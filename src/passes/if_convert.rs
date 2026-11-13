//! If-conversion pass: converts simple branch+phi diamonds to Select instructions.
//!
//! This pass identifies diamond-shaped CFG patterns:
//!
//! ```text
//!     pred_block:
//!         ...
//!         condbranch %cond, true_block, false_block
//!
//!     true_block:
//!         (0-1 simple instructions)
//!         branch merge_block
//!
//!     false_block:
//!         (0-1 simple instructions)
//!         branch merge_block
//!
//!     merge_block:
//!         %result = phi [true_val, true_block], [false_val, false_block]
//!         ...
//! ```
//!
//! And converts them to:
//!
//! ```text
//!     pred_block:
//!         ...
//!         %result = select %cond, true_val, false_val
//!         branch merge_block
//! ```
//!
//! This eliminates branches in favor of conditional moves (cmov/csel),
//! which is critical for performance in tight loops with simple conditionals
//! (e.g., `x >= wsize ? x - wsize : 0` in zlib's slide_hash).
//!
//! Safety: Only converts when both arms are side-effect-free (no stores, calls,
//! or memory operations). This ensures the Select semantics (evaluate both
//! operands) match the original branch semantics.

use crate::ir::reexports::{
    BasicBlock,
    BlockId,
    Instruction,
    IrFunction,
    Operand,
    Terminator,
    Value,
};
use crate::ir::analysis;
use crate::common::types::IrType;
use crate::common::fx_hash::FxHashMap;

/// Run if-conversion on a single function.
pub(crate) fn if_convert_function(func: &mut IrFunction) -> usize {
    let num_blocks = func.blocks.len();
    if num_blocks < 3 {
        return 0; // Need at least 3 blocks (pred + one arm + merge)
    }

    let mut total = 0;

    // Iterate to a fixpoint since converting one diamond may expose another.
    loop {
        let converted = if_convert_once(func);
        if converted == 0 {
            break;
        }
        total += converted;
    }

    total
}

/// Single pass of if-conversion. Returns number of diamonds converted.
fn if_convert_once(func: &mut IrFunction) -> usize {
    let num_blocks = func.blocks.len();
    if num_blocks < 3 {
        return 0;
    }

    // Build CFG
    let label_to_idx = analysis::build_label_map(func);
    let (preds, _succs) = analysis::build_cfg(func, &label_to_idx);

    // Collect diamond candidates
    let mut diamonds: Vec<DiamondInfo> = Vec::new();

    for pred_idx in 0..num_blocks {
        if let Some(diamond) = detect_diamond(func, pred_idx, &label_to_idx, &preds) {
            diamonds.push(diamond);
        } else if let Some(triangle) = detect_triangle(func, pred_idx, &label_to_idx, &preds) {
            diamonds.push(triangle);
        }
    }


    if diamonds.is_empty() {
        return 0;
    }

    // Apply conversions. Track modified blocks to avoid applying overlapping diamonds
    // (e.g., nested ternaries where converting one invalidates another).
    let mut converted = 0;
    let mut modified_blocks: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for diamond in &diamonds {
        // Skip if any of the diamond's blocks were already modified
        if modified_blocks.contains(&diamond.pred_idx)
            || modified_blocks.contains(&diamond.true_idx)
            || modified_blocks.contains(&diamond.false_idx)
            || modified_blocks.contains(&diamond.merge_idx)
        {
            continue;
        }
        if apply_diamond(func, diamond) {
            modified_blocks.insert(diamond.pred_idx);
            modified_blocks.insert(diamond.true_idx);
            modified_blocks.insert(diamond.false_idx);
            modified_blocks.insert(diamond.merge_idx);
            converted += 1;
        }
    }

    // Clean up: run a quick dead block pass to remove the now-empty arm blocks
    // (they'll have no instructions and just an unconditional branch)
    // This is handled by the CFG simplification pass that runs after us.

    converted
}

/// Information about a detected diamond pattern.
struct DiamondInfo {
    /// The block containing the CondBranch
    pred_idx: usize,
    /// The true-branch block index
    true_idx: usize,
    /// The false-branch block index
    false_idx: usize,
    /// The merge block index (with the Phi)
    merge_idx: usize,
    /// The condition operand from the CondBranch
    cond: Operand,
    /// Instructions to hoist from the true arm (before the Select)
    true_arm_insts: Vec<Instruction>,
    /// Instructions to hoist from the false arm (before the Select)
    false_arm_insts: Vec<Instruction>,
    /// Phi nodes in the merge block that can be converted to Select.
    /// Each entry is (phi_dest, phi_ty, true_val, false_val).
    phi_selects: Vec<(Value, IrType, Operand, Operand)>,
}

/// Check if a block contains only simple, side-effect-free instructions
/// that are safe to speculatively execute.
///
/// IMPORTANT: Load is NOT included here because loads can trap (segfault)
/// on invalid pointers. Hoisting a load past a null-pointer guard would
/// cause a crash. For example, `if (!p || !p[0])` has a diamond where
/// one arm loads `*p` — if-converting this would execute the load
/// unconditionally, crashing when `p` is NULL.
fn is_side_effect_free(block: &BasicBlock) -> bool {
    for inst in &block.instructions {
        match inst {
            // Division and remainder can trap with SIGFPE on divide-by-zero.
            // They must not be speculatively executed past a guard condition.
            Instruction::BinOp { op, .. } if op.can_trap() => return false,
            Instruction::BinOp { .. }
            | Instruction::UnaryOp { .. }
            | Instruction::Cmp { .. }
            | Instruction::Cast { .. }
            | Instruction::Copy { .. }
            | Instruction::GetElementPtr { .. }
            | Instruction::GlobalAddr { .. }
            | Instruction::Select { .. } => {}
            // Load can trap on invalid pointers — not safe to speculate.
            // Store, Call, atomics, etc. have write side effects.
            _ => return false,
        }
    }
    true
}

/// Check if a condition operand is a known constant or can be trivially resolved
/// to a constant within the block. When the condition is constant, the branch
/// should be folded by cfg_simplify rather than converted to a Select by
/// if_convert. Converting a constant-condition branch to Select delays dead
/// code elimination: the Select needs additional optimization iterations
/// (simplify + constfold + cfg_simplify) to fold away, and if the diminishing-
/// returns heuristic terminates the optimization loop early, dead code paths
/// (e.g., kernel's conditional calls to restore_tpidr2_context guarded by
/// system_supports_sme() which returns false) survive to the final output.
fn is_constant_condition(block: &BasicBlock, cond: &Operand) -> bool {
    match cond {
        Operand::Const(_) => true,
        Operand::Value(v) => {
            // Check if the value is defined as a constant Copy, or as a Cmp/Select
            // where all operands are constants, within the same block.
            for inst in &block.instructions {
                match inst {
                    Instruction::Copy { dest, src: Operand::Const(_) } if *dest == *v => {
                        return true;
                    }
                    Instruction::Cmp { dest, lhs, rhs, .. } if *dest == *v => {
                        let lhs_const = matches!(lhs, Operand::Const(_)) || is_value_const_in_block(block, lhs);
                        let rhs_const = matches!(rhs, Operand::Const(_)) || is_value_const_in_block(block, rhs);
                        if lhs_const && rhs_const {
                            return true;
                        }
                    }
                    Instruction::Select { dest, true_val, false_val, .. } if *dest == *v => {
                        // Select(cond, x, x) where both arms are the same constant
                        if same_value_or_both_zero(true_val, false_val) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }
    }
}

/// Check if an operand is a constant within the block (either directly or via Copy).
fn is_value_const_in_block(block: &BasicBlock, op: &Operand) -> bool {
    match op {
        Operand::Const(_) => true,
        Operand::Value(v) => {
            for inst in &block.instructions {
                if let Instruction::Copy { dest, src: Operand::Const(_) } = inst {
                    if *dest == *v {
                        return true;
                    }
                }
            }
            false
        }
    }
}

/// Check if two operands are the same value or both integer-zero constants.
/// Used to detect Select(cond, x, x) patterns where the result is the same
/// regardless of the condition, making the condition effectively constant.
fn same_value_or_both_zero(a: &Operand, b: &Operand) -> bool {
    match (a, b) {
        (Operand::Const(ca), Operand::Const(cb)) => {
            ca.to_i64() == Some(0) && cb.to_i64() == Some(0)
        }
        (Operand::Value(va), Operand::Value(vb)) => va.0 == vb.0,
        _ => false,
    }
}

/// Detect a diamond pattern starting from a block with a CondBranch terminator.
fn detect_diamond(
    func: &IrFunction,
    pred_idx: usize,
    label_to_idx: &FxHashMap<BlockId, usize>,
    preds: &analysis::FlatAdj,
) -> Option<DiamondInfo> {
    let pred_block = &func.blocks[pred_idx];

    // Must end with a CondBranch
    let (cond, true_label, false_label) = match &pred_block.terminator {
        Terminator::CondBranch { cond, true_label, false_label } => {
            (cond, true_label, false_label)
        }
        _ => return None,
    };

    // Don't convert branches with constant conditions to Select.
    // cfg_simplify will fold these more efficiently (single pass vs multi-iteration).
    if is_constant_condition(pred_block, cond) {
        return None;
    }

    let true_idx = *label_to_idx.get(true_label)?;
    let false_idx = *label_to_idx.get(false_label)?;

    // The two arms must be different blocks
    if true_idx == false_idx {
        return None;
    }

    let true_block = &func.blocks[true_idx];
    let false_block = &func.blocks[false_idx];

    // Both arms must end with unconditional branches to the same merge block
    let true_target = match &true_block.terminator {
        Terminator::Branch(label) => *label_to_idx.get(label)?,
        _ => return None,
    };
    let false_target = match &false_block.terminator {
        Terminator::Branch(label) => *label_to_idx.get(label)?,
        _ => return None,
    };

    if true_target != false_target {
        return None; // Different merge blocks
    }
    let merge_idx = true_target;

    // The merge block must not be one of the arms
    if merge_idx == true_idx || merge_idx == false_idx || merge_idx == pred_idx {
        return None;
    }

    // The arm blocks should have exactly one predecessor each (the pred block).
    // If they have other predecessors, other code flows into them and we can't
    // eliminate the blocks.
    if preds.len(true_idx) != 1 || preds.len(false_idx) != 1 {
        return None;
    }
    if preds.row(true_idx)[0] as usize != pred_idx || preds.row(false_idx)[0] as usize != pred_idx {
        return None;
    }

    // Both arms must be side-effect-free
    if !is_side_effect_free(true_block) || !is_side_effect_free(false_block) {
        return None;
    }

    // Limit the number of instructions in each arm to prevent over-speculation.
    // cmov is only profitable when the arms are cheap (a few instructions).
    // Allow up to 8 instructions per arm. C's type system generates
    // Load + Cast chains (parameter loads + sign extensions) that inflate
    // the count. A typical arm: Load, Cast, Load, Cast, BinOp, Cast = 6 insts.
    const MAX_ARM_INSTS: usize = 8;
    if true_block.instructions.len() > MAX_ARM_INSTS || false_block.instructions.len() > MAX_ARM_INSTS {
        return None;
    }

    // The merge block must have Phi nodes that reference both arms.
    // Collect phi nodes we can convert.
    let merge_block = &func.blocks[merge_idx];
    let mut phi_selects = Vec::new();

    for inst in &merge_block.instructions {
        if let Instruction::Phi { dest, ty, incoming } = inst {
            // Find the values from each arm
            let mut true_val = None;
            let mut false_val = None;

            for (op, label) in incoming {
                let src_idx = label_to_idx.get(label).copied();
                if src_idx == Some(true_idx) {
                    true_val = Some(*op);
                } else if src_idx == Some(false_idx) {
                    false_val = Some(*op);
                }
                // Other predecessors are fine - they just won't be converted
            }

            match (true_val, false_val) {
                (Some(tv), Some(fv)) => {
                    // Support integer, pointer, F32, and F64 types.
                    // F32/F64 work because all backends implement Select by
                    // moving bit patterns through integer registers (cmov/csel/branch).
                    // Skip long double (F128) and 128-bit integers — they need
                    // multi-register handling that Select doesn't support.
                    if !ty.is_long_double() && !ty.is_128bit() {
                        phi_selects.push((*dest, *ty, tv, fv));
                    } else {
                        // There's an unconvertible phi (F128/I128) referencing
                        // both arms. We must NOT partially convert this diamond
                        // or the remaining phi nodes will reference removed blocks.
                        return None;
                    }
                }
                _ => {
                    // Phi doesn't have entries from both arms - can't convert
                    // This could happen if the phi also has entries from other preds
                }
            }
        }
    }

    if phi_selects.is_empty() {
        return None; // No convertible phis
    }

    // The merge block should only be reached from the two arms (and not from pred directly).
    // If the merge block has other predecessors, we need to preserve the Phi nodes for those.
    let merge_preds_from_diamond = preds.row(merge_idx).iter()
        .filter(|&&p| p as usize == true_idx || p as usize == false_idx)
        .count();

    // If the merge block has predecessors other than the two arms, we can still convert
    // but we need to keep the Phi nodes with those other incoming edges.
    // For simplicity, only convert when the merge block has exactly 2 predecessors
    // (the two arms), so we can fully replace the Phis.
    if preds.len(merge_idx) != 2 || merge_preds_from_diamond != 2 {
        return None;
    }

    Some(DiamondInfo {
        pred_idx,
        true_idx,
        false_idx,
        merge_idx,
        cond: *cond,
        true_arm_insts: true_block.instructions.clone(),
        false_arm_insts: false_block.instructions.clone(),
        phi_selects,
    })
}

/// Detect a triangle pattern: pred branches to arm and merge directly.
///
/// ```text
///     pred: CondBranch(cond, arm, merge)   -- or (cond, merge, arm)
///     arm:  side-effect-free instructions + Branch(merge)
///     merge: phi [arm_val, arm], [pred_val, pred]
/// ```
///
/// This handles ternaries like `a >= t ? a - t : 0` where the false arm
/// is a constant and doesn't need its own block.
fn detect_triangle(
    func: &IrFunction,
    pred_idx: usize,
    label_to_idx: &FxHashMap<BlockId, usize>,
    preds: &analysis::FlatAdj,
) -> Option<DiamondInfo> {
    let pred_block = &func.blocks[pred_idx];

    let (cond, true_label, false_label) = match &pred_block.terminator {
        Terminator::CondBranch { cond, true_label, false_label } => {
            (cond, true_label, false_label)
        }
        _ => return None,
    };

    // Don't convert branches with constant conditions to Select.
    // cfg_simplify will fold these more efficiently (single pass vs multi-iteration).
    if is_constant_condition(pred_block, cond) {
        return None;
    }

    let true_idx = *label_to_idx.get(true_label)?;
    let false_idx = *label_to_idx.get(false_label)?;

    if true_idx == false_idx {
        return None;
    }

    // Determine which arm goes to a separate block and which goes directly to merge.
    // Case 1: true arm is a block, false arm goes directly to merge
    // Case 2: false arm is a block, true arm goes directly to merge
    let (arm_idx, merge_idx, arm_is_true) = {
        let true_block = &func.blocks[true_idx];
        let false_block = &func.blocks[false_idx];

        let true_target = match &true_block.terminator {
            Terminator::Branch(label) => label_to_idx.get(label).copied(),
            _ => None,
        };
        let false_target = match &false_block.terminator {
            Terminator::Branch(label) => label_to_idx.get(label).copied(),
            _ => None,
        };

        if let Some(tt) = true_target {
            if tt == false_idx {
                // true arm branches to false_idx which is the merge block
                (true_idx, false_idx, true)
            } else if let Some(ft) = false_target {
                if ft == true_idx {
                    // false arm branches to true_idx which is the merge block
                    (false_idx, true_idx, false)
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else if let Some(ft) = false_target {
            if ft == true_idx {
                (false_idx, true_idx, false)
            } else {
                return None;
            }
        } else {
            return None;
        }
    };

    // arm block must have exactly one predecessor (pred)
    if preds.len(arm_idx) != 1 || preds.row(arm_idx)[0] as usize != pred_idx {
        return None;
    }

    // merge must have exactly 2 predecessors: pred and arm
    if preds.len(merge_idx) != 2 {
        return None;
    }
    let has_pred = preds.row(merge_idx).iter().any(|&p| p as usize == pred_idx);
    let has_arm = preds.row(merge_idx).iter().any(|&p| p as usize == arm_idx);
    if !has_pred || !has_arm {
        return None;
    }

    let arm_block = &func.blocks[arm_idx];

    // arm must be side-effect-free
    if !is_side_effect_free(arm_block) {
        return None;
    }

    const MAX_ARM_INSTS: usize = 8;
    if arm_block.instructions.len() > MAX_ARM_INSTS {
        return None;
    }

    // Collect phi nodes from merge block
    let merge_block = &func.blocks[merge_idx];
    let mut phi_selects = Vec::new();

    for inst in &merge_block.instructions {
        if let Instruction::Phi { dest, ty, incoming } = inst {
            let mut arm_val = None;
            let mut pred_val = None;

            for (op, label) in incoming {
                let src_idx = label_to_idx.get(label).copied();
                if src_idx == Some(arm_idx) {
                    arm_val = Some(*op);
                } else if src_idx == Some(pred_idx) {
                    pred_val = Some(*op);
                }
            }

            if let (Some(av), Some(pv)) = (arm_val, pred_val) {
                // Support integer, pointer, F32, and F64 types.
                // Skip long double (F128) and 128-bit integers.
                if !ty.is_long_double() && !ty.is_128bit() {
                    // Map to true/false values based on which arm the block is
                    if arm_is_true {
                        phi_selects.push((*dest, *ty, av, pv));
                    } else {
                        phi_selects.push((*dest, *ty, pv, av));
                    }
                } else {
                    // Unconvertible phi — bail out to avoid partial conversion.
                    return None;
                }
            }
        }
    }

    if phi_selects.is_empty() {
        return None;
    }

    // For a triangle, we set the missing arm to merge_idx with empty instructions.
    // apply_diamond will hoist the arm instructions and the empty side is a no-op.
    let (true_idx_out, false_idx_out, true_insts, false_insts) = if arm_is_true {
        (arm_idx, merge_idx, arm_block.instructions.clone(), Vec::new())
    } else {
        (merge_idx, arm_idx, Vec::new(), arm_block.instructions.clone())
    };

    Some(DiamondInfo {
        pred_idx,
        true_idx: true_idx_out,
        false_idx: false_idx_out,
        merge_idx,
        cond: *cond,
        true_arm_insts: true_insts,
        false_arm_insts: false_insts,
        phi_selects,
    })
}

/// Apply a diamond conversion: rewrite the CFG to use Select instructions.
fn apply_diamond(func: &mut IrFunction, diamond: &DiamondInfo) -> bool {
    // Safety check: make sure the blocks haven't been modified by a previous conversion
    // in this same pass iteration
    if diamond.pred_idx >= func.blocks.len()
        || diamond.true_idx >= func.blocks.len()
        || diamond.false_idx >= func.blocks.len()
        || diamond.merge_idx >= func.blocks.len()
    {
        return false;
    }

    // Read the merge label before mutating
    let merge_label = func.blocks[diamond.merge_idx].label;

    // 1. Move instructions from both arms into the pred block (before the branch).
    //    Since both arms are side-effect-free, we can execute all their instructions
    //    unconditionally (both paths are computed).
    let pred_block = &mut func.blocks[diamond.pred_idx];

    // Add true arm instructions (with dummy spans)
    let has_spans = !pred_block.source_spans.is_empty();
    for inst in &diamond.true_arm_insts {
        pred_block.instructions.push(inst.clone());
        if has_spans { pred_block.source_spans.push(crate::common::source::Span::dummy()); }
    }

    // Add false arm instructions
    for inst in &diamond.false_arm_insts {
        pred_block.instructions.push(inst.clone());
        if has_spans { pred_block.source_spans.push(crate::common::source::Span::dummy()); }
    }

    // 2. Add Select instructions for each Phi
    for (dest, ty, true_val, false_val) in &diamond.phi_selects {
        pred_block.instructions.push(Instruction::Select {
            dest: *dest,
            cond: diamond.cond,
            true_val: *true_val,
            false_val: *false_val,
            ty: *ty,
        });
        if has_spans { pred_block.source_spans.push(crate::common::source::Span::dummy()); }
    }

    // 3. Change pred block's terminator to unconditional branch to merge
    pred_block.terminator = Terminator::Branch(merge_label);

    // 4. Remove the converted Phi nodes from the merge block
    let converted_dests: std::collections::HashSet<u32> = diamond.phi_selects.iter()
        .map(|(dest, _, _, _)| dest.0)
        .collect();
    {
        let merge_block = &mut func.blocks[diamond.merge_idx];
        if !merge_block.source_spans.is_empty() {
            let mut idx = 0;
            let insts = &merge_block.instructions;
            merge_block.source_spans.retain(|_| {
                let keep = if let Instruction::Phi { dest, .. } = &insts[idx] {
                    !converted_dests.contains(&dest.0)
                } else {
                    true
                };
                idx += 1;
                keep
            });
        }
    }
    func.blocks[diamond.merge_idx].instructions.retain(|inst| {
        if let Instruction::Phi { dest, .. } = inst {
            !converted_dests.contains(&dest.0)
        } else {
            true
        }
    });

    // 5. Empty the arm blocks (they'll be cleaned up by CFG simplification).
    // Keep them as empty blocks with unconditional branches - CFG simplify
    // will remove them as dead blocks since they'll have no predecessors.
    // For triangle patterns, one arm IS the merge block - don't clear it.
    if diamond.true_idx != diamond.merge_idx {
        func.blocks[diamond.true_idx].instructions.clear();
        func.blocks[diamond.true_idx].source_spans.clear();
    }
    if diamond.false_idx != diamond.merge_idx {
        func.blocks[diamond.false_idx].instructions.clear();
        func.blocks[diamond.false_idx].source_spans.clear();
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::AddressSpace;
    use crate::ir::reexports::{IrBinOp, IrConst};

    #[test]
    fn test_simple_diamond_conversion() {
        // Build a simple diamond:
        //   block0: condbranch %0, block1, block2
        //   block1: branch block3
        //   block2: branch block3
        //   block3: %3 = phi [const(1), block1], [const(0), block2]; return %3
        let mut func = IrFunction::new("test".to_string(), IrType::I32, vec![], false);

        // Block 0: condbranch
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
            source_spans: Vec::new(),
        });

        // Block 1: true arm
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        // Block 2: false arm
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        // Block 3: merge with phi
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(3),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Const(IrConst::I32(1)), BlockId(1)),
                        (Operand::Const(IrConst::I32(0)), BlockId(2)),
                    ],
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 4;

        let converted = if_convert_function(&mut func);
        assert_eq!(converted, 1);

        // Block 0 should now have a Select instruction and branch to block3
        assert_eq!(func.blocks[0].instructions.len(), 1);
        match &func.blocks[0].instructions[0] {
            Instruction::Select { dest, cond, true_val, false_val, ty } => {
                assert_eq!(dest.0, 3);
                assert!(matches!(cond, Operand::Value(Value(0))));
                assert!(matches!(true_val, Operand::Const(IrConst::I32(1))));
                assert!(matches!(false_val, Operand::Const(IrConst::I32(0))));
                assert_eq!(*ty, IrType::I32);
            }
            other => panic!("Expected Select, got {:?}", other),
        }

        // Block 0 should now branch unconditionally to block3
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(3))));

        // Merge block should have no phi
        assert!(!func.blocks[3].instructions.iter().any(|i| matches!(i, Instruction::Phi { .. })));
    }

    #[test]
    fn test_diamond_with_arm_instructions() {
        // Diamond where the true arm computes a value:
        //   block0: condbranch %0, block1, block2
        //   block1: %1 = sub %0, const(5); branch block3
        //   block2: branch block3
        //   block3: %2 = phi [%1, block1], [const(0), block2]; return %2
        let mut func = IrFunction::new("test".to_string(), IrType::I32, vec![], false);

        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
            source_spans: Vec::new(),
        });

        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::BinOp {
                    dest: Value(1),
                    op: IrBinOp::Sub,
                    lhs: Operand::Value(Value(0)),
                    rhs: Operand::Const(IrConst::I32(5)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(2),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(1)), BlockId(1)),
                        (Operand::Const(IrConst::I32(0)), BlockId(2)),
                    ],
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 3;

        let converted = if_convert_function(&mut func);
        assert_eq!(converted, 1);

        // Block 0 should have the hoisted BinOp and the Select
        assert_eq!(func.blocks[0].instructions.len(), 2);
        assert!(matches!(func.blocks[0].instructions[0], Instruction::BinOp { .. }));
        assert!(matches!(func.blocks[0].instructions[1], Instruction::Select { .. }));
    }

    #[test]
    fn test_no_conversion_with_side_effects() {
        // Diamond where the true arm has a store (side effect) - should NOT convert
        let mut func = IrFunction::new("test".to_string(), IrType::I32, vec![], false);

        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Alloca {
                    dest: Value(10),
                    ty: IrType::I32,
                    size: 4,
                    align: 4,
                    volatile: false,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
            source_spans: Vec::new(),
        });

        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                // Side-effecting store!
                Instruction::Store { val: Operand::Const(IrConst::I32(42)), ptr: Value(10), ty: IrType::I32, seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(2),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Const(IrConst::I32(1)), BlockId(1)),
                        (Operand::Const(IrConst::I32(0)), BlockId(2)),
                    ],
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 11;

        let converted = if_convert_function(&mut func);
        assert_eq!(converted, 0); // Should NOT convert due to side effects
    }
}
