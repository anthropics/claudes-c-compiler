//! Copy propagation optimization pass.
//!
//! This pass eliminates redundant Copy instructions by replacing uses of a
//! Copy's destination with the Copy's source operand. This is particularly
//! important because:
//! - Phi elimination generates many Copy instructions
//! - Mem2reg creates Copy instructions when replacing loads
//! - Other optimization passes (simplify, GVN) create Copy instructions
//!
//! Without copy propagation, each Copy becomes a load-to-accumulator then
//! store-to-new-slot in codegen, wasting both instructions and stack space.
//!
//! After this pass runs, the dead Copy instructions are cleaned up by DCE.
//!
//! Performance: Uses a flat Vec<Option<Operand>> indexed by Value ID instead of
//! FxHashMap, since Value IDs are dense sequential u32s. This eliminates hashing
//! overhead and gives O(1) lookups with better cache locality.

use crate::ir::reexports::{
    Instruction,
    IrFunction,
    IrModule,
    Operand,
    Terminator,
    Value,
};

/// Run copy propagation on the entire module.
/// Returns the number of operand replacements made.
pub fn run(module: &mut IrModule) -> usize {
    module.for_each_function(propagate_copies)
}

/// Propagate copies within a single function.
pub(crate) fn propagate_copies(func: &mut IrFunction) -> usize {
    let max_id = func.max_value_id() as usize;

    // Phase 1: Build the copy map as a flat lookup table (dest -> resolved source)
    let (copy_map, has_copies) = build_copy_map(func, max_id);

    // Early exit if no copies found (avoids scanning the entire copy_map Vec)
    if !has_copies {
        return 0;
    }

    // Phase 2: Replace all uses of copied values
    let mut replacements = 0;

    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            replacements += replace_operands_in_instruction(inst, &copy_map);
        }
        replacements += replace_operands_in_terminator(&mut block.terminator, &copy_map);
    }

    replacements
}

/// Build a flat lookup table from Copy destinations to their ultimate sources.
/// Follows chains: if %a = Copy %b and %b = Copy %c, resolves %a -> %c.
/// Returns (copy_map, has_any_entries) to avoid scanning the map for emptiness.
///
/// Uses path compression: when resolving a chain, all intermediate entries are
/// updated to point directly to the final resolved value. This makes
/// resolution amortized O(1) per entry instead of O(chain_length).
/// Also avoids allocating a separate `resolved` vector by resolving in-place.
fn build_copy_map(func: &IrFunction, max_id: usize) -> (Vec<Option<Operand>>, bool) {
    // First pass: collect direct copy relationships into flat table.
    // If a Value has multiple Copy definitions (e.g. from single-phi elimination),
    // we mark it as multi-def using a sentinel (self-referencing copy) and skip it.
    let mut direct: Vec<Option<Operand>> = vec![None; max_id + 1];
    let mut has_copies = false;

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Copy { dest, src } = inst {
                let id = dest.0 as usize;
                if id < direct.len() {
                    if direct[id].is_some() {
                        // Multi-def: mark with self-referencing sentinel
                        direct[id] = Some(Operand::Value(Value(id as u32)));
                    } else {
                        direct[id] = Some(*src);
                        has_copies = true;
                    }
                }
            }
        }
    }

    if !has_copies {
        return (direct, false);
    }

    // Second pass: resolve chains in-place with path compression.
    // We resolve each entry to its ultimate source and memoize the result
    // back into `direct`, so subsequent lookups of intermediate entries
    // are O(1). Uses iterative chain walking to avoid stack overflow.
    let mut any_resolved = false;
    for i in 0..=max_id {
        if direct[i].is_some() {
            resolve_chain_with_compression(&mut direct, i as u32);
            // After compression, check if this is a valid (non-self-ref) entry
            if let Some(Operand::Value(v)) = direct[i] {
                if v.0 == i as u32 {
                    // Self-referencing (multi-def sentinel or cycle) - clear it
                    direct[i] = None;
                    continue;
                }
            }
            if direct[i].is_some() {
                any_resolved = true;
            }
        }
    }

    (direct, any_resolved)
}

/// Resolve a copy chain starting at `start` with path compression.
/// Walks the chain to find the ultimate source, then updates all intermediate
/// entries to point directly to it (path compression / union-find style).
fn resolve_chain_with_compression(copies: &mut [Option<Operand>], start: u32) {
    // First, find the ultimate source by walking the chain.
    let mut current = start;
    let mut depth = 0;
    const MAX_DEPTH: usize = 64;

    let ultimate = loop {
        if depth >= MAX_DEPTH {
            break Operand::Value(Value(current));
        }

        let idx = current as usize;
        match if idx < copies.len() { copies[idx] } else { None } {
            Some(Operand::Value(v)) => {
                if v.0 == current {
                    // Self-reference (multi-def or cycle)
                    break Operand::Value(Value(current));
                }
                current = v.0;
                depth += 1;
            }
            Some(Operand::Const(c)) => {
                break Operand::Const(c);
            }
            None => {
                break Operand::Value(Value(current));
            }
        }
    };

    // Path compression: walk the chain from start and update every
    // intermediate entry to point directly to `ultimate`.
    // For short chains (depth <= 1), we only need to update start itself.
    let start_idx = start as usize;
    if depth <= 1 {
        if start_idx < copies.len() && copies[start_idx].is_some() {
            copies[start_idx] = Some(ultimate);
        }
        return;
    }

    // For longer chains, walk from start and compress each hop.
    // We track the ultimate value ID (if it's a Value) to know when to stop.
    let ultimate_id = match ultimate {
        Operand::Value(v) => Some(v.0),
        Operand::Const(_) => None,
    };
    let mut current = start;
    for _ in 0..depth {
        let idx = current as usize;
        if idx >= copies.len() {
            break;
        }
        match copies[idx] {
            Some(Operand::Value(v)) if v.0 != current => {
                let next = v.0;
                copies[idx] = Some(ultimate);
                // Stop if next is the ultimate target (nothing more to compress)
                if ultimate_id == Some(next) {
                    break;
                }
                current = next;
            }
            _ => break,
        }
    }
}

/// Replace operands in an instruction that reference copied values.
/// Returns the number of replacements made.
fn replace_operands_in_instruction(inst: &mut Instruction, copy_map: &[Option<Operand>]) -> usize {
    let mut count = 0;

    match inst {
        Instruction::Alloca { .. } => {}
        Instruction::DynAlloca { size, .. } => {
            count += replace_operand(size, copy_map);
        }
        Instruction::Store { val, ptr, .. } => {
            count += replace_operand(val, copy_map);
            count += replace_value_in_place(ptr, copy_map);
        }
        Instruction::Load { ptr, .. } => {
            count += replace_value_in_place(ptr, copy_map);
        }
        Instruction::BinOp { lhs, rhs, .. } => {
            count += replace_operand(lhs, copy_map);
            count += replace_operand(rhs, copy_map);
        }
        Instruction::UnaryOp { src, .. } => {
            count += replace_operand(src, copy_map);
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            count += replace_operand(lhs, copy_map);
            count += replace_operand(rhs, copy_map);
        }
        Instruction::Call { info, .. } => {
            for arg in info.args.iter_mut() {
                count += replace_operand(arg, copy_map);
            }
        }
        Instruction::CallIndirect { func_ptr, info } => {
            count += replace_operand(func_ptr, copy_map);
            for arg in info.args.iter_mut() {
                count += replace_operand(arg, copy_map);
            }
        }
        Instruction::GetElementPtr { base, offset, .. } => {
            count += replace_value_in_place(base, copy_map);
            count += replace_operand(offset, copy_map);
        }
        Instruction::Cast { src, .. } => {
            count += replace_operand(src, copy_map);
        }
        Instruction::Copy { src, .. } => {
            count += replace_operand(src, copy_map);
        }
        Instruction::GlobalAddr { .. } => {}
        Instruction::Memcpy { dest, src, .. } => {
            count += replace_value_in_place(dest, copy_map);
            count += replace_value_in_place(src, copy_map);
        }
        Instruction::VaArg { va_list_ptr, .. } => {
            count += replace_value_in_place(va_list_ptr, copy_map);
        }
        Instruction::VaStart { va_list_ptr } => {
            count += replace_value_in_place(va_list_ptr, copy_map);
        }
        Instruction::VaEnd { va_list_ptr } => {
            count += replace_value_in_place(va_list_ptr, copy_map);
        }
        Instruction::VaCopy { dest_ptr, src_ptr } => {
            count += replace_value_in_place(dest_ptr, copy_map);
            count += replace_value_in_place(src_ptr, copy_map);
        }
        Instruction::VaArgStruct { dest_ptr, va_list_ptr, .. } => {
            count += replace_value_in_place(dest_ptr, copy_map);
            count += replace_value_in_place(va_list_ptr, copy_map);
        }
        Instruction::AtomicRmw { ptr, val, .. } => {
            count += replace_operand(ptr, copy_map);
            count += replace_operand(val, copy_map);
        }
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => {
            count += replace_operand(ptr, copy_map);
            count += replace_operand(expected, copy_map);
            count += replace_operand(desired, copy_map);
        }
        Instruction::AtomicLoad { ptr, .. } => {
            count += replace_operand(ptr, copy_map);
        }
        Instruction::AtomicStore { ptr, val, .. } => {
            count += replace_operand(ptr, copy_map);
            count += replace_operand(val, copy_map);
        }
        Instruction::Fence { .. } => {}
        Instruction::Phi { incoming, .. } => {
            for (op, _label) in incoming.iter_mut() {
                count += replace_operand(op, copy_map);
            }
        }
        Instruction::LabelAddr { .. } => {}
        Instruction::GetReturnF64Second { .. } => {}
        Instruction::GetReturnF32Second { .. } => {}
        Instruction::GetReturnF128Second { .. } => {}
        Instruction::SetReturnF64Second { src } => {
            count += replace_operand(src, copy_map);
        }
        Instruction::SetReturnF32Second { src } => {
            count += replace_operand(src, copy_map);
        }
        Instruction::SetReturnF128Second { src } => {
            count += replace_operand(src, copy_map);
        }
        Instruction::InlineAsm { inputs, .. } => {
            for (_constraint, op, _name) in inputs.iter_mut() {
                count += replace_operand(op, copy_map);
            }
        }
        Instruction::Intrinsic { args, .. } => {
            for arg in args.iter_mut() {
                count += replace_operand(arg, copy_map);
            }
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            count += replace_operand(cond, copy_map);
            count += replace_operand(true_val, copy_map);
            count += replace_operand(false_val, copy_map);
        }
        Instruction::StackSave { .. } => {}
        Instruction::StackRestore { ptr } => {
            count += replace_value_in_place(ptr, copy_map);
        }
        Instruction::ParamRef { .. } => {}
    }

    count
}

/// Replace operands in a terminator.
fn replace_operands_in_terminator(term: &mut Terminator, copy_map: &[Option<Operand>]) -> usize {
    let mut count = 0;
    match term {
        Terminator::Return(Some(val)) => {
            count += replace_operand(val, copy_map);
        }
        Terminator::Return(None) => {}
        Terminator::Branch(_) => {}
        Terminator::CondBranch { cond, .. } => {
            count += replace_operand(cond, copy_map);
        }
        Terminator::IndirectBranch { target, .. } => {
            count += replace_operand(target, copy_map);
        }
        Terminator::Switch { val, .. } => {
            count += replace_operand(val, copy_map);
        }
        Terminator::Unreachable => {}
    }
    count
}

/// Replace an Operand if it references a copied value.
/// Returns 1 if a replacement was made, 0 otherwise.
#[inline]
fn replace_operand(op: &mut Operand, copy_map: &[Option<Operand>]) -> usize {
    if let Operand::Value(v) = op {
        let idx = v.0 as usize;
        if let Some(Some(replacement)) = copy_map.get(idx) {
            *op = *replacement;
            return 1;
        }
    }
    0
}

/// Replace a Value in-place if it references a copied value.
/// Only replaces if the resolved source is also a Value (not a Const).
/// Returns 1 if a replacement was made, 0 otherwise.
#[inline]
fn replace_value_in_place(val: &mut Value, copy_map: &[Option<Operand>]) -> usize {
    let idx = val.0 as usize;
    if let Some(Some(Operand::Value(new_val))) = copy_map.get(idx) {
        *val = *new_val;
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IrType;
    use crate::ir::reexports::{BasicBlock, BlockId, IrBinOp, IrConst};

    #[test]
    fn test_simple_copy_propagation() {
        // %1 = Copy %0
        // %2 = Add %1, const(1)
        // Should become:
        // %1 = Copy %0 (dead, will be removed by DCE)
        // %2 = Add %0, const(1)
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Value(Value(0)),
                },
                Instruction::BinOp {
                    dest: Value(2),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        });

        let replacements = propagate_copies(&mut func);
        assert!(replacements > 0);

        // The BinOp should now reference %0 directly
        match &func.blocks[0].instructions[1] {
            Instruction::BinOp { lhs: Operand::Value(v), .. } => {
                assert_eq!(v.0, 0, "Should reference original value %0");
            }
            other => panic!("Expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn test_chain_copy_propagation() {
        // %1 = Copy %0
        // %2 = Copy %1
        // %3 = Add %2, const(1)
        // Should resolve %2 -> %0
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Value(Value(0)),
                },
                Instruction::Copy {
                    dest: Value(2),
                    src: Operand::Value(Value(1)),
                },
                Instruction::BinOp {
                    dest: Value(3),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
            source_spans: Vec::new(),
        });

        let replacements = propagate_copies(&mut func);
        assert!(replacements > 0);

        // The BinOp should now reference %0 directly
        match &func.blocks[0].instructions[2] {
            Instruction::BinOp { lhs: Operand::Value(v), .. } => {
                assert_eq!(v.0, 0, "Should resolve chain to original value %0");
            }
            other => panic!("Expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn test_const_copy_propagation() {
        // %0 = Copy const(42)
        // %1 = Add %0, const(1)
        // Should propagate const(42) into the Add
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I32(42)),
                },
                Instruction::BinOp {
                    dest: Value(1),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(0)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });

        let replacements = propagate_copies(&mut func);
        assert!(replacements > 0);

        // The BinOp should now have const(42) as lhs
        match &func.blocks[0].instructions[1] {
            Instruction::BinOp { lhs: Operand::Const(IrConst::I32(42)), .. } => {}
            other => panic!("Expected BinOp with const 42, got {:?}", other),
        }
    }

    #[test]
    fn test_terminator_propagation() {
        // %1 = Copy %0
        // return %1
        // Should become return %0
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Value(Value(0)),
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });

        let replacements = propagate_copies(&mut func);
        assert!(replacements > 0);

        match &func.blocks[0].terminator {
            Terminator::Return(Some(Operand::Value(v))) => {
                assert_eq!(v.0, 0, "Return should reference %0 directly");
            }
            other => panic!("Expected Return with %0, got {:?}", other),
        }
    }

    #[test]
    fn test_no_propagation_when_no_copies() {
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::BinOp {
                    dest: Value(0),
                    op: IrBinOp::Add,
                    lhs: Operand::Const(IrConst::I32(1)),
                    rhs: Operand::Const(IrConst::I32(2)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(0)))),
            source_spans: Vec::new(),
        });

        let replacements = propagate_copies(&mut func);
        assert_eq!(replacements, 0);
    }
}
