//! Loop Induction Variable Strength Reduction (IVSR) pass.
//!
//! Transforms expensive per-iteration index computations in loops into
//! cheaper pointer increment operations. For array access patterns like:
//!
//!   for (int i = 0; i < n; i++) sum += arr[i];
//!
//! The IR typically generates:
//!   %cast = Cast(%i, I32 -> I64)
//!   %offset = Shl(%cast, 2)          // i * sizeof(int) via shift
//!   %addr = GEP(%base, %offset)      // base + offset
//!   %val = Load(%addr)
//!
//! After IVSR, this becomes:
//!   %ptr = Phi(%initial_ptr, %ptr_next)
//!   %val = Load(%ptr)
//!   %ptr_next = GEP(%ptr, stride)    // ptr += sizeof(int)
//!
//! This eliminates the multiply/shift and cast per iteration, replacing them
//! with a single pointer addition. The dead multiply and cast are then removed
//! by subsequent DCE.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::analysis;
use crate::ir::reexports::{
    Instruction,
    IrBinOp,
    IrConst,
    IrFunction,
    Operand,
    Value,
};
use super::loop_analysis::{self, NaturalLoop};

/// Maximum stride (in bytes) for an induction variable to be eligible for
/// strength reduction. Covers common element sizes up to 1 KB; larger strides
/// are unlikely to benefit and may indicate non-array access patterns.
const MAX_IV_STRIDE: i64 = 1024;

/// Maximum number of Cast/Copy instructions to follow when looking through
/// cast chains to find the root value. Guards against infinite loops on
/// malformed IR with cycles.
const MAX_CAST_CHAIN_LENGTH: usize = 10;

/// A basic induction variable: %iv = phi(init, %iv_next) where %iv_next = %iv + step.
struct BasicIV {
    /// The Phi destination value
    phi_dest: Value,
    /// The type of the IV (typically I32 or I64)
    ty: IrType,
    /// The initial value operand (from outside the loop)
    init: Operand,
    /// The step constant (the additive increment per iteration)
    step: i64,
}

/// A derived expression from a basic IV that can be strength-reduced.
/// Pattern: %derived = %iv_or_cast * const_stride (or Shl), used as offset in GEP
struct DerivedExpr {
    /// The constant stride (element size in bytes)
    stride: i64,
    /// Which basic IV this derives from (index into basic_ivs)
    iv_index: usize,
    /// GEPs that use this multiply/shift result as their offset.
    /// (block_idx, inst_idx, GEP dest, GEP base)
    gep_uses: Vec<(usize, usize, Value, Value)>,
}

/// Run IVSR on a single function.
#[cfg(test)]
pub(crate) fn ivsr_function(func: &mut IrFunction) -> usize {
    let num_blocks = func.blocks.len();
    if num_blocks < 2 {
        return 0;
    }

    // Build CFG and dominator tree
    let cfg = analysis::CfgAnalysis::build(func);
    ivsr_with_analysis(func, &cfg)
}

/// Run IVSR using pre-computed CFG analysis (avoids redundant analysis when
/// called from a pipeline that shares analysis across GVN, LICM, IVSR).
pub(crate) fn ivsr_with_analysis(func: &mut IrFunction, cfg: &analysis::CfgAnalysis) -> usize {
    if cfg.num_blocks < 2 {
        return 0;
    }

    // Find natural loops
    let loops = loop_analysis::find_natural_loops(cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom);
    if loops.is_empty() {
        return 0;
    }

    // Merge loops with same header
    let loops = loop_analysis::merge_loops_by_header(loops);

    let mut total_reductions = 0;

    // Process each loop (innermost first)
    let mut sorted_loops = loops;
    sorted_loops.sort_by_key(|l| l.body.len());

    for natural_loop in &sorted_loops {
        total_reductions += reduce_loop(func, natural_loop, &cfg.preds);
    }

    total_reductions
}

/// Try to strength-reduce induction variables in a single loop.
fn reduce_loop(
    func: &mut IrFunction,
    natural_loop: &NaturalLoop,
    preds: &analysis::FlatAdj,
) -> usize {
    let header = natural_loop.header;

    // Find the preheader (single predecessor outside the loop)
    let preheader = match loop_analysis::find_preheader(header, &natural_loop.body, preds) {
        Some(ph) => ph,
        None => return 0,
    };

    // Find back-edge blocks (predecessors of header that are inside the loop)
    let back_blocks: Vec<usize> = preds.row(header)
        .iter()
        .map(|&p| p as usize)
        .filter(|p| natural_loop.body.contains(p))
        .collect();

    // Only handle simple single-latch loops
    if back_blocks.len() != 1 {
        return 0;
    }

    // Step 1: Identify basic induction variables from phi nodes in the header.
    let basic_ivs = find_basic_ivs(func, header, &natural_loop.body, preheader, &back_blocks);
    if basic_ivs.is_empty() {
        return 0;
    }

    // Step 2: Find derived expressions (iv * const) used in GEPs.
    let derived = find_derived_exprs(func, &basic_ivs, &natural_loop.body);
    if derived.is_empty() {
        return 0;
    }

    // Step 3: Apply strength reduction transformations.
    let mut reductions = 0;
    let mut next_id = func.next_value_id;
    // Track how many phi nodes we've inserted at the header, so subsequent
    // GEP replacements in the header use the correct adjusted index.
    let mut header_phi_insertions = 0usize;

    let preheader_label = func.blocks[preheader].label;
    let back_block_label = func.blocks[back_blocks[0]].label;

    for d in &derived {
        let iv = &basic_ivs[d.iv_index];
        let inc_bytes = iv.step * d.stride;

        for &(gep_block_idx, gep_inst_idx, gep_dest, gep_base) in &d.gep_uses {
            // Only reduce GEPs where the base is loop-invariant (skip loop-variant bases).
            if !is_loop_invariant(gep_base.0, &natural_loop.body, func) {
                continue;
            }

            let ptr_iv_val = Value(next_id);
            next_id += 1;
            let ptr_next_val = Value(next_id);
            next_id += 1;
            let init_ptr_val = Value(next_id);
            next_id += 1;

            // Try to resolve init to a constant (looking through copies)
            let init_const = try_resolve_const(&iv.init, func);
            let init_offset = init_const.map(|v| v * d.stride);

            // Build preheader instructions for computing the initial pointer
            let mut preheader_insts: Vec<Instruction> = Vec::new();

            if let Some(init_off) = init_offset {
                // Constant initial offset case (most common: i = 0)
                if init_off == 0 {
                    preheader_insts.push(Instruction::Copy {
                        dest: init_ptr_val,
                        src: Operand::Value(gep_base),
                    });
                } else {
                    preheader_insts.push(Instruction::GetElementPtr {
                        dest: init_ptr_val,
                        base: gep_base,
                        offset: Operand::Const(IrConst::I64(init_off)),
                        ty: IrType::I8,
                    });
                }
            } else {
                // Non-constant init: compute init * stride at runtime in preheader
                let init_val = match &iv.init {
                    Operand::Value(v) => *v,
                    _ => continue,
                };

                if d.stride == 1 {
                    let init_cast_val = Value(next_id);
                    next_id += 1;
                    if iv.ty != IrType::I64 && iv.ty != IrType::Ptr {
                        preheader_insts.push(Instruction::Cast {
                            dest: init_cast_val,
                            src: Operand::Value(init_val),
                            from_ty: iv.ty,
                            to_ty: IrType::I64,
                        });
                        preheader_insts.push(Instruction::GetElementPtr {
                            dest: init_ptr_val,
                            base: gep_base,
                            offset: Operand::Value(init_cast_val),
                            ty: IrType::I8,
                        });
                    } else {
                        next_id -= 1; // Undo unused init_cast_val allocation
                        preheader_insts.push(Instruction::GetElementPtr {
                            dest: init_ptr_val,
                            base: gep_base,
                            offset: Operand::Value(init_val),
                            ty: IrType::I8,
                        });
                    }
                } else {
                    // Compute init * stride at runtime.
                    // Only allocate a cast value if the IV type needs widening to I64.
                    let needs_cast = iv.ty != IrType::I64 && iv.ty != IrType::Ptr;
                    let mul_operand = if needs_cast {
                        let init_cast_val = Value(next_id);
                        next_id += 1;
                        preheader_insts.push(Instruction::Cast {
                            dest: init_cast_val,
                            src: Operand::Value(init_val),
                            from_ty: iv.ty,
                            to_ty: IrType::I64,
                        });
                        Operand::Value(init_cast_val)
                    } else {
                        Operand::Value(init_val)
                    };

                    let init_mul_val = Value(next_id);
                    next_id += 1;
                    preheader_insts.push(Instruction::BinOp {
                        dest: init_mul_val,
                        op: IrBinOp::Mul,
                        lhs: mul_operand,
                        rhs: Operand::Const(IrConst::I64(d.stride)),
                        ty: IrType::I64,
                    });
                    preheader_insts.push(Instruction::GetElementPtr {
                        dest: init_ptr_val,
                        base: gep_base,
                        offset: Operand::Value(init_mul_val),
                        ty: IrType::I8,
                    });
                }
            }

            // Header: ptr_iv = phi(init_ptr from preheader, ptr_next from back_block)
            let phi_inst = Instruction::Phi {
                dest: ptr_iv_val,
                ty: IrType::Ptr,
                incoming: vec![
                    (Operand::Value(init_ptr_val), preheader_label),
                    (Operand::Value(ptr_next_val), back_block_label),
                ],
            };

            // Back-edge block: ptr_next = GEP(ptr_iv, inc_bytes)
            let inc_inst = Instruction::GetElementPtr {
                dest: ptr_next_val,
                base: ptr_iv_val,
                offset: Operand::Const(IrConst::I64(inc_bytes)),
                ty: IrType::I8,
            };

            // Apply the transformation:
            let ph_has_spans = !func.blocks[preheader].source_spans.is_empty();
            let hdr_has_spans = !func.blocks[header].source_spans.is_empty();
            let bb_has_spans = !func.blocks[back_blocks[0]].source_spans.is_empty();

            // 1. Add init_ptr computation to end of preheader
            for inst in preheader_insts {
                func.blocks[preheader].instructions.push(inst);
                if ph_has_spans {
                    func.blocks[preheader].source_spans.push(
                        crate::common::source::Span::dummy(),
                    );
                }
            }

            // 2. Add phi at beginning of header (after existing phis)
            let insert_pos = func.blocks[header]
                .instructions
                .iter()
                .position(|inst| !matches!(inst, Instruction::Phi { .. }))
                .unwrap_or(func.blocks[header].instructions.len());
            func.blocks[header].instructions.insert(insert_pos, phi_inst);
            if hdr_has_spans {
                func.blocks[header]
                    .source_spans
                    .insert(insert_pos, crate::common::source::Span::dummy());
            }
            header_phi_insertions += 1;

            // 3. Add increment to end of back-edge block
            func.blocks[back_blocks[0]].instructions.push(inc_inst);
            if bb_has_spans {
                func.blocks[back_blocks[0]]
                    .source_spans
                    .push(crate::common::source::Span::dummy());
            }

            // 4. Replace the original GEP instruction with a Copy from ptr_iv.
            // Adjust index for all phis we've inserted at the header so far.
            let adjusted_idx = if gep_block_idx == header {
                gep_inst_idx + header_phi_insertions
            } else {
                gep_inst_idx
            };

            if adjusted_idx < func.blocks[gep_block_idx].instructions.len() {
                let inst = &func.blocks[gep_block_idx].instructions[adjusted_idx];
                if let Some(dest) = inst.dest() {
                    if dest == gep_dest {
                        func.blocks[gep_block_idx].instructions[adjusted_idx] =
                            Instruction::Copy {
                                dest: gep_dest,
                                src: Operand::Value(ptr_iv_val),
                            };
                        reductions += 1;
                    }
                }
            }
        }
    }

    if reductions > 0 {
        func.next_value_id = next_id;
    }

    reductions
}

/// Find basic induction variables: phis in the header of the form
/// %iv = phi(init from preheader, %iv_next from back_block)
/// where %iv_next = %iv + const_step (possibly through casts)
fn find_basic_ivs(
    func: &IrFunction,
    header: usize,
    loop_body: &FxHashSet<usize>,
    preheader: usize,
    back_blocks: &[usize],
) -> Vec<BasicIV> {
    let mut ivs = Vec::new();

    // Build a map from value to its defining instruction within the loop
    let mut loop_defs: FxHashMap<u32, &Instruction> = FxHashMap::default();
    for &bi in loop_body {
        if bi < func.blocks.len() {
            for inst in &func.blocks[bi].instructions {
                if let Some(dest) = inst.dest() {
                    loop_defs.insert(dest.0, inst);
                }
            }
        }
    }

    // Scan phi nodes in the header
    for inst in &func.blocks[header].instructions {
        if let Instruction::Phi { dest, ty, incoming } = inst {
            // Must be integer type (induction variables are integers)
            if !ty.is_integer() && *ty != IrType::Ptr {
                continue;
            }

            // Find the init value (from preheader) and the back-edge value
            let mut init_op = None;
            let mut back_val = None;

            for (op, block_id) in incoming {
                let bi_opt = func.blocks
                    .iter()
                    .enumerate()
                    .find(|(_, b)| b.label == *block_id)
                    .map(|(i, _)| i);
                if let Some(bi) = bi_opt {
                    if bi == preheader {
                        init_op = Some(*op);
                    } else if back_blocks.contains(&bi) {
                        if let Operand::Value(v) = op {
                            back_val = Some(*v);
                        }
                    }
                }
            }

            let init_op = match init_op {
                Some(op) => op,
                None => continue,
            };
            let back_val = match back_val {
                Some(v) => v,
                None => continue,
            };

            // Check if back_val = dest + const_step
            // Also handle: back_val = Cast(Add(dest, step)) or
            //              back_val = Cast(Add(Cast(dest), step))
            // These patterns arise from C integer promotion rules.
            let add_val = look_through_casts(back_val.0, &loop_defs);
            if let Some(Instruction::BinOp { op, lhs, rhs, .. }) = loop_defs.get(&add_val) {
                if *op == IrBinOp::Add {
                    let phi_id = dest.0;
                    let lhs_root = match lhs {
                        Operand::Value(v) => look_through_casts(v.0, &loop_defs),
                        _ => u32::MAX,
                    };
                    let rhs_root = match rhs {
                        Operand::Value(v) => look_through_casts(v.0, &loop_defs),
                        _ => u32::MAX,
                    };

                    let step_operand = if lhs_root == phi_id {
                        Some(rhs)
                    } else if rhs_root == phi_id {
                        Some(lhs)
                    } else {
                        None
                    };

                    if let Some(Operand::Const(c)) = step_operand {
                        if let Some(step) = c.to_i64() {
                            ivs.push(BasicIV {
                                phi_dest: *dest,
                                ty: *ty,
                                init: init_op,
                                step,
                            });
                        }
                    }
                }
            } else if let Some(Instruction::GetElementPtr { base, offset, .. }) = loop_defs.get(&add_val) {
                // GEP-based IV increment: phi(init, GEP(phi, const_stride))
                let base_root = look_through_casts(base.0, &loop_defs);
                if base_root == dest.0 {
                    if let Operand::Const(c) = offset {
                        if let Some(step) = c.to_i64() {
                            ivs.push(BasicIV {
                                phi_dest: *dest,
                                ty: *ty,
                                init: init_op,
                                step,
                            });
                        }
                    }
                }
            }
        }
    }

    ivs
}

/// Find derived expressions: %mul = %iv * const_stride (or Shl by const)
/// that are used as offsets in GEPs within the loop.
fn find_derived_exprs(
    func: &IrFunction,
    basic_ivs: &[BasicIV],
    loop_body: &FxHashSet<usize>,
) -> Vec<DerivedExpr> {
    let mut derived = Vec::new();

    // Build a set of IV phi values for quick lookup
    let mut iv_values: FxHashMap<u32, usize> = FxHashMap::default();
    for (i, iv) in basic_ivs.iter().enumerate() {
        iv_values.insert(iv.phi_dest.0, i);
    }

    // Find Casts and Copies of IV values (common pattern: Cast(i32_iv -> i64),
    // Copy(Cast(iv))). Multiple passes to handle chains.
    let mut iv_derived: FxHashMap<u32, usize> = FxHashMap::default();
    for _ in 0..3 {
        let mut new_entries = Vec::new();
        for &bi in loop_body {
            if bi >= func.blocks.len() {
                continue;
            }
            for inst in &func.blocks[bi].instructions {
                match inst {
                    Instruction::Cast { dest, src: Operand::Value(v), from_ty, to_ty } => {
                        // Only treat widening casts as IV-derived.
                        // Truncating casts (e.g. I32->U8) change the value
                        // and must not be strength-reduced as if linear.
                        if to_ty.size() >= from_ty.size() {
                            let idx = iv_values.get(&v.0).or_else(|| iv_derived.get(&v.0));
                            if let Some(&iv_idx) = idx {
                                if !iv_derived.contains_key(&dest.0) {
                                    new_entries.push((dest.0, iv_idx));
                                }
                            }
                        }
                    }
                    Instruction::Copy { dest, src: Operand::Value(v) } => {
                        let idx = iv_values.get(&v.0).or_else(|| iv_derived.get(&v.0));
                        if let Some(&iv_idx) = idx {
                            if !iv_derived.contains_key(&dest.0) {
                                new_entries.push((dest.0, iv_idx));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if new_entries.is_empty() {
            break;
        }
        for (k, v) in new_entries {
            iv_derived.insert(k, v);
        }
    }

    // Look up whether a value derives from an IV
    let find_iv = |val_id: u32| -> Option<usize> {
        iv_values.get(&val_id).or_else(|| iv_derived.get(&val_id)).copied()
    };

    // Build a set of values used as pointers (in Load/Store) for identifying
    // BinOp::Add-based pointer arithmetic patterns.
    let ptr_uses: FxHashSet<u32> = {
        let mut s = FxHashSet::default();
        for &bi in loop_body {
            if bi >= func.blocks.len() {
                continue;
            }
            for inst in &func.blocks[bi].instructions {
                match inst {
                    Instruction::Load { ptr, .. } => { s.insert(ptr.0); }
                    Instruction::Store { ptr, .. } => { s.insert(ptr.0); }
                    _ => {}
                }
            }
        }
        s
    };

    // Find multiplications/shifts of IV values by constants
    for &bi in loop_body {
        if bi >= func.blocks.len() {
            continue;
        }
        for inst in func.blocks[bi].instructions.iter() {
            let (mul_dest, iv_idx, stride) = match inst {
                // Multiply by constant
                Instruction::BinOp {
                    dest, op: IrBinOp::Mul, lhs, rhs, ..
                } => {
                    match (lhs, rhs) {
                        (Operand::Value(v), Operand::Const(c))
                        | (Operand::Const(c), Operand::Value(v)) => {
                            if let (Some(idx), Some(s)) = (find_iv(v.0), c.to_i64()) {
                                (*dest, idx, s)
                            } else {
                                continue;
                            }
                        }
                        _ => continue,
                    }
                }
                // Shift left by constant (= multiply by 2^k)
                Instruction::BinOp {
                    dest, op: IrBinOp::Shl, lhs: Operand::Value(v), rhs: Operand::Const(c), ..
                } => {
                    if let (Some(idx), Some(shift)) = (find_iv(v.0), c.to_i64()) {
                        if (0..64).contains(&shift) {
                            (*dest, idx, 1i64 << shift)
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };

            // Only worthwhile for strides that are common element sizes
            if stride <= 0 || stride > MAX_IV_STRIDE {
                continue;
            }

            // Find GEPs that use this multiply/shift result as their offset
            let mul_dest_id = mul_dest.0;
            let mut gep_uses = Vec::new();

            for &gbi in loop_body {
                if gbi >= func.blocks.len() {
                    continue;
                }
                for (gii, ginst) in func.blocks[gbi].instructions.iter().enumerate() {
                    if let Instruction::GetElementPtr {
                        dest: gdest,
                        base,
                        offset: Operand::Value(ov),
                        ..
                    } = ginst
                    {
                        if ov.0 == mul_dest_id {
                            gep_uses.push((gbi, gii, *gdest, *base));
                        }
                    }
                }
            }

            // Also find BinOp::Add(ptr, mul_result) where result is used as a pointer.
            // This covers C pointer arithmetic like *(arr + i*sizeof(T)) which lowers
            // to BinOp::Add instead of GetElementPtr.
            for &gbi in loop_body {
                if gbi >= func.blocks.len() {
                    continue;
                }
                for (gii, ginst) in func.blocks[gbi].instructions.iter().enumerate() {
                    if let Instruction::BinOp {
                        dest: adest, op: IrBinOp::Add, lhs, rhs, ..
                    } = ginst
                    {
                        let ptr_op = match (lhs, rhs) {
                            (Operand::Value(v), other) if v.0 == mul_dest_id => Some(other),
                            (other, Operand::Value(v)) if v.0 == mul_dest_id => Some(other),
                            _ => None,
                        };
                        if let Some(Operand::Value(ptr_v)) = ptr_op {
                            if is_loop_invariant(ptr_v.0, loop_body, func)
                                && ptr_uses.contains(&adest.0)
                            {
                                gep_uses.push((gbi, gii, *adest, *ptr_v));
                            }
                        }
                    }
                }
            }

            if !gep_uses.is_empty() {
                derived.push(DerivedExpr {
                    stride,
                    iv_index: iv_idx,
                    gep_uses,
                });
            }
        }
    }

    // Also find IV-derived values used directly as GEP offsets (stride = 1).
    // This handles byte-array patterns like `sieve[j] = 0` where the IV (after
    // cast to I64) is the GEP offset with no multiplication or shift.
    // Collect IV-derived value IDs that are already handled by a Mul/Shl above,
    // so we don't create duplicate reductions.
    let mut already_reduced: FxHashSet<u32> = FxHashSet::default();
    for d in &derived {
        for &(_, _, _, _) in &d.gep_uses {
            // The Mul/Shl dest values are tracked implicitly; what matters is
            // the GEP offset values. We need to avoid reducing a GEP whose
            // offset is already a mul/shl result that we've handled.
        }
    }
    // Actually track which GEPs (by dest) are already reduced
    for d in &derived {
        for &(_, _, gdest, _) in &d.gep_uses {
            already_reduced.insert(gdest.0);
        }
    }

    for &bi in loop_body {
        if bi >= func.blocks.len() {
            continue;
        }
        for (gii, ginst) in func.blocks[bi].instructions.iter().enumerate() {
            if let Instruction::GetElementPtr {
                dest: gdest,
                base,
                offset: Operand::Value(ov),
                ..
            } = ginst
            {
                // Skip if this GEP is already reduced by a Mul/Shl pattern
                if already_reduced.contains(&gdest.0) {
                    continue;
                }
                // Check if the offset value derives from an IV (through Cast/Copy)
                if let Some(iv_idx) = find_iv(ov.0) {
                    derived.push(DerivedExpr {
                        stride: 1,
                        iv_index: iv_idx,
                        gep_uses: vec![(bi, gii, *gdest, *base)],
                    });
                }
            }
        }
    }

    derived
}

/// Look through Cast and Copy instructions to find the root value.
/// Used to match `Cast(Add(Cast(phi_dest), step))` patterns.
fn look_through_casts(val_id: u32, loop_defs: &FxHashMap<u32, &Instruction>) -> u32 {
    let mut current = val_id;
    for _ in 0..MAX_CAST_CHAIN_LENGTH {
        if let Some(inst) = loop_defs.get(&current) {
            match inst {
                Instruction::Cast { src: Operand::Value(v), .. }
                | Instruction::Copy { src: Operand::Value(v), .. } => {
                    current = v.0;
                }
                _ => break,
            }
        } else {
            break;
        }
    }
    current
}

/// Try to resolve an operand to a constant i64 value, looking through Copies.
fn try_resolve_const(op: &Operand, func: &IrFunction) -> Option<i64> {
    match op {
        Operand::Const(c) => c.to_i64(),
        Operand::Value(v) => {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if let Instruction::Copy { dest, src } = inst {
                        if *dest == *v {
                            return try_resolve_const(src, func);
                        }
                    }
                }
            }
            None
        }
    }
}

/// Check if a value is loop-invariant (defined outside the loop).
fn is_loop_invariant(val_id: u32, loop_body: &FxHashSet<usize>, func: &IrFunction) -> bool {
    for &bi in loop_body {
        if bi >= func.blocks.len() {
            continue;
        }
        for inst in &func.blocks[bi].instructions {
            if let Some(dest) = inst.dest() {
                if dest.0 == val_id {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::{AddressSpace, IrType};
    use crate::ir::reexports::{BasicBlock, BlockId, IrCmpOp, Terminator};

    /// Test basic IV detection on a simple counting loop.
    #[test]
    fn test_find_basic_iv() {
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);

        // Block 0 (preheader): init = 0
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![Instruction::Copy {
                dest: Value(0),
                src: Operand::Const(IrConst::I32(0)),
            }],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): i = phi(0, i_next)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(1),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(0)), BlockId(0)),
                        (Operand::Value(Value(3)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(2),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(100)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(2)),
                true_label: BlockId(2),
                false_label: BlockId(3),
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): i_next = i + 1
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![Instruction::BinOp {
                dest: Value(3),
                op: IrBinOp::Add,
                lhs: Operand::Value(Value(1)),
                rhs: Operand::Const(IrConst::I32(1)),
                ty: IrType::I32,
            }],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 4;

        let ivs = find_basic_ivs(&func, 1, &[1, 2].iter().copied().collect(), 0, &[2]);
        assert_eq!(ivs.len(), 1);
        assert_eq!(ivs[0].phi_dest, Value(1));
        assert_eq!(ivs[0].step, 1);
    }

    /// Test full IVSR transformation on a sum-array loop.
    #[test]
    fn test_ivsr_sum_array() {
        let mut func = IrFunction::new("sum_array".into(), IrType::I64, vec![], false);

        // Block 0 (preheader): base = param, n = param, init = 0
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(0x1000)),
                },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Const(IrConst::I32(100)),
                },
                Instruction::Copy {
                    dest: Value(2),
                    src: Operand::Const(IrConst::I32(0)),
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): i = phi(0, i_next)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(3),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(2)), BlockId(0)),
                        (Operand::Value(Value(10)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(4),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Value(Value(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(4)),
                true_label: BlockId(2),
                false_label: BlockId(3),
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): cast, mul, GEP, load, i_next
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Cast {
                    dest: Value(5),
                    src: Operand::Value(Value(3)),
                    from_ty: IrType::I32,
                    to_ty: IrType::I64,
                },
                Instruction::BinOp {
                    dest: Value(6),
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(Value(5)),
                    rhs: Operand::Const(IrConst::I64(4)),
                    ty: IrType::I64,
                },
                Instruction::GetElementPtr {
                    dest: Value(7),
                    base: Value(0),
                    offset: Operand::Value(Value(6)),
                    ty: IrType::I32,
                },
                Instruction::Load {
                    dest: Value(8),
                    ptr: Value(7),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::BinOp {
                    dest: Value(9),
                    op: IrBinOp::Add,
                    lhs: Operand::Const(IrConst::I64(0)),
                    rhs: Operand::Const(IrConst::I64(0)),
                    ty: IrType::I64,
                },
                Instruction::BinOp {
                    dest: Value(10),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I64(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 11;

        let changes = ivsr_function(&mut func);
        assert!(changes > 0, "Expected IVSR to make changes");

        // Check that a phi for the pointer IV was added to the header
        let header_phis: Vec<_> = func.blocks[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Phi { .. }))
            .collect();
        assert!(
            header_phis.len() >= 2,
            "Expected at least 2 phis in header"
        );

        // Check that the original GEP was replaced with a Copy
        let body_copies: Vec<_> = func.blocks[2]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Copy { dest: Value(7), .. }))
            .collect();
        assert_eq!(body_copies.len(), 1, "Expected GEP to be replaced with Copy");
    }

    /// Test GEP-based IV increment detection (Change 1).
    /// Pattern: ptr = phi(init, GEP(ptr, 8))
    #[test]
    fn test_ivsr_gep_increment() {
        let mut func = IrFunction::new("test_gep_inc".into(), IrType::I32, vec![], false);

        // Block 0 (preheader): ptr_init = some base pointer
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![Instruction::Copy {
                dest: Value(0),
                src: Operand::Const(IrConst::I64(0x1000)),
            }],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): ptr = phi(ptr_init from B0, ptr_next from B2)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(1),
                    ty: IrType::Ptr,
                    incoming: vec![
                        (Operand::Value(Value(0)), BlockId(0)),
                        (Operand::Value(Value(3)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(2),
                    op: IrCmpOp::Ult,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I64(0x2000)),
                    ty: IrType::I64,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(2)),
                true_label: BlockId(2),
                false_label: BlockId(3),
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): ptr_next = GEP(ptr, 8)
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![Instruction::GetElementPtr {
                dest: Value(3),
                base: Value(1),
                offset: Operand::Const(IrConst::I64(8)),
                ty: IrType::I8,
            }],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 4;

        let ivs = find_basic_ivs(&func, 1, &[1, 2].iter().copied().collect(), 0, &[2]);
        assert_eq!(ivs.len(), 1, "Should detect GEP-incremented pointer IV");
        assert_eq!(ivs[0].phi_dest, Value(1));
        assert_eq!(ivs[0].step, 8);
        assert_eq!(ivs[0].ty, IrType::Ptr);
    }

    /// Test BinOp::Add-based pointer arithmetic detection (Change 2).
    /// Pattern: Mul(iv, 4) -> BinOp::Add(arr_base, mul) -> Load
    #[test]
    fn test_ivsr_add_based_ptr_arith() {
        let mut func = IrFunction::new("test_add_ptr".into(), IrType::I32, vec![], false);

        // Block 0 (preheader): arr_base, n, init=0
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(0x1000)),
                },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Const(IrConst::I32(100)),
                },
                Instruction::Copy {
                    dest: Value(2),
                    src: Operand::Const(IrConst::I32(0)),
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): i = phi(0, i_next)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(3),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(2)), BlockId(0)),
                        (Operand::Value(Value(9)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(4),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Value(Value(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(4)),
                true_label: BlockId(2),
                false_label: BlockId(3),
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): cast, mul, add(ptr,offset), load, i++
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Cast {
                    dest: Value(5),
                    src: Operand::Value(Value(3)),
                    from_ty: IrType::I32,
                    to_ty: IrType::I64,
                },
                Instruction::BinOp {
                    dest: Value(6),
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(Value(5)),
                    rhs: Operand::Const(IrConst::I64(4)),
                    ty: IrType::I64,
                },
                Instruction::BinOp {
                    dest: Value(7),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(0)),  // arr_base (loop-invariant)
                    rhs: Operand::Value(Value(6)),  // mul result
                    ty: IrType::I64,
                },
                Instruction::Load {
                    dest: Value(8),
                    ptr: Value(7),  // load from add result
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::BinOp {
                    dest: Value(9),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 10;

        let changes = ivsr_function(&mut func);
        assert!(changes > 0, "Expected IVSR to reduce BinOp::Add-based ptr arith");

        // The BinOp::Add (v7) should be replaced with a Copy from a pointer IV
        let body_copies: Vec<_> = func.blocks[2]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Copy { dest: Value(7), .. }))
            .collect();
        assert_eq!(body_copies.len(), 1, "Expected Add to be replaced with Copy");

        // Header should have a new phi for the pointer IV
        let header_phis: Vec<_> = func.blocks[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Phi { .. }))
            .collect();
        assert!(header_phis.len() >= 2, "Expected ptr IV phi in header");
    }

    /// Test IVSR with two arrays using the same IV in the same loop.
    #[test]
    fn test_ivsr_multi_array() {
        let mut func = IrFunction::new("test_multi".into(), IrType::I32, vec![], false);

        // Block 0 (preheader): base1, base2, init=0
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(0x1000)), // base1
                },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Const(IrConst::I64(0x2000)), // base2
                },
                Instruction::Copy {
                    dest: Value(2),
                    src: Operand::Const(IrConst::I32(0)), // init
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): i = phi(0, i_next)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(3),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(2)), BlockId(0)),
                        (Operand::Value(Value(12)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(4),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Const(IrConst::I32(100)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(4)),
                true_label: BlockId(2),
                false_label: BlockId(3),
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): cast, mul, GEP1, load1, GEP2, load2, add, i++
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Cast {
                    dest: Value(5),
                    src: Operand::Value(Value(3)),
                    from_ty: IrType::I32,
                    to_ty: IrType::I64,
                },
                Instruction::BinOp {
                    dest: Value(6),
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(Value(5)),
                    rhs: Operand::Const(IrConst::I64(4)),
                    ty: IrType::I64,
                },
                Instruction::GetElementPtr {
                    dest: Value(7),
                    base: Value(0), // base1
                    offset: Operand::Value(Value(6)),
                    ty: IrType::I32,
                },
                Instruction::Load {
                    dest: Value(8),
                    ptr: Value(7),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::GetElementPtr {
                    dest: Value(9),
                    base: Value(1), // base2
                    offset: Operand::Value(Value(6)),
                    ty: IrType::I32,
                },
                Instruction::Load {
                    dest: Value(10),
                    ptr: Value(9),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::BinOp {
                    dest: Value(11),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(8)),
                    rhs: Operand::Value(Value(10)),
                    ty: IrType::I32,
                },
                Instruction::BinOp {
                    dest: Value(12),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 13;

        let changes = ivsr_function(&mut func);
        assert!(changes >= 2, "Expected at least 2 reductions (one per array)");

        // Both GEPs should be replaced with Copies
        let body_copies: Vec<_> = func.blocks[2]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Copy { .. }))
            .collect();
        assert!(body_copies.len() >= 2, "Expected both GEPs replaced with Copies");

        // Header should have 3 phis (original i + 2 pointer IVs)
        let header_phis: Vec<_> = func.blocks[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Phi { .. }))
            .collect();
        assert_eq!(header_phis.len(), 3, "Expected 3 phis in header (i + 2 ptr IVs)");
    }

    /// Test IVSR for byte array with stride=1 (no multiply).
    #[test]
    fn test_ivsr_stride1_byte_array() {
        let mut func = IrFunction::new("test_byte".into(), IrType::I32, vec![], false);

        // Block 0 (preheader): base, init=0
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(0x1000)), // base
                },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Const(IrConst::I32(0)), // init
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): i = phi(0, i_next)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(2),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(1)), BlockId(0)),
                        (Operand::Value(Value(7)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(3),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I32(100)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(3)),
                true_label: BlockId(2),
                false_label: BlockId(3),
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): cast, GEP(base, cast), store, i++
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Cast {
                    dest: Value(4),
                    src: Operand::Value(Value(2)),
                    from_ty: IrType::I32,
                    to_ty: IrType::I64,
                },
                Instruction::GetElementPtr {
                    dest: Value(5),
                    base: Value(0),
                    offset: Operand::Value(Value(4)), // stride 1, no multiply
                    ty: IrType::I8,
                },
                Instruction::Store {
                    val: Operand::Const(IrConst::I8(0)),
                    ptr: Value(5),
                    ty: IrType::I8,
                    seg_override: AddressSpace::Default,
                },
                Instruction::BinOp {
                    dest: Value(7),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 8;

        let changes = ivsr_function(&mut func);
        assert!(changes > 0, "Expected IVSR to reduce stride-1 byte array");

        // GEP should be replaced with Copy
        let body_copies: Vec<_> = func.blocks[2]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Copy { dest: Value(5), .. }))
            .collect();
        assert_eq!(body_copies.len(), 1, "Expected GEP replaced with Copy");
    }

    /// Test IVSR with negative step (decrementing loop).
    #[test]
    fn test_ivsr_negative_step() {
        let mut func = IrFunction::new("test_neg".into(), IrType::I32, vec![], false);

        // Block 0 (preheader): base, init=99
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(0x1000)), // base
                },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Const(IrConst::I32(99)), // init = 99
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): i = phi(99, i_next)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(2),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(1)), BlockId(0)),
                        (Operand::Value(Value(8)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(3),
                    op: IrCmpOp::Sge,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I32(0)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(3)),
                true_label: BlockId(2),
                false_label: BlockId(3),
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): cast, mul(i,4), GEP, load, i--
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Cast {
                    dest: Value(4),
                    src: Operand::Value(Value(2)),
                    from_ty: IrType::I32,
                    to_ty: IrType::I64,
                },
                Instruction::BinOp {
                    dest: Value(5),
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(Value(4)),
                    rhs: Operand::Const(IrConst::I64(4)),
                    ty: IrType::I64,
                },
                Instruction::GetElementPtr {
                    dest: Value(6),
                    base: Value(0),
                    offset: Operand::Value(Value(5)),
                    ty: IrType::I32,
                },
                Instruction::Load {
                    dest: Value(7),
                    ptr: Value(6),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::BinOp {
                    dest: Value(8),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I32(-1)), // i--
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 9;

        let changes = ivsr_function(&mut func);
        assert!(changes > 0, "Expected IVSR to reduce negative-step loop");

        // GEP should be replaced with Copy
        let body_copies: Vec<_> = func.blocks[2]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Copy { dest: Value(6), .. }))
            .collect();
        assert_eq!(body_copies.len(), 1, "Expected GEP replaced with Copy");

        // The back-edge GEP increment should have negative offset (-4)
        let back_geps: Vec<_> = func.blocks[2]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::GetElementPtr {
                offset: Operand::Const(IrConst::I64(-4)), ..
            }))
            .collect();
        assert_eq!(back_geps.len(), 1, "Expected ptr increment with -4 stride");
    }
}
