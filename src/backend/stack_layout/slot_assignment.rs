//! Slot assignment: classification of instructions into allocation tiers,
//! block-local greedy slot reuse (Tier 3), liveness-based packing (Tier 2),
//! deferred slot finalization, copy alias resolution, and wide value propagation.
//!
//! This module implements Phases 2-7 of the three-tier stack allocation scheme.

use crate::ir::reexports::{
    Instruction,
    IrConst,
    IrFunction,
    Operand,
    Value,
};
use crate::common::types::IrType;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::backend::state::StackSlot;
use crate::backend::regalloc::PhysReg;
use crate::backend::liveness::{
    for_each_operand_in_instruction, for_each_value_use_in_instruction,
    for_each_operand_in_terminator, compute_live_intervals,
};

use super::{
    DeferredSlot, MultiBlockValue, BlockLocalValue, StackLayoutContext,
};

/// Determine if a non-alloca value can be assigned to a block-local pool slot (Tier 3).
/// Returns `Some(def_block_idx)` if the value is defined and used only within a
/// single block, making it safe to share stack space with values from other blocks.
pub(super) fn coalescable_group(
    val_id: u32,
    ctx: &StackLayoutContext,
) -> Option<usize> {
    if !ctx.coalesce {
        return None;
    }
    // Values defined in multiple blocks (from phi elimination) must use Tier 2.
    if ctx.multi_def_values.contains(&val_id) {
        return None;
    }
    if let Some(&def_blk) = ctx.def_block.get(&val_id) {
        if let Some(blocks) = ctx.use_blocks_map.get(&val_id) {
            let mut unique: Vec<usize> = blocks.clone();
            unique.sort_unstable();
            unique.dedup();

            if unique.is_empty() {
                return Some(def_blk); // Dead value, safe to coalesce.
            }

            // Single-block value: defined and used in the same block.
            if unique.len() == 1 && unique[0] == def_blk {
                return Some(def_blk);
            }
        } else {
            return Some(def_blk); // No uses - dead value.
        }
    }
    None
}

/// Walk all instructions and classify each into Tier 1 (permanent alloca slots),
/// Tier 2 (multi-block, liveness-packed), or Tier 3 (block-local, greedy reuse).
pub(super) fn classify_instructions(
    state: &mut crate::backend::state::CodegenState,
    func: &IrFunction,
    ctx: &StackLayoutContext,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
    reg_assigned: &FxHashMap<u32, PhysReg>,
    non_local_space: &mut i64,
    deferred_slots: &mut Vec<DeferredSlot>,
    multi_block_values: &mut Vec<MultiBlockValue>,
    block_local_values: &mut Vec<BlockLocalValue>,
    block_space: &mut FxHashMap<usize, i64>,
    max_block_local_space: &mut i64,
) {
    let mut collected_values: FxHashSet<u32> = FxHashSet::default();

    // Build set of values that are defined (as dest) by non-InlineAsm
    // instructions. This identifies "indirect" asm output pointers:
    //
    // When an InlineAsm output is "=r"(*ptr), the output value is the
    // loaded pointer (from a Load that became a Copy after mem2reg).
    // This value is defined by a Copy/Load/Phi instruction, AND it
    // appears as an InlineAsm output. The asm result must be stored
    // THROUGH the pointer, not directly into a stack slot.
    //
    // When an InlineAsm output is "=r"(x) and x was promoted by mem2reg,
    // the output value is a fresh SSA value created by mem2reg. This
    // value is NOT defined by any other instruction -- it only appears
    // in the InlineAsm outputs. This value DOES need a direct stack slot.
    //
    // The distinction: if an InlineAsm output value is also the dest of
    // a non-InlineAsm instruction, it's an indirect pointer that should
    // NOT be promoted to a direct asm output slot.
    let mut non_asm_defined: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if !matches!(inst, Instruction::InlineAsm { .. }) {
                if let Some(dest) = inst.dest() {
                    non_asm_defined.insert(dest.0);
                }
            }
        }
    }

    // Pre-scan: find param allocas whose memory is modified after the initial
    // emit_store_params Store. The ParamRef optimization (reusing the alloca
    // slot for the initial parameter value) is only safe when the alloca's
    // content is never overwritten. If any additional store targets the param
    // alloca (directly, through a GEP, or via an escaped pointer to a callee),
    // the alloca may hold a different value than the ParamRef expects.
    let modified_param_allocas: FxHashSet<u32> = {
        let param_alloca_set: FxHashSet<u32> =
            func.param_alloca_values.iter().map(|v| v.0).collect();

        // Map GEP dest -> param alloca root (for chained GEPs)
        let mut gep_to_param: FxHashMap<u32, u32> = FxHashMap::default();
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::GetElementPtr { dest, base, .. } = inst {
                    if param_alloca_set.contains(&base.0) {
                        gep_to_param.insert(dest.0, base.0);
                    } else if let Some(&root) = gep_to_param.get(&base.0) {
                        gep_to_param.insert(dest.0, root);
                    }
                }
            }
        }

        // Count stores to each param alloca. The initial emit_store_params
        // generates exactly one store per param. Any additional store means
        // the param alloca is modified.
        let mut store_count: FxHashMap<u32, u32> = FxHashMap::default();
        let mut escaped = FxHashSet::default();
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::Store { ptr, .. } => {
                        // Direct store to param alloca
                        if param_alloca_set.contains(&ptr.0) {
                            *store_count.entry(ptr.0).or_insert(0) += 1;
                        }
                        // Store through GEP of param alloca
                        if let Some(&root) = gep_to_param.get(&ptr.0) {
                            *store_count.entry(root).or_insert(0) += 1;
                        }
                    }
                    Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                        // If param alloca address (or GEP of it) is passed to
                        // a call, the callee may modify it.
                        for arg in &info.args {
                            if let Operand::Value(v) = arg {
                                if let Some(&root) = gep_to_param.get(&v.0) {
                                    escaped.insert(root);
                                }
                                if param_alloca_set.contains(&v.0) {
                                    escaped.insert(v.0);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // A param alloca is "modified" if it has more than 1 store (the
        // initial emit_store_params store) or its address escapes to a call.
        let mut modified = escaped;
        for (&alloca_id, &count) in &store_count {
            if count > 1 {
                modified.insert(alloca_id);
            }
        }
        modified
    };

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, size, ty, align, .. } = inst {
                classify_alloca(
                    state, dest, *size, *ty, *align, ctx,
                    assign_slot, non_local_space, deferred_slots,
                    block_space, max_block_local_space,
                );
            } else if let Instruction::InlineAsm { outputs, operand_types, .. } = inst {
                // Promoted InlineAsm output values need stack slots to hold
                // the output register value. These are "direct" slots (like
                // allocas) -- the slot contains the value itself, not a pointer.
                //
                // However, values defined by non-InlineAsm instructions (Copy,
                // Load, Phi, etc.) are pointer dereference outputs (e.g.,
                // "=r"(*ptr)). After mem2reg, the Load that produced the
                // pointer becomes a Copy, but the value still represents a
                // pre-existing pointer. These must NOT be promoted to direct
                // slots -- their stack slot holds the pointer itself, and
                // store_output_from_reg must store the asm result THROUGH
                // the pointer. Promoting them would cause the result to be
                // written to the slot instead of through the pointer.
                //
                // This check is more robust than checking reg_assigned alone,
                // because it also handles the case where the pointer value
                // is NOT register-allocated (e.g., due to register pressure
                // forcing the pointer to a stack slot).
                for (out_idx, (_, out_val, _)) in outputs.iter().enumerate() {
                    if !state.alloca_values.contains(&out_val.0)
                        && !non_asm_defined.contains(&out_val.0)
                        && collected_values.insert(out_val.0)
                    {
                        let slot_size: i64 = if out_idx < operand_types.len() {
                            match operand_types[out_idx] {
                                IrType::I128 | IrType::U128 | IrType::F128 => 16,
                                _ => 8,
                            }
                        } else {
                            8
                        };
                        state.asm_output_values.insert(out_val.0);
                        // On 32-bit targets, track 64-bit asm output values as "wide"
                        // so that Copy instructions referencing them use the multi-word
                        // copy path (copying both 32-bit halves) instead of only copying
                        // the low 32 bits. Without this, mem2reg-promoted 64-bit inline
                        // asm outputs (e.g., "+r" on unsigned long long) lose their high
                        // 32 bits when the value is copied to subsequent uses.
                        if crate::common::types::target_is_32bit() && out_idx < operand_types.len() {
                            let is_wide = matches!(operand_types[out_idx],
                                IrType::F64 | IrType::I64 | IrType::U64);
                            if is_wide {
                                state.wide_values.insert(out_val.0);
                            }
                        }
                        multi_block_values.push(MultiBlockValue {
                            dest_id: out_val.0,
                            slot_size,
                        });
                    }
                }
            } else if let Instruction::ParamRef { dest, param_idx, ty } = inst {
                // ParamRef loads a parameter value from its alloca slot.
                // Instead of allocating a separate stack slot for the ParamRef
                // dest, reuse the param alloca's slot. This saves 8 bytes per
                // promoted parameter (significant for kernel functions with
                // many parameters where frame size is critical).
                //
                // Safety: the alloca slot is rounded up to 8 bytes (by the
                // assign_slot callback), so storing a full 8-byte movq is safe.
                // emit_param_ref loads from the alloca with sign/zero extension,
                // then stores back to the same slot, which is a valid self-update
                // that sets the upper bytes to the correct extension.
                //
                // Exception: when the param alloca is modified after the initial
                // store (e.g., by an inlined callee writing to it directly, or
                // its address escaping to a call), the ParamRef must have its
                // own separate slot. Otherwise the ParamRef would read back the
                // modified value instead of the original parameter value.
                if *param_idx < func.param_alloca_values.len() {
                    let alloca_val = func.param_alloca_values[*param_idx];
                    if !modified_param_allocas.contains(&alloca_val.0)
                       && !reg_assigned.contains_key(&dest.0) {
                        if let Some(&slot) = state.value_locations.get(&alloca_val.0) {
                            state.value_locations.insert(dest.0, slot);
                            // Propagate type tracking even when reusing the alloca
                            // slot, so downstream Copy instructions use the correct
                            // multi-word paths for wide/i128/f128 values.
                            if matches!(ty, IrType::I128 | IrType::U128) {
                                state.i128_values.insert(dest.0);
                            }
                            if crate::common::types::target_is_32bit()
                                && matches!(ty, IrType::F64 | IrType::I64 | IrType::U64)
                            {
                                state.wide_values.insert(dest.0);
                            }
                            continue;
                        }
                    }
                }
                // Fallthrough: if alloca not found or modified, classify normally.
                classify_value(
                    state, *dest, inst, ctx, reg_assigned,
                    &mut collected_values, multi_block_values,
                    block_local_values,
                );
            } else if let Some(dest) = inst.dest() {
                classify_value(
                    state, dest, inst, ctx, reg_assigned,
                    &mut collected_values, multi_block_values,
                    block_local_values,
                );
            }
        }
    }
}

/// Classify a single Alloca instruction into Tier 1 (permanent) or Tier 3 (block-local).
fn classify_alloca(
    state: &mut crate::backend::state::CodegenState,
    dest: &Value,
    size: usize,
    ty: IrType,
    align: usize,
    ctx: &StackLayoutContext,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
    non_local_space: &mut i64,
    deferred_slots: &mut Vec<DeferredSlot>,
    block_space: &mut FxHashMap<usize, i64>,
    max_block_local_space: &mut i64,
) {
    let effective_align = align;
    let extra = if effective_align > 16 { effective_align - 1 } else { 0 };
    let ptr_size = crate::common::types::target_ptr_size() as i64;
    // Alloca slots must be at least pointer-sized (8 bytes on 64-bit, 4 on 32-bit)
    // to safely hold ParamRef values that store via movq/sd (full register width).
    let raw_size = if size == 0 {
        ptr_size
    } else {
        (size as i64).max(ptr_size)
    };

    state.alloca_values.insert(dest.0);
    state.alloca_types.insert(dest.0, ty);
    if effective_align > 16 {
        state.alloca_alignments.insert(dest.0, effective_align);
    }

    // Skip dead param allocas (still registered so backend recognizes them).
    if ctx.dead_param_allocas.contains(&dest.0) {
        return;
    }

    // Skip dead non-param allocas (never referenced by any instruction).
    if ctx.coalescable_allocas.dead.contains(&dest.0) {
        return;
    }

    // Single-block allocas: use block-local coalescing (Tier 3).
    // Over-aligned allocas (> 16) are excluded because their alignment
    // padding complicates coalescing.
    if effective_align <= 16 {
        if let Some(&use_block) = ctx.coalescable_allocas.single_block.get(&dest.0) {
            let alloca_size = raw_size + extra as i64;
            let alloca_align = align as i64;
            let bs = block_space.entry(use_block).or_insert(0);
            let before = *bs;
            let (_, new_space) = assign_slot(*bs, alloca_size, alloca_align);
            *bs = new_space;
            if new_space > *max_block_local_space {
                *max_block_local_space = new_space;
            }
            deferred_slots.push(DeferredSlot {
                dest_id: dest.0, size: alloca_size, align: alloca_align,
                block_offset: before,
            });
            return;
        }
    }

    // Non-coalescable allocas get permanent Tier 1 slots.
    let (slot, new_space) = assign_slot(*non_local_space, raw_size + extra as i64, align as i64);
    state.value_locations.insert(dest.0, StackSlot(slot));
    *non_local_space = new_space;
}

/// Classify a non-alloca value into Tier 2 (multi-block) or Tier 3 (block-local).
fn classify_value(
    state: &mut crate::backend::state::CodegenState,
    dest: Value,
    inst: &Instruction,
    ctx: &StackLayoutContext,
    reg_assigned: &FxHashMap<u32, PhysReg>,
    collected_values: &mut FxHashSet<u32>,
    multi_block_values: &mut Vec<MultiBlockValue>,
    block_local_values: &mut Vec<BlockLocalValue>,
) {
    let mut is_i128 = matches!(inst.result_type(), Some(IrType::I128) | Some(IrType::U128));
    let is_f128 = matches!(inst.result_type(), Some(IrType::F128))
        || matches!(inst, Instruction::Copy { src: Operand::Const(IrConst::LongDouble(..)), .. });

    // Copy instructions have result_type() = None, so we must check whether
    // the source operand is an I128 value. If it is, the Copy dest also needs
    // a 16-byte slot; otherwise the codegen's emit_copy_i128 will overflow an
    // 8-byte slot into the adjacent stack slot, corrupting other values.
    if !is_i128 {
        if let Instruction::Copy { src: Operand::Value(src_val), .. } = inst {
            if state.i128_values.contains(&src_val.0) {
                is_i128 = true;
            }
        }
    }

    // Detect small values (types that fit in 4 bytes on 64-bit targets).
    // Currently used to populate small_slot_values for future store/load
    // width optimization. Slot allocation remains 8-byte minimum because
    // the backend's store/load paths aren't fully type-safe yet (some paths
    // always use movq/sd/str x0 regardless of IR type).
    let is_small = !crate::common::types::target_is_32bit() && matches!(
        inst.result_type(),
        Some(IrType::I8) | Some(IrType::U8) |
        Some(IrType::I16) | Some(IrType::U16) |
        Some(IrType::I32) | Some(IrType::U32) |
        Some(IrType::F32)
    );
    let slot_size: i64 = if is_i128 || is_f128 {
        16
    } else if is_small {
        4
    } else {
        8
    };

    if is_i128 {
        state.i128_values.insert(dest.0);
    }

    // On 32-bit targets, track values wider than 32 bits for multi-word copy handling.
    if crate::common::types::target_is_32bit() {
        let is_wide = matches!(inst.result_type(),
            Some(IrType::F64) | Some(IrType::I64) | Some(IrType::U64));
        if is_wide {
            state.wide_values.insert(dest.0);
        }
    }

    // Skip register-assigned values (no stack slot needed).
    if reg_assigned.contains_key(&dest.0) {
        return;
    }

    // Skip dead values (defined but never used).
    if !ctx.used_values.contains(&dest.0) {
        return;
    }

    // Skip copy-aliased values (they'll share root's slot). Not for i128/f128.
    if !is_i128 && !is_f128 && ctx.copy_alias.contains_key(&dest.0) {
        return;
    }

    // Skip immediately-consumed values: produced and consumed in adjacent
    // instructions, kept alive in the accumulator register cache without
    // needing a stack slot. Not for i128/f128 (need 16-byte special handling).
    // On 32-bit targets, also exclude F64/I64/U64 ("wide" values) because
    // they can't fit in the 32-bit accumulator (EAX). F64 values use x87
    // and must be stored to memory between operations; I64/U64 need
    // multi-word handling via edx:eax pairs that require stack slots.
    let is_wide_on_32bit = crate::common::types::target_is_32bit() && matches!(
        inst.result_type(),
        Some(IrType::F64) | Some(IrType::I64) | Some(IrType::U64)
    );
    if !is_i128 && !is_f128 && !is_wide_on_32bit && ctx.immediately_consumed.contains(&dest.0) {
        return;
    }

    // Dedup multi-def values (phi results appear in multiple blocks).
    if !collected_values.insert(dest.0) {
        return;
    }

    // Track values that use 4-byte slots so store/load paths can emit
    // 4-byte instructions (movl, sw/lw, str/ldr w-reg) instead of 8-byte.
    if is_small {
        state.small_slot_values.insert(dest.0);
    }

    if let Some(target_blk) = coalescable_group(dest.0, ctx) {
        block_local_values.push(BlockLocalValue {
            dest_id: dest.0,
            slot_size,
            block_idx: target_blk,
        });
    } else {
        multi_block_values.push(MultiBlockValue {
            dest_id: dest.0,
            slot_size,
        });
    }
}

/// Assign stack slots for block-local values using intra-block greedy reuse.
///
/// Within a single block, values have short lifetimes. By tracking when each
/// value is last used, we can reuse its stack slot for later values. This is
/// critical for functions like blake2s_compress_generic where macro expansion
/// creates thousands of short-lived intermediates in a single loop body block.
pub(super) fn assign_tier3_block_local_slots(
    func: &IrFunction,
    ctx: &StackLayoutContext,
    coalesce: bool,
    block_local_values: &[BlockLocalValue],
    deferred_slots: &mut Vec<DeferredSlot>,
    block_space: &mut FxHashMap<usize, i64>,
    max_block_local_space: &mut i64,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
) {
    if block_local_values.is_empty() {
        return;
    }

    if !coalesce {
        // Fallback: no reuse, just accumulate.
        for blv in block_local_values {
            let bs = block_space.entry(blv.block_idx).or_insert(0);
            let before = *bs;
            let (_, new_space) = assign_slot(*bs, blv.slot_size, 0);
            *bs = new_space;
            if new_space > *max_block_local_space {
                *max_block_local_space = new_space;
            }
            deferred_slots.push(DeferredSlot {
                dest_id: blv.dest_id, size: blv.slot_size, align: 0,
                block_offset: before,
            });
        }
        return;
    }

    // Pre-compute per-block last-use and definition instruction indices.
    let block_local_set: FxHashSet<u32> = block_local_values.iter().map(|v| v.dest_id).collect();
    let mut last_use: FxHashMap<u32, usize> = FxHashMap::default();
    let mut def_inst_idx: FxHashMap<u32, usize> = FxHashMap::default();

    for block in &func.blocks {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let Some(dest) = inst.dest() {
                if block_local_set.contains(&dest.0) {
                    def_inst_idx.insert(dest.0, inst_idx);
                }
            }
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    if block_local_set.contains(&v.0) {
                        last_use.insert(v.0, inst_idx);
                    }
                    // Extend copy-alias root's last_use when the aliased
                    // value is used as an operand.
                    if let Some(&root) = ctx.copy_alias.get(&v.0) {
                        if block_local_set.contains(&root) {
                            let root_last = last_use.get(&root).copied().unwrap_or(0);
                            if inst_idx > root_last {
                                last_use.insert(root, inst_idx);
                            }
                        }
                    }
                }
            });
            for_each_value_use_in_instruction(inst, |v| {
                if block_local_set.contains(&v.0) {
                    last_use.insert(v.0, inst_idx);
                }
                // Extend copy-alias root's last_use when the aliased
                // value is used as a value reference (e.g., dest_ptr in
                // Intrinsic, ptr in Store/Load).
                if let Some(&root) = ctx.copy_alias.get(&v.0) {
                    if block_local_set.contains(&root) {
                        let root_last = last_use.get(&root).copied().unwrap_or(0);
                        if inst_idx > root_last {
                            last_use.insert(root, inst_idx);
                        }
                    }
                }
            });
            // Extend last_use for InlineAsm output pointer values: Phase 4 reads
            // these from stack slots AFTER the asm executes. Also extend
            // copy-alias roots which hold the actual slot.
            if let Instruction::InlineAsm { outputs, .. } = inst {
                for (_, v, _) in outputs {
                    let extended = inst_idx + 1;
                    if block_local_set.contains(&v.0) {
                        last_use.insert(v.0, extended);
                    }
                    if let Some(&root) = ctx.copy_alias.get(&v.0) {
                        if block_local_set.contains(&root) {
                            last_use.insert(root, extended);
                        }
                    }
                }
            }
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                if block_local_set.contains(&v.0) {
                    last_use.insert(v.0, block.instructions.len());
                }
            }
        });

        // F128 source pointer liveness extension (Tier 3 block-local mirror).
        //
        // When an F128 Load uses a pointer, the codegen records that pointer so
        // Call emission can reload the full 128-bit value later. The pointer's
        // slot must stay live until the F128 dest's last use, otherwise the
        // greedy slot coloring reuses it and the Call dereferences garbage.
        for inst in &block.instructions {
            if let Instruction::Load { dest, ptr, ty, .. } = inst {
                if *ty == IrType::F128 && block_local_set.contains(&ptr.0) {
                    if let Some(&dest_last) = last_use.get(&dest.0) {
                        let ptr_last = last_use.get(&ptr.0).copied().unwrap_or(0);
                        if dest_last > ptr_last {
                            last_use.insert(ptr.0, dest_last);
                        }
                        if let Some(&root) = ctx.copy_alias.get(&ptr.0) {
                            if block_local_set.contains(&root) {
                                let root_last = last_use.get(&root).copied().unwrap_or(0);
                                if dest_last > root_last {
                                    last_use.insert(root, dest_last);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Group block-local values by block, preserving definition order.
    let mut per_block: FxHashMap<usize, Vec<(u32, i64)>> = FxHashMap::default();
    for blv in block_local_values {
        per_block.entry(blv.block_idx).or_default().push((blv.dest_id, blv.slot_size));
    }

    // For each block, assign slots with greedy coloring.
    for (blk_idx, values) in &per_block {
        let mut active: Vec<(usize, i64, i64)> = Vec::new(); // (last_use, offset, size)
        let mut free_8: Vec<i64> = Vec::new();
        let mut free_16: Vec<i64> = Vec::new();
        let mut block_peak: i64 = block_space.get(blk_idx).copied().unwrap_or(0);

        for &(dest_id, slot_size) in values {
            let my_def = def_inst_idx.get(&dest_id).copied().unwrap_or(0);

            // Release expired slots.
            let mut i = 0;
            while i < active.len() {
                if active[i].0 < my_def {
                    let (_, off, sz) = active.swap_remove(i);
                    if sz == 16 { free_16.push(off); } else { free_8.push(off); }
                } else {
                    i += 1;
                }
            }

            // Try to reuse a freed slot of matching size.
            let free_list = if slot_size == 16 { &mut free_16 } else { &mut free_8 };
            let offset = if let Some(reused) = free_list.pop() {
                reused
            } else {
                let off = block_peak;
                block_peak += slot_size;
                off
            };

            let my_last = last_use.get(&dest_id).copied().unwrap_or(my_def);
            active.push((my_last, offset, slot_size));

            deferred_slots.push(DeferredSlot {
                dest_id,
                size: slot_size,
                align: 0,
                block_offset: offset,
            });
        }

        if block_peak > *max_block_local_space {
            *max_block_local_space = block_peak;
        }
    }
}

/// Assign stack slots for multi-block values using liveness-based packing.
///
/// Uses a greedy interval coloring algorithm: sort by start point, greedily assign
/// to the first slot whose previous occupant's interval has ended. This is optimal
/// for interval graphs (chromatic number equals clique number).
pub(super) fn assign_tier2_liveness_packed_slots(
    state: &mut crate::backend::state::CodegenState,
    coalesce: bool,
    cached_liveness: Option<crate::backend::liveness::LivenessResult>,
    func: &IrFunction,
    multi_block_values: &[MultiBlockValue],
    non_local_space: &mut i64,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
) {
    if multi_block_values.is_empty() {
        return;
    }

    if !coalesce {
        // Fallback: permanent slots for all multi-block values.
        for mbv in multi_block_values {
            let (slot, new_space) = assign_slot(*non_local_space, mbv.slot_size, 0);
            state.value_locations.insert(mbv.dest_id, StackSlot(slot));
            *non_local_space = new_space;
        }
        return;
    }

    // Reuse liveness data from register allocation when available.
    let liveness = cached_liveness.unwrap_or_else(|| compute_live_intervals(func));

    let mut interval_map: FxHashMap<u32, (u32, u32)> = FxHashMap::default();
    for iv in &liveness.intervals {
        interval_map.insert(iv.value_id, (iv.start, iv.end));
    }

    // Separate by slot size for packing (8-byte and 16-byte pools).
    let mut values_8: Vec<(u32, u32, u32)> = Vec::new();
    let mut values_16: Vec<(u32, u32, u32)> = Vec::new();
    let mut no_interval: Vec<(u32, i64)> = Vec::new();

    for mbv in multi_block_values {
        if let Some(&(start, end)) = interval_map.get(&mbv.dest_id) {
            if mbv.slot_size == 16 {
                values_16.push((mbv.dest_id, start, end));
            } else {
                values_8.push((mbv.dest_id, start, end));
            }
        } else {
            no_interval.push((mbv.dest_id, mbv.slot_size));
        }
    }

    pack_values_into_slots(&mut values_8, state, non_local_space, 8, assign_slot);
    pack_values_into_slots(&mut values_16, state, non_local_space, 16, assign_slot);

    // Assign permanent slots for values without interval info.
    for (dest_id, size) in no_interval {
        let (slot, new_space) = assign_slot(*non_local_space, size, 0);
        state.value_locations.insert(dest_id, StackSlot(slot));
        *non_local_space = new_space;
    }
}

/// Pack values with known live intervals into shared stack slots using a min-heap.
/// O(N log S) where N = values and S = slots.
fn pack_values_into_slots(
    values: &mut [(u32, u32, u32)],
    state: &mut crate::backend::state::CodegenState,
    non_local_space: &mut i64,
    slot_size: i64,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
) {
    if values.is_empty() {
        return;
    }

    values.sort_by_key(|&(_, start, _)| start);

    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut slot_offsets: Vec<i64> = Vec::new();

    for &(dest_id, start, end) in values.iter() {
        if let Some(&Reverse((slot_end, slot_idx))) = heap.peek() {
            if slot_end < start {
                heap.pop();
                let slot_offset = slot_offsets[slot_idx];
                heap.push(Reverse((end, slot_idx)));
                state.value_locations.insert(dest_id, StackSlot(slot_offset));
                continue;
            }
        }
        let slot_idx = slot_offsets.len();
        let (slot, new_space) = assign_slot(*non_local_space, slot_size, 0);
        state.value_locations.insert(dest_id, StackSlot(slot));
        *non_local_space = new_space;
        slot_offsets.push(slot);
        heap.push(Reverse((end, slot_idx)));
    }
}

/// Assign final offsets for deferred block-local values. All deferred values
/// share a pool starting at `non_local_space`; each value's final slot is
/// computed by adding its block-local offset to the global base.
///
/// When coalescable allocas with alignment > 8 are mixed with non-aligned
/// block-local values, `assign_slot(nls + block_offset, size, align)` can
/// produce overlapping slot offsets because alignment rounding in assign_slot
/// may cause differently-sized/aligned values to collapse to the same final
/// offset. To prevent this, we align `non_local_space` up to the maximum
/// alignment required by any deferred slot before computing final offsets.
/// This ensures that `nls + block_offset` preserves the alignment invariants
/// that were established during the block-space accumulation phase.
pub(super) fn finalize_deferred_slots(
    state: &mut crate::backend::state::CodegenState,
    deferred_slots: &[DeferredSlot],
    non_local_space: i64,
    max_block_local_space: i64,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
) -> i64 {
    if !deferred_slots.is_empty() && max_block_local_space > 0 {
        // Find the maximum alignment required by any deferred slot and align
        // non_local_space to it. This prevents alignment rounding in assign_slot
        // from causing adjacent slots to overlap when nls is not aligned.
        let max_align = deferred_slots.iter()
            .map(|ds| if ds.align > 0 { ds.align } else { 8 })
            .max()
            .unwrap_or(8);
        let aligned_nls = if max_align > 8 {
            ((non_local_space + max_align - 1) / max_align) * max_align
        } else {
            non_local_space
        };
        for ds in deferred_slots {
            let (slot, _) = assign_slot(aligned_nls + ds.block_offset, ds.size, ds.align);
            state.value_locations.insert(ds.dest_id, StackSlot(slot));
        }
        aligned_nls + max_block_local_space
    } else {
        non_local_space
    }
}

/// Propagate stack slots from root values to their copy aliases.
/// Each aliased value gets the same StackSlot as its root, eliminating
/// a separate slot allocation and making the Copy a harmless self-move.
pub(super) fn resolve_copy_aliases(
    state: &mut crate::backend::state::CodegenState,
    copy_alias: &FxHashMap<u32, u32>,
) {
    for (&dest_id, &root_id) in copy_alias {
        if let Some(&slot) = state.value_locations.get(&root_id) {
            state.value_locations.insert(dest_id, slot);
        }
        // Propagate small-slot property: if the root uses a 4-byte slot,
        // the aliased copy must also use 4-byte store/load to avoid overflow.
        if state.small_slot_values.contains(&root_id) {
            state.small_slot_values.insert(dest_id);
        }
        // Propagate alloca status: if root is an alloca, the aliased copy
        // must also be recognized as an alloca so codegen computes the
        // stack address instead of loading from the slot.
        if state.alloca_values.contains(&root_id) {
            state.alloca_values.insert(dest_id);
        }
        if let Some(&align) = state.alloca_alignments.get(&root_id) {
            state.alloca_alignments.insert(dest_id, align);
        }
        // If root has no slot (optimized away or reg-assigned), the aliased
        // value also gets no slot. The Copy works via accumulator path.
    }
}

/// On 32-bit targets, propagate wide-value status through Copy chains.
///
/// Copy instructions for 64-bit values (F64, I64, U64) need 8-byte copies
/// (two movl instructions) instead of the default 4-byte. The initial
/// wide_values set only includes typed instructions; phi elimination creates
/// Copy chains where the destination has no type info. We propagate using
/// fixpoint iteration to handle cycles from phi copies.
pub(super) fn propagate_wide_values(
    state: &mut crate::backend::state::CodegenState,
    func: &IrFunction,
    copy_alias: &FxHashMap<u32, u32>,
) {
    if !crate::common::types::target_is_32bit() || state.wide_values.is_empty() {
        return;
    }

    let mut copy_edges: Vec<(u32, u32)> = Vec::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                copy_edges.push((dest.0, src_val.0));
            }
        }
    }
    // Also propagate through copy aliases (forward only: root -> dest).
    for (&dest_id, &root_id) in copy_alias {
        copy_edges.push((dest_id, root_id));
    }

    if copy_edges.is_empty() {
        return;
    }

    // Fixpoint iteration: propagate wide status until stable.
    let mut changed = true;
    let mut iters = 0;
    while changed && iters < 100 {
        changed = false;
        iters += 1;
        for &(dest_id, src_id) in &copy_edges {
            if state.wide_values.contains(&src_id) && !state.wide_values.contains(&dest_id) {
                state.wide_values.insert(dest_id);
                changed = true;
            }
        }
    }
}
