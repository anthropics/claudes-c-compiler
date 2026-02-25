//! Module, function, and instruction generation dispatch.
//!
//! This module contains the top-level entry points that drive code generation:
//! - `generate_module`: emits data sections and iterates over functions
//! - `generate_function`: emits prologue, basic blocks, and epilogue
//! - `generate_instruction`: dispatches each IR instruction to arch trait methods
//! - `generate_terminator`: dispatches terminators to arch trait methods
//!
//! These functions are arch-independent — they use the `ArchCodegen` trait to call
//! into the backend-specific implementations.

use std::rc::Rc;
use crate::ir::reexports::{
    BasicBlock,
    GlobalInit,
    Instruction,
    IrConst,
    IrFunction,
    IrModule,
    Operand,
    Terminator,
    Value,
};
use crate::common::types::{AddressSpace, IrType};
use crate::common::source::{Span, SourceManager};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use super::common;
use super::traits::ArchCodegen;
use super::liveness::{for_each_operand_in_instruction, for_each_value_use_in_instruction, for_each_operand_in_terminator};

/// Information about a GEP with a constant offset that can be folded into
/// Load/Store addressing modes. Instead of computing `base + offset` as a
/// separate instruction and spilling to stack, the constant offset is merged
/// directly into the memory operand of the subsequent load/store.
#[derive(Debug, Clone, Copy)]
pub(super) struct GepFoldInfo {
    /// The base pointer value (an alloca or previously-computed pointer).
    pub(super) base: Value,
    /// The constant byte offset to add to the base address.
    pub(super) offset: i64,
}

/// Build a map of GEP destinations that can be folded into Load/Store instructions.
///
/// A GEP is foldable when:
/// 1. Its offset is a compile-time constant (Operand::Const)
/// 2. The constant fits in a 32-bit signed displacement (x86 addressing limit)
/// 3. The GEP result is only used as the ptr operand of Load/Store instructions
///    (not used by other instructions, terminators, or as a value operand)
///
/// When all conditions are met, the GEP instruction is skipped during codegen,
/// and each Load/Store that uses it receives the (base, offset) directly.
fn build_gep_fold_map(func: &IrFunction, use_counts: &[u32]) -> FxHashMap<u32, GepFoldInfo> {
    let mut gep_map: FxHashMap<u32, GepFoldInfo> = FxHashMap::default();

    // Phase 1: Collect all GEPs with constant offsets.
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::GetElementPtr { dest, base, offset: Operand::Const(c), .. } = inst {
                let offset_val = match c.to_i64() {
                    Some(v) => v,
                    None => continue,
                };
                // Offset must fit in 32-bit signed displacement for x86.
                // Also reasonable for ARM (signed 9-bit unscaled or 12-bit scaled)
                // and RISC-V (signed 12-bit).
                // Use i32 range as the safe common limit.
                // Unsigned type constants (e.g. U32 -1 = 4294967295) are sign-narrowed.
                let offset_val = if offset_val >= i32::MIN as i64 && offset_val <= i32::MAX as i64 {
                    offset_val
                } else if offset_val > i32::MAX as i64 && offset_val <= u32::MAX as i64 {
                    offset_val as i32 as i64
                } else {
                    continue;
                };
                gep_map.insert(dest.0, GepFoldInfo { base: *base, offset: offset_val });
            }
        }
    }

    if gep_map.is_empty() {
        return gep_map;
    }

    // Phase 2: Verify that each candidate GEP dest is ONLY used as Load/Store ptr.
    // If it's used anywhere else (as a value operand, in a call, in a terminator,
    // or as a base of another GEP), we cannot fold it.
    //
    // Strategy: Load.ptr and Store.ptr are the ONLY foldable use positions.
    // - Load: ptr is a Value (visited by for_each_value_use), no Operand uses → skip entirely.
    // - Store: ptr (Value) is foldable, but val (Operand) is NOT → check only Operand uses.
    // - All other instructions: ANY reference to a GEP dest invalidates folding.
    let mut non_ptr_uses: FxHashSet<u32> = FxHashSet::default();

    // Helper: mark a GEP dest as non-foldable if used outside Load/Store ptr position.
    let mut mark_non_ptr = |id: u32| {
        if gep_map.contains_key(&id) {
            non_ptr_uses.insert(id);
        }
    };

    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                // Load.ptr is foldable — UNLESS:
                // - The load type is i128/u128: the i128 load path doesn't
                //   support GEP folding and falls through to emit_load.
                // - The load has a segment override (%gs:/%fs:): the segment-
                //   overridden load path (emit_seg_load) returns early before
                //   the GEP fold check, so it needs the pointer value to be
                //   computed by the GEP instruction (not folded away).
                Instruction::Load { ptr, ty, seg_override, .. } => {
                    if matches!(ty, IrType::I128 | IrType::U128)
                        || *seg_override != AddressSpace::Default {
                        mark_non_ptr(ptr.0);
                    }
                }
                // Store.ptr is foldable, but Store.val is an Operand that is NOT foldable.
                // Also invalidate if the store type is i128/u128 or has a segment
                // override, for the same reasons as Load above.
                Instruction::Store { val, ptr, ty, seg_override, .. } => {
                    if let Operand::Value(v) = val { mark_non_ptr(v.0); }
                    if matches!(ty, IrType::I128 | IrType::U128)
                        || *seg_override != AddressSpace::Default {
                        mark_non_ptr(ptr.0);
                    }
                }
                // All other instructions: any reference invalidates folding.
                _ => {
                    for_each_operand_in_instruction(inst, |op| {
                        if let Operand::Value(v) = op { mark_non_ptr(v.0); }
                    });
                    for_each_value_use_in_instruction(inst, |v| mark_non_ptr(v.0));
                }
            }
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op { mark_non_ptr(v.0); }
        });
    }

    // Remove GEPs that have non-ptr uses.
    for val_id in &non_ptr_uses {
        gep_map.remove(val_id);
    }

    // Also remove GEPs that are unused (use_count == 0).
    gep_map.retain(|val_id, _| {
        (*val_id as usize) < use_counts.len() && use_counts[*val_id as usize] > 0
    });

    gep_map
}

/// Build a map from Value IDs to global symbol names (with optional offsets).
/// Maps values produced by `GlobalAddr { name }` to `"name"`, and values
/// produced by `GEP(GlobalAddr { name }, const_offset)` to `"name+offset"`.
/// Used to emit direct symbol(%rip) references for segment-overridden loads/stores.
/// TLS symbols are excluded because they require special access patterns
/// (%fs:/@TPOFF on x86-64, %gs:/@NTPOFF on i686, etc.) and must not be
/// folded into plain RIP-relative accesses.
fn build_global_addr_map(func: &IrFunction, tls_symbols: &FxHashSet<Rc<str>>) -> FxHashMap<u32, String> {
    let mut map: FxHashMap<u32, String> = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::GlobalAddr { dest, name } => {
                    // Skip TLS symbols - they must go through emit_tls_global_addr
                    if !tls_symbols.contains(&**name) {
                        map.insert(dest.0, name.to_string());
                    }
                }
                Instruction::GetElementPtr { dest, base, offset: Operand::Const(c), .. } => {
                    if let Some(base_name) = map.get(&base.0) {
                        let offset_val = match c.to_i64() {
                            Some(v) => v,
                            None => continue,
                        };
                        let sym = if offset_val == 0 {
                            base_name.clone()
                        } else if offset_val > 0 {
                            format!("{}+{}", base_name, offset_val)
                        } else {
                            format!("{}{}", base_name, offset_val)
                        };
                        map.insert(dest.0, sym);
                    }
                }
                _ => {}
            }
        }
    }
    map
}

/// Build a set of GlobalAddr value IDs that are "dead" after the fold optimization.
/// A GlobalAddr is dead when ALL of its uses are as `ptr` in Load/Store instructions
/// that will be folded into direct `symbol(%rip)` accesses. In that case, the
/// `lea symbol(%rip), %rax` instruction for the GlobalAddr is unnecessary.
fn build_foldable_global_addr_set(
    func: &IrFunction,
    global_addr_map: &FxHashMap<u32, String>,
) -> FxHashSet<u32> {
    // Collect all GlobalAddr dest value IDs
    let mut global_addr_ids: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::GlobalAddr { dest, .. } = inst {
                global_addr_ids.insert(dest.0);
            }
        }
    }
    if global_addr_ids.is_empty() {
        return FxHashSet::default();
    }

    // Track which GlobalAddr values have non-foldable uses.
    // A use is "foldable" if it's the `ptr` of a Load/Store AND the ptr is in
    // global_addr_map AND the type is foldable (not wide/F128).
    let mut has_non_foldable_use: FxHashSet<u32> = FxHashSet::default();

    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::Load { ptr, ty, seg_override, .. } => {
                    // The ptr use is foldable if it's in global_addr_map and type is supported
                    let is_foldable = global_addr_ids.contains(&ptr.0)
                        && global_addr_map.contains_key(&ptr.0)
                        && !is_wide_int_type(*ty)
                        && *ty != IrType::F128
                        && *seg_override == AddressSpace::Default;
                    if !is_foldable && global_addr_ids.contains(&ptr.0) {
                        has_non_foldable_use.insert(ptr.0);
                    }
                }
                Instruction::Store { val, ptr, ty, seg_override } => {
                    let is_ptr_foldable = global_addr_ids.contains(&ptr.0)
                        && global_addr_map.contains_key(&ptr.0)
                        && !is_wide_int_type(*ty)
                        && *ty != IrType::F128
                        && *seg_override == AddressSpace::Default;
                    if !is_ptr_foldable && global_addr_ids.contains(&ptr.0) {
                        has_non_foldable_use.insert(ptr.0);
                    }
                    // If Store's val references a GlobalAddr, that's a non-foldable use
                    if let Operand::Value(v) = val {
                        if global_addr_ids.contains(&v.0) {
                            has_non_foldable_use.insert(v.0);
                        }
                    }
                }
                // Any other instruction using a GlobalAddr value means it's not dead
                _ => {
                    for v in inst.used_values() {
                        if global_addr_ids.contains(&v) {
                            has_non_foldable_use.insert(v);
                        }
                    }
                }
            }
        }
        // Check terminator uses too
        for v in block.terminator.used_values() {
            if global_addr_ids.contains(&v) {
                has_non_foldable_use.insert(v);
            }
        }
    }

    // Return GlobalAddr values that have NO non-foldable uses
    global_addr_ids.difference(&has_non_foldable_use).copied().collect()
}

/// Build a set of GlobalAddr value IDs that are used as Load/Store pointers.
/// In kernel code model, GlobalAddr values used only as integer values
/// (e.g., `(unsigned long)_text`) need absolute addressing (R_X86_64_32S)
/// to produce the linked virtual address. But GlobalAddr values used as
/// Load/Store pointers need RIP-relative addressing so they work at any
/// physical/virtual address during early boot.
fn build_global_addr_ptr_set(func: &IrFunction) -> FxHashSet<u32> {
    // First collect all GlobalAddr dest values
    let mut global_addrs: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::GlobalAddr { dest, .. } = inst {
                global_addrs.insert(dest.0);
            }
        }
    }
    // Now find which ones (or values derived from them) are used as memory ptrs.
    // Track derivation through Copy, Cast, GEP, Phi, and Select so that a
    // GlobalAddr flowing through intermediate values to a Load/Store/Atomic
    // ptr is still caught.
    let mut ptr_set: FxHashSet<u32> = FxHashSet::default();
    let mut derived_from: FxHashMap<u32, u32> = FxHashMap::default(); // derived_dest -> original GlobalAddr

    // Helper: if `id` is a GlobalAddr or derived from one, mark it as pointer use
    let mark_val = |id: u32, global_addrs: &FxHashSet<u32>, derived_from: &FxHashMap<u32, u32>, ptr_set: &mut FxHashSet<u32>| {
        if global_addrs.contains(&id) {
            ptr_set.insert(id);
        } else if let Some(&orig) = derived_from.get(&id) {
            ptr_set.insert(orig);
        }
    };
    // Helper: same but for Operand (skips constants)
    let mark_op = |op: &Operand, global_addrs: &FxHashSet<u32>, derived_from: &FxHashMap<u32, u32>, ptr_set: &mut FxHashSet<u32>| {
        if let Operand::Value(v) = op {
            if global_addrs.contains(&v.0) {
                ptr_set.insert(v.0);
            } else if let Some(&orig) = derived_from.get(&v.0) {
                ptr_set.insert(orig);
            }
        }
    };
    // Helper: if src_id is a GlobalAddr or derived from one, record dest_id as derived
    let track_val = |dest_id: u32, src_id: u32, global_addrs: &FxHashSet<u32>, derived_from: &mut FxHashMap<u32, u32>| {
        if global_addrs.contains(&src_id) {
            derived_from.insert(dest_id, src_id);
        } else if let Some(&orig) = derived_from.get(&src_id) {
            derived_from.insert(dest_id, orig);
        }
    };
    // Helper: same but for Operand
    let track_op = |dest_id: u32, op: &Operand, global_addrs: &FxHashSet<u32>, derived_from: &mut FxHashMap<u32, u32>| {
        if let Operand::Value(v) = op {
            if global_addrs.contains(&v.0) {
                derived_from.insert(dest_id, v.0);
            } else if let Some(&orig) = derived_from.get(&v.0) {
                derived_from.insert(dest_id, orig);
            }
        }
    };

    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                // Track derivation: these instructions produce a value that may
                // carry a GlobalAddr through to a later pointer use.
                Instruction::GetElementPtr { dest, base, .. } => {
                    track_val(dest.0, base.0, &global_addrs, &mut derived_from);
                }
                Instruction::Copy { dest, src } => {
                    track_op(dest.0, src, &global_addrs, &mut derived_from);
                }
                Instruction::Cast { dest, src, .. } => {
                    track_op(dest.0, src, &global_addrs, &mut derived_from);
                }
                Instruction::Phi { dest, incoming, .. } => {
                    for (op, _) in incoming {
                        if let Operand::Value(v) = op {
                            if global_addrs.contains(&v.0) || derived_from.contains_key(&v.0) {
                                track_val(dest.0, v.0, &global_addrs, &mut derived_from);
                                break;
                            }
                        }
                    }
                }
                Instruction::Select { dest, true_val, false_val, .. } => {
                    // If either branch carries a GlobalAddr, track the result
                    track_op(dest.0, true_val, &global_addrs, &mut derived_from);
                    if !derived_from.contains_key(&dest.0) {
                        track_op(dest.0, false_val, &global_addrs, &mut derived_from);
                    }
                }
                // Mark pointer uses: Load, Store, Memcpy, and atomic operations
                Instruction::Load { ptr, .. } => {
                    mark_val(ptr.0, &global_addrs, &derived_from, &mut ptr_set);
                }
                Instruction::Store { ptr, .. } => {
                    mark_val(ptr.0, &global_addrs, &derived_from, &mut ptr_set);
                }
                Instruction::Memcpy { dest, src, .. } => {
                    mark_val(dest.0, &global_addrs, &derived_from, &mut ptr_set);
                    mark_val(src.0, &global_addrs, &derived_from, &mut ptr_set);
                }
                Instruction::AtomicLoad { ptr, .. } => {
                    mark_op(ptr, &global_addrs, &derived_from, &mut ptr_set);
                }
                Instruction::AtomicStore { ptr, .. } => {
                    mark_op(ptr, &global_addrs, &derived_from, &mut ptr_set);
                }
                Instruction::AtomicRmw { ptr, .. } => {
                    mark_op(ptr, &global_addrs, &derived_from, &mut ptr_set);
                }
                Instruction::AtomicCmpxchg { ptr, .. } => {
                    mark_op(ptr, &global_addrs, &derived_from, &mut ptr_set);
                }
                // Conservatively mark GlobalAddr passed to function calls as pointer use,
                // since the callee may dereference it
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    for arg in &info.args {
                        mark_op(arg, &global_addrs, &derived_from, &mut ptr_set);
                    }
                }
                _ => {}
            }
        }
    }
    ptr_set
}

/// Returns the number of times each IR Value is used as an operand in
/// instructions or terminators. Indexed by Value ID; used to identify
/// single-use values eligible for compare-branch fusion.
fn count_value_uses(func: &IrFunction) -> Vec<u32> {
    // Find the max value ID to size the vector.
    let mut max_id: u32 = 0;
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                max_id = max_id.max(dest.0);
            }
        }
    }
    let mut counts = vec![0u32; max_id as usize + 1];

    // Helper: increment use count for a value ID, bounds-checked.
    let mut count_id = |id: u32| {
        if (id as usize) < counts.len() {
            counts[id as usize] += 1;
        }
    };

    for block in &func.blocks {
        for inst in &block.instructions {
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op { count_id(v.0); }
            });
            for_each_value_use_in_instruction(inst, |v| count_id(v.0));
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op { count_id(v.0); }
        });
    }
    counts
}

/// Detect if a block's last instruction is a Cmp whose result is only used
/// by the block's CondBranch terminator. Returns the index of the Cmp if
/// fusion is possible, None otherwise.
fn detect_cmp_branch_fusion(block: &BasicBlock, use_counts: &[u32]) -> Option<usize> {
    // Terminator must be a CondBranch
    let (cond, _, _) = match &block.terminator {
        Terminator::CondBranch { cond, true_label, false_label } => (cond, true_label, false_label),
        _ => return None,
    };

    // The condition must be a Value (not a constant)
    let cond_val = match cond {
        Operand::Value(v) => v,
        _ => return None,
    };

    // Find the last instruction that is a Cmp producing this value
    let last_idx = block.instructions.len().checked_sub(1)?;
    let last_inst = &block.instructions[last_idx];

    let (dest, _op, _lhs, _rhs, ty) = match last_inst {
        Instruction::Cmp { dest, op, lhs, rhs, ty } => (dest, op, lhs, rhs, ty),
        _ => return None,
    };

    // The Cmp dest must be the same as the CondBranch cond
    if dest.0 != cond_val.0 {
        return None;
    }

    // Don't fuse wide-int or float comparisons (they have special codegen paths).
    // On 32-bit targets, also exclude I64/U64: the fused compare-and-branch
    // uses 32-bit cmpl which only tests the low half of a 64-bit value.
    if is_wide_int_type(*ty) || ty.is_float() {
        return None;
    }
    if crate::common::types::target_is_32bit() && matches!(ty, IrType::I64 | IrType::U64) {
        return None;
    }

    // The Cmp result must be used exactly once (by the CondBranch terminator)
    if (cond_val.0 as usize) < use_counts.len() && use_counts[cond_val.0 as usize] == 1 {
        Some(last_idx)
    } else {
        None
    }
}

/// Generate assembly for a module using the given architecture's codegen.
/// Generate assembly for an IR module with debug info support.
/// Sets `debug_info` on the codegen state before proceeding.
pub fn generate_module_with_debug(
    cg: &mut dyn ArchCodegen,
    module: &IrModule,
    debug_info: bool,
    source_mgr: Option<&crate::common::source::SourceManager>,
) -> String {
    cg.state().debug_info = debug_info;
    generate_module(cg, module, source_mgr)
}

pub fn generate_module(cg: &mut dyn ArchCodegen, module: &IrModule, source_mgr: Option<&crate::common::source::SourceManager>) -> String {
    pre_size_output_buffer(cg, module);
    collect_symbol_sets(cg, module);
    let file_table = build_and_emit_dwarf_file_table(cg, module, source_mgr);

    let ptr_dir = cg.ptr_directive();
    common::emit_data_sections(&mut cg.state().out, module, ptr_dir);

    // Emit top-level asm("...") directives verbatim (e.g., musl's _start definition).
    // Switch to .text first so that labels/code in the asm land in the correct section.
    if !module.toplevel_asm.is_empty() {
        cg.state().emit(".text");
        for asm_str in &module.toplevel_asm {
            cg.state().emit(asm_str);
        }
    }

    let referenced_symbols = collect_referenced_symbols(module);
    emit_extern_visibility_directives(cg, module, &referenced_symbols);
    emit_functions_and_sections(cg, module, source_mgr, &file_table);
    emit_aliases(cg, module);
    emit_symver_directives(cg, module);
    emit_symbol_attrs(cg, module, &referenced_symbols);
    emit_init_fini_arrays(cg, module, ptr_dir);

    // Emit architecture-specific runtime helper stubs (e.g., i686 __divdi3)
    cg.emit_runtime_stubs();

    // Emit .note.GNU-stack section to indicate non-executable stack
    cg.state().emit("");
    cg.state().emit(".section .note.GNU-stack,\"\",@progbits");

    std::mem::take(&mut cg.state().out.buf)
}

/// Pre-size the output buffer based on total IR instruction count to avoid
/// repeated reallocations. Each IR instruction typically generates ~40 bytes
/// of assembly text.
fn pre_size_output_buffer(cg: &mut dyn ArchCodegen, module: &IrModule) {
    let total_insts: usize = module.functions.iter()
        .map(|f| f.blocks.iter().map(|b| b.instructions.len()).sum::<usize>())
        .sum();
    let estimated_bytes = (total_insts * 40).clamp(256 * 1024, 64 * 1024 * 1024);
    let state = cg.state();
    if state.out.buf.capacity() < estimated_bytes {
        state.out.buf.reserve(estimated_bytes - state.out.buf.capacity());
    }
}

/// Build the sets of locally-defined, thread-local, and weak extern symbols.
/// Local symbols (static or hidden/internal/protected visibility) don't need
/// GOT/PLT indirection in PIC mode. TLS symbols need TLS access patterns.
/// Weak extern symbols need GOT indirection on AArch64.
fn collect_symbol_sets(cg: &mut dyn ArchCodegen, module: &IrModule) {
    let state = cg.state();
    for func in &module.functions {
        if func.is_static || matches!(func.visibility.as_deref(), Some("hidden" | "internal" | "protected")) {
            state.local_symbols.insert(func.name.clone());
        }
    }
    for global in &module.globals {
        if global.is_static || matches!(global.visibility.as_deref(), Some("hidden" | "internal" | "protected")) {
            state.local_symbols.insert(global.name.clone());
        }
        if global.is_thread_local {
            state.tls_symbols.insert(global.name.clone());
        }
        if global.is_weak && global.is_extern {
            state.weak_extern_symbols.insert(global.name.clone());
        }
    }
    for (name, is_weak, visibility) in &module.symbol_attrs {
        if matches!(visibility.as_deref(), Some("hidden" | "internal" | "protected")) {
            state.local_symbols.insert(name.clone());
        }
        if *is_weak {
            state.weak_extern_symbols.insert(name.clone());
        }
    }
    for (label, _) in &module.string_literals {
        state.local_symbols.insert(Rc::from(label.as_str()));
    }
    for (label, _) in &module.wide_string_literals {
        state.local_symbols.insert(Rc::from(label.as_str()));
    }
}

/// Build the DWARF file table and emit .file directives when debug info is enabled.
/// Scans all spans in the module, resolves filenames via SourceManager,
/// and assigns each unique filename a DWARF file number (1-based).
fn build_and_emit_dwarf_file_table(
    cg: &mut dyn ArchCodegen,
    module: &IrModule,
    source_mgr: Option<&crate::common::source::SourceManager>,
) -> FxHashMap<String, u32> {
    if !cg.state_ref().debug_info {
        return FxHashMap::default();
    }
    let sm = match source_mgr {
        Some(sm) => sm,
        None => return FxHashMap::default(),
    };

    let mut table: FxHashMap<String, u32> = FxHashMap::default();
    let mut next_id: u32 = 1;
    for func in &module.functions {
        if func.is_declaration { continue; }
        for block in &func.blocks {
            for span in &block.source_spans {
                if span.start == 0 && span.end == 0 { continue; }
                let loc = sm.resolve_span(*span);
                if let std::collections::hash_map::Entry::Vacant(e) = table.entry(loc.file) {
                    e.insert(next_id);
                    next_id += 1;
                }
            }
        }
    }

    if !table.is_empty() {
        let mut entries: Vec<(&String, &u32)> = table.iter().collect();
        entries.sort_by_key(|(_name, id)| *id);
        for (name, id) in entries {
            cg.state().emit_fmt(format_args!(".file {} \"{}\"", id, name));
        }
    }
    table
}

/// Collect the set of symbols actually referenced in this translation unit.
/// We only emit .weak/.hidden directives for referenced symbols, matching GCC behavior.
fn collect_referenced_symbols(module: &IrModule) -> FxHashSet<Rc<str>> {
    let mut refs = FxHashSet::default();

    // Symbols referenced in function bodies
    for func in &module.functions {
        if func.is_declaration { continue; }
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::Call { func: callee, .. } => {
                        refs.insert(callee.clone());
                    }
                    Instruction::GlobalAddr { name, .. } => {
                        refs.insert(name.clone());
                    }
                    Instruction::InlineAsm { input_symbols, .. } => {
                        for s in input_symbols.iter().flatten() {
                            let base = s.split('+').next().unwrap_or(s);
                            refs.insert(Rc::from(base));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Symbols referenced in global initializers
    for global in &module.globals {
        fn collect_global_refs(init: &GlobalInit, refs: &mut FxHashSet<Rc<str>>) {
            match init {
                GlobalInit::GlobalAddr(name) | GlobalInit::GlobalAddrOffset(name, _) => {
                    refs.insert(name.clone());
                }
                GlobalInit::GlobalLabelDiff(a, b, _) => {
                    refs.insert(a.clone());
                    refs.insert(b.clone());
                }
                GlobalInit::Compound(inits) => {
                    for sub in inits {
                        collect_global_refs(sub, refs);
                    }
                }
                _ => {}
            }
        }
        collect_global_refs(&global.init, &mut refs);
    }

    // Symbols referenced in toplevel asm (conservative substring match)
    for asm_str in &module.toplevel_asm {
        for (sym_name, _, _) in &module.symbol_attrs {
            if asm_str.contains(&**sym_name) {
                refs.insert(sym_name.clone());
            }
        }
    }

    // Defined functions and globals are always considered referenced
    for func in &module.functions {
        if !func.is_declaration {
            refs.insert(func.name.clone());
        }
    }
    for global in &module.globals {
        if !global.is_extern {
            refs.insert(global.name.clone());
        }
    }
    refs
}

/// Emit visibility directives for declaration-only (extern) functions with
/// non-default visibility, but only if they are actually referenced.
fn emit_extern_visibility_directives(cg: &mut dyn ArchCodegen, module: &IrModule, referenced_symbols: &FxHashSet<Rc<str>>) {
    for func in &module.functions {
        if func.is_declaration && referenced_symbols.contains(&*func.name) {
            cg.state().emit_visibility(&func.name, &func.visibility);
        }
    }
}

/// Emit text section, handle custom sections, and generate code for each function.
/// When `-ffunction-sections` is enabled, each function without a custom section
/// attribute gets its own `.text.funcname` section, enabling `--gc-sections` to
/// discard unreferenced functions at link time.
fn emit_functions_and_sections(
    cg: &mut dyn ArchCodegen,
    module: &IrModule,
    source_mgr: Option<&crate::common::source::SourceManager>,
    file_table: &FxHashMap<String, u32>,
) {
    let function_sections = cg.state().function_sections;
    if !function_sections {
        cg.state().emit(".section .text");
    }
    let mut in_custom_section = false;
    for func in &module.functions {
        if !func.is_declaration {
            if let Some(ref sect) = func.section {
                cg.state().emit_fmt(format_args!(".section {},\"ax\",@progbits", sect));
                cg.state().current_text_section = sect.clone();
                in_custom_section = true;
            } else if function_sections {
                // -ffunction-sections: each function gets its own section
                let sect_name = format!(".text.{}", func.name);
                cg.state().emit_fmt(format_args!(".section {},\"ax\",@progbits", sect_name));
                cg.state().current_text_section = sect_name;
                in_custom_section = false;
            } else if in_custom_section {
                cg.state().emit(".section .text");
                cg.state().current_text_section = ".text".to_string();
                in_custom_section = false;
            } else {
                cg.state().current_text_section = ".text".to_string();
            }
            generate_function(cg, func, source_mgr, file_table);
        }
    }
}

/// Emit symbol aliases from __attribute__((alias("target"))).
fn emit_aliases(cg: &mut dyn ArchCodegen, module: &IrModule) {
    for (alias_name, target_name, is_weak) in &module.aliases {
        cg.state().emit("");
        if *is_weak {
            cg.state().emit_fmt(format_args!(".weak {}", alias_name));
        } else {
            cg.state().emit_fmt(format_args!(".globl {}", alias_name));
        }
        cg.state().emit_fmt(format_args!(".set {},{}", alias_name, target_name));
    }
}

/// Emit .symver directives from __attribute__((symver("name@@VERSION"))).
fn emit_symver_directives(cg: &mut dyn ArchCodegen, module: &IrModule) {
    for (func_name, symver_str) in &module.symver_directives {
        cg.state().emit_fmt(format_args!(".symver {},{}", func_name, symver_str));
    }
}

/// Emit .weak/.hidden directives for declaration symbols that are referenced.
fn emit_symbol_attrs(cg: &mut dyn ArchCodegen, module: &IrModule, referenced_symbols: &FxHashSet<Rc<str>>) {
    for (name, is_weak, visibility) in &module.symbol_attrs {
        if !referenced_symbols.contains(&**name) {
            continue;
        }
        if *is_weak {
            cg.state().emit_fmt(format_args!(".weak {}", name));
        }
        cg.state().emit_visibility(name, visibility);
    }
}

/// Emit .init_array and .fini_array sections for constructor/destructor functions.
fn emit_init_fini_arrays(cg: &mut dyn ArchCodegen, module: &IrModule, ptr_dir: super::common::PtrDirective) {
    let align = crate::common::types::target_ptr_size();
    for ctor in &module.constructors {
        cg.state().emit("");
        cg.state().emit(".section .init_array,\"aw\",@init_array");
        cg.state().emit_fmt(format_args!(".align {}", ptr_dir.align_arg(align)));
        cg.state().emit_fmt(format_args!("{} {}", ptr_dir.as_str(), ctor));
    }
    for dtor in &module.destructors {
        cg.state().emit("");
        cg.state().emit(".section .fini_array,\"aw\",@fini_array");
        cg.state().emit_fmt(format_args!(".align {}", ptr_dir.align_arg(align)));
        cg.state().emit_fmt(format_args!("{} {}", ptr_dir.as_str(), dtor));
    }
}

/// Generate code for a single function.
fn generate_function(cg: &mut dyn ArchCodegen, func: &IrFunction, source_mgr: Option<&SourceManager>, file_table: &FxHashMap<String, u32>) {
    cg.state().reset_for_function();

    let type_dir = cg.function_type_directive();
    cg.state().emit_linkage(&func.name, func.is_static, func.is_weak);
    cg.state().emit_visibility(&func.name, &func.visibility);

    // Emit patchable function entry NOP padding (-fpatchable-function-entry=N,M).
    // This is used by the Linux kernel for ftrace and static call patching.
    // Format: M NOPs before the entry point, (N-M) NOPs after, plus a
    // __patchable_function_entries section pointing to the NOP area.
    //
    // Skip patchable entries for inline functions: our compiler emits all static
    // inline functions from headers as separate definitions (since we don't inline
    // them yet). Emitting __patchable_function_entries for each of these would create
    // thousands of entries per file (~1400 instead of ~5), overwhelming the kernel's
    // ftrace initialization and causing boot hangs. GCC avoids this by inlining
    // static inline functions so they never get their own patchable entries.
    let emit_patchable = !func.is_inline;
    if emit_patchable {
        if let Some((total, before)) = cg.state().patchable_function_entry {
            if total > 0 {
                let pfe_id = cg.state().next_label_id();
                let pfe_label = format!(".LPFE{}", pfe_id);

                // Emit __patchable_function_entries section with a pointer to the NOP area
                cg.state().emit_fmt(format_args!(
                    ".section __patchable_function_entries,\"awo\",@progbits,{}",
                    pfe_label
                ));
                let pfe_align = crate::common::types::target_ptr_size();
                let pfe_dir = cg.ptr_directive();
                cg.state().emit_fmt(format_args!(".align {}", pfe_align));
                cg.state().emit_fmt(format_args!("{} {}", pfe_dir.as_str(), pfe_label));

                // Switch back to the function's section (custom or .text)
                if let Some(ref sect) = func.section {
                    cg.state().emit_fmt(format_args!(".section {},\"ax\",@progbits", sect));
                } else {
                    cg.state().emit(".text");
                }

                // Emit the LPFE label and M NOPs before the function entry point
                cg.state().emit_fmt(format_args!("{}:", pfe_label));
                for _ in 0..before {
                    cg.state().emit("nop");
                }
            }
        }
    }

    cg.state().emit_fmt(format_args!(".type {}, {}", func.name, type_dir));
    cg.state().emit_fmt(format_args!("{}:", func.name));
    let emit_cfi = cg.state().emit_cfi;
    if emit_cfi {
        cg.state().emit(".cfi_startproc");
    }

    // Emit (N-M) NOPs after the function entry point for patchable function entry
    if emit_patchable {
        if let Some((total, before)) = cg.state().patchable_function_entry {
            let after = total.saturating_sub(before);
            for _ in 0..after {
                cg.state().emit("nop");
            }
        }
    }

    // Naked functions: emit only inline asm blocks, no prologue/epilogue/params.
    if func.is_naked {
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::InlineAsm { template, .. } = inst {
                    cg.emit_raw_inline_asm(template);
                }
            }
        }
        if emit_cfi {
            cg.state().emit(".cfi_endproc");
        }
        cg.state().emit_fmt(format_args!(".size {}, .-{}", func.name, func.name));
        cg.state().emit("");
        return;
    }

    // Pre-scan for DynAlloca/StackRestore: if present, the epilogue must restore SP from
    // the frame pointer instead of adding back the compile-time frame size.
    let has_dyn_alloca = func.blocks.iter().any(|block| {
        block.instructions.iter().any(|inst| matches!(inst, Instruction::DynAlloca { .. } | Instruction::StackRestore { .. }))
    });
    cg.state().has_dyn_alloca = has_dyn_alloca;
    cg.state().uses_sret = func.uses_sret;

    // Calculate stack space and emit prologue
    let raw_space = cg.calculate_stack_space(func);
    let frame_size = cg.aligned_frame_size(raw_space);
    cg.emit_prologue(func, frame_size);

    // Store parameters
    cg.emit_store_params(func);

    // Generate basic blocks
    let entry_label = func.blocks.first().map(|b| b.label);

    // Pre-scan: count uses of each Value across the entire function to identify
    // single-use Cmp results eligible for compare-branch fusion.
    let value_use_counts = count_value_uses(func);

    // Pre-scan: identify GEPs with constant offsets that can be folded into
    // Load/Store addressing modes, eliminating the GEP instruction entirely.
    let gep_fold_map = build_gep_fold_map(func, &value_use_counts);

    // Pre-scan: map Value IDs to global symbol names (with offsets from GEP).
    // Used to emit direct symbol(%rip) references for segment-overridden loads/stores.
    let global_addr_map = build_global_addr_map(func, &cg.state_ref().tls_symbols);

    // Pre-scan: identify GlobalAddr values used as Load/Store pointers.
    // In kernel code model, non-pointer GlobalAddr values use absolute addressing
    // (R_X86_64_32S) for the linked virtual address, while pointer GlobalAddr
    // values use RIP-relative addressing for position-independent memory access.
    let global_addr_ptr_set = if cg.state_ref().code_model_kernel {
        build_global_addr_ptr_set(func)
    } else {
        FxHashSet::default()
    };

    // Pre-scan: identify GlobalAddr values that can be skipped because ALL of
    // their uses are Load/Store pointers that will be folded into direct
    // `symbol(%rip)` accesses by the generate_load/generate_store fold.
    let dead_global_addrs = if cg.supports_global_addr_fold() {
        build_foldable_global_addr_set(func, &global_addr_map)
    } else {
        FxHashSet::default()
    };

    // Debug info state: track last emitted file/line to suppress redundant .loc directives.
    let emit_debug = cg.state_ref().debug_info && source_mgr.is_some() && !file_table.is_empty();
    let mut last_debug_file: u32 = 0;
    let mut last_debug_line: u32 = 0;

    for block in &func.blocks {
        if Some(block.label) != entry_label {
            // Invalidate register cache at block boundaries: a value in a register
            // from the previous block's fall-through is not guaranteed to be valid
            // if control arrives from a different predecessor.
            cg.state().reg_cache.invalidate_all();
            cg.state().out.emit_block_label(block.label.0);
        }

        // Check for compare-branch fusion opportunity:
        // If the last instruction is a Cmp whose result is only used by the
        // CondBranch terminator, emit a fused compare-and-conditional-jump
        // instead of materializing the boolean result to a register/stack slot.
        let fuse_idx = detect_cmp_branch_fusion(block, &value_use_counts);

        for (idx, inst) in block.instructions.iter().enumerate() {
            if Some(idx) == fuse_idx {
                // Skip this Cmp -- it will be emitted fused with the terminator
                continue;
            }
            // Skip GEP instructions whose offset has been folded into Load/Store.
            // Safe to skip when:
            // 1. Base is an alloca (Direct or OverAligned): alloca slots are stable
            //    and never reused by liveness packing.
            // 2. Base has a register assignment: the liveness analysis has been
            //    extended to keep the base alive through all Load/Store uses of
            //    this GEP result (see extend_gep_base_liveness in liveness.rs),
            //    so the register holds the correct value at the use points.
            if let Instruction::GetElementPtr { dest, base, .. } = inst {
                if gep_fold_map.contains_key(&dest.0) &&
                   (cg.state_ref().is_alloca(base.0) || cg.get_phys_reg_for_value(base.0).is_some()) {
                    continue;
                }
            }

            // Emit .loc directive if source location changed.
            if emit_debug {
                if let Some(span) = block.source_spans.get(idx) {
                    emit_loc_directive(cg, span, source_mgr.expect("debug mode requires source manager"), file_table,
                                       &mut last_debug_file, &mut last_debug_line);
                }
            }

            generate_instruction(cg, inst, &gep_fold_map, &global_addr_map, &global_addr_ptr_set, &dead_global_addrs);
        }

        if let Some(fi) = fuse_idx {
            // Emit fused compare-and-branch: cmp + jCC directly
            if let Instruction::Cmp { dest: _, op, lhs, rhs, ty } = &block.instructions[fi] {
                if let Terminator::CondBranch { cond: _, true_label, false_label } = &block.terminator {
                    cg.emit_fused_cmp_branch_blocks(*op, lhs, rhs, *ty, *true_label, *false_label);
                }
            }
        } else {
            generate_terminator(cg, &block.terminator, frame_size);
        }
    }

    if emit_cfi {
        cg.state().emit(".cfi_endproc");
    }
    cg.state().emit_fmt(format_args!(".size {}, .-{}", func.name, func.name));
    cg.state().emit("");
}

/// Emit a `.loc` directive if the source location for this instruction differs
/// from the previously emitted location. Suppresses redundant directives and
/// skips dummy spans (start==0, end==0).
fn emit_loc_directive(
    cg: &mut dyn ArchCodegen,
    span: &Span,
    source_mgr: &SourceManager,
    file_table: &FxHashMap<String, u32>,
    last_file: &mut u32,
    last_line: &mut u32,
) {
    // Skip dummy spans
    if span.start == 0 && span.end == 0 {
        return;
    }
    let loc = source_mgr.resolve_span(*span);
    if let Some(&dwarf_file_id) = file_table.get(&loc.file) {
        if dwarf_file_id != *last_file || loc.line != *last_line {
            cg.state().emit_fmt(format_args!(".loc {} {} {}", dwarf_file_id, loc.line, loc.column));
            *last_file = dwarf_file_id;
            *last_line = loc.line;
        }
    }
}

/// Dispatch a single IR instruction to the appropriate arch method.
///
/// Register cache management strategy:
/// The cache tracks which IR value is currently in the accumulator register
/// (rax on x86, x0 on ARM, t0 on RISC-V).
///
/// Many instructions follow the pattern: load operand(s) → compute → store_result(dest),
/// which means the accumulator holds dest's value when the instruction completes.
/// For these "acc-preserving" instructions, we keep the cache valid so the next
/// instruction can skip reloading the result.
///
/// Instructions that clobber the accumulator unpredictably (calls, stores, atomics,
/// inline asm, va_arg, memcpy, etc.) invalidate the cache after execution.
fn generate_instruction(cg: &mut dyn ArchCodegen, inst: &Instruction, gep_fold_map: &FxHashMap<u32, GepFoldInfo>, global_addr_map: &FxHashMap<u32, String>, global_addr_ptr_set: &FxHashSet<u32>, dead_global_addrs: &FxHashSet<u32>) {
    match inst {
        Instruction::Alloca { .. } => {
            // Space already allocated in prologue; does not touch registers
        }
        Instruction::Copy { dest, src } => {
            generate_copy(cg, dest, src);
        }

        // ── Acc-preserving instructions ──────────────────────────────────
        // These all end with emit_store_result(dest) or store_rax_to(dest),
        // which sets the reg cache correctly. The accumulator holds dest's
        // value after execution, so we do NOT invalidate.

        Instruction::Load { dest, ptr, ty, seg_override } => {
            generate_load(cg, dest, ptr, *ty, *seg_override, gep_fold_map, global_addr_map);
        }
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            cg.emit_binop(dest, *op, lhs, rhs, *ty);
            if is_wide_int_type(*ty) {
                cg.state().reg_cache.invalidate_all();
            }
        }
        Instruction::UnaryOp { dest, op, src, ty } => {
            cg.emit_unaryop(dest, *op, src, *ty);
            if is_wide_int_type(*ty) {
                cg.state().reg_cache.invalidate_all();
            }
        }
        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            cg.emit_cmp(dest, *op, lhs, rhs, *ty);
        }
        Instruction::Cast { dest, src, from_ty, to_ty } => {
            cg.emit_cast(dest, src, *from_ty, *to_ty);
            if is_wide_int_type(*to_ty) || is_wide_int_type(*from_ty) {
                cg.state().reg_cache.invalidate_all();
            }
        }
        Instruction::GetElementPtr { dest, base, offset, .. } => {
            cg.emit_gep(dest, base, offset);
        }
        Instruction::GlobalAddr { dest, name } => {
            // Skip GlobalAddr when all its uses are folded into direct symbol(%rip)
            // loads/stores by generate_load/generate_store. The needs_got check
            // ensures we don't skip when GOT indirection is required.
            // needs_got_for_addr is used because x86-64 needs GOT for external
            // symbol addresses even in non-PIC mode (for PIE compatibility).
            // TLS symbols must never be folded: they need %fs:sym@TPOFF access,
            // not symbol(%rip).
            let is_dead = dead_global_addrs.contains(&dest.0)
                && !cg.state_ref().needs_got_for_addr(name)
                && !cg.state_ref().tls_symbols.contains(&**name);
            if !is_dead {
                if cg.state_ref().tls_symbols.contains(&**name) {
                    cg.emit_tls_global_addr(dest, name);
                } else if cg.state_ref().code_model_kernel && !global_addr_ptr_set.contains(&dest.0) {
                    cg.emit_global_addr_absolute(dest, name);
                } else {
                    cg.emit_global_addr(dest, name);
                }
            }
        }
        Instruction::Select { dest, cond, true_val, false_val, ty } => {
            cg.emit_select(dest, cond, true_val, false_val, *ty);
        }
        Instruction::LabelAddr { dest, label } => {
            cg.emit_label_addr(dest, &label.as_label());
        }

        // ── Cache-invalidating instructions ──────────────────────────────
        // These clobber the accumulator unpredictably or don't produce a
        // simple acc → dest result. Each arm invalidates the reg cache.

        Instruction::Store { val, ptr, ty, seg_override } => {
            generate_store(cg, val, ptr, *ty, *seg_override, gep_fold_map, global_addr_map);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::DynAlloca { dest, size, align } => {
            cg.emit_dyn_alloca(dest, size, *align);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::Call { func, info } => {
            cg.emit_call(&info.args, &info.arg_types, Some(func), None, info.dest, info.return_type, info.is_variadic, info.num_fixed_args, &info.struct_arg_sizes, &info.struct_arg_aligns, &info.struct_arg_classes, &info.struct_arg_riscv_float_classes, info.is_sret, info.is_fastcall, &info.ret_eightbyte_classes);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::CallIndirect { func_ptr, info } => {
            cg.emit_call(&info.args, &info.arg_types, None, Some(func_ptr), info.dest, info.return_type, info.is_variadic, info.num_fixed_args, &info.struct_arg_sizes, &info.struct_arg_aligns, &info.struct_arg_classes, &info.struct_arg_riscv_float_classes, info.is_sret, info.is_fastcall, &info.ret_eightbyte_classes);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::Memcpy { dest, src, size } => {
            cg.emit_memcpy(dest, src, *size);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::VaArg { dest, va_list_ptr, result_ty } => {
            cg.emit_va_arg(dest, va_list_ptr, *result_ty);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::VaStart { va_list_ptr } => {
            cg.emit_va_start(va_list_ptr);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::VaEnd { va_list_ptr } => {
            cg.emit_va_end(va_list_ptr);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::VaCopy { dest_ptr, src_ptr } => {
            cg.emit_va_copy(dest_ptr, src_ptr);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::VaArgStruct { dest_ptr, va_list_ptr, size, ref eightbyte_classes } => {
            cg.emit_va_arg_struct_ex(dest_ptr, va_list_ptr, *size, eightbyte_classes);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::AtomicRmw { dest, op, ptr, val, ty, ordering } => {
            cg.emit_atomic_rmw(dest, *op, ptr, val, *ty, *ordering);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::AtomicCmpxchg { dest, ptr, expected, desired, ty, success_ordering, failure_ordering, returns_bool } => {
            cg.emit_atomic_cmpxchg(dest, ptr, expected, desired, *ty, *success_ordering, *failure_ordering, *returns_bool);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::AtomicLoad { dest, ptr, ty, ordering } => {
            cg.emit_atomic_load(dest, ptr, *ty, *ordering);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::AtomicStore { ptr, val, ty, ordering } => {
            cg.emit_atomic_store(ptr, val, *ty, *ordering);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::Fence { ordering } => {
            cg.emit_fence(*ordering);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::Phi { .. } => { /* resolved before codegen */ }
        Instruction::GetReturnF64Second { dest } => {
            cg.emit_get_return_f64_second(dest);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::SetReturnF64Second { src } => {
            cg.emit_set_return_f64_second(src);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::GetReturnF32Second { dest } => {
            cg.emit_get_return_f32_second(dest);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::SetReturnF32Second { src } => {
            cg.emit_set_return_f32_second(src);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::GetReturnF128Second { dest } => {
            cg.emit_get_return_f128_second(dest);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::SetReturnF128Second { src } => {
            cg.emit_set_return_f128_second(src);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::InlineAsm { template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols, seg_overrides } => {
            cg.emit_inline_asm_with_segs(template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols, seg_overrides);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::Intrinsic { dest, op, dest_ptr, args } => {
            cg.emit_intrinsic(dest, op, dest_ptr, args);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::StackSave { dest } => {
            cg.emit_stack_save(dest);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::StackRestore { ptr } => {
            cg.emit_stack_restore(ptr);
            cg.state().reg_cache.invalidate_all();
        }
        Instruction::ParamRef { dest, param_idx, ty } => {
            cg.emit_param_ref(dest, *param_idx, *ty);
            cg.state().reg_cache.invalidate_all();
        }
    }
}

/// Generate a Copy instruction, handling coalesced slots, i128, and wide values.
fn generate_copy(cg: &mut dyn ArchCodegen, dest: &Value, src: &Operand) {
    // Skip Copy when dest and src share the same stack slot (from copy coalescing).
    if let Operand::Value(src_val) = src {
        let dest_slot = cg.state_ref().get_slot(dest.0);
        let src_slot = cg.state_ref().get_slot(src_val.0);
        if let (Some(ds), Some(ss)) = (dest_slot, src_slot) {
            if ds.0 == ss.0 {
                if cg.state_ref().reg_cache.acc_has(src_val.0, false) {
                    cg.state().reg_cache.set_acc(dest.0, false);
                }
                return;
            }
        }
    }

    let is_i128_copy = match src {
        Operand::Value(v) => cg.state_ref().is_i128_value(v.0),
        Operand::Const(IrConst::I128(_)) => true,
        _ => false,
    };
    if is_i128_copy {
        cg.state().i128_values.insert(dest.0);
        cg.emit_copy_i128(dest, src);
        cg.state().reg_cache.invalidate_all();
        return;
    }

    // Propagate wide value status through Copy chains on 32-bit targets.
    // IrConst::I64 is the universal container for ALL integer constants,
    // so only mark as wide if the value doesn't fit in 32 bits.
    let is_wide = match src {
        Operand::Value(v) => cg.state_ref().is_wide_value(v.0),
        Operand::Const(IrConst::F64(_)) => crate::common::types::target_is_32bit(),
        Operand::Const(IrConst::I64(val)) => {
            crate::common::types::target_is_32bit()
                && (*val < i32::MIN as i64 || *val > u32::MAX as i64)
        }
        _ => false,
    };
    if is_wide {
        cg.state().wide_values.insert(dest.0);
    }
    cg.emit_copy_value(dest, src);
}

/// Generate a Load instruction with segment override, kernel code model,
/// and GEP folding support.
fn generate_load(
    cg: &mut dyn ArchCodegen,
    dest: &Value, ptr: &Value, ty: IrType, seg_override: AddressSpace,
    gep_fold_map: &FxHashMap<u32, GepFoldInfo>,
    global_addr_map: &FxHashMap<u32, String>,
) {
    if seg_override != AddressSpace::Default {
        if let Some(sym) = global_addr_map.get(&ptr.0) {
            cg.emit_seg_load_symbol(dest, sym, ty, seg_override);
        } else {
            cg.emit_seg_load(dest, ptr, ty, seg_override);
        }
        return;
    }
    // Fold GlobalAddr + Load into a direct PC-relative memory access.
    // On x86-64 this emits `movl symbol(%rip), %eax` instead of separate
    // `leaq symbol(%rip), %rax` + `movl (%rax), %eax`.
    // Works for kernel and default code models. Skipped for symbols
    // that require GOT indirection (the pointer comes from the GOT), and
    // for TLS symbols which require %fs:sym@TPOFF access patterns.
    // Uses needs_got_for_addr to block folding for external symbols even
    // in non-PIC mode (x86-64 needs GOTPCREL for PIE compatibility).
    if cg.supports_global_addr_fold() && !is_wide_int_type(ty) && ty != IrType::F128 {
        if let Some(sym) = global_addr_map.get(&ptr.0) {
            if !cg.state_ref().needs_got_for_addr(sym) && !cg.state_ref().tls_symbols.contains(sym.as_str()) {
                cg.emit_global_load_rip_rel(dest, sym, ty);
                return;
            }
        }
    }
    // Fold GEP with constant offset into Load addressing mode.
    if let Some(gep_info) = gep_fold_map.get(&ptr.0) {
        if !is_wide_int_type(ty) &&
           (cg.state_ref().is_alloca(gep_info.base.0) || cg.get_phys_reg_for_value(gep_info.base.0).is_some()) {
            cg.emit_load_with_const_offset(dest, &gep_info.base, gep_info.offset, ty);
            return;
        }
    }
    cg.emit_load(dest, ptr, ty);
    if is_wide_int_type(ty) {
        cg.state().reg_cache.invalidate_all();
    }
}

/// Generate a Store instruction with segment override, kernel code model,
/// and GEP folding support.
fn generate_store(
    cg: &mut dyn ArchCodegen,
    val: &Operand, ptr: &Value, ty: IrType, seg_override: AddressSpace,
    gep_fold_map: &FxHashMap<u32, GepFoldInfo>,
    global_addr_map: &FxHashMap<u32, String>,
) {
    if seg_override != AddressSpace::Default {
        if let Some(sym) = global_addr_map.get(&ptr.0) {
            cg.emit_seg_store_symbol(val, sym, ty, seg_override);
        } else {
            cg.emit_seg_store(val, ptr, ty, seg_override);
        }
        return;
    }
    // Fold GlobalAddr + Store into a direct PC-relative memory access.
    // Skipped for TLS symbols which require %fs:sym@TPOFF access patterns.
    // Uses needs_got_for_addr: same as Load fold above.
    if cg.supports_global_addr_fold() && !is_wide_int_type(ty) && ty != IrType::F128 {
        if let Some(sym) = global_addr_map.get(&ptr.0) {
            if !cg.state_ref().needs_got_for_addr(sym) && !cg.state_ref().tls_symbols.contains(sym.as_str()) {
                cg.emit_global_store_rip_rel(val, sym, ty);
                return;
            }
        }
    }
    // Fold GEP with constant offset into Store addressing mode.
    if let Some(gep_info) = gep_fold_map.get(&ptr.0) {
        if !is_wide_int_type(ty) &&
           (cg.state_ref().is_alloca(gep_info.base.0) || cg.get_phys_reg_for_value(gep_info.base.0).is_some()) {
            cg.emit_store_with_const_offset(val, &gep_info.base, gep_info.offset, ty);
            return;
        }
    }
    cg.emit_store(val, ptr, ty);
}

/// Dispatch a terminator to the appropriate arch method.
fn generate_terminator(cg: &mut dyn ArchCodegen, term: &Terminator, frame_size: i64) {
    match term {
        Terminator::Return(val) => {
            cg.emit_return(val.as_ref(), frame_size);
        }
        Terminator::Branch(label) => {
            cg.emit_branch_to_block(*label);
        }
        Terminator::CondBranch { cond, true_label, false_label } => {
            cg.emit_cond_branch_blocks(cond, *true_label, *false_label);
        }
        Terminator::IndirectBranch { target, .. } => {
            cg.emit_indirect_branch(target);
        }
        Terminator::Switch { val, cases, default, ty } => {
            cg.emit_switch(val, cases, default, *ty);
        }
        Terminator::Unreachable => {
            cg.emit_unreachable();
        }
    }
}

/// Check if an IR type is a 128-bit integer type (I128 or U128).
pub fn is_i128_type(ty: IrType) -> bool {
    matches!(ty, IrType::I128 | IrType::U128)
}

/// Check if a type is "wide" — needs register-pair operations on the current target.
///
/// Only I128/U128 on all targets. On i686, I64/U64 BinOps are handled via
/// the i686-specific `emit_binop`/`emit_cmp`/`emit_unaryop` overrides which
/// route them through register-pair arithmetic. We don't include I64/U64 here
/// because the framework-level effects (disabling GEP folding, fused branches,
/// cache invalidation) would cause excessive overhead on the common case of
/// widened I32 arithmetic.
pub fn is_wide_int_type(ty: IrType) -> bool {
    matches!(ty, IrType::I128 | IrType::U128)
}

// Re-export stack layout functions so existing `crate::backend::generation::X` imports
// continue to work without changes to downstream code.
pub use super::stack_layout::{
    collect_inline_asm_callee_saved,
    collect_inline_asm_callee_saved_with_generic,
    run_regalloc_and_merge_clobbers,
    filter_available_regs,
    calculate_stack_space_common,
    find_param_alloca,
};

