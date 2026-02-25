//! Loop-Invariant Code Motion (LICM) pass.
//!
//! Identifies natural loops in the CFG and hoists loop-invariant instructions
//! to preheader blocks that execute before the loop. An instruction is
//! loop-invariant if all of its operands are either:
//! - Constants
//! - Defined outside the loop
//! - Defined by other loop-invariant instructions
//!
//! This is particularly important for:
//! - Array index computations (i * n) in inner loops
//! - Address calculations that depend on outer loop variables
//! - Casts and extensions of loop-invariant values
//!
//! The pass requires loops to have a single-entry preheader block. Loops with
//! multiple outside predecessors are skipped (a future improvement could create
//! dedicated preheader blocks for these cases).
//!
//! Safety: Pure (side-effect-free) instructions are always hoisted. Loads are
//! hoisted only when we can prove the memory location is not modified inside
//! the loop:
//! - Loads from allocas that are NOT address-taken and have no stores in the loop
//! - Loads from GlobalAddr pointers when the loop has no function calls and no
//!   stores to any GlobalAddr target (since calls and stores to unknown pointers
//!   could potentially modify any global variable)
//!
//! Address-taken allocas (used in GEP, passed to calls, etc.) are never hoisted
//! because stores through derived pointers may not be tracked in `stored_allocas`.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::analysis;
use crate::ir::reexports::{
    Instruction,
    IrFunction,
    Operand,
    Terminator,
    Value,
};
use super::loop_analysis::{self, NaturalLoop};

/// Run LICM using pre-computed CFG analysis (avoids redundant analysis when
/// called from a pipeline that shares analysis across GVN, LICM, IVSR).
pub(crate) fn licm_with_analysis(func: &mut IrFunction, cfg: &analysis::CfgAnalysis) -> usize {
    if cfg.num_blocks < 2 {
        return 0;
    }

    // Find natural loops
    let loops = loop_analysis::find_natural_loops(cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom);
    if loops.is_empty() {
        return 0;
    }

    // Merge natural loops that share the same header block.
    let loops = loop_analysis::merge_loops_by_header(loops);

    // Pre-compute function-level alloca analysis for load hoisting.
    let alloca_info = analyze_allocas(func);

    let mut total_hoisted = 0;

    // Process loops from innermost to outermost (smaller loops first).
    let mut sorted_loops = loops;
    sorted_loops.sort_by_key(|l| l.body.len());

    for natural_loop in &sorted_loops {
        total_hoisted += hoist_loop_invariants(func, natural_loop, &cfg.preds, &alloca_info);
    }

    total_hoisted
}

/// Check if an instruction is safe to hoist (pure, no side effects, not a phi).
/// Division and remainder can trap with SIGFPE on divide-by-zero, so they must
/// not be speculatively hoisted past a guard condition.
fn is_hoistable(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::BinOp { op, .. } if !op.can_trap()
    ) || matches!(
        inst,
        Instruction::UnaryOp { .. }
            | Instruction::Cmp { .. }
            | Instruction::Cast { .. }
            | Instruction::GetElementPtr { .. }
            | Instruction::Copy { .. }
            | Instruction::GlobalAddr { .. }
            | Instruction::Select { .. }
    ) || matches!(inst, Instruction::Intrinsic { op, .. } if op.is_pure())
}

/// Information about allocas in a function, used for load hoisting analysis.
struct AllocaAnalysis {
    /// Set of value IDs that are alloca destinations.
    alloca_values: FxHashSet<u32>,
    /// Set of alloca value IDs that are "address-taken" — used by anything
    /// other than direct Load/Store (e.g., passed to a call, used in GEP
    /// as a non-base, stored as a value, etc.). Loads from address-taken
    /// allocas cannot be safely hoisted past calls.
    address_taken: FxHashSet<u32>,
}

/// Analyze all allocas in a function to determine which are address-taken.
///
/// An alloca is "address-taken" if its value (the pointer) is used by any
/// instruction other than Load (as ptr) or Store (as ptr). This includes:
/// - Passed as a call argument
/// - Stored as a value (the pointer itself is stored somewhere)
/// - Used as an operand in a BinOp, Cast, etc.
/// - Used in a GEP (the resulting pointer could be passed anywhere)
///
/// Allocas that are NOT address-taken can only be accessed via direct
/// Load/Store, so we can reason precisely about which stores modify them.
fn analyze_allocas(func: &IrFunction) -> AllocaAnalysis {
    let mut alloca_values = FxHashSet::default();
    let mut address_taken = FxHashSet::default();

    // Collect all alloca values from every block. After inlining, allocas
    // from callee entry blocks appear in non-entry blocks of the caller.
    // All such allocas must be tracked so that LICM correctly identifies
    // intrinsic instructions that read from loop-modified alloca memory.
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, .. } = inst {
                alloca_values.insert(dest.0);
            }
        }
    }

    if alloca_values.is_empty() {
        return AllocaAnalysis { alloca_values, address_taken };
    }

    // Scan all instructions to find address-taken allocas.
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                // Load from alloca is fine (direct access).
                Instruction::Load { ptr, .. } => {
                    // ptr is used as a load target - this is fine, not address-taken.
                    let _ = ptr;
                }
                // Store TO an alloca is fine. But storing an alloca's value
                // (as the stored data) means its address escapes.
                Instruction::Store { val, ptr, .. } => {
                    let _ = ptr; // Storing to alloca is fine.
                    // If the stored VALUE is an alloca pointer, it's address-taken.
                    if let Operand::Value(v) = val {
                        if alloca_values.contains(&v.0) {
                            address_taken.insert(v.0);
                        }
                    }
                }
                // Alloca definitions themselves are fine.
                Instruction::Alloca { .. } | Instruction::DynAlloca { .. } => {}
                // All other instructions: any alloca used as an operand is address-taken.
                _ => {
                    for_each_operand_value(inst, |val_id| {
                        if alloca_values.contains(&val_id) {
                            address_taken.insert(val_id);
                        }
                    });
                }
            }
        }

        // Check terminator operands.
        for_each_terminator_value(&block.terminator, |val_id| {
            if alloca_values.contains(&val_id) {
                address_taken.insert(val_id);
            }
        });
    }

    AllocaAnalysis { alloca_values, address_taken }
}

/// Visit each Value ID used as operands by any instruction (for address-taken analysis).
/// Uses a callback to avoid allocating a Vec per call.
#[inline]
fn for_each_operand_value(inst: &Instruction, mut f: impl FnMut(u32)) {
    #[inline]
    fn collect(op: &Operand, f: &mut impl FnMut(u32)) {
        if let Operand::Value(v) = op {
            f(v.0);
        }
    }

    match inst {
        Instruction::BinOp { lhs, rhs, .. } => { collect(lhs, &mut f); collect(rhs, &mut f); }
        Instruction::UnaryOp { src, .. } => collect(src, &mut f),
        Instruction::Cmp { lhs, rhs, .. } => { collect(lhs, &mut f); collect(rhs, &mut f); }
        Instruction::Cast { src, .. } => collect(src, &mut f),
        Instruction::GetElementPtr { base, offset, .. } => {
            f(base.0);
            collect(offset, &mut f);
        }
        Instruction::Copy { src, .. } => collect(src, &mut f),
        Instruction::Call { info, .. } => { for a in &info.args { collect(a, &mut f); } }
        Instruction::CallIndirect { func_ptr, info } => {
            collect(func_ptr, &mut f);
            for a in &info.args { collect(a, &mut f); }
        }
        Instruction::Memcpy { dest, src, .. } => { f(dest.0); f(src.0); }
        Instruction::VaStart { va_list_ptr, .. } => f(va_list_ptr.0),
        Instruction::VaEnd { va_list_ptr } => f(va_list_ptr.0),
        Instruction::VaCopy { dest_ptr, src_ptr } => { f(dest_ptr.0); f(src_ptr.0); }
        Instruction::VaArg { va_list_ptr, .. } => f(va_list_ptr.0),
        Instruction::VaArgStruct { dest_ptr, va_list_ptr, .. } => { f(dest_ptr.0); f(va_list_ptr.0); }
        Instruction::AtomicRmw { ptr, val, .. } => { collect(ptr, &mut f); collect(val, &mut f); }
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => {
            collect(ptr, &mut f); collect(expected, &mut f); collect(desired, &mut f);
        }
        Instruction::AtomicLoad { ptr, .. } => collect(ptr, &mut f),
        Instruction::AtomicStore { ptr, val, .. } => { collect(ptr, &mut f); collect(val, &mut f); }
        Instruction::InlineAsm { outputs, inputs, .. } => {
            // Output pointers are alloca Values that the backend stores asm results into.
            // They must be tracked as address-taken to prevent LICM from hoisting loads
            // of those allocas out of loops containing inline asm.
            for (_, ptr, _) in outputs { f(ptr.0); }
            for (_, op, _) in inputs { collect(op, &mut f); }
        }
        Instruction::Intrinsic { args, .. } => { for a in args { collect(a, &mut f); } }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming { collect(op, &mut f); }
        }
        Instruction::SetReturnF64Second { src } => collect(src, &mut f),
        Instruction::SetReturnF32Second { src } => collect(src, &mut f),
        Instruction::SetReturnF128Second { src } => collect(src, &mut f),
        Instruction::DynAlloca { size, .. } => collect(size, &mut f),
        Instruction::Select { cond, true_val, false_val, .. } => {
            collect(cond, &mut f);
            collect(true_val, &mut f);
            collect(false_val, &mut f);
        }
        // These don't use Value operands (or are already handled above).
        Instruction::Alloca { .. }
        | Instruction::Store { .. }
        | Instruction::Load { .. }
        | Instruction::GlobalAddr { .. }
        | Instruction::LabelAddr { .. }
        | Instruction::GetReturnF64Second { .. }
        | Instruction::GetReturnF32Second { .. }
        | Instruction::GetReturnF128Second { .. }
        | Instruction::Fence { .. }
        | Instruction::StackSave { .. }
        | Instruction::StackRestore { .. }
        | Instruction::ParamRef { .. } => {}
    }
}

/// Visit each Value ID used in a terminator.
#[inline]
fn for_each_terminator_value(term: &Terminator, mut f: impl FnMut(u32)) {
    match term {
        Terminator::Return(Some(Operand::Value(v))) => f(v.0),
        Terminator::CondBranch { cond: Operand::Value(v), .. } => f(v.0),
        Terminator::IndirectBranch { target: Operand::Value(v), .. } => f(v.0),
        Terminator::Switch { val: Operand::Value(v), .. } => f(v.0),
        _ => {}
    }
}

/// Visit each Value ID referenced as operands by a hoistable instruction (not including dest).
/// This is used for the invariance check during hoisting.
/// Uses a callback to avoid allocating a Vec per call in the hot fixpoint loop.
#[inline]
fn for_each_hoistable_operand(inst: &Instruction, mut f: impl FnMut(u32)) {
    #[inline]
    fn collect_op(op: &Operand, f: &mut impl FnMut(u32)) {
        if let Operand::Value(v) = op {
            f(v.0);
        }
    }

    match inst {
        Instruction::BinOp { lhs, rhs, .. } => {
            collect_op(lhs, &mut f);
            collect_op(rhs, &mut f);
        }
        Instruction::UnaryOp { src, .. } => {
            collect_op(src, &mut f);
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            collect_op(lhs, &mut f);
            collect_op(rhs, &mut f);
        }
        Instruction::Cast { src, .. } => {
            collect_op(src, &mut f);
        }
        Instruction::GetElementPtr { base, offset, .. } => {
            f(base.0);
            collect_op(offset, &mut f);
        }
        Instruction::Copy { src, .. } => {
            collect_op(src, &mut f);
        }
        Instruction::GlobalAddr { .. } => {
            // No value operands
        }
        Instruction::Load { ptr, .. } => {
            // The pointer is the operand we need to check for loop-invariance.
            f(ptr.0);
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            collect_op(cond, &mut f);
            collect_op(true_val, &mut f);
            collect_op(false_val, &mut f);
        }
        Instruction::Intrinsic { args, .. } => {
            for a in args { collect_op(a, &mut f); }
        }
        // All other instructions are non-hoistable and should never reach here.
        _ => unreachable!("for_each_hoistable_operand called on non-hoistable instruction")
    }
}

/// Analyze which allocas are stored to within a loop body.
struct LoopMemoryInfo {
    /// Alloca value IDs that have stores targeting them within the loop.
    stored_allocas: FxHashSet<u32>,
    /// Base alloca IDs that are modified (directly or through GEP-derived
    /// pointers) within the loop. This resolves GEP chains so that stores
    /// through `gep(alloca, offset1)` are recognized as modifying the same
    /// base alloca that an intrinsic reads through `gep(alloca, offset2)`.
    modified_base_allocas: FxHashSet<u32>,
    /// Whether the loop body contains any function calls (Call, CallIndirect,
    /// or InlineAsm with clobbers). Calls can modify any global variable,
    /// so loads from globals cannot be hoisted past calls.
    has_calls: bool,
    /// Whether the loop body has any stores through non-alloca pointers,
    /// which could potentially write to global variables. A pointer loaded
    /// from memory at runtime (e.g., through a struct field) may point to
    /// global memory, so any store through a non-alloca pointer must be
    /// treated conservatively as potentially modifying globals.
    has_global_derived_stores: bool,
}

impl LoopMemoryInfo {
    /// Whether the loop has any store operations at all (to any target).
    fn has_any_stores(&self) -> bool {
        !self.stored_allocas.is_empty() || self.has_global_derived_stores
    }
}

/// Build a mapping from GEP/Copy result values to their ultimate base alloca.
/// Follows chains like `gep(gep(alloca, off1), off2)` → alloca.
fn build_value_to_base_alloca(func: &IrFunction, alloca_info: &AllocaAnalysis) -> FxHashMap<u32, u32> {
    let mut map: FxHashMap<u32, u32> = FxHashMap::default();
    // Seed: every alloca maps to itself
    for &alloca_id in &alloca_info.alloca_values {
        map.insert(alloca_id, alloca_id);
    }
    // Propagate through GEP and Copy chains (fixpoint)
    let mut changed = true;
    while changed {
        changed = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::GetElementPtr { dest, base, .. } => {
                        if let Some(&base_alloca) = map.get(&base.0) {
                            if map.insert(dest.0, base_alloca) != Some(base_alloca) {
                                changed = true;
                            }
                        }
                    }
                    Instruction::Copy { dest, src: Operand::Value(src_val) } => {
                        if let Some(&base_alloca) = map.get(&src_val.0) {
                            if map.insert(dest.0, base_alloca) != Some(base_alloca) {
                                changed = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    map
}

/// Scan a loop body to determine which allocas are modified and what
/// memory side effects the loop has (calls, stores to unknown pointers).
fn analyze_loop_memory(
    func: &IrFunction,
    loop_body: &FxHashSet<usize>,
    alloca_info: &AllocaAnalysis,
    _global_addr_values: &FxHashSet<u32>,
    value_to_base_alloca: &FxHashMap<u32, u32>,
) -> LoopMemoryInfo {
    let mut stored_allocas = FxHashSet::default();
    let mut has_calls = false;
    let mut has_global_derived_stores = false;

    let collect_ptr = |op: &Operand, set: &mut FxHashSet<u32>| {
        if let Operand::Value(v) = op {
            set.insert(v.0);
        }
    };

    for &block_idx in loop_body {
        if block_idx >= func.blocks.len() {
            continue;
        }
        for inst in &func.blocks[block_idx].instructions {
            match inst {
                Instruction::Store { ptr, .. } => {
                    stored_allocas.insert(ptr.0);
                    // A store through any pointer that is NOT a known alloca
                    // could potentially modify global memory. For example, a
                    // pointer loaded from a struct field at runtime may point
                    // to a global variable (e.g., linked list node->next->prev
                    // where next points to a global anchor). We must
                    // conservatively flag all non-alloca stores as potentially
                    // modifying globals to prevent incorrect hoisting of global
                    // loads.
                    if !alloca_info.alloca_values.contains(&ptr.0) {
                        has_global_derived_stores = true;
                    }
                }
                Instruction::AtomicRmw { ptr, .. } => {
                    collect_ptr(ptr, &mut stored_allocas);
                    if let Operand::Value(v) = ptr {
                        if !alloca_info.alloca_values.contains(&v.0) {
                            has_global_derived_stores = true;
                        }
                    }
                }
                Instruction::AtomicCmpxchg { ptr, .. } => {
                    collect_ptr(ptr, &mut stored_allocas);
                    if let Operand::Value(v) = ptr {
                        if !alloca_info.alloca_values.contains(&v.0) {
                            has_global_derived_stores = true;
                        }
                    }
                }
                Instruction::AtomicStore { ptr, .. } => {
                    collect_ptr(ptr, &mut stored_allocas);
                    if let Operand::Value(v) = ptr {
                        if !alloca_info.alloca_values.contains(&v.0) {
                            has_global_derived_stores = true;
                        }
                    }
                }
                Instruction::Memcpy { dest, .. } => {
                    stored_allocas.insert(dest.0);
                    if !alloca_info.alloca_values.contains(&dest.0) {
                        has_global_derived_stores = true;
                    }
                }
                // Function calls can modify any global state.
                Instruction::Call { .. } | Instruction::CallIndirect { .. } => {
                    has_calls = true;
                }
                // InlineAsm output operands are pointers (allocas) that the backend
                // stores results into. Track them as stores to prevent LICM from
                // hoisting loads of those allocas out of loops with inline asm.
                // InlineAsm with clobbers is also treated as a call-like barrier.
                Instruction::InlineAsm { outputs, clobbers, .. } => {
                    for (_, ptr, _) in outputs {
                        stored_allocas.insert(ptr.0);
                    }
                    // InlineAsm with "memory" clobber or any clobbers conservatively
                    // treated as potentially modifying globals.
                    if !clobbers.is_empty() {
                        has_calls = true;
                    }
                }
                // Vec128 intrinsics write their result through dest_ptr.
                // Track this as a store so that reads from the same alloca
                // (by other intrinsics or loads) are not incorrectly hoisted.
                Instruction::Intrinsic { dest_ptr: Some(dptr), .. } => {
                    stored_allocas.insert(dptr.0);
                }
                // VaStart/VaEnd/VaCopy/VaArg modify va_list state but not globals.
                _ => {}
            }
        }
    }

    // Resolve stored pointers to their base allocas so that stores through
    // gep(alloca, off1) are recognized as modifying the same base alloca
    // that an intrinsic might read through gep(alloca, off2).
    let mut modified_base_allocas = FxHashSet::default();
    for &stored_id in &stored_allocas {
        if let Some(&base) = value_to_base_alloca.get(&stored_id) {
            modified_base_allocas.insert(base);
        }
    }

    LoopMemoryInfo { stored_allocas, modified_base_allocas, has_calls, has_global_derived_stores }
}

/// Check if a Load instruction is safe to hoist from a loop.
///
/// A load is safe to hoist if:
/// 1. Its pointer operand is loop-invariant
/// 2. The memory it reads is not modified inside the loop:
///    a. If ptr is an alloca that IS address-taken: never hoisted
///    b. If ptr is an alloca that is NOT address-taken: safe if no store targets it
///    c. If ptr is a GlobalAddr: safe if the loop has no calls, no unknown stores,
///    and no store targets that specific GlobalAddr value
fn is_load_hoistable(
    ptr: &Value,
    alloca_info: &AllocaAnalysis,
    loop_mem: &LoopMemoryInfo,
    loop_defined: &FxHashSet<u32>,
    invariant: &FxHashSet<u32>,
    global_addr_values: &FxHashSet<u32>,
) -> bool {
    let ptr_id = ptr.0;

    // The pointer must be loop-invariant (defined outside the loop or already hoisted).
    let ptr_is_invariant = !loop_defined.contains(&ptr_id) || invariant.contains(&ptr_id);
    if !ptr_is_invariant {
        return false;
    }

    // Check if loading from an alloca.
    if alloca_info.alloca_values.contains(&ptr_id) {
        // If the alloca is address-taken (used in GEP, passed to call, etc.),
        // it can be modified through derived pointers. These stores won't
        // appear in stored_allocas under the alloca's own Value ID, so we
        // cannot prove the memory is unmodified. Reject unconditionally.
        if alloca_info.address_taken.contains(&ptr_id) {
            return false;
        }

        // The alloca itself must not be stored to inside the loop.
        if loop_mem.stored_allocas.contains(&ptr_id) {
            return false;
        }

        return true;
    }

    // Check if loading from a GlobalAddr pointer.
    // A load from a GlobalAddr is safe to hoist if:
    // 1. No function calls in the loop (calls could modify any global)
    // 2. No stores to unknown pointers (could alias any global)
    // 3. No store in the loop directly targets this GlobalAddr value
    //
    // This is particularly important for inner rendering loops (e.g., DOOM's
    // R_DrawColumn) where globals like dc_source and dc_colormap are read
    // every iteration but never written.
    if global_addr_values.contains(&ptr_id) {
        if loop_mem.has_calls {
            return false;
        }
        if loop_mem.has_global_derived_stores {
            return false;
        }
        // Check if any store in the loop directly targets this GlobalAddr.
        if loop_mem.stored_allocas.contains(&ptr_id) {
            return false;
        }
        return true;
    }

    // For other non-alloca pointers (e.g., GEP results), we cannot easily
    // determine safety without alias analysis. Be conservative.
    // TODO: Implement alias analysis for GEP-based loads
    false
}

/// Hoist loop-invariant instructions from a natural loop to a preheader.
///
/// Returns the number of instructions hoisted.
fn hoist_loop_invariants(
    func: &mut IrFunction,
    natural_loop: &NaturalLoop,
    preds: &analysis::FlatAdj,
    alloca_info: &AllocaAnalysis,
) -> usize {
    let header = natural_loop.header;

    // Find the preheader: the single predecessor of the header outside the loop.
    let preheader = loop_analysis::find_preheader(header, &natural_loop.body, preds);
    let preheader = match preheader {
        Some(ph) => ph,
        None => return 0, // No suitable preheader found
    };

    // Build the set of Value IDs defined inside the loop.
    // InlineAsm outputs are included: after mem2reg promotion, InlineAsm output
    // pointers become fresh SSA values that define new values, but
    // Instruction::dest() returns None for InlineAsm, so we must add them
    // explicitly to prevent LICM from hoisting uses of those values.
    let mut loop_defined: FxHashSet<u32> = FxHashSet::default();
    for &block_idx in &natural_loop.body {
        if block_idx < func.blocks.len() {
            for inst in &func.blocks[block_idx].instructions {
                if let Some(dest) = inst.dest() {
                    loop_defined.insert(dest.0);
                }
                if let Instruction::InlineAsm { outputs, .. } = inst {
                    for (_, out_val, _) in outputs {
                        loop_defined.insert(out_val.0);
                    }
                }
            }
        }
    }

    // Build a set of Value IDs that point to global memory — this includes
    // direct GlobalAddr instructions AND any GetElementPtr that derives a
    // pointer from a GlobalAddr (transitively). Stores through any of these
    // pointers can modify global variables, so they must be tracked to prevent
    // incorrect hoisting of global loads.
    let mut global_addr_values: FxHashSet<u32> = FxHashSet::default();
    for block in func.blocks.iter() {
        for inst in &block.instructions {
            if let Instruction::GlobalAddr { dest, .. } = inst {
                global_addr_values.insert(dest.0);
            }
        }
    }
    // Transitively include values derived from GlobalAddr: GEP results whose
    // base is a GlobalAddr-derived value, and Copy destinations whose source
    // is a GlobalAddr-derived value. Without this transitive closure, stores
    // through derived pointers (e.g., `p->j = ...` where `p = &global_struct`)
    // are not recognized as global-derived stores, causing LICM to incorrectly
    // hoist loads of globals from loops containing such stores.
    let mut changed_closure = true;
    while changed_closure {
        changed_closure = false;
        for block in func.blocks.iter() {
            for inst in &block.instructions {
                match inst {
                    Instruction::GetElementPtr { dest, base, .. } => {
                        if global_addr_values.contains(&base.0) && global_addr_values.insert(dest.0) {
                            changed_closure = true;
                        }
                    }
                    Instruction::Copy { dest, src: Operand::Value(src_val) } => {
                        if global_addr_values.contains(&src_val.0) && global_addr_values.insert(dest.0) {
                            changed_closure = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Build a mapping from values to their base alloca (following GEP chains).
    let value_to_base_alloca = build_value_to_base_alloca(func, alloca_info);

    // Analyze loop memory for load hoisting.
    let loop_mem = analyze_loop_memory(func, &natural_loop.body, alloca_info, &global_addr_values, &value_to_base_alloca);

    // Iteratively identify loop-invariant instructions.
    // An instruction is loop-invariant if:
    // 1. It is hoistable (pure, no side effects) OR it is a safe-to-hoist load
    // 2. All its Value operands are either:
    //    a. Not defined in the loop (defined outside), OR
    //    b. Already identified as loop-invariant
    let mut invariant: FxHashSet<u32> = FxHashSet::default();
    let mut hoistable_insts: Vec<(usize, usize, Instruction)> = Vec::new(); // (block_idx, inst_idx, inst)

    let mut changed = true;
    while changed {
        changed = false;
        for &block_idx in &natural_loop.body {
            if block_idx >= func.blocks.len() {
                continue;
            }
            let block = &func.blocks[block_idx];
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                let dest = match inst.dest() {
                    Some(d) => d,
                    None => continue,
                };

                // Skip if already identified as invariant
                if invariant.contains(&dest.0) {
                    continue;
                }

                // Determine if this instruction can be hoisted
                let can_hoist = if is_hoistable(inst) {
                    // Pure instruction: check all operands are loop-invariant
                    // Use callback to avoid Vec allocation in this hot loop
                    let mut all_invariant = true;
                    for_each_hoistable_operand(inst, |val_id| {
                        if all_invariant && loop_defined.contains(&val_id) && !invariant.contains(&val_id) {
                            all_invariant = false;
                        }
                    });
                    // For pure intrinsics: also check that pointer args don't read
                    // from allocas that are modified inside the loop. The pointer
                    // value itself may be loop-invariant (e.g., an alloca defined
                    // in the entry block), but the data at that pointer can change.
                    //
                    // Three checks are applied:
                    // 1. Direct: arg IS an alloca that's directly stored to in the loop
                    // 2. GEP-derived: arg resolves through GEP chains to a base alloca
                    //    that's modified (directly or through GEPs) in the loop
                    // 3. Address-taken: arg IS an address-taken alloca (its data can
                    //    be modified through derived pointers like GEPs whose stores
                    //    may not be directly attributed to the base alloca)
                    if all_invariant {
                        if let Instruction::Intrinsic { args, .. } = inst {
                            for arg in args {
                                if let Operand::Value(v) = arg {
                                    // Direct alloca check (original)
                                    if alloca_info.alloca_values.contains(&v.0)
                                        && loop_mem.stored_allocas.contains(&v.0)
                                    {
                                        all_invariant = false;
                                        break;
                                    }
                                    // Address-taken alloca check: if the arg is an alloca
                                    // whose address is taken (used in GEP, passed to calls),
                                    // the data it points to can be modified through derived
                                    // pointers. We must check if ANY store in the loop
                                    // could potentially modify this alloca's data.
                                    if alloca_info.alloca_values.contains(&v.0)
                                        && alloca_info.address_taken.contains(&v.0)
                                        && loop_mem.has_any_stores()
                                    {
                                        all_invariant = false;
                                        break;
                                    }
                                    // GEP-derived pointer check: resolve to base alloca
                                    // and check if that base alloca is modified in the loop
                                    // (including through other GEP-derived pointers)
                                    if let Some(&base) = value_to_base_alloca.get(&v.0) {
                                        if loop_mem.modified_base_allocas.contains(&base) {
                                            all_invariant = false;
                                            break;
                                        }
                                        // Also check if the base alloca is address-taken
                                        // and the loop has stores (which could go through
                                        // derived pointers not tracked by stored_allocas)
                                        if alloca_info.address_taken.contains(&base)
                                            && loop_mem.has_any_stores()
                                        {
                                            all_invariant = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    all_invariant
                } else if let Instruction::Load { ptr, .. } = inst {
                    is_load_hoistable(ptr, alloca_info, &loop_mem,
                                             &loop_defined, &invariant,
                                             &global_addr_values)
                } else {
                    false
                };

                if can_hoist {
                    invariant.insert(dest.0);
                    hoistable_insts.push((block_idx, inst_idx, inst.clone()));
                    changed = true;
                }
            }
        }
    }

    if hoistable_insts.is_empty() {
        return 0;
    }

    // Collect the set of instruction indices to remove from each block
    let mut to_remove: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
    for &(block_idx, inst_idx, _) in &hoistable_insts {
        to_remove.entry(block_idx).or_default().insert(inst_idx);
    }

    // Remove hoisted instructions from their original blocks
    for (&block_idx, indices) in &to_remove {
        if block_idx < func.blocks.len() {
            let block = &mut func.blocks[block_idx];
            let mut new_insts = Vec::with_capacity(block.instructions.len());
            let old_spans = std::mem::take(&mut block.source_spans);
            let has_spans = old_spans.len() == block.instructions.len() && !old_spans.is_empty();
            let mut new_spans = if has_spans { Vec::with_capacity(old_spans.len()) } else { Vec::new() };
            for (i, inst) in block.instructions.drain(..).enumerate() {
                if !indices.contains(&i) {
                    new_insts.push(inst);
                    if has_spans { new_spans.push(old_spans[i]); }
                }
            }
            block.instructions = new_insts;
            block.source_spans = new_spans;
        }
    }

    // Insert hoisted instructions at the end of the preheader block
    // (before its terminator, which is implicit - terminators are separate from instructions).
    // We need to insert in the order they originally appeared to maintain SSA dominance.
    // Sort by (block_idx in RPO order, then inst_idx) to preserve def-before-use.
    hoistable_insts.sort_by_key(|&(block_idx, inst_idx, _)| (block_idx, inst_idx));

    // Deduplicate: if the same instruction was found multiple times (shouldn't happen
    // with our algorithm, but be safe), keep only the first occurrence.
    let mut seen_dests: FxHashSet<u32> = FxHashSet::default();
    let mut unique_insts: Vec<Instruction> = Vec::new();
    for (_, _, inst) in hoistable_insts {
        if let Some(dest) = inst.dest() {
            if seen_dests.insert(dest.0) {
                unique_insts.push(inst);
            }
        } else {
            unique_insts.push(inst);
        }
    }

    let num_hoisted = unique_insts.len();

    // Topologically sort: if instruction A defines a value used by instruction B,
    // A must come before B in the preheader.
    let sorted = topological_sort_instructions(unique_insts);

    // Insert at the end of the preheader (before terminator)
    if preheader < func.blocks.len() {
        let preheader_block = &mut func.blocks[preheader];
        let num_sorted = sorted.len();
        preheader_block.instructions.extend(sorted);
        if !preheader_block.source_spans.is_empty() {
            preheader_block.source_spans.extend(
                std::iter::repeat_n(crate::common::source::Span::dummy(), num_sorted)
            );
        }
    }

    num_hoisted
}

/// Topologically sort instructions so that definitions come before uses.
/// This ensures SSA correctness in the preheader.
fn topological_sort_instructions(mut insts: Vec<Instruction>) -> Vec<Instruction> {
    if insts.len() <= 1 {
        return insts;
    }

    // Build a map from dest value ID to index in insts
    let mut def_to_idx: FxHashMap<u32, usize> = FxHashMap::default();
    for (i, inst) in insts.iter().enumerate() {
        if let Some(dest) = inst.dest() {
            def_to_idx.insert(dest.0, i);
        }
    }

    // Build dependency edges: inst[i] depends on inst[j] if inst[i] uses a value defined by inst[j]
    let n = insts.len();
    let mut in_degree = vec![0u32; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, inst) in insts.iter().enumerate() {
        for_each_hoistable_operand(inst, |val_id| {
            if let Some(&def_idx) = def_to_idx.get(&val_id) {
                if def_idx != i {
                    dependents[def_idx].push(i);
                    in_degree[i] += 1;
                }
            }
        });
    }

    // Kahn's algorithm
    let mut queue: Vec<usize> = Vec::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push(i);
        }
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(idx) = queue.pop() {
        order.push(idx);
        for &dep in &dependents[idx] {
            in_degree[dep] -= 1;
            if in_degree[dep] == 0 {
                queue.push(dep);
            }
        }
    }

    // If there's a cycle (shouldn't happen in SSA), fall back to original order
    if order.len() != n {
        return insts;
    }

    // Reorder instructions according to topological order.
    // Use Option slots to allow taking ownership from arbitrary positions.
    let mut result = Vec::with_capacity(n);
    let mut slots: Vec<Option<Instruction>> = insts.drain(..).map(Some).collect();
    for &idx in &order {
        if let Some(inst) = slots[idx].take() {
            result.push(inst);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::{AddressSpace, IrType};
    use crate::ir::reexports::{BasicBlock, BlockId, IrBinOp, IrCmpOp, IrConst};

    /// Helper to create a simple loop: preheader -> header -> body -> header, header -> exit
    fn make_loop_func() -> IrFunction {
        let mut func = IrFunction::new("test_loop".into(), IrType::I32, vec![], false);

        // Block 0 (preheader): i = 0, n = 10
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I32(0)),
                },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Const(IrConst::I32(10)),
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): phi for i, check i < n
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(2),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(0)), BlockId(0)),
                        (Operand::Value(Value(5)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(3),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Value(Value(1)),
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

        // Block 2 (body): loop-invariant computation (n * 4), then i++
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                // n * 4 is loop-invariant (both n=Value(1) and 4 are outside loop)
                Instruction::BinOp {
                    dest: Value(4),
                    op: IrBinOp::Mul,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(4)),
                    ty: IrType::I32,
                },
                // i + 1 is NOT loop-invariant (uses i = Value(2) which is a phi)
                Instruction::BinOp {
                    dest: Value(5),
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
            terminator: Terminator::Return(Some(Operand::Value(Value(4)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 6;
        func
    }

    #[test]
    fn test_find_natural_loops() {
        let func = make_loop_func();
        let label_to_idx = analysis::build_label_map(&func);
        let (preds, succs) = analysis::build_cfg(&func, &label_to_idx);
        let idom = analysis::compute_dominators(func.blocks.len(), &preds, &succs);
        let loops = loop_analysis::find_natural_loops(func.blocks.len(), &preds, &succs, &idom);

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, 1); // header is block 1
        assert!(loops[0].body.contains(&1)); // header in body
        assert!(loops[0].body.contains(&2)); // loop body block in body
        assert!(!loops[0].body.contains(&0)); // preheader not in body
        assert!(!loops[0].body.contains(&3)); // exit not in body
    }

    #[test]
    fn test_licm_hoists_invariant() {
        let mut func = make_loop_func();
        let alloca_info = analyze_allocas(&func);
        let label_to_idx = analysis::build_label_map(&func);
        let (preds, succs) = analysis::build_cfg(&func, &label_to_idx);
        let idom = analysis::compute_dominators(func.blocks.len(), &preds, &succs);
        let loops = loop_analysis::find_natural_loops(func.blocks.len(), &preds, &succs, &idom);

        let hoisted = hoist_loop_invariants(&mut func, &loops[0], &preds, &alloca_info);

        // n * 4 should be hoisted (1 instruction)
        assert_eq!(hoisted, 1);

        // The preheader (block 0) should now have 3 instructions
        assert_eq!(func.blocks[0].instructions.len(), 3);

        // The loop body (block 2) should have only i+1 (the non-invariant)
        assert_eq!(func.blocks[2].instructions.len(), 1);

        // Check that the hoisted instruction is the multiply
        let last_preheader_inst = func.blocks[0].instructions.last().unwrap();
        match last_preheader_inst {
            Instruction::BinOp { op: IrBinOp::Mul, .. } => {} // correct
            other => panic!("Expected BinOp::Mul, got {:?}", other),
        }
    }

    #[test]
    fn test_topological_sort() {
        // a = 1 + 2, b = a + 3 -> b depends on a, so a must come first
        let insts = vec![
            Instruction::BinOp {
                dest: Value(10),
                op: IrBinOp::Add,
                lhs: Operand::Value(Value(5)), // uses value defined by second inst
                rhs: Operand::Const(IrConst::I32(3)),
                ty: IrType::I32,
            },
            Instruction::BinOp {
                dest: Value(5),
                op: IrBinOp::Add,
                lhs: Operand::Const(IrConst::I32(1)),
                rhs: Operand::Const(IrConst::I32(2)),
                ty: IrType::I32,
            },
        ];

        let sorted = topological_sort_instructions(insts);
        // Value(5) should come before Value(10)
        assert_eq!(sorted[0].dest(), Some(Value(5)));
        assert_eq!(sorted[1].dest(), Some(Value(10)));
    }

    #[test]
    fn test_licm_hoists_load_from_unmodified_alloca() {
        // Test: load from alloca that is NOT address-taken and NOT stored to
        // in the loop should be hoisted.
        //
        // entry:
        //   %0 = alloca i32       (parameter alloca for 'n')
        //   store 42, %0          (initial parameter store)
        //   br loop_header
        //
        // loop_header:
        //   %2 = phi [%init, entry], [%5, body]
        //   %3 = load %0          (load from alloca - HOISTED)
        //   %cmp = cmp slt %2, %3
        //   br %cmp, body, exit
        //
        // body:
        //   %5 = add %2, 1
        //   br loop_header
        //
        // exit:
        //   ret %2
        let mut func = IrFunction::new("test_load_hoist".into(), IrType::I32, vec![], false);

        // Block 0 (entry): alloca + store + init
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Alloca {
                    dest: Value(0),
                    ty: IrType::I32,
                    size: 4,
                    align: 4,
                    volatile: false,
                },
                Instruction::Store { val: Operand::Const(IrConst::I32(42)), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Const(IrConst::I32(0)),
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): phi + load from alloca + compare
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(2),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Value(Value(1)), BlockId(0)),
                        (Operand::Value(Value(5)), BlockId(2)),
                    ],
                },
                Instruction::Load { dest: Value(3), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
                Instruction::Cmp {
                    dest: Value(4),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Value(Value(3)),
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

        // Block 2 (body): i++
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::BinOp {
                    dest: Value(5),
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
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 6;

        let alloca_info = analyze_allocas(&func);
        let label_to_idx = analysis::build_label_map(&func);
        let (preds, succs) = analysis::build_cfg(&func, &label_to_idx);
        let idom = analysis::compute_dominators(func.blocks.len(), &preds, &succs);
        let loops = loop_analysis::find_natural_loops(func.blocks.len(), &preds, &succs, &idom);

        assert_eq!(loops.len(), 1);
        let hoisted = hoist_loop_invariants(&mut func, &loops[0], &preds, &alloca_info);

        // The load from the unmodified, non-address-taken alloca should be hoisted
        assert_eq!(hoisted, 1);

        // The preheader (block 0) should now have 4 instructions (alloca + store + copy + hoisted load)
        assert_eq!(func.blocks[0].instructions.len(), 4);

        // The loop header (block 1) should have lost the load: phi + cmp
        assert_eq!(func.blocks[1].instructions.len(), 2);
    }

    #[test]
    fn test_licm_does_not_hoist_load_from_modified_alloca() {
        // Test: load from an alloca that IS stored to in the loop should NOT be hoisted.
        let mut func = IrFunction::new("test_no_hoist".into(), IrType::I32, vec![], false);

        // Block 0: alloca + initial store
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Alloca {
                    dest: Value(0),
                    ty: IrType::I32,
                    size: 4,
                    align: 4,
                    volatile: false,
                },
                Instruction::Store { val: Operand::Const(IrConst::I32(0)), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (header): load from alloca
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
                Instruction::Cmp {
                    dest: Value(2),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(10)),
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

        // Block 2 (body): store to same alloca (modifies it!)
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::BinOp {
                    dest: Value(3),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
                Instruction::Store { val: Operand::Value(Value(3)), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 4;

        let alloca_info = analyze_allocas(&func);
        let label_to_idx = analysis::build_label_map(&func);
        let (preds, succs) = analysis::build_cfg(&func, &label_to_idx);
        let idom = analysis::compute_dominators(func.blocks.len(), &preds, &succs);
        let loops = loop_analysis::find_natural_loops(func.blocks.len(), &preds, &succs, &idom);

        assert_eq!(loops.len(), 1);
        let hoisted = hoist_loop_invariants(&mut func, &loops[0], &preds, &alloca_info);

        // Nothing should be hoisted because the alloca is stored to in the loop
        assert_eq!(hoisted, 0);
    }

    #[test]
    fn test_licm_does_not_hoist_load_from_inline_asm_output_alloca() {
        // Test: load from an alloca that is written by InlineAsm in the loop
        // should NOT be hoisted. This is the pattern from CAS loops like:
        //   do { asm("cmpxchg8b" : "=r"(succeeded) : ...); } while (!succeeded);
        //
        // entry:
        //   %0 = alloca i32       (succeeded alloca)
        //   br loop_header
        //
        // loop_header (body):
        //   inline_asm outputs=[%0]  (writes to succeeded alloca)
        //   br cond_block
        //
        // cond_block:
        //   %1 = load %0          (load succeeded - must NOT be hoisted)
        //   %2 = cmp eq %1, 0     (check !succeeded)
        //   br %2, loop_header, exit
        //
        // exit:
        //   ret 0
        let mut func = IrFunction::new("test_asm_output".into(), IrType::I32, vec![], false);

        // Block 0 (entry): alloca for succeeded
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Alloca {
                    dest: Value(0),
                    ty: IrType::I32,
                    size: 4,
                    align: 4,
                    volatile: false,
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (loop body): inline asm that writes to the alloca
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::InlineAsm {
                    template: "nop".to_string(),
                    outputs: vec![("=r".to_string(), Value(0), None)],
                    inputs: vec![],
                    clobbers: vec![],
                    operand_types: vec![IrType::I32],
                    goto_labels: vec![],
                    input_symbols: vec![],
                    seg_overrides: vec![AddressSpace::Default],
                },
            ],
            terminator: Terminator::Branch(BlockId(2)),
            source_spans: Vec::new(),
        });

        // Block 2 (cond): load succeeded, check if zero, branch back or exit
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
                Instruction::Cmp {
                    dest: Value(2),
                    op: IrCmpOp::Eq,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Const(IrConst::I32(0)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(2)),
                true_label: BlockId(1),  // loop back
                false_label: BlockId(3), // exit
            },
            source_spans: Vec::new(),
        });

        // Block 3 (exit)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
            source_spans: Vec::new(),
        });

        func.next_value_id = 3;

        let alloca_info = analyze_allocas(&func);
        let label_to_idx = analysis::build_label_map(&func);
        let (preds, succs) = analysis::build_cfg(&func, &label_to_idx);
        let idom = analysis::compute_dominators(func.blocks.len(), &preds, &succs);
        let loops = loop_analysis::find_natural_loops(func.blocks.len(), &preds, &succs, &idom);

        assert_eq!(loops.len(), 1);
        let hoisted = hoist_loop_invariants(&mut func, &loops[0], &preds, &alloca_info);

        // Nothing should be hoisted: the alloca is written by inline asm in the loop
        assert_eq!(hoisted, 0);

        // The cond block should still have its load + cmp
        assert_eq!(func.blocks[2].instructions.len(), 2);
    }

    #[test]
    fn test_promoted_asm_output_not_hoisted() {
        // Regression test: after mem2reg promotes an InlineAsm output alloca to
        // an SSA value, LICM must recognize the fresh SSA output as loop-defined.
        // Without this fix, LICM incorrectly hoists instructions using the
        // InlineAsm output because Instruction::dest() returns None for InlineAsm,
        // so the output value doesn't appear in loop_defined.
        //
        // This is the pattern from Linux kernel's per_cpu_ptr macro:
        //   asm("" : "=r"(__ptr) : "0"(addr))  →  output is fresh SSA Value
        //   result = __ptr + __per_cpu_offset[cpu]
        //
        // After mem2reg, the InlineAsm output is Value(5) (not an alloca).
        // The Copy from Value(5) must NOT be hoisted to the preheader.
        //
        // entry:
        //   %0 = globaladdr @addr        (loop-invariant address)
        //   br loop_header
        //
        // loop_header:
        //   %3 = phi [entry: 0, body: %7]
        //   %4 = cmp lt %3, 10
        //   br %4, body, exit
        //
        // body:
        //   inline_asm outputs=[%5] inputs=[%0]  (%5 is fresh SSA output)
        //   %6 = copy %5                         (must NOT be hoisted)
        //   %7 = binop add %3, 1
        //   br loop_header
        //
        // exit:
        //   ret 0
        use crate::common::types::AddressSpace;

        let mut func = IrFunction::new("test_promoted_asm".into(), IrType::I32, vec![], false);

        // Block 0 (entry): globaladdr + branch
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::GlobalAddr {
                    dest: Value(0),
                    name: "addr".into(),
                },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // Block 1 (loop_header): phi + cmp + condbranch
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(3),
                    ty: IrType::I32,
                    incoming: vec![
                        (Operand::Const(IrConst::I32(0)), BlockId(0)),
                        (Operand::Value(Value(7)), BlockId(2)),
                    ],
                },
                Instruction::Cmp {
                    dest: Value(4),
                    op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Const(IrConst::I32(10)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(4)),
                true_label: BlockId(2),   // body
                false_label: BlockId(3),  // exit
            },
            source_spans: Vec::new(),
        });

        // Block 2 (body): inline asm with promoted output + copy + increment
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                // InlineAsm with promoted output: Value(5) is a fresh SSA value
                // (not an alloca pointer). This is what mem2reg produces.
                Instruction::InlineAsm {
                    template: String::new(), // empty template like per_cpu_ptr
                    outputs: vec![("=r".to_string(), Value(5), None)],
                    inputs: vec![("0".to_string(), Operand::Value(Value(0)), None)],
                    clobbers: vec![],
                    operand_types: vec![IrType::I64],
                    goto_labels: vec![],
                    input_symbols: vec![],
                    seg_overrides: vec![AddressSpace::Default],
                },
                // Copy from the InlineAsm output — this MUST NOT be hoisted
                Instruction::Copy {
                    dest: Value(6),
                    src: Operand::Value(Value(5)),
                },
                // Loop increment
                Instruction::BinOp {
                    dest: Value(7),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(3)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Branch(BlockId(1)), // back to header
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

        let alloca_info = analyze_allocas(&func);
        let label_to_idx = analysis::build_label_map(&func);
        let (preds, succs) = analysis::build_cfg(&func, &label_to_idx);
        let idom = analysis::compute_dominators(func.blocks.len(), &preds, &succs);
        let loops = loop_analysis::find_natural_loops(func.blocks.len(), &preds, &succs, &idom);

        assert_eq!(loops.len(), 1);
        let hoisted = hoist_loop_invariants(&mut func, &loops[0], &preds, &alloca_info);

        // The Copy from Value(5) must NOT be hoisted because Value(5) is
        // defined by InlineAsm inside the loop. The BinOp depends on the
        // phi (also loop-defined), so it shouldn't be hoisted either.
        assert_eq!(hoisted, 0, "Nothing should be hoisted: Copy depends on InlineAsm output defined inside loop");

        // The body block should still have all 3 instructions
        assert_eq!(func.blocks[2].instructions.len(), 3);
    }
}
