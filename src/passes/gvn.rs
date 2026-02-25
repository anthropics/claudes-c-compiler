//! Dominator-based Global Value Numbering (GVN) pass.
//!
//! This pass assigns "value numbers" to expressions and replaces redundant
//! computations with references to previously computed values (CSE).
//!
//! The pass walks the dominator tree in DFS order with scoped hash tables,
//! so expressions computed in dominating blocks are visible to all dominated
//! blocks. On backtracking, the hash tables are restored to their previous
//! state (same scoping pattern as rename_block in mem2reg).
//!
//! Value-numbered instruction types:
//! - BinOp (with commutative operand canonicalization)
//! - UnaryOp
//! - Cmp
//! - Cast (type-to-type conversions)
//! - GetElementPtr (base + offset address computation)
//! - Load (redundant load elimination within dominator scope, invalidated
//!   by stores, calls, and other memory-clobbering instructions)

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::{AddressSpace, IrType};
use crate::ir::reexports::{
    ConstHashKey,
    Instruction,
    IrBinOp,
    IrCmpOp,
    IrFunction,
    IrUnaryOp,
    Operand,
    Value,
};
use crate::ir::analysis;

/// A value number expression key. Two instructions with the same ExprKey
/// compute the same value (assuming their operands are equivalent).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExprKey {
    BinOp { op: IrBinOp, lhs: VNOperand, rhs: VNOperand, ty: IrType },
    UnaryOp { op: IrUnaryOp, src: VNOperand, ty: IrType },
    Cmp { op: IrCmpOp, lhs: VNOperand, rhs: VNOperand, ty: IrType },
    Cast { src: VNOperand, from_ty: IrType, to_ty: IrType },
    Gep { base: VNOperand, offset: VNOperand, ty: IrType },
    /// Load CSE key: two loads from the same pointer with the same type
    /// produce the same value if no intervening memory modification occurs.
    Load { ptr: VNOperand, ty: IrType },
}

/// Returns true if the ExprKey represents a Load (memory-dependent expression).
impl ExprKey {
    fn is_load(&self) -> bool {
        matches!(self, ExprKey::Load { .. })
    }
}

/// A value-numbered operand: either a constant or a value number.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum VNOperand {
    Const(ConstHashKey),
    ValueNum(u32),
}

/// Key for store-to-load forwarding: identifies a memory location by pointer
/// value number and access type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StoreFwdKey {
    ptr_vn: VNOperand,
    ty: IrType,
}

/// Mutable state for the GVN pass, threaded through the dominator-tree DFS.
///
/// Groups the value numbering tables, expression maps, and rollback logs that
/// were previously passed as 9 separate `&mut` parameters. The rollback logs
/// enable scoped hash table semantics: on entering a dominator-tree subtree,
/// save the log positions; on backtracking, restore entries to undo changes
/// made in that subtree.
struct GvnState {
    /// Maps Value ID -> value number. Indexed by `Value.0`.
    value_numbers: Vec<u32>,
    /// Next value number to assign.
    next_vn: u32,
    /// Pure expression -> canonical value (not memory-dependent).
    expr_to_value: FxHashMap<ExprKey, Value>,
    /// Load expression -> (canonical value, generation). Separated from
    /// `expr_to_value` so loads can be invalidated independently via
    /// generation bumping.
    load_expr_to_value: FxHashMap<ExprKey, (Value, u32)>,
    /// Generation counter for O(1) load CSE invalidation. When a memory-
    /// clobbering instruction is encountered, bump this counter; cached
    /// load entries with older generations are considered stale.
    load_generation: u32,
    /// Store-to-load forwarding map: (pointer VN, type) -> (stored operand, generation).
    /// When a Store writes a value to a pointer, we record it here. A subsequent
    /// Load from the same pointer (same VN) with matching type and generation can
    /// be replaced with a Copy of the stored value, eliminating the load entirely.
    /// Uses the same generation counter as load CSE for O(1) invalidation.
    store_fwd_map: FxHashMap<StoreFwdKey, (Operand, u32)>,
    /// Rollback log for `expr_to_value`: (key, previous_value).
    rollback_log: Vec<(ExprKey, Option<Value>)>,
    /// Rollback log for `load_expr_to_value`: (key, previous_entry).
    load_rollback_log: Vec<(ExprKey, Option<(Value, u32)>)>,
    /// Rollback log for `store_fwd_map`: (key, previous_entry).
    store_fwd_rollback_log: Vec<(StoreFwdKey, Option<(Operand, u32)>)>,
    /// Rollback log for `value_numbers`: (index, previous_vn).
    vn_log: Vec<(usize, u32)>,
    /// Total instructions eliminated across all blocks.
    total_eliminated: usize,
    /// Set of param alloca Value IDs whose address has escaped (used in
    /// non-Load/Store contexts). Store-to-load forwarding is disabled for
    /// these allocas because the backend's ParamRef optimization reads
    /// parameter values from the alloca slot, which may be modified by
    /// stores through aliased pointers.
    escaped_param_allocas: FxHashSet<u32>,
}

impl GvnState {
    /// Create a new GVN state sized for `max_value_id` values.
    fn new(max_value_id: usize, escaped_param_allocas: FxHashSet<u32>) -> Self {
        Self {
            value_numbers: vec![u32::MAX; max_value_id + 1],
            next_vn: 0,
            expr_to_value: FxHashMap::default(),
            load_expr_to_value: FxHashMap::default(),
            load_generation: 0,
            store_fwd_map: FxHashMap::default(),
            rollback_log: Vec::new(),
            load_rollback_log: Vec::new(),
            store_fwd_rollback_log: Vec::new(),
            vn_log: Vec::new(),
            total_eliminated: 0,
            escaped_param_allocas,
        }
    }

    /// Assign a fresh value number, returning it.
    fn fresh_vn(&mut self) -> u32 {
        let vn = self.next_vn;
        self.next_vn += 1;
        vn
    }

    /// Assign a fresh value number to `dest` and record it in the rollback log.
    fn assign_fresh_vn(&mut self, dest: Value) {
        let vn = self.fresh_vn();
        let idx = dest.0 as usize;
        if idx < self.value_numbers.len() {
            let old_vn = self.value_numbers[idx];
            self.vn_log.push((idx, old_vn));
            self.value_numbers[idx] = vn;
        }
    }

    /// Convert an Operand to a VNOperand for hashing.
    /// If the value hasn't been assigned a value number yet (e.g. a function
    /// parameter or an alloca whose definition appears later in the block),
    /// assign it a fresh unique VN on the spot to avoid collisions between
    /// different un-numbered values and already-assigned VNs.
    fn operand_to_vn(&mut self, op: &Operand) -> VNOperand {
        match op {
            Operand::Const(c) => VNOperand::Const(c.to_hash_key()),
            Operand::Value(v) => {
                let idx = v.0 as usize;
                // Ensure the table is large enough
                if idx >= self.value_numbers.len() {
                    self.value_numbers.resize(idx + 1, u32::MAX);
                }
                if self.value_numbers[idx] != u32::MAX {
                    VNOperand::ValueNum(self.value_numbers[idx])
                } else {
                    // Assign a fresh VN to this previously un-numbered value
                    let vn = self.fresh_vn();
                    let old_vn = self.value_numbers[idx];
                    self.vn_log.push((idx, old_vn));
                    self.value_numbers[idx] = vn;
                    VNOperand::ValueNum(vn)
                }
            }
        }
    }

    /// Try to create an ExprKey for an instruction (for value numbering).
    /// Returns the expression key and the destination value, or None if
    /// the instruction is not eligible for value numbering.
    fn make_expr_key(&mut self, inst: &Instruction) -> Option<(ExprKey, Value)> {
        match inst {
            Instruction::BinOp { dest, op, lhs, rhs, ty } => {
                let lhs_vn = self.operand_to_vn(lhs);
                let rhs_vn = self.operand_to_vn(rhs);

                // For commutative operations, canonicalize operand order
                let (lhs_vn, rhs_vn) = if op.is_commutative() {
                    canonical_order(lhs_vn, rhs_vn)
                } else {
                    (lhs_vn, rhs_vn)
                };

                Some((ExprKey::BinOp { op: *op, lhs: lhs_vn, rhs: rhs_vn, ty: *ty }, *dest))
            }
            Instruction::UnaryOp { dest, op, src, ty } => {
                let src_vn = self.operand_to_vn(src);
                Some((ExprKey::UnaryOp { op: *op, src: src_vn, ty: *ty }, *dest))
            }
            Instruction::Cmp { dest, op, lhs, rhs, ty } => {
                let lhs_vn = self.operand_to_vn(lhs);
                let rhs_vn = self.operand_to_vn(rhs);
                Some((ExprKey::Cmp { op: *op, lhs: lhs_vn, rhs: rhs_vn, ty: *ty }, *dest))
            }
            Instruction::Cast { dest, src, from_ty, to_ty } => {
                // Don't CSE casts to/from 128-bit types (complex codegen)
                if from_ty.is_128bit() || to_ty.is_128bit() {
                    return None;
                }
                let src_vn = self.operand_to_vn(src);
                Some((ExprKey::Cast { src: src_vn, from_ty: *from_ty, to_ty: *to_ty }, *dest))
            }
            Instruction::GetElementPtr { dest, base, offset, ty } => {
                let base_vn = self.operand_to_vn(&Operand::Value(*base));
                let offset_vn = self.operand_to_vn(offset);
                Some((ExprKey::Gep { base: base_vn, offset: offset_vn, ty: *ty }, *dest))
            }
            // Load CSE: two loads from the same pointer with the same type can be
            // CSE'd if no intervening memory modification occurred. The caller
            // (process_block) handles invalidating Load entries on memory clobbers.
            //
            // Excluded from CSE:
            // - Segment-overridden loads: access thread-local or CPU-local storage
            //   that may differ between accesses even without visible stores
            // - Float, long double, i128 types: use different register paths in
            //   codegen that complicate Copy instruction handling
            // - AtomicLoad: has ordering semantics (falls through to _ => None)
            Instruction::Load { dest, ptr, ty, seg_override } => {
                if *seg_override != AddressSpace::Default {
                    return None;
                }
                if ty.is_float() || ty.is_long_double() || ty.is_128bit() {
                    return None;
                }
                let ptr_vn = self.operand_to_vn(&Operand::Value(*ptr));
                Some((ExprKey::Load { ptr: ptr_vn, ty: *ty }, *dest))
            }
            // Other instructions (Store, Call, AtomicLoad, etc.) are not eligible.
            // AtomicLoad is excluded because it has memory ordering semantics that
            // require the load to actually execute.
            _ => None,
        }
    }

    /// Save the current log positions for later rollback.
    fn save_scope(&self) -> ScopeCheckpoint {
        ScopeCheckpoint {
            rollback_start: self.rollback_log.len(),
            load_rollback_start: self.load_rollback_log.len(),
            store_fwd_rollback_start: self.store_fwd_rollback_log.len(),
            vn_log_start: self.vn_log.len(),
            saved_load_generation: self.load_generation,
        }
    }

    /// Restore state to a previously saved checkpoint, undoing all changes
    /// made since the checkpoint was taken.
    fn restore_scope(&mut self, checkpoint: &ScopeCheckpoint) {
        // Rollback: restore expr_to_value
        while self.rollback_log.len() > checkpoint.rollback_start {
            let (key, old_val) = self.rollback_log.pop()
                .expect("rollback_log length checked by while condition");
            if let Some(val) = old_val {
                self.expr_to_value.insert(key, val);
            } else {
                self.expr_to_value.remove(&key);
            }
        }

        // Rollback: restore load_expr_to_value
        while self.load_rollback_log.len() > checkpoint.load_rollback_start {
            let (key, old_val) = self.load_rollback_log.pop()
                .expect("load_rollback_log length checked by while condition");
            if let Some(val) = old_val {
                self.load_expr_to_value.insert(key, val);
            } else {
                self.load_expr_to_value.remove(&key);
            }
        }

        // Rollback: restore store_fwd_map
        while self.store_fwd_rollback_log.len() > checkpoint.store_fwd_rollback_start {
            let (key, old_val) = self.store_fwd_rollback_log.pop()
                .expect("store_fwd_rollback_log length checked by while condition");
            if let Some(val) = old_val {
                self.store_fwd_map.insert(key, val);
            } else {
                self.store_fwd_map.remove(&key);
            }
        }

        // Rollback: restore value_numbers
        while self.vn_log.len() > checkpoint.vn_log_start {
            let (idx, old_vn) = self.vn_log.pop()
                .expect("vn_log length checked by while condition");
            self.value_numbers[idx] = old_vn;
        }

        // Rollback: restore load_generation
        self.load_generation = checkpoint.saved_load_generation;
    }
}

/// Saved positions in the rollback logs, used by `GvnState::save_scope` /
/// `GvnState::restore_scope` to implement scoped hash table semantics.
struct ScopeCheckpoint {
    rollback_start: usize,
    load_rollback_start: usize,
    store_fwd_rollback_start: usize,
    vn_log_start: usize,
    saved_load_generation: u32,
}

/// Find param allocas whose address has escaped (used in non-Load/Store contexts).
///
/// When a param alloca's address is taken (e.g., via `&x` which becomes a GEP
/// then gets simplified to a Copy), the alloca can be modified through aliased
/// pointers. Store-to-load forwarding must be disabled for these allocas because
/// the backend's ParamRef optimization reads parameter values from the alloca
/// slot at point of use, not at point of definition. If an aliased store
/// modifies the alloca between the original store and the forwarded use, the
/// read will return the wrong value.
fn find_escaped_param_allocas(func: &IrFunction) -> FxHashSet<u32> {
    let param_alloca_set: FxHashSet<u32> = func.param_alloca_values.iter().map(|v| v.0).collect();
    if param_alloca_set.is_empty() {
        return FxHashSet::default();
    }

    let mut escaped = FxHashSet::default();

    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                // Load from param alloca is fine - it reads the value
                Instruction::Load { ptr, .. } => {
                    // ptr is the alloca value itself, this is a normal use
                    let _ = ptr;
                }
                // Store TO param alloca is fine - it writes to the alloca
                Instruction::Store { ptr, val, .. } => {
                    // ptr is the alloca, this is fine
                    let _ = ptr;
                    // But if the alloca value is used as a stored VALUE
                    // (i.e., its address is being stored somewhere), it escapes
                    if let Operand::Value(v) = val {
                        if param_alloca_set.contains(&v.0) {
                            escaped.insert(v.0);
                        }
                    }
                }
                // Copy of param alloca = address taken (e.g., simplified &x GEP)
                Instruction::Copy { src: Operand::Value(v), .. } => {
                    if param_alloca_set.contains(&v.0) {
                        escaped.insert(v.0);
                    }
                }
                // Any other instruction using the alloca value means escape
                _ => {
                    inst.for_each_used_value(|vid| {
                        if param_alloca_set.contains(&vid) {
                            escaped.insert(vid);
                        }
                    });
                }
            }
        }
    }

    escaped
}

/// Run dominator-based GVN on a single function.
pub(crate) fn run_gvn_function(func: &mut IrFunction) -> usize {
    let num_blocks = func.blocks.len();
    if num_blocks == 0 {
        return 0;
    }

    let escaped = find_escaped_param_allocas(func);

    // Fast path for single-block functions: skip CFG/dominator computation.
    if num_blocks == 1 {
        let mut state = GvnState::new(func.max_value_id() as usize, escaped);
        return process_block(0, func, &mut state);
    }

    // Build CFG and dominator tree
    let cfg = analysis::CfgAnalysis::build(func);
    run_gvn_with_analysis(func, &cfg)
}

/// Run GVN using pre-computed CFG analysis (avoids redundant analysis when
/// called from a pipeline that shares analysis across GVN, LICM, IVSR).
pub(crate) fn run_gvn_with_analysis(func: &mut IrFunction, cfg: &analysis::CfgAnalysis) -> usize {
    let num_blocks = func.blocks.len();
    if num_blocks == 0 {
        return 0;
    }

    let escaped = find_escaped_param_allocas(func);

    // Fast path for single-block functions.
    if num_blocks == 1 {
        let mut state = GvnState::new(func.max_value_id() as usize, escaped);
        return process_block(0, func, &mut state);
    }

    let mut state = GvnState::new(func.max_value_id() as usize, escaped);

    // DFS over the dominator tree
    gvn_dfs(0, func, &cfg.dom_children, &cfg.preds, &mut state);

    state.total_eliminated
}

/// Recursive DFS over the dominator tree for GVN.
/// Processes block_idx, then recurses into dominated children.
/// Uses rollback logs to restore state on backtracking.
fn gvn_dfs(
    block_idx: usize,
    func: &mut IrFunction,
    dom_children: &[Vec<usize>],
    preds: &analysis::FlatAdj,
    state: &mut GvnState,
) {
    let checkpoint = state.save_scope();

    // At block entry, decide whether to invalidate inherited Load CSE entries.
    // Load CSE across blocks is safe when this block has exactly one CFG
    // predecessor (straight-line code). At merge points (multiple predecessors),
    // conservatively invalidate all Load entries because a non-dominating
    // predecessor may have stored to memory, making cached loads stale.
    //
    // Invalidation is O(1): just bump load_generation. Entries with older
    // generations are ignored during lookup.
    if block_idx != 0 && preds.len(block_idx) > 1 {
        state.load_generation += 1;
    }

    // Process instructions in this block
    let eliminated = process_block(block_idx, func, state);
    state.total_eliminated += eliminated;

    // Recurse into dominator tree children.
    // Iterate by index to avoid cloning the children Vec.
    let num_children = dom_children[block_idx].len();
    for ci in 0..num_children {
        let child = dom_children[block_idx][ci];
        gvn_dfs(child, func, dom_children, preds, state);
    }

    state.restore_scope(&checkpoint);
}

/// Check if an instruction may modify memory, invalidating cached load values.
/// This is conservative: any instruction that could write to memory or call
/// external code (which could write to memory) returns true.
fn clobbers_memory(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Store { .. }
            | Instruction::Call { .. }
            | Instruction::CallIndirect { .. }
            | Instruction::Memcpy { .. }
            | Instruction::AtomicRmw { .. }
            | Instruction::AtomicCmpxchg { .. }
            | Instruction::AtomicStore { .. }
            | Instruction::Fence { .. }
            | Instruction::InlineAsm { .. }
            | Instruction::VaStart { .. }
            | Instruction::VaEnd { .. }
            | Instruction::VaCopy { .. }
    ) || matches!(inst, Instruction::Intrinsic { dest_ptr: Some(_), .. })
}

/// Check if a Store instruction is eligible for store-to-load forwarding.
/// Same restrictions as Load CSE: no segment overrides, no float/long-double/i128 types.
fn is_forwardable_store(inst: &Instruction) -> bool {
    match inst {
        Instruction::Store { ty, seg_override, .. } => {
            *seg_override == AddressSpace::Default
                && !ty.is_float()
                && !ty.is_long_double()
                && !ty.is_128bit()
        }
        _ => false,
    }
}

/// Process a single basic block for GVN.
/// Returns the number of instructions eliminated.
///
/// Load CSE entries are stored separately in `state.load_expr_to_value`, tagged
/// with a generation counter for O(1) invalidation on memory clobber. Cross-block
/// Load CSE propagation is controlled by `gvn_dfs` which invalidates Load
/// entries at merge points (blocks with multiple CFG predecessors).
///
/// Store-to-load forwarding: when a Store writes value V to pointer P, subsequent
/// Loads from P (same VN, same type, no intervening memory clobber) are replaced
/// with Copy(V). This eliminates redundant loads after stores, a common pattern
/// in struct initialization, local variable access, etc.
fn process_block(
    block_idx: usize,
    func: &mut IrFunction,
    state: &mut GvnState,
) -> usize {
    let mut eliminated = 0;
    let mut new_instructions = Vec::with_capacity(func.blocks[block_idx].instructions.len());
    // GVN replaces instructions 1:1 (original or Copy), so spans stay parallel
    let new_spans = std::mem::take(&mut func.blocks[block_idx].source_spans);

    for inst in func.blocks[block_idx].instructions.drain(..) {
        // Before processing the instruction, check if it clobbers memory.
        // If so, invalidate all cached Load CSE and store forwarding entries
        // by bumping the generation counter. This is O(1) instead of iterating
        // all keys.
        if clobbers_memory(&inst) {
            state.load_generation += 1;
        }

        // Store-to-load forwarding: record stored values for subsequent loads.
        // This happens AFTER the generation bump (since Store itself clobbers memory),
        // so the stored value is recorded at the new generation and will be visible
        // to subsequent loads (which haven't bumped the generation yet).
        //
        // Skip forwarding for stores to escaped param allocas: the backend's
        // ParamRef optimization reads from the alloca slot at point of use, so
        // forwarding a ParamRef through an escaped alloca is unsound when an
        // aliased pointer may write to the same slot.
        if is_forwardable_store(&inst) {
            if let Instruction::Store { val, ptr, ty, .. } = &inst {
                if !state.escaped_param_allocas.contains(&ptr.0) {
                    let ptr_vn = state.operand_to_vn(&Operand::Value(*ptr));
                    let fwd_key = StoreFwdKey { ptr_vn, ty: *ty };
                    let fwd_key_for_log = fwd_key.clone();
                    let old_val = state.store_fwd_map.insert(fwd_key, (*val, state.load_generation));
                    state.store_fwd_rollback_log.push((fwd_key_for_log, old_val));
                }
            }
            // Store has no dest, so no VN to assign. Just keep the instruction.
            new_instructions.push(inst);
            continue;
        }

        match state.make_expr_key(&inst) {
            Some((expr_key, dest)) => {
                let is_load = expr_key.is_load();

                // For loads, first try store-to-load forwarding before load CSE.
                // This catches the pattern: store V -> *P; load *P -> replace with V.
                if is_load {
                    if let ExprKey::Load { ptr: ref ptr_vn, ty } = expr_key {
                        let fwd_key = StoreFwdKey { ptr_vn: ptr_vn.clone(), ty };
                        if let Some((stored_op, gen)) = state.store_fwd_map.get(&fwd_key) {
                            if *gen == state.load_generation {
                                let stored_op = *stored_op;
                                // Forward the stored value to the load destination.
                                // Assign the dest a VN matching the stored value.
                                let dest_idx = dest.0 as usize;
                                let forwarded_vn = match &stored_op {
                                    Operand::Value(v) => {
                                        let idx = v.0 as usize;
                                        if idx < state.value_numbers.len() && state.value_numbers[idx] != u32::MAX {
                                            state.value_numbers[idx]
                                        } else {
                                            state.fresh_vn()
                                        }
                                    }
                                    _ => state.fresh_vn(),
                                };
                                if dest_idx < state.value_numbers.len() {
                                    let old_vn = state.value_numbers[dest_idx];
                                    state.vn_log.push((dest_idx, old_vn));
                                    state.value_numbers[dest_idx] = forwarded_vn;
                                }
                                // Also update load CSE map so subsequent loads from the
                                // same pointer can CSE with this load's dest.
                                let load_key_for_log = expr_key.clone();
                                let old_load = state.load_expr_to_value.insert(
                                    expr_key,
                                    (dest, state.load_generation),
                                );
                                state.load_rollback_log.push((load_key_for_log, old_load));
                                new_instructions.push(Instruction::Copy {
                                    dest,
                                    src: stored_op,
                                });
                                eliminated += 1;
                                continue;
                            }
                        }
                    }
                }

                // Look up: check pure expr map, or load map with generation check
                let existing = if is_load {
                    state.load_expr_to_value.get(&expr_key).and_then(|&(val, gen)| {
                        if gen == state.load_generation { Some(val) } else { None }
                    })
                } else {
                    state.expr_to_value.get(&expr_key).copied()
                };

                if let Some(existing_value) = existing {
                    // This expression was already computed
                    let idx = existing_value.0 as usize;
                    let existing_vn = if idx < state.value_numbers.len() && state.value_numbers[idx] != u32::MAX {
                        state.value_numbers[idx]
                    } else {
                        state.fresh_vn()
                    };
                    let dest_idx = dest.0 as usize;
                    if dest_idx < state.value_numbers.len() {
                        let old_vn = state.value_numbers[dest_idx];
                        state.vn_log.push((dest_idx, old_vn));
                        state.value_numbers[dest_idx] = existing_vn;
                    }
                    new_instructions.push(Instruction::Copy {
                        dest,
                        src: Operand::Value(existing_value),
                    });
                    eliminated += 1;
                } else {
                    // New expression - assign value number and record it
                    let vn = state.fresh_vn();
                    let dest_idx = dest.0 as usize;
                    if dest_idx < state.value_numbers.len() {
                        let old_vn = state.value_numbers[dest_idx];
                        state.vn_log.push((dest_idx, old_vn));
                        state.value_numbers[dest_idx] = vn;
                    }
                    // Record in appropriate map with rollback
                    if is_load {
                        let key_for_log = expr_key.clone();
                        let old_val = state.load_expr_to_value.insert(expr_key, (dest, state.load_generation));
                        state.load_rollback_log.push((key_for_log, old_val));
                    } else {
                        let key_for_log = expr_key.clone();
                        let old_val = state.expr_to_value.insert(expr_key, dest);
                        state.rollback_log.push((key_for_log, old_val));
                    }
                    new_instructions.push(inst);
                }
            }
            None => {
                // Not a numberable expression (store, call, alloca, etc.)
                if let Some(dest) = inst.dest() {
                    state.assign_fresh_vn(dest);
                }
                new_instructions.push(inst);
            }
        }
    }

    func.blocks[block_idx].instructions = new_instructions;
    func.blocks[block_idx].source_spans = new_spans;
    eliminated
}

/// Canonicalize operand order for commutative operations.
/// Ensures (a + b) and (b + a) hash to the same key.
fn canonical_order(lhs: VNOperand, rhs: VNOperand) -> (VNOperand, VNOperand) {
    if should_swap(&lhs, &rhs) {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn should_swap(lhs: &VNOperand, rhs: &VNOperand) -> bool {
    match (lhs, rhs) {
        (VNOperand::ValueNum(_), VNOperand::Const(_)) => true,
        (VNOperand::ValueNum(a), VNOperand::ValueNum(b)) => a > b,
        (VNOperand::Const(a), VNOperand::Const(b)) => a > b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::reexports::{BasicBlock, BlockId, CallInfo, IrConst, IrModule, Terminator};

    #[test]
    fn test_commutative_cse() {
        // Test that a + b and b + a are recognized as the same expression
        let block = BasicBlock {
            label: BlockId(0),
            instructions: vec![
                // %0 = add %a, %b
                Instruction::BinOp {
                    dest: Value(2),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(0)),
                    rhs: Operand::Value(Value(1)),
                    ty: IrType::I32,
                },
                // %1 = add %b, %a  (same expression, reversed operands)
                Instruction::BinOp {
                    dest: Value(3),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Value(Value(0)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
            source_spans: Vec::new(),
        };

        let func = IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::I32,
            blocks: vec![block],
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 4,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        };

        let mut module = IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        };

        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        // Second instruction should be a Copy
        match &module.functions[0].blocks[0].instructions[1] {
            Instruction::Copy { dest, src: Operand::Value(v) } => {
                assert_eq!(dest.0, 3);
                assert_eq!(v.0, 2);
            }
            other => panic!("Expected Copy instruction, got {:?}", other),
        }
    }

    #[test]
    fn test_non_commutative_not_cse() {
        // Test that a - b and b - a are NOT treated as the same
        let func = IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::I32,
            blocks: vec![BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::BinOp {
                        dest: Value(2),
                        op: IrBinOp::Sub,
                        lhs: Operand::Value(Value(0)),
                        rhs: Operand::Value(Value(1)),
                        ty: IrType::I32,
                    },
                    Instruction::BinOp {
                        dest: Value(3),
                        op: IrBinOp::Sub,
                        lhs: Operand::Value(Value(1)),
                        rhs: Operand::Value(Value(0)),
                        ty: IrType::I32,
                    },
                ],
                terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
                source_spans: Vec::new(),
            }],
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 4,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        };

        let mut module = IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        };

        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 0);
    }

    #[test]
    fn test_constant_cse() {
        // Two identical constant expressions should be CSE'd
        let func = IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::I32,
            blocks: vec![BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::BinOp {
                        dest: Value(0),
                        op: IrBinOp::Add,
                        lhs: Operand::Const(IrConst::I32(3)),
                        rhs: Operand::Const(IrConst::I32(4)),
                        ty: IrType::I32,
                    },
                    Instruction::BinOp {
                        dest: Value(1),
                        op: IrBinOp::Add,
                        lhs: Operand::Const(IrConst::I32(3)),
                        rhs: Operand::Const(IrConst::I32(4)),
                        ty: IrType::I32,
                    },
                ],
                terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
                source_spans: Vec::new(),
            }],
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 2,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        };

        let mut module = IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        };

        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);
    }

    #[test]
    fn test_is_commutative() {
        assert!(IrBinOp::Add.is_commutative());
        assert!(IrBinOp::Mul.is_commutative());
        assert!(!IrBinOp::Sub.is_commutative());
        assert!(!IrBinOp::SDiv.is_commutative());
    }

    #[test]
    fn test_cast_cse() {
        // Two identical casts should be CSE'd
        let func = IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::I64,
            blocks: vec![BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Cast {
                        dest: Value(1),
                        src: Operand::Value(Value(0)),
                        from_ty: IrType::I32,
                        to_ty: IrType::I64,
                    },
                    Instruction::Cast {
                        dest: Value(2),
                        src: Operand::Value(Value(0)),
                        from_ty: IrType::I32,
                        to_ty: IrType::I64,
                    },
                ],
                terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
                source_spans: Vec::new(),
            }],
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 3,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        };

        let mut module = IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        };

        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        match &module.functions[0].blocks[0].instructions[1] {
            Instruction::Copy { dest, src: Operand::Value(v) } => {
                assert_eq!(dest.0, 2);
                assert_eq!(v.0, 1);
            }
            other => panic!("Expected Copy instruction, got {:?}", other),
        }
    }

    #[test]
    fn test_gep_cse() {
        // Two identical GEPs should be CSE'd
        let func = IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::Ptr,
            blocks: vec![BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::GetElementPtr {
                        dest: Value(2),
                        base: Value(0),
                        offset: Operand::Value(Value(1)),
                        ty: IrType::Ptr,
                    },
                    Instruction::GetElementPtr {
                        dest: Value(3),
                        base: Value(0),
                        offset: Operand::Value(Value(1)),
                        ty: IrType::Ptr,
                    },
                ],
                terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
                source_spans: Vec::new(),
            }],
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 4,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        };

        let mut module = IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        };

        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);
    }

    #[test]
    fn test_cross_block_cse() {
        // Test that expressions in dominating blocks are visible to dominated blocks
        // CFG: block0 -> block1 (block0 dominates block1)
        let func = IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::I32,
            blocks: vec![
                BasicBlock {
                    label: BlockId(0),
                    instructions: vec![
                        Instruction::BinOp {
                            dest: Value(2),
                            op: IrBinOp::Add,
                            lhs: Operand::Value(Value(0)),
                            rhs: Operand::Value(Value(1)),
                            ty: IrType::I32,
                        },
                    ],
                    terminator: Terminator::Branch(BlockId(1)),
                    source_spans: Vec::new(),
                },
                BasicBlock {
                    label: BlockId(1),
                    instructions: vec![
                        // Same expression as in block0 - should be CSE'd
                        Instruction::BinOp {
                            dest: Value(3),
                            op: IrBinOp::Add,
                            lhs: Operand::Value(Value(0)),
                            rhs: Operand::Value(Value(1)),
                            ty: IrType::I32,
                        },
                    ],
                    terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
                    source_spans: Vec::new(),
                },
            ],
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 4,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        };

        let mut module = IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        };

        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        // The expression in block1 should be replaced with a Copy
        match &module.functions[0].blocks[1].instructions[0] {
            Instruction::Copy { dest, src: Operand::Value(v) } => {
                assert_eq!(dest.0, 3);
                assert_eq!(v.0, 2);
            }
            other => panic!("Expected Copy instruction, got {:?}", other),
        }
    }

    #[test]
    fn test_diamond_no_cse_between_branches() {
        // Diamond CFG: block0 -> {block1, block2} -> block3
        // Expressions in block1 and block2 should NOT be CSE'd with each other,
        // since neither dominates the other.
        let func = IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::I32,
            blocks: vec![
                // block0: entry, branches to block1 or block2
                BasicBlock {
                    label: BlockId(0),
                    instructions: vec![],
                    terminator: Terminator::CondBranch {
                        cond: Operand::Value(Value(0)),
                        true_label: BlockId(1),
                        false_label: BlockId(2),
                    },
                    source_spans: Vec::new(),
                },
                // block1: compute add (only reached via true branch)
                BasicBlock {
                    label: BlockId(1),
                    instructions: vec![
                        Instruction::BinOp {
                            dest: Value(2),
                            op: IrBinOp::Add,
                            lhs: Operand::Value(Value(0)),
                            rhs: Operand::Const(IrConst::I32(1)),
                            ty: IrType::I32,
                        },
                    ],
                    terminator: Terminator::Branch(BlockId(3)),
                    source_spans: Vec::new(),
                },
                // block2: compute same add (only reached via false branch)
                BasicBlock {
                    label: BlockId(2),
                    instructions: vec![
                        Instruction::BinOp {
                            dest: Value(3),
                            op: IrBinOp::Add,
                            lhs: Operand::Value(Value(0)),
                            rhs: Operand::Const(IrConst::I32(1)),
                            ty: IrType::I32,
                        },
                    ],
                    terminator: Terminator::Branch(BlockId(3)),
                    source_spans: Vec::new(),
                },
                // block3: merge
                BasicBlock {
                    label: BlockId(3),
                    instructions: vec![],
                    terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
                    source_spans: Vec::new(),
                },
            ],
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id: 4,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        };

        let mut module = IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        };

        let eliminated = module.for_each_function(run_gvn_function);
        // Neither branch dominates the other, so NO CSE should happen
        assert_eq!(eliminated, 0);

        // Both blocks should still have their original BinOp instructions
        assert!(matches!(
            &module.functions[0].blocks[1].instructions[0],
            Instruction::BinOp { op: IrBinOp::Add, .. }
        ));
        assert!(matches!(
            &module.functions[0].blocks[2].instructions[0],
            Instruction::BinOp { op: IrBinOp::Add, .. }
        ));
    }

    /// Helper to create a minimal IrFunction with given blocks.
    fn make_func(blocks: Vec<BasicBlock>, next_value_id: u32) -> IrFunction {
        IrFunction {
            name: "test".into(),
            params: vec![],
            return_type: IrType::I32,
            blocks,
            is_variadic: false,
            is_fastcall: false,
            is_naked: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_declaration: false,
            next_value_id,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        }
    }

    fn make_module(func: IrFunction) -> IrModule {
        IrModule {
            functions: vec![func],
            globals: vec![],
            string_literals: vec![],
            wide_string_literals: vec![],
            constructors: vec![],
            destructors: vec![],
            aliases: vec![],
            toplevel_asm: vec![],
            symbol_attrs: vec![],
            char16_string_literals: vec![],
            symver_directives: vec![],
        }
    }

    #[test]
    fn test_load_cse_same_block() {
        // Two loads from the same pointer in the same block should be CSE'd
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Load {
                    dest: Value(1),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Load {
                    dest: Value(2),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        }], 3);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        // Second load should be replaced with Copy
        match &module.functions[0].blocks[0].instructions[1] {
            Instruction::Copy { dest, src: Operand::Value(v) } => {
                assert_eq!(dest.0, 2);
                assert_eq!(v.0, 1);
            }
            other => panic!("Expected Copy, got {:?}", other),
        }
    }

    #[test]
    fn test_load_cse_invalidated_by_store() {
        // A store between two loads should prevent CSE
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Load {
                    dest: Value(1),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(42)),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Load {
                    dest: Value(2),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        }], 3);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        // The store invalidates the first load's CSE entry, but store-to-load
        // forwarding can replace the second load with the stored value (42).
        assert_eq!(eliminated, 1);
    }

    #[test]
    fn test_load_cse_invalidated_by_call() {
        // A call between two loads should prevent CSE
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Load {
                    dest: Value(1),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Call {
                    func: "side_effect".into(),
                    info: CallInfo {
                        dest: Some(Value(2)),
                        args: vec![],
                        arg_types: vec![],
                        return_type: IrType::Void,
                        is_variadic: false,
                        num_fixed_args: 0,
                        struct_arg_sizes: vec![],
                        struct_arg_aligns: vec![],
                        struct_arg_classes: vec![],
                        struct_arg_riscv_float_classes: Vec::new(),
                        is_sret: false,
                        is_fastcall: false,
                        ret_eightbyte_classes: Vec::new(),
                    },
                },
                Instruction::Load {
                    dest: Value(3),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
            source_spans: Vec::new(),
        }], 4);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 0); // No CSE: call may modify memory
    }

    #[test]
    fn test_load_cse_across_dominating_block() {
        // Load in block0 should CSE with load in block1 (block0 dominates block1,
        // single predecessor, no memory clobber)
        let func = make_func(vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Load {
                        dest: Value(1),
                        ptr: Value(0),
                        ty: IrType::I32,
                        seg_override: AddressSpace::Default,
                    },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: Vec::new(),
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![
                    Instruction::Load {
                        dest: Value(2),
                        ptr: Value(0),
                        ty: IrType::I32,
                        seg_override: AddressSpace::Default,
                    },
                ],
                terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
                source_spans: Vec::new(),
            },
        ], 3);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        match &module.functions[0].blocks[1].instructions[0] {
            Instruction::Copy { dest, src: Operand::Value(v) } => {
                assert_eq!(dest.0, 2);
                assert_eq!(v.0, 1);
            }
            other => panic!("Expected Copy, got {:?}", other),
        }
    }

    #[test]
    fn test_load_cse_invalidated_at_merge_point() {
        // Diamond CFG: block0 -> {block1, block2} -> block3
        // block1 stores to memory, so Load CSE should be invalidated at block3
        let func = make_func(vec![
            // block0: entry, load and branch
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Load {
                        dest: Value(2),
                        ptr: Value(0),
                        ty: IrType::I32,
                        seg_override: AddressSpace::Default,
                    },
                ],
                terminator: Terminator::CondBranch {
                    cond: Operand::Value(Value(1)),
                    true_label: BlockId(1),
                    false_label: BlockId(2),
                },
                source_spans: Vec::new(),
            },
            // block1: stores to memory
            BasicBlock {
                label: BlockId(1),
                instructions: vec![
                    Instruction::Store {
                        val: Operand::Const(IrConst::I32(42)),
                        ptr: Value(0),
                        ty: IrType::I32,
                        seg_override: AddressSpace::Default,
                    },
                ],
                terminator: Terminator::Branch(BlockId(3)),
                source_spans: Vec::new(),
            },
            // block2: no memory modification
            BasicBlock {
                label: BlockId(2),
                instructions: vec![],
                terminator: Terminator::Branch(BlockId(3)),
                source_spans: Vec::new(),
            },
            // block3: merge point - loads from same pointer
            BasicBlock {
                label: BlockId(3),
                instructions: vec![
                    Instruction::Load {
                        dest: Value(3),
                        ptr: Value(0),
                        ty: IrType::I32,
                        seg_override: AddressSpace::Default,
                    },
                ],
                terminator: Terminator::Return(Some(Operand::Value(Value(3)))),
                source_spans: Vec::new(),
            },
        ], 4);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        // Load in block3 should NOT be CSE'd because block3 is a merge point
        // and block1 (a predecessor) stores to memory.
        assert_eq!(eliminated, 0);

        // block3's load should remain as-is
        assert!(matches!(
            &module.functions[0].blocks[3].instructions[0],
            Instruction::Load { .. }
        ));
    }

    #[test]
    fn test_store_to_load_forwarding_same_block() {
        // store 42 -> *ptr; load *ptr => should be forwarded to Copy(42)
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(42)),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Load {
                    dest: Value(1),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        }], 2);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        // The load should be replaced with a Copy of the stored constant
        match &module.functions[0].blocks[0].instructions[1] {
            Instruction::Copy { dest, src: Operand::Const(IrConst::I32(42)) } => {
                assert_eq!(dest.0, 1);
            }
            other => panic!("Expected Copy of constant 42, got {:?}", other),
        }
    }

    #[test]
    fn test_store_to_load_forwarding_value() {
        // store %v -> *ptr; load *ptr => should be forwarded to Copy(%v)
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Store {
                    val: Operand::Value(Value(1)),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Load {
                    dest: Value(2),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        }], 3);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        match &module.functions[0].blocks[0].instructions[1] {
            Instruction::Copy { dest, src: Operand::Value(v) } => {
                assert_eq!(dest.0, 2);
                assert_eq!(v.0, 1);
            }
            other => panic!("Expected Copy of Value(1), got {:?}", other),
        }
    }

    #[test]
    fn test_store_to_load_forwarding_invalidated_by_call() {
        // store 42 -> *ptr; call foo(); load *ptr => NOT forwarded (call may modify memory)
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(42)),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Call {
                    func: "foo".into(),
                    info: CallInfo {
                        dest: Some(Value(1)),
                        args: vec![],
                        arg_types: vec![],
                        return_type: IrType::Void,
                        is_variadic: false,
                        num_fixed_args: 0,
                        struct_arg_sizes: vec![],
                        struct_arg_aligns: vec![],
                        struct_arg_classes: vec![],
                        struct_arg_riscv_float_classes: Vec::new(),
                        is_sret: false,
                        is_fastcall: false,
                        ret_eightbyte_classes: Vec::new(),
                    },
                },
                Instruction::Load {
                    dest: Value(2),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        }], 3);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 0); // No forwarding: call invalidates the store
    }

    #[test]
    fn test_store_to_load_forwarding_different_store_invalidates() {
        // store 42 -> *ptr_a; store 99 -> *ptr_b; load *ptr_a
        // => NOT forwarded because the second store (to any address) invalidates all
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(42)),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(99)),
                    ptr: Value(1),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Load {
                    dest: Value(2),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: Vec::new(),
        }], 3);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        // The second store bumps the generation, invalidating the first store's entry.
        // However, the second store also records *ptr_b -> 99 at the new generation.
        // The load from *ptr_a should NOT be forwarded because *ptr_a != *ptr_b.
        assert_eq!(eliminated, 0);
    }

    #[test]
    fn test_store_to_load_forwarding_same_ptr_overwrite() {
        // store 42 -> *ptr; store 99 -> *ptr; load *ptr => forwarded to 99
        let func = make_func(vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(42)),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Store {
                    val: Operand::Const(IrConst::I32(99)),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
                Instruction::Load {
                    dest: Value(1),
                    ptr: Value(0),
                    ty: IrType::I32,
                    seg_override: AddressSpace::Default,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        }], 2);

        let mut module = make_module(func);
        let eliminated = module.for_each_function(run_gvn_function);
        assert_eq!(eliminated, 1);

        // Should forward the SECOND store's value (99), not the first (42)
        match &module.functions[0].blocks[0].instructions[2] {
            Instruction::Copy { dest, src: Operand::Const(IrConst::I32(99)) } => {
                assert_eq!(dest.0, 1);
            }
            other => panic!("Expected Copy of constant 99, got {:?}", other),
        }
    }
}
