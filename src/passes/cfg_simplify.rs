//! CFG Simplification pass.
//!
//! Simplifies the control flow graph by:
//! 1. Folding `CondBranch` with a known-constant condition to `Branch`
//! 2. Folding `Switch` with a known-constant value to `Branch` (matching case or default)
//! 3. Converting `CondBranch` where both targets are the same to `Branch`
//! 4. Threading jump chains: if block A branches to empty block B which just
//!    branches to C, redirect A to branch directly to C (only when safe)
//! 5. Removing dead (unreachable) blocks that have no predecessors
//! 6. Simplifying trivial phi nodes (single-entry or all-same-value) to Copy
//! 7. Merging single-predecessor blocks into their predecessor
//!
//! This pass runs to a fixpoint, since one simplification can enable others.
//! Phi nodes in successor blocks are updated when edges are redirected.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::reexports::{
    BasicBlock,
    BlockId,
    Instruction,
    IrConst,
    IrFunction,
    Operand,
    Terminator,
    Value,
};

/// Maximum depth for resolving transitive jump chains (A→B→C→...),
/// to prevent pathological cases.
const MAX_CHAIN_DEPTH: u32 = 32;

/// Maximum depth for walking single-predecessor chains when resolving
/// a value to a constant across blocks.
const MAX_PRED_CHAIN_DEPTH: usize = 8;

/// Maximum recursion depth for global cross-block value resolution.
const MAX_GLOBAL_RESOLVE_DEPTH: usize = 16;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Per-function entry point for dirty-tracking pipeline.
pub(crate) fn run_function(func: &mut IrFunction) -> usize {
    simplify_cfg(func)
}

/// Simplify the CFG of a single function.
/// Iterates until no more simplifications are possible (fixpoint).
///
/// Builds the `label_to_idx` map once per fixpoint iteration and shares it
/// across sub-passes that need block lookups, avoiding redundant HashMap
/// construction.
pub(crate) fn simplify_cfg(func: &mut IrFunction) -> usize {
    if func.blocks.len() <= 1 {
        return 0;
    }

    let mut total = 0;
    loop {
        let mut changed = 0;
        // Build the label-to-index map once per fixpoint iteration.
        // Sub-passes that modify block structure (remove_dead_blocks,
        // merge_single_pred_blocks) invalidate this map, but they run last.
        let label_to_idx = build_label_to_idx(func);
        changed += fold_constant_cond_branches(func, &label_to_idx);
        changed += fold_constant_switches(func, &label_to_idx);
        changed += simplify_redundant_cond_branches(func);
        changed += thread_jump_chains(func, &label_to_idx);
        changed += remove_dead_blocks(func, &label_to_idx);
        changed += simplify_trivial_phis(func);
        changed += merge_single_pred_blocks(func);
        if changed == 0 {
            break;
        }
        total += changed;
    }
    total
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build a map from BlockId -> index in func.blocks for O(1) lookup.
#[inline]
fn build_label_to_idx(func: &IrFunction) -> FxHashMap<BlockId, usize> {
    func.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect()
}

/// Build predecessor count for each block.
///
/// Counts edges from terminators and from InlineAsm goto_labels (which are
/// implicit control flow edges). Also returns a single-predecessor map:
/// `single_pred[B] = A` iff A is B's only predecessor.
fn build_pred_info(func: &IrFunction) -> (FxHashMap<BlockId, u32>, FxHashMap<BlockId, BlockId>) {
    let mut pred_count: FxHashMap<BlockId, u32> = FxHashMap::default();
    let mut single_pred: FxHashMap<BlockId, BlockId> = FxHashMap::default();

    for block in func.blocks.iter() {
        let src = block.label;
        let mut add_edge = |target: BlockId| {
            let count = pred_count.entry(target).or_insert(0);
            *count += 1;
            if *count == 1 {
                single_pred.insert(target, src);
            } else {
                single_pred.remove(&target);
            }
        };
        for_each_terminator_target(&block.terminator, &mut add_edge);
        // InlineAsm goto_labels are implicit control flow edges.
        for inst in &block.instructions {
            if let Instruction::InlineAsm { goto_labels, .. } = inst {
                for (_, label) in goto_labels {
                    add_edge(*label);
                }
            }
        }
    }
    (pred_count, single_pred)
}

/// Remove phi entries that reference `source_label` from all phi nodes in `block`.
fn remove_phi_entries_from(block: &mut BasicBlock, source_label: BlockId) {
    for inst in &mut block.instructions {
        if let Instruction::Phi { incoming, .. } = inst {
            incoming.retain(|(_, label)| *label != source_label);
        }
    }
}

/// Remove phi entries from a block where the source label is in `dead_labels`.
fn remove_phi_entries_from_set(block: &mut BasicBlock, dead_labels: &FxHashSet<BlockId>) {
    for inst in &mut block.instructions {
        if let Instruction::Phi { incoming, .. } = inst {
            incoming.retain(|(_, label)| !dead_labels.contains(label));
        }
    }
}

/// Remap terminator targets using a label substitution map.
/// For each target in the terminator, if `remap[target]` exists, replace it.
fn remap_terminator_targets(term: &mut Terminator, remap: &FxHashMap<BlockId, BlockId>) {
    match term {
        Terminator::Branch(target) => {
            if let Some(&new) = remap.get(target) {
                *target = new;
            }
        }
        Terminator::CondBranch { true_label, false_label, .. } => {
            if let Some(&new) = remap.get(true_label) {
                *true_label = new;
            }
            if let Some(&new) = remap.get(false_label) {
                *false_label = new;
            }
        }
        Terminator::Switch { cases, default, .. } => {
            if let Some(&new) = remap.get(default) {
                *default = new;
            }
            for (_, label) in cases.iter_mut() {
                if let Some(&new) = remap.get(label) {
                    *label = new;
                }
            }
        }
        Terminator::IndirectBranch { possible_targets, .. } => {
            for target in possible_targets.iter_mut() {
                if let Some(&new) = remap.get(target) {
                    *target = new;
                }
            }
        }
        _ => {}
    }
}

/// Visit each branch target of a terminator, calling `f` for each target.
/// Switch targets are deduplicated against the default to avoid double-counting.
#[inline]
fn for_each_terminator_target(term: &Terminator, mut f: impl FnMut(BlockId)) {
    match term {
        Terminator::Branch(target) => f(*target),
        Terminator::CondBranch { true_label, false_label, .. } => {
            f(*true_label);
            f(*false_label);
        }
        Terminator::IndirectBranch { possible_targets, .. } => {
            for &target in possible_targets {
                f(target);
            }
        }
        Terminator::Switch { cases, default, .. } => {
            f(*default);
            for &(_, label) in cases {
                if label != *default {
                    f(label);
                }
            }
        }
        Terminator::Return(_) | Terminator::Unreachable => {}
    }
}

// ---------------------------------------------------------------------------
// Sub-pass: redundant conditional branches
// ---------------------------------------------------------------------------

/// Convert `CondBranch { cond, true_label: X, false_label: X }` to `Branch(X)`.
/// The condition is dead and will be cleaned up by DCE.
fn simplify_redundant_cond_branches(func: &mut IrFunction) -> usize {
    let mut count = 0;
    for block in &mut func.blocks {
        if let Terminator::CondBranch { true_label, false_label, .. } = &block.terminator {
            if true_label == false_label {
                let target = *true_label;
                block.terminator = Terminator::Branch(target);
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Sub-pass: constant branch folding
// ---------------------------------------------------------------------------

/// Fold `CondBranch` with a known-constant condition into an unconditional `Branch`.
///
/// After constant folding + copy propagation, a CondBranch may have a constant
/// condition (e.g., `CondBranch { cond: Const(1), true_label, false_label }`).
/// This arises in switch(sizeof(T)) patterns where the dispatch comparisons
/// fold to constants. Converting these to unconditional branches enables dead
/// block removal to eliminate the unreachable switch cases.
///
/// When folding, we must clean up phi nodes in the not-taken target block:
/// the phi entries referencing the current block must be removed since the edge
/// no longer exists. Without this cleanup, stale phi entries can cause
/// miscompilation when the not-taken block is still reachable from other paths.
fn fold_constant_cond_branches(func: &mut IrFunction, label_to_idx: &FxHashMap<BlockId, usize>) -> usize {
    // Build predecessor info for cross-block value resolution.
    let (_pred_count, single_pred) = build_pred_info(func);

    // Build a global value definition map for cross-block resolution.
    let global_val_map = build_global_value_map(func);

    // First pass: collect the folding decisions.
    // Each entry: (block_index, taken_target, not_taken_target, block_label)
    let mut folds: Vec<(usize, BlockId, BlockId, BlockId)> = Vec::new();

    for (idx, block) in func.blocks.iter().enumerate() {
        if let Terminator::CondBranch { cond, true_label, false_label } = &block.terminator {
            let const_val = match cond {
                Operand::Const(c) => Some(c.is_nonzero()),
                Operand::Value(v) => {
                    resolve_cond_branch_value(func, block, *v, &single_pred, label_to_idx, &global_val_map)
                        .map(|c| c.is_nonzero())
                }
            };
            if let Some(is_true) = const_val {
                let taken = if is_true { *true_label } else { *false_label };
                let not_taken = if is_true { *false_label } else { *true_label };
                folds.push((idx, taken, not_taken, block.label));
            }
        }
    }

    if folds.is_empty() {
        return 0;
    }

    let count = folds.len();

    // Apply the folds and clean up phi nodes.
    for &(idx, taken, not_taken, block_label) in &folds {
        func.blocks[idx].terminator = Terminator::Branch(taken);
        // Remove phi entries in the not-taken target (if it differs from taken).
        if taken != not_taken {
            if let Some(&block_idx) = label_to_idx.get(&not_taken) {
                remove_phi_entries_from(&mut func.blocks[block_idx], block_label);
            }
        }
    }

    count
}

/// Try to resolve a CondBranch condition value to a constant.
///
/// Walks through Copy/Phi/Cmp/Select chains in the current block, then up the
/// single-predecessor chain, then globally across blocks.
fn resolve_cond_branch_value(
    func: &IrFunction,
    block: &BasicBlock,
    v: Value,
    single_pred: &FxHashMap<BlockId, BlockId>,
    label_to_idx: &FxHashMap<BlockId, usize>,
    global_val_map: &FxHashMap<Value, (usize, usize)>,
) -> Option<IrConst> {
    // Try local resolution first.
    if let Some(c) = resolve_value_to_const_in_block(block, v) {
        return Some(c);
    }

    // Walk the single-predecessor chain (blocks with exactly one predecessor
    // that unconditionally branches here).
    let mut current_label = block.label;
    for _ in 0..MAX_PRED_CHAIN_DEPTH {
        let pred_label = *single_pred.get(&current_label)?;
        let &pred_idx = label_to_idx.get(&pred_label)?;
        let pred_block = &func.blocks[pred_idx];
        if !matches!(&pred_block.terminator, Terminator::Branch(t) if *t == current_label) {
            break;
        }
        if let Some(c) = resolve_value_to_const_in_block(pred_block, v) {
            return Some(c);
        }
        current_label = pred_label;
    }

    // Global cross-block resolution (needed for kernel's
    // `alternative_has_cap_unlikely()` pattern).
    resolve_value_globally(func, v, global_val_map, 0)
}

// ---------------------------------------------------------------------------
// Sub-pass: constant switch folding
// ---------------------------------------------------------------------------

/// Fold `Switch` with a known-constant value into an unconditional `Branch`.
///
/// After inlining + constant folding + copy propagation, a Switch may have a
/// constant value. This is critical for the Linux kernel's `cpucap_is_possible()`
/// pattern: a switch on a capability number that should resolve at compile time.
///
/// When folding, we clean up phi nodes in all not-taken target blocks.
fn fold_constant_switches(func: &mut IrFunction, label_to_idx: &FxHashMap<BlockId, usize>) -> usize {
    // Collect folding decisions: (block_index, taken_target, not_taken_targets, block_label)
    let mut folds: Vec<(usize, BlockId, Vec<BlockId>, BlockId)> = Vec::new();

    for (idx, block) in func.blocks.iter().enumerate() {
        if let Terminator::Switch { val, cases, default, .. } = &block.terminator {
            let resolved_const = match val {
                Operand::Const(c) => Some(*c),
                Operand::Value(v) => resolve_value_to_const_in_block(block, *v),
            };
            if let Some(c) = resolved_const {
                if let Some(switch_int) = c.to_i64() {
                    let taken = cases.iter()
                        .find(|(cv, _)| *cv == switch_int)
                        .map(|(_, label)| *label)
                        .unwrap_or(*default);

                    // Collect unique not-taken targets.
                    let mut not_taken = Vec::new();
                    if *default != taken && !not_taken.contains(default) {
                        not_taken.push(*default);
                    }
                    for (_, label) in cases {
                        if *label != taken && !not_taken.contains(label) {
                            not_taken.push(*label);
                        }
                    }

                    folds.push((idx, taken, not_taken, block.label));
                }
            }
        }
    }

    if folds.is_empty() {
        return 0;
    }

    let count = folds.len();

    // Apply folds and clean up phi nodes.
    for (idx, taken, ref not_taken, block_label) in &folds {
        func.blocks[*idx].terminator = Terminator::Branch(*taken);
        for nt_target in not_taken {
            if let Some(&block_idx) = label_to_idx.get(nt_target) {
                remove_phi_entries_from(&mut func.blocks[block_idx], *block_label);
            }
        }
    }

    count
}

// ---------------------------------------------------------------------------
// Sub-pass: jump chain threading
// ---------------------------------------------------------------------------

/// Thread jump chains: if a block branches to an empty forwarding block
/// (no instructions, terminates with unconditional Branch), redirect to
/// skip the intermediate block.
///
/// Special care: when threading would cause a CondBranch's true and false
/// targets to become the same block (both edges merge), AND the target has
/// phi nodes that carry different values from the two paths, we must NOT
/// thread. Otherwise the merge block's phi loses the ability to distinguish
/// the two control flow paths, causing miscompilation.
fn thread_jump_chains(func: &mut IrFunction, label_to_idx: &FxHashMap<BlockId, usize>) -> usize {
    // Build forwarding map: empty blocks that just branch unconditionally.
    let forwarding: FxHashMap<BlockId, BlockId> = func.blocks.iter()
        .filter(|block| {
            block.instructions.is_empty()
                && matches!(&block.terminator, Terminator::Branch(_))
        })
        .map(|block| {
            if let Terminator::Branch(target) = &block.terminator {
                (block.label, *target)
            } else {
                unreachable!("block was filtered to have Branch terminator")
            }
        })
        .collect();

    if forwarding.is_empty() {
        return 0;
    }

    // Resolve transitive chains with cycle detection.
    // resolved[B] = (final_target, immediate_predecessor_of_final)
    let resolved = resolve_forwarding_chains(&forwarding);
    if resolved.is_empty() {
        return 0;
    }

    // Collect the redirections we need to make.
    let redirections = collect_thread_redirections(func, &resolved, label_to_idx);
    if redirections.is_empty() {
        return 0;
    }

    // Apply the redirections.
    apply_thread_redirections(func, &redirections, label_to_idx)
}

/// Resolve transitive forwarding chains.
/// Returns a map: start_block -> (final_target, immediate_pred_of_final).
fn resolve_forwarding_chains(forwarding: &FxHashMap<BlockId, BlockId>) -> FxHashMap<BlockId, (BlockId, BlockId)> {
    let mut resolved = FxHashMap::default();
    for &start in forwarding.keys() {
        let mut prev = start;
        let mut current = start;
        let mut depth = 0;
        while let Some(&next) = forwarding.get(&current) {
            if next == start || depth > MAX_CHAIN_DEPTH {
                break;
            }
            prev = current;
            current = next;
            depth += 1;
        }
        if current != start {
            resolved.insert(start, (current, prev));
        }
    }
    resolved
}

/// Collect edge redirections for jump chain threading.
/// Returns: Vec<(block_idx, Vec<(old_target, new_target, phi_lookup_block)>)>
fn collect_thread_redirections(
    func: &IrFunction,
    resolved: &FxHashMap<BlockId, (BlockId, BlockId)>,
    label_to_idx: &FxHashMap<BlockId, usize>,
) -> Vec<(usize, Vec<(BlockId, BlockId, BlockId)>)> {
    let mut redirections = Vec::new();

    for block_idx in 0..func.blocks.len() {
        let mut edge_changes = Vec::new();
        let block_label = func.blocks[block_idx].label;

        match &func.blocks[block_idx].terminator {
            Terminator::Branch(target) => {
                if let Some(&(resolved_target, phi_block)) = resolved.get(target) {
                    edge_changes.push((*target, resolved_target, phi_block));
                }
            }
            Terminator::CondBranch { true_label, false_label, .. } => {
                let true_resolved = resolved.get(true_label).copied();
                let false_resolved = resolved.get(false_label).copied();
                let true_final = true_resolved.map(|(t, _)| t).unwrap_or(*true_label);
                let false_final = false_resolved.map(|(t, _)| t).unwrap_or(*false_label);

                if true_final == false_final && true_label != false_label {
                    // Threading would merge both edges — check for phi conflict.
                    if would_create_phi_conflict(func, block_label, *true_label,
                                                  *false_label, true_final, resolved,
                                                  label_to_idx) {
                        continue; // Skip: phi conflict
                    }
                }
                // Safe to thread (different final targets, or no phi conflict).
                if let Some((rt, rt_phi)) = true_resolved {
                    edge_changes.push((*true_label, rt, rt_phi));
                }
                if let Some((rf, rf_phi)) = false_resolved {
                    if !edge_changes.iter().any(|(old, new, _)| *old == *false_label && *new == rf) {
                        edge_changes.push((*false_label, rf, rf_phi));
                    }
                }
            }
            // TODO: IndirectBranch targets could also be threaded through
            // empty blocks, but computed goto is rare enough to skip for now.
            _ => {}
        }

        if !edge_changes.is_empty() {
            redirections.push((block_idx, edge_changes));
        }
    }

    redirections
}

/// Apply collected thread redirections: update terminators and phi nodes.
fn apply_thread_redirections(
    func: &mut IrFunction,
    redirections: &[(usize, Vec<(BlockId, BlockId, BlockId)>)],
    label_to_idx: &FxHashMap<BlockId, usize>,
) -> usize {
    let mut count = 0;
    for (block_idx, edge_changes) in redirections {
        let block_label = func.blocks[*block_idx].label;

        // Update the terminator.
        match &mut func.blocks[*block_idx].terminator {
            Terminator::Branch(target) => {
                for (old, new, _) in edge_changes {
                    if target == old { *target = *new; }
                }
            }
            Terminator::CondBranch { true_label, false_label, .. } => {
                for (old, new, _) in edge_changes {
                    if true_label == old { *true_label = *new; }
                    if false_label == old { *false_label = *new; }
                }
            }
            _ => {}
        }
        count += 1;

        // Update phi nodes in new target blocks.
        for (_old, new_target, phi_lookup_block) in edge_changes {
            if let Some(&target_idx) = label_to_idx.get(new_target) {
                let block = &mut func.blocks[target_idx];
                for inst in &mut block.instructions {
                    if let Instruction::Phi { incoming, .. } = inst {
                        let value_from_chain = incoming.iter()
                            .find(|(_, label)| *label == *phi_lookup_block)
                            .map(|(val, _)| *val);
                        if let Some(val) = value_from_chain {
                            if !incoming.iter().any(|(_, label)| *label == block_label) {
                                incoming.push((val, block_label));
                            }
                        }
                    }
                }
            }
        }
    }

    count
}

/// Check if threading a CondBranch's two edges to the same final target would
/// create a phi conflict — i.e., the target has phi nodes that carry different
/// values from the two paths.
fn would_create_phi_conflict(
    func: &IrFunction,
    block_label: BlockId,
    true_label: BlockId,
    false_label: BlockId,
    target: BlockId,
    resolved: &FxHashMap<BlockId, (BlockId, BlockId)>,
    label_to_idx: &FxHashMap<BlockId, usize>,
) -> bool {
    let target_block = match label_to_idx.get(&target) {
        Some(&idx) => &func.blocks[idx],
        None => return false,
    };

    // Determine the phi-lookup label for each path.
    let true_phi_label = resolved.get(&true_label).map(|&(_, pb)| pb).unwrap_or(block_label);
    let false_phi_label = resolved.get(&false_label).map(|&(_, pb)| pb).unwrap_or(block_label);

    if true_phi_label == false_phi_label {
        return false;
    }

    for inst in &target_block.instructions {
        if let Instruction::Phi { incoming, .. } = inst {
            let true_value = incoming.iter().find(|(_, l)| *l == true_phi_label).map(|(v, _)| v);
            let false_value = incoming.iter().find(|(_, l)| *l == false_phi_label).map(|(v, _)| v);
            if let (Some(tv), Some(fv)) = (true_value, false_value) {
                if !operands_equal(tv, fv) {
                    return true;
                }
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Sub-pass: dead block elimination
// ---------------------------------------------------------------------------

/// Remove blocks that have no predecessors (except the entry block, blocks[0]).
/// Returns the number of blocks removed.
fn remove_dead_blocks(func: &mut IrFunction, label_to_idx: &FxHashMap<BlockId, usize>) -> usize {
    if func.blocks.len() <= 1 {
        return 0;
    }

    // Compute the set of blocks reachable from the entry block via BFS.
    let entry = func.blocks[0].label;
    let mut reachable = FxHashSet::default();
    reachable.insert(entry);
    let mut worklist = vec![entry];

    // Blocks referenced by static local initializers (&&label in global data).
    for &block_id in &func.global_init_label_blocks {
        if reachable.insert(block_id) {
            worklist.push(block_id);
        }
    }

    while let Some(block_id) = worklist.pop() {
        if let Some(&idx) = label_to_idx.get(&block_id) {
            for_each_terminator_target(&func.blocks[idx].terminator, |target| {
                if reachable.insert(target) {
                    worklist.push(target);
                }
            });
            for inst in &func.blocks[idx].instructions {
                if let Instruction::LabelAddr { label, .. } = inst {
                    if reachable.insert(*label) {
                        worklist.push(*label);
                    }
                }
                // Always mark asm goto target blocks as reachable.
                // The backend always emits the asm body (using $0 placeholders
                // for unsatisfiable immediate constraints), so goto target labels
                // must exist in the assembly output. Removing them would cause
                // linker errors when the asm template references %l[label].
                if let Instruction::InlineAsm { goto_labels, .. } = inst {
                    for (_, label) in goto_labels {
                        if reachable.insert(*label) {
                            worklist.push(*label);
                        }
                    }
                }
            }
        }
    }

    let dead_blocks: FxHashSet<BlockId> = func.blocks.iter()
        .map(|b| b.label)
        .filter(|label| !reachable.contains(label))
        .collect();

    if dead_blocks.is_empty() {
        return 0;
    }

    // Clean up phi nodes and InlineAsm goto_labels in reachable blocks.
    for block in &mut func.blocks {
        if !reachable.contains(&block.label) {
            continue;
        }
        remove_phi_entries_from_set(block, &dead_blocks);
        // Defensive: clean up InlineAsm goto_labels referencing dead blocks.
        for inst in &mut block.instructions {
            if let Instruction::InlineAsm { goto_labels, .. } = inst {
                goto_labels.retain(|(_, label)| !dead_blocks.contains(label));
            }
        }
    }

    let original_len = func.blocks.len();
    func.blocks.retain(|b| reachable.contains(&b.label));
    original_len - func.blocks.len()
}

// ---------------------------------------------------------------------------
// Sub-pass: trivial phi simplification
// ---------------------------------------------------------------------------

/// Simplify trivial phi nodes: phi nodes with exactly one incoming edge,
/// or where all incoming values are identical, are replaced with Copy.
///
/// This is critical for patterns like `if (1 || expr)` where the `||`
/// short-circuit generates a phi that merges two paths, but after constant
/// branch folding removes the dead path, the phi has only one incoming
/// edge remaining.
fn simplify_trivial_phis(func: &mut IrFunction) -> usize {
    let mut count = 0;
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if let Instruction::Phi { dest, incoming, .. } = inst {
                let replacement = if incoming.len() == 1 {
                    Some(incoming[0].0)
                } else if incoming.len() > 1 {
                    let first = &incoming[0].0;
                    if incoming.iter().all(|(val, _)| operands_equal(val, first)) {
                        Some(*first)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(src) = replacement {
                    *inst = Instruction::Copy { dest: *dest, src };
                    count += 1;
                }
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Sub-pass: single-predecessor block merging
// ---------------------------------------------------------------------------

/// Merge single-predecessor blocks into their predecessor.
///
/// When block A ends with `Branch(B)` and B has exactly one predecessor (A),
/// the two blocks can be fused: B's instructions are appended to A and A
/// inherits B's terminator.
///
/// This is critical for stack frame reduction after inlining: each inlined
/// function body creates a chain of blocks connected by unconditional branches.
/// Without fusion, SSA values crossing these artificial block boundaries are
/// classified as multi-block and each gets a permanent stack slot (8 bytes).
fn merge_single_pred_blocks(func: &mut IrFunction) -> usize {
    if func.blocks.len() <= 1 {
        return 0;
    }

    let (pred_count, _) = build_pred_info(func);
    let label_to_idx = build_label_to_idx(func);

    // Precompute sets of blocks that must keep their identity:
    // - blocks referenced by LabelAddr (computed goto targets)
    // - blocks referenced by InlineAsm goto_labels
    // - blocks referenced by global_init_label_blocks
    let (label_addr_targets, asm_goto_targets) = collect_unmergeable_targets(func);

    let fusions = find_fusion_candidates(
        func, &pred_count, &label_to_idx,
        &label_addr_targets, &asm_goto_targets,
    );

    if fusions.is_empty() {
        return 0;
    }

    let count = fusions.len();
    execute_fusions(func, &fusions);
    patch_absorbed_references(func, &fusions);
    count
}

/// Precompute sets of blocks that cannot be merged because they are referenced
/// by LabelAddr instructions or InlineAsm goto_labels across the function.
///
/// Computing these once avoids O(blocks * instructions) scans per fusion candidate.
fn collect_unmergeable_targets(func: &IrFunction) -> (FxHashSet<BlockId>, FxHashSet<BlockId>) {
    let mut label_addr_targets = FxHashSet::default();
    let mut asm_goto_targets = FxHashSet::default();

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::LabelAddr { label, .. } = inst {
                label_addr_targets.insert(*label);
            }
            if let Instruction::InlineAsm { goto_labels, .. } = inst {
                for (_, label) in goto_labels {
                    asm_goto_targets.insert(*label);
                }
            }
        }
    }

    (label_addr_targets, asm_goto_targets)
}

/// Find (pred_idx, succ_idx) pairs where the predecessor can absorb the successor.
fn find_fusion_candidates(
    func: &IrFunction,
    pred_count: &FxHashMap<BlockId, u32>,
    label_to_idx: &FxHashMap<BlockId, usize>,
    label_addr_targets: &FxHashSet<BlockId>,
    asm_goto_targets: &FxHashSet<BlockId>,
) -> Vec<(usize, usize)> {
    let entry_label = func.blocks[0].label;
    let mut fusions = Vec::new();
    let mut absorbed: FxHashSet<usize> = FxHashSet::default();
    let mut absorbers: FxHashSet<usize> = FxHashSet::default();

    for (idx, block) in func.blocks.iter().enumerate() {
        let target = match &block.terminator {
            Terminator::Branch(t) => t,
            _ => continue,
        };

        if *target == entry_label { continue; }
        if pred_count.get(target).copied().unwrap_or(0) != 1 { continue; }

        let &succ_idx = match label_to_idx.get(target) {
            Some(si) => si,
            None => continue,
        };

        if idx == succ_idx { continue; }
        if absorbed.contains(&succ_idx) || absorbers.contains(&succ_idx) { continue; }
        if absorbed.contains(&idx) { continue; }

        // Successor contains InlineAsm with goto_labels — must stay separate.
        let has_asm_goto = func.blocks[succ_idx].instructions.iter().any(|inst| {
            matches!(inst, Instruction::InlineAsm { goto_labels, .. } if !goto_labels.is_empty())
        });
        if has_asm_goto { continue; }

        // Target is referenced by LabelAddr or asm goto — must keep identity.
        if label_addr_targets.contains(target) { continue; }
        if asm_goto_targets.contains(target) { continue; }

        // Target is referenced by global init (&&label in global data).
        if func.global_init_label_blocks.contains(target) { continue; }

        fusions.push((idx, succ_idx));
        absorbed.insert(succ_idx);
        absorbers.insert(idx);
    }

    fusions
}

/// Execute fusions: move successor instructions/terminator into predecessor.
fn execute_fusions(func: &mut IrFunction, fusions: &[(usize, usize)]) {
    for &(pred_idx, succ_idx) in fusions {
        let succ_instructions = std::mem::take(&mut func.blocks[succ_idx].instructions);
        let succ_terminator = std::mem::replace(
            &mut func.blocks[succ_idx].terminator,
            Terminator::Unreachable,
        );
        let succ_spans = std::mem::take(&mut func.blocks[succ_idx].source_spans);
        let pred_label = func.blocks[pred_idx].label;

        // Convert phi nodes to Copy (single-predecessor, so each Phi has one entry).
        let mut converted_instructions = Vec::with_capacity(succ_instructions.len());
        let mut converted_spans = Vec::with_capacity(succ_spans.len());

        for (i, inst) in succ_instructions.into_iter().enumerate() {
            if let Instruction::Phi { dest, incoming, .. } = &inst {
                let src = incoming.iter()
                    .find(|(_, label)| *label == pred_label)
                    .map(|(op, _)| *op)
                    .unwrap_or_else(|| {
                        // Fallback: single-pred blocks should always have a matching entry.
                        debug_assert!(!incoming.is_empty(), "Phi with no incoming in single-pred block");
                        if !incoming.is_empty() { incoming[0].0 } else { Operand::Const(IrConst::I64(0)) }
                    });
                converted_instructions.push(Instruction::Copy { dest: *dest, src });
            } else {
                converted_instructions.push(inst);
            }
            if i < succ_spans.len() {
                converted_spans.push(succ_spans[i]);
            }
        }

        // Maintain source_spans invariant: either empty or parallel to instructions.
        let pred_has_spans = !func.blocks[pred_idx].source_spans.is_empty();
        let succ_has_spans = !converted_spans.is_empty();

        if pred_has_spans && !succ_has_spans && !converted_instructions.is_empty() {
            converted_spans.resize(converted_instructions.len(), crate::common::source::Span::dummy());
        } else if !pred_has_spans && succ_has_spans {
            let pred_inst_len = func.blocks[pred_idx].instructions.len();
            func.blocks[pred_idx].source_spans.resize(
                pred_inst_len,
                crate::common::source::Span::dummy(),
            );
        }

        func.blocks[pred_idx].instructions.extend(converted_instructions);
        func.blocks[pred_idx].terminator = succ_terminator;
        if !converted_spans.is_empty() {
            func.blocks[pred_idx].source_spans.extend(converted_spans);
        }
    }
}

/// After fusing blocks, update phi nodes and terminators in all blocks to
/// replace references to absorbed block labels with their predecessor labels.
fn patch_absorbed_references(func: &mut IrFunction, fusions: &[(usize, usize)]) {
    let absorbed_to_pred: FxHashMap<BlockId, BlockId> = fusions.iter()
        .map(|&(pred_idx, succ_idx)| (func.blocks[succ_idx].label, func.blocks[pred_idx].label))
        .collect();

    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if let Instruction::Phi { incoming, .. } = inst {
                for (_, label) in incoming.iter_mut() {
                    if let Some(&new_label) = absorbed_to_pred.get(label) {
                        *label = new_label;
                    }
                }
            }
        }
        remap_terminator_targets(&mut block.terminator, &absorbed_to_pred);
    }
}

// ---------------------------------------------------------------------------
// Value resolution helpers
// ---------------------------------------------------------------------------

/// Build a map from Value -> (block_index, instruction_index) for the entire function.
fn build_global_value_map(func: &IrFunction) -> FxHashMap<Value, (usize, usize)> {
    let mut map = FxHashMap::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            let dest = match inst {
                Instruction::Copy { dest, .. } |
                Instruction::Phi { dest, .. } |
                Instruction::Cmp { dest, .. } |
                Instruction::Cast { dest, .. } |
                Instruction::Select { dest, .. } |
                Instruction::BinOp { dest, .. } |
                Instruction::UnaryOp { dest, .. } => Some(*dest),
                _ => None,
            };
            if let Some(d) = dest {
                map.insert(d, (bi, ii));
            }
        }
    }
    map
}

/// Resolve a Value to a constant by following its definition chain across blocks.
///
/// This handles chains like: Phi → Copy → Cmp(Ne, Cast(Phi), 0) where the inner
/// Phi has collapsed to a constant but the outer chain spans multiple blocks.
fn resolve_value_globally(func: &IrFunction, v: Value, val_map: &FxHashMap<Value, (usize, usize)>, depth: usize) -> Option<IrConst> {
    if depth > MAX_GLOBAL_RESOLVE_DEPTH {
        return None;
    }
    let &(bi, ii) = val_map.get(&v)?;
    let inst = &func.blocks[bi].instructions[ii];
    match inst {
        Instruction::Copy { src: Operand::Const(c), .. } => Some(*c),
        Instruction::Copy { src: Operand::Value(sv), .. } => {
            resolve_value_globally(func, *sv, val_map, depth + 1)
        }
        Instruction::Phi { incoming, .. } => {
            // All incoming must be the same constant.
            let mut common_val: Option<i64> = None;
            let mut first_const: Option<IrConst> = None;
            for (op, _) in incoming {
                let c = match op {
                    Operand::Const(c) => Some(*c),
                    Operand::Value(pv) => resolve_value_globally(func, *pv, val_map, depth + 1),
                };
                match c {
                    Some(c) => {
                        let ci = c.to_i64()?;
                        if let Some(prev) = common_val {
                            if prev != ci { return None; }
                        } else {
                            common_val = Some(ci);
                            first_const = Some(c);
                        }
                    }
                    None => return None,
                }
            }
            first_const
        }
        Instruction::Cmp { op, lhs, rhs, ty, .. } => {
            let l = resolve_operand_globally(func, lhs, val_map, depth + 1)?;
            let r = resolve_operand_globally(func, rhs, val_map, depth + 1)?;
            let result = op.eval_i64(ty.truncate_i64(l), ty.truncate_i64(r));
            Some(IrConst::I32(if result { 1 } else { 0 }))
        }
        Instruction::Cast { src: Operand::Const(c), .. } => Some(*c),
        Instruction::Cast { src: Operand::Value(sv), .. } => {
            resolve_value_globally(func, *sv, val_map, depth + 1)
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            let cond_val = resolve_operand_globally(func, cond, val_map, depth + 1)?;
            let chosen = if cond_val != 0 { true_val } else { false_val };
            match chosen {
                Operand::Const(c) => Some(*c),
                Operand::Value(cv) => resolve_value_globally(func, *cv, val_map, depth + 1),
            }
        }
        _ => None,
    }
}

/// Resolve an operand to an i64 constant using global cross-block resolution.
fn resolve_operand_globally(func: &IrFunction, op: &Operand, val_map: &FxHashMap<Value, (usize, usize)>, depth: usize) -> Option<i64> {
    match op {
        Operand::Const(c) => c.to_i64(),
        Operand::Value(v) => resolve_value_globally(func, *v, val_map, depth)?.to_i64(),
    }
}

/// Look through Copy, Phi, Cmp, and Select instructions in a block to resolve
/// a Value to a constant. This allows constant branch/switch folding to see
/// through instructions created by previous simplifications within the same
/// cfg_simplify fixpoint loop, without waiting for a separate copy_prop pass.
fn resolve_value_to_const_in_block(block: &BasicBlock, v: Value) -> Option<IrConst> {
    for inst in &block.instructions {
        match inst {
            Instruction::Copy { dest, src: Operand::Const(c) } if *dest == v => {
                return Some(*c);
            }
            Instruction::Phi { dest, incoming, .. } if *dest == v => {
                // Check if all incoming values are the same constant.
                let mut common_val: Option<i64> = None;
                let mut first_const: Option<IrConst> = None;
                for (op, _) in incoming {
                    match op {
                        Operand::Const(c) => {
                            let ci = c.to_i64()?;
                            if let Some(prev) = common_val {
                                if prev != ci { return None; }
                            } else {
                                common_val = Some(ci);
                                first_const = Some(*c);
                            }
                        }
                        _ => return None,
                    }
                }
                return first_const;
            }
            Instruction::Cmp { dest, op, lhs, rhs, ty } if *dest == v => {
                let l = resolve_operand_to_i64_in_block(block, lhs)?;
                let r = resolve_operand_to_i64_in_block(block, rhs)?;
                let result = op.eval_i64(ty.truncate_i64(l), ty.truncate_i64(r));
                return Some(IrConst::I32(if result { 1 } else { 0 }));
            }
            Instruction::Select { dest, cond, true_val, false_val, .. } if *dest == v => {
                let cond_const = resolve_operand_to_i64_in_block(block, cond)?;
                let chosen = if cond_const != 0 { true_val } else { false_val };
                return match chosen {
                    Operand::Const(c) => Some(*c),
                    Operand::Value(cv) => resolve_value_to_const_in_block(block, *cv),
                };
            }
            _ => {}
        }
    }
    None
}

/// Resolve an operand to an i64 constant within a single block.
fn resolve_operand_to_i64_in_block(block: &BasicBlock, op: &Operand) -> Option<i64> {
    match op {
        Operand::Const(c) => c.to_i64(),
        Operand::Value(v) => resolve_value_to_const_in_block(block, *v)?.to_i64(),
    }
}

/// Compare two operands for structural equality.
fn operands_equal(a: &Operand, b: &Operand) -> bool {
    match (a, b) {
        (Operand::Value(v1), Operand::Value(v2)) => v1.0 == v2.0,
        (Operand::Const(c1), Operand::Const(c2)) => consts_equal_for_phi(c1, c2),
        _ => false,
    }
}

/// Compare two IR constants for phi simplification.
/// Integer constants of different widths but same numeric value are considered
/// equal (e.g., I32(0) == I64(0)).
fn consts_equal_for_phi(a: &IrConst, b: &IrConst) -> bool {
    if a.to_hash_key() == b.to_hash_key() {
        return true;
    }
    match (a.to_i64(), b.to_i64()) {
        (Some(va), Some(vb)) => va == vb,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IrType;
    use crate::ir::reexports::IrCmpOp;

    /// Create a basic block for testing. Reduces boilerplate since
    /// `source_spans` is always empty in tests.
    fn make_block(label: BlockId, instructions: Vec<Instruction>, terminator: Terminator) -> BasicBlock {
        BasicBlock { label, instructions, terminator, source_spans: Vec::new() }
    }

    #[test]
    fn test_redundant_cond_branch() {
        let mut func = IrFunction::new("test".into(), IrType::Void, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(1)) }],
            Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(1),
            },
        ));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Return(None)));

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        // After redundant cond branch -> Branch(1), then merge_single_pred_blocks
        // merges Block 1 into Block 0, so Block 0 ends with Return(None).
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(None)));
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_jump_chain_threading() {
        let mut func = IrFunction::new("test".into(), IrType::Void, vec![], false);
        func.blocks.push(make_block(BlockId(0), vec![], Terminator::Branch(BlockId(1))));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Branch(BlockId(2))));
        func.blocks.push(make_block(BlockId(2), vec![], Terminator::Return(None)));

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        // After threading Block 0 -> Block 2 (skipping Block 1), Block 1 is dead,
        // then merge_single_pred_blocks merges Block 2 into Block 0.
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(None)));
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_dead_block_elimination() {
        let mut func = IrFunction::new("test".into(), IrType::Void, vec![], false);
        func.blocks.push(make_block(BlockId(0), vec![], Terminator::Return(None)));
        func.blocks.push(make_block(
            BlockId(1),
            vec![Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(42)) }],
            Terminator::Return(None),
        ));

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].label, BlockId(0));
    }

    #[test]
    fn test_combined_simplifications() {
        // CondBranch(1,1) -> Branch(1) -> thread to 2 -> dead block removal -> merge.
        // After all simplifications, Block 0 absorbs everything reachable.
        let mut func = IrFunction::new("test".into(), IrType::Void, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(1)) }],
            Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(1),
            },
        ));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Branch(BlockId(2))));
        func.blocks.push(make_block(BlockId(2), vec![], Terminator::Return(None)));
        func.blocks.push(make_block(BlockId(3), vec![], Terminator::Return(None)));

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        // After all passes: Block 3 is dead, intermediate blocks are merged.
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(None)));
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_phi_update_on_thread() {
        // Block 0 -> Block 1 (empty) -> Block 2 (phi referencing Block 1).
        // Threading skips Block 1, then trivial phi simplifies to Copy,
        // then Block 2 merges into Block 0.
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(42)) }],
            Terminator::Branch(BlockId(1)),
        ));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Branch(BlockId(2))));
        func.blocks.push(make_block(
            BlockId(2),
            vec![Instruction::Phi {
                dest: Value(1),
                ty: IrType::I32,
                incoming: vec![(Operand::Value(Value(0)), BlockId(1))],
            }],
            Terminator::Return(Some(Operand::Value(Value(1)))),
        ));

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        // After threading + dead block elimination + trivial phi -> Copy + merge:
        // Block 0 absorbs everything and ends with Return.
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(Some(Operand::Value(Value(1))))));
        // The Copy instruction should be in Block 0 now (from the merged phi-to-copy).
        let has_copy = func.blocks[0].instructions.iter().any(|inst| {
            matches!(inst, Instruction::Copy { dest: Value(1), .. })
        });
        assert!(has_copy, "Block 0 should contain the Copy from the merged phi");
    }

    #[test]
    fn test_no_thread_through_block_with_instructions() {
        // Block 1 has instructions, so it should NOT be threaded.
        // However, merge_single_pred_blocks will merge Block 1 into Block 0
        // (single pred), and then Block 2 into the merged block.
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(make_block(BlockId(0), vec![], Terminator::Branch(BlockId(1))));
        func.blocks.push(make_block(
            BlockId(1),
            vec![Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(42)) }],
            Terminator::Branch(BlockId(2)),
        ));
        func.blocks.push(make_block(
            BlockId(2),
            vec![],
            Terminator::Return(Some(Operand::Value(Value(0)))),
        ));

        let _ = simplify_cfg(&mut func);
        // No threading, but merge_single_pred_blocks merges all three blocks.
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(Some(Operand::Value(Value(0))))));
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_cond_branch_threading() {
        // Block 0 cond-branches to Block 1 and Block 2, both forward to Block 3.
        // Threading makes both targets Block 3, redundant cond branch -> Branch(3),
        // dead blocks removed, then Block 3 merged into Block 0.
        let mut func = IrFunction::new("test".into(), IrType::Void, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(1)) }],
            Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
        ));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Branch(BlockId(3))));
        func.blocks.push(make_block(BlockId(2), vec![], Terminator::Branch(BlockId(3))));
        func.blocks.push(make_block(BlockId(3), vec![], Terminator::Return(None)));

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        // All intermediate blocks absorbed: Block 0 ends with Return(None).
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(None)));
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_no_thread_when_phi_conflict() {
        let mut func = IrFunction::new("test".into(), IrType::I64, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![Instruction::Cmp {
                dest: Value(5),
                op: IrCmpOp::Eq,
                lhs: Operand::Value(Value(2)),
                rhs: Operand::Const(IrConst::I64(0)),
                ty: IrType::I64,
            }],
            Terminator::CondBranch {
                cond: Operand::Value(Value(5)),
                true_label: BlockId(1),
                false_label: BlockId(3),
            },
        ));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Branch(BlockId(3))));
        func.blocks.push(make_block(
            BlockId(3),
            vec![Instruction::Phi {
                dest: Value(8),
                ty: IrType::I64,
                incoming: vec![
                    (Operand::Value(Value(2)), BlockId(0)),
                    (Operand::Const(IrConst::I64(1)), BlockId(1)),
                ],
            }],
            Terminator::Return(Some(Operand::Value(Value(8)))),
        ));

        simplify_cfg(&mut func);

        match &func.blocks[0].terminator {
            Terminator::CondBranch { true_label, false_label, .. } => {
                assert!(
                    *true_label == BlockId(1) || *false_label != *true_label,
                    "Should not thread both branches to same target when phi has different values"
                );
            }
            Terminator::Branch(_) => {
                panic!("CondBranch was incorrectly simplified to unconditional Branch, losing phi distinction");
            }
            _ => panic!("Unexpected terminator"),
        }

        let merge_block = func.blocks.iter().find(|b| b.label == BlockId(3)).unwrap();
        if let Instruction::Phi { incoming, .. } = &merge_block.instructions[0] {
            assert!(incoming.len() >= 2, "Phi must retain at least 2 incoming edges");
            let has_const_1 = incoming.iter().any(|(val, _)| {
                matches!(val, Operand::Const(IrConst::I64(1)))
            });
            assert!(has_const_1, "Phi must still have the Const(1) incoming value");
        } else {
            panic!("Expected Phi instruction in merge block");
        }
    }

    #[test]
    fn test_trivial_phi_simplification() {
        // Simulates `__builtin_constant_p(v) && expr`:
        // Block 0 has constant-false cond branch, so Block 1 becomes dead.
        // The phi in Block 2 loses one incoming edge -> trivial phi -> Copy.
        // Then blocks merge into Block 0.
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![],
            Terminator::CondBranch {
                cond: Operand::Const(IrConst::I64(0)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
        ));
        func.blocks.push(make_block(
            BlockId(1),
            vec![Instruction::Cmp {
                dest: Value(1),
                op: IrCmpOp::Ne,
                lhs: Operand::Value(Value(0)),
                rhs: Operand::Const(IrConst::I64(0)),
                ty: IrType::I64,
            }],
            Terminator::Branch(BlockId(2)),
        ));
        func.blocks.push(make_block(
            BlockId(2),
            vec![Instruction::Phi {
                dest: Value(2),
                ty: IrType::I64,
                incoming: vec![
                    (Operand::Const(IrConst::I64(0)), BlockId(0)),
                    (Operand::Value(Value(1)), BlockId(1)),
                ],
            }],
            Terminator::CondBranch {
                cond: Operand::Value(Value(2)),
                true_label: BlockId(3),
                false_label: BlockId(4),
            },
        ));
        func.blocks.push(make_block(
            BlockId(3),
            vec![],
            Terminator::Return(Some(Operand::Const(IrConst::I32(1)))),
        ));
        func.blocks.push(make_block(
            BlockId(4),
            vec![],
            Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
        ));

        let _ = simplify_cfg(&mut func);

        // Block 1 should be dead (unreachable).
        let has_block_1 = func.blocks.iter().any(|b| b.label == BlockId(1));
        assert!(!has_block_1, "Dead block 1 (RHS of &&) should be eliminated");

        // The phi should have been simplified to a Copy with Const(0).
        // This Copy may be in Block 0 (if Block 2 was merged) or Block 2.
        let has_copy_const_0 = func.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                matches!(inst, Instruction::Copy { dest: Value(2), src: Operand::Const(IrConst::I64(0)) })
            })
        });
        assert!(has_copy_const_0, "Should have Copy(Value(2), Const(0)) somewhere");
    }

    #[test]
    fn test_fold_constant_switch() {
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![],
            Terminator::Switch {
                val: Operand::Const(IrConst::I64(37)),
                cases: vec![(10, BlockId(1)), (20, BlockId(2)), (30, BlockId(3))],
                default: BlockId(4),
                ty: IrType::I64,
            },
        ));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(100))))));
        func.blocks.push(make_block(BlockId(2), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(200))))));
        func.blocks.push(make_block(BlockId(3), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(300))))));
        func.blocks.push(make_block(BlockId(4), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(-1))))));

        let count = simplify_cfg(&mut func);
        assert!(count > 0, "Should have made simplifications");
        // After switch fold -> Branch(4), dead blocks removed, Block 4 merged into Block 0.
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(Some(Operand::Const(IrConst::I32(-1))))),
            "Switch on constant 37 should result in Return(-1) after merge");
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_fold_constant_switch_matching_case() {
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![],
            Terminator::Switch {
                val: Operand::Const(IrConst::I64(20)),
                cases: vec![(10, BlockId(1)), (20, BlockId(2)), (30, BlockId(3))],
                default: BlockId(4),
                ty: IrType::I64,
            },
        ));
        func.blocks.push(make_block(BlockId(1), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(100))))));
        func.blocks.push(make_block(BlockId(2), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(200))))));
        func.blocks.push(make_block(BlockId(3), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(300))))));
        func.blocks.push(make_block(BlockId(4), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(-1))))));

        let count = simplify_cfg(&mut func);
        assert!(count > 0, "Should have made simplifications");
        // After switch fold -> Branch(2), dead blocks removed, Block 2 merged into Block 0.
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(Some(Operand::Const(IrConst::I32(200))))),
            "Switch on constant 20 should result in Return(200) after merge");
        assert_eq!(func.blocks.len(), 1);
    }

    #[test]
    fn test_fold_constant_switch_phi_cleanup() {
        // Block 0: Copy(Value(1), Const(1)), CondBranch(Value(1), true:1, false:3)
        // Block 1: Switch(Const(10), case 10->Block 2, default->Block 3)
        // Block 2: Return(42)
        // Block 3: Phi from Block 0 and Block 1, Return(phi)
        //
        // After folding:
        // - Block 0's CondBranch resolves (cond=1) -> Branch(1)
        // - Block 1's Switch resolves (val=10) -> Branch(2), removes phi entry from Block 3
        // - Block 3 becomes dead (only Block 0 went to it, but that's folded to Block 1)
        // - Result: Block 0 -> Block 1 -> Block 2, all merge.
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(make_block(
            BlockId(0),
            vec![Instruction::Copy { dest: Value(1), src: Operand::Const(IrConst::I32(1)) }],
            Terminator::CondBranch {
                cond: Operand::Value(Value(1)),
                true_label: BlockId(1),
                false_label: BlockId(3),
            },
        ));
        func.blocks.push(make_block(
            BlockId(1),
            vec![],
            Terminator::Switch {
                val: Operand::Const(IrConst::I64(10)),
                cases: vec![(10, BlockId(2))],
                default: BlockId(3),
                ty: IrType::I64,
            },
        ));
        func.blocks.push(make_block(BlockId(2), vec![], Terminator::Return(Some(Operand::Const(IrConst::I32(42))))));
        func.blocks.push(make_block(
            BlockId(3),
            vec![Instruction::Phi {
                dest: Value(0),
                ty: IrType::I32,
                incoming: vec![
                    (Operand::Const(IrConst::I32(1)), BlockId(0)),
                    (Operand::Const(IrConst::I32(2)), BlockId(1)),
                ],
            }],
            Terminator::Return(Some(Operand::Value(Value(0)))),
        ));

        let count = simplify_cfg(&mut func);
        assert!(count > 0, "Should have made simplifications");

        // After all folding and merging, Block 0 should end with Return(42).
        assert!(matches!(func.blocks[0].terminator, Terminator::Return(Some(Operand::Const(IrConst::I32(42))))),
            "After full simplification, should return 42");
    }
}
