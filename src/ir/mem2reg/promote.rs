//! mem2reg: Promote stack allocas to SSA registers with phi insertion.
//!
//! This implements the standard SSA construction algorithm:
//! 1. Identify promotable allocas (scalar, only loaded/stored, not address-taken)
//! 2. Build CFG (from terminators and InlineAsm goto edges)
//! 3. Compute dominator tree (Cooper-Harvey-Kennedy algorithm)
//! 4. Compute dominance frontiers
//! 5. Insert phi nodes at iterated dominance frontiers of defining blocks
//! 6. Rename variables via dominator tree DFS
//!
//! Reference: "A Simple, Fast Dominance Algorithm" by Cooper, Harvey, Kennedy (2001)

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use crate::ir::reexports::{
    BlockId,
    Instruction,
    IrConst,
    IrFunction,
    IrModule,
    Operand,
    Terminator,
    Value,
};
use crate::ir::analysis;
use crate::backend::inline_asm::constraint_is_memory_only;
use crate::common::types::IrType;

/// Maximum byte size for an alloca to be considered promotable to an SSA register.
/// This corresponds to the width of a general-purpose register (8 bytes on 64-bit,
/// 4 bytes on 32-bit). Allocas larger than this are aggregates (arrays, structs)
/// that cannot live in a single register.
const MAX_PROMOTABLE_ALLOCA_SIZE: usize = 8;

/// Promote allocas to SSA form with phi insertion.
/// Only promotes scalar allocas that are exclusively loaded/stored
/// (not address-taken by GEP, memcpy, va_start, etc.).
/// Promote allocas to SSA form, skipping parameter allocas.
/// This is the initial mem2reg pass that runs before inlining.
/// Parameter allocas are not promoted here because the inliner assumes
/// they exist for argument passing.
pub fn promote_allocas(module: &mut IrModule) {
    for func in &mut module.functions {
        if func.is_declaration || func.blocks.is_empty() {
            continue;
        }
        promote_function(func, false);
    }
}

/// Promote allocas to SSA form, including parameter allocas.
/// This is the post-inlining mem2reg pass. Parameter allocas can now be
/// promoted because inlining has already completed and the IR lowering emits
/// explicit ParamRef + Store instructions that make parameter values visible.
pub fn promote_allocas_with_params(module: &mut IrModule) {
    for func in &mut module.functions {
        if func.is_declaration || func.blocks.is_empty() {
            continue;
        }
        promote_function(func, true);
    }
}

/// Information about a promotable alloca.
struct AllocaInfo {
    /// The Value that is the alloca's destination (pointer to the stack slot).
    alloca_value: Value,
    /// The IR type of the stored value.
    ty: IrType,
    /// Block indices where this alloca is stored to (defining blocks).
    def_blocks: FxHashSet<usize>,
    /// Block indices where this alloca is loaded from (use blocks).
    use_blocks: FxHashSet<usize>,
}

/// Promote allocas in a single function to SSA form.
/// If `promote_params` is true, parameter allocas in the entry block are also
/// eligible for promotion.
fn promote_function(func: &mut IrFunction, promote_params: bool) {
    if func.blocks.is_empty() {
        return;
    }

    // Step 1: Identify promotable allocas
    let mut alloca_infos = find_promotable_allocas(func, promote_params);
    if std::env::var("CCC_DEBUG_MEM2REG").is_ok() {
        let total_allocas: usize = func.blocks[0].instructions.iter()
            .filter(|i| matches!(i, Instruction::Alloca { .. }))
            .count();
        eprintln!("[mem2reg] func '{}': {} total allocas, {} promotable, {} params",
            func.name, total_allocas, alloca_infos.len(), func.params.len());
    }
    if alloca_infos.is_empty() {
        return;
    }

    // Step 2: Build CFG
    let num_blocks = func.blocks.len();
    let label_to_idx = analysis::build_label_map(func);
    let (preds, succs) = analysis::build_cfg(func, &label_to_idx);

    // Step 3: Compute dominator tree
    let idom = analysis::compute_dominators(num_blocks, &preds, &succs);

    // Step 4: Compute dominance frontiers
    let df = analysis::compute_dominance_frontiers(num_blocks, &preds, &idom);

    // Step 5: Insert phi nodes with cost limiting.
    //
    // Phi cost limiting: each phi with P predecessors generates ~P copies during
    // phi elimination. For functions with large switch/computed-goto patterns
    // (e.g. Lua's VM dispatch with 84 opcodes), promoting many allocas creates
    // O(cases * allocas) copies. We estimate the total copy cost and exclude
    // expensive allocas from promotion, leaving them as stack variables.
    //
    // 50K copies * 8 bytes per stack slot = ~400KB, well under the typical 8MB
    // stack limit while accommodating moderately complex functions.
    const MAX_PHI_COPY_COST: usize = 50_000;

    let phi_locations = insert_phis(&alloca_infos, &df, num_blocks);

    // Compute per-alloca phi cost: sum of predecessor counts at all phi sites
    let total_phi_cost: usize = alloca_infos.iter().enumerate()
        .map(|(alloca_idx, _)| {
            phi_locations.iter().enumerate()
                .filter(|(_, phi_set)| phi_set.contains(&alloca_idx))
                .map(|(block_idx, _)| preds.len(block_idx))
                .sum::<usize>()
        })
        .sum();

    if total_phi_cost > MAX_PHI_COPY_COST {
        // Compute per-alloca costs and remove the most expensive first
        let mut alloca_phi_costs: Vec<(usize, usize)> = alloca_infos.iter().enumerate()
            .map(|(alloca_idx, _)| {
                let cost: usize = phi_locations.iter().enumerate()
                    .filter(|(_, phi_set)| phi_set.contains(&alloca_idx))
                    .map(|(block_idx, _)| preds.len(block_idx))
                    .sum();
                (alloca_idx, cost)
            })
            .collect();
        alloca_phi_costs.sort_by(|a, b| b.1.cmp(&a.1));

        let mut remaining_cost = total_phi_cost;
        let mut remove_set: FxHashSet<usize> = FxHashSet::default();
        for &(alloca_idx, cost) in &alloca_phi_costs {
            if remaining_cost <= MAX_PHI_COPY_COST {
                break;
            }
            remove_set.insert(alloca_idx);
            remaining_cost -= cost;
        }

        if !remove_set.is_empty() {
            alloca_infos = alloca_infos.into_iter().enumerate()
                .filter(|(idx, _)| !remove_set.contains(idx))
                .map(|(_, info)| info)
                .collect();
        }
    }

    if alloca_infos.is_empty() {
        return;
    }

    // Recompute phi locations after any cost-based filtering
    let phi_locations = insert_phis(&alloca_infos, &df, num_blocks);

    // Step 6: Rename variables
    let dom_children = analysis::build_dom_tree_children(num_blocks, &idom);
    rename_variables(func, &alloca_infos, &phi_locations, &dom_children, &preds, &label_to_idx);
}

/// Find all allocas that can be promoted to SSA registers.
/// Scans all blocks (entry and non-entry) so that inlined locals are also found.
/// A promotable alloca must:
/// - Have scalar type (size <= MAX_PROMOTABLE_ALLOCA_SIZE)
/// - Only be used by Load and Store instructions (not address-taken)
/// - Not be volatile
fn find_promotable_allocas(func: &IrFunction, promote_params: bool) -> Vec<AllocaInfo> {
    let num_params = func.params.len();

    // Collect all allocas from the entry block.
    // When promote_params is false, skip the first num_params allocas (parameter allocas)
    // because the inliner relies on their presence for argument passing.
    // When promote_params is true (post-inlining), parameter allocas are eligible
    // because the IR has explicit ParamRef + Store instructions for their initial values.
    let mut alloca_index = 0;
    let mut all_allocas: Vec<(Value, IrType, usize)> = func.blocks[0]
        .instructions
        .iter()
        .filter_map(|inst| {
            if let Instruction::Alloca { dest, ty, size, volatile, .. } = inst {
                let idx = alloca_index;
                alloca_index += 1;
                // Skip parameter allocas when not promoting params
                if !promote_params && idx < num_params {
                    return None;
                }
                // Never promote volatile allocas -- volatile locals must remain
                // in memory so their values survive setjmp/longjmp and are not
                // cached in registers that longjmp would restore to stale values.
                if *volatile {
                    return None;
                }
                // Only promote scalar allocas that fit in a register.
                // Larger allocas are for arrays/structs passed by value.
                // Note: the alloca size may be larger than the IR type size
                // (e.g., int has type I32 = 4 bytes but alloc size 8 for alignment).
                // We allow promotion as long as the alloca is at most register-width
                // and the type itself is scalar (at most register-width).
                let type_size = ir_type_size(*ty);
                if *size <= MAX_PROMOTABLE_ALLOCA_SIZE && type_size <= MAX_PROMOTABLE_ALLOCA_SIZE {
                    Some((*dest, *ty, *size))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Also collect allocas from non-entry blocks. These come from inlined functions
    // whose entry blocks (containing their local variable allocas) are appended as
    // non-entry blocks in the caller. Without promoting these, inlined parameters
    // remain as store/load pairs through stack slots, preventing constant propagation
    // into inline asm "i" constraints and other optimizations.
    for block in &func.blocks[1..] {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, ty, size, volatile, .. } = inst {
                if *volatile {
                    continue;
                }
                let type_size = ir_type_size(*ty);
                if *size <= MAX_PROMOTABLE_ALLOCA_SIZE && type_size <= MAX_PROMOTABLE_ALLOCA_SIZE {
                    all_allocas.push((*dest, *ty, *size));
                }
            }
        }
    }

    if all_allocas.is_empty() {
        return Vec::new();
    }

    // Build set of candidate alloca values
    let candidate_set: FxHashSet<u32> = all_allocas.iter().map(|(v, _, _)| v.0).collect();

    // Check all uses: only Load and Store targeting the alloca pointer are allowed
    let mut disqualified: FxHashSet<u32> = FxHashSet::default();
    let mut def_blocks: FxHashMap<u32, FxHashSet<usize>> = FxHashMap::default();
    let mut use_blocks: FxHashMap<u32, FxHashSet<usize>> = FxHashMap::default();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            match inst {
                Instruction::Load { ptr, .. } => {
                    if candidate_set.contains(&ptr.0) && !disqualified.contains(&ptr.0) {
                        use_blocks.entry(ptr.0).or_default().insert(block_idx);
                    }
                }
                Instruction::Store { val, ptr, .. } => {
                    if candidate_set.contains(&ptr.0) && !disqualified.contains(&ptr.0) {
                        def_blocks.entry(ptr.0).or_default().insert(block_idx);
                    }
                    // If a candidate alloca value appears as the stored VALUE (not ptr),
                    // it means the alloca's address is being used as data (e.g., array-to-pointer
                    // decay: Y.p = local_array). This is an address-taken use and the alloca
                    // must not be promoted, since promotion would lose the stack address.
                    if let Operand::Value(v) = val {
                        if candidate_set.contains(&v.0) {
                            disqualified.insert(v.0);
                        }
                    }
                }
                // InlineAsm: output pointers act as definitions (like Store),
                // while input operand Values that are allocas disqualify them
                // (the address is taken for memory/address constraints).
                // Memory-only output constraints (=m, =o, =V, =p) also disqualify
                // the alloca because the inline asm writes directly to the alloca's
                // stack memory through the template (the backend substitutes the
                // stack-relative address like -8(%rbp) for the operand).
                Instruction::InlineAsm { outputs, inputs, .. } => {
                    // Output pointers: treat as definitions of the alloca,
                    // unless the constraint is memory-only (=m) in which case
                    // the alloca must keep its stack slot.
                    for (constraint, ptr, _) in outputs {
                        if candidate_set.contains(&ptr.0) {
                            if constraint_is_memory_only(constraint, false) {
                                // Memory-only output: the inline asm template writes
                                // directly to the alloca's memory address. Must keep
                                // the alloca on the stack, not promote to SSA.
                                disqualified.insert(ptr.0);
                            } else if !disqualified.contains(&ptr.0) {
                                def_blocks.entry(ptr.0).or_default().insert(block_idx);
                            }
                        }
                    }
                    // Input operands: if a candidate alloca appears as a Value,
                    // its address is being taken (e.g., for "+m" or "+A" constraints).
                    // This disqualifies it from promotion.
                    for (_, op, _) in inputs {
                        if let Operand::Value(v) = op {
                            if candidate_set.contains(&v.0) {
                                disqualified.insert(v.0);
                            }
                        }
                    }
                }
                // Any other use of the alloca value disqualifies it
                _ => {
                    for used_val in inst.used_values() {
                        if candidate_set.contains(&used_val) {
                            disqualified.insert(used_val);
                        }
                    }
                }
            }
        }

        // Check terminator uses
        for used_val in block.terminator.used_values() {
            if candidate_set.contains(&used_val) {
                disqualified.insert(used_val);
            }
        }
    }

    // Build set of parameter alloca values for filtering below
    let param_alloca_set: FxHashSet<u32> = func.param_alloca_values.iter().map(|v| v.0).collect();

    // Build final list of promotable allocas
    all_allocas
        .into_iter()
        .filter(|(v, _, _)| !disqualified.contains(&v.0))
        .map(|(alloca_value, ty, _)| {
            AllocaInfo {
                alloca_value,
                ty,
                def_blocks: def_blocks.remove(&alloca_value.0).unwrap_or_default(),
                use_blocks: use_blocks.remove(&alloca_value.0).unwrap_or_default(),
            }
        })
        // Only promote allocas that are actually used (have loads or stores)
        .filter(|info| !info.def_blocks.is_empty() || !info.use_blocks.is_empty())
        // Parameter allocas (sret, struct params) that have no IR-visible stores must
        // NOT be promoted: they receive their values from emit_store_params at the
        // backend level. Only param allocas with explicit ParamRef+Store (scalar params)
        // have def_blocks and can safely be promoted.
        .filter(|info| {
            if param_alloca_set.contains(&info.alloca_value.0) && info.def_blocks.is_empty() {
                false // Skip param allocas without any stores
            } else {
                true
            }
        })
        .collect()
}

/// Determine where phi nodes need to be inserted.
/// Returns a map: block_index -> set of alloca indices that need phis there.
fn insert_phis(
    alloca_infos: &[AllocaInfo],
    df: &[FxHashSet<usize>],
    num_blocks: usize,
) -> Vec<FxHashSet<usize>> {
    // phi_locations[block_idx] = set of alloca indices that need a phi at this block
    let mut phi_locations = vec![FxHashSet::default(); num_blocks];

    for (alloca_idx, info) in alloca_infos.iter().enumerate() {
        // Iterated dominance frontier algorithm
        let mut worklist: VecDeque<usize> = info.def_blocks.iter().copied().collect();
        let mut has_phi: FxHashSet<usize> = FxHashSet::default();
        let mut ever_in_worklist: FxHashSet<usize> = info.def_blocks.clone();

        while let Some(block) = worklist.pop_front() {
            for &df_block in &df[block] {
                if has_phi.insert(df_block) {
                    phi_locations[df_block].insert(alloca_idx);
                    if ever_in_worklist.insert(df_block) {
                        worklist.push_back(df_block);
                    }
                }
            }
        }
    }

    phi_locations
}


/// Rename variables to complete SSA construction.
/// This traverses the dominator tree, maintaining stacks of current definitions
/// for each promoted alloca, and rewrites loads/stores to use SSA values.
fn rename_variables(
    func: &mut IrFunction,
    alloca_infos: &[AllocaInfo],
    phi_locations: &[FxHashSet<usize>],
    dom_children: &[Vec<usize>],
    preds: &analysis::FlatAdj,
    label_to_idx: &FxHashMap<BlockId, usize>,
) {
    let num_allocas = alloca_infos.len();

    // Map alloca value -> alloca index for quick lookup
    let alloca_to_idx: FxHashMap<u32, usize> = alloca_infos
        .iter()
        .enumerate()
        .map(|(i, info)| (info.alloca_value.0, i))
        .collect();

    // Use cached next_value_id if available, otherwise scan
    let mut next_value = if func.next_value_id > 0 {
        func.next_value_id
    } else {
        func.max_value_id() + 1
    };

    // First, insert phi instructions at the appropriate blocks.
    // We do this before renaming so the phi dests get fresh values during rename.
    // For now, insert placeholder phis with empty incoming lists.
    // phi_dests[block_idx][alloca_idx] = the Value for the phi's dest (if there is one)
    let mut phi_dests: Vec<FxHashMap<usize, Value>> = vec![FxHashMap::default(); func.blocks.len()];

    for (block_idx, alloca_set) in phi_locations.iter().enumerate() {
        let mut phi_instructions = Vec::new();
        for &alloca_idx in alloca_set {
            let info = &alloca_infos[alloca_idx];
            let dest = Value(next_value);
            next_value += 1;
            phi_dests[block_idx].insert(alloca_idx, dest);
            phi_instructions.push(Instruction::Phi {
                dest,
                ty: info.ty,
                incoming: Vec::new(), // will be filled during rename
            });
        }
        // Prepend phis to block instructions
        if !phi_instructions.is_empty() {
            let num_phis = phi_instructions.len();
            phi_instructions.append(&mut func.blocks[block_idx].instructions);
            func.blocks[block_idx].instructions = phi_instructions;
            // Prepend dummy spans for the phi instructions
            if !func.blocks[block_idx].source_spans.is_empty() {
                let mut new_spans = vec![crate::common::source::Span::dummy(); num_phis];
                new_spans.append(&mut func.blocks[block_idx].source_spans);
                func.blocks[block_idx].source_spans = new_spans;
            }
        }
    }

    // Rename using dominator tree DFS
    // def_stacks[alloca_idx] = stack of current SSA value for that alloca
    let mut def_stacks: Vec<Vec<Operand>> = vec![Vec::new(); num_allocas];

    // Initialize with undef (zero constant of appropriate type) for allocas
    // that might be read before being written
    for (i, info) in alloca_infos.iter().enumerate() {
        def_stacks[i].push(Operand::Const(IrConst::zero(info.ty)));
    }

    rename_block(
        0,
        func,
        &alloca_to_idx,
        alloca_infos,
        &mut def_stacks,
        &mut next_value,
        &phi_dests,
        dom_children,
        preds,
        label_to_idx,
    );

    // Remove promoted allocas from the entry block, and remove dead loads/stores
    remove_promoted_instructions(func, &alloca_to_idx);

    // Update cached next_value_id for downstream passes
    func.next_value_id = next_value;
}

/// Recursive dominator-tree DFS for variable renaming.
fn rename_block(
    block_idx: usize,
    func: &mut IrFunction,
    alloca_to_idx: &FxHashMap<u32, usize>,
    alloca_infos: &[AllocaInfo],
    def_stacks: &mut [Vec<Operand>],
    next_value: &mut u32,
    phi_dests: &[FxHashMap<usize, Value>],
    dom_children: &[Vec<usize>],
    _preds: &analysis::FlatAdj,
    label_to_idx: &FxHashMap<BlockId, usize>,
) {
    // Record stack depths so we can pop on exit
    let stack_depths: Vec<usize> = def_stacks.iter().map(|s| s.len()).collect();

    // Process phi nodes in this block - push their dests onto stacks
    for inst in &func.blocks[block_idx].instructions {
        if let Instruction::Phi { dest, .. } = inst {
            // Find which alloca this phi is for
            if let Some(&alloca_idx) = phi_dests[block_idx]
                .iter()
                .find(|(_, &v)| v == *dest)
                .map(|(idx, _)| idx)
            {
                def_stacks[alloca_idx].push(Operand::Value(*dest));
            }
        }
    }

    // Rewrite instructions in this block.
    //
    // asm goto snapshot: An InlineAsm with goto_labels creates implicit
    // control-flow edges from the *point of the asm* to its label targets.
    // Any definitions that occur *after* the asm goto in the same block
    // (e.g., a subsequent InlineAsm output) must NOT be visible along
    // the goto edge.  We snapshot the def-stack tops at each asm goto
    // and use those snapshots when filling in phi incoming values for
    // the goto targets, instead of the end-of-block def-stack tops.
    let mut new_instructions = Vec::with_capacity(func.blocks[block_idx].instructions.len());
    let mut new_spans = Vec::new();
    let has_spans = !func.blocks[block_idx].source_spans.is_empty();
    if has_spans {
        new_spans.reserve(func.blocks[block_idx].source_spans.len());
    }
    // Map from goto-target BlockId to the def-stack snapshot at the goto point.
    // Each snapshot is a Vec<Operand> indexed by alloca_idx.
    let mut goto_label_snapshots: FxHashMap<BlockId, Vec<Operand>> = FxHashMap::default();
    let old_spans = std::mem::take(&mut func.blocks[block_idx].source_spans);
    for (inst_idx, inst) in func.blocks[block_idx].instructions.drain(..).enumerate() {
        match inst {
            Instruction::Load { dest, ptr, ty, seg_override } => {
                if let Some(&alloca_idx) = alloca_to_idx.get(&ptr.0) {
                    // Replace load with copy from current SSA value
                    let current_val = def_stacks[alloca_idx].last().cloned()
                        .unwrap_or(Operand::Const(IrConst::zero(ty)));
                    new_instructions.push(Instruction::Copy {
                        dest,
                        src: current_val,
                    });
                    if has_spans { new_spans.push(old_spans[inst_idx]); }
                } else {
                    new_instructions.push(Instruction::Load { dest, ptr, ty, seg_override });
                    if has_spans { new_spans.push(old_spans[inst_idx]); }
                }
            }
            Instruction::Store { val, ptr, ty, seg_override } => {
                if let Some(&alloca_idx) = alloca_to_idx.get(&ptr.0) {
                    // Push the stored value onto the def stack.
                    // Narrow constants to match the alloca type: the IR lowering
                    // always produces I64 constants for integer literals, but on
                    // 32-bit targets this causes phi copies to use 64-bit
                    // operations for 32-bit values, leaving high bits
                    // uninitialized in some paths.
                    let narrowed_val = match val {
                        Operand::Const(c) => {
                            Operand::Const(c.narrowed_to(alloca_infos[alloca_idx].ty))
                        }
                        other => other,
                    };
                    def_stacks[alloca_idx].push(narrowed_val);
                    // Remove the store (it's now represented by the SSA def)
                    // (span is dropped along with the instruction)
                } else {
                    new_instructions.push(Instruction::Store { val, ptr, ty, seg_override });
                    if has_spans { new_spans.push(old_spans[inst_idx]); }
                }
            }
            Instruction::InlineAsm {
                template, mut outputs, inputs, clobbers,
                operand_types, goto_labels, input_symbols, seg_overrides,
            } => {
                // Snapshot def stacks BEFORE processing outputs, so goto
                // targets see the values that were live at the point of the
                // asm goto, not any definitions produced by the asm outputs.
                if !goto_labels.is_empty() {
                    let snapshot: Vec<Operand> = (0..alloca_infos.len())
                        .map(|ai| {
                            def_stacks[ai].last().cloned()
                                .unwrap_or(Operand::Const(IrConst::zero(alloca_infos[ai].ty)))
                        })
                        .collect();
                    for (_, label) in &goto_labels {
                        goto_label_snapshots.insert(*label, snapshot.clone());
                    }
                }

                // Replace output pointers that are promoted allocas with fresh SSA values
                for (_, out_ptr, _) in outputs.iter_mut() {
                    if let Some(&alloca_idx) = alloca_to_idx.get(&out_ptr.0) {
                        let fresh = Value(*next_value);
                        *next_value += 1;
                        *out_ptr = fresh;
                        def_stacks[alloca_idx].push(Operand::Value(fresh));
                    }
                }
                new_instructions.push(Instruction::InlineAsm {
                    template, outputs, inputs, clobbers,
                    operand_types, goto_labels, input_symbols, seg_overrides,
                });
                if has_spans { new_spans.push(old_spans[inst_idx]); }
            }
            other => {
                new_instructions.push(other);
                if has_spans { new_spans.push(old_spans[inst_idx]); }
            }
        }
    }
    func.blocks[block_idx].instructions = new_instructions;
    func.blocks[block_idx].source_spans = new_spans;

    // Also rewrite terminator operands if they reference promoted allocas
    // (this shouldn't normally happen, but let's be safe)

    // Fill in phi incoming values in successor blocks
    let mut succ_labels = get_successor_labels(&func.blocks[block_idx].terminator);
    // InlineAsm goto_labels are implicit control flow edges.
    for inst in &func.blocks[block_idx].instructions {
        if let Instruction::InlineAsm { goto_labels, .. } = inst {
            for (_, label) in goto_labels {
                if !succ_labels.contains(label) {
                    succ_labels.push(*label);
                }
            }
        }
    }
    let current_block_label = func.blocks[block_idx].label;

    for succ_label in &succ_labels {
        if let Some(&succ_idx) = label_to_idx.get(succ_label) {
            // For each phi in the successor block, fill in our value.
            // For asm goto targets, use the snapshotted def values from the
            // point of the asm goto, not the end-of-block values.
            let goto_snapshot = goto_label_snapshots.get(succ_label);
            for inst in &mut func.blocks[succ_idx].instructions {
                if let Instruction::Phi { dest, incoming, .. } = inst {
                    // Find which alloca this phi is for
                    if let Some(&alloca_idx) = phi_dests[succ_idx]
                        .iter()
                        .find(|(_, &v)| v == *dest)
                        .map(|(idx, _)| idx)
                    {
                        let current_val = if let Some(snapshot) = goto_snapshot {
                            snapshot[alloca_idx]
                        } else {
                            def_stacks[alloca_idx].last().cloned()
                                .unwrap_or(Operand::Const(IrConst::zero(alloca_infos[alloca_idx].ty)))
                        };
                        incoming.push((current_val, current_block_label));
                    }
                } else {
                    break; // Phis are always at the start of a block
                }
            }
        }
    }

    // Recurse into dominator tree children.
    //
    // For children that are asm-goto targets from this block, we must
    // temporarily adjust the def stacks to the snapshot captured at the
    // goto point. Otherwise the child would see definitions that were
    // produced *after* the asm goto (e.g., InlineAsm outputs) which
    // are not live along the goto edge.
    let children: Vec<usize> = dom_children[block_idx].clone();
    // Build a set of goto-target block indices for quick lookup.
    let goto_target_indices: FxHashMap<usize, &Vec<Operand>> = goto_label_snapshots
        .iter()
        .filter_map(|(label, snapshot)| {
            label_to_idx.get(label).map(|&idx| (idx, snapshot))
        })
        .collect();

    for child in children {
        if let Some(snapshot) = goto_target_indices.get(&child) {
            // This child is an asm-goto target: temporarily push snapshot
            // values so that loads in the child block see the pre-goto defs.
            let child_depths: Vec<usize> = def_stacks.iter().map(|s| s.len()).collect();
            for (ai, snap_val) in snapshot.iter().enumerate() {
                def_stacks[ai].push(*snap_val);
            }
            rename_block(
                child,
                func,
                alloca_to_idx,
                alloca_infos,
                def_stacks,
                next_value,
                phi_dests,
                dom_children,
                _preds,
                label_to_idx,
            );
            for (i, &depth) in child_depths.iter().enumerate() {
                def_stacks[i].truncate(depth);
            }
        } else {
            rename_block(
                child,
                func,
                alloca_to_idx,
                alloca_infos,
                def_stacks,
                next_value,
                phi_dests,
                dom_children,
                _preds,
                label_to_idx,
            );
        }
    }

    // Pop definitions pushed in this block
    for (i, &depth) in stack_depths.iter().enumerate() {
        def_stacks[i].truncate(depth);
    }
}

/// Get successor labels from a terminator.
fn get_successor_labels(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Branch(label) => vec![*label],
        Terminator::CondBranch { true_label, false_label, .. } => {
            if true_label == false_label {
                vec![*true_label]
            } else {
                vec![*true_label, *false_label]
            }
        }
        Terminator::IndirectBranch { possible_targets, .. } => possible_targets.clone(),
        Terminator::Switch { cases, default, .. } => {
            let mut targets = vec![*default];
            for &(_, label) in cases {
                if !targets.contains(&label) {
                    targets.push(label);
                }
            }
            targets
        }
        Terminator::Return(_) | Terminator::Unreachable => Vec::new(),
    }
}

/// Remove promoted alloca, load, and store instructions.
fn remove_promoted_instructions(func: &mut IrFunction, alloca_to_idx: &FxHashMap<u32, usize>) {
    // Count the total number of allocas that are parameters (by position in entry block).
    // The first N allocas (where N = number of params) are parameter allocas.
    // We must NOT remove those because the backend's find_param_alloca uses positional indexing.
    let num_params = func.params.len();

    // Identify which promoted allocas are parameter allocas by their position
    let mut param_alloca_values: FxHashSet<u32> = FxHashSet::default();
    let mut alloca_count = 0;
    for inst in &func.blocks[0].instructions {
        if let Instruction::Alloca { dest, .. } = inst {
            if alloca_count < num_params {
                param_alloca_values.insert(dest.0);
            }
            alloca_count += 1;
        }
    }

    for block in &mut func.blocks {
        let has_spans = !block.source_spans.is_empty();
        if has_spans {
            let mut span_idx = 0;
            let insts = &block.instructions;
            block.source_spans.retain(|_| {
                let keep = match &insts[span_idx] {
                    Instruction::Alloca { dest, .. } => {
                        !alloca_to_idx.contains_key(&dest.0) || param_alloca_values.contains(&dest.0)
                    }
                    Instruction::Store { ptr, .. } => !alloca_to_idx.contains_key(&ptr.0),
                    Instruction::Load { ptr, .. } => !alloca_to_idx.contains_key(&ptr.0),
                    _ => true,
                };
                span_idx += 1;
                keep
            });
        }
        block.instructions.retain(|inst| {
            match inst {
                Instruction::Alloca { dest, .. } => {
                    // Keep parameter allocas, remove promoted non-parameter allocas
                    if alloca_to_idx.contains_key(&dest.0) && !param_alloca_values.contains(&dest.0) {
                        false // remove
                    } else {
                        true // keep
                    }
                }
                Instruction::Store { ptr, .. } => {
                    // Remove stores to promoted allocas
                    !alloca_to_idx.contains_key(&ptr.0)
                }
                Instruction::Load { ptr, .. } => {
                    // Loads to promoted allocas have already been replaced with Copy
                    // But there shouldn't be any left; just in case, keep them
                    !alloca_to_idx.contains_key(&ptr.0)
                }
                _ => true,
            }
        });
    }
}

/// Return the byte size for an IrType.
fn ir_type_size(ty: IrType) -> usize {
    match ty {
        IrType::I8 | IrType::U8 => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::I64 | IrType::U64 | IrType::F64 => 8,
        // Ptr: target-dependent (4 on ILP32, 8 on LP64)
        IrType::Ptr => crate::common::types::target_ptr_size(),
        IrType::I128 | IrType::U128 | IrType::F128 => 16,
        IrType::Void => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::AddressSpace;
    use crate::ir::reexports::{BasicBlock, CallInfo, IrBinOp, IrCmpOp, IrParam};

    /// Helper to build a simple function with one local variable.
    /// int f() { int x = 42; return x; }
    fn make_simple_function() -> IrFunction {
        let mut func = IrFunction::new("f".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                // %0 = alloca i32
                Instruction::Alloca { dest: Value(0), ty: IrType::I32, size: 4, align: 0, volatile: false },
                // store 42, %0
                Instruction::Store { val: Operand::Const(IrConst::I32(42)), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
                // %1 = load %0
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });
        func
    }

    #[test]
    fn test_simple_promotion() {
        let mut module = IrModule::new();
        module.functions.push(make_simple_function());
        promote_allocas(&mut module);

        let func = &module.functions[0];
        // The alloca should be removed, store removed, load replaced with copy
        let entry = &func.blocks[0];
        // Should have just a Copy instruction (load was replaced)
        assert!(entry.instructions.iter().any(|inst| matches!(inst, Instruction::Copy { .. })));
        // Should not have any Store to the promoted alloca
        assert!(!entry.instructions.iter().any(|inst|
            matches!(inst, Instruction::Store { ptr: Value(0), .. })
        ));
    }

    #[test]
    fn test_diamond_phi_insertion() {
        // int f(int cond) {
        //   int x;
        //   if (cond) x = 1; else x = 2;
        //   return x;
        // }
        let mut func = IrFunction::new(
            "f".into(),
            IrType::I32,
            vec![IrParam { ty: IrType::I32, struct_size: None, struct_align: None, struct_eightbyte_classes: Vec::new(), riscv_float_class: None }],
            false,
        );

        // entry: alloca for param, alloca for x, branch
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                // %0 = alloca i32 (param)
                Instruction::Alloca { dest: Value(0), ty: IrType::I32, size: 4, align: 0, volatile: false },
                // %1 = alloca i32 (x)
                Instruction::Alloca { dest: Value(1), ty: IrType::I32, size: 4, align: 0, volatile: false },
                // %2 = load %0 (read param)
                Instruction::Load { dest: Value(2), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
                // %3 = cmp ne %2, 0
                Instruction::Cmp {
                    dest: Value(3), op: IrCmpOp::Ne,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I32(0)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(3)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
            source_spans: Vec::new(),
        });

        // then: store 1 to x, branch to merge
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Store { val: Operand::Const(IrConst::I32(1)), ptr: Value(1), ty: IrType::I32,
                seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        // else: store 2 to x, branch to merge
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Store { val: Operand::Const(IrConst::I32(2)), ptr: Value(1), ty: IrType::I32,
                seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Branch(BlockId(3)),
            source_spans: Vec::new(),
        });

        // merge: load x, return
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![
                Instruction::Load { dest: Value(4), ptr: Value(1), ty: IrType::I32 , seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(4)))),
            source_spans: Vec::new(),
        });

        let mut module = IrModule::new();
        module.functions.push(func);
        promote_allocas(&mut module);

        let func = &module.functions[0];
        // The merge block should have a phi node
        let merge = &func.blocks[3];
        let has_phi = merge.instructions.iter().any(|inst| matches!(inst, Instruction::Phi { .. }));
        assert!(has_phi, "Expected phi node in merge block");

        // Verify phi has two incoming values
        if let Some(Instruction::Phi { incoming, .. }) = merge.instructions.first() {
            assert_eq!(incoming.len(), 2, "Phi should have 2 incoming values");
        }
    }

    #[test]
    fn test_non_promotable_address_taken() {
        // An alloca whose address is passed to a function should not be promoted
        let mut func = IrFunction::new("f".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Alloca { dest: Value(0), ty: IrType::I32, size: 4, align: 0, volatile: false },
                Instruction::Store { val: Operand::Const(IrConst::I32(42)), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
                // Pass address to a function (address-taken)
                Instruction::Call {
                    func: "use_ptr".into(),
                    info: CallInfo {
                        dest: None,
                        args: vec![Operand::Value(Value(0))],
                        arg_types: vec![IrType::Ptr],
                        return_type: IrType::Void,
                        is_variadic: false,
                        num_fixed_args: 1,
                        struct_arg_sizes: vec![None],
                        struct_arg_aligns: vec![],
                        struct_arg_classes: Vec::new(),
                        struct_arg_riscv_float_classes: Vec::new(),
                        is_sret: false,
                        is_fastcall: false,
                        ret_eightbyte_classes: Vec::new(),
                    },
                },
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });

        let mut module = IrModule::new();
        module.functions.push(func);
        promote_allocas(&mut module);

        // The alloca should NOT be promoted (address is taken by call)
        let func = &module.functions[0];
        let has_alloca = func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Alloca { dest: Value(0), .. })
        );
        assert!(has_alloca, "Address-taken alloca should not be promoted");
    }

    #[test]
    fn test_loop_phi() {
        // int f() { int sum = 0; for (int i = 0; i < 10; i++) sum += i; return sum; }
        let mut func = IrFunction::new("f".into(), IrType::I32, vec![], false);

        // entry: allocas, init, branch to loop header
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Alloca { dest: Value(0), ty: IrType::I32, size: 4, align: 0, volatile: false }, // sum
                Instruction::Alloca { dest: Value(1), ty: IrType::I32, size: 4, align: 0, volatile: false }, // i
                Instruction::Store { val: Operand::Const(IrConst::I32(0)), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
                Instruction::Store { val: Operand::Const(IrConst::I32(0)), ptr: Value(1), ty: IrType::I32 , seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // loop_header: load i, cmp, cond branch
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Load { dest: Value(2), ptr: Value(1), ty: IrType::I32 , seg_override: AddressSpace::Default },
                Instruction::Cmp {
                    dest: Value(3), op: IrCmpOp::Slt,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I32(10)),
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

        // loop_body: sum += i, i++, branch back
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Load { dest: Value(4), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
                Instruction::Load { dest: Value(5), ptr: Value(1), ty: IrType::I32 , seg_override: AddressSpace::Default },
                Instruction::BinOp {
                    dest: Value(6), op: IrBinOp::Add,
                    lhs: Operand::Value(Value(4)),
                    rhs: Operand::Value(Value(5)),
                    ty: IrType::I32,
                },
                Instruction::Store { val: Operand::Value(Value(6)), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
                Instruction::BinOp {
                    dest: Value(7), op: IrBinOp::Add,
                    lhs: Operand::Value(Value(5)),
                    rhs: Operand::Const(IrConst::I32(1)),
                    ty: IrType::I32,
                },
                Instruction::Store { val: Operand::Value(Value(7)), ptr: Value(1), ty: IrType::I32 , seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Branch(BlockId(1)),
            source_spans: Vec::new(),
        });

        // exit: load sum, return
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![
                Instruction::Load { dest: Value(8), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(8)))),
            source_spans: Vec::new(),
        });

        let mut module = IrModule::new();
        module.functions.push(func);
        promote_allocas(&mut module);

        let func = &module.functions[0];
        // loop_header should have phi nodes for sum and i
        let header = &func.blocks[1];
        let phi_count = header.instructions.iter()
            .filter(|inst| matches!(inst, Instruction::Phi { .. }))
            .count();
        assert_eq!(phi_count, 2, "Loop header should have 2 phi nodes");
    }

    #[test]
    fn test_volatile_alloca_not_promoted() {
        // A volatile alloca should never be promoted to SSA, even though
        // it is scalar and only used by loads/stores.
        let mut func = IrFunction::new("f".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                // %0 = alloca i32 (volatile)
                Instruction::Alloca { dest: Value(0), ty: IrType::I32, size: 4, align: 0, volatile: true },
                Instruction::Store { val: Operand::Const(IrConst::I32(42)), ptr: Value(0), ty: IrType::I32,
                seg_override: AddressSpace::Default },
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I32 , seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });

        let mut module = IrModule::new();
        module.functions.push(func);
        promote_allocas(&mut module);

        // The volatile alloca should NOT be promoted
        let func = &module.functions[0];
        let has_alloca = func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Alloca { dest: Value(0), volatile: true, .. })
        );
        assert!(has_alloca, "Volatile alloca should not be promoted");
        // Store should still exist
        let has_store = func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Store { ptr: Value(0), .. })
        );
        assert!(has_store, "Store to volatile alloca should not be removed");
    }

    #[test]
    fn test_dominator_computation() {
        // Simple diamond CFG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let succs = analysis::FlatAdj::from_vecs_usize(&[
            vec![1, 2], // 0
            vec![3],    // 1
            vec![3],    // 2
            vec![],     // 3
        ]);
        let preds = analysis::FlatAdj::from_vecs_usize(&[
            vec![],     // 0
            vec![0],    // 1
            vec![0],    // 2
            vec![1, 2], // 3
        ]);
        let idom = analysis::compute_dominators(4, &preds, &succs);
        assert_eq!(idom[0], 0); // entry dominates itself
        assert_eq!(idom[1], 0); // 0 dominates 1
        assert_eq!(idom[2], 0); // 0 dominates 2
        assert_eq!(idom[3], 0); // 0 dominates 3 (join point)
    }

    #[test]
    fn test_dominance_frontier() {
        // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let preds = analysis::FlatAdj::from_vecs_usize(&[
            vec![],     // 0
            vec![0],    // 1
            vec![0],    // 2
            vec![1, 2], // 3
        ]);
        let idom = vec![0, 0, 0, 0];
        let df = analysis::compute_dominance_frontiers(4, &preds, &idom);
        // DF(1) = {3}, DF(2) = {3}
        assert!(df[1].contains(&3));
        assert!(df[2].contains(&3));
        assert!(df[0].is_empty());
        assert!(df[3].is_empty());
    }

    #[test]
    fn test_inline_asm_output_promotion() {
        // Test that mem2reg promotes an alloca used as an InlineAsm output
        // with a register constraint ("=r"). The alloca should be replaced
        // with a fresh SSA value in the InlineAsm output.
        //
        // Pattern: unsigned long __ptr;
        //          asm("" : "=r"(__ptr) : "0"(addr));
        //          return __ptr;
        //
        // Before mem2reg:
        //   %0 = alloca i64
        //   inline_asm outputs=[("=r", %0)] inputs=[("0", ...)]
        //   %1 = load %0
        //   ret %1
        //
        // After mem2reg:
        //   inline_asm outputs=[("=r", %fresh)] inputs=[...]
        //   %1 = copy Value(%fresh)
        //   ret %1
        let mut func = IrFunction::new("test_asm_promote".into(), IrType::I64, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                // %0 = alloca i64
                Instruction::Alloca { dest: Value(0), ty: IrType::I64, size: 8, align: 8, volatile: false },
                // inline_asm outputs=[("=r", %0)]
                Instruction::InlineAsm {
                    template: String::new(),
                    outputs: vec![("=r".to_string(), Value(0), None)],
                    inputs: vec![("0".to_string(), Operand::Const(IrConst::I64(100)), None)],
                    clobbers: vec![],
                    operand_types: vec![IrType::I64],
                    goto_labels: vec![],
                    input_symbols: vec![],
                    seg_overrides: vec![AddressSpace::Default],
                },
                // %1 = load %0
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I64,
                    seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });
        func.next_value_id = 2;

        let mut module = IrModule::new();
        module.functions.push(func);
        promote_allocas(&mut module);

        let func = &module.functions[0];
        // The alloca should be gone (promoted)
        let has_alloca = func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Alloca { .. })
        );
        assert!(!has_alloca, "Alloca should be promoted when used only as InlineAsm output");

        // The InlineAsm should now have a fresh SSA value as output (not the alloca)
        let asm_output = func.blocks[0].instructions.iter().find_map(|inst| {
            if let Instruction::InlineAsm { outputs, .. } = inst {
                Some(outputs[0].1.0) // Value ID of first output
            } else {
                None
            }
        });
        assert!(asm_output.is_some(), "InlineAsm instruction should exist");
        let fresh_id = asm_output.unwrap();
        assert_ne!(fresh_id, 0, "InlineAsm output should be a fresh value, not the original alloca");

        // The load should be replaced with a Copy from the fresh value
        let has_copy_from_fresh = func.blocks[0].instructions.iter().any(|inst| {
            if let Instruction::Copy { src: Operand::Value(v), .. } = inst {
                v.0 == fresh_id
            } else {
                false
            }
        });
        assert!(has_copy_from_fresh, "Load should be replaced with Copy from fresh InlineAsm output");

        // No Load instruction should remain (it was promoted to Copy)
        let has_load = func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Load { .. })
        );
        assert!(!has_load, "Load from promoted alloca should be removed");
    }

    #[test]
    fn test_inline_asm_memory_constraint_not_promoted() {
        // InlineAsm with memory/address constraints ("+m", "+A") should NOT
        // be promoted because the alloca address is passed as a Value operand.
        //
        // Pattern: asm("csrrw %0, satp, %1" : "+m"(*ptr) : ...)
        //
        // The alloca's address is taken (used as input Value), so it must stay.
        let mut func = IrFunction::new("test_asm_no_promote".into(), IrType::I64, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                // %0 = alloca i64
                Instruction::Alloca { dest: Value(0), ty: IrType::I64, size: 8, align: 8, volatile: false },
                // store 42, %0
                Instruction::Store {
                    val: Operand::Const(IrConst::I64(42)),
                    ptr: Value(0),
                    ty: IrType::I64,
                    seg_override: AddressSpace::Default,
                },
                // inline_asm outputs=[("=m", %0)] inputs=[("m", Value(%0))]
                // The input uses the alloca's address (Value(%0)), which means address-taken
                Instruction::InlineAsm {
                    template: "nop".to_string(),
                    outputs: vec![("=r".to_string(), Value(0), None)],
                    inputs: vec![("m".to_string(), Operand::Value(Value(0)), None)],
                    clobbers: vec![],
                    operand_types: vec![IrType::I64],
                    goto_labels: vec![],
                    input_symbols: vec![],
                    seg_overrides: vec![AddressSpace::Default, AddressSpace::Default],
                },
                // %1 = load %0
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I64,
                    seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });
        func.next_value_id = 2;

        let mut module = IrModule::new();
        module.functions.push(func);
        promote_allocas(&mut module);

        let func = &module.functions[0];
        // The alloca should still exist (NOT promoted because address is taken via input)
        let has_alloca = func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Alloca { .. })
        );
        assert!(has_alloca, "Alloca should NOT be promoted when its address is used as InlineAsm input Value");
    }

    #[test]
    fn test_inline_asm_memory_output_only_not_promoted() {
        // InlineAsm with "=m" output-only constraint should NOT be promoted.
        // The alloca appears ONLY in outputs (not inputs), but the inline asm
        // writes directly to the alloca's stack memory. Promoting it would cause
        // the backend to lose the stack address, resulting in writes to garbage.
        //
        // Pattern: asm("mov %1, %0" : "=m"(result) : "r"(value))
        let mut func = IrFunction::new("test_asm_mem_output_only".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                // %0 = alloca i32 (for "=m" output)
                Instruction::Alloca { dest: Value(0), ty: IrType::I32, size: 4, align: 4, volatile: false },
                // inline_asm outputs=[("=m", %0)] inputs=[("r", const 42)]
                // Output is memory-only; alloca must NOT be promoted.
                Instruction::InlineAsm {
                    template: "movl $42, %0".to_string(),
                    outputs: vec![("=m".to_string(), Value(0), None)],
                    inputs: vec![],
                    clobbers: vec![],
                    operand_types: vec![IrType::I32],
                    goto_labels: vec![],
                    input_symbols: vec![],
                    seg_overrides: vec![AddressSpace::Default],
                },
                // %1 = load %0
                Instruction::Load { dest: Value(1), ptr: Value(0), ty: IrType::I32,
                    seg_override: AddressSpace::Default },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: Vec::new(),
        });
        func.next_value_id = 2;

        let mut module = IrModule::new();
        module.functions.push(func);
        promote_allocas(&mut module);

        let func = &module.functions[0];
        // The alloca MUST still exist — it was NOT promoted because the "=m"
        // constraint means the inline asm writes directly to stack memory.
        let has_alloca = func.blocks[0].instructions.iter().any(|inst|
            matches!(inst, Instruction::Alloca { .. })
        );
        assert!(has_alloca, "Alloca should NOT be promoted when used as =m inline asm output");
    }
}
