//! Liveness analysis for IR values.
//!
//! Computes live intervals for each IR value in an IrFunction. A live interval
//! represents the range [def_point, last_use_point] where a value is live and
//! needs to be preserved (either in a register or a stack slot).
//!
//! The analysis supports loops via backward dataflow iteration:
//! 1. First, assign sequential program points to all instructions and terminators.
//! 2. Run backward dataflow to compute live-in/live-out sets for each block.
//!    This correctly handles values that are live across loop back-edges.
//! 3. Build intervals by taking the union of def/use points and live-through blocks.
//!
//! ## Performance
//!
//! The dataflow uses compact bitsets instead of hash sets for gen/kill/live_in/live_out.
//! Value IDs are remapped to a dense [0..N) range so bitsets are minimal size.
//! This eliminates per-iteration heap allocation and replaces hash-table operations
//! with fast word-level bitwise ops (union = OR, difference = AND-NOT, equality = ==).

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::reexports::{
    Instruction,
    IrBinOp,
    IrConst,
    IrFunction,
    Operand,
    Terminator,
    Value,
};

/// A live interval for an IR value: [start, end] in program point numbering.
/// start = the point where the value is defined
/// end = the last point where the value is used
#[derive(Debug, Clone, Copy)]
pub struct LiveInterval {
    pub start: u32,
    pub end: u32,
    pub value_id: u32,
}

/// Result of liveness analysis: maps value IDs to their live intervals.
pub struct LivenessResult {
    pub intervals: Vec<LiveInterval>,
    /// Program points that are Call or CallIndirect instructions.
    /// Used by the register allocator to identify values that cross call boundaries.
    pub call_points: Vec<u32>,
    /// Loop nesting depth for each block (block_index -> depth).
    /// Depth 0 = not in any loop. Depth 1 = in one loop. Depth 2 = nested, etc.
    /// Used by the register allocator to weight uses inside loops more heavily.
    pub block_loop_depth: Vec<u32>,
}

// ── Compact bitset for dataflow ──────────────────────────────────────────────

/// A compact bitset stored as a contiguous slice of u64 words.
/// Supports O(1) insert/contains and O(n/64) union/difference/equality.
#[derive(Clone)]
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    /// Create a new empty bitset that can hold indices [0..num_bits).
    fn new(num_bits: usize) -> Self {
        let num_words = num_bits.div_ceil(64);
        Self { words: vec![0u64; num_words] }
    }

    #[inline(always)]
    fn insert(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.words[word] |= 1u64 << bit;
    }

    #[inline(always)]
    fn contains(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.words[word] >> bit) & 1 != 0
    }

    /// self = self | other. Returns true if self changed.
    fn union_with(&mut self, other: &BitSet) -> bool {
        let mut changed = false;
        for (w, o) in self.words.iter_mut().zip(other.words.iter()) {
            let old = *w;
            *w |= *o;
            changed |= *w != old;
        }
        changed
    }

    /// Computes: self = gen ∪ (out - kill) in one pass. Returns true if self changed.
    fn assign_gen_union_out_minus_kill(&mut self, gen: &BitSet, out: &BitSet, kill: &BitSet) -> bool {
        let mut changed = false;
        for i in 0..self.words.len() {
            let new_val = gen.words[i] | (out.words[i] & !kill.words[i]);
            if new_val != self.words[i] {
                self.words[i] = new_val;
                changed = true;
            }
        }
        changed
    }

    /// Iterate over all set bits, calling f(bit_index) for each.
    fn for_each_set_bit(&self, mut f: impl FnMut(usize)) {
        for (word_idx, &word) in self.words.iter().enumerate() {
            if word == 0 { continue; }
            let base = word_idx * 64;
            let mut w = word;
            while w != 0 {
                let tz = w.trailing_zeros() as usize;
                f(base + tz);
                w &= w - 1; // clear lowest set bit
            }
        }
    }

    /// Clear all bits.
    fn clear(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }
}

/// Intermediate state built during Phase 1 (program point assignment and gen/kill).
struct ProgramPointState {
    block_start_points: Vec<u32>,
    block_end_points: Vec<u32>,
    def_points: Vec<u32>,
    last_use_points: Vec<u32>,
    block_gen: Vec<BitSet>,
    block_kill: Vec<BitSet>,
    block_id_to_idx: FxHashMap<u32, usize>,
    setjmp_block_indices: Vec<usize>,
    call_points: Vec<u32>,
    num_points: u32,
}

/// Compute live intervals for all non-alloca values in a function.
///
/// Uses backward dataflow analysis to correctly handle loops:
/// - live_in[B] = gen[B] ∪ (live_out[B] - kill[B])
/// - live_out[B] = ∪ live_in[S] for all successors S of B
///
/// Values that are live-in to a block have their interval extended to cover
/// from the block's start point through the entire block. This correctly
/// extends intervals through loop back-edges.
pub fn compute_live_intervals(func: &IrFunction) -> LivenessResult {
    let num_blocks = func.blocks.len();
    if num_blocks == 0 {
        return LivenessResult { intervals: Vec::new(), call_points: Vec::new(), block_loop_depth: Vec::new() };
    }

    let alloca_set = collect_alloca_set(func);
    let (value_ids, id_to_dense) = build_dense_value_map(func, &alloca_set);

    let num_values = value_ids.len();
    if num_values == 0 {
        return LivenessResult { intervals: Vec::new(), call_points: Vec::new(), block_loop_depth: Vec::new() };
    }

    // Phase 1: Assign program points and build gen/kill sets.
    let mut ps = assign_program_points(func, num_blocks, num_values, &alloca_set, &id_to_dense);

    // Phase 1b: Extend liveness of GEP base values for GEP folding.
    extend_gep_base_liveness(func, &alloca_set, &id_to_dense,
        &mut ps.last_use_points, &mut ps.block_gen);

    // Phase 1c: Extend liveness for F128 source pointers.
    extend_f128_source_liveness(func, &alloca_set, &id_to_dense,
        &mut ps.last_use_points, &mut ps.block_gen);

    // Phase 2: Build successor lists for the CFG.
    let successors = build_successor_lists(func, num_blocks, &ps.block_id_to_idx);

    // Phase 2b: Compute loop nesting depth per block.
    let block_loop_depth = compute_loop_depth(&successors, num_blocks);

    // Phase 3: Backward dataflow to compute live-in/live-out per block.
    let (live_in, live_out) = run_backward_dataflow(
        num_blocks, num_values, &successors, &ps.block_gen, &ps.block_kill,
    );

    // Phase 4: Extend intervals for values that are live-in or live-out of blocks.
    extend_intervals_from_liveness(
        num_blocks, &live_in, &live_out,
        &ps.block_start_points, &ps.block_end_points,
        &mut ps.def_points, &mut ps.last_use_points,
    );

    // Phase 4b: Handle setjmp/longjmp.
    extend_intervals_for_setjmp(
        &ps.setjmp_block_indices, ps.num_points, &live_in, &live_out,
        &mut ps.last_use_points,
    );

    // Phase 5: Build and sort intervals.
    let intervals = build_intervals(&value_ids, &ps.def_points, &ps.last_use_points);

    LivenessResult {
        intervals,
        call_points: ps.call_points,
        block_loop_depth,
    }
}

/// Collect alloca values (not register-allocatable).
fn collect_alloca_set(func: &IrFunction) -> FxHashSet<u32> {
    let mut alloca_set: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, .. } = inst {
                alloca_set.insert(dest.0);
            }
        }
    }
    alloca_set
}

/// Collect all non-alloca value IDs and build a dense remapping:
/// sparse value_id -> dense index in [0..num_values).
fn build_dense_value_map(func: &IrFunction, alloca_set: &FxHashSet<u32>) -> (Vec<u32>, FxHashMap<u32, usize>) {
    let mut value_ids: Vec<u32> = Vec::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();

    let maybe_add = |id: u32, alloca_set: &FxHashSet<u32>, seen: &mut FxHashSet<u32>, value_ids: &mut Vec<u32>| {
        if !alloca_set.contains(&id) && seen.insert(id) {
            value_ids.push(id);
        }
    };

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                maybe_add(dest.0, alloca_set, &mut seen, &mut value_ids);
            }
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    maybe_add(v.0, alloca_set, &mut seen, &mut value_ids);
                }
            });
            for_each_value_use_in_instruction(inst, |v| {
                maybe_add(v.0, alloca_set, &mut seen, &mut value_ids);
            });
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                maybe_add(v.0, alloca_set, &mut seen, &mut value_ids);
            }
        });
    }

    let mut id_to_dense: FxHashMap<u32, usize> = FxHashMap::default();
    id_to_dense.reserve(value_ids.len());
    for (dense_idx, &vid) in value_ids.iter().enumerate() {
        id_to_dense.insert(vid, dense_idx);
    }

    (value_ids, id_to_dense)
}

/// Phase 1: Assign sequential program points to all instructions/terminators
/// and build per-block gen/kill bitsets, def/use point arrays, call points,
/// and setjmp block indices.
fn assign_program_points(
    func: &IrFunction,
    num_blocks: usize,
    num_values: usize,
    alloca_set: &FxHashSet<u32>,
    id_to_dense: &FxHashMap<u32, usize>,
) -> ProgramPointState {
    let mut point: u32 = 0;
    let mut block_start_points: Vec<u32> = Vec::with_capacity(num_blocks);
    let mut block_end_points: Vec<u32> = Vec::with_capacity(num_blocks);
    let mut def_points: Vec<u32> = vec![u32::MAX; num_values];
    let mut last_use_points: Vec<u32> = vec![u32::MAX; num_values];
    let mut block_gen: Vec<BitSet> = Vec::with_capacity(num_blocks);
    let mut block_kill: Vec<BitSet> = Vec::with_capacity(num_blocks);
    let mut block_id_to_idx: FxHashMap<u32, usize> = FxHashMap::default();
    let mut setjmp_block_indices: Vec<usize> = Vec::new();
    let mut call_points: Vec<u32> = Vec::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        block_id_to_idx.insert(block.label.0, block_idx);
        let block_start = point;
        block_start_points.push(block_start);
        let mut gen = BitSet::new(num_values);
        let mut kill = BitSet::new(num_values);

        for inst in &block.instructions {
            if is_returns_twice_call(inst) {
                setjmp_block_indices.push(block_idx);
            }

            // Track call instruction program points for register allocation.
            // InlineAsm instructions with register operands are treated as call
            // points because they may clobber caller-saved registers (r8-r11 on
            // x86). This ensures values whose live ranges span inline asm get
            // callee-saved registers (which survive inline asm), while values NOT
            // spanning inline asm can safely use caller-saved registers.
            //
            // Empty inline asm barriers (e.g., `asm volatile("" ::: "memory")`)
            // are NOT call points since they don't use any GP registers. These
            // are common in the kernel for memory barriers, preempt_disable/enable,
            // etc. Treating them as call points would unnecessarily force values
            // into callee-saved registers across simple barriers.
            match inst {
                Instruction::Call { .. } | Instruction::CallIndirect { .. } => {
                    call_points.push(point);
                }
                Instruction::InlineAsm { outputs, inputs, .. } => {
                    // Only treat as call point if the asm has register operands
                    // (outputs or inputs that bind to GP registers).
                    if !outputs.is_empty() || !inputs.is_empty() {
                        call_points.push(point);
                    }
                }
                // Memcpy uses rep movsb which clobbers rdi, rsi, rcx.
                // VaArg/VaCopy/VaArgStruct clobber rdi/rsi for struct copy.
                // VaStart is included conservatively for safety.
                // Treat these as call points so caller-saved registers (including
                // rdi/rsi) are not allocated across them.
                Instruction::Memcpy { .. }
                | Instruction::VaArg { .. }
                | Instruction::VaStart { .. }
                | Instruction::VaCopy { .. }
                | Instruction::VaArgStruct { .. } => {
                    call_points.push(point);
                }
                // i128 div/rem emit implicit calls to __divti3/__udivti3/__modti3/__umodti3.
                // These are BinOp instructions at the IR level but generate `call` at the
                // assembly level, clobbering all caller-saved registers. We must treat them
                // as call points so the register allocator doesn't assign caller-saved
                // registers to values whose live ranges span these operations.
                Instruction::BinOp { op, ty, .. }
                    if matches!(ty, IrType::I128 | IrType::U128)
                        && matches!(op, IrBinOp::SDiv | IrBinOp::UDiv | IrBinOp::SRem | IrBinOp::URem) =>
                {
                    call_points.push(point);
                }
                // i128 <-> float casts emit implicit calls to compiler-rt helpers
                // (__floattidf/__fixdfti/etc.). Same reasoning as i128 div/rem above.
                Instruction::Cast { from_ty, to_ty, .. }
                    if (matches!(from_ty, IrType::I128 | IrType::U128) && to_ty.is_float())
                        || (from_ty.is_float() && matches!(to_ty, IrType::I128 | IrType::U128)) =>
                {
                    call_points.push(point);
                }
                _ => {}
            }

            record_instruction_uses_dense(inst, point, alloca_set, id_to_dense, &mut last_use_points);

            // Record InlineAsm output definitions BEFORE gen collection so
            // that promoted (non-alloca) outputs are in the kill set and won't
            // be treated as upward-exposed uses.
            //
            // Only kill output values that are first defined here (promoted asm
            // outputs). Output values already defined earlier (e.g., pointer
            // values passed through for indirect stores like `"=a"(*ptr)`) are
            // merely *used* by the InlineAsm — the asm reads the pointer from
            // the slot and stores through it, but does not overwrite the pointer
            // itself. Killing such values truncates their live interval, letting
            // the slot packer reuse their slot too early, which corrupts the
            // pointer on the next loop iteration.
            if let Instruction::InlineAsm { outputs, .. } = inst {
                for (_, out_val, _) in outputs {
                    if !alloca_set.contains(&out_val.0) {
                        if let Some(&dense) = id_to_dense.get(&out_val.0) {
                            if def_points[dense] == u32::MAX {
                                def_points[dense] = point;
                                kill.insert(dense);
                            }
                        }
                    }
                }
            }

            collect_instruction_gen_dense(inst, alloca_set, id_to_dense, &kill, &mut gen);

            if let Some(dest) = inst.dest() {
                if !alloca_set.contains(&dest.0) {
                    if let Some(&dense) = id_to_dense.get(&dest.0) {
                        if def_points[dense] == u32::MAX {
                            def_points[dense] = point;
                        }
                        kill.insert(dense);
                    }
                }
            }

            point += 1;
        }

        record_terminator_uses_dense(&block.terminator, point, alloca_set, id_to_dense, &mut last_use_points);
        collect_terminator_gen_dense(&block.terminator, alloca_set, id_to_dense, &kill, &mut gen);
        let block_end = point;
        block_end_points.push(block_end);
        point += 1;

        block_gen.push(gen);
        block_kill.push(kill);
    }

    ProgramPointState {
        block_start_points,
        block_end_points,
        def_points,
        last_use_points,
        block_gen,
        block_kill,
        block_id_to_idx,
        setjmp_block_indices,
        call_points,
        num_points: point,
    }
}

/// Phase 2: Build successor lists from block terminators and asm goto labels.
fn build_successor_lists(
    func: &IrFunction,
    num_blocks: usize,
    block_id_to_idx: &FxHashMap<u32, usize>,
) -> Vec<Vec<usize>> {
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    for (idx, block) in func.blocks.iter().enumerate() {
        for target_id in terminator_targets(&block.terminator) {
            if let Some(&target_idx) = block_id_to_idx.get(&target_id) {
                successors[idx].push(target_idx);
            }
        }
        // InlineAsm goto_labels are implicit control flow edges.
        for inst in &block.instructions {
            if let Instruction::InlineAsm { goto_labels, .. } = inst {
                for (_, label) in goto_labels {
                    if let Some(&target_idx) = block_id_to_idx.get(&label.0) {
                        if !successors[idx].contains(&target_idx) {
                            successors[idx].push(target_idx);
                        }
                    }
                }
            }
        }
    }
    successors
}

/// Phase 3: Backward dataflow to compute live-in/live-out per block.
/// live_in[B] = gen[B] ∪ (live_out[B] - kill[B])
/// live_out[B] = ∪ live_in[S] for all successors S of B
fn run_backward_dataflow(
    num_blocks: usize,
    num_values: usize,
    successors: &[Vec<usize>],
    block_gen: &[BitSet],
    block_kill: &[BitSet],
) -> (Vec<BitSet>, Vec<BitSet>) {
    let mut live_in: Vec<BitSet> = (0..num_blocks).map(|_| BitSet::new(num_values)).collect();
    let mut live_out: Vec<BitSet> = (0..num_blocks).map(|_| BitSet::new(num_values)).collect();
    let mut tmp_out = BitSet::new(num_values);

    // Iterate until fixpoint (backward order converges faster).
    // MAX_ITERATIONS is a safety bound for pathological irreducible control flow.
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: u32 = 50;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;

        for idx in (0..num_blocks).rev() {
            tmp_out.clear();
            for &succ in &successors[idx] {
                tmp_out.union_with(&live_in[succ]);
            }

            if tmp_out.words != live_out[idx].words {
                live_out[idx].words.copy_from_slice(&tmp_out.words);
                changed = true;
            }

            let in_changed = live_in[idx].assign_gen_union_out_minus_kill(
                &block_gen[idx], &live_out[idx], &block_kill[idx]
            );
            changed |= in_changed;
        }
    }

    (live_in, live_out)
}

/// Phase 4: Extend intervals for values that are live-in or live-out of blocks.
/// A value live-in to a block has its interval cover the entire block.
fn extend_intervals_from_liveness(
    num_blocks: usize,
    live_in: &[BitSet],
    live_out: &[BitSet],
    block_start_points: &[u32],
    block_end_points: &[u32],
    def_points: &mut [u32],
    last_use_points: &mut [u32],
) {
    for idx in 0..num_blocks {
        let start = block_start_points[idx];
        let end = block_end_points[idx];

        live_in[idx].for_each_set_bit(|dense_idx| {
            let def_entry = &mut def_points[dense_idx];
            if *def_entry == u32::MAX || start < *def_entry {
                *def_entry = start;
            }
            let entry = &mut last_use_points[dense_idx];
            if *entry == u32::MAX {
                *entry = start;
            }
            if end > *entry {
                *entry = end;
            }
        });

        live_out[idx].for_each_set_bit(|dense_idx| {
            let def_entry = &mut def_points[dense_idx];
            if *def_entry == u32::MAX || start < *def_entry {
                *def_entry = start;
            }
            let entry = &mut last_use_points[dense_idx];
            if *entry == u32::MAX {
                *entry = end;
            }
            if end > *entry {
                *entry = end;
            }
        });
    }
}

/// Phase 4b: Handle setjmp/longjmp — extend intervals for values live at
/// setjmp call points to the end of the function, preventing slot reuse.
fn extend_intervals_for_setjmp(
    setjmp_block_indices: &[usize],
    num_points: u32,
    live_in: &[BitSet],
    live_out: &[BitSet],
    last_use_points: &mut [u32],
) {
    if setjmp_block_indices.is_empty() {
        return;
    }
    let func_end = num_points.saturating_sub(1);
    for &sjb in setjmp_block_indices {
        live_in[sjb].for_each_set_bit(|dense_idx| {
            let entry = &mut last_use_points[dense_idx];
            if *entry == u32::MAX || func_end > *entry {
                *entry = func_end;
            }
        });
        live_out[sjb].for_each_set_bit(|dense_idx| {
            let entry = &mut last_use_points[dense_idx];
            if *entry == u32::MAX || func_end > *entry {
                *entry = func_end;
            }
        });
    }
}

/// Phase 5: Build sorted live intervals from def/use point arrays.
fn build_intervals(value_ids: &[u32], def_points: &[u32], last_use_points: &[u32]) -> Vec<LiveInterval> {
    let mut intervals: Vec<LiveInterval> = Vec::new();
    for (dense_idx, &vid) in value_ids.iter().enumerate() {
        let start = def_points[dense_idx];
        if start == u32::MAX { continue; }
        let end = last_use_points[dense_idx];
        let end = if end == u32::MAX { start } else { end.max(start) };
        intervals.push(LiveInterval { start, end, value_id: vid });
    }
    intervals.sort_unstable_by_key(|iv| iv.start);
    intervals
}

/// Extend liveness of GEP base values so that their registers remain valid
/// at Load/Store use points where the GEP offset can be folded into the
/// addressing mode.
///
/// For each GEP `%gep = gep %base, const_offset` whose result is only used
/// as a Load/Store ptr operand:
/// - Find all Load/Store instructions that use %gep as their ptr
/// - Record %base as "used" at those instruction program points
/// - Update the gen bitset for the block containing the Load/Store
///
/// This ensures the register allocator keeps %base alive through the folded
/// Load/Store, enabling safe `offset(%base_reg)` addressing at codegen time.
fn extend_gep_base_liveness(
    func: &IrFunction,
    alloca_set: &FxHashSet<u32>,
    id_to_dense: &FxHashMap<u32, usize>,
    last_use_points: &mut [u32],
    block_gen: &mut [BitSet],
) {
    // Phase A: Identify foldable GEPs with non-alloca bases.
    // Same criteria as build_gep_fold_map in generation.rs.
    let mut gep_info: FxHashMap<u32, (u32, i64)> = FxHashMap::default(); // gep_dest_id -> (base_id, offset)

    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::GetElementPtr { dest, base, offset: Operand::Const(c), .. } = inst {
                // Skip alloca bases (already handled by existing fold logic)
                if alloca_set.contains(&base.0) {
                    continue;
                }
                let offset_val = match c {
                    IrConst::I64(n) => *n,
                    IrConst::I32(n) => *n as i64,
                    IrConst::I16(n) => *n as i64,
                    IrConst::I8(n) => *n as i64,
                    _ => continue,
                };
                if offset_val >= i32::MIN as i64 && offset_val <= i32::MAX as i64 {
                    gep_info.insert(dest.0, (base.0, offset_val));
                }
            }
        }
    }

    if gep_info.is_empty() {
        return;
    }

    // Phase B: Verify each GEP dest is only used as Load/Store ptr operand.
    // If used elsewhere, remove from the map (not foldable).
    let mut non_foldable: FxHashSet<u32> = FxHashSet::default();

    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::Load { ptr, ty, .. } => {
                    // Load.ptr is foldable unless i128
                    if matches!(ty, IrType::I128 | IrType::U128)
                        && gep_info.contains_key(&ptr.0) {
                            non_foldable.insert(ptr.0);
                        }
                }
                Instruction::Store { val, ptr, ty, .. } => {
                    // Store.val is NOT foldable; Store.ptr is (unless i128)
                    if let Operand::Value(v) = val {
                        if gep_info.contains_key(&v.0) {
                            non_foldable.insert(v.0);
                        }
                    }
                    if matches!(ty, IrType::I128 | IrType::U128)
                        && gep_info.contains_key(&ptr.0) {
                            non_foldable.insert(ptr.0);
                        }
                }
                _ => {
                    // Any other use invalidates folding
                    for_each_operand_in_instruction(inst, |op| {
                        if let Operand::Value(v) = op {
                            if gep_info.contains_key(&v.0) {
                                non_foldable.insert(v.0);
                            }
                        }
                    });
                    for_each_value_use_in_instruction(inst, |v| {
                        if gep_info.contains_key(&v.0) {
                            non_foldable.insert(v.0);
                        }
                    });
                }
            }
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                if gep_info.contains_key(&v.0) {
                    non_foldable.insert(v.0);
                }
            }
        });
    }

    for id in &non_foldable {
        gep_info.remove(id);
    }

    if gep_info.is_empty() {
        return;
    }

    // Phase C: Extend base liveness to Load/Store points that use foldable GEP results.
    let mut block_point: u32 = 0;
    for (bi, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            match inst {
                Instruction::Load { .. } | Instruction::Store { .. } => {
                    let ptr_id = match inst {
                        Instruction::Load { ptr, .. } => ptr.0,
                        Instruction::Store { ptr, .. } => ptr.0,
                        _ => unreachable!("GEP analysis matched non-Load/Store instruction"),
                    };
                    if let Some(&(base_id, _offset)) = gep_info.get(&ptr_id) {
                        // Extend base's last_use to this program point
                        if !alloca_set.contains(&base_id) {
                            if let Some(&dense) = id_to_dense.get(&base_id) {
                                let entry = &mut last_use_points[dense];
                                if *entry == u32::MAX || block_point > *entry {
                                    *entry = block_point;
                                }
                                // Also add to block's gen set (the base is "used" here)
                                block_gen[bi].insert(dense);
                            }
                        }
                    }
                }
                _ => {}
            }
            block_point += 1;
        }
        // Account for terminator point
        block_point += 1;
    }
}

/// Extend liveness for F128 load source pointers.
///
/// When an F128 value is loaded from memory, the codegen records which pointer
/// was used (via `track_f128_load` in state.rs). Later, during Call emission,
/// `emit_f128_operand_to_a0_a1` reads the pointer back from its stack slot to
/// reload the full 128-bit value. This creates an implicit dependency: the
/// pointer must remain live until the last use of the F128 dest value.
///
/// Without this extension, the Tier 2 liveness analysis considers the pointer
/// dead after the Load instruction, allowing the register allocator (and
/// subsequently the Tier 3 slot allocator) to reuse its slot. If another value
/// is placed in that slot before the Call, the pointer is corrupted and the
/// Call dereferences garbage (typically causing SIGSEGV).
fn extend_f128_source_liveness(
    func: &IrFunction,
    alloca_set: &FxHashSet<u32>,
    id_to_dense: &FxHashMap<u32, usize>,
    last_use_points: &mut [u32],
    block_gen: &mut [BitSet],
) {
    // Collect (ptr_id, dest_id) pairs for F128 loads with non-alloca pointers.
    let mut f128_loads: Vec<(u32, u32)> = Vec::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Load { dest, ptr, ty, .. } = inst {
                if *ty == IrType::F128 && !alloca_set.contains(&ptr.0) {
                    f128_loads.push((ptr.0, dest.0));
                }
            }
        }
    }

    if f128_loads.is_empty() {
        return;
    }

    // Extend each pointer's last_use_point to match its dest's last_use_point.
    for &(ptr_id, dest_id) in &f128_loads {
        let dest_dense = id_to_dense.get(&dest_id).copied();
        let ptr_dense = id_to_dense.get(&ptr_id).copied();
        if let (Some(dd), Some(pd)) = (dest_dense, ptr_dense) {
            let dest_last = last_use_points[dd];
            if dest_last != u32::MAX {
                let ptr_entry = &mut last_use_points[pd];
                if *ptr_entry == u32::MAX || dest_last > *ptr_entry {
                    *ptr_entry = dest_last;
                }
            }
        }
    }

    // Update gen sets so backward dataflow propagation keeps the pointer live
    // in predecessor blocks when the dest value is used in a successor block.
    for (bi, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            let mut check_use = |vid: u32| {
                for &(ptr_id, dest_id) in &f128_loads {
                    if vid == dest_id {
                        if let Some(&pd) = id_to_dense.get(&ptr_id) {
                            block_gen[bi].insert(pd);
                        }
                    }
                }
            };
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    check_use(v.0);
                }
            });
            for_each_value_use_in_instruction(inst, |v| {
                check_use(v.0);
            });
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                for &(ptr_id, dest_id) in &f128_loads {
                    if v.0 == dest_id {
                        if let Some(&pd) = id_to_dense.get(&ptr_id) {
                            block_gen[bi].insert(pd);
                        }
                    }
                }
            }
        });
    }
}

/// Record uses of operands in an instruction (dense index version).
fn record_instruction_uses_dense(
    inst: &Instruction,
    point: u32,
    alloca_set: &FxHashSet<u32>,
    id_to_dense: &FxHashMap<u32, usize>,
    last_use: &mut [u32],
) {
    let mut record = |vid: u32| {
        if !alloca_set.contains(&vid) {
            if let Some(&dense) = id_to_dense.get(&vid) {
                let entry = &mut last_use[dense];
                if *entry == u32::MAX || point > *entry {
                    *entry = point;
                }
            }
        }
    };

    for_each_operand_in_instruction(inst, |op| {
        if let Operand::Value(v) = op {
            record(v.0);
        }
    });

    for_each_value_use_in_instruction(inst, |v| {
        record(v.0);
    });
}

/// Record uses in a terminator (dense index version).
fn record_terminator_uses_dense(
    term: &Terminator,
    point: u32,
    alloca_set: &FxHashSet<u32>,
    id_to_dense: &FxHashMap<u32, usize>,
    last_use: &mut [u32],
) {
    for_each_operand_in_terminator(term, |op| {
        if let Operand::Value(v) = op {
            if !alloca_set.contains(&v.0) {
                if let Some(&dense) = id_to_dense.get(&v.0) {
                    let entry = &mut last_use[dense];
                    if *entry == u32::MAX || point > *entry {
                        *entry = point;
                    }
                }
            }
        }
    });
}

/// Collect gen set for a block's instruction (dense bitset version).
fn collect_instruction_gen_dense(
    inst: &Instruction,
    alloca_set: &FxHashSet<u32>,
    id_to_dense: &FxHashMap<u32, usize>,
    kill: &BitSet,
    gen: &mut BitSet,
) {
    let mut add_use = |vid: u32| {
        if !alloca_set.contains(&vid) {
            if let Some(&dense) = id_to_dense.get(&vid) {
                if !kill.contains(dense) {
                    gen.insert(dense);
                }
            }
        }
    };

    for_each_operand_in_instruction(inst, |op| {
        if let Operand::Value(v) = op {
            add_use(v.0);
        }
    });

    for_each_value_use_in_instruction(inst, |v| {
        add_use(v.0);
    });
}

/// Collect gen set for a terminator (dense bitset version).
fn collect_terminator_gen_dense(
    term: &Terminator,
    alloca_set: &FxHashSet<u32>,
    id_to_dense: &FxHashMap<u32, usize>,
    kill: &BitSet,
    gen: &mut BitSet,
) {
    for_each_operand_in_terminator(term, |op| {
        if let Operand::Value(v) = op {
            if !alloca_set.contains(&v.0) {
                if let Some(&dense) = id_to_dense.get(&v.0) {
                    if !kill.contains(dense) {
                        gen.insert(dense);
                    }
                }
            }
        }
    });
}

/// Get successor block IDs from a terminator.
fn terminator_targets(term: &Terminator) -> Vec<u32> {
    match term {
        Terminator::Branch(target) => vec![target.0],
        Terminator::CondBranch { true_label, false_label, .. } => {
            vec![true_label.0, false_label.0]
        }
        Terminator::IndirectBranch { possible_targets, .. } => {
            possible_targets.iter().map(|t| t.0).collect()
        }
        Terminator::Switch { cases, default, .. } => {
            let mut targets = vec![default.0];
            for (_, label) in cases {
                targets.push(label.0);
            }
            targets
        }
        _ => vec![],
    }
}

/// Return true if the instruction is a call to setjmp, _setjmp, sigsetjmp, or __sigsetjmp.
/// These functions "return twice": once normally (returning 0) and again when longjmp is called.
/// Values live at the call point must have their intervals extended to prevent stack slot reuse.
fn is_returns_twice_call(inst: &Instruction) -> bool {
    if let Instruction::Call { func, .. } = inst {
        matches!(&**func, "setjmp" | "_setjmp" | "sigsetjmp" | "__sigsetjmp")
    } else {
        false
    }
}

/// Iterate over all Operand references in an instruction.
/// This is the single canonical source of truth for instruction operand traversal.
/// All code that needs to enumerate operands (liveness, use-counting, GEP fold
/// verification) should call this rather than hand-rolling its own match.
pub(super) fn for_each_operand_in_instruction(inst: &Instruction, mut f: impl FnMut(&Operand)) {
    match inst {
        Instruction::Alloca { .. } => {}
        Instruction::DynAlloca { size, .. } => f(size),
        Instruction::Store { val, .. } => f(val),
        Instruction::Load { .. } => {}
        Instruction::BinOp { lhs, rhs, .. } => { f(lhs); f(rhs); }
        Instruction::UnaryOp { src, .. } => f(src),
        Instruction::Cmp { lhs, rhs, .. } => { f(lhs); f(rhs); }
        Instruction::Call { info, .. } => { for a in &info.args { f(a); } }
        Instruction::CallIndirect { func_ptr, info } => { f(func_ptr); for a in &info.args { f(a); } }
        Instruction::GetElementPtr { offset, .. } => f(offset),
        Instruction::Cast { src, .. } => f(src),
        Instruction::Copy { src, .. } => f(src),
        Instruction::GlobalAddr { .. } => {}
        Instruction::Memcpy { .. } => {}
        Instruction::VaArg { .. } => {}
        Instruction::VaStart { .. } => {}
        Instruction::VaEnd { .. } => {}
        Instruction::VaCopy { .. } => {}
        Instruction::VaArgStruct { .. } => {}
        Instruction::AtomicRmw { ptr, val, .. } => { f(ptr); f(val); }
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => { f(ptr); f(expected); f(desired); }
        Instruction::AtomicLoad { ptr, .. } => f(ptr),
        Instruction::AtomicStore { ptr, val, .. } => { f(ptr); f(val); }
        Instruction::Fence { .. } => {}
        Instruction::Phi { incoming, .. } => { for (op, _) in incoming { f(op); } }
        Instruction::LabelAddr { .. } => {}
        Instruction::GetReturnF64Second { .. } => {}
        Instruction::SetReturnF64Second { src } => f(src),
        Instruction::GetReturnF32Second { .. } => {}
        Instruction::SetReturnF32Second { src } => f(src),
        Instruction::GetReturnF128Second { .. } => {},
        Instruction::SetReturnF128Second { src } => f(src),
        Instruction::InlineAsm { inputs, .. } => {
            for (_, op, _) in inputs { f(op); }
        }
        Instruction::Intrinsic { args, .. } => { for a in args { f(a); } }
        Instruction::Select { cond, true_val, false_val, .. } => { f(cond); f(true_val); f(false_val); }
        Instruction::StackSave { .. } => {}
        Instruction::StackRestore { .. } => {}
        Instruction::ParamRef { .. } => {}
    }
}

/// Iterate over Value references (non-Operand) used in an instruction.
/// These are pointer/base values used directly (not wrapped in Operand),
/// e.g., the `ptr` in Store/Load, `base` in GEP, `dest`/`src` in Memcpy.
/// Canonical traversal — shared by liveness, use-counting, and GEP fold analysis.
pub(super) fn for_each_value_use_in_instruction(inst: &Instruction, mut f: impl FnMut(&Value)) {
    match inst {
        Instruction::Store { ptr, .. } => f(ptr),
        Instruction::Load { ptr, .. } => f(ptr),
        Instruction::GetElementPtr { base, .. } => f(base),
        Instruction::Memcpy { dest, src, .. } => { f(dest); f(src); }
        Instruction::VaArg { va_list_ptr, .. } => f(va_list_ptr),
        Instruction::VaStart { va_list_ptr } => f(va_list_ptr),
        Instruction::VaEnd { va_list_ptr } => f(va_list_ptr),
        Instruction::VaCopy { dest_ptr, src_ptr } => { f(dest_ptr); f(src_ptr); }
        Instruction::VaArgStruct { dest_ptr, va_list_ptr, .. } => { f(dest_ptr); f(va_list_ptr); }
        Instruction::InlineAsm { outputs, .. } => {
            for (_, v, _) in outputs { f(v); }
        }
        Instruction::Intrinsic { dest_ptr: Some(dp), .. } => {
            f(dp);
        }
        Instruction::StackRestore { ptr } => f(ptr),
        _ => {}
    }
}

/// Iterate over all Operand references in a terminator.
/// Canonical traversal — shared by liveness, use-counting, and GEP fold analysis.
pub(super) fn for_each_operand_in_terminator(term: &Terminator, mut f: impl FnMut(&Operand)) {
    match term {
        Terminator::Return(Some(op)) => f(op),
        Terminator::CondBranch { cond, .. } => f(cond),
        Terminator::IndirectBranch { target, .. } => f(target),
        Terminator::Switch { val, .. } => f(val),
        _ => {}
    }
}

/// Compute the loop nesting depth for each block in the CFG.
///
/// Uses DFS-based back-edge detection: an edge src -> dst where dst is an
/// ancestor in the DFS tree is a back edge defining a natural loop. For each
/// back edge (src -> header), all blocks on any path from header to src form
/// the loop body. The depth of a block is the number of loop bodies it belongs to.
///
/// This is used by the register allocator to weight uses inside loops more
/// heavily, so that inner-loop temporaries get priority for register allocation.
fn compute_loop_depth(successors: &[Vec<usize>], num_blocks: usize) -> Vec<u32> {
    if num_blocks == 0 {
        return Vec::new();
    }

    let mut depth = vec![0u32; num_blocks];

    // Build predecessor lists from successor lists.
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    for (src, succs) in successors.iter().enumerate() {
        for &dst in succs {
            if dst < num_blocks {
                predecessors[dst].push(src);
            }
        }
    }

    // DFS to classify edges. An edge src -> dst is a back edge if dst is an
    // ancestor of src in the DFS tree (i.e., dst was visited but not finished).
    // State: 0 = unvisited, 1 = in-progress (on stack), 2 = finished.
    let mut state = vec![0u8; num_blocks];
    let mut back_edges: Vec<(usize, usize)> = Vec::new(); // (src, header)

    // Iterative DFS to avoid stack overflow on deeply nested CFGs.
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (block, successor_index)
    state[0] = 1; // Mark entry block as in-progress
    stack.push((0, 0));

    while let Some(&mut (block, ref mut succ_idx)) = stack.last_mut() {
        if *succ_idx < successors[block].len() {
            let next = successors[block][*succ_idx];
            *succ_idx += 1;
            if next < num_blocks {
                match state[next] {
                    0 => {
                        // Unvisited: push to stack
                        state[next] = 1;
                        stack.push((next, 0));
                    }
                    1 => {
                        // Back edge: next is an ancestor (in-progress)
                        back_edges.push((block, next));
                    }
                    _ => {
                        // Cross or forward edge: ignore
                    }
                }
            }
        } else {
            // All successors processed: mark as finished
            state[block] = 2;
            stack.pop();
        }
    }

    // For each back edge (src -> header), find the natural loop body.
    // The loop body consists of all blocks that can reach `src` without going
    // through `header`, plus `header` itself. We compute this by a reverse
    // BFS/DFS from `src` following predecessor edges, stopping at `header`.
    for &(tail, header) in &back_edges {
        // All blocks in the loop body get +1 depth
        depth[header] += 1;
        if tail != header {
            // BFS backwards from tail, stopping at header
            let mut worklist = vec![tail];
            let mut visited = vec![false; num_blocks];
            visited[header] = true; // Don't go past header
            visited[tail] = true;
            depth[tail] += 1;

            while let Some(b) = worklist.pop() {
                for &pred in &predecessors[b] {
                    if pred < num_blocks && !visited[pred] {
                        visited[pred] = true;
                        depth[pred] += 1;
                        worklist.push(pred);
                    }
                }
            }
        }
    }

    depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IrType;
    use crate::ir::reexports::{BasicBlock, BlockId, IrBinOp};

    /// Verify that InlineAsm with register operands is treated as a call point.
    /// This is critical for register allocation: values spanning inline asm with
    /// register constraints must get callee-saved registers, since inline asm may
    /// clobber caller-saved registers (r8-r11 on x86).
    #[test]
    fn test_inline_asm_with_operands_is_call_point() {
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::BinOp {
                    dest: Value(0), op: IrBinOp::Add,
                    lhs: Operand::Const(IrConst::I32(1)),
                    rhs: Operand::Const(IrConst::I32(2)),
                    ty: IrType::I32,
                },
                // Inline asm with an output register constraint
                Instruction::InlineAsm {
                    template: "nop".to_string(),
                    outputs: vec![("=r".to_string(), Value(1), Some("out".to_string()))],
                    inputs: vec![],
                    clobbers: vec![],
                    operand_types: vec![IrType::I32],
                    goto_labels: vec![],
                    input_symbols: vec![],
                    seg_overrides: vec![],
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(0)))),
            source_spans: Vec::new(),
        });
        func.next_value_id = 2;

        let result = compute_live_intervals(&func);
        // The InlineAsm instruction should appear as a call point
        assert!(!result.call_points.is_empty(),
            "InlineAsm with register operands should be a call point");
    }

    /// Verify that empty inline asm barriers (no inputs/outputs) are NOT call points.
    /// Memory barriers like `asm volatile("" ::: "memory")` don't use GP registers
    /// and should not force values into callee-saved registers.
    #[test]
    fn test_empty_inline_asm_barrier_not_call_point() {
        let mut func = IrFunction::new("test".into(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::BinOp {
                    dest: Value(0), op: IrBinOp::Add,
                    lhs: Operand::Const(IrConst::I32(1)),
                    rhs: Operand::Const(IrConst::I32(2)),
                    ty: IrType::I32,
                },
                // Empty inline asm barrier - no outputs or inputs
                Instruction::InlineAsm {
                    template: String::new(),
                    outputs: vec![],
                    inputs: vec![],
                    clobbers: vec!["memory".to_string()],
                    operand_types: vec![],
                    goto_labels: vec![],
                    input_symbols: vec![],
                    seg_overrides: vec![],
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(0)))),
            source_spans: Vec::new(),
        });
        func.next_value_id = 1;

        let result = compute_live_intervals(&func);
        // Call points should only contain the calls, not the empty barrier
        assert!(result.call_points.is_empty(),
            "Empty inline asm barriers should NOT be call points");
    }
}
