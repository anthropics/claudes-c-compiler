//! Sparse Conditional Constant Propagation (SCCP).
//!
//! Implements the Wegman-Zadeck algorithm: propagates constants through the SSA
//! graph across block boundaries via phi nodes, eliminates dead branches, and
//! removes unreachable code. This is strictly more powerful than the existing
//! intra-block constant folding pass because it handles inter-block constant
//! flow through phis and only considers reachable CFG edges.
//!
//! Requires use-chains from UseDefInfo for efficient worklist-driven propagation.

use crate::common::fx_hash::FxHashSet;
use crate::common::types::IrType;
use crate::ir::reexports::{
    BlockId, Instruction, IrBinOp, IrCmpOp, IrConst, IrFunction,
    IrUnaryOp, Operand, Terminator,
};
use crate::passes::use_def::UseDefInfo;
use std::collections::HashMap;

use super::constant_fold;

/// Lattice value for SCCP. Values move monotonically downward:
/// Top → Constant → Bottom.
#[derive(Debug, Clone, Copy)]
enum LatticeVal {
    /// Not yet reached / unknown. Optimistic assumption.
    Top,
    /// Known to be a specific constant.
    Constant(IrConst),
    /// Overdefined: may take multiple values at runtime.
    Bottom,
}

impl LatticeVal {
    /// Lattice meet: Top ∧ x = x, Const(a) ∧ Const(b) = Const(a) if a==b else Bottom,
    /// Bottom ∧ x = Bottom.
    fn meet(self, other: LatticeVal) -> LatticeVal {
        match (self, other) {
            (LatticeVal::Top, x) | (x, LatticeVal::Top) => x,
            (LatticeVal::Bottom, _) | (_, LatticeVal::Bottom) => LatticeVal::Bottom,
            (LatticeVal::Constant(a), LatticeVal::Constant(b)) => {
                if a.to_hash_key() == b.to_hash_key() {
                    LatticeVal::Constant(a)
                } else {
                    LatticeVal::Bottom
                }
            }
        }
    }

    fn is_bottom(self) -> bool {
        matches!(self, LatticeVal::Bottom)
    }

    fn as_const(self) -> Option<IrConst> {
        match self {
            LatticeVal::Constant(c) => Some(c),
            _ => None,
        }
    }
}

/// SCCP algorithm state.
struct SccpState {
    /// Lattice value for each SSA value, indexed by Value.0.
    lattice: Vec<LatticeVal>,
    /// Whether each block has been marked executable.
    block_executable: Vec<bool>,
    /// Set of executable CFG edges (from_block_idx, to_block_idx).
    executable_edges: FxHashSet<(u32, u32)>,
    /// Worklist of block indices to process.
    cfg_worklist: Vec<u32>,
    /// Worklist of value IDs whose lattice changed.
    ssa_worklist: Vec<u32>,
    /// Map from BlockId to block position index.
    label_to_idx: HashMap<BlockId, u32>,
}

/// Entry point: run SCCP on a function with pre-built use-def info.
/// Returns the number of IR changes made.
pub fn run_sccp_with_usedef(func: &mut IrFunction, usedef: &UseDefInfo) -> usize {
    if func.blocks.is_empty() {
        return 0;
    }

    let num_blocks = func.blocks.len();
    let num_values = func.max_value_id() as usize + 1;

    // Build label → index map.
    let mut label_to_idx = HashMap::new();
    for (i, block) in func.blocks.iter().enumerate() {
        label_to_idx.insert(block.label, i as u32);
    }

    let mut state = SccpState {
        lattice: vec![LatticeVal::Top; num_values],
        block_executable: vec![false; num_blocks],
        executable_edges: FxHashSet::default(),
        cfg_worklist: Vec::new(),
        ssa_worklist: Vec::new(),
        label_to_idx,
    };

    // Initialize ParamRef values to Bottom (we don't know parameter values).
    for block in func.blocks.iter() {
        for inst in &block.instructions {
            if let Instruction::ParamRef { dest, .. } = inst {
                let id = dest.0 as usize;
                if id < num_values {
                    state.lattice[id] = LatticeVal::Bottom;
                }
            }
        }
    }

    // Values that are used but have no definition (dangling references from
    // prior passes that removed the defining instruction) must be Bottom.
    // Leaving them as Top would cause SCCP to treat their users as unreachable.
    for i in 0..num_values {
        if usedef.use_count[i] > 0 && usedef.def_loc[i].is_none() {
            state.lattice[i] = LatticeVal::Bottom;
        }
    }

    // Seed: entry block is executable.
    state.cfg_worklist.push(0);

    // Main loop: process both worklists until empty.
    while !state.cfg_worklist.is_empty() || !state.ssa_worklist.is_empty() {
        // Process CFG worklist.
        while let Some(block_idx) = state.cfg_worklist.pop() {
            let bi = block_idx as usize;
            if bi >= num_blocks {
                continue;
            }

            if !state.block_executable[bi] {
                state.block_executable[bi] = true;
                // First time visiting: evaluate all instructions.
                visit_block(func, block_idx, usedef, &mut state);
            } else {
                // Already visited: only need to re-evaluate phis (new edge arrived).
                visit_phis(func, block_idx, usedef, &mut state);
            }
        }

        // Process SSA worklist.
        while let Some(value_id) = state.ssa_worklist.pop() {
            // Re-evaluate all users of this value in executable blocks.
            for &loc in usedef.uses_of(value_id) {
                let bi = loc.block_idx as usize;
                if bi >= num_blocks || !state.block_executable[bi] {
                    continue;
                }
                if loc.is_terminator() {
                    evaluate_terminator(&func.blocks[bi].terminator, loc.block_idx, &mut state);
                } else {
                    let ii = loc.inst_idx as usize;
                    if let Some(inst) = func.blocks[bi].instructions.get(ii) {
                        evaluate_instruction(inst, loc.block_idx, &mut state);
                    }
                }
            }
        }
    }

    // Rewrite phase: apply lattice results to IR.
    rewrite(func, &state)
}

/// Evaluate all instructions and the terminator in a block.
fn visit_block(func: &IrFunction, block_idx: u32, _usedef: &UseDefInfo, state: &mut SccpState) {
    let bi = block_idx as usize;
    let block = &func.blocks[bi];

    for inst in &block.instructions {
        evaluate_instruction(inst, block_idx, state);
    }

    evaluate_terminator(&block.terminator, block_idx, state);
}

/// Re-evaluate only phi nodes in a block (called when a new edge becomes executable).
fn visit_phis(func: &IrFunction, block_idx: u32, _usedef: &UseDefInfo, state: &mut SccpState) {
    let bi = block_idx as usize;
    let block = &func.blocks[bi];

    for inst in &block.instructions {
        if matches!(inst, Instruction::Phi { .. }) {
            evaluate_instruction(inst, block_idx, state);
        }
    }

    // Also re-evaluate the terminator since it may depend on phi results.
    evaluate_terminator(&block.terminator, block_idx, state);
}

/// Resolve an operand to its lattice value.
#[inline]
fn resolve_lattice(op: &Operand, state: &SccpState) -> LatticeVal {
    match op {
        Operand::Const(c) => LatticeVal::Constant(*c),
        Operand::Value(v) => {
            let id = v.0 as usize;
            if id < state.lattice.len() {
                state.lattice[id]
            } else {
                LatticeVal::Bottom
            }
        }
    }
}

/// Update a value's lattice (monotone: only moves downward). If changed,
/// enqueue value on SSA worklist.
#[inline]
fn update_lattice(value_id: u32, new_val: LatticeVal, state: &mut SccpState) {
    let id = value_id as usize;
    if id >= state.lattice.len() {
        return;
    }
    let old = state.lattice[id];
    let merged = old.meet(new_val);

    // Check if the lattice actually changed (moved downward).
    let changed = match (old, merged) {
        (LatticeVal::Top, LatticeVal::Top) => false,
        (LatticeVal::Bottom, _) => false,
        (LatticeVal::Top, _) => true,
        (LatticeVal::Constant(a), LatticeVal::Constant(b)) => {
            a.to_hash_key() != b.to_hash_key()
        }
        (LatticeVal::Constant(_), LatticeVal::Bottom) => true,
        // Lattice is monotone (only moves downward), so Constant → Top can't happen
        // after meet. Include for exhaustiveness.
        (LatticeVal::Constant(_), LatticeVal::Top) => false,
    };

    if changed {
        state.lattice[id] = merged;
        state.ssa_worklist.push(value_id);
    }
}

/// Evaluate a single instruction and update its destination's lattice value.
/// `block_idx` is the index of the block containing this instruction.
fn evaluate_instruction(inst: &Instruction, block_idx: u32, state: &mut SccpState) {
    match inst {
        Instruction::Phi { dest, incoming, ty: _ } => {
            // Meet of incoming values from executable edges only.
            let dest_id = dest.0;
            let mut result = LatticeVal::Top;
            for (op, from_label) in incoming {
                if let Some(&from_idx) = state.label_to_idx.get(from_label) {
                    // Only consider edges that are executable.
                    if state.executable_edges.contains(&(from_idx, block_idx)) {
                        // Skip self-references in phis.
                        if let Operand::Value(v) = op {
                            if v.0 == dest.0 {
                                continue;
                            }
                        }
                        result = result.meet(resolve_lattice(op, state));
                    }
                } else {
                    // Phi references a BlockId not present in the function (stale
                    // entry from a block removed by earlier passes). Be conservative.
                    result = LatticeVal::Bottom;
                }
                if result.is_bottom() {
                    break; // Can't get worse than Bottom.
                }
            }
            update_lattice(dest_id, result, state);
        }

        Instruction::Copy { dest, src } => {
            update_lattice(dest.0, resolve_lattice(src, state), state);
        }

        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            let lv = resolve_lattice(lhs, state);
            let rv = resolve_lattice(rhs, state);
            let result = eval_binop(*op, lv, rv, *ty);
            update_lattice(dest.0, result, state);
        }

        Instruction::UnaryOp { dest, op, src, ty } => {
            // IsConstant: if we can resolve src to a constant, it's Constant(1), else Bottom.
            if *op == IrUnaryOp::IsConstant {
                let sv = resolve_lattice(src, state);
                let result = match sv {
                    LatticeVal::Top => LatticeVal::Top,
                    LatticeVal::Constant(_) => LatticeVal::Constant(IrConst::I32(1)),
                    LatticeVal::Bottom => LatticeVal::Constant(IrConst::I32(0)),
                };
                update_lattice(dest.0, result, state);
                return;
            }
            let sv = resolve_lattice(src, state);
            let result = eval_unaryop(*op, sv, *ty);
            update_lattice(dest.0, result, state);
        }

        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            let lv = resolve_lattice(lhs, state);
            let rv = resolve_lattice(rhs, state);
            let result = eval_cmp(*op, lv, rv, *ty);
            update_lattice(dest.0, result, state);
        }

        Instruction::Cast { dest, src, from_ty, to_ty } => {
            let sv = resolve_lattice(src, state);
            let result = eval_cast(sv, *from_ty, *to_ty);
            update_lattice(dest.0, result, state);
        }

        Instruction::Select { dest, cond, true_val, false_val, .. } => {
            let cv = resolve_lattice(cond, state);
            match cv {
                LatticeVal::Top => {
                    update_lattice(dest.0, LatticeVal::Top, state);
                }
                LatticeVal::Constant(c) => {
                    let taken = if const_is_nonzero(&c) {
                        resolve_lattice(true_val, state)
                    } else {
                        resolve_lattice(false_val, state)
                    };
                    update_lattice(dest.0, taken, state);
                }
                LatticeVal::Bottom => {
                    // Both arms contribute.
                    let tv = resolve_lattice(true_val, state);
                    let fv = resolve_lattice(false_val, state);
                    update_lattice(dest.0, tv.meet(fv), state);
                }
            }
        }

        // Conservative: these always produce Bottom.
        Instruction::Load { dest, .. }
        | Instruction::Alloca { dest, .. }
        | Instruction::DynAlloca { dest, .. }
        | Instruction::GlobalAddr { dest, .. }
        | Instruction::GetElementPtr { dest, .. }
        | Instruction::AtomicRmw { dest, .. }
        | Instruction::AtomicCmpxchg { dest, .. }
        | Instruction::AtomicLoad { dest, .. }
        | Instruction::VaArg { dest, .. }
        | Instruction::LabelAddr { dest, .. }
        | Instruction::GetReturnF64Second { dest, .. }
        | Instruction::GetReturnF32Second { dest, .. }
        | Instruction::GetReturnF128Second { dest, .. }
        | Instruction::StackSave { dest, .. }
        | Instruction::Memcpy { dest, .. } => {
            update_lattice(dest.0, LatticeVal::Bottom, state);
        }

        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
            if let Some(dest) = info.dest {
                update_lattice(dest.0, LatticeVal::Bottom, state);
            }
        }

        Instruction::Intrinsic { dest: Some(dest), .. } => {
            update_lattice(dest.0, LatticeVal::Bottom, state);
        }

        Instruction::ParamRef { dest, .. } => {
            update_lattice(dest.0, LatticeVal::Bottom, state);
        }

        // Instructions with no destination value — nothing to propagate.
        _ => {}
    }
}

/// Evaluate a terminator and mark CFG edges executable.
fn evaluate_terminator(term: &Terminator, block_idx: u32, state: &mut SccpState) {
    match term {
        Terminator::Branch(target) => {
            if let Some(&to_idx) = state.label_to_idx.get(target) {
                mark_edge_executable(block_idx, to_idx, state);
            }
        }

        Terminator::CondBranch { cond, true_label, false_label } => {
            let cv = resolve_lattice(cond, state);
            let true_idx = state.label_to_idx.get(true_label).copied();
            let false_idx = state.label_to_idx.get(false_label).copied();

            match cv {
                LatticeVal::Top => {
                    // Optimistic: don't mark either edge yet.
                }
                LatticeVal::Constant(c) => {
                    // Only mark the taken edge.
                    if const_is_nonzero(&c) {
                        if let Some(ti) = true_idx {
                            mark_edge_executable(block_idx, ti, state);
                        }
                    } else {
                        if let Some(fi) = false_idx {
                            mark_edge_executable(block_idx, fi, state);
                        }
                    }
                }
                LatticeVal::Bottom => {
                    // Both edges may be taken.
                    if let Some(ti) = true_idx {
                        mark_edge_executable(block_idx, ti, state);
                    }
                    if let Some(fi) = false_idx {
                        mark_edge_executable(block_idx, fi, state);
                    }
                }
            }
        }

        Terminator::Switch { val, cases, default, .. } => {
            let vv = resolve_lattice(val, state);
            match vv {
                LatticeVal::Top => {
                    // Optimistic: don't mark any edge.
                }
                LatticeVal::Constant(c) => {
                    // Mark only the matching case.
                    let target = if let Some(cv) = c.to_i64() {
                        cases.iter()
                            .find(|(case_val, _)| *case_val == cv)
                            .map(|(_, label)| label)
                            .unwrap_or(default)
                    } else {
                        default
                    };
                    if let Some(&to_idx) = state.label_to_idx.get(target) {
                        mark_edge_executable(block_idx, to_idx, state);
                    }
                }
                LatticeVal::Bottom => {
                    // All edges may be taken.
                    for (_, label) in cases {
                        if let Some(&to_idx) = state.label_to_idx.get(label) {
                            mark_edge_executable(block_idx, to_idx, state);
                        }
                    }
                    if let Some(&to_idx) = state.label_to_idx.get(default) {
                        mark_edge_executable(block_idx, to_idx, state);
                    }
                }
            }
        }

        Terminator::IndirectBranch { possible_targets, .. } => {
            // Conservative: all targets may be taken.
            for label in possible_targets {
                if let Some(&to_idx) = state.label_to_idx.get(label) {
                    mark_edge_executable(block_idx, to_idx, state);
                }
            }
        }

        Terminator::Return(_) | Terminator::Unreachable => {
            // No successor edges.
        }
    }
}

/// Mark a CFG edge executable. If the target block hasn't been visited yet,
/// add it to the CFG worklist. If it has been visited, re-evaluate its phis
/// (a new incoming edge may change phi lattice values).
fn mark_edge_executable(from: u32, to: u32, state: &mut SccpState) {
    if !state.executable_edges.insert((from, to)) {
        return; // Edge already marked.
    }

    // Always add to worklist — visit_block will handle first-visit vs re-visit.
    state.cfg_worklist.push(to);
}

/// Check if a constant is nonzero (for branch conditions).
fn const_is_nonzero(c: &IrConst) -> bool {
    match c {
        IrConst::I8(v) => *v != 0,
        IrConst::I16(v) => *v != 0,
        IrConst::I32(v) => *v != 0,
        IrConst::I64(v) => *v != 0,
        IrConst::I128(v) => *v != 0,
        IrConst::F32(v) => *v != 0.0,
        IrConst::F64(v) => *v != 0.0,
        IrConst::Zero => false,
        _ => true, // LongDouble: conservatively nonzero
    }
}

// ── Lattice evaluation helpers ──────────────────────────────────────────

fn eval_binop(op: IrBinOp, lhs: LatticeVal, rhs: LatticeVal, ty: IrType) -> LatticeVal {
    match (lhs, rhs) {
        (LatticeVal::Bottom, _) | (_, LatticeVal::Bottom) => LatticeVal::Bottom,
        (LatticeVal::Top, _) | (_, LatticeVal::Top) => LatticeVal::Top,
        (LatticeVal::Constant(lc), LatticeVal::Constant(rc)) => {
            if ty.is_128bit() {
                if let (Some(l), Some(r)) = (lc.to_i128(), rc.to_i128()) {
                    if let Some(result) = op.eval_i128(l, r) {
                        return LatticeVal::Constant(IrConst::I128(result));
                    }
                }
                return LatticeVal::Bottom;
            }
            if ty.is_float() {
                if let (Some(l), Some(r)) = (const_to_f64(&lc), const_to_f64(&rc)) {
                    if let Some(result) = constant_fold::fold_float_binop(op, l, r) {
                        return LatticeVal::Constant(constant_fold::make_float_const(result, ty));
                    }
                }
                return LatticeVal::Bottom;
            }
            if let (Some(l), Some(r)) = (lc.to_i64(), rc.to_i64()) {
                let lt = ty.truncate_i64(l);
                let rt = ty.truncate_i64(r);
                if let Some(result) = constant_fold::fold_binop(op, lt, rt, ty) {
                    return LatticeVal::Constant(IrConst::from_i64(result, ty));
                }
            }
            LatticeVal::Bottom
        }
    }
}

fn eval_unaryop(op: IrUnaryOp, src: LatticeVal, ty: IrType) -> LatticeVal {
    match src {
        LatticeVal::Top => LatticeVal::Top,
        LatticeVal::Bottom => LatticeVal::Bottom,
        LatticeVal::Constant(c) => {
            if ty.is_128bit() {
                if let Some(s) = c.to_i128() {
                    let result = match op {
                        IrUnaryOp::Neg => Some(s.wrapping_neg()),
                        IrUnaryOp::Not => Some(!s),
                        _ => None,
                    };
                    if let Some(r) = result {
                        return LatticeVal::Constant(IrConst::I128(r));
                    }
                }
                return LatticeVal::Bottom;
            }
            if ty.is_float() {
                if let Some(s) = const_to_f64(&c) {
                    if op == IrUnaryOp::Neg {
                        return LatticeVal::Constant(constant_fold::make_float_const(-s, ty));
                    }
                }
                return LatticeVal::Bottom;
            }
            if let Some(s) = c.to_i64() {
                if let Some(result) = constant_fold::fold_unaryop(op, s, ty) {
                    return LatticeVal::Constant(IrConst::from_i64(result, ty));
                }
            }
            LatticeVal::Bottom
        }
    }
}

fn eval_cmp(op: IrCmpOp, lhs: LatticeVal, rhs: LatticeVal, ty: IrType) -> LatticeVal {
    match (lhs, rhs) {
        (LatticeVal::Bottom, _) | (_, LatticeVal::Bottom) => LatticeVal::Bottom,
        (LatticeVal::Top, _) | (_, LatticeVal::Top) => LatticeVal::Top,
        (LatticeVal::Constant(lc), LatticeVal::Constant(rc)) => {
            if ty.is_128bit() {
                if let (Some(l), Some(r)) = (lc.to_i128(), rc.to_i128()) {
                    let result = op.eval_i128(l, r);
                    return LatticeVal::Constant(IrConst::I32(result as i32));
                }
                return LatticeVal::Bottom;
            }
            if ty.is_float() {
                if let (Some(l), Some(r)) = (const_to_f64(&lc), const_to_f64(&rc)) {
                    let result = op.eval_f64(l, r);
                    return LatticeVal::Constant(IrConst::I32(result as i32));
                }
                return LatticeVal::Bottom;
            }
            if let (Some(l), Some(r)) = (lc.to_i64(), rc.to_i64()) {
                let result = op.eval_i64(ty.truncate_i64(l), ty.truncate_i64(r));
                return LatticeVal::Constant(IrConst::I32(result as i32));
            }
            LatticeVal::Bottom
        }
    }
}

fn eval_cast(src: LatticeVal, from_ty: IrType, to_ty: IrType) -> LatticeVal {
    match src {
        LatticeVal::Top => LatticeVal::Top,
        LatticeVal::Bottom => LatticeVal::Bottom,
        LatticeVal::Constant(c) => {
            // 128-bit casts
            if from_ty.is_128bit() || to_ty.is_128bit() {
                if let Some(result) = constant_fold::fold_cast_i128(&c, from_ty, to_ty) {
                    return LatticeVal::Constant(result);
                }
                return LatticeVal::Bottom;
            }
            // Float casts — too complex, conservative.
            if from_ty.is_float() || to_ty.is_float() {
                return LatticeVal::Bottom;
            }
            // Integer cast
            if let Some(val) = c.to_i64() {
                let result = constant_fold::fold_cast(val, from_ty, to_ty);
                return LatticeVal::Constant(IrConst::from_i64(result, to_ty));
            }
            LatticeVal::Bottom
        }
    }
}

fn const_to_f64(c: &IrConst) -> Option<f64> {
    match c {
        IrConst::F32(v) => Some(*v as f64),
        IrConst::F64(v) => Some(*v),
        IrConst::LongDouble(v, _) => Some(*v),
        _ => None,
    }
}

// ── Rewrite phase ───────────────────────────────────────────────────────

/// Apply SCCP results: replace operands with constants, fold branches,
/// mark unreachable blocks. Returns the number of changes.
fn rewrite(func: &mut IrFunction, state: &SccpState) -> usize {
    let mut changes = 0;

    for (bi, block) in func.blocks.iter_mut().enumerate() {
        if !state.block_executable[bi] {
            // Non-executable block: don't rewrite here. cfg_simplify will
            // clean up unreachable blocks after SCCP folds the branches
            // that lead to them.
            continue;
        }

        // Rewrite instruction operands.
        for inst in &mut block.instructions {
            changes += rewrite_instruction_operands(inst, state);
        }

        // Rewrite terminator.
        changes += rewrite_terminator(&mut block.terminator, state);
    }

    changes
}

/// Rewrite operands in an instruction: replace Value operands with Const where
/// the lattice says they're constant. Returns number of operands changed.
fn rewrite_instruction_operands(inst: &mut Instruction, state: &SccpState) -> usize {
    let mut changes = 0;

    match inst {
        Instruction::BinOp { lhs, rhs, .. } => {
            changes += rewrite_operand(lhs, state);
            changes += rewrite_operand(rhs, state);
        }
        Instruction::UnaryOp { src, .. } => {
            changes += rewrite_operand(src, state);
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            changes += rewrite_operand(lhs, state);
            changes += rewrite_operand(rhs, state);
        }
        Instruction::Cast { src, .. } => {
            changes += rewrite_operand(src, state);
        }
        Instruction::Copy { src, .. } => {
            changes += rewrite_operand(src, state);
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            changes += rewrite_operand(cond, state);
            changes += rewrite_operand(true_val, state);
            changes += rewrite_operand(false_val, state);
        }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming.iter_mut() {
                changes += rewrite_operand(op, state);
            }
        }
        // Don't rewrite other instructions (loads, stores, calls, etc.)
        _ => {}
    }

    changes
}

/// Try to replace a Value operand with its constant lattice value.
/// Returns 1 if replaced, 0 otherwise.
#[inline]
fn rewrite_operand(op: &mut Operand, state: &SccpState) -> usize {
    if let Operand::Value(v) = op {
        let id = v.0 as usize;
        if id < state.lattice.len() {
            if let LatticeVal::Constant(c) = state.lattice[id] {
                *op = Operand::Const(c);
                return 1;
            }
        }
    }
    0
}

/// Rewrite a terminator based on lattice values. Fold CondBranch with known
/// condition to unconditional Branch; fold Switch with known value.
fn rewrite_terminator(term: &mut Terminator, state: &SccpState) -> usize {
    match term {
        Terminator::CondBranch { cond, true_label, false_label } => {
            if let Operand::Value(v) = cond {
                let id = v.0 as usize;
                if id < state.lattice.len() {
                    if let LatticeVal::Constant(c) = state.lattice[id] {
                        let target = if const_is_nonzero(&c) {
                            *true_label
                        } else {
                            *false_label
                        };
                        *term = Terminator::Branch(target);
                        return 1;
                    }
                }
            }
            // Also try if cond is already a constant operand.
            if let Operand::Const(c) = cond {
                let target = if const_is_nonzero(c) {
                    *true_label
                } else {
                    *false_label
                };
                *term = Terminator::Branch(target);
                return 1;
            }
            0
        }

        Terminator::Switch { val, cases, default, .. } => {
            let cv = match val {
                Operand::Value(v) => {
                    let id = v.0 as usize;
                    if id < state.lattice.len() {
                        state.lattice[id].as_const()
                    } else {
                        None
                    }
                }
                Operand::Const(c) => Some(*c),
            };
            if let Some(c) = cv {
                if let Some(cv) = c.to_i64() {
                    let target = cases.iter()
                        .find(|(case_val, _)| *case_val == cv)
                        .map(|(_, label)| *label)
                        .unwrap_or(*default);
                    *term = Terminator::Branch(target);
                    return 1;
                }
            }
            0
        }

        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IrType;
    use crate::ir::reexports::*;
    use crate::passes::use_def::UseDefInfo;

    fn make_func(blocks: Vec<BasicBlock>) -> IrFunction {
        let mut f = IrFunction::new("test".into(), IrType::Void, vec![], false);
        f.blocks = blocks;
        let mut max = 0u32;
        for b in &f.blocks {
            for inst in &b.instructions {
                if let Some(v) = inst.dest() {
                    if v.0 > max { max = v.0; }
                }
            }
        }
        f.next_value_id = max + 1;
        f
    }

    #[test]
    fn test_sccp_constant_propagation() {
        // %0 = Copy 3
        // %1 = BinOp Add %0, %0  → should become 6
        // return %1
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(3)),
                },
                Instruction::BinOp {
                    dest: Value(1),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(0)),
                    rhs: Operand::Value(Value(0)),
                    ty: IrType::I64,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: vec![],
        }];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);
        assert!(n > 0, "SCCP should have made changes");

        // %1's operands should have been rewritten to constants.
        match &func.blocks[0].instructions[1] {
            Instruction::BinOp { lhs, rhs, .. } => {
                assert!(matches!(lhs, Operand::Const(IrConst::I64(3))));
                assert!(matches!(rhs, Operand::Const(IrConst::I64(3))));
            }
            other => panic!("Expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sccp_branch_folding() {
        // Block 0: %0 = Copy 1; condbranch %0, Block1, Block2
        // Block 1: return void  (reachable)
        // Block 2: return void  (unreachable — condition is always true)
        let blocks = vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Copy {
                        dest: Value(0),
                        src: Operand::Const(IrConst::I32(1)),
                    },
                ],
                terminator: Terminator::CondBranch {
                    cond: Operand::Value(Value(0)),
                    true_label: BlockId(1),
                    false_label: BlockId(2),
                },
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![],
                terminator: Terminator::Return(None),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(2),
                instructions: vec![],
                terminator: Terminator::Return(None),
                source_spans: vec![],
            },
        ];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);
        assert!(n > 0);

        // Block 0 terminator should be Branch(BlockId(1))
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(1))));
    }

    #[test]
    fn test_sccp_dead_edge_phi() {
        // Block 0: %0 = Copy 42; branch → Block 2
        // Block 1: (unreachable) branch → Block 2
        // Block 2: %1 = Phi [(Block 0, %0), (Block 1, Const(99))]; return %1
        // Since Block 1 is unreachable, %1 should resolve to 42.
        let blocks = vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Copy {
                        dest: Value(0),
                        src: Operand::Const(IrConst::I64(42)),
                    },
                ],
                terminator: Terminator::Branch(BlockId(2)),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![],
                terminator: Terminator::Branch(BlockId(2)),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(2),
                instructions: vec![
                    Instruction::Phi {
                        dest: Value(1),
                        incoming: vec![
                            (Operand::Value(Value(0)), BlockId(0)),
                            (Operand::Const(IrConst::I64(99)), BlockId(1)),
                        ],
                        ty: IrType::I64,
                    },
                ],
                terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
                source_spans: vec![],
            },
        ];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);
        assert!(n > 0);

        // Block 1 is unreachable (no edge from block 0 reaches it),
        // but we leave cleanup to cfg_simplify.
    }

    #[test]
    fn test_sccp_switch_folding() {
        // Block 0: %0 = Copy 2; Switch %0: case 1 → Block1, case 2 → Block2, default → Block3
        // Only Block 2 should be reachable.
        let blocks = vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Copy {
                        dest: Value(0),
                        src: Operand::Const(IrConst::I32(2)),
                    },
                ],
                terminator: Terminator::Switch {
                    val: Operand::Value(Value(0)),
                    cases: vec![(1, BlockId(1)), (2, BlockId(2))],
                    default: BlockId(3),
                    ty: IrType::I32,
                },
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![],
                terminator: Terminator::Return(None),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(2),
                instructions: vec![],
                terminator: Terminator::Return(None),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(3),
                instructions: vec![],
                terminator: Terminator::Return(None),
                source_spans: vec![],
            },
        ];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);
        assert!(n > 0);

        // Block 0 should have Branch(BlockId(2))
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(2))));
    }

    #[test]
    fn test_sccp_transitive_chain() {
        // %0 = Copy 5
        // %1 = Copy %0
        // %2 = BinOp Add %1, %1  → should fold to 10
        // return %2
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I32(5)),
                },
                Instruction::Copy {
                    dest: Value(1),
                    src: Operand::Value(Value(0)),
                },
                Instruction::BinOp {
                    dest: Value(2),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(1)),
                    rhs: Operand::Value(Value(1)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(2)))),
            source_spans: vec![],
        }];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);
        assert!(n > 0);

        // %2's operands should be rewritten to constants.
        match &func.blocks[0].instructions[2] {
            Instruction::BinOp { lhs, rhs, .. } => {
                assert!(matches!(lhs, Operand::Const(IrConst::I32(5))));
                assert!(matches!(rhs, Operand::Const(IrConst::I32(5))));
            }
            other => panic!("Expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sccp_param_stays_bottom() {
        // %0 = ParamRef(0)
        // %1 = BinOp Add %0, Const(1)
        // return %1
        // ParamRef is Bottom → %1 stays Bottom → no constant rewrite.
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::ParamRef {
                    dest: Value(0),
                    param_idx: 0,
                    ty: IrType::I32,
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
            source_spans: vec![],
        }];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);
        // No constant propagation should happen.
        assert_eq!(n, 0, "ParamRef should prevent constant propagation");

        // %1's lhs should still be Value(0).
        match &func.blocks[0].instructions[1] {
            Instruction::BinOp { lhs, .. } => {
                assert!(matches!(lhs, Operand::Value(Value(0))));
            }
            other => panic!("Expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sccp_loop_phi_bottom() {
        // Block 0: %0 = Copy 1; branch → Block 1
        // Block 1: %1 = Phi [(Block 0, %0), (Block 1, %2)]
        //          %2 = BinOp Add %1, Const(1)
        //          branch → Block 1
        // The phi merges a constant (1) with a loop-carried value (%2).
        // %1 should be Bottom (multiple possible values).
        let blocks = vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Copy {
                        dest: Value(0),
                        src: Operand::Const(IrConst::I32(1)),
                    },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![
                    Instruction::Phi {
                        dest: Value(1),
                        incoming: vec![
                            (Operand::Value(Value(0)), BlockId(0)),
                            (Operand::Value(Value(2)), BlockId(1)),
                        ],
                        ty: IrType::I32,
                    },
                    Instruction::BinOp {
                        dest: Value(2),
                        op: IrBinOp::Add,
                        lhs: Operand::Value(Value(1)),
                        rhs: Operand::Const(IrConst::I32(1)),
                        ty: IrType::I32,
                    },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: vec![],
            },
        ];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);

        // %1 should be Bottom (constant + non-constant merge).
        // The BinOp operands should NOT be rewritten to constants.
        match &func.blocks[1].instructions[1] {
            Instruction::BinOp { lhs, .. } => {
                assert!(matches!(lhs, Operand::Value(Value(1))),
                    "Loop phi should be Bottom, operand should remain Value");
            }
            other => panic!("Expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sccp_select_const_cond() {
        // %0 = Copy 1 (true)
        // %1 = Select %0, Const(42), Const(99)  → should resolve to 42
        // return %1
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I32(1)),
                },
                Instruction::Select {
                    dest: Value(1),
                    cond: Operand::Value(Value(0)),
                    true_val: Operand::Const(IrConst::I32(42)),
                    false_val: Operand::Const(IrConst::I32(99)),
                    ty: IrType::I32,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: vec![],
        }];

        let mut func = make_func(blocks);
        let usedef = UseDefInfo::build(&func);
        let n = run_sccp_with_usedef(&mut func, &usedef);
        assert!(n > 0);

        // The Select's condition should have been rewritten to a constant.
        match &func.blocks[0].instructions[1] {
            Instruction::Select { cond, .. } => {
                assert!(matches!(cond, Operand::Const(IrConst::I32(1))));
            }
            other => panic!("Expected Select, got {:?}", other),
        }
    }
}
