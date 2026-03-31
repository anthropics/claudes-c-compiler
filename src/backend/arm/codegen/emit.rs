use crate::delegate_to_impl;
use crate::ir::reexports::{
    AtomicOrdering,
    AtomicRmwOp,
    BlockId,
    Instruction,
    IntrinsicOp,
    IrBinOp,
    IrCmpOp,
    IrConst,
    IrFunction,
    Operand,
    Value,
};
use crate::common::types::IrType;
use crate::common::fx_hash::FxHashMap;
use crate::backend::common::PtrDirective;
use crate::backend::state::{CodegenState, StackSlot};
use crate::backend::traits::ArchCodegen;
use crate::backend::generation::find_param_alloca;
use crate::backend::call_abi::{CallAbiConfig, CallArgClass};
use crate::backend::call_abi::ParamClass;
use crate::backend::inline_asm::emit_inline_asm_common;
use crate::backend::regalloc::PhysReg;
use super::asm_emitter::ARM_GP_SCRATCH;

/// Callee-saved registers available for register allocation: x20-x28.
/// x19 is reserved (some ABIs use it), x29=fp, x30=lr.
pub(super) const ARM_CALLEE_SAVED: [PhysReg; 9] = [
    PhysReg(20), PhysReg(21), PhysReg(22), PhysReg(23), PhysReg(24),
    PhysReg(25), PhysReg(26), PhysReg(27), PhysReg(28),
];

/// Caller-saved registers available for register allocation: x13, x14.
///
/// These are a subset of the AAPCS64 "corruptible" registers (x9-x15).
/// We exclude x9 (primary address register), x10 (memcpy source, secondary
/// scratch), x11 (memcpy loop counter), x12 (memcpy byte transfer), x15
/// (F128 large-offset scratch), and x16/x17/x18 (IP0/IP1/platform-reserved).
///
/// x13 and x14 have NO hardcoded scratch uses in the codegen. They only
/// appear in ARM_TMP_REGS (call argument staging) and ARM_GP_SCRATCH
/// (inline assembly scratch pool). Since caller-saved allocation only assigns
/// values whose live ranges do NOT span any call, the call staging use is safe.
/// Functions with inline assembly have the caller-saved pool disabled entirely.
pub(super) const ARM_CALLER_SAVED: [PhysReg; 2] = [
    PhysReg(13), PhysReg(14),
];

pub(super) fn callee_saved_name(reg: PhysReg) -> &'static str {
    match reg.0 {
        // Caller-saved registers
        13 => "x13", 14 => "x14",
        // Callee-saved registers
        19 => "x19", 20 => "x20", 21 => "x21", 22 => "x22", 23 => "x23", 24 => "x24",
        25 => "x25", 26 => "x26", 27 => "x27", 28 => "x28",
        _ => unreachable!("invalid ARM register index"),
    }
}

pub(super) fn callee_saved_name_32(reg: PhysReg) -> &'static str {
    match reg.0 {
        // Caller-saved registers
        13 => "w13", 14 => "w14",
        // Callee-saved registers
        19 => "w19", 20 => "w20", 21 => "w21", 22 => "w22", 23 => "w23", 24 => "w24",
        25 => "w25", 26 => "w26", 27 => "w27", 28 => "w28",
        _ => unreachable!("invalid ARM register index"),
    }
}

/// Check if a register name is an AArch64 floating-point/SIMD register
/// (s0-s31, d0-d31, v0-v31, q0-q31).
/// This avoids false positives for "sp" (stack pointer) which starts with 's'.
pub(super) fn is_arm_fp_reg(reg: &str) -> bool {
    if let Some(suffix) = reg.strip_prefix('d')
        .or_else(|| reg.strip_prefix('s'))
        .or_else(|| reg.strip_prefix('v'))
        .or_else(|| reg.strip_prefix('q'))
    {
        !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit())
    } else {
        false
    }
}

/// Map IrBinOp to AArch64 mnemonic for simple ALU ops.
pub(super) fn arm_alu_mnemonic(op: IrBinOp) -> &'static str {
    match op {
        IrBinOp::Add => "add",
        IrBinOp::Sub => "sub",
        IrBinOp::And => "and",
        IrBinOp::Or => "orr",
        IrBinOp::Xor => "eor",
        IrBinOp::Mul => "mul",
        _ => unreachable!("unsupported ALU op for arm_alu_mnemonic: {:?}", op),
    }
}

/// Map an IrCmpOp to its AArch64 integer condition code suffix.
pub(super) fn arm_int_cond_code(op: IrCmpOp) -> &'static str {
    match op {
        IrCmpOp::Eq => "eq",
        IrCmpOp::Ne => "ne",
        IrCmpOp::Slt => "lt",
        IrCmpOp::Sle => "le",
        IrCmpOp::Sgt => "gt",
        IrCmpOp::Sge => "ge",
        IrCmpOp::Ult => "lo",
        IrCmpOp::Ule => "ls",
        IrCmpOp::Ugt => "hi",
        IrCmpOp::Uge => "hs",
    }
}

/// Return the inverted AArch64 condition code suffix.
pub(super) fn arm_invert_cond_code(cc: &str) -> &'static str {
    match cc {
        "eq" => "ne",
        "ne" => "eq",
        "lt" => "ge",
        "ge" => "lt",
        "gt" => "le",
        "le" => "gt",
        "lo" => "hs",
        "hs" => "lo",
        "hi" => "ls",
        "ls" => "hi",
        "mi" => "pl",
        "pl" => "mi",
        "vs" => "vc",
        "vc" => "vs",
        _ => unreachable!("unknown ARM condition code: {}", cc),
    }
}

/// AArch64 code generator. Implements the ArchCodegen trait for the shared framework.
/// Uses AAPCS64 calling convention with stack-based allocation.
pub struct ArmCodegen {
    pub(crate) state: CodegenState,
    /// Frame size for the current function (needed for epilogue in terminators).
    pub(super) current_frame_size: i64,
    pub(super) current_return_type: IrType,
    /// For variadic functions: offset from SP where the GP register save area starts (x0-x7).
    pub(super) va_gp_save_offset: i64,
    /// For variadic functions: offset from SP where the FP register save area starts (q0-q7).
    pub(super) va_fp_save_offset: i64,
    /// Number of named (non-variadic) GP params for current variadic function.
    pub(super) va_named_gp_count: usize,
    /// Number of named (non-variadic) FP params for current variadic function.
    pub(super) va_named_fp_count: usize,
    /// Total bytes of named (non-variadic) params passed on the stack.
    /// This includes all stack-passed scalars, F128, I128, and structs with alignment.
    pub(super) va_named_stack_bytes: usize,
    /// Scratch register index for inline asm GP register allocation
    pub(super) asm_scratch_idx: usize,
    /// Scratch register index for inline asm FP register allocation
    pub(super) asm_fp_scratch_idx: usize,
    /// Register allocator: value ID -> physical callee-saved register.
    pub(super) reg_assignments: FxHashMap<u32, PhysReg>,
    /// Which callee-saved registers are actually used (for save/restore).
    pub(super) used_callee_saved: Vec<PhysReg>,
    /// SP offset where callee-saved registers are stored.
    pub(super) callee_save_offset: i64,
    /// For large stack frames: reserved for future x19 frame base optimization.
    /// Currently always None (optimization disabled due to correctness issue).
    pub(super) frame_base_offset: Option<i64>,
    /// Whether -mgeneral-regs-only is set. When true, FP/SIMD registers (q0-q7)
    /// must not be used. Variadic prologues skip saving q0-q7 and va_start
    /// sets __vr_offs=0 (no FP register save area available).
    pub(super) general_regs_only: bool,
}

impl ArmCodegen {
    pub fn new() -> Self {
        Self {
            state: CodegenState::new(),
            current_frame_size: 0,
            current_return_type: IrType::I64,
            va_gp_save_offset: 0,
            va_fp_save_offset: 0,
            va_named_gp_count: 0,
            va_named_fp_count: 0,
            va_named_stack_bytes: 0,
            asm_scratch_idx: 0,
            asm_fp_scratch_idx: 0,
            reg_assignments: FxHashMap::default(),
            used_callee_saved: Vec::new(),
            callee_save_offset: 0,
            frame_base_offset: None,
            general_regs_only: false,
        }
    }

    /// Disable jump table emission (-fno-jump-tables).
    pub fn set_no_jump_tables(&mut self, enabled: bool) {
        self.state.no_jump_tables = enabled;
    }

    /// Enable position-independent code generation (-fPIC/-fpie).
    pub fn set_pic(&mut self, pic: bool) {
        self.state.pic_mode = pic;
    }

    /// Set general-regs-only mode (-mgeneral-regs-only).
    /// When true, FP/SIMD registers are not used in variadic prologues.
    pub fn set_general_regs_only(&mut self, enabled: bool) {
        self.general_regs_only = enabled;
    }

    /// Apply all relevant options from a `CodegenOptions` struct.
    pub fn apply_options(&mut self, opts: &crate::backend::CodegenOptions) {
        self.set_pic(opts.pic);
        self.set_no_jump_tables(opts.no_jump_tables);
        self.set_general_regs_only(opts.general_regs_only);
        self.state.emit_cfi = opts.emit_cfi;
    }

    /// Get the physical register assigned to an operand (if it's a Value with a register).
    pub(super) fn operand_reg(&self, op: &Operand) -> Option<PhysReg> {
        match op {
            Operand::Value(v) => self.reg_assignments.get(&v.0).copied(),
            _ => None,
        }
    }

    /// Get the physical register assigned to a destination value.
    pub(super) fn dest_reg(&self, dest: &Value) -> Option<PhysReg> {
        self.reg_assignments.get(&dest.0).copied()
    }

    /// Load an operand into a specific callee-saved register.
    pub(super) fn operand_to_callee_reg(&mut self, op: &Operand, reg: PhysReg) {
        let reg_name = callee_saved_name(reg);
        match op {
            Operand::Const(_) => {
                self.operand_to_x0(op);
                self.state.emit_fmt(format_args!("    mov {}, x0", reg_name));
            }
            Operand::Value(v) => {
                if let Some(&src_reg) = self.reg_assignments.get(&v.0) {
                    if src_reg.0 != reg.0 {
                        let src_name = callee_saved_name(src_reg);
                        self.state.emit_fmt(format_args!("    mov {}, {}", reg_name, src_name));
                    }
                } else {
                    self.operand_to_x0(op);
                    self.state.emit_fmt(format_args!("    mov {}, x0", reg_name));
                }
            }
        }
    }

    /// Try to extract an immediate value suitable for ARM imm12 encoding.
    pub(super) fn const_as_imm12(op: &Operand) -> Option<i64> {
        match op {
            Operand::Const(c) => {
                let val = match c {
                    IrConst::I8(v) => *v as i64,
                    IrConst::I16(v) => *v as i64,
                    IrConst::I32(v) => *v as i64,
                    IrConst::I64(v) => *v,
                    IrConst::Zero => 0,
                    _ => return None,
                };
                // ARM add/sub imm12: 0..4095
                if (0..=4095).contains(&val) {
                    Some(val)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// If `op` is a constant that is a power of two, return its log2 (shift amount).
    pub(super) fn const_as_power_of_2(op: &Operand) -> Option<u32> {
        match op {
            Operand::Const(c) => {
                let val: u64 = match c {
                    IrConst::I8(v) => *v as u8 as u64,
                    IrConst::I16(v) => *v as u16 as u64,
                    IrConst::I32(v) => *v as u32 as u64,
                    IrConst::I64(v) => *v as u64,
                    IrConst::Zero => return None,
                    _ => return None,
                };
                if val > 0 && val.is_power_of_two() {
                    Some(val.trailing_zeros())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Pre-scan all inline asm instructions in a function to predict which
    /// callee-saved registers will be needed as scratch registers.
    ///
    /// The inline asm scratch allocator (`assign_scratch_reg`) walks through
    /// `ARM_GP_SCRATCH` = [x9..x15, x19, x20, x21], skipping registers that
    /// appear in the clobber/excluded list. When enough caller-saved scratch regs
    /// (x9-x15) are clobbered, the allocator falls through to callee-saved
    /// registers (x19, x20, x21). These must be saved/restored in the prologue,
    /// but the prologue is emitted before inline asm codegen runs. This function
    /// simulates the allocation to discover the callee-saved registers early.
    pub(super) fn prescan_inline_asm_callee_saved(func: &IrFunction, used_callee_saved: &mut Vec<PhysReg>) {
        for block in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::InlineAsm {
                    outputs, inputs, clobbers, ..
                } = instr {
                    // Build excluded set: clobber registers + specific constraint regs
                    let mut excluded: Vec<String> = Vec::new();
                    for clobber in clobbers {
                        if clobber == "cc" || clobber == "memory" {
                            continue;
                        }
                        excluded.push(clobber.clone());
                        // Also exclude the alternate width alias (wN <-> xN)
                        // and normalize rN (GCC AArch64 alias for xN)
                        if let Some(suffix) = clobber.strip_prefix('w') {
                            if suffix.chars().all(|c| c.is_ascii_digit()) {
                                excluded.push(format!("x{}", suffix));
                            }
                        } else if let Some(suffix) = clobber.strip_prefix('x') {
                            if suffix.chars().all(|c| c.is_ascii_digit()) {
                                excluded.push(format!("w{}", suffix));
                            }
                        } else if let Some(suffix) = clobber.strip_prefix('r') {
                            if suffix.chars().all(|c| c.is_ascii_digit()) {
                                if let Ok(n) = suffix.parse::<u32>() {
                                    if n <= 30 {
                                        excluded.push(format!("x{}", n));
                                        excluded.push(format!("w{}", n));
                                    }
                                }
                            }
                        }
                    }

                    // Count GP scratch registers needed:
                    // 1. GpReg operands (outputs + inputs that are "r" type, not tied, not specific)
                    // 2. Memory operands that need indirection (non-alloca pointers get a scratch reg)
                    let mut gp_scratch_needed = 0usize;

                    for (constraint, _, _) in outputs {
                        let c = constraint.trim_start_matches(['=', '+', '&', '%']);
                        if c.starts_with('{') && c.ends_with('}') {
                            let reg_name = &c[1..c.len()-1];
                            // Normalize rN -> xN (GCC AArch64 alias)
                            let normalized = super::asm_emitter::normalize_aarch64_register(reg_name);
                            excluded.push(normalized);
                        } else if c == "m" || c == "Q" || c.contains('Q') || c.contains('m') {
                            // Memory operands may need a scratch reg for indirection.
                            // Conservatively count each one.
                            gp_scratch_needed += 1;
                        } else if c == "w" {
                            // FP register, doesn't consume GP scratch
                        } else if !c.is_empty() && c.chars().all(|ch| ch.is_ascii_digit()) {
                            // Tied operand, doesn't need its own scratch
                        } else {
                            // GpReg
                            gp_scratch_needed += 1;
                        }
                    }

                    // Count "+" read-write outputs that generate synthetic inputs.
                    // Synthetic inputs from "+r" have constraint "r" and consume a
                    // GP scratch slot in phase 1 (even though the register is later
                    // overwritten by copy_metadata_from). We must count these too.
                    let num_plus = outputs.iter().filter(|(c,_,_)| c.contains('+')).count();
                    {
                        let mut plus_idx = 0;
                        for (constraint, _, _) in outputs.iter() {
                            if constraint.contains('+') {
                                let c = constraint.trim_start_matches(['=', '+', '&', '%']);
                                // Synthetic input inherits constraint with '+' stripped
                                // "+r" → "r" (GpReg, consumes scratch), "+m" → "m" (Memory, skip)
                                if c != "m" && c != "Q" && !c.contains('Q') && !c.contains('m') && c != "w"
                                    && !(c.starts_with('{') && c.ends_with('}'))
                                    && (!c.chars().all(|ch| ch.is_ascii_digit()) || c.is_empty())
                                {
                                    gp_scratch_needed += 1;
                                }
                                plus_idx += 1;
                            }
                        }
                        let _ = plus_idx;
                    }

                    for (i, (constraint, val, _)) in inputs.iter().enumerate() {
                        // Skip synthetic inputs (they're already counted above)
                        if i < num_plus {
                            continue;
                        }
                        let c = constraint.trim_start_matches(['=', '+', '&', '%']);
                        if c.starts_with('{') && c.ends_with('}') {
                            let reg_name = &c[1..c.len()-1];
                            // Normalize rN -> xN (GCC AArch64 alias)
                            let normalized = super::asm_emitter::normalize_aarch64_register(reg_name);
                            excluded.push(normalized);
                        } else if c == "m" || c == "Q" || c.contains('Q') || c.contains('m') {
                            gp_scratch_needed += 1;
                        } else if c == "w" {
                            // FP register
                        } else if !c.is_empty() && c.chars().all(|ch| ch.is_ascii_digit()) {
                            // Tied operand
                        } else {
                            // Check if constant input with immediate-capable constraint
                            // would be promoted to Immediate (no scratch needed)
                            let is_const = matches!(val, Operand::Const(_));
                            let has_imm_alt = c.contains('i') || c.contains('I') || c.contains('n');
                            if is_const && has_imm_alt {
                                // Would be promoted to Immediate, no GP scratch needed
                            } else {
                                gp_scratch_needed += 1;
                            }
                        }
                    }

                    // Simulate walking through ARM_GP_SCRATCH, skipping excluded regs
                    let mut scratch_idx = 0;
                    let mut assigned = 0;
                    while assigned < gp_scratch_needed && scratch_idx < ARM_GP_SCRATCH.len() {
                        let reg = ARM_GP_SCRATCH[scratch_idx];
                        scratch_idx += 1;
                        if excluded.iter().any(|e| e == reg) {
                            continue;
                        }
                        assigned += 1;
                        // Check if this is a callee-saved register
                        if let Some(num_str) = reg.strip_prefix('x') {
                            if let Ok(n) = num_str.parse::<u8>() {
                                if (19..=28).contains(&n) {
                                    let phys = PhysReg(n);
                                    if !used_callee_saved.contains(&phys) {
                                        used_callee_saved.push(phys);
                                    }
                                }
                            }
                        }
                    }

                    // Also handle overflow beyond ARM_GP_SCRATCH (format!("x{}", 9 + idx))
                    while assigned < gp_scratch_needed {
                        let idx = scratch_idx;
                        scratch_idx += 1;
                        let reg_num = 9 + idx;
                        let reg_name = format!("x{}", reg_num);
                        if excluded.iter().any(|e| e == &reg_name) {
                            continue;
                        }
                        assigned += 1;
                        if (19..=28).contains(&reg_num) {
                            let phys = PhysReg(reg_num as u8);
                            if !used_callee_saved.contains(&phys) {
                                used_callee_saved.push(phys);
                            }
                        }
                    }
                }
            }
        }
        // Sort for deterministic prologue/epilogue emission
        used_callee_saved.sort_by_key(|r| r.0);
    }

    /// Restore callee-saved registers before epilogue.
    pub(super) fn emit_restore_callee_saved(&mut self) {
        let used_regs = self.used_callee_saved.clone();
        let base = self.callee_save_offset;
        let n = used_regs.len();
        let mut i = 0;
        while i + 1 < n {
            let r1 = callee_saved_name(used_regs[i]);
            let r2 = callee_saved_name(used_regs[i + 1]);
            let offset = base + (i as i64) * 8;
            self.emit_ldp_from_sp(r1, r2, offset);
            i += 2;
        }
        if i < n {
            let r = callee_saved_name(used_regs[i]);
            let offset = base + (i as i64) * 8;
            self.emit_load_from_sp(r, offset, "ldr");
        }
    }

    /// Check if an IrConst is a small unsigned immediate that fits in AArch64
    /// `cmp Xn, #imm12` instruction (0..=4095).
    fn const_as_cmp_imm12(c: &IrConst) -> Option<u64> {
        let v = match c {
            IrConst::I8(v) => *v as i64,
            IrConst::I16(v) => *v as i64,
            IrConst::I32(v) => *v as i64,
            IrConst::I64(v) => *v,
            IrConst::Zero => 0,
            _ => return None,
        };
        // AArch64 cmp (alias of subs) accepts unsigned 12-bit immediate (0..4095),
        // optionally shifted left by 12. We only use the unshifted form.
        if (0..=4095).contains(&v) {
            Some(v as u64)
        } else {
            None
        }
    }

    /// Check if an IrConst is a small negative value that can use `cmn Xn, #imm12`
    /// (i.e., the negated value fits in 0..=4095).
    fn const_as_cmn_imm12(c: &IrConst) -> Option<u64> {
        let v = match c {
            IrConst::I8(v) => *v as i64,
            IrConst::I16(v) => *v as i64,
            IrConst::I32(v) => *v as i64,
            IrConst::I64(v) => *v,
            _ => return None,
        };
        let neg = v.checked_neg()?;
        if (1..=4095).contains(&neg) {
            Some(neg as u64)
        } else {
            None
        }
    }

    /// Get the register name for a Value if it has a register assignment.
    /// Returns (64-bit name, 32-bit name) pair.
    fn value_reg_name(&self, v: &Value) -> Option<(&'static str, &'static str)> {
        self.reg_assignments.get(&v.0).map(|&reg| {
            (callee_saved_name(reg), callee_saved_name_32(reg))
        })
    }

    /// Emit the integer comparison preamble.
    /// Optimized paths:
    ///   1. reg vs #imm12 → `cmp wN/xN, #imm` (1 instruction)
    ///   2. reg vs #neg_imm12 → `cmn wN/xN, #imm` (1 instruction)
    ///   3. reg vs reg → `cmp wN/xN, wM/xM` (1 instruction)
    ///   4. fallback → load lhs→x1, rhs→x0, `cmp w1/x1, w0/x0`
    ///      Used by both emit_cmp and emit_fused_cmp_branch.
    pub(super) fn emit_int_cmp_insn(&mut self, lhs: &Operand, rhs: &Operand, ty: IrType) {
        let use_32bit = ty == IrType::I32 || ty == IrType::U32
            || ty == IrType::I8 || ty == IrType::U8
            || ty == IrType::I16 || ty == IrType::U16;

        // Try optimized path: lhs in register, rhs is immediate
        if let Operand::Value(lv) = lhs {
            if let Some((lhs_x, lhs_w)) = self.value_reg_name(lv) {
                let lhs_reg = if use_32bit { lhs_w } else { lhs_x };

                // cmp reg, #imm12
                if let Operand::Const(c) = rhs {
                    if let Some(imm) = Self::const_as_cmp_imm12(c) {
                        self.state.emit_fmt(format_args!("    cmp {}, #{}", lhs_reg, imm));
                        return;
                    }
                    // cmn reg, #imm12 (for negative constants)
                    if let Some(imm) = Self::const_as_cmn_imm12(c) {
                        self.state.emit_fmt(format_args!("    cmn {}, #{}", lhs_reg, imm));
                        return;
                    }
                }

                // cmp reg, reg
                if let Operand::Value(rv) = rhs {
                    if let Some((rhs_x, rhs_w)) = self.value_reg_name(rv) {
                        let rhs_reg = if use_32bit { rhs_w } else { rhs_x };
                        self.state.emit_fmt(format_args!("    cmp {}, {}", lhs_reg, rhs_reg));
                        return;
                    }
                }

                // lhs in register, rhs needs loading into x0
                self.operand_to_x0(rhs);
                if use_32bit {
                    self.state.emit_fmt(format_args!("    cmp {}, w0", lhs_reg));
                } else {
                    self.state.emit_fmt(format_args!("    cmp {}, x0", lhs_reg));
                }
                return;
            }
        }

        // Try: lhs needs loading, rhs in register
        if let Operand::Value(rv) = rhs {
            if let Some((rhs_x, rhs_w)) = self.value_reg_name(rv) {
                self.operand_to_x0(lhs);
                let rhs_reg = if use_32bit { rhs_w } else { rhs_x };
                if use_32bit {
                    self.state.emit_fmt(format_args!("    cmp w0, {}", rhs_reg));
                } else {
                    self.state.emit_fmt(format_args!("    cmp x0, {}", rhs_reg));
                }
                return;
            }
        }

        // Try: lhs in x0 (accumulator), rhs is immediate
        if let Operand::Const(c) = rhs {
            if let Some(imm) = Self::const_as_cmp_imm12(c) {
                self.operand_to_x0(lhs);
                if use_32bit {
                    self.state.emit_fmt(format_args!("    cmp w0, #{}", imm));
                } else {
                    self.state.emit_fmt(format_args!("    cmp x0, #{}", imm));
                }
                return;
            }
            if let Some(imm) = Self::const_as_cmn_imm12(c) {
                self.operand_to_x0(lhs);
                if use_32bit {
                    self.state.emit_fmt(format_args!("    cmn w0, #{}", imm));
                } else {
                    self.state.emit_fmt(format_args!("    cmn x0, #{}", imm));
                }
                return;
            }
        }

        // Fallback: load both into x0/x1
        self.operand_to_x0(lhs);
        self.state.emit("    mov x1, x0");
        self.operand_to_x0(rhs);
        if use_32bit {
            self.state.emit("    cmp w1, w0");
        } else {
            self.state.emit("    cmp x1, x0");
        }
    }

    // --- AArch64 large-offset helpers ---

    /// Emit a large immediate subtraction from sp. Uses x17 (IP1) as scratch.
    pub(super) fn emit_sub_sp(&mut self, n: i64) {
        if n == 0 { return; }
        if n <= 4095 {
            self.state.emit_fmt(format_args!("    sub sp, sp, #{}", n));
        } else {
            self.emit_load_imm64("x17", n);
            self.state.emit("    sub sp, sp, x17");
        }
    }

    /// Emit a large immediate addition to sp. Uses x17 (IP1) as scratch.
    pub(super) fn emit_add_sp(&mut self, n: i64) {
        if n == 0 { return; }
        if n <= 4095 {
            self.state.emit_fmt(format_args!("    add sp, sp, #{}", n));
        } else {
            self.emit_load_imm64("x17", n);
            self.state.emit("    add sp, sp, x17");
        }
    }

    /// Get the access size in bytes for an AArch64 load/store instruction and register.
    /// For str/ldr, the access size depends on the register:
    /// w registers = 4 bytes, x registers = 8 bytes,
    /// s (single-precision float) = 4 bytes, d (double-precision float) = 8 bytes,
    /// q (SIMD/quad) = 16 bytes.
    fn access_size_for_instr(instr: &str, reg: &str) -> i64 {
        match instr {
            "strb" | "ldrb" | "ldrsb" => 1,
            "strh" | "ldrh" | "ldrsh" => 2,
            "ldrsw" => 4,
            "str" | "ldr" => {
                if reg.starts_with('w') || reg.starts_with('s') {
                    4
                } else if reg.starts_with('q') {
                    16
                } else {
                    // x registers and d registers are both 8 bytes
                    8
                }
            }
            _ => 1, // conservative default
        }
    }

    /// Check if an offset is valid for unsigned immediate addressing on AArch64.
    /// The unsigned offset is a 12-bit field scaled by access size: max = 4095 * access_size.
    /// The offset must also be naturally aligned to the access size.
    pub(super) fn is_valid_imm_offset(offset: i64, instr: &str, reg: &str) -> bool {
        if offset < 0 { return false; }
        let access_size = Self::access_size_for_instr(instr, reg);
        let max_offset = 4095 * access_size;
        offset <= max_offset && offset % access_size == 0
    }

    /// Emit store to [base, #offset], handling large offsets.
    /// For large frames with x19 as frame base register, tries x19-relative addressing
    /// before falling back to the expensive movz+movk+add sequence.
    pub(super) fn emit_store_to_sp(&mut self, reg: &str, offset: i64, instr: &str) {
        // When DynAlloca is present, use x29 (frame pointer) as base.
        let base = if self.state.has_dyn_alloca { "x29" } else { "sp" };
        if Self::is_valid_imm_offset(offset, instr, reg) {
            self.state.emit_fmt(format_args!("    {} {}, [{}, #{}]", instr, reg, base, offset));
        } else if let Some(fb_offset) = self.frame_base_offset {
            // Try x19-relative addressing (x19 = sp + frame_base_offset)
            let rel_offset = offset - fb_offset;
            if Self::is_valid_imm_offset(rel_offset, instr, reg) {
                self.state.emit_fmt(format_args!("    {} {}, [x19, #{}]", instr, reg, rel_offset));
            } else {
                self.load_large_imm("x17", offset);
                self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
                self.state.emit_fmt(format_args!("    {} {}, [x17]", instr, reg));
            }
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
            self.state.emit_fmt(format_args!("    {} {}, [x17]", instr, reg));
        }
    }

    /// Emit load from [base, #offset], handling large offsets.
    /// For large frames with x19 as frame base register, tries x19-relative addressing.
    pub(super) fn emit_load_from_sp(&mut self, reg: &str, offset: i64, instr: &str) {
        let base = if self.state.has_dyn_alloca { "x29" } else { "sp" };
        if Self::is_valid_imm_offset(offset, instr, reg) {
            self.state.emit_fmt(format_args!("    {} {}, [{}, #{}]", instr, reg, base, offset));
        } else if let Some(fb_offset) = self.frame_base_offset {
            let rel_offset = offset - fb_offset;
            if Self::is_valid_imm_offset(rel_offset, instr, reg) {
                self.state.emit_fmt(format_args!("    {} {}, [x19, #{}]", instr, reg, rel_offset));
            } else {
                self.load_large_imm("x17", offset);
                self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
                self.state.emit_fmt(format_args!("    {} {}, [x17]", instr, reg));
            }
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
            self.state.emit_fmt(format_args!("    {} {}, [x17]", instr, reg));
        }
    }

    /// Emit store to [sp, #offset] using the REAL sp register, even when alloca is present.
    /// Used for storing into dynamically-allocated call stack arg areas that live at the
    /// current sp, NOT in the frame (x29-relative).
    pub(super) fn emit_store_to_raw_sp(&mut self, reg: &str, offset: i64, instr: &str) {
        if Self::is_valid_imm_offset(offset, instr, reg) {
            self.state.emit_fmt(format_args!("    {} {}, [sp, #{}]", instr, reg, offset));
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit("    add x17, sp, x17");
            self.state.emit_fmt(format_args!("    {} {}, [x17]", instr, reg));
        }
    }

    /// Emit `stp reg1, reg2, [base, #offset]` handling large offsets.
    /// Uses x19 frame base for large frames when possible.
    pub(super) fn emit_stp_to_sp(&mut self, reg1: &str, reg2: &str, offset: i64) {
        let base = if self.state.has_dyn_alloca { "x29" } else { "sp" };
        // stp supports signed offsets in [-512, 504] range (multiples of 8)
        if (-512..=504).contains(&offset) {
            self.state.emit_fmt(format_args!("    stp {}, {}, [{}, #{}]", reg1, reg2, base, offset));
        } else if let Some(fb_offset) = self.frame_base_offset {
            let rel = offset - fb_offset;
            if (-512..=504).contains(&rel) {
                self.state.emit_fmt(format_args!("    stp {}, {}, [x19, #{}]", reg1, reg2, rel));
            } else {
                self.load_large_imm("x17", offset);
                self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
                self.state.emit_fmt(format_args!("    stp {}, {}, [x17]", reg1, reg2));
            }
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
            self.state.emit_fmt(format_args!("    stp {}, {}, [x17]", reg1, reg2));
        }
    }

    pub(super) fn emit_ldp_from_sp(&mut self, reg1: &str, reg2: &str, offset: i64) {
        let base = if self.state.has_dyn_alloca { "x29" } else { "sp" };
        if (-512..=504).contains(&offset) {
            self.state.emit_fmt(format_args!("    ldp {}, {}, [{}, #{}]", reg1, reg2, base, offset));
        } else if let Some(fb_offset) = self.frame_base_offset {
            let rel = offset - fb_offset;
            if (-512..=504).contains(&rel) {
                self.state.emit_fmt(format_args!("    ldp {}, {}, [x19, #{}]", reg1, reg2, rel));
            } else {
                self.load_large_imm("x17", offset);
                self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
                self.state.emit_fmt(format_args!("    ldp {}, {}, [x17]", reg1, reg2));
            }
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
            self.state.emit_fmt(format_args!("    ldp {}, {}, [x17]", reg1, reg2));
        }
    }

    /// Emit `add dest, sp, #offset` handling large offsets.
    /// Uses x19 frame base when available, falls back to x17 scratch.
    pub(super) fn emit_add_sp_offset(&mut self, dest: &str, offset: i64) {
        let base = if self.state.has_dyn_alloca { "x29" } else { "sp" };
        if (0..=4095).contains(&offset) {
            self.state.emit_fmt(format_args!("    add {}, {}, #{}", dest, base, offset));
        } else if let Some(fb_offset) = self.frame_base_offset {
            let rel = offset - fb_offset;
            if (0..=4095).contains(&rel) {
                self.state.emit_fmt(format_args!("    add {}, x19, #{}", dest, rel));
            } else if (-4095..0).contains(&rel) {
                self.state.emit_fmt(format_args!("    sub {}, x19, #{}", dest, -rel));
            } else {
                self.load_large_imm("x17", offset);
                self.state.emit_fmt(format_args!("    add {}, {}, x17", dest, base));
            }
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit_fmt(format_args!("    add {}, {}, x17", dest, base));
        }
    }

    /// Compute the address of an alloca into `dest`, handling over-aligned allocas.
    /// For normal allocas: `dest = sp + offset`.
    /// For over-aligned allocas: `dest = (sp + offset + align-1) & -align`.
    /// `offset` is the raw stack slot offset (already adjusted for call setup if needed).
    pub(super) fn emit_alloca_addr(&mut self, dest: &str, val_id: u32, offset: i64) {
        if let Some(align) = self.state.alloca_over_align(val_id) {
            self.emit_add_sp_offset(dest, offset);
            self.load_large_imm("x17", (align - 1) as i64);
            self.state.emit_fmt(format_args!("    add {}, {}, x17", dest, dest));
            self.load_large_imm("x17", -(align as i64));
            self.state.emit_fmt(format_args!("    and {}, {}, x17", dest, dest));
        } else {
            self.emit_add_sp_offset(dest, offset);
        }
    }

    /// Emit `add dest, x29, #offset` handling large offsets.
    /// Uses x17 (IP1) as scratch for offsets > 4095.
    pub(super) fn emit_add_fp_offset(&mut self, dest: &str, offset: i64) {
        if (0..=4095).contains(&offset) {
            self.state.emit_fmt(format_args!("    add {}, x29, #{}", dest, offset));
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit_fmt(format_args!("    add {}, x29, x17", dest));
        }
    }

    /// Emit load from an arbitrary base register with offset, handling large offsets via x17.
    /// For offsets that exceed the ARM64 unsigned immediate range, materializes the
    /// effective address into x17 and loads from [x17].
    pub(super) fn emit_load_from_reg(&mut self, dest: &str, base: &str, offset: i64, instr: &str) {
        if Self::is_valid_imm_offset(offset, instr, dest) {
            self.state.emit_fmt(format_args!("    {} {}, [{}, #{}]", instr, dest, base, offset));
        } else {
            self.load_large_imm("x17", offset);
            self.state.emit_fmt(format_args!("    add x17, {}, x17", base));
            self.state.emit_fmt(format_args!("    {} {}, [x17]", instr, dest));
        }
    }

    /// Load an immediate into a register using the most efficient sequence.
    /// Handles all 64-bit values including negatives via MOVZ/MOVK or MOVN/MOVK.
    pub(super) fn load_large_imm(&mut self, reg: &str, val: i64) {
        self.emit_load_imm64(reg, val);
    }

    /// Load a 64-bit immediate value into a register using movz/movn + movk sequence.
    /// Uses MOVN (move-not) for values where most halfwords are 0xFFFF, which
    /// gives shorter sequences for negative numbers and large values.
    pub(super) fn emit_load_imm64(&mut self, reg: &str, val: i64) {
        let bits = val as u64;
        if bits == 0 {
            self.state.emit_fmt(format_args!("    mov {}, #0", reg));
            return;
        }
        if bits == 0xFFFFFFFF_FFFFFFFF {
            // All-ones: MOVN reg, #0 produces NOT(0) = 0xFFFFFFFFFFFFFFFF
            self.state.emit_fmt(format_args!("    movn {}, #0", reg));
            return;
        }

        // Extract 16-bit halfwords
        let hw: [u16; 4] = [
            (bits & 0xffff) as u16,
            ((bits >> 16) & 0xffff) as u16,
            ((bits >> 32) & 0xffff) as u16,
            ((bits >> 48) & 0xffff) as u16,
        ];

        // Count how many halfwords are 0x0000 vs 0xFFFF to pick MOVZ vs MOVN
        let zeros = hw.iter().filter(|&&h| h == 0x0000).count();
        let ones = hw.iter().filter(|&&h| h == 0xFFFF).count();

        if ones > zeros {
            // Use MOVN (move-not) strategy: start with all-ones, patch non-0xFFFF halfwords
            // MOVN sets the register to NOT(imm16 << shift)
            let mut first = true;
            for (i, &h) in hw.iter().enumerate() {
                if h != 0xFFFF {
                    let shift = i * 16;
                    let not_h = (!h) as u64 & 0xffff;
                    if first {
                        if shift == 0 {
                            self.state.emit_fmt(format_args!("    movn {}, #{}", reg, not_h));
                        } else {
                            self.state.emit_fmt(format_args!("    movn {}, #{}, lsl #{}", reg, not_h, shift));
                        }
                        first = false;
                    } else if shift == 0 {
                        self.state.emit_fmt(format_args!("    movk {}, #{}", reg, h as u64));
                    } else {
                        self.state.emit_fmt(format_args!("    movk {}, #{}, lsl #{}", reg, h as u64, shift));
                    }
                }
            }
        } else {
            // Use MOVZ (move-zero) strategy: start with all-zeros, patch non-0x0000 halfwords
            let mut first = true;
            for (i, &h) in hw.iter().enumerate() {
                if h != 0x0000 {
                    let shift = i * 16;
                    if first {
                        if shift == 0 {
                            self.state.emit_fmt(format_args!("    movz {}, #{}", reg, h as u64));
                        } else {
                            self.state.emit_fmt(format_args!("    movz {}, #{}, lsl #{}", reg, h as u64, shift));
                        }
                        first = false;
                    } else if shift == 0 {
                        self.state.emit_fmt(format_args!("    movk {}, #{}", reg, h as u64));
                    } else {
                        self.state.emit_fmt(format_args!("    movk {}, #{}, lsl #{}", reg, h as u64, shift));
                    }
                }
            }
        }
    }

    /// Emit function prologue: allocate stack and save fp/lr.
    pub(super) fn emit_prologue_arm(&mut self, frame_size: i64) {
        const PAGE_SIZE: i64 = 4096;
        if frame_size > 0 && frame_size <= 504 {
            self.state.emit_fmt(format_args!("    stp x29, x30, [sp, #-{}]!", frame_size));
            if self.state.emit_cfi {
                self.state.emit_fmt(format_args!("    .cfi_def_cfa_offset {}", frame_size));
                self.state.emit_fmt(format_args!("    .cfi_offset x29, -{}", frame_size));
                self.state.emit_fmt(format_args!("    .cfi_offset x30, -{}", frame_size - 8));
            }
        } else if frame_size > PAGE_SIZE {
            // Stack probing: for large frames, touch each page so the kernel
            // can grow the stack mapping. Without this, a single large sub
            // can skip guard pages and cause a segfault.
            let probe_label = self.state.fresh_label("stack_probe");
            self.emit_load_imm64("x17", frame_size);
            self.state.emit_fmt(format_args!("{}:", probe_label));
            self.state.emit_fmt(format_args!("    sub sp, sp, #{}", PAGE_SIZE));
            self.state.emit("    str xzr, [sp]");
            self.state.emit_fmt(format_args!("    sub x17, x17, #{}", PAGE_SIZE));
            self.state.emit_fmt(format_args!("    cmp x17, #{}", PAGE_SIZE));
            self.state.emit_fmt(format_args!("    b.hi {}", probe_label));
            self.state.emit("    sub sp, sp, x17");
            self.state.emit("    str xzr, [sp]");
            self.state.emit("    stp x29, x30, [sp]");
            if self.state.emit_cfi {
                self.state.emit_fmt(format_args!("    .cfi_def_cfa_offset {}", frame_size));
                self.state.emit_fmt(format_args!("    .cfi_offset x29, -{}", frame_size));
                self.state.emit_fmt(format_args!("    .cfi_offset x30, -{}", frame_size - 8));
            }
        } else {
            self.emit_sub_sp(frame_size);
            self.state.emit("    stp x29, x30, [sp]");
            if self.state.emit_cfi {
                self.state.emit_fmt(format_args!("    .cfi_def_cfa_offset {}", frame_size));
                self.state.emit_fmt(format_args!("    .cfi_offset x29, -{}", frame_size));
                self.state.emit_fmt(format_args!("    .cfi_offset x30, -{}", frame_size - 8));
            }
        }
        self.state.emit("    mov x29, sp");
        if self.state.emit_cfi {
            self.state.emit("    .cfi_def_cfa_register x29");
        }
    }

    /// Emit function epilogue: restore fp/lr and deallocate stack.
    pub(super) fn emit_epilogue_arm(&mut self, frame_size: i64) {
        if self.state.has_dyn_alloca {
            // DynAlloca modified SP at runtime; restore from frame pointer.
            self.state.emit("    mov sp, x29");
        }
        if frame_size > 0 && frame_size <= 504 {
            self.state.emit_fmt(format_args!("    ldp x29, x30, [sp], #{}", frame_size));
        } else {
            self.state.emit("    ldp x29, x30, [sp]");
            self.emit_add_sp(frame_size);
        }
    }

    /// Load an operand into x0.
    pub(super) fn operand_to_x0(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                self.state.reg_cache.invalidate_acc();
                match c {
                    IrConst::I8(v) => self.state.emit_fmt(format_args!("    mov x0, #{}", v)),
                    IrConst::I16(v) => self.state.emit_fmt(format_args!("    mov x0, #{}", v)),
                    IrConst::I32(v) => {
                        if *v >= -65536 && *v <= 65535 {
                            self.state.emit_fmt(format_args!("    mov x0, #{}", v));
                        } else {
                            // Sign-extend to 64-bit before loading into x0.
                            // Using the i64 path ensures negative I32 values get
                            // proper sign extension (upper 32 bits = 0xFFFFFFFF).
                            self.emit_load_imm64("x0", *v as i64);
                        }
                    }
                    IrConst::I64(v) => {
                        if *v >= -65536 && *v <= 65535 {
                            self.state.emit_fmt(format_args!("    mov x0, #{}", v));
                        } else {
                            self.emit_load_imm64("x0", *v);
                        }
                    }
                    IrConst::F32(v) => self.emit_load_imm64("x0", v.to_bits() as i64),
                    IrConst::F64(v) => self.emit_load_imm64("x0", v.to_bits() as i64),
                    IrConst::LongDouble(v, _) => self.emit_load_imm64("x0", v.to_bits() as i64),
                    IrConst::I128(v) => self.emit_load_imm64("x0", *v as i64), // truncate to 64-bit
                    IrConst::Zero => self.state.emit("    mov x0, #0"),
                }
            }
            Operand::Value(v) => {
                let is_alloca = self.state.is_alloca(v.0);
                if self.state.reg_cache.acc_has(v.0, is_alloca) {
                    return; // Cache hit — x0 already holds this value.
                }
                // Check for callee-saved register assignment.
                if let Some(&reg) = self.reg_assignments.get(&v.0) {
                    let reg_name = callee_saved_name(reg);
                    self.state.emit_fmt(format_args!("    mov x0, {}", reg_name));
                    self.state.reg_cache.set_acc(v.0, false);
                    return;
                }
                if let Some(slot) = self.state.get_slot(v.0) {
                    if is_alloca {
                        self.emit_alloca_addr("x0", v.0, slot.0);
                    } else {
                        self.emit_load_from_sp("x0", slot.0, "ldr");
                    }
                    self.state.reg_cache.set_acc(v.0, is_alloca);
                } else if self.state.reg_cache.acc_has(v.0, false) || self.state.reg_cache.acc_has(v.0, true) {
                    // Value has no slot or register but is in the accumulator cache
                    // (skip-slot optimization: immediately-consumed values stay in x0).
                } else {
                    self.state.emit("    mov x0, #0");
                    self.state.reg_cache.invalidate_acc();
                }
            }
        }
    }

    /// Store x0 to a value's destination (register or stack slot).
    pub(super) fn store_x0_to(&mut self, dest: &Value) {
        if let Some(&reg) = self.reg_assignments.get(&dest.0) {
            // Value has a callee-saved register: store only to register, skip stack.
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov {}, x0", reg_name));
        } else if let Some(slot) = self.state.get_slot(dest.0) {
            self.emit_store_to_sp("x0", slot.0, "str");
        }
        self.state.reg_cache.set_acc(dest.0, false);
    }

    // --- 128-bit integer helpers ---
    // Convention: 128-bit values use x0 (low 64 bits) and x1 (high 64 bits).
    // Stack slots for 128-bit values are 16 bytes: slot(sp) = low, slot+8(sp) = high.

    /// Load a 128-bit operand into x0 (low) : x1 (high).
    pub(super) fn operand_to_x0_x1(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I128(v) => {
                        let low = *v as u64;
                        let high = (*v >> 64) as u64;
                        self.emit_load_imm64("x0", low as i64);
                        self.emit_load_imm64("x1", high as i64);
                    }
                    IrConst::Zero => {
                        self.state.emit("    mov x0, #0");
                        self.state.emit("    mov x1, #0");
                    }
                    _ => {
                        // Other consts: load into x0, zero-extend high half
                        self.operand_to_x0(op);
                        self.state.emit("    mov x1, #0");
                    }
                }
            }
            Operand::Value(v) => {
                if let Some(slot) = self.state.get_slot(v.0) {
                    if self.state.is_alloca(v.0) {
                        // Alloca: address, not a 128-bit value itself
                        self.emit_alloca_addr("x0", v.0, slot.0);
                        self.state.emit("    mov x1, #0");
                    } else if self.state.is_i128_value(v.0) {
                        // 128-bit value in 16-byte stack slot
                        self.emit_load_from_sp("x0", slot.0, "ldr");
                        self.emit_load_from_sp("x1", slot.0 + 8, "ldr");
                    } else {
                        // Non-i128 value (e.g. shift amount): load 8 bytes, zero high
                        // Check register allocation first, since register-allocated values
                        // may not have their stack slot written.
                        if let Some(&reg) = self.reg_assignments.get(&v.0) {
                            let reg_name = callee_saved_name(reg);
                            self.state.emit_fmt(format_args!("    mov x0, {}", reg_name));
                        } else {
                            self.emit_load_from_sp("x0", slot.0, "ldr");
                        }
                        self.state.emit("    mov x1, #0");
                    }
                } else {
                    // No stack slot: check register allocation
                    if let Some(&reg) = self.reg_assignments.get(&v.0) {
                        let reg_name = callee_saved_name(reg);
                        self.state.emit_fmt(format_args!("    mov x0, {}", reg_name));
                        self.state.emit("    mov x1, #0");
                    } else {
                        self.state.emit("    mov x0, #0");
                        self.state.emit("    mov x1, #0");
                    }
                }
            }
        }
    }

    /// Store x0 (low) : x1 (high) to a 128-bit value's stack slot.
    pub(super) fn store_x0_x1_to(&mut self, dest: &Value) {
        if let Some(slot) = self.state.get_slot(dest.0) {
            self.emit_store_to_sp("x0", slot.0, "str");
            self.emit_store_to_sp("x1", slot.0 + 8, "str");
        }
    }

    /// Prepare a 128-bit binary operation: load lhs into x2:x3, rhs into x4:x5.
    /// (Uses x0:x1 as temporaries during loading.)
    pub(super) fn prep_i128_binop(&mut self, lhs: &Operand, rhs: &Operand) {
        self.operand_to_x0_x1(lhs);
        self.state.emit("    mov x2, x0");
        self.state.emit("    mov x3, x1");
        self.operand_to_x0_x1(rhs);
        self.state.emit("    mov x4, x0");
        self.state.emit("    mov x5, x1");
    }

    // emit_i128_binop and emit_i128_cmp use the shared default implementations
    // via ArchCodegen trait defaults, with per-op primitives defined in the trait impl above.

    pub(super) fn str_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 | IrType::U8 => "strb",
            IrType::I16 | IrType::U16 => "strh",
            IrType::I32 | IrType::U32 | IrType::F32 => "str",  // 32-bit store with w register
            _ => "str",  // 64-bit store with x register
        }
    }

    /// Get the appropriate register name for a given base and type.
    pub(super) fn reg_for_type(base: &str, ty: IrType) -> &'static str {
        let use_w = matches!(ty,
            IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 |
            IrType::I32 | IrType::U32 | IrType::F32
        );
        match base {
            "x0" => if use_w { "w0" } else { "x0" },
            "x1" => if use_w { "w1" } else { "x1" },
            "x2" => if use_w { "w2" } else { "x2" },
            "x3" => if use_w { "w3" } else { "x3" },
            "x4" => if use_w { "w4" } else { "x4" },
            "x5" => if use_w { "w5" } else { "x5" },
            "x6" => if use_w { "w6" } else { "x6" },
            "x7" => if use_w { "w7" } else { "x7" },
            "x8" => if use_w { "w8" } else { "x8" },
            _ => "x0",
        }
    }

    /// Parse a load instruction token into the actual ARM instruction and destination register.
    /// ARM's "ldr" instruction is width-polymorphic (the register determines access width),
    /// so load_instr_for_type returns "ldr32"/"ldr64" tokens to distinguish 32-bit from 64-bit.
    pub(super) fn arm_parse_load(instr: &'static str) -> (&'static str, &'static str) {
        match instr {
            "ldr32" => ("ldr", "w0"),
            "ldr64" => ("ldr", "x0"),
            "ldrb" | "ldrh" => (instr, "w0"),
            // ldrsb, ldrsh, ldrsw all sign-extend into x0
            _ => (instr, "x0"),
        }
    }

    /// Like arm_parse_load but returns the w/x variant of the given register number
    /// instead of hardcoded x0/w0. Used when x0 must not be clobbered.
    pub(super) fn arm_parse_load_to_reg(instr: &'static str, xreg: &'static str, wreg: &'static str) -> (&'static str, &'static str) {
        match instr {
            "ldr32" => ("ldr", wreg),
            "ldr64" => ("ldr", xreg),
            "ldrb" | "ldrh" => (instr, wreg),
            // ldrsb, ldrsh, ldrsw all sign-extend into the x-width register
            _ => (instr, xreg),
        }
    }

    // --- Intrinsic helpers (NEON) ---

    /// Load the address represented by a pointer Value into the given register.
    /// For alloca values, computes the address; for others, loads the stored pointer.
    pub(super) fn load_ptr_to_reg(&mut self, ptr: &Value, reg: &str) {
        if let Some(slot) = self.state.get_slot(ptr.0) {
            if self.state.is_alloca(ptr.0) {
                self.emit_alloca_addr(reg, ptr.0, slot.0);
            } else {
                self.emit_load_from_sp(reg, slot.0, "ldr");
            }
        }
    }

    // ── Call register arg helpers ───────────────────────────────────────────

    /// Load an operand into the given destination register, accounting for SP adjustment.
    /// When `needs_adjusted_load` is true, values must be loaded from adjusted stack offsets
    /// or callee-saved registers (since SP has been modified for stack args).
    pub(super) fn emit_load_arg_to_reg(&mut self, arg: &Operand, dest: &str, slot_adjust: i64, extra_sp_adj: i64, needs_adjusted_load: bool) {
        if needs_adjusted_load || extra_sp_adj > 0 {
            match arg {
                Operand::Value(v) => {
                    if let Some(&reg) = self.reg_assignments.get(&v.0) {
                        self.state.emit_fmt(format_args!("    mov {}, {}", dest, callee_saved_name(reg)));
                    } else if let Some(slot) = self.state.get_slot(v.0) {
                        let adjusted = slot.0 + slot_adjust + extra_sp_adj;
                        if self.state.is_alloca(v.0) {
                            self.emit_alloca_addr(dest, v.0, adjusted);
                        } else {
                            self.emit_load_from_sp(dest, adjusted, "ldr");
                        }
                    } else {
                        self.state.emit_fmt(format_args!("    mov {}, #0", dest));
                    }
                }
                Operand::Const(_) => {
                    self.operand_to_x0(arg);
                    if dest != "x0" {
                        self.state.emit_fmt(format_args!("    mov {}, x0", dest));
                    }
                }
            }
        } else {
            // For Value operands, load directly into dest to avoid clobbering x0.
            // The operand_to_x0 path unconditionally uses x0 as scratch, which
            // can destroy previously-loaded argument registers (e.g., when struct
            // arguments are reordered in a call like check(y, x)).
            match arg {
                Operand::Value(v) => {
                    if let Some(&reg) = self.reg_assignments.get(&v.0) {
                        self.state.emit_fmt(format_args!("    mov {}, {}", dest, callee_saved_name(reg)));
                    } else if let Some(slot) = self.state.get_slot(v.0) {
                        if self.state.is_alloca(v.0) {
                            self.emit_alloca_addr(dest, v.0, slot.0);
                        } else {
                            self.emit_load_from_sp(dest, slot.0, "ldr");
                        }
                    } else {
                        self.state.emit_fmt(format_args!("    mov {}, #0", dest));
                    }
                }
                Operand::Const(_) => {
                    self.operand_to_x0(arg);
                    if dest != "x0" {
                        self.state.emit_fmt(format_args!("    mov {}, x0", dest));
                    }
                }
            }
        }
    }

    /// Phase 2a: Load GP integer register args into temp registers (x9-x16).
    pub(super) fn emit_call_gp_to_temps(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                              slot_adjust: i64, needs_adjusted_load: bool) {
        let mut gp_tmp_idx = 0usize;
        for (i, arg) in args.iter().enumerate() {
            if !matches!(arg_classes[i], CallArgClass::IntReg { .. }) { continue; }
            if gp_tmp_idx >= 8 { break; }
            self.emit_load_arg_to_reg(arg, "x0", slot_adjust, 0, needs_adjusted_load);
            self.state.emit_fmt(format_args!("    mov {}, x0", ARM_TMP_REGS[gp_tmp_idx]));
            gp_tmp_idx += 1;
        }
    }

    /// Phase 2b: Load FP register args, handling F128 via temp stack + __extenddftf2.
    pub(super) fn emit_call_fp_reg_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                              arg_types: &[IrType], slot_adjust: i64, needs_adjusted_load: bool) {
        let fp_reg_assignments: Vec<(usize, usize)> = args.iter().enumerate()
            .filter(|(i, _)| matches!(arg_classes[*i], CallArgClass::FloatReg { .. } | CallArgClass::F128Reg { .. }))
            .map(|(i, _)| {
                let reg_idx = match arg_classes[i] {
                    CallArgClass::FloatReg { reg_idx } | CallArgClass::F128Reg { reg_idx } => reg_idx,
                    _ => 0,
                };
                (i, reg_idx)
            })
            .collect();

        let f128_var_count: usize = fp_reg_assignments.iter()
            .filter(|&&(arg_i, _)| matches!(arg_classes[arg_i], CallArgClass::F128Reg { .. }) && matches!(&args[arg_i], Operand::Value(_)))
            .count();
        let f128_temp_space_aligned = (f128_var_count * 16 + 15) & !15;
        if f128_temp_space_aligned > 0 {
            self.emit_sub_sp(f128_temp_space_aligned as i64);
        }

        let extra_sp_adj = f128_temp_space_aligned as i64;
        let f128_temp_slots = self.emit_call_f128_var_args(
            args, arg_classes, &fp_reg_assignments, slot_adjust, extra_sp_adj, needs_adjusted_load,
        );

        self.emit_call_f128_const_args(args, arg_classes, &fp_reg_assignments);

        for &(reg_i, temp_off) in &f128_temp_slots {
            self.state.emit_fmt(format_args!("    ldr q{}, [sp, #{}]", reg_i, temp_off));
        }
        if f128_temp_space_aligned > 0 {
            self.emit_add_sp(f128_temp_space_aligned as i64);
        }

        for &(arg_i, reg_i) in &fp_reg_assignments {
            if matches!(arg_classes[arg_i], CallArgClass::F128Reg { .. }) { continue; }
            let arg_ty = if arg_i < arg_types.len() { Some(arg_types[arg_i]) } else { None };
            self.emit_load_arg_to_reg(&args[arg_i], "x0", slot_adjust, 0, needs_adjusted_load);
            if arg_ty == Some(IrType::F32) {
                self.state.emit_fmt(format_args!("    fmov s{}, w0", reg_i));
            } else {
                self.state.emit_fmt(format_args!("    fmov d{}, x0", reg_i));
            }
        }
    }

    /// Convert F128 variable args to full-precision f128, saving to temp stack.
    pub(super) fn emit_call_f128_var_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                                fp_reg_assignments: &[(usize, usize)],
                                slot_adjust: i64, extra_sp_adj: i64,
                                needs_adjusted_load: bool) -> Vec<(usize, usize)> {
        let mut f128_temp_idx = 0usize;
        let mut f128_temp_slots: Vec<(usize, usize)> = Vec::new();
        for &(arg_i, reg_i) in fp_reg_assignments {
            if !matches!(arg_classes[arg_i], CallArgClass::F128Reg { .. }) { continue; }
            if let Operand::Value(v) = &args[arg_i] {
                let temp_off = f128_temp_idx * 16;
                let loaded_full = self.try_load_f128_full_precision(v.0, slot_adjust + extra_sp_adj, temp_off);

                if !loaded_full {
                    self.emit_load_arg_to_reg(&args[arg_i], "x0", slot_adjust, extra_sp_adj,
                        needs_adjusted_load || extra_sp_adj > 0);
                    self.state.emit("    fmov d0, x0");
                    self.state.emit("    stp x9, x10, [sp, #-16]!");
                    self.state.emit("    bl __extenddftf2");
                    self.state.emit("    ldp x9, x10, [sp], #16");
                    self.state.emit_fmt(format_args!("    str q0, [sp, #{}]", temp_off));
                }

                f128_temp_slots.push((reg_i, temp_off));
                f128_temp_idx += 1;
            }
        }
        f128_temp_slots
    }

    /// Try to load a full-precision f128 value via f128 tracking. Returns true if successful.
    pub(super) fn try_load_f128_full_precision(&mut self, value_id: u32, adjusted_slot_base: i64, temp_off: usize) -> bool {
        if let Some((src_id, offset, is_indirect)) = self.state.get_f128_source(value_id) {
            if !is_indirect {
                if let Some(src_slot) = self.state.get_slot(src_id) {
                    let adj = src_slot.0 + offset + adjusted_slot_base;
                    self.emit_load_from_sp("q0", adj, "ldr");
                    self.state.emit_fmt(format_args!("    str q0, [sp, #{}]", temp_off));
                    return true;
                }
            } else if let Some(src_slot) = self.state.get_slot(src_id) {
                let adj = src_slot.0 + adjusted_slot_base;
                self.emit_load_from_sp("x17", adj, "ldr");
                if offset != 0 {
                    if offset > 0 && offset <= 4095 {
                        self.state.emit_fmt(format_args!("    add x17, x17, #{}", offset));
                    } else {
                        self.load_large_imm("x16", offset);
                        self.state.emit("    add x17, x17, x16");
                    }
                }
                self.state.emit("    ldr q0, [x17]");
                self.state.emit_fmt(format_args!("    str q0, [sp, #{}]", temp_off));
                return true;
            }
        }
        false
    }

    /// Load F128 constants directly into target Q registers using full f128 bytes.
    pub(super) fn emit_call_f128_const_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                                  fp_reg_assignments: &[(usize, usize)]) {
        for &(arg_i, reg_i) in fp_reg_assignments {
            if !matches!(arg_classes[arg_i], CallArgClass::F128Reg { .. }) { continue; }
            if let Operand::Const(c) = &args[arg_i] {
                let bytes = match c {
                    IrConst::LongDouble(_, f128_bytes) => *f128_bytes,
                    _ => {
                        let f64_val = c.to_f64().unwrap_or(0.0);
                        crate::ir::reexports::f64_to_f128_bytes(f64_val)
                    }
                };
                let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
                self.emit_load_imm64("x0", lo as i64);
                self.emit_load_imm64("x1", hi as i64);
                self.state.emit("    stp x0, x1, [sp, #-16]!");
                self.state.emit_fmt(format_args!("    ldr q{}, [sp]", reg_i));
                self.state.emit("    add sp, sp, #16");
            }
        }
    }

    /// Phase 3: Move GP int args from temp regs to actual arg registers.
    pub(super) fn emit_call_move_temps_to_arg_regs(&mut self, args: &[Operand], arg_classes: &[CallArgClass]) {
        let mut int_reg_idx = 0usize;
        let mut gp_tmp_idx = 0usize;
        for (i, _) in args.iter().enumerate() {
            match arg_classes[i] {
                CallArgClass::I128RegPair { .. } => {
                    if int_reg_idx % 2 != 0 { int_reg_idx += 1; }
                    int_reg_idx += 2;
                }
                CallArgClass::StructByValReg { size, .. } => {
                    int_reg_idx += if size <= 8 { 1 } else { 2 };
                }
                CallArgClass::IntReg { .. } => {
                    if gp_tmp_idx < 8 && int_reg_idx < 8 {
                        self.state.emit_fmt(format_args!("    mov {}, {}", ARM_ARG_REGS[int_reg_idx], ARM_TMP_REGS[gp_tmp_idx]));
                        int_reg_idx += 1;
                    }
                    gp_tmp_idx += 1;
                }
                _ => {}
            }
        }
    }

    /// Phase 3b: Load i128 register pair args into paired arg registers.
    pub(super) fn emit_call_i128_reg_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                                slot_adjust: i64, needs_adjusted_load: bool) {
        for (i, arg) in args.iter().enumerate() {
            if let CallArgClass::I128RegPair { base_reg_idx } = arg_classes[i] {
                match arg {
                    Operand::Value(v) => {
                        if let Some(slot) = self.state.get_slot(v.0) {
                            let adj = if needs_adjusted_load { slot.0 + slot_adjust } else { slot.0 };
                            if self.state.is_alloca(v.0) {
                                self.emit_alloca_addr(ARM_ARG_REGS[base_reg_idx], v.0, adj);
                                self.state.emit_fmt(format_args!("    mov {}, #0", ARM_ARG_REGS[base_reg_idx + 1]));
                            } else {
                                self.emit_load_from_sp(ARM_ARG_REGS[base_reg_idx], adj, "ldr");
                                self.emit_load_from_sp(ARM_ARG_REGS[base_reg_idx + 1], adj + 8, "ldr");
                            }
                        }
                    }
                    Operand::Const(c) => {
                        if let IrConst::I128(v) = c {
                            self.emit_load_imm64(ARM_ARG_REGS[base_reg_idx], *v as u64 as i64);
                            self.emit_load_imm64(ARM_ARG_REGS[base_reg_idx + 1], (*v >> 64) as u64 as i64);
                        } else {
                            self.operand_to_x0(arg);
                            if base_reg_idx != 0 {
                                self.state.emit_fmt(format_args!("    mov {}, x0", ARM_ARG_REGS[base_reg_idx]));
                            }
                            self.state.emit_fmt(format_args!("    mov {}, #0", ARM_ARG_REGS[base_reg_idx + 1]));
                        }
                    }
                }
            }
        }
    }

    /// Phase 3c: Load struct-by-value register args. Loads pointer into x17,
    /// then reads struct data from [x17] into arg regs.
    pub(super) fn emit_call_struct_byval_reg_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                                        slot_adjust: i64, needs_adjusted_load: bool) {
        for (i, arg) in args.iter().enumerate() {
            if let CallArgClass::StructByValReg { base_reg_idx, size } = arg_classes[i] {
                let regs_needed = if size <= 8 { 1 } else { 2 };
                self.emit_load_arg_to_reg(arg, "x17", slot_adjust, 0, needs_adjusted_load);
                self.state.emit_fmt(format_args!("    ldr {}, [x17]", ARM_ARG_REGS[base_reg_idx]));
                if regs_needed > 1 {
                    self.state.emit_fmt(format_args!("    ldr {}, [x17, #8]", ARM_ARG_REGS[base_reg_idx + 1]));
                }
            }
        }
    }

    /// Resolve param alloca to (slot, type) for parameter `i`.
    fn resolve_param_slot(&self, func: &IrFunction, i: usize) -> Option<(StackSlot, IrType, Value)> {
        let (dest, ty) = find_param_alloca(func, i)?;
        let slot = self.state.get_slot(dest.0)?;
        Some((slot, ty, dest))
    }

    /// Save variadic function registers to save areas.
    pub(super) fn emit_save_variadic_regs(&mut self) {
        let gp_base = self.va_gp_save_offset;
        for i in (0..8).step_by(2) {
            let offset = gp_base + (i as i64) * 8;
            self.emit_stp_to_sp(&format!("x{}", i), &format!("x{}", i + 1), offset);
        }
        if !self.general_regs_only {
            let fp_base = self.va_fp_save_offset;
            for i in (0..8).step_by(2) {
                let offset = fp_base + (i as i64) * 16;
                self.emit_stp_to_sp(&format!("q{}", i), &format!("q{}", i + 1), offset);
            }
        }
    }

    /// Phase 1: Store GP register params to alloca slots.
    pub(super) fn emit_store_gp_params(&mut self, func: &IrFunction, param_classes: &[ParamClass]) {
        // AArch64 ABI: when a function uses sret, the hidden pointer comes in x8
        // (not x0). All other GP register params shift down by one so that the
        // first real argument is in x0 instead of x1.
        let sret_shift = if self.state.uses_sret { 1usize } else { 0 };

        for (i, _) in func.params.iter().enumerate() {
            let class = param_classes[i];
            if !class.uses_gp_reg() { continue; }

            let (slot, ty, _) = match self.resolve_param_slot(func, i) {
                Some(v) => v,
                None => continue,
            };

            match class {
                ParamClass::IntReg { reg_idx } => {
                    if sret_shift > 0 && reg_idx == 0 && i == 0 {
                        // sret pointer: comes in x8 on AArch64
                        self.emit_store_to_sp("x8", slot.0, "str");
                    } else {
                        let actual_idx = if reg_idx >= sret_shift { reg_idx - sret_shift } else { reg_idx };
                        let store_instr = Self::str_for_type(ty);
                        let reg = Self::reg_for_type(ARM_ARG_REGS[actual_idx], ty);
                        self.emit_store_to_sp(reg, slot.0, store_instr);
                    }
                }
                ParamClass::I128RegPair { base_reg_idx } => {
                    let actual_idx = if base_reg_idx >= sret_shift { base_reg_idx - sret_shift } else { base_reg_idx };
                    self.emit_store_to_sp(ARM_ARG_REGS[actual_idx], slot.0, "str");
                    self.emit_store_to_sp(ARM_ARG_REGS[actual_idx + 1], slot.0 + 8, "str");
                }
                ParamClass::StructByValReg { base_reg_idx, size } => {
                    let actual_idx = if base_reg_idx >= sret_shift { base_reg_idx - sret_shift } else { base_reg_idx };
                    self.emit_store_to_sp(ARM_ARG_REGS[actual_idx], slot.0, "str");
                    if size > 8 {
                        self.emit_store_to_sp(ARM_ARG_REGS[actual_idx + 1], slot.0 + 8, "str");
                    }
                }
                ParamClass::LargeStructByRefReg { reg_idx, size } => {
                    let actual_idx = if reg_idx >= sret_shift { reg_idx - sret_shift } else { reg_idx };
                    let src_reg = ARM_ARG_REGS[actual_idx];
                    let n_dwords = size.div_ceil(8);
                    for qi in 0..n_dwords {
                        let src_off = (qi * 8) as i64;
                        self.emit_load_from_reg("x9", src_reg, src_off, "ldr");
                        self.emit_store_to_sp("x9", slot.0 + src_off, "str");
                    }
                }
                _ => {}
            }
        }
    }

    /// Phase 2: Store FP register params to alloca slots.
    pub(super) fn emit_store_fp_params(&mut self, func: &IrFunction, param_classes: &[ParamClass]) {
        let has_f128_fp_params = param_classes.iter().enumerate().any(|(i, c)| {
            matches!(c, ParamClass::F128FpReg { .. }) &&
            find_param_alloca(func, i).is_some()
        });

        if has_f128_fp_params {
            self.emit_store_fp_params_with_f128(func, param_classes);
        } else {
            self.emit_store_fp_params_simple(func, param_classes);
        }
    }

    /// Store FP params when F128 params are present (save/restore q0-q7).
    fn emit_store_fp_params_with_f128(&mut self, func: &IrFunction, param_classes: &[ParamClass]) {
        self.emit_sub_sp(128);
        for i in 0..8usize {
            self.state.emit_fmt(format_args!("    str q{}, [sp, #{}]", i, i * 16));
        }

        // Process non-F128 float params first (from saved Q area).
        for (i, _) in func.params.iter().enumerate() {
            let reg_idx = match param_classes[i] {
                ParamClass::FloatReg { reg_idx } => reg_idx,
                _ => continue,
            };
            let (slot, ty, _) = match self.resolve_param_slot(func, i) {
                Some(v) => v,
                None => continue,
            };
            let fp_reg_off = (reg_idx * 16) as i64;
            if ty == IrType::F32 {
                self.state.emit_fmt(format_args!("    ldr s0, [sp, #{}]", fp_reg_off));
                self.state.emit("    fmov w9, s0");
            } else {
                self.state.emit_fmt(format_args!("    ldr d0, [sp, #{}]", fp_reg_off));
                self.state.emit("    fmov x9, d0");
            }
            self.emit_store_to_sp("x9", slot.0 + 128, "str");
        }

        // Process F128 FP reg params: store full 16-byte f128, then f64 approx.
        for (i, _) in func.params.iter().enumerate() {
            let reg_idx = match param_classes[i] {
                ParamClass::F128FpReg { reg_idx } => reg_idx,
                _ => continue,
            };
            let (slot, _, dest_val) = match self.resolve_param_slot(func, i) {
                Some(v) => v,
                None => continue,
            };
            let fp_reg_off = (reg_idx * 16) as i64;
            self.state.emit_fmt(format_args!("    ldr q0, [sp, #{}]", fp_reg_off));
            self.emit_store_to_sp("q0", slot.0 + 128, "str");
            self.state.track_f128_self(dest_val.0);
            self.state.emit("    bl __trunctfdf2");
            self.state.emit("    fmov x0, d0");
        }

        self.emit_add_sp(128);
    }

    /// Store FP params when no F128 params are present (simple path).
    fn emit_store_fp_params_simple(&mut self, func: &IrFunction, param_classes: &[ParamClass]) {
        // Use x9/w9 as scratch instead of x0/w0 to avoid clobbering GP argument
        // registers (x0-x7) that may not have been spilled yet (e.g. when mem2reg
        // promoted their allocas and emit_param_ref will read them later).
        for (i, _) in func.params.iter().enumerate() {
            let reg_idx = match param_classes[i] {
                ParamClass::FloatReg { reg_idx } => reg_idx,
                _ => continue,
            };
            let (slot, ty, _) = match self.resolve_param_slot(func, i) {
                Some(v) => v,
                None => continue,
            };
            if ty == IrType::F32 {
                self.state.emit_fmt(format_args!("    fmov w9, s{}", reg_idx));
            } else {
                self.state.emit_fmt(format_args!("    fmov x9, d{}", reg_idx));
            }
            self.emit_store_to_sp("x9", slot.0, "str");
        }
    }

    /// Phase 3: Store stack-passed params to alloca slots.
    /// Uses x9/w9 as scratch instead of x0/w0 to avoid clobbering GP argument
    /// registers (x0-x7) that may not have been spilled yet (e.g. when mem2reg
    /// promoted their allocas and emit_param_ref will read them later).
    pub(super) fn emit_store_stack_params(&mut self, func: &IrFunction, param_classes: &[ParamClass]) {
        let frame_size = self.current_frame_size;
        for (i, _) in func.params.iter().enumerate() {
            let class = param_classes[i];
            if !class.is_stack() { continue; }

            let (slot, ty, dest_val) = match self.resolve_param_slot(func, i) {
                Some(v) => v,
                None => continue,
            };

            match class {
                ParamClass::StructStack { offset, size } | ParamClass::LargeStructStack { offset, size } => {
                    let caller_offset = frame_size + offset;
                    for qi in 0..size.div_ceil(8) {
                        let off = qi as i64 * 8;
                        self.emit_load_from_sp("x9", caller_offset + off, "ldr");
                        self.emit_store_to_sp("x9", slot.0 + off, "str");
                    }
                }
                ParamClass::F128Stack { offset } => {
                    let caller_offset = frame_size + offset;
                    self.emit_load_from_sp("x9", caller_offset, "ldr");
                    self.emit_store_to_sp("x9", slot.0, "str");
                    self.emit_load_from_sp("x9", caller_offset + 8, "ldr");
                    self.emit_store_to_sp("x9", slot.0 + 8, "str");
                    self.state.track_f128_self(dest_val.0);
                }
                ParamClass::I128Stack { offset } => {
                    let caller_offset = frame_size + offset;
                    self.emit_load_from_sp("x9", caller_offset, "ldr");
                    self.emit_store_to_sp("x9", slot.0, "str");
                    self.emit_load_from_sp("x9", caller_offset + 8, "ldr");
                    self.emit_store_to_sp("x9", slot.0 + 8, "str");
                }
                ParamClass::StackScalar { offset } => {
                    let caller_offset = frame_size + offset;
                    // Load from caller stack with extending load, then store
                    // full 64 bits so the slot is valid for any later ldr.
                    // Use x9/w9 to avoid clobbering GP argument registers.
                    let load_instr = self.load_instr_for_type_impl(ty);
                    let (arm_load, dest_reg) = Self::arm_parse_load_to_reg(load_instr, "x9", "w9");
                    self.emit_load_from_sp(dest_reg, caller_offset, arm_load);
                    self.emit_store_to_sp("x9", slot.0, "str");
                }
                ParamClass::LargeStructByRefStack { offset, size } => {
                    let caller_offset = frame_size + offset;
                    self.emit_load_from_sp("x9", caller_offset, "ldr");
                    for qi in 0..size.div_ceil(8) {
                        let off = (qi * 8) as i64;
                        self.emit_load_from_reg("x10", "x9", off, "ldr");
                        self.emit_store_to_sp("x10", slot.0 + off, "str");
                    }
                }
                _ => {}
            }
        }
    }
}

pub(super) const ARM_ARG_REGS: [&str; 8] = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];
const ARM_TMP_REGS: [&str; 8] = ["x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16"];

impl ArchCodegen for ArmCodegen {
    fn state(&mut self) -> &mut CodegenState { &mut self.state }
    fn state_ref(&self) -> &CodegenState { &self.state }

    fn get_phys_reg_for_value(&self, val_id: u32) -> Option<PhysReg> {
        self.reg_assignments.get(&val_id).copied()
    }

    fn emit_reg_to_reg_move(&mut self, src: PhysReg, dest: PhysReg) {
        let s_name = callee_saved_name(src);
        let d_name = callee_saved_name(dest);
        self.state.emit_fmt(format_args!("    mov {}, {}", d_name, s_name));
    }

    fn emit_acc_to_phys_reg(&mut self, dest: PhysReg) {
        let d_name = callee_saved_name(dest);
        self.state.emit_fmt(format_args!("    mov {}, x0", d_name));
    }

    fn jump_mnemonic(&self) -> &'static str { "b" }
    fn trap_instruction(&self) -> &'static str { "brk #0" }

    fn emit_branch_nonzero(&mut self, label: &str) {
        let skip = self.state.fresh_label("skip");
        self.state.emit_fmt(format_args!("    cbz x0, {}", skip));
        self.state.emit_fmt(format_args!("    b {}", label));
        self.state.emit_fmt(format_args!("{}:", skip));
    }

    fn emit_jump_indirect(&mut self) {
        self.state.emit("    br x0");
    }

    fn emit_switch_case_branch(&mut self, case_val: i64, label: &str, ty: IrType) {
        let use_32bit = matches!(ty, IrType::I32 | IrType::U32 | IrType::I16 | IrType::U16 | IrType::I8 | IrType::U8);
        if use_32bit {
            self.emit_load_imm64("w1", case_val as i32 as i64);
            self.state.emit("    cmp w0, w1");
        } else {
            self.emit_load_imm64("x1", case_val);
            self.state.emit("    cmp x0, x1");
        }
        let skip = self.state.fresh_label("skip");
        self.state.emit_fmt(format_args!("    b.ne {}", skip));
        self.state.emit_fmt(format_args!("    b {}", label));
        self.state.emit_fmt(format_args!("{}:", skip));
    }

    fn emit_switch_jump_table(&mut self, val: &Operand, cases: &[(i64, BlockId)], default: &BlockId, _ty: IrType) {
        use crate::backend::traits::build_jump_table;
        let (table, min_val, range) = build_jump_table(cases, default);
        let table_label = self.state.fresh_label("jt");
        let default_label = default.as_label();
        self.operand_to_x0(val);
        if min_val != 0 {
            if min_val > 0 && min_val <= 4095 {
                self.state.emit_fmt(format_args!("    sub x0, x0, #{}", min_val));
            } else if min_val < 0 && (-min_val) <= 4095 {
                self.state.emit_fmt(format_args!("    add x0, x0, #{}", -min_val));
            } else {
                self.load_large_imm("x17", min_val);
                self.state.emit("    sub x0, x0, x17");
            }
        }
        if range <= 4095 {
            self.state.emit_fmt(format_args!("    cmp x0, #{}", range));
        } else {
            self.load_large_imm("x17", range as i64);
            self.state.emit("    cmp x0, x17");
        }
        let range_skip = self.state.fresh_label("range_ok");
        self.state.emit_fmt(format_args!("    b.lo {}", range_skip));
        self.state.emit_fmt(format_args!("    b {}", default_label));
        self.state.emit_fmt(format_args!("{}:", range_skip));
        self.state.emit_fmt(format_args!("    adrp x17, {}", table_label));
        self.state.emit_fmt(format_args!("    add x17, x17, :lo12:{}", table_label));
        self.state.emit("    ldr w16, [x17, x0, lsl #2]");
        self.state.emit("    add x17, x17, w16, sxtw");
        self.state.emit("    br x17");
        self.state.emit(".section .rodata");
        self.state.emit(".align 2");
        self.state.emit_fmt(format_args!("{}:", table_label));
        for target in &table {
            let target_label = target.as_label();
            self.state.emit_fmt(format_args!("    .word {} - {}", target_label, table_label));
        }
        let sect = self.state.current_text_section.clone();
        self.state.emit_fmt(format_args!(".section {},\"ax\",@progbits", sect));
        self.state.reg_cache.invalidate_all();
    }

    fn ptr_directive(&self) -> PtrDirective { PtrDirective::Xword }
    fn function_type_directive(&self) -> &'static str { "%function" }

    // ---- Standard trait methods (kept inline - arch-specific) ----
    fn emit_load_operand(&mut self, op: &Operand) { self.operand_to_x0(op); }
    fn emit_store_result(&mut self, dest: &Value) { self.store_x0_to(dest); }
    fn emit_save_acc(&mut self) { self.state.emit("    mov x1, x0"); }
    fn emit_add_secondary_to_acc(&mut self) { self.state.emit("    add x0, x1, x0"); }
    fn emit_gep_add_const_to_acc(&mut self, offset: i64) { if offset != 0 { self.emit_add_imm_to_acc_impl(offset); } }
    fn emit_acc_to_secondary(&mut self) { self.state.emit("    mov x1, x0"); }
    fn emit_memcpy_store_dest_from_acc(&mut self) { }
    fn emit_memcpy_store_src_from_acc(&mut self) { self.state.emit("    mov x10, x9"); }
    fn emit_call_spill_fptr(&mut self, func_ptr: &Operand) {
        self.operand_to_x0(func_ptr);
        self.state.emit("    str x0, [sp, #-16]!");
    }
    fn emit_call_fptr_spill_size(&self) -> usize { 16 }
    fn emit_call_move_f32_to_acc(&mut self) { self.state.emit("    fmov w0, s0"); }
    fn emit_call_move_f64_to_acc(&mut self) { self.state.emit("    fmov x0, d0"); }

    // AArch64 ABI: sret pointer goes in x8, not x0.
    fn sret_uses_dedicated_reg(&self) -> bool { true }
    fn emit_call_sret_setup(&mut self, sret_operand: &Operand, total_sp_adjust: i64) {
        let slot_adjust = if self.state.has_dyn_alloca { 0 } else { total_sp_adjust };
        let needs_adjusted = total_sp_adjust > 0;
        self.emit_load_arg_to_reg(sret_operand, "x8", slot_adjust, 0, needs_adjusted);
    }

    // ---- Inline asm / intrinsics (kept inline - has extra logic) ----
    fn emit_inline_asm(&mut self, template: &str, outputs: &[(String, Value, Option<String>)], inputs: &[(String, Operand, Option<String>)], clobbers: &[String], operand_types: &[IrType], goto_labels: &[(String, BlockId)], input_symbols: &[Option<String>]) {
        emit_inline_asm_common(self, template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols);
    }
    fn emit_intrinsic(&mut self, dest: &Option<Value>, op: &IntrinsicOp, dest_ptr: &Option<Value>, args: &[Operand]) {
        self.emit_intrinsic_arm(dest, op, dest_ptr, args);
    }

    // ---- Float binop body uses different name ----
    fn emit_float_binop_impl(&mut self, mnemonic: &str, ty: IrType) { self.emit_float_binop_body(mnemonic, ty) }

    // All remaining methods delegate to self.method_name_impl(args...)
    delegate_to_impl! {
        // prologue
        fn calculate_stack_space(&mut self, func: &IrFunction) -> i64 => calculate_stack_space_impl;
        fn aligned_frame_size(&self, raw_space: i64) -> i64 => aligned_frame_size_impl;
        fn emit_prologue(&mut self, func: &IrFunction, frame_size: i64) => emit_prologue_impl;
        fn emit_epilogue(&mut self, frame_size: i64) => emit_epilogue_impl;
        fn emit_store_params(&mut self, func: &IrFunction) => emit_store_params_impl;
        fn emit_param_ref(&mut self, dest: &Value, param_idx: usize, ty: IrType) => emit_param_ref_impl;
        fn emit_epilogue_and_ret(&mut self, frame_size: i64) => emit_epilogue_and_ret_impl;
        fn store_instr_for_type(&self, ty: IrType) -> &'static str => store_instr_for_type_impl;
        fn load_instr_for_type(&self, ty: IrType) -> &'static str => load_instr_for_type_impl;
        // memory
        fn emit_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) => emit_store_impl;
        fn emit_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) => emit_load_impl;
        fn emit_store_with_const_offset(&mut self, val: &Operand, base: &Value, offset: i64, ty: IrType) => emit_store_with_const_offset_impl;
        fn emit_load_with_const_offset(&mut self, dest: &Value, base: &Value, offset: i64, ty: IrType) => emit_load_with_const_offset_impl;
        fn emit_typed_store_to_slot(&mut self, instr: &'static str, ty: IrType, slot: StackSlot) => emit_typed_store_to_slot_impl;
        fn emit_typed_load_from_slot(&mut self, instr: &'static str, slot: StackSlot) => emit_typed_load_from_slot_impl;
        fn emit_load_ptr_from_slot(&mut self, slot: StackSlot, val_id: u32) => emit_load_ptr_from_slot_impl;
        fn emit_typed_store_indirect(&mut self, instr: &'static str, ty: IrType) => emit_typed_store_indirect_impl;
        fn emit_typed_load_indirect(&mut self, instr: &'static str) => emit_typed_load_indirect_impl;
        fn emit_add_offset_to_addr_reg(&mut self, offset: i64) => emit_add_offset_to_addr_reg_impl;
        fn emit_slot_addr_to_secondary(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) => emit_slot_addr_to_secondary_impl;
        fn emit_gep_direct_const(&mut self, slot: StackSlot, offset: i64) => emit_gep_direct_const_impl;
        fn emit_gep_indirect_const(&mut self, slot: StackSlot, offset: i64, val_id: u32) => emit_gep_indirect_const_impl;
        fn emit_add_imm_to_acc(&mut self, imm: i64) => emit_add_imm_to_acc_impl;
        fn emit_round_up_acc_to_16(&mut self) => emit_round_up_acc_to_16_impl;
        fn emit_sub_sp_by_acc(&mut self) => emit_sub_sp_by_acc_impl;
        fn emit_mov_sp_to_acc(&mut self) => emit_mov_sp_to_acc_impl;
        fn emit_mov_acc_to_sp(&mut self) => emit_mov_acc_to_sp_impl;
        fn emit_align_acc(&mut self, align: usize) => emit_align_acc_impl;
        fn emit_memcpy_load_dest_addr(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) => emit_memcpy_load_dest_addr_impl;
        fn emit_memcpy_load_src_addr(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) => emit_memcpy_load_src_addr_impl;
        fn emit_alloca_aligned_addr(&mut self, slot: StackSlot, val_id: u32) => emit_alloca_aligned_addr_impl;
        fn emit_alloca_aligned_addr_to_acc(&mut self, slot: StackSlot, val_id: u32) => emit_alloca_aligned_addr_to_acc_impl;
        fn emit_memcpy_impl(&mut self, size: usize) => emit_memcpy_impl_impl;
        // alu
        fn emit_float_neg(&mut self, ty: IrType) => emit_float_neg_impl;
        fn emit_f128_neg(&mut self, dest: &Value, src: &Operand) => emit_f128_neg_impl;
        fn emit_int_neg(&mut self, ty: IrType) => emit_int_neg_impl;
        fn emit_int_not(&mut self, ty: IrType) => emit_int_not_impl;
        fn emit_int_clz(&mut self, ty: IrType) => emit_int_clz_impl;
        fn emit_int_ctz(&mut self, ty: IrType) => emit_int_ctz_impl;
        fn emit_int_bswap(&mut self, ty: IrType) => emit_int_bswap_impl;
        fn emit_int_popcount(&mut self, ty: IrType) => emit_int_popcount_impl;
        fn emit_int_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_int_binop_impl;
        fn emit_copy_i128(&mut self, dest: &Value, src: &Operand) => emit_copy_i128_impl;
        // comparison
        fn emit_float_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_float_cmp_impl;
        fn emit_int_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_int_cmp_impl;
        fn emit_f128_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand) => emit_f128_cmp_impl;
        fn emit_fused_cmp_branch(&mut self, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType, true_label: &str, false_label: &str) => emit_fused_cmp_branch_impl;
        fn emit_select(&mut self, dest: &Value, cond: &Operand, true_val: &Operand, false_val: &Operand, ty: IrType) => emit_select_impl;
        // calls
        fn call_abi_config(&self) -> CallAbiConfig => call_abi_config_impl;
        fn emit_call_compute_stack_space(&self, arg_classes: &[CallArgClass], arg_types: &[IrType]) -> usize => emit_call_compute_stack_space_impl;
        fn emit_call_f128_pre_convert(&mut self, args: &[Operand], arg_classes: &[CallArgClass], arg_types: &[IrType], stack_arg_space: usize) -> usize => emit_call_f128_pre_convert_impl;
        fn emit_call_stack_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass], arg_types: &[IrType], stack_arg_space: usize, fptr_spill: usize, f128_temp_space: usize) -> i64 => emit_call_stack_args_impl;
        fn emit_call_reg_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass], arg_types: &[IrType], total_sp_adjust: i64, f128_temp_space: usize, stack_arg_space: usize, struct_arg_riscv_float_classes: &[Option<crate::common::types::RiscvFloatClass>]) => emit_call_reg_args_impl;
        fn emit_call_instruction(&mut self, direct_name: Option<&str>, func_ptr: Option<&Operand>, indirect: bool, stack_arg_space: usize) => emit_call_instruction_impl;
        fn emit_call_cleanup(&mut self, stack_arg_space: usize, f128_temp_space: usize, indirect: bool) => emit_call_cleanup_impl;
        fn emit_call_store_i128_result(&mut self, dest: &Value) => emit_call_store_i128_result_impl;
        fn emit_call_store_f128_result(&mut self, dest: &Value) => emit_call_store_f128_result_impl;
        // globals
        fn emit_global_addr(&mut self, dest: &Value, name: &str) => emit_global_addr_impl;
        fn emit_label_addr(&mut self, dest: &Value, label: &str) => emit_label_addr_impl;
        fn emit_tls_global_addr(&mut self, dest: &Value, name: &str) => emit_tls_global_addr_impl;
        // cast
        fn emit_cast_instrs(&mut self, from_ty: IrType, to_ty: IrType) => emit_cast_instrs_impl;
        fn emit_cast(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) => emit_cast_impl;
        // variadic
        fn emit_va_arg(&mut self, dest: &Value, va_list_ptr: &Value, result_ty: IrType) => emit_va_arg_impl;
        fn emit_va_start(&mut self, va_list_ptr: &Value) => emit_va_start_impl;
        fn emit_va_copy(&mut self, dest_ptr: &Value, src_ptr: &Value) => emit_va_copy_impl;
        fn emit_va_arg_struct(&mut self, dest_ptr: &Value, va_list_ptr: &Value, size: usize) => emit_va_arg_struct_impl;
        // returns
        fn current_return_type(&self) -> IrType => current_return_type_impl;
        fn emit_return_i128_to_regs(&mut self) => emit_return_i128_to_regs_impl;
        fn emit_return_f128_to_reg(&mut self) => emit_return_f128_to_reg_impl;
        fn emit_return(&mut self, val: Option<&Operand>, frame_size: i64) => emit_return_impl;
        fn emit_return_f32_to_reg(&mut self) => emit_return_f32_to_reg_impl;
        fn emit_return_f64_to_reg(&mut self) => emit_return_f64_to_reg_impl;
        fn emit_return_int_to_reg(&mut self) => emit_return_int_to_reg_impl;
        fn emit_get_return_f64_second(&mut self, dest: &Value) => emit_get_return_f64_second_impl;
        fn emit_set_return_f64_second(&mut self, src: &Operand) => emit_set_return_f64_second_impl;
        fn emit_get_return_f32_second(&mut self, dest: &Value) => emit_get_return_f32_second_impl;
        fn emit_set_return_f32_second(&mut self, src: &Operand) => emit_set_return_f32_second_impl;
        fn emit_get_return_f128_second(&mut self, dest: &Value) => emit_get_return_f128_second_impl;
        fn emit_set_return_f128_second(&mut self, src: &Operand) => emit_set_return_f128_second_impl;
        // atomics
        fn emit_atomic_rmw(&mut self, dest: &Value, op: AtomicRmwOp, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_rmw_impl;
        fn emit_atomic_cmpxchg(&mut self, dest: &Value, ptr: &Operand, expected: &Operand, desired: &Operand, ty: IrType, success_ordering: AtomicOrdering, failure_ordering: AtomicOrdering, returns_bool: bool) => emit_atomic_cmpxchg_impl;
        fn emit_atomic_load(&mut self, dest: &Value, ptr: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_load_impl;
        fn emit_atomic_store(&mut self, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_store_impl;
        fn emit_fence(&mut self, ordering: AtomicOrdering) => emit_fence_impl;
        // float binop
        fn emit_float_binop(&mut self, dest: &Value, op: crate::backend::cast::FloatOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_float_binop_impl;
        // i128 ops
        fn emit_load_acc_pair(&mut self, op: &Operand) => emit_load_acc_pair_impl;
        fn emit_store_acc_pair(&mut self, dest: &Value) => emit_store_acc_pair_impl;
        fn emit_store_pair_to_slot(&mut self, slot: StackSlot) => emit_store_pair_to_slot_impl;
        fn emit_load_pair_from_slot(&mut self, slot: StackSlot) => emit_load_pair_from_slot_impl;
        fn emit_save_acc_pair(&mut self) => emit_save_acc_pair_impl;
        fn emit_store_pair_indirect(&mut self) => emit_store_pair_indirect_impl;
        fn emit_load_pair_indirect(&mut self) => emit_load_pair_indirect_impl;
        fn emit_i128_neg(&mut self) => emit_i128_neg_impl;
        fn emit_i128_not(&mut self) => emit_i128_not_impl;
        fn emit_sign_extend_acc_high(&mut self) => emit_sign_extend_acc_high_impl;
        fn emit_zero_acc_high(&mut self) => emit_zero_acc_high_impl;
        fn emit_i128_prep_binop(&mut self, lhs: &Operand, rhs: &Operand) => emit_i128_prep_binop_impl;
        fn emit_i128_add(&mut self) => emit_i128_add_impl;
        fn emit_i128_sub(&mut self) => emit_i128_sub_impl;
        fn emit_i128_mul(&mut self) => emit_i128_mul_impl;
        fn emit_i128_and(&mut self) => emit_i128_and_impl;
        fn emit_i128_or(&mut self) => emit_i128_or_impl;
        fn emit_i128_xor(&mut self) => emit_i128_xor_impl;
        fn emit_i128_shl(&mut self) => emit_i128_shl_impl;
        fn emit_i128_lshr(&mut self) => emit_i128_lshr_impl;
        fn emit_i128_ashr(&mut self) => emit_i128_ashr_impl;
        fn emit_i128_prep_shift_lhs(&mut self, lhs: &Operand) => emit_i128_prep_shift_lhs_impl;
        fn emit_i128_shl_const(&mut self, amount: u32) => emit_i128_shl_const_impl;
        fn emit_i128_lshr_const(&mut self, amount: u32) => emit_i128_lshr_const_impl;
        fn emit_i128_ashr_const(&mut self, amount: u32) => emit_i128_ashr_const_impl;
        fn emit_i128_divrem_call(&mut self, func_name: &str, lhs: &Operand, rhs: &Operand) => emit_i128_divrem_call_impl;
        fn emit_i128_store_result(&mut self, dest: &Value) => emit_i128_store_result_impl;
        fn emit_i128_to_float_call(&mut self, src: &Operand, from_signed: bool, to_ty: IrType) => emit_i128_to_float_call_impl;
        fn emit_float_to_i128_call(&mut self, src: &Operand, to_signed: bool, from_ty: IrType) => emit_float_to_i128_call_impl;
        fn emit_i128_cmp_eq(&mut self, is_ne: bool) => emit_i128_cmp_eq_impl;
        fn emit_i128_cmp_ordered(&mut self, op: IrCmpOp) => emit_i128_cmp_ordered_impl;
        fn emit_i128_cmp_store_result(&mut self, dest: &Value) => emit_i128_cmp_store_result_impl;
    }
}

impl Default for ArmCodegen {
    fn default() -> Self {
        Self::new()
    }
}

