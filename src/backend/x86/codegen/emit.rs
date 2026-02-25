use crate::delegate_to_impl;
use crate::ir::reexports::{
    AtomicOrdering,
    AtomicRmwOp,
    BlockId,
    IntrinsicOp,
    IrBinOp,
    IrCmpOp,
    IrConst,
    IrFunction,
    IrUnaryOp,
    Operand,
    Value,
};
use crate::common::types::{AddressSpace, IrType};
use crate::common::fx_hash::FxHashMap;
use crate::backend::common::PtrDirective;
use crate::backend::state::{CodegenState, StackSlot};
use crate::backend::traits::ArchCodegen;
use crate::backend::call_abi::{CallAbiConfig, CallArgClass};
use crate::backend::cast::FloatOp;
use crate::backend::inline_asm::emit_inline_asm_common;
use crate::backend::regalloc::PhysReg;

/// x86-64 callee-saved registers available for register allocation.
/// System V AMD64 ABI callee-saved: rbx, r12, r13, r14, r15.
/// rbp is the frame pointer and cannot be allocated.
/// PhysReg encoding: 1=rbx, 2=r12, 3=r13, 4=r14, 5=r15.
pub(super) const X86_CALLEE_SAVED: [PhysReg; 5] = [
    PhysReg(1), PhysReg(2), PhysReg(3), PhysReg(4), PhysReg(5),
];

/// x86-64 caller-saved registers available for allocation to values that
/// do NOT span function calls. These registers are destroyed by calls, so
/// they can only hold values between calls. No prologue/epilogue save is needed.
///
/// PhysReg encoding: 10=r11, 11=r10, 12=r8, 13=r9, 14=rdi, 15=rsi
/// (IDs 10+ to avoid overlap with callee-saved 1-5)
///
/// rdi and rsi are included because the SysV ABI makes them caller-saved.
/// After the prologue stores incoming parameters from rdi/rsi to their stack
/// slots, these registers are free for use within the function body.
/// Memcpy (rep movsb) and VaArg operations that clobber rdi/rsi are treated
/// as call points by the liveness analysis, so values allocated to rdi/rsi
/// will never have live ranges spanning those operations.
pub(super) const X86_CALLER_SAVED: [PhysReg; 6] = [
    PhysReg(10), PhysReg(11), PhysReg(12), PhysReg(13),
    PhysReg(14), PhysReg(15),
];

/// Convert a 64-bit register name string to its 32-bit sub-register name.
/// Used for `xorl %eXX, %eXX` zeroing idiom (shorter encoding, breaks dependencies).
fn reg_name_to_32(name: &str) -> &'static str {
    match name {
        "rax" => "eax", "rbx" => "ebx", "rcx" => "ecx", "rdx" => "edx",
        "rsi" => "esi", "rdi" => "edi", "rsp" => "esp", "rbp" => "ebp",
        "r8" => "r8d", "r9" => "r9d", "r10" => "r10d", "r11" => "r11d",
        "r12" => "r12d", "r13" => "r13d", "r14" => "r14d", "r15" => "r15d",
        _ => unreachable!("invalid 64-bit register name: {}", name),
    }
}

/// Map a PhysReg index to its x86-64 register name.
/// Handles both callee-saved (1-5) and caller-saved (10-15) registers.
pub(super) fn phys_reg_name(reg: PhysReg) -> &'static str {
    match reg.0 {
        1 => "rbx", 2 => "r12", 3 => "r13", 4 => "r14", 5 => "r15",
        10 => "r11", 11 => "r10", 12 => "r8", 13 => "r9",
        14 => "rdi", 15 => "rsi",
        _ => unreachable!("invalid x86 register index {}", reg.0),
    }
}

/// Map a PhysReg index to its x86-64 32-bit sub-register name.
/// Handles both callee-saved (1-5) and caller-saved (10-15) registers.
pub(super) fn phys_reg_name_32(reg: PhysReg) -> &'static str {
    match reg.0 {
        1 => "ebx", 2 => "r12d", 3 => "r13d", 4 => "r14d", 5 => "r15d",
        10 => "r11d", 11 => "r10d", 12 => "r8d", 13 => "r9d",
        14 => "edi", 15 => "esi",
        _ => unreachable!("invalid x86 register index {}", reg.0),
    }
}

/// Scan inline asm instructions in a function and collect any callee-saved
/// registers that are used via specific constraints or listed in clobbers.
/// These must be saved/restored in the function prologue/epilogue.
///
/// On x86-64, the scratch register pool has 8 caller-saved GP registers
/// (rcx, rdx, rsi, rdi, r8-r11). When an inline asm block needs more than
/// 8 generic "r" operands, the scratch allocator overflows into callee-saved
/// registers (r12-r15). We use `_with_overflow` to detect this and
/// conservatively mark all callee-saved registers as clobbered only when
/// overflow is likely, avoiding unnecessary save/restore in common cases.
pub(super) fn collect_inline_asm_callee_saved_x86(func: &IrFunction, used: &mut Vec<PhysReg>) {
    fn clobber_to_phys(name: &str) -> Option<PhysReg> {
        match name {
            "rbx" | "ebx" | "bx" | "bl" | "bh" => Some(PhysReg(1)),
            "r12" | "r12d" | "r12w" | "r12b" => Some(PhysReg(2)),
            "r13" | "r13d" | "r13w" | "r13b" => Some(PhysReg(3)),
            "r14" | "r14d" | "r14w" | "r14b" => Some(PhysReg(4)),
            "r15" | "r15d" | "r15w" | "r15b" => Some(PhysReg(5)),
            _ => None,
        }
    }
    crate::backend::stack_layout::collect_inline_asm_callee_saved_with_overflow(
        func, used, constraint_to_callee_saved_x86, clobber_to_phys,
        &X86_CALLEE_SAVED, 8,
    );
}

/// Check if a constraint string refers to a specific x86-64 callee-saved register.
fn constraint_to_callee_saved_x86(constraint: &str) -> Option<PhysReg> {
    // Handle explicit register constraint: {regname}
    if constraint.starts_with('{') && constraint.ends_with('}') {
        let reg = &constraint[1..constraint.len()-1];
        return match reg {
            "rbx" | "ebx" | "bx" | "bl" | "bh" => Some(PhysReg(1)),
            "r12" | "r12d" | "r12w" | "r12b" => Some(PhysReg(2)),
            "r13" | "r13d" | "r13w" | "r13b" => Some(PhysReg(3)),
            "r14" | "r14d" | "r14w" | "r14b" => Some(PhysReg(4)),
            "r15" | "r15d" | "r15w" | "r15b" => Some(PhysReg(5)),
            _ => None,
        };
    }
    // Check single-character constraint letters
    // Note: 'a','c','d','S','D' are caller-saved, no save needed
    for ch in constraint.chars() {
        if ch == 'b' {
            return Some(PhysReg(1)); // rbx
        }
    }
    None
}

/// Map an ALU binary op (Add/Sub/And/Or/Xor) to its x86 mnemonic base.
pub(super) fn alu_mnemonic(op: IrBinOp) -> &'static str {
    match op {
        IrBinOp::Add => "add",
        IrBinOp::Sub => "sub",
        IrBinOp::And => "and",
        IrBinOp::Or => "or",
        IrBinOp::Xor => "xor",
        _ => unreachable!("not a simple ALU op"),
    }
}

/// Map a shift op to its x86 32-bit and 64-bit mnemonic.
pub(super) fn shift_mnemonic(op: IrBinOp) -> (&'static str, &'static str) {
    match op {
        IrBinOp::Shl => ("shll", "shlq"),
        IrBinOp::AShr => ("sarl", "sarq"),
        IrBinOp::LShr => ("shrl", "shrq"),
        _ => unreachable!("not a shift op"),
    }
}

/// x86-64 code generator. Implements the ArchCodegen trait for the shared framework.
/// Uses System V AMD64 ABI with linear scan register allocation for callee-saved registers.
pub struct X86Codegen {
    pub(crate) state: CodegenState,
    pub(super) current_return_type: IrType,
    /// SysV ABI eightbyte classification for the current function's return struct.
    /// Set in prologue, used in emit_return_i128_to_regs for the function's own return.
    pub(super) func_ret_classes: Vec<crate::common::types::EightbyteClass>,
    /// SysV ABI eightbyte classification for the most recent call-site's return struct.
    /// Set before processing a call's result, used in emit_call_store_result.
    pub(super) call_ret_classes: Vec<crate::common::types::EightbyteClass>,
    /// For variadic functions: number of named integer/pointer parameters (excluding long double)
    pub(super) num_named_int_params: usize,
    /// For variadic functions: number of named float/double parameters (excluding long double)
    pub(super) num_named_fp_params: usize,
    /// For variadic functions: total bytes of named parameters that are always stack-passed
    /// (e.g. long double = 16 bytes each, struct params passed by value on stack)
    pub(super) num_named_stack_bytes: usize,
    /// For variadic functions: stack offset of the register save area (negative from rbp)
    pub(super) reg_save_area_offset: i64,
    /// Whether the current function is variadic
    pub(super) is_variadic: bool,
    /// Scratch register index for inline asm allocation (GP registers)
    pub(super) asm_scratch_idx: usize,
    /// Scratch register index for inline asm allocation (XMM registers)
    pub(super) asm_xmm_scratch_idx: usize,
    /// Register allocation results for the current function.
    /// Maps value ID -> callee-saved register assignment.
    pub(super) reg_assignments: FxHashMap<u32, PhysReg>,
    /// Which callee-saved registers are used and need save/restore.
    pub(super) used_callee_saved: Vec<PhysReg>,
    /// Whether SSE is disabled (-mno-sse). When true, variadic prologues skip
    /// XMM saves and va_start sets fp_offset to overflow immediately.
    pub(super) no_sse: bool,
}

impl X86Codegen {
    pub fn new() -> Self {
        Self {
            state: CodegenState::new(),
            current_return_type: IrType::I64,
            func_ret_classes: Vec::new(),
            call_ret_classes: Vec::new(),
            num_named_int_params: 0,
            num_named_fp_params: 0,
            num_named_stack_bytes: 0,
            reg_save_area_offset: 0,
            is_variadic: false,
            asm_scratch_idx: 0,
            asm_xmm_scratch_idx: 0,
            reg_assignments: FxHashMap::default(),
            used_callee_saved: Vec::new(),
            no_sse: false,
        }
    }

    /// Enable position-independent code generation.
    pub fn set_pic(&mut self, pic: bool) {
        self.state.pic_mode = pic;
    }

    /// Enable function return thunk (-mfunction-return=thunk-extern).
    pub fn set_function_return_thunk(&mut self, enabled: bool) {
        self.state.function_return_thunk = enabled;
    }

    /// Enable indirect branch thunk (-mindirect-branch=thunk-extern).
    pub fn set_indirect_branch_thunk(&mut self, enabled: bool) {
        self.state.indirect_branch_thunk = enabled;
    }

    /// Set patchable function entry configuration (-fpatchable-function-entry=N,M).
    pub fn set_patchable_function_entry(&mut self, entry: Option<(u32, u32)>) {
        self.state.patchable_function_entry = entry;
    }

    /// Enable CF protection branch (-fcf-protection=branch) to emit endbr64.
    pub fn set_cf_protection_branch(&mut self, enabled: bool) {
        self.state.cf_protection_branch = enabled;
    }

    /// Enable kernel code model (-mcmodel=kernel). All symbols are assumed
    /// to be in the negative 2GB of the virtual address space.
    pub fn set_code_model_kernel(&mut self, enabled: bool) {
        self.state.code_model_kernel = enabled;
    }

    /// Disable jump table emission (-fno-jump-tables). All switch statements
    /// use compare-and-branch chains instead of indirect jumps.
    pub fn set_no_jump_tables(&mut self, enabled: bool) {
        self.state.no_jump_tables = enabled;
    }

    /// Disable SSE (-mno-sse). Prevents emission of any SSE/XMM instructions.
    pub fn set_no_sse(&mut self, enabled: bool) {
        self.no_sse = enabled;
    }

    /// Apply all relevant options from a `CodegenOptions` struct.
    pub fn apply_options(&mut self, opts: &crate::backend::CodegenOptions) {
        self.set_pic(opts.pic);
        self.set_function_return_thunk(opts.function_return_thunk);
        self.set_indirect_branch_thunk(opts.indirect_branch_thunk);
        self.set_patchable_function_entry(opts.patchable_function_entry);
        self.set_cf_protection_branch(opts.cf_protection_branch);
        self.set_no_sse(opts.no_sse);
        self.set_code_model_kernel(opts.code_model_kernel);
        self.set_no_jump_tables(opts.no_jump_tables);
        self.state.emit_cfi = opts.emit_cfi;
    }

    // --- x86 helper methods ---

    /// Get the callee-saved register assigned to an operand, if any.
    pub(super) fn operand_reg(&self, op: &Operand) -> Option<PhysReg> {
        match op {
            Operand::Value(v) => self.reg_assignments.get(&v.0).copied(),
            _ => None,
        }
    }

    /// Get the callee-saved register assigned to a destination value, if any.
    pub(super) fn dest_reg(&self, dest: &Value) -> Option<PhysReg> {
        self.reg_assignments.get(&dest.0).copied()
    }

    /// Emit sign-extension from 32-bit to 64-bit register if the type is signed.
    /// Used after 32-bit ALU operations on callee-saved registers.
    pub(super) fn emit_sext32_if_needed(&mut self, name_32: &str, name_64: &str, is_unsigned: bool) {
        if !is_unsigned {
            self.state.out.emit_instr_reg_reg("    movslq", name_32, name_64);
        }
    }

    /// Emit a comparison instruction, optionally using 32-bit form for I32/U32 types.
    /// When `use_32bit` is true, emits `cmpl` with 32-bit register names instead of `cmpq`.
    pub(super) fn emit_int_cmp_insn_typed(&mut self, lhs: &Operand, rhs: &Operand, use_32bit: bool) {
        let cmp_instr = if use_32bit { "cmpl" } else { "cmpq" };
        let lhs_phys = self.operand_reg(lhs);
        let rhs_phys = self.operand_reg(rhs);
        if let (Some(lhs_r), Some(rhs_r)) = (lhs_phys, rhs_phys) {
            // Both in callee-saved registers: compare directly
            let lhs_name = if use_32bit { phys_reg_name_32(lhs_r) } else { phys_reg_name(lhs_r) };
            let rhs_name = if use_32bit { phys_reg_name_32(rhs_r) } else { phys_reg_name(rhs_r) };
            self.state.emit_fmt(format_args!("    {} %{}, %{}", cmp_instr, rhs_name, lhs_name));
        } else if let Some(imm) = Self::const_as_imm32(rhs) {
            if imm == 0 {
                // test %reg, %reg is shorter than cmp $0, %reg and sets flags identically
                let test_instr = if use_32bit { "testl" } else { "testq" };
                if let Some(lhs_r) = lhs_phys {
                    let lhs_name = if use_32bit { phys_reg_name_32(lhs_r) } else { phys_reg_name(lhs_r) };
                    self.state.emit_fmt(format_args!("    {} %{}, %{}", test_instr, lhs_name, lhs_name));
                } else {
                    self.operand_to_rax(lhs);
                    let reg = if use_32bit { "eax" } else { "rax" };
                    self.state.emit_fmt(format_args!("    {} %{}, %{}", test_instr, reg, reg));
                }
            } else if let Some(lhs_r) = lhs_phys {
                let lhs_name = if use_32bit { phys_reg_name_32(lhs_r) } else { phys_reg_name(lhs_r) };
                self.state.emit_fmt(format_args!("    {} ${}, %{}", cmp_instr, imm, lhs_name));
            } else {
                self.operand_to_rax(lhs);
                let reg = if use_32bit { "eax" } else { "rax" };
                self.state.emit_fmt(format_args!("    {} ${}, %{}", cmp_instr, imm, reg));
            }
        } else if let Some(lhs_r) = lhs_phys {
            let lhs_name = if use_32bit { phys_reg_name_32(lhs_r) } else { phys_reg_name(lhs_r) };
            self.operand_to_rcx(rhs);
            let rcx = if use_32bit { "ecx" } else { "rcx" };
            self.state.emit_fmt(format_args!("    {} %{}, %{}", cmp_instr, rcx, lhs_name));
        } else if let Some(rhs_r) = rhs_phys {
            let rhs_name = if use_32bit { phys_reg_name_32(rhs_r) } else { phys_reg_name(rhs_r) };
            self.operand_to_rax(lhs);
            let reg = if use_32bit { "eax" } else { "rax" };
            self.state.emit_fmt(format_args!("    {} %{}, %{}", cmp_instr, rhs_name, reg));
        } else {
            self.operand_to_rax(lhs);
            self.operand_to_rcx(rhs);
            let (rcx, rax) = if use_32bit { ("ecx", "eax") } else { ("rcx", "rax") };
            self.state.emit_fmt(format_args!("    {} %{}, %{}", cmp_instr, rcx, rax));
        }
    }

    /// Load an operand into a specific callee-saved register.
    /// Handles constants, register-allocated values, and stack values.
    pub(super) fn operand_to_callee_reg(&mut self, op: &Operand, target: PhysReg) {
        let target_name = phys_reg_name(target);
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I8(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", phys_reg_name_32(target))),
                    IrConst::I16(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", phys_reg_name_32(target))),
                    IrConst::I32(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", phys_reg_name_32(target))),
                    IrConst::I64(0) => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", phys_reg_name_32(target))),
                    IrConst::I8(v) => self.state.emit_fmt(format_args!("    movq ${}, %{}", *v as i64, target_name)),
                    IrConst::I16(v) => self.state.emit_fmt(format_args!("    movq ${}, %{}", *v as i64, target_name)),
                    IrConst::I32(v) => self.state.emit_fmt(format_args!("    movq ${}, %{}", *v as i64, target_name)),
                    IrConst::I64(v) => {
                        if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", *v, target_name);
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", *v, target_name);
                        }
                    }
                    IrConst::Zero => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", phys_reg_name_32(target))),
                    _ => {
                        // For float/i128 constants, fall back to loading to rax and moving
                        self.operand_to_rax(op);
                        self.state.out.emit_instr_reg_reg("    movq", "rax", target_name);
                    }
                }
            }
            Operand::Value(v) => {
                if let Some(&reg) = self.reg_assignments.get(&v.0) {
                    if reg.0 != target.0 {
                        let src_name = phys_reg_name(reg);
                        self.state.out.emit_instr_reg_reg("    movq", src_name, target_name);
                    }
                    // If same register, nothing to do
                } else if let Some(slot) = self.state.get_slot(v.0) {
                    if self.state.is_alloca(v.0) {
                        if let Some(align) = self.state.alloca_over_align(v.0) {
                            // Over-aligned alloca: compute aligned address
                            self.state.out.emit_instr_rbp_reg("    leaq", slot.0, target_name);
                            self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, target_name);
                            self.state.out.emit_instr_imm_reg("    andq", -(align as i64), target_name);
                        } else {
                            self.state.out.emit_instr_rbp_reg("    leaq", slot.0, target_name);
                        }
                    } else if self.state.small_slot_values.contains(&v.0) {
                        let target_32 = phys_reg_name_32(target);
                        self.state.out.emit_instr_rbp_reg("    movl", slot.0, target_32);
                    } else {
                        self.state.out.emit_instr_rbp_reg("    movq", slot.0, target_name);
                    }
                } else if self.state.reg_cache.acc_has(v.0, false) || self.state.reg_cache.acc_has(v.0, true) {
                    self.state.out.emit_instr_reg_reg("    movq", "rax", target_name);
                } else {
                    let target_32 = phys_reg_name_32(target);
                    self.state.out.emit_instr_reg_reg("    xorl", target_32, target_32);
                }
            }
        }
    }

    /// Load an operand into %rax. Uses the register cache to skip the load
    /// if the value is already in %rax.
    pub(super) fn operand_to_rax(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                self.state.reg_cache.invalidate_acc();
                match c {
                    IrConst::I8(v) if *v == 0 => self.state.emit("    xorl %eax, %eax"),
                    IrConst::I16(v) if *v == 0 => self.state.emit("    xorl %eax, %eax"),
                    IrConst::I32(v) if *v == 0 => self.state.emit("    xorl %eax, %eax"),
                    IrConst::I64(0) => self.state.emit("    xorl %eax, %eax"),
                    IrConst::I8(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, "rax"),
                    IrConst::I16(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, "rax"),
                    IrConst::I32(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, "rax"),
                    IrConst::I64(v) => {
                        if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", *v, "rax");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", *v, "rax");
                        }
                    }
                    IrConst::F32(v) => {
                        let bits = v.to_bits() as u64;
                        if bits == 0 {
                            self.state.emit("    xorl %eax, %eax");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movq", bits as i64, "rax");
                        }
                    }
                    IrConst::F64(v) => {
                        let bits = v.to_bits();
                        if bits == 0 {
                            self.state.emit("    xorl %eax, %eax");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", bits as i64, "rax");
                        }
                    }
                    // LongDouble at computation level is treated as F64
                    IrConst::LongDouble(v, _) => {
                        let bits = v.to_bits();
                        if bits == 0 {
                            self.state.emit("    xorl %eax, %eax");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", bits as i64, "rax");
                        }
                    }
                    IrConst::I128(v) => {
                        // Truncate to low 64 bits for rax-only path
                        let low = *v as i64;
                        if low == 0 {
                            self.state.emit("    xorl %eax, %eax");
                        } else if low >= i32::MIN as i64 && low <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", low, "rax");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", low, "rax");
                        }
                    }
                    IrConst::Zero => self.state.emit("    xorl %eax, %eax"),
                }
            }
            Operand::Value(v) => {
                let is_alloca = self.state.is_alloca(v.0);
                // Check cache: skip load if value is already in %rax
                if self.state.reg_cache.acc_has(v.0, is_alloca) {
                    return;
                }
                // Check register allocation: load from callee-saved register
                if let Some(&reg) = self.reg_assignments.get(&v.0) {
                    let reg_name = phys_reg_name(reg);
                    self.state.out.emit_instr_reg_reg("    movq", reg_name, "rax");
                    self.state.reg_cache.set_acc(v.0, false);
                } else if self.state.get_slot(v.0).is_some() {
                    self.value_to_reg(v, "rax");
                    self.state.reg_cache.set_acc(v.0, is_alloca);
                } else {
                    self.state.emit("    xorl %eax, %eax");
                    self.state.reg_cache.invalidate_acc();
                }
            }
        }
    }

    /// Store %rax to a value's location (register or stack slot).
    /// Register-only strategy: if the value has a register assignment (callee-saved or caller-saved),
    /// store ONLY to the register (skip the stack write). This eliminates redundant
    /// memory stores for register-allocated values. Values without a register
    /// assignment are stored to their stack slot as before.
    pub(super) fn store_rax_to(&mut self, dest: &Value) {
        if let Some(&reg) = self.reg_assignments.get(&dest.0) {
            // Value has a callee-saved register: store only to register, skip stack.
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", "rax", reg_name);
        } else if let Some(slot) = self.state.get_slot(dest.0) {
            // No register: store to stack slot.
            // Use movl for 4-byte small slots (I8-I32, U8-U32, F32).
            if self.state.small_slot_values.contains(&dest.0) {
                self.state.out.emit_instr_reg_rbp("    movl", "eax", slot.0);
            } else {
                self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
            }
        }
        // After storing to dest, %rax still holds dest's value
        self.state.reg_cache.set_acc(dest.0, false);
    }

    /// Load an operand into an arbitrary named register (e.g. "rdi", "rsi", "r8").
    /// For register-allocated values, emits a direct reg-to-reg move (or nothing
    /// if the value is already in the target). For constants, emits an immediate
    /// load. For stack values, emits a movq/leaq from the stack slot.
    /// This avoids the accumulator-routing pattern of operand_to_rax + movq rax, target.
    pub(super) fn operand_to_named_reg(&mut self, op: &Operand, target: &str) {
        match op {
            Operand::Const(c) => {
                let target_32 = reg_name_to_32(target);
                match c {
                    IrConst::I8(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", target_32)),
                    IrConst::I16(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", target_32)),
                    IrConst::I32(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", target_32)),
                    IrConst::I64(0) => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", target_32)),
                    IrConst::I8(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, target),
                    IrConst::I16(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, target),
                    IrConst::I32(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, target),
                    IrConst::I64(v) => {
                        if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", *v, target);
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", *v, target);
                        }
                    }
                    IrConst::Zero => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", target_32)),
                    _ => {
                        // For float/i128 constants, fall back to loading to rax and moving
                        self.operand_to_rax(op);
                        if target != "rax" {
                            self.state.out.emit_instr_reg_reg("    movq", "rax", target);
                        }
                    }
                }
            }
            Operand::Value(v) => {
                // Check register allocation: direct reg-to-reg
                if let Some(&reg) = self.reg_assignments.get(&v.0) {
                    let reg_name = phys_reg_name(reg);
                    if reg_name != target {
                        self.state.out.emit_instr_reg_reg("    movq", reg_name, target);
                    }
                    // If already in target register, nothing to do
                } else if self.state.get_slot(v.0).is_some() {
                    self.value_to_reg(v, target);
                } else if self.state.reg_cache.acc_has(v.0, false) || self.state.reg_cache.acc_has(v.0, true) {
                    if target != "rax" {
                        self.state.out.emit_instr_reg_reg("    movq", "rax", target);
                    }
                } else {
                    let target_32 = reg_name_to_32(target);
                    self.state.out.emit_instr_reg_reg("    xorl", target_32, target_32);
                }
            }
        }
    }

    /// Load an operand directly into %rcx, avoiding the push/pop pattern.
    /// This is the key optimization: instead of loading to rax, pushing, loading
    /// the other operand to rax, moving rax->rcx, then popping rax, we load
    /// directly to rcx with a single instruction.
    pub(super) fn operand_to_rcx(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I8(v) if *v == 0 => self.state.emit("    xorl %ecx, %ecx"),
                    IrConst::I16(v) if *v == 0 => self.state.emit("    xorl %ecx, %ecx"),
                    IrConst::I32(v) if *v == 0 => self.state.emit("    xorl %ecx, %ecx"),
                    IrConst::I64(0) => self.state.emit("    xorl %ecx, %ecx"),
                    IrConst::I8(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, "rcx"),
                    IrConst::I16(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, "rcx"),
                    IrConst::I32(v) => self.state.out.emit_instr_imm_reg("    movq", *v as i64, "rcx"),
                    IrConst::I64(v) => {
                        if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", *v, "rcx");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", *v, "rcx");
                        }
                    }
                    IrConst::F32(v) => {
                        let bits = v.to_bits() as u64;
                        if bits == 0 {
                            self.state.emit("    xorl %ecx, %ecx");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movq", bits as i64, "rcx");
                        }
                    }
                    IrConst::F64(v) => {
                        let bits = v.to_bits();
                        if bits == 0 {
                            self.state.emit("    xorl %ecx, %ecx");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", bits as i64, "rcx");
                        }
                    }
                    IrConst::LongDouble(v, _) => {
                        let bits = v.to_bits();
                        if bits == 0 {
                            self.state.emit("    xorl %ecx, %ecx");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", bits as i64, "rcx");
                        }
                    }
                    IrConst::I128(v) => {
                        let low = *v as i64;
                        if low == 0 {
                            self.state.emit("    xorl %ecx, %ecx");
                        } else if low >= i32::MIN as i64 && low <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", low, "rcx");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", low, "rcx");
                        }
                    }
                    IrConst::Zero => self.state.emit("    xorl %ecx, %ecx"),
                }
            }
            Operand::Value(v) => {
                // Check register allocation: load from callee-saved register
                if let Some(&reg) = self.reg_assignments.get(&v.0) {
                    let reg_name = phys_reg_name(reg);
                    self.state.out.emit_instr_reg_reg("    movq", reg_name, "rcx");
                } else if self.state.get_slot(v.0).is_some() {
                    self.value_to_reg(v, "rcx");
                } else if self.state.reg_cache.acc_has(v.0, false) || self.state.reg_cache.acc_has(v.0, true) {
                    self.state.out.emit_instr_reg_reg("    movq", "rax", "rcx");
                } else {
                    self.state.emit("    xorl %ecx, %ecx");
                }
            }
        }
    }

    /// Check if an operand is a small constant that fits in a 32-bit immediate.
    /// Returns the immediate value if it fits, None otherwise.
    pub(super) fn const_as_imm32(op: &Operand) -> Option<i64> {
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
                if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
                    Some(val)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Load a Value into a named register. For allocas, loads the address (leaq);
    /// for register-allocated values, copies from the callee-saved register;
    /// for regular values, loads the full 8 bytes from the stack slot with movq.
    ///
    /// The prologue ensures that sub-64-bit parameters are sign/zero-extended
    /// to 64 bits before being stored, so movq always reads valid data.
    pub(super) fn value_to_reg(&mut self, val: &Value, reg: &str) {
        // Check register allocation first (allocas are never register-allocated)
        if let Some(&phys_reg) = self.reg_assignments.get(&val.0) {
            let reg_name = phys_reg_name(phys_reg);
            if reg_name != reg {
                self.state.out.emit_instr_reg_reg("    movq", reg_name, reg);
            }
            return;
        }
        if let Some(slot) = self.state.get_slot(val.0) {
            if self.state.is_alloca(val.0) {
                if let Some(align) = self.state.alloca_over_align(val.0) {
                    // Over-aligned alloca: compute aligned address within the
                    // oversized stack slot. The slot has (align - 1) extra bytes
                    // to guarantee we can find an aligned address within it.
                    self.state.out.emit_instr_rbp_reg("    leaq", slot.0, reg);
                    self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, reg);
                    self.state.out.emit_instr_imm_reg("    andq", -(align as i64), reg);
                } else {
                    self.state.out.emit_instr_rbp_reg("    leaq", slot.0, reg);
                }
            } else if self.state.small_slot_values.contains(&val.0) {
                // 4-byte slot: use movl which zero-extends to 64 bits.
                let reg32 = reg_name_to_32(reg);
                self.state.out.emit_instr_rbp_reg("    movl", slot.0, reg32);
            } else {
                self.state.out.emit_instr_rbp_reg("    movq", slot.0, reg);
            }
        }
    }

    // --- 128-bit integer helpers ---
    // Convention: 128-bit values use %rax (low 64 bits) and %rdx (high 64 bits).
    // Stack slots for 128-bit values are 16 bytes: slot(%rbp) = low, slot+8(%rbp) = high.

    /// Load a 128-bit operand into %rax (low) and %rdx (high).
    pub(super) fn operand_to_rax_rdx(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I128(v) => {
                        let low = *v as u64 as i64;
                        let high = (*v >> 64) as u64 as i64;
                        if low == 0 {
                            self.state.emit("    xorl %eax, %eax");
                        } else if low >= i32::MIN as i64 && low <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", low, "rax");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", low, "rax");
                        }
                        if high == 0 {
                            self.state.emit("    xorl %edx, %edx");
                        } else if high >= i32::MIN as i64 && high <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", high, "rdx");
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", high, "rdx");
                        }
                    }
                    IrConst::Zero => {
                        self.state.emit("    xorl %eax, %eax");
                        self.state.emit("    xorl %edx, %edx");
                    }
                    _ => {
                        // Smaller constant: load into rax, zero/sign-extend to rdx
                        self.operand_to_rax(op);
                        self.state.emit("    xorl %edx, %edx");
                    }
                }
            }
            Operand::Value(v) => {
                if let Some(slot) = self.state.get_slot(v.0) {
                    if self.state.is_alloca(v.0) {
                        // Alloca: load the address (not a 128-bit value itself)
                        if let Some(align) = self.state.alloca_over_align(v.0) {
                            // Over-aligned alloca: compute aligned address
                            self.state.out.emit_instr_rbp_reg("    leaq", slot.0, "rax");
                            self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, "rax");
                            self.state.out.emit_instr_imm_reg("    andq", -(align as i64), "rax");
                        } else {
                            self.state.out.emit_instr_rbp_reg("    leaq", slot.0, "rax");
                        }
                        self.state.emit("    xorl %edx, %edx");
                    } else if self.state.is_i128_value(v.0) {
                        // 128-bit value in 16-byte stack slot
                        self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rax");
                        self.state.out.emit_instr_rbp_reg("    movq", slot.0 + 8, "rdx");
                    } else {
                        // Non-i128 value (e.g. shift amount): load 8 bytes, zero-extend rdx
                        // Check register allocation first, since register-allocated values
                        // may not have their stack slot written.
                        if let Some(&reg) = self.reg_assignments.get(&v.0) {
                            let reg_name = phys_reg_name(reg);
                            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rax");
                        } else {
                            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rax");
                        }
                        self.state.emit("    xorl %edx, %edx");
                    }
                } else {
                    // No stack slot: check register allocation
                    if let Some(&reg) = self.reg_assignments.get(&v.0) {
                        let reg_name = phys_reg_name(reg);
                        self.state.out.emit_instr_reg_reg("    movq", reg_name, "rax");
                        self.state.emit("    xorl %edx, %edx");
                    } else {
                        self.state.emit("    xorl %eax, %eax");
                        self.state.emit("    xorl %edx, %edx");
                    }
                }
            }
        }
    }

    /// Store %rax:%rdx (128-bit) to a value's 16-byte stack slot.
    pub(super) fn store_rax_rdx_to(&mut self, dest: &Value) {
        if let Some(slot) = self.state.get_slot(dest.0) {
            self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
            self.state.out.emit_instr_reg_rbp("    movq", "rdx", slot.0 + 8);
        }
        // rax holds only the low 64 bits of an i128, not a valid scalar IR value.
        self.state.reg_cache.invalidate_all();
    }

    /// Get the store instruction mnemonic for a given type.
    pub(super) fn mov_store_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 | IrType::U8 => "movb",
            IrType::I16 | IrType::U16 => "movw",
            IrType::I32 | IrType::U32 | IrType::F32 => "movl",
            _ => "movq",
        }
    }

    /// Get the load instruction mnemonic for a given type.
    pub(super) fn mov_load_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 => "movsbq",
            IrType::U8 => "movzbq",
            IrType::I16 => "movswq",
            IrType::U16 => "movzwq",
            IrType::I32 => "movslq",
            IrType::U32 | IrType::F32 => "movl",     // movl zero-extends to 64-bit implicitly
            _ => "movq",
        }
    }

    /// Destination register for loads. U32/F32 use movl which needs %eax.
    pub(super) fn load_dest_reg(ty: IrType) -> &'static str {
        match ty {
            IrType::U32 | IrType::F32 => "%eax",
            _ => "%rax",
        }
    }

    /// Map base register name + type to sized sub-register.
    pub(super) fn reg_for_type(base_reg: &str, ty: IrType) -> &'static str {
        let (r8, r16, r32, r64) = match base_reg {
            "rax" => ("al", "ax", "eax", "rax"),
            "rcx" => ("cl", "cx", "ecx", "rcx"),
            "rdx" => ("dl", "dx", "edx", "rdx"),
            "rdi" => ("dil", "di", "edi", "rdi"),
            "rsi" => ("sil", "si", "esi", "rsi"),
            "r8"  => ("r8b", "r8w", "r8d", "r8"),
            "r9"  => ("r9b", "r9w", "r9d", "r9"),
            _ => return "rax",
        };
        match ty {
            IrType::I8 | IrType::U8 => r8,
            IrType::I16 | IrType::U16 => r16,
            IrType::I32 | IrType::U32 | IrType::F32 => r32,
            _ => r64,
        }
    }


    /// Get the type suffix for lock-prefixed instructions (b, w, l, q).
    pub(super) fn type_suffix(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 | IrType::U8 => "b",
            IrType::I16 | IrType::U16 => "w",
            IrType::I32 | IrType::U32 => "l",
            _ => "q",
        }
    }

    /// Emit a cmpxchg-based loop for atomic sub/and/or/xor/nand.
    /// Expects: rax = operand val, rcx = ptr address.
    /// After: rax = old value.
    pub(super) fn emit_x86_atomic_op_loop(&mut self, ty: IrType, op: &str) {
        // Save val to r8
        self.state.emit("    movq %rax, %r8"); // r8 = val
        // Load old value
        let load_instr = Self::mov_load_for_type(ty);
        let load_dest = Self::load_dest_reg(ty);
        self.state.emit_fmt(format_args!("    {} (%rcx), {}", load_instr, load_dest));
        // Loop: rax = old, compute new = op(old, val), try cmpxchg
        let label_id = self.state.next_label_id();
        let loop_label = format!(".Latomic_loop_{}", label_id);
        self.state.out.emit_named_label(&loop_label);
        // rdx = rax (old)
        self.state.emit("    movq %rax, %rdx");
        // Apply operation: rdx = op(rdx, r8)
        let size_suffix = Self::type_suffix(ty);
        let rdx_reg = Self::reg_for_type("rdx", ty);
        let r8_reg = match ty {
            IrType::I8 | IrType::U8 => "r8b",
            IrType::I16 | IrType::U16 => "r8w",
            IrType::I32 | IrType::U32 => "r8d",
            _ => "r8",
        };
        match op {
            "sub" => self.state.emit_fmt(format_args!("    sub{} %{}, %{}", size_suffix, r8_reg, rdx_reg)),
            "and" => self.state.emit_fmt(format_args!("    and{} %{}, %{}", size_suffix, r8_reg, rdx_reg)),
            "or"  => self.state.emit_fmt(format_args!("    or{} %{}, %{}", size_suffix, r8_reg, rdx_reg)),
            "xor" => self.state.emit_fmt(format_args!("    xor{} %{}, %{}", size_suffix, r8_reg, rdx_reg)),
            "nand" => {
                self.state.emit_fmt(format_args!("    and{} %{}, %{}", size_suffix, r8_reg, rdx_reg));
                self.state.emit_fmt(format_args!("    not{} %{}", size_suffix, rdx_reg));
            }
            _ => {}
        }
        // Try cmpxchg: if [rcx] == rax (old), set [rcx] = rdx (new), else rax = [rcx]
        self.state.emit_fmt(format_args!("    lock cmpxchg{} %{}, (%rcx)", size_suffix, rdx_reg));
        self.state.out.emit_jcc_label("    jne", &loop_label);
        // rax = old value on success
    }

    /// Load i128 operands for binary ops: lhs → rax:rdx, rhs → rcx:rsi.
    pub(super) fn prep_i128_binop(&mut self, lhs: &Operand, rhs: &Operand) {
        self.operand_to_rax_rdx(lhs);
        self.state.emit("    pushq %rdx");
        self.state.emit("    pushq %rax");
        self.operand_to_rax_rdx(rhs);
        self.state.emit("    movq %rax, %rcx");
        self.state.emit("    movq %rdx, %rsi");
        self.state.emit("    popq %rax");
        self.state.emit("    popq %rdx");
        self.state.reg_cache.invalidate_all();
    }

    /// Load an operand value into any GP register (returned as string).
    /// Uses rcx as the scratch register.
    pub(super) fn operand_to_reg(&mut self, op: &Operand, reg: &str) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I8(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", reg_name_to_32(reg))),
                    IrConst::I16(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", reg_name_to_32(reg))),
                    IrConst::I32(v) if *v == 0 => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", reg_name_to_32(reg))),
                    IrConst::I64(0) => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", reg_name_to_32(reg))),
                    IrConst::I8(v) => self.state.emit_fmt(format_args!("    movq ${}, %{}", *v as i64, reg)),
                    IrConst::I16(v) => self.state.emit_fmt(format_args!("    movq ${}, %{}", *v as i64, reg)),
                    IrConst::I32(v) => self.state.emit_fmt(format_args!("    movq ${}, %{}", *v as i64, reg)),
                    IrConst::I64(v) => {
                        if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                            self.state.out.emit_instr_imm_reg("    movq", *v, reg);
                        } else {
                            self.state.out.emit_instr_imm_reg("    movabsq", *v, reg);
                        }
                    }
                    _ => self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", reg_name_to_32(reg))),
                }
            }
            Operand::Value(v) => {
                self.value_to_reg(v, reg);
            }
        }
    }

    /// Extract an immediate integer value from an operand.
    /// Used for SSE/AES instructions that require compile-time immediate operands.
    pub(super) fn operand_to_imm_i64(&self, op: &Operand) -> i64 {
        match op {
            Operand::Const(c) => match c {
                IrConst::I8(v) => *v as i64,
                IrConst::I16(v) => *v as i64,
                IrConst::I32(v) => *v as i64,
                IrConst::I64(v) => *v,
                _ => 0,
            },
            Operand::Value(_) => {
                // TODO: this shouldn't happen for compile-time immediate arguments;
                // the frontend should always fold these to constants.
                0
            }
        }
    }

    /// Emit comment annotations for callee-saved registers listed in inline asm
    /// clobber lists. The peephole pass's `eliminate_unused_callee_saves` scans
    /// function bodies for textual register references (e.g., "%rbx") to decide
    /// whether a callee-saved register save/restore can be eliminated. Without
    /// these annotations, an inline asm that clobbers a callee-saved register
    /// (but doesn't mention it in the emitted assembly text) would have its
    /// save/restore incorrectly removed.
    pub(super) fn emit_callee_saved_clobber_annotations(&mut self, clobbers: &[String]) {
        for clobber in clobbers {
            let reg_name = match clobber.as_str() {
                "rbx" | "ebx" | "bx" | "bl" | "bh" => Some("%rbx"),
                "r12" | "r12d" | "r12w" | "r12b" => Some("%r12"),
                "r13" | "r13d" | "r13w" | "r13b" => Some("%r13"),
                "r14" | "r14d" | "r14w" | "r14b" => Some("%r14"),
                "r15" | "r15d" | "r15w" | "r15b" => Some("%r15"),
                _ => None,
            };
            if let Some(reg) = reg_name {
                self.state.emit_fmt(format_args!("    # asm clobber {}", reg));
            }
        }
    }

    /// LEA scale factor for multiply strength reduction.
    /// Returns the LEA scale factor for multipliers 3, 5, 9 (which decompose
    /// as reg + reg*2, reg + reg*4, reg + reg*8 respectively).
    pub(super) fn lea_scale_for_mul(imm: i64) -> Option<u8> {
        match imm {
            3 => Some(2),
            5 => Some(4),
            9 => Some(8),
            _ => None,
        }
    }

    /// Register-direct path for simple ALU ops (add/sub/and/or/xor/mul).
    pub(super) fn emit_alu_reg_direct(&mut self, op: IrBinOp, lhs: &Operand, rhs: &Operand,
                           dest_phys: PhysReg, use_32bit: bool, is_unsigned: bool) {
        let dest_name = phys_reg_name(dest_phys);
        let dest_name_32 = phys_reg_name_32(dest_phys);

        // Immediate form
        if let Some(imm) = Self::const_as_imm32(rhs) {
            self.operand_to_callee_reg(lhs, dest_phys);
            if op == IrBinOp::Mul {
                // LEA strength reduction: replace imul by 3/5/9 with lea.
                // lea (%reg, %reg, scale), %reg computes reg + reg*scale = reg*(scale+1).
                // lea has 1-cycle latency vs 3 cycles for imul on modern x86.
                if let Some(scale) = Self::lea_scale_for_mul(imm) {
                    if use_32bit {
                        self.state.emit_fmt(format_args!(
                            "    leal (%{}, %{}, {}), %{}", dest_name_32, dest_name_32, scale, dest_name_32));
                        self.emit_sext32_if_needed(dest_name_32, dest_name, is_unsigned);
                    } else {
                        self.state.emit_fmt(format_args!(
                            "    leaq (%{}, %{}, {}), %{}", dest_name, dest_name, scale, dest_name));
                    }
                } else if use_32bit {
                    self.state.emit_fmt(format_args!("    imull ${}, %{}, %{}", imm, dest_name_32, dest_name_32));
                    self.emit_sext32_if_needed(dest_name_32, dest_name, is_unsigned);
                } else {
                    self.state.emit_fmt(format_args!("    imulq ${}, %{}, %{}", imm, dest_name, dest_name));
                }
            } else {
                let mnemonic = alu_mnemonic(op);
                if use_32bit && matches!(op, IrBinOp::Add | IrBinOp::Sub) {
                    self.state.emit_fmt(format_args!("    {}l ${}, %{}", mnemonic, imm, dest_name_32));
                    self.emit_sext32_if_needed(dest_name_32, dest_name, is_unsigned);
                } else {
                    self.state.emit_fmt(format_args!("    {}q ${}, %{}", mnemonic, imm, dest_name));
                }
            }
            self.state.reg_cache.invalidate_acc();
            return;
        }

        // Register-register form
        let rhs_phys = self.operand_reg(rhs);
        let rhs_conflicts = rhs_phys.is_some_and(|r| r.0 == dest_phys.0);
        let (rhs_reg_name, rhs_reg_name_32): (String, String) = if rhs_conflicts {
            self.operand_to_rax(rhs);
            self.operand_to_callee_reg(lhs, dest_phys);
            ("rax".to_string(), "eax".to_string())
        } else {
            self.operand_to_callee_reg(lhs, dest_phys);
            if let Some(rhs_phys) = rhs_phys {
                (phys_reg_name(rhs_phys).to_string(), phys_reg_name_32(rhs_phys).to_string())
            } else {
                self.operand_to_rax(rhs);
                ("rax".to_string(), "eax".to_string())
            }
        };

        if op == IrBinOp::Mul {
            if use_32bit {
                self.state.out.emit_instr_reg_reg("    imull", &rhs_reg_name_32, dest_name_32);
                self.emit_sext32_if_needed(dest_name_32, dest_name, is_unsigned);
            } else {
                self.state.out.emit_instr_reg_reg("    imulq", &rhs_reg_name, dest_name);
            }
        } else {
            let mnemonic = alu_mnemonic(op);
            if use_32bit && matches!(op, IrBinOp::Add | IrBinOp::Sub) {
                self.state.emit_fmt(format_args!("    {}l %{}, %{}", mnemonic, rhs_reg_name_32, dest_name_32));
                self.emit_sext32_if_needed(dest_name_32, dest_name, is_unsigned);
            } else {
                self.state.emit_fmt(format_args!("    {}q %{}, %{}", mnemonic, rhs_reg_name, dest_name));
            }
        }
        self.state.reg_cache.invalidate_acc();
    }

    /// Register-direct path for shift operations.
    pub(super) fn emit_shift_reg_direct(&mut self, op: IrBinOp, lhs: &Operand, rhs: &Operand,
                             dest_phys: PhysReg, use_32bit: bool, is_unsigned: bool) {
        let dest_name = phys_reg_name(dest_phys);
        let dest_name_32 = phys_reg_name_32(dest_phys);
        let (mnem32, mnem64) = shift_mnemonic(op);

        if let Some(imm) = Self::const_as_imm32(rhs) {
            self.operand_to_callee_reg(lhs, dest_phys);
            if use_32bit {
                let shift_amount = (imm as u32) & 31;
                self.state.emit_fmt(format_args!("    {} ${}, %{}", mnem32, shift_amount, dest_name_32));
                if !is_unsigned && matches!(op, IrBinOp::Shl | IrBinOp::AShr) {
                    self.state.out.emit_instr_reg_reg("    movslq", dest_name_32, dest_name);
                }
            } else {
                let shift_amount = (imm as u64) & 63;
                self.state.emit_fmt(format_args!("    {} ${}, %{}", mnem64, shift_amount, dest_name));
            }
        } else {
            let rhs_conflicts = self.operand_reg(rhs).is_some_and(|r| r.0 == dest_phys.0);
            if rhs_conflicts {
                self.operand_to_rcx(rhs);
                self.operand_to_callee_reg(lhs, dest_phys);
            } else {
                self.operand_to_callee_reg(lhs, dest_phys);
                self.operand_to_rcx(rhs);
            }
            if use_32bit {
                self.state.emit_fmt(format_args!("    {} %cl, %{}", mnem32, dest_name_32));
                if !is_unsigned && matches!(op, IrBinOp::Shl | IrBinOp::AShr) {
                    self.state.out.emit_instr_reg_reg("    movslq", dest_name_32, dest_name);
                }
            } else {
                self.state.emit_fmt(format_args!("    {} %cl, %{}", mnem64, dest_name));
            }
        }
        self.state.reg_cache.invalidate_acc();
    }

    /// Accumulator-based path: try immediate optimizations first.
    /// Returns true if handled.
    pub(super) fn try_emit_acc_immediate(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand,
                              use_32bit: bool, is_unsigned: bool) -> bool {
        // Immediate ALU ops
        if matches!(op, IrBinOp::Add | IrBinOp::Sub | IrBinOp::And | IrBinOp::Or | IrBinOp::Xor) {
            if let Some(imm) = Self::const_as_imm32(rhs) {
                self.operand_to_rax(lhs);
                let mnemonic = alu_mnemonic(op);
                if use_32bit && matches!(op, IrBinOp::Add | IrBinOp::Sub) {
                    self.state.emit_fmt(format_args!("    {}l ${}, %eax", mnemonic, imm));
                    if !is_unsigned { self.state.emit("    cltq"); }
                } else {
                    self.state.emit_fmt(format_args!("    {}q ${}, %rax", mnemonic, imm));
                }
                self.state.reg_cache.invalidate_acc();
                self.store_rax_to(dest);
                return true;
            }
        }

        // Immediate multiply
        if op == IrBinOp::Mul {
            if let Some(imm) = Self::const_as_imm32(rhs) {
                self.operand_to_rax(lhs);
                // LEA strength reduction: x*3/5/9 → lea (%rax, %rax, scale), %rax.
                // lea has 1-cycle latency vs 3 cycles for imul on modern x86.
                if let Some(scale) = Self::lea_scale_for_mul(imm) {
                    if use_32bit {
                        self.state.emit_fmt(format_args!("    leal (%eax, %eax, {}), %eax", scale));
                        if !is_unsigned { self.state.emit("    cltq"); }
                    } else {
                        self.state.emit_fmt(format_args!("    leaq (%rax, %rax, {}), %rax", scale));
                    }
                } else if use_32bit {
                    self.state.emit_fmt(format_args!("    imull ${}, %eax, %eax", imm));
                    if !is_unsigned { self.state.emit("    cltq"); }
                } else {
                    self.state.emit_fmt(format_args!("    imulq ${}, %rax, %rax", imm));
                }
                self.state.reg_cache.invalidate_acc();
                self.store_rax_to(dest);
                return true;
            }
        }

        // Immediate shift
        if matches!(op, IrBinOp::Shl | IrBinOp::AShr | IrBinOp::LShr) {
            if let Some(imm) = Self::const_as_imm32(rhs) {
                self.operand_to_rax(lhs);
                let (mnem32, mnem64) = shift_mnemonic(op);
                if use_32bit {
                    let shift_amount = (imm as u32) & 31;
                    self.state.emit_fmt(format_args!("    {} ${}, %eax", mnem32, shift_amount));
                    if !is_unsigned && matches!(op, IrBinOp::Shl | IrBinOp::AShr) {
                        self.state.emit("    cltq");
                    }
                } else {
                    let shift_amount = (imm as u64) & 63;
                    self.state.emit_fmt(format_args!("    {} ${}, %rax", mnem64, shift_amount));
                }
                self.state.reg_cache.invalidate_acc();
                self.store_rax_to(dest);
                return true;
            }
        }

        false
    }

    /// Helper to load va_list pointer into %rcx for va_arg operations.
    pub(super) fn load_va_list_ptr_to_rcx(&mut self, va_list_ptr: &Value) {
        if let Some(&reg) = self.reg_assignments.get(&va_list_ptr.0) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rcx");
        } else if let Some(slot) = self.state.get_slot(va_list_ptr.0) {
            if self.state.is_alloca(va_list_ptr.0) {
                self.state.out.emit_instr_rbp_reg("    leaq", slot.0, "rcx");
            } else {
                self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rcx");
            }
        }
    }
}

pub(super) const X86_ARG_REGS: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];


impl ArchCodegen for X86Codegen {
    fn state(&mut self) -> &mut CodegenState { &mut self.state }
    fn state_ref(&self) -> &CodegenState { &self.state }
    fn ptr_directive(&self) -> PtrDirective { PtrDirective::Quad }

    fn get_phys_reg_for_value(&self, val_id: u32) -> Option<PhysReg> {
        self.reg_assignments.get(&val_id).copied()
    }

    fn emit_reg_to_reg_move(&mut self, src: PhysReg, dest: PhysReg) {
        let s_name = phys_reg_name(src);
        let d_name = phys_reg_name(dest);
        self.state.out.emit_instr_reg_reg("    movq", s_name, d_name);
    }

    fn emit_acc_to_phys_reg(&mut self, dest: PhysReg) {
        let d_name = phys_reg_name(dest);
        self.state.out.emit_instr_reg_reg("    movq", "rax", d_name);
    }

    fn jump_mnemonic(&self) -> &'static str { "jmp" }
    fn trap_instruction(&self) -> &'static str { "ud2" }

    fn emit_branch_nonzero(&mut self, label: &str) {
        self.state.emit("    testq %rax, %rax");
        self.state.out.emit_jcc_label("    jne", label);
    }

    fn emit_jump_indirect(&mut self) {
        if self.state.indirect_branch_thunk {
            self.state.emit("    jmp __x86_indirect_thunk_rax");
        } else {
            self.state.emit("    jmpq *%rax");
        }
    }

    fn emit_switch_case_branch(&mut self, case_val: i64, label: &str, ty: IrType) {
        let use_32bit = matches!(ty, IrType::I32 | IrType::U32 | IrType::I16 | IrType::U16 | IrType::I8 | IrType::U8);
        if case_val == 0 {
            if use_32bit {
                self.state.emit("    testl %eax, %eax");
            } else {
                self.state.emit("    testq %rax, %rax");
            }
        } else if use_32bit {
            // Use 32-bit comparison to avoid sign-extension mismatch for int-sized values
            self.state.out.emit_instr_imm_reg("    cmpl", case_val as i32 as i64, "eax");
        } else if case_val >= i32::MIN as i64 && case_val <= i32::MAX as i64 {
            self.state.out.emit_instr_imm_reg("    cmpq", case_val, "rax");
        } else {
            self.state.out.emit_instr_imm_reg("    movabsq", case_val, "rcx");
            self.state.emit("    cmpq %rcx, %rax");
        }
        self.state.out.emit_jcc_label("    je", label);
    }

    fn emit_switch_jump_table(&mut self, val: &Operand, cases: &[(i64, BlockId)], default: &BlockId, ty: IrType) {
        use crate::backend::traits::build_jump_table;
        let (table, min_val, range) = build_jump_table(cases, default);
        let table_label = self.state.fresh_label("jt");
        let default_label = default.as_label();

        self.operand_to_rax(val);

        // For 32-bit switch types, sign-extend to 64-bit for the jump table indexing
        let use_32bit = matches!(ty, IrType::I32 | IrType::U32 | IrType::I16 | IrType::U16 | IrType::I8 | IrType::U8);
        if use_32bit {
            if ty.is_unsigned() {
                // Zero-extend: mov %eax, %eax clears upper 32 bits
                self.state.emit("    movl %eax, %eax");
            } else {
                // Sign-extend 32-bit to 64-bit
                self.state.emit("    cltq");
            }
        }

        if min_val != 0 {
            if min_val >= i32::MIN as i64 && min_val <= i32::MAX as i64 {
                self.state.out.emit_instr_imm_reg("    subq", min_val, "rax");
            } else {
                self.state.out.emit_instr_imm_reg("    movabsq", min_val, "rcx");
                self.state.emit("    subq %rcx, %rax");
            }
        }
        if (range as i64) >= i32::MIN as i64 && (range as i64) <= i32::MAX as i64 {
            self.state.out.emit_instr_imm_reg("    cmpq", range as i64, "rax");
        } else {
            self.state.out.emit_instr_imm_reg("    movabsq", range as i64, "rcx");
            self.state.emit("    cmpq %rcx, %rax");
        }
        self.state.out.emit_jcc_label("    jae", &default_label);

        self.state.out.emit_instr_sym_base_reg("    leaq", &table_label, "rip", "rcx");
        self.state.emit("    movslq (%rcx,%rax,4), %rdx");
        self.state.emit("    addq %rcx, %rdx");
        self.state.emit("    jmp *%rdx");

        self.state.emit(".section .rodata");
        self.state.emit(".align 4");
        self.state.out.emit_named_label(&table_label);
        for target in &table {
            let target_label = target.as_label();
            self.state.emit_fmt(format_args!("    .long {} - {}", target_label, table_label));
        }
        let sect = self.state.current_text_section.clone();
        self.state.emit_fmt(format_args!(".section {},\"ax\",@progbits", sect));

        self.state.reg_cache.invalidate_all();
    }

    // ---- Standard trait methods (kept inline) ----
    fn emit_load_operand(&mut self, op: &Operand) { self.operand_to_rax(op) }
    fn emit_store_result(&mut self, dest: &Value) { self.store_rax_to(dest) }
    fn supports_global_addr_fold(&self) -> bool { true }
    fn emit_call_store_f128_result(&mut self, _dest: &Value) { unreachable!("x86 uses custom emit_call_store_result for F128") }

    fn emit_copy_value(&mut self, dest: &Value, src: &Operand) {
        if let Operand::Value(v) = src {
            if self.state.f128_direct_slots.contains(&v.0) {
                if let (Some(src_slot), Some(dest_slot)) = (self.state.get_slot(v.0), self.state.get_slot(dest.0)) {
                    self.state.out.emit_instr_rbp("    fldt", src_slot.0);
                    self.state.out.emit_instr_rbp("    fstpt", dest_slot.0);
                    self.state.out.emit_instr_rbp("    fldt", dest_slot.0);
                    self.state.emit("    subq $8, %rsp");
                    self.state.emit("    fstpl (%rsp)");
                    self.state.emit("    popq %rax");
                    self.state.reg_cache.set_acc(dest.0, false);
                    self.state.f128_direct_slots.insert(dest.0);
                    return;
                }
            }
        }

        let dest_phys = self.dest_reg(dest);
        let src_phys = self.operand_reg(src);

        match (dest_phys, src_phys) {
            (Some(d), Some(s)) => {
                if d.0 != s.0 {
                    let d_name = phys_reg_name(d);
                    let s_name = phys_reg_name(s);
                    self.state.out.emit_instr_reg_reg("    movq", s_name, d_name);
                }
                self.state.reg_cache.invalidate_acc();
            }
            (Some(d), None) => {
                self.operand_to_rax(src);
                let d_name = phys_reg_name(d);
                self.state.out.emit_instr_reg_reg("    movq", "rax", d_name);
                self.state.reg_cache.invalidate_acc();
            }
            _ => {
                self.operand_to_rax(src);
                self.store_rax_to(dest);
            }
        }
    }

    // ---- Inline asm (kept inline - has extra logic) ----
    fn emit_inline_asm(&mut self, template: &str, outputs: &[(String, Value, Option<String>)], inputs: &[(String, Operand, Option<String>)], clobbers: &[String], operand_types: &[IrType], goto_labels: &[(String, BlockId)], input_symbols: &[Option<String>]) {
        emit_inline_asm_common(self, template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols);
        self.emit_callee_saved_clobber_annotations(clobbers);
        self.state.reg_cache.invalidate_all();
    }

    fn emit_inline_asm_with_segs(&mut self, template: &str, outputs: &[(String, Value, Option<String>)], inputs: &[(String, Operand, Option<String>)], clobbers: &[String], operand_types: &[IrType], goto_labels: &[(String, BlockId)], input_symbols: &[Option<String>], seg_overrides: &[AddressSpace]) {
        crate::backend::inline_asm::emit_inline_asm_common_impl(self, template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols, seg_overrides);
        self.emit_callee_saved_clobber_annotations(clobbers);
        self.state.reg_cache.invalidate_all();
    }

    // ---- Intrinsics (kept inline - has extra logic) ----
    fn emit_intrinsic(&mut self, dest: &Option<Value>, op: &IntrinsicOp, dest_ptr: &Option<Value>, args: &[Operand]) {
        self.emit_intrinsic_impl(dest, op, dest_ptr, args);
        self.state.reg_cache.invalidate_all();
    }

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
        fn emit_save_acc(&mut self) => emit_save_acc_impl;
        fn emit_load_ptr_from_slot(&mut self, slot: StackSlot, val_id: u32) => emit_load_ptr_from_slot_impl;
        fn emit_typed_store_indirect(&mut self, instr: &'static str, ty: IrType) => emit_typed_store_indirect_impl;
        fn emit_typed_load_indirect(&mut self, instr: &'static str) => emit_typed_load_indirect_impl;
        fn emit_add_offset_to_addr_reg(&mut self, offset: i64) => emit_add_offset_to_addr_reg_impl;
        fn emit_slot_addr_to_secondary(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) => emit_slot_addr_to_secondary_impl;
        fn emit_add_secondary_to_acc(&mut self) => emit_add_secondary_to_acc_impl;
        fn emit_gep_direct_const(&mut self, slot: StackSlot, offset: i64) => emit_gep_direct_const_impl;
        fn emit_gep_indirect_const(&mut self, slot: StackSlot, offset: i64, val_id: u32) => emit_gep_indirect_const_impl;
        fn emit_gep_add_const_to_acc(&mut self, offset: i64) => emit_gep_add_const_to_acc_impl;
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
        fn emit_acc_to_secondary(&mut self) => emit_acc_to_secondary_impl;
        fn emit_memcpy_store_dest_from_acc(&mut self) => emit_memcpy_store_dest_from_acc_impl;
        fn emit_memcpy_store_src_from_acc(&mut self) => emit_memcpy_store_src_from_acc_impl;
        fn emit_memcpy_impl(&mut self, size: usize) => emit_memcpy_impl_impl;
        fn emit_seg_load(&mut self, dest: &Value, ptr: &Value, ty: IrType, seg: AddressSpace) => emit_seg_load_impl;
        fn emit_seg_load_symbol(&mut self, dest: &Value, sym: &str, ty: IrType, seg: AddressSpace) => emit_seg_load_symbol_impl;
        fn emit_seg_store(&mut self, val: &Operand, ptr: &Value, ty: IrType, seg: AddressSpace) => emit_seg_store_impl;
        fn emit_seg_store_symbol(&mut self, val: &Operand, sym: &str, ty: IrType, seg: AddressSpace) => emit_seg_store_symbol_impl;
        // alu
        fn emit_float_neg(&mut self, ty: IrType) => emit_float_neg_impl;
        fn emit_int_neg(&mut self, ty: IrType) => emit_int_neg_impl;
        fn emit_int_not(&mut self, ty: IrType) => emit_int_not_impl;
        fn emit_int_clz(&mut self, ty: IrType) => emit_int_clz_impl;
        fn emit_int_ctz(&mut self, ty: IrType) => emit_int_ctz_impl;
        fn emit_int_bswap(&mut self, ty: IrType) => emit_int_bswap_impl;
        fn emit_int_popcount(&mut self, ty: IrType) => emit_int_popcount_impl;
        fn emit_int_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_int_binop_impl;
        fn emit_copy_i128(&mut self, dest: &Value, src: &Operand) => emit_copy_i128_impl;
        // comparison
        fn emit_f128_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand) => emit_f128_cmp_impl;
        fn emit_float_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_float_cmp_impl;
        fn emit_int_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_int_cmp_impl;
        fn emit_fused_cmp_branch(&mut self, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType, true_label: &str, false_label: &str) => emit_fused_cmp_branch_impl;
        fn emit_fused_cmp_branch_blocks(&mut self, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType, true_block: BlockId, false_block: BlockId) => emit_fused_cmp_branch_blocks_impl;
        fn emit_cond_branch_blocks(&mut self, cond: &Operand, true_block: BlockId, false_block: BlockId) => emit_cond_branch_blocks_impl;
        fn emit_select(&mut self, dest: &Value, cond: &Operand, true_val: &Operand, false_val: &Operand, ty: IrType) => emit_select_impl;
        // calls
        fn call_abi_config(&self) -> CallAbiConfig => call_abi_config_impl;
        fn emit_call_compute_stack_space(&self, arg_classes: &[CallArgClass], arg_types: &[IrType]) -> usize => emit_call_compute_stack_space_impl;
        fn emit_call_stack_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass], arg_types: &[IrType], stack_arg_space: usize, fptr_spill: usize, f128_temp_space: usize) -> i64 => emit_call_stack_args_impl;
        fn emit_call_reg_args(&mut self, args: &[Operand], arg_classes: &[CallArgClass], arg_types: &[IrType], total_sp_adjust: i64, f128_temp_space: usize, stack_arg_space: usize, struct_arg_riscv_float_classes: &[Option<crate::common::types::RiscvFloatClass>]) => emit_call_reg_args_impl;
        fn emit_call_instruction(&mut self, direct_name: Option<&str>, func_ptr: Option<&Operand>, indirect: bool, stack_arg_space: usize) => emit_call_instruction_impl;
        fn emit_call_cleanup(&mut self, stack_arg_space: usize, f128_temp_space: usize, indirect: bool) => emit_call_cleanup_impl;
        fn set_call_ret_eightbyte_classes(&mut self, classes: &[crate::common::types::EightbyteClass]) => set_call_ret_eightbyte_classes_impl;
        fn emit_call_store_result(&mut self, dest: &Value, return_type: IrType) => emit_call_store_result_impl;
        fn emit_call_store_i128_result(&mut self, dest: &Value) => emit_call_store_i128_result_impl;
        fn emit_call_move_f32_to_acc(&mut self) => emit_call_move_f32_to_acc_impl;
        fn emit_call_move_f64_to_acc(&mut self) => emit_call_move_f64_to_acc_impl;
        // globals
        fn emit_global_addr(&mut self, dest: &Value, name: &str) => emit_global_addr_impl;
        fn emit_tls_global_addr(&mut self, dest: &Value, name: &str) => emit_tls_global_addr_impl;
        fn emit_global_addr_absolute(&mut self, dest: &Value, name: &str) => emit_global_addr_absolute_impl;
        fn emit_global_load_rip_rel(&mut self, dest: &Value, sym: &str, ty: IrType) => emit_global_load_rip_rel_impl;
        fn emit_global_store_rip_rel(&mut self, val: &Operand, sym: &str, ty: IrType) => emit_global_store_rip_rel_impl;
        fn emit_label_addr(&mut self, dest: &Value, label: &str) => emit_label_addr_impl;
        // cast
        fn emit_cast_instrs(&mut self, from_ty: IrType, to_ty: IrType) => emit_cast_instrs_impl;
        fn emit_cast(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) => emit_cast_impl;
        // variadic
        fn emit_va_arg(&mut self, dest: &Value, va_list_ptr: &Value, result_ty: IrType) => emit_va_arg_impl;
        fn emit_va_arg_struct(&mut self, dest_ptr: &Value, va_list_ptr: &Value, size: usize) => emit_va_arg_struct_impl;
        fn emit_va_arg_struct_ex(&mut self, dest_ptr: &Value, va_list_ptr: &Value, size: usize, eightbyte_classes: &[crate::common::types::EightbyteClass]) => emit_va_arg_struct_ex_impl;
        fn emit_va_start(&mut self, va_list_ptr: &Value) => emit_va_start_impl;
        fn emit_va_copy(&mut self, dest_ptr: &Value, src_ptr: &Value) => emit_va_copy_impl;
        // returns
        fn emit_return(&mut self, val: Option<&Operand>, frame_size: i64) => emit_return_impl;
        fn current_return_type(&self) -> IrType => current_return_type_impl;
        fn emit_return_f32_to_reg(&mut self) => emit_return_f32_to_reg_impl;
        fn emit_return_f64_to_reg(&mut self) => emit_return_f64_to_reg_impl;
        fn emit_return_i128_to_regs(&mut self) => emit_return_i128_to_regs_impl;
        fn emit_get_return_f64_second(&mut self, dest: &Value) => emit_get_return_f64_second_impl;
        fn emit_set_return_f64_second(&mut self, src: &Operand) => emit_set_return_f64_second_impl;
        fn emit_get_return_f32_second(&mut self, dest: &Value) => emit_get_return_f32_second_impl;
        fn emit_set_return_f32_second(&mut self, src: &Operand) => emit_set_return_f32_second_impl;
        fn emit_return_f128_to_reg(&mut self) => emit_return_f128_to_reg_impl;
        fn emit_return_int_to_reg(&mut self) => emit_return_int_to_reg_impl;
        fn emit_get_return_f128_second(&mut self, dest: &Value) => emit_get_return_f128_second_impl;
        fn emit_set_return_f128_second(&mut self, src: &Operand) => emit_set_return_f128_second_impl;
        // atomics
        fn emit_atomic_rmw(&mut self, dest: &Value, op: AtomicRmwOp, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_rmw_impl;
        fn emit_atomic_cmpxchg(&mut self, dest: &Value, ptr: &Operand, expected: &Operand, desired: &Operand, ty: IrType, success_ordering: AtomicOrdering, failure_ordering: AtomicOrdering, returns_bool: bool) => emit_atomic_cmpxchg_impl;
        fn emit_atomic_load(&mut self, dest: &Value, ptr: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_load_impl;
        fn emit_atomic_store(&mut self, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_store_impl;
        fn emit_fence(&mut self, ordering: AtomicOrdering) => emit_fence_impl;
        // float ops
        fn emit_float_binop_mnemonic(&self, op: FloatOp) -> &'static str => emit_float_binop_mnemonic_impl;
        fn emit_unaryop(&mut self, dest: &Value, op: IrUnaryOp, src: &Operand, ty: IrType) => emit_unaryop_impl;
        fn emit_float_binop(&mut self, dest: &Value, op: FloatOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_float_binop_impl;
        fn emit_float_binop_impl(&mut self, mnemonic: &str, ty: IrType) => emit_float_binop_impl_impl;
        // i128 ops
        fn emit_sign_extend_acc_high(&mut self) => emit_sign_extend_acc_high_impl;
        fn emit_zero_acc_high(&mut self) => emit_zero_acc_high_impl;
        fn emit_load_acc_pair(&mut self, op: &Operand) => emit_load_acc_pair_impl;
        fn emit_store_acc_pair(&mut self, dest: &Value) => emit_store_acc_pair_impl;
        fn emit_store_pair_to_slot(&mut self, slot: StackSlot) => emit_store_pair_to_slot_impl;
        fn emit_load_pair_from_slot(&mut self, slot: StackSlot) => emit_load_pair_from_slot_impl;
        fn emit_save_acc_pair(&mut self) => emit_save_acc_pair_impl;
        fn emit_store_pair_indirect(&mut self) => emit_store_pair_indirect_impl;
        fn emit_load_pair_indirect(&mut self) => emit_load_pair_indirect_impl;
        fn emit_i128_neg(&mut self) => emit_i128_neg_impl;
        fn emit_i128_not(&mut self) => emit_i128_not_impl;
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
impl Default for X86Codegen {
    fn default() -> Self {
        Self::new()
    }
}

