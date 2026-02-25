//! X86Codegen: prologue, epilogue, parameter storage.

use crate::ir::reexports::{IrFunction, Instruction, Value};
use crate::common::types::IrType;
use crate::backend::call_abi::{ParamClass, classify_params};
use crate::backend::generation::{calculate_stack_space_common, find_param_alloca};
use crate::backend::regalloc::PhysReg;
use super::emit::{X86Codegen, X86_CALLEE_SAVED, X86_CALLER_SAVED, phys_reg_name,
                     collect_inline_asm_callee_saved_x86, X86_ARG_REGS};

impl X86Codegen {
    pub(super) fn calculate_stack_space_impl(&mut self, func: &IrFunction) -> i64 {
        // Track variadic function info
        self.is_variadic = func.is_variadic;
        // Count named params using the shared ABI classification, so this
        // stays in sync with classify_call_args (caller side) automatically.
        {
            let config = self.call_abi_config_impl();
            let classification = crate::backend::call_abi::classify_params_full(func, &config);
            let mut named_gp = 0usize;
            let mut named_fp = 0usize;
            for class in &classification.classes {
                named_gp += class.gp_reg_count();
                if matches!(class, crate::backend::call_abi::ParamClass::FloatReg { .. }) {
                    named_fp += 1;
                }
            }
            self.num_named_int_params = named_gp;
            self.num_named_fp_params = named_fp;
            self.num_named_stack_bytes =
                crate::backend::call_abi::named_params_stack_bytes(&classification.classes);
        }

        // Run register allocator BEFORE stack space computation so we can
        // skip allocating stack slots for values assigned to registers.
        let mut asm_clobbered_regs: Vec<PhysReg> = Vec::new();
        collect_inline_asm_callee_saved_x86(func, &mut asm_clobbered_regs);
        let available_regs = crate::backend::generation::filter_available_regs(&X86_CALLEE_SAVED, &asm_clobbered_regs);

        let mut caller_saved_regs = X86_CALLER_SAVED.to_vec();
        let mut has_indirect_call = false;
        let mut has_i128_ops = false;
        let mut has_atomic_rmw = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::CallIndirect { .. } => { has_indirect_call = true; }
                    Instruction::BinOp { ty, .. }
                    | Instruction::UnaryOp { ty, .. } => {
                        if matches!(ty, IrType::I128 | IrType::U128) {
                            has_i128_ops = true;
                        }
                    }
                    Instruction::Cast { from_ty, to_ty, .. } => {
                        if matches!(from_ty, IrType::I128 | IrType::U128)
                            || matches!(to_ty, IrType::I128 | IrType::U128) {
                            has_i128_ops = true;
                        }
                    }
                    Instruction::Cmp { ty, .. }
                    | Instruction::Store { ty, .. } => {
                        if matches!(ty, IrType::I128 | IrType::U128) {
                            has_i128_ops = true;
                        }
                    }
                    Instruction::AtomicRmw { .. } => { has_atomic_rmw = true; }
                    _ => {}
                }
            }
        }
        if has_indirect_call {
            caller_saved_regs.retain(|r| r.0 != 11); // r10 = PhysReg(11)
        }
        if has_i128_ops {
            caller_saved_regs.retain(|r| r.0 != 12 && r.0 != 13 && r.0 != 14 && r.0 != 15); // r8, r9, rdi, rsi
        }
        if has_atomic_rmw {
            caller_saved_regs.retain(|r| r.0 != 12); // r8
        }

        let (reg_assigned, cached_liveness) = crate::backend::generation::run_regalloc_and_merge_clobbers(
            func, available_regs, caller_saved_regs, &asm_clobbered_regs,
            &mut self.reg_assignments, &mut self.used_callee_saved,
            false,
        );

        let mut space = calculate_stack_space_common(&mut self.state, func, 0, |space, alloc_size, align| {
            // Allow 4-byte slots for small values (I8-I32, U8-U32, F32).
            // 8-byte and larger values still get 8-byte alignment.
            let min_align = if alloc_size <= 4 { 4 } else { 8 };
            let effective_align = if align > 0 { align.max(min_align) } else { min_align };
            let alloc = (alloc_size + min_align - 1) & !(min_align - 1);
            let new_space = ((space + alloc + effective_align - 1) / effective_align) * effective_align;
            (-new_space, new_space)
        }, &reg_assigned, &X86_CALLEE_SAVED, cached_liveness, false);

        if func.is_variadic {
            if self.no_sse {
                space += 48;
            } else {
                space += 176;
            }
            self.reg_save_area_offset = -space;
        }

        let callee_save_space = (self.used_callee_saved.len() as i64) * 8;
        space + callee_save_space
    }

    pub(super) fn aligned_frame_size_impl(&self, raw_space: i64) -> i64 {
        if raw_space > 0 { (raw_space + 15) & !15 } else { 0 }
    }

    pub(super) fn emit_prologue_impl(&mut self, func: &IrFunction, frame_size: i64) {
        self.current_return_type = func.return_type;
        self.func_ret_classes = func.ret_eightbyte_classes.clone();
        if self.state.cf_protection_branch {
            self.state.emit("    endbr64");
        }
        self.state.emit("    pushq %rbp");
        if self.state.emit_cfi {
            self.state.emit("    .cfi_def_cfa_offset 16");
            self.state.emit("    .cfi_offset %rbp, -16");
        }
        self.state.emit("    movq %rsp, %rbp");
        if self.state.emit_cfi {
            self.state.emit("    .cfi_def_cfa_register %rbp");
        }
        if frame_size > 0 {
            const PAGE_SIZE: i64 = 4096;
            if frame_size > PAGE_SIZE {
                let probe_label = self.state.fresh_label("stack_probe");
                self.state.out.emit_instr_imm_reg("    movq", frame_size, "r11");
                self.state.out.emit_named_label(&probe_label);
                self.state.out.emit_instr_imm_reg("    subq", PAGE_SIZE, "rsp");
                self.state.emit("    orl $0, (%rsp)");
                self.state.out.emit_instr_imm_reg("    subq", PAGE_SIZE, "r11");
                self.state.out.emit_instr_imm_reg("    cmpq", PAGE_SIZE, "r11");
                self.state.out.emit_jcc_label("    ja", &probe_label);
                self.state.emit("    subq %r11, %rsp");
                self.state.emit("    orl $0, (%rsp)");
            } else {
                self.state.out.emit_instr_imm_reg("    subq", frame_size, "rsp");
            }
        }

        let used_regs = self.used_callee_saved.clone();
        for (i, &reg) in used_regs.iter().enumerate() {
            let offset = -frame_size + (i as i64 * 8);
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_rbp("    movq", reg_name, offset);
        }

        if func.is_variadic {
            let base = self.reg_save_area_offset;
            self.state.out.emit_instr_reg_rbp("    movq", "rdi", base);
            self.state.out.emit_instr_reg_rbp("    movq", "rsi", base + 8);
            self.state.out.emit_instr_reg_rbp("    movq", "rdx", base + 16);
            self.state.out.emit_instr_reg_rbp("    movq", "rcx", base + 24);
            self.state.out.emit_instr_reg_rbp("    movq", "r8", base + 32);
            self.state.out.emit_instr_reg_rbp("    movq", "r9", base + 40);
            if !self.no_sse {
                for i in 0..8i64 {
                    self.state.emit_fmt(format_args!("    movdqu %xmm{}, {}(%rbp)", i, base + 48 + i * 16));
                }
            }
        }
    }

    pub(super) fn emit_epilogue_impl(&mut self, frame_size: i64) {
        let used_regs = self.used_callee_saved.clone();
        for (i, &reg) in used_regs.iter().enumerate() {
            let offset = -frame_size + (i as i64 * 8);
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_rbp_reg("    movq", offset, reg_name);
        }
        self.state.emit("    movq %rbp, %rsp");
        self.state.emit("    popq %rbp");
    }

    pub(super) fn emit_store_params_impl(&mut self, func: &IrFunction) {
        let xmm_regs = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"];
        let config = self.call_abi_config_impl();
        let param_classes = classify_params(func, &config);
        self.state.param_classes = param_classes.clone();
        self.state.num_params = func.params.len();
        self.state.func_is_variadic = func.is_variadic;

        self.state.param_alloca_slots = (0..func.params.len()).map(|i| {
            find_param_alloca(func, i).and_then(|(dest, ty)| {
                self.state.get_slot(dest.0).map(|slot| (slot, ty))
            })
        }).collect();

        // Build a map of param_idx -> ParamRef dest Value for fast lookup.
        // This is used to optimize parameter storage: when the ParamRef dest
        // is register-allocated, we can store the ABI arg register directly
        // to the callee-saved register, skipping the alloca slot entirely.
        let mut paramref_dests: Vec<Option<Value>> = vec![None; func.params.len()];
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::ParamRef { dest, param_idx, .. } = inst {
                    if *param_idx < paramref_dests.len() {
                        paramref_dests[*param_idx] = Some(*dest);
                    }
                }
            }
        }

        let stack_base: i64 = 16;

        // Build a map from physical register -> list of param indices that use it,
        // so we can detect when two params share the same callee-saved register.
        let mut reg_to_params: crate::common::fx_hash::FxHashMap<u8, Vec<usize>> = crate::common::fx_hash::FxHashMap::default();
        for (i, _) in func.params.iter().enumerate() {
            if let Some(paramref_dest) = paramref_dests[i] {
                if let Some(&phys_reg) = self.reg_assignments.get(&paramref_dest.0) {
                    reg_to_params.entry(phys_reg.0).or_default().push(i);
                }
            }
        }

        for (i, _param) in func.params.iter().enumerate() {
            let class = param_classes[i];

            // Pre-store optimization: when a param's alloca is dead (eliminated by
            // dead param alloca analysis) but the ParamRef dest is register-assigned,
            // store the ABI arg register directly to the assigned physical register
            // in the prologue. This is critical because:
            // 1. Dead alloca means no stack slot exists for this param
            // 2. The ABI register (rdi, rsi, etc.) is caller-saved and will be clobbered
            // 3. We must save the value NOW, before any other code runs
            // 4. emit_param_ref will see param_pre_stored and skip code generation
            if let Some(paramref_dest) = paramref_dests[i] {
                let has_slot = find_param_alloca(func, i)
                    .and_then(|(dest, _)| self.state.get_slot(dest.0))
                    .is_some();
                if !has_slot {
                    if let Some(&phys_reg) = self.reg_assignments.get(&paramref_dest.0) {
                        // Only pre-store to callee-saved registers (PhysReg 1-5).
                        // Caller-saved registers (rdi, rsi, r8-r11) cannot be used
                        // because they overlap with ABI argument registers that
                        // haven't been saved yet.
                        let is_callee_saved = phys_reg.0 >= 1 && phys_reg.0 <= 5;
                        if is_callee_saved {
                            // Safety check: if another param's dest is also assigned
                            // to this register, skip pre-store to avoid conflicts.
                            let shared = reg_to_params.get(&phys_reg.0)
                                .is_some_and(|users| users.len() > 1);
                            if !shared {
                                let dest_reg = phys_reg_name(phys_reg);
                                if let ParamClass::IntReg { reg_idx } = class {
                                    self.state.out.emit_instr_reg_reg(
                                        "    movq", X86_ARG_REGS[reg_idx], dest_reg);
                                    self.state.param_pre_stored.insert(i);
                                } // TODO: handle StackSlot/SSE params
                            }
                        }
                    }
                    continue;
                }
            }

            let (slot, ty) = if let Some((dest, ty)) = find_param_alloca(func, i) {
                if let Some(slot) = self.state.get_slot(dest.0) {
                    (slot, ty)
                } else {
                    continue;
                }
            } else {
                continue;
            };

            match class {
                ParamClass::IntReg { reg_idx } => {
                    // Always store the full 64-bit register to ensure the entire 8-byte
                    // slot is initialized. Using a typed store (e.g., movl for I32) would
                    // only write 4 bytes, leaving the upper bytes uninitialized. Later
                    // untyped loads via value_to_reg use movq (8 bytes), which would read
                    // uninitialized memory and trigger valgrind errors.
                    // The typed load in emit_param_ref_impl correctly extracts only the
                    // meaningful bytes (e.g., movslq for I32).
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[reg_idx], slot.0);
                }
                ParamClass::FloatReg { reg_idx } => {
                    if ty == IrType::F32 {
                        self.state.out.emit_instr_reg_reg("    movd", xmm_regs[reg_idx], "eax");
                        self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                    } else {
                        self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[reg_idx], slot.0);
                    }
                }
                ParamClass::I128RegPair { base_reg_idx } => {
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx], slot.0);
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx + 1], slot.0 + 8);
                }
                ParamClass::StructByValReg { base_reg_idx, size } => {
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx], slot.0);
                    if size > 8 {
                        self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx + 1], slot.0 + 8);
                    }
                }
                ParamClass::StructSseReg { lo_fp_idx, hi_fp_idx, .. } => {
                    self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[lo_fp_idx], slot.0);
                    if let Some(hi) = hi_fp_idx {
                        self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[hi], slot.0 + 8);
                    }
                }
                ParamClass::StructMixedIntSseReg { int_reg_idx, fp_reg_idx, .. } => {
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[int_reg_idx], slot.0);
                    self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[fp_reg_idx], slot.0 + 8);
                }
                ParamClass::StructMixedSseIntReg { fp_reg_idx, int_reg_idx, .. } => {
                    self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[fp_reg_idx], slot.0);
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[int_reg_idx], slot.0 + 8);
                }
                ParamClass::F128AlwaysStack { offset } => {
                    let src = stack_base + offset;
                    self.state.out.emit_instr_rbp("    fldt", src);
                    self.state.out.emit_instr_rbp("    fstpt", slot.0);
                }
                ParamClass::I128Stack { offset } => {
                    let src = stack_base + offset;
                    self.state.out.emit_instr_rbp_reg("    movq", src, "rax");
                    self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                    self.state.out.emit_instr_rbp_reg("    movq", src + 8, "rax");
                    self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0 + 8);
                }
                ParamClass::StackScalar { offset } => {
                    // Load from caller's stack frame and store full 8 bytes to ensure
                    // the entire slot is initialized (see IntReg comment above).
                    let src = stack_base + offset;
                    self.state.out.emit_instr_rbp_reg("    movq", src, "rax");
                    self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                }
                ParamClass::StructStack { offset, size } | ParamClass::LargeStructStack { offset, size } => {
                    let src = stack_base + offset;
                    let n_qwords = size.div_ceil(8);
                    for qi in 0..n_qwords {
                        let src_off = src + (qi as i64 * 8);
                        let dst_off = slot.0 + (qi as i64 * 8);
                        self.state.out.emit_instr_rbp_reg("    movq", src_off, "rax");
                        self.state.out.emit_instr_reg_rbp("    movq", "rax", dst_off);
                    }
                }
                ParamClass::F128FpReg { .. } | ParamClass::F128GpPair { .. } | ParamClass::F128Stack { .. } |
                ParamClass::LargeStructByRefReg { .. } | ParamClass::LargeStructByRefStack { .. } |
                ParamClass::StructSplitRegStack { .. } |
                ParamClass::ZeroSizeSkip => {}
            }
        }
    }

    pub(super) fn emit_param_ref_impl(&mut self, dest: &Value, param_idx: usize, ty: IrType) {
        if param_idx >= self.state.param_classes.len() {
            return;
        }

        // If this param was pre-stored directly to its register-allocated
        // destination during emit_store_params, the value is already in place.
        // No code needs to be emitted — the register already holds the value.
        if self.state.param_pre_stored.contains(&param_idx) {
            return;
        }

        if param_idx < self.state.param_alloca_slots.len() {
            if let Some((slot, alloca_ty)) = self.state.param_alloca_slots[param_idx] {
                let load_instr = Self::mov_load_for_type(alloca_ty);
                let reg = Self::load_dest_reg(alloca_ty);
                self.state.emit_fmt(format_args!("    {} {}(%rbp), {}", load_instr, slot.0, reg));
                self.store_rax_to(dest);
                return;
            }
        }

        let xmm_regs = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"];
        let class = self.state.param_classes[param_idx];
        let stack_base: i64 = 16;

        match class {
            ParamClass::IntReg { reg_idx } => {
                let src_reg = Self::reg_for_type(X86_ARG_REGS[reg_idx], ty);
                let load_instr = Self::mov_load_for_type(ty);
                let dest_reg = Self::load_dest_reg(ty);
                self.state.emit_fmt(format_args!("    {} %{}, {}", load_instr, src_reg, dest_reg));
                self.store_rax_to(dest);
            }
            ParamClass::FloatReg { reg_idx } => {
                if ty == IrType::F32 {
                    self.state.out.emit_instr_reg_reg("    movd", xmm_regs[reg_idx], "eax");
                    self.store_rax_to(dest);
                } else {
                    self.state.out.emit_instr_reg_reg("    movq", xmm_regs[reg_idx], "rax");
                    self.store_rax_to(dest);
                }
            }
            ParamClass::StackScalar { offset } => {
                let src = stack_base + offset;
                let load_instr = Self::mov_load_for_type(ty);
                let reg = Self::load_dest_reg(ty);
                self.state.emit_fmt(format_args!("    {} {}(%rbp), {}", load_instr, src, reg));
                self.store_rax_to(dest);
            }
            _ => {}
        }
    }

    pub(super) fn emit_epilogue_and_ret_impl(&mut self, frame_size: i64) {
        self.emit_epilogue_impl(frame_size);
        if self.state.function_return_thunk {
            self.state.emit("    jmp __x86_return_thunk");
        } else {
            self.state.emit("    ret");
        }
    }

    pub(super) fn store_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        Self::mov_store_for_type(ty)
    }

    pub(super) fn load_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        Self::mov_load_for_type(ty)
    }
}
