//! X86Codegen: memory operations (load, store, memcpy, GEP, stack).

use crate::ir::reexports::{IrConst, Operand, Value};
use crate::common::types::{AddressSpace, IrType};
use crate::backend::state::{StackSlot, SlotAddr};
use super::emit::{X86Codegen, phys_reg_name};

impl X86Codegen {
    // ---- Store/Load overrides ----

    pub(super) fn emit_store_impl(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        if ty == IrType::F128 {
            if let Operand::Const(IrConst::LongDouble(_, f128_bytes)) = val {
                let x87 = crate::common::long_double::f128_bytes_to_x87_bytes(f128_bytes);
                let lo = u64::from_le_bytes(x87[0..8].try_into().unwrap());
                let hi_bytes: [u8; 8] = [x87[8], x87[9], 0, 0, 0, 0, 0, 0];
                let hi = u64::from_le_bytes(hi_bytes);
                if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                    self.emit_f128_store_raw_bytes(&addr, ptr.0, 0, lo, hi);
                }
                return;
            }
            if let Operand::Value(v) = val {
                if self.state.f128_direct_slots.contains(&v.0) {
                    if let Some(src_slot) = self.state.get_slot(v.0) {
                        if let Some(dest_addr) = self.state.resolve_slot_addr(ptr.0) {
                            self.state.out.emit_instr_rbp("    fldt", src_slot.0);
                            self.emit_f128_fstpt(&dest_addr, ptr.0, 0);
                            return;
                        }
                    }
                }
            }
            self.operand_to_rax(val);
            if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                self.emit_f128_store_f64_via_x87(&addr, ptr.0, 0);
            }
            return;
        }
        crate::backend::traits::emit_store_default(self, val, ptr, ty);
    }

    pub(super) fn emit_load_impl(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if ty == IrType::F128 {
            if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                self.emit_f128_fldt(&addr, ptr.0, 0);
                self.emit_f128_load_finish(dest);
                self.state.track_f128_load(dest.0, ptr.0, 0);
            }
            return;
        }
        crate::backend::traits::emit_load_default(self, dest, ptr, ty);
    }

    pub(super) fn emit_store_with_const_offset_impl(&mut self, val: &Operand, base: &Value, offset: i64, ty: IrType) {
        if ty == IrType::F128 {
            if let Operand::Const(IrConst::LongDouble(_, f128_bytes)) = val {
                let x87 = crate::common::long_double::f128_bytes_to_x87_bytes(f128_bytes);
                let lo = u64::from_le_bytes(x87[0..8].try_into().unwrap());
                let hi_bytes: [u8; 8] = [x87[8], x87[9], 0, 0, 0, 0, 0, 0];
                let hi = u64::from_le_bytes(hi_bytes);
                if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                    self.emit_f128_store_raw_bytes(&addr, base.0, offset, lo, hi);
                }
                return;
            }
            if let Operand::Value(v) = val {
                if self.state.f128_direct_slots.contains(&v.0) {
                    if let Some(src_slot) = self.state.get_slot(v.0) {
                        if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                            self.state.out.emit_instr_rbp("    fldt", src_slot.0);
                            self.emit_f128_fstpt(&addr, base.0, offset);
                            return;
                        }
                    }
                }
            }
            self.operand_to_rax(val);
            if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                self.emit_f128_store_f64_via_x87(&addr, base.0, offset);
            }
            return;
        }
        // Non-F128: use the default GEP fold logic.
        self.operand_to_rax(val);
        let addr = self.state.resolve_slot_addr(base.0);
        if let Some(addr) = addr {
            let store_instr = Self::mov_store_for_type(ty);
            match addr {
                SlotAddr::OverAligned(slot, id) => {
                    self.emit_save_acc_impl();
                    self.emit_alloca_aligned_addr_impl(slot, id);
                    self.emit_add_offset_to_addr_reg_impl(offset);
                    self.emit_typed_store_indirect_impl(store_instr, ty);
                }
                SlotAddr::Direct(slot) => {
                    let folded_slot = StackSlot(slot.0 + offset);
                    self.emit_typed_store_to_slot_impl(store_instr, ty, folded_slot);
                }
                SlotAddr::Indirect(slot) => {
                    if let Some(&reg) = self.reg_assignments.get(&base.0) {
                        let reg_name = phys_reg_name(reg);
                        let store_reg = Self::reg_for_type("rax", ty);
                        if offset != 0 {
                            self.state.emit_fmt(format_args!("    {} %{}, {}(%{})", store_instr, store_reg, offset, reg_name));
                        } else {
                            self.state.emit_fmt(format_args!("    {} %{}, (%{})", store_instr, store_reg, reg_name));
                        }
                    } else {
                        self.emit_save_acc_impl();
                        self.emit_load_ptr_from_slot_impl(slot, base.0);
                        if offset != 0 {
                            self.emit_add_offset_to_addr_reg_impl(offset);
                        }
                        self.emit_typed_store_indirect_impl(store_instr, ty);
                    }
                }
            }
        }
    }

    pub(super) fn emit_load_with_const_offset_impl(&mut self, dest: &Value, base: &Value, offset: i64, ty: IrType) {
        if ty == IrType::F128 {
            if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                self.emit_f128_fldt(&addr, base.0, offset);
                self.emit_f128_load_finish(dest);
            }
            return;
        }
        let addr = self.state.resolve_slot_addr(base.0);
        if let Some(addr) = addr {
            let load_instr = Self::mov_load_for_type(ty);
            match addr {
                SlotAddr::OverAligned(slot, id) => {
                    self.emit_alloca_aligned_addr_impl(slot, id);
                    self.emit_add_offset_to_addr_reg_impl(offset);
                    self.emit_typed_load_indirect_impl(load_instr);
                }
                SlotAddr::Direct(slot) => {
                    let folded_slot = StackSlot(slot.0 + offset);
                    self.emit_typed_load_from_slot_impl(load_instr, folded_slot);
                }
                SlotAddr::Indirect(slot) => {
                    if let Some(&reg) = self.reg_assignments.get(&base.0) {
                        let reg_name = phys_reg_name(reg);
                        let dest_reg = Self::load_dest_reg(ty);
                        if offset != 0 {
                            self.state.emit_fmt(format_args!("    {} {}(%{}), {}", load_instr, offset, reg_name, dest_reg));
                        } else {
                            self.state.emit_fmt(format_args!("    {} (%{}), {}", load_instr, reg_name, dest_reg));
                        }
                    } else {
                        self.emit_load_ptr_from_slot_impl(slot, base.0);
                        if offset != 0 {
                            self.emit_add_offset_to_addr_reg_impl(offset);
                        }
                        self.emit_typed_load_indirect_impl(load_instr);
                    }
                }
            }
            self.store_rax_to(dest);
        }
    }

    pub(super) fn emit_typed_store_to_slot_impl(&mut self, instr: &'static str, ty: IrType, slot: StackSlot) {
        let reg = Self::reg_for_type("rax", ty);
        let out = &mut self.state.out;
        out.write_str("    ");
        out.write_str(instr);
        out.write_str(" %");
        out.write_str(reg);
        out.write_str(", ");
        out.write_i64(slot.0);
        out.write_str("(%rbp)");
        out.newline();
    }

    pub(super) fn emit_typed_load_from_slot_impl(&mut self, instr: &'static str, slot: StackSlot) {
        let dest_reg = if instr == "movl" { "%eax" } else { "%rax" };
        let out = &mut self.state.out;
        out.write_str("    ");
        out.write_str(instr);
        out.write_str(" ");
        out.write_i64(slot.0);
        out.write_str("(%rbp), ");
        out.write_str(dest_reg);
        out.newline();
    }

    pub(super) fn emit_save_acc_impl(&mut self) {
        self.state.emit("    movq %rax, %rdx");
    }

    pub(super) fn emit_load_ptr_from_slot_impl(&mut self, slot: StackSlot, val_id: u32) {
        if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rcx");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rcx");
        }
    }

    pub(super) fn emit_typed_store_indirect_impl(&mut self, instr: &'static str, ty: IrType) {
        let store_reg = Self::reg_for_type("rdx", ty);
        self.state.emit_fmt(format_args!("    {} %{}, (%rcx)", instr, store_reg));
    }

    pub(super) fn emit_typed_load_indirect_impl(&mut self, instr: &'static str) {
        let dest_reg = if instr == "movl" { "%eax" } else { "%rax" };
        self.state.emit_fmt(format_args!("    {} (%rcx), {}", instr, dest_reg));
    }

    pub(super) fn emit_add_offset_to_addr_reg_impl(&mut self, offset: i64) {
        self.state.out.emit_instr_imm_reg("    addq", offset, "rcx");
    }

    /// Compute the address of an alloca into `reg`, handling over-aligned allocas.
    pub(super) fn emit_alloca_addr_to(&mut self, reg: &str, val_id: u32, offset: i64) {
        if let Some(align) = self.state.alloca_over_align(val_id) {
            self.state.out.emit_instr_rbp_reg("    leaq", offset, reg);
            self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, reg);
            self.state.out.emit_instr_imm_reg("    andq", -(align as i64), reg);
        } else {
            self.state.out.emit_instr_rbp_reg("    leaq", offset, reg);
        }
    }

    pub(super) fn emit_slot_addr_to_secondary_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr_to("rcx", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rcx");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rcx");
        }
    }

    pub(super) fn emit_add_secondary_to_acc_impl(&mut self) {
        self.state.emit("    addq %rcx, %rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_gep_direct_const_impl(&mut self, slot: StackSlot, offset: i64) {
        let folded = slot.0 + offset;
        self.state.out.emit_instr_rbp_reg("    leaq", folded, "rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_gep_indirect_const_impl(&mut self, slot: StackSlot, offset: i64, val_id: u32) {
        if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rax");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rax");
        }
        if offset != 0 {
            self.state.out.emit_instr_mem_reg("    leaq", offset, "rax", "rax");
        }
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_gep_add_const_to_acc_impl(&mut self, offset: i64) {
        if offset != 0 {
            self.state.out.emit_instr_imm_reg("    addq", offset, "rax");
        }
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_add_imm_to_acc_impl(&mut self, imm: i64) {
        self.state.out.emit_instr_imm_reg("    addq", imm, "rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_round_up_acc_to_16_impl(&mut self) {
        self.state.emit("    addq $15, %rax");
        self.state.emit("    andq $-16, %rax");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_sub_sp_by_acc_impl(&mut self) {
        self.state.emit("    subq %rax, %rsp");
    }

    pub(super) fn emit_mov_sp_to_acc_impl(&mut self) {
        self.state.emit("    movq %rsp, %rax");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_mov_acc_to_sp_impl(&mut self) {
        self.state.emit("    movq %rax, %rsp");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_align_acc_impl(&mut self, align: usize) {
        self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, "rax");
        self.state.out.emit_instr_imm_reg("    andq", -(align as i64), "rax");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_memcpy_load_dest_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr_to("rdi", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rdi");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rdi");
        }
    }

    pub(super) fn emit_memcpy_load_src_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr_to("rsi", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rsi");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rsi");
        }
    }

    pub(super) fn emit_alloca_aligned_addr_impl(&mut self, slot: StackSlot, val_id: u32) {
        let align = self.state.alloca_over_align(val_id)
            .expect("alloca must have over-alignment for aligned addr emission");
        self.state.out.emit_instr_rbp_reg("    leaq", slot.0, "rcx");
        self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, "rcx");
        self.state.out.emit_instr_imm_reg("    andq", -(align as i64), "rcx");
    }

    pub(super) fn emit_alloca_aligned_addr_to_acc_impl(&mut self, slot: StackSlot, val_id: u32) {
        let align = self.state.alloca_over_align(val_id)
            .expect("alloca must have over-alignment for aligned addr emission");
        self.state.out.emit_instr_rbp_reg("    leaq", slot.0, "rax");
        self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, "rax");
        self.state.out.emit_instr_imm_reg("    andq", -(align as i64), "rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_acc_to_secondary_impl(&mut self) {
        self.state.emit("    movq %rax, %rcx");
    }

    pub(super) fn emit_memcpy_store_dest_from_acc_impl(&mut self) {
        self.state.emit("    movq %rcx, %rdi");
    }

    pub(super) fn emit_memcpy_store_src_from_acc_impl(&mut self) {
        self.state.emit("    movq %rcx, %rsi");
    }

    pub(super) fn emit_memcpy_impl_impl(&mut self, size: usize) {
        let qwords = size / 8;
        let remainder = size % 8;
        if qwords > 0 {
            self.state.out.emit_instr_imm_reg("    movq", qwords as i64, "rcx");
            self.state.emit("    rep movsq");
        }
        if remainder > 0 {
            self.state.out.emit_instr_imm_reg("    movq", remainder as i64, "rcx");
            self.state.emit("    rep movsb");
        }
    }

    // ---- Segment-prefixed memory ops ----

    pub(super) fn emit_seg_load_impl(&mut self, dest: &Value, ptr: &Value, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        self.operand_to_rax(&Operand::Value(*ptr));
        self.state.emit("    movq %rax, %rcx");
        let load_instr = Self::mov_load_for_type(ty);
        let dest_reg = Self::load_dest_reg(ty);
        self.state.emit_fmt(format_args!("    {} {}(%rcx), {}", load_instr, seg_prefix, dest_reg));
        self.store_rax_to(dest);
    }

    pub(super) fn emit_seg_load_symbol_impl(&mut self, dest: &Value, sym: &str, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        let load_instr = Self::mov_load_for_type(ty);
        let dest_reg = Self::load_dest_reg(ty);
        self.state.emit_fmt(format_args!("    {} {}{}(%rip), {}", load_instr, seg_prefix, sym, dest_reg));
        self.store_rax_to(dest);
    }

    pub(super) fn emit_seg_store_impl(&mut self, val: &Operand, ptr: &Value, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        self.operand_to_rax(val);
        self.state.emit("    movq %rax, %rdx");
        self.operand_to_rax(&Operand::Value(*ptr));
        self.state.emit("    movq %rax, %rcx");
        let store_instr = Self::mov_store_for_type(ty);
        let store_reg = Self::reg_for_type("rdx", ty);
        self.state.emit_fmt(format_args!("    {} %{}, {}(%rcx)", store_instr, store_reg, seg_prefix));
    }

    pub(super) fn emit_seg_store_symbol_impl(&mut self, val: &Operand, sym: &str, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        self.operand_to_rax(val);
        let store_instr = Self::mov_store_for_type(ty);
        let store_reg = Self::reg_for_type("rax", ty);
        self.state.emit_fmt(format_args!("    {} %{}, {}{}(%rip)", store_instr, store_reg, seg_prefix, sym));
    }
}
