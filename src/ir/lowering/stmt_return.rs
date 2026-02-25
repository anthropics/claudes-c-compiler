//! Return statement lowering.
//!
//! Extracted from stmt.rs to reduce that file's complexity. The `Stmt::Return`
//! match arm was 194 lines handling sret, two-register returns, complex returns,
//! and scalar-to-complex conversions. This module breaks that into focused helpers.

use crate::frontend::parser::ast::{BinOp, Expr};
use crate::ir::reexports::{
    Instruction,
    IrBinOp,
    IrConst,
    Operand,
    Value,
};
use crate::common::types::{AddressSpace, IrType, CType};
use super::lower::Lowerer;

impl Lowerer {
    /// Check if an expression is a BinaryOp with an arithmetic/bitwise operator.
    /// Such expressions never produce struct/union types, so struct-return checks can be skipped.
    fn expr_is_arithmetic_binop(e: &Expr) -> bool {
        matches!(e, Expr::BinaryOp(op, _, _, _) if matches!(op,
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
            | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor
            | BinOp::Shl | BinOp::Shr
            | BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
            | BinOp::LogicalAnd | BinOp::LogicalOr))
    }

    /// Lower a return statement's expression to an operand.
    /// Handles all return conventions: sret, two-register, complex decomposition,
    /// and scalar returns with implicit casts.
    pub(super) fn lower_return_expr(&mut self, e: &Expr) -> Operand {
        // Try sret return first (large structs and complex types via hidden pointer)
        if let Some(sret_alloca) = self.func().sret_ptr {
            if let Some(op) = self.try_sret_return(e, sret_alloca) {
                return op;
            }
        }

        // Two-register struct return (9-16 bytes packed into I128)
        if let Some(op) = self.try_two_reg_struct_return(e) {
            return op;
        }

        // Small struct return (<= 8 bytes loaded as I64)
        if let Some(op) = self.try_small_struct_return(e) {
            return op;
        }

        // Complex expression returned from complex-returning function
        if let Some(op) = self.try_complex_return(e) {
            return op;
        }

        // Non-complex expression returned from complex-returning function
        if let Some(op) = self.try_scalar_to_complex_return(e) {
            return op;
        }

        // Default: scalar return with implicit cast
        let val = self.lower_expr(e);
        let ret_ty = self.func_mut().return_type;
        let expr_ty = self.get_expr_type(e);
        if self.func_mut().return_is_bool {
            // For _Bool return, normalize at the source type before any truncation.
            self.emit_bool_normalize_typed(val, expr_ty)
        } else {
            self.emit_implicit_cast(val, expr_ty, ret_ty)
        }
    }

    /// Try to handle return via sret (hidden pointer for large structs/complex).
    /// Returns Some(operand) if this is an sret return, None otherwise.
    fn try_sret_return(&mut self, e: &Expr, sret_alloca: Value) -> Option<Operand> {
        // Large struct return via hidden pointer (sret).
        // Threshold is target-dependent: > 8 bytes on 32-bit, > 16 bytes on 64-bit.
        // struct_value_size may return Some(0) for FunctionCall/Conditional expressions
        // where CType::size() returns 0 for Struct/Union types. Fall back to the
        // current function's sret_size from sig metadata.
        let mut struct_size = self.struct_value_size(e).unwrap_or(0);
        if struct_size == 0 && !Self::expr_is_arithmetic_binop(e) {
            if let Some(ctype) = self.get_expr_ctype(e) {
                if ctype.is_struct_or_union() || ctype.is_vector() {
                    let fname = self.func().name.clone();
                    if let Some(size) = self.func_meta.sigs.get(&*fname)
                        .and_then(|s| s.sret_size) {
                        struct_size = size;
                    }
                }
            }
        }
        // On i686, the i386 SysV ABI uses sret for ALL struct sizes (threshold 0).
        // On 64-bit targets, structs > 16 bytes use sret.
        let sret_threshold = if crate::common::types::target_is_32bit() { 0 } else { 16 };
        if struct_size > sret_threshold {
            let src_addr = self.get_struct_base_addr(e);
            let sret_ptr = self.fresh_value();
            self.emit(Instruction::Load { dest: sret_ptr, ptr: sret_alloca, ty: IrType::Ptr , seg_override: AddressSpace::Default });
            self.emit(Instruction::Memcpy { dest: sret_ptr, src: src_addr, size: struct_size });
            return Some(Operand::Value(sret_ptr));
        }

        // Complex expression returned via sret
        let expr_ct = self.expr_ctype(e);
        let fname = self.func().name.clone(); let ret_ct = self.types.func_return_ctypes.get(&fname).cloned();
        if expr_ct.is_complex() {
            let val = self.lower_expr(e);
            let src_addr = if let Some(ref rct) = ret_ct {
                if rct.is_complex() && expr_ct != *rct {
                    let ptr = self.operand_to_value(val);
                    let converted = self.complex_to_complex(ptr, &expr_ct, rct);
                    self.operand_to_value(converted)
                } else {
                    self.operand_to_value(val)
                }
            } else {
                self.operand_to_value(val)
            };
            let complex_size = ret_ct.as_ref().unwrap_or(&expr_ct).size();
            let sret_ptr = self.fresh_value();
            self.emit(Instruction::Load { dest: sret_ptr, ptr: sret_alloca, ty: IrType::Ptr , seg_override: AddressSpace::Default });
            self.emit(Instruction::Memcpy { dest: sret_ptr, src: src_addr, size: complex_size });
            return Some(Operand::Value(sret_ptr));
        }

        // Non-complex expression returned from sret complex function
        if let Some(ref rct) = ret_ct {
            if rct.is_complex() {
                let val = self.lower_expr(e);
                let src_ir_ty = self.get_expr_type(e);
                let rct_clone = rct.clone();
                let complex_val = self.scalar_to_complex(val, src_ir_ty, &rct_clone);
                let src_addr = self.operand_to_value(complex_val);
                let complex_size = rct_clone.size();
                let sret_ptr = self.fresh_value();
                self.emit(Instruction::Load { dest: sret_ptr, ptr: sret_alloca, ty: IrType::Ptr , seg_override: AddressSpace::Default });
                self.emit(Instruction::Memcpy { dest: sret_ptr, src: src_addr, size: complex_size });
                return Some(Operand::Value(sret_ptr));
            }
        }

        None
    }

    /// Try two-register struct return (9-16 bytes packed into I128).
    fn try_two_reg_struct_return(&mut self, e: &Expr) -> Option<Operand> {
        // Complex types are handled by try_complex_return, not struct return paths.
        // Packed complex float params have is_struct=true but must not be treated as
        // struct returns -- they need proper complex-to-scalar conversion.
        let expr_ct = self.expr_ctype(e);
        if expr_ct.is_complex() {
            return None;
        }
        // Get the struct size. struct_value_size may return Some(0) for expressions
        // where CType::size() returns 0 (Struct/Union types without resolved size).
        // In that case, fall back to the function's own two_reg_ret_size from sig
        // metadata, since the return type size is reliably known.
        let mut struct_size = self.struct_value_size(e).unwrap_or(0);
        if struct_size == 0 && !Self::expr_is_arithmetic_binop(e) {
            if let Some(ctype) = self.get_expr_ctype(e) {
                if ctype.is_struct_or_union() || ctype.is_vector() {
                    let fname = self.func().name.clone();
                    if let Some(size) = self.func_meta.sigs.get(&*fname)
                        .and_then(|s| s.two_reg_ret_size) {
                        struct_size = size;
                    }
                }
            }
        }
        // Two-register return: 9-16 bytes on 64-bit targets only.
        // On 32-bit targets, there is no two-register (I128) return path.
        if crate::common::types::target_is_32bit() {
            return None;
        }
        if struct_size <= 8 || struct_size > 16 {
            return None;
        }
        let addr = self.get_struct_base_addr(e);
        // Load low 8 bytes
        let lo = self.fresh_value();
        self.emit(Instruction::Load { dest: lo, ptr: addr, ty: IrType::I64 , seg_override: AddressSpace::Default });
        // Load high bytes
        let hi_ptr = self.fresh_value();
        self.emit(Instruction::GetElementPtr { dest: hi_ptr, base: addr, offset: Operand::Const(IrConst::I64(8)), ty: IrType::I64 });
        let hi = self.fresh_value();
        self.emit(Instruction::Load { dest: hi, ptr: hi_ptr, ty: IrType::I64 , seg_override: AddressSpace::Default });
        // Pack into I128: (hi << 64) | lo (zero-extend both halves)
        let hi_wide = self.fresh_value();
        self.emit(Instruction::Cast { dest: hi_wide, src: Operand::Value(hi), from_ty: IrType::U64, to_ty: IrType::I128 });
        let lo_wide = self.fresh_value();
        self.emit(Instruction::Cast { dest: lo_wide, src: Operand::Value(lo), from_ty: IrType::U64, to_ty: IrType::I128 });
        let shifted = self.fresh_value();
        self.emit(Instruction::BinOp { dest: shifted, op: IrBinOp::Shl, lhs: Operand::Value(hi_wide), rhs: Operand::Const(IrConst::I64(64)), ty: IrType::I128 });
        let packed = self.fresh_value();
        self.emit(Instruction::BinOp { dest: packed, op: IrBinOp::Or, lhs: Operand::Value(shifted), rhs: Operand::Value(lo_wide), ty: IrType::I128 });
        Some(Operand::Value(packed))
    }

    /// Try small struct return (<= 8 bytes loaded as I64).
    /// Only used on 64-bit targets; on i686, ALL structs use sret.
    fn try_small_struct_return(&mut self, e: &Expr) -> Option<Operand> {
        // On i686, the i386 SysV ABI uses sret for ALL struct returns.
        // Small structs must not be packed into eax:edx registers.
        if crate::common::types::target_is_32bit() {
            return None;
        }
        // Complex types are handled by try_complex_return, not struct return paths.
        // Packed complex float params have is_struct=true but must not be treated as
        // struct returns -- they need proper complex-to-scalar conversion.
        let expr_ct = self.expr_ctype(e);
        if expr_ct.is_complex() {
            return None;
        }
        let struct_size = self.struct_value_size(e)?;
        // struct_value_size returns Some(0) for FunctionCall/Conditional expressions
        // where CType::size() returns 0 for Struct/Union types. Don't handle those here;
        // they should be caught by try_sret_return or try_two_reg_struct_return fallbacks.
        if struct_size == 0 || struct_size > 8 {
            return None;
        }
        let addr = self.get_struct_base_addr(e);
        let dest = self.fresh_value();
        self.emit(Instruction::Load { dest, ptr: addr, ty: IrType::I64 , seg_override: AddressSpace::Default });
        Some(Operand::Value(dest))
    }

    /// Try returning a complex expression from a complex-returning function.
    /// Handles decomposition into register pairs or I64 packing.
    fn try_complex_return(&mut self, e: &Expr) -> Option<Operand> {
        let expr_ct = self.expr_ctype(e);
        if !expr_ct.is_complex() {
            return None;
        }

        let fname = self.func().name.clone(); let ret_ct = self.types.func_return_ctypes.get(&fname).cloned();
        if let Some(ref rct) = ret_ct {
            let val = self.lower_expr(e);
            let src_ptr = if rct.is_complex() && expr_ct != *rct {
                let ptr = self.operand_to_value(val);
                let converted = self.complex_to_complex(ptr, &expr_ct, rct);
                self.operand_to_value(converted)
            } else {
                self.operand_to_value(val)
            };

            // _Complex double: return real in xmm0, imag in xmm1
            if *rct == CType::ComplexDouble && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
                let real = self.load_complex_real(src_ptr, rct);
                let imag = self.load_complex_imag(src_ptr, rct);
                self.emit(Instruction::SetReturnF64Second { src: imag });
                return Some(real);
            }

            // _Complex long double on x86-64: return real in st(0), imag in st(1)
            if *rct == CType::ComplexLongDouble && self.returns_complex_long_double_in_regs()
                && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
                let real = self.load_complex_real(src_ptr, rct);
                let imag = self.load_complex_imag(src_ptr, rct);
                self.emit(Instruction::SetReturnF128Second { src: imag });
                return Some(real);
            }

            // _Complex float: platform-specific return convention
            if *rct == CType::ComplexFloat && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
                if self.uses_packed_complex_float() {
                    // x86-64: load packed 8 bytes as F64 for one XMM register return
                    let packed = self.fresh_value();
                    self.emit(Instruction::Load { dest: packed, ptr: src_ptr, ty: IrType::F64 , seg_override: AddressSpace::Default });
                    return Some(Operand::Value(packed));
                } else if !self.decomposes_complex_float() {
                    // i686: load packed 8 bytes as I64 for eax:edx register return
                    let packed = self.fresh_value();
                    self.emit(Instruction::Load { dest: packed, ptr: src_ptr, ty: IrType::I64 , seg_override: AddressSpace::Default });
                    return Some(Operand::Value(packed));
                } else {
                    // ARM/RISC-V: return real in first FP reg (F32), imag in second FP reg (F32)
                    let real = self.load_complex_real(src_ptr, rct);
                    let imag = self.load_complex_imag(src_ptr, rct);
                    self.emit(Instruction::SetReturnF32Second { src: imag });
                    return Some(real);
                }
            }

            // _Complex long double on x86-64/ARM64: return real in FP reg, imag in second FP reg
            if *rct == CType::ComplexLongDouble && self.returns_complex_long_double_in_regs() && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
                let real = self.load_complex_real(src_ptr, rct);
                let imag = self.load_complex_imag(src_ptr, rct);
                self.emit(Instruction::SetReturnF128Second { src: imag });
                return Some(real);
            }
        }

        // Complex expression returned from non-complex function: extract real part
        let ret_ty = self.func_mut().return_type;
        if ret_ty != IrType::Ptr {
            let val2 = self.lower_expr(e);
            let ptr = self.operand_to_value(val2);
            let val2 = if self.func_mut().return_is_bool {
                // For _Bool return, check both real and imag parts per C11 6.3.1.2
                self.lower_complex_to_bool(ptr, &expr_ct)
            } else {
                let real_part = self.load_complex_real(ptr, &expr_ct);
                let from_ty = Self::complex_component_ir_type(&expr_ct);
                self.emit_implicit_cast(real_part, from_ty, ret_ty)
            };
            return Some(val2);
        }

        None
    }

    /// Try converting a non-complex scalar to complex for return from a complex function.
    fn try_scalar_to_complex_return(&mut self, e: &Expr) -> Option<Operand> {
        let expr_ct = self.expr_ctype(e);
        let fname = self.func().name.clone(); let ret_ct = self.types.func_return_ctypes.get(&fname).cloned();
        let rct = ret_ct.as_ref()?;
        if !rct.is_complex() || expr_ct.is_complex() {
            return None;
        }

        let val = self.lower_expr(e);
        let src_ir_ty = self.get_expr_type(e);
        let rct_clone = rct.clone();
        let complex_val = self.scalar_to_complex(val, src_ir_ty, &rct_clone);

        // _Complex double: decompose into two FP return registers
        if rct_clone == CType::ComplexDouble && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
            let src_ptr = self.operand_to_value(complex_val);
            let real = self.load_complex_real(src_ptr, &rct_clone);
            let imag = self.load_complex_imag(src_ptr, &rct_clone);
            self.emit(Instruction::SetReturnF64Second { src: imag });
            return Some(real);
        }

        // _Complex long double on x86-64: return real in st(0), imag in st(1)
        if rct_clone == CType::ComplexLongDouble && self.returns_complex_long_double_in_regs()
            && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
            let src_ptr = self.operand_to_value(complex_val);
            let real = self.load_complex_real(src_ptr, &rct_clone);
            let imag = self.load_complex_imag(src_ptr, &rct_clone);
            self.emit(Instruction::SetReturnF128Second { src: imag });
            return Some(real);
        }

        // For sret returns, copy to the hidden pointer
        if let Some(sret_alloca) = self.func().sret_ptr {
            let src_addr = self.operand_to_value(complex_val);
            let complex_size = rct_clone.size();
            let sret_ptr = self.fresh_value();
            self.emit(Instruction::Load { dest: sret_ptr, ptr: sret_alloca, ty: IrType::Ptr , seg_override: AddressSpace::Default });
            self.emit(Instruction::Memcpy { dest: sret_ptr, src: src_addr, size: complex_size });
            return Some(Operand::Value(sret_ptr));
        }

        // For non-sret complex float: platform-specific return convention
        if rct_clone == CType::ComplexFloat && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
            let ptr = self.operand_to_value(complex_val);
            if self.uses_packed_complex_float() {
                // x86-64: pack into F64 for one XMM register return
                let packed = self.fresh_value();
                self.emit(Instruction::Load { dest: packed, ptr, ty: IrType::F64 , seg_override: AddressSpace::Default });
                return Some(Operand::Value(packed));
            } else if !self.decomposes_complex_float() {
                // i686: pack into I64 for eax:edx register return
                let packed = self.fresh_value();
                self.emit(Instruction::Load { dest: packed, ptr, ty: IrType::I64 , seg_override: AddressSpace::Default });
                return Some(Operand::Value(packed));
            } else {
                // ARM/RISC-V: return real in first FP reg, imag in second
                let real = self.load_complex_real(ptr, &rct_clone);
                let imag = self.load_complex_imag(ptr, &rct_clone);
                self.emit(Instruction::SetReturnF32Second { src: imag });
                return Some(real);
            }
        }

        // _Complex long double on x86-64/ARM64: return real in FP reg, imag in second FP reg
        if rct_clone == CType::ComplexLongDouble && self.returns_complex_long_double_in_regs() && self.func_meta.sigs.get(&fname).and_then(|s| s.sret_size).is_none() {
            let ptr = self.operand_to_value(complex_val);
            let real = self.load_complex_real(ptr, &rct_clone);
            let imag = self.load_complex_imag(ptr, &rct_clone);
            self.emit(Instruction::SetReturnF128Second { src: imag });
            return Some(real);
        }

        Some(complex_val)
    }
}
