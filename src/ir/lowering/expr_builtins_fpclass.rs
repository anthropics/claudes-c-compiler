//! Floating-point classification builtins for the lowerer.
//!
//! Handles __builtin_fpclassify, __builtin_isnan, __builtin_isinf,
//! __builtin_isfinite, __builtin_isnormal, __builtin_signbit, and
//! __builtin_isinf_sign. Uses bit manipulation for F32/F64 and
//! delegates to libc for F128 (long double).

use crate::frontend::parser::ast::{Expr};
use crate::ir::reexports::{
    CallInfo,
    Instruction,
    IrBinOp,
    IrCmpOp,
    IrConst,
    Operand,
    Value,
};
use std::rc::Rc;
use crate::common::types::{AddressSpace, IrType};
use super::lower::Lowerer;

impl Lowerer {
    /// Helper: get the type of the last argument for fp classification builtins.
    fn lower_fp_classify_arg(&mut self, args: &[Expr]) -> (IrType, Operand) {
        let arg_expr = args.last().expect("fp classify builtin requires at least one argument");
        let arg_ty = self.get_expr_type(arg_expr);
        let arg_val = self.lower_expr(arg_expr);
        (arg_ty, arg_val)
    }

    /// Emit a call to a libc FP classification function for F128 (long double) arguments.
    /// Returns the call result as a Value. The libc functions take a long double and return int.
    fn emit_f128_classify_libcall(&mut self, func_name: &str, arg_val: Operand) -> Value {
        let dest = self.fresh_value();
        self.emit(Instruction::Call {
            func: Rc::from(func_name),
            info: CallInfo {
                dest: Some(dest),
                args: vec![arg_val],
                arg_types: vec![IrType::F128],
                return_type: IrType::I32,
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
        });
        dest
    }

    /// Convert a libc classification result to a boolean (non-zero -> 1, zero -> 0).
    fn classify_result_to_bool(&mut self, result: Value) -> Value {
        self.emit_cmp_val(
            IrCmpOp::Ne,
            Operand::Value(result),
            Operand::Const(IrConst::I32(0)),
            IrType::I32,
        )
    }

    /// Zero-extend a narrow value (I8 from Cmp, or I32 from BinOp) to I64.
    /// On 32-bit targets (i686), I64 values occupy two 32-bit stack slots;
    /// without this cast, loading a narrow value as I64 reads uninitialized
    /// bytes from the upper slot, causing incorrect results in I64 arithmetic.
    fn widen_to_i64(&mut self, val: Value) -> Value {
        self.emit_cast_val(Operand::Value(val), IrType::I32, IrType::I64)
    }

    /// Bitcast a float/double to its integer bit representation via memory.
    /// F32 -> I32, F64 -> I64. Uses alloca+store+load for a true bitcast.
    /// Note: F128 classification builtins use libc calls instead of this path.
    fn bitcast_float_to_int(&mut self, val: Operand, float_ty: IrType) -> (Operand, IrType) {
        let (int_ty, store_ty, size) = match float_ty {
            IrType::F32 => (IrType::I32, IrType::F32, 4),
            // F128 is stored as F64 at backend level, treat identically
            _ => (IrType::I64, IrType::F64, 8),
        };

        let tmp_alloca = self.fresh_value();
        self.emit(Instruction::Alloca { dest: tmp_alloca, size, ty: store_ty, align: 0, volatile: false });

        let val_v = self.operand_to_value(val);
        self.emit(Instruction::Store { val: Operand::Value(val_v), ptr: tmp_alloca, ty: store_ty, seg_override: AddressSpace::Default });

        let result = self.fresh_value();
        self.emit(Instruction::Load { dest: result, ptr: tmp_alloca, ty: int_ty, seg_override: AddressSpace::Default });

        (Operand::Value(result), int_ty)
    }

    /// Floating-point bit-layout constants for classification.
    /// Returns (abs_mask, exp_only, exp_shift, exp_field_max, mant_mask).
    /// Note: F128 classification builtins use libc calls instead of these masks.
    pub(super) fn fp_masks(float_ty: IrType) -> (i64, i64, i64, i64, i64) {
        match float_ty {
            IrType::F32 => (
                0x7FFFFFFF_i64,
                0x7F800000_i64,
                23,
                0xFF,
                0x007FFFFF_i64,
            ),
            // F128 uses F64 masks because the backend stores long double as double
            _ => (
                0x7FFFFFFFFFFFFFFF_i64,
                0x7FF0000000000000_u64 as i64,
                52,
                0x7FF,
                0x000FFFFFFFFFFFFF_u64 as i64,
            ),
        }
    }

    /// Compute the absolute value (sign-stripped) of float bits: bits & abs_mask.
    fn fp_abs_bits(&mut self, bits: &Operand, int_ty: IrType, abs_mask: i64) -> Value {
        self.emit_binop_val(IrBinOp::And, *bits, Operand::Const(IrConst::I64(abs_mask)), int_ty)
    }

    /// Lower __builtin_fpclassify(nan_val, inf_val, norm_val, subnorm_val, zero_val, x).
    /// Uses arithmetic select: result = sum of (class_val[i] * is_class[i]).
    /// For F128 (long double), calls __fpclassifyl from libc and maps the result
    /// to the user-provided class values via comparisons.
    pub(super) fn lower_builtin_fpclassify(&mut self, args: &[Expr]) -> Option<Operand> {
        if args.len() < 6 {
            return Some(Operand::Const(IrConst::I64(0)));
        }

        let class_vals: Vec<_> = (0..5).map(|i| self.lower_expr(&args[i])).collect();
        let arg_ty = self.get_expr_type(&args[5]);
        let arg_val = self.lower_expr(&args[5]);

        if arg_ty == IrType::F128 {
            // TODO: FP_* constants assume glibc/Linux; may differ on musl/BSD
            // __fpclassifyl returns: FP_NAN=0, FP_INFINITE=1, FP_ZERO=2, FP_SUBNORMAL=3, FP_NORMAL=4
            let cls = self.emit_f128_classify_libcall("__fpclassifyl", arg_val);
            // Map libc class to user-provided values:
            // class_vals[0]=nan, [1]=inf, [2]=normal, [3]=subnormal, [4]=zero
            // libc: 0=nan, 1=inf, 2=zero, 3=subnormal, 4=normal
            // So: libc_0->class_vals[0], libc_1->class_vals[1], libc_2->class_vals[4],
            //     libc_3->class_vals[3], libc_4->class_vals[2]
            let libc_to_user = [0usize, 1, 4, 3, 2]; // libc class index -> user class_vals index
            // Comparison results (I8) must be widened to I64 before I64 multiply,
            // because on 32-bit targets I64 occupies two stack slots and loading
            // a narrow value as I64 reads uninitialized upper bytes.
            let is_class: Vec<Value> = (0..5).map(|libc_cls| {
                let cmp = self.emit_cmp_val(
                    IrCmpOp::Eq,
                    Operand::Value(cls),
                    Operand::Const(IrConst::I32(libc_cls)),
                    IrType::I32,
                );
                self.widen_to_i64(cmp)
            }).collect();

            let mut result = self.emit_binop_val(
                IrBinOp::Mul,
                class_vals[libc_to_user[0]],
                Operand::Value(is_class[0]),
                IrType::I64,
            );
            for libc_cls in 1..5 {
                let user_idx = libc_to_user[libc_cls];
                let contrib = self.emit_binop_val(
                    IrBinOp::Mul,
                    class_vals[user_idx],
                    Operand::Value(is_class[libc_cls]),
                    IrType::I64,
                );
                result = self.emit_binop_val(
                    IrBinOp::Add,
                    Operand::Value(result),
                    Operand::Value(contrib),
                    IrType::I64,
                );
            }
            return Some(Operand::Value(result));
        }

        let (bits, int_ty) = self.bitcast_float_to_int(arg_val, arg_ty);
        let (_abs_mask, _exp_only, exp_shift, exp_field_max, mant_mask) = Self::fp_masks(arg_ty);

        let shifted = self.emit_binop_val(IrBinOp::LShr, bits, Operand::Const(IrConst::I64(exp_shift)), int_ty);
        let exponent = self.emit_binop_val(IrBinOp::And, Operand::Value(shifted), Operand::Const(IrConst::I64(exp_field_max)), int_ty);
        let mantissa = self.emit_binop_val(IrBinOp::And, bits, Operand::Const(IrConst::I64(mant_mask)), int_ty);

        let exp_is_max = self.emit_cmp_val(IrCmpOp::Eq, Operand::Value(exponent), Operand::Const(IrConst::I64(exp_field_max)), int_ty);
        let exp_is_zero = self.emit_cmp_val(IrCmpOp::Eq, Operand::Value(exponent), Operand::Const(IrConst::I64(0)), int_ty);
        let mant_is_zero = self.emit_cmp_val(IrCmpOp::Eq, Operand::Value(mantissa), Operand::Const(IrConst::I64(0)), int_ty);
        let mant_not_zero = self.emit_cmp_val(IrCmpOp::Ne, Operand::Value(mantissa), Operand::Const(IrConst::I64(0)), int_ty);

        let is_nan = self.emit_binop_val(IrBinOp::And, Operand::Value(exp_is_max), Operand::Value(mant_not_zero), IrType::I32);
        let is_inf = self.emit_binop_val(IrBinOp::And, Operand::Value(exp_is_max), Operand::Value(mant_is_zero), IrType::I32);
        let is_zero = self.emit_binop_val(IrBinOp::And, Operand::Value(exp_is_zero), Operand::Value(mant_is_zero), IrType::I32);
        let is_subnorm = self.emit_binop_val(IrBinOp::And, Operand::Value(exp_is_zero), Operand::Value(mant_not_zero), IrType::I32);

        let mut special = is_nan;
        for &flag in &[is_inf, is_zero, is_subnorm] {
            special = self.emit_binop_val(IrBinOp::Or, Operand::Value(special), Operand::Value(flag), IrType::I32);
        }
        let is_normal = self.emit_binop_val(IrBinOp::Xor, Operand::Value(special), Operand::Const(IrConst::I64(1)), IrType::I32);

        // Order: nan=0, inf=1, normal=2, subnormal=3, zero=4
        // Widen I32 flags to I64 before multiply on 32-bit targets.
        let class_flags = [is_nan, is_inf, is_normal, is_subnorm, is_zero];
        let contribs: Vec<_> = class_vals.iter().zip(class_flags.iter())
            .map(|(val, &flag)| (*val, self.widen_to_i64(flag)))
            .collect();
        let mut result = self.emit_binop_val(IrBinOp::Mul, contribs[0].0, Operand::Value(contribs[0].1), IrType::I64);
        for &(ref val, flag) in &contribs[1..] {
            let contrib = self.emit_binop_val(IrBinOp::Mul, *val, Operand::Value(flag), IrType::I64);
            result = self.emit_binop_val(IrBinOp::Add, Operand::Value(result), Operand::Value(contrib), IrType::I64);
        }

        Some(Operand::Value(result))
    }

    /// __builtin_isnan(x) -> 1 if x is NaN. NaN: abs(bits) > exp_only.
    /// For F128 (long double), delegates to libc __isnanl for correct 80-bit/128-bit handling.
    pub(super) fn lower_builtin_isnan(&mut self, args: &[Expr]) -> Option<Operand> {
        if args.is_empty() { return Some(Operand::Const(IrConst::I64(0))); }
        let (arg_ty, arg_val) = self.lower_fp_classify_arg(args);
        if arg_ty == IrType::F128 {
            let result = self.emit_f128_classify_libcall("__isnanl", arg_val);
            let bool_result = self.classify_result_to_bool(result);
            return Some(Operand::Value(bool_result));
        }
        let (bits, int_ty) = self.bitcast_float_to_int(arg_val, arg_ty);
        let masks = Self::fp_masks(arg_ty);
        let abs_val = self.fp_abs_bits(&bits, int_ty, masks.0);
        let result = self.emit_cmp_val(IrCmpOp::Ugt, Operand::Value(abs_val), Operand::Const(IrConst::I64(masks.1)), int_ty);
        Some(Operand::Value(result))
    }

    /// __builtin_isinf(x) -> nonzero if +/-infinity. Inf: abs(bits) == exp_only.
    /// For F128 (long double), delegates to libc __isinfl.
    pub(super) fn lower_builtin_isinf(&mut self, args: &[Expr]) -> Option<Operand> {
        if args.is_empty() { return Some(Operand::Const(IrConst::I64(0))); }
        let (arg_ty, arg_val) = self.lower_fp_classify_arg(args);
        if arg_ty == IrType::F128 {
            let result = self.emit_f128_classify_libcall("__isinfl", arg_val);
            let bool_result = self.classify_result_to_bool(result);
            return Some(Operand::Value(bool_result));
        }
        let (bits, int_ty) = self.bitcast_float_to_int(arg_val, arg_ty);
        let masks = Self::fp_masks(arg_ty);
        let abs_val = self.fp_abs_bits(&bits, int_ty, masks.0);
        let result = self.emit_cmp_val(IrCmpOp::Eq, Operand::Value(abs_val), Operand::Const(IrConst::I64(masks.1)), int_ty);
        Some(Operand::Value(result))
    }

    /// __builtin_isfinite(x) -> 1 if finite. Finite: (bits & exp_only) != exp_only.
    /// For F128 (long double), delegates to libc __finitel.
    pub(super) fn lower_builtin_isfinite(&mut self, args: &[Expr]) -> Option<Operand> {
        if args.is_empty() { return Some(Operand::Const(IrConst::I64(0))); }
        let (arg_ty, arg_val) = self.lower_fp_classify_arg(args);
        if arg_ty == IrType::F128 {
            let result = self.emit_f128_classify_libcall("__finitel", arg_val);
            let bool_result = self.classify_result_to_bool(result);
            return Some(Operand::Value(bool_result));
        }
        let (bits, int_ty) = self.bitcast_float_to_int(arg_val, arg_ty);
        let masks = Self::fp_masks(arg_ty);
        let exp_bits = self.emit_binop_val(IrBinOp::And, bits, Operand::Const(IrConst::I64(masks.1)), int_ty);
        let result = self.emit_cmp_val(IrCmpOp::Ne, Operand::Value(exp_bits), Operand::Const(IrConst::I64(masks.1)), int_ty);
        Some(Operand::Value(result))
    }

    /// __builtin_isnormal(x) -> 1 if normal. Normal: exp != 0 AND exp != max.
    /// For F128 (long double), delegates to libc __fpclassifyl and compares to FP_NORMAL.
    pub(super) fn lower_builtin_isnormal(&mut self, args: &[Expr]) -> Option<Operand> {
        if args.is_empty() { return Some(Operand::Const(IrConst::I64(0))); }
        let (arg_ty, arg_val) = self.lower_fp_classify_arg(args);
        if arg_ty == IrType::F128 {
            // TODO: FP_NORMAL = 4 assumes glibc/Linux FP_* constants; may differ on musl/BSD
            let cls = self.emit_f128_classify_libcall("__fpclassifyl", arg_val);
            let result = self.emit_cmp_val(
                IrCmpOp::Eq,
                Operand::Value(cls),
                Operand::Const(IrConst::I32(4)),
                IrType::I32,
            );
            return Some(Operand::Value(result));
        }
        let (bits, int_ty) = self.bitcast_float_to_int(arg_val, arg_ty);
        let masks = Self::fp_masks(arg_ty);
        let exp_bits = self.emit_binop_val(IrBinOp::And, bits, Operand::Const(IrConst::I64(masks.1)), int_ty);
        let exponent = self.emit_binop_val(IrBinOp::LShr, Operand::Value(exp_bits), Operand::Const(IrConst::I64(masks.2)), int_ty);
        let exp_nonzero = self.emit_cmp_val(IrCmpOp::Ne, Operand::Value(exponent), Operand::Const(IrConst::I64(0)), int_ty);
        let exp_not_max = self.emit_cmp_val(IrCmpOp::Ne, Operand::Value(exponent), Operand::Const(IrConst::I64(masks.3)), int_ty);
        let result = self.emit_binop_val(IrBinOp::And, Operand::Value(exp_nonzero), Operand::Value(exp_not_max), IrType::I32);
        Some(Operand::Value(result))
    }

    /// __builtin_signbit(x) -> nonzero if sign bit set.
    /// For F128 (long double), delegates to libc __signbitl.
    pub(super) fn lower_builtin_signbit(&mut self, args: &[Expr]) -> Option<Operand> {
        if args.is_empty() { return Some(Operand::Const(IrConst::I64(0))); }
        let (arg_ty, arg_val) = self.lower_fp_classify_arg(args);
        if arg_ty == IrType::F128 {
            let result = self.emit_f128_classify_libcall("__signbitl", arg_val);
            let bool_result = self.classify_result_to_bool(result);
            return Some(Operand::Value(bool_result));
        }
        let (bits, int_ty) = self.bitcast_float_to_int(arg_val, arg_ty);
        let sign_shift = if arg_ty == IrType::F32 { 31_i64 } else { 63_i64 };
        let shifted = self.emit_binop_val(IrBinOp::LShr, bits, Operand::Const(IrConst::I64(sign_shift)), int_ty);
        let result = self.emit_binop_val(IrBinOp::And, Operand::Value(shifted), Operand::Const(IrConst::I64(1)), int_ty);
        Some(Operand::Value(result))
    }

    /// __builtin_isinf_sign(x) -> -1 if -inf, +1 if +inf, 0 otherwise.
    /// For F128 (long double), delegates to libc __isinfl and __signbitl.
    pub(super) fn lower_builtin_isinf_sign(&mut self, args: &[Expr]) -> Option<Operand> {
        if args.is_empty() { return Some(Operand::Const(IrConst::I64(0))); }
        let (arg_ty, arg_val) = self.lower_fp_classify_arg(args);
        if arg_ty == IrType::F128 {
            // __isinfl returns nonzero if inf
            let isinf_result = self.emit_f128_classify_libcall("__isinfl", arg_val);
            let is_inf_narrow = self.classify_result_to_bool(isinf_result);
            // __signbitl returns nonzero if negative
            let signbit_result = self.emit_f128_classify_libcall("__signbitl", arg_val);
            let is_neg_narrow = self.classify_result_to_bool(signbit_result);
            // Widen comparison results (I8) to I64 before I64 arithmetic.
            // On i686, I64 occupies two 32-bit stack slots; without widening,
            // loading a narrow value as I64 reads garbage in the upper slot.
            let is_inf = self.widen_to_i64(is_inf_narrow);
            let is_neg = self.widen_to_i64(is_neg_narrow);
            // direction = 1 - 2*is_neg -> +1 if positive, -1 if negative
            let neg_x2 = self.emit_binop_val(IrBinOp::Mul, Operand::Value(is_neg), Operand::Const(IrConst::I64(2)), IrType::I64);
            let direction = self.emit_binop_val(IrBinOp::Sub, Operand::Const(IrConst::I64(1)), Operand::Value(neg_x2), IrType::I64);
            let result = self.emit_binop_val(IrBinOp::Mul, Operand::Value(is_inf), Operand::Value(direction), IrType::I64);
            return Some(Operand::Value(result));
        }
        let (bits, int_ty) = self.bitcast_float_to_int(arg_val, arg_ty);
        let masks = Self::fp_masks(arg_ty);
        let sign_shift = if arg_ty == IrType::F32 { 31_i64 } else { 63_i64 };
        let abs_val = self.fp_abs_bits(&bits, int_ty, masks.0);
        let is_inf_narrow = self.emit_cmp_val(IrCmpOp::Eq, Operand::Value(abs_val), Operand::Const(IrConst::I64(masks.1)), int_ty);
        // Widen is_inf from I8 (Cmp result) to I64 before I64 multiply.
        // On i686, I64 values use two 32-bit stack slots (eax:edx pair);
        // a narrow Cmp result only writes the low slot, leaving the upper
        // slot uninitialized. The I64 multiply then reads garbage from
        // the upper slot, producing wrong results (e.g. isinf appearing
        // to return non-zero for finite values like 1.23).
        let is_inf = self.widen_to_i64(is_inf_narrow);
        let sign_shifted = self.emit_binop_val(IrBinOp::LShr, bits, Operand::Const(IrConst::I64(sign_shift)), int_ty);
        let sign_bit = self.emit_binop_val(IrBinOp::And, Operand::Value(sign_shifted), Operand::Const(IrConst::I64(1)), int_ty);
        let sign_x2 = self.emit_binop_val(IrBinOp::Mul, Operand::Value(sign_bit), Operand::Const(IrConst::I64(2)), IrType::I64);
        let direction = self.emit_binop_val(IrBinOp::Sub, Operand::Const(IrConst::I64(1)), Operand::Value(sign_x2), IrType::I64);
        let result = self.emit_binop_val(IrBinOp::Mul, Operand::Value(is_inf), Operand::Value(direction), IrType::I64);
        Some(Operand::Value(result))
    }
}
