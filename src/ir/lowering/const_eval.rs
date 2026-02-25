//! Constant expression evaluation for the IR lowerer.
//!
//! Core constant-folding logic: integer and floating-point arithmetic, cast
//! chains, sizeof/offsetof evaluation, and the top-level `eval_const_expr`
//! entry point. Global-address resolution lives in `const_eval_global_addr`
//! and initializer-list size computation in `const_eval_init_size`.
//!
//! Shared pure-evaluation logic (literal eval, builtin folding, sub-int promotion,
//! binary operation arithmetic) lives in `common::const_eval` and `common::const_arith`,
//! called by both this module and `sema::const_eval`.

use crate::frontend::parser::ast::{
    BinOp,
    Expr,
    Initializer,
    SizeofArg,
    TypeSpecifier,
    UnaryOp,
};
use crate::ir::reexports::{GlobalInit, IrConst};
use crate::common::types::{CType, IrType};
use crate::common::const_arith;
use crate::common::const_eval as shared_const_eval;
use super::lower::Lowerer;

impl Lowerer {
    /// Look up a pre-computed constant value from sema's ConstMap.
    /// Returns Some(IrConst) if sema successfully evaluated this expression
    /// at compile time during its pass.
    fn lookup_sema_const(&self, expr: &Expr) -> Option<IrConst> {
        self.sema_const_values.get(&expr.id()).cloned()
    }

    /// Check if an expression tree contains a Sizeof node at any depth.
    /// Used to avoid trusting sema's pre-computed values for expressions involving
    /// sizeof, since sema evaluates sizeof before the lowerer resolves unsized array
    /// dimensions from initializers. For example, `sizeof(cases) / sizeof(cases[0]) + 1`
    /// contains Sizeof nodes in the division subtree, so the entire expression must
    /// be recomputed by the lowerer.
    fn expr_contains_sizeof(expr: &Expr) -> bool {
        match expr {
            Expr::Sizeof(_, _) => true,
            Expr::BinaryOp(_, lhs, rhs, _) => {
                Self::expr_contains_sizeof(lhs) || Self::expr_contains_sizeof(rhs)
            }
            Expr::UnaryOp(_, inner, _) | Expr::PostfixOp(_, inner, _) => {
                Self::expr_contains_sizeof(inner)
            }
            Expr::Cast(_, inner, _) => Self::expr_contains_sizeof(inner),
            Expr::Conditional(cond, then_e, else_e, _) => {
                Self::expr_contains_sizeof(cond)
                    || Self::expr_contains_sizeof(then_e)
                    || Self::expr_contains_sizeof(else_e)
            }
            Expr::GnuConditional(cond, else_e, _) => {
                Self::expr_contains_sizeof(cond) || Self::expr_contains_sizeof(else_e)
            }
            Expr::Comma(lhs, rhs, _) => {
                Self::expr_contains_sizeof(lhs) || Self::expr_contains_sizeof(rhs)
            }
            _ => false,
        }
    }

    /// Try to evaluate a constant expression at compile time.
    ///
    /// First checks sema's pre-computed ConstMap (O(1) lookup for expressions
    /// that sema could evaluate). Falls back to the lowerer's own evaluation
    /// for expressions that require lowering-specific state (global addresses,
    /// const local values, etc.).
    pub(super) fn eval_const_expr(&self, expr: &Expr) -> Option<IrConst> {
        // Fast path: consult sema's pre-computed constant values.
        // This avoids re-evaluating expressions that sema already handled.
        // We skip the sema lookup for:
        //   - Identifiers: the lowerer may have more information (const local values,
        //     static locals) that sema lacks.
        //   - Expressions containing sizeof: sema may have computed sizeof for unsized
        //     arrays (e.g., `PT cases[] = {1,2,3,...}`) before the lowerer resolved the
        //     actual element count from the initializer. The lowerer's sizeof_expr/
        //     sizeof_type uses the correctly-sized global, so we must recompute.
        //     This check covers sizeof itself and any parent expression containing it,
        //     such as `sizeof(x) / sizeof(x[0]) + 1`.
        if !matches!(expr, Expr::Identifier(_, _)) && !Self::expr_contains_sizeof(expr) {
            if let Some(val) = self.lookup_sema_const(expr) {
                return Some(val);
            }
        }
        match expr {
            // Literals: delegate to shared evaluation
            Expr::IntLiteral(..) | Expr::LongLiteral(..) | Expr::LongLongLiteral(..)
            | Expr::UIntLiteral(..) | Expr::ULongLiteral(..) | Expr::ULongLongLiteral(..)
            | Expr::CharLiteral(..) | Expr::FloatLiteral(..)
            | Expr::FloatLiteralF32(..) | Expr::FloatLiteralLongDouble(..) => {
                shared_const_eval::eval_literal(expr)
            }
            Expr::UnaryOp(UnaryOp::Plus, inner, _) => {
                self.eval_const_expr(inner)
            }
            Expr::UnaryOp(UnaryOp::Neg, inner, _) => {
                let val = self.eval_const_expr(inner)?;
                let promoted = shared_const_eval::promote_sub_int(val, self.is_expr_unsigned_for_const(inner));
                const_arith::negate_const(promoted)
            }
            Expr::BinaryOp(op, lhs, rhs, _) => {
                let l = self.eval_const_expr(lhs);
                let r = self.eval_const_expr(rhs);
                if let (Some(l), Some(r)) = (l, r) {
                    // Use infer_expr_type (C semantic types) for proper usual arithmetic
                    // conversions. get_expr_type returns IR storage types (IntLiteral → I64)
                    // which loses 32-bit width info needed for correct folding of
                    // expressions like (1 << 31) / N.
                    let lhs_ty = self.infer_expr_type(lhs);
                    let rhs_ty = self.infer_expr_type(rhs);
                    let result = self.eval_const_binop(op, &l, &r, lhs_ty, rhs_ty);
                    if result.is_some() {
                        return result;
                    }
                }
                // For LogicalOr/LogicalAnd, handle cases where one operand is a
                // string literal or other always-nonzero expression (e.g., address-of).
                // String literals are always non-null pointers, so:
                //   "str" || x  =>  1
                //   x || "str"  =>  1
                //   "str" && x  =>  bool(x) if x is evaluable, else 1 if x is also nonzero
                //   0 && "str"  =>  0
                if *op == BinOp::LogicalOr {
                    let l_nonzero = l.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(lhs);
                    let r_nonzero = r.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(rhs);
                    if l_nonzero || r_nonzero {
                        return Some(IrConst::I64(1));
                    }
                    // Both are zero constants => result is 0
                    if l.as_ref().is_some_and(|v| !v.is_nonzero())
                        && r.as_ref().is_some_and(|v| !v.is_nonzero())
                    {
                        return Some(IrConst::I64(0));
                    }
                }
                if *op == BinOp::LogicalAnd {
                    // If either side is a known zero, result is 0
                    if l.as_ref().is_some_and(|v| !v.is_nonzero())
                        || r.as_ref().is_some_and(|v| !v.is_nonzero())
                    {
                        return Some(IrConst::I64(0));
                    }
                    // If both sides are known nonzero (including string literals), result is 1
                    let l_nonzero = l.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(lhs);
                    let r_nonzero = r.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(rhs);
                    if l_nonzero && r_nonzero {
                        return Some(IrConst::I64(1));
                    }
                }
                // For subtraction, try evaluating as pointer difference:
                // &arr[5] - &arr[0], (char*)&s.c - (char*)&s.a, etc.
                // Both operands may be global address expressions referring to the
                // same symbol; the result is the byte offset difference (possibly
                // divided by the pointed-to element size for typed pointer subtraction).
                if *op == BinOp::Sub {
                    return self.eval_const_ptr_diff(lhs, rhs);
                }
                None
            }
            Expr::UnaryOp(UnaryOp::BitNot, inner, _) => {
                let val = self.eval_const_expr(inner)?;
                let promoted = shared_const_eval::promote_sub_int(val, self.is_expr_unsigned_for_const(inner));
                const_arith::bitnot_const(promoted)
            }
            Expr::Cast(ref target_type, inner, _) => {
                self.eval_const_cast(target_type, inner)
            }
            Expr::Identifier(name, _) => {
                self.eval_const_identifier(name)
            }
            Expr::Sizeof(arg, _) => {
                let size = match arg.as_ref() {
                    SizeofArg::Type(ts) => self.sizeof_type(ts),
                    SizeofArg::Expr(e) => self.sizeof_expr(e),
                };
                Some(IrConst::I64(size as i64))
            }
            Expr::Alignof(ref ts, _) => {
                let align = self.alignof_type(ts);
                Some(IrConst::I64(align as i64))
            }
            Expr::AlignofExpr(ref inner_expr, _) => {
                let align = self.alignof_expr(inner_expr);
                Some(IrConst::I64(align as i64))
            }
            Expr::GnuAlignof(ref ts, _) => {
                let align = self.preferred_alignof_type(ts);
                Some(IrConst::I64(align as i64))
            }
            Expr::GnuAlignofExpr(ref inner_expr, _) => {
                let align = self.preferred_alignof_expr(inner_expr);
                Some(IrConst::I64(align as i64))
            }
            Expr::Conditional(cond, then_e, else_e, _) => {
                // Ternary in constant expr: evaluate condition and pick branch
                let cond_val = self.eval_const_expr(cond)?;
                if cond_val.is_nonzero() {
                    self.eval_const_expr(then_e)
                } else {
                    self.eval_const_expr(else_e)
                }
            }
            Expr::GnuConditional(cond, else_e, _) => {
                let cond_val = self.eval_const_expr(cond)?;
                if cond_val.is_nonzero() {
                    Some(cond_val) // condition value is used as result
                } else {
                    self.eval_const_expr(else_e)
                }
            }
            Expr::UnaryOp(UnaryOp::LogicalNot, inner, _) => {
                if let Some(val) = self.eval_const_expr(inner) {
                    Some(IrConst::I64(if val.is_nonzero() { 0 } else { 1 }))
                } else if Self::expr_is_always_nonzero(inner) {
                    // !("string_literal") is always 0
                    Some(IrConst::I64(0))
                } else {
                    None
                }
            }
            // Handle &((type*)0)->member pattern (offsetof)
            Expr::AddressOf(inner, _) => {
                self.eval_offsetof_pattern(inner)
            }
            Expr::BuiltinTypesCompatibleP(ref type1, ref type2, _) => {
                let result = self.eval_types_compatible(type1, type2);
                Some(IrConst::I64(result as i64))
            }
            // Handle compile-time builtin function calls in constant expressions.
            Expr::FunctionCall(func, args, _) => {
                if let Expr::Identifier(name, _) = func.as_ref() {
                    shared_const_eval::eval_builtin_call(
                        &*name, args, &|e| self.eval_const_expr(e),
                    )
                } else {
                    None
                }
            }
            Expr::CompoundLiteral(ref type_spec, ref init, _) => {
                self.eval_const_compound_literal(type_spec, init)
            }
            Expr::GenericSelection(ref controlling, ref associations, _) => {
                let selected = self.resolve_generic_selection_expr(controlling, associations)?;
                self.eval_const_expr(selected)
            }
            _ => None,
        }
    }

    /// Evaluate a cast expression at compile time.
    fn eval_const_cast(&self, target_type: &TypeSpecifier, inner: &Expr) -> Option<IrConst> {
        let target_ir_ty = self.type_spec_to_ir(target_type);
        let src_val = self.eval_const_expr(inner)?;

        // Handle float source types: use value-based conversion, not bit manipulation
        if let IrConst::LongDouble(fv, bytes) = &src_val {
            return IrConst::cast_long_double_to_target(*fv, bytes, target_ir_ty);
        }
        if let Some(fv) = src_val.to_f64() {
            if matches!(&src_val, IrConst::F32(_) | IrConst::F64(_)) {
                return IrConst::cast_float_to_target(fv, target_ir_ty);
            }
        }

        // Handle I128 source: use full 128-bit value to avoid truncation
        if let IrConst::I128(v128) = src_val {
            let src_ty = self.get_expr_type(inner);
            let src_unsigned = src_ty.is_unsigned();
            return Some(Self::cast_i128_to_ir_type(v128, target_ir_ty, src_unsigned));
        }

        // Integer source: use bit-based cast chain evaluation
        let (bits, src_signed) = self.eval_const_expr_as_bits(inner)?;

        // For 128-bit targets, sign/zero-extend based on source signedness
        if matches!(target_ir_ty, IrType::I128 | IrType::U128) {
            let src_ty = self.get_expr_type(inner);
            let v128 = if src_ty.is_unsigned() {
                bits as i128
            } else {
                (bits as i64) as i128
            };
            return Some(IrConst::I128(v128));
        }

        self.cast_bits_to_ir_const(bits, src_signed, target_ir_ty)
    }

    /// Convert raw bits to an IrConst of the given target type, truncating as needed.
    fn cast_bits_to_ir_const(&self, bits: u64, src_signed: bool, target_ir_ty: IrType) -> Option<IrConst> {
        let target_width = target_ir_ty.size() * 8;
        let truncated = if target_width >= 64 { bits } else { bits & ((1u64 << target_width) - 1) };
        let result = match target_ir_ty {
            IrType::I8 => IrConst::I8(truncated as i8),
            IrType::U8 => IrConst::I64(truncated as u8 as i64),
            IrType::I16 => IrConst::I16(truncated as i16),
            IrType::U16 => IrConst::I64(truncated as u16 as i64),
            IrType::I32 => IrConst::I32(truncated as i32),
            IrType::U32 => IrConst::I64(truncated as u32 as i64),
            IrType::I64 | IrType::U64 => IrConst::I64(truncated as i64),
            IrType::Ptr => IrConst::ptr_int(truncated as i64),
            IrType::I128 | IrType::U128 => unreachable!("handled above"),
            IrType::F32 => {
                let int_val = if src_signed { bits as i64 as f32 } else { bits as f32 };
                IrConst::F32(int_val)
            }
            IrType::F64 => {
                let int_val = if src_signed { bits as i64 as f64 } else { bits as f64 };
                IrConst::F64(int_val)
            }
            _ => return None,
        };
        Some(result)
    }

    fn eval_const_identifier(&self, name: &str) -> Option<IrConst> {
        let is_local = self.func_state.as_ref()
            .is_some_and(|fs| fs.locals.contains_key(name));
        if !is_local {
            if let Some(&val) = self.types.enum_constants.get(name) {
                return Some(IrConst::I64(val));
            }
        }
        if let Some(ref fs) = self.func_state {
            if let Some(&val) = fs.const_local_values.get(name) {
                return Some(IrConst::I64(val));
            }
        }
        None
    }

    /// Evaluate scalar compound literals in constant expressions.
    /// Returns None for multi-field aggregates (those go through struct init path).
    fn eval_const_compound_literal(&self, type_spec: &TypeSpecifier, init: &Initializer) -> Option<IrConst> {
        let cl_ctype = self.type_spec_to_ctype(type_spec);
        let is_multi_field_aggregate = match &cl_ctype {
            CType::Struct(key) | CType::Union(key) => {
                if let Some(layout) = self.types.borrow_struct_layouts().get(&**key) {
                    layout.fields.len() > 1
                        || layout.fields.iter().any(|f| {
                            matches!(f.ty, CType::Array(..) | CType::Struct(..) | CType::Union(..))
                        })
                } else {
                    false
                }
            }
            CType::Array(..) => true,
            _ => false,
        };
        if is_multi_field_aggregate {
            None
        } else {
            self.eval_const_initializer_scalar(init)
        }
    }

    /// Evaluate an initializer to a scalar constant for use in constant expressions.
    /// Handles both direct expressions and brace-wrapped lists (including nested ones
    /// like `{ { 42 } }` which occur when a struct compound literal has a single field).
    fn eval_const_initializer_scalar(&self, init: &Initializer) -> Option<IrConst> {
        match init {
            Initializer::Expr(expr) => self.eval_const_expr(expr),
            Initializer::List(items) => {
                // For a struct/union compound literal with one field,
                // the initializer list has one item whose value is the scalar.
                // Recurse into the first item.
                if let Some(first) = items.first() {
                    self.eval_const_initializer_scalar(&first.init)
                } else {
                    None
                }
            }
        }
    }

    /// Evaluate the offsetof pattern: &((type*)0)->member
    /// Also handles nested member access like &((type*)0)->data.x
    /// Returns Some(IrConst with offset) if the expression matches the pattern.
    pub(super) fn eval_offsetof_pattern(&self, expr: &Expr) -> Option<IrConst> {
        let (offset, _ty) = self.eval_offsetof_pattern_with_type(expr)?;
        Some(IrConst::ptr_int(offset as i64))
    }

    /// Evaluate an offsetof sub-expression, returning both the accumulated byte offset
    /// and the CType of the resulting expression (needed for chained member access).
    fn eval_offsetof_pattern_with_type(&self, expr: &Expr) -> Option<(usize, CType)> {
        match expr {
            Expr::PointerMemberAccess(base, field_name, _) => {
                // base should be (type*)0 - a cast of 0 to a pointer type
                let (type_spec, base_offset) = self.extract_null_pointer_cast_with_offset(base)?;
                let layout = self.get_struct_layout_for_type(&type_spec)?;
                let (field_offset, field_ty) = layout.field_offset(field_name, &*self.types.borrow_struct_layouts())?;
                Some((base_offset + field_offset, field_ty))
            }
            Expr::MemberAccess(base, field_name, _) => {
                // First try: base is *((type*)0) (deref pattern)
                if let Expr::Deref(inner, _) = base.as_ref() {
                    let (type_spec, base_offset) = self.extract_null_pointer_cast_with_offset(inner)?;
                    let layout = self.get_struct_layout_for_type(&type_spec)?;
                    let (field_offset, field_ty) = layout.field_offset(field_name, &*self.types.borrow_struct_layouts())?;
                    return Some((base_offset + field_offset, field_ty));
                }
                // Second try: base is itself an offsetof sub-expression (chained access)
                // e.g., ((type*)0)->data.x where base = ((type*)0)->data
                let (base_offset, base_type) = self.eval_offsetof_pattern_with_type(base)?;
                let struct_key = match &base_type {
                    CType::Struct(key) | CType::Union(key) => key.clone(),
                    _ => return None,
                };
                let layouts = self.types.borrow_struct_layouts();
                let layout = layouts.get(&*struct_key)?;
                let (field_offset, field_ty) = layout.field_offset(field_name, &*layouts)?;
                Some((base_offset + field_offset, field_ty))
            }
            Expr::ArraySubscript(base, index, _) => {
                // Handle &((type*)0)->member[index] pattern
                let (base_offset, base_type) = self.eval_offsetof_pattern_with_type(base)?;
                let idx_val = self.eval_const_expr(index)?;
                let idx = idx_val.to_i64()?;
                let elem_size = match &base_type {
                    CType::Array(elem, _) => self.resolve_ctype_size(elem),
                    _ => return None,
                };
                let elem_ty = match &base_type {
                    CType::Array(elem, _) => (**elem).clone(),
                    _ => return None,
                };
                Some(((base_offset as i64 + idx * elem_size as i64) as usize, elem_ty))
            }
            _ => None,
        }
    }

    /// Extract the struct type from a (type*)0 pattern, returning the base TypeSpecifier
    /// for the struct type and any accumulated offset from nested member access.
    fn extract_null_pointer_cast_with_offset(&self, expr: &Expr) -> Option<(TypeSpecifier, usize)> {
        match expr {
            Expr::Cast(ref type_spec, inner, _) => {
                // The type should be a Pointer to a struct
                if let TypeSpecifier::Pointer(inner_ts, _) = type_spec {
                    // Check that the inner expression is 0
                    if self.is_zero_expr(inner) {
                        return Some((*inner_ts.clone(), 0));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Check if an expression evaluates to 0 (integer literal 0 or cast of 0).
    fn is_zero_expr(&self, expr: &Expr) -> bool {
        const_arith::is_zero_expr(expr)
    }

    /// Evaluate a constant expression, returning raw u64 bits and signedness.
    /// This preserves signedness information through cast chains.
    /// Signedness determines how the value is widened in the next cast.
    fn eval_const_expr_as_bits(&self, expr: &Expr) -> Option<(u64, bool)> {
        match expr {
            Expr::Cast(ref target_type, inner, _) => {
                let (bits, _src_signed) = self.eval_const_expr_as_bits(inner)?;
                let target_ir_ty = self.type_spec_to_ir(target_type);
                let target_width = target_ir_ty.size() * 8;
                let target_signed = matches!(target_ir_ty, IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64);
                Some(const_arith::truncate_and_extend_bits(bits, target_width, target_signed))
            }
            _ => {
                let val = self.eval_const_expr(expr)?;
                Some(shared_const_eval::irconst_to_bits(&val))
            }
        }
    }

    /// Cast a full 128-bit integer value to an IrType without going through u64 truncation.
    ///
    /// For targets <= 64 bits, extracts the lower bits. For 128-bit targets, preserves the
    /// full value. For float targets, uses value-based conversion from the full i128.
    fn cast_i128_to_ir_type(v128: i128, target: IrType, src_unsigned: bool) -> IrConst {
        let bits_lo = v128 as u64; // lower 64 bits
        match target {
            IrType::I8 => IrConst::I8(bits_lo as i8),
            IrType::U8 => IrConst::I64(bits_lo as u8 as i64),
            IrType::I16 => IrConst::I16(bits_lo as i16),
            IrType::U16 => IrConst::I64(bits_lo as u16 as i64),
            IrType::I32 => IrConst::I32(bits_lo as i32),
            IrType::U32 => IrConst::I64(bits_lo as u32 as i64),
            IrType::I64 | IrType::U64 => IrConst::I64(bits_lo as i64),
            IrType::Ptr => IrConst::ptr_int(bits_lo as i64),
            IrType::I128 | IrType::U128 => IrConst::I128(v128),
            IrType::F32 => {
                // int-to-float: signedness comes from the source type
                let fv = if src_unsigned { (v128 as u128) as f32 } else { v128 as f32 };
                IrConst::F32(fv)
            }
            IrType::F64 => {
                let fv = if src_unsigned { (v128 as u128) as f64 } else { v128 as f64 };
                IrConst::F64(fv)
            }
            IrType::F128 => {
                // Use direct integer-to-x87 conversion to preserve full 64-bit
                // mantissa precision (x87 has 64-bit mantissa, unlike f64's 52-bit).
                if src_unsigned {
                    IrConst::long_double_from_u128(v128 as u128)
                } else {
                    IrConst::long_double_from_i128(v128)
                }
            }
            _ => IrConst::I128(v128), // fallback: preserve value
        }
    }

    /// Evaluate a constant binary operation.
    /// Delegates to `shared_const_eval::eval_binop_with_types` which implements
    /// C's usual arithmetic conversions (C11 6.3.1.8).
    fn eval_const_binop(&self, op: &BinOp, lhs: &IrConst, rhs: &IrConst, lhs_ty: IrType, rhs_ty: IrType) -> Option<IrConst> {
        shared_const_eval::eval_binop_with_types(
            op, lhs, rhs,
            lhs_ty.size().max(4), lhs_ty.is_unsigned(),
            rhs_ty.size().max(4), rhs_ty.is_unsigned(),
        )
    }

    /// Check if an expression has an unsigned type for constant evaluation.
    fn is_expr_unsigned_for_const(&self, expr: &Expr) -> bool {
        if let Expr::Cast(ref target_type, _, _) = expr {
            let ty = self.type_spec_to_ir(target_type);
            return ty.is_unsigned();
        }
        let ty = self.infer_expr_type(expr);
        ty.is_unsigned()
    }

    /// Try to constant-fold a binary operation from its parts.
    /// Used by lower_binary_op to avoid generating IR for constant expressions,
    /// ensuring correct C type semantics (especially 32-bit vs 64-bit width).
    pub(super) fn eval_const_expr_from_parts(&self, op: &BinOp, lhs: &Expr, rhs: &Expr) -> Option<IrConst> {
        let l = self.eval_const_expr(lhs)?;
        let r = self.eval_const_expr(rhs)?;
        let lhs_ty = self.infer_expr_type(lhs);
        let rhs_ty = self.infer_expr_type(rhs);
        let result = self.eval_const_binop(op, &l, &r, lhs_ty, rhs_ty)?;
        // Convert to I64 for IR operand compatibility (IR uses I64 for all int operations).
        Some(match result {
            IrConst::I32(v) => IrConst::I64(v as i64),
            IrConst::I8(v) => IrConst::I64(v as i64),
            IrConst::I16(v) => IrConst::I64(v as i64),
            other => other,
        })
    }

    /// Evaluate a constant expression and return as usize (for array index designators).
    pub(super) fn eval_const_expr_for_designator(&self, expr: &Expr) -> Option<usize> {
        self.eval_const_expr(expr).and_then(|v| v.to_usize())
    }

    /// Evaluate a compile-time pointer difference expression.
    /// Handles patterns like:
    ///   &arr[5] - &arr[0]              -> 5  (typed pointer subtraction)
    ///   (char*)&arr[5] - (char*)&arr[0] -> 20 (byte-level subtraction)
    ///   (long)&arr[5] - (long)&arr[0]  -> 20 (cast-to-integer subtraction)
    ///   (char*)&s.c - (char*)&s.a      -> 8  (struct member offset diff)
    ///
    /// Both operands must resolve to addresses within the same global symbol.
    /// The byte offsets are subtracted; for typed pointer subtraction the result
    /// is divided by the pointed-to element size.
    fn eval_const_ptr_diff(&self, lhs: &Expr, rhs: &Expr) -> Option<IrConst> {
        let lhs_addr = self.eval_global_addr_expr(lhs)?;
        let rhs_addr = self.eval_global_addr_expr(rhs)?;

        // Extract (symbol_name, byte_offset) from each side
        let (lhs_name, lhs_offset) = match &lhs_addr {
            GlobalInit::GlobalAddr(name) => (&**name, 0i64),
            GlobalInit::GlobalAddrOffset(name, off) => (&**name, *off),
            _ => return None,
        };
        let (rhs_name, rhs_offset) = match &rhs_addr {
            GlobalInit::GlobalAddr(name) => (&**name, 0i64),
            GlobalInit::GlobalAddrOffset(name, off) => (&**name, *off),
            _ => return None,
        };

        // Both must refer to the same global symbol
        if lhs_name != rhs_name {
            return None;
        }

        let byte_diff = lhs_offset - rhs_offset;

        // Determine if this is typed pointer subtraction (result in elements)
        // or byte/integer subtraction (result in bytes).
        // For typed pointer subtraction (ptr - ptr where both are non-void,
        // non-char pointers), divide by the element size.
        let result = if self.expr_is_pointer(lhs) && self.expr_is_pointer(rhs) {
            let elem_size = self.get_pointer_elem_size_from_expr(lhs) as i64;
            if elem_size > 1 {
                byte_diff / elem_size
            } else {
                byte_diff
            }
        } else {
            // Cast-to-integer subtraction: (long)&x - (long)&y, or
            // char*/void* subtraction: result is in bytes
            byte_diff
        };

        Some(IrConst::I64(result))
    }

    /// Convert an IrConst to i64. Delegates to IrConst::to_i64().
    pub(super) fn const_to_i64(&self, c: &IrConst) -> Option<i64> {
        c.to_i64()
    }

    /// Coerce a constant to the target type, using the source expression's type for signedness.
    pub(super) fn coerce_const_to_type_with_src(&self, val: IrConst, target_ty: IrType, src_ty: IrType) -> IrConst {
        val.coerce_to_with_src(target_ty, Some(src_ty))
    }

    /// Collect array dimensions from nested Array type specifiers.
    /// Extract an integer value from any integer literal expression (Int, UInt, Long, ULong).
    /// Used for array sizes and other compile-time integer expressions.
    pub(super) fn expr_as_array_size(&self, expr: &Expr) -> Option<i64> {
        // Try simple literals first (fast path)
        match expr {
            Expr::IntLiteral(n, _) | Expr::LongLiteral(n, _) | Expr::LongLongLiteral(n, _) => return Some(*n),
            Expr::UIntLiteral(n, _) | Expr::ULongLiteral(n, _) | Expr::ULongLongLiteral(n, _) => return Some(*n as i64),
            _ => {}
        }
        // Fall back to full constant expression evaluation (handles sizeof, arithmetic, etc.)
        if let Some(val) = self.eval_const_expr(expr) {
            return self.const_to_i64(&val);
        }
        None
    }

    /// Check if an expression is always non-zero at compile time, even if we
    /// can't compute its exact numeric value.
    ///
    /// String literals, for example, are always non-null pointers, so they are
    /// always truthy in boolean context. This is needed for static initializers
    /// like `static const int NEED_OPTIONS = 0 || "Lcmwl" || 0;` (used by toybox).
    fn expr_is_always_nonzero(expr: &Expr) -> bool {
        match expr {
            Expr::StringLiteral(..)
            | Expr::WideStringLiteral(..)
            | Expr::Char16StringLiteral(..) => true,
            // A cast of a string literal is also always nonzero
            // (e.g., (int)"hello" in constant context)
            Expr::Cast(_, inner, _) => Self::expr_is_always_nonzero(inner),
            _ => false,
        }
    }
}

