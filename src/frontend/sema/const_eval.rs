//! Constant expression evaluation for semantic analysis.
//!
//! This module provides compile-time constant evaluation using only sema-available
//! state (TypeContext, SymbolTable, ExprTypeChecker). It returns `IrConst` values
//! matching the lowerer's richer result type, enabling:
//! - Float literal evaluation (F32, F64, LongDouble)
//! - Proper cast chain evaluation with bit-width tracking
//! - Binary operations with type-aware signedness semantics
//! - sizeof/alignof via sema's type resolution
//!
//! Shared pure-evaluation logic (literal eval, builtin folding, sub-int promotion,
//! binary operation arithmetic) lives in `common::const_eval` and `common::const_arith`,
//! called by both this module and `ir::lowering::const_eval`.
//!
//! The lowerer's const_eval.rs handles additional lowering-specific cases:
//! - Global address expressions (&x, func, arr)
//! - func_state const local values
//! - Pointer arithmetic on global addresses
//!   These remain in the lowerer since they require IR-level state.

use std::rc::Rc;
use crate::common::types::CType;
use crate::common::types::AddressSpace;
use crate::common::const_arith;
use crate::common::const_eval as shared_const_eval;
use crate::ir::reexports::IrConst;
use crate::frontend::parser::ast::{
    BinOp,
    DerivedDeclarator,
    Expr,
    ExprId,
    SizeofArg,
    StructFieldDecl,
    TypeSpecifier,
    UnaryOp,
};
use super::type_context::TypeContext;
use super::analysis::FunctionInfo;
use crate::common::symbol_table::SymbolTable;
use crate::common::fx_hash::FxHashMap;

/// Map from AST expression node identity to its pre-computed compile-time constant.
///
/// Keyed by [`ExprId`], the same type-safe identity key used by `ExprTypeMap`.
/// During sema's AST walk, expressions that can be evaluated at compile time
/// have their `IrConst` values stored here. The lowerer consults this map as
/// an O(1) fast path before falling back to its own `eval_const_expr`.
pub type ConstMap = FxHashMap<ExprId, IrConst>;

/// Constant expression evaluator using only sema-available state.
///
/// This evaluator is created fresh for each expression evaluation, borrowing
/// the sema state it needs. It is stateless and does not modify anything.
///
/// When `const_values` and `expr_types` caches are provided, previously-computed
/// results are returned in O(1) instead of re-traversing the AST. This prevents
/// exponential blowup on deeply nested expressions.
pub struct SemaConstEval<'a> {
    /// Type context for typedef, enum, and struct layout resolution.
    pub types: &'a TypeContext,
    /// Symbol table for variable type lookup.
    pub symbols: &'a SymbolTable,
    /// Function signatures for return type resolution in sizeof(expr).
    pub functions: &'a FxHashMap<Rc<str>, FunctionInfo>,
    /// Pre-computed constant values from bottom-up sema walk (memoization cache).
    pub const_values: Option<&'a FxHashMap<ExprId, IrConst>>,
    /// Pre-computed expression types from bottom-up sema walk.
    pub expr_types: Option<&'a FxHashMap<ExprId, CType>>,
}

impl<'a> SemaConstEval<'a> {
    /// Try to evaluate a constant expression at compile time.
    ///
    /// Returns `Some(IrConst)` for expressions that can be fully evaluated using
    /// sema state. Returns `None` for expressions that require lowering state
    /// (global addresses, runtime values, etc.).
    pub fn eval_const_expr(&self, expr: &Expr) -> Option<IrConst> {
        // Memoization: if this expression's constant value was already computed
        // during the bottom-up sema walk, return it in O(1).
        if let Some(cache) = self.const_values {
            if let Some(cached) = cache.get(&expr.id()) {
                return Some(*cached);
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
                let promoted = shared_const_eval::promote_sub_int(val, self.is_expr_unsigned(inner));
                const_arith::negate_const(promoted)
            }
            Expr::UnaryOp(UnaryOp::BitNot, inner, _) => {
                let val = self.eval_const_expr(inner)?;
                let is_unsigned = self.is_expr_unsigned(inner);
                let promoted = shared_const_eval::promote_sub_int(val, is_unsigned);
                let result = const_arith::bitnot_const(promoted)?;
                // For unsigned int operands (stored as I64 to preserve unsigned range),
                // the bitwise NOT must be truncated to 32 bits. Without this, ~0u
                // produces I64(-1) (all 64 bits set) instead of I64(0xFFFFFFFF).
                // Check the promoted type's actual width via the inner expression type.
                let inner_ctype = self.lookup_expr_type(inner)
                    .or_else(|| self.infer_expr_ctype(inner));
                let is_32bit_type = inner_ctype.as_ref().map_or(
                    matches!(result, IrConst::I32(_)),
                    |ct| {
                        let size = self.ctype_size(ct);
                        size <= 4
                    },
                );
                if is_32bit_type && is_unsigned {
                    // Truncate to 32-bit unsigned: mask to 0xFFFFFFFF and store as I64
                    if let Some(v) = result.to_i64() {
                        Some(IrConst::I64(v as u32 as i64))
                    } else {
                        Some(result)
                    }
                } else {
                    Some(result)
                }
            }

            // Logical NOT
            Expr::UnaryOp(UnaryOp::LogicalNot, inner, _) => {
                if let Some(val) = self.eval_const_expr(inner) {
                    Some(IrConst::I64(if val.is_nonzero() { 0 } else { 1 }))
                } else if Self::expr_is_always_nonzero(inner) {
                    Some(IrConst::I64(0))
                } else {
                    None
                }
            }

            // Binary operations
            Expr::BinaryOp(op, lhs, rhs, _) => {
                let l = self.eval_const_expr(lhs);
                let r = self.eval_const_expr(rhs);
                if let (Some(ref lv), Some(ref rv)) = (&l, &r) {
                    // Derive operand CTypes for signedness/width determination.
                    let lhs_ctype = self.lookup_expr_type(lhs)
                        .or_else(|| self.infer_expr_ctype(lhs))
                        .or_else(|| Self::ctype_from_ir_const(lv));
                    let rhs_ctype = self.lookup_expr_type(rhs)
                        .or_else(|| self.infer_expr_ctype(rhs))
                        .or_else(|| Self::ctype_from_ir_const(rv));
                    let lhs_size = lhs_ctype.as_ref().map_or(4, |ct| self.ctype_size(ct).max(4));
                    let lhs_unsigned = lhs_ctype.as_ref().is_some_and(|ct| ct.is_unsigned());
                    let rhs_size = rhs_ctype.as_ref().map_or(4, |ct| self.ctype_size(ct).max(4));
                    let rhs_unsigned = rhs_ctype.as_ref().is_some_and(|ct| ct.is_unsigned());
                    let result = shared_const_eval::eval_binop_with_types(
                        op, lv, rv, lhs_size, lhs_unsigned, rhs_size, rhs_unsigned,
                    );
                    if result.is_some() {
                        return result;
                    }
                }
                // For LogicalOr/LogicalAnd, handle string literals and other
                // always-nonzero expressions that can't be folded to numeric values.
                if *op == BinOp::LogicalOr {
                    let l_nonzero = l.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(lhs);
                    let r_nonzero = r.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(rhs);
                    if l_nonzero || r_nonzero {
                        return Some(IrConst::I64(1));
                    }
                    if l.as_ref().is_some_and(|v| !v.is_nonzero())
                        && r.as_ref().is_some_and(|v| !v.is_nonzero())
                    {
                        return Some(IrConst::I64(0));
                    }
                }
                if *op == BinOp::LogicalAnd {
                    if l.as_ref().is_some_and(|v| !v.is_nonzero())
                        || r.as_ref().is_some_and(|v| !v.is_nonzero())
                    {
                        return Some(IrConst::I64(0));
                    }
                    let l_nonzero = l.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(lhs);
                    let r_nonzero = r.as_ref().is_some_and(|v| v.is_nonzero())
                        || Self::expr_is_always_nonzero(rhs);
                    if l_nonzero && r_nonzero {
                        return Some(IrConst::I64(1));
                    }
                }
                None
            }

            // Cast expressions with proper bit-width tracking
            Expr::Cast(ref target_type, inner, _) => {
                let target_ctype = self.type_spec_to_ctype(target_type);
                let src_val = self.eval_const_expr(inner)?;

                // Handle float source types: use value-based conversion
                // For LongDouble, use full x87 precision for integer conversions
                if let IrConst::LongDouble(fv, bytes) = &src_val {
                    return self.cast_long_double_to_ctype(*fv, bytes, &target_ctype);
                }
                if let Some(fv) = src_val.to_f64() {
                    if matches!(&src_val, IrConst::F32(_) | IrConst::F64(_)) {
                        return self.cast_float_to_ctype(fv, &target_ctype);
                    }
                }

                // Handle I128 source: use full 128-bit value to avoid truncation
                // through the u64-based eval_const_expr_as_bits path
                if let IrConst::I128(v128) = src_val {
                    let target_size = self.ctype_size(&target_ctype);
                    // Determine source signedness for int-to-float conversions
                    let src_unsigned = self.lookup_expr_type(inner)
                        .is_some_and(|ct| ct.is_unsigned());
                    return self.cast_i128_to_ctype(v128, &target_ctype, target_size, src_unsigned);
                }

                // Integer source: use bit-based cast chain evaluation
                let (bits, _src_signed) = self.eval_const_expr_as_bits(inner)?;
                let target_size = self.ctype_size(&target_ctype);
                let target_width = target_size * 8;
                let target_signed = !target_ctype.is_unsigned() && !target_ctype.is_pointer_like();

                // For 128-bit targets, we need to sign-extend or zero-extend from 64 bits
                // based on the *source* expression's signedness, not the target's.
                // E.g., (unsigned __int128)(-1) should be all-ones (sign-extend from signed source),
                // but (unsigned __int128)(0xFFFFFFFFFFFFFFFFULL) should be 0x0000...FFFF (zero-extend).
                if target_size == 16 && !matches!(&target_ctype, CType::LongDouble) {
                    // Determine source signedness: try type map first, then infer, default signed
                    let src_signed = self.lookup_expr_type(inner)
                        .or_else(|| self.infer_expr_ctype(inner))
                        .is_none_or(|ct| !ct.is_unsigned());
                    let v128 = if src_signed {
                        // Sign-extend: u64 -> i64 (reinterpret) -> i128 (sign-extend)
                        (bits as i64) as i128
                    } else {
                        // Zero-extend: u64 -> i128
                        bits as i128
                    };
                    return Some(IrConst::I128(v128));
                }

                // Truncate to target width
                let truncated = if target_width >= 64 {
                    bits
                } else if target_width == 0 {
                    return None; // void cast
                } else {
                    bits & ((1u64 << target_width) - 1)
                };

                // For int-to-float conversions, we need the *source* signedness
                // (not target) to correctly interpret the bit pattern. E.g.,
                // (double)(unsigned long)0xFFFFFFFFFFFFFFFF should produce
                // 18446744073709551615.0, not -1.0.
                if matches!(target_ctype, CType::Float | CType::Double | CType::LongDouble) {
                    let src_signed = self.lookup_expr_type(inner)
                        .or_else(|| self.infer_expr_ctype(inner))
                        .is_none_or(|ct| !ct.is_unsigned());
                    return self.bits_to_irconst(truncated, &target_ctype, src_signed);
                }

                // Convert to IrConst based on target CType
                self.bits_to_irconst(truncated, &target_ctype, target_signed)
            }

            // Identifier: enum constants
            Expr::Identifier(name, _) => {
                if let Some(&val) = self.types.enum_constants.get(name) {
                    return Some(IrConst::I64(val));
                }
                None
            }

            // sizeof
            Expr::Sizeof(arg, _) => {
                let size = match arg.as_ref() {
                    SizeofArg::Type(ts) => self.sizeof_type_spec(ts),
                    SizeofArg::Expr(e) => self.sizeof_expr(e),
                };
                size.map(|s| IrConst::I64(s as i64))
            }

            // _Alignof - C11 standard, returns minimum ABI alignment
            Expr::Alignof(ref ts, _) => {
                let align = self.alignof_type_spec(ts);
                Some(IrConst::I64(align as i64))
            }

            // __alignof(type) / __alignof__(type) - GCC extension, returns preferred alignment.
            // On i686: __alignof__(long long) == 8, _Alignof(long long) == 4.
            Expr::GnuAlignof(ref ts, _) => {
                let align = self.preferred_alignof_type_spec(ts);
                Some(IrConst::I64(align as i64))
            }

            // __alignof__(expr) - GCC extension: alignment of expression's type.
            // Per C11 6.2.8p3, if the expression names a variable declared with
            // _Alignas or __attribute__((aligned(N))), the result reflects the
            // declared alignment (max of natural and explicit).
            Expr::AlignofExpr(ref inner_expr, _) => {
                // Check for explicit alignment on a variable identifier
                if let Expr::Identifier(name, _) = inner_expr.as_ref() {
                    if let Some(sym) = self.symbols.lookup(name) {
                        if let Some(explicit_align) = sym.explicit_alignment {
                            let natural = sym.ty.align_ctx(&*self.types.borrow_struct_layouts());
                            return Some(IrConst::I64(natural.max(explicit_align) as i64));
                        }
                    }
                }
                let ctype = self.infer_expr_ctype(inner_expr)?;
                let align = ctype.align_ctx(&*self.types.borrow_struct_layouts());
                Some(IrConst::I64(align as i64))
            }

            // __alignof__(expr) via GnuAlignof path - returns preferred alignment
            Expr::GnuAlignofExpr(ref inner_expr, _) => {
                if let Expr::Identifier(name, _) = inner_expr.as_ref() {
                    if let Some(sym) = self.symbols.lookup(name) {
                        if let Some(explicit_align) = sym.explicit_alignment {
                            let natural = sym.ty.preferred_align_ctx(&*self.types.borrow_struct_layouts());
                            return Some(IrConst::I64(natural.max(explicit_align) as i64));
                        }
                    }
                }
                let ctype = self.infer_expr_ctype(inner_expr)?;
                let align = ctype.preferred_align_ctx(&*self.types.borrow_struct_layouts());
                Some(IrConst::I64(align as i64))
            }

            // Ternary conditional
            Expr::Conditional(cond, then_e, else_e, _) => {
                let cond_val = self.eval_const_expr(cond)?;
                if cond_val.is_nonzero() {
                    self.eval_const_expr(then_e)
                } else {
                    self.eval_const_expr(else_e)
                }
            }

            // GNU conditional (a ?: b)
            Expr::GnuConditional(cond, else_e, _) => {
                let cond_val = self.eval_const_expr(cond)?;
                if cond_val.is_nonzero() {
                    Some(cond_val)
                } else {
                    self.eval_const_expr(else_e)
                }
            }

            // __builtin_types_compatible_p
            Expr::BuiltinTypesCompatibleP(ref type1, ref type2, _) => {
                let ctype1 = self.type_spec_to_ctype(type1);
                let ctype2 = self.type_spec_to_ctype(type2);
                let compatible = self.ctypes_compatible(&ctype1, &ctype2);
                Some(IrConst::I64(compatible as i64))
            }

            // AddressOf for offsetof patterns: &((type*)0)->member
            Expr::AddressOf(inner, _) => {
                self.eval_offsetof_pattern(inner)
            }

            // Handle compile-time builtin function calls in constant expressions.
            Expr::FunctionCall(func, args, _) => {
                if let Expr::Identifier(name, _) = func.as_ref() {
                    shared_const_eval::eval_builtin_call(
                        &**name, args, &|e| self.eval_const_expr(e),
                    )
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Evaluate a constant expression, returning raw u64 bits and signedness.
    /// Preserves signedness through cast chains for proper widening.
    fn eval_const_expr_as_bits(&self, expr: &Expr) -> Option<(u64, bool)> {
        match expr {
            Expr::Cast(ref target_type, inner, _) => {
                let (bits, _src_signed) = self.eval_const_expr_as_bits(inner)?;
                let target_ctype = self.type_spec_to_ctype(target_type);
                let target_width = self.ctype_size(&target_ctype) * 8;
                let target_signed = !target_ctype.is_unsigned() && !target_ctype.is_pointer_like();
                Some(const_arith::truncate_and_extend_bits(bits, target_width, target_signed))
            }
            _ => {
                let val = self.eval_const_expr(expr)?;
                Some(shared_const_eval::irconst_to_bits(&val))
            }
        }
    }

    /// Evaluate the offsetof pattern: &((type*)0)->member
    /// Also handles nested member access like &((type*)0)->data.x
    fn eval_offsetof_pattern(&self, expr: &Expr) -> Option<IrConst> {
        let (offset, _ty) = self.eval_offsetof_pattern_with_type(expr)?;
        Some(IrConst::I64(offset as i64))
    }

    /// Evaluate an offsetof sub-expression, returning both the accumulated byte offset
    /// and the CType of the resulting expression (needed for chained member access).
    fn eval_offsetof_pattern_with_type(&self, expr: &Expr) -> Option<(usize, CType)> {
        match expr {
            Expr::PointerMemberAccess(base, field_name, _) => {
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
                let (base_offset, base_type) = self.eval_offsetof_pattern_with_type(base)?;
                let idx_val = self.eval_const_expr(index)?;
                let idx = idx_val.to_i64()?;
                let elem_size = match &base_type {
                    CType::Array(elem, _) => elem.size_ctx(&*self.types.borrow_struct_layouts()),
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

    /// Extract the struct type from a (type*)0 pattern.
    fn extract_null_pointer_cast_with_offset(&self, expr: &Expr) -> Option<(TypeSpecifier, usize)> {
        match expr {
            Expr::Cast(ref type_spec, inner, _) => {
                if let TypeSpecifier::Pointer(inner_ts, _) = type_spec {
                    if const_arith::is_zero_expr(inner) {
                        return Some((*inner_ts.clone(), 0));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Get the struct layout for a type specifier.
    fn get_struct_layout_for_type(&self, type_spec: &TypeSpecifier) -> Option<crate::common::types::RcLayout> {
        let ctype = self.type_spec_to_ctype(type_spec);
        match &ctype {
            CType::Struct(key) | CType::Union(key) => {
                self.types.borrow_struct_layouts().get(&**key).cloned()
            }
            _ => None,
        }
    }

    /// Check if two CTypes are compatible (for __builtin_types_compatible_p).
    fn ctypes_compatible(&self, t1: &CType, t2: &CType) -> bool {
        // Strip qualifiers (CType doesn't carry them) and compare
        match (t1, t2) {
            (CType::Pointer(a, _), CType::Pointer(b, _)) => self.ctypes_compatible(a, b),
            (CType::Array(a, _), CType::Array(b, _)) => self.ctypes_compatible(a, b),
            _ => t1 == t2,
        }
    }

    // === Type helper methods ===

    /// Look up the pre-annotated CType for an expression from sema's expr_types map.
    /// This is O(1) and preserves signedness information (e.g. unsigned int from a cast).
    fn lookup_expr_type(&self, expr: &Expr) -> Option<CType> {
        self.expr_types.and_then(|m| m.get(&expr.id()).cloned())
    }

    /// Derive a CType from an IrConst value.
    /// This is O(1) and avoids the potentially exponential infer_expr_ctype()
    /// when we already have the evaluated constant value.
    /// Note: This loses signedness info -- IrConst::I64 always maps to signed Long.
    fn ctype_from_ir_const(c: &IrConst) -> Option<CType> {
        match c {
            IrConst::I8(_) => Some(CType::Char),
            IrConst::I16(_) => Some(CType::Short),
            IrConst::I32(_) => Some(CType::Int),
            IrConst::I64(_) => Some(CType::Long),
            IrConst::I128(_) => Some(CType::Int128),
            IrConst::F32(_) => Some(CType::Float),
            IrConst::F64(_) => Some(CType::Double),
            IrConst::LongDouble(..) => Some(CType::LongDouble),
            IrConst::Zero => Some(CType::Int),
        }
    }

    /// Convert a TypeSpecifier to CType using sema's type resolution.
    fn type_spec_to_ctype(&self, spec: &TypeSpecifier) -> CType {
        // Handle typeof(expr) which requires expression type inference - the
        // standalone ctype_from_type_spec function can't handle this because it
        // lacks access to the symbol table and expression type checker.
        if let TypeSpecifier::Typeof(expr) = spec {
            return self.infer_expr_ctype(expr).unwrap_or(CType::Int);
        }
        // Delegate to a simple inline conversion for all other cases.
        ctype_from_type_spec(spec, self.types)
    }

    /// Infer the CType of an expression using ExprTypeChecker.
    fn infer_expr_ctype(&self, expr: &Expr) -> Option<CType> {
        let checker = super::type_checker::ExprTypeChecker {
            symbols: self.symbols,
            types: self.types,
            functions: self.functions,
            expr_types: self.expr_types,
        };
        checker.infer_expr_ctype(expr)
    }

    /// Get the size in bytes for a CType.
    fn ctype_size(&self, ctype: &CType) -> usize {
        ctype.size_ctx(&*self.types.borrow_struct_layouts())
    }

    /// Check if an expression has an unsigned type, using the expr_types cache
    /// or falling back to type inference from the AST structure.
    fn is_expr_unsigned(&self, expr: &Expr) -> bool {
        // First try the pre-annotated type from sema
        if let Some(ctype) = self.lookup_expr_type(expr) {
            return ctype.is_unsigned();
        }
        // For cast expressions, check the target type directly
        if let Expr::Cast(ref target_type, _, _) = expr {
            let ctype = self.type_spec_to_ctype(target_type);
            return ctype.is_unsigned();
        }
        // Fall back to type inference
        if let Some(ctype) = self.infer_expr_ctype(expr) {
            return ctype.is_unsigned();
        }
        false
    }

    /// Cast a long double value (with f128 bytes) to a target CType.
    /// Uses full precision for integer conversions.
    fn cast_long_double_to_ctype(&self, fv: f64, bytes: &[u8; 16], target: &CType) -> Option<IrConst> {
        use crate::common::long_double::{f128_bytes_to_i64, f128_bytes_to_u64, f128_bytes_to_i128, f128_bytes_to_u128};
        Some(match target {
            CType::Float => IrConst::F32(fv as f32),
            CType::Double => IrConst::F64(fv),
            CType::LongDouble => IrConst::long_double_with_bytes(fv, *bytes),
            CType::Char => IrConst::I8(f128_bytes_to_i64(bytes)? as i8),
            CType::UChar => IrConst::I32(f128_bytes_to_u64(bytes)? as u8 as i32),
            CType::Short => IrConst::I16(f128_bytes_to_i64(bytes)? as i16),
            CType::UShort => IrConst::I32(f128_bytes_to_u64(bytes)? as u16 as i32),
            CType::Int => IrConst::I32(f128_bytes_to_i64(bytes)? as i32),
            CType::UInt => IrConst::I32(f128_bytes_to_u64(bytes)? as u32 as i32),
            CType::Long | CType::LongLong => IrConst::I64(f128_bytes_to_i64(bytes)?),
            CType::ULong | CType::ULongLong => IrConst::I64(f128_bytes_to_u64(bytes)? as i64),
            CType::Bool => IrConst::I8(if fv != 0.0 { 1 } else { 0 }),
            CType::Int128 => IrConst::I128(f128_bytes_to_i128(bytes)?),
            CType::UInt128 => IrConst::I128(f128_bytes_to_u128(bytes)? as i128),
            _ => return None,
        })
    }

    /// Cast a float value to a target CType.
    fn cast_float_to_ctype(&self, fv: f64, target: &CType) -> Option<IrConst> {
        Some(match target {
            CType::Float => IrConst::F32(fv as f32),
            CType::Double => IrConst::F64(fv),
            CType::LongDouble => IrConst::long_double(fv),
            CType::Char => IrConst::I8(fv as i8),
            CType::UChar => IrConst::I32(fv as u8 as i32),
            CType::Short => IrConst::I16(fv as i16),
            CType::UShort => IrConst::I32(fv as u16 as i32),
            CType::Int => IrConst::I32(fv as i32),
            CType::UInt => IrConst::I32(fv as u32 as i32),
            CType::Long | CType::LongLong => IrConst::I64(fv as i64),
            CType::ULong | CType::ULongLong => IrConst::I64(fv as u64 as i64),
            CType::Bool => IrConst::I8(if fv != 0.0 { 1 } else { 0 }),
            _ => return None,
        })
    }

    /// Convert raw bits to an IrConst based on target CType.
    ///
    /// For unsigned sub-int types (UChar, UShort, Bool), we store the value
    /// as IrConst::I32 with zero-extension rather than I8/I16, because IrConst
    /// only has signed I8/I16 variants. Without this, `(unsigned short)(-1)` would
    /// be stored as `I16(-1)`, and `to_i64()` would sign-extend to -1 instead of
    /// 65535 -- causing `(unsigned short)(-1) << 1` to evaluate as -2 instead of
    /// 131070. This matches C's integer promotion (sub-int → int).
    fn bits_to_irconst(&self, bits: u64, target: &CType, target_signed: bool) -> Option<IrConst> {
        let size = self.ctype_size(target);
        Some(match size {
            1 => {
                if !target_signed {
                    // Unsigned char/bool: store as I32 to preserve unsigned value
                    // through integer promotion (e.g., (unsigned char)255 → I32(255), not I8(-1))
                    IrConst::I32(bits as u8 as i32)
                } else {
                    IrConst::I8(bits as i8)
                }
            }
            2 => {
                if !target_signed {
                    // Unsigned short: store as I32 to preserve unsigned value
                    // through integer promotion (e.g., (unsigned short)65535 → I32(65535), not I16(-1))
                    IrConst::I32(bits as u16 as i32)
                } else {
                    IrConst::I16(bits as i16)
                }
            }
            4 => {
                if matches!(target, CType::Float) {
                    let int_val = if target_signed { bits as i64 as f32 } else { bits as f32 };
                    IrConst::F32(int_val)
                } else if !target_signed {
                    // Unsigned 32-bit: use I64 to preserve unsigned semantics.
                    // IrConst::I32 is signed, so values >= 0x80000000 would sign-extend
                    // incorrectly when converted to i64 for 64-bit comparisons.
                    IrConst::I64(bits as u32 as i64)
                } else {
                    IrConst::I32(bits as i32)
                }
            }
            8 => {
                if matches!(target, CType::Double) {
                    let int_val = if target_signed { bits as i64 as f64 } else { bits as f64 };
                    IrConst::F64(int_val)
                } else {
                    IrConst::I64(bits as i64)
                }
            }
            16 => {
                if matches!(target, CType::LongDouble) {
                    // Use direct integer-to-x87 conversion to preserve full 64-bit
                    // precision (x87 has 64-bit mantissa, unlike f64's 52-bit).
                    if target_signed {
                        IrConst::long_double_from_i64(bits as i64)
                    } else {
                        IrConst::long_double_from_u64(bits)
                    }
                } else {
                    IrConst::I128(bits as i128) // __int128 / unsigned __int128
                }
            }
            _ => {
                // Pointer types and other 8-byte types
                if target.is_pointer_like() {
                    IrConst::I64(bits as i64)
                } else {
                    return None;
                }
            }
        })
    }

    /// Cast a full 128-bit integer value to a target CType.
    ///
    /// This handles casts from __int128/unsigned __int128 without going through the
    /// u64-based eval_const_expr_as_bits path, which would truncate the upper 64 bits.
    /// For targets <= 64 bits, extract the lower bits. For 128-bit targets, preserve
    /// the full value.
    fn cast_i128_to_ctype(&self, v128: i128, target: &CType, target_size: usize, src_unsigned: bool) -> Option<IrConst> {
        let bits_lo = v128 as u64; // lower 64 bits
        let target_signed = !target.is_unsigned() && !target.is_pointer_like();
        Some(match target_size {
            0 => return None, // void cast
            1 => {
                if !target_signed {
                    IrConst::I32(bits_lo as u8 as i32)
                } else {
                    IrConst::I8(bits_lo as i8)
                }
            }
            2 => {
                if !target_signed {
                    IrConst::I32(bits_lo as u16 as i32)
                } else {
                    IrConst::I16(bits_lo as i16)
                }
            }
            4 => {
                if matches!(target, CType::Float) {
                    // int-to-float: signedness comes from the source type
                    let fv = if src_unsigned { (v128 as u128) as f32 } else { v128 as f32 };
                    IrConst::F32(fv)
                } else if !target_signed {
                    IrConst::I64(bits_lo as u32 as i64)
                } else {
                    IrConst::I32(bits_lo as i32)
                }
            }
            8 => {
                if matches!(target, CType::Double) {
                    let fv = if src_unsigned { (v128 as u128) as f64 } else { v128 as f64 };
                    IrConst::F64(fv)
                } else {
                    IrConst::I64(bits_lo as i64)
                }
            }
            16 => {
                if matches!(target, CType::LongDouble) {
                    // Use direct integer-to-x87 conversion to preserve full 64-bit
                    // mantissa precision (x87 has 64-bit mantissa, unlike f64's 52-bit).
                    if src_unsigned {
                        IrConst::long_double_from_u128(v128 as u128)
                    } else {
                        IrConst::long_double_from_i128(v128)
                    }
                } else {
                    // __int128 / unsigned __int128: preserve full 128-bit value
                    IrConst::I128(v128)
                }
            }
            _ => {
                if target.is_pointer_like() {
                    IrConst::I64(bits_lo as i64)
                } else {
                    return None;
                }
            }
        })
    }

    /// Build an EnumType for a packed enum from its variants or type context.
    fn resolve_packed_enum_type(
        &self,
        name: &Option<Rc<str>>,
        variants: &Option<Vec<crate::frontend::parser::ast::EnumVariant>>,
    ) -> crate::common::types::EnumType {
        // Try looking up previously registered packed enum
        if let Some(tag) = name {
            if let Some(et) = self.types.packed_enum_types.get(tag) {
                return et.clone();
            }
        }
        // Build from variant list
        let variant_values = if let Some(vars) = variants {
            let mut result = Vec::new();
            let mut next_val: i64 = 0;
            for v in vars {
                if let Some(ref val_expr) = v.value {
                    if let Some(val) = self.eval_const_expr(val_expr) {
                        next_val = val.to_i64().unwrap_or(next_val);
                    }
                }
                result.push((v.name.clone(), next_val));
                next_val += 1;
            }
            result
        } else {
            Vec::new()
        };
        crate::common::types::EnumType {
            name: name.clone(),
            variants: variant_values,
            is_packed: true,
        }
    }

    /// Compute sizeof for a type specifier.
    /// Returns None if the type cannot be sized (e.g., typeof(expr) without expr type info).
    fn sizeof_type_spec(&self, spec: &TypeSpecifier) -> Option<usize> {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        match spec {
            TypeSpecifier::Void => Some(0),
            TypeSpecifier::Char | TypeSpecifier::UnsignedChar => Some(1),
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => Some(2),
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Signed | TypeSpecifier::Unsigned => Some(4),
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong => Some(ptr_sz),
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong => Some(8),
            TypeSpecifier::Int128 | TypeSpecifier::UnsignedInt128 => Some(16),
            TypeSpecifier::Float => Some(4),
            TypeSpecifier::Double => Some(8),
            TypeSpecifier::LongDouble => Some(if ptr_sz == 4 { 12 } else { 16 }),
            TypeSpecifier::Bool => Some(1),
            TypeSpecifier::ComplexFloat => Some(8),
            TypeSpecifier::ComplexDouble => Some(16),
            TypeSpecifier::ComplexLongDouble => Some(if ptr_sz == 4 { 24 } else { 32 }),
            TypeSpecifier::Pointer(_, _) | TypeSpecifier::FunctionPointer(_, _, _) => Some(ptr_sz),
            TypeSpecifier::Array(elem, Some(size)) => {
                let elem_size = self.sizeof_type_spec(elem)?;
                let n = self.eval_const_expr(size)?.to_i64()?;
                Some(elem_size * n as usize)
            }
            TypeSpecifier::Array(_, None) => Some(ptr_sz), // incomplete array
            TypeSpecifier::Struct(tag, fields, is_packed, pragma_pack, struct_aligned) => {
                // Look up cached layout for tagged structs
                if let Some(tag) = tag {
                    let key = format!("struct.{}", tag);
                    if let Some(layout) = self.types.borrow_struct_layouts().get(key.as_str()) {
                        return Some(layout.size);
                    }
                }
                if let Some(fields) = fields {
                    let struct_fields = self.convert_struct_fields(fields);
                    if !struct_fields.is_empty() {
                        let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                        let mut layout = crate::common::types::StructLayout::for_struct_with_packing(
                            &struct_fields, max_field_align, &*self.types.borrow_struct_layouts()
                        );
                        if let Some(a) = struct_aligned {
                            if *a > layout.align {
                                layout.align = *a;
                                let mask = layout.align - 1;
                                layout.size = (layout.size + mask) & !mask;
                            }
                        }
                        return Some(layout.size);
                    }
                }
                Some(0)
            }
            TypeSpecifier::Union(tag, fields, is_packed, pragma_pack, struct_aligned) => {
                if let Some(tag) = tag {
                    let key = format!("union.{}", tag);
                    if let Some(layout) = self.types.borrow_struct_layouts().get(key.as_str()) {
                        return Some(layout.size);
                    }
                }
                if let Some(fields) = fields {
                    let union_fields = self.convert_struct_fields(fields);
                    if !union_fields.is_empty() {
                        let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                        let mut layout = crate::common::types::StructLayout::for_union_with_packing(
                            &union_fields, max_field_align, &*self.types.borrow_struct_layouts()
                        );
                        if let Some(a) = struct_aligned {
                            if *a > layout.align {
                                layout.align = *a;
                                let mask = layout.align - 1;
                                layout.size = (layout.size + mask) & !mask;
                            }
                        }
                        return Some(layout.size);
                    }
                }
                Some(0)
            }
            TypeSpecifier::Enum(name, variants, is_packed) => {
                // Check if this is a forward reference to a known packed enum
                let effective_packed = *is_packed || name.as_ref()
                    .and_then(|n| self.types.packed_enum_types.get(n))
                    .is_some();
                if !effective_packed {
                    Some(4)
                } else {
                    // Resolve the packed enum to get its correct size
                    let et = self.resolve_packed_enum_type(name, variants);
                    Some(et.packed_size())
                }
            }
            TypeSpecifier::TypedefName(name) => {
                if let Some(ctype) = self.types.typedefs.get(name) {
                    Some(ctype.size_ctx(&*self.types.borrow_struct_layouts()))
                } else {
                    Some(8) // fallback
                }
            }
            TypeSpecifier::TypeofType(inner) => self.sizeof_type_spec(inner),
            TypeSpecifier::Typeof(expr) => {
                let ctype = self.infer_expr_ctype(expr)?;
                Some(ctype.size_ctx(&*self.types.borrow_struct_layouts()))
            }
            TypeSpecifier::Vector(_, total_bytes) => Some(*total_bytes),
            _ => None,
        }
    }

    /// Compute sizeof for an expression via its CType.
    ///
    /// Special cases: string literals are arrays, not pointers, inside sizeof
    /// (C11 6.3.2.1p3: array-to-pointer decay does not apply to sizeof operands).
    fn sizeof_expr(&self, expr: &Expr) -> Option<usize> {
        match expr {
            // "hello" is char[6], not char* -- sizeof gives array size.
            // Use chars().count() because high-byte escape sequences like \xff
            // are stored as multi-byte UTF-8 in the Rust String but represent
            // single bytes in C.
            Expr::StringLiteral(s, _) => return Some(s.chars().count() + 1),
            // L"hello" is wchar_t[6] (int[6] on Linux) -- sizeof gives array size
            Expr::WideStringLiteral(s, _) => return Some((s.chars().count() + 1) * 4),
            // u"hello" is char16_t[6] (unsigned short[6]) -- sizeof gives array size
            Expr::Char16StringLiteral(s, _) => return Some((s.chars().count() + 1) * 2),
            _ => {}
        }
        let ctype = self.infer_expr_ctype(expr)?;
        Some(ctype.size_ctx(&*self.types.borrow_struct_layouts()))
    }

    /// Compute alignof for a type specifier.
    fn alignof_type_spec(&self, spec: &TypeSpecifier) -> usize {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        match spec {
            TypeSpecifier::Void | TypeSpecifier::Bool => 1,
            TypeSpecifier::Char | TypeSpecifier::UnsignedChar => 1,
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => 2,
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Signed | TypeSpecifier::Unsigned => 4,
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong => ptr_sz,
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong => if ptr_sz == 4 { 4 } else { 8 },
            TypeSpecifier::Int128 | TypeSpecifier::UnsignedInt128 => 16,
            TypeSpecifier::Float => 4,
            TypeSpecifier::Double => if ptr_sz == 4 { 4 } else { 8 },
            // On i686, long double is 80-bit x87 aligned to 4 bytes
            TypeSpecifier::LongDouble => if ptr_sz == 4 { 4 } else { 16 },
            TypeSpecifier::ComplexFloat => 4,
            TypeSpecifier::ComplexDouble => if ptr_sz == 4 { 4 } else { 8 },
            TypeSpecifier::ComplexLongDouble => if ptr_sz == 4 { 4 } else { 16 },
            TypeSpecifier::Pointer(_, _) | TypeSpecifier::FunctionPointer(_, _, _) => ptr_sz,
            TypeSpecifier::Array(elem, _) => self.alignof_type_spec(elem),
            TypeSpecifier::Struct(tag, fields, is_packed, pragma_pack, struct_aligned) => {
                if let Some(tag) = tag {
                    let key = format!("struct.{}", tag);
                    if let Some(layout) = self.types.borrow_struct_layouts().get(key.as_str()) {
                        return layout.align;
                    }
                }
                if let Some(fields) = fields {
                    let struct_fields = self.convert_struct_fields(fields);
                    if !struct_fields.is_empty() {
                        let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                        let mut layout = crate::common::types::StructLayout::for_struct_with_packing(
                            &struct_fields, max_field_align, &*self.types.borrow_struct_layouts()
                        );
                        if let Some(a) = struct_aligned {
                            if *a > layout.align {
                                layout.align = *a;
                            }
                        }
                        return layout.align;
                    }
                }
                struct_aligned.unwrap_or(1)
            }
            TypeSpecifier::Union(tag, fields, is_packed, pragma_pack, struct_aligned) => {
                if let Some(tag) = tag {
                    let key = format!("union.{}", tag);
                    if let Some(layout) = self.types.borrow_struct_layouts().get(key.as_str()) {
                        return layout.align;
                    }
                }
                if let Some(fields) = fields {
                    let union_fields = self.convert_struct_fields(fields);
                    if !union_fields.is_empty() {
                        let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                        let mut layout = crate::common::types::StructLayout::for_union_with_packing(
                            &union_fields, max_field_align, &*self.types.borrow_struct_layouts()
                        );
                        if let Some(a) = struct_aligned {
                            if *a > layout.align {
                                layout.align = *a;
                            }
                        }
                        return layout.align;
                    }
                }
                struct_aligned.unwrap_or(1)
            }
            TypeSpecifier::Enum(name, variants, is_packed) => {
                let effective_packed = *is_packed || name.as_ref()
                    .and_then(|n| self.types.packed_enum_types.get(n))
                    .is_some();
                if !effective_packed {
                    4
                } else {
                    let et = self.resolve_packed_enum_type(name, variants);
                    et.packed_size()
                }
            }
            TypeSpecifier::TypedefName(name) => {
                if let Some(ctype) = self.types.typedefs.get(name) {
                    ctype.align_ctx(&*self.types.borrow_struct_layouts())
                } else {
                    crate::common::types::target_ptr_size()
                }
            }
            TypeSpecifier::TypeofType(inner) => self.alignof_type_spec(inner),
            TypeSpecifier::Typeof(expr) => {
                if let Some(ctype) = self.infer_expr_ctype(expr) {
                    ctype.align_ctx(&*self.types.borrow_struct_layouts())
                } else {
                    crate::common::types::target_ptr_size() // fallback
                }
            }
            TypeSpecifier::Vector(_, total_bytes) => (*total_bytes).min(16),
            _ => crate::common::types::target_ptr_size(),
        }
    }

    /// Compute preferred (natural) alignment for a type specifier.
    /// Used by GCC's __alignof/__alignof__ which returns preferred alignment.
    /// On i686: __alignof__(long long) == 8, __alignof__(double) == 8.
    fn preferred_alignof_type_spec(&self, spec: &TypeSpecifier) -> usize {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        if ptr_sz != 4 {
            return self.alignof_type_spec(spec);
        }
        // On i686: long long and double have preferred alignment of 8
        match spec {
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong => 8,
            TypeSpecifier::Double => 8,
            TypeSpecifier::ComplexDouble => 8,
            TypeSpecifier::Array(elem, _) => self.preferred_alignof_type_spec(elem),
            TypeSpecifier::TypedefName(name) => {
                if let Some(ctype) = self.types.typedefs.get(name) {
                    ctype.preferred_align_ctx(&*self.types.borrow_struct_layouts())
                } else {
                    target_ptr_size()
                }
            }
            TypeSpecifier::TypeofType(inner) => self.preferred_alignof_type_spec(inner),
            TypeSpecifier::Typeof(expr) => {
                if let Some(ctype) = self.infer_expr_ctype(expr) {
                    ctype.preferred_align_ctx(&*self.types.borrow_struct_layouts())
                } else {
                    target_ptr_size()
                }
            }
            TypeSpecifier::Vector(_, total_bytes) => (*total_bytes).min(16),
            _ => self.alignof_type_spec(spec),
        }
    }

    /// Convert struct field declarations to StructField for layout computation.
    fn convert_struct_fields(&self, fields: &[StructFieldDecl]) -> Vec<crate::common::types::StructField> {
        fields.iter().map(|f| {
            let ty = ctype_from_type_spec_with_derived(&f.type_spec, &f.derived, self.types);
            let name = f.name.clone().unwrap_or_default();
            let bit_width = f.bit_width.as_ref().and_then(|bw| {
                self.eval_const_expr(bw)?.to_i64().map(|v| v as u32)
            });
            let field_alignment = {
                let mut align = f.alignment;
                if let TypeSpecifier::TypedefName(td_name) = &f.type_spec {
                    if let Some(&ta) = self.types.typedef_alignments.get(td_name) {
                        align = Some(align.map_or(ta, |a| a.max(ta)));
                    }
                }
                align
            };
            crate::common::types::StructField {
                name,
                ty,
                bit_width,
                alignment: field_alignment,
                is_packed: f.is_packed,
            }
        }).collect()
    }

    /// Check if an expression is always non-zero at compile time, even if we
    /// can't compute its exact numeric value. String literals are always
    /// non-null pointers, so `"hello" || 0` should evaluate to 1 in static
    /// initializers.
    fn expr_is_always_nonzero(expr: &Expr) -> bool {
        match expr {
            Expr::StringLiteral(..)
            | Expr::WideStringLiteral(..)
            | Expr::Char16StringLiteral(..) => true,
            Expr::Cast(_, inner, _) => Self::expr_is_always_nonzero(inner),
            _ => false,
        }
    }
}

/// Convert a TypeSpecifier to CType using the TypeContext for typedef/struct resolution.
/// This is a standalone function that doesn't need the full TypeConvertContext trait.
fn ctype_from_type_spec(spec: &TypeSpecifier, types: &TypeContext) -> CType {
    match spec {
        TypeSpecifier::Void => CType::Void,
        TypeSpecifier::Bool => CType::Bool,
        TypeSpecifier::Char => CType::Char,
        TypeSpecifier::UnsignedChar => CType::UChar,
        TypeSpecifier::Short => CType::Short,
        TypeSpecifier::UnsignedShort => CType::UShort,
        TypeSpecifier::Int | TypeSpecifier::Signed => CType::Int,
        TypeSpecifier::UnsignedInt | TypeSpecifier::Unsigned => CType::UInt,
        TypeSpecifier::Long => CType::Long,
        TypeSpecifier::UnsignedLong => CType::ULong,
        TypeSpecifier::LongLong => CType::LongLong,
        TypeSpecifier::UnsignedLongLong => CType::ULongLong,
        TypeSpecifier::Int128 => CType::Int128,
        TypeSpecifier::UnsignedInt128 => CType::UInt128,
        TypeSpecifier::Float => CType::Float,
        TypeSpecifier::Double => CType::Double,
        TypeSpecifier::LongDouble => CType::LongDouble,
        TypeSpecifier::Pointer(inner, addr_space) => CType::Pointer(Box::new(ctype_from_type_spec(inner, types)), *addr_space),
        TypeSpecifier::Array(elem, size) => {
            let elem_ty = ctype_from_type_spec(elem, types);
            // TODO: evaluate array size expression when available
            let array_size = size.as_ref().and_then(|s| {
                // Try simple literal evaluation for array sizes
                match s.as_ref() {
                    Expr::IntLiteral(n, _) | Expr::LongLiteral(n, _) | Expr::LongLongLiteral(n, _) => Some(*n as usize),
                    Expr::UIntLiteral(n, _) | Expr::ULongLiteral(n, _) | Expr::ULongLongLiteral(n, _) => Some(*n as usize),
                    _ => None,
                }
            });
            CType::Array(Box::new(elem_ty), array_size)
        }
        TypeSpecifier::TypedefName(name) => {
            if let Some(resolved) = types.typedefs.get(name) {
                resolved.clone()
            } else {
                CType::Int // fallback
            }
        }
        TypeSpecifier::Struct(tag, _, _, _, _) => {
            if let Some(tag) = tag {
                CType::Struct(format!("struct.{}", tag).into())
            } else {
                CType::Int // anonymous struct without context
            }
        }
        TypeSpecifier::Union(tag, _, _, _, _) => {
            if let Some(tag) = tag {
                CType::Union(format!("union.{}", tag).into())
            } else {
                CType::Int // anonymous union without context
            }
        }
        TypeSpecifier::Enum(_, _, false) => CType::Int, // non-packed enums are int-sized
        TypeSpecifier::Enum(name, variants, true) => {
            // Packed enum: look up from type context or compute from variants
            if let Some(tag) = name {
                if let Some(et) = types.packed_enum_types.get(tag) {
                    return CType::Enum(et.clone());
                }
            }
            // Inline definition: compute from variants
            if let Some(vars) = variants {
                let mut variant_values = Vec::new();
                for (next_val, v) in (0_i64..).zip(vars.iter()) {
                    if let Some(ref _val_expr) = v.value {
                        // Can't easily eval here, but this path is rare
                    }
                    variant_values.push((v.name.clone(), next_val));
                }
                CType::Enum(crate::common::types::EnumType {
                    name: name.clone(),
                    variants: variant_values,
                    is_packed: true,
                })
            } else {
                CType::Int // packed enum forward ref without definition
            }
        }
        TypeSpecifier::TypeofType(inner) => ctype_from_type_spec(inner, types),
        TypeSpecifier::FunctionPointer(_, _, _) => {
            CType::Pointer(Box::new(CType::Void), AddressSpace::Default) // function pointers are pointer-sized
        }
        TypeSpecifier::Vector(inner, total_bytes) => {
            let elem_ctype = ctype_from_type_spec(inner, types);
            CType::Vector(Box::new(elem_ctype), *total_bytes)
        }
        _ => CType::Int, // fallback
    }
}

/// Convert a TypeSpecifier with derived declarators to CType.
fn ctype_from_type_spec_with_derived(
    spec: &TypeSpecifier,
    derived: &[DerivedDeclarator],
    types: &TypeContext,
) -> CType {
    let mut ty = ctype_from_type_spec(spec, types);
    if derived.is_empty() {
        return ty;
    }
    for d in derived {
        match d {
            DerivedDeclarator::Pointer => {
                ty = CType::Pointer(Box::new(ty), AddressSpace::Default);
            }
            DerivedDeclarator::Array(Some(size_expr)) => {
                let expr: &Expr = size_expr;
                let size = match expr {
                    Expr::IntLiteral(n, _) | Expr::LongLiteral(n, _) | Expr::LongLongLiteral(n, _) => Some(*n as usize),
                    Expr::UIntLiteral(n, _) | Expr::ULongLiteral(n, _) | Expr::ULongLongLiteral(n, _) => Some(*n as usize),
                    _ => None,
                };
                ty = CType::Array(Box::new(ty), size);
            }
            DerivedDeclarator::Array(None) => {
                ty = CType::Array(Box::new(ty), None);
            }
            DerivedDeclarator::Function(_, _) | DerivedDeclarator::FunctionPointer(_, _) => {
                ty = CType::Pointer(Box::new(CType::Void), AddressSpace::Default); // function -> pointer
            }
        }
    }
    ty
}
