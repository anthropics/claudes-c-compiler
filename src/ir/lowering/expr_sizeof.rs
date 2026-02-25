//! Expression sizeof, alignof, and related size computations.
//!
//! This module contains functions for computing the size and alignment of C expressions,
//! extracted from expr_types.rs for maintainability. Includes sizeof_expr, alignof_expr,
//! preferred_alignof_expr, and their helper functions.

use crate::frontend::parser::ast::{
    BinOp,
    Expr,
    Initializer,
    TypeSpecifier,
    UnaryOp,
};
use crate::common::types::CType;
use super::lower::Lowerer;

impl Lowerer {

    /// Get the sizeof for an identifier expression.
    fn sizeof_identifier(&self, name: &str) -> usize {
        if let Some(info) = self.func_state.as_ref().and_then(|fs| fs.locals.get(name)) {
            if info.is_array || info.is_struct {
                return info.alloc_size;
            }
            // Use CType size if available (handles long double, function types, etc.)
            if let Some(ref ct) = info.c_type {
                if matches!(ct, CType::Function(_)) {
                    // GCC extension: sizeof(function_type) == 1
                    return 1;
                }
                let ct_size = self.ctype_size(ct);
                if ct_size > 0 {
                    return ct_size;
                }
            }
            return info.ty.size();
        }
        if let Some(ginfo) = self.globals.get(name) {
            // For arrays, prefer the actual allocation size from the module global
            // (handles implicit-size arrays like `int x[] = {1,2,3}` where
            // CType may still be Array(_, None) returning pointer size).
            if ginfo.is_array {
                for g in &self.module.globals {
                    if &*g.name == name {
                        return g.size;
                    }
                }
            }
            // For structs, prefer CType size (returns the type size without
            // flexible array member data, matching C standard sizeof semantics).
            if let Some(ref ct) = ginfo.c_type {
                let ct_size = self.ctype_size(ct);
                if ct_size > 0 {
                    return ct_size;
                }
            }
            if ginfo.is_struct {
                for g in &self.module.globals {
                    if &*g.name == name {
                        return g.size;
                    }
                }
            }
            return ginfo.ty.size();
        }
        // GCC extension: sizeof(function_name) == 1
        if self.known_functions.contains(name) {
            return 1;
        }
        // Unknown identifier - sema should have caught this, but fallback to int size
        4
    }

    /// Get the sizeof for a dereference expression.
    fn sizeof_deref(&self, inner: &Expr) -> usize {
        // Use CType-based resolution first
        if let Some(inner_ctype) = self.get_expr_ctype(inner) {
            match &inner_ctype {
                CType::Pointer(pointee, _) => {
                    // GCC extension: sizeof(*void_ptr) == 1, sizeof(*func_ptr) == 1
                    if matches!(pointee.as_ref(), CType::Void | CType::Function(_)) {
                        return 1;
                    }
                    let sz = self.resolve_ctype_size(pointee);
                    if sz == 0 {
                        return 1;
                    }
                    return sz;
                }
                CType::Array(elem, _) => return self.resolve_ctype_size(elem).max(1),
                // GCC extension: sizeof(*func) == 1 where func is a function
                CType::Function(_) => return 1,
                _ => {}
            }
        }
        if let Expr::Identifier(name, _) = inner {
            if let Some(vi) = self.lookup_var_info(name) {
                // GCC extension: sizeof(*func_ptr) == 1
                if let Some(ref ct) = vi.c_type {
                    if matches!(ct, CType::Function(_)) {
                        return 1;
                    }
                    if let CType::Pointer(pointee, _) = ct {
                        if matches!(pointee.as_ref(), CType::Function(_)) {
                            return 1;
                        }
                    }
                }
                if vi.elem_size > 0 {
                    return vi.elem_size;
                }
            }
            // GCC extension: sizeof(*func_name) == 1 for known functions
            if self.known_functions.contains(name) {
                return 1;
            }
        }
        crate::common::types::target_ptr_size() // TODO: better type tracking for nested derefs
    }

    /// Get the sizeof for an array subscript expression.
    fn sizeof_subscript(&self, base: &Expr, index: &Expr) -> usize {
        // Use CType-based resolution first (handles string literals, typed pointers, vectors)
        if let Some(base_ctype) = self.get_expr_ctype(base) {
            match &base_ctype {
                CType::Array(elem, _) => return self.resolve_ctype_size(elem).max(1),
                CType::Pointer(pointee, _) => return self.resolve_ctype_size(pointee).max(1),
                CType::Vector(elem, _) => return self.resolve_ctype_size(elem).max(1),
                _ => {}
            }
        }
        // Also check reverse subscript (index[base])
        if let Some(idx_ctype) = self.get_expr_ctype(index) {
            match &idx_ctype {
                CType::Array(elem, _) => return self.resolve_ctype_size(elem).max(1),
                CType::Pointer(pointee, _) => return self.resolve_ctype_size(pointee).max(1),
                CType::Vector(elem, _) => return self.resolve_ctype_size(elem).max(1),
                _ => {}
            }
        }
        if let Expr::Identifier(name, _) = base {
            if let Some(vi) = self.lookup_var_info(name) {
                if vi.elem_size > 0 {
                    return vi.elem_size;
                }
            }
        }
        // Fallback for member access bases (e.g., p->c[0] or x.arr[0])
        match base {
            Expr::MemberAccess(base_expr, field_name, _) | Expr::PointerMemberAccess(base_expr, field_name, _) => {
                let is_ptr = matches!(base, Expr::PointerMemberAccess(..));
                if let Some(ctype) = self.resolve_field_ctype(base_expr, field_name, is_ptr) {
                    match &ctype {
                        CType::Array(elem_ty, _) => return self.resolve_ctype_size(elem_ty).max(1),
                        CType::Pointer(pointee, _) => return self.resolve_ctype_size(pointee).max(1),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        4 // default: int element
    }

    /// Get the sizeof for a member access expression.
    fn sizeof_member_access(&self, base_expr: &Expr, field_name: &str, is_pointer: bool) -> usize {
        if let Some(ctype) = self.resolve_field_ctype(base_expr, field_name, is_pointer) {
            // Trust the CType-based size when we have a resolved field type.
            // Size 0 is valid for zero-length arrays (char arr[0]) and empty structs/unions.
            // The IrType fallback below would incorrectly return pointer size (8) for arrays
            // because IrType::from_ctype decays all arrays to IrType::Ptr.
            return self.ctype_size(&ctype);
        }
        if is_pointer {
            let (_, field_ty) = self.resolve_pointer_member_access(base_expr, field_name);
            field_ty.size()
        } else {
            let (_, field_ty) = self.resolve_member_access(base_expr, field_name);
            field_ty.size()
        }
    }

    /// Get the sizeof for a binary operation expression.
    fn sizeof_binop(&self, op: &BinOp, lhs: &Expr, rhs: &Expr) -> usize {
        match op {
            // Comparison/logical: result is int
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le
            | BinOp::Gt | BinOp::Ge | BinOp::LogicalAnd | BinOp::LogicalOr => 4,
            // Pointer subtraction: result is ptrdiff_t
            BinOp::Sub if self.sizeof_operand_is_pointer_like(lhs)
                && self.sizeof_operand_is_pointer_like(rhs) => crate::common::types::target_ptr_size(),
            // Pointer arithmetic (ptr + int, int + ptr, ptr - int): result is pointer
            BinOp::Add | BinOp::Sub
                if self.sizeof_operand_is_pointer_like(lhs)
                || self.sizeof_operand_is_pointer_like(rhs) => crate::common::types::target_ptr_size(),
            // Shift operators: result type is promoted left operand
            BinOp::Shl | BinOp::Shr => {
                self.sizeof_expr(lhs).max(4) // integer promotion of left operand only
            }
            // Arithmetic/bitwise: usual arithmetic conversions
            _ => {
                let ls = self.sizeof_expr(lhs);
                let rs = self.sizeof_expr(rhs);
                ls.max(rs).max(4) // integer promotion
            }
        }
    }

    /// Check if an expression is pointer-like for sizeof computation.
    /// Arrays decay to pointers in expression context, so both pointers and arrays
    /// produce pointer-typed results in arithmetic.
    fn sizeof_operand_is_pointer_like(&self, expr: &Expr) -> bool {
        if self.expr_is_array_name(expr) {
            return true;
        }
        if let Some(ctype) = self.get_expr_ctype(expr) {
            return matches!(ctype, CType::Pointer(_, _) | CType::Array(_, _));
        }
        false
    }

    /// Compute sizeof for an expression operand (sizeof expr).
    /// Returns the size in bytes of the expression's type.
    pub(super) fn sizeof_expr(&self, expr: &Expr) -> usize {
        match expr {
            // Integer literal: type int (4 bytes), unless value overflows to long
            Expr::IntLiteral(val, _) => {
                if *val >= i32::MIN as i64 && *val <= i32::MAX as i64 {
                    4
                } else {
                    8
                }
            }
            // Unsigned int literal: type unsigned int (4 bytes) if fits, else unsigned long
            Expr::UIntLiteral(val, _) => {
                if *val <= u32::MAX as u64 {
                    4
                } else {
                    8
                }
            }
            // Long literal: on ILP32, long is 4 bytes unless value overflows to long long (8 bytes)
            Expr::LongLiteral(val, _) => {
                if crate::common::types::target_is_32bit() {
                    if *val >= i32::MIN as i64 && *val <= i32::MAX as i64 { 4 } else { 8 }
                } else {
                    8 // LP64: long is always 8 bytes
                }
            }
            // Unsigned long literal: on ILP32, unsigned long is 4 bytes unless value overflows
            Expr::ULongLiteral(val, _) => {
                if crate::common::types::target_is_32bit() {
                    if *val <= u32::MAX as u64 { 4 } else { 8 }
                } else {
                    8 // LP64: unsigned long is always 8 bytes
                }
            }
            // long long is always 8 bytes regardless of target
            Expr::LongLongLiteral(_, _) | Expr::ULongLongLiteral(_, _) => 8,
            // Float literal: type double (8 bytes) by default in C
            Expr::FloatLiteral(_, _) => 8,
            // Float literal with f suffix: type float (4 bytes)
            Expr::FloatLiteralF32(_, _) => 4,
            // Float literal with L suffix: type long double (16 on LP64, 12 on ILP32)
            Expr::FloatLiteralLongDouble(_, _, _) => if crate::common::types::target_is_32bit() { 12 } else { 16 },
            // Char literal: type int in C (4 bytes)
            Expr::CharLiteral(_, _) => 4,
            // String literal: array of char, size = length + 1 (null terminator)
            Expr::StringLiteral(s, _) => s.chars().count() + 1,
            // Wide string literal: array of wchar_t (4 bytes each), size = (chars + 1) * 4
            Expr::WideStringLiteral(s, _) => (s.chars().count() + 1) * 4,
            // char16_t string literal: array of char16_t (2 bytes each), size = (chars + 1) * 2
            Expr::Char16StringLiteral(s, _) => (s.chars().count() + 1) * 2,

            // Variable: look up its alloc_size or type
            Expr::Identifier(name, _) => {
                self.sizeof_identifier(name)
            }

            // Dereference: element type size
            Expr::Deref(inner, _) => {
                self.sizeof_deref(inner)
            }

            // Array subscript: element type size
            Expr::ArraySubscript(base, index, _) => {
                self.sizeof_subscript(base, index)
            }

            // sizeof(sizeof(...)) or sizeof(_Alignof(...)) -> size_t
            Expr::Sizeof(_, _) | Expr::Alignof(_, _) | Expr::AlignofExpr(_, _)
            | Expr::GnuAlignof(_, _) | Expr::GnuAlignofExpr(_, _) => crate::common::types::target_ptr_size(),

            // Cast: size of the target type
            Expr::Cast(target_type, _, _) => {
                self.sizeof_type(target_type)
            }

            // Member access: member field size (use CType for accurate array/struct sizes)
            Expr::MemberAccess(base_expr, field_name, _) => {
                self.sizeof_member_access(base_expr, field_name, false)
            }
            Expr::PointerMemberAccess(base_expr, field_name, _) => {
                self.sizeof_member_access(base_expr, field_name, true)
            }

            // Address-of: pointer
            Expr::AddressOf(_, _) => crate::common::types::target_ptr_size(),

            // Unary operations
            Expr::UnaryOp(op, inner, _) => {
                match op {
                    UnaryOp::LogicalNot => 4, // result is int
                    UnaryOp::Neg | UnaryOp::Plus | UnaryOp::BitNot => {
                        self.sizeof_expr(inner).max(4) // integer promotion
                    }
                    UnaryOp::PreInc | UnaryOp::PreDec => self.sizeof_expr(inner),
                    UnaryOp::RealPart | UnaryOp::ImagPart => {
                        // Result is the component type size
                        let inner_ctype = self.expr_ctype(inner);
                        inner_ctype.complex_component_type().size()
                    }
                }
            }

            // Postfix operations preserve the operand type
            Expr::PostfixOp(_, inner, _) => self.sizeof_expr(inner),

            // Binary operations
            Expr::BinaryOp(op, lhs, rhs, _) => {
                self.sizeof_binop(op, lhs, rhs)
            }

            // Conditional: use composite type for accurate sizeof
            Expr::Conditional(_, then_e, else_e, _) => {
                if let Some(ctype) = self.get_expr_ctype(expr) {
                    return self.ctype_size(&ctype);
                }
                let ts = self.sizeof_expr(then_e);
                let es = self.sizeof_expr(else_e);
                ts.max(es)
            }
            Expr::GnuConditional(cond, else_e, _) => {
                if let Some(ctype) = self.get_expr_ctype(expr) {
                    return self.ctype_size(&ctype);
                }
                let cs = self.sizeof_expr(cond);
                let es = self.sizeof_expr(else_e);
                cs.max(es)
            }

            // Assignment: type of the left-hand side
            Expr::Assign(lhs, _, _) | Expr::CompoundAssign(_, lhs, _, _) => {
                self.sizeof_expr(lhs)
            }

            // Comma: type of the right expression
            Expr::Comma(_, rhs, _) => {
                self.sizeof_expr(rhs)
            }

            // Function call: use the actual return type
            Expr::FunctionCall(_, _, _) => {
                // Prefer CType which has correct struct/union sizes
                if let Some(ctype) = self.get_expr_ctype(expr) {
                    self.ctype_size(&ctype)
                } else {
                    let ret_ty = self.get_expr_type(expr);
                    ret_ty.size()
                }
            }

            // Compound literal: size of the type (handle incomplete array types)
            Expr::CompoundLiteral(ts, ref init, _) => {
                let ctype = self.type_spec_to_ctype(ts);
                match (&ctype, init.as_ref()) {
                    (CType::Array(ref elem_ct, None), Initializer::List(items)) => {
                        self.ctype_size(elem_ct).max(1) * items.len()
                    }
                    _ => self.sizeof_type(ts),
                }
            }

            // _Generic selection: resolve the matching association and compute its size
            Expr::GenericSelection(controlling, associations, _) => {
                if let Some(ctype) = self.resolve_generic_selection_ctype(controlling, associations) {
                    self.ctype_size(&ctype)
                } else {
                    // Fallback to IrType-based sizing
                    let ir_type = self.resolve_generic_selection_type(controlling, associations);
                    ir_type.size()
                }
            }

            // Statement expression: type of the last expression in the block
            Expr::StmtExpr(_, _) => {
                if let Some(ctype) = self.get_expr_ctype(expr) {
                    self.ctype_size(&ctype)
                } else {
                    let ty = self.get_expr_type(expr);
                    ty.size()
                }
            }

            // Default
            _ => 4,
        }
    }

    /// Compute alignof for an expression by inferring its type and returning
    /// the type's alignment. This implements GCC's __alignof__(expr) semantics.
    ///
    /// Per C11 6.2.8p3, if the expression names a variable declared with
    /// _Alignas or __attribute__((aligned(N))), the result reflects that
    /// explicit alignment (taking the max of the natural alignment and the
    /// declared alignment).
    pub(super) fn alignof_expr(&self, expr: &Expr) -> usize {
        // Check if the expression is a variable with an explicit alignment override.
        // _Alignof on an identifier should return the variable's declared alignment,
        // not just its natural type alignment.
        if let Expr::Identifier(name, _) = expr {
            if let Some(vi) = self.lookup_var_info(name) {
                if let Some(explicit_align) = vi.explicit_alignment {
                    // Return max of natural type alignment and explicit alignment
                    let natural = if let Some(ref ctype) = vi.c_type {
                        self.ctype_align(ctype)
                    } else {
                        vi.ty.align()
                    };
                    return natural.max(explicit_align);
                }
            }
        }
        // Try to get the CType of the expression for accurate alignment
        if let Some(ctype) = self.get_expr_ctype(expr) {
            return self.ctype_align(&ctype);
        }
        // Fallback: derive alignment from the expression's size
        // (this handles literals and simple cases where CType isn't available)
        let size = self.sizeof_expr(expr);
        let ptr_sz = crate::common::types::target_ptr_size();
        match size {
            1 => 1,
            2 => 2,
            4 => 4,
            16 => 16,  // __int128
            // On ILP32, 8-byte types (double, long long) are typically aligned to 4
            8 if ptr_sz == 4 => 4,
            _ => ptr_sz,
        }
    }

    /// Compute preferred alignment for an expression by inferring its type and returning
    /// the type's preferred alignment. This implements GCC's __alignof__(expr) semantics
    /// with preferred alignment (8 for long long/double on i686).
    pub(super) fn preferred_alignof_expr(&self, expr: &Expr) -> usize {
        use crate::common::types::target_ptr_size;
        if target_ptr_size() != 4 {
            return self.alignof_expr(expr);
        }
        // Check for explicit alignment on a variable identifier
        if let Expr::Identifier(name, _) = expr {
            if let Some(vi) = self.lookup_var_info(name) {
                if let Some(explicit_align) = vi.explicit_alignment {
                    let natural = if let Some(ref ctype) = vi.c_type {
                        ctype.preferred_align_ctx(&*self.types.borrow_struct_layouts())
                    } else {
                        // Fallback: use size as preferred alignment (min 1, max 16)
                        vi.ty.size().clamp(1, 16)
                    };
                    return natural.max(explicit_align);
                }
            }
        }
        if let Some(ctype) = self.get_expr_ctype(expr) {
            return ctype.preferred_align_ctx(&*self.types.borrow_struct_layouts());
        }
        // Fallback: derive preferred alignment from the expression's size
        let size = self.sizeof_expr(expr);
        match size {
            1 => 1,
            2 => 2,
            4 => 4,
            8 => 8,  // preferred alignment for 8-byte types on i686
            16 => 16,
            _ => target_ptr_size(),
        }
    }

    /// Get the element size for a compound literal type.
    /// For arrays, returns the element size; for scalars/structs, returns the full size.
    pub(super) fn compound_literal_elem_size(&self, ts: &TypeSpecifier) -> usize {
        let ctype = self.type_spec_to_ctype(ts);
        match &ctype {
            CType::Array(elem_ct, _) => self.ctype_size(elem_ct).max(1),
            CType::Vector(elem_ct, _) => elem_ct.size().max(1),
            _ => self.sizeof_type(ts),
        }
    }
}
