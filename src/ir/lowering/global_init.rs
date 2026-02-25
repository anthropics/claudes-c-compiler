//! Global initialization subsystem for the IR lowerer.
//!
//! Lowers AST `Initializer` nodes to IR `GlobalInit` values. Covers:
//! - Scalar initializers (integers, floats, long doubles, booleans)
//! - String literals (narrow, wide, char16) in both array and pointer contexts
//! - Compound literals at file scope (`&(struct S){1, 2}`)
//! - Array initializers (1D, multi-dimensional, designated, pointer arrays)
//! - Struct/union initializers (byte-serialized or compound/relocation-aware)
//! - Complex type arrays (`_Complex float arr[]`)
//! - Vector type arrays (`int __attribute__((vector_size(16))) arr[]`)
//!
//! The top-level entry point `lower_global_init` dispatches to focused helpers
//! for each initializer category, keeping each function short and readable.

use std::rc::Rc;
use crate::frontend::parser::ast::{
    BinOp,
    Designator,
    Expr,
    Initializer,
    InitializerItem,
    TypeSpecifier,
};
use crate::ir::reexports::{GlobalInit, IrConst, IrGlobal};

/// Kind of string literal for extract_string_literal helper.
enum StringLitKind {
    Narrow,
    Wide,
    Char16,
}
use crate::common::types::{IrType, StructLayout, RcLayout, CType};
use super::lower::Lowerer;
use super::global_init_helpers as h;

// =============================================================================
// Top-level entry point
// =============================================================================

impl Lowerer {
    /// Lower a global initializer to a GlobalInit value.
    ///
    /// This is the main dispatch: expressions go to `lower_global_init_expr`,
    /// initializer lists go to `lower_global_init_list`.
    pub(super) fn lower_global_init(
        &mut self,
        init: &Initializer,
        type_spec: &TypeSpecifier,
        base_ty: IrType,
        is_array: bool,
        elem_size: usize,
        total_size: usize,
        struct_layout: &Option<RcLayout>,
        array_dim_strides: &[usize],
    ) -> GlobalInit {
        let is_long_double_target = self.is_type_spec_long_double(type_spec);
        let is_bool_target = self.is_type_bool(type_spec);

        match init {
            Initializer::Expr(expr) => self.lower_global_init_expr(
                expr, type_spec, base_ty, is_array, struct_layout,
                is_long_double_target, is_bool_target,
            ),
            Initializer::List(items) => self.lower_global_init_list(
                items, type_spec, base_ty, is_array, elem_size, total_size,
                struct_layout, array_dim_strides,
                is_long_double_target, is_bool_target,
            ),
        }
    }
}

// =============================================================================
// Expression initializers (Initializer::Expr)
// =============================================================================

impl Lowerer {
    /// Lower an expression initializer: scalar constants, string literals,
    /// compound literals, address expressions, and label differences.
    fn lower_global_init_expr(
        &mut self,
        expr: &Expr,
        type_spec: &TypeSpecifier,
        base_ty: IrType,
        is_array: bool,
        _struct_layout: &Option<RcLayout>,
        is_long_double_target: bool,
        is_bool_target: bool,
    ) -> GlobalInit {
        // Resolve _Generic selections: unwrap to the selected expression before
        // any other processing, so all downstream logic (const eval, string
        // literals, address expressions, etc.) sees the resolved expression.
        if let Expr::GenericSelection(ref controlling, ref associations, _) = expr {
            if let Some(selected) = self.resolve_generic_selection_expr(controlling, associations) {
                return self.lower_global_init_expr(
                    selected, type_spec, base_ty, is_array, _struct_layout,
                    is_long_double_target, is_bool_target,
                );
            }
        }

        // Complex types: handle before scalar evaluation to prevent misinterpretation
        // of scalar values (e.g., `_Complex float g = 1.0f;`).
        {
            let ctype = self.type_spec_to_ctype(type_spec);
            if ctype.is_complex() {
                if let Some(init) = self.eval_complex_global_init(expr, &ctype) {
                    return init;
                }
            }
        }

        // Scalar constant
        if let Some(val) = self.eval_const_expr(expr) {
            return GlobalInit::Scalar(
                self.coerce_scalar_const(val, expr, base_ty, is_long_double_target, is_bool_target)
            );
        }

        // String literal (narrow, wide, or char16)
        if let Some(init) = self.lower_string_literal_init(expr, base_ty, is_array) {
            return init;
        }

        // &(compound_literal) at file scope
        if let Expr::AddressOf(inner, _) = expr {
            if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = inner.as_ref() {
                return self.create_compound_literal_global(cl_type_spec, cl_init);
            }
        }

        // Cast-wrapped compound literal: e.g., (char *)(unsigned char[]){ 0xFD }
        // Also handles (void*) &(CompoundLiteral) pattern used in static initializers.
        {
            let stripped = Self::strip_casts(expr);
            if !std::ptr::eq(expr as *const _, stripped as *const _) {
                if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = stripped {
                    return self.create_compound_literal_global(cl_type_spec, cl_init);
                }
                // (void*) &(CompoundLiteral) – cast wrapping address-of compound literal
                if let Expr::AddressOf(inner, _) = stripped {
                    if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = inner.as_ref() {
                        return self.create_compound_literal_global(cl_type_spec, cl_init);
                    }
                }
            }
        }

        // Compound literal used directly as initializer value
        if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = expr {
            return self.lower_compound_literal_init(
                cl_type_spec, cl_init, base_ty, is_array,
            );
        }

        // String literal with offset: "str" + N or "str" - N
        if let Some(addr_init) = self.eval_string_literal_addr_expr(expr) {
            return addr_init;
        }

        // Pre-materialize compound literals embedded in address arithmetic.
        // This handles patterns like:
        //   (type*)((char*)(int[]){1,2,3} + 8)
        // where a compound literal is used as the base of pointer arithmetic.
        // We create anonymous globals for such compound literals so that
        // eval_global_addr_expr can resolve them as GlobalAddr references.
        self.materialize_compound_literals_in_expr(expr);

        // Global address expression: &x, func, arr, &arr[3], &s.field
        if let Some(addr_init) = self.eval_global_addr_expr(expr) {
            return addr_init;
        }

        // Label difference: &&lab1 - &&lab2 (computed goto dispatch tables)
        if let Some(label_diff) = self.eval_label_diff_expr(expr, base_ty.size().max(4)) {
            return label_diff;
        }

        // Can't evaluate - zero init as fallback
        GlobalInit::Zero
    }

    /// Recursively walk an expression tree and materialize any compound literals
    /// found within arithmetic/cast contexts as anonymous globals. This is needed
    /// so that `eval_global_addr_expr` (which takes `&self`) can resolve compound
    /// literals as `GlobalAddr` references when they appear as bases of pointer
    /// arithmetic in global initializers.
    ///
    /// For example, the Cello library uses:
    ///   `var Alloc = (var)((char*)((var[]){ NULL, ... }) + sizeof(struct Header))`
    /// The compound literal `(var[]){ NULL, ... }` must be materialized as an
    /// anonymous global before the pointer arithmetic can be resolved.
    fn materialize_compound_literals_in_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::CompoundLiteral(ref type_spec, ref init, _) => {
                let key = expr as *const Expr as usize;
                if !self.materialized_compound_literals.contains_key(&key) {
                    // Materialize this compound literal as an anonymous global
                    let init_result = self.create_compound_literal_global(type_spec, init);
                    // Extract the global name from the result
                    if let GlobalInit::GlobalAddr(label) = init_result {
                        self.materialized_compound_literals.insert(key, label.to_string());
                    }
                }
            }
            Expr::Cast(_, inner, _) => {
                self.materialize_compound_literals_in_expr(inner);
            }
            Expr::BinaryOp(_, lhs, rhs, _) => {
                self.materialize_compound_literals_in_expr(lhs);
                self.materialize_compound_literals_in_expr(rhs);
            }
            Expr::UnaryOp(_, inner, _) => {
                self.materialize_compound_literals_in_expr(inner);
            }
            Expr::AddressOf(inner, _) => {
                self.materialize_compound_literals_in_expr(inner);
            }
            Expr::Conditional(cond, then_e, else_e, _) => {
                self.materialize_compound_literals_in_expr(cond);
                self.materialize_compound_literals_in_expr(then_e);
                self.materialize_compound_literals_in_expr(else_e);
            }
            _ => {}
        }
    }

    /// Try to resolve an expression as `&(compound_literal)`, possibly wrapped
    /// in casts. If found, materializes the compound literal as an anonymous global
    /// and returns `GlobalInit::GlobalAddr(label)`.
    ///
    /// This is used by `collect_compound_init_element` to handle `&((struct T){...})`
    /// patterns inside compound literal arrays (e.g., Cello's Instance() macro).
    /// Note: `lower_global_init_expr` has similar logic at lines 106-110 for the
    /// top-level expression case; this function handles the array element case.
    fn try_address_of_compound_literal(&mut self, expr: &Expr) -> Option<GlobalInit> {
        let stripped = Self::strip_casts(expr);
        if let Expr::AddressOf(inner, _) = stripped {
            if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = inner.as_ref() {
                return Some(self.create_compound_literal_global(cl_type_spec, cl_init));
            }
        }
        None
    }

    /// Coerce a scalar constant to the target type, handling bool normalization
    /// and long double promotion.
    fn coerce_scalar_const(
        &self,
        val: IrConst,
        expr: &Expr,
        base_ty: IrType,
        is_long_double_target: bool,
        is_bool_target: bool,
    ) -> IrConst {
        let src_ty = self.get_expr_type(expr);
        let val = if is_bool_target {
            val.bool_normalize()
        } else {
            self.coerce_const_to_type_with_src(val, base_ty, src_ty)
        };
        if is_long_double_target {
            Self::promote_to_long_double_with_signedness(val, src_ty)
        } else {
            val
        }
    }

    /// Promote a constant to long double, respecting signedness for integer sources.
    fn promote_to_long_double_with_signedness(val: IrConst, src_ty: IrType) -> IrConst {
        match val {
            IrConst::F64(v) => IrConst::long_double(v),
            IrConst::F32(v) => IrConst::long_double(v as f64),
            IrConst::I64(v) => {
                if src_ty.is_unsigned() {
                    IrConst::long_double_from_u64(v as u64)
                } else {
                    IrConst::long_double_from_i64(v)
                }
            }
            IrConst::I32(v) => IrConst::long_double_from_i64(v as i64),
            other => other,
        }
    }

    /// Lower a string literal expression to GlobalInit.
    ///
    /// Handles all three string literal kinds (narrow, wide, char16) in a single
    /// path. Returns None if the expression is not a string literal.
    fn lower_string_literal_init(
        &mut self,
        expr: &Expr,
        base_ty: IrType,
        is_array: bool,
    ) -> Option<GlobalInit> {
        let (s, kind) = self.extract_string_literal(expr)?;
        Some(self.string_literal_to_global_init(&s, kind, base_ty, is_array))
    }

    /// Convert a string literal to a GlobalInit based on the target type context.
    ///
    /// When initializing an array, the string is inlined (as bytes, u32s, or u16s
    /// depending on the element type). When initializing a pointer, the string is
    /// interned in .rodata and referenced by label.
    fn string_literal_to_global_init(
        &mut self,
        s: &str,
        kind: StringLitKind,
        base_ty: IrType,
        is_array: bool,
    ) -> GlobalInit {
        if is_array {
            // Inline the string into the array based on the target element type
            match base_ty {
                IrType::I8 | IrType::U8 => GlobalInit::String(s.to_string()),
                IrType::I32 | IrType::U32 => {
                    GlobalInit::WideString(s.chars().map(|c| c as u32).collect())
                }
                IrType::I16 | IrType::U16 => {
                    GlobalInit::Char16String(s.chars().map(|c| c as u16).collect())
                }
                _ => {
                    // Unexpected element type for string init - treat as pointer
                    self.intern_string_as_global_addr(s, kind)
                }
            }
        } else {
            // Pointer context: intern in .rodata and return address
            self.intern_string_as_global_addr(s, kind)
        }
    }

    /// Intern a string literal and return a GlobalAddr referencing it.
    fn intern_string_as_global_addr(&mut self, s: &str, kind: StringLitKind) -> GlobalInit {
        let label = match kind {
            StringLitKind::Narrow => self.intern_string_literal(s),
            StringLitKind::Wide => self.intern_wide_string_literal(s),
            StringLitKind::Char16 => self.intern_char16_string_literal(s),
        };
        GlobalInit::GlobalAddr(Rc::from(label))
    }

    /// Lower a compound literal used directly as an initializer value.
    fn lower_compound_literal_init(
        &mut self,
        cl_type_spec: &TypeSpecifier,
        cl_init: &Initializer,
        _base_ty: IrType,
        is_array: bool,
    ) -> GlobalInit {
        let cl_ctype = self.type_spec_to_ctype(cl_type_spec);
        let is_aggregate = matches!(
            cl_ctype,
            CType::Struct(..) | CType::Union(..) | CType::Array(..)
        );

        if !is_aggregate {
            // Scalar or pointer compound literal: create anonymous global
            return self.create_compound_literal_global(cl_type_spec, cl_init);
        }

        // When a compound literal of array type initializes a non-array variable,
        // it's array-to-pointer decay: create an anonymous global for the array
        // and use its address as the initializer value.
        let cl_is_array_type = matches!(cl_ctype, CType::Array(..));
        if cl_is_array_type && !is_array {
            return self.create_compound_literal_global(cl_type_spec, cl_init);
        }

        // Aggregate compound literal: recursively lower using its own type info.
        let cl_base_ty = self.type_spec_to_ir(cl_type_spec);
        let cl_layout = self.get_struct_layout_for_type(cl_type_spec);
        let cl_size = cl_layout.as_ref()
            .map_or_else(|| self.sizeof_type(cl_type_spec), |l| l.size);
        let cl_is_array = matches!(cl_ctype, CType::Array(..));
        self.lower_global_init(
            cl_init, cl_type_spec, cl_base_ty, cl_is_array,
            0, cl_size, &cl_layout, &[],
        )
    }
}

// =============================================================================
// List initializers (Initializer::List)
// =============================================================================

impl Lowerer {
    /// Lower an initializer list. Dispatches to specialized handlers based on
    /// the target type: brace-wrapped strings, complex arrays, struct arrays,
    /// pointer arrays, vector arrays, flat/multidim arrays, structs, vectors,
    /// and scalar-with-braces.
    fn lower_global_init_list(
        &mut self,
        items: &[InitializerItem],
        type_spec: &TypeSpecifier,
        base_ty: IrType,
        is_array: bool,
        elem_size: usize,
        total_size: usize,
        struct_layout: &Option<RcLayout>,
        array_dim_strides: &[usize],
        is_long_double_target: bool,
        is_bool_target: bool,
    ) -> GlobalInit {
        // Brace-wrapped string literal: char c[] = {"hello"}
        let is_char_not_ptr_array = elem_size <= base_ty.size().max(1);
        if is_array && is_char_not_ptr_array {
            if let Some(init) = self.try_brace_wrapped_string(items, base_ty) {
                return init;
            }
        }

        // Complex array: double _Complex arr[] = { ... }
        let complex_ctype = self.type_spec_to_ctype(type_spec);
        if is_array && complex_ctype.is_complex() {
            return self.lower_complex_array_init(items, &complex_ctype, total_size, elem_size);
        }

        // Array with elements
        if is_array && elem_size > 0 {
            return self.lower_array_init(
                items, type_spec, base_ty, elem_size, total_size,
                struct_layout, array_dim_strides,
                is_long_double_target, is_bool_target,
            );
        }

        // Struct/union initializer list
        if let Some(ref layout) = struct_layout {
            return self.lower_struct_global_init(items, layout);
        }

        // Vector initializer list: __attribute__((vector_size(N)))
        {
            let ctype = self.type_spec_to_ctype(type_spec);
            if let Some((elem_ct, num_elems)) = ctype.vector_info() {
                return self.lower_vector_init(items, elem_ct, num_elems);
            }
        }

        // Scalar with braces: int x = { 1 }; or int x = {{{1}}};
        if !is_array && !items.is_empty() {
            if let Some(init) = self.lower_scalar_with_braces(items, base_ty) {
                return init;
            }
        }

        // Fallback: array of constants coerced to base_ty
        self.lower_fallback_array(items, base_ty)
    }

    /// Try to extract a brace-wrapped string literal: `{"hello"}` for char/wchar_t/char16_t arrays.
    fn try_brace_wrapped_string(
        &mut self,
        items: &[InitializerItem],
        base_ty: IrType,
    ) -> Option<GlobalInit> {
        if items.len() != 1 || !items[0].designators.is_empty() {
            return None;
        }
        let expr = match &items[0].init {
            Initializer::Expr(e) => e,
            _ => return None,
        };
        let (s, _kind) = self.extract_string_literal(expr)?;
        // Convert string to appropriate GlobalInit based on element type
        Some(match base_ty {
            IrType::I8 | IrType::U8 => GlobalInit::String(s),
            IrType::I32 | IrType::U32 => {
                GlobalInit::WideString(s.chars().map(|c| c as u32).collect())
            }
            IrType::I16 | IrType::U16 => {
                GlobalInit::Char16String(s.chars().map(|c| c as u16).collect())
            }
            _ => return None,
        })
    }

    /// Lower a scalar value wrapped in braces: `int x = { 1 };` or `int x = {{{1}}}`.
    fn lower_scalar_with_braces(&mut self, items: &[InitializerItem], base_ty: IrType) -> Option<GlobalInit> {
        let expr = Self::unwrap_nested_init_expr(items)?;
        if let Some(val) = self.eval_const_expr(expr) {
            let expr_ty = self.get_expr_type(expr);
            return Some(GlobalInit::Scalar(self.coerce_const_to_type_with_src(val, base_ty, expr_ty)));
        }
        self.eval_global_addr_expr(expr)
    }

    /// Fallback: emit all items as an array of constants.
    fn lower_fallback_array(&self, items: &[InitializerItem], base_ty: IrType) -> GlobalInit {
        let values: Vec<IrConst> = items.iter().map(|item| {
            if let Initializer::Expr(expr) = &item.init {
                self.eval_const_expr(expr)
                    .map(|v| v.coerce_to(base_ty))
                    .unwrap_or_else(|| self.zero_const(base_ty))
            } else {
                self.zero_const(base_ty)
            }
        }).collect();
        if values.is_empty() { GlobalInit::Zero } else { GlobalInit::Array(values) }
    }
}

// =============================================================================
// Array initializers
// =============================================================================

impl Lowerer {
    /// Lower an array initializer list. Dispatches based on element type:
    /// struct arrays, pointer arrays with relocations, vector arrays,
    /// multi-dimensional arrays, and flat arrays.
    fn lower_array_init(
        &mut self,
        items: &[InitializerItem],
        type_spec: &TypeSpecifier,
        base_ty: IrType,
        elem_size: usize,
        total_size: usize,
        struct_layout: &Option<RcLayout>,
        array_dim_strides: &[usize],
        is_long_double_target: bool,
        is_bool_target: bool,
    ) -> GlobalInit {
        let num_elems = self.compute_num_elems(
            base_ty, elem_size, total_size, struct_layout, is_long_double_target,
        );

        // Struct array (byte-serialized or compound for pointer fields)
        if let Some(ref layout) = struct_layout {
            return self.lower_struct_array_init(
                items, layout, num_elems, total_size, array_dim_strides,
            );
        }

        // Pointer array or array with address expressions (needs .quad directives)
        if self.array_needs_compound_init(items, base_ty, elem_size, array_dim_strides) {
            return self.lower_pointer_array_init(
                items, base_ty, elem_size, total_size, array_dim_strides,
            );
        }

        // Vector array: e.g., __attribute__((vector_size(16))) arr[]
        {
            let ctype = self.type_spec_to_ctype(type_spec);
            if let Some((elem_ct, vec_num_elems)) = ctype.vector_info() {
                return self.lower_vector_array_init(
                    items, elem_ct, vec_num_elems, num_elems,
                );
            }
        }

        // Plain scalar array (possibly multi-dimensional)
        self.lower_scalar_array_init(
            items, base_ty, num_elems, total_size, array_dim_strides,
            is_long_double_target, is_bool_target,
        )
    }

    /// Compute the number of elements in the array.
    fn compute_num_elems(
        &self,
        base_ty: IrType,
        elem_size: usize,
        total_size: usize,
        struct_layout: &Option<RcLayout>,
        is_long_double_target: bool,
    ) -> usize {
        if struct_layout.is_some() || is_long_double_target {
            total_size / elem_size.max(1)
        } else {
            let base_type_size = base_ty.size().max(1);
            if elem_size > base_type_size {
                total_size / elem_size
            } else {
                total_size / base_type_size
            }
        }
    }

    /// Lower a struct array initializer. Uses byte-serialization for structs
    /// without pointer fields, and compound representation for structs with pointers.
    fn lower_struct_array_init(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
        num_elems: usize,
        total_size: usize,
        array_dim_strides: &[usize],
    ) -> GlobalInit {
        let has_ptr_fields = layout.has_pointer_fields(&*self.types.borrow_struct_layouts());
        if has_ptr_fields {
            if array_dim_strides.len() > 1 {
                return self.lower_struct_array_with_ptrs_multidim(
                    items, layout, total_size, array_dim_strides,
                );
            }
            return self.lower_struct_array_with_ptrs(items, layout, num_elems);
        }
        // Byte-serialize the struct array
        let struct_size = layout.size;
        let mut bytes = vec![0u8; total_size];
        self.fill_multidim_struct_array_bytes(
            items, layout, struct_size, array_dim_strides,
            &mut bytes, 0, total_size,
        );
        let values: Vec<IrConst> = bytes.iter().map(|&b| IrConst::I8(b as i8)).collect();
        GlobalInit::Array(values)
    }

    /// Check if an array initializer needs compound (relocation-aware) representation.
    /// This is needed when elements contain address expressions, string literals used
    /// as pointers, or label addresses.
    fn array_needs_compound_init(
        &self,
        items: &[InitializerItem],
        base_ty: IrType,
        elem_size: usize,
        array_dim_strides: &[usize],
    ) -> bool {
        let is_multidim_char_array = matches!(base_ty, IrType::I8 | IrType::U8)
            && array_dim_strides.len() > 1;
        let is_ptr_array = elem_size > base_ty.size().max(1);

        let has_addr_exprs = items.iter().any(|item| {
            if let Initializer::Expr(expr) = &item.init {
                // Resolve _Generic to its selected expression for detection
                let expr = if let Expr::GenericSelection(ref controlling, ref associations, _) = expr {
                    if let Some(selected) = self.resolve_generic_selection_expr(controlling, associations) {
                        selected
                    } else {
                        expr
                    }
                } else {
                    expr
                };
                if matches!(expr, Expr::StringLiteral(_, _)) {
                    return !is_multidim_char_array;
                }
                if matches!(Self::strip_casts(expr), Expr::LabelAddr(_, _)) || Self::expr_contains_label_addr(expr) {
                    return true;
                }
                let is_const = self.eval_const_expr(expr).is_some();
                if !is_const && self.eval_global_addr_expr(expr).is_some() {
                    return true;
                }
                if is_const {
                    return false;
                }
            }
            h::init_contains_addr_expr(item, is_multidim_char_array, &self.types.enum_constants)
        });

        has_addr_exprs || (is_ptr_array && !is_multidim_char_array && items.iter().any(|item| {
            h::init_contains_string_literal(item)
        }))
    }

    /// Lower a pointer array or array with address relocations.
    /// Each element becomes a separate GlobalInit in a Compound representation.
    fn lower_pointer_array_init(
        &mut self,
        items: &[InitializerItem],
        base_ty: IrType,
        elem_size: usize,
        total_size: usize,
        array_dim_strides: &[usize],
    ) -> GlobalInit {
        let base_type_size = base_ty.size().max(1);
        let flat_num_elems = total_size / base_type_size;
        let outer_stride = if array_dim_strides.len() >= 2 {
            array_dim_strides[0] / base_type_size
        } else {
            1
        };
        let mut elements: Vec<GlobalInit> = (0..flat_num_elems).map(|_| GlobalInit::Zero).collect();
        let is_structured = outer_stride > 1 && items.iter().any(|item| {
            matches!(&item.init, Initializer::List(_))
        });

        let mut current_idx = 0usize;
        for item in items {
            let index_designators: Vec<usize> = item.designators.iter().filter_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            }).collect();

            if !index_designators.is_empty() {
                current_idx = index_designators[0];
            }

            let flat_idx = if index_designators.len() > 1 {
                self.compute_flat_index_from_designators(
                    &index_designators, array_dim_strides,
                    elem_size.max(base_ty.size().max(1))
                )
            } else if is_structured {
                current_idx * outer_stride
            } else {
                current_idx
            };

            if flat_idx < flat_num_elems {
                // For inner List initializers, handle designators to place
                // elements at the correct offset within the sub-array.
                if let Initializer::List(sub_items) = &item.init {
                    if is_structured {
                        // Process inner list with designator support.
                        // Compute how many flat elements each inner sub-item spans.
                        // For arr[2][2][3] with strides [48,24,8] and base_type_size=8,
                        // inner_stride = 24/8 = 3. For 2D arrays, inner_stride = 1.
                        let inner_stride = if array_dim_strides.len() >= 2 {
                            array_dim_strides[1] / base_type_size
                        } else {
                            1
                        };
                        let mut inner_idx = 0usize;
                        for sub in sub_items {
                            // Check for inner designator
                            let inner_desig: Vec<usize> = sub.designators.iter().filter_map(|d| {
                                if let Designator::Index(ref idx_expr) = d {
                                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                                } else {
                                    None
                                }
                            }).collect();
                            if !inner_desig.is_empty() {
                                inner_idx = inner_desig[0];
                            }
                            let target_idx = flat_idx + inner_idx * inner_stride;
                            if target_idx < flat_num_elems {
                                let mut elem_parts = Vec::new();
                                self.collect_compound_init_element(&sub.init, &mut elem_parts, elem_size);
                                for (i, elem) in elem_parts.into_iter().enumerate() {
                                    if target_idx + i < flat_num_elems {
                                        let elem = match elem {
                                            GlobalInit::Scalar(val) => {
                                                GlobalInit::Scalar(val.coerce_to(base_ty))
                                            }
                                            other => other,
                                        };
                                        elements[target_idx + i] = elem;
                                    }
                                }
                            }
                            inner_idx += 1;
                        }
                    } else {
                        let mut elem_parts = Vec::new();
                        self.collect_compound_init_element(&item.init, &mut elem_parts, elem_size);
                        for (i, elem) in elem_parts.into_iter().enumerate() {
                            if flat_idx + i < flat_num_elems {
                                let elem = match elem {
                                    GlobalInit::Scalar(val) => {
                                        GlobalInit::Scalar(val.coerce_to(base_ty))
                                    }
                                    other => other,
                                };
                                elements[flat_idx + i] = elem;
                            }
                        }
                    }
                } else {
                    let mut elem_parts = Vec::new();
                    self.collect_compound_init_element(&item.init, &mut elem_parts, elem_size);
                    for (i, elem) in elem_parts.into_iter().enumerate() {
                        if flat_idx + i < flat_num_elems {
                            let elem = match elem {
                                GlobalInit::Scalar(val) => {
                                    GlobalInit::Scalar(val.coerce_to(base_ty))
                                }
                                other => other,
                            };
                            elements[flat_idx + i] = elem;
                        }
                    }
                }
            }
            current_idx += 1;
        }

        // On targets where pointer size < element type size (e.g., i686 with
        // uint64_t[]), convert to byte+pointer representation for correct padding.
        let ptr_size = crate::common::types::target_ptr_size();
        if base_type_size > ptr_size {
            return self.convert_compound_to_bytes_and_ptrs(elements, base_type_size, total_size);
        }
        GlobalInit::Compound(elements)
    }

    /// Convert compound elements to byte+pointer representation for targets where
    /// pointer size < element size.
    fn convert_compound_to_bytes_and_ptrs(
        &self,
        elements: Vec<GlobalInit>,
        base_type_size: usize,
        total_size: usize,
    ) -> GlobalInit {
        let mut bytes = vec![0u8; total_size];
        let mut ptr_ranges: Vec<(usize, GlobalInit)> = Vec::new();
        let ptr_size = crate::common::types::target_ptr_size();
        for (idx, elem) in elements.into_iter().enumerate() {
            let byte_offset = idx * base_type_size;
            match elem {
                GlobalInit::Zero => {}
                GlobalInit::Scalar(ref c) => {
                    let val_bytes = c.to_le_bytes();
                    let len = val_bytes.len().min(base_type_size);
                    bytes[byte_offset..byte_offset + len]
                        .copy_from_slice(&val_bytes[..len]);
                }
                GlobalInit::GlobalAddr(_) | GlobalInit::GlobalAddrOffset(_, _) => {
                    ptr_ranges.push((byte_offset, elem));
                }
                GlobalInit::GlobalLabelDiff(lab1, lab2, _) => {
                    ptr_ranges.push((byte_offset,
                        GlobalInit::GlobalLabelDiff(lab1, lab2, ptr_size)));
                }
                _ => {}
            }
        }
        Self::build_compound_from_bytes_and_ptrs(bytes, ptr_ranges, total_size)
    }

    /// Lower an array of complex types: `_Complex double arr[] = { ... }`.
    fn lower_complex_array_init(
        &mut self,
        items: &[InitializerItem],
        complex_ctype: &CType,
        total_size: usize,
        elem_size: usize,
    ) -> GlobalInit {
        let num_elems = total_size / elem_size.max(1);
        let total_scalars = num_elems * 2;
        let zero_pair = Self::complex_zero_pair(complex_ctype);
        let mut values: Vec<IrConst> = Vec::with_capacity(total_scalars);
        for _ in 0..num_elems {
            values.extend_from_slice(&zero_pair);
        }
        let mut current_idx = 0usize;
        for item in items {
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    current_idx = idx;
                }
            }
            if current_idx < num_elems {
                let expr = match &item.init {
                    Initializer::Expr(e) => Some(e),
                    Initializer::List(sub_items) => Self::unwrap_nested_init_expr(sub_items),
                };
                if let Some(expr) = expr {
                    if let Some((real, imag)) = self.eval_complex_const_public(expr) {
                        let base_offset = current_idx * 2;
                        let (r, i) = Self::complex_pair(complex_ctype, real, imag);
                        values[base_offset] = r;
                        values[base_offset + 1] = i;
                    }
                }
            }
            current_idx += 1;
        }
        GlobalInit::Array(values)
    }

    /// Get the zero pair for a complex type.
    fn complex_zero_pair(ctype: &CType) -> Vec<IrConst> {
        match ctype {
            CType::ComplexFloat => vec![IrConst::F32(0.0), IrConst::F32(0.0)],
            CType::ComplexLongDouble => vec![IrConst::long_double(0.0), IrConst::long_double(0.0)],
            _ => vec![IrConst::F64(0.0), IrConst::F64(0.0)],
        }
    }

    /// Create a (real, imag) pair for a complex type.
    fn complex_pair(ctype: &CType, real: f64, imag: f64) -> (IrConst, IrConst) {
        match ctype {
            CType::ComplexFloat => (IrConst::F32(real as f32), IrConst::F32(imag as f32)),
            CType::ComplexLongDouble => (IrConst::long_double(real), IrConst::long_double(imag)),
            _ => (IrConst::F64(real), IrConst::F64(imag)),
        }
    }

    /// Lower a vector array: `SV arr[] = { (SV){1,2,3,4}, (SV){5,6,7,8} }`.
    fn lower_vector_array_init(
        &mut self,
        items: &[InitializerItem],
        elem_ct: &CType,
        vec_num_elems: usize,
        num_elems: usize,
    ) -> GlobalInit {
        let elem_ir_ty = IrType::from_ctype(elem_ct);
        let total_scalars = num_elems * vec_num_elems;
        let mut values = vec![self.zero_const(elem_ir_ty); total_scalars];
        for (arr_idx, item) in items.iter().enumerate() {
            if arr_idx >= num_elems { break; }
            let base_offset = arr_idx * vec_num_elems;
            match &item.init {
                Initializer::Expr(expr) => {
                    self.collect_vector_scalars_from_expr(
                        expr, elem_ir_ty, vec_num_elems, &mut values, base_offset,
                    );
                }
                Initializer::List(sub_items) => {
                    self.collect_vector_scalars_from_items(
                        sub_items, elem_ir_ty, vec_num_elems, &mut values, base_offset,
                    );
                }
            }
        }
        GlobalInit::Array(values)
    }

    /// Extract scalar values from a vector expression (compound literal or cast-wrapped).
    fn collect_vector_scalars_from_expr(
        &self,
        expr: &Expr,
        elem_ir_ty: IrType,
        vec_num_elems: usize,
        values: &mut [IrConst],
        base_offset: usize,
    ) {
        // Unwrap casts to find the inner compound literal
        let inner = Self::strip_casts(expr);
        if let Expr::CompoundLiteral(_ts, ref cl_init, _) = inner {
            if let Initializer::List(sub_items) = cl_init.as_ref() {
                self.collect_vector_scalars_from_items(
                    sub_items, elem_ir_ty, vec_num_elems, values, base_offset,
                );
            }
        }
    }

    /// Collect scalar values from initializer items into a vector slot.
    fn collect_vector_scalars_from_items(
        &self,
        sub_items: &[InitializerItem],
        elem_ir_ty: IrType,
        vec_num_elems: usize,
        values: &mut [IrConst],
        base_offset: usize,
    ) {
        for (vi, sub) in sub_items.iter().enumerate() {
            if vi >= vec_num_elems { break; }
            if let Initializer::Expr(ref sub_expr) = sub.init {
                if let Some(val) = self.eval_const_expr(sub_expr) {
                    let expr_ty = self.get_expr_type(sub_expr);
                    values[base_offset + vi] = self.coerce_const_to_type_with_src(val, elem_ir_ty, expr_ty);
                }
            }
        }
    }

    /// Lower a plain scalar array (1D or multi-dimensional).
    fn lower_scalar_array_init(
        &mut self,
        items: &[InitializerItem],
        base_ty: IrType,
        num_elems: usize,
        total_size: usize,
        array_dim_strides: &[usize],
        is_long_double_target: bool,
        is_bool_target: bool,
    ) -> GlobalInit {
        let zero_val = self.typed_zero_const(base_ty, is_long_double_target);

        if array_dim_strides.len() > 1 {
            // Multi-dimensional array: flatten nested init lists
            let innermost_stride = array_dim_strides.last().copied().unwrap_or(1).max(1);
            let total_scalar_elems = total_size / innermost_stride;
            let mut values_flat = vec![
                self.typed_zero_const(base_ty, is_long_double_target);
                total_scalar_elems
            ];
            let mut flat = Vec::with_capacity(total_scalar_elems);
            self.flatten_global_array_init_bool(
                items, array_dim_strides, base_ty, &mut flat, is_bool_target,
            );
            for (i, v) in flat.into_iter().enumerate() {
                if i < total_scalar_elems {
                    values_flat[i] = Self::maybe_promote_long_double(v, is_long_double_target);
                }
            }
            GlobalInit::Array(values_flat)
        } else {
            // 1D array with designated initializer support
            let mut values = vec![zero_val; num_elems];
            let mut current_idx = 0usize;
            for item in items {
                if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                    if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                        current_idx = idx;
                    }
                }
                if current_idx < num_elems {
                    let val = self.eval_array_element(&item.init, base_ty, is_bool_target);
                    values[current_idx] = Self::maybe_promote_long_double(val, is_long_double_target);
                }
                current_idx += 1;
            }
            GlobalInit::Array(values)
        }
    }

    /// Evaluate a single array element from its initializer.
    fn eval_array_element(
        &self,
        init: &Initializer,
        base_ty: IrType,
        is_bool_target: bool,
    ) -> IrConst {
        match init {
            Initializer::Expr(expr) => {
                let raw = self.eval_const_expr(expr).unwrap_or(self.zero_const(base_ty));
                if is_bool_target {
                    raw.bool_normalize()
                } else {
                    let expr_ty = self.get_expr_type(expr);
                    self.coerce_const_to_type_with_src(raw, base_ty, expr_ty)
                }
            }
            Initializer::List(sub_items) => {
                let mut sub_vals = Vec::new();
                for sub in sub_items {
                    self.flatten_global_init_item(&sub.init, base_ty, &mut sub_vals);
                }
                let raw = sub_vals.into_iter().next().unwrap_or(self.zero_const(base_ty));
                if is_bool_target {
                    raw.bool_normalize()
                } else {
                    raw.coerce_to(base_ty)
                }
            }
        }
    }

    /// Lower a plain vector init list: `__attribute__((vector_size(N))) v = { 1, 2, 3, 4 }`.
    fn lower_vector_init(
        &mut self,
        items: &[InitializerItem],
        elem_ct: &CType,
        num_elems: usize,
    ) -> GlobalInit {
        let elem_ir_ty = IrType::from_ctype(elem_ct);
        let mut values = vec![self.zero_const(elem_ir_ty); num_elems];
        for (idx, item) in items.iter().enumerate() {
            if idx >= num_elems { break; }
            if let Initializer::Expr(expr) = &item.init {
                if let Some(val) = self.eval_const_expr(expr) {
                    let expr_ty = self.get_expr_type(expr);
                    values[idx] = self.coerce_const_to_type_with_src(val, elem_ir_ty, expr_ty);
                }
            }
        }
        GlobalInit::Array(values)
    }
}

// =============================================================================
// String literal helpers
// =============================================================================

impl Lowerer {
    /// Evaluate a string literal address expression for static initializers.
    /// Handles: `"hello"`, `"hello" + 2`, `"hello" - 1`, `(type*)"hello"`,
    /// `&"hello"[2]`, etc.
    pub(super) fn eval_string_literal_addr_expr(&mut self, expr: &Expr) -> Option<GlobalInit> {
        match expr {
            Expr::BinaryOp(BinOp::Add, lhs, rhs, _) => {
                self.eval_string_literal_with_offset(lhs, rhs, false)
                    .or_else(|| self.eval_string_literal_with_offset(rhs, lhs, false))
            }
            Expr::BinaryOp(BinOp::Sub, lhs, rhs, _) => {
                self.eval_string_literal_with_offset(lhs, rhs, true)
            }
            // &"string"[index] -> intern string literal + byte offset
            // This is equivalent to "string" + index, but expressed as
            // address-of array subscript on a string literal.
            Expr::AddressOf(inner, _) => {
                if let Expr::ArraySubscript(base, index, _) = inner.as_ref() {
                    self.eval_string_literal_with_offset(base, index, false)
                } else {
                    None
                }
            }
            Expr::Cast(_, inner, _) => self.eval_string_literal_addr_expr(inner),
            Expr::StringLiteral(s, _) => {
                Some(GlobalInit::GlobalAddr(Rc::from(self.intern_string_literal(s))))
            }
            Expr::WideStringLiteral(s, _) => {
                Some(GlobalInit::GlobalAddr(Rc::from(self.intern_wide_string_literal(s))))
            }
            Expr::Char16StringLiteral(s, _) => {
                Some(GlobalInit::GlobalAddr(Rc::from(self.intern_char16_string_literal(s))))
            }
            _ => None,
        }
    }

    /// Helper: intern a string literal + offset => GlobalAddr or GlobalAddrOffset.
    fn eval_string_literal_with_offset(
        &mut self,
        str_expr: &Expr,
        offset_expr: &Expr,
        negate: bool,
    ) -> Option<GlobalInit> {
        let (s, kind) = self.extract_string_literal(str_expr)?;
        let offset_val = self.eval_const_expr(offset_expr)?;
        let offset = self.const_to_i64(&offset_val)?;
        let byte_offset = if negate { -offset } else { offset };
        let label: Rc<str> = {
            let s = match kind {
                StringLitKind::Narrow => self.intern_string_literal(&s),
                StringLitKind::Wide => self.intern_wide_string_literal(&s),
                StringLitKind::Char16 => self.intern_char16_string_literal(&s),
            };
            Rc::from(s)
        };
        if byte_offset == 0 {
            Some(GlobalInit::GlobalAddr(label))
        } else {
            Some(GlobalInit::GlobalAddrOffset(label, byte_offset))
        }
    }

    /// Extract a string literal from an expression, possibly through casts.
    fn extract_string_literal(&self, expr: &Expr) -> Option<(String, StringLitKind)> {
        match expr {
            Expr::StringLiteral(s, _) => Some((s.clone(), StringLitKind::Narrow)),
            Expr::WideStringLiteral(s, _) => Some((s.clone(), StringLitKind::Wide)),
            Expr::Char16StringLiteral(s, _) => Some((s.clone(), StringLitKind::Char16)),
            Expr::Cast(_, inner, _) => self.extract_string_literal(inner),
            _ => None,
        }
    }
}

// =============================================================================
// Compound literal globals
// =============================================================================

impl Lowerer {
    /// Create an anonymous global for a compound literal at file scope.
    pub(super) fn create_compound_literal_global(
        &mut self,
        type_spec: &TypeSpecifier,
        init: &Initializer,
    ) -> GlobalInit {
        let label_str = format!(".Lcompound_lit_{}", self.next_anon_struct);
        self.next_anon_struct += 1;
        let label: Rc<str> = Rc::from(label_str);

        let is_array = matches!(type_spec, TypeSpecifier::Array(_, _));
        let (elem_size, base_ty, computed_alloc_size) = if let TypeSpecifier::Array(ref elem_ts, _) = type_spec {
            let elem_ir_ty = self.type_spec_to_ir(elem_ts);
            let e_size = self.sizeof_type(elem_ts);
            let num_elems = if let Initializer::List(items) = init {
                items.len()
            } else {
                1
            };
            (e_size, elem_ir_ty, e_size * num_elems)
        } else {
            let ty = self.type_spec_to_ir(type_spec);
            let size = self.sizeof_type(type_spec);
            (0, ty, size)
        };

        let struct_layout = self.get_struct_layout_for_type(type_spec);
        // For arrays, struct_layout is for the element type, not the whole array,
        // so always use computed_alloc_size which accounts for the element count.
        let alloc_size = if is_array {
            computed_alloc_size
        } else {
            struct_layout.as_ref().map_or(computed_alloc_size, |l| l.size)
        };
        let align = struct_layout.as_ref()
            .map_or(base_ty.align(), |l| l.align.max(base_ty.align()));

        let global_init = self.lower_global_init(
            init, type_spec, base_ty, is_array, elem_size, alloc_size, &struct_layout, &[],
        );

        let global_ty = if matches!(&global_init, GlobalInit::Array(vals) if !vals.is_empty() && matches!(vals[0], IrConst::I8(_)))
            || (struct_layout.is_some() && matches!(global_init, GlobalInit::Array(_)))
        {
            IrType::I8
        } else {
            base_ty
        };

        self.emitted_global_names.insert(label.clone());
        self.module.globals.push(IrGlobal {
            name: label.clone(),
            ty: global_ty,
            size: alloc_size,
            align,
            init: global_init,
            is_static: true,
            is_extern: false,
            is_common: false,
            section: None,
            is_weak: false,
            visibility: None,
            has_explicit_align: false,
            is_const: false,
            is_used: false,
            is_thread_local: false,
        });

        GlobalInit::GlobalAddr(label)
    }
}

// =============================================================================
// Struct initializer helpers
// =============================================================================

impl Lowerer {
    fn lower_struct_global_init(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
    ) -> GlobalInit {
        let has_addr_fields = self.struct_init_has_addr_fields(items, layout);
        if has_addr_fields {
            return self.lower_struct_global_init_compound(items, layout);
        }

        let total_size = layout.size + self.compute_fam_extra_size(items, layout);
        let mut bytes = vec![0u8; total_size];
        self.fill_struct_global_bytes(items, layout, &mut bytes, 0);
        let values: Vec<IrConst> = bytes.iter().map(|&b| IrConst::I8(b as i8)).collect();
        GlobalInit::Array(values)
    }

    /// Compute extra bytes needed for a flexible array member (FAM).
    pub(super) fn compute_fam_extra_size(&self, items: &[InitializerItem], layout: &StructLayout) -> usize {
        let last_field = match layout.fields.last() {
            Some(f) => f,
            None => return 0,
        };
        let (elem_ty, last_field_idx) = match &last_field.ty {
            CType::Array(ref elem_ty, None) => (elem_ty, layout.fields.len() - 1),
            _ => return 0,
        };
        let elem_size = self.resolve_ctype_size(elem_ty);
        let mut current_field_idx = 0usize;
        for (item_idx, item) in items.iter().enumerate() {
            let field_idx = self.resolve_struct_init_field_idx(item, layout, current_field_idx);
            if field_idx == last_field_idx {
                let num_elems = match &item.init {
                    Initializer::List(sub_items) => sub_items.len(),
                    Initializer::Expr(Expr::StringLiteral(s, _)) => {
                        if matches!(elem_ty.as_ref(), CType::Char | CType::UChar) {
                            s.len() + 1
                        } else {
                            items.len() - item_idx
                        }
                    }
                    Initializer::Expr(_) => items.len() - item_idx,
                };
                return num_elems * elem_size;
            }
            if field_idx < layout.fields.len() {
                current_field_idx = field_idx + 1;
            }
        }
        0
    }

    /// Check if a struct initializer contains fields needing address relocations.
    pub(super) fn struct_init_has_addr_fields(&self, items: &[InitializerItem], layout: &StructLayout) -> bool {
        let mut current_field_idx = 0usize;
        let mut item_idx = 0usize;

        while item_idx < items.len() {
            let item = &items[item_idx];
            let field_idx = self.resolve_struct_init_field_idx(item, layout, current_field_idx);

            if field_idx >= layout.fields.len() {
                // Item is beyond all fields. Still check if it has address expressions
                // since brace-elided counting may have caused us to skip past the layout.
                if self.init_has_addr_exprs(&item.init) {
                    return true;
                }
                item_idx += 1;
                continue;
            }

            let field_ty = &layout.fields[field_idx].ty;

            if h::has_nested_field_designator(item) {
                if self.nested_designator_has_addr_fields(item, field_ty) {
                    return true;
                }
                current_field_idx = field_idx + 1;
                item_idx += 1;
                continue;
            }

            if self.field_init_has_addr_refs(item, field_ty, items, item_idx) {
                return true;
            }

            // For brace-elided sub-struct fields, advance by the number of
            // scalar leaf fields, checking each consumed item for address exprs.
            if matches!(field_ty, CType::Struct(_) | CType::Union(_))
                && matches!(&item.init, Initializer::Expr(_))
                && item.designators.is_empty()
            {
                let scalars_needed = h::count_flat_init_scalars(field_ty, &*self.types.borrow_struct_layouts());
                if scalars_needed > 1 {
                    // Check remaining consumed items for address expressions
                    for offset in 1..scalars_needed {
                        let idx = item_idx + offset;
                        if idx >= items.len() { break; }
                        if !items[idx].designators.is_empty() { break; }
                        if self.init_has_addr_exprs(&items[idx].init) {
                            return true;
                        }
                    }
                    item_idx += scalars_needed;
                    current_field_idx = field_idx + 1;
                    continue;
                }
            }

            // Also for array fields with flat init, advance by the total scalars count
            if let CType::Array(elem_ty, Some(arr_size)) = field_ty {
                if matches!(&item.init, Initializer::Expr(_))
                    && item.designators.is_empty()
                {
                    let is_string_literal = matches!(&item.init, Initializer::Expr(Expr::StringLiteral(..)));
                    let is_char_array = matches!(elem_ty.as_ref(), CType::Char | CType::UChar);
                    if !(is_string_literal && is_char_array) {
                        let scalars_per_elem = h::count_flat_init_scalars(elem_ty, &*self.types.borrow_struct_layouts());
                        let total_scalars = arr_size * scalars_per_elem;
                        // Check all consumed items for address expressions
                        for offset in 0..total_scalars {
                            let idx = item_idx + offset;
                            if idx >= items.len() { break; }
                            if !items[idx].designators.is_empty() && offset > 0 { break; }
                            if self.init_has_addr_exprs(&items[idx].init) {
                                return true;
                            }
                        }
                        item_idx += total_scalars.min(items.len() - item_idx);
                        current_field_idx = field_idx + 1;
                        continue;
                    }
                }
            }

            current_field_idx = field_idx + 1;
            item_idx += 1;
        }
        false
    }

    /// Check if a single field's initializer contains address references.
    fn field_init_has_addr_refs(
        &self,
        item: &InitializerItem,
        field_ty: &CType,
        items: &[InitializerItem],
        item_idx: usize,
    ) -> bool {
        match &item.init {
            Initializer::Expr(expr) => {
                if Self::expr_contains_label_addr(expr) {
                    return true;
                }
                if h::expr_contains_string_literal(expr)
                    && h::type_has_pointer_elements(field_ty, &*self.types.borrow_struct_layouts())
                {
                    return true;
                }
                if self.eval_const_expr(expr).is_none() && self.eval_global_addr_expr(expr).is_some() {
                    return true;
                }
                if let Expr::AddressOf(inner, _) = expr {
                    if matches!(inner.as_ref(), Expr::CompoundLiteral(..)) {
                        return true;
                    }
                }
                // Bare compound literal initializing a pointer field (array-to-pointer decay)
                if let Expr::CompoundLiteral(..) = expr {
                    if matches!(field_ty, CType::Pointer(_, _)) {
                        return true;
                    }
                }
                // Check flat array elements following this item
                if let CType::Array(elem_ty, Some(arr_size)) = field_ty {
                    if h::type_has_pointer_elements(elem_ty, &*self.types.borrow_struct_layouts()) {
                        for i in 1..*arr_size {
                            let next = item_idx + i;
                            if next >= items.len() { break; }
                            if !items[next].designators.is_empty() { break; }
                            if self.init_has_addr_exprs(&items[next].init) {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            Initializer::List(nested_items) => {
                if h::type_has_pointer_elements(field_ty, &*self.types.borrow_struct_layouts())
                    && self.init_has_addr_exprs(&item.init)
                {
                    return true;
                }
                if let Some(nested_layout) = self.get_struct_layout_for_ctype(field_ty) {
                    if self.struct_init_has_addr_fields(nested_items, &nested_layout) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Check multi-level designated initializer for address fields.
    fn nested_designator_has_addr_fields(&self, item: &InitializerItem, outer_ty: &CType) -> bool {
        let drill = match self.drill_designators(&item.designators[1..], outer_ty) {
            Some(d) => d,
            None => return false,
        };
        let current_ty = drill.target_ty;

        if matches!(&current_ty, CType::Pointer(_, _) | CType::Function(_)) {
            return self.init_has_addr_exprs(&item.init);
        }

        if let Some(target_layout) = self.get_struct_layout_for_ctype(&current_ty) {
            if target_layout.has_pointer_fields(&*self.types.borrow_struct_layouts())
                && self.init_has_addr_exprs(&item.init)
            {
                return true;
            }
            if let Initializer::List(nested_items) = &item.init {
                if self.struct_init_has_addr_fields(nested_items, &target_layout) {
                    return true;
                }
            }
        }

        if let CType::Array(elem_ty, _) = &current_ty {
            if h::type_has_pointer_elements(elem_ty, &*self.types.borrow_struct_layouts()) {
                return self.init_has_addr_exprs(&item.init);
            }
        }

        false
    }

    /// Check if an initializer contains address expressions (recursive).
    fn init_has_addr_exprs(&self, init: &Initializer) -> bool {
        match init {
            Initializer::Expr(expr) => {
                if Self::expr_contains_label_addr(expr) {
                    return true;
                }
                if h::expr_contains_string_literal(expr) {
                    return true;
                }
                if let Expr::AddressOf(inner, _) = expr {
                    if matches!(inner.as_ref(), Expr::CompoundLiteral(..)) {
                        return true;
                    }
                }
                if let Expr::CompoundLiteral(_, ref cl_init, _) = expr {
                    return self.init_has_addr_exprs(cl_init);
                }
                self.eval_const_expr(expr).is_none() && self.eval_global_addr_expr(expr).is_some()
            }
            Initializer::List(items) => {
                items.iter().any(|item| self.init_has_addr_exprs(&item.init))
            }
        }
    }
}

// =============================================================================
// Array flattening (multi-dimensional designated initializers)
// =============================================================================

impl Lowerer {
    /// Write values from `src` into `dest` starting at `start_idx`, overwriting
    /// existing entries or extending the vector as needed.
    fn overwrite_or_extend_values(
        dest: &mut Vec<IrConst>,
        start_idx: usize,
        src: Vec<IrConst>,
        zero: &dyn Fn() -> IrConst,
    ) {
        for (i, v) in src.into_iter().enumerate() {
            let pos = start_idx + i;
            if pos < dest.len() {
                dest[pos] = v;
            } else {
                while dest.len() < pos {
                    dest.push(zero());
                }
                dest.push(v);
            }
        }
    }

    /// Inline a string literal into a values array for a sub-array element.
    fn inline_string_to_values(
        &self, s: &str, sub_elem_count: usize, base_ty: IrType, values: &mut Vec<IrConst>,
    ) {
        let start_len = values.len();
        for c in s.chars() {
            values.push(IrConst::I64(c as u8 as i64));
        }
        if values.len() < start_len + sub_elem_count {
            values.push(IrConst::I64(0));
        }
        while values.len() < start_len + sub_elem_count {
            values.push(self.zero_const(base_ty));
        }
        if values.len() > start_len + sub_elem_count {
            values.truncate(start_len + sub_elem_count);
        }
    }

    fn flatten_global_array_init_bool(
        &self,
        items: &[InitializerItem],
        array_dim_strides: &[usize],
        base_ty: IrType,
        values: &mut Vec<IrConst>,
        is_bool_target: bool,
    ) {
        let base_type_size = base_ty.size().max(1);
        if array_dim_strides.len() <= 1 {
            self.flatten_1d_array(items, base_ty, values, is_bool_target);
            return;
        }

        let sub_elem_count = if array_dim_strides[0] > 0 && base_type_size > 0 {
            array_dim_strides[0] / base_type_size
        } else {
            1
        };
        let start_len = values.len();
        let mut current_outer_idx = 0usize;

        for item in items {
            let index_designators: Vec<usize> = item.designators.iter().filter_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            }).collect();

            if !index_designators.is_empty() {
                self.flatten_designated_item(
                    item, &index_designators, array_dim_strides, base_ty,
                    base_type_size, sub_elem_count, values, is_bool_target,
                );
                current_outer_idx = index_designators[0] + 1;
                continue;
            }

            // Sequential processing (no designator)
            self.flatten_sequential_item(
                item, array_dim_strides, base_ty, sub_elem_count,
                start_len, &mut current_outer_idx, values, is_bool_target,
            );
        }
    }

    /// Flatten a 1D array with designated initializer support.
    fn flatten_1d_array(
        &self,
        items: &[InitializerItem],
        base_ty: IrType,
        values: &mut Vec<IrConst>,
        is_bool_target: bool,
    ) {
        let mut current_idx = 0usize;
        for item in items {
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    current_idx = idx;
                }
            }
            // Pad up to current_idx if needed (forward jump)
            while values.len() < current_idx {
                values.push(self.zero_const(base_ty));
            }
            if current_idx < values.len() {
                // Backward jump or overwrite
                let mut tmp = Vec::new();
                self.flatten_global_init_item_bool(&item.init, base_ty, &mut tmp, is_bool_target);
                let n = tmp.len();
                let zero_fn = || self.zero_const(base_ty);
                Self::overwrite_or_extend_values(values, current_idx, tmp, &zero_fn);
                current_idx += n.max(1);
            } else {
                // Forward: append at end
                self.flatten_global_init_item_bool(&item.init, base_ty, values, is_bool_target);
                current_idx = values.len();
            }
        }
    }

    /// Flatten a designated item in a multi-dimensional array.
    fn flatten_designated_item(
        &self,
        item: &InitializerItem,
        index_designators: &[usize],
        array_dim_strides: &[usize],
        base_ty: IrType,
        base_type_size: usize,
        sub_elem_count: usize,
        values: &mut Vec<IrConst>,
        is_bool_target: bool,
    ) {
        let flat_idx = self.compute_flat_index_from_designators(
            index_designators, array_dim_strides, base_type_size
        );
        // Pad up to flat_idx if needed
        while values.len() <= flat_idx {
            values.push(self.zero_const(base_ty));
        }

        let zero_fn = || self.zero_const(base_ty);
        match &item.init {
            Initializer::List(sub_items) => {
                let remaining_dims = array_dim_strides.len().saturating_sub(index_designators.len());
                let sub_strides = &array_dim_strides[array_dim_strides.len() - remaining_dims..];
                let mut tmp = Vec::new();
                if sub_strides.is_empty() || remaining_dims == 0 {
                    self.flatten_global_init_item_bool(&item.init, base_ty, &mut tmp, is_bool_target);
                } else {
                    self.flatten_global_array_init_bool(sub_items, sub_strides, base_ty, &mut tmp, is_bool_target);
                }
                Self::overwrite_or_extend_values(values, flat_idx, tmp, &zero_fn);
            }
            Initializer::Expr(expr) => {
                if let Expr::StringLiteral(s, _) = expr {
                    // For designated string literal in a multi-dim char array:
                    // e.g., char arr[2][5] = { [0] = "ABCD" }
                    // index_designators = [0], array_dim_strides = [5, 1]
                    // We need to fill 5 chars (the inner array size).
                    // Use the stride at the designator level (strides[0] = 5), not after it.
                    let string_sub_count = if !index_designators.is_empty()
                        && index_designators.len() < array_dim_strides.len()
                        && base_type_size > 0
                    {
                        // Stride at the last designator level gives the size of the target element
                        array_dim_strides[index_designators.len() - 1] / base_type_size
                    } else if index_designators.len() < array_dim_strides.len() {
                        sub_elem_count
                    } else {
                        1
                    };
                    let mut tmp = Vec::new();
                    self.inline_string_to_values(s, string_sub_count, base_ty, &mut tmp);
                    Self::overwrite_or_extend_values(values, flat_idx, tmp, &zero_fn);
                } else if let Some(val) = self.eval_const_expr(expr) {
                    values[flat_idx] = if is_bool_target {
                        val.bool_normalize()
                    } else {
                        let expr_ty = self.get_expr_type(expr);
                        self.coerce_const_to_type_with_src(val, base_ty, expr_ty)
                    };
                }
            }
        }
    }

    /// Flatten a sequential (non-designated) item in a multi-dimensional array.
    fn flatten_sequential_item(
        &self,
        item: &InitializerItem,
        array_dim_strides: &[usize],
        base_ty: IrType,
        sub_elem_count: usize,
        start_len: usize,
        current_outer_idx: &mut usize,
        values: &mut Vec<IrConst>,
        is_bool_target: bool,
    ) {
        match &item.init {
            Initializer::List(sub_items) => {
                let target_start = start_len + *current_outer_idx * sub_elem_count;
                while values.len() < target_start {
                    values.push(self.zero_const(base_ty));
                }
                // Check for braced string literal initializing a char sub-array
                if sub_items.len() == 1 {
                    if let Initializer::Expr(Expr::StringLiteral(s, _)) = &sub_items[0].init {
                        self.inline_string_to_values(s, sub_elem_count, base_ty, values);
                        *current_outer_idx += 1;
                        return;
                    }
                }
                let mut sub_values = Vec::with_capacity(sub_elem_count);
                self.flatten_global_array_init_bool(
                    sub_items, &array_dim_strides[1..], base_ty, &mut sub_values, is_bool_target,
                );
                while sub_values.len() < sub_elem_count {
                    sub_values.push(self.zero_const(base_ty));
                }
                values.extend(sub_values.into_iter().take(sub_elem_count));
                *current_outer_idx += 1;
            }
            Initializer::Expr(expr) => {
                if let Expr::StringLiteral(s, _) = expr {
                    let target_start = start_len + *current_outer_idx * sub_elem_count;
                    while values.len() < target_start {
                        values.push(self.zero_const(base_ty));
                    }
                    self.inline_string_to_values(s, sub_elem_count, base_ty, values);
                    *current_outer_idx += 1;
                } else if let Some(val) = self.eval_const_expr(expr) {
                    values.push(if is_bool_target {
                        val.bool_normalize()
                    } else {
                        let expr_ty = self.get_expr_type(expr);
                        self.coerce_const_to_type_with_src(val, base_ty, expr_ty)
                    });
                    let relative_pos = values.len() - start_len;
                    if sub_elem_count > 0 {
                        *current_outer_idx = relative_pos.div_ceil(sub_elem_count);
                    }
                } else {
                    values.push(self.zero_const(base_ty));
                    let relative_pos = values.len() - start_len;
                    if sub_elem_count > 0 {
                        *current_outer_idx = relative_pos.div_ceil(sub_elem_count);
                    }
                }
            }
        }
    }

    /// Compute a flat index from multi-dimensional designator indices.
    fn compute_flat_index_from_designators(
        &self,
        indices: &[usize],
        array_dim_strides: &[usize],
        base_type_size: usize,
    ) -> usize {
        let mut flat_idx = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            let elems_per_entry = if i < array_dim_strides.len() && base_type_size > 0 {
                array_dim_strides[i] / base_type_size
            } else {
                1
            };
            flat_idx += idx * elems_per_entry;
        }
        flat_idx
    }
}

// =============================================================================
// Misc helpers
// =============================================================================

impl Lowerer {
    /// Promote an IrConst value to LongDouble.
    fn promote_to_long_double(val: IrConst) -> IrConst {
        match val {
            IrConst::F64(v) => IrConst::long_double(v),
            IrConst::F32(v) => IrConst::long_double(v as f64),
            IrConst::I64(v) => IrConst::long_double_from_i64(v),
            IrConst::I32(v) => IrConst::long_double_from_i64(v as i64),
            IrConst::I16(v) => IrConst::long_double_from_i64(v as i64),
            IrConst::I8(v) => IrConst::long_double_from_i64(v as i64),
            other => other,
        }
    }

    /// Conditionally promote a value to long double.
    fn maybe_promote_long_double(val: IrConst, is_long_double: bool) -> IrConst {
        if is_long_double { Self::promote_to_long_double(val) } else { val }
    }

    /// Get the appropriate zero constant for a type, considering long double.
    fn typed_zero_const(&self, base_ty: IrType, is_long_double: bool) -> IrConst {
        if is_long_double { IrConst::long_double(0.0) } else { self.zero_const(base_ty) }
    }

    /// Collect a single compound initializer element, handling nested lists.
    fn collect_compound_init_element(
        &mut self,
        init: &Initializer,
        elements: &mut Vec<GlobalInit>,
        elem_size: usize,
    ) {
        match init {
            Initializer::Expr(expr) => {
                // Resolve _Generic selections to their selected expression
                // before any other processing.
                let expr = if let Expr::GenericSelection(ref controlling, ref associations, _) = expr {
                    if let Some(selected) = self.resolve_generic_selection_expr(controlling, associations) {
                        selected
                    } else {
                        expr
                    }
                } else {
                    expr
                };
                if let Expr::StringLiteral(s, _) = expr {
                    let label = self.intern_string_literal(s);
                    elements.push(GlobalInit::GlobalAddr(Rc::from(label)));
                } else if let Expr::LabelAddr(label_name, _) = Self::strip_casts(expr) {
                    let scoped_label = self.get_or_create_user_label(label_name);
                    if let Some(ref mut fs) = self.func_state {
                        fs.global_init_label_blocks.push(scoped_label);
                    }
                    elements.push(GlobalInit::GlobalAddr(Rc::from(scoped_label.as_label())));
                // &(compound_literal) or cast-wrapped variant -> materialize and take address
                } else if let Some(addr) = self.try_address_of_compound_literal(expr) {
                    elements.push(addr);
                } else if let Some(val) = self.eval_const_expr(expr) {
                    let val = match val {
                        IrConst::I32(v) if elem_size == 8 => IrConst::I64(v as i64),
                        IrConst::I16(v) if elem_size >= 4 => {
                            if elem_size == 8 { IrConst::I64(v as i64) } else { IrConst::I32(v as i32) }
                        }
                        other => other,
                    };
                    elements.push(GlobalInit::Scalar(val));
                } else if let Some(label_diff) = self.eval_label_diff_expr(expr, elem_size) {
                    elements.push(label_diff);
                } else if let Some(addr) = self.eval_string_literal_addr_expr(expr) {
                    elements.push(addr);
                } else {
                    // Pre-materialize compound literals in arithmetic before resolving
                    self.materialize_compound_literals_in_expr(expr);
                    if let Some(addr) = self.eval_global_addr_expr(expr) {
                        elements.push(addr);
                    } else {
                        elements.push(GlobalInit::Zero);
                    }
                }
            }
            Initializer::List(sub_items) => {
                if sub_items.is_empty() {
                    elements.push(GlobalInit::Zero);
                } else {
                    for sub in sub_items {
                        self.collect_compound_init_element(&sub.init, elements, elem_size);
                    }
                }
            }
        }
    }

    /// Try to evaluate a label difference: `&&lab1 - &&lab2`.
    fn eval_label_diff_expr(&mut self, expr: &Expr, byte_size: usize) -> Option<GlobalInit> {
        self.func_state.as_ref()?;
        if let Expr::BinaryOp(BinOp::Sub, lhs, rhs, _) = expr {
            let lhs_inner = Self::strip_casts(lhs);
            let rhs_inner = Self::strip_casts(rhs);
            if let (Expr::LabelAddr(lab1, _), Expr::LabelAddr(lab2, _)) = (lhs_inner, rhs_inner) {
                let scoped1 = self.get_or_create_user_label(lab1);
                let scoped2 = self.get_or_create_user_label(lab2);
                if let Some(ref mut fs) = self.func_state {
                    fs.global_init_label_blocks.push(scoped1);
                    fs.global_init_label_blocks.push(scoped2);
                }
                return Some(GlobalInit::GlobalLabelDiff(
                    Rc::from(scoped1.as_label()),
                    Rc::from(scoped2.as_label()),
                    byte_size,
                ));
            }
        }
        None
    }

    /// Strip cast expressions to find the underlying expression.
    pub(super) fn strip_casts(expr: &Expr) -> &Expr {
        match expr {
            Expr::Cast(_, inner, _) => Self::strip_casts(inner),
            _ => expr,
        }
    }

    /// Check if an expression tree contains label addresses.
    fn expr_contains_label_addr(expr: &Expr) -> bool {
        match expr {
            Expr::LabelAddr(_, _) => true,
            Expr::BinaryOp(_, lhs, rhs, _) => {
                Self::expr_contains_label_addr(lhs) || Self::expr_contains_label_addr(rhs)
            }
            Expr::Cast(_, inner, _) => Self::expr_contains_label_addr(inner),
            _ => false,
        }
    }

    /// Flatten a single initializer item, recursing into nested lists.
    fn flatten_global_init_item(&self, init: &Initializer, base_ty: IrType, values: &mut Vec<IrConst>) {
        self.flatten_global_init_item_bool(init, base_ty, values, false)
    }

    /// Flatten a single initializer item with _Bool awareness.
    fn flatten_global_init_item_bool(
        &self,
        init: &Initializer,
        base_ty: IrType,
        values: &mut Vec<IrConst>,
        is_bool_target: bool,
    ) {
        match init {
            Initializer::Expr(expr) => {
                if let Expr::StringLiteral(s, _) = expr {
                    for c in s.chars() {
                        values.push(IrConst::I64(c as u8 as i64));
                    }
                    values.push(IrConst::I64(0));
                } else if let Some(val) = self.eval_const_expr(expr) {
                    values.push(if is_bool_target {
                        val.bool_normalize()
                    } else {
                        let expr_ty = self.get_expr_type(expr);
                        self.coerce_const_to_type_with_src(val, base_ty, expr_ty)
                    });
                } else {
                    values.push(self.zero_const(base_ty));
                }
            }
            Initializer::List(items) => {
                for item in items {
                    self.flatten_global_init_item_bool(&item.init, base_ty, values, is_bool_target);
                }
            }
        }
    }
}
