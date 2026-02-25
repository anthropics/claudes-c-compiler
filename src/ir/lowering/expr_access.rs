//! Expression access lowering: cast, compound literals, sizeof, generic selection,
//! address-of, dereference, array subscript, member access, statement expressions,
//! and va_arg.
//!
//! Extracted from expr.rs to keep expression lowering manageable.

use crate::frontend::parser::ast::{
    BlockItem,
    CompoundStmt,
    Designator,
    Expr,
    GenericAssociation,
    Initializer,
    InitializerItem,
    SizeofArg,
    Stmt,
    TypeSpecifier,
};
use crate::ir::reexports::{
    Instruction,
    IrBinOp,
    IrCmpOp,
    IrConst,
    Operand,
    Terminator,
    Value,
};
use std::rc::Rc;
use crate::common::types::{AddressSpace, IrType, CType};
use super::lower::Lowerer;

impl Lowerer {
    // -----------------------------------------------------------------------
    // Cast expressions
    // -----------------------------------------------------------------------

    pub(super) fn lower_cast(&mut self, target_type: &TypeSpecifier, inner: &Expr) -> Operand {
        let target_ctype = self.type_spec_to_ctype(target_type);
        let inner_ctype = self.expr_ctype(inner);

        // Handle complex type casts
        if target_ctype.is_complex() && !inner_ctype.is_complex() {
            let val = self.lower_expr(inner);
            return self.real_to_complex(val, &inner_ctype, &target_ctype);
        }
        if target_ctype.is_complex() && inner_ctype.is_complex() {
            if target_ctype == inner_ctype {
                return self.lower_expr(inner);
            }
            let val = self.lower_expr(inner);
            let ptr = self.operand_to_value(val);
            return self.complex_to_complex(ptr, &inner_ctype, &target_ctype);
        }
        if !target_ctype.is_complex() && inner_ctype.is_complex() {
            let val = self.lower_expr(inner);
            let ptr = self.operand_to_value(val);

            if target_ctype == CType::Bool {
                return self.lower_complex_to_bool(ptr, &inner_ctype);
            }

            let real = self.load_complex_real(ptr, &inner_ctype);
            let comp_ty = Self::complex_component_ir_type(&inner_ctype);
            let to_ty = self.type_spec_to_ir(target_type);
            if comp_ty != to_ty {
                let dest = self.emit_cast_val(real, comp_ty, to_ty);
                return Operand::Value(dest);
            }
            return real;
        }

        // GCC extension: cast to union type, e.g. (union convert)x
        // Creates a temporary union, stores the value into the first matching member at offset 0.
        if let CType::Union(ref key) = target_ctype {
            let union_size = self.types.borrow_struct_layouts().get(&**key).map(|l| l.size).unwrap_or(0);
            let inner_ctype = self.expr_ctype(inner);
            let src = self.lower_expr(inner);

            // Allocate stack space for the union and zero-initialize it
            let alloca = self.fresh_value();
            self.emit(Instruction::Alloca { dest: alloca, size: union_size, ty: IrType::Ptr, align: 0, volatile: false });
            self.zero_init_alloca(alloca, union_size);

            // If the source is an aggregate (struct/union), lower_expr returns a pointer
            // to the data. We need to memcpy the data, not store the pointer value.
            match inner_ctype {
                CType::Struct(_) | CType::Union(_) => {
                    let src_size = self.ctype_size(&inner_ctype);
                    let copy_size = src_size.min(union_size);
                    if copy_size > 0 {
                        let src_val = self.operand_to_value(src);
                        self.emit(Instruction::Memcpy { dest: alloca, src: src_val, size: copy_size });
                    }
                }
                _ => {
                    // Scalar source: store the value directly at offset 0
                    let store_ty = self.get_expr_type(inner);
                    self.emit(Instruction::Store { val: src, ptr: alloca, ty: store_ty, seg_override: AddressSpace::Default });
                }
            }

            return Operand::Value(alloca);
        }

        // GCC extension: cast to struct type (rare, but handle similarly)
        if let CType::Struct(_) = target_ctype {
            // Struct casts are not standard C, but if the source is already a struct pointer, pass through
            let src = self.lower_expr(inner);
            return src;
        }

        // Vector-to-scalar cast: load the vector data as the target scalar type.
        // Vectors are stored in stack allocas and lower_expr returns a pointer.
        // A cast like `(long long)(V2SI){2,2}` must load the 8 bytes of vector
        // data as an integer, not return the alloca pointer itself.
        if inner_ctype.is_vector() && !target_ctype.is_vector() {
            let src = self.lower_expr(inner);
            let ptr = self.operand_to_value(src);
            let to_ty = self.type_spec_to_ir(target_type);
            let dest = self.fresh_value();
            self.emit(Instruction::Load { dest, ptr, ty: to_ty, seg_override: AddressSpace::Default });
            return Operand::Value(dest);
        }

        // Scalar-to-vector cast: store the scalar value into a vector-sized alloca.
        // A cast like `(V2SI)(long long)val` stores the 8-byte integer into an
        // alloca so the result is a pointer to vector data (matching vector repr).
        if !inner_ctype.is_vector() && target_ctype.is_vector() {
            let vec_size = self.ctype_size(&target_ctype);
            let src = self.lower_expr(inner);
            let from_ty = self.get_expr_type(inner);

            let alloca = self.fresh_value();
            self.emit(Instruction::Alloca { dest: alloca, size: vec_size, ty: IrType::Ptr, align: vec_size, volatile: false });
            // Zero-initialize in case source is smaller than vector
            self.zero_init_alloca(alloca, vec_size);
            // Store scalar at offset 0 (bitwise reinterpretation)
            self.emit(Instruction::Store { val: src, ptr: alloca, ty: from_ty, seg_override: AddressSpace::Default });
            return Operand::Value(alloca);
        }

        // Vector-to-vector cast: memcpy between allocas if sizes match.
        if inner_ctype.is_vector() && target_ctype.is_vector() {
            let src = self.lower_expr(inner);
            let src_size = self.ctype_size(&inner_ctype);
            let dst_size = self.ctype_size(&target_ctype);
            if src_size == dst_size {
                // Same size: reinterpret cast, just pass through the pointer
                return src;
            }
            // Different sizes: allocate new vector, memcpy what fits
            let src_val = self.operand_to_value(src);
            let alloca = self.fresh_value();
            self.emit(Instruction::Alloca { dest: alloca, size: dst_size, ty: IrType::Ptr, align: dst_size, volatile: false });
            self.zero_init_alloca(alloca, dst_size);
            let copy_size = src_size.min(dst_size);
            self.emit(Instruction::Memcpy { dest: alloca, src: src_val, size: copy_size });
            return Operand::Value(alloca);
        }

        let src = self.lower_expr(inner);
        let mut from_ty = self.get_expr_type(inner);
        let to_ty = self.type_spec_to_ir(target_type);

        if let Expr::Identifier(name, _) = inner {
            if let Some(vi) = self.lookup_var_info(name) {
                if vi.is_array || vi.is_struct {
                    from_ty = IrType::Ptr;
                }
            }
        }

        // C standard: conversion to _Bool yields 0 or 1
        if target_ctype == CType::Bool {
            if from_ty.is_float() {
                let zero = match from_ty {
                    IrType::F32 => Operand::Const(IrConst::F32(0.0)),
                    IrType::F64 => Operand::Const(IrConst::F64(0.0)),
                    IrType::F128 => Operand::Const(IrConst::long_double(0.0)),
                    _ => Operand::Const(IrConst::F64(0.0)),
                };
                let dest = self.emit_cmp_val(IrCmpOp::Ne, src, zero, from_ty);
                return Operand::Value(dest);
            }
            return self.emit_bool_normalize_typed(src, from_ty);
        }

        if to_ty == from_ty
            || to_ty == IrType::Ptr
            || (from_ty == IrType::Ptr && to_ty.size() == crate::common::types::target_ptr_size())
        {
            return src;
        }

        let dest = self.emit_cast_val(src, from_ty, to_ty);
        Operand::Value(dest)
    }

    // -----------------------------------------------------------------------
    // Compound literals
    // -----------------------------------------------------------------------

    /// Compute the byte size of a compound literal, handling incomplete array types.
    /// For incomplete arrays (e.g., `(int[]){1,2,3}`), sizeof_type returns 0, so we
    /// compute the size from element_size * initializer_count instead.
    fn compound_literal_size(&self, type_spec: &TypeSpecifier, init: &Initializer) -> usize {
        let ctype = self.type_spec_to_ctype(type_spec);
        match (&ctype, init) {
            (CType::Array(ref elem_ct, None), Initializer::List(items)) => {
                let elem_size = elem_ct.size_ctx(&*self.types.borrow_struct_layouts()).max(1);
                // For char/unsigned char arrays with a single string literal initializer,
                // the array size is the string length + 1 (null terminator)
                if elem_size == 1 && items.len() == 1 {
                    if let Initializer::Expr(ref expr) = items[0].init {
                        if let Expr::StringLiteral(ref s, _) | Expr::WideStringLiteral(ref s, _)
                            | Expr::Char16StringLiteral(ref s, _) = expr {
                            if matches!(expr, Expr::StringLiteral(_, _)) {
                                s.chars().count() + 1
                            } else if matches!(expr, Expr::Char16StringLiteral(_, _)) {
                                (s.chars().count() + 1) * 2
                            } else {
                                (s.chars().count() + 1) * 4
                            }
                        } else {
                            elem_size * items.len()
                        }
                    } else {
                        elem_size * items.len()
                    }
                } else {
                    // Use compute_init_list_array_size to correctly handle designated
                    // initializers like (int[]){[1]=10, [8]=80} which need 9 elements,
                    // not just items.len() (4 in this example).
                    elem_size * self.compute_init_list_array_size(items)
                }
            }
            _ => self.sizeof_type(type_spec),
        }
    }

    pub(super) fn lower_compound_literal(&mut self, type_spec: &TypeSpecifier, init: &Initializer) -> Operand {
        let ty = self.type_spec_to_ir(type_spec);
        let ctype = self.type_spec_to_ctype(type_spec);
        let size = self.compound_literal_size(type_spec, init);
        let alloca = self.alloc_and_init_compound_literal(type_spec, init, ty, size);

        let struct_layout = self.get_struct_layout_for_type(type_spec);
        let is_scalar = struct_layout.is_none() && !matches!(ctype, CType::Array(_, _)) && !ctype.is_vector();
        if is_scalar {
            let loaded = self.fresh_value();
            self.emit(Instruction::Load { dest: loaded, ptr: alloca, ty , seg_override: AddressSpace::Default });
            Operand::Value(loaded)
        } else {
            Operand::Value(alloca)
        }
    }

    pub(super) fn init_array_compound_literal(
        &mut self, alloca: Value, items: &[InitializerItem], type_spec: &TypeSpecifier,
        ty: IrType, size: usize,
    ) {
        let elem_size = self.compound_literal_elem_size(type_spec);

        // For vector types, determine the correct element IR type (e.g. F32 for float vectors)
        let vector_elem_ir_ty = {
            let ctype = self.type_spec_to_ctype(type_spec);
            match ctype {
                CType::Vector(elem_ct, _) => Some(IrType::from_ctype(&elem_ct)),
                _ => None,
            }
        };

        // For char/unsigned char array compound literals with a single string literal,
        // copy the string bytes directly instead of storing a pointer.
        if elem_size == 1 && items.len() == 1 && items[0].designators.is_empty() {
            if let Initializer::Expr(Expr::StringLiteral(ref s, _)) = items[0].init {
                self.emit_string_to_alloca(alloca, s, 0, size);
                return;
            }
        }

        // Get the element CType for dispatching struct/union/array element init
        let elem_ctype = {
            let ctype = self.type_spec_to_ctype(type_spec);
            match ctype {
                CType::Array(elem_ct, _) => Some((*elem_ct).clone()),
                _ => None,
            }
        };

        let has_designators = items.iter().any(|item| !item.designators.is_empty());
        // Zero-init if there are designators, or if fewer initializers than total array size
        // (C11 6.7.9p21: uninitialized elements are implicitly zero-initialized)
        let needs_zero = has_designators || (items.len() * elem_size < size);
        if needs_zero {
            self.zero_init_alloca(alloca, size);
        }

        let mut current_idx = 0usize;
        for item in items {
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx_val) = self.eval_const_expr_for_designator(idx_expr) {
                    current_idx = idx_val;
                }
            }

            let elem_offset = current_idx * elem_size;

            match &item.init {
                Initializer::Expr(expr) => {
                    // For struct/union elements, the lowered expression produces an
                    // alloca pointer (not the struct data itself). We must memcpy
                    // from that alloca into the array element slot instead of storing
                    // the pointer value.
                    let is_struct_elem = matches!(
                        &elem_ctype,
                        Some(CType::Struct(_)) | Some(CType::Union(_))
                    ) && self.struct_value_size(expr).is_some();

                    if is_struct_elem {
                        let src_addr = self.get_struct_base_addr(expr);
                        self.emit_memcpy_at_offset(alloca, elem_offset, src_addr, elem_size);
                    } else {
                        let val = self.lower_expr(expr);
                        // For vector compound literals, cast each element to the vector element type
                        let val = if let Some(vec_elem_ty) = vector_elem_ir_ty {
                            let expr_ty = self.get_expr_type(expr);
                            Operand::Value(self.emit_cast_val(val, expr_ty, vec_elem_ty))
                        } else {
                            val
                        };
                        if current_idx == 0 && items.len() == 1 && elem_size == size {
                            self.emit(Instruction::Store { val, ptr: alloca, ty , seg_override: AddressSpace::Default });
                        } else {
                            let offset_val = Operand::Const(IrConst::ptr_int(elem_offset as i64));
                            let elem_ptr = self.fresh_value();
                            self.emit(Instruction::GetElementPtr {
                                dest: elem_ptr, base: alloca, offset: offset_val, ty,
                            });
                            let store_ty = vector_elem_ir_ty.unwrap_or_else(|| Self::ir_type_for_size(elem_size));
                            self.emit(Instruction::Store { val, ptr: elem_ptr, ty: store_ty , seg_override: AddressSpace::Default });
                        }
                    }
                }
                Initializer::List(sub_items) => {
                    // Handle list initializers for array elements (e.g., struct or nested array)
                    match &elem_ctype {
                        Some(CType::Struct(key)) | Some(CType::Union(key)) => {
                            let sub_layout = self.types.borrow_struct_layouts().get(&**key).cloned();
                            if let Some(sub_layout) = sub_layout {
                                // Zero-init the element region first for partial initialization
                                self.zero_init_region(alloca, elem_offset, elem_size);
                                self.emit_struct_init(sub_items, alloca, &sub_layout, elem_offset);
                            }
                        }
                        Some(CType::Array(inner_elem_ty, Some(inner_size))) => {
                            // Nested array: initialize element-by-element
                            let inner_elem_size = self.resolve_ctype_size(inner_elem_ty);
                            let inner_ir_ty = IrType::from_ctype(inner_elem_ty);
                            let inner_is_bool = **inner_elem_ty == CType::Bool;
                            // Zero-init the element region for partial initialization
                            if sub_items.len() < *inner_size {
                                self.zero_init_region(alloca, elem_offset, elem_size);
                            }
                            for (si, sub_item) in sub_items.iter().enumerate() {
                                if si >= *inner_size { break; }
                                if let Initializer::Expr(e) = &sub_item.init {
                                    let inner_offset = elem_offset + si * inner_elem_size;
                                    self.emit_init_expr_to_offset_bool(
                                        e, alloca, inner_offset, inner_ir_ty, inner_is_bool,
                                    );
                                }
                            }
                        }
                        _ => {
                            // Scalar array element with list init: use the first expression
                            if let Some(first) = sub_items.first() {
                                if let Initializer::Expr(expr) = &first.init {
                                    let val = self.lower_expr(expr);
                                    let offset_val = Operand::Const(IrConst::ptr_int(elem_offset as i64));
                                    let elem_ptr = self.fresh_value();
                                    self.emit(Instruction::GetElementPtr {
                                        dest: elem_ptr, base: alloca, offset: offset_val, ty,
                                    });
                                    let store_ty = Self::ir_type_for_size(elem_size);
                                    self.emit(Instruction::Store { val, ptr: elem_ptr, ty: store_ty , seg_override: AddressSpace::Default });
                                }
                            }
                        }
                    }
                }
            }
            current_idx += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Sizeof, Generic selection
    // -----------------------------------------------------------------------

    pub(super) fn lower_sizeof(&mut self, arg: &SizeofArg) -> Operand {
        if let Some(vla_val) = self.get_vla_sizeof(arg) {
            return Operand::Value(vla_val);
        }
        // Check for sizeof(type) where type is or contains VLA dimensions
        if let SizeofArg::Type(ts) = arg {
            if let Some(vla_val) = self.compute_vla_sizeof_for_type(ts) {
                return Operand::Value(vla_val);
            }
        }
        let size = match arg {
            SizeofArg::Type(ts) => self.sizeof_type(ts),
            SizeofArg::Expr(expr) => self.sizeof_expr(expr),
        };
        Operand::Const(IrConst::ptr_int(size as i64))
    }

    fn get_vla_sizeof(&self, arg: &SizeofArg) -> Option<Value> {
        if let SizeofArg::Expr(Expr::Identifier(name, _)) = arg {
            // Check local VLA variables first
            if let Some(info) = self.func().locals.get(name) {
                if info.vla_size.is_some() {
                    return info.vla_size;
                }
            }
            // Then check VLA typedef names (sizeof applied to a typedef identifier)
            if let Some(&vla_size) = self.func().vla_typedef_sizes.get(name) {
                return Some(vla_size);
            }
        }
        None
    }

    /// Compute the runtime sizeof for a type that may contain VLA dimensions.
    /// Handles both typedef names that are VLA types and direct Array types
    /// with non-constant size expressions.
    fn compute_vla_sizeof_for_type(&mut self, ts: &TypeSpecifier) -> Option<Value> {
        // Check if it's a VLA typedef name with a pre-computed runtime size
        if let TypeSpecifier::TypedefName(name) = ts {
            if let Some(&vla_size) = self.func().vla_typedef_sizes.get(name) {
                return Some(vla_size);
            }
        }
        // Check if it's an Array type with non-constant dimensions
        let resolved = self.resolve_type_spec(ts).clone();
        if let TypeSpecifier::Array(ref elem, Some(ref size_expr)) = resolved {
            if self.expr_as_array_size(size_expr).is_none() {
                // Runtime dimension - compute size dynamically
                let elem_clone = elem.clone();
                let size_expr_clone = size_expr.clone();
                let elem_size_opt = self.compute_vla_sizeof_for_type(&elem_clone);
                let dim_val = self.lower_expr(&size_expr_clone);
                let dim_value = self.operand_to_value(dim_val);
                if let Some(elem_sz) = elem_size_opt {
                    // Both element and dimension are runtime
                    let ptr_int_ty = crate::common::types::target_int_ir_type();
                    let mul = self.emit_binop_val(IrBinOp::Mul, Operand::Value(dim_value), Operand::Value(elem_sz), ptr_int_ty);
                    return Some(mul);
                } else {
                    // Element is constant, dimension is runtime
                    let elem_size = self.sizeof_type(&elem_clone) as i64;
                    if elem_size > 1 {
                        let ptr_int_ty = crate::common::types::target_int_ir_type();
                        let mul = self.emit_binop_val(IrBinOp::Mul, Operand::Value(dim_value), Operand::Const(IrConst::ptr_int(elem_size)), ptr_int_ty);
                        return Some(mul);
                    } else {
                        return Some(dim_value);
                    }
                }
            }
            // Constant outer dim but maybe VLA inner dims
            let elem_size_opt = self.compute_vla_sizeof_for_type(elem);
            if let Some(elem_sz) = elem_size_opt {
                let const_dim = self.expr_as_array_size(size_expr).unwrap_or(1);
                if const_dim > 1 {
                    let ptr_int_ty = crate::common::types::target_int_ir_type();
                    let mul = self.emit_binop_val(IrBinOp::Mul, Operand::Value(elem_sz), Operand::Const(IrConst::ptr_int(const_dim)), ptr_int_ty);
                    return Some(mul);
                } else {
                    return Some(elem_sz);
                }
            }
        }
        None
    }

    pub(super) fn lower_generic_selection(&mut self, controlling: &Expr, associations: &[GenericAssociation]) -> Operand {
        let selected = self.resolve_generic_selection(controlling, associations);
        self.lower_expr(selected)
    }

    /// Resolve a _Generic selection to the matching association expression.
    /// Used by both lower_generic_selection (rvalue) and lower_lvalue (lvalue context).
    pub(super) fn resolve_generic_selection<'a>(&mut self, controlling: &Expr, associations: &'a [GenericAssociation]) -> &'a Expr {
        let controlling_ctype = self.get_expr_ctype(controlling);
        let controlling_ir_type = self.get_expr_type(controlling);
        // Per C11 6.5.1.1p2, the controlling expression undergoes lvalue conversion,
        // which includes array-to-pointer and function-to-pointer decay, and strips
        // top-level qualifiers.
        let controlling_ctype = controlling_ctype.map(|ct| match ct {
            CType::Array(elem, _) => CType::Pointer(elem, AddressSpace::Default),
            CType::Function(ft) => CType::Pointer(Box::new(CType::Function(ft)), AddressSpace::Default),
            other => other,
        });
        // Determine if the controlling expression's type has a const-qualified pointee.
        // Lvalue conversion strips top-level qualifiers. So for non-pointer types like
        // `const int x`, the type becomes `int` (ctrl_is_const = false).
        // For pointer types like `const int *p`, lvalue conversion strips the top-level
        // pointer const but preserves the pointee const, so ctrl_is_const reflects
        // whether the pointee is const.
        let ctrl_is_const = if let Some(ref ct) = controlling_ctype {
            matches!(ct, CType::Pointer(_, _)) && self.expr_is_const_qualified(controlling)
        } else {
            false
        };

        // Check if const-aware matching is needed. This is the case when:
        // (a) associations differ in const-ness (some have is_const=true, others false), OR
        // (b) the controlling expression is a pointer with const pointee (ctrl_is_const=true),
        //     which means non-const pointer associations should not match.
        let has_const_differentiated_assocs = {
            let non_default: Vec<_> = associations.iter()
                .filter(|a| a.type_spec.is_some())
                .collect();
            let assocs_differ = non_default.iter().any(|a| a.is_const) && non_default.iter().any(|a| !a.is_const);
            assocs_differ || ctrl_is_const
        };

        let mut default_expr: Option<&Expr> = None;
        let mut matched_expr: Option<&Expr> = None;

        for assoc in associations {
            match &assoc.type_spec {
                None => { default_expr = Some(&assoc.expr); }
                Some(type_spec) => {
                    let assoc_ctype = self.type_spec_to_ctype(type_spec);
                    if let Some(ref ctrl_ct) = controlling_ctype {
                        if self.ctype_matches_generic(ctrl_ct, &assoc_ctype) {
                            // When const-aware matching is active, also check
                            // that the const qualification matches.
                            if has_const_differentiated_assocs && assoc.is_const != ctrl_is_const {
                                continue;
                            }
                            matched_expr = Some(&assoc.expr);
                            break;
                        }
                    } else {
                        let assoc_ir_type = self.type_spec_to_ir(type_spec);
                        if assoc_ir_type == controlling_ir_type {
                            if has_const_differentiated_assocs && assoc.is_const != ctrl_is_const {
                                continue;
                            }
                            matched_expr = Some(&assoc.expr);
                            break;
                        }
                    }
                }
            }
        }

        matched_expr.or(default_expr).unwrap_or(&associations[0].expr)
    }

    pub(super) fn ctype_matches_generic(&self, controlling: &CType, assoc: &CType) -> bool {
        // Per C11 6.5.1.1p2, the controlling expression undergoes lvalue conversion,
        // which includes array-to-pointer decay.
        // Const/volatile qualification is checked separately via is_const flags.

        // Exact structural match (same discriminant)
        if std::mem::discriminant(controlling) == std::mem::discriminant(assoc) {
            match (controlling, assoc) {
                (CType::Pointer(a, _), CType::Pointer(b, _)) => {
                    return self.ctype_matches_generic(a, b);
                }
                (CType::Array(a, _), CType::Array(b, _)) => {
                    return self.ctype_matches_generic(a, b);
                }
                // For types that carry identity data (Struct, Union, Enum, Function,
                // Vector), we must compare the full type, not just the discriminant.
                // Primitive types (Int, Float, etc.) have no data so PartialEq works too.
                _ => return controlling == assoc,
            }
        }
        // Array decays to pointer for _Generic matching
        if let CType::Array(elem, _) = controlling {
            if let CType::Pointer(pointee, _) = assoc {
                return self.ctype_matches_generic(elem, pointee);
            }
        }
        // Enum matches int
        if matches!(controlling, CType::Enum(_)) && matches!(assoc, CType::Int) {
            return true;
        }
        if matches!(controlling, CType::Int) && matches!(assoc, CType::Enum(_)) {
            return true;
        }
        false
    }

    /// Check if an expression's type is const-qualified (for _Generic matching).
    /// Handles local variable identifiers, address-of, and comma expressions.
    pub(super) fn expr_is_const_qualified(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Identifier(name, _) => {
                if let Some(ref fs) = self.func_state {
                    if let Some(info) = fs.locals.get(name) {
                        return info.is_const;
                    }
                }
                false
            }
            // Address-of: &y where y is `const int` produces `const int *`.
            Expr::AddressOf(inner, _) => {
                self.expr_is_const_qualified(inner)
            }
            // Comma: const-ness comes from the right-hand operand.
            Expr::Comma(_, rhs, _) => {
                self.expr_is_const_qualified(rhs)
            }
            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // Address-of, dereference, subscript, member access
    // -----------------------------------------------------------------------

    fn alloc_and_init_compound_literal(
        &mut self, type_spec: &TypeSpecifier, init: &Initializer, ty: IrType, size: usize,
    ) -> Value {
        let alloca = self.fresh_value();
        self.emit(Instruction::Alloca { dest: alloca, size, ty, align: 0, volatile: false });
        let struct_layout = self.get_struct_layout_for_type(type_spec);
        match init {
            Initializer::Expr(expr) => {
                // For char array compound literals with string literal initializer,
                // copy string bytes instead of storing pointer
                let is_char_array = matches!(ty, IrType::I8 | IrType::U8) && {
                    let ctype = self.type_spec_to_ctype(type_spec);
                    matches!(ctype, CType::Array(_, _))
                };
                if is_char_array {
                    if let Expr::StringLiteral(ref s, _) = expr {
                        self.emit_string_to_alloca(alloca, s, 0, size);
                    } else {
                        let val = self.lower_expr(expr);
                        self.emit(Instruction::Store { val, ptr: alloca, ty , seg_override: AddressSpace::Default });
                    }
                } else {
                    let val = self.lower_expr(expr);
                    self.emit(Instruction::Store { val, ptr: alloca, ty , seg_override: AddressSpace::Default });
                }
            }
            Initializer::List(items) => {
                // Check if the type is an array or vector
                let ctype = self.type_spec_to_ctype(type_spec);
                let is_array = matches!(ctype, CType::Array(_, _));
                // For vector compound literals, use the vector init list path
                if let Some((elem_ct, num_elems)) = ctype.vector_info() {
                    let elem_ct = elem_ct.clone();
                    self.lower_vector_init_list(items, alloca, &elem_ct, num_elems);
                } else if !is_array {
                    if let Some(ref layout) = struct_layout {
                        self.zero_init_alloca(alloca, layout.size);
                        self.emit_struct_init(items, alloca, layout, 0);
                    } else {
                        self.init_array_compound_literal(alloca, items, type_spec, ty, size);
                    }
                } else {
                    self.init_array_compound_literal(alloca, items, type_spec, ty, size);
                }
            }
        }
        alloca
    }

    pub(super) fn lower_address_of(&mut self, inner: &Expr) -> Operand {
        // Try to constant-fold offsetof patterns: &((type*)0)->member
        // This is critical for BUILD_BUG_ON / __compiletime_assert in the Linux kernel,
        // which uses __builtin_offsetof (expanded to this pattern) in if-conditions
        // that must be dead-code-eliminated.
        if let Some(constant) = self.eval_offsetof_pattern(inner) {
            return Operand::Const(constant);
        }

        if let Expr::CompoundLiteral(type_spec, init, _) = inner {
            let ty = self.type_spec_to_ir(type_spec);
            let size = self.compound_literal_size(type_spec, init);
            let alloca = self.alloc_and_init_compound_literal(type_spec, init, ty, size);
            return Operand::Value(alloca);
        }

        if let Some(lv) = self.lower_lvalue(inner) {
            let addr = self.lvalue_addr(&lv);
            let result = self.fresh_value();
            self.emit(Instruction::GetElementPtr {
                dest: result, base: addr,
                offset: Operand::Const(IrConst::I64(0)),
                ty: IrType::Ptr,
            });
            return Operand::Value(result);
        }

        if let Expr::Identifier(name, _) = inner {
            let dest = self.fresh_value();
            // Apply __asm__("label") redirect (e.g. stat -> stat64)
            let resolved = self.asm_label_map.get(&*name)
                .cloned()
                .unwrap_or_else(|| name.clone());
            self.emit(Instruction::GlobalAddr { dest, name: Rc::from(resolved) });
            return Operand::Value(dest);
        }

        self.lower_expr(inner)
    }

    /// Check if dereferencing the given expression is a no-op because the
    /// expression is a function pointer (or function designator). In C:
    /// - *f where f is a function pointer → function designator → decays back
    /// - *add where add is a function name → function designator → decays back
    /// - *(s->fnptr) where fnptr is a function pointer member → same no-op
    /// - **f, ***f etc. are also no-ops (recursive application)
    pub(super) fn is_function_pointer_deref(&self, inner: &Expr) -> bool {
        // First, check if the inner expression is actually a pointer-to-function-pointer.
        // Due to our CType representation, int (**fpp)(int,int) is stored as
        // Pointer(Function(FunctionType { return_type: Pointer(Int), ... })) — the same
        // shape as a direct function pointer. We use the is_ptr_to_func_ptr flag
        // (set during declaration analysis based on pointer count in derived declarators)
        // to correctly distinguish these cases.
        if let Expr::Identifier(name, _) = inner {
            if let Some(vi) = self.lookup_var_info(name) {
                if vi.is_ptr_to_func_ptr {
                    // This is a pointer-to-function-pointer (e.g., int (**fpp)(int,int)).
                    // Dereferencing it requires a real load to get the inner function pointer.
                    return false;
                }
            }
        }

        // Check CType-based detection (works for typedef'd function pointers
        // where param_ctype correctly sets CType::Pointer(CType::Function(...)),
        // and also for struct member function pointers via get_expr_ctype)
        if let Some(ref inner_ct) = self.get_expr_ctype(inner) {
            if inner_ct.is_function_pointer() || matches!(inner_ct, CType::Function(_)) {
                return true;
            }
        }
        // Check for known function names (e.g., *add where add is a function)
        match inner {
            Expr::Identifier(name, _) => {
                // Only treat as a function name dereference if there is no local variable
                // with the same name. Local variables shadow function names, so if a local
                // variable called "link" exists, *link should dereference the variable,
                // not be treated as a no-op function pointer dereference.
                if self.known_functions.contains(&*name) && self.lookup_var_info(name).is_none() {
                    return true;
                }
                // Check if this variable is a function pointer (deref is no-op).
                // Pointer-to-function-pointer cases are already handled by the
                // is_ptr_to_func_ptr early-return above, so any Pointer(Function(...))
                // at this point is a genuine direct function pointer.
                if let Some(vi) = self.lookup_var_info(name) {
                    if let Some(ref ct) = vi.c_type {
                        if ct.is_function_pointer() || matches!(ct, CType::Function(_)) {
                            return true;
                        }
                    } else {
                        // Fallback: check ptr_sigs only when c_type is unavailable.
                        // When c_type IS available, the check above is authoritative —
                        // ptr_sigs may contain entries for pointer-to-function-pointers
                        // which are NOT no-op derefs.
                        if self.func_meta.ptr_sigs.contains_key(&*name) {
                            return true;
                        }
                    }
                }
                false
            }
            // For nested derefs: *(*f) where *f is also a no-op → check recursively
            Expr::Deref(deeper_inner, _) => self.is_function_pointer_deref(deeper_inner),
            // For function calls that return a function pointer: *(p(args))
            // If the callee is a function pointer variable whose return type is
            // itself a function pointer, then dereferencing the result is a no-op.
            Expr::FunctionCall(func, _, _) => {
                // Strip deref layers from the callee (since *p, **p are equivalent for fptrs)
                let mut callee: &Expr = func.as_ref();
                while let Expr::Deref(inner, _) = callee {
                    callee = inner;
                }
                if let Expr::Identifier(name, _) = callee {
                    // Check if the callee returns a function pointer by examining its CType
                    if let Some(vi) = self.lookup_var_info(name) {
                        if let Some(ref ct) = vi.c_type {
                            // Extract the return type of the function pointer
                            if let Some(ret_ct) = ct.func_ptr_return_type(true) {
                                // If the return type is a function pointer, deref is no-op
                                if ret_ct.is_function_pointer() || matches!(&ret_ct, CType::Function(_)) {
                                    return true;
                                }
                            }
                        }
                    }
                    // Also check known function signatures
                    if let Some(sig) = self.func_meta.sigs.get(&*name) {
                        if let Some(ref ret_ct) = sig.return_ctype {
                            if ret_ct.is_function_pointer() {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    pub(super) fn lower_deref(&mut self, inner: &Expr) -> Operand {
        // Check if pointee is an aggregate type that doesn't need a Load.
        // Note: Function types are NOT included here — function pointer dereferences
        // are handled by is_function_pointer_deref which correctly distinguishes
        // direct function pointers (no-op deref) from pointer-to-function-pointers
        // (which need a real load despite having similar CType shapes).
        let pointee_is_no_load = |ct: &CType| -> bool {
            if let CType::Pointer(ref pointee, _) = ct {
                matches!(pointee.as_ref(),
                    CType::Array(_, _) | CType::Struct(_) | CType::Union(_) | CType::Vector(_, _))
                    || pointee.is_complex()
            } else {
                false
            }
        };
        // In C, dereferencing a function pointer yields a function designator which
        // immediately decays back to a function pointer. So *f, **f, ***f etc. are
        // all equivalent when f is a function pointer. No Load should be emitted.
        // This also applies to direct function names: (*add)(x, y) is valid C
        // because the function name decays to a pointer, and dereferencing that
        // gives back the function designator.
        if self.is_function_pointer_deref(inner) {
            return self.lower_expr(inner);
        }
        // Check if dereferencing yields an aggregate type.
        // In these cases, the result is an address (no Load needed).
        if self.get_expr_ctype(inner).is_some_and(|ct| pointee_is_no_load(&ct)) {
            return self.lower_expr(inner);
        }
        {
            let inner_ct = self.expr_ctype(inner);
            if pointee_is_no_load(&inner_ct) {
                return self.lower_expr(inner);
            }
            // Handle *array where array is a multi-dimensional array:
            // e.g., *x where x is int[2][3] has inner CType = Array(Array(Int,3), 2).
            // Dereferencing peels off the outer dimension, yielding Array(Int,3) which
            // is an aggregate — so no Load is needed, just return the base address.
            if let CType::Array(ref elem, _) = inner_ct {
                if elem.is_complex()
                    || matches!(elem.as_ref(), CType::Struct(_) | CType::Union(_) | CType::Array(_, _) | CType::Vector(_, _))
                {
                    return self.lower_expr(inner);
                }
            }
        }

        let addr_space = self.get_addr_space_of_ptr_expr(inner);
        let ptr = self.lower_expr(inner);
        let dest = self.fresh_value();
        let deref_ty = self.get_pointee_type_of_expr(inner).unwrap_or(crate::common::types::target_int_ir_type());
        let ptr_val = self.operand_to_value(ptr);
        self.emit(Instruction::Load { dest, ptr: ptr_val, ty: deref_ty, seg_override: addr_space });
        Operand::Value(dest)
    }

    pub(super) fn lower_array_subscript(&mut self, expr: &Expr, base: &Expr, index: &Expr) -> Operand {
        if self.subscript_result_is_array(expr) {
            let addr = self.compute_array_element_addr(base, index);
            return Operand::Value(addr);
        }
        if self.struct_value_size(expr).is_some() {
            let addr = self.compute_array_element_addr(base, index);
            return Operand::Value(addr);
        }
        {
            let elem_ct = self.expr_ctype(expr);
            if elem_ct.is_complex() || elem_ct.is_vector() {
                let addr = self.compute_array_element_addr(base, index);
                return Operand::Value(addr);
            }
        }
        let addr_space = self.get_addr_space_of_ptr_expr(base);
        let elem_ty = self.get_expr_type(expr);
        let addr = self.compute_array_element_addr(base, index);
        let dest = self.fresh_value();
        self.emit(Instruction::Load { dest, ptr: addr, ty: elem_ty, seg_override: addr_space });
        Operand::Value(dest)
    }

    // -----------------------------------------------------------------------
    // Member access (s.field, p->field)
    // -----------------------------------------------------------------------

    fn lower_member_access_impl(&mut self, base_expr: &Expr, field_name: &str, is_pointer: bool) -> Operand {
        let (field_offset, field_ty, bitfield, field_ctype) = if is_pointer {
            self.resolve_pointer_member_access_with_ctype(base_expr, field_name)
        } else {
            self.resolve_member_access_with_ctype(base_expr, field_name)
        };
        // Extract address space: for p->field from the pointer type,
        // for s.field from the struct variable's address space qualifier.
        let addr_space = if is_pointer {
            self.get_addr_space_of_ptr_expr(base_expr)
        } else {
            self.get_addr_space_of_struct_expr(base_expr)
        };

        let base_addr = if is_pointer {
            let ptr_val = self.lower_expr(base_expr);
            self.operand_to_value(ptr_val)
        } else {
            self.get_struct_base_addr(base_expr)
        };

        let field_addr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: field_addr, base: base_addr,
            offset: Operand::Const(IrConst::ptr_int(field_offset as i64)),
            ty: field_ty,
        });

        let is_addr_type = match &field_ctype {
            Some(ct) => {
                Self::is_aggregate_or_complex(ct)
            }
            None => {
                let resolved = self.resolve_field_ctype(base_expr, field_name, is_pointer);
                resolved.as_ref()
                    .map(Self::is_aggregate_or_complex)
                    .unwrap_or(false)
            }
        };
        if is_addr_type {
            return Operand::Value(field_addr);
        }

        // For bitfields, use extract_bitfield_from_addr which handles split loads
        // (packed bitfields that span storage unit boundaries).
        if let Some((bit_offset, bit_width)) = bitfield {
            return self.extract_bitfield_from_addr(field_addr, field_ty, bit_offset, bit_width);
        }

        let dest = self.fresh_value();
        self.emit(Instruction::Load { dest, ptr: field_addr, ty: field_ty, seg_override: addr_space });
        Operand::Value(dest)
    }

    pub(super) fn lower_member_access(&mut self, base_expr: &Expr, field_name: &str) -> Operand {
        self.lower_member_access_impl(base_expr, field_name, false)
    }

    pub(super) fn lower_pointer_member_access(&mut self, base_expr: &Expr, field_name: &str) -> Operand {
        self.lower_member_access_impl(base_expr, field_name, true)
    }

    /// Check if a CType is an aggregate (array/struct/union) or complex type.
    /// These types are always accessed by address rather than loaded by value.
    fn is_aggregate_or_complex(ct: &CType) -> bool {
        matches!(ct,
            CType::Array(_, _) | CType::Struct(_) | CType::Union(_) |
            CType::ComplexFloat | CType::ComplexDouble | CType::ComplexLongDouble |
            CType::Vector(_, _))
    }

    // -----------------------------------------------------------------------
    // Statement expressions (GCC extension)
    // -----------------------------------------------------------------------

    pub(super) fn lower_stmt_expr(&mut self, compound: &CompoundStmt) -> Operand {
        // If this statement expression has __label__ declarations, push a local
        // label scope so that labels are uniquified per expansion.
        let has_local_labels = !compound.local_labels.is_empty();
        if has_local_labels {
            let scope_id = self.next_local_label_scope;
            self.next_local_label_scope += 1;
            let mut scope = crate::common::fx_hash::FxHashMap::default();
            for name in &compound.local_labels {
                scope.insert(name.clone(), Rc::from(format!("{}$ll{}", name, scope_id)));
            }
            self.local_label_scopes.push(scope);
        }

        // Push a scope so that variables declared inside the statement expression
        // don't leak into the enclosing scope.  Without this, nested statement
        // expressions that re-declare the same variable name (extremely common in
        // kernel macros like READ_ONCE, per-CPU accessors, container_of) permanently
        // overwrite the outer binding, producing wrong code.
        let has_declarations = compound.items.iter().any(|item| matches!(item, BlockItem::Declaration(_)));
        if has_declarations {
            self.push_scope();
        }

        let mut last_val = Operand::Const(IrConst::ptr_int(0));
        for item in &compound.items {
            match item {
                BlockItem::Statement(stmt) => {
                    // Peel through Label wrappers to find the innermost statement.
                    // A label like `out: sz;` parses as Label("out", Expr(Some(sz))).
                    // We need to lower the labels (creating branch targets) and then
                    // capture the final expression value, rather than discarding it
                    // via lower_stmt.  This is critical for goto inside statement
                    // expressions: the goto jumps to the label, and the expression
                    // after the label is the statement expression's value.
                    let mut inner = stmt;
                    let mut labels = Vec::new();
                    while let Stmt::Label(name, sub_stmt, _span) = inner {
                        labels.push(&*name);
                        inner = sub_stmt;
                    }
                    if !labels.is_empty() {
                        // Lower each label: terminate current block, start label block
                        for label_name in &labels {
                            let label = self.get_or_create_user_label(label_name);
                            self.terminate(Terminator::Branch(label));
                            self.start_block(label);
                        }
                        // Now lower the innermost statement
                        if let Stmt::Expr(Some(expr)) = inner {
                            last_val = self.lower_expr(expr);
                        } else {
                            self.lower_stmt(inner);
                        }
                    } else if let Stmt::Expr(Some(expr)) = stmt {
                        last_val = self.lower_expr(expr);
                    } else {
                        self.lower_stmt(stmt);
                    }
                }
                BlockItem::Declaration(decl) => {
                    self.collect_enum_constants_scoped(&decl.type_spec);
                    self.lower_local_decl(decl);
                }
            }
        }

        if has_declarations {
            self.pop_scope();
        }

        if has_local_labels {
            self.local_label_scopes.pop();
        }

        last_val
    }

    // -----------------------------------------------------------------------
    // va_arg
    // -----------------------------------------------------------------------

    pub(super) fn lower_va_arg(&mut self, ap_expr: &Expr, type_spec: &TypeSpecifier) -> Operand {
        // VaArg instruction expects a pointer to the va_list "object" so it can
        // both read from and advance the va_list state.
        //
        // The correct pointer depends on the target's va_list representation:
        //
        // x86-64 / AArch64: va_list is an array type (char[24] / char[32]).
        //   - LOCAL va_list:  alloca IS the array, lower_expr returns the alloca
        //     address (array-to-pointer decay) → pointer to the struct. ✓
        //   - PARAM va_list:  array decayed to char* when passed; the alloca holds
        //     a pointer to the caller's va_list struct, lower_expr loads that
        //     pointer value → pointer to the struct. ✓
        //   So lower_expr gives the correct va_list pointer on these targets.
        //
        // RISC-V: va_list is a scalar pointer type (void *).
        //   - LOCAL va_list:  alloca holds the void* value. va_arg needs the
        //     ADDRESS of the alloca (to read the current pointer and write back
        //     the advanced one). lower_address_of returns this. ✓
        //   - PARAM va_list:  alloca holds a copy of the void* parameter.
        //     Same as local: need the ADDRESS of the alloca. lower_address_of
        //     returns this. ✓
        //   So lower_address_of gives the correct va_list pointer on RISC-V.

        // Check if the requested type is a complex type. Complex types map to
        // IrType::Ptr, but the backend va_arg code only handles scalar types
        // (integer, float). We decompose complex va_arg into two component
        // va_arg calls (one for real, one for imaginary) and reassemble.
        let ctype = self.type_spec_to_ctype(type_spec);
        if ctype.is_complex() {
            return self.lower_va_arg_complex(ap_expr, &ctype);
        }

        // Check if the requested type is a struct/union. Struct types are passed
        // by value in variadic arguments (their bytes are inline on the va_list),
        // but IrType maps structs to Ptr. We need to read the struct data as
        // individual slots and store to a temporary alloca.
        // Use the resolved ctype rather than pattern-matching on the TypeSpecifier,
        // because typeof(expr) produces TypeSpecifier::Typeof which would not be
        // recognized by is_type_struct_or_union but resolves correctly via ctype.
        if ctype.is_struct_or_union() {
            return self.lower_va_arg_struct(ap_expr, type_spec, &ctype);
        }

        use crate::backend::Target;
        let ap_val = if self.target == Target::Riscv64 || self.target == Target::I686 {
            // RISC-V / i686: va_list is a pointer, need address of the variable holding it
            self.lower_address_of(ap_expr)
        } else {
            // x86-64 / AArch64: va_list is an array type, lower_expr handles
            // both local (array decay) and parameter (load pointer) cases
            self.lower_expr(ap_expr)
        };
        let va_list_ptr = self.operand_to_value(ap_val);
        let result_ty = self.resolve_va_arg_type(type_spec);
        let dest = self.fresh_value();
        self.emit(Instruction::VaArg { dest, va_list_ptr, result_ty });
        Operand::Value(dest)
    }

    /// Lower va_arg for struct/union types.
    ///
    /// Struct variadic arguments are passed by value: their bytes are stored
    /// inline on the va_list (or in registers, for the first few args).
    /// However, IrType maps structs to IrType::Ptr, so the backend's emit_va_arg
    /// would read a pointer-sized value and treat it as an address — causing a
    /// segfault when dereferenced.
    ///
    /// Fix: read the struct data as individual 8-byte (or 4-byte on i686) slots
    /// via scalar VaArg instructions, store them to a temporary alloca, and
    /// return the alloca address.
    ///
    /// Exception: on ARM64/RISC-V, structs larger than 16 bytes are passed by
    /// reference (a pointer is on the va_list), so we read one pointer.
    fn lower_va_arg_struct(
        &mut self,
        ap_expr: &Expr,
        type_spec: &TypeSpecifier,
        ctype: &CType,
    ) -> Operand {
        use crate::backend::Target;

        let struct_size = self.sizeof_type(type_spec);
        let struct_align = self.alignof_type(type_spec);

        // On ARM64 and RISC-V, structs > 16 bytes are passed by reference:
        // a pointer to the struct is placed in the va_list. Read the pointer
        // and return it directly (the pointer IS the struct address).
        let large_struct_by_ref = matches!(self.target, Target::Aarch64 | Target::Riscv64);
        if large_struct_by_ref && struct_size > 16 {
            let ap_val = self.lower_va_list_pointer(ap_expr);
            let va_list_ptr = self.operand_to_value(ap_val);
            let dest = self.fresh_value();
            self.emit(Instruction::VaArg {
                dest,
                va_list_ptr,
                result_ty: IrType::Ptr,
            });
            return Operand::Value(dest);
        }

        // On x86-64 and AArch64, structs use VaArgStruct for correct ABI handling.
        // Both the SysV x86-64 ABI and AAPCS64 require that multi-register structs
        // are passed entirely in registers OR entirely on the stack - never split
        // across the boundary. By using VaArgStruct, the backend can atomically
        // check if all required register slots are available and fall back to the
        // overflow/stack area if not.
        if self.target == Target::X86_64 || self.target == Target::Aarch64 {
            let eightbyte_classes = if self.target == Target::X86_64 && struct_size <= 16 {
                if let Some(layout) = self.get_struct_layout_for_ctype(ctype) {
                    layout.classify_sysv_eightbytes(&*self.types.borrow_struct_layouts())
                } else {
                    Vec::new()
                }
            } else {
                Vec::new() // AArch64 or >16 bytes on x86 = no eightbyte classes needed
            };

            let ap_val = self.lower_va_list_pointer(ap_expr);
            let va_list_ptr = self.operand_to_value(ap_val);
            let num_slots = struct_size.div_ceil(8);
            let alloc_size = if struct_size > 0 { num_slots * 8 } else { 8 };
            let alloca = self.fresh_value();
            self.emit(Instruction::Alloca {
                dest: alloca,
                size: alloc_size,
                ty: IrType::I64,
                align: struct_align,
                volatile: false,
            });
            self.emit(Instruction::VaArgStruct {
                dest_ptr: alloca,
                va_list_ptr,
                size: struct_size,
                eightbyte_classes,
            });
            return Operand::Value(alloca);
        }

        // Non-x86-64 targets: small structs passed by value inline.
        // Read each slot from the va_list and store to a temporary alloca.
        let slot_size = if self.target == Target::I686 { 4 } else { 8 };
        let default_slot_ty = if self.target == Target::I686 { IrType::I32 } else { IrType::I64 };

        // Number of slots needed (round up)
        let num_slots = struct_size.div_ceil(slot_size);
        let alloc_size = if struct_size > 0 { num_slots * slot_size } else { slot_size };

        // Allocate temporary storage for the struct
        let alloca = self.fresh_value();
        self.emit(Instruction::Alloca {
            dest: alloca,
            size: alloc_size,
            ty: default_slot_ty,
            align: struct_align,
            volatile: false,
        });

        // Get the va_list pointer (same logic as scalar va_arg)
        let ap_val = self.lower_va_list_pointer(ap_expr);
        let va_list_ptr = self.operand_to_value(ap_val);

        // On RISC-V, structs containing long double (f128) require 16-byte alignment
        // in the variadic argument area. The RISC-V LP64D ABI specifies that 2*XLEN-bit
        // scalars (and structs containing them) are aligned to 2*XLEN (16 bytes) on the
        // stack. Without this alignment, reading a struct {long double f;} via va_arg
        // reads from the wrong offset, corrupting the value and misaligning all
        // subsequent va_arg reads.
        //
        // We align the va_list pointer before reading slots:
        //   ptr = load(va_list_ptr)
        //   aligned_ptr = (ptr + align - 1) & ~(align - 1)
        //   store(aligned_ptr, va_list_ptr)
        if self.target == Target::Riscv64 && struct_align > slot_size {
            let cur_ptr = self.fresh_value();
            self.emit(Instruction::Load {
                dest: cur_ptr,
                ptr: va_list_ptr,
                ty: IrType::Ptr,
                seg_override: AddressSpace::Default,
            });
            let align_val = struct_align as i64;
            let added = self.emit_binop_val(
                IrBinOp::Add,
                Operand::Value(cur_ptr),
                Operand::Const(IrConst::I64(align_val - 1)),
                IrType::Ptr,
            );
            let aligned = self.emit_binop_val(
                IrBinOp::And,
                Operand::Value(added),
                Operand::Const(IrConst::I64(-align_val)),
                IrType::Ptr,
            );
            self.emit(Instruction::Store {
                val: Operand::Value(aligned),
                ptr: va_list_ptr,
                ty: IrType::Ptr,
                seg_override: AddressSpace::Default,
            });
        }

        // Read each slot and store it into the alloca.
        for i in 0..num_slots {
            let slot_val = self.fresh_value();
            self.emit(Instruction::VaArg {
                dest: slot_val,
                va_list_ptr,
                result_ty: default_slot_ty,
            });
            if i == 0 {
                // Store directly to the alloca base
                self.emit(Instruction::Store {
                    val: Operand::Value(slot_val),
                    ptr: alloca,
                    ty: default_slot_ty,
                    seg_override: AddressSpace::Default,
                });
            } else {
                // Store at offset i * slot_size
                let offset = (i * slot_size) as i64;
                let offset_ptr = self.fresh_value();
                self.emit(Instruction::BinOp {
                    dest: offset_ptr,
                    op: IrBinOp::Add,
                    lhs: Operand::Value(alloca),
                    rhs: Operand::Const(IrConst::I64(offset)),
                    ty: IrType::Ptr,
                });
                self.emit(Instruction::Store {
                    val: Operand::Value(slot_val),
                    ptr: offset_ptr,
                    ty: default_slot_ty,
                    seg_override: AddressSpace::Default,
                });
            }
        }

        // Return the alloca address — callers will use this as a pointer to the struct data
        Operand::Value(alloca)
    }

    /// Lower va_arg for complex types by decomposing into component va_arg calls.
    ///
    /// Complex float (_Complex float) is special: on x86-64 and RISC-V it's passed
    /// as two F32 values packed into a single 8-byte slot (one XMM register on x86,
    /// one integer register on RISC-V). We read one F64/I64 and bitcast-unpack it.
    /// On ARM64, float _Complex occupies two separate register slots per AAPCS64.
    ///
    /// Complex double (_Complex double) is passed as two F64 values on all platforms.
    ///
    /// Complex long double on x86-64: passed on stack (MEMORY class), reading as
    /// two F128 values from the FP area.
    /// TODO: handle _Complex long double specially on x86-64 if needed.
    fn lower_va_arg_complex(&mut self, ap_expr: &Expr, ctype: &CType) -> Operand {
        use crate::backend::Target;

        // Get va_list pointer using the shared helper (handles target differences)
        let ap_val = self.lower_va_list_pointer(ap_expr);
        let va_list_ptr = self.operand_to_value(ap_val);

        // Handle float _Complex specially: packed into one 8-byte slot on x86-64 and RISC-V.
        // The two F32 components (real, imag) are packed into a single 8-byte value:
        // - x86-64: packed in one XMM register, read as F64
        // - RISC-V: packed in one integer register, read as I64
        if *ctype == CType::ComplexFloat && (self.target == Target::X86_64 || self.target == Target::Riscv64) {
            let read_ty = if self.target == Target::X86_64 { IrType::F64 } else { IrType::I64 };
            let packed = self.fresh_value();
            self.emit(Instruction::VaArg {
                dest: packed,
                va_list_ptr,
                result_ty: read_ty,
            });

            // Store the packed 8 bytes to a temp alloca, then read back as 2 x F32
            let tmp_alloca = self.fresh_value();
            self.emit(Instruction::Alloca {
                dest: tmp_alloca,
                ty: IrType::Ptr,
                size: 8,
                align: 0,
                volatile: false,
            });
            self.emit(Instruction::Store { val: Operand::Value(packed), ptr: tmp_alloca, ty: read_ty,
             seg_override: AddressSpace::Default });

            // Load real part (first F32 at offset 0)
            let real_dest = self.fresh_value();
            self.emit(Instruction::Load { dest: real_dest, ptr: tmp_alloca, ty: IrType::F32,
             seg_override: AddressSpace::Default });

            // Load imag part (second F32 at offset +4)
            let imag_ptr = self.fresh_value();
            let ptr_int_ty = crate::common::types::target_int_ir_type();
            self.emit(Instruction::BinOp {
                dest: imag_ptr,
                op: IrBinOp::Add,
                lhs: Operand::Value(tmp_alloca),
                rhs: Operand::Const(IrConst::ptr_int(4)),
                ty: ptr_int_ty,
            });
            let imag_dest = self.fresh_value();
            self.emit(Instruction::Load { dest: imag_dest, ptr: imag_ptr, ty: IrType::F32,
             seg_override: AddressSpace::Default });

            // Allocate and store the complex float value
            let alloca = self.alloca_complex(ctype);
            self.store_complex_parts(alloca, Operand::Value(real_dest), Operand::Value(imag_dest), ctype);
            return Operand::Value(alloca);
        }

        // For complex double, complex long double, and float complex on ARM64:
        // read two separate values from the va_list
        let comp_ir_ty = Self::complex_component_ir_type(ctype);

        // On x86-64, _Complex double is classified as [Sse, Sse] - two SSE eightbytes.
        // Use VaArgStruct to atomically check that both FP slots are available,
        // preventing the complex value from being split across register and overflow areas.
        // _Complex long double (F128) is >16 bytes so it's MEMORY class and handled
        // via the overflow-only path (empty eightbyte_classes).
        if self.target == Target::X86_64 && comp_ir_ty == IrType::F64 {
            let alloca = self.alloca_complex(ctype);
            self.emit(Instruction::VaArgStruct {
                dest_ptr: alloca,
                va_list_ptr,
                size: 16, // 2 x 8 bytes
                eightbyte_classes: vec![
                    crate::common::types::EightbyteClass::Sse,
                    crate::common::types::EightbyteClass::Sse,
                ],
            });
            return Operand::Value(alloca);
        }

        // Retrieve real part via va_arg
        let real_dest = self.fresh_value();
        self.emit(Instruction::VaArg {
            dest: real_dest,
            va_list_ptr,
            result_ty: comp_ir_ty,
        });

        // Retrieve imaginary part via va_arg
        let imag_dest = self.fresh_value();
        self.emit(Instruction::VaArg {
            dest: imag_dest,
            va_list_ptr,
            result_ty: comp_ir_ty,
        });

        // Allocate stack space and store both components
        let alloca = self.alloca_complex(ctype);
        self.store_complex_parts(alloca, Operand::Value(real_dest), Operand::Value(imag_dest), ctype);

        Operand::Value(alloca)
    }

    /// Get a pointer to the va_list struct from an expression.
    /// Used by va_start, va_end, va_copy builtins.
    ///
    /// Target-dependent behavior (same logic as lower_va_arg):
    /// - x86-64/AArch64: va_list is array type, lower_expr handles both local
    ///   (array decay gives address) and parameter (loads pointer) cases.
    /// - RISC-V: va_list is void*, always need address-of the variable.
    pub(super) fn lower_va_list_pointer(&mut self, ap_expr: &Expr) -> Operand {
        use crate::backend::Target;
        if self.target == Target::Riscv64 || self.target == Target::I686 {
            // RISC-V / i686: va_list is a pointer, need address of the variable
            self.lower_address_of(ap_expr)
        } else {
            self.lower_expr(ap_expr)
        }
    }

    pub(super) fn resolve_va_arg_type(&self, type_spec: &TypeSpecifier) -> IrType {
        // Use type_spec_to_ctype for typedef resolution, then canonical CType-to-IrType
        let ctype = self.type_spec_to_ctype(type_spec);
        IrType::from_ctype(&ctype)
    }
}

