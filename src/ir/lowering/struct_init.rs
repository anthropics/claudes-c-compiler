//! Struct/union initialization lowering.
//!
//! This module handles emitting IR for initializing struct and union values from
//! C initializer lists, including:
//! - Positional (flat) initialization across struct fields
//! - Designated initializers (.field = val, .field[idx] = val, .a.b = val)
//! - Anonymous struct/union member drilling
//! - Nested struct/union fields (recursive)
//! - Array fields (with flat continuation per C11 6.7.9p17)
//! - Complex number fields
//! - Bitfield fields
//! - Scalar fields with implicit casts
//!
//! The main entry point is `emit_struct_init`, which dispatches to per-field-type
//! helpers to keep each case manageable.

use crate::frontend::parser::ast::{
    Designator,
    Expr,
    Initializer,
    InitializerItem,
};
use crate::ir::reexports::{
    Instruction,
    IrConst,
    Operand,
    Value,
};
use crate::common::types::{AddressSpace, IrType, CType, StructLayout, StructFieldLayout, InitFieldResolution};
use super::lower::Lowerer;

impl Lowerer {
    /// Recursively emit struct field initialization from an initializer list.
    ///
    /// `base_alloca` is the alloca for the struct, `base_offset` is the byte offset
    /// from the outermost struct (for nested structs).
    ///
    /// Returns the number of initializer items consumed (for flat init across nested structs).
    pub(super) fn emit_struct_init(
        &mut self,
        items: &[InitializerItem],
        base_alloca: Value,
        layout: &StructLayout,
        base_offset: usize,
    ) -> usize {
        let mut item_idx = 0usize;
        let mut current_field_idx = 0usize;

        while item_idx < items.len() {
            let item = &items[item_idx];

            let desig_name = match item.designators.first() {
                Some(Designator::Field(ref name)) => Some(&**name),
                _ => None,
            };
            // Check for array index designator (e.g., .field[idx] or bare [idx])
            let array_start_idx = self.extract_array_start_index(item, desig_name.is_some());

            // Resolve which field this init item targets
            let resolution = layout.resolve_init_field(desig_name, current_field_idx, &*self.types.borrow_struct_layouts());
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => {
                    let f = &layout.fields[*idx];
                    // For positional init, if this is an anonymous struct/union member,
                    // drill into it and consume multiple init items for inner fields.
                    if desig_name.is_none() && f.name.is_empty() && f.bit_width.is_none() {
                        if let CType::Struct(key) | CType::Union(key) = &f.ty {
                            let sub_layout = self.types.borrow_struct_layouts().get(&**key).cloned();
                            if let Some(sub_layout) = sub_layout {
                                let anon_offset = base_offset + f.offset;
                                // If the current item is a braced sub-initializer (List),
                                // it is the initializer for this anonymous member as a whole.
                                // Unwrap and pass the inner items to the sub-layout.
                                // This handles `(T){ { .field = val } }` where the inner
                                // braces wrap the anonymous union/struct member.
                                if let Initializer::List(sub_items) = &item.init {
                                    self.emit_struct_init(sub_items, base_alloca, &sub_layout, anon_offset);
                                    item_idx += 1;
                                    current_field_idx = *idx + 1;
                                    continue;
                                }
                                let anon_field_count = sub_layout.fields.iter()
                                    .filter(|ff| !ff.name.is_empty() || ff.bit_width.is_none())
                                    .count();
                                let remaining = &items[item_idx..];
                                let consume_count = remaining.len().min(anon_field_count);
                                let consumed = self.emit_struct_init(
                                    &remaining[..consume_count], base_alloca, &sub_layout, anon_offset,
                                );
                                item_idx += consumed.max(1);
                                current_field_idx = *idx + 1;
                                continue;
                            }
                        }
                    }
                    *idx
                }
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Designated init targets a field inside an anonymous struct/union member.
                    let anon_field = &layout.fields[*anon_field_idx].clone();
                    let anon_offset = base_offset + anon_field.offset;
                    let sub_layout = match &anon_field.ty {
                        CType::Struct(key) | CType::Union(key) => {
                            match self.types.borrow_struct_layouts().get(&**key) {
                                Some(l) => l.clone(),
                                None => { item_idx += 1; current_field_idx = *anon_field_idx + 1; continue; }
                            }
                        }
                        _ => { item_idx += 1; current_field_idx = *anon_field_idx + 1; continue; }
                    };
                    let mut synth_desigs = vec![Designator::Field(inner_name.clone())];
                    if item.designators.len() > 1 {
                        synth_desigs.extend(item.designators[1..].iter().cloned());
                    }
                    let sub_item = InitializerItem {
                        designators: synth_desigs,
                        init: item.init.clone(),
                    };
                    self.emit_struct_init(&[sub_item], base_alloca, &sub_layout, anon_offset);
                    item_idx += 1;
                    current_field_idx = *anon_field_idx + 1;
                    continue;
                }
                None => break,
            };
            let field = &layout.fields[field_idx].clone();
            let field_offset = base_offset + field.offset;

            // Detect designators that target nested or anonymous members
            let is_anon_member_designator = desig_name.is_some()
                && field.name.is_empty()
                && matches!(&field.ty, CType::Struct(_) | CType::Union(_));
            let has_nested_designator = item.designators.len() > 1
                && matches!(item.designators.first(), Some(Designator::Field(_)));

            // Dispatch to per-field-type handler
            match &field.ty {
                CType::Struct(key) | CType::Union(key) if has_nested_designator || is_anon_member_designator => {
                    self.emit_field_nested_designator(
                        item, base_alloca, key, field_offset,
                        is_anon_member_designator, has_nested_designator,
                    );
                    item_idx += 1;
                }
                CType::Array(elem_ty, Some(arr_size)) if has_nested_designator => {
                    self.emit_field_array_designated(
                        item, items, &mut item_idx, base_alloca,
                        elem_ty, *arr_size, field_offset,
                    );
                }
                CType::Struct(key) | CType::Union(key) => {
                    self.emit_field_substruct(
                        item, items, &mut item_idx, base_alloca,
                        key, field_offset,
                    );
                }
                CType::Array(elem_ty, Some(arr_size)) => {
                    self.emit_field_array(
                        item, items, &mut item_idx, base_alloca,
                        elem_ty, *arr_size, field_offset, array_start_idx,
                    );
                }
                CType::ComplexFloat | CType::ComplexDouble | CType::ComplexLongDouble => {
                    self.emit_field_complex(item, base_alloca, field_offset, &field.ty);
                    item_idx += 1;
                }
                CType::Vector(ref elem_ty, total_size) => {
                    self.emit_field_vector(
                        item, items, &mut item_idx, base_alloca,
                        elem_ty, *total_size, field_offset,
                    );
                }
                _ => {
                    self.emit_field_scalar(item, base_alloca, field_offset, field);
                    item_idx += 1;
                }
            }

            current_field_idx = field_idx + 1;

            // For unions without a designator, only the first field is initialized (C11 6.7.9).
            if layout.is_union && desig_name.is_none() {
                break;
            }
        }
        item_idx
    }

    // ========================================================================
    // Designator helpers
    // ========================================================================

    /// Extract the array start index from an initializer item's designators.
    /// Handles both `.field[idx]` and bare `[idx]` patterns.
    fn extract_array_start_index(&mut self, item: &InitializerItem, has_field_desig: bool) -> Option<usize> {
        if has_field_desig {
            item.designators.iter().find_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr_for_designator(idx_expr)
                } else {
                    None
                }
            })
        } else {
            match item.designators.first() {
                Some(Designator::Index(ref idx_expr)) => {
                    self.eval_const_expr_for_designator(idx_expr)
                }
                _ => None,
            }
        }
    }

    // ========================================================================
    // Per-field-type handlers
    // ========================================================================

    /// Handle a nested designator targeting a sub-struct field (e.g., `.a.j = 2`)
    /// or an anonymous struct/union member.
    fn emit_field_nested_designator(
        &mut self,
        item: &InitializerItem,
        base_alloca: Value,
        key: &str,
        field_offset: usize,
        is_anon_member_designator: bool,
        has_nested_designator: bool,
    ) {
        let sub_layout = self.types.borrow_struct_layouts().get(key).cloned();
        if let Some(sub_layout) = sub_layout {
            let sub_designators = if is_anon_member_designator && !has_nested_designator {
                item.designators.clone()
            } else {
                item.designators[1..].to_vec()
            };
            let sub_item = InitializerItem {
                designators: sub_designators,
                init: item.init.clone(),
            };
            self.emit_struct_init(&[sub_item], base_alloca, &sub_layout, field_offset);
        }
    }

    /// Handle an array field with a nested designator like `.field[idx] = val`.
    ///
    /// After storing the designated element, continues consuming non-designated items
    /// for subsequent array positions per C11 6.7.9p17.
    fn emit_field_array_designated(
        &mut self,
        item: &InitializerItem,
        items: &[InitializerItem],
        item_idx: &mut usize,
        base_alloca: Value,
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
    ) {
        let elem_size = self.resolve_ctype_size(elem_ty);
        let elem_ir_ty = IrType::from_ctype(elem_ty);

        // Find the first index designator in the remaining designators
        let remaining_desigs = &item.designators[1..];
        let (first_idx_pos, idx) = remaining_desigs.iter().enumerate().find_map(|(i, d)| {
            if let Designator::Index(ref idx_expr) = d {
                self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()).map(|v| (i, v))
            } else {
                None
            }
        }).unwrap_or((0, 0));

        if idx < arr_size {
            let elem_offset = field_offset + idx * elem_size;
            let after_first_idx = &remaining_desigs[first_idx_pos + 1..];
            let has_field_desigs = after_first_idx.iter().any(|d| matches!(d, Designator::Field(_)));
            let has_index_desigs = after_first_idx.iter().any(|d| matches!(d, Designator::Index(_)));

            if has_field_desigs {
                // .a[idx].b = val — drill into struct element
                self.emit_array_elem_struct_drill(item, base_alloca, elem_ty, elem_offset, after_first_idx);
            } else if has_index_desigs {
                // Multi-dimensional: .a[1][2] = val
                let remaining_index_desigs: Vec<_> = after_first_idx.iter()
                    .filter(|d| matches!(d, Designator::Index(_)))
                    .cloned()
                    .collect();
                self.emit_array_elem_multidim(item, base_alloca, elem_ty, elem_offset, &remaining_index_desigs);
            } else {
                // No further designators — store element value
                self.emit_array_elem_value(item, base_alloca, elem_ty, elem_offset, elem_ir_ty);
            }
        }
        *item_idx += 1;

        // Continue consuming subsequent non-designated items for array
        // positions idx+1, idx+2, ... (C11 6.7.9p17)
        let elem_is_bool = *elem_ty == CType::Bool;
        let mut ai = idx + 1;
        while ai < arr_size && *item_idx < items.len() {
            let next_item = &items[*item_idx];
            if !next_item.designators.is_empty() { break; }
            let expr_opt = match &next_item.init {
                Initializer::Expr(e) => Some(e),
                Initializer::List(sub_items) => Self::unwrap_nested_init_expr(sub_items),
            };
            if let Some(e) = expr_opt {
                let elem_offset = field_offset + ai * elem_size;
                self.emit_init_expr_to_offset_bool(e, base_alloca, elem_offset, elem_ir_ty, elem_is_bool);
            } else {
                break;
            }
            *item_idx += 1;
            ai += 1;
        }
    }

    /// Drill into a struct element of an array: `.a[idx].b = val`
    fn emit_array_elem_struct_drill(
        &mut self,
        item: &InitializerItem,
        base_alloca: Value,
        elem_ty: &CType,
        elem_offset: usize,
        after_first_idx: &[Designator],
    ) {
        if let CType::Struct(ref key) | CType::Union(ref key) = elem_ty {
            let sub_layout = self.types.borrow_struct_layouts().get(&**key).cloned();
            if let Some(sub_layout) = sub_layout {
                let sub_desigs: Vec<_> = after_first_idx.to_vec();
                let sub_item = InitializerItem {
                    designators: sub_desigs,
                    init: item.init.clone(),
                };
                self.emit_struct_init(&[sub_item], base_alloca, &sub_layout, elem_offset);
            }
        }
    }

    /// Handle multi-dimensional array designator: `.a[1][2] = val`
    fn emit_array_elem_multidim(
        &mut self,
        item: &InitializerItem,
        base_alloca: Value,
        elem_ty: &CType,
        elem_offset: usize,
        remaining_index_desigs: &[Designator],
    ) {
        if let CType::Array(inner_elem_ty, Some(inner_size)) = elem_ty {
            let inner_idx = remaining_index_desigs.iter().find_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            }).unwrap_or(0);
            if inner_idx < *inner_size {
                let inner_elem_size = self.resolve_ctype_size(inner_elem_ty);
                let inner_ir_ty = IrType::from_ctype(inner_elem_ty);
                let inner_is_bool = **inner_elem_ty == CType::Bool;
                let inner_offset = elem_offset + inner_idx * inner_elem_size;
                if let Initializer::Expr(e) = &item.init {
                    if let Expr::StringLiteral(s, _) = e {
                        if matches!(inner_elem_ty.as_ref(), CType::Char | CType::UChar) {
                            let max_bytes = (*inner_size - inner_idx) * inner_elem_size;
                            self.emit_string_to_alloca(base_alloca, s, inner_offset, max_bytes);
                            return;
                        }
                    }
                    self.emit_init_expr_to_offset_bool(e, base_alloca, inner_offset, inner_ir_ty, inner_is_bool);
                }
            }
        }
    }

    /// Store a single element value at an array offset, handling string literals
    /// and sub-list initializers.
    fn emit_array_elem_value(
        &mut self,
        item: &InitializerItem,
        base_alloca: Value,
        elem_ty: &CType,
        elem_offset: usize,
        elem_ir_ty: IrType,
    ) {
        match &item.init {
            Initializer::Expr(e) => {
                // Check for string literal targeting a char sub-array
                if let Expr::StringLiteral(s, _) = e {
                    if let CType::Array(inner, Some(arr_size)) = elem_ty {
                        if matches!(inner.as_ref(), CType::Char | CType::UChar) {
                            self.emit_string_to_alloca(base_alloca, s, elem_offset, *arr_size);
                            return;
                        }
                    }
                }
                self.emit_init_expr_to_offset_bool(e, base_alloca, elem_offset, elem_ir_ty, *elem_ty == CType::Bool);
            }
            Initializer::List(sub_items) => {
                // Handle list init for array element (e.g., .a[1] = {1,2,3})
                match elem_ty {
                    CType::Array(inner_elem_ty, Some(inner_size)) => {
                        let inner_elem_size = self.resolve_ctype_size(inner_elem_ty);
                        let inner_ir_ty = IrType::from_ctype(inner_elem_ty);
                        let inner_is_bool = **inner_elem_ty == CType::Bool;
                        for (si, sub_item) in sub_items.iter().enumerate() {
                            if si >= *inner_size { break; }
                            if let Initializer::Expr(e) = &sub_item.init {
                                let inner_offset = elem_offset + si * inner_elem_size;
                                self.emit_init_expr_to_offset_bool(e, base_alloca, inner_offset, inner_ir_ty, inner_is_bool);
                            }
                        }
                    }
                    CType::Struct(ref key) | CType::Union(ref key) => {
                        let sub_layout = self.types.borrow_struct_layouts().get(&**key).cloned();
                        if let Some(sub_layout) = sub_layout {
                            self.emit_struct_init(sub_items, base_alloca, &sub_layout, elem_offset);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Handle a struct/union field that is itself a struct/union (non-designated).
    fn emit_field_substruct(
        &mut self,
        item: &InitializerItem,
        items: &[InitializerItem],
        item_idx: &mut usize,
        base_alloca: Value,
        key: &str,
        field_offset: usize,
    ) {
        let sub_layout = self.types.borrow_struct_layouts().get(key).cloned();
        if let Some(sub_layout) = sub_layout {
            match &item.init {
                Initializer::List(sub_items) => {
                    // Zero-init the sub-struct/union region before writing explicit values.
                    // C11 6.7.9p21: unspecified members are implicitly zero-initialized.
                    self.zero_init_region(base_alloca, field_offset, sub_layout.size);
                    self.emit_struct_init(sub_items, base_alloca, &sub_layout, field_offset);
                    *item_idx += 1;
                }
                Initializer::Expr(expr) => {
                    if self.struct_value_size(expr).is_some() {
                        // Struct/union copy in init list
                        let src_addr = self.get_struct_base_addr(expr);
                        self.emit_memcpy_at_offset(base_alloca, field_offset, src_addr, sub_layout.size);
                        *item_idx += 1;
                    } else {
                        // Flat init: consume items for inner struct/union fields.
                        // If the current item has a designator (e.g., `.y = 42` where y is
                        // a struct), strip it so the scalar is treated as a positional init
                        // for the first field of the sub-struct (C11 6.7.9p13).
                        if !item.designators.is_empty() {
                            let stripped = InitializerItem {
                                designators: vec![],
                                init: item.init.clone(),
                            };
                            self.zero_init_region(base_alloca, field_offset, sub_layout.size);
                            self.emit_struct_init(&[stripped], base_alloca, &sub_layout, field_offset);
                            *item_idx += 1;
                        } else {
                            let consumed = self.emit_struct_init(&items[*item_idx..], base_alloca, &sub_layout, field_offset);
                            if consumed == 0 { *item_idx += 1; } else { *item_idx += consumed; }
                        }
                    }
                }
            }
        } else {
            *item_idx += 1;
        }
    }

    /// Handle an array field (non-designated case).
    /// Dispatches between Initializer::List and Initializer::Expr.
    fn emit_field_array(
        &mut self,
        item: &InitializerItem,
        items: &[InitializerItem],
        item_idx: &mut usize,
        base_alloca: Value,
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
        array_start_idx: Option<usize>,
    ) {
        let elem_size = self.resolve_ctype_size(elem_ty);
        match &item.init {
            Initializer::List(sub_items) => {
                self.emit_array_field_list_init(
                    sub_items, base_alloca, elem_ty, arr_size, field_offset, elem_size,
                );
                *item_idx += 1;
            }
            Initializer::Expr(e) => {
                self.emit_array_field_expr_init(
                    e, items, item_idx, base_alloca,
                    elem_ty, arr_size, field_offset, elem_size, array_start_idx,
                );
            }
        }
    }

    // ========================================================================
    // Array field sub-handlers
    // ========================================================================

    /// Handle array field from an Initializer::List.
    fn emit_array_field_list_init(
        &mut self,
        sub_items: &[InitializerItem],
        base_alloca: Value,
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
        elem_size: usize,
    ) {
        // Check for brace-wrapped string literal: {"hello"} for char[]
        if sub_items.len() == 1 && sub_items[0].designators.is_empty() {
            if let Initializer::Expr(Expr::StringLiteral(s, _)) = &sub_items[0].init {
                if matches!(elem_ty, CType::Char | CType::UChar) {
                    self.emit_string_to_alloca(base_alloca, s, field_offset, arr_size * elem_size);
                    return;
                }
            }
        }

        if let CType::Struct(ref key) | CType::Union(ref key) = elem_ty {
            // Array of structs
            let sub_layout = self.types.borrow_struct_layouts().get(&**key).cloned();
            if let Some(sub_layout) = sub_layout {
                self.emit_array_of_structs_list_init(
                    sub_items, base_alloca, &sub_layout, arr_size, field_offset, elem_size,
                );
            }
        } else if elem_ty.is_complex() {
            self.emit_array_of_complex_list_init(
                sub_items, base_alloca, elem_ty, arr_size, field_offset, elem_size,
            );
        } else {
            self.emit_array_of_scalars_list_init(
                sub_items, base_alloca, elem_ty, arr_size, field_offset, elem_size,
            );
        }
    }

    /// Init array of structs from a brace list.
    fn emit_array_of_structs_list_init(
        &mut self,
        sub_items: &[InitializerItem],
        base_alloca: Value,
        sub_layout: &StructLayout,
        arr_size: usize,
        field_offset: usize,
        elem_size: usize,
    ) {
        let mut ai = 0;
        let mut si = 0;
        while si < sub_items.len() && ai < arr_size {
            let elem_offset = field_offset + ai * elem_size;
            match &sub_items[si].init {
                Initializer::List(struct_items) => {
                    self.emit_struct_init(struct_items, base_alloca, sub_layout, elem_offset);
                    si += 1;
                    ai += 1;
                }
                Initializer::Expr(e) => {
                    if self.struct_value_size(e).is_some() {
                        let src_addr = self.get_struct_base_addr(e);
                        self.emit_memcpy_at_offset(base_alloca, elem_offset, src_addr, sub_layout.size);
                        si += 1;
                        ai += 1;
                    } else {
                        let consumed = self.emit_struct_init(&sub_items[si..], base_alloca, sub_layout, elem_offset);
                        si += consumed;
                        ai += 1;
                    }
                }
            }
        }
    }

    /// Init array of complex elements from a brace list.
    fn emit_array_of_complex_list_init(
        &mut self,
        sub_items: &[InitializerItem],
        base_alloca: Value,
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
        elem_size: usize,
    ) {
        let complex_ctype = elem_ty.clone();
        let mut ai = 0usize;
        for sub_item in sub_items {
            if let Some(Designator::Index(ref idx_expr)) = sub_item.designators.first() {
                if let Some(idx) = self.eval_const_expr_for_designator(idx_expr) {
                    ai = idx;
                }
            }
            if ai >= arr_size { break; }
            let elem_offset = field_offset + ai * elem_size;
            match &sub_item.init {
                Initializer::Expr(e) => {
                    self.emit_complex_expr_to_offset(e, base_alloca, elem_offset, &complex_ctype);
                }
                Initializer::List(inner_items) => {
                    let dest_addr = self.emit_gep_offset(base_alloca, elem_offset, IrType::Ptr);
                    self.lower_complex_list_init(inner_items, dest_addr, &complex_ctype);
                }
            }
            ai += 1;
        }
    }

    /// Init array of scalars from a brace list with [idx]=val designator support.
    fn emit_array_of_scalars_list_init(
        &mut self,
        sub_items: &[InitializerItem],
        base_alloca: Value,
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
        elem_size: usize,
    ) {
        let elem_is_bool = *elem_ty == CType::Bool;
        let mut ai = 0usize;
        for sub_item in sub_items {
            if let Some(Designator::Index(ref idx_expr)) = sub_item.designators.first() {
                if let Some(idx) = self.eval_const_expr_for_designator(idx_expr) {
                    ai = idx;
                }
            }
            if ai >= arr_size { break; }
            let elem_offset = field_offset + ai * elem_size;
            match &sub_item.init {
                Initializer::Expr(e) => {
                    // String literal targeting a char sub-array: copy the string
                    // into the array memory instead of storing a pointer.
                    if let Expr::StringLiteral(s, _) = e {
                        if let CType::Array(inner, Some(sub_arr_size)) = elem_ty {
                            if matches!(inner.as_ref(), CType::Char | CType::UChar) {
                                self.emit_string_to_alloca(base_alloca, s, elem_offset, *sub_arr_size);
                                ai += 1;
                                continue;
                            }
                        }
                    }
                    let elem_ir_ty = IrType::from_ctype(elem_ty);
                    self.emit_init_expr_to_offset_bool(e, base_alloca, elem_offset, elem_ir_ty, elem_is_bool);
                }
                Initializer::List(inner_items) => {
                    // Handle braced sub-init for array elements (e.g., int arr[2][3] = {{1,2,3},{4,5,6}})
                    if let CType::Array(inner_elem_ty, Some(inner_size)) = elem_ty {
                        let inner_elem_ir_ty = IrType::from_ctype(inner_elem_ty);
                        let inner_is_bool = **inner_elem_ty == CType::Bool;
                        let inner_elem_size = self.resolve_ctype_size(inner_elem_ty);
                        for (ii, inner_item) in inner_items.iter().enumerate() {
                            if ii >= *inner_size { break; }
                            if let Initializer::Expr(e) = &inner_item.init {
                                let inner_offset = elem_offset + ii * inner_elem_size;
                                // String literal targeting a char sub-array: copy string contents
                                if let Expr::StringLiteral(s, _) = e {
                                    if let CType::Array(inner_inner, Some(sub_arr_size)) = inner_elem_ty.as_ref() {
                                        if matches!(inner_inner.as_ref(), CType::Char | CType::UChar) {
                                            self.emit_string_to_alloca(base_alloca, s, inner_offset, *sub_arr_size);
                                            continue;
                                        }
                                    }
                                }
                                self.emit_init_expr_to_offset_bool(e, base_alloca, inner_offset, inner_elem_ir_ty, inner_is_bool);
                            }
                        }
                    } else if let Some(e) = Self::unwrap_nested_init_expr(inner_items) {
                        // Extra braces around scalar array element: {{{42}}}
                        let elem_ir_ty = IrType::from_ctype(elem_ty);
                        self.emit_init_expr_to_offset_bool(e, base_alloca, elem_offset, elem_ir_ty, elem_is_bool);
                    }
                }
            }
            ai += 1;
        }
    }

    // ========================================================================
    // Flat (non-braced) array field init
    // ========================================================================

    /// Handle an array field initialized by a flat expression (not a braced list).
    fn emit_array_field_expr_init(
        &mut self,
        e: &Expr,
        items: &[InitializerItem],
        item_idx: &mut usize,
        base_alloca: Value,
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
        elem_size: usize,
        array_start_idx: Option<usize>,
    ) {
        // String literal for char array field
        if let Expr::StringLiteral(s, _) = e {
            if matches!(elem_ty, CType::Char | CType::UChar) {
                self.emit_string_to_alloca(base_alloca, s, field_offset, arr_size * elem_size);
                *item_idx += 1;
                return;
            }
        }

        if let CType::Struct(ref key) | CType::Union(ref key) = elem_ty {
            // Flat init for array of structs
            let sub_layout = self.types.borrow_struct_layouts().get(&**key).cloned();
            if let Some(sub_layout) = sub_layout {
                let start_ai = array_start_idx.unwrap_or(0);
                let mut total_consumed = 0usize;
                for ai in start_ai..arr_size {
                    if *item_idx + total_consumed >= items.len() { break; }
                    let elem_offset = field_offset + ai * elem_size;
                    let consumed = self.emit_struct_init(&items[*item_idx + total_consumed..], base_alloca, &sub_layout, elem_offset);
                    total_consumed += consumed;
                }
                *item_idx += total_consumed.max(1);
            } else {
                *item_idx += 1;
            }
        } else if elem_ty.is_complex() {
            // Flat init for array of complex
            let complex_ctype = elem_ty.clone();
            let start_ai = array_start_idx.unwrap_or(0);
            let mut consumed = 0usize;
            let mut ai = start_ai;
            while ai < arr_size && (*item_idx + consumed) < items.len() {
                let cur_item = &items[*item_idx + consumed];
                if !cur_item.designators.is_empty() && consumed > 0 { break; }
                let elem_offset = field_offset + ai * elem_size;
                match &cur_item.init {
                    Initializer::Expr(expr) => {
                        self.emit_complex_expr_to_offset(expr, base_alloca, elem_offset, &complex_ctype);
                    }
                    Initializer::List(inner_items) => {
                        let dest_addr = self.emit_gep_offset(base_alloca, elem_offset, IrType::Ptr);
                        self.lower_complex_list_init(inner_items, dest_addr, &complex_ctype);
                    }
                }
                consumed += 1;
                ai += 1;
            }
            *item_idx += consumed.max(1);
        } else {
            // Flat init for scalar array
            let elem_ir_ty = IrType::from_ctype(elem_ty);
            let elem_is_bool = *elem_ty == CType::Bool;
            let start_ai = array_start_idx.unwrap_or(0);
            let mut consumed = 0usize;
            let mut ai = start_ai;
            while ai < arr_size && (*item_idx + consumed) < items.len() {
                let cur_item = &items[*item_idx + consumed];
                if !cur_item.designators.is_empty() && consumed > 0 { break; }
                let expr_opt = match &cur_item.init {
                    Initializer::Expr(expr) => Some(expr),
                    Initializer::List(sub_items) => Self::unwrap_nested_init_expr(sub_items),
                };
                if let Some(expr) = expr_opt {
                    let elem_offset = field_offset + ai * elem_size;
                    self.emit_init_expr_to_offset_bool(expr, base_alloca, elem_offset, elem_ir_ty, elem_is_bool);
                    consumed += 1;
                    ai += 1;
                } else {
                    break;
                }
            }
            *item_idx += consumed.max(1);
        }
    }

    // ========================================================================
    // Leaf field handlers (scalar and complex)
    // ========================================================================

    /// Handle a complex field (ComplexFloat, ComplexDouble, ComplexLongDouble).
    fn emit_field_complex(
        &mut self,
        item: &InitializerItem,
        base_alloca: Value,
        field_offset: usize,
        field_ty: &CType,
    ) {
        let complex_ctype = field_ty.clone();
        let complex_size = complex_ctype.size();
        let dest_addr = self.emit_gep_offset(base_alloca, field_offset, IrType::Ptr);
        match &item.init {
            Initializer::Expr(e) => {
                let src = self.lower_expr_to_complex(e, &complex_ctype);
                self.emit(Instruction::Memcpy {
                    dest: dest_addr,
                    src,
                    size: complex_size,
                });
            }
            Initializer::List(sub_items) => {
                self.lower_complex_list_init(sub_items, dest_addr, &complex_ctype);
            }
        }
    }

    /// Handle a vector-type field: e.g., `float4 a` inside a struct/union.
    /// When the initializer is a list `{1,2,3,4}`, store elements individually.
    /// When the initializer is an expression, memcpy the vector value.
    /// Also consumes flat (un-braced) trailing items as subsequent vector elements
    /// per C11 6.7.9p17 semantics.
    fn emit_field_vector(
        &mut self,
        item: &InitializerItem,
        items: &[InitializerItem],
        item_idx: &mut usize,
        base_alloca: Value,
        elem_ty: &CType,
        total_size: usize,
        field_offset: usize,
    ) {
        let elem_size = elem_ty.size();
        let num_elems = if elem_size > 0 { total_size / elem_size } else { 0 };
        let elem_ir_ty = IrType::from_ctype(elem_ty);
        let field_addr = self.emit_gep_offset(base_alloca, field_offset, IrType::Ptr);

        match &item.init {
            Initializer::List(sub_items) => {
                // List initializer: {1.0f, 2.0f, 3.0f, 4.0f}
                // Use the existing vector init list helper
                self.lower_vector_init_list(sub_items, field_addr, elem_ty, num_elems);
                *item_idx += 1;
            }
            Initializer::Expr(e) => {
                // Expression initializer: could be a vector expression or a scalar
                let expr_ct = self.expr_ctype(e);
                if expr_ct.is_vector() {
                    // Vector expression: memcpy
                    let src = self.lower_expr(e);
                    let src_val = self.operand_to_value(src);
                    self.emit(Instruction::Memcpy {
                        dest: field_addr,
                        src: src_val,
                        size: total_size,
                    });
                    *item_idx += 1;
                } else {
                    // Scalar: store as first element, then consume trailing flat items
                    // for remaining elements (C11 6.7.9p17: flat init continues to next element)
                    let val = self.lower_expr(e);
                    let expr_ty = self.get_expr_type(e);
                    let val = self.emit_implicit_cast(val, expr_ty, elem_ir_ty);
                    self.emit_array_element_store(field_addr, val, 0, elem_ir_ty);
                    *item_idx += 1;

                    // Consume subsequent non-designated items as remaining vector elements
                    let mut ei = 1;
                    while ei < num_elems && *item_idx < items.len() {
                        let next_item = &items[*item_idx];
                        if !next_item.designators.is_empty() { break; }
                        let expr_opt = match &next_item.init {
                            Initializer::Expr(ne) => Some(ne),
                            Initializer::List(sub_items) => Self::unwrap_nested_init_expr(sub_items),
                        };
                        if let Some(ne) = expr_opt {
                            let nval = self.lower_expr(ne);
                            let net = self.get_expr_type(ne);
                            let nval = self.emit_implicit_cast(nval, net, elem_ir_ty);
                            let offset = ei * elem_size;
                            self.emit_array_element_store(field_addr, nval, offset, elem_ir_ty);
                        } else {
                            break;
                        }
                        *item_idx += 1;
                        ei += 1;
                    }
                }
            }
        }
    }

    /// Handle a scalar field (including bitfields).
    fn emit_field_scalar(
        &mut self,
        item: &InitializerItem,
        base_alloca: Value,
        field_offset: usize,
        field: &StructFieldLayout,
    ) {
        let field_ty = IrType::from_ctype(&field.ty);
        let (val, expr_ty) = match &item.init {
            Initializer::Expr(e) => {
                let et = self.get_expr_type(e);
                (self.lower_expr(e), et)
            }
            Initializer::List(sub_items) => {
                // Scalar with arbitrarily nested braces: {5} or {{{5}}}
                if let Some(e) = Self::unwrap_nested_init_expr(sub_items) {
                    let et = self.get_expr_type(e);
                    (self.lower_expr(e), et)
                } else {
                    (Operand::Const(IrConst::ptr_int(0)), crate::common::types::target_int_ir_type())
                }
            }
        };
        // For _Bool fields, normalize (any nonzero -> 1) before truncation (C11 6.3.1.2).
        let val = if field.ty == CType::Bool {
            self.emit_bool_normalize_typed(val, expr_ty)
        } else {
            self.emit_implicit_cast(val, expr_ty, field_ty)
        };
        let addr = self.emit_gep_offset(base_alloca, field_offset, field_ty);
        if let (Some(bit_offset), Some(bit_width)) = (field.bit_offset, field.bit_width) {
            self.store_bitfield(addr, field_ty, bit_offset, bit_width, val);
        } else {
            self.emit(Instruction::Store { val, ptr: addr, ty: field_ty , seg_override: AddressSpace::Default });
        }
    }
}
