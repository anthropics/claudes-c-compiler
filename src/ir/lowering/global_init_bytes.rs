//! Byte-serialization portion of global initialization.
//!
//! This module contains the byte-level serialization methods used during global
//! variable initialization lowering. It handles writing constants, bitfields,
//! complex numbers, struct layouts, and array fills into byte buffers.

use std::rc::Rc;
use crate::frontend::parser::ast::{
    Designator,
    Expr,
    Initializer,
    InitializerItem,
};
use crate::ir::reexports::{GlobalInit, IrConst};
use crate::common::types::{IrType, StructLayout, CType};
use super::lower::Lowerer;
use super::global_init_helpers as h;

/// Result of filling an array/FAM field in fill_struct_global_bytes,
/// indicating how to advance the item index.
pub(super) struct ArrayFillResult {
    new_item_idx: usize,
    /// If true, caller should `continue` (skip the default field_idx update).
    skip_update: bool,
}

/// Result of drilling through designator chains to find the target type and offset.
pub(super) struct DesignatorDrillResult {
    /// The final target type after following all designators
    pub target_ty: CType,
    /// The byte offset from the outer field's start to the target
    pub byte_offset: usize,
    /// For bitfields: bit offset within the storage unit at `byte_offset`.
    pub bit_offset: Option<u32>,
    /// For bitfields: width in bits.
    pub bit_width: Option<u32>,
}

impl Lowerer {
    /// Recursively fill byte buffer for struct global initialization.
    /// Returns the number of initializer items consumed.
    ///
    /// Handles nested structs/unions, arrays (including multi-dimensional), string
    /// literals, designators, flexible array members, and bitfields.
    pub(super) fn fill_struct_global_bytes(
        &self,
        items: &[InitializerItem],
        layout: &StructLayout,
        bytes: &mut [u8],
        base_offset: usize,
    ) -> usize {
        let mut item_idx = 0usize;
        let mut current_field_idx = 0usize;

        while item_idx < items.len() {
            let item = &items[item_idx];

            let designator_name = h::first_field_designator(item);
            let array_start_idx = self.extract_index_designator(item, designator_name.is_some());
            let resolution = layout.resolve_init_field(designator_name, current_field_idx, &*self.types.borrow_struct_layouts());
            let field_idx = match &resolution {
                Some(crate::common::types::InitFieldResolution::Direct(idx)) => *idx,
                Some(crate::common::types::InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    let extra_desigs = if item.designators.len() > 1 { &item.designators[1..] } else { &[] };
                    if let Some(res) = h::resolve_anonymous_member(layout, *anon_field_idx, inner_name, &item.init, extra_desigs, &self.types.borrow_struct_layouts()) {
                        self.fill_struct_global_bytes(&[res.sub_item], &res.sub_layout, bytes, base_offset + res.anon_offset);
                    }
                    item_idx += 1;
                    current_field_idx = *anon_field_idx + 1;
                    continue;
                }
                None => {
                    if designator_name.is_some() { item_idx += 1; continue; }
                    break;
                }
            };
            let field_layout = &layout.fields[field_idx];
            let field_offset = base_offset + field_layout.offset;

            let is_anon = h::is_anon_member_designator(
                designator_name, &field_layout.name, &field_layout.ty);
            let has_nested = h::has_nested_field_designator(item);

            match &field_layout.ty {
                // Nested designator or anonymous member designator into struct/union
                CType::Struct(key) | CType::Union(key) if has_nested || is_anon => {
                    if let Some(sub_layout) = self.types.borrow_struct_layouts().get(&**key).cloned() {
                        let sub_item = if is_anon && !has_nested {
                            // Pass all designators through for anonymous member field lookup
                            item.clone()
                        } else {
                            InitializerItem {
                                designators: item.designators[1..].to_vec(),
                                init: item.init.clone(),
                            }
                        };
                        self.fill_struct_global_bytes(&[sub_item], &sub_layout, bytes, field_offset);
                    }
                    item_idx += 1;
                }
                // Nested designator into array: .field[idx] = val
                // After the designated element, continue consuming non-designated
                // items for subsequent array positions (C11 6.7.9p17).
                CType::Array(elem_ty, Some(arr_size)) if has_nested => {
                    let arr_size = *arr_size;
                    let desig_idx = self.fill_nested_designator_array(item, elem_ty, arr_size, bytes, field_offset);
                    item_idx += 1;
                    // Continue consuming subsequent non-designated items for positions
                    // desig_idx+1, desig_idx+2, ... within the same array field
                    let elem_size = self.resolve_ctype_size(elem_ty);
                    let elem_ir_ty = IrType::from_ctype(elem_ty);
                    let elem_is_bool = **elem_ty == CType::Bool;
                    let mut ai = desig_idx + 1;
                    while ai < arr_size && item_idx < items.len() {
                        let next_item = &items[item_idx];
                        if !next_item.designators.is_empty() {
                            break;
                        }
                        if let Initializer::Expr(ref expr) = next_item.init {
                            let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                            let val = if elem_is_bool { val.bool_normalize() } else { val };
                            let elem_offset = field_offset + ai * elem_size;
                            self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                        } else {
                            break;
                        }
                        item_idx += 1;
                        ai += 1;
                    }
                }
                // Struct or union field (non-nested designator)
                CType::Struct(key) | CType::Union(key) => {
                    if let Some(sub_layout) = self.types.borrow_struct_layouts().get(&**key).cloned() {
                        item_idx += self.fill_composite_field(
                            &items[item_idx..], &sub_layout, bytes, field_offset,
                        );
                    } else {
                        item_idx += 1;
                    }
                }
                // Fixed-size array field
                CType::Array(elem_ty, Some(arr_size)) => {
                    let advanced = self.fill_array_field(
                        items, item_idx, elem_ty, *arr_size,
                        bytes, field_offset, array_start_idx,
                    );
                    if advanced.skip_update {
                        item_idx = advanced.new_item_idx;
                        current_field_idx = field_idx + 1;
                        continue;
                    }
                    item_idx = advanced.new_item_idx;
                }
                // Flexible array member (FAM)
                CType::Array(elem_ty, None) => {
                    let advanced = self.fill_fam_field(
                        items, item_idx, elem_ty, bytes, field_offset,
                    );
                    if advanced.skip_update {
                        item_idx = advanced.new_item_idx;
                        current_field_idx = field_idx + 1;
                        continue;
                    }
                    item_idx = advanced.new_item_idx;
                }
                // Complex field: write {real, imag} pair to bytes
                CType::ComplexFloat | CType::ComplexDouble | CType::ComplexLongDouble => {
                    self.write_complex_field_to_bytes(bytes, field_offset, &field_layout.ty, &item.init);
                    item_idx += 1;
                }
                // Scalar field (possibly bitfield)
                _ => {
                    let field_ir_ty = IrType::from_ctype(&field_layout.ty);
                    let raw_val = self.eval_init_scalar(&item.init);
                    // _Bool fields: normalize (any nonzero -> 1) per C11 6.3.1.2
                    let val = if field_layout.ty == CType::Bool {
                        raw_val.bool_normalize()
                    } else {
                        raw_val
                    };
                    if let (Some(bit_offset), Some(bit_width)) = (field_layout.bit_offset, field_layout.bit_width) {
                        self.write_bitfield_to_bytes(bytes, field_offset, &val, field_ir_ty, bit_offset, bit_width);
                    } else {
                        self.write_const_to_bytes(bytes, field_offset, &val, field_ir_ty);
                    }
                    item_idx += 1;
                }
            }
            current_field_idx = field_idx + 1;

            if layout.is_union && designator_name.is_none() {
                break;
            }
        }
        item_idx
    }

    /// Drill through a slice of designators to find the target type and byte offset.
    /// This resolves chains like [Field("b"), Index(2), Field("x")] by walking through
    /// struct layouts and array element types to compute the final target type and offset.
    pub(super) fn drill_designators(
        &self,
        designators: &[Designator],
        start_ty: &CType,
    ) -> Option<DesignatorDrillResult> {
        let mut current_ty = start_ty.clone();
        let mut byte_offset = 0usize;
        // Track bitfield info from the last resolved field
        let mut bit_offset: Option<u32> = None;
        let mut bit_width: Option<u32> = None;

        for desig in designators {
            // Reset bitfield info on each step (only the final field matters)
            bit_offset = None;
            bit_width = None;

            match desig {
                Designator::Field(name) => {
                    let sub_layout = self.get_struct_layout_for_ctype(&current_ty)?;
                    let resolution = sub_layout.resolve_init_field(Some(&**name), 0, &*self.types.borrow_struct_layouts())?;
                    match resolution {
                        crate::common::types::InitFieldResolution::Direct(fi) => {
                            byte_offset += sub_layout.fields[fi].offset;
                            current_ty = sub_layout.fields[fi].ty.clone();
                            bit_offset = sub_layout.fields[fi].bit_offset;
                            bit_width = sub_layout.fields[fi].bit_width;
                        }
                        crate::common::types::InitFieldResolution::AnonymousMember { anon_field_idx, inner_name } => {
                            // Add anonymous member's offset, then resolve inner field
                            let anon_field = &sub_layout.fields[anon_field_idx];
                            byte_offset += anon_field.offset;
                            let anon_layout = self.get_struct_layout_for_ctype(&anon_field.ty)?;
                            let inner_fi = anon_layout.resolve_init_field_idx(Some(&*inner_name), 0, &*self.types.borrow_struct_layouts())?;
                            byte_offset += anon_layout.fields[inner_fi].offset;
                            current_ty = anon_layout.fields[inner_fi].ty.clone();
                            bit_offset = anon_layout.fields[inner_fi].bit_offset;
                            bit_width = anon_layout.fields[inner_fi].bit_width;
                        }
                    }
                }
                Designator::Index(idx_expr) => {
                    if let CType::Array(elem_ty, _) = &current_ty {
                        let idx = self.eval_const_expr(idx_expr)
                            .and_then(|c| c.to_usize())
                            .unwrap_or(0);
                        byte_offset += idx * self.resolve_ctype_size(elem_ty);
                        current_ty = elem_ty.as_ref().clone();
                    } else {
                        return None;
                    }
                }
                Designator::Range(_, _) => {
                    // Range designators are expanded to individual Index items during parsing
                    return None;
                }
            }
        }

        Some(DesignatorDrillResult { target_ty: current_ty, byte_offset, bit_offset, bit_width })
    }

    // --- fill_struct_global_bytes helpers ---

    /// Extract an array index designator from an initializer item.
    pub(super) fn extract_index_designator(&self, item: &InitializerItem, has_field_desig: bool) -> Option<usize> {
        if has_field_desig {
            item.designators.iter().find_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            })
        } else {
            match item.designators.first() {
                Some(Designator::Index(ref idx_expr)) => {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                }
                _ => None,
            }
        }
    }

    /// Evaluate an initializer to a scalar constant (handles both Expr and arbitrarily
    /// nested brace-wrapped Lists like `{{{42}}}`).
    pub(super) fn eval_init_scalar(&self, init: &Initializer) -> IrConst {
        match init {
            Initializer::Expr(expr) => self.eval_const_expr(expr).unwrap_or(IrConst::I64(0)),
            Initializer::List(sub_items) => {
                if let Some(first) = sub_items.first() {
                    self.eval_init_scalar(&first.init)
                } else {
                    IrConst::I64(0)
                }
            }
        }
    }

    /// Write a string literal into a byte buffer at the given offset, with null terminator.
    /// Each char in the string is treated as a raw byte value (0-255).
    pub(super) fn write_string_to_bytes(bytes: &mut [u8], offset: usize, s: &str, max_len: usize) {
        let str_chars: Vec<u8> = s.chars().map(|c| c as u8).collect();
        for (i, &b) in str_chars.iter().enumerate() {
            if i >= max_len { break; }
            if offset + i < bytes.len() {
                bytes[offset + i] = b;
            }
        }
        if str_chars.len() < max_len && offset + str_chars.len() < bytes.len() {
            bytes[offset + str_chars.len()] = 0;
        }
    }

    /// Process a nested designator into an array field (e.g., .field[idx] = val).
    /// Returns the array index that was designated.
    pub(super) fn fill_nested_designator_array(
        &self, item: &InitializerItem, elem_ty: &CType, arr_size: usize,
        bytes: &mut [u8], field_offset: usize,
    ) -> usize {
        let elem_size = self.resolve_ctype_size(elem_ty);
        let elem_ir_ty = IrType::from_ctype(elem_ty);

        // Collect all Index and Field designators after the first (which is Field("name"))
        // For .a[1][2].b we have designators: [Field("a"), Index(1), Index(2), Field("b")]
        // After stripping Field("a"), remaining: [Index(1), Index(2), Field("b")]
        let remaining = &item.designators[1..];

        // Find first Index designator (outer array index)
        let (first_idx_pos, idx) = remaining.iter().enumerate().find_map(|(i, d)| {
            if let Designator::Index(ref idx_expr) = d {
                self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()).map(|v| (i, v))
            } else {
                None
            }
        }).unwrap_or((0, 0));

        if idx >= arr_size {
            return idx;
        }

        let elem_offset = field_offset + idx * elem_size;

        // Check for further designators after the first Index
        let after_first_idx = &remaining[first_idx_pos + 1..];

        // Collect remaining Field designators for struct drilling
        let remaining_field_desigs: Vec<_> = after_first_idx.iter()
            .filter(|d| matches!(d, Designator::Field(_)))
            .cloned()
            .collect();

        // Collect remaining Index designators for multi-dimensional array drilling
        let remaining_index_desigs: Vec<_> = after_first_idx.iter()
            .filter(|d| matches!(d, Designator::Index(_)))
            .cloned()
            .collect();

        if !remaining_field_desigs.is_empty() {
            // Drill into struct element: .a[1].b = val
            if !remaining_index_desigs.is_empty() {
                // .a[1].b[2] - remaining has both field and index
                // Build designator list with fields and indices from after_first_idx
                let sub_desigs: Vec<_> = after_first_idx.to_vec();
                if let Some(sub_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                    let sub_item = InitializerItem {
                        designators: sub_desigs,
                        init: item.init.clone(),
                    };
                    self.fill_struct_global_bytes(&[sub_item], &sub_layout, bytes, elem_offset);
                }
            } else if let Some(sub_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                let sub_item = InitializerItem {
                    designators: remaining_field_desigs,
                    init: item.init.clone(),
                };
                self.fill_struct_global_bytes(&[sub_item], &sub_layout, bytes, elem_offset);
            }
        } else if !remaining_index_desigs.is_empty() {
            // Multi-dimensional array: .a[1][2] = val
            // elem_ty is the inner array type (e.g., float[10] for float a[3][10])
            if let CType::Array(inner_elem_ty, Some(inner_size)) = elem_ty {
                // Recursively handle the inner array with the remaining index designators
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
                    let inner_offset = elem_offset + inner_idx * inner_elem_size;

                    // Check for even more Index designators (3D+ arrays)
                    let further_indices: Vec<_> = remaining_index_desigs[1..].to_vec();
                    if !further_indices.is_empty() {
                        if let CType::Array(_deeper_elem, Some(_deeper_size)) = inner_elem_ty.as_ref() {
                            // Build sub-item with remaining indices as [Field("dummy"), Index(...)]
                            // Actually we can recursively call ourselves
                            let sub_item = InitializerItem {
                                designators: {
                                    let mut d = vec![Designator::Field(Rc::from(""))]; // dummy field
                                    d.extend(further_indices);
                                    d
                                },
                                init: item.init.clone(),
                            };
                            self.fill_nested_designator_array(
                                &sub_item, inner_elem_ty, *inner_size,
                                bytes, elem_offset,
                            );
                        }
                    } else if let Initializer::Expr(ref expr) = item.init {
                        if let Expr::StringLiteral(s, _) = expr {
                            if let CType::Array(deep_inner, Some(deep_size)) = inner_elem_ty.as_ref() {
                                if matches!(deep_inner.as_ref(), CType::Char | CType::UChar) {
                                    Self::write_string_to_bytes(bytes, inner_offset, s, *deep_size);
                                } else {
                                    let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                                    self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                                }
                            } else {
                                let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                                self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                            }
                        } else {
                            let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                            self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                        }
                    }
                }
            }
        } else if let Initializer::Expr(ref expr) = item.init {
            // No further designators - store value at elem_offset
            // Handle string literal initializing a char array element
            // (e.g., .a[1] = "abc" where a is char a[3][10])
            if let Expr::StringLiteral(s, _) = expr {
                if let CType::Array(inner, Some(inner_size)) = elem_ty {
                    if matches!(inner.as_ref(), CType::Char | CType::UChar) {
                        Self::write_string_to_bytes(bytes, elem_offset, s, *inner_size);
                    } else {
                        let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                        self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                    }
                } else if matches!(elem_ty, CType::Char | CType::UChar) {
                    let val = s.chars().next().map(|c| IrConst::I8(c as u8 as i8)).unwrap_or(IrConst::I8(0));
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                } else {
                    let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
            } else {
                let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                // _Bool elements: normalize (any nonzero -> 1) per C11 6.3.1.2
                let val = if *elem_ty == CType::Bool { val.bool_normalize() } else { val };
                self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
            }
        } else if let Initializer::List(ref sub_items) = item.init {
            // Handle list init for array element (e.g., .a[1] = {1,2,3})
            match elem_ty {
                CType::Array(inner_elem_ty, Some(inner_size)) => {
                    let inner_elem_size = self.resolve_ctype_size(inner_elem_ty);
                    let inner_ir_ty = IrType::from_ctype(inner_elem_ty);
                    let inner_is_bool = **inner_elem_ty == CType::Bool;
                    for (si, sub_item) in sub_items.iter().enumerate() {
                        if si >= *inner_size { break; }
                        if let Initializer::Expr(ref expr) = sub_item.init {
                            let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                            let val = if inner_is_bool { val.bool_normalize() } else { val };
                            let inner_offset = elem_offset + si * inner_elem_size;
                            self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                        }
                    }
                }
                CType::Struct(ref key) | CType::Union(ref key) => {
                    if let Some(sub_layout) = self.types.borrow_struct_layouts().get(&**key).cloned() {
                        self.fill_struct_global_bytes(sub_items, &sub_layout, bytes, elem_offset);
                    }
                }
                _ => {}
            }
        }
        idx
    }

    /// Fill a composite (struct/union) field from an initializer.
    /// Returns the number of items consumed.
    pub(super) fn fill_composite_field(
        &self, items: &[InitializerItem], sub_layout: &StructLayout,
        bytes: &mut [u8], field_offset: usize,
    ) -> usize {
        match &items[0].init {
            Initializer::List(sub_items) => {
                self.fill_struct_global_bytes(sub_items, sub_layout, bytes, field_offset);
                1
            }
            Initializer::Expr(expr) => {
                // Unwrap compound literal: (type){ init_list } -> use inner init_list
                if let Expr::CompoundLiteral(_, ref cl_init, _) = expr {
                    if let Initializer::List(sub_items) = cl_init.as_ref() {
                        self.fill_struct_global_bytes(sub_items, sub_layout, bytes, field_offset);
                        return 1;
                    }
                }
                // If the item has a designator (e.g., `.y = 42` where y is a struct),
                // strip it so the scalar is treated as a positional init for the first
                // field of the sub-struct (C11 6.7.9p13).
                if !items[0].designators.is_empty() {
                    let stripped = InitializerItem {
                        designators: vec![],
                        init: items[0].init.clone(),
                    };
                    self.fill_struct_global_bytes(&[stripped], sub_layout, bytes, field_offset);
                    1
                } else {
                    let consumed = self.fill_struct_global_bytes(items, sub_layout, bytes, field_offset);
                    if consumed == 0 { 1 } else { consumed }
                }
            }
        }
    }

    /// Fill a fixed-size array field. Handles string literals, arrays of composites,
    /// multi-dimensional arrays, arrays of scalars with designators, and flat initialization.
    pub(super) fn fill_array_field(
        &self,
        items: &[InitializerItem],
        item_idx: usize,
        elem_ty: &CType,
        arr_size: usize,
        bytes: &mut [u8],
        field_offset: usize,
        array_start_idx: Option<usize>,
    ) -> ArrayFillResult {
        let item = &items[item_idx];
        let elem_size = self.resolve_ctype_size(elem_ty);
        let elem_ir_ty = IrType::from_ctype(elem_ty);

        match &item.init {
            Initializer::List(sub_items) => {
                // Check for brace-wrapped string literal: { "hello" }
                if self.try_fill_string_literal_init(sub_items, elem_ty, arr_size, bytes, field_offset) {
                    return ArrayFillResult { new_item_idx: item_idx + 1, skip_update: true };
                }
                if matches!(elem_ty, CType::Struct(_) | CType::Union(_)) {
                    self.fill_array_of_composites(sub_items, elem_ty, arr_size, elem_size, bytes, field_offset);
                } else if let CType::Array(inner_elem, Some(inner_size)) = elem_ty {
                    // Multi-dimensional array field (e.g., int a[2][3] or struct S a[2][2][2]).
                    // Each sub-item is a brace group for one element of the outer dimension.
                    self.fill_multidim_array_field(
                        sub_items, inner_elem, *inner_size, arr_size, elem_size,
                        bytes, field_offset,
                    );
                } else if elem_ty.is_complex() {
                    // Array of complex elements: each sub-item initializes one complex value
                    self.fill_array_of_complex(sub_items, elem_ty, arr_size, elem_size, bytes, field_offset);
                } else {
                    self.fill_array_of_scalars(sub_items, arr_size, elem_size, elem_ir_ty, bytes, field_offset);
                }
                ArrayFillResult { new_item_idx: item_idx + 1, skip_update: false }
            }
            Initializer::Expr(expr) => {
                // String literal for char array
                if let Expr::StringLiteral(s, _) = expr {
                    if matches!(elem_ty, CType::Char | CType::UChar) {
                        Self::write_string_to_bytes(bytes, field_offset, s, arr_size);
                        return ArrayFillResult { new_item_idx: item_idx + 1, skip_update: true };
                    }
                }
                // Flat init from consecutive items.
                // For multi-dimensional arrays (elem_ty is Array), use leaf element
                // info so that flat scalars fill the entire array correctly.
                let start_ai = array_start_idx.unwrap_or(0);
                let leaf_composite = Self::leaf_composite_type(elem_ty);
                let new_idx = if matches!(elem_ty, CType::Struct(_) | CType::Union(_)) {
                    self.fill_flat_array_of_composites(
                        items, item_idx, elem_ty, arr_size, elem_size, elem_ir_ty,
                        bytes, field_offset, start_ai,
                    )
                } else if let Some(composite_ty) = leaf_composite {
                    // Multi-dimensional array of structs/unions: flat init fills composites
                    let composite_size = self.resolve_ctype_size(composite_ty);
                    let composite_ir_ty = IrType::from_ctype(composite_ty);
                    let total_composites = if composite_size > 0 { (arr_size * elem_size) / composite_size } else { 0 };
                    self.fill_flat_array_of_composites(
                        items, item_idx, composite_ty, total_composites, composite_size, composite_ir_ty,
                        bytes, field_offset, start_ai,
                    )
                } else if matches!(elem_ty, CType::Array(_, _)) {
                    // Multi-dimensional array of scalars: use leaf element size
                    let leaf_size = self.leaf_elem_size(elem_ty);
                    let leaf_ir_ty = Self::leaf_ir_type(elem_ty);
                    let total_scalars = if leaf_size > 0 { (arr_size * elem_size) / leaf_size } else { 0 };
                    self.fill_flat_array_of_scalars(
                        items, item_idx, total_scalars, leaf_size, leaf_ir_ty,
                        bytes, field_offset, start_ai,
                    )
                } else if elem_ty.is_complex() {
                    self.fill_flat_array_of_complex(
                        items, item_idx, elem_ty, arr_size, elem_size,
                        bytes, field_offset, start_ai,
                    )
                } else {
                    self.fill_flat_array_of_scalars(
                        items, item_idx, arr_size, elem_size, elem_ir_ty,
                        bytes, field_offset, start_ai,
                    )
                };
                ArrayFillResult { new_item_idx: new_idx, skip_update: true }
            }
        }
    }

    /// Fill a multi-dimensional array field from a braced initializer list.
    /// Handles arrays like `int a[2][3]`, `struct S a[2][2][2]`, `union U a[3][4]`, etc.
    /// Each sub-item at this level represents one element of the outermost dimension.
    pub(super) fn fill_multidim_array_field(
        &self,
        sub_items: &[InitializerItem],
        inner_elem_ty: &CType,
        inner_arr_size: usize,
        outer_arr_size: usize,
        outer_elem_size: usize,
        bytes: &mut [u8],
        field_offset: usize,
    ) {
        let mut sub_idx = 0usize;
        let mut ai = 0usize;
        while ai < outer_arr_size && sub_idx < sub_items.len() {
            let elem_offset = field_offset + ai * outer_elem_size;
            match &sub_items[sub_idx].init {
                Initializer::List(inner_items) => {
                    // Brace group for this outer element: recurse into inner array
                    self.fill_array_field_recursive(
                        inner_items, inner_elem_ty, inner_arr_size,
                        bytes, elem_offset,
                    );
                    sub_idx += 1;
                }
                Initializer::Expr(expr) => {
                    // String literal initializing a char array element (e.g., char arr[3][8] = { "hello", "world", "foo" })
                    // Each string literal initializes one inner char array (e.g., char[8]).
                    if matches!(inner_elem_ty, CType::Char | CType::UChar) {
                        if let Expr::StringLiteral(s, _) = expr {
                            Self::write_string_to_bytes(bytes, elem_offset, s, inner_arr_size);
                            sub_idx += 1;
                            ai += 1;
                            continue;
                        }
                    }
                    // Flat init: fill inner array elements from consecutive Expr items.
                    // Determine the leaf type to figure out how to consume items.
                    let leaf_composite = Self::leaf_composite_type(inner_elem_ty);
                    if let Some(composite_ty) = leaf_composite {
                        // The leaf elements are structs/unions: fill them field by field
                        let composite_layout = self.get_composite_layout(composite_ty);
                        let composite_size = self.resolve_ctype_size(composite_ty);
                        let composites_per_outer = if composite_size > 0 { outer_elem_size / composite_size } else { 0 };
                        let mut ci = 0usize;
                        while sub_idx < sub_items.len() && ci < composites_per_outer {
                            let comp_offset = elem_offset + ci * composite_size;
                            let consumed = self.fill_struct_global_bytes(
                                &sub_items[sub_idx..], &composite_layout, bytes, comp_offset,
                            );
                            sub_idx += consumed.max(1);
                            ci += 1;
                        }
                    } else {
                        // Leaf elements are scalars
                        let leaf_size = self.leaf_elem_size(inner_elem_ty);
                        let scalars_per_outer = if leaf_size > 0 { outer_elem_size / leaf_size } else { 1 };
                        let leaf_ir_ty = Self::leaf_ir_type(inner_elem_ty);
                        let mut filled = 0usize;
                        while sub_idx < sub_items.len() && filled < scalars_per_outer {
                            if let Initializer::Expr(e) = &sub_items[sub_idx].init {
                                let val = self.eval_const_expr(e).unwrap_or(IrConst::I64(0));
                                self.write_const_to_bytes(bytes, elem_offset + filled * leaf_size, &val, leaf_ir_ty);
                                sub_idx += 1;
                                filled += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
            ai += 1;
        }
    }

    /// Recursively fill a (possibly multi-dimensional) array from a braced initializer list.
    /// Called by fill_multidim_array_field to handle inner dimensions.
    fn fill_array_field_recursive(
        &self,
        items: &[InitializerItem],
        elem_ty: &CType,
        arr_size: usize,
        bytes: &mut [u8],
        field_offset: usize,
    ) {
        let elem_size = self.resolve_ctype_size(elem_ty);

        if let CType::Array(inner_elem, Some(inner_size)) = elem_ty {
            // Still multi-dimensional: recurse
            self.fill_multidim_array_field(
                items, inner_elem, *inner_size, arr_size, elem_size,
                bytes, field_offset,
            );
        } else if matches!(elem_ty, CType::Struct(_) | CType::Union(_)) {
            self.fill_array_of_composites(items, elem_ty, arr_size, elem_size, bytes, field_offset);
        } else if matches!(elem_ty, CType::Char | CType::UChar) {
            // Char array: check for string literal initializer (e.g., char[8] init from "hello")
            if items.len() == 1 {
                if let Initializer::Expr(Expr::StringLiteral(s, _)) = &items[0].init {
                    Self::write_string_to_bytes(bytes, field_offset, s, arr_size);
                    return;
                }
            }
            let elem_ir_ty = IrType::from_ctype(elem_ty);
            self.fill_array_of_scalars(items, arr_size, elem_size, elem_ir_ty, bytes, field_offset);
        } else {
            let elem_ir_ty = IrType::from_ctype(elem_ty);
            self.fill_array_of_scalars(items, arr_size, elem_size, elem_ir_ty, bytes, field_offset);
        }
    }

    /// Get the leaf (innermost non-array) element size for a possibly nested array type.
    fn leaf_elem_size(&self, ty: &CType) -> usize {
        match ty {
            CType::Array(inner, _) => self.leaf_elem_size(inner),
            _ => self.resolve_ctype_size(ty),
        }
    }

    /// Get the leaf (innermost non-array) IR type for a possibly nested array type.
    fn leaf_ir_type(ty: &CType) -> IrType {
        match ty {
            CType::Array(inner, _) => Self::leaf_ir_type(inner),
            _ => IrType::from_ctype(ty),
        }
    }

    /// Get the leaf (innermost non-array) composite type, if the leaf is a struct/union.
    /// Returns None if the leaf is a scalar type.
    fn leaf_composite_type(ty: &CType) -> Option<&CType> {
        match ty {
            CType::Array(inner, _) => Self::leaf_composite_type(inner),
            CType::Struct(_) | CType::Union(_) => Some(ty),
            _ => None,
        }
    }

    /// Fill a flexible array member (FAM) field.
    pub(super) fn fill_fam_field(
        &self,
        items: &[InitializerItem],
        item_idx: usize,
        elem_ty: &CType,
        bytes: &mut [u8],
        field_offset: usize,
    ) -> ArrayFillResult {
        let elem_size = self.resolve_ctype_size(elem_ty);
        let elem_ir_ty = IrType::from_ctype(elem_ty);

        match &items[item_idx].init {
            Initializer::List(sub_items) => {
                // Check if FAM element type is a struct/union - if so, fill each element
                // using struct-aware byte filling instead of treating them as scalars.
                let elem_layout = self.get_struct_layout_for_ctype(elem_ty);
                for (ai, sub_item) in sub_items.iter().enumerate() {
                    let elem_offset = field_offset + ai * elem_size;
                    if elem_offset + elem_size > bytes.len() { break; }
                    if let Some(ref layout) = elem_layout {
                        // Struct/union element: recursively fill
                        match &sub_item.init {
                            Initializer::List(nested_items) => {
                                self.fill_struct_global_bytes(nested_items, layout, bytes, elem_offset);
                            }
                            Initializer::Expr(_) => {
                                let val = self.eval_init_scalar(&sub_item.init);
                                self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                            }
                        }
                    } else {
                        let val = self.eval_init_scalar(&sub_item.init);
                        self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                    }
                }
                ArrayFillResult { new_item_idx: item_idx + 1, skip_update: false }
            }
            Initializer::Expr(expr) => {
                // String literal initializing a char FAM: write bytes directly
                if let Expr::StringLiteral(s, _) = expr {
                    if matches!(elem_ty, CType::Char | CType::UChar) {
                        let max_len = bytes.len().saturating_sub(field_offset);
                        Self::write_string_to_bytes(bytes, field_offset, s, max_len);
                        return ArrayFillResult { new_item_idx: item_idx + 1, skip_update: true };
                    }
                }
                let mut ai = 0usize;
                let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                let elem_offset = field_offset + ai * elem_size;
                if elem_offset + elem_size <= bytes.len() {
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
                ai += 1;
                let mut new_idx = item_idx + 1;
                while new_idx < items.len() {
                    let next_item = &items[new_idx];
                    if !next_item.designators.is_empty() { break; }
                    if let Initializer::Expr(e) = &next_item.init {
                        let val = self.eval_const_expr(e).unwrap_or(IrConst::I64(0));
                        let elem_offset = field_offset + ai * elem_size;
                        if elem_offset + elem_size <= bytes.len() {
                            self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                        }
                        ai += 1;
                        new_idx += 1;
                    } else {
                        break;
                    }
                }
                ArrayFillResult { new_item_idx: new_idx, skip_update: true }
            }
        }
    }

    /// Try to interpret a brace-wrapped init list as a string literal for a char array.
    /// Returns true if handled.
    pub(super) fn try_fill_string_literal_init(
        &self, sub_items: &[InitializerItem], elem_ty: &CType, arr_size: usize,
        bytes: &mut [u8], field_offset: usize,
    ) -> bool {
        if sub_items.len() == 1 && sub_items[0].designators.is_empty() {
            if let Initializer::Expr(Expr::StringLiteral(s, _)) = &sub_items[0].init {
                if matches!(elem_ty, CType::Char | CType::UChar) {
                    Self::write_string_to_bytes(bytes, field_offset, s, arr_size);
                    return true;
                }
            }
        }
        false
    }

    /// Fill an array of struct/union elements from a braced sub-item list.
    pub(super) fn fill_array_of_composites(
        &self, sub_items: &[InitializerItem], elem_ty: &CType,
        arr_size: usize, elem_size: usize, bytes: &mut [u8], field_offset: usize,
    ) {
        let sub_layout = self.get_composite_layout(elem_ty);
        let mut sub_idx = 0usize;
        let mut ai = 0usize;
        while ai < arr_size && sub_idx < sub_items.len() {
            // Handle designated array index initializers (e.g., [3] = { ... })
            if let Some(Designator::Index(ref idx_expr)) = sub_items[sub_idx].designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    ai = idx;
                }
            }
            if ai >= arr_size { break; }
            let elem_offset = field_offset + ai * elem_size;
            match &sub_items[sub_idx].init {
                Initializer::List(inner_items) => {
                    self.fill_struct_global_bytes(inner_items, &sub_layout, bytes, elem_offset);
                    sub_idx += 1;
                }
                Initializer::Expr(_) => {
                    let consumed = self.fill_struct_global_bytes(&sub_items[sub_idx..], &sub_layout, bytes, elem_offset);
                    sub_idx += consumed;
                }
            }
            ai += 1;
        }
    }

    /// Fill an array of scalar elements from a braced sub-item list (with designator support).
    pub(super) fn fill_array_of_scalars(
        &self, sub_items: &[InitializerItem], arr_size: usize,
        elem_size: usize, elem_ir_ty: IrType, bytes: &mut [u8], field_offset: usize,
    ) {
        let mut ai = 0usize;
        for sub_item in sub_items {
            if let Some(Designator::Index(ref idx_expr)) = sub_item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    ai = idx;
                }
            }
            if ai >= arr_size { break; }
            let elem_offset = field_offset + ai * elem_size;
            let val = self.eval_init_scalar(&sub_item.init);
            self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
            ai += 1;
        }
    }

    /// Fill an array of complex elements from a braced sub-item list.
    /// Each sub-item initializes one complex element using write_complex_field_to_bytes.
    pub(super) fn fill_array_of_complex(
        &self, sub_items: &[InitializerItem], complex_ctype: &CType,
        arr_size: usize, elem_size: usize, bytes: &mut [u8], field_offset: usize,
    ) {
        let mut ai = 0usize;
        for sub_item in sub_items {
            if let Some(Designator::Index(ref idx_expr)) = sub_item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    ai = idx;
                }
            }
            if ai >= arr_size { break; }
            let elem_offset = field_offset + ai * elem_size;
            self.write_complex_field_to_bytes(bytes, elem_offset, complex_ctype, &sub_item.init);
            ai += 1;
        }
    }

    /// Fill flat init of an array of complex elements from consecutive items.
    /// Each item initializes one complex element. Returns the new item_idx.
    pub(super) fn fill_flat_array_of_complex(
        &self, items: &[InitializerItem], item_idx: usize,
        complex_ctype: &CType, arr_size: usize, elem_size: usize,
        bytes: &mut [u8], field_offset: usize, start_ai: usize,
    ) -> usize {
        let mut consumed = 0usize;
        let mut ai = start_ai;
        while ai < arr_size && (item_idx + consumed) < items.len() {
            let cur_item = &items[item_idx + consumed];
            if !cur_item.designators.is_empty() && consumed > 0 { break; }
            let elem_offset = field_offset + ai * elem_size;
            self.write_complex_field_to_bytes(bytes, elem_offset, complex_ctype, &cur_item.init);
            consumed += 1;
            ai += 1;
        }
        item_idx + consumed.max(1)
    }

    /// Fill flat init of an array of composites from consecutive items.
    /// Returns the new item_idx.
    pub(super) fn fill_flat_array_of_composites(
        &self, items: &[InitializerItem], mut item_idx: usize,
        elem_ty: &CType, arr_size: usize, elem_size: usize, elem_ir_ty: IrType,
        bytes: &mut [u8], field_offset: usize, start_ai: usize,
    ) -> usize {
        let sub_layout = self.get_composite_layout(elem_ty);
        if matches!(elem_ty, CType::Struct(_)) {
            for ai in start_ai..arr_size {
                if item_idx >= items.len() { break; }
                let elem_offset = field_offset + ai * elem_size;
                let consumed = self.fill_struct_global_bytes(&items[item_idx..], &sub_layout, bytes, elem_offset);
                item_idx += consumed;
            }
        } else {
            // Union: take one item per element
            for ai in start_ai..arr_size {
                if item_idx >= items.len() { break; }
                let elem_offset = field_offset + ai * elem_size;
                if let Initializer::Expr(e) = &items[item_idx].init {
                    let val = self.eval_const_expr(e).unwrap_or(IrConst::I64(0));
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
                item_idx += 1;
            }
        }
        item_idx
    }

    /// Fill flat init of an array of scalars from consecutive items.
    /// Returns the new item_idx.
    pub(super) fn fill_flat_array_of_scalars(
        &self, items: &[InitializerItem], item_idx: usize,
        arr_size: usize, elem_size: usize, elem_ir_ty: IrType,
        bytes: &mut [u8], field_offset: usize, start_ai: usize,
    ) -> usize {
        let mut consumed = 0usize;
        let mut ai = start_ai;
        while ai < arr_size && (item_idx + consumed) < items.len() {
            let cur_item = &items[item_idx + consumed];
            if !cur_item.designators.is_empty() && consumed > 0 { break; }
            if let Initializer::Expr(e) = &cur_item.init {
                let val = self.eval_const_expr(e).unwrap_or(IrConst::I64(0));
                let elem_offset = field_offset + ai * elem_size;
                self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                consumed += 1;
                ai += 1;
            } else {
                break;
            }
        }
        item_idx + consumed.max(1)
    }

    /// Fill a multi-dimensional array of structs/unions into a byte buffer.
    ///
    /// Recursively processes brace-enclosed initializer lists to handle arrays with
    /// 3+ dimensions correctly. Each brace nesting level corresponds to one array
    /// dimension, and `array_dim_strides` tells us how many bytes each level spans.
    ///
    /// For `struct S arr[2][2][2]` with struct_size=8, strides=[32, 16, 8]:
    /// - Top-level items fill stride[0]=32 bytes each (a [2][2] sub-array = 4 structs)
    /// - Second-level brace groups fill stride[1]=16 bytes each (a [2] sub-array = 2 structs)
    /// - Innermost brace groups fill stride[2]=8 bytes each (1 struct)
    pub(super) fn fill_multidim_struct_array_bytes(
        &self,
        items: &[InitializerItem],
        layout: &StructLayout,
        struct_size: usize,
        array_dim_strides: &[usize],
        bytes: &mut [u8],
        base_offset: usize,
        region_size: usize,
    ) {
        if struct_size == 0 { return; }

        // Determine the stride for elements at this brace level.
        // If we have strides info, use it; otherwise treat each item as one struct.
        let (this_stride, remaining_strides) = if array_dim_strides.len() > 1 {
            (array_dim_strides[0], &array_dim_strides[1..])
        } else if array_dim_strides.len() == 1 {
            (array_dim_strides[0], &array_dim_strides[0..0])
        } else {
            (struct_size, &array_dim_strides[0..0])
        };

        let num_elems = if this_stride > 0 { region_size / this_stride } else { 0 };
        let mut current_idx = 0usize;
        let mut item_idx = 0usize;

        while item_idx < items.len() {
            let item = &items[item_idx];

            // Check for multi-dimensional index designators: [i][j]...
            // Collect all Index designators to compute the correct element index at this level
            let index_designators: Vec<usize> = item.designators.iter().filter_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            }).collect();

            let has_index_designator = !index_designators.is_empty();
            if has_index_designator {
                // For multi-dim designators like [i][j], the first index selects at this level,
                // and remaining indices drill into sub-dimensions.
                current_idx = index_designators[0];

                // If there are extra index designators beyond the first, we need to compute
                // a byte offset within this_stride and recurse or write directly.
                if index_designators.len() > 1 && !remaining_strides.is_empty() {
                    let elem_offset = base_offset + current_idx * this_stride;
                    // Compute sub-offset from remaining designators
                    let mut sub_byte_offset = 0usize;
                    let mut sub_strides = remaining_strides;
                    for &idx in &index_designators[1..] {
                        let sub_stride = if !sub_strides.is_empty() { sub_strides[0] } else { struct_size };
                        sub_byte_offset += idx * sub_stride;
                        if sub_strides.len() > 1 {
                            sub_strides = &sub_strides[1..];
                        } else {
                            sub_strides = &remaining_strides[0..0]; // empty
                        }
                    }

                    // Handle field designator for [i][j].field = val
                    let field_designator_name = item.designators.iter().find_map(|d| {
                        if let Designator::Field(ref name) = d { Some(name.clone()) } else { None }
                    });

                    match &item.init {
                        Initializer::Expr(expr) => {
                            if let Some(ref fname) = field_designator_name {
                                if let Some(val) = self.eval_const_expr(expr) {
                                    if let Some(field) = layout.fields.iter().find(|f| &f.name == fname) {
                                        let field_ir_ty = IrType::from_ctype(&field.ty);
                                        let write_offset = elem_offset + sub_byte_offset + field.offset;
                                        if let (Some(bo), Some(bw)) = (field.bit_offset, field.bit_width) {
                                            self.write_bitfield_to_bytes(bytes, write_offset, &val, field_ir_ty, bo, bw);
                                        } else if write_offset + field_ir_ty.size() <= bytes.len() {
                                            self.write_const_to_bytes(bytes, write_offset, &val, field_ir_ty);
                                        }
                                    }
                                }
                            } else {
                                // Scalar value at the computed sub-position
                                if let Some(_val) = self.eval_const_expr(expr) {
                                    let write_offset = elem_offset + sub_byte_offset;
                                    // Write struct fields sequentially from this position
                                    self.fill_struct_global_bytes(std::slice::from_ref(item), layout, bytes, write_offset);
                                }
                            }
                        }
                        Initializer::List(sub_items) => {
                            let write_offset = elem_offset + sub_byte_offset;
                            if !sub_strides.is_empty() && sub_strides[0] > struct_size {
                                self.fill_multidim_struct_array_bytes(
                                    sub_items, layout, struct_size, sub_strides,
                                    bytes, write_offset, sub_strides[0],
                                );
                            } else {
                                self.fill_struct_global_bytes(sub_items, layout, bytes, write_offset);
                            }
                        }
                    }
                    item_idx += 1;
                    current_idx += 1;
                    continue;
                }
            }

            if current_idx >= num_elems {
                // Only skip items without a designator that would reset the index;
                // designated items explicitly set current_idx above
                if !has_index_designator { break; }
                item_idx += 1;
                continue;
            }

            // Check for field designator: [idx].field = val
            let field_designator_name = item.designators.iter().find_map(|d| {
                if let Designator::Field(ref name) = d {
                    Some(name.clone())
                } else {
                    None
                }
            });

            let elem_offset = base_offset + current_idx * this_stride;

            match &item.init {
                Initializer::List(sub_items) => {
                    if this_stride > struct_size && !remaining_strides.is_empty() {
                        // This brace group represents a sub-array (not a single struct).
                        // Recurse with the next dimension's strides.
                        self.fill_multidim_struct_array_bytes(
                            sub_items, layout, struct_size, remaining_strides,
                            bytes, elem_offset, this_stride,
                        );
                    } else {
                        // This brace group represents a single struct initializer.
                        self.fill_struct_global_bytes(sub_items, layout, bytes, elem_offset);
                    }
                    item_idx += 1;
                }
                Initializer::Expr(expr) => {
                    if let Some(ref fname) = field_designator_name {
                        // [idx].field = val: write to specific field
                        if let Some(val) = self.eval_const_expr(expr) {
                            if let Some(field) = layout.fields.iter().find(|f| &f.name == fname) {
                                let field_ir_ty = IrType::from_ctype(&field.ty);
                                if let (Some(bo), Some(bw)) = (field.bit_offset, field.bit_width) {
                                    self.write_bitfield_to_bytes(bytes, elem_offset + field.offset, &val, field_ir_ty, bo, bw);
                                } else {
                                    self.write_const_to_bytes(bytes, elem_offset + field.offset, &val, field_ir_ty);
                                }
                            }
                        }
                        item_idx += 1;
                    } else {
                        // Flat init: consume items for struct fields sequentially.
                        // Fill structs one by one across the entire region.
                        let max_structs = if struct_size > 0 { region_size / struct_size } else { 0 };
                        let flat_struct_base = (elem_offset - base_offset) / struct_size;
                        let mut fi = flat_struct_base;
                        while item_idx < items.len() && fi < max_structs {
                            let byte_off = base_offset + fi * struct_size;
                            if byte_off + struct_size > bytes.len() { break; }
                            let consumed = self.fill_struct_global_bytes(&items[item_idx..], layout, bytes, byte_off);
                            item_idx += consumed.max(1);
                            fi += 1;
                        }
                        // Skip the normal current_idx increment since we consumed everything
                        continue;
                    }
                }
            }
            // Only advance current_idx if no field designator (sequential init)
            if field_designator_name.is_none() {
                current_idx += 1;
            }
        }
    }

    /// Get the StructLayout for a composite (struct or union) CType.
    /// Like `get_struct_layout_for_ctype` but returns a fallback empty layout instead of None.
    /// Panics if called on a non-composite type.
    pub(super) fn get_composite_layout(&self, ty: &CType) -> crate::common::types::RcLayout {
        self.get_struct_layout_for_ctype(ty)
            .unwrap_or_else(|| {
                if matches!(ty, CType::Struct(_) | CType::Union(_)) {
                    if matches!(ty, CType::Union(_)) { StructLayout::empty_union_rc() } else { StructLayout::empty_rc() }
                } else {
                    unreachable!("get_composite_layout called on non-composite type")
                }
            })
    }

    /// Fill a byte buffer from a list of scalar initializer items.
    /// Used by the compound path when serializing non-pointer array/scalar fields to bytes.
    /// Each item is evaluated as a constant expression, coerced to the element type,
    /// and written at the appropriate byte offset.
    pub(super) fn fill_scalar_list_to_bytes(
        &self,
        items: &[InitializerItem],
        elem_ty: &CType,
        max_size: usize,
        bytes: &mut [u8],
    ) {
        let elem_ir_ty = IrType::from_ctype(elem_ty);
        let elem_size = elem_ir_ty.size().max(1);
        let mut current_idx = 0usize;
        for item in items {
            // Respect designated array initializer indices (e.g., [3] = 40)
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    current_idx = idx;
                }
            }
            let byte_offset = current_idx * elem_size;
            if byte_offset >= max_size { current_idx += 1; continue; }
            if let Initializer::Expr(ref e) = item.init {
                if let Some(val) = self.eval_const_expr(e) {
                    let e_ty = self.get_expr_type(e);
                    let val = self.coerce_const_to_type_with_src(val, elem_ir_ty, e_ty);
                    self.write_const_to_bytes(bytes, byte_offset, &val, elem_ir_ty);
                }
            }
            current_idx += 1;
        }
    }

    /// Push a constant value as individual bytes into a compound init element list.
    pub(super) fn push_const_as_bytes(&self, elements: &mut Vec<GlobalInit>, val: &IrConst, size: usize) {
        let mut bytes = Vec::with_capacity(size);
        if self.is_x86() {
            val.push_le_bytes_x86(&mut bytes, size);
        } else if self.is_riscv() || self.is_arm() {
            val.push_le_bytes_riscv(&mut bytes, size);
        } else {
            val.push_le_bytes(&mut bytes, size);
        }
        for b in bytes {
            elements.push(GlobalInit::Scalar(IrConst::I8(b as i8)));
        }
    }

    /// Write an IrConst value to a byte buffer at the given offset using the field's IR type.
    pub(super) fn write_const_to_bytes(&self, bytes: &mut [u8], offset: usize, val: &IrConst, ty: IrType) {
        let coerced = val.coerce_to(ty);
        let size = ty.size();
        let mut le_buf = Vec::with_capacity(size);
        if self.is_x86() {
            coerced.push_le_bytes_x86(&mut le_buf, size);
        } else if self.is_riscv() || self.is_arm() {
            coerced.push_le_bytes_riscv(&mut le_buf, size);
        } else {
            coerced.push_le_bytes(&mut le_buf, size);
        }
        for (i, &b) in le_buf.iter().enumerate() {
            if offset + i < bytes.len() {
                bytes[offset + i] = b;
            }
        }
    }

    /// Write a bitfield value into a byte buffer at the given offset.
    /// Uses read-modify-write to pack the value at the correct bit position.
    pub(super) fn write_bitfield_to_bytes(&self, bytes: &mut [u8], offset: usize, val: &IrConst, ty: IrType, bit_offset: u32, bit_width: u32) {
        let int_val = val.to_u64().unwrap_or(0);

        let size = ty.size();
        let mask = if bit_width >= 64 { u64::MAX } else { (1u64 << bit_width) - 1 };
        let field_val = (int_val & mask) << bit_offset;
        let clear_mask = !(mask << bit_offset);

        // Read current storage unit value (little-endian)
        let mut current = 0u64;
        for i in 0..size {
            if offset + i < bytes.len() {
                current |= (bytes[offset + i] as u64) << (i * 8);
            }
        }

        // Modify: clear field bits and OR in new value
        let new_val = (current & clear_mask) | field_val;

        // Write back (little-endian)
        let le = new_val.to_le_bytes();
        for i in 0..size {
            if offset + i < bytes.len() {
                bytes[offset + i] = le[i];
            }
        }
    }

    /// Write a complex field ({real, imag} pair) to a byte buffer.
    /// Evaluates the initializer as a complex constant expression and writes
    /// both components at the correct offsets within the field.
    pub(super) fn write_complex_field_to_bytes(
        &self,
        bytes: &mut [u8],
        field_offset: usize,
        complex_ctype: &CType,
        init: &Initializer,
    ) {
        let comp_size = Self::complex_component_size(complex_ctype);

        // Try to extract (real, imag) from the initializer
        let (real, imag) = match init {
            Initializer::Expr(expr) => {
                self.eval_complex_const_public(expr).unwrap_or((0.0, 0.0))
            }
            Initializer::List(items) => {
                // Handle "braces around scalar" for _Complex (C11 6.7.9):
                // If the list has a single item that is itself a List, unwrap the extra braces.
                if items.len() == 1 && items[0].designators.is_empty() {
                    if let Initializer::List(_) = &items[0].init {
                        return self.write_complex_field_to_bytes(bytes, field_offset, complex_ctype, &items[0].init);
                    }
                }
                // If the list has a single expression that is complex-typed,
                // evaluate it as a whole complex constant.
                if items.len() == 1 && items[0].designators.is_empty() {
                    if let Initializer::Expr(e) = &items[0].init {
                        if let Some(pair) = self.eval_complex_const_public(e) {
                            pair
                        } else {
                            // Fall through to scalar {real} treatment below
                            let real_val = self.eval_const_expr(e).and_then(|c| match c {
                                IrConst::F64(v) => Some(v),
                                IrConst::F32(v) => Some(v as f64),
                                IrConst::I64(v) => Some(v as f64),
                                IrConst::I32(v) => Some(v as f64),
                                IrConst::LongDouble(v, _) => Some(v),
                                _ => None,
                            }).unwrap_or(0.0);
                            (real_val, 0.0)
                        }
                    } else {
                        (0.0, 0.0)
                    }
                } else {
                    // {real, imag} or {real} (imag defaults to 0)
                    let real_val = items.first().and_then(|item| {
                        if let Initializer::Expr(e) = &item.init {
                            self.eval_const_expr(e).and_then(|c| match c {
                                IrConst::F64(v) => Some(v),
                                IrConst::F32(v) => Some(v as f64),
                                IrConst::I64(v) => Some(v as f64),
                                IrConst::I32(v) => Some(v as f64),
                                IrConst::LongDouble(v, _) => Some(v),
                                _ => None,
                            })
                        } else {
                            None
                        }
                    }).unwrap_or(0.0);
                    let imag_val = items.get(1).and_then(|item| {
                        if let Initializer::Expr(e) = &item.init {
                            self.eval_const_expr(e).and_then(|c| match c {
                                IrConst::F64(v) => Some(v),
                                IrConst::F32(v) => Some(v as f64),
                                IrConst::I64(v) => Some(v as f64),
                                IrConst::I32(v) => Some(v as f64),
                                IrConst::LongDouble(v, _) => Some(v),
                                _ => None,
                            })
                        } else {
                            None
                        }
                    }).unwrap_or(0.0);
                    (real_val, imag_val)
                }
            }
        };

        // Write real part at field_offset
        match complex_ctype {
            CType::ComplexFloat => {
                let real_const = IrConst::F32(real as f32);
                let imag_const = IrConst::F32(imag as f32);
                self.write_const_to_bytes(bytes, field_offset, &real_const, IrType::F32);
                self.write_const_to_bytes(bytes, field_offset + comp_size, &imag_const, IrType::F32);
            }
            CType::ComplexLongDouble => {
                let real_const = IrConst::long_double(real);
                let imag_const = IrConst::long_double(imag);
                self.write_const_to_bytes(bytes, field_offset, &real_const, IrType::F128);
                self.write_const_to_bytes(bytes, field_offset + comp_size, &imag_const, IrType::F128);
            }
            _ => {
                // ComplexDouble
                let real_const = IrConst::F64(real);
                let imag_const = IrConst::F64(imag);
                self.write_const_to_bytes(bytes, field_offset, &real_const, IrType::F64);
                self.write_const_to_bytes(bytes, field_offset + comp_size, &imag_const, IrType::F64);
            }
        }
    }
}
