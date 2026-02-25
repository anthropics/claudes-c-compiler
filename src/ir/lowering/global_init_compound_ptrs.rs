//! Bytes-and-pointers hybrid global initialization path.
//!
//! This module handles struct arrays and composite fields where some fields
//! are pointers (requiring relocations) while others are plain byte data.
//! The approach uses a byte buffer for scalar data and a separate `ptr_ranges`
//! list for pointer relocations, merged into a Compound init at the end.
//!
//! Key entry points:
//! - `lower_struct_array_with_ptrs_multidim` — multi-dimensional struct arrays
//! - `lower_struct_array_with_ptrs` — single-dimension struct arrays
//! - `fill_composite_or_array_with_ptrs` — nested composite/array dispatch

use crate::frontend::parser::ast::{
    Designator,
    Expr,
    Initializer,
    InitializerItem,
};
use crate::ir::reexports::{GlobalInit};
use crate::common::types::{AddressSpace, IrType, CType, InitFieldResolution};
use super::lower::Lowerer;
use super::global_init_helpers as h;

impl Lowerer {
    /// Lower a (possibly multi-dimensional) array of structs where some fields are pointers.
    /// Handles multi-dimensional arrays by recursing through dimension strides.
    pub(super) fn lower_struct_array_with_ptrs_multidim(
        &mut self,
        items: &[InitializerItem],
        layout: &crate::common::types::StructLayout,
        total_size: usize,
        array_dim_strides: &[usize],
    ) -> GlobalInit {
        let struct_size = layout.size;
        let mut bytes = vec![0u8; total_size];
        let mut ptr_ranges: Vec<(usize, GlobalInit)> = Vec::new();

        self.fill_multidim_struct_array_with_ptrs(
            items, layout, struct_size, array_dim_strides,
            &mut bytes, &mut ptr_ranges, 0, total_size,
        );

        Self::build_compound_from_bytes_and_ptrs(bytes, ptr_ranges, total_size)
    }

    /// Recursively fill a multi-dimensional struct array into byte buffer + ptr_ranges.
    /// Similar to fill_multidim_struct_array_bytes but with pointer relocation tracking.
    fn fill_multidim_struct_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        layout: &crate::common::types::StructLayout,
        struct_size: usize,
        array_dim_strides: &[usize],
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
        base_offset: usize,
        region_size: usize,
    ) {
        if struct_size == 0 { return; }

        let (this_stride, remaining_strides) = if array_dim_strides.len() > 1 {
            (array_dim_strides[0], &array_dim_strides[1..])
        } else if array_dim_strides.len() == 1 {
            (array_dim_strides[0], &array_dim_strides[0..0])
        } else {
            (struct_size, &array_dim_strides[0..0])
        };

        let num_elems = if this_stride > 0 { region_size / this_stride } else { 0 };

        if h::has_array_field_designators(items) && this_stride == struct_size {
            self.fill_array_field_designator_items(
                items, layout, struct_size, num_elems, base_offset, bytes, ptr_ranges,
            );
        } else {
            // Sequential items: each item maps to one element at this_stride
            let mut current_idx = 0usize;
            for item in items {
                if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                    if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                        current_idx = idx;
                    }
                }
                if current_idx >= num_elems { current_idx += 1; continue; }

                let elem_offset = base_offset + current_idx * this_stride;

                match &item.init {
                    Initializer::List(sub_items) => {
                        if this_stride > struct_size && !remaining_strides.is_empty() {
                            // Sub-array dimension: recurse
                            self.fill_multidim_struct_array_with_ptrs(
                                sub_items, layout, struct_size, remaining_strides,
                                bytes, ptr_ranges, elem_offset, this_stride,
                            );
                        } else {
                            // Single struct element: fill its fields
                            self.fill_nested_struct_with_ptrs(
                                sub_items, layout, elem_offset, bytes, ptr_ranges,
                            );
                        }
                    }
                    Initializer::Expr(expr) => {
                        // Single expression for first field of struct element
                        if !layout.fields.is_empty() {
                            let field = &layout.fields[0];
                            let field_offset = elem_offset + field.offset;
                            self.write_expr_to_bytes_or_ptrs(
                                expr, &field.ty, field_offset,
                                field.bit_offset, field.bit_width,
                                bytes, ptr_ranges,
                            );
                        }
                    }
                }
                current_idx += 1;
            }
        }
    }

    /// Lower an array of structs where some fields are pointers.
    /// Uses byte-level serialization but with Compound for address elements.
    /// Each struct element is emitted as a mix of byte constants and pointer-sized addresses.
    pub(super) fn lower_struct_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        layout: &crate::common::types::StructLayout,
        num_elems: usize,
    ) -> GlobalInit {
        // We emit the entire array as a sequence of Compound elements.
        // Each element is either:
        //   - Scalar(I8) for byte-level constant data
        //   - GlobalAddr for pointer fields
        let struct_size = layout.size;
        let total_size = num_elems * struct_size;

        // Initialize with zero bytes and track pointer relocation ranges
        let mut bytes = vec![0u8; total_size];
        let mut ptr_ranges: Vec<(usize, GlobalInit)> = Vec::new();

        if h::has_array_field_designators(items) {
            // Handle [N].field = value pattern (e.g., postgres mcxt_methods[]).
            self.fill_array_field_designator_items(
                items, layout, struct_size, num_elems, 0, &mut bytes, &mut ptr_ranges,
            );
        } else {
            // Original path: items correspond 1-to-1 to array elements (no [N].field designators).
            // Each item is either a braced list for one struct element or a single expression.
            self.fill_struct_array_sequential(
                items, layout, num_elems,
                &mut bytes, &mut ptr_ranges,
            );
        }

        Self::build_compound_from_bytes_and_ptrs(bytes, ptr_ranges, total_size)
    }

    /// Fill struct fields from an initializer item list into the byte buffer and ptr_ranges.
    /// Used both for braced init lists `{ field1, field2, ... }` and for unwrapped compound
    /// literals `((struct S) { field1, field2, ... })` in struct array initializers.
    ///
    /// Handles brace elision for array fields: when an expression item (not a braced list)
    /// targets an array field, consecutive items are consumed to fill the array elements
    /// (C11 6.7.9p17-21).
    fn fill_struct_fields_from_items(
        &mut self,
        sub_items: &[InitializerItem],
        layout: &crate::common::types::StructLayout,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let mut current_field_idx = 0usize;
        let mut item_idx = 0usize;
        while item_idx < sub_items.len() {
            let sub_item = &sub_items[item_idx];
            let desig_name = h::first_field_designator(sub_item);
            let resolution = layout.resolve_init_field(desig_name, current_field_idx, &*self.types.borrow_struct_layouts());
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Designator targets a field inside an anonymous struct/union member.
                    // Recursively fill the anonymous member's sub-layout.
                    let extra_desigs = if sub_item.designators.len() > 1 { &sub_item.designators[1..] } else { &[] };
                    let anon_res = h::resolve_anonymous_member(layout, *anon_field_idx, inner_name, &sub_item.init, extra_desigs, &self.types.borrow_struct_layouts());
                    if let Some(res) = anon_res {
                        let anon_offset = base_offset + res.anon_offset;
                        self.fill_struct_fields_from_items(
                            &[res.sub_item], &res.sub_layout, anon_offset, bytes, ptr_ranges,
                        );
                    }
                    current_field_idx = *anon_field_idx + 1;
                    item_idx += 1;
                    // For unions, only initialize the first (or designated) member
                    if layout.is_union && desig_name.is_none() { break; }
                    continue;
                }
                None => break,
            };
            let field = &layout.fields[field_idx];
            let field_offset = base_offset + field.offset;

            if h::has_nested_field_designator(sub_item) {
                self.fill_nested_designator_with_ptrs(
                    sub_item, &field.ty, field_offset,
                    bytes, ptr_ranges,
                );
                current_field_idx = field_idx + 1;
                item_idx += 1;
                if layout.is_union && desig_name.is_none() { break; }
                continue;
            }

            // Brace elision for array fields: when an expression (not a braced list) targets
            // an array field, consume consecutive items to fill the array elements.
            if matches!(&sub_item.init, Initializer::Expr(_)) {
                if let CType::Array(ref elem_ty, Some(arr_size)) = field.ty {
                    // Don't apply flat consumption for string literals initializing char arrays
                    let is_char_array_str = matches!(elem_ty.as_ref(), CType::Char | CType::UChar)
                        && matches!(&sub_item.init, Initializer::Expr(Expr::StringLiteral(..)));
                    if !is_char_array_str {
                        let consumed = self.fill_flat_array_field_with_ptrs(
                            &sub_items[item_idx..], elem_ty, arr_size,
                            field_offset, bytes, ptr_ranges,
                        );
                        item_idx += consumed;
                        current_field_idx = field_idx + 1;
                        if layout.is_union && desig_name.is_none() { break; }
                        continue;
                    }
                }
            }

            // Brace elision for struct/union fields: when a scalar expression (not a
            // braced list) targets a struct/union field, pass the remaining items through
            // so multiple items can fill the sub-struct's fields (C11 6.7.9p13-17).
            if matches!(&sub_item.init, Initializer::Expr(_)) {
                let sub_layout_info = match field.ty {
                    CType::Struct(ref key) | CType::Union(ref key) => {
                        let layouts = self.types.borrow_struct_layouts();
                        layouts.get(&**key).map(|l| {
                            let cloned = l.clone();
                            let has_ptrs = cloned.has_pointer_fields(&*layouts);
                            (cloned, has_ptrs)
                        })
                    }
                    _ => None,
                };
                if let Some((sub_layout, has_ptrs)) = sub_layout_info {
                    if has_ptrs {
                        let consumed = self.fill_nested_struct_brace_elided(
                            &sub_items[item_idx..], &sub_layout, field_offset, bytes, ptr_ranges,
                        );
                        item_idx += consumed;
                    } else {
                        let consumed = self.fill_composite_field(
                            &sub_items[item_idx..], &sub_layout, bytes, field_offset,
                        );
                        item_idx += consumed;
                    }
                    current_field_idx = field_idx + 1;
                    if layout.is_union && desig_name.is_none() { break; }
                    continue;
                }
            }

            self.emit_struct_field_init_compound(
                sub_item, field, field_offset,
                bytes, ptr_ranges,
            );
            current_field_idx = field_idx + 1;
            item_idx += 1;
            if layout.is_union && desig_name.is_none() { break; }
        }
    }

    /// Emit a single struct field initialization into the byte buffer and ptr_ranges.
    /// Handles pointer fields, pointer array fields, bitfields, nested structs, and scalars.
    fn emit_struct_field_init_compound(
        &mut self,
        item: &InitializerItem,
        field: &crate::common::types::StructFieldLayout,
        field_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        // Only treat as a flat pointer array if elements are direct pointers/functions,
        // NOT if elements are structs that happen to contain pointer fields.
        let is_ptr_array = matches!(field.ty, CType::Array(ref elem_ty, _)
            if matches!(elem_ty.as_ref(), CType::Pointer(_, _) | CType::Function(_)));

        if let Initializer::Expr(ref expr) = item.init {
            // For pointer arrays with a single expr, the first element is the addr
            let effective_ty = if is_ptr_array {
                // Treat as a pointer write to the first element
                &CType::Pointer(Box::new(CType::Void), AddressSpace::Default)
            } else {
                &field.ty
            };
            self.write_expr_to_bytes_or_ptrs(
                expr, effective_ty, field_offset,
                field.bit_offset, field.bit_width,
                bytes, ptr_ranges,
            );
        } else if let Initializer::List(ref inner_items) = item.init {
            if is_ptr_array {
                let arr_size = match &field.ty {
                    CType::Array(_, Some(s)) => *s,
                    _ => inner_items.len(),
                };
                let ptr_ty = CType::Pointer(Box::new(CType::Void), AddressSpace::Default);
                let ptr_size = crate::common::types::target_ptr_size();
                // Use a sequential index that respects designated initializers.
                // Items may have [idx] designators that jump forward or backward.
                let mut ai = 0usize;
                for inner_item in inner_items.iter() {
                    // Check for index designator
                    if let Some(crate::frontend::parser::ast::Designator::Index(ref idx_expr)) = inner_item.designators.first() {
                        if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                            ai = idx;
                        }
                    }
                    if ai >= arr_size { ai += 1; continue; }
                    let elem_offset = field_offset + ai * ptr_size;
                    if let Initializer::Expr(ref expr) = inner_item.init {
                        self.write_expr_to_bytes_or_ptrs(
                            expr, &ptr_ty, elem_offset, None, None, bytes, ptr_ranges,
                        );
                    }
                    ai += 1;
                }
            } else {
                self.fill_composite_or_array_with_ptrs(
                    inner_items, &field.ty, field_offset, bytes, ptr_ranges,
                );
            }
        }
    }

    /// Fill a struct array where items map to array elements.
    /// Handles both sequential init and `[N] = {...}` designated index patterns.
    /// Each item is either a braced list `{ .field = val, ... }` or a single expression.
    fn fill_struct_array_sequential(
        &mut self,
        items: &[InitializerItem],
        layout: &crate::common::types::StructLayout,
        num_elems: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let struct_size = layout.size;
        // Track the current sequential index for items without designators
        let mut current_idx = 0usize;
        let mut item_idx = 0usize;
        while item_idx < items.len() {
            let item = &items[item_idx];
            // Check if this item has an [N] array index designator
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr_for_designator(idx_expr) {
                    current_idx = idx;
                }
            }
            let elem_idx = current_idx;
            if elem_idx >= num_elems {
                current_idx += 1;
                item_idx += 1;
                continue;
            }
            let base_offset = elem_idx * struct_size;

            match &item.init {
                Initializer::List(sub_items) => {
                    self.fill_struct_fields_from_items(
                        sub_items, layout, base_offset, bytes, ptr_ranges,
                    );
                    item_idx += 1;
                }
                Initializer::Expr(ref expr) => {
                    // Compound literal, e.g., ((struct Wrap) {inc_global}):
                    // unwrap it and process the inner initializer list as struct fields.
                    if let Expr::CompoundLiteral(_, ref inner_init, _) = expr {
                        if let Initializer::List(ref sub_items) = **inner_init {
                            self.fill_struct_fields_from_items(
                                sub_items, layout, base_offset, bytes, ptr_ranges,
                            );
                        } else if let Initializer::Expr(ref inner_expr) = **inner_init {
                            // Scalar compound literal (e.g., ((type){expr})):
                            // treat the inner expression as the value for the first field
                            if !layout.fields.is_empty() {
                                let field = &layout.fields[0];
                                let field_offset = base_offset + field.offset;
                                let synth_item = InitializerItem {
                                    designators: Vec::new(),
                                    init: Initializer::Expr(inner_expr.clone()),
                                };
                                self.emit_struct_field_init_compound(
                                    &synth_item, field, field_offset,
                                    bytes, ptr_ranges,
                                );
                            }
                        }
                        item_idx += 1;
                    } else {
                        // Flat initialization: consume items field-by-field for this struct.
                        // Each item corresponds to one field, not one struct element.
                        let mut current_field_idx = 0usize;
                        while item_idx < items.len() && current_field_idx < layout.fields.len() {
                            let sub_item = &items[item_idx];
                            // If this item has an array index designator, it starts a new element
                            if !sub_item.designators.is_empty() && item_idx != 0 {
                                break;
                            }
                            let field = &layout.fields[current_field_idx];
                            let field_offset = base_offset + field.offset;

                            self.emit_struct_field_init_compound(
                                sub_item, field, field_offset,
                                bytes, ptr_ranges,
                            );
                            current_field_idx += 1;
                            item_idx += 1;
                            // For unions, only initialize the first member
                            if layout.is_union { break; }
                        }
                    }
                }
            }
            current_idx += 1;
        }
    }

    /// Process `[N].field = value` designated initializer items for a struct array.
    /// This is the consolidated loop that handles the `[Index(N), Field("name")]`
    /// designator pattern used by both `fill_multidim_struct_array_with_ptrs` and
    /// `lower_struct_array_with_ptrs`. Each item targets a specific array element and
    /// field within it.
    fn fill_array_field_designator_items(
        &mut self,
        items: &[InitializerItem],
        layout: &crate::common::types::StructLayout,
        struct_size: usize,
        num_elems: usize,
        array_base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let mut current_elem_idx = 0usize;
        let mut current_field_idx = 0usize;
        for item in items {
            let mut elem_idx = current_elem_idx;
            let mut remaining_desigs_start = 0;

            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    if idx != current_elem_idx { current_field_idx = 0; }
                    elem_idx = idx;
                }
                remaining_desigs_start = 1;
            }

            let mut field_desig: Option<&str> = None;
            if let Some(Designator::Field(ref name)) = item.designators.get(remaining_desigs_start) {
                field_desig = Some(&*name);
                remaining_desigs_start += 1;
            }

            if elem_idx >= num_elems { current_elem_idx = elem_idx + 1; continue; }

            let elem_base = array_base_offset + elem_idx * struct_size;
            let resolution = layout.resolve_init_field(field_desig, current_field_idx, &*self.types.borrow_struct_layouts());
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Designator targets a field inside an anonymous struct/union member.
                    let extra_desigs = if item.designators.len() > remaining_desigs_start { &item.designators[remaining_desigs_start..] } else { &[] };
                    let anon_res = h::resolve_anonymous_member(layout, *anon_field_idx, inner_name, &item.init, extra_desigs, &self.types.borrow_struct_layouts());
                    if let Some(res) = anon_res {
                        let anon_offset = elem_base + res.anon_offset;
                        self.fill_struct_fields_from_items(
                            &[res.sub_item], &res.sub_layout, anon_offset, bytes, ptr_ranges,
                        );
                    }
                    current_elem_idx = elem_idx;
                    current_field_idx = *anon_field_idx + 1;
                    continue;
                }
                None => { current_elem_idx = elem_idx; continue; }
            };
            let field = &layout.fields[field_idx];
            let field_offset = elem_base + field.offset;

            if remaining_desigs_start < item.designators.len() {
                let remaining_item = InitializerItem {
                    designators: item.designators[remaining_desigs_start..].to_vec(),
                    init: item.init.clone(),
                };
                self.fill_nested_designator_with_ptrs(
                    &remaining_item, &field.ty, field_offset, bytes, ptr_ranges,
                );
            } else {
                self.emit_struct_field_init_compound(
                    item, field, field_offset, bytes, ptr_ranges,
                );
            }

            current_elem_idx = elem_idx;
            current_field_idx = field_idx + 1;
        }
    }

    /// Handle a multi-level designated initializer within a struct array element.
    /// Drills through the designator chain (starting from the second designator)
    /// to find the actual target sub-field, then writes integer values to the byte
    /// buffer and records pointer relocations in ptr_ranges.
    fn fill_nested_designator_with_ptrs(
        &mut self,
        item: &InitializerItem,
        outer_ty: &CType,
        outer_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        // Drill through designators starting from the second one
        let drill = match self.drill_designators(&item.designators[1..], outer_ty) {
            Some(d) => d,
            None => return,
        };

        let sub_offset = outer_offset + drill.byte_offset;
        let current_ty = drill.target_ty;

        match &item.init {
            Initializer::Expr(expr) => {
                self.write_expr_to_bytes_or_ptrs(
                    expr, &current_ty, sub_offset,
                    drill.bit_offset, drill.bit_width,
                    bytes, ptr_ranges,
                );
            }
            Initializer::List(inner_items) => {
                self.fill_composite_or_array_with_ptrs(
                    inner_items, &current_ty, sub_offset, bytes, ptr_ranges,
                );
            }
        }
    }

    /// Fill a nested struct's fields into the byte buffer and ptr_ranges,
    /// properly handling pointer/function pointer fields that need relocations.
    /// This is used when a struct field with a braced initializer list contains
    /// pointer-type sub-fields within a struct array element context.
    fn fill_nested_struct_with_ptrs(
        &mut self,
        inner_items: &[InitializerItem],
        sub_layout: &crate::common::types::StructLayout,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let mut current_field_idx = 0usize;
        let mut item_idx = 0usize;
        while item_idx < inner_items.len() {
            let inner_item = &inner_items[item_idx];
            let desig_name = h::first_field_designator(inner_item);
            let resolution = sub_layout.resolve_init_field(desig_name, current_field_idx, &*self.types.borrow_struct_layouts());
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Drill into anonymous member
                    let anon_field = &sub_layout.fields[*anon_field_idx];
                    let anon_offset = base_offset + anon_field.offset;
                    if let Some(anon_layout) = self.get_struct_layout_for_ctype(&anon_field.ty) {
                        let mut synth_desigs = vec![Designator::Field(inner_name.clone())];
                        if inner_item.designators.len() > 1 {
                            synth_desigs.extend(inner_item.designators[1..].iter().cloned());
                        }
                        let sub_item = InitializerItem {
                            designators: synth_desigs,
                            init: inner_item.init.clone(),
                        };
                        self.fill_nested_struct_with_ptrs(
                            &[sub_item], &anon_layout, anon_offset, bytes, ptr_ranges);
                    }
                    current_field_idx = *anon_field_idx + 1;
                    item_idx += 1;
                    // For unions, only initialize the first (or designated) member
                    if sub_layout.is_union && desig_name.is_none() { break; }
                    continue;
                }
                None => break,
            };

            // Handle multi-level designators within the nested struct (e.g., .config.i = &val)
            if h::has_nested_field_designator(inner_item)
                && field_idx < sub_layout.fields.len() {
                    let field = &sub_layout.fields[field_idx];
                    let field_abs_offset = base_offset + field.offset;

                    if let Some(drill) = self.drill_designators(&inner_item.designators[1..], &field.ty) {
                        let sub_offset = field_abs_offset + drill.byte_offset;
                        if let Initializer::Expr(ref expr) = inner_item.init {
                            self.write_expr_to_bytes_or_ptrs(
                                expr, &drill.target_ty, sub_offset,
                                drill.bit_offset, drill.bit_width,
                                bytes, ptr_ranges,
                            );
                        }
                    }
                    current_field_idx = field_idx + 1;
                    item_idx += 1;
                    if sub_layout.is_union && desig_name.is_none() { break; }
                    continue;
                }

            let field = &sub_layout.fields[field_idx];
            let field_abs_offset = base_offset + field.offset;

            // Brace elision for array fields: when an expression (not a braced list) targets
            // an array field, consume consecutive items to fill the array elements.
            if matches!(&inner_item.init, Initializer::Expr(_)) {
                if let CType::Array(ref elem_ty, Some(arr_size)) = field.ty {
                    let is_char_array_str = matches!(elem_ty.as_ref(), CType::Char | CType::UChar)
                        && matches!(&inner_item.init, Initializer::Expr(Expr::StringLiteral(..)));
                    if !is_char_array_str {
                        let consumed = self.fill_flat_array_field_with_ptrs(
                            &inner_items[item_idx..], elem_ty, arr_size,
                            field_abs_offset, bytes, ptr_ranges,
                        );
                        item_idx += consumed;
                        current_field_idx = field_idx + 1;
                        if sub_layout.is_union && desig_name.is_none() { break; }
                        continue;
                    }
                }
            }

            if let Initializer::Expr(ref expr) = inner_item.init {
                self.write_expr_to_bytes_or_ptrs(
                    expr, &field.ty, field_abs_offset,
                    field.bit_offset, field.bit_width,
                    bytes, ptr_ranges,
                );
            } else if let Initializer::List(ref nested_items) = inner_item.init {
                self.fill_composite_or_array_with_ptrs(
                    nested_items, &field.ty, field_abs_offset, bytes, ptr_ranges,
                );
            }
            current_field_idx = field_idx + 1;
            item_idx += 1;
            if sub_layout.is_union && desig_name.is_none() { break; }
        }
    }

    /// Write a scalar expression value to either the byte buffer or ptr_ranges,
    /// depending on whether the target type is a pointer/function pointer.
    /// Handles: pointer fields (resolve as GlobalAddr), bitfields, regular scalars.
    fn write_expr_to_bytes_or_ptrs(
        &mut self,
        expr: &Expr,
        ty: &CType,
        offset: usize,
        bit_offset: Option<u32>,
        bit_width: Option<u32>,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let is_ptr = matches!(ty, CType::Pointer(_, _) | CType::Function(_));
        if is_ptr {
            if let Some(addr_init) = self.resolve_ptr_field_init(expr) {
                ptr_ranges.push((offset, addr_init));
            } else if let Some(val) = self.eval_const_expr(expr) {
                let ir_ty = IrType::from_ctype(ty);
                self.write_const_to_bytes(bytes, offset, &val, ir_ty);
            }
        } else if let (Some(bo), Some(bw)) = (bit_offset, bit_width) {
            let ir_ty = IrType::from_ctype(ty);
            let val = self.eval_init_scalar(&Initializer::Expr(expr.clone()));
            self.write_bitfield_to_bytes(bytes, offset, &val, ir_ty, bo, bw);
        } else if let Expr::StringLiteral(s, _) = expr {
            // String literal initializing a char array field (e.g., {"hello", &ptr} in a struct
            // with pointer members). write_string_to_bytes handles copying the string bytes.
            if let CType::Array(ref elem, Some(arr_size)) = ty {
                if matches!(elem.as_ref(), CType::Char | CType::UChar) {
                    Self::write_string_to_bytes(bytes, offset, s, *arr_size);
                }
            }
        } else if let Some(val) = self.eval_const_expr(expr) {
            let ir_ty = IrType::from_ctype(ty);
            self.write_const_to_bytes(bytes, offset, &val, ir_ty);
        }
    }

    /// Consume consecutive expression items from `items` to flat-initialize an array field.
    /// This implements brace elision (C11 6.7.9p17-21) for the bytes+ptrs global init path:
    /// when an expression item targets an array field, subsequent items fill later array elements.
    /// Returns the number of items consumed.
    fn fill_flat_array_field_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) -> usize {
        let elem_size = self.resolve_ctype_size(elem_ty);
        let mut consumed = 0usize;
        while consumed < arr_size && consumed < items.len() {
            let item = &items[consumed];
            // Stop if we hit a designator (it targets a different field)
            if !item.designators.is_empty() && consumed > 0 {
                break;
            }
            // Stop if we hit a braced list (it's a new sub-aggregate)
            if matches!(&item.init, Initializer::List(_)) && consumed > 0 {
                break;
            }
            let elem_offset = field_offset + consumed * elem_size;
            if let Initializer::Expr(ref expr) = item.init {
                // For struct elements within the array, handle recursively
                if let Some(elem_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                    // A single expression initializes the first field of a struct element
                    if !elem_layout.fields.is_empty() {
                        let f = &elem_layout.fields[0];
                        self.write_expr_to_bytes_or_ptrs(
                            expr, &f.ty, elem_offset + f.offset,
                            f.bit_offset, f.bit_width,
                            bytes, ptr_ranges,
                        );
                    }
                } else {
                    self.write_expr_to_bytes_or_ptrs(
                        expr, elem_ty, elem_offset, None, None, bytes, ptr_ranges,
                    );
                }
            } else if let Initializer::List(ref sub_items) = item.init {
                // Braced sub-list for first element
                self.fill_composite_or_array_with_ptrs(
                    sub_items, elem_ty, elem_offset, bytes, ptr_ranges,
                );
            }
            consumed += 1;
        }
        consumed.max(1) // Always consume at least one item
    }

    /// Fill an array of scalar/pointer elements into byte buffer + ptr_ranges.
    /// For pointer types, writes address relocations to ptr_ranges.
    /// For non-pointer types, writes constant values directly to the byte buffer.
    fn fill_scalar_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        elem_ty: &CType,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let elem_size = self.resolve_ctype_size(elem_ty);
        let has_ptrs = h::type_has_pointer_elements(elem_ty, &*self.types.borrow_struct_layouts());
        let elem_ir_ty = IrType::from_ctype(elem_ty);
        let mut current_idx = 0usize;
        for item in items.iter() {
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    current_idx = idx;
                }
            }
            let elem_offset = base_offset + current_idx * elem_size;
            if let Initializer::Expr(ref expr) = item.init {
                if has_ptrs {
                    self.write_expr_to_bytes_or_ptrs(
                        expr, elem_ty, elem_offset, None, None, bytes, ptr_ranges,
                    );
                } else if let Some(val) = self.eval_const_expr(expr) {
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
            }
            current_idx += 1;
        }
    }

    /// Fill a struct/union/array field into byte buffer + ptr_ranges, choosing the
    /// pointer-aware path or plain byte path based on whether the type contains pointers.
    fn fill_composite_or_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        field_ty: &CType,
        offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        if let Some(sub_layout) = self.get_struct_layout_for_ctype(field_ty) {
            if sub_layout.has_pointer_fields(&*self.types.borrow_struct_layouts()) {
                self.fill_nested_struct_with_ptrs(items, &sub_layout, offset, bytes, ptr_ranges);
            } else {
                self.fill_struct_global_bytes(items, &sub_layout, bytes, offset);
            }
        } else if let CType::Array(elem_ty, _) = field_ty {
            // Array field: check if elements are structs/unions with pointer fields
            if let Some(elem_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                if elem_layout.has_pointer_fields(&*self.types.borrow_struct_layouts()) {
                    // Array of structs with pointer fields: handle each element
                    let struct_size = elem_layout.size;
                    for (ai, item) in items.iter().enumerate() {
                        let elem_offset = offset + ai * struct_size;
                        match &item.init {
                            Initializer::List(sub_items) => {
                                self.fill_nested_struct_with_ptrs(
                                    sub_items, &elem_layout, elem_offset, bytes, ptr_ranges,
                                );
                            }
                            Initializer::Expr(expr) => {
                                // Single expression for first field of struct element
                                if !elem_layout.fields.is_empty() {
                                    let field = &elem_layout.fields[0];
                                    let field_offset = elem_offset + field.offset;
                                    self.write_expr_to_bytes_or_ptrs(
                                        expr, &field.ty, field_offset,
                                        field.bit_offset, field.bit_width,
                                        bytes, ptr_ranges,
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // Array of structs without pointer fields: byte serialization.
                    // Handles both braced sub-items (List) and flat/brace-elided
                    // scalar items (Expr) that fill struct fields sequentially.
                    let struct_size = elem_layout.size;
                    let mut sub_idx = 0usize;
                    let mut ai = 0usize;
                    let arr_size = match field_ty {
                        CType::Array(_, Some(s)) => *s,
                        _ => items.len(),
                    };
                    while ai < arr_size && sub_idx < items.len() {
                        let elem_offset = offset + ai * struct_size;
                        match &items[sub_idx].init {
                            Initializer::List(ref sub_items) => {
                                self.fill_struct_global_bytes(sub_items, &elem_layout, bytes, elem_offset);
                                sub_idx += 1;
                            }
                            Initializer::Expr(_) => {
                                // Flat/brace-elided init: pass remaining items to fill
                                // this struct element field-by-field.
                                let consumed = self.fill_struct_global_bytes(&items[sub_idx..], &elem_layout, bytes, elem_offset);
                                sub_idx += consumed;
                            }
                        }
                        ai += 1;
                    }
                }
            } else if let CType::Array(ref inner_elem, Some(inner_size)) = elem_ty.as_ref() {
                // Multi-dimensional array of scalars (e.g., unsigned char hash[2][4]):
                // each item is a braced list for one row of the outer dimension.
                let elem_size = self.resolve_ctype_size(elem_ty);
                self.fill_multidim_array_field(items, inner_elem, *inner_size, items.len(), elem_size, bytes, offset);
            } else {
                // Array of non-composite elements (scalars, pointers, etc.)
                self.fill_scalar_array_with_ptrs(items, elem_ty, offset, bytes, ptr_ranges);
            }
        } else {
            // Non-composite, non-array field: write scalar values
            self.fill_scalar_array_with_ptrs(items, field_ty, offset, bytes, ptr_ranges);
        }
    }

    /// Brace-elided initialization of a struct/union field from remaining items.
    /// When a scalar expression targets a struct/union field without braces, this
    /// distributes consecutive items across the sub-struct's fields (C11 6.7.9p13-17).
    /// Uses the pointer-aware path for sub-fields that may contain pointer types.
    /// Returns the number of items consumed.
    fn fill_nested_struct_brace_elided(
        &mut self,
        items: &[InitializerItem],
        sub_layout: &crate::common::types::StructLayout,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) -> usize {
        match &items[0].init {
            Initializer::List(sub_items) => {
                self.fill_nested_struct_with_ptrs(
                    sub_items, sub_layout, base_offset, bytes, ptr_ranges,
                );
                1
            }
            Initializer::Expr(expr) => {
                // Unwrap compound literal: (type){ init_list } -> use inner init_list
                if let Expr::CompoundLiteral(_, ref cl_init, _) = expr {
                    if let Initializer::List(sub_items) = cl_init.as_ref() {
                        self.fill_nested_struct_with_ptrs(
                            sub_items, sub_layout, base_offset, bytes, ptr_ranges,
                        );
                        return 1;
                    }
                }
                // Brace-elided: pass remaining items to fill_nested_struct_with_ptrs
                // which will consume items field-by-field for the sub-struct.
                let consumed = self.fill_nested_struct_with_ptrs_count(
                    items, sub_layout, base_offset, bytes, ptr_ranges,
                );
                if consumed == 0 { 1 } else { consumed }
            }
        }
    }

    /// Like fill_nested_struct_with_ptrs but returns the number of items consumed.
    /// Used for brace-elided initialization where we need to know how many items
    /// were used to fill the sub-struct.
    fn fill_nested_struct_with_ptrs_count(
        &mut self,
        items: &[InitializerItem],
        sub_layout: &crate::common::types::StructLayout,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) -> usize {
        let mut current_field_idx = 0usize;
        let mut item_idx = 0usize;
        while item_idx < items.len() {
            let inner_item = &items[item_idx];
            let desig_name = h::first_field_designator(inner_item);
            let resolution = sub_layout.resolve_init_field(desig_name, current_field_idx, &*self.types.borrow_struct_layouts());
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    let anon_field = &sub_layout.fields[*anon_field_idx];
                    let anon_offset = base_offset + anon_field.offset;
                    if let Some(anon_layout) = self.get_struct_layout_for_ctype(&anon_field.ty) {
                        let sub_item = InitializerItem {
                            designators: vec![Designator::Field(inner_name.clone())],
                            init: inner_item.init.clone(),
                        };
                        self.fill_nested_struct_with_ptrs(
                            &[sub_item], &anon_layout, anon_offset, bytes, ptr_ranges);
                    }
                    current_field_idx = *anon_field_idx + 1;
                    item_idx += 1;
                    if sub_layout.is_union && desig_name.is_none() { break; }
                    continue;
                }
                None => break,
            };

            let field = &sub_layout.fields[field_idx];
            let field_abs_offset = base_offset + field.offset;

            // Brace elision for array fields within the sub-struct
            if matches!(&inner_item.init, Initializer::Expr(_)) {
                if let CType::Array(ref elem_ty, Some(arr_size)) = field.ty {
                    let is_char_array_str = matches!(elem_ty.as_ref(), CType::Char | CType::UChar)
                        && matches!(&inner_item.init, Initializer::Expr(Expr::StringLiteral(..)));
                    if !is_char_array_str {
                        let consumed = self.fill_flat_array_field_with_ptrs(
                            &items[item_idx..], elem_ty, arr_size,
                            field_abs_offset, bytes, ptr_ranges,
                        );
                        item_idx += consumed;
                        current_field_idx = field_idx + 1;
                        if sub_layout.is_union && desig_name.is_none() { break; }
                        continue;
                    }
                }
            }

            if let Initializer::Expr(ref expr) = inner_item.init {
                self.write_expr_to_bytes_or_ptrs(
                    expr, &field.ty, field_abs_offset,
                    field.bit_offset, field.bit_width,
                    bytes, ptr_ranges,
                );
            } else if let Initializer::List(ref nested_items) = inner_item.init {
                self.fill_composite_or_array_with_ptrs(
                    nested_items, &field.ty, field_abs_offset, bytes, ptr_ranges,
                );
            }
            current_field_idx = field_idx + 1;
            item_idx += 1;
            if sub_layout.is_union && desig_name.is_none() { break; }
        }
        item_idx
    }
}
