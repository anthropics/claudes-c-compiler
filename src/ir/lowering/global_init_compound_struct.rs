//! Struct and union compound global initialization.
//!
//! Contains the primary `lower_struct_global_init_compound` method and its helpers
//! for field-by-field relocation-aware initialization of structs/unions, including:
//! - Flexible array member (FAM) handling
//! - Bitfield packing into storage units
//! - Union single-field emission
//! - Sub-struct compound vs byte-level selection
//! - Nested compound flattening and bytes+ptrs merging

use crate::frontend::parser::ast::{
    Designator,
    Expr,
    Initializer,
    InitializerItem,
};
use std::rc::Rc;
use crate::ir::reexports::{GlobalInit, IrConst};
use crate::common::types::{IrType, StructLayout, CType, InitFieldResolution};
use super::lower::Lowerer;
use super::global_init_helpers as h;
use h::{push_zero_bytes, push_bytes_as_elements, push_string_as_elements};

impl Lowerer {
    /// Lower a struct global init that contains address expressions.
    /// Emits field-by-field using Compound, with padding bytes between fields.
    /// Handles flat init (where multiple items fill an array-of-pointer field),
    /// braced init, and designated init patterns.
    pub(super) fn lower_struct_global_init_compound(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
    ) -> GlobalInit {
        let fam_extra = self.compute_fam_extra_size(items, layout);
        let total_size = layout.size + fam_extra;
        let mut elements: Vec<GlobalInit> = Vec::new();
        let mut current_offset = 0usize;

        // Build a map of field_idx -> list of initializer items.
        // For array fields with flat init, multiple items may map to the same field.
        let mut field_inits: Vec<Vec<&InitializerItem>> = vec![Vec::new(); layout.fields.len()];
        let mut current_field_idx = 0usize;

        // Collect items targeting anonymous members that need synthetic sub-inits.
        // We store them separately to avoid borrow issues with field_inits.
        let mut anon_synth_items: Vec<(usize, InitializerItem)> = Vec::new();

        let mut item_idx = 0;
        while item_idx < items.len() {
            let item = &items[item_idx];

            let designator_name = h::first_field_designator(item);
            let resolution = layout.resolve_init_field(designator_name, current_field_idx, &*self.types.borrow_struct_layouts());

            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Designator targets a field inside an anonymous member.
                    // Create a synthetic init item with the inner designator,
                    // preserving any remaining nested designators from the original item.
                    // e.g., for `.base.cra_name = "val"` where `base` is inside an anonymous
                    // union, the synthetic item must be `.base.cra_name = "val"` (not just `.base`).
                    let mut synth_desigs = vec![Designator::Field(inner_name.clone())];
                    if item.designators.len() > 1 {
                        synth_desigs.extend(item.designators[1..].iter().cloned());
                    }
                    let synth_item = InitializerItem {
                        designators: synth_desigs,
                        init: item.init.clone(),
                    };
                    anon_synth_items.push((*anon_field_idx, synth_item));
                    current_field_idx = *anon_field_idx + 1;
                    item_idx += 1;
                    let has_designator = !item.designators.is_empty();
                    if layout.is_union && !has_designator { break; }
                    continue;
                }
                None => {
                    item_idx += 1;
                    continue;
                }
            };
            if field_idx >= layout.fields.len() {
                item_idx += 1;
                continue;
            }

            let field_ty = &layout.fields[field_idx].ty;

            // Check if this is a flat init filling an array field.
            // A string literal initializing a char array is NOT flat init - it's a single
            // complete initializer for the entire array (e.g., char c[10] = "hello").
            // However, for pointer arrays (e.g., char *s[2] = {"abc", "def"}), each
            // string literal IS a flat-init element (it initializes one pointer).
            let is_string_literal = matches!(&item.init, Initializer::Expr(Expr::StringLiteral(..)));
            if let CType::Array(elem_ty, Some(arr_size)) = field_ty {
                let is_char_array = matches!(elem_ty.as_ref(), CType::Char | CType::UChar);
                let skip_flat = is_string_literal && is_char_array;
                if matches!(&item.init, Initializer::Expr(_)) && !skip_flat {
                    // Flat init: consume the right number of items for this array field.
                    // For arrays of scalars/pointers, each element is 1 item.
                    // For arrays of structs, each element needs multiple items
                    // (one per scalar leaf field in the struct).
                    let scalars_per_elem = h::count_flat_init_scalars(elem_ty, &*self.types.borrow_struct_layouts());
                    let total_scalars = *arr_size * scalars_per_elem;
                    let mut consumed = 0;
                    while consumed < total_scalars && (item_idx + consumed) < items.len() {
                        let cur_item = &items[item_idx + consumed];
                        // Stop if we hit a designator targeting a different field
                        if !cur_item.designators.is_empty() && consumed > 0 {
                            break;
                        }
                        if matches!(&cur_item.init, Initializer::List(_)) && consumed > 0 {
                            break;
                        }
                        field_inits[field_idx].push(cur_item);
                        consumed += 1;
                    }
                    item_idx += consumed;
                    current_field_idx = field_idx + 1;
                    let has_designator = !item.designators.is_empty();
                    if layout.is_union && !has_designator { break; }
                    continue;
                }
            }

            // Check if this is a flat (brace-elided) init filling a struct/union field.
            // When a sub-struct field is initialized without inner braces, we need to
            // consume enough flat items to fill all scalar leaf fields of the sub-struct,
            // rather than just consuming one item and advancing to the next outer field.
            // e.g., `struct { Inner x; void *y; } = {1, 2, 3, ptr}` where Inner has 3 fields:
            // items 0-2 should fill x, and item 3 should fill y.
            if matches!(field_ty, CType::Struct(_) | CType::Union(_))
                && matches!(&item.init, Initializer::Expr(_))
                && item.designators.is_empty()
            {
                let scalars_needed = h::count_flat_init_scalars(field_ty, &*self.types.borrow_struct_layouts());
                if scalars_needed > 1 {
                    let mut consumed = 0;
                    while consumed < scalars_needed && (item_idx + consumed) < items.len() {
                        let cur_item = &items[item_idx + consumed];
                        // Stop if we hit a designator targeting a different field
                        if !cur_item.designators.is_empty() && consumed > 0 {
                            break;
                        }
                        if matches!(&cur_item.init, Initializer::List(_)) && consumed > 0 {
                            break;
                        }
                        field_inits[field_idx].push(cur_item);
                        consumed += 1;
                    }
                    item_idx += consumed;
                    current_field_idx = field_idx + 1;
                    let has_designator = !item.designators.is_empty();
                    if layout.is_union && !has_designator { break; }
                    continue;
                }
            }

            field_inits[field_idx].push(item);
            current_field_idx = field_idx + 1;
            item_idx += 1;
            let has_designator = !item.designators.is_empty();
            if layout.is_union && !has_designator { break; }
        }

        // For unions, find the one initialized field and emit only that,
        // padding to the full union size. Non-union structs emit all fields.
        if layout.is_union {
            // Find which field (if any) has an initializer
            let mut init_fi = None;
            for (i, inits) in field_inits.iter().enumerate() {
                if !inits.is_empty() {
                    init_fi = Some(i);
                    break;
                }
            }
            // Also check anon_synth_items for anonymous member designated inits
            // (e.g., .name = "x" targeting an anonymous struct inside this union)
            if init_fi.is_none() {
                if let Some((idx, _)) = anon_synth_items.first() {
                    init_fi = Some(*idx);
                }
            }
            let union_size = layout.size;
            if let Some(fi) = init_fi {
                let inits = &field_inits[fi];
                let field_size = self.resolve_ctype_size(&layout.fields[fi].ty);
                if !inits.is_empty() {
                    self.emit_field_inits_compound(&mut elements, inits, &layout.fields[fi], field_size);
                } else {
                    // Field has no direct inits but may have anonymous member inits
                    let anon_items_for_fi: Vec<InitializerItem> = anon_synth_items.iter()
                        .filter(|(idx, _)| *idx == fi)
                        .map(|(_, item)| item.clone())
                        .collect();
                    if !anon_items_for_fi.is_empty() {
                        let anon_field_ty = &layout.fields[fi].ty;
                        if let Some(sub_layout) = self.get_struct_layout_for_ctype(anon_field_ty) {
                            let sub_init = self.lower_struct_global_init_compound(&anon_items_for_fi, &sub_layout);
                            Self::append_nested_compound(&mut elements, sub_init, field_size);
                        } else {
                            push_zero_bytes(&mut elements, field_size);
                        }
                    } else {
                        push_zero_bytes(&mut elements, field_size);
                    }
                }
                // Pad to full union size
                if field_size < union_size {
                    push_zero_bytes(&mut elements, union_size - field_size);
                }
                current_offset = union_size;
            } else {
                // No initialized field - zero fill entire union
                push_zero_bytes(&mut elements, union_size);
                current_offset = union_size;
            }
        } else {

        let mut fi = 0;
        while fi < layout.fields.len() {
            let field_offset = layout.fields[fi].offset;
            let field_size = self.resolve_ctype_size(&layout.fields[fi].ty);

            // Check if this is a bitfield: if so, we need to pack all bitfields
            // sharing the same storage unit into a single byte buffer.
            if layout.fields[fi].bit_offset.is_some() {
                let storage_unit_offset = field_offset;
                let storage_unit_size = field_size; // All bitfields in this unit have the same type size

                // Emit padding before this storage unit (only if unit starts after current pos)
                if storage_unit_offset > current_offset {
                    let pad = storage_unit_offset - current_offset;
                    push_zero_bytes(&mut elements, pad);
                    current_offset = storage_unit_offset;
                }

                // Allocate a byte buffer for the storage unit and pack all
                // bitfield values into it using read-modify-write.
                let mut unit_bytes = vec![0u8; storage_unit_size];

                // Track the highest bit used so we know exactly how many bytes
                // the bitfields actually occupy (the storage unit type may be
                // wider than the bits used, and the remaining bytes may belong
                // to subsequent non-bitfield fields).
                let mut max_bit_end: u32 = 0;

                // Process all consecutive bitfield fields sharing this storage unit offset
                while fi < layout.fields.len()
                    && layout.fields[fi].offset == storage_unit_offset
                    && layout.fields[fi].bit_offset.is_some()
                {
                    let bit_offset = layout.fields[fi].bit_offset.expect("bitfield must have bit_offset");
                    let bit_width = layout.fields[fi].bit_width.expect("bitfield must have bit_width");
                    let field_ir_ty = IrType::from_ctype(&layout.fields[fi].ty);

                    let inits = &field_inits[fi];
                    let val = if !inits.is_empty() {
                        self.eval_init_scalar(&inits[0].init)
                    } else {
                        IrConst::I32(0) // Zero-init for missing initializers
                    };

                    // Pack this bitfield value into the storage unit buffer
                    self.write_bitfield_to_bytes(&mut unit_bytes, 0, &val, field_ir_ty, bit_offset, bit_width);
                    if bit_offset + bit_width > max_bit_end {
                        max_bit_end = bit_offset + bit_width;
                    }
                    fi += 1;
                }

                // Compute the actual number of bytes occupied by bitfield data.
                // The storage unit type (e.g. uint32_t = 4 bytes) may be wider
                // than the bits actually used. Bytes beyond the last used bit
                // may belong to subsequent non-bitfield fields and must not be
                // overwritten with zeros from the storage unit buffer.
                let actual_bytes_used = (max_bit_end as usize).div_ceil(8);
                let effective_size = actual_bytes_used.min(storage_unit_size);

                // When the storage unit overlaps with already-written data
                // (due to align_down placement), skip the overlapping bytes.
                let skip = current_offset.saturating_sub(storage_unit_offset);
                // Emit only the non-overlapping portion of the actually-used bytes
                if skip < effective_size {
                    push_bytes_as_elements(&mut elements, &unit_bytes[skip..effective_size]);
                }
                current_offset = storage_unit_offset + effective_size;
                continue;
            }

            // Emit padding before this field
            if field_offset > current_offset {
                let pad = field_offset - current_offset;
                push_zero_bytes(&mut elements, pad);
                current_offset = field_offset;
            }

            let inits = &field_inits[fi];

            // Flexible array member (FAM): use fam_extra as the actual data size
            if let CType::Array(ref elem_ty, None) = layout.fields[fi].ty {
                if fam_extra > 0 && !inits.is_empty() {
                    self.emit_fam_compound(&mut elements, inits, elem_ty, fam_extra);
                    current_offset += fam_extra;
                }
                fi += 1;
                continue;
            }

            // Check for synthetic items from anonymous member designated inits
            if inits.is_empty() {
                let anon_items_for_fi: Vec<InitializerItem> = anon_synth_items.iter()
                    .filter(|(idx, _)| *idx == fi)
                    .map(|(_, item)| item.clone())
                    .collect();
                if !anon_items_for_fi.is_empty() {
                    let anon_field_ty = &layout.fields[fi].ty;
                    if let Some(sub_layout) = self.get_struct_layout_for_ctype(anon_field_ty) {
                        let sub_init = self.lower_struct_global_init_compound(&anon_items_for_fi, &sub_layout);
                        Self::append_nested_compound(&mut elements, sub_init, field_size);
                    } else {
                        push_zero_bytes(&mut elements, field_size);
                    }
                } else {
                    push_zero_bytes(&mut elements, field_size);
                }
            } else {
                self.emit_field_inits_compound(&mut elements, inits, &layout.fields[fi], field_size);
            }
            current_offset += field_size;
            fi += 1;
        }

        } // end of else (non-union struct path)

        // Trailing padding
        if current_offset < total_size {
            push_zero_bytes(&mut elements, total_size - current_offset);
        }

        GlobalInit::Compound(elements)
    }

    /// Emit flexible array member (FAM) elements into compound init.
    /// Each FAM element may be a struct containing pointer fields that need relocations.
    fn emit_fam_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        inits: &[&InitializerItem],
        elem_ty: &CType,
        fam_data_size: usize,
    ) {
        let elem_size = self.resolve_ctype_size(elem_ty);
        if elem_size == 0 { return; }
        let elem_is_pointer = matches!(elem_ty, CType::Pointer(_, _) | CType::Function(_));

        // The FAM init should be a single item with a List initializer containing sub-items
        // (one per FAM element), e.g. .numbers = { { .nr = 0, .ns = &init_pid_ns } }
        let sub_items = if inits.len() == 1 {
            if let Initializer::List(ref list) = inits[0].init {
                list.as_slice()
            } else {
                // Single expression init for a FAM element
                let init_item = inits[0];
                self.emit_compound_field_init(elements, &init_item.init, elem_ty, elem_size, elem_is_pointer);
                let emitted = elem_size;
                if emitted < fam_data_size {
                    push_zero_bytes(elements, fam_data_size - emitted);
                }
                return;
            }
        } else {
            // Multiple flat items - shouldn't normally happen for struct FAMs
            // but handle by emitting each as an element
            let mut emitted = 0;
            for item in inits {
                if emitted + elem_size > fam_data_size { break; }
                self.emit_compound_field_init(elements, &item.init, elem_ty, elem_size, elem_is_pointer);
                emitted += elem_size;
            }
            if emitted < fam_data_size {
                push_zero_bytes(elements, fam_data_size - emitted);
            }
            return;
        };

        // Emit each FAM element
        let num_elems = fam_data_size / elem_size;
        let mut emitted = 0;
        for (i, sub_item) in sub_items.iter().enumerate() {
            if i >= num_elems { break; }
            // Each sub_item is an initializer for one FAM element (e.g., one struct upid)
            if let Some(sub_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                // Struct element - use sub-struct compound emission
                match &sub_item.init {
                    Initializer::List(nested_items) => {
                        self.emit_sub_struct_to_compound(elements, nested_items, &sub_layout, elem_size);
                    }
                    _ => {
                        self.emit_compound_field_init(elements, &sub_item.init, elem_ty, elem_size, elem_is_pointer);
                    }
                }
            } else {
                self.emit_compound_field_init(elements, &sub_item.init, elem_ty, elem_size, elem_is_pointer);
            }
            emitted += elem_size;
        }
        // Zero-fill remaining FAM elements
        if emitted < fam_data_size {
            push_zero_bytes(elements, fam_data_size - emitted);
        }
    }

    /// Emit a sub-struct's initialization into compound elements, choosing between
    /// compound (relocation-aware) and byte-level approaches based on whether any
    /// field contains address expressions.
    pub(super) fn emit_sub_struct_to_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        sub_items: &[InitializerItem],
        sub_layout: &StructLayout,
        field_size: usize,
    ) {
        if self.struct_init_has_addr_fields(sub_items, sub_layout) {
            let nested = self.lower_struct_global_init_compound(sub_items, sub_layout);
            Self::append_nested_compound(elements, nested, field_size);
        } else {
            let mut bytes = vec![0u8; field_size];
            self.fill_struct_global_bytes(sub_items, sub_layout, &mut bytes, 0);
            push_bytes_as_elements(elements, &bytes);
        }
    }

    /// Append the elements from a nested GlobalInit::Compound into `elements`,
    /// padding/truncating to exactly `target_size` bytes.
    fn append_nested_compound(elements: &mut Vec<GlobalInit>, nested: GlobalInit, target_size: usize) {
        if let GlobalInit::Compound(nested_elems) = nested {
            let mut emitted = 0;
            for elem in nested_elems {
                if emitted >= target_size { break; }
                emitted += elem.byte_size();
                elements.push(elem);
            }
            while emitted < target_size {
                elements.push(GlobalInit::Scalar(IrConst::I8(0)));
                emitted += 1;
            }
        } else {
            push_zero_bytes(elements, target_size);
        }
    }

    /// Merge a byte buffer with pointer relocations into a Compound global init.
    /// Pointer offsets in `ptr_ranges` replace the corresponding byte ranges with
    /// GlobalAddr entries; all other positions become I8 scalar bytes.
    /// This is the shared finalization step for struct arrays with pointer fields.
    pub(super) fn build_compound_from_bytes_and_ptrs(
        bytes: Vec<u8>,
        mut ptr_ranges: Vec<(usize, GlobalInit)>,
        total_size: usize,
    ) -> GlobalInit {
        ptr_ranges.sort_by_key(|&(off, _)| off);
        // Deduplicate overlapping pointer ranges that can arise from:
        // 1. Duplicate designated initializers for the same field (C allows
        //    later designators to override earlier ones, e.g.,
        //    `.matches = f, MACRO_THAT_ALSO_SETS_MATCHES`)
        // 2. Union branches with pointer fields at different offsets where
        //    multiple branches are recorded due to sequential field processing
        // For same-offset duplicates, keep the last entry (C last-designator-wins).
        // For overlapping ranges at different offsets, keep the first (already placed).
        let ptr_sz = crate::common::types::target_ptr_size();
        let mut deduped: Vec<(usize, GlobalInit)> = Vec::new();
        for (off, init) in ptr_ranges {
            if let Some(last) = deduped.last_mut() {
                if off == last.0 {
                    // Same offset: last-designator-wins (replace previous)
                    *last = (off, init);
                    continue;
                }
                let last_end = last.0 + ptr_sz;
                if off < last_end {
                    // Overlapping different offsets: skip (keep first)
                    continue;
                }
            }
            deduped.push((off, init));
        }
        let ptr_ranges = deduped;
        let mut elements: Vec<GlobalInit> = Vec::new();
        let mut pos = 0;
        for (ptr_off, ref addr_init) in &ptr_ranges {
            push_bytes_as_elements(&mut elements, &bytes[pos..*ptr_off]);
            elements.push(addr_init.clone());
            pos = ptr_off + ptr_sz;
        }
        push_bytes_as_elements(&mut elements, &bytes[pos..total_size]);
        GlobalInit::Compound(elements)
    }

    /// Emit a single field initializer in compound (relocation-aware) mode.
    pub(super) fn emit_compound_field_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        init: &Initializer,
        field_ty: &CType,
        field_size: usize,
        field_is_pointer: bool,
    ) {
        // Complex fields: emit {real, imag} as byte pairs
        if field_ty.is_complex() {
            let mut bytes = vec![0u8; field_size];
            self.write_complex_field_to_bytes(&mut bytes, 0, field_ty, init);
            push_bytes_as_elements(elements, &bytes);
            return;
        }

        match init {
            Initializer::Expr(expr) => {
                if let Expr::StringLiteral(s, _) = expr {
                    if field_is_pointer {
                        // String literal initializing a pointer field:
                        // create a .rodata string entry and emit GlobalAddr
                        let label = self.intern_string_literal(s);
                        elements.push(GlobalInit::GlobalAddr(Rc::from(label.as_str())));
                    } else {
                        // String literal initializing a char array field
                        push_string_as_elements(elements, s, field_size);
                    }
                } else if let Expr::AddressOf(inner, _) = expr {
                    if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = inner.as_ref() {
                        // &(compound_literal) at file scope: create anonymous global
                        let addr_init = self.create_compound_literal_global(cl_type_spec, cl_init);
                        elements.push(addr_init);
                    } else if let Some(addr_init) = self.eval_string_literal_addr_expr(expr) {
                        elements.push(addr_init);
                    } else if let Some(addr_init) = self.eval_global_addr_expr(expr) {
                        elements.push(addr_init);
                    } else {
                        push_zero_bytes(elements, field_size);
                    }
                } else if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = expr {
                    if field_is_pointer {
                        // Compound literal initializing a pointer field (array-to-pointer decay):
                        // create anonymous global storage and emit its address
                        let addr_init = self.create_compound_literal_global(cl_type_spec, cl_init);
                        elements.push(addr_init);
                    } else {
                        // Non-pointer compound literal: unwrap and use inner initializer
                        self.emit_compound_field_init(elements, cl_init, field_ty, field_size, field_is_pointer);
                    }
                } else {
                    // Scalar constant, string literal addr, global addr, or zero fallback
                    self.emit_expr_to_compound(elements, expr, field_size, Some(field_ty));
                }
            }
            Initializer::List(nested_items) => {
                // Check if this is an array whose elements contain pointers
                if let CType::Array(elem_ty, Some(arr_size)) = field_ty {
                    if h::type_has_pointer_elements(elem_ty, &*self.types.borrow_struct_layouts()) {
                        // Distinguish: direct pointer array vs struct-with-pointer-fields array
                        if matches!(elem_ty.as_ref(), CType::Pointer(_, _) | CType::Function(_)) {
                            // Array of direct pointers: emit each element as a GlobalAddr or zero
                            self.emit_compound_ptr_array_init(elements, nested_items, elem_ty, *arr_size);
                            return;
                        }
                        // Array of structs/unions containing pointer fields:
                        // use the struct-array-with-ptrs path (byte buffer + ptr_ranges)
                        if let Some(elem_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                            let nested = self.lower_struct_array_with_ptrs(nested_items, &elem_layout, *arr_size);
                            if let GlobalInit::Compound(nested_elems) = nested {
                                elements.extend(nested_elems);
                            } else {
                                push_zero_bytes(elements, field_size);
                            }
                            return;
                        }
                        // Multi-dimensional array of structs with pointer fields:
                        // e.g., struct S arr[2][2] where S has pointer fields.
                        // Recursively emit each outer element as a compound sub-array.
                        if let CType::Array(_, _) = elem_ty.as_ref() {
                            let outer_elem_size = self.resolve_ctype_size(elem_ty);
                            self.emit_multidim_array_compound(
                                elements, nested_items, elem_ty, *arr_size, outer_elem_size);
                            return;
                        }
                    }
                }

                // Nested struct/union: delegate to emit_sub_struct_to_compound
                // which handles compound-vs-bytes selection automatically.
                let field_ty_clone = field_ty.clone();
                if let Some(nested_layout) = self.get_struct_layout_for_ctype(&field_ty_clone) {
                    self.emit_sub_struct_to_compound(elements, nested_items, &nested_layout, field_size);
                    return;
                }

                // Non-struct field: serialize to bytes (arrays, scalars)
                let mut bytes = vec![0u8; field_size];
                if let CType::Array(ref inner_ty, Some(arr_size)) = field_ty_clone {
                    if inner_ty.is_complex() {
                        let elem_size = self.resolve_ctype_size(inner_ty);
                        self.fill_array_of_complex(nested_items, inner_ty, arr_size, elem_size, &mut bytes, 0);
                    } else if matches!(inner_ty.as_ref(), CType::Struct(_) | CType::Union(_)) {
                        // Array of structs/unions without pointer fields: use composite
                        // array filler which handles Initializer::List sub-items correctly.
                        let elem_size = self.resolve_ctype_size(inner_ty);
                        self.fill_array_of_composites(nested_items, inner_ty, arr_size, elem_size, &mut bytes, 0);
                    } else if let CType::Array(ref inner_elem, Some(inner_size)) = inner_ty.as_ref() {
                        // Multi-dimensional array (e.g., unsigned char hash[2][24]):
                        // each sub-item is a braced list for one element of the outer dimension.
                        let elem_size = self.resolve_ctype_size(inner_ty);
                        self.fill_multidim_array_field(nested_items, inner_elem, *inner_size, arr_size, elem_size, &mut bytes, 0);
                    } else {
                        self.fill_scalar_list_to_bytes(nested_items, inner_ty, field_size, &mut bytes);
                    }
                } else {
                    self.fill_scalar_list_to_bytes(nested_items, &field_ty_clone, field_size, &mut bytes);
                }
                push_bytes_as_elements(elements, &bytes);
            }
        }
    }

    /// Emit a multi-dimensional array field containing structs with pointer fields.
    /// Recursively emits each outer array element as a compound sub-init, handling
    /// designated initializers at each level.
    fn emit_multidim_array_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        items: &[InitializerItem],
        elem_ty: &CType,
        arr_size: usize,
        elem_size: usize,
    ) {
        // Build a sparse map: for each outer array index, store its initializer.
        let mut index_inits: Vec<Option<&Initializer>> = vec![None; arr_size];
        let mut ai = 0usize;
        for item in items {
            // Handle index designator: [idx] = val
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    ai = idx;
                }
            }
            if ai < arr_size {
                index_inits[ai] = Some(&item.init);
            }
            ai += 1;
        }

        // Emit each outer element
        for slot in &index_inits {
            if let Some(init) = slot {
                // Recursively emit each element using compound field init
                let is_ptr = matches!(elem_ty, CType::Pointer(_, _) | CType::Function(_));
                self.emit_compound_field_init(elements, init, elem_ty, elem_size, is_ptr);
            } else {
                // Uninitialized element - zero fill
                push_zero_bytes(elements, elem_size);
            }
        }
    }

    /// Emit all initializer items for a single struct/union field in compound mode.
    /// Handles the full dispatch: anonymous members, nested designators, flat array init,
    /// and single-expression fields. This is the shared logic between the union and
    /// non-union struct paths in `lower_struct_global_init_compound`.
    fn emit_field_inits_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        inits: &[&InitializerItem],
        field: &crate::common::types::StructFieldLayout,
        field_size: usize,
    ) {
        let field_is_pointer = matches!(field.ty, CType::Pointer(_, _) | CType::Function(_));

        if inits.is_empty() {
            push_zero_bytes(elements, field_size);
        } else if inits.len() == 1 {
            let item = inits[0];
            let desig_name = h::first_field_designator(item);
            let is_anon = h::is_anon_member_designator(
                desig_name, &field.name, &field.ty);

            if is_anon {
                let sub_item = InitializerItem {
                    designators: item.designators.clone(),
                    init: item.init.clone(),
                };
                let sub_layout = self.get_struct_layout_for_ctype(&field.ty)
                    .unwrap_or_else(StructLayout::empty_rc);
                self.emit_sub_struct_to_compound(elements, &[sub_item], &sub_layout, field_size);
            } else if h::has_nested_field_designator(item) {
                self.emit_compound_nested_designator_field(
                    elements, item, &field.ty, field_size);
            } else {
                let has_array_idx_designator = item.designators.iter().any(|d| matches!(d, Designator::Index(_)));
                if has_array_idx_designator {
                    if let CType::Array(elem_ty, Some(arr_size)) = &field.ty {
                        // Only use pointer-array designated init for direct pointer arrays.
                        // For arrays of structs/sub-arrays containing pointers, fall through
                        // to emit_compound_field_init which handles nested List initializers.
                        let is_direct_ptr = matches!(elem_ty.as_ref(), CType::Pointer(_, _) | CType::Function(_));
                        if is_direct_ptr {
                            self.emit_compound_ptr_array_designated_init(
                                elements, &[item], elem_ty, *arr_size);
                            return;
                        }
                    }
                }
                self.emit_compound_field_init(elements, &item.init, &field.ty, field_size, field_is_pointer);
            }
        } else {
            // Multiple items targeting the same outer struct field.
            // Check if they all have nested field designators (e.g., .mmu.f1, .mmu.f2, .mmu.f3)
            // targeting sub-fields of a struct/union field.
            let all_have_nested_desig = inits.iter().all(|item| h::has_nested_field_designator(item));
            let field_is_struct = matches!(&field.ty, CType::Struct(_) | CType::Union(_));

            let is_anon_multi = field.name.is_empty() && field_is_struct;
            if all_have_nested_desig && field_is_struct {
                // Strip the outer field designator from each item and delegate
                // to sub-struct compound init with the inner designators.
                let sub_items: Vec<InitializerItem> = inits.iter().map(|item| {
                    InitializerItem {
                        designators: item.designators[1..].to_vec(),
                        init: item.init.clone(),
                    }
                }).collect();
                let sub_layout = self.get_struct_layout_for_ctype(&field.ty)
                    .unwrap_or_else(StructLayout::empty_rc);
                self.emit_sub_struct_to_compound(elements, &sub_items, &sub_layout, field_size);
            } else if is_anon_multi {
                let sub_items: Vec<InitializerItem> = inits.iter().map(|item| {
                    InitializerItem {
                        designators: item.designators.clone(),
                        init: item.init.clone(),
                    }
                }).collect();
                let sub_layout = self.get_struct_layout_for_ctype(&field.ty)
                    .unwrap_or_else(StructLayout::empty_rc);
                self.emit_sub_struct_to_compound(elements, &sub_items, &sub_layout, field_size);
            } else if field_is_struct && inits.iter().all(|item| item.designators.is_empty()) {
                // Brace-elided sub-struct initialization: multiple flat items
                // (no designators) targeting a struct field. Convert them into
                // positional initializer items for the sub-struct.
                let sub_items: Vec<InitializerItem> = inits.iter().map(|item| {
                    InitializerItem {
                        designators: vec![],
                        init: item.init.clone(),
                    }
                }).collect();
                let sub_layout = self.get_struct_layout_for_ctype(&field.ty)
                    .unwrap_or_else(StructLayout::empty_rc);
                self.emit_sub_struct_to_compound(elements, &sub_items, &sub_layout, field_size);
            } else {
                self.emit_compound_flat_array_init(elements, inits, &field.ty, field_size);
            }
        }
    }
}
