//! Compound (relocation-aware) global initialization — shared helpers.
//!
//! This module contains helper methods for compound global initialization that
//! are used by both the struct/union path (`global_init_compound_struct`) and the
//! bytes+ptrs hybrid path (`global_init_compound_ptrs`). Includes:
//! - Nested designator field emission
//! - Pointer array init (braced and designated)
//! - Flat array init
//! - Expression-to-compound emission
//! - Pointer field resolution
//! - Struct layout lookup

use std::rc::Rc;
use crate::frontend::parser::ast::{
    Designator,
    Expr,
    Initializer,
    InitializerItem,
};
use crate::ir::reexports::{GlobalInit};
use crate::common::types::{IrType, StructLayout, CType};
use super::lower::Lowerer;
use super::global_init_helpers as h;
use h::{push_zero_bytes, push_bytes_as_elements};

impl Lowerer {
    /// Emit a field whose initializer has multi-level designators targeting a sub-field
    /// within this struct/union (e.g., .bs.keyword = {"STORE", -1}).
    ///
    /// Drills through the designator chain to find the target sub-field, then emits
    /// the field data using compound (relocation-aware) initialization for the target
    /// sub-field, with zero-fill for the rest of the outer field.
    pub(super) fn emit_compound_nested_designator_field(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        item: &InitializerItem,
        outer_ty: &CType,
        outer_size: usize,
    ) {
        let drill = match self.drill_designators(&item.designators[1..], outer_ty) {
            Some(d) => d,
            None => {
                push_zero_bytes(elements, outer_size);
                return;
            }
        };

        let sub_offset = drill.byte_offset;
        let current_ty = drill.target_ty;
        let sub_size = self.resolve_ctype_size(&current_ty);
        let sub_is_pointer = matches!(current_ty, CType::Pointer(_, _) | CType::Function(_));

        // Handle bitfield targets: pack the value into the storage unit bytes
        if let (Some(bo), Some(bw)) = (drill.bit_offset, drill.bit_width) {
            let ir_ty = IrType::from_ctype(&current_ty);
            let val = self.eval_init_scalar(&item.init);
            // Emit zero bytes before the storage unit
            push_zero_bytes(elements, sub_offset);
            // Create a small buffer for the storage unit, write bitfield, emit as bytes
            let storage_size = ir_ty.size();
            let mut buf = vec![0u8; storage_size];
            self.write_bitfield_to_bytes(&mut buf, 0, &val, ir_ty, bo, bw);
            push_bytes_as_elements(elements, &buf);
            // Zero-fill remaining
            let remaining = outer_size.saturating_sub(sub_offset + storage_size);
            push_zero_bytes(elements, remaining);
            return;
        }

        // Emit zero bytes before the sub-field
        push_zero_bytes(elements, sub_offset);

        // If the target is a struct/union with a list init, delegate to the
        // sub-struct emission helper which handles compound vs bytes selection.
        let handled = if let (CType::Struct(_) | CType::Union(_), Initializer::List(nested_items)) =
            (&current_ty, &item.init)
        {
            if let Some(target_layout) = self.get_struct_layout_for_ctype(&current_ty) {
                self.emit_sub_struct_to_compound(elements, nested_items, &target_layout, sub_size);
                true
            } else {
                false
            }
        } else {
            false
        };

        if !handled {
            self.emit_compound_field_init(elements, &item.init, &current_ty, sub_size, sub_is_pointer);
        }

        // Zero-fill the remaining outer field after the sub-field
        let remaining = outer_size.saturating_sub(sub_offset + sub_size);
        push_zero_bytes(elements, remaining);
    }

    /// Emit an array-of-pointers field from a braced initializer list.
    /// Each element is either a GlobalAddr (for string literals / address expressions)
    /// or zero-filled (for missing elements).
    ///
    /// Uses a sparse map to correctly handle designated initializers with
    /// out-of-order or backward-jumping indices (e.g., `[3]=&v3, [1]=&v1`).
    pub(super) fn emit_compound_ptr_array_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        items: &[InitializerItem],
        _elem_ty: &CType,
        arr_size: usize,
    ) {
        let ptr_size = crate::common::types::target_ptr_size();

        // Build a sparse map: for each array index, store the initializer (if any).
        // This correctly handles out-of-order designators like [3]=x, [1]=y.
        let mut index_inits: Vec<Option<&Initializer>> = vec![None; arr_size];
        let mut ai = 0usize;

        for item in items {
            if ai >= arr_size && item.designators.is_empty() {
                break;
            }

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

        // Emit each element in order
        for slot in &index_inits {
            if let Some(init) = slot {
                if let Initializer::Expr(ref expr) = init {
                    self.emit_expr_to_compound(elements, expr, ptr_size, None);
                } else {
                    // Nested list - zero fill this element
                    push_zero_bytes(elements, ptr_size);
                }
            } else {
                // Uninitialized element - zero fill
                push_zero_bytes(elements, ptr_size);
            }
        }
    }

    /// Emit a designated initializer for a pointer array field.
    /// Handles cases like `.a[1] = "abc"` where a is `char *a[3]`.
    /// The items may have field+index designators; we extract the index from the designators.
    pub(super) fn emit_compound_ptr_array_designated_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        items: &[&InitializerItem],
        _elem_ty: &CType,
        arr_size: usize,
    ) {
        let ptr_size = crate::common::types::target_ptr_size();
        // Build a sparse map of which indices are initialized
        let mut index_inits: Vec<Option<&Initializer>> = vec![None; arr_size];

        for item in items {
            // Find the Index designator (may be after a Field designator)
            let idx = item.designators.iter().find_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            }).unwrap_or(0);

            if idx < arr_size {
                index_inits[idx] = Some(&item.init);
            }
        }

        // Emit each element
        for ai in 0..arr_size {
            if let Some(init) = index_inits[ai] {
                if let Initializer::Expr(ref expr) = init {
                    self.emit_expr_to_compound(elements, expr, ptr_size, None);
                } else {
                    push_zero_bytes(elements, ptr_size);
                }
            } else {
                // Uninitialized element - zero fill
                push_zero_bytes(elements, ptr_size);
            }
        }
    }

    /// Emit a flat array init for a field that has multiple items (flat init style).
    /// E.g., struct { char *s[2]; } x = { "abc", "def" };
    /// Also handles arrays of structs with pointer fields, where each struct element
    /// consumes multiple flat init items.
    pub(super) fn emit_compound_flat_array_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        inits: &[&InitializerItem],
        field_ty: &CType,
        field_size: usize,
    ) {
        let (elem_ty, arr_size) = match field_ty {
            CType::Array(inner, Some(size)) => (inner.as_ref(), *size),
            _ => {
                // Not an array - just use first item
                if let Some(first) = inits.first() {
                    let field_is_pointer = matches!(field_ty, CType::Pointer(_, _));
                    self.emit_compound_field_init(elements, &first.init, field_ty, field_size, field_is_pointer);
                } else {
                    push_zero_bytes(elements, field_size);
                }
                return;
            }
        };

        let elem_size = self.resolve_ctype_size(elem_ty);

        // Check if the element type is a struct/union - if so, each element
        // consumes multiple flat init items (one per scalar leaf field).
        let elem_is_struct = matches!(elem_ty, CType::Struct(_) | CType::Union(_));
        if elem_is_struct {
            if let Some(elem_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                let scalars_per_elem = h::count_flat_init_scalars(elem_ty, &*self.types.borrow_struct_layouts());
                let mut item_pos = 0usize;
                let mut ai = 0usize;
                while ai < arr_size {
                    if item_pos >= inits.len() {
                        // Remaining elements are zero-initialized
                        push_zero_bytes(elements, elem_size * (arr_size - ai));
                        break;
                    }
                    // Collect scalars_per_elem items for this struct element,
                    // converting them to a flat InitializerItem list for the sub-struct.
                    let chunk_end = (item_pos + scalars_per_elem).min(inits.len());
                    let chunk: Vec<InitializerItem> = inits[item_pos..chunk_end]
                        .iter()
                        .map(|item| (*item).clone())
                        .collect();
                    self.emit_sub_struct_to_compound(elements, &chunk, &elem_layout, elem_size);
                    item_pos += scalars_per_elem;
                    ai += 1;
                }
                return;
            }
        }

        let elem_is_pointer = h::type_has_pointer_elements(elem_ty, &*self.types.borrow_struct_layouts());
        let ptr_size = crate::common::types::target_ptr_size();

        let mut ai = 0usize;
        for item in inits {
            if ai >= arr_size { break; }

            if let Initializer::Expr(ref expr) = item.init {
                if elem_is_pointer {
                    self.emit_expr_to_compound(elements, expr, ptr_size, None);
                } else if let Some(val) = self.eval_const_expr(expr) {
                    let elem_ir_ty = IrType::from_ctype(elem_ty);
                    let coerced = val.coerce_to(elem_ir_ty);
                    self.push_const_as_bytes(elements, &coerced, elem_size);
                } else {
                    push_zero_bytes(elements, elem_size);
                }
            } else {
                // Nested list - zero fill element
                push_zero_bytes(elements, elem_size);
            }
            ai += 1;
        }

        // Zero-fill remaining elements
        while ai < arr_size {
            push_zero_bytes(elements, elem_size);
            ai += 1;
        }
    }

    /// Resolve which struct field a positional or designated initializer targets.
    pub(super) fn resolve_struct_init_field_idx(
        &self,
        item: &InitializerItem,
        layout: &StructLayout,
        current_field_idx: usize,
    ) -> usize {
        let desig_name = h::first_field_designator(item);
        layout.resolve_init_field_idx(desig_name, current_field_idx, &*self.types.borrow_struct_layouts())
            .unwrap_or(current_field_idx)
    }

    /// Emit a single expression as a compound element.
    ///
    /// When `coerce_ty` is Some, tries const eval with type coercion first (for
    /// scalar fields with known types, including _Bool normalization). When None,
    /// tries address resolution first (for pointer fields and pointer array elements).
    ///
    /// Fallback chain: const eval / string literal -> string addr -> global addr -> zero.
    pub(super) fn emit_expr_to_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        expr: &Expr,
        element_size: usize,
        coerce_ty: Option<&CType>,
    ) {
        // For typed scalar fields, try const eval with coercion first
        if let Some(ty) = coerce_ty {
            if let Some(val) = self.eval_const_expr(expr) {
                let coerced = if *ty == CType::Bool {
                    val.bool_normalize()
                } else {
                    val.coerce_to(IrType::from_ctype(ty))
                };
                self.push_const_as_bytes(elements, &coerced, element_size);
                return;
            }
        }
        // For pointer/address contexts, try address resolution first
        if let Expr::LabelAddr(label_name, _) = expr {
            // GCC &&label extension: emit label address as GlobalAddr
            let scoped_label = self.get_or_create_user_label(label_name);
            // Record this block ID so CFG simplify keeps it reachable
            if let Some(ref mut fs) = self.func_state {
                fs.global_init_label_blocks.push(scoped_label);
            }
            elements.push(GlobalInit::GlobalAddr(Rc::from(scoped_label.as_label())));
        } else if let Expr::StringLiteral(s, _) = expr {
            let label = self.intern_string_literal(s);
            elements.push(GlobalInit::GlobalAddr(Rc::from(label)));
        } else if let Some(addr_init) = self.eval_string_literal_addr_expr(expr) {
            elements.push(addr_init);
        } else if let Some(addr_init) = self.eval_global_addr_expr(expr) {
            elements.push(addr_init);
        } else if coerce_ty.is_none() {
            // Untyped path: try raw const eval as last resort
            if let Some(val) = self.eval_const_expr(expr) {
                self.push_const_as_bytes(elements, &val, element_size);
            } else {
                push_zero_bytes(elements, element_size);
            }
        } else {
            push_zero_bytes(elements, element_size);
        }
    }

    /// Resolve a pointer field's initializer expression to a GlobalInit.
    /// Handles string literals, global addresses, function pointers, etc.
    pub(super) fn resolve_ptr_field_init(&mut self, expr: &Expr) -> Option<GlobalInit> {
        // GCC &&label extension: label address in a pointer field
        // Strip casts first since label addresses are often cast, e.g. (Label)&&I_noop
        if let Expr::LabelAddr(label_name, _) = Self::strip_casts(expr) {
            let scoped_label = self.get_or_create_user_label(label_name);
            // Record this block ID so CFG simplify keeps it reachable
            if let Some(ref mut fs) = self.func_state {
                fs.global_init_label_blocks.push(scoped_label);
            }
            return Some(GlobalInit::GlobalAddr(Rc::from(scoped_label.as_label())));
        }
        // String literal: create a .rodata entry and reference it
        if let Expr::StringLiteral(s, _) = expr {
            let label = self.intern_string_literal(s);
            return Some(GlobalInit::GlobalAddr(Rc::from(label)));
        }
        // String literal +/- offset: "str" + N
        if let Some(addr) = self.eval_string_literal_addr_expr(expr) {
            return Some(addr);
        }
        // &(compound_literal) at file scope: create anonymous global and return its address.
        // e.g., .ptr = &(struct in6_addr){ { { 0xfc } } } in a static struct array initializer.
        if let Expr::AddressOf(inner, _) = expr {
            if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = inner.as_ref() {
                return Some(self.create_compound_literal_global(cl_type_spec, cl_init));
            }
        }
        // Bare compound literal used as a pointer value (array-to-pointer decay).
        // e.g., .commands = (const u8 []){ ATA_CMD_ID_ATA, ATA_CMD_ID_ATAPI, 0 }
        // The array compound literal decays to a pointer to its first element.
        if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = expr {
            return Some(self.create_compound_literal_global(cl_type_spec, cl_init));
        }
        // Cast-wrapped compound literal used as a pointer value.
        // e.g., .arr = (char *)(unsigned char[]){ 0xFD, 0x01 }
        // Unwrap casts (arbitrary depth) to find the inner compound literal.
        {
            let stripped = Self::strip_casts(expr);
            if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = stripped {
                return Some(self.create_compound_literal_global(cl_type_spec, cl_init));
            }
            // (void*) &(CompoundLiteral) – cast wrapping address-of compound literal
            // e.g., (void*) &(PyABIInfo) { .major = 1, ... } in static struct initializers
            if let Expr::AddressOf(inner, _) = stripped {
                if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = inner.as_ref() {
                    return Some(self.create_compound_literal_global(cl_type_spec, cl_init));
                }
            }
        }
        // Try as a global address expression (&x, func name, array name, etc.)
        if let Some(addr) = self.eval_global_addr_expr(expr) {
            return Some(addr);
        }
        // Integer constant 0 -> null pointer
        if let Some(val) = self.eval_const_expr(expr) {
            if let Some(v) = self.const_to_i64(&val) {
                if v == 0 {
                    return None; // Will be zero in the byte buffer
                }
            }
            // Non-zero constant pointer (unusual but possible)
            return Some(GlobalInit::Scalar(val));
        }
        None
    }

    /// Get a StructLayout for a CType if it's a struct or union.
    pub(super) fn get_struct_layout_for_ctype(&self, ty: &CType) -> Option<crate::common::types::RcLayout> {
        match ty {
            CType::Struct(key) | CType::Union(key) => {
                self.types.borrow_struct_layouts().get(&**key).cloned()
            }
            _ => None,
        }
    }

}
