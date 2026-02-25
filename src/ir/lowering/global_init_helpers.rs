//! Shared helper functions for initialization code.
//!
//! This module extracts common patterns that were duplicated across
//! `global_init.rs`, `global_init_bytes.rs`, `global_init_compound.rs`, and `stmt.rs`.
//! These include designator inspection, field resolution, anonymous member
//! drilling, and init item classification utilities.

use std::rc::Rc;
use crate::frontend::parser::ast::{
    Designator,
    Expr,
    Initializer,
    InitializerItem,
};
use crate::common::types::{CType, StructLayout, StructLayoutProvider, RcLayout};
use crate::common::fx_hash::FxHashMap;

/// Extract the field name from the first designator of an initializer item.
/// Returns `None` if the item has no designators or the first is not a Field.
pub(super) fn first_field_designator(item: &InitializerItem) -> Option<&str> {
    match item.designators.first() {
        Some(Designator::Field(ref name)) => Some(&**name),
        _ => None,
    }
}

/// Check if an initializer item has a nested field designator
/// (i.e., has 2+ designators with the first being a Field).
/// Used to detect patterns like `.field.subfield = val`.
pub(super) fn has_nested_field_designator(item: &InitializerItem) -> bool {
    item.designators.len() > 1
        && matches!(item.designators.first(), Some(Designator::Field(_)))
}

/// Check if a field is an anonymous struct/union member being targeted by a
/// field designator. This detects the case where a designator like `.x` resolves
/// to an anonymous member (empty name) that is itself a struct/union.
pub(super) fn is_anon_member_designator(
    desig_name: Option<&str>,
    field_name: &str,
    field_ty: &CType,
) -> bool {
    desig_name.is_some()
        && field_name.is_empty()
        && matches!(field_ty, CType::Struct(_) | CType::Union(_))
}

/// Check if any initializer items use the `[N].field` designator pattern
/// (e.g., `[0].name = "hello", [0].value = 42`).
pub(super) fn has_array_field_designators(items: &[InitializerItem]) -> bool {
    items.iter().any(|item| {
        item.designators.len() >= 2
            && matches!(item.designators[0], Designator::Index(_))
            && matches!(item.designators[1], Designator::Field(_))
    })
}

/// Check if an expression contains a string literal anywhere,
/// including through binary operations and casts (e.g., `"str" + N`).
pub(super) fn expr_contains_string_literal(expr: &Expr) -> bool {
    match expr {
        Expr::StringLiteral(_, _) | Expr::WideStringLiteral(_, _)
        | Expr::Char16StringLiteral(_, _) => true,
        Expr::BinaryOp(_, lhs, rhs, _) => {
            expr_contains_string_literal(lhs) || expr_contains_string_literal(rhs)
        }
        Expr::Cast(_, inner, _) => expr_contains_string_literal(inner),
        _ => false,
    }
}

/// Check if an initializer item contains a string literal anywhere
/// (including nested lists).
pub(super) fn init_contains_string_literal(item: &InitializerItem) -> bool {
    match &item.init {
        Initializer::Expr(expr) => expr_contains_string_literal(expr),
        Initializer::List(sub_items) => {
            sub_items.iter().any(init_contains_string_literal)
        }
    }
}

/// Check if an initializer item contains an address expression or string literal.
/// Used when `is_multidim_char_array` is true to suppress treating string
/// literals as address expressions for multi-dim char arrays.
/// `enum_constants` is used to distinguish enum constant identifiers (which are
/// compile-time integer values) from variable/function identifiers (which are addresses).
pub(super) fn init_contains_addr_expr(
    item: &InitializerItem,
    is_multidim_char_array: bool,
    enum_constants: &FxHashMap<Rc<str>, i64>,
) -> bool {
    match &item.init {
        Initializer::Expr(expr) => {
            if matches!(expr, Expr::StringLiteral(_, _)) {
                !is_multidim_char_array
            } else {
                expr_might_be_addr(expr, enum_constants)
            }
        }
        Initializer::List(sub_items) => {
            sub_items.iter().any(|sub| init_contains_addr_expr(sub, is_multidim_char_array, enum_constants))
        }
    }
}

/// Check if an expression might produce an address that requires a relocation.
/// This is used to determine whether a global initializer needs the Compound path.
/// Conservative: false positives are safe (just use the slower Compound path).
/// `enum_constants` is used to exclude known enum constant identifiers, which are
/// compile-time integer values and not addresses.
fn expr_might_be_addr(expr: &Expr, enum_constants: &FxHashMap<Rc<str>, i64>) -> bool {
    match expr {
        Expr::AddressOf(_, _) => true,
        Expr::LabelAddr(_, _) => true,
        // String literals produce addresses (pointers to .rodata entries), so they
        // require relocations in global initializers (e.g., "foo" + 1 in a static array).
        Expr::StringLiteral(_, _)
        | Expr::WideStringLiteral(_, _)
        | Expr::Char16StringLiteral(_, _) => true,
        // Identifiers that are enum constants are compile-time integer values, not addresses.
        // Only treat non-enum identifiers as potential addresses (array/function names).
        Expr::Identifier(name, _) => !enum_constants.contains_key(name),
        // Member access might be array-to-pointer decay (e.g., s.arr where arr is an array field)
        Expr::MemberAccess(_, _, _) => true,
        // Cast of an address expression
        Expr::Cast(_, inner, _) => expr_might_be_addr(inner, enum_constants),
        // Binary ops on addresses (e.g., &x + offset, arr + n)
        Expr::BinaryOp(_, lhs, rhs, _) => expr_might_be_addr(lhs, enum_constants) || expr_might_be_addr(rhs, enum_constants),
        // Compound literals may contain addresses
        Expr::CompoundLiteral(_, _, _) => true,
        _ => false,
    }
}

/// Check if a CType contains pointer elements (directly or through arrays/structs).
pub(super) fn type_has_pointer_elements(ty: &CType, ctx: &dyn StructLayoutProvider) -> bool {
    match ty {
        CType::Pointer(_, _) | CType::Function(_) => true,
        CType::Array(inner, _) => type_has_pointer_elements(inner, ctx),
        CType::Struct(key) | CType::Union(key) => {
            if let Some(layout) = ctx.get_struct_layout(key) {
                layout.fields.iter().any(|f| type_has_pointer_elements(&f.ty, ctx))
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Count the number of flat scalar initializer values needed to fully initialize
/// one instance of the given type. For scalars and pointers, this is 1. For arrays,
/// it's the array size times the count per element. For structs/unions, it's the
/// sum of counts across all fields (union: max of any field, but for flat init we
/// use the first/largest).
///
/// This is used to determine how many flat init items to consume when filling an
/// array-of-structs field without braces.
pub(super) fn count_flat_init_scalars(ty: &CType, ctx: &dyn StructLayoutProvider) -> usize {
    match ty {
        CType::Array(inner, Some(size)) => {
            size * count_flat_init_scalars(inner, ctx)
        }
        CType::Struct(key) | CType::Union(key) => {
            if let Some(layout) = ctx.get_struct_layout(key) {
                if layout.is_union {
                    // For unions, flat init fills the first field
                    if let Some(first) = layout.fields.first() {
                        count_flat_init_scalars(&first.ty, ctx)
                    } else {
                        1
                    }
                } else {
                    layout.fields.iter()
                        .filter(|f| f.bit_width != Some(0)) // skip zero-width bitfields
                        .map(|f| count_flat_init_scalars(&f.ty, ctx))
                        .sum::<usize>()
                        .max(1) // at least 1 for empty structs
                }
            } else {
                1
            }
        }
        _ => 1, // scalar, pointer, enum, etc.
    }
}

/// Append `count` zero bytes as `GlobalInit::Scalar(IrConst::I8(0))` to `elements`.
/// Used throughout global initialization for padding and zero-fill.
pub(super) fn push_zero_bytes(elements: &mut Vec<crate::ir::reexports::GlobalInit>, count: usize) {
    for _ in 0..count {
        elements.push(crate::ir::reexports::GlobalInit::Scalar(crate::ir::reexports::IrConst::I8(0)));
    }
}

/// Convert a byte buffer to GlobalInit elements by pushing each byte as I8.
/// This is the standard bytes-to-compound conversion used throughout the
/// compound initialization path when a sub-structure has no address fields.
pub(super) fn push_bytes_as_elements(elements: &mut Vec<crate::ir::reexports::GlobalInit>, bytes: &[u8]) {
    for &b in bytes {
        elements.push(crate::ir::reexports::GlobalInit::Scalar(crate::ir::reexports::IrConst::I8(b as i8)));
    }
}

/// Convert a byte buffer to GlobalInit elements for a string literal in a char array.
/// Pushes each character as I8, followed by padding zeros up to `field_size`.
/// Characters are mapped via `char as u8` (the internal string representation uses
/// chars in U+0000..U+00FF to represent raw byte values from C string literals).
pub(super) fn push_string_as_elements(elements: &mut Vec<crate::ir::reexports::GlobalInit>, s: &str, field_size: usize) {
    let s_chars: Vec<u8> = s.chars().map(|c| c as u8).collect();
    for (i, &b) in s_chars.iter().enumerate() {
        if i >= field_size { break; }
        elements.push(crate::ir::reexports::GlobalInit::Scalar(crate::ir::reexports::IrConst::I8(b as i8)));
    }
    // null terminator + remaining zero fill
    for _ in s_chars.len()..field_size {
        elements.push(crate::ir::reexports::GlobalInit::Scalar(crate::ir::reexports::IrConst::I8(0)));
    }
}

/// Result of resolving an anonymous member for struct initialization.
/// Contains the sub-layout, the byte offset of the anonymous member, and
/// a synthetic initializer item that re-targets the inner field.
pub(super) struct AnonMemberResolution {
    pub sub_layout: RcLayout,
    pub anon_offset: usize,
    pub sub_item: InitializerItem,
}

/// Resolve an anonymous member during struct initialization.
///
/// When a designator like `.x` targets a field inside an anonymous struct/union
/// member, this function looks up the sub-layout and creates a synthetic
/// initializer item that re-targets the inner field name.
///
/// `extra_designators` contains any remaining designators from the original item
/// beyond the first one (e.g., for `.base.cra_name`, extra_designators would be
/// `[Field("cra_name")]`). These are appended to preserve nested designator chains.
///
/// Returns `None` if the anonymous field's type is not a struct/union or
/// if the layout cannot be found.
pub(super) fn resolve_anonymous_member(
    layout: &StructLayout,
    anon_field_idx: usize,
    inner_name: &str,
    init: &Initializer,
    extra_designators: &[Designator],
    layouts: &crate::common::fx_hash::FxHashMap<Rc<str>, RcLayout>,
) -> Option<AnonMemberResolution> {
    let anon_field = &layout.fields[anon_field_idx];
    let anon_offset = anon_field.offset;
    let key = match &anon_field.ty {
        CType::Struct(key) | CType::Union(key) => key,
        _ => return None,
    };
    let sub_layout = layouts.get(key.as_ref())?.clone(); // Rc::clone, not deep clone
    let mut synth_desigs = vec![Designator::Field(Rc::from(inner_name))];
    synth_desigs.extend(extra_designators.iter().cloned());
    let sub_item = InitializerItem {
        designators: synth_desigs,
        init: init.clone(),
    };
    Some(AnonMemberResolution {
        sub_layout,
        anon_offset,
        sub_item,
    })
}
