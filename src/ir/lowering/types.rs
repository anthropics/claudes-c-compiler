//! Type resolution, predicates, size/alignment computation, and declaration
//! layout for the lowerer.
//!
//! Builtin seeding lives in types_seed.rs; CType conversion and the
//! TypeConvertContext trait impl live in types_ctype.rs.

use std::rc::Rc;
use crate::frontend::parser::ast::{
    DerivedDeclarator,
    Expr,
    StructFieldDecl,
    TypeSpecifier,
};
use crate::common::types::{IrType, StructField, StructLayout, RcLayout, CType};
use super::lower::Lowerer;

impl Lowerer {

    /// Resolve a TypeSpecifier, following TypeofType wrappers.
    /// TypedefName resolution now goes through CType (see type_spec_to_ctype).
    /// This only resolves non-typedef wrappers like TypeofType.
    pub(super) fn resolve_type_spec<'a>(&'a self, ts: &'a TypeSpecifier) -> &'a TypeSpecifier {
        let mut current = ts;
        for _ in 0..32 {
            if let TypeSpecifier::TypeofType(inner) = current {
                current = inner;
                continue;
            }
            break;
        }
        current
    }

    /// Resolve a TypeSpecifier to its underlying CType for typedef/typeof,
    /// returning None for non-typedef/typeof specifiers.
    /// This is a lightweight lookup (no array size evaluation or function pointer building).
    fn resolve_typedef_ctype(&self, ts: &TypeSpecifier) -> Option<CType> {
        match ts {
            TypeSpecifier::TypedefName(name) => self.types.typedefs.get(name).cloned(),
            TypeSpecifier::TypeofType(inner) => self.resolve_typedef_ctype(inner),
            TypeSpecifier::Typeof(expr) => self.get_expr_ctype(expr),
            _ => None,
        }
    }

    /// Check if a TypeSpecifier resolves to a Bool type (through typedefs).
    pub(super) fn is_type_bool(&self, ts: &TypeSpecifier) -> bool {
        matches!(ts, TypeSpecifier::Bool)
            || self.resolve_typedef_ctype(ts).is_some_and(|ct| matches!(ct, CType::Bool))
    }

    /// Check if a TypeSpecifier resolves to a struct or union type (through typedefs).
    pub(super) fn is_type_struct_or_union(&self, ts: &TypeSpecifier) -> bool {
        matches!(ts, TypeSpecifier::Struct(..) | TypeSpecifier::Union(..))
            || self.resolve_typedef_ctype(ts).is_some_and(|ct| ct.is_struct_or_union())
    }

    /// Check if a TypeSpecifier is a transparent union (passed as first member for ABI).
    pub(super) fn is_transparent_union(&self, ts: &TypeSpecifier) -> bool {
        let key = match ts {
            TypeSpecifier::Union(tag, _, _, _, _) => {
                tag.as_ref().map(|t| -> Rc<str> { format!("union.{}", t).into() })
            }
            TypeSpecifier::TypedefName(name) => {
                if let Some(CType::Union(key)) = self.types.typedefs.get(name) {
                    Some(key.clone())
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(key) = key {
            self.types.borrow_struct_layouts().get(&*key).is_some_and(|l| l.is_transparent_union)
        } else {
            false
        }
    }

    /// Check if a TypeSpecifier resolves to a complex type (through typedefs).
    pub(super) fn is_type_complex(&self, ts: &TypeSpecifier) -> bool {
        matches!(ts, TypeSpecifier::ComplexFloat | TypeSpecifier::ComplexDouble | TypeSpecifier::ComplexLongDouble)
            || self.resolve_typedef_ctype(ts).is_some_and(|ct| ct.is_complex())
    }

    /// Check if a TypeSpecifier resolves to a pointer type (through typedefs).
    pub(super) fn is_type_pointer(&self, ts: &TypeSpecifier) -> bool {
        matches!(ts, TypeSpecifier::Pointer(_, _))
            || self.resolve_typedef_ctype(ts).is_some_and(|ct| matches!(ct, CType::Pointer(_, _)))
    }

    /// Resolve typeof(expr) to a concrete TypeSpecifier by analyzing the expression type.
    /// Returns a new TypeSpecifier if the input is Typeof, otherwise returns a clone of the input.
    /// Note: TypedefName resolution now goes through CType, so typeof on a typedef
    /// is handled by resolving the typedef to CType first.
    pub(super) fn resolve_typeof(&self, ts: &TypeSpecifier) -> TypeSpecifier {
        match ts {
            TypeSpecifier::Typeof(expr) => {
                // For _Generic selections inside typeof(), resolve fresh to avoid stale cache.
                let ctype = if let Expr::GenericSelection(controlling, associations, _) = expr.as_ref() {
                    self.resolve_generic_selection_ctype(controlling, associations)
                } else {
                    self.get_expr_ctype(expr)
                };
                if let Some(ctype) = ctype {
                    Self::ctype_to_type_spec(&ctype)
                } else {
                    self.emit_warning(
                        "could not resolve type of 'typeof' expression; defaulting to 'int'",
                        expr.span(),
                    );
                    TypeSpecifier::Int // fallback
                }
            }
            TypeSpecifier::TypeofType(inner) => {
                self.resolve_typeof(inner)
            }
            TypeSpecifier::TypedefName(name) => {
                // Typedefs now store CType. Convert back to TypeSpecifier for
                // code that still needs a TypeSpecifier (e.g., typeof resolution).
                if let Some(ctype) = self.types.typedefs.get(name) {
                    Self::ctype_to_type_spec(ctype)
                } else {
                    ts.clone()
                }
            }
            other => other.clone(),
        }
    }

    /// Evaluate __builtin_types_compatible_p(type1, type2).
    /// Returns 1 if the unqualified types are compatible (same type after resolving
    /// typedefs and typeof), 0 otherwise. Follows GCC semantics: ignores top-level
    /// qualifiers, resolves typedefs, but considers signed/unsigned as distinct.
    pub(super) fn eval_types_compatible(&self, type1: &TypeSpecifier, type2: &TypeSpecifier) -> i32 {
        let ctype1 = self.type_spec_to_ctype(type1);
        let ctype2 = self.type_spec_to_ctype(type2);
        // Strip top-level qualifiers (CType doesn't carry qualifiers, so this is already done).
        // Compare the resolved CTypes. GCC considers enum types as their underlying int type,
        // and considers long/int as distinct even if same size on the platform.
        if Self::ctypes_compatible(&ctype1, &ctype2) { 1 } else { 0 }
    }

    /// Check if two CTypes are compatible for __builtin_types_compatible_p purposes.
    /// This is structural equality with special handling for:
    /// - Arrays: compatible if element types match (ignore size for unsized arrays)
    /// - Pointers: compatible if pointee types are compatible
    /// - Enums: treated as compatible with int
    fn ctypes_compatible(a: &CType, b: &CType) -> bool {
        // Normalize enum to int for compatibility purposes
        let a_norm = match a { CType::Enum(_) => &CType::Int, other => other };
        let b_norm = match b { CType::Enum(_) => &CType::Int, other => other };

        match (a_norm, b_norm) {
            // Pointers: pointee types must be compatible
            (CType::Pointer(p1, _), CType::Pointer(p2, _)) => Self::ctypes_compatible(p1, p2),
            // Arrays: element types must be compatible, sizes must match (or both unsized)
            (CType::Array(e1, s1), CType::Array(e2, s2)) => {
                Self::ctypes_compatible(e1, e2) && s1 == s2
            }
            // Structs/Unions: use derived PartialEq (compares name + fields)
            (CType::Struct(s1), CType::Struct(s2)) => s1 == s2,
            (CType::Union(u1), CType::Union(u2)) => u1 == u2,
            // Function types
            (CType::Function(f1), CType::Function(f2)) => f1 == f2,
            // All other types (scalars, void, bool, etc.): direct enum equality
            _ => a_norm == b_norm,
        }
    }

    pub(super) fn type_spec_to_ir(&self, ts: &TypeSpecifier) -> IrType {
        use crate::common::types::target_is_32bit;
        let is_32bit = target_is_32bit();
        match ts {
            TypeSpecifier::Void => IrType::Void,
            TypeSpecifier::Bool => IrType::U8,
            TypeSpecifier::Char => IrType::I8,
            TypeSpecifier::UnsignedChar => IrType::U8,
            TypeSpecifier::Short => IrType::I16,
            TypeSpecifier::UnsignedShort => IrType::U16,
            TypeSpecifier::Int | TypeSpecifier::Signed => IrType::I32,
            TypeSpecifier::UnsignedInt | TypeSpecifier::Unsigned => IrType::U32,
            TypeSpecifier::Long => if is_32bit { IrType::I32 } else { IrType::I64 },
            TypeSpecifier::UnsignedLong => if is_32bit { IrType::U32 } else { IrType::U64 },
            TypeSpecifier::LongLong => IrType::I64,
            TypeSpecifier::UnsignedLongLong => IrType::U64,
            TypeSpecifier::Int128 => IrType::I128,
            TypeSpecifier::UnsignedInt128 => IrType::U128,
            TypeSpecifier::Float => IrType::F32,
            TypeSpecifier::Double => IrType::F64,
            TypeSpecifier::LongDouble => IrType::F128,
            TypeSpecifier::ComplexFloat | TypeSpecifier::ComplexDouble | TypeSpecifier::ComplexLongDouble => IrType::Ptr,
            TypeSpecifier::Pointer(_, _) => IrType::Ptr,
            TypeSpecifier::Array(_, _) => IrType::Ptr,
            TypeSpecifier::Struct(..) | TypeSpecifier::Union(..) => IrType::Ptr,
            TypeSpecifier::Enum(_, _, false) => {
                // Non-packed enum: resolve to CType to get correct size and signedness.
                // Enums with values like `1U << 31` (0x80000000) are unsigned 32-bit.
                let ctype = self.type_spec_to_ctype(ts);
                IrType::from_ctype(&ctype)
            }
            TypeSpecifier::Enum(_, _, true) => {
                // Packed enum: resolve to CType to get the correct IR type
                let ctype = self.type_spec_to_ctype(ts);
                IrType::from_ctype(&ctype)
            }
            TypeSpecifier::TypedefName(name) => {
                // Resolve typedef through CType
                if let Some(ctype) = self.types.typedefs.get(name) {
                    IrType::from_ctype(ctype)
                } else if is_32bit { IrType::I32 } else { IrType::I64 }
            }
            TypeSpecifier::Typeof(expr) => {
                // For _Generic selections inside typeof(), resolve fresh to avoid stale cache.
                let ctype = if let Expr::GenericSelection(controlling, associations, _) = expr.as_ref() {
                    self.resolve_generic_selection_ctype(controlling, associations)
                } else {
                    self.get_expr_ctype(expr)
                };
                if let Some(ctype) = ctype {
                    IrType::from_ctype(&ctype)
                } else if is_32bit { IrType::I32 } else { IrType::I64 }
            }
            TypeSpecifier::TypeofType(inner) => self.type_spec_to_ir(inner),
            TypeSpecifier::FunctionPointer(_, _, _) => IrType::Ptr, // function pointer is a pointer
            TypeSpecifier::BareFunction(_, _, _) => IrType::Ptr, // bare function type decays to pointer
            // AutoType should be resolved before reaching here (in lower_local_decl)
            TypeSpecifier::AutoType => if is_32bit { IrType::I32 } else { IrType::I64 },
            // Vector type: return the element IR type (used for per-element operations)
            TypeSpecifier::Vector(inner, _) => self.type_spec_to_ir(inner),
        }
    }

    /// Get the (size, alignment) for a scalar type specifier. Returns None for
    /// compound types (arrays, structs, unions) that need recursive computation.
    fn scalar_type_size_align(ts: &TypeSpecifier) -> Option<(usize, usize)> {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        // On i686 (ILP32): long=4, pointer=4, double aligned to 4, long double=12 aligned to 4
        match ts {
            TypeSpecifier::Void | TypeSpecifier::Bool => Some((1, 1)),
            TypeSpecifier::Char | TypeSpecifier::UnsignedChar => Some((1, 1)),
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => Some((2, 2)),
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Signed | TypeSpecifier::Unsigned => Some((4, 4)),
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong => Some((ptr_sz, ptr_sz)),
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong => {
                // On i686, long long is 8 bytes but aligned to 4
                let align = if ptr_sz == 4 { 4 } else { 8 };
                Some((8, align))
            }
            TypeSpecifier::Int128 | TypeSpecifier::UnsignedInt128 => Some((16, 16)),
            TypeSpecifier::Float => Some((4, 4)),
            TypeSpecifier::Double => {
                let align = if ptr_sz == 4 { 4 } else { 8 };
                Some((8, align))
            }
            TypeSpecifier::LongDouble => {
                if ptr_sz == 4 { Some((12, 4)) } else { Some((16, 16)) }
            }
            TypeSpecifier::ComplexFloat => Some((8, 4)),
            TypeSpecifier::ComplexDouble => {
                let align = if ptr_sz == 4 { 4 } else { 8 };
                Some((16, align))
            }
            TypeSpecifier::ComplexLongDouble => {
                if ptr_sz == 4 { Some((24, 4)) } else { Some((32, 16)) }
            }
            TypeSpecifier::Pointer(_, _) => Some((ptr_sz, ptr_sz)),
            TypeSpecifier::Enum(_, _, false) => Some((4, 4)),
            TypeSpecifier::Enum(_, _, true) => {
                // Packed enums need type context to resolve; let caller handle via CType path
                None
            }
            TypeSpecifier::TypedefName(_) => Some((ptr_sz, ptr_sz)), // fallback for unresolved typedefs
            _ => None,
        }
    }

    /// Look up a struct/union layout by tag name, returning a cheap Rc clone.
    fn get_struct_union_layout_by_tag(&self, kind: &str, tag: &str) -> Option<RcLayout> {
        let key = format!("{}.{}", kind, tag);
        self.types.borrow_struct_layouts().get(key.as_str()).cloned()
    }

    /// Get the struct/union layout for a resolved TypeSpecifier.
    /// Handles both inline field definitions and tag-only forward references.
    /// Returns an Rc<StructLayout> for cheap cloning.
    fn struct_union_layout(&self, ts: &TypeSpecifier) -> Option<RcLayout> {
        match ts {
            TypeSpecifier::Struct(tag, Some(fields), is_packed, pragma_pack, _) => {
                // Use cached layout for tagged structs
                if let Some(tag) = tag {
                    let key = format!("struct.{}", tag);
                    if let Some(layout) = self.types.borrow_struct_layouts().get(key.as_str()) {
                        return Some(layout.clone());
                    }
                }
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                Some(Rc::new(self.compute_struct_union_layout_packed(fields, false, max_field_align)))
            }
            TypeSpecifier::Union(tag, Some(fields), is_packed, pragma_pack, _) => {
                // Use cached layout for tagged unions
                if let Some(tag) = tag {
                    let key = format!("union.{}", tag);
                    if let Some(layout) = self.types.borrow_struct_layouts().get(key.as_str()) {
                        return Some(layout.clone());
                    }
                }
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                Some(Rc::new(self.compute_struct_union_layout_packed(fields, true, max_field_align)))
            }
            TypeSpecifier::Struct(Some(tag), None, _, _, _) =>
                self.get_struct_union_layout_by_tag("struct", tag),
            TypeSpecifier::Union(Some(tag), None, _, _, _) =>
                self.get_struct_union_layout_by_tag("union", tag),
            _ => None,
        }
    }

    pub(super) fn compute_struct_union_layout_packed(&self, fields: &[StructFieldDecl], is_union: bool, max_field_align: Option<usize>) -> StructLayout {
        let struct_fields: Vec<StructField> = fields.iter().map(|f| {
            let bit_width = f.bit_width.as_ref().and_then(|bw| {
                self.eval_const_expr(bw).and_then(|c| c.to_u32())
            });
            let mut ty = self.struct_field_ctype(f);
            // GCC treats enum bitfields as unsigned (see struct_or_union_to_ctype).
            // Check both direct enum type specs and typedef'd enum types.
            if bit_width.is_some()
                && self.is_enum_type_spec(&f.type_spec)
                    && ty == CType::Int {
                        ty = CType::UInt;
                    }
            // Merge per-field alignment with typedef alignment.
            // If the field's type is a typedef with __aligned__, that alignment
            // must be applied even when the field itself has no explicit alignment.
            let field_alignment = {
                let mut align = f.alignment;
                if let Some(&ta) = self.typedef_alignment_for_type_spec(&f.type_spec) {
                    align = Some(align.map_or(ta, |a| a.max(ta)));
                }
                align
            };
            StructField {
                name: f.name.clone().unwrap_or_default(),
                ty,
                bit_width,
                alignment: field_alignment,
                is_packed: f.is_packed,
            }
        }).collect();
        if is_union {
            StructLayout::for_union_with_packing(&struct_fields, max_field_align, &*self.types.borrow_struct_layouts())
        } else {
            StructLayout::for_struct_with_packing(&struct_fields, max_field_align, &*self.types.borrow_struct_layouts())
        }
    }

    pub(super) fn sizeof_type(&self, ts: &TypeSpecifier) -> usize {
        // Handle TypedefName through CType
        if let TypeSpecifier::TypedefName(name) = ts {
            if let Some(ctype) = self.types.typedefs.get(name) {
                return ctype.size_ctx(&*self.types.borrow_struct_layouts());
            }
            return crate::common::types::target_ptr_size(); // fallback
        }
        // Handle packed enums (explicit or forward-reference to packed) via CType resolution
        if let TypeSpecifier::Enum(name, _, is_packed) = ts {
            let effective_packed = *is_packed || name.as_ref()
                .and_then(|n| self.types.packed_enum_types.get(n))
                .is_some();
            if effective_packed {
                let ctype = self.type_spec_to_ctype(ts);
                return ctype.size_ctx(&*self.types.borrow_struct_layouts());
            }
        }
        // Handle typeof(expr) by resolving the expression's type
        if let TypeSpecifier::Typeof(expr) = ts {
            if let Some(ctype) = self.get_expr_ctype(expr) {
                return ctype.size_ctx(&*self.types.borrow_struct_layouts());
            }
            return crate::common::types::target_ptr_size(); // fallback
        }
        // Handle typeof(type-name) by recursing on the inner type specifier
        if let TypeSpecifier::TypeofType(inner) = ts {
            return self.sizeof_type(inner);
        }
        // Handle vector types: total_bytes is the vector size
        if let TypeSpecifier::Vector(_, total_bytes) = ts {
            return *total_bytes;
        }
        let ts = self.resolve_type_spec(ts);
        if let Some((size, _)) = Self::scalar_type_size_align(ts) {
            return size;
        }
        if let TypeSpecifier::Array(elem, Some(size_expr)) = ts {
            let elem_size = self.sizeof_type(elem);
            return self.expr_as_array_size(size_expr)
                .map(|n| elem_size * n as usize)
                .unwrap_or(elem_size);
        }
        self.struct_union_layout(ts).map(|l| l.size).unwrap_or(crate::common::types::target_ptr_size())
    }

    /// Compute the alignment of a type in bytes (_Alignof).
    pub(super) fn alignof_type(&self, ts: &TypeSpecifier) -> usize {
        // Handle TypedefName through CType, incorporating typedef alignment overrides
        if let TypeSpecifier::TypedefName(name) = ts {
            let natural = if let Some(ctype) = self.types.typedefs.get(name) {
                self.ctype_align(ctype)
            } else {
                crate::common::types::target_ptr_size() // fallback
            };
            // If the typedef has an __aligned__ override, take the max
            if let Some(&td_align) = self.types.typedef_alignments.get(name) {
                return natural.max(td_align);
            }
            return natural;
        }
        // Handle typeof(expr) by resolving the expression's type
        if let TypeSpecifier::Typeof(expr) = ts {
            if let Some(ctype) = self.get_expr_ctype(expr) {
                return self.ctype_align(&ctype);
            }
            return crate::common::types::target_ptr_size(); // fallback
        }
        // Handle typeof(type-name) by recursing on the inner type specifier
        if let TypeSpecifier::TypeofType(inner) = ts {
            return self.alignof_type(inner);
        }
        // Handle vector types: alignment equals total vector size (power-of-2 aligned)
        if let TypeSpecifier::Vector(_, total_bytes) = ts {
            // Vector alignment is typically the total vector size, capped at 16
            return (*total_bytes).min(16);
        }
        let ts = self.resolve_type_spec(ts);
        if let Some((_, align)) = Self::scalar_type_size_align(ts) {
            return align;
        }
        if let TypeSpecifier::Array(elem, _) = ts {
            return self.alignof_type(elem);
        }
        self.struct_union_layout(ts).map(|l| l.align).unwrap_or(crate::common::types::target_ptr_size())
    }

    /// Compute preferred (natural) alignment of a type in bytes (__alignof__).
    /// On i686: __alignof__(long long) == 8, __alignof__(double) == 8.
    pub(super) fn preferred_alignof_type(&self, ts: &TypeSpecifier) -> usize {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        if ptr_sz != 4 {
            return self.alignof_type(ts);
        }
        // Handle TypedefName through CType preferred alignment
        if let TypeSpecifier::TypedefName(name) = ts {
            let natural = if let Some(ctype) = self.types.typedefs.get(name) {
                ctype.preferred_align_ctx(&*self.types.borrow_struct_layouts())
            } else {
                target_ptr_size()
            };
            if let Some(&td_align) = self.types.typedef_alignments.get(name) {
                return natural.max(td_align);
            }
            return natural;
        }
        if let TypeSpecifier::Typeof(expr) = ts {
            if let Some(ctype) = self.get_expr_ctype(expr) {
                return ctype.preferred_align_ctx(&*self.types.borrow_struct_layouts());
            }
            return target_ptr_size();
        }
        if let TypeSpecifier::TypeofType(inner) = ts {
            return self.preferred_alignof_type(inner);
        }
        if let TypeSpecifier::Vector(_, total_bytes) = ts {
            return (*total_bytes).min(16);
        }
        let ts = self.resolve_type_spec(ts);
        // On i686, check if scalar type has preferred alignment different from ABI
        match ts {
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong
            | TypeSpecifier::Double => return 8,
            TypeSpecifier::ComplexDouble => return 8,
            _ => {}
        }
        // Fall back to normal alignof for all other types
        if let Some((_, align)) = Self::scalar_type_size_align(ts) {
            return align;
        }
        if let TypeSpecifier::Array(elem, _) = ts {
            return self.preferred_alignof_type(elem);
        }
        self.struct_union_layout(ts).map(|l| l.align).unwrap_or(target_ptr_size())
    }

    /// Return the typedef alignment override for a type specifier, if any.
    /// For `TypeSpecifier::TypedefName("foo")`, looks up `foo` in `typedef_alignments`.
    pub(super) fn typedef_alignment_for_type_spec(&self, ts: &TypeSpecifier) -> Option<&usize> {
        if let TypeSpecifier::TypedefName(name) = ts {
            self.types.typedef_alignments.get(name)
        } else {
            None
        }
    }

    /// Collect array dimensions from derived declarators.
    /// Returns None for unsized dimensions (e.g., `int arr[]`).
    fn collect_derived_array_dims(&self, derived: &[DerivedDeclarator]) -> Vec<Option<usize>> {
        derived.iter().filter_map(|d| {
            if let DerivedDeclarator::Array(size_expr) = d {
                Some(size_expr.as_ref().and_then(|e| self.expr_as_array_size(e).map(|n| n as usize)))
            } else {
                None
            }
        }).collect()
    }

    /// Compute strides from an array of dimension sizes and a base element size.
    /// stride[i] = product(dims[i+1..]) * base_elem_size.
    /// E.g., dims=[3,4], base=4 -> strides=[16, 4].
    fn compute_strides_from_dims(dims: &[usize], base_elem_size: usize) -> Vec<usize> {
        let mut strides = Vec::with_capacity(dims.len());
        for i in 0..dims.len() {
            let stride: usize = dims[i+1..].iter().product::<usize>().max(1) * base_elem_size;
            strides.push(stride);
        }
        strides
    }

    /// For a pointer-to-array parameter type (e.g., Pointer(Array(Array(Int, 4), 3))),
    /// compute the array dimension strides for multi-dimensional subscript access.
    /// Returns strides for depth 0, 1, 2, ... where depth 0 is the outermost subscript.
    /// E.g., for int (*arr)[3][4]: strides = [3*4*4=48, 4*4=16, 4]
    /// For int (*arr)[3]: strides = [3*4=12, 4]
    pub(super) fn compute_ptr_array_strides(&self, type_spec: &TypeSpecifier) -> Vec<usize> {
        let ts = self.resolve_type_spec(type_spec);
        if let TypeSpecifier::Pointer(inner, _) = ts {
            // Collect dimensions from nested Array types
            let mut dims: Vec<usize> = Vec::new();
            let mut current = inner;
            loop {
                let resolved = self.resolve_type_spec(current);
                if let TypeSpecifier::Array(elem, size_expr) = resolved {
                    let n = size_expr.as_ref().and_then(|e| self.expr_as_array_size(e)).unwrap_or(1);
                    dims.push(n as usize);
                    current = elem;
                } else {
                    break;
                }
            }
            if dims.is_empty() {
                return vec![];
            }
            // Compute base element size (the innermost non-array type)
            let base_elem_size = self.sizeof_type(current);
            // Strides: [full_size, stride_for_dim_1, ..., base_elem_size]
            // full_size = product(all_dims) * base, then per-dim strides
            let full_size: usize = dims.iter().product::<usize>() * base_elem_size;
            let mut strides = vec![full_size];
            strides.extend(Self::compute_strides_from_dims(&dims, base_elem_size));
            strides
        } else {
            // Fall back to CType for typedef'd pointer-to-array types
            let ctype = self.type_spec_to_ctype(type_spec);
            if let CType::Pointer(ref inner_ct, _) = ctype {
                let mut dims: Vec<usize> = Vec::new();
                let mut current_ct = inner_ct.as_ref();
                while let CType::Array(elem_ct, size) = current_ct {
                    dims.push(size.unwrap_or(1));
                    current_ct = elem_ct.as_ref();
                }
                if dims.is_empty() {
                    return vec![];
                }
                let base_elem_size = current_ct.size_ctx(&*self.types.borrow_struct_layouts()).max(1);
                let full_size: usize = dims.iter().product::<usize>() * base_elem_size;
                let mut strides = vec![full_size];
                strides.extend(Self::compute_strides_from_dims(&dims, base_elem_size));
                strides
            } else {
                vec![]
            }
        }
    }

    /// Compute allocation info for a declaration.
    /// Returns (alloc_size, elem_size, is_array, is_pointer, array_dim_strides).
    /// For multi-dimensional arrays like int a[2][3], array_dim_strides = [12, 4]
    /// (stride for dim 0 = 3*4=12, stride for dim 1 = 4).
    pub(super) fn compute_decl_info(&self, ts: &TypeSpecifier, derived: &[DerivedDeclarator]) -> (usize, usize, bool, bool, Vec<usize>) {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        let ts = self.resolve_type_spec(ts);
        // Resolve the type spec through CType for typedef detection
        let resolved_ctype = self.type_spec_to_ctype(ts);
        // Check for pointer declarators (from derived or from the resolved type itself)
        let has_pointer = derived.iter().any(|d| matches!(d, DerivedDeclarator::Pointer))
            || matches!(ts, TypeSpecifier::Pointer(_, _))
            || matches!(resolved_ctype, CType::Pointer(_, _));

        let has_array = derived.iter().any(|d| matches!(d, DerivedDeclarator::Array(_)))
            || matches!(resolved_ctype, CType::Array(_, _));

        // Handle pointer and array combinations
        if has_pointer && !has_array {
            // Simple pointer: int *p, or typedef'd pointer (e.g., typedef struct Foo *FooPtr)
            let ptr_count = derived.iter().filter(|d| matches!(d, DerivedDeclarator::Pointer)).count();
            let elem_size = if let TypeSpecifier::Pointer(inner, _) = ts {
                if ptr_count >= 1 {
                    ptr_sz
                } else {
                    self.sizeof_type(inner)
                }
            } else if let CType::Pointer(ref inner_ct, _) = resolved_ctype {
                // Pointer from typedef resolution
                if ptr_count >= 1 {
                    ptr_sz
                } else {
                    inner_ct.size_ctx(&*self.types.borrow_struct_layouts())
                }
            } else if ptr_count >= 2 {
                ptr_sz
            } else {
                self.sizeof_type(ts)
            };
            return (ptr_sz, elem_size, false, true, vec![]);
        }
        if has_pointer && has_array {
            // Determine whether this is "array of pointers" or "pointer to array"
            // by looking at the LAST element of the derived list (outermost type wrapper).
            // The last element determines what the overall declaration IS:
            // - Last is Array: it's an array (of pointers to something)
            //   e.g., int *arr[3] -> derived=[Pointer, Array(3)] -> array of ptrs
            //   e.g., int (*ptrs[3])[4] -> derived=[Array(4), Pointer, Array(3)] -> array of ptrs-to-arrays
            // - Last is Pointer: it's a pointer (to an array)
            //   e.g., int (*p)[5] -> derived=[Array(5), Pointer] -> pointer to array
            let has_func_ptr = derived.iter().any(|d| matches!(d,
                DerivedDeclarator::FunctionPointer(_, _) | DerivedDeclarator::Function(_, _)));

            // If pointer is from resolved type spec (not in derived), and array is in derived,
            // this is an array of typedef'd pointers
            let ptr_pos = derived.iter().position(|d| matches!(d, DerivedDeclarator::Pointer));
            let pointer_from_type_spec = ptr_pos.is_none() && (matches!(ts, TypeSpecifier::Pointer(_, _)) || matches!(resolved_ctype, CType::Pointer(_, _)));

            // Check if the outermost (last) derived element is an Array
            let last_is_array = matches!(derived.last(), Some(DerivedDeclarator::Array(_)));

            if has_func_ptr || pointer_from_type_spec || last_is_array {
                // Array of pointers (or array of pointers-to-arrays, etc.)
                // Each element is a pointer (8 bytes).
                // Collect the variable's own array dimensions:
                // - For regular pointer arrays like int *arr[3] (derived=[Pointer, Array(3)]):
                //   Array dims AFTER the last Pointer are the variable's dimensions.
                // - For function pointer arrays like int (*ops[3])(int,int)
                //   (derived=[Array(3), Pointer, FunctionPointer(...)]):
                //   Array dims BEFORE the Pointer are the variable's dimensions,
                //   because the Pointer+FunctionPointer group describes the element type.
                let last_ptr_pos = derived.iter().rposition(|d| matches!(d, DerivedDeclarator::Pointer));
                let array_dims: Vec<Option<usize>> = if let Some(lpp) = last_ptr_pos {
                    // First try: collect Array dims after the last pointer
                    let after_dims: Vec<Option<usize>> = derived[lpp + 1..].iter().filter_map(|d| {
                        if let DerivedDeclarator::Array(size_expr) = d {
                            Some(size_expr.as_ref().and_then(|e| self.expr_as_array_size(e).map(|n| n as usize)))
                        } else {
                            None
                        }
                    }).collect();
                    if !after_dims.is_empty() {
                        after_dims
                    } else if has_func_ptr {
                        // For function pointer arrays, array dims come BEFORE the
                        // Pointer+FunctionPointer group (e.g., [Array(3), Pointer, FuncPtr])
                        derived[..lpp].iter().filter_map(|d| {
                            if let DerivedDeclarator::Array(size_expr) = d {
                                Some(size_expr.as_ref().and_then(|e| self.expr_as_array_size(e).map(|n| n as usize)))
                            } else {
                                None
                            }
                        }).collect()
                    } else {
                        after_dims
                    }
                } else {
                    // Pointer comes from type spec (typedef'd pointer/func ptr),
                    // not from derived list. All array dims in derived belong to
                    // the variable's own dimensions.
                    self.collect_derived_array_dims(derived)
                };
                let resolved_dims: Vec<usize> = array_dims.iter().map(|d| d.unwrap_or(256)).collect();
                let total_size: usize = resolved_dims.iter().product::<usize>() * ptr_sz;
                let strides = if resolved_dims.len() > 1 {
                    Self::compute_strides_from_dims(&resolved_dims, ptr_sz)
                } else {
                    vec![ptr_sz]  // 1D pointer array: stride is just pointer size
                };
                return (total_size, ptr_sz, true, false, strides);
            }
            // Pointer to array (e.g., int (*p)[5]) - treat as pointer
            // The trailing Pointer entries in derived represent the outermost indirection
            // (e.g., `(*p)` adds one trailing Pointer). Everything before the trailing
            // Pointer(s) describes the pointed-to type.
            //
            // Examples:
            //   int (*p)[5]       -> derived=[Array(5), Pointer]          -> trailing=1, rest=[Array(5)]
            //   Node* (*p)[2]     -> derived=[Pointer, Array(2), Pointer] -> trailing=1, rest=[Pointer, Array(2)]
            //   char (**pp)[2]    -> derived=[Array(2), Pointer, Pointer] -> trailing=2, pp is ptr-to-ptr
            let trailing_ptr_count = derived.iter().rev()
                .take_while(|d| matches!(d, DerivedDeclarator::Pointer))
                .count();

            // When there are multiple trailing pointers (e.g., char (**pp)[2]),
            // the outermost pointer is a pointer-to-pointer, not pointer-to-array.
            // pp[i] strides by sizeof(pointer)=8 and requires a load (pointer deref).
            if trailing_ptr_count >= 2 {
                return (ptr_sz, ptr_sz, false, true, vec![]);
            }

            // The non-trailing portion of derived describes the pointed-to type.
            // Collect array dimensions and count any pointer entries (which make the
            // element type a pointer, e.g., Node* (*p)[2] has element type Node*).
            let rest = &derived[..derived.len() - trailing_ptr_count];
            let array_dims: Vec<usize> = rest.iter()
                .filter_map(|d| {
                    if let DerivedDeclarator::Array(size_expr) = d {
                        Some(size_expr.as_ref()
                            .and_then(|e| self.expr_as_array_size(e).map(|n| n as usize))
                            .unwrap_or(1))
                    } else {
                        None
                    }
                }).collect();
            // If the non-trailing part has Pointer entries, the element type includes
            // those pointer levels. E.g., for Node* (*p)[2], rest=[Pointer, Array(2)],
            // the Pointer makes the element type Node* (a pointer), so elem size = ptr_sz.
            let rest_has_pointer = rest.iter().any(|d| matches!(d, DerivedDeclarator::Pointer));
            let base_elem_size = if rest_has_pointer {
                ptr_sz
            } else {
                self.sizeof_type(ts)
            };
            let full_array_size: usize = if array_dims.is_empty() {
                base_elem_size
            } else {
                array_dims.iter().product::<usize>() * base_elem_size
            };

            // strides[0] = full pointed-to array size, then per-dim strides
            let mut strides = vec![full_array_size];
            if !array_dims.is_empty() {
                strides.extend(Self::compute_strides_from_dims(&array_dims, base_elem_size));
            }
            let elem_size = full_array_size;
            return (ptr_sz, elem_size, false, true, strides);
        }

        // If the resolved type itself is an Array (e.g., va_list = Array(Char, 24),
        // or typedef'd multi-dimensional arrays like typedef int arr_t[2][3])
        // and there are no derived array declarators, handle it as an array type.
        let derived_has_array = derived.iter().any(|d| matches!(d, DerivedDeclarator::Array(_)));
        if !derived_has_array && !has_pointer {
            // Check both TypeSpecifier::Array and CType::Array (for typedef'd arrays)
            let is_ts_array = matches!(ts, TypeSpecifier::Array(_, _));
            let is_ctype_array = matches!(resolved_ctype, CType::Array(_, _));
            if is_ts_array {
                let all_dims = self.collect_type_array_dims(ts);
                let mut inner = ts;
                while let TypeSpecifier::Array(elem, _) = inner {
                    inner = elem.as_ref();
                }
                let base_elem_size = self.sizeof_type(inner).max(1);
                let total: usize = all_dims.iter().product::<usize>() * base_elem_size;
                let strides = Self::compute_strides_from_dims(&all_dims, base_elem_size);
                let elem_size = if strides.len() > 1 { strides[0] } else { base_elem_size };
                return (total, elem_size, true, false, strides);
            } else if is_ctype_array && !is_ts_array {
                // Typedef'd array (e.g., va_list = CType::Array(Char, 24))
                let all_dims = Self::collect_ctype_array_dims(&resolved_ctype);
                let base_elem_size = Self::ctype_innermost_elem_size(&resolved_ctype, &self.types.borrow_struct_layouts());
                let total: usize = all_dims.iter().product::<usize>() * base_elem_size;
                let strides = Self::compute_strides_from_dims(&all_dims, base_elem_size);
                let elem_size = if strides.len() > 1 { strides[0] } else { base_elem_size };
                return (total, elem_size, true, false, strides);
            }
        }

        // Check for array declarators - collect all dimensions
        let array_dims = self.collect_derived_array_dims(derived);

        if !array_dims.is_empty() {
            let has_func_ptr = derived.iter().any(|d| matches!(d,
                DerivedDeclarator::Function(_, _) | DerivedDeclarator::FunctionPointer(_, _)));
            // Account for array dimensions in the type specifier itself
            // Check both TypeSpecifier::Array and CType::Array (for typedef'd arrays)
            let type_dims = if matches!(ts, TypeSpecifier::Array(_, _)) {
                self.collect_type_array_dims(ts)
            } else if matches!(resolved_ctype, CType::Array(_, _)) {
                Self::collect_ctype_array_dims(&resolved_ctype)
            } else {
                vec![]
            };

            let base_elem_size = if has_func_ptr {
                crate::common::types::target_ptr_size()
            } else if !type_dims.is_empty() {
                // Use CType for innermost element size (works for both direct and typedef'd arrays)
                Self::ctype_innermost_elem_size(&resolved_ctype, &self.types.borrow_struct_layouts())
            } else {
                resolved_ctype.size_ctx(&*self.types.borrow_struct_layouts()).max(1)
            };

            // Combine: derived dims come first (outermost), then type dims
            let all_dims: Vec<usize> = array_dims.iter().map(|d| d.unwrap_or(256))
                .chain(type_dims.iter().copied())
                .collect();

            // Compute total size = product of all dims * base_elem_size
            let total: usize = all_dims.iter().product::<usize>() * base_elem_size;

            // Compute strides: stride[i] = product of dims[i+1..] * base_elem_size
            let strides = Self::compute_strides_from_dims(&all_dims, base_elem_size);

            // elem_size is the stride of the outermost dimension (for 1D compat, it's base_elem_size)
            let elem_size = if strides.len() > 1 { strides[0] } else { base_elem_size };

            return (total, elem_size, true, false, strides);
        }

        // For struct/union types, use their layout size
        if let Some(layout) = self.get_struct_layout_for_type(ts) {
            return (layout.size, 0, false, false, vec![]);
        }
        // Also check CType for typedef'd structs/unions
        if resolved_ctype.is_struct_or_union() {
            if let Some(layout) = self.struct_layout_from_ctype(&resolved_ctype) {
                return (layout.size, 0, false, false, vec![]);
            }
        }

        // Regular scalar - use sizeof_type for the allocation size
        // Minimum 8 bytes on LP64 (to ensure stack alignment for loads/stores).
        // On ILP32, minimum 4 bytes (pointer-width).
        let min_alloc = if crate::common::types::target_is_32bit() { 4 } else { 8 };
        let scalar_size = resolved_ctype.size_ctx(&*self.types.borrow_struct_layouts()).max(min_alloc);
        (scalar_size, 0, false, false, vec![])
    }

    /// For Array(Array(Int, 3), 2), returns [2, 3] (but we skip the outermost
    /// since that comes from the derived declarator).
    fn collect_type_array_dims(&self, ts: &TypeSpecifier) -> Vec<usize> {
        let mut dims = Vec::new();
        let mut current_owned = ts.clone();
        loop {
            let resolved = self.resolve_type_spec(&current_owned);
            if let TypeSpecifier::Array(inner, Some(size_expr)) = &resolved {
                if let Some(n) = self.expr_as_array_size(size_expr) {
                    dims.push(n as usize);
                }
                current_owned = inner.as_ref().clone();
            } else if let TypeSpecifier::TypedefName(name) = &resolved {
                // Follow typedef to CType for typedef'd arrays
                if let Some(ctype) = self.types.typedefs.get(name) {
                    if matches!(ctype, CType::Array(_, _)) {
                        dims.extend(Self::collect_ctype_array_dims(ctype));
                    }
                }
                break;
            } else {
                break;
            }
        }
        dims
    }

    /// Collect array dimensions from a CType::Array chain.
    /// For CType::Array(CType::Array(Int, Some(3)), Some(2)), returns [2, 3].
    fn collect_ctype_array_dims(ctype: &CType) -> Vec<usize> {
        let mut dims = Vec::new();
        let mut current = ctype;
        while let CType::Array(inner, size) = current {
            dims.push(size.unwrap_or(1));
            current = inner.as_ref();
        }
        dims
    }

    /// Get the innermost element size for a CType::Array chain.
    fn ctype_innermost_elem_size(ctype: &CType, layouts: &crate::common::fx_hash::FxHashMap<Rc<str>, RcLayout>) -> usize {
        let mut current = ctype;
        while let CType::Array(inner, _) = current {
            current = inner.as_ref();
        }
        current.size_ctx(layouts).max(1)
    }

    /// Map an element size in bytes to an appropriate IrType.
    pub(super) fn ir_type_for_elem_size(&self, size: usize) -> IrType {
        match size {
            1 => IrType::I8,
            2 => IrType::I16,
            4 => IrType::I32,
            8 => IrType::I64,
            _ => crate::common::types::target_int_ir_type(),
        }
    }

}
