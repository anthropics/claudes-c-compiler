//! CType conversion for the lowerer.
//!
//! Bidirectional conversion between TypeSpecifier and CType, function
//! pointer parameter handling, struct/union-to-CType conversion, and
//! the TypeConvertContext trait implementation.

use std::rc::Rc;
use crate::common::type_builder;
use crate::frontend::parser::ast::{
    DerivedDeclarator,
    EnumVariant,
    Expr,
    ParamDecl,
    StructFieldDecl,
    TypeSpecifier,
};
use crate::common::types::{AddressSpace, StructField, StructLayout, CType};
use super::lower::Lowerer;

impl Lowerer {
    /// Convert a CType back to a TypeSpecifier (for typeof and __auto_type resolution).
    pub(super) fn ctype_to_type_spec(ctype: &CType) -> TypeSpecifier {
        match ctype {
            CType::Void => TypeSpecifier::Void,
            CType::Bool => TypeSpecifier::Bool,
            CType::Char => TypeSpecifier::Char,
            CType::UChar => TypeSpecifier::UnsignedChar,
            CType::Short => TypeSpecifier::Short,
            CType::UShort => TypeSpecifier::UnsignedShort,
            CType::Int => TypeSpecifier::Int,
            CType::UInt => TypeSpecifier::UnsignedInt,
            CType::Long => TypeSpecifier::Long,
            CType::ULong => TypeSpecifier::UnsignedLong,
            CType::LongLong => TypeSpecifier::LongLong,
            CType::ULongLong => TypeSpecifier::UnsignedLongLong,
            CType::Int128 => TypeSpecifier::Int128,
            CType::UInt128 => TypeSpecifier::UnsignedInt128,
            CType::Float => TypeSpecifier::Float,
            CType::Double => TypeSpecifier::Double,
            CType::LongDouble => TypeSpecifier::LongDouble,
            CType::ComplexFloat => TypeSpecifier::ComplexFloat,
            CType::ComplexDouble => TypeSpecifier::ComplexDouble,
            CType::ComplexLongDouble => TypeSpecifier::ComplexLongDouble,
            CType::Pointer(inner, _) => {
                // Special case: Pointer(Function(...)) -> FunctionPointer TypeSpecifier
                // This preserves function pointer type info through the CType -> TypeSpecifier
                // roundtrip, which is critical for typeof on function pointer variables.
                // Without this, typeof(func_ptr_var) would lose the function type and produce
                // Pointer(Int), causing local variables to be misidentified as extern symbols.
                if let CType::Function(ft) = inner.as_ref() {
                    let ret_ts = Self::ctype_to_type_spec(&ft.return_type);
                    let param_decls: Vec<ParamDecl> = ft.params.iter().map(|(cty, name)| {
                        ParamDecl {
                            type_spec: Self::ctype_to_type_spec(cty),
                            name: name.clone(),
                            fptr_params: None,
                            is_const: false,
                            vla_size_exprs: Vec::new(),
                            fptr_inner_ptr_depth: 0,
                        }
                    }).collect();
                    TypeSpecifier::FunctionPointer(Box::new(ret_ts), param_decls, ft.variadic)
                } else {
                    TypeSpecifier::Pointer(Box::new(Self::ctype_to_type_spec(inner)), AddressSpace::Default)
                }
            }
            CType::Array(elem, size) => TypeSpecifier::Array(
                Box::new(Self::ctype_to_type_spec(elem)),
                size.map(|s| Box::new(Expr::IntLiteral(s as i64, crate::common::source::Span::dummy()))),
            ),
            CType::Struct(key) => {
                // Extract tag name from key (e.g., "struct.Foo" -> "Foo")
                // For anonymous structs (key like "__anon_struct_N"), use the
                // full key as the tag so get_struct_layout_for_type can find it.
                if let Some(tag) = key.strip_prefix("struct.") {
                    TypeSpecifier::Struct(Some(Rc::from(tag)), None, false, None, None)
                } else {
                    TypeSpecifier::Struct(Some(key.clone()), None, false, None, None)
                }
            }
            CType::Union(key) => {
                // Extract tag name from key (e.g., "union.Bar" -> "Bar")
                // For anonymous unions (key like "__anon_struct_N"), use the
                // full key as the tag so get_struct_layout_for_type can find it.
                if let Some(tag) = key.strip_prefix("union.") {
                    TypeSpecifier::Union(Some(Rc::from(tag)), None, false, None, None)
                } else {
                    TypeSpecifier::Union(Some(key.clone()), None, false, None, None)
                }
            }
            CType::Enum(et) => {
                TypeSpecifier::Enum(et.name.clone(), None, et.is_packed)
            }
            CType::Function(ft) => {
                // Bare function type — NOT a pointer. Use BareFunction so that
                // typeof(func_name) preserves the function type without adding a
                // pointer level. FunctionPointer already includes a pointer wrapper.
                let ret_ts = Self::ctype_to_type_spec(&ft.return_type);
                let param_decls: Vec<ParamDecl> = ft.params.iter().map(|(cty, name)| {
                    ParamDecl {
                        type_spec: Self::ctype_to_type_spec(cty),
                        name: name.clone(),
                        fptr_params: None,
                        is_const: false,
                        vla_size_exprs: Vec::new(),
                        fptr_inner_ptr_depth: 0,
                    }
                }).collect();
                TypeSpecifier::BareFunction(Box::new(ret_ts), param_decls, ft.variadic)
            }
            // Vector types fall back to element type for type-spec conversion (used
            // by implicit cast logic). This is safe because vector operations are
            // lowered element-wise and never rely on this round-trip for sizing.
            // TODO: Vector subscript (v[i]) and unary ops (-v, ~v) not yet implemented
            CType::Vector(elem, _) => Self::ctype_to_type_spec(elem),
        }
    }

    /// Build the CType for a function parameter, correctly handling function
    /// pointer parameters (both explicit syntax and typedef'd).
    ///
    /// For explicit function pointer params like `void (*callback)(int, int)`,
    /// the parser sets `fptr_params` to Some(...) and we build
    /// CType::Pointer(CType::Function(...)).
    ///
    /// For typedef'd function pointer params like `lua_Alloc f`, the parser
    /// doesn't set `fptr_params` (the function pointer nature is hidden in the
    /// typedef). We detect this by checking if the resolved type spec has
    /// DerivedDeclarator::FunctionPointer info stored in function_typedefs or
    /// by checking the original typedef declaration.
    pub(super) fn param_ctype(&self, param: &crate::frontend::parser::ast::ParamDecl) -> CType {
        // Case 1: explicit function pointer syntax - fptr_params is set
        if let Some(ref fptr_params) = param.fptr_params {
            let return_ctype = self.type_spec_to_ctype(&param.type_spec);
            // The parser's `is_func_ptr` flag adds exactly one Pointer layer for
            // the (*name) indirection. Peel that single layer; any remaining
            // Pointer layers belong to the actual return type.
            //
            // Examples (type_spec → peel 1 → return type):
            //   int (*f)(...)           → Pointer(Int) → Int
            //   int *(*f)(...)          → Pointer(Pointer(Int)) → Pointer(Int)
            //   struct Node *(*f)(...) → Pointer(Pointer(Struct)) → Pointer(Struct)
            let actual_return = if let CType::Pointer(inner, _) = return_ctype {
                *inner
            } else {
                return_ctype
            };

            let param_types: Vec<(CType, Option<Rc<str>>)> = fptr_params.iter()
                .map(|p| (self.type_spec_to_ctype(&p.type_spec), p.name.clone()))
                .collect();
            let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                return_type: actual_return,
                params: param_types,
                variadic: false,
            }));
            let result = CType::Pointer(Box::new(func_type), AddressSpace::Default);
            return result;
        }

        // Case 2: typedef'd function pointer (e.g., lua_Alloc f)
        // Check if the parameter's type resolves to a typedef that was
        // declared as a function pointer typedef
        if let TypeSpecifier::TypedefName(tname) = &param.type_spec {
            if self.is_typedef_function_pointer(tname) {
                // Build the full function pointer CType from the typedef info
                if let Some(fptr_ctype) = self.build_function_pointer_ctype_from_typedef(tname) {
                    return fptr_ctype;
                }
            }

            // Case 3: bare function typedef (e.g., `typedef int filler_t(void*, void*);`
            // used as parameter `filler_t filler`). Per C11 6.7.6.3p8, a parameter of
            // function type is adjusted to pointer-to-function type.
            if let Some(fti) = self.types.function_typedefs.get(tname).cloned() {
                let return_ctype = self.type_spec_to_ctype(&fti.return_type);
                let param_types: Vec<(CType, Option<Rc<str>>)> = fti.params.iter()
                    .map(|p| (self.param_ctype(p), p.name.clone()))
                    .collect();
                let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                    return_type: return_ctype,
                    params: param_types,
                    variadic: fti.variadic,
                }));
                return CType::Pointer(Box::new(func_type), AddressSpace::Default);
            }
        }

        // Default: use the standard type_spec_to_ctype, then apply parameter adjustments.
        let ctype = self.type_spec_to_ctype(&param.type_spec);
        // C11 6.7.6.3p7: "A declaration of a parameter as 'array of type' shall be
        // adjusted to 'qualified pointer to type'". This handles typedef'd array types
        // like `typedef struct S name[1]` used as function parameters.
        if let CType::Array(elem, _) = ctype {
            return CType::Pointer(elem, AddressSpace::Default);
        }
        ctype
    }

    /// Check if a typedef name refers to a function pointer type.
    /// This handles typedefs like `typedef void *(*lua_Alloc)(void *, ...)`.
    fn is_typedef_function_pointer(&self, tname: &str) -> bool {
        // Check if the typedef's resolved type is a Pointer AND the original
        // typedef had a FunctionPointer derived declarator.
        // We track this via function_typedefs which stores function type info
        // for both bare function typedefs and function pointer typedefs.
        //
        // However, function_typedefs only stores bare function typedefs
        // (typedef int func_t(int)), not function pointer typedefs.
        // For function pointer typedefs, we check the resolved typedef: if it's
        // a Pointer type and the original declaration context implies function pointer.
        //
        // Heuristic: if the typedef resolves to Pointer(X) and there's also a
        // function_typedefs entry, it was a function pointer typedef. But this
        // isn't reliable. Instead, track function pointer typedefs explicitly.
        self.types.func_ptr_typedefs.contains(tname)
    }

    /// Build a CType::Pointer(CType::Function(...)) from a function pointer typedef.
    fn build_function_pointer_ctype_from_typedef(&self, tname: &str) -> Option<CType> {
        // Look up the stored function pointer typedef info
        if let Some(fti) = self.types.func_ptr_typedef_info.get(tname) {
            let return_ctype = self.type_spec_to_ctype(&fti.return_type);
            let param_types: Vec<(CType, Option<Rc<str>>)> = fti.params.iter()
                .map(|p| (self.type_spec_to_ctype(&p.type_spec), p.name.clone()))
                .collect();
            let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                return_type: return_ctype,
                params: param_types,
                variadic: fti.variadic,
            }));
            return Some(CType::Pointer(Box::new(func_type), AddressSpace::Default));
        }
        None
    }

    /// Convert a TypeSpecifier to CType (for struct layout computation).
    /// Delegates to the shared `TypeConvertContext::resolve_type_spec_to_ctype` default
    /// method, which handles all 22 primitive types and delegates struct/union/enum/typedef
    /// to lowering-specific trait methods.
    pub(super) fn type_spec_to_ctype(&self, ts: &TypeSpecifier) -> CType {
        use crate::common::type_builder::TypeConvertContext;
        self.resolve_type_spec_to_ctype(ts)
    }

    /// Convert a struct or union TypeSpecifier to CType.
    /// `is_union` selects between struct and union semantics.
    /// `is_packed` indicates __attribute__((packed)).
    /// `pragma_pack` is the #pragma pack(N) alignment, if any.
    fn struct_or_union_to_ctype(
        &self,
        name: &Option<Rc<str>>,
        fields: &Option<Vec<StructFieldDecl>>,
        is_union: bool,
        is_packed: bool,
        pragma_pack: Option<usize>,
        struct_aligned: Option<usize>,
    ) -> CType {
        let prefix = if is_union { "union" } else { "struct" };
        let wrap = |key: Rc<str>| -> CType {
            if is_union { CType::Union(key) } else { CType::Struct(key) }
        };
        // __attribute__((packed)) forces alignment 1; #pragma pack(N) caps to N.
        let max_field_align = if is_packed { Some(1) } else { pragma_pack };

        if let Some(fs) = fields {
            // Inline definition with fields: check if register_struct_type already
            // inserted the definitive layout for this named struct/union.
            // If so, skip re-computing and re-inserting (which would corrupt the
            // scope undo-log with a redundant shadow entry).
            // Only skip when the existing layout has fields (not a forward-declaration stub).
            if let Some(tag) = name {
                let cache_key: Rc<str> = format!("{}.{}", prefix, tag).into();
                if let Some(existing) = self.types.borrow_struct_layouts().get(&*cache_key) {
                    if !existing.fields.is_empty() {
                        let result = wrap(cache_key.clone());
                        self.types.ctype_cache.borrow_mut().insert(cache_key, result.clone());
                        return result;
                    }
                }
            }
            let struct_fields: Vec<StructField> = fs.iter().map(|f| {
                let bit_width = f.bit_width.as_ref().and_then(|bw| {
                    self.eval_const_expr(bw).and_then(|c| c.to_u32())
                });
                let mut ty = self.struct_field_ctype(f);
                // GCC treats enum bitfields as unsigned: values are zero-extended
                // on load, not sign-extended. Check both direct enum type specs
                // and typedef'd enum types (e.g., typedef enum EFoo EFoo).
                if bit_width.is_some()
                    && self.is_enum_type_spec(&f.type_spec)
                        && ty == CType::Int {
                            ty = CType::UInt;
                        }
                StructField {
                    name: f.name.clone().unwrap_or_default(),
                    ty,
                    bit_width,
                    alignment: f.alignment,
                    is_packed: f.is_packed,
                }
            }).collect();
            let mut layout = if is_union {
                StructLayout::for_union_with_packing(&struct_fields, max_field_align, &*self.types.borrow_struct_layouts())
            } else {
                StructLayout::for_struct_with_packing(&struct_fields, max_field_align, &*self.types.borrow_struct_layouts())
            };
            // Apply struct-level __attribute__((aligned(N))): sets minimum alignment
            if let Some(a) = struct_aligned {
                if a > layout.align {
                    layout.align = a;
                    let mask = layout.align - 1;
                    layout.size = (layout.size + mask) & !mask;
                }
            }
            let key: Rc<str> = if let Some(tag) = name {
                format!("{}.{}", prefix, tag).into()
            } else {
                let id = self.types.next_anon_struct_id();
                format!("__anon_struct_{}", id).into()
            };
            self.types.insert_struct_layout_scoped_from_ref(&key, layout);
            self.types.invalidate_ctype_cache_scoped_from_ref(&key);
            let result = wrap(key.clone());
            self.types.ctype_cache.borrow_mut().insert(key, result.clone());
            result
        } else if let Some(tag) = name {
            // If the tag is already an anonymous struct/union key (e.g. from
            // typeof resolution via ctype_to_type_spec), use it directly instead
            // of prepending the struct/union prefix. This avoids creating a
            // mismatched key like "struct.__anon_struct_N" when the real layout
            // is stored at "__anon_struct_N".
            let key: Rc<str> = if tag.starts_with("__anon_struct_") || tag.starts_with("__anon_union_") {
                tag.clone()
            } else {
                format!("{}.{}", prefix, tag).into()
            };
            // Check cache first
            if let Some(cached) = self.types.ctype_cache.borrow().get(&*key) {
                return cached.clone();
            }
            // Forward declaration: insert an empty layout if not already present
            if self.types.borrow_struct_layouts().get(&*key).is_none() {
                let empty_layout = StructLayout {
                    fields: Vec::new(),
                    size: 0,
                    align: 1,
                    is_union,
                    is_transparent_union: false,
                };
                self.types.insert_struct_layout_from_ref(key.clone(), empty_layout);
            }
            let result = wrap(key.clone());
            self.types.ctype_cache.borrow_mut().insert(key, result.clone());
            result
        } else {
            // Anonymous forward declaration (no name, no fields)
            let id = self.types.next_anon_struct_id();
            let key: Rc<str> = format!("__anon_struct_{}", id).into();
            let empty_layout = StructLayout {
                fields: Vec::new(),
                size: 0,
                align: 1,
                is_union,
                is_transparent_union: false,
            };
            self.types.insert_struct_layout_from_ref(key.clone(), empty_layout);
            wrap(key)
        }
    }

    /// Get the CType for a struct field declaration, accounting for derived declarators.
    /// For simple fields (derived is empty), just converts type_spec.
    /// For complex fields (function pointers, etc.), uses build_full_ctype.
    pub(super) fn struct_field_ctype(&self, f: &StructFieldDecl) -> CType {
        if f.derived.is_empty() {
            self.type_spec_to_ctype(&f.type_spec)
        } else {
            self.build_full_ctype(&f.type_spec, &f.derived)
        }
    }

    /// Build a full CType from a TypeSpecifier and DerivedDeclarator chain.
    /// Delegates to the shared type_builder module for canonical inside-out
    /// declarator application logic.
    pub(super) fn build_full_ctype(&self, type_spec: &TypeSpecifier, derived: &[DerivedDeclarator]) -> CType {
        type_builder::build_full_ctype(self, type_spec, derived)
    }
}

/// Implement TypeConvertContext so shared type_builder functions can call back
/// into the lowerer for type resolution and constant expression evaluation.
///
/// The 4 divergent methods handle lowering-specific behavior:
/// - typedef: also checks function pointer typedefs for richer type info
/// - struct/union: has caching, forward-declaration handling, enum bitfield fixup
/// - enum: returns CType::Int (enums are plain ints at IR level)
/// - typeof: evaluates the expression's actual type
impl type_builder::TypeConvertContext for Lowerer {
    fn resolve_typedef(&self, name: &str) -> CType {
        // Check function pointer typedefs first (they carry richer type info)
        if let Some(fptr_ctype) = self.build_function_pointer_ctype_from_typedef(name) {
            return fptr_ctype;
        }
        // Direct CType lookup from typedef map
        if let Some(ctype) = self.types.typedefs.get(name) {
            return ctype.clone();
        }
        CType::Int // fallback for unresolved typedef
    }

    fn resolve_struct_or_union(
        &self,
        name: &Option<Rc<str>>,
        fields: &Option<Vec<StructFieldDecl>>,
        is_union: bool,
        is_packed: bool,
        pragma_pack: Option<usize>,
        struct_aligned: Option<usize>,
    ) -> CType {
        self.struct_or_union_to_ctype(name, fields, is_union, is_packed, pragma_pack, struct_aligned)
    }

    fn resolve_enum(&self, name: &Option<Rc<str>>, variants: &Option<Vec<EnumVariant>>, is_packed: bool) -> CType {
        // Check if this is a forward reference to a known packed enum
        let effective_packed = is_packed || name.as_ref()
            .and_then(|n| self.types.packed_enum_types.get(n))
            .is_some();
        if !effective_packed {
            // Non-packed enum: normally int (4 bytes), but if any variant
            // value exceeds i32 range while fitting in u32, use unsigned int.
            // Values like `1U << 31` (0x80000000) require unsigned int.
            let needs_unsigned = if let Some(vars) = variants {
                let mut next_val: i64 = 0;
                let mut result = false;
                for v in vars {
                    if let Some(ref expr) = v.value {
                        if let Some(val) = self.eval_const_expr(expr) {
                            if let Some(v) = self.const_to_i64(&val) {
                                next_val = v;
                            }
                        }
                    }
                    if next_val > i32::MAX as i64 {
                        result = true;
                    }
                    next_val += 1;
                }
                result
            } else {
                false
            };
            return if needs_unsigned { CType::UInt } else { CType::Int };
        }
        // For packed enums, compute the minimum integer type from variant values
        let variant_values: Vec<i64> = if let Some(vars) = variants {
            let mut values = Vec::new();
            let mut next_val: i64 = 0;
            for v in vars {
                if let Some(ref expr) = v.value {
                    if let Some(val) = self.eval_const_expr(expr) {
                        if let Some(v) = self.const_to_i64(&val) {
                            next_val = v;
                        }
                    }
                }
                values.push(next_val);
                next_val += 1;
            }
            values
        } else if let Some(n) = name {
            // Forward reference: look up stored packed enum info
            if let Some(et) = self.types.packed_enum_types.get(n) {
                et.variants.iter().map(|(_, v)| *v).collect()
            } else {
                return CType::Char; // packed enum with no known variants, default 1 byte
            }
        } else {
            return CType::Char; // anonymous packed enum with no body
        };

        if variant_values.is_empty() {
            return CType::Char;
        }
        let min_val = *variant_values.iter().min().unwrap();
        let max_val = *variant_values.iter().max().unwrap();
        if min_val >= 0 {
            if max_val <= 0xFF { CType::UChar }
            else if max_val <= 0xFFFF { CType::UShort }
            else { CType::UInt }
        } else if min_val >= -128 && max_val <= 127 { CType::Char }
        else if min_val >= -32768 && max_val <= 32767 { CType::Short }
        else { CType::Int }
    }

    fn resolve_typeof_expr(&self, expr: &Expr) -> CType {
        // For _Generic selections inside typeof(), resolve directly to avoid
        // stale cached None results from get_expr_ctype's memoization cache.
        // The cache can hold None when the same _Generic AST node was
        // first evaluated before local variable types were fully available
        // (e.g., during earlier type resolution passes). By calling
        // resolve_generic_selection_ctype directly we bypass the cache.
        if let Expr::GenericSelection(controlling, associations, _) = expr {
            if let Some(ctype) = self.resolve_generic_selection_ctype(controlling, associations) {
                return ctype;
            }
        }
        self.get_expr_ctype(expr).unwrap_or(CType::Int)
    }

    fn eval_const_expr_as_usize(&self, expr: &Expr) -> Option<usize> {
        self.expr_as_array_size(expr).and_then(|n| {
            if n < 0 {
                None // Negative array sizes are rejected by sema
            } else {
                Some(n as usize)
            }
        })
    }
}
