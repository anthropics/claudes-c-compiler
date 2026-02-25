//! Shared type-building utilities used by both sema and lowering.
//!
//! This module contains the canonical implementations of functions that convert
//! AST type syntax (TypeSpecifier + DerivedDeclarator chains) into CType values.
//!
//! The core `TypeConvertContext` trait provides a single `resolve_type_spec_to_ctype`
//! default method that handles all 22 primitive C types, pointers, arrays, and
//! function pointers identically. Only 4 cases differ between sema and lowering
//! (typedef names, struct/union, enum, typeof), so implementors provide just those
//! via required trait methods. This ensures primitive type mapping can never diverge.

use std::rc::Rc;
use crate::common::types::{AddressSpace, CType, FunctionType};
use crate::frontend::parser::ast::{
    DerivedDeclarator, EnumVariant, Expr, ParamDecl, StructFieldDecl, TypeSpecifier,
};

/// Trait for contexts that can resolve types and evaluate constant expressions.
///
/// Both sema and lowering implement this trait. The shared `resolve_type_spec_to_ctype`
/// default method handles all primitive types, pointers, arrays, and function pointers.
/// Implementors only provide the 4 divergent methods for typedef, struct/union, enum,
/// and typeof resolution.
pub trait TypeConvertContext {
    /// Resolve a typedef name to its CType.
    /// Sema: looks up in type_context.typedefs.
    /// Lowering: also checks function pointer typedefs for richer type info.
    fn resolve_typedef(&self, name: &str) -> CType;

    /// Resolve a struct or union definition to its CType.
    /// Both phases compute layout, but lowering has caching and forward-declaration logic.
    fn resolve_struct_or_union(
        &self,
        name: &Option<Rc<str>>,
        fields: &Option<Vec<StructFieldDecl>>,
        is_union: bool,
        is_packed: bool,
        pragma_pack: Option<usize>,
        struct_aligned: Option<usize>,
    ) -> CType;

    /// Resolve an enum type to its CType.
    /// Sema: returns CType::Enum with name info.
    /// Lowering: returns CType::Int (enums are ints at IR level).
    fn resolve_enum(&self, name: &Option<Rc<str>>, variants: &Option<Vec<EnumVariant>>, is_packed: bool) -> CType;

    /// Resolve typeof(expr) to a CType.
    /// Sema: returns CType::Int (doesn't have full expr type resolution yet).
    /// Lowering: evaluates the expression's type.
    fn resolve_typeof_expr(&self, expr: &Expr) -> CType;

    /// Try to evaluate a constant expression to a usize (for array sizes).
    /// Returns None if the expression cannot be evaluated at compile time.
    fn eval_const_expr_as_usize(&self, expr: &Expr) -> Option<usize>;

    /// Convert a TypeSpecifier to a CType.
    ///
    /// This default implementation handles all shared cases (22 primitive types,
    /// Pointer, Array, FunctionPointer, TypeofType, AutoType) and delegates to
    /// the 4 required trait methods for the divergent cases.
    fn resolve_type_spec_to_ctype(&self, spec: &TypeSpecifier) -> CType {
        match spec {
            // === 22 primitive types (identical in sema and lowering) ===
            TypeSpecifier::Void => CType::Void,
            TypeSpecifier::Char => CType::Char,
            TypeSpecifier::UnsignedChar => CType::UChar,
            TypeSpecifier::Short => CType::Short,
            TypeSpecifier::UnsignedShort => CType::UShort,
            TypeSpecifier::Bool => CType::Bool,
            TypeSpecifier::Int | TypeSpecifier::Signed => CType::Int,
            TypeSpecifier::UnsignedInt | TypeSpecifier::Unsigned => CType::UInt,
            TypeSpecifier::Long => CType::Long,
            TypeSpecifier::UnsignedLong => CType::ULong,
            TypeSpecifier::LongLong => CType::LongLong,
            TypeSpecifier::UnsignedLongLong => CType::ULongLong,
            TypeSpecifier::Int128 => CType::Int128,
            TypeSpecifier::UnsignedInt128 => CType::UInt128,
            TypeSpecifier::Float => CType::Float,
            TypeSpecifier::Double => CType::Double,
            TypeSpecifier::LongDouble => CType::LongDouble,
            TypeSpecifier::ComplexFloat => CType::ComplexFloat,
            TypeSpecifier::ComplexDouble => CType::ComplexDouble,
            TypeSpecifier::ComplexLongDouble => CType::ComplexLongDouble,

            // === Compound types (shared logic) ===
            TypeSpecifier::Pointer(inner, addr_space) => {
                CType::Pointer(Box::new(self.resolve_type_spec_to_ctype(inner)), *addr_space)
            }
            TypeSpecifier::Array(elem, size_expr) => {
                let elem_ctype = self.resolve_type_spec_to_ctype(elem);
                let size = size_expr.as_ref().and_then(|e| self.eval_const_expr_as_usize(e));
                CType::Array(Box::new(elem_ctype), size)
            }
            TypeSpecifier::FunctionPointer(return_type, params, variadic) => {
                let ret_ctype = self.resolve_type_spec_to_ctype(return_type);
                let param_ctypes: Vec<(CType, Option<Rc<str>>)> = params.iter().map(|p| {
                    let ty = self.resolve_type_spec_to_ctype(&p.type_spec);
                    (ty, p.name.clone())
                }).collect();
                CType::Pointer(Box::new(CType::Function(Box::new(FunctionType {
                    return_type: ret_ctype,
                    params: param_ctypes,
                    variadic: *variadic,
                }))), AddressSpace::Default)
            }
            TypeSpecifier::BareFunction(return_type, params, variadic) => {
                // Bare function type (no pointer wrapper) — produced by typeof on
                // function names. Resolves to CType::Function, NOT Pointer(Function).
                let ret_ctype = self.resolve_type_spec_to_ctype(return_type);
                let param_ctypes: Vec<(CType, Option<Rc<str>>)> = params.iter().map(|p| {
                    let ty = self.resolve_type_spec_to_ctype(&p.type_spec);
                    (ty, p.name.clone())
                }).collect();
                CType::Function(Box::new(FunctionType {
                    return_type: ret_ctype,
                    params: param_ctypes,
                    variadic: *variadic,
                }))
            }
            TypeSpecifier::TypeofType(inner) => self.resolve_type_spec_to_ctype(inner),
            TypeSpecifier::AutoType => CType::Int,

            // === Divergent cases (delegated to implementors) ===
            TypeSpecifier::TypedefName(name) => self.resolve_typedef(name),
            TypeSpecifier::Struct(name, fields, is_packed, pragma_pack, struct_aligned) => {
                self.resolve_struct_or_union(name, fields, false, *is_packed, *pragma_pack, *struct_aligned)
            }
            TypeSpecifier::Union(name, fields, is_packed, pragma_pack, struct_aligned) => {
                self.resolve_struct_or_union(name, fields, true, *is_packed, *pragma_pack, *struct_aligned)
            }
            TypeSpecifier::Enum(name, variants, is_packed) => self.resolve_enum(name, variants, *is_packed),
            TypeSpecifier::Typeof(expr) => self.resolve_typeof_expr(expr),
            TypeSpecifier::Vector(inner, total_bytes) => {
                let elem_ctype = self.resolve_type_spec_to_ctype(inner);
                CType::Vector(Box::new(elem_ctype), *total_bytes)
            }
        }
    }
}

/// Find the start index of the function pointer core in a derived declarator list.
///
/// The function pointer core is one of:
/// - `[Pointer, FunctionPointer]` — the `(*name)(params)` syntax
/// - Standalone `FunctionPointer` — direct function pointer declarator
/// - Standalone `Function` — function declaration (not pointer)
///
/// Returns `Some(index)` where the core begins, or `None` if no function
/// pointer/function declarator is present.
fn find_function_pointer_core(derived: &[DerivedDeclarator]) -> Option<usize> {
    for i in 0..derived.len() {
        // Look for Pointer followed by FunctionPointer
        if matches!(&derived[i], DerivedDeclarator::Pointer)
            && i + 1 < derived.len()
                && matches!(&derived[i + 1], DerivedDeclarator::FunctionPointer(_, _))
            {
                return Some(i);
            }
        // Standalone FunctionPointer
        if matches!(&derived[i], DerivedDeclarator::FunctionPointer(_, _)) {
            return Some(i);
        }
        // Standalone Function (for function declarations)
        if matches!(&derived[i], DerivedDeclarator::Function(_, _)) {
            return Some(i);
        }
    }
    None
}

/// Convert a ParamDecl list to a list of (CType, Option<name>) pairs.
///
/// Uses the provided `TypeConvertContext` to resolve each parameter's type.
fn convert_param_decls_to_ctypes(
    ctx: &dyn TypeConvertContext,
    params: &[ParamDecl],
) -> Vec<(CType, Option<Rc<str>>)> {
    params
        .iter()
        .map(|p| {
            let ty = ctx.resolve_type_spec_to_ctype(&p.type_spec);
            (ty, p.name.clone())
        })
        .collect()
}

/// Build a full CType from a TypeSpecifier and DerivedDeclarator chain.
///
/// The derived list is produced by the parser's declarator handling, which stores
/// declarators outer-to-inner. For building the CType, we process inner-to-outer
/// (the C "inside-out" declarator rule).
///
/// This is the single canonical implementation used by both sema and lowering.
///
/// Examples (derived list → CType):
/// - `int **p`: [Pointer, Pointer] → Pointer(Pointer(Int))
/// - `int *arr[3]`: [Pointer, Array(3)] → Array(Pointer(Int), 3)
/// - `int (*fp)(int)`: [Pointer, FunctionPointer([int])] → Pointer(Function(Int→Int))
/// - `int (*fp[3])(int)`: [Array(3), Pointer, FunctionPointer([int])] → Array(Pointer(Function(Int→Int)), 3)
/// - `Page *(*xFetch)(int)`: [Pointer, Pointer, FunctionPointer([int])] → Pointer(Function(Pointer(Page)→...))
pub fn build_full_ctype(
    ctx: &dyn TypeConvertContext,
    type_spec: &crate::frontend::parser::ast::TypeSpecifier,
    derived: &[DerivedDeclarator],
) -> CType {
    let base = ctx.resolve_type_spec_to_ctype(type_spec);
    build_full_ctype_with_base(ctx, base, derived)
}

/// Build a full CType from an already-resolved base type and derived declarators.
/// Use this instead of `build_full_ctype` when the base type has already been
/// resolved (e.g., to avoid re-resolving anonymous struct type specs which would
/// generate different anonymous struct keys for the same declaration).
pub fn build_full_ctype_with_base(
    ctx: &dyn TypeConvertContext,
    base: CType,
    derived: &[DerivedDeclarator],
) -> CType {
    let fptr_idx = find_function_pointer_core(derived);

    if let Some(fp_start) = fptr_idx {
        // Build the function pointer type.
        // Pointer declarators in the prefix (before fp_start) are part of the
        // return type, not outer wrappers. E.g. for `Page *(*xFetch)(int)`:
        //   derived = [Pointer, Pointer, FunctionPointer([int])]
        //   prefix  = [Pointer]  — the `*` on return type `Page *`
        //   core    = [Pointer, FunctionPointer] — the `(*)(int)` syntax
        // We fold prefix Pointer declarators into the base to form the return type.
        let mut result = base;
        for d in &derived[..fp_start] {
            if matches!(d, DerivedDeclarator::Pointer) {
                result = CType::Pointer(Box::new(result), AddressSpace::Default);
            }
            // Array declarators in prefix are outer wrappers, handled after the core.
        }

        // Process from fp_start to end (the function pointer core and any
        // additional inner wrappers after it)
        let mut i = fp_start;
        while i < derived.len() {
            match &derived[i] {
                DerivedDeclarator::Pointer => {
                    if i + 1 < derived.len()
                        && matches!(
                            &derived[i + 1],
                            DerivedDeclarator::FunctionPointer(_, _)
                                | DerivedDeclarator::Function(_, _)
                        )
                    {
                        let (params, variadic) = match &derived[i + 1] {
                            DerivedDeclarator::FunctionPointer(p, v)
                            | DerivedDeclarator::Function(p, v) => (p, *v),
                            _ => unreachable!("expected FunctionPointer/Function declarator after Pointer"),
                        };
                        let param_types = convert_param_decls_to_ctypes(ctx, params);
                        let func_type = CType::Function(Box::new(FunctionType {
                            return_type: result,
                            params: param_types,
                            variadic,
                        }));
                        result = CType::Pointer(Box::new(func_type), AddressSpace::Default);
                        i += 2;
                    } else {
                        result = CType::Pointer(Box::new(result), AddressSpace::Default);
                        i += 1;
                    }
                }
                DerivedDeclarator::FunctionPointer(params, variadic) => {
                    let param_types = convert_param_decls_to_ctypes(ctx, params);
                    let func_type = CType::Function(Box::new(FunctionType {
                        return_type: result,
                        params: param_types,
                        variadic: *variadic,
                    }));
                    result = CType::Pointer(Box::new(func_type), AddressSpace::Default);
                    i += 1;
                }
                DerivedDeclarator::Function(params, variadic) => {
                    let param_types = convert_param_decls_to_ctypes(ctx, params);
                    let func_type = CType::Function(Box::new(FunctionType {
                        return_type: result,
                        params: param_types,
                        variadic: *variadic,
                    }));
                    result = func_type;
                    i += 1;
                }
                DerivedDeclarator::Array(size_expr) => {
                    // Array declarators after the function pointer core are outer
                    // wrappers (e.g., array-of-function-pointers when inner_derived
                    // had [Pointer, Array(N)] which the parser emits as
                    // [Pointer, FunctionPointer, Array(N)]).
                    let size = size_expr
                        .as_ref()
                        .and_then(|e| ctx.eval_const_expr_as_usize(e));
                    result = CType::Array(Box::new(result), size);
                    i += 1;
                }
            }
        }

        // Apply outer wrappers from the prefix (before fp_start).
        // Only Array declarators in the prefix are true outer wrappers
        // (e.g., `int (*fp[10])(void)` = array of function pointers).
        // Pointer declarators in the prefix were already folded into the return type.
        let prefix = &derived[..fp_start];
        for d in prefix.iter().rev() {
            if let DerivedDeclarator::Array(size_expr) = d {
                let size = size_expr
                    .as_ref()
                    .and_then(|e| ctx.eval_const_expr_as_usize(e));
                result = CType::Array(Box::new(result), size);
            }
        }

        result
    } else {
        // No function pointer — simple case: apply pointers and arrays
        let mut result = base;
        let mut i = 0;
        while i < derived.len() {
            match &derived[i] {
                DerivedDeclarator::Pointer => {
                    result = CType::Pointer(Box::new(result), AddressSpace::Default);
                    i += 1;
                }
                DerivedDeclarator::Array(_) => {
                    // Collect consecutive array dimensions
                    let start = i;
                    while i < derived.len()
                        && matches!(&derived[i], DerivedDeclarator::Array(_))
                    {
                        i += 1;
                    }
                    // Apply in reverse: innermost (rightmost) dimension wraps first
                    for j in (start..i).rev() {
                        if let DerivedDeclarator::Array(size_expr) = &derived[j] {
                            let size = size_expr
                                .as_ref()
                                .and_then(|e| ctx.eval_const_expr_as_usize(e));
                            result = CType::Array(Box::new(result), size);
                        }
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }
        result
    }
}
