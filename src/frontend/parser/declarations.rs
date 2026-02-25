// Declaration parsing: external (top-level) and local (block-scope) declarations.
//
// External declarations handle both function definitions and variable/type
// declarations. The key challenge is disambiguating function definitions
// (which have a body) from declarations (which end with ';').
//
// K&R-style function parameters are also handled here, where parameter types
// are declared separately after the parameter name list.

use std::rc::Rc;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::source::Span;
use crate::common::types::AddressSpace;
use crate::frontend::lexer::token::TokenKind;
use super::ast::*;
use super::parse::{ModeKind, ParsedDeclAttrs, Parser};

/// Context for declaration-level attributes that flow from `parse_external_decl`
/// into `parse_declaration_rest` and `parse_function_def`.
///
/// This groups the per-declarator GCC attributes, alignment info, and the
/// `__attribute__((common))` flag that would otherwise be passed as 15+
/// individual parameters.
struct DeclContext {
    /// Per-declarator attributes (constructor, destructor, weak, etc.)
    attrs: DeclAttributes,
    /// Combined alignment from `_Alignas`, declarator attrs, and post-declarator attrs
    alignment: Option<usize>,
    /// `_Alignas(type)` for deferred alignment resolution
    alignas_type: Option<TypeSpecifier>,
    /// `__attribute__((aligned(sizeof(type))))` for deferred sizeof
    alignment_sizeof_type: Option<TypeSpecifier>,
    /// `__attribute__((common))` - emit as common symbol
    is_common: bool,
}

impl Parser {
    pub(super) fn parse_external_decl(&mut self) -> Option<ExternalDecl> {
        // Reset all declaration-level flags before parsing the next declaration.
        self.attrs = ParsedDeclAttrs::default();

        self.skip_gcc_extensions();

        // Handle #pragma pack directives (emitted as synthetic tokens by preprocessor)
        while self.handle_pragma_pack_token() {
            // Consume semicolons after pragma pack synthetic tokens
            self.consume_if(&TokenKind::Semicolon);
        }

        // Handle #pragma GCC visibility push/pop directives
        while self.handle_pragma_visibility_token() {
            self.consume_if(&TokenKind::Semicolon);
        }

        if self.at_eof() {
            return None;
        }

        // Handle top-level asm("..."); directives - emit verbatim in assembly output
        if matches!(self.peek(), TokenKind::Asm) {
            self.advance();
            self.consume_if(&TokenKind::Volatile);
            if matches!(self.peek(), TokenKind::LParen) {
                self.advance(); // consume (
                // Collect all string literal pieces (may be concatenated)
                let mut asm_str = String::new();
                loop {
                    match self.peek() {
                        TokenKind::StringLiteral(s) => {
                            asm_str.push_str(s);
                            self.advance();
                        }
                        TokenKind::RParen | TokenKind::Eof => break,
                        _ => { self.advance(); }
                    }
                }
                if matches!(self.peek(), TokenKind::RParen) {
                    self.advance();
                }
                self.consume_if(&TokenKind::Semicolon);
                if !asm_str.is_empty() {
                    return Some(ExternalDecl::TopLevelAsm(asm_str));
                }
                return Some(ExternalDecl::Declaration(Declaration::empty()));
            }
            self.consume_if(&TokenKind::Semicolon);
            return Some(ExternalDecl::Declaration(Declaration::empty()));
        }

        // Handle _Static_assert at file scope
        if matches!(self.peek(), TokenKind::StaticAssert) {
            self.parse_static_assert();
            return Some(ExternalDecl::Declaration(Declaration::empty()));
        }

        let start = self.peek_span();
        let type_spec = self.parse_type_specifier().or_else(|| {
            // C89 implicit int: if no type specifier is found, check if this looks like
            // a function definition with implicit int return type (e.g., "main() { ... }"
            // or K&R-style "Proc_1(x) int x; { ... }").
            // The pattern is: identifier followed by '('.
            if let TokenKind::Identifier(_) = self.peek() {
                if self.pos + 1 < self.tokens.len()
                    && matches!(self.tokens[self.pos + 1].kind, TokenKind::LParen)
                {
                    return Some(TypeSpecifier::Int);
                }
            }
            None
        })?;

        // Capture constructor/destructor from type-level attributes
        let type_level_ctor = self.attrs.parsing_constructor();
        let type_level_dtor = self.attrs.parsing_destructor();

        // Bare type with no declarator (e.g., struct definition)
        if self.at_eof() || matches!(self.peek(), TokenKind::Semicolon) {
            self.consume_if(&TokenKind::Semicolon);
            let mut d = Declaration::new(
                type_spec,
                Vec::new(),
                None, None, None,
                self.attrs.parsing_address_space,
                self.attrs.parsing_vector_size.take(),
                self.attrs.parsing_ext_vector_nelem.take(),
                start,
            );
            d.set_static(self.attrs.parsing_static());
            d.set_extern(self.attrs.parsing_extern());
            d.set_typedef(self.attrs.parsing_typedef());
            d.set_const(self.attrs.parsing_const());
            d.set_volatile(self.attrs.parsing_volatile());
            d.set_thread_local(self.attrs.parsing_thread_local());
            return Some(ExternalDecl::Declaration(d));
        }

        // Handle post-type storage class specifiers (C allows "struct S typedef name;")
        self.consume_post_type_qualifiers();

        let (name, derived, decl_mode, decl_common, decl_aligned, _) = self.parse_declarator_with_attrs();
        let (post_ctor, post_dtor, post_mode, post_common, post_aligned, first_asm_reg) = self.parse_asm_and_attributes();
        let mode_kind = decl_mode.or(post_mode);
        let is_common = decl_common || post_common;
        // Merge alignment from _Alignas (type specifier), declarator attrs, and post-declarator attrs.
        // Per C11, alignment can only increase (6.7.5), so take the maximum.
        let mut merged_alignment = self.attrs.parsed_alignas.take();
        let alignas_type = self.attrs.parsed_alignas_type.take();
        let alignment_sizeof_type = self.attrs.parsed_alignment_sizeof_type.take();
        for a in [decl_aligned, post_aligned].iter().copied().flatten() {
            merged_alignment = Some(merged_alignment.map_or(a, |prev| prev.max(a)));
        }
        // Merge all sources of constructor/destructor: type-level attrs, declarator-level attrs, post-declarator attrs
        let is_constructor = type_level_ctor || self.attrs.parsing_constructor() || post_ctor;
        let is_destructor = type_level_dtor || self.attrs.parsing_destructor() || post_dtor;
        // Capture alias/weak/visibility/section/error attributes
        let is_weak = self.attrs.parsing_weak();
        let alias_target = self.attrs.parsing_alias_target.take();
        // Use explicit __attribute__((visibility(...))) if present, otherwise fall back
        // to the current #pragma GCC visibility default (if any).
        let visibility = self.attrs.parsing_visibility.take()
            .or_else(|| self.pragma_default_visibility.clone());
        let section = self.attrs.parsing_section.take();
        let is_error_attr = self.attrs.parsing_error_attr();
        let is_noreturn = self.attrs.parsing_noreturn();
        let cleanup_fn = self.attrs.parsing_cleanup_fn.take();
        let symver = self.attrs.parsing_symver.take();
        let is_used = self.attrs.parsing_used();
        let is_fastcall = self.attrs.parsing_fastcall();
        let is_naked = self.attrs.parsing_naked();

        // Build per-declarator attributes struct from the collected flags
        let mut decl_attrs = DeclAttributes::default();
        decl_attrs.set_constructor(is_constructor);
        decl_attrs.set_destructor(is_destructor);
        decl_attrs.set_weak(is_weak);
        decl_attrs.set_error_attr(is_error_attr);
        decl_attrs.set_noreturn(is_noreturn);
        decl_attrs.set_used(is_used);
        decl_attrs.set_fastcall(is_fastcall);
        decl_attrs.set_naked(is_naked);
        decl_attrs.alias_target = alias_target;
        decl_attrs.visibility = visibility;
        decl_attrs.section = section;
        decl_attrs.asm_register = first_asm_reg;
        decl_attrs.cleanup_fn = cleanup_fn;
        decl_attrs.symver = symver;

        // Apply __attribute__((mode(...))): transform type to specified bit-width
        let type_spec = if let Some(mk) = mode_kind {
            mk.apply(type_spec)
        } else {
            type_spec
        };

        // Determine if this is a function definition
        let is_funcdef = !derived.is_empty()
            && matches!(derived.last(), Some(DerivedDeclarator::Function(_, _)))
            && (matches!(self.peek(), TokenKind::LBrace) || self.is_type_specifier());

        if is_funcdef {
            self.parse_function_def(type_spec, name, derived, start, decl_attrs)
        } else {
            let ctx = DeclContext {
                attrs: decl_attrs,
                alignment: merged_alignment,
                alignas_type,
                alignment_sizeof_type,
                is_common,
            };
            self.parse_declaration_rest(type_spec, name, derived, start, ctx)
        }
    }

    /// Parse the rest of a function definition after the declarator.
    fn parse_function_def(
        &mut self,
        type_spec: TypeSpecifier,
        name: Option<Rc<str>>,
        derived: Vec<DerivedDeclarator>,
        start: crate::common::source::Span,
        decl_attrs: DeclAttributes,
    ) -> Option<ExternalDecl> {
        self.attrs.set_typedef(false); // function defs are never typedefs
        let (params, variadic) = if let Some(DerivedDeclarator::Function(p, v)) = derived.last() {
            (p.clone(), *v)
        } else {
            (vec![], false)
        };

        // Handle K&R-style parameter declarations
        let is_kr_style = !matches!(self.peek(), TokenKind::LBrace);
        let final_params = if is_kr_style {
            self.parse_kr_params(params)
        } else {
            params
        };

        let is_static = self.attrs.parsing_static();
        let is_inline = self.attrs.parsing_inline();
        let is_extern = self.attrs.parsing_extern();
        let is_gnu_inline = self.attrs.parsing_gnu_inline();
        let is_always_inline = self.attrs.parsing_always_inline();
        let is_noinline = self.attrs.parsing_noinline();

        // Build return type from derived declarators
        let return_type = self.build_return_type(type_spec, &derived);

        // Shadow typedef names used as parameter names
        let saved_shadowed = self.shadowed_typedefs.clone();
        for param in &final_params {
            if let Some(ref pname) = param.name {
                if self.typedefs.contains(pname) && !self.shadowed_typedefs.contains(pname) {
                    self.shadowed_typedefs.insert(pname.clone());
                }
            }
        }
        // Clear noreturn so that _Noreturn/noreturn from the function definition
        // does not leak into local declarations inside the body.
        // parse_compound_stmt saves/restores attr flags, so this is safe.
        self.attrs.set_noreturn(false);
        let body = self.parse_compound_stmt();
        self.shadowed_typedefs = saved_shadowed;

        Some(ExternalDecl::FunctionDef(FunctionDef {
            return_type,
            name: name.unwrap_or_else(|| Rc::from("")),
            params: final_params,
            variadic,
            body,
            attrs: {
                let mut attrs = FunctionAttributes::new();
                attrs.set_static(is_static);
                attrs.set_inline(is_inline);
                attrs.set_extern(is_extern);
                attrs.set_gnu_inline(is_gnu_inline);
                attrs.set_always_inline(is_always_inline);
                attrs.set_noinline(is_noinline);
                attrs.set_constructor(decl_attrs.is_constructor());
                attrs.set_destructor(decl_attrs.is_destructor());
                attrs.set_weak(decl_attrs.is_weak());
                attrs.set_used(decl_attrs.is_used());
                attrs.set_fastcall(decl_attrs.is_fastcall());
                attrs.set_naked(decl_attrs.is_naked());
                attrs.set_noreturn(decl_attrs.is_noreturn());
                attrs.section = decl_attrs.section;
                attrs.visibility = decl_attrs.visibility;
                attrs.symver = decl_attrs.symver;
                attrs
            },
            is_kr: is_kr_style,
            span: start,
        }))
    }

    /// Build the return type from derived declarators.
    /// For `int (*func())[3]`, we apply post-Function and pre-Function derivations.
    fn build_return_type(
        &self,
        base_type: TypeSpecifier,
        derived: &[DerivedDeclarator],
    ) -> TypeSpecifier {
        let mut return_type = base_type;
        let func_pos = derived.iter().position(|d|
            matches!(d, DerivedDeclarator::Function(_, _)));

        if let Some(fpos) = func_pos {
            // Apply post-Function derivations (Array/Pointer)
            for d in &derived[fpos+1..] {
                match d {
                    DerivedDeclarator::Array(size_expr) => {
                        return_type = TypeSpecifier::Array(
                            Box::new(return_type),
                            size_expr.clone(),
                        );
                    }
                    DerivedDeclarator::Pointer => {
                        return_type = TypeSpecifier::Pointer(Box::new(return_type), AddressSpace::Default);
                    }
                    _ => {}
                }
            }
            // Apply pre-Function derivations
            for d in &derived[..fpos] {
                match d {
                    DerivedDeclarator::Pointer => {
                        return_type = TypeSpecifier::Pointer(Box::new(return_type), AddressSpace::Default);
                    }
                    DerivedDeclarator::Array(size_expr) => {
                        return_type = TypeSpecifier::Array(
                            Box::new(return_type),
                            size_expr.clone(),
                        );
                    }
                    _ => {}
                }
            }
        } else {
            // No Function in derived - just apply pointer derivations
            for d in derived {
                match d {
                    DerivedDeclarator::Pointer => {
                        return_type = TypeSpecifier::Pointer(Box::new(return_type), AddressSpace::Default);
                    }
                    _ => break,
                }
            }
        }
        return_type
    }

    /// Parse K&R-style parameter declarations.
    /// In K&R style, the parameter list is just names, and type declarations follow.
    fn parse_kr_params(&mut self, mut kr_params: Vec<ParamDecl>) -> Vec<ParamDecl> {
        while self.is_type_specifier() && !matches!(self.peek(), TokenKind::LBrace) {
            if let Some(type_spec) = self.parse_type_specifier() {
                loop {
                    let (pname, pderived) = self.parse_declarator();
                    if let Some(ref name) = pname {
                        let (full_type, fptr_params) = self.apply_kr_derivations(&type_spec, &pderived);
                        // Compute fptr_inner_ptr_depth for K&R function pointer params.
                        // Count Pointer entries AFTER FunctionPointer in derived list;
                        // these represent extra indirection (pointer-to-function-pointer).
                        let inner_depth = if fptr_params.is_some() {
                            let mut found_fptr = false;
                            let mut ptrs_after = 0u32;
                            for d in &pderived {
                                match d {
                                    DerivedDeclarator::FunctionPointer(_, _) | DerivedDeclarator::Function(_, _) => {
                                        found_fptr = true;
                                    }
                                    DerivedDeclarator::Pointer if found_fptr => {
                                        ptrs_after += 1;
                                    }
                                    _ => {}
                                }
                            }
                            // inner_ptr_depth = 1 (the (*) syntax star) + ptrs_after
                            1 + ptrs_after
                        } else {
                            0
                        };
                        for param in kr_params.iter_mut() {
                            if param.name.as_deref() == Some(&**name) {
                                param.type_spec = full_type.clone();
                                param.fptr_params = fptr_params.clone();
                                param.fptr_inner_ptr_depth = inner_depth;
                                break;
                            }
                        }
                    }
                    if !self.consume_if(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect_after(&TokenKind::Semicolon, "after parameter declaration");
            } else {
                break;
            }
        }
        kr_params
    }

    /// Apply derived declarators to build a K&R parameter's full type.
    /// Returns (type_specifier, optional_fptr_params).
    /// For function pointer parameters like `int (*fp)(int, int)`, this returns
    /// (Pointer(Int), Some([ParamDecl for int, ParamDecl for int])) to match
    /// the modern-style parameter handling path.
    fn apply_kr_derivations(
        &self,
        type_spec: &TypeSpecifier,
        pderived: &[DerivedDeclarator],
    ) -> (TypeSpecifier, Option<Vec<ParamDecl>>) {
        let mut full_type = type_spec.clone();

        // Check if this is a function pointer parameter.
        // For `int (*fp)(int)`, pderived = [Pointer, FunctionPointer([int], false)]
        // The Pointer is the syntax marker from `(*fp)`, not a real pointer level.
        // We need to: (1) not apply the syntax-marker Pointer, (2) extract the
        // FunctionPointer params, and (3) apply one Pointer wrap for decay.
        let fptr_info = pderived.iter().find_map(|d| {
            if let DerivedDeclarator::FunctionPointer(params, variadic) = d {
                Some((params.clone(), *variadic))
            } else {
                None
            }
        });

        if let Some((fptr_params, _variadic)) = fptr_info {
            // Function pointer parameter in K&R style.
            // Count how many Pointer derivations exist. For `int (*fp)()`:
            //   pderived = [Pointer, FunctionPointer([], false)]
            //   One Pointer is the syntax marker - skip it.
            //   Any additional Pointers are return-type pointer levels.
            // For `int *(*fp)()`:
            //   pderived = [Pointer, Pointer, FunctionPointer([], false)]
            //   First Pointer is return-type pointer, second is syntax marker.
            let ptr_count = pderived.iter().filter(|d| matches!(d, DerivedDeclarator::Pointer)).count();
            // Apply all pointers except the syntax marker (last one)
            for _ in 0..ptr_count.saturating_sub(1) {
                full_type = TypeSpecifier::Pointer(Box::new(full_type), AddressSpace::Default);
            }
            // Apply one Pointer wrapping for the function-pointer-to-pointer decay
            full_type = TypeSpecifier::Pointer(Box::new(full_type), AddressSpace::Default);
            return (full_type, Some(fptr_params));
        }

        // Not a function pointer - apply all derivations normally.
        // Apply pointers
        for d in pderived {
            if let DerivedDeclarator::Pointer = d {
                full_type = TypeSpecifier::Pointer(Box::new(full_type), AddressSpace::Default);
            }
        }
        // Collect array dimensions
        let array_dims: Vec<_> = pderived.iter().filter_map(|d| {
            if let DerivedDeclarator::Array(size) = d {
                Some(size.clone())
            } else {
                None
            }
        }).collect();
        // Array params: outermost dimension decays to pointer
        if !array_dims.is_empty() {
            for dim in array_dims.iter().skip(1).rev() {
                full_type = TypeSpecifier::Array(Box::new(full_type), dim.clone());
            }
            full_type = TypeSpecifier::Pointer(Box::new(full_type), AddressSpace::Default);
        }
        // Function params (bare function names) decay to pointers
        for d in pderived {
            if let DerivedDeclarator::Function(_, _) = d {
                full_type = TypeSpecifier::Pointer(Box::new(full_type), AddressSpace::Default);
            }
        }
        (full_type, None)
    }

    /// Parse the rest of a declaration (not a function definition).
    fn parse_declaration_rest(
        &mut self,
        type_spec: TypeSpecifier,
        name: Option<Rc<str>>,
        derived: Vec<DerivedDeclarator>,
        start: crate::common::source::Span,
        mut ctx: DeclContext,
    ) -> Option<ExternalDecl> {
        let mut declarators = Vec::new();
        let init = if self.consume_if(&TokenKind::Assign) {
            Some(self.parse_initializer())
        } else {
            None
        };
        let section = ctx.attrs.section.clone();
        declarators.push(InitDeclarator {
            name: name.unwrap_or_else(|| Rc::from("")),
            derived,
            init,
            attrs: ctx.attrs,
            span: start,
        });

        let (extra_ctor, extra_dtor, _, extra_common, extra_aligned, extra_asm_reg) = self.parse_asm_and_attributes();
        // Merge post-declarator attributes into the most recently pushed declarator.
        // last_mut() is safe because we just pushed above.
        let last_decl = declarators.last_mut().expect("declarator just pushed");
        if let Some(reg) = extra_asm_reg {
            last_decl.attrs.asm_register = Some(reg);
        }
        if extra_ctor {
            last_decl.attrs.set_constructor(true);
        }
        if extra_dtor {
            last_decl.attrs.set_destructor(true);
        }
        // Merge weak/alias/visibility from post-declarator attributes
        if self.attrs.parsing_weak() {
            last_decl.attrs.set_weak(true);
        }
        if let Some(ref target) = self.attrs.parsing_alias_target {
            last_decl.attrs.alias_target = Some(target.clone());
        }
        if let Some(ref vis) = self.attrs.parsing_visibility {
            last_decl.attrs.visibility = Some(vis.clone());
        }
        if let Some(ref sect) = self.attrs.parsing_section {
            last_decl.attrs.section = Some(sect.clone());
        }
        if let Some(ref cleanup) = self.attrs.parsing_cleanup_fn {
            last_decl.attrs.cleanup_fn = Some(cleanup.clone());
        }
        if self.attrs.parsing_used() {
            last_decl.attrs.set_used(true);
        }
        if self.attrs.parsing_noreturn() {
            last_decl.attrs.set_noreturn(true);
        }
        if self.attrs.parsing_fastcall() {
            last_decl.attrs.set_fastcall(true);
        }
        if self.attrs.parsing_naked() {
            last_decl.attrs.set_naked(true);
        }
        if self.attrs.parsing_error_attr() {
            last_decl.attrs.set_error_attr(true);
        }
        self.attrs.set_weak(false);
        self.attrs.parsing_alias_target = None;
        self.attrs.parsing_visibility = None;
        self.attrs.parsing_section = None;
        self.attrs.set_error_attr(false);
        self.attrs.set_noreturn(false);
        self.attrs.parsing_cleanup_fn = None;
        self.attrs.set_used(false);
        self.attrs.set_fastcall(false);
        self.attrs.set_naked(false);
        ctx.is_common = ctx.is_common || extra_common;
        if let Some(a) = extra_aligned {
            ctx.alignment = Some(ctx.alignment.map_or(a, |prev| prev.max(a)));
        }

        // Parse additional declarators separated by commas
        while self.consume_if(&TokenKind::Comma) {
            let (dname, dderived) = self.parse_declarator();
            let (d_ctor, d_dtor, _, d_common, _, d_asm_reg) = self.parse_asm_and_attributes();
            ctx.is_common = ctx.is_common || d_common;
            let d_weak = self.attrs.parsing_weak();
            let d_alias = self.attrs.parsing_alias_target.take();
            let d_vis = self.attrs.parsing_visibility.take()
                .or_else(|| self.pragma_default_visibility.clone());
            // Per-declarator section attribute overrides, otherwise inherit from
            // the declaration-level attribute (e.g. __attribute__((section(".x"))) int a, b;)
            let d_section = self.attrs.parsing_section.take().or_else(|| section.clone());
            let d_cleanup_fn = self.attrs.parsing_cleanup_fn.take();
            let d_used = self.attrs.parsing_used();
            let d_noreturn = self.attrs.parsing_noreturn();
            let d_error_attr = self.attrs.parsing_error_attr();
            self.attrs.set_weak(false);
            self.attrs.set_used(false);
            self.attrs.set_fastcall(false);
            self.attrs.set_naked(false);
            self.attrs.set_noreturn(false);
            self.attrs.set_error_attr(false);
            let dinit = if self.consume_if(&TokenKind::Assign) {
                Some(self.parse_initializer())
            } else {
                None
            };
            let d_fastcall = self.attrs.parsing_fastcall();
            declarators.push(InitDeclarator {
                name: dname.unwrap_or_else(|| Rc::from("")),
                derived: dderived,
                init: dinit,
                attrs: {
                    let mut da = DeclAttributes::default();
                    da.set_constructor(d_ctor);
                    da.set_destructor(d_dtor);
                    da.set_weak(d_weak);
                    da.set_error_attr(d_error_attr);
                    da.set_noreturn(d_noreturn);
                    da.set_used(d_used);
                    da.set_fastcall(d_fastcall);
                    da.alias_target = d_alias;
                    da.visibility = d_vis;
                    da.section = d_section;
                    da.asm_register = d_asm_reg;
                    da.cleanup_fn = d_cleanup_fn;
                    da
                },
                span: start,
            });
            let (_, skip_aligned, skip_asm_reg) = self.skip_asm_and_attributes();
            if let Some(reg) = skip_asm_reg {
                declarators.last_mut().expect("declarator just pushed").attrs.asm_register = Some(reg);
            }
            if let Some(a) = skip_aligned {
                ctx.alignment = Some(ctx.alignment.map_or(a, |prev| prev.max(a)));
            }
        }

        // Register typedef names
        let is_typedef = self.attrs.parsing_typedef();
        let is_transparent_union = self.attrs.parsing_transparent_union();
        self.attrs.set_transparent_union(false);
        self.register_typedefs(&declarators);

        self.expect_after(&TokenKind::Semicolon, "after declaration");
        let mut d = Declaration::new(
            type_spec,
            declarators,
            ctx.alignment, ctx.alignas_type, ctx.alignment_sizeof_type,
            self.attrs.parsing_address_space,
            self.attrs.parsing_vector_size.take(),
            self.attrs.parsing_ext_vector_nelem.take(),
            start,
        );
        d.set_static(self.attrs.parsing_static());
        d.set_extern(self.attrs.parsing_extern());
        d.set_typedef(is_typedef);
        d.set_const(self.attrs.parsing_const());
        d.set_volatile(self.attrs.parsing_volatile());
        d.set_common(ctx.is_common);
        d.set_thread_local(self.attrs.parsing_thread_local());
        d.set_transparent_union(is_transparent_union);
        d.set_inline(self.attrs.parsing_inline());
        Some(ExternalDecl::Declaration(d))
    }

    pub(super) fn parse_local_declaration(&mut self) -> Option<Declaration> {
        let start = self.peek_span();
        // Selective reset: only storage-class/qualifier flags need clearing here.
        // GCC attribute flags (constructor, weak, etc.) may carry over from the
        // type-specifier parse that already ran before we entered this path.
        // parse_external_decl does a full `attrs = default()` because it starts
        // a completely new declaration context.
        self.attrs.set_static(false);
        self.attrs.set_extern(false);
        self.attrs.set_thread_local(false);
        self.attrs.set_typedef(false);
        self.attrs.set_inline(false);
        self.attrs.set_const(false);
        self.attrs.set_volatile(false);
        self.attrs.parsing_address_space = AddressSpace::Default;
        let type_spec = self.parse_type_specifier()?;

        self.consume_post_type_qualifiers();

        let is_static = self.attrs.parsing_static();
        let is_extern = self.attrs.parsing_extern();

        let mut declarators = Vec::new();

        // Handle bare type with semicolon (struct/enum/union definition)
        if matches!(self.peek(), TokenKind::Semicolon) {
            self.advance();
            let mut d = Declaration::new(
                type_spec,
                declarators,
                None, None, None,
                self.attrs.parsing_address_space,
                self.attrs.parsing_vector_size.take(),
                self.attrs.parsing_ext_vector_nelem.take(),
                start,
            );
            d.set_static(is_static);
            d.set_extern(is_extern);
            d.set_typedef(self.attrs.parsing_typedef());
            d.set_const(self.attrs.parsing_const());
            d.set_volatile(self.attrs.parsing_volatile());
            d.set_thread_local(self.attrs.parsing_thread_local());
            return Some(d);
        }

        let mut mode_kind: Option<ModeKind> = None;
        let mut alignment: Option<usize> = None;
        loop {
            let (name, derived, decl_mode, _, decl_aligned, _) = self.parse_declarator_with_attrs();
            let (skip_mode, skip_aligned, skip_asm_reg) = self.skip_asm_and_attributes();
            let local_cleanup_fn = self.attrs.parsing_cleanup_fn.take();
            let local_section = self.attrs.parsing_section.take();
            mode_kind = mode_kind.or(decl_mode).or(skip_mode);
            // Merge alignment from declarator and post-declarator attributes
            for a in [decl_aligned, skip_aligned].iter().copied().flatten() {
                alignment = Some(alignment.map_or(a, |prev| prev.max(a)));
            }
            let init = if self.consume_if(&TokenKind::Assign) {
                Some(self.parse_initializer())
            } else {
                None
            };
            declarators.push(InitDeclarator {
                name: name.unwrap_or_else(|| Rc::from("")),
                derived,
                init,
                attrs: {
                    let mut da = DeclAttributes::default();
                    da.section = local_section;
                    da.asm_register = skip_asm_reg;
                    da.cleanup_fn = local_cleanup_fn;
                    da.set_used(self.attrs.parsing_used());
                    da.set_noreturn(self.attrs.parsing_noreturn());
                    da.set_constructor(self.attrs.parsing_constructor());
                    da.set_destructor(self.attrs.parsing_destructor());
                    da.set_weak(self.attrs.parsing_weak());
                    da.set_fastcall(self.attrs.parsing_fastcall());
                    da.set_naked(self.attrs.parsing_naked());
                    da.set_error_attr(self.attrs.parsing_error_attr());
                    if let Some(ref target) = self.attrs.parsing_alias_target {
                        da.alias_target = Some(target.clone());
                    }
                    if let Some(ref vis) = self.attrs.parsing_visibility {
                        da.visibility = Some(vis.clone());
                    }
                    da
                },
                span: start,
            });
            let (_, post_init_aligned, _post_asm_reg) = self.skip_asm_and_attributes();
            if let Some(a) = post_init_aligned {
                alignment = Some(alignment.map_or(a, |prev| prev.max(a)));
            }
            if !self.consume_if(&TokenKind::Comma) {
                break;
            }
        }

        // Apply __attribute__((mode(...))): transform type to specified bit-width
        let type_spec = if let Some(mk) = mode_kind {
            mk.apply(type_spec)
        } else {
            type_spec
        };

        // Register typedef names or shadow them for variable declarations
        let is_typedef = self.attrs.parsing_typedef();
        if self.attrs.parsing_typedef() {
            for decl in &declarators {
                if !decl.name.is_empty() {
                    self.typedefs.insert(decl.name.clone());
                    self.shadowed_typedefs.remove(&decl.name);
                }
            }
            self.attrs.set_typedef(false);
        } else {
            for decl in &declarators {
                if !decl.name.is_empty() && self.typedefs.contains(&decl.name) {
                    self.shadowed_typedefs.insert(decl.name.clone());
                }
            }
        }

        self.expect_after(&TokenKind::Semicolon, "after declaration");
        // Merge alignment from _Alignas (captured in parsed_alignas during type specifier parsing)
        // with alignment from __attribute__((aligned(N))) on declarators
        if let Some(a) = self.attrs.parsed_alignas.take() {
            alignment = Some(alignment.map_or(a, |prev| prev.max(a)));
        }
        let alignas_type = self.attrs.parsed_alignas_type.take();
        let alignment_sizeof_type = self.attrs.parsed_alignment_sizeof_type.take();
        let is_transparent_union = self.attrs.parsing_transparent_union();
        self.attrs.set_transparent_union(false);
        let mut d = Declaration::new(
            type_spec,
            declarators,
            alignment, alignas_type, alignment_sizeof_type,
            self.attrs.parsing_address_space,
            self.attrs.parsing_vector_size.take(),
            self.attrs.parsing_ext_vector_nelem.take(),
            start,
        );
        d.set_static(is_static);
        d.set_extern(is_extern);
        d.set_typedef(is_typedef);
        d.set_const(self.attrs.parsing_const());
        d.set_volatile(self.attrs.parsing_volatile());
        d.set_thread_local(self.attrs.parsing_thread_local());
        d.set_transparent_union(is_transparent_union);
        Some(d)
    }

    /// Parse an initializer: either a braced initializer list or a single expression.
    pub(super) fn parse_initializer(&mut self) -> Initializer {
        if matches!(self.peek(), TokenKind::LBrace) {
            let open = self.peek_span();
            self.advance();
            let mut items = Vec::new();
            while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
                let mut designators = Vec::new();
                // Parse designators: [idx] and .field
                loop {
                    if matches!(self.peek(), TokenKind::LBracket) {
                        let open = self.peek_span();
                        self.advance();
                        let lo = self.parse_expr();
                        if self.consume_if(&TokenKind::Ellipsis) {
                            // GCC range designator: [lo ... hi]
                            let hi = self.parse_expr();
                            self.expect_closing(&TokenKind::RBracket, open);
                            designators.push(Designator::Range(lo, hi));
                        } else {
                            self.expect_closing(&TokenKind::RBracket, open);
                            designators.push(Designator::Index(lo));
                        }
                    } else if self.consume_if(&TokenKind::Dot) {
                        if let TokenKind::Identifier(name) = self.peek() {
                            let name = name.clone();
                            self.advance();
                            designators.push(Designator::Field(name));
                        }
                    } else {
                        break;
                    }
                }
                // GNU old-style designator: field: value
                if designators.is_empty() {
                    if let TokenKind::Identifier(name) = self.peek() {
                        let name = name.clone();
                        if self.pos + 1 < self.tokens.len() && matches!(self.tokens[self.pos + 1].kind, TokenKind::Colon) {
                            self.advance(); // consume identifier
                            self.advance(); // consume colon
                            designators.push(Designator::Field(name));
                        }
                    }
                }
                if !designators.is_empty()
                    && matches!(self.peek(), TokenKind::Assign) {
                        self.advance();
                    }
                let init = self.parse_initializer();
                items.push(InitializerItem { designators, init });
                if !self.consume_if(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect_closing(&TokenKind::RBrace, open);
            // Expand range designators [lo ... hi] into individual Index items
            let enums = if self.enum_constants.is_empty() {
                None
            } else {
                Some(&self.enum_constants)
            };
            let items = Self::expand_range_designators(items, enums);
            Initializer::List(items)
        } else {
            Initializer::Expr(self.parse_assignment_expr())
        }
    }

    /// Expand GCC range designators `[lo ... hi] = val` into multiple items
    /// `[lo] = val, [lo+1] = val, ..., [hi] = val` so downstream code only
    /// sees `Designator::Index`.
    fn expand_range_designators(
        items: Vec<InitializerItem>,
        enum_consts: Option<&FxHashMap<Rc<str>, i64>>,
    ) -> Vec<InitializerItem> {
        let mut result = Vec::with_capacity(items.len());
        for item in items {
            if let Some(range_pos) = item.designators.iter().position(|d| matches!(d, Designator::Range(_, _))) {
                if let Designator::Range(ref lo_expr, ref hi_expr) = item.designators[range_pos] {
                    // Try to evaluate lo and hi as integer constants
                    let lo = Self::eval_const_int_expr_with_enums(lo_expr, enum_consts, None);
                    let hi = Self::eval_const_int_expr_with_enums(hi_expr, enum_consts, None);
                    if let (Some(lo_val), Some(hi_val)) = (lo, hi) {
                        for idx in lo_val..=hi_val {
                            let mut new_desigs = item.designators.clone();
                            new_desigs[range_pos] = Designator::Index(
                                Expr::IntLiteral(idx, Span::dummy())
                            );
                            result.push(InitializerItem {
                                designators: new_desigs,
                                init: item.init.clone(),
                            });
                        }
                        continue;
                    }
                }
                // If we can't evaluate, keep as-is (will error later)
                result.push(item);
            } else {
                result.push(item);
            }
        }
        result
    }

    /// Constant expression evaluator for compile-time integer contexts.
    /// Handles integer/char literals, all arithmetic/bitwise/shift/logical/relational
    /// binary ops, unary ops (neg, bitwise not, logical not), and ternary expressions.
    /// Used for range designator bounds, alignment expressions, etc.
    /// The optional `enum_consts` map allows resolving enum constant identifiers.
    pub(super) fn eval_const_int_expr(expr: &Expr) -> Option<i64> {
        Self::eval_const_int_expr_with_enums(expr, None, None)
    }

    /// Evaluate a constant integer expression, with access to known enum constants
    /// and struct/union tag alignments for resolving tag-only __alignof__ references.
    pub(super) fn eval_const_int_expr_with_enums(
        expr: &Expr,
        enum_consts: Option<&FxHashMap<Rc<str>, i64>>,
        tag_aligns: Option<&FxHashMap<Rc<str>, usize>>,
    ) -> Option<i64> {
        match expr {
            Expr::IntLiteral(val, _) => Some(*val),
            Expr::UIntLiteral(val, _) => Some(*val as i64),
            Expr::LongLiteral(val, _) => Some(*val),
            Expr::ULongLiteral(val, _) => Some(*val as i64),
            Expr::LongLongLiteral(val, _) => Some(*val),
            Expr::ULongLongLiteral(val, _) => Some(*val as i64),
            Expr::CharLiteral(val, _) => Some(*val as i64),
            // Identifiers: look up enum constants if available
            Expr::Identifier(name, _) => {
                enum_consts.and_then(|m| m.get(&**name).copied())
            }
            Expr::BinaryOp(op, lhs, rhs, _) => {
                let l = Self::eval_const_int_expr_with_enums(lhs, enum_consts, tag_aligns)?;
                let r = Self::eval_const_int_expr_with_enums(rhs, enum_consts, tag_aligns)?;
                match op {
                    BinOp::Add => Some(l.wrapping_add(r)),
                    BinOp::Sub => Some(l.wrapping_sub(r)),
                    BinOp::Mul => Some(l.wrapping_mul(r)),
                    BinOp::Div if r != 0 => Some(l.wrapping_div(r)),
                    BinOp::Mod if r != 0 => Some(l.wrapping_rem(r)),
                    BinOp::Shl => Some(l.wrapping_shl(r as u32)),
                    BinOp::Shr => {
                        // Use logical (unsigned) shift when the LHS appears unsigned
                        if Self::is_unsigned_int_expr(lhs) {
                            Some((l as u64).wrapping_shr(r as u32) as i64)
                        } else {
                            Some(l.wrapping_shr(r as u32))
                        }
                    }
                    BinOp::BitAnd => Some(l & r),
                    BinOp::BitOr => Some(l | r),
                    BinOp::BitXor => Some(l ^ r),
                    BinOp::Eq => Some(if l == r { 1 } else { 0 }),
                    BinOp::Ne => Some(if l != r { 1 } else { 0 }),
                    BinOp::Lt => {
                        if Self::is_unsigned_int_expr(lhs) || Self::is_unsigned_int_expr(rhs) {
                            Some(if (l as u64) < (r as u64) { 1 } else { 0 })
                        } else {
                            Some(if l < r { 1 } else { 0 })
                        }
                    }
                    BinOp::Le => {
                        if Self::is_unsigned_int_expr(lhs) || Self::is_unsigned_int_expr(rhs) {
                            Some(if (l as u64) <= (r as u64) { 1 } else { 0 })
                        } else {
                            Some(if l <= r { 1 } else { 0 })
                        }
                    }
                    BinOp::Gt => {
                        if Self::is_unsigned_int_expr(lhs) || Self::is_unsigned_int_expr(rhs) {
                            Some(if (l as u64) > (r as u64) { 1 } else { 0 })
                        } else {
                            Some(if l > r { 1 } else { 0 })
                        }
                    }
                    BinOp::Ge => {
                        if Self::is_unsigned_int_expr(lhs) || Self::is_unsigned_int_expr(rhs) {
                            Some(if (l as u64) >= (r as u64) { 1 } else { 0 })
                        } else {
                            Some(if l >= r { 1 } else { 0 })
                        }
                    }
                    BinOp::LogicalAnd => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                    BinOp::LogicalOr => Some(if l != 0 || r != 0 { 1 } else { 0 }),
                    _ => None,
                }
            }
            Expr::UnaryOp(UnaryOp::Neg, inner, _) => {
                Self::eval_const_int_expr_with_enums(inner, enum_consts, tag_aligns).map(|v| v.wrapping_neg())
            }
            Expr::UnaryOp(UnaryOp::BitNot, inner, _) => {
                Self::eval_const_int_expr_with_enums(inner, enum_consts, tag_aligns).map(|v| {
                    let result = !v;
                    // For unsigned int operands, truncate to 32 bits. The evaluator
                    // uses i64 for all values, so ~0u produces i64(-1) (all 64 bits set)
                    // instead of the correct 0xFFFFFFFF (32-bit all-ones).
                    if Self::is_unsigned_int_expr(inner) {
                        result & 0xFFFF_FFFF
                    } else {
                        result
                    }
                })
            }
            Expr::UnaryOp(UnaryOp::LogicalNot, inner, _) => {
                Self::eval_const_int_expr_with_enums(inner, enum_consts, tag_aligns).map(|v| if v == 0 { 1 } else { 0 })
            }
            Expr::UnaryOp(UnaryOp::Plus, inner, _) => {
                Self::eval_const_int_expr_with_enums(inner, enum_consts, tag_aligns)
            }
            Expr::Conditional(cond, then_expr, else_expr, _) => {
                let c = Self::eval_const_int_expr_with_enums(cond, enum_consts, tag_aligns)?;
                if c != 0 {
                    Self::eval_const_int_expr_with_enums(then_expr, enum_consts, tag_aligns)
                } else {
                    Self::eval_const_int_expr_with_enums(else_expr, enum_consts, tag_aligns)
                }
            }
            Expr::Cast(target_type, inner, _) => {
                let val = Self::eval_const_int_expr_with_enums(inner, enum_consts, tag_aligns)?;
                // Apply the cast: truncate to target type's bit width
                if let Some(size) = Self::try_sizeof_type_spec(target_type) {
                    let bits = size * 8;
                    if bits >= 64 && Self::is_unsigned_type_spec(target_type) {
                        // 64-bit unsigned cast: the parser evaluator uses i64 which can't
                        // represent the full unsigned range. Return None to defer to sema.
                        return None;
                    }
                    if bits == 0 || bits >= 64 {
                        Some(val)
                    } else {
                        let mask = (1u64 << bits) - 1;
                        let truncated = (val as u64) & mask;
                        if Self::is_unsigned_type_spec(target_type) {
                            // Unsigned: zero-extend
                            Some(truncated as i64)
                        } else {
                            // Signed: sign-extend
                            let sign_bit = 1u64 << (bits - 1);
                            if truncated & sign_bit != 0 {
                                Some((truncated | !mask) as i64)
                            } else {
                                Some(truncated as i64)
                            }
                        }
                    }
                } else {
                    // Can't determine type size (typedef, struct, etc.) -
                    // return None so static_assert defers to sema-level evaluation
                    None
                }
            }
            Expr::Sizeof(arg, _) => {
                match arg.as_ref() {
                    SizeofArg::Type(ts) => Self::try_sizeof_type_spec(ts).map(|s| s as i64),
                    SizeofArg::Expr(_) => None,
                }
            }
            Expr::Alignof(ts, _) => {
                // Can't resolve typedef alignment at parse time
                if Self::type_spec_has_typedef(ts) { return None; }
                Some(Self::alignof_type_spec(ts, tag_aligns) as i64)
            }
            // __alignof(type) returns preferred alignment (8 for long long/double on i686)
            Expr::GnuAlignof(ts, _) => {
                // Can't resolve typedef alignment at parse time
                if Self::type_spec_has_typedef(ts) { return None; }
                Some(Self::preferred_alignof_type_spec(ts, tag_aligns) as i64)
            }
            // __alignof__(expr): parser-level can't always determine type alignment
            // from an expression, so return None (let sema/lowerer handle it)
            Expr::AlignofExpr(_, _) | Expr::GnuAlignofExpr(_, _) => None,
            _ => None,
        }
    }

    /// Check if an expression has unsigned type.
    /// Used by the parser-level constant evaluator for:
    /// - Truncating bitwise NOT results to the correct width
    /// - Using unsigned comparison semantics for relational operators
    ///
    /// Only detects obvious cases from the AST structure; non-obvious cases are
    /// handled by the sema-level evaluator which has full type information.
    fn is_unsigned_int_expr(expr: &Expr) -> bool {
        match expr {
            Expr::UIntLiteral(..) | Expr::ULongLiteral(..) | Expr::ULongLongLiteral(..) => true,
            Expr::Cast(ts, _, _) => {
                Self::is_unsigned_type_spec(ts)
            }
            // Unary +/- preserve the signedness of the operand.
            // In C, -size_t still has type size_t (unsigned wraparound).
            Expr::UnaryOp(UnaryOp::Plus, inner, _)
            | Expr::UnaryOp(UnaryOp::Neg, inner, _) => Self::is_unsigned_int_expr(inner),
            // Binary ops: unsigned if either operand is unsigned (C promotion)
            Expr::BinaryOp(_, lhs, rhs, _) => {
                Self::is_unsigned_int_expr(lhs) || Self::is_unsigned_int_expr(rhs)
            }
            // sizeof yields size_t (unsigned). _Alignof yields size_t (unsigned).
            // These are unsigned per C11 6.5.3.4 and 6.5.3.
            Expr::Sizeof(..) | Expr::Alignof(..) | Expr::GnuAlignof(..)
            | Expr::AlignofExpr(..) | Expr::GnuAlignofExpr(..) => true,
            _ => false,
        }
    }

    /// Check if a type specifier contains an unresolvable typedef name.
    /// Used to bail out of parser-level alignof/sizeof evaluation when the
    /// actual type can't be determined without typedef resolution.
    fn type_spec_has_typedef(ts: &TypeSpecifier) -> bool {
        match ts {
            TypeSpecifier::TypedefName(_) => true,
            TypeSpecifier::Pointer(inner, _) => Self::type_spec_has_typedef(inner),
            TypeSpecifier::Array(inner, _) => Self::type_spec_has_typedef(inner),
            _ => false,
        }
    }

    /// Consume post-type storage class specifiers and qualifiers.
    /// C allows "struct { int i; } typedef name;" and "char _Alignas(16) x;".
    /// This is shared between parse_external_decl and parse_local_declaration.
    pub(super) fn consume_post_type_qualifiers(&mut self) {
        loop {
            match self.peek() {
                TokenKind::Typedef => { self.advance(); self.attrs.set_typedef(true); }
                TokenKind::Static => { self.advance(); self.attrs.set_static(true); }
                TokenKind::Extern => { self.advance(); self.attrs.set_extern(true); }
                TokenKind::Const => { self.advance(); self.attrs.set_const(true); }
                TokenKind::Volatile => { self.advance(); self.attrs.set_volatile(true); }
                TokenKind::Restrict
                | TokenKind::Inline | TokenKind::Register | TokenKind::Auto => { self.advance(); }
                TokenKind::SegGs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegGs; }
                TokenKind::SegFs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegFs; }
                TokenKind::Alignas => {
                    self.advance();
                    if let Some(align) = self.parse_alignas_argument() {
                        self.attrs.parsed_alignas = Some(self.attrs.parsed_alignas.map_or(align, |prev| prev.max(align)));
                    }
                }
                TokenKind::Attribute => {
                    let (_, aligned, _, _) = self.parse_gcc_attributes();
                    if let Some(a) = aligned {
                        self.attrs.parsed_alignas = Some(self.attrs.parsed_alignas.map_or(a, |prev| prev.max(a)));
                    }
                }
                TokenKind::Extension => { self.advance(); }
                _ => break,
            }
        }
    }

    /// Register typedef names from declarators, if parsing_typedef is set.
    fn register_typedefs(&mut self, declarators: &[InitDeclarator]) {
        if self.attrs.parsing_typedef() {
            for decl in declarators {
                if !decl.name.is_empty() {
                    self.typedefs.insert(decl.name.clone());
                }
            }
            self.attrs.set_typedef(false);
        }
    }

    /// Check whether an expression contains identifiers that are NOT enum constants.
    /// Such identifiers are variables/parameters, which make the expression not a
    /// valid integer constant expression (C11 6.6).
    fn expr_has_non_const_identifier(
        expr: &Expr,
        enum_consts: Option<&FxHashMap<Rc<str>, i64>>,
        unevaluable_consts: Option<&FxHashSet<Rc<str>>>,
    ) -> bool {
        match expr {
            Expr::Identifier(name, _) => {
                // It's a variable/parameter reference if not in enum constants
                // (either evaluated or unevaluable)
                let in_evaluated = enum_consts.is_some_and(|m| m.contains_key(&**name));
                let in_unevaluable = unevaluable_consts.is_some_and(|m| m.contains(&**name));
                !(in_evaluated || in_unevaluable)
            }
            Expr::BinaryOp(_, lhs, rhs, _) => {
                Self::expr_has_non_const_identifier(lhs, enum_consts, unevaluable_consts)
                    || Self::expr_has_non_const_identifier(rhs, enum_consts, unevaluable_consts)
            }
            Expr::UnaryOp(_, inner, _) => {
                Self::expr_has_non_const_identifier(inner, enum_consts, unevaluable_consts)
            }
            Expr::Conditional(cond, then_expr, else_expr, _) => {
                Self::expr_has_non_const_identifier(cond, enum_consts, unevaluable_consts)
                    || Self::expr_has_non_const_identifier(then_expr, enum_consts, unevaluable_consts)
                    || Self::expr_has_non_const_identifier(else_expr, enum_consts, unevaluable_consts)
            }
            Expr::GnuConditional(cond, fallback, _) => {
                Self::expr_has_non_const_identifier(cond, enum_consts, unevaluable_consts)
                    || Self::expr_has_non_const_identifier(fallback, enum_consts, unevaluable_consts)
            }
            Expr::Cast(_, inner, _) => {
                Self::expr_has_non_const_identifier(inner, enum_consts, unevaluable_consts)
            }
            Expr::Comma(lhs, rhs, _) => {
                Self::expr_has_non_const_identifier(lhs, enum_consts, unevaluable_consts)
                    || Self::expr_has_non_const_identifier(rhs, enum_consts, unevaluable_consts)
            }
            // Sizeof, literals, builtin calls, function calls, etc. that aren't
            // evaluable still don't make the expression "definitely non-constant"
            // from a variable-reference perspective. The eval_const_int_expr_with_enums
            // function returning None for these cases will cause silent acceptance.
            _ => false,
        }
    }

    /// Parse and evaluate `_Static_assert(constant-expr, "message")` or
    /// the C23 single-argument form `_Static_assert(constant-expr)`.
    ///
    /// If the constant expression evaluates to zero, emits a compile error
    /// with the provided message (or a default message for the single-arg form).
    /// If the expression cannot be evaluated at compile time, silently accepts
    /// it (matching GCC behavior for complex expressions).
    pub(super) fn parse_static_assert(&mut self) {
        let assert_span = self.peek_span();
        self.advance(); // consume _Static_assert
        let open = self.peek_span();
        self.expect_context(&TokenKind::LParen, "after '_Static_assert'");

        // Parse the constant expression
        let expr = self.parse_assignment_expr();

        // Parse optional message: ", string-literal(s)"
        // Adjacent string literals are concatenated per C standard (translation phase 6).
        // The kernel uses patterns like: "min" "(" "x" ", " "y" ") error"
        let message = if self.consume_if(&TokenKind::Comma) {
            let mut msg = String::new();
            let mut found_string = false;
            while let TokenKind::StringLiteral(s) = self.peek().clone() {
                msg.push_str(&s);
                found_string = true;
                self.advance();
            }
            if found_string { Some(msg) } else { None }
        } else {
            None
        };

        self.expect_closing(&TokenKind::RParen, open);
        self.consume_if(&TokenKind::Semicolon);

        // Evaluate the constant expression
        let enums = if self.enum_constants.is_empty() {
            None
        } else {
            Some(&self.enum_constants)
        };
        let tag_aligns = if self.struct_tag_alignments.is_empty() {
            None
        } else {
            Some(&self.struct_tag_alignments)
        };
        let unevaluable = if self.unevaluable_enum_constants.is_empty() {
            None
        } else {
            Some(&self.unevaluable_enum_constants)
        };
        if let Some(value) = Self::eval_const_int_expr_with_enums(&expr, enums, tag_aligns) {
            if value == 0 {
                // Static assertion failed
                let msg = if let Some(ref m) = message {
                    format!("static assertion failed: {}", m)
                } else {
                    "static assertion failed".to_string()
                };
                self.emit_error(msg, assert_span);
            }
        } else if Self::expr_has_non_const_identifier(&expr, enums, unevaluable) {
            // The expression references variables or non-enum identifiers, which
            // means it's definitely not a valid integer constant expression (C11 6.6).
            self.emit_error("expression in static assertion is not an integer constant expression", assert_span);
        }
        // If we can't evaluate the expression but it doesn't contain variable
        // references (e.g. sizeof, offsetof, compiler builtins), silently accept.
        // This matches real-world behavior where complex expressions may not be
        // evaluable at parse time but are valid constant expressions.
    }

}
