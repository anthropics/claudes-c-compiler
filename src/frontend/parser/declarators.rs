// Declarator parsing: handles the C declarator syntax (the part after the type
// specifier that defines the name and type modifiers like pointers, arrays,
// and function parameters).
//
// C declarators follow an "inside-out" rule: int (*fp)(int) means fp is a
// pointer to a function returning int, read from the name outward. This module
// handles the recursive parsing needed for this grammar.

use std::rc::Rc;
use crate::common::types::AddressSpace;
use crate::frontend::lexer::token::TokenKind;
use super::ast::*;
use super::parse::{ModeKind, Parser};

/// Result of parsing a parenthesized abstract declarator.
pub(super) enum ParenAbstractDecl {
    /// Simple pointer/array grouping: (*), (**), (*[3][4])
    Simple {
        ptr_depth: u32,
        array_dims: Vec<Option<Box<Expr>>>,
    },
    /// Nested function pointer inside outer parens: (*(*)(params))
    /// Used for types like void(*(*)(void*))(void) - pointer to function
    /// returning a function pointer.
    NestedFnPtr {
        /// Outer pointer depth (the * before the inner fn ptr group)
        outer_ptr_depth: u32,
        /// Inner pointer depth (the * in the inner-most (*) group)
        inner_ptr_depth: u32,
        /// Parameter list of the inner function pointer
        inner_params: Vec<ParamDecl>,
        inner_variadic: bool,
    },
}

impl Parser {
    pub(super) fn parse_declarator(&mut self) -> (Option<Rc<str>>, Vec<DerivedDeclarator>) {
        let (name, derived, _, _, _, _) = self.parse_declarator_with_attrs();
        (name, derived)
    }

    /// Parse a declarator, also returning attribute info:
    /// (name, derived, mode_kind, has_common, aligned_value, is_packed)
    pub(super) fn parse_declarator_with_attrs(&mut self) -> (Option<Rc<str>>, Vec<DerivedDeclarator>, Option<ModeKind>, bool, Option<usize>, bool) {
        let mut derived = Vec::new();

        let mut pre_aligned: Option<usize> = None;
        let mut is_packed = false;
        let (pre_packed, pre_align, _, _) = self.parse_gcc_attributes();
        is_packed = is_packed || pre_packed;
        if let Some(a) = pre_align {
            pre_aligned = Some(pre_aligned.map_or(a, |prev: usize| prev.max(a)));
        }

        // Parse pointer(s) with optional qualifiers and attributes
        while self.consume_if(&TokenKind::Star) {
            derived.push(DerivedDeclarator::Pointer);
            self.skip_cv_qualifiers();
            self.skip_gcc_extensions();
        }

        // Parse the direct-declarator part
        let (name, inner_derived) = if let TokenKind::Identifier(ref n) = self.peek() {
            let n = n.clone();
            self.advance();
            (Some(n), Vec::new())
        } else if matches!(self.peek(), TokenKind::LParen) && self.is_paren_declarator() {
            let save = self.pos;
            self.advance(); // consume '('
            let (inner_name, inner_derived) = self.parse_declarator();
            if !self.consume_if(&TokenKind::RParen) {
                self.pos = save;
                (None, Vec::new())
            } else {
                (inner_name, inner_derived)
            }
        } else {
            (None, Vec::new())
        };

        // Parse outer suffixes: array dimensions and function params
        let mut outer_suffixes = Vec::new();
        loop {
            match self.peek() {
                TokenKind::LBracket => {
                    let open_bracket = self.peek_span();
                    self.advance();
                    // C99 array parameter declarators can have qualifiers and 'static':
                    // [static restrict const 10], [restrict n], [const], etc.
                    // Skip all qualifiers and 'static' before the size expression.
                    self.skip_array_qualifiers();
                    let size = if matches!(self.peek(), TokenKind::RBracket) {
                        None
                    } else if matches!(self.peek(), TokenKind::Star)
                        && self.pos + 1 < self.tokens.len()
                        && matches!(self.tokens[self.pos + 1].kind, TokenKind::RBracket) {
                        // C99 VLA star syntax: [*] or [const *] means unspecified VLA size
                        self.advance(); // consume '*'
                        None
                    } else {
                        Some(Box::new(self.parse_expr()))
                    };
                    self.expect_closing(&TokenKind::RBracket, open_bracket);
                    outer_suffixes.push(DerivedDeclarator::Array(size));
                }
                TokenKind::LParen => {
                    let (params, variadic) = self.parse_param_list();
                    outer_suffixes.push(DerivedDeclarator::Function(params, variadic));
                }
                _ => break,
            }
        }

        // Combine using inside-out rule
        let combined = self.combine_declarator_parts(derived, inner_derived, outer_suffixes);

        let (post_packed, post_aligned, mode_kind, has_common) = self.parse_gcc_attributes();
        is_packed = is_packed || post_packed;
        let aligned = match (pre_aligned, post_aligned) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        (name, combined, mode_kind, has_common, aligned, is_packed)
    }

    /// Determine if a '(' starts a parenthesized declarator vs. a parameter list.
    pub(super) fn is_paren_declarator(&self) -> bool {
        if self.pos + 1 >= self.tokens.len() {
            return false;
        }
        match &self.tokens[self.pos + 1].kind {
            TokenKind::Star | TokenKind::Caret => true,
            TokenKind::LParen | TokenKind::LBracket => true,
            TokenKind::Attribute | TokenKind::Extension => true,
            TokenKind::Identifier(name) => {
                // Typedef name -> parameter list; regular name -> declarator
                !self.typedefs.contains(name) || self.shadowed_typedefs.contains(name)
            }
            TokenKind::RParen | TokenKind::Ellipsis => false,
            TokenKind::Void | TokenKind::Char | TokenKind::Short | TokenKind::Int |
            TokenKind::Long | TokenKind::Float | TokenKind::Double | TokenKind::Signed |
            TokenKind::Unsigned | TokenKind::Struct | TokenKind::Union | TokenKind::Enum |
            TokenKind::Const | TokenKind::Volatile | TokenKind::Static | TokenKind::Extern |
            TokenKind::Register | TokenKind::Typedef | TokenKind::Inline | TokenKind::Bool |
            TokenKind::Typeof | TokenKind::Noreturn | TokenKind::Restrict | TokenKind::Complex |
            TokenKind::Atomic | TokenKind::Auto | TokenKind::Alignas |
            TokenKind::Builtin => false,
            _ => false,
        }
    }

    /// Combine declarator parts using C's inside-out rule.
    ///
    /// For `int (*fp)(int)`:
    ///   outer_pointers: []
    ///   inner_derived: [Pointer]
    ///   outer_suffixes: [Function([int])]
    ///   Result: [Pointer, FunctionPointer([int])]
    fn combine_declarator_parts(
        &self,
        mut outer_pointers: Vec<DerivedDeclarator>,
        inner_derived: Vec<DerivedDeclarator>,
        outer_suffixes: Vec<DerivedDeclarator>,
    ) -> Vec<DerivedDeclarator> {
        if inner_derived.is_empty() && outer_suffixes.is_empty() {
            return outer_pointers;
        }

        if inner_derived.is_empty() {
            outer_pointers.extend(outer_suffixes);
            return outer_pointers;
        }

        // Check for function pointer: inner has Pointer(s), outer starts with Function
        let inner_only_ptr_and_array = inner_derived.iter().all(|d|
            matches!(d, DerivedDeclarator::Pointer | DerivedDeclarator::Array(_)));
        let inner_has_pointer = inner_derived.iter().any(|d| matches!(d, DerivedDeclarator::Pointer));
        let outer_starts_with_function = matches!(outer_suffixes.first(), Some(DerivedDeclarator::Function(_, _)));

        if inner_only_ptr_and_array && inner_has_pointer && outer_starts_with_function
            && outer_suffixes.len() == 1
        {
            // In C's inside-out reading of declarations:
            //   outer_pointers = return-type pointers (e.g., the `*` in `int *`)
            //   inner_derived = declarator pointers/arrays from inside parens
            //   outer_suffixes = [Function(...)]
            //
            // The LAST Pointer in inner_derived is the function pointer syntax
            // marker (the `(*` in `(*fp)`). Any preceding Pointers in inner_derived
            // are extra indirection levels (e.g., `(**fpp)` has two Pointers — the
            // first is extra indirection, the second is the syntax marker).
            //
            // Layout: outer_pointers ++ [syntax_marker_ptr, FunctionPointer] ++ extra_inner_ptrs ++ inner_arrays
            //
            // Example: `int *(*fp)(int)` → outer=[Pointer], inner=[Pointer]
            //   result: [Pointer, Pointer, FunctionPointer] — one return-type ptr, one syntax ptr
            //
            // Example: `int (**fpp)(int,int)` → outer=[], inner=[Pointer, Pointer]
            //   result: [Pointer, FunctionPointer, Pointer] — syntax ptr + fptr, then extra indirection
            //
            // Example: `int *(**fpp)(int)` → outer=[Pointer], inner=[Pointer, Pointer]
            //   result: [Pointer, Pointer, FunctionPointer, Pointer]
            let mut result = outer_pointers;

            // Count inner pointers. The last one is the function pointer syntax marker.
            // All others are extra indirection levels placed AFTER the FunctionPointer.
            let inner_ptr_count = inner_derived.iter()
                .filter(|d| matches!(d, DerivedDeclarator::Pointer))
                .count();
            let extra_indirection_ptrs = if inner_ptr_count > 0 { inner_ptr_count - 1 } else { 0 };

            // Emit the function pointer syntax marker + FunctionPointer
            result.push(DerivedDeclarator::Pointer);
            if let Some(DerivedDeclarator::Function(params, variadic)) = outer_suffixes.into_iter().next() {
                result.push(DerivedDeclarator::FunctionPointer(params, variadic));
            }

            // Emit extra indirection Pointers (beyond the syntax marker)
            for _ in 0..extra_indirection_ptrs {
                result.push(DerivedDeclarator::Pointer);
            }

            // Emit inner arrays (for array of function pointers, e.g., `int (*fps[10])(int)`)
            for d in &inner_derived {
                if matches!(d, DerivedDeclarator::Array(_)) {
                    result.push(d.clone());
                }
            }

            return result;
        }

        // Check for pointer-to-array: inner contains Pointer(s) with optional arrays,
        // outer is all Array(s). Handle inside-out rule correctly:
        //
        // For `int (*p)[3][6]`: inner=[Pointer], outer=[Array(3), Array(6)]
        //   Split inner at pointer: pre_ptr=[], post_ptr=[]
        //   Result: [] ++ [Array(3), Array(6)] ++ [Pointer] ++ [] = [Array(3), Array(6), Pointer]
        //
        // For `int ((*(p))[3])[6]`: inner=[Array(3), Pointer], outer=[Array(6)]
        //   Split inner at pointer: pre_ptr=[Array(3)], post_ptr=[]
        //   Result: [Array(3)] ++ [Array(6)] ++ [Pointer] ++ [] = [Array(3), Array(6), Pointer]
        //
        // For `int (*ptrs[2])[4]`: inner=[Pointer, Array(2)], outer=[Array(4)]
        //   Split inner at pointer: pre_ptr=[], post_ptr=[Array(2)]
        //   Result: [] ++ [Array(4)] ++ [Pointer] ++ [Array(2)] = [Array(4), Pointer, Array(2)]
        let outer_only_arrays = outer_suffixes.iter().all(|d| matches!(d, DerivedDeclarator::Array(_)));
        if inner_only_ptr_and_array && inner_has_pointer && outer_only_arrays {
            // Split inner_derived at the last Pointer:
            // - pre_ptr_arrays: arrays before the last pointer (part of pointee type)
            // - post_ptr_arrays: arrays after the last pointer (variable's own array dimensions)
            let last_ptr_idx = inner_derived.iter().rposition(|d| matches!(d, DerivedDeclarator::Pointer))
                .expect("inner_has_pointer is true, so a Pointer must exist");
            let mut result = outer_pointers;
            // 1. Arrays from inner that come before the pointer (pointee array dimensions)
            for d in &inner_derived[..last_ptr_idx] {
                if matches!(d, DerivedDeclarator::Array(_)) {
                    result.push(d.clone());
                }
            }
            // 2. Outer array suffixes (also pointee dimensions)
            result.extend(outer_suffixes);
            // 3. Pointer(s)
            for d in &inner_derived[..=last_ptr_idx] {
                if matches!(d, DerivedDeclarator::Pointer) {
                    result.push(d.clone());
                }
            }
            // 4. Arrays from inner that come after the pointer (variable's array dims)
            for d in &inner_derived[last_ptr_idx + 1..] {
                result.push(d.clone());
            }
            return result;
        }

        // Handle nested function pointer variables like `int (*(*p)(int a, int b))(int c, int d)`.
        // The inner declarator produces [Pointer, Pointer, FunctionPointer(...)], and the
        // outer adds Function(...). We keep inner_derived intact and convert the outer
        // Function to Pointer + FunctionPointer, yielding:
        //   [Pointer, Pointer, FunctionPointer([int,int]), Pointer, FunctionPointer([int,int])]
        //
        // This does NOT match function definitions returning function pointers like
        // `int (*g(int))(int)`, where inner has Function (not FunctionPointer).
        let inner_starts_with_pointer = matches!(inner_derived.first(), Some(DerivedDeclarator::Pointer));
        let inner_has_fptr = inner_derived.iter().any(|d| matches!(d, DerivedDeclarator::FunctionPointer(_, _)));
        if inner_starts_with_pointer && inner_has_fptr && outer_starts_with_function {
            let mut result = outer_pointers;
            // Keep inner_derived in its existing order
            result.extend(inner_derived);
            // The outer Function becomes a Pointer + FunctionPointer pair
            for suffix in outer_suffixes {
                match suffix {
                    DerivedDeclarator::Function(params, variadic) => {
                        result.push(DerivedDeclarator::Pointer);
                        result.push(DerivedDeclarator::FunctionPointer(params, variadic));
                    }
                    other => result.push(other),
                }
            }
            return result;
        }

        // General case: outer_pointers ++ outer_suffixes ++ inner_derived
        outer_pointers.extend(outer_suffixes);
        outer_pointers.extend(inner_derived);
        outer_pointers
    }

    /// Parse a function parameter list: (params...) or (void) or ()
    pub(super) fn parse_param_list(&mut self) -> (Vec<ParamDecl>, bool) {
        let open = self.peek_span();
        self.expect_context(&TokenKind::LParen, "for parameter list");
        let mut params = Vec::new();
        let mut variadic = false;

        if matches!(self.peek(), TokenKind::RParen) {
            self.advance();
            return (params, variadic);
        }

        // Handle (void)
        if matches!(self.peek(), TokenKind::Void) {
            let save = self.pos;
            self.advance();
            if matches!(self.peek(), TokenKind::RParen) {
                self.advance();
                return (params, variadic);
            }
            self.pos = save;
        }

        // Check for K&R-style identifier list
        if let TokenKind::Identifier(ref name) = self.peek() {
            if (!self.typedefs.contains(name) || self.shadowed_typedefs.contains(name)) && !self.is_type_specifier() {
                return self.parse_kr_identifier_list();
            }
        }

        loop {
            if matches!(self.peek(), TokenKind::Ellipsis) {
                self.advance();
                variadic = true;
                break;
            }

            // Save noreturn before skip_gcc_extensions() so that a noreturn attribute
            // on a function pointer parameter (e.g. `__attribute__((__noreturn__)) fn_ptr_t`)
            // doesn't leak to the enclosing function declaration.
            let saved_noreturn = self.attrs.parsing_noreturn();
            self.skip_gcc_extensions();
            // Save and reset parsing_const to detect if this parameter's base type is const.
            let saved_const = self.attrs.parsing_const();
            self.attrs.set_const(false);
            self.attrs.set_noreturn(saved_noreturn);
            if let Some(mut type_spec) = self.parse_type_specifier() {
                // Capture whether the base type (before pointer declarators) was const.
                // For `const int *p`, parsing_const is true here; the `*` is handled below.
                let param_is_const = self.attrs.parsing_const();
                let (name, pointer_depth, array_dims, is_func_ptr, ptr_to_array_dims, fptr_param_decls, inner_ptr_depth) =
                    self.parse_param_declarator_full();
                self.skip_gcc_extensions();

                // Apply pointer levels
                for _ in 0..pointer_depth {
                    type_spec = TypeSpecifier::Pointer(Box::new(type_spec), AddressSpace::Default);
                }

                // Pointer-to-array: int (*p)[N][M]
                if !ptr_to_array_dims.is_empty() {
                    for dim in ptr_to_array_dims.iter().rev() {
                        type_spec = TypeSpecifier::Array(Box::new(type_spec), dim.clone());
                    }
                    type_spec = TypeSpecifier::Pointer(Box::new(type_spec), AddressSpace::Default);
                }

                // Array params: outermost dimension decays to pointer.
                // Preserve the outermost dimension expression (if any) so side effects
                // like `a++` in `int b[a++]` can be evaluated during IR lowering.
                let mut vla_size_exprs = Vec::new();
                if !array_dims.is_empty() {
                    if let Some(Some(expr)) = array_dims.first() {
                        vla_size_exprs.push((**expr).clone());
                    }
                    for dim in array_dims.iter().skip(1).rev() {
                        type_spec = TypeSpecifier::Array(Box::new(type_spec), dim.clone());
                    }
                    type_spec = TypeSpecifier::Pointer(Box::new(type_spec), AddressSpace::Default);
                }

                // Function pointers decay to pointer
                if is_func_ptr {
                    type_spec = TypeSpecifier::Pointer(Box::new(type_spec), AddressSpace::Default);
                }

                self.attrs.set_const(saved_const);
                self.attrs.set_noreturn(saved_noreturn);
                params.push(ParamDecl { type_spec, name, fptr_params: fptr_param_decls, is_const: param_is_const, vla_size_exprs, fptr_inner_ptr_depth: inner_ptr_depth });
            } else {
                self.attrs.set_const(saved_const);
                self.attrs.set_noreturn(saved_noreturn);
                break;
            }

            if !self.consume_if(&TokenKind::Comma) {
                break;
            }
        }

        self.expect_closing(&TokenKind::RParen, open);
        (params, variadic)
    }

    /// Parse a K&R-style identifier list: foo(a, b, c)
    fn parse_kr_identifier_list(&mut self) -> (Vec<ParamDecl>, bool) {
        let mut params = Vec::new();
        while let TokenKind::Identifier(ref n) = self.peek() {
            let n = n.clone();
            self.advance();
            params.push(ParamDecl {
                type_spec: TypeSpecifier::Int, // K&R default type
                name: Some(n),
                fptr_params: None,
                is_const: false,
                vla_size_exprs: Vec::new(),
                fptr_inner_ptr_depth: 0,
            });
            if !self.consume_if(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen);
        (params, false)
    }

    /// Parse a parameter declarator with full type information.
    /// Returns (name, pointer_depth, array_dims, is_func_ptr, ptr_to_array_dims, fptr_params, fptr_inner_ptr_depth).
    pub(super) fn parse_param_declarator_full(&mut self) -> (Option<Rc<str>>, u32, Vec<Option<Box<Expr>>>, bool, Vec<Option<Box<Expr>>>, Option<Vec<ParamDecl>>, u32) {
        let mut pointer_depth: u32 = 0;
        while self.consume_if(&TokenKind::Star) {
            pointer_depth += 1;
            self.skip_cv_qualifiers();
            // Also skip __attribute__(...) after pointer qualifiers.
            // E.g., `int * __attribute__((unused)) p` is valid GNU C.
            self.skip_gcc_extensions();
        }
        let mut array_dims: Vec<Option<Box<Expr>>> = Vec::new();
        let mut is_func_ptr = false;
        let mut ptr_to_array_dims: Vec<Option<Box<Expr>>> = Vec::new();
        let mut fptr_params: Option<Vec<ParamDecl>> = None;
        let mut fptr_inner_ptr_depth: u32 = 0;

        let name = if matches!(self.peek(), TokenKind::LParen) && self.is_paren_declarator() {
            self.parse_paren_param_declarator(&mut pointer_depth, &mut array_dims, &mut is_func_ptr, &mut ptr_to_array_dims, &mut fptr_params, &mut fptr_inner_ptr_depth)
        } else if let TokenKind::Identifier(ref n) = self.peek() {
            let n = n.clone();
            self.advance();
            Some(n)
        } else {
            None
        };

        // Parse trailing array dimensions
        while matches!(self.peek(), TokenKind::LBracket) {
            self.advance();
            // Skip C99 array qualifiers (static, restrict, const, volatile)
            self.skip_array_qualifiers();
            if matches!(self.peek(), TokenKind::RBracket) {
                array_dims.push(None);
                self.advance();
            } else if matches!(self.peek(), TokenKind::Star)
                && self.pos + 1 < self.tokens.len()
                && matches!(self.tokens[self.pos + 1].kind, TokenKind::RBracket) {
                // C99 VLA star syntax: [*] or [static *] means unspecified VLA size
                self.advance(); // consume '*'
                array_dims.push(None);
                self.advance(); // consume ']'
            } else {
                let dim_expr = self.parse_expr();
                array_dims.push(Some(Box::new(dim_expr)));
                self.expect(&TokenKind::RBracket);
            }
        }

        // Trailing function parameter list means this parameter has function type,
        // which in C decays to a function pointer (C11 6.7.6.3p8).
        // E.g., `int f(union U callback(void))` → callback has type `union U (*)(void)`
        // Also handles abstract declarators: `void (Dat *)` → function taking Dat*, returning void
        if matches!(self.peek(), TokenKind::LParen) {
            is_func_ptr = true;
            let (fp_params, _variadic) = self.parse_param_list();
            fptr_params = Some(fp_params);
        }

        (name, pointer_depth, array_dims, is_func_ptr, ptr_to_array_dims, fptr_params, fptr_inner_ptr_depth)
    }

    /// Parse a parenthesized parameter declarator: (*name)(params), (name), etc.
    fn parse_paren_param_declarator(
        &mut self,
        pointer_depth: &mut u32,
        array_dims: &mut Vec<Option<Box<Expr>>>,
        is_func_ptr: &mut bool,
        ptr_to_array_dims: &mut Vec<Option<Box<Expr>>>,
        fptr_params: &mut Option<Vec<ParamDecl>>,
        fptr_inner_ptr_depth: &mut u32,
    ) -> Option<Rc<str>> {
        let save = self.pos;
        self.advance(); // consume '('

        // Skip __attribute__ / __extension__ before pointer declarator
        // e.g. void foo(void (__attribute__((ms_abi)) *handler)(int))
        self.skip_gcc_extensions();

        if matches!(self.peek(), TokenKind::LBracket) {
            // Abstract array declarator in parens: ([4]) or ([])
            // E.g., `int f(int ([4]))` is equivalent to `int f(int *)`
            while matches!(self.peek(), TokenKind::LBracket) {
                self.advance();
                self.skip_array_qualifiers();
                if matches!(self.peek(), TokenKind::RBracket) {
                    array_dims.push(None);
                    self.advance();
                } else {
                    let dim_expr = self.parse_expr();
                    array_dims.push(Some(Box::new(dim_expr)));
                    self.expect(&TokenKind::RBracket);
                }
            }
            self.expect(&TokenKind::RParen);
            None
        } else if matches!(self.peek(), TokenKind::Star) {
            // Function pointer or pointer-to-array: (*name)(params) or (*name)[N]
            let mut inner_ptr_depth = 0u32;
            while self.consume_if(&TokenKind::Star) {
                inner_ptr_depth += 1;
                self.skip_cv_qualifiers();
                // Also skip __attribute__(...) after pointer qualifiers.
                // E.g., `void (*__attribute__((unused)) fp)(int)` is valid GNU C.
                self.skip_gcc_extensions();
            }
            let name = if let TokenKind::Identifier(ref n) = self.peek() {
                let n = n.clone();
                self.advance();
                Some(n)
            } else if matches!(self.peek(), TokenKind::LParen) {
                self.extract_paren_name()
            } else {
                None
            };
            *pointer_depth += inner_ptr_depth.saturating_sub(1);
            // Parse array dimensions inside parens: (*a[]) or (*a[N])
            // These represent "array of pointers" and need to be propagated
            // to the caller so array-to-pointer decay is properly applied.
            // E.g. int (*a[]) = array of ptr-to-int = int **a (after decay)
            let mut inner_array_dims = Vec::new();
            while matches!(self.peek(), TokenKind::LBracket) {
                self.advance();
                self.skip_array_qualifiers();
                if matches!(self.peek(), TokenKind::RBracket) {
                    inner_array_dims.push(None);
                    self.advance();
                } else {
                    let dim_expr = self.parse_expr();
                    inner_array_dims.push(Some(Box::new(dim_expr)));
                    self.expect(&TokenKind::RBracket);
                }
            }
            self.expect(&TokenKind::RParen);
            if !inner_array_dims.is_empty() && !matches!(self.peek(), TokenKind::LParen) {
                // Array of pointers: (*a[]) or (*a[N])
                // The * gives one level of pointer, and the array dims are
                // propagated for the caller to apply array-to-pointer decay.
                // E.g. int (*a[]) → array_dims=[None], pointer_depth+=1 → int **a
                *pointer_depth += 1;
                *array_dims = inner_array_dims;
            } else if matches!(self.peek(), TokenKind::LParen) {
                // Function pointer: (*fp)(params) or (*fp[])(params)
                *is_func_ptr = true;
                *fptr_inner_ptr_depth = inner_ptr_depth;
                let (fp_params, _variadic) = self.parse_param_list();
                *fptr_params = Some(fp_params);
            } else if matches!(self.peek(), TokenKind::LBracket) {
                // Pointer-to-array: (*p)[N]
                while matches!(self.peek(), TokenKind::LBracket) {
                    self.advance();
                    self.skip_array_qualifiers();
                    if matches!(self.peek(), TokenKind::RBracket) {
                        ptr_to_array_dims.push(None);
                        self.advance();
                    } else {
                        let dim_expr = self.parse_expr();
                        ptr_to_array_dims.push(Some(Box::new(dim_expr)));
                        self.expect(&TokenKind::RBracket);
                    }
                }
            } else {
                *pointer_depth += 1;
            }
            name
        } else if self.consume_if(&TokenKind::Caret) {
            // Block pointer (Apple extension)
            let name = if let TokenKind::Identifier(ref n) = self.peek() {
                let n = n.clone();
                self.advance();
                Some(n)
            } else {
                None
            };
            self.expect(&TokenKind::RParen);
            if matches!(self.peek(), TokenKind::LParen) {
                self.skip_balanced_parens();
            }
            name
        } else if let TokenKind::Identifier(_) = self.peek() {
            // Parenthesized name: (name), (name)(params), or (name(params))
            let name = if let TokenKind::Identifier(ref n) = self.peek() {
                let n = n.clone();
                self.advance();
                Some(n)
            } else {
                None
            };
            // Check for function parameter list INSIDE the outer parens:
            // E.g., `int (fn_op(void *, object, object))` where the parameter list
            // is inside the parenthesized declarator. This declares fn_op as having
            // function type `int(void *, object, object)` which decays to a function
            // pointer per C11 6.7.6.3p8.
            if matches!(self.peek(), TokenKind::LParen) {
                *is_func_ptr = true;
                let (fp_params, _variadic) = self.parse_param_list();
                *fptr_params = Some(fp_params);
            }
            self.expect(&TokenKind::RParen);
            self.skip_array_dimensions();
            // Trailing (params) outside the parens means function-type parameter decay.
            // E.g., `int (f)(int)` → f has function type, decays to function pointer.
            // Parse the param list to preserve function type information.
            if !*is_func_ptr && matches!(self.peek(), TokenKind::LParen) {
                *is_func_ptr = true;
                let (fp_params, _variadic) = self.parse_param_list();
                *fptr_params = Some(fp_params);
            }
            name
        } else if matches!(self.peek(), TokenKind::LParen) {
            // Nested parens: ((name)), ((*name)), ((name)(params)), or ((type))
            let inner_save = self.pos;
            let name = self.extract_paren_name();
            if name.is_some() {
                // Successfully extracted a name. Check for function param lists
                // inside the outer parens, e.g. ((fnc)(int)) or ((*fp)(int))
                self.skip_array_dimensions();
                // Trailing (params) inside outer parens means function pointer.
                // E.g., `int ((*f)(int))` or `int ((f)(int))`
                // Parse the param list to preserve function type information.
                if matches!(self.peek(), TokenKind::LParen) {
                    *is_func_ptr = true;
                    let (fp_params, _variadic) = self.parse_param_list();
                    *fptr_params = Some(fp_params);
                }
            } else {
                // extract_paren_name failed (e.g. ((int)) where inner content
                // is a type, not a name). Restore position and skip the inner
                // content as balanced parentheses.
                self.pos = inner_save;
                self.skip_balanced_parens();
            }
            self.expect(&TokenKind::RParen);
            // Also skip trailing suffixes after the outer parens
            self.skip_array_dimensions();
            // Trailing (params) after outer parens also means function type decay.
            // Parse the param list to preserve function type information.
            if matches!(self.peek(), TokenKind::LParen) {
                *is_func_ptr = true;
                let (fp_params, _variadic) = self.parse_param_list();
                *fptr_params = Some(fp_params);
            }
            name
        } else {
            self.pos = save;
            None
        }
    }

    /// Extract a name from nested parentheses: (name), ((name)), (*(name)), etc.
    pub(super) fn extract_paren_name(&mut self) -> Option<Rc<str>> {
        if !matches!(self.peek(), TokenKind::LParen) {
            if let TokenKind::Identifier(ref n) = self.peek() {
                let n = n.clone();
                self.advance();
                return Some(n);
            }
            return None;
        }
        self.advance(); // consume '('
        if matches!(self.peek(), TokenKind::Star) {
            self.advance();
            self.skip_cv_qualifiers();
        }
        let name = if matches!(self.peek(), TokenKind::LParen) {
            self.extract_paren_name()
        } else if let TokenKind::Identifier(ref n) = self.peek() {
            let n = n.clone();
            self.advance();
            Some(n)
        } else {
            None
        };
        self.consume_if(&TokenKind::RParen);
        name
    }

    /// Try to parse a parenthesized abstract declarator: (*), ((*)), (**), (*[3][4])
    /// Also handles nested function pointers: (*(*)(params))
    /// Returns a ParenAbstractDecl if successful, None otherwise.
    /// Restores position on failure.
    pub(super) fn try_parse_paren_abstract_declarator(&mut self) -> Option<ParenAbstractDecl> {
        if !matches!(self.peek(), TokenKind::LParen) {
            return None;
        }
        let save = self.pos;
        self.advance(); // consume '('

        let mut total_ptrs = 0u32;

        // Skip __attribute__ / __extension__ before pointer declarator
        // e.g. (int(__attribute__((noinline)) *)(void)) function_pointer
        self.skip_gcc_extensions();

        while self.consume_if(&TokenKind::Star) {
            total_ptrs += 1;
            self.skip_cv_qualifiers();
            // Also skip attributes after each pointer star, e.g. (* __attribute__((unused)))
            self.skip_gcc_extensions();
        }

        // Check for nested: (* (...))
        if matches!(self.peek(), TokenKind::LParen) {
            if let Some(inner) = self.try_parse_paren_abstract_declarator() {
                match inner {
                    ParenAbstractDecl::Simple { ptr_depth: inner_ptrs, array_dims: inner_dims } => {
                        // After inner (*), check if a parameter list follows — making this
                        // a nested function pointer: (*(*)(params))
                        if matches!(self.peek(), TokenKind::LParen) {
                            // This is a function pointer inside the outer parens
                            // e.g., (*(*)(void*)) — inner (*) + (void*) forms a fn ptr
                            let (params, variadic) = self.parse_param_list();
                            // Now expect the closing ')' of the outer group
                            if self.consume_if(&TokenKind::RParen) {
                                return Some(ParenAbstractDecl::NestedFnPtr {
                                    outer_ptr_depth: total_ptrs,
                                    inner_ptr_depth: inner_ptrs,
                                    inner_params: params,
                                    inner_variadic: variadic,
                                });
                            } else {
                                self.pos = save;
                                return None;
                            }
                        }
                        // Otherwise, handle as before: simple nested grouping
                        let combined_ptrs = total_ptrs + inner_ptrs;
                        let mut array_dims = inner_dims;
                        while matches!(self.peek(), TokenKind::LBracket) {
                            self.advance();
                            let size = if matches!(self.peek(), TokenKind::RBracket) {
                                None
                            } else {
                                Some(Box::new(self.parse_expr()))
                            };
                            self.expect(&TokenKind::RBracket);
                            array_dims.push(size);
                        }
                        if self.consume_if(&TokenKind::RParen) {
                            return Some(ParenAbstractDecl::Simple {
                                ptr_depth: combined_ptrs,
                                array_dims,
                            });
                        } else {
                            self.pos = save;
                            return None;
                        }
                    }
                    ParenAbstractDecl::NestedFnPtr { .. } => {
                        // Deeply nested fn ptrs: just close the outer group
                        // TODO: support deeper nesting if needed
                        if self.consume_if(&TokenKind::RParen) {
                            return Some(inner);
                        } else {
                            self.pos = save;
                            return None;
                        }
                    }
                }
            } else {
                self.pos = save;
                return None;
            }
        }

        // Parse array dimensions after pointer(s): (*[3][4])
        let mut array_dims = Vec::new();
        while matches!(self.peek(), TokenKind::LBracket) {
            self.advance();
            let size = if matches!(self.peek(), TokenKind::RBracket) {
                None
            } else {
                Some(Box::new(self.parse_expr()))
            };
            self.expect(&TokenKind::RBracket);
            array_dims.push(size);
        }

        if self.consume_if(&TokenKind::RParen) {
            if total_ptrs > 0 || !array_dims.is_empty() {
                Some(ParenAbstractDecl::Simple { ptr_depth: total_ptrs, array_dims })
            } else {
                self.pos = save;
                None
            }
        } else {
            self.pos = save;
            None
        }
    }
}
