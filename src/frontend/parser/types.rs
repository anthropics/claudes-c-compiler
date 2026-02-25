// Type specifier parsing: handles all C type specifiers including struct/union/enum
// definitions, typedef names, and GNU extensions like typeof and _Complex.
//
// The main complexity here is that C allows type specifier tokens in any order
// (e.g., "long unsigned int" == "unsigned long int"), so we collect flags
// and resolve them at the end.

use std::rc::Rc;
use crate::common::types::AddressSpace;
use crate::frontend::lexer::token::TokenKind;
use super::ast::*;
use super::parse::{ModeKind, Parser};

/// Collected type specifier flags during parsing.
///
/// C allows type specifier keywords in any order ("long unsigned int" ==
/// "unsigned long int"), so we accumulate boolean flags and a `long_count`
/// while scanning tokens, then resolve them into a concrete `TypeSpecifier`
/// via `Parser::resolve_type_flags`.
#[derive(Default)]
struct TypeSpecFlags {
    has_void: bool,
    has_bool: bool,
    has_float: bool,
    has_double: bool,
    has_complex: bool,
    has_char: bool,
    has_short: bool,
    has_int: bool,
    has_unsigned: bool,
    has_signed: bool,
    has_struct: bool,
    has_union: bool,
    has_enum: bool,
    has_typeof: bool,
    long_count: u32,
    typedef_name: Option<Rc<str>>,
}

impl Parser {
    /// Parse a complete type specifier. Returns None if no type specifier found.
    ///
    /// Handles arbitrary ordering of type keywords (C allows "long unsigned int"
    /// or "unsigned long int"), struct/union/enum definitions, typedef names,
    /// typeof expressions, and _Complex types.
    pub(super) fn parse_type_specifier(&mut self) -> Option<TypeSpecifier> {
        self.skip_gcc_extensions();

        let mut flags = TypeSpecFlags::default();
        let mut mode_kind: Option<ModeKind> = None;
        let mut any_base_specifier = false;
        let mut any_storage_class = false;

        // Collect qualifiers, storage classes, and type specifiers
        loop {
            match self.peek() {
                // Qualifiers
                TokenKind::Const => {
                    self.advance();
                    self.attrs.set_const(true);
                }
                TokenKind::Volatile => {
                    self.advance();
                    self.attrs.set_volatile(true);
                }
                TokenKind::Restrict => {
                    self.advance();
                }
                TokenKind::Register
                | TokenKind::Auto => {
                    self.advance();
                    any_storage_class = true;
                }
                TokenKind::Noreturn => {
                    self.advance();
                    self.attrs.set_noreturn(true);
                }
                // GCC named address space qualifiers (__seg_gs / __seg_fs)
                TokenKind::SegGs => {
                    self.advance();
                    self.attrs.parsing_address_space = AddressSpace::SegGs;
                }
                TokenKind::SegFs => {
                    self.advance();
                    self.attrs.parsing_address_space = AddressSpace::SegFs;
                }
                // __auto_type - GCC extension: type inferred from initializer
                TokenKind::AutoType => {
                    self.advance();
                    return Some(TypeSpecifier::AutoType);
                }
                TokenKind::Inline => {
                    self.advance();
                    self.attrs.set_inline(true);
                }
                // Storage classes
                TokenKind::Static => {
                    self.advance();
                    self.attrs.set_static(true);
                    any_storage_class = true;
                }
                TokenKind::Extern => {
                    self.advance();
                    self.attrs.set_extern(true);
                    any_storage_class = true;
                }
                TokenKind::Typedef => {
                    self.advance();
                    self.attrs.set_typedef(true);
                    any_storage_class = true;
                }
                // Thread-local storage class (__thread / _Thread_local)
                TokenKind::ThreadLocal => {
                    self.advance();
                    self.attrs.set_thread_local(true);
                    any_storage_class = true;
                }
                // _Complex modifier
                TokenKind::Complex => {
                    self.advance();
                    flags.has_complex = true;
                    any_base_specifier = true;
                }
                // GNU extensions
                TokenKind::Attribute => {
                    let (_, aligned, mk, _) = self.parse_gcc_attributes();
                    mode_kind = mode_kind.or(mk);
                    if let Some(a) = aligned {
                        self.attrs.parsed_alignas = Some(self.attrs.parsed_alignas.map_or(a, |prev| prev.max(a)));
                    }
                    // parse_gcc_attributes already sets self.attrs.parsing_constructor/destructor
                }
                TokenKind::Extension => {
                    self.advance();
                }
                // _Atomic as type specifier: _Atomic(type-name)
                // C11 §6.7.2.4: _Atomic(T) is a type specifier equivalent to T
                // with atomic qualification. Since we don't track atomic-ness,
                // we just parse and return the inner type.
                TokenKind::Atomic => {
                    self.advance();
                    if matches!(self.peek(), TokenKind::LParen) {
                        let open = self.peek_span();
                        self.advance(); // consume '('
                        // Save and restore const qualifier across inner type parse
                        // to prevent leakage (e.g., _Atomic(const int) should not
                        // make the outer declaration const).
                        let saved_const = self.attrs.parsing_const();
                        self.attrs.set_const(false);
                        let inner = self.parse_type_specifier();
                        self.attrs.set_const(saved_const);
                        if let Some(inner_type) = inner {
                            let result = self.parse_abstract_declarator_suffix(inner_type);
                            self.expect_closing(&TokenKind::RParen, open);
                            return Some(result);
                        }
                        // Fallback: if we can't parse a type, emit error and skip
                        let err_span = self.peek_span();
                        self.emit_error("expected type name in _Atomic(...)", err_span);
                        while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
                            self.advance();
                        }
                        self.consume_if(&TokenKind::RParen);
                        return Some(TypeSpecifier::Int);
                    }
                    // _Atomic without parens is a type qualifier; since we don't
                    // track atomic-ness, it falls through to continue collecting
                    // type specifiers.
                }
                // Alignas
                TokenKind::Alignas => {
                    self.advance();
                    if let Some(align) = self.parse_alignas_argument() {
                        self.attrs.parsed_alignas = Some(self.attrs.parsed_alignas.map_or(align, |prev| prev.max(align)));
                    }
                }
                // Type specifier tokens
                TokenKind::Void => {
                    self.advance(); flags.has_void = true; any_base_specifier = true;
                    break; // void can't combine with others
                }
                TokenKind::Char => {
                    self.advance(); flags.has_char = true; any_base_specifier = true;
                    break; // char only combines with signed/unsigned
                }
                TokenKind::Short => {
                    self.advance(); flags.has_short = true; any_base_specifier = true;
                }
                TokenKind::Int => {
                    self.advance(); flags.has_int = true; any_base_specifier = true;
                }
                TokenKind::Long => {
                    self.advance(); flags.long_count += 1; any_base_specifier = true;
                }
                TokenKind::Float => {
                    self.advance(); flags.has_float = true; any_base_specifier = true;
                    break;
                }
                TokenKind::Double => {
                    self.advance(); flags.has_double = true; any_base_specifier = true;
                    break;
                }
                TokenKind::Bool => {
                    self.advance(); flags.has_bool = true; any_base_specifier = true;
                    break;
                }
                TokenKind::Signed => {
                    self.advance(); flags.has_signed = true; any_base_specifier = true;
                }
                TokenKind::Unsigned => {
                    self.advance(); flags.has_unsigned = true; any_base_specifier = true;
                }
                // __int128 can combine with signed/unsigned
                TokenKind::Int128 => {
                    let span = self.peek_span();
                    self.advance();
                    // GCC rejects __int128 on 32-bit targets
                    if crate::common::types::target_is_32bit() {
                        self.emit_error("__int128 is not supported on this target", span);
                        // Fall through with Int (to avoid cascading errors)
                        return Some(TypeSpecifier::Int);
                    }
                    // __int128 already implies signed unless unsigned is present
                    if flags.has_unsigned {
                        return Some(TypeSpecifier::UnsignedInt128);
                    } else {
                        return Some(TypeSpecifier::Int128);
                    }
                }
                // __uint128_t is always unsigned
                TokenKind::UInt128 => {
                    let span = self.peek_span();
                    self.advance();
                    // GCC rejects __uint128_t on 32-bit targets
                    if crate::common::types::target_is_32bit() {
                        self.emit_error("__uint128_t is not supported on this target", span);
                        return Some(TypeSpecifier::UnsignedInt);
                    }
                    return Some(TypeSpecifier::UnsignedInt128);
                }
                TokenKind::Struct => {
                    self.advance(); flags.has_struct = true; any_base_specifier = true;
                    break;
                }
                TokenKind::Union => {
                    self.advance(); flags.has_union = true; any_base_specifier = true;
                    break;
                }
                TokenKind::Enum => {
                    self.advance(); flags.has_enum = true; any_base_specifier = true;
                    break;
                }
                TokenKind::Typeof => {
                    self.advance(); flags.has_typeof = true; any_base_specifier = true;
                    break;
                }
                TokenKind::Builtin => {
                    if !any_base_specifier {
                        flags.typedef_name = Some(Rc::from("__builtin_va_list"));
                        self.advance();
                        any_base_specifier = true;
                        break;
                    } else {
                        break;
                    }
                }
                TokenKind::Identifier(ref name) if self.typedefs.contains(name) && !self.shadowed_typedefs.contains(name) => {
                    if !any_base_specifier {
                        flags.typedef_name = Some(name.clone());
                        self.advance();
                        any_base_specifier = true;
                        break;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        // After the main loop, collect trailing specifiers that can follow
        // certain base types (e.g., "short unsigned int", "double long", "float _Complex")
        self.collect_trailing_specifiers(&mut flags, &mut mode_kind);

        if !any_base_specifier {
            // C89 implicit int: if a storage class specifier was consumed but no
            // type specifier was found, the type defaults to int.
            // E.g., "static x = 5;" means "static int x = 5;",
            //       "register y;" means "register int y;".
            if any_storage_class {
                return Some(TypeSpecifier::Int);
            }
            return None;
        }

        // Resolve collected flags into a TypeSpecifier
        let base = self.resolve_type_flags(&flags);

        // Handle trailing _Complex, qualifiers, and storage classes after the base type
        let base = self.consume_trailing_qualifiers(base);

        // Apply __attribute__((mode(...))): transform type to the specified bit-width
        let base = if let Some(mk) = mode_kind {
            mk.apply(base)
        } else {
            base
        };

        Some(base)
    }

    /// Collect additional type specifier tokens that follow the initial base type.
    /// E.g., "short" can be followed by "unsigned int", "double" by "long",
    /// "float" by "_Complex".
    fn collect_trailing_specifiers(
        &mut self,
        flags: &mut TypeSpecFlags,
        mode_kind: &mut Option<ModeKind>,
    ) {
        if flags.has_char || flags.has_short || flags.has_int || flags.long_count > 0 {
            loop {
                match self.peek() {
                    TokenKind::Signed => { self.advance(); flags.has_signed = true; }
                    TokenKind::Unsigned => { self.advance(); flags.has_unsigned = true; }
                    TokenKind::Int => { self.advance(); flags.has_int = true; }
                    TokenKind::Long => { self.advance(); flags.long_count += 1; }
                    TokenKind::Short => { self.advance(); flags.has_short = true; }
                    TokenKind::Char => { self.advance(); flags.has_char = true; }
                    TokenKind::Complex => { self.advance(); flags.has_complex = true; }
                    TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => { self.advance(); }
                    TokenKind::SegGs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegGs; }
                    TokenKind::SegFs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegFs; }
                    TokenKind::Static => { self.advance(); self.attrs.set_static(true); }
                    TokenKind::Extern => { self.advance(); self.attrs.set_extern(true); }
                    TokenKind::Auto | TokenKind::Register => { self.advance(); }
                    TokenKind::ThreadLocal => { self.advance(); self.attrs.set_thread_local(true); }
                    TokenKind::Noreturn => { self.advance(); self.attrs.set_noreturn(true); }
                    TokenKind::Inline => { self.advance(); self.attrs.set_inline(true); }
                    TokenKind::Attribute => {
                        let (_, aligned, mk, _) = self.parse_gcc_attributes();
                        *mode_kind = mode_kind.or(mk);
                        if let Some(a) = aligned {
                            self.attrs.parsed_alignas = Some(self.attrs.parsed_alignas.map_or(a, |prev| prev.max(a)));
                        }
                    }
                    TokenKind::Extension => { self.advance(); }
                    _ => break,
                }
            }
        } else if flags.has_float {
            // "float" can be followed by "_Complex" and storage class / qualifiers
            loop {
                match self.peek() {
                    TokenKind::Complex => { self.advance(); flags.has_complex = true; }
                    TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => { self.advance(); }
                    TokenKind::SegGs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegGs; }
                    TokenKind::SegFs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegFs; }
                    TokenKind::Static => { self.advance(); self.attrs.set_static(true); }
                    TokenKind::Extern => { self.advance(); self.attrs.set_extern(true); }
                    TokenKind::Auto | TokenKind::Register => { self.advance(); }
                    TokenKind::ThreadLocal => { self.advance(); self.attrs.set_thread_local(true); }
                    TokenKind::Noreturn => { self.advance(); self.attrs.set_noreturn(true); }
                    TokenKind::Inline => { self.advance(); self.attrs.set_inline(true); }
                    TokenKind::Extension => { self.advance(); }
                    _ => break,
                }
            }
        } else if flags.has_double {
            // "double" can be followed by "long", "_Complex", and storage class / qualifiers
            loop {
                match self.peek() {
                    TokenKind::Long => { self.advance(); flags.long_count += 1; }
                    TokenKind::Complex => { self.advance(); flags.has_complex = true; }
                    TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => { self.advance(); }
                    TokenKind::SegGs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegGs; }
                    TokenKind::SegFs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegFs; }
                    TokenKind::Static => { self.advance(); self.attrs.set_static(true); }
                    TokenKind::Extern => { self.advance(); self.attrs.set_extern(true); }
                    TokenKind::Auto | TokenKind::Register => { self.advance(); }
                    TokenKind::ThreadLocal => { self.advance(); self.attrs.set_thread_local(true); }
                    TokenKind::Noreturn => { self.advance(); self.attrs.set_noreturn(true); }
                    TokenKind::Inline => { self.advance(); self.attrs.set_inline(true); }
                    TokenKind::Extension => { self.advance(); }
                    _ => break,
                }
            }
        }
    }

    /// Resolve the collected type specifier flags into a concrete TypeSpecifier.
    fn resolve_type_flags(
        &mut self,
        flags: &TypeSpecFlags,
    ) -> TypeSpecifier {
        let TypeSpecFlags {
            has_void, has_bool, has_float, has_double, has_complex,
            has_char, has_short, has_int: _, has_unsigned, has_signed: _,
            has_struct, has_union, has_enum, has_typeof, long_count,
            ref typedef_name,
        } = *flags;
        if has_void {
            TypeSpecifier::Void
        } else if has_bool {
            TypeSpecifier::Bool
        } else if has_float {
            if has_complex { TypeSpecifier::ComplexFloat } else { TypeSpecifier::Float }
        } else if has_double {
            if has_complex {
                if long_count > 0 { TypeSpecifier::ComplexLongDouble } else { TypeSpecifier::ComplexDouble }
            } else if long_count > 0 {
                TypeSpecifier::LongDouble
            } else {
                TypeSpecifier::Double
            }
        } else if has_complex && !has_struct && !has_union && !has_enum {
            // standalone _Complex defaults to _Complex double
            TypeSpecifier::ComplexDouble
        } else if has_struct {
            self.parse_struct_or_union(true)
        } else if has_union {
            self.parse_struct_or_union(false)
        } else if has_enum {
            self.parse_enum_specifier()
        } else if has_typeof {
            self.parse_typeof_specifier()
        } else if let Some(ref name) = typedef_name {
            TypeSpecifier::TypedefName(name.clone())
        } else if has_char {
            if has_unsigned { TypeSpecifier::UnsignedChar } else { TypeSpecifier::Char }
        } else if has_short {
            if has_unsigned { TypeSpecifier::UnsignedShort } else { TypeSpecifier::Short }
        } else if long_count >= 2 {
            if has_unsigned { TypeSpecifier::UnsignedLongLong } else { TypeSpecifier::LongLong }
        } else if long_count == 1 {
            if has_unsigned { TypeSpecifier::UnsignedLong } else { TypeSpecifier::Long }
        } else if has_unsigned {
            TypeSpecifier::UnsignedInt
        } else {
            // signed, int, or signed int
            TypeSpecifier::Int
        }
    }

    /// Parse a struct or union definition/reference.
    fn parse_struct_or_union(&mut self, is_struct: bool) -> TypeSpecifier {
        let (mut is_packed, mut struct_aligned, _, _) = self.parse_gcc_attributes();
        let name = if let TokenKind::Identifier(n) = self.peek() {
            let n = n.clone();
            self.advance();
            Some(n)
        } else {
            None
        };
        let (packed2, aligned2, _, _) = self.parse_gcc_attributes();
        is_packed = is_packed || packed2;
        if aligned2.is_some() { struct_aligned = aligned2; }
        let fields = if matches!(self.peek(), TokenKind::LBrace) {
            // Save and restore parsing_const across struct field parsing.
            // Field types may contain `const` (e.g., `const int *p`), and without
            // this save/restore the const from field types leaks into the outer
            // declaration, incorrectly marking non-const variables as const.
            let saved_const = self.attrs.parsing_const();
            let f = self.parse_struct_fields();
            self.attrs.set_const(saved_const);
            Some(f)
        } else {
            None
        };
        let (packed3, aligned3, _, _) = self.parse_gcc_attributes();
        is_packed = is_packed || packed3;
        if aligned3.is_some() { struct_aligned = aligned3; }
        // Apply current #pragma pack alignment to struct definition
        let max_field_align = self.pragma_pack_align;
        let ts = if is_struct {
            TypeSpecifier::Struct(name, fields, is_packed, max_field_align, struct_aligned)
        } else {
            TypeSpecifier::Union(name, fields, is_packed, max_field_align, struct_aligned)
        };
        // Record alignment for named struct/union definitions so that later
        // tag-only references (e.g., __alignof__(struct foo)) can look it up.
        match &ts {
            TypeSpecifier::Struct(Some(ref tag), Some(_), ..)
            | TypeSpecifier::Union(Some(ref tag), Some(_), ..) => {
                let align = Self::alignof_type_spec(&ts, None);
                self.struct_tag_alignments.insert(tag.clone(), align);
            }
            _ => {}
        }
        ts
    }

    /// Parse an enum definition/reference.
    fn parse_enum_specifier(&mut self) -> TypeSpecifier {
        let (mut is_packed, _, _, _) = self.parse_gcc_attributes();
        let name = if let TokenKind::Identifier(n) = self.peek() {
            let n = n.clone();
            self.advance();
            Some(n)
        } else {
            None
        };
        let (packed2, _, _, _) = self.parse_gcc_attributes();
        is_packed = is_packed || packed2;
        let variants = if matches!(self.peek(), TokenKind::LBrace) {
            let v = self.parse_enum_variants();
            // Register enum constant values so that later constant expressions
            // (e.g., in __attribute__((aligned(1 << ENUM_CONST)))) can resolve them.
            self.register_enum_constants(&v);
            Some(v)
        } else {
            None
        };
        let (packed3, _, _, _) = self.parse_gcc_attributes();
        is_packed = is_packed || packed3;
        TypeSpecifier::Enum(name, variants, is_packed)
    }

    /// Parse typeof(expr) or typeof(type-name).
    fn parse_typeof_specifier(&mut self) -> TypeSpecifier {
        let open = self.peek_span();
        self.expect_context(&TokenKind::LParen, "after 'typeof'");
        // Save attrs.flags so that storage-class specifiers from declarations
        // inside a statement expression (e.g., `typeof(({ extern void f(); x; }))`)
        // don't leak into the outer declaration.
        let saved_flags = self.attrs.flags;
        let save = self.pos;
        // Try parsing as a type first
        if self.is_type_specifier() {
            if let Some(ts) = self.parse_type_specifier() {
                let result_type = self.parse_abstract_declarator_suffix(ts);
                if matches!(self.peek(), TokenKind::RParen) {
                    self.advance();
                    self.attrs.flags = saved_flags;
                    return TypeSpecifier::TypeofType(Box::new(result_type));
                }
            }
            // Didn't work as type, backtrack
            self.pos = save;
            self.attrs.flags = saved_flags;
            self.expect_context(&TokenKind::LParen, "after 'typeof'");
        }
        // Parse as expression
        let expr = self.parse_expr();
        self.expect_closing(&TokenKind::RParen, open);
        self.attrs.flags = saved_flags;
        TypeSpecifier::Typeof(Box::new(expr))
    }

    /// Consume trailing qualifiers and _Complex that may follow a resolved base type.
    /// C allows "int static x;" and "double _Complex".
    fn consume_trailing_qualifiers(&mut self, mut base: TypeSpecifier) -> TypeSpecifier {
        loop {
            match self.peek() {
                TokenKind::Complex => {
                    self.advance();
                    base = match base {
                        TypeSpecifier::Float => TypeSpecifier::ComplexFloat,
                        TypeSpecifier::Double => TypeSpecifier::ComplexDouble,
                        TypeSpecifier::LongDouble => TypeSpecifier::ComplexLongDouble,
                        _ => TypeSpecifier::ComplexDouble,
                    };
                }
                TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => {
                    self.advance();
                }
                TokenKind::Static => {
                    self.advance();
                    self.attrs.set_static(true);
                }
                TokenKind::Extern => {
                    self.advance();
                    self.attrs.set_extern(true);
                }
                TokenKind::Auto | TokenKind::Register => {
                    self.advance();
                }
                TokenKind::ThreadLocal => {
                    self.advance();
                    self.attrs.set_thread_local(true);
                }
                TokenKind::Noreturn => {
                    self.advance();
                    self.attrs.set_noreturn(true);
                }
                TokenKind::Inline => {
                    self.advance();
                    self.attrs.set_inline(true);
                }
                TokenKind::Attribute => {
                    let (_, aligned, _, _) = self.parse_gcc_attributes();
                    if let Some(a) = aligned {
                        self.attrs.parsed_alignas = Some(self.attrs.parsed_alignas.map_or(a, |prev| prev.max(a)));
                    }
                }
                TokenKind::Extension => {
                    self.advance();
                }
                TokenKind::SegGs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegGs; }
                TokenKind::SegFs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegFs; }
                _ => break,
            }
        }
        base
    }

    /// Parse struct or union field declarations inside braces.
    pub(super) fn parse_struct_fields(&mut self) -> Vec<StructFieldDecl> {
        let mut fields = Vec::new();
        let open = self.peek_span();
        self.expect_context(&TokenKind::LBrace, "for struct/union body");
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            self.skip_gcc_extensions();
            if matches!(self.peek(), TokenKind::Semicolon) {
                self.advance();
                continue;
            }
            // C11 6.7.2.1: _Static_assert is allowed as a struct-declaration
            if matches!(self.peek(), TokenKind::StaticAssert) {
                self.parse_static_assert();
                continue;
            }
            if let Some(type_spec) = self.parse_type_specifier() {
                if matches!(self.peek(), TokenKind::Semicolon) {
                    // Anonymous field (e.g., anonymous struct/union)
                    let alignment = self.attrs.parsed_alignas.take();
                    fields.push(StructFieldDecl { type_spec, name: None, bit_width: None, derived: Vec::new(), alignment, is_packed: false });
                } else {
                    self.parse_struct_field_declarators(&type_spec, &mut fields);
                }
                self.skip_gcc_extensions();
                self.expect_after(&TokenKind::Semicolon, "after struct field declaration");
            } else {
                self.advance(); // skip unknown
            }
        }
        self.expect_closing(&TokenKind::RBrace, open);
        fields
    }

    /// Parse one or more declarators for a struct field using the general
    /// declarator parser. This correctly handles complex declarators like
    /// function pointers returning function pointers (e.g.,
    /// `void (*(*xDlSym)(sqlite3_vfs*, void*, const char *))(void)`).
    fn parse_struct_field_declarators(
        &mut self,
        type_spec: &TypeSpecifier,
        fields: &mut Vec<StructFieldDecl>,
    ) {
        // Capture _Alignas value that was parsed during type specifier parsing
        let mut alignas_from_type = self.attrs.parsed_alignas.take();

        // Consume post-type qualifiers that may appear between type and declarator
        // in struct field declarations: e.g., "char _Alignas(32) c;"
        self.consume_struct_field_qualifiers(&mut alignas_from_type);

        loop {
            // Handle unnamed bitfield: `: constant-expr`
            if matches!(self.peek(), TokenKind::Colon) {
                self.advance();
                // Use parse_assignment_expr to avoid consuming comma as comma-operator.
                // Bitfield width is a constant expression (no comma operator at top level).
                let bit_width = Some(Box::new(self.parse_assignment_expr()));
                fields.push(StructFieldDecl {
                    type_spec: type_spec.clone(),
                    name: None,
                    bit_width,
                    derived: Vec::new(),
                    alignment: alignas_from_type,
                    is_packed: false,
                });
                if !self.consume_if(&TokenKind::Comma) { break; }
                continue;
            }

            // Use the general-purpose declarator parser. This handles all cases:
            // simple pointers, arrays, function pointers, and nested function
            // pointer declarators of arbitrary depth.
            // parse_declarator_with_attrs also consumes trailing __attribute__ and
            // returns the aligned value from __attribute__((aligned(N))).
            let (name, derived, _, _, decl_aligned, decl_packed) = self.parse_declarator_with_attrs();

            // Parse optional bitfield width (constant-expression, not full expr with comma)
            let bit_width = if self.consume_if(&TokenKind::Colon) {
                Some(Box::new(self.parse_assignment_expr()))
            } else {
                None
            };

            // Parse any additional trailing GCC __attribute__ (e.g., after bitfield width)
            let (extra_packed, extra_aligned, _, _) = self.parse_gcc_attributes();

            // Combine alignment sources: explicit attribute on declarator,
            // extra attribute after bitfield, or _Alignas from type specifier
            let alignment = decl_aligned.or(extra_aligned).or(alignas_from_type);
            let is_packed = decl_packed || extra_packed;

            // For backward compatibility with downstream code that reads type_spec
            // directly, fold simple derived declarators (pointers, arrays) into
            // type_spec. Only use the derived field for complex cases with function
            // pointers that require build_full_ctype().
            let (field_type, field_derived) = Self::fold_simple_derived(type_spec, &derived);

            fields.push(StructFieldDecl {
                type_spec: field_type,
                name,
                bit_width,
                derived: field_derived,
                alignment,
                is_packed,
            });

            if !self.consume_if(&TokenKind::Comma) { break; }
        }
    }

    /// Consume qualifiers that may appear between type specifier and declarator
    /// in struct field declarations. Handles _Alignas, const, volatile, etc.
    fn consume_struct_field_qualifiers(&mut self, alignas: &mut Option<usize>) {
        loop {
            match self.peek() {
                TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => {
                    self.advance();
                }
                TokenKind::SegGs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegGs; }
                TokenKind::SegFs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegFs; }
                TokenKind::Alignas => {
                    self.advance();
                    if let Some(align) = self.parse_alignas_argument() {
                        *alignas = Some(alignas.map_or(align, |prev| prev.max(align)));
                    }
                }
                TokenKind::Attribute => {
                    let (_, attr_aligned, _, _) = self.parse_gcc_attributes();
                    if attr_aligned.is_some() {
                        *alignas = attr_aligned;
                    }
                }
                TokenKind::Extension => { self.advance(); }
                _ => break,
            }
        }
    }

    /// For simple derived declarators (just pointers and/or arrays), fold them
    /// into the TypeSpecifier directly to maintain backward compatibility with
    /// downstream code. For complex cases (function pointers), return the
    /// derived list for downstream to process with build_full_ctype().
    fn fold_simple_derived(base: &TypeSpecifier, derived: &[DerivedDeclarator]) -> (TypeSpecifier, Vec<DerivedDeclarator>) {
        // If derived contains any function-related declarators, pass it through
        let has_function = derived.iter().any(|d| matches!(d,
            DerivedDeclarator::Function(_, _) | DerivedDeclarator::FunctionPointer(_, _)));

        if has_function {
            return (base.clone(), derived.to_vec());
        }

        if derived.is_empty() {
            return (base.clone(), Vec::new());
        }

        // Simple case: only Pointer and Array declarators. Fold into type_spec.
        let mut result = base.clone();
        let mut i = 0;
        while i < derived.len() {
            match &derived[i] {
                DerivedDeclarator::Pointer => {
                    result = TypeSpecifier::Pointer(Box::new(result), AddressSpace::Default);
                    i += 1;
                }
                DerivedDeclarator::Array(_) => {
                    // Collect consecutive array dims, apply in reverse (innermost first)
                    let start = i;
                    while i < derived.len() && matches!(&derived[i], DerivedDeclarator::Array(_)) {
                        i += 1;
                    }
                    for j in (start..i).rev() {
                        if let DerivedDeclarator::Array(size_expr) = &derived[j] {
                            result = TypeSpecifier::Array(Box::new(result), size_expr.clone());
                        }
                    }
                }
                _ => { i += 1; }
            }
        }
        (result, Vec::new())
    }

    /// Parse enum variant declarations inside braces.
    pub(super) fn parse_enum_variants(&mut self) -> Vec<EnumVariant> {
        let mut variants = Vec::new();
        let open = self.peek_span();
        self.expect_context(&TokenKind::LBrace, "for enum body");
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            if let TokenKind::Identifier(name) = self.peek() {
                let name = name.clone();
                self.advance();
                let value = if self.consume_if(&TokenKind::Assign) {
                    Some(Box::new(self.parse_assignment_expr()))
                } else {
                    None
                };
                variants.push(EnumVariant { name, value });
                self.consume_if(&TokenKind::Comma);
            } else {
                self.advance();
            }
        }
        self.expect_closing(&TokenKind::RBrace, open);
        variants
    }

    /// Register enum constant values from parsed variants into the parser's
    /// enum_constants map. This allows later constant expressions (e.g., in
    /// __attribute__((aligned(1 << ENUM_CONST)))) to resolve these identifiers.
    pub(super) fn register_enum_constants(&mut self, variants: &[super::ast::EnumVariant]) {
        let mut next_value: Option<i64> = Some(0);
        for variant in variants {
            let evaluated = if let Some(ref expr) = variant.value {
                // Evaluate the explicit value expression.
                // Use eval_const_int_expr_with_enums so references to previously
                // defined enum constants are resolved.
                let tag_aligns = if self.struct_tag_alignments.is_empty() { None } else { Some(&self.struct_tag_alignments) };
                Self::eval_const_int_expr_with_enums(expr, Some(&self.enum_constants), tag_aligns)
            } else {
                next_value
            };
            if let Some(val) = evaluated {
                self.enum_constants.insert(variant.name.clone(), val);
                next_value = Some(val + 1);
            } else {
                // Value not evaluable at parse time (e.g., sizeof(struct)).
                // Don't store in the value map, but record the name so that
                // expr_has_non_const_identifier knows this is a constant (not a variable).
                self.unevaluable_enum_constants.insert(variant.name.clone());
                next_value = None;
            }
        }
    }

    /// Parse a type-name for __builtin_va_arg: type-specifier + abstract declarator.
    pub(super) fn parse_va_arg_type(&mut self) -> TypeSpecifier {
        if let Some(type_spec) = self.parse_type_specifier() {
            let mut result_type = type_spec;
            // Parse pointer declarators
            while self.consume_if(&TokenKind::Star) {
                result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                self.skip_cv_qualifiers();
            }
            // Handle function pointer: type (*)(args)
            if matches!(self.peek(), TokenKind::LParen) {
                let save2 = self.pos;
                self.advance();
                if self.consume_if(&TokenKind::Star) {
                    while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
                        self.advance();
                    }
                    self.consume_if(&TokenKind::RParen);
                    if matches!(self.peek(), TokenKind::LParen) {
                        self.skip_balanced_parens();
                    }
                    result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                } else {
                    self.pos = save2;
                }
            }
            // Parse array dimensions
            while matches!(self.peek(), TokenKind::LBracket) {
                let open = self.peek_span();
                self.advance();
                let size = if matches!(self.peek(), TokenKind::RBracket) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()))
                };
                self.expect_closing(&TokenKind::RBracket, open);
                result_type = TypeSpecifier::Array(Box::new(result_type), size);
            }
            result_type
        } else {
            let span = self.peek_span();
            self.emit_error("expected type in __builtin_va_arg", span);
            TypeSpecifier::Int
        }
    }

    /// Parse an abstract declarator suffix: pointer(s), parenthesized pointer groups,
    /// and array dimensions after a type name. Used by cast expressions, sizeof,
    /// typeof, and _Alignof to avoid duplicating this logic.
    ///
    /// Input: base type already parsed.
    /// Output: type wrapped with pointer/array/function-pointer modifiers.
    pub(super) fn parse_abstract_declarator_suffix(&mut self, mut result_type: TypeSpecifier) -> TypeSpecifier {
        // Consume address space qualifiers that appear before the first '*'
        // e.g., typeof(var) __seg_gs * → __seg_gs sets parsing_address_space
        self.skip_cv_qualifiers();
        // Parse leading pointer(s)
        while self.consume_if(&TokenKind::Star) {
            // Capture any address space qualifier that preceded the '*'
            // (e.g., __seg_gs in "typeof(var) __seg_gs *")
            let addr_space = std::mem::take(&mut self.attrs.parsing_address_space);
            result_type = TypeSpecifier::Pointer(Box::new(result_type), addr_space);
            self.skip_cv_qualifiers();
        }
        // Handle parenthesized abstract declarators: (*), (*)(params), (*)[N], (*[3][4])
        // Also handles nested function pointers: (*(*)(params))(params)
        if matches!(self.peek(), TokenKind::LParen) {
            let save = self.pos;
            if let Some(paren_decl) = self.try_parse_paren_abstract_declarator() {
                use super::declarators::ParenAbstractDecl;
                match paren_decl {
                    ParenAbstractDecl::Simple { ptr_depth, array_dims: inner_array_dims } => {
                        if matches!(self.peek(), TokenKind::LParen) {
                            // Function pointer cast: (*)(params) or (**)(params)
                            let (params, variadic) = self.parse_param_list();
                            result_type = TypeSpecifier::FunctionPointer(Box::new(result_type), params, variadic);
                            // Extra pointer levels for multi-indirection (e.g., ** means ptr-to-func-ptr)
                            for _ in 0..ptr_depth.saturating_sub(1) {
                                result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                            }
                            // Wrap with inner array dims (for array of function pointers)
                            for dim in inner_array_dims.into_iter().rev() {
                                result_type = TypeSpecifier::Array(Box::new(result_type), dim);
                            }
                        } else if matches!(self.peek(), TokenKind::LBracket) || !inner_array_dims.is_empty() {
                            // Pointer to array: (*)[N] or (*[3][4])[2]
                            let mut outer_dims: Vec<Option<Box<Expr>>> = Vec::new();
                            while matches!(self.peek(), TokenKind::LBracket) {
                                let open_bracket = self.peek_span();
                                self.advance();
                                let size = if matches!(self.peek(), TokenKind::RBracket) {
                                    None
                                } else {
                                    Some(Box::new(self.parse_expr()))
                                };
                                self.expect_closing(&TokenKind::RBracket, open_bracket);
                                outer_dims.push(size);
                            }
                            for dim in outer_dims.into_iter().rev() {
                                result_type = TypeSpecifier::Array(Box::new(result_type), dim);
                            }
                            for _ in 0..ptr_depth {
                                result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                            }
                            for dim in inner_array_dims.into_iter().rev() {
                                result_type = TypeSpecifier::Array(Box::new(result_type), dim);
                            }
                        } else {
                            for _ in 0..ptr_depth {
                                result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                            }
                        }
                    }
                    ParenAbstractDecl::NestedFnPtr {
                        outer_ptr_depth, inner_ptr_depth,
                        inner_params, inner_variadic,
                    } => {
                        // Nested function pointer: (*(*)(inner_params))(outer_params)
                        // e.g., void(*(*)(void*))(void):
                        //   base = void, inner_params = [void*], outer_params = [void]
                        //   Result: Pointer(FunctionPointer(
                        //       return_type = FunctionPointer(return_type=void, params=[], variadic=false),
                        //       params = [Pointer(Void)],
                        //       variadic = false))
                        if matches!(self.peek(), TokenKind::LParen) {
                            let (outer_params, outer_variadic) = self.parse_param_list();
                            // Build the return type: function pointer returning base type
                            for _ in 0..inner_ptr_depth.saturating_sub(1) {
                                result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                            }
                            let return_fn_type = TypeSpecifier::FunctionPointer(
                                Box::new(result_type), outer_params, outer_variadic
                            );
                            // Build the outer function: takes inner_params, returns return_fn_type
                            result_type = TypeSpecifier::FunctionPointer(
                                Box::new(return_fn_type), inner_params, inner_variadic
                            );
                            // Apply extra outer pointer levels (for outer_ptr_depth > 1)
                            for _ in 0..outer_ptr_depth.saturating_sub(1) {
                                result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                            }
                        } else {
                            // No outer params - treat as simple pointer
                            let total = outer_ptr_depth + inner_ptr_depth;
                            for _ in 0..total {
                                result_type = TypeSpecifier::Pointer(Box::new(result_type), AddressSpace::Default);
                            }
                        }
                    }
                }
            } else {
                self.pos = save;
            }
        }
        // Parse trailing array dimensions, collecting them first so we can
        // apply in reverse order. In C, `int[1][3]` is "1 element of int[3]",
        // so the rightmost dimension wraps first (innermost).
        let mut array_dims: Vec<Option<Box<Expr>>> = Vec::new();
        while matches!(self.peek(), TokenKind::LBracket) {
            let open = self.peek_span();
            self.advance();
            let size = if matches!(self.peek(), TokenKind::RBracket) {
                None
            } else {
                Some(Box::new(self.parse_expr()))
            };
            self.expect_closing(&TokenKind::RBracket, open);
            array_dims.push(size);
        }
        // Apply in reverse: innermost (rightmost) dimension wraps first
        for dim in array_dims.into_iter().rev() {
            result_type = TypeSpecifier::Array(Box::new(result_type), dim);
        }
        result_type
    }
}
