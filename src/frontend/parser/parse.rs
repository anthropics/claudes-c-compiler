// Core Parser struct and basic helpers.
//
// The parser is split into focused modules:
//   - expressions.rs: operator precedence climbing (comma through primary)
//   - types.rs: type specifier collection and resolution
//   - statements.rs: all statement types + inline assembly
//   - declarations.rs: external and local declarations, initializers
//   - declarators.rs: C declarator syntax (pointers, arrays, function pointers)
//
// Each module adds methods to the Parser struct via `impl Parser` blocks.
// Methods are pub(super) so they can be called across modules within the parser.

use std::rc::Rc;
use crate::common::error::DiagnosticEngine;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::source::Span;
use crate::common::types::AddressSpace;
use crate::frontend::lexer::token::{Token, TokenKind};
use super::ast::*;

/// GCC __attribute__((mode(...))) integer mode specifier.
/// Controls the bit-width of an integer type regardless of the base type keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ModeKind {
    QI,  // 8-bit (quarter integer)
    HI,  // 16-bit (half integer)
    SI,  // 32-bit (single integer)
    DI,  // 64-bit (double integer)
    TI,  // 128-bit (tetra integer)
}

impl ModeKind {
    /// Apply mode to a type specifier, preserving signedness.
    pub(super) fn apply(self, ts: TypeSpecifier) -> TypeSpecifier {
        let is_unsigned = matches!(ts,
            TypeSpecifier::UnsignedInt | TypeSpecifier::UnsignedLong
            | TypeSpecifier::UnsignedLongLong | TypeSpecifier::Unsigned
            | TypeSpecifier::UnsignedChar | TypeSpecifier::UnsignedShort
        );
        match self {
            ModeKind::QI => if is_unsigned { TypeSpecifier::UnsignedChar } else { TypeSpecifier::Char },
            ModeKind::HI => if is_unsigned { TypeSpecifier::UnsignedShort } else { TypeSpecifier::Short },
            ModeKind::SI => if is_unsigned { TypeSpecifier::UnsignedInt } else { TypeSpecifier::Int },
            ModeKind::DI => if is_unsigned { TypeSpecifier::UnsignedLongLong } else { TypeSpecifier::LongLong },
            ModeKind::TI => if is_unsigned { TypeSpecifier::UnsignedInt128 } else { TypeSpecifier::Int128 },
        }
    }
}

/// Bit masks for boolean flags in `ParsedDeclAttrs::flags`.
///
/// Each flag occupies one bit of a `u32`, matching the pattern used by
/// `FunctionAttributes`, `Declaration`, and `DeclAttributes` in ast.rs.
/// New flags can be added by defining the next power-of-two constant.
pub(super) mod parsed_attr_flag {
    // --- Storage-class specifiers ---
    /// `typedef` keyword encountered.
    pub const TYPEDEF: u32          = 1 << 0;
    /// `static` keyword encountered.
    pub const STATIC: u32           = 1 << 1;
    /// `extern` keyword encountered.
    pub const EXTERN: u32           = 1 << 2;
    /// `_Thread_local` or `__thread` encountered.
    pub const THREAD_LOCAL: u32     = 1 << 3;
    /// `inline` keyword encountered.
    pub const INLINE: u32           = 1 << 4;

    // --- Type qualifiers ---
    /// `const` qualifier encountered.
    pub const CONST: u32            = 1 << 5;
    /// `volatile` qualifier encountered.
    pub const VOLATILE: u32         = 1 << 6;

    // --- GCC function attributes ---
    /// `__attribute__((constructor))` encountered.
    pub const CONSTRUCTOR: u32      = 1 << 7;
    /// `__attribute__((destructor))` encountered.
    pub const DESTRUCTOR: u32       = 1 << 8;
    /// `__attribute__((weak))` encountered.
    pub const WEAK: u32             = 1 << 9;
    /// `__attribute__((used))` encountered.
    pub const USED: u32             = 1 << 10;
    /// `__attribute__((gnu_inline))` encountered.
    pub const GNU_INLINE: u32       = 1 << 11;
    /// `__attribute__((always_inline))` encountered.
    pub const ALWAYS_INLINE: u32    = 1 << 12;
    /// `__attribute__((noinline))` encountered.
    pub const NOINLINE: u32         = 1 << 13;
    /// `__attribute__((noreturn))` or `_Noreturn` encountered.
    pub const NORETURN: u32         = 1 << 14;
    /// `__attribute__((error("...")))` encountered.
    pub const ERROR_ATTR: u32       = 1 << 15;
    /// `__attribute__((transparent_union))` encountered.
    pub const TRANSPARENT_UNION: u32 = 1 << 16;
    /// `__attribute__((fastcall))` encountered (i386 fastcall calling convention).
    pub const FASTCALL: u32         = 1 << 17;
    /// `__attribute__((naked))` encountered — emit no prologue/epilogue.
    pub const NAKED: u32            = 1 << 18;
}

/// Accumulated storage-class specifiers, type qualifiers, and GCC attributes
/// parsed during declaration/type-specifier processing.
///
/// These fields are set by `parse_type_specifier` and `parse_gcc_attribute` as
/// keywords/attributes are encountered, then consumed and reset by declaration
/// builders in `declarations.rs`. Grouping them into a single struct replaces
/// scattered fields on `Parser`, enabling bulk reset via `Default::default()`
/// and making the "set then consume" lifecycle explicit.
///
/// Boolean attributes are stored as a packed bitfield (`flags`) for memory
/// efficiency — 19 booleans collapse from 19 bytes into 4 bytes. Accessor
/// methods provide the same API as the old struct fields.
#[derive(Default)]
pub(super) struct ParsedDeclAttrs {
    /// Packed boolean flags — see `parsed_attr_flag` constants.
    pub(super) flags: u32,

    /// `__seg_gs` or `__seg_fs` qualifier encountered.
    pub parsing_address_space: AddressSpace,

    // --- GCC attributes with values ---
    /// `__attribute__((alias("target")))` target symbol name.
    pub parsing_alias_target: Option<String>,
    /// `__attribute__((visibility("...")))` visibility string.
    pub parsing_visibility: Option<String>,
    /// `__attribute__((section("...")))` section name.
    pub parsing_section: Option<String>,
    /// `__attribute__((cleanup(func)))` cleanup function name.
    pub parsing_cleanup_fn: Option<Rc<str>>,
    /// `__attribute__((symver("name@@VERSION")))` symbol version string.
    pub parsing_symver: Option<String>,
    /// `__attribute__((vector_size(N)))` total vector size in bytes.
    pub parsing_vector_size: Option<usize>,
    /// `__attribute__((ext_vector_type(N)))` number of vector elements.
    /// Converted to total byte size in lowering using sizeof(element_type) * N.
    pub parsing_ext_vector_nelem: Option<usize>,

    // --- Alignment ---
    /// `_Alignas(N)` or `__attribute__((aligned(N)))` value.
    pub parsed_alignas: Option<usize>,
    /// `_Alignas(type)` type specifier (for deferred alignment resolution).
    pub parsed_alignas_type: Option<TypeSpecifier>,
    /// `__attribute__((aligned(sizeof(type))))` type (for deferred sizeof).
    pub parsed_alignment_sizeof_type: Option<TypeSpecifier>,
}


impl ParsedDeclAttrs {
    // --- flag getters ---

    #[inline] pub fn parsing_typedef(&self) -> bool          { self.flags & parsed_attr_flag::TYPEDEF != 0 }
    #[inline] pub fn parsing_static(&self) -> bool           { self.flags & parsed_attr_flag::STATIC != 0 }
    #[inline] pub fn parsing_extern(&self) -> bool           { self.flags & parsed_attr_flag::EXTERN != 0 }
    #[inline] pub fn parsing_thread_local(&self) -> bool     { self.flags & parsed_attr_flag::THREAD_LOCAL != 0 }
    #[inline] pub fn parsing_inline(&self) -> bool           { self.flags & parsed_attr_flag::INLINE != 0 }
    #[inline] pub fn parsing_const(&self) -> bool            { self.flags & parsed_attr_flag::CONST != 0 }
    #[inline] pub fn parsing_volatile(&self) -> bool         { self.flags & parsed_attr_flag::VOLATILE != 0 }
    #[inline] pub fn parsing_constructor(&self) -> bool      { self.flags & parsed_attr_flag::CONSTRUCTOR != 0 }
    #[inline] pub fn parsing_destructor(&self) -> bool       { self.flags & parsed_attr_flag::DESTRUCTOR != 0 }
    #[inline] pub fn parsing_weak(&self) -> bool             { self.flags & parsed_attr_flag::WEAK != 0 }
    #[inline] pub fn parsing_used(&self) -> bool             { self.flags & parsed_attr_flag::USED != 0 }
    #[inline] pub fn parsing_gnu_inline(&self) -> bool       { self.flags & parsed_attr_flag::GNU_INLINE != 0 }
    #[inline] pub fn parsing_always_inline(&self) -> bool    { self.flags & parsed_attr_flag::ALWAYS_INLINE != 0 }
    #[inline] pub fn parsing_noinline(&self) -> bool         { self.flags & parsed_attr_flag::NOINLINE != 0 }
    #[inline] pub fn parsing_noreturn(&self) -> bool         { self.flags & parsed_attr_flag::NORETURN != 0 }
    #[inline] pub fn parsing_error_attr(&self) -> bool       { self.flags & parsed_attr_flag::ERROR_ATTR != 0 }
    #[inline] pub fn parsing_transparent_union(&self) -> bool { self.flags & parsed_attr_flag::TRANSPARENT_UNION != 0 }
    #[inline] pub fn parsing_fastcall(&self) -> bool         { self.flags & parsed_attr_flag::FASTCALL != 0 }
    #[inline] pub fn parsing_naked(&self) -> bool            { self.flags & parsed_attr_flag::NAKED != 0 }

    // --- flag setters ---

    #[inline] pub fn set_typedef(&mut self, v: bool)          { self.set_flag(parsed_attr_flag::TYPEDEF, v) }
    #[inline] pub fn set_static(&mut self, v: bool)           { self.set_flag(parsed_attr_flag::STATIC, v) }
    #[inline] pub fn set_extern(&mut self, v: bool)           { self.set_flag(parsed_attr_flag::EXTERN, v) }
    #[inline] pub fn set_thread_local(&mut self, v: bool)     { self.set_flag(parsed_attr_flag::THREAD_LOCAL, v) }
    #[inline] pub fn set_inline(&mut self, v: bool)           { self.set_flag(parsed_attr_flag::INLINE, v) }
    #[inline] pub fn set_const(&mut self, v: bool)            { self.set_flag(parsed_attr_flag::CONST, v) }
    #[inline] pub fn set_volatile(&mut self, v: bool)         { self.set_flag(parsed_attr_flag::VOLATILE, v) }
    #[inline] pub fn set_constructor(&mut self, v: bool)      { self.set_flag(parsed_attr_flag::CONSTRUCTOR, v) }
    #[inline] pub fn set_destructor(&mut self, v: bool)       { self.set_flag(parsed_attr_flag::DESTRUCTOR, v) }
    #[inline] pub fn set_weak(&mut self, v: bool)             { self.set_flag(parsed_attr_flag::WEAK, v) }
    #[inline] pub fn set_used(&mut self, v: bool)             { self.set_flag(parsed_attr_flag::USED, v) }
    #[inline] pub fn set_gnu_inline(&mut self, v: bool)       { self.set_flag(parsed_attr_flag::GNU_INLINE, v) }
    #[inline] pub fn set_always_inline(&mut self, v: bool)    { self.set_flag(parsed_attr_flag::ALWAYS_INLINE, v) }
    #[inline] pub fn set_noinline(&mut self, v: bool)         { self.set_flag(parsed_attr_flag::NOINLINE, v) }
    #[inline] pub fn set_noreturn(&mut self, v: bool)         { self.set_flag(parsed_attr_flag::NORETURN, v) }
    #[inline] pub fn set_error_attr(&mut self, v: bool)       { self.set_flag(parsed_attr_flag::ERROR_ATTR, v) }
    #[inline] pub fn set_transparent_union(&mut self, v: bool) { self.set_flag(parsed_attr_flag::TRANSPARENT_UNION, v) }
    #[inline] pub fn set_fastcall(&mut self, v: bool)         { self.set_flag(parsed_attr_flag::FASTCALL, v) }
    #[inline] pub fn set_naked(&mut self, v: bool)           { self.set_flag(parsed_attr_flag::NAKED, v) }

    #[inline]
    fn set_flag(&mut self, mask: u32, v: bool) {
        if v { self.flags |= mask; } else { self.flags &= !mask; }
    }

    /// Save the packed boolean flags for later restoration.
    /// Used to prevent storage-class flags from leaking out of compound
    /// statements and typeof expressions.
    #[inline]
    pub fn save_flags(&self) -> u32 {
        self.flags
    }

    /// Restore previously saved packed boolean flags.
    #[inline]
    pub fn restore_flags(&mut self, saved: u32) {
        self.flags = saved;
    }
}

impl std::fmt::Debug for ParsedDeclAttrs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParsedDeclAttrs")
            .field("parsing_typedef", &self.parsing_typedef())
            .field("parsing_static", &self.parsing_static())
            .field("parsing_extern", &self.parsing_extern())
            .field("parsing_thread_local", &self.parsing_thread_local())
            .field("parsing_inline", &self.parsing_inline())
            .field("parsing_const", &self.parsing_const())
            .field("parsing_volatile", &self.parsing_volatile())
            .field("parsing_address_space", &self.parsing_address_space)
            .field("parsing_constructor", &self.parsing_constructor())
            .field("parsing_destructor", &self.parsing_destructor())
            .field("parsing_weak", &self.parsing_weak())
            .field("parsing_used", &self.parsing_used())
            .field("parsing_gnu_inline", &self.parsing_gnu_inline())
            .field("parsing_always_inline", &self.parsing_always_inline())
            .field("parsing_noinline", &self.parsing_noinline())
            .field("parsing_noreturn", &self.parsing_noreturn())
            .field("parsing_error_attr", &self.parsing_error_attr())
            .field("parsing_transparent_union", &self.parsing_transparent_union())
            .field("parsing_fastcall", &self.parsing_fastcall())
            .field("parsing_alias_target", &self.parsing_alias_target)
            .field("parsing_visibility", &self.parsing_visibility)
            .field("parsing_section", &self.parsing_section)
            .field("parsing_cleanup_fn", &self.parsing_cleanup_fn)
            .field("parsing_vector_size", &self.parsing_vector_size)
            .field("parsing_ext_vector_nelem", &self.parsing_ext_vector_nelem)
            .field("parsed_alignas", &self.parsed_alignas)
            .field("parsed_alignas_type", &self.parsed_alignas_type)
            .field("parsed_alignment_sizeof_type", &self.parsed_alignment_sizeof_type)
            .finish()
    }
}

/// Recursive descent parser for C.
pub struct Parser {
    pub(super) tokens: Vec<Token>,
    pub(super) pos: usize,
    pub(super) typedefs: FxHashSet<Rc<str>>,
    /// Typedef names shadowed by local variable declarations in the current scope.
    pub(super) shadowed_typedefs: FxHashSet<Rc<str>>,
    /// Accumulated declaration attributes from the current parse_type_specifier pass.
    /// Reset at the start of each top-level or local declaration.
    pub(super) attrs: ParsedDeclAttrs,
    /// Stack for #pragma pack alignment values.
    /// Current effective alignment is the last element (or None for default).
    pub(super) pragma_pack_stack: Vec<Option<usize>>,
    /// Current #pragma pack alignment. None means default (natural) alignment.
    pub(super) pragma_pack_align: Option<usize>,
    /// Stack for #pragma GCC visibility push/pop.
    /// Each entry is the visibility string (e.g., "hidden", "default").
    pub(super) pragma_visibility_stack: Vec<String>,
    /// Current default visibility from #pragma GCC visibility push(...).
    /// None means default visibility (no pragma active).
    pub(super) pragma_default_visibility: Option<String>,
    /// Count of parse errors encountered (invalid tokens at top level, etc.)
    pub error_count: usize,
    /// Structured diagnostic engine for error/warning reporting with source snippets.
    pub(super) diagnostics: DiagnosticEngine,
    /// Map of enum constant names to their integer values.
    /// Populated as enum definitions are parsed, so that later constant expressions
    /// (e.g., in __attribute__((aligned(1 << ENUM_CONST)))) can resolve them.
    pub(super) enum_constants: FxHashMap<Rc<str>, i64>,
    /// Set of enum constant names whose values couldn't be evaluated at parse time
    /// (e.g., `MY_SIZE = sizeof(some_typedef)`). These are still valid constants,
    /// just not evaluable by our constant-expression evaluator.
    pub(super) unevaluable_enum_constants: FxHashSet<Rc<str>>,
    /// Map of struct/union tag names to their computed alignments.
    /// Populated when a struct/union with fields is parsed, so that later
    /// __alignof__(struct tag) references can look up the correct alignment
    /// (especially important for packed structs where tag-only refs would
    /// otherwise incorrectly default to ptr_size).
    pub(super) struct_tag_alignments: FxHashMap<Rc<str>, usize>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            typedefs: Self::builtin_typedefs(),
            shadowed_typedefs: FxHashSet::default(),
            attrs: ParsedDeclAttrs::default(),
            pragma_pack_stack: Vec::new(),
            pragma_pack_align: None,
            pragma_visibility_stack: Vec::new(),
            pragma_default_visibility: None,
            error_count: 0,
            diagnostics: DiagnosticEngine::new(),
            enum_constants: FxHashMap::default(),
            unevaluable_enum_constants: FxHashSet::default(),
            struct_tag_alignments: FxHashMap::default(),
        }
    }

    /// Set a pre-configured diagnostic engine on the parser.
    /// Called by the driver after setting up the source manager on the engine.
    pub fn set_diagnostics(&mut self, engine: DiagnosticEngine) {
        self.diagnostics = engine;
    }

    /// Take the diagnostic engine back from the parser (transfers ownership).
    /// The driver uses this to continue using the same engine across phases.
    pub fn take_diagnostics(&mut self) -> DiagnosticEngine {
        std::mem::take(&mut self.diagnostics)
    }

    /// Emit a parse error at the given span. Updates error_count and prints
    /// the error with source location and snippet (if source manager is set).
    pub(super) fn emit_error(&mut self, message: impl Into<String>, span: Span) {
        self.error_count += 1;
        self.diagnostics.error(message, span);
    }

    /// Standard C typedef names commonly provided by system headers.
    /// Since we don't actually include system headers, we pre-seed these.
    fn builtin_typedefs() -> FxHashSet<Rc<str>> {
        [
            // <stddef.h>
            "size_t", "ssize_t", "ptrdiff_t", "wchar_t", "wint_t",
            // <stdint.h>
            "int8_t", "int16_t", "int32_t", "int64_t",
            "uint8_t", "uint16_t", "uint32_t", "uint64_t",
            "intptr_t", "uintptr_t",
            "intmax_t", "uintmax_t",
            "int_least8_t", "int_least16_t", "int_least32_t", "int_least64_t",
            "uint_least8_t", "uint_least16_t", "uint_least32_t", "uint_least64_t",
            "int_fast8_t", "int_fast16_t", "int_fast32_t", "int_fast64_t",
            "uint_fast8_t", "uint_fast16_t", "uint_fast32_t", "uint_fast64_t",
            // <stdio.h>
            "FILE", "fpos_t",
            // <signal.h>
            "sig_atomic_t",
            // <time.h>
            "time_t", "clock_t", "timer_t", "clockid_t",
            // <sys/types.h>
            "off_t", "pid_t", "uid_t", "gid_t", "mode_t", "dev_t", "ino_t",
            "nlink_t", "blksize_t", "blkcnt_t",
            // GNU/glibc common types
            "ulong", "ushort", "uint",
            "__u8", "__u16", "__u32", "__u64",
            "__s8", "__s16", "__s32", "__s64",
            // <stdarg.h>
            "va_list", "__builtin_va_list", "__gnuc_va_list",
            // <locale.h>
            "locale_t",
            // <pthread.h>
            "pthread_t", "pthread_mutex_t", "pthread_cond_t",
            "pthread_key_t", "pthread_attr_t", "pthread_once_t",
            "pthread_mutexattr_t", "pthread_condattr_t",
            // <setjmp.h>
            "jmp_buf", "sigjmp_buf",
            // <dirent.h>
            "DIR",
            // GCC builtin NEON vector types (AArch64).
            // These are compiler-internal types used by bits/math-vector.h when
            // __GNUC_PREREQ(9, 0). They only appear in typedef/function declarations
            // behind #ifdef __aarch64__ guards, so listing them unconditionally is safe.
            "__Float32x4_t", "__Float64x2_t",
            // GCC builtin SVE scalable vector types (AArch64).
            // Used by bits/math-vector.h when __GNUC_PREREQ(10, 0).
            // SVE types are runtime-sized in real GCC, but we model them as fixed
            // 16-byte vectors for parsing (the functions using these are never called).
            "__SVFloat32_t", "__SVFloat64_t", "__SVBool_t",
            "__SVInt8_t", "__SVInt16_t", "__SVInt32_t", "__SVInt64_t",
            "__SVUint8_t", "__SVUint16_t", "__SVUint32_t", "__SVUint64_t",
            "__SVFloat16_t",
        ].iter().map(|s| Rc::from(*s)).collect()
    }

    pub fn parse(&mut self) -> TranslationUnit {
        let mut decls = Vec::new();
        while !self.at_eof() {
            if let Some(decl) = self.parse_external_decl() {
                decls.push(decl);
            } else {
                // Report error for unrecognized token at top level
                if !matches!(self.peek(), TokenKind::Semicolon | TokenKind::Eof) {
                    let span = self.peek_span();
                    self.emit_error(format!("expected declaration before {}", self.peek()), span);
                }
                self.advance();
            }
        }
        TranslationUnit { decls }
    }

    // === Token access helpers ===

    pub(super) fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.tokens[self.pos].kind, TokenKind::Eof)
    }

    pub(super) fn peek(&self) -> &TokenKind {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos].kind
        } else {
            &TokenKind::Eof
        }
    }

    pub(super) fn peek_span(&self) -> Span {
        if self.pos < self.tokens.len() {
            self.tokens[self.pos].span
        } else {
            Span::dummy()
        }
    }

    pub(super) fn advance(&mut self) -> &Token {
        if self.pos < self.tokens.len() {
            let tok = &self.tokens[self.pos];
            self.pos += 1;
            tok
        } else {
            &self.tokens[self.tokens.len() - 1]
        }
    }

    pub(super) fn expect(&mut self, expected: &TokenKind) -> Span {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(expected) {
            let span = self.peek_span();
            self.advance();
            span
        } else {
            let span = self.peek_span();
            self.emit_error(format!("expected {} before {}", expected, self.peek()), span);
            span
        }
    }

    /// Expect a token with a contextual description of what construct we're parsing.
    /// Produces messages like "expected ';' after return statement" instead of
    /// the generic "expected ';' before '}'".
    ///
    /// For semicolons: if the next token starts a new statement (e.g., '}', keyword),
    /// the error points at the previous token's end position and suggests inserting
    /// the missing token, which matches GCC's behavior.
    pub(super) fn expect_after(&mut self, expected: &TokenKind, context: &str) -> Span {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(expected) {
            let span = self.peek_span();
            self.advance();
            span
        } else {
            let span = self.peek_span();
            let diag = crate::common::error::Diagnostic::error(
                format!("expected {} {} before {}", expected, context, self.peek())
            )
            .with_span(span)
            .with_fix_hint(format!("insert {}", expected));
            self.error_count += 1;
            self.diagnostics.emit(&diag);
            span
        }
    }

    /// Expect a closing delimiter (paren, brace, bracket) and if missing,
    /// attach a note pointing back to where the opening delimiter was,
    /// plus a fix-it hint suggesting the insertion.
    /// Produces messages like:
    ///   error: expected ')' before ';'
    ///   note: to match this '(' (at file.c:10:5)
    ///   fix-it hint: insert ')'
    pub(super) fn expect_closing(&mut self, expected: &TokenKind, open_span: Span) -> Span {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(expected) {
            let span = self.peek_span();
            self.advance();
            span
        } else {
            let span = self.peek_span();
            let open_tok = match expected {
                TokenKind::RParen => "'('",
                TokenKind::RBrace => "'{'",
                TokenKind::RBracket => "'['",
                _ => "'?'",
            };
            let diag = crate::common::error::Diagnostic::error(
                format!("expected {} before {}", expected, self.peek())
            )
            .with_span(span)
            .with_fix_hint(format!("insert {}", expected))
            .with_note(
                crate::common::error::Diagnostic::note(
                    format!("to match this {}", open_tok)
                ).with_span(open_span)
            );
            self.error_count += 1;
            self.diagnostics.emit(&diag);
            span
        }
    }

    /// Expect a token with a contextual description of the construct being parsed.
    /// Unlike `expect()`, produces messages like "expected '(' after 'if'" instead of
    /// the generic "expected '(' before 'x'", and includes a fix-it hint.
    pub(super) fn expect_context(&mut self, expected: &TokenKind, context: &str) -> Span {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(expected) {
            let span = self.peek_span();
            self.advance();
            span
        } else {
            let span = self.peek_span();
            let diag = crate::common::error::Diagnostic::error(
                format!("expected {} {} before {}", expected, context, self.peek())
            )
            .with_span(span)
            .with_fix_hint(format!("insert {}", expected));
            self.error_count += 1;
            self.diagnostics.emit(&diag);
            span
        }
    }

    pub(super) fn consume_if(&mut self, kind: &TokenKind) -> bool {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    // === Type and qualifier helpers ===

    /// Check if the current position is a typedef name followed by ':',
    /// which means it's a label (not a declaration). In C, label names
    /// can shadow typedef names: `typedef struct Ins Ins; ... Ins: stmt;`
    pub(super) fn is_typedef_label(&self) -> bool {
        if let TokenKind::Identifier(name) = self.peek() {
            if self.typedefs.contains(name) && !self.shadowed_typedefs.contains(name) {
                // Check if next token is ':'
                if self.pos + 1 < self.tokens.len() {
                    return matches!(self.tokens[self.pos + 1].kind, TokenKind::Colon);
                }
            }
        }
        false
    }

    pub(super) fn is_type_specifier(&self) -> bool {
        match self.peek() {
            TokenKind::Void | TokenKind::Char | TokenKind::Short | TokenKind::Int |
            TokenKind::Long | TokenKind::Float | TokenKind::Double | TokenKind::Signed |
            TokenKind::Unsigned | TokenKind::Struct | TokenKind::Union | TokenKind::Enum |
            TokenKind::Const | TokenKind::Volatile | TokenKind::Static | TokenKind::Extern |
            TokenKind::Register | TokenKind::Typedef | TokenKind::Inline | TokenKind::Bool |
            TokenKind::Typeof | TokenKind::Attribute | TokenKind::Extension |
            TokenKind::Noreturn | TokenKind::Restrict | TokenKind::Complex |
            TokenKind::Atomic | TokenKind::Auto | TokenKind::AutoType | TokenKind::Alignas |
            TokenKind::Builtin | TokenKind::Int128 | TokenKind::UInt128 |
            TokenKind::ThreadLocal | TokenKind::SegGs | TokenKind::SegFs => true,
            TokenKind::Identifier(name) => self.typedefs.contains(name) && !self.shadowed_typedefs.contains(name),
            _ => false,
        }
    }

    pub(super) fn skip_cv_qualifiers(&mut self) {
        loop {
            match self.peek() {
                TokenKind::Const | TokenKind::Restrict => {
                    self.advance();
                }
                TokenKind::Volatile => {
                    self.advance();
                    // Propagate volatile to the declaration so that
                    // `T *volatile p` marks `p` as volatile (preventing
                    // mem2reg from promoting it to a register).
                    self.attrs.set_volatile(true);
                }
                TokenKind::SegGs => {
                    self.advance();
                    self.attrs.parsing_address_space = AddressSpace::SegGs;
                }
                TokenKind::SegFs => {
                    self.advance();
                    self.attrs.parsing_address_space = AddressSpace::SegFs;
                }
                _ => break,
            }
        }
    }

    /// Skip C99 type qualifiers and 'static' inside array brackets.
    /// In C99 function parameter declarations, array dimensions can include:
    ///   [static restrict const 10], [restrict n], [const], [static 10], etc.
    /// We skip these qualifiers since they only affect optimization hints and
    /// don't change the type semantics (array params decay to pointers).
    pub(super) fn skip_array_qualifiers(&mut self) {
        while let TokenKind::Static | TokenKind::Const | TokenKind::Volatile
            | TokenKind::Restrict | TokenKind::Atomic = self.peek()
        {
            self.advance();
        }
    }

    pub(super) fn skip_array_dimensions(&mut self) {
        while matches!(self.peek(), TokenKind::LBracket) {
            self.advance();
            while !matches!(self.peek(), TokenKind::RBracket | TokenKind::Eof) {
                self.advance();
            }
            self.consume_if(&TokenKind::RBracket);
        }
    }

    pub(super) fn compound_assign_op(&self) -> Option<BinOp> {
        match self.peek() {
            TokenKind::PlusAssign => Some(BinOp::Add),
            TokenKind::MinusAssign => Some(BinOp::Sub),
            TokenKind::StarAssign => Some(BinOp::Mul),
            TokenKind::SlashAssign => Some(BinOp::Div),
            TokenKind::PercentAssign => Some(BinOp::Mod),
            TokenKind::AmpAssign => Some(BinOp::BitAnd),
            TokenKind::PipeAssign => Some(BinOp::BitOr),
            TokenKind::CaretAssign => Some(BinOp::BitXor),
            TokenKind::LessLessAssign => Some(BinOp::Shl),
            TokenKind::GreaterGreaterAssign => Some(BinOp::Shr),
            _ => None,
        }
    }

    // === GCC extension helpers ===

    pub(super) fn skip_gcc_extensions(&mut self) {
        let (_, aligned, _, _) = self.parse_gcc_attributes();
        if let Some(a) = aligned {
            self.attrs.parsed_alignas = Some(self.attrs.parsed_alignas.map_or(a, |prev| prev.max(a)));
        }
    }

    /// Parse __attribute__((...)) and __extension__, returning struct attribute flags.
    /// Returns (is_packed, aligned_value, mode_kind, is_common).
    pub(super) fn parse_gcc_attributes(&mut self) -> (bool, Option<usize>, Option<ModeKind>, bool) {
        let mut is_packed = false;
        let mut aligned = None;
        let mut mode_kind: Option<ModeKind> = None;
        let mut is_common = false;
        loop {
            match self.peek() {
                TokenKind::Extension => { self.advance(); }
                TokenKind::Attribute => {
                    self.advance();
                    if matches!(self.peek(), TokenKind::LParen) {
                        self.advance(); // outer (
                        if matches!(self.peek(), TokenKind::LParen) {
                            self.advance(); // inner (
                            self.parse_gcc_attribute_list(
                                &mut is_packed, &mut aligned, &mut mode_kind, &mut is_common,
                            );
                            if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
                        } else {
                            // Single-paren form
                            while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
                                if let TokenKind::Identifier(name) = self.peek() {
                                    if &**name == "packed" || &**name == "__packed__" {
                                        is_packed = true;
                                    }
                                }
                                self.advance();
                            }
                        }
                        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
                    }
                }
                _ => break,
            }
        }
        (is_packed, aligned, mode_kind, is_common)
    }

    /// Parse the comma-separated attribute list inside __attribute__((...)).
    fn parse_gcc_attribute_list(&mut self, is_packed: &mut bool, aligned: &mut Option<usize>,
                                mode_kind: &mut Option<ModeKind>, is_common: &mut bool) {
        loop {
            match self.peek() {
                TokenKind::Comma => { self.advance(); }
                TokenKind::RParen | TokenKind::Eof => break,
                TokenKind::Noreturn => {
                    self.attrs.set_noreturn(true);
                    self.advance();
                }
                TokenKind::Identifier(name) => {
                    self.dispatch_gcc_attribute(&name.clone(), is_packed, aligned, mode_kind, is_common);
                }
                _ => { self.advance(); }
            }
        }
    }

    /// Dispatch a single GCC attribute by name.
    fn dispatch_gcc_attribute(&mut self, name: &str, is_packed: &mut bool,
                              aligned: &mut Option<usize>, mode_kind: &mut Option<ModeKind>,
                              is_common: &mut bool) {
        match name {
            "constructor" | "__constructor__" => {
                self.attrs.set_constructor(true);
                self.advance();
                if matches!(self.peek(), TokenKind::LParen) { self.skip_balanced_parens(); }
            }
            "destructor" | "__destructor__" => {
                self.attrs.set_destructor(true);
                self.advance();
                if matches!(self.peek(), TokenKind::LParen) { self.skip_balanced_parens(); }
            }
            "packed" | "__packed__" => { *is_packed = true; self.advance(); }
            "aligned" | "__aligned__" => {
                self.advance();
                if matches!(self.peek(), TokenKind::LParen) {
                    if let Some(align) = self.parse_alignment_expr() { *aligned = Some(align); }
                }
            }
            "common" | "__common__" => { *is_common = true; self.advance(); }
            "transparent_union" | "__transparent_union__" => { self.attrs.set_transparent_union(true); self.advance(); }
            "weak" | "__weak__" => { self.attrs.set_weak(true); self.advance(); }
            "alias" | "__alias__" => {
                self.advance();
                self.attrs.parsing_alias_target = self.parse_string_attr_arg();
            }
            "weakref" | "__weakref__" => {
                self.attrs.set_weak(true);
                self.advance();
                self.attrs.parsing_alias_target = self.parse_string_attr_arg();
            }
            "visibility" | "__visibility__" => {
                self.advance();
                self.attrs.parsing_visibility = self.parse_string_attr_arg();
            }
            "section" | "__section__" => {
                self.advance();
                self.attrs.parsing_section = self.parse_string_attr_arg();
            }
            "symver" | "__symver__" => {
                self.advance();
                self.attrs.parsing_symver = self.parse_string_attr_arg();
            }
            "error" | "__error__" | "warning" | "__warning__" => {
                self.attrs.set_error_attr(true);
                self.advance();
                self.skip_optional_paren_arg();
            }
            "noreturn" => { self.attrs.set_noreturn(true); self.advance(); }
            "gnu_inline" | "__gnu_inline__" => { self.attrs.set_gnu_inline(true); self.advance(); }
            "always_inline" | "__always_inline__" => { self.attrs.set_always_inline(true); self.advance(); }
            "noinline" | "__noinline__" => { self.attrs.set_noinline(true); self.advance(); }
            "cleanup" | "__cleanup__" => { self.advance(); self.parse_cleanup_attr(); }
            "mode" | "__mode__" => { self.advance(); self.parse_mode_attr(mode_kind); }
            "vector_size" | "__vector_size__" => { self.advance(); self.parse_vector_size_attr(); }
            "ext_vector_type" | "__ext_vector_type__" => { self.advance(); self.parse_ext_vector_type_attr(); }
            "used" | "__used__" => { self.attrs.set_used(true); self.advance(); }
            "fastcall" | "__fastcall__" => { self.attrs.set_fastcall(true); self.advance(); }
            "naked" | "__naked__" => { self.attrs.set_naked(true); self.advance(); }
            "optimize" | "__optimize__" => {
                self.advance();
                // Recognize optimize("omit-frame-pointer") as equivalent to naked.
                // GCC uses this attribute on functions like nlr_push whose inline asm
                // assumes (%rsp) is the return address (no frame pointer push).
                let mut found_omit_fp = false;
                if matches!(self.peek(), TokenKind::LParen) {
                    // Peek at the string argument inside parens without consuming
                    if self.pos + 1 < self.tokens.len() {
                        if let TokenKind::StringLiteral(ref s) = self.tokens[self.pos + 1].kind {
                            if s.contains("omit-frame-pointer") {
                                found_omit_fp = true;
                            }
                        }
                    }
                    self.skip_balanced_parens();
                }
                if found_omit_fp {
                    self.attrs.set_naked(true);
                }
            }
            "address_space" | "__address_space__" => { self.advance(); self.parse_address_space_attr(); }
            _ => {
                self.advance();
                if matches!(self.peek(), TokenKind::LParen) { self.skip_balanced_parens(); }
            }
        }
    }

    /// Parse a parenthesized string argument: ("string1" "string2"...).
    /// Returns Some(concatenated) if non-empty, None otherwise.
    fn parse_string_attr_arg(&mut self) -> Option<String> {
        if !matches!(self.peek(), TokenKind::LParen) { return None; }
        self.advance(); // consume (
        let mut result = String::new();
        while let TokenKind::StringLiteral(s) = self.peek() {
            result.push_str(s);
            self.advance();
        }
        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
        if result.is_empty() { None } else { Some(result) }
    }

    /// Skip an optional parenthesized argument (consuming everything inside).
    fn skip_optional_paren_arg(&mut self) {
        if !matches!(self.peek(), TokenKind::LParen) { return; }
        self.advance();
        while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) {
            self.advance();
        }
        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
    }

    /// Parse cleanup(func_name) attribute.
    fn parse_cleanup_attr(&mut self) {
        if !matches!(self.peek(), TokenKind::LParen) { return; }
        self.advance();
        if let TokenKind::Identifier(func_name) = self.peek() {
            self.attrs.parsing_cleanup_fn = Some(func_name.clone());
            self.advance();
        }
        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
    }

    /// Parse mode(QI|HI|SI|DI|TI|word|pointer) attribute.
    fn parse_mode_attr(&mut self, mode_kind: &mut Option<ModeKind>) {
        if !matches!(self.peek(), TokenKind::LParen) { return; }
        self.advance();
        if let TokenKind::Identifier(mode_name) = self.peek() {
            let is_32bit = crate::common::types::target_is_32bit();
            *mode_kind = match &**mode_name {
                "QI" | "__QI__" | "byte" | "__byte__" => Some(ModeKind::QI),
                "HI" | "__HI__" => Some(ModeKind::HI),
                "SI" | "__SI__" => Some(ModeKind::SI),
                "DI" | "__DI__" => Some(ModeKind::DI),
                "TI" | "__TI__" => {
                    if is_32bit {
                        let span = self.peek_span();
                        self.emit_error("TI mode is not supported on 32-bit targets", span);
                        *mode_kind
                    }
                    else { Some(ModeKind::TI) }
                }
                "word" | "__word__" | "pointer" | "__pointer__" => {
                    if is_32bit { Some(ModeKind::SI) } else { Some(ModeKind::DI) }
                }
                _ => *mode_kind,
            };
        }
        while !matches!(self.peek(), TokenKind::RParen | TokenKind::Eof) { self.advance(); }
        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
    }

    /// Parse vector_size(expr) attribute.
    fn parse_vector_size_attr(&mut self) {
        if !matches!(self.peek(), TokenKind::LParen) { return; }
        self.advance();
        let expr = self.parse_assignment_expr();
        let enums = if self.enum_constants.is_empty() { None } else { Some(&self.enum_constants) };
        let tag_aligns = if self.struct_tag_alignments.is_empty() { None } else { Some(&self.struct_tag_alignments) };
        if let Some(size) = Self::eval_const_int_expr_with_enums(&expr, enums, tag_aligns) {
            self.attrs.parsing_vector_size = Some(size as usize);
        }
        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
    }

    /// Parse ext_vector_type(N) attribute (Clang-style vector type).
    /// Stores the element count N; the total size is computed in lowering as N * sizeof(elem).
    fn parse_ext_vector_type_attr(&mut self) {
        if !matches!(self.peek(), TokenKind::LParen) { return; }
        self.advance();
        let expr = self.parse_assignment_expr();
        let enums = if self.enum_constants.is_empty() { None } else { Some(&self.enum_constants) };
        let tag_aligns = if self.struct_tag_alignments.is_empty() { None } else { Some(&self.struct_tag_alignments) };
        if let Some(nelem) = Self::eval_const_int_expr_with_enums(&expr, enums, tag_aligns) {
            self.attrs.parsing_ext_vector_nelem = Some(nelem as usize);
        }
        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
    }

    /// Parse address_space(__seg_gs|__seg_fs) attribute.
    fn parse_address_space_attr(&mut self) {
        if !matches!(self.peek(), TokenKind::LParen) { return; }
        self.advance();
        match self.peek() {
            TokenKind::SegGs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegGs; }
            TokenKind::SegFs => { self.advance(); self.attrs.parsing_address_space = AddressSpace::SegFs; }
            _ => { self.advance(); }
        }
        if matches!(self.peek(), TokenKind::RParen) { self.advance(); }
    }

    /// Skip __asm__("..."), __attribute__(...), and __extension__ after declarators.
    /// Returns (mode_kind, aligned_value, asm_register).
    pub(super) fn skip_asm_and_attributes(&mut self) -> (Option<ModeKind>, Option<usize>, Option<String>) {
        let (_, _, mk, _, aligned, asm_reg) = self.parse_asm_and_attributes();
        (mk, aligned, asm_reg)
    }

    /// Parse __asm__("..."), __attribute__(...), and __extension__ after declarators.
    /// Returns (is_constructor, is_destructor, mode_kind, is_common, aligned_value, asm_register).
    /// The asm_register captures the register name from `register var __asm__("regname")`.
    pub(super) fn parse_asm_and_attributes(&mut self) -> (bool, bool, Option<ModeKind>, bool, Option<usize>, Option<String>) {
        let mut is_constructor = false;
        let mut is_destructor = false;
        let mut mode_kind: Option<ModeKind> = None;
        let mut has_common = false;
        let mut aligned: Option<usize> = None;
        let mut asm_register: Option<String> = None;
        loop {
            match self.peek() {
                TokenKind::Asm => {
                    self.advance();
                    self.consume_if(&TokenKind::Volatile);
                    if matches!(self.peek(), TokenKind::LParen) {
                        // Try to extract the asm register name: __asm__("regname")
                        // This is a single string literal inside parentheses.
                        asm_register = asm_register.or_else(|| self.try_parse_asm_register_name());
                    }
                }
                TokenKind::Attribute => {
                    let (_, attr_aligned, mk, common) = self.parse_gcc_attributes();
                    mode_kind = mode_kind.or(mk);
                    has_common = has_common || common;
                    if let Some(a) = attr_aligned {
                        aligned = Some(aligned.map_or(a, |prev| prev.max(a)));
                    }
                    if self.attrs.parsing_constructor() { is_constructor = true; }
                    if self.attrs.parsing_destructor() { is_destructor = true; }
                }
                TokenKind::Extension => {
                    self.advance();
                }
                _ => break,
            }
        }
        (is_constructor, is_destructor, mode_kind, has_common, aligned, asm_register)
    }

    /// Try to extract a label/register name from __asm__("name") on a declaration.
    /// Called when the current token is LParen after __asm__. Handles both:
    /// - Register pinning: `register int x __asm__("rbx")`
    /// - Linker symbol redirect: `extern int foo(...) __asm__("" "__xpg_strerror_r")`
    ///
    /// Supports concatenation of adjacent string literals (e.g., `__asm__("" "name")`).
    /// Returns Some("name") if a non-empty label was found, None otherwise.
    fn try_parse_asm_register_name(&mut self) -> Option<String> {
        // Save position so we can fall back
        let saved_pos = self.pos;
        self.advance(); // consume '('
        // Concatenate adjacent string literals: __asm__("" "name") -> "name"
        let mut combined = String::new();
        let mut found_string = false;
        while let TokenKind::StringLiteral(s) = self.peek() {
            combined.push_str(s);
            found_string = true;
            self.advance();
        }
        if found_string && matches!(self.peek(), TokenKind::RParen) {
            self.advance(); // consume ')'
            if !combined.is_empty() {
                // Strip leading '%' prefix from register names.
                // In C source, register names in asm() may use the GCC inline asm
                // convention with a '%' prefix (e.g., asm("%rdx") or asm("%" "rdx")),
                // but the actual register name used internally should be just "rdx".
                let combined = combined.trim_start_matches('%').to_string();
                if !combined.is_empty() {
                    return Some(combined);
                }
            }
            return None;
        }
        // Not a simple string literal sequence, restore and skip
        self.pos = saved_pos;
        self.skip_balanced_parens();
        None
    }

    // === Pragma pack handling ===

    /// Check if current token is a pragma pack directive and handle it.
    /// Returns true if a pragma pack token was consumed.
    pub(super) fn handle_pragma_pack_token(&mut self) -> bool {
        match self.peek() {
            TokenKind::PragmaPackSet(n) => {
                let n = *n;
                self.advance();
                // pack(0) means reset to default (natural alignment)
                self.pragma_pack_align = if n == 0 { None } else { Some(n) };
                true
            }
            TokenKind::PragmaPackPush(n) => {
                let n = *n;
                self.advance();
                // Push current alignment onto stack
                self.pragma_pack_stack.push(self.pragma_pack_align);
                // pack(push, 0) means push and reset to default (natural alignment)
                // pack(push, N) means push and set to N
                if n == 0 {
                    self.pragma_pack_align = None;
                } else {
                    self.pragma_pack_align = Some(n);
                }
                true
            }
            TokenKind::PragmaPackPushOnly => {
                self.advance();
                // pack(push) - push current alignment without changing it
                self.pragma_pack_stack.push(self.pragma_pack_align);
                true
            }
            TokenKind::PragmaPackPop => {
                self.advance();
                // Pop previous alignment from stack
                if let Some(prev) = self.pragma_pack_stack.pop() {
                    self.pragma_pack_align = prev;
                } else {
                    // Stack underflow: reset to default
                    self.pragma_pack_align = None;
                }
                true
            }
            TokenKind::PragmaPackReset => {
                self.advance();
                self.pragma_pack_align = None;
                true
            }
            _ => false,
        }
    }

    /// Handle #pragma GCC visibility push/pop synthetic tokens.
    /// Returns true if a token was consumed.
    pub(super) fn handle_pragma_visibility_token(&mut self) -> bool {
        match self.peek() {
            TokenKind::PragmaVisibilityPush(vis) => {
                let vis = vis.clone();
                self.advance();
                // Push current visibility and set new default
                if let Some(ref current) = self.pragma_default_visibility {
                    self.pragma_visibility_stack.push(current.clone());
                } else {
                    // Push a sentinel for "no pragma active"
                    self.pragma_visibility_stack.push(String::new());
                }
                if vis == "default" {
                    // "default" means no special visibility
                    self.pragma_default_visibility = None;
                } else {
                    self.pragma_default_visibility = Some(vis);
                }
                true
            }
            TokenKind::PragmaVisibilityPop => {
                self.advance();
                // Pop previous visibility
                if let Some(prev) = self.pragma_visibility_stack.pop() {
                    if prev.is_empty() {
                        self.pragma_default_visibility = None;
                    } else {
                        self.pragma_default_visibility = Some(prev);
                    }
                } else {
                    // Stack underflow: reset to no pragma
                    self.pragma_default_visibility = None;
                }
                true
            }
            _ => false,
        }
    }

    pub(super) fn skip_balanced_parens(&mut self) {
        if !matches!(self.peek(), TokenKind::LParen) {
            return;
        }
        let mut depth = 0i32;
        loop {
            match self.peek() {
                TokenKind::LParen => { depth += 1; self.advance(); }
                TokenKind::RParen => {
                    depth -= 1;
                    self.advance();
                    if depth <= 0 { break; }
                }
                TokenKind::Eof => break,
                _ => { self.advance(); }
            }
        }
    }

    /// Skip optional GNU label attributes after a label colon.
    /// GNU C allows `label: __attribute__((unused));` to suppress unused-label warnings.
    /// We consume the entire `__attribute__((...))` token sequence and discard it.
    pub(super) fn skip_label_attributes(&mut self) {
        while matches!(self.peek(), TokenKind::Attribute) {
            self.advance(); // consume __attribute__
            // Expect __attribute__((...)) — two levels of parens
            if matches!(self.peek(), TokenKind::LParen) {
                self.skip_balanced_parens();
            }
        }
    }

    /// Parse the parenthesized argument of `aligned(expr)` in __attribute__.
    /// Expects the opening `(` to be the current token (not yet consumed).
    /// Parses and evaluates a constant expression, consuming through the closing `)`.
    /// Returns Some(alignment) on success, None on failure.
    pub(super) fn parse_alignment_expr(&mut self) -> Option<usize> {
        if !matches!(self.peek(), TokenKind::LParen) {
            return None;
        }
        self.advance(); // consume opening (
        let expr = self.parse_assignment_expr();
        // Consume closing )
        if matches!(self.peek(), TokenKind::RParen) {
            self.advance();
        }
        // If the expression is sizeof(type), capture the type so sema/lowerer can
        // recompute with accurate struct/union layout info (the parser's sizeof
        // uses a conservative default for struct/union types).
        if let Expr::Sizeof(ref arg, _) = expr {
            if let SizeofArg::Type(ref ts) = **arg {
                self.attrs.parsed_alignment_sizeof_type = Some(ts.clone());
            }
        }
        let enums = if self.enum_constants.is_empty() { None } else { Some(&self.enum_constants) };
        let tag_aligns = if self.struct_tag_alignments.is_empty() { None } else { Some(&self.struct_tag_alignments) };
        Self::eval_const_int_expr_with_enums(&expr, enums, tag_aligns).map(|v| v as usize)
    }

    /// Parse the parenthesized argument of `_Alignas(...)`.
    /// _Alignas can take either a type-name or a constant expression.
    /// Returns Some(alignment) on success, None on failure.
    pub(super) fn parse_alignas_argument(&mut self) -> Option<usize> {
        if !matches!(self.peek(), TokenKind::LParen) {
            return None;
        }
        // Try type-name first using save/restore
        let save = self.pos;
        let save_typedef = self.attrs.parsing_typedef();
        self.advance(); // consume (
        if self.is_type_specifier() {
            if let Some(ts) = self.parse_type_specifier() {
                let result_type = self.parse_abstract_declarator_suffix(ts);
                if matches!(self.peek(), TokenKind::RParen) {
                    self.advance(); // consume )
                    // Save the type specifier so the lowerer can resolve typedefs
                    // and compute accurate alignment (parser can't resolve typedefs).
                    self.attrs.parsed_alignas_type = Some(result_type.clone());
                    let tag_aligns = if self.struct_tag_alignments.is_empty() { None } else { Some(&self.struct_tag_alignments) };
                    return Some(Self::alignof_type_spec(&result_type, tag_aligns));
                }
            }
        }
        // Backtrack and try as constant expression
        self.pos = save;
        self.attrs.set_typedef(save_typedef);
        self.advance(); // consume (
        let expr = self.parse_assignment_expr();
        if matches!(self.peek(), TokenKind::RParen) {
            self.advance();
        }
        let enums = if self.enum_constants.is_empty() { None } else { Some(&self.enum_constants) };
        let tag_aligns = if self.struct_tag_alignments.is_empty() { None } else { Some(&self.struct_tag_alignments) };
        Self::eval_const_int_expr_with_enums(&expr, enums, tag_aligns).map(|v| v as usize)
    }

    /// Try to compute sizeof for a type specifier. Returns None for types
    /// whose size cannot be determined at parse time (struct, union, typedef).
    /// Use this in constant expression evaluation where a wrong default would
    /// cause spurious errors (e.g., _Static_assert).
    pub(super) fn try_sizeof_type_spec(ts: &TypeSpecifier) -> Option<usize> {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        match ts {
            TypeSpecifier::Void | TypeSpecifier::Bool
            | TypeSpecifier::Char | TypeSpecifier::UnsignedChar => Some(1),
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => Some(2),
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Signed | TypeSpecifier::Unsigned
            | TypeSpecifier::Float => Some(4),
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong => Some(ptr_sz),
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong
            | TypeSpecifier::Double => Some(8),
            TypeSpecifier::Pointer(_, _) | TypeSpecifier::FunctionPointer(_, _, _) => Some(ptr_sz),
            TypeSpecifier::Int128 | TypeSpecifier::UnsignedInt128 => Some(16),
            TypeSpecifier::LongDouble => Some(if ptr_sz == 4 { 12 } else { 16 }),
            TypeSpecifier::ComplexFloat => Some(8),
            TypeSpecifier::ComplexDouble => Some(16),
            TypeSpecifier::ComplexLongDouble => Some(if ptr_sz == 4 { 24 } else { 32 }),
            TypeSpecifier::Array(elem, Some(size_expr)) => {
                let elem_size = Self::try_sizeof_type_spec(elem)?;
                let count = Self::eval_const_int_expr(size_expr)? as usize;
                Some(elem_size * count)
            }
            TypeSpecifier::Array(_, None) => Some(0),
            // Enum, struct, union, typedef: size not reliably known at parse time.
            // Enums can be __packed (1 byte), and struct/union sizes depend on layout.
            _ => None,
        }
    }

    /// Check if a type specifier is an unsigned type.
    /// Used by the parser-level constant evaluator for cast truncation.
    pub(super) fn is_unsigned_type_spec(ts: &TypeSpecifier) -> bool {
        matches!(ts,
            TypeSpecifier::UnsignedChar
            | TypeSpecifier::UnsignedShort
            | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Unsigned
            | TypeSpecifier::UnsignedLong
            | TypeSpecifier::UnsignedLongLong
            | TypeSpecifier::UnsignedInt128
            | TypeSpecifier::Bool
            | TypeSpecifier::Pointer(_, _)
        )
    }

    /// Compute alignment (in bytes) for a type specifier.
    /// Used by _Alignas(type) to determine the alignment value.
    /// The optional `tag_aligns` map provides previously-computed alignments for
    /// struct/union tags, enabling correct alignment for tag-only references
    /// (e.g., `__alignof__(struct packed_tag)` where the definition is elsewhere).
    pub(super) fn alignof_type_spec(
        ts: &TypeSpecifier,
        tag_aligns: Option<&FxHashMap<Rc<str>, usize>>,
    ) -> usize {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        match ts {
            TypeSpecifier::Void | TypeSpecifier::Bool
            | TypeSpecifier::Char | TypeSpecifier::UnsignedChar => 1,
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => 2,
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Signed | TypeSpecifier::Unsigned
            | TypeSpecifier::Float => 4,
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong => ptr_sz,
            // On i686 (ILP32), long long and double are aligned to 4 bytes per i386 SysV ABI
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong
            | TypeSpecifier::Double => if ptr_sz == 4 { 4 } else { 8 },
            TypeSpecifier::Pointer(_, _) | TypeSpecifier::FunctionPointer(_, _, _) => ptr_sz,
            TypeSpecifier::Int128 | TypeSpecifier::UnsignedInt128 => 16,
            // On i686, long double is 80-bit x87 but aligned to 4 bytes
            TypeSpecifier::LongDouble => if ptr_sz == 4 { 4 } else { 16 },
            TypeSpecifier::ComplexFloat => 4,
            TypeSpecifier::ComplexDouble => if ptr_sz == 4 { 4 } else { 8 },
            TypeSpecifier::ComplexLongDouble => if ptr_sz == 4 { 4 } else { 16 },
            TypeSpecifier::Array(elem, _) => Self::alignof_type_spec(elem, tag_aligns),
            TypeSpecifier::Struct(name, fields, is_packed, _, struct_aligned)
            | TypeSpecifier::Union(name, fields, is_packed, _, struct_aligned) => {
                if *is_packed { return 1; }
                // Explicit struct/union-level __attribute__((aligned(N))) overrides
                let mut align = struct_aligned.unwrap_or(0);
                // Compute max alignment from fields if available
                if let Some(field_list) = fields {
                    for field in field_list {
                        let field_align = if let Some(fa) = field.alignment {
                            fa
                        } else {
                            Self::alignof_type_spec(&field.type_spec, tag_aligns)
                        };
                        align = align.max(field_align);
                    }
                } else if let Some(tag_name) = name {
                    // Tag-only reference: look up previously stored alignment
                    if let Some(ta) = tag_aligns {
                        if let Some(&stored) = ta.get(&**tag_name) {
                            return stored;
                        }
                    }
                }
                // Fallback for empty struct/union or tag-only (no fields available)
                if align == 0 { ptr_sz } else { align }
            }
            TypeSpecifier::Enum(_, _, _) => 4,
            TypeSpecifier::TypedefName(_) => ptr_sz, // conservative default
            _ => ptr_sz,
        }
    }

    /// Compute preferred (natural) alignment for a type specifier.
    /// Used by GCC's __alignof/__alignof__ which returns preferred alignment.
    /// On i686: __alignof__(long long) == 8, __alignof__(double) == 8,
    /// while _Alignof returns 4 for both (minimum ABI alignment).
    pub(super) fn preferred_alignof_type_spec(
        ts: &TypeSpecifier,
        tag_aligns: Option<&FxHashMap<Rc<str>, usize>>,
    ) -> usize {
        use crate::common::types::target_ptr_size;
        let ptr_sz = target_ptr_size();
        if ptr_sz != 4 {
            // On 64-bit targets, preferred == ABI alignment
            return Self::alignof_type_spec(ts, tag_aligns);
        }
        // On i686: long long and double have preferred alignment of 8
        match ts {
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong
            | TypeSpecifier::Double => 8,
            TypeSpecifier::ComplexDouble => 8,
            TypeSpecifier::Array(elem, _) => Self::preferred_alignof_type_spec(elem, tag_aligns),
            _ => Self::alignof_type_spec(ts, tag_aligns),
        }
    }
}
