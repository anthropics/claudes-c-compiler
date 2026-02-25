use std::rc::Rc;
use crate::common::source::Span;

/// All token kinds recognized by the C lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    IntLiteral(i64),       // no suffix or value > i32::MAX
    UIntLiteral(u64),      // u/U suffix or value > i64::MAX
    LongLiteral(i64),      // l/L suffix (signed long)
    ULongLiteral(u64),     // ul/UL suffix (unsigned long)
    LongLongLiteral(i64),  // ll/LL suffix (signed long long, always 64-bit)
    ULongLongLiteral(u64), // ull/ULL suffix (unsigned long long, always 64-bit)
    FloatLiteral(f64),             // no suffix (double)
    FloatLiteralF32(f64),          // f/F suffix (float, 32-bit)
    /// Long double literal (l/L suffix). Stores (f64_approx, f128_bytes).
    /// f128_bytes is IEEE 754 binary128 format with full 112-bit mantissa precision.
    FloatLiteralLongDouble(f64, [u8; 16]),
    /// Imaginary double literal (e.g. 1.0i) - GCC extension
    ImaginaryLiteral(f64),
    /// Imaginary float literal (e.g. 1.0fi or 1.0if) - GCC extension
    ImaginaryLiteralF32(f64),
    /// Imaginary long double literal (e.g. 1.0Li or 1.0il) - GCC extension. Stores (f64_approx, f128_bytes).
    ImaginaryLiteralLongDouble(f64, [u8; 16]),
    StringLiteral(String),
    /// Wide string literal (L"..."), stores content as Rust chars (each becomes wchar_t = i32)
    WideStringLiteral(String),
    /// char16_t string literal (u"..."), stores content as Rust chars (each becomes char16_t = u16)
    Char16StringLiteral(String),
    CharLiteral(char),

    // Identifiers and keywords
    Identifier(Rc<str>),

    // Keywords
    Auto,
    Break,
    Case,
    Char,
    Const,
    Continue,
    Default,
    Do,
    Double,
    Else,
    Enum,
    Extern,
    Float,
    For,
    Goto,
    If,
    Inline,
    Int,
    Long,
    Register,
    Restrict,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Struct,
    Switch,
    Typedef,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,
    // C11 keywords
    Alignas,
    Alignof,
    Atomic,
    Bool,
    Complex,
    Generic,
    Imaginary,
    Noreturn,
    StaticAssert,
    ThreadLocal,

    // GCC extensions
    Typeof,
    Asm,
    Attribute,
    Extension,
    Builtin,         // __builtin_va_list (used as type name)
    BuiltinVaArg,    // __builtin_va_arg(expr, type) - special syntax
    BuiltinTypesCompatibleP, // __builtin_types_compatible_p(type, type) - special syntax
    /// __int128 / __int128_t type keyword (GCC extension, signed)
    Int128,
    /// __uint128_t type keyword (GCC extension, unsigned)
    UInt128,
    /// __real__ - extract real part of complex number (GCC extension)
    RealPart,
    /// __imag__ - extract imaginary part of complex number (GCC extension)
    ImagPart,
    /// __auto_type - GCC extension for type inference from initializer
    AutoType,
    /// __alignof / __alignof__ - GCC extension returning preferred (natural) alignment.
    /// Differs from C11 _Alignof which returns minimum ABI alignment.
    /// On i686: __alignof__(long long) == 8, _Alignof(long long) == 4.
    GnuAlignof,
    /// __label__ - GCC extension for local label declarations in block scope
    GnuLabel,
    /// __seg_gs - GCC named address space qualifier (x86 %gs segment)
    SegGs,
    /// __seg_fs - GCC named address space qualifier (x86 %fs segment)
    SegFs,

    /// #pragma pack directive, emitted by preprocessor as synthetic token.
    /// Variants: Set(N), Push(N), PushOnly (push without change), Pop, Reset (pack())
    PragmaPackSet(usize),
    PragmaPackPush(usize),
    /// #pragma pack(push) - push current alignment without changing it
    PragmaPackPushOnly,
    PragmaPackPop,
    PragmaPackReset,

    /// #pragma GCC visibility push(hidden|default|protected|internal), emitted by preprocessor.
    PragmaVisibilityPush(String),
    /// #pragma GCC visibility pop
    PragmaVisibilityPop,

    // Punctuation
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]
    Semicolon,  // ;
    Comma,      // ,
    Dot,        // .
    Arrow,      // ->
    Ellipsis,   // ...

    // Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Amp,        // &
    Pipe,       // |
    Caret,      // ^
    Tilde,      // ~
    Bang,       // !
    Assign,     // =
    Less,       // <
    Greater,    // >
    Question,   // ?
    Colon,      // :

    // Compound operators
    PlusPlus,   // ++
    MinusMinus, // --
    PlusAssign, // +=
    MinusAssign,// -=
    StarAssign, // *=
    SlashAssign,// /=
    PercentAssign, // %=
    AmpAssign,  // &=
    PipeAssign, // |=
    CaretAssign,// ^=
    LessLess,   // <<
    GreaterGreater, // >>
    LessLessAssign, // <<=
    GreaterGreaterAssign, // >>=
    EqualEqual, // ==
    BangEqual,  // !=
    LessEqual,  // <=
    GreaterEqual, // >=
    AmpAmp,     // &&
    PipePipe,   // ||
    Hash,       // # (used in preprocessor)
    HashHash,   // ## (used in preprocessor)

    // Special
    Eof,
}

/// A token with its kind and source span.
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn is_eof(&self) -> bool {
        matches!(self.kind, TokenKind::Eof)
    }
}

impl std::fmt::Display for TokenKind {
    /// Display tokens in GCC-style human-readable format for diagnostics.
    ///
    /// Punctuation and operators are shown as quoted symbols: `';'`, `')'`
    /// Keywords are shown as quoted keyword text: `'int'`, `'return'`
    /// Identifiers include the name: `'foo'`
    /// Literals are described generically: `integer constant`, `string literal`
    /// EOF is shown as: `end of input`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Literals - described generically like GCC
            TokenKind::IntLiteral(_) | TokenKind::UIntLiteral(_) |
            TokenKind::LongLiteral(_) | TokenKind::ULongLiteral(_) |
            TokenKind::LongLongLiteral(_) | TokenKind::ULongLongLiteral(_) =>
                write!(f, "integer constant"),
            TokenKind::FloatLiteral(_) | TokenKind::FloatLiteralF32(_) |
            TokenKind::FloatLiteralLongDouble(_, _) =>
                write!(f, "floating constant"),
            TokenKind::ImaginaryLiteral(_) | TokenKind::ImaginaryLiteralF32(_) |
            TokenKind::ImaginaryLiteralLongDouble(_, _) =>
                write!(f, "imaginary constant"),
            TokenKind::StringLiteral(_) => write!(f, "string literal"),
            TokenKind::WideStringLiteral(_) => write!(f, "wide string literal"),
            TokenKind::Char16StringLiteral(_) => write!(f, "char16_t string literal"),
            TokenKind::CharLiteral(_) => write!(f, "character constant"),

            // Identifiers
            TokenKind::Identifier(name) => write!(f, "'{}'", name),

            // Keywords - shown as quoted keyword text
            TokenKind::Auto => write!(f, "'auto'"),
            TokenKind::Break => write!(f, "'break'"),
            TokenKind::Case => write!(f, "'case'"),
            TokenKind::Char => write!(f, "'char'"),
            TokenKind::Const => write!(f, "'const'"),
            TokenKind::Continue => write!(f, "'continue'"),
            TokenKind::Default => write!(f, "'default'"),
            TokenKind::Do => write!(f, "'do'"),
            TokenKind::Double => write!(f, "'double'"),
            TokenKind::Else => write!(f, "'else'"),
            TokenKind::Enum => write!(f, "'enum'"),
            TokenKind::Extern => write!(f, "'extern'"),
            TokenKind::Float => write!(f, "'float'"),
            TokenKind::For => write!(f, "'for'"),
            TokenKind::Goto => write!(f, "'goto'"),
            TokenKind::If => write!(f, "'if'"),
            TokenKind::Inline => write!(f, "'inline'"),
            TokenKind::Int => write!(f, "'int'"),
            TokenKind::Long => write!(f, "'long'"),
            TokenKind::Register => write!(f, "'register'"),
            TokenKind::Restrict => write!(f, "'restrict'"),
            TokenKind::Return => write!(f, "'return'"),
            TokenKind::Short => write!(f, "'short'"),
            TokenKind::Signed => write!(f, "'signed'"),
            TokenKind::Sizeof => write!(f, "'sizeof'"),
            TokenKind::Static => write!(f, "'static'"),
            TokenKind::Struct => write!(f, "'struct'"),
            TokenKind::Switch => write!(f, "'switch'"),
            TokenKind::Typedef => write!(f, "'typedef'"),
            TokenKind::Union => write!(f, "'union'"),
            TokenKind::Unsigned => write!(f, "'unsigned'"),
            TokenKind::Void => write!(f, "'void'"),
            TokenKind::Volatile => write!(f, "'volatile'"),
            TokenKind::While => write!(f, "'while'"),

            // C11 keywords
            TokenKind::Alignas => write!(f, "'_Alignas'"),
            TokenKind::Alignof => write!(f, "'_Alignof'"),
            TokenKind::Atomic => write!(f, "'_Atomic'"),
            TokenKind::Bool => write!(f, "'_Bool'"),
            TokenKind::Complex => write!(f, "'_Complex'"),
            TokenKind::Generic => write!(f, "'_Generic'"),
            TokenKind::Imaginary => write!(f, "'_Imaginary'"),
            TokenKind::Noreturn => write!(f, "'_Noreturn'"),
            TokenKind::StaticAssert => write!(f, "'_Static_assert'"),
            TokenKind::ThreadLocal => write!(f, "'_Thread_local'"),

            // GCC extensions
            TokenKind::Typeof => write!(f, "'typeof'"),
            TokenKind::Asm => write!(f, "'asm'"),
            TokenKind::Attribute => write!(f, "'__attribute__'"),
            TokenKind::Extension => write!(f, "'__extension__'"),
            TokenKind::Builtin => write!(f, "'__builtin_va_list'"),
            TokenKind::BuiltinVaArg => write!(f, "'__builtin_va_arg'"),
            TokenKind::BuiltinTypesCompatibleP => write!(f, "'__builtin_types_compatible_p'"),
            TokenKind::Int128 => write!(f, "'__int128'"),
            TokenKind::UInt128 => write!(f, "'__uint128_t'"),
            TokenKind::RealPart => write!(f, "'__real__'"),
            TokenKind::ImagPart => write!(f, "'__imag__'"),
            TokenKind::AutoType => write!(f, "'__auto_type'"),
            TokenKind::GnuAlignof => write!(f, "'__alignof__'"),
            TokenKind::GnuLabel => write!(f, "'__label__'"),
            TokenKind::SegGs => write!(f, "'__seg_gs'"),
            TokenKind::SegFs => write!(f, "'__seg_fs'"),

            // Pragma tokens
            TokenKind::PragmaPackSet(_) | TokenKind::PragmaPackPush(_) |
            TokenKind::PragmaPackPushOnly | TokenKind::PragmaPackPop |
            TokenKind::PragmaPackReset => write!(f, "'#pragma pack'"),
            TokenKind::PragmaVisibilityPush(_) | TokenKind::PragmaVisibilityPop =>
                write!(f, "'#pragma GCC visibility'"),

            // Punctuation - shown as quoted symbols
            TokenKind::LParen => write!(f, "'('"),
            TokenKind::RParen => write!(f, "')'"),
            TokenKind::LBrace => write!(f, "'{{'"),
            TokenKind::RBrace => write!(f, "'}}'"),
            TokenKind::LBracket => write!(f, "'['"),
            TokenKind::RBracket => write!(f, "']'"),
            TokenKind::Semicolon => write!(f, "';'"),
            TokenKind::Comma => write!(f, "','"),
            TokenKind::Dot => write!(f, "'.'"),
            TokenKind::Arrow => write!(f, "'->'"),
            TokenKind::Ellipsis => write!(f, "'...'"),

            // Operators
            TokenKind::Plus => write!(f, "'+'"),
            TokenKind::Minus => write!(f, "'-'"),
            TokenKind::Star => write!(f, "'*'"),
            TokenKind::Slash => write!(f, "'/'"),
            TokenKind::Percent => write!(f, "'%'"),
            TokenKind::Amp => write!(f, "'&'"),
            TokenKind::Pipe => write!(f, "'|'"),
            TokenKind::Caret => write!(f, "'^'"),
            TokenKind::Tilde => write!(f, "'~'"),
            TokenKind::Bang => write!(f, "'!'"),
            TokenKind::Assign => write!(f, "'='"),
            TokenKind::Less => write!(f, "'<'"),
            TokenKind::Greater => write!(f, "'>'"),
            TokenKind::Question => write!(f, "'?'"),
            TokenKind::Colon => write!(f, "':'"),

            // Compound operators
            TokenKind::PlusPlus => write!(f, "'++'"),
            TokenKind::MinusMinus => write!(f, "'--'"),
            TokenKind::PlusAssign => write!(f, "'+='"),
            TokenKind::MinusAssign => write!(f, "'-='"),
            TokenKind::StarAssign => write!(f, "'*='"),
            TokenKind::SlashAssign => write!(f, "'/='"),
            TokenKind::PercentAssign => write!(f, "'%='"),
            TokenKind::AmpAssign => write!(f, "'&='"),
            TokenKind::PipeAssign => write!(f, "'|='"),
            TokenKind::CaretAssign => write!(f, "'^='"),
            TokenKind::LessLess => write!(f, "'<<'"),
            TokenKind::GreaterGreater => write!(f, "'>>'"),
            TokenKind::LessLessAssign => write!(f, "'<<='"),
            TokenKind::GreaterGreaterAssign => write!(f, "'>>='"),
            TokenKind::EqualEqual => write!(f, "'=='"),
            TokenKind::BangEqual => write!(f, "'!='"),
            TokenKind::LessEqual => write!(f, "'<='"),
            TokenKind::GreaterEqual => write!(f, "'>='"),
            TokenKind::AmpAmp => write!(f, "'&&'"),
            TokenKind::PipePipe => write!(f, "'||'"),
            TokenKind::Hash => write!(f, "'#'"),
            TokenKind::HashHash => write!(f, "'##'"),

            // Special
            TokenKind::Eof => write!(f, "end of input"),
        }
    }
}

impl TokenKind {
    /// Convert a keyword string to its token kind.
    /// When `gnu_extensions` is false (strict C standard mode, e.g. -std=c99),
    /// bare GNU keywords like `typeof` and `asm` are treated as identifiers.
    /// The double-underscore forms (`__typeof__`, `__asm__`) are always keywords.
    ///
    /// Performance: Uses a two-stage filter to quickly reject non-keywords.
    /// Stage 1: reject by length (keywords are 2-17 or 28 chars).
    /// Stage 2: reject by first character (only 16 possible first chars).
    /// Most identifiers in typical C code are rejected by these filters
    /// without entering the match statement.
    pub fn from_keyword(s: &str, gnu_extensions: bool) -> Option<TokenKind> {
        // Fast reject by length: C/GCC keywords have lengths 2-17 or 28.
        let len = s.len();
        if len < 2 || (len > 17 && len != 28) {
            return None;
        }
        // Fast reject by first character: keywords only start with these 16 chars.
        let first = s.as_bytes()[0];
        if !matches!(first,
            b'_' | b'a' | b'b' | b'c' | b'd' | b'e' | b'f' | b'g' |
            b'i' | b'l' | b'r' | b's' | b't' | b'u' | b'v' | b'w'
        ) {
            return None;
        }
        match s {
            "auto" => Some(TokenKind::Auto),
            "break" => Some(TokenKind::Break),
            "case" => Some(TokenKind::Case),
            "char" => Some(TokenKind::Char),
            "const" => Some(TokenKind::Const),
            "continue" => Some(TokenKind::Continue),
            "default" => Some(TokenKind::Default),
            "do" => Some(TokenKind::Do),
            "double" => Some(TokenKind::Double),
            "else" => Some(TokenKind::Else),
            "enum" => Some(TokenKind::Enum),
            "extern" => Some(TokenKind::Extern),
            "float" => Some(TokenKind::Float),
            "for" => Some(TokenKind::For),
            "goto" => Some(TokenKind::Goto),
            "if" => Some(TokenKind::If),
            "inline" => Some(TokenKind::Inline),
            "int" => Some(TokenKind::Int),
            "long" => Some(TokenKind::Long),
            "register" => Some(TokenKind::Register),
            "restrict" => Some(TokenKind::Restrict),
            "return" => Some(TokenKind::Return),
            "short" => Some(TokenKind::Short),
            "signed" => Some(TokenKind::Signed),
            "sizeof" => Some(TokenKind::Sizeof),
            "static" => Some(TokenKind::Static),
            "struct" => Some(TokenKind::Struct),
            "switch" => Some(TokenKind::Switch),
            "typedef" => Some(TokenKind::Typedef),
            "union" => Some(TokenKind::Union),
            "unsigned" => Some(TokenKind::Unsigned),
            "void" => Some(TokenKind::Void),
            "volatile" | "__volatile__" | "__volatile" => Some(TokenKind::Volatile),
            "__const" | "__const__" => Some(TokenKind::Const),
            "__inline" | "__inline__" => Some(TokenKind::Inline),
            "__restrict" | "__restrict__" => Some(TokenKind::Restrict),
            "__signed__" => Some(TokenKind::Signed),
            "while" => Some(TokenKind::While),
            "_Alignas" => Some(TokenKind::Alignas),
            "_Alignof" => Some(TokenKind::Alignof),
            "_Atomic" => Some(TokenKind::Atomic),
            "_Bool" => Some(TokenKind::Bool),
            "_Complex" | "__complex__" | "__complex" => Some(TokenKind::Complex),
            "_Generic" => Some(TokenKind::Generic),
            "_Imaginary" => Some(TokenKind::Imaginary),
            "_Noreturn" | "__noreturn__" => Some(TokenKind::Noreturn),
            "_Static_assert" | "static_assert" => Some(TokenKind::StaticAssert),
            "_Thread_local" | "__thread" => Some(TokenKind::ThreadLocal),
            "typeof" if gnu_extensions => Some(TokenKind::Typeof),
            "__typeof__" | "__typeof" => Some(TokenKind::Typeof),
            "asm" if gnu_extensions => Some(TokenKind::Asm),
            "__asm__" | "__asm" => Some(TokenKind::Asm),
            "__attribute__" | "__attribute" => Some(TokenKind::Attribute),
            "__extension__" => Some(TokenKind::Extension),
            "__builtin_va_list" => Some(TokenKind::Builtin),
            "__builtin_va_arg" => Some(TokenKind::BuiltinVaArg),
            "__builtin_types_compatible_p" => Some(TokenKind::BuiltinTypesCompatibleP),
            "__int128" | "__int128_t" => Some(TokenKind::Int128),
            "__uint128_t" => Some(TokenKind::UInt128),
            "__real__" | "__real" => Some(TokenKind::RealPart),
            "__imag__" | "__imag" => Some(TokenKind::ImagPart),
            "__auto_type" => Some(TokenKind::AutoType),
            "__alignof" | "__alignof__" => Some(TokenKind::GnuAlignof),
            "__label__" => Some(TokenKind::GnuLabel),
            "__seg_gs" => Some(TokenKind::SegGs),
            "__seg_fs" => Some(TokenKind::SegFs),
            // __builtin_va_start, __builtin_va_end, __builtin_va_copy remain as
            // Identifier tokens so they flow through the normal builtin call path
            _ => None,
        }
    }
}
