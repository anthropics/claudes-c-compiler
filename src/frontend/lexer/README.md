# Lexer

The lexer transforms preprocessed C source text into a flat sequence of tokens. It operates on a single byte buffer (the output of the preprocessor), scanning left-to-right in a single pass with no backtracking beyond local lookahead. Every token carries a `Span` that records its exact byte range and file identity, enabling precise diagnostics downstream.

The lexer lives in two main files (plus a minimal `mod.rs` that re-exports `Lexer`):

| File | Purpose |
|------|---------|
| `token.rs` | `Token`, `TokenKind` enum, keyword lookup table |
| `scan.rs` | `Lexer` struct and the scanning/parsing logic |

---

## Token Representation

A `Token` is a pair of a `TokenKind` and a `Span`:

```rust
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}
```

`TokenKind` is a large enum whose variants fall into nine categories:

### 1. Integer Literals

The type of an integer token is determined at lex time using the C11 promotion rules, which depend on the numeric value, the base (decimal vs. hex/octal), and any explicit suffix. Six variants cover the full matrix:

| Variant | C Type | Rust Payload |
|---------|--------|-------------|
| `IntLiteral(i64)` | `int` / `long` (signed, fits i64) | signed 64-bit |
| `UIntLiteral(u64)` | `unsigned int` | unsigned 64-bit |
| `LongLiteral(i64)` | `long` (signed) | signed 64-bit |
| `ULongLiteral(u64)` | `unsigned long` | unsigned 64-bit |
| `LongLongLiteral(i64)` | `long long` (signed, always 64-bit) | signed 64-bit |
| `ULongLongLiteral(u64)` | `unsigned long long` (always 64-bit) | unsigned 64-bit |

### 2. Floating-Point and Imaginary Literals

Three widths of float, each mirrored by an imaginary counterpart (GCC `_Complex` extension):

| Variant | Meaning |
|---------|---------|
| `FloatLiteral(f64)` | `double` (no suffix) |
| `FloatLiteralF32(f64)` | `float` (`f`/`F` suffix); stored as f64, narrowed later |
| `FloatLiteralLongDouble(f64, [u8; 16])` | `long double` (`l`/`L` suffix); carries both an f64 approximation and the full IEEE 754 binary128 bytes |
| `ImaginaryLiteral(f64)` | `_Complex double` imaginary part (`i`/`j` suffix) |
| `ImaginaryLiteralF32(f64)` | `_Complex float` imaginary part (`fi`/`if` suffix) |
| `ImaginaryLiteralLongDouble(f64, [u8; 16])` | `_Complex long double` imaginary part (`li`/`il` suffix) |

The `[u8; 16]` payload on long-double variants holds the constant in binary128 encoding so that downstream codegen does not lose mantissa bits through an f64 round-trip. Decimal constants get it from `common::long_double::parse_long_double_to_f128_bytes`. Hexadecimal constants get it from `common::long_double::hex_float_to_long_double`, which on x87 targets (x86-64, i686) stores the value already rounded to the 64-bit x87 significand.

### 3. String and Character Literals

| Variant | Source Syntax | Element Type |
|---------|--------------|-------------|
| `StringLiteral(String)` | `"..."` or `u8"..."` | `char` (byte) |
| `WideStringLiteral(String)` | `L"..."` or `U"..."` | `wchar_t` / `char32_t` (code point) |
| `Char16StringLiteral(String)` | `u"..."` | `char16_t` (16-bit) |
| `CharLiteral(char)` | `'x'` | `char` (single byte) |

Multi-character constants like `'AB'` produce an `IntLiteral` with the bytes packed left-to-right (matching GCC behavior).

### 4. Identifiers

Any identifier that is not a keyword produces `Identifier(String)`. This includes variable names, function names, type names, macro-expanded identifiers, and GCC builtins like `__builtin_va_start` that are handled as ordinary function calls rather than special syntax.

### 5. Keywords

All C89, C99, and C11 keywords have dedicated variants (`Auto`, `Break`, `Case`, `Char`, ..., `Bool`, `Complex`, `Generic`, `StaticAssert`, `ThreadLocal`, etc.). The C23 spelling `static_assert` (without underscore prefix) is also accepted alongside `_Static_assert`. The parser never string-compares against keyword text; it pattern-matches on enum variants directly.

### 6. GCC Extension Keywords

The lexer recognizes a broad set of GCC built-in keywords:

| Variant | Keyword(s) |
|---------|-----------|
| `Typeof` | `typeof`, `__typeof__`, `__typeof` |
| `Asm` | `asm`, `__asm__`, `__asm` |
| `Attribute` | `__attribute__`, `__attribute` |
| `Extension` | `__extension__` |
| `Builtin` | `__builtin_va_list` |
| `BuiltinVaArg` | `__builtin_va_arg` |
| `BuiltinTypesCompatibleP` | `__builtin_types_compatible_p` |
| `Int128` / `UInt128` | `__int128`, `__int128_t` / `__uint128_t` |
| `RealPart` / `ImagPart` | `__real__`, `__real` / `__imag__`, `__imag` |
| `AutoType` | `__auto_type` |
| `GnuAlignof` | `__alignof__`, `__alignof` |
| `GnuLabel` | `__label__` |
| `SegGs` / `SegFs` | `__seg_gs` / `__seg_fs` |

Double-underscore forms are always recognized. The bare forms `typeof` and `asm` are only keywords when `gnu_extensions` is enabled (the default); in strict standard mode (`-std=c99` etc.) they become ordinary identifiers.

GCC-style qualifier aliases (`__const`/`__const__`, `__volatile`/`__volatile__`, `__inline`/`__inline__`, `__restrict`/`__restrict__`, `__signed__`, `__complex`/`__complex__`, `__noreturn__`, `__thread`) map to their standard C counterparts. Both single- and double-underscore-suffix forms are accepted where shown.

### 7. Pragma Tokens

The preprocessor rewrites `#pragma pack(...)` and `#pragma GCC visibility ...` directives into synthetic identifier tokens (`__ccc_pack_set_N`, `__ccc_visibility_push_hidden`, etc.). The lexer's `lex_identifier` path recognizes these by prefix and emits structured pragma token variants:

- `PragmaPackSet(usize)`, `PragmaPackPush(usize)`, `PragmaPackPushOnly`, `PragmaPackPop`, `PragmaPackReset`
- `PragmaVisibilityPush(String)`, `PragmaVisibilityPop`

This keeps pragma semantics out of the preprocessor while giving the parser typed tokens to work with.

### 8. Punctuation and Operators

All C punctuation and operators have individual variants (`LParen`, `RParen`, `Plus`, `Arrow`, `Ellipsis`, `LessLessAssign`, etc.). There are no "generic operator" fallbacks -- every valid C operator has its own variant.

### 9. Special

`Eof` marks the end of input.

---

## Scanning Algorithm

The `Lexer` struct holds the input as a `Vec<u8>`, a byte position cursor, a file ID, and a GNU-extensions flag:

```rust
pub struct Lexer {
    input: Vec<u8>,
    pos: usize,
    file_id: u32,
    gnu_extensions: bool,
}
```

### Tokenization Loop

`tokenize()` pre-allocates a token vector (estimated at 1 token per 5 input bytes, a reasonable heuristic for C code) and repeatedly calls `next_token()` until `Eof`:

```
tokenize():
    tokens = Vec with capacity input.len() / 5
    loop:
        tok = next_token()
        tokens.push(tok)
        if tok is Eof: break
    return tokens
```

### Dispatch in `next_token`

Each call to `next_token` first skips whitespace and comments, then dispatches on the current byte:

1. **Digit or `.` followed by digit** -- number literal (`lex_number`)
2. **`"`** -- string literal (`lex_string`)
3. **`'`** -- character literal (`lex_char`)
4. **`_`, `$`, or ASCII letter** -- identifier or keyword (`lex_identifier`)
5. **Everything else** -- punctuation/operator (`lex_punctuation`)

This ordering means that `.5` is correctly lexed as a float (not dot-then-5), and string prefixes like `L"..."` are handled inside `lex_identifier` when it discovers a quote character immediately after a recognized prefix.

### Whitespace and Comment Skipping

`skip_whitespace_and_comments` runs in a loop, consuming:

- ASCII whitespace bytes
- GCC-style line markers (`# <digit>...`), which are preprocessor artifacts that must not become tokens. A line marker is identified by `#` at position 0 or immediately after a newline, followed by optional spaces and then a digit.
- Line comments (`//` to end of line)
- Block comments (`/*` to `*/`)

The loop restarts after each comment or marker, so interleaved whitespace and comments are consumed in one call.

---

## Number Literal Parsing

Number parsing starts in `lex_number` and immediately branches by prefix:

```
0x / 0X  -->  lex_hex_number
0b / 0B  -->  lex_binary_number
0[0-7]   -->  lex_octal_number (may fall back to decimal)
otherwise -->  lex_decimal_number
```

### Hexadecimal Integers and Hex Floats

`lex_hex_number` consumes `0x` then hex digits. It looks ahead for a `.` or `p`/`P` to decide between integer and hex-float. Hex floats follow the C99 format `0x<int>.<frac>p<exp>`. The lexer only finds the extent of the token and its `f`/`F`/`l`/`L` suffix; the value is then computed from the exact digit string (`common::long_double::hex_float_to_f64`, `hex_float_to_f32` and `hex_float_to_long_double`) with a single round-to-nearest-even to the type of the constant (C11 6.4.4.2p3: a hexadecimal floating constant is correctly rounded). The conversion is exact for any number of digits and any exponent: digits beyond what the format can hold only act as a sticky bit, results below the smallest normal number become subnormals, and results that are too large become infinity. A `long double` constant is rounded to the target's format, which is why the lexer consults the target's long double format: binary128 on AArch64/RISC-V, the 64-bit x87 significand on x86-64/i686.

Hex floats use a simplified suffix parser that recognizes `f`/`F` and `l`/`L` but does not handle imaginary suffixes (`i`/`j`). This matches typical usage -- imaginary hex floats are extremely rare in practice.

### Binary Integers

`lex_binary_number` consumes `0b` then binary digits (`0`/`1`) and parses via `u64::from_str_radix(s, 2)`.

### Octal Integers

`lex_octal_number` is called speculatively when the current token starts with `0` followed by a digit. It consumes octal digits (`0`-`7`), but if it encounters `8`, `9`, `.`, `e`, or `E`, it backtracks and lets `lex_decimal_number` handle the token instead. There is a special case: `0` followed by `...` (ellipsis, as in `case 0 ... 5:`) is correctly recognized as octal zero followed by the ellipsis operator, not a decimal-point trigger.

### Decimal Integers and Floats

`lex_decimal_number` consumes decimal digits, then checks for a decimal point (but *not* if the dot begins an `...` ellipsis -- this matters for GCC case ranges like `case 2...15`). If a decimal point or exponent (`e`/`E`) is found, the token becomes a float. Otherwise it is an integer.

### Integer Suffix Parsing

`parse_int_suffix` consumes any combination of:

- `u` / `U` (unsigned)
- `l` / `L` (long) or `ll` / `LL` (long long)
- `i` / `I` / `j` / `J` (imaginary, GCC extension)

The suffix order is flexible: `ULL`, `LLU`, `uLL`, etc. are all accepted. A standalone `i`/`I` suffix is recognized as imaginary only if the next character is not alphanumeric or underscore (to avoid consuming the `i` in an identifier like `int`).

### Integer Type Promotion

`make_int_token` implements the C11 SS6.4.4.1 integer promotion rules:

- **Decimal literals** without suffix: `int` -> `long` -> `long long`. Unsigned only if the value exceeds `i64::MAX`.
- **Hex/octal literals** without suffix: `int` -> `unsigned int` -> `long` -> `unsigned long`. The unsigned intermediate step is the key difference from decimal.
- **Explicit suffix** (`U`, `L`, `UL`, `LL`, `ULL`): the stated type is used, with further promotion only for hex/octal when the value overflows the signed range.
- **32-bit targets**: `L`-suffixed hex/octal literals follow `long` -> `unsigned long` -> `long long` -> `unsigned long long` promotion through the 32-bit long range.

### Float Suffix Parsing

`parse_float_suffix` handles:

| Suffix | Meaning | `float_kind` |
|--------|---------|:---:|
| (none) | `double` | 0 |
| `f` / `F` | `float` | 1 |
| `l` / `L` | `long double` | 2 |

Each of these can be combined with an imaginary suffix. The `i`/`I` suffix can appear before or after the type suffix (`1.0fi`, `1.0if`, `1.0Li`, `1.0iL`). The `j`/`J` suffix can appear after the type suffix or standalone (`1.0fj`, `1.0Lj`, `1.0j`).

---

## String and Character Literal Handling

### Narrow Strings (`"..."` and `u8"..."`)

`lex_string` processes the body character-by-character:

- **Escape sequences** (`\n`, `\t`, `\x4F`, `\0`, `\u00E9`, `\U0001F600`, etc.) are handled by `lex_escape_char`.
- **Unicode escapes in narrow strings**: `\u` and `\U` produce Unicode code points that are then UTF-8-encoded byte-by-byte into the string (matching GCC/Clang behavior for narrow string literals).
- **PUA-encoded bytes**: Non-UTF-8 source bytes (encoded as PUA code points by the encoding layer) are decoded back to their original byte values via `decode_pua_byte`.

`u8"..."` strings are lexed identically to `"..."` and produce the same `StringLiteral` token.

### Wide Strings (`L"..."`, `U"..."`)

`lex_wide_string` produces `WideStringLiteral` tokens. The key difference from narrow strings is that Unicode escapes store code points directly (not UTF-8-encoded), and multi-byte UTF-8 characters in the source are decoded to their Unicode code point rather than being stored as raw bytes.

### UTF-16 Strings (`u"..."`)

`lex_char16_string` produces `Char16StringLiteral` tokens. Parsing is identical to wide strings; the distinction is semantic -- downstream the code converts each character to a `u16`, truncating code points above 0xFFFF.

### Character Literals (`'x'`)

`lex_char` handles:

- **Single characters**: produce `CharLiteral(char)`.
- **Multi-character constants** (e.g., `'AB'`, `'ABCD'`): bytes are packed left-to-right with left-shift, producing an `IntLiteral` with the combined value. This matches GCC behavior.
- **Unicode escapes in narrow chars**: `\u` and `\U` values above 0xFF are UTF-8-encoded and the resulting bytes are packed into the multi-byte integer value, matching GCC.

### Wide Character Literals (`L'x'`, `U'x'`, `u'x'`)

Detected inside `lex_identifier` when a prefix (`L`, `U`, `u`) is immediately followed by `'`. `lex_wide_char` decodes the character to its Unicode code point and produces an `IntLiteral` (since `wchar_t` is `int` on the target platform).

### Escape Sequences

`lex_escape_char` supports the full set:

| Escape | Value | Notes |
|--------|-------|-------|
| `\n` `\t` `\r` `\\` `\'` `\"` | Standard | |
| `\a` `\b` `\f` `\v` | Standard | BEL, BS, FF, VT |
| `\e` `\E` | `0x1B` | GNU extension (ESC) |
| `\0`..`\377` | Octal | 1-3 octal digits, truncated to byte |
| `\xNN...` | Hex | Consumes all hex digits, truncated to byte |
| `\uNNNN` | Unicode | Exactly 4 hex digits, returns code point |
| `\UNNNNNNNN` | Unicode | Exactly 8 hex digits, returns code point |

Invalid Unicode code points (e.g., surrogates) fall back to U+FFFD (replacement character).

---

## Source Location Tracking

Every token carries a `Span`:

```rust
pub struct Span {
    pub start: u32,    // byte offset of first character
    pub end: u32,      // byte offset past last character
    pub file_id: u32,  // index into SourceManager's file table
}
```

Spans use `u32` offsets, which support source files up to 4 GB -- more than sufficient for any C translation unit. The `file_id` connects a span to its source file for multi-file diagnostic resolution.

The `SourceManager` (in `common/source.rs`) converts spans to human-readable `SourceLocation` values (`file:line:column`) using either direct binary search over precomputed line-offset tables, or a line-map built from GCC-style `# linenum "filename"` markers in preprocessed output.

---

## Non-UTF-8 Source Handling via PUA Encoding

C source files may contain non-UTF-8 bytes in string and character literals (e.g., EUC-JP, Latin-1, Windows-1252). Since Rust requires valid UTF-8 for `String` and `str`, the encoding layer (`common/encoding.rs`) uses a Private Use Area (PUA) encoding scheme:

**Encoding** (at file load time): Each non-UTF-8 byte `0xNN` (where NN >= 0x80) is mapped to the Unicode code point `U+E080 + (NN - 0x80)`, which falls in the PUA range `U+E080..U+E0FF`. Valid UTF-8 sequences are passed through unchanged.

**Decoding** (at lex time): Whenever the lexer reads a byte from a string or character literal, it calls `decode_pua_byte`. If the current position holds a 3-byte UTF-8 sequence corresponding to `U+E080..U+E0FF`, the function returns the original raw byte and advances by 3. Otherwise it returns the input byte as-is and advances by 1.

This scheme is transparent to the rest of the compiler: string literals end up containing exactly the bytes that appeared in the original source file, regardless of encoding. UTF-8 BOM bytes at the start of a file are silently stripped, matching GCC and Clang behavior.

---

## GNU Extensions Support

GNU extension support is controlled by the `gnu_extensions` field on `Lexer`, toggled via `set_gnu_extensions()`. It defaults to `true`.

### Dollar Signs in Identifiers

The lexer allows `$` in identifiers unconditionally (matching GCC's default `-fdollars-in-identifiers` behavior). The identifier scanner treats `$` the same as `_` or an ASCII letter:

```rust
// In lex_identifier:
while ... (input[pos] == b'_' || input[pos] == b'$' || input[pos].is_ascii_alphanumeric()) {
    pos += 1;
}
```

### GNU Keywords

When `gnu_extensions` is `true`, bare `typeof` and `asm` are recognized as keywords. When `false` (strict standard mode), only the double-underscore forms (`__typeof__`, `__asm__`) are keywords; `typeof` and `asm` become regular identifiers.

All other GCC extension keywords (`__attribute__`, `__extension__`, `__builtin_va_list`, `__int128`, `__auto_type`, `__label__`, `__seg_gs`, `__seg_fs`, etc.) are recognized regardless of the `gnu_extensions` flag, because they use the reserved `__` prefix and cannot conflict with user identifiers.

### Imaginary Suffixes

Integer and float literals accept `i`, `I`, `j`, and `J` suffixes to denote imaginary constants (GCC `_Complex` extension). The `i`/`I` suffix can appear alone (`5i`) or combined with type suffixes in either order (`1.0fi`, `1.0if`, `5Li`). The `j`/`J` suffix works standalone or after type suffixes (`1.0fj`, `5uLLj`).

### `\e` / `\E` Escape

The escape sequence `\e` (and `\E`) produces `0x1B` (ASCII ESC), a GNU extension not part of the C standard.

---

## Keyword Lookup Performance

`TokenKind::from_keyword` converts an identifier string to a keyword variant (or `None`). It uses a two-stage fast-rejection filter before entering the `match` statement:

1. **Length filter**: C and GCC keywords have lengths 2-17 or exactly 28 (`__builtin_types_compatible_p`). Any identifier outside this range is immediately rejected.
2. **First-character filter**: Only 16 ASCII characters can start a keyword (`_`, `a`, `b`, `c`, `d`, `e`, `f`, `g`, `i`, `l`, `r`, `s`, `t`, `u`, `v`, `w`). Any other starting character is immediately rejected.

These filters ensure that the vast majority of identifiers in typical C code (variable names, function names, struct fields) never reach the match statement, keeping keyword lookup fast.

---

## Design Decisions and Tradeoffs

### Byte-Level Scanning

The lexer operates on `Vec<u8>` rather than `&str`. This avoids UTF-8 validation overhead during scanning (most of the input is ASCII) and allows direct byte comparisons for all dispatch decisions. UTF-8 decoding is only performed when necessary (wide string/char literals, identifier text extraction for keyword lookup).

### Eager Type Resolution for Integer Literals

Integer literal types are resolved at lex time rather than being deferred to the parser or type checker. This front-loads the C11 promotion logic but means each integer token carries its final C type. The tradeoff is some complexity in `make_int_token` (which must handle the decimal vs. hex/octal promotion difference, the suffix matrix, and 32-bit target support), but it eliminates ambiguity for downstream consumers and avoids re-parsing the literal text later.

### Long Double as `[u8; 16]`

`long double` values are stored as both an `f64` approximation and a `[u8; 16]` IEEE 754 binary128 representation. The `f64` is convenient for constant folding where full precision is not critical. The `[u8; 16]` preserves the full 112-bit mantissa for codegen, ensuring that `long double` constants are emitted with the precision the programmer specified.

### String Literals as `String`

All string literal contents are stored as Rust `String` values, even though narrow C strings are really byte sequences. This works because (a) most C source is ASCII or valid UTF-8, (b) non-UTF-8 bytes are PUA-encoded to valid UTF-8 at load time and decoded back at lex time, and (c) the char-per-element abstraction maps cleanly to both narrow (byte) and wide (code point) interpretation.

### Pragma Tokens via Synthetic Identifiers

Rather than having the lexer parse `#pragma` directives directly (which would require context-sensitive state), pragmas are rewritten by the preprocessor into magic identifier names (`__ccc_pack_set_4`, `__ccc_visibility_push_hidden`, etc.) that the lexer recognizes by prefix. This keeps the lexer context-free and puts the pragma parsing complexity in the preprocessor, where it has access to macro expansion and conditional compilation state.

### No Separate Lexer Error Token

The lexer does not produce error tokens. Unknown or invalid characters (non-ASCII bytes outside of string literals, stray characters) are silently skipped by consuming any UTF-8 continuation bytes and re-entering `next_token`. This is a pragmatic choice: real-world preprocessed C code rarely contains genuinely invalid characters, and the parser will catch structural problems.

### Pre-Allocation Heuristic

`tokenize()` pre-allocates the token vector with capacity `input.len() / 5`, estimating roughly one token per 5 bytes of source. This is a reasonable average for C code (a short keyword like `int` is 3 bytes + 1 space = ~4 bytes per token; a semicolon is 1 byte + whitespace ~ 2 bytes; longer identifiers and string literals bring the average up). The heuristic avoids repeated reallocation for large translation units while not drastically over-allocating for small ones.
