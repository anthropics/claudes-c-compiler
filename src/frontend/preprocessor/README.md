# Preprocessor

The preprocessor is a **text-to-text transformation pass** that processes C
preprocessing directives before lexing. It consumes raw source text and produces
expanded source text annotated with GCC-compatible line markers, ready for the
lexer/parser pipeline.

## Overview

The preprocessor implements the ISO C11 translation phases 2-4:

1. **Line splicing** -- backslash-newline sequences are joined.
2. **Comment replacement** -- `/* ... */` block comments are replaced with a
   single space (per C11 5.1.1.2 phase 3); `// ...` line comments are
   stripped.
3. **Directive processing and macro expansion** -- `#define`, `#include`,
   `#if`/`#ifdef`/`#elif`/`#else`/`#endif`, `#pragma`, `#error`, `#warning`,
   `#line`, and `#undef` are handled. All macro references in non-directive
   lines are expanded.

The entire pass operates on **raw text** (not tokens). All scanning is done on
byte slices (`&[u8]`) for performance, since C preprocessor tokens are ASCII.
UTF-8 multi-byte sequences inside string and character literals are copied
verbatim without interpretation.

## Architecture

```
                 ┌───────────────────────────────────────────┐
  raw source ──> │  join_continued_lines  (phase 2)         │
                 │  strip_block_comments  (phase 3)         │
                 │  ┌─────────────────────────────────────┐ │
                 │  │  per-line loop:                      │ │
                 │  │    directive?  → process_directive() │ │
                 │  │    active?    → expand macros        │ │
                 │  │    inactive?  → emit blank line      │ │
                 │  └─────────────────────────────────────┘ │
                 └──────────────┬────────────────────────────┘
                                │
                    expanded text + line markers
                                │
         ┌──────────────────────┼──────────────────────────┐
         │ side channels:       │                          │
         │  errors[]            │  warnings[]              │
         │  weak_pragmas[]      │  redefine_extname[]      │
         │  macro_expansion_info[]                         │
         └─────────────────────────────────────────────────┘
```

The output is a single `String` containing expanded C source with embedded line
markers of the form `# <line> "<file>" [flags]` (following GCC conventions:
flag 1 = entering an include, flag 2 = returning from an include). The
downstream lexer uses these markers to maintain accurate source locations.

## Public Interface

### Construction and Preprocessing

| Method | Description |
|--------|-------------|
| `Preprocessor::new()` | Create a new instance with predefined macros and builtin macro definitions already loaded. |
| `preprocess(&mut self, source: &str) -> String` | Main entry point. Expands the source and returns text with line markers. |
| `preprocess_force_include(&mut self, content, path)` | Process a `-include` file. Macros persist; pragma synthetic tokens are collected for prepending. |

### Configuration (called before `preprocess`)

| Method | Description |
|--------|-------------|
| `set_filename(name)` | Set `__FILE__`, `__BASE_FILE__`, and push onto the include stack. |
| `define_macro(name, value)` | Define an object-like macro (equivalent to `-D`). |
| `undefine_macro(name)` | Remove a macro (equivalent to `-U`). |
| `add_include_path(path)` | Append to the `-I` search list. |
| `add_quote_include_path(path)` | Append to the `-iquote` search list. |
| `add_system_include_path(path)` | Append to the `-isystem` search list. |
| `add_after_include_path(path)` | Append to the `-idirafter` search list. |
| `set_nostdinc(nostdinc)` | Skip the default system paths (`-nostdinc`). |
| `set_target(target)` | Switch architecture (`"aarch64"`, `"riscv64"`, `"i686"`, `"i386"`), updating predefined macros and include paths. |
| `set_pic(enabled)` | Define or undefine `__PIC__`/`__pic__`. |
| `set_optimize(opt, size)` | Define `__OPTIMIZE__` and `__OPTIMIZE_SIZE__`. |
| `set_gnu89_inline(gnu89)` | Toggle between GNU89 and C99 inline semantics macros. |
| `set_sse_macros(no_sse)` | Control `__SSE__`/`__SSE2__`/`__MMX__` definitions. |
| `set_extended_simd_macros(...)` | Define `__SSE3__`, `__SSSE3__`, `__SSE4_1__`, `__SSE4_2__`, `__AVX__`, `__AVX2__` based on flags. |
| `set_asm_mode(asm_mode)` | Enable assembly preprocessing mode: `$` is not treated as an identifier character (for AT&T assembly `$MACRO_NAME` expansion), and `#pragma pack` / `#pragma GCC visibility` synthetic tokens are suppressed (since these are C parser-specific and would cause assembler errors). |
| `set_riscv_abi(abi)` | Override RISC-V float ABI macros (`lp64` = soft-float, `lp64f` = single, `lp64d` = double). |
| `set_riscv_march(march)` | Override RISC-V extension macros based on `-march=` (removes F/D macros when extensions not present). |
| `set_strict_ansi(strict)` | Define or undefine `__STRICT_ANSI__` (set for `-std=cXX` non-GNU modes; checked by glibc and other headers to gate GNU extensions). |

### Output Inspection

| Method | Description |
|--------|-------------|
| `errors() -> &[PreprocessorDiagnostic]` | Collected `#error` directives and unresolved `#include` failures. |
| `warnings() -> &[PreprocessorDiagnostic]` | Collected `#warning` directives. |
| `weak_pragmas` | `Vec<(String, Option<String>)>` -- `#pragma weak` declarations. |
| `redefine_extname_pragmas` | `Vec<(String, String)>` -- `#pragma redefine_extname` declarations. |
| `take_macro_expansion_info()` | Per-line records of which macros were expanded, for "in expansion of macro" diagnostics. |
| `dump_defines() -> String` | GCC `-dM` compatible dump of all defined macros. |

---

## Macro Expansion

### Macro Definitions (`MacroDef`)

Every macro is stored in a `MacroTable` (a `FxHashMap<String, MacroDef>`)
which also carries an `asm_mode` flag. When `asm_mode` is true, `$` is not
treated as an identifier character during macro expansion, so that AT&T
assembly immediates like `$MACRO_NAME` correctly trigger expansion of
`MACRO_NAME`. Each macro definition carries these fields:

```rust
pub struct MacroDef {
    pub name: String,
    pub is_function_like: bool,
    pub params: Vec<String>,       // empty for object-like macros
    pub is_variadic: bool,         // last param is `...`
    pub has_named_variadic: bool,  // e.g., `args...` (GNU extension)
    pub body: String,              // replacement text
}
```

### Object-Like Macros

```c
#define VERSION 42
```

Expansion is straightforward: every occurrence of the identifier in
non-directive text is replaced with the body. After substitution, the result
is rescanned for further macro expansion.

### Function-Like Macros

```c
#define MAX(a, b) ((a) > (b) ? (a) : (b))
```

Expansion follows the C11 algorithm (sections 6.10.3 -- 6.10.3.4):

1. **Argument collection** -- `parse_macro_args` matches the opening `(`
   through the balanced closing `)`, splitting on commas at depth 0. String
   and character literals are skipped so that commas and parentheses inside
   them do not confuse the parser.

2. **Argument prescan** -- every argument is fully macro-expanded
   (`expand_text`) before substitution. Arguments adjacent to `#` or `##`
   use the *raw* (unexpanded) form instead.

3. **Stringification and token pasting** -- `handle_stringify_and_paste`
   processes `#param` and `a ## b` operators (see below).

4. **Parameter substitution** -- `substitute_params` replaces parameter
   names in the body with (expanded) argument text.

5. **Rescanning** -- the substituted body is re-expanded with the macro's
   own name added to the "currently expanding" set, preventing
   self-referential recursion.

### Stringification (`#`)

`#param` wraps the raw (unexpanded) argument text in double quotes, escaping
embedded `"` and `\` characters per C11 6.10.3.2.

### Token Pasting (`##`)

`a ## b` concatenates the text of the left and right operands into a single
token. When an operand is a parameter, the *raw* argument text is used (not
the prescan-expanded text). The pasted result is wrapped in internal
paste-protection markers (`0x02`/`0x03`) to prevent re-substitution during
the same parameter replacement pass; these markers are stripped before the
final output.

An anti-paste guard (`would_paste_tokens`) inserts protective spaces between
adjacent tokens that would otherwise form unintended multi-character tokens
(e.g., two `/` becoming a `//` comment, two `+` becoming `++`).

### `__VA_ARGS__` and Variadic Macros

```c
#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)
```

The variadic arguments (everything after the last named parameter) are
collected as a single comma-separated string. Both the standard `...` /
`__VA_ARGS__` form and the GNU named-variadic extension (`args...`) are
supported.

The GNU `, ## __VA_ARGS__` extension is implemented: when `__VA_ARGS__` is
empty and appears to the right of `##` preceded by a comma, the comma is
removed. This enables the common pattern:

```c
#define DBG(fmt, ...) fprintf(stderr, fmt, ## __VA_ARGS__)
DBG("hello")  // expands to: fprintf(stderr, "hello")
```

### Recursive Expansion Prevention (C11 6.10.3.4)

A "currently expanding" set (`FxHashSet<String>`) tracks which macros are
actively being expanded. When an identifier matches a macro in this set, it is
**blue-painted** -- prefixed with a sentinel byte (`0x01`) that prevents it
from being recognized as a macro during rescanning. All blue-paint markers are
stripped from the final output.

This set is allocated once per preprocessing run and cleared/reused for each
source line (`expand_line_reuse`), avoiding per-line allocation overhead.

### Multi-Line Macro Argument Accumulation

When a source line has unbalanced parentheses (more `(` than `)`), the
preprocessor accumulates subsequent lines into a `pending_line` buffer until
parentheses balance. This handles macro invocations whose arguments span
multiple lines:

```c
SOME_MACRO(
    arg1,
    arg2,
    arg3
)
```

A safety limit (`MAX_PENDING_NEWLINES = 100,000`) prevents runaway
accumulation from genuinely unbalanced source. The preprocessor also detects
when a line ends with a function-like macro name (even after expansion) and
accumulates the next line in case it starts with `(`.

Conditional directives (`#ifdef`, `#endif`, etc.) that appear inside
accumulated multi-line arguments are still processed to keep the conditional
stack correct, but are not added to the pending text.

---

## Conditional Compilation

### Directive Handling

| Directive | Effect |
|-----------|--------|
| `#if <expr>` | Push a conditional; evaluate the constant expression. |
| `#ifdef <name>` | Push; true if the macro is defined. |
| `#ifndef <name>` | Push; true if the macro is *not* defined. |
| `#elif <expr>` | Flip the current conditional; evaluate if no earlier branch was taken. |
| `#else` | Flip to the remaining branch. |
| `#endif` | Pop the conditional stack. |

The `ConditionalStack` maintains a `Vec<ConditionalState>` where each entry
tracks:
- Whether any branch in the `#if`/`#elif`/`#else` chain has been taken.
- Whether the current branch is active.
- Whether the parent context is active (for proper nesting).

Lines inside an inactive branch are replaced with empty lines to preserve
line numbering. Nested `#if` blocks within inactive branches are tracked
(push/pop) but their conditions are not evaluated.

### Expression Evaluator

`#if` and `#elif` expressions go through a multi-stage pipeline:

1. **`resolve_defined_in_expr`** -- Replace `defined(X)`, `defined X`,
   `__has_builtin(X)`, `__has_attribute(X)`, `__has_feature(X)`,
   `__has_extension(X)`, `__has_include(X)`, and `__has_include_next(X)`
   with `1` or `0`.

2. **Macro expansion** -- The resolved expression is expanded through the
   standard macro expansion pipeline.

3. **Second `resolve_defined_in_expr` pass** -- Catches cases where macro
   expansion introduced new `__has_*()` calls.

4. **`replace_remaining_idents_with_zero`** -- Per the C standard, any
   remaining identifiers (except `true`/`false`) evaluate to `0`.

5. **`evaluate_condition`** -- First expands simple object-like macros in
   the condition, then evaluates via `eval_const_expr`.

The expression evaluator (`eval_const_expr`) is a recursive-descent parser
supporting:

- Integer literals (decimal, hex `0x`, octal `0`) with
  `U`/`L`/`LL`/`ULL` suffixes.
- Character literals with escape sequences.
- `defined(X)` / `defined X`.
- The ternary operator `? :`.
- Logical: `&&`, `||`, `!`.
- Bitwise: `&`, `|`, `^`, `~`.
- Shifts: `<<`, `>>`.
- Relational: `==`, `!=`, `<`, `>`, `<=`, `>=`.
- Arithmetic: `+`, `-`, `*`, `/`, `%` (including unary `-` and `+`).
- Parenthesized sub-expressions.

Per C99 6.10.1, arithmetic uses `intmax_t` (signed) or `uintmax_t` (unsigned)
depending on whether any operand has a `U` suffix. Hex and octal literals
that exceed `INT64_MAX` are implicitly unsigned.

---

## Include File Resolution

### Search Order

The search order is GCC-compatible and depends on the include style:

**Quoted includes** (`#include "file.h"`):
1. Directory of the currently-processed file.
2. Directory of the original source file.
3. `-iquote` paths.
4. `-I` paths.
5. `-isystem` paths.
6. Default system paths (bundled headers, `/usr/include`, GCC internal headers).
7. `-idirafter` paths.

**System includes** (`#include <file.h>`):
1. `-I` paths.
2. `-isystem` paths.
3. Default system paths.
4. `-idirafter` paths.

With `-nostdinc` (`set_nostdinc(true)`), the default system paths are skipped
for both include styles and for `#include_next`, so only the explicitly
specified directories are searched. This includes the bundled headers: they
are found again only if their directory is passed explicitly.

Path resolution results are cached in an `include_resolve_cache`
(`FxHashMap`) keyed by `(include_path, is_system, current_dir)` to avoid
repeated filesystem `stat()` calls. `set_nostdinc` clears the cache when it
changes the search list.

Paths are made absolute **without resolving symlinks** (`make_absolute`
instead of `canonicalize`), matching GCC behavior where `#include "..."`
searches are relative to the symlink location, not the symlink target.

### `#include_next` (GCC Extension)

`#include_next` searches for the header starting from the *next* include
path after the one that contained the currently-processed file. This is used
by system headers to layer architecture-specific headers over generic ones.
Falls back to regular `#include` resolution if the current file cannot be
located in any search path.

### Computed Includes

When the `#include` argument does not start with `"` or `<`, it is
macro-expanded first. All spaces in the resulting path are stripped by
`normalize_include_path` to remove both anti-paste spaces (inserted by
`would_paste_tokens` to prevent `//` comment formation) and inter-token
spaces preserved from macro bodies. This normalization is only applied to
macro-expanded paths; direct include paths are left unchanged.

### Recursive Inclusion Protection

- Maximum depth of 200 (matching GCC's default).
- Files without `#pragma once` or include guards may be re-included
  (intentional, to support patterns like TCC's conditional self-inclusion).
- Only excessive nesting of the *same* file path triggers the depth limit.

### Include Guard Optimization

After preprocessing an included file, the raw source is scanned for the
classic include guard pattern:

```c
#ifndef GUARD_MACRO
#define GUARD_MACRO
  ...
#endif
```

Detection is intentionally conservative -- it returns `None` for files with
`#else`/`#elif` at the outermost level, code before `#ifndef`, or content
after `#endif`. When a guard is detected and the guard macro is still defined
on subsequent `#include` of the same file, re-processing is skipped entirely
(same optimization as GCC and Clang).

### Builtin Header Injection

When well-known standard headers are included (e.g., `stdarg.h`, `stdbool.h`,
`complex.h`), the preprocessor injects compiler-builtin macro definitions
regardless of whether the real system header is found. For example, including
`stdarg.h` defines `va_start`, `va_end`, `va_copy`, and `va_arg` as macros
expanding to `__builtin_*` forms. The `va_list`, `__builtin_va_list`, and
`__gnuc_va_list` types are handled natively by the parser/sema/lowerer
pipeline (not injected as typedef text, which would break when stdarg.h
is included from nested headers).

### Fallback Declarations

When a standard header file cannot be found on disk (no system headers
installed or cross-compiling without a sysroot), the preprocessor injects
minimal fallback declarations so that compilation can proceed. These are
only injected when the real header is absent -- when a project provides its
own headers (e.g., musl, dietlibc), the fallbacks are not used and cannot
cause conflicts.

| Header | Fallback Declarations |
|--------|-----------------------|
| `stdio.h` | `typedef struct _IO_FILE FILE;` and `extern FILE *stdin, *stdout, *stderr;` |
| `errno.h` | `extern int errno;` |
| `complex.h` | `creal`, `crealf`, `creall`, `cimag`, `cimagf`, `cimagl`, `conj`, `conjf`, `conjl`, `cabs`, `cabsf`, `carg`, `cargf` function declarations |

---

## Pragma Handling

Pragmas are dispatched by `handle_pragma` and produce either side-channel
data or synthetic tokens injected into the output for the parser.

| Pragma | Behavior |
|--------|----------|
| `#pragma once` | Marks the current file in `pragma_once_files`; subsequent `#include` of the same file returns empty. |
| `#pragma pack(N)` | Emits `__ccc_pack_set_N ;` synthetic token. Suppressed in `asm_mode`. |
| `#pragma pack()` | Emits `__ccc_pack_reset ;`. Suppressed in `asm_mode`. |
| `#pragma pack(push, N)` | Emits `__ccc_pack_push_N ;`. Suppressed in `asm_mode`. |
| `#pragma pack(push)` | Emits `__ccc_pack_push_only ;`. Suppressed in `asm_mode`. |
| `#pragma pack(pop)` | Emits `__ccc_pack_pop ;`. Suppressed in `asm_mode`. |
| `#pragma push_macro("X")` | Saves the current definition of macro `X` onto `macro_save_stack`. |
| `#pragma pop_macro("X")` | Restores the previously saved definition of macro `X`. |
| `#pragma weak sym` | Appends `(sym, None)` to `weak_pragmas`. |
| `#pragma weak sym = tgt` | Appends `(sym, Some(tgt))` to `weak_pragmas`. |
| `#pragma redefine_extname old new` | Appends `(old, new)` to `redefine_extname_pragmas`. |
| `#pragma GCC visibility push(V)` | Emits `__ccc_visibility_push_V ;` synthetic token. Suppressed in `asm_mode`. |
| `#pragma GCC visibility pop` | Emits `__ccc_visibility_pop ;`. Suppressed in `asm_mode`. |

Unrecognized pragmas (including `#pragma GCC diagnostic ...`) are silently
ignored.

---

## Predefined Macros

On construction, `define_predefined_macros` populates the macro table from a
static table of `(name, body)` pairs. Categories include:

| Category | Examples |
|----------|----------|
| **Standard C** | `__STDC__`, `__STDC_VERSION__` (201710L / C17), `__STDC_HOSTED__` |
| **Platform** | `__linux__`, `__unix__`, `__LP64__`, `__ELF__` |
| **Architecture** | `__x86_64__`, `__amd64__` (default); `__aarch64__`, `__ARM_NEON` (via `set_target`) |
| **GCC compat** | `__GNUC__` (14), `__GNUC_MINOR__` (2), `__VERSION__`, `__GNUC_STDC_INLINE__` |
| **sizeof** | `__SIZEOF_POINTER__` (8), `__SIZEOF_INT__` (4), `__SIZEOF_LONG__` (8), etc. |
| **Type limits** | `__INT_MAX__`, `__LONG_MAX__`, `__SIZE_MAX__`, etc. |
| **Type names** | `__SIZE_TYPE__` (`long unsigned int`), `__PTRDIFF_TYPE__` (`long int`), etc. |
| **Float characteristics** | `__FLT_MAX__`, `__DBL_EPSILON__`, `__LDBL_MANT_DIG__`, etc. |
| **Byte order** | `__BYTE_ORDER__` (`__ORDER_LITTLE_ENDIAN__`) |
| **Atomics** | `__GCC_ATOMIC_INT_LOCK_FREE` (2), etc. |
| **SIMD** | `__SSE__`, `__SSE2__`, `__MMX__` (x86_64 only) |
| **Date/Time** | `__DATE__`, `__TIME__` (static values) |

### Special Built-in Macros

`__LINE__` and `__COUNTER__` are **not** stored in the macro table. They use
`Cell<usize>` fields on `MacroTable` and are expanded by special-case code in
`expand_identifier`, avoiding per-line `MacroDef` allocation:

| Macro | Behavior |
|-------|----------|
| `__LINE__` | Current line number; stored in a `Cell<usize>` and updated each line via `set_line()`. Respects `#line` overrides. |
| `__COUNTER__` | Monotonically incrementing counter (`Cell<usize>`); increments on each expansion. |

`__FILE__` and `__BASE_FILE__` **are** stored in the macro table as regular
object-like macros but have dedicated update APIs to avoid full `MacroDef`
reallocation on every `#include`:

| Macro | Behavior |
|-------|----------|
| `__FILE__` | Current filename; updated in-place via `MacroTable::set_file()` (mutates the existing entry's body rather than inserting a new `MacroDef`). |
| `__BASE_FILE__` | Always the top-level input filename (set once by `set_filename()`, does not change during `#include`). |

### Preprocessor Operator Intrinsics

These are recognized by `is_defined()` for `#ifdef` and evaluated specially
in `#if` expressions by `resolve_defined_in_expr`:

- `__has_builtin(X)` -- returns `1` for supported compiler builtins.
- `__has_attribute(X)` -- returns `1` for supported GCC attributes.
- `__has_feature(X)` / `__has_extension(X)` -- always `0`.
- `__has_include(X)` / `__has_include_next(X)` -- probes the include path
  resolution machinery and returns `1` if the header exists.

---

## Builtin Macros (System Header Substitutes)

Since the preprocessor can operate without system headers, `builtin_macros.rs`
defines essential macros that would normally come from standard headers:

| Header | Macros Provided |
|--------|-----------------|
| `<limits.h>` | `CHAR_BIT`, `INT_MAX`, `LONG_MAX`, `LLONG_MIN`, etc. |
| `<stdint.h>` | `INT8_MIN` .. `UINT64_MAX`, `INTPTR_MAX`, `SIZE_MAX`, `INT64_C(x)`, `UINT32_C(x)`, etc. |
| `<stddef.h>` | `NULL`, `offsetof(type, member)` |
| `<stdbool.h>` | `bool`, `true`, `false` (injected on `#include <stdbool.h>`) |
| `<stdatomic.h>` | `__ATOMIC_RELAXED` .. `__ATOMIC_SEQ_CST`, `memory_order_*` |
| `<float.h>` | `FLT_MAX`, `DBL_EPSILON`, `LDBL_DIG`, `FLT_EVAL_METHOD`, etc. |
| `<inttypes.h>` | `PRId8` .. `PRIx64`, `PRIuMAX`, `PRIuPTR`, etc. |

Additionally, `define_type_traits_macros` provides macros that supplement
the standard headers or serve as fallbacks when system headers are unavailable:

| Category | Macros Provided |
|----------|-----------------|
| **Type widths** (GCC extension) | `__CHAR_WIDTH__` (8), `__SHRT_WIDTH__` (16), `__INT_WIDTH__` (32), `__LONG_WIDTH__` (64), `__LONG_LONG_WIDTH__` (64), `__PTRDIFF_WIDTH__`, `__SIZE_WIDTH__`, `__WCHAR_WIDTH__`, `__WINT_WIDTH__`, `__SIZEOF_WCHAR_T__` |
| **Byte order** | `__BYTE_ORDER__` (1234), `__ORDER_LITTLE_ENDIAN__` (1234), `__ORDER_BIG_ENDIAN__` (4321), `__ORDER_PDP_ENDIAN__` (3412) |
| **C11/C17 feature test** | `__STDC_UTF_16__`, `__STDC_UTF_32__`, `__STDC_NO_ATOMICS__`, `__STDC_NO_VLA__` |
| **`<stdlib.h>`** | `EXIT_SUCCESS` (0), `EXIT_FAILURE` (1), `RAND_MAX` |
| **`<stdio.h>`** | `EOF` (-1), `BUFSIZ` (8192), `SEEK_SET` (0), `SEEK_CUR` (1), `SEEK_END` (2) |
| **`<errno.h>`** | `ENOENT`, `EINTR`, `EIO`, `ENOMEM`, `EACCES`, `EEXIST`, `ENOTDIR`, `EISDIR`, `EINVAL`, `ENFILE`, `EMFILE`, `ENOSPC`, `EPIPE`, `ERANGE`, `ENOSYS` |
| **`<signal.h>`** | `SIGABRT`, `SIGFPE`, `SIGILL`, `SIGINT`, `SIGSEGV`, `SIGTERM`, `SIG_DFL`, `SIG_IGN`, `SIG_ERR` |
| **GCC version** | `__GNUC_PATCHLEVEL__` (0) |

The integer constant macros (`INT64_C`, `UINT32_C`, etc.) are function-like
macros that use `##` for suffix pasting (e.g., `INT64_C(x)` expands to
`x ## LL`).

---

## Output Contract

### Expanded Text

The returned `String` from `preprocess()` contains:

- An initial line marker: `# 1 "filename"`.
- Expanded source lines with macros resolved.
- Blank lines preserving the original line numbering.
- Line markers at include boundaries (`# 1 "header.h" 1` on entry,
  `# N "parent.c" 2` on return).
- Synthetic pragma tokens (e.g., `__ccc_pack_set_4 ;`).

### Side-Channel Data

| Field | Type | Content |
|-------|------|---------|
| `errors` | `Vec<PreprocessorDiagnostic>` | `#error` messages and unresolved `#include` paths, with file/line/col. |
| `warnings` | `Vec<PreprocessorDiagnostic>` | `#warning` messages with file/line/col. |
| `weak_pragmas` | `Vec<(String, Option<String>)>` | Weak symbol declarations and aliases. |
| `redefine_extname_pragmas` | `Vec<(String, String)>` | External name redirections. |
| `macro_expansion_info` | `Vec<MacroExpansionInfo>` | Maps output line numbers to expanded macro names for diagnostic rendering. |

`PreprocessorDiagnostic` carries `file`, `line`, `col`, and `message` fields,
enabling GCC-compatible `file:line:col: error:` / `warning:` output formatting.

---

## File Inventory

| File | Purpose |
|------|---------|
| `mod.rs` | Module declaration and re-exports. Exposes `Preprocessor` as the public type. |
| `pipeline.rs` | Core struct, `preprocess()` pipeline, directive dispatch (`process_directive`), line accumulation, and public configuration API. |
| `macro_defs.rs` | `MacroDef` and `MacroTable` types. All macro expansion logic: `expand_text`, `expand_function_macro`, `handle_stringify_and_paste`, `substitute_params`, argument parsing, blue-paint markers, paste-protection markers, and anti-paste guards. |
| `conditionals.rs` | `ConditionalStack` (push/pop state machine for `#if` nesting). `evaluate_condition` and `expand_condition_macros` for `#if` expression preprocessing. `eval_const_expr` recursive-descent expression evaluator and tokenizer. |
| `expr_eval.rs` | `resolve_defined_in_expr` (replaces `defined()`, `__has_builtin()`, `__has_attribute()`, `__has_include()`, etc. with `0`/`1`). `replace_remaining_idents_with_zero` (C standard: undefined identifiers in `#if` are `0`). |
| `includes.rs` | `#include` / `#include_next` handling. Include path resolution with GCC-compatible search order. Include guard detection (`detect_include_guard`). Path caching, symlink-preserving absolute path construction, computed include normalization, recursive inclusion depth limiting, and builtin header injection. |
| `predefined_macros.rs` | Static tables of predefined macros (standard C, platform, GCC compat, sizeof, type limits, float characteristics). `set_target` for architecture switching (`"aarch64"`, `"riscv64"`, `"i686"`, `"i386"`). `set_pic`, `set_optimize`, `set_gnu89_inline`, `set_sse_macros`, `set_extended_simd_macros`, `set_riscv_abi`, `set_riscv_march`, `set_strict_ansi` configuration methods. `bundled_include_dir` and `default_system_include_paths`. |
| `pragmas.rs` | `handle_pragma` dispatcher. Handlers for `once`, `pack`, `push_macro`/`pop_macro`, `weak`, `redefine_extname`, and `GCC visibility push`/`pop`. Synthetic token emission for pack and visibility pragmas. |
| `builtin_macros.rs` | Macro definitions substituting for `<limits.h>`, `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`, `<stdatomic.h>`, `<float.h>`, and `<inttypes.h>`. Includes both object-like and function-like (suffix-pasting) macros. |
| `text_processing.rs` | Low-level text transformations: `strip_block_comments` (with `LineMap` for line number remapping), `join_continued_lines`, `has_unbalanced_parens`, `strip_line_comment` (strips `//` comments outside string literals, returns `Cow<str>`), and `split_first_word` (splits directive keyword from arguments, treating `(` as a word boundary). |
| `utils.rs` | Shared character/byte classification (`is_ident_start`, `is_ident_cont`, byte variants), `bytes_to_str` (`&[u8]` to `&str` for ASCII identifiers), string/char literal skipping and copying helpers. |

---

## Design Decisions

### Text-Level Processing (Not Token-Based)

The preprocessor operates on raw text strings rather than a token stream. This
is a deliberate trade-off:

- **Simplicity**: no need for a tokenizer that handles both pre- and
  post-expansion token grammars.
- **Performance**: byte-slice scanning with `&[u8]` avoids allocation overhead.
  Hot paths use `bytes_to_str` (unchecked UTF-8 conversion for ASCII
  identifiers) and batch slice copies instead of per-character operations.
- **Compatibility**: the output is a flat string with line markers, which is
  the same interface GCC and Clang present to their parsers via `-E`.

### Reusable Expansion State

The `FxHashSet<String>` used for tracking "currently expanding" macros is
allocated once and reused across all lines via `expand_line_reuse`. A separate
reusable set (`directive_expanding`) is kept for directive-level expansions
(`#if`, `#elif`, `#line`, `#error`). This eliminates a measurable per-line
allocation overhead when preprocessing large headers.

### Blue-Paint Marker Byte

Self-referential macro names are marked with a `0x01` prefix byte during
expansion. This is a single-byte sentinel chosen because it cannot appear in
valid C source. The markers survive through rescanning (preventing
re-expansion) and are stripped in one final pass over the output string.
Paste-protection markers (`0x02`/`0x03`) serve a similar role for `##` results.

### Cow-Based Short-Circuit

Several expansion helpers return `Cow<str>` to avoid allocating when the
common-case fast path applies (e.g., `handle_stringify_and_paste` returns
`Cow::Borrowed(&mac.body)` when the body contains no `#` characters,
`strip_block_comments` returns `Cow::Borrowed(source)` when there are no
block comments).

### GCC Compatibility Posture

The preprocessor claims `__GNUC__ 14`, `__GNUC_MINOR__ 2` to satisfy version
checks in the Linux kernel, glibc, QEMU, and other major projects. The
predefined macro set is tuned for an LP64 Linux/ELF/x86-64 target by
default, with `set_target` providing overrides for aarch64, riscv64,
i686, and i386.

---

## Known Limitations

- **`__DATE__` and `__TIME__` are static** -- they expand to compile-time
  constants rather than the actual build timestamp.
- **`_Float128` is aliased to `long double`** -- on x86-64, this loses the
  difference between 80-bit extended precision and IEEE binary128.
- **`__has_feature` and `__has_extension` always return `0`** -- no Clang
  feature set is modeled.
- **`_Pragma("...")` is skipped** -- the C99 `_Pragma` operator is consumed
  and discarded during macro expansion rather than being executed.
- **No `#embed` support** -- the C23 `#embed` directive is not implemented.
- **No digraph/trigraph processing** -- trigraphs (`??=`, `??/`, etc.) and
  digraphs (`<:`, `:>`) are not translated in the preprocessor phase.
- **Expression evaluator does not support floating-point** -- `#if`
  expressions are evaluated using integer arithmetic only (standard-compliant,
  since preprocessor expressions are integer constant expressions).
