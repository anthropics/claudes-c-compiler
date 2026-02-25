# Driver -- Design Document

The driver is the entry point and orchestrator of the entire compiler. It parses
GCC-compatible command-line arguments, selects the target architecture, and
dispatches input files through the correct compilation pipeline -- from
preprocessing through code generation, assembly, and linking. It produces ELF
executables without requiring any external toolchain (via the builtin assembler
and linker), though it can optionally delegate to GCC for assembly and linking.

---

## Overview

The driver receives command-line arguments, classifies input files, and routes
each through the appropriate pipeline stage. A single invocation can process
multiple input files (C source, assembly, object files, archives) and link them
together into an executable.

```
  Command-line arguments (GCC-compatible)
       |
       |  cli.rs: parse_cli_args()
       v
  Driver struct (all configuration fields populated)
       |
       |  pipeline.rs: run()
       v
  Dispatch by CompileMode:
       |
       +-- PreprocessOnly (-E)  -->  preprocess --> stdout / file
       |
       +-- AssemblyOnly (-S)    -->  preprocess -> lex -> parse -> sema ->
       |                             lower -> mem2reg -> optimize -> phi-elim ->
       |                             codegen --> .s file
       |
       +-- ObjectOnly (-c)      -->  [compile to asm] -> assemble --> .o file
       |
       +-- Full (default)       -->  [compile to asm] -> assemble -> link --> executable
```

Each C source file passes through the full internal pipeline
(`compile_to_assembly`). Assembly source files (`.s`/`.S`) are assembled
directly. Object files and archives (`.o`/`.a`/`.so`) pass through to the
linker.

---

## Module Layout

| File | Lines | Responsibility |
|------|-------|---------------|
| `mod.rs` | 7 | Module declarations and public re-exports (`Driver`, `CompileMode`). |
| `pipeline.rs` | ~1120 | `Driver` struct with all configuration fields, `new()` constructor, `run()` dispatcher, compilation mode handlers (`run_preprocess_only`, `run_assembly_only`, `run_object_only`, `run_full`), and the core `compile_to_assembly` pipeline. |
| `cli.rs` | ~700 | GCC-compatible CLI argument parsing: `parse_cli_args()`, query flag handling, response file expansion, and the main argument loop. |
| `external_tools.rs` | ~350 | External tool invocation and assembly source file handling: GCC `-m16` delegation, source `.s`/`.S` file assembly (both builtin and GCC-backed), assembler argument construction, linker argument construction, and dependency file generation. |
| `file_types.rs` | ~110 | Input file classification: object/archive detection by extension and magic bytes, C source detection, assembly source detection, explicit language override, and line marker stripping for `-P`. |

---

## The Driver Struct

The `Driver` struct holds every piece of configuration parsed from the command
line. All fields are `pub(super)` (visible only within the driver module) and
populated exclusively by `parse_cli_args()`. The struct is created with
`Driver::new()` which provides sensible defaults.

### Configuration Categories

**Target and output:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target` | `Target` | `X86_64` | Architecture (detected from binary name or `-m32`/`-m16`) |
| `output_path` | `String` | `"a.out"` | Output file path (from `-o`) |
| `output_path_set` | `bool` | `false` | Whether `-o` was explicitly given |
| `input_files` | `Vec<String>` | `[]` | Input source/object/archive paths |
| `mode` | `CompileMode` | `Full` | Pipeline stop point (from `-E`/`-S`/`-c`) |

**Optimization:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opt_level` | `u32` | `2` | Internal optimization level: 0=minimal, 1=basic, 2=full, 3=aggressive |
| `optimize` | `bool` | `false` | Whether user passed `-O1` or higher (defines `__OPTIMIZE__`) |
| `optimize_size` | `bool` | `false` | Whether `-Os`/`-Oz` (defines `__OPTIMIZE_SIZE__`) |

Optimization levels: `-O0` (no passes), `-O1` (single-iteration basics), `-O2` (full pipeline, 3 iterations),
`-O3` (aggressive: 5 iterations, 2% diminishing-returns threshold). `-Os`/`-Oz` map to `-O2`.
The `optimize` and `optimize_size` booleans also control predefined macros (`__OPTIMIZE__`,
`__OPTIMIZE_SIZE__`), which build systems like the Linux kernel rely on.

**Preprocessor:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `defines` | `Vec<CliDefine>` | `[]` | `-D` macro definitions |
| `include_paths` | `Vec<String>` | `[]` | `-I` include search paths |
| `quote_include_paths` | `Vec<String>` | `[]` | `-iquote` paths (searched only for `#include "file"`) |
| `isystem_include_paths` | `Vec<String>` | `[]` | `-isystem` system include paths |
| `after_include_paths` | `Vec<String>` | `[]` | `-idirafter` paths (searched last) |
| `force_includes` | `Vec<String>` | `[]` | `-include` files (processed before main source) |
| `undef_macros` | `Vec<String>` | `[]` | `-U` macro undefinitions |
| `undef_all` | `bool` | `false` | `-undef` (suppress all predefined macros) |
| `gnu_extensions` | `bool` | `true` | GNU C extensions enabled (false with `-std=c99` etc.) |
| `gnu89_inline` | `bool` | `false` | GNU89 inline semantics (`-fgnu89-inline` or `-std=gnu89`) |
| `nostdinc` | `bool` | `false` | `-nostdinc` (no default system include paths) |
| `suppress_line_markers` | `bool` | `false` | `-P` (strip `# line "file"` from `-E` output) |
| `dump_defines` | `bool` | `false` | `-dM` (dump all `#define`s instead of preprocessed text) |
| `explicit_language` | `Option<String>` | `None` | `-x` language override (e.g., `"c"`, `"assembler-with-cpp"`) |

**Code generation:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `debug_info` | `bool` | `false` | `-g` (emit DWARF `.file`/`.loc` directives) |
| `pic` | `bool` | `false` | `-fPIC`/`-fpic` (position-independent code) |
| `function_return_thunk` | `bool` | `false` | `-mfunction-return=thunk-extern` (Spectre v2 retpoline) |
| `indirect_branch_thunk` | `bool` | `false` | `-mindirect-branch=thunk-extern` (retpoline for indirect calls) |
| `patchable_function_entry` | `Option<(u32,u32)>` | `None` | `-fpatchable-function-entry=N,M` (NOP padding for ftrace) |
| `cf_protection_branch` | `bool` | `false` | `-fcf-protection=branch` (Intel CET `endbr64`) |
| `no_sse` | `bool` | `false` | `-mno-sse` (avoid all SSE/XMM instructions) |
| `enable_sse3` .. `enable_avx2` | `bool` | `false` | `-msse3` through `-mavx2` (SIMD feature flags) |
| `general_regs_only` | `bool` | `false` | `-mgeneral-regs-only` (no FP/SIMD registers; AArch64 kernel) |
| `code_model_kernel` | `bool` | `false` | `-mcmodel=kernel` (negative 2GB address space) |
| `no_jump_tables` | `bool` | `false` | `-fno-jump-tables` (compare-and-branch chains for switches) |
| `function_sections` | `bool` | `false` | `-ffunction-sections` (each function in its own section) |
| `data_sections` | `bool` | `false` | `-fdata-sections` (each global in its own section) |
| `code16gcc` | `bool` | `false` | `-m16` (16-bit real mode boot code) |
| `regparm` | `u8` | `0` | `-mregparm=N` (i686: pass first N int args in registers) |
| `omit_frame_pointer` | `bool` | `false` | `-fomit-frame-pointer` (free EBP as GP register) |
| `no_unwind_tables` | `bool` | `false` | `-fno-asynchronous-unwind-tables` (suppress `.eh_frame`) |
| `fcommon` | `bool` | `false` | `-fcommon` (COMMON linkage for tentative definitions) |

**RISC-V specific:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `riscv_abi` | `Option<String>` | `None` | `-mabi=` override (e.g., `lp64`, `lp64d`) |
| `riscv_march` | `Option<String>` | `None` | `-march=` override (e.g., `rv64imac_zicsr_zifencei`) |
| `riscv_no_relax` | `bool` | `false` | `-mno-relax` (suppress linker relaxation) |

**Linker:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `linker_paths` | `Vec<String>` | `[]` | `-L` library search paths |
| `linker_ordered_items` | `Vec<String>` | `[]` | Ordered list of `-l`, `-Wl,`, and object/archive paths |
| `static_link` | `bool` | `false` | `-static` |
| `shared_lib` | `bool` | `false` | `-shared` |
| `nostdlib` | `bool` | `false` | `-nostdlib` |
| `relocatable` | `bool` | `false` | `-r` (relocatable link; merge `.o` files) |

**Diagnostics:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `warning_config` | `WarningConfig` | all enabled | Warning enable/disable/error state (from `-W` flags) |
| `color_mode` | `ColorMode` | `Auto` | `-fdiagnostics-color={auto,always,never}` |
| `verbose` | `bool` | `false` | `-v` / `--verbose` |

**Dependency generation:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dep_file` | `Option<String>` | `None` | Dependency file path (from `-MF` or `-Wp,-MMD,path`) |
| `dep_only` | `bool` | `false` | `-M`/`-MM` (output dependency rules, no compilation) |
| `dep_target` | `Option<String>` | `None` | `-MT` (override target name in dependency rule) |

**Other:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pthread` | `bool` | `false` | `-pthread` (defines `_REENTRANT` for configure scripts like `ax_pthread.m4`) |
| `assembler_extra_args` | `Vec<String>` | `[]` | `-Wa,` assembler passthrough flags |
| `raw_args` | `Vec<String>` | `[]` | Raw CLI args for GCC `-m16` passthrough |

---

## CLI Parsing (`cli.rs`)

The CLI parser is a hand-written `while` loop with a flat `match` on each
argument. No external parser library is used. The design priorities are:

1. **GCC compatibility.** Build systems (Linux kernel, Meson, autoconf) pass a
   wide variety of GCC flags. The parser accepts all of them, silently ignoring
   unknown `-f` and `-m` flags to match GCC's behavior.

2. **Positional ordering.** Linker items (`-l`, `-Wl,`, object files) are
   collected into `linker_ordered_items` preserving their command-line order.
   This is critical for flags like `-Wl,--whole-archive` which must appear
   before the archive they affect.

3. **Early-exit query flags.** Build system probes like `-dumpmachine`,
   `-dumpversion`, `--version`, `-v` (alone), `-print-search-dirs`, and
   `-print-file-name=` are handled before input files are required. These
   print information and exit immediately.

### Target Detection

The target architecture is detected from the binary name (`argv[0]`):

| Binary name contains | Target |
|---------------------|--------|
| `arm` or `aarch64` | AArch64 |
| `riscv` | RISC-V 64 |
| `i686` or `i386` | i686 |
| anything else | x86-64 |

The `-m32` flag overrides the target to i686, and `-m16` selects i686 with the
`code16gcc` flag set (for 16-bit real mode boot code).

### Response Files

The `@file` syntax (GCC/MSVC convention) is supported. When an argument starts
with `@`, the file is read and its contents are split into tokens respecting
single/double quotes and backslash escaping. Build systems like Meson use this
when command lines exceed OS limits.

### GCC Compatibility Probes

The driver reports as GCC 14.2.0 for build system compatibility:

| Query | Response |
|-------|----------|
| `-dumpmachine` | Target triple (e.g., `x86_64-linux-gnu`) |
| `-dumpversion` | `14` |
| `--version` | `ccc (Claude's C Compiler, GCC-compatible) 14.2.0` + FSF copyright + backend mode |
| `-v` (alone) | Target and version info |
| `-print-search-dirs` | Standard library directory layout |
| `-print-file-name=NAME` | Searches standard GCC library paths; returns bundled include dir for `include` |
| `-Wl,--version` (no inputs) | With `gcc_linker`: delegates to GCC; otherwise prints `GNU ld (Claude's C Compiler built-in) 2.42` for Meson linker detection |

The `--version` output includes "Free Software Foundation" text because Meson
detects GCC by grepping for that string. It also prints the backend mode
("standalone" or a list of enabled GCC fallback features like `gcc_linker`,
`gcc_assembler`, `gcc_m16`).

### SIMD Feature Flag Implication Chain

SIMD feature flags follow GCC's implication chain where each flag implies all
lower-tier flags:

```
-mavx2 → -mavx → -msse4.2 → -msse4.1 → -mssse3 → -msse3
```

When any of these are set, the corresponding `__SSE3__`, `__AVX__`, `__AVX2__`,
etc. predefined macros are defined so projects like blosc can compile
SIMD-optimized code paths.

### Standard Version and GNU Extensions

The `-std=` flag controls GNU extensions and inline semantics:

- `gnu89`, `gnu99`, `gnu11`, `gnu17`, `gnu23`: GNU extensions enabled
- `c89`, `c99`, `c11`, `c17`, `c23`, `iso9899:*`: strict ISO C (GNU extensions
  disabled)
- `gnu89`, `c89`, `gnu90`, `c90`, `iso9899:1990`, `iso9899:199409`: GNU89
  inline semantics (`__GNUC_GNU_INLINE__`)
- All others: C99+ inline semantics (`__GNUC_STDC_INLINE__`)

When GNU extensions are disabled, bare keywords like `typeof` and `asm` are
treated as identifiers; the double-underscore forms (`__typeof__`, `__asm__`)
always work.

---

## Compilation Pipeline (`pipeline.rs`)

### Core Pipeline: `compile_to_assembly`

The heart of the driver is `compile_to_assembly()`, which transforms a C source
file into target-specific assembly text. The pipeline has 9 phases, each
producing output consumed by the next:

```
 Source text (String)
      |
 [1]  |  Preprocessor     macro expansion, #include, #ifdef, #pragma
      v
 Preprocessed text (String)
      |
 [2]  |  Lexer             tokenize with source locations
      v
 Token stream (Vec<Token>)
      |
 [3]  |  Parser            recursive descent → spanned AST
      v
 AST (TranslationUnit)
      |
 [4]  |  Sema              type-check, symbol resolution, const eval
      v
 Typed AST + TypeContext + FunctionInfo map
      |
 [5]  |  Lowerer           AST → alloca-based IR
      v
 IrModule (alloca-form)
      |
 [6]  |  mem2reg           promote allocas to SSA
      v
 IrModule (SSA-form)
      |
 [7]  |  Optimization      constant fold, DCE, GVN, inline, LICM, etc.
      v
 IrModule (optimized SSA)
      |
 [8]  |  Phi elimination   SSA phi nodes → Copy instructions
      v
 IrModule (non-SSA, copy-form)
      |
 [9]  |  Codegen           IR → target-specific assembly text
      v
 Assembly text (String)
```

Between lowering and mem2reg, the driver applies `#pragma weak` and
`#pragma redefine_extname` directives from the preprocessor, and applies
`-fcommon` by marking qualifying tentative definitions as COMMON symbols.

### Phase Timing

Set `CCC_TIME_PHASES=1` in the environment to print per-phase wall-clock timing
to stderr:

```
[TIME] preprocess: 0.012s
[TIME] lex: 0.003s (1847 tokens)
[TIME] parse: 0.005s
[TIME] sema: 0.008s
[TIME] lowering: 0.015s (42 functions)
[TIME] mem2reg: 0.002s
[TIME] opt passes: 0.025s
[TIME] phi elimination: 0.001s
[TIME] codegen: 0.018s (24576 bytes asm)
[TIME] total compile input.c: 0.089s
```

### Diagnostic Engine Threading

The diagnostic engine (`DiagnosticEngine`) owns the source manager and is
threaded through the pipeline:

1. Created after preprocessing, configured with the warning config and color
   mode. Preprocessor warnings are immediately routed through it.
2. Source manager created during lexing; transferred to the diagnostic engine
   before parsing (holds file content, line maps, and macro expansion info).
3. Passed to the parser for span-based error reporting.
4. Retrieved from the parser and passed to sema.
5. Retrieved from sema; source manager extracted for debug info emission.

Preprocessor warnings are routed through the diagnostic engine with `Cpp` kind
so they can be controlled via `-Wcpp`/`-Wno-cpp`/`-Werror=cpp`. After sema
completes, the driver checks `diagnostics.has_errors()` to catch warnings
promoted to errors by `-Werror`.

### Preprocessor Configuration

The `configure_preprocessor()` method sets up the preprocessor with all
target-specific and user-specified state:

- **Target macros**: Architecture-specific predefined macros (set via
  `set_target("x86_64")` etc.)
- **RISC-V overrides**: `-mabi=` and `-march=` values override default
  RV64GC/lp64d macros
- **Strict ANSI**: `__STRICT_ANSI__` defined when `-std=c*` (non-GNU) mode
  is selected
- **Inline semantics**: `__GNUC_GNU_INLINE__` vs `__GNUC_STDC_INLINE__`
- **Optimization macros**: `__OPTIMIZE__` for `-O1+`, `__OPTIMIZE_SIZE__` for
  `-Os`/`-Oz`
- **PIC macro**: `__PIC__`/`__pic__` when `-fPIC` is active
- **SSE/MMX macros**: `__SSE__`, `__SSE2__`, `__MMX__` (always for x86-64,
  unless `-mno-sse`)
- **Extended SIMD macros**: `__SSE3__`, `__AVX__`, etc. per feature flags
- **User defines**: `-D` macros
- **`_FORTIFY_SOURCE` suppression**: Always undefined to prevent glibc
  fortification wrappers that use unsupported GCC builtins
  (`__builtin_va_arg_pack`)
- **Include paths**: `-I`, `-iquote`, `-isystem`, `-idirafter`

Force-include files (`-include`) are processed after the preprocessor is
configured but before the main source. They are searched in the current
working directory first, then through the configured include paths.

### Source File Reading

The driver handles non-UTF-8 C source files. Raw bytes with values 0x80-0xFF
are encoded as Private Use Area code points (U+E080-U+E0FF) so the source can
be stored as a Rust `String`. The lexer decodes these back to raw bytes inside
string and character literals. This allows compiling code with Latin-1 string
literals or other non-UTF-8 content.

---

## Run Modes

### PreprocessOnly (`-E`)

Two sub-modes:

- **Assembly source** (`.S` files or `-x assembler-with-cpp`): With the
  `gcc_assembler` feature, delegates preprocessing to GCC, forwarding all
  include paths, defines, force-include files, `-nostdinc`, `-U` flags,
  `-undef`, `-x` language override, and dependency file generation flags.
  Without `gcc_assembler`, uses the built-in C preprocessor with
  `__ASSEMBLER__` defined and assembly-mode tokenization enabled.

- **C source**: Runs the built-in preprocessor. Special cases:
  - `-dM` mode: preprocesses the source then dumps all resulting `#define`
    macros instead of the preprocessed text.
  - `-M`/`-MM` mode: outputs Make-compatible dependency rules and exits.
  - `-P` flag: strips line markers (`# <line> "file"`) from output.

### AssemblyOnly (`-S`)

Compiles C source files through `compile_to_assembly()` and writes the
resulting `.s` file. With the `gcc_m16` feature enabled and `-m16`, delegates
to GCC instead (see GCC `-m16` Delegation below).

### ObjectOnly (`-c`)

Compiles and assembles. Assembly source files (`.s`/`.S`) are assembled
directly (via the builtin assembler or GCC). C source files go through
`compile_to_assembly()` then assembly. RISC-V-specific assembler flags
(`-mabi`, `-march`, `-mno-relax`, `-fno-pic`) are forwarded.

### Full (default)

Compiles, assembles, and links. Input files are classified:

- **C source** (`.c`): compiled to assembly, assembled to a temp `.o` file
- **Assembly source** (`.s`/`.S`): assembled to a temp `.o` file
- **Object/archive** (`.o`/`.a`/`.so`): passed through to the linker via
  `linker_ordered_items` (preserving command-line position)
- **Unknown extension with ELF/ar magic bytes**: detected at runtime by
  `looks_like_binary_object()` and passed through to the linker

Temporary `.o` files use RAII-based `TempFile` guards that clean up on all
exit paths (success, error, or panic). Freshly compiled objects are placed
first in the link order, followed by passthrough items in their original
command-line order.

In verbose mode (`-v`), the driver prints a synthetic link line containing
`/usr/bin/ld` and `-L` flags. This exists for CMake compatibility -- CMake's
`CMakeParseImplicitLinkInfo.cmake` extracts `-L` paths from it to populate
`CMAKE_C_IMPLICIT_LINK_DIRECTORIES`.

Linker arguments are constructed by `build_linker_args()`:
1. Order-independent flags: `-nostdlib` (if `-r`), `-shared`, `-static`,
   `-nostdlib`, `-L` paths
2. Positional items from `linker_ordered_items`: object files, `-l` flags,
   `-Wl,` pass-through flags in their original CLI order

---

## Assembly and Linking

### Assembler Selection

Assembler selection is a **compile-time** decision via Cargo features:

| Build Configuration | Behavior |
|---------------------|----------|
| Default (no features) | Use the per-architecture **builtin assembler** |
| `--features gcc_assembler` | Use GCC as the assembler |

When using the builtin assembler for a source `.S` file, the driver runs its
own C preprocessor first (with `__ASSEMBLER__` defined and assembly-mode
tokenization enabled) before passing the result to the builtin assembler. For
`.s` files, the content is read directly.

The builtin assembler path also handles the `-Wa,--version` probe: when
detected, it prints `GNU assembler (Claude's C Compiler built-in) 2.42` to satisfy the Linux
kernel's `scripts/as-version.sh`.

### Linker Selection

Linker selection is also a **compile-time** decision (handled in
`backend/common.rs` and `backend/mod.rs`):

| Build Configuration | Behavior |
|---------------------|----------|
| Default (no features) | Use the per-architecture **builtin linker** |
| `--features gcc_linker` | Use GCC as the linker |

The driver calls `Target::link()` or `Target::link_with_args()` which dispatch
to the selected linker implementation.

### GCC `-m16` Delegation

The `-m16` flag generates i386 code with `.code16gcc` prepended for 16-bit real
mode execution (used by the Linux kernel boot code at `arch/x86/boot/`). Because
the internal i686 backend currently produces code that exceeds the 32KB real-mode
size limit, `-m16` compilation of C source files is delegated entirely to GCC.

The delegation preserves the raw CLI arguments (`raw_args`) to maintain flag
ordering semantics. It strips `-o`, `-c`, and `-S` flags (added back by the
driver) and suppresses GCC warnings with `-w`.

This is explicitly a temporary hack (`// TODO: Remove this once i686 code size
optimizations bring boot code under 32KB`).

---

## File Type Classification (`file_types.rs`)

Input files are classified by extension and magic bytes:

### Object/Archive Detection (`is_object_or_archive`)

Standard extensions: `.o`, `.a`, `.so`

Non-standard extensions used by build systems:
- `.os` / `.od` (heatshrink static/dynamic variants)
- `.lo` (libtool objects)
- `.obj` (Windows-style, cross-platform projects)

Versioned shared libraries: `.so.1`, `.so.1.2.3`, etc.

Suffixed static archives: `.a.xyzzy` (skarnet.org build system). Only matched
in the filename component to avoid false positives in directory names.

### Magic Byte Detection (`looks_like_binary_object`)

Files with unrecognized extensions are probed at runtime by reading the first
8 bytes:
- `\x7fELF` → ELF object file
- `!<arch>\n` → ar archive

These are added to the link order as extra passthrough objects.

### Other Classifications

- **C source**: `.c`, `.h`, `.i`
- **Assembly source**: `.s` (pure assembly), `.S` (assembly with C preprocessor)
- **Explicit language**: `-x assembler`, `-x assembler-with-cpp` overrides
  extension-based detection

---

## Dependency File Generation

The driver generates Make-compatible dependency files (`.d`) for build system
integration. Three mechanisms:

- **`-MD`/`-MMD`**: Derive `.d` path from output path (replace extension)
- **`-MF path`**: Explicit dependency file path
- **`-Wp,-MMD,path`** / **`-Wp,-MD,path`**: Preprocessor passthrough syntax

The generated dependency rule has the format `output: input\n`. The current
implementation is minimal -- it lists only the source file as a dependency,
not included headers. This is sufficient for the Linux kernel's `fixdep`
processing.

The `-M`/`-MM` flags enter dependency-only mode: the driver outputs Make rules
to stdout (or to `-o` path) instead of compiling. `-MT` overrides the target
name in the dependency rule.

---

## Codegen Options

The `compile_to_assembly()` method constructs a `CodegenOptions` struct that
captures all CLI-driven flags affecting code generation. This struct is passed
to `Target::generate_assembly_with_opts_and_debug()`:

```rust
CodegenOptions {
    pic,                       // -fPIC or -shared
    function_return_thunk,     // -mfunction-return=thunk-extern
    indirect_branch_thunk,     // -mindirect-branch=thunk-extern
    patchable_function_entry,  // -fpatchable-function-entry=N,M
    cf_protection_branch,      // -fcf-protection=branch
    no_sse,                    // -mno-sse
    general_regs_only,         // -mgeneral-regs-only
    code_model_kernel,         // -mcmodel=kernel
    no_jump_tables,            // -fno-jump-tables
    no_relax,                  // -mno-relax (RISC-V)
    debug_info,                // -g
    function_sections,         // -ffunction-sections
    data_sections,             // -fdata-sections
    code16gcc,                 // -m16
    regparm,                   // -mregparm=N
    omit_frame_pointer,        // -fomit-frame-pointer
    emit_cfi,                  // !(-fno-asynchronous-unwind-tables)
}
```

Note that `pic` is forced on when `-shared` is specified (shared libraries
require PIC).

---

## Post-Lowering Transformations

After IR lowering and before mem2reg, the driver applies several
transformations to the alloca-form IR:

1. **`#pragma weak`**: Marks symbols as weak or creates weak aliases.
2. **`#pragma redefine_extname`**: Creates `.set` aliases to rename external
   symbols.
3. **`-fcommon`**: Marks tentative definitions (file-scope globals with no
   initializer, no `extern`, no `static`, not thread-local, with zero
   initialization) as COMMON symbols for cross-TU merging.

---

## Main Entry Point (`lib.rs` / `compiler_main`)

The shared entry point `compiler_main()` (in `lib.rs`, called by each
binary's `main()`) spawns a worker thread with a 64 MB stack (the default
~8 MB is insufficient for deeply recursive descent parsing of large generated
C files like Bison parsers). The worker thread:

1. Collects command-line arguments
2. Creates a `Driver` and calls `parse_cli_args()`
3. Exits early if a query flag was handled
4. Checks for input files
5. Calls `driver.run()`

Panics from the worker thread are caught and printed as "internal error"
messages to avoid silent failures.

The `run()` method sets two thread-local configuration values before
dispatching:
- `set_target_ptr_size()`: pointer size (4 for i686, 8 for 64-bit targets)
- `set_target_long_double_is_f128()`: `true` for AArch64/RISC-V (IEEE binary128),
  `false` for x86/i686 (x87 80-bit)

These are used throughout the type system for `sizeof`, alignment, and ABI
computations.

---

## Environment Variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `CCC_TIME_PHASES` | `pipeline.rs` | Print per-phase compilation timing to stderr |
| `CCC_ASM_DEBUG` | `external_tools.rs` | Dump preprocessed assembly to `/tmp/asm_debug_<name>.s` |
| `CCC_KEEP_ASM` | `common::temp_files`, `backend::common` | Preserve intermediate `.s` files next to output (for debugging) |

Note: Assembler/linker selection is a compile-time decision via Cargo features
(`gcc_assembler`, `gcc_linker`), not environment variables. See the top-level
[README.md](../../README.md) for details.

---

## Design Decisions

### No External Parser Library

The CLI parser is a flat `while`/`match` loop. This avoids dependencies and
gives fine-grained control over GCC compatibility edge cases (e.g., `-Wl,`
comma splitting, `-Wp,-MMD,path` parsing, `-D` with and without space). The
parser handles approximately 80 distinct flag patterns.

### Silent Ignore of Unknown Flags

Unknown `-f` and `-m` flags are silently ignored (logged only in verbose mode).
This matches GCC's behavior and is critical for build system compatibility --
the Linux kernel build passes many architecture-specific flags that only newer
GCC versions understand. Rejecting unknown flags would break kernel builds.

### Positional Linker Item Ordering

Object files, `-l` flags, and `-Wl,` flags are collected in a single ordered
list (`linker_ordered_items`) rather than separate vectors. This preserves the
command-line ordering that the linker depends on -- flags like
`-Wl,--whole-archive` must precede the archive they affect, and library
ordering determines symbol resolution priority.

### RAII Temp File Cleanup

Temporary `.o` files in the full compilation mode use `TempFile` guards
(from `common::temp_files`) that delete the file when dropped. This ensures
cleanup on all exit paths: normal completion, early `?` returns, and panics.
The `CCC_KEEP_ASM` mechanism can override this for debugging.

### Bundled Include Directory

The `-print-file-name=include` query returns the path to the compiler's bundled
include directory (containing intrinsic headers like `arm_neon.h`,
`emmintrin.h`, etc.) so that build systems pick up compatible headers instead
of the host GCC's headers which use incompatible builtins.

### _FORTIFY_SOURCE Suppression

The driver unconditionally undefines `_FORTIFY_SOURCE` because glibc's
fortification headers emit `extern always_inline` wrappers using
`__builtin_va_arg_pack()` and `__builtin_va_arg_pack_len()`, which are
GCC-specific constructs that cannot be fully supported. Without this
suppression, the wrappers produce incorrect code (infinite recursion).

---

## File Inventory

| File | Purpose |
|------|---------|
| `mod.rs` | Module declarations, re-exports `Driver` and `CompileMode`. |
| `pipeline.rs` | `Driver` struct definition (all config fields), constructor, `run()` dispatcher, mode handlers (`run_preprocess_only`, `run_assembly_only`, `run_object_only`, `run_full`), `compile_to_assembly` core pipeline, preprocessor configuration, force-include processing, source file reading with non-UTF-8 support. |
| `cli.rs` | `parse_cli_args()` entry point, target detection from binary name, query flag handling (`-dumpmachine`, `--version`, `-print-file-name=`, etc.), response file expansion with quote handling, main argument parsing loop (~80 flag patterns), `-Wl,--version` linker detection probe. |
| `external_tools.rs` | `compile_with_gcc_m16()` delegation for `-m16` boot code, `assemble_source_file()` with builtin/GCC/custom assembler dispatch, `assemble_source_file_builtin()` with built-in C preprocessor for `.S` files, `build_asm_extra_args()` for RISC-V assembler flag construction, `build_linker_args()` for ordered linker argument construction, `write_dep_file()` for Make-compatible dependency generation. |
| `file_types.rs` | `is_object_or_archive()` extension-based detection, `is_c_source()`, `looks_like_binary_object()` magic-byte probing, `is_assembly_source()`, `is_explicit_assembly()` for `-x` override, `strip_line_markers()` for `-P` output. |
