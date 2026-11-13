# Common Module -- Design Document

The `common` module is the shared foundation of the compiler. It contains
types, data structures, and algorithms that are used across multiple compiler
phases -- from the preprocessor and parser through semantic analysis, IR
lowering, optimization passes, and code generation. Nothing in `common`
depends on any specific phase; all dependencies flow inward.

This document describes the 12 sub-modules, explains the key design decisions,
and provides enough context to understand the system without reading code.

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [Dependency Diagram](#dependency-diagram)
3. [The Dual Type System: CType vs IrType](#the-dual-type-system-ctype-vs-irtype)
4. [Module Reference](#module-reference)
   - [types.rs -- Type Representations](#typesrs----type-representations)
   - [error.rs -- Diagnostic Infrastructure](#errorrs----diagnostic-infrastructure)
   - [source.rs -- Source Location Tracking](#sourcers----source-location-tracking)
   - [symbol_table.rs -- Scoped Symbol Table](#symbol_tablers----scoped-symbol-table)
   - [type_builder.rs -- Shared Type Resolution](#type_builderrs----shared-type-resolution)
   - [const_arith.rs -- Constant Arithmetic Primitives](#const_arithrs----constant-arithmetic-primitives)
   - [const_eval.rs -- Shared Constant Expression Evaluation](#const_evalrs----shared-constant-expression-evaluation)
   - [long_double.rs -- Long Double Precision Support](#long_doublers----long-double-precision-support)
   - [encoding.rs -- Non-UTF-8 Source File Handling](#encodingrs----non-utf-8-source-file-handling)
   - [asm_constraints.rs -- Inline Assembly Constraint Classification](#asm_constraintsrs----inline-assembly-constraint-classification)
   - [fx_hash.rs -- Fast Non-Cryptographic Hash](#fx_hashrs----fast-non-cryptographic-hash)
   - [temp_files.rs -- RAII Temporary File Handling](#temp_filesrs----raii-temporary-file-handling)
5. [Eliminating Duplication: The const_eval / const_arith Pattern](#eliminating-duplication-the-const_eval--const_arith-pattern)

---

## Module Overview

| Module              | Purpose                                              | Primary Consumers                              |
|---------------------|------------------------------------------------------|------------------------------------------------|
| `types.rs`          | CType, IrType, StructLayout, ABI classification      | Every phase                                    |
| `error.rs`          | Diagnostic engine with GCC-compatible output          | Parser, sema, driver                           |
| `source.rs`         | Span, SourceLocation, SourceManager                  | Lexer, parser, sema, backend                   |
| `symbol_table.rs`   | Scoped symbol table for name resolution              | Sema, sema const_eval                          |
| `type_builder.rs`   | Shared AST-to-CType conversion trait                 | Sema, lowering                                 |
| `const_arith.rs`    | Low-level integer/float constant arithmetic          | Sema const_eval, lowering const_eval           |
| `const_eval.rs`     | Shared constant expression evaluation logic          | Sema const_eval, lowering const_eval           |
| `long_double.rs`    | x87 80-bit and IEEE binary128 arithmetic/conversion  | const_arith, constant_fold pass, IR constants  |
| `encoding.rs`       | PUA encoding for non-UTF-8 source bytes              | Lexer                                          |
| `asm_constraints.rs`| Inline asm constraint classification                 | Inline pass, cfg_simplify, backend             |
| `fx_hash.rs`        | FxHashMap / FxHashSet type aliases                   | Every phase                                    |
| `temp_files.rs`     | RAII temp file management                            | Driver, backend (assembler invocation)          |

---

## Dependency Diagram

The following diagram shows which compiler phases use which common modules.
Arrows point from consumer to provider.

```
                          +--------------------------------------------------+
                          |                  common/                          |
                          |                                                  |
  Compiler Phases         |  Modules                                         |
  ===============         |  =======                                         |
                          |                                                  |
  Preprocessor  --------->|  fx_hash, encoding                               |
                          |                                                  |
  Lexer  ---------------->|  source (Span), encoding (decode_pua_byte)       |
                          |                                                  |
  Parser  --------------->|  source (Span), error (DiagnosticEngine),        |
                          |  fx_hash, types (AddressSpace)                   |
                          |                                                  |
  Sema  ----------------->|  types (CType, StructLayout), symbol_table,      |
                          |  type_builder, const_arith, const_eval,          |
                          |  error, fx_hash, long_double                     |
                          |                                                  |
  IR Lowering  ---------->|  types (CType, IrType, StructLayout),            |
                          |  type_builder, const_arith, const_eval,          |
                          |  source (Span), fx_hash, long_double             |
                          |                                                  |
  Optimization Passes --->|  types (IrType), fx_hash, long_double,           |
                          |  asm_constraints                                 |
                          |                                                  |
  Backend  -------------->|  types (IrType, EightbyteClass, RiscvFloatClass, |
                          |        AddressSpace), source, fx_hash,           |
                          |  asm_constraints, temp_files                     |
                          |                                                  |
  Driver  --------------->|  error (DiagnosticEngine, WarningConfig,         |
                          |        ColorMode), source (SourceManager),       |
                          |  temp_files                                      |
                          +--------------------------------------------------+
```

Key observations:

- `types.rs` and `fx_hash.rs` are truly universal -- imported by every phase.
- `const_arith.rs` and `const_eval.rs` are consumed by exactly two callers
  (sema and lowering), which is their entire reason for existing.
- `encoding.rs` is used only by the lexer.
- `temp_files.rs` is used only by the driver and backend for assembler/linker
  invocation.

---

## The Dual Type System: CType vs IrType

One of the most important design decisions in the compiler is maintaining two
separate type representations. Understanding why they exist and how they relate
is essential for working on any phase.

### CType -- The C-Level Type

`CType` represents types as the C programmer sees them. It preserves all
semantic distinctions that matter for type checking, sizeof, and ABI:

```
CType::Void, Bool,
       Char, UChar, Short, UShort, Int, UInt,
       Long, ULong, LongLong, ULongLong, Int128, UInt128,
       Float, Double, LongDouble,
       ComplexFloat, ComplexDouble, ComplexLongDouble,
       Pointer(Box<CType>, AddressSpace),
       Array(Box<CType>, Option<usize>),
       Function(Box<FunctionType>),
       Struct(RcStr), Union(RcStr), Enum(EnumType),
       Vector(Box<CType>, usize)
```

All 27 variants are shown above: 17 primitive scalar types (Void through
LongDouble), 3 complex types, 4 compound type constructors (Pointer, Array,
Function, Vector), and 3 named aggregate types (Struct, Union, Enum).

CType distinguishes `int` from `long` even when the underlying sizes match
(e.g., both are 32 bits on ILP32), because C requires them to be distinct
types for type compatibility checks and correct format specifier warnings.
CType also carries struct/union identity
through `RcStr` keys (e.g., `"struct.Foo"`) that index into a layout table.

CType is the primary type during parsing, semantic analysis, and the early
stages of IR lowering. It is used for:

- Type checking and implicit conversion rules
- sizeof / alignof evaluation
- Struct layout computation (field offsets, padding, bitfields)
- ABI classification (determining register vs. stack passing)

### IrType -- The IR-Level Type

`IrType` is a flat enumeration of machine-level types with no nesting:

```
IrType::I8, I16, I32, I64, I128,
        U8, U16, U32, U64, U128,
        F32, F64, F128,
        Ptr, Void
```

IrType is what the IR instructions, optimization passes, and code generator
work with. It collapses C-level distinctions that are irrelevant at the
machine level:

- `long` and `long long` on LP64 both become `I64` (while `int` is always `I32`)
- All pointer types become `Ptr`
- Struct and array types are decomposed into sequences of scalar loads/stores

IrType knows its own size and alignment, which vary by target. On i686, for
example, `I64` is 4-byte aligned (not 8), and `F128` (long double) is 12
bytes with 4-byte alignment. These platform differences are captured by
thread-local target configuration (`set_target_ptr_size`,
`set_target_long_double_is_f128`) that the driver sets at startup.

### Why Two Systems?

The lowering phase is the bridge. It reads CType information from the AST and
sema, computes struct layouts, performs ABI classification, and produces IR
instructions annotated with IrType. Once lowering is complete, CType is no
longer needed -- the optimization passes and backend work exclusively with
IrType.

If there were only one type system, it would either be too detailed for the
backend (carrying struct names and C-level distinctions through every
optimization pass) or too coarse for semantic analysis (losing the ability to
distinguish `int*` from `long*`).

### Target Configuration

Both type systems depend on target parameters stored in thread-locals:

- `TARGET_PTR_SIZE` (4 for i686/ILP32, 8 for x86-64/AArch64/RISC-V LP64)
- `TARGET_LONG_DOUBLE_IS_F128` (true for AArch64/RISC-V, false for x86)

The driver calls `set_target_ptr_size()` and `set_target_long_double_is_f128()`
before compilation begins. Helper functions like `target_int_ir_type()` and
`widened_op_type()` use these to return target-appropriate types, preventing
hardcoded assumptions from leaking into phase-specific code.

---

## Module Reference

### types.rs -- Type Representations

The largest module in `common/`. Contains:

**CType enum** -- All 27 C-level type variants, from `Void` through `Vector`.
Struct and union variants carry an `RcStr` (reference-counted string) key that
indexes into a `StructLayout` table. This makes cloning a struct type a cheap
refcount bump instead of a heap allocation.

CType has an extensive public method surface (20+ methods):
- Size/alignment: `size()`, `size_ctx(ctx)`, `align_ctx(ctx)`,
  `preferred_align_ctx(ctx)` -- the `_ctx` variants accept a
  `StructLayoutProvider` to resolve struct/union sizes.
- Classification: `is_integer()`, `is_signed()`, `is_unsigned()`,
  `is_floating()`, `is_complex()`, `is_arithmetic()`, `is_vector()`,
  `is_pointer_like()`, `is_function_pointer()`, `is_struct_or_union()`.
- Type arithmetic: `usual_arithmetic_conversion(lhs, rhs)` -- implements C11
  6.3.1.8 usual arithmetic conversions. `integer_rank()` -- returns the
  integer conversion rank per C11 6.3.1.1. `integer_promoted()` -- promotes
  sub-int types to `int`. `conditional_composite_type(then, else)` -- merges
  types for the ternary operator.
- Complex support: `complex_component_type()` -- returns the component type
  (e.g., `Float` for `ComplexFloat`).
- Pointer/function introspection: `get_function_type()`,
  `func_ptr_return_type(strict)`, `vector_info()`.
- Display: implements `std::fmt::Display` for user-friendly C type names in
  diagnostics (e.g., `unsigned long *`, `void (*)(int, double)`).

**IrType enum** -- Flat machine-level types (15 variants). Methods:
- `size()`, `align()` -- target-dependent size and alignment.
- `is_signed()`, `is_unsigned()`, `is_integer()`, `is_float()`,
  `is_long_double()`, `is_128bit()`.
- `to_unsigned()` -- converts signed variants to unsigned (e.g., I32 to U32).
- `truncate_i64(val)` -- truncates an i64 to the width of this type, then
  sign-extends back, implementing C truncation semantics.
- `from_ctype(ct)` -- converts a CType to the corresponding IrType.

**RcLayout type alias** -- `Rc<StructLayout>`. Cloning is a cheap reference
count increment instead of deep-copying all field names, types, and offsets.
This eliminates the most expensive cloning in the lowering phase.

**StructField struct** -- Input description of a field for layout computation:
name, CType, optional bitfield width, optional per-field alignment override,
and per-field `is_packed` flag.

**StructLayout** -- Computed layout of a struct or union: field offsets,
total size, alignment, and `is_union` / `is_transparent_union` flags.
Fields are represented as `StructFieldLayout` entries, each carrying a name,
byte offset, CType, and optional bitfield offset/width. Layout computation
follows the System V ABI rules, including:
- Natural alignment with padding
- Bitfield packing across storage units (both standard and packed modes)
- `__attribute__((packed))` and `#pragma pack(N)` support
- `_Alignas` / `__attribute__((aligned(N)))` per-field overrides
- Zero-width bitfield alignment forcing

A `StructLayoutBuilder` handles the stateful layout algorithm, tracking the
current byte offset, maximum alignment, and bitfield accumulation state.

StructLayout public methods:
- Construction: `for_struct_with_packing(fields, max_field_align, ctx)`,
  `for_union_with_packing(fields, max_field_align, ctx)`, `empty()`,
  `empty_union()`, `empty_rc()`, `empty_union_rc()`.
- ABI classification: `classify_sysv_eightbytes(ctx)` -- System V AMD64 ABI
  eightbyte classification for register passing. Returns a vector of
  `EightbyteClass` values (one per eightbyte).
  `classify_riscv_float_fields(ctx)` -- RISC-V LP64D floating-point calling
  convention classification. Returns `Option<RiscvFloatClass>`.
- Field resolution: `resolve_init_field(designator, idx, ctx)` and
  `resolve_init_field_idx(...)` -- resolve designated initializer field names,
  including drill-down into anonymous struct/union members. Returns
  `InitFieldResolution` (Direct or AnonymousMember).
  `field_offset(name, ctx)`, `field_layout(name)`,
  `field_offset_with_bitfield(...)`.
- Analysis: `has_pointer_fields(ctx)` -- recursively checks for pointer fields.

**InitFieldResolution enum** -- Result of resolving a designated initializer:
`Direct(usize)` for a field found at a given index, or
`AnonymousMember { anon_field_idx, inner_name }` when the field is inside an
anonymous struct/union member.

**EnumType** -- Stores an optional `name: Option<String>`, variant name/value
pairs, and a `is_packed` flag. The `packed_size()` method computes the minimum
integer size needed to hold all variant values (1, 2, 4, or 8 bytes), or
returns 4 for non-packed enums (with a GCC extension to use 8 bytes for
values exceeding 32-bit range).

**FunctionType** -- Return type, parameter list with optional names, and
variadic flag.

**ABI classification types**:
- `EightbyteClass` -- System V AMD64 ABI classification (NoClass, Sse, Integer)
  with a `merge()` method implementing the standard merging rules: NoClass
  yields to anything, Integer dominates, and Sse merges with itself.
- `RiscvFloatClass` -- RISC-V LP64D hardware floating-point calling convention
  classification. Variants:
  - `OneFloat { is_double }` -- a single float or double field.
  - `TwoFloats { lo_is_double, hi_is_double }` -- two float/double fields
    (two booleans indicating whether each is double).
  - `FloatAndInt { float_is_double, float_offset, int_offset, int_size }` --
    one float/double followed by one integer, with byte offsets and the
    integer's size.
  - `IntAndFloat { float_is_double, int_offset, int_size, float_offset }` --
    one integer followed by one float/double, with byte offsets and the
    integer's size.
- `AddressSpace` -- GCC named address space extension for x86 segment-relative
  addressing: `Default`, `SegGs` (`__seg_gs`), `SegFs` (`__seg_fs`).

**StructLayoutProvider trait** -- Abstraction that lets CType methods compute
sizes and alignments for struct types without depending on the lowering module.
Both `TypeContext` (in sema) and `FxHashMap<String, RcLayout>` implement this
trait.

**align_up() function** -- Aligns an offset up to the next multiple of a given
alignment. Used throughout layout computation.

**Target helpers** -- `set_target_ptr_size()`, `target_ptr_size()`,
`target_is_32bit()`, `set_target_long_double_is_f128()`,
`target_long_double_is_f128()`, `target_int_ir_type()`, `widened_op_type()`.
The `widened_op_type()` function returns the widened type for integer
arithmetic: on LP64 everything widens to I64, while on i686 most types widen
to I32 but I64/U64 stay at I64 (requiring register pairs). Float types, Void,
and I128/U128 pass through unchanged.

### error.rs -- Diagnostic Infrastructure

A complete diagnostic system that renders GCC-compatible error messages with
source snippets, caret highlighting, and ANSI color output.

**Severity** -- Three levels: `Error`, `Warning`, `Note`.

**Diagnostic** -- A single diagnostic message with:
- Severity and message text
- Optional `Span` for source location
- Optional `WarningKind` for filtering and `-Werror` promotion
- Attached `notes` vector (sub-diagnostics providing additional context)
- Optional fix-it hint text (rendered below the snippet)
- Optional explicit location string (for preprocessor-phase diagnostics
  where the SourceManager is not yet available)

Diagnostic builder methods:
- Constructors: `error(msg)`, `warning(msg)`, `warning_with_kind(msg, kind)`,
  `note(msg)`.
- Chainable modifiers: `with_span(span)`, `with_note(diag)`,
  `with_fix_hint(text)`, `with_location(file, line, col)`.

**WarningKind** -- Enumeration of warning categories matching GCC flag names:
`Undeclared`, `ImplicitFunctionDeclaration`, `Cpp`, `ReturnType`. Each
variant maps to a `-W<name>` flag (e.g., `-Wimplicit-function-declaration`).
Supports `-Wall` and `-Wextra` groupings. The `Undeclared` variant is
retained for CLI flag parsing compatibility but is no longer emitted as a
warning (undeclared variables are now hard errors).

**WarningConfig** -- Per-warning enabled/disabled and error-promotion state.
Processes CLI flags left-to-right so that later flags override earlier ones:
`-Wall -Wno-return-type` enables all warnings except `return-type`. The
`process_flag()` method handles the full set of supported flags:
- `-Werror` (global promotion of all warnings to errors)
- `-Wno-error` (global demotion, clears the `-Werror` flag)
- `-Werror=<name>` (per-warning promotion, which also implicitly enables it)
- `-Wno-error=<name>` (demotion of a specific warning back from error)
- `-Wall` (enable the standard warning set)
- `-Wextra` (enable additional warnings, superset of `-Wall`)
- `-W<name>` (enable a specific warning by name)
- `-Wno-<name>` (disable a specific warning entirely)

**ColorMode** -- Three modes matching `-fdiagnostics-color={auto,always,never}`.
The `auto` mode detects whether stderr is a terminal. Color scheme matches GCC:
bold red for errors, bold magenta for warnings, bold cyan for notes, bold green
for carets and fix-it hints, bold white for file:line:col locations.

**DiagnosticEngine** -- Central diagnostic collector. Tracks error and warning
counts, holds a `WarningConfig` and an owned `Option<SourceManager>`. Emits
diagnostics to stderr with source snippets and include-chain traces (GCC-style
"In file included from X:Y:" headers, with deduplication to avoid repeating the
same chain for consecutive errors in the same included file). The source manager
is attached after preprocessing creates it, so early-phase diagnostics use
explicit location strings instead.

DiagnosticEngine convenience methods:
- `error(msg, span)`, `warning(msg, span)`, `warning_with_kind(msg, span, kind)`
  -- shorthand methods that construct a Diagnostic and emit it in one call.
- `has_errors()`, `error_count()`, `warning_count()` -- query counts.
- `take_source_manager()` -- transfers ownership of the source manager out
  (e.g., for passing to codegen for debug info).

**Macro expansion traces** -- When a diagnostic occurs on a line produced by
macro expansion, the engine calls `render_macro_expansion_trace()` to emit
"note: in expansion of macro 'X'" messages. The `is_uninteresting_macro()`
filter suppresses noise from ubiquitous predefined macros (`NULL`, `INT_MAX`,
`__builtin_*`, etc.), showing only user-defined macros likely relevant to the
error. Traces are limited to 3 levels of nesting.

### source.rs -- Source Location Tracking

Maps byte offsets in compiler output back to original source file locations.

**Span** -- A byte-offset range (`start`, `end`, `file_id`) in source code,
all stored as `u32`. Compact (12 bytes) and cheap to copy. Constructors:
`Span::new(start, end, file_id)` and `Span::dummy()` (zero-valued placeholder
for generated code).

**SourceLocation** -- Human-readable location: filename, line number, column.
Implements `Display` as `file:line:col`.

**SourceManager** -- Resolves spans to locations. Holds a `Vec<SourceFile>`
(supporting multiple files via `add_file()`). Operates in two modes:

1. *Simple mode*: Files registered via `add_file()`. Spans are resolved
   by binary-searching a precomputed line-offset table within the file
   identified by `span.file_id`.
2. *Line-map mode*: Preprocessed output with embedded `# linenum "filename"`
   markers. The line map is built by `build_line_map()` and maps byte offsets
   in the preprocessed stream back to original source files and line numbers.

The source manager also tracks:
- **Include chains** (`IncludeOrigin`): When a `# 1 "file.h" 1` marker
  appears (flag 1 = enter-include), it records that `file.h` was included
  from the previously active file at a specific line. This enables "In file
  included from X:Y:" traces in error diagnostics.
- **Macro expansion info** (`MacroExpansionInfo`): Records which
  preprocessed-output lines involved macro expansion, enabling "in expansion
  of macro 'X'" diagnostic notes. Stored sorted by preprocessed line number
  for binary search lookup.

Public methods beyond `add_file()` and `resolve_span()`:
- `get_content(file_id)` -- returns the raw source text for a file.
- `get_source_line(span)` -- returns the full source line text containing the
  span start position (for error snippet display).
- `get_include_chain(filename)` -- walks the include origin chain from a file
  back to the main source file.
- `set_macro_expansions(expansions)` -- stores macro expansion metadata
  collected by the preprocessor.
- `get_macro_expansion_at(span)` -- looks up macro expansion info for a span
  via binary search.
- `build_line_map()` -- scans preprocessed output for line markers and builds
  the mapping tables.

Internally, filenames are deduplicated into a table indexed by `u16`, so
`LineMapEntry` structs are compact (12 bytes: `pp_offset` as u32 +
`filename_idx` as u16 + 2 bytes padding + `orig_line` as u32) even for large
translation units.

### symbol_table.rs -- Scoped Symbol Table

A simple, stack-based symbol table for lexical scoping during semantic analysis.

**Symbol** -- A declared symbol with a name, CType, and optional explicit
alignment (from `_Alignas` or `__attribute__((aligned(N)))`). The explicit
alignment is needed because `_Alignof(var)` must return the declared
alignment per C11 6.2.8p3, not the natural type alignment.

**SymbolTable** -- A stack of `Scope` objects, each containing an
`FxHashMap<String, Symbol>`. Operations:
- `push_scope()` / `pop_scope()` -- Enter/leave a lexical scope.
- `declare(symbol)` -- Insert a symbol into the current (innermost) scope.
- `lookup(name)` -- Search scopes from innermost to outermost, returning the
  first match. This implements C's name shadowing rules.

The table is initialized with a single scope (file scope) and is used by the
semantic analysis phase and its constant expression evaluator.

### type_builder.rs -- Shared Type Resolution

Prevents divergence in how AST type syntax is converted to CType between sema
and lowering.

**TypeConvertContext trait** -- A trait with one large default method
(`resolve_type_spec_to_ctype`) and five required methods. The default method
handles all shared cases:

- 22 primitive C types (Void through ComplexLongDouble)
- Pointer, Array, FunctionPointer, and BareFunction construction (BareFunction
  produces `CType::Function` directly, not wrapped in a Pointer -- this is
  used for `typeof` on function names)
- TypeofType and AutoType

Only four cases differ between sema and lowering and must be implemented by
each:
1. `resolve_typedef(name)` -- Typedef name resolution (lowering also checks
   function pointer typedefs for richer type info).
2. `resolve_struct_or_union(...)` -- Struct/union layout computation (lowering
   has caching and forward-declaration logic).
3. `resolve_enum(...)` -- Enum type resolution (sema preserves `CType::Enum`,
   lowering collapses to `CType::Int`).
4. `resolve_typeof_expr(expr)` -- typeof(expr) evaluation (sema returns
   `CType::Int` as a placeholder; lowering evaluates the full expression type).

A fifth required method, `eval_const_expr_as_usize(expr)`, handles
compile-time evaluation of array dimension expressions.

This design guarantees that the mapping from `TypeSpecifier::Int` to
`CType::Int` (and all 21 other primitives) is defined in exactly one place.

**`build_full_ctype(ctx, type_spec, derived)`** and its variant
**`build_full_ctype_with_base(ctx, base, derived)`** -- Public functions
(~140 lines of shared implementation) that build a complete CType from a
DerivedDeclarator chain. The first resolves the TypeSpecifier to a base
CType, while the second accepts an already-resolved base type (used to
avoid re-resolving anonymous struct type specs which would generate
different anonymous struct keys). Together they are the single canonical
implementation of the C "inside-out" declarator rule, used by both sema
and lowering. They handle:
- Simple pointer and array chains (`int **p`, `int arr[3][4]`)
- Function pointer declarators (`int (*fp)(int)`)
- Arrays of function pointers (`int (*fp[10])(void)`)
- Nested function pointer return types (`Page *(*fetch)(int)`)
- Bare function declarators (`int main(int, char**)`)

The function identifies the function-pointer "core" in the declarator list,
folds prefix pointer declarators into the return type, processes the core and
any trailing declarators, then applies outer array wrappers from the prefix.
Helper functions `find_function_pointer_core()` and
`convert_param_decls_to_ctypes()` support the implementation.

### const_arith.rs -- Constant Arithmetic Primitives

Low-level arithmetic functions for compile-time constant evaluation with
proper C semantics.

The functions here handle pure arithmetic: given `IrConst` operands and
width/signedness parameters, they compute the result. Callers (sema and
lowering) determine width and signedness from their own type systems before
calling these shared functions.

Key internal helpers:
- `wrap_result(v, is_32bit)` -- Truncate an i64 to 32-bit width when needed,
  preserving C truncation semantics (cast to i32, then sign-extend back).
- `unsigned_op(l, r, is_32bit, op)` -- Apply an operation in the unsigned
  domain with correct width handling.
- `bool_to_i64(b)` -- Convert boolean to 0/1.

Key evaluators:
- `eval_const_binop(op, lhs, rhs, is_32bit, is_unsigned, lhs_unsigned, rhs_unsigned)`
  -- (public) Top-level dispatch entry point for all constant binary
  evaluation. Dispatches to float, i128, or i64 paths based on operand types.
- `eval_const_binop_int(op, l, r, is_32bit, is_unsigned)` -- (module-private)
  Integer binary operations (add, sub, mul, div, mod, shifts, bitwise,
  comparisons) with wrapping, division-by-zero checking, and proper width
  truncation.
- `eval_const_binop_float(op, lhs, rhs)` -- (public) Floating-point binary
  operations on `&IrConst` parameters. Handles F32, F64, and LongDouble
  formats: for LongDouble, uses full-precision f128 software arithmetic or
  x87 80-bit software arithmetic depending on the target. For F32/F64, uses
  native Rust arithmetic. Comparison and logical operations always return
  `IrConst::I64`.
- `eval_const_binop_i128(op, lhs, rhs, ...)` -- (module-private) Native i128
  arithmetic for `__int128` operations.
- `negate_const(val)` -- (public) Unary negation with sub-int promotion.
  For LongDouble, negates by flipping the sign bit in the f128 bytes directly,
  preserving full 112-bit precision.
- `bitnot_const(val)` -- (public) Bitwise NOT with sub-int promotion to i32.
- `is_zero_expr(expr)` -- (public) Detects zero-literal expressions (0 or
  cast of 0), used for offsetof pattern detection (`&((type*)0)->member`).
- `is_null_pointer_constant(expr)` -- (public) Checks whether an expression
  is a null pointer constant per C11 6.3.2.3p3 (integer literal 0, or a
  `(void*)` cast of an integer constant expression that evaluates to zero).
  Used by sema and lowering for ternary operator type checking.
- `truncate_and_extend_bits(bits, target_width, target_signed)` -- (public)
  Raw bit truncation to a target width and optional sign-extension back to
  64 bits, used for evaluating cast chains at compile time.

### const_eval.rs -- Shared Constant Expression Evaluation

Higher-level constant evaluation logic extracted from the near-identical
implementations that previously existed in both `sema::const_eval` and
`ir::lowering::const_eval`.

The functions are parameterized by closures that abstract over the
sema-vs-lowering differences, allowing both callers to share the same
evaluation logic for:

- **Literal evaluation** (`eval_literal`) -- Converts `Expr::IntLiteral`,
  `Expr::FloatLiteral`, `Expr::CharLiteral`, and all other literal kinds to
  `IrConst` values. `IntLiteral` produces `IrConst::I32` when the value fits
  in 32 bits, otherwise `IrConst::I64`. Other integer literal kinds
  (`LongLiteral`, `LongLongLiteral`, `UIntLiteral`, `ULongLiteral`,
  `ULongLongLiteral`) always produce `IrConst::I64`. Char literals are
  sign-extended from `signed char` to `int`, matching GCC behavior. Long
  double literals produce `IrConst::long_double_with_bytes` with full-precision
  byte storage.

- **Builtin constant folding** (`eval_builtin_call`) -- Evaluates
  `__builtin_bswap`, `__builtin_clz`, `__builtin_ctz`, `__builtin_popcount`,
  `__builtin_ffs`, `__builtin_parity`, `__builtin_clrsb`, and similar
  builtins at compile time when their arguments are constants. Also handles
  `__builtin_nan`/`__builtin_inf`/`__builtin_huge_val` families for
  float/double/long-double NaN and infinity constants. The `l` suffix variants
  (e.g., `__builtin_clzl`) are target-aware, using 32-bit or 64-bit
  operations based on `target_is_32bit()`.

- **Sub-int promotion** (`promote_sub_int`) -- Handles unary operations on
  types narrower than `int` (I8/I16), applying C integer promotion rules with
  correct zero-extension for unsigned types.

- **IrConst-to-bits conversion** (`irconst_to_bits`) -- Converts an IrConst
  to raw u64 bits for bit-level operations.

- **Binary operations with type parameters** (`eval_binop_with_types`) --
  Wraps `const_arith::eval_const_binop` with C usual arithmetic conversion
  logic (C11 6.3.1.8). For shifts, only the LHS type determines the result
  type (C11 6.5.7); for other ops, the wider of both types is used.

Functions that require caller-specific state (global address resolution,
sizeof/alignof, binary operations with type inference) remain in the
respective callers. See the final section for the full explanation of how this
eliminates duplication.

### long_double.rs -- Long Double Precision Support

Full-precision support for the two `long double` formats used across
target architectures:

| Architecture     | Format              | Storage                | Exponent Bias |
|------------------|---------------------|------------------------|---------------|
| x86-64, i686     | x87 80-bit extended | 16 bytes (6 padding)   | 16383         |
| AArch64, RISC-V  | IEEE 754 binary128  | 16 bytes               | 16383         |

Both formats use 15-bit exponent fields with bias 16383. The key structural
difference is that x87 has an explicit integer bit in the mantissa (64 bits
total significand), while binary128 uses an implicit leading 1 with 112
stored mantissa bits.

**Parsing:**
- `parse_long_double_to_f128_bytes(text)` -- Parses a decimal or hex float
  string to IEEE binary128 bytes. Handles infinity, NaN, hex floats with
  binary exponents, and decimal floats via a big-integer algorithm that
  preserves all 112 bits of mantissa precision. Uses a pure-Rust `BigUint`
  implementation with no external dependencies.
- `hex_float_to_f64(text)`, `hex_float_to_f32(text)`,
  `hex_float_to_long_double(text, long_double_is_f128)` -- Convert a hex float
  constant (with `0x` prefix, no sign) to `double`, `float` or the target's
  `long double`. The digit string is read exactly (any length, any exponent)
  and rounded once to nearest-even, including subnormals and overflow to
  infinity. For x87 targets the 64-bit-significand result is stored exactly
  in the binary128 carrier. Hex float parsing in
  `parse_long_double_to_f128_bytes` uses the same rounding.

Internal preprocessing (`preparse_long_double`) strips suffixes, detects
signs, and classifies the input as hex, infinity, NaN, or decimal, shared
by both x87 and f128 parsing paths.

**Format conversion:**
- `x87_bytes_to_f128_bytes(x87)` -- Convert x87 80-bit to IEEE binary128.
- `x87_bytes_to_f64(bytes)` -- Convert x87 80-bit to f64.
- `x87_to_f64(bytes)` -- Convenience alias for `x87_bytes_to_f64`.
- `f128_bytes_to_x87_bytes(f128_bytes)` -- Convert IEEE binary128 to x87.
- `f128_bytes_to_f64(f128_bytes)` -- Convert IEEE binary128 to f64.
- `f64_to_x87_bytes_simple(val)` -- Convert f64 to x87 bytes.
- `f64_to_f128_bytes_lossless(val)` -- Convert f64 to IEEE binary128 losslessly.

**Integer-to-float conversion:**
- `i64_to_f128_bytes(val)`, `u64_to_f128_bytes(val)` -- Signed/unsigned 64-bit
  integer to binary128.
- `i128_to_f128_bytes(val)`, `u128_to_f128_bytes(val)` -- Signed/unsigned
  128-bit integer to binary128.

**Float-to-integer conversion** (all return `Option`, `None` on overflow):
- `f128_bytes_to_i64(bytes)`, `f128_bytes_to_u64(bytes)` -- Binary128 to
  64-bit integer.
- `f128_bytes_to_i128(bytes)`, `f128_bytes_to_u128(bytes)` -- Binary128 to
  128-bit integer.

Note that all float-to-integer conversions operate on f128 bytes. Code working
with x87 values should first convert to f128 via `x87_bytes_to_f128_bytes`,
then call the appropriate `f128_bytes_to_*` function.

**x87 arithmetic (pure software):**
All x87 operations are implemented in pure Rust without inline assembly,
producing bit-identical results to x87 FPU hardware:
- `x87_add(a, b)`, `x87_sub(a, b)`, `x87_mul(a, b)`, `x87_div(a, b)` --
  Full 64-bit mantissa precision arithmetic using `_soft` implementations.
- `x87_rem(a, b)` -- Software remainder with `fprem` semantics.
- `x87_neg(a)` -- Negates by flipping the sign bit (`result[9] ^= 0x80`).
- `x87_cmp(a, b)` -- Software comparison returning: -1 if a < b, 0 if
  equal, 1 if a > b, or `i32::MIN` for unordered (NaN) comparisons.

Because all implementations are pure software, no cross-compilation
fallbacks are needed -- the same code runs identically on any host.

**f128 arithmetic (pure Rust software implementation):**
For AArch64/RISC-V targets (or cross-compilation on non-x86 hosts), all
binary128 operations are implemented in pure Rust without inline assembly:
- `f128_add(a, b)`, `f128_sub(a, b)`, `f128_mul(a, b)`, `f128_div(a, b)` --
  Full 112-bit mantissa precision with correct rounding.
- `f128_rem(a, b)` -- Falls back to a lossy f64 approximation for the
  remainder computation. This is acceptable because remainder is rarely used
  in long double constant folding; the special cases (NaN, infinity, zero)
  are handled correctly.
- `f128_cmp(a, b)` -- Returns -1, 0, 1, or `i32::MIN` for unordered (NaN).

These decompose the 128-bit representation into sign, exponent, and mantissa
components using shared `f128_decompose` helpers, then perform the operation
with correct rounding.

The constant folding optimization pass uses these arithmetic functions to
evaluate long double expressions at compile time without precision loss. The
choice between x87 and f128 functions is determined by the
`target_long_double_is_f128()` flag.

### encoding.rs -- Non-UTF-8 Source File Handling

C source files may contain non-UTF-8 bytes in string and character literals
(e.g., EUC-JP, Latin-1 encoded files). Since Rust strings must be valid
UTF-8, this module provides a round-trip encoding scheme using Unicode
Private Use Area (PUA) code points.

**Encoding scheme:** Byte `0x80+n` maps to code point `U+E080+n` (PUA range
`U+E080..U+E0FF`).

**Public API:**
- `bytes_to_string(bytes)` -- Converts raw file bytes to a valid UTF-8 String.
  If the input is already valid UTF-8, it is returned as-is (zero-copy fast
  path). Otherwise, bytes are processed one at a time: valid UTF-8 multi-byte
  sequences are preserved, and invalid bytes `0x80-0xFF` are encoded as PUA
  code points. A UTF-8 BOM at the start of the file is stripped, matching
  GCC/Clang behavior.
- `decode_pua_byte(input: &[u8], pos: usize) -> (u8, usize)` -- Recovers the
  original byte from a PUA-encoded UTF-8 sequence in a byte slice. Returns
  the decoded byte and the number of input bytes consumed. Used by the lexer
  when processing string/character literals to emit the correct raw bytes
  into the compiled output.

### asm_constraints.rs -- Inline Assembly Constraint Classification

A single function shared by three consumers: the inline pass (symbol
resolution after inlining), cfg_simplify (dead block detection), and the
backend (operand emission).

**`constraint_is_immediate_only(constraint)`** -- Returns `true` for
constraints that accept only compile-time constants (`"i"`, `"I"`, `"n"`,
`"N"`, `"e"`, `"E"`, and x86-specific letters like `"K"`, `"M"`, `"G"`,
`"H"`, `"J"`, `"L"`, `"O"`). Returns `false` for multi-alternative
constraints that also accept registers or memory (`"ri"`, `"g"`, `"Ir"`,
etc.).

The function strips output/early-clobber/commutative modifiers (`=`, `+`,
`&`, `%`) before classification, and rejects named operand references
(`[name]`). It checks that at least one immediate-class letter is present,
and that no GP register-class letters (`r`, `q`, `R`, `l`), FP/SIMD
register-class letters (`x`, `v`, `Y`), specific register letters (`a`, `b`,
`c`, `d`, `S`, `D`), memory-class letters (`m`, `o`, `V`, `p`, `Q`), or
general-class letters (`g`) appear. Numeric digit references (operand reuse
constraints) also cause a `false` return.

### fx_hash.rs -- Fast Non-Cryptographic Hash

A copy of the FxHash algorithm used by the Rust compiler (rustc). Replaces
the default SipHash in `HashMap`/`HashSet` with a much faster hash for
compiler workloads where DoS resistance is unnecessary.

**Type aliases:**
- `FxHashMap<K, V>` -- `HashMap` with `FxHasher`.
- `FxHashSet<V>` -- `HashSet` with `FxHasher`.

The `FxHasher` struct implements `std::hash::Hasher`. The hash function uses
rotate-left-5-XOR followed by multiply-by-`0x517cc1b727220a95`. It is used
throughout the entire compiler for all hash maps and hash sets.

### temp_files.rs -- RAII Temporary File Handling

Provides safe temporary file management with automatic cleanup.

**`temp_dir()`** -- Returns the platform temp directory, respecting `$TMPDIR`
on Unix (falling back to `/tmp`).

**`make_temp_path(prefix, stem, extension)`** -- Generates a unique temp file
path. The filename includes the process ID and an atomic counter to avoid
collisions in parallel builds and multi-file compilations. Format:
`{prefix}_{pid}_{stem}.{counter}.{extension}`.

**`TempFile`** -- An RAII guard that deletes the file when dropped. Ensures
cleanup happens even on early returns, panics, or errors. Has a `keep` flag
for debugging that prevents deletion on drop. Used by the driver when invoking
external assemblers and by the backend for intermediate object files.

---

## Eliminating Duplication: The const_eval / const_arith Pattern

Compile-time constant expression evaluation is needed in two distinct phases:

1. **Semantic analysis (sema)** -- Evaluates constant expressions for
   `_Static_assert`, array dimensions, enum values, and `case` labels. Works
   with `CType` to determine operand width and signedness.

2. **IR lowering** -- Evaluates constant expressions for global variable
   initializers, constant folding during lowering, and static address
   computation. Works with `IrType`.

Before extraction, both phases contained near-identical implementations of
constant binary operations, literal evaluation, and builtin folding. The
`common` module eliminates this duplication with a two-layer design:

```
  sema::const_eval                  ir::lowering::const_eval
       |                                   |
       |  (determines width/signedness     |  (determines width/signedness
       |   from CType)                     |   from IrType)
       |                                   |
       +--------->  common::const_eval  <--+
                         |
                         | (pure evaluation logic,
                         |  parameterized by closures)
                         |
                    common::const_arith
                         |
                         | (pure arithmetic: wrapping,
                         |  unsigned ops, width truncation)
                         |
                    common::long_double
                         |
                         | (f128/x87 arithmetic for
                         |  long double constants)
```

**const_arith** is the bottom layer. Its functions like
`eval_const_binop_int(op, l, r, is_32bit, is_unsigned)` take explicit
width/signedness flags and return `IrConst` results. The callers compute
`is_32bit` and `is_unsigned` from their own type systems:
- Sema: `(ctype_size <= 4, ctype.is_unsigned())`
- Lowering: `(ir_type.size() <= 4, ir_type.is_unsigned())`

**const_eval** is the upper layer. Its functions like `eval_literal(expr)` and
builtin evaluation take AST expressions and return `IrConst` values directly.
Functions that need type information are parameterized by closures, so sema
passes CType-based logic and lowering passes IrType-based logic.

This structure means that:
- Adding a new integer literal kind requires changing only `eval_literal`.
- Adding a new binary operator requires changing only `eval_const_binop_int`.
- Adding a new builtin requires changing only the shared builtin evaluator.
- The two callers remain thin wrappers that supply type resolution and delegate
  to the shared layer.

Functions that inherently require phase-specific context -- global address
resolution (lowering only), sizeof/alignof evaluation (different symbol tables),
and type inference for binary operations -- remain in their respective callers.
The boundary is drawn precisely at the point where the logic diverges.
