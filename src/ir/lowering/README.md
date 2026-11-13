# IR Lowering

Translates the parsed AST into alloca-based IR. This is the largest module in the
compiler because it handles every C language construct: declarations, statements,
expressions, type conversions, ABI conventions, and global initializers. The
`mem2reg` pass later promotes allocas to SSA form.

## Module Organization

| File | Responsibility |
|---|---|
| `definitions.rs` | Shared data structures: `VarInfo`, `LocalInfo`, `GlobalInfo`, `DeclAnalysis`, `LValue`, `SwitchFrame`, `FuncSig`, `FunctionMeta`, `ParamKind`, `IrParamBuildResult` |
| `func_state.rs` | `FunctionBuildState` (per-function build state) and `FuncScopeFrame` (undo-log scope tracking for locals/statics/consts) |
| `func_lowering.rs` | Function lowering pipeline orchestration |
| `lower.rs` | `Lowerer` struct, `lower()` entry point, scope management delegation, IR emission helpers |
| `stmt.rs` | Statement lowering: thin `lower_stmt` dispatcher delegates to per-statement helpers |
| `stmt_init.rs` | Local variable init helpers: expr-init, list-init, extern/func-decl handling, `register_block_func_meta` |
| `stmt_return.rs` | Return statement: sret, two-reg struct, complex decomposition, scalar returns |
| `stmt_switch.rs` | Switch statement lowering |
| `stmt_control_flow.rs` | Control flow lowering (if/while/for/goto/labels) |
| `stmt_asm.rs` | Inline assembly lowering |
| `struct_init.rs` | Struct/union initializer list lowering (`emit_struct_init`): field dispatch loop, per-field-type handlers |
| `expr.rs` | Expression lowering dispatcher, identifier resolution, literal lowering, type inference (`infer_expr_type`, `common_type`), implicit casts |
| `expr_sizeof.rs` | sizeof/alignof expression lowering |
| `expr_ops.rs` | Binary/unary operator lowering |
| `expr_access.rs` | Cast, compound literals, sizeof, generic selection, address-of, dereference, array subscript, member access, statement expressions, va_arg |
| `expr_builtins.rs` | `__builtin_*` dispatch hub and X86 SSE/CRC intrinsics; delegates to sub-modules for bit manipulation, overflow, and FP classification |
| `expr_builtins_intrin.rs` | Integer bit-manipulation intrinsics: `__builtin_clz`, `__builtin_ctz`, `__builtin_ffs`, `__builtin_clrsb`, `__builtin_bswap`, `__builtin_popcount`, `__builtin_parity` |
| `expr_builtins_overflow.rs` | Overflow builtins (`__builtin_add_overflow`, etc.) |
| `expr_builtins_fpclass.rs` | FP classification builtins |
| `expr_atomics.rs` | `__atomic_*` and `__sync_*` operations via table-driven dispatch |
| `expr_calls.rs` | Function call lowering with ABI handling (sret, complex args, struct passing) |
| `expr_assign.rs` | Assignment/compound-assign, bitfield read-modify-write, type promotions |
| `expr_types.rs` | Expression type inference (`get_expr_type`, `get_expr_ctype`) |
| `lvalue.rs` | L-value resolution and array address computation |
| `types.rs` | `TypeSpecifier` to `IrType`/`CType`, sizeof/alignof |
| `types_ctype.rs` | Bidirectional TypeSpecifier/CType conversion, function pointer parameter handling, struct/union-to-CType, `TypeConvertContext` trait |
| `types_seed.rs` | Builtin typedef and libc math function signature seeding |
| `structs.rs` | Struct/union layout cache, field offset resolution, `mark_transparent_union` |
| `complex.rs` | `_Complex` arithmetic, assignment, conversions, and `expr_ctype` helper |
| `global_decl.rs` | Global declaration lowering |
| `global_init.rs` | Global initializer dispatch: routes to byte or compound path based on pointer content |
| `global_init_bytes.rs` | Byte-level global init serialization; shared `drill_designators` for nested designator resolution |
| `global_init_compound.rs` | Relocation-aware global init for structs/arrays containing pointer fields; `emit_expr_to_compound` |
| `global_init_compound_ptrs.rs` | Compound pointer handling: `write_expr_to_bytes_or_ptrs`, `fill_composite_or_array_with_ptrs` |
| `global_init_compound_struct.rs` | Compound struct handling: `emit_field_inits_compound`, `build_compound_from_bytes_and_ptrs` |
| `global_init_helpers.rs` | Shared utilities for the global init subsystem (designator inspection, field resolution, init classification) |
| `const_eval.rs` | Compile-time constant expression evaluation |
| `const_eval_global_addr.rs` | Global address constant evaluation |
| `const_eval_init_size.rs` | Initializer size constant evaluation |
| `pointer_analysis.rs` | Pointer type analysis: pointee sizing, struct layout resolution, `ctype_size`/`ctype_align` wrappers |
| `ref_collection.rs` | Pre-pass to collect referenced static/inline functions for dead code elimination |
| `test_util.rs` | Test-only helper (`lower_source`): runs the lexer, parser, sema and lowering on a C snippet and returns the `IrModule` |

## Architecture

### Multi-Pass Structure

The `Lowerer` processes a `TranslationUnit` through a carefully ordered sequence of
pre-passes and main passes. Sema has already populated `TypeContext` with typedefs,
enum constants, struct/union layouts, and function typedef info before lowering begins.

#### Pre-passes

These run before any function signatures are collected, establishing the type
environment that signature registration depends on:

1. **Seed builtin typedefs** (`seed_builtin_typedefs`): Registers target-dependent
   standard types (`size_t`, `ptrdiff_t`, `va_list`, `int64_t`, etc.) that sema does
   not know about. These are needed so that function parameter and return types using
   these names resolve correctly.

2. **Seed libc math function signatures** (`seed_libc_math_functions`): Registers
   known signatures for libc math functions (`sin`, `cos`, `sqrt`, etc.) so that
   calls to them get correct return types and parameter casts without requiring
   headers.

3. **Mark transparent_union**: Walks all declarations looking for
   `__attribute__((transparent_union))` and marks the corresponding `StructLayout`.
   This must happen before function signature registration so that transparent union
   parameters are excluded from `param_struct_sizes`.

4. **Apply vector_size to typedefs**: Sema does not handle `__attribute__((vector_size(N)))`,
   so typedef entries still hold the unwrapped base type. This pre-pass scans all
   typedef declarations for `vector_size` or `ext_vector_type` attributes and wraps
   the CType in `CType::Vector`. This must happen before function metadata
   registration so that return types using vector typedefs resolve correctly.

5. **Recompute vector-dependent struct layouts**: If any vector typedefs were updated
   in step 4, struct/union layouts containing those typedef fields may have incorrect
   sizes. This pass recomputes the affected layouts in-place (by key, preserving
   CType references from sema).

6. **Register struct/union layouts for SysV eightbyte classification**: Walks all
   declarations and function definitions to register struct/union types from type
   specifiers. This ensures that `classify_sysv_eightbytes` can look up struct layouts
   when computing function signature ABI details in the first pass.

#### First pass: collect function signatures

Iterates over all `ExternalDecl` entries to register function metadata:

- **Function definitions**: Calls `register_function_meta` with the function's return
  type, parameters, variadic flag, static flag, and K&R flag.
- **Function declarations**: Finds the `Function` derived declarator, computes pointer
  depth, and calls `register_function_meta`. Also collects:
  - `__asm__("label")` linker symbol redirects (stored in `asm_label_map`)
  - `has_non_inline_decl` tracking for C99 inline semantics
  - Function typedef-based declarations (e.g., `func_t add;`)
- **Constructor/destructor/alias attributes**: Collected from both function definitions
  and declarations into `module.constructors`, `module.destructors`, `module.aliases`.
- **Error/noreturn/fastcall attributes**: Collected into `error_functions`,
  `noreturn_functions`, `fastcall_functions` sets.
- **Symbol attributes**: Weak/visibility/symver attributes on extern declarations are
  collected into `module.symbol_attrs` and `module.symver_directives`.

#### Pass 2.5: collect referenced static functions

Calls `collect_referenced_static_functions` (in `ref_collection.rs`) to determine which
static and inline functions are actually referenced. The algorithm:

1. Collects direct references from root (non-skippable) function bodies and global
   initializers.
2. Builds a per-function reference map for skippable functions.
3. Uses a worklist to transitively close the reference set.

Unreferenced static/inline functions are skipped in the third pass, providing a
significant performance win for translation units that include many header-defined
static inline functions.

#### Third pass: lower everything

Iterates over all `ExternalDecl` entries and lowers them:

- **Function definitions**: Skipped if unreferenced and skippable (static, static
  inline, extern inline with gnu_inline, C99 inline-only). Otherwise, lowered via
  `lower_function`.
- **Declarations**: Lowered as global variables/externs via `lower_global_decl`.
- **Top-level asm**: Collected into `module.toplevel_asm`.

### Key Types

- **`VarInfo`** (`definitions.rs`) -- Shared type metadata (ty, elem_size, is_array,
  struct_layout, c_type, address_space, explicit_alignment, etc.) embedded in both
  `LocalInfo` and `GlobalInfo` via `Deref`. The `lookup_var_info(name)` helper returns
  `&VarInfo` for cases that only need these shared fields, eliminating the duplicated
  locals-then-globals lookup pattern.

- **`LocalInfo`** (`definitions.rs`) -- Local variable info. Wraps `VarInfo` (via Deref)
  and adds alloca, alloc_size, is_bool, static_global_name, VLA strides/size,
  asm_register, cleanup_fn, and is_const.

- **`GlobalInfo`** (`definitions.rs`) -- Global variable info. Wraps `VarInfo` (via Deref)
  and adds asm_register for global register variables.

- **`DeclAnalysis`** (`definitions.rs`) -- Computed once per declaration, bundles all
  type properties (base_ty, var_ty, alloc_size, elem_size, is_array, is_pointer,
  struct_layout, c_type, etc.). Used by both local and global lowering to avoid
  duplicating ~80 lines of type analysis. Has `resolve_global_ty(init)` to determine
  whether a global should use I8 element type (byte-array struct init) or its
  declared type.

- **`FuncSig`** (`definitions.rs`) -- Consolidated ABI-adjusted function signature.
  Contains IrType return type, return CType, param types/ctypes, sret/two-reg info,
  SysV eightbyte classification, RISC-V float classification, and bool flags.
  Use `FuncSig::for_ptr(ret, params)` to create minimal function pointer signatures.

- **`FunctionMeta`** (`definitions.rs`) -- Maps function names to `FuncSig` via `sigs`
  (for direct calls) and `ptr_sigs` (for function pointer variables).

- **`ParamKind`** (`definitions.rs`) -- Classifies how each C parameter maps to IR
  params after ABI decomposition: `Normal`, `Struct`, `ComplexDecomposed`,
  `ComplexFloatPacked`.

- **`IrParamBuildResult`** (`definitions.rs`) -- Result of `build_ir_params`: the IR
  param list, per-original-parameter `ParamKind` mapping, and sret flag.

- **`LValue`** (`definitions.rs`) -- L-value enum: `Variable(Value)` for direct allocas,
  `Address(Value, AddressSpace)` for computed addresses with optional segment override.

- **`SwitchFrame`** (`definitions.rs`) -- Switch statement context: cases, GNU case
  ranges, default label, and expression type.

- **`Lowerer`** (`lower.rs`) -- Main lowerer struct. Holds module-level state (target,
  IrModule, globals, known_functions, types, func_meta, sema data) and an
  `Option<FunctionBuildState>` that is `Some` during function lowering and `None`
  between functions. Entry point is `lower()`.

- **`FunctionBuildState`** (`func_state.rs`) -- Per-function build state: basic blocks,
  current instructions, locals, break/continue labels, switch stack, user labels,
  scope stack, sret pointer, return type. Created fresh for each function and
  discarded after.

- **`FuncScopeFrame`** (`func_state.rs`) -- Undo-log scope tracking for function-local
  state. Records additions and shadows for locals, static_local_names,
  const_local_values, var_ctypes, and vla_typedef_sizes. Also tracks VLA stack saves
  and cleanup variables. On scope exit, `pop_scope()` undoes changes in O(changes)
  rather than cloning entire HashMaps.

- **`FunctionInfo`** (`sema::analysis`, defined in `sema/analysis.rs`, re-exported from
  `sema/mod.rs`) -- C-level function signature from semantic analysis: CType return
  type, CType parameter types, variadic flag. Stored in `Lowerer::sema_functions`.
  Provides authoritative CType info that the lowerer falls back to when its own
  ABI-adjusted `FuncSig` doesn't have the information (e.g., for expression type
  inference via `get_expr_ctype`). Also pre-populates `known_functions` at
  construction time.

- **`TypeContext`** (`sema::type_context`, defined in `sema/type_context.rs`) --
  Module-level type state: struct layouts, typedefs, enum constants, function typedef
  info, CType cache, function return CTypes. Persists across functions. Uses
  `TypeScopeFrame` undo-log for block-scoped type definitions.

- **`FunctionTypedefInfo`** (`sema::type_context`) -- Function/function-pointer typedef
  metadata: return type, params, variadic flag. Used to recognize declarations like
  `func_t add;` as function declarations rather than variable declarations.

- **`TypeScopeFrame`** (`sema::type_context`) -- Undo-log for type scopes in
  `TypeContext`. Records additions and shadows for typedefs, struct layouts,
  enum constants, typedef alignments, and the CType cache. Popped on scope exit
  to restore previous state.

### Key Helpers

- `extract_fptr_typedef_info(base, derived)` (`sema::type_context`) -- Extract
  function-pointer typedef info from a base TypeSpecifier and derived declarator list.

- `shadow_local_for_scope(name)` (in `lower.rs`, delegates to `func_state.rs`) --
  Remove a local variable and record it in the current `FuncScopeFrame` for
  restoration on scope exit.

- `register_block_func_meta(name, ...)` (in `stmt_init.rs`) -- Register function
  metadata for block-scope function declarations (e.g., `extern int foo(int);`
  inside a function body).

- `lower_return_expr(e)` (in `stmt_return.rs`) -- Handles all return conventions:
  sret hidden pointer, two-register struct packing, complex decomposition, and
  scalar returns.

- `lower_local_init_expr(...)` / `lower_local_init_list(...)` (in `stmt_init.rs`) --
  Dispatch local variable initialization by type: scalar, array, struct, VLA.

- `lower_stmt` (in `stmt.rs`) -- Main statement dispatcher: routes each `Stmt`
  variant to the appropriate handler.

- `emit_struct_init` (in `struct_init.rs`) -- Lowers struct/union initializer lists
  to a sequence of field stores.

- `mark_transparent_union(decl)` (in `structs.rs`) -- Marks the `transparent_union`
  flag on the `StructLayout` for a union declaration.

- `ctype_size(ct)` (in `pointer_analysis.rs`) -- Convenience wrapper for
  `CType::size_ctx()`, passing the current struct layout context.

- `ctype_align(ct)` (in `pointer_analysis.rs`) -- Convenience wrapper for
  `CType::align_ctx()`, passing the current struct layout context.

- `get_expr_type` (in `expr_types.rs`) -- Returns the `IrType` of an expression.

- `get_expr_ctype` (in `expr_types.rs`) -- Returns the `CType` of an expression
  (with memoization cache and sema fallback).

- `expr_ctype` (in `complex.rs`) -- Returns the `CType` of an expression in
  complex-number contexts.

### Function Lowering Pipeline

`lower_function` (in `func_lowering.rs`) orchestrates function lowering via focused
sub-methods:

1. **`compute_function_return_type`** -- Resolve the IR return type, applying ABI
   overrides (sret, two-register, complex). Registers complex return CType in
   `TypeContext` for expression type resolution.

2. **`build_ir_params`** -- Build the IR parameter list with ABI decomposition:
   `_Complex double` decomposes to two FP params (real, imag), structs get
   pass-by-value handling, `_Complex float` packs into a single I64 on x86-64.
   Produces `IrParamBuildResult` with the param list and `ParamKind` mapping.

3. **`allocate_function_params`** -- 3-phase parameter alloca emission:
   - Phase 1: sret pointer parameter (hidden first arg for large struct returns)
   - Phase 2: normal and struct parameters (alloca + store)
   - Phase 3: complex decomposed/packed reconstruction (reconstruct the
     stack-allocated {real, imag} pair from individual register values)

4. **`evaluate_vla_param_side_effects`** -- Evaluate VLA parameter size expressions
   for their side effects (e.g., `void foo(int a, int b[a++])` -- the `a++` must
   be evaluated so that `a` is incremented before the function body runs).

5. **`handle_kr_float_promotion`** -- For K&R (old-style) function definitions,
   float parameters are promoted to double by the caller. This step emits
   narrowing stores (FPTrunc double to float) for parameters declared as float.

6. **`prescan_label_depths`** -- Pre-scans the function body for label definitions
   and their scope depths, needed so that `goto` can determine how many cleanup
   scopes to exit for forward references (goto before label definition).

7. **`lower_compound_stmt`** -- Lower the function body.

8. **`finalize_function`** -- Emit implicit return (void functions, or unreachable
   fallthrough), register the completed IR function in the module.

### Global Initialization Subsystem

The global init code handles C's complex initialization rules for global and static
variables. The key architectural decision is the **two-path split** based on whether
an initializer contains pointer/address fields:

```
global_init.rs: lower_struct_global_init()
    |-- struct_init_has_addr_fields() == true  --> compound path (relocation-aware)
    |-- struct_init_has_addr_fields() == false --> bytes path (flat byte array)
```

**Why two paths?** Pointer fields need `GlobalAddr` entries that become linker
relocations (e.g., `const char *s = "hello"` needs a `.quad .L.str` directive).
Non-pointer structs can be fully serialized to a `Vec<u8>` byte buffer, which is
simpler and more efficient.

#### Bytes path (`global_init_bytes.rs`)

Serializes the entire struct/array to a flat byte array. All scalar values are
written directly to a `Vec<u8>` at their computed offsets. This path handles:

- Simple structs with integer/float/enum fields
- Arrays of scalars
- Bitfield packing
- Nested structs without pointer members

**Shared data helpers** (in `global_init_bytes.rs`):
- `drill_designators(designators, start_ty)` -- Walks a chain of field/index
  designators to resolve the target type and byte offset. Used by both paths
  for nested designators like `.u.keyword` or `[3].field.subfield`.
- `fill_scalar_list_to_bytes(items, elem_ty, max_size, bytes)` -- Fills a byte
  buffer from a list of scalar initializer items. Used by the compound path for
  non-pointer fields.
- `write_const_to_bytes()`, `write_bitfield_to_bytes()` -- Low-level byte buffer
  operations.

#### Compound path (3 files)

Handles struct/array initializers that contain pointer fields, using a hybrid
approach: serializes scalar fields to bytes, collects pointer relocations
separately, then merges them into a single `GlobalInit::Compound` vector.

**`global_init_compound.rs`**:
- `emit_expr_to_compound(elements, expr, size, coerce_ty)` -- Unified
  expression-to-compound emission. When `coerce_ty` is Some, tries const eval
  with type coercion first (scalar fields, `_Bool` normalization). When None,
  tries address resolution first (pointer fields). Consolidates the
  resolve-to-GlobalInit cascade that previously appeared in 5+ places.

**`global_init_compound_struct.rs`**:
- `emit_field_inits_compound(elements, inits, field, field_size)` -- Dispatches
  all initializer items for a single struct/union field: anonymous members,
  nested designators, flat array init, and single expressions. Shared between
  union and non-union struct paths.
- `build_compound_from_bytes_and_ptrs()` -- Merges a byte buffer and sorted
  pointer relocation list into a `GlobalInit::Compound`. Used by both 1D and
  multidimensional struct array paths.

**`global_init_compound_ptrs.rs`**:
- `write_expr_to_bytes_or_ptrs()` -- Writes a scalar expression to either the
  byte buffer (for non-pointer types) or the ptr_ranges relocation list (for
  pointer/function types). Handles bitfields too. Eliminates duplicated
  is-ptr/bitfield/scalar dispatch.
- `fill_composite_or_array_with_ptrs()` -- Routes a braced init list through
  either the pointer-aware path (`fill_nested_struct_with_ptrs`) or the plain
  byte path (`fill_struct_global_bytes`) based on `StructLayout::has_pointer_fields()`.

#### Shared init helpers (`global_init_helpers.rs`)

The bytes and compound paths share many patterns for inspecting initializer items
and designator chains. These are extracted into free functions to avoid duplication:

- `first_field_designator(item)` -- Extracts field name from first designator
- `has_nested_field_designator(item)` -- Checks for multi-level `.field.subfield` patterns
- `is_anon_member_designator(name, field_name, field_ty)` -- Detects anonymous
  struct/union members
- `resolve_anonymous_member(layout, idx, name, init, layouts)` -- Resolves anonymous
  struct/union member during init: looks up sub-layout and creates synthetic
  `InitializerItem`. Used by both local struct init (`struct_init.rs`) and global
  byte init (`global_init_bytes.rs`) to avoid duplicating the sub-layout lookup +
  synthetic item construction pattern
- `has_array_field_designators(items)` -- Detects `[N].field` designated init patterns
- `expr_contains_string_literal(expr)` -- Recursive check for string literals in
  expressions
- `init_contains_string_literal(item)` -- Recursive init-level string literal check
- `init_contains_addr_expr(item, enum_constants)` -- Recursive init-level address
  expression check (takes an enum_constants map to exclude enum identifiers from
  address detection)
- `type_has_pointer_elements(ty, ctx)` -- Recursive pointer content check for CTypes
- `push_zero_bytes(elements, count)` -- Zero-fill for compound element lists
- `push_bytes_as_elements(elements, bytes)` -- Converts a byte buffer to I8 compound
  elements. Used throughout the compound path to convert byte-serialized data to
  GlobalInit elements.
- `push_string_as_elements(elements, s, size)` -- Converts a string literal to I8
  elements with null terminator and zero padding for char array fields.

## Design Decisions

- **Alloca-based**: Every local variable gets an `Alloca` instruction. The `mem2reg`
  pass promotes these to SSA registers later. This simplifies lowering because every
  variable has a stable address, making assignment, address-of, and complex control
  flow straightforward.

- **Scope stack with undo-log**: Push/pop `FuncScopeFrame` (for function-local state)
  and `TypeScopeFrame` (for type definitions) instead of cloning HashMaps at block
  boundaries. On scope exit, only the changes made within that scope are undone,
  giving O(changes) cost instead of O(total entries). Cleanup functions
  (`__attribute__((cleanup))`) are emitted in reverse declaration order at scope exit.

- **Complex ABI**: `_Complex double` decomposes to two FP registers (real, imag).
  `_Complex float` packs into a single I64 on x86-64. `_Complex long double`
  return convention varies by target: x86-64 returns in x87 registers st(0)/st(1)
  (COMPLEX_X87 class), AArch64 decomposes to two F128 registers (HFA in q0/q1),
  while RISC-V and i686 use sret hidden pointer. For parameter passing, x86-64
  passes `_Complex long double` on the stack (MEMORY class), AArch64 decomposes
  to two F128 values, and RISC-V passes by reference. The ABI logic is centralized
  in `func_lowering.rs` for parameters and `stmt_return.rs` for return values.

- **Short-circuit evaluation**: `&&` and `||` use conditional branches (jump to
  true/false blocks), not boolean AND/OR instructions. This ensures that the
  right-hand side is only evaluated when needed, as required by C semantics.

- **Two-path global init**: The global initialization subsystem splits on whether
  pointer fields are present. Pointer-free structs serialize to flat byte arrays
  (simpler, faster). Structs with pointer fields use a compound representation that
  tracks relocations separately, merging bytes and pointers at the end.

## Relationship to Other Modules

```
parser/AST + sema/types  -->  lowering  -->  ir::Module  -->  mem2reg --> passes --> codegen
```

### Data flow from sema to lowerer

The lowerer receives the following from semantic analysis:

- **`TypeContext`** (ownership transfer): typedefs, struct layouts, enum constants,
  function typedef info. The lowerer augments this with builtin typedefs and
  vector_size adjustments during pre-passes.

- **`FxHashMap<String, FunctionInfo>`** (`sema_functions`): C-level function signatures
  with CType return types and parameter types. Defined in `sema/analysis.rs`,
  re-exported from `sema/mod.rs`.

- **`ExprTypeMap`** (`sema_expr_types`): Maps `ExprId` keys to sema-inferred CTypes.
  Consulted as a fast O(1) fallback in `get_expr_ctype()` before the lowerer does
  its own (more expensive) type inference.

- **`ConstMap`** (`sema_const_values`): Pre-computed constant expression values from
  sema. Consulted as an O(1) fast path in `eval_const_expr()` before the lowerer
  falls back to its own evaluation.

The lowerer uses sema's function signatures in two ways:

1. **Pre-population**: `known_functions` is seeded from sema's function map at
   construction time, so function names are recognized immediately.

2. **Fallback CType resolution**: `register_function_meta` uses sema's CType as
   source-of-truth for return types and param CTypes instead of re-deriving from
   AST. Expression type inference (`get_expr_ctype`, `get_call_return_type`) falls
   back to `sema_functions` when the lowerer's own `func_meta.sigs` doesn't have
   the info.

Note: The lowerer's `FuncSig` contains ABI-adjusted information (IrType, sret_size,
two_reg_ret_size, param_struct_sizes, eightbyte classifications) that sema does not
compute. Sema provides C-level CTypes; the lowerer adds target-specific ABI details.
