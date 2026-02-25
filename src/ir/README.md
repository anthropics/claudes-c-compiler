# IR Subsystem Design Document

This document describes the intermediate representation (IR) used by the compiler.
It is intended to be a complete reference: a developer should be able to understand
the entire IR design without reading the source code.

---

## Table of Contents

1. [Module Structure](#module-structure)
2. [Compilation Pipeline](#compilation-pipeline)
3. [Key Identifiers](#key-identifiers)
4. [Type System](#type-system)
5. [IrModule](#irmodule)
6. [IrFunction](#irfunction)
7. [BasicBlock](#basicblock)
8. [Instruction Set](#instruction-set)
9. [Terminators](#terminators)
10. [Operations](#operations)
11. [Constants](#constants)
12. [Intrinsics](#intrinsics)
13. [Global Variables](#global-variables)
14. [Global Initializers](#global-initializers)
15. [CFG Analysis](#cfg-analysis)
16. [SSA Construction](#ssa-construction)
17. [Phi Elimination](#phi-elimination)
18. [Pass Interface](#pass-interface)
19. [IR Debugging](#ir-debugging)
20. [Key Design Decisions](#key-design-decisions)

---

## Module Structure

The IR subsystem lives at `src/ir/` and is organized into the following files
and subdirectories:

| File / Directory | Purpose |
|------------------|---------|
| `mod.rs` | Module declarations |
| `reexports.rs` | Re-export hub for all IR types (re-exports from `constants`, `ops`, `intrinsics`, `instruction`, `module`) |
| `module.rs` | `IrModule`, `IrFunction`, `IrParam`, `IrGlobal`, `GlobalInit` |
| `instruction.rs` | `Instruction`, `Terminator`, `BasicBlock`, `BlockId`, `Value`, `Operand`, `CallInfo` |
| `ops.rs` | `IrBinOp`, `IrUnaryOp`, `IrCmpOp`, `AtomicRmwOp`, `AtomicOrdering` with eval methods |
| `constants.rs` | `IrConst`, `ConstHashKey`, float encoding utilities |
| `intrinsics.rs` | `IntrinsicOp` (SIMD, crypto, math, fences) |
| `analysis.rs` | `FlatAdj`, CFG construction, dominator tree, dominance frontiers, `CfgAnalysis` |
| `lowering/` | AST-to-IR translation (alloca-based) |
| `mem2reg/` | SSA promotion (`promote.rs`) and phi elimination (`phi_eliminate.rs`) |

All public IR types are re-exported through `reexports.rs`, so consumers use
`use crate::ir::reexports::*`.

---

## Compilation Pipeline

```
  AST
   |
   |  lowering/
   v
  Alloca-based IR  (every local is a stack slot: alloca + load/store)
   |
   |  mem2reg/
   v
  SSA IR           (scalars promoted to virtual registers with phi nodes)
   |
   |  optimization passes, then phi elimination
   v
  Backend-ready IR (phi nodes lowered to copies)
```

The lowering phase translates the AST into a flat, alloca-based IR where every
local variable is represented as a stack slot (an `Alloca` instruction) accessed
through `Load` and `Store` instructions. The `mem2reg` pass then promotes
eligible allocas into SSA virtual registers, inserting `Phi` nodes at
control-flow join points. After optimization passes run on the SSA form, phi
elimination converts `Phi` instructions into `Copy` instructions, producing
backend-ready IR.

---

## Key Identifiers

| Type | Representation | Description |
|------|---------------|-------------|
| `Value(u32)` | Newtype wrapper around `u32` | Names an SSA value (virtual register) |
| `BlockId(u32)` | Newtype wrapper around `u32` | Names a basic block; formats as `.LBB{id}` in assembly output |
| `Operand` | Enum: `Value(Value)` or `Const(IrConst)` | An instruction operand that is either a virtual register reference or a compile-time constant |

`BlockId` uses the `.LBB` prefix (rather than `.L`) to avoid collisions with
the GNU assembler's internal local labels used for RISC-V `PCREL_HI20`/`PCREL_LO12`
relocation pairs.

---

## Type System

`IrType` is defined in `common/types` and represents the target-independent
scalar types used throughout the IR:

| Type | Description |
|------|-------------|
| `I8` | Signed 8-bit integer |
| `I16` | Signed 16-bit integer |
| `I32` | Signed 32-bit integer |
| `I64` | Signed 64-bit integer |
| `I128` | Signed 128-bit integer |
| `U8` | Unsigned 8-bit integer |
| `U16` | Unsigned 16-bit integer |
| `U32` | Unsigned 32-bit integer |
| `U64` | Unsigned 64-bit integer |
| `U128` | Unsigned 128-bit integer |
| `F32` | 32-bit IEEE 754 float |
| `F64` | 64-bit IEEE 754 float |
| `F128` | 128-bit long double (binary128 on ARM64/RISC-V, x87 80-bit extended on x86) |
| `Ptr` | Pointer (width determined by target: 4 bytes on ILP32, 8 bytes on LP64) |
| `Void` | Void (for function return types) |

The type system is intentionally target-independent. Backend-specific ABI
details (struct passing conventions, register classification) are carried as
metadata on `IrFunction`, `IrParam`, and `CallInfo` rather than encoded in the
type system itself. Pointers are opaque at the IR level -- the `IrType` attached
to load/store instructions determines the access width; the pointer itself
carries no pointee type information.

### AddressSpace

`AddressSpace` (defined in `common/types`) is used by `Load`, `Store`, and
`InlineAsm` to express x86 segment register overrides:

| Variant | Description |
|---------|-------------|
| `Default` | Normal memory access (no segment prefix) |
| `SegGs` | `__seg_gs`: GS segment (used for per-CPU variables in the Linux kernel) |
| `SegFs` | `__seg_fs`: FS segment (used for TLS on some platforms) |

On non-x86 targets this is always `Default`.

---

## IrModule

`IrModule` is the top-level compilation unit. It contains all functions, globals,
string literals, and linker directives for a translation unit.

| Field | Type | Description |
|-------|------|-------------|
| `functions` | `Vec<IrFunction>` | All functions (both definitions and declarations) |
| `globals` | `Vec<IrGlobal>` | All global variables |
| `string_literals` | `Vec<(String, String)>` | String literals as `(label, value)` pairs |
| `wide_string_literals` | `Vec<(String, Vec<u32>)>` | Wide string literals `L"..."` as `(label, u32 chars)` |
| `char16_string_literals` | `Vec<(String, Vec<u16>)>` | `char16_t` string literals `u"..."` as `(label, u16 chars)` |
| `constructors` | `Vec<Rc<str>>` | Functions with `__attribute__((constructor))` |
| `destructors` | `Vec<Rc<str>>` | Functions with `__attribute__((destructor))` |
| `aliases` | `Vec<(Rc<str>, Rc<str>, bool)>` | Symbol aliases: `(alias_name, target_name, is_weak)` |
| `toplevel_asm` | `Vec<String>` | Top-level `asm("...")` directives, emitted verbatim |
| `symbol_attrs` | `Vec<(Rc<str>, bool, Option<String>)>` | Symbol attribute directives: `(name, is_weak, visibility)` |
| `symver_directives` | `Vec<(Rc<str>, Rc<str>)>` | `__attribute__((symver(...)))` directives: `(symbol_name, version_string)` |

`IrModule` provides a `for_each_function` method that runs a transformation on
each defined (non-declaration) function, returning the total count of changes
made. This is used by optimization passes (see [Pass Interface](#pass-interface)).

---

## IrFunction

`IrFunction` represents a single function with its parameter list, basic blocks,
and ABI metadata. It has **23 fields**:

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `name` | `Rc<str>` | Function name (reference-counted for O(1) clone) |
| 2 | `return_type` | `IrType` | Return type |
| 3 | `params` | `Vec<IrParam>` | Parameter list |
| 4 | `blocks` | `Vec<BasicBlock>` | Basic blocks (entry block is `blocks[0]`) |
| 5 | `is_variadic` | `bool` | Whether the function accepts variadic arguments |
| 6 | `is_declaration` | `bool` | `true` if no body (extern declaration) |
| 7 | `is_static` | `bool` | `true` if declared with `static` linkage |
| 8 | `is_inline` | `bool` | `true` if declared with `inline` |
| 9 | `is_always_inline` | `bool` | `__attribute__((always_inline))` -- must always be inlined |
| 10 | `is_noinline` | `bool` | `__attribute__((noinline))` -- must never be inlined |
| 11 | `next_value_id` | `u32` | Cached upper bound on Value IDs; all Values in this function are `< next_value_id`. Set by lowering/mem2reg/phi_eliminate. A value of 0 means "not yet computed" (falls back to scanning). |
| 12 | `section` | `Option<String>` | `__attribute__((section("...")))` -- specific ELF section placement |
| 13 | `visibility` | `Option<String>` | `__attribute__((visibility(...)))` -- symbol visibility |
| 14 | `is_weak` | `bool` | `__attribute__((weak))` -- emit as weak symbol (STB_WEAK) |
| 15 | `is_used` | `bool` | `__attribute__((used))` -- prevent dead code elimination |
| 16 | `is_fastcall` | `bool` | i386 fastcall calling convention (first two int/ptr args in ecx/edx) |
| 17 | `is_naked` | `bool` | `__attribute__((naked))` -- no prologue/epilogue; body is pure asm |
| 18 | `has_inlined_calls` | `bool` | Set by the inliner when calls were inlined into this function |
| 19 | `param_alloca_values` | `Vec<Value>` | Tracks parameter allocas for dead alloca detection in the backend |
| 20 | `uses_sret` | `bool` | Returns large struct via hidden pointer; on i386 SysV ABI uses `ret $4` |
| 21 | `global_init_label_blocks` | `Vec<BlockId>` | Blocks referenced by static local variable initializers via `&&label`; must not be merged away |
| 22 | `ret_eightbyte_classes` | `Vec<EightbyteClass>` | SysV AMD64 eightbyte classification for struct return (rax/rdx vs xmm0/xmm1); empty for non-struct returns |
| 23 | `is_gnu_inline_def` | `bool` | `extern inline __attribute__((gnu_inline))` -- body used for inlining only, converted to declaration afterward to prevent infinite recursion |

### IrParam

Each function parameter carries ABI metadata beyond its type:

| Field | Type | Description |
|-------|------|-------------|
| `ty` | `IrType` | Parameter type |
| `struct_size` | `Option<usize>` | Byte size if struct/union passed by value; `None` for non-struct params |
| `struct_align` | `Option<usize>` | Struct alignment in bytes; used on RISC-V for even-aligning register pairs for 2xXLEN-aligned structs |
| `struct_eightbyte_classes` | `Vec<EightbyteClass>` | Per-eightbyte SysV ABI classification (x86-64 only); empty for non-struct params |
| `riscv_float_class` | `Option<RiscvFloatClass>` | RISC-V LP64D float field classification for FP register passing |

### max_value_id()

`IrFunction` provides a `max_value_id()` method that returns the highest `Value`
ID defined in the function. It uses the cached `next_value_id` when available,
falling back to an expensive full-IR scan otherwise. This is useful for sizing
flat lookup tables indexed by Value ID.

---

## BasicBlock

A basic block is a labeled, straight-line sequence of instructions ending in a
terminator.

| Field | Type | Description |
|-------|------|-------------|
| `label` | `BlockId` | Block identifier |
| `instructions` | `Vec<Instruction>` | Instructions (non-terminating) |
| `terminator` | `Terminator` | Block terminator (branch, return, switch, etc.) |
| `source_spans` | `Vec<Span>` | Source location spans parallel to `instructions`; used by the backend to emit `.file`/`.loc` directives when compiling with `-g`. Empty when debug info is not tracked. |

---

## Instruction Set

The `Instruction` enum has exactly **38 variants**, organized into the following
categories.

### Memory Operations (9)

| Variant | Fields | Description |
|---------|--------|-------------|
| `Alloca` | `dest, ty, size, align, volatile` | Static stack slot allocation. The `volatile` flag prevents `mem2reg` from promoting the alloca (needed for `volatile`-qualified locals that must survive `setjmp`/`longjmp`). |
| `DynAlloca` | `dest, size, align` | Dynamic stack allocation (`__builtin_alloca`); adjusts stack pointer at runtime. |
| `Load` | `dest, ptr, ty, seg_override` | Load from memory. `seg_override` is an `AddressSpace` for x86 segment register prefixes (`%gs:`, `%fs:`). |
| `Store` | `val, ptr, ty, seg_override` | Store to memory. Same `seg_override` semantics as `Load`. |
| `Memcpy` | `dest, src, size` | Block memory copy. |
| `GetElementPtr` | `dest, base, offset, ty` | Pointer arithmetic (array/struct element access). |
| `GlobalAddr` | `dest, name` | Load the address of a global symbol. |
| `StackSave` | `dest` | Capture the current stack pointer (for VLA support). |
| `StackRestore` | `ptr` | Restore a previously saved stack pointer (reclaim VLA stack space). |

### Arithmetic and Logic (6)

| Variant | Fields | Description |
|---------|--------|-------------|
| `BinOp` | `dest, op, lhs, rhs, ty` | Binary operation (see [IrBinOp](#irbinop-13-variants)). |
| `UnaryOp` | `dest, op, src, ty` | Unary operation (see [IrUnaryOp](#irunaryop-7-variants)). |
| `Cmp` | `dest, op, lhs, rhs, ty` | Comparison producing an `I8` boolean result (see [IrCmpOp](#ircmpop-10-variants)). |
| `Cast` | `dest, src, from_ty, to_ty` | Type cast/conversion. |
| `Copy` | `dest, src` | Copy a value (used by phi elimination). |
| `Select` | `dest, cond, true_val, false_val, ty` | Conditional select: `cond != 0 ? true_val : false_val`. Lowered to `cmov` on x86, `csel` on ARM, branch-based on RISC-V. Both operands are always evaluated. |

### Function Calls (2)

| Variant | Fields | Description |
|---------|--------|-------------|
| `Call` | `func, info: CallInfo` | Direct function call. |
| `CallIndirect` | `func_ptr, info: CallInfo` | Indirect function call through a pointer. |

#### CallInfo

`CallInfo` is a shared metadata struct used by both `Call` and `CallIndirect`:

| Field | Type | Description |
|-------|------|-------------|
| `dest` | `Option<Value>` | Destination value, or `None` for void calls |
| `args` | `Vec<Operand>` | Argument operands |
| `arg_types` | `Vec<IrType>` | Type of each argument (parallel to `args`) |
| `return_type` | `IrType` | Return type of the callee |
| `is_variadic` | `bool` | Whether the callee is variadic |
| `num_fixed_args` | `usize` | Number of named (non-variadic) parameters |
| `struct_arg_sizes` | `Vec<Option<usize>>` | `Some(size)` for struct args passed by value |
| `struct_arg_aligns` | `Vec<Option<usize>>` | `Some(align)` for struct args (RISC-V register pair alignment) |
| `struct_arg_classes` | `Vec<Vec<EightbyteClass>>` | Per-eightbyte SysV ABI classification for struct args |
| `struct_arg_riscv_float_classes` | `Vec<Option<RiscvFloatClass>>` | RISC-V LP64D float classification for struct args |
| `is_sret` | `bool` | Hidden pointer argument for struct returns (i386 SysV ABI) |
| `is_fastcall` | `bool` | Callee uses fastcall calling convention |
| `ret_eightbyte_classes` | `Vec<EightbyteClass>` | SysV ABI eightbyte classification for two-register struct return |

### Variadic (5)

| Variant | Key Fields | Description |
|---------|------------|-------------|
| `VaStart` | `va_list_ptr` | Initialize a `va_list`. |
| `VaEnd` | `va_list_ptr` | Clean up a `va_list` (typically a no-op). |
| `VaCopy` | `dest_ptr, src_ptr` | Copy one `va_list` to another. |
| `VaArg` | `dest, va_list_ptr, result_ty` | Extract the next scalar variadic argument. |
| `VaArgStruct` | `dest_ptr, va_list_ptr, size, eightbyte_classes` | Extract a struct/union variadic argument into a pre-allocated buffer. The `eightbyte_classes` field carries SysV AMD64 ABI eightbyte classification so the backend can determine whether struct eightbytes come from register save areas or the overflow area. |

### Atomics (5)

| Variant | Key Fields | Description |
|---------|------------|-------------|
| `AtomicRmw` | `dest, op, ptr, val, ordering, ty` | Atomic read-modify-write (see [AtomicRmwOp](#atomicrmwop-8-variants)). |
| `AtomicCmpxchg` | `dest, ptr, expected, desired, success_ordering, failure_ordering, ty, returns_bool` | Atomic compare-exchange. When `returns_bool` is `true`, `dest` receives the success/failure boolean; when `false`, `dest` receives the old value. |
| `AtomicLoad` | `dest, ptr, ordering, ty` | Atomic load. |
| `AtomicStore` | `ptr, val, ordering, ty` | Atomic store. |
| `Fence` | `ordering` | Memory fence. |

### SSA (2)

| Variant | Key Fields | Description |
|---------|------------|-------------|
| `Phi` | `dest, ty, incoming: Vec<(Operand, BlockId)>` | SSA phi node: merges values from predecessor blocks. Each entry in `incoming` is `(value, block_id)` indicating which value flows in from which predecessor. |
| `ParamRef` | `dest, param_idx, ty` | Reference to a function parameter value. Emitted in the entry block alongside param alloca + store, making parameter initial values visible for `mem2reg` promotion and constant propagation. |

### Complex Number Returns (6)

These instructions support `_Complex` return values that use two registers.

| Variant | Fields | Placement | Description |
|---------|--------|-----------|-------------|
| `GetReturnF64Second` | `dest` | After `Call`/`CallIndirect` | Read second F64 return value (xmm1 on x86-64, d1 on ARM64, fa1 on RISC-V). |
| `SetReturnF64Second` | `src` | Before `Return` | Set second F64 return value. |
| `GetReturnF32Second` | `dest` | After `Call`/`CallIndirect` | Read second F32 return value (s1 on ARM64, fa1 on RISC-V). |
| `SetReturnF32Second` | `src` | Before `Return` | Set second F32 return value. |
| `GetReturnF128Second` | `dest` | After `Call`/`CallIndirect` | Read second F128 return value (st(0) on x86-64 after first fstpt). |
| `SetReturnF128Second` | `src` | Before `Return` | Set second F128 return value (load onto x87 FPU stack as st(1)). |

### Inline Assembly and Intrinsics (3)

| Variant | Key Fields | Description |
|---------|------------|-------------|
| `InlineAsm` | `template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols, seg_overrides` | Inline assembly statement. `outputs` is `Vec<(constraint, Value, Option<name>)>`. `inputs` is `Vec<(constraint, Operand, Option<name>)>`. `goto_labels` contains `(C_label, BlockId)` pairs for `asm goto`. `input_symbols` provides symbol names for `"i"` constraints (used by `%P`/`%a` modifiers). `seg_overrides` carries per-operand `AddressSpace` overrides. Goto labels also generate implicit CFG edges in `build_cfg`. |
| `Intrinsic` | `op, dest, dest_ptr, args` | Target-independent intrinsic operation (see [IntrinsicOp](#intrinsics)). `dest` is `Option<Value>` for the result. `dest_ptr` is `Option<Value>` for store-type intrinsics. |
| `LabelAddr` | `dest, label` | Get the address of a label (GCC computed goto extension: `&&label`). |

### Canonical Value Visitor

`Instruction` provides a `for_each_used_value(&self, f: impl FnMut(u32))` method
that calls `f` for every `Value` ID used as an operand. This is the canonical
value visitor -- all passes that need to enumerate instruction operands use this
method to avoid duplicating the 38-arm match block. A convenience `used_values()`
method collects the results into a `Vec<u32>`.

`Terminator` has an analogous `for_each_used_value` and `used_values` pair.

The `dest()` method returns the `Option<Value>` defined by the instruction, and
`result_type()` returns the `Option<IrType>` of the produced value.

---

## Terminators

The `Terminator` enum defines the **6 block-ending** control flow operations:

| Variant | Fields | Description |
|---------|--------|-------------|
| `Return` | `Option<Operand>` | Return from function (with optional return value). |
| `Branch` | `BlockId` | Unconditional branch. |
| `CondBranch` | `cond, true_label, false_label` | Conditional branch. |
| `IndirectBranch` | `target, possible_targets` | Indirect branch (computed goto: `goto *addr`). `possible_targets` lists all labels that could be reached. |
| `Switch` | `val, cases: Vec<(i64, BlockId)>, default, ty` | Switch dispatch via jump table. The backend emits a jump table instead of a chain of compare-and-branch instructions. `cases` maps case values to target block IDs. |
| `Unreachable` | (none) | Unreachable code (e.g., after a `noreturn` call). |

---

## Operations

### IrBinOp (13 variants)

| Variant | Description |
|---------|-------------|
| `Add` | Integer addition (wrapping) |
| `Sub` | Integer subtraction (wrapping) |
| `Mul` | Integer multiplication (wrapping) |
| `SDiv` | Signed division |
| `UDiv` | Unsigned division |
| `SRem` | Signed remainder |
| `URem` | Unsigned remainder |
| `And` | Bitwise AND |
| `Or` | Bitwise OR |
| `Xor` | Bitwise XOR |
| `Shl` | Shift left |
| `AShr` | Arithmetic shift right (sign-extending) |
| `LShr` | Logical shift right (zero-extending) |

Methods:
- `is_commutative()` -- returns `true` for `Add`, `Mul`, `And`, `Or`, `Xor`.
- `can_trap()` -- returns `true` for `SDiv`, `UDiv`, `SRem`, `URem` (divide by
  zero causes SIGFPE). Such operations must not be speculatively executed by
  if-conversion.
- `eval_i64(lhs, rhs) -> Option<i64>` -- evaluate on two `i64` operands using
  wrapping arithmetic. Returns `None` for division/remainder by zero. Unsigned
  operations (UDiv, URem, LShr) reinterpret the bits as `u64`.
- `eval_i128(lhs, rhs) -> Option<i128>` -- same for `i128` operands.

### IrUnaryOp (7 variants)

| Variant | Description |
|---------|-------------|
| `Neg` | Arithmetic negation |
| `Not` | Bitwise NOT |
| `Clz` | Count leading zeros |
| `Ctz` | Count trailing zeros |
| `Bswap` | Byte swap |
| `Popcount` | Population count (number of set bits) |
| `IsConstant` | `__builtin_constant_p`: returns 1 if operand is a compile-time constant, 0 otherwise. Lowered as an IR instruction so it can be resolved after inlining and constant propagation. |

### IrCmpOp (10 variants)

| Variant | Description |
|---------|-------------|
| `Eq` | Equal |
| `Ne` | Not equal |
| `Slt` | Signed less than |
| `Sle` | Signed less than or equal |
| `Sgt` | Signed greater than |
| `Sge` | Signed greater than or equal |
| `Ult` | Unsigned less than |
| `Ule` | Unsigned less than or equal |
| `Ugt` | Unsigned greater than |
| `Uge` | Unsigned greater than or equal |

Methods:
- `eval_i64(lhs, rhs) -> bool` -- evaluate on two `i64` operands. Signed
  comparisons use Rust's native `i64` ordering; unsigned comparisons reinterpret
  the bits as `u64`.
- `eval_i128(lhs, rhs) -> bool` -- same for `i128` operands.
- `eval_f64(lhs, rhs) -> bool` -- evaluate on two `f64` operands using IEEE 754
  semantics. For floats, signed and unsigned comparison variants are equivalent
  since IEEE 754 defines a total ordering.

### AtomicRmwOp (8 variants)

| Variant | Semantics |
|---------|-----------|
| `Add` | `*ptr += val` |
| `Sub` | `*ptr -= val` |
| `And` | `*ptr &= val` |
| `Or` | `*ptr \|= val` |
| `Xor` | `*ptr ^= val` |
| `Nand` | `*ptr = ~(*ptr & val)` |
| `Xchg` | `*ptr = val` (returns old value) |
| `TestAndSet` | `*ptr = 1` (returns old value) |

### AtomicOrdering (5 variants)

| Variant | Description |
|---------|-------------|
| `Relaxed` | No ordering constraints |
| `Acquire` | Subsequent loads/stores not reordered before this |
| `Release` | Previous loads/stores not reordered after this |
| `AcqRel` | Both acquire and release semantics |
| `SeqCst` | Sequentially consistent (total order) |

---

## Constants

### IrConst (9 variants)

| Variant | Payload | Description |
|---------|---------|-------------|
| `I8` | `i8` | 8-bit signed integer |
| `I16` | `i16` | 16-bit signed integer |
| `I32` | `i32` | 32-bit signed integer |
| `I64` | `i64` | 64-bit signed integer |
| `I128` | `i128` | 128-bit signed integer |
| `F32` | `f32` | 32-bit float |
| `F64` | `f64` | 64-bit float |
| `LongDouble` | `(f64, [u8; 16])` | Long double with dual representation: an `f64` approximation for computations (lossy, 52-bit mantissa) and raw IEEE 754 binary128 bytes with full 112-bit mantissa precision. For ARM64/RISC-V the bytes are emitted directly; for x86 they are converted to x87 80-bit format at emission time. |
| `Zero` | (none) | Zero value for any type |

Key methods on `IrConst`:

- **Predicates:** `is_zero()`, `is_one()`, `is_nonzero()`
- **Value extraction:** `to_i64()`, `to_i128()`, `to_u64()`, `to_f64()` (returns `None` when not applicable)
- **Construction:** `from_i64(val, ty)`, `zero(ty)`, `one(ty)`, `ptr_int(val)` (pointer-width `I32` or `I64`)
- **Type coercion:** `coerce_to(target_ty)`, `coerce_to_with_src(target_ty, src_ty)` (unsigned-aware), `narrowed_to(ty)`, `bool_normalize()` (to `I8(0)`/`I8(1)` per C11 6.3.1.2)
- **Long double constructors:** `long_double(v)`, `long_double_with_bytes(v, bytes)`, `long_double_from_i64(val)`, `long_double_from_u64(val)`, `long_double_from_i128(val)`, `long_double_from_u128(val)`
- **Float casts:** `cast_float_to_target(fv, target)`, `cast_long_double_to_target(fv, bytes, target)`
- **Serialization:** `push_le_bytes(out, size)` (target-default), `push_le_bytes_x86(out, size)` (x87 80-bit for long doubles), `push_le_bytes_riscv(out, size)` (binary128 for long doubles), `to_le_bytes()`, `x87_bytes()`
- **Hashing:** `to_hash_key()` converts to `ConstHashKey` which uses raw bit patterns for floats, enabling constants as `HashMap` keys for value numbering.

### Float Encoding Utilities

| Function | Description |
|----------|-------------|
| `f64_to_f128_bytes(val: f64) -> [u8; 16]` | Convert `f64` to IEEE 754 binary128 (quad-precision) encoding, 16 bytes little-endian. Quad format: 1 sign bit, 15 exponent bits (bias 16383), 112 mantissa bits (implicit leading 1). Used for long double on AArch64 and RISC-V. |
| `f64_to_x87_bytes(val: f64) -> [u8; 10]` | Convert `f64` to x87 80-bit extended precision encoding, 10 bytes little-endian. x87 format: 1 sign bit, 15 exponent bits (bias 16383), 64 mantissa bits (explicit integer bit). Used for long double on x86-64. |

---

## Intrinsics

`IntrinsicOp` defines **81 target-independent intrinsic operations**. Each
backend emits the appropriate native instructions for its architecture. The
variants are organized by ISA extension:

| Category | Count | Examples |
|----------|-------|---------|
| Fences and barriers | 5 | `Lfence`, `Mfence`, `Sfence`, `Pause`, `Clflush` |
| Non-temporal stores | 4 | `Movnti`, `Movnti64`, `Movntdq`, `Movntpd` |
| 128-bit load/store | 4 | `Loaddqu`, `Storedqu`, `Loadldi128`, `Storeldi128` |
| SSE2 packed integer | 10 | `Pcmpeqb128`, `Pcmpeqd128`, `Psubusb128`, `Psubsb128`, `Por128`, `Pand128`, `Pxor128`, `Pmovmskb128`, `SetEpi8`, `SetEpi32` |
| SSE2 packed 16-bit | 6 | `Paddw128`, `Psubw128`, `Pmulhw128`, `Pmaddwd128`, `Pcmpgtw128`, `Pcmpgtb128` |
| SSE2 packed 16-bit shifts | 3 | `Psllwi128`, `Psrlwi128`, `Psrawi128` |
| SSE2 packed 32-bit | 2 | `Paddd128`, `Psubd128` |
| SSE2 packed 32-bit shifts | 3 | `Psradi128`, `Pslldi128`, `Psrldi128` |
| SSE2 byte/bit shifts | 4 | `Pslldqi128`, `Psrldqi128`, `Psllqi128`, `Psrlqi128` |
| SSE2 shuffle | 3 | `Pshufd128`, `Pshuflw128`, `Pshufhw128` |
| SSE2 pack/unpack | 7 | `Packssdw128`, `Packsswb128`, `Packuswb128`, `Punpcklbw128`, `Punpckhbw128`, `Punpcklwd128`, `Punpckhwd128` |
| SSE2 insert/extract/convert | 6 | `SetEpi16`, `Pinsrw128`, `Pextrw128`, `Cvtsi128Si32`, `Cvtsi32Si128`, `Cvtsi128Si64` |
| SSE4.1 insert/extract | 6 | `Pinsrd128`, `Pextrd128`, `Pinsrb128`, `Pextrb128`, `Pinsrq128`, `Pextrq128` |
| AES-NI | 6 | `Aesenc128`, `Aesenclast128`, `Aesdec128`, `Aesdeclast128`, `Aesimc128`, `Aeskeygenassist128` |
| CLMUL | 1 | `Pclmulqdq128` |
| CRC32 | 4 | `Crc32_8`, `Crc32_16`, `Crc32_32`, `Crc32_64` |
| Scalar math | 4 | `SqrtF32`, `SqrtF64`, `FabsF32`, `FabsF64` |
| Frame/return address | 3 | `FrameAddress`, `ReturnAddress`, `ThreadPointer` |

Each variant name maps directly to an x86 instruction mnemonic (e.g.,
`Pcmpeqb128` -> `pcmpeqb`, `Aesenc128` -> `aesenc`). On ARM and RISC-V,
backends emit the equivalent native instructions. See `intrinsics.rs` for the
complete variant list with per-variant doc comments.

### Purity

The `is_pure()` method returns `true` for intrinsics with no side effects --
pure intrinsics can be dead-code eliminated when their result is unused. Most
SIMD arithmetic, shifts, shuffles, pack/unpack, insert/extract, AES-NI, CLMUL,
and scalar math operations are marked pure. Intrinsics with memory or ordering
side effects (fences, non-temporal stores, loads/stores) are not pure. See the
`is_pure()` match in `intrinsics.rs` for the authoritative list.

---

## Global Variables

`IrGlobal` defines a global variable with its initializer and linkage attributes:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `Rc<str>` | Symbol name (reference-counted for O(1) clone) |
| `ty` | `IrType` | Element type |
| `size` | `usize` | Size in bytes (for arrays: `elem_size * count`) |
| `align` | `usize` | Alignment in bytes |
| `init` | `GlobalInit` | Initializer |
| `is_static` | `bool` | File-scope (static) variable |
| `is_extern` | `bool` | Extern declaration (no storage emitted) |
| `is_common` | `bool` | `__attribute__((common))` -- use COMMON linkage |
| `is_weak` | `bool` | `__attribute__((weak))` -- weak symbol (STB_WEAK) |
| `section` | `Option<String>` | `__attribute__((section("...")))` -- specific ELF section |
| `is_const` | `bool` | Const-qualified (placed in `.rodata`) |
| `is_thread_local` | `bool` | `_Thread_local` or `__thread` (placed in `.tdata`/`.tbss`) |
| `visibility` | `Option<String>` | `__attribute__((visibility(...)))` |
| `is_used` | `bool` | `__attribute__((used))` -- prevent DCE |
| `has_explicit_align` | `bool` | User specified alignment via `__attribute__((aligned(N)))` or `_Alignas`; prevents auto-promotion to 16-byte alignment |

---

## Global Initializers

`GlobalInit` represents the initializer for a global variable. It has **10 variants**:

| Variant | Payload | Description |
|---------|---------|-------------|
| `Zero` | (none) | Zero-initialized (placed in `.bss`) |
| `Scalar` | `IrConst` | Single scalar constant |
| `Array` | `Vec<IrConst>` | Array of scalar constants |
| `String` | `String` | String literal (stored as bytes with null terminator) |
| `WideString` | `Vec<u32>` | Wide string literal (`wchar_t` values); backend adds null terminator |
| `Char16String` | `Vec<u16>` | `char16_t` string literal; backend adds null terminator |
| `GlobalAddr` | `Rc<str>` | Address of another global (e.g., `const char *s = "hello"`) |
| `GlobalAddrOffset` | `(Rc<str>, i64)` | Address of a global plus a byte offset (e.g., `&arr[3]`, `&s.field`) |
| `Compound` | `Vec<GlobalInit>` | Compound initializer sequence for arrays/structs with address expressions (e.g., `int *ptrs[] = {&a, &b, 0}`) |
| `GlobalLabelDiff` | `(Rc<str>, Rc<str>, usize)` | Difference of two labels `(label1, label2, byte_size)` for computed goto dispatch tables; `byte_size` is the width of the resulting integer (4 for int, 8 for long) |

Methods on `GlobalInit`:

| Method | Description |
|--------|-------------|
| `for_each_ref(f)` | Visit every symbol name referenced by this initializer. Calls `f` with each global/label name found in `GlobalAddr`, `GlobalAddrOffset`, and `GlobalLabelDiff` variants, recursing into `Compound` children. |
| `byte_size()` | Returns the byte size of this initializer element in a compound context. Used when flattening nested `Compound` elements into a parent. |
| `emitted_byte_size()` | Returns the total number of bytes that will be emitted for this initializer. Unlike `byte_size()`, this correctly accounts for pointer-sized `GlobalAddr` entries (4 bytes on i686, 8 bytes on 64-bit) and null terminators in string variants. |

---

## CFG Analysis

The analysis module (`analysis.rs`) provides shared CFG and dominator tree
utilities used by `mem2reg`, GVN, LICM, if-conversion, and IVSR.

### FlatAdj (Compressed Sparse Row Adjacency List)

`FlatAdj` stores `n` variable-length rows in two flat arrays:

- `offsets[i]..offsets[i+1]` is the range of indices into `data` for row `i`
- `data[offsets[i]..offsets[i+1]]` contains the neighbors of node `i`

This uses exactly **2 heap allocations** regardless of the number of rows
(compared to `n+1` for `Vec<Vec<usize>>`), and provides better cache locality
when iterating over adjacency lists.

Access: `row(i) -> &[u32]` returns the neighbor slice for node `i`.
Length: `len(i) -> usize` returns the number of neighbors of node `i`.

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_label_map` | `(func) -> FxHashMap<BlockId, usize>` | Build a map from block label (`BlockId`) to block index. |
| `build_cfg` | `(func, label_to_idx) -> (FlatAdj, FlatAdj)` | Build predecessor and successor `FlatAdj` lists from terminators, including `InlineAsm` goto edges. Uses 4 heap allocations total. |
| `compute_reverse_postorder` | `(num_blocks, succs) -> Vec<usize>` | DFS-based reverse postorder from the entry block. |
| `compute_dominators` | `(num_blocks, preds, succs) -> Vec<usize>` | Compute immediate dominators using the Cooper-Harvey-Kennedy algorithm ("A Simple, Fast Dominance Algorithm", 2001). Iterative over RPO, typically converges in 2-3 passes. Entry block: `idom[0] = 0`. Unreachable blocks: `usize::MAX`. |
| `compute_dominance_frontiers` | `(num_blocks, preds, idom) -> Vec<FxHashSet<usize>>` | Compute dominance frontiers from predecessors and idom. For join points (blocks with >= 2 predecessors), walks runners up the dominator tree. |
| `build_dom_tree_children` | `(num_blocks, idom) -> Vec<Vec<usize>>` | Invert the `idom` array into children lists. `children[b]` contains all blocks whose immediate dominator is `b`. |

### CfgAnalysis

`CfgAnalysis` is a cached bundle of pre-computed analysis results that can be
shared across multiple passes within a single pipeline iteration. Since passes
like GVN only replace instruction operands without modifying the CFG, these
results remain valid across GVN, LICM, and IVSR, avoiding redundant recomputation.

| Field | Type | Description |
|-------|------|-------------|
| `preds` | `FlatAdj` | Predecessor adjacency list |
| `succs` | `FlatAdj` | Successor adjacency list |
| `idom` | `Vec<usize>` | Immediate dominator for each block |
| `dom_children` | `Vec<Vec<usize>>` | Dominator tree children lists |
| `num_blocks` | `usize` | Number of blocks |

Constructed via `CfgAnalysis::build(func)`, which calls `build_label_map`,
`build_cfg`, `compute_dominators`, and `build_dom_tree_children` in sequence.

### Algorithm Details

**CFG construction** (`build_cfg`) iterates over each block's terminator to
extract successor edges (`Branch`, `CondBranch`, `IndirectBranch`, `Switch`).
It also scans instructions for `InlineAsm` goto labels, which create implicit
control flow edges. The function returns both predecessor and successor adjacency
lists in CSR format, using only 4 heap allocations total.

**Reverse postorder** (`compute_reverse_postorder`) performs a DFS from the
entry block (block 0) and collects blocks in postorder, then reverses. This
ordering is required by the dominator algorithm and ensures that each block is
processed after its dominator.

**Dominator computation** (`compute_dominators`) uses the Cooper-Harvey-Kennedy
algorithm. It iterates over blocks in reverse postorder, computing the immediate
dominator of each block as the intersection of its already-processed
predecessors' dominators. The `intersect` function walks two fingers up the
dominator tree, guided by RPO numbering, until they meet at the common
dominator. The algorithm converges in a small number of iterations for reducible
CFGs (typically 2-3 passes over the RPO).

**Dominance frontiers** (`compute_dominance_frontiers`) are computed from the
predecessor lists and the idom array. For each join point (block with 2 or more
predecessors), a runner walks up the dominator tree from each predecessor,
adding the join point to every block's frontier along the way, stopping when
it reaches the join point's immediate dominator. The result is
`Vec<FxHashSet<usize>>` where `df[b]` is the dominance frontier of block `b`.

---

## SSA Construction

SSA construction follows a two-phase approach, modeled after the LLVM strategy.

### Phase 1: Alloca-Based Lowering (`lowering/`)

Every local variable is lowered to an `Alloca` instruction (a stack slot)
accessed through `Load` and `Store` instructions. Function parameters get a
`ParamRef` instruction to make their initial value visible, followed by a `Store`
to the parameter's alloca. This representation is simple to generate because
every variable reference translates to a load or store through the alloca
pointer, regardless of control flow complexity. No SSA renaming or phi placement
is needed at this stage.

```
entry:
    %1 = alloca i32, 4, 4       // int x;
    %2 = alloca i32, 4, 4       // int y;
    %3 = paramref 0, i32        // incoming parameter value
    store %3, %1, i32           // x = param0
    ...
    %5 = load %1, i32           // read x
    %6 = binop add %5, const(1) // x + 1
    store %6, %2, i32           // y = x + 1
```

### Phase 2: mem2reg (`mem2reg/promote.rs`)

The `mem2reg` pass promotes eligible allocas to SSA registers using the standard
algorithm, implemented in **6 steps** (as numbered in the implementation):

1. **Identify promotable allocas** -- an alloca is promotable if it is scalar
   (size <= `MAX_PROMOTABLE_ALLOCA_SIZE` = 8 bytes), accessed only by
   `Load`/`Store` (not address-taken by `GetElementPtr`, `Memcpy`, `VaStart`,
   inline asm memory-only constraints, etc.), and not marked `volatile`.

2. **Build CFG** -- construct predecessor and successor adjacency lists from
   block terminators (and inline asm goto edges).

3. **Compute dominator tree** -- using the Cooper-Harvey-Kennedy algorithm.

4. **Compute dominance frontiers** -- for phi placement.

5. **Insert phi nodes** -- at the iterated dominance frontiers of each alloca's
   defining blocks (blocks containing stores to that alloca).

6. **Rename variables** -- via dominator-tree DFS, maintaining a stack of
   reaching definitions for each alloca. Replace loads with the current reaching
   definition and update the stack at stores. Fill in phi node operands from
   predecessor reaching definitions.

After mem2reg, the original alloca/load/store sequences are replaced with direct
value references and phi nodes, yielding proper SSA form.

The pass **runs twice** during compilation:

- **Before inlining** (`promote_allocas`): promotes non-parameter allocas.
  Parameter allocas are kept because the inliner assumes they exist for argument
  passing.
- **After inlining** (`promote_allocas_with_params`): promotes all allocas
  including parameters, since inlining is complete and parameter values are now
  explicit via `ParamRef` + `Store`.

---

## Phi Elimination

Phi elimination (`mem2reg/phi_eliminate.rs`) runs after all SSA optimizations and
before backend codegen. It converts each `Phi` instruction into `Copy`
instructions. Three cases are handled:

### Non-Conflicting (Common Case)

Direct copies are inserted at the end of each predecessor block (before the
terminator):

```
pred_block:
    %phi1_dest = copy src1
    %phi2_dest = copy src2
    <terminator>
```

### Conflicting (Cycles)

When phis form a cycle (e.g., a swap pattern where phi1 reads from phi2's dest
and vice versa), a two-phase temporary protocol is used to avoid the lost-copy
problem. The pass analyzes the copy graph per predecessor to detect cycles and
introduces shared temporaries:

```
pred_block:
    %tmp1 = copy src1     // save source before it's overwritten
    <terminator>

target_block:
    %phi1_dest = copy %tmp1  // restore from temporary
    ... rest of block ...
```

### Critical Edges (Trampoline Blocks)

When a predecessor block has multiple successors (e.g., a `CondBranch`) and the
target block has phis, placing copies at the end of the predecessor would execute
them on ALL outgoing paths, corrupting values on other edges. To fix this, the
critical edge is split by inserting a new **trampoline block** that contains only
the phi copies and branches unconditionally to the target. Trampoline block IDs
are allocated from a global counter (computed as the maximum block ID across all
functions) to avoid label collisions.

---

## Pass Interface

Optimization passes consume and transform the SSA IR. The pass pipeline lives
in `src/passes/` (see `passes/README.md` for the complete pass-by-pass
reference). This section describes how passes interact with the IR.

### Pass Signature

A per-function optimization pass is a function with signature:

```rust
fn my_pass(func: &mut IrFunction) -> usize
```

It takes a mutable reference to a single function and returns the number of
changes made (0 = no changes). This change count drives the fixed-point
iteration: the pipeline re-runs passes until no more changes occur or a
diminishing-returns threshold is reached.

### Running Passes

`IrModule::for_each_function(f)` iterates over all defined (non-declaration)
functions and calls `f` on each, accumulating the total change count. This is
the standard way to apply a per-function pass to the entire module.

For passes that need CFG/dominator analysis (GVN, LICM, IVSR), the pipeline
builds a shared `CfgAnalysis` once per function and passes it to all three.
Since GVN only replaces operands without modifying the CFG, the analysis remains
valid across all three passes.

### Pipeline Structure

The optimization pipeline runs inlining first (with its own mem2reg passes),
then enters a fixed-point loop (up to 3 iterations) over these per-function
passes:

1. CFG simplification
2. Copy propagation
3. Division-by-constant strength reduction (first iteration only, 64-bit targets only)
4. Integer narrowing
5. Algebraic simplification
6. Constant folding
7. GVN (Global Value Numbering) -- shares `CfgAnalysis` with LICM and IVSR
8. LICM (Loop-Invariant Code Motion)
9. IVSR (Induction Variable Strength Reduction, first iteration only)
10. If-conversion
11. Copy propagation (second round)
12. DCE (Dead Code Elimination)
13. CFG simplification (second round)
14. IPCP (Interprocedural Constant Propagation)

After the fixed-point loop, dead static function elimination removes
unreferenced `static` functions.  Each per-function pass has dependency
tracking: it only runs if it or its upstream passes made changes in the
previous iteration, avoiding redundant work.  See `passes/README.md` for
the complete dependency graph and per-pass documentation.

---

## IR Debugging

There is no structured textual IR format (unlike LLVM's `.ll` files). All IR
types derive `Debug`, so Rust's `{:?}` formatter can dump any IR structure,
though the output is verbose.

The inliner contains a private `dump_function_ir` helper that prints a
human-readable IR dump to stderr. It is activated via the `CCC_INLINE_DUMP_IR`
environment variable.

Other useful environment variables for debugging the IR pipeline:

| Variable | Effect |
|----------|--------|
| `CCC_DISABLE_PASSES=pass1,pass2,...` | Disable specific optimization passes (e.g., `gvn,licm`). Use `all` to skip the entire optimizer. |
| `CCC_TIME_PASSES=1` | Print per-pass timing and change counts to stderr. |
| `CCC_INLINE_DUMP_IR=1` | Dump function IR to stderr after each inlining event. |

---

## Key Design Decisions

1. **Alloca-based lowering, then mem2reg.** This separates concerns: the
   lowering pass handles C semantics without worrying about SSA, and mem2reg
   handles SSA construction without knowing C. This is the same strategy used
   by LLVM.

2. **Numeric `Value(u32)` IDs.** Values are identified by `Value(u32)`, not
   string names. This simplifies the IR, avoids name collision issues, enables
   flat lookup tables indexed by Value ID, and makes value comparison a simple
   integer comparison.

3. **Target-independent types.** The IR uses abstract types (`I8` through `I128`,
   `U8` through `U128`, `F32`, `F64`, `F128`, `Ptr`, `Void`) and delegates ABI
   details (calling conventions, register assignment, struct layout) to the
   backend. Backend-specific metadata is carried on `IrFunction`, `IrParam`,
   and `CallInfo`.

4. **Dual-representation `LongDouble`.** `IrConst::LongDouble(f64, [u8; 16])`
   carries both an `f64` approximation for constant folding and the raw IEEE 754
   binary128 bytes for precise code emission. This avoids the need for a software
   `f128` implementation while preserving precision where it matters.

5. **Canonical `for_each_used_value()` visitor.** A single method on
   `Instruction` that visits all used values. All optimization passes use this
   instead of duplicating the 38-arm match block, ensuring consistency and
   reducing maintenance burden.

6. **CSR adjacency lists.** The CFG uses flat CSR storage (`FlatAdj`) instead of
   `Vec<Vec<usize>>` because `build_cfg` is called per-function by multiple
   passes (GVN, LICM, if-conversion, mem2reg). The flat layout reduces heap
   allocations from `n+1` to 2 per `build_cfg` call and improves cache locality.

7. **`Rc<str>` for symbol names.** Function names, global names, call targets,
   and global initializer symbol references use `Rc<str>` instead of `String`.
   This makes `.clone()` O(1) instead of O(n), shrinks per-instance size from
   24 to 16 bytes, and auto-derefs to `&str` so most read sites need no changes.
   Optimization passes (inline, ipcp) that build hash maps keyed by function name
   benefit from O(1) key cloning.
