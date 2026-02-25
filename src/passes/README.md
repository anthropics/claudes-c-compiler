# Optimization Passes

This document describes the design and implementation of the optimization pass
pipeline. The pipeline transforms the compiler's intermediate representation (IR)
to produce better machine code by eliminating redundant computation, simplifying
control flow, and replacing expensive operations with cheaper equivalents.

Optimization levels control which passes run and how aggressively:

- **`-O0`**: Minimal — only mem2reg, resolve_asm, and dead_statics (for correctness).
- **`-O1`**: Basic — cfg_simplify, copy_prop, narrow, simplify, constant_fold, dce. Single iteration.
- **`-O2`**: Full pipeline — all passes, up to 3 iterations with dirty-tracking and diminishing-returns early exit.
- **`-O3`**: Aggressive — same passes as -O2 but with 5 iterations and a tighter 2% diminishing-returns threshold (vs 5%).
- **`-Os`/`-Oz`**: Same as -O2.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Phase Structure](#phase-structure)
3. [Dirty Tracking and Iteration Strategy](#dirty-tracking-and-iteration-strategy)
4. [Shared CFG Analysis](#shared-cfg-analysis)
5. [Pass Descriptions](#pass-descriptions)
6. [Pass Dependency Graph](#pass-dependency-graph)
7. [Disabling Individual Passes](#disabling-individual-passes)
8. [Files](#files)

---

## Pipeline Overview

The optimizer executes in four sequential phases:

```
 Phase 0: Inlining
     inline
     (convert gnu_inline defs to declarations)
     mem2reg
     constant_fold
     copy_prop
     simplify
     constant_fold
     copy_prop
     resolve_asm
         |
         v
 Phase 0.5: IsConstant Resolution
     resolve_remaining_is_constant
         |
         v
 Main Loop (up to 3 iterations, with dirty tracking)
   +-------------------------------------------------------+
   |  1.  cfg_simplify                                     |
   |  2.  copy_prop                                        |
   |  2a. div_by_const      (iteration 0 only, 64-bit targets only) |
   |  2b. narrow                                           |
   |  3.  simplify                                         |
   |  4.  constant_fold                                    |
   |  5.  gvn              \                               |
   |  6.  licm               > shared CFG analysis         |
   |  6a. iv_strength_reduce /  (iteration 0 only)         |
   |  7.  if_convert                                       |
   |  8.  copy_prop        (second round)                  |
   |  9.  dce                                              |
   | 10.  cfg_simplify     (second round)                  |
   | 10.5 ipcp             (interprocedural)               |
   +-------------------------------------------------------+
         |
         v
 Phase 11: Dead Static Elimination
     dead_statics
```

## Phase Structure

### Phase 0 -- Inlining

Function inlining runs first, before the main optimization loop. The inliner
substitutes the bodies of small static and `static inline` functions (as well as
`__attribute__((always_inline))` functions) into their call sites. After
inlining, `extern inline` functions with the `gnu_inline` attribute are
converted to declarations: their bodies existed only for the inliner, and
emitting them as definitions would cause infinite recursion when their internal
calls resolve to the local definition instead of the intended external library
symbol.

After inlining, a sequence of cleanup passes runs once on the freshly inlined
code:

1. **mem2reg** -- promotes stack allocations created during inlining back into
   SSA registers.
2. **constant_fold** -- folds constants that became visible after inlining
   (e.g., an inlined function that returns a constant).
3. **copy_prop** -- propagates copies introduced by phi elimination and mem2reg.
4. **simplify** -- applies algebraic identities to the simplified code.
5. **constant_fold** (again) -- catches additional folding opportunities exposed
   by simplification.
6. **copy_prop** (again) -- cleans up any remaining copies.
7. **resolve_asm** -- resolves inline assembly symbol references that became
   computable after inlining with constant arguments.

### Phase 0.5 -- IsConstant Resolution

After inlining and its cleanup passes, any `__builtin_constant_p` call whose
operand became a compile-time constant has already been resolved to `1` (true)
by constant folding. The remaining `IsConstant` instructions have operands that
are definitively not compile-time constants (parameters, globals, etc.), so this
phase resolves them all to `0` (false).

This resolution is critical: it allows `cfg_simplify` in the main loop to fold
conditional branches that test `__builtin_constant_p` results and eliminate dead
code paths. Without it, unreachable function calls guarded by
`__builtin_constant_p` would survive into the object file and cause linker
errors for intentionally undefined symbols (e.g., the Linux kernel's
`__bad_udelay()`, which is left undefined to generate a link error for invalid
`udelay()` arguments).

### Main Loop

The main optimization loop runs for up to 3 iterations. Each iteration executes
the full sequence of intraprocedural passes plus one interprocedural pass (IPCP)
at the end. The loop uses per-function dirty tracking and per-pass skip logic to
avoid redundant work, and it terminates early when either:

- No pass made any changes (the IR has reached a fixed point), or
- The iteration produced fewer than 5% as many changes as the first iteration
  (diminishing returns), unless IPCP made changes that require a full cleanup
  iteration.

Two passes run only during the first iteration: **div_by_const** (the expansion
is a one-time rewrite, and subsequent iterations clean up the generated code) and
**iv_strength_reduce** (a loop transformation that creates work for later cleanup
rather than benefiting from iteration).

### Phase 11 -- Dead Static Elimination

After all intraprocedural optimizations are complete, a final interprocedural
pass removes internal-linkage (`static`) functions and `static const` globals
that are no longer referenced by any live symbol. This is essential for `static
inline` functions from headers: after optimization eliminates dead code paths,
some callees become completely unreferenced and can be removed. Without this
cleanup, dead functions may reference undefined external symbols and cause
linker errors.

---

## Dirty Tracking and Iteration Strategy

The optimizer maintains two boolean vectors, each with one entry per function in
the module: **dirty** and **changed**.

- **dirty** indicates which functions should be visited during the current
  iteration. At the start of the first iteration, every function is dirty.
- **changed** accumulates which functions were modified by any pass during the
  current iteration.

The helper function `run_on_visited` enforces the dirty-tracking contract: it
skips declaration-only functions entirely, skips non-dirty functions, and marks
a function as changed whenever a pass reports modifications.

At the end of each iteration, `dirty` and `changed` are swapped: only functions
that were actually modified become dirty for the next iteration. This ensures
that functions which have already converged are not revisited.

### Per-Pass Skip Logic

In addition to per-function dirty tracking, the pipeline tracks how many changes
each pass made in the previous iteration. Before running a pass, the pipeline
checks whether the pass itself or any of its upstream dependencies made changes
last time. If neither the pass nor its upstream passes produced any changes, the
pass is skipped entirely for the current iteration.

This is driven by an explicit dependency graph encoded in the `should_run!`
macro. For example, `constant_fold` depends on `copy_prop`, `narrow`,
`simplify`, `if_convert`, and the second `copy_prop`; if none of those passes
made changes in the previous iteration, constant folding is skipped.

On the first iteration, all previous-change counts are set to `MAX` so every
pass runs unconditionally.

### Diminishing Returns and Early Exit

The pipeline tracks the total change count (excluding DCE) from the first
iteration as a baseline. If a subsequent iteration produces fewer than 5% of
that baseline, and IPCP did not make changes, the loop exits early. DCE is
excluded from this comparison because it is a cleanup pass whose large change
count (often thousands of removed dead instructions) would inflate the baseline
and make productive later iterations look like diminishing returns.

An exception is made for IPCP: if interprocedural constant propagation made
changes in an iteration, the pipeline always runs at least one more iteration
regardless of diminishing returns, because IPCP changes (constant arguments,
dead call elimination) require a full round of constant folding, DCE, and CFG
simplification to clean up.

The pipeline also guarantees at least two full iterations before the diminishing-
returns check can trigger, because multi-step constant propagation chains
(e.g., switch folding through layers of inlined helper functions) need at least
two iterations to fully propagate.

---

## Shared CFG Analysis

Three passes in the main loop -- GVN, LICM, and IVSR -- all require the same
expensive CFG analysis: label maps, predecessor/successor adjacency, dominator
trees, and natural loop detection. Rather than computing this independently in
each pass, the function `run_gvn_licm_ivsr_shared` builds a single
`CfgAnalysis` per function and passes it to all three.

This sharing is sound because:

- **GVN** only replaces operands within existing instructions; it does not add,
  remove, or reorder blocks, so the CFG structure is unchanged after GVN.
- **LICM** hoists instructions into preheader blocks but does not add or remove
  blocks from the CFG, so the analysis remains valid for IVSR.

For single-block functions, only GVN runs (via a fast path that skips CFG
analysis entirely), since LICM and IVSR require at least two blocks to form a
loop.

---

## Pass Descriptions

### inline -- Function Inlining

Substitutes callee function bodies into call sites to eliminate call overhead
and expose cross-function optimization opportunities. The inliner runs before
the main optimization loop because inlining creates the most significant
optimization opportunities: constant propagation through function arguments,
dead code elimination of unused branches, and elimination of small wrapper
functions.

**Eligibility.** Not all functions can be inlined. Variadic functions, functions
containing `VaStart`/`VaEnd`/`VaArg`, `DynAlloca`, `StackSave`/`StackRestore`,
`IndirectBranch`, or static locals with label references are excluded.
`__attribute__((noinline))` functions and recursive calls are also excluded.

**Heuristics.** The inliner uses a tiered size-based heuristic:

| Category | Instruction limit | Block limit | Notes |
|---|---|---|---|
| Tiny | 5 | 1 | Always inlined, ignores caller size |
| Small / static inline | 20 | 3 | Always inlined if budget not exhausted, or if `always_inline` |
| Normal static | 30 | 4 | Inlined if under caller budget |
| Normal eligible | 60 | 6 | Inlined if under caller budget |
| `always_inline` | 500 | 500 | Separate 200-instruction budget (main) + 400 (second pass) |

Inlining respects per-caller budgets: normal inlining stops when the caller
exceeds 200 instructions or 800 total inlined instructions, with a hard cap at
500 instructions (1000 absolute cap). The `always_inline` attribute has its own
200-instruction budget in the main loop, kept low to prevent stack frame bloat
(CCC allocates ~8 bytes per SSA value on the stack). When the caller has a
section attribute (e.g., `.init.text` in the Linux kernel), `always_inline`
callees bypass the budget entirely, ensuring kernel initialization code is always
fully inlined.

Each caller function is processed for up to 200 inlining rounds to handle chains
of inlined calls (e.g., A calls B calls C, where B is inlined into A, exposing
the call to C).

After the main loop, a **second pass** processes any remaining `always_inline`
call sites with an independent 400-instruction budget and up to 300 rounds. This
handles correctness-critical cases where the main loop's budget or round limit
was exhausted before resolving all always_inline chains (e.g., KVM nVHE functions
referencing section-specific symbols, or `cpucap_is_possible()` chains that must
be inlined to eliminate `__attribute__((error))` calls and undefined symbol
references). Small `always_inline` callees (≤20 instructions, ≤3 blocks) bypass
the budget in both passes because they have correctness requirements: inline asm
`"i"` constraints (e.g., `arch_static_branch`'s `__jump_table` entries) need
resolved symbol references. The two budgets are independent, so the combined
maximum is 200 + 400 = 600 always_inline instructions per caller. The second
pass does not enforce caller size caps (hard cap, absolute cap) because these
are correctness-critical inlines that must proceed regardless of caller size.

**Mechanics.** The inliner clones the callee's blocks with remapped value and
block IDs, wires arguments by inserting stores into the callee's parameter
allocas, creates a merge block for the return value (using a phi if the callee
has multiple return paths), and splices the cloned blocks into the caller.
After all inlining is complete, inline assembly symbol references are resolved
by tracing value chains back through `Load`, `Store`, `Copy`, `GEP`, and
`GlobalAddr` instructions.

### cfg_simplify -- CFG Simplification

Simplifies the control flow graph through seven sub-passes that run to a
fixpoint (repeating until no sub-pass makes changes):

1. **Fold constant conditional branches.** Resolves `CondBranch` conditions
   by walking Copy, Phi, Cmp, Select, and Cast chains -- both locally within a
   block and globally across single-predecessor chains (up to 8 hops) and across
   all blocks (up to 16 recursion depth). When the condition resolves to a
   constant, replaces the branch with an unconditional jump to the taken target.

2. **Fold constant switches.** Same idea for `Switch` terminators: if the
   discriminant resolves to a constant, replaces the switch with a direct branch
   to the matching case (or default).

3. **Simplify redundant conditional branches.** When a `CondBranch` has both
   targets pointing to the same block, replaces it with an unconditional branch.

4. **Thread jump chains.** Identifies empty blocks (no instructions,
   unconditional branch terminator) and redirects their predecessors to the
   final target. Chains are followed transitively with cycle detection, up to
   a depth limit of 32 hops. A safety check prevents threading when it would
   merge `CondBranch` edges that carry conflicting phi values.

5. **Remove dead blocks.** BFS from the entry block to compute reachability
   (also marking blocks referenced by `LabelAddr`, inline asm goto labels, and
   global initializer label blocks). Unreachable blocks are removed and their
   phi entries in surviving blocks are cleaned up.

6. **Simplify trivial phis.** Phi nodes with a single incoming edge, or where
   all incoming values are identical, are replaced with `Copy` instructions.

7. **Merge single-predecessor blocks.** When block A ends with an unconditional
   branch to block B and B has exactly one predecessor (A), fuses B into A.
   Skips blocks referenced by `LabelAddr`, inline asm goto labels, or global
   initializer labels.

This pass appears twice in each iteration of the main loop: once at the
beginning (to clean up from the previous iteration's constant folding and DCE)
and once at the end (to clean up after the current iteration's DCE and
if-conversion).

### copy_prop -- Copy Propagation

Replaces uses of a `Copy` instruction's destination with the `Copy`'s source
operand, transitively following chains of copies. This is important because many
other passes produce copies: phi simplification, mem2reg, algebraic
simplification (when an identity like `x + 0` reduces to `x`), and GVN (when a
redundant computation is replaced with a reference to an earlier equivalent).
Without copy propagation, each copy becomes a redundant register move or
load-store pair in code generation.

Uses a flat `Vec<Option<Operand>>` indexed by value ID for O(1) lookups, taking
advantage of the dense sequential nature of SSA value IDs. Chain resolution uses
union-find-style path compression: when resolving `%a = Copy %b` and
`%b = Copy %c`, all intermediate entries are updated to point directly to `%c`
in a single pass, giving amortized O(1) resolution. Multi-def detection prevents
propagating ambiguous values (e.g., a value defined by `Copy` in two different
branches). Chain depth is limited to 64 hops to prevent pathological cases.

After copy propagation, the dead `Copy` instructions themselves are cleaned up
by DCE. This pass appears twice in each iteration: once early (after CFG
simplification) and once late (after GVN, LICM, and if-conversion), to
propagate copies generated by intermediate passes.

### div_by_const -- Division by Constant Strength Reduction

Replaces integer division and modulo by compile-time constants with equivalent
multiply-and-shift sequences. On x86, `div`/`idiv` instructions cost 20-90
cycles, while the replacement sequence costs 3-5 cycles, making this one of the
most impactful single optimizations for integer-heavy code.

Supported transformations:

- **Unsigned division** (`x /u C`): replaced with a widened multiply by a magic
  number followed by a right shift, using the algorithm from Hacker's Delight
  by Henry S. Warren Jr. When the magic number exceeds 32 bits, an
  add-and-shift fixup is used.
- **Signed division by power-of-2** (`x /s 2^k`): replaced with a biased
  arithmetic right shift that handles negative rounding correctly.
- **Signed division by constant** (`x /s C`): replaced with a magic-number
  multiply, shift, and sign correction.
- **Signed division by negative constant** (`x /s -C`): delegates to the
  positive-constant case and negates the result.
- **Modulo** (`x % C`): rewritten as `x - (x / C) * C` using the optimized
  division from above. Power-of-2 unsigned modulo is handled separately by
  the simplify pass as `x & (C - 1)`.

**Only 32-bit operand values are currently optimized.** For native I32/U32
types, the pass applies directly. For I64/U64 operations, the pass performs
value-range analysis to detect operands that are provably 32-bit (e.g., values
produced by zero-extending or sign-extending from a 32-bit type, or loaded from
a 32-bit memory location). When both the dividend and divisor fit in 32 bits,
the pass generates the 32-bit magic-number sequence operating within the 64-bit
type. Genuine 64-bit divisions (where the dividend may exceed 32-bit range) fall
through to native `div`/`idiv` instructions.

The pass is disabled on 32-bit targets (i686) because the generated 64-bit
multiply-and-shift sequences cannot be executed correctly by the 32-bit backend.
It runs only during the first iteration of the main loop; subsequent iterations
clean up the expanded instruction sequences.

### narrow -- Integer Narrowing

Eliminates unnecessary widening introduced by C's integer promotion rules. The
compiler's frontend promotes sub-64-bit operands to I64 before performing
arithmetic, then narrows the result back. This creates a
widen-operate-narrow pattern that generates redundant sign/zero-extension
instructions.

The pass operates in three transformation phases:

**Phase 1: Narrow binary operations with explicit narrowing cast.** Detects the
canonical widen-operate-narrow pattern:

```
%w = Cast %x, I32 -> I64       (widen)
%r = BinOp add %w, %y, I64    (operate at I64)
%n = Cast %r, I64 -> I32       (narrow)
```

And rewrites to operate directly at the narrow type:

```
%r = BinOp add %x, narrow(%y), I32
```

This is safe for operations where the low bits are width-independent when
truncated: Add, Sub, Mul, And, Or, Xor, and Shl. Right shifts are safe when
the extension type matches the shift type (arithmetic shift with sign extension,
logical shift with zero extension). The BinOp result must have exactly one use
(the narrowing cast).

**Phase 2: Narrow binary operations without explicit cast.** On 64-bit targets,
narrows I64 bitwise operations (And, Or, Xor) to I32 when all operands are
provably 32-bit values, even without an explicit narrowing cast. This handles
cases where the result is consumed by another operation rather than an explicit
truncation. Uses both the widening cast map and a load-type map to identify
narrow operands. Disabled on 32-bit targets and restricted to I32/U32 output
to avoid sub-int register complications.

**Phase 3: Narrow comparisons.** When both operands of a comparison were
widened from the same type, narrows the comparison to operate at the original
type. Signed comparisons (Slt/Sle/Sgt/Sge) require the source type to be
signed; unsigned comparisons (Ult/Ule/Ugt/Uge) require unsigned; Eq/Ne are
safe with either sign. The RHS operand can be a constant if it round-trips
safely through extension to the comparison type.

### simplify -- Algebraic Simplification

Applies algebraic identities, strength reductions, and peephole optimizations
to individual instructions. This is one of the broadest passes in the pipeline.

**Identity simplifications** (integer-only, to preserve IEEE 754 float
semantics):

```
x + 0 => x          x - 0 => x          x - x => 0
x * 0 => 0          x * 1 => x          x / 1 => x
x / x => 1          x % 1 => 0          x % x => 0
x & 0 => 0          x & ~0 => x         x & x => x
x | 0 => x          x | ~0 => ~0        x | x => x
x ^ 0 => x          x ^ x => 0
x << 0 => x          x >> 0 => x
0 << x => 0          0 >> x => 0
```

Float identities are restricted to cases that respect IEEE 754: `x * 1.0` and
`x / 1.0` are safe, but `x + 0.0`, `x * 0.0`, `x - x`, and `x / x` are not
(due to signed zeros, NaN, and infinity).

**Strength reductions** (integer-only):

```
x * 2     => x + x
x * 2^k   => x << k
x * (-1)  => neg(x)
x /u 2^k  => x >>l k    (unsigned only)
x %u 2^k  => x & (2^k - 1)   (unsigned only)
```

**Constant reassociation.** When a binary operation's operand is itself a binary
operation with a constant, the two constants are combined. For example:
`(x + 3) + 5 => x + 8`, `(x * 4) * 8 => x * 32`, `(x & 0xFF) & 0x0F => x & 0x0F`,
`(x << 2) << 3 => x << 5`. Handles all commutative variants and cross-operator
interactions (e.g., `(x + C1) - C2 => x + (C1 - C2)`). When the combined
constant is an identity element, the entire operation is eliminated.

**Negation elimination.** Rewrites `x - (neg y)` as `x + y` by tracking
`UnaryOp::Neg` definitions.

**Cast chain optimization:**

```
Cast(x, T -> T)                       => Copy(x)       (identity cast)
Cast(const, A -> B)                   => Copy(folded)   (constant cast)
Cast(Cast(x:A, A->B), B->A) where A<=B => Copy(x)     (widen then narrow back)
Cast(Cast(x, A->B), B->C) where A<B<C  => Cast(x, A->C)  (double widen, same sign)
Cast(Cast(x, A->B), B->C) where A>B>C  => Cast(x, A->C)  (double narrow)
```

**Comparison simplifications:**

- Self-comparison: `Cmp(Eq, x, x)` => 1, `Cmp(Slt, x, x)` => 0, etc.
  (integer-only, float self-comparison is not safe due to NaN).
- Boolean test elimination: `Cmp(Ne, bool, 0)` => `bool`,
  `Cmp(Eq, bool, 0)` => inverted comparison (when the boolean comes from a Cmp).
  Boolean-ness is tracked transitively through And, Or, Xor of boolean values.
- Unsigned-zero: `Ult x, 0` => 0 (always false), `Uge x, 0` => 1 (always true),
  `Ule x, 0` => `Eq x, 0`, `Ugt x, 0` => `Ne x, 0`.

**Operand canonicalization.** Commutative BinOps and Cmp instructions with a
constant on the left are swapped to place the constant on the right, normalizing
the representation for downstream passes.

**Select simplification:**

```
select cond, x, x          => x
select Const(0), a, b      => b
select Const(nonzero), a, b => a
```

**GEP simplification:**

```
GEP(base, 0)                  => Copy(base)
GEP(GEP(base, C1), C2)        => GEP(base, C1 + C2)
```

**Math library call lowering.** Recognized calls are replaced with IR
intrinsics: `sqrt`/`sqrtf` => `SqrtF64`/`SqrtF32`, `fabs`/`fabsf` =>
`FabsF64`/`FabsF32`, `pow(x, 0.0)` => `1.0`, `pow(x, 1.0)` => `x`,
`pow(x, 2.0)` => `x * x`, `pow(x, -1.0)` => `1.0 / x`,
`pow(x, 0.5)` => `sqrt(x)`.

### constant_fold -- Constant Folding

Evaluates operations whose operands are all compile-time constants, replacing
the instruction with the computed result. Runs to a fixpoint within each
function, rebuilding the constant map each iteration so that folded results
chain into subsequent folds.

**Folded operation types:**

- **BinOp**: all integer arithmetic, bitwise, and shift operations. Float
  (F32/F64) arithmetic via native IEEE 754. F128/long double via platform-
  specific x87 (x86/i686) or software f128 (ARM/RISC-V) arithmetic. I128 via
  native Rust i128.
- **UnaryOp**: Neg, Not, Clz, Ctz, Popcount (width-sensitive), Bswap
  (16/32/64-bit), and IsConstant.
- **Cmp**: all 10 comparison operators for integer, float, F128 (with correct
  NaN unordered handling), and I128.
- **Cast**: integer-to-integer (with proper width truncation), float-to-float,
  float-to-int (with NaN/Inf/out-of-range safety), int-to-float, and I128
  casts with proper sign/zero extension.
- **Select**: both-arms-same folding, constant-condition folding, and
  both-arms-same-constant folding (even across different integer widths).
- **GetElementPtr**: constant base + constant offset.

**Width sensitivity.** Sub-32-bit types (I8, U8, I16, U16) require careful
promotion: the constant map tracks the target type of Cast instructions so that
I8 values are sign-extended while U8 values are zero-extended, implementing C11
integer promotion semantics.

**Division by zero.** Integer division/remainder by zero is not folded
(preserving undefined behavior). Float division by zero is folded (producing
Inf or NaN per IEEE 754).

**IsConstant handling.** The pass folds `IsConstant` to 1 when the operand is a
compile-time constant. Non-constant operands are left for the Phase 0.5
resolution step (see above).

### gvn -- Global Value Numbering

A dominator-based common subexpression elimination (CSE) pass with redundant
load elimination and store-to-load forwarding. Walks the dominator tree in
depth-first order, maintaining scoped hash tables that map expression keys to
previously computed values.

**Value-numbered expression types:**

- **BinOp** -- with commutative operand canonicalization (smaller value number
  first), so `a + b` and `b + a` receive the same value number.
- **UnaryOp** -- unary operations.
- **Cmp** -- comparisons (without commutative canonicalization).
- **Cast** -- type conversions keyed by source and destination type. Excluded
  for 128-bit types.
- **GetElementPtr** -- base + offset address computations.
- **Load** -- redundant load elimination (see below). Excluded for float,
  long double, and 128-bit types.

**Redundant load elimination.** Two loads from the same pointer with the same
type produce the same value if no intervening memory modification occurs. Load
value numbers use a generation counter: any memory-clobbering instruction
(Store, Call, CallIndirect, Memcpy, atomic operations, InlineAsm, fences, etc.)
bumps the generation, instantly invalidating all cached load entries. At block
merge points (blocks with >1 predecessor), the generation is also bumped to
conservatively invalidate inherited load entries.

**Store-to-load forwarding.** When a Store writes value V to pointer P, a
subsequent Load from P (matching value number and type, same generation) is
replaced with `Copy(V)`. This is tracked with the same generation scheme as
load CSE. Forwarding is disabled for escaped parameter allocas to preserve
correctness of the backend's ParamRef optimization.

**Scoped hash tables.** The pass maintains four rollback logs (pure expressions,
load entries, store-forwarding entries, and value number assignments). On
entering a dominator subtree, the current log positions are saved. On
backtracking, all entries beyond the saved position are undone, giving scoped
hash-table semantics without cloning. This is the same scoping pattern used by
mem2reg's SSA rename phase.

### licm -- Loop-Invariant Code Motion

Identifies natural loops in the CFG and hoists loop-invariant instructions to
preheader blocks that execute before the loop. An instruction is loop-invariant
if all its operands are constants, defined outside the loop, or themselves
defined by loop-invariant instructions (computed to a fixpoint).

Loops are processed innermost-first (sorted by body size). Each loop must have a
single-entry preheader block; loops with multiple outside predecessors are
skipped. Hoisted instructions are topologically sorted (Kahn's algorithm) before
insertion into the preheader to ensure def-before-use ordering.

**Safety rules:**

- **Pure instructions** (arithmetic, casts, GEP, Copy, GlobalAddr, Select, pure
  Intrinsics) are always hoistable. Division and remainder are excluded because
  they can trap (SIGFPE on divide-by-zero) and must not be speculatively
  executed.
- **Loads from non-address-taken allocas** are hoistable when the alloca is not
  stored to inside the loop body.
- **Loads from GlobalAddr pointers** are hoistable when the loop contains no
  calls, no stores to any global-derived pointer, and no direct stores to that
  specific global.
- **Loads from address-taken allocas** are never hoistable because stores
  through derived pointers may not be tracked.
- **All other loads** (runtime-computed pointers) are conservatively rejected
  since there is no alias analysis.

### iv_strength_reduce -- Induction Variable Strength Reduction

Transforms expensive per-iteration index computations in loops into cheaper
pointer increment operations. For a typical array access pattern:

```c
for (int i = 0; i < n; i++) sum += arr[i];
```

The IR computes `base + i * sizeof(int)` every iteration via a cast, shift (or
multiply), and GEP. After strength reduction, the loop maintains a running
pointer that is incremented by `sizeof(int)` each iteration, eliminating the
per-iteration multiply and cast.

**Finding induction variables.** The pass scans phi nodes in the loop header for
basic induction variables: `%iv = phi(init, %iv_next)` where
`%iv_next = %iv + constant_step`. Supports looking through Cast and Copy chains
to find the back-edge increment. Only single-latch loops (exactly one back-edge
block) are supported.

**Finding derived expressions.** Once basic IVs are identified, the pass looks
for multiplications or shifts by constants where one operand derives from a
basic IV (following Cast and Copy chains for up to 3 propagation passes). These
derived expressions that feed into GEP instructions become candidates for
strength reduction.

**Applying the transformation.** For each eligible GEP, the pass creates a new
pointer induction variable (a phi in the loop header) that starts at the initial
pointer and increments by `iv.step * stride` bytes each iteration. The original
GEP is replaced with a copy of the new pointer IV. The dead original
computations are removed by subsequent DCE.

**Limits:**

| Limit | Value | Purpose |
|---|---|---|
| Max stride | 1024 bytes | Avoids unusual access patterns |
| Max cast chain depth | 10 | Prevents pathological chain following |
| Shift amount range | 0..64 | Validates shift amounts before computing stride |

Runs only during the first iteration of the main loop. Uses shared CFG analysis
with GVN and LICM.

### if_convert -- If-Conversion

Converts simple diamond and triangle branch-and-phi patterns into `Select`
instructions, which lower to conditional moves (`cmov` on x86, `csel` on
AArch64).

**Diamond pattern:**

```
pred:
    condbranch %cond, true_block, false_block
true_block:                          false_block:
    (0-8 side-effect-free instrs)       (0-8 side-effect-free instrs)
    branch merge                        branch merge
merge:
    %result = phi [true_val, true_block], [false_val, false_block]
```

**Triangle pattern** (one arm falls through to merge):

```
pred:
    condbranch %cond, arm, merge
arm:
    (0-8 side-effect-free instructions)
    branch merge
merge:
    %result = phi [arm_val, arm], [pred_val, pred]
```

Both patterns are rewritten to:

```
pred:
    (hoisted arm instructions)
    %result = select %cond, true_val, false_val
    branch merge
```

Each arm may contain up to 8 instructions (to accommodate C type system
patterns like Load + Cast chains for parameter loads and sign extensions that
inflate instruction count). Both arms must be side-effect-free: no stores,
calls, loads, atomics, division/remainder (can trap), or inline assembly.
Branches with already-constant conditions are skipped (deferred to
cfg_simplify). F128 and I128 phi types are rejected. The pass iterates to a
fixpoint within each invocation, since converting one diamond may expose
another. Overlapping diamonds within a single iteration are detected and skipped
to avoid conflicts.

**Cost model.** The pass applies a cost model to avoid converting diamonds
where branchless code is slower than well-predicted branches:

| Limit | Value | Rationale |
|---|---|---|
| MAX_SELECTS | 1 | Each Select becomes a cmov chain (~6 x86 instructions). 2+ selects produce 12+ instructions, worse than a branch diamond of ~4-6 instructions with good prediction. |
| MAX_TOTAL_COST | 12 | Hoisted arm instructions + selects×5. Limits total speculated work even when select count is within bounds. |

These limits directly target the pathological case of nested if/else-if chains
(e.g., `if (isspace(*s)) {...} else if (!in_word) {...}` in tight loops), where
the fixpoint loop previously converted multiple diamonds iteratively, producing
4+ cmov chains in a single block. With the cost model, such patterns keep their
branches, allowing the CPU's branch predictor to skip untaken paths entirely.
Simple ternary expressions (1 select) are still converted, while multi-select
diamonds keep their branches for the branch predictor.

### ipcp -- Interprocedural Constant Propagation

An interprocedural pass that performs three optimizations across function
boundaries:

1. **Constant return propagation.** Identifies defined, non-weak, non-variadic,
   side-effect-free functions that return the same constant on every return path.
   Replaces all `Call` instructions to these functions with the constant value.
   Side-effect-free means no stores, calls, loads, atomics, inline asm, memcpy,
   or any va_arg operations.

2. **Dead call elimination.** Identifies defined, non-weak, side-effect-free,
   void-returning functions with no `Unreachable` terminators (traps are
   observable side effects). Removes entire `Call` instructions to these
   functions from all callers. This eliminates references to symbols that would
   otherwise cause linker errors.

3. **Constant argument propagation.** For static, defined, non-weak,
   non-variadic functions: when all call sites pass the same constant for a
   given parameter, replaces the `ParamRef` instruction in the function body
   with a copy of that constant. Functions whose address is taken (via
   `GlobalAddr` or global initializer references) have all parameters marked as
   varying since they may be called indirectly with unknown arguments.

IPCP runs at the end of every iteration (not just the first), because earlier
passes within the same iteration may simplify call arguments to constants (e.g.,
phi nodes collapsed after CFG simplification resolves dead branches). When IPCP
makes changes, the pipeline marks all functions dirty and always runs at least
one more iteration to allow cleanup passes to process the newly exposed
constants.

### dce -- Dead Code Elimination

Removes instructions whose results are never used by any other instruction or
terminator. Uses a use-count-based worklist algorithm with O(n) complexity:

1. Build use counts for all values in a single forward pass. Self-referencing
   phi edges (`v = phi [..., v, ...]`) are excluded from the count to prevent
   dead self-referencing phis from surviving.
2. Build a definition map from value IDs to their defining instruction
   locations, excluding side-effecting instructions.
3. Seed the worklist with non-side-effecting instructions that have zero use
   count.
4. Process the worklist: for each dead instruction, decrement its operands' use
   counts (saturating); if any operand's count drops to zero, add its defining
   instruction to the worklist.
5. Sweep all dead instructions in a single pass.

**Side-effecting instructions** are never removed regardless of whether their
result is used: Store, Call, CallIndirect, Alloca, DynAlloca, Memcpy, all atomic
operations (including Fence), InlineAsm, StackRestore, VaStart/VaEnd/VaCopy/
VaArg/VaArgStruct, multi-return helpers (GetReturn*Second, SetReturn*Second),
and Intrinsics that are non-pure or have a destination pointer (the store
through the destination pointer is a side effect even if the intrinsic
computation itself is pure). Alloca is conservatively kept because the backend
uses positional indexing for parameter mapping.

### dead_statics -- Dead Static Function and Global Elimination

Removes internal-linkage (`static`) functions and `static const` globals that
are unreachable from any externally visible symbol. Uses a BFS reachability
analysis:

**Roots** (seeds for BFS):
- All non-static defined functions and non-static defined globals
- Functions and globals with the `is_used` attribute
- All aliases (both the alias name and its target)
- All constructors and destructors
- Address-taken `static always_inline` functions
- Static symbols whose names appear in `toplevel_asm` (conservative check)
- Common globals

**References** are scanned from function bodies (`Call`, `GlobalAddr`,
`InlineAsm.input_symbols`) and global initializer trees. Any static symbol not
reached during BFS is removed from the module. After removal, the pass also
cleans up `module.symbol_attrs` to remove entries for unreferenced symbols.

### resolve_asm -- Inline Assembly Symbol Resolution

A post-inlining fixup pass that resolves symbolic references in inline assembly
operands. After inlining a function like `_static_cpu_has(u16 bit)` with a
constant argument, the IR may contain a `GlobalAddr` + `GEP` chain that
represents a specific offset into a global variable. This pass builds a
lightweight definition map (`DefInfo` enum tracking GlobalAddr, GEP, Add, and
Cast/Copy instructions) and traces those chains to produce symbol strings like
`"boot_cpu_data+74"` for the backend.

Only immediate-constraint ("i") operands with unresolved `input_symbols` entries
are processed. Non-constant GEP offsets cause resolution to fail gracefully.

### loop_analysis -- Shared Loop Analysis Utilities

A utility module (not a standalone pass) providing natural loop detection and
loop body computation used by both LICM and IVSR.

- **Back edge detection.** For each edge `tail -> header` in the CFG, checks
  if `header` dominates `tail` by walking the immediate dominator chain.
- **Loop body computation.** Seeds the body with the header. For each back edge,
  adds the tail and performs reverse BFS: adds all predecessors not already in
  the body, collecting all blocks that can reach the tail without going through
  the header.
- **Loop merging.** Multiple back edges targeting the same header produce
  separate loop entries; these are merged by taking the union of all bodies.
- **Preheader detection.** Finds the single predecessor of the header that is
  not in the loop body. Returns `None` if there is not exactly one such
  predecessor.

---

## Pass Dependency Graph

The following diagram shows the actual `should_run!` macro dependencies encoded
in `mod.rs`. An arrow from A to B means "B checks whether A made changes; if A
(or B itself) made no changes in the previous iteration, B is skipped."

```
                     +-> narrow -------> simplify -> constant_fold
                     |                      |    \       |
cfg_simplify1 -> copy_prop1 --------> gvn --+     +-----+
     ^               |                  |          |     |
     |               +--------> licm ---+     if_convert |
     |               |                         |   |     |
     |          constant_fold <------ copy_prop2 <-+     |
     |               |                  |                |
     |               v                  v                |
     +------------ cfg_simplify2 <---- dce <-------------+
```

**Upstream dependencies (encoded in `should_run!` macro):**

| Pass | Upstream dependencies |
|---|---|
| cfg_simplify (1st) | constant_fold, dce |
| copy_prop (1st) | cfg_simplify, gvn, licm, if_convert |
| narrow | copy_prop |
| simplify | copy_prop, narrow |
| constant_fold | copy_prop, narrow, simplify, if_convert, copy_prop (2nd) |
| gvn | cfg_simplify, copy_prop, simplify |
| licm | cfg_simplify, copy_prop, gvn |
| if_convert | cfg_simplify, constant_fold |
| copy_prop (2nd) | gvn, licm, if_convert + current-iter simplify, constant_fold |
| dce | gvn, licm, if_convert, copy_prop (2nd) |
| cfg_simplify (2nd) | constant_fold, if_convert, dce |

**IPCP** is not part of the `should_run!` system. When IPCP makes changes, it
marks all functions dirty, which forces all passes to re-run in the next
iteration via the dirty tracking mechanism rather than the per-pass skip logic.

---

## Disabling Individual Passes

Individual passes can be disabled at runtime for debugging by setting the
`CCC_DISABLE_PASSES` environment variable to a comma-separated list of pass
names:

```
CCC_DISABLE_PASSES=gvn,licm ./ccc input.c -o output.o
```

Recognized names: `all`, `inline`, `cfg`, `copyprop`, `narrow`, `simplify`,
`constfold`, `gvn`, `licm`, `ifconv`, `dce`, `ipcp`, `divconst`, `ivsr`.

Setting the variable to `all` skips the entire optimization pipeline.

Pass timing information can be enabled with:

```
CCC_TIME_PASSES=1 ./ccc input.c -o output.o
```

This prints per-pass, per-function timing and change counts to stderr, which is
useful for identifying performance bottlenecks and understanding convergence
behavior across iterations.

---

## Files

| File                     | Description                                            |
|--------------------------|--------------------------------------------------------|
| `mod.rs`                 | Pipeline orchestration, dirty tracking, shared analysis |
| `cfg_simplify.rs`        | CFG simplification (branch folding, jump threading)    |
| `constant_fold.rs`       | Constant expression evaluation at compile time         |
| `copy_prop.rs`           | Copy propagation with path compression                 |
| `dce.rs`                 | Dead code elimination (use-count worklist)              |
| `dead_statics.rs`        | Dead static function/global elimination (BFS)          |
| `div_by_const.rs`        | Division by constant strength reduction                |
| `gvn.rs`                 | Dominator-based GVN/CSE with load elimination          |
| `if_convert.rs`          | Diamond/triangle to Select conversion                  |
| `inline.rs`              | Function inlining with tiered size heuristics          |
| `ipcp.rs`                | Interprocedural constant propagation                   |
| `iv_strength_reduce.rs`  | Loop induction variable strength reduction             |
| `licm.rs`                | Loop-invariant code motion                             |
| `loop_analysis.rs`       | Shared loop detection and body computation utilities   |
| `narrow.rs`              | Integer narrowing (3-phase C promotion elimination)    |
| `resolve_asm.rs`         | Post-inline assembly symbol resolution                 |
| `simplify.rs`            | Algebraic simplification, strength reduction, and peephole |
