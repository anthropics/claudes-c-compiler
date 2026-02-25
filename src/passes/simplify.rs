//! Algebraic simplification and strength reduction pass.
//!
//! Applies algebraic identities to simplify instructions (integer types):
//! - x + 0 => x, 0 + x => x
//! - x - 0 => x, x - x => 0
//! - x * 0 => 0, x * 1 => x
//! - x / 1 => x, x / x => 1
//! - x % 1 => 0, x % x => 0
//! - x & 0 => 0, x & x => x, x & all_ones => x
//! - x | 0 => x, x | x => x, x | all_ones => all_ones
//! - x ^ 0 => x, x ^ x => 0
//! - x << 0 => x, x >> 0 => x
//!
//! Float-unsafe simplifications (x + 0, 0 + x, x * 0, x - x, x / x) are
//! restricted to integer types to preserve IEEE 754 semantics (signed zeros,
//! NaN propagation, infinity arithmetic).
//!
//! Strength reduction (integer types only):
//! - x * 2^k => x << k  (multiply by power-of-2 to shift)
//! - x * 2 => x + x     (slightly cheaper on some uarches)
//! - x * (-1) => neg(x)  (single neg instruction vs imul)
//! - x / 2^k => x >> k  (unsigned divide by power-of-2 to logical shift)
//!
//! Constant reassociation (requires def lookup):
//! - (x + C1) + C2 => x + (C1 + C2)
//! - (x - C1) - C2 => x - (C1 + C2)
//! - (x * C1) * C2 => x * (C1 * C2)
//! - (x & C1) & C2 => x & (C1 & C2)
//! - (x | C1) | C2 => x | (C1 | C2)
//! - (x ^ C1) ^ C2 => x ^ (C1 ^ C2)
//! - (x << C1) << C2 => x << (C1 + C2)
//!
//! Negation elimination (requires def lookup):
//! - x - (neg y) => x + y
//!
//! Redundant instruction elimination:
//! - Cast where from_ty == to_ty => Copy
//! - GetElementPtr chain: GEP(GEP(base, c1), c2) => GEP(base, c1+c2)
//! - GetElementPtr with constant zero offset => Copy of base
//!
//! Cast chain optimization (requires def lookup):
//! - Cast(Cast(x, A->B), B->A) where A fits in B => Copy of x (widen-then-narrow)
//! - Cast(Cast(x, A->B), B->C) => Cast(x, A->C) (double widen/narrow)
//! - Cast of constant => constant (fold at compile time)
//!
//! Comparison simplification (requires def lookup):
//! - Cmp(Ne, cmp_result, 0) => Copy(cmp_result) -- redundant boolean test
//! - Cmp(Eq, cmp_result, 0) => Cmp(inverted_op, orig_lhs, orig_rhs) -- negated comparison
//! - Cmp(Ne, cmp_result, 1) => Cmp(inverted_op, orig_lhs, orig_rhs) -- negated boolean
//! - Cmp(Eq, cmp_result, 1) => Copy(cmp_result) -- redundant boolean test
//! - Cmp(op, x, x) => constant -- self-comparison
//! - Cmp(Ule, x, 0) => Cmp(Eq, x, 0) -- unsigned x <= 0 is x == 0
//! - Cmp(Ugt, x, 0) => Cmp(Ne, x, 0) -- unsigned x > 0 is x != 0
//!
//! Operand canonicalization (for better codegen):
//! - Cmp(op, Const, Value) => Cmp(swapped_op, Value, Const)
//! - BinOp(commutative_op, Const, Value) => BinOp(op, Value, Const)
//!
//! Select simplification:
//! - select cond, x, x => x (both arms same)
//! - select Const(0), a, b => b (constant false condition)
//! - select Const(nonzero), a, b => a (constant true condition)

use crate::common::types::IrType;
use crate::ir::reexports::{
    Instruction,
    IntrinsicOp,
    IrBinOp,
    IrCmpOp,
    IrConst,
    IrFunction,
    IrModule,
    IrUnaryOp,
    Operand,
    Value,
};

/// Run algebraic simplification on the module.
/// Returns the number of instructions simplified.
pub fn run(module: &mut IrModule) -> usize {
    module.for_each_function(simplify_function)
}

pub(crate) fn simplify_function(func: &mut IrFunction) -> usize {
    let mut total = 0;

    // Build def maps for chain optimizations: Value -> defining instruction
    // We use flat Vecs indexed by Value ID for O(1) lookup.
    let max_id = func.max_value_id() as usize;
    let mut cast_defs: Vec<Option<CastDef>> = vec![None; max_id + 1];
    let mut gep_defs: Vec<Option<GepDef>> = vec![None; max_id + 1];
    let mut cmp_defs: Vec<Option<CmpDef>> = vec![None; max_id + 1];
    let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; max_id + 1];
    let mut neg_defs: Vec<Option<NegDef>> = vec![None; max_id + 1];
    // Track values known to be boolean (0 or 1). This includes Cmp results
    // and bitwise And/Or/Xor of boolean values.
    let mut is_boolean = vec![false; max_id + 1];

    // Collect definitions - first pass: mark Cmp results as boolean
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::Cast { dest, src, from_ty, to_ty } => {
                    set_def(&mut cast_defs, dest.0, CastDef {
                        src: *src, from_ty: *from_ty, to_ty: *to_ty,
                    });
                }
                Instruction::GetElementPtr { dest, base, offset, .. } => {
                    set_def(&mut gep_defs, dest.0, GepDef {
                        base: *base, offset: *offset,
                    });
                }
                Instruction::Cmp { dest, op, lhs, rhs, ty } => {
                    set_def(&mut cmp_defs, dest.0, CmpDef {
                        op: *op, lhs: *lhs, rhs: *rhs, ty: *ty,
                    });
                    let id = dest.0 as usize;
                    if id < is_boolean.len() {
                        is_boolean[id] = true;
                    }
                }
                Instruction::BinOp { dest, op, lhs, rhs, ty } => {
                    // Track BinOp definitions where at least one operand is a constant.
                    // This enables constant reassociation: (x + C1) + C2 => x + (C1+C2),
                    // and multiply reassociation: (x * C1) * C2 => x * (C1*C2).
                    if matches!(lhs, Operand::Const(_)) || matches!(rhs, Operand::Const(_)) {
                        set_def(&mut binop_defs, dest.0, BinOpDef {
                            op: *op, lhs: *lhs, rhs: *rhs, ty: *ty,
                        });
                    }
                }
                Instruction::UnaryOp { dest, op: IrUnaryOp::Neg, src, .. } => {
                    set_def(&mut neg_defs, dest.0, NegDef { src: *src });
                }
                _ => {}
            }
        }
    }

    // Second pass: propagate boolean-ness through And/Or/Xor of boolean values.
    // A single pass is sufficient for most patterns since And/Or/Xor of Cmp results
    // are the dominant case. For deeply nested boolean expressions, this misses some
    // cases, but those are rare in practice.
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::BinOp { dest, op, lhs, rhs, .. } = inst {
                if matches!(op, IrBinOp::And | IrBinOp::Or | IrBinOp::Xor)
                    && operand_is_boolean(lhs, &is_boolean) && operand_is_boolean(rhs, &is_boolean) {
                        let id = dest.0 as usize;
                        if id < is_boolean.len() {
                            is_boolean[id] = true;
                        }
                    }
            }
        }
    }

    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if let Some(simplified) = try_simplify(inst, &cast_defs, &gep_defs, &cmp_defs, &binop_defs, &neg_defs, &is_boolean) {
                *inst = simplified;
                total += 1;
            }
        }
    }
    total
}

/// Store a definition in a flat Vec, bounds-checked.
fn set_def<T>(defs: &mut [Option<T>], id: u32, val: T) {
    let idx = id as usize;
    if idx < defs.len() {
        defs[idx] = Some(val);
    }
}

/// Check whether an operand is known to be boolean (0 or 1).
fn operand_is_boolean(op: &Operand, is_boolean: &[bool]) -> bool {
    match op {
        Operand::Value(v) => {
            let id = v.0 as usize;
            id < is_boolean.len() && is_boolean[id]
        }
        Operand::Const(c) => matches!(c.to_i64(), Some(0) | Some(1)),
    }
}

/// Cached information about a Cast instruction for chain elimination.
#[derive(Clone, Copy)]
struct CastDef {
    src: Operand,
    from_ty: IrType,
    to_ty: IrType,
}

/// Cached information about a GEP instruction for chain folding.
#[derive(Clone, Copy)]
struct GepDef {
    base: Value,
    offset: Operand,
}

/// Cached information about a Cmp instruction for comparison simplification.
#[derive(Clone, Copy)]
struct CmpDef {
    op: IrCmpOp,
    lhs: Operand,
    rhs: Operand,
    ty: IrType,
}

/// Cached information about a UnaryOp::Neg instruction.
/// Used to simplify `x - (neg y)` => `x + y`.
#[derive(Clone, Copy)]
struct NegDef {
    src: Operand,
}

/// Cached information about a BinOp instruction for constant reassociation.
/// We only store BinOps where at least one operand is a constant, since
/// reassociation requires folding two constants together.
#[derive(Clone, Copy)]
struct BinOpDef {
    op: IrBinOp,
    lhs: Operand,
    rhs: Operand,
    ty: IrType,
}

/// Try to simplify an instruction using algebraic identities and strength reduction.
fn try_simplify(
    inst: &Instruction,
    cast_defs: &[Option<CastDef>],
    gep_defs: &[Option<GepDef>],
    cmp_defs: &[Option<CmpDef>],
    binop_defs: &[Option<BinOpDef>],
    neg_defs: &[Option<NegDef>],
    is_boolean: &[bool],
) -> Option<Instruction> {
    match inst {
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            simplify_binop(*dest, *op, lhs, rhs, *ty, binop_defs, neg_defs)
        }
        Instruction::Cast { dest, src, from_ty, to_ty } => {
            simplify_cast(*dest, src, *from_ty, *to_ty, cast_defs)
        }
        Instruction::GetElementPtr { dest, base, offset, ty } => {
            simplify_gep(*dest, *base, offset, *ty, gep_defs)
        }
        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            simplify_cmp(*dest, *op, lhs, rhs, *ty, cmp_defs, is_boolean)
        }
        Instruction::Select { dest, cond, true_val, false_val, .. } => {
            // select cond, x, x => x (both arms are the same)
            if same_operand(true_val, false_val) {
                return Some(Instruction::Copy { dest: *dest, src: *true_val });
            }
            // select Const(0), a, b => b (constant false condition)
            // select Const(nonzero), a, b => a (constant true condition)
            if let Operand::Const(c) = cond {
                if c.is_zero() {
                    return Some(Instruction::Copy { dest: *dest, src: *false_val });
                } else {
                    return Some(Instruction::Copy { dest: *dest, src: *true_val });
                }
            }
            None
        }
        Instruction::Call { func, info } => {
            if let Some(d) = info.dest {
                simplify_math_call(d, func, &info.args, info.return_type)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Simplify a Cast instruction.
///
/// Handles:
/// - Cast from type T to same type T => Copy
/// - Cast chain: Cast(Cast(x, A->B), B->C) optimizations
/// - Cast of constant => constant (fold at compile time)
fn simplify_cast(
    dest: Value,
    src: &Operand,
    from_ty: IrType,
    to_ty: IrType,
    def_map: &[Option<CastDef>],
) -> Option<Instruction> {
    // Identity cast: same type
    if from_ty == to_ty {
        return Some(Instruction::Copy { dest, src: *src });
    }

    // Constant folding: cast of a constant => new constant
    if let Operand::Const(c) = src {
        if let Some(folded) = fold_const_cast(c, from_ty, to_ty) {
            return Some(Instruction::Copy {
                dest,
                src: Operand::Const(folded),
            });
        }
    }

    // Cast chain optimization: if src is defined by another Cast, try to fold.
    // Only handle the widen-then-narrow-back case which is safe and common.
    if let Operand::Value(v) = src {
        let idx = v.0 as usize;
        if let Some(Some(inner_cast)) = def_map.get(idx) {
            let inner_from = inner_cast.from_ty;
            let inner_to = inner_cast.to_ty;
            let inner_src = inner_cast.src;

            // Verify chain consistency: inner output type must match our input type
            if inner_to == from_ty {
                // Widen then narrow back to exact same type (most common C pattern).
                // E.g., Cast(Cast(x:I32, I32->I64), I64->I32) => Copy of x
                // Safe because widening preserves all bits, then narrowing discards
                // the high bits we added - yielding the original value unchanged.
                // The size guard (inner_from.size() <= from_ty.size()) ensures the
                // first cast was a widening, not a narrowing (which loses bits).
                if inner_from == to_ty
                    && inner_from.is_integer() && from_ty.is_integer()
                    && inner_from.size() <= from_ty.size()
                {
                    return Some(Instruction::Copy { dest, src: inner_src });
                }

                // Double widen: Cast(Cast(x, A->B), B->C) where A < B < C (all ints)
                // => Cast(x, A->C) - skip intermediate.
                // ONLY safe when A and B have the same signedness, because the
                // extension from B to C uses B's signedness. If we skip B and go
                // directly A->C, the extension uses A's signedness instead. When
                // A and B differ in signedness (e.g., I8->U16->I32), the
                // intermediate unsigned type changes the extension behavior and
                // skipping it would be incorrect.
                if inner_from.is_integer() && from_ty.is_integer() && to_ty.is_integer() {
                    let a = inner_from.size();
                    let b = from_ty.size();
                    let c = to_ty.size();
                    if a < b && b < c {
                        // Only safe when inner source and intermediate have same signedness
                        let same_sign = inner_from.is_signed() == from_ty.is_signed()
                            || inner_from.is_unsigned() == from_ty.is_unsigned();
                        if same_sign {
                            return Some(Instruction::Cast {
                                dest,
                                src: inner_src,
                                from_ty: inner_from,
                                to_ty,
                            });
                        }
                    }

                    // Double narrow: Cast(Cast(x, A->B), B->C) where A > B > C (all ints)
                    // => Cast(x, A->C) - skip intermediate narrow.
                    // Safe because narrowing only keeps the low bits, so
                    // narrow(narrow(x, A->B), B->C) = narrow(x, A->C) regardless
                    // of signedness (both just truncate to C bits).
                    if a > b && b > c {
                        return Some(Instruction::Cast {
                            dest,
                            src: inner_src,
                            from_ty: inner_from,
                            to_ty,
                        });
                    }
                }
            }
        }
    }

    None
}

/// Fold a cast of a constant at compile time.
/// Returns the new constant if the cast can be folded, None otherwise.
///
/// `from_ty` is needed because IrConst::to_i64() always sign-extends from
/// the storage type (I8/I16/I32/I64), but unsigned source types require
/// zero-extension. For example, IrConst::I8(-1) with from_ty=U8 represents
/// the value 255, not -1.
fn fold_const_cast(c: &IrConst, from_ty: IrType, to_ty: IrType) -> Option<IrConst> {
    // Handle I128 source directly to avoid truncation through to_i64().
    // to_i64() truncates the upper 64 bits, which loses data for values like (U)1 << 127.
    if let IrConst::I128(v128) = c {
        return fold_i128_cast(*v128, from_ty, to_ty);
    }

    // Integer-to-integer/float constant cast
    if let Some(raw_val) = c.to_i64() {
        // Normalize source value according to from_ty signedness.
        // Signed types sign-extend; unsigned types zero-extend.
        let val = from_ty.truncate_i64(raw_val);
        return fold_int_cast(val, from_ty, to_ty);
    }

    // Float-to-other constant cast
    // For LongDouble, use full x87 precision for integer targets
    if let IrConst::LongDouble(fv, bytes) = c {
        return IrConst::cast_long_double_to_target(*fv, bytes, to_ty);
    }
    if let Some(fval) = c.to_f64() {
        return IrConst::cast_float_to_target(fval, to_ty);
    }

    None
}

/// Convert an i128 constant to the target type.
fn fold_i128_cast(v128: i128, from_ty: IrType, to_ty: IrType) -> Option<IrConst> {
    let bits_lo = v128 as u64;
    Some(match to_ty {
        IrType::I8 | IrType::U8 => IrConst::I8(bits_lo as i8),
        IrType::I16 | IrType::U16 => IrConst::I16(bits_lo as i16),
        IrType::I32 => IrConst::I32(bits_lo as i32),
        IrType::U32 => IrConst::I64(bits_lo as u32 as i64),
        IrType::I64 | IrType::U64 | IrType::Ptr => IrConst::I64(bits_lo as i64),
        IrType::I128 | IrType::U128 => IrConst::I128(v128),
        IrType::F32 => {
            if from_ty.is_unsigned() {
                IrConst::F32((v128 as u128) as f32)
            } else {
                IrConst::F32(v128 as f32)
            }
        }
        IrType::F64 => {
            let fv = if from_ty.is_unsigned() { (v128 as u128) as f64 } else { v128 as f64 };
            IrConst::F64(fv)
        }
        IrType::F128 => {
            // Use direct integer-to-x87 conversion to preserve full 64-bit
            // mantissa precision (x87 has 64-bit mantissa, unlike f64's 52-bit).
            if from_ty.is_unsigned() {
                IrConst::long_double_from_u128(v128 as u128)
            } else {
                IrConst::long_double_from_i128(v128)
            }
        }
        _ => return None,
    })
}

/// Convert a normalized i64 value to the target type.
fn fold_int_cast(val: i64, from_ty: IrType, to_ty: IrType) -> Option<IrConst> {
    Some(match to_ty {
        IrType::I8 | IrType::U8 => IrConst::I8(val as i8),
        IrType::I16 | IrType::U16 => IrConst::I16(val as i16),
        IrType::I32 => IrConst::I32(val as i32),
        // U32: keep as I64 with zero-extended value so that loading into
        // a 64-bit register preserves the unsigned 32-bit value. If we
        // stored as I32(val as i32), the constant load would sign-extend
        // (e.g. -13 -> 0xFFFFFFFFFFFFFFF3) which corrupts 64-bit unsigned
        // operations like divq.
        IrType::U32 => IrConst::I64(val as u32 as i64),
        IrType::I64 | IrType::U64 | IrType::Ptr => IrConst::I64(val),
        IrType::I128 | IrType::U128 => {
            if from_ty.is_unsigned() {
                IrConst::I128(val as u64 as u128 as i128)
            } else {
                IrConst::I128(val as i128)
            }
        }
        IrType::F32 => IrConst::F32(if from_ty.is_unsigned() { val as u64 as f32 } else { val as f32 }),
        IrType::F64 => IrConst::F64(if from_ty.is_unsigned() { val as u64 as f64 } else { val as f64 }),
        IrType::F128 => {
            // Use direct integer-to-x87 conversion for full 64-bit precision
            if from_ty.is_unsigned() {
                IrConst::long_double_from_u64(val as u64)
            } else {
                IrConst::long_double_from_i64(val)
            }
        }
        _ => return None,
    })
}

/// Invert a comparison operator (negate the condition).
fn invert_cmp_op(op: IrCmpOp) -> IrCmpOp {
    match op {
        IrCmpOp::Eq => IrCmpOp::Ne,
        IrCmpOp::Ne => IrCmpOp::Eq,
        IrCmpOp::Slt => IrCmpOp::Sge,
        IrCmpOp::Sle => IrCmpOp::Sgt,
        IrCmpOp::Sgt => IrCmpOp::Sle,
        IrCmpOp::Sge => IrCmpOp::Slt,
        IrCmpOp::Ult => IrCmpOp::Uge,
        IrCmpOp::Ule => IrCmpOp::Ugt,
        IrCmpOp::Ugt => IrCmpOp::Ule,
        IrCmpOp::Uge => IrCmpOp::Ult,
    }
}

/// Swap a comparison operator (reverse operand order).
fn swap_cmp_op(op: IrCmpOp) -> IrCmpOp {
    match op {
        IrCmpOp::Eq | IrCmpOp::Ne => op,
        IrCmpOp::Slt => IrCmpOp::Sgt,
        IrCmpOp::Sle => IrCmpOp::Sge,
        IrCmpOp::Sgt => IrCmpOp::Slt,
        IrCmpOp::Sge => IrCmpOp::Sle,
        IrCmpOp::Ult => IrCmpOp::Ugt,
        IrCmpOp::Ule => IrCmpOp::Uge,
        IrCmpOp::Ugt => IrCmpOp::Ult,
        IrCmpOp::Uge => IrCmpOp::Ule,
    }
}

/// Simplify a Cmp instruction.
///
/// Handles:
/// - Cmp(op, x, x) => constant (self-comparison: Eq/Sle/Sge/Ule/Uge => 1, others => 0)
/// - Cmp(Ne, cmp_result, 0) => Copy(cmp_result) when cmp_result is from another Cmp
///   (a Cmp already produces 0/1, so comparing != 0 is a no-op)
/// - Cmp(Eq, cmp_result, 0) => Cmp(inverted_op, orig_lhs, orig_rhs)
///   (negating a boolean comparison result)
/// - Cmp(Ne, cmp_result, 1) => Cmp(inverted_op, orig_lhs, orig_rhs)
///   (checking if comparison result is not-true = inverted)
/// - Cmp(Eq, cmp_result, 1) => Copy(cmp_result) when cmp_result is from another Cmp
///   (comparing a 0/1 value against 1 is the same as the original comparison)
fn simplify_cmp(
    dest: Value,
    op: IrCmpOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    cmp_defs: &[Option<CmpDef>],
    is_boolean: &[bool],
) -> Option<Instruction> {
    // Self-comparison: Cmp(op, x, x) => constant
    if same_value_operands(lhs, rhs) && ty.is_integer() {
        let result: i8 = match op {
            // x == x, x <= x, x >= x are always true
            IrCmpOp::Eq | IrCmpOp::Sle | IrCmpOp::Sge | IrCmpOp::Ule | IrCmpOp::Uge => 1,
            // x != x, x < x, x > x are always false
            IrCmpOp::Ne | IrCmpOp::Slt | IrCmpOp::Sgt | IrCmpOp::Ult | IrCmpOp::Ugt => 0,
        };
        return Some(Instruction::Copy {
            dest,
            src: Operand::Const(IrConst::I8(result)),
        });
    }

    // Canonicalize to (value, const) form by swapping the comparison operator.
    // This eliminates the need for separate swapped/non-swapped handling below.
    let (val, cval, effective_op) = match (lhs, rhs) {
        (Operand::Value(v), Operand::Const(c)) => (v, c, op),
        (Operand::Const(c), Operand::Value(v)) => (v, c, swap_cmp_op(op)),
        _ => return None,
    };

    let const_i64 = cval.to_i64();
    let is_zero_const = const_i64 == Some(0);
    let is_one_const = const_i64 == Some(1);

    // Check if the value is known boolean (0 or 1). This includes:
    // - Values defined by Cmp instructions
    // - Values from And/Or/Xor of boolean values
    let idx = val.0 as usize;
    let val_is_boolean = idx < is_boolean.len() && is_boolean[idx];

    // If the value is from a Cmp, we can do precise inversion
    let inner_cmp = if let Some(Some(cdef)) = cmp_defs.get(idx) { Some(cdef) } else { None };

    if val_is_boolean {
        match effective_op {
            IrCmpOp::Eq => {
                if is_zero_const {
                    // Cmp(Eq, boolean_val, 0) => logical NOT of boolean_val
                    // If from a Cmp, we can invert the comparison directly
                    if let Some(ic) = inner_cmp {
                        // For float ordered comparisons (Slt, Sle, Sgt, Sge),
                        // inversion is NOT valid because of NaN: !(a <= b) is
                        // NOT the same as (a > b) when either operand is NaN.
                        // !(NaN <= x) should be true (since NaN <= x is false),
                        // but (NaN > x) is also false. Only Eq<->Ne inversion
                        // is safe for floats since both handle NaN consistently.
                        let is_float_ordered = ic.ty.is_float()
                            && !matches!(ic.op, IrCmpOp::Eq | IrCmpOp::Ne);
                        if !is_float_ordered {
                            return Some(Instruction::Cmp {
                                dest,
                                op: invert_cmp_op(ic.op),
                                lhs: ic.lhs,
                                rhs: ic.rhs,
                                ty: ic.ty,
                            });
                        }
                    }
                    // For non-Cmp boolean values (e.g., And/Or of booleans),
                    // Cmp(Eq, val, 0) = Xor(val, 1) - but we can't easily
                    // express this without creating new instructions, so skip.
                }
                if is_one_const {
                    // Cmp(Eq, boolean_val, 1) => Copy(boolean_val)
                    // For a boolean value, "== 1" is identity
                    return Some(Instruction::Copy {
                        dest,
                        src: Operand::Value(*val),
                    });
                }
            }
            IrCmpOp::Ne => {
                if is_zero_const {
                    // Cmp(Ne, boolean_val, 0) => Copy(boolean_val)
                    // For a boolean value, "!= 0" is identity
                    return Some(Instruction::Copy {
                        dest,
                        src: Operand::Value(*val),
                    });
                }
                if is_one_const {
                    // Cmp(Ne, boolean_val, 1) => logical NOT
                    if let Some(ic) = inner_cmp {
                        // Skip inversion for float ordered comparisons (NaN safety).
                        // See the Eq/is_zero_const case above for the full explanation.
                        let is_float_ordered = ic.ty.is_float()
                            && !matches!(ic.op, IrCmpOp::Eq | IrCmpOp::Ne);
                        if !is_float_ordered {
                            return Some(Instruction::Cmp {
                                dest,
                                op: invert_cmp_op(ic.op),
                                lhs: ic.lhs,
                                rhs: ic.rhs,
                                ty: ic.ty,
                            });
                        }
                    }
                }
            }
            _ => {
                // For ordered comparisons (Slt, Sgt, etc.) with a boolean value:
                // Boolean values are 0 or 1, so we can simplify some patterns.
                // However, these are rare enough that we skip them for now.
            }
        }
    }

    // Unsigned comparisons with zero (already canonicalized to value op const form).
    if is_zero_const {
        match effective_op {
            // Cmp(Ult, x, 0) is always false (nothing is less than 0 unsigned)
            IrCmpOp::Ult => {
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::I8(0)),
                });
            }
            // Cmp(Uge, x, 0) is always true (everything is >= 0 unsigned)
            IrCmpOp::Uge => {
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::I8(1)),
                });
            }
            // Cmp(Ule, x, 0) => Cmp(Eq, x, 0) (unsigned x <= 0 means x == 0)
            IrCmpOp::Ule => {
                return Some(Instruction::Cmp {
                    dest, op: IrCmpOp::Eq,
                    lhs: Operand::Value(*val),
                    rhs: Operand::Const(*cval),
                    ty,
                });
            }
            // Cmp(Ugt, x, 0) => Cmp(Ne, x, 0) (unsigned x > 0 means x != 0)
            IrCmpOp::Ugt => {
                return Some(Instruction::Cmp {
                    dest, op: IrCmpOp::Ne,
                    lhs: Operand::Value(*val),
                    rhs: Operand::Const(*cval),
                    ty,
                });
            }
            _ => {}
        }
    }

    // Operand canonicalization: if the constant is on the LHS, rewrite to place it
    // on the RHS with the comparison operator swapped. This enables the backend to
    // use immediate-form compare instructions (e.g., `cmp $imm, %reg` or
    // `test %reg, %reg` for zero) instead of loading the constant into a register.
    // Only rewrite when the instruction actually has the constant on the LHS.
    if matches!(lhs, Operand::Const(_)) && matches!(rhs, Operand::Value(_)) {
        let swapped_op = swap_cmp_op(op);
        return Some(Instruction::Cmp {
            dest,
            op: swapped_op,
            lhs: *rhs,
            rhs: *lhs,
            ty,
        });
    }

    None
}

/// Simplify a GetElementPtr instruction.
///
/// - GEP with constant zero offset => Copy of base pointer
/// - GEP chain: GEP(GEP(base, c1), c2) => GEP(base, c1 + c2) when both offsets are constants
fn simplify_gep(
    dest: Value,
    base: Value,
    offset: &Operand,
    ty: IrType,
    gep_defs: &[Option<GepDef>],
) -> Option<Instruction> {
    if is_zero(offset) {
        return Some(Instruction::Copy { dest, src: Operand::Value(base) });
    }

    // GEP chain folding: GEP(GEP(inner_base, c1), c2) => GEP(inner_base, c1 + c2)
    // Only safe when both offsets are constants (we can compute the sum at compile time).
    if let Operand::Const(outer_c) = offset {
        if let Some(outer_val) = outer_c.to_i64() {
            let base_idx = base.0 as usize;
            if let Some(Some(inner_gep)) = gep_defs.get(base_idx) {
                if let Operand::Const(inner_c) = &inner_gep.offset {
                    if let Some(inner_val) = inner_c.to_i64() {
                        let combined = inner_val.wrapping_add(outer_val);
                        if combined == 0 {
                            // Combined offset is zero => just copy the original base
                            return Some(Instruction::Copy {
                                dest,
                                src: Operand::Value(inner_gep.base),
                            });
                        }
                        return Some(Instruction::GetElementPtr {
                            dest,
                            base: inner_gep.base,
                            offset: Operand::Const(IrConst::I64(combined)),
                            ty,
                        });
                    }
                }
            }
        }
    }

    None
}

/// Return the log2 of a positive i64 value if it is a power of 2, or None otherwise.
fn const_power_of_two(op: &Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => {
            let val = c.to_i64()?;
            if val > 0 && (val & (val - 1)) == 0 {
                Some(val.trailing_zeros())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to strength-reduce a multiply by a power of 2 on either side.
/// Returns the shift instruction if successful.
fn try_mul_power_of_two(dest: Value, val: &Operand, pow2: &Operand, ty: IrType) -> Option<Instruction> {
    let shift = const_power_of_two(pow2)?;
    if shift == 1 {
        // x * 2 => x + x
        Some(Instruction::BinOp {
            dest,
            op: IrBinOp::Add,
            lhs: *val,
            rhs: *val,
            ty,
        })
    } else {
        // x * 2^k => x << k
        Some(Instruction::BinOp {
            dest,
            op: IrBinOp::Shl,
            lhs: *val,
            rhs: Operand::Const(IrConst::from_i64(shift as i64, ty)),
            ty,
        })
    }
}

fn simplify_binop(
    dest: Value,
    op: IrBinOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    binop_defs: &[Option<BinOpDef>],
    neg_defs: &[Option<NegDef>],
) -> Option<Instruction> {
    let lhs_zero = is_zero(lhs);
    let rhs_zero = is_zero(rhs);
    let lhs_one = is_one(lhs);
    let rhs_one = is_one(rhs);
    let same_value = same_value_operands(lhs, rhs);
    let is_float = ty.is_float();

    match op {
        IrBinOp::Add => {
            // For floats, x + 0 => x is invalid because -0.0 + 0.0 = +0.0, not -0.0.
            // IEEE 754: the sign of a sum differs from at most one addend.
            if !is_float {
                if rhs_zero {
                    // x + 0 => x (integers only)
                    return Some(Instruction::Copy { dest, src: *lhs });
                }
                if lhs_zero {
                    // 0 + x => x (integers only)
                    return Some(Instruction::Copy { dest, src: *rhs });
                }
                // Reassociation: (x + C1) + C2 => x + (C1 + C2)
                if let Some(inst) = try_reassociate_add(dest, lhs, rhs, ty, binop_defs) {
                    return Some(inst);
                }
            }
        }
        IrBinOp::Sub => {
            if rhs_zero && (!is_float || is_positive_zero(rhs)) {
                // x - 0 => x (for floats, only valid when rhs is +0.0; -0.0 - (-0.0) = +0.0)
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            if !is_float && same_value {
                // x - x => 0 (integers only; for floats, Inf - Inf = NaN)
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::zero(ty)),
                });
            }
            if !is_float {
                // Reassociation: (x + C1) - C2 => x + (C1 - C2)
                // Also: (x - C1) - C2 => x - (C1 + C2)
                if let Some(inst) = try_reassociate_sub(dest, lhs, rhs, ty, binop_defs) {
                    return Some(inst);
                }
                // x - (neg y) => x + y (eliminate negation)
                if let Some(neg_src) = get_neg_def(rhs, neg_defs) {
                    return Some(Instruction::BinOp {
                        dest, op: IrBinOp::Add, lhs: *lhs, rhs: neg_src, ty,
                    });
                }
            }
        }
        IrBinOp::Mul => {
            if !is_float && (rhs_zero || lhs_zero) {
                // x * 0 or 0 * x => 0 (integers only; for floats, NaN*0=NaN, -x*0=-0)
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::zero(ty)),
                });
            }
            if rhs_one {
                // x * 1 => x (valid for both int and float)
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            if lhs_one {
                // 1 * x => x (valid for both int and float)
                return Some(Instruction::Copy { dest, src: *rhs });
            }
            // Strength reduction: multiply by power-of-2 => shift left (integers only)
            if ty.is_integer() || ty == IrType::Ptr {
                if let Some(inst) = try_mul_power_of_two(dest, lhs, rhs, ty) {
                    return Some(inst);
                }
                if let Some(inst) = try_mul_power_of_two(dest, rhs, lhs, ty) {
                    return Some(inst);
                }
            }
            // x * (-1) => neg(x), (-1) * x => neg(x) (integers only)
            // A single neg instruction is cheaper than imul by -1.
            if !is_float {
                if is_neg_one(rhs, ty) {
                    return Some(Instruction::UnaryOp {
                        dest, op: IrUnaryOp::Neg, src: *lhs, ty,
                    });
                }
                if is_neg_one(lhs, ty) {
                    return Some(Instruction::UnaryOp {
                        dest, op: IrUnaryOp::Neg, src: *rhs, ty,
                    });
                }
            }
            // Reassociation: (x * C1) * C2 => x * (C1 * C2) (integers only)
            if !is_float {
                if let Some(inst) = try_reassociate_mul(dest, lhs, rhs, ty, binop_defs) {
                    return Some(inst);
                }
            }
        }
        IrBinOp::SDiv | IrBinOp::UDiv => {
            if rhs_one {
                // x / 1 => x (valid for both int and float)
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            if !is_float && same_value {
                // x / x => 1 (integers only; for floats, 0/0=NaN, Inf/Inf=NaN)
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::one(ty)),
                });
            }
            // Strength reduction: unsigned divide by power-of-2 => logical shift right
            // Note: We check the opcode (UDiv) for unsigned semantics, not the type.
            // On 64-bit targets, C integer promotion widens unsigned int ops to I64,
            // but UDiv still has unsigned semantics regardless of type signedness.
            if op == IrBinOp::UDiv && (ty.is_integer() || ty == IrType::Ptr) {
                if let Some(shift) = const_power_of_two(rhs) {
                    return Some(Instruction::BinOp {
                        dest,
                        op: IrBinOp::LShr,
                        lhs: *lhs,
                        rhs: Operand::Const(IrConst::from_i64(shift as i64, ty)),
                        ty,
                    });
                }
            }
        }
        IrBinOp::SRem | IrBinOp::URem => {
            if rhs_one || same_value {
                // x % 1 => 0, x % x => 0
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::zero(ty)),
                });
            }
            // Strength reduction: unsigned rem by power-of-2 => bitwise AND with mask
            // Note: We check the opcode (URem) for unsigned semantics, not the type.
            // On 64-bit targets, C integer promotion widens unsigned int ops to I64,
            // but URem still has unsigned semantics regardless of type signedness.
            if op == IrBinOp::URem && (ty.is_integer() || ty == IrType::Ptr) {
                if let Some(shift) = const_power_of_two(rhs) {
                    // x % 2^k => x & (2^k - 1)
                    let mask = (1i64 << shift) - 1;
                    return Some(Instruction::BinOp {
                        dest,
                        op: IrBinOp::And,
                        lhs: *lhs,
                        rhs: Operand::Const(IrConst::from_i64(mask, ty)),
                        ty,
                    });
                }
            }
        }
        IrBinOp::And => {
            if rhs_zero || lhs_zero {
                // x & 0 => 0
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::zero(ty)),
                });
            }
            if is_all_ones(rhs, ty) {
                // x & all_ones => x (identity element for AND)
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            if is_all_ones(lhs, ty) {
                // all_ones & x => x
                return Some(Instruction::Copy { dest, src: *rhs });
            }
            if same_value {
                // x & x => x
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            // Reassociation: (x & C1) & C2 => x & (C1 & C2)
            if let Some(inst) = try_reassociate_bitwise(dest, IrBinOp::And, lhs, rhs, ty, binop_defs) {
                return Some(inst);
            }
        }
        IrBinOp::Or => {
            if rhs_zero {
                // x | 0 => x
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            if lhs_zero {
                // 0 | x => x
                return Some(Instruction::Copy { dest, src: *rhs });
            }
            if is_all_ones(rhs, ty) || is_all_ones(lhs, ty) {
                // x | all_ones => all_ones (annihilator for OR)
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(all_ones_const(ty)),
                });
            }
            if same_value {
                // x | x => x
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            // Reassociation: (x | C1) | C2 => x | (C1 | C2)
            if let Some(inst) = try_reassociate_bitwise(dest, IrBinOp::Or, lhs, rhs, ty, binop_defs) {
                return Some(inst);
            }
        }
        IrBinOp::Xor => {
            if rhs_zero {
                // x ^ 0 => x
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            if lhs_zero {
                // 0 ^ x => x
                return Some(Instruction::Copy { dest, src: *rhs });
            }
            if same_value {
                // x ^ x => 0
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::zero(ty)),
                });
            }
            // Reassociation: (x ^ C1) ^ C2 => x ^ (C1 ^ C2)
            if let Some(inst) = try_reassociate_bitwise(dest, IrBinOp::Xor, lhs, rhs, ty, binop_defs) {
                return Some(inst);
            }
        }
        IrBinOp::Shl | IrBinOp::AShr | IrBinOp::LShr => {
            if rhs_zero {
                // x << 0 or x >> 0 => x
                return Some(Instruction::Copy { dest, src: *lhs });
            }
            if lhs_zero {
                // 0 << x or 0 >> x => 0
                return Some(Instruction::Copy {
                    dest,
                    src: Operand::Const(IrConst::zero(ty)),
                });
            }
            // Reassociation: (x << C1) << C2 => x << (C1 + C2)
            // Also for >>: (x >> C1) >> C2 => x >> (C1 + C2)
            if let Some(inst) = try_reassociate_shift(dest, op, lhs, rhs, ty, binop_defs) {
                return Some(inst);
            }
        }
    }

    // Commutative operand canonicalization: for commutative operations
    // (Add, Mul, And, Or, Xor), place the constant on the RHS so the backend
    // can use immediate-form instructions (e.g., `addq $imm, %reg` instead of
    // loading the constant into a register first).
    if op.is_commutative() && matches!(lhs, Operand::Const(_)) && matches!(rhs, Operand::Value(_)) {
        return Some(Instruction::BinOp {
            dest, op, lhs: *rhs, rhs: *lhs, ty,
        });
    }

    None
}

/// Check if an operand is the all-ones bit pattern for the given integer type.
/// All-ones is the identity element for AND and the annihilator for OR.
/// Examples: 0xFF for I8/U8, 0xFFFFFFFF for I32/U32, -1 for I64/U64.
fn is_all_ones(op: &Operand, ty: IrType) -> bool {
    if !ty.is_integer() {
        return false;
    }
    match op {
        Operand::Const(c) => {
            // For 128-bit types, check the full 128-bit value directly.
            // We must NOT use to_i64() for I128 constants because it truncates
            // the upper 64 bits, causing false positives (e.g., I128(0xFFFFFFFFFFFFFFFF)
            // would truncate to i64 -1, which would incorrectly match "all ones" for U128).
            if ty.is_128bit() {
                if let IrConst::I128(v) = c {
                    *v == -1i128
                } else {
                    false
                }
            } else {
                match c.to_i64() {
                    Some(val) => {
                        // After truncation to the type's width, all-ones is -1.
                        ty.truncate_i64(val) == ty.truncate_i64(-1)
                    }
                    None => false,
                }
            }
        }
        _ => false,
    }
}

/// Create the all-ones constant for a given integer type.
fn all_ones_const(ty: IrType) -> IrConst {
    if ty.is_128bit() {
        IrConst::I128(-1i128)
    } else {
        IrConst::from_i64(-1, ty)
    }
}

/// Check if an operand is zero (including both +0.0 and -0.0 for floats).
fn is_zero(op: &Operand) -> bool {
    matches!(op, Operand::Const(c) if c.is_zero())
}

/// Check if a float operand is positive zero (+0.0), using the bit pattern.
/// Returns false for -0.0, non-zero values, and non-float types.
fn is_positive_zero(op: &Operand) -> bool {
    match op {
        Operand::Const(IrConst::F32(v)) => *v == 0.0 && !v.is_sign_negative(),
        Operand::Const(IrConst::F64(v)) => *v == 0.0 && !v.is_sign_negative(),
        Operand::Const(IrConst::LongDouble(v, _)) => *v == 0.0 && !v.is_sign_negative(),
        _ => false,
    }
}

/// Check if an operand is one.
fn is_one(op: &Operand) -> bool {
    matches!(op, Operand::Const(c) if c.is_one())
}

/// Check if an operand is -1 for the given integer type.
/// This is the two's complement representation: all bits set.
/// Used for x * (-1) => neg(x) optimization.
fn is_neg_one(op: &Operand, ty: IrType) -> bool {
    if !ty.is_integer() {
        return false;
    }
    match op {
        Operand::Const(c) => {
            // For 128-bit types, check the full 128-bit value directly to avoid
            // truncation through to_i64() (same issue as is_all_ones).
            if ty.is_128bit() {
                if let IrConst::I128(v) = c {
                    *v == -1i128
                } else {
                    false
                }
            } else {
                match c.to_i64() {
                    Some(val) => ty.truncate_i64(val) == ty.truncate_i64(-1),
                    None => false,
                }
            }
        }
        _ => false,
    }
}

/// Look up a NegDef for a Value operand.
/// Returns the source operand of the negation if the operand is defined by UnaryOp::Neg.
fn get_neg_def(op: &Operand, neg_defs: &[Option<NegDef>]) -> Option<Operand> {
    if let Operand::Value(v) = op {
        let idx = v.0 as usize;
        if let Some(Some(def)) = neg_defs.get(idx) {
            return Some(def.src);
        }
    }
    None
}

/// Check if two operands refer to the same value.
fn same_value_operands(lhs: &Operand, rhs: &Operand) -> bool {
    match (lhs, rhs) {
        (Operand::Value(a), Operand::Value(b)) => a.0 == b.0,
        _ => false,
    }
}

/// Check if two operands are identical (same value or same constant).
fn same_operand(a: &Operand, b: &Operand) -> bool {
    match (a, b) {
        (Operand::Value(va), Operand::Value(vb)) => va.0 == vb.0,
        (Operand::Const(ca), Operand::Const(cb)) => ca.to_hash_key() == cb.to_hash_key(),
        _ => false,
    }
}

/// Look up a BinOpDef for a Value operand.
fn get_binop_def<'a>(op: &Operand, binop_defs: &'a [Option<BinOpDef>]) -> Option<&'a BinOpDef> {
    if let Operand::Value(v) = op {
        let idx = v.0 as usize;
        if let Some(Some(def)) = binop_defs.get(idx) {
            return Some(def);
        }
    }
    None
}

/// Combine two i64 constants with wrapping addition, truncated to the type's width.
fn combine_add_consts(c1: i64, c2: i64, ty: IrType) -> i64 {
    ty.truncate_i64(c1.wrapping_add(c2))
}

/// Try reassociating addition: (x + C1) + C2 => x + (C1 + C2)
/// Also handles: C2 + (x + C1) => x + (C1 + C2)
fn try_reassociate_add(
    dest: Value,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    binop_defs: &[Option<BinOpDef>],
) -> Option<Instruction> {
    // Skip I128/U128: to_i64() truncates, which would silently corrupt 128-bit constants.
    if matches!(ty, IrType::I128 | IrType::U128) { return None; }
    // Pattern: (x + C1) + C2 or (x - C1) + C2
    if let Operand::Const(c2) = rhs {
        let c2_val = c2.to_i64()?;
        if let Some(def) = get_binop_def(lhs, binop_defs) {
            if def.ty == ty {
                match def.op {
                    IrBinOp::Add => {
                        // (x + C1) + C2 => x + (C1 + C2)
                        if let Operand::Const(c1) = &def.rhs {
                            let c1_val = c1.to_i64()?;
                            let combined = combine_add_consts(c1_val, c2_val, ty);
                            if combined == 0 {
                                return Some(Instruction::Copy { dest, src: def.lhs });
                            }
                            return Some(Instruction::BinOp {
                                dest, op: IrBinOp::Add,
                                lhs: def.lhs,
                                rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                                ty,
                            });
                        }
                        // (C1 + x) + C2 => x + (C1 + C2)
                        if let Operand::Const(c1) = &def.lhs {
                            let c1_val = c1.to_i64()?;
                            let combined = combine_add_consts(c1_val, c2_val, ty);
                            if combined == 0 {
                                return Some(Instruction::Copy { dest, src: def.rhs });
                            }
                            return Some(Instruction::BinOp {
                                dest, op: IrBinOp::Add,
                                lhs: def.rhs,
                                rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                                ty,
                            });
                        }
                    }
                    IrBinOp::Sub => {
                        // (x - C1) + C2 => x + (C2 - C1)
                        if let Operand::Const(c1) = &def.rhs {
                            let c1_val = c1.to_i64()?;
                            let combined = ty.truncate_i64(c2_val.wrapping_sub(c1_val));
                            if combined == 0 {
                                return Some(Instruction::Copy { dest, src: def.lhs });
                            }
                            return Some(Instruction::BinOp {
                                dest, op: IrBinOp::Add,
                                lhs: def.lhs,
                                rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                                ty,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // Pattern: C2 + (x + C1) -- commuted case
    if let Operand::Const(c2) = lhs {
        let c2_val = c2.to_i64()?;
        if let Some(def) = get_binop_def(rhs, binop_defs) {
            if def.ty == ty && def.op == IrBinOp::Add {
                if let Operand::Const(c1) = &def.rhs {
                    let c1_val = c1.to_i64()?;
                    let combined = combine_add_consts(c1_val, c2_val, ty);
                    if combined == 0 {
                        return Some(Instruction::Copy { dest, src: def.lhs });
                    }
                    return Some(Instruction::BinOp {
                        dest, op: IrBinOp::Add,
                        lhs: def.lhs,
                        rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                        ty,
                    });
                }
                if let Operand::Const(c1) = &def.lhs {
                    let c1_val = c1.to_i64()?;
                    let combined = combine_add_consts(c1_val, c2_val, ty);
                    if combined == 0 {
                        return Some(Instruction::Copy { dest, src: def.rhs });
                    }
                    return Some(Instruction::BinOp {
                        dest, op: IrBinOp::Add,
                        lhs: def.rhs,
                        rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                        ty,
                    });
                }
            }
        }
    }
    None
}

/// Try reassociating subtraction:
/// - (x + C1) - C2 => x + (C1 - C2)
/// - (x - C1) - C2 => x - (C1 + C2)
fn try_reassociate_sub(
    dest: Value,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    binop_defs: &[Option<BinOpDef>],
) -> Option<Instruction> {
    // Skip I128/U128: to_i64() truncates, which would silently corrupt 128-bit constants.
    if matches!(ty, IrType::I128 | IrType::U128) { return None; }
    let c2_val = match rhs {
        Operand::Const(c) => c.to_i64()?,
        _ => return None,
    };
    let def = get_binop_def(lhs, binop_defs)?;
    if def.ty != ty {
        return None;
    }
    match def.op {
        IrBinOp::Add => {
            // (x + C1) - C2 => x + (C1 - C2)
            if let Operand::Const(c1) = &def.rhs {
                let c1_val = c1.to_i64()?;
                let combined = ty.truncate_i64(c1_val.wrapping_sub(c2_val));
                if combined == 0 {
                    return Some(Instruction::Copy { dest, src: def.lhs });
                }
                return Some(Instruction::BinOp {
                    dest, op: IrBinOp::Add,
                    lhs: def.lhs,
                    rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                    ty,
                });
            }
            // (C1 + x) - C2 => x + (C1 - C2)
            if let Operand::Const(c1) = &def.lhs {
                let c1_val = c1.to_i64()?;
                let combined = ty.truncate_i64(c1_val.wrapping_sub(c2_val));
                if combined == 0 {
                    return Some(Instruction::Copy { dest, src: def.rhs });
                }
                return Some(Instruction::BinOp {
                    dest, op: IrBinOp::Add,
                    lhs: def.rhs,
                    rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                    ty,
                });
            }
        }
        IrBinOp::Sub => {
            // (x - C1) - C2 => x - (C1 + C2)
            if let Operand::Const(c1) = &def.rhs {
                let c1_val = c1.to_i64()?;
                let combined = combine_add_consts(c1_val, c2_val, ty);
                if combined == 0 {
                    return Some(Instruction::Copy { dest, src: def.lhs });
                }
                return Some(Instruction::BinOp {
                    dest, op: IrBinOp::Sub,
                    lhs: def.lhs,
                    rhs: Operand::Const(IrConst::from_i64(combined, ty)),
                    ty,
                });
            }
        }
        _ => {}
    }
    None
}

/// Try reassociating bitwise operations: (x & C1) & C2 => x & (C1 & C2)
/// Works for And, Or, and Xor (all associative and commutative).
fn try_reassociate_bitwise(
    dest: Value,
    op: IrBinOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    binop_defs: &[Option<BinOpDef>],
) -> Option<Instruction> {
    // Skip I128/U128: to_i64() truncates, which would silently corrupt 128-bit constants.
    if matches!(ty, IrType::I128 | IrType::U128) { return None; }
    // Pattern: (x op C1) op C2
    let c2_val = match rhs {
        Operand::Const(c) => c.to_i64()?,
        _ => return None,
    };
    let def = get_binop_def(lhs, binop_defs)?;
    if def.op != op || def.ty != ty {
        return None;
    }
    // (x op C1) op C2 => x op (C1 op C2)
    let (non_const_operand, c1_val) = if let Operand::Const(c1) = &def.rhs {
        (def.lhs, c1.to_i64()?)
    } else if let Operand::Const(c1) = &def.lhs {
        (def.rhs, c1.to_i64()?)
    } else {
        return None;
    };

    let combined = match op {
        IrBinOp::And => c1_val & c2_val,
        IrBinOp::Or => c1_val | c2_val,
        IrBinOp::Xor => c1_val ^ c2_val,
        _ => return None,
    };
    let combined = ty.truncate_i64(combined);

    // Check if combined result is identity and can eliminate the instruction
    let is_identity = match op {
        IrBinOp::And => is_all_ones(&Operand::Const(IrConst::from_i64(combined, ty)), ty),
        IrBinOp::Or | IrBinOp::Xor => combined == 0,
        _ => false,
    };
    if is_identity {
        return Some(Instruction::Copy { dest, src: non_const_operand });
    }

    // Check if combined result is annihilator
    let is_annihilator = match op {
        IrBinOp::And => combined == 0,
        IrBinOp::Or => is_all_ones(&Operand::Const(IrConst::from_i64(combined, ty)), ty),
        _ => false,
    };
    if is_annihilator {
        return Some(Instruction::Copy {
            dest,
            src: Operand::Const(IrConst::from_i64(combined, ty)),
        });
    }

    Some(Instruction::BinOp {
        dest, op,
        lhs: non_const_operand,
        rhs: Operand::Const(IrConst::from_i64(combined, ty)),
        ty,
    })
}

/// Try reassociating shift operations: (x << C1) << C2 => x << (C1 + C2)
/// Works for Shl, AShr, and LShr (but only when same shift direction).
fn try_reassociate_shift(
    dest: Value,
    op: IrBinOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    binop_defs: &[Option<BinOpDef>],
) -> Option<Instruction> {
    // Skip I128/U128: to_i64() truncates, which would silently corrupt 128-bit constants.
    if matches!(ty, IrType::I128 | IrType::U128) { return None; }
    let c2_val = match rhs {
        Operand::Const(c) => c.to_i64()?,
        _ => return None,
    };
    if c2_val < 0 { return None; }
    let def = get_binop_def(lhs, binop_defs)?;
    // Must be the same shift operation
    if def.op != op || def.ty != ty {
        return None;
    }
    // (x shift C1) shift C2 => x shift (C1 + C2)
    let c1_val = match &def.rhs {
        Operand::Const(c) => c.to_i64()?,
        _ => return None,
    };
    if c1_val < 0 { return None; }

    let combined = c1_val + c2_val;
    let bit_width = (ty.size() * 8) as i64;

    // If combined shift exceeds bit width, result is 0 (for Shl/LShr)
    // or sign-extended (for AShr). Conservatively only fold to 0 for Shl/LShr.
    if combined >= bit_width {
        if matches!(op, IrBinOp::Shl | IrBinOp::LShr) {
            return Some(Instruction::Copy {
                dest,
                src: Operand::Const(IrConst::zero(ty)),
            });
        }
        // For AShr, don't fold -- the result depends on the sign bit
        return None;
    }

    Some(Instruction::BinOp {
        dest, op,
        lhs: def.lhs,
        rhs: Operand::Const(IrConst::from_i64(combined, ty)),
        ty,
    })
}

/// Try reassociating multiplication: (x * C1) * C2 => x * (C1 * C2)
/// Also handles: C2 * (x * C1) => x * (C1 * C2) (commuted)
/// Integer multiplication is associative, so folding two constant factors into
/// one eliminates an instruction.
fn try_reassociate_mul(
    dest: Value,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    binop_defs: &[Option<BinOpDef>],
) -> Option<Instruction> {
    // Skip I128/U128: to_i64() truncates, which would silently corrupt 128-bit constants.
    if matches!(ty, IrType::I128 | IrType::U128) { return None; }

    // Helper: try pattern (expr * C1) * C2 where expr is on one side and C2 on the other.
    let try_pattern = |expr_side: &Operand, const_side: &Operand| -> Option<Instruction> {
        let c2_val = match const_side {
            Operand::Const(c) => c.to_i64()?,
            _ => return None,
        };
        let def = get_binop_def(expr_side, binop_defs)?;
        if def.op != IrBinOp::Mul || def.ty != ty {
            return None;
        }
        // Find the constant in the inner multiply: (x * C1) or (C1 * x)
        let (non_const, c1_val) = if let Operand::Const(c1) = &def.rhs {
            (def.lhs, c1.to_i64()?)
        } else if let Operand::Const(c1) = &def.lhs {
            (def.rhs, c1.to_i64()?)
        } else {
            return None;
        };
        let combined = ty.truncate_i64(c1_val.wrapping_mul(c2_val));
        if combined == 0 {
            return Some(Instruction::Copy {
                dest,
                src: Operand::Const(IrConst::zero(ty)),
            });
        }
        if combined == 1 || ty.truncate_i64(combined) == ty.truncate_i64(1) {
            return Some(Instruction::Copy { dest, src: non_const });
        }
        Some(Instruction::BinOp {
            dest, op: IrBinOp::Mul,
            lhs: non_const,
            rhs: Operand::Const(IrConst::from_i64(combined, ty)),
            ty,
        })
    };

    // Try both orderings: (x*C1)*C2 and C2*(x*C1)
    if let Some(inst) = try_pattern(lhs, rhs) {
        return Some(inst);
    }
    try_pattern(rhs, lhs)
}

/// Table of unary math functions that map directly to intrinsic instructions.
const UNARY_INTRINSICS: &[(&str, IntrinsicOp)] = &[
    ("sqrt", IntrinsicOp::SqrtF64),
    ("sqrtf", IntrinsicOp::SqrtF32),
    ("fabs", IntrinsicOp::FabsF64),
    ("fabsf", IntrinsicOp::FabsF32),
];

/// Create a float constant (F32 or F64) appropriate for the return type.
fn float_const(return_type: IrType, val: f64) -> IrConst {
    if return_type == IrType::F32 {
        IrConst::F32(val as f32)
    } else {
        IrConst::F64(val)
    }
}

/// Simplify calls to known math library functions.
///
/// Optimizations:
/// - pow(x, 2.0) / powf(x, 2.0f) => x * x  (avoids expensive libm call)
/// - pow(x, 0.0) => 1.0, pow(x, 1.0) => x
/// - pow(x, 0.5) => sqrt(x), powf(x, 0.5f) => sqrtf(x)
/// - pow(x, -1.0) => 1.0 / x
/// - sqrt(x) / sqrtf(x) => SqrtF64/SqrtF32 intrinsic (inline instruction)
/// - fabs(x) / fabsf(x) => FabsF64/FabsF32 intrinsic (inline instruction)
fn simplify_math_call(
    dest: Value,
    func: &str,
    args: &[Operand],
    return_type: IrType,
) -> Option<Instruction> {
    // Check unary intrinsic table first (sqrt, sqrtf, fabs, fabsf).
    for &(name, intrinsic_op) in UNARY_INTRINSICS {
        if func == name {
            if args.len() != 1 { return None; }
            return Some(Instruction::Intrinsic {
                dest: Some(dest),
                op: intrinsic_op,
                dest_ptr: None,
                args: args.to_vec(),
            });
        }
    }

    // pow/powf with constant exponent
    if func == "pow" || func == "powf" {
        if args.len() != 2 { return None; }
        let exp = match &args[1] {
            Operand::Const(IrConst::F64(v)) => *v,
            Operand::Const(IrConst::F32(v)) => *v as f64,
            Operand::Const(IrConst::LongDouble(v, _)) => *v,
            _ => return None,
        };
        let base = args[0];

        if exp == 0.0 {
            // pow(x, 0.0) => 1.0 (C11 7.12.7.4: pow(x,+-0) returns 1 for all x, even NaN)
            return Some(Instruction::Copy { dest, src: Operand::Const(float_const(return_type, 1.0)) });
        }
        if exp == 1.0 {
            // pow(x, 1.0) => x
            return Some(Instruction::Copy { dest, src: base });
        }
        if exp == 2.0 {
            // pow(x, 2.0) => x * x
            return Some(Instruction::BinOp { dest, op: IrBinOp::Mul, lhs: base, rhs: base, ty: return_type });
        }
        if exp == -1.0 {
            // pow(x, -1.0) => 1.0 / x
            return Some(Instruction::BinOp {
                dest,
                op: IrBinOp::SDiv, // float division
                lhs: Operand::Const(float_const(return_type, 1.0)),
                rhs: base,
                ty: return_type,
            });
        }
        if exp == 0.5 {
            // pow(x, 0.5) => sqrt(x)
            let sqrt_op = if return_type == IrType::F32 { IntrinsicOp::SqrtF32 } else { IntrinsicOp::SqrtF64 };
            return Some(Instruction::Intrinsic {
                dest: Some(dest),
                op: sqrt_op,
                dest_ptr: None,
                args: vec![base],
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::reexports::CallInfo;

    // === Test helpers ===

    /// Shorthand: try_simplify with empty def maps (no chain optimization context).
    fn simplify_default(inst: &Instruction) -> Option<Instruction> {
        try_simplify(inst, &[], &[], &[], &[], &[], &[])
    }

    /// Create a BinOp instruction with standard test values.
    fn binop(op: IrBinOp, lhs: Operand, rhs: Operand, ty: IrType) -> Instruction {
        Instruction::BinOp { dest: Value(0), op, lhs, rhs, ty }
    }

    /// Create a BinOp with Value(1) as lhs and the given const rhs.
    fn binop_val_const(op: IrBinOp, rhs: IrConst, ty: IrType) -> Instruction {
        binop(op, Operand::Value(Value(1)), Operand::Const(rhs), ty)
    }

    /// Create a BinOp with both operands being Value(1).
    fn binop_self(op: IrBinOp, ty: IrType) -> Instruction {
        Instruction::BinOp {
            dest: Value(2), op,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Value(Value(1)),
            ty,
        }
    }

    /// Assert result is a Copy of the given value id.
    fn assert_copy_value(result: &Instruction, expected_id: u32) {
        match result {
            Instruction::Copy { src: Operand::Value(v), .. } => assert_eq!(v.0, expected_id),
            _ => panic!("Expected Copy of Value({}), got {:?}", expected_id, result),
        }
    }

    /// Assert result is a Copy of the given constant.
    fn assert_copy_const_i32(result: &Instruction, expected: i32) {
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::I32(v)), .. } => assert_eq!(*v, expected),
            _ => panic!("Expected Copy with I32({}), got {:?}", expected, result),
        }
    }

    /// Assert result is a Copy of I8 constant (used for boolean results).
    fn assert_copy_bool(result: &Instruction, expected: i8) {
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::I8(v)), .. } => assert_eq!(*v, expected),
            _ => panic!("Expected Copy with I8({}), got {:?}", expected, result),
        }
    }

    fn make_call(func_name: &str, args: Vec<Operand>, return_type: IrType) -> Instruction {
        Instruction::Call {
            func: func_name.into(),
            info: CallInfo {
                dest: Some(Value(10)),
                args,
                arg_types: vec![],
                return_type,
                is_variadic: false,
                num_fixed_args: 0,
                struct_arg_sizes: vec![],
                struct_arg_aligns: vec![],
                struct_arg_classes: Vec::new(),
                struct_arg_riscv_float_classes: Vec::new(),
                is_sret: false,
                is_fastcall: false,
                ret_eightbyte_classes: Vec::new(),
            },
        }
    }

    // === BinOp identity tests ===

    #[test]
    fn test_add_zero() {
        let result = simplify_default(&binop_val_const(IrBinOp::Add, IrConst::I32(0), IrType::I32)).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_mul_zero() {
        let result = simplify_default(&binop_val_const(IrBinOp::Mul, IrConst::I32(0), IrType::I32)).unwrap();
        assert_copy_const_i32(&result, 0);
    }

    #[test]
    fn test_mul_one() {
        let result = simplify_default(&binop_val_const(IrBinOp::Mul, IrConst::I32(1), IrType::I32)).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_sub_self() {
        let result = simplify_default(&binop_self(IrBinOp::Sub, IrType::I32)).unwrap();
        assert_copy_const_i32(&result, 0);
    }

    #[test]
    fn test_xor_self() {
        let result = simplify_default(&binop_self(IrBinOp::Xor, IrType::I32)).unwrap();
        assert_copy_const_i32(&result, 0);
    }

    #[test]
    fn test_and_self() {
        let result = simplify_default(&binop_self(IrBinOp::And, IrType::I32)).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_no_simplify() {
        // x + y (non-trivial) should not simplify
        let inst = binop(IrBinOp::Add, Operand::Value(Value(0)), Operand::Value(Value(1)), IrType::I32);
        assert!(simplify_default(&inst).is_none());
    }

    // === Strength reduction tests ===

    #[test]
    fn test_mul_power_of_two_to_shift() {
        // x * 4 => x << 2
        let result = simplify_default(&binop_val_const(IrBinOp::Mul, IrConst::I64(4), IrType::I64)).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Shl, rhs: Operand::Const(IrConst::I64(2)), .. } => {}
            _ => panic!("Expected Shl by 2, got {:?}", result),
        }
    }

    #[test]
    fn test_mul_two_to_add() {
        // x * 2 => x + x
        let result = simplify_default(&binop_val_const(IrBinOp::Mul, IrConst::I64(2), IrType::I64)).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Add, lhs: Operand::Value(a), rhs: Operand::Value(b), .. } => {
                assert_eq!(a.0, 1);
                assert_eq!(b.0, 1);
            }
            _ => panic!("Expected Add x,x, got {:?}", result),
        }
    }

    #[test]
    fn test_mul_power_of_two_i32() {
        // x * 8 => x << 3 (I32)
        let result = simplify_default(&binop_val_const(IrBinOp::Mul, IrConst::I32(8), IrType::I32)).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Shl, rhs: Operand::Const(IrConst::I32(3)), .. } => {}
            _ => panic!("Expected Shl by 3, got {:?}", result),
        }
    }

    #[test]
    fn test_mul_non_power_of_two_no_change() {
        // x * 3 should NOT be simplified to shift
        assert!(simplify_default(&binop_val_const(IrBinOp::Mul, IrConst::I64(3), IrType::I64)).is_none());
    }

    #[test]
    fn test_mul_float_no_strength_reduction() {
        // x * 2.0 should NOT be simplified (float type)
        assert!(simplify_default(&binop_val_const(IrBinOp::Mul, IrConst::I64(2), IrType::F64)).is_none());
    }

    #[test]
    fn test_udiv_power_of_two() {
        // x /u 8 => x >> 3 (unsigned)
        let inst = binop_val_const(IrBinOp::UDiv, IrConst::I64(8), IrType::U64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::LShr, rhs: Operand::Const(IrConst::I64(3)), .. } => {}
            _ => panic!("Expected LShr by 3, got {:?}", result),
        }
    }

    #[test]
    fn test_urem_power_of_two() {
        // x %u 8 => x & 7 (unsigned)
        let inst = binop_val_const(IrBinOp::URem, IrConst::I64(8), IrType::U64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::And, rhs: Operand::Const(IrConst::I64(7)), .. } => {}
            _ => panic!("Expected And with 7, got {:?}", result),
        }
    }

    // === Cast tests ===

    #[test]
    fn test_cast_same_type() {
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Value(Value(1)),
            from_ty: IrType::I32,
            to_ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_cast_chain_widen_narrow() {
        // Cast(Cast(x, I32->I64), I64->I32) => Copy of x
        let mut defs: Vec<Option<CastDef>> = vec![None; 3];
        defs[1] = Some(CastDef {
            src: Operand::Value(Value(0)),
            from_ty: IrType::I32,
            to_ty: IrType::I64,
        });
        let inst = Instruction::Cast {
            dest: Value(2),
            src: Operand::Value(Value(1)),
            from_ty: IrType::I64,
            to_ty: IrType::I32,
        };
        let result = try_simplify(&inst, &defs, &[], &[], &[], &[], &[]).unwrap();
        assert_copy_value(&result, 0);
    }

    #[test]
    fn test_cast_const_fold() {
        // Cast const I32(42) to I64 => const I64(42)
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Const(IrConst::I32(42)),
            from_ty: IrType::I32,
            to_ty: IrType::I64,
        };
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::I64(42)), .. } => {}
            _ => panic!("Expected Copy with I64(42)"),
        }
    }

    #[test]
    fn test_cast_different_type_no_change() {
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Value(Value(1)),
            from_ty: IrType::I32,
            to_ty: IrType::I64,
        };
        assert!(simplify_default(&inst).is_none());
    }

    // === GEP tests ===

    #[test]
    fn test_gep_zero_offset() {
        let inst = Instruction::GetElementPtr {
            dest: Value(0),
            base: Value(1),
            offset: Operand::Const(IrConst::I64(0)),
            ty: IrType::Ptr,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_gep_nonzero_no_change() {
        let inst = Instruction::GetElementPtr {
            dest: Value(0),
            base: Value(1),
            offset: Operand::Const(IrConst::I64(4)),
            ty: IrType::Ptr,
        };
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_gep_chain_fold() {
        // GEP(GEP(base, 8), 4) => GEP(base, 12)
        let mut gep_defs: Vec<Option<GepDef>> = vec![None; 3];
        gep_defs[1] = Some(GepDef {
            base: Value(0),
            offset: Operand::Const(IrConst::I64(8)),
        });
        let inst = Instruction::GetElementPtr {
            dest: Value(2),
            base: Value(1),
            offset: Operand::Const(IrConst::I64(4)),
            ty: IrType::Ptr,
        };
        let result = try_simplify(&inst, &[], &gep_defs, &[], &[], &[], &[]).unwrap();
        match result {
            Instruction::GetElementPtr { base, offset: Operand::Const(IrConst::I64(12)), .. } => {
                assert_eq!(base.0, 0, "Should use original base");
            }
            _ => panic!("Expected GEP with combined offset 12, got {:?}", result),
        }
    }

    #[test]
    fn test_gep_chain_fold_to_zero() {
        // GEP(GEP(base, 4), -4) => Copy of base (offsets cancel)
        let mut gep_defs: Vec<Option<GepDef>> = vec![None; 3];
        gep_defs[1] = Some(GepDef {
            base: Value(0),
            offset: Operand::Const(IrConst::I64(4)),
        });
        let inst = Instruction::GetElementPtr {
            dest: Value(2),
            base: Value(1),
            offset: Operand::Const(IrConst::I64(-4)),
            ty: IrType::Ptr,
        };
        let result = try_simplify(&inst, &[], &gep_defs, &[], &[], &[], &[]).unwrap();
        assert_copy_value(&result, 0);
    }

    // === IEEE 754 float safety tests ===

    #[test]
    fn test_float_add_zero_not_simplified() {
        // -0.0 + 0.0 must NOT be simplified to -0.0 (result should be +0.0)
        let inst = binop(
            IrBinOp::Add,
            Operand::Const(IrConst::F64(-0.0)),
            Operand::Const(IrConst::F64(0.0)),
            IrType::F64,
        );
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_float_add_zero_lhs_not_simplified() {
        // 0.0 + (-0.0) must NOT simplify to -0.0
        let inst = binop(
            IrBinOp::Add,
            Operand::Const(IrConst::F64(0.0)),
            Operand::Const(IrConst::F64(-0.0)),
            IrType::F64,
        );
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_float_mul_zero_not_simplified() {
        // x * 0.0 must NOT simplify for floats (NaN*0=NaN, -5.0*0=-0.0)
        let inst = binop_val_const(IrBinOp::Mul, IrConst::F64(0.0), IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_float_sub_self_not_simplified() {
        // x - x must NOT simplify for floats (Inf - Inf = NaN)
        assert!(simplify_default(&binop_self(IrBinOp::Sub, IrType::F64)).is_none());
    }

    #[test]
    fn test_float_div_self_not_simplified() {
        // x / x must NOT simplify for floats (0/0=NaN, Inf/Inf=NaN)
        assert!(simplify_default(&binop_self(IrBinOp::SDiv, IrType::F64)).is_none());
    }

    #[test]
    fn test_float_sub_positive_zero_ok() {
        // x - (+0.0) is safe (subtracting +0 preserves value)
        let inst = binop_val_const(IrBinOp::Sub, IrConst::F64(0.0), IrType::F64);
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_float_sub_negative_zero_not_simplified() {
        // x - (-0.0) is NOT safe (equivalent to x + 0.0, changes -0.0 to +0.0)
        let inst = binop_val_const(IrBinOp::Sub, IrConst::F64(-0.0), IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_float_mul_one_not_simplified() {
        // x * F64(1.0) is not simplified because is_one() only matches integer 1.
        let inst = binop_val_const(IrBinOp::Mul, IrConst::F64(1.0), IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_float_div_one_not_simplified() {
        // x / F64(1.0) is not simplified because is_one() only matches integer 1.
        let inst = binop_val_const(IrBinOp::SDiv, IrConst::F64(1.0), IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    // === Math call simplification tests ===

    #[test]
    fn test_sqrt_to_intrinsic() {
        let inst = make_call("sqrt", vec![Operand::Value(Value(1))], IrType::F64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Intrinsic { op: IntrinsicOp::SqrtF64, args, .. } => {
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected SqrtF64 intrinsic, got {:?}", result),
        }
    }

    #[test]
    fn test_sqrtf_to_intrinsic() {
        let inst = make_call("sqrtf", vec![Operand::Value(Value(1))], IrType::F32);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Intrinsic { op: IntrinsicOp::SqrtF32, args, .. } => {
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected SqrtF32 intrinsic, got {:?}", result),
        }
    }

    #[test]
    fn test_fabs_to_intrinsic() {
        let inst = make_call("fabs", vec![Operand::Value(Value(1))], IrType::F64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Intrinsic { op: IntrinsicOp::FabsF64, args, .. } => {
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected FabsF64 intrinsic, got {:?}", result),
        }
    }

    #[test]
    fn test_fabsf_to_intrinsic() {
        let inst = make_call("fabsf", vec![Operand::Value(Value(1))], IrType::F32);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Intrinsic { op: IntrinsicOp::FabsF32, args, .. } => {
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected FabsF32 intrinsic, got {:?}", result),
        }
    }

    #[test]
    fn test_pow_x_zero() {
        let inst = make_call("pow", vec![
            Operand::Value(Value(1)),
            Operand::Const(IrConst::F64(0.0)),
        ], IrType::F64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::F64(v)), .. } => assert_eq!(v, 1.0),
            _ => panic!("Expected Copy with 1.0, got {:?}", result),
        }
    }

    #[test]
    fn test_pow_x_one() {
        let inst = make_call("pow", vec![
            Operand::Value(Value(1)),
            Operand::Const(IrConst::F64(1.0)),
        ], IrType::F64);
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_pow_x_two() {
        let inst = make_call("pow", vec![
            Operand::Value(Value(1)),
            Operand::Const(IrConst::F64(2.0)),
        ], IrType::F64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Mul, lhs: Operand::Value(a), rhs: Operand::Value(b), .. } => {
                assert_eq!(a.0, 1);
                assert_eq!(b.0, 1);
            }
            _ => panic!("Expected Mul x*x, got {:?}", result),
        }
    }

    #[test]
    fn test_pow_x_neg_one() {
        let inst = make_call("pow", vec![
            Operand::Value(Value(1)),
            Operand::Const(IrConst::F64(-1.0)),
        ], IrType::F64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::SDiv, lhs: Operand::Const(IrConst::F64(v)), .. } => {
                assert_eq!(v, 1.0);
            }
            _ => panic!("Expected 1.0/x, got {:?}", result),
        }
    }

    #[test]
    fn test_pow_x_half() {
        let inst = make_call("pow", vec![
            Operand::Value(Value(1)),
            Operand::Const(IrConst::F64(0.5)),
        ], IrType::F64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Intrinsic { op: IntrinsicOp::SqrtF64, .. } => {}
            _ => panic!("Expected SqrtF64 intrinsic, got {:?}", result),
        }
    }

    #[test]
    fn test_pow_non_special_exponent() {
        let inst = make_call("pow", vec![
            Operand::Value(Value(1)),
            Operand::Const(IrConst::F64(3.0)),
        ], IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_math_call_wrong_arg_count() {
        // sqrt with 0 args should not simplify
        let inst = make_call("sqrt", vec![], IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_unknown_func_no_simplify() {
        let inst = make_call("sin", vec![Operand::Value(Value(1))], IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_pow_variable_exponent_no_simplify() {
        let inst = make_call("pow", vec![
            Operand::Value(Value(1)),
            Operand::Value(Value(2)),
        ], IrType::F64);
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_powf_x_two_f32() {
        let inst = make_call("powf", vec![
            Operand::Value(Value(1)),
            Operand::Const(IrConst::F32(2.0)),
        ], IrType::F32);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Mul, ty: IrType::F32, .. } => {}
            _ => panic!("Expected F32 Mul, got {:?}", result),
        }
    }

    // === Cmp simplification tests ===

    #[test]
    fn test_cmp_self_eq() {
        // Cmp(Eq, x, x) => 1 (always true)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Eq,
            lhs: Operand::Value(Value(1)), rhs: Operand::Value(Value(1)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_bool(&result, 1);
    }

    #[test]
    fn test_cmp_self_ne() {
        // Cmp(Ne, x, x) => 0 (always false)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Ne,
            lhs: Operand::Value(Value(1)), rhs: Operand::Value(Value(1)),
            ty: IrType::I64,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_bool(&result, 0);
    }

    #[test]
    fn test_cmp_self_sle() {
        // Cmp(Sle, x, x) => 1 (always true)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Sle,
            lhs: Operand::Value(Value(1)), rhs: Operand::Value(Value(1)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_bool(&result, 1);
    }

    #[test]
    fn test_cmp_self_slt() {
        // Cmp(Slt, x, x) => 0 (always false)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Slt,
            lhs: Operand::Value(Value(1)), rhs: Operand::Value(Value(1)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_bool(&result, 0);
    }

    #[test]
    fn test_cmp_ne_cmp_result_zero() {
        // Cmp(Ne, cmp_result, 0) => Copy(cmp_result)
        let mut cmp_defs: Vec<Option<CmpDef>> = vec![None; 3];
        cmp_defs[1] = Some(CmpDef {
            op: IrCmpOp::Eq,
            lhs: Operand::Value(Value(10)),
            rhs: Operand::Value(Value(11)),
            ty: IrType::I64,
        });
        let mut booleans = vec![false; 3];
        booleans[1] = true;

        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Ne,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::I64,
        };
        let result = try_simplify(&inst, &[], &[], &cmp_defs, &[], &[], &booleans).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_cmp_eq_cmp_result_zero() {
        // Cmp(Eq, cmp_result, 0) => Cmp(inverted, orig_lhs, orig_rhs)
        let mut cmp_defs: Vec<Option<CmpDef>> = vec![None; 3];
        cmp_defs[1] = Some(CmpDef {
            op: IrCmpOp::Slt,
            lhs: Operand::Value(Value(10)),
            rhs: Operand::Value(Value(11)),
            ty: IrType::I64,
        });
        let mut booleans = vec![false; 3];
        booleans[1] = true;

        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Eq,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::I64,
        };
        let result = try_simplify(&inst, &[], &[], &cmp_defs, &[], &[], &booleans).unwrap();
        match result {
            Instruction::Cmp { op: IrCmpOp::Sge, lhs: Operand::Value(a), rhs: Operand::Value(b), .. } => {
                assert_eq!(a.0, 10);
                assert_eq!(b.0, 11);
            }
            _ => panic!("Expected Cmp(Sge, V10, V11), got {:?}", result),
        }
    }

    #[test]
    fn test_cmp_eq_cmp_result_one() {
        // Cmp(Eq, cmp_result, 1) => Copy(cmp_result)
        let mut cmp_defs: Vec<Option<CmpDef>> = vec![None; 3];
        cmp_defs[1] = Some(CmpDef {
            op: IrCmpOp::Eq,
            lhs: Operand::Value(Value(10)),
            rhs: Operand::Value(Value(11)),
            ty: IrType::I64,
        });
        let mut booleans = vec![false; 3];
        booleans[1] = true;

        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Eq,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(1)),
            ty: IrType::I64,
        };
        let result = try_simplify(&inst, &[], &[], &cmp_defs, &[], &[], &booleans).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_cmp_ne_cmp_result_one() {
        // Cmp(Ne, cmp_result, 1) => Cmp(inverted)
        let mut cmp_defs: Vec<Option<CmpDef>> = vec![None; 3];
        cmp_defs[1] = Some(CmpDef {
            op: IrCmpOp::Ne,
            lhs: Operand::Value(Value(10)),
            rhs: Operand::Value(Value(11)),
            ty: IrType::I64,
        });
        let mut booleans = vec![false; 3];
        booleans[1] = true;

        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Ne,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(1)),
            ty: IrType::I64,
        };
        let result = try_simplify(&inst, &[], &[], &cmp_defs, &[], &[], &booleans).unwrap();
        match result {
            Instruction::Cmp { op: IrCmpOp::Eq, lhs: Operand::Value(a), rhs: Operand::Value(b), .. } => {
                assert_eq!(a.0, 10);
                assert_eq!(b.0, 11);
            }
            _ => panic!("Expected Cmp(Eq, V10, V11), got {:?}", result),
        }
    }

    #[test]
    fn test_cmp_ne_boolean_and_zero() {
        // Cmp(Ne, And(cmp1, cmp2), 0) => Copy(And_result)
        let mut booleans = vec![false; 5];
        booleans[3] = true; // Value(3) = And(cmp1, cmp2) is boolean

        let inst = Instruction::Cmp {
            dest: Value(4), op: IrCmpOp::Ne,
            lhs: Operand::Value(Value(3)),
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::I64,
        };
        let result = try_simplify(&inst, &[], &[], &[], &[], &[], &booleans).unwrap();
        assert_copy_value(&result, 3);
    }

    #[test]
    fn test_cmp_ult_zero_always_false() {
        // Cmp(Ult, x, 0) is always false (nothing is < 0 unsigned)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Ult,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::U64,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_bool(&result, 0);
    }

    #[test]
    fn test_cmp_uge_zero_always_true() {
        // Cmp(Uge, x, 0) is always true (everything is >= 0 unsigned)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Uge,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::U64,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_bool(&result, 1);
    }

    #[test]
    fn test_cmp_no_simplify_non_boolean() {
        // Cmp(Ne, non_cmp_value, 0) should NOT simplify (value could be any integer)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Ne,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::I64,
        };
        assert!(simplify_default(&inst).is_none());
    }

    // === All-ones pattern tests ===

    #[test]
    fn test_and_all_ones_i32() {
        // x & 0xFFFFFFFF => x (I32)
        let inst = binop_val_const(IrBinOp::And, IrConst::I32(-1), IrType::I32);
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_and_all_ones_i64() {
        // x & -1 => x (I64)
        let inst = binop_val_const(IrBinOp::And, IrConst::I64(-1), IrType::I64);
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_or_all_ones_i32() {
        // x | 0xFFFFFFFF => 0xFFFFFFFF (I32)
        let inst = binop_val_const(IrBinOp::Or, IrConst::I32(-1), IrType::I32);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Copy { src: Operand::Const(c), .. } => {
                assert_eq!(c.to_i64(), Some(-1i32 as i64));
            }
            _ => panic!("Expected Copy with all-ones, got {:?}", result),
        }
    }

    #[test]
    fn test_or_all_ones_i64() {
        // x | -1 => -1 (I64)
        let inst = binop_val_const(IrBinOp::Or, IrConst::I64(-1), IrType::I64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::I64(-1)), .. } => {}
            _ => panic!("Expected Copy with I64(-1), got {:?}", result),
        }
    }

    // === is_all_ones helper tests ===

    #[test]
    fn test_is_all_ones_i8() {
        assert!(is_all_ones(&Operand::Const(IrConst::I8(-1)), IrType::I8));
        assert!(is_all_ones(&Operand::Const(IrConst::I8(-1)), IrType::U8));
        assert!(!is_all_ones(&Operand::Const(IrConst::I8(0)), IrType::I8));
        assert!(!is_all_ones(&Operand::Const(IrConst::I8(1)), IrType::I8));
    }

    #[test]
    fn test_is_all_ones_i32() {
        assert!(is_all_ones(&Operand::Const(IrConst::I32(-1)), IrType::I32));
        assert!(!is_all_ones(&Operand::Const(IrConst::I32(0)), IrType::I32));
    }

    #[test]
    fn test_is_all_ones_not_float() {
        assert!(!is_all_ones(&Operand::Const(IrConst::F64(-1.0)), IrType::F64));
    }

    // === Select simplification tests ===

    #[test]
    fn test_select_const_true() {
        // select 1, a, b => a
        let inst = Instruction::Select {
            dest: Value(3),
            cond: Operand::Const(IrConst::I32(1)),
            true_val: Operand::Value(Value(1)),
            false_val: Operand::Value(Value(2)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    #[test]
    fn test_select_const_false() {
        // select 0, a, b => b
        let inst = Instruction::Select {
            dest: Value(3),
            cond: Operand::Const(IrConst::I32(0)),
            true_val: Operand::Value(Value(1)),
            false_val: Operand::Value(Value(2)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 2);
    }

    #[test]
    fn test_select_const_nonzero() {
        // select 42, a, b => a (any nonzero is true)
        let inst = Instruction::Select {
            dest: Value(3),
            cond: Operand::Const(IrConst::I32(42)),
            true_val: Operand::Value(Value(1)),
            false_val: Operand::Value(Value(2)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        assert_copy_value(&result, 1);
    }

    // === Constant reassociation tests ===

    #[test]
    fn test_reassociate_add() {
        // (x + 10) + 20 => x + 30
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Add,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(10)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Add,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(20)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Add, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(30)), .. } => {
                assert_eq!(v.0, 0);
            }
            _ => panic!("Expected Add(V0, 30), got {:?}", result),
        }
    }

    #[test]
    fn test_reassociate_add_cancels() {
        // (x + 5) + (-5) => x
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Add,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(5)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Add,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(-5)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        assert_copy_value(&result, 0);
    }

    #[test]
    fn test_reassociate_sub() {
        // (x - 10) - 20 => x - 30
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Sub,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(10)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Sub,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(20)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Sub, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(30)), .. } => {
                assert_eq!(v.0, 0);
            }
            _ => panic!("Expected Sub(V0, 30), got {:?}", result),
        }
    }

    #[test]
    fn test_reassociate_and() {
        // (x & 0xFF) & 0x0F => x & 0x0F
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::And,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(0xFF)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::And,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(0x0F)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::And, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(0x0F)), .. } => {
                assert_eq!(v.0, 0);
            }
            _ => panic!("Expected And(V0, 0x0F), got {:?}", result),
        }
    }

    #[test]
    fn test_reassociate_xor_cancels() {
        // (x ^ 0xFF) ^ 0xFF => x (cancels out)
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Xor,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(0xFF)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Xor,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(0xFF)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        assert_copy_value(&result, 0);
    }

    #[test]
    fn test_reassociate_shift() {
        // (x << 2) << 3 => x << 5
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Shl,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(2)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Shl,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(3)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Shl, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(5)), .. } => {
                assert_eq!(v.0, 0);
            }
            _ => panic!("Expected Shl(V0, 5), got {:?}", result),
        }
    }

    #[test]
    fn test_reassociate_shift_overflow_to_zero() {
        // (x << 20) << 20 => 0 (combined shift >= 32 bits for I32)
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Shl,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(20)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Shl,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(20)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        assert_copy_const_i32(&result, 0);
    }

    // === Multiply by -1 tests ===

    #[test]
    fn test_mul_neg_one_rhs() {
        // x * (-1) => neg(x)
        let inst = binop_val_const(IrBinOp::Mul, IrConst::I32(-1), IrType::I32);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::UnaryOp { op: IrUnaryOp::Neg, src: Operand::Value(v), ty: IrType::I32, .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected UnaryOp::Neg, got {:?}", result),
        }
    }

    #[test]
    fn test_mul_neg_one_lhs() {
        // (-1) * x => neg(x)
        let inst = binop(IrBinOp::Mul, Operand::Const(IrConst::I64(-1)), Operand::Value(Value(1)), IrType::I64);
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::UnaryOp { op: IrUnaryOp::Neg, src: Operand::Value(v), ty: IrType::I64, .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected UnaryOp::Neg, got {:?}", result),
        }
    }

    #[test]
    fn test_mul_neg_one_float_not_simplified() {
        // Float x * (-1) should NOT be converted to neg (keep mul for IEEE semantics)
        let inst = binop_val_const(IrBinOp::Mul, IrConst::I64(-1), IrType::F64);
        // Note: is_neg_one checks ty.is_integer(), so this won't match
        assert!(simplify_default(&inst).is_none());
    }

    // === Multiply reassociation tests ===

    #[test]
    fn test_reassociate_mul() {
        // (x * 3) * 5 => x * 15
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Mul,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I32(3)),
            ty: IrType::I32,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Mul,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(5)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Mul, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(15)), .. } => {
                assert_eq!(v.0, 0, "Should use original non-const operand");
            }
            _ => panic!("Expected Mul(V0, 15), got {:?}", result),
        }
    }

    #[test]
    fn test_reassociate_mul_to_zero() {
        // (x * 3) * 0 => 0 (already handled by mul-by-zero, but also via reassociation)
        // This test verifies the case is handled correctly
        let inst = binop_val_const(IrBinOp::Mul, IrConst::I32(0), IrType::I32);
        let result = simplify_default(&inst).unwrap();
        assert_copy_const_i32(&result, 0);
    }

    #[test]
    fn test_reassociate_mul_commuted() {
        // 5 * (x * 3) => x * 15
        let mut binop_defs: Vec<Option<BinOpDef>> = vec![None; 4];
        binop_defs[1] = Some(BinOpDef {
            op: IrBinOp::Mul,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Const(IrConst::I64(3)),
            ty: IrType::I64,
        });
        let inst = Instruction::BinOp {
            dest: Value(2),
            op: IrBinOp::Mul,
            lhs: Operand::Const(IrConst::I64(5)),
            rhs: Operand::Value(Value(1)),
            ty: IrType::I64,
        };
        let result = try_simplify(&inst, &[], &[], &[], &binop_defs, &[], &[]).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Mul, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I64(15)), .. } => {
                assert_eq!(v.0, 0);
            }
            _ => panic!("Expected Mul(V0, 15), got {:?}", result),
        }
    }

    // === Subtract-of-negation tests ===

    #[test]
    fn test_sub_neg_to_add() {
        // x - (neg y) => x + y
        let mut neg_defs: Vec<Option<NegDef>> = vec![None; 4];
        neg_defs[2] = Some(NegDef { src: Operand::Value(Value(1)) });
        let inst = Instruction::BinOp {
            dest: Value(3),
            op: IrBinOp::Sub,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Value(Value(2)),
            ty: IrType::I32,
        };
        let result = try_simplify(&inst, &[], &[], &[], &[], &neg_defs, &[]).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Add, lhs: Operand::Value(a), rhs: Operand::Value(b), .. } => {
                assert_eq!(a.0, 0, "lhs should be original lhs");
                assert_eq!(b.0, 1, "rhs should be negation source");
            }
            _ => panic!("Expected Add(V0, V1), got {:?}", result),
        }
    }

    #[test]
    fn test_sub_neg_float_not_simplified() {
        // Float x - (neg y) should NOT be simplified (IEEE 754 concerns)
        let mut neg_defs: Vec<Option<NegDef>> = vec![None; 4];
        neg_defs[2] = Some(NegDef { src: Operand::Value(Value(1)) });
        let inst = Instruction::BinOp {
            dest: Value(3),
            op: IrBinOp::Sub,
            lhs: Operand::Value(Value(0)),
            rhs: Operand::Value(Value(2)),
            ty: IrType::F64,
        };
        let result = try_simplify(&inst, &[], &[], &[], &[], &neg_defs, &[]);
        assert!(result.is_none());
    }

    // === Unsigned comparison simplification tests ===

    #[test]
    fn test_cmp_ule_zero() {
        // Cmp(Ule, x, 0) => Cmp(Eq, x, 0)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Ule,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(0)),
            ty: IrType::U32,
        };
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Cmp { op: IrCmpOp::Eq, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(0)), .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected Cmp(Eq, V1, 0), got {:?}", result),
        }
    }

    #[test]
    fn test_cmp_ugt_zero() {
        // Cmp(Ugt, x, 0) => Cmp(Ne, x, 0)
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Ugt,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::U64,
        };
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Cmp { op: IrCmpOp::Ne, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I64(0)), .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected Cmp(Ne, V1, 0), got {:?}", result),
        }
    }

    // === Operand canonicalization tests ===

    #[test]
    fn test_cmp_canonicalize_const_lhs_slt() {
        // Cmp(Slt, Const(5), Value(1)) => Cmp(Sgt, Value(1), Const(5))
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Slt,
            lhs: Operand::Const(IrConst::I32(5)),
            rhs: Operand::Value(Value(1)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Cmp { op: IrCmpOp::Sgt, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(5)), .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected Cmp(Sgt, V1, I32(5)), got {:?}", result),
        }
    }

    #[test]
    fn test_cmp_canonicalize_const_lhs_eq() {
        // Cmp(Eq, Const(42), Value(1)) => Cmp(Eq, Value(1), Const(42))
        // Eq is symmetric so swap_cmp_op returns Eq
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Eq,
            lhs: Operand::Const(IrConst::I64(42)),
            rhs: Operand::Value(Value(1)),
            ty: IrType::I64,
        };
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::Cmp { op: IrCmpOp::Eq, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I64(42)), .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected Cmp(Eq, V1, I64(42)), got {:?}", result),
        }
    }

    #[test]
    fn test_cmp_no_canonicalize_already_canonical() {
        // Cmp(Slt, Value(1), Const(5)) - already canonical, no change
        let inst = Instruction::Cmp {
            dest: Value(2), op: IrCmpOp::Slt,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(5)),
            ty: IrType::I32,
        };
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_binop_canonicalize_commutative_add() {
        // BinOp(Add, Const(10), Value(1)) => BinOp(Add, Value(1), Const(10))
        let inst = Instruction::BinOp {
            dest: Value(2), op: IrBinOp::Add,
            lhs: Operand::Const(IrConst::I32(10)),
            rhs: Operand::Value(Value(1)),
            ty: IrType::I32,
        };
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::Add, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I32(10)), .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected BinOp(Add, V1, I32(10)), got {:?}", result),
        }
    }

    #[test]
    fn test_binop_canonicalize_commutative_and() {
        // BinOp(And, Const(0xFF), Value(1)) => BinOp(And, Value(1), Const(0xFF))
        let inst = Instruction::BinOp {
            dest: Value(2), op: IrBinOp::And,
            lhs: Operand::Const(IrConst::I64(0xFF)),
            rhs: Operand::Value(Value(1)),
            ty: IrType::I64,
        };
        let result = simplify_default(&inst).unwrap();
        match result {
            Instruction::BinOp { op: IrBinOp::And, lhs: Operand::Value(v), rhs: Operand::Const(IrConst::I64(0xFF)), .. } => {
                assert_eq!(v.0, 1);
            }
            _ => panic!("Expected BinOp(And, V1, I64(0xFF)), got {:?}", result),
        }
    }

    #[test]
    fn test_binop_no_canonicalize_non_commutative() {
        // BinOp(Sub, Const(10), Value(1)) - Sub is NOT commutative, should NOT swap
        let inst = Instruction::BinOp {
            dest: Value(2), op: IrBinOp::Sub,
            lhs: Operand::Const(IrConst::I32(10)),
            rhs: Operand::Value(Value(1)),
            ty: IrType::I32,
        };
        // Sub has its own simplifications (x-0, x-x, reassoc) but NOT canonicalization
        assert!(simplify_default(&inst).is_none());
    }

    #[test]
    fn test_binop_no_canonicalize_already_canonical() {
        // BinOp(Add, Value(1), Const(10)) - already canonical, no change
        let inst = Instruction::BinOp {
            dest: Value(2), op: IrBinOp::Add,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(10)),
            ty: IrType::I32,
        };
        assert!(simplify_default(&inst).is_none());
    }
}
