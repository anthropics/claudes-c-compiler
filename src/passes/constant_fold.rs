//! Constant folding optimization pass.
//!
//! This pass evaluates operations on constant operands at compile time,
//! replacing the instruction with the computed constant. This eliminates
//! redundant computation and enables further optimizations (DCE, etc.).

use crate::ir::reexports::{
    Instruction,
    IrBinOp,
    IrCmpOp,
    IrConst,
    IrFunction,
    IrModule,
    IrUnaryOp,
    Operand,
    Value,
};
use crate::common::types::IrType;

/// Run constant folding on the entire module.
/// Returns the number of instructions folded.
pub fn run(module: &mut IrModule) -> usize {
    module.for_each_function(fold_function)
}

/// Resolve all remaining `UnaryOp::IsConstant` instructions to `Copy(0)`.
///
/// After inlining and the post-inline constant folding passes, any `IsConstant`
/// whose operand became constant has already been resolved to `Copy(1)`. The
/// remaining `IsConstant` instructions have operands that are definitively not
/// compile-time constants (e.g., global variables, function parameters in non-inlined
/// contexts).
///
/// Resolving these to `Copy(0)` before the main optimization loop is critical:
/// it enables `cfg_simplify` to fold `CondBranch` instructions that test the
/// result of `__builtin_constant_p` and eliminate dead code paths. Without this,
/// unreachable function calls (like the kernel's `__bad_udelay()`, which is
/// intentionally undefined to generate a link error for invalid `udelay()` arguments)
/// survive into the object file and cause linker errors.
pub fn resolve_remaining_is_constant(module: &mut IrModule) {
    for func in &mut module.functions {
        if func.is_declaration || func.blocks.is_empty() {
            continue;
        }
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if let Instruction::UnaryOp { dest, op: IrUnaryOp::IsConstant, src, .. } = inst {
                    // Check if the operand is a constant - if so, resolve to 1 (true).
                    let is_const = matches!(src, Operand::Const(_));
                    *inst = Instruction::Copy {
                        dest: *dest,
                        src: Operand::Const(IrConst::I32(if is_const { 1 } else { 0 })),
                    };
                }
            }
        }
    }
}

/// Fold constants within a single function.
///
/// Builds a const-value map from Copy instructions so that operands referencing
/// values defined by `Copy { src: Const(c) }` can be resolved without waiting
/// for a separate copy propagation pass. This is critical for deeply nested
/// constant expressions like `-(unsigned short)(-1)` where each sub-expression
/// folds to a Copy of a constant, and outer expressions need to see through
/// the chain within a single pass invocation.
///
/// The const map tracks both the constant value and the defining Cast's target
/// type (if any), so that sub-int constants (I8/I16) can be correctly zero-
/// extended (for U8/U16 targets) or sign-extended (for I8/I16 targets) when
/// used as operands of UnaryOp/BitNot instructions.
pub(crate) fn fold_function(func: &mut IrFunction) -> usize {
    let max_id = func.max_value_id() as usize;
    let mut total = 0;

    loop {
        // Build a map of Value -> (IrConst, defining_cast_target_type).
        // The defining_cast_target_type is the `to_ty` of the Cast instruction
        // that was folded to produce this constant, or None for non-Cast sources.
        let mut const_map: Vec<Option<ConstMapEntry>> = vec![None; max_id + 1];

        // First pass: record Cast target types for all Cast instructions.
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::Cast { dest, to_ty, .. } = inst {
                    let id = dest.0 as usize;
                    if id <= max_id {
                        const_map[id] = Some(ConstMapEntry { konst: None, cast_to_ty: Some(*to_ty) });
                    }
                }
            }
        }

        // Second pass: record constants from Copy instructions, preserving
        // any Cast target type already recorded for this value.
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::Copy { dest, src: Operand::Const(c) } = inst {
                    let id = dest.0 as usize;
                    if id <= max_id {
                        let cast_ty = const_map[id].and_then(|e| e.cast_to_ty);
                        const_map[id] = Some(ConstMapEntry { konst: Some(*c), cast_to_ty: cast_ty });
                    }
                }
            }
        }

        let mut folded = 0;
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if let Some(folded_inst) = try_fold_with_map(inst, &const_map) {
                    // Update the const map immediately so later instructions in
                    // the same block can see through newly-folded constants.
                    if let Instruction::Copy { dest, src: Operand::Const(c) } = &folded_inst {
                        let id = dest.0 as usize;
                        if id <= max_id {
                            let cast_ty = const_map[id].and_then(|e| e.cast_to_ty);
                            const_map[id] = Some(ConstMapEntry { konst: Some(*c), cast_to_ty: cast_ty });
                        }
                    }
                    *inst = folded_inst;
                    folded += 1;
                }
            }
        }

        if folded == 0 {
            break;
        }
        total += folded;
    }
    total
}

/// Entry in the constant value map, tracking both the constant value and the
/// defining Cast's target type for correct sign/zero extension of sub-int types.
#[derive(Clone, Copy)]
struct ConstMapEntry {
    /// The constant value, if known (from a Copy { src: Const(c) } instruction).
    konst: Option<IrConst>,
    /// The target type of the Cast instruction that defined this value, if any.
    /// Used to determine whether I8/I16 constants represent unsigned (U8/U16)
    /// or signed (I8/I16) sub-int values for correct extension in UnaryOp folding.
    cast_to_ty: Option<IrType>,
}

/// Try to fold a single instruction without const map lookup (for tests and
/// simple cases where all operands are already Operand::Const).
#[cfg(test)]
fn try_fold(inst: &Instruction) -> Option<Instruction> {
    try_fold_with_map(inst, &[])
}

/// Try to fold a single instruction, using the const map to resolve Value
/// operands that are known to be constants (from prior Copy instructions).
fn try_fold_with_map(inst: &Instruction, const_map: &[Option<ConstMapEntry>]) -> Option<Instruction> {
    match inst {
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            // For 128-bit types, use native i128 arithmetic to avoid truncation
            if ty.is_128bit() {
                let lc = resolve_const(lhs, const_map)?;
                let rc = resolve_const(rhs, const_map)?;
                let l = lc.to_i128()?;
                let r = rc.to_i128()?;
                let result = op.eval_i128(l, r)?;
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: Operand::Const(IrConst::I128(result)),
                });
            }
            // Try float folding first
            if ty.is_float() {
                // For F128 (long double), use full x87 precision arithmetic
                if *ty == IrType::F128 {
                    let lc = resolve_const(lhs, const_map)?;
                    let rc = resolve_const(rhs, const_map)?;
                    let result = fold_f128_binop(*op, &lc, &rc)?;
                    return Some(Instruction::Copy {
                        dest: *dest,
                        src: Operand::Const(result),
                    });
                }
                let l = as_f64_const_mapped(lhs, const_map)?;
                let r = as_f64_const_mapped(rhs, const_map)?;
                let result = fold_float_binop(*op, l, r)?;
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: Operand::Const(make_float_const(result, *ty)),
                });
            }
            let lhs_const = as_i64_const_mapped(lhs, const_map)?;
            let rhs_const = as_i64_const_mapped(rhs, const_map)?;
            // Truncate operands to the BinOp's type width before folding.
            // This is needed because constants may be stored in wider IrConst
            // variants (e.g., a U32 value stored as IrConst::I32) and to_i64()
            // sign-extends them. Operations like LShr and UDiv are sensitive
            // to upper bits, so we must normalize first.
            let lhs_trunc = ty.truncate_i64(lhs_const);
            let rhs_trunc = ty.truncate_i64(rhs_const);
            let result = fold_binop(*op, lhs_trunc, rhs_trunc, *ty)?;
            Some(Instruction::Copy {
                dest: *dest,
                src: Operand::Const(IrConst::from_i64(result, *ty)),
            })
        }
        Instruction::UnaryOp { dest, op, src, ty } => {
            // IsConstant (__builtin_constant_p): resolve based on whether operand is constant.
            // After inlining and constant propagation, the operand may have become a constant.
            if *op == IrUnaryOp::IsConstant {
                // If the operand is already a constant (literal or known via const_map),
                // resolve to 1. Otherwise, leave unresolved so that later passes
                // (copy_prop, simplify) can simplify the operand, and a subsequent
                // constant_fold run can try again.
                if resolve_const(src, const_map).is_some() {
                    return Some(Instruction::Copy {
                        dest: *dest,
                        src: Operand::Const(IrConst::I32(1)),
                    });
                }
                // Not yet known to be constant - don't fold yet
                return None;
            }
            // For 128-bit types, fold Neg and Not using native i128
            if ty.is_128bit() {
                let sc = resolve_const(src, const_map)?;
                let s = sc.to_i128()?;
                let result = match op {
                    IrUnaryOp::Neg => s.wrapping_neg(),
                    IrUnaryOp::Not => !s,
                    _ => return None,
                };
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: Operand::Const(IrConst::I128(result)),
                });
            }
            // Try float unary folding
            if ty.is_float() {
                // For F128 (long double), preserve full x87 precision on negation
                if *ty == IrType::F128 && *op == IrUnaryOp::Neg {
                    let sc = resolve_const(src, const_map)?;
                    let result = fold_f128_neg(&sc);
                    return Some(Instruction::Copy {
                        dest: *dest,
                        src: Operand::Const(result),
                    });
                }
                let s = as_f64_const_mapped(src, const_map)?;
                let result = fold_float_unaryop(*op, s)?;
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: Operand::Const(make_float_const(result, *ty)),
                });
            }
            // Promote sub-int constants (I8/I16) for C11 integer promotion.
            // Zero-extend unsigned sub-int (U8/U16 cast targets) and sign-extend
            // signed sub-int (I8/I16 cast targets) to match the C promotion rules.
            let src_const = as_i64_promoted_mapped(src, const_map)?;
            let result = fold_unaryop(*op, src_const, *ty)?;
            Some(Instruction::Copy {
                dest: *dest,
                src: Operand::Const(IrConst::from_i64(result, *ty)),
            })
        }
        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            // For 128-bit types, fold comparisons using native i128
            if ty.is_128bit() {
                let lc = resolve_const(lhs, const_map)?;
                let rc = resolve_const(rhs, const_map)?;
                let l = lc.to_i128()?;
                let r = rc.to_i128()?;
                let result = op.eval_i128(l, r);
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: Operand::Const(IrConst::I32(result as i32)),
                });
            }
            // Try float comparison folding
            if ty.is_float() {
                // For F128 (long double), use full x87 precision comparison
                if *ty == IrType::F128 {
                    let lc = resolve_const(lhs, const_map)?;
                    let rc = resolve_const(rhs, const_map)?;
                    let result = fold_f128_cmp(*op, &lc, &rc);
                    return Some(Instruction::Copy {
                        dest: *dest,
                        src: Operand::Const(IrConst::I32(result as i32)),
                    });
                }
                let l = as_f64_const_mapped(lhs, const_map)?;
                let r = as_f64_const_mapped(rhs, const_map)?;
                let result = op.eval_f64(l, r);
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: Operand::Const(IrConst::I32(result as i32)),
                });
            }
            let lhs_const = as_i64_const_mapped(lhs, const_map)?;
            let rhs_const = as_i64_const_mapped(rhs, const_map)?;
            // Truncate operands to the comparison type's width before comparing.
            // This is needed because constants may be stored in wider IrConst
            // variants (e.g., a U32 value stored as IrConst::I32) and to_i64()
            // sign-extends them. By truncating to the comparison type, we ensure
            // bit-level equality is correct regardless of storage representation.
            let result = op.eval_i64(ty.truncate_i64(lhs_const), ty.truncate_i64(rhs_const));
            Some(Instruction::Copy {
                dest: *dest,
                src: Operand::Const(IrConst::I32(result as i32)),
            })
        }
        Instruction::Cast { dest, src, from_ty, to_ty } => {
            // For casts involving 128-bit types, use native i128 arithmetic
            if from_ty.is_128bit() || to_ty.is_128bit() {
                let sc = resolve_const(src, const_map)?;
                let result = fold_cast_i128(&sc, *from_ty, *to_ty)?;
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: Operand::Const(result),
                });
            }
            // Handle float-to-int and int-to-float casts
            if from_ty.is_float() || to_ty.is_float() {
                return try_fold_float_cast_mapped(*dest, src, *from_ty, *to_ty, const_map);
            }
            let src_const = as_i64_const_mapped(src, const_map)?;
            let result = fold_cast(src_const, *from_ty, *to_ty);
            Some(Instruction::Copy {
                dest: *dest,
                src: Operand::Const(IrConst::from_i64(result, *to_ty)),
            })
        }
        Instruction::Select { dest, cond, true_val, false_val, .. } => {
            // If both arms are the same operand, the result is that operand
            // regardless of the condition. This handles patterns like
            // Select(cond, 0, 0) from dead short-circuit branches.
            if operands_equal(true_val, false_val) {
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: *true_val,
                });
            }
            // If the condition is a known constant, fold to the appropriate value
            if let Some(cond_const) = as_i64_const_mapped(cond, const_map) {
                let result = if cond_const != 0 { *true_val } else { *false_val };
                return Some(Instruction::Copy {
                    dest: *dest,
                    src: result,
                });
            }
            // If both arms resolve to the same constant, the Select is redundant
            // regardless of the condition. This is critical for patterns like:
            //   if (test_large(x) && always_false(x)) dead_code();
            // where if-conversion creates Select(cond, 0, 0) but one arm is
            // still a Value (from cmp ne 0,0) that constant fold hasn't yet
            // propagated into the Select operands.
            if let (Some(tv), Some(fv)) = (resolve_const(true_val, const_map), resolve_const(false_val, const_map)) {
                // Compare constants across potentially different widths (e.g.,
                // I32(0) from a cmp result vs I64(0) from a short-circuit default).
                let same = tv.to_hash_key() == fv.to_hash_key()
                    || matches!((tv.to_i64(), fv.to_i64()), (Some(a), Some(b)) if a == b);
                if same {
                    return Some(Instruction::Copy {
                        dest: *dest,
                        src: Operand::Const(tv),
                    });
                }
            }
            None
        }
        _ => None,
    }
}

/// Check if two operands are structurally equal.
fn operands_equal(a: &Operand, b: &Operand) -> bool {
    match (a, b) {
        (Operand::Value(va), Operand::Value(vb)) => va.0 == vb.0,
        (Operand::Const(ca), Operand::Const(cb)) => ca.to_hash_key() == cb.to_hash_key(),
        _ => false,
    }
}

/// Resolve an operand to its constant value, looking through the const map
/// for Value operands that are defined by Copy instructions with const sources.
fn resolve_const(op: &Operand, const_map: &[Option<ConstMapEntry>]) -> Option<IrConst> {
    match op {
        Operand::Const(c) => Some(*c),
        Operand::Value(v) => {
            let id = v.0 as usize;
            if id < const_map.len() {
                const_map[id].as_ref()?.konst
            } else {
                None
            }
        }
    }
}

/// Extract a constant integer value from an operand, resolving through const map.
fn as_i64_const_mapped(op: &Operand, const_map: &[Option<ConstMapEntry>]) -> Option<i64> {
    resolve_const(op, const_map)?.to_i64()
}

/// Extract a constant integer with proper extension for sub-int types (I8/I16),
/// resolving through the const map.
///
/// For sub-int constants, we need to determine whether to zero-extend (unsigned)
/// or sign-extend (signed) based on the Cast target type that produced the value:
/// - Constants from Cast to U8/U16 are zero-extended (e.g., (unsigned short)(-1) = 65535)
/// - Constants from Cast to I8/I16 are sign-extended (e.g., (signed char)(-1) = -1)
/// - Literal I8/I16 constants (not from a Cast) are zero-extended per the original
///   invariant: the lowerer promotes signed sub-int to I32, so remaining I8/I16 are unsigned
fn as_i64_promoted_mapped(op: &Operand, const_map: &[Option<ConstMapEntry>]) -> Option<i64> {
    let c = resolve_const(op, const_map)?;
    match &c {
        IrConst::I8(v) => {
            // Check if the value came from a Cast to a signed type
            if let Operand::Value(val) = op {
                let id = val.0 as usize;
                if id < const_map.len() {
                    if let Some(entry) = &const_map[id] {
                        if entry.cast_to_ty == Some(IrType::I8) {
                            // Signed char: sign-extend
                            return Some(*v as i64);
                        }
                    }
                }
            }
            // Default: zero-extend (unsigned char or literal)
            Some(*v as u8 as i64)
        }
        IrConst::I16(v) => {
            // Check if the value came from a Cast to a signed type
            if let Operand::Value(val) = op {
                let id = val.0 as usize;
                if id < const_map.len() {
                    if let Some(entry) = &const_map[id] {
                        if entry.cast_to_ty == Some(IrType::I16) {
                            // Signed short: sign-extend
                            return Some(*v as i64);
                        }
                    }
                }
            }
            // Default: zero-extend (unsigned short or literal)
            Some(*v as u16 as i64)
        }
        _ => c.to_i64(),
    }
}

/// Extract a constant floating-point value, resolving through const map.
fn as_f64_const_mapped(op: &Operand, const_map: &[Option<ConstMapEntry>]) -> Option<f64> {
    match resolve_const(op, const_map)? {
        IrConst::F32(v) => Some(v as f64),
        IrConst::F64(v) => Some(v),
        IrConst::LongDouble(v, _) => Some(v),
        _ => None,
    }
}

/// Create a float constant of the appropriate type from an f64 value.
pub(crate) fn make_float_const(val: f64, ty: IrType) -> IrConst {
    match ty {
        IrType::F32 => IrConst::F32(val as f32),
        IrType::F64 => IrConst::F64(val),
        IrType::F128 => IrConst::long_double(val),
        _ => unreachable!("make_float_const called with non-float type"),
    }
}

/// Evaluate a binary operation on two constant floats.
/// Uses Rust's native f64 arithmetic which is IEEE 754 compliant.
pub(crate) fn fold_float_binop(op: IrBinOp, lhs: f64, rhs: f64) -> Option<f64> {
    Some(match op {
        IrBinOp::Add => lhs + rhs,
        IrBinOp::Sub => lhs - rhs,
        IrBinOp::Mul => lhs * rhs,
        // For division, allow folding even for division by zero (produces Inf/-Inf/NaN per IEEE 754)
        IrBinOp::SDiv | IrBinOp::UDiv => lhs / rhs,
        IrBinOp::SRem | IrBinOp::URem => lhs % rhs,
        // Bitwise ops don't apply to floats
        _ => return None,
    })
}

/// Evaluate a unary operation on a constant float.
fn fold_float_unaryop(op: IrUnaryOp, src: f64) -> Option<f64> {
    match op {
        IrUnaryOp::Neg => Some(-src),
        _ => None,
    }
}

/// Fold a binary operation on two F128 (long double) constants.
///
/// On ARM64/RISC-V (where long double is IEEE binary128), uses full 112-bit mantissa
/// f128 software arithmetic to match the runtime __addtf3/__subtf3/__multf3/__divtf3.
///
/// On x86/i686 (where long double is x87 80-bit), uses x87 arithmetic to match the
/// runtime x87 FPU behavior.
fn fold_f128_binop(op: IrBinOp, lhs: &IrConst, rhs: &IrConst) -> Option<IrConst> {
    use crate::common::long_double;
    use crate::common::types::target_long_double_is_f128;

    if target_long_double_is_f128() {
        // ARM64/RISC-V: full f128 software arithmetic
        let la = lhs.long_double_bytes()?;
        let ra = rhs.long_double_bytes()?;

        let result_f128 = match op {
            IrBinOp::Add => long_double::f128_add(la, ra),
            IrBinOp::Sub => long_double::f128_sub(la, ra),
            IrBinOp::Mul => long_double::f128_mul(la, ra),
            IrBinOp::SDiv | IrBinOp::UDiv => long_double::f128_div(la, ra),
            IrBinOp::SRem | IrBinOp::URem => long_double::f128_rem(la, ra),
            _ => return None,
        };

        let approx = long_double::f128_bytes_to_f64(&result_f128);
        Some(IrConst::long_double_with_bytes(approx, result_f128))
    } else {
        // x86/i686: x87 80-bit arithmetic
        let la = lhs.x87_bytes();
        let ra = rhs.x87_bytes();

        let result_x87 = match op {
            IrBinOp::Add => long_double::x87_add(&la, &ra),
            IrBinOp::Sub => long_double::x87_sub(&la, &ra),
            IrBinOp::Mul => long_double::x87_mul(&la, &ra),
            IrBinOp::SDiv | IrBinOp::UDiv => long_double::x87_div(&la, &ra),
            IrBinOp::SRem | IrBinOp::URem => long_double::x87_rem(&la, &ra),
            _ => return None,
        };

        let result_f128 = long_double::x87_bytes_to_f128_bytes(&result_x87);
        let approx = long_double::x87_to_f64(&result_x87);
        Some(IrConst::long_double_with_bytes(approx, result_f128))
    }
}

/// Negate an F128 (long double) constant with full precision by flipping the sign bit.
fn fold_f128_neg(src: &IrConst) -> IrConst {
    use crate::common::long_double;

    // For negation, we can operate directly on f128 bytes (just flip sign bit)
    if let IrConst::LongDouble(fv, f128_bytes) = src {
        let val = u128::from_le_bytes(*f128_bytes);
        let neg_val = val ^ (1u128 << 127); // flip sign bit
        let neg_f128 = neg_val.to_le_bytes();
        return IrConst::long_double_with_bytes(-fv, neg_f128);
    }
    // Fallback via x87
    let bytes = src.x87_bytes();
    let neg_bytes = long_double::x87_neg(&bytes);
    let neg_f128 = long_double::x87_bytes_to_f128_bytes(&neg_bytes);
    let approx = long_double::x87_to_f64(&neg_bytes);
    IrConst::long_double_with_bytes(approx, neg_f128)
}

/// Compare two F128 (long double) constants using x87 80-bit precision.
fn fold_f128_cmp(op: IrCmpOp, lhs: &IrConst, rhs: &IrConst) -> bool {
    use crate::common::long_double;

    // Convert f128 -> x87 for comparison
    let la = lhs.x87_bytes();
    let ra = rhs.x87_bytes();
    let cmp = long_double::x87_cmp(&la, &ra);

    // cmp: -1 = a<b, 0 = a==b, 1 = a>b, i32::MIN = unordered (NaN)
    match op {
        IrCmpOp::Eq => cmp == 0,
        IrCmpOp::Ne => cmp != 0, // NaN != NaN is true per IEEE 754 (unordered gives i32::MIN which is != 0)
        IrCmpOp::Slt | IrCmpOp::Ult => cmp == -1,
        IrCmpOp::Sle | IrCmpOp::Ule => cmp == -1 || cmp == 0,
        IrCmpOp::Sgt | IrCmpOp::Ugt => cmp == 1,
        IrCmpOp::Sge | IrCmpOp::Uge => cmp == 1 || cmp == 0,
    }
}

/// Try to fold a cast involving float types.
// TODO: simplify.rs also has float cast folding via IrConst::cast_float_to_target
// which doesn't check for NaN/Inf. These should be unified eventually.
fn try_fold_float_cast_mapped(dest: Value, src: &Operand, from_ty: IrType, to_ty: IrType, const_map: &[Option<ConstMapEntry>]) -> Option<Instruction> {
    let src_const = resolve_const(src, const_map)?;
    let result = match (from_ty.is_float(), to_ty.is_float()) {
        (true, true) => {
            // float-to-float conversion
            let val = as_f64_const_mapped(src, const_map)?;
            make_float_const(val, to_ty)
        }
        (true, false) => {
            // float-to-int conversion
            // For LongDouble, use full x87 precision to avoid mantissa loss
            if let Some(IrConst::LongDouble(fv, bytes)) = resolve_const(src, const_map) {
                return IrConst::cast_long_double_to_target(fv, &bytes, to_ty).map(|c| {
                    Instruction::Copy { dest, src: Operand::Const(c) }
                });
            }
            let val = as_f64_const_mapped(src, const_map)?;
            // Don't fold if value can't be represented as i64
            if !val.is_finite() || val < i64::MIN as f64 || val > i64::MAX as f64 {
                return None;
            }
            IrConst::from_i64(val as i64, to_ty)
        }
        (false, true) => {
            // int-to-float conversion
            // Normalize to source type width first (sign-extend or zero-extend)
            let raw_val = src_const.to_i64()?;
            let val = match from_ty {
                IrType::I8 => raw_val as i8 as i64,
                IrType::U8 => raw_val as u8 as i64,
                IrType::I16 => raw_val as i16 as i64,
                IrType::U16 => raw_val as u16 as i64,
                IrType::I32 => raw_val as i32 as i64,
                IrType::U32 => raw_val as u32 as i64,
                _ => raw_val,
            };
            if to_ty == IrType::F128 {
                // For long double, use direct integer-to-x87 conversion to preserve
                // full 64-bit precision (x87 has 64-bit mantissa, unlike f64's 52-bit).
                if from_ty.is_unsigned() {
                    IrConst::long_double_from_u64(val as u64)
                } else {
                    IrConst::long_double_from_i64(val)
                }
            } else if from_ty.is_unsigned() {
                make_float_const(val as u64 as f64, to_ty)
            } else {
                make_float_const(val as f64, to_ty)
            }
        }
        _ => return None,
    };
    Some(Instruction::Copy {
        dest,
        src: Operand::Const(result),
    })
}

/// Fold a cast involving 128-bit types.
pub(crate) fn fold_cast_i128(src: &IrConst, from_ty: IrType, to_ty: IrType) -> Option<IrConst> {
    let val = src.to_i128()?;

    if to_ty.is_128bit() {
        // Widening to i128: need to respect source signedness.
        // to_i128() sign-extends from i64, which is correct for signed sources.
        // For unsigned sources, we need to zero-extend: e.g., u64 0xCAFEBABE12345678
        // should become u128 0x00000000_00000000_CAFEBABE12345678, not
        // 0xFFFFFFFF_FFFFFFFF_CAFEBABE12345678.
        if from_ty.is_unsigned() && !from_ty.is_128bit() {
            // Zero-extend: reinterpret the i128 (which was sign-extended from i64)
            // as just the low 64 bits zero-extended.
            let zero_extended = (val as u64 as u128) as i128;
            Some(IrConst::I128(zero_extended))
        } else {
            Some(IrConst::I128(val))
        }
    } else if to_ty.is_float() {
        // i128 to float
        let fval = if from_ty.is_unsigned() {
            (val as u128) as f64
        } else {
            val as f64
        };
        Some(match to_ty {
            IrType::F32 => IrConst::F32(fval as f32),
            IrType::F64 => IrConst::F64(fval),
            _ => return None,
        })
    } else {
        // Narrowing from i128 to smaller int
        let i64_val = val as i64;
        Some(IrConst::from_i64(fold_cast(i64_val, IrType::I64, to_ty), to_ty))
    }
}

/// Evaluate a binary operation on two constant integers.
/// `ty` is needed for width-sensitive unsigned operations (LShr, UDiv, URem)
/// where operands stored as sign-extended i64 must be masked to the correct
/// bit width to get the proper unsigned representation.
pub(crate) fn fold_binop(op: IrBinOp, lhs: i64, rhs: i64, ty: IrType) -> Option<i64> {
    let is_32bit = ty == IrType::I32 || ty == IrType::U32
        || ty == IrType::I16 || ty == IrType::U16
        || ty == IrType::I8 || ty == IrType::U8;
    Some(match op {
        IrBinOp::Add => lhs.wrapping_add(rhs),
        IrBinOp::Sub => lhs.wrapping_sub(rhs),
        IrBinOp::Mul => lhs.wrapping_mul(rhs),
        IrBinOp::SDiv => {
            if rhs == 0 { return None; } // division by zero is UB, don't fold
            lhs.wrapping_div(rhs)
        }
        IrBinOp::UDiv => {
            if rhs == 0 { return None; }
            if is_32bit {
                // For 32-bit types, mask to u32 to get correct unsigned value
                (lhs as u32).wrapping_div(rhs as u32) as i64
            } else {
                (lhs as u64).wrapping_div(rhs as u64) as i64
            }
        }
        IrBinOp::SRem => {
            if rhs == 0 { return None; }
            lhs.wrapping_rem(rhs)
        }
        IrBinOp::URem => {
            if rhs == 0 { return None; }
            if is_32bit {
                (lhs as u32).wrapping_rem(rhs as u32) as i64
            } else {
                (lhs as u64).wrapping_rem(rhs as u64) as i64
            }
        }
        IrBinOp::And => lhs & rhs,
        IrBinOp::Or => lhs | rhs,
        IrBinOp::Xor => lhs ^ rhs,
        IrBinOp::Shl => lhs.wrapping_shl(rhs as u32),
        IrBinOp::AShr => {
            if is_32bit {
                // For 32-bit types, use i32 arithmetic shift to stay in 32-bit range
                (lhs as i32).wrapping_shr(rhs as u32) as i64
            } else {
                lhs.wrapping_shr(rhs as u32)
            }
        }
        IrBinOp::LShr => {
            if is_32bit {
                // For 32-bit types, mask to u32 to get correct unsigned bit pattern
                (lhs as u32).wrapping_shr(rhs as u32) as i64
            } else {
                (lhs as u64).wrapping_shr(rhs as u32) as i64
            }
        }
    })
}

/// Evaluate a unary operation on a constant integer.
/// Width-sensitive operations (CLZ, CTZ, Popcount, Bswap) use `ty` to determine
/// whether to operate on 32 or 64 bits, matching the runtime semantics of
/// __builtin_clz vs __builtin_clzll, etc.
pub(crate) fn fold_unaryop(op: IrUnaryOp, src: i64, ty: IrType) -> Option<i64> {
    let is_32bit = ty == IrType::I32 || ty == IrType::U32
        || ty == IrType::I16 || ty == IrType::U16
        || ty == IrType::I8 || ty == IrType::U8;
    Some(match op {
        IrUnaryOp::Neg => src.wrapping_neg(),
        IrUnaryOp::Not => !src,
        IrUnaryOp::Clz => {
            if is_32bit {
                (src as u32).leading_zeros() as i64
            } else {
                (src as u64).leading_zeros() as i64
            }
        }
        IrUnaryOp::Ctz => {
            if src == 0 {
                if is_32bit { 32 } else { 64 }
            } else if is_32bit {
                (src as u32).trailing_zeros() as i64
            } else {
                (src as u64).trailing_zeros() as i64
            }
        }
        IrUnaryOp::Bswap => {
            // Bswap is width-sensitive: swapping 2 bytes vs 4 vs 8 produces
            // different results, unlike CLZ/CTZ/Popcount where sub-32-bit
            // values are implicitly zero-extended into 32 bits.
            if ty == IrType::I16 || ty == IrType::U16 {
                (src as u16).swap_bytes() as i64
            } else if ty == IrType::I32 || ty == IrType::U32 {
                (src as u32).swap_bytes() as i64
            } else {
                (src as u64).swap_bytes() as i64
            }
        }
        IrUnaryOp::Popcount => {
            if is_32bit {
                (src as u32).count_ones() as i64
            } else {
                (src as u64).count_ones() as i64
            }
        }
        IrUnaryOp::IsConstant => {
            // If we got here, the operand was already constant
            1
        }
    })
}

/// Evaluate a type cast on a constant.
///
/// For signed source types, we sign-extend to get the correct i64 representation.
/// For unsigned source types, we zero-extend (mask to type width).
/// Same logic applies to the target type.
pub(crate) fn fold_cast(val: i64, from_ty: crate::common::types::IrType, to_ty: crate::common::types::IrType) -> i64 {
    // Normalize source to its width/signedness, then convert to target.
    to_ty.truncate_i64(from_ty.truncate_i64(val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IrType;

    #[test]
    fn test_fold_binop_add() {
        assert_eq!(fold_binop(IrBinOp::Add, 3, 4, IrType::I64), Some(7));
    }

    #[test]
    fn test_fold_binop_sub() {
        assert_eq!(fold_binop(IrBinOp::Sub, 10, 3, IrType::I64), Some(7));
    }

    #[test]
    fn test_fold_binop_mul() {
        assert_eq!(fold_binop(IrBinOp::Mul, 6, 7, IrType::I64), Some(42));
    }

    #[test]
    fn test_fold_binop_div() {
        assert_eq!(fold_binop(IrBinOp::SDiv, 10, 3, IrType::I64), Some(3));
        assert_eq!(fold_binop(IrBinOp::SDiv, -10, 3, IrType::I64), Some(-3));
    }

    #[test]
    fn test_fold_binop_div_by_zero() {
        assert_eq!(fold_binop(IrBinOp::SDiv, 10, 0, IrType::I64), None);
        assert_eq!(fold_binop(IrBinOp::UDiv, 10, 0, IrType::I64), None);
        assert_eq!(fold_binop(IrBinOp::SRem, 10, 0, IrType::I64), None);
    }

    #[test]
    fn test_fold_binop_bitwise() {
        assert_eq!(fold_binop(IrBinOp::And, 0xFF, 0x0F, IrType::I64), Some(0x0F));
        assert_eq!(fold_binop(IrBinOp::Or, 0xF0, 0x0F, IrType::I64), Some(0xFF));
        assert_eq!(fold_binop(IrBinOp::Xor, 0xFF, 0xFF, IrType::I64), Some(0));
    }

    #[test]
    fn test_fold_binop_shift() {
        assert_eq!(fold_binop(IrBinOp::Shl, 1, 3, IrType::I64), Some(8));
        assert_eq!(fold_binop(IrBinOp::AShr, -8, 2, IrType::I64), Some(-2));
        // 64-bit LShr: -1 >> 32 = 0x00000000FFFFFFFF
        assert_eq!(fold_binop(IrBinOp::LShr, -1i64, 32, IrType::I64), Some(0xFFFFFFFF));
        // 32-bit LShr: -1 (as u32 = 0xFFFFFFFF) >> 31 = 1
        assert_eq!(fold_binop(IrBinOp::LShr, -1i64, 31, IrType::I32), Some(1));
        // 32-bit AShr: -1 >> 31 = -1 (sign bit propagates)
        assert_eq!(fold_binop(IrBinOp::AShr, -1i64, 31, IrType::I32), Some(-1));
    }

    #[test]
    fn test_fold_binop_udiv_32bit() {
        // 32-bit UDiv: -1 (as u32 = 0xFFFFFFFF = 4294967295) / 2 = 2147483647
        assert_eq!(fold_binop(IrBinOp::UDiv, -1i64, 2, IrType::I32), Some(2147483647));
        // 32-bit URem: -1 (as u32 = 4294967295) % 2 = 1
        assert_eq!(fold_binop(IrBinOp::URem, -1i64, 2, IrType::I32), Some(1));
    }

    #[test]
    fn test_fold_unaryop() {
        assert_eq!(fold_unaryop(IrUnaryOp::Neg, 5, IrType::I64), Some(-5));
        assert_eq!(fold_unaryop(IrUnaryOp::Not, 0, IrType::I64), Some(-1));
        // 32-bit popcount of -33 (0xFFFFFFDF) = 31 set bits
        assert_eq!(fold_unaryop(IrUnaryOp::Popcount, -33, IrType::I32), Some(31));
        // 64-bit popcount of -33 (0xFFFFFFFFFFFFFFDF) = 63 set bits
        assert_eq!(fold_unaryop(IrUnaryOp::Popcount, -33, IrType::I64), Some(63));
        // 32-bit CLZ of 1 = 31
        assert_eq!(fold_unaryop(IrUnaryOp::Clz, 1, IrType::I32), Some(31));
        // 64-bit CLZ of 1 = 63
        assert_eq!(fold_unaryop(IrUnaryOp::Clz, 1, IrType::I64), Some(63));
    }

    #[test]
    fn test_fold_cmp() {
        assert!(IrCmpOp::Eq.eval_i64(5, 5));
        assert!(!IrCmpOp::Eq.eval_i64(5, 6));
        assert!(IrCmpOp::Slt.eval_i64(-1, 0));
        // -1 as u64 is large, so unsigned comparison flips
        assert!(!IrCmpOp::Ult.eval_i64(-1i64, 0));
        assert!(IrCmpOp::Ugt.eval_i64(-1i64, 0));
    }

    #[test]
    fn test_fold_cast() {
        // Sign-extend i8 to i32
        assert_eq!(fold_cast(-1, IrType::I8, IrType::I32), -1);
        // Truncate i32 to i8 (signed)
        assert_eq!(fold_cast(256, IrType::I32, IrType::I8), 0);
        assert_eq!(fold_cast(255, IrType::I32, IrType::I8), -1);
    }

    #[test]
    fn test_fold_cast_unsigned_source() {
        // Zero-extend U8 to I32: 0xFF as u8 = 255, zero-extended to 255
        assert_eq!(fold_cast(-1, IrType::U8, IrType::I32), 255);
        // Zero-extend U8 to I64
        assert_eq!(fold_cast(-1, IrType::U8, IrType::I64), 255);
        // Zero-extend U16 to I32: 0xFFFF as u16 = 65535
        assert_eq!(fold_cast(-1, IrType::U16, IrType::I32), 65535);
        // Zero-extend U16 to I64
        assert_eq!(fold_cast(-1, IrType::U16, IrType::I64), 65535);
        // Zero-extend U32 to I64: 0xFFFFFFFF as u32 = 4294967295
        assert_eq!(fold_cast(-1, IrType::U32, IrType::I64), 4294967295);
    }

    #[test]
    fn test_fold_cast_unsigned_target() {
        // Truncate I32 to U8: 0x1FF & 0xFF = 0xFF = 255
        assert_eq!(fold_cast(0x1FF, IrType::I32, IrType::U8), 255);
        // Truncate I32 to U8: -1 & 0xFF = 255
        assert_eq!(fold_cast(-1, IrType::I32, IrType::U8), 255);
        // Truncate I32 to U16: 0x1FFFF & 0xFFFF = 65535
        assert_eq!(fold_cast(0x1FFFF, IrType::I32, IrType::U16), 65535);
        // Truncate I64 to U32: 0x1FFFFFFFF & 0xFFFFFFFF = 4294967295
        assert_eq!(fold_cast(0x1FFFFFFFF_i64, IrType::I64, IrType::U32), 4294967295);
        // Truncate I64 to U32: -1 & 0xFFFFFFFF = 4294967295
        assert_eq!(fold_cast(-1, IrType::I64, IrType::U32), 4294967295);
    }

    #[test]
    fn test_fold_cast_unsigned_to_unsigned() {
        // U8 255 to U16: zero-extend to 255
        assert_eq!(fold_cast(255, IrType::U8, IrType::U16), 255);
        // U16 to U8: truncate 0x1FF to 0xFF = 255
        assert_eq!(fold_cast(0x1FF, IrType::U16, IrType::U8), 255);
        // U32 to U8: truncate 0x1FF to 0xFF = 255
        assert_eq!(fold_cast(0x1FF, IrType::U32, IrType::U8), 255);
    }

    #[test]
    fn test_try_fold_binop() {
        let inst = Instruction::BinOp {
            dest: Value(0),
            op: IrBinOp::Add,
            lhs: Operand::Const(IrConst::I32(3)),
            rhs: Operand::Const(IrConst::I32(4)),
            ty: IrType::I32,
        };
        let folded = try_fold(&inst).unwrap();
        match folded {
            Instruction::Copy { src: Operand::Const(IrConst::I32(7)), .. } => {}
            _ => panic!("Expected Copy with constant 7"),
        }
    }

    #[test]
    fn test_no_fold_with_value_operand() {
        let inst = Instruction::BinOp {
            dest: Value(0),
            op: IrBinOp::Add,
            lhs: Operand::Value(Value(1)),
            rhs: Operand::Const(IrConst::I32(4)),
            ty: IrType::I32,
        };
        assert!(try_fold(&inst).is_none());
    }

    // === Float constant folding tests ===

    #[test]
    fn test_fold_float_neg_zero_add() {
        // -0.0 + 0.0 = +0.0 (IEEE 754)
        let result = fold_float_binop(IrBinOp::Add, -0.0, 0.0).unwrap();
        assert!(result == 0.0 && !result.is_sign_negative(), "Expected +0.0");
    }

    #[test]
    fn test_fold_float_neg_zero_add_self() {
        // -0.0 + -0.0 = -0.0 (IEEE 754)
        let result = fold_float_binop(IrBinOp::Add, -0.0, -0.0).unwrap();
        assert!(result == 0.0 && result.is_sign_negative(), "Expected -0.0");
    }

    #[test]
    fn test_fold_float_nan_mul_zero() {
        // NaN * 0.0 = NaN (IEEE 754)
        let result = fold_float_binop(IrBinOp::Mul, f64::NAN, 0.0).unwrap();
        assert!(result.is_nan(), "Expected NaN");
    }

    #[test]
    fn test_fold_float_neg_mul_zero() {
        // -5.0 * 0.0 = -0.0 (IEEE 754)
        let result = fold_float_binop(IrBinOp::Mul, -5.0, 0.0).unwrap();
        assert!(result == 0.0 && result.is_sign_negative(), "Expected -0.0");
    }

    #[test]
    fn test_fold_float_div_by_zero() {
        // 1.0 / 0.0 = +Inf (IEEE 754)
        let result = fold_float_binop(IrBinOp::SDiv, 1.0, 0.0).unwrap();
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[test]
    fn test_fold_float_neg_div_zero() {
        // -1.0 / 0.0 = -Inf (IEEE 754)
        let result = fold_float_binop(IrBinOp::SDiv, -1.0, 0.0).unwrap();
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_fold_float_cmp_neg_zero_eq() {
        // -0.0 == 0.0 is true (IEEE 754)
        assert!(IrCmpOp::Eq.eval_f64(-0.0, 0.0));
    }

    #[test]
    fn test_fold_float_cmp_nan_ne() {
        // NaN != NaN is true (IEEE 754)
        assert!(IrCmpOp::Ne.eval_f64(f64::NAN, f64::NAN));
        // NaN == NaN is false
        assert!(!IrCmpOp::Eq.eval_f64(f64::NAN, f64::NAN));
        // NaN < 1.0 is false
        assert!(!IrCmpOp::Slt.eval_f64(f64::NAN, 1.0));
    }

    #[test]
    fn test_fold_float_unary_neg() {
        let result = fold_float_unaryop(IrUnaryOp::Neg, 5.0).unwrap();
        assert_eq!(result, -5.0);
        // Negating -0.0 gives +0.0
        let result = fold_float_unaryop(IrUnaryOp::Neg, -0.0).unwrap();
        assert!(result == 0.0 && !result.is_sign_negative());
    }

    #[test]
    fn test_fold_float_cast_int_to_float() {
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Const(IrConst::I32(42)),
            from_ty: IrType::I32,
            to_ty: IrType::F64,
        };
        let result = try_fold(&inst).unwrap();
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::F64(v)), .. } => {
                assert_eq!(v, 42.0);
            }
            _ => panic!("Expected Copy with F64(42.0)"),
        }
    }

    #[test]
    fn test_fold_float_cast_float_to_int() {
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Const(IrConst::F64(3.125)),
            from_ty: IrType::F64,
            to_ty: IrType::I32,
        };
        let result = try_fold(&inst).unwrap();
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::I32(v)), .. } => {
                assert_eq!(v, 3);
            }
            _ => panic!("Expected Copy with I32(3)"),
        }
    }

    #[test]
    fn test_fold_float_cast_nan_to_int_no_fold() {
        // NaN to int should not fold
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Const(IrConst::F64(f64::NAN)),
            from_ty: IrType::F64,
            to_ty: IrType::I32,
        };
        assert!(try_fold(&inst).is_none());
    }

    #[test]
    fn test_fold_float_cast_overflow_to_int_no_fold() {
        // 1e20 exceeds i32 range, should not fold
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Const(IrConst::F64(1e20)),
            from_ty: IrType::F64,
            to_ty: IrType::I32,
        };
        assert!(try_fold(&inst).is_none());
    }

    #[test]
    fn test_fold_float_cast_inf_to_int_no_fold() {
        let inst = Instruction::Cast {
            dest: Value(0),
            src: Operand::Const(IrConst::F64(f64::INFINITY)),
            from_ty: IrType::F64,
            to_ty: IrType::I64,
        };
        assert!(try_fold(&inst).is_none());
    }

    #[test]
    fn test_fold_float_binop_instruction() {
        // Full instruction folding: F64(3.0) + F64(4.0) => F64(7.0)
        let inst = Instruction::BinOp {
            dest: Value(0),
            op: IrBinOp::Add,
            lhs: Operand::Const(IrConst::F64(3.0)),
            rhs: Operand::Const(IrConst::F64(4.0)),
            ty: IrType::F64,
        };
        let result = try_fold(&inst).unwrap();
        match result {
            Instruction::Copy { src: Operand::Const(IrConst::F64(v)), .. } => {
                assert_eq!(v, 7.0);
            }
            _ => panic!("Expected Copy with F64(7.0)"),
        }
    }
}
