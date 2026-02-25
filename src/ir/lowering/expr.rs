//! Expression lowering: converts AST Expr nodes to IR instructions.
//!
//! The main entry point is `lower_expr()`, which dispatches to focused helpers
//! for each expression category. Large subsystems are split into submodules:
//! - `expr_ops`: binary/unary ops, conditional, short-circuit, inc/dec
//! - `expr_access`: cast, compound literals, sizeof, address-of, member access, va_arg
//! - `expr_builtins`: __builtin_* intrinsics and FP classification
//! - `expr_atomics`: __atomic_* and __sync_* operations
//! - `expr_calls`: function call lowering, arguments, dispatch
//! - `expr_assign`: assignment, compound assignment, bitfield helpers

use std::rc::Rc;
use crate::frontend::parser::ast::{
    BinOp,
    Expr,
    UnaryOp,
};
use crate::ir::reexports::{
    Instruction,
    IrBinOp,
    IrCmpOp,
    IrConst,
    Operand,
    Value,
};
use crate::common::types::{AddressSpace, IrType, CType};
use super::lower::Lowerer;
use super::definitions::GlobalInfo;

impl Lowerer {
    /// Mask off the sign bit of a float value for truthiness testing.
    /// This ensures -0.0 is treated as falsy (same as +0.0) while NaN remains truthy.
    /// For F128 (long double), uses a proper float comparison against zero instead of
    /// bit-masking, because x87 80-bit values can't be reliably tested via integer AND.
    pub(super) fn mask_float_sign_for_truthiness(&mut self, val: Operand, float_ty: IrType) -> Operand {
        if !matches!(float_ty, IrType::F32 | IrType::F64 | IrType::F128) {
            return val;
        }
        if float_ty == IrType::F128 {
            // For F128 (long double), we cannot use bit-masking because the value
            // is stored as 80-bit x87 and the low 8 bytes are the mantissa (not f64 bits).
            // Instead, emit a proper F128 comparison against zero which uses x87 fucomip.
            let zero = Operand::Const(IrConst::long_double(0.0));
            let result = self.emit_cmp_val(IrCmpOp::Ne, val, zero, IrType::F128);
            return Operand::Value(result);
        }
        if crate::common::types::target_is_32bit() && float_ty == IrType::F64 {
            // On i686 (ILP32), the I64 bit-masking approach doesn't work because
            // the result is a 64-bit integer that gets truncated to 32 bits when
            // tested in conditional branches and comparisons. Use a proper F64
            // comparison against zero instead, which handles -0.0 correctly
            // (IEEE 754: +0.0 == -0.0) and NaN (unordered → truthy via setp).
            let zero = Operand::Const(IrConst::F64(0.0));
            let result = self.emit_cmp_val(IrCmpOp::Ne, val, zero, IrType::F64);
            return Operand::Value(result);
        }
        let (abs_mask, _, _, _, _) = Self::fp_masks(float_ty);
        let masked = self.emit_binop_val(IrBinOp::And, val, Operand::Const(IrConst::I64(abs_mask)), IrType::I64);
        if crate::common::types::target_is_32bit() {
            // On 32-bit targets, the masked result is typed as I64 but needs to
            // be reduced to a boolean for the conditional branch, since the
            // backend's branch-on-nonzero only tests the low 32 bits.
            let result = self.emit_cmp_val(IrCmpOp::Ne, Operand::Value(masked), Operand::Const(IrConst::I64(0)), IrType::I64);
            Operand::Value(result)
        } else {
            Operand::Value(masked)
        }
    }

    /// Lower a condition expression, ensuring floating-point values are properly
    /// tested for truthiness (masking sign bit so -0.0 is falsy).
    /// For complex types, tests (real != 0) || (imag != 0) per C11 6.3.1.2.
    pub(super) fn lower_condition_expr(&mut self, expr: &Expr) -> Operand {
        let expr_ct = self.expr_ctype(expr);
        if expr_ct.is_complex() {
            let val = self.lower_expr(expr);
            let ptr = self.operand_to_value(val);
            return self.lower_complex_to_bool(ptr, &expr_ct);
        }
        let expr_ty = self.infer_expr_type(expr);
        let val = self.lower_expr(expr);
        let val = self.mask_float_sign_for_truthiness(val, expr_ty);

        // On i686 (ILP32), 64-bit integer condition values (I64/U64 from long long
        // expressions) don't fit in a single 32-bit register. Reduce them to a
        // boolean (I8) via comparison against zero so that downstream CondBranch
        // and comparisons in short-circuit evaluation only need 32-bit values.
        // F64 conditions are already handled above by mask_float_sign_for_truthiness
        // which emits an F64 != 0.0 comparison on i686.
        if crate::common::types::target_is_32bit() && matches!(expr_ty, IrType::I64 | IrType::U64) {
            let zero = Operand::Const(IrConst::I64(0));
            let result = self.emit_cmp_val(IrCmpOp::Ne, val, zero, IrType::I64);
            return Operand::Value(result);
        }

        // Sub-int types (U8/I8/U16/I16): the narrow optimization pass may convert
        // widen-op-narrow patterns into narrow-type BinOps. The codegen only handles
        // I32/U32 and I64/U64 operand widths, so after a narrow BinOp the upper
        // register bits may contain stale data. CondBranch tests the full 64-bit
        // register (`testq %rax, %rax`), which would see those stale bits as nonzero.
        // Emit a Cast to widen the value to the machine word type, which forces
        // proper zero/sign-extension (movzbq/movzwq) to clean the upper bits.
        if matches!(expr_ty, IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16) {
            let widen_ty = crate::common::types::widened_op_type(expr_ty);
            let widened = self.emit_cast_val(val, expr_ty, widen_ty);
            return Operand::Value(widened);
        }

        val
    }

    pub(super) fn lower_expr(&mut self, expr: &Expr) -> Operand {
        match expr {
            // Literals
            Expr::IntLiteral(val, _) => Operand::Const(IrConst::I64(*val)),
            Expr::UIntLiteral(val, _) => Operand::Const(IrConst::I64(*val as i64)),
            Expr::LongLiteral(val, _) => Operand::Const(IrConst::I64(*val)),
            Expr::ULongLiteral(val, _) => Operand::Const(IrConst::I64(*val as i64)),
            Expr::LongLongLiteral(val, _) => Operand::Const(IrConst::I64(*val)),
            Expr::ULongLongLiteral(val, _) => Operand::Const(IrConst::I64(*val as i64)),
            Expr::FloatLiteral(val, _) => Operand::Const(IrConst::F64(*val)),
            Expr::FloatLiteralF32(val, _) => Operand::Const(IrConst::F32(*val as f32)),
            Expr::FloatLiteralLongDouble(val, bytes, _) => Operand::Const(IrConst::long_double_with_bytes(*val, *bytes)),
            Expr::CharLiteral(ch, _) => {
                // Sign-extend from signed char to int, matching GCC behavior.
                // '\xEF' should be -17, not 239, when char is signed.
                let val = *ch as u8 as i8 as i32;
                Operand::Const(IrConst::I32(val))
            }

            // Imaginary literals
            Expr::ImaginaryLiteral(val, _) => self.lower_imaginary_literal(*val, &CType::ComplexDouble),
            Expr::ImaginaryLiteralF32(val, _) => self.lower_imaginary_literal(*val, &CType::ComplexFloat),
            Expr::ImaginaryLiteralLongDouble(val, bytes, _) => self.lower_imaginary_literal_ld(*val, bytes, &CType::ComplexLongDouble),

            Expr::StringLiteral(s, _) => self.lower_string_literal(s, false),
            Expr::WideStringLiteral(s, _) => self.lower_string_literal(s, true),
            Expr::Char16StringLiteral(s, _) => self.lower_char16_string_literal(s),
            Expr::Identifier(name, _) => self.lower_identifier(name),
            Expr::BinaryOp(op, lhs, rhs, _) => self.lower_binary_op(op, lhs, rhs),
            Expr::UnaryOp(op, inner, _) => self.lower_unary_op(*op, inner),
            Expr::PostfixOp(op, inner, _) => self.lower_post_inc_dec(inner, *op),
            Expr::Assign(lhs, rhs, _) => self.lower_assign(lhs, rhs),
            Expr::CompoundAssign(op, lhs, rhs, _) => self.lower_compound_assign(op, lhs, rhs),
            Expr::FunctionCall(func, args, _) => self.lower_function_call(func, args),
            Expr::Conditional(cond, then_expr, else_expr, _) => {
                self.lower_conditional(cond, then_expr, else_expr)
            }
            Expr::GnuConditional(cond, else_expr, _) => {
                self.lower_gnu_conditional(cond, else_expr)
            }
            Expr::Cast(ref target_type, inner, _) => self.lower_cast(target_type, inner),
            Expr::CompoundLiteral(ref type_spec, ref init, _) => {
                self.lower_compound_literal(type_spec, init)
            }
            Expr::Sizeof(arg, _) => self.lower_sizeof(arg),
            Expr::Alignof(ref type_spec, _) => {
                let align = self.alignof_type(type_spec);
                Operand::Const(IrConst::I64(align as i64))
            }
            Expr::AlignofExpr(ref inner_expr, _) => {
                let align = self.alignof_expr(inner_expr);
                Operand::Const(IrConst::I64(align as i64))
            }
            Expr::GnuAlignof(ref type_spec, _) => {
                let align = self.preferred_alignof_type(type_spec);
                Operand::Const(IrConst::I64(align as i64))
            }
            Expr::GnuAlignofExpr(ref inner_expr, _) => {
                let align = self.preferred_alignof_expr(inner_expr);
                Operand::Const(IrConst::I64(align as i64))
            }
            Expr::AddressOf(inner, _) => self.lower_address_of(inner),
            Expr::Deref(inner, _) => self.lower_deref(inner),
            Expr::ArraySubscript(base, index, _) => self.lower_array_subscript(expr, base, index),
            Expr::MemberAccess(base_expr, field_name, _) => {
                self.lower_member_access(base_expr, field_name)
            }
            Expr::PointerMemberAccess(base_expr, field_name, _) => {
                self.lower_pointer_member_access(base_expr, field_name)
            }
            Expr::Comma(lhs, rhs, _) => {
                self.lower_expr(lhs);
                self.lower_expr(rhs)
            }
            Expr::StmtExpr(compound, _) => self.lower_stmt_expr(compound),
            Expr::VaArg(ap_expr, type_spec, _) => self.lower_va_arg(ap_expr, type_spec),
            Expr::GenericSelection(controlling, associations, _) => {
                self.lower_generic_selection(controlling, associations)
            }
            Expr::LabelAddr(label_name, _) => {
                let scoped_label = self.get_or_create_user_label(label_name);
                let dest = self.fresh_value();
                self.emit(Instruction::LabelAddr { dest, label: scoped_label });
                Operand::Value(dest)
            }
            Expr::BuiltinTypesCompatibleP(ref type1, ref type2, _) => {
                let result = self.eval_types_compatible(type1, type2);
                Operand::Const(IrConst::I64(result as i64))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Literal and identifier helpers
    // -----------------------------------------------------------------------

    fn lower_string_literal(&mut self, s: &str, wide: bool) -> Operand {
        let label = if wide {
            self.intern_wide_string_literal(s)
        } else {
            self.intern_string_literal(s)
        };
        let dest = self.fresh_value();
        self.emit(Instruction::GlobalAddr { dest, name: Rc::from(label) });
        Operand::Value(dest)
    }

    fn lower_char16_string_literal(&mut self, s: &str) -> Operand {
        let label = self.intern_char16_string_literal(s);
        let dest = self.fresh_value();
        self.emit(Instruction::GlobalAddr { dest, name: Rc::from(label) });
        Operand::Value(dest)
    }

    /// Emit an inline asm to read a global register variable (e.g., `register long x asm("rsp")`).
    /// Creates a temporary alloca, uses the inline asm output constraint `={regname}` to
    /// store the register value into the alloca, then loads the result.
    pub(super) fn read_global_register(&mut self, reg_name: &str, ty: IrType) -> Operand {
        // Create a temporary alloca for the inline asm output
        let tmp_alloca = self.fresh_value();
        self.emit(Instruction::Alloca { dest: tmp_alloca, ty, size: ty.size(), align: ty.align(), volatile: false });
        let output_constraint = format!("={{{}}}", reg_name);
        self.emit(Instruction::InlineAsm {
            template: String::new(),
            outputs: vec![(output_constraint, tmp_alloca, None)],
            inputs: vec![],
            clobbers: vec![],
            operand_types: vec![ty],
            goto_labels: vec![],
            input_symbols: vec![],
            seg_overrides: vec![AddressSpace::Default],
        });
        // Load the result from the alloca
        let result = self.fresh_value();
        self.emit(Instruction::Load { dest: result, ptr: tmp_alloca, ty , seg_override: AddressSpace::Default });
        Operand::Value(result)
    }

    fn load_global_var(&mut self, global_name: Rc<str>, ginfo: &GlobalInfo) -> Operand {
        let addr = self.fresh_value();
        self.emit(Instruction::GlobalAddr { dest: addr, name: global_name });
        if ginfo.is_array || ginfo.is_struct {
            return Operand::Value(addr);
        }
        if let Some(ref ct) = ginfo.c_type {
            if ct.is_complex() || ct.is_vector() {
                return Operand::Value(addr);
            }
            // Function names in expression context decay to function pointers (addresses).
            // Don't load from the function's address - just return the address itself.
            if matches!(ct, CType::Function(_)) {
                return Operand::Value(addr);
            }
        }
        let dest = self.fresh_value();
        self.emit(Instruction::Load { dest, ptr: addr, ty: ginfo.ty, seg_override: ginfo.address_space });
        Operand::Value(dest)
    }

    fn lower_identifier(&mut self, name: &str) -> Operand {
        if name == "__func__" || name == "__FUNCTION__" || name == "__PRETTY_FUNCTION__" {
            let name = self.func().name.clone(); return self.lower_string_literal(&name, false);
        }
        if name == "NULL" {
            return Operand::Const(IrConst::I64(0));
        }

        // Check locals first (shadows enum constants).
        // Extract only the cheap scalar fields we need, avoiding a full LocalInfo clone
        // (which includes StructLayout, CType, Vecs, and Strings on the heap).
        if let Some(info) = self.func_mut().locals.get(name) {
            let alloca = info.alloca;
            let ty = info.ty;
            let is_array = info.is_array;
            let is_struct = info.is_struct;
            let is_complex = info.c_type.as_ref().is_some_and(|ct| ct.is_complex());
            let is_vector = info.c_type.as_ref().is_some_and(|ct| ct.is_vector());
            let static_global_name = info.static_global_name.clone();
            let asm_register = info.asm_register.clone();
            let asm_register_has_init = info.asm_register_has_init;

            // Local register variable WITHOUT initializer AND not yet written by
            // inline asm: read the hardware register directly via inline asm.
            // E.g.:  register unsigned long sp __asm__("rsp"); return sp;
            //
            // Local register variables WITH initializer, or that have been written
            // by an inline asm output, read from their alloca like normal variables.
            // The register binding only affects inline asm constraint resolution.
            //
            // Note: asm_register_has_init is set to true either by the declaration
            // having an initializer, or by the inline asm lowering when the variable
            // is used as an output operand (see stmt_asm.rs).
            if let Some(ref reg_name) = asm_register {
                if !asm_register_has_init {
                    return self.read_global_register(reg_name, ty);
                }
            }

            if let Some(global_name) = static_global_name {
                let addr = self.fresh_value();
                self.emit(Instruction::GlobalAddr { dest: addr, name: Rc::from(global_name) });
                if is_array || is_struct || is_vector {
                    return Operand::Value(addr);
                }
                let dest = self.fresh_value();
                self.emit(Instruction::Load { dest, ptr: addr, ty , seg_override: AddressSpace::Default });
                return Operand::Value(dest);
            }
            if is_array || is_struct {
                return Operand::Value(alloca);
            }
            if is_complex || is_vector {
                return Operand::Value(alloca);
            }
            let dest = self.fresh_value();
            self.emit(Instruction::Load { dest, ptr: alloca, ty , seg_override: AddressSpace::Default });
            return Operand::Value(dest);
        }

        if let Some(&val) = self.types.enum_constants.get(name) {
            return Operand::Const(IrConst::I64(val));
        }

        if let Some(mangled) = self.func_mut().static_local_names.get(name).cloned() {
            if let Some(ginfo) = self.globals.get(&mangled).cloned() {
                return self.load_global_var(mangled, &ginfo);
            }
        }

        if let Some(ginfo) = self.globals.get(name).cloned() {
            // Global register variable: read the register directly via inline asm
            if let Some(ref reg_name) = ginfo.asm_register {
                return self.read_global_register(reg_name, ginfo.ty);
            }
            return self.load_global_var(Rc::from(name), &ginfo);
        }

        // Note: implicit declaration warnings are emitted during sema, not here.
        // Apply __asm__("label") linker symbol redirect if present.
        let resolved_name = self.asm_label_map.get(name)
            .cloned()
            .unwrap_or_else(|| Rc::from(name));
        let dest = self.fresh_value();
        self.emit(Instruction::GlobalAddr { dest, name: resolved_name });
        Operand::Value(dest)
    }


    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    /// Convert an Operand to a Value, copying constants to a temp if needed.
    pub(super) fn operand_to_value(&mut self, op: Operand) -> Value {
        match op {
            Operand::Value(v) => v,
            Operand::Const(_) => {
                let tmp = self.fresh_value();
                self.emit(Instruction::Copy { dest: tmp, src: op });
                tmp
            }
        }
    }

    pub(super) fn maybe_narrow(&mut self, val: Value, ty: IrType) -> Operand {
        let wt = crate::common::types::widened_op_type(ty);
        if ty != wt {
            let narrowed = self.emit_cast_val(Operand::Value(val), wt, ty);
            Operand::Value(narrowed)
        } else {
            Operand::Value(val)
        }
    }

    /// Map a byte size to the smallest IR integer type that fits.
    pub(super) fn ir_type_for_size(size: usize) -> IrType {
        if size <= 1 { IrType::I8 }
        else if size <= 2 { IrType::I16 }
        else if size <= 4 { IrType::I32 }
        else { IrType::I64 }
    }

    // -----------------------------------------------------------------------
    // Type inference for binary operations
    // -----------------------------------------------------------------------

    /// Infer the C semantic type of an expression for arithmetic conversions.
    ///
    /// This differs from `get_expr_type` in that it returns the C-level type
    /// (e.g., IntLiteral → I32 if it fits, CharLiteral → I8, comparisons → I32)
    /// whereas `get_expr_type` returns the IR storage type (literals → I64,
    /// comparisons → I64). Use this for binary operation type selection and
    /// arithmetic promotion decisions.
    ///
    /// For cases that don't differ, delegates to `get_expr_type`.
    pub(super) fn infer_expr_type(&self, expr: &Expr) -> IrType {
        match expr {
            // Literals: use C semantic types (narrower than get_expr_type's I64)
            Expr::IntLiteral(val, _) => {
                if *val >= i32::MIN as i64 && *val <= i32::MAX as i64 { IrType::I32 } else { IrType::I64 }
            }
            Expr::UIntLiteral(val, _) => {
                if *val <= u32::MAX as u64 { IrType::U32 } else { IrType::U64 }
            }
            // On ILP32, long is 32-bit; values that overflow promote to long long (64-bit)
            Expr::LongLiteral(val, _) => {
                if crate::common::types::target_is_32bit() {
                    if *val >= i32::MIN as i64 && *val <= i32::MAX as i64 { IrType::I32 } else { IrType::I64 }
                } else {
                    IrType::I64
                }
            }
            Expr::ULongLiteral(val, _) => {
                if crate::common::types::target_is_32bit() {
                    if *val <= u32::MAX as u64 { IrType::U32 } else { IrType::U64 }
                } else {
                    IrType::U64
                }
            }
            // long long is always 64-bit, regardless of target
            Expr::LongLongLiteral(_, _) => IrType::I64,
            Expr::ULongLongLiteral(_, _) => IrType::U64,
            Expr::CharLiteral(_, _) => IrType::I8,
            // Comparisons and logical ops produce C int (I32), not I64
            Expr::BinaryOp(op, lhs, rhs, _) => {
                if op.is_comparison() || matches!(op, BinOp::LogicalAnd | BinOp::LogicalOr) {
                    IrType::I32
                } else if matches!(op, BinOp::Shl | BinOp::Shr) {
                    Self::integer_promote(self.infer_expr_type(lhs))
                } else {
                    // Iterate left-skewed chains to avoid O(2^n) recursion
                    let rhs_ty = self.infer_expr_type(rhs);
                    let mut result = rhs_ty;
                    let mut cur = lhs.as_ref();
                    loop {
                        match cur {
                            Expr::BinaryOp(op2, inner_lhs, inner_rhs, _)
                                if !op2.is_comparison()
                                    && !matches!(op2, BinOp::LogicalAnd | BinOp::LogicalOr | BinOp::Shl | BinOp::Shr) =>
                            {
                                let r_ty = self.infer_expr_type(inner_rhs);
                                result = Self::common_type(result, r_ty);
                                cur = inner_lhs.as_ref();
                            }
                            _ => {
                                let l_ty = self.infer_expr_type(cur);
                                result = Self::common_type(result, l_ty);
                                break;
                            }
                        }
                    }
                    result
                }
            }
            Expr::UnaryOp(UnaryOp::LogicalNot, _, _) => IrType::I32,
            Expr::UnaryOp(UnaryOp::Neg | UnaryOp::BitNot | UnaryOp::Plus, inner, _) => {
                let inner_ty = self.infer_expr_type(inner);
                if inner_ty.is_float() { inner_ty } else { Self::integer_promote(inner_ty) }
            }
            // Recursive cases that must use infer_expr_type (not get_expr_type)
            // to propagate narrow literal types through the expression tree
            Expr::UnaryOp(_, inner, _) => self.infer_expr_type(inner),
            Expr::PostfixOp(_, inner, _) => self.infer_expr_type(inner),
            Expr::Assign(lhs, _, _) | Expr::CompoundAssign(_, lhs, _, _) => {
                self.infer_expr_type(lhs)
            }
            Expr::Conditional(_, then_expr, else_expr, _) => {
                Self::common_type(self.infer_expr_type(then_expr), self.infer_expr_type(else_expr))
            }
            Expr::GnuConditional(cond, else_expr, _) => {
                Self::common_type(self.infer_expr_type(cond), self.infer_expr_type(else_expr))
            }
            Expr::Comma(_, rhs, _) => self.infer_expr_type(rhs),
            // All other cases: delegate to get_expr_type (same result)
            _ => self.get_expr_type(expr),
        }
    }

    /// Determine common type for usual arithmetic conversions.
    /// On ILP32 targets, Ptr is equivalent to I32 (4 bytes); on LP64, Ptr maps to I64.
    pub(super) fn common_type(a: IrType, b: IrType) -> IrType {
        // Map Ptr to the target-appropriate signed integer type so that all
        // the size-based ranking below works correctly on both 32-bit and 64-bit.
        let a = if a == IrType::Ptr { crate::common::types::target_int_ir_type() } else { a };
        let b = if b == IrType::Ptr { crate::common::types::target_int_ir_type() } else { b };

        if a == IrType::I128 || a == IrType::U128 || b == IrType::I128 || b == IrType::U128 {
            if a == IrType::U128 || b == IrType::U128 { return IrType::U128; }
            return IrType::I128;
        }
        if a == IrType::F128 || b == IrType::F128 { return IrType::F128; }
        if a == IrType::F64 || b == IrType::F64 { return IrType::F64; }
        if a == IrType::F32 || b == IrType::F32 { return IrType::F32; }
        if a == IrType::I64 || a == IrType::U64
            || b == IrType::I64 || b == IrType::U64
        {
            if a == IrType::U64 || b == IrType::U64 { return IrType::U64; }
            return IrType::I64;
        }
        if a == IrType::U32 || b == IrType::U32 { return IrType::U32; }
        if a == IrType::I32 || b == IrType::I32 { return IrType::I32; }
        IrType::I32
    }

    /// Lower expression and cast to target type if needed.
    pub(super) fn lower_expr_with_type(&mut self, expr: &Expr, target_ty: IrType) -> Operand {
        let src = self.lower_expr(expr);
        let src_ty = self.get_expr_type(expr);
        self.emit_implicit_cast(src, src_ty, target_ty)
    }

    /// Insert an implicit type cast if src_ty differs from target_ty.
    pub(super) fn emit_implicit_cast(&mut self, src: Operand, src_ty: IrType, target_ty: IrType) -> Operand {
        if src_ty == target_ty { return src; }
        if target_ty == IrType::Ptr || target_ty == IrType::Void { return src; }
        if src_ty == IrType::Ptr && target_ty.is_integer() { return src; }
        // Pointer <-> float conversions are invalid in C; skip the cast
        // (sema should have already emitted an error for this)
        if src_ty == IrType::Ptr && target_ty.is_float() { return src; }
        if src_ty.is_float() && target_ty == IrType::Ptr { return src; }

        let needs_cast = (target_ty.is_float() && !src_ty.is_float())
            || (!target_ty.is_float() && src_ty.is_float())
            || (target_ty.is_float() && src_ty.is_float() && target_ty != src_ty)
            || (src_ty.is_integer() && target_ty.is_integer() && src_ty != target_ty);

        if needs_cast {
            let dest = self.emit_cast_val(src, src_ty, target_ty);
            return Operand::Value(dest);
        }
        src
    }

    /// Normalize a value for _Bool storage at the given source type.
    /// Emits (val != 0) for integers, (val != 0.0) for floats.
    /// This must be called BEFORE any truncation to avoid losing high bits.
    pub(super) fn emit_bool_normalize_typed(&mut self, val: Operand, src_ty: IrType) -> Operand {
        let zero = match src_ty {
            IrType::F32 => Operand::Const(IrConst::F32(0.0)),
            IrType::F64 => Operand::Const(IrConst::F64(0.0)),
            IrType::F128 => Operand::Const(IrConst::long_double(0.0)),
            _ => Operand::Const(IrConst::I64(0)),
        };
        let dest = self.emit_cmp_val(IrCmpOp::Ne, val, zero, src_ty);
        Operand::Value(dest)
    }

}
