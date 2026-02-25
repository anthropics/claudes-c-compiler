//! Complex number lowering: handles _Complex type operations.
//!
//! Complex types (_Complex float, _Complex double, _Complex long double) are
//! represented in IR as stack-allocated pairs of floats: {real, imag}.
//! A complex value is always passed around as a pointer to this pair.
//!
//! Layout:
//!   - _Complex float:       [f32 real, f32 imag] = 8 bytes, align 4
//!   - _Complex double:      [f64 real, f64 imag] = 16 bytes, align 8
//!   - _Complex long double: [f128 real, f128 imag] = 32 bytes, align 16 (x86-64)
//!     24 bytes, align 4  (i686)

use crate::frontend::parser::ast::{BinOp, Expr, UnaryOp};
use crate::ir::reexports::{
    GlobalInit,
    Instruction,
    IrBinOp,
    IrCmpOp,
    IrConst,
    IrUnaryOp,
    Operand,
    Value,
};
use crate::common::types::{AddressSpace, IrType, CType};
use super::lower::Lowerer;

impl Lowerer {
    /// Get the IR float type for a complex type's component.
    pub(super) fn complex_component_ir_type(ctype: &CType) -> IrType {
        match ctype {
            CType::ComplexFloat => IrType::F32,
            CType::ComplexDouble => IrType::F64,
            CType::ComplexLongDouble => IrType::F128,
            _ => IrType::F64, // fallback
        }
    }

    /// Get the component size in bytes for a complex type.
    /// On i686, long double is 12 bytes (80-bit x87 padded to 12); on x86-64 it's 16 bytes.
    pub(super) fn complex_component_size(ctype: &CType) -> usize {
        use crate::common::types::target_is_32bit;
        match ctype {
            CType::ComplexFloat => 4,
            CType::ComplexDouble => 8,
            CType::ComplexLongDouble => if target_is_32bit() { 12 } else { 16 },
            _ => 8,
        }
    }

    /// Convert a complex value (given as a pointer to {real, imag}) to _Bool.
    /// Returns (real != 0) || (imag != 0) per C11 6.3.1.2.
    pub(super) fn lower_complex_to_bool(&mut self, ptr: Value, ctype: &CType) -> Operand {
        let real = self.load_complex_real(ptr, ctype);
        let imag = self.load_complex_imag(ptr, ctype);
        let comp_ty = Self::complex_component_ir_type(ctype);
        let zero = Self::complex_zero(comp_ty);
        let real_nz = self.emit_cmp_val(IrCmpOp::Ne, real, zero, comp_ty);
        let imag_nz = self.emit_cmp_val(IrCmpOp::Ne, imag, zero, comp_ty);
        let result = self.emit_binop_val(IrBinOp::Or, Operand::Value(real_nz), Operand::Value(imag_nz), crate::common::types::target_int_ir_type());
        Operand::Value(result)
    }

    /// Get the zero constant for a complex component type.
    pub(super) fn complex_zero(comp_ty: IrType) -> Operand {
        match comp_ty {
            IrType::F32 => Operand::Const(IrConst::F32(0.0)),
            IrType::F128 => Operand::Const(IrConst::long_double(0.0)),
            _ => Operand::Const(IrConst::F64(0.0)),
        }
    }

    /// Allocate stack space for a complex value and return the alloca pointer.
    pub(super) fn alloca_complex(&mut self, ctype: &CType) -> Value {
        let size = ctype.size();
        let alloca = self.fresh_value();
        self.emit(Instruction::Alloca {
            dest: alloca,
            ty: IrType::Ptr,
            size,
            align: 0,
            volatile: false,
        });
        alloca
    }

    /// Store a complex value (real, imag) into an alloca.
    pub(super) fn store_complex_parts(&mut self, ptr: Value, real: Operand, imag: Operand, ctype: &CType) {
        let comp_ty = Self::complex_component_ir_type(ctype);
        let comp_size = Self::complex_component_size(ctype);

        // Store real part at offset 0
        self.emit(Instruction::Store {
            val: real,
            ptr,
            ty: comp_ty,
            seg_override: AddressSpace::Default,
        });

        // Store imag part at offset comp_size
        let imag_ptr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: imag_ptr,
            base: ptr,
            offset: Operand::Const(IrConst::ptr_int(comp_size as i64)),
            ty: IrType::I8, // byte offset
        });
        self.emit(Instruction::Store { val: imag, ptr: imag_ptr, ty: comp_ty,
         seg_override: AddressSpace::Default });
    }

    /// Load the real part of a complex value from a pointer.
    pub(super) fn load_complex_real(&mut self, ptr: Value, ctype: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(ctype);
        let dest = self.fresh_value();
        self.emit(Instruction::Load {
            dest,
            ptr,
            ty: comp_ty,
            seg_override: AddressSpace::Default,
        });
        Operand::Value(dest)
    }

    /// Load the imaginary part of a complex value from a pointer.
    pub(super) fn load_complex_imag(&mut self, ptr: Value, ctype: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(ctype);
        let comp_size = Self::complex_component_size(ctype);
        let imag_ptr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: imag_ptr,
            base: ptr,
            offset: Operand::Const(IrConst::ptr_int(comp_size as i64)),
            ty: IrType::I8,
        });
        let dest = self.fresh_value();
        self.emit(Instruction::Load {
            dest,
            ptr: imag_ptr,
            ty: comp_ty,
            seg_override: AddressSpace::Default,
        });
        Operand::Value(dest)
    }

    /// Lower an imaginary literal to a complex value on the stack.
    /// Creates {0.0, val} and returns a pointer to it.
    pub(super) fn lower_imaginary_literal(&mut self, val: f64, ctype: &CType) -> Operand {
        let alloca = self.alloca_complex(ctype);
        let comp_ty = Self::complex_component_ir_type(ctype);
        let zero = match comp_ty {
            IrType::F32 => Operand::Const(IrConst::F32(0.0)),
            IrType::F128 => Operand::Const(IrConst::long_double(0.0)),
            _ => Operand::Const(IrConst::F64(0.0)),
        };
        let imag_val = match comp_ty {
            IrType::F32 => Operand::Const(IrConst::F32(val as f32)),
            IrType::F128 => Operand::Const(IrConst::long_double(val)),
            _ => Operand::Const(IrConst::F64(val)),
        };
        self.store_complex_parts(alloca, zero, imag_val, ctype);
        Operand::Value(alloca)
    }

    /// Lower a long double imaginary literal, preserving full x87 precision bytes.
    pub(super) fn lower_imaginary_literal_ld(&mut self, val: f64, bytes: &[u8; 16], ctype: &CType) -> Operand {
        let alloca = self.alloca_complex(ctype);
        let zero = Operand::Const(IrConst::long_double(0.0));
        let imag_val = Operand::Const(IrConst::long_double_with_bytes(val, *bytes));
        self.store_complex_parts(alloca, zero, imag_val, ctype);
        Operand::Value(alloca)
    }

    /// Lower __real__ expr - extract real part of a complex expression.
    pub(super) fn lower_complex_real_part(&mut self, inner: &Expr) -> Operand {
        let inner_ctype = self.expr_ctype(inner);
        if inner_ctype.is_complex() {
            let ptr = self.lower_expr(inner);
            let ptr_val = self.operand_to_value(ptr);
            self.load_complex_real(ptr_val, &inner_ctype)
        } else {
            // __real__ on a non-complex value just returns the value
            self.lower_expr(inner)
        }
    }

    /// Lower __imag__ expr - extract imaginary part of a complex expression.
    pub(super) fn lower_complex_imag_part(&mut self, inner: &Expr) -> Operand {
        let inner_ctype = self.expr_ctype(inner);
        if inner_ctype.is_complex() {
            let ptr = self.lower_expr(inner);
            let ptr_val = self.operand_to_value(ptr);
            self.load_complex_imag(ptr_val, &inner_ctype)
        } else {
            // __imag__ on a non-complex value returns 0
            let ir_ty = self.get_expr_type(inner);
            if ir_ty.is_float() {
                match ir_ty {
                    IrType::F32 => Operand::Const(IrConst::F32(0.0)),
                    IrType::F128 => Operand::Const(IrConst::long_double(0.0)),
                    _ => Operand::Const(IrConst::F64(0.0)),
                }
            } else {
                Operand::Const(IrConst::ptr_int(0))
            }
        }
    }

    /// Lower a componentwise complex binary operation (add or sub).
    /// Both real and imaginary parts use the same `op`:
    ///   add: (a+bi) + (c+di) = (a+c) + (b+d)i
    ///   sub: (a+bi) - (c+di) = (a-c) + (b-d)i
    fn lower_complex_componentwise_binop(&mut self, op: IrBinOp, lhs_ptr: Value, rhs_ptr: Value, ctype: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(ctype);

        let lr = self.load_complex_real(lhs_ptr, ctype);
        let li = self.load_complex_imag(lhs_ptr, ctype);
        let rr = self.load_complex_real(rhs_ptr, ctype);
        let ri = self.load_complex_imag(rhs_ptr, ctype);

        let real_result = self.emit_binop_val(op, lr, rr, comp_ty);
        let imag_result = self.emit_binop_val(op, li, ri, comp_ty);

        let result = self.alloca_complex(ctype);
        self.store_complex_parts(result, Operand::Value(real_result), Operand::Value(imag_result), ctype);
        Operand::Value(result)
    }

    /// Lower complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
    pub(super) fn lower_complex_add(&mut self, lhs_ptr: Value, rhs_ptr: Value, ctype: &CType) -> Operand {
        self.lower_complex_componentwise_binop(IrBinOp::Add, lhs_ptr, rhs_ptr, ctype)
    }

    /// Lower complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
    pub(super) fn lower_complex_sub(&mut self, lhs_ptr: Value, rhs_ptr: Value, ctype: &CType) -> Operand {
        self.lower_complex_componentwise_binop(IrBinOp::Sub, lhs_ptr, rhs_ptr, ctype)
    }

    /// Lower real - complex: real - (c+di) = (real-c) + (-d)i
    /// Uses negation for the imaginary part instead of 0.0-d to preserve
    /// IEEE 754 negative zero: -(+0.0) = -0.0, but +0.0 - (+0.0) = +0.0.
    pub(super) fn lower_real_minus_complex(&mut self, real_val: Operand, real_type: &CType, rhs_ptr: Value, ctype: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(ctype);

        // Cast the real scalar to the component type
        let real_ir = IrType::from_ctype(real_type);
        let converted_real = if real_ir != comp_ty {
            let dest = self.emit_cast_val(real_val, real_ir, comp_ty);
            Operand::Value(dest)
        } else {
            real_val
        };

        let rr = self.load_complex_real(rhs_ptr, ctype);
        let ri = self.load_complex_imag(rhs_ptr, ctype);

        // real_part = real - rhs_real
        let real_result = self.emit_binop_val(IrBinOp::Sub, converted_real, rr, comp_ty);
        // imag_part = -rhs_imag (not 0.0 - rhs_imag)
        let neg_i = self.fresh_value();
        self.emit(Instruction::UnaryOp { dest: neg_i, op: IrUnaryOp::Neg, src: ri, ty: comp_ty });

        let result = self.alloca_complex(ctype);
        self.store_complex_parts(result, Operand::Value(real_result), Operand::Value(neg_i), ctype);
        Operand::Value(result)
    }

    /// Lower complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    pub(super) fn lower_complex_mul(&mut self, lhs_ptr: Value, rhs_ptr: Value, ctype: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(ctype);

        let a = self.load_complex_real(lhs_ptr, ctype);
        let b = self.load_complex_imag(lhs_ptr, ctype);
        let c = self.load_complex_real(rhs_ptr, ctype);
        let d = self.load_complex_imag(rhs_ptr, ctype);

        // ac
        let ac = self.emit_binop_val(IrBinOp::Mul, a, c, comp_ty);
        // bd
        let bd = self.emit_binop_val(IrBinOp::Mul, b, d, comp_ty);
        // ad
        let ad = self.emit_binop_val(IrBinOp::Mul, a, d, comp_ty);
        // bc
        let bc = self.emit_binop_val(IrBinOp::Mul, b, c, comp_ty);

        // real = ac - bd
        let real = self.emit_binop_val(IrBinOp::Sub, Operand::Value(ac), Operand::Value(bd), comp_ty);
        // imag = ad + bc
        let imag = self.emit_binop_val(IrBinOp::Add, Operand::Value(ad), Operand::Value(bc), comp_ty);

        let result = self.alloca_complex(ctype);
        self.store_complex_parts(result, Operand::Value(real), Operand::Value(imag), ctype);
        Operand::Value(result)
    }

    /// Lower complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
    pub(super) fn lower_complex_div(&mut self, lhs_ptr: Value, rhs_ptr: Value, ctype: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(ctype);

        let a = self.load_complex_real(lhs_ptr, ctype);
        let b = self.load_complex_imag(lhs_ptr, ctype);
        let c = self.load_complex_real(rhs_ptr, ctype);
        let d = self.load_complex_imag(rhs_ptr, ctype);

        // denom = c*c + d*d
        let cc = self.emit_binop_val(IrBinOp::Mul, c, c, comp_ty);
        let dd = self.emit_binop_val(IrBinOp::Mul, d, d, comp_ty);
        let denom = self.emit_binop_val(IrBinOp::Add, Operand::Value(cc), Operand::Value(dd), comp_ty);

        // real_num = a*c + b*d
        let ac = self.emit_binop_val(IrBinOp::Mul, a, c, comp_ty);
        let bd = self.emit_binop_val(IrBinOp::Mul, b, d, comp_ty);
        let real_num = self.emit_binop_val(IrBinOp::Add, Operand::Value(ac), Operand::Value(bd), comp_ty);

        // imag_num = b*c - a*d
        let bc = self.emit_binop_val(IrBinOp::Mul, b, c, comp_ty);
        let ad = self.emit_binop_val(IrBinOp::Mul, a, d, comp_ty);
        let imag_num = self.emit_binop_val(IrBinOp::Sub, Operand::Value(bc), Operand::Value(ad), comp_ty);

        // real = real_num / denom, imag = imag_num / denom
        let real = self.emit_binop_val(IrBinOp::SDiv, Operand::Value(real_num), Operand::Value(denom), comp_ty);
        let imag = self.emit_binop_val(IrBinOp::SDiv, Operand::Value(imag_num), Operand::Value(denom), comp_ty);

        let result = self.alloca_complex(ctype);
        self.store_complex_parts(result, Operand::Value(real), Operand::Value(imag), ctype);
        Operand::Value(result)
    }

    /// Lower complex negation: -(a+bi) = (-a) + (-b)i
    pub(super) fn lower_complex_neg(&mut self, ptr: Value, ctype: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(ctype);

        let r = self.load_complex_real(ptr, ctype);
        let i = self.load_complex_imag(ptr, ctype);

        let neg_r = self.fresh_value();
        self.emit(Instruction::UnaryOp { dest: neg_r, op: IrUnaryOp::Neg, src: r, ty: comp_ty });
        let neg_i = self.fresh_value();
        self.emit(Instruction::UnaryOp { dest: neg_i, op: IrUnaryOp::Neg, src: i, ty: comp_ty });

        let result = self.alloca_complex(ctype);
        self.store_complex_parts(result, Operand::Value(neg_r), Operand::Value(neg_i), ctype);
        Operand::Value(result)
    }

    /// Lower complex conjugate: conj(a+bi) = a + (-b)i
    pub(super) fn lower_complex_conj(&mut self, inner: &Expr) -> Operand {
        let inner_ctype = self.expr_ctype(inner);
        if inner_ctype.is_complex() {
            let ptr = self.lower_expr(inner);
            let ptr_val = self.operand_to_value(ptr);
            let comp_ty = Self::complex_component_ir_type(&inner_ctype);

            let r = self.load_complex_real(ptr_val, &inner_ctype);
            let i = self.load_complex_imag(ptr_val, &inner_ctype);

            let neg_i = self.fresh_value();
            self.emit(Instruction::UnaryOp { dest: neg_i, op: IrUnaryOp::Neg, src: i, ty: comp_ty });

            let result = self.alloca_complex(&inner_ctype);
            self.store_complex_parts(result, r, Operand::Value(neg_i), &inner_ctype);
            Operand::Value(result)
        } else {
            // conj on a non-complex value just returns the value
            self.lower_expr(inner)
        }
    }

    /// Create a complex value from a real scalar (real part = scalar, imag = 0).
    pub(super) fn real_to_complex(&mut self, real_val: Operand, real_type: &CType, complex_type: &CType) -> Operand {
        let real_ir = IrType::from_ctype(real_type);
        self.scalar_to_complex(real_val, real_ir, complex_type)
    }

    /// Create a complex value from a scalar given its IR type (real part = scalar, imag = 0).
    /// This is preferred over real_to_complex when the CType may be inaccurate (e.g., for
    /// function calls where expr_ctype may not correctly resolve the return type).
    pub(super) fn scalar_to_complex(&mut self, real_val: Operand, src_ir_ty: IrType, complex_type: &CType) -> Operand {
        let comp_ty = Self::complex_component_ir_type(complex_type);
        let zero = match comp_ty {
            IrType::F32 => Operand::Const(IrConst::F32(0.0)),
            IrType::F128 => Operand::Const(IrConst::long_double(0.0)),
            _ => Operand::Const(IrConst::F64(0.0)),
        };

        // Convert the real value to the component type if needed
        let converted_real = if src_ir_ty != comp_ty {
            let dest = self.emit_cast_val(real_val, src_ir_ty, comp_ty);
            Operand::Value(dest)
        } else {
            real_val
        };

        let result = self.alloca_complex(complex_type);
        self.store_complex_parts(result, converted_real, zero, complex_type);
        Operand::Value(result)
    }

    /// Convert a complex value from one complex type to another (e.g., ComplexFloat -> ComplexDouble).
    pub(super) fn complex_to_complex(&mut self, ptr: Value, from_type: &CType, to_type: &CType) -> Operand {
        let from_comp = Self::complex_component_ir_type(from_type);
        let to_comp = Self::complex_component_ir_type(to_type);

        let r = self.load_complex_real(ptr, from_type);
        let i = self.load_complex_imag(ptr, from_type);

        let conv_r = if from_comp != to_comp {
            let dest = self.emit_cast_val(r, from_comp, to_comp);
            Operand::Value(dest)
        } else {
            r
        };
        let conv_i = if from_comp != to_comp {
            let dest = self.emit_cast_val(i, from_comp, to_comp);
            Operand::Value(dest)
        } else {
            i
        };

        let result = self.alloca_complex(to_type);
        self.store_complex_parts(result, conv_r, conv_i, to_type);
        Operand::Value(result)
    }

    /// Determine the CType of an expression, with complex-type-aware overrides.
    ///
    /// Delegates to `get_expr_ctype` for most expressions, but adds handling for
    /// complex-specific patterns that `get_expr_ctype` doesn't cover:
    /// - `__builtin_complex`: return type depends on argument type
    /// - `conj/conjf/conjl`: return type matches argument, not function signature
    /// - Arithmetic on complex operands: uses `common_complex_type` for promotion
    /// - Conditionals with complex branches: complex-aware type resolution
    pub(super) fn expr_ctype(&self, expr: &Expr) -> CType {
        // Handle complex-specific cases that get_expr_ctype doesn't cover
        match expr {
            Expr::BinaryOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div
                | BinOp::Mod | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor
                | BinOp::Shl | BinOp::Shr, lhs, rhs, _) => {
                let lt = self.expr_ctype(lhs);
                let rt = self.expr_ctype(rhs);
                // GCC vector extensions: if either operand is a vector, result is that vector type
                if lt.is_vector() {
                    return lt;
                }
                if rt.is_vector() {
                    return rt;
                }
                if lt.is_complex() || rt.is_complex() {
                    return self.common_complex_type(&lt, &rt);
                }
                // Fall through to get_expr_ctype for non-complex/non-vector arithmetic
            }
            Expr::FunctionCall(callee, args, _) => {
                if let Expr::Identifier(name, _) = callee.as_ref() {
                    // __builtin_complex returns complex type based on argument types
                    if &**name == "__builtin_complex" {
                        if let Some(first_arg) = args.first() {
                            let arg_ct = self.expr_ctype(first_arg);
                            return match arg_ct {
                                CType::Float => CType::ComplexFloat,
                                CType::LongDouble => CType::ComplexLongDouble,
                                _ => CType::ComplexDouble,
                            };
                        }
                        return CType::ComplexDouble;
                    }
                    // conj/conjf/conjl preserve the argument's complex type, but
                    // the registered function signature always says ComplexDouble.
                    if matches!(&**name, "conj" | "conjf" | "conjl"
                        | "__builtin_conj" | "__builtin_conjf" | "__builtin_conjl") {
                        if let Some(first_arg) = args.first() {
                            let arg_ct = self.expr_ctype(first_arg);
                            if arg_ct.is_complex() {
                                return arg_ct;
                            }
                        }
                    }
                }
                // Fall through to get_expr_ctype for other function calls
            }
            Expr::Conditional(_, then_expr, else_expr, _) => {
                let then_ct = self.expr_ctype(then_expr);
                let else_ct = self.expr_ctype(else_expr);
                if then_ct.is_complex() || else_ct.is_complex() {
                    return self.common_complex_type(&then_ct, &else_ct);
                }
                // Fall through to get_expr_ctype for non-complex conditionals
            }
            Expr::GnuConditional(cond, else_expr, _) => {
                let cond_ct = self.expr_ctype(cond);
                let else_ct = self.expr_ctype(else_expr);
                if cond_ct.is_complex() || else_ct.is_complex() {
                    return self.common_complex_type(&cond_ct, &else_ct);
                }
                // Fall through to get_expr_ctype for non-complex conditionals
            }
            _ => {}
        }
        // Delegate to the general-purpose CType inference (lowerer + sema fallback)
        self.get_expr_ctype(expr).unwrap_or(CType::Int)
    }

    /// Determine the common complex type for binary operations.
    /// Per C11 6.3.1.8: integer types rank below float in the type hierarchy,
    /// so they adopt the complex type of the other operand rather than forcing
    /// promotion to ComplexDouble.
    pub(super) fn common_complex_type(&self, a: &CType, b: &CType) -> CType {
        // Per C11 6.3.1.8: if one operand is complex and the other is real/integer,
        // the result type is determined by the "usual arithmetic conversions" applied
        // to the real type of the complex operand and the type of the other operand.
        // Integer types smaller than double don't force promotion to ComplexDouble;
        // instead, the result keeps the complex type unless the other operand is wider.
        let rank_of = |ct: &CType| -> u8 {
            match ct {
                CType::ComplexLongDouble | CType::LongDouble => 3,
                CType::ComplexDouble | CType::Double => 2,
                CType::ComplexFloat | CType::Float => 1,
                // Integer types: C standard says integer types convert to double
                // when combined with a complex type that is double or wider,
                // but when combined with complex float, they convert to float.
                // However, long long / unsigned long may exceed float precision.
                // For simplicity and correctness with most tests, we use the
                // C rule: "the corresponding real type of the complex" wins unless
                // the real operand has higher floating rank.
                CType::Long | CType::ULong | CType::LongLong | CType::ULongLong => 0,
                _ => 0, // char, short, int, uint -> rank 0 (no floating rank)
            }
        };

        let a_rank = rank_of(a);
        let b_rank = rank_of(b);

        // Result complex type is the wider of the two ranks.
        // If one is complex and the other is integer (rank 0), the complex type wins.
        let max_rank = a_rank.max(b_rank);
        match max_rank {
            3 => CType::ComplexLongDouble,
            2 => CType::ComplexDouble,
            1 => CType::ComplexFloat,
            _ => {
                // Both are integer types - promote to ComplexDouble (C default)
                // But check if one is already complex
                if a.is_complex() { return a.clone(); }
                if b.is_complex() { return b.clone(); }
                CType::ComplexDouble
            }
        }
    }

    /// Convert a value to a complex type, handling both real-to-complex and complex-to-complex.
    pub(super) fn convert_to_complex(&mut self, val: Operand, from_type: &CType, to_type: &CType) -> Operand {
        if from_type.is_complex() {
            if from_type == to_type {
                val
            } else {
                let ptr = self.operand_to_value(val);
                self.complex_to_complex(ptr, from_type, to_type)
            }
        } else {
            // Real to complex
            self.real_to_complex(val, from_type, to_type)
        }
    }

    /// Lower assignment to a complex variable.
    pub(super) fn lower_complex_assign(&mut self, lhs: &Expr, rhs: &Expr, lhs_ct: &CType) -> Operand {
        let rhs_ct = self.expr_ctype(rhs);
        let rhs_val = self.lower_expr(rhs);

        // Convert RHS to the LHS complex type
        let rhs_complex = if rhs_ct.is_complex() {
            if rhs_ct == *lhs_ct {
                rhs_val
            } else {
                self.convert_to_complex(rhs_val, &rhs_ct, lhs_ct)
            }
        } else {
            // Real value -> complex
            self.real_to_complex(rhs_val, &rhs_ct, lhs_ct)
        };

        // Get LHS pointer
        let lhs_ptr = self.lower_complex_lvalue(lhs);

        // Copy real and imag parts from rhs to lhs
        let rhs_ptr = self.operand_to_value(rhs_complex);
        let comp_size = Self::complex_component_size(lhs_ct);
        let total_size = comp_size * 2;
        self.emit(Instruction::Memcpy {
            dest: lhs_ptr,
            src: rhs_ptr,
            size: total_size,
        });

        // Return the lhs pointer
        Operand::Value(lhs_ptr)
    }

    /// Get the lvalue (address) of a complex expression for assignment purposes.
    pub(super) fn lower_complex_lvalue(&mut self, expr: &Expr) -> Value {
        match expr {
            Expr::Identifier(name, _) => {
                // Look up in locals
                if let Some(info) = self.func_state.as_ref().and_then(|fs| fs.locals.get(name).cloned()) {
                    return info.alloca;
                }
                // Check static locals
                if let Some(mangled) = self.func_state.as_ref().and_then(|fs| fs.static_local_names.get(name).cloned()) {
                    let addr = self.fresh_value();
                    self.emit(Instruction::GlobalAddr { dest: addr, name: mangled.clone() });
                    return addr;
                }
                // Global
                let addr = self.fresh_value();
                self.emit(Instruction::GlobalAddr { dest: addr, name: name.clone() });
                addr
            }
            Expr::Deref(inner, _) => {
                let ptr = self.lower_expr(inner);
                self.operand_to_value(ptr)
            }
            Expr::ArraySubscript(base, index, _) => {
                let base_val = self.lower_expr(base);
                let idx_val = self.lower_expr(index);
                let ct = self.expr_ctype(expr);
                let elem_size = self.resolve_ctype_size(&ct);
                let scaled = self.scale_index(idx_val, elem_size);
                
                self.emit_binop_val(IrBinOp::Add, base_val, scaled, crate::common::types::target_int_ir_type())
            }
            Expr::MemberAccess(base, field, _) => {
                // Get base struct address and add field offset
                let base_ct = self.expr_ctype(base);
                if let CType::Struct(ref key) | CType::Union(ref key) = base_ct {
                    let field_offset = {
                        let layouts = self.types.borrow_struct_layouts();
                        layouts.get(&**key)
                            .and_then(|layout| layout.field_offset(field, &*layouts))
                            .map(|(offset, _)| offset)
                    };
                    if let Some(offset) = field_offset {
                        let base_ptr = self.lower_complex_lvalue(base);
                        if offset == 0 {
                            return base_ptr;
                        }
                        let dest = self.fresh_value();
                        self.emit(Instruction::GetElementPtr {
                            dest, base: base_ptr,
                            offset: Operand::Const(IrConst::ptr_int(offset as i64)),
                            ty: IrType::I8,
                        });
                        return dest;
                    }
                }
                // Fallback
                let val = self.lower_expr(expr);
                self.operand_to_value(val)
            }
            Expr::PointerMemberAccess(base, field, _) => {
                let base_val = self.lower_expr(base);
                let base_ptr = self.operand_to_value(base_val);
                let base_ct = self.expr_ctype(base);
                if let CType::Pointer(ref inner, _) = base_ct {
                    if let CType::Struct(ref key) | CType::Union(ref key) = **inner {
                        let field_offset = {
                            let layouts = self.types.borrow_struct_layouts();
                            layouts.get(&**key)
                                .and_then(|layout| layout.field_offset(field, &*layouts))
                                .map(|(offset, _)| offset)
                        };
                        if let Some(offset) = field_offset {
                            if offset == 0 {
                                return base_ptr;
                            }
                            let dest = self.fresh_value();
                            self.emit(Instruction::GetElementPtr {
                                dest, base: base_ptr,
                                offset: Operand::Const(IrConst::ptr_int(offset as i64)),
                                ty: IrType::I8,
                            });
                            return dest;
                        }
                    }
                }
                base_ptr
            }
            _ => {
                // Fallback: treat expression result as pointer
                let val = self.lower_expr(expr);
                self.operand_to_value(val)
            }
        }
    }

    /// Lower complex equality/inequality comparison.
    /// Two complex numbers are equal iff both real and imaginary parts are equal.
    pub(super) fn lower_complex_comparison(&mut self, op: &BinOp, lhs: &Expr, rhs: &Expr, lhs_ct: &CType, rhs_ct: &CType) -> Operand {
        let result_ct = self.common_complex_type(lhs_ct, rhs_ct);
        let comp_ty = Self::complex_component_ir_type(&result_ct);

        // Lower and convert both to common complex type
        let lhs_val = self.lower_expr(lhs);
        let rhs_val = self.lower_expr(rhs);
        let lhs_complex = self.convert_to_complex(lhs_val, lhs_ct, &result_ct);
        let rhs_complex = self.convert_to_complex(rhs_val, rhs_ct, &result_ct);
        let lhs_ptr = self.operand_to_value(lhs_complex);
        let rhs_ptr = self.operand_to_value(rhs_complex);

        // Compare real parts
        let lr = self.load_complex_real(lhs_ptr, &result_ct);
        let rr = self.load_complex_real(rhs_ptr, &result_ct);
        let real_eq = self.emit_cmp_val(IrCmpOp::Eq, lr, rr, comp_ty);

        // Compare imag parts
        let li = self.load_complex_imag(lhs_ptr, &result_ct);
        let ri = self.load_complex_imag(rhs_ptr, &result_ct);
        let imag_eq = self.emit_cmp_val(IrCmpOp::Eq, li, ri, comp_ty);

        // Combine: both must be equal for == (either not equal for !=)
        let int_ty = crate::common::types::target_int_ir_type();
        let result = match op {
            BinOp::Eq => {
                self.emit_binop_val(IrBinOp::And, Operand::Value(real_eq), Operand::Value(imag_eq), int_ty)
            }
            BinOp::Ne => {
                // a != b iff !(a.re == b.re && a.im == b.im)
                let both_eq = self.emit_binop_val(IrBinOp::And, Operand::Value(real_eq), Operand::Value(imag_eq), int_ty);
                self.emit_cmp_val(IrCmpOp::Eq, Operand::Value(both_eq), Operand::Const(IrConst::ptr_int(0)), int_ty)
            }
            _ => unreachable!("complex comparison only supports Eq/Ne, got: {:?}", op),
        };
        Operand::Value(result)
    }

    /// Evaluate a complex expression for global initialization.
    /// Returns a GlobalInit::Array with [real, imag] constant values.
    pub(super) fn eval_complex_global_init(&self, expr: &Expr, target_ctype: &CType) -> Option<GlobalInit> {
        let (real, imag) = self.eval_complex_const(expr)?;
        match target_ctype {
            CType::ComplexFloat => Some(GlobalInit::Array(vec![
                IrConst::F32(real as f32),
                IrConst::F32(imag as f32),
            ])),
            CType::ComplexLongDouble => Some(GlobalInit::Array(vec![
                IrConst::long_double(real),
                IrConst::long_double(imag),
            ])),
            _ => Some(GlobalInit::Array(vec![
                IrConst::F64(real),
                IrConst::F64(imag),
            ])),
        }
    }

    /// Public wrapper for eval_complex_const, used by global_init for complex array elements.
    pub(super) fn eval_complex_const_public(&self, expr: &Expr) -> Option<(f64, f64)> {
        self.eval_complex_const(expr)
    }

    /// Try to evaluate a complex constant expression, returning (real, imag) as f64.
    fn eval_complex_const(&self, expr: &Expr) -> Option<(f64, f64)> {
        match expr {
            // Imaginary literals: 4.0i -> (0, 4.0)
            Expr::ImaginaryLiteral(val, _) => Some((0.0, *val)),
            Expr::ImaginaryLiteralF32(val, _) => Some((0.0, *val)),
            Expr::ImaginaryLiteralLongDouble(val, _, _) => Some((0.0, *val)),
            // Binary add: real + imag, or complex + complex
            Expr::BinaryOp(BinOp::Add, lhs, rhs, _) => {
                let l = self.eval_complex_const(lhs)?;
                let r = self.eval_complex_const(rhs)?;
                Some((l.0 + r.0, l.1 + r.1))
            }
            // Binary sub: real - imag
            Expr::BinaryOp(BinOp::Sub, lhs, rhs, _) => {
                let l = self.eval_complex_const(lhs)?;
                let r = self.eval_complex_const(rhs)?;
                Some((l.0 - r.0, l.1 - r.1))
            }
            // Binary mul: for simple cases like 2*I or val*I
            Expr::BinaryOp(BinOp::Mul, lhs, rhs, _) => {
                let l = self.eval_complex_const(lhs)?;
                let r = self.eval_complex_const(rhs)?;
                // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                let real = l.0 * r.0 - l.1 * r.1;
                let imag = l.0 * r.1 + l.1 * r.0;
                Some((real, imag))
            }
            // Unary negation
            Expr::UnaryOp(UnaryOp::Neg, inner, _) => {
                let v = self.eval_complex_const(inner)?;
                Some((-v.0, -v.1))
            }
            // Unary plus
            Expr::UnaryOp(UnaryOp::Plus, inner, _) => {
                self.eval_complex_const(inner)
            }
            // Cast to complex
            Expr::Cast(_, inner, _) => {
                self.eval_complex_const(inner)
            }
            // Real number literal -> (val, 0)
            _ => {
                if let Some(val) = self.eval_const_expr(expr) {
                    let f = val.to_f64()?;
                    Some((f, 0.0))
                } else {
                    None
                }
            }
        }
    }

    /// Decompose complex arguments at a call site for ABI compliance.
    /// Replaces complex double/float pointer arguments with (real, imag) pairs.
    /// ABI conventions:
    /// - _Complex float on x86-64: pack two F32 into one F64 (one XMM register)
    /// - _Complex float on RISC-V variadic: pack two F32 into one I64 (one GP register)
    /// - _Complex float on ARM: decompose to 2 separate F32 values
    /// - _Complex float on RISC-V non-variadic: decompose to 2 separate F32 values
    /// - _Complex double: decompose to 2 F64 values (all platforms)
    /// - _Complex long double: keep as pointer (passed on stack)
    pub(super) fn decompose_complex_call_args(
        &mut self,
        arg_vals: &mut Vec<Operand>,
        arg_types: &mut Vec<IrType>,
        struct_arg_sizes: &mut Vec<Option<usize>>,
        struct_arg_aligns: &mut Vec<Option<usize>>,
        struct_arg_classes: &mut Vec<Vec<crate::common::types::EightbyteClass>>,
        param_ctypes: &Option<Vec<CType>>,
        args: &[Expr],
        is_variadic_call: bool,
    ) {
        let pctypes = match param_ctypes {
            Some(ref ct) => ct.clone(),
            None => {
                // No param type info available - try to infer from expressions
                args.iter().map(|a| self.expr_ctype(a)).collect()
            }
        };
        let n_fixed_params = param_ctypes.as_ref().map_or(0, |v| v.len());

        let uses_packed_cf = self.uses_packed_complex_float();
        let packs_cf_variadic = self.packs_complex_float_variadic();
        let mut new_vals = Vec::with_capacity(arg_vals.len() * 2);
        let mut new_types = Vec::with_capacity(arg_types.len() * 2);
        let mut new_struct_sizes = Vec::with_capacity(struct_arg_sizes.len() * 2);
        let mut new_struct_aligns = Vec::with_capacity(struct_arg_aligns.len() * 2);
        let mut new_struct_classes: Vec<Vec<crate::common::types::EightbyteClass>> = Vec::with_capacity(struct_arg_classes.len() * 2);

        for (i, (val, ty)) in arg_vals.iter().zip(arg_types.iter()).enumerate() {
            let ctype = pctypes.get(i);
            // Is this arg beyond the fixed params (i.e., a variadic argument)?
            let is_variadic_arg = is_variadic_call && i >= n_fixed_params;

            let is_complex_float = match ctype {
                Some(CType::ComplexFloat) => true,
                None => {
                    *ty == IrType::Ptr && i < args.len() && matches!(self.expr_ctype(&args[i]), CType::ComplexFloat)
                }
                _ => false,
            };

            // Check for ComplexFloat with packed convention:
            // - x86-64: always pack (2xF32 into 1xF64 in XMM register)
            // - RISC-V variadic: pack (2xF32 into 1xI64 in GP register)
            let should_pack = is_complex_float && (uses_packed_cf || (packs_cf_variadic && is_variadic_arg));

            if should_pack {
                let ptr = self.operand_to_value(*val);
                let packed = self.fresh_value();
                if packs_cf_variadic && is_variadic_arg && !uses_packed_cf {
                    // RISC-V variadic: load as I64 (two packed F32s in one GP register)
                    self.emit(Instruction::Load { dest: packed, ptr, ty: IrType::I64 , seg_override: AddressSpace::Default });
                    new_vals.push(Operand::Value(packed));
                    new_types.push(IrType::I64);
                } else {
                    // x86-64: load as F64 (two packed F32s in one XMM register)
                    self.emit(Instruction::Load { dest: packed, ptr, ty: IrType::F64 , seg_override: AddressSpace::Default });
                    new_vals.push(Operand::Value(packed));
                    new_types.push(IrType::F64);
                }
                new_struct_sizes.push(None); // packed scalar, not a struct
                new_struct_aligns.push(None);
                new_struct_classes.push(Vec::new());
                continue;
            }

            // On 64-bit targets, complex types are decomposed into (real, imag) scalar pairs:
            // - ComplexFloat: decomposed on ARM/RISC-V (not x86-64 where it's packed)
            // - ComplexDouble: decomposed on ARM/RISC-V (not x86-64, see below)
            // - ComplexLongDouble: only decomposed on ARM64 (HFA in Q regs)
            // On i686, no complex types are decomposed (all passed as structs on the stack).
            //
            // On x86-64, _Complex double is NOT decomposed into two F64 args because the
            // SysV ABI treats it as a [Sse, Sse] two-eightbyte argument that must be passed
            // entirely in registers or entirely on the stack. Decomposing would break the
            // atomic register allocation check. Instead, it's passed as a 16-byte struct-like
            // argument, which classify_sysv_struct handles correctly.
            let decomposes_cld = self.decomposes_complex_long_double();
            let decomposes_cd = self.decomposes_complex_double();
            let decomposes_cf = self.decomposes_complex_float();
            let is_x86_64 = self.target == crate::backend::Target::X86_64;

            let resolved_ctype = ctype.cloned().unwrap_or_else(|| {
                if *ty == IrType::Ptr && i < args.len() { self.expr_ctype(&args[i]) } else { CType::Int }
            });

            // On x86-64, don't decompose _Complex double - pass as 16-byte struct [Sse, Sse]
            let is_complex_double = matches!(resolved_ctype, CType::ComplexDouble);
            if is_x86_64 && is_complex_double {
                // Pass as a struct-like 16-byte [Sse, Sse] argument
                new_vals.push(*val);
                new_types.push(*ty);
                new_struct_sizes.push(Some(16));
                new_struct_aligns.push(Some(8));
                new_struct_classes.push(vec![
                    crate::common::types::EightbyteClass::Sse,
                    crate::common::types::EightbyteClass::Sse,
                ]);
                continue;
            }

            let should_decompose = match &resolved_ctype {
                CType::ComplexFloat => decomposes_cf && !uses_packed_cf,
                CType::ComplexDouble => decomposes_cd,
                CType::ComplexLongDouble => decomposes_cld,
                _ => false,
            };

            if should_decompose {
                let comp_ty = Self::complex_component_ir_type(&resolved_ctype);
                let ptr = self.operand_to_value(*val);

                // Load real part
                let real = self.load_complex_real(ptr, &resolved_ctype);
                // Load imag part
                let imag = self.load_complex_imag(ptr, &resolved_ctype);

                new_vals.push(real);
                new_types.push(comp_ty);
                new_struct_sizes.push(None); // decomposed scalar
                new_struct_aligns.push(None);
                new_struct_classes.push(Vec::new());
                new_vals.push(imag);
                new_types.push(comp_ty);
                new_struct_sizes.push(None); // decomposed scalar
                new_struct_aligns.push(None);
                new_struct_classes.push(Vec::new());
            } else {
                new_vals.push(*val);
                new_types.push(*ty);
                new_struct_sizes.push(struct_arg_sizes.get(i).copied().flatten());
                new_struct_aligns.push(struct_arg_aligns.get(i).copied().flatten());
                new_struct_classes.push(struct_arg_classes.get(i).cloned().unwrap_or_default());
            }
        }

        *arg_vals = new_vals;
        *arg_types = new_types;
        *struct_arg_sizes = new_struct_sizes;
        *struct_arg_aligns = new_struct_aligns;
        *struct_arg_classes = new_struct_classes;
    }
}
