//! Function call lowering: argument preparation, call dispatch, and result handling.
//!
//! Extracted from expr.rs. This module handles:
//! - `lower_function_call`: the main entry point for FunctionCall expressions
//! - `lower_call_arguments`: argument evaluation with implicit casts and promotions
//! - `emit_call_instruction`: dispatch to direct, function-pointer, or indirect calls
//! - `classify_struct_return`: shared sret/two-reg classification logic
//! - Helpers: maybe_narrow_call_result, is_function_variadic, get_func_ptr_return_ir_type

use std::rc::Rc;
use crate::frontend::parser::ast::Expr;
use crate::ir::reexports::{
    CallInfo,
    Instruction,
    IrBinOp,
    IrConst,
    Operand,
    Terminator,
    Value,
};
use crate::common::types::{AddressSpace, IrType, CType, target_int_ir_type};
use super::lower::Lowerer;

impl Lowerer {
    /// Classify a struct return size into sret (hidden pointer) or two-register return.
    /// Returns (sret_size, two_reg_size).
    ///
    /// On x86-64/arm/riscv (64-bit targets):
    ///   - size > 16: sret (hidden pointer)
    ///   - 9-16 bytes: two-register return (packed into I128, returned in rax:rdx / x0:x1 / a0:a1)
    ///   - 1-8 bytes: small struct (packed into I64, returned in rax / x0 / a0)
    ///
    /// On i686 (32-bit target):
    ///   - ALL structs use sret (hidden pointer) regardless of size.
    ///     The i386 System V ABI always returns structs via a caller-provided
    ///     hidden pointer (first stack argument), with `ret $4` to pop it.
    pub(super) fn classify_struct_return(size: usize) -> (Option<usize>, Option<usize>) {
        let ptr_size = crate::common::types::target_ptr_size();
        if ptr_size <= 4 {
            // 32-bit target (i686): i386 SysV ABI always uses sret for structs.
            (Some(size), None)
        } else {
            // 64-bit target: register pair is 16 bytes
            if size > 16 {
                (Some(size), None)
            } else if size > 8 {
                (None, Some(size))
            } else {
                (None, None)
            }
        }
    }

    /// Check if an identifier is a function pointer variable rather than a direct function.
    /// A local variable always shadows a known function of the same name (e.g., a parameter
    /// named `free` that is a function pointer should produce an indirect call, not a direct
    /// call to the libc `free` symbol).
    pub(super) fn is_func_ptr_variable(&self, name: &str) -> bool {
        // When called outside a function context (e.g., during global constant
        // evaluation), there are no locals, so this is not a func ptr variable.
        let func_state = match self.func_state.as_ref() {
            Some(fs) => fs,
            None => return self.globals.contains_key(name) && !self.known_functions.contains(name),
        };
        if func_state.locals.contains_key(name) {
            // Local variable exists with this name. Check if it's a function pointer
            // via ptr_sigs (set for function pointer params and locals) or c_type.
            if self.func_meta.ptr_sigs.contains_key(name) {
                return true;
            }
            // Also check the local's CType for function pointer types not in ptr_sigs
            if let Some(local_info) = func_state.locals.get(name) {
                if let Some(ref cty) = local_info.c_type {
                    if cty.is_function_pointer() {
                        return true;
                    }
                }
            }
            // Local exists but is not a function pointer - it's a regular variable
            // that happens to shadow a known function name. Not a func ptr variable.
            false
        } else {
            // No local with this name. Check globals, excluding known functions.
            self.globals.contains_key(name) && !self.known_functions.contains(name)
        }
    }

    pub(super) fn lower_function_call(&mut self, func: &Expr, args: &[Expr]) -> Operand {
        // Strip Deref layers that are semantically no-ops.
        // In C, dereferencing a function pointer is a no-op: (*fp)() == fp().
        // But dereferencing a pointer-to-function-pointer is a real load:
        // (*fpp)() where fpp is int (**)(int,int) must load the function pointer.
        // Only strip a Deref if the inner expression is itself a function pointer
        // or function designator (checked via is_function_pointer_deref).
        let mut stripped_func = func;
        while let Expr::Deref(inner, _) = stripped_func {
            if self.is_function_pointer_deref(inner) {
                stripped_func = inner;
            } else {
                break;
            }
        }

        // Strip AddressOf wrapping a function name: (&func)(args) => func(args).
        // In C, taking the address of a function and calling through it is equivalent
        // to calling the function directly. The kernel's static_call mechanism uses
        // this pattern: ({ ...; (&__SCT__trampoline); })(args) and requires a direct
        // call instruction so the call site can be patched at runtime.
        if let Expr::AddressOf(inner, _) = stripped_func {
            if let Expr::Identifier(name, _) = inner.as_ref() {
                if !self.is_func_ptr_variable(name) {
                    stripped_func = inner;
                }
            }
        }

        // Resolve _Generic selections to the selected expression.
        // Without this, _Generic(expr, type1: func1, type2: func2)(args) would
        // fall through to the catch-all indirect call path, which dereferences
        // the function address instead of calling directly.
        if let Expr::GenericSelection(controlling, associations, _) = stripped_func {
            stripped_func = self.resolve_generic_selection(controlling, associations);
        }
        // Use the resolved expression for emit_call_instruction too
        let effective_func = stripped_func;

        // Resolve __builtin_* functions first
        if let Expr::Identifier(name, _) = stripped_func {
            if let Some(result) = self.try_lower_builtin_call(name, args) {
                return result;
            }

            // Functions declared with __attribute__((error("..."))) are compile-time
            // assertion functions (e.g., kernel's __bad_mask, __field_overflow).
            // In GCC, these calls are eliminated by inlining + constant folding,
            // and if they survive, GCC emits a compile error. Since CCC's inliner
            // may not inline all call sites (due to budget/round limits), we must
            // handle surviving calls gracefully.
            //
            // Instead of emitting a trap (which causes kernel BRK crashes when
            // always_inline functions like cpucap_is_possible aren't fully inlined),
            // we simply skip the call. This avoids referencing the undefined symbol
            // and makes the standalone function body harmless. After inlining, the
            // dead code containing this no-op is eliminated by constant folding and DCE.
            //
            // TODO: Once the inliner can fully inline all always_inline chains
            // (eliminating all error function call sites), restore this to emit a
            // compile error (matching GCC behavior) instead of silently dropping.
            if self.error_functions.contains(name) {
                // No-op: skip the call. The code path should be unreachable after
                // inlining and constant folding. If it IS reached at runtime,
                // execution continues harmlessly (which is better than crashing).
                return Operand::Const(IrConst::ptr_int(0));
            }
        }

        // Determine sret/two-reg return convention
        let (sret_size, two_reg_size, call_ret_classes) = if let Expr::Identifier(name, _) = stripped_func {
            if self.is_func_ptr_variable(name) {
                // Indirect call through function pointer variable
                // TODO: compute ret_eightbyte_classes from the function pointer's
                // return type to support mixed SSE/INTEGER struct returns via fptrs
                match self.get_call_return_struct_size(effective_func) {
                    Some(size) => {
                        let (s, t) = Self::classify_struct_return(size);
                        (s, t, Vec::new())
                    }
                    None => (None, None, Vec::new()),
                }
            } else {
                // Direct function call - look up by function name
                let sig = self.func_meta.sigs.get(&*name);
                (
                    sig.and_then(|s| s.sret_size),
                    sig.and_then(|s| s.two_reg_ret_size),
                    sig.map(|s| s.ret_eightbyte_classes.clone()).unwrap_or_default(),
                )
            }
        } else {
            // Non-identifier function expression (e.g., array[i]())
            // TODO: compute ret_eightbyte_classes from expression return type
            // to support mixed SSE/INTEGER struct returns via indirect calls
            match self.get_call_return_struct_size(effective_func) {
                Some(size) => {
                    let (s, t) = Self::classify_struct_return(size);
                    (s, t, Vec::new())
                }
                None => (None, None, Vec::new()),
            }
        };

        // Lower arguments with implicit casts
        let (mut arg_vals, mut arg_types, mut struct_arg_sizes, mut struct_arg_aligns, mut struct_arg_classes, mut struct_arg_riscv_float_classes) = self.lower_call_arguments(effective_func, args);

        // Detect variadic status early (needed for complex arg decomposition)
        let call_is_variadic = if let Expr::Identifier(name, _) = stripped_func {
            self.is_function_variadic(name)
        } else {
            false
        };

        // Decompose complex double/float arguments into (real, imag) pairs for ABI compliance
        let param_ctypes_for_decompose = if let Expr::Identifier(name, _) = stripped_func {
            let sig_for_decompose = if self.is_func_ptr_variable(name) {
                self.func_meta.ptr_sigs.get(&*name).or_else(|| self.func_meta.sigs.get(&*name))
            } else {
                self.func_meta.sigs.get(&*name)
            };
            sig_for_decompose.map(|s| s.param_ctypes.clone()).filter(|v| !v.is_empty())
        } else {
            None
        };
        self.decompose_complex_call_args(&mut arg_vals, &mut arg_types, &mut struct_arg_sizes, &mut struct_arg_aligns, &mut struct_arg_classes, &param_ctypes_for_decompose, args, call_is_variadic);

        let dest = self.fresh_value();

        // For sret calls, allocate space and prepend hidden pointer argument
        let sret_alloca = if let Some(size) = sret_size {
            let alloca = self.fresh_value();
            self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size, align: 0, volatile: false });
            arg_vals.insert(0, Operand::Value(alloca));
            arg_types.insert(0, IrType::Ptr);
            struct_arg_sizes.insert(0, None); // sret pointer is not a struct arg
            struct_arg_aligns.insert(0, None);
            struct_arg_classes.insert(0, Vec::new()); // sret pointer has no eightbyte classes
            struct_arg_riscv_float_classes.insert(0, None); // sret pointer has no riscv float class
            Some(alloca)
        } else {
            None
        };

        // Determine number of fixed args for variadic calls
        let (call_variadic, num_fixed_args) = if let Expr::Identifier(name, _) = stripped_func {
            let variadic = call_is_variadic;
            let n_fixed = if variadic {
                let variadic_sig = if self.is_func_ptr_variable(name) {
                    self.func_meta.ptr_sigs.get(&*name).or_else(|| self.func_meta.sigs.get(&*name))
                } else {
                    self.func_meta.sigs.get(&*name)
                };
                if let Some(sig) = variadic_sig {
                    if !sig.param_ctypes.is_empty() {
                        let decomposes_cld = self.decomposes_complex_long_double();
                        let decomposes_cd = self.decomposes_complex_double();
                        let decomposes_cf = self.decomposes_complex_float();
                        sig.param_ctypes.iter().map(|ct| {
                            match ct {
                                CType::ComplexDouble if decomposes_cd => 2,
                                // Fixed ComplexFloat params are decomposed into 2 FP regs
                                // on 64-bit targets (not x86-64 packed, not i686 struct)
                                CType::ComplexFloat if decomposes_cf && !self.uses_packed_complex_float() => 2,
                                CType::ComplexLongDouble if decomposes_cld => 2,
                                _ => 1,
                            }
                        }).sum()
                    } else if !sig.param_types.is_empty() {
                        sig.param_types.len()
                    } else {
                        arg_vals.len()
                    }
                } else {
                    arg_vals.len()
                }
            } else {
                arg_vals.len()
            };
            (variadic, n_fixed)
        } else {
            (false, arg_vals.len())
        };

        // Dispatch: direct call, function pointer call, or indirect call
        let call_ret_ty = self.emit_call_instruction(effective_func, dest, arg_vals, arg_types, struct_arg_sizes, struct_arg_aligns, struct_arg_classes, struct_arg_riscv_float_classes, call_variadic, num_fixed_args, two_reg_size, sret_size, call_ret_classes);

        // After call to noreturn function, emit unreachable and start dead block.
        // Unlike error_functions (which skip the call entirely), noreturn functions
        // are real functions that get called but never return (e.g., panic, abort).
        if let Expr::Identifier(name, _) = stripped_func {
            if self.noreturn_functions.contains(name) {
                self.terminate(Terminator::Unreachable);
                let dead_label = self.fresh_label();
                self.start_block(dead_label);
                return Operand::Const(IrConst::ptr_int(0));
            }
        }

        // For sret calls, the struct data is now in the alloca - return its address
        if let Some(alloca) = sret_alloca {
            return Operand::Value(alloca);
        }

        // For two-register struct returns (9-16 bytes), unpack I128 into struct alloca
        if let Some(size) = two_reg_size {
            return self.unpack_two_reg_return(dest, size);
        }

        // For complex returns (non-sret), store into complex alloca
        if sret_size.is_none() {
            // First try the function name lookup (for direct calls)
            let ret_ct = if let Expr::Identifier(name, _) = stripped_func {
                self.types.func_return_ctypes.get(name).cloned()
            } else {
                None
            };
            // Fall back to extracting return CType from the expression's type
            // (handles indirect calls through function pointers)
            let ret_ct = ret_ct.or_else(|| self.get_func_ptr_return_ctype(stripped_func));
            if let Some(ret_ct) = ret_ct {
                if let Some(result) = self.handle_complex_return(dest, &ret_ct) {
                    return result;
                }
            }
        }

        // Narrow the result if the return type is sub-64-bit integer
        self.maybe_narrow_call_result(dest, call_ret_ty)
    }

    /// Unpack a two-register (I128) struct return into a struct alloca.
    fn unpack_two_reg_return(&mut self, dest: Value, size: usize) -> Operand {
        let alloca = self.fresh_value();
        self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size, align: 0, volatile: false });
        // Extract low 8 bytes (rax)
        let lo = self.fresh_value();
        self.emit(Instruction::Cast { dest: lo, src: Operand::Value(dest), from_ty: IrType::I128, to_ty: IrType::I64 });
        self.emit(Instruction::Store { val: Operand::Value(lo), ptr: alloca, ty: IrType::I64 , seg_override: AddressSpace::Default });
        // Extract high bytes (rdx): shift right by 64
        let shifted = self.fresh_value();
        self.emit(Instruction::BinOp { dest: shifted, op: IrBinOp::LShr, lhs: Operand::Value(dest), rhs: Operand::Const(IrConst::I64(64)), ty: IrType::I128 });
        let hi = self.fresh_value();
        self.emit(Instruction::Cast { dest: hi, src: Operand::Value(shifted), from_ty: IrType::I128, to_ty: IrType::I64 });
        let hi_ptr = self.fresh_value();
        self.emit(Instruction::GetElementPtr { dest: hi_ptr, base: alloca, offset: Operand::Const(IrConst::I64(8)), ty: IrType::I64 });
        self.emit(Instruction::Store { val: Operand::Value(hi), ptr: hi_ptr, ty: IrType::I64 , seg_override: AddressSpace::Default });
        Operand::Value(alloca)
    }

    /// Handle complex return types (ComplexFloat, ComplexDouble, ComplexLongDouble).
    /// Returns Some(result) if the return was complex, None otherwise.
    /// Note: on i686, ComplexDouble and ComplexLongDouble use sret (handled before this),
    /// so this function only needs to handle ComplexFloat on i686.
    fn handle_complex_return(&mut self, dest: Value, ret_ct: &CType) -> Option<Operand> {
        match ret_ct {
            CType::ComplexFloat => {
                if self.uses_packed_complex_float() {
                    // x86-64: two packed F32 returned in xmm0 as F64
                    // Store the raw 8 bytes (two F32s) into an alloca
                    let alloca = self.fresh_value();
                    self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size: 8, align: 0, volatile: false });
                    self.emit(Instruction::Store { val: Operand::Value(dest), ptr: alloca, ty: IrType::F64 , seg_override: AddressSpace::Default });
                    Some(Operand::Value(alloca))
                } else if !self.decomposes_complex_float() {
                    // i686: two packed F32 returned in eax:edx as I64
                    // Store the raw 8 bytes (two F32s) into an alloca
                    let alloca = self.fresh_value();
                    self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size: 8, align: 0, volatile: false });
                    self.emit(Instruction::Store { val: Operand::Value(dest), ptr: alloca, ty: IrType::I64 , seg_override: AddressSpace::Default });
                    Some(Operand::Value(alloca))
                } else {
                    // ARM/RISC-V: real F32 in first FP reg (dest), imag F32 in second FP reg
                    let imag_val = self.fresh_value();
                    self.emit(Instruction::GetReturnF32Second { dest: imag_val });
                    let alloca = self.fresh_value();
                    self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size: 8, align: 0, volatile: false });
                    // Store real part (F32) at offset 0
                    self.emit(Instruction::Store { val: Operand::Value(dest), ptr: alloca, ty: IrType::F32 , seg_override: AddressSpace::Default });
                    // Store imag part (F32) at offset 4
                    let imag_ptr = self.fresh_value();
                    let ptr_int_ty = crate::common::types::target_int_ir_type();
                    self.emit(Instruction::BinOp {
                        dest: imag_ptr, op: IrBinOp::Add,
                        lhs: Operand::Value(alloca), rhs: Operand::Const(IrConst::ptr_int(4)),
                        ty: ptr_int_ty,
                    });
                    self.emit(Instruction::Store { val: Operand::Value(imag_val), ptr: imag_ptr, ty: IrType::F32 , seg_override: AddressSpace::Default });
                    Some(Operand::Value(alloca))
                }
            }
            CType::ComplexDouble if self.decomposes_complex_double() => {
                // x86-64/ARM64/RISC-V: _Complex double: real in first FP reg (dest),
                // imag in second FP reg (second return)
                let imag_val = self.fresh_value();
                self.emit(Instruction::GetReturnF64Second { dest: imag_val });
                let alloca = self.fresh_value();
                self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size: 16, align: 0, volatile: false });
                self.emit(Instruction::Store { val: Operand::Value(dest), ptr: alloca, ty: IrType::F64 , seg_override: AddressSpace::Default });
                let imag_ptr = self.fresh_value();
                let ptr_int_ty = crate::common::types::target_int_ir_type();
                self.emit(Instruction::BinOp {
                    dest: imag_ptr, op: IrBinOp::Add,
                    lhs: Operand::Value(alloca), rhs: Operand::Const(IrConst::ptr_int(8)),
                    ty: ptr_int_ty,
                });
                self.emit(Instruction::Store { val: Operand::Value(imag_val), ptr: imag_ptr, ty: IrType::F64 , seg_override: AddressSpace::Default });
                Some(Operand::Value(alloca))
            }
            CType::ComplexLongDouble if self.returns_complex_long_double_in_regs() => {
                // x86-64: _Complex long double returns real in x87 st(0), imag in x87 st(1).
                // After the call, emit_call_store_result stores st(0) (real) into dest.
                // GetReturnF128Second reads the next x87 value (former st(1), now st(0) after first fstpt).
                let imag_val = self.fresh_value();
                self.emit(Instruction::GetReturnF128Second { dest: imag_val });
                let alloca = self.fresh_value();
                self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size: 32, align: 16, volatile: false });
                // Store real part (F128) at offset 0
                self.emit(Instruction::Store { val: Operand::Value(dest), ptr: alloca, ty: IrType::F128, seg_override: AddressSpace::Default });
                // Store imag part (F128) at offset 16
                let imag_ptr = self.fresh_value();
                let ptr_int_ty = crate::common::types::target_int_ir_type();
                self.emit(Instruction::BinOp {
                    dest: imag_ptr, op: IrBinOp::Add,
                    lhs: Operand::Value(alloca), rhs: Operand::Const(IrConst::ptr_int(16)),
                    ty: ptr_int_ty,
                });
                self.emit(Instruction::Store { val: Operand::Value(imag_val), ptr: imag_ptr, ty: IrType::F128, seg_override: AddressSpace::Default });
                Some(Operand::Value(alloca))
            }
            _ => None,
        }
    }

    /// Lower function call arguments, applying implicit casts for parameter types
    /// and default argument promotions for variadic args.
    /// Returns (arg_vals, arg_types, struct_arg_sizes, struct_arg_aligns, struct_arg_classes) where struct_arg_sizes[i] is
    /// Some(size) if the ith argument is a struct/union passed by value, and struct_arg_aligns[i]
    /// is Some(align) for struct args.
    pub(super) fn lower_call_arguments(&mut self, func: &Expr, args: &[Expr]) -> (Vec<Operand>, Vec<IrType>, Vec<Option<usize>>, Vec<Option<usize>>, Vec<Vec<crate::common::types::EightbyteClass>>, Vec<Option<crate::common::types::RiscvFloatClass>>) {
        // Extract function name from direct calls, or the underlying variable name
        // from indirect calls through function pointers (e.g., (*afp)(args) -> "afp").
        let func_name = match func {
            Expr::Identifier(name, _) => Some(&*name),
            Expr::Deref(inner, _) => {
                if let Expr::Identifier(name, _) = inner.as_ref() {
                    Some(&*name)
                } else { None }
            }
            _ => None,
        };
        // When the callee is a local function pointer variable, prefer ptr_sigs
        // over sigs. This prevents a parameter named e.g. `round` from picking up
        // the seeded `double round(double)` library signature instead of the
        // actual function pointer's signature.
        let sig = func_name.and_then(|name| {
            if self.is_func_ptr_variable(name) {
                self.func_meta.ptr_sigs.get(name)
                    .or_else(|| self.func_meta.sigs.get(name))
            } else {
                self.func_meta.sigs.get(name)
                    .or_else(|| self.func_meta.ptr_sigs.get(name))
            }
        });

        // When calling through a complex expression (e.g., struct member function
        // pointer like stats->compute_stats(...)), sig will be None because there
        // is no named function to look up. In that case, try to extract parameter
        // type information from the callee expression's CType so that implicit
        // argument conversions (e.g., int -> double) are performed correctly.
        let inferred_from_ctype = if sig.is_none() {
            self.extract_fn_ptr_param_info(func)
        } else {
            None
        };

        // If the signature has an empty parameter list (unprototyped function like `int f()`),
        // all arguments need default argument promotions (float->double, char/short->int).
        let is_unprototyped = sig.is_some_and(|s| s.param_types.is_empty());
        let param_types: Option<Vec<IrType>> = sig.map(|s| s.param_types.clone()).filter(|v| !v.is_empty())
            .or_else(|| inferred_from_ctype.as_ref().map(|(pt, _, _, _)| pt.clone()).filter(|v| !v.is_empty()));
        let param_ctypes: Option<Vec<CType>> = sig.map(|s| s.param_ctypes.clone()).filter(|v| !v.is_empty())
            .or_else(|| inferred_from_ctype.as_ref().map(|(_, pc, _, _)| pc.clone()).filter(|v| !v.is_empty()));
        let param_bool_flags: Option<Vec<bool>> = sig.map(|s| s.param_bool_flags.clone()).filter(|v| !v.is_empty())
            .or_else(|| inferred_from_ctype.as_ref().map(|(_, _, pb, _)| pb.clone()).filter(|v| !v.is_empty()));
        let pre_call_variadic = func_name.is_some_and(|name|
            self.is_function_variadic(name)
        ) || inferred_from_ctype.as_ref().is_some_and(|(_, _, _, variadic)| *variadic);

        let mut arg_types = Vec::with_capacity(args.len());
        // Track argument indices where a complex expression was converted to a
        // scalar type (e.g., complex-to-bool). These should NOT get a
        // struct_arg_size based on the original expression's CType.
        let mut complex_converted_to_scalar: Vec<bool> = vec![false; args.len()];
        let arg_vals: Vec<Operand> = args.iter().enumerate().map(|(i, a)| {
            let mut val = self.lower_expr(a);
            let arg_ty = self.get_expr_type(a);

            // Convert complex arguments to the declared parameter complex type if they differ
            if let Some(ref pctypes) = param_ctypes {
                if i < pctypes.len() && pctypes[i].is_complex() {
                    let arg_ct = self.expr_ctype(a);
                    if arg_ct.is_complex() && arg_ct != pctypes[i] {
                        let ptr = self.operand_to_value(val);
                        val = self.complex_to_complex(ptr, &arg_ct, &pctypes[i]);
                    } else if !arg_ct.is_complex() {
                        val = self.real_to_complex(val, &arg_ct, &pctypes[i]);
                    }
                } else if i < pctypes.len() && !pctypes[i].is_complex() {
                    let arg_ct = self.expr_ctype(a);
                    if arg_ct.is_complex() {
                        let ptr = self.operand_to_value(val);
                        // Check if target param is _Bool: use both real and imag per C11 6.3.1.2
                        let is_param_bool = param_bool_flags.as_ref()
                            .and_then(|bf| bf.get(i).copied())
                            .unwrap_or(false);
                        if is_param_bool {
                            let cast_val = self.lower_complex_to_bool(ptr, &arg_ct);
                            arg_types.push(IrType::I8);
                            complex_converted_to_scalar[i] = true;
                            return cast_val;
                        }
                        let real_part = self.load_complex_real(ptr, &arg_ct);
                        let comp_ir_ty = Self::complex_component_ir_type(&arg_ct);
                        let param_ty = param_types.as_ref()
                            .and_then(|pt| pt.get(i).copied())
                            .unwrap_or(comp_ir_ty);
                        let cast_val = self.emit_implicit_cast(real_part, comp_ir_ty, param_ty);
                        arg_types.push(param_ty);
                        complex_converted_to_scalar[i] = true;
                        return cast_val;
                    }
                }
            }

            // Spill packed struct data to a temporary alloca so that the call
            // argument is a pointer (address) rather than raw data.  This is
            // needed for any expression that produces packed struct data:
            // direct function calls, statement expressions wrapping calls,
            // ternaries, comma expressions, etc.
            {
                let is_struct_ret = matches!(
                    self.get_expr_ctype(a),
                    Some(CType::Struct(_)) | Some(CType::Union(_))
                );
                if is_struct_ret && self.expr_produces_packed_struct_data(a) {
                    let struct_size = self.struct_value_size(a).unwrap_or(8);
                    let alloc_size = if struct_size > 0 { struct_size } else { 8 };
                    let alloca = self.fresh_value();
                    let store_ty = Self::packed_store_type(alloc_size);
                    self.emit(Instruction::Alloca { dest: alloca, size: alloc_size, ty: store_ty, align: 0, volatile: false });
                    self.emit(Instruction::Store { val, ptr: alloca, ty: store_ty , seg_override: AddressSpace::Default });
                    val = Operand::Value(alloca);
                }
            }

            let is_bool_param = param_bool_flags.as_ref()
                .is_some_and(|flags| i < flags.len() && flags[i]);

            if let Some(ref ptypes) = param_types {
                if i < ptypes.len() {
                    let param_ty = ptypes[i];
                    arg_types.push(param_ty);
                    if is_bool_param {
                        // For _Bool params, normalize at source type before truncation.
                        return self.emit_bool_normalize_typed(val, arg_ty);
                    }
                    let cast_val = self.emit_implicit_cast(val, arg_ty, param_ty);
                    return cast_val;
                }
            }
            // Default argument promotions (C11 6.5.2.2p6): for variadic args,
            // args beyond known params, or all args for unprototyped functions:
            // - float -> double
            // - char/short (signed or unsigned) -> int
            let needs_promotion = pre_call_variadic || param_types.is_some() || is_unprototyped;
            if needs_promotion {
                if arg_ty == IrType::F32 {
                    arg_types.push(IrType::F64);
                    return self.emit_implicit_cast(val, IrType::F32, IrType::F64);
                }
                if matches!(arg_ty, IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16) {
                    arg_types.push(IrType::I32);
                    return self.emit_implicit_cast(val, arg_ty, IrType::I32);
                }
            }
            arg_types.push(arg_ty);
            val
        }).collect();

        // Build struct_arg_sizes: for each arg, check if it's a struct/union by value
        let func_name = if let Expr::Identifier(name, _) = func { Some(&*name) } else { None };
        let struct_arg_sizes: Vec<Option<usize>> = if let Some(ref sizes) = func_name.and_then(|n| self.func_meta.sigs.get(n).map(|s| s.param_struct_sizes.clone())) {
            // Use pre-registered struct sizes from function metadata.
            // For variadic _Complex long double args beyond fixed params, infer size
            // (param_struct_sizes only covers declared parameters).
            // Also override None entries for vector types, since the first-pass
            // registration may not have resolved vector typedefs yet.
            let decomposes_cld = self.decomposes_complex_long_double();
            args.iter().enumerate().map(|(i, a)| {
                // If Stage A already converted this complex argument to a scalar
                // (e.g., complex-to-bool or complex-to-real), the value is no
                // longer a struct-like type and must not get a struct_arg_size.
                if complex_converted_to_scalar.get(i).copied().unwrap_or(false) {
                    return None;
                }
                let registered = if i < sizes.len() { sizes[i] } else { None };
                if registered.is_some() {
                    return registered;
                }
                // Fall back to expression-based inference for unregistered types
                let ctype = self.get_expr_ctype(a);
                let decomposes_cd = self.decomposes_complex_double();
                let decomposes_cf = self.decomposes_complex_float();
                match ctype {
                    Some(CType::Struct(_)) | Some(CType::Union(_)) => {
                        self.struct_value_size(a)
                    }
                    Some(CType::Vector(_, total_size)) => Some(total_size),
                    Some(CType::ComplexLongDouble) if !decomposes_cld => {
                        Some(CType::ComplexLongDouble.size())
                    }
                    // i686: ComplexDouble (16 bytes) and ComplexFloat (8 bytes) are
                    // passed as structs on the stack when not decomposed.
                    Some(CType::ComplexDouble) if !decomposes_cd => {
                        Some(CType::ComplexDouble.size())
                    }
                    Some(CType::ComplexFloat) if !decomposes_cf => {
                        Some(CType::ComplexFloat.size())
                    }
                    _ => None,
                }
            }).collect()
        } else {
            // Infer from argument expressions
            let decomposes_cld = self.decomposes_complex_long_double();
            let decomposes_cd = self.decomposes_complex_double();
            let decomposes_cf = self.decomposes_complex_float();
            args.iter().enumerate().map(|(i, a)| {
                // If Stage A already converted this complex argument to a scalar
                // (e.g., complex-to-bool or complex-to-real), the value is no
                // longer a struct-like type and must not get a struct_arg_size.
                if complex_converted_to_scalar.get(i).copied().unwrap_or(false) {
                    return None;
                }
                let ctype = self.get_expr_ctype(a);
                match ctype {
                    Some(CType::Struct(_)) | Some(CType::Union(_)) => {
                        self.struct_value_size(a)
                    }
                    Some(CType::Vector(_, total_size)) => {
                        Some(total_size) // Vector types are passed by value like structs
                    }
                    Some(CType::ComplexLongDouble) if !decomposes_cld => {
                        Some(CType::ComplexLongDouble.size())
                    }
                    // i686: ComplexDouble and ComplexFloat passed as structs on stack
                    Some(CType::ComplexDouble) if !decomposes_cd => {
                        Some(CType::ComplexDouble.size())
                    }
                    Some(CType::ComplexFloat) if !decomposes_cf => {
                        Some(CType::ComplexFloat.size())
                    }
                    _ => None,
                }
            }).collect()
        };

        // Build struct_arg_aligns: for each struct arg, record its alignment.
        // This is used by RISC-V to even-align register pairs for 2×XLEN-aligned structs
        // (e.g., struct containing long double with 16-byte alignment).
        let struct_arg_aligns: Vec<Option<usize>> = args.iter().enumerate().map(|(i, a)| {
            struct_arg_sizes.get(i).copied().flatten()?;
            let ctype = self.get_expr_ctype(a);
            match ctype {
                Some(ref ct @ CType::Struct(_)) | Some(ref ct @ CType::Union(_)) => {
                    self.get_struct_layout_for_ctype(ct).map(|layout| layout.align)
                }
                _ => None,
            }
        }).collect();

        // Build struct_arg_classes: propagate per-eightbyte SysV ABI classification from FuncSig.
        // For variadic args beyond fixed params, infer classification from expression CType
        // so that struct fields are correctly split between GP and SSE registers.
        let struct_arg_classes: Vec<Vec<crate::common::types::EightbyteClass>> = if let Some(ref classes) = func_name.and_then(|n| self.func_meta.sigs.get(n).map(|s| s.param_struct_classes.clone())) {
            args.iter().enumerate().map(|(i, a)| {
                if i < classes.len() && !classes[i].is_empty() {
                    classes[i].clone()
                } else {
                    // Infer eightbyte classification from expression CType for variadic struct args
                    self.infer_struct_eightbyte_classes(a)
                }
            }).collect()
        } else {
            args.iter().map(|a| self.infer_struct_eightbyte_classes(a)).collect()
        };

        // Build struct_arg_riscv_float_classes: propagate RISC-V float classification from FuncSig.
        // Variadic args beyond fixed params get None — on RISC-V LP64D, variadic
        // struct args are passed in GP registers, not FP registers.
        let struct_arg_riscv_float_classes: Vec<Option<crate::common::types::RiscvFloatClass>> = if let Some(ref classes) = func_name.and_then(|n| self.func_meta.sigs.get(n).map(|s| s.param_riscv_float_classes.clone())) {
            args.iter().enumerate().map(|(i, _a)| {
                if i < classes.len() {
                    classes[i]
                } else {
                    None
                }
            }).collect()
        } else {
            args.iter().map(|_a| None).collect()
        };

        (arg_vals, arg_types, struct_arg_sizes, struct_arg_aligns, struct_arg_classes, struct_arg_riscv_float_classes)
    }

    /// Infer SysV ABI eightbyte classification for a struct argument expression.
    /// This is used for variadic struct args that don't have pre-registered classification
    /// from the function signature (since the signature only covers fixed params).
    fn infer_struct_eightbyte_classes(&self, expr: &Expr) -> Vec<crate::common::types::EightbyteClass> {
        if let Some(ctype) = self.get_expr_ctype(expr) {
            if let Some(layout) = self.get_struct_layout_for_ctype(&ctype) {
                return layout.classify_sysv_eightbytes(&*self.types.borrow_struct_layouts());
            }
        }
        Vec::new()
    }

    /// Emit the actual call instruction (direct, indirect via fptr, or general indirect).
    /// Returns the effective return type for narrowing.
    fn emit_call_instruction(
        &mut self,
        func: &Expr,
        dest: Value,
        arg_vals: Vec<Operand>,
        arg_types: Vec<IrType>,
        struct_arg_sizes: Vec<Option<usize>>,
        struct_arg_aligns: Vec<Option<usize>>,
        struct_arg_classes: Vec<Vec<crate::common::types::EightbyteClass>>,
        struct_arg_riscv_float_classes: Vec<Option<crate::common::types::RiscvFloatClass>>,
        is_variadic: bool,
        num_fixed_args: usize,
        two_reg_size: Option<usize>,
        sret_size: Option<usize>,
        call_ret_classes: Vec<crate::common::types::EightbyteClass>,
    ) -> IrType {
        let mut indirect_ret_ty = self.get_func_ptr_return_ir_type(func);
        if two_reg_size.is_some() {
            indirect_ret_ty = IrType::I128;
        }

        match func {
            Expr::Identifier(name, _) => {
                if self.is_func_ptr_variable(name) {
                    // Function pointer variable: load pointer and call indirect
                    let func_ptr = self.load_func_ptr_variable(name);
                    // TODO: Support fastcall through function pointers (requires tracking
                    // calling convention in function pointer types)
                    self.emit(Instruction::CallIndirect {
                        func_ptr: Operand::Value(func_ptr),
                        info: CallInfo {
                            dest: Some(dest), args: arg_vals, arg_types,
                            return_type: indirect_ret_ty, is_variadic, num_fixed_args,
                            struct_arg_sizes, struct_arg_aligns, struct_arg_classes,
                            struct_arg_riscv_float_classes,
                            is_sret: sret_size.is_some(), is_fastcall: false,
                            ret_eightbyte_classes: call_ret_classes,
                        },
                    });
                    indirect_ret_ty
                } else {
                    // Direct call - apply __asm__("label") linker symbol redirect if present
                    let call_name = self.asm_label_map.get(&*name)
                        .cloned()
                        .unwrap_or_else(|| name.clone());
                    let sig = self.func_meta.sigs.get(&*name);
                    let mut ret_ty = sig.map(|s| s.return_type).unwrap_or(target_int_ir_type());
                    if sig.and_then(|s| s.two_reg_ret_size).is_some() {
                        ret_ty = IrType::I128;
                    }
                    // On 32-bit targets, all struct/union returns use sret (the
                    // i386 SysV ABI never returns structs in registers). This
                    // block handles the fallback case where an indirect call
                    // has a missing signature.
                    if crate::common::types::target_is_32bit() {
                        if let Some(s) = sig {
                            if s.sret_size.is_none() && s.two_reg_ret_size.is_none() {
                                if let Some(ref rc) = s.return_ctype {
                                    if rc.is_struct_or_union() {
                                        ret_ty = IrType::Ptr;
                                    }
                                }
                            }
                        }
                    }
                    let callee_is_fastcall = self.fastcall_functions.contains(&*name);
                    self.emit(Instruction::Call {
                        func: Rc::from(call_name),
                        info: CallInfo {
                            dest: Some(dest), args: arg_vals, arg_types,
                            return_type: ret_ty, is_variadic, num_fixed_args,
                            struct_arg_sizes, struct_arg_aligns, struct_arg_classes,
                            struct_arg_riscv_float_classes,
                            is_sret: sret_size.is_some(), is_fastcall: callee_is_fastcall,
                            ret_eightbyte_classes: call_ret_classes,
                        },
                    });
                    ret_ty
                }
            }
            Expr::Deref(inner, _) => {
                // In C, dereferencing a function pointer is a no-op: (*fp)(args) == fp(args).
                // But dereferencing a pointer-to-function-pointer is a real load:
                // (*fpp)(args) where fpp is func_ptr* needs to load the func_ptr first.
                // Use the comprehensive is_function_pointer_deref check which also handles
                // cases where c_type is None but ptr_sigs or known_functions provide info.
                let is_noop_deref = self.is_function_pointer_deref(inner);
                let n = arg_vals.len();
                let sas = struct_arg_sizes;
                let saa = struct_arg_aligns;
                let sac = struct_arg_classes;
                let sarfc = struct_arg_riscv_float_classes;
                let func_ptr = if is_noop_deref {
                    // No-op dereference: (*fp)() == fp()
                    self.lower_expr(inner)
                } else {
                    // Real dereference needed: load through the pointer to get the func ptr
                    self.lower_expr(func)
                };
                self.emit(Instruction::CallIndirect {
                    func_ptr,
                    info: CallInfo {
                        dest: Some(dest), args: arg_vals, arg_types,
                        return_type: indirect_ret_ty, is_variadic: false, num_fixed_args: n,
                        struct_arg_sizes: sas, struct_arg_aligns: saa, struct_arg_classes: sac,
                        struct_arg_riscv_float_classes: sarfc,
                        is_sret: sret_size.is_some(), is_fastcall: false,
                        ret_eightbyte_classes: call_ret_classes,
                    },
                });
                indirect_ret_ty
            }
            _ => {
                let sas = struct_arg_sizes;
                let saa = struct_arg_aligns;
                let sac = struct_arg_classes;
                let sarfc = struct_arg_riscv_float_classes;
                let func_ptr = self.lower_expr(func);

                // Check if the lowered expression is a GlobalAddr of a known function.
                // This handles patterns like ({ ...; (&func); })(args) where a
                // statement expression yields a function address. The kernel's
                // static_call mechanism uses this pattern and requires a direct call
                // instruction so the call site can be patched at runtime.
                let direct_func_name = if let Operand::Value(v) = func_ptr {
                    let instrs = &self.func_state.as_ref()
                        .expect("func_state must exist during function lowering").instrs;
                    let found = instrs.iter().rev().find_map(|inst| {
                        if let Instruction::GlobalAddr { dest, ref name } = *inst {
                            if dest == v && self.known_functions.contains(&**name) {
                                return Some(name.clone());
                            }
                        }
                        None
                    });
                    found
                } else {
                    None
                };

                if let Some(call_name) = direct_func_name {
                    // Emit a direct call instead of indirect.
                    let call_name = self.asm_label_map.get(&*call_name)
                        .cloned()
                        .unwrap_or(call_name);
                    let sig = self.func_meta.sigs.get(&*call_name);
                    let mut ret_ty = sig.map(|s| s.return_type).unwrap_or(indirect_ret_ty);
                    if sig.and_then(|s| s.two_reg_ret_size).is_some() {
                        ret_ty = IrType::I128;
                    }
                    if crate::common::types::target_is_32bit() {
                        if let Some(s) = sig {
                            if s.sret_size.is_none() && s.two_reg_ret_size.is_none() {
                                if let Some(ref rc) = s.return_ctype {
                                    if rc.is_struct_or_union() {
                                        ret_ty = IrType::I64;
                                    }
                                }
                            }
                        }
                    }
                    let callee_is_fastcall = self.fastcall_functions.contains(&*call_name);
                    self.emit(Instruction::Call {
                        func: Rc::from(call_name),
                        info: CallInfo {
                            dest: Some(dest), args: arg_vals, arg_types,
                            return_type: ret_ty, is_variadic, num_fixed_args,
                            struct_arg_sizes: sas, struct_arg_aligns: saa, struct_arg_classes: sac,
                            struct_arg_riscv_float_classes: sarfc,
                            is_sret: sret_size.is_some(), is_fastcall: callee_is_fastcall,
                            ret_eightbyte_classes: call_ret_classes,
                        },
                    });
                    ret_ty
                } else {
                    self.emit(Instruction::CallIndirect {
                        func_ptr,
                        info: CallInfo {
                            dest: Some(dest), args: arg_vals, arg_types,
                            return_type: indirect_ret_ty, is_variadic, num_fixed_args,
                            struct_arg_sizes: sas, struct_arg_aligns: saa, struct_arg_classes: sac,
                            struct_arg_riscv_float_classes: sarfc,
                            is_sret: sret_size.is_some(), is_fastcall: false,
                            ret_eightbyte_classes: call_ret_classes,
                        },
                    });
                    indirect_ret_ty
                }
            }
        }
    }

    /// Load a function pointer from a local or global variable.
    fn load_func_ptr_variable(&mut self, name: &str) -> Value {
        let base_addr = if let Some(info) = self.func_mut().locals.get(name).cloned() {
            if let Some(ref global_name) = info.static_global_name {
                let addr = self.fresh_value();
                self.emit(Instruction::GlobalAddr { dest: addr, name: global_name.clone() });
                addr
            } else {
                info.alloca
            }
        } else {
            // Global function pointer
            let addr = self.fresh_value();
            self.emit(Instruction::GlobalAddr { dest: addr, name: Rc::from(name) });
            addr
        };
        let ptr_val = self.fresh_value();
        self.emit(Instruction::Load { dest: ptr_val, ptr: base_addr, ty: IrType::Ptr , seg_override: AddressSpace::Default });
        ptr_val
    }

    /// Narrow call result if return type is smaller than the widened operation type.
    /// 128-bit return values are already correctly handled.
    pub(super) fn maybe_narrow_call_result(&mut self, dest: Value, ret_ty: IrType) -> Operand {
        let wt = crate::common::types::widened_op_type(ret_ty);
        if ret_ty != wt && ret_ty != IrType::Ptr
            && ret_ty != IrType::Void && ret_ty.is_integer()
        {
            let narrowed = self.emit_cast_val(Operand::Value(dest), wt, ret_ty);
            Operand::Value(narrowed)
        } else {
            Operand::Value(dest)
        }
    }

    /// Check if a function is variadic.
    pub(super) fn is_function_variadic(&self, name: &str) -> bool {
        if let Some(sig) = self.func_meta.sigs.get(name) {
            return sig.is_variadic;
        }
        matches!(name, "printf" | "fprintf" | "sprintf" | "snprintf" | "scanf" | "sscanf"
            | "fscanf" | "dprintf" | "vprintf" | "vfprintf" | "vsprintf" | "vsnprintf"
            | "syslog" | "err" | "errx" | "warn" | "warnx" | "asprintf" | "vasprintf"
            | "open" | "fcntl" | "ioctl" | "execl" | "execlp" | "execle")
    }

    /// Extract the return CType from a function pointer expression.
    /// Used to detect complex return types for indirect calls.
    pub(super) fn get_func_ptr_return_ctype(&self, func_expr: &Expr) -> Option<CType> {
        let mut expr = func_expr;
        loop {
            if let Some(ctype) = self.get_expr_ctype(expr) {
                return Self::extract_return_ctype_from_func_type(&ctype);
            }
            if let Expr::Deref(inner, _) = expr {
                expr = inner;
            } else {
                return None;
            }
        }
    }

    /// Extract the return CType from a function or function pointer CType.
    fn extract_return_ctype_from_func_type(ctype: &CType) -> Option<CType> {
        match ctype {
            CType::Pointer(inner, _) => match inner.as_ref() {
                CType::Function(ft) => Some(ft.return_type.clone()),
                _ => None,
            },
            CType::Function(ft) => Some(ft.return_type.clone()),
            _ => None,
        }
    }

    /// Determine the return type of a function pointer expression for indirect calls.
    /// Strips Deref layers since dereferencing a function pointer is a no-op in C.
    fn get_func_ptr_return_ir_type(&self, func_expr: &Expr) -> IrType {
        if let Some(ctype) = self.get_expr_ctype(func_expr) {
            return self.extract_return_type_from_ctype(&ctype);
        }
        // Strip all Deref layers (dereferencing function pointers is a no-op)
        let mut expr = func_expr;
        while let Expr::Deref(inner, _) = expr {
            expr = inner;
            if let Some(ctype) = self.get_expr_ctype(expr) {
                return self.extract_return_type_from_ctype(&ctype);
            }
        }
        target_int_ir_type()
    }

    pub(super) fn extract_return_type_from_ctype(&self, ctype: &CType) -> IrType {
        let returns_cld_in_regs = self.returns_complex_long_double_in_regs();
        match ctype {
            CType::Pointer(inner, _) => {
                match inner.as_ref() {
                    CType::Function(ft) => Self::func_return_ir_type(&ft.return_type, returns_cld_in_regs),
                    CType::Pointer(ret, _) => {
                        match ret.as_ref() {
                            CType::Float => IrType::F32,
                            CType::Double => IrType::F64,
                            _ => target_int_ir_type(),
                        }
                    }
                    CType::Float => IrType::F32,
                    CType::Double => IrType::F64,
                    _ => target_int_ir_type(),
                }
            }
            CType::Function(ft) => Self::func_return_ir_type(&ft.return_type, returns_cld_in_regs),
            _ => target_int_ir_type(),
        }
    }

    /// Map a function's return CType to the IR type used for the call instruction.
    /// Complex types return in FP registers (F64 for both ComplexFloat and ComplexDouble),
    /// not as Ptr which is what IrType::from_ctype gives.
    fn func_return_ir_type(ret_ctype: &CType, _returns_cld_in_regs: bool) -> IrType {
        let is_32bit = crate::common::types::target_is_32bit();
        match ret_ctype {
            // Complex float:
            //   x86-64: both F32 parts packed into xmm0 as F64
            //   ARM/RISC-V: real part in first FP reg as F32
            //   i686: both F32 parts packed into eax:edx as I64
            CType::ComplexFloat => {
                if is_32bit { IrType::I64 } else { IrType::F64 }
            }
            // Complex double:
            //   x86-64/ARM64/RISC-V: real part in first FP reg as F64
            //   i686: 16 bytes, uses sret (shouldn't reach here, but default to Ptr)
            CType::ComplexDouble => {
                if is_32bit { IrType::Ptr } else { IrType::F64 }
            }
            // Complex long double:
            //   x86-64: returns via x87 st(0)/st(1), real part as F128
            //   i686: 24 bytes, uses sret (shouldn't reach here, but default to Ptr)
            //   Other targets: handled via sret before reaching this point
            CType::ComplexLongDouble => {
                if is_32bit { IrType::Ptr } else { IrType::F128 }
            }
            // On 32-bit targets, ALL struct/union returns use sret (hidden pointer).
            // The i386 SysV ABI never returns structs in registers.
            // This case shouldn't normally be reached for sret calls, but return Ptr
            // as a safe default matching the hidden pointer convention.
            CType::Struct(_) | CType::Union(_) if is_32bit => IrType::Ptr,
            other => IrType::from_ctype(other),
        }
    }

    /// Extract parameter type information from a function pointer expression's CType.
    /// This is used as a fallback when the callee is a complex expression (e.g.,
    /// struct member function pointer like `stats->compute_stats(...)`) and we
    /// don't have a named function signature to look up.
    ///
    /// Returns `Some((param_types, param_ctypes, param_bool_flags))` if the
    /// callee's CType is a function pointer with known parameter types.
    fn extract_fn_ptr_param_info(&self, func_expr: &Expr) -> Option<(Vec<IrType>, Vec<CType>, Vec<bool>, bool)> {
        let ctype = self.get_expr_ctype(func_expr)?;
        let ft = ctype.get_function_type()?;
        if ft.params.is_empty() {
            return None; // Unprototyped function - no parameter info
        }
        let param_types: Vec<IrType> = ft.params.iter().map(|(ct, _)| IrType::from_ctype(ct)).collect();
        let param_ctypes: Vec<CType> = ft.params.iter().map(|(ct, _)| ct.clone()).collect();
        let param_bool_flags: Vec<bool> = ft.params.iter().map(|(ct, _)| matches!(ct, CType::Bool)).collect();
        Some((param_types, param_ctypes, param_bool_flags, ft.variadic))
    }
}
