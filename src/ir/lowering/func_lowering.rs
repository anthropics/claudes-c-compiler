//! Function lowering pipeline: AST function definitions -> IR functions.
//!
//! Handles the complete function lowering pipeline:
//! 1. Set up return type (handling sret, two-reg, complex ABI overrides)
//! 2. Build IR parameter list (decomposing complex params for ABI)
//! 3. Allocate parameters as locals (3-phase: basic allocas, struct setup, complex reconstruction)
//! 4. Handle K&R float promotion
//! 5. Lower the function body
//! 6. Finalize (implicit return, emit IrFunction)
//!
//! Also handles VLA parameter stride computation and dimension collection.

use std::rc::Rc;
use crate::common::fx_hash::FxHashMap;
use crate::frontend::parser::ast::{
    BlockItem,
    CompoundStmt,
    Expr,
    ForInit,
    FunctionDef,
    ParamDecl,
    Stmt,
    TypeSpecifier,
};
use crate::ir::reexports::{
    Instruction,
    IrBinOp,
    IrConst,
    IrFunction,
    IrParam,
    Operand,
    Terminator,
    Value,
};
use crate::common::types::{AddressSpace, IrType, CType};
use super::lower::Lowerer;
use super::definitions::{LocalInfo, VarInfo, FuncSig, IrParamBuildResult, ParamKind, VlaDimInfo};
use super::func_state::FunctionBuildState;

impl Lowerer {
    /// Lower a function definition to IR.
    ///
    /// Orchestrates the function lowering pipeline:
    /// 1. Set up return type (handling sret, two-reg, complex ABI overrides)
    /// 2. Build IR parameter list (decomposing complex params for ABI)
    /// 3. Allocate parameters as locals (3-phase: basic allocas, struct setup, complex reconstruction)
    /// 4. Handle K&R float promotion
    /// 5. Lower the function body
    /// 6. Finalize (implicit return, emit IrFunction)
    pub(super) fn lower_function(&mut self, func: &FunctionDef) {
        if self.defined_functions.contains(&func.name) {
            return;
        }
        self.defined_functions.insert(func.name.clone());

        let return_is_bool = self.is_type_bool(&func.return_type);
        let base_return_type = self.type_spec_to_ir(&func.return_type);
        self.func_state = Some(FunctionBuildState::new(
            func.name.clone(), base_return_type, return_is_bool,
        ));
        // Clear get_expr_ctype memoization cache: results depend on
        // per-function state (local variables), so cannot span functions.
        self.expr_ctype_cache.borrow_mut().clear();
        // Mark whether this function is an inline candidate. __builtin_constant_p
        // in non-inline functions always resolves to 0 for non-constant expressions
        // at lowering time. In inline candidates, it emits an IsConstant IR instruction
        // that can be resolved to 1 after inlining if the parameter becomes constant.
        self.func_mut().is_inline_candidate =
            func.attrs.is_always_inline() || func.attrs.is_inline() || func.attrs.is_static();
        self.push_scope();

        // Step 1: Compute ABI-adjusted return type
        let return_type = self.compute_function_return_type(func);
        self.func_mut().return_type = return_type;

        // Step 2: Build IR parameter list with ABI decomposition
        let param_info = self.build_ir_params(func);

        // Step 3: Allocate parameters as locals (3-phase)
        let entry_label = self.fresh_label();
        self.start_block(entry_label);
        self.func_mut().sret_ptr = None;
        self.allocate_function_params(func, &param_info);

        // Step 4: Evaluate VLA parameter size expressions for side effects.
        // E.g., `void foo(int a, int b[a++])` - the `a++` must be evaluated
        // so that `a` is incremented before the function body runs.
        self.evaluate_vla_param_side_effects(func);

        // Step 5: K&R float promotion
        self.handle_kr_float_promotion(func);

        // Step 5.5: Pre-scan the function body for label definitions and their scope depths.
        // This is needed so that `goto` can determine how many cleanup scopes to exit,
        // even for forward references (goto before label definition).
        let label_depths = Self::prescan_label_depths(&func.body);
        self.func_mut().user_label_depths = label_depths;

        // Step 6: Lower body
        self.lower_compound_stmt(&func.body);

        // Step 7: Finalize
        let uses_sret = param_info.uses_sret;
        self.finalize_function(func, return_type, param_info.params, uses_sret);
    }

    /// Compute the IR return type for a function, applying ABI overrides.
    fn compute_function_return_type(&mut self, func: &FunctionDef) -> IrType {
        // Record complex return type for expr_ctype resolution
        let ret_ctype = self.type_spec_to_ctype(&func.return_type);
        if ret_ctype.is_complex() {
            self.types.func_return_ctypes.insert(func.name.clone(), ret_ctype.clone());
        }

        // Use the return type from the already-registered signature, which has
        // complex ABI overrides applied (e.g., ComplexDouble -> F64, ComplexFloat -> F64/F32).
        if let Some(sig) = self.func_meta.sigs.get(&func.name) {
            // Two-register struct returns (9-16 bytes) are packed into I128 by the
            // IR lowering (Shl+Or), so the function's return type must be I128 to
            // ensure the codegen uses the register-pair return path (a0+a1).
            if sig.two_reg_ret_size.is_some() {
                return IrType::I128;
            }
            // Small vector returns (≤8 bytes) are packed into I64 by the
            // lowering. On 64-bit targets, small struct returns also use I64.
            // On 32-bit targets (i686), ALL struct/union returns use sret, so
            // this struct branch is now unreachable for i686; kept as a safety
            // fallback for indirect calls with missing signatures.
            if sig.sret_size.is_none() && sig.two_reg_ret_size.is_none() {
                if let Some(ref rc) = sig.return_ctype {
                    if rc.is_struct_or_union() && crate::common::types::target_is_32bit() {
                        return IrType::Ptr;
                    }
                    if rc.is_vector() {
                        // Small vectors (≤8 bytes) must return I64 so the packed
                        // data is returned in a register, not treated as a pointer.
                        return IrType::I64;
                    }
                }
            }
            return sig.return_type;
        }

        // Fallback: compute directly (shouldn't normally be reached)
        self.type_spec_to_ir(&func.return_type)
    }

    /// Build the IR parameter list for a function, handling ABI decomposition.
    fn build_ir_params(&mut self, func: &FunctionDef) -> IrParamBuildResult {
        let mut params: Vec<IrParam> = Vec::new();
        let mut param_kinds: Vec<ParamKind> = Vec::new();
        let mut uses_sret = false;

        // Check if function returns a large struct via sret
        if let Some(sig) = self.func_meta.sigs.get(&func.name) {
            if let Some(sret_size) = sig.sret_size {
                params.push(IrParam { ty: IrType::Ptr, struct_size: None, struct_align: None, struct_eightbyte_classes: Vec::new(), riscv_float_class: None });
                uses_sret = true;
                let _ = sret_size; // used for alloca sizing in allocate_function_params
            }
        }

        for param in func.params.iter() {
            let param_ctype = self.type_spec_to_ctype(&param.type_spec);

            // Complex parameter decomposition
            let decompose_cld = self.decomposes_complex_long_double();
            if param_ctype.is_complex() {
                // ComplexLongDouble: only decompose on ARM64 (HFA in Q regs);
                // on x86-64/RISC-V it's passed as a struct (on stack / by reference).
                if matches!(param_ctype, CType::ComplexLongDouble) && !decompose_cld {
                    // Fall through to struct handling below
                } else if matches!(param_ctype, CType::ComplexDouble) && !self.decomposes_complex_double() {
                    // i686: _Complex double (16 bytes) is passed as a struct on the stack.
                    // Fall through to struct handling below.
                } else if matches!(param_ctype, CType::ComplexFloat) && !self.decomposes_complex_float() {
                    // i686: _Complex float (8 bytes) is passed as a struct on the stack.
                    // Fall through to struct handling below.
                } else if matches!(param_ctype, CType::ComplexFloat) && self.uses_packed_complex_float() {
                    // x86-64: _Complex float packed into single F64
                    let ir_idx = params.len();
                    params.push(IrParam { ty: IrType::F64, struct_size: None, struct_align: None, struct_eightbyte_classes: Vec::new(), riscv_float_class: None });
                    param_kinds.push(ParamKind::ComplexFloatPacked(ir_idx));
                    continue;
                } else {
                    // Decompose into two FP params (ComplexFloat/ComplexDouble on 64-bit,
                    // ComplexLongDouble on ARM64 only)
                    let comp_ty = Self::complex_component_ir_type(&param_ctype);
                    let real_idx = params.len();
                    params.push(IrParam { ty: comp_ty, struct_size: None, struct_align: None, struct_eightbyte_classes: Vec::new(), riscv_float_class: None });
                    let imag_idx = params.len();
                    params.push(IrParam { ty: comp_ty, struct_size: None, struct_align: None, struct_eightbyte_classes: Vec::new(), riscv_float_class: None });
                    param_kinds.push(ParamKind::ComplexDecomposed(real_idx, imag_idx));
                    continue;
                }
            }

            // Struct/union/vector parameter (pass by value), including ComplexLongDouble
            // on x86-64/RISC-V where it's not decomposed, and ComplexDouble/ComplexFloat
            // on i686 where they are passed as structs on the stack.
            // Transparent unions are passed as their first member (a pointer),
            // not as a by-value aggregate, so struct_size is None for them.
            // Vector types (e.g. __attribute__((vector_size(N)))) are passed by value
            // like structs: the data is copied into the callee's stack slot.
            let is_complex_as_struct = matches!(param_ctype, CType::ComplexLongDouble)
                || (matches!(param_ctype, CType::ComplexDouble) && !self.decomposes_complex_double())
                || (matches!(param_ctype, CType::ComplexFloat) && !self.decomposes_complex_float());
            if self.is_type_struct_or_union(&param.type_spec)
                || is_complex_as_struct
                || param_ctype.is_vector()
            {
                let ir_idx = params.len();
                let struct_size = if self.is_transparent_union(&param.type_spec) {
                    None
                } else {
                    Some(self.sizeof_type(&param.type_spec))
                };
                // Compute struct alignment, per-eightbyte SysV ABI classification,
                // and RISC-V LP64D float field classification
                let (struct_align, struct_eightbyte_classes, riscv_float_class) = if struct_size.is_some() {
                    if let Some(layout) = self.get_struct_layout_for_type(&param.type_spec) {
                        let ctx = &*self.types.borrow_struct_layouts();
                        let classes = layout.classify_sysv_eightbytes(ctx);
                        let rv_class = layout.classify_riscv_float_fields(ctx);
                        (Some(layout.align), classes, rv_class)
                    } else {
                        (None, Vec::new(), None)
                    }
                } else {
                    (None, Vec::new(), None)
                };
                params.push(IrParam { ty: IrType::Ptr, struct_size, struct_align, struct_eightbyte_classes, riscv_float_class });
                param_kinds.push(ParamKind::Struct(ir_idx));
                continue;
            }

            // Normal scalar parameter
            let ir_idx = params.len();
            let mut ty = self.type_spec_to_ir(&param.type_spec);
            // K&R default argument promotions: float->double, char/short->int
            if func.is_kr {
                ty = match ty {
                    IrType::F32 => IrType::F64,
                    IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 => IrType::I32,
                    other => other,
                };
            }
            params.push(IrParam { ty, struct_size: None, struct_align: None, struct_eightbyte_classes: Vec::new(), riscv_float_class: None });
            param_kinds.push(ParamKind::Normal(ir_idx));
        }

        IrParamBuildResult { params, param_kinds, uses_sret }
    }

    /// Allocate function parameters as local variables (3-phase process).
    fn allocate_function_params(&mut self, func: &FunctionDef, info: &IrParamBuildResult) {
        // Phase 1: Emit allocas for all IR params
        let mut ir_allocas: Vec<Value> = Vec::new();
        for param in &info.params {
            let alloca = self.fresh_value();
            if info.uses_sret && ir_allocas.is_empty() {
                let ptr_size = crate::common::types::target_ptr_size();
                self.emit(Instruction::Alloca { dest: alloca, ty: IrType::Ptr, size: ptr_size, align: 0, volatile: false });
                self.func_mut().sret_ptr = Some(alloca);
                ir_allocas.push(alloca);
                continue;
            }
            let size = param.ty.size().max(param.struct_size.unwrap_or(param.ty.size()));
            self.emit(Instruction::Alloca { dest: alloca, ty: param.ty, size, align: 0, volatile: false });
            ir_allocas.push(alloca);
        }

        // Save param alloca values for the backend to detect unused param allocas.
        self.func_mut().param_alloca_values = ir_allocas.clone();

        // Phase 1b: Emit explicit ParamRef + Store for scalar parameter allocas.
        // This makes the parameter->alloca store visible in the IR so mem2reg
        // can promote parameter allocas to SSA values, enabling constant
        // propagation when parameters are reassigned (e.g., `x &= 0`).
        // Only non-struct scalar params are eligible: struct params (struct_size != None)
        // use special by-value handling in emit_store_params that we must not duplicate.
        // Also skip sret pointer params (first param when uses_sret is true).
        for (i, param) in info.params.iter().enumerate() {
            // Skip struct params - they have special ABI handling
            if param.struct_size.is_some() {
                continue;
            }
            // Skip sret pointer param
            if info.uses_sret && i == 0 {
                continue;
            }
            let ty_size = param.ty.size();
            if ty_size <= 8 {
                let param_val = self.fresh_value();
                self.emit(Instruction::ParamRef { dest: param_val, param_idx: i, ty: param.ty });
                self.emit(Instruction::Store {
                    val: Operand::Value(param_val),
                    ptr: ir_allocas[i],
                    ty: param.ty,
                    seg_override: crate::common::types::AddressSpace::Default,
                });
            }
        }

        // Phase 2 & 3: Process each original parameter by kind
        for (orig_idx, kind) in info.param_kinds.iter().enumerate() {
            let orig_param = &func.params[orig_idx];
            match kind {
                ParamKind::Normal(ir_idx) => {
                    self.register_normal_param(orig_param, &info.params[*ir_idx], ir_allocas[*ir_idx]);
                }
                ParamKind::Struct(ir_idx) => {
                    self.register_struct_or_complex_param(orig_param, ir_allocas[*ir_idx]);
                }
                ParamKind::ComplexFloatPacked(ir_idx) => {
                    self.register_packed_complex_float_param(orig_param, ir_allocas[*ir_idx]);
                }
                ParamKind::ComplexDecomposed(real_ir_idx, imag_ir_idx) => {
                    self.reconstruct_decomposed_complex_param(
                        orig_param, ir_allocas[*real_ir_idx], ir_allocas[*imag_ir_idx],
                    );
                }
            }
        }

        self.compute_vla_param_strides(func);
    }

    /// Register a normal (non-struct, non-complex) parameter as a local variable.
    fn register_normal_param(&mut self, orig_param: &ParamDecl, ir_param: &IrParam, alloca: Value) {
        let ty = ir_param.ty;
        let param_size = self.sizeof_type(&orig_param.type_spec).max(ty.size());
        let elem_size = if ty == IrType::Ptr { self.pointee_elem_size(&orig_param.type_spec) } else { 0 };
        let pointee_type = if ty == IrType::Ptr { self.pointee_ir_type(&orig_param.type_spec) } else { None };
        let struct_layout = if ty == IrType::Ptr { self.get_struct_layout_for_pointer_param(&orig_param.type_spec) } else { None };
        let c_type = Some(self.param_ctype(orig_param));
        let is_bool = self.is_type_bool(&orig_param.type_spec);
        let array_dim_strides = if ty == IrType::Ptr { self.compute_ptr_array_strides(&orig_param.type_spec) } else { vec![] };

        // Detect pointer-to-function-pointer parameters using the parser's
        // fptr_inner_ptr_depth field. This records how many `*` were inside the
        // parenthesized declarator: (*fp) has depth 1, (**fpp) has depth 2.
        // A depth >= 2 means pointer-to-function-pointer. This correctly
        // distinguishes `int (**fpp)(int)` (depth 2, ptr-to-func-ptr) from
        // `void *(*fp)(size_t)` (depth 1, func ptr returning void*), which
        // have identical CType representations.
        let is_ptr_to_func_ptr = orig_param.fptr_params.is_some()
            && orig_param.fptr_inner_ptr_depth >= 2;

        let name: Rc<str> = orig_param.name.clone().unwrap_or_else(|| Rc::from(""));
        self.insert_local_scoped(name, LocalInfo {
            var: VarInfo { ty, elem_size, is_array: false, pointee_type, struct_layout, is_struct: false, array_dim_strides, c_type, is_ptr_to_func_ptr, address_space: AddressSpace::Default, explicit_alignment: None },
            alloca, alloc_size: param_size, is_bool, static_global_name: None, vla_strides: vec![], vla_size: None, asm_register: None, asm_register_has_init: false, cleanup_fn: None,
            is_const: orig_param.is_const,
        });

        // Register function pointer parameter signatures for indirect calls
        if let Some(ref fptr_params) = orig_param.fptr_params {
            let ret_ty = match &orig_param.type_spec {
                TypeSpecifier::Pointer(inner, _) => self.type_spec_to_ir(inner),
                _ => self.type_spec_to_ir(&orig_param.type_spec),
            };
            if let Some(ref name) = orig_param.name {
                let param_tys: Vec<IrType> = fptr_params.iter().map(|fp| self.type_spec_to_ir(&fp.type_spec)).collect();
                self.func_meta.ptr_sigs.insert(name.clone(), FuncSig::for_ptr(ret_ty, param_tys));
            }
        } else if let Some(ref name) = orig_param.name {
            // Fallback: check if the parameter type is a bare function typedef
            // (e.g., `typedef int filler_t(void*, void*); void foo(filler_t filler)`).
            // Per C11 6.7.6.3p8, function-type parameters decay to pointer-to-function.
            // The parser doesn't set fptr_params for typedef-based function types, so
            // we must register the ptr_sig from the function typedef info here.
            if let TypeSpecifier::TypedefName(tname) = &orig_param.type_spec {
                if let Some(fti) = self.types.function_typedefs.get(tname).cloned() {
                    let ret_ty = self.type_spec_to_ir(&fti.return_type);
                    let param_tys: Vec<IrType> = fti.params.iter()
                        .map(|fp| self.type_spec_to_ir(&fp.type_spec))
                        .collect();
                    self.func_meta.ptr_sigs.insert(name.clone(), FuncSig::for_ptr(ret_ty, param_tys));
                }
            }
        }
    }

    /// Register a struct/union or non-decomposed complex parameter as a local variable.
    fn register_struct_or_complex_param(&mut self, orig_param: &ParamDecl, alloca: Value) {
        let is_struct = self.is_type_struct_or_union(&orig_param.type_spec);
        let layout = if is_struct { self.get_struct_layout_for_type(&orig_param.type_spec) } else { None };
        let size = if is_struct { layout.as_ref().map_or(8, |l| l.size) } else { self.sizeof_type(&orig_param.type_spec) };
        let c_type = Some(self.type_spec_to_ctype(&orig_param.type_spec));

        let name = orig_param.name.clone().unwrap_or_else(|| Rc::from(""));
        self.insert_local_scoped(name, LocalInfo {
            var: VarInfo { ty: IrType::Ptr, elem_size: 0, is_array: false, pointee_type: None, struct_layout: layout, is_struct: true, array_dim_strides: vec![], c_type, is_ptr_to_func_ptr: false, address_space: AddressSpace::Default, explicit_alignment: None },
            alloca, alloc_size: size, is_bool: false, static_global_name: None, vla_strides: vec![], vla_size: None, asm_register: None, asm_register_has_init: false, cleanup_fn: None,
            is_const: orig_param.is_const,
        });
    }

    /// Register a packed complex float parameter (x86-64 only) as a local variable.
    fn register_packed_complex_float_param(&mut self, orig_param: &ParamDecl, alloca: Value) {
        let ct = self.type_spec_to_ctype(&orig_param.type_spec);
        let name = orig_param.name.clone().unwrap_or_else(|| Rc::from(""));
        self.insert_local_scoped(name, LocalInfo {
            var: VarInfo { ty: IrType::Ptr, elem_size: 0, is_array: false, pointee_type: None, struct_layout: None, is_struct: true, array_dim_strides: vec![], c_type: Some(ct), is_ptr_to_func_ptr: false, address_space: AddressSpace::Default, explicit_alignment: None },
            alloca, alloc_size: 8, is_bool: false, static_global_name: None, vla_strides: vec![], vla_size: None, asm_register: None, asm_register_has_init: false, cleanup_fn: None,
            is_const: orig_param.is_const,
        });
    }

    /// Reconstruct a decomposed complex parameter from its real/imag Phase 1 allocas.
    fn reconstruct_decomposed_complex_param(&mut self, orig_param: &ParamDecl, real_alloca: Value, imag_alloca: Value) {
        let ct = self.type_spec_to_ctype(&orig_param.type_spec);
        let comp_ty = Self::complex_component_ir_type(&ct);
        let comp_size = Self::complex_component_size(&ct);
        let complex_size = ct.size();

        let complex_alloca = self.fresh_value();
        self.emit(Instruction::Alloca { dest: complex_alloca, ty: IrType::Ptr, size: complex_size, align: 0, volatile: false });

        let real_val = self.fresh_value();
        self.emit(Instruction::Load { dest: real_val, ptr: real_alloca, ty: comp_ty , seg_override: AddressSpace::Default });
        self.emit(Instruction::Store { val: Operand::Value(real_val), ptr: complex_alloca, ty: comp_ty , seg_override: AddressSpace::Default });

        let imag_val = self.fresh_value();
        self.emit(Instruction::Load { dest: imag_val, ptr: imag_alloca, ty: comp_ty , seg_override: AddressSpace::Default });
        let imag_ptr = self.fresh_value();
        self.emit(Instruction::GetElementPtr { dest: imag_ptr, base: complex_alloca, offset: Operand::Const(IrConst::I64(comp_size as i64)), ty: IrType::I8 });
        self.emit(Instruction::Store { val: Operand::Value(imag_val), ptr: imag_ptr, ty: comp_ty , seg_override: AddressSpace::Default });

        let name = orig_param.name.clone().unwrap_or_else(|| Rc::from(""));
        self.func_mut().locals.insert(name, LocalInfo {
            var: VarInfo { ty: IrType::Ptr, elem_size: 0, is_array: false, pointee_type: None, struct_layout: None, is_struct: true, array_dim_strides: vec![], c_type: Some(ct), is_ptr_to_func_ptr: false, address_space: AddressSpace::Default, explicit_alignment: None },
            alloca: complex_alloca, alloc_size: complex_size, is_bool: false, static_global_name: None, vla_strides: vec![], vla_size: None, asm_register: None, asm_register_has_init: false, cleanup_fn: None,
            is_const: orig_param.is_const,
        });
    }

    /// Handle K&R default argument promotions: narrow promoted params back to declared types.
    /// float->double promotion: narrow double back to float.
    /// char/short->int promotion: narrow int back to char/short.
    fn handle_kr_float_promotion(&mut self, func: &FunctionDef) {
        if !func.is_kr { return; }
        for param in &func.params {
            let declared_ty = self.type_spec_to_ir(&param.type_spec);
            let name: Rc<str> = param.name.clone().unwrap_or_else(|| Rc::from(""));
            let local_info = match self.func_mut().locals.get(&name).cloned() { Some(i) => i, None => continue };
            match declared_ty {
                IrType::F32 => {
                    // Received as F64, narrow to F32
                    let f64_val = self.fresh_value();
                    self.emit(Instruction::Load { dest: f64_val, ptr: local_info.alloca, ty: IrType::F64 , seg_override: AddressSpace::Default });
                    let f32_val = self.emit_cast_val(Operand::Value(f64_val), IrType::F64, IrType::F32);
                    let f32_alloca = self.fresh_value();
                    self.emit(Instruction::Alloca { dest: f32_alloca, ty: IrType::F32, size: 4, align: 0, volatile: false });
                    self.emit(Instruction::Store { val: Operand::Value(f32_val), ptr: f32_alloca, ty: IrType::F32 , seg_override: AddressSpace::Default });
                    if let Some(local) = self.func_mut().locals.get_mut(&name) {
                        local.alloca = f32_alloca; local.ty = IrType::F32; local.alloc_size = 4;
                    }
                }
                IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 => {
                    // Received as I32, narrow to declared type
                    let i32_val = self.fresh_value();
                    self.emit(Instruction::Load { dest: i32_val, ptr: local_info.alloca, ty: IrType::I32 , seg_override: AddressSpace::Default });
                    let narrow_val = self.emit_cast_val(Operand::Value(i32_val), IrType::I32, declared_ty);
                    let narrow_alloca = self.fresh_value();
                    let size = declared_ty.size().max(1);
                    self.emit(Instruction::Alloca { dest: narrow_alloca, ty: declared_ty, size, align: 0, volatile: false });
                    self.emit(Instruction::Store { val: Operand::Value(narrow_val), ptr: narrow_alloca, ty: declared_ty , seg_override: AddressSpace::Default });
                    if let Some(local) = self.func_mut().locals.get_mut(&name) {
                        local.alloca = narrow_alloca; local.ty = declared_ty; local.alloc_size = size;
                    }
                }
                _ => {}
            }
        }
    }

    /// Finalize a function: add implicit return, build IrFunction, push to module.
    fn finalize_function(&mut self, func: &FunctionDef, return_type: IrType, params: Vec<IrParam>, uses_sret: bool) {
        if !self.func_mut().instrs.is_empty() || self.func_mut().blocks.is_empty()
           || !matches!(self.func_mut().blocks.last().map(|b| &b.terminator), Some(Terminator::Return(_)))
        {
            let ret_op = if return_type == IrType::Void { None } else { Some(Operand::Const(IrConst::I32(0))) };
            self.terminate(Terminator::Return(ret_op));
        }

        // Merge deferred entry-block allocas into blocks[0], right after
        // the existing parameter allocas. mem2reg skips the first N allocas
        // (where N = param count) so deferred local allocas must come after them.
        let entry_allocas = std::mem::take(&mut self.func_mut().entry_allocas);
        if !entry_allocas.is_empty() {
            if let Some(entry_block) = self.func_mut().blocks.first_mut() {
                // Find the insertion point: right after the initial contiguous run
                // of Alloca instructions at the start of the entry block (parameter allocas).
                // We must NOT use rposition (last alloca anywhere in the block) because
                // SSE intrinsic lowering emits inline allocas deep in the block via
                // self.emit(Alloca), and placing deferred allocas after those would put
                // them after their uses, violating dominance.
                let insert_pos = entry_block.instructions.iter()
                    .position(|inst| !matches!(inst, Instruction::Alloca { .. }))
                    .unwrap_or(entry_block.instructions.len());
                // Splice deferred allocas at the insertion point.
                let num_allocas = entry_allocas.len();
                entry_block.instructions.splice(insert_pos..insert_pos, entry_allocas);
                // Insert dummy spans for the spliced allocas
                if !entry_block.source_spans.is_empty() {
                    let dummy_spans = vec![crate::common::source::Span::dummy(); num_allocas];
                    entry_block.source_spans.splice(insert_pos..insert_pos, dummy_spans);
                }
            }
        }

        // Inline linkage rules (GNU89 vs C99):
        //
        // GNU89 mode (explicit __attribute__((gnu_inline)) OR -fgnu89-inline / -std=c89):
        //   extern inline = inline definition only, no external def → static
        //   inline (no extern) = external definition → global
        //
        // C99 mode (default for -std=c99 and later):
        //   extern inline = provides external definition → global
        //   plain inline (no extern) = inline definition only → static (local)
        //
        // C99 6.7.4p7 additional rule: if ANY file-scope declaration of the function
        // does NOT include `inline`, then the definition provides an external definition.
        // This handles cases like jq's tsd_dtoa_context_get() where the header declares
        // the function without `inline` and the .c file defines it with `inline`.
        let is_gnu_inline_no_extern_def = self.is_gnu_inline_no_extern_def(&func.attrs);
        // C99 6.7.4p7: A plain `inline` definition (without `extern`) does not
        // provide an external definition ONLY if ALL file-scope declarations include
        // `inline`. If any declaration lacks `inline`, this is an external definition.
        // Note: in GNU89 mode, `inline` without `extern` provides an external def,
        // so this rule does not apply.
        let is_c99_inline_def = !self.gnu89_inline
            && func.attrs.is_inline() && !func.attrs.is_extern()
            && !func.attrs.is_static() && !func.attrs.is_gnu_inline()
            && !self.has_non_inline_decl.contains(&func.name);
        // C99 inline-only definitions (inline without extern/static, all declarations
        // have inline) don't provide an external definition per C99 6.7.4p7.
        // We lower them as static so their bodies are available for inlining.
        // If all call sites are inlined, dead code elimination removes them.
        // If not inlined, they're emitted as local symbols (safe fallback).
        let is_static = func.attrs.is_static() || self.static_functions.contains(&func.name)
            || is_gnu_inline_no_extern_def || is_c99_inline_def;
        let next_val = self.func_mut().next_value;
        let param_alloca_vals = std::mem::take(&mut self.func_mut().param_alloca_values);
        let global_init_labels = std::mem::take(&mut self.func_mut().global_init_label_blocks);
        // Get return eightbyte classes from the function's signature metadata
        let ret_eightbyte_classes = self.func_meta.sigs.get(&func.name)
            .map(|s| s.ret_eightbyte_classes.clone())
            .unwrap_or_default();
        let ir_func = IrFunction {
            name: func.name.clone(), return_type, params,
            blocks: std::mem::take(&mut self.func_mut().blocks),
            is_variadic: func.variadic, is_declaration: false, is_static,
            is_inline: func.attrs.is_inline(),
            is_always_inline: func.attrs.is_always_inline(),
            is_noinline: func.attrs.is_noinline(),
            next_value_id: next_val,
            section: func.attrs.section.clone(),
            visibility: func.attrs.visibility.clone(),
            is_weak: func.attrs.is_weak(),
            is_used: func.attrs.is_used(),
            has_inlined_calls: false,
            param_alloca_values: param_alloca_vals,
            uses_sret,
            is_fastcall: func.attrs.is_fastcall(),
            is_naked: func.attrs.is_naked(),
            global_init_label_blocks: global_init_labels,
            ret_eightbyte_classes,
            is_gnu_inline_def: is_gnu_inline_no_extern_def,
        };
        // Collect __attribute__((symver("..."))) directives
        if let Some(ref sv) = func.attrs.symver {
            self.module.symver_directives.push((func.name.clone(), Rc::from(sv.as_str())));
        }
        self.module.functions.push(ir_func);
        self.pop_scope();
        self.func_state = None;
    }

    /// Evaluate VLA parameter size expressions for their side effects.
    ///
    /// In C, array parameter declarations like `void foo(int a, int b[a++])` have
    /// the outermost dimension decayed to a pointer, but the size expression (here
    /// `a++`) must still be evaluated for its side effects. The expression was
    /// preserved in `ParamDecl::vla_size_exprs` during parsing; we evaluate it here
    /// (discarding the result) so that side effects take effect before the body runs.
    fn evaluate_vla_param_side_effects(&mut self, func: &FunctionDef) {
        for param in &func.params {
            for expr in &param.vla_size_exprs {
                // Evaluate the expression for its side effects; discard the result.
                let _ = self.lower_expr(expr);
            }
        }
    }

    /// For pointer-to-array function parameters with VLA (runtime) dimensions,
    /// compute strides at runtime and store them in the LocalInfo.
    fn compute_vla_param_strides(&mut self, func: &FunctionDef) {
        // Collect VLA info first, then emit code (avoids borrow issues)
        let mut vla_params: Vec<(Rc<str>, Vec<VlaDimInfo>)> = Vec::new();

        for param in &func.params {
            let param_name = match &param.name {
                Some(n) => n.clone(),
                None => continue,
            };

            // Check if this parameter is a pointer-to-array with VLA dimensions
            let ts = self.resolve_type_spec(&param.type_spec);
            if let TypeSpecifier::Pointer(inner, _) = ts {
                let dim_infos = self.collect_vla_dims(inner);
                if dim_infos.iter().any(|d| d.is_vla) {
                    vla_params.push((param_name, dim_infos));
                }
            } else {
                // Check CType for typedef'd pointer-to-array
                let ctype = self.type_spec_to_ctype(&param.type_spec);
                if let CType::Pointer(ref inner_ct, _) = ctype {
                    if matches!(inner_ct.as_ref(), CType::Array(_, _)) {
                        // TypeSpecifier-based VLA detection won't work for typedef'd types,
                        // but VLA dimensions in typedef'd pointers are rare
                    }
                }
            }
        }

        // Now emit runtime stride computations
        for (param_name, dim_infos) in vla_params {
            let num_strides = dim_infos.len() + 1; // +1 for base element size
            let mut vla_strides: Vec<Option<Value>> = vec![None; num_strides];

            let base_elem_size = dim_infos.last().map_or(1, |d| d.base_elem_size);
            let mut current_stride: Option<Value> = None;
            let mut current_const_stride = base_elem_size;

            // Process dimensions from innermost to outermost
            for (i, dim_info) in dim_infos.iter().enumerate().rev() {
                if dim_info.is_vla {
                    let ptr_int_ty = crate::common::types::target_int_ir_type();
                    let dim_val = self.load_vla_dim_value(&dim_info.dim_expr_name);
                    let stride_val = if let Some(prev) = current_stride {
                        self.emit_binop_val(IrBinOp::Mul, Operand::Value(dim_val), Operand::Value(prev), ptr_int_ty)
                    } else {
                        self.emit_binop_val(IrBinOp::Mul, Operand::Value(dim_val), Operand::Const(IrConst::ptr_int(current_const_stride as i64)), ptr_int_ty)
                    };
                    vla_strides[i] = Some(stride_val);
                    current_stride = Some(stride_val);
                    current_const_stride = 0;
                } else {
                    let const_dim = dim_info.const_size.unwrap_or(1) as usize;
                    if let Some(prev) = current_stride {
                        let ptr_int_ty = crate::common::types::target_int_ir_type();
                        let result = self.emit_binop_val(IrBinOp::Mul, Operand::Value(prev), Operand::Const(IrConst::ptr_int(const_dim as i64)), ptr_int_ty);
                        vla_strides[i] = Some(result);
                        current_stride = Some(result);
                    } else {
                        current_const_stride *= const_dim;
                    }
                }
            }

            if let Some(local) = self.func_mut().locals.get_mut(&param_name) {
                local.vla_strides = vla_strides;
            }
        }
    }

    /// Load the value of a VLA dimension variable (a function parameter).
    fn load_vla_dim_value(&mut self, dim_name: &str) -> Value {
        if let Some(info) = self.func_mut().locals.get(dim_name).cloned() {
            let loaded = self.fresh_value();
            self.emit(Instruction::Load { dest: loaded, ptr: info.alloca, ty: info.ty,
             seg_override: AddressSpace::Default });
            loaded
        } else {
            // Fallback: use constant 1
            let val = self.fresh_value();
            self.emit(Instruction::Copy {
                dest: val,
                src: Operand::Const(IrConst::ptr_int(1)),
            });
            val
        }
    }

    /// Collect VLA dimension information from a pointer-to-array type.
    fn collect_vla_dims(&self, inner: &TypeSpecifier) -> Vec<VlaDimInfo> {
        let mut dims = Vec::new();
        let mut current = inner;
        loop {
            let resolved = self.resolve_type_spec(current);
            if let TypeSpecifier::Array(elem, size_expr) = resolved {
                let (is_vla, dim_name, const_size) = if let Some(expr) = size_expr {
                    if let Some(val) = self.expr_as_array_size(expr) {
                        (false, Rc::from(""), Some(val))
                    } else {
                        let name = Self::extract_dim_expr_name(expr);
                        (true, name, None)
                    }
                } else {
                    (false, Rc::from(""), None)
                };

                let base_elem_size = self.sizeof_type(elem);

                dims.push(VlaDimInfo {
                    is_vla,
                    dim_expr_name: dim_name,
                    const_size,
                    base_elem_size,
                });
                current = elem;
            } else {
                break;
            }
        }
        dims
    }

    /// Extract variable name from a VLA dimension expression.
    fn extract_dim_expr_name(expr: &Expr) -> Rc<str> {
        match expr {
            Expr::Identifier(name, _) => name.clone(),
            _ => Rc::from(""),
        }
    }

    /// Pre-scan a function body to find all label definitions and their scope depths.
    ///
    /// This walks the AST without generating IR, just tracking compound statement
    /// nesting depth. The result maps each label name to its scope depth (number
    /// of compound statements it's nested within, starting from 1 for the function
    /// body itself, matching the scope_stack depth during lowering since push_scope
    /// is called before lowering the function body).
    ///
    /// This information is used by `lower_goto_stmt` to determine which cleanup
    /// scopes need to be exited: only scopes deeper than the target label's depth
    /// should have their cleanup destructors called.
    fn prescan_label_depths(body: &CompoundStmt) -> FxHashMap<Rc<str>, usize> {
        let mut result = FxHashMap::default();
        // depth starts at 1 because lower_function calls push_scope() before
        // lower_compound_stmt, and then lower_compound_stmt calls push_scope again
        // for the function body's own compound statement.
        Self::prescan_compound_stmt(body, 1, &mut result);
        result
    }

    fn prescan_compound_stmt(compound: &CompoundStmt, depth: usize, result: &mut FxHashMap<Rc<str>, usize>) {
        // Match the lowering behavior: only push a scope (increment depth) when the
        // compound statement contains declarations. Declaration-free compound statements
        // don't push a scope in lower_compound_stmt, so we must not increment depth here.
        let has_declarations = compound.items.iter().any(|item| matches!(item, BlockItem::Declaration(_)));
        let inner_depth = if has_declarations { depth + 1 } else { depth };
        for item in &compound.items {
            match item {
                BlockItem::Statement(stmt) => Self::prescan_stmt(stmt, inner_depth, result),
                BlockItem::Declaration(_) => {}
            }
        }
    }

    fn prescan_stmt(stmt: &Stmt, depth: usize, result: &mut FxHashMap<Rc<str>, usize>) {
        match stmt {
            Stmt::Label(name, inner_stmt, _) => {
                // Record the label at the current scope depth.
                result.insert(name.clone(), depth);
                Self::prescan_stmt(inner_stmt, depth, result);
            }
            Stmt::Compound(compound) => {
                Self::prescan_compound_stmt(compound, depth, result);
            }
            Stmt::If(_, then_stmt, else_stmt, _) => {
                Self::prescan_stmt(then_stmt, depth, result);
                if let Some(else_stmt) = else_stmt {
                    Self::prescan_stmt(else_stmt, depth, result);
                }
            }
            Stmt::While(_, body, _) | Stmt::DoWhile(body, _, _) => {
                Self::prescan_stmt(body, depth, result);
            }
            Stmt::For(init, _, _, body, _) => {
                // C99: for-init declarations have their own scope, matching
                // the push_scope() in lower_for_stmt.
                let has_decl_init = init.as_ref().is_some_and(|i| matches!(i.as_ref(), ForInit::Declaration(_)));
                let for_depth = if has_decl_init { depth + 1 } else { depth };
                Self::prescan_stmt(body, for_depth, result);
            }
            Stmt::Switch(_, body, _) => {
                Self::prescan_stmt(body, depth, result);
            }
            Stmt::Case(_, inner, _) | Stmt::CaseRange(_, _, inner, _) | Stmt::Default(inner, _) => {
                Self::prescan_stmt(inner, depth, result);
            }
            // Leaf statements: no labels inside
            Stmt::Expr(_) | Stmt::Return(_, _) | Stmt::Break(_) | Stmt::Continue(_) |
            Stmt::Goto(_, _) | Stmt::GotoIndirect(_, _) | Stmt::Declaration(_) |
            Stmt::InlineAsm { .. } => {}
        }
    }
}
