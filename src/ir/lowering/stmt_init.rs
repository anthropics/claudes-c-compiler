//! Local variable initialization lowering.
//!
//! Extracted from stmt.rs to reduce that file's size. Contains the logic for
//! lowering `Initializer::Expr` and `Initializer::List` for local variable
//! declarations, plus helpers for registering block-scope function declarations.

use crate::frontend::parser::ast::{
    Declaration,
    DerivedDeclarator,
    Designator,
    Expr,
    InitDeclarator,
    Initializer,
    InitializerItem,
    ParamDecl,
    TypeSpecifier,
};
use crate::ir::reexports::{
    Instruction,
    IrConst,
    Operand,
    Value,
};
use std::rc::Rc;
use crate::common::types::{AddressSpace, IrType, CType};
use super::lower::Lowerer;
use super::definitions::{GlobalInfo, DeclAnalysis, FuncSig};

impl Lowerer {
    /// Handle extern declarations inside function bodies.
    /// They reference a global symbol, not a local variable.
    pub(super) fn lower_extern_decl(
        &mut self,
        decl: &Declaration,
        declarator: &InitDeclarator,
    ) -> bool {
        // Remove from locals, but track in scope frame so pop_scope()
        // restores the local when this block exits.
        self.shadow_local_for_scope(&declarator.name);
        // Also remove from static_local_names so that the extern name
        // resolves to the true global, not a same-named static local
        self.shadow_static_for_scope(&declarator.name);

        // Check if this is a function declaration (extern int f(int))
        let is_func_decl = declarator.derived.iter().any(|d| matches!(d, DerivedDeclarator::Function(_, _)));
        if is_func_decl {
            return false; // Fall through to the function declaration handler
        }

        if !self.globals.contains_key(&declarator.name) {
            let mut ext_da = self.analyze_declaration(&decl.type_spec, &declarator.derived);
            let elem_size = ext_da.c_type.as_ref().map_or(0, |ct| ct.size());
            if let Some(vs) = decl.resolve_vector_size(elem_size) {
                ext_da.apply_vector_size(vs);
            }
            self.globals.insert(declarator.name.clone(), GlobalInfo::from_analysis(&ext_da));
        }
        true // Handled, caller should continue to next declarator
    }

    /// Handle block-scope function declarations: `int f(int);`
    /// These declare an external function, not a local variable.
    /// Returns true if this was a function declaration (caller should continue to next declarator).
    pub(super) fn try_lower_block_func_decl(
        &mut self,
        decl: &Declaration,
        declarator: &InitDeclarator,
    ) -> bool {
        // If there's an initializer, this is a variable declaration, not a function declaration
        if declarator.init.is_some() {
            return false;
        }

        // Check for direct function declarator: int f(int, int)
        // A block-scope function declaration has the form: type name(params);
        // The derived list starts with Function (possibly preceded by Pointer for
        // return type indirection like `int *f(int)`).
        // If we encounter a FunctionPointer before Function, this is a variable
        // with a function pointer type, not a function declaration.
        let mut ptr_count = 0;
        let mut func_info = None;
        let mut has_fptr_before_func = false;
        for d in &declarator.derived {
            match d {
                DerivedDeclarator::Pointer => ptr_count += 1,
                DerivedDeclarator::Function(p, v) => {
                    func_info = Some((p.clone(), *v));
                    break;
                }
                DerivedDeclarator::FunctionPointer(_, _) => {
                    has_fptr_before_func = true;
                    break;
                }
                _ => {}
            }
        }
        // If we found a FunctionPointer before a Function, this is a function pointer
        // variable (e.g., int (*(*p)(int))(int)), not a function declaration.
        if has_fptr_before_func {
            return false;
        }
        if let Some((params, variadic)) = func_info {
            self.register_block_func_meta(&declarator.name, &decl.type_spec, ptr_count, &params, variadic);
            self.shadow_local_for_scope(&declarator.name);
            // Propagate __attribute__((error("..."))) and __attribute__((noreturn)) / _Noreturn
            // from block-scope function declarations. Without this, the kernel's BUILD_BUG_ON
            // macro (which uses block-scope `extern void __compiletime_assert_NNN(void)
            // __attribute__((error(...)))`) would emit real calls to undefined symbols
            // instead of replacing them with Unreachable.
            if declarator.attrs.is_error_attr() && !declarator.name.is_empty() {
                self.error_functions.insert(declarator.name.clone());
            }
            if declarator.attrs.is_noreturn() && !declarator.name.is_empty() {
                self.noreturn_functions.insert(declarator.name.clone());
            }
            // Propagate __attribute__((weak)) and __attribute__((visibility(...)))
            // from block-scope extern function declarations.
            if !declarator.name.is_empty()
                && (declarator.attrs.is_weak() || declarator.attrs.visibility.is_some())
            {
                self.module.symbol_attrs.push((
                    declarator.name.clone(),
                    declarator.attrs.is_weak(),
                    declarator.attrs.visibility.clone(),
                ));
            }
            return true;
        }

        // Also handle typedef-based function declarations in block scope
        // (e.g., `func_t add;` where func_t is `typedef int func_t(int);`)
        if declarator.derived.is_empty() && declarator.init.is_none() {
            if let TypeSpecifier::TypedefName(tname) = &decl.type_spec {
                if let Some(fti) = self.types.function_typedefs.get(tname).cloned() {
                    self.register_block_func_meta(&declarator.name, &fti.return_type, 0, &fti.params, fti.variadic);
                    self.shadow_local_for_scope(&declarator.name);
                    if declarator.attrs.is_error_attr() && !declarator.name.is_empty() {
                        self.error_functions.insert(declarator.name.clone());
                    }
                    if declarator.attrs.is_noreturn() && !declarator.name.is_empty() {
                        self.noreturn_functions.insert(declarator.name.clone());
                    }
                    if declarator.attrs.is_weak() || declarator.attrs.visibility.is_some() {
                        self.module.symbol_attrs.push((
                            declarator.name.clone(),
                            declarator.attrs.is_weak(),
                            declarator.attrs.visibility.clone(),
                        ));
                    }
                    return true;
                }
            }
        }

        false
    }

    /// Register function metadata for a block-scope function declaration.
    /// Must compute the same ABI metadata as `register_function_meta` so that
    /// member access on struct return values (e.g., `stfunc1().field`) works
    /// correctly even when the function is only forward-declared at block scope.
    fn register_block_func_meta(
        &mut self,
        name: &str,
        ret_type_spec: &TypeSpecifier,
        ptr_count: usize,
        params: &[ParamDecl],
        variadic: bool,
    ) {
        self.known_functions.insert(Rc::from(name));
        let mut ret_ty = self.type_spec_to_ir(ret_type_spec);
        if ptr_count > 0 {
            ret_ty = IrType::Ptr;
        }

        // Complex return types need special IR type overrides (same as register_function_meta)
        if ptr_count == 0 {
            let ret_ctype = self.type_spec_to_ctype(ret_type_spec);
            if matches!(ret_ctype, CType::ComplexDouble) && self.decomposes_complex_double() {
                ret_ty = IrType::F64;
            } else if matches!(ret_ctype, CType::ComplexFloat) {
                if self.uses_packed_complex_float() {
                    ret_ty = IrType::F64;
                } else if !self.decomposes_complex_float() {
                    // i686: _Complex float (8 bytes) packed in eax:edx as I64
                    ret_ty = IrType::I64;
                } else {
                    ret_ty = IrType::F32;
                }
            }
        }

        // Track CType for pointer-returning and struct-returning functions.
        // Without this, member access on a call result (e.g., `func().field`)
        // cannot resolve the struct layout and falls back to offset 0.
        let mut return_ctype = None;
        if ret_ty == IrType::Ptr {
            let base_ctype = self.type_spec_to_ctype(ret_type_spec);
            let ret_ctype = if ptr_count > 0 {
                let mut ct = base_ctype;
                for _ in 0..ptr_count {
                    ct = CType::Pointer(Box::new(ct), AddressSpace::Default);
                }
                ct
            } else {
                base_ctype
            };
            return_ctype = Some(ret_ctype);
        }

        // Record complex return types for expr_ctype resolution
        if ptr_count == 0 {
            let ret_ct = self.type_spec_to_ctype(ret_type_spec);
            if ret_ct.is_complex() {
                self.types.func_return_ctypes.insert(Rc::from(name), ret_ct);
            }
        }

        // Detect struct/complex/vector returns that need special ABI handling
        let mut sret_size = None;
        let mut two_reg_ret_size = None;
        if ptr_count == 0 {
            let ret_ct = self.type_spec_to_ctype(ret_type_spec);
            if ret_ct.is_struct_or_union() {
                let size = self.sizeof_type(ret_type_spec);
                let (s, t) = Self::classify_struct_return(size);
                sret_size = s;
                two_reg_ret_size = t;
            }
            // Vector types use the same by-value return convention as structs
            if ret_ct.is_vector() {
                let size = self.sizeof_type(ret_type_spec);
                let (s, t) = Self::classify_struct_return(size);
                sret_size = s;
                two_reg_ret_size = t;
                // Small vector returns (<=8 bytes, no sret/two_reg) need I64 return type
                if s.is_none() && t.is_none() {
                    ret_ty = IrType::I64;
                }
            }
            if matches!(ret_ct, CType::ComplexLongDouble) {
                // On x86-64, _Complex long double returns via st(0)/st(1), no sret needed.
                // On all other targets (including i686), use sret.
                if !self.returns_complex_long_double_in_regs() {
                    let size = self.sizeof_type(ret_type_spec);
                    sret_size = Some(size);
                }
            }
            // On i686, _Complex double (16 bytes) exceeds 8-byte reg pair, needs sret.
            if matches!(ret_ct, CType::ComplexDouble) && !self.decomposes_complex_double() {
                sret_size = Some(ret_ct.size());
            }
        }

        // Compute SysV eightbyte classification for two-register struct returns
        let ret_eightbyte_classes = if two_reg_ret_size.is_some() && ptr_count == 0 {
            let ret_ct = self.type_spec_to_ctype(ret_type_spec);
            if ret_ct.is_struct_or_union() || ret_ct.is_vector() {
                if let Some(layout) = self.get_struct_layout_for_type(ret_type_spec) {
                    layout.classify_sysv_eightbytes(&*self.types.borrow_struct_layouts())
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let param_tys: Vec<IrType> = params.iter().map(|p| {
            self.type_spec_to_ir(&p.type_spec)
        }).collect();
        let param_bool_flags: Vec<bool> = params.iter().map(|p| {
            self.is_type_bool(&p.type_spec)
        }).collect();
        let param_ctypes: Vec<CType> = params.iter().map(|p| {
            self.type_spec_to_ctype(&p.type_spec)
        }).collect();
        let decomposes_cld = self.decomposes_complex_long_double();
        let decomposes_cd = self.decomposes_complex_double();
        let decomposes_cf = self.decomposes_complex_float();
        let param_struct_sizes: Vec<Option<usize>> = params.iter().map(|p| {
            let ctype = self.type_spec_to_ctype(&p.type_spec);
            if self.is_type_struct_or_union(&p.type_spec) {
                Some(self.sizeof_type(&p.type_spec))
            } else if ctype.is_vector() {
                // Vector types are passed by value like structs
                Some(self.sizeof_type(&p.type_spec))
            } else if !decomposes_cld && matches!(ctype, CType::ComplexLongDouble) {
                Some(self.sizeof_type(&p.type_spec))
            } else if !decomposes_cd && matches!(ctype, CType::ComplexDouble) {
                Some(CType::ComplexDouble.size())
            } else if !decomposes_cf && matches!(ctype, CType::ComplexFloat) {
                Some(CType::ComplexFloat.size())
            } else {
                None
            }
        }).collect();

        // Compute per-eightbyte SysV ABI classification for struct params
        let param_struct_classes: Vec<Vec<crate::common::types::EightbyteClass>> = params.iter().enumerate().map(|(i, p)| {
            if param_struct_sizes.get(i).copied().flatten().is_some() {
                if let Some(layout) = self.get_struct_layout_for_type(&p.type_spec) {
                    layout.classify_sysv_eightbytes(&*self.types.borrow_struct_layouts())
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        }).collect();

        // Compute RISC-V LP64D float field classification for struct params
        let param_riscv_float_classes: Vec<Option<crate::common::types::RiscvFloatClass>> = params.iter().enumerate().map(|(i, p)| {
            if param_struct_sizes.get(i).copied().flatten().is_some() {
                if let Some(layout) = self.get_struct_layout_for_type(&p.type_spec) {
                    layout.classify_riscv_float_fields(&*self.types.borrow_struct_layouts())
                } else {
                    None
                }
            } else {
                None
            }
        }).collect();

        let sig = if !variadic || !param_tys.is_empty() {
            FuncSig {
                return_type: ret_ty,
                return_ctype,
                param_types: param_tys,
                param_ctypes,
                param_bool_flags,
                is_variadic: variadic,
                sret_size,
                two_reg_ret_size,
                ret_eightbyte_classes: ret_eightbyte_classes.clone(),
                param_struct_sizes,
                param_struct_classes,
                param_riscv_float_classes,
            }
        } else {
            FuncSig {
                return_type: ret_ty,
                return_ctype,
                param_types: Vec::new(),
                param_ctypes: Vec::new(),
                param_bool_flags: Vec::new(),
                is_variadic: variadic,
                sret_size,
                two_reg_ret_size,
                ret_eightbyte_classes,
                param_struct_sizes: Vec::new(),
                param_struct_classes: Vec::new(),
                param_riscv_float_classes: Vec::new(),
            }
        };
        // Don't overwrite an existing, more complete sig from the first pass
        // (e.g., a K&R definition) with a less complete block-scope declaration
        // that has empty/unspecified parameters like `int *f();`.
        if let Some(existing) = self.func_meta.sigs.get(name) {
            if sig.param_types.is_empty() && !existing.param_types.is_empty() {
                return;
            }
            if sig.param_struct_sizes.is_empty() && !existing.param_struct_sizes.is_empty() {
                return;
            }
        }
        self.func_meta.sigs.insert(Rc::from(name), sig);
    }

    /// Lower an `Initializer::Expr` for a local variable declaration.
    /// Handles char arrays from string literals, struct copy-init, complex init,
    /// and scalar init with implicit casts.
    pub(super) fn lower_local_init_expr(
        &mut self,
        expr: &Expr,
        alloca: Value,
        da: &DeclAnalysis,
        is_complex: bool,
        decl: &Declaration,
    ) {
        if da.is_array && (da.base_ty == IrType::I8 || da.base_ty == IrType::U8) {
            self.lower_char_array_init_expr(expr, alloca, da);
        } else if da.is_array && (da.base_ty == IrType::I32 || da.base_ty == IrType::U32) {
            self.lower_wchar_array_init_expr(expr, alloca, da);
        } else if da.is_array && (da.base_ty == IrType::I16 || da.base_ty == IrType::U16) {
            self.lower_char16_array_init_expr(expr, alloca, da);
        } else if da.is_struct {
            self.lower_struct_copy_init(expr, alloca, da);
        } else if is_complex {
            self.lower_complex_var_init(expr, alloca, da, decl);
        } else if let Some(ref ct) = da.c_type {
            if ct.is_vector() {
                // Vector init from expression: memcpy from source to destination.
                // When the source is a function call returning a small vector (<=8 bytes),
                // the return value is packed data in a register (I64), not a pointer.
                // We must spill it to an alloca before memcpy.
                let total_size = ct.size();
                let is_small_vec_call = self.rhs_is_small_vector_call(expr, total_size);
                let src = self.lower_expr(expr);
                let src_val = self.operand_to_value(src);
                if is_small_vec_call {
                    // Packed register value: store to alloca, then memcpy from alloca
                    let tmp_alloca = self.fresh_value();
                    let store_ty = Self::packed_store_type(total_size);
                    self.emit(Instruction::Alloca { dest: tmp_alloca, size: total_size, ty: store_ty, align: 0, volatile: false });
                    self.emit(Instruction::Store { val: Operand::Value(src_val), ptr: tmp_alloca, ty: store_ty, seg_override: AddressSpace::Default });
                    self.emit(Instruction::Memcpy { dest: alloca, src: tmp_alloca, size: total_size });
                } else {
                    self.emit(Instruction::Memcpy { dest: alloca, src: src_val, size: total_size });
                }
            } else {
                self.lower_scalar_init_expr(expr, alloca, da, decl);
            }
        } else {
            self.lower_scalar_init_expr(expr, alloca, da, decl);
        }
    }

    /// Char array from string literal: `char s[] = "hello"`
    fn lower_char_array_init_expr(&mut self, expr: &Expr, alloca: Value, da: &DeclAnalysis) {
        match expr {
            Expr::StringLiteral(s, _) | Expr::WideStringLiteral(s, _)
            | Expr::Char16StringLiteral(s, _) => {
                let arr_size = da.alloc_size;
                self.emit_string_to_alloca(alloca, s, 0, arr_size);
                // Zero-fill remaining bytes if string is shorter than array
                let str_len = s.chars().count() + 1; // +1 for null terminator
                for i in str_len..arr_size {
                    let val = Operand::Const(IrConst::I8(0));
                    self.emit_store_at_offset(alloca, i, val, IrType::I8);
                }
            }
            _ => {
                let val = self.lower_expr(expr);
                self.emit(Instruction::Store { val, ptr: alloca, ty: da.var_ty , seg_override: AddressSpace::Default });
            }
        }
    }

    /// wchar_t/int array from wide string: `wchar_t s[] = L"hello"`
    fn lower_wchar_array_init_expr(&mut self, expr: &Expr, alloca: Value, da: &DeclAnalysis) {
        match expr {
            Expr::WideStringLiteral(s, _) | Expr::StringLiteral(s, _)
            | Expr::Char16StringLiteral(s, _) => {
                self.emit_wide_string_to_alloca(alloca, s, 0);
            }
            _ => {
                let val = self.lower_expr(expr);
                self.emit(Instruction::Store { val, ptr: alloca, ty: da.var_ty , seg_override: AddressSpace::Default });
            }
        }
    }

    /// char16_t array from u"..." string literal: `char16_t s[] = u"hello"`
    fn lower_char16_array_init_expr(&mut self, expr: &Expr, alloca: Value, da: &DeclAnalysis) {
        match expr {
            Expr::Char16StringLiteral(s, _) | Expr::StringLiteral(s, _)
            | Expr::WideStringLiteral(s, _) => {
                self.emit_char16_string_to_alloca(alloca, s, 0);
                // Zero-fill remaining bytes if string is shorter than array
                let str_len = (s.chars().count() + 1) * 2; // +1 for null, *2 for u16
                let arr_size = da.alloc_size;
                if arr_size > str_len {
                    // Zero remaining bytes
                    for i in (str_len..arr_size).step_by(2) {
                        let val = Operand::Const(IrConst::I16(0));
                        self.emit_store_at_offset(alloca, i, val, IrType::U16);
                    }
                }
            }
            _ => {
                let val = self.lower_expr(expr);
                self.emit(Instruction::Store { val, ptr: alloca, ty: da.var_ty, seg_override: AddressSpace::Default });
            }
        }
    }

    /// Struct copy-initialization: `struct Point b = a;`
    fn lower_struct_copy_init(&mut self, expr: &Expr, alloca: Value, da: &DeclAnalysis) {
        // For expressions producing packed struct data (small struct
        // function call returns, ternaries over them, etc.), the value
        // IS the struct data, not an address. Store directly.
        if self.expr_produces_packed_struct_data(expr) && da.actual_alloc_size <= 8 {
            let val = self.lower_expr(expr);
            self.emit(Instruction::Store { val, ptr: alloca, ty: Self::packed_store_type(da.actual_alloc_size) , seg_override: AddressSpace::Default });
        } else {
            let src_addr = self.get_struct_base_addr(expr);
            self.emit(Instruction::Memcpy {
                dest: alloca,
                src: src_addr,
                size: da.actual_alloc_size,
            });
        }
    }

    /// Complex variable initialization: `_Complex double z = expr;`
    fn lower_complex_var_init(&mut self, expr: &Expr, alloca: Value, da: &DeclAnalysis, decl: &Declaration) {
        let complex_ctype = self.type_spec_to_ctype(&decl.type_spec);
        let src = self.lower_expr_to_complex(expr, &complex_ctype);
        self.emit(Instruction::Memcpy {
            dest: alloca,
            src,
            size: da.actual_alloc_size,
        });
    }

    /// Scalar variable initialization with implicit casts and const tracking.
    fn lower_scalar_init_expr(&mut self, expr: &Expr, alloca: Value, da: &DeclAnalysis, decl: &Declaration) {
        // Track const-qualified integer variable values for compile-time
        // array size evaluation (e.g., const int len = 5000; int arr[len];)
        if decl.is_const() && !da.is_pointer && !da.is_array && !da.is_struct {
            if let Some(const_val) = self.eval_const_expr(expr) {
                if let Some(_ival) = self.const_to_i64(&const_val) {
                    // Use declarator name from the declaration context
                    // (passed via da which carries the name context)
                    // Actually we need the name; we'll handle this at the call site
                }
            }
        }
        // Check if RHS is complex but LHS is non-complex:
        // extract real part first, then convert to target type.
        let rhs_ctype = self.expr_ctype(expr);
        let val = if rhs_ctype.is_complex() {
            let complex_val = self.lower_expr(expr);
            let ptr = self.operand_to_value(complex_val);
            if da.is_bool {
                // For _Bool targets, check both real and imag parts per C11 6.3.1.2
                self.lower_complex_to_bool(ptr, &rhs_ctype)
            } else {
                let real_part = self.load_complex_real(ptr, &rhs_ctype);
                let from_ty = Self::complex_component_ir_type(&rhs_ctype);
                self.emit_implicit_cast(real_part, from_ty, da.var_ty)
            }
        } else if da.is_bool {
            // For _Bool targets, normalize at the source type before any truncation.
            // Truncating first (e.g. 0x100 -> U8 = 0) then normalizing gives wrong results.
            let val = self.lower_expr(expr);
            let expr_ty = self.get_expr_type(expr);
            self.emit_bool_normalize_typed(val, expr_ty)
        } else {
            let val = self.lower_expr(expr);
            let expr_ty = self.get_expr_type(expr);
            self.emit_implicit_cast(val, expr_ty, da.var_ty)
        };
        self.emit(Instruction::Store { val, ptr: alloca, ty: da.var_ty , seg_override: AddressSpace::Default });
    }

    /// Lower `Initializer::List` for a local variable declaration.
    /// Dispatches to complex, struct, array, or scalar-with-braces handlers.
    pub(super) fn lower_local_init_list(
        &mut self,
        items: &[InitializerItem],
        alloca: Value,
        da: &DeclAnalysis,
        is_complex: bool,
        complex_elem_ctype: &Option<CType>,
        decl: &Declaration,
        declarator_name: &str,
    ) {
        if is_complex {
            self.lower_complex_init_list(items, alloca, decl);
        } else if da.is_struct {
            self.lower_struct_init_list(items, alloca, declarator_name);
        } else if da.is_array && da.elem_size > 0 {
            self.lower_array_init_list_dispatch(items, alloca, da, complex_elem_ctype, decl, declarator_name);
        } else if let Some(ref ct) = da.c_type {
            if let Some((elem_ct, num_elems)) = ct.vector_info() {
                self.lower_vector_init_list(items, alloca, elem_ct, num_elems);
            } else {
                // Scalar with braces: int x = { 1 };
                self.lower_scalar_braced_init(items, alloca, da);
            }
        } else {
            // Scalar with braces: int x = { 1 };
            self.lower_scalar_braced_init(items, alloca, da);
        }
    }

    /// Complex initializer list: `_Complex double z = {real, imag}`
    fn lower_complex_init_list(&mut self, items: &[InitializerItem], alloca: Value, decl: &Declaration) {
        let complex_ctype = self.type_spec_to_ctype(&decl.type_spec);
        let comp_ty = Self::complex_component_ir_type(&complex_ctype);
        // Store real part (first item)
        if let Some(item) = items.first() {
            if let Initializer::Expr(expr) = &item.init {
                let val = self.lower_expr(expr);
                let expr_ty = self.get_expr_type(expr);
                let val = self.emit_implicit_cast(val, expr_ty, comp_ty);
                self.emit(Instruction::Store { val, ptr: alloca, ty: comp_ty , seg_override: AddressSpace::Default });
            }
        } else {
            let zero = Self::complex_zero(comp_ty);
            self.emit(Instruction::Store { val: zero, ptr: alloca, ty: comp_ty , seg_override: AddressSpace::Default });
        }
        // Store imag part (second item) at offset
        let comp_size = Self::complex_component_size(&complex_ctype);
        let imag_ptr = self.emit_gep_offset(alloca, comp_size, IrType::I8);
        if let Some(item) = items.get(1) {
            if let Initializer::Expr(expr) = &item.init {
                let val = self.lower_expr(expr);
                let expr_ty = self.get_expr_type(expr);
                let val = self.emit_implicit_cast(val, expr_ty, comp_ty);
                self.emit(Instruction::Store { val, ptr: imag_ptr, ty: comp_ty , seg_override: AddressSpace::Default });
            }
        } else {
            let zero = Self::complex_zero(comp_ty);
            self.emit(Instruction::Store { val: zero, ptr: imag_ptr, ty: comp_ty , seg_override: AddressSpace::Default });
        }
    }

    /// Struct initializer list with designated initializer support.
    fn lower_struct_init_list(&mut self, items: &[InitializerItem], alloca: Value, declarator_name: &str) {
        if let Some(layout) = self.func_mut().locals.get(declarator_name).and_then(|l| l.struct_layout.clone()) {
            // Always zero-initialize the entire struct before writing explicit values.
            // The C standard (C11 6.7.9p21) requires that all members not explicitly
            // initialized in a brace-enclosed list are implicitly zero-initialized.
            // This handles partial array field init (e.g., struct { int arr[8]; } x = {0};)
            // where a single initializer item covers only one element of an array field.
            self.zero_init_alloca(alloca, layout.size);
            self.emit_struct_init(items, alloca, &layout, 0);
        }
    }

    /// Dispatch array initializer list handling based on array kind.
    fn lower_array_init_list_dispatch(
        &mut self,
        items: &[InitializerItem],
        alloca: Value,
        da: &DeclAnalysis,
        complex_elem_ctype: &Option<CType>,
        decl: &Declaration,
        declarator_name: &str,
    ) {
        // Handle brace-wrapped string literal for char arrays: char b[5] = {"def"}
        // The C standard (C11 6.7.9 p14) allows a string literal optionally enclosed
        // in braces as an initializer for an array of character type. Redirect to
        // lower_char_array_init_expr which uses the full array size (da.alloc_size)
        // rather than the per-element size (da.elem_size = 1).
        if (da.base_ty == IrType::I8 || da.base_ty == IrType::U8)
            && !da.is_array_of_pointers
            && items.len() == 1 && items[0].designators.is_empty()
        {
            if let Initializer::Expr(ref expr) = items[0].init {
                match expr {
                    Expr::StringLiteral(..) | Expr::WideStringLiteral(..) | Expr::Char16StringLiteral(..) => {
                        self.lower_char_array_init_expr(expr, alloca, da);
                        return;
                    }
                    _ => {}
                }
            }
        }
        // Also handle brace-wrapped wide string for wchar_t (I32/U32) arrays
        if (da.base_ty == IrType::I32 || da.base_ty == IrType::U32)
            && !da.is_array_of_pointers
            && items.len() == 1 && items[0].designators.is_empty()
        {
            if let Initializer::Expr(ref expr @ Expr::WideStringLiteral(..)) = items[0].init {
                self.lower_wchar_array_init_expr(expr, alloca, da);
                return;
            }
        }
        // Also handle brace-wrapped char16_t string for char16_t (I16/U16) arrays
        if (da.base_ty == IrType::I16 || da.base_ty == IrType::U16)
            && !da.is_array_of_pointers
            && items.len() == 1 && items[0].designators.is_empty()
        {
            if let Initializer::Expr(ref expr @ Expr::Char16StringLiteral(..)) = items[0].init {
                self.lower_char16_array_init_expr(expr, alloca, da);
                return;
            }
        }

        // For arrays of function pointers or pointer arrays, the struct layout
        // (from the return type or pointee type) must not trigger struct init path.
        let elem_struct_layout = if da.is_array_of_func_ptrs || da.is_array_of_pointers {
            None
        } else {
            self.func_mut().locals.get(declarator_name)
                .and_then(|l| l.struct_layout.clone())
        };

        if let Some(ref cplx_ctype) = complex_elem_ctype {
            // Array of complex elements (handles both 1D and multi-dimensional)
            self.lower_array_of_complex_init(items, alloca, da, cplx_ctype);
        } else if da.is_array_of_pointers || da.is_array_of_func_ptrs {
            // Array of pointers (including pointer-to-struct): use scalar init with pointer stride.
            // Note: elem_struct_layout may be set for p->field access, but the array stride
            // is sizeof(pointer) not sizeof(struct), so we must NOT route to lower_array_of_structs_init.
            if da.array_dim_strides.len() > 1 {
                self.zero_init_alloca(alloca, da.alloc_size);
                self.lower_array_init_list(items, alloca, IrType::Ptr, &da.array_dim_strides);
            } else {
                self.lower_1d_array_init(items, alloca, da, decl);
            }
        } else if da.array_dim_strides.len() > 1 && elem_struct_layout.is_none() {
            // Multi-dimensional array of scalars
            self.zero_init_alloca(alloca, da.alloc_size);
            let md_elem_ty = da.elem_ir_ty;
            self.lower_array_init_list(items, alloca, md_elem_ty, &da.array_dim_strides);
        } else if let Some(ref s_layout) = elem_struct_layout {
            // Array of structs
            self.lower_array_of_structs_init(items, alloca, da, s_layout);
        } else {
            // 1D array of scalars
            self.lower_1d_array_init(items, alloca, da, decl);
        }
    }

    /// Array of structs initialization with designator support.
    /// Handles arbitrary dimensionality (1D, 2D, 3D, etc.) via recursive
    /// stride peeling through `lower_struct_array_init_recursive`.
    fn lower_array_of_structs_init(
        &mut self,
        items: &[InitializerItem],
        alloca: Value,
        da: &DeclAnalysis,
        s_layout: &crate::common::types::StructLayout,
    ) {
        let struct_size = s_layout.size;
        if struct_size == 0 {
            return;
        }
        self.zero_init_alloca(alloca, da.alloc_size);
        let strides = da.array_dim_strides.clone();
        let s_layout_cloned = s_layout.clone();
        let mut flat_idx = 0usize;
        self.lower_struct_array_init_recursive(items, alloca, &strides, struct_size, &s_layout_cloned, &mut flat_idx);
    }

    /// Recursively initialize a (possibly multi-dimensional) array of structs.
    ///
    /// `dim_strides` contains the byte strides for each remaining dimension.
    /// For `struct t a[2][2][2]` with struct_size=8, strides are [32, 16, 8].
    /// At each level, if stride > struct_size, the brace-delimited `List` items
    /// represent sub-arrays that must be recursed into. When stride == struct_size,
    /// we've reached leaf elements and each `List` is a single struct initializer.
    fn lower_struct_array_init_recursive(
        &mut self,
        items: &[InitializerItem],
        alloca: Value,
        dim_strides: &[usize],
        struct_size: usize,
        s_layout: &crate::common::types::StructLayout,
        flat_idx: &mut usize,
    ) {
        let (this_stride, remaining) = if dim_strides.len() > 1 {
            (dim_strides[0], &dim_strides[1..])
        } else if dim_strides.len() == 1 {
            (dim_strides[0], &dim_strides[0..0])
        } else {
            (struct_size, &dim_strides[0..0])
        };

        // How many leaf structs fit in one element at this dimension level
        let elems_per_slot = if struct_size > 0 { this_stride / struct_size } else { 1 };
        let is_subarray = this_stride > struct_size && !remaining.is_empty();

        let mut item_idx = 0usize;
        while item_idx < items.len() {
            let item = &items[item_idx];

            // Handle designators for index positioning
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx_val) = self.eval_const_expr_for_designator(idx_expr) {
                    *flat_idx = idx_val * elems_per_slot;
                }
            }

            let base_byte_offset = *flat_idx * struct_size;
            let field_designator_name = item.designators.iter().find_map(|d| {
                if let Designator::Field(ref name) = d {
                    Some(name.clone())
                } else {
                    None
                }
            });

            match &item.init {
                Initializer::List(sub_items) => {
                    if let Some(ref fname) = field_designator_name {
                        // Field-designated list init: [idx].field = { ... }
                        self.lower_struct_array_field_designated_list(
                            sub_items, alloca, base_byte_offset, fname, s_layout,
                        );
                    } else if is_subarray {
                        // Sub-array: recurse with next dimension level
                        let start_flat = *flat_idx;
                        self.lower_struct_array_init_recursive(
                            sub_items, alloca, remaining, struct_size, s_layout, flat_idx,
                        );
                        // After recursion, advance flat_idx to the next sub-array boundary
                        // to handle partial initialization (C11 6.7.9p21: uninitialized
                        // elements are zero-initialized, already done by zero_init_alloca).
                        let boundary = start_flat + elems_per_slot;
                        if *flat_idx < boundary {
                            *flat_idx = boundary;
                        }
                        item_idx += 1;
                        continue; // flat_idx already advanced
                    } else {
                        // Leaf: this List is a single struct initializer
                        let elem_base = self.emit_gep_offset(alloca, base_byte_offset, IrType::I8);
                        self.lower_local_struct_init(sub_items, elem_base, s_layout);
                    }
                }
                Initializer::Expr(e) => {
                    if let Some(ref fname) = field_designator_name {
                        self.lower_struct_array_field_designated_expr(
                            e, alloca, base_byte_offset, fname, s_layout,
                        );
                    } else if self.struct_value_size(e).is_some() {
                        let src_addr = self.get_struct_base_addr(e);
                        self.emit_memcpy_at_offset(alloca, base_byte_offset, src_addr, struct_size);
                    } else {
                        // Flat scalar init without braces (e.g., `struct t a[2] = {1,2,3,4}`)
                        let consumed = self.emit_struct_init(&items[item_idx..], alloca, s_layout, base_byte_offset);
                        item_idx += consumed.max(1);
                        *flat_idx += 1;
                        continue;
                    }
                }
            }
            item_idx += 1;
            *flat_idx += 1;
        }
    }

    /// Handle field-designated list init for a struct array element:
    /// `[idx].field = { ... }`
    fn lower_struct_array_field_designated_list(
        &mut self,
        sub_items: &[InitializerItem],
        alloca: Value,
        base_byte_offset: usize,
        fname: &str,
        s_layout: &crate::common::types::StructLayout,
    ) {
        if let Some(field) = s_layout.fields.iter().find(|f| &*f.name == fname) {
            let field_offset = base_byte_offset + field.offset;
            if field.ty.is_complex() {
                let dest_addr = self.emit_gep_offset(alloca, field_offset, IrType::Ptr);
                self.lower_complex_list_init(sub_items, dest_addr, &field.ty);
            } else if let CType::Array(ref elem_ty, Some(arr_size)) = field.ty {
                let elem_size = self.resolve_ctype_size(elem_ty);
                if elem_ty.is_complex() {
                    let complex_ctype = elem_ty.as_ref().clone();
                    for (ai, sub_item) in sub_items.iter().enumerate() {
                        if ai >= arr_size { break; }
                        let elem_offset = field_offset + ai * elem_size;
                        match &sub_item.init {
                            Initializer::Expr(e) => {
                                self.emit_complex_expr_to_offset(e, alloca, elem_offset, &complex_ctype);
                            }
                            Initializer::List(inner_items) => {
                                let dest = self.emit_gep_offset(alloca, elem_offset, IrType::Ptr);
                                self.lower_complex_list_init(inner_items, dest, &complex_ctype);
                            }
                        }
                    }
                } else {
                    let elem_ir_ty = IrType::from_ctype(elem_ty);
                    let elem_is_bool = **elem_ty == CType::Bool;
                    for (ai, sub_item) in sub_items.iter().enumerate() {
                        if ai >= arr_size { break; }
                        if let Initializer::Expr(e) = &sub_item.init {
                            let elem_offset = field_offset + ai * elem_size;
                            self.emit_init_expr_to_offset_bool(e, alloca, elem_offset, elem_ir_ty, elem_is_bool);
                        }
                    }
                }
            } else if let CType::Struct(ref key) | CType::Union(ref key) = field.ty {
                let sub_layout = self.types.borrow_struct_layouts().get(&**key).cloned();
                if let Some(sub_layout) = sub_layout {
                    let dest = self.emit_gep_offset(alloca, field_offset, IrType::I8);
                    self.lower_local_struct_init(sub_items, dest, &sub_layout);
                }
            } else if let Some(first) = sub_items.first() {
                if let Initializer::Expr(e) = &first.init {
                    let field_ir_ty = IrType::from_ctype(&field.ty);
                    let val = self.lower_and_cast_init_expr(e, field_ir_ty);
                    self.emit_store_at_offset(alloca, field_offset, val, field_ir_ty);
                }
            }
        }
    }

    /// Handle field-designated expression init for a struct array element:
    /// `[idx].field = expr`
    fn lower_struct_array_field_designated_expr(
        &mut self,
        e: &Expr,
        alloca: Value,
        base_byte_offset: usize,
        fname: &str,
        s_layout: &crate::common::types::StructLayout,
    ) {
        if let Some(field) = s_layout.fields.iter().find(|f| &*f.name == fname) {
            if field.ty.is_complex() {
                let field_offset = base_byte_offset + field.offset;
                self.emit_complex_expr_to_offset(e, alloca, field_offset, &field.ty);
            } else {
                let field_ty = IrType::from_ctype(&field.ty);
                let val = if field.ty == CType::Bool {
                    let v = self.lower_expr(e);
                    let et = self.get_expr_type(e);
                    self.emit_bool_normalize_typed(v, et)
                } else {
                    self.lower_and_cast_init_expr(e, field_ty)
                };
                self.emit_store_at_offset(alloca, base_byte_offset + field.offset, val, field_ty);
            }
        }
    }

    /// Array of complex elements initialization.
    /// Handles both 1D and multi-dimensional arrays of complex types.
    fn lower_array_of_complex_init(
        &mut self,
        items: &[InitializerItem],
        alloca: Value,
        da: &DeclAnalysis,
        cplx_ctype: &CType,
    ) {
        self.zero_init_alloca(alloca, da.alloc_size);
        let mut flat_idx = 0usize;
        self.lower_complex_init_recursive(items, alloca, da, cplx_ctype, &da.array_dim_strides.clone(), &mut flat_idx);
    }

    /// Recursive helper for multi-dimensional complex array initialization.
    /// Flattens nested brace initializers into flat element indices.
    fn lower_complex_init_recursive(
        &mut self,
        items: &[InitializerItem],
        alloca: Value,
        da: &DeclAnalysis,
        cplx_ctype: &CType,
        dim_strides: &[usize],
        flat_idx: &mut usize,
    ) {
        // The leaf complex element size is always the last stride
        let leaf_size = *da.array_dim_strides.last().unwrap_or(&da.elem_size);
        // How many leaf elements per sub-array at this level
        let sub_elem_count = if dim_strides.len() > 1 && leaf_size > 0 {
            dim_strides[0] / leaf_size
        } else {
            1
        };

        for item in items.iter() {
            let start_index = *flat_idx;

            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx_val) = self.eval_const_expr_for_designator(idx_expr) {
                    if dim_strides.len() > 1 {
                        // Designator at outer dimension: set flat index to start of that row
                        *flat_idx = idx_val * sub_elem_count;
                    } else {
                        *flat_idx = idx_val;
                    }
                }
            }

            match &item.init {
                Initializer::List(sub_items) => {
                    if dim_strides.len() > 1 {
                        // Recurse into nested brace list for inner dimensions
                        self.lower_complex_init_recursive(sub_items, alloca, da, cplx_ctype, &dim_strides[1..], flat_idx);
                        // Advance to next sub-array boundary
                        let boundary = start_index + sub_elem_count;
                        if *flat_idx < boundary {
                            *flat_idx = boundary;
                        }
                    } else {
                        // At leaf dimension, unwrap single-element brace list
                        if let Some(e) = Self::unwrap_nested_init_expr(sub_items) {
                            let src = self.lower_expr_to_complex(e, cplx_ctype);
                            self.emit_memcpy_at_offset(alloca, *flat_idx * leaf_size, src, leaf_size);
                        }
                        *flat_idx += 1;
                    }
                }
                Initializer::Expr(e) => {
                    let src = self.lower_expr_to_complex(e, cplx_ctype);
                    self.emit_memcpy_at_offset(alloca, *flat_idx * leaf_size, src, leaf_size);
                    *flat_idx += 1;
                }
            }
        }
    }

    /// 1D array initialization with designator and string literal support.
    fn lower_1d_array_init(
        &mut self,
        items: &[InitializerItem],
        alloca: Value,
        da: &DeclAnalysis,
        decl: &Declaration,
    ) {
        let num_elems = da.alloc_size / da.elem_size.max(1);
        let has_designators = items.iter().any(|item| !item.designators.is_empty());
        if has_designators || items.len() < num_elems {
            self.zero_init_alloca(alloca, da.alloc_size);
        }

        let is_complex_elem_array = self.is_type_complex(&decl.type_spec);
        let is_bool_elem_array = self.is_type_bool(&decl.type_spec)
            && !da.is_array_of_pointers && !da.is_array_of_func_ptrs;

        let elem_store_ty = if da.is_array_of_pointers || da.is_array_of_func_ptrs { IrType::Ptr } else { da.elem_ir_ty };

        let mut current_idx = 0usize;
        for item in items.iter() {
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx_val) = self.eval_const_expr_for_designator(idx_expr) {
                    current_idx = idx_val;
                }
            }

            let init_expr = match &item.init {
                Initializer::Expr(e) => Some(e),
                Initializer::List(sub_items) => {
                    Self::unwrap_nested_init_expr(sub_items)
                }
            };
            if let Some(e) = init_expr {
                if !da.is_array_of_pointers && (da.elem_ir_ty == IrType::I8 || da.elem_ir_ty == IrType::U8) {
                    if let Expr::StringLiteral(s, _) = e {
                        self.emit_string_to_alloca(alloca, s, current_idx * da.elem_size, da.elem_size);
                        current_idx += 1;
                        continue;
                    }
                }
                // wchar_t array with braced wide string: wchar_t w[] = {L"hello"};
                if !da.is_array_of_pointers && (da.elem_ir_ty == IrType::I32 || da.elem_ir_ty == IrType::U32) {
                    if let Expr::WideStringLiteral(s, _) = e {
                        self.emit_wide_string_to_alloca(alloca, s, current_idx * da.elem_size);
                        current_idx += 1;
                        continue;
                    }
                }
                // char16_t array with braced char16 string: char16_t w[] = {u"hello"};
                if !da.is_array_of_pointers && (da.elem_ir_ty == IrType::I16 || da.elem_ir_ty == IrType::U16) {
                    if let Expr::Char16StringLiteral(s, _) = e {
                        self.emit_char16_string_to_alloca(alloca, s, current_idx * da.elem_size);
                        current_idx += 1;
                        continue;
                    }
                }
                if is_complex_elem_array {
                    let val = self.lower_expr(e);
                    let src = self.operand_to_value(val);
                    self.emit_memcpy_at_offset(alloca, current_idx * da.elem_size, src, da.elem_size);
                } else if is_bool_elem_array {
                    // _Bool array elements: normalize (any nonzero -> 1) per C11 6.3.1.2
                    let val = self.lower_expr(e);
                    let expr_ty = self.get_expr_type(e);
                    let val = self.emit_bool_normalize_typed(val, expr_ty);
                    self.emit_array_element_store(alloca, val, current_idx * da.elem_size, elem_store_ty);
                } else {
                    let val = self.lower_expr(e);
                    let expr_ty = self.get_expr_type(e);
                    let val = self.emit_implicit_cast(val, expr_ty, elem_store_ty);
                    self.emit_array_element_store(alloca, val, current_idx * da.elem_size, elem_store_ty);
                }
            }
            current_idx += 1;
        }
    }

    /// Vector init list: `v4hi a = {1, 2, 3, 4};`
    /// Stores each element at elem_size * index offset from the alloca.
    pub(super) fn lower_vector_init_list(&mut self, items: &[InitializerItem], alloca: Value, elem_ct: &CType, num_elems: usize) {
        let elem_ir_ty = IrType::from_ctype(elem_ct);
        let elem_size = elem_ct.size();
        // Zero-init first if fewer initializers than elements
        if items.len() < num_elems {
            self.zero_init_alloca(alloca, elem_size * num_elems);
        }
        for (idx, item) in items.iter().enumerate() {
            if idx >= num_elems { break; }
            let init_expr = match &item.init {
                Initializer::Expr(e) => Some(e),
                Initializer::List(sub_items) => Self::unwrap_nested_init_expr(sub_items),
            };
            if let Some(e) = init_expr {
                let val = self.lower_expr(e);
                let expr_ty = self.get_expr_type(e);
                let val = self.emit_implicit_cast(val, expr_ty, elem_ir_ty);
                let offset = idx * elem_size;
                self.emit_array_element_store(alloca, val, offset, elem_ir_ty);
            }
        }
    }

    /// Scalar with braces: `int x = { 1 };` or `int x = {{{1}}};`
    fn lower_scalar_braced_init(&mut self, items: &[InitializerItem], alloca: Value, da: &DeclAnalysis) {
        if let Some(expr) = Self::unwrap_nested_init_expr(items) {
            let val = self.lower_expr(expr);
            let expr_ty = self.get_expr_type(expr);
            // For _Bool, normalize before any truncation to preserve high bits
            let val = if da.is_bool {
                self.emit_bool_normalize_typed(val, expr_ty)
            } else {
                self.emit_implicit_cast(val, expr_ty, da.var_ty)
            };
            self.emit(Instruction::Store { val, ptr: alloca, ty: da.var_ty , seg_override: AddressSpace::Default });
        }
    }
}
