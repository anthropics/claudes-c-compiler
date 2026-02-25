//! Global address resolution for compile-time constant evaluation.
//!
//! This module resolves expressions to global addresses (GlobalInit::GlobalAddr
//! or GlobalInit::GlobalAddrOffset) for use in global variable initializers.
//! Handles patterns including:
//! - `&x` (address of a global variable)
//! - `func` (function name as pointer)
//! - `arr` (array name decay to pointer)
//! - `arr[i]` (multidimensional array subscript with sub-array decay to pointer)
//! - `&arr[i][j]` (chained array subscripts with constant indices)
//! - `&s.a.b.c` (chained struct member access)
//! - `&((type*)0)->member` patterns (resolved via offsetof in const_eval.rs)
//! - Pointer arithmetic on global addresses (`&x + n`, `arr - n`)

use std::rc::Rc;
use crate::frontend::parser::ast::{
    BinOp,
    Expr,
    Initializer,
    TypeSpecifier,
    UnaryOp,
};
use crate::ir::reexports::{GlobalInit, IrConst};
use crate::common::types::{CType, StructLayout};
use super::lower::Lowerer;

impl Lowerer {
    /// Try to extract a global address from an initializer.
    /// Recurses into brace-wrapped lists to find the first pointer/address value.
    /// This handles compound literals like `((struct Wrap) {inc_global})` where
    /// the inner initializer contains a function pointer or global address.
    pub(super) fn eval_global_addr_from_initializer(&self, init: &Initializer) -> Option<GlobalInit> {
        match init {
            Initializer::Expr(expr) => self.eval_global_addr_expr(expr),
            Initializer::List(items) => {
                if let Some(first) = items.first() {
                    self.eval_global_addr_from_initializer(&first.init)
                } else {
                    None
                }
            }
        }
    }

    /// Resolve a variable name to its global name, checking static local names first.
    fn resolve_to_global_name(&self, name: &str) -> Option<Rc<str>> {
        if let Some(ref fs) = self.func_state {
            if let Some(mangled) = fs.static_local_names.get(name) {
                return Some(mangled.clone());
            }
        }
        if self.globals.contains_key(name) {
            Some(Rc::from(name))
        } else {
            None
        }
    }

    /// Try to evaluate an expression as a global address constant.
    /// This handles patterns like:
    /// - `&x` (address of a global variable)
    /// - `func` (function name used as pointer)
    /// - `arr` (array name decays to pointer)
    /// - `arr[i]` (multidimensional array subscript, sub-array decays to pointer)
    /// - `&arr[3]` (address of array element with constant index)
    /// - `&s.field` (address of struct field)
    /// - `(type *)&x` (cast of address expression)
    /// - `&x + n` or `&x - n` (pointer arithmetic on global address)
    pub(super) fn eval_global_addr_expr(&self, expr: &Expr) -> Option<GlobalInit> {
        match expr {
            // &x -> GlobalAddr("x")
            Expr::AddressOf(inner, _) => {
                match inner.as_ref() {
                    Expr::Identifier(name, _) => {
                        // Check static local names first (local statics shadow globals)
                        if let Some(ref fs) = self.func_state {
                            if let Some(mangled) = fs.static_local_names.get(name) {
                                return Some(GlobalInit::GlobalAddr(mangled.clone()));
                            }
                        }
                        // Address of a global variable or function
                        if self.globals.contains_key(name) || self.known_functions.contains(name) {
                            // Apply __asm__("label") redirect (e.g. stat -> stat64)
                            let resolved = self.asm_label_map.get(&**name)
                                .cloned()
                                .unwrap_or_else(|| name.clone());
                            return Some(GlobalInit::GlobalAddr(resolved));
                        }
                        None
                    }
                    // &arr[i] or &arr[i][j][k] -> GlobalAddrOffset("arr", total_offset)
                    Expr::ArraySubscript(_, _, _) => {
                        self.resolve_chained_array_subscript(inner)
                    }
                    // &s.field or &s.a.b.c or &arr[i].field -> GlobalAddrOffset
                    Expr::MemberAccess(_, _, _) => {
                        self.resolve_chained_member_access(inner)
                    }
                    // &(base->field) where base is pointer arithmetic on a global array
                    // e.g., &((Upgrade_items + 1)->uaattrid)
                    Expr::PointerMemberAccess(base, field, _) => {
                        self.resolve_pointer_member_access_addr(base, field)
                    }
                    // &(compound_literal) - look up previously materialized compound literal
                    Expr::CompoundLiteral(_, _, _) => {
                        let key = inner.as_ref() as *const Expr as usize;
                        self.materialized_compound_literals.get(&key)
                            .map(|label| GlobalInit::GlobalAddr(Rc::from(label.as_str())))
                    }
                    _ => None,
                }
            }
            // Function name as pointer: void (*fp)(void) = func;
            Expr::Identifier(name, _) => {
                if self.known_functions.contains(name) {
                    // Apply __asm__("label") linker symbol redirect if present.
                    // E.g., `stat` declared with __asm__("stat64") should emit "stat64".
                    // Without this, glibc's __REDIRECT mechanism (used for LFS stat/fstat
                    // when _FILE_OFFSET_BITS=64) would store the non-redirected symbol
                    // in global initializers like sqlite's aSyscall[] table.
                    let resolved = self.asm_label_map.get(&**name)
                        .cloned()
                        .unwrap_or_else(|| name.clone());
                    return Some(GlobalInit::GlobalAddr(resolved));
                }
                // Check static local array names first (they shadow globals)
                if let Some(ref fs) = self.func_state {
                    if let Some(mangled) = fs.static_local_names.get(name) {
                        if let Some(ginfo) = self.globals.get(mangled) {
                            if ginfo.is_array {
                                return Some(GlobalInit::GlobalAddr(mangled.clone()));
                            }
                        }
                    }
                }
                // Array name decays to pointer: int *p = arr;
                if let Some(ginfo) = self.globals.get(name) {
                    if ginfo.is_array {
                        return Some(GlobalInit::GlobalAddr(name.clone()));
                    }
                }
                None
            }
            // Array subscript on a multidimensional global array where the result
            // is a sub-array that decays to a pointer. For example:
            //   extern const u8 regs[10][16];
            //   .ptr = regs[3]   // regs[3] is u8[16], decays to u8*
            // This resolves to GlobalAddrOffset("regs", 3*16).
            // Critical for Linux kernel's clk_alpha_pll_regs patterns.
            Expr::ArraySubscript(_, _, _) => {
                self.resolve_array_subscript_decay(expr)
            }
            // Member access on a global struct where the field is an array:
            // e.g., `global_struct.array_field` decays to &global_struct.array_field[0]
            // This handles patterns like `init_files.open_fds_init` in the Linux kernel
            // where open_fds_init is an `unsigned long[1]` field.
            Expr::MemberAccess(_, _, _) => {
                self.resolve_member_access_array_decay(expr)
            }
            // (type *)expr -> try evaluating the inner expression
            Expr::Cast(_, inner, _) => {
                self.eval_global_addr_expr(inner)
            }
            // Compound literal -> check if it was materialized as an anonymous global
            // by materialize_compound_literals_in_expr(), then fall back to trying
            // to extract a global address from the initializer content.
            // The materialized path handles patterns like:
            //   (char*)(int[]){1,2,3} + 8
            // where the compound literal is the base of pointer arithmetic.
            // The initializer path handles patterns like:
            //   ((struct Wrap) {func_ptr})
            // where the compound literal wraps a function pointer.
            Expr::CompoundLiteral(_, ref init, _) => {
                // Check if this compound literal was pre-materialized as an anonymous global
                let key = expr as *const Expr as usize;
                if let Some(label) = self.materialized_compound_literals.get(&key) {
                    return Some(GlobalInit::GlobalAddr(Rc::from(label.as_str())));
                }
                self.eval_global_addr_from_initializer(init)
            }
            // &x + n or arr + n -> GlobalAddrOffset with byte offset
            Expr::BinaryOp(BinOp::Add, lhs, rhs, _) => {
                // Try lhs as address, rhs as constant offset
                if let Some(addr) = self.eval_global_addr_base_and_offset(lhs, rhs) {
                    return Some(addr);
                }
                // Try rhs as address, lhs as constant offset (commutative)
                if let Some(addr) = self.eval_global_addr_base_and_offset(rhs, lhs) {
                    return Some(addr);
                }
                None
            }
            // &x - n -> GlobalAddrOffset with negative byte offset
            Expr::BinaryOp(BinOp::Sub, lhs, rhs, _) => {
                if let Some(base_init) = self.eval_global_addr_expr(lhs) {
                    if let Some(offset_val) = self.eval_const_expr(rhs) {
                        if let Some(offset) = self.const_to_i64(&offset_val) {
                            // If the expression was cast to an integer type
                            // (e.g., (uintptr_t)ptr - 1), use byte-level arithmetic.
                            let elem_size = if self.expr_is_pointer(lhs) {
                                self.get_pointer_elem_size_from_expr(lhs)
                            } else {
                                1
                            };
                            let byte_offset = -(offset * elem_size as i64);
                            match base_init {
                                GlobalInit::GlobalAddr(name) => {
                                    if byte_offset == 0 {
                                        return Some(GlobalInit::GlobalAddr(name));
                                    }
                                    return Some(GlobalInit::GlobalAddrOffset(name, byte_offset));
                                }
                                GlobalInit::GlobalAddrOffset(name, base_off) => {
                                    let total = base_off + byte_offset;
                                    if total == 0 {
                                        return Some(GlobalInit::GlobalAddr(name));
                                    }
                                    return Some(GlobalInit::GlobalAddrOffset(name, total));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                None
            }
            // Conditional (ternary): cond ? then_expr : else_expr
            // Evaluate the condition as a scalar constant, then resolve the
            // selected branch as a global address. This is critical for the
            // Linux kernel's _OF_DECLARE macro pattern:
            //   .data = (fn == (fn_type)NULL) ? fn : fn
            // Without this, the ternary falls through to GlobalInit::Zero and
            // the function pointer is lost, causing NULL dereference at boot.
            Expr::Conditional(cond, then_e, else_e, _) => {
                // Try normal scalar constant evaluation of the condition first.
                // If that fails (e.g., condition involves a function pointer
                // comparison like `fn == NULL`), try address-aware evaluation.
                let cond_val = self.eval_const_expr(cond)
                    .or_else(|| self.eval_addr_comparison_cond(cond));
                let cond_val = cond_val?;
                let selected = if cond_val.is_nonzero() { then_e } else { else_e };
                // Try the selected branch as a global address first
                if let Some(addr) = self.eval_global_addr_expr(selected) {
                    return Some(addr);
                }
                // If the selected branch is a scalar constant (e.g., 0 for NULL),
                // return None and let the caller's eval_const_expr handle it.
                None
            }
            // _Generic(controlling, type1: expr1, ...) -> resolve to the matching
            // expression and evaluate it as a global address. This is critical for
            // QEMU's OUTOP macro which uses _Generic in designated initializers:
            //   [INDEX_op_st32] = _Generic(outop_st, TCGOutOpStore: &outop_st.base)
            Expr::GenericSelection(ref controlling, ref associations, _) => {
                let selected = self.resolve_generic_selection_expr(controlling, associations)?;
                self.eval_global_addr_expr(selected)
            }
            _ => None,
        }
    }

    /// Helper for pointer arithmetic: base_expr + offset_expr where base is an address
    fn eval_global_addr_base_and_offset(&self, base_expr: &Expr, offset_expr: &Expr) -> Option<GlobalInit> {
        let base_init = self.eval_global_addr_expr(base_expr)?;
        let offset_val = self.eval_const_expr(offset_expr)?;
        let offset = self.const_to_i64(&offset_val)?;
        // If the expression was cast to an integer type (e.g., (uintptr_t)ptr + 1),
        // arithmetic is byte-level (scale factor 1), not pointer-scaled.
        let elem_size = if self.expr_is_pointer(base_expr) {
            self.get_pointer_elem_size_from_expr(base_expr)
        } else {
            1
        };
        let byte_offset = offset * elem_size as i64;
        match base_init {
            GlobalInit::GlobalAddr(name) => {
                if byte_offset == 0 {
                    Some(GlobalInit::GlobalAddr(name))
                } else {
                    Some(GlobalInit::GlobalAddrOffset(name, byte_offset))
                }
            }
            GlobalInit::GlobalAddrOffset(name, base_off) => {
                let total = base_off + byte_offset;
                if total == 0 {
                    Some(GlobalInit::GlobalAddr(name))
                } else {
                    Some(GlobalInit::GlobalAddrOffset(name, total))
                }
            }
            _ => None,
        }
    }

    /// Resolve a chained array subscript expression like `arr[i][j][k]` to a global
    /// address with computed offset. Walks nested ArraySubscript nodes from outermost
    /// to innermost, collecting (index, stride) pairs to compute the total byte offset.
    /// Handles 1D (`&arr[i]`), 2D (`&arr[i][j]`), and higher-dimensional arrays.
    fn resolve_chained_array_subscript(&self, expr: &Expr) -> Option<GlobalInit> {
        // Collect subscripts from outer to inner: arr[i][j] has outer=[j] inner=[i]
        let mut subscripts: Vec<&Expr> = Vec::new();
        let mut current = expr;
        while let Expr::ArraySubscript(base, index, _) = current {
            subscripts.push(index);
            current = base.as_ref();
        }

        // Try to resolve the base. It can be:
        // 1. A plain Identifier (global array)
        // 2. A Cast of a global address expression, e.g.:
        //    (const char *)boot_cpu_data.x86_capability
        //    which reinterprets the member as a different pointer/array type.
        match current {
            Expr::Identifier(name, _) => {
                let global_name = self.resolve_to_global_name(name)?;
                let ginfo = self.globals.get(&global_name)?;
                if !ginfo.is_array {
                    return None;
                }
                subscripts.reverse();

                let mut total_offset: i64 = 0;
                let strides = &ginfo.array_dim_strides;
                for (dim, idx_expr) in subscripts.iter().enumerate() {
                    let idx_val = self.eval_const_expr(idx_expr)?;
                    let idx = self.const_to_i64(&idx_val)?;
                    let stride = if !strides.is_empty() && dim < strides.len() {
                        strides[dim] as i64
                    } else if dim == 0 {
                        ginfo.elem_size as i64
                    } else {
                        return None;
                    };
                    total_offset += idx * stride;
                }

                if total_offset == 0 {
                    Some(GlobalInit::GlobalAddr(global_name))
                } else {
                    Some(GlobalInit::GlobalAddrOffset(global_name, total_offset))
                }
            }
            Expr::Cast(type_spec, inner, _) => {
                // Handle patterns like ((const char *)boot_cpu_data.x86_capability)[N]
                // The inner expression should resolve to a global address, and the cast's
                // pointee type determines the element size for subscript strides.
                self.resolve_cast_array_subscript(type_spec, inner, &subscripts)
            }
            Expr::MemberAccess(_, _, _) => {
                // Handle patterns like global_struct.array_member[i]
                // e.g., rcu_state.node[0] where node is an array field inside a global struct.
                // Resolve the member access chain to get base global + field offset,
                // then apply array subscript offsets using the member's array element size.
                self.resolve_member_array_subscript(current, &subscripts)
            }
            _ => None,
        }
    }

    /// Resolve an array subscript expression that results in array-to-pointer decay.
    /// This handles `regs[i]` where `regs` is a multidimensional array, so `regs[i]`
    /// yields a sub-array that decays to a pointer. For example:
    ///   extern const u8 regs[10][16];
    ///   const u8 *p = regs[3];  // -> GlobalAddrOffset("regs", 48)
    /// We reuse resolve_chained_array_subscript for offset computation, then verify
    /// that the subscript count is less than the array dimension count (meaning the
    /// result is a sub-array, not a scalar element).
    fn resolve_array_subscript_decay(&self, expr: &Expr) -> Option<GlobalInit> {
        // Count subscripts and find the base
        let mut num_subscripts = 0usize;
        let mut current = expr;
        while let Expr::ArraySubscript(base, _, _) = current {
            num_subscripts += 1;
            current = base.as_ref();
        }
        // The base must be a global array with more dimensions than subscripts
        // (so the result is a sub-array that decays to a pointer)
        match current {
            Expr::Identifier(name, _) => {
                let global_name = self.resolve_to_global_name(name)?;
                let ginfo = self.globals.get(&global_name)?;
                if !ginfo.is_array {
                    return None;
                }
                // Check that this is a partial subscript (result is still an array)
                let num_dims = ginfo.array_dim_strides.len();
                if num_dims == 0 || num_subscripts >= num_dims {
                    return None;
                }
                // Use the existing chained subscript resolver for offset computation
                self.resolve_chained_array_subscript(expr)
            }
            // TODO: handle MemberAccess/Cast bases for patterns like
            // global_struct.field_2d_array[i] decay
            _ => None,
        }
    }

    /// Resolve an array subscript where the base is a member access on a global struct.
    /// For example: `rcu_state.node[0]` resolves to `rcu_state + offsetof(node) + 0 * sizeof(node[0])`.
    fn resolve_member_array_subscript(
        &self,
        member_expr: &Expr,
        subscripts: &[&Expr],
    ) -> Option<GlobalInit> {
        // Walk the member access chain to collect field names and find the base identifier
        let mut fields: Vec<Rc<str>> = Vec::new();
        let mut cur = member_expr;
        loop {
            match cur {
                Expr::MemberAccess(base, field, _) => {
                    fields.push(field.clone());
                    cur = base.as_ref();
                }
                Expr::Identifier(name, _) => {
                    let global_name = self.resolve_to_global_name(name)?;
                    let ginfo = self.globals.get(&global_name)?;
                    let start_layout = ginfo.struct_layout.clone()?;

                    // Walk field chain (in reverse, since we collected outer-to-inner)
                    // to compute the member offset and find the final field type
                    let mut member_offset: i64 = 0;
                    let mut current_layout = start_layout;
                    let mut final_field_ty: Option<CType> = None;
                    for field_name in fields.iter().rev() {
                        if let Some((foff, fty)) = current_layout.field_offset(field_name, &*self.types.borrow_struct_layouts()) {
                            member_offset += foff as i64;
                            final_field_ty = Some(fty.clone());
                            current_layout = match &fty {
                                CType::Struct(key) | CType::Union(key) => {
                                    self.types.borrow_struct_layouts().get(&**key).cloned()
                                        .unwrap_or_else(StructLayout::empty_rc)
                                }
                                _ => StructLayout::empty_rc(),
                            };
                        } else {
                            return None;
                        }
                    }

                    // The final field type should be an array for subscript access
                    let elem_size = match &final_field_ty {
                        Some(CType::Array(elem_ty, _)) => self.ctype_size(elem_ty) as i64,
                        _ => return None,
                    };
                    if elem_size == 0 {
                        return None;
                    }

                    // Apply subscript offsets (subscripts are in reverse order: outer first)
                    let mut total_offset = member_offset;
                    let mut subs = subscripts.to_vec();
                    subs.reverse();

                    // For the first dimension, use elem_size.
                    // For additional dimensions, we need to drill into the element type's
                    // array dimensions. Handle multi-dimensional arrays like node[i][j].
                    let mut current_elem_ty = match &final_field_ty {
                        Some(CType::Array(elem_ty, _)) => elem_ty.as_ref().clone(),
                        _ => return None,
                    };

                    for (dim, idx_expr) in subs.iter().enumerate() {
                        let idx_val = self.eval_const_expr(idx_expr)?;
                        let idx = self.const_to_i64(&idx_val)?;
                        let stride = if dim == 0 {
                            elem_size
                        } else {
                            // For deeper dimensions, get the size of the current element type
                            self.ctype_size(&current_elem_ty) as i64
                        };
                        total_offset += idx * stride;
                        // Drill into the next array dimension's element type
                        current_elem_ty = match &current_elem_ty {
                            CType::Array(inner, _) => inner.as_ref().clone(),
                            _ => current_elem_ty.clone(),
                        };
                    }

                    return if total_offset == 0 {
                        Some(GlobalInit::GlobalAddr(Rc::from(global_name)))
                    } else {
                        Some(GlobalInit::GlobalAddrOffset(Rc::from(global_name), total_offset))
                    };
                }
                _ => return None,
            }
        }
    }

    /// Resolve an array subscript where the base is a cast of a global address
    /// expression. For example: `((const char *)boot_cpu_data.x86_capability)[1]`
    /// resolves to `boot_cpu_data + offsetof(x86_capability) + 1 * sizeof(char)`.
    fn resolve_cast_array_subscript(
        &self,
        cast_type: &TypeSpecifier,
        inner_expr: &Expr,
        subscripts: &[&Expr],
    ) -> Option<GlobalInit> {
        // The cast type should be a pointer type; the pointee is the element type.
        let cast_ctype = self.type_spec_to_ctype(cast_type);
        let elem_size = match &cast_ctype {
            CType::Pointer(pointee, _) => self.ctype_size(pointee),
            // If it's an array type, use the element size
            CType::Array(elem, _) => self.ctype_size(elem),
            _ => return None,
        };
        if elem_size == 0 {
            return None;
        }

        // Resolve the inner expression as a global address.
        // This handles MemberAccess on globals (e.g., boot_cpu_data.x86_capability),
        // AddressOf patterns, identifiers, etc.
        let base_init = self.resolve_inner_as_global_addr(inner_expr)?;
        let (global_name_rc, base_offset) = match &base_init {
            GlobalInit::GlobalAddr(name) => (name.clone(), 0i64),
            GlobalInit::GlobalAddrOffset(name, off) => (name.clone(), *off),
            _ => return None,
        };

        // Compute the total subscript offset. Only a single subscript dimension
        // makes sense here since the cast reinterprets the base as a flat pointer.
        // subscripts are in outer-to-inner order; reverse for left-to-right.
        let mut total_offset = base_offset;
        for idx_expr in subscripts.iter().rev() {
            let idx_val = self.eval_const_expr(idx_expr)?;
            let idx = self.const_to_i64(&idx_val)?;
            total_offset += idx * elem_size as i64;
        }

        if total_offset == 0 {
            Some(GlobalInit::GlobalAddr(global_name_rc))
        } else {
            Some(GlobalInit::GlobalAddrOffset(global_name_rc, total_offset))
        }
    }

    /// Resolve an expression to a global address, handling member access on global
    /// structs (e.g., `boot_cpu_data.x86_capability` -> GlobalAddrOffset("boot_cpu_data", 8)).
    /// This is used as the base of cast+subscript patterns in inline asm operands.
    fn resolve_inner_as_global_addr(&self, expr: &Expr) -> Option<GlobalInit> {
        match expr {
            // Direct global identifier - treat as address of the global
            Expr::Identifier(name, _) => {
                let global_name = self.resolve_to_global_name(name)?;
                Some(GlobalInit::GlobalAddr(Rc::from(global_name)))
            }
            // struct_var.field -> global + field_offset
            Expr::MemberAccess(base, field, _) => {
                // Resolve the base to a global address
                let base_init = self.resolve_inner_as_global_addr(base)?;
                let (global_name_rc, base_off) = match &base_init {
                    GlobalInit::GlobalAddr(name) => (name.clone(), 0i64),
                    GlobalInit::GlobalAddrOffset(name, off) => (name.clone(), *off),
                    _ => return None,
                };
                // Look up the struct layout to get the field offset
                let ginfo = self.globals.get(&*global_name_rc)?;
                let layout = ginfo.struct_layout.clone()?;
                let (field_offset, _field_ty) = layout.field_offset(field, &*self.types.borrow_struct_layouts())?;
                let total = base_off + field_offset as i64;
                if total == 0 {
                    Some(GlobalInit::GlobalAddr(global_name_rc))
                } else {
                    Some(GlobalInit::GlobalAddrOffset(global_name_rc, total))
                }
            }
            // AddressOf(&x) -> address of x
            Expr::AddressOf(_inner, _) => {
                self.eval_global_addr_expr(expr)
            }
            // Cast preserves the address
            Expr::Cast(_, inner, _) => {
                self.resolve_inner_as_global_addr(inner)
            }
            _ => None,
        }
    }

    /// Resolve a member access expression where the field is an array type,
    /// implementing array-to-pointer decay for global struct fields.
    /// e.g., `global_struct.arr_field` -> GlobalAddrOffset("global_struct", field_offset)
    /// This is needed for patterns like `.open_fds = init_files.open_fds_init` in the
    /// Linux kernel, where `open_fds_init` is a `unsigned long[1]` array field.
    fn resolve_member_access_array_decay(&self, expr: &Expr) -> Option<GlobalInit> {
        // Use resolve_inner_as_global_addr to get the address of the field,
        // falling back to resolve_chained_member_access for complex chains
        // involving PointerMemberAccess (e.g., (&g.member)->field.array_field)
        let addr = self.resolve_inner_as_global_addr(expr)
            .or_else(|| self.resolve_chained_member_access(expr))?;
        // Now check if the field type is an array (needs decay)
        // Walk the member access chain to find the final field type
        let field_ty = self.get_member_access_field_type(expr)?;
        match field_ty {
            CType::Array(_, _) => Some(addr),
            // Also handle function types which decay to function pointers
            CType::Function(_) => Some(addr),
            _ => None,
        }
    }

    /// Get the C type of the result of a member access expression.
    /// Walks chains like `s.a.b` to return the type of the final field `b`.
    fn get_member_access_field_type(&self, expr: &Expr) -> Option<CType> {
        match expr {
            Expr::MemberAccess(base, field, _) => {
                // Get the struct layout of the base expression
                let layout = self.get_struct_layout_of_expr(base)?;
                let (_offset, field_ty) = layout.field_offset(field, &*self.types.borrow_struct_layouts())?;
                Some(field_ty)
            }
            _ => None,
        }
    }

    /// Get the struct layout for the type of an expression (for member access resolution).
    fn get_struct_layout_of_expr(&self, expr: &Expr) -> Option<StructLayout> {
        match expr {
            Expr::Identifier(name, _) => {
                let global_name = self.resolve_to_global_name(name)?;
                let ginfo = self.globals.get(&global_name)?;
                ginfo.struct_layout.as_ref().map(|rc| (**rc).clone())
            }
            Expr::MemberAccess(base, field, _) => {
                // For chained access s.a.b: get layout of s, find field a's type,
                // then get layout of a's type (must be struct/union)
                let base_layout = self.get_struct_layout_of_expr(base)?;
                let (_offset, field_ty) = base_layout.field_offset(field, &*self.types.borrow_struct_layouts())?;
                match &field_ty {
                    CType::Struct(key) | CType::Union(key) => {
                        self.types.borrow_struct_layouts().get(&**key).map(|rc| (**rc).clone())
                    }
                    _ => None,
                }
            }
            // arr[i] where arr is a global array of structs: return the element's struct layout.
            // This handles patterns like `arr[0].text` where we need the layout of the
            // struct element to resolve the `.text` field access.
            Expr::ArraySubscript(base, _index, _) => {
                let mut cur = base.as_ref();
                // Walk through nested subscripts to find the base identifier
                while let Expr::ArraySubscript(inner, _, _) = cur {
                    cur = inner.as_ref();
                }
                match cur {
                    Expr::Identifier(name, _) => {
                        let global_name = self.resolve_to_global_name(name)?;
                        let ginfo = self.globals.get(&global_name)?;
                        if ginfo.is_array {
                            // Return the struct layout of the array element type
                            ginfo.struct_layout.as_ref().map(|rc| (**rc).clone())
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            // (&struct_expr)->field: dereference is a no-op for &expr, resolve field type
            Expr::PointerMemberAccess(base, field, _) => {
                // The base should be &something; get the pointee's struct layout
                let pointee_layout = match base.as_ref() {
                    Expr::AddressOf(inner, _) => self.get_struct_layout_of_expr(inner)?,
                    _ => return None,
                };
                let (_offset, field_ty) = pointee_layout.field_offset(field, &*self.types.borrow_struct_layouts())?;
                match &field_ty {
                    CType::Struct(key) | CType::Union(key) => {
                        self.types.borrow_struct_layouts().get(&**key).map(|rc| (**rc).clone())
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Resolve a chained member access expression like `s.a.b.c` to a global address.
    /// Walks the chain from the root identifier, accumulating field offsets.
    /// Also handles `arr[i].field` and `(arr + i)->field` patterns.
    fn resolve_chained_member_access(&self, expr: &Expr) -> Option<GlobalInit> {
        // Collect the chain of field names from innermost to outermost
        let mut fields = Vec::new();
        let mut current = expr;
        loop {
            match current {
                Expr::MemberAccess(base, field, _) => {
                    fields.push(field.clone());
                    current = base.as_ref();
                }
                Expr::Identifier(name, _) => {
                    let global_name = self.resolve_to_global_name(name)?;
                    let ginfo = self.globals.get(&global_name)?;
                    let base_offset: i64 = 0;
                    let start_layout = ginfo.struct_layout.clone()?;
                    return self.apply_field_chain_offsets(&global_name, base_offset, &start_layout, &fields);
                }
                // Handle &arr[i].field - ArraySubscript as base of member chain
                // Also handles &global.member[i][j].field (member access + subscripts)
                Expr::ArraySubscript(_, _, _) => {
                    // Collect all subscript indices from outermost to innermost
                    let mut subscript_indices: Vec<i64> = Vec::new();
                    let mut sub_cur = current;
                    while let Expr::ArraySubscript(sub_base, sub_index, _) = sub_cur {
                        let idx_val = self.eval_const_expr(sub_index)?;
                        let idx = self.const_to_i64(&idx_val)?;
                        subscript_indices.push(idx);
                        sub_cur = sub_base.as_ref();
                    }
                    subscript_indices.reverse(); // inner-to-outer -> left-to-right

                    match sub_cur {
                        Expr::Identifier(name, _) => {
                            let global_name = self.resolve_to_global_name(name)?;
                            let ginfo = self.globals.get(&global_name)?;
                            if ginfo.is_array {
                                // Compute offset from subscripts using array dim strides
                                let mut base_offset: i64 = 0;
                                let strides = &ginfo.array_dim_strides;
                                for (dim, &idx) in subscript_indices.iter().enumerate() {
                                    let stride = if !strides.is_empty() && dim < strides.len() {
                                        strides[dim] as i64
                                    } else if dim == 0 {
                                        ginfo.elem_size as i64
                                    } else {
                                        return None;
                                    };
                                    base_offset += idx * stride;
                                }
                                let start_layout = ginfo.struct_layout.clone()?;
                                return self.apply_field_chain_offsets(&global_name, base_offset, &start_layout, &fields);
                            }
                        }
                        // Handle global.member[i][j].field pattern:
                        // The base of the subscripts is a MemberAccess chain on a global.
                        // Resolve the member chain to get the global + field offset + field type,
                        // then apply subscript offsets within the array field.
                        Expr::MemberAccess(_, _, _) => {
                            // Walk the member access chain below the subscripts (global.member[i][j].field)
                            let mut member_fields: Vec<Rc<str>> = Vec::new();
                            let mut mcur = sub_cur;
                            loop {
                                match mcur {
                                    Expr::MemberAccess(mbase, mfield, _) => {
                                        member_fields.push(mfield.clone());
                                        mcur = mbase.as_ref();
                                    }
                                    Expr::Identifier(name, _) => {
                                        let global_name = self.resolve_to_global_name(name)?;
                                        let ginfo = self.globals.get(&global_name)?;
                                        let start_layout = ginfo.struct_layout.clone()?;

                                        // Apply member field offsets to find the array field
                                        let mut member_offset: i64 = 0;
                                        let mut current_layout = start_layout;
                                        let mut final_field_ty: Option<CType> = None;
                                        for field_name in member_fields.iter().rev() {
                                            let (foff, fty) = current_layout.field_offset(field_name, &*self.types.borrow_struct_layouts())?;
                                            member_offset += foff as i64;
                                            final_field_ty = Some(fty.clone());
                                            current_layout = match &fty {
                                                CType::Struct(key) | CType::Union(key) => {
                                                    self.types.borrow_struct_layouts().get(&**key).cloned()
                                                        .unwrap_or_else(StructLayout::empty_rc)
                                                }
                                                _ => StructLayout::empty_rc(),
                                            };
                                        }

                                        // The field type should be an array for subscript access
                                        let mut arr_ty = final_field_ty?;
                                        let mut total_offset = member_offset;

                                        // Apply each subscript index
                                        for &idx in &subscript_indices {
                                            let elem_size = match &arr_ty {
                                                CType::Array(elem_ty, _) => {
                                                    let es = self.ctype_size(elem_ty) as i64;
                                                    // Advance to the element type for the next subscript
                                                    arr_ty = elem_ty.as_ref().clone();
                                                    es
                                                }
                                                _ => return None,
                                            };
                                            total_offset += idx * elem_size;
                                        }

                                        // Now arr_ty is the element type after all subscripts.
                                        // Get its struct layout for applying remaining field chain.
                                        let elem_layout = match &arr_ty {
                                            CType::Struct(key) | CType::Union(key) => {
                                                self.types.borrow_struct_layouts().get(&**key).cloned()?
                                            }
                                            _ => return None,
                                        };

                                        return self.apply_field_chain_offsets(&global_name, total_offset, &elem_layout, &fields);
                                    }
                                    _ => return None,
                                }
                            }
                        }
                        _ => {}
                    }
                    return None;
                }
                // Handle (&global)->field.subfield pattern:
                // PointerMemberAccess(AddressOf(Identifier("global")), "field")
                // This is equivalent to global.field so treat it as a member chain entry
                Expr::PointerMemberAccess(base, field, _) => {
                    fields.push(field.clone());
                    // The base must be &global_identifier or &global.member
                    match base.as_ref() {
                        Expr::AddressOf(inner, _) => {
                            match inner.as_ref() {
                                Expr::Identifier(name, _) => {
                                    let global_name = self.resolve_to_global_name(name)?;
                                    let ginfo = self.globals.get(&global_name)?;
                                    let base_offset: i64 = 0;
                                    let start_layout = ginfo.struct_layout.clone()?;
                                    return self.apply_field_chain_offsets(&global_name, base_offset, &start_layout, &fields);
                                }
                                // Handle &(g.member)->field pattern:
                                // e.g., (&(g._main_interpreter))->dtoa.preallocated
                                // Resolve inner member access to get the global + offset,
                                // then continue applying remaining fields from there.
                                Expr::MemberAccess(_, _, _) => {
                                    if let Some(init) = self.resolve_chained_member_access(inner) {
                                        let (global_name, base_off) = match &init {
                                            GlobalInit::GlobalAddr(name) => (name.clone(), 0i64),
                                            GlobalInit::GlobalAddrOffset(name, off) => (name.clone(), *off),
                                            _ => return None,
                                        };
                                        // Get the struct layout for the member access result type
                                        let member_ty = self.get_member_access_field_type(inner)?;
                                        let start_layout = match &member_ty {
                                            CType::Struct(key) | CType::Union(key) => {
                                                self.types.borrow_struct_layouts().get(&**key).cloned()?
                                            }
                                            _ => return None,
                                        };
                                        return self.apply_field_chain_offsets(&global_name, base_off, &start_layout, &fields);
                                    }
                                    return None;
                                }
                                _ => return None,
                            }
                        }
                        // Also handle identifier directly (e.g., ptr->field where ptr is a global pointer)
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
    }

    /// Apply a chain of field names to a base global+offset, accumulating field offsets.
    /// `fields` are in reverse order (outermost field last, innermost first).
    fn apply_field_chain_offsets(
        &self,
        global_name: &str,
        base_offset: i64,
        start_layout: &std::rc::Rc<StructLayout>,
        fields: &[Rc<str>],
    ) -> Option<GlobalInit> {
        let mut total_offset = base_offset;
        let mut current_layout = start_layout.clone();
        // fields are in reverse order (outermost field last)
        for field_name in fields.iter().rev() {
            let mut found = false;
            // Try field_offset which handles anonymous structs/unions
            if let Some((foff, fty)) = current_layout.field_offset(field_name, &*self.types.borrow_struct_layouts()) {
                total_offset += foff as i64;
                current_layout = match &fty {
                    CType::Struct(key) | CType::Union(key) => {
                        self.types.borrow_struct_layouts().get(&**key).cloned()
                            .unwrap_or_else(StructLayout::empty_rc)
                    }
                    _ => StructLayout::empty_rc(),
                };
                found = true;
            }
            if !found {
                return None;
            }
        }
        if total_offset == 0 {
            Some(GlobalInit::GlobalAddr(Rc::from(global_name)))
        } else {
            Some(GlobalInit::GlobalAddrOffset(Rc::from(global_name), total_offset))
        }
    }

    /// Resolve &(base->field) where base is a constant pointer expression
    /// involving a global array (e.g., &((Upgrade_items + 1)->uaattrid)).
    /// The base must resolve to a global address (possibly with offset).
    fn resolve_pointer_member_access_addr(&self, base: &Expr, field: &str) -> Option<GlobalInit> {
        // The base expression should be a pointer to a global (array element).
        // Try to evaluate it as a global address expression.
        let base_init = self.eval_global_addr_expr(base)?;
        let (global_name_rc, base_offset) = match &base_init {
            GlobalInit::GlobalAddr(name) => (name.clone(), 0i64),
            GlobalInit::GlobalAddrOffset(name, off) => (name.clone(), *off),
            _ => return None,
        };
        // Get the struct layout for the element type.
        // The global should be an array of structs.
        let ginfo = self.globals.get(&*global_name_rc)?;
        let layout = ginfo.struct_layout.clone()?;
        let (field_offset, _field_ty) = layout.field_offset(field, &*self.types.borrow_struct_layouts())?;
        let total_offset = base_offset + field_offset as i64;
        if total_offset == 0 {
            Some(GlobalInit::GlobalAddr(global_name_rc))
        } else {
            Some(GlobalInit::GlobalAddrOffset(global_name_rc, total_offset))
        }
    }

    /// Evaluate a comparison condition that involves a global address (function
    /// pointer or array) compared against a scalar constant (typically NULL/0).
    ///
    /// A defined function or global variable always has a non-zero address, so:
    /// - `func == 0` / `func == NULL` -> 0 (false)
    /// - `func != 0` / `func != NULL` -> 1 (true)
    /// - `0 == func` / `NULL == func` -> 0 (false)
    /// - `0 != func` / `NULL != func` -> 1 (true)
    ///
    /// This is used by the Linux kernel's _OF_DECLARE macro:
    ///   .data = (fn == (fn_type)NULL) ? fn : fn
    /// where the condition must be evaluated at compile time for a static initializer.
    fn eval_addr_comparison_cond(&self, expr: &Expr) -> Option<IrConst> {
        // Unwrap casts (e.g., (fn_type)NULL is Cast(fn_type, 0))
        let expr = match expr {
            Expr::Cast(_, inner, _) => inner.as_ref(),
            _ => expr,
        };
        match expr {
            Expr::BinaryOp(op, lhs, rhs, _) => {
                let is_eq = match op {
                    BinOp::Eq => true,
                    BinOp::Ne => false,
                    _ => return None,
                };
                // Check if one side is a global address and the other is zero/NULL
                let (_, scalar_side) = if self.eval_global_addr_expr(lhs).is_some() {
                    (lhs.as_ref(), rhs.as_ref())
                } else if self.eval_global_addr_expr(rhs).is_some() {
                    (rhs.as_ref(), lhs.as_ref())
                } else {
                    return None;
                };
                // The scalar side must be zero (NULL)
                let scalar_val = self.eval_const_expr(scalar_side)?;
                if !scalar_val.is_zero() {
                    return None;
                }
                // A defined function/global address is always non-zero
                // (already validated by eval_global_addr_expr above),
                // so fn == 0 is false (0), fn != 0 is true (1).
                Some(IrConst::I64(if is_eq { 0 } else { 1 }))
            }
            // Handle !func (logical not of function pointer)
            Expr::UnaryOp(UnaryOp::LogicalNot, inner, _) => {
                if self.eval_global_addr_expr(inner).is_some() {
                    // !func is false (0) since func is always non-zero
                    Some(IrConst::I64(0))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
