use crate::frontend::parser::ast::{
    BinOp,
    Declaration,
    Expr,
    StructFieldDecl,
    TypeSpecifier,
    UnaryOp,
};
use crate::ir::reexports::{
    Instruction,
    IrConst,
    Operand,
    Value,
};
use std::rc::Rc;
use crate::common::types::{AddressSpace, IrType, StructLayout, RcLayout, CType};
use super::lower::Lowerer;

impl Lowerer {
    /// Register a struct/union type definition from a TypeSpecifier, computing and
    /// caching its layout in self.types.struct_layouts. Also recursively registers any
    /// nested struct/union types defined in the fields.
    pub(super) fn register_struct_type(&mut self, ts: &TypeSpecifier) {
        match ts {
            TypeSpecifier::Struct(tag, Some(fields), is_packed, pragma_pack, struct_aligned) => {
                // Recursively register nested struct/union types in fields
                self.register_nested_struct_types(fields);
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                let mut layout = self.compute_struct_union_layout_packed(fields, false, max_field_align);
                // Apply struct-level __attribute__((aligned(N))): sets minimum alignment
                if let Some(a) = struct_aligned {
                    if *a > layout.align {
                        layout.align = *a;
                        let mask = layout.align - 1;
                        layout.size = (layout.size + mask) & !mask;
                    }
                }
                let key = self.struct_layout_key(tag, false);
                self.insert_struct_layout_scoped(key.clone(), layout);
                // Also invalidate the ctype_cache for this tag so sizeof picks
                // up the new definition
                if tag.is_some() {
                    self.invalidate_ctype_cache_scoped(&key);
                }
            }
            TypeSpecifier::Union(tag, Some(fields), is_packed, pragma_pack, struct_aligned) => {
                // Recursively register nested struct/union types in fields
                self.register_nested_struct_types(fields);
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                let mut layout = self.compute_struct_union_layout_packed(fields, true, max_field_align);
                // Apply struct-level __attribute__((aligned(N))): sets minimum alignment
                if let Some(a) = struct_aligned {
                    if *a > layout.align {
                        layout.align = *a;
                        let mask = layout.align - 1;
                        layout.size = (layout.size + mask) & !mask;
                    }
                }
                let key = self.struct_layout_key(tag, true);
                self.insert_struct_layout_scoped(key.clone(), layout);
                if tag.is_some() {
                    self.invalidate_ctype_cache_scoped(&key);
                }
            }
            _ => {}
        }
    }

    /// Re-compute struct/union layouts that contain vector typedef fields.
    /// Sema computed these layouts before vector_size was applied to typedefs,
    /// so their field sizes may be wrong (e.g., float4 treated as float instead
    /// of Vector(float, 16)). This updates the EXISTING layout entries by key
    /// rather than creating new ones, since CType::Union/Struct references
    /// use the key assigned by sema.
    pub(super) fn recompute_vector_struct_layouts(
        &mut self,
        tu: &crate::frontend::parser::ast::TranslationUnit,
    ) {
        use crate::frontend::parser::ast::ExternalDecl;
        let mut type_specs_to_recompute: Vec<&TypeSpecifier> = Vec::new();
        for decl in &tu.decls {
            match decl {
                ExternalDecl::Declaration(decl) => {
                    Self::collect_struct_union_type_specs(&decl.type_spec, &mut type_specs_to_recompute);
                }
                ExternalDecl::FunctionDef(func) => {
                    Self::collect_struct_union_type_specs(&func.return_type, &mut type_specs_to_recompute);
                    for p in &func.params {
                        Self::collect_struct_union_type_specs(&p.type_spec, &mut type_specs_to_recompute);
                    }
                }
                ExternalDecl::TopLevelAsm(_) => {}
            }
        }
        for ts in type_specs_to_recompute {
            self.recompute_layout_if_vector_fields(ts);
        }
    }

    /// Collect all struct/union TypeSpecifiers with inline field definitions from a type.
    fn collect_struct_union_type_specs<'a>(ts: &'a TypeSpecifier, out: &mut Vec<&'a TypeSpecifier>) {
        match ts {
            TypeSpecifier::Struct(_, Some(fields), _, _, _) |
            TypeSpecifier::Union(_, Some(fields), _, _, _) => {
                out.push(ts);
                for f in fields {
                    Self::collect_struct_union_type_specs(&f.type_spec, out);
                }
            }
            _ => {}
        }
    }

    /// Re-compute a struct/union layout if any of its fields use vector typedefs.
    /// Updates the existing layout entry in the map using the key from sema.
    fn recompute_layout_if_vector_fields(&mut self, ts: &TypeSpecifier) {
        let (tag, fields, is_union, is_packed, pragma_pack, struct_aligned) = match ts {
            TypeSpecifier::Struct(tag, Some(fields), is_packed, pragma_pack, struct_aligned) =>
                (tag, fields, false, *is_packed, *pragma_pack, *struct_aligned),
            TypeSpecifier::Union(tag, Some(fields), is_packed, pragma_pack, struct_aligned) =>
                (tag, fields, true, *is_packed, *pragma_pack, *struct_aligned),
            _ => return,
        };
        // Check if any field uses a vector typedef
        let has_vector_field = fields.iter().any(|f| {
            let ctype = self.struct_field_ctype(f);
            ctype.is_vector()
        });
        if !has_vector_field { return; }

        // Find the existing layout key. For tagged types, use the tag-based key.
        // For anonymous types (e.g., typedef union { ... } name), find the key
        // that sema assigned by searching typedefs for a matching CType.
        let existing_key: Option<Rc<str>> = if let Some(name) = tag {
            let prefix = if is_union { "union." } else { "struct." };
            Some(Rc::from(format!("{}{}", prefix, name)))
        } else {
            let layouts = self.types.borrow_struct_layouts();
            let mut found_key = None;
            for (_, ctype) in self.types.typedefs.iter() {
                match ctype {
                    CType::Struct(key) | CType::Union(key) => {
                        if let Some(layout) = layouts.get(&**key) {
                            if layout.is_union == is_union && layout.fields.len() == fields.len() {
                                found_key = Some(key.clone());
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
            drop(layouts);
            found_key
        };

        if let Some(key) = existing_key {
            let max_field_align = if is_packed { Some(1) } else { pragma_pack };
            let mut layout = self.compute_struct_union_layout_packed(fields, is_union, max_field_align);
            if let Some(a) = struct_aligned {
                if a > layout.align {
                    layout.align = a;
                    let mask = layout.align - 1;
                    layout.size = (layout.size + mask) & !mask;
                }
            }
            self.types.insert_struct_layout_from_ref(key, layout);
        }
    }

    /// Insert a struct layout into the cache, tracking the change in the current
    /// scope frame so it can be undone on scope exit.
    fn insert_struct_layout_scoped(&mut self, key: Rc<str>, layout: StructLayout) {
        self.types.insert_struct_layout_scoped(key, layout);
    }

    /// Invalidate a ctype_cache entry for a struct/union tag, tracking the change
    /// in the current scope frame so it can be restored on scope exit.
    fn invalidate_ctype_cache_scoped(&mut self, key: &str) {
        self.types.invalidate_ctype_cache_scoped(key);
    }

    /// Recursively register any struct/union types defined inline in field declarations.
    /// This handles cases like:
    ///   struct Outer { struct Inner { int x; } field; };
    /// where `struct Inner` needs to be registered so it can be referenced later.
    fn register_nested_struct_types(&mut self, fields: &[StructFieldDecl]) {
        for field in fields {
            self.register_nested_in_type_spec(&field.type_spec);
        }
    }

    /// Walk a TypeSpecifier and register any struct/union definitions found within it.
    fn register_nested_in_type_spec(&mut self, ts: &TypeSpecifier) {
        match ts {
            TypeSpecifier::Struct(_tag, Some(_fields), _, _, _) => {
                // This is a struct definition inside a field (named or anonymous) - register it
                self.register_struct_type(ts);
            }
            TypeSpecifier::Union(_tag, Some(_fields), _, _, _) => {
                // This is a union definition inside a field (named or anonymous) - register it
                self.register_struct_type(ts);
            }
            TypeSpecifier::Pointer(inner, _) => {
                // Walk through pointer types to find nested struct defs
                self.register_nested_in_type_spec(inner);
            }
            TypeSpecifier::Array(inner, _) => {
                self.register_nested_in_type_spec(inner);
            }
            _ => {}
        }
    }

    /// Compute a layout key for a struct/union.
    fn struct_layout_key(&mut self, tag: &Option<Rc<str>>, is_union: bool) -> Rc<str> {
        let prefix = if is_union { "union." } else { "struct." };
        if let Some(name) = tag {
            Rc::from(format!("{}{}", prefix, name))
        } else {
            let id = self.next_anon_struct;
            self.next_anon_struct += 1;
            Rc::from(format!("{}__anon_{}", prefix, id))
        }
    }

    /// Get the StructLayout key for a union TypeSpecifier.
    /// Returns the layout map key if the type is a union (directly or via typedef).
    pub(super) fn union_layout_key(&self, ts: &TypeSpecifier) -> Option<Rc<str>> {
        match ts {
            TypeSpecifier::Union(tag, _, _, _, _) => {
                tag.as_ref().map(|name| Rc::from(format!("union.{}", name)))
            }
            TypeSpecifier::TypedefName(name) => {
                if let Some(CType::Union(key)) = self.types.typedefs.get(name) {
                    return Some(key.clone());
                }
                None
            }
            _ => None,
        }
    }

    /// Mark a transparent_union attribute on the union's StructLayout.
    ///
    /// Resolves the union layout key either from the type specifier directly
    /// or by searching through declarator names for typedef aliases.
    /// This is needed in two places: the pre-pass in `lower()` and `lower_global_decl`.
    pub(super) fn mark_transparent_union(&mut self, decl: &Declaration) {
        let mut found_key = self.union_layout_key(&decl.type_spec);
        if found_key.is_none() {
            for declarator in &decl.declarators {
                if !declarator.name.is_empty() {
                    if let Some(CType::Union(key)) = self.types.typedefs.get(&declarator.name) {
                        found_key = Some(key.clone());
                        break;
                    }
                }
            }
        }
        if let Some(key) = found_key {
            if let Some(layout) = self.types.borrow_struct_layouts_mut().get_mut(&*key) {
                Rc::make_mut(layout).is_transparent_union = true;
            }
        }
    }

    /// Get the cached struct layout for a TypeSpecifier, if it's a struct/union type.
    /// Prefers cached layout from struct_layouts when a tag name is available.
    /// Returns Rc<StructLayout> for cheap cloning.
    pub(super) fn get_struct_layout_for_type(&self, ts: &TypeSpecifier) -> Option<RcLayout> {
        // For TypedefName, resolve through CType
        if let TypeSpecifier::TypedefName(name) = ts {
            if let Some(ctype) = self.types.typedefs.get(name) {
                return self.struct_layout_from_ctype(ctype);
            }
            return None;
        }
        // For typeof(expr), resolve through the expression's CType.
        // This handles patterns like typeof(*ptr) where ptr is a struct pointer,
        // which appear in offsetof patterns: &((typeof(*ptr) *)0)->member
        if let TypeSpecifier::Typeof(expr) = ts {
            if let Some(ctype) = self.get_expr_ctype(expr) {
                return self.struct_layout_from_ctype(&ctype);
            }
            return None;
        }
        let ts = self.resolve_type_spec(ts);
        match ts {
            TypeSpecifier::Struct(tag, Some(fields), is_packed, pragma_pack, _) => {
                if let Some(tag) = tag {
                    let layouts = self.types.borrow_struct_layouts();
                    let key = format!("struct.{}", tag);
                    if let Some(layout) = layouts.get(key.as_str())
                        .or_else(|| layouts.get(&**tag))
                    {
                        return Some(layout.clone());
                    }
                }
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                Some(Rc::new(self.compute_struct_union_layout_packed(fields, false, max_field_align)))
            }
            TypeSpecifier::Struct(Some(tag), None, _, _, _) => {
                let layouts = self.types.borrow_struct_layouts();
                let key = format!("struct.{}", tag);
                layouts.get(key.as_str()).cloned()
                    .or_else(|| {
                        // Anonymous structs from typeof/ctype_to_type_spec use the
                        // raw CType key (e.g., "__anon_struct_N") as the tag.
                        layouts.get(&**tag).cloned()
                    })
            }
            TypeSpecifier::Union(tag, Some(fields), is_packed, pragma_pack, _) => {
                if let Some(tag) = tag {
                    let layouts = self.types.borrow_struct_layouts();
                    let key = format!("union.{}", tag);
                    if let Some(layout) = layouts.get(key.as_str())
                        .or_else(|| layouts.get(&**tag))
                    {
                        return Some(layout.clone());
                    }
                }
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                Some(Rc::new(self.compute_struct_union_layout_packed(fields, true, max_field_align)))
            }
            TypeSpecifier::Union(Some(tag), None, _, _, _) => {
                let layouts = self.types.borrow_struct_layouts();
                let key = format!("union.{}", tag);
                layouts.get(key.as_str()).cloned()
                    .or_else(|| {
                        // Anonymous unions from typeof/ctype_to_type_spec use the
                        // raw CType key (e.g., "__anon_struct_N") as the tag.
                        layouts.get(&**tag).cloned()
                    })
            }
            // For typedef'd array types like `typedef S arr_t[4]`, peel the
            // Array wrapper(s) to find the inner struct/union element type.
            TypeSpecifier::Array(inner, _) => {
                self.get_struct_layout_for_type(inner)
            }
            _ => None,
        }
    }

    /// Get the base address of a struct variable (for member access).
    /// For struct locals, the alloca IS the struct base.
    /// For struct globals, we emit GlobalAddr.
    pub(super) fn get_struct_base_addr(&mut self, expr: &Expr) -> Value {
        match expr {
            Expr::Identifier(name, _) => {
                // Static local variables: resolve via mangled global name
                if let Some(mangled) = self.func_mut().static_local_names.get(name).cloned() {
                    let addr = self.fresh_value();
                    self.emit(Instruction::GlobalAddr { dest: addr, name: mangled });
                    return addr;
                }
                if let Some(info) = self.func_mut().locals.get(name).cloned() {
                    // Static locals: emit fresh GlobalAddr
                    if let Some(ref global_name) = info.static_global_name {
                        let addr = self.fresh_value();
                        self.emit(Instruction::GlobalAddr { dest: addr, name: global_name.clone() });
                        return addr;
                    }
                    if info.is_struct {
                        // The alloca is the struct base address
                        return info.alloca;
                    }
                    // Vector types also store data inline at their alloca address
                    if info.c_type.as_ref().is_some_and(|ct| ct.is_vector()) {
                        return info.alloca;
                    }
                    // It's a pointer to struct: load the pointer
                    let loaded = self.fresh_value();
                    self.emit(Instruction::Load { dest: loaded, ptr: info.alloca, ty: IrType::Ptr , seg_override: AddressSpace::Default });
                    return loaded;
                }
                if self.globals.contains_key(&**name) {
                    let addr = self.fresh_value();
                    self.emit(Instruction::GlobalAddr { dest: addr, name: name.clone() });
                    return addr;
                }
                // Unknown - try to evaluate as expression
                let val = self.lower_expr(expr);
                match val {
                    Operand::Value(v) => v,
                    Operand::Const(_) => {
                        let tmp = self.fresh_value();
                        self.emit(Instruction::Copy { dest: tmp, src: val });
                        tmp
                    }
                }
            }
            Expr::Deref(inner, _) => {
                // (*ptr).field - evaluate ptr to get base address
                let val = self.lower_expr(inner);
                match val {
                    Operand::Value(v) => v,
                    Operand::Const(_) => {
                        let tmp = self.fresh_value();
                        self.emit(Instruction::Copy { dest: tmp, src: val });
                        tmp
                    }
                }
            }
            Expr::ArraySubscript(base, index, _) => {
                // array[i].field
                self.compute_array_element_addr(base, index)
            }
            Expr::MemberAccess(inner_base, inner_field, _) => {
                // Nested member access: s.inner.field
                let (inner_offset, _) = self.resolve_member_access(inner_base, inner_field);
                let inner_base_addr = self.get_struct_base_addr(inner_base);
                let inner_addr = self.fresh_value();
                self.emit(Instruction::GetElementPtr {
                    dest: inner_addr,
                    base: inner_base_addr,
                    offset: Operand::Const(IrConst::ptr_int(inner_offset as i64)),
                    ty: IrType::Ptr,
                });
                inner_addr
            }
            Expr::PointerMemberAccess(inner_base, inner_field, _) => {
                // Nested access via pointer: p->inner.field
                // Load the pointer, compute offset of the inner field, return address
                let ptr_val = self.lower_expr(inner_base);
                let base_addr = match ptr_val {
                    Operand::Value(v) => v,
                    Operand::Const(_) => {
                        let tmp = self.fresh_value();
                        self.emit(Instruction::Copy { dest: tmp, src: ptr_val });
                        tmp
                    }
                };
                let (inner_offset, _) = self.resolve_pointer_member_access(inner_base, inner_field);
                let inner_addr = self.fresh_value();
                self.emit(Instruction::GetElementPtr {
                    dest: inner_addr,
                    base: base_addr,
                    offset: Operand::Const(IrConst::ptr_int(inner_offset as i64)),
                    ty: IrType::Ptr,
                });
                inner_addr
            }
            Expr::FunctionCall(func_expr, _, _) => {
                // Function returning a struct: call the function, then store
                // the return value to a temporary alloca so we have an address.
                let struct_size = self.struct_value_size(expr).unwrap_or(8);

                // Check if this is an sret call (struct > 16 bytes with hidden pointer)
                // or a two-register return (9-16 bytes) - both return an alloca address.
                // On i686 (32-bit), ALL struct returns use sret regardless of size.
                let is_32bit = crate::common::types::target_is_32bit();
                let returns_address = if is_32bit {
                    // i686: all struct returns use hidden pointer (sret)
                    true
                } else if let Expr::Identifier(name, _) = func_expr.as_ref() {
                    // Detect function pointer variables: identifiers that are
                    // local/global variables rather than known function names
                    let is_fptr_var = (self.func_mut().locals.contains_key(name) && !self.known_functions.contains(&**name))
                        || (!self.func_mut().locals.contains_key(name) && self.globals.contains_key(&**name) && !self.known_functions.contains(&**name));
                    if is_fptr_var {
                        // Indirect call through variable: use struct size to determine ABI
                        struct_size > 8
                    } else {
                        self.func_meta.sigs.get(&**name).is_some_and(|s| s.sret_size.is_some() || s.two_reg_ret_size.is_some())
                    }
                } else {
                    // Indirect call through expression: determine from return type
                    struct_size > 8
                };

                if returns_address {
                    // For sret and two-register calls, lower_expr returns the alloca
                    // address directly (the struct data is already there)
                    let val = self.lower_expr(expr);
                    match val {
                        Operand::Value(v) => v,
                        Operand::Const(_) => {
                            let tmp = self.fresh_value();
                            self.emit(Instruction::Copy { dest: tmp, src: val });
                            tmp
                        }
                    }
                } else {
                    // Small struct (<= 8 bytes): return value in rax IS
                    // the packed struct data, not an address.
                    let val = self.lower_expr(expr);
                    let alloca = self.fresh_value();
                    let alloc_size = if struct_size > 0 { struct_size } else { 8 };
                    let store_ty = Self::packed_store_type(alloc_size);
                    self.emit(Instruction::Alloca { dest: alloca, size: alloc_size, ty: store_ty, align: 0, volatile: false });
                    self.emit(Instruction::Store { val, ptr: alloca, ty: store_ty , seg_override: AddressSpace::Default });
                    alloca
                }
            }
            // Ternary and assignment on struct values produce rvalues (temporaries).
            // We must copy the struct data to a fresh temporary so that subsequent
            // member writes don't modify the original objects.
            Expr::Conditional(_, _, _, _) | Expr::GnuConditional(_, _, _) | Expr::Assign(_, _, _) => {
                if let Some(struct_size) = self.struct_value_size(expr) {
                    if self.expr_produces_packed_struct_data(expr) {
                        // Small struct packed in a register: spill to alloca
                        let val = self.lower_expr(expr);
                        let alloca = self.fresh_value();
                        let alloc_size = if struct_size > 0 { struct_size } else { 8 };
                        let store_ty = Self::packed_store_type(alloc_size);
                        self.emit(Instruction::Alloca { dest: alloca, size: alloc_size, ty: store_ty, align: 0, volatile: false });
                        self.emit(Instruction::Store { val, ptr: alloca, ty: store_ty, seg_override: AddressSpace::Default });
                        alloca
                    } else {
                        // Struct returned by address: copy to a fresh temporary
                        let src_val = self.lower_expr(expr);
                        let src_addr = self.operand_to_value(src_val);
                        let tmp_alloca = self.fresh_value();
                        self.emit(Instruction::Alloca { dest: tmp_alloca, size: struct_size, ty: IrType::Ptr, align: 0, volatile: false });
                        self.emit(Instruction::Memcpy { dest: tmp_alloca, src: src_addr, size: struct_size });
                        tmp_alloca
                    }
                } else {
                    // Not a struct type - evaluate normally
                    let val = self.lower_expr(expr);
                    match val {
                        Operand::Value(v) => v,
                        Operand::Const(_) => {
                            let tmp = self.fresh_value();
                            self.emit(Instruction::Copy { dest: tmp, src: val });
                            tmp
                        }
                    }
                }
            }
            _ => {
                // For expressions that might produce packed struct data (e.g. ternary
                // with struct-returning function calls), detect and spill to an alloca.
                if self.expr_produces_packed_struct_data(expr) {
                    let struct_size = self.struct_value_size(expr).unwrap_or(8);
                    let val = self.lower_expr(expr);
                    let alloca = self.fresh_value();
                    let alloc_size = if struct_size > 0 { struct_size } else { 8 };
                    let store_ty = Self::packed_store_type(alloc_size);
                    self.emit(Instruction::Alloca { dest: alloca, size: alloc_size, ty: store_ty, align: 0, volatile: false });
                    self.emit(Instruction::Store { val, ptr: alloca, ty: store_ty , seg_override: AddressSpace::Default });
                    alloca
                } else {
                    let val = self.lower_expr(expr);
                    match val {
                        Operand::Value(v) => v,
                        Operand::Const(_) => {
                            let tmp = self.fresh_value();
                            self.emit(Instruction::Copy { dest: tmp, src: val });
                            tmp
                        }
                    }
                }
            }
        }
    }

    /// Check if an expression produces packed struct data (non-address value)
    /// rather than a pointer to struct data. This happens for small structs
    /// (<= 8 bytes) returned from non-sret, non-two-reg function calls, either
    /// directly or through ternary/comma expressions.
    pub(super) fn expr_produces_packed_struct_data(&self, expr: &Expr) -> bool {
        match expr {
            Expr::FunctionCall(func_expr, _, _) => {
                // On i686 (32-bit), ALL struct returns use sret, never packed data
                if crate::common::types::target_is_32bit() {
                    return false;
                }
                let struct_size = self.struct_value_size(expr).unwrap_or(8);
                if struct_size > 8 {
                    // 9+ byte structs: sret or two-reg return, both produce addresses
                    return false;
                }
                // Small struct (<= 8 bytes): produces packed data unless somehow sret
                if let Expr::Identifier(name, _) = func_expr.as_ref() {
                    self.func_meta.sigs.get(&**name).is_none_or(|s| s.sret_size.is_none() && s.two_reg_ret_size.is_none())
                } else {
                    true
                }
            }
            Expr::Conditional(_, then_expr, else_expr, _) => {
                // If either branch produces packed struct data, the ternary does too
                self.expr_produces_packed_struct_data(then_expr)
                    || self.expr_produces_packed_struct_data(else_expr)
            }
            Expr::GnuConditional(cond, else_expr, _) => {
                self.expr_produces_packed_struct_data(cond)
                    || self.expr_produces_packed_struct_data(else_expr)
            }
            Expr::Comma(_, last, _) => {
                self.expr_produces_packed_struct_data(last)
            }
            Expr::StmtExpr(compound, _) => {
                // Statement expression: check if the last expression statement
                // produces packed struct data (e.g., ({ pfn_pte(...); }))
                if let Some(crate::frontend::parser::ast::BlockItem::Statement(
                    crate::frontend::parser::ast::Stmt::Expr(Some(inner_expr))
                )) = compound.items.last() {
                    return self.expr_produces_packed_struct_data(inner_expr);
                }
                false
            }
            _ => false,
        }
    }

    /// Look up the struct layout for a member access expression.
    /// For direct access (s.field), gets the layout of the base expression.
    /// For pointer access (p->field), gets the layout that the pointer points to.
    fn get_member_layout(&self, base_expr: &Expr, is_pointer_access: bool) -> Option<RcLayout> {
        if is_pointer_access {
            self.get_pointed_struct_layout(base_expr)
        } else {
            self.get_layout_for_expr(base_expr)
        }
    }

    /// Resolve member access: returns (byte_offset, ir_type_of_field).
    /// Works for both direct (s.field) and pointer (p->field) access.
    pub(super) fn resolve_member_access(&self, base_expr: &Expr, field_name: &str) -> (usize, IrType) {
        self.resolve_member_access_impl(base_expr, field_name, false)
    }

    /// Resolve pointer member access (p->field): returns (byte_offset, ir_type_of_field).
    pub(super) fn resolve_pointer_member_access(&self, base_expr: &Expr, field_name: &str) -> (usize, IrType) {
        self.resolve_member_access_impl(base_expr, field_name, true)
    }

    fn resolve_member_access_impl(&self, base_expr: &Expr, field_name: &str, is_pointer: bool) -> (usize, IrType) {
        if let Some(layout) = self.get_member_layout(base_expr, is_pointer) {
            if let Some((offset, ctype)) = layout.field_offset(field_name, &*self.types.borrow_struct_layouts()) {
                return (offset, IrType::from_ctype(&ctype));
            }
        }
        (0, IrType::I32)
    }

    /// Resolve member access with full bitfield info.
    /// Returns (byte_offset, ir_type, Option<(bit_offset, bit_width)>).
    pub(super) fn resolve_member_access_full(&self, base_expr: &Expr, field_name: &str) -> (usize, IrType, Option<(u32, u32)>) {
        self.resolve_member_access_full_impl(base_expr, field_name, false)
    }

    /// Resolve pointer member access with full bitfield info.
    pub(super) fn resolve_pointer_member_access_full(&self, base_expr: &Expr, field_name: &str) -> (usize, IrType, Option<(u32, u32)>) {
        self.resolve_member_access_full_impl(base_expr, field_name, true)
    }

    fn resolve_member_access_full_impl(&self, base_expr: &Expr, field_name: &str, is_pointer: bool) -> (usize, IrType, Option<(u32, u32)>) {
        if let Some(layout) = self.get_member_layout(base_expr, is_pointer) {
            if let Some(fl) = layout.field_layout(field_name) {
                let ir_ty = IrType::from_ctype(&fl.ty);
                let bf = match (fl.bit_offset, fl.bit_width) {
                    (Some(bo), Some(bw)) => Some((bo, bw)),
                    _ => None,
                };
                return (fl.offset, ir_ty, bf);
            }
            // Fallback: search anonymous struct/union members recursively,
            // preserving bitfield metadata (bit_offset, bit_width).
            if let Some((offset, ctype, bit_offset, bit_width)) = layout.field_offset_with_bitfield(field_name, &*self.types.borrow_struct_layouts()) {
                let bf = match (bit_offset, bit_width) {
                    (Some(bo), Some(bw)) => Some((bo, bw)),
                    _ => None,
                };
                return (offset, IrType::from_ctype(&ctype), bf);
            }
        }
        (0, IrType::I32, None)
    }

    /// Try to determine the struct layout that an expression (a pointer) points to.
    fn get_pointed_struct_layout(&self, expr: &Expr) -> Option<RcLayout> {
        match expr {
            Expr::Identifier(name, _) => {
                // Check static locals first (only in function context)
                if let Some(ref fs) = self.func_state {
                    if let Some(mangled) = fs.static_local_names.get(name) {
                        if let Some(ginfo) = self.globals.get(&**mangled) {
                            if ginfo.struct_layout.is_some() {
                                return ginfo.struct_layout.clone();
                            }
                        }
                    }
                }
                // Check if this variable has struct layout info (pointer to struct).
                // Prefer live lookup from struct_layouts over the VarInfo snapshot,
                // because the snapshot may be stale when a struct tag is redefined
                // in the current scope (the VarInfo was captured at declaration time
                // before the full struct definition was processed).
                if let Some(vi) = self.lookup_var_info(name) {
                    // First try live lookup via c_type -> struct_layouts map
                    if let Some(ref ct) = vi.c_type {
                        if let Some(layout) = self.resolve_struct_from_pointer_ctype(ct) {
                            return Some(layout);
                        }
                    }
                    // Fall back to the snapshot captured at declaration time
                    if let Some(ref layout) = vi.struct_layout {
                        return Some(layout.clone());
                    }
                }
                // Fallback: resolve struct layout from the variable's CType
                // (handles forward-declared struct pointers where layout was None at decl time,
                // and array typedef parameters like `typedef struct S arr[1]`)
                match self.get_expr_ctype(expr).as_ref() {
                    Some(CType::Pointer(pointee, _)) | Some(CType::Array(pointee, _)) => {
                        return self.struct_layout_from_ctype(pointee);
                    }
                    _ => {}
                }
                None
            }
            Expr::MemberAccess(base, field, _) | Expr::PointerMemberAccess(base, field, _) => {
                let is_ptr = matches!(expr, Expr::PointerMemberAccess(..));
                self.resolve_field_struct_layout(base, field, is_ptr, true)
            }
            Expr::UnaryOp(UnaryOp::PreInc | UnaryOp::PreDec, inner, _) => {
                self.get_pointed_struct_layout(inner)
            }
            Expr::PostfixOp(_, inner, _) => {
                self.get_pointed_struct_layout(inner)
            }
            Expr::BinaryOp(op, lhs, rhs, _) if matches!(op, BinOp::Add | BinOp::Sub) => {
                // Pointer arithmetic: (p + i) or (p - i) preserves the pointed-to struct type
                if let Some(layout) = self.get_pointed_struct_layout(lhs) {
                    return Some(layout);
                }
                // Try rhs for commutative case: (i + p)
                if matches!(op, BinOp::Add) {
                    if let Some(layout) = self.get_pointed_struct_layout(rhs) {
                        return Some(layout);
                    }
                }
                None
            }
            Expr::Cast(type_spec, inner, _) => {
                // Cast to struct pointer type
                if let Some(layout) = self.get_struct_layout_for_pointer_type(type_spec) {
                    return Some(layout);
                }
                // Try inner expression
                self.get_pointed_struct_layout(inner)
            }
            Expr::AddressOf(inner, _) => {
                // &expr - result is a pointer to expr's type
                self.get_layout_for_expr(inner)
            }
            Expr::Deref(inner, _) => {
                // *pp where pp is a pointer to pointer to struct
                // or *pa where pa is a pointer to array of struct (array decays to struct pointer)
                if let Some(CType::Pointer(inner_ct, _)) = self.get_expr_ctype(inner).as_ref() {
                    if let CType::Pointer(pointee, _) = inner_ct.as_ref() {
                        return self.struct_layout_from_ctype(pointee);
                    }
                    // Pointer to array of struct: *pa dereferences to the array,
                    // which decays to a pointer to the element struct type.
                    // This handles patterns like cpumask_var_t (typedef struct cpumask[1])
                    // where &var is pointer-to-array and *&var decays to struct pointer.
                    if let CType::Array(elem, _) = inner_ct.as_ref() {
                        return self.struct_layout_from_ctype(elem);
                    }
                }
                // Fallback: propagate through inner
                self.get_pointed_struct_layout(inner)
            }
            Expr::ArraySubscript(base, _, _) => {
                // pp[i] where pp is an array of struct pointers
                if let Some(CType::Array(elem, _) | CType::Pointer(elem, _)) = self.get_expr_ctype(base).as_ref() {
                    if let CType::Pointer(pointee, _) = elem.as_ref() {
                        return self.struct_layout_from_ctype(pointee);
                    }
                }
                // Fallback: try base directly
                self.get_pointed_struct_layout(base)
            }
            Expr::FunctionCall(func, _, _) => {
                self.resolve_func_call_struct_layout(func, expr, true)
            }
            Expr::Conditional(_, then_expr, else_expr, _) => {
                // Try both branches: one may be a typed struct pointer and
                // the other may be (void*)0 (e.g., ql_last macro pattern).
                self.get_pointed_struct_layout(then_expr)
                    .or_else(|| self.get_pointed_struct_layout(else_expr))
            }
            Expr::GnuConditional(cond, else_expr, _) => {
                self.get_pointed_struct_layout(cond)
                    .or_else(|| self.get_pointed_struct_layout(else_expr))
            }
            Expr::Comma(_, last, _) => {
                self.get_pointed_struct_layout(last)
            }
            Expr::StmtExpr(..) => {
                // Statement expression: use CType resolution to find the
                // pointed-to struct type, since inner variables may not be
                // in scope yet during layout lookup.
                if let Some(CType::Pointer(pointee, _)) =
                    self.get_expr_ctype(expr).as_ref()
                {
                    return self.struct_layout_from_ctype(pointee);
                }
                None
            }
            Expr::Assign(lhs, rhs, _) | Expr::CompoundAssign(_, lhs, rhs, _) => {
                // For assignment, the result has the LHS type; try LHS first, then RHS
                if let Some(layout) = self.get_pointed_struct_layout(lhs) {
                    return Some(layout);
                }
                self.get_pointed_struct_layout(rhs)
            }
            _ => {
                // Generic fallback: try CType-based resolution
                if let Some(ctype) = self.get_expr_ctype(expr) {
                    if let CType::Pointer(pointee, _) = &ctype {
                        return self.struct_layout_from_ctype(pointee);
                    }
                    // Array of struct decays to struct pointer (e.g., after deref of
                    // pointer-to-array-of-struct, as in typedef struct S arr_t[1] patterns).
                    if let CType::Array(elem, _) = &ctype {
                        return self.struct_layout_from_ctype(elem);
                    }
                }
                None
            }
        }
    }

    /// Get struct layout from a CType (struct or union).
    /// Prefers cached layout from struct_layouts when available.
    pub(super) fn struct_layout_from_ctype(&self, ctype: &CType) -> Option<RcLayout> {
        match ctype {
            CType::Struct(key) | CType::Union(key) => {
                self.types.borrow_struct_layouts().get(key.as_ref()).cloned()
            }
            _ => None,
        }
    }

    /// Get struct layout from a type specifier that should be a pointer to struct
    fn get_struct_layout_for_pointer_type(&self, type_spec: &TypeSpecifier) -> Option<RcLayout> {
        let ctype = self.type_spec_to_ctype(type_spec);
        if let CType::Pointer(pointee, _) = &ctype {
            return self.struct_layout_from_ctype(pointee);
        }
        None
    }

    /// Try to get the struct layout for an expression (value, not pointer).
    fn get_layout_for_expr(&self, expr: &Expr) -> Option<RcLayout> {
        match expr {
            Expr::Identifier(name, _) => {
                // Check static locals first (resolved via mangled name)
                // Use func_state directly to avoid panic when called from global initializer context
                if let Some(ref fs) = self.func_state {
                    if let Some(mangled) = fs.static_local_names.get(name) {
                        if let Some(ginfo) = self.globals.get(&**mangled) {
                            return ginfo.struct_layout.clone();
                        }
                    }
                }
                if let Some(vi) = self.lookup_var_info(name) {
                    return vi.struct_layout.clone();
                }
                None
            }
            Expr::MemberAccess(base, field, _) | Expr::PointerMemberAccess(base, field, _) => {
                let is_ptr = matches!(expr, Expr::PointerMemberAccess(..));
                self.resolve_field_struct_layout(base, field, is_ptr, false)
            }
            Expr::ArraySubscript(base, _, _) => {
                // array[i] where array is of struct type
                // First try getting the layout directly from base
                if let Some(layout) = self.get_layout_for_expr(base) {
                    return Some(layout);
                }
                // If base is a member access yielding an array of structs,
                // use CType to resolve the element type
                if let Some(base_ctype) = self.get_expr_ctype(base) {
                    let inner = match &base_ctype {
                        CType::Array(elem, _) => Some(elem.as_ref()),
                        CType::Pointer(pointee, _) => Some(pointee.as_ref()),
                        _ => None,
                    };
                    if let Some(inner_ct) = inner {
                        if let Some(layout) = self.struct_layout_from_ctype(inner_ct) {
                            return Some(layout);
                        }
                    }
                }
                // Also try: base is a pointer to struct/union (for ptr[i] patterns)
                self.get_pointed_struct_layout(base)
            }
            Expr::CompoundLiteral(type_spec, _, _) => {
                // Compound literal: (struct tag){...} - get layout from the type specifier
                self.get_struct_layout_for_type(type_spec)
            }
            Expr::Deref(inner, _) => {
                // (*ptr) where ptr points to a struct
                self.get_pointed_struct_layout(inner)
            }
            Expr::Cast(type_spec, inner, _) => {
                // Cast to struct type, or cast wrapping a compound literal
                self.get_struct_layout_for_type(type_spec)
                    .or_else(|| self.get_layout_for_expr(inner))
            }
            Expr::FunctionCall(func, _, _) => {
                self.resolve_func_call_struct_layout(func, expr, false)
            }
            Expr::Assign(lhs, _, _) | Expr::CompoundAssign(_, lhs, _, _) => {
                self.get_layout_for_expr(lhs)
            }
            Expr::Conditional(_, then_expr, _, _) => {
                self.get_layout_for_expr(then_expr)
            }
            Expr::GnuConditional(cond, _, _) => {
                self.get_layout_for_expr(cond)
            }
            Expr::Comma(_, last, _) => {
                self.get_layout_for_expr(last)
            }
            Expr::StmtExpr(..) => {
                // Statement expression: use CType resolution (which handles
                // inner scopes via sema) to find the struct type, since the
                // inner variables may not be in scope yet during layout lookup.
                if let Some(ctype) = self.get_expr_ctype(expr) {
                    return self.struct_layout_from_ctype(&ctype);
                }
                None
            }
            Expr::VaArg(_, type_spec, _) => {
                self.get_struct_layout_for_type(type_spec)
            }
            _ => None,
        }
    }

    /// Shared helper: given a field access (base.field or base->field), resolve the
    /// struct layout of the field's type. Used by both get_pointed_struct_layout and
    /// get_layout_for_expr for their MemberAccess/PointerMemberAccess arms.
    ///
    /// `is_pointer_base`: whether the base is accessed via pointer (->)
    /// `want_pointer_deref`: whether the caller wants the field treated as a pointer
    ///   (true for get_pointed_struct_layout: field is a pointer to struct, resolve its pointee)
    ///   (false for get_layout_for_expr: field is a struct, resolve its own layout)
    fn resolve_field_struct_layout(&self, base: &Expr, field: &str, is_pointer_base: bool, want_pointer_deref: bool) -> Option<RcLayout> {
        let base_layout = self.get_member_layout(base, is_pointer_base)?;
        let (_offset, ctype) = base_layout.field_offset(field, &*self.types.borrow_struct_layouts())?;
        if want_pointer_deref {
            // Caller wants what the field points to (get_pointed_struct_layout path)
            if let Some(layout) = self.resolve_struct_from_pointer_ctype(&ctype) {
                return Some(layout);
            }
        } else {
            // Caller wants the field's own layout (get_layout_for_expr path)
            if let Some(layout) = self.struct_layout_from_ctype(&ctype) {
                return Some(layout);
            }
        }
        // Both paths: if the field is an array of structs, resolve the element type
        if let CType::Array(ref elem, _) = ctype {
            return self.struct_layout_from_ctype(elem);
        }
        None
    }

    /// Shared helper: resolve struct layout from a function call expression.
    /// For get_pointed_struct_layout: the return type is a pointer to struct, resolve its pointee.
    /// For get_layout_for_expr: the return type is a struct, resolve its layout.
    fn resolve_func_call_struct_layout(&self, func: &Expr, call_expr: &Expr, want_pointer_deref: bool) -> Option<RcLayout> {
        // Try direct function name first
        if let Expr::Identifier(name, _) = func {
            if let Some(ctype) = self.func_meta.sigs.get(&**name).and_then(|s| s.return_ctype.as_ref()) {
                if want_pointer_deref {
                    if let CType::Pointer(pointee, _) = ctype {
                        return self.struct_layout_from_ctype(pointee);
                    }
                } else if let Some(layout) = self.struct_layout_from_ctype(ctype) {
                    return Some(layout);
                }
            }
        }
        // For indirect calls through function pointers, resolve from CType
        if let Some(ctype) = self.get_expr_ctype(call_expr) {
            if want_pointer_deref {
                if let CType::Pointer(pointee, _) = &ctype {
                    return self.struct_layout_from_ctype(pointee);
                }
            } else {
                return self.struct_layout_from_ctype(&ctype);
            }
        }
        None
    }

    /// Resolve member access to get the CType of the field (not just the IrType).
    /// This is needed to check if a field is an array type (for array decay behavior).
    /// Handles both direct (s.field) and pointer (p->field) access.
    pub(super) fn resolve_member_field_ctype_impl(&self, base_expr: &Expr, field_name: &str, is_pointer: bool) -> Option<CType> {
        if let Some(layout) = self.get_member_layout(base_expr, is_pointer) {
            if let Some((_offset, ctype)) = layout.field_offset(field_name, &*self.types.borrow_struct_layouts()) {
                return Some(ctype.clone());
            }
        }
        None
    }

    /// Resolve member access and return full info including CType.
    /// Returns (byte_offset, ir_type, bitfield_info, field_ctype).
    pub(super) fn resolve_member_access_with_ctype(&self, base_expr: &Expr, field_name: &str) -> (usize, IrType, Option<(u32, u32)>, Option<CType>) {
        self.resolve_member_access_with_ctype_impl(base_expr, field_name, false)
    }

    /// Resolve pointer member access and return full info including CType.
    /// Returns (byte_offset, ir_type, bitfield_info, field_ctype).
    pub(super) fn resolve_pointer_member_access_with_ctype(&self, base_expr: &Expr, field_name: &str) -> (usize, IrType, Option<(u32, u32)>, Option<CType>) {
        self.resolve_member_access_with_ctype_impl(base_expr, field_name, true)
    }

    fn resolve_member_access_with_ctype_impl(&self, base_expr: &Expr, field_name: &str, is_pointer: bool) -> (usize, IrType, Option<(u32, u32)>, Option<CType>) {
        if let Some(layout) = self.get_member_layout(base_expr, is_pointer) {
            if let Some(fl) = layout.field_layout(field_name) {
                let ir_ty = IrType::from_ctype(&fl.ty);
                let bf = match (fl.bit_offset, fl.bit_width) {
                    (Some(bo), Some(bw)) => Some((bo, bw)),
                    _ => None,
                };
                return (fl.offset, ir_ty, bf, Some(fl.ty.clone()));
            }
            // Search anonymous struct/union members recursively,
            // preserving bitfield metadata (bit_offset, bit_width).
            if let Some((offset, ctype, bit_offset, bit_width)) = layout.field_offset_with_bitfield(field_name, &*self.types.borrow_struct_layouts()) {
                let bf = match (bit_offset, bit_width) {
                    (Some(bo), Some(bw)) => Some((bo, bw)),
                    _ => None,
                };
                return (offset, IrType::from_ctype(&ctype), bf, Some(ctype.clone()));
            }
        }
        (0, IrType::I32, None, None)
    }

    /// Given a CType that should be a Pointer to a struct, resolve the struct layout.
    /// Handles self-referential structs by looking up the cache when fields are empty.
    /// Also handles array types (e.g., `typedef struct S my_arr[1]` parameters decay
    /// to pointers, so the element type is the pointed-to struct).
    fn resolve_struct_from_pointer_ctype(&self, ctype: &CType) -> Option<RcLayout> {
        match ctype {
            CType::Pointer(inner, _) | CType::Array(inner, _) => self.struct_layout_from_ctype(inner),
            _ => None,
        }
    }
}
