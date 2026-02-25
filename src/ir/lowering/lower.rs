//! Core lowering logic: AST -> alloca-based IR.
//!
//! This file contains the `Lowerer` struct and its primary methods:
//! - Construction and top-level orchestration (`lower()`)
//! - IR emission helpers (fresh_value, emit, terminate, etc.)
//! - Scope management delegation
//! - Enum/label helpers and string/array init helpers
//!
//! Function lowering pipeline is in `func_lowering.rs`, global declaration
//! lowering and declaration analysis are in `global_decl.rs`.
//!
//! Data structure definitions are in `definitions.rs`, per-function state in
//! `func_state.rs`, and type-system state in `frontend::sema::type_context`.

use std::cell::RefCell;
use std::mem::Discriminant;
use std::rc::Rc;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::error::DiagnosticEngine;
use crate::common::source::Span;
use crate::frontend::parser::ast::{
    DerivedDeclarator,
    Expr,
    ExprId,
    ExternalDecl,
    ParamDecl,
    TranslationUnit,
    TypeSpecifier,
};
use crate::frontend::sema::{FunctionInfo, ExprTypeMap, ConstMap};
use crate::ir::reexports::{
    BasicBlock,
    BlockId,
    CallInfo,
    Instruction,
    IrBinOp,
    IrCmpOp,
    IrConst,
    IrModule,
    Operand,
    Terminator,
    Value,
};
use crate::common::types::{AddressSpace, IrType, CType};
use crate::backend::Target;
use super::definitions::*;
use super::func_state::FunctionBuildState;
use crate::frontend::sema::type_context::TypeContext;

/// Lowers AST to IR (alloca-based, not yet SSA).
pub struct Lowerer {
    /// Target architecture, used for ABI-specific lowering decisions
    pub(super) target: Target,
    pub(super) next_label: u32,
    pub(super) next_string: u32,
    /// Deduplication map for string literals: maps string content to its label.
    /// When the same string literal appears multiple times, they share the same
    /// .rodata entry (matching GCC's -fmerge-constants default behavior).
    pub(super) string_dedup: FxHashMap<String, String>,
    pub(super) next_anon_struct: u32,
    /// Counter for unique static local variable names
    pub(super) next_static_local: u32,
    pub(super) module: IrModule,
    /// Per-function build state. None between functions, Some during lowering.
    pub(super) func_state: Option<FunctionBuildState>,
    // Global variable tracking
    pub(super) globals: FxHashMap<Rc<str>, GlobalInfo>,
    // Set of known function names
    pub(super) known_functions: FxHashSet<Rc<str>>,
    // Set of already-defined function bodies
    pub(super) defined_functions: FxHashSet<Rc<str>>,
    // Set of function names declared with static linkage
    pub(super) static_functions: FxHashSet<Rc<str>>,
    /// Set of function names declared with __attribute__((error("..."))) or __attribute__((warning("..."))).
    /// Calls to these functions should be treated as unreachable (they are compile-time assertion traps).
    pub(super) error_functions: FxHashSet<Rc<str>>,
    /// Set of function names declared with __attribute__((noreturn)) or _Noreturn.
    /// After calls to these functions, emit Unreachable to avoid generating dead epilogue code.
    pub(super) noreturn_functions: FxHashSet<Rc<str>>,
    /// Set of function names declared with __attribute__((fastcall)).
    /// On i386, these use ecx/edx for the first two integer/pointer args.
    pub(super) fastcall_functions: FxHashSet<Rc<str>>,
    /// Set of function names that have at least one file-scope declaration
    /// without the `inline` specifier OR with `extern`. Per C99 6.7.4p7,
    /// an inline definition is only an "inline definition" (no external def)
    /// if ALL file-scope declarations include `inline` WITHOUT `extern`.
    /// If any declaration lacks `inline` or has `extern`, the definition
    /// provides an external definition.
    pub(super) has_non_inline_decl: FxHashSet<Rc<str>>,
    /// Type-system state (struct layouts, typedefs, enum constants, type caches)
    pub(super) types: TypeContext,
    /// Metadata about known functions (consolidated FuncSig)
    pub(super) func_meta: FunctionMeta,
    /// Set of emitted global variable names (O(1) dedup)
    pub(super) emitted_global_names: FxHashSet<Rc<str>>,
    /// Function signatures from semantic analysis.
    /// Used as authoritative source for function return types and parameter types,
    /// reducing the lowerer's need to re-derive type information from the raw AST.
    pub(super) sema_functions: FxHashMap<Rc<str>, FunctionInfo>,
    /// Expression type annotations from semantic analysis.
    /// Maps `ExprId` keys to their sema-inferred CTypes.
    /// Consulted as a fast O(1) fallback in get_expr_ctype() before the lowerer
    /// does its own (more expensive) type inference using lowering-specific state.
    pub(super) sema_expr_types: ExprTypeMap,
    /// Pre-computed constant expression values from semantic analysis.
    /// Maps `ExprId` keys to their IrConst values.
    /// Consulted as an O(1) fast path in eval_const_expr() before the lowerer
    /// falls back to its own evaluation (which handles lowering-specific cases
    /// like global addresses, func_state const locals, etc.).
    pub(super) sema_const_values: ConstMap,
    /// Stack of local label scopes from GNU __label__ declarations.
    /// Each entry maps a label name to its scope-qualified name.
    /// When resolving a label, the stack is searched top-down so that
    /// inner __label__ declarations shadow outer ones.
    pub(super) local_label_scopes: Vec<FxHashMap<Rc<str>, Rc<str>>>,
    /// Counter for generating unique local label scope IDs.
    pub(super) next_local_label_scope: u32,
    /// Maps C function/variable names to linker symbol overrides from __asm__("label").
    /// E.g., `extern int strerror_r(...) __asm__("__xpg_strerror_r")` maps
    /// "strerror_r" -> "__xpg_strerror_r". Used to redirect calls/references at IR emission.
    pub(super) asm_label_map: FxHashMap<Rc<str>, Rc<str>>,
    /// Memoization cache for get_expr_ctype().
    /// Maps `ExprId` keys to their resolved CType plus the Expr discriminant at
    /// insertion time. The discriminant is checked on cache hit to detect address
    /// reuse (ABA): expressions inside TypeSpecifier trees (typeof, _Generic) can
    /// share addresses with different Expr variants in the main AST, so a stale
    /// hit from a prior discriminant must be treated as a miss to avoid returning
    /// the wrong type.
    ///
    /// TODO: Once ExprId is backed by a counter-based scheme instead of pointer
    /// addresses, the ABA discriminant check can be removed entirely.
    pub(super) expr_ctype_cache: RefCell<FxHashMap<ExprId, (Discriminant<Expr>, Option<CType>)>>,
    /// Diagnostic engine for emitting structured errors and warnings during lowering.
    /// Threaded from the driver through the compilation pipeline so that lowering-phase
    /// diagnostics get the same source location rendering and warning control as
    /// parser/sema diagnostics.
    /// Wrapped in RefCell because many lowering methods take &self (not &mut self)
    /// but still need to emit diagnostics (same pattern as expr_ctype_cache).
    pub(super) diagnostics: RefCell<DiagnosticEngine>,
    /// Maps `*const Expr` identity of compound literal nodes to their materialized
    /// anonymous global names. Populated by `materialize_compound_literals_in_expr()`
    /// before `eval_global_addr_expr()` is called, so that compound literals
    /// embedded in pointer arithmetic (e.g., `(char*)(int[]){1,2,3} + 8`)
    /// can be resolved as global addresses.
    ///
    /// Lifetime: entries persist for the entire lowering session since the AST
    /// (and therefore Expr node addresses) lives through the full lowering pass.
    /// If the same compound literal appears in multiple globals, it will be
    /// materialized once and reused via this cache.
    ///
    /// Precondition: callers of `eval_global_addr_expr()` that may encounter
    /// compound literals in arithmetic contexts must first call
    /// `materialize_compound_literals_in_expr()` to populate this map.
    pub(super) materialized_compound_literals: FxHashMap<usize, String>,
    /// Whether GNU89 inline semantics are in effect (-fgnu89-inline, -std=c89, etc).
    /// When true, `extern inline` without `__attribute__((gnu_inline__))` is treated
    /// as an inline-only definition (no external def emitted), matching the behaviour
    /// of GCC in GNU89 mode. The default (false) uses C99 inline semantics.
    pub(super) gnu89_inline: bool,
}

impl Lowerer {
    /// Create a new Lowerer with pre-populated state from semantic analysis.
    ///
    /// The sema-provided TypeContext contains typedefs, enum constants, struct layouts,
    /// and function typedef info collected during the sema pass.
    ///
    /// The sema-provided function map contains function signatures (return types,
    /// parameter types, variadic status) collected during sema. This is used to:
    /// - Pre-populate `known_functions` so function names are recognized immediately
    /// - Pre-populate `func_return_ctypes` for function call return type resolution
    /// - Provide authoritative CType info for `get_expr_ctype` fallback
    ///
    /// The sema-provided expr_types map contains CType annotations for AST expression
    /// nodes, keyed by their pointer address. The lowerer consults this as a fast
    /// fallback in get_expr_ctype() before doing its own type inference.
    pub fn with_type_context(
        target: Target,
        type_context: TypeContext,
        sema_functions: FxHashMap<Rc<str>, FunctionInfo>,
        sema_expr_types: ExprTypeMap,
        sema_const_values: ConstMap,
        diagnostics: DiagnosticEngine,
        gnu89_inline: bool,
    ) -> Self {
        // Pre-populate known_functions from sema's function map.
        // This means the lowerer knows about all functions before the first pass,
        // which helps with early identifier resolution.
        let mut known_functions = FxHashSet::default();
        for name in sema_functions.keys() {
            known_functions.insert(name.clone());
        }

        Self {
            target,
            next_label: 0,
            next_string: 0,
            string_dedup: FxHashMap::default(),
            next_anon_struct: 0,
            next_static_local: 0,
            module: IrModule::new(),
            func_state: None,
            globals: FxHashMap::default(),
            known_functions,
            defined_functions: FxHashSet::default(),
            static_functions: FxHashSet::default(),
            error_functions: FxHashSet::default(),
            noreturn_functions: FxHashSet::default(),
            fastcall_functions: FxHashSet::default(),
            has_non_inline_decl: FxHashSet::default(),
            types: type_context,
            func_meta: FunctionMeta::default(),
            emitted_global_names: FxHashSet::default(),
            sema_functions,
            sema_expr_types,
            sema_const_values,
            local_label_scopes: Vec::new(),
            next_local_label_scope: 0,
            asm_label_map: FxHashMap::default(),
            expr_ctype_cache: RefCell::new(FxHashMap::default()),
            diagnostics: RefCell::new(diagnostics),
            materialized_compound_literals: FxHashMap::default(),
            gnu89_inline,
        }
    }

    // --- State accessors ---

    /// Access the current function build state (panics if not inside a function).
    #[inline]
    pub(super) fn func(&self) -> &FunctionBuildState {
        self.func_state.as_ref().expect("not inside a function")
    }

    /// Mutably access the current function build state (panics if not inside a function).
    #[inline]
    pub(super) fn func_mut(&mut self) -> &mut FunctionBuildState {
        self.func_state.as_mut().expect("not inside a function")
    }

    /// Returns true if the target is x86 (either x86-64 or i686).
    /// Used to select x87 80-bit extended precision format for long double memory layout.
    pub(super) fn is_x86(&self) -> bool {
        self.target == Target::X86_64 || self.target == Target::I686
    }

    /// Returns true if the target is RISC-V 64.
    /// RISC-V stores full IEEE binary128 long doubles in memory.
    pub(super) fn is_riscv(&self) -> bool {
        self.target == Target::Riscv64
    }

    /// Returns true if the target is AArch64.
    /// ARM64 stores full IEEE binary128 long doubles in memory.
    pub(super) fn is_arm(&self) -> bool {
        self.target == Target::Aarch64
    }

    /// Returns true if the function has `extern inline` semantics that do NOT
    /// emit an external definition. This covers two cases:
    ///
    ///   1. `extern inline` + `__attribute__((gnu_inline__))` (explicit GNU inline attr)
    ///   2. `extern inline` in GNU89 mode (`-std=c89`, `-fgnu89-inline`) without the attr
    ///
    /// In both cases, the function body is available for inlining but the compiler
    /// must not emit a standalone global symbol.
    pub(super) fn is_gnu_inline_no_extern_def(&self, attrs: &crate::frontend::parser::ast::FunctionAttributes) -> bool {
        attrs.is_inline() && attrs.is_extern() &&
            (attrs.is_gnu_inline() || self.gnu89_inline)
    }

    /// Returns true if the target uses x86-64 style packed _Complex float ABI
    /// (two F32s packed into a single F64/xmm register).
    /// Returns false for ARM/RISC-V which pass _Complex float as two separate F32 registers.
    /// Returns false for i686 where _Complex float is passed/returned as a struct (8 bytes
    /// in eax:edx or on the stack), not packed into an XMM register.
    pub(super) fn uses_packed_complex_float(&self) -> bool {
        self.target == Target::X86_64
    }

    /// Returns true if the target packs _Complex float into a single 8-byte value
    /// for variadic function calls. On RISC-V, variadic float args go through GP
    /// registers, so _Complex float (2xF32) must be packed into one I64 GP register.
    pub(super) fn packs_complex_float_variadic(&self) -> bool {
        self.target == Target::Riscv64
    }

    /// Returns true if the target decomposes _Complex long double into 2 F128 scalar
    /// components for function argument/parameter passing.
    /// On ARM64 (AAPCS64): _Complex long double is an HFA passed in Q0/Q1 registers,
    ///   so we decompose into 2 F128 values.
    /// On x86-64: _Complex long double is passed on the stack (MEMORY class), not decomposed.
    /// On RISC-V: _Complex long double is passed by reference (pointer), not decomposed.
    pub(super) fn decomposes_complex_long_double(&self) -> bool {
        self.target == Target::Aarch64
    }

    /// Returns true if the target returns _Complex long double via register pairs
    /// rather than via sret hidden pointer.
    /// On x86-64: true (COMPLEX_X87 class, returned in st(0) and st(1)).
    /// On ARM64: true (HFA with 2 quad-precision members, returned in q0/q1).
    /// On i686: false (24 bytes > 8-byte reg pair; uses sret hidden pointer like GCC).
    /// On RISC-V: false (returned via sret hidden pointer).
    pub(super) fn returns_complex_long_double_in_regs(&self) -> bool {
        self.target == Target::X86_64 || self.target == Target::Aarch64
    }

    /// Returns true if the target decomposes _Complex double into two separate
    /// FP scalar arguments/returns (real, imag) rather than passing as a struct.
    /// On x86-64: true (xmm0/xmm1 for return, separate XMM regs for params).
    /// On ARM64: true (d0/d1 for return, separate FP regs for params).
    /// On RISC-V: true (fa0/fa1 for return, separate FP regs for params).
    /// On i686: false (16 bytes > 8; uses sret for return, struct-on-stack for params).
    pub(super) fn decomposes_complex_double(&self) -> bool {
        self.target != Target::I686
    }

    /// Returns true if the target decomposes _Complex float into two separate
    /// FP scalar arguments rather than passing as a struct.
    /// On i686: false (8 bytes fits in eax:edx for return, passed as struct on stack).
    /// On all 64-bit targets: true (decomposed into two FP register args).
    pub(super) fn decomposes_complex_float(&self) -> bool {
        self.target != Target::I686
    }

    // --- Diagnostic helpers ---

    /// Emit a warning diagnostic with a source span.
    /// Uses RefCell for interior mutability so this can be called from &self methods.
    pub(super) fn emit_warning(&self, message: impl Into<String>, span: Span) {
        self.diagnostics.borrow_mut().warning(message, span);
    }

    /// Look up the shared type metadata for a variable by name.
    ///
    /// Checks locals first, then globals. Returns `&VarInfo` which provides
    /// access to the shared fields (ty, elem_size, is_array, pointee_type,
    /// struct_layout, is_struct, array_dim_strides, c_type).
    pub(super) fn lookup_var_info(&self, name: &str) -> Option<&VarInfo> {
        if let Some(ref fs) = self.func_state {
            if let Some(info) = fs.locals.get(name) {
                return Some(&info.var);
            }
        }
        if let Some(ginfo) = self.globals.get(name) {
            return Some(&ginfo.var);
        }
        None
    }

    /// Resolve the CType of a struct/union field, handling both direct member access
    /// (s.field) and pointer member access (p->field) through a single entry point.
    pub(super) fn resolve_field_ctype(&self, base_expr: &Expr, field_name: &str, is_pointer_access: bool) -> Option<CType> {
        self.resolve_member_field_ctype_impl(base_expr, field_name, is_pointer_access)
    }

    // --- Scope management delegation ---

    /// Push a new scope frame onto both TypeContext and FunctionBuildState scope stacks.
    /// Call this at the start of a compound statement or function body.
    pub(super) fn push_scope(&mut self) {
        self.types.push_scope();
        self.func_mut().push_scope();
    }

    /// Pop the top scope frame from both TypeContext and FunctionBuildState,
    /// undoing all scoped changes made in that scope.
    /// Emits cleanup function calls for variables with __attribute__((cleanup(func)))
    /// in reverse declaration order, then restores the stack pointer for VLAs.
    pub(super) fn pop_scope(&mut self) {
        let (scope_stack_save, cleanup_vars) = self.func_mut().pop_scope();
        self.types.pop_scope();
        // Emit cleanup calls in reverse declaration order (C semantics: last declared = first cleaned up)
        self.emit_cleanup_calls(&cleanup_vars);
        // If this scope contained VLA declarations, restore the stack pointer
        // to reclaim the dynamically-allocated stack space.
        if let Some(save_val) = scope_stack_save {
            self.emit(Instruction::StackRestore { ptr: save_val });
        }
    }

    /// Emit cleanup function calls for variables with __attribute__((cleanup(func))).
    /// Calls func(&var) for each cleanup variable, in reverse declaration order.
    pub(super) fn emit_cleanup_calls(&mut self, cleanup_vars: &[(Rc<str>, Value)]) {
        for (func_name, alloca_val) in cleanup_vars.iter().rev() {
            let dest = Some(self.fresh_value());
            self.emit(Instruction::Call {
                func: func_name.clone(),
                info: CallInfo {
                    dest,
                    args: vec![Operand::Value(*alloca_val)],
                    arg_types: vec![IrType::Ptr],
                    return_type: IrType::Void,
                    is_variadic: false,
                    num_fixed_args: 1,
                    struct_arg_sizes: vec![None],
                    struct_arg_aligns: vec![],
                    struct_arg_classes: Vec::new(),
                    struct_arg_riscv_float_classes: Vec::new(),
                    is_sret: false,
                    is_fastcall: false,
                    ret_eightbyte_classes: Vec::new(),
                },
            });
        }
    }

    /// Collect all cleanup variables from all active scopes (for return statements).
    /// Returns cleanup vars from innermost scope to outermost scope, each scope's
    /// vars in reverse declaration order.
    pub(super) fn collect_all_scope_cleanup_vars(&self) -> Vec<(Rc<str>, Value)> {
        self.collect_scope_cleanup_vars_above_depth(0)
    }

    /// Collect cleanup variables from scopes above `target_depth` (for break/continue).
    /// This collects from the innermost scope down to (but not including) the scope at
    /// target_depth, with each scope's vars in reverse declaration order.
    pub(super) fn collect_scope_cleanup_vars_above_depth(&self, target_depth: usize) -> Vec<(Rc<str>, Value)> {
        let func = self.func();
        let mut all_cleanups = Vec::new();
        // Walk scopes from innermost to outermost, stopping at target_depth
        for frame in func.scope_stack[target_depth..].iter().rev() {
            for (func_name, alloca_val) in frame.cleanup_vars.iter().rev() {
                all_cleanups.push((func_name.clone(), *alloca_val));
            }
        }
        all_cleanups
    }

    /// Remove a local variable, tracking the removal in the current scope frame.
    pub(super) fn shadow_local_for_scope(&mut self, name: &str) {
        self.func_mut().shadow_local_for_scope(name);
    }

    /// Remove a static local name, tracking the removal in the current scope frame.
    pub(super) fn shadow_static_for_scope(&mut self, name: &str) {
        self.func_mut().shadow_static_for_scope(name);
    }

    /// Insert a local variable, tracking the change in the current scope frame.
    pub(super) fn insert_local_scoped(&mut self, name: Rc<str>, info: LocalInfo) {
        self.func_mut().insert_local_scoped(name, info);
    }

    /// Insert an enum constant, tracking the change in the current scope frame.
    pub(super) fn insert_enum_scoped(&mut self, name: Rc<str>, value: i64) {
        self.types.insert_enum_scoped(name, value);
    }

    /// Insert a static local name, tracking the change in the current scope frame.
    pub(super) fn insert_static_local_scoped(&mut self, name: Rc<str>, mangled: Rc<str>) {
        self.func_mut().insert_static_local_scoped(name, mangled);
    }

    /// Insert a const local value, tracking the change in the current scope frame.
    pub(super) fn insert_const_local_scoped(&mut self, name: Rc<str>, value: i64) {
        self.func_mut().insert_const_local_scoped(name, value);
    }

    // --- Top-level orchestration ---

    pub fn lower(mut self, tu: &TranslationUnit) -> (IrModule, DiagnosticEngine) {
        // Sema has already populated TypeContext with typedefs, enum constants,
        // struct/union layouts, function typedefs, and function pointer typedefs.
        // We only need to seed target-dependent builtin typedefs (va_list, size_t, etc.)
        // and libc math function signatures that sema doesn't know about.
        self.seed_builtin_typedefs();
        self.seed_libc_math_functions();

        // Mark transparent_union on union StructLayouts before the first pass,
        // so that register_function_meta can exclude them from param_struct_sizes.
        for decl in &tu.decls {
            if let ExternalDecl::Declaration(decl) = decl {
                if decl.is_transparent_union() {
                    self.mark_transparent_union(decl);
                }
            }
        }

        // Pre-pass: update typedefs that have __attribute__((vector_size(N))) or
        // __attribute__((ext_vector_type(N))).
        // Sema doesn't handle vector_size, so the typedef map has the unwrapped base type.
        // We need to apply vector_size wrapping before registering function metadata,
        // so that function return types using vector typedefs are correctly resolved.
        let mut has_vector_typedefs = false;
        for decl in &tu.decls {
            if let ExternalDecl::Declaration(decl) = decl {
                if decl.is_typedef() {
                    for declarator in &decl.declarators {
                        if !declarator.name.is_empty() {
                            let base_ctype = self.build_full_ctype(&decl.type_spec, &declarator.derived);
                            let elem_size = base_ctype.size();
                            if let Some(vs) = decl.resolve_vector_size(elem_size) {
                                has_vector_typedefs = true;
                                let vec_ctype = CType::Vector(Box::new(base_ctype), vs);
                                self.types.typedefs.insert(declarator.name.clone(), vec_ctype);
                            }
                        }
                    }
                }
            }
        }

        // After updating vector typedefs, re-compute struct/union layouts that may
        // contain vector typedef fields. Sema computed these layouts before vector_size
        // was applied, so their field sizes may be wrong (e.g., float4 was treated as
        // float instead of vector(float, 16)). We must update the EXISTING layout
        // entries (by key) rather than creating new ones, because CType::Union/Struct
        // references use the key assigned by sema.
        if has_vector_typedefs {
            self.recompute_vector_struct_layouts(tu);
        }

        // Pre-pass: register struct/union layouts from all declarations and function
        // definitions so that function signature registration (below) can compute
        // correct SysV eightbyte classification for struct return types.
        // Without this, structs/unions defined via typedef before a function that
        // returns them would not have their layout available during classify_sysv_eightbytes.
        for decl in &tu.decls {
            match decl {
                ExternalDecl::Declaration(decl) => {
                    self.register_struct_type(&decl.type_spec);
                }
                ExternalDecl::FunctionDef(func) => {
                    self.register_struct_type(&func.return_type);
                    for p in &func.params {
                        self.register_struct_type(&p.type_spec);
                    }
                }
                ExternalDecl::TopLevelAsm(_) => {}
            }
        }

        // First pass: collect all function signatures (return types, param types,
        // variadic status, sret) so we can distinguish functions from globals and
        // insert proper casts/ABI handling during lowering.
        for decl in &tu.decls {
            if let ExternalDecl::FunctionDef(func) = decl {
                self.register_function_meta(
                    &func.name, &func.return_type, 0,
                    &func.params, func.variadic, func.attrs.is_static(), func.is_kr,
                );
            }
            if let ExternalDecl::Declaration(decl) = decl {
                for declarator in &decl.declarators {
                    // Collect __asm__("label") linker symbol redirects on function declarations.
                    // When a function declaration has __asm__("symbol"), calls to that function
                    // should emit the asm label as the linker symbol instead of the C name.
                    // E.g.: extern int strerror_r(...) __asm__("__xpg_strerror_r");
                    // Note: register variables also use asm_register for register pinning
                    // (e.g., `register int x __asm__("rbx")`), which is handled separately
                    // in lower_global_decl. We only redirect function declarations here.
                    if let Some(ref asm_label) = declarator.attrs.asm_register {
                        let is_function_decl = declarator.derived.iter().any(|d|
                            matches!(d, DerivedDeclarator::Function(_, _))
                        );
                        if is_function_decl {
                            self.asm_label_map.insert(
                                declarator.name.clone(),
                                Rc::from(asm_label.as_str()),
                            );
                        }
                    }

                    // Find the Function derived declarator and count preceding Pointer derivations
                    let mut ptr_count = 0;
                    let mut func_info = None;
                    for d in &declarator.derived {
                        match d {
                            DerivedDeclarator::Pointer => ptr_count += 1,
                            DerivedDeclarator::Function(p, v) => {
                                func_info = Some((p.clone(), *v));
                                break;
                            }
                            _ => {}
                        }
                    }
                    if let Some((params, variadic)) = func_info {
                        self.register_function_meta(
                            &declarator.name, &decl.type_spec, ptr_count,
                            &params, variadic, decl.is_static(), false,
                        );
                        // C99 6.7.4p7: Track function declarations that would make
                        // an inline definition provide an external definition.
                        // An inline definition is "inline only" (no external def) ONLY
                        // if ALL file-scope declarations include `inline` WITHOUT `extern`.
                        // So: if any declaration lacks `inline` OR has `extern`, mark it.
                        if !declarator.name.is_empty()
                            && (!decl.is_inline() || decl.is_extern())
                        {
                            self.has_non_inline_decl.insert(declarator.name.clone());
                        }
                    } else if declarator.derived.is_empty() {
                        // Check if the base type is a function typedef
                        // (e.g., `func_t add;` where func_t is typedef int func_t(int);)
                        // Only when derived is empty — `func_t *callback;` is a variable,
                        // not a function declaration.
                        if let TypeSpecifier::TypedefName(tname) = &decl.type_spec {
                            if let Some(fti) = self.types.function_typedefs.get(tname).cloned() {
                                self.register_function_meta(
                                    &declarator.name, &fti.return_type, 0,
                                    &fti.params, fti.variadic, false, false,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Collect constructor/destructor/alias attributes from function definitions and declarations
        for decl in &tu.decls {
            match decl {
                ExternalDecl::FunctionDef(func) => {
                    if func.attrs.is_constructor() && !self.module.constructors.iter().any(|c| **c == *func.name) {
                        self.module.constructors.push(func.name.clone());
                    }
                    if func.attrs.is_destructor() && !self.module.destructors.iter().any(|c| **c == *func.name) {
                        self.module.destructors.push(func.name.clone());
                    }
                    if func.attrs.is_fastcall() {
                        self.fastcall_functions.insert(func.name.clone());
                    }
                }
                ExternalDecl::Declaration(decl) => {
                    for declarator in &decl.declarators {
                        if declarator.attrs.is_constructor() && !declarator.name.is_empty()
                            && !self.module.constructors.iter().any(|c| **c == *declarator.name)
                        {
                            self.module.constructors.push(declarator.name.clone());
                        }
                        if declarator.attrs.is_destructor() && !declarator.name.is_empty()
                            && !self.module.destructors.iter().any(|c| **c == *declarator.name)
                        {
                            self.module.destructors.push(declarator.name.clone());
                        }
                        // Collect __attribute__((alias("target"))) declarations
                        if let Some(ref target) = declarator.attrs.alias_target {
                            if !declarator.name.is_empty() {
                                self.module.aliases.push((
                                    declarator.name.clone(),
                                    Rc::from(target.as_str()),  // alias_target is String
                                    declarator.attrs.is_weak(),
                                ));
                            }
                        }
                        // Collect __attribute__((symver("..."))) declarations
                        if let Some(ref sv) = declarator.attrs.symver {
                            if !declarator.name.is_empty() {
                                self.module.symver_directives.push((
                                    declarator.name.clone(),
                                    Rc::from(sv.as_str()),  // symver is String
                                ));
                            }
                        }
                        // Collect __attribute__((error("..."))) / __attribute__((warning("...")))
                        if declarator.attrs.is_error_attr() && !declarator.name.is_empty() {
                            self.error_functions.insert(declarator.name.clone());
                        }
                        // Collect __attribute__((noreturn)) / _Noreturn
                        if declarator.attrs.is_noreturn() && !declarator.name.is_empty() {
                            self.noreturn_functions.insert(declarator.name.clone());
                        }
                        // Collect __attribute__((fastcall))
                        if declarator.attrs.is_fastcall() && !declarator.name.is_empty() {
                            self.fastcall_functions.insert(declarator.name.clone());
                        }
                        // Collect weak/visibility attributes on extern declarations (not aliases).
                        // Skip typedefs: they are not linker symbols and should never get
                        // .weak/.hidden directives. This matters when #pragma GCC visibility
                        // push(hidden) is active (e.g., kernel EFI stub) because it would
                        // otherwise emit .hidden for thousands of typedef names.
                        if !decl.is_typedef()
                            && declarator.attrs.alias_target.is_none()
                            && !declarator.name.is_empty()
                            && (declarator.attrs.is_weak() || declarator.attrs.visibility.is_some())
                        {
                            self.module.symbol_attrs.push((
                                declarator.name.clone(),
                                declarator.attrs.is_weak(),
                                declarator.attrs.visibility.clone(),
                            ));
                        }
                    }
                }
                ExternalDecl::TopLevelAsm(_) => {
                    // Handled in the third pass
                }
            }
        }

        // Pass 2.5: collect referenced static functions so we can skip unreferenced ones.
        // Static/inline functions from headers that are never called don't need to be lowered.
        let referenced_statics = self.collect_referenced_static_functions(tu);

        // Third pass: lower everything
        for decl in tu.decls.iter() {
            match decl {
                ExternalDecl::FunctionDef(func) => {
                    // Skip unreferenced static/static-inline functions (e.g., static inline
                    // from headers that are never called).
                    //
                    // GNU89/gnu_inline semantics (explicit __attribute__((gnu_inline)) OR
                    // -fgnu89-inline / -std=c89 mode):
                    //   extern inline = inline definition only, no external def
                    //   → treat like static inline (can skip if unreferenced, local if emitted)
                    //   inline (no extern) = external definition (must emit, global)
                    //
                    // C99 semantics (default for -std=c99 and later):
                    //   extern inline = provides external definition (must emit, global)
                    //   inline (alone) = inline definition only (no external def)
                    let is_gnu_inline_no_extern_def = self.is_gnu_inline_no_extern_def(&func.attrs);
                    // C99 6.7.4p7: A plain `inline` definition (without `extern`)
                    // does not provide an external definition. We lower these as
                    // static functions so their bodies are available for inlining.
                    // If unreferenced, they are skipped. If referenced but all calls
                    // are inlined, dead code elimination removes them.
                    // Note: this only applies in C99 mode; in GNU89 mode, `inline`
                    // without `extern` provides the external definition.
                    let is_c99_inline_only = !self.gnu89_inline
                        && func.attrs.is_inline() && !func.attrs.is_extern()
                        && !func.attrs.is_static() && !func.attrs.is_gnu_inline()
                        && !self.has_non_inline_decl.contains(&func.name);
                    let can_skip = if is_c99_inline_only {
                        // C99 inline-only functions don't provide an external definition.
                        // Lower them as static so their bodies are available for inlining.
                        // If all call sites are inlined, dead code elimination removes them.
                        // If not inlined, they're emitted as local symbols (safe fallback).
                        true
                    } else if func.attrs.is_static() {
                        // static or static inline: internal linkage, safe to skip if unreferenced
                        true
                    } else if is_gnu_inline_no_extern_def {
                        // extern inline (gnu89 or gnu_inline attr): no external def, skip if unreferenced
                        true
                    } else {
                        false
                    };
                    if can_skip && !func.attrs.is_used()
                        && !func.attrs.is_constructor() && !self.module.constructors.iter().any(|c| **c == *func.name)
                        && !func.attrs.is_destructor() && !self.module.destructors.iter().any(|c| **c == *func.name)
                        && !referenced_statics.contains(&func.name) {
                        continue;
                    }
                    self.lower_function(func);
                }
                ExternalDecl::Declaration(decl) => {
                    self.lower_global_decl(decl);
                }
                ExternalDecl::TopLevelAsm(asm_str) => {
                    self.module.toplevel_asm.push(asm_str.clone());
                }
            }
        }
        (self.module, self.diagnostics.into_inner())
    }

    /// Register function metadata (return type, param types, variadic, sret) for
    /// a function name. This shared helper eliminates the triplicated pattern in `lower()`
    /// where function definitions, extern declarations, and typedef-based declarations
    /// all needed to register the same metadata fields.
    ///
    /// When sema has already collected the function's CType info (via FunctionInfo),
    /// this method uses sema's return CType as source-of-truth instead of re-computing
    /// it from the AST TypeSpecifier. This reduces duplicated work and establishes
    /// sema as the authority on function type information.
    fn register_function_meta(
        &mut self,
        name: &Rc<str>,
        ret_type_spec: &TypeSpecifier,
        ptr_count: usize,
        params: &[ParamDecl],
        variadic: bool,
        is_static: bool,
        is_kr: bool,
    ) {
        self.known_functions.insert(name.clone());
        if is_static {
            self.static_functions.insert(name.clone());
        }

        // Compute the return CType once. Prefer sema's authoritative CType if available,
        // falling back to re-computing from the AST TypeSpecifier.
        // When sema provides the return type, it already includes pointer levels from
        // the declarator chain, so we must NOT add ptr_count layers again.
        // Only the AST fallback path needs ptr_count wrapping, since type_spec_to_ctype
        // only resolves the base type specifier without pointer derivators.
        let full_ret_ctype = if let Some(func_info) = self.sema_functions.get(name) {
            let sema_ct = func_info.return_type.clone();
            // Sema doesn't resolve vector_size attributes, so if the lowerer's
            // type_spec_to_ctype produces a Vector but sema doesn't, prefer the
            // lowerer's result which correctly includes vector information.
            if !sema_ct.is_vector() && ptr_count == 0 {
                let lowerer_ct = self.type_spec_to_ctype(ret_type_spec);
                if lowerer_ct.is_vector() {
                    lowerer_ct
                } else {
                    sema_ct
                }
            } else {
                sema_ct
            }
        } else {
            let mut ct = self.type_spec_to_ctype(ret_type_spec);
            for _ in 0..ptr_count {
                ct = CType::Pointer(Box::new(ct), AddressSpace::Default);
            }
            ct
        };

        // Compute return type, wrapping with pointer levels if needed
        let mut ret_ty = IrType::from_ctype(&full_ret_ctype);
        // Complex return types need special IR type overrides based on target ABI:
        //
        // x86-64:
        //   _Complex double: real in xmm0 (F64), imag in xmm1 -> IR ret type F64
        //   _Complex float: packed two F32 in xmm0 as F64 -> IR ret type F64
        //   _Complex long double: real in x87 st(0) -> IR ret type F128
        //
        // ARM64/RISC-V:
        //   _Complex double: real in d0 (F64), imag in d1 -> IR ret type F64
        //   _Complex float: real in s0 (F32), imag in s1 -> IR ret type F32
        //
        // i686:
        //   _Complex float (8 bytes): packed in eax:edx -> IR ret type I64
        //   _Complex double (16 bytes): sret hidden pointer (ret_ty stays Ptr)
        //   _Complex long double (24 bytes): sret hidden pointer (ret_ty stays Ptr)
        if ptr_count == 0 {
            if matches!(full_ret_ctype, CType::ComplexLongDouble) && self.returns_complex_long_double_in_regs() {
                // x86-64: real part in x87 st(0); ARM64: real part in q0
                ret_ty = IrType::F128;
            } else if matches!(full_ret_ctype, CType::ComplexDouble) && self.decomposes_complex_double() {
                // x86-64/ARM64/RISC-V: _Complex double returns real in FP reg
                ret_ty = IrType::F64;
            } else if matches!(full_ret_ctype, CType::ComplexFloat) {
                if self.uses_packed_complex_float() {
                    // x86-64: packed two F32 in one xmm register as F64
                    ret_ty = IrType::F64;
                } else if self.decomposes_complex_float() {
                    // ARM64/RISC-V: real in first FP register as F32
                    ret_ty = IrType::F32;
                } else {
                    // i686: _Complex float (8 bytes) fits in eax:edx, pack as I64
                    ret_ty = IrType::I64;
                }
            }
            // i686: ComplexDouble (16 bytes) and ComplexLongDouble (24 bytes) keep
            // ret_ty = IrType::Ptr (the default from_ctype mapping for complex types).
            // They will be handled via sret below.
        }

        // Track CType for pointer-returning functions
        let return_ctype = if ret_ty == IrType::Ptr {
            Some(full_ret_ctype.clone())
        } else {
            None
        };

        // Record complex return types for expr_ctype resolution
        if ptr_count == 0 && full_ret_ctype.is_complex() {
            self.types.func_return_ctypes.insert(name.clone(), full_ret_ctype.clone());
        }

        // Detect struct/complex/vector returns that need special ABI handling.
        let mut sret_size = None;
        let mut two_reg_ret_size = None;
        if ptr_count == 0 {
            if full_ret_ctype.is_struct_or_union() {
                let size = self.sizeof_type(ret_type_spec);
                let (s, t) = Self::classify_struct_return(size);
                sret_size = s;
                two_reg_ret_size = t;
            }
            // Vector types use the same by-value return convention as structs:
            // small vectors (<=8 bytes) are packed into a register, larger ones use sret.
            if full_ret_ctype.is_vector() {
                let size = self.sizeof_type(ret_type_spec);
                let (s, t) = Self::classify_struct_return(size);
                sret_size = s;
                two_reg_ret_size = t;
                // Small vector returns (<=8 bytes, no sret/two_reg) need I64 return type
                // so the packed vector data is returned in a register, not as a pointer.
                if s.is_none() && t.is_none() {
                    ret_ty = IrType::I64;
                }
            }
            if matches!(full_ret_ctype, CType::ComplexLongDouble) {
                // On x86-64, _Complex long double is returned via x87 st(0)/st(1),
                // not via hidden pointer (sret). On i686 and other targets, use sret.
                if !self.returns_complex_long_double_in_regs() {
                    let size = self.sizeof_type(ret_type_spec);
                    sret_size = Some(size);
                }
            }
            // On i686, _Complex double (16 bytes) exceeds the 8-byte register pair
            // (eax:edx), so it must use sret (hidden pointer return).
            // On 64-bit targets, _Complex double is returned in two FP registers.
            if matches!(full_ret_ctype, CType::ComplexDouble) && !self.decomposes_complex_double() {
                let size = full_ret_ctype.size();
                sret_size = Some(size);
            }
        }

        // Compute SysV eightbyte classification for two-register struct returns
        let ret_eightbyte_classes = if two_reg_ret_size.is_some() && ptr_count == 0 {
            if full_ret_ctype.is_struct_or_union() || full_ret_ctype.is_vector() {
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

        // Collect parameter types, with K&R default argument promotions.
        // Use sema's param CTypes when available to avoid re-computing from AST.
        let sema_param_ctypes = self.sema_functions.get(name)
            .map(|fi| fi.params.iter().map(|(ct, _)| ct.clone()).collect::<Vec<_>>());

        let param_tys: Vec<IrType> = params.iter().enumerate().map(|(i, p)| {
            let ty = if let Some(ref sema_cts) = sema_param_ctypes {
                if let Some(ct) = sema_cts.get(i) {
                    IrType::from_ctype(ct)
                } else {
                    self.type_spec_to_ir(&p.type_spec)
                }
            } else {
                self.type_spec_to_ir(&p.type_spec)
            };
            if is_kr {
                match ty {
                    IrType::F32 => IrType::F64,
                    IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 => IrType::I32,
                    other => other,
                }
            } else { ty }
        }).collect();
        let param_bool_flags: Vec<bool> = params.iter().map(|p| {
            self.is_type_bool(&p.type_spec)
        }).collect();
        // Collect parameter CTypes for complex argument conversion.
        // Prefer sema's authoritative CTypes when available.
        let param_ctypes: Vec<CType> = if let Some(sema_cts) = sema_param_ctypes {
            params.iter().enumerate().map(|(i, p)| {
                if let Some(ct) = sema_cts.get(i) {
                    ct.clone()
                } else {
                    self.type_spec_to_ctype(&p.type_spec)
                }
            }).collect()
        } else {
            params.iter().map(|p| {
                self.type_spec_to_ctype(&p.type_spec)
            }).collect()
        };

        // Collect per-parameter struct sizes for by-value struct passing ABI.
        // ComplexLongDouble is included as a struct on platforms that don't decompose it
        // (x86-64, RISC-V) since it's passed like a struct (on stack / by reference).
        // On i686, ComplexDouble (16 bytes) and ComplexFloat (8 bytes) are also passed
        // as structs on the stack since they aren't decomposed into separate FP args.
        // Transparent unions are excluded — they are passed as their first member.
        let decomposes_cld = self.decomposes_complex_long_double();
        let decomposes_cd = self.decomposes_complex_double();
        let decomposes_cf = self.decomposes_complex_float();
        let param_struct_sizes: Vec<Option<usize>> = params.iter().map(|p| {
            let ctype = self.type_spec_to_ctype(&p.type_spec);
            if self.is_type_struct_or_union(&p.type_spec) && !self.is_transparent_union(&p.type_spec) {
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
        self.func_meta.sigs.insert(name.clone(), sig);
    }

    // --- IR emission helpers ---

    pub(super) fn fresh_value(&mut self) -> Value {
        let v = Value(self.func_mut().next_value);
        self.func_mut().next_value += 1;
        v
    }

    pub(super) fn fresh_label(&mut self) -> BlockId {
        let l = BlockId(self.next_label);
        self.next_label += 1;
        l
    }

    /// Intern a string literal: add it to the module's .rodata string table and
    /// return its unique label. Deduplicates identical strings so they share
    /// the same .rodata entry (matching GCC's -fmerge-constants behavior).
    pub(super) fn intern_string_literal(&mut self, s: &str) -> String {
        if let Some(existing_label) = self.string_dedup.get(s) {
            return existing_label.clone();
        }
        let label = format!(".Lstr{}", self.next_string);
        self.next_string += 1;
        self.string_dedup.insert(s.to_string(), label.clone());
        self.module.string_literals.push((label.clone(), s.to_string()));
        label
    }

    /// Intern a wide string literal (L"...") and return its label.
    /// Each character is stored as a u32 (wchar_t), plus a null terminator.
    pub(super) fn intern_wide_string_literal(&mut self, s: &str) -> String {
        let label = format!(".Lwstr{}", self.next_string);
        self.next_string += 1;
        let mut chars: Vec<u32> = s.chars().map(|c| c as u32).collect();
        chars.push(0); // null terminator
        self.module.wide_string_literals.push((label.clone(), chars));
        label
    }

    /// Intern a char16_t string literal (u"...") and return its label.
    /// Each character is stored as a u16 (char16_t), plus a null terminator.
    pub(super) fn intern_char16_string_literal(&mut self, s: &str) -> String {
        let label = format!(".Lc16str{}", self.next_string);
        self.next_string += 1;
        let mut chars: Vec<u16> = s.chars().map(|c| c as u16).collect();
        chars.push(0); // null terminator
        self.module.char16_string_literals.push((label.clone(), chars));
        label
    }

    pub(super) fn emit(&mut self, inst: Instruction) {
        let span = self.func_mut().current_span;
        self.func_mut().instrs.push(inst);
        self.func_mut().instr_spans.push(span);
    }

    /// Emit an alloca into the entry block buffer.
    /// Used for local variable declarations so that variables whose
    /// declarations are skipped by `goto` still have valid stack slots.
    pub(super) fn emit_entry_alloca(&mut self, ty: IrType, size: usize, align: usize, volatile: bool) -> Value {
        let dest = self.fresh_value();
        self.func_mut().entry_allocas.push(Instruction::Alloca {
            dest, ty, size, align, volatile,
        });
        dest
    }

    /// Emit a binary operation and return the result Value.
    pub(super) fn emit_binop_val(&mut self, op: IrBinOp, lhs: Operand, rhs: Operand, ty: IrType) -> Value {
        let dest = self.fresh_value();
        self.emit(Instruction::BinOp { dest, op, lhs, rhs, ty });
        dest
    }

    /// Emit a comparison and return the result Value (I32: 0 or 1).
    pub(super) fn emit_cmp_val(&mut self, op: IrCmpOp, lhs: Operand, rhs: Operand, ty: IrType) -> Value {
        let dest = self.fresh_value();
        self.emit(Instruction::Cmp { dest, op, lhs, rhs, ty });
        dest
    }

    /// Emit a type cast and return the result Value.
    pub(super) fn emit_cast_val(&mut self, src: Operand, from_ty: IrType, to_ty: IrType) -> Value {
        let dest = self.fresh_value();
        self.emit(Instruction::Cast { dest, src, from_ty, to_ty });
        dest
    }

    /// Emit a GEP + Store: store `val` at `base + byte_offset` with the given type.
    pub(super) fn emit_store_at_offset(&mut self, base: Value, byte_offset: usize, val: Operand, ty: IrType) {
        let addr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: addr,
            base,
            offset: Operand::Const(IrConst::ptr_int(byte_offset as i64)),
            ty,
        });
        self.emit(Instruction::Store { val, ptr: addr, ty , seg_override: AddressSpace::Default });
    }

    /// Lower an expression, cast to target type, then store at base + byte_offset.
    /// When target_is_bool is true, normalizes the value (any nonzero -> 1) per C11 6.3.1.2.
    pub(super) fn emit_init_expr_to_offset_bool(&mut self, e: &Expr, base: Value, byte_offset: usize, target_ty: IrType, target_is_bool: bool) {
        let expr_ty = self.get_expr_type(e);
        let val = self.lower_expr(e);
        let val = if target_is_bool {
            self.emit_bool_normalize_typed(val, expr_ty)
        } else {
            self.emit_implicit_cast(val, expr_ty, target_ty)
        };
        self.emit_store_at_offset(base, byte_offset, val, target_ty);
    }

    /// Emit a GEP to compute base + byte_offset and return the address Value.
    pub(super) fn emit_gep_offset(&mut self, base: Value, byte_offset: usize, ty: IrType) -> Value {
        let addr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: addr,
            base,
            offset: Operand::Const(IrConst::ptr_int(byte_offset as i64)),
            ty,
        });
        addr
    }

    /// Emit memcpy from src to base + byte_offset.
    pub(super) fn emit_memcpy_at_offset(&mut self, base: Value, byte_offset: usize, src: Value, size: usize) {
        let dest = self.emit_gep_offset(base, byte_offset, IrType::Ptr);
        self.emit(Instruction::Memcpy { dest, src, size });
    }

    pub(super) fn terminate(&mut self, term: Terminator) {
        let block = BasicBlock {
            label: self.func_mut().current_label,
            instructions: std::mem::take(&mut self.func_mut().instrs),
            source_spans: std::mem::take(&mut self.func_mut().instr_spans),
            terminator: term,
        };
        self.func_mut().blocks.push(block);
    }

    pub(super) fn start_block(&mut self, label: BlockId) {
        self.func_mut().current_label = label;
        self.func_mut().instrs.clear();
        self.func_mut().instr_spans.clear();
    }

    // --- Enum and label helpers ---

    /// Collect enum constants from a type specifier.
    fn collect_enum_constants_impl(&mut self, ts: &TypeSpecifier, scoped: bool) {
        match ts {
            TypeSpecifier::Enum(name, Some(variants), is_packed) => {
                let mut next_val: i64 = 0;
                let mut variant_values = Vec::new();
                for variant in variants {
                    if let Some(ref expr) = variant.value {
                        if let Some(val) = self.eval_const_expr(expr) {
                            if let Some(v) = self.const_to_i64(&val) {
                                next_val = v;
                            }
                        }
                    }
                    if scoped {
                        self.insert_enum_scoped(variant.name.clone(), next_val);
                    } else {
                        self.types.enum_constants.insert(variant.name.clone(), next_val);
                    }
                    variant_values.push((variant.name.clone(), next_val));
                    next_val += 1;
                }
                // Store packed enum info for forward-reference lookups
                if *is_packed {
                    if let Some(tag) = name {
                        self.types.packed_enum_types.insert(
                            tag.clone(),
                            crate::common::types::EnumType {
                                name: Some(tag.clone()),
                                variants: variant_values,
                                is_packed: true,
                            },
                        );
                    }
                }
            }
            TypeSpecifier::Struct(_, Some(fields), _, _, _) | TypeSpecifier::Union(_, Some(fields), _, _, _) => {
                for field in fields {
                    self.collect_enum_constants_impl(&field.type_spec, scoped);
                }
            }
            TypeSpecifier::Array(inner, _) | TypeSpecifier::Pointer(inner, _) => {
                self.collect_enum_constants_impl(inner, scoped);
            }
            _ => {}
        }
    }

    /// Collect enum constants from a type specifier (file-scope, direct insertion).
    pub(super) fn collect_enum_constants(&mut self, ts: &TypeSpecifier) {
        self.collect_enum_constants_impl(ts, false);
    }

    /// Collect enum constants from a type specifier, using scoped insertion.
    pub(super) fn collect_enum_constants_scoped(&mut self, ts: &TypeSpecifier) {
        self.collect_enum_constants_impl(ts, true);
    }

    /// Check if a TypeSpecifier refers to an enum type (directly or via typedef).
    pub(super) fn is_enum_type_spec(&self, ts: &TypeSpecifier) -> bool {
        match ts {
            TypeSpecifier::Enum(..) => true,
            TypeSpecifier::TypedefName(name) => self.types.enum_typedefs.contains(name),
            TypeSpecifier::TypeofType(inner) => self.is_enum_type_spec(inner),
            _ => false,
        }
    }

    /// Check if a user-defined goto label has already been defined (i.e., the
    /// label statement was already lowered). Used to determine backward vs forward gotos.
    /// This checks `defined_user_labels` (set when `label:` is lowered), NOT `user_labels`
    /// (which is also populated by forward `goto` statements creating placeholder blocks).
    pub(super) fn user_label_exists(&self, name: &str) -> bool {
        let resolved_name = self.resolve_local_label(name);
        let func_name = &self.func().name;
        let key: Rc<str> = Rc::from(format!("{}::{}", func_name, resolved_name));
        self.func().defined_user_labels.contains(&key)
    }

    /// Get or create a unique IR label for a user-defined goto label.
    /// If the label name is declared via __label__ in a local scope,
    /// uses the scope-qualified name so that different invocations of
    /// a macro with __label__ don't collide.
    pub(super) fn get_or_create_user_label(&mut self, name: &str) -> BlockId {
        // Check local label scopes from innermost to outermost
        let resolved_name = self.resolve_local_label(name);
        let func_name = self.func_mut().name.clone();
        let key: Rc<str> = Rc::from(format!("{}::{}", func_name, resolved_name));
        if let Some(&label) = self.func_mut().user_labels.get(&key) {
            label
        } else {
            let label = self.fresh_label();
            self.func_mut().user_labels.insert(key, label);
            label
        }
    }

    /// Resolve a label name through the local label scope stack.
    /// Returns a scope-qualified name if the label is declared via __label__,
    /// or the original name if not.
    pub(super) fn resolve_local_label(&self, name: &str) -> Rc<str> {
        // Search scopes from innermost to outermost
        for scope in self.local_label_scopes.iter().rev() {
            if let Some(qualified) = scope.get(name) {
                return qualified.clone();
            }
        }
        Rc::from(name)
    }

    // --- String and array init helpers ---

    /// Copy a string literal's bytes into an alloca at a given byte offset,
    /// followed by a null terminator (if it fits within `max_bytes`).
    ///
    /// Per C11 6.7.9 p14, when a char array is initialized from a string literal
    /// that is exactly one byte longer than the array (due to the implicit NUL
    /// terminator), the trailing NUL is silently dropped. This function respects
    /// that rule by only writing up to `max_bytes` total bytes.
    pub(super) fn emit_string_to_alloca(&mut self, alloca: Value, s: &str, base_offset: usize, max_bytes: usize) {
        let str_bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
        let bytes_to_copy = str_bytes.len().min(max_bytes);
        for j in 0..bytes_to_copy {
            let byte = str_bytes[j];
            let val = Operand::Const(IrConst::I8(byte as i8));
            let offset = Operand::Const(IrConst::ptr_int((base_offset + j) as i64));
            let addr = self.fresh_value();
            self.emit(Instruction::GetElementPtr {
                dest: addr, base: alloca, offset, ty: IrType::I8,
            });
            self.emit(Instruction::Store { val, ptr: addr, ty: IrType::I8 , seg_override: AddressSpace::Default });
        }
        // Null terminator -- only write if there's room within max_bytes
        let null_pos = str_bytes.len();
        if null_pos < max_bytes {
            let null_offset = Operand::Const(IrConst::ptr_int((base_offset + null_pos) as i64));
            let null_addr = self.fresh_value();
            self.emit(Instruction::GetElementPtr {
                dest: null_addr, base: alloca, offset: null_offset, ty: IrType::I8,
            });
            self.emit(Instruction::Store { val: Operand::Const(IrConst::I8(0)), ptr: null_addr, ty: IrType::I8,
             seg_override: AddressSpace::Default });
        }
    }

    /// Emit a wide string (L"...") to a local alloca. Each character is stored as I32 (wchar_t).
    pub(super) fn emit_wide_string_to_alloca(&mut self, alloca: Value, s: &str, base_offset: usize) {
        for (j, ch) in s.chars().enumerate() {
            let val = Operand::Const(IrConst::I32(ch as i32));
            let byte_offset = base_offset + j * 4;
            let offset = Operand::Const(IrConst::ptr_int(byte_offset as i64));
            let addr = self.fresh_value();
            self.emit(Instruction::GetElementPtr {
                dest: addr, base: alloca, offset, ty: IrType::I8,
            });
            self.emit(Instruction::Store { val, ptr: addr, ty: IrType::I32 , seg_override: AddressSpace::Default });
        }
        // Null terminator
        let null_byte_offset = base_offset + s.chars().count() * 4;
        let null_offset = Operand::Const(IrConst::ptr_int(null_byte_offset as i64));
        let null_addr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: null_addr, base: alloca, offset: null_offset, ty: IrType::I8,
        });
        self.emit(Instruction::Store { val: Operand::Const(IrConst::I32(0)), ptr: null_addr, ty: IrType::I32,
         seg_override: AddressSpace::Default });
    }

    /// Emit a char16_t string (u"...") to a local alloca. Each character is stored as U16.
    pub(super) fn emit_char16_string_to_alloca(&mut self, alloca: Value, s: &str, base_offset: usize) {
        for (j, ch) in s.chars().enumerate() {
            let val = Operand::Const(IrConst::I16(ch as u16 as i16));
            let byte_offset = base_offset + j * 2;
            let offset = Operand::Const(IrConst::ptr_int(byte_offset as i64));
            let addr = self.fresh_value();
            self.emit(Instruction::GetElementPtr {
                dest: addr, base: alloca, offset, ty: IrType::I8,
            });
            self.emit(Instruction::Store { val, ptr: addr, ty: IrType::U16, seg_override: AddressSpace::Default });
        }
        // Null terminator
        let null_byte_offset = base_offset + s.chars().count() * 2;
        let null_offset = Operand::Const(IrConst::ptr_int(null_byte_offset as i64));
        let null_addr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: null_addr, base: alloca, offset: null_offset, ty: IrType::I8,
        });
        self.emit(Instruction::Store { val: Operand::Const(IrConst::I16(0)), ptr: null_addr, ty: IrType::U16,
         seg_override: AddressSpace::Default });
    }

    /// Emit a single element store at a given byte offset in an alloca.
    pub(super) fn emit_array_element_store(
        &mut self, alloca: Value, val: Operand, offset: usize, ty: IrType,
    ) {
        let offset_val = Operand::Const(IrConst::ptr_int(offset as i64));
        let elem_addr = self.fresh_value();
        self.emit(Instruction::GetElementPtr {
            dest: elem_addr, base: alloca, offset: offset_val, ty,
        });
        self.emit(Instruction::Store { val, ptr: elem_addr, ty , seg_override: AddressSpace::Default });
    }

    /// Zero-initialize a region of memory within an alloca at the given byte offset.
    pub(super) fn zero_init_region(&mut self, alloca: Value, base_offset: usize, region_size: usize) {
        let mut offset = base_offset;
        let end = base_offset + region_size;
        while offset + 8 <= end {
            let addr = self.fresh_value();
            self.emit(Instruction::GetElementPtr {
                dest: addr,
                base: alloca,
                offset: Operand::Const(IrConst::ptr_int(offset as i64)),
                ty: IrType::I64,
            });
            self.emit(Instruction::Store { val: Operand::Const(IrConst::I64(0)), ptr: addr, ty: IrType::I64,
             seg_override: AddressSpace::Default });
            offset += 8;
        }
        while offset < end {
            let addr = self.fresh_value();
            self.emit(Instruction::GetElementPtr {
                dest: addr,
                base: alloca,
                offset: Operand::Const(IrConst::ptr_int(offset as i64)),
                ty: IrType::I8,
            });
            self.emit(Instruction::Store { val: Operand::Const(IrConst::I8(0)), ptr: addr, ty: IrType::I8,
             seg_override: AddressSpace::Default });
            offset += 1;
        }
    }

    /// Zero-initialize an entire alloca.
    pub(super) fn zero_init_alloca(&mut self, alloca: Value, total_size: usize) {
        self.zero_init_region(alloca, 0, total_size);
    }

}
