//! Semantic analysis pass.
//!
//! Walks the AST to:
//! - Build a scoped symbol table of declarations
//! - Track function signatures for call validation
//! - Resolve typedef names and typeof(expr) via ExprTypeChecker
//! - Collect information needed by the IR lowering phase
//! - Map __builtin_* identifiers to their libc equivalents
//!
//! Expression CType inference is available via `type_checker::ExprTypeChecker`,
//! which uses SymbolTable + TypeContext + FunctionInfo to infer types without
//! depending on lowering state. This enables typeof(expr) resolution and
//! will eventually support type annotations on AST nodes.
//!
//! This pass does NOT reject programs with type errors (yet); it collects
//! information for the lowerer. Full type checking is TODO.

use crate::common::error::DiagnosticEngine;
use crate::common::source::Span;
use crate::common::symbol_table::{Symbol, SymbolTable};
use crate::common::type_builder;
use crate::common::types::{AddressSpace, CType, FunctionType, StructLayout};
use crate::frontend::parser::ast::{
    BinOp,
    BlockItem,
    CompoundStmt,
    Declaration,
    DerivedDeclarator,
    Designator,
    EnumVariant,
    Expr,
    ExprId,
    ExternalDecl,
    ForInit,
    FunctionDef,
    Initializer,
    SizeofArg,
    Stmt,
    StructFieldDecl,
    TranslationUnit,
    TypeSpecifier,
};
use crate::frontend::sema::builtins;
use super::type_context::{TypeContext, FunctionTypedefInfo};
use super::const_eval::{SemaConstEval, ConstMap};

use std::cell::RefCell;
use std::rc::Rc;
use crate::common::fx_hash::{FxHashMap, FxHashSet};

/// Outcome of a case segment in a switch statement for -Wreturn-type analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SwitchSegmentOutcome {
    /// Segment returns or otherwise diverges (goto, continue to outer loop, infinite loop)
    Returns,
    /// Segment breaks out of the switch
    Breaks,
    /// Segment falls through (to next case or end of switch body)
    FallsThrough,
}

/// Map from AST expression node identity to its inferred CType.
///
/// Keyed by [`ExprId`], a type-safe wrapper around each `Expr` node's identity.
/// The AST is allocated once during parsing and is not moved or reallocated
/// before the lowerer consumes it, so node identities are stable throughout
/// the compilation pipeline.
///
/// This is the core data structure for Step 3 of the typed-AST plan: sema
/// annotates every expression it can type-check, and the lowerer consults
/// these annotations as a fallback before doing its own (more expensive)
/// type inference.
pub type ExprTypeMap = FxHashMap<ExprId, CType>;

/// Information about a function collected during semantic analysis.
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub return_type: CType,
    pub params: Vec<(CType, Option<Rc<str>>)>,
    pub variadic: bool,
    pub is_defined: bool,
    /// Whether the function is declared with __attribute__((noreturn)) or _Noreturn
    pub is_noreturn: bool,
}

/// Results of semantic analysis, used by the lowering phase.
#[derive(Debug)]
pub struct SemaResult {
    /// Function signatures discovered during analysis.
    pub functions: FxHashMap<Rc<str>, FunctionInfo>,
    /// Type context populated by sema: typedefs, enum constants, struct layouts,
    /// function typedefs, function pointer typedefs.
    pub type_context: TypeContext,
    /// Expression type annotations: maps `ExprId` keys to their inferred
    /// CTypes. Populated during sema's analyze_expr walk using
    /// ExprTypeChecker. The lowerer consults this as a fallback in
    /// get_expr_ctype() after its own lowering-state-based inference fails.
    pub expr_types: ExprTypeMap,
    /// Pre-computed constant expression values: maps `ExprId` keys to their
    /// compile-time IrConst values. Populated during sema's walk
    /// using SemaConstEval (handles float literals, cast chains, sizeof,
    /// binary ops with proper signedness semantics). The lowerer consults
    /// this as an O(1) fast path before its own eval_const_expr.
    pub const_values: ConstMap,
}

impl Default for SemaResult {
    fn default() -> Self {
        Self {
            functions: FxHashMap::default(),
            type_context: TypeContext::new(),
            expr_types: ExprTypeMap::default(),
            const_values: ConstMap::default(),
        }
    }
}

/// Semantic analyzer that builds a scoped symbol table and collects type info.
pub struct SemanticAnalyzer {
    /// The scoped symbol table for name resolution.
    symbol_table: SymbolTable,
    /// Accumulated results for the lowerer.
    result: SemaResult,
    /// Current enum counter for auto-incrementing enum values.
    enum_counter: i64,
    // Anonymous struct/union keys are generated via TypeContext::next_anon_struct_id()
    // to ensure a single shared counter across both declaration processing and
    // expression type inference, avoiding key collisions between the two paths.
    /// Structured diagnostic engine for error/warning reporting.
    /// All sema errors and warnings are emitted with source spans through this
    /// engine, which handles rendering, filtering (-Wall/-Werror), and counting.
    /// Uses RefCell for interior mutability so that &self methods (e.g.,
    /// eval_const_expr_as_usize from TypeConvertContext) can emit diagnostics.
    diagnostics: RefCell<DiagnosticEngine>,
    /// Set of struct/union keys that have been defined (with a body, even if
    /// empty). Used to distinguish `struct X {}` (defined, complete) from
    /// `struct X;` (forward declaration, incomplete) for the incomplete type
    /// check. Uses RefCell for interior mutability since resolve_struct_or_union
    /// takes &self.
    defined_structs: RefCell<FxHashSet<Rc<str>>>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = Self {
            symbol_table: SymbolTable::new(),
            result: SemaResult::default(),
            enum_counter: 0,
            diagnostics: RefCell::new(DiagnosticEngine::new()),
            defined_structs: RefCell::new(FxHashSet::default()),
        };
        // Pre-populate with common implicit declarations
        analyzer.declare_implicit_functions();
        analyzer
    }

    /// Analyze a translation unit. Builds symbol table and collects semantic info.
    /// Returns Ok(()) on success, or an error count on failure.
    /// All errors are emitted through the DiagnosticEngine with source spans.
    pub fn analyze(&mut self, tu: &TranslationUnit) -> Result<(), usize> {
        for decl in &tu.decls {
            match decl {
                ExternalDecl::FunctionDef(func) => {
                    self.analyze_function_def(func);
                }
                ExternalDecl::Declaration(decl) => {
                    self.analyze_declaration(decl, /* is_global */ true);
                }
                ExternalDecl::TopLevelAsm(_) => {
                    // Top-level asm is passed through verbatim; no semantic analysis needed
                }
            }
        }
        let diag = self.diagnostics.borrow();
        if diag.has_errors() {
            Err(diag.error_count())
        } else {
            Ok(())
        }
    }

    /// Get the analysis results (consumed after analysis).
    pub fn into_result(self) -> SemaResult {
        // Both sema's declaration processing and expression type inference now use
        // TypeContext::next_anon_struct_id(), so no counter synchronization is needed.
        self.result
    }

    /// Set a pre-configured diagnostic engine on the analyzer.
    pub fn set_diagnostics(&mut self, engine: DiagnosticEngine) {
        self.diagnostics = RefCell::new(engine);
    }

    /// Take the diagnostic engine back from the analyzer.
    pub fn take_diagnostics(&mut self) -> DiagnosticEngine {
        self.diagnostics.replace(DiagnosticEngine::new())
    }

    // === Function analysis ===

    fn analyze_function_def(&mut self, func: &FunctionDef) {
        let return_type = self.type_spec_to_ctype(&func.return_type);

        // Register enum constants from the return type (e.g., anonymous enums
        // used as return types: `static enum { A, B } func(void) { ... }`).
        // Must happen at file scope before pushing the function body scope.
        self.collect_enum_constants_from_type_spec(&func.return_type);

        let params: Vec<(CType, Option<Rc<str>>)> = func.params.iter().map(|p| {
            let ty = self.type_spec_to_ctype(&p.type_spec);
            (ty, p.name.clone())
        }).collect();

        // Preserve noreturn from a prior declaration (e.g., the prototype may have
        // __attribute__((noreturn)) even if the definition doesn't repeat it).
        let prior_noreturn = self.result.functions.get(&func.name)
            .is_some_and(|fi| fi.is_noreturn);
        let func_info = FunctionInfo {
            return_type: return_type.clone(),
            params: params.clone(),
            variadic: func.variadic,
            is_defined: true,
            is_noreturn: func.attrs.is_noreturn() || prior_noreturn,
        };

        // Register in function table
        self.result.functions.insert(func.name.clone(), func_info);

        // Register in symbol table
        let func_ctype = CType::Function(Box::new(FunctionType {
            return_type: return_type.clone(),
            params: params.clone(),
            variadic: func.variadic,
        }));
        self.symbol_table.declare(Symbol {
            name: func.name.clone(),
            ty: func_ctype,
            explicit_alignment: None,
        });

        // Push scope for function body (both symbol table and type context,
        // so local struct/union definitions don't overwrite global layouts)
        self.symbol_table.push_scope();
        self.result.type_context.push_scope();

        // Declare parameters in function scope
        for param in &func.params {
            if let Some(name) = &param.name {
                let ty = self.type_spec_to_ctype(&param.type_spec);
                self.symbol_table.declare(Symbol {
                    name: name.clone(),
                    ty,
                    explicit_alignment: None,
                });
            }
        }

        // Analyze function body
        self.analyze_compound_stmt(&func.body);

        // -Wreturn-type: warn if a non-void function can fall through without returning.
        // Skip the check for:
        //   - void functions (no return value expected)
        //   - noreturn functions (never return by contract)
        //   - naked functions (body is pure inline asm; return is handled by asm)
        //   - main() (C99 5.1.2.2.3: reaching } is equivalent to return 0)
        if !matches!(return_type, CType::Void)
            && !func.attrs.is_noreturn()
            && !func.attrs.is_naked()
            && &*func.name != "main"
            && self.compound_can_fall_through(&func.body)
        {
            self.diagnostics.borrow_mut().warning_with_kind(
                "control reaches end of non-void function",
                func.span,
                crate::common::error::WarningKind::ReturnType,
            );
        }

        // Pop function scope
        self.result.type_context.pop_scope();
        self.symbol_table.pop_scope();
    }

    // === Declaration analysis ===

    fn analyze_declaration(&mut self, decl: &Declaration, _is_global: bool) {
        // Register enum constants from this declaration's type specifier.
        // This recursively walks into struct/union fields to find inline
        // enum definitions (e.g., `struct { enum { A, B } mode; }`).
        self.collect_enum_constants_from_type_spec(&decl.type_spec);

        // Handle __auto_type: infer base type from the first declarator's
        // initializer expression rather than defaulting to int.
        let base_type = if matches!(&decl.type_spec, TypeSpecifier::AutoType) {
            if let Some(first) = decl.declarators.first() {
                if let Some(Initializer::Expr(ref init_expr)) = first.init {
                    let checker = super::type_checker::ExprTypeChecker {
                        symbols: &self.symbol_table,
                        types: &self.result.type_context,
                        functions: &self.result.functions,
                        expr_types: Some(&self.result.expr_types),
                    };
                    checker.infer_expr_ctype(init_expr).unwrap_or(CType::Int)
                } else {
                    CType::Int
                }
            } else {
                CType::Int
            }
        } else {
            self.type_spec_to_ctype(&decl.type_spec)
        };

        // Handle typedef declarations: populate TypeContext with typedef info
        if decl.is_typedef() {
            for declarator in &decl.declarators {
                if declarator.name.is_empty() {
                    continue;
                }
                // Check for function typedef (e.g., typedef int func_t(int, int);)
                let has_func_derived = declarator.derived.iter().any(|d|
                    matches!(d, DerivedDeclarator::Function(_, _)));
                let has_fptr_derived = declarator.derived.iter().any(|d|
                    matches!(d, DerivedDeclarator::FunctionPointer(_, _)));

                if has_func_derived && !has_fptr_derived {
                    // Function typedef like: typedef int func_t(int x);
                    if let Some(DerivedDeclarator::Function(params, variadic)) =
                        declarator.derived.iter().find(|d| matches!(d, DerivedDeclarator::Function(_, _)))
                    {
                        let ptr_count = declarator.derived.iter()
                            .take_while(|d| matches!(d, DerivedDeclarator::Pointer))
                            .count();
                        let mut return_type = decl.type_spec.clone();
                        for _ in 0..ptr_count {
                            return_type = TypeSpecifier::Pointer(Box::new(return_type), AddressSpace::Default);
                        }
                        self.result.type_context.function_typedefs.insert(
                            declarator.name.clone(),
                            FunctionTypedefInfo {
                                return_type,
                                params: params.clone(),
                                variadic: *variadic,
                            },
                        );
                    }
                }

                // Function pointer typedef (e.g., typedef void *(*lua_Alloc)(void *, ...))
                if has_fptr_derived {
                    if let Some(DerivedDeclarator::FunctionPointer(params, variadic)) = declarator.derived.iter().find(|d|
                        matches!(d, DerivedDeclarator::FunctionPointer(_, _)))
                    {
                        let ptr_count = declarator.derived.iter()
                            .take_while(|d| matches!(d, DerivedDeclarator::Pointer))
                            .count();
                        let ret_ptr_count = if ptr_count > 0 { ptr_count - 1 } else { 0 };
                        let mut return_type = decl.type_spec.clone();
                        for _ in 0..ret_ptr_count {
                            return_type = TypeSpecifier::Pointer(Box::new(return_type), AddressSpace::Default);
                        }
                        self.result.type_context.func_ptr_typedefs.insert(declarator.name.clone());
                        self.result.type_context.func_ptr_typedef_info.insert(
                            declarator.name.clone(),
                            FunctionTypedefInfo {
                                return_type,
                                params: params.clone(),
                                variadic: *variadic,
                            },
                        );
                    }
                }

                // Store resolved CType for the typedef.
                // Use build_full_ctype_with_base to reuse the already-resolved base
                // type, avoiding re-resolution of anonymous struct type specs.
                let resolved_ctype = if declarator.derived.is_empty() {
                    base_type.clone()
                } else {
                    type_builder::build_full_ctype_with_base(self, base_type.clone(), &declarator.derived)
                };
                self.result.type_context.typedefs.insert(declarator.name.clone(), resolved_ctype);

                // Preserve alignment override from __attribute__((aligned(N))) on the typedef.
                // E.g. typedef struct S aligned_S __attribute__((aligned(32)));
                // Also handle _Alignas(type) on typedefs by computing alignment from the type.
                //
                // When alignment_sizeof_type is set, the parser saw aligned(sizeof(type))
                // but may have computed sizeof incorrectly for struct/union types.  Recompute
                // sizeof here with full struct layout information and use that as alignment.
                let effective_alignment = {
                    let mut align = decl.alignment;
                    if let Some(ref sizeof_ts) = decl.alignment_sizeof_type {
                        let ctype = self.type_spec_to_ctype(sizeof_ts);
                        let real_sizeof = ctype.size_ctx(&*self.result.type_context.borrow_struct_layouts());
                        align = Some(align.map_or(real_sizeof, |a| a.max(real_sizeof)));
                    }
                    align.or_else(|| {
                        decl.alignas_type.as_ref().map(|ts| {
                            // _Alignas(type) means align to alignof(type)
                            let ctype = self.type_spec_to_ctype(ts);
                            ctype.align_ctx(&*self.result.type_context.borrow_struct_layouts())
                        })
                    })
                };
                if let Some(align) = effective_alignment {
                    self.result.type_context.typedef_alignments.insert(declarator.name.clone(), align);
                }
            }
            return; // typedefs don't declare variables
        }

        for init_decl in &decl.declarators {
            let mut full_type = if init_decl.derived.is_empty() {
                base_type.clone()
            } else {
                // Use build_full_ctype_with_base to reuse the already-resolved base
                // type. This is critical for anonymous structs: re-resolving the type
                // spec would call next_anon_struct_id() again and produce a different
                // key for each declarator, breaking pointer-array compatibility checks.
                type_builder::build_full_ctype_with_base(self, base_type.clone(), &init_decl.derived)
            };

            // Resolve incomplete array sizes from initializers (e.g., int arr[] = {1,2,3})
            // This must happen before storing the symbol so sizeof(arr) works in
            // subsequent global initializer const-evaluation.
            if let CType::Array(ref elem, None) = full_type {
                if let Some(ref init) = init_decl.init {
                    if let Some(count) = self.count_initializer_elements(init, elem) {
                        full_type = CType::Array(elem.clone(), Some(count));
                    }
                }
            }

            // Check if this is a function declaration (prototype)
            if let CType::Function(ref ft) = full_type {
                let is_noreturn = init_decl.attrs.is_noreturn();
                // If redeclared with noreturn, update existing entry
                if is_noreturn {
                    if let Some(existing) = self.result.functions.get_mut(&init_decl.name) {
                        existing.is_noreturn = true;
                    } else {
                        let func_info = FunctionInfo {
                            return_type: ft.return_type.clone(),
                            params: ft.params.clone(),
                            variadic: ft.variadic,
                            is_defined: false,
                            is_noreturn: true,
                        };
                        self.result.functions.insert(init_decl.name.clone(), func_info);
                    }
                } else if !self.result.functions.contains_key(&init_decl.name) {
                    let func_info = FunctionInfo {
                        return_type: ft.return_type.clone(),
                        params: ft.params.clone(),
                        variadic: ft.variadic,
                        is_defined: false,
                        is_noreturn: false,
                    };
                    self.result.functions.insert(init_decl.name.clone(), func_info);
                }
            }

            // Check for incomplete struct/union types in variable declarations.
            // A variable with an incomplete type (forward-declared struct/union with no
            // definition) cannot be allocated. This is only an error for definitions,
            // not for extern declarations (which don't allocate storage).
            // Also catches arrays of incomplete element types.
            // GCC error: "storage size of 'x' isn't known"
            if !decl.is_extern() && !matches!(full_type, CType::Function(_)) {
                // Extract the innermost element type (unwrap arrays)
                let mut elem_type = &full_type;
                while let CType::Array(inner, _) = elem_type {
                    elem_type = inner;
                }
                let is_incomplete = match elem_type {
                    CType::Struct(key) | CType::Union(key) => {
                        !self.defined_structs.borrow().contains(&**key)
                    }
                    _ => false,
                };
                if is_incomplete {
                    self.diagnostics.borrow_mut().error(
                        format!("storage size of '{}' isn't known", init_decl.name),
                        init_decl.span,
                    );
                }
            }

            // Resolve explicit alignment from _Alignas or __attribute__((aligned(N))).
            // For _Alignas(type), resolve the type alignment via sema's type resolution.
            // For _Alignas(N) or __attribute__((aligned(N))), use the parsed numeric value.
            let explicit_alignment = if let Some(ref alignas_ts) = decl.alignas_type {
                let ct = self.type_spec_to_ctype(alignas_ts);
                let a = ct.align_ctx(&*self.result.type_context.borrow_struct_layouts());
                if a > 0 { Some(a) } else { None }
            } else {
                decl.alignment
            };

            self.symbol_table.declare(Symbol {
                name: init_decl.name.clone(),
                ty: full_type,
                explicit_alignment,
            });

            // Analyze array size expressions in derived declarators
            // (catches undeclared identifiers in e.g. `int arr[UNDECLARED];`)
            for derived in &init_decl.derived {
                if let DerivedDeclarator::Array(Some(size_expr)) = derived {
                    self.analyze_expr(size_expr);
                }
            }

            // Analyze initializer expressions
            if let Some(init) = &init_decl.init {
                self.analyze_initializer(init);
                // Check for invalid pointer <-> float conversions in scalar initializers
                if let Initializer::Expr(init_expr) = init {
                    let var_ty = self.symbol_table.lookup(&init_decl.name)
                        .map(|s| s.ty.clone());
                    if let Some(var_ty) = var_ty {
                        let checker = super::type_checker::ExprTypeChecker {
                            symbols: &self.symbol_table,
                            types: &self.result.type_context,
                            functions: &self.result.functions,
                            expr_types: Some(&self.result.expr_types),
                        };
                        if let Some(init_ty) = checker.infer_expr_ctype(init_expr) {
                            self.check_pointer_float_conversion(
                                &init_ty, &var_ty, init_expr.span(),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Count the number of array elements from an initializer.
    /// Used to resolve incomplete array types (e.g., `int arr[] = {1,2,3}`)
    /// so that sizeof(arr) works correctly in subsequent const evaluation.
    ///
    /// Uses the same index-tracking algorithm as lowering's
    /// `compute_init_list_array_size`. Handles initializer lists with
    /// designators, string literals for char arrays, and brace-wrapped
    /// string literals.
    fn count_initializer_elements(&self, init: &Initializer, elem_ty: &CType) -> Option<usize> {
        match init {
            Initializer::List(items) => {
                // Check for brace-wrapped string literal: char c[] = {"hello"}
                let is_char_elem = matches!(elem_ty, CType::Char | CType::UChar);
                let is_int_elem = matches!(elem_ty, CType::Int | CType::UInt);
                if items.len() == 1 && items[0].designators.is_empty() {
                    if is_char_elem {
                        if let Initializer::Expr(Expr::StringLiteral(s, _)) = &items[0].init {
                            return Some(s.chars().count() + 1);
                        }
                    }
                    if is_int_elem {
                        if let Initializer::Expr(Expr::WideStringLiteral(s, _)) = &items[0].init {
                            // For wchar_t arrays, each Unicode codepoint is one element
                            return Some(s.chars().count() + 1);
                        }
                    }
                    // char16_t array: unsigned short c[] = {u"hello"}
                    if matches!(elem_ty, CType::UShort | CType::Short) {
                        if let Initializer::Expr(Expr::Char16StringLiteral(s, _)) = &items[0].init {
                            return Some(s.chars().count() + 1);
                        }
                    }
                }

                // C11 6.7.9: flat initializers for struct arrays fill fields
                // sequentially, so array members consume multiple scalar items.
                // E.g. struct { int arr[4]; int b, e, k; } needs 7 items per element.
                let flat_scalars_per_elem = self.flat_scalar_count_for_type(elem_ty);

                // General case: track current index through the initializer list.
                let mut max_idx = 0usize;
                let mut current_idx = 0usize;
                let mut fields_consumed = 0usize;
                let has_any_designator = items.iter()
                    .any(|item| !item.designators.is_empty());

                for item in items {
                    // If this item has an array index designator, jump to that index
                    if let Some(designator) = item.designators.first() {
                        match designator {
                            Designator::Index(idx_expr) => {
                                if let Some(idx) = self.eval_designator_index(idx_expr) {
                                    current_idx = idx;
                                    fields_consumed = 0;
                                }
                            }
                            Designator::Range(_lo, hi) => {
                                // GNU range designator [lo ... hi]: size determined by hi
                                if let Some(idx) = self.eval_designator_index(hi) {
                                    current_idx = idx;
                                    fields_consumed = 0;
                                }
                            }
                            Designator::Field(_) => {
                                // Field designator on struct element - don't advance
                            }
                        }
                    }
                    if current_idx >= max_idx {
                        max_idx = current_idx + 1;
                    }
                    // Only advance if this item doesn't have a field designator
                    // (field designators target fields within the same struct element)
                    let has_field_designator = item.designators.iter()
                        .any(|d| matches!(d, Designator::Field(_)));
                    if !has_field_designator {
                        if flat_scalars_per_elem > 1 && !has_any_designator {
                            // Struct with array fields: group flat scalars per struct.
                            // Designators disable grouping (conservative per-item count).
                            let is_braced = matches!(item.init, Initializer::List(_));
                            if is_braced {
                                // Braced sub-list counts as one complete element
                                current_idx += 1;
                                fields_consumed = 0;
                            } else {
                                fields_consumed += 1;
                                if fields_consumed >= flat_scalars_per_elem {
                                    current_idx += 1;
                                    fields_consumed = 0;
                                }
                            }
                        } else {
                            current_idx += 1;
                        }
                    }
                }

                // Count partial struct element at the end
                if flat_scalars_per_elem > 1 && fields_consumed > 0
                    && current_idx >= max_idx {
                        max_idx = current_idx + 1;
                    }

                if has_any_designator {
                    Some(max_idx)
                } else {
                    Some(max_idx.max(items.len().div_ceil(flat_scalars_per_elem.max(1))))
                }
            }
            Initializer::Expr(expr) => {
                // String literal initializer for char arrays: char s[] = "hello"
                match (elem_ty, expr) {
                    (CType::Char | CType::UChar, Expr::StringLiteral(s, _)) => {
                        Some(s.chars().count() + 1) // +1 for null terminator
                    }
                    (CType::Int | CType::UInt, Expr::WideStringLiteral(s, _)) => {
                        // For wchar_t/char32_t arrays, each Unicode codepoint is one element
                        Some(s.chars().count() + 1)
                    }
                    (CType::UShort | CType::Short, Expr::Char16StringLiteral(s, _)) => {
                        // For char16_t arrays, each Unicode codepoint is one element
                        Some(s.chars().count() + 1)
                    }
                    _ => None,
                }
            }
        }
    }

    /// Compute the flat scalar count for a type - how many scalar initializer
    /// items are needed to fully initialize one value of this type.
    /// Scalar types = 1, arrays = element_count * per_element, structs = sum of fields.
    fn flat_scalar_count_for_type(&self, ty: &CType) -> usize {
        match ty {
            CType::Array(elem_ty, Some(size)) => {
                *size * self.flat_scalar_count_for_type(elem_ty)
            }
            CType::Array(_, None) => 0, // unsized arrays contribute 0 scalars
            CType::Struct(key) | CType::Union(key) => {
                let layouts = self.result.type_context.borrow_struct_layouts();
                if let Some(layout) = layouts.get(&**key) {
                    if layout.fields.is_empty() {
                        return 0;
                    }
                    if layout.is_union {
                        layout.fields.iter()
                            .map(|f| self.flat_scalar_count_for_type(&f.ty))
                            .max()
                            .unwrap_or(1)
                    } else {
                        layout.fields.iter()
                            .map(|f| self.flat_scalar_count_for_type(&f.ty))
                            .sum()
                    }
                } else {
                    1
                }
            }
            _ => 1,
        }
    }

    /// Try to evaluate a designator index expression as a usize.
    /// Handles integer and char literals, and falls back to the full
    /// constant expression evaluator for enum values, sizeof, etc.
    fn eval_designator_index(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::IntLiteral(n, _) | Expr::LongLiteral(n, _) | Expr::LongLongLiteral(n, _) => Some(*n as usize),
            Expr::UIntLiteral(n, _) | Expr::ULongLiteral(n, _) | Expr::ULongLongLiteral(n, _) => Some(*n as usize),
            Expr::CharLiteral(n, _) => Some(*n as usize),
            _ => {
                let val = self.eval_const_expr(expr)?;
                if val >= 0 { Some(val as usize) } else { None }
            }
        }
    }

    fn process_enum_variants(&mut self, variants: &[EnumVariant]) {
        self.enum_counter = 0;
        for variant in variants {
            if let Some(val_expr) = &variant.value {
                // Walk the enum value expression through analyze_expr first to
                // populate the expr_types cache. Without this, infer_expr_ctype
                // called from eval_const_expr would have no cached sub-expression
                // types and could exhibit O(2^N) blowup on deep expression chains
                // (e.g., `enum { NUM = +1+1+1+...+1 }`).
                self.analyze_expr(val_expr);
                // Try to evaluate constant expression
                if let Some(val) = self.eval_const_expr(val_expr) {
                    self.enum_counter = val;
                }
            }
            // Use insert_enum_scoped so that enum constants defined inside
            // function bodies are properly removed when the scope exits.
            // At global scope the scope stack is empty, so insert_enum_scoped
            // behaves identically to a direct insert.
            self.result.type_context.insert_enum_scoped(variant.name.clone(), self.enum_counter);
            // GCC extension: enum constants that don't fit in int are promoted
            let sym_ty = super::type_checker::enum_constant_type(self.enum_counter);
            self.symbol_table.declare(Symbol {
                name: variant.name.clone(),
                ty: sym_ty,
                explicit_alignment: None,
            });
            self.enum_counter += 1;
        }
    }

    /// Recursively collect enum constants from a type specifier, walking into
    /// struct/union field types. This ensures that inline enum definitions
    /// within structs (e.g., `struct { enum { A, B } mode; }`) have their
    /// constants registered in the enclosing scope, matching GCC/Clang behavior.
    fn collect_enum_constants_from_type_spec(&mut self, ts: &TypeSpecifier) {
        match ts {
            TypeSpecifier::Enum(name, Some(variants), is_packed) => {
                self.process_enum_variants(variants);
                // Store packed enum info for forward-reference lookups
                if *is_packed {
                    if let Some(tag) = name {
                        let mut variant_values = Vec::new();
                        let mut next_val: i64 = 0;
                        for v in variants {
                            if let Some(ref val_expr) = v.value {
                                if let Some(val) = self.eval_const_expr(val_expr) {
                                    next_val = val;
                                }
                            }
                            variant_values.push((v.name.clone(), next_val));
                            next_val += 1;
                        }
                        self.result.type_context.packed_enum_types.insert(
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
            TypeSpecifier::Struct(_, Some(fields), _, _, _)
            | TypeSpecifier::Union(_, Some(fields), _, _, _) => {
                for field in fields {
                    self.collect_enum_constants_from_type_spec(&field.type_spec);
                }
            }
            TypeSpecifier::Array(inner, _) | TypeSpecifier::Pointer(inner, _) => {
                self.collect_enum_constants_from_type_spec(inner);
            }
            _ => {}
        }
    }

    // === Statement analysis ===

    fn analyze_compound_stmt(&mut self, compound: &CompoundStmt) {
        self.symbol_table.push_scope();
        self.result.type_context.push_scope();
        for item in &compound.items {
            match item {
                BlockItem::Declaration(decl) => {
                    self.analyze_declaration(decl, /* is_global */ false);
                }
                BlockItem::Statement(stmt) => {
                    self.analyze_stmt(stmt);
                }
            }
        }
        self.result.type_context.pop_scope();
        self.symbol_table.pop_scope();
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr(Some(expr)) => {
                self.analyze_expr(expr);
            }
            Stmt::Expr(None) => {}
            Stmt::Return(Some(expr), _) => {
                self.analyze_expr(expr);
            }
            Stmt::Return(None, _) => {}
            Stmt::If(cond, then_br, else_br, _) => {
                self.analyze_expr(cond);
                self.analyze_stmt(then_br);
                if let Some(else_stmt) = else_br {
                    self.analyze_stmt(else_stmt);
                }
            }
            Stmt::While(cond, body, _) => {
                self.analyze_expr(cond);
                self.analyze_stmt(body);
            }
            Stmt::DoWhile(body, cond, _) => {
                self.analyze_stmt(body);
                self.analyze_expr(cond);
            }
            Stmt::For(init, cond, inc, body, _) => {
                self.symbol_table.push_scope();
                self.result.type_context.push_scope();
                if let Some(init) = init {
                    match init.as_ref() {
                        ForInit::Declaration(decl) => {
                            self.analyze_declaration(decl, false);
                        }
                        ForInit::Expr(expr) => {
                            self.analyze_expr(expr);
                        }
                    }
                }
                if let Some(cond) = cond {
                    self.analyze_expr(cond);
                }
                if let Some(inc) = inc {
                    self.analyze_expr(inc);
                }
                self.analyze_stmt(body);
                self.result.type_context.pop_scope();
                self.symbol_table.pop_scope();
            }
            Stmt::Compound(compound) => {
                self.analyze_compound_stmt(compound);
            }
            Stmt::Switch(expr, body, _) => {
                self.analyze_expr(expr);
                // C11 6.8.4.2: The controlling expression of a switch statement
                // shall have integer type.
                if let Some(ctype) = self.result.expr_types.get(&expr.id()) {
                    if !ctype.is_integer() {
                        let diag = crate::common::error::Diagnostic::error(
                            "switch quantity is not an integer"
                        ).with_span(expr.span())
                         .with_note(crate::common::error::Diagnostic::note(
                            format!("expression has type '{}'", ctype)
                         ).with_span(expr.span()));
                        self.diagnostics.borrow_mut().emit(&diag);
                    }
                }
                self.analyze_stmt(body);
            }
            Stmt::Case(expr, body, _) => {
                self.analyze_expr(expr);
                self.analyze_stmt(body);
            }
            Stmt::CaseRange(low, high, body, _) => {
                self.analyze_expr(low);
                self.analyze_expr(high);
                self.analyze_stmt(body);
            }
            Stmt::Default(body, _) => {
                self.analyze_stmt(body);
            }
            Stmt::Label(_, body, _) => {
                self.analyze_stmt(body);
            }
            Stmt::Declaration(decl) => {
                self.analyze_declaration(decl, /* is_global */ false);
            }
            Stmt::Break(_) | Stmt::Continue(_) | Stmt::Goto(_, _) => {}
            Stmt::GotoIndirect(expr, _) => {
                self.analyze_expr(expr);
            }
            Stmt::InlineAsm { outputs, inputs, .. } => {
                for out in outputs {
                    self.analyze_expr(&out.expr);
                }
                for inp in inputs {
                    self.analyze_expr(&inp.expr);
                }
            }
        }
    }

    // === Return-type analysis (-Wreturn-type) ===

    /// Check whether a compound statement (function body) can fall through
    /// without executing a return statement. Used by `-Wreturn-type` to detect
    /// non-void functions that may not return a value.
    ///
    /// Returns `true` if control can reach the end of the compound statement,
    /// `false` if all paths through the block are guaranteed to return/diverge.
    fn compound_can_fall_through(&self, compound: &CompoundStmt) -> bool {
        if compound.items.is_empty() {
            return true;
        }
        // Check each item - if any item doesn't fall through, the rest is unreachable
        for item in &compound.items {
            let falls_through = match item {
                BlockItem::Statement(stmt) => self.stmt_can_fall_through(stmt),
                BlockItem::Declaration(_) => true,
            };
            if !falls_through {
                return false;
            }
        }
        true
    }

    /// Check whether a statement can fall through (i.e., control can reach the
    /// point immediately after this statement without a return/goto/diverge).
    fn stmt_can_fall_through(&self, stmt: &Stmt) -> bool {
        match stmt {
            // return always diverges
            Stmt::Return(_, _) => false,

            // goto always diverges (transfers control elsewhere)
            Stmt::Goto(_, _) | Stmt::GotoIndirect(_, _) => false,

            // break/continue diverge from the current construct
            // (conservative for our purposes: they exit the current statement)
            Stmt::Break(_) | Stmt::Continue(_) => false,

            // if/else: falls through only if either branch can fall through
            // if without else: can fall through unless the condition is constant true
            // and the then-branch diverges (e.g., `if(1) noreturn_call()`)
            Stmt::If(cond, then_br, else_br, _) => {
                match else_br {
                    Some(else_stmt) => {
                        // Both branches must not fall through for the if/else to not fall through
                        self.stmt_can_fall_through(then_br) || self.stmt_can_fall_through(else_stmt)
                    }
                    None => {
                        // if without else: can fall through unless condition is constant true
                        // and the body diverges (handles `if(1) abort()` and BUILD_BUG patterns)
                        !self.is_constant_true_expr(cond) || self.stmt_can_fall_through(then_br)
                    }
                }
            }

            // while/do-while/for: conservatively can fall through, unless the
            // condition is a compile-time constant true (infinite loop), or the
            // body itself diverges (contains __builtin_unreachable(), etc.)
            Stmt::While(cond, _body, _) => {
                if self.is_constant_true_expr(cond) {
                    false // infinite loop
                } else {
                    true // might not execute body at all, so can fall through
                }
            }
            Stmt::DoWhile(body, cond, _) => {
                // Can fall through only if not an infinite loop AND body can fall through
                !self.is_constant_true_expr(cond) && self.stmt_can_fall_through(body)
            }
            Stmt::For(_init, cond, _inc, _body, _) => {
                // for(;;) with no condition is an infinite loop
                match cond {
                    None => false, // for(;;) is infinite
                    Some(c) => !self.is_constant_true_expr(c),
                }
            }

            // switch: can fall through unless it has a default case and every
            // case segment ends with a return/goto/diverge (not break/fallthrough).
            Stmt::Switch(_, body, _) => self.switch_can_fall_through(body),

            // compound: delegate to compound analysis
            Stmt::Compound(compound) => self.compound_can_fall_through(compound),

            // labels: the relevant question is whether the labeled statement falls through
            Stmt::Label(_, inner, _) => self.stmt_can_fall_through(inner),
            Stmt::Case(_, inner, _) => self.stmt_can_fall_through(inner),
            Stmt::CaseRange(_, _, inner, _) => self.stmt_can_fall_through(inner),
            Stmt::Default(inner, _) => self.stmt_can_fall_through(inner),

            // expression statements: fall through unless they call a noreturn function
            Stmt::Expr(Some(expr)) => !self.is_noreturn_call(expr),
            Stmt::Expr(None) => true,
            Stmt::Declaration(_) => true,
            Stmt::InlineAsm { .. } => true,
        }
    }

    /// Check whether a switch statement can fall through to the next statement.
    ///
    /// A switch cannot fall through if:
    /// 1. It has a `default` label (so all values are covered), AND
    /// 2. No case segment contains a `break` that exits the switch, AND
    /// 3. The last segment in the compound body doesn't fall through.
    ///
    /// Segments that fall through to the next case (without break) are allowed,
    /// since control just continues to the next case label.
    fn switch_can_fall_through(&self, body: &Stmt) -> bool {
        let compound = match body {
            Stmt::Compound(c) => c,
            // Non-compound switch body: conservatively say it can fall through
            _ => return true,
        };

        if compound.items.is_empty() {
            return true;
        }

        // Must have a default label to cover all values
        if !self.switch_has_default(&compound.items) {
            return true;
        }

        // Collect case segment boundaries
        let mut segment_starts: Vec<usize> = Vec::new();
        for (i, item) in compound.items.iter().enumerate() {
            if let BlockItem::Statement(stmt) = item {
                if self.is_case_label(stmt) {
                    segment_starts.push(i);
                }
            }
        }

        if segment_starts.is_empty() {
            return true;
        }

        // Check each segment:
        // - If ANY segment ends with a break, the switch falls through
        // - If the LAST segment falls through, the switch falls through
        for (seg_idx, &start) in segment_starts.iter().enumerate() {
            let end = if seg_idx + 1 < segment_starts.len() {
                segment_starts[seg_idx + 1]
            } else {
                compound.items.len()
            };

            let segment = &compound.items[start..end];
            let outcome = self.segment_outcome(segment);
            match outcome {
                SwitchSegmentOutcome::Returns => {
                    // This segment returns/diverges - doesn't cause fallthrough
                }
                SwitchSegmentOutcome::Breaks => {
                    // This segment breaks out of the switch - switch falls through
                    return true;
                }
                SwitchSegmentOutcome::FallsThrough => {
                    // Falls through to next segment. Only a problem if this is the last segment.
                    if seg_idx + 1 >= segment_starts.len() {
                        return true;
                    }
                    // Otherwise, fallthrough to next case is fine (C semantics)
                }
            }
        }

        false
    }

    /// Determine the outcome of a case segment: does it return, break, or fall through?
    ///
    /// Walks items sequentially: if an earlier item diverges (returns/breaks),
    /// subsequent items are dead code and don't affect the outcome. This handles
    /// patterns like `return NULL; break;` where the break is unreachable.
    fn segment_outcome(&self, segment: &[BlockItem]) -> SwitchSegmentOutcome {
        if segment.is_empty() {
            return SwitchSegmentOutcome::FallsThrough;
        }

        for item in segment {
            let outcome = match item {
                BlockItem::Declaration(_) => continue,
                BlockItem::Statement(stmt) => {
                    let inner = self.unwrap_case_label(stmt);
                    self.stmt_switch_outcome(inner)
                }
            };
            // If this item returns or breaks, subsequent items are dead code
            if outcome != SwitchSegmentOutcome::FallsThrough {
                return outcome;
            }
        }
        SwitchSegmentOutcome::FallsThrough
    }

    /// Determine what a statement does in a switch context: return, break, or fall through.
    fn stmt_switch_outcome(&self, stmt: &Stmt) -> SwitchSegmentOutcome {
        match stmt {
            Stmt::Return(_, _) | Stmt::Goto(_, _) | Stmt::GotoIndirect(_, _) => {
                SwitchSegmentOutcome::Returns
            }
            Stmt::Break(_) => SwitchSegmentOutcome::Breaks,
            Stmt::Continue(_) => SwitchSegmentOutcome::Returns,

            Stmt::Compound(compound) => {
                // Walk items sequentially: if an item diverges (returns/breaks),
                // subsequent items are dead code. This handles `return x; break;`.
                for item in &compound.items {
                    let outcome = match item {
                        BlockItem::Declaration(_) => continue,
                        BlockItem::Statement(s) => self.stmt_switch_outcome(s),
                    };
                    if outcome != SwitchSegmentOutcome::FallsThrough {
                        return outcome;
                    }
                }
                SwitchSegmentOutcome::FallsThrough
            }

            Stmt::If(_, then_br, Some(else_stmt), _) => {
                let then_out = self.stmt_switch_outcome(then_br);
                let else_out = self.stmt_switch_outcome(else_stmt);
                if then_out == SwitchSegmentOutcome::Returns
                    && else_out == SwitchSegmentOutcome::Returns
                {
                    SwitchSegmentOutcome::Returns
                } else if then_out == SwitchSegmentOutcome::Breaks
                    || else_out == SwitchSegmentOutcome::Breaks
                {
                    SwitchSegmentOutcome::Breaks
                } else {
                    SwitchSegmentOutcome::FallsThrough
                }
            }
            Stmt::If(..) => SwitchSegmentOutcome::FallsThrough,

            Stmt::While(cond, _, _) => {
                if self.is_constant_true_expr(cond) {
                    SwitchSegmentOutcome::Returns
                } else {
                    SwitchSegmentOutcome::FallsThrough
                }
            }
            Stmt::DoWhile(body, cond, _) => {
                // Returns if: infinite loop OR body diverges
                if self.is_constant_true_expr(cond) || !self.stmt_can_fall_through(body) {
                    SwitchSegmentOutcome::Returns
                } else {
                    SwitchSegmentOutcome::FallsThrough
                }
            }
            Stmt::For(_, cond, _, _, _) => match cond {
                None => SwitchSegmentOutcome::Returns,
                Some(c) => {
                    if self.is_constant_true_expr(c) {
                        SwitchSegmentOutcome::Returns
                    } else {
                        SwitchSegmentOutcome::FallsThrough
                    }
                }
            },

            Stmt::Label(_, inner, _)
            | Stmt::Case(_, inner, _)
            | Stmt::CaseRange(_, _, inner, _)
            | Stmt::Default(inner, _) => self.stmt_switch_outcome(inner),

            Stmt::Switch(_, inner_body, _) => {
                if self.switch_can_fall_through(inner_body) {
                    SwitchSegmentOutcome::FallsThrough
                } else {
                    SwitchSegmentOutcome::Returns
                }
            }

            Stmt::Expr(Some(expr)) => {
                if self.is_noreturn_call(expr) {
                    SwitchSegmentOutcome::Returns
                } else {
                    SwitchSegmentOutcome::FallsThrough
                }
            }

            _ => SwitchSegmentOutcome::FallsThrough,
        }
    }

    /// Unwrap case/default label wrappers to get to the inner statement.
    fn unwrap_case_label<'b>(&self, stmt: &'b Stmt) -> &'b Stmt {
        match stmt {
            Stmt::Case(_, inner, _) | Stmt::CaseRange(_, _, inner, _) | Stmt::Default(inner, _) => {
                self.unwrap_case_label(inner)
            }
            other => other,
        }
    }

    /// Check if a statement is a case/default label.
    fn is_case_label(&self, stmt: &Stmt) -> bool {
        matches!(stmt, Stmt::Case(_, _, _) | Stmt::CaseRange(_, _, _, _) | Stmt::Default(_, _))
    }

    /// Check if any item in a switch body contains a `default` label.
    fn switch_has_default(&self, items: &[BlockItem]) -> bool {
        for item in items {
            if let BlockItem::Statement(stmt) = item {
                if self.stmt_contains_default(stmt) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if a statement is or contains a `default` label (handles nested case labels).
    fn stmt_contains_default(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Default(_, _) => true,
            Stmt::Case(_, inner, _) | Stmt::CaseRange(_, _, inner, _) => {
                self.stmt_contains_default(inner)
            }
            _ => false,
        }
    }

    /// Check if an expression diverges (never falls through to subsequent code).
    /// This includes calls to noreturn functions (`abort()`, `exit()`,
    /// `__builtin_unreachable()`, or any function declared with
    /// `__attribute__((noreturn))`), and GCC statement expressions whose
    /// compound body cannot fall through (e.g. `({ return val; })`).
    fn is_noreturn_call(&self, expr: &Expr) -> bool {
        match expr {
            Expr::FunctionCall(callee, _, _) => {
                if let Expr::Identifier(name, _) = callee.as_ref() {
                    // Check built-in noreturn functions
                    if matches!(
                        &**name,
                        "__builtin_unreachable"
                            | "__builtin_trap"
                            | "__builtin_abort"
                            | "abort"
                            | "exit"
                            | "_exit"
                            | "_Exit"
                            | "quick_exit"
                            | "__assert_fail"
                            | "__assert_rtn"
                            | "longjmp"
                            | "siglongjmp"
                            | "__longjmp_chk"
                    ) {
                        return true;
                    }
                    // Check user-declared noreturn functions
                    if let Some(func_info) = self.result.functions.get(name) {
                        return func_info.is_noreturn;
                    }
                    false
                } else {
                    false
                }
            }
            // Handle comma expressions: (void)0, __builtin_unreachable()
            Expr::Comma(_, rhs, _) => self.is_noreturn_call(rhs),
            // Handle cast expressions: (void)noreturn_call()
            Expr::Cast(_, inner, _) => self.is_noreturn_call(inner),
            // Handle GCC statement expressions: ({ ...; return val; })
            // A return inside a statement expression returns from the enclosing
            // function, so if the compound body cannot fall through, the
            // expression itself diverges.
            Expr::StmtExpr(compound, _) => !self.compound_can_fall_through(compound),
            _ => false,
        }
    }

    /// Check if an expression is a compile-time constant that evaluates to true (non-zero).
    /// Used to detect infinite loops like `while(1)` and `for(;1;)`, and for
    /// `if(1)` patterns like `if(!(0)) noreturn_call()` in BUILD_BUG macros.
    fn is_constant_true_expr(&self, expr: &Expr) -> bool {
        self.try_eval_constant_bool(expr).unwrap_or_default()
    }

    /// Try to evaluate an expression as a compile-time boolean (true=non-zero, false=zero).
    /// Returns None if the expression is not a compile-time constant.
    fn try_eval_constant_bool(&self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::IntLiteral(val, _) => Some(*val != 0),
            Expr::UIntLiteral(val, _) => Some(*val != 0),
            Expr::LongLiteral(val, _) => Some(*val != 0),
            Expr::ULongLiteral(val, _) => Some(*val != 0),
            Expr::LongLongLiteral(val, _) => Some(*val != 0),
            Expr::ULongLongLiteral(val, _) => Some(*val != 0),
            // !expr: invert the constant
            Expr::UnaryOp(crate::frontend::parser::ast::UnaryOp::LogicalNot, inner, _) => {
                self.try_eval_constant_bool(inner).map(|v| !v)
            }
            // Cast to another type preserves truthiness
            Expr::Cast(_, inner, _) => self.try_eval_constant_bool(inner),
            // Enum constants (e.g., kernel's `true` defined as `enum { false = 0, true = 1 }`)
            Expr::Identifier(name, _) => {
                self.result.type_context.enum_constants.get(name).map(|&val| val != 0)
            }
            _ => None,
        }
    }

    // === Expression analysis ===

    fn analyze_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Identifier(name, span) => {
                // Check if it's a known symbol
                if self.symbol_table.lookup(name).is_none()
                    && !self.result.type_context.enum_constants.contains_key(name)
                    && !builtins::is_builtin(name)
                    && !self.result.functions.contains_key(name)
                    && &**name != "__func__" && &**name != "__FUNCTION__"
                    && &**name != "__PRETTY_FUNCTION__"
                {
                    self.diagnostics.borrow_mut().error(
                        format!("'{}' undeclared", name),
                        *span,
                    );
                }
            }
            Expr::FunctionCall(callee, args, _) => {
                // Check for builtin calls
                if let Expr::Identifier(name, callee_span) = callee.as_ref() {
                    if builtins::is_builtin(name) {
                        // Valid builtin call - will be resolved during lowering
                    } else if !self.result.functions.contains_key(name)
                        && self.symbol_table.lookup(name).is_none()
                    {
                        // Implicit function declaration (C89 style) - register it
                        self.diagnostics.borrow_mut().warning_with_kind(
                            format!("implicit declaration of function '{}'", name),
                            *callee_span,
                            crate::common::error::WarningKind::ImplicitFunctionDeclaration,
                        );
                        let func_info = FunctionInfo {
                            return_type: CType::Int, // implicit return int
                            params: Vec::new(),
                            variadic: true, // unknown params
                            is_defined: false,
                            is_noreturn: false,
                        };
                        self.result.functions.insert(name.clone(), func_info);
                    }
                }
                self.analyze_expr(callee);
                for arg in args {
                    self.analyze_expr(arg);
                }
                // Check for invalid pointer <-> float conversions in arguments
                if let Expr::Identifier(name, _) = callee.as_ref() {
                    if let Some(func_info) = self.result.functions.get(name) {
                        // Clone to release borrow on self.result.functions before
                        // creating ExprTypeChecker which also borrows it
                        let params = func_info.params.clone();
                        let checker = super::type_checker::ExprTypeChecker {
                            symbols: &self.symbol_table,
                            types: &self.result.type_context,
                            functions: &self.result.functions,
                            expr_types: Some(&self.result.expr_types),
                        };
                        for (i, arg) in args.iter().enumerate() {
                            if i < params.len() {
                                if let Some(arg_ty) = checker.infer_expr_ctype(arg) {
                                    self.check_pointer_float_conversion(
                                        &arg_ty, &params[i].0, arg.span(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
            Expr::BinaryOp(op, lhs, rhs, span) => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
                // Check pointer subtraction type compatibility (C11 6.5.6p3):
                // both operands must point to compatible types.
                if *op == BinOp::Sub {
                    self.check_pointer_subtraction_compat(lhs, rhs, *span);
                }
            }
            Expr::UnaryOp(_, operand, _) => {
                self.analyze_expr(operand);
            }
            Expr::PostfixOp(_, operand, _) => {
                self.analyze_expr(operand);
            }
            Expr::Assign(lhs, rhs, span) => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
                // Check for invalid pointer <-> float conversions in assignment
                let checker = super::type_checker::ExprTypeChecker {
                    symbols: &self.symbol_table,
                    types: &self.result.type_context,
                    functions: &self.result.functions,
                    expr_types: Some(&self.result.expr_types),
                };
                if let (Some(lhs_ty), Some(rhs_ty)) = (
                    checker.infer_expr_ctype(lhs),
                    checker.infer_expr_ctype(rhs),
                ) {
                    self.check_pointer_float_conversion(&rhs_ty, &lhs_ty, *span);
                }
            }
            Expr::CompoundAssign(_, lhs, rhs, _) => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
            }
            Expr::Conditional(cond, then_expr, else_expr, _) => {
                self.analyze_expr(cond);
                self.analyze_expr(then_expr);
                self.analyze_expr(else_expr);
            }
            Expr::GnuConditional(cond, else_expr, _) => {
                self.analyze_expr(cond);
                self.analyze_expr(else_expr);
            }
            Expr::ArraySubscript(arr, idx, _) => {
                self.analyze_expr(arr);
                self.analyze_expr(idx);
            }
            Expr::MemberAccess(obj, field_name, span) => {
                self.analyze_expr(obj);
                self.check_member_exists(obj, field_name, *span, false);
            }
            Expr::PointerMemberAccess(obj, field_name, span) => {
                self.analyze_expr(obj);
                self.check_member_exists(obj, field_name, *span, true);
            }
            Expr::Cast(_, inner, _) => {
                self.analyze_expr(inner);
            }
            Expr::Sizeof(arg, span) => {
                // sizeof is unevaluated context, but identifiers must still be declared
                match arg.as_ref() {
                    SizeofArg::Expr(inner) => {
                        self.analyze_expr(inner);
                    }
                    SizeofArg::Type(ts) => {
                        // Check for sizeof on incomplete struct/union types.
                        // sizeof(struct X) where struct X has no definition is an error.
                        // (Pointers to incomplete types are fine.)
                        self.check_sizeof_incomplete_type(ts, *span);
                    }
                }
            }
            Expr::Alignof(..) | Expr::GnuAlignof(..) => {} // alignof(type) - no expr to check
            Expr::AlignofExpr(inner, _) | Expr::GnuAlignofExpr(inner, _) => {
                self.analyze_expr(inner);
            }
            Expr::AddressOf(inner, _) => {
                self.analyze_expr(inner);
            }
            Expr::Deref(inner, _) => {
                self.analyze_expr(inner);
            }
            Expr::Comma(lhs, rhs, _) => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
            }
            Expr::CompoundLiteral(_, init, _) => {
                self.analyze_initializer(init);
            }
            Expr::StmtExpr(compound, _) => {
                self.analyze_compound_stmt(compound);
            }
            Expr::VaArg(ap_expr, _, _) => {
                self.analyze_expr(ap_expr);
            }
            Expr::GenericSelection(controlling, associations, _) => {
                self.analyze_expr(controlling);
                for assoc in associations {
                    self.analyze_expr(&assoc.expr);
                }
            }
            // Literals don't need analysis
            Expr::IntLiteral(_, _)
            | Expr::UIntLiteral(_, _)
            | Expr::LongLiteral(_, _)
            | Expr::ULongLiteral(_, _)
            | Expr::LongLongLiteral(_, _)
            | Expr::ULongLongLiteral(_, _)
            | Expr::FloatLiteral(_, _)
            | Expr::FloatLiteralF32(_, _)
            | Expr::FloatLiteralLongDouble(_, _, _)
            | Expr::ImaginaryLiteral(_, _)
            | Expr::ImaginaryLiteralF32(_, _)
            | Expr::ImaginaryLiteralLongDouble(_, _, _)
            | Expr::StringLiteral(_, _)
            | Expr::WideStringLiteral(_, _)
            | Expr::Char16StringLiteral(_, _)
            | Expr::CharLiteral(_, _) => {}
            // Label address (&&label) - just a compile-time address
            Expr::LabelAddr(_, _) => {}
            // __builtin_types_compatible_p(type1, type2) - compile-time constant
            Expr::BuiltinTypesCompatibleP(_, _, _) => {}
        }

        // After analyzing sub-expressions, infer this expression's CType using
        // the ExprTypeChecker and store it in the annotation map. The lowerer
        // will consult this map as a fast O(1) fallback before doing its own
        // (more expensive) type inference that requires lowering state.
        self.annotate_expr_type(expr);

        // Also try to evaluate this expression as a compile-time constant.
        // The lowerer will consult the const_values map as an O(1) fast path
        // before doing its own eval_const_expr.
        self.annotate_const_value(expr);
    }

    /// Infer the CType of an expression via ExprTypeChecker and record it
    /// in the expr_types annotation map, keyed by the expression's `ExprId`.
    fn annotate_expr_type(&mut self, expr: &Expr) {
        let checker = super::type_checker::ExprTypeChecker {
            symbols: &self.symbol_table,
            types: &self.result.type_context,
            functions: &self.result.functions,
            expr_types: Some(&self.result.expr_types),
        };
        if let Some(ctype) = checker.infer_expr_ctype(expr) {
            self.result.expr_types.insert(expr.id(), ctype);
        }
    }

    /// Try to evaluate an expression as a compile-time constant using
    /// SemaConstEval and record the result in the const_values map.
    /// The lowerer will consult this as an O(1) fast path.
    fn annotate_const_value(&mut self, expr: &Expr) {
        let evaluator = SemaConstEval {
            types: &self.result.type_context,
            symbols: &self.symbol_table,
            functions: &self.result.functions,
            const_values: Some(&self.result.const_values),
            expr_types: Some(&self.result.expr_types),
        };
        if let Some(val) = evaluator.eval_const_expr(expr) {
            self.result.const_values.insert(expr.id(), val);
        }
    }

    /// Check that a struct/union member access refers to an existing field.
    /// Emits an error if the base type is a known struct/union with a complete layout
    /// and the field name is not found.
    fn check_member_exists(&self, base_expr: &Expr, field_name: &str, span: Span, is_pointer: bool) {
        let checker = super::type_checker::ExprTypeChecker {
            symbols: &self.symbol_table,
            types: &self.result.type_context,
            functions: &self.result.functions,
            expr_types: Some(&self.result.expr_types),
        };

        // Get the base type, dereferencing for pointer member access (->)
        let base_ctype = if is_pointer {
            match checker.infer_expr_ctype(base_expr) {
                Some(CType::Pointer(inner, _)) => Some(*inner),
                Some(CType::Array(inner, _)) => Some(*inner),
                _ => None,
            }
        } else {
            checker.infer_expr_ctype(base_expr)
        };

        let base_ctype = match base_ctype {
            Some(ct) => ct,
            None => return, // Can't determine base type; skip check
        };

        let key = match &base_ctype {
            CType::Struct(key) | CType::Union(key) => key.clone(),
            _ => return, // Not a struct/union; skip check
        };

        // If the struct tag has been redefined in an inner scope, the current layout
        // in the map may not match the layout the variable was actually declared with.
        // Skip the check to avoid false positives (codegen uses layout snapshots).
        if self.result.type_context.is_struct_key_shadowed(key.as_ref()) {
            return;
        }

        let layouts = self.result.type_context.borrow_struct_layouts();
        let layout = match layouts.get(key.as_ref()) {
            Some(l) => l,
            None => {
                // No layout at all - struct was never declared or defined.
                // This happens with implicit struct references in casts like
                // ((struct foo *)0)->bar where struct foo is never defined.
                // C11 6.5.2.3: member access requires the struct to be complete.
                if !self.defined_structs.borrow().contains(key.as_ref()) {
                    let type_name = format!("{}", base_ctype);
                    self.diagnostics.borrow_mut().error(
                        format!("invalid use of undefined type '{}'", type_name),
                        span,
                    );
                }
                return;
            }
        };

        // Check if the field exists (including anonymous struct/union members)
        if layout.field_offset(field_name, &*layouts).is_none() {
            let type_name = format!("{}", base_ctype);
            self.diagnostics.borrow_mut().error(
                format!("'{}' has no member named '{}'", type_name, field_name),
                span,
            );
        }
    }

    /// Check that pointer subtraction operands have compatible pointee types.
    /// C11 6.5.6p3: both operands must point to compatible types.
    /// Emits an error if both operands are pointers to different struct/union types.
    fn check_pointer_subtraction_compat(&self, lhs: &Expr, rhs: &Expr, span: Span) {
        let checker = super::type_checker::ExprTypeChecker {
            symbols: &self.symbol_table,
            types: &self.result.type_context,
            functions: &self.result.functions,
            expr_types: Some(&self.result.expr_types),
        };
        let lct = match checker.infer_expr_ctype(lhs) {
            Some(ct) => ct,
            None => return,
        };
        let rct = match checker.infer_expr_ctype(rhs) {
            Some(ct) => ct,
            None => return,
        };
        // Extract pointee types; only check when both sides are pointers
        let (l_inner, r_inner) = match (&lct, &rct) {
            (CType::Pointer(li, _), CType::Pointer(ri, _)) => (li.as_ref(), ri.as_ref()),
            (CType::Array(li, _), CType::Pointer(ri, _)) => (li.as_ref(), ri.as_ref()),
            (CType::Pointer(li, _), CType::Array(ri, _)) => (li.as_ref(), ri.as_ref()),
            _ => return, // Not both pointers; not a ptr-ptr subtraction
        };
        // void* is compatible with any pointer for subtraction purposes
        if matches!(l_inner, CType::Void) || matches!(r_inner, CType::Void) {
            return;
        }
        // Check if pointee types are compatible
        if !Self::pointee_types_compatible(l_inner, r_inner) {
            self.diagnostics.borrow_mut().error(
                format!(
                    "invalid operands to binary - (have '{}' and '{}')",
                    lct, rct
                ),
                span,
            );
        }
    }

    /// Check for invalid implicit conversions between pointer and floating-point types.
    /// C11 6.5.16.1p1: pointer types cannot be implicitly converted to/from floating-point types.
    /// This catches cases like `float f = ptr;`, `func_taking_float(ptr)`, etc.
    fn check_pointer_float_conversion(&self, from_ty: &CType, to_ty: &CType, span: Span) {
        let from_is_ptr = from_ty.is_pointer_like() || matches!(from_ty, CType::Function(_));
        let to_is_ptr = to_ty.is_pointer_like() || matches!(to_ty, CType::Function(_));
        let from_is_float = from_ty.is_floating() || from_ty.is_complex();
        let to_is_float = to_ty.is_floating() || to_ty.is_complex();

        if (from_is_ptr && to_is_float) || (from_is_float && to_is_ptr) {
            self.diagnostics.borrow_mut().error(
                format!(
                    "incompatible types (have '{}' but expected '{}')",
                    from_ty, to_ty
                ),
                span,
            );
        }
    }

    /// Check if two pointee types are compatible for pointer arithmetic.
    /// This is a simplified check: types are compatible if they are structurally
    /// the same (ignoring qualifiers at the top level).
    fn pointee_types_compatible(a: &CType, b: &CType) -> bool {
        match (a, b) {
            // Same struct/union name -> compatible
            (CType::Struct(ka), CType::Struct(kb)) => ka == kb,
            (CType::Union(ka), CType::Union(kb)) => ka == kb,
            // Pointers: recurse on pointee types
            (CType::Pointer(pa, _), CType::Pointer(pb, _)) => {
                Self::pointee_types_compatible(pa, pb)
            }
            // Arrays: compare element types
            (CType::Array(ea, _), CType::Array(eb, _)) => {
                Self::pointee_types_compatible(ea, eb)
            }
            // Function types: just check broad compatibility
            (CType::Function(_), CType::Function(_)) => {
                // Simplification: treat all function pointers as compatible
                true
            }
            // Enum types are integer types in C (C11 6.7.2.2p4), so enum pointers
            // are compatible with integer pointers of the same size for pointer
            // arithmetic. This is needed for QEMU's VMSTATE_UINT32 macro pattern:
            // (uint32_t*)0 - (typeof(enum_field)*)0
            (CType::Enum(e), other) | (other, CType::Enum(e)) if other.is_integer() => {
                e.packed_size() == other.size()
            }
            // For basic types, use equality (ignoring qualifiers)
            _ => a == b,
        }
    }

    fn analyze_initializer(&mut self, init: &Initializer) {
        match init {
            Initializer::Expr(expr) => {
                self.analyze_expr(expr);
            }
            Initializer::List(items) => {
                for item in items {
                    self.analyze_initializer(&item.init);
                }
            }
        }
    }

    /// Check if a sizeof type argument refers to an incomplete struct/union.
    /// Reports an error like GCC: "invalid application of 'sizeof' to incomplete type"
    fn check_sizeof_incomplete_type(&self, ts: &TypeSpecifier, span: Span) {
        match ts {
            TypeSpecifier::Struct(Some(tag), None, _, _, _) => {
                // Reference to a named struct with no inline body.
                // Check if it was previously defined (key format: "struct.tag").
                let key = format!("struct.{}", tag);
                if !self.defined_structs.borrow().contains(key.as_str()) {
                    self.diagnostics.borrow_mut().error(
                        format!("storage size of 'struct {}' isn't known", tag),
                        span,
                    );
                }
            }
            TypeSpecifier::Union(Some(tag), None, _, _, _) => {
                let key = format!("union.{}", tag);
                if !self.defined_structs.borrow().contains(key.as_str()) {
                    self.diagnostics.borrow_mut().error(
                        format!("storage size of 'union {}' isn't known", tag),
                        span,
                    );
                }
            }
            // Pointers to incomplete types are fine - sizeof gives pointer size
            TypeSpecifier::Pointer(_, _) => {}
            // Arrays of incomplete element types - check the element type
            TypeSpecifier::Array(elem, _) => {
                self.check_sizeof_incomplete_type(elem, span);
            }
            _ => {}
        }
    }

    // === Type conversion utilities ===

    /// Convert an AST TypeSpecifier to a CType.
    /// Delegates to the shared `TypeConvertContext::resolve_type_spec_to_ctype` default
    /// method, which handles all 22 primitive types and delegates struct/union/enum/typedef
    /// to sema-specific trait methods.
    fn type_spec_to_ctype(&self, spec: &TypeSpecifier) -> CType {
        use crate::common::type_builder::TypeConvertContext;
        self.resolve_type_spec_to_ctype(spec)
    }

    fn convert_struct_fields(&self, fields: &[StructFieldDecl]) -> Vec<crate::common::types::StructField> {
        fields.iter().map(|f| {
            let ty = if f.derived.is_empty() {
                self.type_spec_to_ctype(&f.type_spec)
            } else {
                // Delegate to shared build_full_ctype for correct inside-out type construction
                type_builder::build_full_ctype(self, &f.type_spec, &f.derived)
            };
            let name = f.name.clone().unwrap_or_default();
            let bit_width = f.bit_width.as_ref().and_then(|bw| {
                self.eval_const_expr(bw).map(|v| v as u32)
            });
            // Merge per-field alignment with typedef alignment.
            // If the field's type is a typedef with __aligned__, that alignment
            // must be applied even when the field itself has no explicit alignment.
            let field_alignment = {
                let mut align = f.alignment;
                // Check if the field type is a typedef with an alignment override
                let typedef_align = self.resolve_typedef_alignment(&f.type_spec);
                if let Some(ta) = typedef_align {
                    align = Some(align.map_or(ta, |a| a.max(ta)));
                }
                align
            };
            crate::common::types::StructField {
                name,
                ty,
                bit_width,
                alignment: field_alignment,
                is_packed: f.is_packed,
            }
        }).collect()
    }

    /// Resolve the alignment override carried by a typedef, if any.
    /// For a `TypeSpecifier::TypedefName("foo")`, looks up `foo` in
    /// `typedef_alignments`.  Returns `None` when the type specifier is not
    /// a typedef name or the typedef has no alignment attribute.
    fn resolve_typedef_alignment(&self, ts: &TypeSpecifier) -> Option<usize> {
        if let TypeSpecifier::TypedefName(name) = ts {
            self.result.type_context.typedef_alignments.get(name).copied()
        } else {
            None
        }
    }

    // === Constant expression evaluation ===

    /// Try to evaluate a constant expression at compile time.
    /// Returns None if the expression cannot be evaluated.
    ///
    /// This delegates to the richer SemaConstEval which returns IrConst,
    /// then extracts the i64 value. This ensures enum values, bitfield
    /// widths, and array sizes all use the same evaluation logic.
    fn eval_const_expr(&self, expr: &Expr) -> Option<i64> {
        let evaluator = SemaConstEval {
            types: &self.result.type_context,
            symbols: &self.symbol_table,
            functions: &self.result.functions,
            const_values: Some(&self.result.const_values),
            expr_types: Some(&self.result.expr_types),
        };
        evaluator.eval_const_expr(expr)?.to_i64()
    }

    // === Implicit declarations ===

    /// Pre-declare common implicit functions that C programs expect.
    fn declare_implicit_functions(&mut self) {
        // Common libc functions that tests may use without headers
        let implicit_funcs = [
            ("printf", CType::Int, true),
            ("fprintf", CType::Int, true),
            ("sprintf", CType::Int, true),
            ("snprintf", CType::Int, true),
            ("puts", CType::Int, false),
            ("putchar", CType::Int, false),
            ("getchar", CType::Int, false),
            ("malloc", CType::Pointer(Box::new(CType::Void), AddressSpace::Default), false),
            ("calloc", CType::Pointer(Box::new(CType::Void), AddressSpace::Default), false),
            ("realloc", CType::Pointer(Box::new(CType::Void), AddressSpace::Default), false),
            ("free", CType::Void, false),
            ("exit", CType::Void, false),
            ("abort", CType::Void, false),
            ("memcpy", CType::Pointer(Box::new(CType::Void), AddressSpace::Default), false),
            ("memmove", CType::Pointer(Box::new(CType::Void), AddressSpace::Default), false),
            ("memset", CType::Pointer(Box::new(CType::Void), AddressSpace::Default), false),
            ("memcmp", CType::Int, false),
            ("strlen", CType::ULong, false),
            ("strcmp", CType::Int, false),
            ("strcpy", CType::Pointer(Box::new(CType::Char), AddressSpace::Default), false),
            ("atoi", CType::Int, false),
            ("atol", CType::Long, false),
            ("abs", CType::Int, false),
        ];

        for (name, ret_type, variadic) in &implicit_funcs {
            let is_noreturn = matches!(*name, "exit" | "abort" | "_Exit" | "quick_exit");
            let func_info = FunctionInfo {
                return_type: ret_type.clone(),
                params: Vec::new(),
                variadic: *variadic,
                is_defined: false,
                is_noreturn,
            };
            self.result.functions.insert(Rc::from(*name), func_info);
        }
    }
}

/// Implement TypeConvertContext so shared type_builder functions can call back
/// into sema for type resolution and constant expression evaluation.
///
/// The 4 divergent methods handle sema-specific behavior:
/// - typedef: looks up in type_context.typedefs
/// - struct/union: converts fields and computes layout
/// - enum: returns CType::Enum with name info (preserves enum identity)
/// - typeof: returns CType::Int (sema doesn't have full expr type resolution yet)
impl type_builder::TypeConvertContext for SemanticAnalyzer {
    fn resolve_typedef(&self, name: &str) -> CType {
        if let Some(resolved) = self.result.type_context.typedefs.get(name) {
            resolved.clone()
        } else {
            CType::Int
        }
    }

    fn resolve_struct_or_union(
        &self,
        name: &Option<Rc<str>>,
        fields: &Option<Vec<StructFieldDecl>>,
        is_union: bool,
        is_packed: bool,
        pragma_pack: Option<usize>,
        struct_aligned: Option<usize>,
    ) -> CType {
        let prefix = if is_union { "union" } else { "struct" };
        let struct_fields = fields.as_ref().map(|f| self.convert_struct_fields(f)).unwrap_or_default();
        let max_field_align = if is_packed { Some(1) } else { pragma_pack };
        let key = if let Some(tag) = name {
            format!("{}.{}", prefix, tag)
        } else {
            let id = self.result.type_context.next_anon_struct_id();
            format!("__anon_struct_{}", id)
        };
        // Track whether this struct/union has been defined (has a body in the
        // AST, e.g. `struct X { ... }` or `struct X {}`), as opposed to just
        // forward-declared (`struct X;`). This distinction is needed for the
        // incomplete type check in analyze_declaration.
        if fields.is_some() {
            self.defined_structs.borrow_mut().insert(Rc::from(key.as_str()));
        }
        if !struct_fields.is_empty() {
            let mut layout = if is_union {
                StructLayout::for_union_with_packing(&struct_fields, max_field_align, &*self.result.type_context.borrow_struct_layouts())
            } else {
                StructLayout::for_struct_with_packing(
                    &struct_fields, max_field_align, &*self.result.type_context.borrow_struct_layouts()
                )
            };
            if let Some(a) = struct_aligned {
                if a > layout.align {
                    layout.align = a;
                    let mask = layout.align - 1;
                    layout.size = (layout.size + mask) & !mask;
                }
            }
            self.result.type_context.insert_struct_layout_scoped_from_ref(&key, layout);
        } else if self.result.type_context.borrow_struct_layouts().get(key.as_str()).is_none() {
            let align = struct_aligned.unwrap_or(1);
            let layout = StructLayout {
                fields: Vec::new(),
                size: 0,
                align,
                is_union,
                is_transparent_union: false,
            };
            self.result.type_context.insert_struct_layout_scoped_from_ref(&key, layout);
        }
        if is_union { CType::Union(key.into()) } else { CType::Struct(key.into()) }
    }

    fn resolve_enum(&self, name: &Option<Rc<str>>, variants: &Option<Vec<EnumVariant>>, is_packed: bool) -> CType {
        // Check if this is a forward reference to a previously-defined packed enum
        let effective_packed = is_packed || name.as_ref()
            .and_then(|n| self.result.type_context.packed_enum_types.get(&**n))
            .is_some();
        // Sema preserves enum identity for diagnostics. Variant processing is
        // done separately via process_enum_variants (requires &mut self).
        // We carry variant values so packed_size() can compute the correct size.
        let variant_values = if let Some(vars) = variants {
            let mut result = Vec::new();
            let mut next_val: i64 = 0;
            for v in vars {
                if let Some(ref val_expr) = v.value {
                    if let Some(val) = self.eval_const_expr(val_expr) {
                        next_val = val;
                    }
                }
                result.push((v.name.clone(), next_val));
                next_val += 1;
            }
            result
        } else if let Some(n) = name {
            // Forward reference: look up previously stored packed enum info
            if let Some(et) = self.result.type_context.packed_enum_types.get(n) {
                et.variants.clone()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        CType::Enum(crate::common::types::EnumType {
            name: name.clone(),
            variants: variant_values,
            is_packed: effective_packed,
        })
    }

    fn resolve_typeof_expr(&self, expr: &Expr) -> CType {
        // Use the ExprTypeChecker to infer typeof(expr) from sema state.
        let checker = super::type_checker::ExprTypeChecker {
            symbols: &self.symbol_table,
            types: &self.result.type_context,
            functions: &self.result.functions,
            expr_types: Some(&self.result.expr_types),
        };
        checker.infer_expr_ctype(expr).unwrap_or(CType::Int)
    }

    fn eval_const_expr_as_usize(&self, expr: &Expr) -> Option<usize> {
        self.eval_const_expr(expr).and_then(|v| {
            if v < 0 {
                // C standard requires array sizes to be positive (constraint violation).
                // Critical for autoconf AC_CHECK_SIZEOF which uses negative array sizes
                // as compile-time assertions to detect type sizes during cross-compilation.
                self.diagnostics.borrow_mut().error(
                    "size of array is negative",
                    expr.span(),
                );
                None
            } else {
                Some(v as usize)
            }
        })
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
