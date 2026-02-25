//! Dead static function elimination via reference collection.
//!
//! This module implements dead static function elimination by walking AST compound
//! statements to collect all referenced function names. Functions that are declared
//! as static but never referenced (transitively) from any non-static function body
//! or global initializer can be safely skipped during lowering, providing a significant
//! performance win for translation units that include many header-defined static
//! or inline functions.

use std::collections::VecDeque;
use std::rc::Rc;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::frontend::parser::ast::{
    BlockItem,
    CompoundStmt,
    Expr,
    ExternalDecl,
    ForInit,
    FunctionDef,
    Initializer,
    Stmt,
    TranslationUnit,
};
use super::lower::Lowerer;

impl Lowerer {
    /// Returns true if a function definition can be skipped when unreferenced.
    /// This mirrors the skip logic in `lower()` — static, static inline,
    /// extern inline with gnu_inline attribute or in GNU89 mode, and C99
    /// inline-only definitions can all be skipped.
    fn is_skippable_function(&self, func: &FunctionDef) -> bool {
        let is_gnu_inline_no_extern_def = self.is_gnu_inline_no_extern_def(&func.attrs);
        // C99 plain `inline` (without `extern` or `static`) does not provide
        // an external definition. These are lowered as static when referenced,
        // so they can be skipped when unreferenced.
        // Per C99 6.7.4p7: unless there's a non-inline declaration of this function.
        // Note: in GNU89 mode, `inline` without `extern` provides an external def,
        // so this rule does not apply.
        let is_c99_inline_only = !self.gnu89_inline
            && func.attrs.is_inline() && !func.attrs.is_extern()
            && !func.attrs.is_static() && !func.attrs.is_gnu_inline()
            && !self.has_non_inline_decl.contains(&func.name);
        func.attrs.is_static() || is_gnu_inline_no_extern_def || is_c99_inline_only
    }

    /// Collect all function names that are transitively referenced from root
    /// (non-skippable) function bodies and global initializers.
    ///
    /// Algorithm:
    /// 1. Collect direct references from root functions and global initializers
    /// 2. Build a per-function reference map for skippable functions
    /// 3. Worklist-based transitive closure: when a skippable function becomes
    ///    reachable, add its references to the worklist
    pub(super) fn collect_referenced_static_functions(&self, tu: &TranslationUnit) -> FxHashSet<Rc<str>> {
        let mut referenced = FxHashSet::default();
        // Map from skippable function name -> set of functions it references
        let mut skippable_refs: FxHashMap<Rc<str>, FxHashSet<Rc<str>>> = FxHashMap::default();

        for decl in &tu.decls {
            match decl {
                ExternalDecl::FunctionDef(func) => {
                    if self.is_skippable_function(func) {
                        // For skippable functions, collect their refs into a separate map
                        // so we can do transitive closure later
                        let mut func_refs = FxHashSet::default();
                        self.collect_refs_from_compound(&func.body, &mut func_refs);
                        skippable_refs.insert(func.name.clone(), func_refs);
                    } else {
                        // Root function: collect refs directly into the referenced set
                        self.collect_refs_from_compound(&func.body, &mut referenced);
                    }
                }
                ExternalDecl::Declaration(decl) => {
                    // Check global variable initializers for function references
                    for declarator in &decl.declarators {
                        if let Some(ref init) = declarator.init {
                            self.collect_refs_from_initializer(init, &mut referenced);
                        }
                        // Alias targets reference the aliased function
                        if let Some(ref target) = declarator.attrs.alias_target {
                            referenced.insert(Rc::from(target.as_str()));
                        }
                    }
                }
                ExternalDecl::TopLevelAsm(_) => {
                    // Top-level asm doesn't reference C functions
                }
            }
        }

        // Constructor and destructor functions are roots: they will be called by
        // the runtime via .init_array/.fini_array, so any functions they reference
        // must also be preserved. The constructor/destructor names were collected
        // into module.constructors/destructors in an earlier pass.
        for ctor in &self.module.constructors {
            referenced.insert(ctor.clone());
        }
        for dtor in &self.module.destructors {
            referenced.insert(dtor.clone());
        }

        // Transitive closure: use a worklist to propagate reachability through
        // skippable functions. When a skippable function is found to be referenced,
        // add all of its own references to the worklist.
        let mut worklist: VecDeque<Rc<str>> = referenced.iter().cloned().collect();
        while let Some(name) = worklist.pop_front() {
            if let Some(func_refs) = skippable_refs.get(&name) {
                for r in func_refs {
                    if referenced.insert(r.clone()) {
                        worklist.push_back(r.clone());
                    }
                }
            }
        }

        referenced
    }

    /// Collect function name references from a compound statement.
    pub(super) fn collect_refs_from_compound(&self, compound: &CompoundStmt, refs: &mut FxHashSet<Rc<str>>) {
        for item in &compound.items {
            match item {
                BlockItem::Declaration(decl) => {
                    for declarator in &decl.declarators {
                        if let Some(ref init) = declarator.init {
                            self.collect_refs_from_initializer(init, refs);
                        }
                        // __attribute__((cleanup(func))) references func
                        if let Some(ref cleanup_fn) = declarator.attrs.cleanup_fn {
                            if self.known_functions.contains(&**cleanup_fn) {
                                refs.insert(Rc::from(&**cleanup_fn));
                            }
                        }
                    }
                }
                BlockItem::Statement(stmt) => {
                    self.collect_refs_from_stmt(stmt, refs);
                }
            }
        }
    }

    /// Collect function name references from a statement.
    pub(super) fn collect_refs_from_stmt(&self, stmt: &Stmt, refs: &mut FxHashSet<Rc<str>>) {
        match stmt {
            Stmt::Expr(Some(expr)) => {
                self.collect_refs_from_expr(expr, refs);
            }
            Stmt::Return(Some(expr), _) => {
                self.collect_refs_from_expr(expr, refs);
            }
            Stmt::Compound(compound) => {
                self.collect_refs_from_compound(compound, refs);
            }
            Stmt::If(cond, then_s, else_s, _) => {
                self.collect_refs_from_expr(cond, refs);
                self.collect_refs_from_stmt(then_s, refs);
                if let Some(e) = else_s {
                    self.collect_refs_from_stmt(e, refs);
                }
            }
            Stmt::While(cond, body, _) => {
                self.collect_refs_from_expr(cond, refs);
                self.collect_refs_from_stmt(body, refs);
            }
            Stmt::DoWhile(body, cond, _) => {
                self.collect_refs_from_stmt(body, refs);
                self.collect_refs_from_expr(cond, refs);
            }
            Stmt::For(init, cond, inc, body, _) => {
                if let Some(init) = init {
                    match init.as_ref() {
                        ForInit::Expr(e) => self.collect_refs_from_expr(e, refs),
                        ForInit::Declaration(d) => {
                            for declarator in &d.declarators {
                                if let Some(ref init) = declarator.init {
                                    self.collect_refs_from_initializer(init, refs);
                                }
                                // __attribute__((cleanup(func))) references func
                                if let Some(ref cleanup_fn) = declarator.attrs.cleanup_fn {
                                    if self.known_functions.contains(&**cleanup_fn) {
                                        refs.insert(Rc::from(&**cleanup_fn));
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(c) = cond { self.collect_refs_from_expr(c, refs); }
                if let Some(i) = inc { self.collect_refs_from_expr(i, refs); }
                self.collect_refs_from_stmt(body, refs);
            }
            Stmt::Switch(expr, body, _) => {
                self.collect_refs_from_expr(expr, refs);
                self.collect_refs_from_stmt(body, refs);
            }
            Stmt::Case(expr, stmt, _) => {
                self.collect_refs_from_expr(expr, refs);
                self.collect_refs_from_stmt(stmt, refs);
            }
            Stmt::CaseRange(low_expr, high_expr, stmt, _) => {
                self.collect_refs_from_expr(low_expr, refs);
                self.collect_refs_from_expr(high_expr, refs);
                self.collect_refs_from_stmt(stmt, refs);
            }
            Stmt::Default(stmt, _) | Stmt::Label(_, stmt, _) => {
                self.collect_refs_from_stmt(stmt, refs);
            }
            Stmt::Declaration(decl) => {
                for declarator in &decl.declarators {
                    if let Some(ref init) = declarator.init {
                        self.collect_refs_from_initializer(init, refs);
                    }
                    if let Some(ref cleanup_fn) = declarator.attrs.cleanup_fn {
                        if self.known_functions.contains(&**cleanup_fn) {
                            refs.insert(Rc::from(&**cleanup_fn));
                        }
                    }
                }
            }
            Stmt::GotoIndirect(expr, _) => {
                self.collect_refs_from_expr(expr, refs);
            }
            Stmt::InlineAsm { inputs, outputs, .. } => {
                for op in inputs.iter().chain(outputs.iter()) {
                    self.collect_refs_from_expr(&op.expr, refs);
                }
            }
            _ => {}
        }
    }

    /// Collect function name references from an expression.
    pub(super) fn collect_refs_from_expr(&self, expr: &Expr, refs: &mut FxHashSet<Rc<str>>) {
        match expr {
            Expr::Identifier(name, _) => {
                if self.known_functions.contains(name) {
                    refs.insert(name.clone());
                }
            }
            Expr::FunctionCall(callee, args, _) => {
                self.collect_refs_from_expr(callee, refs);
                for arg in args { self.collect_refs_from_expr(arg, refs); }
            }
            Expr::BinaryOp(_, lhs, rhs, _) | Expr::Assign(lhs, rhs, _)
            | Expr::CompoundAssign(_, lhs, rhs, _) => {
                self.collect_refs_from_expr(lhs, refs);
                self.collect_refs_from_expr(rhs, refs);
            }
            Expr::UnaryOp(_, operand, _) | Expr::PostfixOp(_, operand, _)
            | Expr::Cast(_, operand, _) => {
                self.collect_refs_from_expr(operand, refs);
            }
            Expr::Conditional(c, t, f, _) => {
                self.collect_refs_from_expr(c, refs);
                self.collect_refs_from_expr(t, refs);
                self.collect_refs_from_expr(f, refs);
            }
            Expr::GnuConditional(c, f, _) => {
                self.collect_refs_from_expr(c, refs);
                self.collect_refs_from_expr(f, refs);
            }
            Expr::Comma(lhs, rhs, _) => {
                self.collect_refs_from_expr(lhs, refs);
                self.collect_refs_from_expr(rhs, refs);
            }
            Expr::MemberAccess(base, _, _) | Expr::PointerMemberAccess(base, _, _) => {
                self.collect_refs_from_expr(base, refs);
            }
            Expr::ArraySubscript(base, idx, _) => {
                self.collect_refs_from_expr(base, refs);
                self.collect_refs_from_expr(idx, refs);
            }
            Expr::Deref(inner, _) | Expr::AddressOf(inner, _) | Expr::VaArg(inner, _, _) => {
                self.collect_refs_from_expr(inner, refs);
            }
            Expr::CompoundLiteral(_, init, _) => {
                self.collect_refs_from_initializer(init, refs);
            }
            Expr::StmtExpr(compound, _) => {
                self.collect_refs_from_compound(compound, refs);
            }
            Expr::GenericSelection(ctrl, assocs, _) => {
                self.collect_refs_from_expr(ctrl, refs);
                for assoc in assocs {
                    self.collect_refs_from_expr(&assoc.expr, refs);
                }
            }
            _ => {}
        }
    }

    /// Collect function name references from an initializer.
    pub(super) fn collect_refs_from_initializer(&self, init: &Initializer, refs: &mut FxHashSet<Rc<str>>) {
        match init {
            Initializer::Expr(e) => self.collect_refs_from_expr(e, refs),
            Initializer::List(items) => {
                for item in items {
                    self.collect_refs_from_initializer(&item.init, refs);
                }
            }
        }
    }
}
