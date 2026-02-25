//! Control flow statement lowering: if/else, loops (while/for/do-while),
//! break/continue, goto (direct and computed), and labels.

use std::rc::Rc;
use crate::frontend::parser::ast::{
    Expr,
    ForInit,
    Stmt,
};
use crate::ir::reexports::{BlockId, Instruction, Terminator};
use super::lower::Lowerer;

impl Lowerer {
    pub(super) fn lower_if_stmt(&mut self, cond: &Expr, then_stmt: &Stmt, else_stmt: Option<&Stmt>) {
        let cond_val = self.lower_condition_expr(cond);
        let then_label = self.fresh_label();
        let else_label = self.fresh_label();
        let end_label = self.fresh_label();

        let false_target = if else_stmt.is_some() { else_label } else { end_label };
        self.terminate(Terminator::CondBranch {
            cond: cond_val,
            true_label: then_label,
            false_label: false_target,
        });

        self.start_block(then_label);
        self.lower_stmt(then_stmt);
        self.terminate(Terminator::Branch(end_label));

        if let Some(else_stmt) = else_stmt {
            self.start_block(else_label);
            self.lower_stmt(else_stmt);
            self.terminate(Terminator::Branch(end_label));
        }

        self.start_block(end_label);
    }

    pub(super) fn lower_while_stmt(&mut self, cond: &Expr, body: &Stmt) {
        let cond_label = self.fresh_label();
        let body_label = self.fresh_label();
        let end_label = self.fresh_label();

        let scope_depth = self.func().scope_stack.len();
        self.func_mut().break_labels.push((end_label, scope_depth));
        self.func_mut().continue_labels.push((cond_label, scope_depth));

        self.terminate(Terminator::Branch(cond_label));

        self.start_block(cond_label);
        let cond_val = self.lower_condition_expr(cond);
        self.terminate(Terminator::CondBranch {
            cond: cond_val,
            true_label: body_label,
            false_label: end_label,
        });

        self.start_block(body_label);
        self.lower_stmt(body);
        let continue_target = self.func().continue_labels.last()
            .expect("continue_labels must be non-empty inside a loop body").0;
        self.terminate(Terminator::Branch(continue_target));

        self.func_mut().break_labels.pop();
        self.func_mut().continue_labels.pop();
        self.start_block(end_label);
    }

    pub(super) fn lower_for_stmt(
        &mut self,
        init: &Option<Box<ForInit>>,
        cond: &Option<Expr>,
        inc: &Option<Expr>,
        body: &Stmt,
    ) {
        // C99: for-init declarations have their own scope.
        let has_decl_init = init.as_ref().is_some_and(|i| matches!(i.as_ref(), ForInit::Declaration(_)));
        if has_decl_init {
            self.push_scope();
        }

        // Init
        if let Some(init) = init {
            match init.as_ref() {
                ForInit::Declaration(decl) => {
                    self.collect_enum_constants_scoped(&decl.type_spec);
                    self.lower_local_decl(decl);
                }
                ForInit::Expr(expr) => { self.lower_expr(expr); },
            }
        }

        let cond_label = self.fresh_label();
        let body_label = self.fresh_label();
        let inc_label = self.fresh_label();
        let end_label = self.fresh_label();

        let scope_depth = self.func().scope_stack.len();
        self.func_mut().break_labels.push((end_label, scope_depth));
        self.func_mut().continue_labels.push((inc_label, scope_depth));

        self.terminate(Terminator::Branch(cond_label));

        // Condition
        self.start_block(cond_label);
        if let Some(cond) = cond {
            let cond_val = self.lower_condition_expr(cond);
            self.terminate(Terminator::CondBranch {
                cond: cond_val,
                true_label: body_label,
                false_label: end_label,
            });
        } else {
            self.terminate(Terminator::Branch(body_label));
        }

        // Body
        self.start_block(body_label);
        self.lower_stmt(body);
        self.terminate(Terminator::Branch(inc_label));

        // Increment
        self.start_block(inc_label);
        if let Some(inc) = inc {
            self.lower_expr(inc);
        }
        self.terminate(Terminator::Branch(cond_label));

        self.func_mut().break_labels.pop();
        self.func_mut().continue_labels.pop();
        self.start_block(end_label);

        if has_decl_init {
            self.pop_scope();
        }
    }

    pub(super) fn lower_do_while_stmt(&mut self, body: &Stmt, cond: &Expr) {
        let body_label = self.fresh_label();
        let cond_label = self.fresh_label();
        let end_label = self.fresh_label();

        let scope_depth = self.func().scope_stack.len();
        self.func_mut().break_labels.push((end_label, scope_depth));
        self.func_mut().continue_labels.push((cond_label, scope_depth));

        self.terminate(Terminator::Branch(body_label));

        self.start_block(body_label);
        self.lower_stmt(body);
        self.terminate(Terminator::Branch(cond_label));

        self.start_block(cond_label);
        let cond_val = self.lower_condition_expr(cond);
        self.terminate(Terminator::CondBranch {
            cond: cond_val,
            true_label: body_label,
            false_label: end_label,
        });

        self.func_mut().break_labels.pop();
        self.func_mut().continue_labels.pop();
        self.start_block(end_label);
    }

    pub(super) fn lower_break_stmt(&mut self) {
        if let Some(&(label, scope_depth)) = self.func().break_labels.last() {
            // Emit cleanup calls for all scopes being exited by break
            let cleanups = self.collect_scope_cleanup_vars_above_depth(scope_depth);
            self.emit_cleanup_calls(&cleanups);
            // Restore the stack pointer for VLA deallocation when breaking out
            // of scopes that contain VLAs. Without this, VLA stack space leaks.
            self.emit_vla_restore_for_scope_exit(scope_depth);
            self.terminate(Terminator::Branch(label));
            let dead = self.fresh_label();
            self.start_block(dead);
        }
    }

    pub(super) fn lower_continue_stmt(&mut self) {
        if let Some(&(label, scope_depth)) = self.func().continue_labels.last() {
            // Emit cleanup calls for all scopes being exited by continue
            let cleanups = self.collect_scope_cleanup_vars_above_depth(scope_depth);
            self.emit_cleanup_calls(&cleanups);
            // Restore the stack pointer for VLA deallocation when continuing
            // past scopes that contain VLAs. Without this, VLA stack space leaks
            // on each loop iteration, causing stack overflow.
            self.emit_vla_restore_for_scope_exit(scope_depth);
            self.terminate(Terminator::Branch(label));
            let dead = self.fresh_label();
            self.start_block(dead);
        }
    }

    /// Emit StackRestore for VLA deallocation when exiting scopes via break/continue.
    ///
    /// Checks all scopes from `target_depth` to the current depth for VLA stack saves.
    /// If any exited scope has a saved stack pointer (from VLA allocation), restores
    /// the outermost one to reclaim all VLA stack space in the exited scopes.
    fn emit_vla_restore_for_scope_exit(&mut self, target_depth: usize) {
        let current_depth = self.func().scope_stack.len();
        if target_depth < current_depth {
            // Exiting one or more scopes. Find the outermost exited scope with VLAs
            // and restore from its save point (this reclaims all VLA space in the
            // exited scopes since their VLAs were allocated after that save point).
            let vla_restore = self.func().scope_stack[target_depth..]
                .iter()
                .find_map(|frame| frame.scope_stack_save);
            if let Some(save_val) = vla_restore {
                self.emit(Instruction::StackRestore { ptr: save_val });
            }
        }
    }

    pub(super) fn lower_goto_stmt(&mut self, label: &str) {
        // Determine the target label's scope depth (populated by prescan_label_depths).
        // Only emit cleanup calls for scopes being exited by the goto, i.e. scopes
        // deeper than the target label's scope depth.
        let current_depth = self.func().scope_stack.len();
        let target_depth = self.func().user_label_depths
            .get(label)
            .copied()
            // Fallback: if the label depth is unknown (shouldn't happen after prescan),
            // conservatively clean up all scopes.
            .unwrap_or(0);
        // Clamp target_depth to the current scope depth. If the goto jumps into a
        // deeper or same-level scope (target_depth >= current_depth), no cleanup is
        // needed. If jumping out to a shallower scope, clean up the exited scopes.
        let cleanup_depth = target_depth.min(current_depth);
        let cleanups = self.collect_scope_cleanup_vars_above_depth(cleanup_depth);
        self.emit_cleanup_calls(&cleanups);

        // Restore the stack pointer for VLA deallocation when needed.
        //
        // There are two cases where VLA stack space must be reclaimed:
        // 1. The goto exits one or more scopes that contain VLAs — restore using
        //    the outermost exited scope's stack save point.
        // 2. The goto jumps backward within the same scope past a VLA declaration
        //    (e.g., a goto-based loop around a VLA). Without restoration, the VLA
        //    would be re-allocated on each iteration, causing stack overflow.
        //
        // Forward gotos within the same scope must NOT restore, because the VLA
        // data may still be live (pointers into it may be used after the label).
        // We distinguish forward from backward gotos by checking whether the
        // target label has already been defined during lowering.
        if cleanup_depth < current_depth {
            // Case 1: exiting scopes — use the shared helper to restore VLA stack.
            self.emit_vla_restore_for_scope_exit(cleanup_depth);
        } else if let Some(save_val) = self.func().vla_stack_save {
            // Case 2: same-scope goto. Only restore for backward jumps (label
            // already defined), which may re-enter VLA declarations.
            if self.user_label_exists(label) {
                self.emit(Instruction::StackRestore { ptr: save_val });
            }
        }
        let scoped_label = self.get_or_create_user_label(label);
        self.terminate(Terminator::Branch(scoped_label));
        let dead = self.fresh_label();
        self.start_block(dead);
    }

    pub(super) fn lower_goto_indirect_stmt(&mut self, expr: &Expr) {
        // Emit cleanup calls for all active scopes before indirect jump (same as goto).
        let all_cleanups = self.collect_all_scope_cleanup_vars();
        self.emit_cleanup_calls(&all_cleanups);
        // If the function has VLA declarations, restore the saved stack pointer before
        // indirect jumps too (computed gotos).
        if let Some(save_val) = self.func().vla_stack_save {
            self.emit(Instruction::StackRestore { ptr: save_val });
        }
        let target = self.lower_expr(expr);
        let possible_targets: Vec<BlockId> = self.func_mut().user_labels.values().copied().collect();
        self.terminate(Terminator::IndirectBranch { target, possible_targets });
        let dead = self.fresh_label();
        self.start_block(dead);
    }

    pub(super) fn lower_label_stmt(&mut self, name: &str, stmt: &Stmt) {
        let label = self.get_or_create_user_label(name);
        // Mark this label as defined (the label: statement has been lowered).
        // This distinguishes it from labels merely referenced by a forward goto.
        let resolved_name = self.resolve_local_label(name);
        let func_name = self.func().name.clone();
        let key: Rc<str> = Rc::from(format!("{}::{}", func_name, resolved_name));
        self.func_mut().defined_user_labels.insert(key);
        self.terminate(Terminator::Branch(label));
        self.start_block(label);
        self.lower_stmt(stmt);
    }
}
