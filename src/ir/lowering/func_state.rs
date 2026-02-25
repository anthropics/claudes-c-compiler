//! Per-function build state for IR lowering.
//!
//! `FunctionBuildState` holds all state that is created fresh for each function
//! being lowered and discarded afterward. This makes the function-vs-module
//! lifecycle boundary explicit: the Lowerer wraps this in `Option<FunctionBuildState>`,
//! which is `None` between functions.
//!
//! Scope management uses an undo-log pattern (`FuncScopeFrame`) rather than
//! cloning entire HashMaps at scope boundaries. On scope exit, only the changes
//! made within that scope are undone, giving O(changes) cost instead of O(total).

use std::rc::Rc;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::source::Span;
use crate::ir::reexports::{
    BasicBlock,
    BlockId,
    Instruction,
    Value,
};
use crate::common::types::{IrType, CType};
use super::definitions::{LocalInfo, SwitchFrame};

/// Records undo operations for function-local scoped variables.
///
/// Pushed on scope entry, popped on scope exit to restore previous state.
/// Tracks both newly-added keys (removed on undo) and shadowed keys
/// (restored to previous value on undo).
#[derive(Debug)]
pub(super) struct FuncScopeFrame {
    /// Keys that were newly inserted into `locals` (not present before scope entry).
    pub locals_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `locals`: (key, previous_value).
    pub locals_shadowed: Vec<(Rc<str>, LocalInfo)>,
    /// Keys newly inserted into `static_local_names`.
    pub statics_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `static_local_names`: (key, previous_value).
    pub statics_shadowed: Vec<(Rc<str>, Rc<str>)>,
    /// Keys newly inserted into `const_local_values`.
    pub consts_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `const_local_values`: (key, previous_value).
    pub consts_shadowed: Vec<(Rc<str>, i64)>,
    /// Keys newly inserted into `var_ctypes`.
    pub var_ctypes_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `var_ctypes`: (key, previous_value).
    pub var_ctypes_shadowed: Vec<(Rc<str>, CType)>,
    /// Keys newly inserted into `vla_typedef_sizes`.
    pub vla_typedef_sizes_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `vla_typedef_sizes`: (key, previous_value).
    pub vla_typedef_sizes_shadowed: Vec<(Rc<str>, Value)>,
    /// Saved stack pointer before the first VLA in this scope.
    /// When set, StackRestore is emitted at scope exit to reclaim VLA stack space.
    pub scope_stack_save: Option<Value>,
    /// Variables with __attribute__((cleanup(func))) in this scope.
    /// Stored in declaration order; cleanup calls are emitted in reverse order at scope exit.
    /// Each entry is (cleanup_function_name, alloca_value) where alloca_value is the
    /// address of the variable to pass as &var to the cleanup function.
    pub cleanup_vars: Vec<(Rc<str>, Value)>,
}

impl FuncScopeFrame {
    fn new() -> Self {
        Self {
            locals_added: Vec::new(),
            locals_shadowed: Vec::new(),
            statics_added: Vec::new(),
            statics_shadowed: Vec::new(),
            consts_added: Vec::new(),
            consts_shadowed: Vec::new(),
            var_ctypes_added: Vec::new(),
            var_ctypes_shadowed: Vec::new(),
            vla_typedef_sizes_added: Vec::new(),
            vla_typedef_sizes_shadowed: Vec::new(),
            scope_stack_save: None,
            cleanup_vars: Vec::new(),
        }
    }
}

/// Per-function build state, extracted from Lowerer to make the function-vs-module
/// state boundary explicit. Created fresh at the start of each function, replacing
/// the old pattern of clearing individual fields.
#[derive(Debug)]
pub(super) struct FunctionBuildState {
    /// Basic blocks accumulated for the current function
    pub blocks: Vec<BasicBlock>,
    /// Instructions for the current basic block being built
    pub instrs: Vec<Instruction>,
    /// Label of the current basic block
    pub current_label: BlockId,
    /// Name of the function currently being lowered
    pub name: Rc<str>,
    /// Return type of the function currently being lowered
    pub return_type: IrType,
    /// Whether the current function returns _Bool
    pub return_is_bool: bool,
    /// sret pointer alloca for current function (struct returns > 16 bytes)
    pub sret_ptr: Option<Value>,
    /// Variable -> alloca mapping with metadata
    pub locals: FxHashMap<Rc<str>, LocalInfo>,
    /// Loop context: (label, scope_depth) to jump to on `break`.
    /// scope_depth records the scope_stack length when the loop was entered,
    /// so break can emit cleanup calls for scopes being exited.
    pub break_labels: Vec<(BlockId, usize)>,
    /// Loop context: (label, scope_depth) to jump to on `continue`.
    /// scope_depth records the scope_stack length when the loop was entered,
    /// so continue can emit cleanup calls for scopes being exited.
    pub continue_labels: Vec<(BlockId, usize)>,
    /// Stack of switch statement contexts
    pub switch_stack: Vec<SwitchFrame>,
    /// User-defined goto labels -> unique IR labels
    pub user_labels: FxHashMap<Rc<str>, BlockId>,
    /// Set of user-defined goto labels that have been defined (label statement lowered).
    /// Used to distinguish forward gotos (label not yet defined) from backward gotos
    /// (label already defined) for VLA stack restore decisions.
    pub defined_user_labels: FxHashSet<Rc<str>>,
    /// User-defined goto labels -> scope depth at label definition site.
    /// Populated by a prescan of the function body before lowering, so that
    /// `goto` cleanup emission can determine which scopes are actually exited.
    pub user_label_depths: FxHashMap<Rc<str>, usize>,
    /// Scope stack for function-local variable undo tracking
    pub scope_stack: Vec<FuncScopeFrame>,
    /// Static local variable name -> mangled global name
    pub static_local_names: FxHashMap<Rc<str>, Rc<str>>,
    /// Const-qualified local variable values
    pub const_local_values: FxHashMap<Rc<str>, i64>,
    /// CType for each local variable
    pub var_ctypes: FxHashMap<Rc<str>, CType>,
    /// Runtime sizeof Values for VLA typedef types (e.g., `typedef char buf[n][m]`).
    /// Keyed by typedef name, value is the IR Value holding the runtime byte size.
    pub vla_typedef_sizes: FxHashMap<Rc<str>, Value>,
    /// Per-function value counter (reset for each function)
    pub next_value: u32,
    /// Saved stack pointer Value for VLA deallocation.
    /// When a function contains VLA declarations and gotos, the stack pointer is saved
    /// at function entry so it can be restored before backward jumps that cross VLA scopes.
    pub vla_stack_save: Option<Value>,
    /// Whether the function has any VLA declarations.
    /// Used to decide whether to emit StackSave/StackRestore around gotos.
    pub has_vla: bool,
    /// Alloca instructions deferred to the entry block.
    /// Local variable allocas are collected here during lowering so that
    /// variables whose declarations are skipped by `goto` still have valid
    /// stack slots at runtime. Merged into blocks[0] in finalize_function.
    pub entry_allocas: Vec<Instruction>,
    /// Values corresponding to the allocas created for function parameters.
    /// Populated during allocate_function_params and transferred to IrFunction
    /// in finalize_function. Used by the backend to detect dead param allocas.
    pub param_alloca_values: Vec<Value>,
    /// Source location spans for the current basic block being built,
    /// parallel to `instrs`. Each entry corresponds to one instruction.
    pub instr_spans: Vec<Span>,
    /// The current source span, set before lowering each statement/expression.
    /// Emitted instructions inherit this span for debug info tracking.
    pub current_span: Span,
    /// Whether this function is a candidate for inlining (always_inline, inline, or static).
    /// Used by __builtin_constant_p lowering: in non-inline-candidate functions, non-constant
    /// expressions resolve immediately to 0. In inline candidates, they emit IsConstant
    /// instructions that can be resolved to 1 after inlining if the argument becomes constant.
    pub is_inline_candidate: bool,
    /// Block IDs referenced by static local variable initializers via &&label.
    /// Transferred to IrFunction in finalize_function so CFG simplify can keep
    /// these blocks reachable (their labels appear in global data like .quad .LBB3).
    pub global_init_label_blocks: Vec<BlockId>,
}

impl FunctionBuildState {
    /// Create a new function build state for the given function.
    pub fn new(name: Rc<str>, return_type: IrType, return_is_bool: bool) -> Self {
        Self {
            blocks: Vec::new(),
            instrs: Vec::new(),
            current_label: BlockId(0),
            name,
            return_type,
            return_is_bool,
            sret_ptr: None,
            locals: FxHashMap::default(),
            break_labels: Vec::new(),
            continue_labels: Vec::new(),
            switch_stack: Vec::new(),
            user_labels: FxHashMap::default(),
            defined_user_labels: FxHashSet::default(),
            user_label_depths: FxHashMap::default(),
            scope_stack: Vec::new(),
            static_local_names: FxHashMap::default(),
            const_local_values: FxHashMap::default(),
            var_ctypes: FxHashMap::default(),
            vla_typedef_sizes: FxHashMap::default(),
            next_value: 0,
            vla_stack_save: None,
            has_vla: false,
            entry_allocas: Vec::new(),
            param_alloca_values: Vec::new(),
            instr_spans: Vec::new(),
            current_span: Span::dummy(),
            is_inline_candidate: false,
            global_init_label_blocks: Vec::new(),
        }
    }

    /// Push a new function-local scope frame.
    pub fn push_scope(&mut self) {
        self.scope_stack.push(FuncScopeFrame::new());
    }

    /// Pop the top function-local scope frame and undo changes to locals,
    /// static_local_names, const_local_values, and var_ctypes.
    /// Returns (scope_stack_save, cleanup_vars) - the VLA save value and cleanup variables.
    pub fn pop_scope(&mut self) -> (Option<Value>, Vec<(Rc<str>, Value)>) {
        if let Some(frame) = self.scope_stack.pop() {
            let scope_stack_save = frame.scope_stack_save;
            let cleanup_vars = frame.cleanup_vars;
            for key in frame.locals_added {
                self.locals.remove(&key);
            }
            for (key, val) in frame.locals_shadowed {
                self.locals.insert(key, val);
            }
            for key in frame.statics_added {
                self.static_local_names.remove(&key);
            }
            for (key, val) in frame.statics_shadowed {
                self.static_local_names.insert(key, val);
            }
            for key in frame.consts_added {
                self.const_local_values.remove(&key);
            }
            for (key, val) in frame.consts_shadowed {
                self.const_local_values.insert(key, val);
            }
            for key in frame.var_ctypes_added {
                self.var_ctypes.remove(&key);
            }
            for (key, val) in frame.var_ctypes_shadowed {
                self.var_ctypes.insert(key, val);
            }
            for key in frame.vla_typedef_sizes_added {
                self.vla_typedef_sizes.remove(&key);
            }
            for (key, val) in frame.vla_typedef_sizes_shadowed {
                self.vla_typedef_sizes.insert(key, val);
            }
            (scope_stack_save, cleanup_vars)
        } else {
            (None, Vec::new())
        }
    }

    /// Insert a VLA typedef runtime size, tracking for scope management.
    pub fn insert_vla_typedef_size_scoped(&mut self, name: Rc<str>, size: Value) {
        if let Some(frame) = self.scope_stack.last_mut() {
            if let Some(prev) = self.vla_typedef_sizes.remove(&name) {
                frame.vla_typedef_sizes_shadowed.push((name.clone(), prev));
            } else {
                frame.vla_typedef_sizes_added.push(name.clone());
            }
        }
        self.vla_typedef_sizes.insert(name, size);
    }

    /// Insert a local variable, tracking the change in the current scope frame.
    pub fn insert_local_scoped(&mut self, name: Rc<str>, info: LocalInfo) {
        if let Some(frame) = self.scope_stack.last_mut() {
            if let Some(prev) = self.locals.remove(&name) {
                frame.locals_shadowed.push((name.clone(), prev));
            } else {
                frame.locals_added.push(name.clone());
            }
        }
        self.locals.insert(name, info);
    }

    /// Insert a static local name, tracking the change in the current scope frame.
    pub fn insert_static_local_scoped(&mut self, name: Rc<str>, mangled: Rc<str>) {
        if let Some(frame) = self.scope_stack.last_mut() {
            if let Some(prev) = self.static_local_names.remove(&name) {
                frame.statics_shadowed.push((name.clone(), prev));
            } else {
                frame.statics_added.push(name.clone());
            }
        }
        self.static_local_names.insert(name, mangled);
    }

    /// Insert a const local value, tracking the change in the current scope frame.
    pub fn insert_const_local_scoped(&mut self, name: Rc<str>, value: i64) {
        if let Some(frame) = self.scope_stack.last_mut() {
            if let Some(prev) = self.const_local_values.remove(&name) {
                frame.consts_shadowed.push((name.clone(), prev));
            } else {
                frame.consts_added.push(name.clone());
            }
        }
        self.const_local_values.insert(name, value);
    }

    /// Remove a local variable from `locals`, tracking the removal in the
    /// current scope frame so `pop_scope()` restores it.
    pub fn shadow_local_for_scope(&mut self, name: &str) {
        if let Some(prev_local) = self.locals.remove(name) {
            if let Some(frame) = self.scope_stack.last_mut() {
                frame.locals_shadowed.push((Rc::from(name), prev_local));
            }
        }
    }

    /// Remove a static local name, tracking the removal in the current scope frame.
    pub fn shadow_static_for_scope(&mut self, name: &str) {
        if let Some(prev_static) = self.static_local_names.remove(name) {
            if let Some(frame) = self.scope_stack.last_mut() {
                frame.statics_shadowed.push((Rc::from(name), prev_static));
            }
        }
    }
}
