/// IR module, function, and global variable definitions.
///
/// `IrModule` is the top-level compilation unit containing functions, globals,
/// string literals, and linker directives. `IrFunction` represents a single
/// function with its parameter list, basic blocks, and ABI metadata.
/// `IrGlobal` defines a global variable with its initializer and linkage.
use std::rc::Rc;

use crate::common::types::IrType;
use super::constants::IrConst;
use super::instruction::{BasicBlock, Value};

/// A compilation unit in the IR.
#[derive(Debug)]
pub struct IrModule {
    pub functions: Vec<IrFunction>,
    pub globals: Vec<IrGlobal>,
    pub string_literals: Vec<(String, String)>, // (label, value)
    /// Wide string literals (L"..."): (label, chars as u32 values including null terminator)
    pub wide_string_literals: Vec<(String, Vec<u32>)>,
    /// char16_t string literals (u"..."): (label, chars as u16 values including null terminator)
    pub char16_string_literals: Vec<(String, Vec<u16>)>,
    pub constructors: Vec<Rc<str>>, // functions with __attribute__((constructor))
    pub destructors: Vec<Rc<str>>,  // functions with __attribute__((destructor))
    /// Symbol aliases: (alias_name, target_name, is_weak)
    /// From __attribute__((alias("target"))) and __attribute__((weak))
    pub aliases: Vec<(Rc<str>, Rc<str>, bool)>,
    /// Top-level asm("...") directives - emitted verbatim in assembly output
    pub toplevel_asm: Vec<String>,
    /// Symbol attribute directives for extern declarations:
    /// (name, is_weak, visibility) - emitted as .weak/.hidden/.protected directives
    pub symbol_attrs: Vec<(Rc<str>, bool, Option<String>)>,
    /// Symbol version directives: (function_name, symver_string)
    /// From __attribute__((symver("name@@VERSION"))) - emitted as .symver directives
    pub symver_directives: Vec<(Rc<str>, Rc<str>)>,
}

/// A global variable.
#[derive(Debug, Clone)]
pub struct IrGlobal {
    pub name: Rc<str>,
    pub ty: IrType,
    /// Size of the global in bytes (for arrays, this is elem_size * count).
    pub size: usize,
    /// Alignment in bytes.
    pub align: usize,
    /// Initializer for the global variable.
    pub init: GlobalInit,
    /// Whether this is a static (file-scope) variable.
    pub is_static: bool,
    /// Whether this is an extern declaration (no storage emitted).
    pub is_extern: bool,
    /// Whether this has __attribute__((common)) - use COMMON linkage.
    pub is_common: bool,
    /// __attribute__((section("..."))) - place in specific ELF section.
    pub section: Option<String>,
    /// __attribute__((weak)) - emit as a weak symbol (STB_WEAK).
    pub is_weak: bool,
    /// __attribute__((visibility("hidden"|"default"|...))) or #pragma GCC visibility.
    pub visibility: Option<String>,
    /// Whether the user specified an explicit alignment via __attribute__((aligned(N))) or _Alignas.
    /// When true, we respect the user's alignment exactly and don't auto-promote to 16.
    pub has_explicit_align: bool,
    /// Whether this global has const qualification (should be placed in .rodata).
    pub is_const: bool,
    /// __attribute__((used)) - prevent dead code elimination of this symbol.
    pub is_used: bool,
    /// Whether this global has _Thread_local or __thread storage class.
    /// Thread-local globals are placed in .tdata/.tbss and accessed via TLS mechanisms.
    pub is_thread_local: bool,
}

/// Initializer for a global variable.
#[derive(Debug, Clone)]
pub enum GlobalInit {
    /// No initializer (zero-initialized in .bss).
    Zero,
    /// Single scalar constant.
    Scalar(IrConst),
    /// Array of scalar constants.
    Array(Vec<IrConst>),
    /// String literal (stored as bytes with null terminator).
    String(String),
    /// Wide string literal (stored as array of u32 wchar_t values, no null terminator in vec).
    /// The backend emits each value as .long and adds a null terminator.
    WideString(Vec<u32>),
    /// char16_t string literal (stored as array of u16 values, no null terminator in vec).
    /// The backend emits each value as .short and adds a null terminator.
    Char16String(Vec<u16>),
    /// Address of another global (for pointer globals like `const char *s = "hello"`).
    GlobalAddr(Rc<str>),
    /// Address of a global plus a byte offset (for `&arr[3]`, `&s.field`, etc.).
    GlobalAddrOffset(Rc<str>, i64),
    /// Compound initializer: a sequence of initializer elements (for arrays/structs
    /// containing address expressions, e.g., `int *ptrs[] = {&a, &b, 0}`).
    Compound(Vec<GlobalInit>),
    /// Difference of two labels (&&lab1 - &&lab2) for computed goto dispatch tables.
    /// Fields: (label1, label2, byte_size) where byte_size is the width of the
    /// resulting integer (4 for int, 8 for long).
    GlobalLabelDiff(Rc<str>, Rc<str>, usize),
}

impl GlobalInit {
    /// Visit every symbol name referenced by this initializer.
    /// Calls `f` with each global/label name found in GlobalAddr, GlobalAddrOffset,
    /// and GlobalLabelDiff variants, recursing into Compound children.
    pub fn for_each_ref<F: FnMut(&str)>(&self, f: &mut F) {
        match self {
            GlobalInit::GlobalAddr(name) | GlobalInit::GlobalAddrOffset(name, _) => {
                f(name);
            }
            GlobalInit::GlobalLabelDiff(label1, label2, _) => {
                f(label1);
                f(label2);
            }
            GlobalInit::Compound(fields) => {
                for field in fields {
                    field.for_each_ref(f);
                }
            }
            _ => {}
        }
    }

    /// Returns the byte size of this initializer element in a compound context.
    /// Used when flattening nested Compound elements into a parent Compound.
    pub fn byte_size(&self) -> usize {
        match self {
            GlobalInit::Scalar(_) => 1,
            // Use target pointer size: 4 bytes on i686, 8 bytes on 64-bit targets
            GlobalInit::GlobalAddr(_) | GlobalInit::GlobalAddrOffset(_, _) => {
                crate::common::types::target_ptr_size()
            }
            GlobalInit::Compound(inner) => inner.len(),
            GlobalInit::Array(vals) => vals.len(),
            GlobalInit::Zero => 0,
            GlobalInit::String(s) => s.chars().count(),
            GlobalInit::WideString(ws) => ws.len() * 4,
            GlobalInit::Char16String(cs) => cs.len() * 2,
            GlobalInit::GlobalLabelDiff(_, _, size) => *size,
        }
    }

    /// Returns the total number of bytes that will be emitted for this initializer.
    /// Unlike `byte_size()`, this correctly accounts for GlobalAddr entries
    /// (pointer-sized: 4 bytes on i686, 8 bytes on 64-bit) inside Compound initializers.
    pub fn emitted_byte_size(&self) -> usize {
        match self {
            GlobalInit::Scalar(_) => 1,
            // Use target pointer size: 4 bytes on i686, 8 bytes on 64-bit targets
            GlobalInit::GlobalAddr(_) | GlobalInit::GlobalAddrOffset(_, _) => {
                crate::common::types::target_ptr_size()
            }
            GlobalInit::Compound(inner) => inner.iter().map(|e| e.emitted_byte_size()).sum(),
            GlobalInit::Array(vals) => vals.len(),
            GlobalInit::Zero => 0,
            GlobalInit::String(s) => s.chars().count() + 1,
            GlobalInit::WideString(ws) => (ws.len() + 1) * 4,
            GlobalInit::Char16String(cs) => (cs.len() + 1) * 2,
            GlobalInit::GlobalLabelDiff(_, _, size) => *size,
        }
    }
}

/// An IR function.
#[derive(Debug)]
pub struct IrFunction {
    pub name: Rc<str>,
    pub return_type: IrType,
    pub params: Vec<IrParam>,
    pub blocks: Vec<BasicBlock>,
    pub is_variadic: bool,
    pub is_declaration: bool, // true if no body (extern)
    pub is_static: bool,      // true if declared with `static` linkage
    pub is_inline: bool,      // true if declared with `inline` (used to skip patchable function entries)
    /// True when __attribute__((always_inline)) is present.
    /// These functions must always be inlined at call sites.
    pub is_always_inline: bool,
    /// True when __attribute__((noinline)) is present.
    /// These functions must never be inlined.
    pub is_noinline: bool,
    /// Cached upper bound on Value IDs: all Value IDs in this function are < next_value_id.
    /// Set by lowering/mem2reg/phi_eliminate to avoid expensive full-IR scans.
    /// A value of 0 means "not yet computed" (will fall back to scanning).
    pub next_value_id: u32,
    /// __attribute__((section("..."))) - place in specific ELF section.
    pub section: Option<String>,
    /// __attribute__((visibility("hidden"|"default"|...)))
    pub visibility: Option<String>,
    /// __attribute__((weak)) - emit as a weak symbol (STB_WEAK).
    pub is_weak: bool,
    /// __attribute__((used)) - prevent dead code elimination of this symbol.
    pub is_used: bool,
    /// __attribute__((fastcall)) - i386 fastcall calling convention.
    /// First two integer/pointer args passed in ecx/edx instead of stack.
    pub is_fastcall: bool,
    /// __attribute__((naked)) - emit no prologue/epilogue; function body is pure asm.
    pub is_naked: bool,
    /// Set by the inlining pass when call sites were inlined into this function.
    /// Used by post-inlining passes (mem2reg re-run, symbol resolution) to know
    /// that non-entry blocks may contain allocas from inlined callees.
    pub has_inlined_calls: bool,
    /// Values corresponding to the allocas created for function parameters.
    /// Tracked explicitly because lowering creates these allocas, but they may
    /// become unused after optimization. The backend uses this to detect dead
    /// param allocas and skip stack slot allocation, reducing frame size.
    pub param_alloca_values: Vec<Value>,
    /// True when the function returns a large struct via hidden pointer (sret).
    /// On i386 SysV ABI, such functions must use `ret $4` to pop the hidden
    /// pointer argument from the caller's stack.
    pub uses_sret: bool,
    /// Block IDs referenced by static local variable initializers via &&label.
    /// These blocks must be kept reachable and not merged away by CFG simplify,
    /// since their labels appear in global data (.quad .LBB3) and must resolve.
    pub global_init_label_blocks: Vec<super::instruction::BlockId>,
    /// SysV AMD64 ABI eightbyte classification for the return struct (if 9-16 bytes).
    /// Used by the x86-64 backend to determine whether each eightbyte should be
    /// returned in a GP register (rax/rdx) or SSE register (xmm0/xmm1).
    /// Empty for non-struct returns, sret, or non-x86-64 targets.
    pub ret_eightbyte_classes: Vec<crate::common::types::EightbyteClass>,
    /// True for `extern inline __attribute__((gnu_inline))` functions (or `extern inline`
    /// in GNU89 mode). These function bodies are lowered for inlining, but must NOT be
    /// emitted as standalone definitions. After the inlining pass, they are converted to
    /// declarations so any remaining calls resolve to the external libc/library symbol.
    /// This prevents infinite recursion when the inline body calls the same symbol
    /// (e.g., glibc's `btowc` inline calling `__btowc_alias` with asm name "btowc").
    pub is_gnu_inline_def: bool,
}

/// A function parameter.
#[derive(Debug, Clone)]
pub struct IrParam {
    pub ty: IrType,
    /// If this param is a struct/union passed by value, its byte size. None for non-struct params.
    pub struct_size: Option<usize>,
    /// Struct alignment in bytes. Used on RISC-V to even-align register pairs for
    /// 2×XLEN-aligned structs. None for non-struct params.
    pub struct_align: Option<usize>,
    /// Per-eightbyte SysV ABI classification for struct params (x86-64 only).
    /// Empty for non-struct params or when classification is not applicable.
    /// Each entry indicates whether that eightbyte should use SSE or GP registers.
    pub struct_eightbyte_classes: Vec<crate::common::types::EightbyteClass>,
    /// RISC-V LP64D float field classification for struct params.
    /// When Some, indicates this struct should use FP registers per the psABI
    /// hardware floating-point calling convention. None for non-struct params
    /// or structs that don't qualify for FP register passing.
    pub riscv_float_class: Option<crate::common::types::RiscvFloatClass>,
}

impl IrModule {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            globals: Vec::new(),
            string_literals: Vec::new(),
            wide_string_literals: Vec::new(),
            char16_string_literals: Vec::new(),
            constructors: Vec::new(),
            destructors: Vec::new(),
            aliases: Vec::new(),
            toplevel_asm: Vec::new(),
            symbol_attrs: Vec::new(),
            symver_directives: Vec::new(),
        }
    }

    /// Run a transformation on each defined (non-declaration) function, returning
    /// the total count of changes made. Used by optimization passes.
    pub fn for_each_function<F>(&mut self, mut f: F) -> usize
    where
        F: FnMut(&mut IrFunction) -> usize,
    {
        let mut total = 0;
        for func in &mut self.functions {
            if !func.is_declaration {
                total += f(func);
            }
        }
        total
    }
}

impl Default for IrModule {
    fn default() -> Self {
        Self::new()
    }
}

impl IrFunction {
    #[cfg(test)]
    pub fn new(name: Rc<str>, return_type: IrType, params: Vec<IrParam>, is_variadic: bool) -> Self {
        Self {
            name,
            return_type,
            params,
            blocks: Vec::new(),
            is_variadic,
            is_declaration: false,
            is_static: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            next_value_id: 0,
            section: None,
            visibility: None,
            is_weak: false,
            is_used: false,
            has_inlined_calls: false,
            param_alloca_values: Vec::new(),
            uses_sret: false,
            is_fastcall: false,
            is_naked: false,
            global_init_label_blocks: Vec::new(),
            ret_eightbyte_classes: Vec::new(),
            is_gnu_inline_def: false,
        }
    }

    /// Return the highest Value ID defined (as a destination) in this function, or 0 if empty.
    /// Uses the cached `next_value_id` if available, otherwise falls back to scanning.
    /// Useful for sizing flat lookup tables indexed by Value ID.
    #[inline]
    pub fn max_value_id(&self) -> u32 {
        if self.next_value_id > 0 {
            // next_value_id is the first *unused* ID, so max used is one less
            return self.next_value_id - 1;
        }
        // Fallback: scan all instructions (expensive)
        let mut max_id: u32 = 0;
        for block in &self.blocks {
            for inst in &block.instructions {
                if let Some(v) = inst.dest() {
                    if v.0 > max_id {
                        max_id = v.0;
                    }
                }
            }
        }
        max_id
    }
}
