//! Shared codegen state, slot addressing types, and register value cache.
//!
//! All four backends use the same `CodegenState` to track stack slot assignments,
//! alloca metadata, label generation, and the register value cache during code generation.
//! The `SlotAddr` enum captures the 3-way addressing pattern (over-aligned alloca /
//! direct alloca / indirect) that repeats across store, load, GEP, and memcpy emission.
//!
//! The `RegCache` tracks which IR values are currently known to be in registers,
//! enabling backends to skip redundant stack loads. This is the foundation for
//! eventually replacing the pure stack-slot model with a register allocator.

use std::rc::Rc;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use super::common::AsmOutput;

/// Stack slot location for a value. Interpretation varies by arch:
/// - x86: negative offset from %rbp
/// - ARM: positive offset from sp
/// - RISC-V: negative offset from s0
#[derive(Debug, Clone, Copy)]
pub struct StackSlot(pub i64);

/// Register cache entry: tracks which IR value is known to be in a register.
/// The `is_alloca` flag distinguishes whether the register holds the alloca's
/// address (leaq/adr) or the value loaded from the stack slot (movq/ldr).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegCacheEntry {
    pub value_id: u32,
    pub is_alloca: bool,
}

/// Register value cache. Tracks which IR values are currently in the accumulator
/// and secondary registers, avoiding redundant stack loads.
///
/// The cache is conservative: it is invalidated on any operation that might clobber
/// a register (calls, inline asm, complex operations that use scratch registers).
/// This is safe because a stale entry would just cause a redundant load (the same
/// behavior as before the cache existed), while a missing invalidation could cause
/// incorrect code by skipping a needed load.
///
/// Architecture mapping:
/// - x86:    acc = %rax,  sec = %rcx
/// - ARM64:  acc = x0,    sec = x1
/// - RISC-V: acc = t0,    sec = t1
#[derive(Debug, Default)]
pub struct RegCache {
    /// Which value is currently in the primary accumulator register.
    pub acc: Option<RegCacheEntry>,
}

impl RegCache {
    /// Record that the accumulator now holds the given value.
    #[inline]
    pub fn set_acc(&mut self, value_id: u32, is_alloca: bool) {
        self.acc = Some(RegCacheEntry { value_id, is_alloca });
    }

    /// Check if the accumulator holds the given value (with matching alloca status).
    #[inline]
    pub fn acc_has(&self, value_id: u32, is_alloca: bool) -> bool {
        self.acc == Some(RegCacheEntry { value_id, is_alloca })
    }

    /// Invalidate the accumulator cache.
    #[inline]
    pub fn invalidate_acc(&mut self) {
        self.acc = None;
    }

    /// Invalidate all cached register values. Called on operations that may
    /// clobber any register (calls, inline asm, etc.).
    #[inline]
    pub fn invalidate_all(&mut self) {
        self.acc = None;
    }
}

/// Shared codegen state, used by all backends.
pub struct CodegenState {
    pub out: AsmOutput,
    pub stack_offset: i64,
    pub value_locations: FxHashMap<u32, StackSlot>,
    /// Values that are allocas (their stack slot IS the data, not a pointer to data).
    pub alloca_values: FxHashSet<u32>,
    /// Type associated with each alloca (for type-aware loads/stores).
    pub alloca_types: FxHashMap<u32, IrType>,
    /// Alloca values that need runtime alignment > 16 bytes.
    pub alloca_alignments: FxHashMap<u32, usize>,
    /// Values that are 128-bit integers (need 16-byte copy).
    pub i128_values: FxHashSet<u32>,
    /// Values that are 64-bit types on 32-bit targets (F64, I64/U64).
    /// These values don't fit in a single 32-bit GPR and need special
    /// copy handling (e.g., x87 fldl/fstpl for F64, two-word copy for I64).
    /// Only populated on i686; empty on 64-bit targets.
    pub wide_values: FxHashSet<u32>,
    /// Counter for generating unique labels (e.g., memcpy loops).
    label_counter: u32,
    /// Whether position-independent code (PIC) generation is enabled.
    pub pic_mode: bool,
    /// Set of symbol names that are locally defined (not extern) and have internal
    /// linkage (static) — these can use direct addressing even in PIC mode.
    pub local_symbols: FxHashSet<Rc<str>>,
    /// Set of symbol names that are thread-local (_Thread_local / __thread).
    /// These require TLS-specific access patterns (e.g., %fs:x@TPOFF on x86-64).
    pub tls_symbols: FxHashSet<Rc<str>>,
    /// Whether the current function contains DynAlloca instructions.
    /// When true, the epilogue must restore SP from the frame pointer instead of
    /// adding back the compile-time frame size.
    pub has_dyn_alloca: bool,
    /// Register value cache: tracks which IR values are in the accumulator and
    /// secondary registers to skip redundant loads.
    pub reg_cache: RegCache,
    /// Whether to replace `ret` with `jmp __x86_return_thunk` (-mfunction-return=thunk-extern).
    /// Used by the Linux kernel for Spectre v2 (retbleed) mitigation.
    pub function_return_thunk: bool,
    /// Whether to replace indirect calls/jumps with retpoline thunks (-mindirect-branch=thunk-extern).
    /// Used by the Linux kernel for Spectre v2 (retpoline) mitigation.
    pub indirect_branch_thunk: bool,
    /// Patchable function entry: (total_nops, nops_before_entry).
    /// When set, emits NOP padding around function entry points and records
    /// them in __patchable_function_entries for runtime patching (ftrace).
    pub patchable_function_entry: Option<(u32, u32)>,
    /// Whether to emit endbr64 at function entry points (-fcf-protection=branch).
    pub cf_protection_branch: bool,
    /// Maps an F128 value ID to the memory location it was loaded from.
    /// This enables full-precision F128 operations (casts, comparisons, stores)
    /// by reloading directly from the original memory location instead of using
    /// a lossy f64 intermediate.
    ///
    /// The tuple is `(source_value_id, byte_offset, is_indirect)`:
    /// - `source_value_id`: The alloca or pointer value whose slot contains the data.
    /// - `byte_offset`: Offset from the source slot's base address.
    /// - `is_indirect`: If true, the source slot holds a pointer that must be
    ///   dereferenced. If false, the F128 data is directly at the slot.
    ///
    /// x86 uses this for x87 `fldt` from original memory (typically offset=0, is_indirect=false).
    /// ARM/RISC-V use this for IEEE binary128 soft-float operations with full 16-byte precision.
    pub f128_load_sources: FxHashMap<u32, (u32, i64, bool)>,
    /// Values whose 16-byte slots contain full x87 80-bit data (via fstpt),
    /// not a pointer. These are F128 call results or other values where
    /// full precision was preserved directly in the slot.
    pub f128_direct_slots: FxHashSet<u32>,
    /// The current text section name for this function. Defaults to ".text" but
    /// may be a custom section (e.g., ".init.text") for functions with
    /// __attribute__((section("..."))). Used to restore the correct section
    /// after emitting data (e.g., jump tables) in other sections.
    pub current_text_section: String,
    /// Whether to use the kernel code model (-mcmodel=kernel). All symbols
    /// are assumed to be in the negative 2GB of the virtual address space.
    /// Uses absolute sign-extended 32-bit addressing (movq $symbol) for
    /// global address references, producing R_X86_64_32S relocations.
    pub code_model_kernel: bool,
    /// Whether to disable jump table emission for switch statements (-fno-jump-tables).
    pub no_jump_tables: bool,
    /// Set of symbol names declared as weak extern (e.g., `extern __weak`).
    /// On AArch64, these need GOT-indirect addressing because the linker
    /// rejects R_AARCH64_ADR_PREL_PG_HI21 against symbols that may bind externally.
    pub weak_extern_symbols: FxHashSet<Rc<str>>,
    /// SSA values that use 4-byte (32-bit) stack slots instead of the default 8-byte.
    /// On 64-bit targets, I32/U32/F32 and smaller types can use 4-byte slots,
    /// reducing stack frame sizes by ~40%. Store/load paths check this set to
    /// emit 4-byte instructions (movl, sw/lw, str/ldr w-reg) instead of 8-byte.
    pub small_slot_values: FxHashSet<u32>,
    /// Values that were assigned to callee-saved registers and have no stack slot.
    /// Used by resolve_slot_addr to return a dummy Indirect slot for these values,
    /// which is safe because all Indirect codepaths check reg_assignments first.
    pub reg_assigned_values: FxHashSet<u32>,
    /// Values that are promoted InlineAsm output results. Like allocas, their
    /// stack slot holds the value directly (not a pointer). The asm emitter
    /// stores the output register to this slot after the asm, and subsequent
    /// uses load the value from it.
    pub asm_output_values: FxHashSet<u32>,
    /// Whether to emit .file/.loc debug directives for source-level debugging.
    pub debug_info: bool,
    /// Pre-computed parameter classifications for the current function.
    /// Populated by `emit_store_params` so that `emit_param_ref` can access them.
    pub param_classes: Vec<crate::backend::call_abi::ParamClass>,
    /// Number of function parameters (for ParamRef bounds checking).
    pub num_params: usize,
    /// Whether the current function is variadic (for ParamRef ABI handling).
    pub func_is_variadic: bool,
    /// Stack slots for parameter allocas, indexed by param_idx.
    /// Populated by `emit_store_params` so that `emit_param_ref` can load the
    /// parameter value from its alloca slot (where emit_store_params saved it)
    /// instead of reading from ABI registers that may have been clobbered.
    pub param_alloca_slots: Vec<Option<(StackSlot, IrType)>>,
    /// Set of param indices whose values have been pre-stored directly to a
    /// callee-saved register during `emit_store_params`. `emit_param_ref` can
    /// skip the alloca load for these and just emit a no-op (the value is
    /// already in the register-allocated destination).
    pub param_pre_stored: FxHashSet<usize>,
    /// Whether the current function returns a struct via hidden pointer (sret).
    /// On i386 SysV ABI, such functions must use `ret $4` to pop the hidden
    /// pointer argument from the caller's stack.
    pub uses_sret: bool,
    /// Whether to place each function in its own section (-ffunction-sections).
    pub function_sections: bool,
    /// Whether to place each data object in its own section (-fdata-sections).
    pub data_sections: bool,
    /// Whether any 64-bit division/modulo runtime helpers (__divdi3, __udivdi3,
    /// __moddi3, __umoddi3) were referenced during code generation.
    /// When true, the i686 backend emits weak implementations of these functions
    /// so that standalone builds (without libgcc) can link successfully.
    pub needs_divdi3_helpers: bool,
    /// Whether to emit CFI directives (.cfi_startproc, .cfi_endproc, etc.)
    /// for generating .eh_frame unwind tables. Enabled by default (like GCC).
    pub emit_cfi: bool,
}

impl CodegenState {
    pub fn new() -> Self {
        Self {
            out: AsmOutput::new(),
            stack_offset: 0,
            value_locations: FxHashMap::default(),
            alloca_values: FxHashSet::default(),
            alloca_types: FxHashMap::default(),
            alloca_alignments: FxHashMap::default(),
            i128_values: FxHashSet::default(),
            wide_values: FxHashSet::default(),
            label_counter: 0,
            pic_mode: false,
            local_symbols: FxHashSet::default(),
            tls_symbols: FxHashSet::default(),
            has_dyn_alloca: false,
            reg_cache: RegCache::default(),
            function_return_thunk: false,
            indirect_branch_thunk: false,
            patchable_function_entry: None,
            cf_protection_branch: false,
            f128_load_sources: FxHashMap::default(),
            f128_direct_slots: FxHashSet::default(),
            current_text_section: ".text".to_string(),
            code_model_kernel: false,
            no_jump_tables: false,
            weak_extern_symbols: FxHashSet::default(),
            small_slot_values: FxHashSet::default(),
            reg_assigned_values: FxHashSet::default(),
            asm_output_values: FxHashSet::default(),
            debug_info: false,
            param_classes: Vec::new(),
            num_params: 0,
            func_is_variadic: false,
            param_alloca_slots: Vec::new(),
            param_pre_stored: FxHashSet::default(),
            uses_sret: false,
            function_sections: false,
            data_sections: false,
            needs_divdi3_helpers: false,
            emit_cfi: true,
        }
    }

    pub fn next_label_id(&mut self) -> u32 {
        let id = self.label_counter;
        self.label_counter += 1;
        id
    }

    /// Generate a fresh label with the given prefix.
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        let id = self.next_label_id();
        format!(".L{}_{}", prefix, id)
    }

    pub fn emit(&mut self, s: &str) {
        self.out.emit(s);
    }

    /// Emit formatted assembly directly (no temporary String allocation).
    #[inline]
    pub fn emit_fmt(&mut self, args: std::fmt::Arguments<'_>) {
        self.out.emit_fmt(args);
    }

    /// Emit a visibility directive (.hidden, .protected, .internal) if the symbol
    /// has non-default visibility. No-op if `visibility` is None or "default".
    pub fn emit_visibility(&mut self, name: &str, visibility: &Option<String>) {
        if let Some(ref vis) = visibility {
            match vis.as_str() {
                "hidden" => self.emit_fmt(format_args!(".hidden {}", name)),
                "protected" => self.emit_fmt(format_args!(".protected {}", name)),
                "internal" => self.emit_fmt(format_args!(".internal {}", name)),
                _ => {} // "default" or unknown: no directive needed
            }
        }
    }

    /// Emit linkage directives (.globl or .weak) for a non-static symbol.
    /// No-op if `is_static` is true.
    pub fn emit_linkage(&mut self, name: &str, is_static: bool, is_weak: bool) {
        if !is_static {
            if is_weak {
                self.emit_fmt(format_args!(".weak {}", name));
            } else {
                self.emit_fmt(format_args!(".globl {}", name));
            }
        }
    }

    pub fn reset_for_function(&mut self) {
        self.stack_offset = 0;
        self.value_locations.clear();
        self.alloca_values.clear();
        self.alloca_types.clear();
        self.alloca_alignments.clear();
        self.i128_values.clear();
        self.wide_values.clear();
        self.has_dyn_alloca = false;
        self.reg_cache.invalidate_all();
        self.f128_direct_slots.clear();
        self.f128_load_sources.clear();
        self.small_slot_values.clear();
        self.reg_assigned_values.clear();
        self.asm_output_values.clear();
        self.param_pre_stored.clear();
        self.uses_sret = false;
    }

    /// Get the over-alignment requirement for an alloca (> 16 bytes), or None.
    pub fn alloca_over_align(&self, v: u32) -> Option<usize> {
        self.alloca_alignments.get(&v).copied()
    }

    pub fn is_alloca(&self, v: u32) -> bool {
        self.alloca_values.contains(&v)
    }

    /// Check if a value has a direct stack slot (the slot holds the value itself,
    /// not a pointer to the value). This is true for allocas and promoted InlineAsm
    /// output values.
    pub fn is_direct_slot(&self, v: u32) -> bool {
        self.alloca_values.contains(&v) || self.asm_output_values.contains(&v)
    }

    pub fn get_slot(&self, v: u32) -> Option<StackSlot> {
        self.value_locations.get(&v).copied()
    }

    pub fn is_i128_value(&self, v: u32) -> bool {
        self.i128_values.contains(&v)
    }

    /// Check if a value uses a 4-byte (small) stack slot.
    /// Used by store/load paths to emit 4-byte instructions instead of 8-byte.
    /// Currently infrastructure-only: backends don't use this yet because
    /// store/load paths aren't fully type-safe (some always use 8-byte ops).
    #[allow(dead_code)]
    pub fn is_small_slot(&self, v: u32) -> bool {
        self.small_slot_values.contains(&v)
    }

    /// Check if a value is a "wide" type on 32-bit targets (F64, I64, U64).
    /// These need multi-word copy handling instead of the 32-bit accumulator path.
    pub fn is_wide_value(&self, v: u32) -> bool {
        self.wide_values.contains(&v)
    }

    /// Track that `dest_id` was loaded from `source_id` at the given byte offset.
    /// Automatically determines `is_indirect` based on whether the source is an alloca.
    /// Allocas have data directly in the slot; non-allocas hold pointers that must be dereferenced.
    #[inline]
    pub fn track_f128_load(&mut self, dest_id: u32, source_id: u32, offset: i64) {
        let is_indirect = !self.is_alloca(source_id);
        self.f128_load_sources.insert(dest_id, (source_id, offset, is_indirect));
    }

    /// Track that `value_id` has full F128 data stored directly in its own slot.
    /// (Used after operations like negation or cast that produce full-precision F128 results.)
    #[inline]
    pub fn track_f128_self(&mut self, value_id: u32) {
        self.f128_load_sources.insert(value_id, (value_id, 0, false));
    }

    /// Look up the F128 load source for a value: `(source_id, offset, is_indirect)`.
    #[inline]
    pub fn get_f128_source(&self, value_id: u32) -> Option<(u32, i64, bool)> {
        self.f128_load_sources.get(&value_id).copied()
    }

    /// Returns true if the given symbol needs GOT indirection in PIC mode.
    /// A symbol needs GOT if PIC is enabled AND it's not a local (static) symbol.
    /// Local labels (starting with '.') are always PIC-safe via RIP-relative.
    pub fn needs_got(&self, name: &str) -> bool {
        if !self.pic_mode {
            return false;
        }
        if name.starts_with('.') {
            return false;
        }
        !self.local_symbols.contains(name)
    }

    /// Returns true if taking the address of a symbol requires GOT indirection.
    /// Unlike needs_got(), this returns true for external symbols even in non-PIC
    /// mode (x86-64 only). Modern toolchains default to PIE, so object files must
    /// use GOTPCREL for external symbol addresses to be compatible with PIE linking
    /// by the system linker. Locally-defined symbols can still use direct leaq.
    pub fn needs_got_for_addr(&self, name: &str) -> bool {
        if self.code_model_kernel {
            return false;
        }
        if name.starts_with('.') {
            return false;
        }
        !self.local_symbols.contains(name)
    }

    /// Returns true if a function call needs PLT indirection in PIC mode.
    pub fn needs_plt(&self, name: &str) -> bool {
        self.needs_got(name)
    }

    /// Returns true if a symbol needs GOT indirection on AArch64.
    ///
    /// In PIC mode (-fPIC/-fpic/-shared), external symbols that may bind
    /// externally must use GOT-indirect ADRP+LDR :got: sequences instead of
    /// direct ADRP+ADD. The system linker rejects R_AARCH64_ADR_PREL_PG_HI21
    /// against symbols that may bind externally in shared objects.
    ///
    /// Weak extern symbols always need GOT indirection regardless of PIC mode,
    /// because they may resolve to zero at runtime.
    pub fn needs_got_aarch64(&self, name: &str) -> bool {
        if name.starts_with('.') {
            return false;
        }
        // Weak extern symbols always need GOT indirection on AArch64
        if self.weak_extern_symbols.contains(name) {
            return true;
        }
        // In PIC mode, external symbols need GOT indirection (same as x86)
        if self.pic_mode {
            return !self.local_symbols.contains(name);
        }
        false
    }
}

/// How a value's effective address is accessed. This captures the 3-way decision
/// (alloca with over-alignment / alloca direct / non-alloca indirect) that repeats
/// across emit_store, emit_load, emit_gep, and emit_memcpy.
#[derive(Debug, Clone, Copy)]
pub enum SlotAddr {
    /// Alloca with alignment > 16: runtime-aligned address must be computed.
    OverAligned(StackSlot, u32),
    /// Normal alloca: slot IS the data, access directly.
    Direct(StackSlot),
    /// Non-alloca: slot holds a pointer that must be loaded first.
    Indirect(StackSlot),
}

impl CodegenState {
    /// Classify how to access a value's effective address.
    /// Returns `None` if the value has no assigned stack slot (and isn't register-assigned).
    pub fn resolve_slot_addr(&self, val_id: u32) -> Option<SlotAddr> {
        if let Some(slot) = self.get_slot(val_id) {
            if self.is_alloca(val_id) {
                if self.alloca_over_align(val_id).is_some() {
                    Some(SlotAddr::OverAligned(slot, val_id))
                } else {
                    Some(SlotAddr::Direct(slot))
                }
            } else {
                Some(SlotAddr::Indirect(slot))
            }
        } else if self.reg_assigned_values.contains(&val_id) {
            // Value lives in a callee-saved register with no stack slot.
            // Return a dummy Indirect slot — all Indirect codepaths in both
            // x86 and RISC-V backends check reg_assignments before accessing
            // the slot, so the dummy offset is never actually used.
            Some(SlotAddr::Indirect(StackSlot(0)))
        } else {
            None
        }
    }
}
