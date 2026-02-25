/// IR instruction definitions: the core SSA instruction set.
///
/// This module defines the SSA IR instruction enum with 38 variants covering
/// memory operations, arithmetic, control flow, atomics, SIMD intrinsics,
/// inline assembly, and ABI support (va_arg, sret, complex returns).
///
/// Key types:
/// - `BlockId`: basic block identifier (u32 index, formats as ".LBB{id}")
/// - `Value`: SSA value reference (u32 index)
/// - `Operand`: either a `Value` or an `IrConst`
/// - `Instruction`: the main instruction enum
/// - `CallInfo`: shared metadata for direct and indirect calls
/// - `Terminator`: block terminators (return, branch, switch)
/// - `BasicBlock`: a labeled sequence of instructions ending in a terminator
use std::rc::Rc;

use crate::common::source::Span;
use crate::common::types::{AddressSpace, EightbyteClass, IrType};
use super::constants::IrConst;
use super::intrinsics::IntrinsicOp;
use super::ops::{AtomicOrdering, AtomicRmwOp, IrBinOp, IrCmpOp, IrUnaryOp};

/// A basic block identifier. Uses a u32 index for zero-cost copies
/// instead of heap-allocated String labels. The block's assembly label
/// is generated on-the-fly during codegen as ".LBB{id}".
/// We use ".LBB" instead of ".L" to avoid collisions with the GNU
/// assembler's internal local labels (e.g., .L0, .L1) used for
/// RISC-V PCREL_HI20/PCREL_LO12 relocation pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Format this block ID as an assembly label (e.g., ".LBB5").
    #[inline]
    pub fn as_label(&self) -> String {
        format!(".LBB{}", self.0)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".LBB{}", self.0)
    }
}

/// An SSA value reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub u32);

/// An operand (either a value reference or a constant).
#[derive(Debug, Clone, Copy)]
pub enum Operand {
    Value(Value),
    Const(IrConst),
}

/// A basic block in the CFG.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub label: BlockId,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
    /// Source location spans for each instruction, parallel to `instructions`.
    /// Used by the backend to emit .file/.loc directives when compiling with -g.
    /// Empty when debug info is not being tracked (non-debug builds).
    pub source_spans: Vec<Span>,
}

/// Shared call metadata for both direct and indirect function calls.
///
/// This struct consolidates the fields that are common to `Instruction::Call` and
/// `Instruction::CallIndirect`, avoiding duplication across match arms and making
/// it easier to add new call-related fields in the future.
#[derive(Debug, Clone)]
pub struct CallInfo {
    /// Destination value for the return, or None for void calls.
    pub dest: Option<Value>,
    /// Argument operands.
    pub args: Vec<Operand>,
    /// Type of each argument (parallel to `args`).
    pub arg_types: Vec<IrType>,
    /// Return type of the callee.
    pub return_type: IrType,
    /// Whether the callee is variadic.
    pub is_variadic: bool,
    /// Number of named (non-variadic) parameters in the callee's prototype.
    /// For non-variadic calls, this equals args.len().
    pub num_fixed_args: usize,
    /// Which args are struct/union by-value: Some(size) for struct args, None otherwise.
    pub struct_arg_sizes: Vec<Option<usize>>,
    /// Struct alignment: Some(align) for struct args, None otherwise.
    /// Used on RISC-V to even-align register pairs for 2×XLEN-aligned structs.
    pub struct_arg_aligns: Vec<Option<usize>>,
    /// Per-eightbyte SysV ABI classification for struct args (for x86-64 SSE-class passing).
    pub struct_arg_classes: Vec<Vec<EightbyteClass>>,
    /// RISC-V LP64D float field classification for struct args.
    pub struct_arg_riscv_float_classes: Vec<Option<crate::common::types::RiscvFloatClass>>,
    /// True if the call uses a hidden pointer argument for struct returns (i386 SysV ABI).
    pub is_sret: bool,
    /// True if the callee uses the fastcall calling convention.
    pub is_fastcall: bool,
    /// SysV ABI eightbyte classification for the return struct (if 9-16 byte two-reg return).
    /// Used by the x86-64 backend to read SSE eightbytes from xmm0 instead of rdx.
    pub ret_eightbyte_classes: Vec<EightbyteClass>,
}

/// An IR instruction.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Allocate stack space: %dest = alloca ty
    /// `align` is the alignment override (0 means use default platform alignment).
    /// `volatile` prevents mem2reg from promoting this alloca to an SSA register.
    /// This is needed for volatile-qualified locals that must survive setjmp/longjmp.
    Alloca { dest: Value, ty: IrType, size: usize, align: usize, volatile: bool },

    /// Dynamic stack allocation: %dest = dynalloca size_operand, align
    /// Used for __builtin_alloca - adjusts stack pointer at runtime.
    DynAlloca { dest: Value, size: Operand, align: usize },

    /// Store to memory: store val, ptr (type indicates size of store)
    /// seg_override: segment register override for x86 (%gs:/%fs:) from named address spaces.
    Store { val: Operand, ptr: Value, ty: IrType, seg_override: AddressSpace },

    /// Load from memory: %dest = load ptr
    /// seg_override: segment register override for x86 (%gs:/%fs:) from named address spaces.
    Load { dest: Value, ptr: Value, ty: IrType, seg_override: AddressSpace },

    /// Binary operation: %dest = op lhs, rhs
    BinOp { dest: Value, op: IrBinOp, lhs: Operand, rhs: Operand, ty: IrType },

    /// Unary operation: %dest = op src
    UnaryOp { dest: Value, op: IrUnaryOp, src: Operand, ty: IrType },

    /// Comparison: %dest = cmp op lhs, rhs
    Cmp { dest: Value, op: IrCmpOp, lhs: Operand, rhs: Operand, ty: IrType },

    /// Direct function call: %dest = call func(args...)
    Call { func: Rc<str>, info: CallInfo },

    /// Indirect function call through a pointer: %dest = call_indirect ptr(args...)
    CallIndirect { func_ptr: Operand, info: CallInfo },

    /// Get element pointer (for arrays/structs)
    GetElementPtr { dest: Value, base: Value, offset: Operand, ty: IrType },

    /// Type cast/conversion
    Cast { dest: Value, src: Operand, from_ty: IrType, to_ty: IrType },

    /// Copy a value
    Copy { dest: Value, src: Operand },

    /// Get address of a global
    GlobalAddr { dest: Value, name: Rc<str> },

    /// Memory copy: memcpy(dest, src, size)
    Memcpy { dest: Value, src: Value, size: usize },

    /// va_arg: extract the next variadic argument from a va_list.
    /// va_list_ptr is a pointer to the va_list struct/pointer.
    /// result_ty is the type of the argument being extracted.
    VaArg { dest: Value, va_list_ptr: Value, result_ty: IrType },

    /// va_arg for struct/union types: read a struct from the va_list into a
    /// pre-allocated buffer. `dest_ptr` is a pointer to the buffer (an alloca),
    /// `size` is the struct size in bytes. The backend reads the appropriate
    /// number of bytes from the va_list (registers or overflow area) and stores
    /// them at `dest_ptr`, advancing the va_list state appropriately.
    ///
    /// `eightbyte_classes` carries the SysV AMD64 ABI eightbyte classification
    /// for small structs (<=16 bytes). When non-empty, the x86 backend uses this
    /// to check if all required register slots (GP and/or FP) are available:
    ///
    /// - If sufficient registers exist, each eightbyte is read from the register
    ///   save area (GP or FP depending on classification).
    /// - If not, the ENTIRE struct is read from the overflow area.
    ///
    /// This ensures the ABI rule that multi-eightbyte structs must be entirely
    /// in registers or entirely on the stack is respected.
    VaArgStruct {
        dest_ptr: Value,
        va_list_ptr: Value,
        size: usize,
        eightbyte_classes: Vec<crate::common::types::EightbyteClass>,
    },

    /// va_start: initialize a va_list. last_named_param is the pointer to the last named parameter.
    VaStart { va_list_ptr: Value, },

    /// va_end: cleanup a va_list (typically a no-op).
    VaEnd { va_list_ptr: Value },

    /// va_copy: copy one va_list to another.
    VaCopy { dest_ptr: Value, src_ptr: Value },

    /// Atomic read-modify-write: %dest = atomicrmw op ptr, val
    /// Performs: old = *ptr; *ptr = op(old, val); dest = old (fetch_and_*) or dest = op(old, val) (*_and_fetch)
    AtomicRmw {
        dest: Value,
        op: AtomicRmwOp,
        ptr: Operand,
        val: Operand,
        ty: IrType,
        ordering: AtomicOrdering,
    },

    /// Atomic compare-exchange: %dest = cmpxchg ptr, expected, desired
    /// Returns whether the exchange succeeded (as a boolean i8 for __atomic_compare_exchange_n)
    /// or the old value (for __sync_val_compare_and_swap).
    AtomicCmpxchg {
        dest: Value,
        ptr: Operand,
        expected: Operand,
        desired: Operand,
        ty: IrType,
        success_ordering: AtomicOrdering,
        failure_ordering: AtomicOrdering,
        /// If true, dest gets the success/failure boolean; if false, dest gets the old value.
        returns_bool: bool,
    },

    /// Atomic load: %dest = atomic_load ptr
    AtomicLoad {
        dest: Value,
        ptr: Operand,
        ty: IrType,
        ordering: AtomicOrdering,
    },

    /// Atomic store: atomic_store ptr, val
    AtomicStore {
        ptr: Operand,
        val: Operand,
        ty: IrType,
        ordering: AtomicOrdering,
    },

    /// Memory fence
    Fence {
        ordering: AtomicOrdering,
    },

    /// SSA Phi node: merges values from different predecessor blocks.
    /// Each entry in `incoming` is (value, block_id) indicating which value
    /// flows in from which predecessor block.
    Phi {
        dest: Value,
        ty: IrType,
        incoming: Vec<(Operand, BlockId)>,
    },

    /// Get the address of a label (GCC computed goto extension: &&label)
    LabelAddr { dest: Value, label: BlockId },

    /// Get the second F64 return value from a function call (for _Complex double returns).
    /// On x86-64: reads xmm1, on ARM64: reads d1, on RISC-V: reads fa1.
    /// Must appear immediately after a Call/CallIndirect instruction.
    GetReturnF64Second { dest: Value },

    /// Set the second F64 return value before a return (for _Complex double returns).
    /// On x86-64: writes xmm1, on ARM64: writes d1, on RISC-V: writes fa1.
    /// Must appear immediately before a Return terminator.
    SetReturnF64Second { src: Operand },

    /// Get the second F32 return value from a function call (for _Complex float returns on ARM/RISC-V).
    /// On ARM64: reads s1, on RISC-V: reads fa1 as float.
    /// Must appear immediately after a Call/CallIndirect instruction.
    GetReturnF32Second { dest: Value },

    /// Set the second F32 return value before a return (for _Complex float returns on ARM/RISC-V).
    /// On ARM64: writes s1, on RISC-V: writes fa1 as float.
    /// Must appear immediately before a Return terminator.
    SetReturnF32Second { src: Operand },

    /// Get the second F128 return value from a function call (for _Complex long double returns on x86-64).
    /// On x86-64: reads st(0) after the first fstpt has already popped the real part.
    /// Must appear immediately after a Call/CallIndirect instruction.
    GetReturnF128Second { dest: Value },

    /// Set the second F128 return value before a return (for _Complex long double returns on x86-64).
    /// On x86-64: loads an additional value onto the x87 FPU stack as st(1).
    /// Must appear immediately before a Return terminator.
    SetReturnF128Second { src: Operand },

    /// Inline assembly statement
    InlineAsm {
        /// Assembly template string (with \n\t separators)
        template: String,
        /// Output operands: (constraint, value_ptr, optional_name)
        outputs: Vec<(String, Value, Option<String>)>,
        /// Input operands: (constraint, operand, optional_name)
        inputs: Vec<(String, Operand, Option<String>)>,
        /// Clobber list (register names and "memory", "cc")
        clobbers: Vec<String>,
        /// Types of operands (outputs first, then inputs) for register size selection
        operand_types: Vec<IrType>,
        /// Goto labels for asm goto: (C label name, resolved block ID)
        goto_labels: Vec<(String, BlockId)>,
        /// Symbol names for input operands with "i" constraints (e.g., function names).
        /// One entry per input; None if the input is not a symbol reference.
        /// Used by %P and %a modifiers to emit raw symbol names in inline asm.
        input_symbols: Vec<Option<String>>,
        /// Per-operand address space overrides (outputs first, then inputs).
        /// Non-Default entries cause segment prefix on memory operands (e.g., %gs:).
        seg_overrides: Vec<AddressSpace>,
    },

    /// Target-independent intrinsic operation (fences, SIMD, CRC32, etc.).
    /// Each backend emits the appropriate native instructions for these operations.
    Intrinsic {
        dest: Option<Value>,
        op: IntrinsicOp,
        /// For store ops: destination pointer
        dest_ptr: Option<Value>,
        /// Operand arguments (varies by op)
        args: Vec<Operand>,
    },

    /// Conditional select: %dest = select cond, true_val, false_val
    /// Equivalent to: cond != 0 ? true_val : false_val
    /// Lowered to cmov on x86, csel on ARM, branch-based on RISC-V.
    /// Unlike a branch diamond, both operands are always evaluated.
    Select {
        dest: Value,
        cond: Operand,
        true_val: Operand,
        false_val: Operand,
        ty: IrType,
    },

    /// Save the current stack pointer: %dest = stacksave
    /// Used to capture the SP before VLA allocations so it can be restored later.
    StackSave { dest: Value },

    /// Restore the stack pointer: stackrestore %ptr
    /// Used to reclaim VLA stack space when jumping backward past VLA declarations.
    StackRestore { ptr: Value },

    /// Reference to a function parameter value: %dest = paramref param_idx
    /// Represents the incoming value of the function's `param_idx`-th parameter.
    /// Emitted in the entry block alongside param alloca + store to make parameter
    /// initial values visible in the IR, allowing mem2reg to promote param allocas
    /// to SSA and enabling constant propagation through reassigned parameters.
    /// The backend translates this to a load from the appropriate argument register
    /// or stack slot according to the calling convention.
    ParamRef { dest: Value, param_idx: usize, ty: IrType },
}

/// Block terminator.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return from function
    Return(Option<Operand>),

    /// Unconditional branch
    Branch(BlockId),

    /// Conditional branch
    CondBranch { cond: Operand, true_label: BlockId, false_label: BlockId },

    /// Indirect branch (computed goto): goto *addr
    /// possible_targets lists all labels that could be jumped to (for optimization/validation)
    IndirectBranch { target: Operand, possible_targets: Vec<BlockId> },

    /// Switch dispatch via jump table.
    /// Implements dense switch statements efficiently: the backend emits a jump table
    /// instead of a chain of compare-and-branch instructions.
    /// `val` is the switch expression (must be integer type).
    /// `cases` maps case values to target block IDs.
    /// `default` is the fallback block.
    Switch {
        val: Operand,
        cases: Vec<(i64, BlockId)>,
        default: BlockId,
        ty: IrType,
    },

    /// Unreachable (e.g., after noreturn call)
    Unreachable,
}

// === Instruction impl: dest, result_type, value visitors ===

impl Instruction {
    /// Get the destination value defined by this instruction, if any.
    /// Instructions like Store, Memcpy, VaStart, VaEnd, VaCopy, AtomicStore,
    /// Fence, and InlineAsm produce no value.
    pub fn dest(&self) -> Option<Value> {
        match self {
            Instruction::Alloca { dest, .. }
            | Instruction::DynAlloca { dest, .. }
            | Instruction::Load { dest, .. }
            | Instruction::BinOp { dest, .. }
            | Instruction::UnaryOp { dest, .. }
            | Instruction::Cmp { dest, .. }
            | Instruction::GetElementPtr { dest, .. }
            | Instruction::Cast { dest, .. }
            | Instruction::Copy { dest, .. }
            | Instruction::GlobalAddr { dest, .. }
            | Instruction::VaArg { dest, .. }
            | Instruction::AtomicRmw { dest, .. }
            | Instruction::AtomicCmpxchg { dest, .. }
            | Instruction::AtomicLoad { dest, .. }
            | Instruction::Phi { dest, .. }
            | Instruction::LabelAddr { dest, .. }
            | Instruction::GetReturnF64Second { dest }
            | Instruction::GetReturnF32Second { dest }
            | Instruction::GetReturnF128Second { dest }
            | Instruction::Select { dest, .. }
            | Instruction::StackSave { dest }
            | Instruction::ParamRef { dest, .. } => Some(*dest),
            Instruction::Call { info, .. }
            | Instruction::CallIndirect { info, .. } => info.dest,
            Instruction::Intrinsic { dest, .. } => *dest,
            Instruction::Store { .. }
            | Instruction::Memcpy { .. }
            | Instruction::VaStart { .. }
            | Instruction::VaEnd { .. }
            | Instruction::VaCopy { .. }
            | Instruction::VaArgStruct { .. }
            | Instruction::AtomicStore { .. }
            | Instruction::Fence { .. }
            | Instruction::SetReturnF64Second { .. }
            | Instruction::SetReturnF32Second { .. }
            | Instruction::SetReturnF128Second { .. }
            | Instruction::InlineAsm { .. }
            | Instruction::StackRestore { .. } => None,
        }
    }

    /// Returns the result IR type of this instruction, if any.
    /// Used to determine stack slot sizes for 128-bit values.
    pub fn result_type(&self) -> Option<IrType> {
        match self {
            Instruction::Load { ty, .. } => Some(*ty),
            Instruction::BinOp { ty, .. } => Some(*ty),
            Instruction::UnaryOp { ty, .. } => Some(*ty),
            Instruction::Cmp { .. } => Some(IrType::I8), // comparisons produce i8
            Instruction::Cast { to_ty, .. } => Some(*to_ty),
            Instruction::Call { info, .. }
            | Instruction::CallIndirect { info, .. } => Some(info.return_type),
            Instruction::VaArg { result_ty, .. } => Some(*result_ty),
            Instruction::AtomicRmw { ty, .. } => Some(*ty),
            Instruction::AtomicCmpxchg { ty, returns_bool, .. } => {
                if *returns_bool { Some(IrType::I8) } else { Some(*ty) }
            }
            Instruction::AtomicLoad { ty, .. } => Some(*ty),
            // Alloca, GEP, GlobalAddr, Copy, DynAlloca, LabelAddr, StackSave produce pointers or copy types
            Instruction::Alloca { .. } | Instruction::DynAlloca { .. }
            | Instruction::GetElementPtr { .. } | Instruction::GlobalAddr { .. }
            | Instruction::LabelAddr { .. }
            | Instruction::StackSave { .. } => Some(IrType::Ptr),
            Instruction::Copy { .. } => None, // unknown without tracking
            Instruction::Phi { ty, .. } => Some(*ty),
            Instruction::Select { ty, .. } => Some(*ty),
            Instruction::Intrinsic { op, .. } => match op {
                IntrinsicOp::SqrtF32 | IntrinsicOp::FabsF32 => Some(IrType::F32),
                IntrinsicOp::SqrtF64 | IntrinsicOp::FabsF64 => Some(IrType::F64),
                _ => None,
            },
            Instruction::GetReturnF128Second { .. } => Some(IrType::F128),
            Instruction::ParamRef { ty, .. } => Some(*ty),
            _ => None,
        }
    }

    /// Call `f(value_id)` for every Value ID used as an operand in this instruction.
    ///
    /// This is the canonical value visitor. All passes that need to enumerate
    /// instruction operands should use this method to avoid duplicating the
    /// match block.
    #[inline]
    pub fn for_each_used_value(&self, mut f: impl FnMut(u32)) {
        #[inline(always)]
        fn visit_op(op: &Operand, f: &mut impl FnMut(u32)) {
            if let Operand::Value(v) = op { f(v.0); }
        }
        match self {
            Instruction::Alloca { .. } | Instruction::GlobalAddr { .. }
            | Instruction::LabelAddr { .. } | Instruction::StackSave { .. }
            | Instruction::Fence { .. } | Instruction::GetReturnF64Second { .. }
            | Instruction::GetReturnF32Second { .. }
            | Instruction::GetReturnF128Second { .. }
            | Instruction::ParamRef { .. } => {}

            Instruction::Load { ptr, .. } => f(ptr.0),
            Instruction::Store { val, ptr, .. } => { visit_op(val, &mut f); f(ptr.0); }
            Instruction::DynAlloca { size, .. } => visit_op(size, &mut f),
            Instruction::BinOp { lhs, rhs, .. }
            | Instruction::Cmp { lhs, rhs, .. } => { visit_op(lhs, &mut f); visit_op(rhs, &mut f); }
            Instruction::UnaryOp { src, .. }
            | Instruction::Cast { src, .. }
            | Instruction::Copy { src, .. } => visit_op(src, &mut f),
            Instruction::Call { info, .. } => {
                for arg in &info.args { visit_op(arg, &mut f); }
            }
            Instruction::CallIndirect { func_ptr, info } => {
                visit_op(func_ptr, &mut f);
                for arg in &info.args { visit_op(arg, &mut f); }
            }
            Instruction::GetElementPtr { base, offset, .. } => { f(base.0); visit_op(offset, &mut f); }
            Instruction::Memcpy { dest, src, .. } => { f(dest.0); f(src.0); }
            Instruction::VaArg { va_list_ptr, .. }
            | Instruction::VaStart { va_list_ptr }
            | Instruction::VaEnd { va_list_ptr } => f(va_list_ptr.0),
            Instruction::VaCopy { dest_ptr, src_ptr } => { f(dest_ptr.0); f(src_ptr.0); }
            Instruction::VaArgStruct { dest_ptr, va_list_ptr, .. } => { f(dest_ptr.0); f(va_list_ptr.0); }
            Instruction::AtomicRmw { ptr, val, .. } => { visit_op(ptr, &mut f); visit_op(val, &mut f); }
            Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => {
                visit_op(ptr, &mut f); visit_op(expected, &mut f); visit_op(desired, &mut f);
            }
            Instruction::AtomicLoad { ptr, .. } => visit_op(ptr, &mut f),
            Instruction::AtomicStore { ptr, val, .. } => { visit_op(ptr, &mut f); visit_op(val, &mut f); }
            Instruction::Phi { incoming, .. } => {
                for (op, _) in incoming { visit_op(op, &mut f); }
            }
            Instruction::SetReturnF64Second { src }
            | Instruction::SetReturnF32Second { src }
            | Instruction::SetReturnF128Second { src } => visit_op(src, &mut f),
            Instruction::InlineAsm { outputs, inputs, .. } => {
                for (_, ptr, _) in outputs { f(ptr.0); }
                for (_, op, _) in inputs { visit_op(op, &mut f); }
            }
            Instruction::Intrinsic { dest_ptr, args, .. } => {
                if let Some(ptr) = dest_ptr { f(ptr.0); }
                for arg in args { visit_op(arg, &mut f); }
            }
            Instruction::Select { cond, true_val, false_val, .. } => {
                visit_op(cond, &mut f); visit_op(true_val, &mut f); visit_op(false_val, &mut f);
            }
            Instruction::StackRestore { ptr } => f(ptr.0),
        }
    }

    /// Collect all Value IDs used (as operands, not defined) by this instruction.
    pub fn used_values(&self) -> Vec<u32> {
        let mut used = Vec::new();
        self.for_each_used_value(|id| used.push(id));
        used
    }
}

impl Terminator {
    /// Call `f(value_id)` for every Value ID used as an operand in this terminator.
    #[inline]
    pub fn for_each_used_value(&self, mut f: impl FnMut(u32)) {
        match self {
            Terminator::Return(Some(Operand::Value(v))) => f(v.0),
            Terminator::CondBranch { cond: Operand::Value(v), .. } => f(v.0),
            Terminator::IndirectBranch { target: Operand::Value(v), .. } => f(v.0),
            Terminator::Switch { val: Operand::Value(v), .. } => f(v.0),
            _ => {}
        }
    }

    /// Collect all Value IDs used by this terminator.
    pub fn used_values(&self) -> Vec<u32> {
        let mut used = Vec::new();
        self.for_each_used_value(|id| used.push(id));
        used
    }
}
