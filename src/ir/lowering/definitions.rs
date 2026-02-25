//! Data structure definitions shared across the lowering module.
//!
//! Contains the core types that multiple lowering sub-modules reference:
//! variable metadata (VarInfo/LocalInfo/GlobalInfo), declaration analysis
//! (DeclAnalysis), lvalue representation, switch context, function signature
//! metadata, and typedef helpers.

use std::rc::Rc;
use crate::common::fx_hash::FxHashMap;
use crate::ir::reexports::{
    BlockId,
    GlobalInit,
    IrConst,
    IrParam,
    Value,
};
use crate::common::types::{AddressSpace, IrType, RcLayout, CType};

/// Type metadata shared between local and global variables.
///
/// Both `LocalInfo` and `GlobalInfo` embed this struct via `Deref`, so field
/// access like `info.ty` or `info.is_array` works transparently on either type.
/// The `Lowerer::lookup_var_info()` helper returns `&VarInfo` for cases that
/// only need these shared fields, eliminating the duplicated locals-then-globals
/// lookup pattern.
#[derive(Debug, Clone)]
pub(super) struct VarInfo {
    /// The IR type of the variable (I8 for char, I32 for int, I64 for long, Ptr for pointers).
    pub ty: IrType,
    /// Element size for arrays (used for pointer arithmetic on subscript).
    /// For non-arrays this is 0.
    pub elem_size: usize,
    /// Whether this is an array (the alloca IS the base address, not a pointer to one).
    pub is_array: bool,
    /// For pointers and arrays, the type of the pointed-to/element type.
    /// Used for correct loads through pointer dereference and subscript.
    pub pointee_type: Option<IrType>,
    /// If this is a struct/union variable, its layout for member access.
    /// Uses Rc for cheap cloning (struct layouts are shared, never mutated after creation).
    pub struct_layout: Option<RcLayout>,
    /// Whether this variable is a struct (not a pointer to struct).
    pub is_struct: bool,
    /// For multi-dimensional arrays: stride (in bytes) per dimension level.
    /// E.g., for int a[2][3][4], strides = [48, 16, 4] (row_size, inner_row, elem).
    /// Empty for non-arrays or 1D arrays (use elem_size instead).
    pub array_dim_strides: Vec<usize>,
    /// Full C type for precise multi-level pointer type resolution.
    pub c_type: Option<CType>,
    /// Whether this is a pointer-to-function-pointer (e.g., int (**fpp)(int, int)).
    /// Distinguished from direct function pointers despite similar CType representation.
    pub is_ptr_to_func_ptr: bool,
    /// Address space qualifier on the variable itself (not on a pointer type).
    /// E.g., `extern struct pcpu_hot __seg_gs const_pcpu_hot` has SegGs.
    /// Used to emit %gs:/%fs: segment prefixes on member access loads/stores.
    pub address_space: AddressSpace,
    /// Explicit alignment from _Alignas or __attribute__((aligned(N))) on the
    /// variable declaration. When set, _Alignof(var) returns this value (or the
    /// natural type alignment, whichever is larger) per C11 6.2.8p3.
    pub explicit_alignment: Option<usize>,
}

/// Information about a local variable stored in an alloca.
/// Derefs to `VarInfo` for shared field access.
#[derive(Debug, Clone)]
pub(super) struct LocalInfo {
    /// Shared type metadata (ty, elem_size, is_array, pointee_type, etc.)
    pub var: VarInfo,
    /// The Value (alloca) holding the address of this local.
    pub alloca: Value,
    /// The total allocation size of this variable (for sizeof).
    pub alloc_size: usize,
    /// Whether this variable has _Bool type (needs value clamping to 0/1).
    pub is_bool: bool,
    /// For static local variables: the mangled global name. When set, accesses should
    /// emit a fresh GlobalAddr instruction instead of using `alloca`, because the
    /// declaration may be in an unreachable basic block (skipped by goto/switch).
    pub static_global_name: Option<Rc<str>>,
    /// For VLA function parameters: runtime stride Values per dimension level.
    /// Parallel to `array_dim_strides`. When `Some(value)`, use the runtime Value
    /// instead of the compile-time stride. This supports parameters like
    /// `int m[rows][cols]` where `cols` is a runtime variable.
    pub vla_strides: Vec<Option<Value>>,
    /// For VLA local variables: the runtime Value holding sizeof(this_variable).
    /// Used when sizeof is applied to a VLA local variable.
    pub vla_size: Option<Value>,
    /// For register variables with __asm__("regname"): the specific register name.
    /// Used to rewrite inline asm "r" constraints to specific register constraints.
    pub asm_register: Option<Rc<str>>,
    /// Whether this register variable has been "initialized" -- either by a declaration
    /// initializer (e.g., `register long x8 __asm__("x8") = n;`) or by being used as an
    /// inline asm output operand. When true, reads come from the alloca; when false,
    /// reads emit inline asm to sample the physical hardware register.
    pub asm_register_has_init: bool,
    /// __attribute__((cleanup(func))): cleanup function to call with &var when scope exits.
    /// The function is called as func(&var) with a pointer to the variable.
    pub cleanup_fn: Option<Rc<str>>,
    /// Whether this variable was declared with `const` qualifier.
    /// Used by _Generic matching to distinguish e.g. `const int *` from `int *`,
    /// since CType does not track const/volatile qualifiers.
    pub is_const: bool,
}

impl std::ops::Deref for LocalInfo {
    type Target = VarInfo;
    fn deref(&self) -> &VarInfo { &self.var }
}

impl std::ops::DerefMut for LocalInfo {
    fn deref_mut(&mut self) -> &mut VarInfo { &mut self.var }
}

/// Information about a global variable tracked by the lowerer.
/// Derefs to `VarInfo` for shared field access.
#[derive(Debug, Clone)]
pub(super) struct GlobalInfo {
    /// Shared type metadata (ty, elem_size, is_array, pointee_type, etc.)
    pub var: VarInfo,
    /// For global register variables declared with `register <type> <name> __asm__("reg")`.
    /// When set, no storage is emitted; reads/writes map directly to the named register.
    pub asm_register: Option<Rc<str>>,
}

impl std::ops::Deref for GlobalInfo {
    type Target = VarInfo;
    fn deref(&self) -> &VarInfo { &self.var }
}

impl std::ops::DerefMut for GlobalInfo {
    fn deref_mut(&mut self) -> &mut VarInfo { &mut self.var }
}

/// Pre-computed declaration analysis shared between `lower_local_decl` and
/// `lower_global_decl`. Extracts the common type analysis (base type, array info,
/// pointer info, struct layout, etc.) that both paths need, eliminating the
/// ~80 lines of duplicated computation.
#[derive(Debug)]
pub(super) struct DeclAnalysis {
    /// The base IR type from the type specifier (before pointer/array derivation).
    pub base_ty: IrType,
    /// The final variable IR type (Ptr for pointers/arrays-of-pointers, else base_ty).
    pub var_ty: IrType,
    /// Total allocation size in bytes.
    pub alloc_size: usize,
    /// Element size for arrays (stride for indexing).
    pub elem_size: usize,
    /// Whether this declaration is an array.
    pub is_array: bool,
    /// Whether this declaration is a pointer.
    pub is_pointer: bool,
    /// Per-dimension strides for multi-dimensional arrays.
    pub array_dim_strides: Vec<usize>,
    /// Whether this is an array of pointers (int *arr[N]).
    pub is_array_of_pointers: bool,
    /// Whether this is an array of function pointers.
    pub is_array_of_func_ptrs: bool,
    /// Struct/union layout (for struct variables or pointer-to-struct).
    /// Uses Rc for cheap cloning.
    pub struct_layout: Option<RcLayout>,
    /// Whether this is a direct struct variable (not pointer-to or array-of).
    pub is_struct: bool,
    /// Actual allocation size (uses struct layout size for non-array structs).
    pub actual_alloc_size: usize,
    /// Pointee type for pointer/array types.
    pub pointee_type: Option<IrType>,
    /// Full C type for multi-level pointer resolution.
    pub c_type: Option<CType>,
    /// Whether this is a _Bool variable (not pointer or array of _Bool).
    pub is_bool: bool,
    /// The element IR type for arrays (accounts for typedef'd arrays).
    pub elem_ir_ty: IrType,
    /// Whether this is a pointer-to-function-pointer (e.g., int (**fpp)(int, int)).
    /// Distinguished from direct function pointers despite similar CType representation.
    pub is_ptr_to_func_ptr: bool,
}

impl DeclAnalysis {
    /// Apply `__attribute__((vector_size(N)))` to this declaration analysis.
    /// When a non-typedef variable is declared with an inline vector_size attribute,
    /// the CType, allocation size, and IR type must be updated to reflect the vector type.
    /// Without this, the variable is treated as a plain scalar (e.g., `float` instead of
    /// `CType::Vector(Float, 16)`), causing incorrect element-wise access and codegen.
    pub fn apply_vector_size(&mut self, total_size: usize) {
        // Wrap the C type in CType::Vector
        if let Some(ref ct) = self.c_type {
            self.c_type = Some(CType::Vector(Box::new(ct.clone()), total_size));
        }
        // Vectors are aggregates (like structs): passed by pointer, allocated as a blob
        self.actual_alloc_size = total_size;
        self.alloc_size = total_size;
        self.var_ty = IrType::Ptr;
        // base_ty stays as the element type (e.g., F32 for float vectors)
        // This matches what happens when typedef resolution produces CType::Vector,
        // which maps to IrType::Ptr via from_ctype but the element type is still
        // accessible through the CType.
    }
}

/// Information about a VLA dimension in a function parameter type.
#[derive(Debug)]
pub(super) struct VlaDimInfo {
    /// Whether this dimension is a VLA (runtime variable).
    pub is_vla: bool,
    /// The name of the variable providing the dimension (e.g., "cols").
    pub dim_expr_name: Rc<str>,
    /// If not VLA, the constant size value.
    pub const_size: Option<i64>,
    /// The sizeof the element type at this level (for computing strides).
    pub base_elem_size: usize,
}

/// Represents an lvalue - something that can be assigned to.
/// Contains the address (as an IR Value) where the data resides.
#[derive(Debug, Clone)]
pub(super) enum LValue {
    /// A direct variable: the alloca is the address.
    Variable(Value),
    /// An address computed at runtime (e.g., arr[i], *ptr).
    /// The AddressSpace tracks segment overrides (e.g., __seg_gs for per-CPU vars).
    Address(Value, AddressSpace),
}

/// A single level of switch statement context, pushed/popped as switches nest.
#[derive(Debug)]
pub(super) struct SwitchFrame {
    pub cases: Vec<(i64, BlockId)>,
    /// GNU case ranges: (low, high, label)
    pub case_ranges: Vec<(i64, i64, BlockId)>,
    pub default_label: Option<BlockId>,
    pub expr_type: IrType,
}

/// Consolidated function signature metadata.
/// Replaces 10 parallel HashMaps with a single struct per function.
#[derive(Debug, Clone)]
pub(super) struct FuncSig {
    /// IR return type for inserting narrowing casts after calls.
    pub return_type: IrType,
    /// CType of the return value (for pointer-returning and struct-returning functions).
    pub return_ctype: Option<CType>,
    /// IR types of each parameter, for inserting implicit argument casts.
    pub param_types: Vec<IrType>,
    /// CTypes of each parameter, for complex type argument conversions.
    pub param_ctypes: Vec<CType>,
    /// Flags indicating which parameters are _Bool (need normalization to 0/1).
    pub param_bool_flags: Vec<bool>,
    /// Whether this function is variadic.
    pub is_variadic: bool,
    /// If the function returns a struct > 16 bytes, the struct size (uses hidden sret pointer).
    pub sret_size: Option<usize>,
    /// If the function returns a struct of 9-16 bytes via two registers, the struct size.
    pub two_reg_ret_size: Option<usize>,
    /// SysV ABI eightbyte classification for the return struct (if two_reg_ret_size is set).
    /// Used to determine which eightbytes go in GP vs SSE registers on return.
    /// Empty if not applicable (non-struct return, sret, or non-x86-64).
    pub ret_eightbyte_classes: Vec<crate::common::types::EightbyteClass>,
    /// Per-parameter struct sizes for by-value struct passing ABI.
    /// Each entry is Some(size) if that parameter is a struct/union, None otherwise.
    pub param_struct_sizes: Vec<Option<usize>>,
    /// Per-parameter SysV ABI eightbyte classification for struct params.
    /// Each entry is the classification for that parameter (empty vec for non-struct params).
    pub param_struct_classes: Vec<Vec<crate::common::types::EightbyteClass>>,
    /// Per-parameter RISC-V LP64D float field classification for struct params.
    /// Each entry is Some(..) for struct params that qualify for FP register passing.
    pub param_riscv_float_classes: Vec<Option<crate::common::types::RiscvFloatClass>>,
}

impl FuncSig {
    /// Create a minimal FuncSig for a function pointer variable.
    /// Sets all optional/ABI fields to their defaults (empty/None/false).
    pub fn for_ptr(return_type: IrType, param_types: Vec<IrType>) -> Self {
        FuncSig {
            return_type,
            return_ctype: None,
            param_types,
            param_ctypes: Vec::new(),
            param_bool_flags: Vec::new(),
            is_variadic: false,
            sret_size: None,
            two_reg_ret_size: None,
            ret_eightbyte_classes: Vec::new(),
            param_struct_sizes: Vec::new(),
            param_struct_classes: Vec::new(),
            param_riscv_float_classes: Vec::new(),
        }
    }
}

impl DeclAnalysis {
    /// Determine the IR element type for a global variable's initializer.
    ///
    /// When struct initializers are emitted as byte arrays (Vec<IrConst::I8>),
    /// the global's element type must be I8 instead of the declared type.
    /// This logic is shared between `lower_global_decl` and `lower_local_static_decl`.
    pub fn resolve_global_ty(&self, init: &GlobalInit) -> IrType {
        if matches!(init, GlobalInit::Array(vals) if !vals.is_empty() && matches!(vals[0], IrConst::I8(_)))
            || (self.is_struct && matches!(init, GlobalInit::Array(_)))
        {
            IrType::I8
        } else if matches!(init, GlobalInit::Array(_)) {
            // Check if this is a vector type or array of vectors.
            // Vector globals store element values (e.g., 4x I32 for int vector_size(16)).
            // Use the element type instead of Ptr so each element is emitted at the
            // correct width (e.g., .long for I32 instead of .quad for Ptr on 64-bit).
            if let Some(elem_ir_ty) = self.vector_element_ir_type() {
                return elem_ir_ty;
            }
            // Complex arrays store scalar component pairs (e.g., _Complex float arr[N]
            // stores 2*N F32 values). The fallback type must be the component's IR type
            // (F32/F64/F128) rather than Ptr, so that zero coalescing emits the correct
            // number of bytes per zero element.
            if let Some(complex_ir_ty) = self.complex_component_ir_type() {
                return complex_ir_ty;
            }
            self.var_ty
        } else {
            self.var_ty
        }
    }

    /// Get the IR element type if this is a vector or array-of-vectors type.
    /// Returns None if this is not a vector-related type.
    fn vector_element_ir_type(&self) -> Option<IrType> {
        let ct = self.c_type.as_ref()?;
        // Direct vector type: CType::Vector(elem, size)
        if let Some((elem_ct, _)) = ct.vector_info() {
            return Some(IrType::from_ctype(elem_ct));
        }
        // Array of vectors: CType::Array(Vector(elem, size), count)
        let mut inner = ct;
        while let CType::Array(ref elem, _) = inner {
            inner = elem.as_ref();
        }
        if let Some((elem_ct, _)) = inner.vector_info() {
            return Some(IrType::from_ctype(elem_ct));
        }
        None
    }

    /// Get the IR component type if this is a complex or array-of-complex type.
    /// Complex arrays are stored as flat scalar pairs (real, imag), so the fallback
    /// type for data emission must be the component type, not Ptr.
    fn complex_component_ir_type(&self) -> Option<IrType> {
        let ct = self.c_type.as_ref()?;
        // Unwrap array layers to find the element type
        let mut inner = ct;
        while let CType::Array(ref elem, _) = inner {
            inner = elem.as_ref();
        }
        if inner.is_complex() {
            Some(IrType::from_ctype(&inner.complex_component_type()))
        } else {
            None
        }
    }
}

/// Metadata about known functions (signatures, variadic status, ABI handling).
/// Uses a consolidated FuncSig per function instead of parallel HashMaps.
#[derive(Debug, Default)]
pub(super) struct FunctionMeta {
    /// Function name -> consolidated signature.
    pub sigs: FxHashMap<Rc<str>, FuncSig>,
    /// Function pointer variable name -> signature (return type + param types).
    pub ptr_sigs: FxHashMap<Rc<str>, FuncSig>,
}

/// Tracks how each original C parameter maps to IR parameters after ABI decomposition.
///
/// Used by `build_ir_params` to record what happened to each original parameter,
/// so `allocate_function_params` knows which registration method to call.
#[derive(Debug)]
pub(super) enum ParamKind {
    /// Normal parameter: 1 IR param at the given index
    Normal(usize),
    /// Struct/union or non-decomposed complex parameter passed by value
    Struct(usize),
    /// Complex parameter passed as two decomposed FP params (real_ir_idx, imag_ir_idx)
    ComplexDecomposed(usize, usize),
    /// Complex float packed into single F64 (x86-64 only)
    ComplexFloatPacked(usize),
}

/// Result of building the IR parameter list for a function.
///
/// Produced by `build_ir_params`, consumed by `allocate_function_params` and
/// `finalize_function`.
pub(super) struct IrParamBuildResult {
    /// The IR parameters to use for the function signature
    pub params: Vec<IrParam>,
    /// Per-original-parameter: how it maps to IR params (indexed by original param index)
    pub param_kinds: Vec<ParamKind>,
    /// Whether the function uses sret (hidden first pointer param)
    pub uses_sret: bool,
}

// --- Construction helpers ---

impl VarInfo {
    /// Construct VarInfo from a DeclAnalysis (shared by both LocalInfo and GlobalInfo).
    pub(super) fn from_analysis(da: &DeclAnalysis) -> Self {
        VarInfo {
            ty: da.var_ty,
            elem_size: da.elem_size,
            is_array: da.is_array,
            pointee_type: da.pointee_type,
            struct_layout: da.struct_layout.clone(),
            is_struct: da.is_struct,
            array_dim_strides: da.array_dim_strides.clone(),
            c_type: da.c_type.clone(),
            is_ptr_to_func_ptr: da.is_ptr_to_func_ptr,
            address_space: AddressSpace::Default,
            explicit_alignment: None,
        }
    }
}

impl GlobalInfo {
    /// Construct a GlobalInfo from a DeclAnalysis, avoiding repeated field construction.
    pub(super) fn from_analysis(da: &DeclAnalysis) -> Self {
        GlobalInfo { var: VarInfo::from_analysis(da), asm_register: None }
    }
}

impl LocalInfo {
    /// Construct a LocalInfo for a regular (non-static) local variable from DeclAnalysis.
    pub(super) fn from_analysis(da: &DeclAnalysis, alloca: Value, is_const: bool) -> Self {
        LocalInfo {
            var: VarInfo::from_analysis(da),
            alloca,
            alloc_size: da.actual_alloc_size,
            is_bool: da.is_bool,
            static_global_name: None,
            vla_strides: vec![],
            vla_size: None,
            asm_register: None,
            asm_register_has_init: false,
            cleanup_fn: None,
            is_const,
        }
    }

    /// Construct a LocalInfo for a static local variable from DeclAnalysis.
    pub(super) fn for_static(da: &DeclAnalysis, static_name: Rc<str>, is_const: bool) -> Self {
        LocalInfo {
            var: VarInfo::from_analysis(da),
            alloca: Value(0), // placeholder; not used for static locals
            alloc_size: da.actual_alloc_size,
            is_bool: da.is_bool,
            static_global_name: Some(static_name),
            vla_strides: vec![],
            vla_size: None,
            asm_register: None,
            asm_register_has_init: false,
            cleanup_fn: None,
            is_const,
        }
    }
}
