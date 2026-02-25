use std::rc::Rc;
use crate::common::source::Span;
use crate::common::types::AddressSpace;

/// A complete translation unit (one C source file).
#[derive(Debug)]
pub struct TranslationUnit {
    pub decls: Vec<ExternalDecl>,
}

/// Top-level declarations in a translation unit.
#[derive(Debug)]
pub enum ExternalDecl {
    FunctionDef(FunctionDef),
    Declaration(Declaration),
    /// Top-level asm("...") directive - emitted verbatim in assembly output.
    TopLevelAsm(String),
}

/// Attributes that can be applied to function definitions via storage-class
/// specifiers (static, inline, extern) and GCC __attribute__((...)) syntax.
///
/// Boolean attributes are stored as a packed bitfield (`flags`) for memory
/// efficiency — 13 booleans collapse from 13 bytes into 2 bytes. Accessor
/// methods provide the same API as the old struct fields.
///
/// Non-boolean attributes (`section`, `visibility`) remain as `Option<String>`.
#[derive(Clone, Default)]
pub struct FunctionAttributes {
    /// Packed boolean flags — see `FuncAttrFlag` constants.
    flags: u16,
    /// __attribute__((section("..."))) - place in specific ELF section
    pub section: Option<String>,
    /// __attribute__((visibility("hidden"|"default"|...)))
    pub visibility: Option<String>,
    /// __attribute__((symver("name@@VERSION"))) - symbol version alias
    pub symver: Option<String>,
}

/// Bit masks for boolean flags in `FunctionAttributes::flags`.
///
/// Each attribute occupies one bit, allowing cheap test/set/clear operations.
/// New attributes can be added by defining the next power-of-two constant.
pub mod func_attr_flag {
    pub const STATIC: u16        = 1 << 0;
    pub const INLINE: u16        = 1 << 1;
    /// `extern` storage class on the function definition.
    pub const EXTERN: u16        = 1 << 2;
    /// `__attribute__((gnu_inline))` — forces GNU89 inline semantics.
    pub const GNU_INLINE: u16    = 1 << 3;
    /// `__attribute__((always_inline))` — must always be inlined.
    pub const ALWAYS_INLINE: u16 = 1 << 4;
    /// `__attribute__((noinline))` — must never be inlined.
    pub const NOINLINE: u16      = 1 << 5;
    /// `__attribute__((constructor))` — run before main.
    pub const CONSTRUCTOR: u16   = 1 << 6;
    /// `__attribute__((destructor))` — run after main.
    pub const DESTRUCTOR: u16    = 1 << 7;
    /// `__attribute__((weak))` — emit as a weak symbol.
    pub const WEAK: u16          = 1 << 8;
    /// `__attribute__((used))` — prevent dead code elimination.
    pub const USED: u16          = 1 << 9;
    /// `__attribute__((fastcall))` — i386 fastcall convention (first 2 int args in ecx/edx).
    pub const FASTCALL: u16      = 1 << 10;
    /// `__attribute__((naked))` — emit no prologue/epilogue; function body is pure asm.
    pub const NAKED: u16         = 1 << 11;
    /// `__attribute__((noreturn))` or `_Noreturn` — function never returns.
    pub const NORETURN: u16      = 1 << 12;
}

impl FunctionAttributes {
    /// Create a new `FunctionAttributes` with all flags cleared.
    pub fn new() -> Self {
        Self::default()
    }

    // --- flag getters ---

    #[inline] pub fn is_static(&self) -> bool        { self.flags & func_attr_flag::STATIC != 0 }
    #[inline] pub fn is_inline(&self) -> bool         { self.flags & func_attr_flag::INLINE != 0 }
    #[inline] pub fn is_extern(&self) -> bool         { self.flags & func_attr_flag::EXTERN != 0 }
    #[inline] pub fn is_gnu_inline(&self) -> bool     { self.flags & func_attr_flag::GNU_INLINE != 0 }
    #[inline] pub fn is_always_inline(&self) -> bool  { self.flags & func_attr_flag::ALWAYS_INLINE != 0 }
    #[inline] pub fn is_noinline(&self) -> bool       { self.flags & func_attr_flag::NOINLINE != 0 }
    #[inline] pub fn is_constructor(&self) -> bool    { self.flags & func_attr_flag::CONSTRUCTOR != 0 }
    #[inline] pub fn is_destructor(&self) -> bool     { self.flags & func_attr_flag::DESTRUCTOR != 0 }
    #[inline] pub fn is_weak(&self) -> bool           { self.flags & func_attr_flag::WEAK != 0 }
    #[inline] pub fn is_used(&self) -> bool           { self.flags & func_attr_flag::USED != 0 }
    #[inline] pub fn is_fastcall(&self) -> bool       { self.flags & func_attr_flag::FASTCALL != 0 }
    #[inline] pub fn is_naked(&self) -> bool          { self.flags & func_attr_flag::NAKED != 0 }
    #[inline] pub fn is_noreturn(&self) -> bool       { self.flags & func_attr_flag::NORETURN != 0 }

    // --- flag setters ---

    #[inline] pub fn set_static(&mut self, v: bool)        { self.set_flag(func_attr_flag::STATIC, v) }
    #[inline] pub fn set_inline(&mut self, v: bool)        { self.set_flag(func_attr_flag::INLINE, v) }
    #[inline] pub fn set_extern(&mut self, v: bool)        { self.set_flag(func_attr_flag::EXTERN, v) }
    #[inline] pub fn set_gnu_inline(&mut self, v: bool)    { self.set_flag(func_attr_flag::GNU_INLINE, v) }
    #[inline] pub fn set_always_inline(&mut self, v: bool) { self.set_flag(func_attr_flag::ALWAYS_INLINE, v) }
    #[inline] pub fn set_noinline(&mut self, v: bool)      { self.set_flag(func_attr_flag::NOINLINE, v) }
    #[inline] pub fn set_constructor(&mut self, v: bool)   { self.set_flag(func_attr_flag::CONSTRUCTOR, v) }
    #[inline] pub fn set_destructor(&mut self, v: bool)    { self.set_flag(func_attr_flag::DESTRUCTOR, v) }
    #[inline] pub fn set_weak(&mut self, v: bool)          { self.set_flag(func_attr_flag::WEAK, v) }
    #[inline] pub fn set_used(&mut self, v: bool)          { self.set_flag(func_attr_flag::USED, v) }
    #[inline] pub fn set_fastcall(&mut self, v: bool)      { self.set_flag(func_attr_flag::FASTCALL, v) }
    #[inline] pub fn set_naked(&mut self, v: bool)        { self.set_flag(func_attr_flag::NAKED, v) }
    #[inline] pub fn set_noreturn(&mut self, v: bool)     { self.set_flag(func_attr_flag::NORETURN, v) }

    #[inline]
    fn set_flag(&mut self, mask: u16, v: bool) {
        if v { self.flags |= mask; } else { self.flags &= !mask; }
    }
}

impl std::fmt::Debug for FunctionAttributes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionAttributes")
            .field("is_static", &self.is_static())
            .field("is_inline", &self.is_inline())
            .field("is_extern", &self.is_extern())
            .field("is_gnu_inline", &self.is_gnu_inline())
            .field("is_always_inline", &self.is_always_inline())
            .field("is_noinline", &self.is_noinline())
            .field("is_constructor", &self.is_constructor())
            .field("is_destructor", &self.is_destructor())
            .field("is_weak", &self.is_weak())
            .field("is_used", &self.is_used())
            .field("is_fastcall", &self.is_fastcall())
            .field("is_naked", &self.is_naked())
            .field("is_noreturn", &self.is_noreturn())
            .field("section", &self.section)
            .field("visibility", &self.visibility)
            .finish()
    }
}

/// A function definition.
#[derive(Debug)]
pub struct FunctionDef {
    pub return_type: TypeSpecifier,
    pub name: Rc<str>,
    pub params: Vec<ParamDecl>,
    pub variadic: bool,
    pub body: CompoundStmt,
    /// Function attributes (storage class, inline hints, GCC __attribute__).
    pub attrs: FunctionAttributes,
    pub is_kr: bool,
    pub span: Span,
}

/// A parameter declaration.
#[derive(Debug, Clone)]
pub struct ParamDecl {
    pub type_spec: TypeSpecifier,
    pub name: Option<Rc<str>>,
    /// For function pointer parameters, the parameter types of the pointed-to function.
    /// E.g., for `float (*func)(float, float)`, this holds the two float param decls.
    pub fptr_params: Option<Vec<ParamDecl>>,
    /// Whether this parameter's base type has a `const` qualifier.
    /// Used by _Generic matching to distinguish e.g. `const int *` from `int *`.
    pub is_const: bool,
    /// VLA size expressions from the outermost array dimension that was decayed to pointer.
    /// E.g., for `void foo(int a, int b[a++])`, the expression `a++` is stored here
    /// so its side effects can be evaluated at function entry during IR lowering.
    pub vla_size_exprs: Vec<Expr>,
    /// The number of `*` inside the parenthesized declarator for function pointer params.
    /// `(*fp)` has depth 1, `(**fpp)` has depth 2. A depth >= 2 means pointer-to-function-
    /// pointer. This distinguishes `int (**fpp)(int)` from `void *(*fp)(size_t)` which
    /// otherwise have identical CType representations.
    pub fptr_inner_ptr_depth: u32,
}

/// Bit masks for boolean flags in `Declaration::flags`.
///
/// Each flag occupies one bit of a `u16`, matching the pattern used by
/// `FunctionAttributes` and `DeclAttributes`.  New flags can be added by
/// defining the next power-of-two constant.
pub mod decl_flag {
    /// `static` storage class.
    pub const STATIC: u16            = 1 << 0;
    /// `extern` storage class.
    pub const EXTERN: u16            = 1 << 1;
    /// `typedef` storage class.
    pub const TYPEDEF: u16           = 1 << 2;
    /// `const` type qualifier.
    pub const CONST: u16             = 1 << 3;
    /// `volatile` type qualifier.
    pub const VOLATILE: u16          = 1 << 4;
    /// GCC `-fcommon` tentative definition (no initialiser, no `extern`).
    pub const COMMON: u16            = 1 << 5;
    /// `_Thread_local` or `__thread` storage class.
    pub const THREAD_LOCAL: u16      = 1 << 6;
    /// `__attribute__((transparent_union))` applied to a typedef.
    pub const TRANSPARENT_UNION: u16 = 1 << 7;
    /// `inline` function specifier on a declaration (not a definition).
    /// Used to implement C99 6.7.4p7: a function provides an external definition
    /// only if not ALL file-scope declarations include `inline`.
    pub const INLINE: u16            = 1 << 8;
}

/// A variable/type declaration.
///
/// Boolean storage-class / qualifier / attribute flags are stored as a packed
/// bitfield (`flags`) for memory efficiency — 9 booleans collapse from 9 bytes
/// into 2 bytes.  Accessor methods provide the same API as the old struct
/// fields.
#[derive(Clone)]
pub struct Declaration {
    pub type_spec: TypeSpecifier,
    pub declarators: Vec<InitDeclarator>,
    /// Packed boolean flags — see `decl_flag` constants.
    flags: u16,
    /// Alignment override from _Alignas(N) or __attribute__((aligned(N))).
    pub alignment: Option<usize>,
    /// Type specifier from _Alignas(type) that the lowerer should resolve for alignment.
    /// The parser can't resolve typedef names, so when _Alignas takes a type argument,
    /// we store the type specifier here for proper resolution during lowering.
    pub alignas_type: Option<TypeSpecifier>,
    /// Type from `__attribute__((aligned(sizeof(type))))`.  The parser can't compute
    /// sizeof for struct/union types accurately, so we capture the type here and let
    /// sema/lowerer recompute `sizeof(type)` with full layout information.
    pub alignment_sizeof_type: Option<TypeSpecifier>,
    /// Address space qualifier on the variable itself (not on a pointer).
    /// E.g., `extern const struct pcpu_hot __seg_gs const_pcpu_hot;` has SegGs.
    /// Used for x86 per-CPU variable access with %gs:/%fs: segment prefixes.
    pub address_space: AddressSpace,
    /// GCC __attribute__((vector_size(N))): total vector size in bytes.
    /// When present on a typedef, wraps the base type in CType::Vector.
    pub vector_size: Option<usize>,
    /// Clang __attribute__((ext_vector_type(N))): number of vector elements.
    /// Resolved to total bytes in lowering via N * sizeof(element_type).
    pub ext_vector_nelem: Option<usize>,
    pub span: Span,
}

impl Declaration {
    // --- flag getters ---

    #[inline] pub fn is_static(&self) -> bool            { self.flags & decl_flag::STATIC != 0 }
    #[inline] pub fn is_extern(&self) -> bool             { self.flags & decl_flag::EXTERN != 0 }
    #[inline] pub fn is_typedef(&self) -> bool             { self.flags & decl_flag::TYPEDEF != 0 }
    #[inline] pub fn is_const(&self) -> bool              { self.flags & decl_flag::CONST != 0 }
    #[inline] pub fn is_volatile(&self) -> bool           { self.flags & decl_flag::VOLATILE != 0 }
    #[inline] pub fn is_common(&self) -> bool             { self.flags & decl_flag::COMMON != 0 }
    #[inline] pub fn is_thread_local(&self) -> bool       { self.flags & decl_flag::THREAD_LOCAL != 0 }
    #[inline] pub fn is_transparent_union(&self) -> bool  { self.flags & decl_flag::TRANSPARENT_UNION != 0 }
    #[inline] pub fn is_inline(&self) -> bool              { self.flags & decl_flag::INLINE != 0 }

    // --- flag setters ---

    #[inline] pub fn set_static(&mut self, v: bool)            { self.set_flag(decl_flag::STATIC, v) }
    #[inline] pub fn set_extern(&mut self, v: bool)             { self.set_flag(decl_flag::EXTERN, v) }
    #[inline] pub fn set_typedef(&mut self, v: bool)             { self.set_flag(decl_flag::TYPEDEF, v) }
    #[inline] pub fn set_const(&mut self, v: bool)              { self.set_flag(decl_flag::CONST, v) }
    #[inline] pub fn set_volatile(&mut self, v: bool)           { self.set_flag(decl_flag::VOLATILE, v) }
    #[inline] pub fn set_common(&mut self, v: bool)             { self.set_flag(decl_flag::COMMON, v) }
    #[inline] pub fn set_thread_local(&mut self, v: bool)       { self.set_flag(decl_flag::THREAD_LOCAL, v) }
    #[inline] pub fn set_transparent_union(&mut self, v: bool)  { self.set_flag(decl_flag::TRANSPARENT_UNION, v) }
    #[inline] pub fn set_inline(&mut self, v: bool)             { self.set_flag(decl_flag::INLINE, v) }

    #[inline]
    fn set_flag(&mut self, mask: u16, v: bool) {
        if v { self.flags |= mask; } else { self.flags &= !mask; }
    }

    /// Create a `Declaration` with all flags cleared.  Non-boolean fields must
    /// be supplied; boolean qualifiers are set afterwards via the `set_*` methods.
    pub fn new(
        type_spec: TypeSpecifier,
        declarators: Vec<InitDeclarator>,
        alignment: Option<usize>,
        alignas_type: Option<TypeSpecifier>,
        alignment_sizeof_type: Option<TypeSpecifier>,
        address_space: AddressSpace,
        vector_size: Option<usize>,
        ext_vector_nelem: Option<usize>,
        span: Span,
    ) -> Self {
        Self {
            type_spec,
            declarators,
            flags: 0,
            alignment,
            alignas_type,
            alignment_sizeof_type,
            address_space,
            vector_size,
            ext_vector_nelem,
            span,
        }
    }

    /// Create an empty declaration (used for skipped constructs like asm directives).
    pub fn empty() -> Self {
        Self::new(
            TypeSpecifier::Void,
            Vec::new(),
            None,
            None,
            None,
            AddressSpace::Default,
            None,
            None,
            Span::dummy(),
        )
    }

    /// Resolve the total vector size in bytes, considering both `vector_size` (total bytes)
    /// and `ext_vector_type` (element count). `elem_size` is the sizeof the base element type.
    /// Returns `None` if neither attribute is present.
    pub fn resolve_vector_size(&self, elem_size: usize) -> Option<usize> {
        if let Some(vs) = self.vector_size {
            Some(vs)
        } else {
            self.ext_vector_nelem.map(|n| n * elem_size)
        }
    }
}

impl std::fmt::Debug for Declaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Declaration")
            .field("type_spec", &self.type_spec)
            .field("declarators", &self.declarators)
            .field("is_static", &self.is_static())
            .field("is_extern", &self.is_extern())
            .field("is_typedef", &self.is_typedef())
            .field("is_const", &self.is_const())
            .field("is_volatile", &self.is_volatile())
            .field("is_common", &self.is_common())
            .field("is_thread_local", &self.is_thread_local())
            .field("is_transparent_union", &self.is_transparent_union())
            .field("alignment", &self.alignment)
            .field("alignas_type", &self.alignas_type)
            .field("alignment_sizeof_type", &self.alignment_sizeof_type)
            .field("address_space", &self.address_space)
            .field("vector_size", &self.vector_size)
            .field("ext_vector_nelem", &self.ext_vector_nelem)
            .field("span", &self.span)
            .finish()
    }
}

/// Attributes that can be applied to individual declarators via GCC
/// __attribute__((...)) syntax.
///
/// Boolean attributes are stored as a packed bitfield (`flags`). Non-boolean
/// attributes (`alias_target`, `visibility`, `section`, `asm_register`,
/// `cleanup_fn`) remain as `Option<String>`.
#[derive(Clone, Default)]
pub struct DeclAttributes {
    /// Packed boolean flags — see `decl_attr_flag` constants.
    flags: u16,
    /// __attribute__((alias("target"))) - this symbol is an alias for target
    pub alias_target: Option<String>,
    /// __attribute__((visibility("hidden"))) etc.
    pub visibility: Option<String>,
    /// __attribute__((section("..."))) - place in specific ELF section
    pub section: Option<String>,
    /// register var __asm__("regname") - pin to specific register for inline asm
    pub asm_register: Option<String>,
    /// __attribute__((cleanup(func))) - call func(&var) when var goes out of scope.
    /// Used for RAII-style cleanup (e.g., Linux kernel guard()/scoped_guard() for mutex_unlock).
    pub cleanup_fn: Option<Rc<str>>,
    /// __attribute__((symver("name@@VERSION"))) - symbol version alias
    pub symver: Option<String>,
}

/// Bit masks for boolean flags in `DeclAttributes::flags`.
pub mod decl_attr_flag {
    /// `__attribute__((constructor))` — run before main.
    pub const CONSTRUCTOR: u16 = 1 << 0;
    /// `__attribute__((destructor))` — run after main.
    pub const DESTRUCTOR: u16  = 1 << 1;
    /// `__attribute__((weak))` — emit as a weak symbol.
    pub const WEAK: u16        = 1 << 2;
    /// `__attribute__((error("...")))` or `__attribute__((warning("...")))`.
    pub const ERROR_ATTR: u16  = 1 << 3;
    /// `__attribute__((noreturn))` or `_Noreturn`.
    pub const NORETURN: u16    = 1 << 4;
    /// `__attribute__((used))` — prevent dead code elimination.
    pub const USED: u16        = 1 << 5;
    /// `__attribute__((fastcall))` — i386 fastcall calling convention.
    pub const FASTCALL: u16    = 1 << 6;
    /// `__attribute__((naked))` — emit no prologue/epilogue.
    pub const NAKED: u16       = 1 << 7;
}

impl DeclAttributes {
    // --- flag getters ---

    #[inline] pub fn is_constructor(&self) -> bool { self.flags & decl_attr_flag::CONSTRUCTOR != 0 }
    #[inline] pub fn is_destructor(&self) -> bool  { self.flags & decl_attr_flag::DESTRUCTOR != 0 }
    #[inline] pub fn is_weak(&self) -> bool        { self.flags & decl_attr_flag::WEAK != 0 }
    #[inline] pub fn is_error_attr(&self) -> bool  { self.flags & decl_attr_flag::ERROR_ATTR != 0 }
    #[inline] pub fn is_noreturn(&self) -> bool    { self.flags & decl_attr_flag::NORETURN != 0 }
    #[inline] pub fn is_used(&self) -> bool        { self.flags & decl_attr_flag::USED != 0 }
    #[inline] pub fn is_fastcall(&self) -> bool    { self.flags & decl_attr_flag::FASTCALL != 0 }
    #[inline] pub fn is_naked(&self) -> bool       { self.flags & decl_attr_flag::NAKED != 0 }

    // --- flag setters ---

    #[inline] pub fn set_constructor(&mut self, v: bool) { self.set_flag(decl_attr_flag::CONSTRUCTOR, v) }
    #[inline] pub fn set_destructor(&mut self, v: bool)  { self.set_flag(decl_attr_flag::DESTRUCTOR, v) }
    #[inline] pub fn set_weak(&mut self, v: bool)        { self.set_flag(decl_attr_flag::WEAK, v) }
    #[inline] pub fn set_error_attr(&mut self, v: bool)  { self.set_flag(decl_attr_flag::ERROR_ATTR, v) }
    #[inline] pub fn set_noreturn(&mut self, v: bool)    { self.set_flag(decl_attr_flag::NORETURN, v) }
    #[inline] pub fn set_used(&mut self, v: bool)        { self.set_flag(decl_attr_flag::USED, v) }
    #[inline] pub fn set_fastcall(&mut self, v: bool)    { self.set_flag(decl_attr_flag::FASTCALL, v) }
    #[inline] pub fn set_naked(&mut self, v: bool)       { self.set_flag(decl_attr_flag::NAKED, v) }

    #[inline]
    fn set_flag(&mut self, mask: u16, v: bool) {
        if v { self.flags |= mask; } else { self.flags &= !mask; }
    }
}

impl std::fmt::Debug for DeclAttributes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeclAttributes")
            .field("is_constructor", &self.is_constructor())
            .field("is_destructor", &self.is_destructor())
            .field("is_weak", &self.is_weak())
            .field("is_error_attr", &self.is_error_attr())
            .field("is_noreturn", &self.is_noreturn())
            .field("is_used", &self.is_used())
            .field("alias_target", &self.alias_target)
            .field("visibility", &self.visibility)
            .field("section", &self.section)
            .field("asm_register", &self.asm_register)
            .field("cleanup_fn", &self.cleanup_fn)
            .finish()
    }
}

/// A declarator with optional initializer.
#[derive(Debug, Clone)]
pub struct InitDeclarator {
    pub name: Rc<str>,
    pub derived: Vec<DerivedDeclarator>,
    pub init: Option<Initializer>,
    /// Declarator attributes (GCC __attribute__, asm register, etc.).
    pub attrs: DeclAttributes,
    pub span: Span,
}

/// Derived parts of a declarator (pointers, arrays, function params).
#[derive(Debug, Clone)]
pub enum DerivedDeclarator {
    Pointer,
    Array(Option<Box<Expr>>),
    Function(Vec<ParamDecl>, bool), // params, variadic
    /// Function pointer: (*name)(params) - distinguishes from pointer-to-return-type
    FunctionPointer(Vec<ParamDecl>, bool), // params, variadic
}

/// An initializer.
#[derive(Debug, Clone)]
pub enum Initializer {
    Expr(Expr),
    List(Vec<InitializerItem>),
}

/// An item in an initializer list.
#[derive(Debug, Clone)]
pub struct InitializerItem {
    pub designators: Vec<Designator>,
    pub init: Initializer,
}

/// A designator in an initializer.
#[derive(Debug, Clone)]
pub enum Designator {
    Index(Expr),
    /// GCC range designator: [lo ... hi]
    Range(Expr, Expr),
    Field(Rc<str>),
}

/// Type specifiers.
#[derive(Debug, Clone)]
pub enum TypeSpecifier {
    Void,
    Char,
    Short,
    Int,
    Long,
    LongLong,
    Float,
    Double,
    LongDouble,
    #[allow(dead_code)] // Matched in type resolution but not currently emitted by parser
    Signed,
    #[allow(dead_code)] // Matched in type resolution but not currently emitted by parser
    Unsigned,
    UnsignedChar,
    UnsignedShort,
    UnsignedInt,
    UnsignedLong,
    UnsignedLongLong,
    Int128,
    UnsignedInt128,
    Bool,
    ComplexFloat,
    ComplexDouble,
    ComplexLongDouble,
    /// Struct: (name, fields, is_packed, max_field_align from #pragma pack, struct-level aligned attribute)
    Struct(Option<Rc<str>>, Option<Vec<StructFieldDecl>>, bool, Option<usize>, Option<usize>),
    /// Union: (name, fields, is_packed, max_field_align from #pragma pack, struct-level aligned attribute)
    Union(Option<Rc<str>>, Option<Vec<StructFieldDecl>>, bool, Option<usize>, Option<usize>),
    /// Enum: (name, variants, is_packed)
    Enum(Option<Rc<str>>, Option<Vec<EnumVariant>>, bool),
    TypedefName(Rc<str>),
    Pointer(Box<TypeSpecifier>, AddressSpace),
    Array(Box<TypeSpecifier>, Option<Box<Expr>>),
    /// Function pointer type from cast/sizeof: return_type, params, variadic
    /// E.g., `(jv (*)(void*, jv))` produces FunctionPointer(jv, [void*, jv], false)
    FunctionPointer(Box<TypeSpecifier>, Vec<ParamDecl>, bool),
    /// Bare function type (not a pointer): return_type, params, variadic
    /// This is produced by ctype_to_type_spec for CType::Function, used in typeof
    /// resolution. Unlike FunctionPointer which represents Pointer(Function(...)),
    /// this represents just Function(...) without the pointer wrapper.
    BareFunction(Box<TypeSpecifier>, Vec<ParamDecl>, bool),
    /// typeof(expr) - GCC extension: type of an expression
    Typeof(Box<Expr>),
    /// typeof(type-name) - GCC extension: type from a type name
    TypeofType(Box<TypeSpecifier>),
    /// __auto_type - GCC extension: type inferred from initializer expression
    AutoType,
    /// __attribute__((vector_size(N))) applied directly in cast/compound-literal/sizeof.
    /// Wraps the base element type and total vector size in bytes.
    /// E.g., `(__attribute__((vector_size(16))) float){...}` becomes Vector(Float, 16).
    Vector(Box<TypeSpecifier>, usize),
}

/// A field declaration in a struct/union.
#[derive(Debug, Clone)]
pub struct StructFieldDecl {
    pub type_spec: TypeSpecifier,
    pub name: Option<Rc<str>>,
    pub bit_width: Option<Box<Expr>>,
    /// Derived declarator parts (pointers, arrays, function pointers) from the declarator.
    /// For simple fields like `int x` or `int *p`, this is empty (the pointer is in type_spec).
    /// For complex declarators like `void (*(*fp)(int))(void)`, this carries the full
    /// derived declarator chain that must be applied to type_spec to get the final type.
    pub derived: Vec<DerivedDeclarator>,
    /// Per-field alignment from _Alignas(N) or __attribute__((aligned(N))).
    pub alignment: Option<usize>,
    /// Per-field __attribute__((packed)) - forces this field's alignment to 1.
    pub is_packed: bool,
}

/// An enum variant.
#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: Rc<str>,
    pub value: Option<Box<Expr>>,
}

/// A compound statement (block).
#[derive(Debug, Clone)]
pub struct CompoundStmt {
    pub items: Vec<BlockItem>,
    /// GNU __label__ declarations: local label names scoped to this block.
    /// When non-empty, label definitions and gotos within this block use
    /// scope-qualified names to avoid collisions (e.g., in statement expressions).
    pub local_labels: Vec<Rc<str>>,
}

/// Items within a block.
#[derive(Debug, Clone)]
pub enum BlockItem {
    Declaration(Declaration),
    Statement(Stmt),
}

/// Statements.
#[derive(Debug, Clone)]
pub enum Stmt {
    Expr(Option<Expr>),
    Return(Option<Expr>, Span),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>, Span),
    While(Expr, Box<Stmt>, Span),
    DoWhile(Box<Stmt>, Expr, Span),
    For(Option<Box<ForInit>>, Option<Expr>, Option<Expr>, Box<Stmt>, Span),
    Compound(CompoundStmt),
    Break(Span),
    Continue(Span),
    Switch(Expr, Box<Stmt>, Span),
    Case(Expr, Box<Stmt>, Span),
    /// GNU case range: `case low ... high:` (GCC extension)
    CaseRange(Expr, Expr, Box<Stmt>, Span),
    Default(Box<Stmt>, Span),
    Goto(Rc<str>, Span),
    /// Computed goto: goto *expr (GCC extension, labels-as-values)
    GotoIndirect(Box<Expr>, Span),
    Label(Rc<str>, Box<Stmt>, Span),
    /// A declaration in statement position (C23: declarations allowed after labels,
    /// and in other statement contexts like `case`/`default`).
    Declaration(Declaration),
    InlineAsm {
        template: String,
        outputs: Vec<AsmOperand>,
        inputs: Vec<AsmOperand>,
        clobbers: Vec<String>,
        /// Goto labels for asm goto (e.g., `asm goto("..." : : : : label1, label2)`)
        goto_labels: Vec<Rc<str>>,
    },
}

impl Stmt {
    /// Extract the source span from this statement, if available.
    /// Used by IR lowering to propagate source locations for debug info.
    pub fn span(&self) -> Option<Span> {
        match self {
            Stmt::Return(_, span) | Stmt::If(_, _, _, span) | Stmt::While(_, _, span) |
            Stmt::DoWhile(_, _, span) | Stmt::For(_, _, _, _, span) |
            Stmt::Break(span) | Stmt::Continue(span) | Stmt::Switch(_, _, span) |
            Stmt::Case(_, _, span) | Stmt::CaseRange(_, _, _, span) |
            Stmt::Default(_, span) | Stmt::Goto(_, span) |
            Stmt::GotoIndirect(_, span) | Stmt::Label(_, _, span) => Some(*span),
            Stmt::Expr(Some(expr)) => Some(expr.span()),
            Stmt::Declaration(decl) => Some(decl.span),
            Stmt::Expr(None) | Stmt::Compound(_) | Stmt::InlineAsm { .. } => None,
        }
    }
}

/// An operand in an inline asm statement.
#[derive(Debug, Clone)]
pub struct AsmOperand {
    /// Symbolic name (e.g., [name]) if present
    pub name: Option<String>,
    /// Constraint string (e.g., "=r", "+r", "r")
    pub constraint: String,
    /// The C expression for the operand
    pub expr: Expr,
}

/// For loop initializer.
#[derive(Debug, Clone)]
pub enum ForInit {
    Declaration(Declaration),
    Expr(Expr),
}

/// A type-safe identity key for AST `Expr` nodes, used as a HashMap key for
/// expression type and constant value caches.
///
/// Currently derived from the heap address of the Expr node (`&Expr as *const _
/// as usize`), which is stable because the AST is allocated once during parsing
/// and not moved before lowering consumes it.  Wrapping the raw address in a
/// newtype provides:
/// - **Type safety**: prevents accidental mixing with other `usize` values.
/// - **Self-documenting API**: call sites use `expr.id()` instead of a raw cast.
/// - **Future-proofing**: the underlying representation can be changed to a
///   counter-based scheme (assigned during parsing) without touching call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(usize);

impl ExprId {
    /// Create an ExprId from a reference to an Expr node.
    #[inline]
    pub fn of(expr: &Expr) -> Self {
        ExprId(expr as *const Expr as usize)
    }
}

/// Expressions.
#[derive(Debug, Clone)]
pub enum Expr {
    IntLiteral(i64, Span),
    UIntLiteral(u64, Span),
    LongLiteral(i64, Span),
    ULongLiteral(u64, Span),
    LongLongLiteral(i64, Span),   // ll/LL suffix (signed long long, always 64-bit)
    ULongLongLiteral(u64, Span),  // ull/ULL suffix (unsigned long long, always 64-bit)
    FloatLiteral(f64, Span),            // double literal (no suffix)
    FloatLiteralF32(f64, Span),         // float literal (f/F suffix)
    /// Long double literal (l/L suffix). Stores (f64_approx, f128_bytes, span).
    /// f128_bytes is IEEE 754 binary128 with full 112-bit mantissa precision.
    FloatLiteralLongDouble(f64, [u8; 16], Span),
    /// Imaginary literal: value * I (double imaginary, e.g. 1.0i)
    ImaginaryLiteral(f64, Span),
    /// Float imaginary literal (e.g. 1.0fi)
    ImaginaryLiteralF32(f64, Span),
    /// Long double imaginary literal (e.g. 1.0Li). Stores (f64_approx, f128_bytes, span).
    ImaginaryLiteralLongDouble(f64, [u8; 16], Span),
    StringLiteral(String, Span),
    /// Wide string literal (L"...") - each char is a wchar_t (32-bit int)
    WideStringLiteral(String, Span),
    /// char16_t string literal (u"...") - each char is a char16_t (16-bit unsigned)
    Char16StringLiteral(String, Span),
    CharLiteral(char, Span),
    Identifier(Rc<str>, Span),
    BinaryOp(BinOp, Box<Expr>, Box<Expr>, Span),
    UnaryOp(UnaryOp, Box<Expr>, Span),
    PostfixOp(PostfixOp, Box<Expr>, Span),
    Assign(Box<Expr>, Box<Expr>, Span),
    CompoundAssign(BinOp, Box<Expr>, Box<Expr>, Span),
    Conditional(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    /// GNU extension: `cond ? : else_expr` - condition is evaluated once and used as the
    /// then-value if truthy. Semantically: `({ auto __tmp = cond; __tmp ? __tmp : else_expr; })`
    GnuConditional(Box<Expr>, Box<Expr>, Span),
    FunctionCall(Box<Expr>, Vec<Expr>, Span),
    ArraySubscript(Box<Expr>, Box<Expr>, Span),
    MemberAccess(Box<Expr>, Rc<str>, Span),
    PointerMemberAccess(Box<Expr>, Rc<str>, Span),
    Cast(TypeSpecifier, Box<Expr>, Span),
    CompoundLiteral(TypeSpecifier, Box<Initializer>, Span),
    StmtExpr(CompoundStmt, Span),
    Sizeof(Box<SizeofArg>, Span),
    /// __builtin_va_arg(ap, type): extract next variadic argument of given type
    VaArg(Box<Expr>, TypeSpecifier, Span),
    /// _Alignof(type) - C11 standard, returns minimum ABI alignment
    Alignof(TypeSpecifier, Span),
    /// _Alignof(expr) - alignment of expression's type (via _Alignof macro path)
    AlignofExpr(Box<Expr>, Span),
    /// __alignof(type) / __alignof__(type) - GCC extension, returns preferred alignment
    /// On i686: __alignof__(long long) == 8, _Alignof(long long) == 4
    GnuAlignof(TypeSpecifier, Span),
    /// __alignof__(expr) - GCC extension, preferred alignment of expression's type
    GnuAlignofExpr(Box<Expr>, Span),
    Comma(Box<Expr>, Box<Expr>, Span),
    AddressOf(Box<Expr>, Span),
    Deref(Box<Expr>, Span),
    /// _Generic(controlling_expr, type1: expr1, type2: expr2, ..., default: exprN)
    GenericSelection(Box<Expr>, Vec<GenericAssociation>, Span),
    /// GCC extension: &&label (address of label, for computed goto)
    LabelAddr(Rc<str>, Span),
    /// GCC extension: __builtin_types_compatible_p(type1, type2)
    /// Compile-time constant: 1 if the two types are compatible, 0 otherwise.
    BuiltinTypesCompatibleP(TypeSpecifier, TypeSpecifier, Span),
}

/// A _Generic association: either a type-expression pair, or a default expression.
#[derive(Debug, Clone)]
pub struct GenericAssociation {
    pub type_spec: Option<TypeSpecifier>, // None for "default"
    pub expr: Expr,
    /// Whether the association type has a const qualifier on the top-level type
    /// (for non-pointer types) or on the pointee (for pointer types like `const int *`).
    /// Used by _Generic matching to distinguish e.g. `const int *` from `int *`,
    /// since CType does not track const/volatile qualifiers.
    pub is_const: bool,
}

/// Sizeof argument can be a type or expression.
#[derive(Debug, Clone)]
pub enum SizeofArg {
    Type(TypeSpecifier),
    Expr(Expr),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogicalAnd,
    LogicalOr,
}

impl BinOp {
    /// Returns true for comparison operators (==, !=, <, <=, >, >=, &&, ||).
    pub fn is_comparison(self) -> bool {
        matches!(self, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le
            | BinOp::Gt | BinOp::Ge | BinOp::LogicalAnd | BinOp::LogicalOr)
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Neg,
    BitNot,
    LogicalNot,
    PreInc,
    PreDec,
    /// __real__ expr - extract real part of complex number (GCC extension)
    RealPart,
    /// __imag__ expr - extract imaginary part of complex number (GCC extension)
    ImagPart,
}

/// Postfix operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostfixOp {
    PostInc,
    PostDec,
}

impl Expr {
    /// Returns a type-safe identity key for this expression node.
    /// Used as a HashMap key for caching expression types and constant values.
    #[inline]
    pub fn id(&self) -> ExprId {
        ExprId::of(self)
    }

    pub fn span(&self) -> Span {
        match self {
            Expr::IntLiteral(_, s) | Expr::UIntLiteral(_, s)
            | Expr::LongLiteral(_, s) | Expr::ULongLiteral(_, s)
            | Expr::LongLongLiteral(_, s) | Expr::ULongLongLiteral(_, s)
            | Expr::FloatLiteral(_, s) | Expr::FloatLiteralF32(_, s) | Expr::FloatLiteralLongDouble(_, _, s)
            | Expr::ImaginaryLiteral(_, s) | Expr::ImaginaryLiteralF32(_, s) | Expr::ImaginaryLiteralLongDouble(_, _, s)
            | Expr::StringLiteral(_, s) | Expr::WideStringLiteral(_, s)
            | Expr::Char16StringLiteral(_, s)
            | Expr::CharLiteral(_, s) | Expr::Identifier(_, s)
            | Expr::BinaryOp(_, _, _, s) | Expr::UnaryOp(_, _, s) | Expr::PostfixOp(_, _, s)
            | Expr::Assign(_, _, s) | Expr::CompoundAssign(_, _, _, s) | Expr::Conditional(_, _, _, s)
            | Expr::GnuConditional(_, _, s)
            | Expr::FunctionCall(_, _, s) | Expr::ArraySubscript(_, _, s)
            | Expr::MemberAccess(_, _, s) | Expr::PointerMemberAccess(_, _, s)
            | Expr::Cast(_, _, s) | Expr::CompoundLiteral(_, _, s) | Expr::StmtExpr(_, s)
            | Expr::Sizeof(_, s) | Expr::VaArg(_, _, s) | Expr::Alignof(_, s)
            | Expr::AlignofExpr(_, s) | Expr::GnuAlignof(_, s)
            | Expr::GnuAlignofExpr(_, s) | Expr::Comma(_, _, s)
            | Expr::AddressOf(_, s) | Expr::Deref(_, s)
            | Expr::GenericSelection(_, _, s) | Expr::LabelAddr(_, s)
            | Expr::BuiltinTypesCompatibleP(_, _, s) => *s,
        }
    }
}
