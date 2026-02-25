//! Shared type-system state for semantic analysis and IR lowering.
//!
//! `TypeContext` holds all type information that persists across function boundaries:
//! struct/union layouts, typedefs, enum constants, function typedef metadata, and
//! a type cache. It is populated by the sema pass and consumed/extended by the lowerer.
//!
//! This module lives in `frontend/sema/` because it is semantically part of the
//! sema output: sema creates, populates, and owns the TypeContext until it is
//! transferred to the lowerer. The lowerer imports from here, which is the correct
//! dependency direction (IR depends on frontend output, not the reverse).
//!
//! Scope management uses an undo-log pattern (`TypeScopeFrame`) rather than cloning
//! entire HashMaps at scope boundaries. This gives O(changes-in-scope) cost for
//! scope push/pop instead of O(total-map-size).

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::rc::Rc;
use crate::common::types::{AddressSpace, StructLayout, RcLayout, CType};
use crate::frontend::parser::ast::{TypeSpecifier, ParamDecl, DerivedDeclarator};

/// Information about a function typedef (e.g., `typedef int func_t(int, int);`).
/// Used to detect when a declaration like `func_t add;` is a function declaration
/// rather than a variable declaration.
///
/// Shared between sema (which collects typedef info) and lowering (which uses it
/// to resolve function declarations through typedefs).
#[derive(Debug, Clone)]
pub struct FunctionTypedefInfo {
    /// The return TypeSpecifier of the function typedef
    pub return_type: TypeSpecifier,
    /// Parameters of the function typedef
    pub params: Vec<ParamDecl>,
    /// Whether the function is variadic
    pub variadic: bool,
}

/// Extract function pointer typedef info from a declarator with `FunctionPointer`
/// derived declarators.
///
/// For typedefs like `typedef void *(*lua_Alloc)(void *, ...)`, finds the
/// `FunctionPointer` derived and builds the return type. The last `Pointer` before
/// `FunctionPointer` is the `(*)` indirection, not a return-type pointer.
pub fn extract_fptr_typedef_info(
    base_type: &TypeSpecifier,
    derived: &[DerivedDeclarator],
) -> Option<FunctionTypedefInfo> {
    let (params, variadic) = derived.iter().find_map(|d| {
        if let DerivedDeclarator::FunctionPointer(p, v) = d { Some((p, v)) } else { None }
    })?;
    let ptr_count_before_fptr = derived.iter()
        .take_while(|d| !matches!(d, DerivedDeclarator::FunctionPointer(_, _)))
        .filter(|d| matches!(d, DerivedDeclarator::Pointer))
        .count();
    let ret_ptr_count = ptr_count_before_fptr.saturating_sub(1);
    let mut return_type = base_type.clone();
    for _ in 0..ret_ptr_count {
        return_type = TypeSpecifier::Pointer(Box::new(return_type), AddressSpace::Default);
    }
    Some(FunctionTypedefInfo {
        return_type,
        params: params.clone(),
        variadic: *variadic,
    })
}

/// Records undo operations for type-system scoped state.
///
/// Pushed on scope entry, popped on scope exit. Tracks both newly-added keys
/// (removed on undo) and shadowed keys (restored to previous value on undo)
/// across enum_constants, struct_layouts, ctype_cache, and typedefs.
#[derive(Debug)]
pub struct TypeScopeFrame {
    /// Keys newly inserted into `enum_constants`.
    pub enums_added: Vec<Rc<str>>,
    /// Keys newly inserted into `struct_layouts`.
    pub struct_layouts_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `struct_layouts`: (key, previous_value).
    /// Uses Rc<StructLayout> so saving/restoring is a cheap refcount bump.
    pub struct_layouts_shadowed: Vec<(Rc<str>, RcLayout)>,
    /// Keys newly inserted into `ctype_cache`.
    pub ctype_cache_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `ctype_cache`: (key, previous_value).
    pub ctype_cache_shadowed: Vec<(Rc<str>, CType)>,
    /// Keys newly inserted into `typedefs`.
    pub typedefs_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `typedefs`: (key, previous_value).
    pub typedefs_shadowed: Vec<(Rc<str>, CType)>,
    /// Keys newly inserted into `typedef_alignments`.
    pub typedef_alignments_added: Vec<Rc<str>>,
    /// Keys that were overwritten in `typedef_alignments`: (key, previous_value).
    pub typedef_alignments_shadowed: Vec<(Rc<str>, usize)>,
}

impl TypeScopeFrame {
    fn new() -> Self {
        Self {
            enums_added: Vec::new(),
            struct_layouts_added: Vec::new(),
            struct_layouts_shadowed: Vec::new(),
            ctype_cache_added: Vec::new(),
            ctype_cache_shadowed: Vec::new(),
            typedefs_added: Vec::new(),
            typedefs_shadowed: Vec::new(),
            typedef_alignments_added: Vec::new(),
            typedef_alignments_shadowed: Vec::new(),
        }
    }
}

/// Module-level type-system state produced by sema and consumed by lowering.
///
/// Holds struct/union layouts, typedefs, enum constants, and type caches.
/// Populated by the sema pass during semantic analysis, then transferred
/// to the lowerer by ownership for IR emission.
#[derive(Debug)]
pub struct TypeContext {
    /// Struct/union layouts indexed by tag name.
    /// Uses Rc<StructLayout> so lookups/clones are cheap refcount bumps
    /// instead of deep-copying all field names and types.
    /// Wrapped in RefCell for interior mutability: type resolution methods
    /// that take &self (via the TypeConvertContext trait) may need to insert
    /// forward-declaration layouts when encountering struct/union types.
    pub struct_layouts: RefCell<FxHashMap<Rc<str>, RcLayout>>,
    /// Enum constant values
    pub enum_constants: FxHashMap<Rc<str>, i64>,
    /// Typedef mappings (name -> resolved CType)
    pub typedefs: FxHashMap<Rc<str>, CType>,
    /// Per-typedef alignment overrides from `__attribute__((aligned(N)))` on typedef
    /// declarations.  E.g. `typedef struct S aligned_S __attribute__((aligned(32)));`
    /// stores `"aligned_S" -> 32`.  Consulted when computing field / variable alignment
    /// for declarations that use the typedef name.
    pub typedef_alignments: FxHashMap<Rc<str>, usize>,
    /// Function typedef info (bare function typedefs like `typedef int func_t(int)`)
    pub function_typedefs: FxHashMap<Rc<str>, FunctionTypedefInfo>,
    /// Set of typedef names that are function pointer types
    /// (e.g., `typedef void *(*lua_Alloc)(void *, ...)`)
    pub func_ptr_typedefs: FxHashSet<Rc<str>>,
    /// Function pointer typedef info (return type, params, variadic)
    pub func_ptr_typedef_info: FxHashMap<Rc<str>, FunctionTypedefInfo>,
    /// Set of typedef names that alias enum types.
    /// Used to treat enum-typedef bitfields as unsigned (GCC compat).
    pub enum_typedefs: FxHashSet<Rc<str>>,
    /// Packed enum type info, keyed by tag name.
    /// Stored when a packed enum definition is processed so that forward
    /// references can look up the correct size.
    pub packed_enum_types: FxHashMap<Rc<str>, crate::common::types::EnumType>,
    /// Return CType for known functions
    pub func_return_ctypes: FxHashMap<Rc<str>, CType>,
    /// Cache for CType of named struct/union types.
    /// Uses RefCell because type_spec_to_ctype takes &self.
    pub ctype_cache: RefCell<FxHashMap<Rc<str>, CType>>,
    /// Scope stack for type-system undo tracking (enum_constants, struct_layouts, ctype_cache).
    /// Wrapped in RefCell for interior mutability: scoped insertion methods
    /// called from &self contexts need to record undo entries.
    pub scope_stack: RefCell<Vec<TypeScopeFrame>>,
    /// Counter for anonymous struct/union CType keys generated from &self contexts.
    /// Uses Cell for interior mutability since type_spec_to_ctype takes &self.
    anon_ctype_counter: std::cell::Cell<u32>,
}

// We cannot directly implement `StructLayoutProvider` for `TypeContext` because
// `struct_layouts` is behind a `RefCell`, and the trait returns `Option<&StructLayout>`
// which would borrow from a temporary `Ref` guard. Instead, callers should use
// `tc.borrow_struct_layouts()` to get the guard, then pass `&*guard` as the
// `&dyn StructLayoutProvider` (since `FxHashMap<String, RcLayout>` implements the trait).

impl TypeContext {
    pub fn new() -> Self {
        let mut tc = Self {
            struct_layouts: RefCell::new(FxHashMap::default()),
            enum_constants: FxHashMap::default(),
            typedefs: FxHashMap::default(),
            typedef_alignments: FxHashMap::default(),
            function_typedefs: FxHashMap::default(),
            func_ptr_typedefs: FxHashSet::default(),
            func_ptr_typedef_info: FxHashMap::default(),
            enum_typedefs: FxHashSet::default(),
            packed_enum_types: FxHashMap::default(),
            func_return_ctypes: FxHashMap::default(),
            ctype_cache: RefCell::new(FxHashMap::default()),
            scope_stack: RefCell::new(Vec::new()),
            anon_ctype_counter: std::cell::Cell::new(0),
        };
        tc.seed_builtin_typedefs();
        tc
    }

    /// Pre-populate typedef mappings for builtin/standard C types so that
    /// sema can correctly resolve types like `uint64_t`, `size_t`, etc.
    /// even when the source code does not `#include <stdint.h>`.
    ///
    /// Without this, sema's `resolve_typedef` falls back to `CType::Int`
    /// for unresolved typedefs, causing functions returning `uint64_t`
    /// to be recorded with a 32-bit return type.
    fn seed_builtin_typedefs(&mut self) {
        use crate::common::types::target_is_32bit;
        let is_32bit = target_is_32bit();

        // On ILP32 (i686): long=32bit, so 64-bit types must use LongLong
        // On LP64 (x86-64, arm64, riscv64): long=64bit
        let i64_type = if is_32bit { CType::LongLong } else { CType::Long };
        let u64_type = if is_32bit { CType::ULongLong } else { CType::ULong };
        let size_type = if is_32bit { CType::UInt } else { CType::ULong };
        let ssize_type = if is_32bit { CType::Int } else { CType::Long };
        let ptrdiff_type = if is_32bit { CType::Int } else { CType::Long };
        let intptr_type = if is_32bit { CType::Int } else { CType::Long };
        let uintptr_type = if is_32bit { CType::UInt } else { CType::ULong };
        let fast_s = if is_32bit { CType::Int } else { CType::Long };
        let fast_u = if is_32bit { CType::UInt } else { CType::ULong };
        let long_s = CType::Long;
        let long_u = CType::ULong;

        let builtins: &[(&str, CType)] = &[
            // <stddef.h>
            ("size_t", size_type.clone()),
            ("ssize_t", ssize_type.clone()),
            ("ptrdiff_t", ptrdiff_type),
            ("wchar_t", CType::Int),
            ("wint_t", CType::UInt),
            // <stdint.h> - exact width types
            ("int8_t", CType::Char),
            ("int16_t", CType::Short),
            ("int32_t", CType::Int),
            ("int64_t", i64_type.clone()),
            ("uint8_t", CType::UChar),
            ("uint16_t", CType::UShort),
            ("uint32_t", CType::UInt),
            ("uint64_t", u64_type.clone()),
            ("intptr_t", intptr_type),
            ("uintptr_t", uintptr_type),
            ("intmax_t", i64_type.clone()),
            ("uintmax_t", u64_type.clone()),
            // least types
            ("int_least8_t", CType::Char),
            ("int_least16_t", CType::Short),
            ("int_least32_t", CType::Int),
            ("int_least64_t", i64_type.clone()),
            ("uint_least8_t", CType::UChar),
            ("uint_least16_t", CType::UShort),
            ("uint_least32_t", CType::UInt),
            ("uint_least64_t", u64_type.clone()),
            // fast types
            ("int_fast8_t", CType::Char),
            ("int_fast16_t", fast_s.clone()),
            ("int_fast32_t", fast_s.clone()),
            ("int_fast64_t", i64_type.clone()),
            ("uint_fast8_t", CType::UChar),
            ("uint_fast16_t", fast_u.clone()),
            ("uint_fast32_t", fast_u),
            ("uint_fast64_t", u64_type.clone()),
            // <signal.h>
            ("sig_atomic_t", CType::Int),
            // <time.h>
            ("time_t", long_s.clone()),
            ("clock_t", long_s.clone()),
            ("timer_t", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("clockid_t", CType::Int),
            // <sys/types.h>
            ("off_t", long_s.clone()),
            ("pid_t", CType::Int),
            ("uid_t", CType::UInt),
            ("gid_t", CType::UInt),
            ("mode_t", CType::UInt),
            ("dev_t", u64_type.clone()),
            ("ino_t", u64_type.clone()),
            ("nlink_t", u64_type.clone()),
            ("blksize_t", long_s.clone()),
            ("blkcnt_t", long_s),
            // GNU/glibc common
            ("ulong", long_u),
            ("ushort", CType::UShort),
            ("uint", CType::UInt),
            ("__u8", CType::UChar),
            ("__u16", CType::UShort),
            ("__u32", CType::UInt),
            ("__u64", u64_type.clone()),
            ("__s8", CType::Char),
            ("__s16", CType::Short),
            ("__s32", CType::Int),
            ("__s64", i64_type.clone()),
            // <locale.h>
            ("locale_t", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            // <pthread.h>
            ("pthread_t", size_type),
            ("pthread_mutex_t", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("pthread_cond_t", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("pthread_key_t", CType::UInt),
            ("pthread_attr_t", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("pthread_once_t", CType::Int),
            ("pthread_mutexattr_t", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("pthread_condattr_t", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            // <setjmp.h>
            ("jmp_buf", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("sigjmp_buf", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            // <stdio.h>
            ("FILE", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("fpos_t", ssize_type),
            // <dirent.h>
            ("DIR", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            // POSIX internal names
            ("__u_char", CType::UChar),
            ("__u_short", CType::UShort),
            ("__u_int", CType::UInt),
            ("__u_long", CType::ULong),
            ("__int8_t", CType::Char),
            ("__int16_t", CType::Short),
            ("__int32_t", CType::Int),
            ("__uint8_t", CType::UChar),
            ("__uint16_t", CType::UShort),
            ("__uint32_t", CType::UInt),
            // __int64_t/__uint64_t: must be LongLong on ILP32, Long on LP64
            ("__int64_t", i64_type),
            ("__uint64_t", u64_type),
            // <stdarg.h> - va_list and related types.
            // Sema uses a pointer approximation; the IR lowerer (types_seed.rs)
            // applies the target-specific concrete ABI type.
            ("va_list", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("__builtin_va_list", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
            ("__gnuc_va_list", CType::Pointer(Box::new(CType::Void), AddressSpace::Default)),
        ];
        for (name, ct) in builtins {
            self.typedefs.insert(Rc::from(*name), ct.clone());
        }
    }

    /// Borrow the struct layouts map immutably.
    /// Returns a `Ref` guard that derefs to `FxHashMap<String, RcLayout>`.
    /// The underlying `FxHashMap` implements `StructLayoutProvider`, so
    /// `&*guard` can be passed wherever `&dyn StructLayoutProvider` is needed.
    pub fn borrow_struct_layouts(&self) -> std::cell::Ref<'_, FxHashMap<Rc<str>, RcLayout>> {
        self.struct_layouts.borrow()
    }

    /// Borrow the struct layouts map mutably.
    /// Returns a `RefMut` guard that derefs to `FxHashMap<Rc<str>, RcLayout>`.
    pub fn borrow_struct_layouts_mut(&self) -> std::cell::RefMut<'_, FxHashMap<Rc<str>, RcLayout>> {
        self.struct_layouts.borrow_mut()
    }

    /// Get the next anonymous struct/union ID for CType key generation.
    /// Safe to call from &self contexts (uses Cell for interior mutability).
    pub fn next_anon_struct_id(&self) -> u32 {
        let id = self.anon_ctype_counter.get();
        self.anon_ctype_counter.set(id + 1);
        id
    }

    /// Insert a struct layout from a &self context (interior mutability via RefCell).
    pub fn insert_struct_layout_from_ref(&self, key: Rc<str>, layout: StructLayout) {
        self.struct_layouts.borrow_mut().insert(key, Rc::new(layout));
    }

    /// Check if a struct key is currently shadowed by an inner scope redefinition.
    /// Returns true if any scope frame has saved a previous layout for this key,
    /// meaning the current layout in the map differs from an outer scope's definition.
    pub fn is_struct_key_shadowed(&self, key: &str) -> bool {
        let stack = self.scope_stack.borrow();
        for frame in stack.iter() {
            for (k, _) in &frame.struct_layouts_shadowed {
                if &**k == key {
                    return true;
                }
            }
        }
        false
    }

    /// Push a new type-system scope frame.
    pub fn push_scope(&mut self) {
        self.scope_stack.get_mut().push(TypeScopeFrame::new());
    }

    /// Pop the top type-system scope frame and undo changes to
    /// enum_constants, struct_layouts, ctype_cache, and typedefs.
    pub fn pop_scope(&mut self) {
        if let Some(frame) = self.scope_stack.get_mut().pop() {
            for key in frame.enums_added {
                self.enum_constants.remove(&key);
            }
            let layouts = self.struct_layouts.get_mut();
            for key in frame.struct_layouts_added {
                layouts.remove(&key);
            }
            for (key, val) in frame.struct_layouts_shadowed {
                // Don't restore an empty forward-declaration layout over a full
                // definition.
                if val.fields.is_empty() {
                    if let Some(current) = layouts.get(&key) {
                        if !current.fields.is_empty() {
                            continue;
                        }
                    }
                }
                layouts.insert(key, val);
            }
            {
                let cache = self.ctype_cache.get_mut();
                for key in frame.ctype_cache_added {
                    cache.remove(&key);
                }
                for (key, val) in frame.ctype_cache_shadowed {
                    cache.insert(key, val);
                }
            }
            for key in frame.typedefs_added {
                self.typedefs.remove(&key);
            }
            for (key, val) in frame.typedefs_shadowed {
                self.typedefs.insert(key, val);
            }
            for key in frame.typedef_alignments_added {
                self.typedef_alignments.remove(&key);
            }
            for (key, val) in frame.typedef_alignments_shadowed {
                self.typedef_alignments.insert(key, val);
            }
        }
    }

    /// Insert an enum constant, tracking the change in the current scope frame.
    pub fn insert_enum_scoped(&mut self, name: Rc<str>, value: i64) {
        let track = !self.enum_constants.contains_key(&name);
        if track {
            if let Some(frame) = self.scope_stack.get_mut().last_mut() {
                frame.enums_added.push(name.clone());
            }
        }
        self.enum_constants.insert(name, value);
    }

    /// Insert a struct layout, tracking the change in the current scope frame
    /// so it can be undone on scope exit.
    pub fn insert_struct_layout_scoped(&mut self, key: Rc<str>, layout: StructLayout) {
        let layouts = self.struct_layouts.get_mut();
        if let Some(frame) = self.scope_stack.get_mut().last_mut() {
            if let Some(prev) = layouts.get(&key).cloned() {
                frame.struct_layouts_shadowed.push((key.clone(), prev));
            } else {
                frame.struct_layouts_added.push(key.clone());
            }
        }
        layouts.insert(key, Rc::new(layout));
    }

    /// Insert a typedef, tracking the change in the current scope frame
    /// so it can be undone on scope exit.
    pub fn insert_typedef_scoped(&mut self, name: Rc<str>, ctype: CType) {
        if let Some(frame) = self.scope_stack.get_mut().last_mut() {
            if let Some(prev) = self.typedefs.get(&name).cloned() {
                frame.typedefs_shadowed.push((name.clone(), prev));
            } else {
                frame.typedefs_added.push(name.clone());
            }
        }
        self.typedefs.insert(name, ctype);
    }

    /// Insert a typedef alignment, tracking the change in the current scope frame
    /// so it can be undone on scope exit.
    pub fn insert_typedef_alignment_scoped(&mut self, name: Rc<str>, align: usize) {
        if let Some(frame) = self.scope_stack.get_mut().last_mut() {
            if let Some(prev) = self.typedef_alignments.get(&name).copied() {
                frame.typedef_alignments_shadowed.push((name.clone(), prev));
            } else {
                frame.typedef_alignments_added.push(name.clone());
            }
        }
        self.typedef_alignments.insert(name, align);
    }

    /// Invalidate a ctype_cache entry, tracking the change in the current scope frame
    /// so it can be restored on scope exit.
    pub fn invalidate_ctype_cache_scoped(&mut self, key: &str) {
        let prev = self.ctype_cache.get_mut().remove(key);
        if let Some(frame) = self.scope_stack.get_mut().last_mut() {
            if let Some(prev) = prev {
                frame.ctype_cache_shadowed.push((Rc::from(key), prev));
            } else {
                frame.ctype_cache_added.push(Rc::from(key));
            }
        }
    }

    /// Insert a struct layout from a &self context (interior mutability via RefCell),
    /// tracking the change in the current scope frame so it can be undone on scope exit.
    ///
    /// Used by `type_spec_to_ctype` which takes &self but still needs to
    /// properly scope struct layout insertions within function bodies.
    pub fn insert_struct_layout_scoped_from_ref(&self, key: &str, layout: StructLayout) {
        let rc_key: Rc<str> = Rc::from(key);
        let mut layouts = self.struct_layouts.borrow_mut();
        let mut stack = self.scope_stack.borrow_mut();
        if let Some(frame) = stack.last_mut() {
            if let Some(prev) = layouts.get(key).cloned() {
                frame.struct_layouts_shadowed.push((rc_key.clone(), prev));
            } else {
                frame.struct_layouts_added.push(rc_key.clone());
            }
        }
        layouts.insert(rc_key, Rc::new(layout));
    }

    /// Invalidate a ctype_cache entry from a &self context, tracking the change
    /// in the current scope frame so it can be restored on scope exit.
    pub fn invalidate_ctype_cache_scoped_from_ref(&self, key: &str) {
        let prev = self.ctype_cache.borrow_mut().remove(key);
        let mut stack = self.scope_stack.borrow_mut();
        if let Some(frame) = stack.last_mut() {
            if let Some(prev) = prev {
                frame.ctype_cache_shadowed.push((Rc::from(key), prev));
            } else {
                frame.ctype_cache_added.push(Rc::from(key));
            }
        }
    }
}
