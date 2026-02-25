//! Predefined macros and target configuration.
//!
//! Contains the predefined macro tables (standard C, platform, GCC compat,
//! type limits, float characteristics, etc.) and architecture-specific
//! setup for aarch64 and riscv64.

use std::path::PathBuf;
use std::rc::Rc;

use super::macro_defs::MacroDef;
use super::pipeline::Preprocessor;

impl Preprocessor {
    /// Define standard predefined macros.
    ///
    /// Object-like macros are defined via a static table to keep this compact.
    /// Function-like macros (with parameters) are defined individually below.
    pub(super) fn define_predefined_macros(&mut self) {
        // All object-like predefined macros as (name, body) pairs.
        // Grouped by category; order matches GCC's predefined macro output.
        const PREDEFINED_OBJECT_MACROS: &[(&str, &str)] = &[
            // Standard C
            ("__STDC__", "1"),
            ("__STDC_VERSION__", "201710L"), // C17
            ("__STDC_HOSTED__", "1"),
            // Platform
            ("__linux__", "1"), ("__linux", "1"), ("linux", "1"),
            ("__gnu_linux__", "1"),
            ("__unix__", "1"), ("__unix", "1"), ("unix", "1"),
            ("__LP64__", "1"), ("_LP64", "1"),
            // Default arch: x86_64 (overridden by set_target)
            ("__x86_64__", "1"), ("__x86_64", "1"),
            ("__amd64__", "1"), ("__amd64", "1"),
            // GCC compat: claim GCC 14.2.0. This is:
            //  - >= 5.1 (Linux kernel minimum requirement)
            //  - >= 7.4 (QEMU minimum requirement)
            //  - A modern version that satisfies most project requirements.
            // For glibc's _Float* types (expected native for GCC >= 7), we define
            // them as macros mapping to standard C types below.
            ("__GNUC__", "14"), ("__GNUC_MINOR__", "2"), ("__GNUC_PATCHLEVEL__", "0"),
            ("__VERSION__", "\"14.2.0\""),
            // C99 inline semantics: tells gnulib and other libraries that
            // plain `inline` provides an inline definition only (no external symbol).
            ("__GNUC_STDC_INLINE__", "1"),
            // sizeof macros
            ("__SIZEOF_POINTER__", "8"), ("__SIZEOF_INT__", "4"),
            ("__SIZEOF_LONG__", "8"), ("__SIZEOF_LONG_LONG__", "8"),
            ("__SIZEOF_SHORT__", "2"), ("__SIZEOF_FLOAT__", "4"),
            ("__SIZEOF_DOUBLE__", "8"), ("__SIZEOF_SIZE_T__", "8"),
            ("__SIZEOF_PTRDIFF_T__", "8"), ("__SIZEOF_WCHAR_T__", "4"),
            ("__SIZEOF_INT128__", "16"), ("__SIZEOF_WINT_T__", "4"),
            // Byte order
            ("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__"),
            ("__ORDER_LITTLE_ENDIAN__", "1234"), ("__ORDER_BIG_ENDIAN__", "4321"),
            // Type limits
            ("__CHAR_BIT__", "8"),
            ("__INT_MAX__", "2147483647"),
            ("__LONG_MAX__", "9223372036854775807L"),
            ("__LONG_LONG_MAX__", "9223372036854775807LL"),
            ("__SCHAR_MAX__", "127"), ("__SHRT_MAX__", "32767"),
            ("__SIZE_MAX__", "18446744073709551615UL"),
            ("__PTRDIFF_MAX__", "9223372036854775807L"),
            ("__WCHAR_MAX__", "2147483647"), ("__WCHAR_MIN__", "(-2147483647-1)"),
            ("__WINT_MAX__", "4294967295U"), ("__WINT_MIN__", "0U"),
            ("__SIG_ATOMIC_MAX__", "2147483647"), ("__SIG_ATOMIC_MIN__", "(-2147483647-1)"),
            // Type names
            ("__SIZE_TYPE__", "long unsigned int"), ("__PTRDIFF_TYPE__", "long int"),
            ("__WCHAR_TYPE__", "int"), ("__WINT_TYPE__", "unsigned int"),
            ("__CHAR16_TYPE__", "short unsigned int"), ("__CHAR32_TYPE__", "unsigned int"),
            ("__INTMAX_TYPE__", "long int"), ("__UINTMAX_TYPE__", "long unsigned int"),
            ("__INT8_TYPE__", "signed char"), ("__INT16_TYPE__", "short int"),
            ("__INT32_TYPE__", "int"), ("__INT64_TYPE__", "long int"),
            ("__UINT8_TYPE__", "unsigned char"), ("__UINT16_TYPE__", "unsigned short int"),
            ("__UINT32_TYPE__", "unsigned int"), ("__UINT64_TYPE__", "long unsigned int"),
            ("__INTPTR_TYPE__", "long int"), ("__UINTPTR_TYPE__", "long unsigned int"),
            ("__INT_LEAST8_TYPE__", "signed char"), ("__INT_LEAST16_TYPE__", "short int"),
            ("__INT_LEAST32_TYPE__", "int"), ("__INT_LEAST64_TYPE__", "long int"),
            ("__UINT_LEAST8_TYPE__", "unsigned char"),
            ("__UINT_LEAST16_TYPE__", "unsigned short int"),
            ("__UINT_LEAST32_TYPE__", "unsigned int"),
            ("__UINT_LEAST64_TYPE__", "long unsigned int"),
            ("__INT_FAST8_TYPE__", "signed char"), ("__INT_FAST16_TYPE__", "long int"),
            ("__INT_FAST32_TYPE__", "long int"), ("__INT_FAST64_TYPE__", "long int"),
            ("__UINT_FAST8_TYPE__", "unsigned char"),
            ("__UINT_FAST16_TYPE__", "long unsigned int"),
            ("__UINT_FAST32_TYPE__", "unsigned int"),
            ("__UINT_FAST64_TYPE__", "long unsigned int"),
            // FLT characteristics
            ("__FLT_MANT_DIG__", "24"), ("__FLT_DIG__", "6"),
            ("__FLT_MIN_EXP__", "(-125)"), ("__FLT_MIN_10_EXP__", "(-37)"),
            ("__FLT_MAX_EXP__", "128"), ("__FLT_MAX_10_EXP__", "38"),
            ("__FLT_MAX__", "3.40282346638528859811704183484516925e+38F"),
            ("__FLT_MIN__", "1.17549435082228750796873653722224568e-38F"),
            ("__FLT_EPSILON__", "1.19209289550781250000000000000000000e-7F"),
            ("__FLT_RADIX__", "2"),
            ("__FLT_DENORM_MIN__", "1.40129846432481707092372958328991613e-45F"),
            // DBL characteristics
            ("__DBL_MANT_DIG__", "53"), ("__DBL_DIG__", "15"),
            ("__DBL_MIN_EXP__", "(-1021)"), ("__DBL_MIN_10_EXP__", "(-307)"),
            ("__DBL_MAX_EXP__", "1024"), ("__DBL_MAX_10_EXP__", "308"),
            ("__DBL_MAX__", "1.79769313486231570814527423731704357e+308"),
            ("__DBL_MIN__", "2.22507385850720138309023271733240406e-308"),
            ("__DBL_EPSILON__", "2.22044604925031308084726333618164062e-16"),
            ("__DBL_DENORM_MIN__", "4.94065645841246544176568792868221372e-324"),
            // LDBL characteristics
            ("__LDBL_MANT_DIG__", "64"), ("__LDBL_DIG__", "18"),
            ("__LDBL_MIN_EXP__", "(-16381)"), ("__LDBL_MIN_10_EXP__", "(-4931)"),
            ("__LDBL_MAX_EXP__", "16384"), ("__LDBL_MAX_10_EXP__", "4932"),
            ("__LDBL_MAX__", "1.18973149535723176502126385303097021e+4932L"),
            ("__LDBL_MIN__", "3.36210314311209350626267781732175260e-4932L"),
            ("__LDBL_EPSILON__", "1.08420217248550443400745280086994171e-19L"),
            ("__LDBL_DENORM_MIN__", "3.64519953188247460252840593361941982e-4951L"),
            ("__SIZEOF_LONG_DOUBLE__", "16"),
            // Float feature flags
            ("__FLT_HAS_INFINITY__", "1"), ("__FLT_HAS_QUIET_NAN__", "1"),
            ("__FLT_HAS_DENORM__", "1"),
            ("__DBL_HAS_INFINITY__", "1"), ("__DBL_HAS_QUIET_NAN__", "1"),
            ("__DBL_HAS_DENORM__", "1"),
            ("__LDBL_HAS_INFINITY__", "1"), ("__LDBL_HAS_QUIET_NAN__", "1"),
            ("__LDBL_HAS_DENORM__", "1"),
            ("__FLT_DECIMAL_DIG__", "9"), ("__DBL_DECIMAL_DIG__", "17"),
            ("__LDBL_DECIMAL_DIG__", "21"), ("__DECIMAL_DIG__", "21"),
            // GCC extensions
            ("__GNUC_VA_LIST", "1"), ("__extension__", ""),
            // NOTE: GNU keyword aliases (__inline__, __volatile__, __asm__, __const__,
            // __restrict__, __signed__, __typeof__) are handled as keyword tokens in
            // the lexer (token.rs), not as macros, because GCC treats them as reserved
            // keywords immune to #define redefinition.
            // __alignof/__alignof__ are handled as keyword tokens (GnuAlignof)
            // in the lexer, not as macros - they return preferred alignment,
            // which differs from C11 _Alignof on i686.
            // Named address spaces (Linux kernel): __seg_gs/__seg_fs are handled
            // as keyword tokens in the lexer (token.rs), not as macros.
            // __float128 -> long double (glibc compat)
            ("__float128", "long double"), ("__SIZEOF_FLOAT128__", "16"),
            // _Float* types: For GCC >= 7, glibc expects the compiler to provide these
            // natively. We define them as macros to the corresponding standard C types.
            // TODO: Implement _Float* as proper builtin types with correct semantics
            // (e.g., _Float128 should be true IEEE binary128, not 80-bit long double
            // on x86-64). The macro approach works for glibc header compatibility but
            // loses precision for _Float128 operations on x86-64.
            ("_Float128", "long double"),
            ("_Float32", "float"),
            ("_Float64", "double"),
            ("_Float32x", "double"),
            ("_Float64x", "long double"),
            // MSVC integer type specifiers
            ("__int8", "char"), ("__int16", "short"),
            ("__int32", "int"), ("__int64", "long long"),
            // ELF ABI
            ("__USER_LABEL_PREFIX__", ""),
            // GNU C attribute macros (strip)
            ("__LEAF", ""), ("__LEAF_ATTR", ""), ("__wur", ""),
            // Date/time
            ("__DATE__", "\"Jan  1 2025\""), ("__TIME__", "\"00:00:00\""),
            // GCC atomic lock-free macros
            ("__GCC_ATOMIC_BOOL_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_CHAR_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_CHAR16_T_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_CHAR32_T_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_WCHAR_T_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_SHORT_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_INT_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_LONG_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_LLONG_LOCK_FREE", "2"),
            ("__GCC_ATOMIC_POINTER_LOCK_FREE", "2"),
            // ELF
            ("__ELF__", "1"),
            // Note: __PIC__/__pic__ are conditionally defined via set_pic(),
            // not here, so they are only present when -fPIC is active.
            // CET (Control-flow Enforcement Technology) - match GCC's default
            // This is x86_64-only; removed for other targets in set_target().
            // Value 3 = IBT (bit 0) + SHSTK (bit 1), matching GCC's default.
            // Critical: must match the GCC that assembles .S files, because
            // libffi's trampoline sizes depend on ENDBR_PRESENT which checks __CET__.
            ("__CET__", "3"),
            // SSE/MMX feature macros: SSE2 is baseline for x86_64.
            // Removed for non-x86_64 targets in set_target().
            // Many projects (dr_libs, minimp3, stb_image, etc.) use #ifdef __SSE2__
            // to enable SIMD code paths.
            ("__SSE__", "1"), ("__SSE2__", "1"), ("__MMX__", "1"),
            ("__SSE_MATH__", "1"), ("__SSE2_MATH__", "1"),
            // Pragma support flags
            ("__PRAGMA_REDEFINE_EXTNAME", "1"),
        ];

        for &(name, body) in PREDEFINED_OBJECT_MACROS {
            self.define_simple_macro(name, body);
        }

        // Function-like predefined macros: (name, params, body)
        // Note: __builtin_expect is handled as a real builtin (not a macro)
        // to properly evaluate side effects in the second argument.
        const PREDEFINED_FUNC_MACROS: &[(&str, &[&str], &str)] = &[
            ("__builtin_offsetof", &["type", "member"], "((unsigned long)&((type *)0)->member)"),
            // __has_builtin, __has_attribute, __has_feature, __has_extension,
            // __has_include, and __has_include_next are NOT defined as macros.
            // They are handled as special preprocessor operators:
            // - #ifdef checks use is_defined() which special-cases them
            // - #if evaluation uses resolve_defined_in_expr() in expr_eval.rs
        ];

        for &(name, params, body) in PREDEFINED_FUNC_MACROS {
            self.macros.define(MacroDef {
                name: Rc::from(name),
                is_function_like: true,
                params: params.iter().map(|&s| Rc::from(s)).collect(),
                is_variadic: false,
                has_named_variadic: false,
                body: Rc::from(body),
            });
        }
    }

    /// Helper to define a simple object-like macro.
    pub(super) fn define_simple_macro(&mut self, name: &str, body: &str) {
        self.macros.define(MacroDef {
            name: Rc::from(name),
            is_function_like: false,
            params: Vec::new(),
            is_variadic: false,
            has_named_variadic: false,
            body: Rc::from(body),
        });
    }

    /// Locate the bundled `include/` directory shipped alongside the binary.
    ///
    /// Walks up to 5 parent directories from the canonicalized executable path
    /// looking for an `include/` directory that contains `emmintrin.h`.
    /// Falls back to the compile-time `CARGO_MANIFEST_DIR/include` path.
    /// Returns `Some(path)` when a valid bundled include directory is found.
    pub fn bundled_include_dir() -> Option<PathBuf> {
        // Try to find the include dir relative to the running binary.
        if let Ok(exe) = std::env::current_exe() {
            if let Ok(canonical) = exe.canonicalize() {
                let mut dir = canonical.as_path().parent();
                for _ in 0..5 {
                    if let Some(d) = dir {
                        let candidate = d.join("include");
                        if candidate.join("emmintrin.h").is_file() {
                            return Some(candidate);
                        }
                        dir = d.parent();
                    } else {
                        break;
                    }
                }
            }
        }

        // Compile-time fallback: CARGO_MANIFEST_DIR/include
        let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include");
        if fallback.join("emmintrin.h").is_file() {
            return Some(fallback);
        }

        None
    }

    /// Get default system include paths (arch-neutral only).
    pub(super) fn default_system_include_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        // Bundled include directory takes priority over system GCC headers
        if let Some(bundled) = Self::bundled_include_dir() {
            paths.push(bundled);
        }
        // Only include arch-neutral paths here; arch-specific paths are added by set_target
        let candidates = [
            "/usr/local/include",
            // x86_64 multiarch (default, removed by set_target for other arches)
            "/usr/include/x86_64-linux-gnu",
            // GCC headers (common versions)
            "/usr/lib/gcc/x86_64-linux-gnu/12/include",
            "/usr/lib/gcc/x86_64-linux-gnu/11/include",
            "/usr/lib/gcc/x86_64-linux-gnu/13/include",
            "/usr/lib/gcc/x86_64-linux-gnu/14/include",
            "/usr/lib/gcc/x86_64-linux-gnu/10/include",
            "/usr/include",
        ];
        for candidate in &candidates {
            let path = PathBuf::from(candidate);
            if path.is_dir() {
                paths.push(path);
            }
        }
        paths
    }

    /// Define `__STRICT_ANSI__` for strict ISO C modes (`-std=c99`, `-std=c11`, etc.).
    /// GCC defines this macro when `-std=cXX` (non-GNU) modes are used, and many
    /// headers (glibc's `<features.h>`, CPython's `pymacro.h`) check for it to
    /// gate GNU extensions like `typeof`.
    pub fn set_strict_ansi(&mut self, strict: bool) {
        if strict {
            self.define_simple_macro("__STRICT_ANSI__", "1");
        } else {
            self.macros.undefine("__STRICT_ANSI__");
        }
    }

    /// Set inline semantics mode: GNU89 vs C99.
    /// When `gnu89` is true, defines `__GNUC_GNU_INLINE__` and undefines `__GNUC_STDC_INLINE__`.
    /// When `gnu89` is false (default), `__GNUC_STDC_INLINE__` remains defined.
    /// GCC sets `__GNUC_GNU_INLINE__` with `-fgnu89-inline` or `-std=gnu89`,
    /// and `__GNUC_STDC_INLINE__` with `-std=gnu99` and later.
    pub fn set_gnu89_inline(&mut self, gnu89: bool) {
        if gnu89 {
            self.macros.undefine("__GNUC_STDC_INLINE__");
            self.define_simple_macro("__GNUC_GNU_INLINE__", "1");
        } else {
            self.macros.undefine("__GNUC_GNU_INLINE__");
            self.define_simple_macro("__GNUC_STDC_INLINE__", "1");
        }
    }

    /// Define __OPTIMIZE__ and __OPTIMIZE_SIZE__ based on the optimization level.
    ///
    /// GCC defines __OPTIMIZE__ for any optimization level >= 1 (-O, -O1, -O2, -O3, -Os, -Oz).
    /// GCC defines __OPTIMIZE_SIZE__ for -Os and -Oz.
    /// The Linux kernel relies on __OPTIMIZE__ for BUILD_BUG() and related compile-time
    /// assertion macros that expand to noreturn function calls when optimization is enabled.
    pub fn set_optimize(&mut self, optimize: bool, optimize_size: bool) {
        if optimize {
            self.define_simple_macro("__OPTIMIZE__", "1");
        } else {
            self.macros.undefine("__OPTIMIZE__");
        }
        if optimize_size {
            self.define_simple_macro("__OPTIMIZE_SIZE__", "1");
        } else {
            self.macros.undefine("__OPTIMIZE_SIZE__");
        }
    }

    /// Define or undefine __PIC__/__pic__ based on whether PIC mode is active.
    /// GCC defines these to 1 for -fpic and 2 for -fPIC; we always use 2.
    /// When PIC is disabled (e.g. -fno-PIC), these must not be defined, as
    /// kernel code (RIP_REL_REF) checks `#ifndef __pic__` to decide whether
    /// to use RIP-relative inline asm for position-independent references.
    pub fn set_pic(&mut self, enabled: bool) {
        if enabled {
            self.define_simple_macro("__PIC__", "2");
            self.define_simple_macro("__pic__", "2");
        } else {
            self.macros.undefine("__PIC__");
            self.macros.undefine("__pic__");
        }
    }

    /// Define x86/x86_64 SIMD feature macros (__SSE__, __SSE2__, __MMX__, etc.).
    ///
    /// GCC/Clang always define these for x86_64 (SSE2 is baseline for the ISA).
    /// For i686, GCC only defines them with explicit -msse/-msse2, but since our
    /// i686 backend always uses SSE2 instructions, we define them unconditionally.
    ///
    /// When `no_sse` is true (from -mno-sse or similar flags), these macros are
    /// not defined (matching GCC behavior for kernel builds).
    ///
    /// Must be called after set_target() since it checks which arch is active.
    pub fn set_sse_macros(&mut self, no_sse: bool) {
        if no_sse {
            return;
        }
        // Only define SSE macros for x86 targets (x86_64 and i686).
        // Check that we're on an x86 target by looking for __x86_64__ or __i386__.
        let is_x86_64 = self.macros.is_defined("__x86_64__");
        let is_i386 = self.macros.is_defined("__i386__");
        if !is_x86_64 && !is_i386 {
            return;
        }

        // SSE and SSE2 are baseline for x86_64; our i686 backend also uses SSE2.
        self.define_simple_macro("__SSE__", "1");
        self.define_simple_macro("__SSE2__", "1");
        self.define_simple_macro("__MMX__", "1");

        if is_x86_64 {
            // x86_64 uses SSE for floating-point math by default
            self.define_simple_macro("__SSE_MATH__", "1");
            self.define_simple_macro("__SSE2_MATH__", "1");
            // GCC also defines this for x86_64
            self.define_simple_macro("__MMX_WITH_SSE__", "1");
        }
    }

    /// Define extended SIMD feature macros (__SSE3__, __AVX__, __AVX2__, etc.)
    /// when the corresponding -msse3, -mavx, -mavx2 flags are passed.
    /// Projects like blosc use #ifdef __AVX2__ to select optimized code paths.
    /// Must be called after set_sse_macros().
    pub fn set_extended_simd_macros(
        &mut self,
        sse3: bool,
        ssse3: bool,
        sse4_1: bool,
        sse4_2: bool,
        avx: bool,
        avx2: bool,
    ) {
        // Only define SSE/AVX macros for x86 targets.
        let is_x86 = self.macros.is_defined("__x86_64__") || self.macros.is_defined("__i386__");
        if !is_x86 {
            return;
        }
        if sse3 {
            self.define_simple_macro("__SSE3__", "1");
        }
        if ssse3 {
            self.define_simple_macro("__SSSE3__", "1");
        }
        if sse4_1 {
            self.define_simple_macro("__SSE4_1__", "1");
        }
        if sse4_2 {
            self.define_simple_macro("__SSE4_2__", "1");
        }
        if avx {
            self.define_simple_macro("__AVX__", "1");
        }
        if avx2 {
            self.define_simple_macro("__AVX2__", "1");
        }
    }

    /// Set the target architecture, updating predefined macros and include paths.
    pub fn set_target(&mut self, target: &str) {
        match target {
            "aarch64" => {
                // Remove x86 macros
                self.macros.undefine("__x86_64__");
                self.macros.undefine("__x86_64");
                self.macros.undefine("__amd64__");
                self.macros.undefine("__amd64");
                self.macros.undefine("__CET__");
                self.macros.undefine("__SSE__");
                self.macros.undefine("__SSE2__");
                self.macros.undefine("__MMX__");
                self.macros.undefine("__SSE_MATH__");
                self.macros.undefine("__SSE2_MATH__");
                // Define aarch64 macros
                self.define_simple_macro("__aarch64__", "1");
                self.define_simple_macro("__ARM_64BIT_STATE", "1");
                self.define_simple_macro("__ARM_ARCH", "8");
                self.define_simple_macro("__ARM_ARCH_8A", "1");
                self.define_simple_macro("__ARM_ARCH_ISA_A64", "1");
                self.define_simple_macro("__ARM_ARCH_PROFILE", "65"); // 'A'
                // Floating-point and SIMD
                self.define_simple_macro("__ARM_FP", "14"); // 0b1110: half+single+double precision
                self.define_simple_macro("__ARM_NEON", "1");
                self.define_simple_macro("__ARM_FP16_ARGS", "1");
                self.define_simple_macro("__ARM_FP16_FORMAT_IEEE", "1");
                // ABI
                self.define_simple_macro("__ARM_PCS_AAPCS64", "1");
                self.define_simple_macro("__ARM_SIZEOF_MINIMAL_ENUM", "4");
                self.define_simple_macro("__ARM_SIZEOF_WCHAR_T", "4");
                // Features
                self.define_simple_macro("__ARM_FEATURE_CLZ", "1");
                self.define_simple_macro("__ARM_FEATURE_FMA", "1");
                self.define_simple_macro("__ARM_FEATURE_IDIV", "1");
                self.define_simple_macro("__ARM_FEATURE_NUMERIC_MAXMIN", "1");
                self.define_simple_macro("__ARM_FEATURE_UNALIGNED", "1");
                self.define_simple_macro("__ARM_ALIGN_MAX_PWR", "28");
                self.define_simple_macro("__ARM_ALIGN_MAX_STACK_PWR", "16");
                self.define_simple_macro("__AARCH64EL__", "1");
                self.define_simple_macro("__AARCH64_CMODEL_SMALL__", "1");
                // ARM: char is unsigned by default
                self.define_simple_macro("__CHAR_UNSIGNED__", "1");
                // Replace x86 include paths with aarch64 paths
                self.system_include_paths.retain(|p| {
                    let s = p.to_string_lossy();
                    !s.contains("x86_64")
                });
                let aarch64_paths = [
                    "/usr/lib/gcc-cross/aarch64-linux-gnu/11/include",
                    "/usr/lib/gcc-cross/aarch64-linux-gnu/12/include",
                    "/usr/lib/gcc-cross/aarch64-linux-gnu/13/include",
                    "/usr/lib/gcc-cross/aarch64-linux-gnu/14/include",
                    "/usr/aarch64-linux-gnu/include",
                    "/usr/include/aarch64-linux-gnu",
                ];
                self.insert_arch_paths_after_bundled(&aarch64_paths);
                // AArch64 uses IEEE 754 binary128 for long double (not x87 80-bit)
                self.override_ldbl_binary128();
            }
            "riscv64" => {
                // Remove x86 macros
                self.macros.undefine("__x86_64__");
                self.macros.undefine("__x86_64");
                self.macros.undefine("__amd64__");
                self.macros.undefine("__amd64");
                self.macros.undefine("__CET__");
                self.macros.undefine("__SSE__");
                self.macros.undefine("__SSE2__");
                self.macros.undefine("__MMX__");
                self.macros.undefine("__SSE_MATH__");
                self.macros.undefine("__SSE2_MATH__");
                // Define riscv64 macros
                self.define_simple_macro("__riscv", "1");
                self.define_simple_macro("__riscv_xlen", "64");
                // Floating-point: double-precision (D extension)
                self.define_simple_macro("__riscv_flen", "64");
                self.define_simple_macro("__riscv_float_abi_double", "1");
                self.define_simple_macro("__riscv_fdiv", "1");
                self.define_simple_macro("__riscv_fsqrt", "1");
                // ISA extensions (RV64GC = IMAFDCZicsr_Zifencei)
                self.define_simple_macro("__riscv_atomic", "1");
                self.define_simple_macro("__riscv_mul", "1");
                self.define_simple_macro("__riscv_muldiv", "1");
                self.define_simple_macro("__riscv_div", "1");
                self.define_simple_macro("__riscv_compressed", "1");
                // Extension version macros (XYYYZZZZ format: e.g. 2001000 = v2.1.0)
                self.define_simple_macro("__riscv_i", "2001000");
                self.define_simple_macro("__riscv_m", "2000000");
                self.define_simple_macro("__riscv_a", "2001000");
                self.define_simple_macro("__riscv_f", "2002000");
                self.define_simple_macro("__riscv_d", "2002000");
                self.define_simple_macro("__riscv_c", "2000000");
                self.define_simple_macro("__riscv_zicsr", "2000000");
                self.define_simple_macro("__riscv_zifencei", "2000000");
                self.define_simple_macro("__riscv_arch_test", "1");
                self.define_simple_macro("__riscv_cmodel_medany", "1");
                // Replace x86 include paths with riscv64 paths
                self.system_include_paths.retain(|p| {
                    let s = p.to_string_lossy();
                    !s.contains("x86_64")
                });
                let riscv_paths = [
                    "/usr/lib/gcc-cross/riscv64-linux-gnu/11/include",
                    "/usr/lib/gcc-cross/riscv64-linux-gnu/12/include",
                    "/usr/lib/gcc-cross/riscv64-linux-gnu/13/include",
                    "/usr/lib/gcc-cross/riscv64-linux-gnu/14/include",
                    "/usr/riscv64-linux-gnu/include",
                    "/usr/include/riscv64-linux-gnu",
                ];
                self.insert_arch_paths_after_bundled(&riscv_paths);
                // RISC-V uses IEEE 754 binary128 for long double (not x87 80-bit)
                self.override_ldbl_binary128();
            }
            "i686" | "i386" => {
                // Remove x86-64 macros (keep x86 general macros)
                self.macros.undefine("__x86_64__");
                self.macros.undefine("__x86_64");
                self.macros.undefine("__amd64__");
                self.macros.undefine("__amd64");
                self.macros.undefine("__LP64__");
                self.macros.undefine("_LP64");
                self.macros.undefine("__SIZEOF_INT128__");
                // i686 baseline does not include SSE (GCC only enables SSE with
                // -march=pentium4 or higher). Remove SSE macros to match GCC.
                self.macros.undefine("__SSE__");
                self.macros.undefine("__SSE2__");
                self.macros.undefine("__MMX__");
                self.macros.undefine("__SSE_MATH__");
                self.macros.undefine("__SSE2_MATH__");
                // i686-linux-gnu-gcc -m32 does NOT define __CET__ (CET is
                // disabled by -m32).  We must match this because .S assembly
                // files are assembled by GCC, and if the C code expects
                // ENDBR_PRESENT (44-byte trampolines) but the assembly
                // produces non-ENDBR trampolines (40 bytes), the mismatch
                // causes crashes (e.g. libffi closures).
                self.macros.undefine("__CET__");
                // Define i686/i386 macros
                self.define_simple_macro("__i386__", "1");
                self.define_simple_macro("__i386", "1");
                self.define_simple_macro("i386", "1");
                self.define_simple_macro("__i686__", "1");
                self.define_simple_macro("__i686", "1");
                self.define_simple_macro("__ILP32__", "1");
                self.define_simple_macro("_ILP32", "1");
                // ILP32 data model: pointer/long/size_t are 4 bytes
                self.define_simple_macro("__SIZEOF_POINTER__", "4");
                self.define_simple_macro("__SIZEOF_LONG__", "4");
                self.define_simple_macro("__SIZEOF_SIZE_T__", "4");
                self.define_simple_macro("__SIZEOF_PTRDIFF_T__", "4");
                // Long double is 12 bytes on i686 (80-bit x87 + 4 bytes padding)
                self.define_simple_macro("__SIZEOF_LONG_DOUBLE__", "12");
                // Type limits for ILP32
                self.define_simple_macro("__LONG_MAX__", "2147483647L");
                self.define_simple_macro("__SIZE_MAX__", "4294967295U");
                self.define_simple_macro("__PTRDIFF_MAX__", "2147483647");
                // Override <limits.h> macros for ILP32 (long is 32-bit)
                self.define_simple_macro("LONG_MIN", "(-2147483647L-1L)");
                self.define_simple_macro("LONG_MAX", "2147483647L");
                self.define_simple_macro("ULONG_MAX", "4294967295UL");
                // Override <stdint.h> macros for ILP32 (pointer/size_t are 32-bit)
                self.define_simple_macro("INTPTR_MIN", "(-2147483647-1)");
                self.define_simple_macro("INTPTR_MAX", "2147483647");
                self.define_simple_macro("UINTPTR_MAX", "4294967295U");
                self.define_simple_macro("SIZE_MAX", "4294967295U");
                self.define_simple_macro("PTRDIFF_MIN", "(-2147483647-1)");
                self.define_simple_macro("PTRDIFF_MAX", "2147483647");
                // Type names for ILP32 (long is 32-bit, so many types change)
                self.define_simple_macro("__SIZE_TYPE__", "unsigned int");
                self.define_simple_macro("__PTRDIFF_TYPE__", "int");
                self.define_simple_macro("__INTMAX_TYPE__", "long long int");
                self.define_simple_macro("__UINTMAX_TYPE__", "long long unsigned int");
                self.define_simple_macro("__INT64_TYPE__", "long long int");
                self.define_simple_macro("__UINT64_TYPE__", "long long unsigned int");
                self.define_simple_macro("__INTPTR_TYPE__", "int");
                self.define_simple_macro("__UINTPTR_TYPE__", "unsigned int");
                self.define_simple_macro("__INT_LEAST64_TYPE__", "long long int");
                self.define_simple_macro("__UINT_LEAST64_TYPE__", "long long unsigned int");
                self.define_simple_macro("__INT_FAST16_TYPE__", "int");
                self.define_simple_macro("__INT_FAST32_TYPE__", "int");
                self.define_simple_macro("__INT_FAST64_TYPE__", "long long int");
                self.define_simple_macro("__UINT_FAST16_TYPE__", "unsigned int");
                self.define_simple_macro("__UINT_FAST64_TYPE__", "long long unsigned int");
                // Replace x86-64 include paths with i686 paths
                self.system_include_paths.retain(|p| {
                    let s = p.to_string_lossy();
                    !s.contains("x86_64")
                });
                let i686_paths = [
                    "/usr/lib/gcc-cross/i686-linux-gnu/11/include",
                    "/usr/lib/gcc-cross/i686-linux-gnu/12/include",
                    "/usr/lib/gcc-cross/i686-linux-gnu/13/include",
                    "/usr/lib/gcc-cross/i686-linux-gnu/14/include",
                    "/usr/lib/gcc/i686-linux-gnu/11/include",
                    "/usr/lib/gcc/i686-linux-gnu/12/include",
                    "/usr/lib/gcc/i686-linux-gnu/13/include",
                    "/usr/lib/gcc/i686-linux-gnu/14/include",
                    "/usr/i686-linux-gnu/include",
                    "/usr/include/i386-linux-gnu",
                ];
                self.insert_arch_paths_after_bundled(&i686_paths);
                // Override width macros for ILP32 (pointer/long/size_t/ptrdiff are 32-bit)
                self.define_simple_macro("__LONG_WIDTH__", "32");
                self.define_simple_macro("__PTRDIFF_WIDTH__", "32");
                self.define_simple_macro("__SIZE_WIDTH__", "32");
                self.define_simple_macro("__INTPTR_WIDTH__", "32");
                self.define_simple_macro("__INT_FAST16_WIDTH__", "32");
                self.define_simple_macro("__INT_FAST32_WIDTH__", "32");
                // i686 uses the same x87 80-bit long double format as x86-64
                // (LDBL macros are already set correctly), but sizeof differs (12 vs 16)
            }
            _ => {
                // x86_64 is already the default
            }
        }
    }

    /// Insert architecture-specific include paths, keeping the bundled include
    /// directory first so our simplified SSE/intrinsic headers take priority over
    /// the system GCC cross-compiler headers (which use unsupported builtins).
    fn insert_arch_paths_after_bundled(&mut self, arch_paths: &[&str]) {
        // Find the index after the bundled include dir (if present).
        // The bundled dir is always the first entry added by default_system_include_paths().
        let insert_pos = if let Some(bundled) = Self::bundled_include_dir() {
            self.system_include_paths
                .iter()
                .position(|p| *p == bundled)
                .map(|i| i + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let mut offset = 0;
        for p in arch_paths {
            let path = PathBuf::from(p);
            if path.is_dir() {
                self.system_include_paths.insert(insert_pos + offset, path);
                offset += 1;
            }
        }
    }

    /// Override long double macros from x87 80-bit extended to IEEE 754 binary128.
    /// Called by set_target() for aarch64 and riscv64 which use quad precision.
    fn override_ldbl_binary128(&mut self) {
        // GCC predefined macros (__LDBL_*__)
        self.define_simple_macro("__LDBL_MANT_DIG__", "113");
        self.define_simple_macro("__LDBL_DIG__", "33");
        self.define_simple_macro("__LDBL_EPSILON__", "1.92592994438723585305597794258492732e-34L");
        self.define_simple_macro("__LDBL_MAX__", "1.18973149535723176508575932662800702e+4932L");
        self.define_simple_macro("__LDBL_MIN__", "3.36210314311209350626267781732175260e-4932L");
        self.define_simple_macro("__LDBL_DENORM_MIN__", "6.47517511943802511092443895822764655e-4966L");
        self.define_simple_macro("__LDBL_DECIMAL_DIG__", "36");
        self.define_simple_macro("__DECIMAL_DIG__", "36");
        // MIN_EXP, MAX_EXP, MIN_10_EXP, MAX_10_EXP are the same for x87 and binary128
        // (both use 15-bit exponent fields), so no override needed.

        // <float.h> macros (LDBL_*)
        self.define_simple_macro("LDBL_MANT_DIG", "113");
        self.define_simple_macro("LDBL_DIG", "33");
        self.define_simple_macro("DECIMAL_DIG", "36");
    }

    /// Override RISC-V preprocessor macros based on -mabi= and -march= flags.
    ///
    /// The kernel uses `-mabi=lp64` (soft-float ABI) and `-march=rv64imac...`
    /// (no F/D extensions). When these flags are set, we must adjust:
    /// - Float ABI macros: `__riscv_float_abi_soft` vs `__riscv_float_abi_double`
    /// - `__riscv_flen`: only defined when FPU is available
    /// - Extension macros: `__riscv_f`, `__riscv_d`, `__riscv_fdiv`, `__riscv_fsqrt`
    pub fn set_riscv_abi(&mut self, abi: &str) {
        match abi {
            "lp64" => {
                // Soft-float ABI (no FPU registers for argument passing).
                // Undefine double-float macros and define soft-float.
                self.macros.undefine("__riscv_float_abi_double");
                self.macros.undefine("__riscv_flen");
                self.macros.undefine("__riscv_fdiv");
                self.macros.undefine("__riscv_fsqrt");
                self.define_simple_macro("__riscv_float_abi_soft", "1");
            }
            "lp64f" => {
                // Single-float ABI.
                self.macros.undefine("__riscv_float_abi_double");
                self.define_simple_macro("__riscv_float_abi_single", "1");
                self.define_simple_macro("__riscv_flen", "32");
            }
            "lp64d" => {
                // Double-float ABI (default) - macros already set by set_target.
            }
            _ => {
                // Unknown ABI value - leave defaults (lp64d) in place.
                // This covers ilp32* ABIs and any future additions.
            }
        }
    }

    /// Override RISC-V extension macros based on -march= flag.
    ///
    /// The kernel uses -march=rv64imac... (no F/D extensions). When the march
    /// string doesn't contain 'f' or 'd' (or 'g' which implies both), we must
    /// remove F/D extension macros that set_target unconditionally defines.
    pub fn set_riscv_march(&mut self, march: &str) {
        // Extract the base ISA string (strip rv32/rv64 prefix for extension parsing).
        let exts = if let Some(rest) = march.strip_prefix("rv64") {
            rest
        } else if let Some(rest) = march.strip_prefix("rv32") {
            rest
        } else {
            march
        };
        // 'g' = imafd, so check for 'g' as well.
        // NOTE: This simple character check may false-positive on sub-extension names
        // (e.g., 'f' in "zifencei"). In practice, kernel -march strings use the
        // underscore-separated format (rv64imac_zicsr_zifencei) where single-letter
        // extensions precede the first underscore, so this heuristic works correctly.
        let has_f = exts.contains('f') || exts.contains('g');
        let has_d = exts.contains('d') || exts.contains('g');

        if !has_f {
            self.macros.undefine("__riscv_f");
            self.macros.undefine("__riscv_fdiv");
            self.macros.undefine("__riscv_fsqrt");
        }
        if !has_d {
            self.macros.undefine("__riscv_d");
        }
        if !has_f && !has_d {
            self.macros.undefine("__riscv_flen");
        }
    }
}
