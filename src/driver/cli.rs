//! CLI argument parsing for GCC-compatible command-line flags.
//!
//! Handles the full range of GCC flags that build systems like the Linux kernel,
//! Meson, and autoconf expect: optimization levels, debug info, preprocessor
//! directives, linker flags, target-specific machine flags, and query flags
//! like `--dumpmachine` and `--version`.
//!
//! Design: The parser is a simple `while` loop with a flat `match` on each
//! argument. No external parser library is used. Unknown flags are silently
//! ignored (matching GCC's behavior for unrecognized `-f` and `-m` flags),
//! which is critical for build system compatibility.

use super::pipeline::{Driver, CompileMode, CliDefine};
use crate::backend::Target;
use crate::common::error::ColorMode;

impl Driver {
    /// Parse GCC-compatible command-line arguments and populate driver fields.
    /// Returns `Ok(true)` if early exit was handled (query flags like -dumpmachine),
    /// `Ok(false)` if normal compilation should proceed, or `Err` for invalid args.
    pub fn parse_cli_args(&mut self, args: &[String]) -> Result<bool, String> {
        // Detect target from binary name (argv[0])
        let binary_name = std::path::Path::new(&args[0])
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("ccc");

        self.target = if binary_name.contains("arm") || binary_name.contains("aarch64") {
            Target::Aarch64
        } else if binary_name.contains("riscv") {
            Target::Riscv64
        } else if binary_name.contains("i686") || binary_name.contains("i386") {
            Target::I686
        } else {
            Target::X86_64
        };

        // Handle GCC query flags that exit immediately (before requiring input files).
        // These are used by configure scripts to detect the compiler and target.
        if Self::handle_query_flags(args, &self.target)? {
            return Ok(true);
        }

        // Expand @response_file arguments (GCC/MSVC convention).
        // Response files contain additional command-line arguments, one per line
        // or whitespace-separated. Build systems like Meson use them when the
        // command line would exceed OS limits.
        let expanded_args = Self::expand_response_files(&args[1..]);
        self.parse_main_args(&expanded_args)?;

        // Store raw args for GCC -m16 passthrough. We keep everything except
        // argv[0], -o <output>, -c/-S/-E (we set mode ourselves), and input files.
        // GCC understands all the same flags we accept, so forwarding them directly
        // preserves ordering semantics (e.g., -fcf-protection=none after =branch).
        if self.code16gcc {
            self.raw_args = args[1..].iter()
                .filter(|a| !self.input_files.contains(a))
                .cloned()
                .collect();
        }

        // Special case: no input files but -Wl,--version is present.
        // Build systems like Meson run `compiler -Wl,--version` without source files
        // to detect the linker type. Invoke our linker driver (GCC) directly.
        if self.input_files.is_empty()
            && self.linker_ordered_items.iter().any(|a| a.contains("--version"))
        {
            Self::run_linker_version_query(&self.target, &self.linker_ordered_items);
            return Ok(true);
        }

        Ok(false)
    }

    /// Handle early-exit query flags (--dumpmachine, --version, etc.).
    /// Returns Ok(true) if a query flag was handled and the process should exit.
    fn handle_query_flags(args: &[String], target: &Target) -> Result<bool, String> {
        for arg in &args[1..] {
            match arg.as_str() {
                "-dumpmachine" => {
                    println!("{}", target.triple());
                    return Ok(true);
                }
                "-dumpversion" => {
                    println!("14");
                    return Ok(true);
                }
                "--version" => {
                    // Meson detects GCC by checking for "Free Software Foundation"
                    // in the --version output. We claim GCC 14.2.0 compatibility
                    // (matching our __GNUC__/__GNUC_MINOR__/__GNUC_PATCHLEVEL__).
                    println!("ccc (Claude's C Compiler, GCC-compatible) 14.2.0");
                    println!("GCC is maintained by the Free Software Foundation, Inc.");
                    println!("This program was written by Claude Opus 4.6;");
                    println!("It is not intended for production use.");
                    // Show which GCC fallback features are enabled (if any)
                    let mut features = Vec::new();
                    if cfg!(feature = "gcc_linker") {
                        features.push("gcc_linker");
                    }
                    if cfg!(feature = "gcc_assembler") {
                        features.push("gcc_assembler");
                    }
                    if cfg!(feature = "gcc_m16") {
                        features.push("gcc_m16");
                    }
                    if features.is_empty() {
                        println!("Backend: standalone");
                    } else {
                        println!("Backend: {}", features.join(", "));
                    }
                    return Ok(true);
                }
                "-v" if args.len() == 2 => {
                    println!("ccc (Claude's C Compiler, GCC-compatible) 14.2.0");
                    println!("Target: {}", target.triple());
                    return Ok(true);
                }
                "-print-search-dirs" => {
                    println!("install: /usr/lib/gcc/{}/13/", target.triple());
                    println!("programs: /usr/bin/");
                    println!("libraries: {}", target.implicit_library_paths());
                    return Ok(true);
                }
                _ if arg.starts_with("-print-file-name=") => {
                    let name = &arg["-print-file-name=".len()..];
                    // Special case: "include" should return our bundled include
                    // directory so that build systems (e.g., Linux kernel) pick up
                    // our intrinsic headers (arm_neon.h, emmintrin.h, etc.) instead
                    // of the host GCC's headers which use incompatible builtins.
                    if name == "include" {
                        if let Some(bundled) = crate::frontend::preprocessor::Preprocessor::bundled_include_dir() {
                            println!("{}", bundled.display());
                            return Ok(true);
                        }
                    }
                    // Search standard library directories for the requested file.
                    // If found, print the full path; otherwise echo the name back
                    // (matching GCC behavior).
                    let triple = target.triple();
                    let search_dirs = [
                        format!("/usr/lib/gcc/{}/13/", triple),
                        format!("/usr/lib/gcc-cross/{}/13/", triple),
                        format!("/usr/lib/{}/", triple),
                        format!("/usr/{}/lib/", triple),
                        "/usr/lib/".to_string(),
                    ];
                    let mut found = false;
                    for dir in &search_dirs {
                        let path = format!("{}{}", dir, name);
                        if std::path::Path::new(&path).exists() {
                            println!("{}", path);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        println!("{}", name);
                    }
                    return Ok(true);
                }
                _ => {}
            }
        }
        Ok(false)
    }

    /// Expand `@file` response file arguments.
    /// Each `@path` argument is replaced by the contents of the file at `path`,
    /// split on whitespace. Non-`@` arguments are passed through unchanged.
    fn expand_response_files(args: &[String]) -> Vec<String> {
        let mut result = Vec::new();
        for arg in args {
            if let Some(path) = arg.strip_prefix('@') {
                if let Ok(contents) = std::fs::read_to_string(path) {
                    // Split on whitespace, respecting simple quoting
                    for token in Self::split_response_file(&contents) {
                        result.push(token);
                    }
                } else {
                    // If the file can't be read, pass the arg through unchanged
                    result.push(arg.clone());
                }
            } else {
                result.push(arg.clone());
            }
        }
        result
    }

    /// Split response file contents into tokens, handling simple quoting.
    fn split_response_file(contents: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut in_single_quote = false;
        let mut in_double_quote = false;
        let mut escape = false;

        for ch in contents.chars() {
            if escape {
                current.push(ch);
                escape = false;
                continue;
            }
            match ch {
                '\\' if !in_single_quote => {
                    escape = true;
                }
                '\'' if !in_double_quote => {
                    in_single_quote = !in_single_quote;
                }
                '"' if !in_single_quote => {
                    in_double_quote = !in_double_quote;
                }
                c if c.is_ascii_whitespace() && !in_single_quote && !in_double_quote => {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                }
                _ => {
                    current.push(ch);
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// Parse the main argument list (everything after argv[0]).
    fn parse_main_args(&mut self, args: &[String]) -> Result<(), String> {
        let mut explicit_language: Option<String> = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                // Output file
                "-o" => {
                    i += 1;
                    if i < args.len() {
                        self.output_path = args[i].clone();
                        self.output_path_set = true;
                    } else {
                        return Err("-o requires an argument".to_string());
                    }
                }

                // Compilation mode flags
                "-S" => self.mode = CompileMode::AssemblyOnly,
                "-c" => self.mode = CompileMode::ObjectOnly,
                "-E" => self.mode = CompileMode::PreprocessOnly,
                "-P" => self.suppress_line_markers = true,
                "-dM" => self.dump_defines = true,

                // Optimization levels
                //
                // -O0: Minimal optimization (mem2reg + resolve_asm + dead_statics only).
                //      Fast compile, debuggable output.
                // -O1: Basic optimizations (cfg_simplify, copy_prop, constant_fold,
                //      simplify, dce). Single iteration, no expensive analyses.
                // -O2: Full optimization pipeline (all passes, 3 iterations with
                //      GVN, LICM, IVSR, if-conversion, inlining, IPCP).
                // -O3: Aggressive — more iterations, tighter diminishing-returns
                //      threshold, more aggressive inlining.
                //
                // The `optimize` and `optimize_size` booleans also control predefined
                // macros (__OPTIMIZE__, __OPTIMIZE_SIZE__), which build systems like
                // the Linux kernel rely on.
                "-O0" => {
                    self.opt_level = 0;
                    self.optimize = false;
                    self.optimize_size = false;
                    self.omit_frame_pointer = false;
                }
                "-O" | "-O1" => {
                    self.opt_level = 1;
                    self.optimize = true;
                    self.optimize_size = false;
                    self.omit_frame_pointer = true;
                }
                "-O2" => {
                    self.opt_level = 2;
                    self.optimize = true;
                    self.optimize_size = false;
                    self.omit_frame_pointer = true;
                }
                "-O3" => {
                    self.opt_level = 3;
                    self.optimize = true;
                    self.optimize_size = false;
                    self.omit_frame_pointer = true;
                }
                "-Os" | "-Oz" => {
                    self.opt_level = 2;
                    self.optimize = true;
                    self.optimize_size = true;
                    self.omit_frame_pointer = true;
                }

                // Debug info
                "-g" => self.debug_info = true,
                arg if arg.starts_with("-g") && arg.len() > 2 => self.debug_info = true,

                // Verbose/diagnostic flags
                "-v" | "--verbose" => self.verbose = true,

                // Linker library flags: -lfoo
                arg if arg.starts_with("-l") => {
                    self.linker_ordered_items.push(arg.to_string());
                }

                // Linker pass-through: -Wl,flag1,flag2,...
                // Keep the whole -Wl argument together so that multi-part flags
                // like -Wl,-soname,libfoo.so and -Wl,-rpath,/path stay intact.
                // The linker code splits on commas internally.
                arg if arg.starts_with("-Wl,") => {
                    self.linker_ordered_items.push(arg.to_string());
                }

                // Linker pass-through: -Xlinker ARG
                // Each -Xlinker passes exactly one argument to the linker.
                // Convert to -Wl,ARG format for uniform downstream handling.
                "-Xlinker" => {
                    i += 1;
                    if i < args.len() {
                        self.linker_ordered_items.push(format!("-Wl,{}", args[i]));
                    }
                }

                // Assembler pass-through: -Wa,flag1,flag2,...
                arg if arg.starts_with("-Wa,") => {
                    for flag in arg[4..].split(',') {
                        if !flag.is_empty() {
                            self.assembler_extra_args.push(flag.to_string());
                        }
                    }
                }

                // Preprocessor pass-through: -Wp,-MMD,path or -Wp,-MD,path
                arg if arg.starts_with("-Wp,") => {
                    let flags: Vec<&str> = arg[4..].splitn(2, ',').collect();
                    if flags.len() == 2 && (flags[0] == "-MMD" || flags[0] == "-MD") {
                        self.dep_file = Some(flags[1].to_string());
                    }
                }

                // Warning flags
                arg if arg.starts_with("-W") => {
                    let flag = &arg[2..];
                    if !flag.is_empty() {
                        self.warning_config.process_flag(flag);
                    }
                }

                // Preprocessor defines
                "-D" => {
                    i += 1;
                    if i < args.len() {
                        self.add_define(&args[i]);
                    } else {
                        return Err("-D requires an argument".to_string());
                    }
                }
                arg if arg.starts_with("-D") => self.add_define(&arg[2..]),

                // Force-include files
                "-include" => {
                    i += 1;
                    if i < args.len() {
                        self.force_includes.push(args[i].clone());
                    } else {
                        return Err("-include requires an argument".to_string());
                    }
                }

                // Include paths
                "-I" => {
                    i += 1;
                    if i < args.len() {
                        self.add_include_path(&args[i]);
                    } else {
                        return Err("-I requires an argument".to_string());
                    }
                }
                arg if arg.starts_with("-I") => self.add_include_path(&arg[2..]),

                // Quote-only include paths (-iquote)
                "-iquote" => {
                    i += 1;
                    if i < args.len() {
                        self.quote_include_paths.push(args[i].clone());
                    } else {
                        return Err("-iquote requires an argument".to_string());
                    }
                }

                // System include paths (-isystem)
                "-isystem" => {
                    i += 1;
                    if i < args.len() {
                        self.isystem_include_paths.push(args[i].clone());
                    } else {
                        return Err("-isystem requires an argument".to_string());
                    }
                }

                // After include paths (-idirafter)
                "-idirafter" => {
                    i += 1;
                    if i < args.len() {
                        self.after_include_paths.push(args[i].clone());
                    } else {
                        return Err("-idirafter requires an argument".to_string());
                    }
                }

                // Library search paths
                "-L" => {
                    i += 1;
                    if i < args.len() {
                        self.linker_paths.push(args[i].clone());
                    }
                }
                arg if arg.starts_with("-L") => {
                    self.linker_paths.push(arg[2..].to_string());
                }

                // Suppress all predefined macros (-undef)
                // Must come before -U prefix match since -undef starts with -U
                "-undef" => {
                    self.undef_all = true;
                }
                // Undefine macro
                "-U" => {
                    i += 1;
                    if i < args.len() {
                        self.undef_macros.push(args[i].clone());
                    }
                }
                arg if arg.starts_with("-U") => {
                    self.undef_macros.push(arg[2..].to_string());
                }

                // Standard version flag: -std=c99 disables GNU extensions,
                // -std=gnu99 (or no flag) enables them.
                arg if arg.starts_with("-std=") => {
                    let std_value = &arg[5..];
                    // GNU dialects: gnu89, gnu99, gnu11, gnu17, gnu23, etc.
                    // Strict ISO: c89, c99, c11, c17, c23, iso9899:*, etc.
                    self.gnu_extensions = std_value.starts_with("gnu");
                    // gnu89 and c89 use GNU inline semantics by default;
                    // gnu99+ and c99+ use C99 inline semantics.
                    // Note: -fgnu89-inline can override this later on the command line.
                    self.gnu89_inline = matches!(std_value, "gnu89" | "c89" | "gnu90" | "c90"
                        | "iso9899:1990" | "iso9899:199409");
                }

                // Machine/target flags
                "-mfunction-return=thunk-extern" => self.function_return_thunk = true,
                "-mindirect-branch=thunk-extern" => self.indirect_branch_thunk = true,
                "-m16" => {
                    // -m16 generates i386 code with .code16gcc prepended so the
                    // GNU assembler adds operand/address-size override prefixes
                    // for 16-bit real mode execution. Used by the Linux kernel
                    // boot code (arch/x86/boot/).
                    self.target = Target::I686;
                    self.code16gcc = true;
                }
                "-m32" => {
                    // Switch to 32-bit i686 target. If already targeting i686
                    // (e.g. invoked as ccc-i686), this is a no-op.
                    if self.target != Target::I686 {
                        self.target = Target::I686;
                    }
                }
                "-mno-sse" | "-mno-sse2" | "-mno-mmx" | "-mno-sse3" | "-mno-ssse3"
                | "-mno-sse4" | "-mno-sse4.1" | "-mno-sse4.2" | "-mno-avx"
                | "-mno-avx2" | "-mno-avx512f" | "-mno-3dnow" => {
                    self.no_sse = true;
                }
                // Positive SIMD feature flags: define corresponding macros.
                // -mavx2 implies -mavx implies -msse4.2 implies -msse4.1 implies
                // -mssse3 implies -msse3 (matching GCC's implication chain).
                "-mavx2" => {
                    self.enable_avx2 = true;
                    self.enable_avx = true;
                    self.enable_sse4_2 = true;
                    self.enable_sse4_1 = true;
                    self.enable_ssse3 = true;
                    self.enable_sse3 = true;
                }
                "-mavx" => {
                    self.enable_avx = true;
                    self.enable_sse4_2 = true;
                    self.enable_sse4_1 = true;
                    self.enable_ssse3 = true;
                    self.enable_sse3 = true;
                }
                "-msse4.2" => {
                    self.enable_sse4_2 = true;
                    self.enable_sse4_1 = true;
                    self.enable_ssse3 = true;
                    self.enable_sse3 = true;
                }
                "-msse4.1" | "-msse4" => {
                    self.enable_sse4_1 = true;
                    self.enable_ssse3 = true;
                    self.enable_sse3 = true;
                }
                "-mssse3" => {
                    self.enable_ssse3 = true;
                    self.enable_sse3 = true;
                }
                "-msse3" => {
                    self.enable_sse3 = true;
                }
                "-mgeneral-regs-only" => self.general_regs_only = true,
                "-mcmodel=kernel" => self.code_model_kernel = true,
                "-mcmodel=small" | "-mcmodel=medlow" | "-mcmodel=medium" | "-mcmodel=medany" | "-mcmodel=large" => {
                    self.code_model_kernel = false;
                }
                arg if arg.starts_with("-mabi=") => {
                    self.riscv_abi = Some(arg["-mabi=".len()..].to_string());
                }
                arg if arg.starts_with("-march=") => {
                    self.riscv_march = Some(arg["-march=".len()..].to_string());
                }
                "-mlittle-endian" => {
                    // ARM64 target indicator: only arm64-gcc accepts -mlittle-endian.
                    // This allows `ccc -mlittle-endian` to build ARM code without
                    // requiring the binary to be named aarch64-linux-gnu-ccc.
                    if self.target == Target::X86_64 {
                        self.target = Target::Aarch64;
                    }
                }
                "-mno-relax" => self.riscv_no_relax = true,
                arg if arg.starts_with("-mregparm=") => {
                    let n: u8 = arg["-mregparm=".len()..].parse().unwrap_or(0);
                    self.regparm = n.min(3);
                }
                arg if arg.starts_with("-m") => {}

                // Feature flags
                "-fPIC" | "-fpic" | "-fPIE" | "-fpie" => self.pic = true,
                "-fno-PIC" | "-fno-pic" | "-fno-PIE" | "-fno-pie" => self.pic = false,
                "-fcf-protection=branch" | "-fcf-protection=full" => self.cf_protection_branch = true,
                "-fcf-protection=none" => self.cf_protection_branch = false,
                arg if arg.starts_with("-fpatchable-function-entry=") => {
                    let val = &arg["-fpatchable-function-entry=".len()..];
                    let parts: Vec<&str> = val.split(',').collect();
                    let total: u32 = parts[0].parse().unwrap_or(0);
                    let before: u32 = if parts.len() > 1 { parts[1].parse().unwrap_or(0) } else { 0 };
                    self.patchable_function_entry = Some((total, before));
                }
                "-fomit-frame-pointer" => self.omit_frame_pointer = true,
                "-fno-omit-frame-pointer" => self.omit_frame_pointer = false,
                "-fno-asynchronous-unwind-tables" | "-fno-unwind-tables" => self.no_unwind_tables = true,
                "-fasynchronous-unwind-tables" | "-funwind-tables" => self.no_unwind_tables = false,
                "-fno-jump-tables" => self.no_jump_tables = true,
                "-ffunction-sections" => self.function_sections = true,
                "-fno-function-sections" => self.function_sections = false,
                "-fdata-sections" => self.data_sections = true,
                "-fno-data-sections" => self.data_sections = false,
                "-fcommon" => self.fcommon = true,
                "-fno-common" => self.fcommon = false,
                "-fgnu89-inline" => self.gnu89_inline = true,
                "-fno-gnu89-inline" => self.gnu89_inline = false,
                // Diagnostic color: -fdiagnostics-color, -fdiagnostics-color={auto,always,never}
                "-fdiagnostics-color" | "-fcolor-diagnostics" => {
                    self.color_mode = ColorMode::Always;
                }
                "-fno-diagnostics-color" | "-fno-color-diagnostics" => {
                    self.color_mode = ColorMode::Never;
                }
                arg if arg.starts_with("-fdiagnostics-color=") => {
                    let value = &arg["-fdiagnostics-color=".len()..];
                    if let Some(mode) = ColorMode::from_flag(value) {
                        self.color_mode = mode;
                    }
                    // Unknown values silently ignored (matching GCC)
                }
                arg if arg.starts_with("-f") => {}

                // Linker flags
                "-static" => self.static_link = true,
                "-shared" => self.shared_lib = true,
                "-r" | "-relocatable" => self.relocatable = true,
                "-no-pie" | "-pie" => {}
                "-nostdlib" => self.nostdlib = true,
                "-nostdinc" => self.nostdinc = true,
                "-nodefaultlibs" => {}

                // Language selection
                "-x" => {
                    i += 1;
                    if i < args.len() {
                        let lang = args[i].as_str();
                        if lang == "none" {
                            explicit_language = None;
                        } else {
                            explicit_language = Some(args[i].clone());
                        }
                    } else {
                        return Err("-x requires an argument".to_string());
                    }
                }

                // Dependency generation flags
                "-MD" | "-MMD" => {
                    if self.dep_file.is_none() {
                        self.dep_file = Some(String::new());
                    }
                }
                "-MP" => {}
                "-M" | "-MM" => {
                    // -M/-MM: dependency-only mode. Preprocess and output
                    // make rules instead of compiling. GCC treats -M/-MM
                    // as implying -E.
                    self.dep_only = true;
                    self.mode = CompileMode::PreprocessOnly;
                }
                "-MF" => {
                    i += 1;
                    if i < args.len() {
                        self.dep_file = Some(args[i].clone());
                    }
                }
                "-MT" | "-MQ" => {
                    i += 1;
                    if i < args.len() {
                        self.dep_target = Some(args[i].clone());
                    }
                }

                // Misc flags
                "-rdynamic" => {
                    self.linker_ordered_items.push("-rdynamic".to_string());
                }
                "-pipe" | "-Xa" | "-Xc" | "-Xt" => {}
                "-pthread" => {
                    self.pthread = true;
                }

                // GCC --param flag: --param <name>=<value> or --param=<name>=<value>
                // Used by nix CC wrapper for hardening flags like ssp-buffer-size=4
                "--param" => {
                    // Skip the next argument (the parameter value)
                    i += 1;
                }
                arg if arg.starts_with("--param=") => {
                    // Single-argument form: --param=ssp-buffer-size=4
                    // Silently ignore
                }

                // Stdin input
                "-" => {
                    self.input_files.push("-".to_string());
                    self.explicit_language = explicit_language.clone();
                }

                // Unknown flags
                arg if arg.starts_with('-') => {
                    if self.verbose {
                        eprintln!("warning: unknown flag: {}", arg);
                    }
                }

                // Input file
                _ => {
                    if explicit_language.is_some() {
                        self.explicit_language = explicit_language.clone();
                    }
                    // Track object/archive files in the ordered linker items list
                    // to preserve their position relative to -l and -Wl, flags.
                    // C source files are compiled to temp objects and placed first.
                    if Self::is_object_or_archive(&args[i]) {
                        self.linker_ordered_items.push(args[i].clone());
                    }
                    self.input_files.push(args[i].clone());
                }
            }
            i += 1;
        }

        Ok(())
    }

    /// Handle -Wl,--version when no input files are given (Meson linker detection).
    ///
    /// When the `gcc_linker` feature is enabled, delegates to GCC for version info.
    /// When disabled, prints built-in linker version info.
    fn run_linker_version_query(target: &Target, linker_items: &[String]) {
        #[cfg(feature = "gcc_linker")]
        {
            let config = target.linker_config();
            let mut cmd = std::process::Command::new(config.command);
            cmd.args(config.extra_args);
            for item in linker_items {
                cmd.arg(item);
            }
            cmd.stdout(std::process::Stdio::inherit());
            cmd.stderr(std::process::Stdio::inherit());
            let _ = cmd.status();
        }
        #[cfg(not(feature = "gcc_linker"))]
        {
            let _ = (target, linker_items);
            // Print GNU ld-compatible version info for build system detection
            println!("GNU ld (Claude's C Compiler built-in) 2.42");
        }
    }

    /// Add a -D define from command line.
    pub fn add_define(&mut self, arg: &str) {
        if let Some(eq_pos) = arg.find('=') {
            self.defines.push(CliDefine {
                name: arg[..eq_pos].to_string(),
                value: arg[eq_pos + 1..].to_string(),
            });
        } else {
            self.defines.push(CliDefine {
                name: arg.to_string(),
                value: "1".to_string(),
            });
        }
    }

    /// Add a -I include path from command line.
    pub fn add_include_path(&mut self, path: &str) {
        self.include_paths.push(path.to_string());
    }
}
