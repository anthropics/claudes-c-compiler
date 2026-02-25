//! Core compiler driver: struct definition and compilation pipeline.
//!
//! The `Driver` struct holds all configuration parsed from the command line.
//! The compilation pipeline is: preprocess -> lex -> parse -> sema -> lower ->
//! mem2reg -> optimize -> phi-eliminate -> codegen.
//!
//! Submodules handle distinct concerns:
//! - `cli.rs`: GCC-compatible CLI argument parsing
//! - `external_tools.rs`: assembler, linker, and dependency file invocation
//! - `file_types.rs`: input file classification by extension/magic bytes

use crate::backend::Target;
use crate::common::error::{ColorMode, DiagnosticEngine, WarningConfig};
use crate::common::source::SourceManager;
use crate::frontend::preprocessor::Preprocessor;
use crate::frontend::lexer::Lexer;
use crate::frontend::parser::Parser;
use crate::frontend::sema::SemanticAnalyzer;
use crate::ir::lowering::Lowerer;
use crate::ir::mem2reg::{promote_allocas, eliminate_phis};
use crate::passes::run_passes;

/// Compilation mode - determines where in the pipeline to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    /// Full compilation: preprocess -> compile -> assemble -> link (default)
    Full,
    /// -S: Stop after generating assembly, output .s file
    AssemblyOnly,
    /// -c: Stop after assembling, output .o file
    ObjectOnly,
    /// -E: Stop after preprocessing, output preprocessed source to stdout
    PreprocessOnly,
}

/// A command-line define: -Dname or -Dname=value
#[derive(Debug, Clone)]
pub(crate) struct CliDefine {
    pub(crate) name: String,
    pub(crate) value: String,
}

/// The compiler driver orchestrates all compilation phases.
///
/// All fields are private; configuration is done through `parse_cli_args()`.
/// Use `has_input_files()` to check if input was provided before calling `run()`.
pub struct Driver {
    pub(super) target: Target,
    pub(super) output_path: String,
    pub(super) output_path_set: bool,
    pub(super) input_files: Vec<String>,
    pub(super) opt_level: u32,
    /// Whether optimization is enabled (any -O level except -O0).
    /// Used to define __OPTIMIZE__ predefined macro.
    pub(super) optimize: bool,
    /// Whether size optimization is requested (-Os or -Oz).
    /// Used to define __OPTIMIZE_SIZE__ predefined macro.
    pub(super) optimize_size: bool,
    pub(super) verbose: bool,
    pub(super) mode: CompileMode,
    pub(super) debug_info: bool,
    pub(super) defines: Vec<CliDefine>,
    pub(super) include_paths: Vec<String>,
    /// Quote-only include paths (from -iquote flags).
    /// Searched only for `#include "file"` (quoted includes), after the current
    /// file's directory and before -I paths.
    pub(super) quote_include_paths: Vec<String>,
    /// System include paths (from -isystem flags).
    /// Searched after -I paths, before default system paths.
    pub(super) isystem_include_paths: Vec<String>,
    /// After include paths (from -idirafter flags).
    /// Searched last, after all other include paths.
    pub(super) after_include_paths: Vec<String>,
    /// Library search paths (from -L flags)
    pub(super) linker_paths: Vec<String>,
    /// Ordered list of linker items preserving command-line order.
    /// Contains passthrough object/archive paths, -l flags, and -Wl, flags
    /// in the order they appeared on the command line. This is critical for
    /// flags like -Wl,--whole-archive which must precede the .a they affect.
    pub(super) linker_ordered_items: Vec<String>,
    /// Whether to link statically (-static)
    pub(super) static_link: bool,
    /// Whether to produce a shared library (-shared)
    pub(super) shared_lib: bool,
    /// Whether to omit standard library linking (-nostdlib)
    pub(super) nostdlib: bool,
    /// Whether to produce a relocatable object file (-r / -relocatable)
    pub(super) relocatable: bool,
    /// Whether to generate position-independent code (-fPIC/-fpic)
    pub(super) pic: bool,
    /// Files to force-include before the main source (-include flag)
    pub(super) force_includes: Vec<String>,
    /// Whether to replace `ret` with `jmp __x86_return_thunk` (-mfunction-return=thunk-extern)
    pub(super) function_return_thunk: bool,
    /// Whether to replace indirect calls/jumps with retpoline thunks (-mindirect-branch=thunk-extern)
    pub(super) indirect_branch_thunk: bool,
    /// Patchable function entry: (total_nops, nops_before_entry).
    /// Set by -fpatchable-function-entry=N[,M] where N is total NOPs and M is
    /// how many go before the entry point (the rest go after).
    /// Used by the Linux kernel for ftrace and static call patching.
    pub(super) patchable_function_entry: Option<(u32, u32)>,
    /// Whether to emit endbr64 at function entry points (-fcf-protection=branch).
    /// Required by the Linux kernel for Intel CET/IBT (Indirect Branch Tracking).
    pub(super) cf_protection_branch: bool,
    /// Whether SSE is disabled (-mno-sse). When true, the compiler must not emit
    /// any SSE/SSE2/AVX instructions (movdqu, movss, movsd, etc.).
    /// The Linux kernel uses -mno-sse to avoid FPU state in kernel code.
    pub(super) no_sse: bool,
    /// Explicit SIMD feature flags from -msse3, -msse4.1, -mavx, -mavx2, etc.
    /// When set, the corresponding __SSE3__, __AVX__, __AVX2__ macros are defined.
    /// These flags enable projects like blosc to compile SIMD-optimized code paths.
    pub(super) enable_sse3: bool,
    pub(super) enable_ssse3: bool,
    pub(super) enable_sse4_1: bool,
    pub(super) enable_sse4_2: bool,
    pub(super) enable_avx: bool,
    pub(super) enable_avx2: bool,
    /// Whether to use only general-purpose registers (-mgeneral-regs-only).
    /// On AArch64, this prevents FP/SIMD register usage. The Linux kernel uses
    /// this to avoid touching NEON/FP state. When set, variadic function prologues
    /// must not save q0-q7, and va_start sets __vr_offs=0 (no FP save area).
    pub(super) general_regs_only: bool,
    /// Whether to use the kernel code model (-mcmodel=kernel). All symbols
    /// are assumed to be in the negative 2GB of the virtual address space.
    /// Uses absolute sign-extended 32-bit addressing (movq $symbol) for
    /// global address references, producing R_X86_64_32S relocations.
    pub(super) code_model_kernel: bool,
    /// Whether to disable jump table emission for switch statements (-fno-jump-tables).
    /// The Linux kernel uses this with -mindirect-branch=thunk-extern (retpoline) to
    /// prevent indirect jumps that objtool would reject.
    pub(super) no_jump_tables: bool,
    /// RISC-V ABI override from -mabi= flag (e.g., "lp64", "lp64d", "lp64f").
    /// When set, overrides the default "lp64d" passed to the assembler.
    /// The Linux kernel uses -mabi=lp64 (soft-float) for kernel code.
    pub(super) riscv_abi: Option<String>,
    /// RISC-V architecture override from -march= flag (e.g., "rv64imac_zicsr_zifencei").
    /// When set, overrides the default "rv64gc" passed to the assembler.
    pub(super) riscv_march: Option<String>,
    /// RISC-V -mno-relax flag: suppress linker relaxation.
    /// When true, the codegen emits `.option norelax` and the assembler is
    /// invoked with `-mno-relax`. This prevents R_RISCV_RELAX relocations
    /// that would allow the linker to introduce absolute symbol references.
    /// Required by the Linux kernel's EFI stub (built with -fpic -mno-relax).
    pub(super) riscv_no_relax: bool,
    /// Explicit language override from -x flag.
    /// When set, overrides file extension detection for input language.
    /// Values: "c", "assembler", "assembler-with-cpp", "none" (reset).
    /// Used for stdin input ("-") and also for files like /dev/null where
    /// the extension doesn't indicate the language.
    pub(super) explicit_language: Option<String>,
    /// Extra arguments to pass to the assembler (from -Wa, flags).
    /// Used by kernel build to query assembler version via -Wa,--version.
    pub(super) assembler_extra_args: Vec<String>,
    /// Path to write dependency file (from -MF or -Wp,-MMD,path or -Wp,-MD,path).
    /// When set, the compiler writes a Make-compatible dependency file listing
    /// the input source as a dependency of the output object. Used by the Linux
    /// kernel build system (fixdep) to track header dependencies.
    pub(super) dep_file: Option<String>,
    /// Whether to prepend `.code16gcc` to the assembly output (-m16).
    /// When true, generated 32-bit i686 code is prefixed with `.code16gcc` so
    /// the assembler adds operand/address-size override prefixes for 16-bit
    /// real mode execution. Used by the Linux kernel boot code.
    pub(super) code16gcc: bool,
    /// Whether -M or -MM was specified (dependency-only mode).
    /// In this mode, the compiler preprocesses the source and outputs
    /// make-compatible dependency rules instead of compiling. -M includes
    /// system headers in the output, -MM skips them (but our minimal
    /// implementation doesn't list headers at all).
    pub(super) dep_only: bool,
    /// Dependency rule target from -MT flag (e.g., `-MT src/foo.o`).
    /// When set, overrides the default target in the dependency rule.
    /// Default: derive from input filename (replace extension with .o).
    pub(super) dep_target: Option<String>,
    /// Whether to suppress line markers in preprocessor output (-P flag).
    /// When true, `# <line> "<file>"` directives are stripped from -E output.
    /// Used by the Linux kernel's cc-version.sh to detect the compiler.
    pub(super) suppress_line_markers: bool,
    /// Whether -nostdinc was passed. When delegating to gcc for assembly
    /// preprocessing, this must be forwarded to prevent system header
    /// interference.
    pub(super) nostdinc: bool,
    /// Macro undefinitions from -U flags. These need to be forwarded when
    /// delegating preprocessing to gcc (e.g., -Uriscv for kernel linker scripts).
    pub(super) undef_macros: Vec<String>,
    /// Whether -undef was passed. Prevents the preprocessor from predefining
    /// any system-specific or GCC-specific macros. Must be forwarded when
    /// delegating to gcc (e.g., for DTS preprocessing in kernel builds).
    pub(super) undef_all: bool,
    /// Warning configuration parsed from -W flags. Controls which warnings are
    /// enabled, disabled, or promoted to errors. Processed left-to-right from
    /// the command line to match GCC semantics.
    pub(super) warning_config: WarningConfig,
    /// Whether GNU C extensions are enabled. Defaults to true.
    /// Set to false when -std=c99, -std=c11, etc. (strict ISO C mode) is used.
    /// When false, bare GNU keywords like `typeof` and `asm` are treated as
    /// identifiers (the __typeof__/__asm__ forms always work).
    pub(super) gnu_extensions: bool,
    /// Whether to place each function in its own section (-ffunction-sections).
    pub(super) function_sections: bool,
    /// Whether to place each data object in its own section (-fdata-sections).
    pub(super) data_sections: bool,
    /// Whether GNU89 inline semantics are in effect (-fgnu89-inline or -std=gnu89).
    /// When true, the preprocessor defines __GNUC_GNU_INLINE__ instead of __GNUC_STDC_INLINE__.
    /// This affects how projects like mpack select their inline linkage model.
    pub(super) gnu89_inline: bool,
    /// Whether to dump preprocessor defines instead of preprocessed output (-dM).
    /// When true with -E, outputs `#define NAME VALUE` for all predefined and
    /// user-defined macros instead of the preprocessed source. Used by build
    /// systems like Meson to detect compiler version via __GNUC__ etc.
    pub(super) dump_defines: bool,
    /// Color mode for diagnostic output, controlled by -fdiagnostics-color={auto,always,never}.
    /// Defaults to Auto (colorize when stderr is a terminal).
    pub(super) color_mode: ColorMode,
    /// Whether to use COMMON linkage for tentative definitions (-fcommon).
    /// When true, uninitialized global variables (tentative definitions) are emitted
    /// as COMMON symbols, allowing multiple TUs to define the same global without
    /// linker "multiple definition" errors. GCC defaulted to -fcommon before GCC 10.
    /// When false (-fno-common, the default), tentative definitions go to BSS with
    /// strong linkage, and duplicate definitions cause link errors.
    pub(super) fcommon: bool,
    /// Number of integer arguments to pass in registers for i686 (-mregparm=N).
    /// 0 = standard cdecl (all on stack), 1-3 = pass first N int args in EAX/EDX/ECX.
    pub(super) regparm: u8,
    /// Whether to omit the frame pointer (-fomit-frame-pointer).
    pub(super) omit_frame_pointer: bool,
    /// Whether to suppress .eh_frame unwind table generation
    /// (-fno-asynchronous-unwind-tables / -fno-unwind-tables).
    pub(super) no_unwind_tables: bool,
    /// Raw CLI arguments (excluding argv[0], -o, output path, and input files).
    /// Used for GCC -m16 passthrough: we forward all flags directly to GCC
    /// rather than trying to reconstruct them from parsed state.
    pub(super) raw_args: Vec<String>,
    /// Whether -pthread was specified. When true, the preprocessor defines
    /// _REENTRANT=1 (matching GCC/Clang behavior). Build systems that detect
    /// pthread support via configure (ax_pthread.m4) add -lpthread themselves.
    pub(super) pthread: bool,
}

impl Driver {
    pub fn new() -> Self {
        Self {
            target: Target::X86_64,
            output_path: "a.out".to_string(),
            output_path_set: false,
            input_files: Vec::new(),
            opt_level: 2, // Default to full optimization
            optimize: false, // Only set to true when user explicitly passes -O1 or higher
            optimize_size: false,
            verbose: false,
            mode: CompileMode::Full,
            debug_info: false,
            defines: Vec::new(),
            include_paths: Vec::new(),
            quote_include_paths: Vec::new(),
            isystem_include_paths: Vec::new(),
            after_include_paths: Vec::new(),
            linker_paths: Vec::new(),
            linker_ordered_items: Vec::new(),
            static_link: false,
            shared_lib: false,
            nostdlib: false,
            relocatable: false,
            pic: false,
            force_includes: Vec::new(),
            function_return_thunk: false,
            indirect_branch_thunk: false,
            patchable_function_entry: None,
            cf_protection_branch: false,
            no_sse: false,
            enable_sse3: false,
            enable_ssse3: false,
            enable_sse4_1: false,
            enable_sse4_2: false,
            enable_avx: false,
            enable_avx2: false,
            general_regs_only: false,
            code_model_kernel: false,
            no_jump_tables: false,
            riscv_abi: None,
            riscv_march: None,
            riscv_no_relax: false,
            explicit_language: None,
            assembler_extra_args: Vec::new(),
            dep_file: None,
            code16gcc: false,
            dep_only: false,
            dep_target: None,
            suppress_line_markers: false,
            nostdinc: false,
            undef_macros: Vec::new(),
            undef_all: false,
            warning_config: WarningConfig::new(),
            gnu_extensions: true,
            function_sections: false,
            data_sections: false,
            gnu89_inline: false,
            dump_defines: false,
            color_mode: ColorMode::Auto,
            fcommon: false,
            regparm: 0,
            omit_frame_pointer: false,
            no_unwind_tables: false,
            raw_args: Vec::new(),
            pthread: false,
        }
    }

    /// Whether the driver has any input files to process.
    pub fn has_input_files(&self) -> bool {
        !self.input_files.is_empty()
    }

    /// Run the compiler pipeline.
    pub fn run(&self) -> Result<(), String> {
        if self.input_files.is_empty() {
            return Err("No input files".to_string());
        }

        // Set the thread-local target pointer size for type system queries.
        // Must be done before any CType/IrType size computations.
        crate::common::types::set_target_ptr_size(self.target.ptr_size());
        crate::common::types::set_target_long_double_is_f128(
            matches!(self.target, Target::Aarch64 | Target::Riscv64)
        );

        match self.mode {
            CompileMode::PreprocessOnly => self.run_preprocess_only(),
            CompileMode::AssemblyOnly => self.run_assembly_only(),
            CompileMode::ObjectOnly => self.run_object_only(),
            CompileMode::Full => self.run_full(),
        }
    }

    // ---- Run modes ----

    fn run_preprocess_only(&self) -> Result<(), String> {
        for input_file in &self.input_files {
            if Self::is_assembly_source(input_file) || self.is_explicit_assembly() {
                // For .S files, delegate preprocessing to gcc which understands
                // assembly-specific preprocessor behavior
                self.preprocess_assembly(input_file)?;
                continue;
            }

            // -M/-MM: dependency-only mode. Output make rules and exit.
            // TODO: Currently only lists the source file as a dependency.
            // A full implementation should preprocess and list all #included
            // headers in the dependency rule (like GCC's -M output).
            if self.dep_only {
                // Determine the target for the dependency rule.
                let target = if let Some(ref t) = self.dep_target {
                    t.clone()
                } else {
                    // Default: derive from input filename by replacing extension with .o
                    let p = std::path::Path::new(input_file);
                    let stem = p.file_stem().unwrap_or_default().to_string_lossy();
                    format!("{}.o", stem)
                };
                let input_name = if input_file == "-" { "<stdin>" } else { input_file };
                let dep_line = format!("{}: {}\n", target, input_name);

                if self.output_path_set {
                    std::fs::write(&self.output_path, &dep_line)
                        .map_err(|e| format!("Cannot write {}: {}", self.output_path, e))?;
                } else {
                    print!("{}", dep_line);
                }
                continue;
            }

            let source = Self::read_source(input_file)?;

            let mut preprocessor = Preprocessor::new();
            self.configure_preprocessor(&mut preprocessor);
            let filename = if input_file == "-" { "<stdin>" } else { input_file };
            preprocessor.set_filename(filename);
            self.process_force_includes(&mut preprocessor)?;

            if self.dump_defines {
                // -dM mode: preprocess the source (to process #define/#undef
                // directives) then dump all resulting macro definitions.
                let _ = preprocessor.preprocess(&source);

                // Check for preprocessor errors (missing includes, #error, etc.)
                let pp_errors = preprocessor.errors();
                if !pp_errors.is_empty() {
                    for err in pp_errors {
                        eprintln!("{}:{}:{}: error: {}", err.file, err.line, err.col, err.message);
                    }
                    return Err(format!("{} preprocessor error(s) in {}", pp_errors.len(), filename));
                }

                let output = preprocessor.dump_defines();
                if self.output_path_set {
                    std::fs::write(&self.output_path, format!("{}\n", output))
                        .map_err(|e| format!("Cannot write {}: {}", self.output_path, e))?;
                } else {
                    println!("{}", output);
                }
            } else {
                let preprocessed = preprocessor.preprocess(&source);

                // Output the preprocessed text first, even if there are #error
                // directives. GCC and Clang also emit the full preprocessed
                // output before exiting non-zero, and many configure scripts
                // (e.g., privoxy) rely on grepping this output while ignoring
                // the exit code.
                let output = if self.suppress_line_markers {
                    Self::strip_line_markers(&preprocessed)
                } else {
                    preprocessed
                };

                if self.output_path_set {
                    std::fs::write(&self.output_path, &output)
                        .map_err(|e| format!("Cannot write {}: {}", self.output_path, e))?;
                    // Write dependency file if requested (e.g., -Wp,-MMD,<depfile>).
                    // The kernel build uses this when preprocessing linker scripts
                    // (.lds.S -> .lds) and fixdep expects the .d file to exist.
                    self.write_dep_file(input_file, &self.output_path);
                } else {
                    print!("{}", output);
                }

                // Check for preprocessor errors (missing includes, #error,
                // etc.) AFTER emitting output so the result is still available
                // for downstream tools that ignore the exit code.
                let pp_errors = preprocessor.errors();
                if !pp_errors.is_empty() {
                    for err in pp_errors {
                        eprintln!("{}:{}:{}: error: {}", err.file, err.line, err.col, err.message);
                    }
                    return Err(format!("{} preprocessor error(s) in {}", pp_errors.len(), filename));
                }
            }
        }
        Ok(())
    }

    /// Preprocess an assembly file.
    ///
    /// Uses the built-in C preprocessor by default. When the `gcc_assembler`
    /// feature is enabled, falls back to GCC for preprocessing.
    fn preprocess_assembly(&self, input_file: &str) -> Result<(), String> {
        // When gcc_assembler is enabled, delegate to GCC for assembly preprocessing.
        // GCC handles certain assembly-specific preprocessing behaviors that our
        // built-in preprocessor may not fully support in all edge cases.
        #[cfg(feature = "gcc_assembler")]
        {
            self.preprocess_assembly_gcc(input_file)
        }

        // Default: use built-in preprocessor
        #[cfg(not(feature = "gcc_assembler"))]
        {
            let source = Self::read_source(input_file)?;
            let mut preprocessor = Preprocessor::new();
            self.configure_preprocessor(&mut preprocessor);
            preprocessor.define_macro("__ASSEMBLER__", "1");
            preprocessor.define_macro("_CET_H_INCLUDED", "1");
            if self.target == crate::backend::Target::X86_64 {
                preprocessor.define_macro("_CET_ENDBR", "endbr64");
            } else {
                preprocessor.define_macro("_CET_ENDBR", "endbr32");
            }
            preprocessor.define_macro("_CET_NOTRACK", "notrack");
            preprocessor.set_asm_mode(true);
            preprocessor.set_filename(input_file);
            self.process_force_includes(&mut preprocessor)
                .map_err(|e| format!("Preprocessing {} failed: {}", input_file, e))?;
            let preprocessed = preprocessor.preprocess(&source);
            let output = if self.suppress_line_markers {
                Self::strip_line_markers(&preprocessed)
            } else {
                preprocessed
            };

            if self.output_path_set {
                std::fs::write(&self.output_path, &output)
                    .map_err(|e| format!("Cannot write {}: {}", self.output_path, e))?;
                self.write_dep_file(input_file, &self.output_path);
            } else {
                print!("{}", output);
            }
            Ok(())
        }
    }

    /// Preprocess an assembly file by delegating to gcc -E.
    ///
    /// Only compiled when the `gcc_assembler` feature is enabled.
    #[cfg(feature = "gcc_assembler")]
    fn preprocess_assembly_gcc(&self, input_file: &str) -> Result<(), String> {
        let config = self.target.assembler_config();
        let mut cmd = std::process::Command::new(config.command);
        cmd.args(config.extra_args);
        for path in &self.include_paths {
            cmd.arg("-I").arg(path);
        }
        for path in &self.quote_include_paths {
            cmd.arg("-iquote").arg(path);
        }
        for path in &self.isystem_include_paths {
            cmd.arg("-isystem").arg(path);
        }
        for path in &self.after_include_paths {
            cmd.arg("-idirafter").arg(path);
        }
        for def in &self.defines {
            if def.value == "1" {
                cmd.arg(format!("-D{}", def.name));
            } else {
                cmd.arg(format!("-D{}={}", def.name, def.value));
            }
        }
        for inc in &self.force_includes {
            cmd.arg("-include").arg(inc);
        }
        if self.nostdinc {
            cmd.arg("-nostdinc");
        }
        for undef in &self.undef_macros {
            cmd.arg(format!("-U{}", undef));
        }
        if self.undef_all {
            cmd.arg("-undef");
        }
        if let Some(ref lang) = self.explicit_language {
            cmd.arg("-x").arg(lang);
        }
        cmd.arg("-E");
        if self.suppress_line_markers {
            cmd.arg("-P");
        }
        if let Some(ref dep_path) = self.dep_file {
            if !dep_path.is_empty() {
                cmd.arg(format!("-Wp,-MMD,{}", dep_path));
            }
        }
        cmd.arg(input_file);
        if self.output_path_set {
            cmd.arg("-o").arg(&self.output_path);
        }
        let result = cmd.output()
            .map_err(|e| format!("Failed to preprocess {}: {}", input_file, e))?;
        if !self.output_path_set {
            print!("{}", String::from_utf8_lossy(&result.stdout));
        }
        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(format!("Preprocessing {} failed: {}", input_file, stderr));
        }
        Ok(())
    }

    fn run_assembly_only(&self) -> Result<(), String> {
        for input_file in &self.input_files {
            let out_path = self.output_for_input(input_file);
            // When gcc_m16 feature is enabled, delegate -m16 C compilation to GCC
            #[cfg(feature = "gcc_m16")]
            if self.code16gcc && Self::is_c_source(input_file) {
                use super::external_tools::GccM16Mode;
                self.compile_with_gcc_m16(input_file, &out_path, GccM16Mode::Assembly)?;
                self.write_dep_file(input_file, &out_path);
                if self.verbose {
                    eprintln!("Assembly output (GCC -m16): {}", out_path);
                }
                continue;
            }
            // Default path: compile with internal codegen (handles .code16gcc prepend)
            let asm = self.compile_to_assembly(input_file)?;
            std::fs::write(&out_path, &asm)
                .map_err(|e| format!("Cannot write {}: {}", out_path, e))?;
            self.write_dep_file(input_file, &out_path);
            if self.verbose {
                eprintln!("Assembly output: {}", out_path);
            }
        }
        Ok(())
    }

    fn run_object_only(&self) -> Result<(), String> {
        for input_file in &self.input_files {
            let out_path = self.output_for_input(input_file);
            if Self::is_assembly_source(input_file) || self.is_explicit_assembly() {
                // .s/.S files (or -x assembler): pass directly to the assembler
                self.assemble_source_file(input_file, &out_path)?;
            } else {
                // When gcc_m16 feature is enabled, delegate -m16 C compilation to GCC
                #[cfg(feature = "gcc_m16")]
                if self.code16gcc {
                    use super::external_tools::GccM16Mode;
                    self.compile_with_gcc_m16(input_file, &out_path, GccM16Mode::Object)?;
                    self.write_dep_file(input_file, &out_path);
                    if self.verbose {
                        eprintln!("Object output (GCC -m16): {}", out_path);
                    }
                    continue;
                }
                // Default path: compile with internal codegen
                let asm = self.compile_to_assembly(input_file)?;
                let extra = self.build_asm_extra_args();
                self.target.assemble_with_extra(&asm, &out_path, &extra)?;
            }
            self.write_dep_file(input_file, &out_path);
            if self.verbose {
                eprintln!("Object output: {}", out_path);
            }
        }
        Ok(())
    }

    fn run_full(&self) -> Result<(), String> {
        use crate::common::temp_files::TempFile;

        // TempFile guards ensure cleanup on all exit paths (success, error, panic).
        let mut temp_guards: Vec<TempFile> = Vec::new();

        let mut extra_passthrough: Vec<String> = Vec::new();

        for input_file in &self.input_files {
            if Self::is_object_or_archive(input_file) {
                // Object/archive files are passed to the linker via linker_ordered_items
                // (populated during argument parsing) to preserve their position relative
                // to -Wl, and -l flags. This is critical for --whole-archive support.
            } else if Self::is_assembly_source(input_file) || self.is_explicit_assembly() {
                // .s/.S files (or -x assembler): pass to assembler, then link
                let tmp = TempFile::new("ccc", Self::input_stem(input_file), "o");
                self.assemble_source_file(input_file, tmp.to_str())?;
                temp_guards.push(tmp);
            } else if !Self::is_c_source(input_file)
                && Self::looks_like_binary_object(input_file)
            {
                // Unrecognized extension but file has ELF/archive magic bytes -
                // treat as object file. These weren't caught by is_object_or_archive
                // at parse time, so add them to the extra passthrough list.
                extra_passthrough.push(input_file.clone());
            } else {
                // When gcc_m16 feature is enabled, delegate -m16 C compilation to GCC
                #[cfg(feature = "gcc_m16")]
                if self.code16gcc {
                    use super::external_tools::GccM16Mode;
                    let tmp = TempFile::new("ccc", Self::input_stem(input_file), "o");
                    self.compile_with_gcc_m16(input_file, tmp.to_str(), GccM16Mode::Object)?;
                    if self.verbose {
                        eprintln!("Compiled (GCC -m16): {}", input_file);
                    }
                    self.write_dep_file(input_file, &self.output_path);
                    temp_guards.push(tmp);
                    continue;
                }
                // Compile .c files to .o (handles .code16gcc prepend via internal codegen)
                let asm = self.compile_to_assembly(input_file)?;

                let tmp = TempFile::new("ccc", Self::input_stem(input_file), "o");
                let extra = self.build_asm_extra_args();
                self.target.assemble_with_extra(&asm, tmp.to_str(), &extra)?;
                // Write dependency file for this source file. When compiling and
                // linking in one step, GCC's -Wp,-MMD uses the .o name as the
                // dependency target. We use the output executable path as target,
                // which is sufficient for kernel build's fixdep processing.
                self.write_dep_file(input_file, &self.output_path);
                temp_guards.push(tmp);
            }
        }

        // Compiled temp objects go first (order relative to linker flags doesn't matter
        // for freshly compiled objects). Passthrough objects, -l flags, and -Wl, flags
        // follow in their original command-line order via linker_ordered_items.
        // Extra passthrough objects (detected by magic bytes at runtime) go after temps.
        let mut all_objects: Vec<&str> = temp_guards.iter().map(|t| t.to_str()).collect();
        for obj in &extra_passthrough {
            all_objects.push(obj.as_str());
        }

        // Build linker args from -l, -L, -static flags, preserving positional ordering
        let linker_args = self.build_linker_args();

        // Emit a synthetic verbose link line for build system compatibility.
        // CMake's CMakeParseImplicitLinkInfo.cmake looks for a line matching
        // `collect2` or `ld` and extracts -L paths from it, which populate
        // CMAKE_C_IMPLICIT_LINK_DIRECTORIES (used by find_library()).
        // Without this, CMake can't find libraries in multiarch paths like
        // /usr/lib/x86_64-linux-gnu/.
        // NOTE: This line is intentionally incomplete -- it only contains -L
        // flags for CMake detection, not actual linker arguments. Our built-in
        // linker is invoked internally, not via a subprocess.
        if self.verbose {
            let lib_paths = self.target.implicit_library_paths();
            let l_flags: String = lib_paths
                .split(':')
                .filter(|p| !p.is_empty())
                .map(|p| format!(" -L{}", p))
                .collect();
            eprintln!(
                " /usr/bin/ld -dynamic-linker {} -o {}{}",
                self.target.dynamic_linker(), self.output_path, l_flags,
            );
            eprintln!("LIBRARY_PATH={}", lib_paths);
        }

        if linker_args.is_empty() {
            self.target.link(&all_objects, &self.output_path)?;
        } else {
            self.target.link_with_args(&all_objects, &self.output_path, &linker_args)?;
        }

        // temp_guards drop here, cleaning up all temp .o files automatically.
        // (Also cleans up on early return via ? above.)

        if self.verbose {
            eprintln!("Output: {}", self.output_path);
        }

        Ok(())
    }

    // ---- Helper methods ----

    /// Determine the output path for a given input file and mode.
    fn output_for_input(&self, input_file: &str) -> String {
        if self.output_path_set {
            return self.output_path.clone();
        }
        let stem = if input_file == "-" {
            "stdin"
        } else {
            std::path::Path::new(input_file)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("a")
        };
        match self.mode {
            CompileMode::AssemblyOnly => format!("{}.s", stem),
            CompileMode::ObjectOnly => format!("{}.o", stem),
            CompileMode::PreprocessOnly => String::new(),
            CompileMode::Full => self.output_path.clone(),
        }
    }

    /// Get a short stem name for an input file (for temp file naming).
    fn input_stem(input_file: &str) -> &str {
        if input_file == "-" {
            "stdin"
        } else {
            std::path::Path::new(input_file)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("out")
        }
    }

    /// Read a C source file, tolerating non-UTF-8 content.
    /// Valid UTF-8 files are returned as-is. Non-UTF-8 bytes (0x80-0xFF) are
    /// encoded as PUA code points (U+E080-U+E0FF) which the lexer decodes
    /// back to raw bytes inside string/character literals.
    fn read_c_source_file(path: &str) -> Result<String, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Cannot read {}: {}", path, e))?;
        Ok(crate::common::encoding::bytes_to_string(bytes))
    }

    /// Read source from an input file. If the file is "-", reads from stdin.
    pub(super) fn read_source(input_file: &str) -> Result<String, String> {
        if input_file == "-" {
            use std::io::Read;
            let mut bytes = Vec::new();
            std::io::stdin().read_to_end(&mut bytes)
                .map_err(|e| format!("Cannot read from stdin: {}", e))?;
            Ok(crate::common::encoding::bytes_to_string(bytes))
        } else {
            Self::read_c_source_file(input_file)
        }
    }

    /// Configure the preprocessor with CLI-defined macros and target.
    pub(super) fn configure_preprocessor(&self, preprocessor: &mut Preprocessor) {
        // Set target architecture macros
        match self.target {
            Target::Aarch64 => preprocessor.set_target("aarch64"),
            Target::Riscv64 => preprocessor.set_target("riscv64"),
            Target::I686 => preprocessor.set_target("i686"),
            Target::X86_64 => preprocessor.set_target("x86_64"),
        }
        // Apply RISC-V ABI/arch overrides from -mabi= and -march= flags.
        // These must come after set_target() which sets defaults for RV64GC/lp64d.
        if self.target == Target::Riscv64 {
            if let Some(ref abi) = self.riscv_abi {
                preprocessor.set_riscv_abi(abi);
            }
            if let Some(ref march) = self.riscv_march {
                preprocessor.set_riscv_march(march);
            }
        }
        // Define __STRICT_ANSI__ for strict ISO C modes (-std=c99, -std=c11, etc.).
        // GCC defines this when non-GNU standard modes are used. Headers like
        // glibc's <features.h> and CPython's pymacro.h check for it to gate
        // GNU extensions like typeof.
        if !self.gnu_extensions {
            preprocessor.set_strict_ansi(true);
        }
        // Set inline semantics mode: -fgnu89-inline or -std=gnu89 uses GNU89
        // inline semantics (__GNUC_GNU_INLINE__), while the default C99+ mode
        // uses __GNUC_STDC_INLINE__. Projects like mpack use these macros to
        // select the correct inline linkage model.
        if self.gnu89_inline {
            preprocessor.set_gnu89_inline(true);
        }
        // Set optimization macros: __OPTIMIZE__ for -O1+, __OPTIMIZE_SIZE__ for -Os/-Oz.
        // The Linux kernel's BUILD_BUG() relies on __OPTIMIZE__ to expand to a noreturn
        // function call instead of a no-op.
        preprocessor.set_optimize(self.optimize, self.optimize_size);
        // Set PIC mode: defines __PIC__/__pic__ only when -fPIC is active.
        // This is critical for kernel code where RIP_REL_REF() checks #ifndef __pic__
        // to decide whether to use RIP-relative inline asm for early boot code.
        preprocessor.set_pic(self.pic);
        // Set SSE/SSE2/MMX predefined macros for x86 targets.
        // GCC/Clang always define __SSE__, __SSE2__, __MMX__ for x86_64 (baseline ISA).
        // Our i686 backend also uses SSE2, so we define them for i686 as well.
        // Projects like stb_image, minimp3, dr_libs use #ifdef __SSE2__ to enable SIMD paths.
        preprocessor.set_sse_macros(self.no_sse);
        // Define extended SIMD feature macros (__SSE3__, __AVX__, __AVX2__, etc.)
        // when the corresponding -msse3, -mavx, -mavx2 flags are passed.
        if !self.no_sse {
            preprocessor.set_extended_simd_macros(
                self.enable_sse3,
                self.enable_ssse3,
                self.enable_sse4_1,
                self.enable_sse4_2,
                self.enable_avx,
                self.enable_avx2,
            );
        }
        // Define _REENTRANT when -pthread is used.
        // GCC and Clang automatically define _REENTRANT=1 when -pthread is passed.
        // Many configure scripts (e.g., iperf3's ax_pthread.m4) check for this macro
        // to verify that pthread support is properly configured.
        if self.pthread {
            preprocessor.define_macro("_REENTRANT", "1");
        }
        for def in &self.defines {
            preprocessor.define_macro(&def.name, &def.value);
        }
        // Disable _FORTIFY_SOURCE: glibc's fortification headers define extern
        // always_inline wrapper functions that use __builtin_va_arg_pack() and
        // __builtin_va_arg_pack_len(), which are GCC-specific constructs that
        // only work when the wrapper is inlined into the caller. Since we cannot
        // fully support these constructs, the wrappers produce incorrect code
        // (infinite recursion or wrong control flow). Undefining _FORTIFY_SOURCE
        // prevents these wrappers from being emitted.
        preprocessor.undefine_macro("_FORTIFY_SOURCE");
        for path in &self.include_paths {
            preprocessor.add_include_path(path);
        }
        for path in &self.quote_include_paths {
            preprocessor.add_quote_include_path(path);
        }
        for path in &self.isystem_include_paths {
            preprocessor.add_system_include_path(path);
        }
        for path in &self.after_include_paths {
            preprocessor.add_after_include_path(path);
        }
    }

    /// Process force-included files (-include flag) through the preprocessor before
    /// the main source. Matches GCC's behavior: `-include file` acts as if
    /// `#include "file"` appeared at the top of the primary source file.
    /// The file is searched in this order:
    /// 1. Current working directory (for relative paths)
    /// 2. Include paths (-I, -isystem, system defaults, -idirafter)
    pub(super) fn process_force_includes(&self, preprocessor: &mut Preprocessor) -> Result<(), String> {
        for path in &self.force_includes {
            let resolved = if std::path::Path::new(path).is_absolute() {
                std::path::PathBuf::from(path)
            } else {
                // Try CWD first
                let cwd_path = std::env::current_dir()
                    .map(|cwd| cwd.join(path))
                    .unwrap_or_else(|_| std::path::PathBuf::from(path));
                if cwd_path.is_file() {
                    cwd_path
                } else {
                    // Search include paths (like #include "file")
                    preprocessor.resolve_include_path(path, false)
                        .unwrap_or(cwd_path)
                }
            };

            let content = Self::read_c_source_file(&resolved.to_string_lossy())
                .map_err(|e| format!("{}: {}", path, e))?;
            preprocessor.preprocess_force_include(&content, &resolved.to_string_lossy());
        }
        Ok(())
    }

    // ---- Core compilation pipeline ----

    /// Core pipeline: preprocess, lex, parse, sema, lower, optimize, codegen.
    ///
    /// Set `CCC_TIME_PHASES=1` in the environment to print per-phase timing to stderr.
    fn compile_to_assembly(&self, input_file: &str) -> Result<String, String> {
        let source = Self::read_source(input_file)?;

        let time_phases = std::env::var("CCC_TIME_PHASES").is_ok();
        let t0 = std::time::Instant::now();

        // Preprocess
        let mut preprocessor = Preprocessor::new();
        self.configure_preprocessor(&mut preprocessor);
        let filename = if input_file == "-" { "<stdin>" } else { input_file };
        preprocessor.set_filename(filename);
        self.process_force_includes(&mut preprocessor)?;
        let preprocessed = preprocessor.preprocess(&source);
        if time_phases { eprintln!("[TIME] preprocess: {:.3}s", t0.elapsed().as_secs_f64()); }

        // Create diagnostic engine for structured error/warning reporting
        let mut diagnostics = DiagnosticEngine::new();
        diagnostics.set_warning_config(self.warning_config.clone());
        diagnostics.set_color_mode(self.color_mode);

        // Emit preprocessor warnings through diagnostic engine with Cpp kind
        // so they can be controlled via -Wcpp / -Wno-cpp / -Werror=cpp.
        // Preprocessor diagnostics carry file:line:col location info for GCC-compatible output.
        for warn in preprocessor.warnings() {
            let diag = crate::common::error::Diagnostic::warning_with_kind(
                warn.message.clone(),
                crate::common::error::WarningKind::Cpp,
            ).with_location(&warn.file, warn.line, warn.col);
            diagnostics.emit(&diag);
        }

        // Check for #error directives and missing #include errors
        let pp_errors = preprocessor.errors();
        if !pp_errors.is_empty() {
            for err in pp_errors {
                let diag = crate::common::error::Diagnostic::error(err.message.clone())
                    .with_location(&err.file, err.line, err.col);
                diagnostics.emit(&diag);
            }
            return Err(format!("{} preprocessor error(s) in {}", pp_errors.len(), input_file));
        }

        // Lex
        let t1 = std::time::Instant::now();
        let mut source_manager = SourceManager::new();
        // Move preprocessed output into source manager (avoids cloning the entire string).
        // Then borrow a reference back for the lexer.
        let file_id = source_manager.add_file(input_file.to_string(), preprocessed);
        // Build line map from preprocessor line markers for source location tracking.
        // Uses stored content from add_file() and reuses already-computed line offsets.
        source_manager.build_line_map();
        // Transfer macro expansion metadata from preprocessor to source manager
        // for "in expansion of macro 'X'" diagnostic notes.
        let macro_expansions = preprocessor.take_macro_expansion_info();
        source_manager.set_macro_expansions(macro_expansions);
        let mut lexer = Lexer::new(source_manager.get_content(file_id), file_id);
        lexer.set_gnu_extensions(self.gnu_extensions);
        let tokens = lexer.tokenize();
        if time_phases { eprintln!("[TIME] lex: {:.3}s ({} tokens)", t1.elapsed().as_secs_f64(), tokens.len()); }

        if self.verbose {
            eprintln!("Lexed {} tokens from {}", tokens.len(), input_file);
        }

        // Parse -- the diagnostic engine holds the source manager for span
        // resolution and snippet rendering. It's also set on the parser for
        // backward-compatible span_to_location() calls.
        let t2 = std::time::Instant::now();
        diagnostics.set_source_manager(source_manager);
        let mut parser = Parser::new(tokens);
        parser.set_diagnostics(diagnostics);
        let ast = parser.parse();
        if time_phases { eprintln!("[TIME] parse: {:.3}s", t2.elapsed().as_secs_f64()); }

        if parser.error_count > 0 {
            return Err(format!("{}: {} parse error(s)", input_file, parser.error_count));
        }

        // Retrieve diagnostic engine (which holds the source manager) for subsequent phases
        let diagnostics = parser.take_diagnostics();

        if self.verbose {
            eprintln!("Parsed {} declarations", ast.decls.len());
        }

        // Semantic analysis -- pass diagnostic engine to sema (still holds
        // the source manager so sema diagnostics can resolve spans)
        let t3 = std::time::Instant::now();
        let mut sema = SemanticAnalyzer::new();
        sema.set_diagnostics(diagnostics);
        if let Err(error_count) = sema.analyze(&ast) {
            // Errors already emitted through diagnostic engine with source spans
            return Err(format!("{} error(s) during semantic analysis", error_count));
        }
        let mut diagnostics = sema.take_diagnostics();
        let sema_result = sema.into_result();
        // Extract source manager for debug info emission (-g) after sema is done
        let source_manager = diagnostics.take_source_manager();
        if time_phases { eprintln!("[TIME] sema: {:.3}s", t3.elapsed().as_secs_f64()); }

        // Check for warnings promoted to errors by -Werror / -Werror=<name>.
        // The sema pass may have returned Ok (no hard errors), but the diagnostic
        // engine may have accumulated promoted-warning-errors that should stop compilation.
        if diagnostics.has_errors() {
            return Err(format!("{} error(s) (warnings promoted by -Werror)", diagnostics.error_count()));
        }

        // Log diagnostic summary if there were any warnings
        if self.verbose && diagnostics.warning_count() > 0 {
            eprintln!("{} warning(s) generated", diagnostics.warning_count());
        }

        // Lower to IR (target-aware for ABI-specific lowering decisions)
        // Pass sema's TypeContext, function signatures, and expression type annotations
        // to the lowerer so it has pre-populated type info upfront.
        let t4 = std::time::Instant::now();
        let lowerer = Lowerer::with_type_context(
            self.target,
            sema_result.type_context,
            sema_result.functions,
            sema_result.expr_types,
            sema_result.const_values,
            diagnostics,
            self.gnu89_inline,
        );
        let (mut module, diagnostics) = lowerer.lower(&ast);

        // Apply #pragma weak directives from the preprocessor.
        for (symbol, target) in &preprocessor.weak_pragmas {
            if let Some(ref alias_target) = target {
                // #pragma weak symbol = alias -> create weak alias
                module.aliases.push((std::rc::Rc::clone(symbol), std::rc::Rc::clone(alias_target), true));
            } else {
                // #pragma weak symbol -> mark as weak
                module.symbol_attrs.push((std::rc::Rc::clone(symbol), true, None));
            }
        }

        // Apply #pragma redefine_extname directives from the preprocessor.
        // TODO: This uses .set aliases which works when both symbols are defined
        // locally, but a proper implementation would rename symbol references
        // during lowering/codegen for the case where new_name is external.
        for (old_name, new_name) in &preprocessor.redefine_extname_pragmas {
            module.aliases.push((std::rc::Rc::clone(old_name), std::rc::Rc::clone(new_name), false));
        }

        // Apply -fcommon: mark tentative definitions as COMMON symbols.
        // A tentative definition is a global variable at file scope with no initializer
        // and no extern/static storage class. With -fcommon, these use COMMON linkage
        // so the linker merges duplicates across TUs instead of reporting errors.
        if self.fcommon {
            for global in &mut module.globals {
                if !global.is_common && !global.is_extern && !global.is_static
                    && !global.is_thread_local
                    && matches!(global.init, crate::ir::module::GlobalInit::Zero)
                {
                    global.is_common = true;
                }
            }
        }

        if time_phases { eprintln!("[TIME] lowering: {:.3}s ({} functions)", t4.elapsed().as_secs_f64(), module.functions.len()); }

        // Check for errors emitted during lowering (e.g., unresolved types, invalid constructs)
        if diagnostics.has_errors() {
            return Err(format!("{} error(s) during IR lowering", diagnostics.error_count()));
        }

        // Log diagnostic summary if there were any warnings during lowering
        if self.verbose && diagnostics.warning_count() > 0 {
            eprintln!("{} warning(s) during lowering", diagnostics.warning_count());
        }

        if self.verbose {
            eprintln!("Lowered to {} IR functions", module.functions.len());
        }

        // Run optimization passes
        let t5 = std::time::Instant::now();
        promote_allocas(&mut module);
        if time_phases { eprintln!("[TIME] mem2reg: {:.3}s", t5.elapsed().as_secs_f64()); }

        let t6 = std::time::Instant::now();
        run_passes(&mut module, self.opt_level, self.target);
        if time_phases { eprintln!("[TIME] opt passes: {:.3}s", t6.elapsed().as_secs_f64()); }

        // Lower SSA phi nodes to copies before codegen
        let t7 = std::time::Instant::now();
        eliminate_phis(&mut module);
        if time_phases { eprintln!("[TIME] phi elimination: {:.3}s", t7.elapsed().as_secs_f64()); }

        // Note: we intentionally do NOT run copy_prop after phi elimination.
        // The IR is no longer in SSA form at this point - Copy instructions from
        // phi elimination represent moves at specific program points. Propagating
        // through them can change semantics (reading a value before it's defined
        // in the current iteration of a loop). Stack size reduction is handled
        // by copy coalescing in codegen instead.

        // Generate assembly using target-specific codegen
        let t8 = std::time::Instant::now();
        let opts = crate::backend::CodegenOptions {
            pic: self.pic || self.shared_lib,
            function_return_thunk: self.function_return_thunk,
            indirect_branch_thunk: self.indirect_branch_thunk,
            patchable_function_entry: self.patchable_function_entry,
            cf_protection_branch: self.cf_protection_branch,
            no_sse: self.no_sse,
            general_regs_only: self.general_regs_only,
            code_model_kernel: self.code_model_kernel,
            no_jump_tables: self.no_jump_tables,
            no_relax: self.riscv_no_relax,
            debug_info: self.debug_info,
            function_sections: self.function_sections,
            data_sections: self.data_sections,
            code16gcc: self.code16gcc,
            regparm: self.regparm,
            omit_frame_pointer: self.omit_frame_pointer,
            emit_cfi: !self.no_unwind_tables,
        };
        let asm = self.target.generate_assembly_with_opts_and_debug(
            &module, &opts, source_manager.as_ref(),
        );
        if time_phases { eprintln!("[TIME] codegen: {:.3}s ({} bytes asm)", t8.elapsed().as_secs_f64(), asm.len()); }

        if time_phases { eprintln!("[TIME] total compile {}: {:.3}s", input_file, t0.elapsed().as_secs_f64()); }

        if self.verbose {
            eprintln!("Generated {:?} assembly ({} bytes)", self.target, asm.len());
        }

        Ok(asm)
    }
}

impl Default for Driver {
    fn default() -> Self {
        Self::new()
    }
}
