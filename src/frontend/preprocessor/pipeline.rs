//! Full C preprocessor implementation.
//!
//! Struct definition, core preprocessing pipeline, directive dispatch,
//! and public configuration API. Predefined macros and target configuration
//! live in `predefined_macros`, pragma handling in `pragmas`, and text
//! processing (comment stripping, line joining) in `text_processing`.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use std::fmt::Write;
use std::path::PathBuf;
use std::rc::Rc;

use super::macro_defs::{MacroDef, MacroTable, parse_define};
use super::conditionals::{ConditionalStack, evaluate_condition};
use super::builtin_macros::define_builtin_macros;
use super::utils::{is_ident_start, is_ident_cont};
use super::text_processing::{strip_line_comment, split_first_word};

/// Deduplicate a list of macro names, preserving order (first occurrence wins).
/// Used to remove duplicate names from nested macro expansions.
fn dedup_macro_names(names: Vec<Rc<str>>) -> Vec<Rc<str>> {
    let mut unique = Vec::new();
    for name in names {
        if !unique.contains(&name) {
            unique.push(name);
        }
    }
    unique
}

/// Maximum number of newlines to accumulate while joining lines for unbalanced
/// parentheses in macro arguments. Prevents runaway accumulation when a source
/// file has a genuinely unbalanced parenthesis. Must be large enough to handle
/// real-world macro calls that span many lines (e.g., QEMU's qapi-introspect.c
/// has a single QLIT_QLIST() macro invocation spanning ~32000 lines).
const MAX_PENDING_NEWLINES: usize = 100_000;

/// A preprocessor diagnostic with source location information.
///
/// Stores the file name, line number, and column where the diagnostic was
/// generated, enabling GCC-compatible `file:line:col: error:` / `file:line:col: warning:`
/// output for `#error`, `#warning`, and missing `#include` errors.
#[derive(Debug, Clone)]
pub struct PreprocessorDiagnostic {
    /// The source file where the diagnostic originated.
    pub file: String,
    /// The 1-based line number in the source file.
    pub line: usize,
    /// The 1-based column number in the source file.
    pub col: usize,
    /// The diagnostic message text (e.g., `#error "bad config"`).
    pub message: String,
}

pub struct Preprocessor {
    pub(super) macros: MacroTable,
    conditionals: ConditionalStack,
    pub(super) includes: Vec<String>,
    pub(super) filename: String,
    pub(super) errors: Vec<PreprocessorDiagnostic>,
    /// Collected preprocessor warnings (e.g., #warning directives).
    pub(super) warnings: Vec<PreprocessorDiagnostic>,
    /// Include search paths (from -I flags)
    pub(super) include_paths: Vec<PathBuf>,
    /// Quote include paths (from -iquote flags), searched only for #include "..."
    pub(super) quote_include_paths: Vec<PathBuf>,
    /// System include paths explicitly added (from -isystem flags)
    pub(super) isystem_include_paths: Vec<PathBuf>,
    /// After include paths (from -idirafter flags), searched last
    pub(super) after_include_paths: Vec<PathBuf>,
    /// System include paths (default search paths)
    pub(super) system_include_paths: Vec<PathBuf>,
    /// Files currently being processed (for recursion detection)
    pub(super) include_stack: Vec<PathBuf>,
    /// Files that have been included with #pragma once
    pub(super) pragma_once_files: FxHashSet<PathBuf>,
    /// Whether to actually resolve includes (can be disabled for testing)
    pub(super) resolve_includes: bool,
    /// Declarations to inject into the output (from #include processing).
    pub(super) pending_injections: Vec<String>,
    /// Stack for #pragma push_macro / pop_macro.
    /// Maps macro name -> stack of saved definitions (None = was undefined).
    pub(super) macro_save_stack: FxHashMap<Rc<str>, Vec<Option<MacroDef>>>,
    /// Line offset set by #line directive: effective_line = line_offset + (source_line - line_offset_base)
    /// When None, no #line has been issued and __LINE__ uses the source line directly.
    line_override: Option<(usize, usize)>, // (target_line, source_line_at_directive)
    /// #pragma weak directives: (symbol, optional_alias_target)
    /// - (symbol, None) means "mark symbol as weak"
    /// - (symbol, Some(target)) means "symbol is a weak alias for target"
    pub weak_pragmas: Vec<(Rc<str>, Option<Rc<str>>)>,
    /// #pragma redefine_extname directives: (old_name, new_name)
    pub redefine_extname_pragmas: Vec<(Rc<str>, Rc<str>)>,
    /// Accumulated output from force-included files (-include).
    /// Prepended to the main source's preprocessed output so that pragma
    /// synthetic tokens (e.g., visibility push/pop) take effect.
    force_include_output: String,
    /// Cache for include path resolution.
    /// Maps (include_path, is_system, current_dir_key) to the resolved filesystem path.
    /// This avoids repeated `stat()` calls when the same header is included from
    /// multiple locations with the same include search path configuration.
    /// The current_dir_key is the parent directory of the including file for quoted
    /// includes (since resolution depends on it), or empty for system includes.
    pub(super) include_resolve_cache: FxHashMap<(String, bool, PathBuf), Option<PathBuf>>,
    /// Include guard detection: maps file paths to their guard macro names.
    ///
    /// After preprocessing an included file, we scan the raw source to detect if
    /// the entire file is wrapped in a classic include guard pattern:
    ///   #ifndef GUARD_MACRO
    ///   #define GUARD_MACRO
    ///   ...
    ///   #endif
    ///
    /// On subsequent #include of the same file, if the guard macro is still defined,
    /// we skip re-processing entirely (same optimization as GCC/Clang).
    pub(super) include_guard_macros: FxHashMap<PathBuf, Rc<str>>,
    /// Reusable FxHashSet for directive-level macro expansion (handle_if, handle_elif,
    /// handle_line_directive, #error). Avoids allocating a new FxHashSet per directive.
    directive_expanding: FxHashSet<Rc<str>>,
    /// Macro expansion metadata: maps preprocessed output line numbers to
    /// the macros expanded on that line. Populated during preprocessing and
    /// passed to the SourceManager for diagnostic rendering.
    macro_expansion_info: Vec<crate::common::source::MacroExpansionInfo>,
}

impl Preprocessor {
    pub fn new() -> Self {
        let mut pp = Self {
            macros: MacroTable::new(),
            conditionals: ConditionalStack::new(),
            includes: Vec::new(),
            filename: String::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            include_paths: Vec::new(),
            quote_include_paths: Vec::new(),
            isystem_include_paths: Vec::new(),
            after_include_paths: Vec::new(),
            system_include_paths: Self::default_system_include_paths(),
            include_stack: Vec::new(),
            pragma_once_files: FxHashSet::default(),
            resolve_includes: true,
            pending_injections: Vec::new(),
            macro_save_stack: FxHashMap::default(),
            line_override: None,
            weak_pragmas: Vec::new(),
            redefine_extname_pragmas: Vec::new(),
            force_include_output: String::new(),
            include_resolve_cache: FxHashMap::default(),
            include_guard_macros: FxHashMap::default(),
            directive_expanding: FxHashSet::default(),
            macro_expansion_info: Vec::new(),
        };
        pp.define_predefined_macros();
        define_builtin_macros(&mut pp.macros);
        pp
    }

    /// Process source code, expanding macros and handling conditionals.
    /// Returns the preprocessed source with embedded line markers for source tracking.
    pub fn preprocess(&mut self, source: &str) -> String {
        // Emit initial line marker for the main file
        let line_marker = format!("# 1 \"{}\"\n", self.filename);
        let main_output = self.preprocess_source(source, false);
        // Prepend any output from force-included files (e.g., pragma synthetic tokens)
        let (result, prefix_lines) = if self.force_include_output.is_empty() {
            let prefix_lines = line_marker.as_bytes().iter().filter(|&&b| b == b'\n').count() as u32;
            let mut result = line_marker;
            result.push_str(&main_output);
            (result, prefix_lines)
        } else {
            let mut result = std::mem::take(&mut self.force_include_output);
            result.push_str(&line_marker);
            let prefix_lines = result.as_bytes().iter().filter(|&&b| b == b'\n').count() as u32;
            result.push_str(&main_output);
            (result, prefix_lines)
        };
        // Adjust macro expansion line numbers to account for prepended content
        // (line markers, force-include output) that shifts all line numbers.
        if prefix_lines > 0 {
            for info in &mut self.macro_expansion_info {
                info.pp_line += prefix_lines;
            }
        }
        result
    }

    /// Process source code from an included file. Same pipeline as preprocess()
    /// but saves/restores the conditional stack and skips pending_injections.
    pub(super) fn preprocess_included(&mut self, source: &str) -> String {
        self.preprocess_source(source, true)
    }

    /// Unified preprocessing pipeline for both top-level and included sources.
    ///
    /// When `is_include` is true:
    /// - Saves and restores the conditional stack (each file gets its own)
    /// - Does not emit pending_injections (those only apply to top-level)
    /// - Only processes directives when no multi-line accumulation is pending
    fn preprocess_source(&mut self, source: &str, is_include: bool) -> String {
        // Per C standard (C11 5.1.1.2), translation phases are:
        // Phase 2: Line splicing (backslash-newline removal)
        // Phase 3: Comment replacement
        // So we must join continued lines BEFORE stripping comments.
        let source = self.join_continued_lines(source);
        let (source, line_map) = Self::strip_block_comments(&source);
        let mut output = String::with_capacity(source.len());

        // For included files, save and reset the conditional stack and line override
        let saved_conditionals = if is_include {
            Some(std::mem::take(&mut self.conditionals))
        } else {
            None
        };
        let saved_line_override = if is_include {
            self.line_override.take()
        } else {
            None
        };

        // Buffer for accumulating multi-line macro invocations
        let mut pending_line = String::new();
        let mut pending_newlines: usize = 0;

        // Track current line number in the preprocessed output for macro expansion metadata.
        // Incremented whenever a newline is appended to the output string.
        let mut pp_output_line: u32 = 0;

        // Reusable FxHashSet for macro expansion, avoiding per-line allocation.
        // This set tracks which macros are currently being expanded (to prevent
        // infinite recursion per C11 §6.10.3.4). It's cleared before each use
        // by expand_line_reuse().
        let mut expanding: crate::common::fx_hash::FxHashSet<Rc<str>> = crate::common::fx_hash::FxHashSet::default();

        // Enable macro expansion tracking for diagnostic "in expansion of macro" notes.
        // Only track at top level (not within included files) to avoid duplicate entries.
        if !is_include {
            self.macros.set_track_expansions(true);
        }

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            // Map output line number to original source line number using the
            // line_map from comment stripping (accounts for removed newlines in
            // block comments).
            let source_line_num = line_map.get(line_num);

            // Update __LINE__, accounting for any #line directive override
            let effective_line = if let Some((target_line, source_line_at_directive)) = self.line_override {
                // After #line N, __LINE__ = N + (current_source_line - source_line_of_directive)
                let offset = source_line_num.saturating_sub(source_line_at_directive);
                target_line + offset
            } else {
                source_line_num + 1
            };
            self.macros.set_line(effective_line);

            // Directive handling: #if/#ifdef/#ifndef/#elif/#else/#endif must always
            // be processed regardless of pending multi-line accumulation. Other
            // directives (#include, #define, etc.) are only processed when there's
            // no pending line in included files.
            let is_directive = trimmed.starts_with('#');
            let is_conditional_directive = if is_directive {
                let after_hash = trimmed[1..].trim_start();
                after_hash.starts_with("if")
                    || after_hash.starts_with("elif")
                    || after_hash.starts_with("else")
                    || after_hash.starts_with("endif")
            } else {
                false
            };

            // When we're accumulating a multi-line macro invocation (pending_line
            // is non-empty) and hit a conditional directive, we must process the
            // directive to update the conditional stack but NOT flush the pending
            // line. The macro argument collection must continue across
            // #ifdef/#endif boundaries. This handles cases like:
            //   FOO(1, #ifdef BAR 2, #endif 3)
            // where the #ifdef/#endif must be evaluated but the macro args keep
            // accumulating until the closing ')'.
            if is_conditional_directive && !pending_line.is_empty() {
                // Process the conditional directive (updates conditional stack)
                // Column of the directive keyword (1-based, after '#')
                let hash_col = line.find('#').map(|i| i + 2).unwrap_or(1);
                self.process_directive(trimmed, source_line_num + 1, hash_col);
                // Don't add the directive line to pending_line, don't flush.
                // Just count a newline for line numbering preservation.
                pending_newlines += 1;
                continue;
            }

            // Similarly, when accumulating a multi-line macro invocation in an
            // included file, #define/#undef directives must be processed (to
            // register/remove macros) without interrupting the accumulation.
            // Without this, #define directives inside #ifdef blocks in included
            // files would leak into the output as literal text. This mirrors the
            // conditional directive handling above.
            let is_define_undef_directive = if is_directive && !is_conditional_directive {
                let after_hash = trimmed[1..].trim_start();
                after_hash.starts_with("define") || after_hash.starts_with("undef")
            } else {
                false
            };
            if is_define_undef_directive && is_include && !pending_line.is_empty() {
                if self.conditionals.is_active() {
                    let hash_col = line.find('#').map(|i| i + 2).unwrap_or(1);
                    self.process_directive(trimmed, source_line_num + 1, hash_col);
                }
                pending_newlines += 1;
                continue;
            }

            // Handle #line directives during multi-line accumulation.
            // Generated code (e.g. gforth's prim.i) can have #line directives
            // between every line of code, including in the middle of multi-line
            // expressions. These must be consumed (updating line tracking state)
            // without being appended to the pending accumulation buffer.
            let is_line_directive = if is_directive && !is_conditional_directive {
                let after_hash = trimmed[1..].trim_start();
                after_hash.starts_with("line ")
                    || after_hash.starts_with("line\t")
                    || after_hash.chars().next().is_some_and(|c| c.is_ascii_digit())
            } else {
                false
            };
            if is_line_directive && !pending_line.is_empty() {
                if self.conditionals.is_active() {
                    let hash_col = line.find('#').map(|i| i + 2).unwrap_or(1);
                    self.process_directive(trimmed, source_line_num + 1, hash_col);
                }
                pending_newlines += 1;
                continue;
            }

            let process_directive = is_directive
                && (!is_include || pending_line.is_empty());

            if process_directive {
                // When accumulating a multi-line expression (pending_line is non-empty)
                // and a #define or #undef directive appears, we must expand macros in
                // the accumulated text BEFORE the directive modifies the macro table.
                // This ensures tokens like D(0) are expanded using the macro definition
                // that was active when those tokens were encountered, not the definition
                // (or lack thereof) after the directive. Per C standard, directives take
                // effect for tokens that follow them, not tokens that precede them.
                let output_len_before_directive = output.len();
                if !pending_line.is_empty() && !is_include {
                    let after_hash = trimmed[1..].trim_start();
                    if after_hash.starts_with("define") || after_hash.starts_with("undef") {
                        let expanded = self.macros.expand_line_reuse(&pending_line, &mut expanding);
                        pending_line.clear();
                        pending_line.push_str(&expanded);
                    } else if after_hash.starts_with("include") {
                        // When #include appears in the middle of a multi-line expression
                        // (e.g. a function call with args spanning lines), flush the
                        // pending tokens to output first. The included content must appear
                        // after the preceding tokens, not before them.
                        let expanded = self.macros.expand_line_reuse(&pending_line, &mut expanding);
                        output.push_str(&expanded);
                        output.push('\n');
                        for _ in 1..pending_newlines {
                            output.push('\n');
                        }
                        pending_line.clear();
                        pending_newlines = 0;
                    }
                }
                // Column of the directive keyword (1-based, after '#')
                let hash_col = line.find('#').map(|i| i + 2).unwrap_or(1);
                let include_result = self.process_directive(trimmed, source_line_num + 1, hash_col);
                if let Some(included_content) = include_result {
                    output.push_str(&included_content);
                    // After included content, emit a return-to-parent line marker
                    // so the source manager can map subsequent lines back to the
                    // correct file and line number. source_line_num is 0-based,
                    // and the next line of the parent file is source_line_num + 2
                    // (since the #include directive itself was source_line_num + 1).
                    let parent_file = self.include_stack.last()
                        .map(|p| super::includes::format_path_for_line_directive(p))
                        .unwrap_or_else(|| self.filename.clone());
                    // Flag 2 indicates returning from an include file (GCC convention)
                    let _ = writeln!(output, "# {} \"{}\" 2", source_line_num + 2, parent_file);
                } else if is_include {
                    // Included files always emit a newline for non-include directives
                    output.push('\n');
                }
                // Top-level: emit injected declarations from #include processing
                if !is_include && !self.pending_injections.is_empty() {
                    for decl in std::mem::take(&mut self.pending_injections) {
                        output.push_str(&decl);
                    }
                }
                // Preserve line numbering during multi-line accumulation
                if !is_include {
                    if !pending_line.is_empty() {
                        pending_newlines += 1;
                    } else {
                        output.push('\n');
                    }
                    // Update pp_output_line for directive path
                    let added = &output.as_bytes()[output_len_before_directive..];
                    pp_output_line += added.iter().filter(|&&b| b == b'\n').count() as u32;
                }
            } else if self.conditionals.is_active() {
                // Regular line (or directive during include with pending line) -
                // expand macros, handling multi-line macro invocations
                let output_len_before = output.len();
                self.accumulate_and_expand(
                    line, &mut pending_line, &mut pending_newlines, &mut output,
                    &mut expanding,
                );
                // Collect macro expansion info for diagnostics.
                // If expansion added content to the output, check if macros were used.
                if !is_include && output.len() > output_len_before {
                    let expanded_names = self.macros.take_expanded_macros();
                    if !expanded_names.is_empty() {
                        let unique_names = dedup_macro_names(expanded_names);
                        self.macro_expansion_info.push(
                            crate::common::source::MacroExpansionInfo {
                                pp_line: pp_output_line,
                                macro_names: unique_names,
                            }
                        );
                    }
                }
                // Update pp_output_line by counting newlines in the newly added content
                if !is_include {
                    let added = &output.as_bytes()[output_len_before..];
                    pp_output_line += added.iter().filter(|&&b| b == b'\n').count() as u32;
                }
            } else {
                // Inactive conditional block - skip the line but preserve numbering
                if !pending_line.is_empty() {
                    pending_newlines += 1;
                } else {
                    output.push('\n');
                    if !is_include {
                        pp_output_line += 1;
                    }
                }
            }
        }

        // Flush any remaining pending line
        if !pending_line.is_empty() {
            let expanded = self.macros.expand_line_reuse(&pending_line, &mut expanding);
            output.push_str(&expanded);
            output.push('\n');
            // Collect macro expansion info for the flushed line
            if !is_include {
                let expanded_names = self.macros.take_expanded_macros();
                if !expanded_names.is_empty() {
                    let unique_names = dedup_macro_names(expanded_names);
                    self.macro_expansion_info.push(
                        crate::common::source::MacroExpansionInfo {
                            pp_line: pp_output_line,
                            macro_names: unique_names,
                        }
                    );
                }
            }
        }

        // Disable macro expansion tracking
        if !is_include {
            self.macros.set_track_expansions(false);
        }

        // Restore conditional stack and line override for included files
        if let Some(saved) = saved_conditionals {
            self.conditionals = saved;
        }
        if is_include {
            self.line_override = saved_line_override;
        }

        output
    }

    /// Accumulate lines for multi-line macro invocations (unbalanced parens)
    /// and expand when complete. This is the shared logic for both preprocess
    /// paths, avoiding the previous duplication of ~40 lines.
    ///
    /// The `expanding` parameter is a reusable FxHashSet that avoids per-line
    /// allocation for macro expansion tracking.
    fn accumulate_and_expand(
        &self,
        line: &str,
        pending_line: &mut String,
        pending_newlines: &mut usize,
        output: &mut String,
        expanding: &mut crate::common::fx_hash::FxHashSet<Rc<str>>,
    ) {
        if pending_line.is_empty() {
            if Self::has_unbalanced_parens(line) {
                *pending_line = line.to_string();
                *pending_newlines = 1;
            } else if self.ends_with_funclike_macro(line) {
                // Line ends with a function-like macro name without '(' on same line.
                // Per C standard, whitespace (including newlines) between macro name
                // and '(' is allowed, so accumulate to check next line for '('.
                *pending_line = line.to_string();
                *pending_newlines = 1;
            } else {
                let expanded = self.macros.expand_line_reuse(line, expanding);
                // After expansion, check if the expanded result ends with a
                // function-like macro name. This handles chained macros like:
                //   #define STEP1(x) x STEP2
                //   #define STEP2(y) y STEP3
                //   STEP1(int)    <- expands to "int STEP2"
                //   (foo)         <- should be STEP2's argument
                // The original source line doesn't end with a func-like macro,
                // but the expanded result does. We need to accumulate the next
                // line so STEP2 can find its '(' argument.
                if self.ends_with_funclike_macro(&expanded) {
                    *pending_line = line.to_string();
                    *pending_newlines = 1;
                } else {
                    output.push_str(&expanded);
                    output.push('\n');
                }
            }
        } else {
            // Check if this continuation line starts with '(' (after whitespace)
            // when we were accumulating for a trailing function-like macro name.
            let needs_more = Self::has_unbalanced_parens(pending_line);
            pending_line.push('\n');
            pending_line.push_str(line);
            *pending_newlines += 1;

            if needs_more {
                // Was accumulating for unbalanced parens
                if !Self::has_unbalanced_parens(pending_line) || *pending_newlines > MAX_PENDING_NEWLINES {
                    let expanded = self.macros.expand_line_reuse(pending_line, expanding);
                    // Check if the expanded result ends with a function-like macro
                    // that needs args from the next line (chained macros).
                    if self.ends_with_funclike_macro(&expanded) && *pending_newlines <= MAX_PENDING_NEWLINES {
                        // Don't clear pending_line - keep accumulating
                    } else {
                        output.push_str(&expanded);
                        output.push('\n');
                        for _ in 1..*pending_newlines {
                            output.push('\n');
                        }
                        pending_line.clear();
                        *pending_newlines = 0;
                    }
                }
            } else {
                // Was accumulating for trailing function-like macro name.
                // Now we have the next line joined. Check if parens are balanced.
                if Self::has_unbalanced_parens(pending_line) && *pending_newlines <= MAX_PENDING_NEWLINES {
                    // The joined text has unbalanced parens (macro args span more lines)
                    // Keep accumulating.
                } else {
                    // Parens balanced or next line didn't start with '(' - expand now.
                    let expanded = self.macros.expand_line_reuse(pending_line, expanding);
                    // Check if the expanded result itself ends with a function-like
                    // macro name that needs args from the next line (chained macros).
                    if self.ends_with_funclike_macro(&expanded) && *pending_newlines <= MAX_PENDING_NEWLINES {
                        // Don't clear pending_line - keep accumulating
                    } else {
                        output.push_str(&expanded);
                        output.push('\n');
                        for _ in 1..*pending_newlines {
                            output.push('\n');
                        }
                        pending_line.clear();
                        *pending_newlines = 0;
                    }
                }
            }
        }
    }

    /// Check if a line ends with an identifier that is a defined function-like macro.
    /// This is used to detect cases where the macro arguments '(' might be on the next line.
    fn ends_with_funclike_macro(&self, line: &str) -> bool {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            return false;
        }
        // Extract the last identifier from the line
        let bytes = trimmed.as_bytes();
        let end = bytes.len();
        // Walk backwards to find end of last identifier
        if !is_ident_cont(bytes[end - 1] as char) {
            return false;
        }
        let mut start = end - 1;
        while start > 0 && is_ident_cont(bytes[start - 1] as char) {
            start -= 1;
        }
        // Check that the identifier starts with a valid start character
        if !is_ident_start(bytes[start] as char) {
            return false;
        }
        let ident = &trimmed[start..end];
        // Check if this identifier is a defined function-like macro
        if let Some(mac) = self.macros.get(ident) {
            mac.is_function_like
        } else {
            false
        }
    }

    /// Enable assembly preprocessing mode. In this mode, '$' is not treated
    /// as an identifier character during macro expansion, so that AT&T assembly
    /// immediates like `$MACRO_NAME` correctly expand the macro.
    pub fn set_asm_mode(&mut self, asm_mode: bool) {
        self.macros.asm_mode = asm_mode;
    }

    /// Set the filename for __FILE__ and __BASE_FILE__ macros and set as the base include directory.
    pub fn set_filename(&mut self, filename: &str) {
        self.filename = filename.to_string();
        self.macros.set_file(format!("\"{}\"", filename));
        // __BASE_FILE__ always expands to the main input file name,
        // unlike __FILE__ which changes during #include processing.
        self.macros.define(MacroDef {
            name: Rc::from("__BASE_FILE__"),
            is_function_like: false,
            params: Vec::new(),
            is_variadic: false,
            has_named_variadic: false,
            body: Rc::from(format!("\"{}\"", filename).as_str()),
        });
        // Push the file path onto the include stack for relative includes.
        // Use make_absolute (not canonicalize) to preserve symlinks, matching GCC
        // behavior: #include "..." searches relative to the directory where the
        // including file was found (through symlinks), not the symlink target.
        let path = PathBuf::from(filename);
        let abs = super::includes::make_absolute(&path);
        self.include_stack.push(abs);
    }

    /// Get preprocessing errors.
    pub fn errors(&self) -> &[PreprocessorDiagnostic] {
        &self.errors
    }

    /// Get preprocessing warnings.
    pub fn warnings(&self) -> &[PreprocessorDiagnostic] {
        &self.warnings
    }

    /// Get the current source file name for diagnostic purposes.
    /// Returns the top of the include stack (the file currently being preprocessed),
    /// falling back to `self.filename` for the main translation unit.
    pub(super) fn current_file(&self) -> String {
        self.include_stack.last()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| self.filename.clone())
    }

    /// Take macro expansion metadata collected during preprocessing.
    /// This metadata maps preprocessed output line numbers to the macros
    /// that were expanded on each line, for use in diagnostic rendering.
    pub fn take_macro_expansion_info(&mut self) -> Vec<crate::common::source::MacroExpansionInfo> {
        std::mem::take(&mut self.macro_expansion_info)
    }

    /// Dump all macro definitions in GCC `-dM` format.
    ///
    /// Returns a string with one `#define` per line for every currently-defined
    /// macro, sorted alphabetically. Function-like macros include their parameter
    /// lists. This output is used by build systems like Meson to detect compiler
    /// features via preprocessor defines (e.g., `__GNUC__`, `__GNUC_MINOR__`).
    pub fn dump_defines(&self) -> String {
        let mut defs: Vec<String> = self.macros.iter().map(|def| {
            if def.is_function_like {
                let params = def.params.join(",");
                if def.body.is_empty() {
                    format!("#define {}({})", def.name, params)
                } else {
                    format!("#define {}({}) {}", def.name, params, def.body)
                }
            } else if def.body.is_empty() {
                format!("#define {}", def.name)
            } else {
                format!("#define {} {}", def.name, def.body)
            }
        }).collect();
        defs.sort();
        defs.join("\n")
    }

    /// Define a macro from a command-line -D flag.
    /// Takes a name and value (e.g., name="FOO", value="1").
    pub fn define_macro(&mut self, name: &str, value: &str) {
        self.macros.define(MacroDef {
            name: Rc::from(name),
            is_function_like: false,
            params: Vec::new(),
            is_variadic: false,
            has_named_variadic: false,
            body: Rc::from(value),
        });
    }

    /// Undefine a macro by name.
    pub fn undefine_macro(&mut self, name: &str) {
        self.macros.undefine(name);
    }

    /// Add an include search path for #include directives (-I flag).
    /// Adds regardless of whether the directory currently exists.
    pub fn add_include_path(&mut self, path: &str) {
        self.include_paths.push(PathBuf::from(path));
    }

    /// Add a quote-only include search path (-iquote flag).
    /// These paths are searched only for `#include "file"`, not `#include <file>`.
    /// Searched after current directory, before -I paths.
    pub fn add_quote_include_path(&mut self, path: &str) {
        self.quote_include_paths.push(PathBuf::from(path));
    }

    /// Add a system include path (-isystem flag).
    /// These paths are searched after -I paths, before default system paths.
    pub fn add_system_include_path(&mut self, path: &str) {
        self.isystem_include_paths.push(PathBuf::from(path));
    }

    /// Add an after include path (-idirafter flag).
    /// These paths are searched last, after all other include paths.
    pub fn add_after_include_path(&mut self, path: &str) {
        self.after_include_paths.push(PathBuf::from(path));
    }

    /// Process a force-included file (-include flag). This preprocesses the file content
    /// as if it were #include'd at the very beginning of the main source file.
    /// All #define directives in the file take effect, and the preprocessed output
    /// is discarded (macros/typedefs persist in the preprocessor state).
    pub fn preprocess_force_include(&mut self, content: &str, resolved_path: &str) {
        let resolved = PathBuf::from(resolved_path);

        // Check for #pragma once
        if self.pragma_once_files.contains(&resolved) {
            return;
        }

        // Check for include guard
        if let Some(guard) = self.include_guard_macros.get(&resolved) {
            if self.macros.is_defined(guard) {
                return;
            }
        }

        // Push onto include stack
        self.include_stack.push(resolved.clone());

        // Save and set __FILE__ (uses set_file to avoid full MacroDef allocation)
        let old_file = self.macros.get_file_body().map(|s| s.to_string());
        self.macros.set_file(format!("\"{}\"", resolved.display()));

        // Preprocess the included content (macros persist; any pragma synthetic tokens
        // like __ccc_visibility_push_hidden are collected and prepended to main output)
        let output = self.preprocess_included(content);
        // Collect any non-whitespace output (e.g., pragma synthetic tokens) for prepending
        // to the main source's preprocessed output. This ensures that pragmas like
        // #pragma GCC visibility push(hidden) in force-included files take effect.
        let trimmed = output.trim();
        if !trimmed.is_empty() {
            self.force_include_output.push_str(trimmed);
            self.force_include_output.push('\n');
        }

        // Restore __FILE__
        if let Some(old) = old_file {
            self.macros.set_file(old);
        }

        // Pop include stack
        self.include_stack.pop();
    }


    /// Process a preprocessor directive line.
    /// Returns Some(content) if an #include was processed and should be inserted.
    /// `line_num` is the 1-based source line number, `col` is the 1-based column
    /// of the `#` character (used for diagnostic source locations).
    fn process_directive(&mut self, line: &str, line_num: usize, col: usize) -> Option<String> {
        // Strip leading # and whitespace
        let after_hash = line.trim_start_matches('#').trim();

        // Strip trailing comments (// style)
        let after_hash = strip_line_comment(after_hash);

        // Get directive keyword
        let (keyword, rest) = split_first_word(&after_hash);

        // Handle #include<file> and #include"file" (no space between include and path)
        // This handles the common C pattern: #include<stdio.h>
        let (keyword, rest) = if keyword.starts_with("include<") || keyword.starts_with("include\"") {
            ("include", &after_hash["include".len()..])
        } else if keyword.starts_with("include_next<") || keyword.starts_with("include_next\"") {
            ("include_next", &after_hash["include_next".len()..])
        } else {
            (keyword, rest)
        };

        // Some directives are processed even in inactive conditional blocks
        match keyword {
            "ifdef" | "ifndef" | "if" => {
                if !self.conditionals.is_active() {
                    // In an inactive block, just push a nested inactive conditional
                    self.conditionals.push_if(false);
                    return None;
                }
            }
            "elif" => {
                self.handle_elif(rest);
                return None;
            }
            "else" => {
                self.conditionals.handle_else();
                return None;
            }
            "endif" => {
                self.conditionals.handle_endif();
                return None;
            }
            _ => {
                // Other directives only processed in active blocks
                if !self.conditionals.is_active() {
                    return None;
                }
            }
        }

        // Process directive in active block
        match keyword {
            "include" => {
                return self.handle_include(rest, line_num, col);
            }
            "include_next" => {
                // GCC extension: include_next searches from the next path after the
                // current file's directory in the include search list
                return self.handle_include_next(rest, line_num, col);
            }
            "define" => self.handle_define(rest),
            "undef" => self.handle_undef(rest),
            "ifdef" => self.handle_ifdef(rest, false),
            "ifndef" => self.handle_ifdef(rest, true),
            "if" => self.handle_if(rest),
            "pragma" => {
                return self.handle_pragma(rest);
            }
            "error" => {
                // Expand macros in error message
                let expanded = self.macros.expand_line_reuse(rest, &mut self.directive_expanding);
                self.errors.push(PreprocessorDiagnostic {
                    file: self.current_file(),
                    line: line_num,
                    col,
                    message: format!("#error {}", expanded),
                });
            }
            "warning" => {
                // GCC extension, collect warning for structured diagnostic output
                self.warnings.push(PreprocessorDiagnostic {
                    file: self.current_file(),
                    line: line_num,
                    col,
                    message: format!("#warning {}", rest),
                });
            }
            "line" => {
                self.handle_line_directive(rest, line_num);
            }
            "" => {
                // Empty # directive (null directive), valid in C
            }
            _ => {
                // Handle GNU linemarker: # <digit-sequence> ["filename" [flags]]
                // This is equivalent to #line <digit-sequence> ["filename"]
                if keyword.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    let line_rest = format!("{} {}", keyword, rest);
                    self.handle_line_directive(&line_rest, line_num);
                }
                // Otherwise unknown directive, ignore silently
            }
        }

        None
    }

    fn handle_define(&mut self, rest: &str) {
        if let Some(def) = parse_define(rest) {
            self.macros.define(def);
        }
    }

    fn handle_undef(&mut self, rest: &str) {
        let name = rest.split_whitespace().next().unwrap_or("");
        if !name.is_empty() {
            self.macros.undefine(name);
        }
    }

    fn handle_ifdef(&mut self, rest: &str, negate: bool) {
        let name = rest.split_whitespace().next().unwrap_or("");
        let defined = self.macros.is_defined(name);
        let condition = if negate { !defined } else { defined };
        self.conditionals.push_if(condition);
    }

    fn handle_if(&mut self, expr: &str) {
        // First resolve `defined(X)` and `__has_*()` before macro expansion
        let resolved = self.resolve_defined_in_expr(expr);
        // Expand macros in the resolved expression (reuse directive_expanding set)
        let expanded = self.macros.expand_line_reuse(&resolved, &mut self.directive_expanding);
        // Resolve again after macro expansion, in case macros expanded to
        // __has_attribute(), __has_builtin(), __has_include(), etc.
        let expanded = self.resolve_defined_in_expr(&expanded);
        // Replace any remaining identifiers with 0 (standard C behavior for #if)
        let final_expr = self.replace_remaining_idents_with_zero(&expanded);
        let condition = evaluate_condition(&final_expr, &self.macros);
        self.conditionals.push_if(condition);
    }

    fn handle_elif(&mut self, expr: &str) {
        let resolved = self.resolve_defined_in_expr(expr);
        let expanded = self.macros.expand_line_reuse(&resolved, &mut self.directive_expanding);
        // Resolve again after macro expansion (same reason as handle_if)
        let expanded = self.resolve_defined_in_expr(&expanded);
        let final_expr = self.replace_remaining_idents_with_zero(&expanded);
        let condition = evaluate_condition(&final_expr, &self.macros);
        self.conditionals.handle_elif(condition);
    }

    fn handle_line_directive(&mut self, rest: &str, source_line_num: usize) {
        // #line digit-sequence ["filename"]
        // The argument undergoes macro expansion first
        let expanded = self.macros.expand_line_reuse(rest, &mut self.directive_expanding);
        let expanded = expanded.trim();

        // Parse the line number (first token)
        let mut parts = expanded.split_whitespace();
        if let Some(line_str) = parts.next() {
            if let Ok(target_line) = line_str.parse::<usize>() {
                // source_line_num is 1-based (the line where #line appears)
                // The line AFTER #line should be target_line, so we record:
                // (target_line, source_line_num_0based_of_directive)
                // source_line_num is already 1-based from process_directive caller
                self.line_override = Some((target_line, source_line_num));

                // If there's a filename argument, update __FILE__
                if let Some(filename_str) = parts.next() {
                    let filename = filename_str.trim_matches('"');
                    if !filename.is_empty() {
                        self.macros.set_file(format!("\"{}\"", filename));
                    }
                }
            }
        }
    }


}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}


