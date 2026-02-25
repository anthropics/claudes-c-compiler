//! Handles `#include` and `#include_next` directive resolution, file path lookup,
//! and synthetic header declaration injection.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::macro_defs::{MacroDef, parse_define};
use super::pipeline::Preprocessor;

/// Maximum recursive inclusion depth, matching GCC's default of 200.
/// Prevents infinite inclusion loops in files without `#pragma once`.
const MAX_INCLUDE_DEPTH: usize = 200;

/// Detect if a source file has a classic include guard pattern.
///
/// The pattern we detect is:
///   - First non-blank, non-comment directive is `#ifndef GUARD_MACRO`
///   - Second directive is `#define GUARD_MACRO` (same macro name, no value or any value)
///   - Last directive is `#endif`
///   - No code/tokens exist outside the `#ifndef`/`#endif` wrapper
///
/// Returns `Some(guard_macro_name)` if the pattern is detected, `None` otherwise.
///
/// This function operates on the raw source text (before preprocessing) and
/// uses a lightweight scan that doesn't require full lexing. It handles:
///   - C-style (`/* ... */`) and C++-style (`// ...`) comments
///   - Line continuations (`\` at end of line)
///   - Whitespace and blank lines before/after the guard
///
/// Intentionally conservative: returns None for anything unusual (e.g., code
/// before the `#ifndef`, `#else` branches, multiple `#endif`s at the same level).
fn detect_include_guard(source: &str) -> Option<String> {
    // We scan directive lines at the top level. We need to track:
    // 1. Whether we've seen #ifndef MACRO as the first directive
    // 2. Whether the next directive is #define MACRO
    // 3. Whether #endif is the last directive at depth 0
    // 4. That no non-whitespace content exists outside the guard

    let mut guard_macro: Option<String> = None;
    let mut found_ifndef = false;
    let mut found_define = false;
    let mut found_endif = false;
    let mut if_depth: i32 = 0;
    let mut has_content_before_guard = false;
    let mut has_content_after_endif = false;
    let mut in_block_comment = false;

    for raw_line in source.lines() {
        // Handle block comments that span lines
        let line = if in_block_comment {
            if let Some(end_pos) = raw_line.find("*/") {
                in_block_comment = false;
                &raw_line[end_pos + 2..]
            } else {
                continue;
            }
        } else {
            raw_line
        };

        // Strip block comments within the line (simple single-line handling)
        let line = strip_inline_comments(line, &mut in_block_comment);
        let trimmed = line.trim();

        // Skip empty lines
        if trimmed.is_empty() {
            continue;
        }

        // If we already saw the final #endif, any non-empty line means
        // there's content after the guard -> not a valid include guard
        if found_endif {
            has_content_after_endif = true;
            break;
        }

        if let Some(after_hash_raw) = trimmed.strip_prefix('#') {
            let after_hash = after_hash_raw.trim_start();

            // Handle line continuations in directives
            let after_hash = if after_hash.ends_with('\\') {
                // For guard detection, we only care about simple single-line directives.
                // Multi-line #define is fine as long as the macro name is on the first line.
                after_hash.trim_end_matches('\\').trim_end()
            } else {
                after_hash
            };

            if after_hash.starts_with("ifndef") && !found_ifndef && if_depth == 0 {
                // First directive must be #ifndef
                if has_content_before_guard {
                    return None;
                }
                let rest = after_hash["ifndef".len()..].trim();
                let macro_name = extract_identifier(rest)?;
                if macro_name.is_empty() {
                    return None;
                }
                guard_macro = Some(macro_name);
                found_ifndef = true;
                if_depth = 1;
            } else if !found_ifndef {
                // First directive is not #ifndef -> no guard
                return None;
            } else if !found_define && found_ifndef && if_depth == 1 {
                // Second directive should be #define with the same macro
                if after_hash.starts_with("define") {
                    let rest = after_hash.strip_prefix("define").unwrap().trim();
                    let macro_name = extract_identifier(rest)?;
                    if let Some(ref guard) = guard_macro {
                        if macro_name == *guard {
                            found_define = true;
                        } else {
                            return None; // #define of a different macro
                        }
                    }
                } else {
                    return None; // Second directive is not #define
                }
            } else {
                // Inside the guard body - track nesting depth
                if after_hash.starts_with("if") {
                    // Covers #if, #ifdef, #ifndef
                    if_depth += 1;
                } else if after_hash.starts_with("endif") {
                    if_depth -= 1;
                    if if_depth == 0 {
                        // This #endif closes the guard
                        found_endif = true;
                    }
                } else if (after_hash.starts_with("else") || after_hash.starts_with("elif")) && if_depth == 1 {
                    // An #else/#elif at the outermost guard level means the
                    // header has different behavior on re-inclusion (e.g.,
                    // libev's ev_wrap.h defines macros on first include and
                    // #undef's them on second include via the #else branch).
                    // Such headers must NOT be skipped on re-inclusion.
                    return None;
                }
                // Other directives (#define, #include, #elif, #else, etc.)
                // inside nested #if blocks are fine
            }
        } else {
            // Non-directive, non-empty line
            if !found_ifndef {
                // Content before the #ifndef -> no guard
                has_content_before_guard = true;
            }
            // Content inside the guard is fine
            // Content after the guard (found_endif) is handled at the top of the loop
        }
    }

    if found_ifndef && found_define && found_endif && !has_content_after_endif {
        guard_macro
    } else {
        None
    }
}

/// Extract a C identifier from the beginning of a string.
/// Returns Some(identifier) or None if no valid identifier starts the string.
fn extract_identifier(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() {
        return Some(String::new());
    }
    let bytes = s.as_bytes();
    if !super::utils::is_ident_start(bytes[0] as char) {
        return None;
    }
    let mut end = 1;
    while end < bytes.len() && super::utils::is_ident_cont(bytes[end] as char) {
        end += 1;
    }
    Some(s[..end].to_string())
}

/// Strip inline block comments from a single line.
/// Updates `in_block_comment` state for multi-line block comments.
/// Also strips line comments (// ...).
fn strip_inline_comments<'a>(line: &'a str, in_block_comment: &mut bool) -> std::borrow::Cow<'a, str> {
    // Fast path: no comment markers at all
    if !line.contains("/*") && !line.contains("//") {
        return std::borrow::Cow::Borrowed(line);
    }

    let mut result = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if *in_block_comment {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                *in_block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            *in_block_comment = true;
            i += 2;
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            // Line comment - ignore rest of line
            break;
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    std::borrow::Cow::Owned(result)
}

/// Read a C source file, tolerating non-UTF-8 content.
/// Valid UTF-8 files are returned as-is. Non-UTF-8 bytes are encoded as PUA
/// code points which the lexer decodes back to raw bytes.
fn read_c_source_file(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(crate::common::encoding::bytes_to_string(bytes))
}

/// Make a path absolute without resolving symlinks.
///
/// Unlike `std::fs::canonicalize`, this preserves symlinks in the path.
/// This is important for `#include "..."` resolution: GCC searches for
/// included files relative to the directory where the including file was
/// found (through symlinks), not relative to the symlink target's directory.
///
/// For example, if `build/local_scan.h -> ../src/local_scan.h` includes
/// `"config.h"`, we should search in `build/` (the symlink's directory),
/// not in `../src/` (the target's directory).
pub(super) fn make_absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        // Clean up . and .. components without resolving symlinks
        clean_path(path)
    } else if let Ok(cwd) = std::env::current_dir() {
        clean_path(&cwd.join(path))
    } else {
        path.to_path_buf()
    }
}

/// Format a path for use in `#line` directives.
///
/// GCC emits relative paths in `#line` directives when the file is under the
/// current working directory, and absolute paths otherwise. We replicate this
/// behavior because some build systems (notably Perl's `makedepend`) parse
/// `#line` directives and add compilation recipes for `.c` dependencies that
/// contain a `/` in their path. Using absolute paths would cause files like
/// `vutil.c` (which is `#include`d from `util.c`) to match the pattern and
/// generate a spurious inline recipe that overrides the correct `.c.o` rule.
pub(super) fn format_path_for_line_directive(path: &Path) -> String {
    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(relative) = path.strip_prefix(&cwd) {
            return relative.display().to_string();
        }
    }
    path.display().to_string()
}

/// Clean a path by resolving `.` and `..` components without following symlinks.
fn clean_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => { /* skip */ }
            std::path::Component::ParentDir => {
                result.pop();
            }
            other => {
                result.push(other);
            }
        }
    }
    result
}

/// Normalize a macro-expanded include path by removing spaces inserted during
/// expansion.
///
/// During macro expansion, the preprocessor may insert spaces between tokens for
/// two reasons:
/// 1. Anti-paste spaces to prevent "//" from being lexed as a comment start
///    (see `would_paste_tokens` in macro_defs.rs).
/// 2. Spaces preserved from the original macro body between token positions.
///
/// Both corrupt file paths in computed `#include` directives. For example:
///
///   #define incdir tests/
///   #define funnyname 42test.h
///   #define incname < incdir funnyname >
///   #include incname
///
/// produces "tests/ 42test.h" instead of "tests/42test.h". Removing all spaces
/// fixes this. This should only be called for macro-expanded paths; direct
/// include paths (e.g., `#include "path with spaces/foo.h"`) are not affected.
fn normalize_include_path(path: String) -> String {
    if path.contains(' ') {
        path.replace(' ', "")
    } else {
        path
    }
}

impl Preprocessor {
    /// Handle #include directive. Returns the preprocessed content of the included file,
    /// or None if the include couldn't be resolved (falls back to old behavior).
    /// `line_num` is the 1-based source line and `col` is the 1-based column of `#`
    /// for diagnostics.
    pub(super) fn handle_include(&mut self, path: &str, line_num: usize, col: usize) -> Option<String> {
        let path = path.trim();

        // Expand macros in include path (for computed includes)
        let (path, was_macro_expanded) = if !path.starts_with('<') && !path.starts_with('"') {
            (self.macros.expand_line(path), true)
        } else {
            (path.to_string(), false)
        };
        let path = path.trim();

        let (include_path, is_system) = if path.starts_with('<') {
            let end = path.find('>').unwrap_or(path.len());
            (path[1..end].to_string(), true)
        } else if let Some(rest) = path.strip_prefix('"') {
            let end = rest.find('"').unwrap_or(rest.len());
            (rest[..end].to_string(), false)
        } else {
            (path.to_string(), false)
        };

        // Only normalize paths that were produced by macro expansion, since
        // those may have spurious spaces from token separation. Direct paths
        // (e.g., #include "My Headers/foo.h") should be left as-is.
        let include_path = if was_macro_expanded {
            normalize_include_path(include_path)
        } else {
            include_path
        };

        self.includes.push(include_path.clone());

        // Always inject compiler-builtin macros for well-known headers
        // (e.g., stdarg.h's va_start/va_end, stdbool.h's true/false).
        // These use compiler builtins and should always be available.
        self.inject_builtin_macros_for_header(&include_path);

        if !self.resolve_includes {
            // When not resolving includes, inject all fallback declarations
            // since we won't have any real headers to provide them.
            self.inject_fallback_declarations_for_header(&include_path);
            return None;
        }

        // Resolve the include path to an actual file
        if let Some(resolved_path) = self.resolve_include_path(&include_path, is_system) {
            // Check for #pragma once
            if self.pragma_once_files.contains(&resolved_path) {
                return Some(String::new());
            }

            // Check for include guard: if this file has a known guard macro and
            // that macro is still defined, skip re-processing entirely.
            if let Some(guard) = self.include_guard_macros.get(&resolved_path) {
                if self.macros.is_defined(guard) {
                    return Some(String::new());
                }
            }

            // Check for excessive recursive inclusion.
            // Files WITHOUT #pragma once are allowed to be re-included with different
            // macro definitions active (e.g., TCC's x86_64-gen.c includes itself via
            // tcc.h with TARGET_DEFS_ONLY defined). Only block when nesting is excessive.
            {
                let depth = self.include_stack.iter().filter(|p| *p == &resolved_path).count();
                if depth >= MAX_INCLUDE_DEPTH {
                    return Some(String::new());
                }
            }

            // Read the file
            match read_c_source_file(&resolved_path) {
                Ok(content) => {
                    // Detect include guard pattern in the raw source before preprocessing.
                    // We do this before preprocessing so we're analyzing the original
                    // structure of the file, not the expanded output.
                    let detected_guard = detect_include_guard(&content);

                    // Push onto include stack
                    self.include_stack.push(resolved_path.clone());

                    // Format path for line directive (relative when under cwd, matching GCC)
                    let display_path = format_path_for_line_directive(&resolved_path);

                    // Update __FILE__ (uses set_file to avoid full MacroDef allocation)
                    let old_file = self.macros.get_file_body().map(|s| s.to_string());
                    self.macros.set_file(format!("\"{}\"", display_path));

                    // Emit line marker for entering the included file
                    // Flag 1 indicates entering a new include file (GCC convention)
                    let mut result = format!("# 1 \"{}\" 1\n", display_path);

                    // Preprocess the included content
                    result.push_str(&self.preprocess_included(&content));

                    // Restore __FILE__
                    if let Some(old) = old_file {
                        self.macros.set_file(old);
                    }

                    // Pop include stack
                    self.include_stack.pop();

                    // Register the include guard for future fast-path skipping.
                    // We do this after preprocessing so that the guard macro is
                    // now defined (the #define inside the file was processed).
                    if let Some(guard) = detected_guard {
                        self.include_guard_macros.insert(resolved_path, Rc::from(guard));
                    }

                    Some(result)
                }
                Err(_) => {
                    // Silently skip unresolvable includes (many system headers
                    // may not be needed if builtins provide their macros).
                    // Inject fallback declarations since the real header failed to load.
                    self.inject_fallback_declarations_for_header(&include_path);
                    None
                }
            }
        } else {
            // Header not found - inject fallback type/extern declarations for
            // well-known standard headers so compilation can still proceed.
            self.inject_fallback_declarations_for_header(&include_path);

            // Emit a fatal error with source location.
            // Compute approximate column of the include path for better diagnostics.
            // The include path token is at col (of '#') + len("include") + whitespace.
            let include_path_col = col + "include ".len();
            self.errors.push(super::pipeline::PreprocessorDiagnostic {
                file: self.current_file(),
                line: line_num,
                col: include_path_col,
                message: format!("{}: No such file or directory", include_path),
            });
            None
        }
    }

    /// Handle #include_next directive (GCC extension).
    /// Searches for the header starting from the next include path after the one
    /// that contained the current file.
    /// `line_num` is the 1-based source line and `col` is the 1-based column of `#`
    /// for diagnostics.
    pub(super) fn handle_include_next(&mut self, path: &str, line_num: usize, col: usize) -> Option<String> {
        let path = path.trim();

        // Parse the include path
        let (include_path, _is_system, was_macro_expanded) = if path.starts_with('<') {
            let end = path.find('>').unwrap_or(path.len());
            (path[1..end].to_string(), true, false)
        } else if let Some(rest) = path.strip_prefix('"') {
            let end = rest.find('"').unwrap_or(rest.len());
            (rest[..end].to_string(), false, false)
        } else {
            // Try macro expansion
            let expanded = self.macros.expand_line(path);
            let expanded = expanded.trim().to_string();
            if expanded.starts_with('<') {
                let end = expanded.find('>').unwrap_or(expanded.len());
                (expanded[1..end].to_string(), true, true)
            } else if let Some(rest) = expanded.strip_prefix('"') {
                let end = rest.find('"').unwrap_or(rest.len());
                (rest[..end].to_string(), false, true)
            } else {
                (expanded, false, true)
            }
        };

        let include_path = if was_macro_expanded {
            normalize_include_path(include_path)
        } else {
            include_path
        };

        if !self.resolve_includes {
            return None;
        }

        // Get the current file path for include_next resolution
        let current_file = self.include_stack.last().cloned();

        // Resolve using include_next semantics
        if let Some(resolved_path) = self.resolve_include_next_path(&include_path, current_file.as_ref()) {
            // Check for #pragma once
            if self.pragma_once_files.contains(&resolved_path) {
                return Some(String::new());
            }

            // Check for include guard
            if let Some(guard) = self.include_guard_macros.get(&resolved_path) {
                if self.macros.is_defined(guard) {
                    return Some(String::new());
                }
            }

            // Check for excessive recursive inclusion
            {
                let depth = self.include_stack.iter().filter(|p| *p == &resolved_path).count();
                if depth >= MAX_INCLUDE_DEPTH {
                    return Some(String::new());
                }
            }

            // Read and preprocess the file
            match read_c_source_file(&resolved_path) {
                Ok(content) => {
                    let detected_guard = detect_include_guard(&content);

                    self.include_stack.push(resolved_path.clone());

                    // Format path for line directive (relative when under cwd, matching GCC)
                    let display_path = format_path_for_line_directive(&resolved_path);

                    let old_file = self.macros.get_file_body().map(|s| s.to_string());
                    self.macros.set_file(format!("\"{}\"", display_path));

                    // Emit line marker for entering the included file
                    // Flag 1 indicates entering a new include file (GCC convention)
                    let mut result = format!("# 1 \"{}\" 1\n", display_path);

                    result.push_str(&self.preprocess_included(&content));

                    if let Some(old) = old_file {
                        self.macros.set_file(old);
                    }

                    self.include_stack.pop();

                    if let Some(guard) = detected_guard {
                        self.include_guard_macros.insert(resolved_path, Rc::from(guard));
                    }

                    Some(result)
                }
                Err(_) => None,
            }
        } else {
            // Fall back to regular include if include_next can't find it
            self.handle_include(path, line_num, col)
        }
    }

    /// Resolve an include path using #include_next semantics: search from the
    /// next include path after the one containing the current file.
    /// `current_file` is the full path to the file containing the #include_next.
    pub(super) fn resolve_include_next_path(&self, include_path: &str, current_file: Option<&PathBuf>) -> Option<PathBuf> {
        // Collect all search paths in order:
        // -iquote -> -I -> -isystem -> default system -> -idirafter
        let all_paths: Vec<&Path> = self.quote_include_paths.iter()
            .chain(self.include_paths.iter())
            .chain(self.isystem_include_paths.iter())
            .chain(self.system_include_paths.iter())
            .chain(self.after_include_paths.iter())
            .map(|p| p.as_path())
            .collect();

        // Canonicalize the current file path for comparison
        let current_file_canon = current_file.and_then(|f| std::fs::canonicalize(f).ok());

        // Find which search path contains the current file by checking if
        // search_path/include_path resolves to the same file as the current file.
        // This correctly handles subdirectory includes (e.g., sys/types.h).
        let mut found_current = false;
        if let Some(ref cur_canon) = current_file_canon {
            for search_path in &all_paths {
                let candidate = search_path.join(include_path);
                if candidate.is_file() {
                    if let Ok(candidate_canon) = std::fs::canonicalize(&candidate) {
                        if &candidate_canon == cur_canon {
                            found_current = true;
                            continue;
                        }
                    }
                }
                if found_current {
                    let candidate = search_path.join(include_path);
                    if candidate.is_file() {
                        return Some(make_absolute(&candidate));
                    }
                }
            }
        }

        // Fallback: if we couldn't find the current file in any search path,
        // search all paths but skip any that resolve to the current file.
        if !found_current {
            for search_path in &all_paths {
                let candidate = search_path.join(include_path);
                if candidate.is_file() {
                    // Use canonicalize for comparison to detect same-file
                    let candidate_canon = std::fs::canonicalize(&candidate).ok();
                    if let (Some(ref cur), Some(ref cand)) = (&current_file_canon, &candidate_canon) {
                        if cur == cand {
                            continue;
                        }
                    }
                    return Some(make_absolute(&candidate));
                }
            }
        }

        None
    }

    /// Resolve an include path to an actual file path.
    /// For "file.h": search current dir, then -I paths, then system paths.
    /// For <file.h>: search -I paths, then system paths.
    ///
    /// Returns the path WITHOUT resolving symlinks, matching GCC behavior.
    /// This ensures that `#include "..."` searches relative to the directory
    /// where the including file was found (through symlinks), not the target.
    ///
    /// Uses a cache to avoid repeated filesystem probing for the same include
    /// path from the same context. The cache key includes the current directory
    /// for quoted includes (since resolution depends on it).
    pub fn resolve_include_path(&mut self, include_path: &str, is_system: bool) -> Option<PathBuf> {
        // Compute cache key: (include_path, is_system, current_dir_for_quoted_includes)
        let current_dir_key = if !is_system {
            self.include_stack.last()
                .and_then(|f| f.parent().map(|p| p.to_path_buf()))
                .unwrap_or_default()
        } else {
            PathBuf::new()
        };
        let cache_key = (include_path.to_string(), is_system, current_dir_key);

        if let Some(cached) = self.include_resolve_cache.get(&cache_key) {
            return cached.clone();
        }

        let result = self.resolve_include_path_uncached(include_path, is_system);
        self.include_resolve_cache.insert(cache_key, result.clone());
        result
    }

    /// Uncached include path resolution. Called by `resolve_include_path` on cache miss.
    fn resolve_include_path_uncached(&self, include_path: &str, is_system: bool) -> Option<PathBuf> {
        // For quoted includes (#include "..."), search in this order:
        //   1. Current file's directory
        //   2. -iquote paths
        //   3. -I paths
        //   4. -isystem paths
        //   5. Default system paths
        //   6. -idirafter paths
        //
        // For system includes (#include <...>), search in this order:
        //   1. -I paths
        //   2. -isystem paths
        //   3. Default system paths
        //   4. -idirafter paths

        if !is_system {
            // Step 1: Search relative to the current file's directory
            if let Some(current_file) = self.include_stack.last() {
                if let Some(current_dir) = current_file.parent() {
                    let candidate = current_dir.join(include_path);
                    if candidate.is_file() {
                        return Some(make_absolute(&candidate));
                    }
                }
            }
            // Also try relative to the original source file directory
            if !self.filename.is_empty() {
                if let Some(parent) = Path::new(&self.filename).parent() {
                    let candidate = parent.join(include_path);
                    if candidate.is_file() {
                        return Some(make_absolute(&candidate));
                    }
                }
            }

            // Step 2: Search -iquote paths (quoted includes only)
            for dir in &self.quote_include_paths {
                let candidate = dir.join(include_path);
                if candidate.is_file() {
                    return Some(make_absolute(&candidate));
                }
            }
        }

        // Step 3: Search -I paths
        for dir in &self.include_paths {
            let candidate = dir.join(include_path);
            if candidate.is_file() {
                return Some(make_absolute(&candidate));
            }
        }

        // Step 4: Search -isystem paths
        for dir in &self.isystem_include_paths {
            let candidate = dir.join(include_path);
            if candidate.is_file() {
                return Some(make_absolute(&candidate));
            }
        }

        // Step 5: Search default system include paths
        for dir in &self.system_include_paths {
            let candidate = dir.join(include_path);
            if candidate.is_file() {
                return Some(make_absolute(&candidate));
            }
        }

        // Step 6: Search -idirafter paths
        for dir in &self.after_include_paths {
            let candidate = dir.join(include_path);
            if candidate.is_file() {
                return Some(make_absolute(&candidate));
            }
        }

        None
    }

    /// Inject compiler-builtin macros for well-known standard headers.
    /// These are always injected regardless of whether the real header is found,
    /// because they expand to compiler builtins (`__builtin_*`) that only our
    /// compiler understands.
    fn inject_builtin_macros_for_header(&mut self, header: &str) {
        match header {
            "stdbool.h" => {
                // Define true/false macros only when stdbool.h is explicitly included
                crate::frontend::preprocessor::builtin_macros::define_stdbool_true_false(&mut self.macros);
            }
            "complex.h" => {
                // C99 <complex.h> support - define standard complex macros
                self.macros.define(parse_define("complex _Complex").expect("static define"));
                self.macros.define(parse_define("_Complex_I (__extension__ 1.0fi)").expect("static define"));
                self.macros.define(parse_define("I _Complex_I").expect("static define"));
                self.macros.define(parse_define("__STDC_IEC_559_COMPLEX__ 1").expect("static define"));
            }
            "stdarg.h" => {
                // Define va_start/va_arg/va_end/va_copy as macros expanding to builtins.
                // These are always needed because they expand to __builtin_* forms.
                // Note: va_list / __gnuc_va_list / __builtin_va_list typedefs are NOT
                // injected as text here. The parser pre-registers them as type names
                // (parse.rs), sema seeds them in type_context.rs, and the IR lowerer
                // provides the target-specific ABI types (types_seed.rs). Injecting
                // typedef text via pending_injections would break if stdarg.h is included
                // from within a nested header, since the injected text gets emitted at
                // the include boundary -- potentially in the middle of an initializer.
                self.macros.define(MacroDef {
                    name: Rc::from("va_start"),
                    is_function_like: true,
                    params: vec![Rc::from("ap"), Rc::from("last")],
                    is_variadic: false,
                    has_named_variadic: false,
                    body: Rc::from("__builtin_va_start(ap,last)"),
                });
                self.macros.define(MacroDef {
                    name: Rc::from("va_end"),
                    is_function_like: true,
                    params: vec![Rc::from("ap")],
                    is_variadic: false,
                    has_named_variadic: false,
                    body: Rc::from("__builtin_va_end(ap)"),
                });
                self.macros.define(MacroDef {
                    name: Rc::from("va_copy"),
                    is_function_like: true,
                    params: vec![Rc::from("dest"), Rc::from("src")],
                    is_variadic: false,
                    has_named_variadic: false,
                    body: Rc::from("__builtin_va_copy(dest,src)"),
                });
                // va_arg is special syntax: __builtin_va_arg(ap, type)
                // It's handled by the parser as a special built-in, so we define
                // the macro to expand to __builtin_va_arg which the lexer recognizes.
                self.macros.define(MacroDef {
                    name: Rc::from("va_arg"),
                    is_function_like: true,
                    params: vec![Rc::from("ap"), Rc::from("type")],
                    is_variadic: false,
                    has_named_variadic: false,
                    body: Rc::from("__builtin_va_arg(ap,type)"),
                });
                // __gnuc_va_list is also handled natively by the parser/sema/lowerer
                // (see comment above about not injecting typedef text).
            }
            _ => {}
        }
    }

    /// Inject fallback type definitions and extern declarations for standard
    /// headers. Only called when no real header file was found, to provide
    /// minimal definitions so compilation can proceed. When the project provides
    /// its own headers (e.g., dietlibc, musl), these are NOT injected to avoid
    /// conflicting type definitions.
    fn inject_fallback_declarations_for_header(&mut self, header: &str) {
        match header {
            "stdio.h" => {
                // FILE type and standard streams (fallback only)
                self.pending_injections.push("typedef struct _IO_FILE FILE;\n".to_string());
                self.pending_injections.push("extern FILE *stdin;\n".to_string());
                self.pending_injections.push("extern FILE *stdout;\n".to_string());
                self.pending_injections.push("extern FILE *stderr;\n".to_string());
            }
            "errno.h" => {
                // errno is typically a macro expanding to (*__errno_location())
                // but for our purposes, treat it as an extern int
                self.pending_injections.push("extern int errno;\n".to_string());
            }
            "complex.h" => {
                // Declare complex math functions (fallback only)
                self.pending_injections.push(concat!(
                    "double creal(double _Complex __z);\n",
                    "float crealf(float _Complex __z);\n",
                    "long double creall(long double _Complex __z);\n",
                    "double cimag(double _Complex __z);\n",
                    "float cimagf(float _Complex __z);\n",
                    "long double cimagl(long double _Complex __z);\n",
                    "double _Complex conj(double _Complex __z);\n",
                    "float _Complex conjf(float _Complex __z);\n",
                    "long double _Complex conjl(long double _Complex __z);\n",
                    "double cabs(double _Complex __z);\n",
                    "float cabsf(float _Complex __z);\n",
                    "double carg(double _Complex __z);\n",
                    "float cargf(float _Complex __z);\n",
                ).to_string());
            }
            _ => {}
        }
    }
}
