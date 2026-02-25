/// A byte-offset span in source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
    pub file_id: u32,
}

impl Span {
    pub fn new(start: u32, end: u32, file_id: u32) -> Self {
        Self { start, end, file_id }
    }

    pub fn dummy() -> Self {
        Self { start: 0, end: 0, file_id: 0 }
    }

}

/// A human-readable source location.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Entry in the line map: maps a byte offset in the preprocessed output
/// to an original filename and line number. Uses a filename index into the
/// deduplicated `line_map_filenames` table to avoid per-entry String allocation.
#[derive(Debug, Clone, Copy)]
struct LineMapEntry {
    /// Byte offset in preprocessed output where this mapping starts.
    pp_offset: u32,
    /// Index into SourceManager::line_map_filenames.
    filename_idx: u16,
    /// Original line number (1-based) at pp_offset.
    orig_line: u32,
}

/// Records where a file was included from: the parent file and the line number
/// of the `#include` directive. Used to render GCC-style "In file included from"
/// traces in error diagnostics.
#[derive(Debug, Clone)]
pub struct IncludeOrigin {
    /// The filename that contained the #include directive.
    pub file: String,
    /// The line number of the #include directive (1-based).
    pub line: u32,
}

/// Records that macros were expanded on a particular line of preprocessed output.
/// Used by the diagnostic engine to emit "in expansion of macro 'X'" notes.
#[derive(Debug, Clone)]
pub struct MacroExpansionInfo {
    /// Line number in the preprocessed output (0-based) where the expansion occurred.
    pub pp_line: u32,
    /// Names of macros that were expanded (outermost first).
    /// Only the first (outermost) macro is typically shown in diagnostics.
    pub macro_names: Vec<std::rc::Rc<str>>,
}

/// Manages source files and provides span-to-location resolution.
///
/// Supports two modes:
/// 1. Simple mode: a single file registered via `add_file()`, spans resolved
///    directly via byte-offset-to-line binary search.
/// 2. Line-map mode: preprocessed output with embedded `# linenum "filename"`
///    markers. The line map is built by `build_line_map()` and used to resolve
///    spans back to original source files and line numbers.
///
/// Also tracks include chains: when flag 1 appears in a line marker
/// (`# 1 "file.h" 1`), it records that file.h was included from the
/// previously active file at the line of the preceding marker. This enables
/// "In file included from X:Y:" diagnostic traces.
///
/// Also tracks macro expansion info: when the preprocessor records that a
/// line involved macro expansion, the diagnostic engine can query this to
/// emit "in expansion of macro 'X'" notes.
#[derive(Debug, Default)]
pub struct SourceManager {
    files: Vec<SourceFile>,
    /// Line map entries sorted by pp_offset. When non-empty, resolve_span uses
    /// this instead of per-file line_offsets.
    line_map: Vec<LineMapEntry>,
    /// Deduplicated filename strings referenced by LineMapEntry::filename_idx.
    /// Avoids allocating the same filename string for every line marker.
    line_map_filenames: Vec<String>,
    /// Include chain map: filename_idx -> IncludeOrigin.
    /// Records where each included file was included from.
    /// Only populated for files that have flag 1 (enter-include) markers.
    include_origins: Vec<Option<IncludeOrigin>>,
    /// Macro expansion info: maps preprocessed output line numbers to the
    /// macros that were expanded on that line. Sorted by pp_line for binary search.
    macro_expansions: Vec<MacroExpansionInfo>,
}

#[derive(Debug)]
struct SourceFile {
    name: String,
    content: String,
    line_offsets: Vec<u32>,
}

impl SourceManager {
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            line_map: Vec::new(),
            line_map_filenames: Vec::new(),
            include_origins: Vec::new(),
            macro_expansions: Vec::new(),
        }
    }

    pub fn add_file(&mut self, name: String, content: String) -> u32 {
        let line_offsets = compute_line_offsets(&content);
        let id = self.files.len() as u32;
        self.files.push(SourceFile { name, content, line_offsets });
        id
    }

    pub fn get_content(&self, file_id: u32) -> &str {
        &self.files[file_id as usize].content
    }

    /// Build a line map from GCC-style line markers in preprocessed output.
    ///
    /// Scans the stored file content (files[0]) for lines matching
    /// `# <number> "<filename>" [flags]`. These markers are emitted by the
    /// preprocessor at `#include` boundaries and indicate that subsequent lines
    /// originate from the named file starting at the given line number.
    ///
    /// GCC-style flags after the filename:
    /// - Flag `1`: entering a new include file
    /// - Flag `2`: returning to a file after an include
    ///
    /// When flag 1 is seen, this records the include origin (which file and line
    /// included the new file) for later use in "In file included from" traces.
    ///
    /// Reuses the line offsets already computed by `add_file()` for column
    /// calculation, avoiding a redundant scan of the preprocessed output.
    ///
    /// Must be called after `add_file()` has stored the preprocessed content.
    pub fn build_line_map(&mut self) {
        if self.files.is_empty() {
            return;
        }

        // Line offsets for column calculation are reused directly from
        // files[0].line_offsets (computed once during add_file), avoiding
        // both a redundant O(n) scan and a Vec clone.

        let bytes = self.files[0].content.as_bytes();
        let len = bytes.len();

        // Track the last filename (byte range) and its index to avoid redundant
        // lookups. Consecutive line markers usually reference the same file.
        let mut last_fname_start: usize = 0;
        let mut last_fname_end: usize = 0;
        let mut last_fname_idx: u16 = 0;
        use crate::common::fx_hash::FxHashMap;
        let mut fname_map: FxHashMap<&[u8], u16> = FxHashMap::default();

        // Track the current file and line for include chain building.
        // When we see a flag-1 marker (entering include), the current file/line
        // is the origin of the #include directive.
        let mut current_filename_idx: u16 = 0;
        let mut current_line: u32 = 1;
        // Byte offset of the line after the most recent line marker.
        // Used to count newlines between markers for accurate line tracking.
        let mut last_marker_next_offset: usize = 0;

        let mut i = 0;
        while i < len {
            // Find end of this line using fast newline search
            let line_end = if let Some(rel) = memchr_newline(&bytes[i..]) {
                i + rel
            } else {
                len
            };

            // Quick check: line markers start with '#' (possibly after whitespace).
            // Skip lines that don't start with '#' for fast rejection.
            let mut j = i;
            while j < line_end && bytes[j] == b' ' {
                j += 1;
            }

            if j < line_end && bytes[j] == b'#' {
                j += 1;
                // Skip whitespace after #
                while j < line_end && bytes[j] == b' ' {
                    j += 1;
                }
                // Parse line number directly from bytes (avoids from_utf8 + parse)
                let num_start = j;
                while j < line_end && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > num_start {
                    if let Some(line_num) = parse_u32_from_digits(&bytes[num_start..j]) {
                        // Skip whitespace
                        while j < line_end && bytes[j] == b' ' {
                            j += 1;
                        }
                        // Parse "filename"
                        if j < line_end && bytes[j] == b'"' {
                            j += 1;
                            let fname_start = j;
                            while j < line_end && bytes[j] != b'"' {
                                j += 1;
                            }
                            let fname_bytes = &bytes[fname_start..j];

                            // Deduplicate filenames: check last-used cache first
                            // (consecutive markers usually reference the same file),
                            // then fall back to the hash map for non-consecutive repeats.
                            let filename_idx =
                                if last_fname_end > last_fname_start
                                    && fname_bytes == &bytes[last_fname_start..last_fname_end]
                                {
                                    // Same filename as previous marker (common case)
                                    last_fname_idx
                                } else if let Some(&idx) = fname_map.get(fname_bytes) {
                                    // Previously seen filename
                                    idx
                                } else {
                                    // New unique filename - allocate once
                                    let s = std::str::from_utf8(fname_bytes)
                                        .unwrap_or("<unknown>")
                                        .to_string();
                                    let idx = self.line_map_filenames.len() as u16;
                                    self.line_map_filenames.push(s);
                                    fname_map.insert(fname_bytes, idx);
                                    idx
                                };
                            last_fname_start = fname_start;
                            last_fname_end = j;
                            last_fname_idx = filename_idx;

                            // Parse optional GCC-style flags after the closing quote
                            let mut has_flag_1 = false;
                            let mut k = j + 1; // skip closing quote
                            while k < line_end {
                                // Skip whitespace
                                while k < line_end && bytes[k] == b' ' {
                                    k += 1;
                                }
                                if k < line_end && bytes[k].is_ascii_digit() {
                                    let flag_start = k;
                                    while k < line_end && bytes[k].is_ascii_digit() {
                                        k += 1;
                                    }
                                    if let Some(flag) = parse_u32_from_digits(&bytes[flag_start..k]) {
                                        if flag == 1 {
                                            has_flag_1 = true;
                                        }
                                    }
                                } else {
                                    break;
                                }
                            }

                            // If flag 1 (entering include), record the include origin.
                            // The current file/line before this marker is where the
                            // #include directive appeared.
                            if has_flag_1 {
                                // Ensure include_origins vec is large enough
                                while self.include_origins.len() <= filename_idx as usize {
                                    self.include_origins.push(None);
                                }
                                // Only record the first include origin for each file
                                // (a header may be included multiple times but only the
                                // first inclusion chain matters for diagnostics).
                                if self.include_origins[filename_idx as usize].is_none() {
                                    let parent_file = self.line_map_filenames
                                        [current_filename_idx as usize]
                                        .clone();
                                    // Count newlines between the last marker and the
                                    // current line to get the actual line number of the
                                    // #include directive in the parent file.
                                    let newlines_since_marker = bytes
                                        [last_marker_next_offset..i]
                                        .iter()
                                        .filter(|&&b| b == b'\n')
                                        .count() as u32;
                                    let include_line = current_line + newlines_since_marker;
                                    self.include_origins[filename_idx as usize] =
                                        Some(IncludeOrigin {
                                            file: parent_file,
                                            line: include_line,
                                        });
                                }
                            }

                            // Update current file/line tracking
                            current_filename_idx = filename_idx;
                            current_line = line_num;

                            // The next line (after this marker) maps to filename:line_num.
                            // Record the byte offset of the line after the marker.
                            let next_line_offset = if line_end < len {
                                line_end + 1 // skip the '\n'
                            } else {
                                line_end
                            };

                            // Update last marker position for newline counting
                            last_marker_next_offset = next_line_offset;

                            self.line_map.push(LineMapEntry {
                                pp_offset: next_line_offset as u32,
                                filename_idx,
                                orig_line: line_num,
                            });
                        }
                    }
                }
            }

            // Advance past the newline
            i = if line_end < len { line_end + 1 } else { len };
        }
    }

    /// Resolve a span to a human-readable source location.
    ///
    /// When a line map is available (preprocessor emitted line markers),
    /// resolves through the line map to the original file and line number.
    /// Otherwise falls back to direct file-based resolution.
    pub fn resolve_span(&self, span: Span) -> SourceLocation {
        if !self.line_map.is_empty() {
            return self.resolve_via_line_map(span);
        }

        // Fallback: direct file-based resolution
        if (span.file_id as usize) >= self.files.len() {
            return SourceLocation {
                file: "<unknown>".to_string(),
                line: 0,
                column: 0,
            };
        }
        let file = &self.files[span.file_id as usize];
        let line = match file.line_offsets.binary_search(&span.start) {
            Ok(i) => i as u32,
            Err(i) => if i > 0 { (i - 1) as u32 } else { 0 },
        };
        let col = span.start.saturating_sub(file.line_offsets[line as usize]);
        SourceLocation {
            file: file.name.clone(),
            line: line + 1,
            column: col + 1,
        }
    }

    /// Resolve a span using the line map built from preprocessor line markers.
    /// Assumes files[0] contains the preprocessed output (set by the driver via add_file).
    fn resolve_via_line_map(&self, span: Span) -> SourceLocation {
        let offset = span.start;

        // Find the line map entry that covers this offset.
        // Binary search for the last entry with pp_offset <= offset.
        let idx = match self.line_map.binary_search_by_key(&offset, |e| e.pp_offset) {
            Ok(i) => i,
            Err(i) => if i > 0 { i - 1 } else { 0 },
        };

        let entry = &self.line_map[idx];

        // Count how many newlines are between entry.pp_offset and offset
        // to determine the line offset within this mapped region.
        let mut lines_past = 0u32;
        let entry_filename = &self.line_map_filenames[entry.filename_idx as usize];
        let file_content = if !self.files.is_empty() {
            self.files[0].content.as_bytes()
        } else {
            return SourceLocation {
                file: entry_filename.clone(),
                line: entry.orig_line,
                column: 1,
            };
        };

        let start = entry.pp_offset as usize;
        let end = offset as usize;
        if end <= file_content.len() && start <= end {
            for &b in &file_content[start..end] {
                if b == b'\n' {
                    lines_past += 1;
                }
            }
        }

        // Compute column: distance from the start of the current line.
        // Uses files[0].line_offsets (computed once during add_file) instead of
        // maintaining a separate pp_line_offsets copy.
        let line_offsets = &self.files[0].line_offsets;
        let col = if !line_offsets.is_empty() {
            let pp_line = match line_offsets.binary_search(&offset) {
                Ok(i) => i,
                Err(i) => if i > 0 { i - 1 } else { 0 },
            };
            offset.saturating_sub(line_offsets[pp_line]) + 1
        } else {
            1
        };

        SourceLocation {
            file: entry_filename.clone(),
            line: entry.orig_line + lines_past,
            column: col,
        }
    }

    /// Get the source line text for a given span (for error snippet display).
    /// Returns the full line containing the span start position.
    /// Assumes files[0] contains the preprocessed output (set by the driver via add_file).
    pub fn get_source_line(&self, span: Span) -> Option<String> {
        if self.files.is_empty() {
            return None;
        }
        let content = self.files[0].content.as_bytes();
        let offset = span.start as usize;
        if offset >= content.len() {
            return None;
        }

        // Find start of the line
        let mut line_start = offset;
        while line_start > 0 && content[line_start - 1] != b'\n' {
            line_start -= 1;
        }

        // Find end of the line
        let mut line_end = offset;
        while line_end < content.len() && content[line_end] != b'\n' {
            line_end += 1;
        }

        let line_bytes = &content[line_start..line_end];

        // Skip line markers (# <digit>... "filename" pattern), but not
        // other preprocessor directives like #define or #if which are valid
        // source lines that users may want to see in error snippets.
        if is_line_marker(line_bytes) {
            return None;
        }

        std::str::from_utf8(line_bytes).ok().map(|s| s.to_string())
    }

    /// Get the include chain for a file, from innermost to outermost.
    ///
    /// For a file included as: main.c -> header1.h -> header2.h,
    /// calling this with "header2.h" returns:
    ///   [IncludeOrigin("header1.h", line), IncludeOrigin("main.c", line)]
    ///
    /// Returns an empty vec for the main source file or files without
    /// include tracking information.
    pub fn get_include_chain(&self, filename: &str) -> Vec<IncludeOrigin> {
        let mut chain = Vec::new();

        // Find the filename index for the given filename
        let mut current_file = filename.to_string();

        // Walk up the include chain, following parent links
        // Limit depth to avoid infinite loops from malformed data
        for _ in 0..200 {
            // Find the filename index for current_file
            let idx = self.line_map_filenames.iter().position(|f| f == &current_file);
            let idx = match idx {
                Some(i) => i,
                None => break,
            };

            // Check if this file has an include origin
            if idx >= self.include_origins.len() {
                break;
            }
            match &self.include_origins[idx] {
                Some(origin) => {
                    chain.push(origin.clone());
                    current_file = origin.file.clone();
                }
                None => break,
            }
        }

        chain
    }

    /// Set macro expansion metadata collected by the preprocessor.
    /// Each entry maps a preprocessed output line to the macros expanded on it.
    /// Entries should be sorted by pp_line for efficient lookup.
    pub fn set_macro_expansions(&mut self, expansions: Vec<MacroExpansionInfo>) {
        self.macro_expansions = expansions;
    }

    /// Look up macro expansion info for a given span.
    /// Returns the list of macro names if the span falls on a line that had
    /// macro expansion, or None if the span is not in a macro expansion region.
    pub fn get_macro_expansion_at(&self, span: Span) -> Option<&[std::rc::Rc<str>]> {
        if self.macro_expansions.is_empty() || self.files.is_empty() {
            return None;
        }

        // Convert byte offset to preprocessed output line number
        let line_offsets = &self.files[0].line_offsets;
        let pp_line = match line_offsets.binary_search(&span.start) {
            Ok(i) => i as u32,
            Err(i) => if i > 0 { (i - 1) as u32 } else { 0 },
        };

        // Binary search for this pp_line in the macro expansions
        match self.macro_expansions.binary_search_by_key(&pp_line, |e| e.pp_line) {
            Ok(idx) => {
                let names = &self.macro_expansions[idx].macro_names;
                if names.is_empty() { None } else { Some(names) }
            }
            Err(_) => None,
        }
    }
}

/// Check if a line (as bytes) is a GCC-style line marker: # <digit>... "filename"
/// Returns true only for line markers, not for preprocessor directives like #define.
fn is_line_marker(line: &[u8]) -> bool {
    let mut i = 0;
    // Skip leading whitespace
    while i < line.len() && line[i] == b' ' {
        i += 1;
    }
    // Must start with '#'
    if i >= line.len() || line[i] != b'#' {
        return false;
    }
    i += 1;
    // Skip whitespace after '#'
    while i < line.len() && line[i] == b' ' {
        i += 1;
    }
    // Next character must be a digit (this distinguishes line markers from directives)
    i < line.len() && line[i].is_ascii_digit()
}

fn compute_line_offsets(content: &str) -> Vec<u32> {
    let bytes = content.as_bytes();
    let len = bytes.len();
    // Pre-allocate: estimate ~60 bytes per line (typical for C code)
    let mut offsets = Vec::with_capacity(len / 60 + 1);
    offsets.push(0u32);
    let mut pos = 0;
    while pos < len {
        // Use memchr-style scanning: check bytes in chunks for newlines.
        // This is faster than enumerate() because it avoids the tuple overhead
        // and lets the compiler vectorize the inner search.
        if let Some(rel) = memchr_newline(&bytes[pos..]) {
            offsets.push((pos + rel + 1) as u32);
            pos += rel + 1;
        } else {
            break;
        }
    }
    offsets
}

/// Fast newline search. Returns the position of the first b'\n' in `haystack`,
/// or None if not found. Uses a simple loop that the compiler can auto-vectorize.
#[inline]
fn memchr_newline(haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == b'\n')
}

/// Parse a u32 directly from ASCII digit bytes, avoiding from_utf8 + parse overhead.
#[inline]
fn parse_u32_from_digits(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() {
        return None;
    }
    let mut result: u32 = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        result = result.checked_mul(10)?.checked_add((b - b'0') as u32)?;
    }
    Some(result)
}
