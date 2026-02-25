//! Structured diagnostic infrastructure for the compiler.
//!
//! Provides a `DiagnosticEngine` that collects errors, warnings, and notes
//! with source locations and renders them in GCC-compatible format with
//! optional source snippet display and ANSI color output.
//!
//! # Warning control
//! The engine supports GCC-compatible warning flags:
//! - `-Werror`: promote all warnings to errors
//! - `-Werror=<name>`: promote a specific warning to an error
//! - `-Wno-error=<name>`: demote a specific warning back from error
//! - `-Wall`: enable standard warning set
//! - `-Wextra`: enable additional warnings
//! - `-W<name>`: enable a specific warning
//! - `-Wno-<name>`: disable a specific warning
//!
//! Flags are processed left-to-right, so `-Wall -Wno-unused-variable` enables
//! all warnings except unused-variable.
//!
//! # Color output
//! Controlled by `-fdiagnostics-color={auto,always,never}`:
//! - `auto` (default): colorize when stderr is a terminal
//! - `always`: always emit ANSI color codes
//! - `never`: plain text output
//!
//! Color scheme (matching GCC):
//! - Location (file:line:col): **bold white**
//! - `error:` label: **bold red**
//! - `warning:` label: **bold magenta**
//! - `note:` label: **bold cyan**
//! - Caret/underline (`^~~~`): **bold green**
//! - Fix-it hints: **bold green**
//!
//! # Output format
//! ```text
//! file.c:10:5: error: expected ';', got '}'
//!     int x = 42
//!             ^
//! ```

use crate::common::source::{SourceManager, Span};

/// Controls whether diagnostic output uses ANSI color escape codes.
///
/// Matches GCC's `-fdiagnostics-color` flag with the same three modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// Colorize output only when stderr is a terminal (default).
    #[default]
    Auto,
    /// Always emit ANSI color codes.
    Always,
    /// Never emit color codes; plain text output.
    Never,
}

impl ColorMode {
    /// Parse a `-fdiagnostics-color` flag value.
    /// Returns `None` for unrecognized values.
    pub fn from_flag(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(ColorMode::Auto),
            "always" => Some(ColorMode::Always),
            "never" => Some(ColorMode::Never),
            _ => None,
        }
    }

    /// Resolve whether colors should actually be used, considering
    /// the mode and whether stderr is a terminal.
    fn use_color(self) -> bool {
        match self {
            ColorMode::Always => true,
            ColorMode::Never => false,
            ColorMode::Auto => {
                use std::io::IsTerminal;
                std::io::stderr().is_terminal()
            }
        }
    }
}


/// Categories of warnings, matching GCC's -W<name> flag names.
///
/// Each variant corresponds to a `-W<name>` flag. For example,
/// `WarningKind::ImplicitFunctionDeclaration` maps to
/// `-Wimplicit-function-declaration`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningKind {
    /// An undeclared identifier was used.
    /// NOTE: As of the fix in sema.rs, undeclared variables are now hard errors
    /// (not warnings). This variant is kept for CLI flag parsing compatibility
    /// (-Wno-undeclared, -Werror=undeclared) but is no longer emitted.
    Undeclared,
    /// A function was called without a prior declaration (C89 implicit int).
    /// GCC flag: -Wimplicit-function-declaration
    ImplicitFunctionDeclaration,
    /// A `#warning` preprocessor directive was encountered.
    /// GCC flag: -Wcpp
    Cpp,
    /// Control reaches end of non-void function, or non-void function does
    /// not return a value in all control paths.
    /// GCC flag: -Wreturn-type
    ReturnType,
    // Future categories (add as warnings are implemented):
    // UnusedVariable,         // -Wunused-variable
    // UnusedFunction,         // -Wunused-function
    // UnusedParameter,        // -Wunused-parameter
    // UninitializedVariable,  // -Wuninitialized
    // ImplicitConversion,     // -Wimplicit-int-conversion
    // SignCompare,            // -Wsign-compare
    // Parentheses,            // -Wparentheses
    // Pointer,                // -Wpointer-sign
}

impl WarningKind {
    /// The GCC-compatible flag name for this warning (without the -W prefix).
    pub fn flag_name(self) -> &'static str {
        match self {
            WarningKind::Undeclared => "undeclared",
            WarningKind::ImplicitFunctionDeclaration => "implicit-function-declaration",
            WarningKind::Cpp => "cpp",
            WarningKind::ReturnType => "return-type",
        }
    }

    /// Parse a warning flag name (without the -W prefix) into a WarningKind.
    /// Returns None for unrecognized names (which are silently ignored, matching GCC).
    pub fn from_flag_name(name: &str) -> Option<Self> {
        match name {
            "undeclared" => Some(WarningKind::Undeclared),
            "implicit-function-declaration" => Some(WarningKind::ImplicitFunctionDeclaration),
            "implicit" => Some(WarningKind::ImplicitFunctionDeclaration),
            "cpp" => Some(WarningKind::Cpp),
            "return-type" => Some(WarningKind::ReturnType),
            _ => None,
        }
    }

    /// All warning kinds that are enabled by -Wall.
    pub fn wall_set() -> &'static [WarningKind] {
        &[
            WarningKind::ImplicitFunctionDeclaration,
            WarningKind::Cpp,
            WarningKind::ReturnType,
            // WarningKind::Undeclared is now a hard error, not a warning
        ]
    }

    /// All warning kinds that are enabled by -Wextra (in addition to -Wall).
    pub fn wextra_set() -> &'static [WarningKind] {
        // Currently no extra warnings beyond -Wall.
        &[]
    }

    /// All known warning kinds.
    fn all() -> &'static [WarningKind] {
        &[
            WarningKind::Undeclared,
            WarningKind::ImplicitFunctionDeclaration,
            WarningKind::Cpp,
            WarningKind::ReturnType,
        ]
    }
}

/// Configuration for warning behavior: which warnings are enabled,
/// and which are promoted to errors.
///
/// Processed from CLI flags in left-to-right order.
#[derive(Debug, Clone)]
pub struct WarningConfig {
    /// Per-warning enabled state. Warnings not in this set are suppressed.
    enabled: std::collections::HashSet<WarningKind>,
    /// Per-warning error promotion. Warnings in this set are emitted as errors.
    errors: std::collections::HashSet<WarningKind>,
    /// Global -Werror flag: promote ALL warnings to errors.
    pub werror_all: bool,
}

impl WarningConfig {
    /// Create a default config with all warnings enabled and none promoted to errors.
    pub fn new() -> Self {
        let mut enabled = std::collections::HashSet::new();
        for &kind in WarningKind::all() {
            enabled.insert(kind);
        }
        Self {
            enabled,
            errors: std::collections::HashSet::new(),
            werror_all: false,
        }
    }

    /// Enable a specific warning.
    pub fn enable(&mut self, kind: WarningKind) {
        self.enabled.insert(kind);
    }

    /// Disable a specific warning (-Wno-<name>).
    pub fn disable(&mut self, kind: WarningKind) {
        self.enabled.remove(&kind);
    }

    /// Enable all -Wall warnings.
    pub fn enable_wall(&mut self) {
        for &kind in WarningKind::wall_set() {
            self.enabled.insert(kind);
        }
    }

    /// Enable all -Wextra warnings (superset of -Wall).
    pub fn enable_wextra(&mut self) {
        self.enable_wall();
        for &kind in WarningKind::wextra_set() {
            self.enabled.insert(kind);
        }
    }

    /// Promote a specific warning to error (-Werror=<name>).
    pub fn set_werror(&mut self, kind: WarningKind) {
        self.errors.insert(kind);
        // -Werror=<name> also implicitly enables the warning
        self.enabled.insert(kind);
    }

    /// Demote a specific warning from error (-Wno-error=<name>).
    pub fn clear_werror(&mut self, kind: WarningKind) {
        self.errors.remove(&kind);
    }

    /// Check if a warning of the given kind should be emitted at all.
    pub fn is_enabled(&self, kind: WarningKind) -> bool {
        self.enabled.contains(&kind)
    }

    /// Check if a warning of the given kind should be promoted to an error.
    pub fn is_error(&self, kind: WarningKind) -> bool {
        self.werror_all || self.errors.contains(&kind)
    }

    /// Process a single -W flag argument (the part after -W).
    /// Returns true if the flag was recognized, false if unknown (silently ignored).
    pub fn process_flag(&mut self, flag: &str) -> bool {
        match flag {
            "error" => {
                self.werror_all = true;
                true
            }
            "no-error" => {
                self.werror_all = false;
                true
            }
            "all" => {
                self.enable_wall();
                true
            }
            "extra" => {
                self.enable_wextra();
                true
            }
            _ if flag.starts_with("error=") => {
                let name = &flag[6..];
                if let Some(kind) = WarningKind::from_flag_name(name) {
                    self.set_werror(kind);
                    true
                } else {
                    false // unknown warning name, silently ignored
                }
            }
            _ if flag.starts_with("no-error=") => {
                let name = &flag[9..];
                if let Some(kind) = WarningKind::from_flag_name(name) {
                    self.clear_werror(kind);
                    true
                } else {
                    false
                }
            }
            _ if flag.starts_with("no-") => {
                let name = &flag[3..];
                if let Some(kind) = WarningKind::from_flag_name(name) {
                    self.disable(kind);
                    true
                } else {
                    false // unknown warning name, silently ignored
                }
            }
            _ => {
                // Try as a positive warning name: -W<name> enables it
                if let Some(kind) = WarningKind::from_flag_name(flag) {
                    self.enable(kind);
                    true
                } else {
                    false // unknown, silently ignored (matching GCC behavior)
                }
            }
        }
    }
}

impl Default for WarningConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Severity level for a diagnostic message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A fatal error that prevents compilation from continuing.
    Error,
    /// A warning that does not prevent compilation.
    Warning,
    /// A supplementary note attached to a previous error or warning.
    Note,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Note => write!(f, "note"),
        }
    }
}

/// A single diagnostic message with severity, location, and message text.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
    /// Optional warning category for filtering and -Werror promotion.
    pub warning_kind: Option<WarningKind>,
    /// Optional follow-up notes providing additional context.
    pub notes: Vec<Diagnostic>,
    /// Optional fix-it hint: a short suggestion for how to fix the problem.
    /// Rendered below the snippet as "fix-it hint: insert ';'" etc.
    pub fix_hint: Option<String>,
    /// Explicit source location string for diagnostics without a span.
    /// Used for preprocessor-phase diagnostics (e.g., `#error`, `#warning`,
    /// missing `#include`) where the source manager is not yet available.
    /// Format: "file:line:" (rendered as the location prefix in output).
    pub explicit_location: Option<String>,
}

impl Diagnostic {
    /// Create a new error diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            span: None,
            warning_kind: None,
            notes: Vec::new(),
            fix_hint: None,
            explicit_location: None,
        }
    }

    /// Create a new warning diagnostic with a category.
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            span: None,
            warning_kind: None,
            notes: Vec::new(),
            fix_hint: None,
            explicit_location: None,
        }
    }

    /// Create a new warning diagnostic with a specific kind for filtering.
    pub fn warning_with_kind(message: impl Into<String>, kind: WarningKind) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            span: None,
            warning_kind: Some(kind),
            notes: Vec::new(),
            fix_hint: None,
            explicit_location: None,
        }
    }

    /// Create a new note diagnostic.
    pub fn note(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Note,
            message: message.into(),
            span: None,
            warning_kind: None,
            notes: Vec::new(),
            fix_hint: None,
            explicit_location: None,
        }
    }

    /// Attach a source span to this diagnostic.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a follow-up note to this diagnostic.
    /// Used for "note: expression has type '...'" and similar context.
    pub fn with_note(mut self, note: Diagnostic) -> Self {
        self.notes.push(note);
        self
    }

    /// Add a fix-it hint suggestion.
    /// Rendered below the source snippet, e.g., "fix-it hint: insert ';' after expression".
    pub fn with_fix_hint(mut self, hint: impl Into<String>) -> Self {
        self.fix_hint = Some(hint.into());
        self
    }

    /// Set an explicit source location string for diagnostics without a span.
    /// Used for preprocessor-phase diagnostics where the source manager is
    /// not yet available. The location string is rendered as the prefix
    /// (e.g., "file.c:10:1:") before the severity label.
    pub fn with_location(mut self, file: &str, line: usize, col: usize) -> Self {
        self.explicit_location = Some(format!("{}:{}:{}:", file, line, col));
        self
    }
}

/// Collects and renders compiler diagnostics with source context.
///
/// The engine is designed to be threaded through compilation phases. Each
/// phase calls `emit()` to report errors and warnings. After each phase,
/// the driver checks `has_errors()` to decide whether to continue.
///
/// Diagnostics are printed to stderr immediately on emit, matching GCC
/// behavior where errors appear as soon as they are discovered.
///
/// Warning filtering and -Werror promotion are controlled by the
/// `WarningConfig` set on the engine.
pub struct DiagnosticEngine {
    error_count: usize,
    warning_count: usize,
    /// Warning configuration: controls which warnings are enabled and
    /// which are promoted to errors.
    warning_config: WarningConfig,
    /// Reference to the source manager for span resolution and snippet display.
    /// Set after the preprocessing/lexing phase creates the SourceManager.
    source_manager: Option<SourceManager>,
    /// Whether to use ANSI color codes in diagnostic output.
    /// Resolved once at creation from the ColorMode setting.
    use_color: bool,
    /// The file for which we last emitted an "In file included from" chain.
    /// Used to avoid repeating the same chain for consecutive errors in the
    /// same included file (matching GCC behavior).
    last_include_trace_file: Option<String>,
}

impl DiagnosticEngine {
    /// Create a new diagnostic engine with no source manager and default warning config.
    /// Color mode defaults to `Auto` (colorize when stderr is a terminal).
    pub fn new() -> Self {
        Self {
            error_count: 0,
            warning_count: 0,
            warning_config: WarningConfig::new(),
            source_manager: None,
            use_color: ColorMode::Auto.use_color(),
            last_include_trace_file: None,
        }
    }

    /// Set the warning configuration (parsed from CLI flags).
    pub fn set_warning_config(&mut self, config: WarningConfig) {
        self.warning_config = config;
    }

    /// Set the source manager for span resolution and snippet display.
    pub fn set_source_manager(&mut self, sm: SourceManager) {
        self.source_manager = Some(sm);
    }

    /// Take ownership of the source manager (for passing to codegen for debug info).
    pub fn take_source_manager(&mut self) -> Option<SourceManager> {
        self.source_manager.take()
    }

    /// Set the color mode for diagnostic output.
    /// Resolves the mode immediately (e.g., checking isatty for Auto).
    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.use_color = mode.use_color();
    }

    /// Emit a diagnostic: apply warning filtering/promotion, print to stderr,
    /// and update counts.
    ///
    /// For warnings with a `WarningKind`:
    /// - If the warning is disabled in the config, it is silently suppressed.
    /// - If -Werror or -Werror=<name> applies, it is promoted to an error.
    /// - The GCC-style `[-W<name>]` suffix is appended to the message.
    pub fn emit(&mut self, diag: &Diagnostic) {
        match diag.severity {
            Severity::Warning => {
                if let Some(kind) = diag.warning_kind {
                    // Check if this warning category is enabled
                    if !self.warning_config.is_enabled(kind) {
                        return; // suppressed by -Wno-<name>
                    }

                    // Check if this warning should be promoted to an error
                    if self.warning_config.is_error(kind) {
                        // Promote: render as error with [-Werror=<name>] suffix
                        let promoted = Diagnostic {
                            severity: Severity::Error,
                            message: format!("{} [-Werror={}]", diag.message, kind.flag_name()),
                            span: diag.span,
                            warning_kind: diag.warning_kind,
                            notes: diag.notes.clone(),
                            fix_hint: diag.fix_hint.clone(),
                            explicit_location: diag.explicit_location.clone(),
                        };
                        self.render_diagnostic(&promoted);
                        self.error_count += 1;
                        return;
                    }

                    // Render as warning with [-W<name>] suffix
                    let annotated = Diagnostic {
                        severity: Severity::Warning,
                        message: format!("{} [-W{}]", diag.message, kind.flag_name()),
                        span: diag.span,
                        warning_kind: diag.warning_kind,
                        notes: diag.notes.clone(),
                        fix_hint: diag.fix_hint.clone(),
                        explicit_location: diag.explicit_location.clone(),
                    };
                    self.render_diagnostic(&annotated);
                    self.warning_count += 1;
                } else {
                    // Warning without a kind: always emit, subject to global -Werror
                    if self.warning_config.werror_all {
                        let promoted = Diagnostic {
                            severity: Severity::Error,
                            message: format!("{} [-Werror]", diag.message),
                            span: diag.span,
                            warning_kind: None,
                            notes: diag.notes.clone(),
                            fix_hint: diag.fix_hint.clone(),
                            explicit_location: diag.explicit_location.clone(),
                        };
                        self.render_diagnostic(&promoted);
                        self.error_count += 1;
                    } else {
                        self.render_diagnostic(diag);
                        self.warning_count += 1;
                    }
                }
            }
            Severity::Error => {
                self.render_diagnostic(diag);
                self.error_count += 1;
            }
            Severity::Note => {
                self.render_diagnostic(diag);
                // notes don't count separately
            }
        }
    }

    /// Emit an error with a message and span.
    pub fn error(&mut self, message: impl Into<String>, span: Span) {
        let diag = Diagnostic::error(message).with_span(span);
        self.emit(&diag);
    }

    /// Emit a warning with a message and span.
    pub fn warning(&mut self, message: impl Into<String>, span: Span) {
        let diag = Diagnostic::warning(message).with_span(span);
        self.emit(&diag);
    }

    /// Emit a categorized warning with a message, span, and kind.
    pub fn warning_with_kind(&mut self, message: impl Into<String>, span: Span, kind: WarningKind) {
        let diag = Diagnostic::warning_with_kind(message, kind).with_span(span);
        self.emit(&diag);
    }

    /// Returns true if any errors have been emitted (including promoted warnings).
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Number of errors emitted so far (including promoted warnings).
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Number of warnings emitted so far (excludes promoted warnings).
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Render "In file included from X:Y:" traces for errors in included files.
    ///
    /// GCC emits these traces before the first diagnostic in an included file,
    /// showing the chain of #include directives that led to the current file.
    /// Example output:
    /// ```text
    /// In file included from main.c:4:
    /// In file included from header1.h:2:
    /// header2.h:10:5: error: unknown type name 'foo'
    /// ```
    ///
    /// The trace is only emitted once per file -- consecutive errors in the
    /// same included file do not repeat the chain.
    fn render_include_trace(&mut self, diag: &Diagnostic) {
        // Only emit traces for diagnostics that have a span and are errors/warnings
        // (not for notes, which are sub-diagnostics)
        if diag.severity == Severity::Note {
            return;
        }

        let span = match diag.span {
            Some(s) => s,
            None => return,
        };

        let sm = match &self.source_manager {
            Some(sm) => sm,
            None => return,
        };

        let loc = sm.resolve_span(span);

        // Check if we already emitted a trace for this file
        if let Some(ref last_file) = self.last_include_trace_file {
            if *last_file == loc.file {
                return; // Same file as last diagnostic, skip trace
            }
        }

        // Get the include chain for this file
        let chain = sm.get_include_chain(&loc.file);
        if chain.is_empty() {
            // Main file or no include info -- no trace needed
            self.last_include_trace_file = Some(loc.file.clone());
            return;
        }

        // Emit the include trace matching GCC format:
        //   In file included from header1.h:3,
        //                    from main.c:1:
        // The first entry uses "In file included from", subsequent entries use
        // "from" with indentation to align with the first entry.
        for (i, origin) in chain.iter().enumerate() {
            let is_last = i == chain.len() - 1;
            let suffix = if is_last { ":" } else { "," };
            if i == 0 {
                if self.use_color {
                    eprintln!("\x1b[1mIn file included from {}:{}{}\x1b[0m",
                        origin.file, origin.line, suffix);
                } else {
                    eprintln!("In file included from {}:{}{}",
                        origin.file, origin.line, suffix);
                }
            } else {
                // "                 from " aligns with "In file included from "
                if self.use_color {
                    eprintln!("\x1b[1m                 from {}:{}{}\x1b[0m",
                        origin.file, origin.line, suffix);
                } else {
                    eprintln!("                 from {}:{}{}",
                        origin.file, origin.line, suffix);
                }
            }
        }

        self.last_include_trace_file = Some(loc.file.clone());
    }

    /// Render a single diagnostic to stderr, including location, severity,
    /// message, source snippet with caret, fix-it hints, and any follow-up notes.
    ///
    /// When color is enabled, output matches GCC's color scheme:
    /// - Location prefix: bold white
    /// - "error:": bold red
    /// - "warning:": bold magenta
    /// - "note:": bold cyan
    /// - Message text: bold white
    /// - Source snippet: no special color
    /// - Caret/underline: bold green
    /// - Fix-it hints: bold green
    fn render_diagnostic(&mut self, diag: &Diagnostic) {
        // Emit "In file included from" traces for errors in included files
        self.render_include_trace(diag);

        use std::fmt::Write;
        let mut msg = String::new();

        if self.use_color {
            // Colored output matching GCC
            // Location prefix in bold white
            if let Some(span) = diag.span {
                if let Some(ref sm) = self.source_manager {
                    let loc = sm.resolve_span(span);
                    let _ = write!(msg, "\x1b[1m{}:{}:{}: \x1b[0m",
                        loc.file, loc.line, loc.column);
                }
            } else if let Some(ref loc) = diag.explicit_location {
                let _ = write!(msg, "\x1b[1m{} \x1b[0m", loc);
            }
            // Severity label in its color, message in bold white
            let (sev_color, sev_text) = match diag.severity {
                Severity::Error => ("\x1b[1;31m", "error"),     // bold red
                Severity::Warning => ("\x1b[1;35m", "warning"), // bold magenta
                Severity::Note => ("\x1b[1;36m", "note"),       // bold cyan
            };
            let _ = write!(msg, "{}{}\x1b[0m: \x1b[1m{}\x1b[0m",
                sev_color, sev_text, diag.message);
        } else {
            // Plain text output
            if let Some(span) = diag.span {
                if let Some(ref sm) = self.source_manager {
                    let loc = sm.resolve_span(span);
                    let _ = write!(msg, "{}:{}:{}: ", loc.file, loc.line, loc.column);
                }
            } else if let Some(ref loc) = diag.explicit_location {
                let _ = write!(msg, "{} ", loc);
            }
            let _ = write!(msg, "{}: {}", diag.severity, diag.message);
        }
        eprintln!("{}", msg);

        // Source snippet with caret underline
        if let Some(span) = diag.span {
            self.render_snippet(span);
        }

        // Render fix-it hint if present
        if let Some(ref hint) = diag.fix_hint {
            if self.use_color {
                eprintln!("  \x1b[1;32mfix-it hint:\x1b[0m {}", hint);
            } else {
                eprintln!("  fix-it hint: {}", hint);
            }
        }

        // Render "in expansion of macro 'X'" note for errors in macro expansions.
        // Only for primary diagnostics (errors/warnings), not for sub-notes.
        if diag.severity != Severity::Note {
            self.render_macro_expansion_trace(diag);
        }

        // Render any follow-up notes
        for note in &diag.notes {
            self.render_diagnostic(note);
        }
    }

    /// Render "note: in expansion of macro 'X'" traces for diagnostics that
    /// occur in lines produced by macro expansion.
    ///
    /// GCC emits these notes when an error or warning is triggered inside
    /// macro-expanded code, helping the user understand that the error is
    /// in the expansion of a specific macro.
    ///
    /// Example output:
    /// ```text
    /// file.c:10:5: error: expected ';' before '}' token
    ///    10 |   FOO(x)
    ///       |   ^
    /// file.c:10:5: note: in expansion of macro 'FOO'
    /// ```
    fn render_macro_expansion_trace(&self, diag: &Diagnostic) {
        let span = match diag.span {
            Some(s) => s,
            None => return,
        };

        let sm = match &self.source_manager {
            Some(sm) => sm,
            None => return,
        };

        let macro_names = match sm.get_macro_expansion_at(span) {
            Some(names) => names,
            None => return,
        };

        // Filter out predefined/builtin macros that are expanded ubiquitously
        // and would produce noise in diagnostics. Only show user-defined macros
        // that are likely relevant to the error.
        let interesting: Vec<&str> = macro_names.iter()
            .filter(|name| !is_uninteresting_macro(name))
            .map(|s| &**s)
            .collect();

        if interesting.is_empty() {
            return;
        }

        let loc = sm.resolve_span(span);

        // Emit a note for each macro in the expansion chain (outermost first).
        // Typically there is only one, but nested expansions may have several.
        // Limit to 3 levels to avoid noisy output for deeply nested macros.
        for name in interesting.iter().take(3) {
            if self.use_color {
                eprintln!("\x1b[1m{}:{}:{}: \x1b[0m\x1b[1;36mnote:\x1b[0m \x1b[1min expansion of macro '{}'\x1b[0m",
                    loc.file, loc.line, loc.column, name);
            } else {
                eprintln!("{}:{}:{}: note: in expansion of macro '{}'",
                    loc.file, loc.line, loc.column, name);
            }
        }
    }

    /// Render a source code snippet with a caret pointing to the error location.
    ///
    /// Output format:
    /// ```text
    ///     int x = 42
    ///             ^~
    /// ```
    ///
    /// When color is enabled, the caret and underline are rendered in bold green
    /// (matching GCC's behavior).
    fn render_snippet(&self, span: Span) {
        let sm = match &self.source_manager {
            Some(sm) => sm,
            None => return,
        };

        let source_line = match sm.get_source_line(span) {
            Some(line) => line,
            None => return,
        };

        // Don't render snippets for empty or whitespace-only lines
        if source_line.trim().is_empty() {
            return;
        }

        // Resolve the column for caret positioning
        let loc = sm.resolve_span(span);
        let col = loc.column as usize;

        // Print the source line with indentation
        eprintln!(" {}", source_line);

        // Build the caret line: spaces up to the column, then ^ with tildes
        if col > 0 {
            let padding = " ".repeat(col);
            let span_len = (span.end.saturating_sub(span.start)) as usize;
            let underline = if span_len > 1 {
                format!("^{}", "~".repeat(span_len - 1))
            } else {
                "^".to_string()
            };
            if self.use_color {
                eprintln!("{}\x1b[1;32m{}\x1b[0m", padding, underline);
            } else {
                eprintln!("{}{}", padding, underline);
            }
        }
    }
}

impl Default for DiagnosticEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a macro name is a predefined/builtin macro that should not be shown
/// in "in expansion of macro" diagnostic notes. These macros are expanded
/// ubiquitously in C code and showing them would produce noise rather than
/// helping the user understand the error.
fn is_uninteresting_macro(name: &str) -> bool {
    // GCC/Clang builtin function-style macros
    if name.starts_with("__builtin_") {
        return true;
    }
    // Common standard library macros that are uninteresting
    matches!(name, "NULL" | "EOF" | "CHAR_BIT" | "CHAR_MAX" | "CHAR_MIN" |
        "INT_MAX" | "INT_MIN" | "UINT_MAX" | "LONG_MAX" | "LONG_MIN" |
        "ULONG_MAX" | "LLONG_MAX" | "LLONG_MIN" | "ULLONG_MAX" |
        "SIZE_MAX" | "PTRDIFF_MAX" | "PTRDIFF_MIN" |
        "true" | "false" | "bool" |
        "INT8_MAX" | "INT16_MAX" | "INT32_MAX" | "INT64_MAX" |
        "UINT8_MAX" | "UINT16_MAX" | "UINT32_MAX" | "UINT64_MAX" |
        "INT8_MIN" | "INT16_MIN" | "INT32_MIN" | "INT64_MIN" |
        "WCHAR_MAX" | "WCHAR_MIN" |
        "INTPTR_MAX" | "INTPTR_MIN" | "UINTPTR_MAX" |
        "SSIZE_MAX" | "PATH_MAX" |
        "SEEK_SET" | "SEEK_CUR" | "SEEK_END" |
        "STDIN_FILENO" | "STDOUT_FILENO" | "STDERR_FILENO" |
        "EXIT_SUCCESS" | "EXIT_FAILURE"
    )
}
