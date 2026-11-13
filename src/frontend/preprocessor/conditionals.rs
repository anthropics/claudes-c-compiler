//! Conditional compilation tracking for the C preprocessor.
//!
//! Handles #if, #ifdef, #ifndef, #elif, #else, #endif directives
//! by maintaining a stack of conditional states.
//!
//! Performance: All scanning operates on byte slices (`&[u8]`) to avoid
//! the overhead of `Vec<char>` allocation. Since C preprocessor expressions
//! are ASCII, this is safe and efficient. Identifiers and numbers are parsed
//! directly from byte spans using `bytes_to_str()`, eliminating intermediate
//! String allocations.

use super::macro_defs::MacroTable;
use super::utils::{is_ident_start_byte, is_ident_cont_byte, bytes_to_str};

/// State of a single conditional (#if/#ifdef/#ifndef block).
#[derive(Debug, Clone, Copy)]
struct ConditionalState {
    /// Whether any branch in this if/elif/else chain has been true
    any_branch_taken: bool,
    /// Whether the current branch is active (code should be emitted)
    current_branch_active: bool,
    /// Whether the parent context is active
    parent_active: bool,
}

/// Tracks the stack of nested conditionals.
#[derive(Debug)]
pub struct ConditionalStack {
    stack: Vec<ConditionalState>,
}

impl ConditionalStack {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    /// Returns true if code should currently be emitted.
    pub fn is_active(&self) -> bool {
        self.stack.last().is_none_or(|s| s.current_branch_active && s.parent_active)
    }

    /// Push a new #if/#ifdef/#ifndef.
    pub fn push_if(&mut self, condition: bool) {
        let parent_active = self.is_active();
        let active = parent_active && condition;
        self.stack.push(ConditionalState {
            any_branch_taken: condition,
            current_branch_active: active,
            parent_active,
        });
    }

    /// Handle #elif.
    pub fn handle_elif(&mut self, condition: bool) {
        if let Some(state) = self.stack.last_mut() {
            if state.any_branch_taken {
                // A previous branch was taken, skip this one
                state.current_branch_active = false;
            } else if state.parent_active && condition {
                state.current_branch_active = true;
                state.any_branch_taken = true;
            } else {
                state.current_branch_active = false;
            }
        }
    }

    /// Handle #else.
    pub fn handle_else(&mut self) {
        if let Some(state) = self.stack.last_mut() {
            if state.any_branch_taken {
                state.current_branch_active = false;
            } else {
                state.current_branch_active = state.parent_active;
                state.any_branch_taken = true;
            }
        }
    }

    /// Handle #endif.
    pub fn handle_endif(&mut self) {
        self.stack.pop();
    }
}

impl Default for ConditionalStack {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate a preprocessor #if expression.
/// This handles basic constant expressions, defined() operator, and macro expansion.
pub fn evaluate_condition(expr: &str, macros: &MacroTable) -> bool {
    // First expand macros in the expression (except inside defined())
    let expanded = expand_condition_macros(expr, macros);
    // Then evaluate the constant expression
    eval_const_expr(&expanded)
}

/// Expand macros in a condition expression, handling `defined(X)` and `defined X` specially.
/// Character and string literals are preserved verbatim (not subject to macro expansion).
///
/// Operates on byte slices for performance: avoids allocating Vec<char>.
fn expand_condition_macros(expr: &str, macros: &MacroTable) -> String {
    let mut result = String::with_capacity(expr.len());
    let bytes = expr.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];

        // Skip character literals verbatim (don't expand macros inside)
        if b == b'\'' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'\'' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < len {
                i += 1; // closing quote
            }
            result.push_str(bytes_to_str(bytes, start, i));
            continue;
        }

        // Skip string literals verbatim
        if b == b'"' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < len {
                i += 1; // closing quote
            }
            result.push_str(bytes_to_str(bytes, start, i));
            continue;
        }

        if i + 7 <= len && &bytes[i..i + 7] == b"defined" {
            // Check it's not part of a larger identifier
            let before_ok = i == 0 || !is_ident_cont_byte(bytes[i - 1]);
            let after_ok = i + 7 >= len || !is_ident_cont_byte(bytes[i + 7]);
            if before_ok && after_ok {
                let start = i;
                i += 7;

                // Skip whitespace
                while i < len && (bytes[i] == b' ' || bytes[i] == b'\t') {
                    i += 1;
                }

                if i < len && bytes[i] == b'(' {
                    // defined(MACRO) -- copy through closing paren
                    i += 1;
                    while i < len && bytes[i] != b')' {
                        i += 1;
                    }
                    if i < len {
                        i += 1; // closing paren
                    }
                } else if i < len && is_ident_start_byte(bytes[i]) {
                    // defined MACRO
                    while i < len && is_ident_cont_byte(bytes[i]) {
                        i += 1;
                    }
                }
                result.push_str(bytes_to_str(bytes, start, i));
                continue;
            }
        }

        // Skip pp-number tokens: a digit (or .digit) followed by alphanumeric chars,
        // dots, underscores, and exponent signs (e+, e-, p+, p-).
        // This prevents macro expansion of suffixes like 'u' in '1u', 'L' in '1L', etc.
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            while i < len {
                if bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'.'
                    || bytes[i] == b'_'
                    || ((bytes[i] == b'+' || bytes[i] == b'-')
                        && i > 0
                        && matches!(bytes[i - 1], b'e' | b'E' | b'p' | b'P'))
                {
                    i += 1;
                } else {
                    break;
                }
            }
            result.push_str(bytes_to_str(bytes, start, i));
            continue;
        }

        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_cont_byte(bytes[i]) {
                i += 1;
            }
            let ident = bytes_to_str(bytes, start, i);

            // Check if this identifier is a pp-number suffix (e.g., 1u, 0xFFULL, 1.0f).
            // If the previous character in result is a digit, this is part of a number
            // token and should NOT be expanded as a macro.
            let is_ppnum_suffix = {
                let rb = result.as_bytes();
                !rb.is_empty() && rb[rb.len() - 1].is_ascii_alphanumeric()
            };
            if is_ppnum_suffix {
                result.push_str(ident);
                continue;
            }

            // Expand macro if it exists
            if let Some(mac) = macros.get(ident) {
                if !mac.is_function_like {
                    result.push_str(&mac.body);
                } else {
                    result.push_str(ident);
                }
            } else {
                // Undefined identifiers become 0 in #if expressions
                result.push_str(ident);
            }
            continue;
        }

        result.push(b as char);
        i += 1;
    }

    result
}

/// Evaluate a constant expression for #if directives.
/// Supports: integer literals, defined(X), !, &&, ||, ==, !=, <, >, <=, >=, +, -, *, /, %, (), unary -, ~
/// Per C99 6.10.1, preprocessor integer expressions use intmax_t (signed) or uintmax_t (unsigned)
/// depending on whether any operand has a 'u'/'U' suffix.
pub fn eval_const_expr(expr: &str) -> bool {
    let expr = expr.trim();
    if expr.is_empty() {
        return false;
    }

    let tokens = tokenize_expr(expr);
    let mut parser = ExprParser::new(&tokens);
    let (result, _is_unsigned) = parser.parse_ternary();
    result != 0
}

/// Simple token for expression evaluation.
/// Num holds (value_as_i64, is_unsigned). The i64 stores the bit pattern;
/// for unsigned values, interpret as u64 via `as u64`.
#[derive(Debug, Clone, PartialEq)]
enum ExprToken {
    Num(i64, bool),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
    Defined,
}

/// Tokenize a preprocessor expression, operating on byte slices for performance.
/// Avoids allocating Vec<char>; numbers are parsed directly from byte spans.
fn tokenize_expr(expr: &str) -> Vec<ExprToken> {
    let mut tokens = Vec::new();
    let bytes = expr.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];

        // Skip whitespace
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Number (decimal, hex, octal)
        if b.is_ascii_digit() {
            let start = i;
            let (raw_val, is_hex_or_oct) = if b == b'0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
                i += 2;
                let hex_start = i;
                while i < len && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let hex_str = bytes_to_str(bytes, hex_start, i);
                (u64::from_str_radix(hex_str, 16).unwrap_or(0), true)
            } else if b == b'0' && i + 1 < len && bytes[i + 1].is_ascii_digit() {
                // Octal
                i += 1;
                let oct_start = i;
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let oct_str = bytes_to_str(bytes, oct_start, i);
                (u64::from_str_radix(oct_str, 8).unwrap_or(0), true)
            } else {
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let num_str = bytes_to_str(bytes, start, i);
                (num_str.parse::<u64>().unwrap_or(0), false)
            };
            // Parse suffixes (U, L, UL, LL, ULL) - track unsigned
            let mut is_unsigned = false;
            while i < len && matches!(bytes[i], b'u' | b'U' | b'l' | b'L') {
                if bytes[i] == b'u' || bytes[i] == b'U' {
                    is_unsigned = true;
                }
                i += 1;
            }
            // Per C99: decimal without U suffix: int -> long -> long long
            // Hex/octal without U suffix: int -> unsigned int -> long -> unsigned long -> long long -> unsigned long long
            // With U suffix: always unsigned
            if !is_unsigned && is_hex_or_oct {
                // Hex/octal: unsigned if value exceeds signed range
                if raw_val > i64::MAX as u64 {
                    is_unsigned = true;
                }
            }
            // Decimal without suffix: stays signed even for large values (they become long long)
            // But if value > i64::MAX (can't happen for decimal u64 parse unless really huge), treat as unsigned
            if !is_unsigned && !is_hex_or_oct && raw_val > i64::MAX as u64 {
                is_unsigned = true;
            }
            tokens.push(ExprToken::Num(raw_val as i64, is_unsigned));
            continue;
        }

        // Character literal
        if b == b'\'' {
            i += 1;
            let val = if i < len && bytes[i] == b'\\' {
                i += 1;
                let c = if i < len {
                    let c = match bytes[i] {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'r' => b'\r',
                        b'0' => b'\0',
                        b'\\' => b'\\',
                        b'\'' => b'\'',
                        b'a' => 0x07,
                        b'b' => 0x08,
                        b'f' => 0x0C,
                        b'v' => 0x0B,
                        other => other,
                    };
                    i += 1;
                    c
                } else {
                    0
                };
                c as i64
            } else if i < len {
                let c = bytes[i] as i64;
                i += 1;
                c
            } else {
                0
            };
            // Skip closing quote
            if i < len && bytes[i] == b'\'' {
                i += 1;
            }
            tokens.push(ExprToken::Num(val, false));
            continue;
        }

        // Identifier or 'defined'
        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_cont_byte(bytes[i]) {
                i += 1;
            }
            let ident = bytes_to_str(bytes, start, i);
            if ident == "defined" {
                tokens.push(ExprToken::Defined);
            } else {
                tokens.push(ExprToken::Ident(ident.to_string()));
            }
            continue;
        }

        // Operators (two-character operators checked first)
        if i + 1 < len {
            let next = bytes[i + 1];
            let two_char_op: Option<&'static str> = match (b, next) {
                (b'!', b'=') => Some("!="),
                (b'&', b'&') => Some("&&"),
                (b'|', b'|') => Some("||"),
                (b'=', b'=') => Some("=="),
                (b'<', b'=') => Some("<="),
                (b'<', b'<') => Some("<<"),
                (b'>', b'=') => Some(">="),
                (b'>', b'>') => Some(">>"),
                _ => None,
            };
            if let Some(op) = two_char_op {
                tokens.push(ExprToken::Op(op));
                i += 2;
                continue;
            }
        }

        // Single-character operators and parens
        match b {
            b'(' => { tokens.push(ExprToken::LParen); i += 1; }
            b')' => { tokens.push(ExprToken::RParen); i += 1; }
            b'!' => { tokens.push(ExprToken::Op("!")); i += 1; }
            b'&' => { tokens.push(ExprToken::Op("&")); i += 1; }
            b'|' => { tokens.push(ExprToken::Op("|")); i += 1; }
            b'+' => { tokens.push(ExprToken::Op("+")); i += 1; }
            b'-' => { tokens.push(ExprToken::Op("-")); i += 1; }
            b'*' => { tokens.push(ExprToken::Op("*")); i += 1; }
            b'/' => { tokens.push(ExprToken::Op("/")); i += 1; }
            b'%' => { tokens.push(ExprToken::Op("%")); i += 1; }
            b'~' => { tokens.push(ExprToken::Op("~")); i += 1; }
            b'^' => { tokens.push(ExprToken::Op("^")); i += 1; }
            b'?' => { tokens.push(ExprToken::Op("?")); i += 1; }
            b':' => { tokens.push(ExprToken::Op(":")); i += 1; }
            b'<' => { tokens.push(ExprToken::Op("<")); i += 1; }
            b'>' => { tokens.push(ExprToken::Op(">")); i += 1; }
            _ => { i += 1; } // skip unknown
        }
    }

    tokens
}

/// Recursive descent parser for preprocessor constant expressions.
/// All parse functions return (value_as_i64, is_unsigned).
/// The i64 stores the bit pattern; when is_unsigned is true, interpret via `as u64`.
/// Per C99 6.10.1: if either operand is unsigned, convert both to unsigned for the operation.
struct ExprParser<'a> {
    tokens: &'a [ExprToken],
    pos: usize,
}

/// Result of a preprocessor expression: (bit_pattern_as_i64, is_unsigned)
type PpVal = (i64, bool);

impl<'a> ExprParser<'a> {
    fn new(tokens: &'a [ExprToken]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&ExprToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&ExprToken> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// Check if the next token matches a given operator string
    fn peek_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(ExprToken::Op(s)) if *s == op)
    }

    fn parse_ternary(&mut self) -> PpVal {
        let cond = self.parse_or();
        if self.peek_op("?") {
            self.advance(); // ?
            let then_val = self.parse_ternary();
            if self.peek_op(":") {
                self.advance(); // :
            }
            let else_val = self.parse_ternary();
            // The result has the common type of both arms (C99 6.5.15p5), so
            // it is unsigned if either arm is, whichever arm is selected.
            let is_unsigned = then_val.1 || else_val.1;
            let value = if cond.0 != 0 { then_val.0 } else { else_val.0 };
            (value, is_unsigned)
        } else {
            cond
        }
    }

    fn parse_or(&mut self) -> PpVal {
        let mut left = self.parse_and();
        while self.peek_op("||") {
            self.advance();
            let right = self.parse_and();
            // Logical operators always produce signed int result
            left = (if left.0 != 0 || right.0 != 0 { 1 } else { 0 }, false);
        }
        left
    }

    fn parse_and(&mut self) -> PpVal {
        let mut left = self.parse_bitor();
        while self.peek_op("&&") {
            self.advance();
            let right = self.parse_bitor();
            left = (if left.0 != 0 && right.0 != 0 { 1 } else { 0 }, false);
        }
        left
    }

    fn parse_bitor(&mut self) -> PpVal {
        let mut left = self.parse_bitxor();
        while self.peek_op("|") {
            self.advance();
            let right = self.parse_bitxor();
            let u = left.1 || right.1;
            left = (left.0 | right.0, u);
        }
        left
    }

    fn parse_bitxor(&mut self) -> PpVal {
        let mut left = self.parse_bitand();
        while self.peek_op("^") {
            self.advance();
            let right = self.parse_bitand();
            let u = left.1 || right.1;
            left = (left.0 ^ right.0, u);
        }
        left
    }

    fn parse_bitand(&mut self) -> PpVal {
        let mut left = self.parse_equality();
        while self.peek_op("&") {
            self.advance();
            let right = self.parse_equality();
            let u = left.1 || right.1;
            left = (left.0 & right.0, u);
        }
        left
    }

    fn parse_equality(&mut self) -> PpVal {
        let mut left = self.parse_relational();
        loop {
            if self.peek_op("==") {
                self.advance();
                let right = self.parse_relational();
                let u = left.1 || right.1;
                let eq = if u {
                    (left.0 as u64) == (right.0 as u64)
                } else {
                    left.0 == right.0
                };
                left = (if eq { 1 } else { 0 }, false);
            } else if self.peek_op("!=") {
                self.advance();
                let right = self.parse_relational();
                let u = left.1 || right.1;
                let ne = if u {
                    (left.0 as u64) != (right.0 as u64)
                } else {
                    left.0 != right.0
                };
                left = (if ne { 1 } else { 0 }, false);
            } else {
                break;
            }
        }
        left
    }

    fn parse_relational(&mut self) -> PpVal {
        let mut left = self.parse_shift();
        loop {
            if self.peek_op("<") {
                self.advance();
                let right = self.parse_shift();
                let u = left.1 || right.1;
                let cmp = if u {
                    (left.0 as u64) < (right.0 as u64)
                } else {
                    left.0 < right.0
                };
                left = (if cmp { 1 } else { 0 }, false);
            } else if self.peek_op(">") {
                self.advance();
                let right = self.parse_shift();
                let u = left.1 || right.1;
                let cmp = if u {
                    (left.0 as u64) > (right.0 as u64)
                } else {
                    left.0 > right.0
                };
                left = (if cmp { 1 } else { 0 }, false);
            } else if self.peek_op("<=") {
                self.advance();
                let right = self.parse_shift();
                let u = left.1 || right.1;
                let cmp = if u {
                    (left.0 as u64) <= (right.0 as u64)
                } else {
                    left.0 <= right.0
                };
                left = (if cmp { 1 } else { 0 }, false);
            } else if self.peek_op(">=") {
                self.advance();
                let right = self.parse_shift();
                let u = left.1 || right.1;
                let cmp = if u {
                    (left.0 as u64) >= (right.0 as u64)
                } else {
                    left.0 >= right.0
                };
                left = (if cmp { 1 } else { 0 }, false);
            } else {
                break;
            }
        }
        left
    }

    fn parse_shift(&mut self) -> PpVal {
        let mut left = self.parse_additive();
        loop {
            if self.peek_op("<<") {
                self.advance();
                let right = self.parse_additive();
                // Shift: result has type of the left operand
                let shift_amt = right.0 as u32;
                if left.1 {
                    left = ((left.0 as u64).wrapping_shl(shift_amt) as i64, true);
                } else {
                    left = (left.0.wrapping_shl(shift_amt), left.1);
                }
            } else if self.peek_op(">>") {
                self.advance();
                let right = self.parse_additive();
                let shift_amt = right.0 as u32;
                if left.1 {
                    // Unsigned: logical shift right
                    left = ((left.0 as u64).wrapping_shr(shift_amt) as i64, true);
                } else {
                    // Signed: arithmetic shift right
                    left = (left.0.wrapping_shr(shift_amt), false);
                }
            } else {
                break;
            }
        }
        left
    }

    fn parse_additive(&mut self) -> PpVal {
        let mut left = self.parse_multiplicative();
        loop {
            if self.peek_op("+") {
                self.advance();
                let right = self.parse_multiplicative();
                let u = left.1 || right.1;
                // Wrapping add works the same for signed and unsigned at bit level
                left = (left.0.wrapping_add(right.0), u);
            } else if self.peek_op("-") {
                self.advance();
                let right = self.parse_multiplicative();
                let u = left.1 || right.1;
                left = (left.0.wrapping_sub(right.0), u);
            } else {
                break;
            }
        }
        left
    }

    fn parse_multiplicative(&mut self) -> PpVal {
        let mut left = self.parse_unary();
        loop {
            if self.peek_op("*") {
                self.advance();
                let right = self.parse_unary();
                let u = left.1 || right.1;
                if u {
                    left = ((left.0 as u64).wrapping_mul(right.0 as u64) as i64, true);
                } else {
                    left = (left.0.wrapping_mul(right.0), false);
                }
            } else if self.peek_op("/") {
                self.advance();
                let right = self.parse_unary();
                let u = left.1 || right.1;
                if right.0 == 0 {
                    left = (0, u);
                } else if u {
                    left = ((left.0 as u64).wrapping_div(right.0 as u64) as i64, true);
                } else {
                    // Avoid signed overflow on i64::MIN / -1
                    if left.0 == i64::MIN && right.0 == -1 {
                        left = (i64::MIN, false);
                    } else {
                        left = (left.0 / right.0, false);
                    }
                }
            } else if self.peek_op("%") {
                self.advance();
                let right = self.parse_unary();
                let u = left.1 || right.1;
                if right.0 == 0 {
                    left = (0, u);
                } else if u {
                    left = ((left.0 as u64).wrapping_rem(right.0 as u64) as i64, true);
                } else if left.0 == i64::MIN && right.0 == -1 {
                    left = (0, false);
                } else {
                    left = (left.0 % right.0, false);
                }
            } else {
                break;
            }
        }
        left
    }

    fn parse_unary(&mut self) -> PpVal {
        if self.peek_op("!") {
            self.advance();
            let val = self.parse_unary();
            // Logical NOT always produces signed int
            return (if val.0 == 0 { 1 } else { 0 }, false);
        }
        if self.peek_op("-") {
            self.advance();
            let val = self.parse_unary();
            // Unary minus: if unsigned, the result is still unsigned (wrapping)
            if val.1 {
                return ((val.0 as u64).wrapping_neg() as i64, true);
            } else {
                return (val.0.wrapping_neg(), false);
            }
        }
        if self.peek_op("+") {
            self.advance();
            return self.parse_unary();
        }
        if self.peek_op("~") {
            self.advance();
            let val = self.parse_unary();
            // Bitwise NOT preserves signedness
            return (!val.0, val.1);
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> PpVal {
        match self.peek().cloned() {
            Some(ExprToken::Num(n, u)) => {
                self.advance();
                (n, u)
            }
            Some(ExprToken::LParen) => {
                self.advance();
                let val = self.parse_ternary();
                if self.peek() == Some(&ExprToken::RParen) {
                    self.advance();
                }
                val
            }
            Some(ExprToken::Defined) => {
                self.advance();
                // defined(X) or defined X
                let has_paren = if self.peek() == Some(&ExprToken::LParen) {
                    self.advance();
                    true
                } else {
                    false
                };
                // The identifier - in our evaluation, we've already resolved
                // defined() to 0 or 1, so identifiers here mean "not defined"
                if let Some(ExprToken::Ident(_)) = self.peek() {
                    self.advance();
                }
                if has_paren && self.peek() == Some(&ExprToken::RParen) {
                    self.advance();
                }
                // If we reach here during evaluation, the macro was not resolved
                // at expansion time, so it's not defined
                (0, false)
            }
            Some(ExprToken::Ident(ref name)) => {
                let name = name.clone();
                self.advance();
                // Undefined identifiers in #if evaluate to 0
                // Special case: "true" and "false"
                match name.as_str() {
                    "true" => (1, false),
                    "false" => (0, false),
                    _ => (0, false),
                }
            }
            _ => {
                self.advance(); // skip unknown
                (0, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conditional operator converts its second and third operands to a
    /// common type (C99 6.5.15p5), so one unsigned arm makes the result
    /// unsigned even when the signed arm is the one selected.
    #[test]
    fn test_ternary_result_has_common_type_of_both_arms() {
        // -1 converted to uintmax_t is UINTMAX_MAX.
        assert!(!eval_const_expr("(1 ? -1 : 0u) < 0"));
        assert!(eval_const_expr("(1 ? -1 : 0u) > 0"));
        assert!(!eval_const_expr("(0 ? 0u : -1) < 0"));
        assert!(eval_const_expr("(0 ? 1u : -1) / 2 == 9223372036854775807"));
        // A literal too large for intmax_t is unsigned without a suffix.
        assert!(!eval_const_expr("(1 ? -1 : 0xFFFFFFFFFFFFFFFF) < 0"));
        // The common type propagates out of a nested conditional.
        assert!(!eval_const_expr("(1 ? (1 ? -1 : 0) : 0u) < 0"));
    }

    #[test]
    fn test_ternary_with_signed_arms_stays_signed() {
        assert!(eval_const_expr("(1 ? -1 : 0) < 0"));
        assert!(eval_const_expr("(0 ? 0 : -1) < 0"));
        // The type of the condition does not affect the result type.
        assert!(eval_const_expr("(1u ? -1 : 0) < 0"));
    }
}
