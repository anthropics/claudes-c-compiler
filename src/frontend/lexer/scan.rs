use std::rc::Rc;
use crate::common::encoding::decode_pua_byte;
use crate::common::source::Span;
use super::token::{Token, TokenKind};

/// C lexer that tokenizes source input with source locations.
pub struct Lexer {
    input: Vec<u8>,
    pos: usize,
    file_id: u32,
    gnu_extensions: bool,
}

impl Lexer {
    pub fn new(input: &str, file_id: u32) -> Self {
        Self {
            input: input.bytes().collect(),
            pos: 0,
            file_id,
            gnu_extensions: true,
        }
    }

    pub fn set_gnu_extensions(&mut self, enabled: bool) {
        self.gnu_extensions = enabled;
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        // Estimate ~1 token per 5 bytes of source (typical for C code).
        let mut tokens = Vec::with_capacity(self.input.len() / 5);
        loop {
            let tok = self.next_token();
            let is_eof = tok.is_eof();
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        tokens
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        if self.pos >= self.input.len() {
            return Token::new(TokenKind::Eof, Span::new(self.pos as u32, self.pos as u32, self.file_id));
        }

        let start = self.pos;
        let ch = self.input[self.pos];

        // Number literals
        if ch.is_ascii_digit() || (ch == b'.' && self.peek_next().is_some_and(|c| c.is_ascii_digit())) {
            return self.lex_number(start);
        }

        // String literals
        if ch == b'"' {
            return self.lex_string(start);
        }

        // Character literals
        if ch == b'\'' {
            return self.lex_char(start);
        }

        // Identifiers and keywords
        // GCC extension: '$' is allowed in identifiers (-fdollars-in-identifiers, on by default)
        if ch == b'_' || ch == b'$' || ch.is_ascii_alphabetic() {
            return self.lex_identifier(start);
        }

        // Punctuation and operators
        self.lex_punctuation(start)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }

            if self.pos >= self.input.len() {
                return;
            }

            // Skip GCC-style line markers: # <number> "filename"
            // These are emitted by the preprocessor and must not be lexed as tokens.
            // A line marker is a '#' at the start of a line (after optional whitespace)
            // followed by a digit.
            if self.input[self.pos] == b'#' && self.is_line_marker() {
                // Skip the entire line
                while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }

            // Skip line comments
            if self.pos + 1 < self.input.len() && self.input[self.pos] == b'/' && self.input[self.pos + 1] == b'/' {
                while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }

            // Skip block comments
            if self.pos + 1 < self.input.len() && self.input[self.pos] == b'/' && self.input[self.pos + 1] == b'*' {
                self.pos += 2;
                while self.pos + 1 < self.input.len() {
                    if self.input[self.pos] == b'*' && self.input[self.pos + 1] == b'/' {
                        self.pos += 2;
                        break;
                    }
                    self.pos += 1;
                }
                continue;
            }

            break;
        }
    }

    /// Check if the current position is at a GCC-style line marker.
    /// A line marker is `# <digit>` at the start of a line (i.e., the '#' is
    /// either at position 0 or preceded by a newline).
    fn is_line_marker(&self) -> bool {
        // Must be at '#'
        if self.pos >= self.input.len() || self.input[self.pos] != b'#' {
            return false;
        }
        // '#' must be at the start of a line
        if self.pos > 0 && self.input[self.pos - 1] != b'\n' {
            return false;
        }
        // Next non-space char must be a digit
        let mut j = self.pos + 1;
        while j < self.input.len() && self.input[j] == b' ' {
            j += 1;
        }
        j < self.input.len() && self.input[j].is_ascii_digit()
    }

    fn peek_next(&self) -> Option<u8> {
        if self.pos + 1 < self.input.len() {
            Some(self.input[self.pos + 1])
        } else {
            None
        }
    }

    fn lex_number(&mut self, start: usize) -> Token {
        if self.pos + 1 < self.input.len() && self.input[self.pos] == b'0'
            && (self.input[self.pos + 1] == b'x' || self.input[self.pos + 1] == b'X')
        {
            return self.lex_hex_number(start);
        }

        if self.pos + 1 < self.input.len() && self.input[self.pos] == b'0'
            && (self.input[self.pos + 1] == b'b' || self.input[self.pos + 1] == b'B')
        {
            return self.lex_binary_number(start);
        }

        if let Some(tok) = self.lex_octal_number(start) {
            return tok;
        }

        self.lex_decimal_number(start)
    }

    /// Lex a hexadecimal integer or hex float literal (0x/0X prefix).
    fn lex_hex_number(&mut self, start: usize) -> Token {
        self.pos += 2;
        let hex_start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_hexdigit() {
            self.pos += 1;
        }

        // Check for hex float: 0x<digits>.<digits>p<exp> or 0x<digits>p<exp>
        let has_dot = self.pos < self.input.len() && self.input[self.pos] == b'.';
        let after_dot_has_p = if has_dot {
            let mut look = self.pos + 1;
            while look < self.input.len() && self.input[look].is_ascii_hexdigit() {
                look += 1;
            }
            look < self.input.len() && (self.input[look] == b'p' || self.input[look] == b'P')
        } else {
            false
        };
        let has_p = self.pos < self.input.len() && (self.input[self.pos] == b'p' || self.input[self.pos] == b'P');

        if has_dot && after_dot_has_p || has_p {
            return self.lex_hex_float(start, hex_start, has_dot);
        }

        // Regular hex integer
        let hex_str = std::str::from_utf8(&self.input[hex_start..self.pos]).unwrap_or("0");
        let value = u64::from_str_radix(hex_str, 16).unwrap_or(0);
        self.finish_int_literal(value, true, start)
    }

    /// Lex a hex float literal: 0x<int_hex>.<frac_hex>p<+/->exp
    fn lex_hex_float(&mut self, start: usize, hex_start: usize, has_dot: bool) -> Token {
        let int_hex = std::str::from_utf8(&self.input[hex_start..self.pos]).unwrap_or("0");

        let frac_hex = if has_dot {
            self.pos += 1; // skip '.'
            let frac_start = self.pos;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_hexdigit() {
                self.pos += 1;
            }
            std::str::from_utf8(&self.input[frac_start..self.pos]).unwrap_or("")
        } else {
            ""
        };

        // Parse 'p'/'P' exponent (mandatory for hex floats)
        let exp: i64 = if self.pos < self.input.len() && (self.input[self.pos] == b'p' || self.input[self.pos] == b'P') {
            self.pos += 1;
            let exp_neg = if self.pos < self.input.len() && self.input[self.pos] == b'-' {
                self.pos += 1;
                true
            } else {
                if self.pos < self.input.len() && self.input[self.pos] == b'+' {
                    self.pos += 1;
                }
                false
            };
            let exp_start = self.pos;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            let exp_str = std::str::from_utf8(&self.input[exp_start..self.pos]).unwrap_or("0");
            let e: i64 = exp_str.parse().unwrap_or(0);
            if exp_neg { -e } else { e }
        } else {
            0
        };

        // Convert hex float to f64: value = (int_part + frac_part) * 2^exp
        let int_val = u64::from_str_radix(int_hex, 16).unwrap_or(0) as f64;
        let frac_val: f64 = if !frac_hex.is_empty() {
            let frac_int = u64::from_str_radix(frac_hex, 16).unwrap_or(0) as f64;
            frac_int / (16.0_f64).powi(frac_hex.len() as i32)
        } else {
            0.0
        };
        let value = (int_val + frac_val) * (2.0_f64).powi(exp as i32);

        // Check float suffix: f/F = float, l/L = long double
        let float_kind = self.parse_simple_float_suffix();
        let span = Span::new(start as u32, self.pos as u32, self.file_id);
        match float_kind {
            1 => Token::new(TokenKind::FloatLiteralF32(value), span),
            2 => {
                let hex_text = std::str::from_utf8(&self.input[start..self.pos]).unwrap_or("0x0p0");
                let f128_bytes = crate::common::long_double::parse_long_double_to_f128_bytes(hex_text);
                Token::new(TokenKind::FloatLiteralLongDouble(value, f128_bytes), span)
            }
            _ => Token::new(TokenKind::FloatLiteral(value), span),
        }
    }

    /// Parse a simple float suffix (f/F → 1, l/L → 2, else 0). No imaginary handling.
    fn parse_simple_float_suffix(&mut self) -> u8 {
        if self.pos < self.input.len() && (self.input[self.pos] == b'f' || self.input[self.pos] == b'F') {
            self.pos += 1;
            1
        } else if self.pos < self.input.len() && (self.input[self.pos] == b'l' || self.input[self.pos] == b'L') {
            self.pos += 1;
            2
        } else {
            0
        }
    }

    /// Lex a binary integer literal (0b/0B prefix).
    fn lex_binary_number(&mut self, start: usize) -> Token {
        self.pos += 2;
        let bin_start = self.pos;
        while self.pos < self.input.len() && (self.input[self.pos] == b'0' || self.input[self.pos] == b'1') {
            self.pos += 1;
        }
        let bin_str = std::str::from_utf8(&self.input[bin_start..self.pos]).unwrap_or("0");
        let value = u64::from_str_radix(bin_str, 2).unwrap_or(0);
        self.finish_int_literal(value, true, start)
    }

    /// Try to lex an octal literal. Returns None if the token turns out to be decimal/float.
    fn lex_octal_number(&mut self, start: usize) -> Option<Token> {
        if self.input[self.pos] != b'0' || !self.peek_next().is_some_and(|c| c.is_ascii_digit()) {
            return None;
        }
        let saved_pos = self.pos;
        self.pos += 1;
        let oct_start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos] >= b'0' && self.input[self.pos] <= b'7' {
            self.pos += 1;
        }
        // Float indicator or non-octal digit → backtrack to decimal.
        // But '.' followed by '..' is ellipsis, not a decimal point — keep the octal.
        if self.pos < self.input.len() && matches!(self.input[self.pos], b'.' | b'e' | b'E' | b'8' | b'9') {
            let is_ellipsis = self.input[self.pos] == b'.'
                && self.pos + 2 < self.input.len()
                && self.input[self.pos + 1] == b'.'
                && self.input[self.pos + 2] == b'.';
            if !is_ellipsis {
                self.pos = saved_pos;
                return None;
            }
        }
        let oct_str = std::str::from_utf8(&self.input[oct_start..self.pos]).unwrap_or("0");
        let value = u64::from_str_radix(oct_str, 8).unwrap_or(0);
        Some(self.finish_int_literal(value, true, start))
    }

    /// Common integer literal finish: parse suffix, return token.
    fn finish_int_literal(&mut self, value: u64, is_hex_or_octal: bool, start: usize) -> Token {
        let (is_unsigned, is_long, is_long_long, is_imaginary) = self.parse_int_suffix();
        if is_imaginary {
            let span = Span::new(start as u32, self.pos as u32, self.file_id);
            return Token::new(TokenKind::ImaginaryLiteral(value as f64), span);
        }
        self.make_int_token(value, is_unsigned, is_long, is_long_long, is_hex_or_octal, start)
    }

    /// Lex a decimal integer or float literal.
    fn lex_decimal_number(&mut self, start: usize) -> Token {
        let mut is_float = false;
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
        }

        // Check for decimal point, but NOT if it's the start of '...' (ellipsis).
        // E.g. `2...15` from GCC case range `case 2 ... 15:` must lex as `2` `...` `15`,
        // not as float `2.` followed by invalid `..15`.
        if self.pos < self.input.len() && self.input[self.pos] == b'.'
            && !(self.pos + 2 < self.input.len() && self.input[self.pos + 1] == b'.' && self.input[self.pos + 2] == b'.')
        {
            is_float = true;
            self.pos += 1;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }

        if self.pos < self.input.len() && (self.input[self.pos] == b'e' || self.input[self.pos] == b'E') {
            is_float = true;
            self.pos += 1;
            if self.pos < self.input.len() && (self.input[self.pos] == b'+' || self.input[self.pos] == b'-') {
                self.pos += 1;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }

        // Save the end position of the number digits before parsing suffixes
        // (which advance self.pos further).
        let num_end = self.pos;

        if is_float {
            let (float_kind, is_imaginary) = self.parse_float_suffix();
            // Borrow the digit text as &str without allocating a String.
            let text = std::str::from_utf8(&self.input[start..num_end]).unwrap_or("0");
            self.make_float_token(text, float_kind, is_imaginary, start)
        } else {
            // Parse the integer value directly from the &str borrow (no heap allocation).
            let text = std::str::from_utf8(&self.input[start..num_end]).unwrap_or("0");
            let uvalue: u64 = text.parse().unwrap_or(0);
            self.finish_int_literal(uvalue, false, start)
        }
    }

    /// Parse float suffix with imaginary support (GCC extension).
    /// Returns (float_kind, is_imaginary) where float_kind: 0=double, 1=float, 2=long double.
    fn parse_float_suffix(&mut self) -> (u8, bool) {
        let mut is_imaginary = false;
        let float_kind = if self.pos < self.input.len() && (self.input[self.pos] == b'f' || self.input[self.pos] == b'F') {
            self.pos += 1;
            if self.pos < self.input.len() && (self.input[self.pos] == b'i' || self.input[self.pos] == b'I') {
                self.pos += 1;
                is_imaginary = true;
            }
            1
        } else if self.pos < self.input.len() && (self.input[self.pos] == b'l' || self.input[self.pos] == b'L') {
            self.pos += 1;
            if self.pos < self.input.len() && (self.input[self.pos] == b'i' || self.input[self.pos] == b'I') {
                self.pos += 1;
                is_imaginary = true;
            }
            2
        } else if self.pos < self.input.len() && (self.input[self.pos] == b'i' || self.input[self.pos] == b'I') {
            self.pos += 1;
            is_imaginary = true;
            if self.pos < self.input.len() && (self.input[self.pos] == b'f' || self.input[self.pos] == b'F') {
                self.pos += 1;
                1
            } else if self.pos < self.input.len() && (self.input[self.pos] == b'l' || self.input[self.pos] == b'L') {
                self.pos += 1;
                2
            } else {
                0
            }
        } else {
            0
        };
        // Also consume trailing 'j'/'J' suffix (C99/GCC alternative for imaginary)
        if !is_imaginary && self.pos < self.input.len() && (self.input[self.pos] == b'j' || self.input[self.pos] == b'J') {
            self.pos += 1;
            is_imaginary = true;
        }
        (float_kind, is_imaginary)
    }

    /// Construct a float/imaginary token from parsed components.
    fn make_float_token(&self, text: &str, float_kind: u8, is_imaginary: bool, start: usize) -> Token {
        let value: f64 = text.parse().unwrap_or(0.0);
        let span = Span::new(start as u32, self.pos as u32, self.file_id);
        if is_imaginary {
            match float_kind {
                1 => Token::new(TokenKind::ImaginaryLiteralF32(value), span),
                2 => {
                    let f128_bytes = crate::common::long_double::parse_long_double_to_f128_bytes(text);
                    Token::new(TokenKind::ImaginaryLiteralLongDouble(value, f128_bytes), span)
                }
                _ => Token::new(TokenKind::ImaginaryLiteral(value), span),
            }
        } else {
            match float_kind {
                1 => Token::new(TokenKind::FloatLiteralF32(value), span),
                2 => {
                    let f128_bytes = crate::common::long_double::parse_long_double_to_f128_bytes(text);
                    Token::new(TokenKind::FloatLiteralLongDouble(value, f128_bytes), span)
                }
                _ => Token::new(TokenKind::FloatLiteral(value), span),
            }
        }
    }

    /// Parse integer suffix and return (is_unsigned, is_long, is_imaginary).
    /// is_long is true for l/L or ll/LL suffixes.
    /// is_imaginary is true for trailing 'i' suffix (GCC extension: 5i).
    /// Parse integer suffix and return (is_unsigned, is_long, is_long_long, is_imaginary).
    /// is_long is true for single l/L suffix. is_long_long is true for ll/LL suffix.
    fn parse_int_suffix(&mut self) -> (bool, bool, bool, bool) {
        let mut is_imaginary = false;
        // First check for standalone 'i'/'I' imaginary suffix (GCC extension: 5i, 5I)
        // Must check this before the main loop since 'i'/'I' alone means imaginary, not a regular suffix
        if self.pos < self.input.len() && (self.input[self.pos] == b'i' || self.input[self.pos] == b'I') {
            // Check it's not the start of an identifier (like 'int')
            let next = if self.pos + 1 < self.input.len() { self.input[self.pos + 1] } else { 0 };
            if !next.is_ascii_alphanumeric() && next != b'_' {
                self.pos += 1; // consume 'i'/'I' as imaginary suffix
                return (false, false, false, true);
            }
        }

        let mut is_unsigned = false;
        let mut is_long = false;
        let mut is_long_long = false;
        loop {
            if self.pos < self.input.len() && (self.input[self.pos] == b'u' || self.input[self.pos] == b'U') {
                is_unsigned = true;
                self.pos += 1;
            } else if self.pos < self.input.len() && (self.input[self.pos] == b'l' || self.input[self.pos] == b'L') {
                self.pos += 1;
                // Check for second l/L for ll/LL
                if self.pos < self.input.len() && (self.input[self.pos] == b'l' || self.input[self.pos] == b'L') {
                    is_long_long = true;
                    self.pos += 1;
                } else {
                    is_long = true;
                }
            } else {
                break;
            }
        }
        // Consume trailing 'i'/'I'/'j'/'J' for GCC imaginary suffix (e.g., 5li, 5ui, 5ULi, 5I)
        if self.pos < self.input.len() && (self.input[self.pos] == b'i' || self.input[self.pos] == b'I' || self.input[self.pos] == b'j' || self.input[self.pos] == b'J') {
            let next = if self.pos + 1 < self.input.len() { self.input[self.pos + 1] } else { 0 };
            if !next.is_ascii_alphanumeric() && next != b'_' {
                self.pos += 1;
                is_imaginary = true;
            }
        }
        (is_unsigned, is_long, is_long_long, is_imaginary)
    }

    /// Create the appropriate token kind based on integer value, suffix, and base info.
    /// For hex/octal literals, C promotes: int -> unsigned int -> long -> unsigned long -> long long -> unsigned long long.
    /// For decimal literals: int -> long -> long long (no implicit unsigned).
    fn make_int_token(&self, value: u64, is_unsigned: bool, is_long: bool, is_long_long: bool, is_hex_or_octal: bool, start: usize) -> Token {
        let span = Span::new(start as u32, self.pos as u32, self.file_id);
        if is_unsigned && is_long_long {
            // Explicit ULL suffix: always unsigned long long (64-bit)
            Token::new(TokenKind::ULongLongLiteral(value), span)
        } else if is_unsigned && is_long {
            // Explicit UL suffix: unsigned long
            Token::new(TokenKind::ULongLiteral(value), span)
        } else if is_unsigned {
            if value > u32::MAX as u64 {
                Token::new(TokenKind::ULongLiteral(value), span)
            } else {
                Token::new(TokenKind::UIntLiteral(value), span)
            }
        } else if is_long_long {
            // Explicit LL suffix: always long long (64-bit)
            if is_hex_or_octal && value > i64::MAX as u64 {
                Token::new(TokenKind::ULongLongLiteral(value), span)
            } else {
                Token::new(TokenKind::LongLongLiteral(value as i64), span)
            }
        } else if is_long {
            // Explicit L suffix: C11 6.4.4.1 type promotion for hex/octal:
            //   long -> unsigned long -> long long -> unsigned long long
            // On ILP32, long is 32-bit so values > i32::MAX need promotion.
            if is_hex_or_octal && crate::common::types::target_is_32bit() {
                if value <= i32::MAX as u64 {
                    Token::new(TokenKind::LongLiteral(value as i64), span)
                } else if value <= u32::MAX as u64 {
                    Token::new(TokenKind::ULongLiteral(value), span)
                } else if value <= i64::MAX as u64 {
                    Token::new(TokenKind::LongLongLiteral(value as i64), span)
                } else {
                    Token::new(TokenKind::ULongLongLiteral(value), span)
                }
            } else if is_hex_or_octal && value > i64::MAX as u64 {
                // LP64: long is 64-bit, value exceeds signed range -> unsigned long
                Token::new(TokenKind::ULongLiteral(value), span)
            } else {
                Token::new(TokenKind::LongLiteral(value as i64), span)
            }
        } else if is_hex_or_octal {
            // Hex/octal: int -> unsigned int -> long -> unsigned long
            if value <= i32::MAX as u64 {
                Token::new(TokenKind::IntLiteral(value as i64), span)
            } else if value <= u32::MAX as u64 {
                Token::new(TokenKind::UIntLiteral(value), span)
            } else if value <= i64::MAX as u64 {
                Token::new(TokenKind::LongLiteral(value as i64), span)
            } else {
                Token::new(TokenKind::ULongLiteral(value), span)
            }
        } else {
            // Decimal with no suffix: C11 6.4.4.1 Table 6
            // Type sequence: int -> long int -> long long int
            if value > i64::MAX as u64 {
                // Doesn't fit in any signed type; implementation-defined, use unsigned long
                Token::new(TokenKind::ULongLiteral(value), span)
            } else if crate::common::types::target_is_32bit() {
                // ILP32: int (32) -> long (32) -> long long (64)
                if value <= i32::MAX as u64 {
                    Token::new(TokenKind::IntLiteral(value as i64), span)
                } else {
                    // Doesn't fit in int or long (both 32-bit), promote to long long
                    Token::new(TokenKind::LongLongLiteral(value as i64), span)
                }
            } else {
                // LP64: int (32) -> long (64)
                if value <= i32::MAX as u64 {
                    Token::new(TokenKind::IntLiteral(value as i64), span)
                } else {
                    // Doesn't fit in int, promote to long
                    Token::new(TokenKind::LongLiteral(value as i64), span)
                }
            }
        }
    }

    fn lex_string(&mut self, start: usize) -> Token {
        self.pos += 1; // skip opening "
        let mut s = String::new();
        while self.pos < self.input.len() && self.input[self.pos] != b'"' {
            if self.input[self.pos] == b'\\' {
                self.pos += 1;
                if self.pos < self.input.len() {
                    let ch = self.lex_escape_char();
                    // C narrow strings store raw bytes, so Unicode escapes (\u, \U)
                    // must be UTF-8 encoded to match GCC/Clang behavior.
                    if (ch as u32) > 0xFF {
                        let mut buf = [0u8; 4];
                        let utf8 = ch.encode_utf8(&mut buf);
                        for byte in utf8.bytes() {
                            s.push(byte as char);
                        }
                    } else {
                        s.push(ch);
                    }
                }
            } else {
                // Decode PUA-encoded bytes back to original values for
                // non-UTF-8 source files (e.g., EUC-JP string literals)
                let (byte, consumed) = decode_pua_byte(&self.input, self.pos);
                s.push(byte as char);
                self.pos += consumed;
            }
        }
        if self.pos < self.input.len() {
            self.pos += 1; // skip closing "
        }
        Token::new(TokenKind::StringLiteral(s), Span::new(start as u32, self.pos as u32, self.file_id))
    }

    fn lex_wide_string(&mut self, start: usize) -> Token {
        self.pos += 1; // skip opening "
        let mut s = String::new();
        while self.pos < self.input.len() && self.input[self.pos] != b'"' {
            if self.input[self.pos] == b'\\' {
                self.pos += 1;
                if self.pos < self.input.len() {
                    // Wide strings store code points directly (no UTF-8 encoding needed)
                    s.push(self.lex_escape_char());
                }
            } else {
                // Check for PUA-encoded bytes first (from non-UTF-8 source files)
                let (byte, consumed) = decode_pua_byte(&self.input, self.pos);
                if consumed > 1 {
                    // PUA byte: decode back to original byte value
                    s.push(byte as char);
                    self.pos += consumed;
                } else {
                    // Decode UTF-8 character for wide string
                    let byte = self.input[self.pos];
                    if byte < 0x80 {
                        s.push(byte as char);
                        self.pos += 1;
                    } else {
                        // Multi-byte UTF-8: decode to a single Unicode code point
                        let remaining = &self.input[self.pos..];
                        let end = std::cmp::min(4, remaining.len());
                        if let Ok(text) = std::str::from_utf8(&remaining[..end]) {
                            if let Some(ch) = text.chars().next() {
                                s.push(ch);
                                self.pos += ch.len_utf8();
                            } else {
                                s.push(byte as char);
                                self.pos += 1;
                            }
                        } else if let Ok(text) = std::str::from_utf8(remaining) {
                            if let Some(ch) = text.chars().next() {
                                s.push(ch);
                                self.pos += ch.len_utf8();
                            } else {
                                s.push(byte as char);
                                self.pos += 1;
                            }
                        } else {
                            s.push(byte as char);
                            self.pos += 1;
                        }
                    }
                }
            }
        }
        if self.pos < self.input.len() {
            self.pos += 1; // skip closing "
        }
        Token::new(TokenKind::WideStringLiteral(s), Span::new(start as u32, self.pos as u32, self.file_id))
    }

    /// Lex a u"..." char16_t string literal. Same parsing as wide string but produces
    /// Char16StringLiteral token. The Rust String stores Unicode chars; the downstream
    /// pipeline converts each to a u16 value (truncating code points > 0xFFFF).
    fn lex_char16_string(&mut self, start: usize) -> Token {
        self.pos += 1; // skip opening "
        let mut s = String::new();
        while self.pos < self.input.len() && self.input[self.pos] != b'"' {
            if self.input[self.pos] == b'\\' {
                self.pos += 1;
                if self.pos < self.input.len() {
                    s.push(self.lex_escape_char());
                }
            } else {
                // Check for PUA-encoded bytes first
                let (byte, consumed) = decode_pua_byte(&self.input, self.pos);
                if consumed > 1 {
                    s.push(byte as char);
                    self.pos += consumed;
                } else {
                    let byte = self.input[self.pos];
                    if byte < 0x80 {
                        s.push(byte as char);
                        self.pos += 1;
                    } else {
                        let remaining = &self.input[self.pos..];
                        let end = std::cmp::min(4, remaining.len());
                        if let Ok(text) = std::str::from_utf8(&remaining[..end]) {
                            if let Some(ch) = text.chars().next() {
                                s.push(ch);
                                self.pos += ch.len_utf8();
                            } else {
                                s.push(byte as char);
                                self.pos += 1;
                            }
                        } else if let Ok(text) = std::str::from_utf8(remaining) {
                            if let Some(ch) = text.chars().next() {
                                s.push(ch);
                                self.pos += ch.len_utf8();
                            } else {
                                s.push(byte as char);
                                self.pos += 1;
                            }
                        } else {
                            s.push(byte as char);
                            self.pos += 1;
                        }
                    }
                }
            }
        }
        if self.pos < self.input.len() {
            self.pos += 1; // skip closing "
        }
        Token::new(TokenKind::Char16StringLiteral(s), Span::new(start as u32, self.pos as u32, self.file_id))
    }

    fn lex_wide_char(&mut self, start: usize) -> Token {
        self.pos += 1; // skip opening '
        let mut value: u32 = 0;
        if self.pos < self.input.len() && self.input[self.pos] != b'\'' {
            if self.input[self.pos] == b'\\' {
                self.pos += 1;
                let ch = self.lex_escape_char();
                value = ch as u32; // Unicode escapes return code point directly
            } else {
                // Check for PUA-encoded bytes first
                let (byte, consumed) = decode_pua_byte(&self.input, self.pos);
                if consumed > 1 {
                    value = byte as u32;
                    self.pos += consumed;
                } else {
                    // Decode UTF-8 to get Unicode code point
                    let byte = self.input[self.pos];
                    if byte < 0x80 {
                        value = byte as u32;
                        self.pos += 1;
                    } else {
                        let remaining = &self.input[self.pos..];
                        let end = std::cmp::min(remaining.len(), 4);
                        if let Ok(text) = std::str::from_utf8(&remaining[..end]) {
                            if let Some(ch) = text.chars().next() {
                                value = ch as u32;
                                self.pos += ch.len_utf8();
                            }
                        } else if let Ok(text) = std::str::from_utf8(remaining) {
                            if let Some(ch) = text.chars().next() {
                                value = ch as u32;
                                self.pos += ch.len_utf8();
                            }
                        } else {
                            value = byte as u32;
                            self.pos += 1;
                        }
                    }
                }
            }
        }
        // Skip any remaining chars until closing quote
        while self.pos < self.input.len() && self.input[self.pos] != b'\'' {
            self.pos += 1;
        }
        if self.pos < self.input.len() && self.input[self.pos] == b'\'' {
            self.pos += 1; // skip closing '
        }
        let span = Span::new(start as u32, self.pos as u32, self.file_id);
        // Wide char literals have type int (wchar_t)
        Token::new(TokenKind::IntLiteral(value as i64), span)
    }

    fn lex_char(&mut self, start: usize) -> Token {
        self.pos += 1; // skip opening '
        let mut value: i32 = 0;
        let mut char_count = 0;
        while self.pos < self.input.len() && self.input[self.pos] != b'\'' {
            let ch = if self.input[self.pos] == b'\\' {
                self.pos += 1;
                self.lex_escape_char()
            } else {
                // Decode PUA-encoded bytes for non-UTF-8 source files
                let (byte, consumed) = decode_pua_byte(&self.input, self.pos);
                self.pos += consumed;
                byte as char
            };
            // C narrow char literals encode Unicode escapes as UTF-8 bytes
            // combined into a multi-byte int value, matching GCC behavior.
            if (ch as u32) > 0xFF {
                let mut buf = [0u8; 4];
                let utf8 = ch.encode_utf8(&mut buf);
                for byte in utf8.bytes() {
                    value = (value << 8) | (byte as i32);
                    char_count += 1;
                }
            } else {
                // Multi-character constant: shift previous value and add new byte
                value = (value << 8) | (ch as u8 as i32);
                char_count += 1;
            }
        }
        if self.pos < self.input.len() && self.input[self.pos] == b'\'' {
            self.pos += 1; // skip closing '
        }
        let span = Span::new(start as u32, self.pos as u32, self.file_id);
        if char_count <= 1 {
            // Single character: use CharLiteral with the char value
            let ch = if value == 0 { '\0' } else { (value as u8) as char };
            Token::new(TokenKind::CharLiteral(ch), span)
        } else {
            // Multi-character constant: produce an IntLiteral with the combined value
            Token::new(TokenKind::IntLiteral(value as i64), span)
        }
    }

    fn lex_escape_char(&mut self) -> char {
        if self.pos >= self.input.len() {
            return '\0';
        }
        let ch = self.input[self.pos];
        self.pos += 1;
        match ch {
            b'n' => '\n',
            b't' => '\t',
            b'r' => '\r',
            b'\\' => '\\',
            b'\'' => '\'',
            b'"' => '"',
            b'a' => '\x07',
            b'b' => '\x08',
            b'e' | b'E' => '\x1b', // GNU extension: ESC
            b'f' => '\x0c',
            b'v' => '\x0b',
            b'x' => {
                // Hex escape: \xNN - consumes all hex digits, value truncated to byte
                let mut val = 0u32;
                while self.pos < self.input.len() && self.input[self.pos].is_ascii_hexdigit() {
                    val = val * 16 + hex_digit_val(self.input[self.pos]) as u32;
                    self.pos += 1;
                }
                // Truncate to byte value and use direct char mapping to avoid
                // multi-byte UTF-8 encoding issues with values > 127
                (val as u8) as char
            }
            b'u' => {
                // Universal character name: \uNNNN (exactly 4 hex digits)
                self.lex_unicode_escape(4)
            }
            b'U' => {
                // Universal character name: \UNNNNNNNN (exactly 8 hex digits)
                self.lex_unicode_escape(8)
            }
            b'0'..=b'7' => {
                // Octal escape: \0 through \377 (1-3 octal digits)
                // Note: \0 alone produces null; \040 produces space (32), etc.
                let mut val = (ch - b'0') as u32;
                for _ in 0..2 {
                    if self.pos < self.input.len() && self.input[self.pos] >= b'0' && self.input[self.pos] <= b'7' {
                        val = val * 8 + (self.input[self.pos] - b'0') as u32;
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                // Truncate to byte value to match C semantics
                (val as u8) as char
            }
            _ => ch as char,
        }
    }

    /// Parse a universal character name (\uNNNN or \UNNNNNNNN).
    /// `num_digits` is 4 for \u or 8 for \U.
    /// Returns the Unicode code point as a Rust char.
    // TODO: C11 requires exactly num_digits hex digits; emit diagnostic if fewer provided.
    // TODO: C11 §6.4.3 disallows certain code points (below 0x00A0 except 0x24/0x40/0x60,
    //       and surrogates 0xD800-0xDFFF). Validate and emit diagnostics for these.
    fn lex_unicode_escape(&mut self, num_digits: usize) -> char {
        let mut val = 0u32;
        for _ in 0..num_digits {
            if self.pos < self.input.len() && self.input[self.pos].is_ascii_hexdigit() {
                val = val * 16 + hex_digit_val(self.input[self.pos]) as u32;
                self.pos += 1;
            } else {
                break;
            }
        }
        // TODO: Emit a diagnostic for invalid code points (e.g. surrogates) instead of
        //       silently using the replacement character.
        char::from_u32(val).unwrap_or('\u{FFFD}')
    }

    fn lex_identifier(&mut self, start: usize) -> Token {
        while self.pos < self.input.len() && (self.input[self.pos] == b'_' || self.input[self.pos] == b'$' || self.input[self.pos].is_ascii_alphanumeric()) {
            self.pos += 1;
        }

        // Check for wide/unicode char/string prefixes: L'x', L"...", u'x', u"...", U'x', U"..."
        if self.pos < self.input.len() {
            let text_len = self.pos - start;
            let next = self.input[self.pos];
            if next == b'\'' || next == b'"' {
                let prefix = &self.input[start..self.pos];
                let is_wide_prefix = match text_len {
                    1 => prefix[0] == b'L' || prefix[0] == b'u' || prefix[0] == b'U',
                    2 => prefix == b"u8",
                    _ => false,
                };
                if is_wide_prefix {
                    if next == b'\'' {
                        // Wide/unicode char literal: value is the Unicode code point
                        return self.lex_wide_char(start);
                    } else {
                        // Wide/unicode string literal: L"...", u"...", U"...", u8"..."
                        let is_wide_32 = text_len == 1 && (prefix[0] == b'L' || prefix[0] == b'U');
                        let is_char16 = text_len == 1 && prefix[0] == b'u';
                        if is_wide_32 {
                            return self.lex_wide_string(start);
                        } else if is_char16 {
                            // u"..." - char16_t string (16-bit elements)
                            return self.lex_char16_string(start);
                        } else {
                            // u8"..." - UTF-8 string, same as narrow string
                            return self.lex_string(start);
                        }
                    }
                }
            }
        }

        let text = std::str::from_utf8(&self.input[start..self.pos]).unwrap_or("");
        let span = Span::new(start as u32, self.pos as u32, self.file_id);

        // Check for synthetic pragma pack directives emitted by preprocessor
        if let Some(pack_tok) = Self::try_pragma_pack_token(text) {
            return Token::new(pack_tok, span);
        }

        // Check for synthetic pragma visibility directives emitted by preprocessor
        if let Some(vis_tok) = Self::try_pragma_visibility_token(text) {
            return Token::new(vis_tok, span);
        }

        if let Some(kw) = TokenKind::from_keyword(text, self.gnu_extensions) {
            Token::new(kw, span)
        } else {
            Token::new(TokenKind::Identifier(Rc::from(text)), span)
        }
    }

    /// Recognize synthetic pragma pack identifiers emitted by the preprocessor.
    /// Format: __ccc_pack_set_N, __ccc_pack_push_N, __ccc_pack_push_only,
    ///         __ccc_pack_pop, __ccc_pack_reset
    fn try_pragma_pack_token(text: &str) -> Option<TokenKind> {
        if let Some(rest) = text.strip_prefix("__ccc_pack_") {
            if rest == "pop" {
                Some(TokenKind::PragmaPackPop)
            } else if rest == "reset" {
                Some(TokenKind::PragmaPackReset)
            } else if rest == "push_only" {
                Some(TokenKind::PragmaPackPushOnly)
            } else if let Some(n_str) = rest.strip_prefix("set_") {
                if let Ok(n) = n_str.parse::<usize>() {
                    Some(TokenKind::PragmaPackSet(n))
                } else {
                    None
                }
            } else if let Some(n_str) = rest.strip_prefix("push_") {
                if let Ok(n) = n_str.parse::<usize>() {
                    Some(TokenKind::PragmaPackPush(n))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Recognize synthetic pragma visibility identifiers emitted by the preprocessor.
    /// Format: __ccc_visibility_push_VISIBILITY, __ccc_visibility_pop
    fn try_pragma_visibility_token(text: &str) -> Option<TokenKind> {
        if let Some(rest) = text.strip_prefix("__ccc_visibility_") {
            if rest == "pop" {
                Some(TokenKind::PragmaVisibilityPop)
            } else if let Some(vis) = rest.strip_prefix("push_") {
                match vis {
                    "hidden" | "default" | "protected" | "internal" => {
                        Some(TokenKind::PragmaVisibilityPush(vis.to_string()))
                    }
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    fn lex_punctuation(&mut self, start: usize) -> Token {
        let ch = self.input[self.pos];
        self.pos += 1;

        let kind = match ch {
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b';' => TokenKind::Semicolon,
            b',' => TokenKind::Comma,
            b'~' => TokenKind::Tilde,
            b'?' => TokenKind::Question,
            b':' => TokenKind::Colon,
            b'#' => {
                if self.pos < self.input.len() && self.input[self.pos] == b'#' {
                    self.pos += 1;
                    TokenKind::HashHash
                } else {
                    TokenKind::Hash
                }
            }
            b'.' => {
                if self.pos + 1 < self.input.len() && self.input[self.pos] == b'.' && self.input[self.pos + 1] == b'.' {
                    self.pos += 2;
                    TokenKind::Ellipsis
                } else {
                    TokenKind::Dot
                }
            }
            b'+' => {
                if self.pos < self.input.len() {
                    match self.input[self.pos] {
                        b'+' => { self.pos += 1; TokenKind::PlusPlus }
                        b'=' => { self.pos += 1; TokenKind::PlusAssign }
                        _ => TokenKind::Plus,
                    }
                } else {
                    TokenKind::Plus
                }
            }
            b'-' => {
                if self.pos < self.input.len() {
                    match self.input[self.pos] {
                        b'-' => { self.pos += 1; TokenKind::MinusMinus }
                        b'=' => { self.pos += 1; TokenKind::MinusAssign }
                        b'>' => { self.pos += 1; TokenKind::Arrow }
                        _ => TokenKind::Minus,
                    }
                } else {
                    TokenKind::Minus
                }
            }
            b'*' => {
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::StarAssign
                } else {
                    TokenKind::Star
                }
            }
            b'/' => {
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::SlashAssign
                } else {
                    TokenKind::Slash
                }
            }
            b'%' => {
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::PercentAssign
                } else {
                    TokenKind::Percent
                }
            }
            b'&' => {
                if self.pos < self.input.len() {
                    match self.input[self.pos] {
                        b'&' => { self.pos += 1; TokenKind::AmpAmp }
                        b'=' => { self.pos += 1; TokenKind::AmpAssign }
                        _ => TokenKind::Amp,
                    }
                } else {
                    TokenKind::Amp
                }
            }
            b'|' => {
                if self.pos < self.input.len() {
                    match self.input[self.pos] {
                        b'|' => { self.pos += 1; TokenKind::PipePipe }
                        b'=' => { self.pos += 1; TokenKind::PipeAssign }
                        _ => TokenKind::Pipe,
                    }
                } else {
                    TokenKind::Pipe
                }
            }
            b'^' => {
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::CaretAssign
                } else {
                    TokenKind::Caret
                }
            }
            b'!' => {
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::BangEqual
                } else {
                    TokenKind::Bang
                }
            }
            b'=' => {
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::EqualEqual
                } else {
                    TokenKind::Assign
                }
            }
            b'<' => {
                if self.pos < self.input.len() {
                    match self.input[self.pos] {
                        b'<' => {
                            self.pos += 1;
                            if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                                self.pos += 1;
                                TokenKind::LessLessAssign
                            } else {
                                TokenKind::LessLess
                            }
                        }
                        b'=' => { self.pos += 1; TokenKind::LessEqual }
                        _ => TokenKind::Less,
                    }
                } else {
                    TokenKind::Less
                }
            }
            b'>' => {
                if self.pos < self.input.len() {
                    match self.input[self.pos] {
                        b'>' => {
                            self.pos += 1;
                            if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                                self.pos += 1;
                                TokenKind::GreaterGreaterAssign
                            } else {
                                TokenKind::GreaterGreater
                            }
                        }
                        b'=' => { self.pos += 1; TokenKind::GreaterEqual }
                        _ => TokenKind::Greater,
                    }
                } else {
                    TokenKind::Greater
                }
            }
            _ => {
                // Non-ASCII or unknown character: skip any remaining bytes of
                // a multi-byte UTF-8 sequence (including PUA-encoded bytes from
                // non-UTF-8 source files) and continue tokenizing.
                // TODO: emit a diagnostic for genuinely unknown/invalid characters
                while self.pos < self.input.len() && (self.input[self.pos] & 0xC0) == 0x80 {
                    self.pos += 1; // skip continuation bytes
                }
                return self.next_token();
            }
        };

        Token::new(kind, Span::new(start as u32, self.pos as u32, self.file_id))
    }
}

fn hex_digit_val(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0,
    }
}
