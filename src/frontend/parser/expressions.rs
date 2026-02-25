// Expression parsing: precedence climbing from comma expression down to primary.
//
// The expression parser implements C's operator precedence through a table-driven
// approach. Binary operators (logical, bitwise, relational, arithmetic) are handled
// by a shared `parse_binary_expr` method parameterized on `PrecedenceLevel`, which
// maps tokens to operators and recurses to the next-tighter level. This eliminates
// the repetitive per-level parsing functions while keeping the code clear.
//
// Call hierarchy (loosest to tightest binding):
//   parse_expr -> parse_assignment_expr -> parse_conditional_expr
//   -> parse_binary_expr(LogicalOr) -> ... -> parse_binary_expr(Multiplicative)
//   -> parse_cast_expr -> parse_unary_expr -> parse_postfix_expr
//   -> parse_primary_expr

use crate::frontend::lexer::token::TokenKind;
use std::rc::Rc;
use super::ast::*;
use super::parse::Parser;

/// C operator precedence levels (loosest to tightest binding).
/// Used by the table-driven binary expression parser.
#[derive(Debug, Clone, Copy)]
enum PrecedenceLevel {
    LogicalOr,
    LogicalAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseAnd,
    Equality,
    Relational,
    Shift,
    Additive,
    Multiplicative,
}

impl Parser {
    /// Consume any pending `__attribute__((vector_size(N)))` or `__attribute__((ext_vector_type(N)))`
    /// that was parsed during a type specifier and wrap the given TypeSpecifier in a
    /// `TypeSpecifier::Vector` node.  This is needed for casts, compound literals, sizeof,
    /// and _Alignof where the vector_size attribute is not captured in a Declaration.
    pub(super) fn apply_pending_vector_attr(&mut self, ts: TypeSpecifier) -> TypeSpecifier {
        if let Some(total_bytes) = self.attrs.parsing_vector_size.take() {
            return TypeSpecifier::Vector(Box::new(ts), total_bytes);
        }
        if let Some(nelem) = self.attrs.parsing_ext_vector_nelem.take() {
            // Compute total bytes from element count * element size
            let elem_size = self.estimate_type_size(&ts);
            return TypeSpecifier::Vector(Box::new(ts), nelem * elem_size);
        }
        ts
    }

    /// Rough size estimate for a scalar type (for ext_vector_type element size computation).
    /// TODO: Long/UnsignedLong are 4 bytes on i686 and LongDouble is 12 on i686;
    /// this currently hardcodes 64-bit sizes. Only affects ext_vector_type (not vector_size).
    fn estimate_type_size(&self, ts: &TypeSpecifier) -> usize {
        match ts {
            TypeSpecifier::Char | TypeSpecifier::UnsignedChar | TypeSpecifier::Bool => 1,
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => 2,
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Signed | TypeSpecifier::Unsigned => 4,
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong => 8, // 64-bit default
            TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong => 8,
            TypeSpecifier::Float => 4,
            TypeSpecifier::Double => 8,
            TypeSpecifier::LongDouble => 16,
            _ => 4, // fallback
        }
    }

    pub(super) fn parse_expr(&mut self) -> Expr {
        let lhs = self.parse_assignment_expr();
        if matches!(self.peek(), TokenKind::Comma) {
            let span = self.peek_span();
            self.advance();
            let rhs = self.parse_expr();
            Expr::Comma(Box::new(lhs), Box::new(rhs), span)
        } else {
            lhs
        }
    }

    pub(super) fn parse_assignment_expr(&mut self) -> Expr {
        let lhs = self.parse_conditional_expr();

        match self.peek() {
            TokenKind::Assign => {
                let span = self.peek_span();
                self.advance();
                let rhs = self.parse_assignment_expr();
                Expr::Assign(Box::new(lhs), Box::new(rhs), span)
            }
            _ => {
                if let Some(op) = self.compound_assign_op() {
                    let span = self.peek_span();
                    self.advance();
                    let rhs = self.parse_assignment_expr();
                    Expr::CompoundAssign(op, Box::new(lhs), Box::new(rhs), span)
                } else {
                    lhs
                }
            }
        }
    }

    fn parse_conditional_expr(&mut self) -> Expr {
        let cond = self.parse_binary_expr(PrecedenceLevel::LogicalOr);
        if self.consume_if(&TokenKind::Question) {
            let span = cond.span();
            // GNU extension: `cond ? : else_expr` (omitted middle operand)
            // Condition is evaluated once and used as the then-value if truthy.
            if self.peek() == &TokenKind::Colon {
                self.expect_context(&TokenKind::Colon, "in conditional expression");
                let else_expr = self.parse_conditional_expr();
                Expr::GnuConditional(Box::new(cond), Box::new(else_expr), span)
            } else {
                let then_expr = self.parse_expr();
                self.expect_context(&TokenKind::Colon, "in conditional expression");
                let else_expr = self.parse_conditional_expr();
                Expr::Conditional(Box::new(cond), Box::new(then_expr), Box::new(else_expr), span)
            }
        } else {
            cond
        }
    }

    /// Map a token to a binary operator at the current precedence level, if applicable.
    fn token_to_binop(&self, token: &TokenKind, level: PrecedenceLevel) -> Option<BinOp> {
        match (token, level) {
            (TokenKind::PipePipe, PrecedenceLevel::LogicalOr) => Some(BinOp::LogicalOr),
            (TokenKind::AmpAmp, PrecedenceLevel::LogicalAnd) => Some(BinOp::LogicalAnd),
            (TokenKind::Pipe, PrecedenceLevel::BitwiseOr) => Some(BinOp::BitOr),
            (TokenKind::Caret, PrecedenceLevel::BitwiseXor) => Some(BinOp::BitXor),
            (TokenKind::Amp, PrecedenceLevel::BitwiseAnd) => Some(BinOp::BitAnd),
            (TokenKind::EqualEqual, PrecedenceLevel::Equality) => Some(BinOp::Eq),
            (TokenKind::BangEqual, PrecedenceLevel::Equality) => Some(BinOp::Ne),
            (TokenKind::Less, PrecedenceLevel::Relational) => Some(BinOp::Lt),
            (TokenKind::LessEqual, PrecedenceLevel::Relational) => Some(BinOp::Le),
            (TokenKind::Greater, PrecedenceLevel::Relational) => Some(BinOp::Gt),
            (TokenKind::GreaterEqual, PrecedenceLevel::Relational) => Some(BinOp::Ge),
            (TokenKind::LessLess, PrecedenceLevel::Shift) => Some(BinOp::Shl),
            (TokenKind::GreaterGreater, PrecedenceLevel::Shift) => Some(BinOp::Shr),
            (TokenKind::Plus, PrecedenceLevel::Additive) => Some(BinOp::Add),
            (TokenKind::Minus, PrecedenceLevel::Additive) => Some(BinOp::Sub),
            (TokenKind::Star, PrecedenceLevel::Multiplicative) => Some(BinOp::Mul),
            (TokenKind::Slash, PrecedenceLevel::Multiplicative) => Some(BinOp::Div),
            (TokenKind::Percent, PrecedenceLevel::Multiplicative) => Some(BinOp::Mod),
            _ => None,
        }
    }

    /// Parse a left-associative binary expression at the given precedence level.
    /// This is the shared core that replaces 10 nearly-identical parsing functions.
    fn parse_binary_expr(&mut self, level: PrecedenceLevel) -> Expr {
        let mut lhs = self.parse_next_tighter(level);
        while let Some(op) = self.token_to_binop(self.peek(), level) {
            let span = self.peek_span();
            self.advance();
            let rhs = self.parse_next_tighter(level);
            lhs = Expr::BinaryOp(op, Box::new(lhs), Box::new(rhs), span);
        }
        lhs
    }

    /// Parse the next tighter precedence level.
    fn parse_next_tighter(&mut self, level: PrecedenceLevel) -> Expr {
        match level {
            PrecedenceLevel::LogicalOr => self.parse_binary_expr(PrecedenceLevel::LogicalAnd),
            PrecedenceLevel::LogicalAnd => self.parse_binary_expr(PrecedenceLevel::BitwiseOr),
            PrecedenceLevel::BitwiseOr => self.parse_binary_expr(PrecedenceLevel::BitwiseXor),
            PrecedenceLevel::BitwiseXor => self.parse_binary_expr(PrecedenceLevel::BitwiseAnd),
            PrecedenceLevel::BitwiseAnd => self.parse_binary_expr(PrecedenceLevel::Equality),
            PrecedenceLevel::Equality => self.parse_binary_expr(PrecedenceLevel::Relational),
            PrecedenceLevel::Relational => self.parse_binary_expr(PrecedenceLevel::Shift),
            PrecedenceLevel::Shift => self.parse_binary_expr(PrecedenceLevel::Additive),
            PrecedenceLevel::Additive => self.parse_binary_expr(PrecedenceLevel::Multiplicative),
            PrecedenceLevel::Multiplicative => self.parse_cast_expr(),
        }
    }

    /// Parse a cast expression: (type-name)expr, compound literal (type-name){...},
    /// or fall through to unary expression.
    pub(super) fn parse_cast_expr(&mut self) -> Expr {
        if matches!(self.peek(), TokenKind::LParen) {
            let save = self.pos;
            let save_typedef = self.attrs.parsing_typedef();
            let save_const = self.attrs.parsing_const();
            let save_vector_size = self.attrs.parsing_vector_size.take();
            let save_ext_vector = self.attrs.parsing_ext_vector_nelem.take();
            self.advance();
            if self.is_type_specifier() {
                if let Some(type_spec) = self.parse_type_specifier() {
                    let mut result_type = self.parse_abstract_declarator_suffix(type_spec);
                    // If __attribute__((vector_size(N))) was parsed, wrap the type
                    result_type = self.apply_pending_vector_attr(result_type);
                    if matches!(self.peek(), TokenKind::RParen) {
                        let span = self.peek_span();
                        self.advance();
                        // Check for compound literal: (type){...}
                        if matches!(self.peek(), TokenKind::LBrace) {
                            let init = self.parse_initializer();
                            let lit = Expr::CompoundLiteral(result_type, Box::new(init), span);
                            self.attrs.set_const(save_const);
                            // Restore outer vector attrs so the enclosing declaration can use them
                            self.attrs.parsing_vector_size = save_vector_size;
                            self.attrs.parsing_ext_vector_nelem = save_ext_vector;
                            return self.parse_postfix_ops(lit);
                        }
                        let expr = self.parse_cast_expr();
                        self.attrs.set_const(save_const);
                        // Restore outer vector attrs so the enclosing declaration can use them
                        self.attrs.parsing_vector_size = save_vector_size;
                        self.attrs.parsing_ext_vector_nelem = save_ext_vector;
                        return Expr::Cast(result_type, Box::new(expr), span);
                    }
                }
            }
            self.pos = save;
            self.attrs.set_typedef(save_typedef);
            self.attrs.set_const(save_const);
            self.attrs.parsing_vector_size = save_vector_size;
            self.attrs.parsing_ext_vector_nelem = save_ext_vector;
        }
        self.parse_unary_expr()
    }

    fn parse_unary_expr(&mut self) -> Expr {
        match self.peek() {
            TokenKind::AmpAmp => {
                // GCC extension: &&label (address of label, for computed goto)
                let span = self.peek_span();
                if self.pos + 1 < self.tokens.len() {
                    if let TokenKind::Identifier(ref name) = self.tokens[self.pos + 1].kind {
                        let label_name = name.clone();
                        self.advance(); // consume &&
                        self.advance(); // consume identifier
                        return Expr::LabelAddr(label_name, span);
                    }
                }
                self.parse_postfix_expr()
            }
            TokenKind::RealPart => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::UnaryOp(UnaryOp::RealPart, Box::new(expr), span)
            }
            TokenKind::ImagPart => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::UnaryOp(UnaryOp::ImagPart, Box::new(expr), span)
            }
            TokenKind::PlusPlus => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_unary_expr();
                Expr::UnaryOp(UnaryOp::PreInc, Box::new(expr), span)
            }
            TokenKind::MinusMinus => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_unary_expr();
                Expr::UnaryOp(UnaryOp::PreDec, Box::new(expr), span)
            }
            TokenKind::Plus => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::UnaryOp(UnaryOp::Plus, Box::new(expr), span)
            }
            TokenKind::Minus => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::UnaryOp(UnaryOp::Neg, Box::new(expr), span)
            }
            TokenKind::Tilde => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::UnaryOp(UnaryOp::BitNot, Box::new(expr), span)
            }
            TokenKind::Bang => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::UnaryOp(UnaryOp::LogicalNot, Box::new(expr), span)
            }
            TokenKind::Amp => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::AddressOf(Box::new(expr), span)
            }
            TokenKind::Star => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_cast_expr();
                Expr::Deref(Box::new(expr), span)
            }
            TokenKind::Sizeof => {
                self.parse_sizeof_expr()
            }
            TokenKind::Alignof => {
                let span = self.peek_span();
                self.advance();
                // _Alignof(type) - C11 standard, returns minimum ABI alignment
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after '_Alignof'");
                if let Some(ts) = self.parse_type_specifier() {
                    let mut result_type = self.parse_abstract_declarator_suffix(ts);
                    result_type = self.apply_pending_vector_attr(result_type);
                    self.expect_closing(&TokenKind::RParen, open);
                    Expr::Alignof(result_type, span)
                } else {
                    // GCC extension: __alignof__(expr) - alignment of expression's type
                    let expr = self.parse_assignment_expr();
                    self.expect_closing(&TokenKind::RParen, open);
                    Expr::AlignofExpr(Box::new(expr), span)
                }
            }
            TokenKind::GnuAlignof => {
                let span = self.peek_span();
                self.advance();
                // __alignof / __alignof__ - GCC extension, returns preferred alignment
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after '__alignof__'");
                if let Some(ts) = self.parse_type_specifier() {
                    let mut result_type = self.parse_abstract_declarator_suffix(ts);
                    result_type = self.apply_pending_vector_attr(result_type);
                    self.expect_closing(&TokenKind::RParen, open);
                    Expr::GnuAlignof(result_type, span)
                } else {
                    let expr = self.parse_assignment_expr();
                    self.expect_closing(&TokenKind::RParen, open);
                    Expr::GnuAlignofExpr(Box::new(expr), span)
                }
            }
            _ => self.parse_postfix_expr(),
        }
    }

    /// Parse sizeof expression. Handles both sizeof(type-name) and sizeof expr.
    fn parse_sizeof_expr(&mut self) -> Expr {
        let span = self.peek_span();
        self.advance(); // consume 'sizeof'
        if matches!(self.peek(), TokenKind::LParen) {
            let save = self.pos;
            let save_typedef = self.attrs.parsing_typedef();
            let save_const = self.attrs.parsing_const();
            let save_vector_size = self.attrs.parsing_vector_size.take();
            let save_ext_vector = self.attrs.parsing_ext_vector_nelem.take();
            self.advance();
            if self.is_type_specifier() {
                if let Some(ts) = self.parse_type_specifier() {
                    let mut result_type = self.parse_abstract_declarator_suffix(ts);
                    // If __attribute__((vector_size(N))) was parsed, wrap the type
                    result_type = self.apply_pending_vector_attr(result_type);
                    if matches!(self.peek(), TokenKind::RParen) {
                        self.expect(&TokenKind::RParen);
                        self.attrs.set_const(save_const);
                        // Restore outer vector attrs so the enclosing declaration can use them
                        self.attrs.parsing_vector_size = save_vector_size;
                        self.attrs.parsing_ext_vector_nelem = save_ext_vector;
                        return Expr::Sizeof(Box::new(SizeofArg::Type(result_type)), span);
                    }
                }
            }
            self.pos = save;
            self.attrs.set_typedef(save_typedef);
            self.attrs.set_const(save_const);
            self.attrs.parsing_vector_size = save_vector_size;
            self.attrs.parsing_ext_vector_nelem = save_ext_vector;
        }
        let expr = self.parse_unary_expr();
        Expr::Sizeof(Box::new(SizeofArg::Expr(expr)), span)
    }

    fn parse_postfix_expr(&mut self) -> Expr {
        let expr = self.parse_primary_expr();
        self.parse_postfix_ops(expr)
    }

    /// Parse postfix operators ([], ., ->, ++, --, function call) applied to an initial expression.
    fn parse_postfix_ops(&mut self, mut expr: Expr) -> Expr {
        loop {
            match self.peek() {
                TokenKind::LParen => {
                    // Function call
                    let open = self.peek_span();
                    self.advance();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), TokenKind::RParen) {
                        args.push(self.parse_assignment_expr());
                        while self.consume_if(&TokenKind::Comma) {
                            args.push(self.parse_assignment_expr());
                        }
                    }
                    self.expect_closing(&TokenKind::RParen, open);
                    expr = Expr::FunctionCall(Box::new(expr), args, open);
                }
                TokenKind::LBracket => {
                    let open = self.peek_span();
                    self.advance();
                    let index = self.parse_expr();
                    self.expect_closing(&TokenKind::RBracket, open);
                    expr = Expr::ArraySubscript(Box::new(expr), Box::new(index), open);
                }
                TokenKind::Dot => {
                    let span = self.peek_span();
                    self.advance();
                    let field = if let TokenKind::Identifier(name) = self.peek() {
                        let name = name.clone();
                        self.advance();
                        name
                    } else {
                        Rc::from("")
                    };
                    expr = Expr::MemberAccess(Box::new(expr), field, span);
                }
                TokenKind::Arrow => {
                    let span = self.peek_span();
                    self.advance();
                    let field = if let TokenKind::Identifier(name) = self.peek() {
                        let name = name.clone();
                        self.advance();
                        name
                    } else {
                        Rc::from("")
                    };
                    expr = Expr::PointerMemberAccess(Box::new(expr), field, span);
                }
                TokenKind::PlusPlus => {
                    let span = self.peek_span();
                    self.advance();
                    expr = Expr::PostfixOp(PostfixOp::PostInc, Box::new(expr), span);
                }
                TokenKind::MinusMinus => {
                    let span = self.peek_span();
                    self.advance();
                    expr = Expr::PostfixOp(PostfixOp::PostDec, Box::new(expr), span);
                }
                _ => break,
            }
        }

        expr
    }

    fn parse_primary_expr(&mut self) -> Expr {
        match self.peek() {
            TokenKind::IntLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::IntLiteral(val, span)
            }
            TokenKind::UIntLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::UIntLiteral(val, span)
            }
            TokenKind::LongLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::LongLiteral(val, span)
            }
            TokenKind::ULongLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::ULongLiteral(val, span)
            }
            TokenKind::LongLongLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::LongLongLiteral(val, span)
            }
            TokenKind::ULongLongLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::ULongLongLiteral(val, span)
            }
            TokenKind::FloatLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::FloatLiteral(val, span)
            }
            TokenKind::FloatLiteralF32(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::FloatLiteralF32(val, span)
            }
            TokenKind::FloatLiteralLongDouble(val, bytes) => {
                let val = *val;
                let bytes = *bytes;
                let span = self.peek_span();
                self.advance();
                Expr::FloatLiteralLongDouble(val, bytes, span)
            }
            TokenKind::ImaginaryLiteral(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::ImaginaryLiteral(val, span)
            }
            TokenKind::ImaginaryLiteralF32(val) => {
                let val = *val;
                let span = self.peek_span();
                self.advance();
                Expr::ImaginaryLiteralF32(val, span)
            }
            TokenKind::ImaginaryLiteralLongDouble(val, bytes) => {
                let val = *val;
                let bytes = *bytes;
                let span = self.peek_span();
                self.advance();
                Expr::ImaginaryLiteralLongDouble(val, bytes, span)
            }
            TokenKind::StringLiteral(s) => {
                let mut result = s.clone();
                let span = self.peek_span();
                self.advance();
                // Concatenate adjacent string literals. If any is wide/char16, result upgrades.
                let mut is_wide = false;
                let mut is_char16 = false;
                loop {
                    match self.peek() {
                        TokenKind::StringLiteral(s2) => {
                            result.push_str(s2);
                            self.advance();
                        }
                        TokenKind::WideStringLiteral(s2) => {
                            result.push_str(s2);
                            is_wide = true;
                            self.advance();
                        }
                        TokenKind::Char16StringLiteral(s2) => {
                            result.push_str(s2);
                            is_char16 = true;
                            self.advance();
                        }
                        _ => break,
                    }
                }
                if is_wide {
                    Expr::WideStringLiteral(result, span)
                } else if is_char16 {
                    Expr::Char16StringLiteral(result, span)
                } else {
                    Expr::StringLiteral(result, span)
                }
            }
            TokenKind::WideStringLiteral(s) => {
                let mut result = s.clone();
                let span = self.peek_span();
                self.advance();
                // Concatenate adjacent string literals (wide + narrow = wide)
                while let TokenKind::StringLiteral(s2) | TokenKind::WideStringLiteral(s2)
                    | TokenKind::Char16StringLiteral(s2) = self.peek()
                {
                    result.push_str(s2);
                    self.advance();
                }
                Expr::WideStringLiteral(result, span)
            }
            TokenKind::Char16StringLiteral(s) => {
                let mut result = s.clone();
                let span = self.peek_span();
                self.advance();
                // Concatenate adjacent string literals (char16 + narrow = char16, char16 + wide = wide)
                let mut is_wide = false;
                loop {
                    match self.peek() {
                        TokenKind::StringLiteral(s2) | TokenKind::Char16StringLiteral(s2) => {
                            result.push_str(s2);
                            self.advance();
                        }
                        TokenKind::WideStringLiteral(s2) => {
                            result.push_str(s2);
                            is_wide = true;
                            self.advance();
                        }
                        _ => break,
                    }
                }
                if is_wide {
                    Expr::WideStringLiteral(result, span)
                } else {
                    Expr::Char16StringLiteral(result, span)
                }
            }
            TokenKind::CharLiteral(c) => {
                let c = *c;
                let span = self.peek_span();
                self.advance();
                Expr::CharLiteral(c, span)
            }
            TokenKind::Identifier(name) => {
                let name = name.clone();
                let span = self.peek_span();
                self.advance();
                Expr::Identifier(name, span)
            }
            TokenKind::LParen => {
                let open = self.peek_span();
                self.advance();
                // Check for GCC statement expression: ({ stmt; stmt; expr; })
                if matches!(self.peek(), TokenKind::LBrace) {
                    let span = self.peek_span();
                    let compound = self.parse_compound_stmt();
                    self.expect_closing(&TokenKind::RParen, open);
                    Expr::StmtExpr(compound, span)
                } else {
                    let expr = self.parse_expr();
                    self.expect_closing(&TokenKind::RParen, open);
                    expr
                }
            }
            TokenKind::Generic => {
                self.parse_generic_selection()
            }
            TokenKind::Asm => {
                // GCC asm expression
                let span = self.peek_span();
                self.advance();
                self.consume_if(&TokenKind::Volatile);
                if matches!(self.peek(), TokenKind::LParen) {
                    self.skip_balanced_parens();
                }
                Expr::IntLiteral(0, span)
            }
            TokenKind::BuiltinVaArg => {
                let span = self.peek_span();
                self.advance();
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after '__builtin_va_arg'");
                let ap_expr = self.parse_assignment_expr();
                self.expect_context(&TokenKind::Comma, "between '__builtin_va_arg' arguments");
                let type_spec = self.parse_va_arg_type();
                self.expect_closing(&TokenKind::RParen, open);
                Expr::VaArg(Box::new(ap_expr), type_spec, span)
            }
            TokenKind::BuiltinTypesCompatibleP => {
                // __builtin_types_compatible_p(type1, type2)
                // Compile-time: 1 if types are compatible (ignoring qualifiers), 0 otherwise.
                let span = self.peek_span();
                self.advance();
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after '__builtin_types_compatible_p'");
                let type1 = self.parse_va_arg_type();
                self.expect_context(&TokenKind::Comma, "between '__builtin_types_compatible_p' arguments");
                let type2 = self.parse_va_arg_type();
                self.expect_closing(&TokenKind::RParen, open);
                Expr::BuiltinTypesCompatibleP(type1, type2, span)
            }
            TokenKind::Typeof => {
                let span = self.peek_span();
                self.advance();
                if matches!(self.peek(), TokenKind::LParen) {
                    self.skip_balanced_parens();
                }
                Expr::IntLiteral(0, span)
            }
            TokenKind::Builtin => {
                let span = self.peek_span();
                self.advance();
                Expr::Identifier(Rc::from("__builtin_va_list"), span)
            }
            TokenKind::Extension => {
                self.advance();
                self.parse_cast_expr()
            }
            _ => {
                let span = self.peek_span();
                self.emit_error(format!("expected expression before {}", self.peek()), span);
                self.advance();
                Expr::IntLiteral(0, span)
            }
        }
    }

    /// Parse _Generic(controlling_expr, type: expr, ..., default: expr)
    fn parse_generic_selection(&mut self) -> Expr {
        let span = self.peek_span();
        self.advance(); // consume _Generic
        let open = self.peek_span();
        self.expect_context(&TokenKind::LParen, "after '_Generic'");
        let controlling = self.parse_assignment_expr();
        self.expect_context(&TokenKind::Comma, "after '_Generic' controlling expression");
        let mut associations = Vec::new();
        loop {
            if matches!(self.peek(), TokenKind::RParen) {
                break;
            }
            // Save and reset parsing_const before parsing each association type,
            // so we can detect whether `const` appeared in the type specifier.
            let saved_const = self.attrs.parsing_const();
            self.attrs.set_const(false);
            let (type_spec, is_const) = if matches!(self.peek(), TokenKind::Default) {
                self.advance();
                (None, false)
            } else if let Some(ts) = self.parse_type_specifier() {
                // Capture whether the base type had `const` before pointer declarators.
                // For `const int *`, parsing_const is true after parse_type_specifier
                // (from the `const` keyword), and the `*` is applied in
                // parse_abstract_declarator_suffix. This means the pointee is const.
                let base_const = self.attrs.parsing_const();
                let full_type = self.parse_abstract_declarator_suffix(ts);
                (Some(full_type), base_const)
            } else {
                (None, false)
            };
            self.attrs.set_const(saved_const);
            self.expect_context(&TokenKind::Colon, "in '_Generic' association");
            let expr = self.parse_assignment_expr();
            associations.push(GenericAssociation { type_spec, expr, is_const });
            if !self.consume_if(&TokenKind::Comma) {
                break;
            }
        }
        self.expect_closing(&TokenKind::RParen, open);
        Expr::GenericSelection(Box::new(controlling), associations, span)
    }
}
