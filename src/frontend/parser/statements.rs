// Statement parsing: all C statement types including inline assembly.
//
// Handles: return, if/else, while, do-while, for, switch/case/default,
// break, continue, goto (including computed goto), labels, compound
// statements, and inline assembly (GCC syntax).

use std::rc::Rc;
use crate::frontend::lexer::token::TokenKind;
use super::ast::*;
use super::parse::Parser;

impl Parser {
    pub(super) fn parse_compound_stmt(&mut self) -> CompoundStmt {
        let open_brace = self.peek_span();
        self.expect(&TokenKind::LBrace);
        let mut items = Vec::new();
        let mut local_labels = Vec::new();

        // Save typedef shadowing state for this scope
        let saved_shadowed = self.shadowed_typedefs.clone();

        // Save declaration attribute flags so that storage-class specifiers
        // (extern, static, typedef, etc.) from declarations inside this
        // compound statement do not leak into the enclosing context.
        // This is critical for statement expressions inside typeof():
        //   typeof(({ extern void f(void); 42; })) x = 10;
        // Without this, the `extern` from `f` leaks and makes `x` extern.
        let saved_attr_flags = self.attrs.save_flags();

        // Parse GNU __label__ declarations at the start of the block.
        // These must appear before any statements or declarations.
        // Syntax: __label__ ident1, ident2, ... ;
        while matches!(self.peek(), TokenKind::GnuLabel) {
            self.advance(); // consume __label__
            // Parse comma-separated list of label names
            loop {
                if let TokenKind::Identifier(name) = self.peek() {
                    local_labels.push(name.clone());
                    self.advance();
                }
                if !self.consume_if(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect_after(&TokenKind::Semicolon, "after __label__ declaration");
        }

        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            self.skip_gcc_extensions();
            // Handle #pragma pack directives within function bodies
            while self.handle_pragma_pack_token() {
                self.consume_if(&TokenKind::Semicolon);
            }
            // Handle #pragma GCC visibility push/pop within function bodies
            while self.handle_pragma_visibility_token() {
                self.consume_if(&TokenKind::Semicolon);
            }
            if matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
                break;
            }
            // Handle __label__ declarations that appear after __extension__
            if matches!(self.peek(), TokenKind::GnuLabel) {
                self.advance();
                loop {
                    if let TokenKind::Identifier(name) = self.peek() {
                        local_labels.push(name.clone());
                        self.advance();
                    }
                    if !self.consume_if(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect_after(&TokenKind::Semicolon, "after __label__ declaration");
                continue;
            }
            if matches!(self.peek(), TokenKind::StaticAssert) {
                self.parse_static_assert();
            } else if self.is_type_specifier() && !self.is_typedef_label() {
                if let Some(decl) = self.parse_local_declaration() {
                    items.push(BlockItem::Declaration(decl));
                }
            } else {
                let stmt = self.parse_stmt();
                items.push(BlockItem::Statement(stmt));
            }
        }

        self.expect_closing(&TokenKind::RBrace, open_brace);
        self.shadowed_typedefs = saved_shadowed;
        self.attrs.restore_flags(saved_attr_flags);
        CompoundStmt { items, local_labels }
    }

    pub(super) fn parse_stmt(&mut self) -> Stmt {
        // C23 / GNU extension: declarations are allowed in statement position.
        // This handles declarations after labels (e.g., `label: int x = 5;`),
        // after case/default, and other contexts where parse_stmt() is called.
        // We skip __extension__ first since it can precede declarations.
        self.skip_gcc_extensions();
        if self.is_type_specifier() && !self.is_typedef_label() {
            if let Some(decl) = self.parse_local_declaration() {
                return Stmt::Declaration(decl);
            }
            // If parse_local_declaration returns None (e.g. _Static_assert),
            // fall through to parse a null statement or the next thing
            return Stmt::Expr(None);
        }
        match self.peek() {
            TokenKind::Return => {
                let span = self.peek_span();
                self.advance();
                let expr = if matches!(self.peek(), TokenKind::Semicolon) {
                    None
                } else {
                    Some(self.parse_expr())
                };
                self.expect_after(&TokenKind::Semicolon, "after return statement");
                Stmt::Return(expr, span)
            }
            TokenKind::If => {
                let span = self.peek_span();
                self.advance();
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after 'if'");
                let cond = self.parse_expr();
                self.expect_closing(&TokenKind::RParen, open);
                let then_stmt = self.parse_stmt();
                let else_stmt = if self.consume_if(&TokenKind::Else) {
                    Some(Box::new(self.parse_stmt()))
                } else {
                    None
                };
                Stmt::If(cond, Box::new(then_stmt), else_stmt, span)
            }
            TokenKind::While => {
                let span = self.peek_span();
                self.advance();
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after 'while'");
                let cond = self.parse_expr();
                self.expect_closing(&TokenKind::RParen, open);
                let body = self.parse_stmt();
                Stmt::While(cond, Box::new(body), span)
            }
            TokenKind::Do => {
                let span = self.peek_span();
                self.advance();
                let body = self.parse_stmt();
                self.expect_after(&TokenKind::While, "at end of do-while statement");
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after 'while'");
                let cond = self.parse_expr();
                self.expect_closing(&TokenKind::RParen, open);
                self.expect_after(&TokenKind::Semicolon, "after do-while statement");
                Stmt::DoWhile(Box::new(body), cond, span)
            }
            TokenKind::For => {
                self.parse_for_stmt()
            }
            TokenKind::LBrace => {
                let compound = self.parse_compound_stmt();
                Stmt::Compound(compound)
            }
            TokenKind::Break => {
                let span = self.peek_span();
                self.advance();
                self.expect_after(&TokenKind::Semicolon, "after break statement");
                Stmt::Break(span)
            }
            TokenKind::Continue => {
                let span = self.peek_span();
                self.advance();
                self.expect_after(&TokenKind::Semicolon, "after continue statement");
                Stmt::Continue(span)
            }
            TokenKind::Switch => {
                let span = self.peek_span();
                self.advance();
                let open = self.peek_span();
                self.expect_context(&TokenKind::LParen, "after 'switch'");
                let expr = self.parse_expr();
                self.expect_closing(&TokenKind::RParen, open);
                let body = self.parse_stmt();
                Stmt::Switch(expr, Box::new(body), span)
            }
            TokenKind::Case => {
                let span = self.peek_span();
                self.advance();
                let expr = self.parse_expr();
                if self.consume_if(&TokenKind::Ellipsis) {
                    // GNU case range extension: case low ... high:
                    let high = self.parse_expr();
                    self.expect_context(&TokenKind::Colon, "after 'case' expression");
                    let stmt = self.parse_stmt();
                    Stmt::CaseRange(expr, high, Box::new(stmt), span)
                } else {
                    self.expect_context(&TokenKind::Colon, "after 'case' expression");
                    let stmt = self.parse_stmt();
                    Stmt::Case(expr, Box::new(stmt), span)
                }
            }
            TokenKind::Default => {
                let span = self.peek_span();
                self.advance();
                self.expect_context(&TokenKind::Colon, "after 'default'");
                let stmt = self.parse_stmt();
                Stmt::Default(Box::new(stmt), span)
            }
            TokenKind::Goto => {
                let span = self.peek_span();
                self.advance();
                if matches!(self.peek(), TokenKind::Star) {
                    // Computed goto: goto *expr;
                    self.advance();
                    let expr = self.parse_expr();
                    self.expect_after(&TokenKind::Semicolon, "after goto statement");
                    Stmt::GotoIndirect(Box::new(expr), span)
                } else {
                    let label = if let TokenKind::Identifier(name) = self.peek() {
                        let name = name.clone();
                        self.advance();
                        name
                    } else {
                        Rc::from("")
                    };
                    self.expect_after(&TokenKind::Semicolon, "after goto statement");
                    Stmt::Goto(label, span)
                }
            }
            TokenKind::Identifier(name) => {
                // Check for label (identifier followed by colon)
                let name_clone = name.clone();
                let span = self.peek_span();
                if self.pos + 1 < self.tokens.len() && matches!(self.tokens[self.pos + 1].kind, TokenKind::Colon) {
                    self.advance(); // identifier
                    self.advance(); // colon
                    // Skip optional label attributes: `label: __attribute__((unused));`
                    // In GNU C, labels can have attributes (e.g., unused, hot, cold).
                    // We consume and discard them since they only affect diagnostics.
                    self.skip_label_attributes();
                    let stmt = self.parse_stmt();
                    Stmt::Label(name_clone, Box::new(stmt), span)
                } else {
                    let expr = self.parse_expr();
                    self.expect_after(&TokenKind::Semicolon, "after expression");
                    Stmt::Expr(Some(expr))
                }
            }
            TokenKind::Asm => {
                self.parse_inline_asm()
            }
            TokenKind::Semicolon => {
                self.advance();
                Stmt::Expr(None)
            }
            _ => {
                let expr = self.parse_expr();
                self.expect_after(&TokenKind::Semicolon, "after expression");
                Stmt::Expr(Some(expr))
            }
        }
    }

    /// Parse a for statement: for (init; cond; inc) body
    fn parse_for_stmt(&mut self) -> Stmt {
        let span = self.peek_span();
        self.advance();
        let open = self.peek_span();
        self.expect_context(&TokenKind::LParen, "after 'for'");

        let init = if matches!(self.peek(), TokenKind::Semicolon) {
            self.advance();
            None
        } else if self.is_type_specifier() {
            let decl = self.parse_local_declaration();
            decl.map(|d| Box::new(ForInit::Declaration(d)))
        } else {
            let expr = self.parse_expr();
            self.expect_after(&TokenKind::Semicolon, "in for statement initializer");
            Some(Box::new(ForInit::Expr(expr)))
        };

        let cond = if matches!(self.peek(), TokenKind::Semicolon) {
            None
        } else {
            Some(self.parse_expr())
        };
        self.expect_after(&TokenKind::Semicolon, "in for statement condition");

        let inc = if matches!(self.peek(), TokenKind::RParen) {
            None
        } else {
            Some(self.parse_expr())
        };
        self.expect_closing(&TokenKind::RParen, open);

        let body = self.parse_stmt();
        Stmt::For(init, cond, inc, Box::new(body), span)
    }

    // === Inline assembly parsing ===

    fn parse_inline_asm(&mut self) -> Stmt {
        self.advance(); // consume 'asm' / '__asm__'
        // Skip optional qualifiers: volatile, goto, inline
        while matches!(self.peek(), TokenKind::Volatile)
            || matches!(self.peek(), TokenKind::Goto)
            || matches!(self.peek(), TokenKind::Inline)
        {
            self.advance();
        }
        let open = self.peek_span();
        self.expect_context(&TokenKind::LParen, "after 'asm'");

        let template = self.parse_asm_string();

        let mut outputs = Vec::new();
        let mut inputs = Vec::new();
        let mut clobbers = Vec::new();
        let mut goto_labels = Vec::new();

        // First colon: outputs
        if matches!(self.peek(), TokenKind::Colon) {
            self.advance();
            outputs = self.parse_asm_operands();

            // Second colon: inputs
            if matches!(self.peek(), TokenKind::Colon) {
                self.advance();
                inputs = self.parse_asm_operands();

                // Third colon: clobbers
                if matches!(self.peek(), TokenKind::Colon) {
                    self.advance();
                    clobbers = self.parse_asm_clobbers();

                    // Fourth colon: goto labels
                    if matches!(self.peek(), TokenKind::Colon) {
                        self.advance();
                        goto_labels = self.parse_asm_goto_labels();
                    }
                }
            }
        }

        self.expect_closing(&TokenKind::RParen, open);
        self.consume_if(&TokenKind::Semicolon);

        Stmt::InlineAsm { template, outputs, inputs, clobbers, goto_labels }
    }

    fn parse_asm_string(&mut self) -> String {
        let mut result = String::new();
        while let TokenKind::StringLiteral(ref s) = self.peek() {
            result.push_str(s);
            self.advance();
        }
        result
    }

    fn parse_asm_operands(&mut self) -> Vec<AsmOperand> {
        let mut operands = Vec::new();
        if matches!(self.peek(), TokenKind::Colon | TokenKind::RParen) {
            return operands;
        }
        loop {
            let operand = self.parse_one_asm_operand();
            operands.push(operand);
            if !self.consume_if(&TokenKind::Comma) {
                break;
            }
        }
        operands
    }

    fn parse_one_asm_operand(&mut self) -> AsmOperand {
        // Optional [name]
        let name = if matches!(self.peek(), TokenKind::LBracket) {
            let open = self.peek_span();
            self.advance();
            let n = if let TokenKind::Identifier(ref id) = self.peek() {
                let id = id.clone();
                self.advance();
                Some(id)
            } else {
                None
            };
            self.expect_closing(&TokenKind::RBracket, open);
            n
        } else {
            None
        };

        // Constraint string (may be concatenated)
        let constraint = if let TokenKind::StringLiteral(ref s) = self.peek() {
            let mut full = s.clone();
            self.advance();
            while let TokenKind::StringLiteral(ref s2) = self.peek() {
                full.push_str(s2);
                self.advance();
            }
            full
        } else {
            String::new()
        };

        // (expr)
        let open = self.peek_span();
        self.expect_context(&TokenKind::LParen, "in asm operand");
        let expr = self.parse_expr();
        self.expect_closing(&TokenKind::RParen, open);

        AsmOperand { name: name.map(|n| n.to_string()), constraint, expr }
    }

    fn parse_asm_clobbers(&mut self) -> Vec<String> {
        let mut clobbers = Vec::new();
        if matches!(self.peek(), TokenKind::Colon | TokenKind::RParen) {
            return clobbers;
        }
        while let TokenKind::StringLiteral(ref s) = self.peek() {
            clobbers.push(s.clone());
            self.advance();
            if !self.consume_if(&TokenKind::Comma) {
                break;
            }
        }
        clobbers
    }

    /// Parse the goto labels section (fourth colon) of an asm goto statement.
    /// Labels are comma-separated identifiers: `asm goto("..." : : : : label1, label2)`
    fn parse_asm_goto_labels(&mut self) -> Vec<Rc<str>> {
        let mut labels = Vec::new();
        if matches!(self.peek(), TokenKind::RParen) {
            return labels;
        }
        while let TokenKind::Identifier(ref name) = self.peek() {
            labels.push(name.clone());
            self.advance();
            if !self.consume_if(&TokenKind::Comma) {
                break;
            }
        }
        labels
    }
}
