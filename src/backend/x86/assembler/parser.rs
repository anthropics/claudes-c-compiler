//! Parser for AT&T syntax x86-64 assembly as emitted by our codegen.
//!
//! Parses assembly text line-by-line into structured `AsmItem` values.
//! Handles directives, labels, and instructions with AT&T operand ordering
//! (source, destination).

use std::fmt;
use crate::backend::asm_expr;
use crate::backend::asm_preprocess::{self, CommentStyle};
use crate::backend::elf;

/// A parsed assembly item (one per line, roughly).
#[derive(Debug, Clone)]
pub enum AsmItem {
    /// Switch to a named section: `.section .text`, `.section .rodata`, etc.
    Section(SectionDirective),
    /// Global symbol: `.globl name`
    Global(String),
    /// Weak symbol: `.weak name`
    Weak(String),
    /// Hidden visibility: `.hidden name`
    Hidden(String),
    /// Protected visibility: `.protected name`
    Protected(String),
    /// Internal visibility: `.internal name`
    Internal(String),
    /// Symbol type: `.type name, @function` or `@object` or `@tls_object`
    SymbolType(String, SymbolKind),
    /// Symbol size: `.size name, expr`
    Size(String, SizeExpr),
    /// Label definition: `name:`
    Label(String),
    /// Alignment: `.align N`
    Align(u32),
    /// Emit bytes: `.byte val, val, ...` (can contain label expressions)
    Byte(Vec<DataValue>),
    /// Emit 16-bit values: `.short val, ...` (can be symbol references)
    Short(Vec<DataValue>),
    /// Emit 32-bit values: `.long val, ...` (can be symbol references)
    Long(Vec<DataValue>),
    /// Emit 64-bit values: `.quad val, ...` (can be symbol references)
    Quad(Vec<DataValue>),
    /// Emit zero bytes: `.zero N`
    Zero(u32),
    /// Deferred `.skip` with expression: evaluated after all labels are known.
    /// Used by kernel alternatives framework for label-arithmetic expressions
    /// like `.skip -(((6651f-6641f)-(662b-661b)) > 0) * ((6651f-6641f)-(662b-661b)), 0x90`.
    /// Fields: (expression_string, fill_byte)
    SkipExpr(String, u8),
    /// NUL-terminated string: `.asciz "str"`
    Asciz(Vec<u8>),
    /// String without NUL: `.ascii "str"`
    Ascii(Vec<u8>),
    /// Common symbol: `.comm name, size, align`
    Comm(String, u64, u32),
    /// Symbol alias: `.set alias, target`
    Set(String, String),
    /// CFI directive (ignored for code generation, kept for .eh_frame)
    #[allow(dead_code)] // Parsed by parser but skipped during encoding (DWARF .eh_frame not yet emitted)
    Cfi(CfiDirective),
    /// Debug file directive: `.file N "filename"`
    #[allow(dead_code)] // Parsed but skipped during encoding (debug info not yet emitted)
    File(u32, String),
    /// Debug location: `.loc filenum line column`
    #[allow(dead_code)] // Parsed but skipped during encoding (debug info not yet emitted)
    Loc(u32, u32, u32),
    /// x86-64 instruction
    Instruction(Instruction),
    /// `.option norelax` (RISC-V, ignored for x86)
    #[allow(dead_code)] // Parsed for RISC-V compatibility in shared parser; ignored on x86
    OptionDirective(String),
    /// Push current section and switch to a new one: `.pushsection name,"flags",@type`
    PushSection(SectionDirective),
    /// Pop section stack and restore previous section: `.popsection`
    PopSection,
    /// Swap current and previous sections: `.previous`
    Previous,
    /// `.org` directive: advance to position symbol + offset within the section.
    Org(String, i64),
    /// `.incbin "file"[, skip[, count]]` — include binary file contents
    Incbin { path: String, skip: u64, count: Option<u64> },
    /// Symbol version: `.symver name, name2@@VERSION` or `.symver name, name2@VERSION`
    Symver(String, String),
    /// Code mode switch: `.code16`, `.code32`, `.code64`
    /// The value is the bit width (16, 32, or 64).
    CodeMode(u8),
    /// Blank line or comment-only line
    Empty,
}

/// Section directive with optional flags and type.
#[derive(Debug, Clone)]
pub struct SectionDirective {
    pub name: String,
    pub flags: Option<String>,
    pub section_type: Option<String>,
    /// For sections with linked-to or group info (4th arg), e.g.
    /// `__patchable_function_entries,"awo",@progbits,.LPFE0`
    #[allow(dead_code)] // Parsed for completeness; will be used for SHF_LINK_ORDER sections
    pub extra: Option<String>,
    /// COMDAT group name from `.section name,"axG",@progbits,group_name,comdat`
    /// Set when flags contain 'G' and a 5th argument is "comdat".
    pub comdat_group: Option<String>,
}

/// Symbol kind from `.type` directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Object,
    TlsObject,
    NoType,
}

/// Size expression: either a constant or `.-name` (current position minus symbol).
#[derive(Debug, Clone)]
pub enum SizeExpr {
    Constant(u64),
    CurrentMinusSymbol(String),
    /// end_label - start_label (resolved by ELF writer after relaxation)
    SymbolDiff(String, String),
    /// Symbol reference (e.g., .L__sym_size_*) - resolved through .set aliases
    SymbolRef(String),
}

/// A data value that can be a constant, a symbol, or a symbol expression.
#[derive(Debug, Clone)]
pub enum DataValue {
    Integer(i64),
    Symbol(String),
    /// symbol + offset (e.g., `.quad GD_struct+128`)
    SymbolOffset(String, i64),
    /// symbol - symbol + addend (e.g., `.long .LBB3 - .Ljt_0`, `.short tr_gdt_end - tr_gdt - 1`)
    SymbolDiff(String, String),
    /// symbol - symbol with constant addend
    SymbolDiffAddend(String, String, i64),
}

/// CFI directives (call frame information).
#[derive(Debug, Clone)]
#[allow(dead_code)] // Variants constructed by parser; not yet consumed for .eh_frame emission
pub enum CfiDirective {
    StartProc,
    EndProc,
    DefCfaOffset(i32),
    DefCfaRegister(String),
    Offset(String, i32),
    Other(String),
}

/// An x86-64 instruction with mnemonic and operands.
#[derive(Debug, Clone)]
pub struct Instruction {
    /// Optional prefix (e.g., "lock", "rep")
    pub prefix: Option<String>,
    /// Instruction mnemonic (e.g., "movq", "addl", "ret")
    pub mnemonic: String,
    /// Operands in AT&T order (source first, destination last)
    pub operands: Vec<Operand>,
}

/// An instruction operand.
#[derive(Debug, Clone)]
pub enum Operand {
    /// Register: %rax, %eax, %al, %xmm0, %st(0), etc.
    Register(Register),
    /// Immediate: $42, $-1, $symbol
    Immediate(ImmediateValue),
    /// Memory reference: disp(%base, %index, scale) with optional segment
    Memory(MemoryOperand),
    /// Direct label/symbol reference (for jmp/call targets)
    Label(String),
    /// Indirect jump/call target: *%reg or *addr
    Indirect(Box<Operand>),
}

/// Register reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Register {
    pub name: String,
}

impl Register {
    pub fn new(name: &str) -> Self {
        Register { name: name.to_string() }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.name)
    }
}

/// Immediate value.
#[derive(Debug, Clone)]
pub enum ImmediateValue {
    Integer(i64),
    Symbol(String),
    /// Symbol with integer offset, e.g., init_top_pgt - 0xffffffff80000000
    SymbolPlusOffset(String, i64),
    /// Symbol with @modifier, e.g., symbol@GOTPCREL
    #[allow(dead_code)] // Constructed by parser; handled by encoder for GOT/TLS relocations
    SymbolMod(String, String),
    /// Symbol difference: sym_a - sym_b (e.g., $_DYNAMIC-1b)
    SymbolDiff(String, String),
}

/// Memory operand: optional_segment:disp(%base, %index, scale)
#[derive(Debug, Clone)]
pub struct MemoryOperand {
    pub segment: Option<String>,
    pub displacement: Displacement,
    pub base: Option<Register>,
    pub index: Option<Register>,
    pub scale: Option<u8>,
}

/// Memory displacement.
#[derive(Debug, Clone)]
pub enum Displacement {
    None,
    Integer(i64),
    Symbol(String),
    /// Symbol with an addend offset: symbol+offset or symbol-offset (e.g., GD_struct+128(%rip))
    #[allow(dead_code)] // Constructed by elf.rs numeric label resolver; consumed by encoder
    SymbolAddend(String, i64),
    /// Symbol with relocation modifier: symbol@GOT, symbol@GOTPCREL, symbol@TPOFF, etc.
    SymbolMod(String, String),
    /// Symbol plus integer offset: symbol+N or symbol-N
    SymbolPlusOffset(String, i64),
}

/// Comment style for x86 AT&T GAS: `#` only.
const COMMENT_STYLE: CommentStyle = CommentStyle::Hash;

/// Expand .rept/.endr blocks by repeating contained lines.
/// Note: .irp/.endr blocks are handled separately by expand_gas_macros, so we must
/// track .irp depth to avoid swallowing .endr lines that belong to .irp blocks.
fn expand_rept_blocks(lines: &[&str]) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut irp_depth = 0; // Track .irp nesting to avoid consuming their .endr
    while i < lines.len() {
        let trimmed = strip_comment(lines[i]).trim().to_string();
        if trimmed.starts_with(".rept ") || trimmed.starts_with(".rept\t") {
            if irp_depth > 0 {
                // Inside .irp block - pass through as-is
                result.push(lines[i].to_string());
                i += 1;
                continue;
            }
            let count_str = trimmed[".rept".len()..].trim();
            let count = parse_integer_expr(count_str)
                .map_err(|e| format!(".rept: bad count '{}': {}", count_str, e))? as usize;
            let mut depth = 1;
            let mut body = Vec::new();
            i += 1;
            while i < lines.len() {
                let inner = strip_comment(lines[i]).trim().to_string();
                if inner.starts_with(".rept ") || inner.starts_with(".rept\t") {
                    depth += 1;
                } else if inner == ".endr" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                body.push(lines[i]);
                i += 1;
            }
            if depth != 0 {
                return Err(".rept without matching .endr".to_string());
            }
            let expanded_body = expand_rept_blocks(&body)?;
            for _ in 0..count {
                result.extend(expanded_body.iter().cloned());
            }
        } else if trimmed.starts_with(".irp ") || trimmed.starts_with(".irp\t") {
            irp_depth += 1;
            result.push(lines[i].to_string());
        } else if trimmed == ".endr" {
            if irp_depth > 0 {
                irp_depth -= 1;
                result.push(lines[i].to_string());
            }
            // else: stray .endr without .rept or .irp - skip
        } else {
            result.push(lines[i].to_string());
        }
        i += 1;
    }
    Ok(result)
}

/// Parse assembly text into a list of AsmItems.
pub fn parse_asm(text: &str) -> Result<Vec<AsmItem>, String> {
    let mut items = Vec::new();

    // Strip C-style /* */ comments (used in hand-written assembly like musl)
    let text = asm_preprocess::strip_c_comments(text);
    // Pre-split on ';' (GAS statement separator) so that .rept/.irp directives
    // embedded in semicolon-separated lines (e.g., from C preprocessor macro
    // expansion like PMDS) are properly detected by expand_rept_blocks.
    let raw_lines: Vec<&str> = text.lines().collect();
    let mut lines: Vec<String> = Vec::new();
    for line in &raw_lines {
        let parts = asm_preprocess::split_on_semicolons(line);
        if parts.len() > 1 {
            for part in parts {
                let part = part.trim();
                if !part.is_empty() {
                    lines.push(part.to_string());
                }
            }
        } else {
            lines.push(line.to_string());
        }
    }
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let expanded = expand_rept_blocks(&line_refs)?;
    let expanded = expand_gas_macros(&expanded)?;

    for (line_num, line) in expanded.iter().enumerate() {
        let line_num = line_num + 1; // 1-based

        // Strip comments (# to end of line, but not inside strings)
        let stripped = strip_comment(line);
        let trimmed = stripped.trim();

        if trimmed.is_empty() {
            items.push(AsmItem::Empty);
            continue;
        }

        // Handle ';' as instruction separator (GAS syntax)
        // Split the line on ';' and parse each part independently.
        let parts: Vec<&str> = asm_preprocess::split_on_semicolons(trimmed);
        for part in parts {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            match parse_line_items(part) {
                Ok(line_items) => items.extend(line_items),
                Err(e) => {
                    return Err(format!("line {}: {}: '{}'", line_num, e, part));
                }
            }
        }
    }

    Ok(items)
}

/// Strip trailing comment from a line using x86 comment style (`#`).
fn strip_comment(line: &str) -> &str {
    asm_preprocess::strip_comment(line, &COMMENT_STYLE)
}

/// Parse a single non-empty assembly line into one or more AsmItems.
///
/// A line may contain a label followed by an instruction on the same line
/// (e.g., `1: stmxcsr -8(%rsp)`), which produces two items.
fn parse_line_items(line: &str) -> Result<Vec<AsmItem>, String> {
    let mut items = Vec::new();

    // Check for label (may be followed by instruction on same line)
    let rest = if let Some((label, rest)) = try_parse_label(line) {
        items.push(AsmItem::Label(label));
        rest
    } else {
        line.trim()
    };

    // If there's nothing after the label, we're done
    if rest.is_empty() {
        return Ok(items);
    }

    // Check for GAS symbol assignment: symbol = expr (e.g., L4_PAGE_OFFSET = 42)
    // This is equivalent to .set symbol, expr
    if let Some(eq_pos) = rest.find('=') {
        let before = rest[..eq_pos].trim();
        // Make sure it looks like a symbol name (no spaces, not a register, not starting with special chars)
        if !before.is_empty()
            && !before.contains(' ')
            && !before.contains('\t')
            && !before.starts_with('$')
            && !before.starts_with('%')
            && (before.as_bytes()[0].is_ascii_alphabetic() || before.starts_with('_'))
        {
            let expr = rest[eq_pos + 1..].trim();
            items.push(AsmItem::Set(before.to_string(), expr.to_string()));
            return Ok(items);
        }
    }

    // Handle multiple labels on same line (e.g. "771: 999: .pushsection ...")
    let mut rest = rest;
    while let Some((label, remaining)) = try_parse_label(rest) {
        items.push(AsmItem::Label(label));
        rest = remaining;
    }

    // If there's nothing after the labels, we're done
    if rest.is_empty() {
        return Ok(items);
    }

    // Parse the remaining content as a directive or instruction
    if rest.starts_with('.') {
        items.push(parse_directive(rest)?);
    } else if rest.starts_with("lock ") || rest.starts_with("rep ") || rest.starts_with("repz ") || rest.starts_with("repe ") || rest.starts_with("repnz ") || rest.starts_with("repne ") || rest.starts_with("notrack ") {
        items.push(parse_prefixed_instruction(rest)?);
    } else {
        items.push(parse_instruction(rest, None)?);
    }

    Ok(items)
}

/// Try to parse a label definition. Returns the label name and any remaining
/// content after the label (which may be an instruction or directive on the same line).
///
/// For example:
///   "foo:"           -> Some(("foo", ""))
///   "1:  stmxcsr -8(%rsp)"  -> Some(("1", "stmxcsr -8(%rsp)"))
///   ".Lfoo: ret"     -> Some((".Lfoo", "ret"))
///   "mov %rax, %rbx" -> None (no label)
fn try_parse_label(line: &str) -> Option<(String, &str)> {
    let trimmed = line.trim();
    if let Some(colon_pos) = trimmed.find(':') {
        let candidate = trimmed[..colon_pos].trim();
        // Verify it's a valid label (no spaces, starts with letter/dot/digit)
        if !candidate.is_empty()
            && !candidate.contains(' ')
            && !candidate.contains('\t')
            && !candidate.contains(',')
            && !candidate.starts_with('$')
            && !candidate.starts_with('%')
        {
            let rest = trimmed[colon_pos + 1..].trim();
            return Some((candidate.to_string(), rest));
        }
    }
    None
}

/// Parse a directive line (starts with '.').
fn parse_directive(line: &str) -> Result<AsmItem, String> {
    let parts: Vec<&str> = line.splitn(2, |c: char| c.is_whitespace()).collect();
    let directive = parts[0];
    let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match directive {
        ".section" => parse_section_directive(args),
        ".text" => Ok(AsmItem::Section(SectionDirective {
            name: ".text".to_string(),
            flags: None,
            section_type: None,
            extra: None,
            comdat_group: None,
        })),
        ".data" => Ok(AsmItem::Section(SectionDirective {
            name: ".data".to_string(),
            flags: None,
            section_type: None,
            extra: None,
            comdat_group: None,
        })),
        ".bss" => Ok(AsmItem::Section(SectionDirective {
            name: ".bss".to_string(),
            flags: None,
            section_type: None,
            extra: None,
            comdat_group: None,
        })),
        ".rodata" => Ok(AsmItem::Section(SectionDirective {
            name: ".rodata".to_string(),
            flags: None,
            section_type: None,
            extra: None,
            comdat_group: None,
        })),
        ".globl" | ".global" => Ok(AsmItem::Global(args.trim().to_string())),
        ".weak" => Ok(AsmItem::Weak(args.trim().to_string())),
        ".hidden" => Ok(AsmItem::Hidden(args.trim().to_string())),
        ".protected" => Ok(AsmItem::Protected(args.trim().to_string())),
        ".internal" => Ok(AsmItem::Internal(args.trim().to_string())),
        ".type" => parse_type_directive(args),
        ".size" => parse_size_directive(args),
        ".align" | ".p2align" | ".balign" => {
            let val_str = args.split(',').next().unwrap_or("1").trim();
            let val: u32 = parse_integer_expr(val_str)
                .map_err(|_| format!("bad alignment: {}", args))? as u32;
            // .p2align is power-of-2, .align/.balign on x86 gas is byte count
            if directive == ".p2align" {
                Ok(AsmItem::Align(1 << val))
            } else {
                Ok(AsmItem::Align(val))
            }
        }
        ".org" => {
            let args = args.trim();
            if let Ok(val) = parse_integer_expr(args) {
                Ok(AsmItem::Org(String::new(), val))
            } else {
                let mut sym = args.to_string();
                let mut offset = 0i64;
                if let Some(plus_pos) = args.find('+') {
                    if plus_pos > 0 {
                        sym = args[..plus_pos].trim().to_string();
                        let offset_str = args[plus_pos + 1..].trim();
                        offset = parse_integer_expr(offset_str)
                            .map_err(|_| format!("bad .org offset: {}", offset_str))?;
                    }
                }
                Ok(AsmItem::Org(sym, offset))
            }
        }
        ".byte" => {
            let vals = parse_data_values(args)?;
            Ok(AsmItem::Byte(vals))
        }
        ".short" | ".value" | ".2byte" | ".word" | ".hword" => {
            let vals = parse_data_values(args)?;
            Ok(AsmItem::Short(vals))
        }
        ".long" | ".4byte" | ".int" => {
            let vals = parse_data_values(args)?;
            Ok(AsmItem::Long(vals))
        }
        ".quad" | ".8byte" => {
            let vals = parse_data_values(args)?;
            Ok(AsmItem::Quad(vals))
        }
        ".zero" | ".skip" => {
            let (expr_str, fill) = split_skip_args(args);
            if let Ok(val) = parse_integer_expr(expr_str) {
                if fill == 0 {
                    Ok(AsmItem::Zero(val as u32))
                } else {
                    Ok(AsmItem::SkipExpr(expr_str.to_string(), fill))
                }
            } else {
                // Expression contains labels or complex arithmetic - defer to ELF writer
                Ok(AsmItem::SkipExpr(expr_str.to_string(), fill))
            }
        }
        ".fill" => {
            // .fill repeat, size, value
            // Emits repeat copies of value, each size bytes wide (LE, max 8).
            let parts: Vec<&str> = args.splitn(3, ',').collect();
            let repeat_str = parts[0].trim();
            match parse_integer_expr(repeat_str) {
                Ok(repeat) => {
                    let repeat = repeat as u64;
                    let size = if parts.len() > 1 {
                        parse_integer_expr(parts[1].trim())
                            .map_err(|_| format!("bad .fill size: {}", parts[1].trim()))? as u64
                    } else {
                        1
                    };
                    let value = if parts.len() > 2 {
                        parse_integer_expr(parts[2].trim())
                            .map_err(|_| format!("bad .fill value: {}", parts[2].trim()))? as u64
                    } else {
                        0
                    };
                    let total_bytes = repeat * size.min(8);
                    if value == 0 {
                        Ok(AsmItem::Zero(total_bytes as u32))
                    } else {
                        let mut data = Vec::with_capacity(total_bytes as usize);
                        let value_bytes = value.to_le_bytes();
                        for _ in 0..repeat {
                            for j in 0..size.min(8) as usize {
                                data.push(value_bytes[j]);
                            }
                        }
                        Ok(AsmItem::Ascii(data))
                    }
                }
                Err(_) => {
                    // Repeat expression contains labels/symbols - defer to ELF writer.
                    // Parse size and value as constants (these are always simple integers).
                    let size = if parts.len() > 1 {
                        parse_integer_expr(parts[1].trim())
                            .map_err(|_| format!("bad .fill size: {}", parts[1].trim()))? as u64
                    } else {
                        1
                    };
                    let value = if parts.len() > 2 {
                        parse_integer_expr(parts[2].trim())
                            .map_err(|_| format!("bad .fill value: {}", parts[2].trim()))? as u8
                    } else {
                        0
                    };
                    if size == 1 {
                        // size=1: equivalent to .skip repeat, value
                        Ok(AsmItem::SkipExpr(repeat_str.to_string(), value))
                    } else if value == 0 {
                        // value=0: equivalent to .skip (repeat * size), 0
                        // Wrap expression: (repeat_expr) * size
                        let expr = format!("({}) * {}", repeat_str, size);
                        Ok(AsmItem::SkipExpr(expr, 0))
                    } else {
                        Err(format!("bad .fill repeat: {}: deferred .fill with size > 1 and non-zero value not supported", repeat_str))
                    }
                }
            }
        }
        ".asciz" | ".string" => {
            let s = elf::parse_string_literal(args)?;
            let mut bytes = s;
            bytes.push(0); // NUL terminator
            Ok(AsmItem::Asciz(bytes))
        }
        ".ascii" => {
            let s = elf::parse_string_literal(args)?;
            Ok(AsmItem::Ascii(s))
        }
        ".comm" => parse_comm_directive(args),
        ".set" => parse_set_directive(args),
        ".symver" => parse_symver_directive(args),
        ".cfi_startproc" => Ok(AsmItem::Cfi(CfiDirective::StartProc)),
        ".cfi_endproc" => Ok(AsmItem::Cfi(CfiDirective::EndProc)),
        ".cfi_def_cfa_offset" => {
            let val: i32 = args.trim().parse()
                .map_err(|_| format!("bad cfi offset: {}", args))?;
            Ok(AsmItem::Cfi(CfiDirective::DefCfaOffset(val)))
        }
        ".cfi_def_cfa_register" => {
            let reg = args.trim().trim_start_matches('%').to_string();
            Ok(AsmItem::Cfi(CfiDirective::DefCfaRegister(reg)))
        }
        ".cfi_offset" => {
            // .cfi_offset %rbp, -16
            let parts: Vec<&str> = args.splitn(2, ',').collect();
            if parts.len() != 2 {
                return Ok(AsmItem::Cfi(CfiDirective::Other(line.to_string())));
            }
            let reg = parts[0].trim().trim_start_matches('%').to_string();
            let off: i32 = parts[1].trim().parse()
                .map_err(|_| format!("bad cfi offset value: {}", args))?;
            Ok(AsmItem::Cfi(CfiDirective::Offset(reg, off)))
        }
        ".file" => {
            // .file N "filename"
            let parts: Vec<&str> = args.splitn(2, |c: char| c.is_whitespace()).collect();
            if parts.len() == 2 {
                let num: u32 = parts[0].trim().parse().unwrap_or(0);
                let filename = parts[1].trim().trim_matches('"').to_string();
                Ok(AsmItem::File(num, filename))
            } else {
                Ok(AsmItem::Empty) // ignore malformed .file
            }
        }
        ".loc" => {
            // .loc filenum line column
            let nums: Vec<u32> = args.split_whitespace()
                .take(3)
                .filter_map(|s| s.parse().ok())
                .collect();
            if nums.len() >= 2 {
                Ok(AsmItem::Loc(nums[0], nums[1], nums.get(2).copied().unwrap_or(0)))
            } else {
                Ok(AsmItem::Empty)
            }
        }
        ".pushsection" => {
            // .pushsection name,"flags",@type - same syntax as .section
            parse_section_directive(args).map(|item| {
                if let AsmItem::Section(dir) = item {
                    AsmItem::PushSection(dir)
                } else {
                    AsmItem::Empty
                }
            })
        }
        ".popsection" => Ok(AsmItem::PopSection),
        ".previous" => Ok(AsmItem::Previous),
        ".code16gcc" | ".code16" => Ok(AsmItem::CodeMode(16)),
        ".code32" => Ok(AsmItem::CodeMode(32)),
        ".code64" => Ok(AsmItem::CodeMode(64)),
        ".option" => Ok(AsmItem::OptionDirective(args.to_string())),
        ".incbin" => {
            let parts: Vec<&str> = args.splitn(3, ',').collect();
            let path = elf::parse_string_literal(parts[0].trim())
                .map_err(|e| format!(".incbin path: {}", e))?;
            let path = String::from_utf8(path)
                .map_err(|_| ".incbin: invalid UTF-8 in path".to_string())?;
            let skip = if parts.len() > 1 {
                parts[1].trim().parse::<u64>().unwrap_or(0)
            } else { 0 };
            let count = if parts.len() > 2 {
                Some(parts[2].trim().parse::<u64>().unwrap_or(0))
            } else { None };
            Ok(AsmItem::Incbin { path, skip, count })
        }
        _ => {
            // Unknown directive - just ignore it with a warning
            // This handles .ident, .addrsig, etc. that GCC might emit
            Ok(AsmItem::Empty)
        }
    }
}

/// Parse `.section name,"flags",@type` directive.
fn parse_section_directive(args: &str) -> Result<AsmItem, String> {
    // Split by comma, but handle quoted strings
    let parts = split_section_args(args);

    let name = parts.first()
        .map(|s| s.trim().trim_matches('"').to_string())
        .unwrap_or_else(|| ".text".to_string());

    let flags = parts.get(1).map(|s| s.trim().trim_matches('"').to_string());
    let section_type = parts.get(2).map(|s| s.trim().to_string());
    let extra = parts.get(3).map(|s| s.trim().to_string());

    // If the flags contain 'G' (SHF_GROUP), the 4th arg is the group signature
    // name and the 5th arg should be "comdat".
    let comdat_group = if let Some(ref f) = flags {
        if f.contains('G') {
            if let Some(ref group_name) = extra {
                let fifth = parts.get(4).map(|s| s.trim().to_string());
                if fifth.as_deref() == Some("comdat") || fifth.is_none() {
                    // Accept both explicit "comdat" and implicit (GNU as default)
                    Some(group_name.clone())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(AsmItem::Section(SectionDirective {
        name,
        flags,
        section_type,
        extra,
        comdat_group,
    }))
}

/// Split section directive args, respecting quoted strings.
fn split_section_args(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for c in s.chars() {
        if c == '"' {
            in_quotes = !in_quotes;
            current.push(c);
        } else if c == ',' && !in_quotes {
            parts.push(current.clone());
            current.clear();
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Parse `.type name, @function` or `@object` or `@tls_object`.
/// Also handles Linux kernel format: `.type name STT_FUNC`.
fn parse_type_directive(args: &str) -> Result<AsmItem, String> {
    let (name, kind_str) = if let Some(comma_pos) = args.find(',') {
        // Standard GAS format: .type name, @function
        (args[..comma_pos].trim(), args[comma_pos+1..].trim())
    } else {
        // Linux kernel format: .type name STT_FUNC (space-separated, no comma)
        let parts: Vec<&str> = args.splitn(2, char::is_whitespace).collect();
        if parts.len() != 2 {
            return Err(format!("bad .type directive: {}", args));
        }
        (parts[0].trim(), parts[1].trim())
    };
    let name = name.to_string();
    let kind = match kind_str {
        "@function" | "%function" | "STT_FUNC" => SymbolKind::Function,
        "@object" | "%object" | "STT_OBJECT" => SymbolKind::Object,
        "@tls_object" | "%tls_object" | "STT_TLS" => SymbolKind::TlsObject,
        "@notype" | "%notype" | "STT_NOTYPE" => SymbolKind::NoType,
        _ => return Err(format!("unknown symbol type: {}", kind_str)),
    };
    Ok(AsmItem::SymbolType(name, kind))
}

/// Parse `.size name, expr`.
fn parse_size_directive(args: &str) -> Result<AsmItem, String> {
    let parts: Vec<&str> = args.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err(format!("bad .size directive: {}", args));
    }
    let name = parts[0].trim().to_string();
    let expr_str = parts[1].trim();

    // Match ". - sym" or ".-sym" (current position minus symbol)
    let normalized = expr_str.replace(' ', "");
    if let Some(rest) = normalized.strip_prefix(".-") {
        let sym = rest.trim().to_string();
        Ok(AsmItem::Size(name, SizeExpr::CurrentMinusSymbol(sym)))
    } else if let Ok(val) = parse_integer_expr(expr_str) {
        Ok(AsmItem::Size(name, SizeExpr::Constant(val as u64)))
    } else if expr_str.starts_with('.') || expr_str.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_') {
        // Symbol reference (e.g., .L__sym_size_*) - resolve through .set aliases
        // Treat as a symbol that will be resolved during ELF writing
        Ok(AsmItem::Size(name, SizeExpr::SymbolRef(expr_str.to_string())))
    } else {
        Err(format!("bad size expr: {}", expr_str))
    }
}

/// Parse `.comm name, size, align`.
fn parse_comm_directive(args: &str) -> Result<AsmItem, String> {
    let parts: Vec<&str> = args.split(',').collect();
    if parts.len() < 2 {
        return Err(format!("bad .comm directive: {}", args));
    }
    let name = parts[0].trim().to_string();
    let size: u64 = parts[1].trim().parse()
        .map_err(|_| format!("bad .comm size: {}", args))?;
    let align: u32 = parts.get(2)
        .map(|s| s.trim().parse().unwrap_or(1))
        .unwrap_or(1);
    Ok(AsmItem::Comm(name, size, align))
}

/// Parse `.set alias, target`.
fn parse_set_directive(args: &str) -> Result<AsmItem, String> {
    let parts: Vec<&str> = args.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err(format!("bad .set directive: {}", args));
    }
    Ok(AsmItem::Set(
        parts[0].trim().to_string(),
        parts[1].trim().to_string(),
    ))
}

/// Parse `.symver name, name2@@VERSION` or `.symver name, name2@VERSION`.
fn parse_symver_directive(args: &str) -> Result<AsmItem, String> {
    let parts: Vec<&str> = args.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err(format!("bad .symver directive: {}", args));
    }
    Ok(AsmItem::Symver(
        parts[0].trim().to_string(),
        parts[1].trim().to_string(),
    ))
}

/// Parse a prefix instruction like "lock cmpxchgq ..." or "rep movsb".
fn parse_prefixed_instruction(line: &str) -> Result<AsmItem, String> {
    let parts: Vec<&str> = line.splitn(2, |c: char| c.is_whitespace()).collect();
    let prefix = parts[0].to_string();
    let rest = parts.get(1).map(|s| s.trim()).unwrap_or("");
    parse_instruction(rest, Some(prefix))
}

/// Parse an instruction line.
fn parse_instruction(line: &str, prefix: Option<String>) -> Result<AsmItem, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(AsmItem::Empty);
    }

    // Split mnemonic from operands
    let (mnemonic, operand_str) = split_mnemonic_operands(trimmed);

    if mnemonic.is_empty() {
        return Err(format!("empty mnemonic in: {}", line));
    }

    let operands = if operand_str.is_empty() {
        Vec::new()
    } else {
        parse_operands(operand_str)?
    };

    Ok(AsmItem::Instruction(Instruction {
        prefix,
        mnemonic: mnemonic.to_string(),
        operands,
    }))
}

/// Split a line into mnemonic and operand string.
fn split_mnemonic_operands(line: &str) -> (&str, &str) {
    if let Some(pos) = line.find(|c: char| c.is_whitespace()) {
        let mnemonic = &line[..pos];
        let rest = line[pos..].trim();
        (mnemonic, rest)
    } else {
        (line, "")
    }
}

/// Parse comma-separated operands.
fn parse_operands(s: &str) -> Result<Vec<Operand>, String> {
    let parts = split_operands(s);
    let mut operands = Vec::new();
    for part in &parts {
        operands.push(parse_operand(part.trim())?);
    }
    Ok(operands)
}

/// Split operand string by commas, respecting parentheses.
fn split_operands(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut paren_depth = 0;

    for c in s.chars() {
        if c == '(' {
            paren_depth += 1;
            current.push(c);
        } else if c == ')' {
            paren_depth -= 1;
            current.push(c);
        } else if c == ',' && paren_depth == 0 {
            parts.push(current.clone());
            current.clear();
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Parse a single operand.
fn parse_operand(s: &str) -> Result<Operand, String> {
    let s = s.trim();

    // Indirect: *%reg or *addr
    if let Some(rest) = s.strip_prefix('*') {
        let inner = parse_operand(rest)?;
        return Ok(Operand::Indirect(Box::new(inner)));
    }

    // Register: %rax, %st(0), etc.
    if s.starts_with('%') {
        return parse_register_operand(s);
    }

    // Immediate: $42, $symbol, $symbol@GOTPCREL
    if let Some(rest) = s.strip_prefix('$') {
        return parse_immediate_operand(rest);
    }

    // Memory or label reference
    // Could be: offset(%base), symbol(%rip), (%base,%idx,scale), or just a label
    if s.contains('(') || s.contains(':') {
        return parse_memory_operand(s);
    }

    // Check if this is a bare numeric address like `42` (used in `movl 42, %eax`).
    // In AT&T syntax, a bare number without `$` prefix is an absolute memory address.
    // Only treat it as a memory operand if it parses as a pure integer and doesn't
    // look like a numeric label reference (e.g., `1f`, `1b`).
    if s.bytes().next().is_some_and(|c| c.is_ascii_digit() || c == b'-')
        && !s.ends_with('f') && !s.ends_with('b')
    {
        if let Ok(val) = crate::backend::asm_expr::parse_integer_expr(s) {
            return Ok(Operand::Memory(MemoryOperand {
                segment: None,
                displacement: Displacement::Integer(val),
                base: None,
                index: None,
                scale: None,
            }));
        }
    }

    // Plain label reference (for jmp/call targets)
    // Could be: .LBB42, funcname, funcname@PLT, 1f, 1b
    Ok(Operand::Label(s.to_string()))
}

/// Parse a register operand like %rax, %st(0).
fn parse_register_operand(s: &str) -> Result<Operand, String> {
    let name = &s[1..]; // strip %

    // Handle %st(N)
    if name.starts_with("st(") && name.ends_with(')') {
        return Ok(Operand::Register(Register::new(name)));
    }

    // Handle segment:memory patterns like %fs:0
    if let Some(colon_pos) = name.find(':') {
        let seg = &name[..colon_pos];
        let rest = &s[1 + colon_pos + 1..]; // after the colon
        if seg == "fs" || seg == "gs" {
            let mut mem = parse_memory_inner(rest)?;
            mem.segment = Some(seg.to_string());
            return Ok(Operand::Memory(mem));
        }
    }

    Ok(Operand::Register(Register::new(name)))
}

/// Parse an immediate operand (after the '$').
fn parse_immediate_operand(s: &str) -> Result<Operand, String> {
    let s = s.trim();

    // Strip outer parentheses used for grouping, e.g. $(init_top_pgt - 0xffffffff80000000)
    // In AT&T syntax, $(...) wraps the expression inside parens for grouping.
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        // Verify balanced parens
        let mut depth = 0i32;
        let mut balanced = true;
        for c in inner.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        balanced = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if balanced && depth == 0 {
            return parse_immediate_operand(inner);
        }
    }

    // Try integer
    if let Ok(val) = parse_integer_expr(s) {
        return Ok(Operand::Immediate(ImmediateValue::Integer(val)));
    }

    // Symbol with modifier: symbol@GOTPCREL, etc.
    if let Some(at_pos) = s.find('@') {
        let sym = s[..at_pos].to_string();
        let modifier = s[at_pos + 1..].to_string();
        return Ok(Operand::Immediate(ImmediateValue::SymbolMod(sym, modifier)));
    }

    // Check for symbol difference: SYM-LABEL (e.g., $_DYNAMIC-1b, $4f-1b)
    // Scan for '-' after position 0 where both sides look like labels.
    if let Some(diff) = parse_immediate_label_diff(s) {
        return Ok(Operand::Immediate(diff));
    }

    // Check for symbol+/-offset expression (e.g., init_top_pgt - 0xffffffff80000000)
    if let Some(sym_offset) = try_parse_immediate_symbol_offset(s) {
        return Ok(Operand::Immediate(sym_offset));
    }

    // Plain symbol
    Ok(Operand::Immediate(ImmediateValue::Symbol(s.to_string())))
}

/// Try to parse a symbol difference expression in an immediate value.
/// E.g., "_DYNAMIC-1b" -> SymbolDiff("_DYNAMIC", "1b"), "4f-1b" -> SymbolDiff("4f", "1b")
fn parse_immediate_label_diff(s: &str) -> Option<ImmediateValue> {
    // Scan for '-' after position 0. We skip position 0 since a leading '-'
    // is a negation, not a difference.
    for (i, c) in s.char_indices().skip(1) {
        if c == '-' {
            let lhs = &s[..i];
            let rhs = &s[i + 1..];
            if !lhs.is_empty() && !rhs.is_empty()
                && is_label_like(lhs) && is_label_like(rhs)
            {
                return Some(ImmediateValue::SymbolDiff(lhs.to_string(), rhs.to_string()));
            }
        }
    }
    None
}

/// Parse a memory operand like `offset(%base, %index, scale)` or `symbol(%rip)`.
fn parse_memory_operand(s: &str) -> Result<Operand, String> {
    // Check for segment prefix: %fs:..., %gs:...
    if s.starts_with('%') {
        if let Some(colon_pos) = s.find(':') {
            let seg = &s[1..colon_pos];
            if seg == "fs" || seg == "gs" {
                let rest = &s[colon_pos + 1..];
                let mut mem = parse_memory_inner(rest)?;
                mem.segment = Some(seg.to_string());
                return Ok(Operand::Memory(mem));
            }
        }
    }

    let mem = parse_memory_inner(s)?;
    Ok(Operand::Memory(mem))
}

/// Parse memory operand inner part: `offset(%base, %index, scale)`.
fn parse_memory_inner(s: &str) -> Result<MemoryOperand, String> {
    let s = s.trim();

    // Find the register-enclosing parentheses: the LAST balanced (...) group.
    // This handles nested parens in the displacement like ((6*8) + 8*16)(%rsp).
    // We scan backwards from the end to find the matching '(' for the final ')'.
    if let Some(paren_end) = s.rfind(')') {
        let mut depth = 0i32;
        let mut paren_start = None;
        for (i, c) in s[..=paren_end].char_indices().rev() {
            match c {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        paren_start = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let paren_start = paren_start
            .ok_or_else(|| format!("unmatched paren in memory operand: {}", s))?;
        let disp_str = s[..paren_start].trim();
        let inner = &s[paren_start + 1..paren_end];

        // If there's no displacement before the parens AND the inner content doesn't
        // look like register references (i.e., first part doesn't start with %),
        // then this is a parenthesized displacement expression like (pcpu_hot + 16),
        // not a register reference like (%rsp). Treat the whole thing as displacement.
        if disp_str.is_empty() && !inner.is_empty() {
            let first_part = inner.split(',').next().unwrap_or("").trim();
            if !first_part.starts_with('%') && !first_part.is_empty() {
                // This is a parenthesized expression used as displacement
                let displacement = parse_displacement(inner)?;
                return Ok(MemoryOperand {
                    segment: None,
                    displacement,
                    base: None,
                    index: None,
                    scale: None,
                });
            }
        }

        let displacement = parse_displacement(disp_str)?;

        // Parse base, index, scale from inside parens
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();

        let base = if !parts.is_empty() && !parts[0].is_empty() {
            Some(Register::new(parts[0].trim_start_matches('%')))
        } else {
            None
        };

        let index = if parts.len() > 1 && !parts[1].is_empty() {
            Some(Register::new(parts[1].trim_start_matches('%')))
        } else {
            None
        };

        let scale = if parts.len() > 2 && !parts[2].is_empty() {
            Some(parts[2].parse::<u8>()
                .map_err(|_| format!("bad scale: {}", parts[2]))?)
        } else {
            None
        };

        Ok(MemoryOperand {
            segment: None,
            displacement,
            base,
            index,
            scale,
        })
    } else {
        // No parens - could be just a displacement/symbol
        let displacement = parse_displacement(s)?;
        Ok(MemoryOperand {
            segment: None,
            displacement,
            base: None,
            index: None,
            scale: None,
        })
    }
}

/// Parse a displacement: integer, symbol, symbol+offset, or symbol@modifier.
fn parse_displacement(s: &str) -> Result<Displacement, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Displacement::None);
    }

    // Strip outer parentheses used for grouping expressions like (pcpu_hot + 16).
    // In AT&T syntax, parenthesized displacement expressions like %gs:(pcpu_hot + 16)(%rip)
    // produce a displacement string of "(pcpu_hot + 16)" that we need to unwrap.
    // Only strip if the entire string is wrapped in balanced parens.
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        // Verify the parens are balanced (not e.g. "(a)+(b)")
        let mut depth = 0i32;
        let mut balanced = true;
        for c in inner.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        balanced = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if balanced && depth == 0 {
            return parse_displacement(inner);
        }
    }

    // Try integer
    if let Ok(val) = parse_integer_expr(s) {
        return Ok(Displacement::Integer(val));
    }

    // Symbol with modifier: symbol@GOTPCREL, symbol@TPOFF, etc.
    if let Some(at_pos) = s.find('@') {
        let sym = s[..at_pos].to_string();
        let modifier = s[at_pos + 1..].to_string();
        return Ok(Displacement::SymbolMod(sym, modifier));
    }

    // Check for symbol+offset or symbol-offset (e.g., `.Lstr0+1`, `foo-4`)
    // Find the last '+' or '-' that's not at position 0 (to avoid splitting
    // negative numbers or labels starting with '.')
    if let Some(offset_disp) = try_parse_symbol_plus_offset(s) {
        return Ok(offset_disp);
    }

    // Plain symbol
    Ok(Displacement::Symbol(s.to_string()))
}

/// Try to parse a `symbol+offset` or `symbol-offset` expression.
/// Returns None if the string doesn't match this pattern.
fn try_parse_symbol_plus_offset(s: &str) -> Option<Displacement> {
    // Scan for '+' or '-' that separates the symbol from the offset.
    // Skip the first character to avoid splitting on leading sign/dot.
    for (i, c) in s.char_indices().skip(1) {
        if c == '+' || c == '-' {
            let left = s[..i].trim();
            let right_with_sign = s[i..].trim(); // includes the + or -
            if left.is_empty() {
                continue;
            }

            let is_valid_sym = |s: &str| -> bool {
                !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '$')
            };

            // Case 1: symbol+offset (e.g. "maxmin_avx+32")
            if let Ok(offset) = parse_integer_expr(right_with_sign) {
                if is_valid_sym(left) {
                    return Some(Displacement::SymbolPlusOffset(left.to_string(), offset));
                }
            }

            // Case 2: offset+symbol (e.g. "32+maxmin_avx") - only for '+'
            if c == '+' {
                if let Ok(offset) = parse_integer_expr(left) {
                    let sym = right_with_sign[1..].trim(); // skip the '+'
                    if is_valid_sym(sym) {
                        return Some(Displacement::SymbolPlusOffset(sym.to_string(), offset));
                    }
                }
            }
        }
    }
    None
}


/// Try to parse a `symbol+offset` or `symbol-offset` expression as an immediate value.
/// This handles cases like `init_top_pgt - 0xffffffff80000000` where the expression
/// mixes a symbol with a large integer constant.
fn try_parse_immediate_symbol_offset(s: &str) -> Option<ImmediateValue> {
    let is_valid_sym = |s: &str| -> bool {
        !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '$')
    };

    // Scan for '+' or '-' that separates the symbol from the offset.
    for (i, c) in s.char_indices().skip(1) {
        if c == '+' || c == '-' {
            let left = s[..i].trim();
            let right_with_sign = s[i..].trim();
            if left.is_empty() {
                continue;
            }

            // Case 1: symbol+/-offset (e.g. "init_top_pgt - 0xffffffff80000000")
            if let Ok(offset) = parse_integer_expr(right_with_sign) {
                if is_valid_sym(left) {
                    return Some(ImmediateValue::SymbolPlusOffset(left.to_string(), offset));
                }
            }

            // Case 2: offset+symbol (e.g. "32+init_top_pgt") - only for '+'
            if c == '+' {
                if let Ok(offset) = parse_integer_expr(left) {
                    let sym = right_with_sign[1..].trim();
                    if is_valid_sym(sym) {
                        return Some(ImmediateValue::SymbolPlusOffset(sym.to_string(), offset));
                    }
                }
            }
        }
    }
    None
}

/// Split `.skip expr, fill` arguments, respecting parentheses.
/// Returns the expression string and the fill byte (default 0).
fn split_skip_args(args: &str) -> (&str, u8) {
    let mut depth = 0i32;
    let mut last_comma = None;
    for (i, c) in args.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => last_comma = Some(i),
            _ => {}
        }
    }
    if let Some(pos) = last_comma {
        let expr = &args[..pos];
        let fill_str = args[pos + 1..].trim();
        let fill = if let Ok(v) = parse_integer_expr(fill_str) { v as u8 } else { 0u8 };
        (expr, fill)
    } else {
        (args, 0u8)
    }
}

/// Parse a label diff without spaces, e.g., `663b-661b`.
/// Returns a SymbolDiff DataValue if both sides look like labels.
fn parse_label_diff(s: &str) -> Option<DataValue> {
    for (i, c) in s.char_indices().skip(1) {
        if c == '-' {
            let lhs = &s[..i];
            let rhs = &s[i + 1..];
            if !lhs.is_empty() && !rhs.is_empty()
                && is_label_like(lhs) && is_label_like(rhs) {
                return Some(DataValue::SymbolDiff(lhs.to_string(), rhs.to_string()));
            }
        }
    }
    None
}

/// Check if a string looks like a label (alphanumeric, underscore, dot).
fn is_label_like(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.as_bytes()[0];
    if !(first.is_ascii_alphabetic() || first == b'_' || first == b'.' || first.is_ascii_digit()) {
        return false;
    }
    s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.')
}

/// Strip balanced outer parentheses from an expression.
/// E.g. "((1b) - .)" -> "(1b) - ." -> "1b - ." (recursive).
fn strip_outer_parens(s: &str) -> &str {
    let s = s.trim();
    if !s.starts_with('(') || !s.ends_with(')') {
        return s;
    }
    let inner = &s[1..s.len() - 1];
    let mut depth = 0i32;
    for ch in inner.bytes() {
        match ch {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth < 0 {
                    return s;
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        strip_outer_parens(inner)
    } else {
        s
    }
}

/// Strip outer parentheses from a symbol name. E.g. "(1b)" -> "1b".
fn strip_sym_parens(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        let mut depth = 0i32;
        for ch in inner.bytes() {
            match ch {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth < 0 {
                        return s;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 { strip_sym_parens(inner) } else { s }
    } else {
        s
    }
}

/// Parse data values (integers or symbol references).
fn parse_data_values(s: &str) -> Result<Vec<DataValue>, String> {
    let mut vals = Vec::new();
    for part in s.split(',') {
        let trimmed = strip_outer_parens(part.trim());
        if trimmed.is_empty() {
            continue;
        }

        // Check for symbol difference: .LBB3 - .Ljt_0, or with addend: tr_gdt_end - tr_gdt - 1
        if let Some(minus_pos) = trimmed.find(" - ") {
            let lhs = strip_sym_parens(trimmed[..minus_pos].trim()).to_string();
            let rhs_full = trimmed[minus_pos + 3..].trim();
            // Check if rhs has an addend: "sym - N" or "sym + N"
            if let Some(rhs_minus) = rhs_full.rfind(" - ") {
                let rhs_sym = strip_sym_parens(rhs_full[..rhs_minus].trim());
                let rhs_add = rhs_full[rhs_minus + 3..].trim();
                if is_label_like(rhs_sym) {
                    if let Ok(addend) = parse_integer_expr(rhs_add) {
                        vals.push(DataValue::SymbolDiffAddend(lhs, rhs_sym.to_string(), -addend));
                        continue;
                    }
                }
            }
            if let Some(rhs_plus) = rhs_full.rfind(" + ") {
                let rhs_sym = strip_sym_parens(rhs_full[..rhs_plus].trim());
                let rhs_add = rhs_full[rhs_plus + 3..].trim();
                if is_label_like(rhs_sym) {
                    if let Ok(addend) = parse_integer_expr(rhs_add) {
                        vals.push(DataValue::SymbolDiffAddend(lhs, rhs_sym.to_string(), addend));
                        continue;
                    }
                }
            }
            vals.push(DataValue::SymbolDiff(lhs, strip_sym_parens(rhs_full).to_string()));
            continue;
        }

        // Try integer
        if let Ok(val) = parse_integer_expr(trimmed) {
            vals.push(DataValue::Integer(val));
            continue;
        }

        // Check for label diff without spaces (e.g., 663b-661b)
        if let Some(val) = parse_label_diff(trimmed) {
            vals.push(val);
            continue;
        }

        // Check for symbol+offset or symbol-offset (e.g., GD_struct+128, arr+33)
        if let Some(val) = parse_symbol_offset(trimmed) {
            vals.push(val);
            continue;
        }

        // Symbol reference
        vals.push(DataValue::Symbol(trimmed.to_string()));
    }
    Ok(vals)
}

/// Parse symbol+offset or symbol-offset expressions (e.g., GD_struct+128).
/// Also handles offset+symbol (e.g., 0x9b000000 + pa_real_mode_base).
/// Returns a DataValue::SymbolOffset if the string matches this pattern.
fn parse_symbol_offset(s: &str) -> Option<DataValue> {
    // Look for + or - that separates symbol from offset
    // Don't match the leading character (could be .-prefixed label)
    for (i, c) in s.char_indices().skip(1) {
        if c == '+' || c == '-' {
            let left = s[..i].trim();
            let right_with_sign = &s[i..]; // includes the sign

            // Case 1: symbol+offset or symbol-offset (e.g., "GD_struct+128")
            if let Ok(offset) = parse_integer_expr(right_with_sign) {
                if !left.is_empty() && !left.contains(' ') {
                    return Some(DataValue::SymbolOffset(left.to_string(), offset));
                }
            }

            // Case 2: offset+symbol (e.g., "0x9b000000 + pa_real_mode_base") - only for '+'
            if c == '+' {
                if let Ok(offset) = parse_integer_expr(left) {
                    let sym = right_with_sign[1..].trim(); // skip the '+'
                    if !sym.is_empty() && !sym.contains(' ') && is_label_like(sym) {
                        return Some(DataValue::SymbolOffset(sym.to_string(), offset));
                    }
                }
            }
        }
    }
    None
}

/// Parse an integer expression (delegates to the shared expression evaluator).
fn parse_integer_expr(s: &str) -> Result<i64, String> {
    asm_expr::parse_integer_expr(s)
}

/// GAS macro definition
#[derive(Clone, Debug)]
struct GasMacro {
    params: Vec<(String, Option<String>)>, // (name, default_value)
    body: Vec<String>,
}

/// Expand GAS macro directives: .macro/.endm, .purgem, .irp/.endr, .ifc/.endif, .if/.endif, .set
///
/// This runs as a text-level expansion pass before instruction parsing.
/// It handles the Linux kernel's extable_type_reg pattern and similar constructs.
fn expand_gas_macros(lines: &[String]) -> Result<Vec<String>, String> {
    let mut macros = std::collections::HashMap::new();
    let mut symbols = std::collections::HashMap::new();
    expand_gas_macros_with_state(lines, &mut macros, &mut symbols)
}

fn expand_gas_macros_with_state(
    lines: &[String],
    macros: &mut std::collections::HashMap<String, GasMacro>,
    symbols: &mut std::collections::HashMap<String, i64>,
) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = strip_comment(&lines[i]).trim().to_string();

        // .macro name param1:req param2:req ...
        if trimmed.starts_with(".macro ") || trimmed.starts_with(".macro\t") {
            let rest = trimmed[".macro".len()..].trim();
            let (name, params) = parse_macro_def(rest)?;
            let mut body = Vec::new();
            let mut depth = 1;
            i += 1;
            while i < lines.len() {
                let inner = strip_comment(&lines[i]).trim().to_string();
                if inner.starts_with(".macro ") || inner.starts_with(".macro\t") {
                    depth += 1;
                } else if inner == ".endm" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                body.push(lines[i].clone());
                i += 1;
            }
            if depth != 0 {
                return Err(".macro without matching .endm".to_string());
            }
            macros.insert(name, GasMacro { params, body });
            i += 1;
            continue;
        }

        // .purgem name
        if trimmed.starts_with(".purgem ") || trimmed.starts_with(".purgem\t") {
            let name = trimmed[".purgem".len()..].trim().to_string();
            macros.remove(&name);
            i += 1;
            continue;
        }

        // .set symbol, expr (with symbol resolution)
        if trimmed.starts_with(".set ") || trimmed.starts_with(".set\t") {
            let rest = trimmed[".set".len()..].trim();
            if let Some(comma_pos) = rest.find(',') {
                let sym_name = rest[..comma_pos].trim().to_string();
                let expr_str = rest[comma_pos+1..].trim();
                // Try to evaluate expression with current symbol values
                let resolved = resolve_set_expr(expr_str, symbols);
                if let Ok(val) = parse_integer_expr(&resolved) {
                    symbols.insert(sym_name.clone(), val);
                    // For local symbols (.L*), don't emit the .set - we handle them internally
                    if sym_name.starts_with(".L") {
                        i += 1;
                        continue;
                    }
                }
                // Emit the .set directive for non-local or unresolvable symbols
                let resolved_line = format!(".set {}, {}", sym_name, resolved);
                result.push(resolved_line);
            } else {
                result.push(lines[i].clone());
            }
            i += 1;
            continue;
        }

        // GAS-style symbol = expr assignment (equivalent to .set symbol, expr)
        // Used by kernel code like: i = 0 / i = i + 1 inside .rept blocks
        if let Some(eq_pos) = trimmed.find('=') {
            // Make sure it's not ==, !=, <=, >= (comparison operators)
            let not_comparison = (eq_pos + 1 >= trimmed.len() || trimmed.as_bytes()[eq_pos + 1] != b'=')
                && (eq_pos == 0 || (trimmed.as_bytes()[eq_pos - 1] != b'!'
                    && trimmed.as_bytes()[eq_pos - 1] != b'<'
                    && trimmed.as_bytes()[eq_pos - 1] != b'>'));
            if not_comparison {
                let before = trimmed[..eq_pos].trim();
                // Check if before looks like a symbol name: starts with letter or _, no spaces
                if !before.is_empty()
                    && !before.contains(' ')
                    && !before.contains('\t')
                    && !before.contains(':')
                    && !before.starts_with('$')
                    && !before.starts_with('%')
                    && !before.starts_with('.')
                    && (before.as_bytes()[0].is_ascii_alphabetic() || before.as_bytes()[0] == b'_')
                {
                    let expr_str = trimmed[eq_pos + 1..].trim();
                    let resolved = resolve_set_expr(expr_str, symbols);
                    if let Ok(val) = parse_integer_expr(&resolved) {
                        symbols.insert(before.to_string(), val);
                        // Emit as .set so the parser can also handle it
                        let resolved_line = format!(".set {}, {}", before, resolved);
                        result.push(resolved_line);
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // .irp var, item1, item2, ...  /  .endr
        if trimmed.starts_with(".irp ") || trimmed.starts_with(".irp\t") {
            let rest = trimmed[".irp".len()..].trim();
            let (var, items) = parse_irp_header(rest)?;
            let mut body = Vec::new();
            let mut depth = 1;
            i += 1;
            while i < lines.len() {
                let inner = strip_comment(&lines[i]).trim().to_string();
                if inner.starts_with(".irp ") || inner.starts_with(".irp\t")
                    || inner.starts_with(".rept ") || inner.starts_with(".rept\t") {
                    depth += 1;
                } else if inner == ".endr" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                body.push(lines[i].clone());
                i += 1;
            }
            // Expand: for each item, substitute \var with the item, then recursively process
            let mut all_expanded = Vec::new();
            for item in &items {
                for bline in &body {
                    let mut expanded = asm_preprocess::replace_macro_param(bline, &format!("\\{}", var), item);
                    // Strip GAS macro argument delimiters: \() resolves to empty string
                    expanded = expanded.replace("\\()", "");
                    all_expanded.push(expanded);
                }
            }
            let processed = expand_gas_macros_with_state(&all_expanded, macros, symbols)?;
            result.extend(processed);
            i += 1;
            continue;
        }

        // .if expr / .elseif expr / .else / .endif
        if trimmed.starts_with(".if ") || trimmed.starts_with(".if\t") || trimmed.starts_with(".if(") {
            let rest = if trimmed.starts_with(".if(") {
                &trimmed[".if".len()..]
            } else {
                trimmed[".if".len()..].trim()
            };
            let cond = eval_if_expr(rest, symbols);
            // Collect branches: a chain of (condition, lines) pairs ending with optional else
            let mut branches: Vec<(bool, Vec<String>)> = vec![(cond, Vec::new())];
            let mut current_idx = 0;
            let mut depth = 1;
            i += 1;
            while i < lines.len() {
                let inner = strip_comment(&lines[i]).trim().to_string();
                if is_if_start(&inner) {
                    depth += 1;
                    branches[current_idx].1.push(lines[i].clone());
                } else if inner == ".endif" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    branches[current_idx].1.push(lines[i].clone());
                } else if depth == 1 && (inner.starts_with(".elseif ") || inner.starts_with(".elseif\t")) {
                    let elseif_rest = inner[".elseif".len()..].trim();
                    // All branch conditions are evaluated eagerly; harmless for pure comparisons.
                    let elseif_cond = eval_if_expr(elseif_rest, symbols);
                    branches.push((elseif_cond, Vec::new()));
                    current_idx += 1;
                } else if inner == ".else" && depth == 1 {
                    // .else is like .elseif with condition=true (fallback)
                    branches.push((true, Vec::new()));
                    current_idx += 1;
                } else {
                    branches[current_idx].1.push(lines[i].clone());
                }
                i += 1;
            }
            // Choose the first branch whose condition is true
            let empty: Vec<String> = Vec::new();
            let mut chosen_lines: &Vec<String> = &empty;
            for (bcond, blines) in &branches {
                if *bcond {
                    chosen_lines = blines;
                    break;
                }
            }
            let expanded = expand_gas_macros_with_state(chosen_lines, macros, symbols)?;
            result.extend(expanded);
            i += 1;
            continue;
        }

        // .ifc str1, str2 / .endif
        if trimmed.starts_with(".ifc ") || trimmed.starts_with(".ifc\t") {
            let rest = trimmed[".ifc".len()..].trim();
            let cond = eval_ifc(rest);
            let mut branches: Vec<(bool, Vec<String>)> = vec![(cond, Vec::new())];
            let mut current_idx = 0;
            let mut depth = 1;
            i += 1;
            while i < lines.len() {
                let inner = strip_comment(&lines[i]).trim().to_string();
                if is_if_start(&inner) {
                    depth += 1;
                    branches[current_idx].1.push(lines[i].clone());
                } else if inner == ".endif" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    branches[current_idx].1.push(lines[i].clone());
                } else if depth == 1 && (inner.starts_with(".elseif ") || inner.starts_with(".elseif\t")) {
                    let elseif_rest = inner[".elseif".len()..].trim();
                    // All branch conditions are evaluated eagerly; harmless for pure comparisons.
                    let elseif_cond = eval_if_expr(elseif_rest, symbols);
                    branches.push((elseif_cond, Vec::new()));
                    current_idx += 1;
                } else if inner == ".else" && depth == 1 {
                    branches.push((true, Vec::new()));
                    current_idx += 1;
                } else {
                    branches[current_idx].1.push(lines[i].clone());
                }
                i += 1;
            }
            let empty: Vec<String> = Vec::new();
            let mut chosen_lines: &Vec<String> = &empty;
            for (bcond, blines) in &branches {
                if *bcond {
                    chosen_lines = blines;
                    break;
                }
            }
            let expanded = expand_gas_macros_with_state(chosen_lines, macros, symbols)?;
            result.extend(expanded);
            i += 1;
            continue;
        }

        // .ifnb arg / .ifb arg / .endif — test if macro argument is non-blank / blank
        if trimmed == ".ifnb" || trimmed == ".ifb"
            || trimmed.starts_with(".ifnb ") || trimmed.starts_with(".ifnb\t")
            || trimmed.starts_with(".ifb ") || trimmed.starts_with(".ifb\t")
        {
            let is_ifnb = trimmed.starts_with(".ifnb");
            let dir_len = if is_ifnb { ".ifnb".len() } else { ".ifb".len() };
            let arg = if trimmed.len() > dir_len { trimmed[dir_len..].trim() } else { "" };
            let cond = if is_ifnb { !arg.is_empty() } else { arg.is_empty() };
            let mut branches: Vec<(bool, Vec<String>)> = vec![(cond, Vec::new())];
            let mut current_idx = 0;
            let mut depth = 1;
            i += 1;
            while i < lines.len() {
                let inner = strip_comment(&lines[i]).trim().to_string();
                if is_if_start(&inner) {
                    depth += 1;
                    branches[current_idx].1.push(lines[i].clone());
                } else if inner == ".endif" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    branches[current_idx].1.push(lines[i].clone());
                } else if depth == 1 && (inner.starts_with(".elseif ") || inner.starts_with(".elseif\t")) {
                    let elseif_rest = inner[".elseif".len()..].trim();
                    let elseif_cond = eval_if_expr(elseif_rest, symbols);
                    branches.push((elseif_cond, Vec::new()));
                    current_idx += 1;
                } else if inner == ".else" && depth == 1 {
                    branches.push((true, Vec::new()));
                    current_idx += 1;
                } else {
                    branches[current_idx].1.push(lines[i].clone());
                }
                i += 1;
            }
            let empty: Vec<String> = Vec::new();
            let mut chosen_lines: &Vec<String> = &empty;
            for (bcond, blines) in &branches {
                if *bcond {
                    chosen_lines = blines;
                    break;
                }
            }
            let expanded = expand_gas_macros_with_state(chosen_lines, macros, symbols)?;
            result.extend(expanded);
            i += 1;
            continue;
        }

        // .error "message" - assembler error directive
        if trimmed.starts_with(".error ") || trimmed.starts_with(".error\t") {
            return Err(format!("assembler error: {}", trimmed[".error".len()..].trim()));
        }

        // Check if line is a macro invocation.
        // First, split on semicolons to handle "macrocall args; other_stuff" patterns.
        // GAS treats ';' as a statement separator, so a macro invocation may be
        // followed by other statements on the same line.
        let semi_parts = crate::backend::asm_preprocess::split_on_semicolons(&trimmed);
        let first_part = semi_parts[0].trim();
        let first_word = first_part.split_whitespace().next().unwrap_or("");
        // Strip label prefix if present (e.g., "label: macroname args")
        let macro_name_candidate = if first_word.ends_with(':') {
            // There might be a macro after the label
            first_part[first_word.len()..].split_whitespace().next().unwrap_or("")
        } else {
            first_word
        };
        // Also check for C-preprocessor-style invocations: MACRO_NAME(args)
        // In GAS, "MACRO_NAME(arg)" is treated as macro invocation with "(arg)" as arg text
        let macro_name = if macros.contains_key(macro_name_candidate) {
            macro_name_candidate
        } else if let Some(paren_pos) = macro_name_candidate.find('(') {
            let candidate = &macro_name_candidate[..paren_pos];
            if macros.contains_key(candidate) { candidate } else { macro_name_candidate }
        } else {
            macro_name_candidate
        };

        if macros.contains_key(macro_name) {
            let mac = macros[macro_name].clone();
            let args_str = if first_word.ends_with(':') {
                // Label before macro
                let after_label = first_part[first_word.len()..].trim();
                let after_name = after_label[macro_name.len()..].trim();
                // Emit label first
                result.push(first_word.to_string());
                after_name
            } else {
                first_part[macro_name.len()..].trim()
            };
            let args = parse_macro_args(args_str, &mac.params)?;
            // Sort parameters by name length (longest first) to avoid partial
            // substitution: e.g., \orig must not match before \orig_len.
            let mut sorted_args: Vec<(String, String)> = args.clone();
            sorted_args.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
            // Substitute parameters in body
            let mut expanded_body: Vec<String> = mac.body.iter().map(|line| {
                let mut l = line.clone();
                for (pname, pval) in &sorted_args {
                    l = asm_preprocess::replace_macro_param(&l, &format!("\\{}", pname), pval);
                }
                // Strip GAS macro argument delimiters: \() resolves to empty string.
                // Used to separate parameter names from adjacent text,
                // e.g., \op\()_safe_regs -> rdmsr_safe_regs
                l = l.replace("\\()", "");
                l
            }).collect();
            // Recursively expand the body (handles nested .irp, .set, .if, etc.)
            expanded_body = expand_gas_macros_with_state(&expanded_body, macros, symbols)?;
            result.extend(expanded_body);
            // Emit remaining semicolon-separated parts as separate lines
            for sp in &semi_parts[1..] {
                let sp = sp.trim();
                if !sp.is_empty() {
                    result.push(sp.to_string());
                }
            }
            i += 1;
            continue;
        }

        // For regular lines, resolve any .set symbol references
        if !symbols.is_empty() {
            let resolved = resolve_set_expr(&lines[i], symbols);
            result.push(resolved);
        } else {
            result.push(lines[i].clone());
        }
        i += 1;
    }

    Ok(result)
}

/// Parse a .macro definition header: "name param1:req param2:req ..."
fn parse_macro_def(rest: &str) -> Result<(String, Vec<(String, Option<String>)>), String> {
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.is_empty() {
        return Err(".macro: missing name".to_string());
    }
    let name = parts[0].to_string();
    let mut params = Vec::new();
    for &part in &parts[1..] {
        // Handle param:req, param=default, or just param
        let part = part.trim_end_matches(',');
        if part.contains(':') {
            let pname = part.split(':').next().unwrap().to_string();
            params.push((pname, None));
        } else if part.contains('=') {
            let mut split = part.splitn(2, '=');
            let pname = split.next().unwrap().to_string();
            let default = split.next().map(|s| s.to_string());
            params.push((pname, default));
        } else {
            params.push((part.to_string(), None));
        }
    }
    Ok((name, params))
}

/// Parse macro invocation arguments: "param1=val1, param2=val2" or positional
fn parse_macro_args(args_str: &str, params: &[(String, Option<String>)]) -> Result<Vec<(String, String)>, String> {
    let mut result = Vec::new();
    if args_str.is_empty() {
        // Use defaults for all params
        for (name, default) in params {
            result.push((name.clone(), default.clone().unwrap_or_default()));
        }
        return Ok(result);
    }

    // Split on commas (respecting parentheses)
    let arg_parts = split_macro_args(args_str);

    // Helper: strip surrounding quotes from GAS macro arguments
    // In GAS, "" means empty string, "foo bar" means foo bar
    let strip_quotes = |s: &str| -> String {
        let t = s.trim();
        if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
            t[1..t.len()-1].to_string()
        } else {
            t.to_string()
        }
    };

    // GAS supports mixed positional and named arguments.
    // Positional args fill params left-to-right, named args (key=val) override by name.
    // Example: "0 asm_foo exc_foo has_error_code=0" with params (vector, asmsym, cfunc, has_error_code)
    // → positional: vector=0, asmsym=asm_foo, cfunc=exc_foo; named: has_error_code=0
    let mut positional_idx = 0;
    let mut arg_map = std::collections::HashMap::new();
    let mut positional_vals: Vec<(usize, String)> = Vec::new();

    for part in &arg_parts {
        let part = part.trim();
        if let Some(eq_pos) = part.find('=') {
            let key = part[..eq_pos].trim();
            // Only treat as named if the key matches a known parameter name
            if params.iter().any(|(pname, _)| pname == key) {
                let val = strip_quotes(part[eq_pos+1..].trim());
                arg_map.insert(key.to_string(), val);
                continue;
            }
        }
        // Positional argument
        positional_vals.push((positional_idx, strip_quotes(part)));
        positional_idx += 1;
    }

    // Build result: start with positional, then override with named
    let mut pos_iter = positional_vals.into_iter();
    for (name, default) in params {
        if let Some(val) = arg_map.get(name) {
            result.push((name.clone(), val.clone()));
        } else if let Some((_, val)) = pos_iter.next() {
            result.push((name.clone(), val));
        } else {
            result.push((name.clone(), default.clone().unwrap_or_default()));
        }
    }

    Ok(result)
}

/// Split macro arguments on commas and spaces, respecting parentheses.
///
/// GAS allows both commas and spaces/tabs as macro argument separators.
/// e.g., `base=%rsp offset=0` or `base=%rsp, offset=0` both give two args.
/// Consecutive separators are collapsed (no empty args from double spaces).
fn split_macro_args(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut in_quotes = false;
    let mut current = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            '(' if !in_quotes => { depth += 1; current.push(ch); }
            ')' if !in_quotes => { depth -= 1; current.push(ch); }
            ',' if depth == 0 && !in_quotes => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
                current.clear();
                // Skip any whitespace after comma
                while i + 1 < bytes.len() && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\t') {
                    i += 1;
                }
            }
            ' ' | '\t' if depth == 0 && !in_quotes => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                    current.clear();
                }
                // Skip remaining whitespace
                while i + 1 < bytes.len() && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\t') {
                    i += 1;
                }
                // If next char is comma, let the comma handler deal with it
            }
            _ => current.push(ch),
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }
    parts
}

/// Parse .irp header: "var, item1, item2, ..."
fn parse_irp_header(rest: &str) -> Result<(String, Vec<String>), String> {
    // Format: var,item1,item2,... or var item1,item2,...
    let rest = rest.trim();
    // First find the variable name (before the first comma)
    let comma_pos = rest.find(',').ok_or(".irp: missing comma after variable")?;
    let var = rest[..comma_pos].trim().to_string();
    let items_str = rest[comma_pos+1..].trim();
    let items: Vec<String> = items_str.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Ok((var, items))
}

/// Evaluate .ifc string comparison: "str1, str2" => true if equal
fn eval_ifc(rest: &str) -> bool {
    if let Some(comma_pos) = rest.find(',') {
        let s1 = rest[..comma_pos].trim();
        let s2 = rest[comma_pos+1..].trim();
        s1 == s2
    } else {
        false
    }
}

/// Resolve symbols in a .set expression string.
/// Uses whole-word matching to avoid replacing substrings inside identifiers,
/// register names, or instruction mnemonics (e.g., replacing `i` inside `rip`).
fn resolve_set_expr(expr: &str, symbols: &std::collections::HashMap<String, i64>) -> String {
    let mut result = expr.to_string();
    // Sort by length (longest first) to avoid partial replacements
    let mut sym_list: Vec<_> = symbols.iter().collect();
    sym_list.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (name, val) in sym_list {
        result = replace_whole_word(&result, name, &val.to_string());
    }
    result
}

/// Replace all whole-word occurrences of `word` with `replacement`.
/// A word boundary is any position where the adjacent character is not
/// alphanumeric or underscore (i.e., not an identifier character).
fn replace_whole_word(text: &str, word: &str, replacement: &str) -> String {
    if word.is_empty() || text.is_empty() || text.len() < word.len() {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let word_bytes = word.as_bytes();
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i + word_bytes.len() <= bytes.len() {
        if &bytes[i..i + word_bytes.len()] == word_bytes {
            // Check left boundary: either start of string or non-identifier char
            let left_ok = i == 0 || !is_ident_char(bytes[i - 1]);
            // Check right boundary: either end of string or non-identifier char
            let right_ok = i + word_bytes.len() >= bytes.len()
                || !is_ident_char(bytes[i + word_bytes.len()]);
            if left_ok && right_ok {
                result.push_str(replacement);
                i += word_bytes.len();
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    // Append any remaining characters
    while i < bytes.len() {
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Check if a byte is an identifier character (alphanumeric or underscore).
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Check if a line starts a new conditional assembly block (.if, .ifc, .ifdef, .ifndef).
fn is_if_start(trimmed: &str) -> bool {
    trimmed.starts_with(".if ") || trimmed.starts_with(".if\t") || trimmed.starts_with(".if(")
        || trimmed.starts_with(".ifc ") || trimmed.starts_with(".ifc\t")
        || trimmed.starts_with(".ifdef ") || trimmed.starts_with(".ifndef ")
        || trimmed == ".ifnb" || trimmed.starts_with(".ifnb ") || trimmed.starts_with(".ifnb\t")
        || trimmed == ".ifb" || trimmed.starts_with(".ifb ") || trimmed.starts_with(".ifb\t")
}

/// Evaluate a `.if` expression for the x86 assembler.
///
/// Resolves `.set` symbols and x86 register names (%rsp, %rbp, etc.) before
/// evaluating the expression. Supports `==`, `!=`, `>=`, `<=`, `>`, `<`
/// comparison operators and full arithmetic.
fn eval_if_expr(expr: &str, symbols: &std::collections::HashMap<String, i64>) -> bool {
    let resolved = resolve_set_expr(expr, symbols);
    asm_preprocess::eval_if_condition_with_resolver(&resolved, |s| {
        asm_preprocess::resolve_x86_registers(s)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let asm = r#"
.section .text
.globl main
.type main, @function
main:
.cfi_startproc
    pushq %rbp
    movq %rsp, %rbp
    xorl %eax, %eax
    popq %rbp
    ret
.cfi_endproc
.size main, .-main
"#;
        let items = parse_asm(asm).unwrap();
        // Should parse without errors
        let labels: Vec<_> = items.iter().filter(|i| matches!(i, AsmItem::Label(_))).collect();
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn test_parse_memory_operand() {
        let mem = parse_memory_inner("-8(%rbp)").unwrap();
        assert!(matches!(mem.displacement, Displacement::Integer(-8)));
        assert_eq!(mem.base.as_ref().unwrap().name, "rbp");
    }

    #[test]
    fn test_parse_rip_relative() {
        let mem = parse_memory_inner("x(%rip)").unwrap();
        assert!(matches!(mem.displacement, Displacement::Symbol(ref s) if s == "x"));
        assert_eq!(mem.base.as_ref().unwrap().name, "rip");
    }

    #[test]
    fn test_parse_integer() {
        assert_eq!(parse_integer_expr("42").unwrap(), 42);
        assert_eq!(parse_integer_expr("-1").unwrap(), -1);
        assert_eq!(parse_integer_expr("0xff").unwrap(), 255);
        assert_eq!(parse_integer_expr("0").unwrap(), 0);
    }
}
