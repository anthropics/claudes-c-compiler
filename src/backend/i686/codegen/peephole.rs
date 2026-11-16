//! i686 peephole optimizer for assembly text.
//!
//! Operates on generated assembly text to eliminate redundant patterns from the
//! stack-based codegen. Adapted from the x86-64 peephole optimizer for 32-bit
//! i686 assembly (uses %ebp instead of %rbp, %eax instead of %rax, etc.).
//!
//! ## Pass structure
//!
//! 1. **Local passes** (iterative, up to 8 rounds): adjacent store/load elimination,
//!    self-move elimination, redundant jump elimination, branch inversion, reverse
//!    move elimination.
//!
//! 2. **Global passes** (once): dead register move elimination, dead store elimination,
//!    compare+branch fusion, memory operand folding.
//!
//! 3. **Local cleanup** (up to 4 rounds): re-run local and global passes to clean up
//!    opportunities exposed by the first round.
//!
//! 4. **Never-read store elimination**: global analysis to remove stores to
//!    stack slots that are never read anywhere in the function.

// ── Constants ────────────────────────────────────────────────────────────────

const MAX_LOCAL_PASS_ITERATIONS: usize = 8;
const MAX_POST_GLOBAL_ITERATIONS: usize = 4;

// Register IDs (i686 has fewer registers)
type RegId = u8;
const REG_NONE: RegId = 255;
const REG_EAX: RegId = 0;
const REG_ECX: RegId = 1;
const REG_EDX: RegId = 2;
const REG_EBX: RegId = 3;
const REG_ESP: RegId = 4;
const REG_EBP: RegId = 5;
const REG_ESI: RegId = 6;
const REG_EDI: RegId = 7;
const REG_GP_MAX: RegId = 7;

/// Sentinel value for ebp_offset meaning "no %ebp reference" or "complex reference".
const EBP_OFFSET_NONE: i32 = i32::MIN;

// ── Line classification ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineKind {
    Nop,
    Empty,
    StoreEbp { reg: RegId, offset: i32, size: MoveSize },
    LoadEbp  { reg: RegId, offset: i32, size: MoveSize },
    Move { dst: RegId, src: RegId },
    SelfMove,
    Label,
    Jmp,
    JmpIndirect,
    CondJmp,
    Call,
    Ret,
    Push { reg: RegId },
    Pop { reg: RegId },
    SetCC { reg: RegId },
    Cmp,
    Directive,
    Other { dest_reg: RegId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveSize {
    L,   // movl (32-bit)
    W,   // movw (16-bit)
    B,   // movb (8-bit)
}

impl MoveSize {
    fn mnemonic(self) -> &'static str {
        match self {
            MoveSize::L => "movl",
            MoveSize::W => "movw",
            MoveSize::B => "movb",
        }
    }
    fn byte_size(self) -> i32 {
        match self {
            MoveSize::L => 4,
            MoveSize::W => 2,
            MoveSize::B => 1,
        }
    }
}

/// Check if two byte ranges `[a, a+a_size)` and `[b, b+b_size)` overlap.
#[inline]
fn ranges_overlap(a_off: i32, a_size: i32, b_off: i32, b_size: i32) -> bool {
    a_off < b_off + b_size && b_off < a_off + a_size
}

#[derive(Clone, Copy)]
struct LineInfo {
    kind: LineKind,
    trim_start: u16,
    has_indirect_mem: bool,
    ebp_offset: i32,
}

impl LineInfo {
    #[inline]
    fn is_nop(self) -> bool { self.kind == LineKind::Nop }
    #[inline]
    fn is_barrier(self) -> bool {
        matches!(self.kind,
            LineKind::Label | LineKind::Call | LineKind::Jmp | LineKind::JmpIndirect |
            LineKind::CondJmp | LineKind::Ret | LineKind::Directive)
    }
}

#[inline]
fn line_info(kind: LineKind, ts: u16) -> LineInfo {
    LineInfo { kind, trim_start: ts, has_indirect_mem: false, ebp_offset: EBP_OFFSET_NONE }
}

// ── Register parsing ─────────────────────────────────────────────────────────

/// Map i686 register name to family ID.
fn register_family(name: &str) -> RegId {
    let name = name.trim_start_matches('%');
    match name {
        "eax" | "ax" | "al" | "ah" => REG_EAX,
        "ecx" | "cx" | "cl" | "ch" => REG_ECX,
        "edx" | "dx" | "dl" | "dh" => REG_EDX,
        "ebx" | "bx" | "bl" | "bh" => REG_EBX,
        "esp" | "sp" => REG_ESP,
        "ebp" | "bp" => REG_EBP,
        "esi" | "si" => REG_ESI,
        "edi" | "di" => REG_EDI,
        _ => REG_NONE,
    }
}

/// Get the 32-bit register name for a family ID.
fn reg32_name(id: RegId) -> &'static str {
    match id {
        REG_EAX => "%eax",
        REG_ECX => "%ecx",
        REG_EDX => "%edx",
        REG_EBX => "%ebx",
        REG_ESP => "%esp",
        REG_EBP => "%ebp",
        REG_ESI => "%esi",
        REG_EDI => "%edi",
        _ => "%???",
    }
}

/// Check if a register is caller-saved (clobbered by calls).
fn is_caller_saved(reg: RegId) -> bool {
    matches!(reg, REG_EAX | REG_ECX | REG_EDX)
}

// ── Store/Load parsing ───────────────────────────────────────────────────────

/// Parse `movX %reg, offset(%ebp)` → (reg_name, offset_str, MoveSize)
fn parse_store_to_ebp(s: &str) -> Option<(&str, &str, MoveSize)> {
    let (rest, size) = if let Some(r) = s.strip_prefix("movl ") {
        (r, MoveSize::L)
    } else if let Some(r) = s.strip_prefix("movw ") {
        (r, MoveSize::W)
    } else if let Some(r) = s.strip_prefix("movb ") {
        (r, MoveSize::B)
    } else {
        return None;
    };
    // rest = "%eax, -8(%ebp)"
    let rest = rest.trim();
    if !rest.starts_with('%') { return None; }
    let comma = rest.find(',')?;
    let reg = &rest[..comma];
    let mem = rest[comma + 1..].trim();
    if !mem.ends_with("(%ebp)") { return None; }
    // Reject indirect memory (pointer dereference, not stack slot)
    if mem.contains("(%e") && !mem.ends_with("(%ebp)") { return None; }
    let offset_str = &mem[..mem.len() - 6]; // strip "(%ebp)"
    Some((reg.trim(), offset_str, size))
}

/// Parse `movX offset(%ebp), %reg` → (offset_str, reg_name, MoveSize)
fn parse_load_from_ebp(s: &str) -> Option<(&str, &str, MoveSize)> {
    let (rest, size) = if let Some(r) = s.strip_prefix("movl ") {
        (r, MoveSize::L)
    } else if let Some(r) = s.strip_prefix("movw ") {
        (r, MoveSize::W)
    } else if let Some(r) = s.strip_prefix("movb ") {
        (r, MoveSize::B)
    } else if let Some(r) = s.strip_prefix("movzbl ") {
        (r, MoveSize::L) // movzbl from stack, dest is 32-bit
    } else if let Some(r) = s.strip_prefix("movzwl ") {
        (r, MoveSize::L)
    } else if let Some(r) = s.strip_prefix("movsbl ") {
        (r, MoveSize::L)
    } else if let Some(r) = s.strip_prefix("movswl ") {
        (r, MoveSize::L)
    } else {
        return None;
    };
    let rest = rest.trim();
    // Must start with an offset or directly with (%ebp)
    if !rest.contains("(%ebp)") { return None; }
    let paren_start = rest.find("(%ebp)")?;
    let offset_str = &rest[..paren_start];
    let after = rest[paren_start + 6..].trim();
    if !after.starts_with(',') { return None; }
    let reg = after[1..].trim();
    if !reg.starts_with('%') { return None; }
    Some((offset_str, reg, size))
}

/// Parse `movl %src, %dst` (register-to-register move).
fn parse_reg_to_reg_move(s: &str) -> Option<(RegId, RegId)> {
    let rest = s.strip_prefix("movl ")?.trim();
    if !rest.starts_with('%') { return None; }
    let comma = rest.find(',')?;
    let src_name = rest[..comma].trim();
    let dst_name = rest[comma + 1..].trim();
    if !dst_name.starts_with('%') { return None; }
    // Must not be memory operands
    if src_name.contains('(') || dst_name.contains('(') { return None; }
    let src = register_family(src_name);
    let dst = register_family(dst_name);
    if src <= REG_GP_MAX && dst <= REG_GP_MAX {
        Some((src, dst))
    } else {
        None
    }
}

/// Parse integer offset from string.
fn parse_offset(s: &str) -> i32 {
    if s.is_empty() { return 0; }
    s.parse::<i32>().unwrap_or(EBP_OFFSET_NONE)
}

/// Check if a line has indirect memory access (pointer dereference through a register).
fn has_indirect_memory_access(s: &str) -> bool {
    // Pattern: offset(%eXX) where XX is not bp or sp
    // or (%eXX) where XX is not bp or sp
    // or (%eXX, %eYY, N)
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'(' && i + 4 < bytes.len() && bytes[i + 1] == b'%' {
            // Check if it's (%ebp) or (%esp) - those are stack accesses, not indirect
            if i + 5 < bytes.len() && (&bytes[i + 1..i + 5] == b"%ebp" || &bytes[i + 1..i + 5] == b"%esp") {
                continue;
            }
            return true;
        }
    }
    false
}

/// Parse the %ebp offset from a line, or return EBP_OFFSET_NONE.
fn parse_ebp_offset_in_line(s: &str) -> i32 {
    if let Some(pos) = s.find("(%ebp)") {
        let before = &s[..pos];
        // Find the start of the offset number
        let offset_start = before.rfind(|c: char| !c.is_ascii_digit() && c != '-').map(|p| p + 1).unwrap_or(0);
        let offset_str = &before[offset_start..];
        if offset_str.is_empty() {
            0
        } else {
            offset_str.parse::<i32>().unwrap_or(EBP_OFFSET_NONE)
        }
    } else {
        EBP_OFFSET_NONE
    }
}

/// Parse the destination register of a generic instruction.
/// For two-operand instructions (AT&T syntax), the destination is the last operand.
fn parse_dest_reg(s: &str) -> RegId {
    // Find the last %reg
    if let Some(comma) = s.rfind(',') {
        let after = s[comma + 1..].trim();
        if after.starts_with('%') && !after.contains('(') {
            return register_family(after);
        }
    }
    REG_NONE
}

/// Check if a line references a specific register family.
/// This includes both explicit register operands and implicit register uses
/// by instructions like cltd, idivl, rep movsb, etc.
fn line_references_reg(s: &str, reg: RegId) -> bool {
    // Check explicit register operands
    let names: &[&str] = match reg {
        REG_EAX => &["%eax", "%ax", "%al", "%ah"],
        REG_ECX => &["%ecx", "%cx", "%cl", "%ch"],
        REG_EDX => &["%edx", "%dx", "%dl", "%dh"],
        REG_EBX => &["%ebx", "%bx", "%bl", "%bh"],
        REG_ESP => &["%esp", "%sp"],
        REG_EBP => &["%ebp", "%bp"],
        REG_ESI => &["%esi", "%si"],
        REG_EDI => &["%edi", "%di"],
        _ => return false,
    };
    for name in names {
        if s.contains(name) { return true; }
    }
    // Check implicit register uses by specific instructions
    if implicit_reg_use(s, reg) { return true; }
    false
}

/// Check if an instruction implicitly uses a register (not mentioned in text).
fn implicit_reg_use(s: &str, reg: RegId) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() { return false; }
    match bytes[0] {
        b'c' => {
            // cmpxchg8b (without lock prefix): reads/writes eax, edx, ecx, ebx
            if s.starts_with("cmpxchg8b") {
                return reg == REG_EAX || reg == REG_EDX || reg == REG_ECX || reg == REG_EBX;
            }
            // cmpxchg{l,w,b} (without lock prefix): implicitly reads eax
            if s.starts_with("cmpxchg") {
                return reg == REG_EAX;
            }
            // cltd/cdq: reads eax, writes edx
            if s == "cltd" || s == "cdq" {
                return reg == REG_EAX || reg == REG_EDX;
            }
            // cbw/cwde: reads/writes eax
            if s == "cbw" || s == "cwde" || s == "cwtl" {
                return reg == REG_EAX;
            }
        }
        b'i' => {
            // idivl/idivw: implicitly reads edx:eax, writes eax and edx
            if s.starts_with("idivl") || s.starts_with("idivw") || s.starts_with("idivb") {
                return reg == REG_EAX || reg == REG_EDX;
            }
            // imull with 1 operand: reads eax, writes edx:eax
            // imull with 2 or 3 operands has explicit regs
            if s.starts_with("imull ") && !s.contains(',') {
                return reg == REG_EAX || reg == REG_EDX;
            }
        }
        b'd' => {
            // divl/divw: implicitly reads edx:eax, writes eax and edx
            if s.starts_with("divl") || s.starts_with("divw") || s.starts_with("divb") {
                return reg == REG_EAX || reg == REG_EDX;
            }
        }
        b'm' => {
            // mul: reads eax, writes edx:eax
            if s.starts_with("mull ") || s.starts_with("mulw ") || s.starts_with("mulb ") {
                return reg == REG_EAX || reg == REG_EDX;
            }
        }
        b'r' => {
            // rep movsb/movsl: uses esi, edi, ecx
            // rep stosb/stosl: uses edi, ecx, eax
            if s.starts_with("rep") {
                if s.contains("movs") {
                    return reg == REG_ESI || reg == REG_EDI || reg == REG_ECX;
                }
                if s.contains("stos") {
                    return reg == REG_EAX || reg == REG_EDI || reg == REG_ECX;
                }
                if s.contains("scas") || s.contains("cmps") {
                    return reg == REG_ESI || reg == REG_EDI || reg == REG_ECX || reg == REG_EAX;
                }
                // Unknown rep instruction - assume all regs used
                return true;
            }
        }
        b'l' => {
            // lock cmpxchg8b: implicitly reads/writes eax, edx, ecx, ebx
            // cmpxchg8b compares edx:eax with memory, stores ecx:ebx on match
            if s.starts_with("lock cmpxchg8b") {
                return reg == REG_EAX || reg == REG_EDX || reg == REG_ECX || reg == REG_EBX;
            }
            // lock cmpxchg{l,w,b}: implicitly reads eax (compared with memory)
            if s.starts_with("lock cmpxchg") {
                return reg == REG_EAX;
            }
            // loop/loope/loopne: reads ecx
            if s.starts_with("loop") {
                return reg == REG_ECX;
            }
        }
        _ => {}
    }
    false
}

// ── Line classifier ──────────────────────────────────────────────────────────

fn classify_line(raw: &str) -> LineInfo {
    let trim_start = raw.len() - raw.trim_start().len();
    let s = &raw[trim_start..];

    if s.is_empty() {
        return line_info(LineKind::Empty, trim_start as u16);
    }

    let bytes = s.as_bytes();
    let first = bytes[0];
    let last = bytes[bytes.len() - 1];
    let ts = trim_start as u16;

    // Label
    if last == b':' {
        return line_info(LineKind::Label, ts);
    }

    // Directive
    if first == b'.' {
        return line_info(LineKind::Directive, ts);
    }

    // Comment
    if first == b'#' {
        return line_info(LineKind::Directive, ts);
    }

    // mov instructions - check store/load/self-move/reg-reg
    if first == b'm' && bytes.len() >= 4 && bytes[1] == b'o' && bytes[2] == b'v' {
        if let Some((reg_str, offset_str, size)) = parse_store_to_ebp(s) {
            let reg = register_family(reg_str);
            if reg <= REG_GP_MAX {
                let offset = parse_offset(offset_str);
                return line_info(LineKind::StoreEbp { reg, offset, size }, ts);
            }
        }
        if let Some((offset_str, reg_str, size)) = parse_load_from_ebp(s) {
            let reg = register_family(reg_str);
            if reg <= REG_GP_MAX {
                let offset = parse_offset(offset_str);
                return line_info(LineKind::LoadEbp { reg, offset, size }, ts);
            }
        }
        if let Some((src, dst)) = parse_reg_to_reg_move(s) {
            if src == dst {
                return line_info(LineKind::SelfMove, ts);
            }
            return line_info(LineKind::Move { dst, src }, ts);
        }
    }

    // Control flow
    if first == b'j' {
        if bytes.len() >= 4 && bytes[1] == b'm' && bytes[2] == b'p' {
            if bytes.len() > 4 && bytes[4] == b'*' {
                return line_info(LineKind::JmpIndirect, ts);
            }
            if bytes[3] == b' ' {
                if s.contains("indirect_thunk") || s.contains("*%") {
                    return line_info(LineKind::JmpIndirect, ts);
                }
                return line_info(LineKind::Jmp, ts);
            }
        }
        if is_conditional_jump(s) {
            return line_info(LineKind::CondJmp, ts);
        }
    }

    if first == b'c' {
        if bytes.len() >= 4 && bytes[1] == b'a' && bytes[2] == b'l' && bytes[3] == b'l' {
            return line_info(LineKind::Call, ts);
        }
        if bytes.len() >= 4 && bytes[1] == b'm' && bytes[2] == b'p' {
            return line_info(LineKind::Cmp, ts);
        }
    }

    if first == b'r' && s == "ret" {
        return line_info(LineKind::Ret, ts);
    }

    // test instructions
    if first == b't' && bytes.len() >= 5 && bytes[1] == b'e' && bytes[2] == b's' && bytes[3] == b't' {
        return line_info(LineKind::Cmp, ts);
    }

    // push/pop
    if first == b'p' {
        if let Some(rest) = s.strip_prefix("pushl ") {
            let reg = register_family(rest.trim());
            return line_info(LineKind::Push { reg }, ts);
        }
        if let Some(rest) = s.strip_prefix("popl ") {
            let reg = register_family(rest.trim());
            return line_info(LineKind::Pop { reg }, ts);
        }
    }

    // setCC
    if first == b's' && bytes.len() >= 4 && bytes[1] == b'e' && bytes[2] == b't' && parse_setcc(s).is_some() {
        let setcc_reg = if let Some(space_pos) = s.rfind(' ') {
            register_family(s[space_pos + 1..].trim())
        } else {
            REG_EAX
        };
        return line_info(LineKind::SetCC { reg: setcc_reg }, ts);
    }

    // Other instruction
    let dest_reg = parse_dest_reg(s);
    let has_indirect = has_indirect_memory_access(s);
    let ebp_off = if has_indirect { EBP_OFFSET_NONE } else { parse_ebp_offset_in_line(s) };
    LineInfo {
        kind: LineKind::Other { dest_reg },
        trim_start: ts,
        has_indirect_mem: has_indirect,
        ebp_offset: ebp_off,
    }
}

// ── Conditional jump helpers ─────────────────────────────────────────────────

fn is_conditional_jump(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 3 || b[0] != b'j' { return false; }
    // jCC where CC is one of: e, ne, l, le, g, ge, b, be, a, ae, s, ns, o, no, p, np, z, nz
    matches!(&s[1..2], "e" | "a" | "b" | "g" | "l" | "s" | "o" | "p" | "z" | "n")
        && s.contains(' ')
}

/// Invert a condition code.
fn invert_cc(cc: &str) -> Option<&'static str> {
    match cc {
        "e" | "z" => Some("ne"),
        "ne" | "nz" => Some("e"),
        "l" => Some("ge"),
        "ge" => Some("l"),
        "le" => Some("g"),
        "g" => Some("le"),
        "b" => Some("ae"),
        "ae" => Some("b"),
        "be" => Some("a"),
        "a" => Some("be"),
        "s" => Some("ns"),
        "ns" => Some("s"),
        "o" => Some("no"),
        "no" => Some("o"),
        "p" => Some("np"),
        "np" => Some("p"),
        _ => None,
    }
}

/// Extract condition code and target from a conditional jump.
fn parse_condjmp(s: &str) -> Option<(&str, &str)> {
    if !s.starts_with('j') { return None; }
    let space = s.find(' ')?;
    let cc = &s[1..space];
    let target = s[space + 1..].trim();
    Some((cc, target))
}

/// Parse setCC instruction → condition code.
fn parse_setcc(s: &str) -> Option<&str> {
    if !s.starts_with("set") { return None; }
    let rest = &s[3..];
    let space = rest.find(' ')?;
    let cc = &rest[..space];
    // Validate it's a real condition code
    match cc {
        "e" | "ne" | "z" | "nz" | "l" | "le" | "g" | "ge" |
        "b" | "be" | "a" | "ae" | "s" | "ns" | "o" | "no" |
        "p" | "np" => Some(cc),
        _ => None,
    }
}

/// Extract the jump target from a jmp instruction.
fn parse_jmp_target(s: &str) -> Option<&str> {
    s.strip_prefix("jmp ")
}

// ── Line store ───────────────────────────────────────────────────────────────

/// Efficient line storage that avoids reallocating strings.
/// Lines are stored as byte offsets into the original assembly string.
/// Replaced lines are stored in a side buffer.
// Re-export the shared LineStore from peephole_common.
// See backend/peephole_common.rs for the implementation.
use crate::backend::peephole_common::LineStore;

// ── Trimmed line helper ──────────────────────────────────────────────────────

#[inline]
fn trimmed<'a>(store: &'a LineStore, info: &LineInfo, idx: usize) -> &'a str {
    &store.get(idx)[info.trim_start as usize..]
}

/// Check if the next instruction reads the carry flag (CF).
/// Instructions like `adcl`, `sbbl`, `rcl`, `rcr` depend on CF.
/// `incl`/`decl` do NOT set CF (unlike `addl`/`subl`), so converting
/// `addl $1` → `incl` or `subl $1` → `decl` is invalid when the next
/// instruction reads CF.
fn next_reads_carry_flag(store: &LineStore, infos: &[LineInfo], start: usize) -> bool {
    let len = infos.len();
    for j in (start + 1)..len {
        let s = store.get(j).trim();
        if s.is_empty() || s.starts_with('#') || s.starts_with("//") || s.ends_with(':') {
            continue;
        }
        // Check if the instruction reads CF
        return s.starts_with("adcl ")
            || s.starts_with("adcb ")
            || s.starts_with("adcw ")
            || s.starts_with("sbbl ")
            || s.starts_with("sbbb ")
            || s.starts_with("sbbw ")
            || s.starts_with("rcl ")
            || s.starts_with("rcr ")
            || s.starts_with("setc ")
            || s.starts_with("setb ")
            || s.starts_with("jc ")
            || s.starts_with("jb ")
            || s.starts_with("jnc ")
            || s.starts_with("jnb ")
            || s.starts_with("jae ")
            || s.starts_with("cmc");
    }
    false
}

// ── Pass 1: Local patterns ───────────────────────────────────────────────────

/// Combined local pass: scan once, apply multiple patterns.
fn combined_local_pass(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = infos.len();
    let mut changed = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Pattern 1: Self-move elimination
        if infos[i].kind == LineKind::SelfMove {
            infos[i].kind = LineKind::Nop;
            changed = true;
            i += 1;
            continue;
        }

        // Pattern 1b: Strength reduction for code size
        // - addl $1, %reg → incl %reg (saves 2 bytes, critical for 16-bit boot code)
        // - subl $1, %reg → decl %reg (saves 2 bytes)
        // - movl $0, %reg → xorl %reg, %reg (saves 3 bytes)
        // - addl $-1, %reg → decl %reg (saves 2 bytes)
        // - subl $-1, %reg → incl %reg (saves 2 bytes)
        if let LineKind::Other { dest_reg } = infos[i].kind {
            if dest_reg != REG_NONE && dest_reg <= REG_GP_MAX && dest_reg != REG_ESP && dest_reg != REG_EBP {
                let s = trimmed(store, &infos[i], i);
                let rn = reg32_name(dest_reg);
                // addl $1, %reg → incl %reg
                // SAFETY: incl does NOT set the carry flag (CF), so this
                // conversion is invalid if the next instruction reads CF
                // (e.g., adcl used in 64-bit add-with-carry chains).
                if s.starts_with("addl $1, ") && s.ends_with(rn)
                    && !next_reads_carry_flag(store, infos, i)
                {
                    store.replace(i, format!("    incl {}", rn));
                    infos[i] = LineInfo {
                        kind: LineKind::Other { dest_reg },
                        trim_start: 4,
                        has_indirect_mem: false,
                        ebp_offset: EBP_OFFSET_NONE,
                    };
                    changed = true;
                    i += 1;
                    continue;
                }
                // subl $1, %reg → decl %reg
                // SAFETY: decl does NOT set CF, skip if next reads CF.
                if s.starts_with("subl $1, ") && s.ends_with(rn)
                    && !next_reads_carry_flag(store, infos, i)
                {
                    store.replace(i, format!("    decl {}", rn));
                    infos[i] = LineInfo {
                        kind: LineKind::Other { dest_reg },
                        trim_start: 4,
                        has_indirect_mem: false,
                        ebp_offset: EBP_OFFSET_NONE,
                    };
                    changed = true;
                    i += 1;
                    continue;
                }
                // addl $-1, %reg → decl %reg
                // SAFETY: decl does NOT set CF, skip if next reads CF.
                if s.starts_with("addl $-1, ") && s.ends_with(rn)
                    && !next_reads_carry_flag(store, infos, i)
                {
                    store.replace(i, format!("    decl {}", rn));
                    infos[i] = LineInfo {
                        kind: LineKind::Other { dest_reg },
                        trim_start: 4,
                        has_indirect_mem: false,
                        ebp_offset: EBP_OFFSET_NONE,
                    };
                    changed = true;
                    i += 1;
                    continue;
                }
                // subl $-1, %reg → incl %reg
                // SAFETY: incl does NOT set CF, skip if next reads CF.
                if s.starts_with("subl $-1, ") && s.ends_with(rn)
                    && !next_reads_carry_flag(store, infos, i)
                {
                    store.replace(i, format!("    incl {}", rn));
                    infos[i] = LineInfo {
                        kind: LineKind::Other { dest_reg },
                        trim_start: 4,
                        has_indirect_mem: false,
                        ebp_offset: EBP_OFFSET_NONE,
                    };
                    changed = true;
                    i += 1;
                    continue;
                }
            }
        }
        // movl $0, %reg → xorl %reg, %reg (saves 3 bytes, clears flags)
        if let LineKind::Other { dest_reg } = infos[i].kind {
            if dest_reg != REG_NONE && dest_reg <= REG_GP_MAX && dest_reg != REG_ESP && dest_reg != REG_EBP {
                let s = trimmed(store, &infos[i], i);
                let rn = reg32_name(dest_reg);
                if s == format!("movl $0, {}", rn) {
                    store.replace(i, format!("    xorl {}, {}", rn, rn));
                    infos[i] = LineInfo {
                        kind: LineKind::Other { dest_reg },
                        trim_start: 4,
                        has_indirect_mem: false,
                        ebp_offset: EBP_OFFSET_NONE,
                    };
                    changed = true;
                    i += 1;
                    continue;
                }
            }
        }

        // Find next non-nop line
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len { i += 1; continue; }

        // Pattern 2: Adjacent store/load with same offset
        if let LineKind::StoreEbp { reg: store_reg, offset: store_off, size: store_size } = infos[i].kind {
            if let LineKind::LoadEbp { reg: load_reg, offset: load_off, size: load_size } = infos[j].kind {
                if store_off == load_off && store_size == load_size {
                    if store_reg == load_reg {
                        // movl %eax, -8(%ebp); movl -8(%ebp), %eax → keep store only
                        infos[j].kind = LineKind::Nop;
                        changed = true;
                        i += 1;
                        continue;
                    } else {
                        // movl %eax, -8(%ebp); movl -8(%ebp), %ecx → movl %eax, -8(%ebp); movl %eax, %ecx
                        let new_line = format!("    {} {}, {}", store_size.mnemonic(), reg32_name(store_reg), reg32_name(load_reg));
                        store.replace(j, new_line);
                        infos[j] = LineInfo {
                            kind: LineKind::Move { dst: load_reg, src: store_reg },
                            trim_start: 4,
                            has_indirect_mem: false,
                            ebp_offset: EBP_OFFSET_NONE,
                        };
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Pattern 3: Redundant jump to next label
        if infos[i].kind == LineKind::Jmp && infos[j].kind == LineKind::Label {
            let jmp_s = trimmed(store, &infos[i], i);
            let label_s = trimmed(store, &infos[j], j);
            if let Some(target) = parse_jmp_target(jmp_s) {
                if let Some(label_name) = label_s.strip_suffix(':') {
                    if target.trim() == label_name {
                        infos[i].kind = LineKind::Nop;
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Pattern 4: Branch inversion: jCC .L1; jmp .L2; .L1: → j!CC .L2; .L1:
        if infos[i].kind == LineKind::CondJmp {
            let mut k = j + 1;
            while k < len && infos[k].is_nop() { k += 1; }
            if k < len && infos[j].kind == LineKind::Jmp && infos[k].kind == LineKind::Label {
                let cond_s = trimmed(store, &infos[i], i);
                let jmp_s = trimmed(store, &infos[j], j);
                let label_s = trimmed(store, &infos[k], k);
                if let (Some((cc, cond_target)), Some(jmp_target)) =
                    (parse_condjmp(cond_s), parse_jmp_target(jmp_s))
                {
                    if let Some(label_name) = label_s.strip_suffix(':') {
                        if cond_target == label_name {
                            if let Some(inv_cc) = invert_cc(cc) {
                                let new_line = format!("    j{} {}", inv_cc, jmp_target.trim());
                                store.replace(i, new_line);
                                infos[i].kind = LineKind::CondJmp;
                                infos[j].kind = LineKind::Nop;
                                changed = true;
                                i += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Pattern 5b: Redundant movsbl %al, %eax after movsbl (...), %eax
        // The first sign-extension already produces a properly sign-extended 32-bit result,
        // so the second `movsbl %al, %eax` is a no-op.
        if let LineKind::Other { dest_reg: REG_EAX } = infos[i].kind {
            let si = trimmed(store, &infos[i], i);
            if si.starts_with("movsbl ") && si.ends_with(", %eax") {
                if let LineKind::Other { dest_reg: REG_EAX } = infos[j].kind {
                    let sj = trimmed(store, &infos[j], j);
                    if sj == "movsbl %al, %eax" {
                        infos[j].kind = LineKind::Nop;
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Pattern 5: Reverse move elimination: movl %A, %B; movl %B, %A → keep first only
        if let LineKind::Move { dst: dst1, src: src1 } = infos[i].kind {
            if let LineKind::Move { dst: dst2, src: src2 } = infos[j].kind {
                if dst1 == src2 && src1 == dst2 {
                    infos[j].kind = LineKind::Nop;
                    changed = true;
                    i += 1;
                    continue;
                }
            }
        }

        i += 1;
    }

    changed
}

// ── Pass 2: Global store forwarding ──────────────────────────────────────────

/// Track which register value is stored at each stack slot.
/// When we see `movl %eax, -8(%ebp)`, record that slot -8 contains eax.
/// When we see `movl -8(%ebp), %ecx`, forward to `movl %eax, %ecx` or eliminate if same reg.
// TODO: Disabled - causes 21 regressions in FP computation tests (matrix/FP operations
// produce wrong numerical results). Needs investigation into FP load/store forwarding patterns.
#[allow(dead_code)]
fn global_store_forwarding(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = infos.len();
    let mut changed = false;

    // Mapping: offset → (reg, line_idx)
    // Small flat array for common offsets (-256..0)
    const SLOT_COUNT: usize = 256;
    let mut slots: [(RegId, MoveSize); SLOT_COUNT] = [(REG_NONE, MoveSize::L); SLOT_COUNT];

    // Collect jump targets so we can invalidate at them
    let mut jump_targets = std::collections::HashSet::new();
    for i in 0..len {
        if infos[i].is_nop() { continue; }
        let s = trimmed(store, &infos[i], i);
        match infos[i].kind {
            LineKind::Jmp | LineKind::JmpIndirect => {
                if let Some(target) = parse_jmp_target(s) {
                    jump_targets.insert(target.trim().to_string());
                }
            }
            LineKind::CondJmp => {
                if let Some((_, target)) = parse_condjmp(s) {
                    jump_targets.insert(target.to_string());
                }
            }
            _ => {}
        }
    }

    for i in 0..len {
        if infos[i].is_nop() { continue; }

        match infos[i].kind {
            LineKind::Label => {
                // Check if this label is a jump target (invalidate all)
                let s = trimmed(store, &infos[i], i);
                if let Some(name) = s.strip_suffix(':') {
                    if jump_targets.contains(name) {
                        // This label is a jump target - invalidate all mappings
                        slots = [(REG_NONE, MoveSize::L); SLOT_COUNT];
                    }
                    // If it's just a fallthrough label, keep mappings
                }
            }
            LineKind::StoreEbp { reg, offset, size } => {
                // Record that this slot now contains this register's value
                if offset < 0 && (-offset as usize) <= SLOT_COUNT {
                    slots[(-offset - 1) as usize] = (reg, size);
                }
            }
            LineKind::LoadEbp { reg: load_reg, offset, size: load_size } => {
                // Check if we know what register value is in this slot
                let mut forwarded = false;
                if offset < 0 && (-offset as usize) <= SLOT_COUNT {
                    let (stored_reg, stored_size) = slots[(-offset - 1) as usize];
                    if stored_reg != REG_NONE && stored_size == load_size {
                        if stored_reg == load_reg {
                            // Same register - just eliminate the load
                            infos[i].kind = LineKind::Nop;
                            changed = true;
                            forwarded = true;
                        } else {
                            // Different register - forward as reg-reg move
                            let new_line = format!("    {} {}, {}", load_size.mnemonic(), reg32_name(stored_reg), reg32_name(load_reg));
                            store.replace(i, new_line);
                            infos[i] = LineInfo {
                                kind: LineKind::Move { dst: load_reg, src: stored_reg },
                                trim_start: 4,
                                has_indirect_mem: false,
                                ebp_offset: EBP_OFFSET_NONE,
                            };
                            changed = true;
                            forwarded = true;
                        }
                    }
                }
                // The load writes to load_reg, so invalidate any slot
                // that maps to load_reg (its value has changed).
                // This must happen even if we forwarded, because the
                // destination register now has a new value.
                for slot in slots.iter_mut() {
                    if slot.0 == load_reg {
                        *slot = (REG_NONE, MoveSize::L);
                    }
                }
                if forwarded { continue; }
            }
            LineKind::Call => {
                // Calls clobber caller-saved registers (eax, ecx, edx)
                // Invalidate all mappings involving these registers
                for slot in slots.iter_mut() {
                    if is_caller_saved(slot.0) {
                        *slot = (REG_NONE, MoveSize::L);
                    }
                }
            }
            LineKind::Jmp | LineKind::JmpIndirect | LineKind::Ret => {
                // Control flow change - invalidate all
                slots = [(REG_NONE, MoveSize::L); SLOT_COUNT];
            }
            LineKind::Move { dst, .. } => {
                // Invalidate any slot that was mapped to the overwritten register
                for slot in slots.iter_mut() {
                    if slot.0 == dst {
                        *slot = (REG_NONE, MoveSize::L);
                    }
                }
            }
            LineKind::SetCC { reg } => {
                // setCC modifies a byte register, invalidate its family
                for slot in slots.iter_mut() {
                    if slot.0 == reg {
                        *slot = (REG_NONE, MoveSize::L);
                    }
                }
            }
            LineKind::Other { dest_reg } => {
                // Invalidate any slot mapped to the destination register
                if dest_reg != REG_NONE {
                    for slot in slots.iter_mut() {
                        if slot.0 == dest_reg {
                            *slot = (REG_NONE, MoveSize::L);
                        }
                    }
                }
                // If line has indirect memory access or might clobber stack,
                // invalidate all (conservative)
                let s = trimmed(store, &infos[i], i);
                if infos[i].has_indirect_mem || s.contains("(%ebp)") {
                    // Only invalidate the specific slot if we can parse it
                    let off = infos[i].ebp_offset;
                    if off != EBP_OFFSET_NONE && off < 0 && (-off as usize) <= SLOT_COUNT {
                        slots[(-off - 1) as usize] = (REG_NONE, MoveSize::L);
                    } else if infos[i].has_indirect_mem {
                        // Indirect memory - could write anywhere, invalidate all
                        slots = [(REG_NONE, MoveSize::L); SLOT_COUNT];
                    }
                }
                // Check for inline asm or instructions that clobber multiple regs
                if s.contains(';') || s.starts_with("rdmsr") || s.starts_with("cpuid")
                    || s.starts_with("syscall") || s.starts_with("int ") || s.starts_with("int$")
                    || s.starts_with("rep") || s.starts_with("cld") {
                    slots = [(REG_NONE, MoveSize::L); SLOT_COUNT];
                }
            }
            LineKind::Push { .. } | LineKind::Pop { .. } => {
                // Push/pop modify esp but don't affect ebp-relative slots
                if let LineKind::Pop { reg } = infos[i].kind {
                    // Pop writes to a register, invalidate mappings
                    for slot in slots.iter_mut() {
                        if slot.0 == reg {
                            *slot = (REG_NONE, MoveSize::L);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    changed
}

// ── Pass: Dead store elimination ─────────────────────────────────────────────

/// Remove stores to stack slots that are immediately overwritten.
fn eliminate_dead_stores(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let len = infos.len();
    let mut changed = false;
    const WINDOW: usize = 16;

    for i in 0..len {
        if infos[i].is_nop() { continue; }
        if let LineKind::StoreEbp { offset: store_off, size: store_size, reg: store_reg } = infos[i].kind {
            // Look ahead for another store to the same slot (meaning this one is dead)
            // or a load from the same slot (meaning this one is alive)
            let mut j = i + 1;
            let mut count = 0;
            while j < len && count < WINDOW {
                if infos[j].is_nop() { j += 1; continue; }

                let store_bytes = store_size.byte_size();
                match infos[j].kind {
                    LineKind::StoreEbp { offset, size, .. }
                        if offset == store_off && size == store_size =>
                    {
                        // Another store to the exact same slot - this store is dead
                        infos[i].kind = LineKind::Nop;
                        changed = true;
                        break;
                    }
                    LineKind::StoreEbp { offset, size, .. }
                        if ranges_overlap(store_off, store_bytes, offset, size.byte_size()) =>
                    {
                        // Overlapping store but not identical - conservatively keep alive
                        break;
                    }
                    LineKind::LoadEbp { offset, size, .. }
                        if ranges_overlap(store_off, store_bytes, offset, size.byte_size()) =>
                    {
                        // Load overlaps this store's byte range - this store is alive
                        break;
                    }
                    _ => {}
                }

                // Stop at barriers
                if infos[j].is_barrier() { break; }
                // Stop if the stored register is modified (value may have changed)
                match infos[j].kind {
                    LineKind::Other { dest_reg } if dest_reg == store_reg => break,
                    LineKind::Move { dst, .. } if dst == store_reg => break,
                    LineKind::SetCC { reg } if reg == store_reg => break,
                    _ => {}
                }
                // Stop at indirect memory access (could read the slot)
                let s = trimmed(store, &infos[j], j);
                if infos[j].has_indirect_mem { break; }
                // If line references ebp with same offset, it's alive
                if infos[j].ebp_offset == store_off { break; }
                // leaq N(%ebp) takes address of slot
                if s.contains("(%ebp)") && !matches!(infos[j].kind, LineKind::StoreEbp { .. } | LineKind::LoadEbp { .. }) {
                    break;
                }

                j += 1;
                count += 1;
            }
        }
    }

    changed
}

// ── Pass: Dead register move elimination ─────────────────────────────────────

/// Remove register moves where the destination is overwritten before being read.
fn eliminate_dead_reg_moves(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let len = infos.len();
    let mut changed = false;
    const WINDOW: usize = 16;

    for i in 0..len {
        if infos[i].is_nop() { continue; }
        let dst_reg = match infos[i].kind {
            LineKind::Move { dst, .. } => dst,
            _ => continue,
        };
        // Don't eliminate moves to esp/ebp
        if dst_reg == REG_ESP || dst_reg == REG_EBP { continue; }

        // Look ahead: if dst is overwritten before being read, this move is dead
        let mut j = i + 1;
        let mut count = 0;
        while j < len && count < WINDOW {
            if infos[j].is_nop() { j += 1; continue; }
            if infos[j].is_barrier() { break; }

            // Check if dst is read by this instruction
            let s = trimmed(store, &infos[j], j);
            match infos[j].kind {
                LineKind::StoreEbp { reg, .. } if reg == dst_reg => {
                    // dst is read (stored to stack) - move is alive
                    break;
                }
                LineKind::Move { src, dst } => {
                    if src == dst_reg {
                        // dst is read - move is alive
                        break;
                    }
                    if dst == dst_reg {
                        // dst is overwritten - this move is dead
                        infos[i].kind = LineKind::Nop;
                        changed = true;
                        break;
                    }
                }
                LineKind::Other { dest_reg } => {
                    if dest_reg == dst_reg && !line_references_reg(s, dst_reg) {
                        // Destination only writes to dst, doesn't read it: dead
                        // But we need to make sure it doesn't ALSO read it
                        // Actually, just check if the line references the reg at all
                        // For "movl $5, %eax", dest is eax and it doesn't read eax
                        // For "addl $5, %eax", dest is eax and it reads eax
                        // parse_dest_reg returns the last operand. If the instruction writes
                        // to dst_reg but the source doesn't reference it, then this move is dead.
                        infos[i].kind = LineKind::Nop;
                        changed = true;
                        break;
                    }
                    if line_references_reg(s, dst_reg) {
                        break; // dst is read
                    }
                }
                _ => {
                    if line_references_reg(s, dst_reg) {
                        break; // dst is read
                    }
                }
            }

            j += 1;
            count += 1;
        }
    }

    changed
}

// ── Pass: Compare and branch fusion ──────────────────────────────────────────

/// Maximum number of store/load offsets tracked during compare-and-branch fusion.
const MAX_TRACKED_STORE_LOAD_OFFSETS: usize = 4;

/// Size of the instruction lookahead window for compare-and-branch fusion.
const CMP_FUSION_LOOKAHEAD: usize = 8;

/// Collect up to N non-NOP line indices following `start_idx` (exclusive).
/// Returns the number of indices collected.
fn collect_non_nop_indices<const N: usize>(
    infos: &[LineInfo], start_idx: usize, len: usize, out: &mut [usize; N],
) -> usize {
    let mut count = 0;
    let mut j = start_idx + 1;
    while j < len && count < N {
        if !infos[j].is_nop() {
            out[count] = j;
            count += 1;
        }
        j += 1;
    }
    count
}

/// Fuse `cmpl/testl + setCC %al + movzbl %al, %eax + [store/load] + testl %eax, %eax + jne/je`
/// into a single `jCC`/`j!CC` directly.
///
/// This enhanced version can skip over store/load pairs between the movzbl and
/// testl, allowing fusion even when the boolean is temporarily spilled to the
/// stack. It tracks stored offsets and verifies each has a matching load nearby,
/// ensuring the stored boolean is only consumed locally.
fn fuse_compare_and_branch(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = infos.len();
    let mut changed = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() || infos[i].kind != LineKind::Cmp {
            i += 1;
            continue;
        }

        // Collect next non-NOP lines: cmp itself + (CMP_FUSION_LOOKAHEAD-1) following
        let mut seq_indices = [0usize; CMP_FUSION_LOOKAHEAD];
        seq_indices[0] = i;
        let mut rest = [0usize; CMP_FUSION_LOOKAHEAD - 1];
        let rest_count = collect_non_nop_indices::<{ CMP_FUSION_LOOKAHEAD - 1 }>(infos, i, len, &mut rest);
        seq_indices[1..(rest_count + 1)].copy_from_slice(&rest[..rest_count]);
        let seq_count = 1 + rest_count;

        if seq_count < 4 {
            i += 1;
            continue;
        }

        // Second must be setCC %al
        let setcc_cc = if let LineKind::SetCC { reg: REG_EAX } = infos[seq_indices[1]].kind {
            let s = trimmed(store, &infos[seq_indices[1]], seq_indices[1]);
            parse_setcc(s)
        } else {
            None
        };
        if setcc_cc.is_none() {
            i += 1;
            continue;
        }
        let setcc_cc = setcc_cc.unwrap();

        // Scan for testl %eax, %eax pattern.
        // Track StoreEbp offsets so we can bail out if any store's slot is
        // potentially read by another basic block (no matching load nearby).
        let mut test_idx = None;
        let mut store_offsets: [i32; MAX_TRACKED_STORE_LOAD_OFFSETS] = [0; MAX_TRACKED_STORE_LOAD_OFFSETS];
        let mut store_count = 0usize;
        let mut scan = 2;
        while scan < seq_count {
            let si = seq_indices[scan];
            let line = trimmed(store, &infos[si], si);

            // Skip zero-extend of setcc result
            if line == "movzbl %al, %eax" {
                scan += 1;
                continue;
            }
            // Skip store/load to ebp (pre-parsed fast check).
            if let LineKind::StoreEbp { offset, .. } = infos[si].kind {
                if store_count < MAX_TRACKED_STORE_LOAD_OFFSETS {
                    store_offsets[store_count] = offset;
                    store_count += 1;
                } else {
                    store_count = usize::MAX;
                    break;
                }
                scan += 1;
                continue;
            }
            if matches!(infos[si].kind, LineKind::LoadEbp { .. }) {
                scan += 1;
                continue;
            }
            // Skip cwtl (sign-extend ax->eax, i686 equivalent of cltq)
            if line == "cwtl" || line.starts_with("movswl ") || line.starts_with("movsbl ") {
                scan += 1;
                continue;
            }
            // Check for test
            if line == "testl %eax, %eax" {
                test_idx = Some(scan);
                break;
            }
            break;
        }

        let test_scan = match test_idx {
            Some(t) => t,
            None => { i += 1; continue; }
        };

        // If there are stores in the sequence, verify each has a matching load nearby.
        if store_count == usize::MAX {
            i += 1;
            continue;
        }
        if store_count > 0 {
            let range_start = seq_indices[1];
            let range_end = seq_indices[test_scan];
            let mut load_offsets: [i32; MAX_TRACKED_STORE_LOAD_OFFSETS] = [0; MAX_TRACKED_STORE_LOAD_OFFSETS];
            let mut load_count = 0usize;
            for ri in range_start..=range_end {
                let off = match infos[ri].kind {
                    LineKind::LoadEbp { offset, .. } => Some(offset),
                    // Check NOP'd lines too - earlier passes (store/load forwarding)
                    // may have NOP'd a load that originally matched a store.
                    LineKind::Nop => {
                        let orig = classify_line(store.get(ri));
                        match orig.kind {
                            LineKind::LoadEbp { offset, .. } => Some(offset),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(o) = off {
                    if load_count < MAX_TRACKED_STORE_LOAD_OFFSETS {
                        load_offsets[load_count] = o;
                        load_count += 1;
                    }
                }
            }
            let has_unmatched_store = (0..store_count).any(|si| {
                !(0..load_count).any(|li| load_offsets[li] == store_offsets[si])
            });
            if has_unmatched_store {
                i += 1;
                continue;
            }
        }

        if test_scan + 1 >= seq_count {
            i += 1;
            continue;
        }

        // Find jne/je after test
        let jmp_line = trimmed(store, &infos[seq_indices[test_scan + 1]], seq_indices[test_scan + 1]);
        let (is_jne, branch_target) = if let Some(target) = jmp_line.strip_prefix("jne ") {
            (true, target.trim())
        } else if let Some(target) = jmp_line.strip_prefix("je ") {
            (false, target.trim())
        } else {
            i += 1;
            continue;
        };

        let fused_cc = if is_jne {
            setcc_cc
        } else {
            match invert_cc(setcc_cc) {
                Some(inv) => inv,
                None => { i += 1; continue; }
            }
        };

        let fused_jcc = format!("    j{} {}", fused_cc, branch_target);

        // NOP out everything from setCC through testl
        for s in 1..=test_scan {
            infos[seq_indices[s]].kind = LineKind::Nop;
        }
        // Replace the jne/je with the fused conditional jump
        let idx = seq_indices[test_scan + 1];
        store.replace(idx, fused_jcc);
        infos[idx] = LineInfo {
            kind: LineKind::CondJmp,
            trim_start: 4,
            has_indirect_mem: false,
            ebp_offset: EBP_OFFSET_NONE,
        };

        changed = true;
        i = idx + 1;
    }

    changed
}

// ── Pass: Memory operand folding ─────────────────────────────────────────────

/// Fold `movl -N(%ebp), %ecx; addl %ecx, %eax` into `addl -N(%ebp), %eax`.
fn fold_memory_operands(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = infos.len();
    let mut changed = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Look for load from stack slot
        if let LineKind::LoadEbp { reg: load_reg, offset, size } = infos[i].kind {
            // Only fold scratch registers (eax, ecx, edx)
            if !is_caller_saved(load_reg) && load_reg != REG_EAX {
                i += 1; continue;
            }

            // Find next non-nop instruction
            let j = next_non_nop(infos, i + 1);
            if j >= len { i += 1; continue; }

            // Check if next instruction uses this register as a source operand
            // Pattern: load into %ecx, then `addl %ecx, %eax` etc.
            let s = trimmed(store, &infos[j], j);
            if let Some(folded) = try_fold_memory_operand(s, load_reg, offset, size) {
                store.replace(j, format!("    {}", folded));
                let dest_reg = parse_dest_reg(&folded);
                infos[j] = LineInfo {
                    kind: LineKind::Other { dest_reg },
                    trim_start: 4,
                    has_indirect_mem: false,
                    ebp_offset: offset,
                };
                infos[i].kind = LineKind::Nop; // Remove the load
                changed = true;
            }
        }
        i += 1;
    }

    changed
}

/// Try to fold a stack slot into an ALU instruction.
/// Returns the folded instruction string if successful.
fn try_fold_memory_operand(s: &str, load_reg: RegId, offset: i32, _size: MoveSize) -> Option<String> {
    let reg_name = reg32_name(load_reg);

    // Try patterns: `OPCODE %load_reg, %other_reg`
    for op in &["addl", "subl", "andl", "orl", "xorl", "cmpl", "testl", "imull"] {
        if let Some(rest) = s.strip_prefix(op) {
            let rest = rest.trim();
            // Pattern: `%load_reg, %dst` → `OPCODE offset(%ebp), %dst`
            if let Some(after) = rest.strip_prefix(reg_name) {
                let after = after.trim();
                if let Some(after_comma) = after.strip_prefix(',') {
                    let dst = after_comma.trim();
                    if dst.starts_with('%') && !dst.contains('(') {
                        // Don't fold if dst is the same as load_reg (would be read after free)
                        if register_family(dst) != load_reg {
                            return Some(format!("{} {}(%ebp), {}", op, offset, dst));
                        }
                    }
                }
            }
        }
    }

    None
}

// ── Pass: Never-read store elimination ───────────────────────────────────────

/// Global pass: find stack slots that are never loaded and remove all stores to them.
fn eliminate_never_read_stores(store: &LineStore, infos: &mut [LineInfo]) {
    let len = infos.len();

    // Collect all loaded byte ranges (offset, size)
    let mut read_ranges: Vec<(i32, i32)> = Vec::new();
    let mut addr_taken = false;

    for i in 0..len {
        if infos[i].is_nop() { continue; }
        match infos[i].kind {
            LineKind::LoadEbp { offset, size, .. } => {
                read_ranges.push((offset, size.byte_size()));
            }
            _ => {
                let s = trimmed(store, &infos[i], i);
                // Check for address-of-slot patterns (leal N(%ebp), %reg or leal N(%esp), %reg)
                if s.starts_with("leal ") && (s.contains("(%ebp)") || s.contains("(%esp)")) {
                    addr_taken = true;
                }
                // Indirect memory access means we can't know what's read
                if infos[i].has_indirect_mem {
                    addr_taken = true;
                }
                // Track %ebp-relative reads from non-Load/Store instructions
                // (e.g. folded memory operands like "cmpl -44(%ebp), %eax")
                let ebp_off = infos[i].ebp_offset;
                if ebp_off != EBP_OFFSET_NONE {
                    // Conservatively treat as a 4-byte read (max store size on i686)
                    read_ranges.push((ebp_off, 4));
                } else if !matches!(infos[i].kind, LineKind::StoreEbp { .. }) && s.contains("(%ebp)") {
                    // Unknown %ebp reference - bail out
                    addr_taken = true;
                }
            }
        }
    }

    if addr_taken { return; }

    // Remove stores to slots whose byte range is never overlapped by any load
    for i in 0..len {
        if infos[i].is_nop() { continue; }
        if let LineKind::StoreEbp { offset, size, .. } = infos[i].kind {
            let store_bytes = size.byte_size();
            let is_read = read_ranges.iter().any(|&(r_off, r_sz)| {
                ranges_overlap(offset, store_bytes, r_off, r_sz)
            });
            if !is_read {
                infos[i].kind = LineKind::Nop;
            }
        }
    }
}

// ── Pass: Unused callee-saved register elimination ───────────────────────────

/// Remove pushl/popl of callee-saved registers that are never referenced in the function body.
// TODO: Disabled - buggy leal -N(%ebp),%esp adjustment causes stack misalignment and 97+
// segfault regressions. Needs proper understanding of frame layout before re-enabling.
#[allow(dead_code)]
fn eliminate_unused_callee_saves(store: &LineStore, infos: &mut [LineInfo]) {
    let len = infos.len();

    // Find function boundaries
    let mut func_start = 0;
    for i in 0..len {
        if infos[i].is_nop() { continue; }
        // Look for the prologue pattern: pushl %ebp; movl %esp, %ebp
        if let LineKind::Push { reg: REG_EBP } = infos[i].kind {
            func_start = i;
            break;
        }
        if infos[i].kind == LineKind::Label {
            let s = trimmed(store, &infos[i], i);
            if s.ends_with(':') && !s.starts_with('.') {
                func_start = i;
            }
        }
    }

    // Identify callee-saved registers that are pushed in the prologue
    // and check if they're used in the function body
    for reg in [REG_EBX, REG_ESI, REG_EDI] {
        // Find the push of this register
        let mut push_idx = None;
        let mut pop_idx = None;
        let mut used = false;

        for i in func_start..len {
            if infos[i].is_nop() { continue; }
            match infos[i].kind {
                LineKind::Push { reg: r } if r == reg && push_idx.is_none() => {
                    push_idx = Some(i);
                }
                LineKind::Pop { reg: r } if r == reg => {
                    pop_idx = Some(i);
                }
                _ => {
                    if push_idx.is_some() && pop_idx.is_none() {
                        // Check if the register is referenced in the body
                        let s = trimmed(store, &infos[i], i);
                        if line_references_reg(s, reg) {
                            used = true;
                        }
                    }
                }
            }
        }

        if !used {
            if let Some(pi) = push_idx {
                infos[pi].kind = LineKind::Nop;
                if let Some(qi) = pop_idx {
                    infos[qi].kind = LineKind::Nop;
                }
                // For noreturn functions (no pop), still eliminate the push
            }
        }
    }
}

// ── Pass: Push/pop elimination ───────────────────────────────────────────────

/// Eliminate push/pop pairs where the register is not modified between them.
// TODO: Disabled - removes function-level callee-save push/pops which breaks the
// leal -12(%ebp),%esp epilogue pattern. Needs awareness of function boundaries.
#[allow(dead_code)]
fn eliminate_push_pop_pairs(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let len = infos.len();
    let mut changed = false;

    for i in 0..len {
        if infos[i].is_nop() { continue; }
        let push_reg = match infos[i].kind {
            LineKind::Push { reg } if reg <= REG_GP_MAX => reg,
            _ => continue,
        };

        // Scan forward for matching pop, checking reg is unmodified
        let mut j = i + 1;
        let mut depth = 0; // Track nested push/pops
        let mut safe = true;
        while j < len {
            if infos[j].is_nop() { j += 1; continue; }

            match infos[j].kind {
                LineKind::Push { .. } => { depth += 1; }
                LineKind::Pop { _reg } if depth > 0 => { depth -= 1; }
                LineKind::Pop { reg } if reg == push_reg && depth == 0 => {
                    // Found matching pop
                    if safe {
                        infos[i].kind = LineKind::Nop;
                        infos[j].kind = LineKind::Nop;
                        changed = true;
                    }
                    break;
                }
                LineKind::Pop { .. } if depth == 0 => { break; } // Different register popped
                _ => {}
            }

            // Check if the register is modified
            match infos[j].kind {
                LineKind::Move { dst, .. } if dst == push_reg => { safe = false; }
                LineKind::LoadEbp { reg, .. } if reg == push_reg => { safe = false; }
                LineKind::Other { dest_reg } if dest_reg == push_reg => { safe = false; }
                LineKind::Other { .. } => {
                    // For unknown instructions, check if the raw text references the register
                    // or if it's an instruction that implicitly clobbers registers (rep movsb, etc.)
                    let raw = store.get(j).trim();
                    if raw.starts_with("rep") || raw.starts_with("cld") {
                        // rep movsb/movsl/stosb etc. clobber esi, edi, ecx
                        if push_reg == REG_ESI || push_reg == REG_EDI || push_reg == REG_ECX {
                            safe = false;
                        }
                    }
                    if line_references_reg(raw, push_reg) {
                        safe = false;
                    }
                }
                LineKind::Call => { if is_caller_saved(push_reg) { safe = false; } }
                LineKind::SetCC { reg } if reg == push_reg => { safe = false; }
                _ => {}
            }

            // Stop at barriers that change control flow
            if matches!(infos[j].kind, LineKind::Jmp | LineKind::JmpIndirect | LineKind::Ret | LineKind::Label) {
                break;
            }

            j += 1;
        }
    }

    changed
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// Find the next non-nop line after index `start`.
fn next_non_nop(infos: &[LineInfo], start: usize) -> usize {
    let mut i = start;
    while i < infos.len() && (infos[i].is_nop() || infos[i].kind == LineKind::Empty) {
        i += 1;
    }
    i
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// Run peephole optimization on i686 assembly text.
/// Returns the optimized assembly string.
pub fn peephole_optimize(asm: String) -> String {
    let mut store = LineStore::new(asm);
    let line_count = store.len();
    let mut infos: Vec<LineInfo> = (0..line_count).map(|i| classify_line(store.get(i))).collect();

    // Phase 1: Iterative local passes
    let mut changed = true;
    let mut pass_count = 0;
    while changed && pass_count < MAX_LOCAL_PASS_ITERATIONS {
        changed = false;
        changed |= combined_local_pass(&mut store, &mut infos);
        pass_count += 1;
    }

    // Phase 2: Global passes (run once)
    let global_changed = eliminate_dead_reg_moves(&store, &mut infos);
    let global_changed = global_changed | eliminate_dead_stores(&store, &mut infos);
    let global_changed = global_changed | fuse_compare_and_branch(&mut store, &mut infos);
    let global_changed = global_changed | fold_memory_operands(&mut store, &mut infos);

    // Phase 3: Local cleanup after global passes
    if global_changed {
        let mut changed2 = true;
        let mut pass_count2 = 0;
        while changed2 && pass_count2 < MAX_POST_GLOBAL_ITERATIONS {
            changed2 = false;
            changed2 |= combined_local_pass(&mut store, &mut infos);
            changed2 |= eliminate_dead_reg_moves(&store, &mut infos);
            changed2 |= eliminate_dead_stores(&store, &mut infos);
            changed2 |= fold_memory_operands(&mut store, &mut infos);
            pass_count2 += 1;
        }
    }

    // Phase 4: Never-read store elimination
    eliminate_never_read_stores(&store, &mut infos);

    store.build_result(|i| infos[i].is_nop())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redundant_store_load() {
        let asm = "    movl %eax, -8(%ebp)\n    movl -8(%ebp), %eax\n".to_string();
        let result = peephole_optimize(asm);
        // After store/load elimination, the load is removed. Then never-read
        // store elimination removes the now-unread store too. Both gone.
        assert_eq!(result.trim(), "");
    }

    #[test]
    fn test_store_load_different_reg() {
        let asm = "    movl %eax, -8(%ebp)\n    movl -8(%ebp), %ecx\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("movl %eax, %ecx"), "should forward: {}", result);
        assert!(!result.contains("-8(%ebp), %ecx"), "should eliminate load: {}", result);
    }

    #[test]
    fn test_self_move() {
        let asm = "    movl %eax, %eax\n".to_string();
        let result = peephole_optimize(asm);
        assert_eq!(result.trim(), "");
    }

    #[test]
    fn test_redundant_jump() {
        let asm = "    jmp .Lfoo\n.Lfoo:\n".to_string();
        let result = peephole_optimize(asm);
        assert!(!result.contains("jmp"), "should eliminate redundant jmp: {}", result);
        assert!(result.contains(".Lfoo:"), "should keep label: {}", result);
    }

    #[test]
    fn test_branch_inversion() {
        let asm = [
            "    jl .LBB2",
            "    jmp .LBB4",
            ".LBB2:",
            "    movl %eax, %ecx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jge .LBB4"), "should invert to jge: {}", result);
        assert!(!result.contains("jmp .LBB4"), "should remove jmp: {}", result);
    }

    #[test]
    fn test_compare_branch_fusion() {
        let asm = [
            "    cmpl %ecx, %eax",
            "    setl %al",
            "    movzbl %al, %eax",
            "    testl %eax, %eax",
            "    jne .LBB2",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jl .LBB2"), "should fuse to jl: {}", result);
        assert!(!result.contains("setl"), "should eliminate setl: {}", result);
    }

    #[test]
    fn test_compare_branch_fusion_with_store_load() {
        // Pattern: cmp + setCC + movzbl + store + load + test + jne
        // The store/load pair should be skipped, allowing fusion.
        let asm = [
            "    cmpl %ecx, %eax",
            "    setge %al",
            "    movzbl %al, %eax",
            "    movl %eax, -16(%ebp)",
            "    movl -16(%ebp), %eax",
            "    testl %eax, %eax",
            "    jne .LBB5",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jge .LBB5"), "should fuse to jge: {}", result);
        assert!(!result.contains("setge"), "should eliminate setge: {}", result);
        assert!(!result.contains("movzbl"), "should eliminate movzbl: {}", result);
        assert!(!result.contains("testl"), "should eliminate testl: {}", result);
    }

    #[test]
    fn test_compare_branch_fusion_unmatched_store_bails() {
        // If the store has no matching load, we should NOT fuse (the boolean
        // escapes to another basic block).
        let asm = [
            "    cmpl %ecx, %eax",
            "    setge %al",
            "    movzbl %al, %eax",
            "    movl %eax, -16(%ebp)",
            "    testl %eax, %eax",
            "    jne .LBB5",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        // Should NOT fuse because the store has no matching load
        assert!(result.contains("setge"), "should keep setge (unmatched store): {}", result);
    }

    #[test]
    fn test_compare_branch_fusion_inverted() {
        // Test je (inverted condition)
        let asm = [
            "    cmpl %ecx, %eax",
            "    setl %al",
            "    movzbl %al, %eax",
            "    testl %eax, %eax",
            "    je .LBB3",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jge .LBB3"), "should fuse to jge (inverted): {}", result);
        assert!(!result.contains("setl"), "should eliminate setl: {}", result);
    }

    #[test]
    fn test_dead_store() {
        // Two consecutive stores to the same slot: first is dead, second survives.
        // But never-read store elimination also removes the second store if no
        // loads exist. Use a load after the second store to keep it alive.
        let asm = [
            "    movl %eax, -8(%ebp)",
            "    movl %ecx, -8(%ebp)",
            "    movl -8(%ebp), %edx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(!result.contains("%eax, -8(%ebp)"), "first store dead: {}", result);
        assert!(result.contains("%ecx"), "second store alive: {}", result);
    }

    #[test]
    fn test_memory_fold() {
        let asm = [
            "    movl -48(%ebp), %ecx",
            "    addl %ecx, %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("addl -48(%ebp), %eax"), "should fold: {}", result);
    }

    // Note: store forwarding tests removed - global_store_forwarding is disabled
    // due to FP computation regressions.

    #[test]
    fn test_reverse_move_elimination() {
        let asm = [
            "    movl %eax, %ecx",
            "    movl %ecx, %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("movl").count(), 1, "should eliminate reverse: {}", result);
    }

    // Note: push/pop elimination test removed - eliminate_push_pop_pairs is disabled
    // due to callee-save/leal epilogue interactions.

    #[test]
    fn test_addl_1_to_incl() {
        let asm = "    addl $1, %eax\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("incl %eax"), "should convert to incl: {}", result);
        assert!(!result.contains("addl"), "should eliminate addl: {}", result);
    }

    #[test]
    fn test_subl_1_to_decl() {
        let asm = "    subl $1, %ecx\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("decl %ecx"), "should convert to decl: {}", result);
        assert!(!result.contains("subl"), "should eliminate subl: {}", result);
    }

    #[test]
    fn test_movl_0_to_xorl() {
        let asm = "    movl $0, %ebx\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("xorl %ebx, %ebx"), "should convert to xorl: {}", result);
        assert!(!result.contains("movl"), "should eliminate movl: {}", result);
    }

    #[test]
    fn test_redundant_movsbl() {
        let asm = [
            "    movsbl (%ecx), %eax",
            "    movsbl %al, %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("movsbl").count(), 1,
            "should eliminate redundant movsbl: {}", result);
    }

    #[test]
    fn test_addl_neg1_to_decl() {
        let asm = "    addl $-1, %edx\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("decl %edx"), "should convert to decl: {}", result);
    }

    #[test]
    fn test_subl_neg1_to_incl() {
        let asm = "    subl $-1, %esi\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("incl %esi"), "should convert to incl: {}", result);
    }

    #[test]
    fn test_addl_1_not_incl_before_adcl() {
        // addl $1 followed by adcl must NOT be converted to incl,
        // because incl does not set the carry flag (CF).
        // This pattern is used in 64-bit negation: notl+notl+addl+adcl.
        let asm = [
            "    notl %eax",
            "    notl %edx",
            "    addl $1, %eax",
            "    adcl $0, %edx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("addl $1, %eax"), "must keep addl before adcl: {}", result);
        assert!(!result.contains("incl"), "must NOT convert to incl before adcl: {}", result);
    }

    #[test]
    fn test_subl_1_not_decl_before_sbbl() {
        // subl $1 followed by sbbl must NOT be converted to decl,
        // because decl does not set the carry flag.
        let asm = [
            "    subl $1, %eax",
            "    sbbl $0, %edx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("subl $1, %eax"), "must keep subl before sbbl: {}", result);
        assert!(!result.contains("decl"), "must NOT convert to decl before sbbl: {}", result);
    }
}
