//! Shared helper functions used across peephole optimization passes.
//!
//! These are small utility functions for register manipulation, label parsing,
//! epilogue detection, and instruction analysis that multiple passes need.

use super::super::types::*;

// ── Register helpers ─────────────────────────────────────────────────────────

/// Check if a register ID refers to a valid general-purpose register (0..=15).
#[inline]
pub(super) fn is_valid_gp_reg(reg: RegId) -> bool {
    reg != REG_NONE && reg <= REG_GP_MAX
}

/// Check if a register family ID is callee-saved (rbx=3, r12=12, r13=13, r14=14, r15=15).
pub(super) fn is_callee_saved_reg(reg: RegId) -> bool {
    matches!(reg, 3 | 12 | 13 | 14 | 15)
}

/// Check if a line of assembly text references a given register family.
/// Uses the pre-computed reg_refs bitmask for O(1) lookup.
#[inline]
pub(super) fn line_references_reg_fast(info: &LineInfo, reg: RegId) -> bool {
    info.reg_refs & (1u16 << reg) != 0
}

// ── Register rewriting ───────────────────────────────────────────────────────

/// Replace all register-family occurrences of `old_family` with `new_family` in `line`.
/// Handles all register sizes: 64-bit (%rax), 32-bit (%eax), 16-bit (%ax), 8-bit (%al).
/// For example, replacing family 0 (rax) with family 1 (rcx) will convert:
///   %rax -> %rcx, %eax -> %ecx, %ax -> %cx, %al -> %cl
pub(super) fn replace_reg_family(line: &str, old_id: RegId, new_id: RegId) -> String {
    let mut result = line.to_string();
    // Replace in order from longest to shortest to avoid partial matches.
    // 64-bit names are longest (e.g., %r10, %rax), then 32-bit, 16-bit, 8-bit.
    for size_idx in 0..4 {
        let old_name = REG_NAMES[size_idx][old_id as usize];
        let new_name = REG_NAMES[size_idx][new_id as usize];
        if old_name == new_name {
            continue;
        }
        result = replace_reg_name_exact(&result, old_name, new_name);
    }
    result
}

/// Replace all complete occurrences of `old_reg` with `new_reg` in `line`.
/// A "complete" occurrence means `old_reg` is not a prefix of a longer register
/// name (e.g., replacing `%r8` must not match `%r8d` or `%r8b`).
/// After `old_reg`, the next character must be a delimiter: `,`, `)`, ` `, or end-of-string.
pub(super) fn replace_reg_name_exact(line: &str, old_reg: &str, new_reg: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let old_bytes = old_reg.as_bytes();
    let old_len = old_bytes.len();
    let mut pos = 0;

    while pos < bytes.len() {
        if pos + old_len <= bytes.len() && &bytes[pos..pos + old_len] == old_bytes {
            // Check that this is a complete register name
            let after = pos + old_len;
            let is_complete = after >= bytes.len()
                || matches!(bytes[after], b',' | b')' | b' ' | b'\t' | b'\n');
            if is_complete {
                result.push_str(new_reg);
                pos += old_len;
                continue;
            }
        }
        result.push(bytes[pos] as char);
        pos += 1;
    }
    result
}

/// Replace register family occurrences only in the source operand part
/// (before the last comma). The destination operand is left unchanged.
pub(super) fn replace_reg_family_in_source(line: &str, old_id: RegId, new_id: RegId) -> String {
    if let Some(comma_pos) = line.rfind(',') {
        let src_part = &line[..comma_pos];
        let dst_part = &line[comma_pos..];
        let new_src = replace_reg_family(src_part, old_id, new_id);
        format!("{}{}", new_src, dst_part)
    } else {
        replace_reg_family(line, old_id, new_id)
    }
}

// ── Instruction analysis ─────────────────────────────────────────────────────

/// Check if an instruction has implicit register usage that makes register
/// substitution unsafe (div/idiv/mul use rax/rdx, shifts use cl, etc.).
pub(super) fn has_implicit_reg_usage(trimmed: &str) -> bool {
    let nb = trimmed.as_bytes();
    if nb.len() < 3 {
        return false;
    }
    (nb[0] == b'd' && nb[1] == b'i' && nb[2] == b'v') || // div, divl, divq
    (nb[0] == b'i' && nb[1] == b'd' && nb[2] == b'i') || // idiv
    (nb[0] == b'm' && nb[1] == b'u' && nb[2] == b'l') || // mul, mulq
    trimmed.starts_with("cltq") || // sign-extend eax -> rax (implicit read+write rax)
    trimmed.starts_with("cbw") ||  // sign-extend al -> ax (implicit read+write rax)
    trimmed.starts_with("cqto") || trimmed.starts_with("cdq") ||
    trimmed.starts_with("cqo") || trimmed.starts_with("cwd") ||
    trimmed.starts_with("rep ") || trimmed.starts_with("repne ") ||
    trimmed.starts_with("cpuid") || trimmed.starts_with("syscall") ||
    trimmed.starts_with("rdtsc") || trimmed.starts_with("rdmsr") ||
    trimmed.starts_with("wrmsr") ||
    trimmed.starts_with("xchg") || trimmed.starts_with("cmpxchg") ||
    trimmed.starts_with("lock ") ||
    // x87 FPU status/control instructions that write to %ax or memory.
    // fnstsw writes to %ax (implicit destination), and the peephole's
    // copy propagation must not rewrite %ax to another register since
    // the x86 ISA only allows %ax or memory for fnstsw.
    trimmed.starts_with("fnstsw") || trimmed.starts_with("fnstcw") ||
    trimmed.starts_with("fstcw") || trimmed.starts_with("fnstenv") ||
    trimmed.starts_with("fldenv") || trimmed.starts_with("fldcw")
}

/// Check if an instruction is a shift/rotate that implicitly uses %cl.
pub(super) fn is_shift_or_rotate(trimmed: &str) -> bool {
    let nb = trimmed.as_bytes();
    nb.len() >= 3 && (
        (nb[0] == b's' && (nb[1] == b'h' || nb[1] == b'a')) || // shl, shr, sal, sar
        (nb[0] == b'r' && (nb[1] == b'o' || nb[1] == b'c'))    // rol, ror, rcl, rcr
    )
}

/// Parse a movq %src, %dst instruction. Returns (src_family, dst_family) if valid.
pub(super) fn parse_reg_to_reg_movq(info: &LineInfo, trimmed: &str) -> Option<(RegId, RegId)> {
    if let LineKind::Other { dest_reg } = info.kind {
        if dest_reg == REG_NONE || dest_reg > REG_GP_MAX {
            return None;
        }
        if let Some(rest) = trimmed.strip_prefix("movq ") {
            if let Some((src_part, dst_part)) = rest.split_once(',') {
                let src = src_part.trim();
                let dst = dst_part.trim();
                if !src.starts_with("%r") || !dst.starts_with("%r") {
                    return None;
                }
                let sfam = register_family_fast(src);
                let dfam = register_family_fast(dst);
                if sfam == REG_NONE || sfam > REG_GP_MAX || sfam == 4 || sfam == 5
                    || dfam == REG_NONE || dfam > REG_GP_MAX || dfam == 4 || dfam == 5
                    || sfam == dfam
                {
                    return None;
                }
                return Some((sfam, dfam));
            }
        }
    }
    None
}

/// Get the destination register of an instruction (the register it writes to).
pub(super) fn get_dest_reg(info: &LineInfo) -> RegId {
    match info.kind {
        LineKind::Other { dest_reg } => dest_reg,
        LineKind::StoreRbp { .. } => REG_NONE, // stores don't write to a register
        LineKind::LoadRbp { reg, .. } => reg,
        LineKind::SetCC { reg } => reg,
        LineKind::Pop { reg } => reg,
        LineKind::Cmp => REG_NONE,
        LineKind::Push { .. } => REG_NONE,
        _ => REG_NONE,
    }
}

// ── Label/jump parsing ───────────────────────────────────────────────────────

/// Parse ".LBB<number>:" label into its number, e.g. ".LBB123:" -> Some(123)
#[inline]
pub(super) fn parse_label_number(label_with_colon: &str) -> Option<u32> {
    let s = label_with_colon.strip_suffix(':')?;
    parse_dotl_number(s)
}

/// Parse ".LBB<number>" into its number, e.g. ".LBB123" -> Some(123)
#[inline]
pub(super) fn parse_dotl_number(s: &str) -> Option<u32> {
    let rest = s.strip_prefix(".LBB")?;
    let b = rest.as_bytes();
    if b.is_empty() || b[0] < b'0' || b[0] > b'9' {
        return None;
    }
    let mut v: u32 = 0;
    for &c in b {
        if c.is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((c - b'0') as u32);
        } else {
            return None;
        }
    }
    Some(v)
}

/// Extract the jump target label from a jump/branch instruction.
pub(super) fn extract_jump_target(s: &str) -> Option<&str> {
    // Handle: jmp .LBB1, je .LBB1, jne .LBB1, etc.
    if let Some(rest) = s.strip_prefix("jmp ") {
        return Some(rest.trim());
    }
    // Conditional jumps: j<cc> <target>
    if s.starts_with('j') {
        if let Some(space) = s.find(' ') {
            return Some(s[space + 1..].trim());
        }
    }
    None
}

// ── Epilogue detection ───────────────────────────────────────────────────────

/// Check if a LoadRbp at position `pos` is part of the function epilogue.
/// The epilogue pattern is: callee-save restores, then `movq %rbp, %rsp; popq %rbp; ret/jmp`.
/// We look forward from `pos` for a Ret or Jmp (for __x86_return_thunk) within a small window,
/// only allowing other LoadRbp or Pop or Other (for stack teardown) instructions between.
pub(super) fn is_near_epilogue(infos: &[LineInfo], pos: usize) -> bool {
    let limit = (pos + 20).min(infos.len());
    for j in (pos + 1)..limit {
        if infos[j].is_nop() {
            continue;
        }
        match infos[j].kind {
            // More callee-save restores or stack teardown moves are expected
            LineKind::LoadRbp { .. } | LineKind::Pop { .. } | LineKind::SelfMove => continue,
            // Stack pointer restoration (movq %rbp, %rsp) is classified as Other
            LineKind::Other { .. } => continue,
            // Found the return instruction - this is an epilogue
            LineKind::Ret | LineKind::Jmp | LineKind::JmpIndirect => return true,
            // Any other instruction type means we're not in the epilogue
            _ => return false,
        }
    }
    false
}

/// Check if an x86 instruction is a read-modify-write for its destination register.
/// For most two-operand instructions like `addq %src, %dst`, the destination is read
/// before being written.  For move instructions like `movq %src, %dst` or `leaq ...(%rip), %dst`,
/// the destination is only written.
///
/// Returns true if the instruction reads the destination register (read-modify-write).
pub(super) fn is_read_modify_write(trimmed: &str) -> bool {
    let b = trimmed.as_bytes();
    if b.len() < 3 {
        return true; // conservative
    }

    // Move instructions are write-only for the destination:
    // movq, movl, movw, movb, movabs, movzbl, movzbq, movzwl, movzwq,
    // movslq, movsbl, movsbq, movswl, movswq
    if b[0] == b'm' && b[1] == b'o' && b[2] == b'v' {
        return false;
    }

    // LEA is write-only for the destination, UNLESS the destination register
    // also appears in the address computation (source operand). For example:
    //   leaq 4(%rax), %rax   -- reads %rax (base) and writes %rax (dest)
    //   leaq (%rax,%rcx), %rax  -- reads %rax and writes %rax
    //   leaq 8(%rcx), %rax   -- does NOT read %rax, only writes it
    if b[0] == b'l' && b[1] == b'e' && b[2] == b'a' {
        // Check if dest register appears in the source (memory) operand.
        // Source operand is everything before the last comma.
        if let Some(comma) = trimmed.rfind(',') {
            let dest_part = trimmed[comma + 1..].trim();
            let src_part = &trimmed[..comma];
            // If the dest register name appears in the source part, it's a read
            if src_part.contains(dest_part) {
                return true;
            }
        }
        return false;
    }

    // Conditional moves (cmovXX) are read-modify-write because the old value
    // is kept when the condition is false.
    if b[0] == b'c' && b[1] == b'm' && b[2] == b'o' {
        return true; // cmov reads the original destination
    }

    // Check if it's a setCC instruction (write-only for byte reg)
    if b[0] == b's' && b[1] == b'e' && b[2] == b't' {
        return false;
    }

    // Default: assume read-modify-write (conservative)
    true
}

// ── Jump target analysis ──────────────────────────────────────────────────────

/// Jump target analysis result, shared by store forwarding and copy propagation.
pub(super) struct JumpTargets {
    pub is_jump_target: Vec<bool>,
    pub has_non_numeric_jump_targets: bool,
}

/// Collect all jump targets in the assembly to distinguish fallthrough labels
/// from labels that are actual branch targets. Labels only reached by fallthrough
/// don't break linear instruction flow, so optimizations can propagate state
/// across them safely.
pub(super) fn collect_jump_targets(store: &LineStore, infos: &[LineInfo], len: usize) -> JumpTargets {
    let mut max_label_num: u32 = 0;
    for i in 0..len {
        if infos[i].kind == LineKind::Label {
            let trimmed = infos[i].trimmed(store.get(i));
            if let Some(n) = parse_label_number(trimmed) {
                if n > max_label_num {
                    max_label_num = n;
                }
            }
        }
    }
    let mut is_jump_target = vec![false; (max_label_num + 1) as usize];
    let mut has_non_numeric_jump_targets = false;
    let mut has_indirect_jump = false;
    for i in 0..len {
        match infos[i].kind {
            LineKind::Jmp | LineKind::CondJmp => {
                let trimmed = infos[i].trimmed(store.get(i));
                if let Some(target) = extract_jump_target(trimmed) {
                    if let Some(n) = parse_dotl_number(target) {
                        if (n as usize) < is_jump_target.len() {
                            is_jump_target[n as usize] = true;
                        }
                    } else {
                        has_non_numeric_jump_targets = true;
                    }
                }
            }
            LineKind::JmpIndirect => {
                has_indirect_jump = true;
            }
            _ => {}
        }
    }
    if has_indirect_jump {
        for v in is_jump_target.iter_mut() {
            *v = true;
        }
        has_non_numeric_jump_targets = true;
    }
    JumpTargets { is_jump_target, has_non_numeric_jump_targets }
}
