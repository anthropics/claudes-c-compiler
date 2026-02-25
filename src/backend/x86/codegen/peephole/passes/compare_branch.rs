//! Compare-and-branch fusion pass.
//!
//! Fuses cmp + setCC + test + jCC sequences into a single conditional jump,
//! eliminating the boolean materialization overhead from the codegen model.

use super::super::types::*;

/// Maximum number of store/load offsets tracked during compare-and-branch fusion.
const MAX_TRACKED_STORE_LOAD_OFFSETS: usize = 4;

/// Size of the instruction lookahead window for compare-and-branch fusion.
const CMP_FUSION_LOOKAHEAD: usize = 8;

pub(super) fn fuse_compare_and_branch(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    let mut i = 0;
    while i < len {
        if infos[i].kind != LineKind::Cmp {
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

        // Second must be setCC
        if !matches!(infos[seq_indices[1]].kind, LineKind::SetCC { .. }) {
            i += 1;
            continue;
        }
        let set_line = infos[seq_indices[1]].trimmed(store.get(seq_indices[1]));
        let cc = match parse_setcc(set_line) {
            Some(c) => c,
            None => { i += 1; continue; }
        };

        // Scan for testq %rax, %rax pattern.
        // Track StoreRbp offsets so we can bail out if any store's slot is
        // potentially read by another basic block (no matching load nearby).
        let mut test_idx = None;
        let mut store_offsets: [i32; MAX_TRACKED_STORE_LOAD_OFFSETS] = [0; MAX_TRACKED_STORE_LOAD_OFFSETS];
        let mut store_count = 0usize;
        let mut scan = 2;
        while scan < seq_count {
            let si = seq_indices[scan];
            let line = infos[si].trimmed(store.get(si));

            // Skip zero-extend of setcc result
            if line.starts_with("movzbq %al,") || line.starts_with("movzbl %al,") {
                scan += 1;
                continue;
            }
            // Skip store/load to rbp (pre-parsed fast check).
            if let LineKind::StoreRbp { offset, .. } = infos[si].kind {
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
            if matches!(infos[si].kind, LineKind::LoadRbp { .. }) {
                scan += 1;
                continue;
            }
            // Skip cltq and movslq
            if line == "cltq" || line.starts_with("movslq ") {
                scan += 1;
                continue;
            }
            // Check for test
            if line == "testq %rax, %rax" || line == "testl %eax, %eax" {
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
                    LineKind::LoadRbp { offset, .. } => Some(offset),
                    LineKind::Nop => {
                        let orig = classify_line(store.get(ri));
                        match orig.kind {
                            LineKind::LoadRbp { offset, .. } => Some(offset),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(o) = off {
                    if load_count < MAX_TRACKED_STORE_LOAD_OFFSETS { load_offsets[load_count] = o; load_count += 1; }
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

        let jmp_line = infos[seq_indices[test_scan + 1]].trimmed(store.get(seq_indices[test_scan + 1]));
        let (is_jne, branch_target) = if let Some(target) = jmp_line.strip_prefix("jne ") {
            (true, target.trim())
        } else if let Some(target) = jmp_line.strip_prefix("je ") {
            (false, target.trim())
        } else {
            i += 1;
            continue;
        };

        let fused_cc = if is_jne { cc } else { invert_cc(cc) };
        let fused_jcc = format!("    j{} {}", fused_cc, branch_target);

        // NOP out everything from setCC through testq
        for s in 1..=test_scan {
            mark_nop(&mut infos[seq_indices[s]]);
        }
        // Replace the jne/je with the fused conditional jump
        let idx = seq_indices[test_scan + 1];
        replace_line(store, &mut infos[idx], idx, fused_jcc);

        changed = true;
        i = idx + 1;
    }

    changed
}

// ── AND/test/branch fusion ──────────────────────────────────────────────────

/// Eliminate redundant test instructions after flag-setting AND operations,
/// and convert dead ANDs to non-destructive TEST instructions.
///
/// Pattern: `andl $IMM, %rB; [mov chain]; testl %rC, %rC; jCC target`
///
/// The `andl` already sets ZF/SF/PF flags based on its result. Since `mov`
/// instructions do not modify flags, any intervening `movl`/`movq` preserve
/// the flags from the `andl`. The `testl` is therefore redundant.
///
/// Step 1: NOP the redundant `testl` (always valid).
/// Step 2: If the AND result register is dead after the `jCC`, convert the
///         `andl $IMM, %reg` to `testl $IMM, %orig_reg` (non-destructive).
///         This traces back through a preceding `movq` to use the original
///         source register, and NOPs the now-dead intermediate moves.
pub(super) fn fuse_and_test_branch(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // Look for andl/andq $IMM, %reg
        if !matches!(infos[i].kind, LineKind::Other { .. }) {
            i += 1;
            continue;
        }

        let trimmed_i = infos[i].trimmed(store.get(i));
        let (and_imm, and_dest_reg, is_andq) = match parse_and_imm(trimmed_i) {
            Some(v) => v,
            None => { i += 1; continue; }
        };

        // Don't touch rsp/rbp
        if and_dest_reg == 4 || and_dest_reg == 5 {
            i += 1;
            continue;
        }

        // Track which registers hold the AND result
        let mut result_regs = 1u16 << and_dest_reg;

        // Scan forward, tracking mov chain, looking for testl/testq + jCC
        let mut j = i + 1;
        let scan_end = (i + 6).min(len);
        let mut test_idx = None;
        let mut intermediate_mov_indices: [usize; 4] = [0; 4];
        let mut mov_count: usize = 0;

        while j < scan_end {
            if infos[j].is_nop() {
                j += 1;
                continue;
            }

            if infos[j].is_barrier() {
                break;
            }

            // Check for testl/testq of a result register
            if infos[j].kind == LineKind::Cmp {
                let trimmed_j = infos[j].trimmed(store.get(j));
                if let Some(test_reg) = parse_test_self(trimmed_j) {
                    if result_regs & (1u16 << test_reg) != 0 {
                        test_idx = Some(j);
                    }
                }
                break;
            }

            let trimmed_j = infos[j].trimmed(store.get(j));

            // Track movl/movq from a result reg to another reg
            if let Some((src_reg, dst_reg)) = parse_any_reg_mov(trimmed_j) {
                if result_regs & (1u16 << src_reg) != 0 && dst_reg != 4 && dst_reg != 5 {
                    result_regs |= 1u16 << dst_reg;
                    if mov_count < intermediate_mov_indices.len() {
                        intermediate_mov_indices[mov_count] = j;
                        mov_count += 1;
                    }
                    j += 1;
                    continue;
                }
            }

            // Any instruction that modifies flags -> stop scanning
            if !is_flag_preserving(trimmed_j) {
                break;
            }

            j += 1;
        }

        let test_j = match test_idx {
            Some(tj) => tj,
            None => { i += 1; continue; }
        };

        // Find the jCC after the testl
        let mut jcc_idx = None;
        {
            let mut k = test_j + 1;
            while k < len {
                if infos[k].is_nop() {
                    k += 1;
                    continue;
                }
                if matches!(infos[k].kind, LineKind::CondJmp) {
                    jcc_idx = Some(k);
                }
                break;
            }
        }

        let jcc_k = match jcc_idx {
            Some(jk) => jk,
            None => { i += 1; continue; }
        };

        // ── Step 1: NOP the redundant testl ──
        mark_nop(&mut infos[test_j]);
        changed = true;

        // ── Step 2: Try to convert andl → testl ──
        // Check if the AND result and all intermediate mov destinations are dead.
        let and_result_dead = super::local_patterns::is_reg_unused_after_ext(
            infos, store, jcc_k + 1, len, and_dest_reg, &targets
        );

        let mut movs_all_dead = true;
        for mi in 0..mov_count {
            let mov_dest = super::helpers::get_dest_reg(&infos[intermediate_mov_indices[mi]]);
            if mov_dest == REG_NONE || !super::local_patterns::is_reg_unused_after_ext(
                infos, store, jcc_k + 1, len, mov_dest, &targets
            ) {
                movs_all_dead = false;
                break;
            }
        }

        if and_result_dead && movs_all_dead {
            // NOP intermediate movs
            for mi in 0..mov_count {
                mark_nop(&mut infos[intermediate_mov_indices[mi]]);
            }

            // Look for a preceding movq %rA, %and_dest to trace to original source.
            let mut orig_reg = and_dest_reg;
            let mut prev_mov_idx: Option<usize> = None;

            if i > 0 {
                let mut pi = i.saturating_sub(1);
                while pi > 0 && infos[pi].is_nop() {
                    pi -= 1;
                }
                if !infos[pi].is_nop() {
                    let trimmed_pi = infos[pi].trimmed(store.get(pi));
                    if let Some((src, dst)) = super::helpers::parse_reg_to_reg_movq(&infos[pi], trimmed_pi) {
                        if dst == and_dest_reg && src != and_dest_reg {
                            orig_reg = src;
                            prev_mov_idx = Some(pi);
                        }
                    }
                }
            }

            // Convert andl → testl using the (possibly traced-back) source register
            let test_suffix = if is_andq { "q" } else { "l" };
            let size_idx: usize = if is_andq { 0 } else { 1 };
            let reg_name = REG_NAMES[size_idx][orig_reg as usize];
            let new_text = format!("    test{} ${}, {}", test_suffix, and_imm, reg_name);
            replace_line(store, &mut infos[i], i, new_text);

            // NOP the preceding movq if we traced through it
            if prev_mov_idx.is_some() {
                mark_nop(&mut infos[prev_mov_idx.unwrap()]);
            }
        }

        i = jcc_k + 1;
    }

    changed
}

// ── Helpers for AND/test fusion ─────────────────────────────────────────────

/// Parse `andl $IMM, %reg` or `andq $IMM, %reg`.
/// Returns (immediate_string, dest_register_family, is_andq).
fn parse_and_imm(trimmed: &str) -> Option<(&str, RegId, bool)> {
    let (rest, is_andq) = if let Some(r) = trimmed.strip_prefix("andl $") {
        (r, false)
    } else if let Some(r) = trimmed.strip_prefix("andq $") {
        (r, true)
    } else {
        return None;
    };

    let comma_pos = rest.find(", ")?;
    let imm = &rest[..comma_pos];
    let reg_str = rest[comma_pos + 2..].trim();
    let reg_fam = register_family_fast(reg_str);
    if reg_fam == REG_NONE || reg_fam > REG_GP_MAX {
        return None;
    }
    Some((imm, reg_fam, is_andq))
}

/// Parse `testl %reg, %reg` or `testq %reg, %reg` (self-test).
/// Returns the register family if both operands are the same register.
fn parse_test_self(trimmed: &str) -> Option<RegId> {
    let rest = if let Some(r) = trimmed.strip_prefix("testl ") {
        r
    } else if let Some(r) = trimmed.strip_prefix("testq ") {
        r
    } else {
        return None;
    };

    let (left, right) = rest.split_once(", ")?;
    let left = left.trim();
    let right = right.trim();
    if left != right {
        return None;
    }
    let fam = register_family_fast(left);
    if fam == REG_NONE || fam > REG_GP_MAX {
        return None;
    }
    Some(fam)
}

/// Parse a reg-to-reg mov of any size: `movl %src, %dst` or `movq %src, %dst`.
/// Returns (src_family, dst_family). Excludes memory operands.
fn parse_any_reg_mov(trimmed: &str) -> Option<(RegId, RegId)> {
    let rest = if let Some(r) = trimmed.strip_prefix("movl ") {
        r
    } else if let Some(r) = trimmed.strip_prefix("movq ") {
        r
    } else {
        return None;
    };

    let (src, dst) = rest.split_once(", ")?;
    let src = src.trim();
    let dst = dst.trim();
    if !src.starts_with('%') || !dst.starts_with('%') || src.contains('(') || dst.contains('(') {
        return None;
    }
    let sfam = register_family_fast(src);
    let dfam = register_family_fast(dst);
    if sfam == REG_NONE || sfam > REG_GP_MAX || dfam == REG_NONE || dfam > REG_GP_MAX {
        return None;
    }
    Some((sfam, dfam))
}

/// Check if an instruction preserves flags (does not modify EFLAGS).
fn is_flag_preserving(trimmed: &str) -> bool {
    let b = trimmed.as_bytes();
    if b.len() < 3 {
        return false;
    }
    // mov* (movq, movl, movb, movw, movzbl, movzbq, movslq, movabs, etc.)
    if b[0] == b'm' && b[1] == b'o' && b[2] == b'v' {
        return true;
    }
    // lea* (leaq, leal)
    if b[0] == b'l' && b[1] == b'e' && b[2] == b'a' {
        return true;
    }
    // pushq / popq
    if trimmed.starts_with("pushq ") || trimmed.starts_with("popq ") {
        return true;
    }
    false
}
