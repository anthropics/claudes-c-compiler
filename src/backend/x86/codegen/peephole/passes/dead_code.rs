//! Dead code elimination passes.
//!
//! Three passes that eliminate dead instructions:
//! - `eliminate_dead_reg_moves`: removes reg-to-reg moves where the destination
//!   is overwritten before being read (local, forward scan within a window).
//! - `eliminate_dead_stores`: removes stores to stack slots that are overwritten
//!   before being read (local, 16-instruction window).
//! - `eliminate_never_read_stores`: removes stores to stack slots that are never
//!   read anywhere in the function (global, whole-function analysis).

use super::super::types::*;
use super::helpers::*;

// ── Dead register move elimination ──────────────────────────────────────────

/// Maximum forward scan window for dead register move detection.
const DEAD_MOVE_WINDOW: usize = 24;

pub(super) fn eliminate_dead_reg_moves(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let mut i = 0;

    while i < len {
        if infos[i].is_nop() || infos[i].is_barrier() {
            i += 1;
            continue;
        }

        // Check if this is a reg-to-reg movq.
        let dst_reg = match infos[i].kind {
            LineKind::Other { dest_reg } => {
                let trimmed = infos[i].trimmed(store.get(i));
                if parse_reg_to_reg_movq(&infos[i], trimmed).is_some() {
                    dest_reg
                } else {
                    i += 1;
                    continue;
                }
            }
            _ => {
                i += 1;
                continue;
            }
        };

        if dst_reg == REG_NONE || dst_reg > REG_GP_MAX {
            i += 1;
            continue;
        }
        // Don't eliminate moves to %rsp or %rbp.
        if dst_reg == 4 || dst_reg == 5 {
            i += 1;
            continue;
        }

        let dst_mask = 1u16 << dst_reg;
        let mut dead = false;

        // Scan forward within the same basic block.
        let mut j = i + 1;
        let scan_end = (i + DEAD_MOVE_WINDOW).min(len);
        while j < scan_end {
            if infos[j].is_nop() {
                j += 1;
                continue;
            }

            if infos[j].is_barrier() {
                break;
            }

            {
                let trimmed_j = infos[j].trimmed(store.get(j));
                if has_implicit_reg_usage(trimmed_j) {
                    break;
                }
            }

            let refs_dst = infos[j].reg_refs & dst_mask != 0;
            let writes_dst = get_dest_reg(&infos[j]) == dst_reg;

            if writes_dst {
                let also_reads = match infos[j].kind {
                    LineKind::LoadRbp { .. } => false,
                    LineKind::Pop { .. } => false,
                    LineKind::SetCC { .. } => false,
                    LineKind::Other { .. } => {
                        if !refs_dst {
                            false
                        } else {
                            let t = infos[j].trimmed(store.get(j));
                            if is_read_modify_write(t) {
                                true
                            } else {
                                // Defense-in-depth: is_read_modify_write uses exact
                                // string matching for LEA source checks, which misses
                                // cross-size register references (e.g., %eax in source
                                // vs %rax in dest). Use REG_NAMES to match all size
                                // variants of the dest register in the source operand.
                                if let Some(comma_pos) = t.rfind(',') {
                                    let src_part = &t[..comma_pos];
                                    REG_NAMES.iter().any(|row|
                                        src_part.contains(row[dst_reg as usize])
                                    )
                                } else {
                                    // Single-operand instruction that both reads
                                    // and writes (e.g., negq %rax) - conservative
                                    true
                                }
                            }
                        }
                    }
                    _ => refs_dst,
                };

                if also_reads {
                    break;
                } else {
                    dead = true;
                    break;
                }
            }

            if refs_dst {
                break;
            }

            j += 1;
        }

        if dead {
            mark_nop(&mut infos[i]);
            changed = true;
        }

        i += 1;
    }

    changed
}

// ── Extended dead register move elimination ──────────────────────────────────
//
// Like eliminate_dead_reg_moves but uses extended liveness analysis (depth=5)
// to prove dead moves across conditional jumps and labels. Uses the call-safe
// variant that treats function calls as barriers, preventing incorrect
// elimination of argument-setup moves.

pub(super) fn eliminate_dead_reg_moves_ext(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() || infos[i].is_barrier() {
            i += 1;
            continue;
        }

        let dst_reg = match infos[i].kind {
            LineKind::Other { dest_reg } => {
                let trimmed = infos[i].trimmed(store.get(i));
                if parse_reg_to_reg_movq(&infos[i], trimmed).is_some() {
                    dest_reg
                } else {
                    i += 1;
                    continue;
                }
            }
            _ => {
                i += 1;
                continue;
            }
        };

        if dst_reg == REG_NONE || dst_reg > REG_GP_MAX || dst_reg == 4 || dst_reg == 5 {
            i += 1;
            continue;
        }

        if super::local_patterns::is_reg_unused_after_ext(infos, store, i + 1, len, dst_reg, &targets) {
            mark_nop(&mut infos[i]);
            changed = true;
        }

        i += 1;
    }

    changed
}

// ── Dead argument register move elimination ──────────────────────────────────
//
// Eliminates moves to argument registers (%rdi, %rsi, %rdx, %rcx, %r8, %r9)
// that precede a call to a known zero-argument function. In System V AMD64 ABI,
// these registers are used to pass the first 6 integer/pointer arguments. When
// the call target takes zero arguments, any setup moves to these registers are
// dead code.
//
// Example eliminated (strprocess hot loop):
//   movq %rax, %rsi       ; dead — __ctype_b_loc takes 0 args
//   movq %rax, %rdi       ; dead — __ctype_b_loc takes 0 args
//   movq %rax, %r14
//   xorl %eax, %eax
//   call __ctype_b_loc

/// Known zero-argument C library functions.
fn is_zero_arg_call(trimmed: &str) -> bool {
    if let Some(target) = trimmed.strip_prefix("call ") {
        let target = target.trim();
        matches!(target,
            "__ctype_b_loc" | "__ctype_toupper_loc" | "__ctype_tolower_loc"
            | "clock" | "getpid" | "getppid" | "getuid" | "geteuid"
            | "getgid" | "getegid" | "fork" | "__errno_location"
        )
    } else {
        false
    }
}

/// Argument register family IDs for System V AMD64 ABI.
const ARG_REGS: [u8; 6] = [7, 6, 2, 1, 8, 9]; // %rdi, %rsi, %rdx, %rcx, %r8, %r9

pub(super) fn eliminate_dead_arg_moves(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() || infos[i].is_barrier() {
            i += 1;
            continue;
        }

        // Check if this is a move to an argument register.
        let dst_reg = match infos[i].kind {
            LineKind::Other { dest_reg } => dest_reg,
            _ => {
                i += 1;
                continue;
            }
        };

        if !ARG_REGS.contains(&dst_reg) {
            i += 1;
            continue;
        }

        // Scan forward for a call within a short window.
        let dst_mask = 1u16 << dst_reg;
        let mut j = i + 1;
        let scan_end = (i + 12).min(len);
        let mut found_dead = false;

        while j < scan_end {
            if infos[j].is_nop() {
                j += 1;
                continue;
            }

            match infos[j].kind {
                LineKind::Call => {
                    // Check if the call is a known 0-arg function.
                    let trimmed_j = infos[j].trimmed(store.get(j));
                    if is_zero_arg_call(trimmed_j) {
                        found_dead = true;
                    }
                    break;
                }
                // Stop at other control flow.
                LineKind::Label | LineKind::Jmp | LineKind::JmpIndirect
                | LineKind::CondJmp | LineKind::Ret => break,
                _ => {}
            }

            // If any intervening instruction reads our register, it's not dead.
            if infos[j].reg_refs & dst_mask != 0 {
                let dest_j = get_dest_reg(&infos[j]);
                if dest_j == dst_reg {
                    // Overwritten — our move is dead (but not because of the call).
                    // Leave this to the regular dead-move pass.
                    break;
                }
                // Register is read by an intervening instruction — not dead.
                break;
            }

            j += 1;
        }

        if found_dead {
            mark_nop(&mut infos[i]);
            changed = true;
        }

        i += 1;
    }

    changed
}

// ── Dead store elimination (local, windowed) ─────────────────────────────────

pub(super) fn eliminate_dead_stores(store: &LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    const WINDOW: usize = 16;

    let mut pattern_bytes = [0u8; 24];

    for i in 0..len {
        let (store_offset, store_size) = match infos[i].kind {
            LineKind::StoreRbp { offset, size, .. } => (offset, size),
            _ => continue,
        };

        let store_bytes = store_size.byte_size();

        let end = std::cmp::min(i + WINDOW, len);
        let mut slot_read = false;
        let mut slot_overwritten = false;
        let mut pattern_len: usize = 0;

        for j in (i + 1)..end {
            if infos[j].is_nop() {
                continue;
            }

            if infos[j].is_barrier() {
                slot_read = true;
                break;
            }

            if let LineKind::LoadRbp { offset: load_off, size: load_sz, .. } = infos[j].kind {
                if ranges_overlap(store_offset, store_bytes, load_off, load_sz.byte_size()) {
                    slot_read = true;
                    break;
                }
            }

            if let LineKind::StoreRbp { offset: new_off, size: new_sz, .. } = infos[j].kind {
                let new_bytes = new_sz.byte_size();
                if new_off <= store_offset && new_off + new_bytes >= store_offset + store_bytes {
                    slot_overwritten = true;
                    break;
                }
                if ranges_overlap(store_offset, store_bytes, new_off, new_bytes) {
                    slot_read = true;
                    break;
                }
            }

            // Check Other and Cmp lines for rbp references. Cmp lines can
            // have memory operands after memory fold (e.g., cmpq -N(%rbp), %rax).
            if matches!(infos[j].kind, LineKind::Other { .. } | LineKind::Cmp) {
                if infos[j].has_indirect_mem {
                    slot_read = true;
                    break;
                }
                let rbp_off = infos[j].rbp_offset;
                if rbp_off != RBP_OFFSET_NONE {
                    if rbp_off >= store_offset && rbp_off < store_offset + store_bytes {
                        slot_read = true;
                        break;
                    }
                    if rbp_off < store_offset && rbp_off + 8 > store_offset {
                        slot_read = true;
                        break;
                    }
                    continue;
                }
                if pattern_len == 0 {
                    pattern_len = write_rbp_pattern(&mut pattern_bytes, store_offset);
                }
                let pattern = std::str::from_utf8(&pattern_bytes[..pattern_len])
                    .expect("rbp pattern produced non-UTF8");
                let line = infos[j].trimmed(store.get(j));
                if line.contains(pattern) {
                    slot_read = true;
                    break;
                }
                if store_bytes > 1 {
                    let mut sub_pattern_bytes = [0u8; 24];
                    for byte_off in 1..store_bytes {
                        let check_off = store_offset + byte_off;
                        let check_len = write_rbp_pattern(&mut sub_pattern_bytes, check_off);
                        let check_pattern = std::str::from_utf8(&sub_pattern_bytes[..check_len])
                            .expect("rbp pattern produced non-UTF8");
                        let line = infos[j].trimmed(store.get(j));
                        if line.contains(check_pattern) {
                            slot_read = true;
                            break;
                        }
                    }
                    if slot_read { break; }
                }
            }
        }

        if slot_overwritten && !slot_read {
            mark_nop(&mut infos[i]);
            changed = true;
        }
    }

    changed
}

// ── Global dead store elimination for never-read stack slots ─────────────────

pub(super) fn eliminate_never_read_stores(store: &LineStore, infos: &mut [LineInfo]) {
    let len = store.len();
    if len == 0 {
        return;
    }

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }
        if !matches!(infos[i].kind, LineKind::Push { reg: 5 }) {
            i += 1;
            continue;
        }

        let mut j = next_non_nop(infos, i + 1, len);
        if j >= len {
            i = j;
            continue;
        }
        let mov_line = infos[j].trimmed(store.get(j));
        if mov_line != "movq %rsp, %rbp" {
            i = j + 1;
            continue;
        }
        j += 1;

        j = next_non_nop(infos, j, len);
        if j >= len {
            i = j;
            continue;
        }
        let subq_line = infos[j].trimmed(store.get(j));
        let is_subq = if let Some(rest) = subq_line.strip_prefix("subq $") {
            rest.strip_suffix(", %rsp").and_then(|v| v.parse::<i64>().ok()).is_some()
        } else {
            false
        };
        if !is_subq {
            i = j + 1;
            continue;
        }
        j += 1;

        // Skip callee-saved register saves
        j = next_non_nop(infos, j, len);
        let mut callee_save_end = j;
        while callee_save_end < len {
            if infos[callee_save_end].is_nop() {
                callee_save_end += 1;
                continue;
            }
            if let LineKind::StoreRbp { reg, size: MoveSize::Q, .. } = infos[callee_save_end].kind {
                if is_callee_saved_reg(reg) {
                    callee_save_end += 1;
                    continue;
                }
            }
            break;
        }
        let body_start = callee_save_end;

        // Find the end of this function
        let mut func_end = len;
        for k in body_start..len {
            if infos[k].is_nop() {
                continue;
            }
            let line = infos[k].trimmed(store.get(k));
            if line.starts_with(".size ") {
                func_end = k + 1;
                break;
            }
        }

        // Phase 1: Collect all "read" byte ranges
        let mut has_indirect = false;
        let mut read_ranges: Vec<(i32, i32)> = Vec::new();

        for k in body_start..func_end {
            if infos[k].is_nop() {
                continue;
            }

            if infos[k].has_indirect_mem {
                has_indirect = true;
                break;
            }

            match infos[k].kind {
                LineKind::StoreRbp { .. } => {}
                LineKind::LoadRbp { offset, size, .. } => {
                    read_ranges.push((offset, size.byte_size()));
                }
                LineKind::Other { .. } => {
                    let rbp_off = infos[k].rbp_offset;
                    if rbp_off != RBP_OFFSET_NONE {
                        let line = infos[k].trimmed(store.get(k));
                        if line.starts_with("leaq ") {
                            has_indirect = true;
                            break;
                        }
                        read_ranges.push((rbp_off, 32));
                    } else {
                        let line = infos[k].trimmed(store.get(k));
                        if line.contains("(%rbp)") {
                            has_indirect = true;
                            break;
                        }
                    }
                }
                LineKind::Nop | LineKind::Empty | LineKind::SelfMove
                | LineKind::Label | LineKind::Jmp | LineKind::CondJmp
                | LineKind::JmpIndirect | LineKind::Ret | LineKind::Directive => {}
                _ => {
                    let line = infos[k].trimmed(store.get(k));
                    let rbp_off = parse_rbp_offset(line);
                    if rbp_off != RBP_OFFSET_NONE {
                        read_ranges.push((rbp_off, 8));
                    } else if line.contains("(%rbp)") {
                        has_indirect = true;
                        break;
                    }
                }
            }
        }

        if has_indirect {
            i = func_end;
            continue;
        }

        // Phase 2: Eliminate stores to unread slots
        for k in body_start..func_end {
            if infos[k].is_nop() {
                continue;
            }
            if let LineKind::StoreRbp { offset, size, .. } = infos[k].kind {
                let store_bytes = size.byte_size();
                let is_read = read_ranges.iter().any(|&(r_off, r_sz)| {
                    ranges_overlap(offset, store_bytes, r_off, r_sz)
                });
                if !is_read {
                    mark_nop(&mut infos[k]);
                }
            }
        }

        i = func_end;
    }
}
