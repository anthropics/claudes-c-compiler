//! Local peephole pattern matching passes.
//!
//! Merges 7 simple local passes into a single linear scan (`combined_local_pass`)
//! to avoid redundant iteration over the lines array. Also includes
//! `fuse_movq_ext_truncation` which fuses movq + extension/truncation patterns.
//!
//! Merged passes in `combined_local_pass`:
//!   1. eliminate_redundant_movq_self: movq %reg, %reg (same src/dst)
//!   2. eliminate_reverse_move: movq %A,%B + movq %B,%A -> remove second
//!   3. eliminate_redundant_jumps: jmp to the immediately following label
//!   4. eliminate_cond_branch_inversion: jCC+jmp+label -> j!CC (inverted)
//!   5. eliminate_adjacent_store_load: store/load at same %rbp offset
//!   6. eliminate_redundant_zero_extend: redundant zero/sign extensions
//!   7. eliminate_redundant_xorl_zero: xorl %eax,%eax when %rax already zero

use super::super::types::*;
use super::helpers::is_valid_gp_reg;

pub(super) fn combined_local_pass(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    // Track whether %rax is known to be zero for redundant xorl elimination.
    // This is set to true after `xorl %eax, %eax` and stays true across
    // StoreRbp instructions (which don't modify register values), but is
    // invalidated by anything that writes %rax, or by control flow barriers.
    let mut rax_is_zero = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // --- Pattern: redundant xorl %eax, %eax elimination ---
        // When %rax is already known to be zero (from a previous xorl %eax, %eax),
        // and only StoreRbp instructions intervene (which read but don't modify
        // registers), the repeated xorl is redundant.
        //
        // Common pattern from codegen zeroing multiple local variables:
        //   xorl %eax, %eax          # sets rax = 0
        //   movq %rax, -N(%rbp)      # stores 0, rax still 0
        //   xorl %eax, %eax          # REDUNDANT
        //   movq %rax, -M(%rbp)      # stores 0, rax still 0
        if rax_is_zero {
            if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
                let trimmed = infos[i].trimmed(store.get(i));
                if trimmed == "xorl %eax, %eax" {
                    mark_nop(&mut infos[i]);
                    changed = true;
                    i += 1;
                    continue;
                }
            }
        }

        // Update rax_is_zero tracking based on current instruction.
        match infos[i].kind {
            LineKind::StoreRbp { .. } => {
                // Stores to stack don't modify registers, rax_is_zero unchanged.
            }
            LineKind::Other { dest_reg: 0 } => {
                // Something writes to %rax. Check if it's xorl %eax, %eax.
                let trimmed = infos[i].trimmed(store.get(i));
                rax_is_zero = trimmed == "xorl %eax, %eax";
            }
            LineKind::Other { dest_reg } if dest_reg != 0 => {
                // Writes to a non-rax register, rax_is_zero unchanged.
                // But check if it also reads/clobbers rax implicitly.
                // Most Other instructions only write their dest_reg.
                // Conservative: only keep rax_is_zero if the instruction
                // doesn't reference rax at all (via reg_refs).
                if infos[i].reg_refs & 1 != 0 {
                    // References rax - could be a read or write, invalidate
                    // But actually a read of rax is fine for rax_is_zero.
                    // Only a write to rax matters. Since dest_reg != 0,
                    // rax is not the destination, so it's a read - OK.
                    // Exception: instructions like div/idiv/mul/cqto that
                    // implicitly clobber rax through dest_reg rdx.
                    let trimmed = infos[i].trimmed(store.get(i));
                    if trimmed.starts_with("div") || trimmed.starts_with("idiv")
                        || trimmed.starts_with("mul") || trimmed.starts_with("imul")
                        || trimmed == "cqto" || trimmed == "cqo" || trimmed == "cdq"
                        || trimmed.starts_with("xchg") || trimmed.starts_with("cmpxchg") {
                        rax_is_zero = false;
                    }
                    // Otherwise rax is only read, not written - keep tracking.
                }
            }
            LineKind::LoadRbp { reg: 0, .. } => {
                // Load to rax - rax is no longer zero
                rax_is_zero = false;
            }
            LineKind::LoadRbp { .. } => {
                // Load to non-rax register, rax_is_zero unchanged.
            }
            LineKind::Label | LineKind::Jmp | LineKind::JmpIndirect
            | LineKind::CondJmp | LineKind::Ret | LineKind::Call => {
                // Control flow or label - invalidate tracking
                rax_is_zero = false;
            }
            LineKind::Pop { reg: 0 } | LineKind::SetCC { reg: 0 } => {
                rax_is_zero = false;
            }
            LineKind::Pop { .. } | LineKind::SetCC { .. }
            | LineKind::Push { .. } | LineKind::Cmp | LineKind::Directive => {
                // Don't affect rax
            }
            _ => {
                // Conservative: invalidate
                rax_is_zero = false;
            }
        }

        // --- Pattern: self-move elimination (movq %reg, %reg) ---
        // Pre-classified as SelfMove during classify_line, avoiding string parsing.
        if infos[i].kind == LineKind::SelfMove {
            mark_nop(&mut infos[i]);
            changed = true;
            i += 1;
            continue;
        }

        // --- Pattern: reverse-move elimination ---
        // Detects `movq %regA, %regB` followed by `movq %regB, %regA` and
        // eliminates the second mov (since %regA still holds the original value).
        //
        // Safety: We only skip NOPs and StoreRbp between the two instructions.
        // StoreRbp reads registers but never modifies any GP register value.
        // Any other instruction type causes the search to stop via `break`.
        if let LineKind::Other { dest_reg: dest_a } = infos[i].kind {
            if is_valid_gp_reg(dest_a) {
                let line_i = infos[i].trimmed(store.get(i));
                // Parse "movq %srcReg, %dstReg" pattern
                if let Some(rest) = line_i.strip_prefix("movq ") {
                    if let Some((src_str, dst_str)) = rest.split_once(',') {
                        let src = src_str.trim();
                        let dst = dst_str.trim();
                        let src_fam = register_family_fast(src);
                        let dst_fam = register_family_fast(dst);
                        // Both must be GP registers, different families, both register operands
                        if is_valid_gp_reg(src_fam) && is_valid_gp_reg(dst_fam)
                            && src_fam != dst_fam
                            && src.starts_with('%') && dst.starts_with('%')
                        {
                            // Find the next non-NOP, non-StoreRbp instruction.
                            // Limit search to 8 lines to avoid pathological scanning.
                            let mut j = i + 1;
                            let search_limit = (i + 8).min(len);
                            while j < search_limit {
                                if infos[j].is_nop() {
                                    j += 1;
                                    continue;
                                }
                                if matches!(infos[j].kind, LineKind::StoreRbp { .. }) {
                                    j += 1;
                                    continue;
                                }
                                break;
                            }
                            if j < search_limit {
                                // Check if line j is the reverse: movq %dstReg, %srcReg
                                if let LineKind::Other { dest_reg: dest_b } = infos[j].kind {
                                    if dest_b == src_fam {
                                        let line_j = infos[j].trimmed(store.get(j));
                                        if let Some(rest_j) = line_j.strip_prefix("movq ") {
                                            if let Some((src_j, dst_j)) = rest_j.split_once(',') {
                                                let src_j = src_j.trim();
                                                let dst_j = dst_j.trim();
                                                let src_j_fam = register_family_fast(src_j);
                                                let dst_j_fam = register_family_fast(dst_j);
                                                if src_j_fam == dst_fam && dst_j_fam == src_fam {
                                                    mark_nop(&mut infos[j]);
                                                    changed = true;
                                                    i += 1;
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern: redundant jump to next label ---
        if infos[i].kind == LineKind::Jmp {
            let jmp_line = infos[i].trimmed(store.get(i));
            if let Some(target) = jmp_line.strip_prefix("jmp ") {
                let target = target.trim();
                // Find the next non-NOP, non-empty line
                let mut found_redundant = false;
                for j in (i + 1)..len {
                    if infos[j].is_nop() || infos[j].kind == LineKind::Empty {
                        continue;
                    }
                    if infos[j].kind == LineKind::Label {
                        let next = infos[j].trimmed(store.get(j));
                        if let Some(label) = next.strip_suffix(':') {
                            if label == target {
                                mark_nop(&mut infos[i]);
                                changed = true;
                                found_redundant = true;
                            }
                        }
                    }
                    break;
                }
                if found_redundant {
                    i += 1;
                    continue;
                }
            }
        }

        // --- Pattern: conditional branch inversion for fall-through ---
        // Detects:
        //   jCC .Ltrue        (conditional jump)
        //   jmp .Lfalse       (unconditional jump)
        //   .Ltrue:           (label matching the conditional target)
        //
        // Transforms to:
        //   j!CC .Lfalse      (inverted condition, jump to false target)
        //   .Ltrue:           (fall through naturally)
        if infos[i].kind == LineKind::CondJmp {
            let cond_line = infos[i].trimmed(store.get(i));
            // Parse: "jCC target" -> extract CC and target
            if let Some(space_pos) = cond_line.find(' ') {
                let cc = &cond_line[1..space_pos]; // e.g., "l", "ge", "ne"
                let cond_target = cond_line[space_pos + 1..].trim();
                // Find the next non-NOP line (should be jmp)
                let mut j = i + 1;
                while j < len && infos[j].is_nop() {
                    j += 1;
                }
                if j < len && infos[j].kind == LineKind::Jmp {
                    let jmp_line = infos[j].trimmed(store.get(j));
                    if let Some(jmp_target) = jmp_line.strip_prefix("jmp ") {
                        let jmp_target = jmp_target.trim();
                        // Find the next non-NOP/non-empty line after jmp (should be a label)
                        let mut k = j + 1;
                        while k < len && (infos[k].is_nop() || infos[k].kind == LineKind::Empty) {
                            k += 1;
                        }
                        if k < len && infos[k].kind == LineKind::Label {
                            let label_line = infos[k].trimmed(store.get(k));
                            if let Some(label_name) = label_line.strip_suffix(':') {
                                if label_name == cond_target {
                                    let inv_cc = invert_cc(cc);
                                    if inv_cc != cc {
                                        let new_line = format!("    j{} {}", inv_cc, jmp_target);
                                        replace_line(store, &mut infos[i], i, new_line);
                                        mark_nop(&mut infos[j]); // Remove the jmp
                                        changed = true;
                                        i += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern: adjacent store/load at same %rbp offset ---
        if let LineKind::StoreRbp { reg: sr, offset: so, size: ss } = infos[i].kind {
            if i + 1 < len && !infos[i + 1].is_nop() {
                if let LineKind::LoadRbp { reg: lr, offset: lo, size: ls } = infos[i + 1].kind {
                    // Different register cases are handled by global_store_forwarding
                    if so == lo && ss == ls && sr == lr && sr != REG_NONE {
                        // Same register: load is redundant
                        mark_nop(&mut infos[i + 1]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // --- Pattern: redundant zero/sign extension (including cltq) ---
        // Uses pre-classified ExtKind to avoid repeated starts_with/ends_with
        // string comparisons on every iteration.
        let mut ext_idx = i + 1;
        while ext_idx < len && ext_idx < i + 10 {
            if infos[ext_idx].is_nop() {
                ext_idx += 1;
                continue;
            }
            if matches!(infos[ext_idx].kind, LineKind::StoreRbp { .. }) {
                ext_idx += 1;
                continue;
            }
            // Skip non-rax-writing instructions: these don't change %rax,
            // so an extension on %rax further ahead is still redundant.
            // This catches patterns like: movsbq (%r15), %rax; movq %rax, %r13; movsbq %al, %rax
            match infos[ext_idx].kind {
                LineKind::Other { dest_reg } if dest_reg != 0 => {
                    ext_idx += 1;
                    continue;
                }
                LineKind::LoadRbp { reg, .. } if reg != 0 => {
                    ext_idx += 1;
                    continue;
                }
                _ => break,
            }
        }

        if ext_idx < len && !infos[ext_idx].is_nop() {
            let next_ext = infos[ext_idx].ext_kind;
            let prev_ext = infos[i].ext_kind;

            let is_redundant_ext = match next_ext {
                ExtKind::MovzbqAlRax => matches!(prev_ext, ExtKind::ProducerMovzbqToRax | ExtKind::MovzbqAlRax),
                ExtKind::MovzwqAxRax => matches!(prev_ext, ExtKind::ProducerMovzwqToRax | ExtKind::MovzwqAxRax),
                ExtKind::MovsbqAlRax => matches!(prev_ext, ExtKind::ProducerMovsbqToRax | ExtKind::MovsbqAlRax),
                ExtKind::MovslqEaxRax => matches!(prev_ext, ExtKind::ProducerMovslqToRax | ExtKind::MovslqEaxRax),
                ExtKind::Cltq => matches!(prev_ext,
                    ExtKind::ProducerMovslqToRax | ExtKind::ProducerMovqConstRax |
                    ExtKind::MovslqEaxRax | ExtKind::Cltq |
                    // Zero-extend producers always produce values with bit 31 = 0,
                    // so cltq (sign-extend from 32 to 64) is a no-op after them.
                    ExtKind::ProducerMovzbToEax | ExtKind::ProducerMovzwToEax |
                    ExtKind::ProducerMovzbqToRax | ExtKind::ProducerMovzwqToRax |
                    ExtKind::MovzbqAlRax | ExtKind::MovzwqAxRax),
                ExtKind::MovlEaxEax => matches!(prev_ext,
                    ExtKind::ProducerArith32 | ExtKind::ProducerMovlToEax |
                    ExtKind::ProducerMovzbToEax | ExtKind::ProducerMovzbqToRax |
                    ExtKind::ProducerMovzwToEax | ExtKind::ProducerMovzwqToRax |
                    ExtKind::ProducerDiv32 |
                    ExtKind::MovlEaxEax),
                _ => false,
            };

            if is_redundant_ext {
                mark_nop(&mut infos[ext_idx]);
                changed = true;
                i += 1;
                continue;
            }

            // --- Extended scan: cltq past non-rax-clobbering instructions ---
            if next_ext == ExtKind::Cltq && !is_redundant_ext {
                let i_writes_rax = match infos[i].kind {
                    LineKind::Other { dest_reg } => dest_reg == 0,
                    LineKind::LoadRbp { reg, .. } => reg == 0,
                    LineKind::StoreRbp { .. } => false,
                    LineKind::Nop | LineKind::Empty => false,
                    _ => true, // conservative: barriers, calls, etc. may write rax
                };

                if !i_writes_rax && i > 0 {
                    let mut found_producer = false;
                    let scan_limit = i.saturating_sub(6);
                    let mut k = i - 1;
                    while k >= scan_limit {
                        if infos[k].is_nop() {
                            if k == 0 { break; }
                            k -= 1;
                            continue;
                        }
                        if matches!(infos[k].kind, LineKind::StoreRbp { .. }) {
                            if k == 0 { break; }
                            k -= 1;
                            continue;
                        }
                        // Stop at barriers (labels, calls, jumps, ret)
                        if infos[k].is_barrier() {
                            break;
                        }
                        // Check if this instruction is a sign-extension producer for rax
                        let k_ext = infos[k].ext_kind;
                        if matches!(k_ext,
                            ExtKind::ProducerMovslqToRax | ExtKind::ProducerMovqConstRax |
                            ExtKind::MovslqEaxRax | ExtKind::Cltq)
                        {
                            found_producer = true;
                            break;
                        }
                        // Check if this instruction writes to %rax (family 0)
                        let writes_rax = match infos[k].kind {
                            LineKind::Other { dest_reg } => dest_reg == 0,
                            LineKind::LoadRbp { reg, .. } => reg == 0,
                            _ => true, // conservative: treat unknown as writing rax
                        };
                        if writes_rax {
                            break;
                        }
                        if k == 0 { break; }
                        k -= 1;
                    }
                    if found_producer {
                        mark_nop(&mut infos[ext_idx]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

// ── Movq + extension/truncation fusion ───────────────────────────────────────
//
// Fuses `movq %REG, %rax` followed by a cast instruction into a single
// instruction. The two-instruction pattern arises from the accumulator-based
// codegen model: emit_load_operand loads a 64-bit value into %rax, then
// emit_cast_instrs emits an extension/truncation on %rax/%eax/%ax/%al.
//
// Fused patterns (all require REG != rax, no intervening non-NOP instructions):
//   movq %REG, %rax + movl %eax, %eax   -> movl %REGd, %eax    (truncate to u32)
//   movq %REG, %rax + movslq %eax, %rax -> movslq %REGd, %rax  (sign-extend i32->i64)
//   movq %REG, %rax + cltq              -> movslq %REGd, %rax   (sign-extend i32->i64)
//   movq %REG, %rax + movzbq %al, %rax  -> movzbl %REGb, %eax  (zero-extend u8->i64)
//   movq %REG, %rax + movzwq %ax, %rax  -> movzwl %REGw, %eax  (zero-extend u16->i64)
//   movq %REG, %rax + movsbq %al, %rax  -> movsbq %REGb, %rax  (sign-extend i8->i64)

pub(super) fn fuse_movq_ext_truncation(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    let mut i = 0;
    while i + 1 < len {
        // Look for ProducerMovqRegToRax or ProducerMovqMemToRax
        let is_reg_src = infos[i].ext_kind == ExtKind::ProducerMovqRegToRax;
        let is_mem_src = infos[i].ext_kind == ExtKind::ProducerMovqMemToRax;
        if !is_reg_src && !is_mem_src {
            i += 1;
            continue;
        }

        // Find next non-NOP instruction (skip only NOPs, not stores)
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        // Check if next instruction is a fusable extension/truncation on %rax
        let next_ext = infos[j].ext_kind;
        let fusable = matches!(next_ext,
            ExtKind::MovlEaxEax | ExtKind::MovslqEaxRax | ExtKind::Cltq |
            ExtKind::MovzbqAlRax | ExtKind::MovzwqAxRax |
            ExtKind::MovsbqAlRax);
        if !fusable {
            i += 1;
            continue;
        }

        let movq_line = infos[i].trimmed(store.get(i));

        if is_mem_src {
            // Memory source: movq N(%rbp), %rax + cltq -> movslq N(%rbp), %rax
            // Extract the memory operand (everything between "movq " and ", %rax")
            let mem_operand = if let Some(rest) = movq_line.strip_prefix("movq ") {
                if let Some((src, _)) = rest.rsplit_once(", %rax") {
                    Some(src.trim().to_string())
                } else { None }
            } else { None };

            if let Some(mem_op) = mem_operand {
                let new_text = match next_ext {
                    ExtKind::MovslqEaxRax | ExtKind::Cltq => {
                        // movq N(%rbp), %rax + cltq -> movslq N(%rbp), %rax
                        format!("    movslq {}, %rax", mem_op)
                    }
                    ExtKind::MovlEaxEax => {
                        // movq N(%rbp), %rax + movl %eax, %eax -> movl N(%rbp), %eax
                        format!("    movl {}, %eax", mem_op)
                    }
                    ExtKind::MovzbqAlRax => {
                        // movq N(%rbp), %rax + movzbq %al, %rax -> movzbl N(%rbp), %eax
                        format!("    movzbl {}, %eax", mem_op)
                    }
                    ExtKind::MovzwqAxRax => {
                        // movq N(%rbp), %rax + movzwq %ax, %rax -> movzwl N(%rbp), %eax
                        format!("    movzwl {}, %eax", mem_op)
                    }
                    ExtKind::MovsbqAlRax => {
                        // movq N(%rbp), %rax + movsbq %al, %rax -> movsbl N(%rbp), %eax
                        format!("    movsbq {}, %rax", mem_op)
                    }
                    _ => unreachable!(),
                };
                replace_line(store, &mut infos[i], i, new_text);
                mark_nop(&mut infos[j]);
                changed = true;
                i = j + 1;
                continue;
            }
            i += 1;
            continue;
        }

        // Register source: extract source register family from the movq instruction
        let src_family = if let Some(rest) = movq_line.strip_prefix("movq ") {
            if let Some((src, _dst)) = rest.split_once(',') {
                let src = src.trim();
                let fam = register_family_fast(src);
                if fam != REG_NONE && fam != 0 { fam } else { REG_NONE }
            } else { REG_NONE }
        } else { REG_NONE };

        if src_family == REG_NONE {
            i += 1;
            continue;
        }

        // Build the fused instruction based on the extension type
        let new_text = match next_ext {
            ExtKind::MovlEaxEax => {
                let src_32 = REG_NAMES[1][src_family as usize];
                format!("    movl {}, %eax", src_32)
            }
            ExtKind::MovslqEaxRax | ExtKind::Cltq => {
                let src_32 = REG_NAMES[1][src_family as usize];
                format!("    movslq {}, %rax", src_32)
            }
            ExtKind::MovzbqAlRax => {
                let src_8 = REG_NAMES[3][src_family as usize];
                format!("    movzbl {}, %eax", src_8)
            }
            ExtKind::MovzwqAxRax => {
                let src_16 = REG_NAMES[2][src_family as usize];
                format!("    movzwl {}, %eax", src_16)
            }
            ExtKind::MovsbqAlRax => {
                let src_8 = REG_NAMES[3][src_family as usize];
                format!("    movsbq {}, %rax", src_8)
            }
            _ => unreachable!("mov+ext fusion matched unexpected ExtKind"),
        };

        replace_line(store, &mut infos[i], i, new_text);
        mark_nop(&mut infos[j]);
        changed = true;
        i = j + 1;
        continue;
    }
    changed
}

// ── XMM-through-accumulator folding ──────────────────────────────────────────
//
// Folds `movq %xmm0, %rax` + `movq %rax, <dest>` into `movq %xmm0, <dest>`.
// The accumulator-based codegen routes FP values through %rax when storing
// double/float results to stack slots or callee-saved registers. This pattern
// is safe because `movq %xmm0, <gp_reg>` and `movq %xmm0, <memory>` are both
// valid x86-64 instructions (SSE2 MOVQ encoding).
//
// Also handles `movd %xmm0, %eax` + `movl %eax, <dest>` → `movd %xmm0, <dest>`.

/// Fold `movq %xmm0, %rax; movq %rax, <dest>` into `movq %xmm0, <dest>`.
///
/// The accumulator-based codegen routes floating-point values through %rax,
/// producing two-move chains. This fold eliminates the intermediate step.
///
/// Safety: The fold removes the definition of %rax. We must verify that %rax
/// is dead after the second move (not read before being overwritten or before
/// a control flow boundary).
pub(super) fn fold_xmm_through_accumulator(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    const RAX: u8 = 0; // register family 0 = rax/eax/ax/al

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        let trimmed_i = infos[i].trimmed(store.get(i));

        // Match `movq %xmm0, %rax`
        if trimmed_i != "movq %xmm0, %rax" {
            i += 1;
            continue;
        }

        // Find next non-NOP instruction
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        let trimmed_j = infos[j].trimmed(store.get(j));

        // Match `movq %rax, <dest>` where dest is a register (NOT memory).
        // We must NOT fold to memory destinations because it creates a
        // cross-domain store forwarding stall: an XMM store to a stack slot
        // followed by a GP load from the same slot incurs ~10-15 cycle penalty.
        if let Some(dest) = trimmed_j.strip_prefix("movq %rax, ") {
            let dest = dest.trim();
            if dest == "%rax" || dest.contains('(') {
                i += 1;
                continue;
            }

            // Check that %rax is dead after line j.
            // Scan forward from j+1: if %rax is referenced before being
            // purely overwritten (or before a control flow boundary), the fold
            // is unsafe.
            if !is_reg_dead_after(infos, store, j + 1, len, RAX) {
                i += 1;
                continue;
            }

            let new_text = format!("    movq %xmm0, {}", dest);
            replace_line(store, &mut infos[i], i, new_text);
            mark_nop(&mut infos[j]);
            changed = true;
            i = j + 1;
            continue;
        }

        i += 1;
    }
    changed
}

/// Check if a register is dead (not read before being overwritten) starting
/// from position `start`. Scans at most 16 instructions forward and gives up
/// conservatively (returns false) at control flow boundaries.
fn is_reg_dead_after(infos: &[LineInfo], store: &LineStore, start: usize, len: usize, reg: u8) -> bool {
    is_reg_dead_scan(infos, store, start, len, reg, 16, 0, None)
}

/// Extended liveness check that can look past conditional jumps, unconditional
/// jumps, and fallthrough-only labels by following control flow. The `depth`
/// parameter limits recursion through branches/jumps.
fn is_reg_dead_after_ext(
    infos: &[LineInfo], store: &LineStore, start: usize, len: usize, reg: u8,
    targets: &super::helpers::JumpTargets,
) -> bool {
    is_reg_dead_scan(infos, store, start, len, reg, 24, 5, Some(targets))
}

/// Call-safe variant of is_reg_dead_after_ext for dead move elimination.
/// Treats Calls as barriers (returns false) instead of assuming calls kill
/// registers. This prevents incorrectly eliminating moves that set up
/// function call arguments.
pub(super) fn is_reg_unused_after_ext(
    infos: &[LineInfo], store: &LineStore, start: usize, len: usize, reg: u8,
    targets: &super::helpers::JumpTargets,
) -> bool {
    is_reg_dead_scan_call_safe(infos, store, start, len, reg, 24, 5, Some(targets))
}

/// Like is_reg_dead_scan but treats Calls as barriers (returns false).
fn is_reg_dead_scan_call_safe(
    infos: &[LineInfo], store: &LineStore, start: usize, len: usize,
    reg: u8, max_scan: usize, depth: usize,
    targets: Option<&super::helpers::JumpTargets>,
) -> bool {
    let reg_bit = 1u16 << reg;
    let mut scanned = 0;
    let mut k = start;
    while k < len && scanned < max_scan {
        if infos[k].is_nop() {
            k += 1;
            continue;
        }

        match infos[k].kind {
            LineKind::Label => {
                if let Some(tgt) = targets {
                    let label_text = infos[k].trimmed(store.get(k));
                    let is_jump_target = if let Some(n) = super::helpers::parse_label_number(label_text) {
                        (n as usize) < tgt.is_jump_target.len() && tgt.is_jump_target[n as usize]
                    } else {
                        tgt.has_non_numeric_jump_targets
                    };
                    if !is_jump_target {
                        k += 1;
                        continue;
                    }
                    if depth > 0 {
                        let dead_here = is_reg_dead_scan_call_safe(
                            infos, store, k + 1, len, reg, 16, depth - 1, targets
                        );
                        if dead_here {
                            k += 1;
                            continue;
                        }
                    }
                }
                return false;
            }
            LineKind::Jmp => {
                if depth > 0 {
                    let trimmed = infos[k].trimmed(store.get(k));
                    if let Some(target_label) = super::helpers::extract_jump_target(trimmed) {
                        if let Some(tp) = find_label_pos(infos, store, len, target_label) {
                            return is_reg_dead_scan_call_safe(
                                infos, store, tp, len, reg, 16, depth - 1, targets
                            );
                        }
                    }
                }
                return false;
            }
            LineKind::JmpIndirect => return false,
            LineKind::CondJmp => {
                if depth == 0 {
                    return false;
                }
                let trimmed = infos[k].trimmed(store.get(k));
                let target = super::helpers::extract_jump_target(trimmed);

                // Fallthrough doesn't consume depth
                let fall_dead = is_reg_dead_scan_call_safe(
                    infos, store, k + 1, len, reg, 16, depth, targets
                );
                if !fall_dead {
                    return false;
                }

                if let Some(target_label) = target {
                    if let Some(tp) = find_label_pos(infos, store, len, target_label) {
                        return is_reg_dead_scan_call_safe(
                            infos, store, tp, len, reg, 16, depth - 1, targets
                        );
                    }
                }
                return false;
            }
            // Conservative: treat calls as barriers — the register might be
            // read as a function argument.
            LineKind::Call => return false,
            LineKind::Ret => return reg != 0,
            _ => {}
        }

        let refs_reg = infos[k].reg_refs & reg_bit != 0;
        if refs_reg {
            let dest = super::helpers::get_dest_reg(&infos[k]);
            if dest == reg {
                let trimmed = infos[k].trimmed(store.get(k));
                if trimmed.starts_with("movq ") || trimmed.starts_with("movl ")
                    || trimmed.starts_with("movb ") || trimmed.starts_with("movw ")
                    || trimmed.starts_with("movabs")
                    || trimmed.starts_with("xorl %eax, %eax")
                    || trimmed.starts_with("movzbl ") || trimmed.starts_with("movzbq ")
                    || trimmed.starts_with("movzwl ") || trimmed.starts_with("movzwq ")
                    || trimmed.starts_with("movslq ") || trimmed.starts_with("movsbq ")
                    || trimmed.starts_with("movsbl ")
                {
                    return true;
                }
                if trimmed.starts_with("leaq ") || trimmed.starts_with("leal ") {
                    if let Some(comma_pos) = trimmed.rfind(", ") {
                        let src_part = &trimmed[..comma_pos];
                        let mut reg_in_src = false;
                        for size_idx in 0..4 {
                            let name = REG_NAMES[size_idx][reg as usize];
                            if src_part.contains(name) {
                                reg_in_src = true;
                                break;
                            }
                        }
                        if !reg_in_src {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        if reg == 0 {
            let trimmed = infos[k].trimmed(store.get(k));
            if super::helpers::has_implicit_reg_usage(trimmed) {
                return false;
            }
        }

        scanned += 1;
        k += 1;
    }

    false
}

/// Core liveness scan with depth-limited cross-block analysis.
/// `depth` controls how many control flow boundaries (CondJmp, Jmp, jump-target
/// Labels) the scan can look past. depth=0 is the basic local-only scan.
fn is_reg_dead_scan(
    infos: &[LineInfo], store: &LineStore, start: usize, len: usize,
    reg: u8, max_scan: usize, depth: usize,
    targets: Option<&super::helpers::JumpTargets>,
) -> bool {
    let reg_bit = 1u16 << reg;
    let mut scanned = 0;
    let mut k = start;
    while k < len && scanned < max_scan {
        if infos[k].is_nop() {
            k += 1;
            continue;
        }

        // Control flow boundary handling
        match infos[k].kind {
            LineKind::Label => {
                if let Some(tgt) = targets {
                    let label_text = infos[k].trimmed(store.get(k));
                    let is_jump_target = if let Some(n) = super::helpers::parse_label_number(label_text) {
                        (n as usize) < tgt.is_jump_target.len() && tgt.is_jump_target[n as usize]
                    } else {
                        tgt.has_non_numeric_jump_targets
                    };
                    if !is_jump_target {
                        // Fallthrough-only label — safe to scan past
                        k += 1;
                        continue;
                    }
                    // Jump-target label: code here may be reached from multiple paths.
                    // Check if register is dead starting from here (using a sub-scan).
                    if depth > 0 {
                        let dead_here = is_reg_dead_scan(
                            infos, store, k + 1, len, reg, 16, depth - 1, targets
                        );
                        if dead_here {
                            k += 1;
                            continue;
                        }
                    }
                }
                return false;
            }
            LineKind::Jmp => {
                if depth > 0 {
                    // Follow the unconditional jump to its target.
                    let trimmed = infos[k].trimmed(store.get(k));
                    if let Some(target_label) = super::helpers::extract_jump_target(trimmed) {
                        if let Some(tp) = find_label_pos(infos, store, len, target_label) {
                            return is_reg_dead_scan(
                                infos, store, tp, len, reg, 16, depth - 1, targets
                            );
                        }
                    }
                }
                return false;
            }
            LineKind::JmpIndirect => return false,
            LineKind::CondJmp => {
                if depth == 0 {
                    return false;
                }
                // Check both paths: fallthrough and jump target.
                let trimmed = infos[k].trimmed(store.get(k));
                let target = super::helpers::extract_jump_target(trimmed);

                // Fallthrough path — doesn't consume depth since it's the
                // natural code continuation (not a new code path).
                let fall_dead = is_reg_dead_scan(
                    infos, store, k + 1, len, reg, 16, depth, targets
                );
                if !fall_dead {
                    return false;
                }

                // Jump target path — consumes depth (new code path)
                if let Some(target_label) = target {
                    if let Some(tp) = find_label_pos(infos, store, len, target_label) {
                        return is_reg_dead_scan(
                            infos, store, tp, len, reg, 16, depth - 1, targets
                        );
                    }
                }
                return false; // couldn't find target — conservative
            }
            LineKind::Call => {
                return reg != 4 && reg != 5 && !super::helpers::is_callee_saved_reg(reg);
            }
            // At ret, %rax (reg=0) is live — it holds the function return value.
            // All other GP registers are dead at the function return.
            LineKind::Ret => return reg != 0,
            _ => {}
        }

        let refs_reg = infos[k].reg_refs & reg_bit != 0;
        if refs_reg {
            // This line references the register. Check if it's a pure overwrite
            // (writes the reg without reading it).
            let dest = super::helpers::get_dest_reg(&infos[k]);
            if dest == reg {
                // dest_reg == our reg. But read-modify-write instructions
                // (addq %rax, ...; subq ..., %rax) also read it.
                // If it's a simple mov/lea with reg as dest only, it's a pure overwrite.
                let trimmed = infos[k].trimmed(store.get(k));
                if trimmed.starts_with("movq ") || trimmed.starts_with("movl ")
                    || trimmed.starts_with("movb ") || trimmed.starts_with("movw ")
                    || trimmed.starts_with("movabs")
                    || trimmed.starts_with("xorl %eax, %eax")
                    || trimmed.starts_with("movzbl ") || trimmed.starts_with("movzbq ")
                    || trimmed.starts_with("movzwl ") || trimmed.starts_with("movzwq ")
                    || trimmed.starts_with("movslq ") || trimmed.starts_with("movsbq ")
                    || trimmed.starts_with("movsbl ")
                {
                    // Pure overwrite — reg is dead here
                    return true;
                }
                // leaq/leal: pure overwrite ONLY if dest reg doesn't appear in src operand.
                // e.g. `leaq A(%rip), %rax` is pure overwrite (rax not in src),
                // but  `leaq 8(%rax), %rax` is read-modify-write (rax IS in src).
                if trimmed.starts_with("leaq ") || trimmed.starts_with("leal ") {
                    if let Some(comma_pos) = trimmed.rfind(", ") {
                        let src_part = &trimmed[..comma_pos];
                        // Check if any name variant of the dest register appears in src
                        let mut reg_in_src = false;
                        for size_idx in 0..4 {
                            let name = REG_NAMES[size_idx][reg as usize];
                            if src_part.contains(name) {
                                reg_in_src = true;
                                break;
                            }
                        }
                        if !reg_in_src {
                            return true; // Pure overwrite
                        }
                    }
                    // dest reg appears in src → read-modify-write, fall through to return false
                }
            }
            // Referenced but not a pure overwrite → reg is read, fold unsafe
            return false;
        }

        // Implicit rax usage by div/mul/cltq etc. that reg_refs might miss
        if reg == 0 {
            let trimmed = infos[k].trimmed(store.get(k));
            if super::helpers::has_implicit_reg_usage(trimmed) {
                return false;
            }
        }

        scanned += 1;
        k += 1;
    }

    // Reached scan limit without finding a definitive answer — conservatively unsafe
    false
}

/// Find the position of a label definition (the instruction after the label line).
/// Returns the index of the first non-NOP instruction after the label.
fn find_label_pos(infos: &[LineInfo], store: &LineStore, len: usize, target: &str) -> Option<usize> {
    for idx in 0..len {
        if infos[idx].kind == LineKind::Label {
            let label_text = infos[idx].trimmed(store.get(idx));
            // Label text includes colon, e.g. ".LBB3:"
            if label_text.len() > 1
                && label_text.ends_with(':')
                && &label_text[..label_text.len() - 1] == target
            {
                // Return position after the label
                let mut pos = idx + 1;
                while pos < len && infos[pos].is_nop() {
                    pos += 1;
                }
                return Some(pos);
            }
        }
    }
    None
}

// ── 64-bit → 32-bit operation narrowing ─────────────────────────────────────
//
// Narrows 64-bit operations to 32-bit equivalents when the upper 32 bits are
// provably zero. On x86-64, 32-bit register operations implicitly zero-extend
// the upper 32 bits of the 64-bit register, so narrowing is always safe when:
//
//   1. `andq $imm, %reg` where 0 <= imm <= 0x7FFFFFFF → `andl $imm, %regd`
//      The AND result fits in 32 bits since the immediate limits the output range.
//
//   2. `testq %reg, %reg` after a 32-bit operation → `testl %regd, %regd`
//      The value is already zero-extended, so 64-bit test is equivalent to 32-bit.
//
//   3. `movslq %regd, %rax` after a 32-bit operation that zero-extends →
//      eliminate entirely. The 32-bit op already zero-extended bit 31=0, so
//      movslq (sign-extend from 32 to 64) is a no-op.
//
// These patterns arise from CCC's accumulator-based codegen which emits 64-bit
// instructions even when 32-bit would suffice (the C type system doesn't propagate
// down to instruction selection). The strprocess benchmark's count_words hot loop
// has exactly this pattern: `andq $8192, %rdi; movslq %edi, %rax; testq %rax, %rax`.

pub(super) fn narrow_64_to_32(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // --- Pattern 1: andq $imm, %reg → andl $imm, %regd ---
        // Safe when: immediate is non-negative and fits in 32 bits (0..=0x7FFFFFFF).
        // After andl, the result is zero-extended to 64 bits automatically.
        if let LineKind::Other { dest_reg } = infos[i].kind {
            if is_valid_gp_reg(dest_reg) {
                let trimmed = infos[i].trimmed(store.get(i));
                if let Some(rest) = trimmed.strip_prefix("andq $") {
                    if let Some((imm_str, reg_str)) = rest.split_once(',') {
                        let imm_str = imm_str.trim();
                        let reg_str = reg_str.trim();
                        // Parse the immediate value
                        if let Ok(imm) = imm_str.parse::<i64>() {
                            // Safe to narrow if immediate is in range [0, 0x7FFFFFFF].
                            // Negative immediates or values > 2^31-1 need 64-bit AND.
                            if imm >= 0 && imm <= 0x7FFFFFFF {
                                let reg_fam = register_family_fast(reg_str);
                                if is_valid_gp_reg(reg_fam) {
                                    let reg32 = REG_NAMES[1][reg_fam as usize];
                                    let new_line = format!("    andl ${}, {}", imm, reg32);
                                    replace_line(store, &mut infos[i], i, new_line);
                                    changed = true;
                                    // After andl, the dest register is known to be zero-extended.
                                    // Check the next instruction for further narrowing opportunities.
                                    narrow_after_32bit_op(store, infos, i, len, dest_reg, &mut changed);
                                    i += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern 2: movslq %regd, %reg → eliminate/narrow ---
        // Self-extension (movslq %eax, %rax): eliminate when source is known 32-bit.
        // Cross-register (movslq %edi, %rax): narrow to movl %edi, %eax.
        if let LineKind::Other { dest_reg } = infos[i].kind {
            if is_valid_gp_reg(dest_reg) {
                let trimmed = infos[i].trimmed(store.get(i));
                if let Some(rest) = trimmed.strip_prefix("movslq ") {
                    if let Some((src, dst)) = rest.split_once(',') {
                        let src = src.trim();
                        let dst = dst.trim();
                        let src_fam = register_family_fast(src);
                        let dst_fam = register_family_fast(dst);
                        if is_valid_gp_reg(src_fam) && is_valid_gp_reg(dst_fam)
                            && is_known_32bit_value(infos, store, i, src_fam)
                        {
                            if src_fam == dst_fam {
                                // Self-extension: eliminate entirely
                                mark_nop(&mut infos[i]);
                                changed = true;
                                i += 1;
                                continue;
                            } else {
                                // Cross-register: narrow to movl
                                let src_32 = REG_NAMES[1][src_fam as usize];
                                let dst_32 = REG_NAMES[1][dst_fam as usize];
                                let new_line = format!("    movl {}, {}", src_32, dst_32);
                                replace_line(store, &mut infos[i], i, new_line);
                                changed = true;
                                i += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern 3: testq %reg, %reg → testl %regd, %regd ---
        // testq is classified as LineKind::Cmp, so we check separately.
        // Safe when the value in %reg is known to be zero-extended (i.e.,
        // produced by a 32-bit operation). Look backward for a producer.
        if infos[i].kind == LineKind::Cmp {
            let trimmed = infos[i].trimmed(store.get(i));
            if let Some(rest) = trimmed.strip_prefix("testq ") {
                if let Some((src, dst)) = rest.split_once(',') {
                    let src = src.trim();
                    let dst = dst.trim();
                    // Only handle testq %reg, %reg (same register)
                    if src == dst && src.starts_with('%') {
                        let reg_fam = register_family_fast(src);
                        if is_valid_gp_reg(reg_fam) {
                            // Scan backward to find if the value is 32-bit
                            if is_known_32bit_value(infos, store, i, reg_fam) {
                                let reg32 = REG_NAMES[1][reg_fam as usize];
                                let new_line = format!("    testl {}, {}", reg32, reg32);
                                replace_line(store, &mut infos[i], i, new_line);
                                changed = true;
                                i += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

/// After rewriting a 64-bit op to 32-bit (e.g., andq→andl), check if the next
/// instruction is a movslq or testq that can be narrowed/eliminated.
fn narrow_after_32bit_op(
    store: &mut LineStore,
    infos: &mut [LineInfo],
    producer_idx: usize,
    len: usize,
    producer_reg: u8,
    changed: &mut bool,
) {
    // Find next non-NOP instruction
    let mut j = producer_idx + 1;
    while j < len && infos[j].is_nop() {
        j += 1;
    }
    if j >= len {
        return;
    }

    let trimmed_j = infos[j].trimmed(store.get(j));

    if trimmed_j.starts_with("movslq ") {
        if let Some(rest) = trimmed_j.strip_prefix("movslq ") {
            if let Some((src, dst)) = rest.split_once(',') {
                let src = src.trim();
                let dst = dst.trim();
                let src_fam = register_family_fast(src);
                let dst_fam = register_family_fast(dst);
                if src_fam == producer_reg {
                    if src_fam == dst_fam {
                        // Self-extension (e.g., movslq %eax, %rax) after a 32-bit op.
                        // Since the 32-bit op zero-extends (bit 31 = 0 for positive
                        // results like AND with a positive mask), movslq is a no-op.
                        mark_nop(&mut infos[j]);
                        *changed = true;
                    } else if is_valid_gp_reg(dst_fam) {
                        // Cross-register movslq (e.g., movslq %edi, %rax).
                        // Since the source is known to be zero-extended (from the 32-bit
                        // op), sign-extend = zero-extend, so movslq is equivalent to
                        // movl (which is a shorter encoding and also zero-extends).
                        let src_32 = REG_NAMES[1][src_fam as usize];
                        let dst_32 = REG_NAMES[1][dst_fam as usize];
                        let new_line = format!("    movl {}, {}", src_32, dst_32);
                        replace_line(store, &mut infos[j], j, new_line);
                        *changed = true;
                    }
                }
            }
        }
    }
}

/// Check if a register's value at position `pos` is known to be 32-bit
/// (upper 32 bits are zero). Scans backward looking for a 32-bit producer.
fn is_known_32bit_value(infos: &[LineInfo], store: &LineStore, pos: usize, reg: u8) -> bool {
    if pos == 0 {
        return false;
    }
    let reg_bit = 1u16 << reg;
    let mut scanned = 0;
    let mut k = pos - 1;
    loop {
        if scanned >= 12 {
            return false;
        }
        if infos[k].is_nop() {
            if k == 0 { return false; }
            k -= 1;
            continue;
        }

        // Stop at control flow boundaries
        if infos[k].is_barrier() {
            return false;
        }

        // Skip stores — they don't modify registers
        if matches!(infos[k].kind, LineKind::StoreRbp { .. }) {
            if k == 0 { return false; }
            k -= 1;
            scanned += 1;
            continue;
        }

        // Check if this instruction writes to our register
        let dest = super::helpers::get_dest_reg(&infos[k]);
        if dest == reg {
            let trimmed = infos[k].trimmed(store.get(k));
            // 32-bit arithmetic operations: andl, addl, subl, orl, xorl, etc.
            // These all zero-extend the result to 64 bits.
            if trimmed.starts_with("andl ") || trimmed.starts_with("addl ")
                || trimmed.starts_with("subl ") || trimmed.starts_with("orl ")
                || trimmed.starts_with("xorl ") || trimmed.starts_with("shll ")
                || trimmed.starts_with("shrl ") || trimmed.starts_with("sarl ")
                || trimmed.starts_with("imull ")
            {
                return true;
            }
            // 32-bit moves: movl, movzbl, movzwl
            if trimmed.starts_with("movl ") || trimmed.starts_with("movzbl ")
                || trimmed.starts_with("movzwl ")
            {
                return true;
            }
            // movslq to this register also produces a 64-bit value but the upper
            // bits may be set — so it's NOT a 32-bit producer in general.
            // However, if the source is positive (e.g., after andl with positive mask),
            // it would be. We conservatively say no.
            return false;
        }

        // Check if this instruction modifies a different register (skip past it)
        if infos[k].reg_refs & reg_bit != 0 {
            // References our register but doesn't write it — it reads it.
            // We can't determine the value from here; give up.
            // Actually: if the instruction reads our reg but writes a different reg,
            // we can skip past it. Only stop if it modifies our reg.
            // The `dest != reg` check above already handled the write case.
            // But implicit writes (div, cltq) could also modify our reg:
            if reg == 0 {
                if super::helpers::has_implicit_reg_usage(infos[k].trimmed(store.get(k))) {
                    return false;
                }
            }
        }

        scanned += 1;
        if k == 0 { return false; }
        k -= 1;
    }
}

// ── Address-through-secondary register folding ──────────────────────────────
//
// Folds `movq %rN, %rcx; <mem-op> (%rcx), ...` into `<mem-op> (%rN), ...`
// and NOP's the movq. The accumulator-based codegen routes all pointer
// dereferences through %rcx (the secondary register), producing two-instruction
// chains where a single instruction suffices.
//
// Handles both loads and stores through (%rcx), including displacement forms
// like `N(%rcx)`:
//   movq %r15, %rcx; movsbq (%rcx), %rax  → movsbq (%r15), %rax
//   movq %rax, %rcx; movq (%rcx), %rax    → movq (%rax), %rax
//   movq %r15, %rcx; movb %dl, (%rcx)     → movb %dl, (%r15)
//   movq %r14, %rcx; leaq 8(%rcx), %rax   → leaq 8(%r14), %rax
//
// Safety: the fold removes the definition of %rcx. We verify that %rcx is
// dead after the memory operation (not read before being overwritten).
// We also verify %rcx is not used as a register operand (outside parentheses)
// in the memory instruction.

pub(super) fn fold_address_through_secondary(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    // Build jump target map to distinguish fallthrough-only labels from real targets.
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // Match: movq %rSrc, %rDst (any GP register pair, excluding rsp/rbp)
        let (src_fam, dst_fam) = match infos[i].kind {
            LineKind::Other { dest_reg } if is_valid_gp_reg(dest_reg)
                && dest_reg != 4 && dest_reg != 5 =>
            {
                let trimmed_i = infos[i].trimmed(store.get(i));
                match super::helpers::parse_reg_to_reg_movq(&infos[i], trimmed_i) {
                    Some((s, d)) => (s, d),
                    None => { i += 1; continue; }
                }
            }
            _ => { i += 1; continue; }
        };

        let src_reg_name: &str = REG_NAMES[0][src_fam as usize]; // &'static str
        let dst_reg_name: &str = REG_NAMES[0][dst_fam as usize]; // &'static str

        // Find next non-NOP instruction
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        let trimmed_j = infos[j].trimmed(store.get(j));

        // Check that the instruction uses %rDst as a memory base
        // (contains "%rDst)" as a substring — covers (%rDst), N(%rDst), etc.)
        let mem_pattern = format!("{})", dst_reg_name);
        if !trimmed_j.contains(&mem_pattern) {
            i += 1;
            continue;
        }

        // Trial replacement: replace %rDst) with %rSrc) in the instruction text.
        let new_instr = trimmed_j.replace(&mem_pattern, &format!("{})", src_reg_name));

        // Verify %rDst doesn't appear elsewhere (as a non-memory register operand).
        let has_other_ref = (0..4).any(|size_idx| {
            let name = REG_NAMES[size_idx][dst_fam as usize];
            new_instr.contains(name)
        });
        if has_other_ref {
            i += 1;
            continue;
        }

        // Check that %rDst is dead after the memory instruction.
        if !is_reg_dead_after_ext(infos, store, j + 1, len, dst_fam, &targets) {
            i += 1;
            continue;
        }

        // Safe to fold: NOP the movq, rewrite the memory instruction
        mark_nop(&mut infos[i]);
        let new_text = format!("    {}", new_instr);
        replace_line(store, &mut infos[j], j, new_text);
        changed = true;
        i = j + 1;
    }
    changed
}

// ── Double-register add to LEA fold ─────────────────────────────────────────
//
// When a register is copied and then added to itself, producing a multiply-by-2,
// the movq + addq can be replaced with a single LEA:
//
//   movq %rA, %rB; addq %rA, %rB → leaq (%rA, %rA), %rB
//
// LEA doesn't set flags, so this is only safe when the addq's flags are dead
// (overwritten before being consumed).

pub(super) fn fold_double_to_leaq(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // Match: movq %rA, %rB
        let (src_a, dst_b) = match infos[i].kind {
            LineKind::Other { dest_reg } if is_valid_gp_reg(dest_reg)
                && dest_reg != 4 && dest_reg != 5 =>
            {
                let trimmed = infos[i].trimmed(store.get(i));
                match super::helpers::parse_reg_to_reg_movq(&infos[i], trimmed) {
                    Some((s, d)) => (s, d),
                    None => { i += 1; continue; }
                }
            }
            _ => { i += 1; continue; }
        };

        // Find next non-NOP instruction
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len { i += 1; continue; }

        // Match: addq %rA, %rB (same source register, same destination)
        if !matches!(infos[j].kind, LineKind::Other { dest_reg } if dest_reg == dst_b) {
            i += 1;
            continue;
        }

        let trimmed_j = infos[j].trimmed(store.get(j));
        let addq_match = if let Some(rest) = trimmed_j.strip_prefix("addq ") {
            if let Some((src, dst)) = rest.split_once(", ") {
                let src = src.trim();
                let dst = dst.trim();
                register_family_fast(src) == src_a && register_family_fast(dst) == dst_b
            } else { false }
        } else { false };

        if !addq_match {
            i += 1;
            continue;
        }

        // Check: flags from addq are dead (will be overwritten before use)
        if !are_flags_dead_after(infos, store, j + 1, len, None) {
            i += 1;
            continue;
        }

        // Build replacement LEA
        let src_name = REG_NAMES[0][src_a as usize];
        let dst_name = REG_NAMES[0][dst_b as usize];
        let new_text = format!("    leaq ({}, {}), {}", src_name, src_name, dst_name);

        mark_nop(&mut infos[i]);
        replace_line(store, &mut infos[j], j, new_text);
        changed = true;
        i = j + 1;
    }

    changed
}

/// Check if CPU flags are dead (will be overwritten before being read) starting
/// from position `start`. Scans forward looking for the next flag-relevant
/// instruction and returns true if it sets (rather than reads) flags.
/// When `targets` is provided, non-jump-target labels are safely skipped.
fn are_flags_dead_after(
    infos: &[LineInfo], store: &LineStore, start: usize, len: usize,
    _targets: Option<&super::helpers::JumpTargets>,
) -> bool {
    let scan_end = (start + 24).min(len);
    let mut k = start;
    while k < scan_end {
        if infos[k].is_nop() {
            k += 1;
            continue;
        }

        // Cmp/test always sets flags → previous flags dead
        if infos[k].kind == LineKind::Cmp {
            return true;
        }

        // Labels: always skip. Flag liveness is forward-only: "does the code
        // from here forward consume flags before setting new ones?" This question
        // is the same regardless of which path reached this label. Both jump-target
        // and fallthrough-only labels are safe to scan past.
        if infos[k].kind == LineKind::Label {
            k += 1;
            continue;
        }

        // Other control flow barriers → conservative
        if infos[k].is_barrier() {
            return false;
        }

        // CondJmp and SetCC consume flags
        if infos[k].kind == LineKind::CondJmp { return false; }
        if matches!(infos[k].kind, LineKind::SetCC { .. }) { return false; }

        let trimmed = infos[k].trimmed(store.get(k));

        // cmov reads flags
        if trimmed.starts_with("cmov") { return false; }
        // adc/sbb read carry flag
        if trimmed.starts_with("adc") || trimmed.starts_with("sbb") { return false; }

        // Flag-preserving instructions: mov, lea, push, pop → continue
        let b = trimmed.as_bytes();
        if b.len() >= 3 {
            if (b[0] == b'm' && b[1] == b'o' && b[2] == b'v')
                || (b[0] == b'l' && b[1] == b'e' && b[2] == b'a')
            {
                k += 1;
                continue;
            }
        }
        if trimmed.starts_with("pushq ") || trimmed.starts_with("popq ") {
            k += 1;
            continue;
        }

        // Any other instruction (add, sub, and, or, xor, shl, shr, etc.)
        // likely sets flags → previous flags dead
        return true;
    }
    false // conservative: couldn't determine
}

// ── movq + addq $imm → leaq fold ────────────────────────────────────────────
//
// When a register is copied and then an immediate is added to the copy,
// the movq + addq can be replaced with a single LEA (which doesn't set flags):
//
//   movq %rA, %rB
//   addq $IMM, %rB     →   leaq IMM(%rA), %rB
//
// Also handles subq:
//   movq %rA, %rB
//   subq $IMM, %rB     →   leaq -IMM(%rA), %rB

pub(super) fn fold_movq_addimm_to_leaq(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match: movq %rA, %rB
        let (src_a, dst_b) = match infos[i].kind {
            LineKind::Other { dest_reg } if is_valid_gp_reg(dest_reg)
                && dest_reg != 4 && dest_reg != 5 =>
            {
                let trimmed = infos[i].trimmed(store.get(i));
                match super::helpers::parse_reg_to_reg_movq(&infos[i], trimmed) {
                    Some((s, d)) => (s, d),
                    None => { i += 1; continue; }
                }
            }
            _ => { i += 1; continue; }
        };

        // Find next non-NOP
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len { i += 1; continue; }

        // Match: addq $IMM, %rB or subq $IMM, %rB (same destination as movq)
        if !matches!(infos[j].kind, LineKind::Other { dest_reg } if dest_reg == dst_b) {
            i += 1; continue;
        }

        let trimmed_j = infos[j].trimmed(store.get(j));
        let imm_val: Option<i64> = if let Some(rest) = trimmed_j.strip_prefix("addq $") {
            if let Some((imm_s, dst_s)) = rest.split_once(", ") {
                let dst_s = dst_s.trim();
                if register_family_fast(dst_s) == dst_b {
                    imm_s.trim().parse::<i64>().ok()
                } else { None }
            } else { None }
        } else if let Some(rest) = trimmed_j.strip_prefix("subq $") {
            if let Some((imm_s, dst_s)) = rest.split_once(", ") {
                let dst_s = dst_s.trim();
                if register_family_fast(dst_s) == dst_b {
                    imm_s.trim().parse::<i64>().ok().map(|v| -v)
                } else { None }
            } else { None }
        } else { None };

        let imm = match imm_val {
            Some(v) => v,
            None => { i += 1; continue; }
        };

        // Check: flags from addq/subq are dead
        if !are_flags_dead_after(infos, store, j + 1, len, Some(&targets)) {
            i += 1; continue;
        }

        // Build replacement LEA
        let src_name = REG_NAMES[0][src_a as usize];
        let dst_name = REG_NAMES[0][dst_b as usize];
        let new_text = format!("    leaq {}({}), {}", imm, src_name, dst_name);

        mark_nop(&mut infos[i]);
        replace_line(store, &mut infos[j], j, new_text);
        changed = true;
        i = j + 1;
    }

    changed
}

// ── Scaled address into memory operand fold ─────────────────────────────────
//
// Folds address computation chains into x86 addressing modes:
//
// Pattern 1 (3-instruction, saves 2):
//   leaq (%rA, %rA), %rT    ; rT = rA * 2
//   addq %rT, %rB           ; rB = rB + rA*2
//   <mem_op> (%rB), %rC     ; load/store using rB
//   →
//   <mem_op> (%rB, %rA, 2), %rC  ; fold scaled address into operand
//
// Pattern 2 (2-instruction, saves 1):
//   addq %rT, %rB           ; rB = rB + rT
//   <mem_op> (%rB), %rC     ; load/store using rB
//   →
//   <mem_op> (%rB, %rT), %rC     ; fold simple address into operand
//
// Conditions: modified rB dead after mem_op, addq flags dead, temps dead.

pub(super) fn fold_scaled_address_into_load(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() { i += 1; continue; }

        // Match: addq %rSrc, %rDst (register-to-register, src != dst)
        let trimmed_i = infos[i].trimmed(store.get(i));
        let (add_src, add_dst) = match parse_addq_reg_reg(trimmed_i) {
            Some((s, d)) => (s, d),
            None => { i += 1; continue; }
        };

        // Find next non-NOP instruction
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len { i += 1; continue; }

        // Must not be a barrier, label, or jump
        if infos[j].is_barrier() { i += 1; continue; }
        match infos[j].kind {
            LineKind::Label | LineKind::Jmp | LineKind::CondJmp | LineKind::Call => {
                i += 1; continue;
            }
            _ => {}
        }

        // Check: instruction j uses (%rB) as a simple memory base (no index/scale)
        let trimmed_j_owned = infos[j].trimmed(store.get(j)).to_string();
        let trimmed_j = trimmed_j_owned.as_str();
        let base_name = REG_NAMES[0][add_dst as usize];
        let base_pattern = format!("({})", base_name);
        if !trimmed_j.contains(&base_pattern) { i += 1; continue; }

        // Safety: if mem_op writes to add_dst (the base register), it must be
        // a pure overwrite (mov/lea) — NOT a read-modify-write like addq.
        let mem_dest = super::helpers::get_dest_reg(&infos[j]);
        if mem_dest == add_dst {
            if !trimmed_j.starts_with("mov") && !trimmed_j.starts_with("lea") {
                i += 1; continue;
            }
        }

        // Check: flags from addq are dead (not consumed before overwritten)
        if !are_flags_dead_after(infos, store, j, len, Some(&targets)) { i += 1; continue; }

        // Check: modified rB (= rB_orig + rT) is dead after mem_op
        let modified_rb_dead = if mem_dest == add_dst {
            true // mem_op overwrites rB
        } else {
            is_reg_dead_after_ext(infos, store, j + 1, len, add_dst, &targets)
        };
        if !modified_rb_dead { i += 1; continue; }

        // Try 3-instruction fold: look back for leaq (%rA, %rA), %rT
        let mut did_three_fold = false;
        if i > 0 {
            let mut pi = i.saturating_sub(1);
            while pi > 0 && infos[pi].is_nop() { pi -= 1; }
            if !infos[pi].is_nop() {
                let trimmed_pi = infos[pi].trimmed(store.get(pi));
                if let Some((reg_a, dst_t)) = parse_leaq_double(trimmed_pi) {
                    // leaq wrote to dst_t, which must be the addq source
                    // reg_a must not be add_dst (since addq modified it)
                    if dst_t == add_src && reg_a != add_dst && reg_a != add_src {
                        // Check: add_src (rT) is safe to eliminate
                        let add_src_safe = mem_dest == add_src
                            || is_reg_dead_after_ext(
                                infos, store, j + 1, len, add_src, &targets,
                            );
                        if add_src_safe {
                            let index_name = REG_NAMES[0][reg_a as usize];
                            let new_mem = format!("({}, {}, 2)", base_name, index_name);
                            let new_text = format!(
                                "    {}", trimmed_j.replace(&base_pattern, &new_mem)
                            );
                            mark_nop(&mut infos[pi]); // NOP leaq
                            mark_nop(&mut infos[i]);  // NOP addq
                            replace_line(store, &mut infos[j], j, new_text);
                            changed = true;
                            did_three_fold = true;
                            i = j + 1;
                        }
                    }
                }
            }
        }

        if did_three_fold { continue; }

        // 2-instruction fold: addq + mem → mem with index register
        let index_name = REG_NAMES[0][add_src as usize];
        let new_mem = format!("({}, {})", base_name, index_name);
        let new_text = format!("    {}", trimmed_j.replace(&base_pattern, &new_mem));
        mark_nop(&mut infos[i]); // NOP addq
        replace_line(store, &mut infos[j], j, new_text);
        changed = true;
        i = j + 1;
    }
    changed
}

/// Parse `leaq (%rA, %rA), %rT` where both registers in parens are the same.
/// Returns Some((rA_family, rT_family)).
fn parse_leaq_double(trimmed: &str) -> Option<(RegId, RegId)> {
    let rest = trimmed.strip_prefix("leaq (")?;
    let (inner, after) = rest.split_once(')')?;
    let after = after.strip_prefix(", ")?;
    let dst = after.trim();
    let dst_fam = register_family_fast(dst);
    if dst_fam == REG_NONE || dst_fam > REG_GP_MAX || dst_fam == 4 || dst_fam == 5 {
        return None;
    }
    let (reg1, reg2) = inner.split_once(", ")?;
    let reg1 = reg1.trim();
    let reg2 = reg2.trim();
    let fam1 = register_family_fast(reg1);
    let fam2 = register_family_fast(reg2);
    if fam1 == REG_NONE || fam1 > REG_GP_MAX || fam1 != fam2 {
        return None;
    }
    if fam1 == 4 || fam1 == 5 { return None; }
    Some((fam1, dst_fam))
}

/// Parse `addq %rSrc, %rDst` → Some((src_family, dst_family)).
/// Both must be GP registers, not rsp/rbp, and src != dst.
fn parse_addq_reg_reg(trimmed: &str) -> Option<(RegId, RegId)> {
    let rest = trimmed.strip_prefix("addq ")?;
    let (src, dst) = rest.split_once(", ")?;
    let src = src.trim();
    let dst = dst.trim();
    if !src.starts_with('%') || !dst.starts_with('%') {
        return None;
    }
    let sfam = register_family_fast(src);
    let dfam = register_family_fast(dst);
    if sfam == REG_NONE || sfam > REG_GP_MAX || sfam == 4 || sfam == 5 {
        return None;
    }
    if dfam == REG_NONE || dfam > REG_GP_MAX || dfam == 4 || dfam == 5 {
        return None;
    }
    if sfam == dfam { return None; }
    Some((sfam, dfam))
}

// ── Commutative binop through temp fold ─────────────────────────────────────
//
// When a commutative binary operation uses a temporary register to swap
// operands, the entire save/overwrite/binop sequence can be replaced with
// a single binop using the original operands:
//
//   movq %rA, %rT       ; save rA in temp
//   movq %rB, %rA       ; overwrite rA with rB
//   addq %rT, %rA       ; rA = rB + rA_orig = rA_orig + rB (commutative)
//   →
//   addq %rB, %rA       ; rA = rA + rB (same result, when %rT dead)

pub(super) fn fold_commutative_through_temp(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i + 2 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // Match: movq %rA, %rT
        let (src_a, temp_reg) = match infos[i].kind {
            LineKind::Other { dest_reg } if is_valid_gp_reg(dest_reg)
                && dest_reg != 4 && dest_reg != 5 =>
            {
                let trimmed = infos[i].trimmed(store.get(i));
                match super::helpers::parse_reg_to_reg_movq(&infos[i], trimmed) {
                    Some((s, d)) => (s, d),
                    None => { i += 1; continue; }
                }
            }
            _ => { i += 1; continue; }
        };

        // Find next two non-NOP instructions
        let mut j = i + 1;
        while j < len && infos[j].is_nop() { j += 1; }
        if j >= len { i += 1; continue; }

        let mut k = j + 1;
        while k < len && infos[k].is_nop() { k += 1; }
        if k >= len { i += 1; continue; }

        // Match: movq %rB, %rA (overwrite the source of the first movq)
        let src_b = match infos[j].kind {
            LineKind::Other { dest_reg } if dest_reg == src_a => {
                let trimmed = infos[j].trimmed(store.get(j));
                match super::helpers::parse_reg_to_reg_movq(&infos[j], trimmed) {
                    Some((sb, _da)) if sb != src_a && sb != temp_reg => sb,
                    _ => { i += 1; continue; }
                }
            }
            _ => { i += 1; continue; }
        };

        // Match: commutative binop %rT, %rA at position k
        if !matches!(infos[k].kind, LineKind::Other { dest_reg } if dest_reg == src_a) {
            i += 1;
            continue;
        }

        let trimmed_k = infos[k].trimmed(store.get(k));
        let (op, op_src, op_dst) = match parse_binop_reg_reg(trimmed_k) {
            Some(v) => v,
            None => { i += 1; continue; }
        };

        if op_src != temp_reg || op_dst != src_a {
            i += 1;
            continue;
        }

        if !is_commutative_op(op) {
            i += 1;
            continue;
        }

        // Check: %rT is dead after the binop
        if !is_reg_dead_after_ext(infos, store, k + 1, len, temp_reg, &targets) {
            i += 1;
            continue;
        }

        // Build replacement: replace temp_reg family with src_b family in the binop
        let new_instr = super::helpers::replace_reg_family(trimmed_k, temp_reg, src_b);
        let new_text = format!("    {}", new_instr);

        mark_nop(&mut infos[i]);
        mark_nop(&mut infos[j]);
        replace_line(store, &mut infos[k], k, new_text);
        changed = true;
        i = k + 1;
    }

    changed
}

/// Parse `<opcode> %src, %dst` binary operation with two register operands.
fn parse_binop_reg_reg(trimmed: &str) -> Option<(&str, RegId, RegId)> {
    let space_pos = trimmed.find(' ')?;
    let opcode = &trimmed[..space_pos];
    let rest = &trimmed[space_pos + 1..];

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
    Some((opcode, sfam, dfam))
}

/// Check if a binary operation is commutative (a op b = b op a).
fn is_commutative_op(op: &str) -> bool {
    matches!(op, "addq" | "addl" | "addw"
        | "orq" | "orl" | "orw"
        | "xorq" | "xorl" | "xorw"
        | "andq" | "andl" | "andw"
        | "imulq" | "imull")
}

// ── Accumulator routing fold ────────────────────────────────────────────────
//
// CCC routes most values through %rax (the accumulator), producing two-movq
// chains where a single instruction suffices:
//
//   movq $1, %rax; movq %rax, %r10   → movq $1, %r10
//   movq %rbx, %rax; movq %rax, %r11 → movq %rbx, %r11
//
// The pattern matches any `movq <src>, %rT; movq %rT, %rN` where:
// - <src> is an immediate ($N) or register (%reg)
// - %rT is dead after the second movq
// - %rN is a different GP register from %rT
//
// This is a local two-instruction fold; the global copy propagation pass
// handles wider chains but can't fold across control flow barriers.

pub(super) fn fold_accumulator_routing(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // Match: movq <src>, %rT (where src is $imm or %reg, not memory)
        if let LineKind::Other { dest_reg: temp_reg } = infos[i].kind {
            if !is_valid_gp_reg(temp_reg) || temp_reg == 4 || temp_reg == 5 {
                i += 1;
                continue;
            }

            let trimmed_i = infos[i].trimmed(store.get(i));
            let rest = match trimmed_i.strip_prefix("movq ") {
                Some(r) => r,
                None => { i += 1; continue; }
            };
            let (src_str, dst_str) = match rest.split_once(", ") {
                Some(pair) => pair,
                None => { i += 1; continue; }
            };

            let temp_64 = REG_NAMES[0][temp_reg as usize];
            if dst_str != temp_64 {
                i += 1;
                continue;
            }

            // Source must be $imm or %reg (not memory — no parentheses)
            let is_imm = src_str.starts_with('$');
            let is_reg = src_str.starts_with('%') && !src_str.contains('(');
            if !is_imm && !is_reg {
                i += 1;
                continue;
            }

            // If source is a register, it must not be the same as temp
            if is_reg {
                let src_fam = register_family_fast(src_str);
                if src_fam == temp_reg {
                    i += 1;
                    continue;
                }
            }

            // Find next non-NOP instruction
            let mut j = i + 1;
            while j < len && infos[j].is_nop() {
                j += 1;
            }
            if j >= len {
                i += 1;
                continue;
            }

            // Match: movq %rT, %rN
            let trimmed_j = infos[j].trimmed(store.get(j));
            let expected_prefix = format!("movq {}, ", temp_64);
            if let Some(dest_str) = trimmed_j.strip_prefix(expected_prefix.as_str()) {
                let dest_str = dest_str.trim();
                if !dest_str.starts_with('%') || dest_str.contains('(') {
                    i += 1;
                    continue;
                }
                let dest_fam = register_family_fast(dest_str);
                if !is_valid_gp_reg(dest_fam) || dest_fam == temp_reg
                    || dest_fam == 4 || dest_fam == 5
                {
                    i += 1;
                    continue;
                }

                // Check temp_reg dead after j (extended: cross-block)
                if is_reg_dead_after_ext(infos, store, j + 1, len, temp_reg, &targets) {
                    // Fold: NOP instruction i, rewrite j as movq <src>, <dest>
                    let new_text = format!("    movq {}, {}", src_str, dest_str);
                    mark_nop(&mut infos[i]);
                    replace_line(store, &mut infos[j], j, new_text);
                    changed = true;
                    i = j + 1;
                    continue;
                }
            }
        }

        i += 1;
    }
    changed
}

// ── Load destination redirect ──────────────────────────────────────────────
//
// CCC routes loads through %rax (the accumulator) before copying to the
// real destination, producing patterns like:
//
//   movsbq (%r15), %rax      ; load to temp
//   movq %rax, %r13          ; copy temp to dest
//   testq %rax, %rax         ; use temp (optional)
//
// When %rax (temp) is dead after the use(s), we can redirect the load
// directly to %r13 and rewrite later uses:
//
//   movsbq (%r15), %r13
//   testq %r13, %r13
//
// This saves 1 instruction per occurrence.

pub(super) fn redirect_load_destination(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // Match: <load> ..., %rT  where load is from memory (contains parentheses)
        // Load instructions: movq, movl, movsbq, movsbl, movslq, movzbl, movzbq, movzwl, movzwq
        if let LineKind::Other { dest_reg: temp_reg } = infos[i].kind {
            if !is_valid_gp_reg(temp_reg) || temp_reg == 4 || temp_reg == 5 {
                i += 1;
                continue;
            }

            let trimmed_i = infos[i].trimmed(store.get(i)).to_string();

            // Must be a load instruction (has memory operand with parentheses)
            let is_load = trimmed_i.contains('(') && (
                trimmed_i.starts_with("movq ") || trimmed_i.starts_with("movl ") ||
                trimmed_i.starts_with("movsbq ") || trimmed_i.starts_with("movsbl ") ||
                trimmed_i.starts_with("movslq ") || trimmed_i.starts_with("movzbl ") ||
                trimmed_i.starts_with("movzbq ") || trimmed_i.starts_with("movzwl ") ||
                trimmed_i.starts_with("movzwq ") || trimmed_i.starts_with("movb ") ||
                trimmed_i.starts_with("movw ")
            );
            if !is_load {
                i += 1;
                continue;
            }

            let temp_64 = REG_NAMES[0][temp_reg as usize];

            // The temp register must NOT appear in the source operand (address computation).
            // e.g., movq (%rax), %rax  — can't redirect because rax is used in the address.
            if let Some(comma_pos) = trimmed_i.rfind(", ") {
                let src_part = &trimmed_i[..comma_pos];
                let mut temp_in_src = false;
                for size_idx in 0..4 {
                    if src_part.contains(REG_NAMES[size_idx][temp_reg as usize]) {
                        temp_in_src = true;
                        break;
                    }
                }
                if temp_in_src {
                    i += 1;
                    continue;
                }
            } else {
                i += 1;
                continue;
            }

            // Find next non-NOP: should be movq %rT, %rN
            let mut j = i + 1;
            while j < len && infos[j].is_nop() {
                j += 1;
            }
            if j >= len {
                i += 1;
                continue;
            }

            let trimmed_j = infos[j].trimmed(store.get(j));
            let expected_prefix = format!("movq {}, ", temp_64);
            let dest_fam;
            if let Some(dest_str) = trimmed_j.strip_prefix(expected_prefix.as_str()) {
                let dest_str = dest_str.trim();
                if !dest_str.starts_with('%') || dest_str.contains('(') {
                    i += 1;
                    continue;
                }
                dest_fam = register_family_fast(dest_str);
                if !is_valid_gp_reg(dest_fam) || dest_fam == temp_reg
                    || dest_fam == 4 || dest_fam == 5
                {
                    i += 1;
                    continue;
                }
            } else {
                i += 1;
                continue;
            }

            // Now check if there are 0-2 more uses of temp_reg between j+1 and the
            // point where it's dead. We collect these uses and rewrite them.
            // We scan forward from j+1 looking for:
            //  - uses of temp_reg that we can rewrite to dest_fam
            //  - the point where temp_reg is overwritten or dead
            // We limit to 3 additional uses max for safety.
            let mut uses: Vec<usize> = Vec::new();
            let mut scan_ok = true;
            let mut k = j + 1;
            let scan_limit = (j + 8).min(len);
            while k < scan_limit {
                if infos[k].is_nop() {
                    k += 1;
                    continue;
                }
                // Stop at control flow barriers
                match infos[k].kind {
                    LineKind::Label | LineKind::Jmp | LineKind::JmpIndirect |
                    LineKind::CondJmp | LineKind::Call | LineKind::Ret => break,
                    _ => {}
                }

                let refs_temp = infos[k].reg_refs & (1u16 << temp_reg) != 0;
                let refs_dest = infos[k].reg_refs & (1u16 << dest_fam) != 0;

                if refs_temp {
                    // Check if this instruction writes dest_fam — conflict
                    if refs_dest {
                        // Both temp and dest referenced — not safe to redirect
                        scan_ok = false;
                        break;
                    }

                    let dest_k = super::helpers::get_dest_reg(&infos[k]);
                    if dest_k == temp_reg {
                        // Temp is overwritten here — done scanning
                        break;
                    }

                    // temp_reg is used (read) — we can potentially rewrite
                    if uses.len() >= 3 {
                        scan_ok = false;
                        break;
                    }
                    // Make sure the instruction doesn't have implicit reg usage
                    let trimmed_k = infos[k].trimmed(store.get(k));
                    if super::helpers::has_implicit_reg_usage(trimmed_k) {
                        scan_ok = false;
                        break;
                    }
                    uses.push(k);
                    k += 1;
                    continue;
                }

                // Check if this instruction writes dest_fam — conflict
                let dest_k = super::helpers::get_dest_reg(&infos[k]);
                if dest_k == dest_fam {
                    // dest is overwritten before temp is dead — can't redirect
                    scan_ok = false;
                    break;
                }

                k += 1;
            }

            if !scan_ok {
                i += 1;
                continue;
            }

            // Check that temp_reg is dead after all the uses we found
            let check_pos = if uses.is_empty() { j + 1 } else { uses[uses.len() - 1] + 1 };
            if !is_reg_dead_after_ext(infos, store, check_pos, len, temp_reg, &targets) {
                i += 1;
                continue;
            }

            // Also check that dest_fam is NOT read between j+1 and the last use
            // (since we're moving its definition earlier)
            if !uses.is_empty() {
                let mut dest_conflict = false;
                let mut m = j + 1;
                while m <= uses[uses.len() - 1] {
                    if infos[m].is_nop() {
                        m += 1;
                        continue;
                    }
                    if infos[m].reg_refs & (1u16 << dest_fam) != 0 {
                        // Check if this is one of our use-sites that we're rewriting
                        if !uses.contains(&m) {
                            dest_conflict = true;
                            break;
                        }
                    }
                    m += 1;
                }
                if dest_conflict {
                    i += 1;
                    continue;
                }
            }

            // Safe to redirect. Rewrite:
            // 1. Load instruction: change destination from temp to dest
            let new_load = super::helpers::replace_reg_family(&trimmed_i, temp_reg, dest_fam);
            let new_load = format!("    {}", new_load);
            replace_line(store, &mut infos[i], i, new_load);

            // 2. NOP the movq %rT, %rN
            mark_nop(&mut infos[j]);

            // 3. Rewrite uses of temp_reg to dest_fam
            for &u in &uses {
                let trimmed_u = infos[u].trimmed(store.get(u)).to_string();
                let new_text = super::helpers::replace_reg_family(&trimmed_u, temp_reg, dest_fam);
                let new_text = format!("    {}", new_text);
                replace_line(store, &mut infos[u], u, new_text);
            }

            changed = true;
            i = if uses.is_empty() { j + 1 } else { uses[uses.len() - 1] + 1 };
            continue;
        }

        i += 1;
    }
    changed
}

// ── Increment-in-place fold ─────────────────────────────────────────────────
//
// CCC's codegen produces three-instruction sequences to modify a value in a
// register, routing through a temporary:
//
//   movq %r15, %rsi; addq $1, %rsi; movq %rsi, %r15  → addq $1, %r15
//   movq %rbx, %rsi; subq $1, %rsi; movq %rsi, %rbx  → subq $1, %rbx
//
// This fold replaces the 3-instruction pattern with 1, eliminating 2 instructions.
// Safety: the temporary register (%rsi in the examples) must be dead after the
// third instruction.

/// Match `addq/subq $imm, %rT` and return the prefix and immediate string.
fn match_arith_imm_reg<'a>(trimmed: &'a str, reg_64: &str) -> Option<(&'a str, &'a str)> {
    for prefix in &["addq $", "subq $"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if let Some((imm, dst)) = rest.split_once(", ") {
                if dst.trim() == reg_64 {
                    return Some((prefix, imm));
                }
            }
        }
    }
    None
}

pub(super) fn fold_increment_in_place(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    let targets = super::helpers::collect_jump_targets(store, infos, len);

    let mut i = 0;
    while i + 2 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // Match: movq %rN, %rT (reg-to-reg copy)
        let trimmed_i = infos[i].trimmed(store.get(i));
        let (src_fam, tmp_fam) = match super::helpers::parse_reg_to_reg_movq(&infos[i], trimmed_i) {
            Some(pair) => pair,
            None => { i += 1; continue; }
        };

        let src_64 = REG_NAMES[0][src_fam as usize];
        let tmp_64 = REG_NAMES[0][tmp_fam as usize];

        // Find next non-NOP: should be addq/subq $imm, %rT
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        let trimmed_j = infos[j].trimmed(store.get(j));

        // Match addq/subq $imm, %rT
        let op_match = match_arith_imm_reg(trimmed_j, tmp_64);
        let (op_prefix, imm_str) = match op_match {
            Some(pair) => pair,
            None => { i += 1; continue; }
        };

        // Find next non-NOP: should be movq %rT, %rN
        let mut k = j + 1;
        while k < len && infos[k].is_nop() {
            k += 1;
        }
        if k >= len {
            i += 1;
            continue;
        }

        let trimmed_k = infos[k].trimmed(store.get(k));
        let expected = format!("movq {}, {}", tmp_64, src_64);
        if trimmed_k != expected {
            i += 1;
            continue;
        }

        // Check tmp_reg dead after k (extended: cross-block)
        if !is_reg_dead_after_ext(infos, store, k + 1, len, tmp_fam, &targets) {
            i += 1;
            continue;
        }

        // Fold: NOP first two, rewrite third as op $imm, %rN
        let new_text = format!("    {}{}, {}", op_prefix, imm_str, src_64);
        mark_nop(&mut infos[i]);
        mark_nop(&mut infos[j]);
        replace_line(store, &mut infos[k], k, new_text);
        changed = true;
        i = k + 1;
    }
    changed
}

