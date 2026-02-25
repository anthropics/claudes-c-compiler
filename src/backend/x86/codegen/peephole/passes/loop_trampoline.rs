//! Loop trampoline elimination pass.
//!
//! SSA codegen creates "trampoline" blocks for loop back-edges to resolve phi
//! nodes. Instead of updating loop variables in-place, it creates new SSA values
//! in fresh registers and uses a separate block to shuffle them back:
//!
//!   .LOOP:
//!       ; ... loop body using %r9, %r10, %r11 ...
//!       movq %r9, %r14           ; copy old dest to new reg
//!       addq $320, %r14          ; modify new dest
//!       movq %r10, %r15          ; copy old frac to new reg
//!       addl %r8d, %r15d         ; modify new frac
//!       ; ... loop condition ...
//!       jne .TRAMPOLINE
//!   .TRAMPOLINE:
//!       movq %r14, %r9           ; shuffle new dest back
//!       movq %r15, %r10          ; shuffle new frac back
//!       jmp .LOOP
//!
//! This pass detects trampoline blocks and coalesces the register copies:
//!   1. For each trampoline copy %src -> %dst, find where %src was created in
//!      the predecessor (as a copy from %dst followed by modifications).
//!   2. Rewrite those modifications to target %dst directly.
//!   3. NOP the initial copy and the trampoline copy.
//!   4. Redirect the branch directly to the loop header.

use super::super::types::*;
use super::helpers::*;

/// Look up a register family from a name string that does NOT have the '%' prefix.
/// This avoids allocating a `format!("%{}", name)` just to call `register_family_fast`.
/// Intentionally duplicates the core lookup logic from `register_family_fast` in types.rs
/// to avoid the allocation overhead on this hot path.
#[inline]
fn register_family_no_prefix(name: &str) -> RegId {
    let b = name.as_bytes();
    let len = b.len();
    if len < 2 {
        return REG_NONE;
    }
    // Dispatch on first character (same logic as register_family_fast but without '%' prefix)
    match b[0] {
        b'r' | b'e' => {
            if len < 3 {
                // len==2: only r8, r9 are valid
                return if b[0] == b'r' { reg_digit_to_id(b[1]) } else { REG_NONE };
            }
            match (b[1], b[2]) {
                (b'a', b'x') => 0,  // rax / eax
                (b'c', b'x') => 1,  // rcx / ecx
                (b'd', b'x') => 2,  // rdx / edx
                (b'd', b'i') => 7,  // rdi / edi
                (b'b', b'x') => 3,  // rbx / ebx
                (b'b', b'p') => 5,  // rbp / ebp
                (b's', b'p') => 4,  // rsp / esp
                (b's', b'i') => 6,  // rsi / esi
                (b'8', _)    => 8,  // r8d / r8w / r8b
                (b'9', _)    => 9,  // r9d / r9w / r9b
                (b'1', b'0') => 10, (b'1', b'1') => 11, (b'1', b'2') => 12,
                (b'1', b'3') => 13, (b'1', b'4') => 14, (b'1', b'5') => 15,
                _ => REG_NONE,
            }
        }
        // 16-bit / 8-bit short forms: ax, al, ah, cx, cl, etc.
        b'a' => if matches!(b[1], b'x' | b'l' | b'h') { 0 } else { REG_NONE },
        b'c' => if matches!(b[1], b'x' | b'l' | b'h') { 1 } else { REG_NONE },
        b'd' => match b[1] { b'i' => 7, b'x' | b'l' | b'h' => 2, _ => REG_NONE },
        b'b' => match b[1] { b'p' => 5, b'x' | b'l' | b'h' => 3, _ => REG_NONE },
        b's' => match b[1] { b'p' => 4, b'i' => 6, _ => REG_NONE },
        _ => REG_NONE,
    }
}

/// Check if trimmed instruction matches "movq <first_reg>, <second_reg>" exactly.
/// `first_reg` and `second_reg` should include the '%' prefix (e.g., "%rax").
/// Avoids allocating a format!() string for comparison.
#[inline]
fn is_movq_reg_reg(trimmed: &str, first_reg: &str, second_reg: &str) -> bool {
    // Expected: "movq %rXX, %rYY" (regs include '%' prefix from REG_NAMES)
    let b = trimmed.as_bytes();
    let expected_len = 5 + first_reg.len() + 2 + second_reg.len(); // "movq " + first + ", " + second
    if b.len() != expected_len {
        return false;
    }
    trimmed.starts_with("movq ")
        && trimmed[5..].starts_with(first_reg)
        && trimmed[5 + first_reg.len()..].starts_with(", ")
        && trimmed[5 + first_reg.len() + 2..] == *second_reg
}

/// Check if trimmed instruction matches "movslq <first_reg>, <second_reg>" exactly.
#[inline]
fn is_movslq_reg_reg(trimmed: &str, first_reg: &str, second_reg: &str) -> bool {
    let expected_len = 7 + first_reg.len() + 2 + second_reg.len(); // "movslq " + first + ", " + second
    if trimmed.len() != expected_len {
        return false;
    }
    trimmed.starts_with("movslq ")
        && trimmed[7..].starts_with(first_reg)
        && trimmed[7 + first_reg.len()..].starts_with(", ")
        && trimmed[7 + first_reg.len() + 2..] == *second_reg
}

pub(super) fn eliminate_loop_trampolines(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    if len < 4 {
        return false;
    }

    let mut changed = false;

    // Build a map of label_name -> line_index for all labels.
    let mut label_positions: Vec<(u32, usize)> = Vec::new();
    let mut max_label_num: u32 = 0;

    for i in 0..len {
        if infos[i].is_nop() { continue; }
        if infos[i].kind == LineKind::Label {
            let trimmed = infos[i].trimmed(store.get(i));
            if let Some(n) = parse_label_number(trimmed) {
                label_positions.push((n, i));
                if n > max_label_num { max_label_num = n; }
            }
        }
    }

    if label_positions.is_empty() {
        return false;
    }

    let table_size = (max_label_num + 1) as usize;

    // Build label_num -> line_index lookup
    let mut label_line: Vec<usize> = vec![usize::MAX; table_size];
    for &(num, idx) in &label_positions {
        label_line[num as usize] = idx;
    }

    // Count branch references to each label AND build reverse index from
    // label_num -> first conditional branch line targeting it.
    // This eliminates the O(n) scan per trampoline candidate that was the
    // dominant bottleneck (previously ~20% of total compile time on large files).
    let mut label_branch_count: Vec<u32> = vec![0; table_size];
    let mut cond_branch_for_label: Vec<usize> = vec![usize::MAX; table_size];

    for i in 0..len {
        if infos[i].is_nop() { continue; }
        match infos[i].kind {
            LineKind::Jmp | LineKind::CondJmp => {
                let trimmed = infos[i].trimmed(store.get(i));
                if let Some(target) = extract_jump_target(trimmed) {
                    if let Some(n) = parse_dotl_number(target) {
                        if (n as usize) < table_size {
                            label_branch_count[n as usize] += 1;
                            // Record the first conditional branch targeting this label
                            if infos[i].kind == LineKind::CondJmp
                                && cond_branch_for_label[n as usize] == usize::MAX
                            {
                                cond_branch_for_label[n as usize] = i;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Find trampoline blocks
    for &(tramp_num, tramp_label_idx) in &label_positions {
        if label_branch_count[tramp_num as usize] != 1 {
            continue;
        }

        // Parse the trampoline block contents
        let mut tramp_moves: Vec<(RegId, RegId)> = Vec::new();
        let mut tramp_jmp_target: Option<u32> = None;
        let mut has_stack_load = false;
        let mut tramp_stack_loads: Vec<(i32, RegId, usize, usize)> = Vec::new();
        let mut tramp_all_lines: Vec<usize> = Vec::new();

        let mut j = tramp_label_idx + 1;
        while j < len {
            if infos[j].is_nop() || infos[j].kind == LineKind::Empty {
                j += 1;
                continue;
            }
            let trimmed = infos[j].trimmed(store.get(j));

            // Check for movq %regA, %regB
            if let Some(rest) = trimmed.strip_prefix("movq %") {
                if let Some((src_str, dst_str)) = rest.split_once(", %") {
                    let src_fam = register_family_no_prefix(src_str);
                    let dst_fam = register_family_no_prefix(dst_str.trim());
                    if src_fam != REG_NONE && dst_fam != REG_NONE && src_fam != dst_fam {
                        tramp_moves.push((src_fam, dst_fam));
                        tramp_all_lines.push(j);
                        j += 1;
                        continue;
                    }
                }
            }

            // Check for movslq %regA, %regB
            if let Some(rest) = trimmed.strip_prefix("movslq %") {
                if let Some((src_str, dst_str)) = rest.split_once(", %") {
                    let src_fam = register_family_no_prefix(src_str);
                    let dst_fam = register_family_no_prefix(dst_str.trim());
                    if src_fam != REG_NONE && dst_fam != REG_NONE && src_fam != dst_fam {
                        tramp_moves.push((src_fam, dst_fam));
                        tramp_all_lines.push(j);
                        j += 1;
                        continue;
                    }
                }
            }

            // Check for stack load pattern: movq -N(%rbp), %rax
            if let LineKind::LoadRbp { reg: 0, offset, .. } = infos[j].kind {
                let mut k = j + 1;
                while k < len && (infos[k].is_nop() || infos[k].kind == LineKind::Empty) {
                    k += 1;
                }
                if k < len {
                    let next_trimmed = infos[k].trimmed(store.get(k));
                    if let Some(rest) = next_trimmed.strip_prefix("movq %rax, %") {
                        let dst_fam = register_family_no_prefix(rest.trim());
                        if dst_fam != REG_NONE && dst_fam != 0 {
                            has_stack_load = true;
                            tramp_stack_loads.push((offset, dst_fam, j, k));
                            tramp_all_lines.push(j);
                            tramp_all_lines.push(k);
                            j = k + 1;
                            continue;
                        }
                    }
                }
                break;
            }

            // Check for jmp .LBB_N (final instruction)
            if infos[j].kind == LineKind::Jmp {
                if let Some(target) = extract_jump_target(trimmed) {
                    if let Some(n) = parse_dotl_number(target) {
                        tramp_jmp_target = Some(n);
                        tramp_all_lines.push(j);
                    }
                }
                break;
            }

            break;
        }

        let target_num = match tramp_jmp_target {
            Some(n) => n,
            None => continue,
        };

        if tramp_moves.is_empty() && !has_stack_load {
            continue;
        }

        // Find the conditional branch that targets this trampoline using
        // the pre-built reverse index (O(1) instead of O(n) scan).
        let branch_idx = cond_branch_for_label[tramp_num as usize];
        if branch_idx == usize::MAX {
            continue;
        }

        // Per-move coalescing
        let mut move_coalesced: Vec<bool> = Vec::with_capacity(tramp_moves.len());
        let mut coalesce_actions: Vec<(usize, RegId, RegId)> = Vec::new();
        let mut copy_nop_lines: Vec<usize> = Vec::new();
        for &(src_fam, dst_fam) in &tramp_moves {
            let src_64 = REG_NAMES[0][src_fam as usize];
            let dst_64 = REG_NAMES[0][dst_fam as usize];

            let mut copy_idx = None;
            let mut modifications: Vec<usize> = Vec::new();
            let mut scan_ok = true;

            let mut k = branch_idx;
            while k > 0 {
                k -= 1;
                if infos[k].is_nop() || infos[k].kind == LineKind::Empty {
                    continue;
                }
                if infos[k].kind == LineKind::Label {
                    break;
                }
                if matches!(infos[k].kind, LineKind::Call | LineKind::Jmp |
                    LineKind::JmpIndirect | LineKind::Ret) {
                    break;
                }

                let trimmed = infos[k].trimmed(store.get(k));

                let modifies_src = match infos[k].kind {
                    LineKind::Other { dest_reg } => dest_reg == src_fam,
                    LineKind::StoreRbp { .. } => false,
                    LineKind::LoadRbp { reg, .. } => reg == src_fam,
                    LineKind::SetCC { reg } => {
                        // SetCC is a partial write (only 1 byte) — it cannot be
                        // safely rewritten to a different register family because
                        // it does not clear the upper bytes. Bail out of coalescing
                        // entirely when SetCC modifies src_fam.
                        if reg == src_fam {
                            scan_ok = false;
                            break;
                        }
                        false
                    }
                    LineKind::Pop { reg } => reg == src_fam,
                    _ => false,
                };

                if modifies_src {
                    // Check if this is the initial copy: "movq <dst_64>, <src_64>"
                    if is_movq_reg_reg(trimmed, dst_64, src_64) {
                        copy_idx = Some(k);
                        break;
                    }
                    // Check for movslq variant
                    let dst_32 = REG_NAMES[1][dst_fam as usize];
                    if is_movslq_reg_reg(trimmed, dst_32, src_64) {
                        scan_ok = false;
                        break;
                    }
                    modifications.push(k);
                    continue;
                }

                if infos[k].reg_refs & (1u16 << src_fam) != 0 {
                    if infos[k].reg_refs & (1u16 << dst_fam) != 0 {
                        scan_ok = false;
                        break;
                    }
                    modifications.push(k);
                    continue;
                }

                if infos[k].reg_refs & (1u16 << dst_fam) != 0 {
                    scan_ok = false;
                    break;
                }
            }

            if !scan_ok || copy_idx.is_none() {
                move_coalesced.push(false);
                continue;
            }

            let copy_idx = copy_idx.unwrap();

            // Verify fall-through safety
            let check_regs = [dst_fam, src_fam];
            let mut fall_through_safe = true;
            let mut m = branch_idx + 1;
            let mut killed = [false; 2];
            let mut jumps_followed = 0u32;
            'ft_scan: while m < len {
                if infos[m].is_nop() || infos[m].kind == LineKind::Empty
                    || infos[m].kind == LineKind::Label {
                    m += 1;
                    continue;
                }
                if infos[m].kind == LineKind::Jmp {
                    if jumps_followed < 2 && (!killed[0] || !killed[1]) {
                        let trimmed = infos[m].trimmed(store.get(m));
                        if let Some(target) = extract_jump_target(trimmed) {
                            if let Some(n) = parse_dotl_number(target) {
                                if (n as usize) < label_line.len()
                                    && label_line[n as usize] != usize::MAX
                                {
                                    m = label_line[n as usize] + 1;
                                    jumps_followed += 1;
                                    continue 'ft_scan;
                                }
                            }
                        }
                    }
                    break;
                }
                if matches!(infos[m].kind, LineKind::JmpIndirect | LineKind::Ret) {
                    break;
                }
                if infos[m].kind == LineKind::CondJmp {
                    fall_through_safe = false;
                    break;
                }
                for i in 0..2 {
                    if killed[i] {
                        continue;
                    }
                    let reg = check_regs[i];
                    if infos[m].reg_refs & (1u16 << reg) != 0 {
                        fall_through_safe = false;
                        break;
                    }
                    let writes_reg = match infos[m].kind {
                        LineKind::Other { dest_reg } => dest_reg == reg,
                        LineKind::LoadRbp { reg: r, .. } => r == reg,
                        LineKind::SetCC { reg: r } => r == reg,
                        LineKind::Pop { reg: r } => r == reg,
                        _ => false,
                    };
                    if writes_reg {
                        killed[i] = true;
                    }
                }
                if !fall_through_safe {
                    break;
                }
                if killed[0] && killed[1] {
                    break;
                }
                m += 1;
            }
            if !fall_through_safe {
                move_coalesced.push(false);
                continue;
            }

            copy_nop_lines.push(copy_idx);

            for &mod_idx in &modifications {
                coalesce_actions.push((mod_idx, src_fam, dst_fam));
            }

            move_coalesced.push(true);
        }

        // Stack-load coalescing is not attempted (see comment in original code).
        let stack_coalesced: Vec<bool> = vec![false; tramp_stack_loads.len()];
        let stack_nop_lines: Vec<usize> = Vec::new();
        let stack_store_rewrites: Vec<(usize, String)> = Vec::new();

        let num_moves_coalesced = move_coalesced.iter().filter(|&&c| c).count();
        let num_stack_coalesced = stack_coalesced.iter().filter(|&&c| c).count();
        let total_coalesced = num_moves_coalesced + num_stack_coalesced;

        if total_coalesced == 0 {
            continue;
        }

        let all_coalesced = num_moves_coalesced == tramp_moves.len()
            && num_stack_coalesced == tramp_stack_loads.len();

        // Apply the register-register coalescing actions
        for &nop_idx in &copy_nop_lines {
            mark_nop(&mut infos[nop_idx]);
        }

        for &(mod_idx, old_fam, new_fam) in &coalesce_actions {
            let old_line = infos[mod_idx].trimmed(store.get(mod_idx)).to_string();
            if let Some(new_line) = rewrite_instruction_register(&old_line, old_fam, new_fam) {
                replace_line(store, &mut infos[mod_idx], mod_idx, format!("    {}", new_line));
            }
        }

        for &(store_idx, ref new_line) in &stack_store_rewrites {
            replace_line(store, &mut infos[store_idx], store_idx, new_line.clone());
        }
        for &nop_idx in &stack_nop_lines {
            mark_nop(&mut infos[nop_idx]);
        }

        if all_coalesced {
            for &line_idx in &tramp_all_lines {
                mark_nop(&mut infos[line_idx]);
            }
            mark_nop(&mut infos[tramp_label_idx]);

            let branch_trimmed = infos[branch_idx].trimmed(store.get(branch_idx)).to_string();
            if let Some(space_pos) = branch_trimmed.find(' ') {
                let cc = &branch_trimmed[..space_pos];
                let target_label = format!(".LBB{}", target_num);
                let new_branch = format!("    {} {}", cc, target_label);
                replace_line(store, &mut infos[branch_idx], branch_idx, new_branch);
            }
        } else {
            for (idx, &(src_fam, dst_fam)) in tramp_moves.iter().enumerate() {
                if !move_coalesced[idx] { continue; }
                let src_64 = REG_NAMES[0][src_fam as usize];
                let dst_64 = REG_NAMES[0][dst_fam as usize];
                for &line_idx in &tramp_all_lines {
                    if infos[line_idx].is_nop() { continue; }
                    let trimmed = infos[line_idx].trimmed(store.get(line_idx));
                    if is_movq_reg_reg(trimmed, src_64, dst_64) {
                        mark_nop(&mut infos[line_idx]);
                        break;
                    }
                    let src_32 = REG_NAMES[1][src_fam as usize];
                    if is_movslq_reg_reg(trimmed, src_32, dst_64) {
                        mark_nop(&mut infos[line_idx]);
                        break;
                    }
                }
            }
        }

        changed = true;
    }

    changed
}

/// Rewrite an instruction to use a different register family.
fn rewrite_instruction_register(inst: &str, old_fam: RegId, new_fam: RegId) -> Option<String> {
    let result = replace_reg_family(inst, old_fam, new_fam);
    if result == inst {
        None
    } else {
        Some(result)
    }
}

/// Inline join blocks: blocks that consist only of register moves followed by
/// a jmp (or fallthrough into another join block). For each predecessor that
/// jumps to a join block, substitute the moves using the predecessor's register
/// state and redirect directly to the final target.
///
/// This handles multi-level SSA phi-resolution chains like:
///   .LBB45: movq %rbx, %r11; movq %r12, %r10; jmp .LBB9
///   .LBB9:  movq %r11, %r8;  movq %r10, %r9   (fallthrough)
///   .LBB6:  addq $1, %r15;   movq %r8, %rbx;  movq %r9, %r12; jmp .LBB1
///
/// After inlining into LBB45: the net effect is identity (rbx→rbx, r12→r12),
/// so LBB45 becomes: addq $1, %r15; jmp .LBB1
pub(super) fn inline_join_blocks(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    if len < 4 { return false; }

    // Build label_num -> line_index map
    let mut max_label: u32 = 0;
    for i in 0..len {
        if infos[i].is_nop() { continue; }
        if infos[i].kind == LineKind::Label {
            let trimmed = infos[i].trimmed(store.get(i));
            if let Some(n) = parse_label_number(trimmed) {
                if n > max_label { max_label = n; }
            }
        }
    }
    let table_size = (max_label + 1) as usize;
    let mut label_line: Vec<usize> = vec![usize::MAX; table_size];
    for i in 0..len {
        if infos[i].is_nop() { continue; }
        if infos[i].kind == LineKind::Label {
            let trimmed = infos[i].trimmed(store.get(i));
            if let Some(n) = parse_label_number(trimmed) {
                label_line[n as usize] = i;
            }
        }
    }

    // Parse join block contents for each label.
    // A join block = sequence of simple movq/xorl/movl instructions + jmp (or fallthrough).
    // Returns: (moves as instruction strings, final jmp target label_num or fallthrough label_num)
    struct JoinBlock {
        /// Instructions to inline (the actual text lines, with indentation)
        insts: Vec<String>,
        /// Target label number (from jmp or fallthrough)
        target: u32,
    }

    let mut join_blocks: Vec<(u32, JoinBlock)> = Vec::new();

    for label_num in 0..table_size {
        let label_idx = label_line[label_num];
        if label_idx == usize::MAX { continue; }

        let mut insts = Vec::new();
        let mut target: Option<u32> = None;
        let mut valid = true;
        let mut inst_count = 0;

        let mut j = label_idx + 1;
        while j < len {
            if infos[j].is_nop() || infos[j].kind == LineKind::Empty {
                j += 1;
                continue;
            }
            let trimmed = infos[j].trimmed(store.get(j));

            // jmp = end of block
            if infos[j].kind == LineKind::Jmp {
                if let Some(tgt) = extract_jump_target(trimmed) {
                    if let Some(n) = parse_dotl_number(tgt) {
                        target = Some(n);
                    }
                }
                break;
            }

            // Label = fallthrough to next block
            if infos[j].kind == LineKind::Label {
                if let Some(n) = parse_label_number(trimmed) {
                    target = Some(n);
                }
                break;
            }

            // Barrier = not a join block
            if matches!(infos[j].kind, LineKind::CondJmp | LineKind::Call
                | LineKind::JmpIndirect | LineKind::Ret) {
                valid = false;
                break;
            }

            // Only allow simple register moves and small ALU ops (up to 6 insts)
            inst_count += 1;
            if inst_count > 6 {
                valid = false;
                break;
            }

            // Must be a simple instruction (movq reg,reg / xorl / movl / addq imm,reg)
            let is_simple = trimmed.starts_with("movq %")
                || trimmed.starts_with("xorl %")
                || trimmed.starts_with("movl $")
                || trimmed.starts_with("movl %")
                || (trimmed.starts_with("addq $") && !trimmed.contains("("))
                || (trimmed.starts_with("movq $") && !trimmed.contains("("));
            if !is_simple {
                valid = false;
                break;
            }

            insts.push(store.get(j).to_string());
            j += 1;
        }

        if !valid || target.is_none() || insts.is_empty() {
            continue;
        }

        join_blocks.push((label_num as u32, JoinBlock {
            insts,
            target: target.unwrap(),
        }));
    }

    if join_blocks.is_empty() { return false; }

    // Build a lookup from label_num to join_block index
    let mut join_lookup: Vec<usize> = vec![usize::MAX; table_size];
    for (idx, &(num, _)) in join_blocks.iter().enumerate() {
        join_lookup[num as usize] = idx;
    }

    let mut changed = false;

    // For each jmp instruction, check if it targets a join block chain.
    // If so, resolve the full chain and inline the combined instructions.
    for i in 0..len {
        if infos[i].is_nop() { continue; }
        if infos[i].kind != LineKind::Jmp { continue; }

        let trimmed = infos[i].trimmed(store.get(i));
        let target_label = match extract_jump_target(trimmed) {
            Some(t) => t,
            None => continue,
        };
        let first_target = match parse_dotl_number(target_label) {
            Some(n) if (n as usize) < table_size => n,
            _ => continue,
        };

        // Check the predecessor block before this jmp — it must have simple
        // moves only (similar to the join block itself). We'll compose them.
        // Actually, for the inline approach, we just need to collect the chain
        // of join blocks and substitute.

        if join_lookup[first_target as usize] == usize::MAX {
            continue;
        }

        // Resolve the chain of join blocks, tracking which registers the
        // last block in the chain writes (the "output" registers).
        let mut chain_insts: Vec<String> = Vec::new();
        let mut final_target: u32 = first_target;
        let mut visited: Vec<u32> = Vec::new();
        let mut last_block_dsts: u16 = 0; // bitmask of output registers
        let mut cur = first_target;
        loop {
            let jb_idx = join_lookup[cur as usize];
            if jb_idx == usize::MAX { break; }
            let (_, ref jb) = join_blocks[jb_idx];
            if visited.contains(&cur) { break; } // cycle
            visited.push(cur);
            // Track destinations in this block
            last_block_dsts = 0;
            for inst in &jb.insts {
                let t = inst.trim();
                // Extract destination register from movq/xorl/addq etc.
                if let Some(rest) = t.strip_prefix("movq %").or_else(|| t.strip_prefix("movq $")) {
                    if let Some((_, dst_s)) = rest.split_once(", %") {
                        let dst = register_family_no_prefix(dst_s.trim());
                        if dst != REG_NONE { last_block_dsts |= 1 << dst; }
                    }
                } else if let Some(rest) = t.strip_prefix("xorl %") {
                    if let Some((_, b)) = rest.split_once(", %") {
                        let rb = register_family_no_prefix(b.trim());
                        if rb != REG_NONE { last_block_dsts |= 1 << rb; }
                    }
                } else if let Some(rest) = t.strip_prefix("addq $") {
                    if let Some((_, dst_s)) = rest.split_once(", %") {
                        let dst = register_family_no_prefix(dst_s.trim());
                        if dst != REG_NONE { last_block_dsts |= 1 << dst; }
                    }
                }
            }
            chain_insts.extend(jb.insts.iter().cloned());
            final_target = jb.target;
            cur = jb.target;
        }

        if chain_insts.is_empty() || final_target == first_target {
            continue;
        }

        // Now compose: collect the predecessor's moves (between previous label
        // and this jmp) + the chain's moves, and apply substitutions.
        // Strategy: build a register mapping from the combined moves, then
        // emit only the net-effect moves.

        // Collect predecessor instructions (moves before this jmp)
        let mut pred_start = i;
        while pred_start > 0 {
            pred_start -= 1;
            if infos[pred_start].is_nop() || infos[pred_start].kind == LineKind::Empty {
                continue;
            }
            if infos[pred_start].kind == LineKind::Label {
                pred_start += 1;
                break;
            }
            if matches!(infos[pred_start].kind, LineKind::Call | LineKind::Jmp
                | LineKind::JmpIndirect | LineKind::CondJmp | LineKind::Ret) {
                pred_start += 1;
                break;
            }
        }

        // Collect all predecessor instructions as text
        let mut pred_insts: Vec<(usize, String)> = Vec::new();
        let mut pred_all_simple = true;
        for k in pred_start..i {
            if infos[k].is_nop() || infos[k].kind == LineKind::Empty { continue; }
            let t = infos[k].trimmed(store.get(k));
            let is_simple = t.starts_with("movq %")
                || t.starts_with("xorl %")
                || t.starts_with("movl $")
                || t.starts_with("movl %")
                || (t.starts_with("addq $") && !t.contains("("))
                || (t.starts_with("movq $") && !t.contains("("));
            if !is_simple {
                pred_all_simple = false;
                break;
            }
            pred_insts.push((k, store.get(k).to_string()));
        }

        if !pred_all_simple || pred_insts.is_empty() { continue; }

        // Build register substitution map from predecessor + chain.
        // For each movq %A, %B: map[B] = A.
        // For movq $imm, %B or xorl %B, %B: map[B] = literal.
        // Then compose: for chain moves, substitute sources using the map.
        // Finally emit only net-effect instructions.

        #[derive(Clone)]
        enum RegVal {
            Reg(RegId),
            Literal(String), // e.g. "$0", "$1"
        }

        let mut reg_map: [Option<RegVal>; 16] = Default::default();

        // Parse predecessor moves into reg_map
        for (_, ref line) in &pred_insts {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("movq %") {
                if let Some((src_s, dst_s)) = rest.split_once(", %") {
                    let src = register_family_no_prefix(src_s);
                    let dst = register_family_no_prefix(dst_s.trim());
                    if src != REG_NONE && dst != REG_NONE {
                        // Resolve: if src has a mapping, use that
                        let val = match &reg_map[src as usize] {
                            Some(v) => v.clone(),
                            None => RegVal::Reg(src),
                        };
                        reg_map[dst as usize] = Some(val);
                    }
                }
            } else if let Some(rest) = t.strip_prefix("movq $") {
                if let Some((imm, dst_s)) = rest.split_once(", %") {
                    let dst = register_family_no_prefix(dst_s.trim());
                    if dst != REG_NONE {
                        reg_map[dst as usize] = Some(RegVal::Literal(format!("${}", imm)));
                    }
                }
            } else if let Some(rest) = t.strip_prefix("xorl %") {
                if let Some((a, b)) = rest.split_once(", %") {
                    let ra = register_family_no_prefix(a);
                    let rb = register_family_no_prefix(b.trim());
                    if ra == rb && ra != REG_NONE {
                        reg_map[ra as usize] = Some(RegVal::Literal("$0".to_string()));
                    }
                }
            }
        }

        // Apply chain moves to the register map
        for line in &chain_insts {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("movq %") {
                if let Some((src_s, dst_s)) = rest.split_once(", %") {
                    let src = register_family_no_prefix(src_s);
                    let dst = register_family_no_prefix(dst_s.trim());
                    if src != REG_NONE && dst != REG_NONE {
                        let val = match &reg_map[src as usize] {
                            Some(v) => v.clone(),
                            None => RegVal::Reg(src),
                        };
                        reg_map[dst as usize] = Some(val);
                    }
                }
            } else if let Some(rest) = t.strip_prefix("movq $") {
                if let Some((imm, dst_s)) = rest.split_once(", %") {
                    let dst = register_family_no_prefix(dst_s.trim());
                    if dst != REG_NONE {
                        reg_map[dst as usize] = Some(RegVal::Literal(format!("${}", imm)));
                    }
                }
            } else if let Some(rest) = t.strip_prefix("xorl %") {
                if let Some((a, b)) = rest.split_once(", %") {
                    let ra = register_family_no_prefix(a);
                    let rb = register_family_no_prefix(b.trim());
                    if ra == rb && ra != REG_NONE {
                        reg_map[ra as usize] = Some(RegVal::Literal("$0".to_string()));
                    }
                }
            } else if let Some(rest) = t.strip_prefix("addq $") {
                if let Some((_imm, dst_s)) = rest.split_once(", %") {
                    let dst = register_family_no_prefix(dst_s.trim());
                    if dst != REG_NONE {
                        // addq breaks the simple mapping — emit as-is
                        reg_map[dst as usize] = None;
                    }
                }
            }
        }

        // Now emit the net-effect: for each register that has a non-trivial mapping,
        // emit the appropriate instruction. Also include non-move chain instructions
        // (like addq) that aren't captured in the map.

        // Collect non-move chain instructions (addq, etc.)
        let mut extra_insts: Vec<String> = Vec::new();
        for line in &chain_insts {
            let t = line.trim();
            if t.starts_with("addq $") || t.starts_with("subq $") {
                // Substitute source reg if mapped
                extra_insts.push(line.clone());
            }
        }

        // Build net-effect moves (only for output registers of the last chain block)
        let mut net_insts: Vec<String> = Vec::new();
        for reg in 0..16u8 {
            if last_block_dsts & (1 << reg) == 0 { continue; } // skip intermediates
            if let Some(ref val) = reg_map[reg as usize] {
                let dst_64 = REG_NAMES[0][reg as usize];
                match val {
                    RegVal::Reg(src) => {
                        if *src != reg { // Skip identity
                            let src_64 = REG_NAMES[0][*src as usize];
                            net_insts.push(format!("    movq {}, {}", src_64, dst_64));
                        }
                    }
                    RegVal::Literal(lit) => {
                        if lit == "$0" {
                            let dst_32 = REG_NAMES[1][reg as usize];
                            net_insts.push(format!("    xorl {}, {}", dst_32, dst_32));
                        } else {
                            net_insts.push(format!("    movq {}, {}", lit, dst_64));
                        }
                    }
                }
            }
        }

        // Safety check: don't produce more instructions than we're replacing
        let orig_count = pred_insts.len() + 1; // +1 for jmp
        let new_count = extra_insts.len() + net_insts.len() + 1; // +1 for jmp
        if new_count > orig_count { continue; }

        // Apply: NOP predecessor instructions, write new instructions, redirect jmp
        for (k, _) in &pred_insts {
            mark_nop(&mut infos[*k]);
        }

        // Write extra insts (addq etc) + net moves into the NOP'd slots
        let mut write_slots: Vec<usize> = pred_insts.iter().map(|(k, _)| *k).collect();
        write_slots.push(i); // the jmp line itself

        let mut all_new: Vec<String> = Vec::new();
        all_new.extend(extra_insts);
        all_new.extend(net_insts);
        all_new.push(format!("    jmp .LBB{}", final_target));

        // Pad with NOPs if we have fewer new insts
        while all_new.len() < write_slots.len() {
            all_new.push(String::new()); // will be NOP'd
        }

        for (slot_idx, slot) in write_slots.iter().enumerate() {
            if slot_idx < all_new.len() && !all_new[slot_idx].is_empty() {
                replace_line(store, &mut infos[*slot], *slot, all_new[slot_idx].clone());
            } else {
                mark_nop(&mut infos[*slot]);
            }
        }

        changed = true;
    }

    changed
}
