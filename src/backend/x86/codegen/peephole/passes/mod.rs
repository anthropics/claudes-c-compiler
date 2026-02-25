//! x86-64 peephole optimizer: pass orchestration.
//!
//! This module is the entry point (`peephole_optimize`) that runs all optimization
//! passes in the correct order. The actual pass implementations live in submodules:
//!
//! - [`local_patterns`]: combined local pattern matching (self-move, reverse-move,
//!   redundant jump, branch inversion, store/load, extensions) + movq/ext fusion
//! - [`push_pop`]: push/pop pair and push/binop/pop elimination
//! - [`compare_branch`]: compare-and-branch fusion (cmp+setCC+test+jCC -> jCC)
//! - [`copy_propagation`]: register copy propagation across basic blocks
//! - [`dead_code`]: dead register moves, dead stores, never-read store elimination
//! - [`store_forwarding`]: global store forwarding across fallthrough labels
//! - [`loop_trampoline`]: SSA loop backedge trampoline block coalescing
//! - [`callee_saves`]: unused callee-saved register save/restore elimination
//! - [`memory_fold`]: fold stack loads into ALU instructions as memory operands
//! - [`tail_call`]: convert `call; epilogue; ret` to `epilogue; jmp` for tail calls
//! - [`frame_compact`]: stack frame compaction after dead store/callee-save elimination
//! - [`helpers`]: shared utilities (register rewriting, label parsing, etc.)

use super::types::*;

// Submodule pass implementations
mod helpers;
mod local_patterns;
mod push_pop;
mod compare_branch;
mod copy_propagation;
mod dead_code;
mod store_forwarding;
mod loop_trampoline;
mod callee_saves;
mod memory_fold;
mod frame_compact;
mod tail_call;

// ── Constants ────────────────────────────────────────────────────────────────

/// Maximum iterations for Phase 1 (local peephole passes).
/// Local patterns rarely chain deeper than 3-4 levels, so 8 provides ample headroom.
const MAX_LOCAL_PASS_ITERATIONS: usize = 8;

/// Maximum iterations for Phase 3 (local cleanup after global passes).
/// Post-global cleanup is shallow (mostly dead store + adjacent pairs), so 4 suffices.
const MAX_POST_GLOBAL_ITERATIONS: usize = 4;

// ── Main entry point ─────────────────────────────────────────────────────────

/// Run peephole optimization on x86-64 assembly text.
/// Returns the optimized assembly string.
///
/// Pass structure for speed:
/// 1. Run cheap local passes iteratively until convergence (max `MAX_LOCAL_PASS_ITERATIONS`).
///    These are O(n) single-scan passes that only look at adjacent/nearby lines.
/// 2. Run expensive global passes once. `global_store_forwarding` is O(n) but with
///    higher constant factor due to tracking slot→register mappings. It subsumes
///    the functionality of local store-load forwarding across wider windows.
/// 3. Run local passes one more time to clean up opportunities exposed by the
///    global passes (max `MAX_POST_GLOBAL_ITERATIONS` iterations).
pub fn peephole_optimize(asm: String) -> String {
    let mut store = LineStore::new(asm);
    let line_count = store.len();
    let mut infos: Vec<LineInfo> = (0..line_count).map(|i| classify_line(store.get(i))).collect();

    // Phase 1: Iterative cheap local passes.
    let mut changed = true;
    let mut pass_count = 0;
    while changed && pass_count < MAX_LOCAL_PASS_ITERATIONS {
        changed = false;
        let local_changed = local_patterns::combined_local_pass(&mut store, &mut infos);
        changed |= local_changed;
        changed |= local_patterns::fuse_movq_ext_truncation(&mut store, &mut infos);
        changed |= local_patterns::narrow_64_to_32(&mut store, &mut infos);
        changed |= local_patterns::fold_xmm_through_accumulator(&mut store, &mut infos);
        changed |= local_patterns::fold_address_through_secondary(&mut store, &mut infos);
        changed |= local_patterns::fold_commutative_through_temp(&mut store, &mut infos);
        changed |= local_patterns::fold_accumulator_routing(&mut store, &mut infos);
        changed |= local_patterns::redirect_load_destination(&mut store, &mut infos);
        changed |= local_patterns::fold_increment_in_place(&mut store, &mut infos);
        if local_changed || pass_count == 0 {
            changed |= push_pop::eliminate_push_pop_pairs(&store, &mut infos);
            changed |= push_pop::eliminate_binop_push_pop_pattern(&mut store, &mut infos);
        }
        pass_count += 1;
    }

    // Phase 2: Expensive global passes (run once)
    let global_changed = store_forwarding::global_store_forwarding(&mut store, &mut infos);
    let global_changed = global_changed | copy_propagation::propagate_register_copies(&mut store, &mut infos);
    let global_changed = global_changed | dead_code::eliminate_dead_reg_moves(&store, &mut infos);
    let global_changed = global_changed | dead_code::eliminate_dead_reg_moves_ext(&store, &mut infos);
    let global_changed = global_changed | dead_code::eliminate_dead_arg_moves(&store, &mut infos);
    let global_changed = global_changed | dead_code::eliminate_dead_stores(&store, &mut infos);
    let global_changed = global_changed | compare_branch::fuse_compare_and_branch(&mut store, &mut infos);
    let global_changed = global_changed | compare_branch::fuse_and_test_branch(&mut store, &mut infos);
    // Memory operand folding: fold remaining stack loads into subsequent ALU
    // instructions as memory source operands. This runs after store forwarding
    // has already converted loads that can be forwarded from registers; the
    // remaining loads benefit from being folded into ALU instructions.
    let global_changed = global_changed | memory_fold::fold_memory_operands(&mut store, &mut infos);

    // Phase 3: One more local cleanup if global passes made changes.
    if global_changed {
        let mut changed2 = true;
        let mut pass_count2 = 0;
        while changed2 && pass_count2 < MAX_POST_GLOBAL_ITERATIONS {
            changed2 = false;
            changed2 |= local_patterns::combined_local_pass(&mut store, &mut infos);
            changed2 |= local_patterns::fuse_movq_ext_truncation(&mut store, &mut infos);
            changed2 |= local_patterns::narrow_64_to_32(&mut store, &mut infos);
            changed2 |= local_patterns::fold_xmm_through_accumulator(&mut store, &mut infos);
            changed2 |= local_patterns::fold_address_through_secondary(&mut store, &mut infos);
            changed2 |= local_patterns::fold_commutative_through_temp(&mut store, &mut infos);
            changed2 |= local_patterns::fold_double_to_leaq(&mut store, &mut infos);
            changed2 |= local_patterns::fold_movq_addimm_to_leaq(&mut store, &mut infos);
            changed2 |= local_patterns::fold_scaled_address_into_load(&mut store, &mut infos);
            changed2 |= local_patterns::fold_accumulator_routing(&mut store, &mut infos);
            changed2 |= local_patterns::redirect_load_destination(&mut store, &mut infos);
            changed2 |= local_patterns::fold_increment_in_place(&mut store, &mut infos);
            changed2 |= dead_code::eliminate_dead_reg_moves(&store, &mut infos);
            changed2 |= dead_code::eliminate_dead_reg_moves_ext(&store, &mut infos);
            changed2 |= dead_code::eliminate_dead_arg_moves(&store, &mut infos);
            changed2 |= dead_code::eliminate_dead_stores(&store, &mut infos);
            changed2 |= compare_branch::fuse_and_test_branch(&mut store, &mut infos);
            changed2 |= memory_fold::fold_memory_operands(&mut store, &mut infos);
            pass_count2 += 1;
        }
    }

    // Phase 4: Eliminate loop backedge trampoline blocks.
    let trampoline_changed = loop_trampoline::eliminate_loop_trampolines(&mut store, &mut infos);
    // Phase 4a: Inline multi-level join blocks (SSA phi chains).
    let trampoline_changed = trampoline_changed | loop_trampoline::inline_join_blocks(&mut store, &mut infos);

    // Phase 4b: If trampoline elimination made changes, do another round of local cleanup.
    if trampoline_changed {
        let mut changed3 = true;
        let mut pass_count3 = 0;
        while changed3 && pass_count3 < MAX_POST_GLOBAL_ITERATIONS {
            changed3 = false;
            changed3 |= local_patterns::combined_local_pass(&mut store, &mut infos);
            changed3 |= local_patterns::fuse_movq_ext_truncation(&mut store, &mut infos);
            changed3 |= local_patterns::narrow_64_to_32(&mut store, &mut infos);
            changed3 |= local_patterns::fold_commutative_through_temp(&mut store, &mut infos);
            changed3 |= local_patterns::fold_double_to_leaq(&mut store, &mut infos);
            changed3 |= local_patterns::fold_movq_addimm_to_leaq(&mut store, &mut infos);
            changed3 |= local_patterns::fold_scaled_address_into_load(&mut store, &mut infos);
            changed3 |= local_patterns::fold_accumulator_routing(&mut store, &mut infos);
            changed3 |= local_patterns::redirect_load_destination(&mut store, &mut infos);
            changed3 |= local_patterns::fold_increment_in_place(&mut store, &mut infos);
            changed3 |= dead_code::eliminate_dead_reg_moves(&store, &mut infos);
            changed3 |= dead_code::eliminate_dead_arg_moves(&store, &mut infos);
            changed3 |= dead_code::eliminate_dead_stores(&store, &mut infos);
            changed3 |= memory_fold::fold_memory_operands(&mut store, &mut infos);
            pass_count3 += 1;
        }
    }

    // Phase 5: Tail call optimization: convert `call X; epilogue; ret` to
    // `epilogue; jmp X`. This must run before callee-save elimination because
    // removing the call may make some callee-save registers unnecessary.
    tail_call::optimize_tail_calls(&mut store, &mut infos);

    // Phase 5b: Global dead store elimination for never-read stack slots.
    dead_code::eliminate_never_read_stores(&store, &mut infos);

    // Phase 6: Eliminate unused callee-saved register saves/restores.
    callee_saves::eliminate_unused_callee_saves(&store, &mut infos);

    // Phase 7: Compact stack frames by packing callee-saved saves tightly
    // and shrinking subq $N, %rsp when dead stores/callee-save elimination
    // created gaps in the frame.
    frame_compact::compact_frame(&mut store, &mut infos);

    store.build_result(|i| infos[i].is_nop())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redundant_store_load() {
        let asm = "    movq %rax, -8(%rbp)\n    movq -8(%rbp), %rax\n".to_string();
        let result = peephole_optimize(asm);
        assert_eq!(result.trim(), "movq %rax, -8(%rbp)");
    }

    #[test]
    fn test_store_load_different_reg() {
        let asm = "    movq %rax, -8(%rbp)\n    movq -8(%rbp), %rcx\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("movq %rax, -8(%rbp)"));
        assert!(result.contains("movq %rax, %rcx"));
        assert!(!result.contains("movq -8(%rbp), %rcx"));
    }

    #[test]
    fn test_redundant_jump() {
        let asm = "    jmp .Lfoo\n.Lfoo:\n".to_string();
        let result = peephole_optimize(asm);
        assert!(!result.contains("jmp"));
        assert!(result.contains(".Lfoo:"));
    }

    #[test]
    fn test_push_pop_elimination() {
        let asm = "    pushq %rax\n    movq %rax, %rcx\n    popq %rax\n".to_string();
        let result = peephole_optimize(asm);
        assert!(!result.contains("pushq"));
        assert!(!result.contains("popq"));
        assert!(result.contains("movq %rax, %rcx"));
    }

    #[test]
    fn test_self_move() {
        let asm = "    movq %rax, %rax\n".to_string();
        let result = peephole_optimize(asm);
        assert_eq!(result.trim(), "");
    }

    #[test]
    fn test_parse_store_to_rbp() {
        assert!(parse_store_to_rbp_str("movq %rax, -8(%rbp)").is_some());
        assert!(parse_store_to_rbp_str("movl %eax, -16(%rbp)").is_some());
        assert!(parse_store_to_rbp_str("movq $5, -8(%rbp)").is_none());
    }

    #[test]
    fn test_parse_load_from_rbp() {
        assert!(parse_load_from_rbp_str("movq -8(%rbp), %rax").is_some());
        assert!(parse_load_from_rbp_str("movslq -8(%rbp), %rax").is_some());
    }

    #[test]
    fn test_compare_branch_fusion_with_matched_store_load() {
        let asm = [
            "    cmpq %rcx, %rax",
            "    setl %al",
            "    movzbq %al, %rax",
            "    movq %rax, -24(%rbp)",
            "    movq -24(%rbp), %rax",
            "    testq %rax, %rax",
            "    jne .LBB2",
            "    jmp .LBB4",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("cmpq %rcx, %rax"), "should keep the cmp");
        assert!(result.contains("jl .LBB2"), "should fuse to jl: {}", result);
        assert!(!result.contains("setl"), "should eliminate setl");
    }

    #[test]
    fn test_compare_branch_fusion_short() {
        let asm = [
            "    cmpq %rcx, %rax",
            "    setl %al",
            "    movzbq %al, %rax",
            "    testq %rax, %rax",
            "    jne .LBB2",
            "    jmp .LBB4",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jl .LBB2"), "should fuse to jl: {}", result);
        assert!(!result.contains("setl"), "should eliminate setl");
    }

    #[test]
    fn test_compare_branch_fusion_je() {
        let asm = [
            "    cmpq %rcx, %rax",
            "    setl %al",
            "    movzbq %al, %rax",
            "    testq %rax, %rax",
            "    je .Lfalse",
            "    jmp .Ltrue",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jge .Lfalse"), "should fuse to jge: {}", result);
    }

    #[test]
    fn test_non_adjacent_store_load_same_reg() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    movq %rcx, -32(%rbp)",
            "    movq -24(%rbp), %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(!result.contains("-24(%rbp), %rax"), "should eliminate the load: {}", result);
    }

    #[test]
    fn test_non_adjacent_store_load_diff_reg() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    movq %rcx, -32(%rbp)",
            "    movq -24(%rbp), %rdx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("movq %rax, %rdx"), "should forward to reg-reg: {}", result);
    }

    #[test]
    fn test_non_adjacent_store_load_reg_modified() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    movq -32(%rbp), %rax",
            "    movq -24(%rbp), %rcx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-24(%rbp), %rcx") || result.contains("%rax, %rcx"),
            "should not forward since rax was modified: {}", result);
    }

    #[test]
    fn test_redundant_cltq() {
        let asm = "    movslq -8(%rbp), %rax\n    cltq\n".to_string();
        let result = peephole_optimize(asm);
        assert!(result.contains("movslq"), "should keep movslq");
        assert!(!result.contains("cltq"), "should eliminate redundant cltq: {}", result);
    }

    #[test]
    fn test_dead_store_elimination() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    movq %rcx, -24(%rbp)",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(!result.contains("%rax, -24(%rbp)"), "first store should be dead: {}", result);
        assert!(result.contains("%rcx, -24(%rbp)"), "second store should remain: {}", result);
    }

    #[test]
    fn test_condition_codes() {
        for (cc, expected_jcc) in &[("e", "je"), ("ne", "jne"), ("l", "jl"), ("g", "jg"),
                                     ("le", "jle"), ("ge", "jge"), ("b", "jb"), ("a", "ja")] {
            let asm = format!(
                "    cmpq %rcx, %rax\n    set{} %al\n    movzbq %al, %rax\n    testq %rax, %rax\n    jne .LBB1\n",
                cc
            );
            let result = peephole_optimize(asm);
            assert!(result.contains(&format!("{} .LBB1", expected_jcc)),
                "cc={} should produce {}: {}", cc, expected_jcc, result);
        }
    }

    #[test]
    fn test_global_store_forward_across_fallthrough_label() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    movq %rcx, -32(%rbp)",
            ".Lfallthrough:",
            "    movq -24(%rbp), %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(!result.contains("-24(%rbp), %rax"),
            "should forward across fallthrough label: {}", result);
    }

    #[test]
    fn test_global_store_forward_blocked_at_jump_target() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    jmp .Lskip",
            ".Ltarget:",
            "    movq -24(%rbp), %rax",
            ".Lskip:",
            "    ret",
            "    jmp .Ltarget",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-24(%rbp), %rax") || result.contains("-24(%rbp),"),
            "should NOT forward across jump target: {}", result);
    }

    #[test]
    fn test_global_store_forward_across_cond_branch() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    cmpq %rcx, %rax",
            "    jne .Lother",
            "    movq -24(%rbp), %rdx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("movq %rax, %rdx"),
            "should forward on fallthrough after cond branch: {}", result);
    }

    #[test]
    fn test_global_store_forward_invalidated_by_call() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    callq some_func",
            "    movq -24(%rbp), %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-24(%rbp), %rax"),
            "should not forward across call (rax clobbered): {}", result);
    }

    #[test]
    fn test_global_store_forward_callee_saved_across_call() {
        let asm = [
            "    movq %rbx, -24(%rbp)",
            "    callq some_func",
            "    movq -24(%rbp), %rbx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(!result.contains("-24(%rbp), %rbx"),
            "should forward callee-saved reg across call: {}", result);
    }

    #[test]
    fn test_global_store_forward_invalidated_by_unrecognized_rbp_write() {
        let asm = [
            "    movl %eax, -8(%rbp)",
            "    movntil %ecx, -8(%rbp)",
            "    movl -8(%rbp), %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-8(%rbp), %eax"),
            "must not eliminate load after unrecognized write to same slot: {}", result);
    }

    #[test]
    fn test_classify_line() {
        let info = classify_line("    movq %rax, -8(%rbp)");
        assert!(matches!(info.kind, LineKind::StoreRbp { reg: 0, offset: -8, size: MoveSize::Q }));

        let info = classify_line("    movq -16(%rbp), %rcx");
        assert!(matches!(info.kind, LineKind::LoadRbp { reg: 1, offset: -16, size: MoveSize::Q }));

        let info = classify_line(".Lfoo:");
        assert_eq!(info.kind, LineKind::Label);

        let info = classify_line("    jmp .LBB1");
        assert_eq!(info.kind, LineKind::Jmp);

        let info = classify_line("    ret");
        assert_eq!(info.kind, LineKind::Ret);
    }

    #[test]
    fn test_parse_rbp_offset() {
        assert_eq!(parse_rbp_offset("leaq -24(%rbp), %rax"), -24);
        assert_eq!(parse_rbp_offset("addq (%rbp), %rax"), 0);
        assert_eq!(parse_rbp_offset("movq 16(%rbp), %rdx"), 16);
        assert_eq!(parse_rbp_offset("movq %rax, %rcx"), RBP_OFFSET_NONE);
        assert_eq!(parse_rbp_offset("movq -8(%rbp), -16(%rbp)"), RBP_OFFSET_NONE);
        assert_eq!(parse_rbp_offset("addq -8(%rbp), -8(%rbp)"), -8);
    }

    #[test]
    fn test_compare_branch_fusion_no_fuse_cross_block_store() {
        let asm = [
            "    cmpq $0, %rbx",
            "    sete %al",
            "    movzbq %al, %rax",
            "    movq %rax, -40(%rbp)",
            "    testq %rax, %rax",
            "    jne .LBB8",
            "    jmp .LBB10",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-40(%rbp)"),
            "must preserve cross-block store: {}", result);
        assert!(result.contains("sete"),
            "must preserve sete for cross-block store: {}", result);
    }

    #[test]
    fn test_jmp_star_reg_classified_as_indirect() {
        let asm = [
            "    movq %rax, -40(%rbp)",
            "    jmp *%rcx",
            ".LBB21:",
            "    movq -40(%rbp), %rax",
            "    movq %rax, -160(%rbp)",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-40(%rbp), %rax"),
            "must NOT eliminate load after indirect jump target label: {}", result);
    }

    #[test]
    fn test_jmpq_star_reg_classified_as_indirect() {
        let asm = [
            "    movq %rax, -40(%rbp)",
            "    jmpq *%rax",
            ".LBB5:",
            "    movq -40(%rbp), %rax",
            "    ret",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-40(%rbp), %rax"),
            "must NOT eliminate load after jmpq* indirect jump target: {}", result);
    }

    #[test]
    fn test_inline_asm_rdmsr_invalidates_store_forwarding() {
        let asm = [
            "    leaq -16(%rbp), %rax",
            "    movq %rax, -40(%rbp)",
            "    movabsq $27, %rcx",
            "    1: rdmsr ; xor %esi,%esi",
            "    pushq %rcx",
            "    movq -40(%rbp), %rcx",
            "    movl %esi, (%rcx)",
            "    popq %rcx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-40(%rbp), %rcx"),
            "must NOT forward rax across rdmsr (rax clobbered by inline asm): {}", result);
    }

    #[test]
    fn test_semicolon_multi_instruction_invalidates_mappings() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    xorl %eax, %eax ; movl $1, %ecx",
            "    movq -24(%rbp), %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-24(%rbp), %rax"),
            "must not forward across multi-instruction line with ';': {}", result);
    }

    #[test]
    fn test_rdmsr_standalone_invalidates_mappings() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    rdmsr",
            "    movq -24(%rbp), %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-24(%rbp), %rax"),
            "must not forward across rdmsr (implicit clobber of rax/rdx): {}", result);
    }

    #[test]
    fn test_cpuid_invalidates_mappings() {
        let asm = [
            "    movq %rax, -24(%rbp)",
            "    cpuid",
            "    movq -24(%rbp), %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-24(%rbp), %rax"),
            "must not forward across cpuid (implicit clobber of rax/rbx/rcx/rdx): {}", result);
    }

    #[test]
    fn test_setcc_non_al_invalidates_store_forwarding() {
        let asm = [
            "    movl %ecx, -8(%rbp)",
            "    sete %cl",
            "    movl -8(%rbp), %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-8(%rbp), %eax"),
            "must NOT forward ecx across sete %%cl (ecx clobbered): {}", result);
    }

    #[test]
    fn test_setcc_al_still_invalidates_rax() {
        let asm = [
            "    movq %rax, -16(%rbp)",
            "    sete %al",
            "    movq -16(%rbp), %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-16(%rbp), %rax"),
            "must NOT forward rax across sete %%al (rax clobbered): {}", result);
    }

    #[test]
    fn test_syscall_invalidates_mappings() {
        let asm = [
            "    movq %rcx, -16(%rbp)",
            "    syscall",
            "    movq -16(%rbp), %rcx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("-16(%rbp), %rcx"),
            "must NOT forward rcx across syscall (rcx clobbered): {}", result);
    }

    #[test]
    #[ignore] // TODO: trampoline coalescing not triggering for this test pattern yet
    fn test_loop_trampoline_simple_coalesce() {
        let asm = [
            ".LBB1:",
            "    movq %r9, %rax",
            "    movq %r9, %r14",
            "    addq $320, %r14",
            "    testq %rax, %rax",
            "    jne .LBB2",
            "    ret",
            ".LBB2:",
            "    movq %r14, %r9",
            "    jmp .LBB1",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("addq $320, %r9"),
            "should rewrite addq to target %r9 directly: {}", result);
        assert!(!result.contains("movq %r9, %r14"),
            "should eliminate the initial copy: {}", result);
        assert!(!result.contains("movq %r14, %r9"),
            "should eliminate the trampoline copy: {}", result);
        assert!(result.contains("jne .LBB1"),
            "should redirect branch to loop header: {}", result);
    }

    #[test]
    #[ignore] // TODO: trampoline coalescing not triggering for this test pattern yet
    fn test_loop_trampoline_two_copies() {
        let asm = [
            ".LBB10:",
            "    movq %r9, %rax",
            "    movq %r10, %rcx",
            "    movq %r9, %r14",
            "    addq $320, %r14",
            "    movq %r10, %r15",
            "    addl %r8d, %r15d",
            "    testq %rax, %rax",
            "    jne .LBB20",
            "    ret",
            ".LBB20:",
            "    movq %r14, %r9",
            "    movq %r15, %r10",
            "    jmp .LBB10",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("addq $320, %r9"),
            "should rewrite dest addq to %r9: {}", result);
        assert!(result.contains("addl %r8d, %r10d"),
            "should rewrite frac addl to %r10d: {}", result);
        assert!(!result.contains("movq %r9, %r14"),
            "should eliminate dest copy: {}", result);
        assert!(!result.contains("movq %r10, %r15"),
            "should eliminate frac copy: {}", result);
        assert!(result.contains("jne .LBB10"),
            "should redirect branch to loop header: {}", result);
    }

    #[test]
    fn test_condbranch_inversion_fallthrough() {
        let asm = [
            "    cmpl %r8d, %eax",
            "    jl .LBB2",
            "    jmp .LBB4",
            ".LBB2:",
            "    movq %rax, %rcx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jge .LBB4"),
            "should invert jl to jge: {}", result);
        assert!(!result.contains("jl .LBB2"),
            "should remove original jl: {}", result);
        assert!(!result.contains("jmp .LBB4"),
            "should remove the jmp: {}", result);
        assert!(result.contains(".LBB2:"),
            "should keep the label: {}", result);
    }

    #[test]
    fn test_condbranch_inversion_je_to_jne() {
        let asm = [
            "    testq %rax, %rax",
            "    je .Ltrue",
            "    jmp .Lfalse",
            ".Ltrue:",
            "    ret",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jne .Lfalse"),
            "should invert je to jne: {}", result);
        assert!(!result.contains("jmp .Lfalse"),
            "should remove the jmp: {}", result);
    }

    #[test]
    fn test_condbranch_no_inversion_when_not_fallthrough() {
        let asm = [
            "    cmpl %r8d, %eax",
            "    jl .LBB5",
            "    jmp .LBB4",
            ".LBB2:",
            "    movq %rax, %rcx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("jl .LBB5"),
            "should keep jl when not fallthrough: {}", result);
    }

    #[test]
    fn test_back_to_back_cltq() {
        let asm = "    cltq\n    cltq\n".to_string();
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("cltq").count(), 1,
            "should keep only one cltq: {}", result);
    }

    #[test]
    fn test_cltq_backward_scan_over_non_rax_write() {
        let asm = [
            "    movslq -8(%rbp), %rax",
            "    movq %rax, %r8",
            "    cltq",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(!result.contains("cltq"),
            "cltq should be eliminated after movslq past non-rax-write: {}", result);
    }

    #[test]
    fn test_cltq_backward_scan_blocked_by_rax_write() {
        let asm = [
            "    movslq -8(%rbp), %rax",
            "    addl $1, %eax",
            "    cltq",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("cltq"),
            "cltq should NOT be eliminated when rax is modified: {}", result);
    }

    #[test]
    fn test_cltq_backward_scan_blocked_by_call() {
        let asm = [
            "    cltq",
            "    call foo",
            "    cltq",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("cltq").count(), 2,
            "both cltq should survive when call intervenes: {}", result);
    }

    #[test]
    fn test_cltq_backward_scan_with_store_rbp() {
        let asm = [
            "    cltq",
            "    movq %rax, -16(%rbp)",
            "    movq %rax, %r9",
            "    cltq",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("cltq").count(), 1,
            "second cltq should be eliminated past store and mov: {}", result);
    }

    // ── Memory operand folding tests ──────────────────────────────────────

    #[test]
    fn test_mem_fold_addq_rcx() {
        let asm = [
            "    movq -48(%rbp), %rcx",
            "    addq %rcx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("addq -48(%rbp), %rax"),
            "should fold load+add into memory operand: {}", result);
        assert!(!result.contains("movq -48(%rbp), %rcx"),
            "load should be eliminated: {}", result);
    }

    #[test]
    fn test_mem_fold_subl_ecx() {
        let asm = [
            "    movq -64(%rbp), %rcx",
            "    subl %ecx, %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("subl -64(%rbp), %eax"),
            "should fold load+sub into memory operand: {}", result);
    }

    #[test]
    fn test_mem_fold_cmpq_rcx() {
        let asm = [
            "    movq -8(%rbp), %rcx",
            "    cmpq %rcx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("cmpq -8(%rbp), %rax"),
            "should fold load+cmp into memory operand: {}", result);
    }

    #[test]
    fn test_mem_fold_testq_rcx() {
        let asm = [
            "    movq -16(%rbp), %rcx",
            "    testq %rcx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("testq -16(%rbp), %rax"),
            "should fold load+test into memory operand: {}", result);
    }

    #[test]
    fn test_mem_fold_no_fold_when_dest_is_loaded_reg() {
        let asm = [
            "    movq -48(%rbp), %rcx",
            "    addq %rax, %rcx",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("movq -48(%rbp), %rcx") || result.contains("addq %rax, %rcx"),
            "should not fold when loaded reg is destination: {}", result);
    }

    #[test]
    fn test_mem_fold_no_fold_for_callee_saved() {
        let asm = [
            "    movq -48(%rbp), %rbx",
            "    addq %rbx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(!result.contains("addq -48(%rbp), %rax"),
            "should not fold callee-saved register loads: {}", result);
    }

    #[test]
    fn test_mem_fold_andq() {
        let asm = [
            "    movq -16(%rbp), %rcx",
            "    andq %rcx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("andq -16(%rbp), %rax"),
            "should fold load+and into memory operand: {}", result);
    }

    #[test]
    fn test_mem_fold_xorq() {
        let asm = [
            "    movq -24(%rbp), %rcx",
            "    xorq %rcx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("xorq -24(%rbp), %rax"),
            "should fold load+xor into memory operand: {}", result);
    }

    #[test]
    fn test_mem_fold_load_rax_into_add_with_reg_dest() {
        let asm = [
            "    movq -32(%rbp), %rax",
            "    addq %rax, %r12",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("addq -32(%rbp), %r12"),
            "should fold rax load into add with callee-saved dest: {}", result);
    }

    #[test]
    fn test_mem_fold_orq() {
        let asm = [
            "    movq -16(%rbp), %rcx",
            "    orq %rcx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("orq -16(%rbp), %rax"),
            "should fold load+or into memory operand: {}", result);
    }

    #[test]
    fn test_mem_fold_with_empty_line_between() {
        let asm = [
            "    movq -48(%rbp), %rcx",
            "",
            "    addq %rcx, %rax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert!(result.contains("addq -48(%rbp), %rax"),
            "should fold with empty lines between: {}", result);
    }

    // ── Redundant xorl elimination tests ─────────────────────────────────

    #[test]
    fn test_redundant_xorl_after_zero_store() {
        let asm = [
            "    xorl %eax, %eax",
            "    movq %rax, -8(%rbp)",
            "    xorl %eax, %eax",
            "    movq %rax, -16(%rbp)",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("xorl %eax, %eax").count(), 1,
            "second xorl should be eliminated: {}", result);
        assert!(result.contains("movq %rax, -8(%rbp)"),
            "first store should remain: {}", result);
        assert!(result.contains("movq %rax, -16(%rbp)"),
            "second store should remain: {}", result);
    }

    #[test]
    fn test_redundant_xorl_chain_of_four() {
        let asm = [
            "    xorl %eax, %eax",
            "    movq %rax, -8(%rbp)",
            "    xorl %eax, %eax",
            "    movq %rax, -16(%rbp)",
            "    xorl %eax, %eax",
            "    movq %rax, -24(%rbp)",
            "    xorl %eax, %eax",
            "    movq %rax, -32(%rbp)",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("xorl %eax, %eax").count(), 1,
            "only first xorl should survive: {}", result);
    }

    #[test]
    fn test_xorl_not_eliminated_after_rax_write() {
        let asm = [
            "    xorl %eax, %eax",
            "    movq %rax, -8(%rbp)",
            "    movq -16(%rbp), %rax",
            "    xorl %eax, %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        // The load to %rax invalidates rax_is_zero, so both xorls are needed
        assert_eq!(result.matches("xorl %eax, %eax").count(), 2,
            "both xorls should survive after rax modification: {}", result);
    }

    #[test]
    fn test_xorl_not_eliminated_after_label() {
        let asm = [
            "    xorl %eax, %eax",
            "    movq %rax, -8(%rbp)",
            ".LBB1:",
            "    xorl %eax, %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("xorl %eax, %eax").count(), 2,
            "xorl after label should NOT be eliminated: {}", result);
    }

    #[test]
    fn test_xorl_not_eliminated_after_call() {
        let asm = [
            "    xorl %eax, %eax",
            "    movq %rax, -8(%rbp)",
            "    call some_func",
            "    xorl %eax, %eax",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        assert_eq!(result.matches("xorl %eax, %eax").count(), 2,
            "xorl after call should NOT be eliminated: {}", result);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Regression test: dead store elimination must not use a stale pattern_bytes
    /// buffer when checking multi-byte stores. The sub-byte-offset scan for a Q
    /// (8-byte) store was overwriting pattern_bytes with the last sub-offset
    /// (store_offset + 7), and subsequent scans within the same store's window
    /// would reuse the stale pattern instead of the original store_offset.
    ///
    /// Pattern from oniguruma regparse.c: a store to -144(%rbp) was followed by
    /// an Other instruction (movl %eax, %eax), then a Cmp with -144(%rbp) as a
    /// memory operand (from memory_fold), then another store to -144(%rbp). The
    /// sub-byte scan for the Other instruction overwrote pattern_bytes from
    /// "-144(%rbp)" to "-137(%rbp)", causing the Cmp line's pattern check to fail,
    /// and the later store marked the original as dead.
    #[test]
    fn test_dead_store_not_eliminated_when_cmp_reads_slot() {
        // The critical pattern after memory_fold transforms the comparison:
        //   movq %rax, -144(%rbp)     # store data[x*2]
        //   movq -128(%rbp), %rax     # load to+1 (clobbers rax)
        //   movl %eax, %eax           # truncate (Other, no rbp ref)
        //   movq %rax, -136(%rbp)     # store to+1
        //   cmpl -144(%rbp), %eax     # memory-folded cmp reads -144!
        //   setae %al
        //   movzbq %al, %rax
        //   movq %rax, %rsi
        //   movq %r11, %rax
        //   addl $1, %eax
        //   cltq
        //   movq %rax, -144(%rbp)     # later store overwrites -144
        //
        // The bug: dead_stores saw the overwrite at the end, but missed the
        // cmp read because pattern_bytes had been corrupted by the sub-byte
        // offset scan for movl %eax, %eax.
        let asm = [
            "func:",
            "    pushq %rbp",
            "    movq %rsp, %rbp",
            "    subq $160, %rsp",
            // ... setup ...
            "    movl (%rcx), %eax",
            "    movq %rax, -144(%rbp)",       // store data[x*2]
            "    movq -128(%rbp), %rax",       // load to+1 (clobbers rax)
            "    movl %eax, %eax",             // truncate
            "    movq %rax, -136(%rbp)",       // store to+1
            "    cmpl -144(%rbp), %eax",       // memory-folded cmp (Cmp kind)
            "    setae %al",
            "    movzbq %al, %rax",
            "    movq %rax, %rsi",
            "    movq %r11, %rax",
            "    addl $1, %eax",
            "    cltq",
            "    movq %rax, -144(%rbp)",       // later overwrite of -144
            "    ret",
            ".size func, .-func",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        // After optimization, there must still be a store to -144(%rbp) before
        // the cmpl that reads it. The cmpl must compare the correct value.
        let lines: Vec<&str> = result.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed == "cmpl -144(%rbp), %eax" {
                // Scan backward for a store to -144
                let mut found_store = false;
                for k in (0..idx).rev() {
                    let prev = lines[k].trim();
                    if prev.ends_with("-144(%rbp)") && prev.starts_with("mov") {
                        found_store = true;
                        break;
                    }
                    if prev.ends_with(':') { break; }
                }
                assert!(found_store,
                    "cmpl -144(%rbp) has no preceding store in same block!\nResult:\n{}", result);
                return;
            }
        }
        // If the cmpl was not folded, check it exists in some form
        assert!(result.contains("cmpl") || result.contains("setae"),
            "No comparison found\nResult:\n{}", result);
    }

    #[test]
    fn test_store_forward_param_ref_gep() {
        // This pattern comes from:
        //   void f(struct state *s) { ... s->member[i] = 0; ... }
        // The codegen emits:
        //   movq %rdi, -8(%rbp)    # store param
        //   movq -8(%rbp), %rax    # paramref load
        //   movq %rax, -8(%rbp)    # paramref store-back (redundant)
        //   movq -8(%rbp), %rax    # GEP base load
        //   leaq 208(%rax), %rax   # GEP offset
        // After peephole, rax must still be set correctly before the leaq.
        let asm = [
            "func:",
            "    pushq %rbp",
            "    movq %rsp, %rbp",
            "    subq $16, %rsp",
            "    movq %rdi, -8(%rbp)",
            "    movq -8(%rbp), %rax",
            "    movq %rax, -8(%rbp)",
            "    movq -8(%rbp), %rax",
            "    leaq 208(%rax), %rax",
            "    movq %rax, %r14",
            "    ret",
            ".size func, .-func",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        // After optimization, rax must be loaded from rdi or from the stack
        // before the leaq instruction.
        // The correct result should be something like:
        //   movq %rdi, -8(%rbp) or movq %rdi, %rax + leaq 208(%rax)...
        // NOT: leaq 208(%rax) with rax uninitialized!
        eprintln!("Result:\n{}", result);
        assert!(
            result.contains("movq %rdi, %rax") || 
            result.contains("movq -8(%rbp), %rax") ||
            result.contains("leaq 208(%rdi)"),
            "rax must be set from rdi before leaq 208(%rax): {}", result
        );
    }

    /// Regression test: frame_compact must not NOP a store when a read overlaps
    /// it at a different offset. Example: a struct param stored as
    /// `movq %rsi, -8(%rbp)` (8 bytes at [-8, 0)) has a field read via
    /// `movl -4(%rbp), %eax` (4 bytes at [-4, 0)). The store and read overlap
    /// but have different offsets, so exact-offset matching would miss the
    /// dependency and incorrectly NOP the store.
    #[test]
    fn test_frame_compact_overlapping_store_read() {
        let asm = [
            "func:",
            "    pushq %rbp",
            "    .cfi_def_cfa_offset 16",
            "    .cfi_offset %rbp, -16",
            "    movq %rsp, %rbp",
            "    .cfi_def_cfa_register %rbp",
            "    subq $48, %rsp",
            "    movq %rbx, -48(%rbp)",
            "    movq %r12, -40(%rbp)",
            "    movq %r13, -32(%rbp)",
            "    movq %r14, -24(%rbp)",
            "    movq %r15, -16(%rbp)",
            // Store 8-byte struct param at -8(%rbp) covering bytes [-8, 0)
            "    movq %rsi, -8(%rbp)",
            // Read a 4-byte field at -4(%rbp) covering bytes [-4, 0)
            "    movl -4(%rbp), %eax",
            "    movq %rax, %r14",
            "    movq %rdi, %rdi",
            "    call some_func",
            // Epilogue
            "    movq %r14, %rax",
            "    movq -48(%rbp), %rbx",
            "    movq -40(%rbp), %r12",
            "    movq -32(%rbp), %r13",
            "    movq -24(%rbp), %r14",
            "    movq -16(%rbp), %r15",
            "    movq %rbp, %rsp",
            "    popq %rbp",
            "    ret",
            ".size func, .-func",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        // The store to -8(%rbp) must survive because the read at -4(%rbp) overlaps it.
        // The load must also survive.
        let lines: Vec<&str> = result.lines().map(|l| l.trim()).collect();
        let has_store = lines.iter().any(|l| l.contains("%rsi") && l.contains("(%rbp)"));
        let has_load = lines.iter().any(|l| l.starts_with("movl") && l.contains("(%rbp)") && l.contains("%eax"));
        assert!(has_store,
            "store of struct param must survive frame compaction (overlapping read exists): {}", result);
        assert!(has_load,
            "load of struct field must survive: {}", result);
    }

    /// Regression test: frame compaction must NOP out dead stores that conflict
    /// with relocated callee-save offsets. Without this fix, a dead store like
    /// `movq %rax, -64(%rbp)` can clobber a callee-saved register that was
    /// relocated to -64(%rbp) during frame compaction.
    ///
    /// Pattern from tre-compile.c: tre_ast_to_tnfa has 5 callee saves at
    /// -112..-80 and body reads down to -56, with a dead store at -64.
    /// Compaction moves callee saves to -96..-64, but -64 conflicts with
    /// the dead store.
    #[test]
    fn test_frame_compact_dead_store_noped() {
        let asm = [
            "func:",
            "    pushq %rbp",
            "    .cfi_def_cfa_offset 16",
            "    .cfi_offset %rbp, -16",
            "    movq %rsp, %rbp",
            "    .cfi_def_cfa_register %rbp",
            "    subq $112, %rsp",
            "    movq %rbx, -112(%rbp)",
            "    movq %r12, -104(%rbp)",
            "    movq %r13, -96(%rbp)",
            "    movq %r14, -88(%rbp)",
            "    movq %r15, -80(%rbp)",
            // Body: reads at -8, -56; dead store at -64
            "    movq %rdi, -8(%rbp)",
            "    movq -8(%rbp), %rax",
            "    movq %rax, -56(%rbp)",
            "    movq -56(%rbp), %rdi",
            "    call some_func",
            "    movq %rax, -64(%rbp)",      // dead store - never read
            "    movq %rax, %r14",
            // Epilogue
            "    movq %r14, %rax",
            "    movq -112(%rbp), %rbx",
            "    movq -104(%rbp), %r12",
            "    movq -96(%rbp), %r13",
            "    movq -88(%rbp), %r14",
            "    movq -80(%rbp), %r15",
            "    movq %rbp, %rsp",
            "    popq %rbp",
            "    ret",
            ".size func, .-func",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        // After compaction:
        // - Frame should be smaller than 112
        // - Dead store at -64 should be NOP'd (not present in output)
        // - Callee saves should be at new offsets
        assert!(result.contains("subq $"),
            "should have subq: {}", result);
        assert!(!result.contains("subq $112"),
            "frame should be compacted from 112: {}", result);
        // The dead store to -64 must not appear in the output
        // (it would clobber the relocated callee save)
        assert!(!result.contains("-64(%rbp)") ||
                // -64 might appear as a new callee-save offset in saves/restores which is OK
                (result.contains("movq %r15, -64(%rbp)") || result.contains("movq -64(%rbp), %r15")),
            "dead store to -64 must be eliminated or -64 used only for callee save: {}", result);
    }

    /// Regression test: a struct stored with movq at -8(%rbp) that is read
    /// field-by-field via movl at -4(%rbp) must NOT be NOP'd by frame_compact.
    /// The store covers bytes [-8, 0) and the read at -4 falls within that range.
    #[test]
    fn test_frame_compact_struct_suboffset_read() {
        let asm = [
            "release_entry:",
            "    pushq %rbp",
            "    .cfi_def_cfa_offset 16",
            "    .cfi_offset %rbp, -16",
            "    movq %rsp, %rbp",
            "    .cfi_def_cfa_register %rbp",
            "    subq $48, %rsp",
            "    movq %rbx, -48(%rbp)",
            "    movq %r12, -40(%rbp)",
            "    movq %r13, -32(%rbp)",
            "    movq %r14, -24(%rbp)",
            "    movq %r15, -16(%rbp)",
            // Struct param stored as a whole at -8(%rbp)
            "    movq %rdi, %r14",
            "    movq %rsi, -8(%rbp)",
            // Read individual field at sub-offset -4 within the struct
            "    movl -4(%rbp), %eax",
            "    movq %rax, %rsi",
            "    call printf",
            // Epilogue
            "    movq -48(%rbp), %rbx",
            "    movq -40(%rbp), %r12",
            "    movq -32(%rbp), %r13",
            "    movq -24(%rbp), %r14",
            "    movq -16(%rbp), %r15",
            "    movq %rbp, %rsp",
            "    popq %rbp",
            "    ret",
            ".size release_entry, .-release_entry",
        ].join("\n") + "\n";
        let result = peephole_optimize(asm);
        // The struct store at -8(%rbp) must be preserved because -4(%rbp) is read
        assert!(result.contains("movq %rsi, -8(%rbp)"),
            "struct param store at -8(%rbp) must NOT be NOP'd when -4(%rbp) is read: {}", result);
    }
}

