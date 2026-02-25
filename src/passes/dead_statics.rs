//! Dead static function and global elimination.
//!
//! After optimization passes eliminate dead code paths, some static inline
//! functions and static const globals from headers may no longer be referenced.
//! Keeping them wastes code size and may cause linker errors if they reference
//! symbols that don't exist in this translation unit.
//!
//! Uses BFS reachability analysis from roots (non-static symbols) to find all
//! live symbols, then removes unreachable static functions and globals.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::reexports::{GlobalInit, Instruction, IrModule};

/// Remove internal-linkage (static) functions and globals that are unreachable.
///
/// Uses reachability analysis from roots (non-static symbols) to find all live symbols,
/// then removes unreachable static functions and globals.
pub(crate) fn eliminate_dead_static_functions(module: &mut IrModule) {
    // Phase 1: Build name-to-index mapping for all symbols.
    let (mut name_to_id, mut next_id, id_func_idx, id_global_idx, func_id, global_id) =
        build_symbol_index(module);

    // Phase 2: Build reference lists per function and global (using symbol IDs).
    let func_refs = build_func_refs(module, &mut name_to_id, &mut next_id);
    let global_refs_lists = build_global_refs(module, &mut name_to_id, &mut next_id);

    // Phase 3: Build address_taken set (moved before BFS so we can use it as roots).
    let address_taken = build_address_taken(module, &name_to_id, next_id as usize);

    // Phase 4: Reachability BFS from roots, including address-taken functions.
    let reachable = compute_reachability(
        module, &func_id, &global_id, &id_func_idx, &id_global_idx,
        &func_refs, &global_refs_lists, &address_taken,
        &mut name_to_id, &mut next_id,
    );

    // Drop the borrow on module strings so we can mutate module below.
    drop(name_to_id);

    // Phase 5: Remove unreachable symbols.
    remove_unreachable(module, &func_id, &global_id, &reachable, &address_taken);

    // Phase 6: Filter symbol_attrs for surviving symbols only.
    filter_symbol_attrs(module);
}

/// Phase 1: Assign compact integer IDs to all function and global names.
/// Returns (name_to_id, next_id, id_func_idx, id_global_idx, func_id, global_id).
fn build_symbol_index(module: &IrModule) -> (
    FxHashMap<&str, u32>, u32,
    Vec<Option<usize>>, Vec<Option<usize>>,
    Vec<u32>, Vec<u32>,
) {
    let mut name_to_id: FxHashMap<&str, u32> = FxHashMap::default();
    let mut next_id: u32 = 0;
    let mut id_func_idx: Vec<Option<usize>> = Vec::new();
    let mut id_global_idx: Vec<Option<usize>> = Vec::new();

    let mut func_id: Vec<u32> = Vec::with_capacity(module.functions.len());
    for (i, func) in module.functions.iter().enumerate() {
        let id = *name_to_id.entry(&*func.name).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id_func_idx.push(None);
            id_global_idx.push(None);
            id
        });
        id_func_idx[id as usize] = Some(i);
        func_id.push(id);
    }

    let mut global_id: Vec<u32> = Vec::with_capacity(module.globals.len());
    for (i, global) in module.globals.iter().enumerate() {
        let id = *name_to_id.entry(&*global.name).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id_func_idx.push(None);
            id_global_idx.push(None);
            id
        });
        id_global_idx[id as usize] = Some(i);
        global_id.push(id);
    }

    (name_to_id, next_id, id_func_idx, id_global_idx, func_id, global_id)
}

/// Look up or create an ID for a name that may not already exist.
fn get_or_create_id<'a>(name: &'a str, name_to_id: &mut FxHashMap<&'a str, u32>, next_id: &mut u32) -> u32 {
    *name_to_id.entry(name).or_insert_with(|| {
        let id = *next_id;
        *next_id += 1;
        id
    })
}

/// Phase 2a: Build per-function reference lists using symbol IDs.
fn build_func_refs<'a>(module: &'a IrModule, name_to_id: &mut FxHashMap<&'a str, u32>, next_id: &mut u32) -> Vec<Vec<u32>> {
    let mut func_refs: Vec<Vec<u32>> = Vec::with_capacity(module.functions.len());
    for func in &module.functions {
        if func.is_declaration {
            func_refs.push(Vec::new());
            continue;
        }
        let mut refs = Vec::new();
        for block in &func.blocks {
            for inst in &block.instructions {
                collect_instruction_symbol_refs(inst, name_to_id, next_id, &mut refs);
            }
        }
        func_refs.push(refs);
    }
    func_refs
}

/// Phase 2b: Build per-global reference lists from initializers.
fn build_global_refs(module: &IrModule, name_to_id: &mut FxHashMap<&str, u32>, next_id: &mut u32) -> Vec<Vec<u32>> {
    let mut global_refs_lists: Vec<Vec<u32>> = Vec::with_capacity(module.globals.len());
    for global in &module.globals {
        let mut id_refs = Vec::new();
        global.init.for_each_ref(&mut |name| {
            // Use inline lookup since for_each_ref's callback lifetime is too short for get_or_create_id.
            let id = if let Some(&id) = name_to_id.get(name) {
                id
            } else {
                let id = *next_id;
                *next_id += 1;
                id
            };
            id_refs.push(id);
        });
        global_refs_lists.push(id_refs);
    }
    global_refs_lists
}

/// Mark a symbol ID as reachable if not already, growing the reachable vec as needed.
fn mark_reachable(id: u32, reachable: &mut Vec<bool>, worklist: &mut Vec<u32>, next_id: u32) {
    let idx = id as usize;
    if idx >= reachable.len() { reachable.resize(next_id as usize, false); }
    if !reachable[idx] {
        reachable[idx] = true;
        worklist.push(id);
    }
}

/// Phase 4: Compute reachability from roots via BFS.
///
/// Roots include: non-static functions, non-static globals, aliases, constructors,
/// destructors, toplevel asm references, and address-taken static always_inline functions
/// (which survive dead elimination because they're used as function pointers).
fn compute_reachability<'a>(
    module: &'a IrModule,
    func_id: &[u32], global_id: &[u32],
    id_func_idx: &[Option<usize>], id_global_idx: &[Option<usize>],
    func_refs: &[Vec<u32>], global_refs_lists: &[Vec<u32>],
    address_taken: &[bool],
    name_to_id: &mut FxHashMap<&'a str, u32>, next_id: &mut u32,
) -> Vec<bool> {
    let mut reachable = vec![false; *next_id as usize];
    let mut worklist: Vec<u32> = Vec::new();

    // Roots: non-static functions
    for (i, func) in module.functions.iter().enumerate() {
        if func.is_declaration { continue; }
        if !func.is_static || func.is_used {
            mark_reachable(func_id[i], &mut reachable, &mut worklist, *next_id);
        }
    }

    // Roots: non-static globals
    for (i, global) in module.globals.iter().enumerate() {
        if global.is_extern { continue; }
        if !global.is_static || global.is_common || global.is_used {
            mark_reachable(global_id[i], &mut reachable, &mut worklist, *next_id);
        }
    }

    // Roots: aliases (both the alias name and its target are reachable)
    for (alias_name, target, _) in &module.aliases {
        let tid = get_or_create_id(target, name_to_id, next_id);
        mark_reachable(tid, &mut reachable, &mut worklist, *next_id);
        let aid = get_or_create_id(alias_name, name_to_id, next_id);
        mark_reachable(aid, &mut reachable, &mut worklist, *next_id);
    }

    // Roots: constructors and destructors
    for ctor in &module.constructors {
        let id = get_or_create_id(ctor, name_to_id, next_id);
        mark_reachable(id, &mut reachable, &mut worklist, *next_id);
    }
    for dtor in &module.destructors {
        let id = get_or_create_id(dtor, name_to_id, next_id);
        mark_reachable(id, &mut reachable, &mut worklist, *next_id);
    }

    if reachable.len() < *next_id as usize {
        reachable.resize(*next_id as usize, false);
    }

    // Roots: address-taken static always_inline functions.
    // These survive dead elimination (Phase 5) because their address is used as a
    // function pointer, so their referenced globals/functions must also survive.
    for (i, func) in module.functions.iter().enumerate() {
        if func.is_declaration { continue; }
        if func.is_static && func.is_always_inline {
            let fid = func_id[i] as usize;
            if fid < address_taken.len() && address_taken[fid] {
                mark_reachable(func_id[i], &mut reachable, &mut worklist, *next_id);
            }
        }
    }

    // Toplevel asm: conservatively mark static symbols whose names appear in asm
    if !module.toplevel_asm.is_empty() {
        for (i, func) in module.functions.iter().enumerate() {
            if func.is_static && !func.is_declaration {
                let fid = func_id[i] as usize;
                if !reachable[fid] && module.toplevel_asm.iter().any(|s| s.contains(&*func.name)) {
                    reachable[fid] = true;
                    worklist.push(fid as u32);
                }
            }
        }
        for (i, global) in module.globals.iter().enumerate() {
            if global.is_static && !global.is_extern {
                let gid = global_id[i] as usize;
                if !reachable[gid] && module.toplevel_asm.iter().any(|s| s.contains(&*global.name)) {
                    reachable[gid] = true;
                    worklist.push(gid as u32);
                }
            }
        }
    }

    // BFS: propagate reachability through function and global references.
    while let Some(sym_id) = worklist.pop() {
        let sid = sym_id as usize;
        if sid < id_func_idx.len() {
            if let Some(fi) = id_func_idx[sid] {
                if fi < func_refs.len() {
                    for &ref_id in &func_refs[fi] {
                        let rid = ref_id as usize;
                        if rid < reachable.len() && !reachable[rid] {
                            reachable[rid] = true;
                            worklist.push(ref_id);
                        }
                    }
                }
            }
        }
        if sid < id_global_idx.len() {
            if let Some(gi) = id_global_idx[sid] {
                if gi < global_refs_lists.len() {
                    for &ref_id in &global_refs_lists[gi] {
                        let rid = ref_id as usize;
                        if rid < reachable.len() && !reachable[rid] {
                            reachable[rid] = true;
                            worklist.push(ref_id);
                        }
                    }
                }
            }
        }
    }

    reachable
}

/// Phase 3: Build address_taken bitvector from GlobalAddr and InlineAsm instructions.
fn build_address_taken<'a>(module: &'a IrModule, name_to_id: &FxHashMap<&'a str, u32>, len: usize) -> Vec<bool> {
    let mut address_taken = vec![false; len];

    for func in &module.functions {
        if func.is_declaration { continue; }
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::GlobalAddr { name, .. } => {
                        if let Some(&id) = name_to_id.get(&**name) {
                            if (id as usize) < address_taken.len() {
                                address_taken[id as usize] = true;
                            }
                        }
                    }
                    Instruction::InlineAsm { input_symbols, .. } => {
                        for s in input_symbols.iter().flatten() {
                            let base = s.split('+').next().unwrap_or(s);
                            if let Some(&id) = name_to_id.get(base) {
                                if (id as usize) < address_taken.len() {
                                    address_taken[id as usize] = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    for global in &module.globals {
        global.init.for_each_ref(&mut |name| {
            if let Some(&id) = name_to_id.get(name) {
                if (id as usize) < address_taken.len() {
                    address_taken[id as usize] = true;
                }
            }
        });
    }

    address_taken
}

/// Phase 5: Remove unreachable static functions and globals.
fn remove_unreachable(module: &mut IrModule, func_id: &[u32], global_id: &[u32], reachable: &[bool], address_taken: &[bool]) {
    let mut func_pos = 0usize;
    module.functions.retain(|func| {
        let pos = func_pos;
        func_pos += 1;
        if func.is_declaration { return true; }
        let id = func_id[pos] as usize;
        if func.is_static && func.is_always_inline {
            return (id < address_taken.len() && address_taken[id])
                || (id < reachable.len() && reachable[id]);
        }
        if !func.is_static { return true; }
        id < reachable.len() && reachable[id]
    });

    let mut global_pos = 0usize;
    module.globals.retain(|global| {
        let pos = global_pos;
        global_pos += 1;
        if global.is_extern { return true; }
        if !global.is_static { return true; }
        if global.is_common { return true; }
        let id = global_id[pos] as usize;
        id < reachable.len() && reachable[id]
    });
}

/// Phase 6: Filter symbol_attrs to only keep directives for referenced symbols.
/// Visibility directives for unreferenced symbols cause assembler/linker errors.
fn filter_symbol_attrs(module: &mut IrModule) {
    let mut referenced_symbols: FxHashSet<&str> = FxHashSet::default();
    for func in &module.functions {
        if func.is_declaration { continue; }
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::Call { func: callee, .. } => {
                        referenced_symbols.insert(&**callee);
                    }
                    Instruction::GlobalAddr { name, .. } => {
                        referenced_symbols.insert(&**name);
                    }
                    Instruction::InlineAsm { input_symbols, .. } => {
                        for s in input_symbols.iter().flatten() {
                            let base = s.split('+').next().unwrap_or(s);
                            referenced_symbols.insert(base);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    for global in &module.globals {
        collect_global_init_refs_set(&global.init, &mut referenced_symbols);
    }
    for func in &module.functions {
        referenced_symbols.insert(&*func.name);
    }
    for global in &module.globals {
        referenced_symbols.insert(&*global.name);
    }

    module.symbol_attrs.retain(|(name, is_weak, visibility)| {
        if *is_weak && visibility.is_none() {
            return true;
        }
        referenced_symbols.contains(&**name)
    });
}

/// Extract symbol references from a single instruction into the ID list.
///
/// Shared by the reference-collection phase to avoid repeating the same
/// `match inst { Call | GlobalAddr | InlineAsm }` pattern.
fn collect_instruction_symbol_refs<'a>(
    inst: &'a Instruction,
    name_to_id: &mut FxHashMap<&'a str, u32>,
    next_id: &mut u32,
    refs: &mut Vec<u32>,
) {
    match inst {
        Instruction::Call { func: callee, .. } => {
            refs.push(get_or_create_id(callee, name_to_id, next_id));
        }
        Instruction::GlobalAddr { name, .. } => {
            refs.push(get_or_create_id(name, name_to_id, next_id));
        }
        Instruction::InlineAsm { input_symbols, .. } => {
            for s in input_symbols.iter().flatten() {
                let base = s.split('+').next().unwrap_or(s);
                refs.push(get_or_create_id(base, name_to_id, next_id));
            }
        }
        _ => {}
    }
}

/// Collect symbol references from a global initializer into a HashSet of borrowed strings.
///
/// This requires an explicit lifetime annotation since the borrowed `&str` references come
/// from the `GlobalInit`'s owned String fields, which is what `GlobalInit::for_each_ref`
/// cannot express through its closure-based API.
fn collect_global_init_refs_set<'a>(init: &'a GlobalInit, refs: &mut FxHashSet<&'a str>) {
    match init {
        GlobalInit::GlobalAddr(name) | GlobalInit::GlobalAddrOffset(name, _) => {
            refs.insert(&**name);
        }
        GlobalInit::GlobalLabelDiff(label1, label2, _) => {
            refs.insert(&**label1);
            refs.insert(&**label2);
        }
        GlobalInit::Compound(fields) => {
            for field in fields {
                collect_global_init_refs_set(field, refs);
            }
        }
        _ => {}
    }
}
