//! Global store forwarding pass.
//!
//! Tracks register→slot mappings across the function, forwarding stored values
//! to subsequent loads. At a label reached only by fallthrough (not a jump
//! target), register state from the previous instruction is fully known,
//! so we can safely forward across such labels.
//!
//! For labels that ARE jump targets, all mappings are invalidated because the
//! jump source may have different register values.

use super::super::types::*;
use super::helpers::*;

// ── Data structures ──────────────────────────────────────────────────────────

/// A tracked store mapping: we know that stack slot at `offset` contains the
/// value that was in register `reg_id` with the given `size`.
#[derive(Clone, Copy)]
struct SlotMapping {
    reg_id: RegId,
    size: MoveSize,
}

/// A slot entry for flat-array store forwarding.
#[derive(Clone, Copy)]
struct SlotEntry {
    offset: i32,
    mapping: SlotMapping,
    active: bool,
}

/// Small inline vector for register->offset tracking (avoids heap allocation
/// for the common case of <=4 offsets per register).
#[derive(Clone, Default)]
struct SmallVec {
    inline: [i32; 4],
    len: u8,
    overflow: Option<Vec<i32>>,
}

impl SmallVec {
    #[inline]
    fn push(&mut self, val: i32) {
        if let Some(ref mut ov) = self.overflow {
            ov.push(val);
        } else if (self.len as usize) < 4 {
            self.inline[self.len as usize] = val;
            self.len += 1;
        } else {
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&self.inline[..4]);
            v.push(val);
            self.overflow = Some(v);
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
        self.overflow = None;
    }

    #[inline]
    fn remove_val(&mut self, val: i32) {
        if let Some(ref mut ov) = self.overflow {
            ov.retain(|&v| v != val);
        } else {
            let n = self.len as usize;
            for j in 0..n {
                if self.inline[j] == val {
                    self.inline[j] = self.inline[n - 1];
                    self.len -= 1;
                    return;
                }
            }
        }
    }

    #[inline]
    fn iter(&self) -> SmallVecIter<'_> {
        SmallVecIter { sv: self, idx: 0 }
    }
}

struct SmallVecIter<'a> {
    sv: &'a SmallVec,
    idx: usize,
}

impl<'a> Iterator for SmallVecIter<'a> {
    type Item = i32;
    #[inline]
    fn next(&mut self) -> Option<i32> {
        if let Some(ref ov) = self.sv.overflow {
            if self.idx < ov.len() {
                let v = ov[self.idx];
                self.idx += 1;
                Some(v)
            } else {
                None
            }
        } else if self.idx < self.sv.len as usize {
            let v = self.sv.inline[self.idx];
            self.idx += 1;
            Some(v)
        } else {
            None
        }
    }
}

// JumpTargets struct is now in helpers.rs
use super::helpers::JumpTargets;

// ── State management helpers ─────────────────────────────────────────────────

/// Clear all slot→register mappings.
#[inline]
fn invalidate_all_mappings(slot_entries: &mut Vec<SlotEntry>, reg_offsets: &mut [SmallVec; 16]) {
    slot_entries.clear();
    for rs in reg_offsets.iter_mut() {
        rs.clear();
    }
}

/// Deactivate a single slot entry and remove its offset from the per-register tracking.
#[inline]
fn deactivate_entry(entry: &mut SlotEntry, reg_offsets: &mut [SmallVec; 16]) {
    let old_reg = entry.mapping.reg_id;
    entry.active = false;
    reg_offsets[old_reg as usize].remove_val(entry.offset);
}

/// Invalidate slot mappings at a given offset.
fn invalidate_slots_at(
    slot_entries: &mut [SlotEntry], reg_offsets: &mut [SmallVec; 16],
    offset: i32, access_size: i32,
) {
    for entry in slot_entries.iter_mut().filter(|e| e.active) {
        let hit = if access_size == 0 {
            entry.offset == offset
        } else {
            ranges_overlap(offset, access_size, entry.offset, entry.mapping.size.byte_size())
        };
        if hit {
            deactivate_entry(entry, reg_offsets);
        }
    }
}

/// Remove all slot mappings backed by a given register (flat array version).
fn invalidate_reg_flat(
    slot_entries: &mut [SlotEntry],
    reg_offsets: &mut [SmallVec; 16],
    reg_id: RegId,
) {
    let offsets = &reg_offsets[reg_id as usize];
    for offset in offsets.iter() {
        for entry in slot_entries.iter_mut().rev() {
            if entry.active && entry.offset == offset && entry.mapping.reg_id == reg_id {
                entry.active = false;
                break;
            }
        }
    }
    reg_offsets[reg_id as usize].clear();
}

// collect_jump_targets is now in helpers.rs
use super::helpers::collect_jump_targets;

// ── Per-instruction handlers ─────────────────────────────────────────────────

fn gsf_handle_label(
    store: &LineStore, infos: &[LineInfo], i: usize,
    targets: &JumpTargets,
    slot_entries: &mut Vec<SlotEntry>, reg_offsets: &mut [SmallVec; 16],
    prev_was_unconditional_jump: bool,
) {
    let label_name = infos[i].trimmed(store.get(i));
    let is_target = if let Some(n) = parse_label_number(label_name) {
        (n as usize) < targets.is_jump_target.len() && targets.is_jump_target[n as usize]
    } else {
        targets.has_non_numeric_jump_targets
    };
    if is_target || prev_was_unconditional_jump {
        invalidate_all_mappings(slot_entries, reg_offsets);
    }
}

fn gsf_handle_store(
    reg: RegId, offset: i32, size: MoveSize,
    slot_entries: &mut Vec<SlotEntry>, reg_offsets: &mut [SmallVec; 16],
) {
    invalidate_slots_at(slot_entries, reg_offsets, offset, size.byte_size());
    if is_valid_gp_reg(reg) {
        slot_entries.push(SlotEntry {
            offset,
            mapping: SlotMapping { reg_id: reg, size },
            active: true,
        });
        reg_offsets[reg as usize].push(offset);
    }
    if slot_entries.len() > 64 {
        slot_entries.retain(|e| e.active);
    }
}

fn gsf_handle_load(
    store: &mut LineStore, infos: &mut [LineInfo], i: usize,
    load_reg: RegId, load_offset: i32, load_size: MoveSize,
    slot_entries: &mut [SlotEntry], reg_offsets: &mut [SmallVec; 16],
) -> bool {
    let mut changed = false;
    let mapping = slot_entries.iter().rev()
        .find(|e| e.active && e.offset == load_offset)
        .map(|e| e.mapping);
    if let Some(mapping) = mapping {
        if mapping.size == load_size && mapping.reg_id != REG_NONE {
            let is_epilogue_restore = matches!(load_reg, 3 | 12 | 13 | 14 | 15)
                && load_offset < 0
                && is_near_epilogue(infos, i);
            if load_reg == mapping.reg_id && !is_epilogue_restore {
                mark_nop(&mut infos[i]);
                changed = true;
            } else if load_reg != REG_NONE && load_reg != mapping.reg_id {
                let store_reg_str = reg_id_to_name(mapping.reg_id, load_size);
                let load_reg_str = reg_id_to_name(load_reg, load_size);
                let new_text = format!("    {} {}, {}",
                    load_size.mnemonic(), store_reg_str, load_reg_str);
                replace_line(store, &mut infos[i], i, new_text);
                changed = true;
            }
        }
    }
    if is_valid_gp_reg(load_reg) {
        invalidate_reg_flat(slot_entries, reg_offsets, load_reg);
    }
    changed
}

fn gsf_handle_other(
    store: &LineStore, infos: &[LineInfo], i: usize, dest_reg: RegId,
    slot_entries: &mut Vec<SlotEntry>, reg_offsets: &mut [SmallVec; 16],
) {
    if is_valid_gp_reg(dest_reg) {
        invalidate_reg_flat(slot_entries, reg_offsets, dest_reg);
        if dest_reg == 0 {
            let trimmed = infos[i].trimmed(store.get(i));
            if trimmed.starts_with("div") || trimmed.starts_with("idiv")
                || trimmed.starts_with("mul")
                || trimmed == "cqto" || trimmed == "cqo" || trimmed == "cdq"
            {
                invalidate_reg_flat(slot_entries, reg_offsets, 2);
            }
        }
    }

    if dest_reg == REG_NONE && infos[i].rbp_offset != RBP_OFFSET_NONE {
        invalidate_slots_at(slot_entries, reg_offsets, infos[i].rbp_offset, 0);
    }

    if infos[i].has_indirect_mem {
        invalidate_all_mappings(slot_entries, reg_offsets);
    } else if infos[i].rbp_offset != RBP_OFFSET_NONE {
        invalidate_slots_at(slot_entries, reg_offsets, infos[i].rbp_offset, 1);
    }
}

// ── Main entry point ─────────────────────────────────────────────────────────

pub(super) fn global_store_forwarding(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let len = store.len();
    if len == 0 {
        return false;
    }

    let jump_targets = collect_jump_targets(store, infos, len);

    let mut slot_entries: Vec<SlotEntry> = Vec::new();
    let mut reg_offsets: [SmallVec; 16] = Default::default();
    let mut changed = false;
    let mut prev_was_unconditional_jump = false;

    for i in 0..len {
        if infos[i].is_nop() || infos[i].kind == LineKind::Empty {
            continue;
        }

        let was_uncond_jump = prev_was_unconditional_jump;
        prev_was_unconditional_jump = false;

        match infos[i].kind {
            LineKind::Label => {
                gsf_handle_label(store, infos, i, &jump_targets,
                    &mut slot_entries, &mut reg_offsets, was_uncond_jump);
            }

            LineKind::StoreRbp { reg, offset, size } => {
                gsf_handle_store(reg, offset, size,
                    &mut slot_entries, &mut reg_offsets);
            }

            LineKind::LoadRbp { reg: load_reg, offset: load_offset, size: load_size } => {
                changed |= gsf_handle_load(store, infos, i, load_reg, load_offset, load_size,
                    &mut slot_entries, &mut reg_offsets);
            }

            LineKind::Jmp | LineKind::JmpIndirect | LineKind::Ret => {
                invalidate_all_mappings(&mut slot_entries, &mut reg_offsets);
                prev_was_unconditional_jump = true;
            }

            LineKind::Call => {
                for &r in &[0u8, 1, 2, 6, 7, 8, 9, 10, 11] {
                    invalidate_reg_flat(&mut slot_entries, &mut reg_offsets, r);
                }
            }

            LineKind::Pop { reg } | LineKind::SetCC { reg } => {
                if is_valid_gp_reg(reg) {
                    invalidate_reg_flat(&mut slot_entries, &mut reg_offsets, reg);
                }
            }

            LineKind::Other { dest_reg } => {
                gsf_handle_other(store, infos, i, dest_reg,
                    &mut slot_entries, &mut reg_offsets);
            }

            LineKind::CondJmp | LineKind::Cmp | LineKind::Push { .. }
            | LineKind::Directive => {}

            _ => {}
        }
    }

    changed
}
