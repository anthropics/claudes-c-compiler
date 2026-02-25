//! Inline assembly symbol resolution.
//!
//! After inlining, callee functions like `_static_cpu_has(u16 bit)` may have
//! "i" (immediate) constraint operands like `&boot_cpu_data.x86_capability[bit >> 3]`
//! that couldn't be resolved at lowering time because `bit` was a parameter.
//! After inlining with a constant argument and running mem2reg + constant folding,
//! the IR contains a def chain like:
//!
//! ```text
//!   v1 = GlobalAddr @boot_cpu_data
//!   v2 = GEP v1, Const(74)
//! ```
//!
//! This module traces such chains and sets `input_symbols[i] = "boot_cpu_data+74"`
//! so the backend can emit correct symbol references in inline assembly.

use crate::common::fx_hash::FxHashMap;
use crate::ir::reexports::{IrFunction, IrModule, Instruction, Operand, Value};
use std::rc::Rc;

/// Resolve InlineAsm input symbols across all functions in the module.
pub(crate) fn resolve_inline_asm_symbols(module: &mut IrModule) {
    for func in &mut module.functions {
        if func.is_declaration || func.blocks.is_empty() {
            continue;
        }
        resolve_in_function(func);
    }
}

/// Information about a value's defining instruction, used for symbol resolution.
enum DefInfo {
    GlobalAddr(Rc<str>),
    Gep(Value, Operand),
    Add(Operand, Operand),
    Cast(Operand),
}

fn resolve_in_function(func: &mut IrFunction) {
    // Build a map from Value ID to its defining instruction for fast lookup.
    let mut value_defs: FxHashMap<u32, DefInfo> = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::GlobalAddr { dest, name } => {
                    value_defs.insert(dest.0, DefInfo::GlobalAddr(name.clone()));
                }
                Instruction::GetElementPtr { dest, base, offset, .. } => {
                    value_defs.insert(dest.0, DefInfo::Gep(*base, *offset));
                }
                Instruction::BinOp { dest, op: crate::ir::reexports::IrBinOp::Add, lhs, rhs, .. } => {
                    value_defs.insert(dest.0, DefInfo::Add(*lhs, *rhs));
                }
                Instruction::Cast { dest, src, .. } => {
                    value_defs.insert(dest.0, DefInfo::Cast(*src));
                }
                Instruction::Copy { dest, src } => {
                    value_defs.insert(dest.0, DefInfo::Cast(*src));
                }
                _ => {}
            }
        }
    }

    // Now scan InlineAsm instructions and try to resolve unresolved input_symbols.
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if let Instruction::InlineAsm { inputs, input_symbols, .. } = inst {
                for (i, (constraint, operand, _)) in inputs.iter().enumerate() {
                    // Only process "i" constraint inputs that don't have a symbol yet
                    if i >= input_symbols.len() { break; }
                    if input_symbols[i].is_some() { continue; }
                    // Only for immediate-only constraints
                    if !crate::backend::inline_asm::constraint_is_immediate_only(constraint) {
                        continue;
                    }
                    // Try to resolve the operand to a symbol+offset
                    if let Operand::Value(v) = operand {
                        if let Some(sym) = try_resolve_global_symbol(v, &value_defs) {
                            input_symbols[i] = Some(sym);
                        }
                    }
                }
            }
        }
    }
}

/// Try to resolve a Value to a global symbol + constant offset string.
/// Returns e.g., "boot_cpu_data" or "boot_cpu_data+74".
fn try_resolve_global_symbol(val: &Value, defs: &FxHashMap<u32, DefInfo>) -> Option<String> {
    let (name, offset) = try_resolve_global_with_offset(val, defs, 0)?;
    if offset == 0 {
        Some(name)
    } else if offset > 0 {
        Some(format!("{}+{}", name, offset))
    } else {
        Some(format!("{}{}", name, offset))
    }
}

/// Recursively trace a Value back through its def chain to find
/// GlobalAddr + constant offsets (from GEP, Add, Cast/Copy).
fn try_resolve_global_with_offset(val: &Value, defs: &FxHashMap<u32, DefInfo>, accum_offset: i64) -> Option<(String, i64)> {
    let def = defs.get(&val.0)?;
    match def {
        DefInfo::GlobalAddr(name) => Some(((*name).to_string(), accum_offset)),
        DefInfo::Gep(base, offset) => {
            let off = match offset {
                Operand::Const(c) => c.to_i64()?,
                Operand::Value(_) => {
                    // Non-constant GEP offset - can't resolve to symbol+offset
                    return None;
                }
            };
            try_resolve_global_with_offset(base, defs, accum_offset + off)
        }
        DefInfo::Add(lhs, rhs) => {
            // Pattern: base + const_offset or const_offset + base
            match (lhs, rhs) {
                (Operand::Value(base), Operand::Const(c)) => {
                    let off = c.to_i64()?;
                    try_resolve_global_with_offset(base, defs, accum_offset + off)
                }
                (Operand::Const(c), Operand::Value(base)) => {
                    let off = c.to_i64()?;
                    try_resolve_global_with_offset(base, defs, accum_offset + off)
                }
                _ => None,
            }
        }
        DefInfo::Cast(src) => {
            // Look through casts and copies
            match src {
                Operand::Value(v) => try_resolve_global_with_offset(v, defs, accum_offset),
                _ => None,
            }
        }
    }
}
