//! Inline assembly statement lowering.
//!
//! Processes GCC-style `asm(template : outputs : inputs : clobbers : goto_labels)`
//! statements into IR InlineAsm instructions. Handles:
//! - Output operand processing (including "+" read-write constraints)
//! - Input operand processing (immediate, memory, register constraints)
//! - Register variable constraint rewriting (`register x asm("reg")`)
//! - Symbol name extraction for `%P`/`%c`/`%a` modifiers
//! - Address space detection for segment-override operands
//! - Goto label resolution

use std::rc::Rc;
use crate::frontend::parser::ast::{AsmOperand, Expr};
use crate::ir::reexports::{
    BlockId,
    Instruction,
    Operand,
    Value,
};
use crate::common::types::{AddressSpace, IrType};
use super::lower::Lowerer;
use super::definitions::LValue;
use crate::backend::inline_asm::{constraint_has_immediate_alt, constraint_is_memory_only, constraint_needs_address};

impl Lowerer {
    pub(super) fn lower_inline_asm_stmt(
        &mut self,
        template: &str,
        outputs: &[AsmOperand],
        inputs: &[AsmOperand],
        clobbers: &[String],
        goto_labels: &[Rc<str>],
    ) {
        let mut ir_outputs = Vec::new();
        let mut ir_inputs = Vec::new();
        let mut operand_types = Vec::new();
        let mut seg_overrides = Vec::new();

        // Process output operands and synthetic "+" inputs
        let mut plus_input_types = Vec::new();
        let mut plus_input_segs = Vec::new();
        let mut plus_input_symbols: Vec<Option<String>> = Vec::new();
        for out in outputs {
            let mut constraint = out.constraint.clone();
            let name = out.name.clone();
            // Rewrite output constraint for register variables with __asm__("regname")
            if let Expr::Identifier(ref var_name, _) = out.expr {
                if let Some(asm_reg) = self.get_asm_register(var_name) {
                    let stripped = constraint.trim_start_matches(['=', '+', '&', '%']);
                    if stripped.contains('r') || stripped == "g" {
                        let prefix: String = constraint.chars().take_while(|c| *c == '=' || *c == '+' || *c == '&' || *c == '%').collect();
                        constraint = format!("{}{{{}}}", prefix, asm_reg);
                    }
                }
                // Mark local register asm variables as "initialized" when used as
                // an inline asm output. This ensures subsequent reads of the variable
                // load from the alloca (which the asm output writes to) instead of
                // emitting a new inline asm to read the physical hardware register.
                if let Some(fs) = self.func_state.as_mut() {
                    if let Some(info) = fs.locals.get_mut(var_name) {
                        if info.asm_register.is_some() {
                            info.asm_register_has_init = true;
                        }
                    }
                }
            }
            let out_ty = self.asm_operand_ir_type(&self.expr_ctype(&out.expr));
            // Detect address space for memory operands (e.g., __seg_gs pointer dereferences)
            let out_seg = self.get_asm_operand_addr_space(&out.expr);
            let ptr = if let Some(lv) = self.lower_lvalue(&out.expr) {
                match lv {
                    LValue::Variable(v) | LValue::Address(v, _) => v,
                }
            } else if let Expr::Identifier(ref var_name, _) = out.expr {
                // Global register variables are pinned to specific hardware registers
                // via constraint rewriting above (e.g., "+r" -> "+{rsp}"). They have no
                // backing storage, so lower_lvalue returns None. Create a temporary alloca
                // to preserve GCC operand numbering -- without this, subsequent operand
                // references (e.g., %P4) would be off-by-one and unresolvable.
                // Note: local register variables now return their alloca from lower_lvalue,
                // so they are handled by the branch above.
                if self.get_asm_register(var_name).is_some() {
                    let tmp = self.fresh_value();
                    self.emit(Instruction::Alloca {
                        dest: tmp, ty: out_ty, size: out_ty.size(),
                        align: out_ty.align(), volatile: false,
                    });
                    tmp
                } else {
                    continue;
                }
            } else {
                // Non-lvalue expression used as asm output (e.g., compound literal
                // `(long){0}` used as throwaway output: `[tmp] "=&r"((long){0})`).
                // Create a temporary alloca to preserve operand numbering -- without
                // this, the operand is skipped and named references like %[tmp] in
                // the template would not be substituted.
                let tmp = self.fresh_value();
                self.emit(Instruction::Alloca {
                    dest: tmp, ty: out_ty, size: out_ty.size(),
                    align: out_ty.align(), volatile: false,
                });
                tmp
            };
            if constraint.contains('+') {
                // For global register variables (e.g., `register long x asm("rsp")`),
                // the alloca is just a placeholder and is uninitialized. We must read
                // the current register value via an inline asm instead of loading from
                // the uninitialized alloca, which would corrupt the register (especially
                // critical for stack pointer register variables).
                //
                // Local register variables (e.g., `register void *tos asm("r11")`)
                // have real allocas that hold their assigned values. For "+" constraints,
                // we load from the alloca (not the hardware register), because the
                // programmer may have assigned a value via `tos = expr;` and expects
                // the inline asm to see that value, not a stale hardware register.
                let is_global_reg = if let Expr::Identifier(ref var_name, _) = out.expr {
                    self.get_asm_register(var_name).is_some() && self.lower_lvalue(&out.expr).is_none()
                } else {
                    false
                };
                let stripped_for_mem_check = constraint.replace('+', "");
                let needs_address = constraint_needs_address(&stripped_for_mem_check, self.is_riscv(), self.is_arm());
                let input_operand = if is_global_reg {
                    if let Expr::Identifier(ref var_name, _) = &out.expr {
                        let asm_reg = self.get_asm_register(var_name)
                            .expect("global register variable must have an asm register");
                        self.read_global_register(&asm_reg, out_ty)
                    } else {
                        unreachable!("asm output for global register variable must be an identifier")
                    }
                } else if needs_address {
                    // For "+m" (memory-only read-write) and "+A" (RISC-V address for
                    // AMO/LR/SC) constraints, do NOT emit a Load of the current value.
                    // These constraints need the ADDRESS (lvalue), not the value (rvalue).
                    //
                    // For "+m": the inline asm reads/writes memory directly through the
                    // template. Emitting a Load can crash when the memory address is only
                    // valid with a segment prefix (e.g., %gs: for per-CPU variables).
                    //
                    // For "+A": the address is loaded into a register and formatted as
                    // "(reg)" for AMO instructions. Loading the value would cause the
                    // atomic op to use the memory CONTENTS as an address, leading to
                    // crashes (e.g., kernel clear_bit using flag bits as an address).
                    //
                    // We still need a placeholder operand for correct operand numbering.
                    Operand::Value(ptr)
                } else if matches!(self.expr_ctype(&out.expr), crate::common::types::CType::Vector(_, _)) {
                    // For "+x" read-write constraints on vector types, pass the alloca pointer
                    // directly instead of emitting a Load instruction. Vector types (e.g.,
                    // __attribute__((vector_size(16)))) are 128 bits but IrType::Ptr is only
                    // 64 bits, so a Load would truncate the value to 64 bits, zeroing the
                    // upper half of the XMM register. By passing the alloca directly, the
                    // backend's load_input_to_reg will use movdqu to load the full 128-bit
                    // vector from the alloca's stack slot.
                    Operand::Value(ptr)
                } else {
                    let cur_val = self.fresh_value();
                    self.emit(Instruction::Load { dest: cur_val, ptr, ty: out_ty, seg_override: out_seg });
                    Operand::Value(cur_val)
                };
                ir_inputs.push((constraint.replace('+', "").to_string(), input_operand, name.clone()));
                plus_input_types.push(out_ty);
                plus_input_segs.push(out_seg);
                // For "+m" constraints, extract the symbol name so the backend can emit
                // direct symbol(%rip) references instead of register-indirect addressing.
                let stripped_constraint = constraint.replace('+', "");
                if constraint_is_memory_only(&stripped_constraint, self.is_arm()) {
                    plus_input_symbols.push(self.extract_mem_operand_symbol(&out.expr));
                } else {
                    plus_input_symbols.push(None);
                }
            }
            ir_outputs.push((constraint, ptr, name));
            operand_types.push(out_ty);
            seg_overrides.push(out_seg);
        }
        for ty in plus_input_types {
            operand_types.push(ty);
        }
        for seg in plus_input_segs {
            seg_overrides.push(seg);
        }

        // Process input operands
        // Start with symbols for synthetic "+" inputs (from output operands),
        // since those appear first in ir_inputs.
        let mut input_symbols: Vec<Option<String>> = plus_input_symbols;
        for inp in inputs {
            let mut constraint = inp.constraint.clone();
            let name = inp.name.clone();
            // Rewrite constraint for register variables with __asm__("regname"):
            // when the constraint allows "r", pin to the exact requested register.
            if let Expr::Identifier(ref var_name, _) = inp.expr {
                if let Some(asm_reg) = self.get_asm_register(var_name) {
                    let stripped = constraint.trim_start_matches(['=', '+', '&', '%']);
                    if stripped.contains('r') || stripped == "g" {
                        constraint = format!("{{{}}}", asm_reg);
                    }
                }
            }
            let inp_ty = self.asm_operand_ir_type(&self.expr_ctype(&inp.expr));
            let inp_seg = self.get_asm_operand_addr_space(&inp.expr);
            let mut sym_name: Option<String> = None;
            let val = if constraint_has_immediate_alt(&constraint) {
                if let Some(const_val) = self.eval_const_expr(&inp.expr) {
                    Operand::Const(const_val)
                } else if let Some(s) = Self::peel_to_string_literal(&inp.expr) {
                    // String literals (possibly wrapped in casts) have assembly-time-
                    // constant addresses (their .rodata label), so they are valid "i"
                    // constraint immediates. Intern once and use the label as both the
                    // symbol name and the GlobalAddr operand.
                    let label = self.intern_string_literal(&s);
                    sym_name = Some(label.clone());
                    let dest = self.fresh_value();
                    self.emit(Instruction::GlobalAddr { dest, name: Rc::from(label) });
                    Operand::Value(dest)
                } else if let Some(const_op) = self.try_recover_local_const(&inp.expr, &constraint) {
                    // For immediate-alternative constraints like "rK", try to recover the
                    // constant value of a local variable by scanning recent Store instructions.
                    // This handles the common kernel pattern:
                    //   unsigned long __v = (unsigned long)(0ULL);
                    //   asm("csrrw %0, satp, %1" : "=r"(__v) : "rK"(__v) : "memory");
                    // Without this, __v is lowered as a stack load, causing stack memory
                    // accesses between CSR writes that crash when paging is enabled.
                    const_op
                } else {
                    sym_name = self.extract_symbol_name(&inp.expr);
                    self.lower_expr(&inp.expr)
                }
            } else if constraint_needs_address(&constraint, self.is_riscv(), self.is_arm()) {
                // For memory-only and address constraints (m, o, V), provide the
                // address (lvalue) rather than the loaded value (rvalue).
                // On AArch64, 'Q' is also memory-only (single base register addressing).
                // On x86, 'Q' is a register constraint (legacy byte register).
                // For memory constraints, also try to extract the symbol name so the
                // backend can emit direct symbol(%rip) references instead of loading
                // addresses into registers, which is critical when the inline asm
                // template adds a segment prefix like %%gs:.
                // For RISC-V "A" constraints (AMO/LR/SC), the address is loaded into
                // a register and formatted as "(reg)" in the template.
                if constraint_is_memory_only(&constraint, self.is_arm()) {
                    sym_name = self.extract_mem_operand_symbol(&inp.expr);
                }
                if let Some(lv) = self.lower_lvalue(&inp.expr) {
                    let ptr = match lv {
                        LValue::Variable(v) | LValue::Address(v, _) => v,
                    };
                    Operand::Value(ptr)
                } else {
                    self.lower_expr(&inp.expr)
                }
            } else {
                // For local register variables, load from the alloca instead of
                // reading the hardware register. The variable's value was written
                // to the alloca by prior C assignments (e.g., `tos = stack`), and
                // the constraint rewriting already ensures the value is placed in
                // the correct hardware register for the inline asm.
                // Global register variables have no alloca, so lower_expr correctly
                // reads the hardware register for those.
                if let Expr::Identifier(ref var_name, _) = inp.expr {
                    if let Some(alloca) = self.get_local_alloca(var_name) {
                        if self.get_asm_register(var_name).is_some() {
                            let dest = self.fresh_value();
                            self.emit(Instruction::Load { dest, ptr: alloca, ty: inp_ty, seg_override: inp_seg });
                            Operand::Value(dest)
                        } else {
                            self.lower_expr(&inp.expr)
                        }
                    } else {
                        self.lower_expr(&inp.expr)
                    }
                } else {
                    self.lower_expr(&inp.expr)
                }
            };
            ir_inputs.push((constraint, val, name));
            operand_types.push(inp_ty);
            input_symbols.push(sym_name);
            seg_overrides.push(inp_seg);
        }

        // Resolve goto labels
        let ir_goto_labels: Vec<(String, BlockId)> = goto_labels.iter().map(|name| {
            let block = self.get_or_create_user_label(name);
            (name.to_string(), block)
        }).collect();

        self.emit(Instruction::InlineAsm {
            template: template.to_string(),
            outputs: ir_outputs.clone(),
            inputs: ir_inputs.clone(),
            clobbers: clobbers.to_vec(),
            operand_types,
            goto_labels: ir_goto_labels,
            input_symbols,
            seg_overrides,
        });

        // Dead store elimination for inline asm output allocas.
        //
        // When an inline asm has a write-only output ("=r") that writes to an alloca,
        // and no input reads from that same alloca (e.g., because the input was promoted
        // to a constant by try_recover_local_const), the store that initialized the
        // alloca before the inline asm is dead. Removing it prevents stack accesses
        // in critical sections like the RISC-V set_satp_mode function where the MMU
        // is in identity-mapping mode and the stack is not mapped.
        //
        // Pattern:
        //   Store(const, alloca_v)          <-- dead store (to be removed)
        //   InlineAsm { outputs: [("=r", alloca_v)], inputs: [("rK", Const(0))], ... }
        self.eliminate_dead_stores_before_inline_asm(&ir_outputs, &ir_inputs);
    }

    /// Remove stores to write-only inline asm output allocas when the inline asm
    /// doesn't read from those allocas (input was promoted to constant/immediate).
    fn eliminate_dead_stores_before_inline_asm(
        &mut self,
        outputs: &[(String, Value, Option<String>)],
        inputs: &[(String, Operand, Option<String>)],
    ) {
        // Collect write-only output alloca values (constraint starts with "=" but not "+")
        let mut writeonly_allocas: Vec<Value> = Vec::new();
        for (constraint, ptr, _) in outputs {
            if constraint.starts_with('=') && !constraint.contains('+') {
                writeonly_allocas.push(*ptr);
            }
        }
        if writeonly_allocas.is_empty() {
            return;
        }

        // Check which output allocas are NOT read by any input
        // An input reads from an alloca if:
        //   - It's Operand::Value(v) where v == alloca (direct reference)
        //   - It's Operand::Value(v) where v is a Load from the alloca (found via recent instrs)
        let fs = self.func_state.as_ref()
            .expect("func_state must exist during asm lowering");
        let mut alloca_read_by_input: Vec<bool> = vec![false; writeonly_allocas.len()];

        // Build a set of Values that are Loads from our target allocas
        let mut load_from_alloca: Vec<(u32, usize)> = Vec::new(); // (load_dest_id, alloca_idx)
        for inst in fs.instrs.iter().rev().take(50) {
            if let Instruction::Load { dest, ptr, .. } = inst {
                for (i, alloca_v) in writeonly_allocas.iter().enumerate() {
                    if *ptr == *alloca_v {
                        load_from_alloca.push((dest.0, i));
                    }
                }
            }
        }

        for (_, val, _) in inputs {
            match val {
                Operand::Value(v) => {
                    // Direct reference to the alloca
                    for (i, alloca_v) in writeonly_allocas.iter().enumerate() {
                        if *v == *alloca_v {
                            alloca_read_by_input[i] = true;
                        }
                    }
                    // Reference via a Load from the alloca
                    for (load_dest, alloca_idx) in &load_from_alloca {
                        if v.0 == *load_dest {
                            alloca_read_by_input[*alloca_idx] = true;
                        }
                    }
                }
                Operand::Const(_) => {
                    // Constant inputs don't read from allocas
                }
            }
        }

        // For each unread output alloca, find and remove the preceding dead store
        let fs = self.func_state.as_mut()
            .expect("func_state must exist during asm lowering");
        let instrs_len = fs.instrs.len();
        // The InlineAsm is at instrs[instrs_len - 1], scan backwards from instrs_len - 2
        for (i, is_read) in alloca_read_by_input.iter().enumerate() {
            if *is_read {
                continue;
            }
            let target_alloca = writeonly_allocas[i];
            // Scan backwards (skip the InlineAsm itself)
            // Stop at function calls, other inline asm, or after 30 instructions
            let start = if instrs_len >= 2 { instrs_len - 2 } else { continue };
            let limit = start.saturating_sub(30);
            for idx in (limit..=start).rev() {
                match &fs.instrs[idx] {
                    Instruction::Store { ptr, .. } if *ptr == target_alloca => {
                        // Found the dead store - remove it
                        fs.instrs.remove(idx);
                        fs.instr_spans.remove(idx);
                        break;
                    }
                    // Stop at barriers: calls, other inline asm, or loads from this alloca
                    Instruction::Call { .. } | Instruction::InlineAsm { .. } => break,
                    Instruction::Load { ptr, .. } if *ptr == target_alloca => break,
                    _ => {}
                }
            }
        }
    }

    /// Extract a global symbol name (possibly with offset) from an expression,
    /// for use with inline asm `"i"` constraint operands and `%P`/`%c`/`%a`
    /// modifiers. Handles:
    /// - `func_name` (bare function identifier that is a known function or global)
    /// - `&var_name` (address-of global variable)
    /// - Casts of the above (e.g., `(void *)func_name`)
    /// - Complex global address expressions like `&((const char *)s.member)[N]`
    ///   which produce `"symbol+offset"` strings (e.g., `"boot_cpu_data+9"`)
    ///
    /// Returns `None` for local variables/parameters, since those are not valid
    /// assembly symbols and would produce invalid assembly if emitted literally.
    fn extract_symbol_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Identifier(name, _) => {
                // Only return the name if it is a global symbol or known function,
                // NOT a local variable or function parameter.
                if self.is_global_or_function(name) {
                    Some(name.to_string())
                } else {
                    None
                }
            }
            Expr::AddressOf(inner, _) => {
                if let Expr::Identifier(name, _) = inner.as_ref() {
                    if self.is_global_or_function(name) {
                        Some(name.to_string())
                    } else {
                        None
                    }
                } else {
                    // Try resolving complex address expressions like
                    // &((const char *)boot_cpu_data.x86_capability)[12 >> 3]
                    // via the global address evaluator.
                    self.extract_symbol_from_global_addr_expr(expr)
                }
            }
            Expr::Cast(_, inner, _) => self.extract_symbol_name(inner),
            _ => None,
        }
    }

    /// Peel through casts to find a narrow string literal. Returns the string
    /// content if the expression is a (possibly cast-wrapped) `StringLiteral`.
    /// Only handles narrow strings; wide/char16 literals are not expected in
    /// inline asm "i" constraint operands.
    fn peel_to_string_literal(expr: &Expr) -> Option<String> {
        match expr {
            Expr::StringLiteral(s, _) => Some(s.clone()),
            Expr::Cast(_, inner, _) => Self::peel_to_string_literal(inner),
            _ => None,
        }
    }

    /// Extract a global symbol name (with optional offset) from an expression used
    /// as a memory ("m") constraint operand in inline assembly.
    /// The expression is typically a dereference like `*(type*)(uintptr_t)(&global.field)`.
    /// We look through dereferences and casts to find the underlying address expression,
    /// then try to resolve it to a symbol+offset.
    fn extract_mem_operand_symbol(&self, expr: &Expr) -> Option<String> {
        match expr {
            // Dereference: look at the pointer expression
            Expr::Deref(inner, _) => {
                // The inner expression is a pointer. Try to get its symbol name as an address.
                self.extract_symbol_from_global_addr_expr(inner)
                    .or_else(|| self.extract_mem_operand_symbol(inner))
            }
            // Cast: look through it
            Expr::Cast(_, inner, _) => self.extract_mem_operand_symbol(inner),
            // Direct identifier: if it's a global, return its name
            Expr::Identifier(name, _) => {
                if self.is_global_or_function(name) {
                    Some(name.to_string())
                } else {
                    None
                }
            }
            // Address-of: try to resolve as global address expression
            Expr::AddressOf(_, _) => self.extract_symbol_from_global_addr_expr(expr),
            // Member access on a global struct
            Expr::MemberAccess(base, _, _) | Expr::PointerMemberAccess(base, _, _) => {
                // Try to resolve the full expression as a global address (will include offset)
                self.extract_symbol_from_global_addr_expr(&Expr::AddressOf(Box::new(expr.clone()), expr.span()))
                    .or_else(|| self.extract_mem_operand_symbol(base))
            }
            _ => None,
        }
    }

    /// Try to resolve an expression to a global symbol name with optional offset,
    /// using the global address expression evaluator. Returns strings like
    /// `"boot_cpu_data"` or `"boot_cpu_data+9"`.
    fn extract_symbol_from_global_addr_expr(&self, expr: &Expr) -> Option<String> {
        use crate::ir::reexports::GlobalInit;
        let init = self.eval_global_addr_expr(expr)?;
        match init {
            GlobalInit::GlobalAddr(name) => Some(name.to_string()),
            GlobalInit::GlobalAddrOffset(name, offset) => {
                if offset >= 0 {
                    Some(format!("{}+{}", name, offset))
                } else {
                    Some(format!("{}{}", name, offset))
                }
            }
            _ => None,
        }
    }

    /// Check whether a name refers to a global variable, known function, or
    /// enum constant (i.e., something that is a valid assembly-level symbol or
    /// compile-time constant), as opposed to a local variable or parameter.
    fn is_global_or_function(&self, name: &str) -> bool {
        // Check if it's a local variable/parameter first -- if so, it's NOT global
        if let Some(ref fs) = self.func_state {
            if fs.locals.contains_key(name) {
                return false;
            }
        }
        // It's a global if it's in the globals map, known functions, or enum constants
        self.globals.contains_key(name)
            || self.known_functions.contains(name)
            || self.types.enum_constants.contains_key(name)
    }

    /// Try to recover a constant value for a local variable used as an inline
    /// asm input with an immediate-alternative constraint (e.g., "rK", "rI").
    ///
    /// When a non-const local variable like `unsigned long __v = 0;` is used as
    /// an inline asm input with constraint "rK", `eval_const_expr` returns None
    /// because `__v` is not const-qualified. However, we can scan recent IR
    /// instructions to find the most recent Store to the variable's alloca and,
    /// if it stored a constant value that fits the constraint, return it as
    /// `Operand::Const`. This avoids unnecessary register materialization (and
    /// associated stack spills) for values that could be immediates.
    ///
    /// This is critical for the RISC-V Linux kernel's CSR macros like:
    ///   `unsigned long __v = 0; asm("csrrw %0, satp, %1" : "=r"(__v) : "rK"(__v));`
    /// Without this, the 0 is spilled to the stack between csrw/csrrw instructions,
    /// causing a fault when paging is enabled but the stack is not identity-mapped.
    fn try_recover_local_const(&self, expr: &Expr, _constraint: &str) -> Option<Operand> {
        let name = match expr {
            Expr::Identifier(name, _) => name,
            _ => return None,
        };
        let fs = self.func_state.as_ref()?;
        let local_info = fs.locals.get(name)?;
        let alloca = local_info.alloca;

        // Scan recent instructions backwards looking for a Store to this alloca
        // with a constant value. Limit the scan to avoid performance issues.
        for inst in fs.instrs.iter().rev().take(20) {
            match inst {
                Instruction::Store { val: Operand::Const(c), ptr, .. } if *ptr == alloca => {
                    return Some(Operand::Const(*c));
                }
                Instruction::Store { val: Operand::Value(v), ptr, .. } if *ptr == alloca => {
                    // The store holds a Value, not a direct constant. This commonly
                    // happens with casts like `(unsigned long)(5)` which produce:
                    //   Cast { dest: vN, src: Const(5), from_ty: I64, to_ty: U64 }
                    //   Store { val: Value(vN), ptr: alloca }
                    // Follow one level of indirection to check if the defining
                    // instruction is a Cast or Copy of a constant.
                    let defining = *v;
                    for def_inst in fs.instrs.iter().rev().take(20) {
                        match def_inst {
                            Instruction::Cast { dest, src: Operand::Const(c), .. }
                                if *dest == defining =>
                            {
                                return Some(Operand::Const(*c));
                            }
                            Instruction::Copy { dest, src: Operand::Const(c) }
                                if *dest == defining =>
                            {
                                return Some(Operand::Const(*c));
                            }
                            _ => {}
                        }
                    }
                    // Non-constant store, stop searching.
                    return None;
                }
                // If we find any other store to this alloca, stop.
                Instruction::Store { ptr, .. } if *ptr == alloca => {
                    return None;
                }
                _ => {}
            }
        }
        None
    }

    /// Look up the asm register name for a variable declared with
    /// `register <type> <name> __asm__("regname")`.
    /// Checks local variables first, then global register variables.
    pub(super) fn get_asm_register(&self, name: &str) -> Option<Rc<str>> {
        // Check locals first
        if let Some(reg) = self.func_state.as_ref()
            .and_then(|fs| fs.locals.get(name))
            .and_then(|info| info.asm_register.clone())
        {
            return Some(reg);
        }
        // Check globals (for global register variables like `current_stack_pointer`)
        self.globals.get(name)
            .and_then(|info| info.asm_register.clone())
    }

    /// Get the alloca for a local variable by name, if it exists.
    /// Returns None for global variables (including global register variables)
    /// and for unknown variable names.
    fn get_local_alloca(&self, name: &str) -> Option<Value> {
        self.func_state.as_ref()
            .and_then(|fs| fs.locals.get(name))
            .map(|info| info.alloca)
    }

    /// Detect address space for an inline asm operand expression.
    /// For expressions like `*(typeof(var) __seg_gs *)(uintptr_t)&var`,
    /// returns the address space from the pointer type in the deref.
    fn get_asm_operand_addr_space(&self, expr: &Expr) -> AddressSpace {
        match expr {
            Expr::Deref(inner, _) => self.get_addr_space_of_ptr_expr(inner),
            _ => AddressSpace::Default,
        }
    }

    /// Map a CType to an IrType for inline asm operand sizing.
    ///
    /// Unlike `IrType::from_ctype()` which maps aggregate types (vectors, structs)
    /// to `Ptr` (8 bytes on 64-bit), this preserves the actual size so the backend's
    /// inline asm emitter can choose the correct load/store width.
    ///
    /// For 128-bit types (GCC vectors with `__attribute__((vector_size(16)))` or
    /// struct-based NEON types like `uint8x16_t` that are 16 bytes), we return
    /// `IrType::I128` (size 16) so the ARM backend uses `ldr qN`/`str qN` instead
    /// of `fmov dN, xN` which only transfers 64 bits and zeroes the upper lane.
    fn asm_operand_ir_type(&self, ctype: &crate::common::types::CType) -> IrType {
        use crate::common::types::CType;
        match ctype {
            CType::Vector(_, total_size) if *total_size == 16 => IrType::I128,
            CType::Vector(_, total_size) if *total_size == 8 => IrType::I64,
            CType::Struct(_) | CType::Union(_) => {
                // For struct/union types used with NEON "w" constraints (e.g.,
                // uint8x16_t which is `struct { unsigned char __val[16]; }`),
                // preserve the actual size so the backend uses 128-bit loads/stores.
                let layouts = self.types.borrow_struct_layouts();
                let size = ctype.size_ctx(&*layouts);
                match size {
                    16 => IrType::I128,
                    8 => IrType::I64,
                    4 => IrType::I32,
                    _ => IrType::Ptr,
                }
            }
            _ => IrType::from_ctype(ctype),
        }
    }
}
