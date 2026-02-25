//! Shared use-def information for optimization passes.
//!
//! Built once per function before each optimization iteration and consumed
//! read-only by passes (DCE, narrow, etc.) that need use-counts or definition
//! locations. This eliminates redundant full-function scans that each pass
//! previously performed independently.
//!
//! The UseDefInfo is NOT incrementally maintained — passes that mutate the IR
//! invalidate it, and it's rebuilt on demand by the next consumer.

use crate::ir::reexports::{
    Instruction,
    IrFunction,
    Operand,
    Terminator,
};

/// Compact use-site: identifies where a value is used.
///
/// `inst_idx == u32::MAX` means the value is used in the block's terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UseLoc {
    pub block_idx: u32,
    pub inst_idx: u32,
}

impl UseLoc {
    pub const TERMINATOR: u32 = u32::MAX;

    #[inline]
    pub fn instruction(block: u32, inst: u32) -> Self {
        UseLoc { block_idx: block, inst_idx: inst }
    }

    #[inline]
    pub fn terminator(block: u32) -> Self {
        UseLoc { block_idx: block, inst_idx: Self::TERMINATOR }
    }

    #[inline]
    pub fn is_terminator(self) -> bool {
        self.inst_idx == Self::TERMINATOR
    }
}

/// Compact definition location encoded as a single u64.
///
/// Encoding:
/// - `u64::MAX` = no definition found (gap, external, or unknown)
/// - `(1 << 63) | param_idx` = function parameter
/// - `(block_idx << 32) | inst_idx` = instruction (block_idx < 2^31)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefLoc(u64);

impl DefLoc {
    const NONE_SENTINEL: u64 = u64::MAX;
    const PARAM_BIT: u64 = 1 << 63;

    /// No definition found.
    #[inline]
    pub fn none() -> Self {
        DefLoc(Self::NONE_SENTINEL)
    }

    /// Defined by an instruction at (block_idx, inst_idx).
    #[inline]
    pub fn instruction(block: u32, inst: u32) -> Self {
        DefLoc(((block as u64) << 32) | (inst as u64))
    }

    /// Defined as a function parameter.
    #[inline]
    pub fn parameter(idx: u32) -> Self {
        DefLoc(Self::PARAM_BIT | (idx as u64))
    }

    /// Returns true if this is the "no definition" sentinel.
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == Self::NONE_SENTINEL
    }

    /// If this is an instruction definition, returns `(block_idx, inst_idx)`.
    #[inline]
    pub fn as_instruction(self) -> Option<(u32, u32)> {
        if self.0 == Self::NONE_SENTINEL || (self.0 & Self::PARAM_BIT) != 0 {
            None
        } else {
            Some(((self.0 >> 32) as u32, self.0 as u32))
        }
    }
}

/// Per-function use-def information, built once before each optimization
/// iteration and consumed read-only by passes.
///
/// All arrays are indexed by Value ID (`Value.0 as usize`). Length is
/// `max_value_id + 1`, matching the pattern used by DCE, narrow, etc.
pub struct UseDefInfo {
    /// `use_count[v]` = number of times `Value(v)` appears as an operand.
    /// For Phi nodes, self-references are excluded (matching DCE behavior).
    /// Terminator uses are included.
    pub use_count: Vec<u32>,

    /// `def_loc[v]` = where `Value(v)` is defined.
    pub def_loc: Vec<DefLoc>,

    /// CSR offsets for use-chains. Length = `size + 1`.
    /// Uses of value `v` are: `use_sites[use_offsets[v] .. use_offsets[v+1]]`
    pub use_offsets: Vec<u32>,

    /// Flat array of use-sites, grouped by value ID (CSR data).
    pub use_sites: Vec<UseLoc>,
}

impl UseDefInfo {
    /// Build use-def information for a function in two passes.
    ///
    /// Pass 1: count uses + record def-locs (unchanged from before).
    /// Pass 2: prefix-sum use_count → use_offsets, then scan instructions
    /// again to fill the CSR use_sites array.
    ///
    /// Cost: O(instructions * avg_operands), two scans.
    pub fn build(func: &IrFunction) -> Self {
        let max_id = func.max_value_id() as usize;
        let size = max_id + 1;
        let mut use_count: Vec<u32> = vec![0; size];
        let mut def_loc: Vec<DefLoc> = vec![DefLoc::none(); size];

        // --- Pass 1: count uses + record def-locs ---
        for (bi, block) in func.blocks.iter().enumerate() {
            let bi32 = bi as u32;
            for (ii, inst) in block.instructions.iter().enumerate() {
                let ii32 = ii as u32;

                // Record definition location.
                if let Some(dest) = inst.dest() {
                    let id = dest.0 as usize;
                    if id < size {
                        if let Instruction::ParamRef { param_idx, .. } = inst {
                            def_loc[id] = DefLoc::parameter(*param_idx as u32);
                        } else {
                            def_loc[id] = DefLoc::instruction(bi32, ii32);
                        }
                    }
                }

                // Count uses, excluding Phi self-references (matching DCE).
                if let Instruction::Phi { dest, incoming, .. } = inst {
                    for (op, _) in incoming {
                        if let Operand::Value(v) = op {
                            if v.0 != dest.0 {
                                let idx = v.0 as usize;
                                if idx < size {
                                    use_count[idx] += 1;
                                }
                            }
                        }
                    }
                } else {
                    inst.for_each_used_value(|id| {
                        let idx = id as usize;
                        if idx < size {
                            use_count[idx] += 1;
                        }
                    });
                }
            }

            // Count terminator uses.
            block.terminator.for_each_used_value(|id| {
                let idx = id as usize;
                if idx < size {
                    use_count[idx] += 1;
                }
            });
        }

        // --- Pass 2: build CSR use-chains ---
        // Prefix-sum to build offsets.
        let mut use_offsets: Vec<u32> = vec![0; size + 1];
        for i in 0..size {
            use_offsets[i + 1] = use_offsets[i] + use_count[i];
        }
        let total_uses = use_offsets[size] as usize;
        let mut use_sites: Vec<UseLoc> = vec![UseLoc { block_idx: 0, inst_idx: 0 }; total_uses];

        // Cursor array tracks where to insert the next use for each value.
        let mut cursor: Vec<u32> = use_offsets[..size].to_vec();

        // Scan instructions again to fill use_sites.
        for (bi, block) in func.blocks.iter().enumerate() {
            let bi32 = bi as u32;
            for (ii, inst) in block.instructions.iter().enumerate() {
                let ii32 = ii as u32;
                let loc = UseLoc::instruction(bi32, ii32);

                if let Instruction::Phi { dest, incoming, .. } = inst {
                    for (op, _) in incoming {
                        if let Operand::Value(v) = op {
                            if v.0 != dest.0 {
                                let idx = v.0 as usize;
                                if idx < size {
                                    let pos = cursor[idx] as usize;
                                    use_sites[pos] = loc;
                                    cursor[idx] += 1;
                                }
                            }
                        }
                    }
                } else {
                    inst.for_each_used_value(|id| {
                        let idx = id as usize;
                        if idx < size {
                            let pos = cursor[idx] as usize;
                            use_sites[pos] = loc;
                            cursor[idx] += 1;
                        }
                    });
                }
            }

            // Terminator uses.
            let term_loc = UseLoc::terminator(bi32);
            block.terminator.for_each_used_value(|id| {
                let idx = id as usize;
                if idx < size {
                    let pos = cursor[idx] as usize;
                    use_sites[pos] = term_loc;
                    cursor[idx] += 1;
                }
            });
        }

        UseDefInfo { use_count, def_loc, use_offsets, use_sites }
    }

    /// Check if a value has no uses (use_count == 0).
    #[inline]
    pub fn is_dead(&self, v: u32) -> bool {
        let idx = v as usize;
        idx < self.use_count.len() && self.use_count[idx] == 0
    }

    /// Look up the instruction defining a value. Returns `None` if the value
    /// is a parameter, has no recorded definition, or is out of bounds.
    #[inline]
    pub fn def_inst<'a>(&self, v: u32, func: &'a IrFunction) -> Option<&'a Instruction> {
        let idx = v as usize;
        if idx >= self.def_loc.len() {
            return None;
        }
        let (bi, ii) = self.def_loc[idx].as_instruction()?;
        func.blocks.get(bi as usize)
            .and_then(|b| b.instructions.get(ii as usize))
    }

    /// Get all use-sites for a value as a slice. O(1).
    #[inline]
    pub fn uses_of(&self, v: u32) -> &[UseLoc] {
        let idx = v as usize;
        if idx + 1 >= self.use_offsets.len() {
            return &[];
        }
        let start = self.use_offsets[idx] as usize;
        let end = self.use_offsets[idx + 1] as usize;
        &self.use_sites[start..end]
    }

    /// Look up the instruction at a UseLoc. Returns `None` if the UseLoc
    /// refers to a terminator or is out of bounds.
    #[inline]
    pub fn use_inst<'a>(&self, loc: UseLoc, func: &'a IrFunction) -> Option<&'a Instruction> {
        if loc.is_terminator() {
            return None;
        }
        func.blocks.get(loc.block_idx as usize)
            .and_then(|b| b.instructions.get(loc.inst_idx as usize))
    }

    /// Look up the terminator at a UseLoc. Returns `None` if the UseLoc
    /// refers to an instruction (not a terminator) or is out of bounds.
    #[inline]
    pub fn use_terminator<'a>(&self, loc: UseLoc, func: &'a IrFunction) -> Option<&'a Terminator> {
        if !loc.is_terminator() {
            return None;
        }
        func.blocks.get(loc.block_idx as usize)
            .map(|b| &b.terminator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::reexports::*;
    use crate::common::types::IrType;

    /// Helper: create a minimal function with given blocks.
    fn make_func(blocks: Vec<BasicBlock>) -> IrFunction {
        let mut f = IrFunction::new("test".into(), IrType::Void, vec![], false);
        f.blocks = blocks;
        // Set next_value_id to cover all values
        let mut max = 0u32;
        for b in &f.blocks {
            for inst in &b.instructions {
                if let Some(v) = inst.dest() {
                    if v.0 > max { max = v.0; }
                }
            }
        }
        f.next_value_id = max + 1;
        f
    }

    #[test]
    fn test_def_loc_encoding() {
        let none = DefLoc::none();
        assert!(none.is_none());
        assert_eq!(none.as_instruction(), None);

        let inst = DefLoc::instruction(3, 7);
        assert!(!inst.is_none());
        assert_eq!(inst.as_instruction(), Some((3, 7)));

        let param = DefLoc::parameter(2);
        assert!(!param.is_none());
        assert_eq!(param.as_instruction(), None);
    }

    #[test]
    fn test_basic_use_count() {
        // %0 = Copy 42
        // %1 = BinOp Add %0, %0
        // return %1
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(42)),
                },
                Instruction::BinOp {
                    dest: Value(1),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(0)),
                    rhs: Operand::Value(Value(0)),
                    ty: IrType::I64,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: vec![],
        }];

        let func = make_func(blocks);
        let info = UseDefInfo::build(&func);

        // %0 used twice (lhs + rhs of BinOp)
        assert_eq!(info.use_count[0], 2);
        // %1 used once (return)
        assert_eq!(info.use_count[1], 1);

        // %0 defined at block 0, inst 0
        assert_eq!(info.def_loc[0].as_instruction(), Some((0, 0)));
        // %1 defined at block 0, inst 1
        assert_eq!(info.def_loc[1].as_instruction(), Some((0, 1)));

        assert!(!info.is_dead(0));
        assert!(!info.is_dead(1));
    }

    #[test]
    fn test_dead_value() {
        // %0 = Copy 42  (unused)
        // return void
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(42)),
                },
            ],
            terminator: Terminator::Return(None),
            source_spans: vec![],
        }];

        let func = make_func(blocks);
        let info = UseDefInfo::build(&func);

        assert_eq!(info.use_count[0], 0);
        assert!(info.is_dead(0));
    }

    #[test]
    fn test_basic_use_chains() {
        // %0 = Copy 42
        // %1 = BinOp Add %0, %0
        // return %1
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(42)),
                },
                Instruction::BinOp {
                    dest: Value(1),
                    op: IrBinOp::Add,
                    lhs: Operand::Value(Value(0)),
                    rhs: Operand::Value(Value(0)),
                    ty: IrType::I64,
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
            source_spans: vec![],
        }];

        let func = make_func(blocks);
        let info = UseDefInfo::build(&func);

        // %0 used twice (lhs + rhs of BinOp at block 0, inst 1)
        let uses0 = info.uses_of(0);
        assert_eq!(uses0.len(), 2);
        assert_eq!(uses0[0], UseLoc::instruction(0, 1));
        assert_eq!(uses0[1], UseLoc::instruction(0, 1));

        // %1 used once (return terminator of block 0)
        let uses1 = info.uses_of(1);
        assert_eq!(uses1.len(), 1);
        assert!(uses1[0].is_terminator());
        assert_eq!(uses1[0].block_idx, 0);

        // Accessor methods
        assert!(info.use_inst(uses0[0], &func).is_some());
        assert!(info.use_terminator(uses1[0], &func).is_some());
        assert!(info.use_inst(uses1[0], &func).is_none()); // terminator, not inst
    }

    #[test]
    fn test_phi_self_ref_use_chains() {
        // Block 0: %0 = Copy 0; branch -> Block 1
        // Block 1: %1 = Phi [(Block 0, %0), (Block 1, %1)]; branch -> Block 1
        // Self-ref (%1 -> %1) should NOT appear in use chains.
        let blocks = vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Copy {
                        dest: Value(0),
                        src: Operand::Const(IrConst::I64(0)),
                    },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![
                    Instruction::Phi {
                        dest: Value(1),
                        incoming: vec![
                            (Operand::Value(Value(0)), BlockId(0)),
                            (Operand::Value(Value(1)), BlockId(1)), // self-ref
                        ],
                        ty: IrType::I64,
                    },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: vec![],
            },
        ];

        let func = make_func(blocks);
        let info = UseDefInfo::build(&func);

        // %0 used once in Phi (block 1, inst 0)
        let uses0 = info.uses_of(0);
        assert_eq!(uses0.len(), 1);
        assert_eq!(uses0[0], UseLoc::instruction(1, 0));

        // %1 self-ref excluded: 0 uses
        let uses1 = info.uses_of(1);
        assert_eq!(uses1.len(), 0);
    }

    #[test]
    fn test_terminator_use_chains() {
        // %0 = Copy 1
        // condbranch %0, Block 1, Block 2
        let blocks = vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Copy {
                        dest: Value(0),
                        src: Operand::Const(IrConst::I32(1)),
                    },
                ],
                terminator: Terminator::CondBranch {
                    cond: Operand::Value(Value(0)),
                    true_label: BlockId(1),
                    false_label: BlockId(2),
                },
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![],
                terminator: Terminator::Return(None),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(2),
                instructions: vec![],
                terminator: Terminator::Return(None),
                source_spans: vec![],
            },
        ];

        let func = make_func(blocks);
        let info = UseDefInfo::build(&func);

        let uses0 = info.uses_of(0);
        assert_eq!(uses0.len(), 1);
        assert!(uses0[0].is_terminator());
        assert_eq!(uses0[0].block_idx, 0);
    }

    #[test]
    fn test_unused_value_empty_chain() {
        // %0 = Copy 42 (unused)
        // return void
        let blocks = vec![BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy {
                    dest: Value(0),
                    src: Operand::Const(IrConst::I64(42)),
                },
            ],
            terminator: Terminator::Return(None),
            source_spans: vec![],
        }];

        let func = make_func(blocks);
        let info = UseDefInfo::build(&func);

        assert_eq!(info.uses_of(0).len(), 0);
        assert!(info.is_dead(0));
        // Out-of-bounds value also returns empty
        assert_eq!(info.uses_of(999).len(), 0);
    }

    #[test]
    fn test_phi_self_ref_excluded() {
        // Block 0: %0 = Copy 0; branch -> Block 1
        // Block 1: %1 = Phi [(Block 0, %0), (Block 1, %1)]; branch -> Block 1
        // The self-ref (%1 -> %1) should NOT be counted.
        let blocks = vec![
            BasicBlock {
                label: BlockId(0),
                instructions: vec![
                    Instruction::Copy {
                        dest: Value(0),
                        src: Operand::Const(IrConst::I64(0)),
                    },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: vec![],
            },
            BasicBlock {
                label: BlockId(1),
                instructions: vec![
                    Instruction::Phi {
                        dest: Value(1),
                        incoming: vec![
                            (Operand::Value(Value(0)), BlockId(0)),
                            (Operand::Value(Value(1)), BlockId(1)), // self-ref
                        ],
                        ty: IrType::I64,
                    },
                ],
                terminator: Terminator::Branch(BlockId(1)),
                source_spans: vec![],
            },
        ];

        let func = make_func(blocks);
        let info = UseDefInfo::build(&func);

        // %0 used once (in Phi from Block 0)
        assert_eq!(info.use_count[0], 1);
        // %1: self-ref excluded, so only 0 external uses (phi not used by anything else)
        assert_eq!(info.use_count[1], 0);
        assert!(info.is_dead(1));
    }
}
