use super::*;
use crate::backend::arm::assembler::parser::Operand;

// ── System instructions ──────────────────────────────────────────────────

pub(crate) fn encode_dmb(operands: &[Operand]) -> Result<EncodeResult, String> {
    let option = match operands.first() {
        Some(Operand::Barrier(b)) | Some(Operand::Symbol(b)) => match b.to_lowercase().as_str() {
            "sy" => 0b1111u32,
            "st" => 0b1110,
            "ld" => 0b1101,
            "ish" => 0b1011,
            "ishst" => 0b1010,
            "ishld" => 0b1001,
            "nsh" => 0b0111,
            "nshst" => 0b0110,
            "nshld" => 0b0101,
            "osh" => 0b0011,
            "oshst" => 0b0010,
            "oshld" => 0b0001,
            _ => return Err(format!("unknown dmb option: {}", b)),
        },
        _ => 0b1111,
    };
    // DMB: 0xD50330BF | (CRm << 8)
    let word = 0xd50330bf | (option << 8);
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_dsb(operands: &[Operand]) -> Result<EncodeResult, String> {
    let option = match operands.first() {
        Some(Operand::Barrier(b)) | Some(Operand::Symbol(b)) => match b.to_lowercase().as_str() {
            "sy" => 0b1111u32,
            "st" => 0b1110,
            "ld" => 0b1101,
            "ish" => 0b1011,
            "ishst" => 0b1010,
            "ishld" => 0b1001,
            "nsh" => 0b0111,
            "nshst" => 0b0110,
            "nshld" => 0b0101,
            "osh" => 0b0011,
            "oshst" => 0b0010,
            "oshld" => 0b0001,
            _ => return Err(format!("unknown dsb option: {}", b)),
        },
        _ => 0b1111,
    };
    // DSB: 0xD503309F | (option << 8)
    let word = 0xd503309f | (option << 8);
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_mrs(operands: &[Operand]) -> Result<EncodeResult, String> {
    // MRS Xt, system_reg
    let (rt, _) = get_reg(operands, 0)?;
    let sysreg = match operands.get(1) {
        Some(Operand::Symbol(s)) => s.to_lowercase(),
        _ => return Err("mrs needs system register name".to_string()),
    };

    let encoding = match sysreg.as_str() {
        "sp_el0" => 0xc208u32,
        "tpidr_el0" => 0xde82,
        "tpidr_el1" => 0xc684,
        "tpidr_el2" => 0xe682,
        "tpidrro_el0" => 0xde83,
        "tcr_el1" => 0xc102,
        "ttbr0_el1" => 0xc100,
        "sctlr_el1" => 0xc080,
        "mdscr_el1" => 0x8012,
        "id_aa64mmfr0_el1" => 0xc038,
        "id_aa64mmfr1_el1" => 0xc039,
        "cpacr_el1" => 0xc082,
        "par_el1" => 0xc3a0,
        "osdlr_el1" => 0x809c,
        "currentel" => 0xc212,
        "elr_el1" => 0xc201,
        "spsr_el1" => 0xc200,
        "esr_el1" => 0xc290,
        "far_el1" => 0xc300,
        "vbar_el1" => 0xc600,
        "mpidr_el1" => 0xc005,
        "contextidr_el1" => 0xc681,
        "mair_el1" => 0xc510,
        "isr_el1" => 0xc608,
        "oslsr_el1" => 0x808c,
        "midr_el1" => 0xc000,
        "revidr_el1" => 0xc006,
        "id_aa64pfr0_el1" => 0xc020,
        "id_aa64pfr1_el1" => 0xc021,
        "id_aa64isar0_el1" => 0xc030,
        "id_aa64isar1_el1" => 0xc031,
        "id_aa64isar2_el1" => 0xc032,
        "amair_el1" => 0xc518,
        "hcr_el2" => 0xe088,
        "cptr_el2" => 0xe08a,
        "hstr_el2" => 0xe08b,
        "hacr_el2" => 0xe08f,
        "vpidr_el2" => 0xe000,
        "vmpidr_el2" => 0xe005,
        "actlr_el2" => 0xe081,
        "elr_el2" => 0xe201,
        "esr_el2" => 0xe290,
        "afsr0_el2" => 0xe288,
        "afsr1_el2" => 0xe289,
        "far_el2" => 0xe300,
        "hpfar_el2" => 0xe304,
        "spsr_el2" => 0xe200,
        "sctlr_el2" => 0xe080,
        "mdcr_el2" => 0xe089,
        "tcr_el2" => 0xe102,
        "ttbr0_el2" => 0xe100,
        "vttbr_el2" => 0xe108,
        "vtcr_el2" => 0xe10a,
        "vbar_el2" => 0xe600,
        "mair_el2" => 0xe510,
        "amair_el2" => 0xe518,
        "sp_el1" => 0xe208,
        "pmuserenr_el0" => 0xdcf0,
        "cntfrq_el0" => 0xdf00,
        "cntpct_el0" => 0xdf01,
        "cntv_ctl_el0" => 0xdf19,
        "cntp_ctl_el0" => 0xdf11,
        "cntv_cval_el0" => 0xdf1c,
        "cntp_cval_el0" => 0xdf12,
        "ctr_el0" => 0xd801,
        "ttbr1_el1" => 0xc101,
        "cntkctl_el1" => 0xc708,
        "id_aa64dfr0_el1" => 0xc028,
        "oslar_el1" => 0x8084,
        "cntvct_el0" => 0xdf02,
        "clidr_el1" => 0xc801,
        "ccsidr_el1" => 0xc800,
        "csselr_el1" => 0xd000,
        "id_aa64mmfr2_el1" => 0xc03a,
        "id_aa64dfr1_el1" => 0xc029,
        "actlr_el1" => 0xc081,
        "afsr0_el1" => 0xc288,
        "afsr1_el1" => 0xc289,
        "id_pfr0_el1" => 0xc008,
        "id_pfr1_el1" => 0xc009,
        "cnthctl_el2" => 0xe708,
        "cntvoff_el2" => 0xe703,
        "sp_el2" => 0xf208,
        "pmintenset_el1" => 0xc4f1,
        "pmintenclr_el1" => 0xc4f2,
        "pmcr_el0" => 0xdce0,
        "pmcntenset_el0" => 0xdce1,
        "pmcntenclr_el0" => 0xdce2,
        "pmovsclr_el0" => 0xdce3,
        "pmselr_el0" => 0xdce5,
        "pmceid0_el0" => 0xdce6,
        "pmceid1_el0" => 0xdce7,
        "pmccntr_el0" => 0xdce8,
        "pmxevtyper_el0" => 0xdce9,
        "pmxevcntr_el0" => 0xdcea,
        "pmccfiltr_el0" => 0xdf7f,
        "dczid_el0" => 0xd807,
        "daif" => 0xda11,
        "fpcr" => 0xda20,
        "fpsr" => 0xda21,
        "nzcv" => 0xda10,
        "spsel" => 0xc210,
        "mdccint_el1" => 0x8010,
        "fpexc32_el2" => 0xe298,
        "dbgauthstatus_el1" => 0x83f6,
        "spsr_abt" => 0xe219,
        "spsr_und" => 0xe21a,
        "spsr_irq" => 0xe218,
        "spsr_fiq" => 0xe21b,
        "ifsr32_el2" => 0xe281,
        "dacr32_el2" => 0xe180,
        _ => parse_generic_sysreg(&sysreg)?,
    };

    // MRS encoding: 0xd520_0000 has L=1 (bit 21) for read.
    // Bits [20:19] = op0, supplied entirely by the sysreg encoding field.
    let word = 0xd5200000 | (encoding << 5) | rt;
    Ok(EncodeResult::Word(word))
}

/// Compute sysreg encoding from (op0, op1, CRn, CRm, op2) fields.
pub(crate) fn sysreg_encoding(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    ((op0 & 3) << 14) | ((op1 & 7) << 11) | ((crn & 0xF) << 7) | ((crm & 0xF) << 3) | (op2 & 7)
}

/// Try to parse a numbered debug/performance register family name like
/// `dbgbcr15_el1` or `dbgwvr0_el1` into its encoding. Returns None if not matched.
pub(crate) fn parse_numbered_sysreg(name: &str) -> Option<u32> {
    // Debug breakpoint/watchpoint registers: dbg{b,w}{c,v}r<n>_el1
    // dbgbcr<n>_el1: op0=2, op1=0, CRn=0, CRm=n, op2=5
    // dbgbvr<n>_el1: op0=2, op1=0, CRn=0, CRm=n, op2=4
    // dbgwcr<n>_el1: op0=2, op1=0, CRn=0, CRm=n, op2=7
    // dbgwvr<n>_el1: op0=2, op1=0, CRn=0, CRm=n, op2=6
    let prefixes: &[(&str, &str, u32)] = &[
        ("dbgbcr", "_el1", 5),
        ("dbgbvr", "_el1", 4),
        ("dbgwcr", "_el1", 7),
        ("dbgwvr", "_el1", 6),
    ];
    for &(prefix, suffix, op2) in prefixes {
        if let Some(rest) = name.strip_prefix(prefix) {
            if let Some(num_str) = rest.strip_suffix(suffix) {
                if let Ok(n) = num_str.parse::<u32>() {
                    if n <= 15 {
                        return Some(sysreg_encoding(2, 0, 0, n, op2));
                    }
                }
            }
        }
    }

    // Performance monitor event count registers: pmevcntr<n>_el0, pmevtyper<n>_el0
    // pmevcntr<n>_el0: op0=3, op1=3, CRn=14, CRm=8+n/8, op2=n%8
    // pmevtyper<n>_el0: op0=3, op1=3, CRn=14, CRm=12+n/8, op2=n%8
    if let Some(rest) = name.strip_prefix("pmevcntr") {
        if let Some(num_str) = rest.strip_suffix("_el0") {
            if let Ok(n) = num_str.parse::<u32>() {
                if n <= 30 {
                    return Some(sysreg_encoding(3, 3, 14, 8 + n / 8, n % 8));
                }
            }
        }
    }
    if let Some(rest) = name.strip_prefix("pmevtyper") {
        if let Some(num_str) = rest.strip_suffix("_el0") {
            if let Ok(n) = num_str.parse::<u32>() {
                if n <= 30 {
                    return Some(sysreg_encoding(3, 3, 14, 12 + n / 8, n % 8));
                }
            }
        }
    }

    None
}

/// Parse generic system register name like `s3_0_c1_c0_1` into encoding bits.
/// Also handles numbered register families like `dbgbcr15_el1`.
pub(crate) fn parse_generic_sysreg(name: &str) -> Result<u32, String> {
    // Try numbered register families first
    if let Some(enc) = parse_numbered_sysreg(name) {
        return Ok(enc);
    }

    // Format: s<op0>_<op1>_c<CRn>_c<CRm>_<op2>
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() == 5 && parts[0].starts_with('s') && parts[2].starts_with('c') && parts[3].starts_with('c') {
        let op0: u32 = parts[0][1..].parse().map_err(|_| format!("unsupported system register: {}", name))?;
        let op1: u32 = parts[1].parse().map_err(|_| format!("unsupported system register: {}", name))?;
        let crn: u32 = parts[2][1..].parse().map_err(|_| format!("unsupported system register: {}", name))?;
        let crm: u32 = parts[3][1..].parse().map_err(|_| format!("unsupported system register: {}", name))?;
        let op2: u32 = parts[4].parse().map_err(|_| format!("unsupported system register: {}", name))?;
        // Each field must fit its architectural width; sysreg_encoding masks, so an
        // out-of-range value would otherwise alias a different register.
        if op0 > 3 || op1 > 7 || crn > 15 || crm > 15 || op2 > 7 {
            return Err(format!("unsupported system register: {}", name));
        }
        let enc = sysreg_encoding(op0, op1, crn, crm, op2);
        Ok(enc)
    } else {
        Err(format!("unsupported system register: {}", name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_sysreg_in_range_fields_are_accepted() {
        assert_eq!(parse_generic_sysreg("s0_0_c0_c0_0"), Ok(0));
        assert_eq!(parse_generic_sysreg("s3_7_c15_c15_7"), Ok(0xffff));
    }

    #[test]
    fn generic_sysreg_out_of_range_fields_are_rejected() {
        // Each of these used to be masked to a valid register instead of being rejected.
        for name in [
            "s4_0_c0_c0_0",
            "s0_8_c0_c0_0",
            "s0_0_c16_c0_0",
            "s0_0_c0_c16_0",
            "s0_0_c0_c0_8",
        ] {
            assert!(parse_generic_sysreg(name).is_err(), "{} should be rejected", name);
        }
    }
}

pub(crate) fn encode_msr(operands: &[Operand]) -> Result<EncodeResult, String> {
    let sysreg = match operands.first() {
        Some(Operand::Symbol(s)) => s.to_lowercase(),
        _ => return Err("msr needs system register name".to_string()),
    };

    // MSR (immediate): msr <pstatefield>, #imm
    // Encoding: 1101_0101_0000_0 op1[18:16] 0100 CRm[11:8] op2[7:5] 11111[4:0]
    // daifset: op1=3, op2=6; daifclr: op1=3, op2=7; spsel: op1=0, op2=5
    match sysreg.as_str() {
        "daifset" => {
            let imm = get_imm(operands, 1)? as u32 & 0xF;
            let word = 0xd5034000 | (imm << 8) | (0b110 << 5) | 0x1F;
            return Ok(EncodeResult::Word(word));
        }
        "daifclr" => {
            let imm = get_imm(operands, 1)? as u32 & 0xF;
            let word = 0xd5034000 | (imm << 8) | (0b111 << 5) | 0x1F;
            return Ok(EncodeResult::Word(word));
        }
        "spsel" => {
            // SPSel: op1=0, op2=5 (MSR immediate form)
            // If the second operand is a register, fall through to MSR register form
            if let Ok(imm) = get_imm(operands, 1) {
                let imm = imm as u32 & 0xF;
                let word = 0xd5004000 | (imm << 8) | (0b101 << 5) | 0x1F;
                return Ok(EncodeResult::Word(word));
            }
        }
        _ => {}
    }

    // MSR (register): msr sysreg, Xt
    let (rt, _) = get_reg(operands, 1)?;

    let encoding = match sysreg.as_str() {
        "sp_el0" => 0xc208u32,
        "tpidr_el0" => 0xde82,
        "tpidr_el1" => 0xc684,
        "tpidr_el2" => 0xe682,
        "tpidrro_el0" => 0xde83,
        "tcr_el1" => 0xc102,
        "ttbr0_el1" => 0xc100,
        "sctlr_el1" => 0xc080,
        "mdscr_el1" => 0x8012,
        "cpacr_el1" => 0xc082,
        "par_el1" => 0xc3a0,
        "osdlr_el1" => 0x809c,
        "oslar_el1" => 0x8084,
        "oslsr_el1" => 0x808c,
        "elr_el1" => 0xc201,
        "spsr_el1" => 0xc200,
        "esr_el1" => 0xc290,
        "far_el1" => 0xc300,
        "vbar_el1" => 0xc600,
        "contextidr_el1" => 0xc681,
        "mair_el1" => 0xc510,
        "amair_el1" => 0xc518,
        "hcr_el2" => 0xe088,
        "cptr_el2" => 0xe08a,
        "hstr_el2" => 0xe08b,
        "elr_el2" => 0xe201,
        "esr_el2" => 0xe290,
        "far_el2" => 0xe300,
        "spsr_el2" => 0xe200,
        "sctlr_el2" => 0xe080,
        "mdcr_el2" => 0xe089,
        "tcr_el2" => 0xe102,
        "ttbr0_el2" => 0xe100,
        "vttbr_el2" => 0xe108,
        "vtcr_el2" => 0xe10a,
        "vbar_el2" => 0xe600,
        "mair_el2" => 0xe510,
        "sp_el1" => 0xe208,
        "csselr_el1" => 0xd000,
        "actlr_el1" => 0xc081,
        "cnthctl_el2" => 0xe708,
        "cntvoff_el2" => 0xe703,
        "sp_el2" => 0xf208,
        "vpidr_el2" => 0xe000,
        "vmpidr_el2" => 0xe005,
        "hacr_el2" => 0xe08f,
        "actlr_el2" => 0xe081,
        "afsr0_el2" => 0xe288,
        "afsr1_el2" => 0xe289,
        "amair_el2" => 0xe518,
        "hpfar_el2" => 0xe304,
        "pmintenset_el1" => 0xc4f1,
        "pmintenclr_el1" => 0xc4f2,
        "pmcr_el0" => 0xdce0,
        "pmcntenset_el0" => 0xdce1,
        "pmcntenclr_el0" => 0xdce2,
        "pmovsclr_el0" => 0xdce3,
        "pmselr_el0" => 0xdce5,
        "pmccntr_el0" => 0xdce8,
        "pmxevtyper_el0" => 0xdce9,
        "pmxevcntr_el0" => 0xdcea,
        "pmuserenr_el0" => 0xdcf0,
        "pmccfiltr_el0" => 0xdf7f,
        "cntv_ctl_el0" => 0xdf19,
        "cntp_ctl_el0" => 0xdf11,
        "cntp_cval_el0" => 0xdf12,
        "cntv_cval_el0" => 0xdf1c,
        "ttbr1_el1" => 0xc101,
        "cntkctl_el1" => 0xc708,
        "daif" => 0xda11,
        "fpcr" => 0xda20,
        "fpsr" => 0xda21,
        "nzcv" => 0xda10,
        "spsel" => 0xc210,
        "mdccint_el1" => 0x8010,
        "fpexc32_el2" => 0xe298,
        "spsr_abt" => 0xe219,
        "spsr_und" => 0xe21a,
        "spsr_irq" => 0xe218,
        "spsr_fiq" => 0xe21b,
        "ifsr32_el2" => 0xe281,
        "dacr32_el2" => 0xe180,
        _ => parse_generic_sysreg(&sysreg)?,
    };

    // MSR encoding: 0xd500_0000 has L=0 (bit 21) for write.
    // Bits [20:19] = op0, supplied entirely by the sysreg encoding field.
    let word = 0xd5000000 | (encoding << 5) | rt;
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_svc(operands: &[Operand]) -> Result<EncodeResult, String> {
    let imm = get_imm(operands, 0)?;
    let word = 0xd4000001 | ((imm as u32 & 0xFFFF) << 5);
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_hvc(operands: &[Operand]) -> Result<EncodeResult, String> {
    let imm = get_imm(operands, 0)?;
    let word = 0xd4000002 | ((imm as u32 & 0xFFFF) << 5);
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_ic(raw_operands: &str) -> Result<EncodeResult, String> {
    let parts: Vec<&str> = raw_operands.splitn(2, ',').collect();
    let op_name = parts[0].trim().to_lowercase();
    let rt = if parts.len() > 1 {
        let reg_str = parts[1].trim();
        parse_reg_num(reg_str).ok_or_else(|| format!("ic: invalid register '{}'", reg_str))?
    } else {
        31 // xzr
    };
    let base = match op_name.as_str() {
        "ialluis" => 0xd508711fu32,
        "iallu"   => 0xd508751f,
        "ivau"    => 0xd50b7520,
        _ => return Err(format!("unsupported ic operation: {}", op_name)),
    };
    let word = (base & !0x1F) | rt;
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_smc(operands: &[Operand]) -> Result<EncodeResult, String> {
    let imm = get_imm(operands, 0)?;
    let word = 0xd4000003 | ((imm as u32 & 0xFFFF) << 5);
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_at(_operands: &[Operand], raw_operands: &str) -> Result<EncodeResult, String> {
    let parts: Vec<&str> = raw_operands.splitn(2, ',').collect();
    let op_name = parts[0].trim().to_lowercase();
    let rt = if parts.len() > 1 {
        let reg_str = parts[1].trim();
        parse_reg_num(reg_str).ok_or_else(|| format!("at: invalid register '{}'", reg_str))?
    } else {
        31
    };
    // AT encoding: SYS instruction. Base words from GCC:
    let base = match op_name.as_str() {
        "s1e1r" => 0xd5087800u32,
        "s1e1w" => 0xd5087820,
        "s1e0r" => 0xd5087840,
        "s1e0w" => 0xd5087860,
        _ => return Err(format!("unsupported at operation: {}", op_name)),
    };
    let word = (base & !0x1F) | rt;
    Ok(EncodeResult::Word(word))
}

/// Encode `sys #op1, Cn, Cm, #op2, Xt` instruction.
pub(crate) fn encode_sys(raw_operands: &str) -> Result<EncodeResult, String> {
    let parts: Vec<&str> = raw_operands.split(',').map(|s| s.trim()).collect();
    if parts.len() < 4 {
        return Err(format!("sys needs at least 4 operands, got: {}", raw_operands));
    }
    let op1: u32 = parts[0].trim_start_matches('#').trim().parse()
        .map_err(|_| format!("sys: invalid op1: {}", parts[0]))?;
    let crn: u32 = parts[1].trim().to_lowercase().trim_start_matches('c').parse()
        .map_err(|_| format!("sys: invalid CRn: {}", parts[1]))?;
    let crm: u32 = parts[2].trim().to_lowercase().trim_start_matches('c').parse()
        .map_err(|_| format!("sys: invalid CRm: {}", parts[2]))?;
    let op2: u32 = parts[3].trim_start_matches('#').trim().parse()
        .map_err(|_| format!("sys: invalid op2: {}", parts[3]))?;
    let rt = if parts.len() >= 5 {
        let reg = parts[4].trim().to_lowercase();
        parse_reg_num(&reg).ok_or_else(|| format!("sys: invalid register: {}", parts[4]))?
    } else {
        31 // xzr if no register specified
    };
    let word = 0xd5080000 | ((op1 & 7) << 16) | ((crn & 0xF) << 12) | ((crm & 0xF) << 8) | ((op2 & 7) << 5) | rt;
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_brk(operands: &[Operand]) -> Result<EncodeResult, String> {
    let imm = get_imm(operands, 0)?;
    let word = 0xd4200000 | ((imm as u32 & 0xFFFF) << 5);
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_tlbi(_operands: &[Operand], raw_operands: &str) -> Result<EncodeResult, String> {
    let parts: Vec<&str> = raw_operands.splitn(2, ',').collect();
    let op_name = parts[0].trim().to_lowercase();
    let rt = if parts.len() > 1 {
        let reg_str = parts[1].trim();
        parse_reg_num(reg_str).ok_or_else(|| format!("tlbi: invalid register '{}'", reg_str))?
    } else {
        31 // xzr
    };
    // TLBI encoding: SYS instruction with fixed fields
    // Full word from GCC objdump for known ops (with Rt=x0):
    let base = match op_name.as_str() {
        // Standard ARMv8.0 TLBI operations
        "vmalle1is" => 0xd508831fu32,
        "vmalle1"   => 0xd508871f,
        "alle1is"   => 0xd50c839f,
        "alle1"     => 0xd50c879f,
        "alle2is"   => 0xd50c831f,
        "vale1is"   => 0xd50883a0,
        "vale1"     => 0xd50887a0,
        "vale2is"   => 0xd50c83a0,
        "vale2"     => 0xd50c87a0,
        "vaae1is"   => 0xd5088360,
        "vaae1"     => 0xd5088760,
        "vaale1is"  => 0xd50883e0,
        "vaale1"    => 0xd50887e0,
        "vae1is"    => 0xd5088320,
        "vae1"      => 0xd5088720,
        "vae2is"    => 0xd50c8320,
        "vae2"      => 0xd50c8720,
        "aside1is"  => 0xd5088340,
        "aside1"    => 0xd5088740,
        "vmalls12e1is" => 0xd50c83df,
        "vmalls12e1"   => 0xd50c87df,
        "ipas2e1is"    => 0xd50c8020,
        "ipas2e1"      => 0xd50c8420,
        "ipas2le1is"   => 0xd50c80a0,
        "ipas2le1"     => 0xd50c84a0,
        // FEAT_TLBIRANGE: range TLBI operations (ARMv8.4-A)
        "rvae1is"      => 0xd5088220,
        "rvale1is"     => 0xd50882a0,
        "rvaae1is"     => 0xd5088260,
        "rvaale1is"    => 0xd50882e0,
        "rvae1"        => 0xd5088620,
        "rvale1"       => 0xd50886a0,
        "rvaae1"       => 0xd5088660,
        "rvaale1"      => 0xd50886e0,
        "rvae1os"      => 0xd5088520,
        "rvale1os"     => 0xd50885a0,
        "rvaae1os"     => 0xd5088560,
        "rvaale1os"    => 0xd50885e0,
        "ripas2e1is"   => 0xd50c8040,
        "ripas2e1"     => 0xd50c8440,
        "ripas2e1os"   => 0xd50c8460,
        "ripas2le1is"  => 0xd50c80c0,
        "ripas2le1"    => 0xd50c84c0,
        "ripas2le1os"  => 0xd50c84e0,
        _ => return Err(format!("unsupported tlbi operation: {}", op_name)),
    };
    // Replace Rt field (bits 4:0)
    let word = (base & !0x1F) | rt;
    Ok(EncodeResult::Word(word))
}

/// Encode HINT #imm (system hint instruction)
pub(crate) fn encode_bti(raw_operands: &str) -> Result<EncodeResult, String> {
    let target = raw_operands.trim().to_lowercase();
    let word = match target.as_str() {
        "" => 0xd503241f,    // bti (no target)
        "c" => 0xd503245f,   // bti c
        "j" => 0xd503249f,   // bti j
        "jc" => 0xd50324df,  // bti jc
        _ => return Err(format!("unsupported bti target: {}", target)),
    };
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_hint(operands: &[Operand]) -> Result<EncodeResult, String> {
    let imm = get_imm(operands, 0)?;
    // HINT: 11010101 00000011 0010 CRm op2 11111
    // CRm = imm >> 3, op2 = imm & 7
    let crm = ((imm as u32) >> 3) & 0xF;
    let op2 = (imm as u32) & 0x7;
    let word = 0xd503201f | (crm << 8) | (op2 << 5);
    Ok(EncodeResult::Word(word))
}

pub(crate) fn encode_dc(operands: &[Operand], raw_operands: &str) -> Result<EncodeResult, String> {
    // Check for the operation type in the operands or raw string
    let op = match operands.first() {
        Some(Operand::Symbol(s)) => s.to_lowercase(),
        _ => raw_operands.to_lowercase(),
    };

    // Find the register operand (second operand or last operand)
    let rt = match operands.get(1) {
        Some(Operand::Reg(name)) => parse_reg_num(name).ok_or("invalid register for dc")?,
        _ => {
            if let Some(Operand::Reg(name)) = operands.last() {
                parse_reg_num(name).ok_or("invalid register for dc")?
            } else {
                0
            }
        }
    };

    if op.contains("civac") {
        // DC CIVAC: sys #3, c7, c14, #1, Xt
        let word = 0xd50b7e20 | rt;
        return Ok(EncodeResult::Word(word));
    }
    if op.contains("cvac") {
        // DC CVAC: sys #3, c7, c10, #1, Xt
        let word = 0xd50b7a20 | rt;
        return Ok(EncodeResult::Word(word));
    }
    if op.contains("cvap") {
        // DC CVAP: sys #3, c7, c12, #1, Xt
        let word = 0xd50b7c20 | rt;
        return Ok(EncodeResult::Word(word));
    }
    if op.contains("cvau") {
        let word = 0xd50b7b20 | rt;
        return Ok(EncodeResult::Word(word));
    }
    if op.contains("ivac") {
        let word = 0xd5087620 | rt;
        return Ok(EncodeResult::Word(word));
    }
    if op.contains("zva") {
        // DC ZVA: sys #3, c7, c4, #1, Xt
        let word = 0xd50b7420 | rt;
        return Ok(EncodeResult::Word(word));
    }

    Err(format!("unsupported dc variant: {}", raw_operands))
}
