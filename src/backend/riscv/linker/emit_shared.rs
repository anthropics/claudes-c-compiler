//! RISC-V shared library emission.
//!
//! Produces an ET_DYN ELF with base address 0 (position-independent).
//! All defined global symbols are exported to .dynsym.
//! R_RISCV_RELATIVE dynamic relocations are emitted for internal absolute addresses.
//!
//! Called from `link.rs::link_shared` after input loading, section merging,
//! and symbol resolution.

use std::collections::{HashMap, HashSet};
use super::elf_read::*;
use super::relocations::{
    GlobalSym, MergedSection, R_RISCV_64, R_RISCV_CALL_PLT,
    write_phdr_at, align_up, section_order,
};
use super::{reloc, symbols};
use crate::backend::linker_common::{self, DynStrTab};

use crate::backend::elf::{
    ET_DYN, PT_LOAD, PT_DYNAMIC, PT_TLS, PT_GNU_STACK, PT_GNU_RELRO,
    PF_X, PF_W, PF_R,
    DT_NULL, DT_NEEDED, DT_STRTAB, DT_SYMTAB, DT_RELA, DT_RELASZ,
    DT_RELAENT, DT_STRSZ, DT_SYMENT, DT_GNU_HASH,
    DT_INIT_ARRAY, DT_INIT_ARRAYSZ, DT_FINI_ARRAY, DT_FINI_ARRAYSZ,
    EM_RISCV as EM_RISCV_ELF,
};

const R_RISCV_RELATIVE: u64 = 3;
const R_RISCV_JUMP_SLOT: u64 = 5;
const R_RISCV_GLOB_DAT: u64 = 6;
const DT_SONAME: u64 = 14;
const DT_RELACOUNT: u64 = 0x6ffffff9;
const SHT_DYNAMIC: u32 = 6;
const SHT_DYNSYM: u32 = 11;
const SHT_RELA: u32 = 4;

const PAGE_SIZE: u64 = 0x1000;
const PT_RISCV_ATTRIBUTES: u32 = 0x70000003;

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;

/// Emit a RISC-V shared library (.so) from pre-resolved linking state.
///
/// `input_objs`, `merged_sections`, `sec_mapping`, `global_syms` etc.
/// have been prepared by the orchestration in `link.rs::link_shared`.
pub fn emit_shared_library(
    input_objs: &[(String, ElfObject)],
    merged_sections: &mut [MergedSection],
    _merged_map: &mut HashMap<String, usize>,
    sec_mapping: &HashMap<(usize, usize), (usize, u64)>,
    global_syms: &mut HashMap<String, GlobalSym>,
    got_symbols: &[String],
    _tls_got_symbols: &HashSet<String>,
    _local_got_sym_info: &HashMap<String, (usize, usize, i64)>,
    needed_sonames: &[String],
    soname: Option<String>,
    output_path: &str,
) -> Result<(), String> {
    // ── Phase 3c: Identify PLT entries needed for external function calls ──
    let mut plt_symbols: Vec<String> = Vec::new();
    {
        let mut plt_set: HashSet<String> = HashSet::new();
        for (_, obj) in input_objs.iter() {
            for relocs in &obj.relocations {
                for reloc in relocs {
                    if reloc.rela_type == R_RISCV_CALL_PLT {
                        let sym = &obj.symbols[reloc.sym_idx as usize];
                        if !sym.name.is_empty() && sym.binding() != STB_LOCAL {
                            if let Some(gsym) = global_syms.get(&sym.name) {
                                if !gsym.defined && !plt_set.contains(&sym.name) {
                                    plt_set.insert(sym.name.clone());
                                    plt_symbols.push(sym.name.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let plt_header_size: u64 = if plt_symbols.is_empty() { 0 } else { 32 };
    let plt_entry_size: u64 = 16;
    let plt_size: u64 = if plt_symbols.is_empty() { 0 } else {
        plt_header_size + plt_symbols.len() as u64 * plt_entry_size
    };
    let got_plt_entries: u64 = if plt_symbols.is_empty() { 0 } else { 2 + plt_symbols.len() as u64 };
    let got_plt_size: u64 = got_plt_entries * 8;
    let rela_plt_size: u64 = plt_symbols.len() as u64 * 24;

    // ── Phase 4: Layout sections ────────────────────────────────────
    let mut sec_indices: Vec<usize> = (0..merged_sections.len()).collect();
    sec_indices.sort_by_key(|&i| {
        let ms = &merged_sections[i];
        section_order(&ms.name, ms.sh_flags)
    });

    let got_size = got_symbols.len() as u64 * 8;

    // Estimate max R_RISCV_RELATIVE entries
    let mut max_rela_count: usize = 0;
    for (_, obj) in input_objs.iter() {
        for sec_relas in &obj.relocations {
            for rela in sec_relas {
                if rela.rela_type == R_RISCV_64 {
                    max_rela_count += 1;
                }
            }
        }
    }
    for ms in merged_sections.iter() {
        if ms.name == ".init_array" || ms.name == ".fini_array" {
            max_rela_count += ms.data.len() / 8;
        }
    }
    max_rela_count += got_symbols.len();

    let has_init_array = merged_sections.iter().any(|ms| ms.name == ".init_array" && !ms.data.is_empty());
    let has_fini_array = merged_sections.iter().any(|ms| ms.name == ".fini_array" && !ms.data.is_empty());
    let has_tls = merged_sections.iter().any(|ms| ms.sh_flags & SHF_TLS != 0);

    let mut dyn_count: u64 = needed_sonames.len() as u64 + 10;
    if soname.is_some() { dyn_count += 1; }
    if has_init_array { dyn_count += 2; }
    if has_fini_array { dyn_count += 2; }
    if !plt_symbols.is_empty() { dyn_count += 4; }
    let dynamic_size = dyn_count * 16;

    let has_riscv_attrs = merged_sections.iter().any(|ms| ms.name == ".riscv.attributes");
    let mut phdr_count: u64 = 8;
    if has_tls { phdr_count += 1; }
    if has_riscv_attrs { phdr_count += 1; }
    let phdr_total_size = phdr_count * 56;

    // ── Layout ──
    let base_addr: u64 = 0;
    let mut offset = 64 + phdr_total_size;

    // Collect exported symbol names for .dynsym
    let mut dyn_sym_names: Vec<String> = Vec::new();
    let mut exported: Vec<String> = global_syms.iter()
        .filter(|(_, g)| g.defined && g.binding != STB_LOCAL)
        .map(|(n, _)| n.clone())
        .collect();
    exported.sort();
    for name in exported { dyn_sym_names.push(name); }
    for (name, gsym) in global_syms.iter() {
        if !gsym.defined && !dyn_sym_names.contains(name) {
            dyn_sym_names.push(name.clone());
        }
    }

    // Build dynamic string table
    let mut dynstr = DynStrTab::new();
    for lib in needed_sonames { dynstr.add(lib); }
    if let Some(ref sn) = soname { dynstr.add(sn); }
    for name in &dyn_sym_names { dynstr.add(name); }

    let dynsym_count = 1 + dyn_sym_names.len();
    let dynsym_size = dynsym_count as u64 * 24;
    let dynstr_bytes = dynstr.as_bytes();
    let dynstr_size = dynstr_bytes.len() as u64;

    // Build .gnu.hash
    let num_hashed = dyn_sym_names.len();
    let gnu_hash_nbuckets = if num_hashed == 0 { 1 } else { num_hashed.next_power_of_two().max(1) } as u32;
    let gnu_hash_bloom_size: u32 = 1;
    let gnu_hash_bloom_shift: u32 = 6;
    let gnu_hash_symoffset: usize = 1;

    let hashed_sym_hashes: Vec<u32> = dyn_sym_names.iter()
        .map(|name| linker_common::gnu_hash(name.as_bytes()))
        .collect();

    let mut bloom_word: u64 = 0;
    for &h in &hashed_sym_hashes {
        bloom_word |= 1u64 << (h as u64 % 64);
        bloom_word |= 1u64 << ((h >> gnu_hash_bloom_shift) as u64 % 64);
    }

    // Sort hashed symbols by bucket
    if num_hashed > 0 {
        let mut hashed_with_hash: Vec<(String, u32)> = dyn_sym_names.iter()
            .zip(hashed_sym_hashes.iter())
            .map(|(n, &h)| (n.clone(), h))
            .collect();
        hashed_with_hash.sort_by_key(|(_, h)| h % gnu_hash_nbuckets);
        for (i_idx, (name, _)) in hashed_with_hash.iter().enumerate() {
            dyn_sym_names[i_idx] = name.clone();
        }
    }

    let hashed_sym_hashes: Vec<u32> = dyn_sym_names.iter()
        .map(|name| linker_common::gnu_hash(name.as_bytes()))
        .collect();

    let mut gnu_hash_buckets = vec![0u32; gnu_hash_nbuckets as usize];
    let mut gnu_hash_chains = vec![0u32; num_hashed];
    for (i_idx, &h) in hashed_sym_hashes.iter().enumerate() {
        let bucket = (h % gnu_hash_nbuckets) as usize;
        if gnu_hash_buckets[bucket] == 0 {
            gnu_hash_buckets[bucket] = (gnu_hash_symoffset + i_idx) as u32;
        }
        gnu_hash_chains[i_idx] = h & !1;
    }
    for bucket_idx in 0..gnu_hash_nbuckets as usize {
        if gnu_hash_buckets[bucket_idx] == 0 { continue; }
        let mut last_in_bucket = 0;
        for (i_idx, &h) in hashed_sym_hashes.iter().enumerate() {
            if (h % gnu_hash_nbuckets) as usize == bucket_idx {
                last_in_bucket = i_idx;
            }
        }
        gnu_hash_chains[last_in_bucket] |= 1;
    }

    let gnu_hash_size: u64 = 16 + (gnu_hash_bloom_size as u64 * 8)
        + (gnu_hash_nbuckets as u64 * 4) + (num_hashed as u64 * 4);

    // RO segment: ELF header + phdrs + .gnu.hash + .dynsym + .dynstr
    offset = align_up(offset, 8);
    let gnu_hash_offset = offset;
    let gnu_hash_addr = base_addr + offset;
    offset += gnu_hash_size;
    offset = align_up(offset, 8);
    let dynsym_offset = offset;
    let dynsym_addr = base_addr + offset;
    offset += dynsym_size;
    let dynstr_offset = offset;
    let dynstr_addr = base_addr + offset;
    offset += dynstr_size;
    let rela_plt_offset = if rela_plt_size > 0 {
        offset = align_up(offset, 8);
        let o = offset;
        offset += rela_plt_size;
        o
    } else { 0 };
    let rela_plt_addr = base_addr + rela_plt_offset;
    let ro_seg_end = offset;

    // Text segment (RX)
    offset = align_up(offset, PAGE_SIZE);
    let text_page_offset = offset;
    let text_page_addr = base_addr + offset;
    for &si in &sec_indices {
        let flags = merged_sections[si].sh_flags;
        let a = merged_sections[si].align.max(1);
        let dlen = merged_sections[si].data.len() as u64;
        if flags & SHF_EXECINSTR != 0 && flags & SHF_ALLOC != 0 {
            offset = align_up(offset, a);
            merged_sections[si].vaddr = base_addr + offset;
            offset += dlen;
        }
    }
    let plt_offset = if plt_size > 0 {
        offset = align_up(offset, 16);
        let o = offset;
        offset += plt_size;
        o
    } else { 0 };
    let plt_vaddr = base_addr + plt_offset;
    let text_total_size = offset - text_page_offset;

    // Rodata segment (RO, non-exec)
    offset = align_up(offset, PAGE_SIZE);
    let rodata_page_offset = offset;
    let rodata_page_addr = base_addr + offset;
    for &si in &sec_indices {
        let flags = merged_sections[si].sh_flags;
        let sh_type = merged_sections[si].sh_type;
        let is_riscv_attr = merged_sections[si].name == ".riscv.attributes";
        let a = merged_sections[si].align.max(1);
        let dlen = merged_sections[si].data.len() as u64;
        if flags & SHF_ALLOC != 0 && flags & SHF_EXECINSTR == 0
            && flags & SHF_WRITE == 0 && sh_type != SHT_NOBITS
            && !is_riscv_attr
        {
            offset = align_up(offset, a);
            merged_sections[si].vaddr = base_addr + offset;
            offset += dlen;
        }
    }
    let rodata_total_size = offset - rodata_page_offset;

    // RW segment
    offset = align_up(offset, PAGE_SIZE);
    let rw_page_offset = offset;
    let rw_page_addr = base_addr + offset;

    let mut init_array_addr = 0u64;
    let mut init_array_size = 0u64;
    let mut fini_array_addr = 0u64;
    let mut fini_array_size = 0u64;

    for &si in &sec_indices {
        if merged_sections[si].name == ".init_array" {
            let a = merged_sections[si].align.max(8);
            offset = align_up(offset, a);
            merged_sections[si].vaddr = base_addr + offset;
            init_array_addr = merged_sections[si].vaddr;
            init_array_size = merged_sections[si].data.len() as u64;
            offset += merged_sections[si].data.len() as u64;
        }
    }
    for &si in &sec_indices {
        if merged_sections[si].name == ".fini_array" {
            let a = merged_sections[si].align.max(8);
            offset = align_up(offset, a);
            merged_sections[si].vaddr = base_addr + offset;
            fini_array_addr = merged_sections[si].vaddr;
            fini_array_size = merged_sections[si].data.len() as u64;
            offset += merged_sections[si].data.len() as u64;
        }
    }

    // .rela.dyn
    offset = align_up(offset, 8);
    let rela_dyn_offset = offset;
    let rela_dyn_addr = base_addr + offset;
    let rela_dyn_max_size = max_rela_count as u64 * 24;
    offset += rela_dyn_max_size;

    // .dynamic
    offset = align_up(offset, 8);
    let dynamic_offset = offset;
    let dynamic_addr = base_addr + offset;
    offset += dynamic_size;

    // GOT (covered by RELRO)
    offset = align_up(offset, 8);
    let got_offset = offset;
    let got_vaddr = base_addr + offset;
    offset += got_size;

    // GOT.PLT (NOT covered by RELRO)
    let got_plt_offset = if got_plt_size > 0 {
        offset = align_up(offset, 8);
        let o = offset;
        offset += got_plt_size;
        o
    } else { 0 };
    let got_plt_vaddr = base_addr + got_plt_offset;

    // Writable data sections
    for &si in &sec_indices {
        let flags = merged_sections[si].sh_flags;
        let sh_type = merged_sections[si].sh_type;
        let is_init = merged_sections[si].name == ".init_array";
        let is_fini = merged_sections[si].name == ".fini_array";
        let a = merged_sections[si].align.max(1);
        let dlen = merged_sections[si].data.len() as u64;
        if flags & SHF_ALLOC != 0 && flags & SHF_WRITE != 0
            && sh_type != SHT_NOBITS && !is_init && !is_fini
            && flags & SHF_TLS == 0
        {
            offset = align_up(offset, a);
            merged_sections[si].vaddr = base_addr + offset;
            offset += dlen;
        }
    }

    // TLS sections
    let mut tls_vaddr = 0u64;
    let mut tls_file_offset = 0u64;
    let mut tls_file_size = 0u64;
    let mut tls_mem_size = 0u64;
    let mut tls_align = 1u64;
    for &si in &sec_indices {
        let flags = merged_sections[si].sh_flags;
        let sh_type = merged_sections[si].sh_type;
        let a = merged_sections[si].align.max(1);
        let dlen = merged_sections[si].data.len() as u64;
        if flags & SHF_TLS != 0 && flags & SHF_ALLOC != 0 && sh_type != SHT_NOBITS {
            offset = align_up(offset, a);
            merged_sections[si].vaddr = base_addr + offset;
            if tls_vaddr == 0 { tls_vaddr = merged_sections[si].vaddr; tls_file_offset = offset; tls_align = a; }
            tls_file_size += dlen;
            tls_mem_size += dlen;
            offset += dlen;
        }
    }
    if tls_vaddr == 0 && has_tls {
        tls_vaddr = base_addr + offset;
        tls_file_offset = offset;
    }
    for &si in &sec_indices {
        let flags = merged_sections[si].sh_flags;
        let sh_type = merged_sections[si].sh_type;
        let a = merged_sections[si].align.max(1);
        let dlen = merged_sections[si].data.len() as u64;
        if flags & SHF_TLS != 0 && sh_type == SHT_NOBITS {
            let aligned = align_up(tls_mem_size, a);
            merged_sections[si].vaddr = tls_vaddr + aligned;
            tls_mem_size = aligned + dlen;
            if a > tls_align { tls_align = a; }
        }
    }
    tls_mem_size = align_up(tls_mem_size, tls_align);

    // BSS
    let bss_addr = base_addr + offset;
    let mut bss_size = 0u64;
    for &si in &sec_indices {
        let sh_type = merged_sections[si].sh_type;
        let flags = merged_sections[si].sh_flags;
        let a = merged_sections[si].align.max(1);
        let dlen = merged_sections[si].data.len() as u64;
        if sh_type == SHT_NOBITS && flags & SHF_ALLOC != 0 && flags & SHF_TLS == 0 {
            let aligned = align_up(bss_addr + bss_size, a);
            bss_size = aligned - bss_addr + dlen;
            merged_sections[si].vaddr = aligned;
        }
    }

    // ── Update global symbol addresses ──────────────────────────────
    let section_vaddrs: Vec<u64> = merged_sections.iter().map(|ms| ms.vaddr).collect();
    for (_, gsym) in global_syms.iter_mut() {
        if gsym.defined {
            if let Some(si) = gsym.section_idx {
                gsym.value += section_vaddrs[si];
            }
        }
    }

    let local_sym_vaddrs = symbols::build_local_sym_vaddrs(
        input_objs, sec_mapping, &section_vaddrs, global_syms,
    );

    // Assign GOT offsets to symbols
    let mut got_sym_offsets: HashMap<String, u64> = HashMap::new();
    for (gi, name) in got_symbols.iter().enumerate() {
        let got_off = gi as u64 * 8;
        got_sym_offsets.insert(name.clone(), got_off);
        if let Some(gsym) = global_syms.get_mut(name) {
            gsym.got_offset = Some(got_off);
        }
    }

    // ── Phase 5: Emit ELF ───────────────────────────────────────────
    let file_size = offset as usize;
    let mut elf = vec![0u8; file_size];

    // ELF header
    elf[0..4].copy_from_slice(&ELF_MAGIC);
    elf[4] = ELFCLASS64; elf[5] = ELFDATA2LSB; elf[6] = 1;
    elf[7] = 0;
    elf[16..18].copy_from_slice(&ET_DYN.to_le_bytes());
    elf[18..20].copy_from_slice(&EM_RISCV_ELF.to_le_bytes());
    elf[20..24].copy_from_slice(&1u32.to_le_bytes());
    elf[24..32].copy_from_slice(&0u64.to_le_bytes()); // e_entry = 0
    elf[32..40].copy_from_slice(&64u64.to_le_bytes()); // e_phoff
    elf[40..48].copy_from_slice(&0u64.to_le_bytes()); // e_shoff placeholder
    elf[48..52].copy_from_slice(&0x5u32.to_le_bytes()); // e_flags
    elf[52..54].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
    elf[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize
    elf[56..58].copy_from_slice(&(phdr_count as u16).to_le_bytes());
    elf[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
    elf[60..62].copy_from_slice(&0u16.to_le_bytes()); // e_shnum placeholder
    elf[62..64].copy_from_slice(&0u16.to_le_bytes()); // e_shstrndx placeholder

    // Program headers
    let mut ph = 64usize;
    // PT_PHDR
    write_phdr_at(&mut elf, ph, 6, PF_R, 64, base_addr + 64, base_addr + 64, phdr_total_size, phdr_total_size, 8);
    ph += 56;
    // PT_LOAD (RO)
    write_phdr_at(&mut elf, ph, PT_LOAD, PF_R, 0, base_addr, base_addr, ro_seg_end, ro_seg_end, PAGE_SIZE);
    ph += 56;
    // PT_LOAD (text)
    write_phdr_at(&mut elf, ph, PT_LOAD, PF_R | PF_X, text_page_offset, text_page_addr, text_page_addr,
                  text_total_size, text_total_size, PAGE_SIZE);
    ph += 56;
    // PT_LOAD (rodata)
    write_phdr_at(&mut elf, ph, PT_LOAD, PF_R, rodata_page_offset, rodata_page_addr, rodata_page_addr,
                  rodata_total_size, rodata_total_size, PAGE_SIZE);
    ph += 56;
    // PT_LOAD (RW)
    let rw_filesz = offset - rw_page_offset;
    let rw_memsz = if bss_size > 0 { (bss_addr + bss_size) - rw_page_addr } else { rw_filesz };
    write_phdr_at(&mut elf, ph, PT_LOAD, PF_R | PF_W, rw_page_offset, rw_page_addr, rw_page_addr,
                  rw_filesz, rw_memsz, PAGE_SIZE);
    ph += 56;
    // PT_DYNAMIC
    write_phdr_at(&mut elf, ph, PT_DYNAMIC, PF_R | PF_W, dynamic_offset, dynamic_addr, dynamic_addr,
                  dynamic_size, dynamic_size, 8);
    ph += 56;
    // PT_GNU_RELRO
    {
        let relro_start_offset = rw_page_offset;
        let relro_start_addr = rw_page_addr;
        let relro_end = if got_size > 0 {
            align_up(got_vaddr + got_size, PAGE_SIZE)
        } else {
            align_up(dynamic_addr + dynamic_size, PAGE_SIZE)
        };
        let relro_end = if got_plt_size > 0 && relro_end > got_plt_vaddr {
            got_vaddr + got_size
        } else {
            relro_end
        };
        let relro_filesz = relro_end - relro_start_addr;
        let relro_memsz = relro_filesz;
        write_phdr_at(&mut elf, ph, PT_GNU_RELRO, PF_R, relro_start_offset, relro_start_addr,
                      relro_start_addr, relro_filesz, relro_memsz, 1);
    }
    ph += 56;
    // PT_GNU_STACK
    write_phdr_at(&mut elf, ph, PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 0, 0x10);
    ph += 56;
    // PT_TLS (optional)
    if has_tls {
        write_phdr_at(&mut elf, ph, PT_TLS, PF_R, tls_file_offset, tls_vaddr, tls_vaddr,
                      tls_file_size, tls_mem_size, tls_align);
        ph += 56;
    }
    // PT_RISCV_ATTRIBUTES (optional)
    if has_riscv_attrs {
        if let Some(_ms) = merged_sections.iter().find(|ms| ms.name == ".riscv.attributes") {
            write_phdr_at(&mut elf, ph, PT_RISCV_ATTRIBUTES, PF_R, 0, 0, 0, 0, 0, 1);
        }
    }

    // Write .gnu.hash
    {
        let gh = gnu_hash_offset as usize;
        elf[gh..gh+4].copy_from_slice(&gnu_hash_nbuckets.to_le_bytes());
        elf[gh+4..gh+8].copy_from_slice(&(gnu_hash_symoffset as u32).to_le_bytes());
        elf[gh+8..gh+12].copy_from_slice(&gnu_hash_bloom_size.to_le_bytes());
        elf[gh+12..gh+16].copy_from_slice(&gnu_hash_bloom_shift.to_le_bytes());
        let bloom_off = gh + 16;
        elf[bloom_off..bloom_off+8].copy_from_slice(&bloom_word.to_le_bytes());
        let buckets_off = bloom_off + (gnu_hash_bloom_size as usize * 8);
        for (bi, &b) in gnu_hash_buckets.iter().enumerate() {
            let off_b = buckets_off + bi * 4;
            elf[off_b..off_b+4].copy_from_slice(&b.to_le_bytes());
        }
        let chains_off = buckets_off + (gnu_hash_nbuckets as usize * 4);
        for (ci, &c) in gnu_hash_chains.iter().enumerate() {
            let off_c = chains_off + ci * 4;
            elf[off_c..off_c+4].copy_from_slice(&c.to_le_bytes());
        }
    }

    // Write .dynsym
    let mut merged_to_shdr: HashMap<usize, u16> = HashMap::new();
    {
        let mut shdr_idx = 4u16;
        for &si in &sec_indices {
            let ms = &merged_sections[si];
            if ms.name == ".riscv.attributes" { continue; }
            if ms.sh_flags & SHF_ALLOC == 0 && ms.sh_type != SHT_RISCV_ATTRIBUTES { continue; }
            merged_to_shdr.insert(si, shdr_idx);
            shdr_idx += 1;
        }
    }
    {
        let mut ds = dynsym_offset as usize + 24;
        for name in &dyn_sym_names {
            let no = dynstr.get_offset(name) as u32;
            elf[ds..ds+4].copy_from_slice(&no.to_le_bytes());
            if let Some(gsym) = global_syms.get(name) {
                if gsym.defined {
                    let info = (gsym.binding << 4) | gsym.sym_type;
                    elf[ds+4] = info;
                    elf[ds+5] = 0;
                    let shndx = gsym.section_idx
                        .and_then(|si| merged_to_shdr.get(&si).copied())
                        .unwrap_or(1);
                    elf[ds+6..ds+8].copy_from_slice(&shndx.to_le_bytes());
                    elf[ds+8..ds+16].copy_from_slice(&gsym.value.to_le_bytes());
                    elf[ds+16..ds+24].copy_from_slice(&gsym.size.to_le_bytes());
                } else {
                    let bind = gsym.binding;
                    let stype = if gsym.sym_type != 0 { gsym.sym_type } else { STT_FUNC };
                    elf[ds+4] = (bind << 4) | stype;
                    elf[ds+5] = 0;
                    elf[ds+6..ds+8].copy_from_slice(&0u16.to_le_bytes());
                    elf[ds+8..ds+16].copy_from_slice(&0u64.to_le_bytes());
                    elf[ds+16..ds+24].copy_from_slice(&0u64.to_le_bytes());
                }
            } else {
                elf[ds+4] = (STB_GLOBAL << 4) | STT_FUNC;
                elf[ds+5] = 0;
                elf[ds+6..ds+8].copy_from_slice(&0u16.to_le_bytes());
                elf[ds+8..ds+16].copy_from_slice(&0u64.to_le_bytes());
                elf[ds+16..ds+24].copy_from_slice(&0u64.to_le_bytes());
            }
            ds += 24;
        }
    }

    // Write .dynstr
    {
        let db = dynstr.as_bytes();
        let start = dynstr_offset as usize;
        if start + db.len() <= elf.len() {
            elf[start..start + db.len()].copy_from_slice(db);
        }
    }

    // Write section data
    for ms in merged_sections.iter() {
        if ms.sh_type == SHT_NOBITS || ms.data.is_empty() || ms.vaddr == 0 { continue; }
        if ms.name == ".riscv.attributes" { continue; }
        let file_off = (ms.vaddr - base_addr) as usize;
        if file_off + ms.data.len() <= elf.len() {
            elf[file_off..file_off + ms.data.len()].copy_from_slice(&ms.data);
        }
    }

    // Fill GOT entries
    for (gi, name) in got_symbols.iter().enumerate() {
        let entry_off = (got_offset + gi as u64 * 8) as usize;
        if let Some(gsym) = global_syms.get(name) {
            if gsym.defined && entry_off + 8 <= elf.len() {
                elf[entry_off..entry_off+8].copy_from_slice(&gsym.value.to_le_bytes());
            }
        }
    }

    // Build PLT stubs and GOT.PLT
    let mut plt_sym_addrs: HashMap<String, u64> = HashMap::new();
    if !plt_symbols.is_empty() {
        let plt0_addr = plt_vaddr;
        for i in 0..plt_symbols.len() {
            let off = got_plt_offset as usize + (2 + i) * 8;
            if off + 8 <= elf.len() {
                elf[off..off + 8].copy_from_slice(&plt0_addr.to_le_bytes());
            }
        }

        // PLT[0] header
        let got_plt_rel = got_plt_vaddr as i64 - plt0_addr as i64;
        let hi = ((got_plt_rel + 0x800) >> 12) as u32;
        let lo = (got_plt_rel & 0xFFF) as u32;
        let po = plt_offset as usize;

        let insn0 = 0x00000397u32 | (hi << 12);
        elf[po..po+4].copy_from_slice(&insn0.to_le_bytes());
        let insn1 = 0x41c30333u32;
        elf[po+4..po+8].copy_from_slice(&insn1.to_le_bytes());
        let insn2 = 0x0003be03u32 | (lo << 20);
        elf[po+8..po+12].copy_from_slice(&insn2.to_le_bytes());
        let neg_hdr = (-((plt_header_size as i32) + 12)) as u32;
        let insn3 = 0x00030313u32 | ((neg_hdr & 0xFFF) << 20);
        elf[po+12..po+16].copy_from_slice(&insn3.to_le_bytes());
        let insn4 = 0x00038293u32 | (lo << 20);
        elf[po+16..po+20].copy_from_slice(&insn4.to_le_bytes());
        let insn5 = 0x00135313u32;
        elf[po+20..po+24].copy_from_slice(&insn5.to_le_bytes());
        let insn6 = 0x0082b283u32;
        elf[po+24..po+28].copy_from_slice(&insn6.to_le_bytes());
        let insn7 = 0x000e0067u32;
        elf[po+28..po+32].copy_from_slice(&insn7.to_le_bytes());

        // PLT entries
        for (i, name) in plt_symbols.iter().enumerate() {
            let entry_file_off = plt_offset as usize + plt_header_size as usize + i * plt_entry_size as usize;
            let entry_addr = plt_vaddr + plt_header_size + i as u64 * plt_entry_size;
            let got_entry_addr = got_plt_vaddr + (2 + i) as u64 * 8;

            plt_sym_addrs.insert(name.clone(), entry_addr);

            let rel = got_entry_addr as i64 - entry_addr as i64;
            let hi = ((rel + 0x800) >> 12) as u32;
            let lo = (rel & 0xFFF) as u32;

            let insn0 = 0x00000e17u32 | (hi << 12);
            elf[entry_file_off..entry_file_off+4].copy_from_slice(&insn0.to_le_bytes());
            let insn1 = 0x000e3e03u32 | (lo << 20);
            elf[entry_file_off+4..entry_file_off+8].copy_from_slice(&insn1.to_le_bytes());
            let insn2 = 0x000e0367u32;
            elf[entry_file_off+8..entry_file_off+12].copy_from_slice(&insn2.to_le_bytes());
            let insn3 = 0x00000013u32;
            elf[entry_file_off+12..entry_file_off+16].copy_from_slice(&insn3.to_le_bytes());
        }
    }

    // ── Apply relocations and collect R_RISCV_RELATIVE entries ──────
    let mut rela_dyn_entries: Vec<(u64, u64)> = Vec::new();
    let mut glob_dat_entries: Vec<(u64, String)> = Vec::new();

    for (gi, name) in got_symbols.iter().enumerate() {
        let gea = got_vaddr + gi as u64 * 8;
        if let Some(gsym) = global_syms.get(name) {
            if gsym.defined {
                rela_dyn_entries.push((gea, gsym.value));
            } else {
                glob_dat_entries.push((gea, name.clone()));
            }
        } else {
            glob_dat_entries.push((gea, name.clone()));
        }
    }

    // Apply relocations with RELATIVE collection
    let empty_relax = HashMap::new();
    let empty_nop = HashSet::new();
    let ctx = reloc::RelocContext {
        sec_mapping,
        section_vaddrs: &section_vaddrs,
        local_sym_vaddrs: &local_sym_vaddrs,
        global_syms,
        got_vaddr,
        got_symbols,
        got_plt_vaddr: 0,
        tls_vaddr,
        gd_tls_relax_info: &empty_relax,
        gd_tls_call_nop: &empty_nop,
        collect_relatives: true,
        got_sym_offsets: &got_sym_offsets,
        plt_sym_addrs: &plt_sym_addrs,
    };
    let reloc_result = reloc::apply_relocations(input_objs, merged_sections, &ctx)?;
    rela_dyn_entries.extend(reloc_result.relative_entries);

    // Write updated section data back to elf buffer
    for ms in merged_sections.iter() {
        if ms.sh_type == SHT_NOBITS || ms.data.is_empty() || ms.vaddr == 0 { continue; }
        if ms.name == ".riscv.attributes" { continue; }
        let file_off = (ms.vaddr - base_addr) as usize;
        if file_off + ms.data.len() <= elf.len() {
            elf[file_off..file_off + ms.data.len()].copy_from_slice(&ms.data);
        }
    }

    // Write GOT entries (re-fill after relocations)
    for (gi, name) in got_symbols.iter().enumerate() {
        let entry_off = (got_offset + gi as u64 * 8) as usize;
        if let Some(gsym) = global_syms.get(name) {
            if gsym.defined && entry_off + 8 <= elf.len() {
                elf[entry_off..entry_off+8].copy_from_slice(&gsym.value.to_le_bytes());
            }
        }
    }

    // Add RELATIVE entries for init_array/fini_array pointers
    for ms in merged_sections.iter() {
        if ms.name == ".init_array" || ms.name == ".fini_array" {
            let num_entries = ms.data.len() / 8;
            for ei in 0..num_entries {
                let ptr_off = ei * 8;
                if ptr_off + 8 <= ms.data.len() {
                    let val = u64::from_le_bytes(ms.data[ptr_off..ptr_off+8].try_into().unwrap());
                    if val != 0 {
                        let runtime_addr = ms.vaddr + ptr_off as u64;
                        rela_dyn_entries.push((runtime_addr, val));
                    }
                }
            }
        }
    }

    // Write .rela.dyn entries
    let relative_count = rela_dyn_entries.len();
    let actual_rela_count = relative_count + glob_dat_entries.len();
    let rela_dyn_size = actual_rela_count as u64 * 24;
    {
        let mut rd = rela_dyn_offset as usize;
        for &(rel_offset, rel_addend) in &rela_dyn_entries {
            if rd + 24 <= elf.len() {
                elf[rd..rd+8].copy_from_slice(&rel_offset.to_le_bytes());
                elf[rd+8..rd+16].copy_from_slice(&R_RISCV_RELATIVE.to_le_bytes());
                elf[rd+16..rd+24].copy_from_slice(&rel_addend.to_le_bytes());
                rd += 24;
            }
        }
        for (rel_offset, sym_name) in &glob_dat_entries {
            let si = dyn_sym_names.iter().position(|n| n == sym_name).map(|j| j + 1).unwrap_or(0) as u64;
            if rd + 24 <= elf.len() {
                elf[rd..rd+8].copy_from_slice(&rel_offset.to_le_bytes());
                elf[rd+8..rd+16].copy_from_slice(&((si << 32) | R_RISCV_GLOB_DAT).to_le_bytes());
                elf[rd+16..rd+24].copy_from_slice(&0u64.to_le_bytes());
                rd += 24;
            }
        }
    }

    // Write .rela.plt entries
    if !plt_symbols.is_empty() {
        let mut rp = rela_plt_offset as usize;
        for (i, name) in plt_symbols.iter().enumerate() {
            let got_entry_addr = got_plt_vaddr + (2 + i) as u64 * 8;
            let si = dyn_sym_names.iter().position(|n| n == name).map(|j| j + 1).unwrap_or(0) as u64;
            let r_info = (si << 32) | R_RISCV_JUMP_SLOT;
            if rp + 24 <= elf.len() {
                elf[rp..rp+8].copy_from_slice(&got_entry_addr.to_le_bytes());
                elf[rp+8..rp+16].copy_from_slice(&r_info.to_le_bytes());
                elf[rp+16..rp+24].copy_from_slice(&0i64.to_le_bytes());
                rp += 24;
            }
        }
    }

    // Write .dynamic section
    {
        let mut dd = dynamic_offset as usize;
        for lib in needed_sonames {
            let so = dynstr.get_offset(lib) as u64;
            elf[dd..dd+8].copy_from_slice(&(DT_NEEDED as u64).to_le_bytes());
            elf[dd+8..dd+16].copy_from_slice(&so.to_le_bytes());
            dd += 16;
        }
        if let Some(ref sn) = soname {
            let so = dynstr.get_offset(sn) as u64;
            elf[dd..dd+8].copy_from_slice(&DT_SONAME.to_le_bytes());
            elf[dd+8..dd+16].copy_from_slice(&so.to_le_bytes());
            dd += 16;
        }
        let dyn_entries: Vec<(u64, u64)> = vec![
            (DT_STRTAB as u64, dynstr_addr),
            (DT_SYMTAB as u64, dynsym_addr),
            (DT_STRSZ as u64, dynstr_size),
            (DT_SYMENT as u64, 24),
            (DT_RELA as u64, rela_dyn_addr),
            (DT_RELASZ as u64, rela_dyn_size),
            (DT_RELAENT as u64, 24),
            (DT_RELACOUNT, relative_count as u64),
            (DT_GNU_HASH as u64, gnu_hash_addr),
        ];
        for (tag, val) in dyn_entries {
            elf[dd..dd+8].copy_from_slice(&tag.to_le_bytes());
            elf[dd+8..dd+16].copy_from_slice(&val.to_le_bytes());
            dd += 16;
        }
        if has_init_array {
            elf[dd..dd+8].copy_from_slice(&(DT_INIT_ARRAY as u64).to_le_bytes());
            elf[dd+8..dd+16].copy_from_slice(&init_array_addr.to_le_bytes());
            dd += 16;
            elf[dd..dd+8].copy_from_slice(&(DT_INIT_ARRAYSZ as u64).to_le_bytes());
            elf[dd+8..dd+16].copy_from_slice(&init_array_size.to_le_bytes());
            dd += 16;
        }
        if has_fini_array {
            elf[dd..dd+8].copy_from_slice(&(DT_FINI_ARRAY as u64).to_le_bytes());
            elf[dd+8..dd+16].copy_from_slice(&fini_array_addr.to_le_bytes());
            dd += 16;
            elf[dd..dd+8].copy_from_slice(&(DT_FINI_ARRAYSZ as u64).to_le_bytes());
            elf[dd+8..dd+16].copy_from_slice(&fini_array_size.to_le_bytes());
            dd += 16;
        }
        // PLT-related dynamic entries
        if !plt_symbols.is_empty() {
            const DT_PLTGOT_TAG: u64 = 3;
            const DT_PLTRELSZ_TAG: u64 = 2;
            const DT_PLTREL_TAG: u64 = 20;
            const DT_JMPREL_TAG: u64 = 23;
            let plt_dyn_entries: Vec<(u64, u64)> = vec![
                (DT_PLTGOT_TAG, got_plt_vaddr),
                (DT_PLTRELSZ_TAG, rela_plt_size),
                (DT_PLTREL_TAG, 7),
                (DT_JMPREL_TAG, rela_plt_addr),
            ];
            for (tag, val) in plt_dyn_entries {
                elf[dd..dd+8].copy_from_slice(&tag.to_le_bytes());
                elf[dd+8..dd+16].copy_from_slice(&val.to_le_bytes());
                dd += 16;
            }
        }
        // DT_NULL terminator
        elf[dd..dd+8].copy_from_slice(&(DT_NULL as u64).to_le_bytes());
        elf[dd+8..dd+16].copy_from_slice(&0u64.to_le_bytes());
    }

    // ── Append section headers ──────────────────────────────────────
    let mut shstrtab_data: Vec<u8> = vec![0];
    let mut section_headers: Vec<(String, u32, u64, u64, u64, u64, u32, u32, u64, u64)> = Vec::new();

    let add_shstrtab_name = |name: &str, strtab: &mut Vec<u8>| -> u32 {
        let off = strtab.len() as u32;
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
        off
    };

    // Section 0: null
    section_headers.push(("".into(), 0, 0, 0, 0, 0, 0, 0, 0, 0));

    // .gnu.hash
    let _sh_name = add_shstrtab_name(".gnu.hash", &mut shstrtab_data);
    section_headers.push((".gnu.hash".into(), 0x6ffffff6, SHF_ALLOC,
        gnu_hash_addr, gnu_hash_offset, gnu_hash_size, 3, 0, 8, 0));

    // .dynsym
    let _sh_name = add_shstrtab_name(".dynsym", &mut shstrtab_data);
    section_headers.push((".dynsym".into(), SHT_DYNSYM, SHF_ALLOC,
        dynsym_addr, dynsym_offset, dynsym_size, 3, 1, 8, 24));

    // .dynstr
    let _sh_name = add_shstrtab_name(".dynstr", &mut shstrtab_data);
    section_headers.push((".dynstr".into(), SHT_STRTAB, SHF_ALLOC,
        dynstr_addr, dynstr_offset, dynstr_size, 0, 0, 1, 0));

    // Fix .gnu.hash link to point to .dynsym (index 2)
    section_headers[1].6 = 2;

    // Add merged sections as section headers
    for &si in &sec_indices {
        let ms = &merged_sections[si];
        if ms.name == ".riscv.attributes" { continue; }
        if ms.sh_flags & SHF_ALLOC == 0 && ms.sh_type != SHT_RISCV_ATTRIBUTES { continue; }
        let _sh_name = add_shstrtab_name(&ms.name, &mut shstrtab_data);
        let sh_offset = if ms.sh_type == SHT_NOBITS { 0 } else { ms.vaddr - base_addr };
        let sh_size = ms.data.len() as u64;
        section_headers.push((ms.name.clone(), ms.sh_type, ms.sh_flags,
            ms.vaddr, sh_offset, sh_size, 0, 0, ms.align, 0));
    }

    // .rela.dyn
    let _sh_name = add_shstrtab_name(".rela.dyn", &mut shstrtab_data);
    section_headers.push((".rela.dyn".into(), SHT_RELA, SHF_ALLOC,
        rela_dyn_addr, rela_dyn_offset, rela_dyn_size, 2, 0, 8, 24));

    // .dynamic
    let _sh_name = add_shstrtab_name(".dynamic", &mut shstrtab_data);
    section_headers.push((".dynamic".into(), SHT_DYNAMIC, SHF_ALLOC | SHF_WRITE,
        dynamic_addr, dynamic_offset, dynamic_size, 3, 0, 8, 16));

    // .got
    if got_size > 0 {
        let _sh_name = add_shstrtab_name(".got", &mut shstrtab_data);
        section_headers.push((".got".into(), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
            got_vaddr, got_offset, got_size, 0, 0, 8, 8));
    }

    // .got.plt
    if got_plt_size > 0 {
        let _sh_name = add_shstrtab_name(".got.plt", &mut shstrtab_data);
        section_headers.push((".got.plt".into(), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
            got_plt_vaddr, got_plt_offset, got_plt_size, 0, 0, 8, 8));
    }

    // .plt
    if plt_size > 0 {
        let _sh_name = add_shstrtab_name(".plt", &mut shstrtab_data);
        section_headers.push((".plt".into(), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
            plt_vaddr, plt_offset, plt_size, 0, 0, 16, plt_entry_size));
    }

    // .rela.plt
    if rela_plt_size > 0 {
        let _sh_name = add_shstrtab_name(".rela.plt", &mut shstrtab_data);
        section_headers.push((".rela.plt".into(), SHT_RELA, SHF_ALLOC,
            rela_plt_addr, rela_plt_offset, rela_plt_size, 2, 0, 8, 24));
    }

    // .riscv.attributes (non-loadable)
    let mut attr_file_offset = 0u64;
    let mut attr_size = 0u64;
    if let Some(ms) = merged_sections.iter().find(|ms| ms.name == ".riscv.attributes") {
        let _sh_name = add_shstrtab_name(".riscv.attributes", &mut shstrtab_data);
        attr_file_offset = elf.len() as u64;
        attr_size = ms.data.len() as u64;
        elf.extend_from_slice(&ms.data);
        section_headers.push((".riscv.attributes".into(), SHT_RISCV_ATTRIBUTES, 0,
            0, attr_file_offset, attr_size, 0, 0, 1, 0));
    }

    // .shstrtab
    let _shstrtab_name_off = add_shstrtab_name(".shstrtab", &mut shstrtab_data);
    let shstrtab_idx = section_headers.len();
    let shstrtab_file_offset = elf.len() as u64;
    elf.extend_from_slice(&shstrtab_data);
    section_headers.push((".shstrtab".into(), SHT_STRTAB, 0,
        0, shstrtab_file_offset, shstrtab_data.len() as u64, 0, 0, 1, 0));

    // Write section headers
    while elf.len() % 8 != 0 { elf.push(0); }
    let shdr_offset = elf.len() as u64;
    let shdr_count = section_headers.len();

    // Rebuild shstrtab name offsets
    let mut name_offsets: Vec<u32> = Vec::new();
    {
        let mut _strtab_pos = 0u32;
        for (name, ..) in &section_headers {
            if name.is_empty() {
                name_offsets.push(0);
            } else {
                let name_bytes = name.as_bytes();
                let mut found = false;
                for pos in 0..shstrtab_data.len() {
                    if pos + name_bytes.len() < shstrtab_data.len()
                        && &shstrtab_data[pos..pos + name_bytes.len()] == name_bytes
                        && shstrtab_data[pos + name_bytes.len()] == 0
                    {
                        name_offsets.push(pos as u32);
                        found = true;
                        break;
                    }
                }
                if !found { name_offsets.push(_strtab_pos); }
            }
            _strtab_pos += name.len() as u32 + 1;
        }
    }

    for (idx, (_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize)) in section_headers.iter().enumerate() {
        let mut shdr = [0u8; 64];
        let name_off = name_offsets[idx];
        shdr[0..4].copy_from_slice(&name_off.to_le_bytes());
        shdr[4..8].copy_from_slice(&sh_type.to_le_bytes());
        shdr[8..16].copy_from_slice(&sh_flags.to_le_bytes());
        shdr[16..24].copy_from_slice(&sh_addr.to_le_bytes());
        shdr[24..32].copy_from_slice(&sh_offset.to_le_bytes());
        shdr[32..40].copy_from_slice(&sh_size.to_le_bytes());
        shdr[40..44].copy_from_slice(&sh_link.to_le_bytes());
        shdr[44..48].copy_from_slice(&sh_info.to_le_bytes());
        shdr[48..56].copy_from_slice(&sh_addralign.to_le_bytes());
        shdr[56..64].copy_from_slice(&sh_entsize.to_le_bytes());
        elf.extend_from_slice(&shdr);
    }

    // Update ELF header with section header info
    elf[40..48].copy_from_slice(&shdr_offset.to_le_bytes());
    elf[60..62].copy_from_slice(&(shdr_count as u16).to_le_bytes());
    elf[62..64].copy_from_slice(&(shstrtab_idx as u16).to_le_bytes());

    // Update PT_RISCV_ATTRIBUTES phdr if present
    if has_riscv_attrs && attr_size > 0 {
        let mut ph_off = 64;
        for _ in 0..phdr_count {
            let p_type = u32::from_le_bytes(elf[ph_off..ph_off+4].try_into().unwrap());
            if p_type == PT_RISCV_ATTRIBUTES {
                elf[ph_off+8..ph_off+16].copy_from_slice(&attr_file_offset.to_le_bytes());
                elf[ph_off+16..ph_off+24].copy_from_slice(&0u64.to_le_bytes());
                elf[ph_off+24..ph_off+32].copy_from_slice(&0u64.to_le_bytes());
                elf[ph_off+32..ph_off+40].copy_from_slice(&attr_size.to_le_bytes());
                elf[ph_off+40..ph_off+48].copy_from_slice(&0u64.to_le_bytes());
                break;
            }
            ph_off += 56;
        }
    }

    // Write output
    std::fs::write(output_path, &elf)
        .map_err(|e| format!("failed to write '{}': {}", output_path, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(output_path, std::fs::Permissions::from_mode(0o755));
    }

    Ok(())
}
