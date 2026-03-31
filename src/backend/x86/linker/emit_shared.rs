//! Shared library (.so) emission for the x86-64 linker.
//!
//! Emits an ELF64 shared library (ET_DYN) with PIC relocations, PLT stubs,
//! `.dynamic` section, and GNU hash tables.

use std::collections::{HashMap, HashSet, BTreeSet};

use super::elf::*;
use super::types::{GlobalSymbol, PAGE_SIZE};
use super::emit_exec::resolve_sym;
use crate::backend::linker_common::{self, DynStrTab, OutputSection};

pub(super) fn emit_shared_library(
    objects: &[ElfObject], globals: &mut HashMap<String, GlobalSymbol>,
    output_sections: &mut [OutputSection],
    section_map: &HashMap<(usize, usize), (usize, u64)>,
    needed_sonames: &[String], output_path: &str,
    soname: Option<String>, rpath_entries: &[String], use_runpath: bool,
) -> Result<(), String> {
    let base_addr: u64 = 0;

    let mut dynstr = DynStrTab::new();
    for lib in needed_sonames { dynstr.add(lib); }
    if let Some(ref sn) = soname { dynstr.add(sn); }
    let rpath_string = if rpath_entries.is_empty() { None } else {
        let s = rpath_entries.join(":");
        dynstr.add(&s);
        Some(s)
    };

    // Identify symbols that need PLT entries: any symbol referenced via
    // R_X86_64_PLT32 or R_X86_64_PC32 that is not defined locally.
    // In shared libraries, undefined symbols are resolved at runtime by the
    // dynamic linker, so we need PLT entries for all of them.
    let mut plt_names: Vec<String> = Vec::new();
    for obj in objects.iter() {
        for sec_relas in &obj.relocations {
            for rela in sec_relas {
                let si = rela.sym_idx as usize;
                if si >= obj.symbols.len() { continue; }
                let sym = &obj.symbols[si];
                if sym.name.is_empty() { continue; }
                // Skip local symbols - they don't need PLT entries
                if sym.is_local() { continue; }
                match rela.rela_type {
                    R_X86_64_PLT32 | R_X86_64_PC32 => {
                        if let Some(gsym) = globals.get(&sym.name) {
                            // Need PLT for: dynamic symbols from shared libs,
                            // or undefined symbols not defined in any loaded object
                            let needs_plt = gsym.is_dynamic
                                || (gsym.defined_in.is_none() && gsym.section_idx == SHN_UNDEF);
                            if needs_plt && !plt_names.contains(&sym.name) {
                                plt_names.push(sym.name.clone());
                            }
                        }
                        // Don't create PLT for symbols not in globals - they are
                        // local/section symbols resolved directly
                    }
                    _ => {}
                }
            }
        }
    }

    // Ensure PLT symbols that are not yet in globals get entries (e.g. libc symbols
    // when libc is not explicitly linked). Create global entries for them so they
    // appear in dynsym and can be resolved by the dynamic linker at runtime.
    for name in &plt_names {
        if !globals.contains_key(name) {
            globals.insert(name.clone(), GlobalSymbol {
                value: 0, size: 0, info: (STB_GLOBAL << 4) | STT_FUNC,
                defined_in: None, from_lib: None, section_idx: SHN_UNDEF,
                is_dynamic: true, copy_reloc: false, lib_sym_value: 0, version: None,
                plt_idx: None, got_idx: None,
            });
        }
    }

    // Assign PLT indices to global symbols
    for (plt_idx, name) in plt_names.iter().enumerate() {
        if let Some(gsym) = globals.get_mut(name) {
            gsym.plt_idx = Some(plt_idx);
        }
    }

    // Collect symbols that need GOT entries (GOTPCREL references).
    // For undefined symbols, these need R_X86_64_GLOB_DAT relocations
    // (or R_X86_64_TPOFF64 for TLS symbols referenced via GOTTPOFF).
    let mut got_needed_names: Vec<String> = Vec::new();
    let mut tls_got_names: HashSet<String> = HashSet::new();
    for obj in objects.iter() {
        for sec_relas in &obj.relocations {
            for rela in sec_relas {
                let si = rela.sym_idx as usize;
                if si >= obj.symbols.len() { continue; }
                let sym = &obj.symbols[si];
                if sym.name.is_empty() { continue; }
                // Skip local symbols - they don't need GOT entries in dynsym
                // (e.g. static _Thread_local variables referenced via GOTTPOFF)
                if sym.is_local() { continue; }
                match rela.rela_type {
                    R_X86_64_GOTPCREL | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX | R_X86_64_GOTTPOFF => {
                        if !got_needed_names.contains(&sym.name) {
                            got_needed_names.push(sym.name.clone());
                        }
                        // Track TLS symbols for proper dynamic relocation emission
                        if sym.sym_type() == STT_TLS {
                            tls_got_names.insert(sym.name.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // Ensure GOT-referenced undefined symbols are in globals for dynsym
    for name in &got_needed_names {
        if !globals.contains_key(name) {
            // Use STT_TLS for TLS symbols so the dynamic symbol table has the
            // correct type, allowing the dynamic linker to resolve them properly.
            let stype = if tls_got_names.contains(name) { STT_TLS } else { STT_FUNC };
            globals.insert(name.clone(), GlobalSymbol {
                value: 0, size: 0, info: (STB_GLOBAL << 4) | stype,
                defined_in: None, from_lib: None, section_idx: SHN_UNDEF,
                is_dynamic: true, copy_reloc: false, lib_sym_value: 0, version: None,
                plt_idx: None, got_idx: None,
            });
        }
    }

    // Pre-scan: collect named global symbols referenced by R_X86_64_64 relocations.
    // These must appear in the dynamic symbol table so the dynamic linker can
    // resolve them (supporting symbol interposition at runtime).
    let mut abs64_sym_names: BTreeSet<String> = BTreeSet::new();
    for obj in objects.iter() {
        for sec_relas in &obj.relocations {
            for rela in sec_relas {
                if rela.rela_type == R_X86_64_64 {
                    let si = rela.sym_idx as usize;
                    if si >= obj.symbols.len() { continue; }
                    let sym = &obj.symbols[si];
                    if !sym.name.is_empty() && !sym.is_local() && sym.sym_type() != STT_SECTION {
                        abs64_sym_names.insert(sym.name.clone());
                    }
                }
            }
        }
    }

    // Collect all defined global symbols for export
    let mut dyn_sym_names: Vec<String> = Vec::new();
    let mut exported: Vec<String> = globals.iter()
        .filter(|(_, g)| {
            g.defined_in.is_some() && !g.is_dynamic
                && (g.info >> 4) != 0 // not STB_LOCAL
                && g.section_idx != SHN_UNDEF
        })
        .map(|(n, _)| n.clone())
        .collect();
    exported.sort();
    for name in exported {
        if !dyn_sym_names.contains(&name) {
            dyn_sym_names.push(name);
        }
    }

    // Also add undefined/dynamic symbols (from -l libs and PLT imports)
    for (name, gsym) in globals.iter() {
        if (gsym.is_dynamic || (gsym.defined_in.is_none() && gsym.section_idx == SHN_UNDEF))
            && !dyn_sym_names.contains(name)
        {
            dyn_sym_names.push(name.clone());
        }
    }

    // Ensure all symbols referenced by R_X86_64_64 data relocations are in dynsym
    for name in &abs64_sym_names {
        if !dyn_sym_names.contains(name) {
            dyn_sym_names.push(name.clone());
        }
    }

    for name in &dyn_sym_names { dynstr.add(name); }

    let dynsym_count = 1 + dyn_sym_names.len();
    let dynsym_size = dynsym_count as u64 * 24;
    let dynstr_size = dynstr.as_bytes().len() as u64;

    // Build .gnu.hash
    // Separate defined (hashed) from undefined (unhashed) symbols.
    // .gnu.hash only includes defined symbols; undefined symbols must come
    // first in the symbol table (before symoffset).
    let mut undef_syms: Vec<String> = Vec::new();
    let mut defined_syms: Vec<String> = Vec::new();
    for name in &dyn_sym_names {
        if let Some(g) = globals.get(name) {
            if g.defined_in.is_some() && g.section_idx != SHN_UNDEF {
                defined_syms.push(name.clone());
            } else {
                undef_syms.push(name.clone());
            }
        } else {
            undef_syms.push(name.clone());
        }
    }
    // Reorder: undefined first, then defined
    dyn_sym_names.clear();
    dyn_sym_names.extend(undef_syms.iter().cloned());
    dyn_sym_names.extend(defined_syms.iter().cloned());

    let gnu_hash_symoffset: usize = 1 + undef_syms.len(); // 1 for null entry + undefs
    let num_hashed = defined_syms.len();
    let gnu_hash_nbuckets = if num_hashed == 0 { 1 } else { num_hashed.next_power_of_two().max(1) } as u32;
    // Scale bloom filter size with number of symbols for efficient lookup.
    // Each 64-bit bloom word can effectively track ~32 symbols (2 bits each).
    // Use next power of two for the number of words needed, minimum 1.
    let gnu_hash_bloom_size: u32 = if num_hashed <= 32 { 1 }
        else { num_hashed.div_ceil(32).next_power_of_two() as u32 };
    let gnu_hash_bloom_shift: u32 = 6;

    let hashed_sym_hashes: Vec<u32> = defined_syms.iter()
        .map(|name| linker_common::gnu_hash(name.as_bytes()))
        .collect();

    let mut bloom_words: Vec<u64> = vec![0u64; gnu_hash_bloom_size as usize];
    for &h in &hashed_sym_hashes {
        let word_idx = ((h / 64) % gnu_hash_bloom_size) as usize;
        bloom_words[word_idx] |= 1u64 << (h as u64 % 64);
        bloom_words[word_idx] |= 1u64 << ((h >> gnu_hash_bloom_shift) as u64 % 64);
    }

    // Sort hashed (defined) symbols by bucket
    if num_hashed > 0 {
        let mut hashed_with_hash: Vec<(String, u32)> = defined_syms.iter()
            .zip(hashed_sym_hashes.iter())
            .map(|(n, &h)| (n.clone(), h))
            .collect();
        hashed_with_hash.sort_by_key(|(_, h)| h % gnu_hash_nbuckets);
        // Update defined portion of dyn_sym_names
        for (i, (name, _)) in hashed_with_hash.iter().enumerate() {
            dyn_sym_names[undef_syms.len() + i] = name.clone();
        }
    }

    let hashed_sym_hashes: Vec<u32> = dyn_sym_names[undef_syms.len()..].iter()
        .map(|name| linker_common::gnu_hash(name.as_bytes()))
        .collect();

    let mut gnu_hash_buckets = vec![0u32; gnu_hash_nbuckets as usize];
    let mut gnu_hash_chains = vec![0u32; num_hashed];
    for (i, &h) in hashed_sym_hashes.iter().enumerate() {
        let bucket = (h % gnu_hash_nbuckets) as usize;
        if gnu_hash_buckets[bucket] == 0 {
            gnu_hash_buckets[bucket] = (gnu_hash_symoffset + i) as u32;
        }
        gnu_hash_chains[i] = h & !1;
    }
    for bucket_idx in 0..gnu_hash_nbuckets as usize {
        if gnu_hash_buckets[bucket_idx] == 0 { continue; }
        let mut last_in_bucket = 0;
        for (i, &h) in hashed_sym_hashes.iter().enumerate() {
            if (h % gnu_hash_nbuckets) as usize == bucket_idx {
                last_in_bucket = i;
            }
        }
        gnu_hash_chains[last_in_bucket] |= 1;
    }

    let gnu_hash_size: u64 = 16 + (gnu_hash_bloom_size as u64 * 8)
        + (gnu_hash_nbuckets as u64 * 4) + (num_hashed as u64 * 4);

    let plt_size = if plt_names.is_empty() { 0u64 } else { 16 + 16 * plt_names.len() as u64 };
    let got_plt_count = if plt_names.is_empty() { 0 } else { 3 + plt_names.len() };
    let got_plt_size = got_plt_count as u64 * 8;
    let rela_plt_size = plt_names.len() as u64 * 24;

    // Count R_X86_64_RELATIVE relocations needed (for internal absolute addresses)
    // We'll collect them during relocation processing
    let has_init_array = output_sections.iter().any(|s| s.name == ".init_array" && s.mem_size > 0);
    let has_fini_array = output_sections.iter().any(|s| s.name == ".fini_array" && s.mem_size > 0);
    let mut dyn_count = needed_sonames.len() as u64 + 10; // 9 fixed entries + DT_NULL
    if soname.is_some() { dyn_count += 1; }
    if has_init_array { dyn_count += 2; }
    if has_fini_array { dyn_count += 2; }
    if !plt_names.is_empty() { dyn_count += 4; } // DT_PLTGOT, DT_PLTRELSZ, DT_PLTREL, DT_JMPREL
    if rpath_string.is_some() { dyn_count += 1; } // DT_RUNPATH or DT_RPATH
    let dynamic_size = dyn_count * 16;

    let has_tls_sections = output_sections.iter().any(|s| s.flags & SHF_TLS != 0 && s.flags & SHF_ALLOC != 0);

    // Identify output sections that have R_X86_64_64 relocations (need RELATIVE
    // relocations at load time). These must go in a writable segment so the
    // dynamic linker can patch them. We track them by output section index.
    let mut sections_with_abs_relocs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for obj in objects.iter() {
        for (sec_idx, sec_relas) in obj.relocations.iter().enumerate() {
            for rela in sec_relas {
                if rela.rela_type == R_X86_64_64 {
                    // Find which output section this input section maps to
                    let obj_idx_search = objects.iter().position(|o| std::ptr::eq(o, obj));
                    if let Some(oi) = obj_idx_search {
                        if let Some(&(out_idx, _)) = section_map.get(&(oi, sec_idx)) {
                            sections_with_abs_relocs.insert(out_idx);
                        }
                    }
                }
            }
        }
    }

    // A section is "pure rodata" if it's read-only and has no absolute relocations.
    // Sections with absolute relocations go in the RW segment (as .data.rel.ro).
    let is_pure_rodata = |idx: usize, sec: &OutputSection| -> bool {
        sec.flags & SHF_ALLOC != 0
            && sec.flags & SHF_EXECINSTR == 0
            && sec.flags & SHF_WRITE == 0
            && sec.flags & SHF_TLS == 0
            && sec.sh_type != SHT_NOBITS
            && !sections_with_abs_relocs.contains(&idx)
    };
    let is_relro_rodata = |idx: usize, sec: &OutputSection| -> bool {
        sec.flags & SHF_ALLOC != 0
            && sec.flags & SHF_EXECINSTR == 0
            && sec.flags & SHF_WRITE == 0
            && sec.flags & SHF_TLS == 0
            && sec.sh_type != SHT_NOBITS
            && sections_with_abs_relocs.contains(&idx)
    };

    // phdrs: PHDR, LOAD(ro), LOAD(text), LOAD(rodata), LOAD(rw), DYNAMIC, GNU_STACK, [GNU_RELRO], [TLS]
    let has_relro = !sections_with_abs_relocs.is_empty();
    let mut phdr_count: u64 = 7; // base count
    if has_tls_sections { phdr_count += 1; }
    if has_relro { phdr_count += 1; }
    let phdr_total_size = phdr_count * 56;

    // === Layout ===
    let mut offset = 64 + phdr_total_size;

    offset = (offset + 7) & !7;
    let gnu_hash_offset = offset; let gnu_hash_addr = base_addr + offset; offset += gnu_hash_size;
    offset = (offset + 7) & !7;
    let dynsym_offset = offset; let dynsym_addr = base_addr + offset; offset += dynsym_size;
    let dynstr_offset = offset; let dynstr_addr = base_addr + offset; offset += dynstr_size;

    // Text segment
    offset = (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let text_page_offset = offset;
    let text_page_addr = base_addr + offset;
    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_EXECINSTR != 0 && sec.flags & SHF_ALLOC != 0 {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = base_addr + offset;
            sec.file_offset = offset;
            offset += sec.mem_size;
        }
    }
    // PLT goes at the end of the text segment
    let (plt_addr, plt_offset) = if plt_size > 0 {
        offset = (offset + 15) & !15;
        let a = base_addr + offset; let o = offset; offset += plt_size; (a, o)
    } else { (0u64, 0u64) };
    let text_total_size = offset - text_page_offset;

    // Rodata segment - only pure rodata (no absolute relocations)
    offset = (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let rodata_page_offset = offset;
    let rodata_page_addr = base_addr + offset;
    for (idx, sec) in output_sections.iter_mut().enumerate() {
        if is_pure_rodata(idx, sec) {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = base_addr + offset;
            sec.file_offset = offset;
            offset += sec.mem_size;
        }
    }
    let rodata_total_size = offset - rodata_page_offset;

    // RW segment - includes RELRO sections (rodata with abs relocs), then linker
    // data structures, then actual writable data
    offset = (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let rw_page_offset = offset;
    let rw_page_addr = base_addr + offset;

    // First: RELRO sections (rodata that needs dynamic relocations)
    let _relro_start_offset = offset;
    for (idx, sec) in output_sections.iter_mut().enumerate() {
        if is_relro_rodata(idx, sec) {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = base_addr + offset;
            sec.file_offset = offset;
            offset += sec.mem_size;
        }
    }

    let mut init_array_addr = 0u64; let mut init_array_size = 0u64;
    let mut fini_array_addr = 0u64; let mut fini_array_size = 0u64;

    for sec in output_sections.iter_mut() {
        if sec.name == ".init_array" {
            let a = sec.alignment.max(8);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = base_addr + offset; sec.file_offset = offset;
            init_array_addr = sec.addr; init_array_size = sec.mem_size;
            offset += sec.mem_size; break;
        }
    }
    for sec in output_sections.iter_mut() {
        if sec.name == ".fini_array" {
            let a = sec.alignment.max(8);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = base_addr + offset; sec.file_offset = offset;
            fini_array_addr = sec.addr; fini_array_size = sec.mem_size;
            offset += sec.mem_size; break;
        }
    }

    // GOT entries were already collected into got_needed_names above.
    let got_needed = &got_needed_names;

    // Reserve space for .rela.dyn (will be filled later)
    offset = (offset + 7) & !7;
    let rela_dyn_offset = offset;
    let rela_dyn_addr = base_addr + offset;
    // Each R_X86_64_64 reloc in input becomes one R_X86_64_RELATIVE entry.
    let mut max_rela_count: usize = 0;
    for obj in objects.iter() {
        for sec_relas in &obj.relocations {
            for rela in sec_relas {
                if rela.rela_type == R_X86_64_64 {
                    max_rela_count += 1;
                }
            }
        }
    }
    // Also init_array/fini_array entries are pointers
    for sec in output_sections.iter() {
        if sec.name == ".init_array" || sec.name == ".fini_array" {
            max_rela_count += (sec.mem_size / 8) as usize;
        }
    }
    // GOT entries need either RELATIVE (local) or GLOB_DAT (external) relocations
    max_rela_count += got_needed.len();
    let rela_dyn_max_size = max_rela_count as u64 * 24;
    offset += rela_dyn_max_size;

    // .rela.plt (JMPREL) for PLT GOT entries
    offset = (offset + 7) & !7;
    let rela_plt_offset = offset;
    let rela_plt_addr = base_addr + offset;
    offset += rela_plt_size;

    offset = (offset + 7) & !7;
    let dynamic_offset = offset; let dynamic_addr = base_addr + offset; offset += dynamic_size;

    // End of RELRO region (page-aligned up for PT_GNU_RELRO).
    // Everything after this must be on a new page so that mprotect(PROT_READ)
    // on the RELRO region doesn't affect writable data (GOT.PLT, GOT, .data, .bss).
    let relro_end_offset = (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let relro_end_addr = base_addr + relro_end_offset;
    if has_relro {
        offset = relro_end_offset; // advance to page boundary
    }

    // .got.plt entries - MUST be after RELRO boundary since dynamic linker
    // needs to write to them during lazy PLT resolution
    offset = (offset + 7) & !7;
    let got_plt_offset = offset;
    let got_plt_addr = base_addr + offset;
    offset += got_plt_size;

    // GOT for locally-resolved symbols
    let got_offset = offset; let got_addr = base_addr + offset;
    let got_size = got_needed.len() as u64 * 8;
    offset += got_size;

    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_WRITE != 0 &&
           sec.sh_type != SHT_NOBITS && sec.name != ".init_array" && sec.name != ".fini_array" &&
           sec.flags & SHF_TLS == 0 {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = base_addr + offset; sec.file_offset = offset;
            offset += sec.mem_size;
        }
    }

    // TLS sections
    let mut tls_addr = 0u64;
    let mut tls_file_offset = 0u64;
    let mut tls_file_size = 0u64;
    let mut tls_mem_size = 0u64;
    let mut tls_align = 1u64;
    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_TLS != 0 && sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = base_addr + offset; sec.file_offset = offset;
            if tls_addr == 0 { tls_addr = sec.addr; tls_file_offset = offset; tls_align = a; }
            tls_file_size += sec.mem_size;
            tls_mem_size += sec.mem_size;
            offset += sec.mem_size;
        }
    }
    if tls_addr == 0 && has_tls_sections {
        tls_addr = base_addr + offset;
        tls_file_offset = offset;
    }
    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_TLS != 0 && sec.sh_type == SHT_NOBITS {
            let a = sec.alignment.max(1);
            let aligned = (tls_mem_size + a - 1) & !(a - 1);
            sec.addr = tls_addr + aligned; sec.file_offset = offset;
            tls_mem_size = aligned + sec.mem_size;
            if a > tls_align { tls_align = a; }
        }
    }
    tls_mem_size = (tls_mem_size + tls_align - 1) & !(tls_align - 1);
    let has_tls = tls_addr != 0;

    let bss_addr = base_addr + offset;
    let mut bss_size = 0u64;
    for sec in output_sections.iter_mut() {
        if sec.sh_type == SHT_NOBITS && sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_TLS == 0 {
            let a = sec.alignment.max(1);
            let aligned = (bss_addr + bss_size + a - 1) & !(a - 1);
            bss_size = aligned - bss_addr + sec.mem_size;
            sec.addr = aligned; sec.file_offset = offset;
        }
    }

    // Merge section data
    for sec in output_sections.iter_mut() {
        if sec.sh_type == SHT_NOBITS { continue; }
        let mut data = vec![0u8; sec.mem_size as usize];
        for input in &sec.inputs {
            let sd = &objects[input.object_idx].section_data[input.section_idx];
            let s = input.output_offset as usize;
            let e = s + sd.len();
            if e <= data.len() && !sd.is_empty() { data[s..e].copy_from_slice(sd); }
        }
        sec.data = data;
    }

    // Update global symbol addresses
    for (_, gsym) in globals.iter_mut() {
        if let Some(obj_idx) = gsym.defined_in {
            if gsym.section_idx == SHN_COMMON || gsym.section_idx == 0xffff {
                if let Some(bss_sec) = output_sections.iter().find(|s| s.name == ".bss") {
                    gsym.value += bss_sec.addr;
                }
            } else if gsym.section_idx != SHN_UNDEF && gsym.section_idx != SHN_ABS {
                let si = gsym.section_idx as usize;
                if let Some(&(oi, so)) = section_map.get(&(obj_idx, si)) {
                    gsym.value += output_sections[oi].addr + so;
                }
            }
        }
    }

    // Define linker-provided symbols
    let linker_addrs = LinkerSymbolAddresses {
        base_addr,
        got_addr,
        dynamic_addr,
        bss_addr,
        bss_size,
        text_end: text_page_addr + text_total_size,
        data_start: rw_page_addr,
        init_array_start: init_array_addr,
        init_array_size,
        fini_array_start: fini_array_addr,
        fini_array_size,
        preinit_array_start: 0,
        preinit_array_size: 0,
        rela_iplt_start: 0,
        rela_iplt_size: 0,
    };
    for sym in &get_standard_linker_symbols(&linker_addrs) {
        let entry = globals.entry(sym.name.to_string()).or_insert(GlobalSymbol {
            value: 0, size: 0, info: (sym.binding << 4),
            defined_in: None, from_lib: None, plt_idx: None, got_idx: None,
            section_idx: SHN_ABS, is_dynamic: false, copy_reloc: false, lib_sym_value: 0, version: None,
        });
        if entry.defined_in.is_none() && !entry.is_dynamic {
            entry.value = sym.value;
            entry.defined_in = Some(usize::MAX);
            entry.section_idx = SHN_ABS;
        }
    }

    // Auto-generate __start_<section> / __stop_<section> symbols (GNU ld feature)
    for (name, addr) in linker_common::resolve_start_stop_symbols(output_sections) {
        if let Some(entry) = globals.get_mut(&name) {
            if entry.defined_in.is_none() && !entry.is_dynamic {
                entry.value = addr;
                entry.defined_in = Some(usize::MAX);
                entry.section_idx = SHN_ABS;
            }
        }
    }

    // === Build output buffer ===
    let file_size = offset as usize;
    let mut out = vec![0u8; file_size];

    // ELF header
    out[0..4].copy_from_slice(&ELF_MAGIC);
    out[4] = ELFCLASS64; out[5] = ELFDATA2LSB; out[6] = 1;
    w16(&mut out, 16, ET_DYN); // Shared object
    w16(&mut out, 18, EM_X86_64); w32(&mut out, 20, 1);
    w64(&mut out, 24, 0); // e_entry = 0 for shared libraries
    w64(&mut out, 32, 64); // e_phoff
    w64(&mut out, 40, 0); // e_shoff = 0 (no section headers for now)
    w32(&mut out, 48, 0); w16(&mut out, 52, 64); w16(&mut out, 54, 56);
    w16(&mut out, 56, phdr_count as u16); w16(&mut out, 58, 64); w16(&mut out, 60, 0); w16(&mut out, 62, 0);

    // Program headers
    let mut ph = 64usize;
    wphdr(&mut out, ph, PT_PHDR, PF_R, 64, base_addr + 64, phdr_total_size, phdr_total_size, 8); ph += 56;
    let ro_seg_end = dynstr_offset + dynstr_size;
    wphdr(&mut out, ph, PT_LOAD, PF_R, 0, base_addr, ro_seg_end, ro_seg_end, PAGE_SIZE); ph += 56;
    if text_total_size > 0 {
        wphdr(&mut out, ph, PT_LOAD, PF_R|PF_X, text_page_offset, text_page_addr, text_total_size, text_total_size, PAGE_SIZE); ph += 56;
    } else {
        wphdr(&mut out, ph, PT_LOAD, PF_R|PF_X, text_page_offset, text_page_addr, 0, 0, PAGE_SIZE); ph += 56;
    }
    wphdr(&mut out, ph, PT_LOAD, PF_R, rodata_page_offset, rodata_page_addr, rodata_total_size, rodata_total_size, PAGE_SIZE); ph += 56;
    let rw_filesz = offset - rw_page_offset;
    let rw_memsz = if bss_size > 0 { (bss_addr + bss_size) - rw_page_addr } else { rw_filesz };
    wphdr(&mut out, ph, PT_LOAD, PF_R|PF_W, rw_page_offset, rw_page_addr, rw_filesz, rw_memsz, PAGE_SIZE); ph += 56;
    wphdr(&mut out, ph, PT_DYNAMIC, PF_R|PF_W, dynamic_offset, dynamic_addr, dynamic_size, dynamic_size, 8); ph += 56;
    wphdr(&mut out, ph, PT_GNU_STACK, PF_R|PF_W, 0, 0, 0, 0, 0x10); ph += 56;
    if has_relro {
        let relro_filesz = relro_end_addr - rw_page_addr;
        wphdr(&mut out, ph, PT_GNU_RELRO, PF_R, rw_page_offset, rw_page_addr, relro_filesz, relro_filesz, 1); ph += 56;
    }
    if has_tls {
        wphdr(&mut out, ph, PT_TLS, PF_R, tls_file_offset, tls_addr, tls_file_size, tls_mem_size, tls_align);
    }

    // .gnu.hash
    let gh = gnu_hash_offset as usize;
    w32(&mut out, gh, gnu_hash_nbuckets);
    w32(&mut out, gh+4, gnu_hash_symoffset as u32);
    w32(&mut out, gh+8, gnu_hash_bloom_size);
    w32(&mut out, gh+12, gnu_hash_bloom_shift);
    let bloom_off = gh + 16;
    for (i, &bw) in bloom_words.iter().enumerate() {
        w64(&mut out, bloom_off + i * 8, bw);
    }
    let buckets_off = bloom_off + (gnu_hash_bloom_size as usize * 8);
    for (i, &b) in gnu_hash_buckets.iter().enumerate() {
        w32(&mut out, buckets_off + i * 4, b);
    }
    let chains_off = buckets_off + (gnu_hash_nbuckets as usize * 4);
    for (i, &c) in gnu_hash_chains.iter().enumerate() {
        w32(&mut out, chains_off + i * 4, c);
    }

    // .dynsym
    let mut ds = dynsym_offset as usize + 24; // skip null entry
    for name in &dyn_sym_names {
        let no = dynstr.get_offset(name) as u32;
        w32(&mut out, ds, no);
        if let Some(gsym) = globals.get(name) {
            if gsym.defined_in.is_some() && !gsym.is_dynamic && gsym.section_idx != SHN_UNDEF {
                // Exported defined symbol: preserve original st_info (type + binding)
                if ds+5 < out.len() { out[ds+4] = gsym.info; out[ds+5] = 0; }
                // shndx=1: marks symbol as defined (non-UNDEF). The dynamic linker
                // only checks UNDEF vs defined, not the actual section index.
                w16(&mut out, ds+6, 1);
                // For TLS symbols, the value must be the offset within the TLS segment,
                // not the virtual address. The dynamic linker uses this offset to
                // compute the thread-pointer-relative address.
                let sym_val = if (gsym.info & 0xf) == STT_TLS && tls_mem_size > 0 {
                    gsym.value - tls_addr
                } else {
                    gsym.value
                };
                w64(&mut out, ds+8, sym_val);
                w64(&mut out, ds+16, gsym.size);
            } else {
                // Undefined symbol (from -l dependencies or weak refs)
                // Preserve original binding (STB_WEAK vs STB_GLOBAL) and type
                let bind = gsym.info >> 4;
                let stype = gsym.info & 0xf;
                let st_info = (bind << 4) | if stype != 0 { stype } else { STT_FUNC };
                if ds+5 < out.len() { out[ds+4] = st_info; out[ds+5] = 0; }
                w16(&mut out, ds+6, 0); w64(&mut out, ds+8, 0); w64(&mut out, ds+16, 0);
            }
        } else {
            if ds+5 < out.len() { out[ds+4] = (STB_GLOBAL << 4) | STT_FUNC; out[ds+5] = 0; }
            w16(&mut out, ds+6, 0); w64(&mut out, ds+8, 0); w64(&mut out, ds+16, 0);
        }
        ds += 24;
    }

    // .dynstr
    write_bytes(&mut out, dynstr_offset as usize, dynstr.as_bytes());

    // Section data
    for sec in output_sections.iter() {
        if sec.sh_type == SHT_NOBITS || sec.data.is_empty() { continue; }
        write_bytes(&mut out, sec.file_offset as usize, &sec.data);
    }

    // .plt - PLT stubs for external dynamic symbols
    if plt_size > 0 {
        let po = plt_offset as usize;
        // PLT[0] - the resolver stub (16 bytes)
        out[po] = 0xff; out[po+1] = 0x35; // push [GOT+8] (link_map)
        w32(&mut out, po+2, ((got_plt_addr+8) as i64 - (plt_addr+6) as i64) as u32);
        out[po+6] = 0xff; out[po+7] = 0x25; // jmp [GOT+16] (resolver)
        w32(&mut out, po+8, ((got_plt_addr+16) as i64 - (plt_addr+12) as i64) as u32);
        for i in 12..16 { out[po+i] = 0x90; } // nop padding

        // PLT[1..N] - per-symbol stubs (16 bytes each)
        for (i, _) in plt_names.iter().enumerate() {
            let ep = po + 16 + i * 16;
            let pea = plt_addr + 16 + i as u64 * 16;
            let gea = got_plt_addr + 24 + i as u64 * 8;
            out[ep] = 0xff; out[ep+1] = 0x25; // jmp [GOT.PLT slot]
            w32(&mut out, ep+2, (gea as i64 - (pea+6) as i64) as u32);
            out[ep+6] = 0x68; w32(&mut out, ep+7, i as u32); // push <plt_index>
            out[ep+11] = 0xe9; // jmp PLT[0]
            w32(&mut out, ep+12, (plt_addr as i64 - (pea+16) as i64) as u32);
        }
    }

    // .got.plt
    if got_plt_size > 0 {
        let gp = got_plt_offset as usize;
        w64(&mut out, gp, dynamic_addr);  // GOT[0] = _DYNAMIC
        w64(&mut out, gp+8, 0);           // GOT[1] = 0 (link_map, filled by ld.so)
        w64(&mut out, gp+16, 0);          // GOT[2] = 0 (resolver, filled by ld.so)
        for (i, _) in plt_names.iter().enumerate() {
            // GOT[3+i] = address of "push <index>" in PLT stub (lazy binding)
            w64(&mut out, gp+24+i*8, plt_addr + 16 + i as u64 * 16 + 6);
        }
    }

    // .rela.plt - JMPREL relocations for GOT.PLT entries
    if rela_plt_size > 0 {
        let mut rp = rela_plt_offset as usize;
        let gpb = got_plt_addr + 24; // base of per-symbol GOT.PLT slots
        for (i, name) in plt_names.iter().enumerate() {
            let gea = gpb + i as u64 * 8;
            // Find symbol index in dynsym
            let si = dyn_sym_names.iter().position(|n| n == name).map(|j| j+1).unwrap_or(0) as u64;
            w64(&mut out, rp, gea);             // r_offset = GOT.PLT slot address
            w64(&mut out, rp+8, (si << 32) | R_X86_64_JUMP_SLOT as u64);
            w64(&mut out, rp+16, 0);            // r_addend = 0
            rp += 24;
        }
    }

    // Build GOT entries map
    let mut got_sym_addrs: HashMap<String, u64> = HashMap::new();
    for (i, name) in got_needed.iter().enumerate() {
        let gea = got_addr + i as u64 * 8;
        got_sym_addrs.insert(name.clone(), gea);
        // Fill GOT with resolved symbol value (skip TLS - handled in reloc loop below)
        if !tls_got_names.contains(name) {
            if let Some(gsym) = globals.get(name) {
                if gsym.defined_in.is_some() && !gsym.is_dynamic {
                    w64(&mut out, (got_offset + i as u64 * 8) as usize, gsym.value);
                }
            }
        }
    }

    // Apply relocations and collect dynamic relocation entries
    let globals_snap: HashMap<String, GlobalSymbol> = globals.clone();
    let mut rela_dyn_entries: Vec<(u64, u64)> = Vec::new(); // (offset, value) for RELATIVE relocs
    let mut glob_dat_entries: Vec<(u64, String)> = Vec::new(); // (offset, sym_name) for GLOB_DAT relocs
    let mut tpoff64_entries: Vec<(u64, String)> = Vec::new(); // (offset, sym_name) for R_X86_64_TPOFF64 relocs
    let mut abs64_entries: Vec<(u64, String, i64)> = Vec::new(); // (offset, sym_name, addend) for R_X86_64_64 relocs

    // Add RELATIVE entries for GOT entries that point to local symbols,
    // GLOB_DAT entries for GOT entries that point to external non-TLS symbols,
    // and TPOFF64 entries for GOT entries that point to external TLS symbols.
    for (i, name) in got_needed.iter().enumerate() {
        let gea = got_addr + i as u64 * 8;
        let is_tls = tls_got_names.contains(name);
        if let Some(gsym) = globals_snap.get(name) {
            if gsym.defined_in.is_some() && !gsym.is_dynamic && gsym.section_idx != SHN_UNDEF {
                if is_tls {
                    // Locally-defined TLS symbol: compute TPOFF and store in GOT
                    let tpoff = (gsym.value as i64 - tls_addr as i64) - tls_mem_size as i64;
                    w64(&mut out, (got_offset + i as u64 * 8) as usize, tpoff as u64);
                    // No dynamic relocation needed - statically resolved
                } else {
                    rela_dyn_entries.push((gea, gsym.value));
                }
            } else if is_tls {
                // External TLS symbol - needs R_X86_64_TPOFF64 dynamic relocation
                tpoff64_entries.push((gea, name.clone()));
            } else {
                // External non-TLS symbol - needs GLOB_DAT
                glob_dat_entries.push((gea, name.clone()));
            }
        } else if is_tls {
            // Unknown TLS symbol - needs R_X86_64_TPOFF64
            tpoff64_entries.push((gea, name.clone()));
        } else {
            // Unknown symbol - needs GLOB_DAT
            glob_dat_entries.push((gea, name.clone()));
        }
    }

    for obj_idx in 0..objects.len() {
        for sec_idx in 0..objects[obj_idx].sections.len() {
            let relas = &objects[obj_idx].relocations[sec_idx];
            if relas.is_empty() { continue; }
            let (out_idx, sec_off) = match section_map.get(&(obj_idx, sec_idx)) {
                Some(&v) => v, None => continue,
            };
            let sa = output_sections[out_idx].addr;
            let sfo = output_sections[out_idx].file_offset;

            for rela in relas {
                let si = rela.sym_idx as usize;
                if si >= objects[obj_idx].symbols.len() { continue; }
                let sym = &objects[obj_idx].symbols[si];
                let p = sa + sec_off + rela.offset;
                let fp = (sfo + sec_off + rela.offset) as usize;
                let a = rela.addend;
                let s = resolve_sym(obj_idx, sym, &globals_snap, section_map, output_sections, plt_addr);

                match rela.rela_type {
                    R_X86_64_64 => {
                        let val = (s as i64 + a) as u64;
                        w64(&mut out, fp, val);
                        // Determine what kind of dynamic relocation to emit.
                        // Named global/weak symbols need R_X86_64_64 dynamic relocs
                        // (with symbol index) to support symbol interposition.
                        // Section symbols and local symbols use R_X86_64_RELATIVE.
                        let is_named_global = !sym.name.is_empty()
                            && !sym.is_local()
                            && sym.sym_type() != STT_SECTION;
                        if is_named_global {
                            abs64_entries.push((p, sym.name.clone(), a));
                        } else if s != 0 {
                            rela_dyn_entries.push((p, val));
                        }
                    }
                    R_X86_64_PC32 | R_X86_64_PLT32 => {
                        // For dynamic symbols, redirect through PLT
                        let t = if !sym.name.is_empty() && !sym.is_local() {
                            if let Some(g) = globals_snap.get(&sym.name) {
                                if let Some(pi) = g.plt_idx {
                                    plt_addr + 16 + pi as u64 * 16
                                } else { s }
                            } else { s }
                        } else { s };
                        w32(&mut out, fp, (t as i64 + a - p as i64) as u32);
                    }
                    // TODO: R_X86_64_32/32S are not position-independent and should
                    // ideally emit a diagnostic when used in shared libraries. For now
                    // we apply them statically which works for simple cases but may fail
                    // if the library is loaded at a high address.
                    R_X86_64_32 => { w32(&mut out, fp, (s as i64 + a) as u32); }
                    R_X86_64_32S => { w32(&mut out, fp, (s as i64 + a) as u32); }
                    R_X86_64_GOTPCREL | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => {
                        if let Some(&gea) = got_sym_addrs.get(&sym.name) {
                            w32(&mut out, fp, (gea as i64 + a - p as i64) as u32);
                        } else if (rela.rela_type == R_X86_64_GOTPCRELX || rela.rela_type == R_X86_64_REX_GOTPCRELX)
                                  && !sym.name.is_empty() {
                            // GOT relaxation: convert to LEA
                            if let Some(g) = globals_snap.get(&sym.name) {
                                if g.defined_in.is_some() {
                                    if fp >= 2 && fp < out.len() && out[fp-2] == 0x8b {
                                        out[fp-2] = 0x8d;
                                    }
                                    w32(&mut out, fp, (s as i64 + a - p as i64) as u32);
                                    continue;
                                }
                            }
                            w32(&mut out, fp, (s as i64 + a - p as i64) as u32);
                        } else {
                            w32(&mut out, fp, (s as i64 + a - p as i64) as u32);
                        }
                    }
                    R_X86_64_PC64 => { w64(&mut out, fp, (s as i64 + a - p as i64) as u64); }
                    R_X86_64_GOTTPOFF => {
                        // TLS Initial-Exec: point the instruction at the GOT entry.
                        // For locally-defined TLS symbols, the GOT entry was already
                        // filled with the static TPOFF value above. For external TLS
                        // symbols, the dynamic linker fills the GOT slot at load time
                        // via R_X86_64_TPOFF64.
                        if let Some(&gea) = got_sym_addrs.get(&sym.name) {
                            // Only fill the GOT entry statically for locally-defined symbols
                            let is_local_tls = if let Some(g) = globals_snap.get(&sym.name) {
                                g.defined_in.is_some() && !g.is_dynamic && g.section_idx != SHN_UNDEF
                            } else { false };
                            if is_local_tls {
                                let tpoff = (s as i64 - tls_addr as i64) - tls_mem_size as i64;
                                w64(&mut out, (got_offset + (gea - got_addr)) as usize, tpoff as u64);
                            }
                            // Patch the instruction to reference the GOT entry
                            w32(&mut out, fp, (gea as i64 + a - p as i64) as u32);
                        } else {
                            // No GOT entry: IE-to-LE relaxation for locally-resolved symbols
                            let tpoff = (s as i64 - tls_addr as i64) - tls_mem_size as i64;
                            if fp >= 2 && fp + 4 <= out.len() && out[fp-2] == 0x8b {
                                let modrm = out[fp-1];
                                let reg = (modrm >> 3) & 7;
                                out[fp-2] = 0xc7;
                                out[fp-1] = 0xc0 | reg;
                                w32(&mut out, fp, (tpoff + a) as u32);
                            }
                        }
                    }
                    R_X86_64_TPOFF32 => {
                        let tpoff = (s as i64 - tls_addr as i64) - tls_mem_size as i64;
                        w32(&mut out, fp, (tpoff + a) as u32);
                    }
                    R_X86_64_NONE => {}
                    other => {
                        eprintln!("warning: unsupported relocation type {} for '{}' in shared library", other, sym.name);
                    }
                }
            }
        }
    }

    // Write .rela.dyn entries
    let relative_count = rela_dyn_entries.len();
    let total_rela_count = relative_count + glob_dat_entries.len() + tpoff64_entries.len() + abs64_entries.len();
    let rela_dyn_size = total_rela_count as u64 * 24;
    let mut rd = rela_dyn_offset as usize;
    // First: R_X86_64_RELATIVE entries (type 8, no symbol)
    for (rel_offset, rel_value) in &rela_dyn_entries {
        if rd + 24 <= out.len() {
            w64(&mut out, rd, *rel_offset);     // r_offset
            w64(&mut out, rd+8, R_X86_64_RELATIVE as u64); // r_info (sym 0)
            w64(&mut out, rd+16, *rel_value);   // r_addend = runtime value
            rd += 24;
        }
    }
    // Then: R_X86_64_GLOB_DAT entries (type 6, with symbol index)
    for (rel_offset, sym_name) in &glob_dat_entries {
        let si = dyn_sym_names.iter().position(|n| n == sym_name).map(|j| j + 1).unwrap_or(0) as u64;
        if rd + 24 <= out.len() {
            w64(&mut out, rd, *rel_offset);         // r_offset = GOT entry address
            w64(&mut out, rd+8, (si << 32) | R_X86_64_GLOB_DAT as u64);
            w64(&mut out, rd+16, 0);                 // r_addend = 0
            rd += 24;
        }
    }
    // Then: R_X86_64_TPOFF64 entries (type 18, with symbol index) for TLS GOT entries.
    // The dynamic linker fills these GOT slots with the thread-pointer offset of the
    // TLS symbol, so that `%fs:0 + GOT[n]` gives the correct address.
    for (rel_offset, sym_name) in &tpoff64_entries {
        let si = dyn_sym_names.iter().position(|n| n == sym_name).map(|j| j + 1).unwrap_or(0) as u64;
        if rd + 24 <= out.len() {
            w64(&mut out, rd, *rel_offset);         // r_offset = GOT entry address
            w64(&mut out, rd+8, (si << 32) | R_X86_64_TPOFF64 as u64);
            w64(&mut out, rd+16, 0);                 // r_addend = 0
            rd += 24;
        }
    }
    // Then: R_X86_64_64 entries (type 1, with symbol index) for named symbol
    // references in data sections (function pointer tables, vtables, etc.)
    for (rel_offset, sym_name, addend) in &abs64_entries {
        let si = dyn_sym_names.iter().position(|n| n == sym_name).map(|j| j + 1).unwrap_or(0) as u64;
        if rd + 24 <= out.len() {
            w64(&mut out, rd, *rel_offset);         // r_offset
            w64(&mut out, rd+8, (si << 32) | R_X86_64_64 as u64);
            w64(&mut out, rd+16, *addend as u64);   // r_addend
            rd += 24;
        }
    }

    // .dynamic
    let mut dd = dynamic_offset as usize;
    for lib in needed_sonames {
        let so = dynstr.get_offset(lib);
        w64(&mut out, dd, DT_NEEDED as u64); w64(&mut out, dd+8, so as u64); dd += 16;
    }
    if let Some(ref sn) = soname {
        let so = dynstr.get_offset(sn);
        w64(&mut out, dd, DT_SONAME as u64); w64(&mut out, dd+8, so as u64); dd += 16;
    }
    for &(tag, val) in &[
        (DT_STRTAB, dynstr_addr), (DT_SYMTAB, dynsym_addr), (DT_STRSZ, dynstr_size),
        (DT_SYMENT, 24),
        (DT_RELA, rela_dyn_addr), (DT_RELASZ, rela_dyn_size), (DT_RELAENT, 24),
        (DT_RELACOUNT, relative_count as u64),
        (DT_GNU_HASH, gnu_hash_addr),
        // DT_TEXTREL not needed since we use PIC
    ] {
        w64(&mut out, dd, tag as u64); w64(&mut out, dd+8, val); dd += 16;
    }
    if has_init_array {
        w64(&mut out, dd, DT_INIT_ARRAY as u64); w64(&mut out, dd+8, init_array_addr); dd += 16;
        w64(&mut out, dd, DT_INIT_ARRAYSZ as u64); w64(&mut out, dd+8, init_array_size); dd += 16;
    }
    if has_fini_array {
        w64(&mut out, dd, DT_FINI_ARRAY as u64); w64(&mut out, dd+8, fini_array_addr); dd += 16;
        w64(&mut out, dd, DT_FINI_ARRAYSZ as u64); w64(&mut out, dd+8, fini_array_size); dd += 16;
    }
    if !plt_names.is_empty() {
        w64(&mut out, dd, DT_PLTGOT as u64); w64(&mut out, dd+8, got_plt_addr); dd += 16;
        w64(&mut out, dd, DT_PLTRELSZ as u64); w64(&mut out, dd+8, rela_plt_size); dd += 16;
        w64(&mut out, dd, DT_PLTREL as u64); w64(&mut out, dd+8, DT_RELA as u64); dd += 16;
        w64(&mut out, dd, DT_JMPREL as u64); w64(&mut out, dd+8, rela_plt_addr); dd += 16;
    }
    if let Some(ref rp) = rpath_string {
        let rp_off = dynstr.get_offset(rp) as u64;
        let tag = if use_runpath { DT_RUNPATH } else { DT_RPATH };
        w64(&mut out, dd, tag as u64); w64(&mut out, dd+8, rp_off); dd += 16;
    }
    w64(&mut out, dd, DT_NULL as u64); w64(&mut out, dd+8, 0);

    // === Append section headers ===
    // Build .shstrtab string table
    let mut shstrtab = vec![0u8]; // null byte at offset 0
    let mut shstr_offsets: HashMap<String, u32> = HashMap::new();
    let known_names = [
        ".gnu.hash", ".dynsym", ".dynstr",
        ".rela.dyn", ".rela.plt", ".plt", ".dynamic",
        ".got", ".got.plt", ".init_array", ".fini_array",
        ".tdata", ".tbss", ".bss", ".shstrtab",
    ];
    for name in &known_names {
        let off = shstrtab.len() as u32;
        shstr_offsets.insert(name.to_string(), off);
        shstrtab.extend_from_slice(name.as_bytes());
        shstrtab.push(0);
    }
    // Add merged section names not already in known list
    for sec in output_sections.iter() {
        if !sec.name.is_empty() && !shstr_offsets.contains_key(&sec.name) {
            let off = shstrtab.len() as u32;
            shstr_offsets.insert(sec.name.clone(), off);
            shstrtab.extend_from_slice(sec.name.as_bytes());
            shstrtab.push(0);
        }
    }

    let get_shname = |n: &str| -> u32 { shstr_offsets.get(n).copied().unwrap_or(0) };

    // Helper: write a 64-byte ELF64 section header
    // Use shared write_elf64_shdr from linker_common (aliased locally for brevity)
    let write_shdr_so = linker_common::write_elf64_shdr;

    // Pre-count section indices for cross-references
    let dynsym_shidx: u32 = 2; // NULL=0, .gnu.hash=1, .dynsym=2
    let dynstr_shidx: u32 = 3; // .dynstr=3

    // Count total sections to determine .shstrtab index
    let mut sh_count: u16 = 4; // NULL + .gnu.hash + .dynsym + .dynstr
    if rela_dyn_size > 0 { sh_count += 1; }
    if rela_plt_size > 0 { sh_count += 1; }
    if plt_size > 0 { sh_count += 1; }
    // Merged output sections (non-BSS, non-TLS, non-init/fini)
    for sec in output_sections.iter() {
        if sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS && sec.flags & SHF_TLS == 0
           && sec.name != ".init_array" && sec.name != ".fini_array" {
            sh_count += 1;
        }
    }
    // TLS data + TLS BSS
    for sec in output_sections.iter() {
        if sec.flags & SHF_TLS != 0 && sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS { sh_count += 1; }
    }
    for sec in output_sections.iter() {
        if sec.flags & SHF_TLS != 0 && sec.sh_type == SHT_NOBITS { sh_count += 1; }
    }
    if has_init_array { sh_count += 1; }
    if has_fini_array { sh_count += 1; }
    sh_count += 1; // .dynamic
    if got_plt_size > 0 { sh_count += 1; }
    if got_size > 0 { sh_count += 1; }
    // BSS sections (non-TLS)
    for sec in output_sections.iter() {
        if sec.sh_type == SHT_NOBITS && sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_TLS == 0 { sh_count += 1; }
    }
    let shstrtab_shidx = sh_count; // .shstrtab is the last section
    sh_count += 1;

    // Align and append .shstrtab data
    while out.len() % 8 != 0 { out.push(0); }
    let shstrtab_data_offset = out.len() as u64;
    out.extend_from_slice(&shstrtab);

    // Align section header table to 8 bytes
    while out.len() % 8 != 0 { out.push(0); }
    let shdr_offset = out.len() as u64;

    // Write section headers
    // [0] NULL
    write_shdr_so(&mut out, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // .gnu.hash
    write_shdr_so(&mut out, get_shname(".gnu.hash"), SHT_GNU_HASH, SHF_ALLOC,
               gnu_hash_addr, gnu_hash_offset, gnu_hash_size, dynsym_shidx, 0, 8, 0);
    // .dynsym
    write_shdr_so(&mut out, get_shname(".dynsym"), SHT_DYNSYM, SHF_ALLOC,
               dynsym_addr, dynsym_offset, dynsym_size, dynstr_shidx, 1, 8, 24);
    // .dynstr
    write_shdr_so(&mut out, get_shname(".dynstr"), SHT_STRTAB, SHF_ALLOC,
               dynstr_addr, dynstr_offset, dynstr_size, 0, 0, 1, 0);
    // .rela.dyn
    if rela_dyn_size > 0 {
        write_shdr_so(&mut out, get_shname(".rela.dyn"), SHT_RELA, SHF_ALLOC,
                   rela_dyn_addr, rela_dyn_offset, rela_dyn_size, dynsym_shidx, 0, 8, 24);
    }
    // .rela.plt
    if rela_plt_size > 0 {
        write_shdr_so(&mut out, get_shname(".rela.plt"), SHT_RELA, SHF_ALLOC | 0x40,
                   rela_plt_addr, rela_plt_offset, rela_plt_size, dynsym_shidx, 0, 8, 24);
    }
    // .plt
    if plt_size > 0 {
        write_shdr_so(&mut out, get_shname(".plt"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
                   plt_addr, plt_offset, plt_size, 0, 0, 16, 16);
    }
    // Merged output sections (text/rodata/data, excluding BSS/TLS/init_array/fini_array)
    for sec in output_sections.iter() {
        if sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS && sec.flags & SHF_TLS == 0
           && sec.name != ".init_array" && sec.name != ".fini_array" {
            write_shdr_so(&mut out, get_shname(&sec.name), sec.sh_type, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // TLS data sections (.tdata)
    for sec in output_sections.iter() {
        if sec.flags & SHF_TLS != 0 && sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS {
            write_shdr_so(&mut out, get_shname(&sec.name), sec.sh_type, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // TLS BSS sections (.tbss)
    for sec in output_sections.iter() {
        if sec.flags & SHF_TLS != 0 && sec.sh_type == SHT_NOBITS {
            write_shdr_so(&mut out, get_shname(&sec.name), SHT_NOBITS, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // .init_array
    if has_init_array {
        if let Some(ia_sec) = output_sections.iter().find(|s| s.name == ".init_array") {
            write_shdr_so(&mut out, get_shname(".init_array"), SHT_INIT_ARRAY, SHF_ALLOC | SHF_WRITE,
                       init_array_addr, ia_sec.file_offset, init_array_size, 0, 0, 8, 8);
        }
    }
    // .fini_array
    if has_fini_array {
        if let Some(fa_sec) = output_sections.iter().find(|s| s.name == ".fini_array") {
            write_shdr_so(&mut out, get_shname(".fini_array"), SHT_FINI_ARRAY, SHF_ALLOC | SHF_WRITE,
                       fini_array_addr, fa_sec.file_offset, fini_array_size, 0, 0, 8, 8);
        }
    }
    // .dynamic
    write_shdr_so(&mut out, get_shname(".dynamic"), SHT_DYNAMIC, SHF_ALLOC | SHF_WRITE,
               dynamic_addr, dynamic_offset, dynamic_size, dynstr_shidx, 0, 8, 16);
    // .got.plt
    if got_plt_size > 0 {
        write_shdr_so(&mut out, get_shname(".got.plt"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                   got_plt_addr, got_plt_offset, got_plt_size, 0, 0, 8, 8);
    }
    // .got
    if got_size > 0 {
        write_shdr_so(&mut out, get_shname(".got"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                   got_addr, got_offset, got_size, 0, 0, 8, 8);
    }
    // BSS sections (non-TLS)
    for sec in output_sections.iter() {
        if sec.sh_type == SHT_NOBITS && sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_TLS == 0 {
            write_shdr_so(&mut out, get_shname(&sec.name), SHT_NOBITS, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // .shstrtab (last section)
    write_shdr_so(&mut out, get_shname(".shstrtab"), SHT_STRTAB, 0,
               0, shstrtab_data_offset, shstrtab.len() as u64, 0, 0, 1, 0);

    // Patch ELF header with section header info
    out[40..48].copy_from_slice(&shdr_offset.to_le_bytes());     // e_shoff
    out[58..60].copy_from_slice(&64u16.to_le_bytes());           // e_shentsize
    out[60..62].copy_from_slice(&sh_count.to_le_bytes());        // e_shnum
    out[62..64].copy_from_slice(&shstrtab_shidx.to_le_bytes()); // e_shstrndx

    std::fs::write(output_path, &out).map_err(|e| format!("failed to write '{}': {}", output_path, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(output_path, std::fs::Permissions::from_mode(0o755));
    }
    Ok(())
}
