//! Executable emission for the x86-64 linker.
//!
//! Emits either a statically-linked or dynamically-linked ELF64 executable,
//! depending on whether dynamic symbols are present. Handles PLT/GOT,
//! `.dynamic` section, TLS, IFUNC/IRELATIVE, and copy relocations.

use std::collections::{HashMap, BTreeSet};

use super::elf::*;
use super::types::{GlobalSymbol, BASE_ADDR, PAGE_SIZE, INTERP};
use crate::backend::linker_common::{self, DynStrTab, OutputSection};

pub(super) fn emit_executable(
    objects: &[ElfObject], globals: &mut HashMap<String, GlobalSymbol>,
    output_sections: &mut [OutputSection],
    section_map: &HashMap<(usize, usize), (usize, u64)>,
    plt_names: &[String], got_entries: &[(String, bool)],
    needed_sonames: &[String], output_path: &str,
    export_dynamic: bool, rpath_entries: &[String], use_runpath: bool,
    is_static: bool, ifunc_symbols: &[String],
) -> Result<(), String> {
    let mut dynstr = DynStrTab::new();
    for lib in needed_sonames { dynstr.add(lib); }
    let rpath_string = if rpath_entries.is_empty() { None } else {
        let s = rpath_entries.join(":");
        dynstr.add(&s);
        Some(s)
    };

    // Build dyn_sym_names in two parts:
    // 1. Non-hashed symbols (PLT imports, GLOB_DAT imports) - these are undefined
    // 2. Hashed symbols (copy-reloc symbols) - these are defined and must be
    //    findable through .gnu.hash so the dynamic linker can redirect references
    let mut dyn_sym_names: Vec<String> = Vec::new();
    for name in plt_names {
        if !dyn_sym_names.contains(name) { dyn_sym_names.push(name.clone()); }
    }
    for (name, is_plt) in got_entries {
        if !name.is_empty() && !*is_plt && !dyn_sym_names.contains(name) {
            if let Some(gsym) = globals.get(name) {
                if gsym.is_dynamic && !gsym.copy_reloc {
                    dyn_sym_names.push(name.clone());
                }
            }
        }
    }
    // symoffset = index of first hashed symbol (1-indexed: null symbol is index 0)
    let gnu_hash_symoffset = 1 + dyn_sym_names.len(); // +1 for null entry

    // Collect copy relocation symbols - these go AFTER non-hashed symbols
    // and are included in the .gnu.hash table
    let copy_reloc_syms: Vec<(String, u64)> = globals.iter()
        .filter(|(_, g)| g.copy_reloc)
        .map(|(n, g)| (n.clone(), g.size))
        .collect();
    for (name, _) in &copy_reloc_syms {
        if !dyn_sym_names.contains(name) {
            dyn_sym_names.push(name.clone());
        }
    }

    // When --export-dynamic is used, add all defined global symbols to the
    // dynamic symbol table so shared libraries loaded at runtime (via dlopen)
    // can find symbols from this executable.
    if export_dynamic {
        let mut exported: Vec<String> = globals.iter()
            .filter(|(_, g)| {
                // Export defined, non-dynamic (local to this executable) global symbols
                g.section_idx != SHN_UNDEF && !g.is_dynamic && !g.copy_reloc
                    && (g.info >> 4) != 0 // not STB_LOCAL
            })
            .map(|(n, _)| n.clone())
            .collect();
        exported.sort(); // deterministic output
        for name in exported {
            if !dyn_sym_names.contains(&name) {
                dyn_sym_names.push(name);
            }
        }
    }

    for name in &dyn_sym_names { dynstr.add(name); }

    // ── Build .gnu.version (versym) and .gnu.version_r (verneed) data ──
    //
    // Collect version requirements from dynamic symbols, grouped by library.
    let mut lib_versions: HashMap<String, BTreeSet<String>> = HashMap::new();
    for name in &dyn_sym_names {
        if let Some(gs) = globals.get(name) {
            if gs.is_dynamic {
                if let Some(ref ver) = gs.version {
                    if let Some(ref lib) = gs.from_lib {
                        lib_versions.entry(lib.clone())
                            .or_default()
                            .insert(ver.clone());
                    }
                }
            }
        }
    }

    // Build version index mapping: (library, version_string) -> version index (starting at 2)
    let mut ver_index_map: HashMap<(String, String), u16> = HashMap::new();
    let mut ver_idx: u16 = 2;
    let mut lib_ver_list: Vec<(String, Vec<String>)> = Vec::new();
    let mut sorted_libs: Vec<String> = lib_versions.keys().cloned().collect();
    sorted_libs.sort();
    for lib in &sorted_libs {
        let vers: Vec<String> = lib_versions[lib].iter().cloned().collect();
        for v in &vers {
            ver_index_map.insert((lib.clone(), v.clone()), ver_idx);
            ver_idx += 1;
            // Add version string to dynstr
            dynstr.add(v);
        }
        lib_ver_list.push((lib.clone(), vers));
    }

    // Build .gnu.version_r (verneed) section
    let mut verneed_data: Vec<u8> = Vec::new();
    let mut verneed_count: u32 = 0;
    // Only include libraries that are in our needed list
    let lib_ver_needed: Vec<(String, Vec<String>)> = lib_ver_list.iter()
        .filter(|(lib, _)| needed_sonames.contains(lib))
        .cloned()
        .collect();
    for (lib_i, (lib, vers)) in lib_ver_needed.iter().enumerate() {
        let lib_name_off = dynstr.get_offset(lib);
        let is_last_lib = lib_i == lib_ver_needed.len() - 1;

        // Verneed entry header (16 bytes)
        verneed_data.extend_from_slice(&1u16.to_le_bytes()); // vn_version = 1
        verneed_data.extend_from_slice(&(vers.len() as u16).to_le_bytes()); // vn_cnt
        verneed_data.extend_from_slice(&(lib_name_off as u32).to_le_bytes()); // vn_file
        verneed_data.extend_from_slice(&16u32.to_le_bytes()); // vn_aux (right after header)
        let next_off = if is_last_lib {
            0u32
        } else {
            16 + vers.len() as u32 * 16
        };
        verneed_data.extend_from_slice(&next_off.to_le_bytes()); // vn_next
        verneed_count += 1;

        // Vernaux entries for each version (16 bytes each)
        for (v_i, ver) in vers.iter().enumerate() {
            let ver_name_off = dynstr.get_offset(ver);
            let vidx = ver_index_map[&(lib.clone(), ver.clone())];
            let is_last_ver = v_i == vers.len() - 1;

            let vna_hash = linker_common::sysv_hash(ver.as_bytes());
            verneed_data.extend_from_slice(&vna_hash.to_le_bytes()); // vna_hash
            verneed_data.extend_from_slice(&0u16.to_le_bytes()); // vna_flags
            verneed_data.extend_from_slice(&vidx.to_le_bytes()); // vna_other
            verneed_data.extend_from_slice(&(ver_name_off as u32).to_le_bytes()); // vna_name
            let vna_next: u32 = if is_last_ver { 0 } else { 16 };
            verneed_data.extend_from_slice(&vna_next.to_le_bytes()); // vna_next
        }
    }

    let verneed_size = verneed_data.len() as u64;

    let dynsym_count = 1 + dyn_sym_names.len();
    let dynsym_size = dynsym_count as u64 * 24;
    let dynstr_size = dynstr.as_bytes().len() as u64;
    let rela_plt_size = plt_names.len() as u64 * 24;
    let rela_dyn_glob_count = got_entries.iter().filter(|(n, p)| {
        !n.is_empty() && !*p && globals.get(n).map(|g| g.is_dynamic && !g.copy_reloc && g.plt_idx.is_none()).unwrap_or(false)
    }).count();
    let rela_dyn_count = rela_dyn_glob_count + copy_reloc_syms.len();
    let rela_dyn_size = rela_dyn_count as u64 * 24;

    // Build .gnu.hash table for hashed symbols (copy-reloc + exported)
    // Number of hashed symbols = total symbols after the non-hashed imports
    let num_hashed = dyn_sym_names.len() - (gnu_hash_symoffset - 1);
    let gnu_hash_nbuckets = if num_hashed == 0 { 1 } else { num_hashed.next_power_of_two().max(1) } as u32;
    let gnu_hash_bloom_size: u32 = 1;
    let gnu_hash_bloom_shift: u32 = 6;

    // Compute hashes for hashed symbols
    let hashed_sym_hashes: Vec<u32> = dyn_sym_names[gnu_hash_symoffset - 1..]
        .iter()
        .map(|name| linker_common::gnu_hash(name.as_bytes()))
        .collect();

    // Build bloom filter (single 64-bit word)
    let mut bloom_word: u64 = 0;
    for &h in &hashed_sym_hashes {
        let bit1 = h as u64 % 64;
        let bit2 = (h >> gnu_hash_bloom_shift) as u64 % 64;
        bloom_word |= 1u64 << bit1;
        bloom_word |= 1u64 << bit2;
    }

    // Sort hashed symbols by bucket (hash % nbuckets) for proper chain grouping
    // We need to reorder the hashed portion of dyn_sym_names
    if num_hashed > 0 {
        let hashed_start = gnu_hash_symoffset - 1;
        let mut hashed_with_hash: Vec<(String, u32)> = dyn_sym_names[hashed_start..]
            .iter()
            .zip(hashed_sym_hashes.iter())
            .map(|(n, &h)| (n.clone(), h))
            .collect();
        hashed_with_hash.sort_by_key(|(_, h)| h % gnu_hash_nbuckets);
        for (i, (name, _)) in hashed_with_hash.iter().enumerate() {
            dyn_sym_names[hashed_start + i] = name.clone();
        }
    }

    // Build .gnu.version (versym) - one u16 per dynsym entry
    // Must be built AFTER gnu_hash bucket sort so versym indices match final dynsym order
    let mut versym_data: Vec<u8> = Vec::new();
    // Entry 0: VER_NDX_LOCAL for the null symbol
    versym_data.extend_from_slice(&0u16.to_le_bytes());
    for name in &dyn_sym_names {
        if let Some(gs) = globals.get(name) {
            if gs.is_dynamic {
                if let Some(ref ver) = gs.version {
                    if let Some(ref lib) = gs.from_lib {
                        let idx = ver_index_map.get(&(lib.clone(), ver.clone()))
                            .copied().unwrap_or(1);
                        versym_data.extend_from_slice(&idx.to_le_bytes());
                    } else {
                        versym_data.extend_from_slice(&1u16.to_le_bytes()); // VER_NDX_GLOBAL
                    }
                } else {
                    versym_data.extend_from_slice(&1u16.to_le_bytes()); // VER_NDX_GLOBAL
                }
            } else if gs.section_idx != SHN_UNDEF && gs.value != 0 {
                // Defined/exported symbol: VER_NDX_GLOBAL
                versym_data.extend_from_slice(&1u16.to_le_bytes());
            } else {
                versym_data.extend_from_slice(&0u16.to_le_bytes()); // VER_NDX_LOCAL
            }
        } else {
            versym_data.extend_from_slice(&0u16.to_le_bytes());
        }
    }

    let versym_size = versym_data.len() as u64;

    // Recompute hashes after sorting
    let hashed_sym_hashes: Vec<u32> = dyn_sym_names[gnu_hash_symoffset - 1..]
        .iter()
        .map(|name| linker_common::gnu_hash(name.as_bytes()))
        .collect();

    // Build buckets and chains
    let mut gnu_hash_buckets = vec![0u32; gnu_hash_nbuckets as usize];
    let mut gnu_hash_chains = vec![0u32; num_hashed];
    for (i, &h) in hashed_sym_hashes.iter().enumerate() {
        let bucket = (h % gnu_hash_nbuckets) as usize;
        if gnu_hash_buckets[bucket] == 0 {
            gnu_hash_buckets[bucket] = (gnu_hash_symoffset + i) as u32;
        }
        // Chain value = hash with bit 0 indicating end of chain
        gnu_hash_chains[i] = h & !1; // clear bit 0 (will set later for last in chain)
    }
    // Mark the last symbol in each bucket chain with bit 0 set
    for bucket_idx in 0..gnu_hash_nbuckets as usize {
        if gnu_hash_buckets[bucket_idx] == 0 { continue; }
        let mut last_in_bucket = 0;
        for (i, &h) in hashed_sym_hashes.iter().enumerate() {
            if (h % gnu_hash_nbuckets) as usize == bucket_idx {
                last_in_bucket = i;
            }
        }
        gnu_hash_chains[last_in_bucket] |= 1; // set end-of-chain bit
    }

    // gnu_hash_size = header(16) + bloom(bloom_size*8) + buckets(nbuckets*4) + chains(num_hashed*4)
    let gnu_hash_size: u64 = if is_static { 0 } else {
        16 + (gnu_hash_bloom_size as u64 * 8)
            + (gnu_hash_nbuckets as u64 * 4) + (num_hashed as u64 * 4)
    };
    let plt_size = if is_static || plt_names.is_empty() { 0u64 } else { 16 + 16 * plt_names.len() as u64 };
    let got_plt_size = if is_static { 0u64 } else { (3 + plt_names.len()) as u64 * 8 };
    let got_globdat_count = got_entries.iter().filter(|(n, p)| !n.is_empty() && !*p).count();
    let got_size = got_globdat_count as u64 * 8; // GOT needed even for static (TLS, GOTPCREL)

    let has_init_array = output_sections.iter().any(|s| s.name == ".init_array" && s.mem_size > 0);
    let has_fini_array = output_sections.iter().any(|s| s.name == ".fini_array" && s.mem_size > 0);
    let dynamic_size = if is_static { 0u64 } else {
        let mut dyn_count = needed_sonames.len() as u64 + 14; // fixed entries + NULL
        if has_init_array { dyn_count += 2; }
        if has_fini_array { dyn_count += 2; }
        if rpath_string.is_some() { dyn_count += 1; }
        if verneed_size > 0 { dyn_count += 3; } // DT_VERSYM + DT_VERNEED + DT_VERNEEDNUM
        dyn_count * 16
    };
    // Override other dynamic sizes for static linking
    let dynsym_size = if is_static { 0u64 } else { dynsym_size };
    let dynstr_size = if is_static { 0u64 } else { dynstr_size };
    let rela_plt_size = if is_static { 0u64 } else { rela_plt_size };
    let rela_dyn_size = if is_static { 0u64 } else { rela_dyn_size };
    let versym_size = if is_static { 0u64 } else { versym_size };
    let verneed_size = if is_static { 0u64 } else { verneed_size };

    let has_tls_sections = output_sections.iter().any(|s| s.flags & SHF_TLS != 0 && s.flags & SHF_ALLOC != 0);
    // Static: PHDR, LOAD(ro), LOAD(text), LOAD(rodata), LOAD(rw), GNU_STACK, [TLS]
    // Dynamic: PHDR, INTERP, LOAD(ro), LOAD(text), LOAD(rodata), LOAD(rw), DYNAMIC, GNU_STACK, [TLS]
    let phdr_count: u64 = if is_static {
        if has_tls_sections { 7 } else { 6 }
    } else if has_tls_sections { 9 } else { 8 };
    let phdr_total_size = phdr_count * 56;

    // === Layout ===
    let mut offset = 64 + phdr_total_size;
    let interp_offset = offset;
    let interp_addr = BASE_ADDR + offset;
    if !is_static { offset += INTERP.len() as u64; }

    offset = (offset + 7) & !7;
    let gnu_hash_offset = offset; let gnu_hash_addr = BASE_ADDR + offset; offset += gnu_hash_size;
    offset = (offset + 7) & !7;
    let dynsym_offset = offset; let dynsym_addr = BASE_ADDR + offset; offset += dynsym_size;
    let dynstr_offset = offset; let dynstr_addr = BASE_ADDR + offset; offset += dynstr_size;
    // .gnu.version (versym) - right after dynstr, aligned to 2
    offset = (offset + 1) & !1;
    let versym_offset = offset; let versym_addr = BASE_ADDR + offset;
    if versym_size > 0 { offset += versym_size; }
    // .gnu.version_r (verneed) - aligned to 4
    offset = (offset + 3) & !3;
    let verneed_offset = offset; let verneed_addr = BASE_ADDR + offset;
    if verneed_size > 0 { offset += verneed_size; }
    offset = (offset + 7) & !7;
    let rela_dyn_offset = offset; let rela_dyn_addr = BASE_ADDR + offset; offset += rela_dyn_size;
    offset = (offset + 7) & !7;
    let rela_plt_offset = offset; let rela_plt_addr = BASE_ADDR + offset; offset += rela_plt_size;

    // Text segment
    offset = (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let text_page_offset = offset;
    let text_page_addr = BASE_ADDR + offset;
    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_EXECINSTR != 0 && sec.flags & SHF_ALLOC != 0 {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = BASE_ADDR + offset;
            sec.file_offset = offset;
            offset += sec.mem_size;
        }
    }
    let (plt_addr, plt_offset) = if plt_size > 0 {
        offset = (offset + 15) & !15;
        let a = BASE_ADDR + offset; let o = offset; offset += plt_size; (a, o)
    } else { (0u64, 0u64) };

    // .iplt (IFUNC PLT entries for static linking)
    let num_ifunc = ifunc_symbols.len();
    let iplt_entry_size: u64 = 16; // each IPLT entry: jmp *got(%rip) + padding
    let iplt_total_size = num_ifunc as u64 * iplt_entry_size;
    let (iplt_addr, iplt_offset) = if iplt_total_size > 0 {
        offset = (offset + 15) & !15;
        let a = BASE_ADDR + offset; let o = offset; offset += iplt_total_size; (a, o)
    } else { (0u64, 0u64) };

    let text_total_size = offset - text_page_offset;

    // Rodata segment
    offset = (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let rodata_page_offset = offset;
    let rodata_page_addr = BASE_ADDR + offset;
    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_EXECINSTR == 0 &&
           sec.flags & SHF_WRITE == 0 && sec.sh_type != SHT_NOBITS {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = BASE_ADDR + offset;
            sec.file_offset = offset;
            offset += sec.mem_size;
        }
    }
    let rodata_total_size = offset - rodata_page_offset;

    // RW segment
    offset = (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let rw_page_offset = offset;
    let rw_page_addr = BASE_ADDR + offset;

    let mut init_array_addr = 0u64; let mut init_array_size = 0u64;
    let mut fini_array_addr = 0u64; let mut fini_array_size = 0u64;

    for sec in output_sections.iter_mut() {
        if sec.name == ".init_array" {
            let a = sec.alignment.max(8);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = BASE_ADDR + offset; sec.file_offset = offset;
            init_array_addr = sec.addr; init_array_size = sec.mem_size;
            offset += sec.mem_size; break;
        }
    }
    for sec in output_sections.iter_mut() {
        if sec.name == ".fini_array" {
            let a = sec.alignment.max(8);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = BASE_ADDR + offset; sec.file_offset = offset;
            fini_array_addr = sec.addr; fini_array_size = sec.mem_size;
            offset += sec.mem_size; break;
        }
    }

    offset = (offset + 7) & !7;
    let dynamic_offset = offset; let dynamic_addr = BASE_ADDR + offset; offset += dynamic_size;
    offset = (offset + 7) & !7;
    let got_offset = offset; let got_addr = BASE_ADDR + offset; offset += got_size;
    offset = (offset + 7) & !7;
    let got_plt_offset = offset; let got_plt_addr = BASE_ADDR + offset; offset += got_plt_size;

    // IFUNC GOT (8 bytes per entry, stores resolver addresses initially)
    offset = (offset + 7) & !7;
    let ifunc_got_offset = offset; let ifunc_got_addr = BASE_ADDR + offset;
    let ifunc_got_size = num_ifunc as u64 * 8;
    offset += ifunc_got_size;

    // .rela.iplt (24 bytes per RELA entry for R_X86_64_IRELATIVE)
    offset = (offset + 7) & !7;
    let rela_iplt_offset = offset; let rela_iplt_addr = BASE_ADDR + offset;
    let rela_iplt_size = num_ifunc as u64 * 24;
    offset += rela_iplt_size;

    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_WRITE != 0 &&
           sec.sh_type != SHT_NOBITS && sec.name != ".init_array" && sec.name != ".fini_array" &&
           sec.flags & SHF_TLS == 0 {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = BASE_ADDR + offset; sec.file_offset = offset;
            offset += sec.mem_size;
        }
    }

    // TLS sections (.tdata, .tbss) - place in RW segment, track for PT_TLS
    let mut tls_addr = 0u64;
    let mut tls_file_offset = 0u64;
    let mut tls_file_size = 0u64;
    let mut tls_mem_size = 0u64;
    let mut tls_align = 1u64;
    for sec in output_sections.iter_mut() {
        if sec.flags & SHF_TLS != 0 && sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS {
            let a = sec.alignment.max(1);
            offset = (offset + a - 1) & !(a - 1);
            sec.addr = BASE_ADDR + offset; sec.file_offset = offset;
            if tls_addr == 0 { tls_addr = sec.addr; tls_file_offset = offset; tls_align = a; }
            tls_file_size += sec.mem_size;
            tls_mem_size += sec.mem_size;
            offset += sec.mem_size;
        }
    }
    // If only .tbss (NOBITS TLS) exists with no .tdata, we still need a TLS segment.
    // Set tls_addr/tls_file_offset to the current position so TPOFF calculations work.
    if tls_addr == 0 && has_tls_sections {
        tls_addr = BASE_ADDR + offset;
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
    // Align TLS size to TLS alignment
    tls_mem_size = (tls_mem_size + tls_align - 1) & !(tls_align - 1);
    let has_tls = tls_addr != 0;

    let bss_addr = BASE_ADDR + offset;
    let mut bss_size = 0u64;
    for sec in output_sections.iter_mut() {
        if sec.sh_type == SHT_NOBITS && sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_TLS == 0 {
            let a = sec.alignment.max(1);
            let aligned = (bss_addr + bss_size + a - 1) & !(a - 1);
            bss_size = aligned - bss_addr + sec.mem_size;
            sec.addr = aligned; sec.file_offset = offset;
        }
    }

    // Allocate BSS space for copy-relocated symbols.
    // Symbols that are aliases (same from_lib + lib_sym_value) share the same BSS slot.
    let mut copy_reloc_addr_map: HashMap<(String, u64), u64> = HashMap::new(); // (lib, lib_value) -> bss_addr
    for (name, size) in &copy_reloc_syms {
        let gsym = globals.get(name).cloned();
        let key = gsym.as_ref().and_then(|g| {
            g.from_lib.as_ref().map(|lib| (lib.clone(), g.lib_sym_value))
        });
        let addr = if let Some(ref k) = key {
            if let Some(&existing_addr) = copy_reloc_addr_map.get(k) {
                existing_addr // reuse existing BSS slot for alias
            } else {
                let aligned = (bss_addr + bss_size + 7) & !7;
                bss_size = aligned - bss_addr + size;
                copy_reloc_addr_map.insert(k.clone(), aligned);
                aligned
            }
        } else {
            let aligned = (bss_addr + bss_size + 7) & !7;
            bss_size = aligned - bss_addr + size;
            aligned
        };
        if let Some(gsym) = globals.get_mut(name) {
            gsym.value = addr;
            gsym.defined_in = Some(usize::MAX); // sentinel: defined via copy reloc
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

    // Define linker-provided symbols using shared infrastructure (consistent
    // with i686/ARM/RISC-V backends via get_standard_linker_symbols)
    let text_seg_end = text_page_addr + text_total_size;
    let data_seg_start = rw_page_addr;
    let linker_addrs = LinkerSymbolAddresses {
        base_addr: BASE_ADDR,
        got_addr: got_plt_addr,
        dynamic_addr,
        bss_addr,
        bss_size,
        text_end: text_seg_end,
        data_start: data_seg_start,
        init_array_start: init_array_addr,
        init_array_size,
        fini_array_start: fini_array_addr,
        fini_array_size,
        preinit_array_start: 0,
        preinit_array_size: 0,
        rela_iplt_start: rela_iplt_addr,
        rela_iplt_size,
    };
    for sym in &get_standard_linker_symbols(&linker_addrs) {
        let entry = globals.entry(sym.name.to_string()).or_insert(GlobalSymbol {
            value: 0, size: 0, info: (sym.binding << 4),
            defined_in: None, from_lib: None, plt_idx: None, got_idx: None,
            section_idx: SHN_ABS, is_dynamic: false, copy_reloc: false, lib_sym_value: 0, version: None,
        });
        if entry.defined_in.is_none() && !entry.is_dynamic {
            entry.value = sym.value;
            entry.defined_in = Some(usize::MAX); // sentinel: linker-defined
            entry.section_idx = SHN_ABS;
        }
    }

    // Auto-generate __start_<section> / __stop_<section> symbols (GNU ld feature).
    // These are created for output sections whose names are valid C identifiers,
    // when there are undefined references to those symbols.
    for (name, addr) in linker_common::resolve_start_stop_symbols(output_sections) {
        if let Some(entry) = globals.get_mut(&name) {
            if entry.defined_in.is_none() && !entry.is_dynamic {
                entry.value = addr;
                entry.defined_in = Some(usize::MAX);
                entry.section_idx = SHN_ABS;
            }
        }
    }

    // Override IFUNC symbol addresses to point to IPLT entries.
    // Save the original (resolver) addresses for IFUNC GOT initialization.
    let mut ifunc_resolver_addrs: Vec<u64> = Vec::new();
    for (i, name) in ifunc_symbols.iter().enumerate() {
        if let Some(gsym) = globals.get_mut(name) {
            ifunc_resolver_addrs.push(gsym.value);
            gsym.value = iplt_addr + (i as u64) * iplt_entry_size;
        }
    }

    let entry_addr = globals.get("_start").map(|s| s.value).unwrap_or(text_page_addr);

    // === Build output buffer ===
    let file_size = offset as usize;
    let mut out = vec![0u8; file_size];

    // ELF header
    out[0..4].copy_from_slice(&ELF_MAGIC);
    out[4] = ELFCLASS64; out[5] = ELFDATA2LSB; out[6] = 1;
    w16(&mut out, 16, ET_EXEC); w16(&mut out, 18, EM_X86_64); w32(&mut out, 20, 1);
    w64(&mut out, 24, entry_addr); w64(&mut out, 32, 64); w64(&mut out, 40, 0);
    w32(&mut out, 48, 0); w16(&mut out, 52, 64); w16(&mut out, 54, 56);
    w16(&mut out, 56, phdr_count as u16); w16(&mut out, 58, 64); w16(&mut out, 60, 0); w16(&mut out, 62, 0);

    // Program headers
    let mut ph = 64usize;
    wphdr(&mut out, ph, PT_PHDR, PF_R, 64, BASE_ADDR+64, phdr_total_size, phdr_total_size, 8); ph += 56;
    if !is_static {
        wphdr(&mut out, ph, PT_INTERP, PF_R, interp_offset, interp_addr, INTERP.len() as u64, INTERP.len() as u64, 1); ph += 56;
    }
    let ro_seg_end = rela_plt_offset + rela_plt_size;
    wphdr(&mut out, ph, PT_LOAD, PF_R, 0, BASE_ADDR, ro_seg_end, ro_seg_end, PAGE_SIZE); ph += 56;
    wphdr(&mut out, ph, PT_LOAD, PF_R|PF_X, text_page_offset, text_page_addr, text_total_size, text_total_size, PAGE_SIZE); ph += 56;
    wphdr(&mut out, ph, PT_LOAD, PF_R, rodata_page_offset, rodata_page_addr, rodata_total_size, rodata_total_size, PAGE_SIZE); ph += 56;
    let rw_filesz = offset - rw_page_offset;
    let rw_memsz = if bss_size > 0 { (bss_addr + bss_size) - rw_page_addr } else { rw_filesz };
    wphdr(&mut out, ph, PT_LOAD, PF_R|PF_W, rw_page_offset, rw_page_addr, rw_filesz, rw_memsz, PAGE_SIZE); ph += 56;
    if !is_static {
        wphdr(&mut out, ph, PT_DYNAMIC, PF_R|PF_W, dynamic_offset, dynamic_addr, dynamic_size, dynamic_size, 8); ph += 56;
    }
    wphdr(&mut out, ph, PT_GNU_STACK, PF_R|PF_W, 0, 0, 0, 0, 0x10); ph += 56;
    if has_tls {
        wphdr(&mut out, ph, PT_TLS, PF_R, tls_file_offset, tls_addr, tls_file_size, tls_mem_size, tls_align);
    }

    // Section data (needed for both static and dynamic)
    for sec in output_sections.iter() {
        if sec.sh_type == SHT_NOBITS || sec.data.is_empty() { continue; }
        write_bytes(&mut out, sec.file_offset as usize, &sec.data);
    }

    // Dynamic linking sections (skipped for static executables)
    if !is_static {
        // .interp
        write_bytes(&mut out, interp_offset as usize, INTERP);

        // .gnu.hash - proper hash table so dynamic linker can find copy-reloc symbols
        let gh = gnu_hash_offset as usize;
        w32(&mut out, gh, gnu_hash_nbuckets);
        w32(&mut out, gh+4, gnu_hash_symoffset as u32);
        w32(&mut out, gh+8, gnu_hash_bloom_size);
        w32(&mut out, gh+12, gnu_hash_bloom_shift);
        // Bloom filter
        let bloom_off = gh + 16;
        w64(&mut out, bloom_off, bloom_word);
        // Buckets
        let buckets_off = bloom_off + (gnu_hash_bloom_size as usize * 8);
        for (i, &b) in gnu_hash_buckets.iter().enumerate() {
            w32(&mut out, buckets_off + i * 4, b);
        }
        // Chains
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
                if gsym.copy_reloc {
                    if ds+5 < out.len() { out[ds+4] = (STB_GLOBAL << 4) | STT_OBJECT; out[ds+5] = 0; }
                    w16(&mut out, ds+6, 1);
                    w64(&mut out, ds+8, gsym.value);
                    w64(&mut out, ds+16, gsym.size);
                } else if !gsym.is_dynamic && gsym.section_idx != SHN_UNDEF && gsym.value != 0 {
                    let stt = gsym.info & 0xf;
                    let stb = gsym.info >> 4;
                    let st_info = (stb << 4) | stt;
                    if ds+5 < out.len() { out[ds+4] = st_info; out[ds+5] = 0; }
                    w16(&mut out, ds+6, 1);
                    // For TLS symbols, the dynsym value must be the offset within
                    // the TLS segment, not the virtual address.
                    let sym_val = if stt == STT_TLS && tls_addr != 0 {
                        gsym.value - tls_addr
                    } else {
                        gsym.value
                    };
                    w64(&mut out, ds+8, sym_val);
                    w64(&mut out, ds+16, gsym.size);
                } else {
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

        // .gnu.version (versym)
        if !versym_data.is_empty() {
            write_bytes(&mut out, versym_offset as usize, &versym_data);
        }

        // .gnu.version_r (verneed)
        if !verneed_data.is_empty() {
            write_bytes(&mut out, verneed_offset as usize, &verneed_data);
        }

        // .rela.dyn (GLOB_DAT for dynamic GOT symbols, R_X86_64_COPY for copy relocs)
        let mut rd = rela_dyn_offset as usize;
        let mut gd_a = got_addr;
        for (name, is_plt) in got_entries {
            if name.is_empty() || *is_plt { continue; }
            let gsym_info = globals.get(name);
            let is_dynamic = gsym_info.map(|g| g.is_dynamic && !g.copy_reloc).unwrap_or(false);
            let has_plt = gsym_info.map(|g| g.plt_idx.is_some()).unwrap_or(false);
            // Skip GLOB_DAT for dynamic symbols that also have a PLT entry:
            // their GOT entry is statically filled with the PLT address to match
            // the canonical address used by R_X86_64_64 data relocations.
            if is_dynamic && !has_plt {
                let si = dyn_sym_names.iter().position(|n| n == name).map(|i| i+1).unwrap_or(0) as u64;
                w64(&mut out, rd, gd_a); w64(&mut out, rd+8, (si << 32) | R_X86_64_GLOB_DAT as u64); w64(&mut out, rd+16, 0);
                rd += 24;
            }
            gd_a += 8;
        }
        // R_X86_64_COPY relocations for copy-relocated symbols
        for (name, _) in &copy_reloc_syms {
            if let Some(gsym) = globals.get(name) {
                let si = dyn_sym_names.iter().position(|n| n == name).map(|i| i+1).unwrap_or(0) as u64;
                let copy_addr = gsym.value;
                w64(&mut out, rd, copy_addr); w64(&mut out, rd+8, (si << 32) | 5); w64(&mut out, rd+16, 0);
                rd += 24;
            }
        }

        // .rela.plt
        let mut rp = rela_plt_offset as usize;
        let gpb = got_plt_addr + 24;
        for (i, name) in plt_names.iter().enumerate() {
            let gea = gpb + i as u64 * 8;
            let si = dyn_sym_names.iter().position(|n| n == name).map(|j| j+1).unwrap_or(0) as u64;
            w64(&mut out, rp, gea); w64(&mut out, rp+8, (si << 32) | R_X86_64_JUMP_SLOT as u64); w64(&mut out, rp+16, 0);
            rp += 24;
        }

        // .plt
        if plt_size > 0 {
            let po = plt_offset as usize;
            out[po] = 0xff; out[po+1] = 0x35;
            w32(&mut out, po+2, ((got_plt_addr+8) as i64 - (plt_addr+6) as i64) as u32);
            out[po+6] = 0xff; out[po+7] = 0x25;
            w32(&mut out, po+8, ((got_plt_addr+16) as i64 - (plt_addr+12) as i64) as u32);
            for i in 12..16 { out[po+i] = 0x90; }

            for (i, _) in plt_names.iter().enumerate() {
                let ep = po + 16 + i * 16;
                let pea = plt_addr + 16 + i as u64 * 16;
                let gea = got_plt_addr + 24 + i as u64 * 8;
                out[ep] = 0xff; out[ep+1] = 0x25;
                w32(&mut out, ep+2, (gea as i64 - (pea+6) as i64) as u32);
                out[ep+6] = 0x68; w32(&mut out, ep+7, i as u32);
                out[ep+11] = 0xe9; w32(&mut out, ep+12, (plt_addr as i64 - (pea+16) as i64) as u32);
            }
        }

        // .dynamic
        let mut dd = dynamic_offset as usize;
        for lib in needed_sonames {
            let so = dynstr.get_offset(lib);
            w64(&mut out, dd, DT_NEEDED as u64); w64(&mut out, dd+8, so as u64); dd += 16;
        }
        for &(tag, val) in &[
            (DT_STRTAB, dynstr_addr), (DT_SYMTAB, dynsym_addr), (DT_STRSZ, dynstr_size),
            (DT_SYMENT, 24), (DT_DEBUG, 0), (DT_PLTGOT, got_plt_addr),
            (DT_PLTRELSZ, rela_plt_size), (DT_PLTREL, DT_RELA as u64), (DT_JMPREL, rela_plt_addr),
            (DT_RELA, rela_dyn_addr), (DT_RELASZ, rela_dyn_size), (DT_RELAENT, 24),
            (DT_GNU_HASH, gnu_hash_addr),
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
        if let Some(ref rp) = rpath_string {
            let rp_off = dynstr.get_offset(rp) as u64;
            let tag = if use_runpath { DT_RUNPATH } else { DT_RPATH };
            w64(&mut out, dd, tag as u64); w64(&mut out, dd+8, rp_off); dd += 16;
        }
        if verneed_size > 0 {
            w64(&mut out, dd, DT_VERSYM as u64); w64(&mut out, dd+8, versym_addr); dd += 16;
            w64(&mut out, dd, DT_VERNEED as u64); w64(&mut out, dd+8, verneed_addr); dd += 16;
            w64(&mut out, dd, DT_VERNEEDNUM as u64); w64(&mut out, dd+8, verneed_count as u64); dd += 16;
        }
        w64(&mut out, dd, DT_NULL as u64); w64(&mut out, dd+8, 0);

        // .got.plt
        let gp = got_plt_offset as usize;
        w64(&mut out, gp, dynamic_addr);
        w64(&mut out, gp+8, 0); w64(&mut out, gp+16, 0);
        for (i, _) in plt_names.iter().enumerate() {
            w64(&mut out, gp+24+i*8, plt_addr + 16 + i as u64 * 16 + 6);
        }
    } // end if !is_static

    // .got (needed for both static and dynamic: TLS GOTTPOFF, GOTPCREL entries)
    if got_size > 0 {
        let mut go = got_offset as usize;
        for (name, is_plt) in got_entries {
            if name.is_empty() || *is_plt { continue; }
            if let Some(gsym) = globals.get(name) {
                if gsym.defined_in.is_some() && !gsym.is_dynamic {
                    let sym_val = gsym.value;
                    if has_tls && (gsym.info & 0xf) == STT_TLS {
                        // TLS GOT entry: store the TPOFF value
                        let tpoff = (sym_val as i64 - tls_addr as i64) - tls_mem_size as i64;
                        w64(&mut out, go, tpoff as u64);
                    } else {
                        w64(&mut out, go, sym_val);
                    }
                } else if gsym.copy_reloc && gsym.value != 0 {
                    w64(&mut out, go, gsym.value);
                } else if gsym.is_dynamic {
                    if let Some(plt_idx) = gsym.plt_idx {
                        // Dynamic function with both PLT and GOTPCREL: fill GOT with
                        // PLT entry address so address-of via GOTPCREL matches the
                        // canonical PLT address used by R_X86_64_64 data relocations.
                        let plt_entry_addr = plt_addr + 16 + plt_idx as u64 * 16;
                        w64(&mut out, go, plt_entry_addr);
                    }
                }
            }
            go += 8;
        }
    }

    // IFUNC: write .iplt, IFUNC GOT, and .rela.iplt data
    if num_ifunc > 0 {
        // .iplt - each entry is: jmp *ifunc_got_entry(%rip); nop padding
        for i in 0..num_ifunc {
            let ep = iplt_offset as usize + i * iplt_entry_size as usize;
            let pea = iplt_addr + i as u64 * iplt_entry_size; // address of this IPLT entry
            let gea = ifunc_got_addr + i as u64 * 8; // address of IFUNC GOT entry
            // ff 25 XX XX XX XX = jmp *disp32(%rip)
            out[ep] = 0xff; out[ep+1] = 0x25;
            w32(&mut out, ep+2, (gea as i64 - (pea + 6) as i64) as u32);
            // Pad remaining 10 bytes with NOPs
            for j in 6..iplt_entry_size as usize { out[ep+j] = 0x90; }
        }

        // IFUNC GOT - initialized to resolver function addresses
        for (i, &resolver_addr) in ifunc_resolver_addrs.iter().enumerate() {
            let go = ifunc_got_offset as usize + i * 8;
            w64(&mut out, go, resolver_addr);
        }

        // .rela.iplt - R_X86_64_IRELATIVE relocations
        for i in 0..num_ifunc {
            let rp = rela_iplt_offset as usize + i * 24;
            let r_offset = ifunc_got_addr + i as u64 * 8;
            // r_info: (0 << 32) | R_X86_64_IRELATIVE
            w64(&mut out, rp, r_offset);
            w64(&mut out, rp+8, R_X86_64_IRELATIVE as u64);
            w64(&mut out, rp+16, ifunc_resolver_addrs[i]); // r_addend = resolver address
        }
    }

    // === Apply relocations ===
    // Snapshot globals to avoid borrow issues
    let globals_snap: HashMap<String, GlobalSymbol> = globals.clone();

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
                let s = resolve_sym(obj_idx, sym, &globals_snap, section_map, output_sections,
                                    plt_addr);

                match rela.rela_type {
                    R_X86_64_64 => {
                        let t = if !sym.name.is_empty() && !sym.is_local() {
                            if let Some(g) = globals_snap.get(&sym.name) {
                                if g.is_dynamic && !g.copy_reloc {
                                    if let Some(pi) = g.plt_idx { plt_addr + 16 + pi as u64 * 16 } else { s }
                                } else { s }
                            } else { s }
                        } else { s };
                        w64(&mut out, fp, (t as i64 + a) as u64);
                    }
                    R_X86_64_PC32 | R_X86_64_PLT32 => {
                        let t = if !sym.name.is_empty() && !sym.is_local() {
                            if let Some(g) = globals_snap.get(&sym.name) {
                                if let Some(pi) = g.plt_idx { plt_addr + 16 + pi as u64 * 16 } else { s }
                            } else { s }
                        } else { s };
                        w32(&mut out, fp, (t as i64 + a - p as i64) as u32);
                    }
                    R_X86_64_32 => { w32(&mut out, fp, (s as i64 + a) as u32); }
                    R_X86_64_32S => { w32(&mut out, fp, (s as i64 + a) as u32); }
                    R_X86_64_GOTTPOFF => {
                        // Initial Exec TLS via GOT: GOT entry contains TPOFF value
                        let mut resolved = false;
                        if !sym.name.is_empty() && !sym.is_local() {
                            if let Some(g) = globals_snap.get(&sym.name) {
                                if let Some(gi) = g.got_idx {
                                    let entry = &got_entries[gi];
                                    let gea = if entry.1 {
                                        got_plt_addr + 24 + g.plt_idx.unwrap_or(0) as u64 * 8
                                    } else {
                                        let nb = got_entries[..gi].iter().filter(|(n,p)| !n.is_empty() && !*p).count();
                                        got_addr + nb as u64 * 8
                                    };
                                    w32(&mut out, fp, (gea as i64 + a - p as i64) as u32);
                                    resolved = true;
                                }
                            }
                        }
                        if !resolved {
                            // IE-to-LE relaxation: convert GOT-indirect to immediate TPOFF.
                            // Transform: movq GOT(%rip), %reg -> movq $tpoff, %reg
                            // Encoding: 48 8b XX YY YY YY YY -> 48 c7 CX YY YY YY YY
                            //   where XX encodes the register via ModR/M
                            let tpoff = (s as i64 - tls_addr as i64) - tls_mem_size as i64;
                            if fp >= 2 && fp + 4 <= out.len() && out[fp-2] == 0x8b {
                                // Get the register from ModR/M byte
                                let modrm = out[fp-1];
                                let reg = (modrm >> 3) & 7;
                                // Change mov r/m64,reg to mov $imm32,reg (opcode 0xc7, /0)
                                out[fp-2] = 0xc7;
                                out[fp-1] = 0xc0 | reg;
                                w32(&mut out, fp, (tpoff + a) as u32);
                            } else {
                                return Err(format!(
                                    "GOTTPOFF IE-to-LE relaxation failed: unrecognized instruction pattern at offset 0x{:x} for symbol '{}' (expected movq GOT(%rip), %reg)",
                                    fp, sym.name
                                ));
                            }
                        }
                    }
                    R_X86_64_GOTPCREL | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => {
                        if !sym.name.is_empty() && !sym.is_local() {
                            if let Some(g) = globals_snap.get(&sym.name) {
                                if let Some(gi) = g.got_idx {
                                    let entry = &got_entries[gi];
                                    let gea = if entry.1 {
                                        got_plt_addr + 24 + g.plt_idx.unwrap_or(0) as u64 * 8
                                    } else {
                                        let nb = got_entries[..gi].iter().filter(|(n,p)| !n.is_empty() && !*p).count();
                                        got_addr + nb as u64 * 8
                                    };
                                    w32(&mut out, fp, (gea as i64 + a - p as i64) as u32);
                                    continue;
                                }
                                if (rela.rela_type == R_X86_64_GOTPCRELX || rela.rela_type == R_X86_64_REX_GOTPCRELX)
                                   && g.defined_in.is_some() {
                                    if fp >= 2 && fp < out.len() && out[fp-2] == 0x8b {
                                        out[fp-2] = 0x8d;
                                    }
                                    w32(&mut out, fp, (s as i64 + a - p as i64) as u32);
                                    continue;
                                }
                            }
                        }
                        w32(&mut out, fp, (s as i64 + a - p as i64) as u32);
                    }
                    R_X86_64_PC64 => { w64(&mut out, fp, (s as i64 + a - p as i64) as u64); }
                    R_X86_64_TPOFF32 => {
                        // Initial Exec TLS: value = (sym_addr - tls_addr) - tls_mem_size
                        // %fs:0 points past end of TLS block on x86-64
                        let tpoff = (s as i64 - tls_addr as i64) - tls_mem_size as i64;
                        w32(&mut out, fp, (tpoff + a) as u32);
                    }
                    R_X86_64_NONE => {}
                    other => {
                        return Err(format!(
                            "unsupported x86-64 relocation type {} for '{}' in {}",
                            other, sym.name, objects[obj_idx].source_name
                        ));
                    }
                }
            }
        }
    }

    // === Append section headers ===
    // Build .shstrtab string table
    let mut shstrtab = vec![0u8]; // null byte at offset 0
    let mut shstr_offsets: HashMap<String, u32> = HashMap::new();
    let known_names = [
        ".interp", ".gnu.hash", ".dynsym", ".dynstr",
        ".gnu.version", ".gnu.version_r",
        ".rela.dyn", ".rela.plt", ".plt", ".dynamic",
        ".got", ".got.plt", ".init_array", ".fini_array",
        ".tdata", ".tbss", ".bss", ".shstrtab",
        ".iplt", ".rela.iplt",
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

    // Use shared write_elf64_shdr from linker_common (aliased locally for brevity)
    let write_shdr = linker_common::write_elf64_shdr;

    // Pre-count section indices for cross-references (dynsym_shidx, dynstr_shidx)
    // These are only meaningful for dynamic linking, but define them unconditionally for convenience
    let dynsym_shidx: u32 = if is_static { 0 } else { 3 }; // NULL=0, .interp=1, .gnu.hash=2, .dynsym=3
    let dynstr_shidx: u32 = if is_static { 0 } else { 4 }; // .dynstr=4

    // Count total sections to determine .shstrtab index
    let mut sh_count: u16 = if is_static {
        1 // NULL only
    } else {
        5 // NULL + .interp + .gnu.hash + .dynsym + .dynstr
    };
    if !is_static && verneed_size > 0 { sh_count += 2; } // .gnu.version + .gnu.version_r
    if !is_static && rela_dyn_size > 0 { sh_count += 1; }
    if !is_static && rela_plt_size > 0 { sh_count += 1; }
    if !is_static && plt_size > 0 { sh_count += 1; }
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
    if !is_static { sh_count += 1; } // .dynamic
    if got_size > 0 { sh_count += 1; } // .got (needed for static too: TLS, GOTPCREL)
    if !is_static { sh_count += 1; } // .got.plt
    if iplt_total_size > 0 { sh_count += 1; } // .iplt
    if rela_iplt_size > 0 { sh_count += 1; } // .rela.iplt
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
    write_shdr(&mut out, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // Dynamic linking section headers (skipped for static executables)
    if !is_static {
        // .interp
        write_shdr(&mut out, get_shname(".interp"), SHT_PROGBITS, SHF_ALLOC,
                   interp_addr, interp_offset, INTERP.len() as u64, 0, 0, 1, 0);
        // .gnu.hash
        write_shdr(&mut out, get_shname(".gnu.hash"), SHT_GNU_HASH, SHF_ALLOC,
                   gnu_hash_addr, gnu_hash_offset, gnu_hash_size, dynsym_shidx, 0, 8, 0);
        // .dynsym
        write_shdr(&mut out, get_shname(".dynsym"), SHT_DYNSYM, SHF_ALLOC,
                   dynsym_addr, dynsym_offset, dynsym_size, dynstr_shidx, 1, 8, 24);
        // .dynstr
        write_shdr(&mut out, get_shname(".dynstr"), SHT_STRTAB, SHF_ALLOC,
                   dynstr_addr, dynstr_offset, dynstr_size, 0, 0, 1, 0);
        // .gnu.version (versym)
        if verneed_size > 0 {
            write_shdr(&mut out, get_shname(".gnu.version"), SHT_GNU_VERSYM, SHF_ALLOC,
                       versym_addr, versym_offset, versym_size, dynsym_shidx, 0, 2, 2);
        }
        // .gnu.version_r (verneed)
        if verneed_size > 0 {
            write_shdr(&mut out, get_shname(".gnu.version_r"), SHT_GNU_VERNEED, SHF_ALLOC,
                       verneed_addr, verneed_offset, verneed_size, dynstr_shidx, verneed_count, 4, 0);
        }
        // .rela.dyn
        if rela_dyn_size > 0 {
            write_shdr(&mut out, get_shname(".rela.dyn"), SHT_RELA, SHF_ALLOC,
                       rela_dyn_addr, rela_dyn_offset, rela_dyn_size, dynsym_shidx, 0, 8, 24);
        }
        // .rela.plt
        if rela_plt_size > 0 {
            write_shdr(&mut out, get_shname(".rela.plt"), SHT_RELA, SHF_ALLOC | 0x40,
                       rela_plt_addr, rela_plt_offset, rela_plt_size, dynsym_shidx, 0, 8, 24);
        }
        // .plt
        if plt_size > 0 {
            write_shdr(&mut out, get_shname(".plt"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
                       plt_addr, plt_offset, plt_size, 0, 0, 16, 16);
        }
    }
    // Merged output sections (text/rodata/data, excluding BSS/TLS/init_array/fini_array)
    for sec in output_sections.iter() {
        if sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS && sec.flags & SHF_TLS == 0
           && sec.name != ".init_array" && sec.name != ".fini_array" {
            write_shdr(&mut out, get_shname(&sec.name), sec.sh_type, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // TLS data sections (.tdata)
    for sec in output_sections.iter() {
        if sec.flags & SHF_TLS != 0 && sec.flags & SHF_ALLOC != 0 && sec.sh_type != SHT_NOBITS {
            write_shdr(&mut out, get_shname(&sec.name), sec.sh_type, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // TLS BSS sections (.tbss)
    for sec in output_sections.iter() {
        if sec.flags & SHF_TLS != 0 && sec.sh_type == SHT_NOBITS {
            write_shdr(&mut out, get_shname(&sec.name), SHT_NOBITS, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // .init_array
    if has_init_array {
        if let Some(ia_sec) = output_sections.iter().find(|s| s.name == ".init_array") {
            write_shdr(&mut out, get_shname(".init_array"), SHT_INIT_ARRAY, SHF_ALLOC | SHF_WRITE,
                       init_array_addr, ia_sec.file_offset, init_array_size, 0, 0, 8, 8);
        }
    }
    // .fini_array
    if has_fini_array {
        if let Some(fa_sec) = output_sections.iter().find(|s| s.name == ".fini_array") {
            write_shdr(&mut out, get_shname(".fini_array"), SHT_FINI_ARRAY, SHF_ALLOC | SHF_WRITE,
                       fini_array_addr, fa_sec.file_offset, fini_array_size, 0, 0, 8, 8);
        }
    }
    if !is_static {
        // .dynamic
        write_shdr(&mut out, get_shname(".dynamic"), SHT_DYNAMIC, SHF_ALLOC | SHF_WRITE,
                   dynamic_addr, dynamic_offset, dynamic_size, dynstr_shidx, 0, 8, 16);
    }
    // .got (needed for both static and dynamic: TLS GOTTPOFF, GOTPCREL)
    if got_size > 0 {
        write_shdr(&mut out, get_shname(".got"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                   got_addr, got_offset, got_size, 0, 0, 8, 8);
    }
    if !is_static {
        // .got.plt
        write_shdr(&mut out, get_shname(".got.plt"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                   got_plt_addr, got_plt_offset, got_plt_size, 0, 0, 8, 8);
    }
    // .iplt (IFUNC PLT for static linking)
    if iplt_total_size > 0 {
        write_shdr(&mut out, get_shname(".iplt"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
                   iplt_addr, iplt_offset, iplt_total_size, 0, 0, 16, 16);
    }
    // .rela.iplt (IRELATIVE relocations for static linking)
    if rela_iplt_size > 0 {
        write_shdr(&mut out, get_shname(".rela.iplt"), SHT_RELA, SHF_ALLOC,
                   rela_iplt_addr, rela_iplt_offset, rela_iplt_size, 0, 0, 8, 24);
    }
    // BSS sections (non-TLS)
    for sec in output_sections.iter() {
        if sec.sh_type == SHT_NOBITS && sec.flags & SHF_ALLOC != 0 && sec.flags & SHF_TLS == 0 {
            write_shdr(&mut out, get_shname(&sec.name), SHT_NOBITS, sec.flags,
                       sec.addr, sec.file_offset, sec.mem_size, 0, 0, sec.alignment.max(1), 0);
        }
    }
    // .shstrtab (last section)
    write_shdr(&mut out, get_shname(".shstrtab"), SHT_STRTAB, 0,
               0, shstrtab_data_offset, shstrtab.len() as u64, 0, 0, 1, 0);

    // Patch ELF header with section header info
    // e_shoff at offset 40 (8 bytes)
    out[40..48].copy_from_slice(&shdr_offset.to_le_bytes());
    // e_shnum at offset 60 (2 bytes)
    out[60..62].copy_from_slice(&sh_count.to_le_bytes());
    // e_shstrndx at offset 62 (2 bytes)
    out[62..64].copy_from_slice(&shstrtab_shidx.to_le_bytes());

    std::fs::write(output_path, &out).map_err(|e| format!("failed to write '{}': {}", output_path, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(output_path, std::fs::Permissions::from_mode(0o755));
    }
    Ok(())
}

pub(super) fn resolve_sym(
    obj_idx: usize, sym: &Symbol, globals: &HashMap<String, GlobalSymbol>,
    section_map: &HashMap<(usize, usize), (usize, u64)>, output_sections: &[OutputSection],
    plt_addr: u64,
) -> u64 {
    if sym.sym_type() == STT_SECTION {
        let si = sym.shndx as usize;
        return section_map.get(&(obj_idx, si)).map(|&(oi, so)| output_sections[oi].addr + so).unwrap_or(0);
    }
    // Local (STB_LOCAL) symbols must NOT be resolved via globals, since a
    // local symbol named e.g. "opts" must not be confused with a global "opts"
    // from another object file.
    if !sym.name.is_empty() && !sym.is_local() {
        if let Some(g) = globals.get(&sym.name) {
            if g.defined_in.is_some() { return g.value; }
            if g.is_dynamic {
                return g.plt_idx.map(|pi| plt_addr + 16 + pi as u64 * 16).unwrap_or(0);
            }
        }
        if sym.is_weak() { return 0; }
    }
    if sym.is_undefined() { return 0; }
    if sym.shndx == SHN_ABS { return sym.value; }
    section_map.get(&(obj_idx, sym.shndx as usize))
        .map(|&(oi, so)| output_sections[oi].addr + so + sym.value).unwrap_or(sym.value)
}
