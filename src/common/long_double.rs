//! Long double precision support.
//!
//! On x86-64, `long double` is 80-bit x87 extended precision (stored in 16 bytes with 6 padding bytes).
//! On AArch64/RISC-V, `long double` is IEEE 754 binary128 (quad precision, 16 bytes).
//!
//! This module provides:
//! - Parsing decimal/hex strings to x87 80-bit or f128 128-bit format
//! - Correctly rounded conversion of hex float constants to f32, f64 and long double
//! - Conversion between x87, f128, and f64 formats
//! - Full-precision arithmetic on x87 and f128 (pure software, no inline asm)
//! - Conversion from floating-point bytes to integer types
//!
//! # Architecture
//!
//! Both x87 and f128 formats use 16383-biased exponents (15-bit exponent field).
//! The key structural difference: x87 has an explicit integer bit in the mantissa
//! (64 bits total), while f128 uses an implicit leading 1 (112-bit stored mantissa).
//!
//! Shared decompose helpers (`x87_decompose`, `f64_decompose`, `f128_decompose`)
//! extract (sign, exponent, mantissa) tuples, eliminating boilerplate repetition
//! across the many conversion and arithmetic functions.

/// Result of preprocessing a long double string literal.
enum PreparsedFloat<'a> {
    /// Hex float (sign, text including "0x" prefix)
    Hex(bool, &'a str),
    /// Infinity
    Infinity(bool),
    /// NaN
    NaN(bool),
    /// Decimal float (sign, text without suffix/sign)
    Decimal(bool, &'a str),
}

/// Preprocess a long double string: strip suffix, detect sign, classify format.
/// Shared by both x87 and f128 parsing entry points.
fn preparse_long_double(text: &str) -> PreparsedFloat<'_> {
    let text = text.trim();
    let text = if text.ends_with('L') || text.ends_with('l') {
        &text[..text.len() - 1]
    } else {
        text
    };

    let (negative, text) = if let Some(rest) = text.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = text.strip_prefix('+') {
        (false, rest)
    } else {
        (false, text)
    };

    if text.len() > 2 && (text.starts_with("0x") || text.starts_with("0X")) {
        return PreparsedFloat::Hex(negative, text);
    }

    let text_lower = text.to_ascii_lowercase();
    if text_lower == "inf" || text_lower == "infinity" {
        return PreparsedFloat::Infinity(negative);
    }
    if text_lower == "nan" || text_lower.starts_with("nan(") {
        return PreparsedFloat::NaN(negative);
    }

    PreparsedFloat::Decimal(negative, text)
}

// Simple big integer using Vec<u32> limbs (little-endian: limbs[0] is least significant)
struct BigUint {
    limbs: Vec<u32>,
}

impl BigUint {
    fn from_decimal_digits(digits: &[u8]) -> Self {
        // Convert decimal digits to binary limbs
        let mut limbs = vec![0u32];
        for &d in digits {
            // Multiply by 10 and add digit
            let mut carry: u64 = d as u64;
            for limb in limbs.iter_mut() {
                let val = (*limb as u64) * 10 + carry;
                *limb = val as u32;
                carry = val >> 32;
            }
            if carry > 0 {
                limbs.push(carry as u32);
            }
        }
        BigUint { limbs }
    }

    fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    fn mul_u32(&mut self, factor: u32) {
        let mut carry: u64 = 0;
        for limb in self.limbs.iter_mut() {
            let val = (*limb as u64) * (factor as u64) + carry;
            *limb = val as u32;
            carry = val >> 32;
        }
        if carry > 0 {
            self.limbs.push(carry as u32);
        }
    }

    /// Shift left by n bits.
    fn shl(&mut self, n: u32) {
        if n == 0 || self.is_zero() {
            return;
        }
        let word_shift = (n / 32) as usize;
        let bit_shift = n % 32;

        if word_shift > 0 {
            let old_len = self.limbs.len();
            self.limbs.resize(old_len + word_shift, 0);
            // Shift words
            for i in (0..old_len).rev() {
                self.limbs[i + word_shift] = self.limbs[i];
            }
            for i in 0..word_shift {
                self.limbs[i] = 0;
            }
        }

        if bit_shift > 0 {
            let mut carry: u32 = 0;
            for limb in self.limbs.iter_mut() {
                let new_carry = *limb >> (32 - bit_shift);
                *limb = (*limb << bit_shift) | carry;
                carry = new_carry;
            }
            if carry > 0 {
                self.limbs.push(carry);
            }
        }
    }

    /// Get the number of significant bits.
    fn bit_length(&self) -> u32 {
        if self.is_zero() {
            return 0;
        }
        let top_limb = *self.limbs.last().unwrap();
        let top_bits = 32 - top_limb.leading_zeros();
        (self.limbs.len() as u32 - 1) * 32 + top_bits
    }

    /// Check if bit at position `pos` (0-indexed from LSB) is set.
    fn bit_at(&self, pos: u32) -> bool {
        let word_idx = (pos / 32) as usize;
        let bit_idx = pos % 32;
        if word_idx < self.limbs.len() {
            (self.limbs[word_idx] >> bit_idx) & 1 != 0
        } else {
            false
        }
    }

    /// Check if any bit in the range [0, pos) is set (i.e., any of the bottom `pos` bits).
    fn any_bits_below(&self, pos: u32) -> bool {
        let full_words = (pos / 32) as usize;
        let remainder = pos % 32;
        for i in 0..full_words {
            if i < self.limbs.len() && self.limbs[i] != 0 {
                return true;
            }
        }
        if remainder > 0 && full_words < self.limbs.len() {
            let mask = (1u32 << remainder) - 1;
            if self.limbs[full_words] & mask != 0 {
                return true;
            }
        }
        false
    }

    /// Extract the top N bits (up to 128), with the MSB at bit (N-1).
    /// Returns (top_val, bits_shifted) where the value is approximately top_val * 2^bits_shifted.
    fn top_n_bits(&self, n: u32) -> (u128, i32) {
        assert!(n > 0 && n <= 128);
        let bl = self.bit_length();
        if bl == 0 {
            return (0, 0);
        }
        if bl <= n {
            // Value fits in n bits
            let mut val: u128 = 0;
            for (i, &limb) in self.limbs.iter().enumerate() {
                val |= (limb as u128) << (i * 32);
            }
            return (val, 0);
        }

        // We need bits [bl-1 .. bl-n] of the big integer
        let shift = bl - n;
        let word_shift = (shift / 32) as usize;
        let bit_shift = shift % 32;

        let mut val: u128 = 0;
        // We need up to (n/32 + 2) limbs due to bit_shift straddling
        let limbs_needed = (n / 32 + 2) as usize;
        for j in 0..limbs_needed {
            let idx = word_shift + j;
            if idx < self.limbs.len() {
                let limb = self.limbs[idx] as u128;
                if j == 0 {
                    val |= limb >> bit_shift;
                } else {
                    let bit_pos = j as u32 * 32 - bit_shift;
                    if bit_pos < 128 {
                        val |= limb << bit_pos;
                    }
                }
            }
        }
        // Mask to n bits
        if n < 128 {
            val &= (1u128 << n) - 1;
        }

        (val, shift as i32)
    }

    /// Divide dividend by divisor, returning the quotient (floor division).
    /// The remainder is discarded. Used for dividing by large powers of 10.
    fn div_big(dividend: &BigUint, divisor: &BigUint) -> BigUint {
        if divisor.is_zero() {
            return BigUint { limbs: vec![0] };
        }

        let d_bits = dividend.bit_length();
        let v_bits = divisor.bit_length();

        if d_bits < v_bits {
            return BigUint { limbs: vec![0] };
        }

        // Simple long division: shift divisor up, subtract repeatedly
        // For our purposes, we only need ~70 bits of quotient precision
        let mut quotient_limbs = vec![0u32; ((d_bits - v_bits) / 32 + 2) as usize];
        let mut remainder = dividend.limbs.clone();

        // Process from MSB down
        let shift_max = d_bits - v_bits;
        for shift in (0..=shift_max).rev() {
            // Check if (divisor << shift) <= remainder
            if cmp_shifted(&remainder, &divisor.limbs, shift) {
                // Subtract divisor << shift from remainder
                sub_shifted(&mut remainder, &divisor.limbs, shift);
                // Set bit in quotient
                let word = (shift / 32) as usize;
                let bit = shift % 32;
                if word < quotient_limbs.len() {
                    quotient_limbs[word] |= 1u32 << bit;
                }
            }
        }

        // Strip leading zeros
        while quotient_limbs.len() > 1 && *quotient_limbs.last().unwrap() == 0 {
            quotient_limbs.pop();
        }

        BigUint { limbs: quotient_limbs }
    }
}

/// Compare remainder >= divisor_limbs << shift
fn cmp_shifted(remainder: &[u32], divisor: &[u32], shift: u32) -> bool {
    let word_shift = (shift / 32) as usize;
    let bit_shift = shift % 32;

    // Find the top word of the shifted divisor
    let div_top = divisor.len() + word_shift + if bit_shift > 0 { 1 } else { 0 };

    if remainder.len() > div_top {
        // Remainder has more limbs, check if upper limbs are non-zero
        for i in div_top..remainder.len() {
            if remainder[i] != 0 {
                return true;
            }
        }
    }

    // Compare from MSB
    for i in (0..div_top.max(remainder.len())).rev() {
        let r = if i < remainder.len() { remainder[i] } else { 0 };
        // Get the shifted divisor bit at position i
        let d = shifted_limb(divisor, i, word_shift, bit_shift);
        if r > d {
            return true;
        }
        if r < d {
            return false;
        }
    }
    true // equal
}

/// Get limb `i` of (divisor << shift)
fn shifted_limb(divisor: &[u32], i: usize, word_shift: usize, bit_shift: u32) -> u32 {
    if i < word_shift {
        return 0;
    }
    let di = i - word_shift;
    if bit_shift == 0 {
        if di < divisor.len() { divisor[di] } else { 0 }
    } else {
        let lo = if di < divisor.len() { divisor[di] } else { 0 };
        let hi = if di > 0 && di - 1 < divisor.len() { divisor[di - 1] } else { 0 };
        (lo << bit_shift) | (hi >> (32 - bit_shift))
    }
}

/// Subtract divisor << shift from remainder (in place).
fn sub_shifted(remainder: &mut [u32], divisor: &[u32], shift: u32) {
    let word_shift = (shift / 32) as usize;
    let bit_shift = shift % 32;

    let mut borrow: i64 = 0;
    for i in word_shift..remainder.len() {
        let d = shifted_limb(divisor, i, word_shift, bit_shift) as i64;
        let val = remainder[i] as i64 - d - borrow;
        if val < 0 {
            remainder[i] = (val + (1i64 << 32)) as u32;
            borrow = 1;
        } else {
            remainder[i] = val as u32;
            borrow = 0;
        }
    }
}

/// Build a big integer for 10^n
fn pow10_big(n: u32) -> BigUint {
    let mut result = BigUint { limbs: vec![1] };
    // Multiply by 10 in chunks for efficiency
    let mut remaining = n;
    // Use 10^9 = 1_000_000_000 as a chunk (fits in u32)
    while remaining >= 9 {
        result.mul_u32(1_000_000_000);
        remaining -= 9;
    }
    // Remaining powers
    let small_pow10: [u32; 10] = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000];
    if remaining > 0 {
        result.mul_u32(small_pow10[remaining as usize]);
    }
    result
}

/// Parsed decimal float: digits and exponent extracted from a decimal float string.
/// Used by the shared parsing pipeline for both x87 and f128 formats.
struct ParsedDecimal {
    digits: Vec<u8>,
    decimal_exp: i32,
}

/// Parse a decimal float string into digits and decimal exponent.
/// Returns None if the string represents zero.
/// This is the shared front-end for both x87 and f128 decimal parsing.
fn parse_decimal_digits(text: &str, capacity: usize) -> Option<ParsedDecimal> {
    let bytes = text.as_bytes();

    let mut digits: Vec<u8> = Vec::with_capacity(capacity);
    let mut decimal_point_offset: Option<usize> = None;
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'.' {
            decimal_point_offset = Some(digits.len());
            i += 1;
        } else if bytes[i].is_ascii_digit() {
            digits.push(bytes[i] - b'0');
            i += 1;
        } else {
            break;
        }
    }

    // Parse optional exponent
    let mut exp10: i32 = 0;
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        let exp_neg = if i < bytes.len() && bytes[i] == b'-' {
            i += 1;
            true
        } else {
            if i < bytes.len() && bytes[i] == b'+' {
                i += 1;
            }
            false
        };
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            exp10 = exp10.saturating_mul(10).saturating_add((bytes[i] - b'0') as i32);
            i += 1;
        }
        if exp_neg {
            exp10 = -exp10;
        }
    }

    let frac_digits = if let Some(dp) = decimal_point_offset {
        (digits.len() - dp) as i32
    } else {
        0
    };
    let decimal_exp = exp10 - frac_digits;

    // Strip leading zeros
    while digits.len() > 1 && digits[0] == 0 {
        digits.remove(0);
    }

    if digits.is_empty() || (digits.len() == 1 && digits[0] == 0) {
        return None;
    }

    Some(ParsedDecimal { digits, decimal_exp })
}

/// Apply IEEE 754 round-to-nearest-even to a BigUint value, extracting a 113-bit mantissa.
/// Returns (mantissa113, binary_exp) after rounding, or None if zero.
fn round_to_nearest_even_113(big_val: &BigUint) -> Option<(u128, i32)> {
    let (top113, shift) = big_val.top_n_bits(113);
    if top113 == 0 {
        return None;
    }

    let lz = top113.leading_zeros() - (128 - 113);
    let mut mantissa113 = top113 << lz;
    let effective_shift = shift - lz as i32;
    let binary_exp = shift + 112 - lz as i32;

    // IEEE 754 round-to-nearest-even:
    // guard bit = bit at position (effective_shift - 1)
    // sticky bits = any bits below the guard bit
    if effective_shift > 0 {
        let guard_pos = (effective_shift - 1) as u32;
        let guard = big_val.bit_at(guard_pos);
        let sticky = if guard_pos > 0 { big_val.any_bits_below(guard_pos) } else { false };

        if guard && (sticky || (mantissa113 & 1 != 0)) {
            // Round up
            mantissa113 = mantissa113.wrapping_add(1);
            // If mantissa overflowed past 113 bits, we need to adjust
            if mantissa113 >> 113 != 0 {
                // This means we went from 0x1FFF...FFF to 0x2000...000 (114 bits)
                // Shift right and increment exponent
                return Some((mantissa113 >> 1, binary_exp + 1));
            }
        }
    }

    Some((mantissa113, binary_exp))
}

/// Shared decimal-to-float bigint conversion for f128 (113-bit mantissa).
fn decimal_to_float_bigint_f128(negative: bool, digits: &[u8], decimal_exp: i32) -> [u8; 16] {
    if decimal_exp >= 0 {
        let mut big_val = BigUint::from_decimal_digits(digits);
        let p10 = pow10_big(decimal_exp as u32);
        big_val = mul_big(&big_val, &p10);

        if big_val.is_zero() {
            return make_f128_zero(negative);
        }

        match round_to_nearest_even_113(&big_val) {
            Some((mantissa113, binary_exp)) => encode_f128(negative, binary_exp, mantissa113),
            None => make_f128_zero(negative),
        }
    } else {
        let neg_exp = (-decimal_exp) as u32;
        let big_d = BigUint::from_decimal_digits(digits);
        if big_d.is_zero() {
            return make_f128_zero(negative);
        }

        let extra_bits = (128 + neg_exp * 4).min(200000);
        let mut shifted_d = big_d;
        shifted_d.shl(extra_bits);

        let p10 = pow10_big(neg_exp);
        let quotient = BigUint::div_big(&shifted_d, &p10);
        if quotient.is_zero() {
            return make_f128_zero(negative);
        }

        match round_to_nearest_even_113(&quotient) {
            Some((mantissa113, binary_exp)) => {
                encode_f128(negative, binary_exp - extra_bits as i32, mantissa113)
            }
            None => make_f128_zero(negative),
        }
    }
}

/// Multiply two big integers.
fn mul_big(a: &BigUint, b: &BigUint) -> BigUint {
    let mut result = vec![0u32; a.limbs.len() + b.limbs.len()];
    for (i, &al) in a.limbs.iter().enumerate() {
        let mut carry: u64 = 0;
        for (j, &bl) in b.limbs.iter().enumerate() {
            let val = result[i + j] as u64 + (al as u64) * (bl as u64) + carry;
            result[i + j] = val as u32;
            carry = val >> 32;
        }
        if carry > 0 {
            result[i + b.limbs.len()] += carry as u32;
        }
    }
    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }
    BigUint { limbs: result }
}

fn make_x87_zero(negative: bool) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    if negative {
        bytes[9] = 0x80;
    }
    bytes
}

fn make_x87_infinity(negative: bool) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[7] = 0x80; // integer bit set, fraction = 0
    bytes[8] = 0xFF;
    bytes[9] = 0x7F | (if negative { 0x80 } else { 0 });
    bytes
}

fn make_x87_nan(negative: bool) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    // quiet NaN: integer bit set, top fraction bit set
    bytes[7] = 0xC0;
    bytes[8] = 0xFF;
    bytes[9] = 0x7F | (if negative { 0x80 } else { 0 });
    bytes
}

// =============================================================================
// Decompose helpers: extract (sign, exponent, mantissa) from raw bytes
// =============================================================================

/// Decomposed x87 80-bit extended precision value.
/// For normal numbers: `mantissa` has bit 63 set (the explicit integer bit).
/// Biased exponent uses the standard 16383 bias.
struct X87Decomposed {
    sign: bool,
    /// Biased exponent (0 = zero/subnormal, 0x7FFF = inf/NaN).
    biased_exp: u16,
    /// Full 64-bit mantissa including explicit integer bit.
    mantissa: u64,
}

impl X87Decomposed {
    fn is_zero(&self) -> bool { self.biased_exp == 0 && self.mantissa == 0 }
    fn is_special(&self) -> bool { self.biased_exp == 0x7FFF }
    fn is_inf(&self) -> bool { self.is_special() && self.mantissa == 0x8000_0000_0000_0000 }
    fn unbiased_exp(&self) -> i32 { self.biased_exp as i32 - 16383 }
}

/// Decompose x87 80-bit extended precision bytes into sign, exponent, and mantissa.
/// This is the single extraction point used by all x87 conversion functions.
fn x87_decompose(bytes: &[u8; 16]) -> X87Decomposed {
    let mantissa = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let exp_sign = u16::from_le_bytes([bytes[8], bytes[9]]);
    X87Decomposed {
        sign: (exp_sign >> 15) & 1 == 1,
        biased_exp: exp_sign & 0x7FFF,
        mantissa,
    }
}

/// Decomposed f64 value.
struct F64Decomposed {
    sign: bool,
    /// Biased exponent (0 = zero/subnormal, 0x7FF = inf/NaN).
    biased_exp: u16,
    /// 52-bit stored mantissa (no implicit leading 1).
    mantissa: u64,
}

impl F64Decomposed {
    fn is_zero(&self) -> bool { self.biased_exp == 0 && self.mantissa == 0 }
    fn is_special(&self) -> bool { self.biased_exp == 0x7FF }
    fn is_inf(&self) -> bool { self.is_special() && self.mantissa == 0 }
}

/// Decompose an f64 into sign, exponent, and mantissa.
fn f64_decompose(val: f64) -> F64Decomposed {
    let bits = val.to_bits();
    F64Decomposed {
        sign: (bits >> 63) & 1 == 1,
        biased_exp: ((bits >> 52) & 0x7FF) as u16,
        mantissa: bits & 0x000F_FFFF_FFFF_FFFF,
    }
}

/// Encode x87 80-bit extended precision bytes from sign, biased exponent, and mantissa.
fn x87_encode(sign: bool, biased_exp: u16, mantissa64: u64) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&mantissa64.to_le_bytes());
    let exp_sign = biased_exp | (if sign { 0x8000 } else { 0 });
    bytes[8] = (exp_sign & 0xFF) as u8;
    bytes[9] = (exp_sign >> 8) as u8;
    bytes
}

/// Convert x87 80-bit bytes back to f64 (lossy - for computations that need f64).
/// `bytes[0..10]` contain the x87 extended value in little-endian.
pub fn x87_bytes_to_f64(bytes: &[u8; 16]) -> f64 {
    let d = x87_decompose(bytes);
    let sign_u64 = if d.sign { 1u64 } else { 0u64 };

    if d.is_zero() {
        return if d.sign { -0.0 } else { 0.0 };
    }
    if d.is_special() {
        if d.is_inf() {
            return if d.sign { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        return f64::NAN;
    }

    // Normal number
    // x87: value = mantissa64 * 2^(exp15 - 16383 - 63)
    // f64: value = (1 + mantissa52/2^52) * 2^(exp11 - 1023)
    let unbiased = d.unbiased_exp();

    // f64 exponent range: -1022 to 1023
    if unbiased > 1023 {
        return if d.sign { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    if unbiased < -1074 {
        return if d.sign { -0.0 } else { 0.0 };
    }

    // Extract top 52 mantissa bits (below the integer bit)
    // mantissa64 bit 63 = integer bit (=1 for normals)
    // We want bits 62..11 (52 bits) for the f64 mantissa
    let mantissa52 = (d.mantissa >> 11) & 0x000F_FFFF_FFFF_FFFF;

    // Round to nearest: check bit 10 (the first dropped bit)
    let round_bit = (d.mantissa >> 10) & 1;
    let sticky = d.mantissa & 0x3FF; // bits 9..0
    let mantissa52 = if round_bit == 1 && (sticky != 0 || mantissa52 & 1 != 0) {
        mantissa52 + 1
    } else {
        mantissa52
    };

    // Handle mantissa overflow from rounding
    if mantissa52 > 0x000F_FFFF_FFFF_FFFF {
        let f64_biased_exp = (unbiased + 1024) as u64;
        if f64_biased_exp >= 0x7FF {
            return if d.sign { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        let f64_bits = (sign_u64 << 63) | (f64_biased_exp << 52);
        return f64::from_bits(f64_bits);
    }

    if unbiased >= -1022 {
        let f64_biased_exp = (unbiased + 1023) as u64;
        let f64_bits = (sign_u64 << 63) | (f64_biased_exp << 52) | mantissa52;
        f64::from_bits(f64_bits)
    } else {
        // Subnormal in f64 - not common for constants, just convert approximately
        let val = d.mantissa as f64 * 2.0_f64.powi(unbiased - 63);
        if d.sign { -val } else { val }
    }
}

/// Convert x87 80-bit bytes `[u8; 16]` to IEEE 754 binary128 bytes (16 bytes, little-endian).
/// Used for ARM64/RISC-V long double emission.
pub fn x87_bytes_to_f128_bytes(x87: &[u8; 16]) -> [u8; 16] {
    let d = x87_decompose(x87);

    if d.is_zero() {
        return make_f128_zero(d.sign);
    }
    if d.is_special() {
        return if d.is_inf() { make_f128_infinity(d.sign) } else { make_f128_nan(d.sign) };
    }

    // Normal number
    // x87 and f128 both use exponent bias 16383, so exponent bits are the same!
    // x87 mantissa: 64 bits with explicit integer bit at position 63
    // f128 mantissa: 112 bits, implicit leading 1 (no integer bit stored)
    // Take lower 63 bits of x87 mantissa and shift left by (112-63) = 49
    let mantissa_no_int = d.mantissa & 0x7FFF_FFFF_FFFF_FFFF;
    let mantissa112: u128 = (mantissa_no_int as u128) << 49;
    let sign_bit: u128 = if d.sign { 1u128 << 127 } else { 0 };
    let val: u128 = mantissa112 | ((d.biased_exp as u128) << 112) | sign_bit;
    val.to_le_bytes()
}

// =============================================================================
// IEEE 754 binary128 (f128) native functions
// =============================================================================

/// Parse a float string directly to IEEE 754 binary128 (f128) bytes with full 112-bit
/// mantissa precision. Used for long double constants on ARM64/RISC-V where long double
/// is quad precision.
///
/// f128 has 112 bits of mantissa (vs 64 for x87), so parsing directly to f128
/// preserves more precision than parsing to x87 and converting.
pub fn parse_long_double_to_f128_bytes(text: &str) -> [u8; 16] {
    match preparse_long_double(text) {
        PreparsedFloat::Hex(neg, text) => parse_hex_float_to_f128(neg, text),
        PreparsedFloat::Infinity(neg) => make_f128_infinity(neg),
        PreparsedFloat::NaN(neg) => make_f128_nan(neg),
        PreparsedFloat::Decimal(neg, text) => parse_decimal_float_to_f128(neg, text),
    }
}

/// Parse a decimal float string to IEEE 754 binary128 format.
fn parse_decimal_float_to_f128(negative: bool, text: &str) -> [u8; 16] {
    match parse_decimal_digits(text, 40) {
        Some(parsed) => decimal_to_float_bigint_f128(negative, &parsed.digits, parsed.decimal_exp),
        None => make_f128_zero(negative),
    }
}


/// Encode an IEEE 754 binary128 value from sign, binary exponent, and 113-bit mantissa.
/// `binary_exp` is the exponent of the MSB (bit 112) of `mantissa113`.
/// That is, value = mantissa113 * 2^(binary_exp - 112).
fn encode_f128(negative: bool, binary_exp: i32, mantissa113: u128) -> [u8; 16] {
    if mantissa113 == 0 {
        return make_f128_zero(negative);
    }

    // f128 format: biased_exponent = unbiased_exponent + 16383
    let biased_exp = binary_exp + 16383;

    if biased_exp >= 0x7FFF {
        return make_f128_infinity(negative);
    }

    if biased_exp <= 0 {
        // Subnormal or underflow
        // TODO: the right-shift here truncates without rounding
        let shift = 1 - biased_exp;
        if shift >= 113 {
            return make_f128_zero(negative);
        }
        let mantissa_denorm = mantissa113 >> shift as u32;
        // f128: implicit bit is NOT stored, mantissa is bits [111:0]
        let mantissa_stored = mantissa_denorm & ((1u128 << 112) - 1);
        let sign_bit = if negative { 1u128 << 127 } else { 0 };
        let val = sign_bit | mantissa_stored;
        return val.to_le_bytes();
    }

    let exp15 = biased_exp as u128;
    // Remove implicit integer bit (bit 112), store only lower 112 bits
    let mantissa_stored = mantissa113 & ((1u128 << 112) - 1);
    let sign_bit = if negative { 1u128 << 127 } else { 0 };
    let val = sign_bit | (exp15 << 112) | mantissa_stored;
    val.to_le_bytes()
}

/// Parse a hex float string (0x..p..) to f128 format, rounded to nearest even.
fn parse_hex_float_to_f128(negative: bool, text: &str) -> [u8; 16] {
    let value = scan_hex_float(text);
    let magnitude = encode_binary(round_to_format(&value, F128_FORMAT), F128_FORMAT);
    let sign_bit: u128 = if negative { 1u128 << 127 } else { 0 };
    (sign_bit | magnitude).to_le_bytes()
}

// =============================================================================
// Hexadecimal floating constants
// =============================================================================
//
// C11 6.4.4.2p3: the significand of a hexadecimal floating constant is
// interpreted as an exact rational number scaled by 2^exponent, and when
// FLT_RADIX is a power of 2 the result is correctly rounded (to nearest even
// in the default rounding mode, which is what this compiler uses). The code
// below therefore rounds once, from the exact digit string, to the target
// format, rather than assembling the value through intermediate floating
// point operations that would each round.

/// The parameters of a binary interchange format that rounding depends on.
#[derive(Clone, Copy)]
struct BinaryFormat {
    /// Significand bits, including the leading bit that is implicit in IEEE encodings.
    precision: u32,
    /// Exponent bias. The largest finite exponent is `bias`, the smallest
    /// normal exponent is `1 - bias` and the all-ones exponent field is
    /// infinity.
    bias: i64,
}

const F32_FORMAT: BinaryFormat = BinaryFormat { precision: 24, bias: 127 };
const F64_FORMAT: BinaryFormat = BinaryFormat { precision: 53, bias: 1023 };
/// The x87 extended format has the exponent range of binary128 but only a
/// 64-bit significand (with an explicit leading bit). It is used for rounding
/// only; the result is stored in the binary128 carrier format.
const X87_FORMAT: BinaryFormat = BinaryFormat { precision: 64, bias: 16383 };
const F128_FORMAT: BinaryFormat = BinaryFormat { precision: 113, bias: 16383 };

/// The exact value of a hex float's digits: `(mantissa + f) * 2^exp2` with
/// `0 <= f < 1`, where `f` is nonzero exactly when `sticky` is set.
struct HexFloatValue {
    mantissa: u128,
    exp2: i64,
    sticky: bool,
}

/// A value rounded to some `BinaryFormat`.
enum Rounded {
    /// The magnitude `significand * 2^lsb_exp`, with `significand < 2^precision`.
    /// A zero significand is zero.
    Finite { significand: u128, lsb_exp: i64 },
    /// The rounded magnitude is too large for the format.
    Infinity,
}

/// Read the digits of a hex float (`0x<int>.<frac>p<exp>`, without sign; the
/// "0x" prefix and everything after the exponent are optional/ignored).
///
/// The first 124 or more bits after the leading zeros go into `mantissa`
/// (that is enough for any supported format plus rounding bits). Later digits
/// only record that the value is not exactly `mantissa * 2^exp2` in `sticky`;
/// integer digits among them still scale the exponent, so no digit is lost or
/// misweighted however long the constant is.
fn scan_hex_float(text: &str) -> HexFloatValue {
    let text = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")).unwrap_or(text);
    let bytes = text.as_bytes();
    let mut value = HexFloatValue { mantissa: 0, exp2: 0, sticky: false };
    let mut seen_point = false;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'.' && !seen_point {
            seen_point = true;
        } else if let Some(digit) = (bytes[i] as char).to_digit(16) {
            if value.mantissa >> 124 == 0 {
                value.mantissa = (value.mantissa << 4) | digit as u128;
                if seen_point {
                    value.exp2 -= 4;
                }
            } else {
                value.sticky |= digit != 0;
                if !seen_point {
                    value.exp2 += 4;
                }
            }
        } else {
            break;
        }
        i += 1;
    }

    if i < bytes.len() && (bytes[i] == b'p' || bytes[i] == b'P') {
        i += 1;
        let exp_neg = i < bytes.len() && bytes[i] == b'-';
        if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
            i += 1;
        }
        let mut exp: i64 = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            // Saturate: an exponent this large is out of range for every format
            // whatever the mantissa is, and the cap avoids i64 overflow.
            exp = (exp * 10 + (bytes[i] - b'0') as i64).min(1 << 40);
            i += 1;
        }
        value.exp2 += if exp_neg { -exp } else { exp };
    }
    value
}

/// Round `value` to `format` with round-to-nearest, ties-to-even, including
/// gradual underflow to subnormals and overflow to infinity.
fn round_to_format(value: &HexFloatValue, format: BinaryFormat) -> Rounded {
    let precision = format.precision as i64;
    let mantissa = value.mantissa;
    if mantissa == 0 {
        return Rounded::Finite { significand: 0, lsb_exp: 0 };
    }

    // Exponent of the leading bit, and of the last significand bit: below
    // the smallest normal exponent the spacing stays fixed (subnormals).
    let msb_exp = value.exp2 + 127 - mantissa.leading_zeros() as i64;
    let mut lsb_exp = (msb_exp - (precision - 1)).max(1 - format.bias - (precision - 1));

    // Number of low bits of `mantissa` that lie below the last significand bit.
    let shift = lsb_exp - value.exp2;
    let mut significand = if shift <= 0 {
        // Nothing is discarded. (`sticky` is only ever set for 125 or more
        // mantissa bits, which always need a positive shift.)
        mantissa << (-shift) as u32
    } else {
        let (kept, half, below_half) = if shift > 128 {
            (0, false, true)
        } else if shift == 128 {
            (0, mantissa >> 127 != 0, mantissa << 1 != 0)
        } else {
            let s = shift as u32;
            (mantissa >> s, (mantissa >> (s - 1)) & 1 == 1, mantissa & ((1u128 << (s - 1)) - 1) != 0)
        };
        if half && (below_half || value.sticky || kept & 1 == 1) { kept + 1 } else { kept }
    };

    // Rounding up can carry out of the significand.
    if significand >> precision != 0 {
        significand >>= 1;
        lsb_exp += 1;
    }
    if significand == 0 {
        return Rounded::Finite { significand: 0, lsb_exp: 0 };
    }
    if lsb_exp + (127 - significand.leading_zeros() as i64) > format.bias {
        return Rounded::Infinity;
    }
    Rounded::Finite { significand, lsb_exp }
}

/// The IEEE 754 bit pattern of a rounded magnitude (sign bit clear), for
/// binary32, binary64 and binary128.
fn encode_binary(rounded: Rounded, format: BinaryFormat) -> u128 {
    let frac_bits = format.precision - 1;
    match rounded {
        Rounded::Infinity => ((2 * format.bias + 1) as u128) << frac_bits,
        Rounded::Finite { significand, lsb_exp } => {
            if significand >> frac_bits == 0 {
                // Zero or subnormal: biased exponent 0, no implicit bit.
                significand
            } else {
                let biased_exp = (lsb_exp + frac_bits as i64 + format.bias) as u128;
                (biased_exp << frac_bits) | (significand & ((1u128 << frac_bits) - 1))
            }
        }
    }
}

/// Value of the hex float constant `text` (unsigned, with "0x" prefix) as a
/// `double`, rounded once from the exact value.
pub fn hex_float_to_f64(text: &str) -> f64 {
    let value = scan_hex_float(text);
    f64::from_bits(encode_binary(round_to_format(&value, F64_FORMAT), F64_FORMAT) as u64)
}

/// Value of the hex float constant `text` (unsigned, with "0x" prefix) as a
/// `float`, rounded once from the exact value. Going through `double` would
/// round twice and can give a different result.
pub fn hex_float_to_f32(text: &str) -> f32 {
    let value = scan_hex_float(text);
    f32::from_bits(encode_binary(round_to_format(&value, F32_FORMAT), F32_FORMAT) as u32)
}

/// Value of the hex float constant `text` (unsigned, with "0x" prefix) as a
/// `long double` of the target. Returns the f64 approximation and the
/// binary128 carrier bytes that long double tokens hold.
///
/// The constant is rounded once from the exact value to the target's format:
/// binary128 if `long_double_is_f128`, otherwise the 64-bit x87 significand.
/// An x87 result is stored exactly in the carrier, so the later (truncating)
/// conversion `f128_bytes_to_x87_bytes` does not lose anything. The f64
/// approximation is that long double value converted to double, i.e. rounded
/// a second time, as a cast would do (C11 6.3.1.5p1).
pub fn hex_float_to_long_double(text: &str, long_double_is_f128: bool) -> (f64, [u8; 16]) {
    let value = scan_hex_float(text);
    let format = if long_double_is_f128 { F128_FORMAT } else { X87_FORMAT };
    let (approx_bits, carrier) = match round_to_format(&value, format) {
        Rounded::Infinity => (encode_binary(Rounded::Infinity, F64_FORMAT), encode_binary(Rounded::Infinity, F128_FORMAT)),
        Rounded::Finite { significand, lsb_exp } => {
            // The rounded value is exact here, so there is no sticky tail.
            let exact = HexFloatValue { mantissa: significand, exp2: lsb_exp, sticky: false };
            (
                encode_binary(round_to_format(&exact, F64_FORMAT), F64_FORMAT),
                encode_binary(round_to_format(&exact, F128_FORMAT), F128_FORMAT),
            )
        }
    };
    (f64::from_bits(approx_bits as u64), carrier.to_le_bytes())
}

/// Construct raw f128 bytes from sign, biased exponent, and stored mantissa (no implicit bit).
fn make_f128_raw(negative: bool, biased_exp: u16, stored_mantissa: u128) -> [u8; 16] {
    let sign_bit: u128 = if negative { 1u128 << 127 } else { 0 };
    let val: u128 = sign_bit | ((biased_exp as u128) << 112) | stored_mantissa;
    val.to_le_bytes()
}

fn make_f128_zero(negative: bool) -> [u8; 16] {
    make_f128_raw(negative, 0, 0)
}

fn make_f128_infinity(negative: bool) -> [u8; 16] {
    make_f128_raw(negative, 0x7FFF, 0)
}

fn make_f128_nan(negative: bool) -> [u8; 16] {
    make_f128_raw(negative, 0x7FFF, 1u128 << 111)
}

/// Convert f128 bytes to x87 80-bit bytes. This is a lossy narrowing conversion
/// (112-bit mantissa → 64-bit mantissa) used when x87 format is needed (x86 backend,
/// x87 FPU constant folding).
pub fn f128_bytes_to_x87_bytes(f128_bytes: &[u8; 16]) -> [u8; 16] {
    let (sign, biased_exp, mantissa) = f128_decompose(f128_bytes);

    if biased_exp == 0 && mantissa == 0 {
        return make_x87_zero(sign);
    }
    if biased_exp == 0x7FFF {
        return if mantissa == 0 { make_x87_infinity(sign) } else { make_x87_nan(sign) };
    }

    // Normal number
    // f128 mantissa: 112 bits, implicit leading 1 (bit 112 set by f128_decompose)
    // x87 mantissa: 64 bits, explicit leading 1
    // Take top 64 bits of the 113-bit mantissa (mantissa >> 49)
    // Exponent bias is the same for both formats (16383)
    let mantissa64 = (mantissa >> 49) as u64;
    x87_encode(sign, biased_exp as u16, mantissa64)
}

/// Convert f128 bytes to f64 (lossy narrowing).
pub fn f128_bytes_to_f64(f128_bytes: &[u8; 16]) -> f64 {
    let (sign, biased_exp, mantissa) = f128_decompose(f128_bytes);
    let sign_u64 = if sign { 1u64 } else { 0u64 };

    if biased_exp == 0 && mantissa == 0 {
        return if sign { -0.0 } else { 0.0 };
    }
    if biased_exp == 0x7FFF {
        return if mantissa == 0 {
            if sign { f64::NEG_INFINITY } else { f64::INFINITY }
        } else {
            f64::NAN
        };
    }

    // Normal f128: value = (-1)^sign * 2^(biased_exp - 16383) * (1 + stored_mantissa/2^112)
    // Note: f128_decompose already adds the implicit bit, so mantissa has bit 112 set.
    let unbiased = (biased_exp - 16383) as i64;

    if (-1022..=1023).contains(&unbiased) {
        let f64_biased_exp = (unbiased + 1023) as u64;
        // Take top 52 bits of the 112-bit stored mantissa (strip implicit bit first)
        let stored = mantissa & ((1u128 << 112) - 1);
        let mantissa52 = (stored >> 60) as u64;
        let f64_bits = (sign_u64 << 63) | (f64_biased_exp << 52) | mantissa52;
        f64::from_bits(f64_bits)
    } else if unbiased > 1023 {
        if sign { f64::NEG_INFINITY } else { f64::INFINITY }
    } else {
        // Subnormal in f64
        let val = mantissa as f64 * 2.0_f64.powi(unbiased as i32 - 112);
        if sign { -val } else { val }
    }
}

/// Convert a signed i64 to f128 bytes with full precision.
/// f128 has 112-bit mantissa, so all i64 values are representable exactly.
pub fn i64_to_f128_bytes(val: i64) -> [u8; 16] {
    if val == 0 {
        return make_f128_zero(false);
    }
    let negative = val < 0;
    let abs_val: u64 = if val == i64::MIN {
        1u64 << 63
    } else if negative {
        (-val) as u64
    } else {
        val as u64
    };
    u64_to_f128_bytes_with_sign(abs_val, negative)
}

/// Convert an unsigned u64 to f128 bytes with full precision.
pub fn u64_to_f128_bytes(val: u64) -> [u8; 16] {
    if val == 0 {
        return make_f128_zero(false);
    }
    u64_to_f128_bytes_with_sign(val, false)
}

fn u64_to_f128_bytes_with_sign(val: u64, negative: bool) -> [u8; 16] {
    if val == 0 {
        return make_f128_zero(negative);
    }
    let bl = 64 - val.leading_zeros(); // number of significant bits
    // binary_exp = bl - 1 (position of MSB)
    // We need to normalize to 113-bit mantissa with bit 112 set
    let mantissa113: u128 = (val as u128) << (113 - bl);
    let binary_exp = (bl as i32) - 1;
    encode_f128(negative, binary_exp, mantissa113)
}

/// Convert an unsigned u128 to f128 bytes.
/// Values with more than 113 significant bits will be rounded.
pub fn u128_to_f128_bytes(val: u128) -> [u8; 16] {
    if val == 0 {
        return make_f128_zero(false);
    }
    if val <= u64::MAX as u128 {
        return u64_to_f128_bytes(val as u64);
    }
    let bl = 128 - val.leading_zeros(); // number of significant bits
    let mantissa113: u128 = if bl > 113 {
        val >> (bl - 113)
    } else if bl < 113 {
        val << (113 - bl)
    } else {
        val
    };
    let binary_exp = (bl as i32) - 1;
    encode_f128(false, binary_exp, mantissa113)
}

/// Convert a signed i128 to f128 bytes.
pub fn i128_to_f128_bytes(val: i128) -> [u8; 16] {
    if val == 0 {
        return make_f128_zero(false);
    }
    let negative = val < 0;
    let abs_val: u128 = if val == i128::MIN {
        1u128 << 127
    } else if negative {
        (-val) as u128
    } else {
        val as u128
    };
    if abs_val <= u64::MAX as u128 {
        return u64_to_f128_bytes_with_sign(abs_val as u64, negative);
    }
    let bl = 128 - abs_val.leading_zeros();
    let mantissa113: u128 = if bl > 113 {
        abs_val >> (bl - 113)
    } else if bl < 113 {
        abs_val << (113 - bl)
    } else {
        abs_val
    };
    let binary_exp = (bl as i32) - 1;
    encode_f128(negative, binary_exp, mantissa113)
}

/// Convert f64 to f128 bytes (widening, zero-fills extra mantissa bits).
pub fn f64_to_f128_bytes_lossless(val: f64) -> [u8; 16] {
    let d = f64_decompose(val);

    if d.is_zero() {
        return make_f128_zero(d.sign);
    }
    if d.is_special() {
        return if d.is_inf() { make_f128_infinity(d.sign) } else { make_f128_nan(d.sign) };
    }

    // Normal f64: exp = biased_exp - 1023, mantissa = 1.mantissa52
    // f128: exp = biased_exp - 1023 + 16383, mantissa112 = mantissa52 << (112 - 52)
    let exp15 = (d.biased_exp as u128 - 1023 + 16383) as u128;
    let mantissa112: u128 = (d.mantissa as u128) << 60; // 112 - 52 = 60
    let sign_bit: u128 = if d.sign { 1u128 << 127 } else { 0 };
    let val128: u128 = sign_bit | (exp15 << 112) | mantissa112;
    val128.to_le_bytes()
}

// =============================================================================
// f128 to integer conversion: direct extraction without lossy x87 intermediate
// =============================================================================

/// Core f128-to-unsigned-integer conversion. Extracts the absolute value of an f128
/// float as u128, with truncation toward zero. `max_bits` limits the output width.
/// Returns None for inf/NaN/overflow.
fn f128_to_abs_uint(bytes: &[u8; 16], max_bits: u32) -> Option<(bool, u128)> {
    let (sign, biased_exp, mantissa) = f128_decompose(bytes);

    if biased_exp == 0 && mantissa == 0 {
        return Some((sign, 0));
    }
    if biased_exp == 0x7FFF {
        return None; // inf or NaN
    }

    // For normals, f128_decompose returns mantissa with bit 112 set (the implicit bit).
    // value = mantissa * 2^(biased_exp - 16383 - 112)
    let unbiased = biased_exp - 16383;
    let shift = unbiased - 112;

    if shift >= 0 {
        let shift = shift as u32;
        if shift >= max_bits {
            return None; // overflow
        }
        let abs_val = mantissa << shift;
        if max_bits < 128 && abs_val >> max_bits != 0 {
            return None;
        }
        Some((sign, abs_val))
    } else {
        let rshift = (-shift) as u32;
        if rshift >= 128 {
            return Some((sign, 0));
        }
        Some((sign, mantissa >> rshift))
    }
}

/// Convert f128 bytes to i64 (for constant folding). Direct extraction with full
/// 112-bit mantissa precision (no lossy x87 intermediate).
pub fn f128_bytes_to_i64(bytes: &[u8; 16]) -> Option<i64> {
    let (sign, abs_val) = f128_to_abs_uint(bytes, 64)?;
    if sign {
        if abs_val > i64::MAX as u128 + 1 {
            return None;
        }
        Some(-(abs_val as i64))
    } else {
        if abs_val > i64::MAX as u128 {
            return None;
        }
        Some(abs_val as i64)
    }
}

/// Convert f128 bytes to u64 (for constant folding).
pub fn f128_bytes_to_u64(bytes: &[u8; 16]) -> Option<u64> {
    let (sign, abs_val) = f128_to_abs_uint(bytes, 64)?;
    if sign {
        let signed_val = f128_bytes_to_i64(bytes)?;
        return Some(signed_val as u64);
    }
    Some(abs_val as u64)
}

/// Convert f128 bytes to i128 (for constant folding).
pub fn f128_bytes_to_i128(bytes: &[u8; 16]) -> Option<i128> {
    let (sign, abs_val) = f128_to_abs_uint(bytes, 128)?;
    if sign {
        if abs_val > i128::MAX as u128 + 1 {
            return None;
        }
        Some(-(abs_val as i128))
    } else {
        if abs_val > i128::MAX as u128 {
            return None;
        }
        Some(abs_val as i128)
    }
}

/// Convert f128 bytes to u128 (for constant folding).
pub fn f128_bytes_to_u128(bytes: &[u8; 16]) -> Option<u128> {
    let (sign, abs_val) = f128_to_abs_uint(bytes, 128)?;
    if sign {
        let signed_val = f128_bytes_to_i128(bytes)?;
        return Some(signed_val as u128);
    }
    Some(abs_val)
}

/// Create x87 bytes from an f64 value (for when we don't have the original text).
/// This is a widening conversion that zero-fills the extra mantissa bits.
pub fn f64_to_x87_bytes_simple(val: f64) -> [u8; 16] {
    let d = f64_decompose(val);

    if d.is_zero() {
        return make_x87_zero(d.sign);
    }
    if d.is_special() {
        return if d.is_inf() { make_x87_infinity(d.sign) } else { make_x87_nan(d.sign) };
    }

    // Normal f64: x87 mantissa has explicit integer bit at 63, then 52 bits at 62..11
    let exp15 = (d.biased_exp as i32 - 1023 + 16383) as u16;
    let mantissa64 = (1u64 << 63) | (d.mantissa << 11);
    x87_encode(d.sign, exp15, mantissa64)
}

// ============================================================================
// x87 full-precision arithmetic on raw 80-bit bytes
// ============================================================================
//
// These functions perform arithmetic on x87 80-bit extended precision values
// stored as [u8; 16] byte arrays (10 bytes of x87 data + 6 padding bytes).
// All operations use pure software implementations (no inline asm), producing
// bit-identical results to the x87 FPU hardware.

/// Add two x87 80-bit extended precision values with full precision.
pub fn x87_add(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    x87_add_soft(a, b)
}

/// Subtract two x87 80-bit extended precision values with full precision.
pub fn x87_sub(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    x87_sub_soft(a, b)
}

/// Multiply two x87 80-bit extended precision values with full precision.
pub fn x87_mul(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    x87_mul_soft(a, b)
}

/// Divide two x87 80-bit extended precision values with full precision.
pub fn x87_div(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    x87_div_soft(a, b)
}

/// Software multiply for x87 80-bit extended precision values.
/// Produces bit-identical results to the x87 FPU fmulp instruction.
///
/// x87 extended format: 1 sign bit, 15-bit biased exponent (bias=16383),
/// 64-bit mantissa with explicit integer bit at position 63.
pub fn x87_mul_soft(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let da = x87_decompose(a);
    let db = x87_decompose(b);
    let sign = da.sign ^ db.sign;

    // x87 "indefinite" QNaN: negative, exp=0x7FFF, mantissa=0xC000_0000_0000_0000
    let indefinite_nan = x87_encode(true, 0x7FFF, 0xC000_0000_0000_0000);

    // Unnormals and pseudo-values: biased_exp > 0 but integer bit (bit 63) not set.
    // x87 raises #IA and returns the indefinite NaN for these invalid encodings.
    // This includes pseudo-NaN/pseudo-infinity (exp=0x7FFF, integer bit clear).
    let a_is_unnormal = da.biased_exp > 0 && (da.mantissa >> 63) == 0;
    let b_is_unnormal = db.biased_exp > 0 && (db.mantissa >> 63) == 0;
    if a_is_unnormal || b_is_unnormal {
        return indefinite_nan;
    }

    // At this point, all values with biased_exp > 0 have the integer bit set.
    // Infinity: exp=0x7FFF, mantissa=exactly 0x8000_0000_0000_0000
    let a_is_inf = da.biased_exp == 0x7FFF && da.mantissa == 0x8000_0000_0000_0000;
    let b_is_inf = db.biased_exp == 0x7FFF && db.mantissa == 0x8000_0000_0000_0000;
    // NaN: exp=0x7FFF with integer bit set but fraction bits nonzero
    let a_is_nan = da.biased_exp == 0x7FFF && !a_is_inf;
    let b_is_nan = db.biased_exp == 0x7FFF && !b_is_inf;

    // NaN propagation: return the input NaN with quiet bit set
    if a_is_nan {
        let mut result = *a;
        result[7] |= 0x40; // set quiet NaN bit
        return result;
    }
    if b_is_nan {
        let mut result = *b;
        result[7] |= 0x40; // set quiet NaN bit
        return result;
    }

    // Inf * 0 = indefinite NaN (invalid operation)
    if a_is_inf && db.is_zero() {
        return indefinite_nan;
    }
    if b_is_inf && da.is_zero() {
        return indefinite_nan;
    }

    // Inf * anything (non-zero, non-NaN) = Inf
    if a_is_inf || b_is_inf {
        return make_x87_infinity(sign);
    }

    // Zero * anything = Zero
    if da.is_zero() || db.is_zero() {
        return make_x87_zero(sign);
    }

    // Both operands are finite and nonzero.
    // Compute unbiased exponents. For denormals (biased_exp==0), the true
    // exponent is 1-16383 = -16382 (not 0-16383).
    let exp_a = if da.biased_exp == 0 { -16382i32 } else { da.biased_exp as i32 - 16383 };
    let exp_b = if db.biased_exp == 0 { -16382i32 } else { db.biased_exp as i32 - 16383 };
    let mut exp = exp_a + exp_b;

    // Multiply the two 64-bit mantissas to get a 128-bit product.
    let product = (da.mantissa as u128) * (db.mantissa as u128);

    if product == 0 {
        return make_x87_zero(sign);
    }

    // For two normal numbers (integer bit set), the product's MSB is at bit 126
    // of the 128-bit product. We need to place the result so the integer bit
    // is at bit 63 of a 64-bit mantissa, then do a single rounding step.
    let msb = 127 - product.leading_zeros() as i32;
    // Adjust exponent for where the MSB actually landed vs the expected bit 126.
    exp += msb - 126;

    // Now we need to extract a 64-bit mantissa from the product, with the MSB
    // at bit 63. The total right-shift from the 128-bit product to get there:
    let norm_shift = msb - 63; // bits to shift right for normalization

    // Check if result will be denormal (biased_exp would be <= 0).
    // If so, we need additional right-shift to force biased_exp = 0.
    let biased_exp = exp + 16383;
    let (total_shift, final_biased_exp) = if biased_exp <= 0 {
        // Extra shift to denormalize
        let extra = 1 - biased_exp;
        (norm_shift as i64 + extra as i64, 0i32)
    } else {
        (norm_shift as i64, biased_exp)
    };

    // Check for overflow → infinity
    if final_biased_exp >= 0x7FFF {
        return make_x87_infinity(sign);
    }

    // Extract 64-bit mantissa with a single rounding step on the full 128-bit product.
    if total_shift > 128 {
        // Even the full 128-bit product is less than half the minimum denormal.
        return make_x87_zero(sign);
    }
    if total_shift == 128 {
        // The mantissa is 0; the entire product is the rounding tail.
        // halfway = 1 << 127. Round up if product > halfway, or product == halfway
        // and result bit 0 would be 1 (but result is 0, so round-to-even → no).
        if product > (1u128 << 127) {
            return x87_encode(sign, final_biased_exp as u16, 1);
        }
        return make_x87_zero(sign);
    }

    let (mut mantissa, round_up) = if total_shift > 0 {
        let ts = total_shift as u32;
        let m = if ts >= 128 { 0u64 } else { (product >> ts) as u64 };
        // Rounding bits from the shifted-away portion
        let halfway = if ts >= 128 { 0u128 } else { 1u128 << (ts - 1) };
        let mask = if ts >= 128 { u128::MAX } else { (1u128 << ts) - 1 };
        let tail = product & mask;
        let round = tail > halfway || (tail == halfway && m & 1 != 0);
        (m, round)
    } else if total_shift == 0 {
        (product as u64, false)
    } else {
        // total_shift < 0: shift left (very small denormal inputs)
        let ls = (-total_shift) as u32;
        if ls >= 128 { return make_x87_zero(sign); }
        let m = (product << ls) as u64;
        (m, false)
    };

    if round_up {
        let old_mantissa = mantissa;
        mantissa = mantissa.wrapping_add(1);
        if mantissa == 0 {
            // Overflow from 0xFFFF_FFFF_FFFF_FFFF → 0: result is 1 << 63 with exp+1
            mantissa = 1u64 << 63;
            if final_biased_exp > 0 {
                let new_exp = final_biased_exp + 1;
                if new_exp >= 0x7FFF {
                    return make_x87_infinity(sign);
                }
                return x87_encode(sign, new_exp as u16, mantissa);
            }
            // Was denormal, rounding carried into normal range
            return x87_encode(sign, 1, mantissa);
        }
        // Check if rounding a denormal caused carry into normal range:
        // the integer bit (bit 63) was clear before but is now set.
        if final_biased_exp == 0 && (old_mantissa >> 63) == 0 && (mantissa >> 63) == 1 {
            return x87_encode(sign, 1, mantissa);
        }
    }

    if mantissa == 0 {
        return make_x87_zero(sign);
    }

    x87_encode(sign, final_biased_exp as u16, mantissa)
}

/// Classify an x87 value for arithmetic special-case handling.
/// Returns (is_unnormal, is_inf, is_nan).
/// For unnormals/pseudo-values, the caller should return indefinite NaN.
fn x87_classify(d: &X87Decomposed) -> (bool, bool, bool) {
    let unnormal = d.biased_exp > 0 && (d.mantissa >> 63) == 0;
    let inf = d.biased_exp == 0x7FFF && d.mantissa == 0x8000_0000_0000_0000;
    let nan = d.biased_exp == 0x7FFF && !inf && !unnormal;
    (unnormal, inf, nan)
}

/// Get the true (unbiased) exponent for an x87 value.
/// Denormals (biased_exp=0) have true exponent -16382.
fn x87_true_exp(d: &X87Decomposed) -> i32 {
    if d.biased_exp == 0 { -16382 } else { d.biased_exp as i32 - 16383 }
}

/// x87 indefinite QNaN constant.
const X87_INDEFINITE_NAN: [u8; 16] = {
    let mut b = [0u8; 16];
    // mantissa = 0xC000_0000_0000_0000, exp_sign = 0xFFFF (sign=1, exp=0x7FFF)
    b[7] = 0xC0;
    b[8] = 0xFF;
    b[9] = 0xFF;
    b
};

/// Propagate a NaN operand: set the quiet bit and return it.
fn x87_quiet_nan(a: &[u8; 16]) -> [u8; 16] {
    let mut result = *a;
    result[7] |= 0x40;
    result
}

/// Software addition/subtraction for x87 80-bit extended precision.
/// If `negate_b` is true, performs a - b; otherwise a + b.
fn x87_addsub_soft(a: &[u8; 16], b: &[u8; 16], negate_b: bool) -> [u8; 16] {
    let da = x87_decompose(a);
    let db = x87_decompose(b);
    let (a_unnorm, a_inf, a_nan) = x87_classify(&da);
    let (b_unnorm, b_inf, b_nan) = x87_classify(&db);

    // Effective sign of b after potential negation
    let b_sign = db.sign ^ negate_b;

    if a_unnorm || b_unnorm { return X87_INDEFINITE_NAN; }
    if a_nan { return x87_quiet_nan(a); }
    if b_nan { return x87_quiet_nan(b); }

    if a_inf && b_inf {
        // inf + inf = inf (same sign), inf - inf = NaN (different sign)
        if da.sign == b_sign {
            return make_x87_infinity(da.sign);
        } else {
            return X87_INDEFINITE_NAN;
        }
    }
    if a_inf { return make_x87_infinity(da.sign); }
    if b_inf { return make_x87_infinity(b_sign); }

    if da.is_zero() && db.is_zero() {
        // +0 + +0 = +0, -0 + -0 = -0, +0 + -0 = +0 (round to nearest)
        return make_x87_zero(da.sign && b_sign);
    }
    if da.is_zero() { return x87_encode(b_sign, db.biased_exp, db.mantissa); }
    if db.is_zero() { return *a; }

    // Both finite, nonzero. Align mantissas and add/subtract.
    let exp_a = x87_true_exp(&da);
    let exp_b = x87_true_exp(&db);

    // Use 128-bit mantissas for precision during alignment shift.
    // Place the 64-bit mantissa in the upper 64 bits, leaving room for guard/round/sticky.
    let mut ma = (da.mantissa as u128) << 64;
    let mut mb = (db.mantissa as u128) << 64;
    let mut exp = exp_a;

    if exp_a > exp_b {
        let shift = (exp_a - exp_b) as u32;
        if shift < 128 {
            let sticky = mb & ((1u128 << shift) - 1) != 0;
            mb >>= shift;
            if sticky { mb |= 1; }
        } else {
            mb = if mb != 0 { 1 } else { 0 };
        }
    } else if exp_b > exp_a {
        let shift = (exp_b - exp_a) as u32;
        if shift < 128 {
            let sticky = ma & ((1u128 << shift) - 1) != 0;
            ma >>= shift;
            if sticky { ma |= 1; }
        } else {
            ma = if ma != 0 { 1 } else { 0 };
        }
        exp = exp_b;
    }

    // Perform addition or subtraction based on effective signs.
    // For addition of same-sign values, the sum can overflow u128 (carry bit),
    // so we track the carry separately.
    let (result_sign, result_mag, carry) = if da.sign == b_sign {
        // Same sign: add magnitudes
        let (sum, overflow) = ma.overflowing_add(mb);
        (da.sign, sum, overflow)
    } else {
        // Different signs: subtract magnitudes (no carry possible)
        if ma >= mb {
            (da.sign, ma - mb, false)
        } else {
            (b_sign, mb - ma, false)
        }
    };

    if result_mag == 0 && !carry {
        return make_x87_zero(false); // +0 in round-to-nearest mode
    }

    // Normalize the result. We have a 129-bit value: carry bit + 128-bit result_mag.
    // We want to produce a 128-bit normalized value with bit 127 as the integer bit,
    // then extract the top 64 bits with rounding.
    if carry {
        // The true value is 2^128 + result_mag. The MSB is at bit 128 (virtual).
        // We need to shift right by 1 to fit in 128 bits, preserving sticky.
        exp += 1; // carry means the result is one bit wider
        let sticky = result_mag & 1 != 0;
        let normalized = (result_mag >> 1) | (1u128 << 127); // restore the carry bit
        let normalized = if sticky { normalized | 1 } else { normalized };

        // Extract top 64 bits with rounding.
        let mantissa_hi = (normalized >> 64) as u64;
        let tail = normalized as u64;
        let halfway = 1u64 << 63;
        let mut mantissa = mantissa_hi;
        let round_up = tail > halfway || (tail == halfway && mantissa & 1 != 0);

        let biased_exp = exp + 16383;
        if round_up {
            mantissa = mantissa.wrapping_add(1);
            if mantissa == 0 {
                mantissa = 1u64 << 63;
                let new_exp = biased_exp + 1;
                if new_exp >= 0x7FFF { return make_x87_infinity(result_sign); }
                return x87_encode(result_sign, new_exp as u16, mantissa);
            }
        }
        if biased_exp >= 0x7FFF { return make_x87_infinity(result_sign); }
        return x87_encode(result_sign, biased_exp as u16, mantissa);
    }

    // No carry — normalize by finding the MSB position
    let msb = 127 - result_mag.leading_zeros() as i32;
    let shift_to_normalize = msb - 127;
    exp += shift_to_normalize;

    let normalized = if shift_to_normalize > 0 {
        // Result grew — shift right (shouldn't happen without carry, but handle it)
        let s = shift_to_normalize as u32;
        let sticky = result_mag & ((1u128 << s) - 1) != 0;
        let shifted = result_mag >> s;
        if sticky { shifted | 1 } else { shifted }
    } else if shift_to_normalize < 0 {
        // Result shrank (cancellation in subtraction) — shift left
        let s = (-shift_to_normalize) as u32;
        result_mag << s
    } else {
        result_mag
    };

    // Now the integer bit is at bit 127. Extract top 64 bits with rounding.
    let mantissa_hi = (normalized >> 64) as u64;
    let tail = normalized as u64;
    let halfway = 1u64 << 63;
    let mut mantissa = mantissa_hi;
    let round_up = tail > halfway || (tail == halfway && mantissa & 1 != 0);

    let biased_exp = exp + 16383;
    if biased_exp <= 0 {
        // Denormalize
        let denorm_shift = 1 - biased_exp;
        if denorm_shift > 128 {
            return make_x87_zero(result_sign);
        }
        let total_shift = 64 + denorm_shift as i64;
        if total_shift >= 128 {
            if total_shift == 128 && normalized > (1u128 << 127) {
                return x87_encode(result_sign, 0, 1);
            }
            return make_x87_zero(result_sign);
        }
        let ts = total_shift as u32;
        let m = (normalized >> ts) as u64;
        let mask = (1u128 << ts) - 1;
        let t = normalized & mask;
        let hw = 1u128 << (ts - 1);
        let mut dm = m;
        if t > hw || (t == hw && dm & 1 != 0) {
            dm += 1;
            if dm >> 63 == 1 && m >> 63 == 0 {
                return x87_encode(result_sign, 1, dm);
            }
        }
        if dm == 0 { return make_x87_zero(result_sign); }
        return x87_encode(result_sign, 0, dm);
    }

    if round_up {
        mantissa = mantissa.wrapping_add(1);
        if mantissa == 0 {
            mantissa = 1u64 << 63;
            let new_exp = biased_exp + 1;
            if new_exp >= 0x7FFF { return make_x87_infinity(result_sign); }
            return x87_encode(result_sign, new_exp as u16, mantissa);
        }
    }

    if biased_exp >= 0x7FFF { return make_x87_infinity(result_sign); }

    x87_encode(result_sign, biased_exp as u16, mantissa)
}

/// Software addition for x87 80-bit extended precision.
pub fn x87_add_soft(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    x87_addsub_soft(a, b, false)
}

/// Software subtraction for x87 80-bit extended precision.
pub fn x87_sub_soft(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    x87_addsub_soft(a, b, true)
}

/// Software division for x87 80-bit extended precision.
pub fn x87_div_soft(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let da = x87_decompose(a);
    let db = x87_decompose(b);
    let sign = da.sign ^ db.sign;
    let (a_unnorm, a_inf, a_nan) = x87_classify(&da);
    let (b_unnorm, b_inf, b_nan) = x87_classify(&db);

    if a_unnorm || b_unnorm { return X87_INDEFINITE_NAN; }
    if a_nan { return x87_quiet_nan(a); }
    if b_nan { return x87_quiet_nan(b); }

    // inf / inf = NaN
    if a_inf && b_inf { return X87_INDEFINITE_NAN; }
    // inf / finite = inf
    if a_inf { return make_x87_infinity(sign); }
    // finite / inf = 0
    if b_inf { return make_x87_zero(sign); }
    // 0 / 0 = NaN
    if da.is_zero() && db.is_zero() { return X87_INDEFINITE_NAN; }
    // 0 / finite = 0
    if da.is_zero() { return make_x87_zero(sign); }
    // finite / 0 = inf (division by zero)
    if db.is_zero() { return make_x87_infinity(sign); }

    // Both finite, nonzero
    let exp_a = x87_true_exp(&da);
    let exp_b = x87_true_exp(&db);
    let mut exp = exp_a - exp_b;

    // Divide mantissas: we need 64 bits of quotient plus guard bits for rounding.
    // Compute (da.mantissa << 64) / db.mantissa to get a 64-bit quotient
    // with the decimal point after bit 63.
    let dividend = (da.mantissa as u128) << 64;
    let divisor = db.mantissa as u128;
    let quotient = dividend / divisor;
    let remainder = dividend % divisor;

    if quotient == 0 {
        return make_x87_zero(sign);
    }

    // The quotient has the integer bit somewhere. For normal/normal where both
    // have the integer bit at position 63, quotient ≈ 2^64 * (1.xxx / 1.yyy).
    // If mantissa_a >= mantissa_b, quotient is in [2^64, 2^65).
    // If mantissa_a < mantissa_b, quotient is in [2^63, 2^64).
    let msb = 127 - quotient.leading_zeros() as i32;
    // We want the integer bit at position 63 of a 64-bit result.
    let shift = msb - 63;
    exp += shift - 1; // quotient = (mantissa_a << 64) / mantissa_b, value = quotient * 2^(-64) * 2^(exp_a-exp_b); after shifting right by (msb-63), true_exp = exp + msb - 63 - 64 + 63 = exp + shift - 1

    // When shift < 0, extend the quotient by computing more division bits.
    let (quotient, remainder, shift) = if shift < 0 {
        let s = (-shift) as u32;
        // Compute (dividend << s) / divisor to get s more quotient bits.
        // dividend << s = (quotient * divisor + remainder) << s
        //               = quotient << s * divisor + remainder << s
        // New quotient = quotient << s + (remainder << s) / divisor
        let extra_dividend = remainder << s;
        let extra_q = extra_dividend / divisor;
        let extra_r = extra_dividend % divisor;
        let new_q = (quotient << s) | extra_q;
        (new_q, extra_r, 0i32)
    } else {
        (quotient, remainder, shift)
    };

    let (mut mantissa, round_up) = if shift > 0 {
        let s = shift as u32;
        let m = (quotient >> s) as u64;
        let tail = quotient & ((1u128 << s) - 1);
        let halfway = 1u128 << (s - 1);
        // For exact halfway, also check remainder (sticky bit from division)
        let round = tail > halfway || (tail == halfway && (m & 1 != 0 || remainder != 0));
        (m, round)
    } else {
        // shift == 0 (shift < 0 was handled above)
        let m = quotient as u64;
        // remainder != 0 means the true value is quotient + remainder/divisor.
        // Round up if remainder/divisor > 0.5 (i.e., 2*remainder > divisor),
        // or at exactly 0.5 and m is odd (round-to-even).
        let double_rem = remainder << 1;
        let round = double_rem > divisor || (double_rem == divisor && m & 1 != 0);
        (m, round)
    };

    let biased_exp = exp + 16383;

    // Handle denormal
    if biased_exp <= 0 {
        let denorm_shift = 1 - biased_exp;
        // Work on the full quotient to avoid double-rounding
        let total_shift = shift as i64 + denorm_shift as i64;
        if total_shift > 127 {
            return make_x87_zero(sign);
        }
        if total_shift >= 0 {
            let ts = total_shift as u32;
            let m = if ts >= 128 { 0u64 } else { (quotient >> ts) as u64 };
            let mask = if ts >= 128 { u128::MAX } else { (1u128 << ts) - 1 };
            let tail = quotient & mask;
            let hw = if ts == 0 { 0u128 } else { 1u128 << (ts - 1) };
            let mut dm = m;
            let round = tail > hw || (tail == hw && (dm & 1 != 0 || remainder != 0));
            if round {
                dm += 1;
                if dm >> 63 == 1 && m >> 63 == 0 {
                    return x87_encode(sign, 1, dm);
                }
            }
            if dm == 0 { return make_x87_zero(sign); }
            return x87_encode(sign, 0, dm);
        }
        // total_shift < 0: shift left
        let s = (-total_shift) as u32;
        let m = (quotient << s) as u64;
        if m == 0 { return make_x87_zero(sign); }
        return x87_encode(sign, 0, m);
    }

    if round_up {
        mantissa = mantissa.wrapping_add(1);
        if mantissa == 0 {
            mantissa = 1u64 << 63;
            let new_exp = biased_exp + 1;
            if new_exp >= 0x7FFF { return make_x87_infinity(sign); }
            return x87_encode(sign, new_exp as u16, mantissa);
        }
    }

    if biased_exp >= 0x7FFF { return make_x87_infinity(sign); }
    x87_encode(sign, biased_exp as u16, mantissa)
}

/// Software remainder for x87 80-bit extended precision (fprem semantics).
/// Computes a - trunc(a/b) * b (truncation toward zero, like C fmod).
pub fn x87_rem_soft(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let da = x87_decompose(a);
    let db = x87_decompose(b);
    let (a_unnorm, a_inf, a_nan) = x87_classify(&da);
    let (b_unnorm, b_inf, b_nan) = x87_classify(&db);

    if a_unnorm || b_unnorm { return X87_INDEFINITE_NAN; }
    if a_nan { return x87_quiet_nan(a); }
    if b_nan { return x87_quiet_nan(b); }
    // inf % anything = NaN
    if a_inf { return X87_INDEFINITE_NAN; }
    // finite % inf = finite (unchanged)
    if b_inf { return *a; }
    // anything % 0 = NaN
    if db.is_zero() { return X87_INDEFINITE_NAN; }
    // 0 % finite = 0
    if da.is_zero() { return *a; }

    // Both finite, nonzero. Compute a - trunc(a/b) * b (fprem semantics, like C fmod).
    // Result sign = sign of dividend (a). The result is exact (no rounding needed).
    let result_sign = da.sign;
    let exp_a = x87_true_exp(&da);
    let exp_b = x87_true_exp(&db);

    // Compare |a| vs |b|. If |a| < |b|, result is a (possibly normalized).
    // x87 fprem normalizes pseudo-denormals (biased_exp=0, integer bit set) → biased_exp=1.
    if exp_a < exp_b || (exp_a == exp_b && da.mantissa < db.mantissa) {
        if da.biased_exp == 0 && (da.mantissa >> 63) == 1 {
            return x87_encode(result_sign, 1, da.mantissa);
        }
        return *a;
    }

    // Iterative bit-by-bit reduction (like hardware fprem).
    // At each step, if rem >= div * 2^k, subtract div * 2^k.
    // This avoids precision loss from integer division with truncation.
    //
    // Both mantissas have MSB at bit 63. The exponent difference tells us
    // how many "extra" bits of magnitude |a| has over |b|.
    let exp_diff = exp_a - exp_b;

    // Work with the 64-bit mantissas directly (remainder is exact).
    let mut rem = da.mantissa as u128;
    let div = db.mantissa as u128;

    // Compute (mantissa_a * 2^exp_diff) % mantissa_b, which gives the remainder
    // in terms of the divisor's mantissa scale (at exponent exp_b).
    // Since exp_diff can be large (up to ~32K), we shift-and-reduce in chunks.
    let mut remaining_shift = exp_diff;
    loop {
        let chunk = if remaining_shift > 63 { 63 } else { remaining_shift };
        if chunk > 0 {
            rem <<= chunk as u32;
        }
        rem %= div;
        remaining_shift -= chunk;
        if remaining_shift == 0 { break; }
    }

    if rem == 0 {
        return make_x87_zero(result_sign);
    }

    // rem now contains the integer remainder at exponent exp_b.
    // Normalize: find MSB position and set the exponent accordingly.
    let msb = 127 - rem.leading_zeros() as i32;
    let target_msb = 63;
    let shift = msb - target_msb;
    let result_exp = exp_b + shift;
    let mantissa = if shift > 0 {
        (rem >> shift as u32) as u64
    } else if shift < 0 {
        (rem << (-shift) as u32) as u64
    } else {
        rem as u64
    };

    let biased = result_exp + 16383;
    if biased <= 0 {
        let ds = 1 - biased;
        if ds >= 64 { return make_x87_zero(result_sign); }
        let m = mantissa >> ds as u32;
        if m == 0 { return make_x87_zero(result_sign); }
        return x87_encode(result_sign, 0, m);
    }

    x87_encode(result_sign, biased as u16, mantissa)
}

/// Software comparison for x87 80-bit extended precision.
/// Returns: -1 if a < b, 0 if a == b, 1 if a > b, i32::MIN if unordered (NaN).
pub fn x87_cmp_soft(a: &[u8; 16], b: &[u8; 16]) -> i32 {
    let da = x87_decompose(a);
    let db = x87_decompose(b);
    let (a_unnorm, _, a_nan) = x87_classify(&da);
    let (b_unnorm, _, b_nan) = x87_classify(&db);

    // Any NaN or unnormal → unordered
    if a_nan || b_nan || a_unnorm || b_unnorm {
        return i32::MIN;
    }

    // Handle zeros: +0 == -0
    if da.is_zero() && db.is_zero() {
        return 0;
    }

    // Different signs (and not both zero)
    if da.sign != db.sign {
        return if da.sign { -1 } else { 1 };
    }

    // Same sign: compare magnitude, flip result if negative
    let neg = da.sign;

    // Compare biased exponents first, then mantissa
    let ord = if da.biased_exp != db.biased_exp {
        da.biased_exp.cmp(&db.biased_exp)
    } else {
        da.mantissa.cmp(&db.mantissa)
    };

    match ord {
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => if neg { -1 } else { 1 },
        std::cmp::Ordering::Less => if neg { 1 } else { -1 },
    }
}

/// Negate an x87 80-bit extended precision value by flipping the sign bit.
/// This preserves full precision since it only changes one bit.
pub fn x87_neg(a: &[u8; 16]) -> [u8; 16] {
    let mut result = *a;
    result[9] ^= 0x80; // flip the sign bit (bit 15 of exponent+sign word)
    result
}

/// Compute the remainder of two x87 80-bit extended precision values (fprem semantics).
pub fn x87_rem(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    x87_rem_soft(a, b)
}

/// Compare two x87 80-bit extended precision values.
/// Returns: -1 if a < b, 0 if a == b, 1 if a > b, i32::MIN if unordered (NaN).
pub fn x87_cmp(a: &[u8; 16], b: &[u8; 16]) -> i32 {
    x87_cmp_soft(a, b)
}

/// Get the f64 approximation from x87 bytes, for use when we still need f64.
/// This is a convenience alias for x87_bytes_to_f64.
pub fn x87_to_f64(bytes: &[u8; 16]) -> f64 {
    x87_bytes_to_f64(bytes)
}

// =============================================================================
// IEEE 754 binary128 (f128) software arithmetic
// =============================================================================
//
// These functions perform arithmetic directly on [u8; 16] f128 byte arrays,
// giving full 112-bit mantissa precision. This avoids the precision loss that
// occurs when converting f128 → x87 (64-bit mantissa) → f128 for constant
// folding on ARM/RISC-V targets where long double is f128.

/// Helper: decompose f128 bytes into (sign, biased_exponent, mantissa_with_implicit_bit).
/// For normal numbers, mantissa has bit 112 set (the implicit leading 1).
/// For subnormals, mantissa does NOT have bit 112 set.
/// Returns (sign: bool, biased_exp: i32, mantissa: u128)
fn f128_decompose(bytes: &[u8; 16]) -> (bool, i32, u128) {
    let val = u128::from_le_bytes(*bytes);
    let sign = (val >> 127) != 0;
    let biased_exp = ((val >> 112) & 0x7FFF) as i32;
    let mantissa = val & ((1u128 << 112) - 1);

    if biased_exp == 0 {
        // Zero or subnormal (no implicit bit)
        (sign, 0, mantissa)
    } else if biased_exp == 0x7FFF {
        // Inf or NaN: keep exponent, mantissa distinguishes them
        (sign, 0x7FFF, mantissa)
    } else {
        // Normal: add implicit leading 1 at bit 112
        (sign, biased_exp, mantissa | (1u128 << 112))
    }
}

/// Helper: check if f128 is zero
fn f128_is_zero(bytes: &[u8; 16]) -> bool {
    let val = u128::from_le_bytes(*bytes);
    (val & !(1u128 << 127)) == 0
}

/// Helper: check if f128 is infinity
fn f128_is_inf(bytes: &[u8; 16]) -> bool {
    let val = u128::from_le_bytes(*bytes);
    let exp = (val >> 112) & 0x7FFF;
    let mantissa = val & ((1u128 << 112) - 1);
    exp == 0x7FFF && mantissa == 0
}

/// Helper: check if f128 is NaN
fn f128_is_nan(bytes: &[u8; 16]) -> bool {
    let val = u128::from_le_bytes(*bytes);
    let exp = (val >> 112) & 0x7FFF;
    let mantissa = val & ((1u128 << 112) - 1);
    exp == 0x7FFF && mantissa != 0
}

/// Helper: round a 128-bit mantissa with guard/round/sticky bits and normalize to f128.
/// `mantissa` is the 113+ bit result, `binary_exp` is the unbiased exponent of bit 112.
/// `guard`, `round`, `sticky` are the IEEE rounding bits for round-to-nearest-even.
fn f128_round_and_encode(sign: bool, binary_exp: i32, mantissa: u128, guard: bool, round: bool, sticky: bool) -> [u8; 16] {
    let mut m = mantissa;
    let mut exp = binary_exp;

    // Round to nearest, ties to even
    let lsb = (m & 1) != 0;
    let round_up = guard && (round || sticky || lsb);

    if round_up {
        m = m.wrapping_add(1);
        // Check if rounding caused carry past bit 113 (overflow of 113-bit mantissa)
        if m & (1u128 << 113) != 0 {
            m >>= 1;
            exp += 1;
        }
    }

    encode_f128(sign, exp, m)
}

/// Add two IEEE 754 binary128 values with full 112-bit mantissa precision.
pub fn f128_add(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    f128_add_sub(a, b, false)
}

/// Subtract two IEEE 754 binary128 values with full 112-bit mantissa precision.
pub fn f128_sub(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    f128_add_sub(a, b, true)
}

/// Internal add/sub implementation.
fn f128_add_sub(a: &[u8; 16], b: &[u8; 16], subtract: bool) -> [u8; 16] {
    // Handle NaN
    if f128_is_nan(a) { return *a; }
    if f128_is_nan(b) { return *b; }

    let (a_sign, a_exp, a_mant) = f128_decompose(a);
    let (b_sign_orig, b_exp, b_mant) = f128_decompose(b);
    let b_sign = b_sign_orig ^ subtract; // flip sign of b for subtraction

    // Handle infinities
    if a_exp == 0x7FFF {
        if b_exp == 0x7FFF {
            if a_sign == b_sign { return *a; } // inf + inf = inf
            return make_f128_nan(false); // inf - inf = NaN
        }
        return *a;
    }
    if b_exp == 0x7FFF {
        if subtract {
            // Return infinity with flipped sign
            return make_f128_infinity(!b_sign_orig);
        }
        return *b;
    }

    // Handle zeros
    if f128_is_zero(a) && f128_is_zero(b) {
        // -0 + -0 = -0, otherwise +0
        return make_f128_zero(a_sign && b_sign);
    }
    if f128_is_zero(a) {
        if subtract {
            // 0 - b = -b
            let val = u128::from_le_bytes(*b);
            return (val ^ (1u128 << 127)).to_le_bytes();
        }
        return *b;
    }
    if f128_is_zero(b) { return *a; }

    // Get unbiased exponents. For subnormals, effective exponent is 1 (biased) - 16383 = -16382.
    let a_ue = if a_exp == 0 { -16382 } else { a_exp - 16383 };
    let b_ue = if b_exp == 0 { -16382 } else { b_exp - 16383 };

    // Align mantissas. We work with 3 extra bits (guard, round, sticky) for rounding.
    // Mantissas are 113 bits (bit 112 is MSB for normals). Shift left by 3 to make room.
    let mut a_m: u128 = a_mant << 3;
    let mut b_m: u128 = b_mant << 3;
    let mut exp_result = a_ue;
    let mut sticky = false;

    let exp_diff = a_ue - b_ue;
    if exp_diff > 0 {
        // Shift b right
        let shift = exp_diff as u32;
        if shift >= 128 {
            sticky = b_m != 0;
            b_m = 0;
        } else {
            sticky = (b_m & ((1u128 << shift) - 1)) != 0;
            b_m >>= shift;
        }
    } else if exp_diff < 0 {
        // Shift a right
        let shift = (-exp_diff) as u32;
        if shift >= 128 {
            sticky = a_m != 0;
            a_m = 0;
        } else {
            sticky = (a_m & ((1u128 << shift) - 1)) != 0;
            a_m >>= shift;
        }
        exp_result = b_ue;
    }

    // Add or subtract mantissas
    let (result_sign, result_m) = if a_sign == b_sign {
        // Same sign: add magnitudes
        let sum = a_m + b_m;
        (a_sign, sum)
    } else {
        // Different signs: subtract magnitudes
        if a_m > b_m {
            (a_sign, a_m - b_m)
        } else if b_m > a_m {
            (b_sign, b_m - a_m)
        } else {
            // Exact cancellation
            return make_f128_zero(false); // +0 for round-to-nearest
        }
    };

    if result_m == 0 {
        return make_f128_zero(false);
    }

    // Normalize: the result mantissa should have its MSB at bit 115 (= 112 + 3 guard bits).
    let mut m = result_m;
    let target_bit = 115; // bit 112 + 3 guard bits
    let msb = 127 - m.leading_zeros() as i32;

    if msb > target_bit {
        // Overflow: shift right, collect sticky bits
        let shift = (msb - target_bit) as u32;
        let lost = m & ((1u128 << shift) - 1);
        sticky = sticky || (lost != 0);
        m >>= shift;
        exp_result += shift as i32;
    } else if msb < target_bit {
        // Underflow: shift left
        let shift = (target_bit - msb) as u32;
        m <<= shift;
        exp_result -= shift as i32;
    }

    // Extract guard, round, sticky bits
    let guard_bit = (m & 4) != 0;
    let round_bit = (m & 2) != 0;
    let sticky_bit = sticky || (m & 1) != 0;
    m >>= 3; // Remove guard bits, now m is 113-bit mantissa

    f128_round_and_encode(result_sign, exp_result, m, guard_bit, round_bit, sticky_bit)
}

/// Multiply two IEEE 754 binary128 values with full 112-bit mantissa precision.
pub fn f128_mul(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    // Handle NaN
    if f128_is_nan(a) { return *a; }
    if f128_is_nan(b) { return *b; }

    let (a_sign, a_exp, a_mant) = f128_decompose(a);
    let (b_sign, b_exp, b_mant) = f128_decompose(b);
    let result_sign = a_sign ^ b_sign;

    // Handle infinities
    if a_exp == 0x7FFF || b_exp == 0x7FFF {
        if f128_is_zero(a) || f128_is_zero(b) {
            return make_f128_nan(false); // inf * 0 = NaN
        }
        return make_f128_infinity(result_sign);
    }

    // Handle zeros
    if f128_is_zero(a) || f128_is_zero(b) {
        return make_f128_zero(result_sign);
    }

    // Compute unbiased exponents
    let a_ue = if a_exp == 0 { -16382 } else { a_exp - 16383 };
    let b_ue = if b_exp == 0 { -16382 } else { b_exp - 16383 };

    // Multiply mantissas: 113 bits * 113 bits = up to 226 bits.
    // We need to use 256-bit multiplication (two u128 halves).
    let (prod_hi, prod_lo) = mul_u128(a_mant, b_mant);

    // The product has MSB at bit 224 or 225 (of the 226-bit result).
    // Exponent of the product = a_ue + b_ue (for the MSB of the product, adjusted).
    // Product is in the form: mantissa_a (bit 112 MSB) * mantissa_b (bit 112 MSB)
    // So the product MSB is at bit 224 or 225.
    let result_exp_base = a_ue + b_ue;

    // Normalize: we need a 113-bit mantissa (bit 112 is MSB).
    // The product MSB is at position 224 (= 112+112) or 225.
    // We need to shift right by (MSB_pos - 112) bits and collect sticky.
    let prod_bits = if prod_hi == 0 {
        if prod_lo == 0 { return make_f128_zero(result_sign); }
        127 - prod_lo.leading_zeros() as i32
    } else {
        128 + 127 - prod_hi.leading_zeros() as i32
    };

    // Number of bits to shift right to get mantissa MSB at bit 112
    let shift = prod_bits - 112;
    // The product of two mantissas with MSB at bit 112 has expected MSB at bit 224.
    // Only the excess above 224 adjusts the exponent (e.g. if MSB is at 225, add 1).
    let result_exp = result_exp_base + (prod_bits - 224);

    let (mantissa, guard, round, sticky) = if shift <= 0 {
        // Product fits in 113 bits - shift left
        let s = (-shift) as u32;
        let m = if prod_hi == 0 { prod_lo << s } else { (prod_hi << (s + 128)) | (prod_lo << s) };
        (m, false, false, false)
    } else {
        shift_right_256_with_grs(prod_hi, prod_lo, shift as u32)
    };

    f128_round_and_encode(result_sign, result_exp, mantissa, guard, round, sticky)
}

/// Divide two IEEE 754 binary128 values with full 112-bit mantissa precision.
pub fn f128_div(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    // Handle NaN
    if f128_is_nan(a) { return *a; }
    if f128_is_nan(b) { return *b; }

    let (a_sign, a_exp, a_mant) = f128_decompose(a);
    let (b_sign, b_exp, b_mant) = f128_decompose(b);
    let result_sign = a_sign ^ b_sign;

    // Handle infinities and zeros
    if a_exp == 0x7FFF {
        if b_exp == 0x7FFF { return make_f128_nan(false); } // inf / inf = NaN
        return make_f128_infinity(result_sign);
    }
    if b_exp == 0x7FFF {
        return make_f128_zero(result_sign); // x / inf = 0
    }
    if f128_is_zero(b) {
        if f128_is_zero(a) { return make_f128_nan(false); } // 0 / 0 = NaN
        return make_f128_infinity(result_sign); // x / 0 = inf
    }
    if f128_is_zero(a) {
        return make_f128_zero(result_sign);
    }

    // Unbiased exponents
    let a_ue = if a_exp == 0 { -16382 } else { a_exp - 16383 };
    let b_ue = if b_exp == 0 { -16382 } else { b_exp - 16383 };

    // Division: we need a_mant / b_mant with 113+ bits of precision in the quotient.
    // Both mantissas are 113-bit (bit 112 is MSB for normals).
    //
    // Strategy: compute (a_mant << 115) / b_mant to get a quotient with 115+ bits.
    // Since both mantissas have MSB at bit 112, their ratio is in [0.5, 2.0),
    // so the quotient MSB is at bit 114 or 115. We normalize to bit 115
    // (= 112 mantissa bits + 3 guard/round/sticky bits).
    //
    // We use 256-bit dividend: (a_mant << 115) is at most 228 bits.
    let shift = 115; // 112 (for mantissa) + 3 (for GRS)
    let (dividend_hi, dividend_lo) = shl_u128(a_mant, shift);

    // Divide 256-bit dividend by 128-bit divisor
    let (quot, rem) = div_256_by_128(dividend_hi, dividend_lo, b_mant);

    let result_exp = a_ue - b_ue;

    if quot == 0 {
        return make_f128_zero(result_sign);
    }

    // Normalize quotient: MSB should be at bit 115 (112 + 3 guard bits)
    let msb = 127 - quot.leading_zeros() as i32;
    let target = 115;
    let (m, exp_adjust, extra_sticky) = if msb > target {
        let s = (msb - target) as u32;
        let lost = quot & ((1u128 << s) - 1);
        (quot >> s, msb - target, lost != 0)
    } else if msb < target {
        let s = (target - msb) as u32;
        (quot << s, -(s as i32), false)
    } else {
        (quot, 0, false)
    };

    let guard = (m & 4) != 0;
    let round = (m & 2) != 0;
    let sticky = extra_sticky || (rem != 0) || (m & 1) != 0;
    let mantissa = m >> 3;

    f128_round_and_encode(result_sign, result_exp + exp_adjust, mantissa, guard, round, sticky)
}

/// Compute the remainder of two IEEE 754 binary128 values.
pub fn f128_rem(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    // rem(a, b) = a - trunc(a/b) * b
    // For simplicity, compute using the identity: rem = a - q * b where q = trunc(a/b)
    // Handle special cases
    if f128_is_nan(a) { return *a; }
    if f128_is_nan(b) { return *b; }
    if f128_is_inf(a) { return make_f128_nan(false); }
    if f128_is_zero(b) { return make_f128_nan(false); }
    if f128_is_zero(a) { return *a; }
    if f128_is_inf(b) { return *a; }

    // For now, fall back to f64 approximation for remainder
    // (remainder is less commonly used for long double constant folding)
    let fa = f128_bytes_to_f64(a);
    let fb = f128_bytes_to_f64(b);
    let result = fa % fb;
    f64_to_f128_bytes_lossless(result)
}

/// Compare two f128 values. Returns -1, 0, 1, or i32::MIN for unordered (NaN).
pub fn f128_cmp(a: &[u8; 16], b: &[u8; 16]) -> i32 {
    if f128_is_nan(a) || f128_is_nan(b) {
        return i32::MIN;
    }
    let a_val = u128::from_le_bytes(*a);
    let b_val = u128::from_le_bytes(*b);
    let a_sign = (a_val >> 127) != 0;
    let b_sign = (b_val >> 127) != 0;
    let a_mag = a_val & !(1u128 << 127);
    let b_mag = b_val & !(1u128 << 127);

    // Both zero (may differ in sign)
    if a_mag == 0 && b_mag == 0 {
        return 0;
    }

    // Different signs: negative < positive
    if a_sign != b_sign {
        return if a_sign { -1 } else { 1 };
    }

    // Same sign: compare magnitudes
    let cmp = if a_mag < b_mag { -1 } else if a_mag > b_mag { 1 } else { 0 };
    // If negative, reverse the comparison
    if a_sign { -cmp } else { cmp }
}

// --- Helper functions for wide arithmetic ---

/// Multiply two u128 values, returning (hi, lo) of the 256-bit result.
fn mul_u128(a: u128, b: u128) -> (u128, u128) {
    let a_lo = a as u64 as u128;
    let a_hi = (a >> 64) as u64 as u128;
    let b_lo = b as u64 as u128;
    let b_hi = (b >> 64) as u64 as u128;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + hl;
    let carry1 = if mid < lh { 1u128 } else { 0 };

    let lo = ll.wrapping_add(mid << 64);
    let carry2 = if lo < ll { 1u128 } else { 0 };
    let hi = hh + (mid >> 64) + (carry1 << 64) + carry2;

    (hi, lo)
}

/// Shift a u128 value left, returning (hi, lo) of the 256-bit result.
fn shl_u128(val: u128, shift: u32) -> (u128, u128) {
    if shift == 0 {
        (0, val)
    } else if shift < 128 {
        let hi = val >> (128 - shift);
        let lo = val << shift;
        (hi, lo)
    } else if shift < 256 {
        let hi = val << (shift - 128);
        (hi, 0)
    } else {
        (0, 0)
    }
}

/// Shift right a 256-bit value (hi:lo) by `shift` bits, extracting guard/round/sticky.
/// Returns (shifted_lo_128, guard, round, sticky).
fn shift_right_256_with_grs(hi: u128, lo: u128, shift: u32) -> (u128, bool, bool, bool) {
    if shift == 0 {
        return (lo, false, false, false);
    }

    // Reconstruct the bits we'll lose for GRS
    // We need to shift right by `shift` bits and extract the top bit lost (guard),
    // next bit (round), and OR of all remaining lost bits (sticky).

    // For very large shifts, everything becomes sticky
    if shift >= 256 {
        let sticky = hi != 0 || lo != 0;
        return (0, false, false, sticky);
    }

    // Compute the shifted value and the lost bits
    let result;
    let guard_bit;
    let round_bit;
    let sticky_bits;

    if shift < 128 {
        // Result comes from lo (shifted right) with bits from hi
        result = (lo >> shift) | (hi << (128 - shift));
        // Guard bit is bit (shift-1) of the original (hi:lo)
        guard_bit = if shift >= 1 { (lo >> (shift - 1)) & 1 != 0 } else { false };
        // Round bit is bit (shift-2)
        round_bit = if shift >= 2 { (lo >> (shift - 2)) & 1 != 0 } else { false };
        // Sticky: OR of all bits below round
        sticky_bits = if shift >= 3 { lo & ((1u128 << (shift - 2)) - 1) != 0 } else { false };
    } else if shift == 128 {
        result = hi;
        guard_bit = (lo >> 127) & 1 != 0;
        round_bit = (lo >> 126) & 1 != 0;
        sticky_bits = lo & ((1u128 << 126) - 1) != 0;
    } else {
        // shift > 128
        let s = shift - 128;
        result = hi >> s;
        if s == 0 {
            guard_bit = (lo >> 127) & 1 != 0;
            round_bit = (lo >> 126) & 1 != 0;
            sticky_bits = lo & ((1u128 << 126) - 1) != 0;
        } else {
            guard_bit = if s >= 1 { (hi >> (s - 1)) & 1 != 0 } else { false };
            round_bit = if s >= 2 { (hi >> (s - 2)) & 1 != 0 } else { lo >> 127 != 0 };
            let hi_sticky = if s >= 3 { hi & ((1u128 << (s - 2)) - 1) != 0 } else { false };
            sticky_bits = hi_sticky || lo != 0;
        }
    }

    (result, guard_bit, round_bit, sticky_bits)
}

/// Divide a 256-bit number (hi:lo) by a 128-bit divisor.
/// Returns (quotient, remainder) both as u128.
/// The quotient must fit in u128 (caller ensures this by appropriate shifting).
fn div_256_by_128(hi: u128, lo: u128, divisor: u128) -> (u128, u128) {
    if divisor == 0 {
        return (u128::MAX, 0); // Division by zero
    }
    if hi == 0 {
        return (lo / divisor, lo % divisor);
    }

    // Long division: divide (hi:lo) by divisor.
    // Process one bit at a time from the most significant bit.
    let mut remainder: u128 = 0;
    let mut quotient: u128 = 0;

    // Process high 128 bits
    for i in (0..128).rev() {
        remainder <<= 1;
        remainder |= (hi >> i) & 1;
        if remainder >= divisor {
            remainder -= divisor;
            // This contributes to upper bits of quotient (beyond 128), but we know
            // the quotient fits in 128 bits, so we just track the remainder.
        }
    }

    // Now process low 128 bits, building the actual quotient
    for i in (0..128).rev() {
        remainder <<= 1;
        remainder |= (lo >> i) & 1;
        if remainder >= divisor {
            remainder -= divisor;
            quotient |= 1u128 << i;
        }
    }

    (quotient, remainder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x87_to_f128() {
        // Test zero
        let x87 = [0u8; 16];
        let f128 = x87_bytes_to_f128_bytes(&x87);
        assert!(f128.iter().all(|&b| b == 0));

        // Test 1.0: x87 bytes should be exp=16383=0x3FFF, mantissa=1<<63
        let bytes_1 = f64_to_x87_bytes_simple(1.0);
        let f128_1 = x87_bytes_to_f128_bytes(&bytes_1);
        // f128 for 1.0: sign=0, exp=16383=0x3FFF, mantissa=0
        // Bytes (LE): 00..00 FF 3F
        assert_eq!(f128_1[15], 0x3F);
        assert_eq!(f128_1[14], 0xFF);
    }

    #[test]
    fn test_f64_to_x87_simple() {
        let bytes = f64_to_x87_bytes_simple(1.0);
        let f = x87_bytes_to_f64(&bytes);
        assert!((f - 1.0).abs() < 1e-15);

        let bytes = f64_to_x87_bytes_simple(-0.0);
        assert_eq!(bytes[9] & 0x80, 0x80);
    }
}

#[cfg(test)]
mod f128_tests {
    use super::*;

    #[test]
    fn test_f128_parse_integer() {
        // 9223372036854775807 = 2^63 - 1
        // In f128: exp=62+16383=16445=0x403D, mantissa has all lower bits set
        let bytes = parse_long_double_to_f128_bytes("9223372036854775807.0L");
        let val = u128::from_le_bytes(bytes);
        let exp = (val >> 112) & 0x7FFF;
        let mantissa = val & ((1u128 << 112) - 1);

        // Expected: exp = 0x403D (biased 62)
        assert_eq!(exp, 0x403D, "exponent for 2^63-1 should be 0x403D");
        // Mantissa should have high bits set (not all zeros)
        assert_ne!(mantissa, 0, "mantissa for 2^63-1 should not be zero");
    }

    #[test]
    fn test_f128_parse_pi() {
        let bytes = parse_long_double_to_f128_bytes("3.14159265358979323846264338327950288L");
        let val = u128::from_le_bytes(bytes);
        let exp = (val >> 112) & 0x7FFF;

        // pi: exp = 1 + 16383 = 16384 = 0x4000
        assert_eq!(exp, 0x4000, "exponent for pi should be 0x4000");
    }

    #[test]
    fn test_f128_parse_one() {
        let bytes = parse_long_double_to_f128_bytes("1.0L");
        let val = u128::from_le_bytes(bytes);
        let exp = (val >> 112) & 0x7FFF;
        let mantissa = val & ((1u128 << 112) - 1);

        assert_eq!(exp, 0x3FFF, "exponent for 1.0 should be 0x3FFF");
        assert_eq!(mantissa, 0, "mantissa for 1.0 should be zero");
    }

    #[test]
    fn test_f128_roundtrip_x87() {
        // Parse to f128, convert to x87, and back
        let f128 = parse_long_double_to_f128_bytes("1.0L");
        let x87 = f128_bytes_to_x87_bytes(&f128);
        let f128_back = x87_bytes_to_f128_bytes(&x87);
        assert_eq!(f128, f128_back, "roundtrip should preserve value");
    }
}

#[cfg(test)]
mod soft_mul_tests {
    use super::*;

    fn assert_mul_matches(a: &[u8; 16], b: &[u8; 16], label: &str) {
        let hw = x87_mul(a, b);
        let sw = x87_mul_soft(a, b);
        assert_eq!(
            hw, sw,
            "{}: x87_mul != x87_mul_soft\n  a = {:02x?}\n  b = {:02x?}\n  hw = {:02x?}\n  sw = {:02x?}",
            label, &a[..10], &b[..10], &hw[..10], &sw[..10]
        );
    }

    fn from_f64(v: f64) -> [u8; 16] {
        f64_to_x87_bytes_simple(v)
    }

    #[test]
    fn test_soft_mul_basic() {
        // 1 * 1 = 1
        assert_mul_matches(&from_f64(1.0), &from_f64(1.0), "1*1");
        // 2 * 3 = 6
        assert_mul_matches(&from_f64(2.0), &from_f64(3.0), "2*3");
        // -2 * 3 = -6
        assert_mul_matches(&from_f64(-2.0), &from_f64(3.0), "-2*3");
        // -2 * -3 = 6
        assert_mul_matches(&from_f64(-2.0), &from_f64(-3.0), "-2*-3");
    }

    #[test]
    fn test_soft_mul_zero() {
        assert_mul_matches(&from_f64(0.0), &from_f64(5.0), "0*5");
        assert_mul_matches(&from_f64(5.0), &from_f64(0.0), "5*0");
        assert_mul_matches(&from_f64(0.0), &from_f64(0.0), "0*0");
        assert_mul_matches(&from_f64(-0.0), &from_f64(1.0), "-0*1");
        assert_mul_matches(&from_f64(1.0), &from_f64(-0.0), "1*-0");
        assert_mul_matches(&from_f64(-0.0), &from_f64(-0.0), "-0*-0");
    }

    #[test]
    fn test_soft_mul_infinity() {
        let inf = make_x87_infinity(false);
        let neg_inf = make_x87_infinity(true);
        assert_mul_matches(&inf, &from_f64(2.0), "inf*2");
        assert_mul_matches(&from_f64(2.0), &inf, "2*inf");
        assert_mul_matches(&inf, &inf, "inf*inf");
        assert_mul_matches(&neg_inf, &from_f64(2.0), "-inf*2");
        assert_mul_matches(&neg_inf, &neg_inf, "-inf*-inf");
        assert_mul_matches(&inf, &neg_inf, "inf*-inf");
    }

    #[test]
    fn test_soft_mul_nan() {
        let nan = make_x87_nan(false);
        assert_mul_matches(&nan, &from_f64(2.0), "nan*2");
        assert_mul_matches(&from_f64(2.0), &nan, "2*nan");
        assert_mul_matches(&nan, &nan, "nan*nan");
    }

    #[test]
    fn test_soft_mul_inf_times_zero() {
        // inf * 0 = indefinite NaN on x87
        let inf = make_x87_infinity(false);
        let hw = x87_mul(&inf, &from_f64(0.0));
        let sw = x87_mul_soft(&inf, &from_f64(0.0));
        // x87 returns the "indefinite" QNaN: compare byte patterns
        assert_eq!(&hw[..10], &sw[..10],
            "inf*0: hw={:02x?} sw={:02x?}", &hw[..10], &sw[..10]);
    }

    #[test]
    fn test_soft_mul_powers_of_two() {
        for i in -20i32..=20 {
            for j in -20i32..=20 {
                let a = from_f64(2.0f64.powi(i));
                let b = from_f64(2.0f64.powi(j));
                assert_mul_matches(&a, &b, &format!("2^{i} * 2^{j}"));
            }
        }
    }

    #[test]
    fn test_soft_mul_various_values() {
        let values: Vec<f64> = vec![
            1.0, -1.0, 0.5, -0.5, 1.5, -1.5,
            3.14159265358980, -2.72,
            1e10, 1e-10, 1e20, 1e-20,
            1e100, 1e-100, 1e300, 1e-300,
            f64::MIN_POSITIVE, // smallest normal f64
            f64::MAX,
            1.0 + f64::EPSILON,
            0.1, 0.2, 0.3, 0.7, 0.9,
            123456789.0, 987654321.0,
            1.0 / 3.0, 1.0 / 7.0, 1.0 / 11.0,
        ];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                let xa = from_f64(a);
                let xb = from_f64(b);
                assert_mul_matches(&xa, &xb, &format!("values[{i}]*values[{j}] ({a}*{b})"));
            }
        }
    }

    #[test]
    fn test_soft_mul_near_overflow() {
        // Large exponents that are near the x87 overflow boundary
        let big = from_f64(f64::MAX); // ~1.8e308, exp in x87 ~= 1023+16383
        assert_mul_matches(&big, &from_f64(0.5), "MAX*0.5");
        assert_mul_matches(&big, &from_f64(1.0), "MAX*1.0");
        assert_mul_matches(&big, &from_f64(2.0), "MAX*2.0"); // overflow → inf
    }

    #[test]
    fn test_soft_mul_near_underflow() {
        let tiny = from_f64(f64::MIN_POSITIVE);
        assert_mul_matches(&tiny, &from_f64(1.0), "MIN_POS*1");
        assert_mul_matches(&tiny, &from_f64(0.5), "MIN_POS*0.5");
        assert_mul_matches(&tiny, &tiny, "MIN_POS*MIN_POS");
    }

    #[test]
    fn test_soft_mul_x87_denormals() {
        // x87 denormal: biased_exp=0, integer bit clear
        // Smallest x87 denormal: mantissa=1 → 2^(-16382-63) = 2^-16445
        let min_denorm = x87_encode(false, 0, 1);
        let max_denorm = x87_encode(false, 0, (1u64 << 63) - 1);
        let one = from_f64(1.0);
        let two = from_f64(2.0);
        let half = from_f64(0.5);

        assert_mul_matches(&min_denorm, &one, "min_denorm*1");
        assert_mul_matches(&min_denorm, &two, "min_denorm*2");
        assert_mul_matches(&min_denorm, &half, "min_denorm*0.5");
        assert_mul_matches(&max_denorm, &one, "max_denorm*1");
        assert_mul_matches(&max_denorm, &two, "max_denorm*2");
        assert_mul_matches(&max_denorm, &half, "max_denorm*0.5");
        assert_mul_matches(&min_denorm, &min_denorm, "min_denorm*min_denorm");
        assert_mul_matches(&max_denorm, &max_denorm, "max_denorm*max_denorm");
        assert_mul_matches(&min_denorm, &max_denorm, "min_denorm*max_denorm");
    }

    #[test]
    fn test_soft_mul_pseudo_denormal() {
        // Pseudo-denormal: biased_exp=0 but integer bit set.
        // x87 treats these as valid with effective exponent = -16382.
        let pseudo_denorm = x87_encode(false, 0, 1u64 << 63);
        let one = from_f64(1.0);
        let two = from_f64(2.0);
        assert_mul_matches(&pseudo_denorm, &one, "pseudo_denorm*1");
        assert_mul_matches(&pseudo_denorm, &two, "pseudo_denorm*2");
        assert_mul_matches(&pseudo_denorm, &pseudo_denorm, "pseudo_denorm*pseudo_denorm");
    }

    #[test]
    fn test_soft_mul_unnormal() {
        // Unnormal: biased_exp > 0 but integer bit clear → indefinite NaN
        let unnormal = x87_encode(false, 100, 0x7FFF_FFFF_FFFF_FFFF);
        let one = from_f64(1.0);
        assert_mul_matches(&unnormal, &one, "unnormal*1");
        assert_mul_matches(&one, &unnormal, "1*unnormal");
    }

    #[test]
    fn test_soft_mul_signaling_nan() {
        // Signaling NaN: exp=0x7FFF, integer bit set, quiet bit clear, fraction nonzero
        let snan = x87_encode(false, 0x7FFF, 0x8000_0000_0000_0001);
        let one = from_f64(1.0);
        assert_mul_matches(&snan, &one, "sNaN*1");
        assert_mul_matches(&one, &snan, "1*sNaN");
        // sNaN with payload
        let snan2 = x87_encode(false, 0x7FFF, 0x8000_0000_DEAD_BEEF);
        assert_mul_matches(&snan2, &one, "sNaN_payload*1");
    }

    #[test]
    fn test_soft_mul_max_normal() {
        // Largest x87 normal: exp=0x7FFE, mantissa=0xFFFF_FFFF_FFFF_FFFF
        let max_norm = x87_encode(false, 0x7FFE, u64::MAX);
        let one = from_f64(1.0);
        let two = from_f64(2.0);
        let half = from_f64(0.5);
        assert_mul_matches(&max_norm, &one, "max_norm*1");
        assert_mul_matches(&max_norm, &two, "max_norm*2"); // overflow
        assert_mul_matches(&max_norm, &half, "max_norm*0.5");
        assert_mul_matches(&max_norm, &max_norm, "max_norm*max_norm"); // overflow
    }

    #[test]
    fn test_soft_mul_min_normal() {
        // Smallest x87 normal: exp=1, mantissa=0x8000_0000_0000_0000
        let min_norm = x87_encode(false, 1, 1u64 << 63);
        let one = from_f64(1.0);
        let half = from_f64(0.5);
        assert_mul_matches(&min_norm, &one, "min_norm*1");
        assert_mul_matches(&min_norm, &half, "min_norm*0.5"); // → denormal
        assert_mul_matches(&min_norm, &min_norm, "min_norm*min_norm"); // deep underflow
    }

    #[test]
    fn test_soft_mul_rounding_carry() {
        // Values where rounding causes a carry that bumps the exponent.
        // mantissa=0xFFFF_FFFF_FFFF_FFFF with exp such that product rounds up
        let almost_two = x87_encode(false, 16383, u64::MAX); // just below 2.0
        let just_above_one = x87_encode(false, 16383, (1u64 << 63) | 1); // just above 1.0
        assert_mul_matches(&almost_two, &just_above_one, "almost_2 * just_above_1");
        assert_mul_matches(&almost_two, &almost_two, "almost_2 * almost_2");
    }

    #[test]
    fn test_soft_mul_negative_combos() {
        // Verify sign handling with various special values
        let neg_inf = make_x87_infinity(true);
        let pos_inf = make_x87_infinity(false);
        let neg_one = from_f64(-1.0);
        let neg_zero = from_f64(-0.0);
        let pos_zero = from_f64(0.0);
        assert_mul_matches(&neg_inf, &neg_inf, "-inf*-inf");
        assert_mul_matches(&neg_inf, &neg_one, "-inf*-1");
        assert_mul_matches(&pos_inf, &neg_one, "inf*-1");
        assert_mul_matches(&neg_zero, &neg_zero, "-0*-0");
        assert_mul_matches(&neg_zero, &pos_zero, "-0*0");
        assert_mul_matches(&neg_one, &neg_zero, "-1*-0");
        // inf * -0 = NaN
        assert_mul_matches(&pos_inf, &neg_zero, "inf*-0");
        assert_mul_matches(&neg_inf, &pos_zero, "-inf*0");
        assert_mul_matches(&neg_inf, &neg_zero, "-inf*-0");
    }

    #[test]
    fn test_soft_mul_exact_halfway_denormal() {
        // Construct a case at the exact denormal boundary where rounding matters.
        // min_normal * 0.5 should give the largest denormal (exact, no rounding).
        let min_norm = x87_encode(false, 1, 1u64 << 63);
        let half = from_f64(0.5);
        assert_mul_matches(&min_norm, &half, "min_norm*0.5_exact");

        // min_normal * value_just_below_0.5 should give a denormal
        let just_below_half = x87_encode(false, 16382, u64::MAX); // largest value < 1.0
        assert_mul_matches(&min_norm, &just_below_half, "min_norm*just_below_half");
    }

    #[test]
    fn test_soft_mul_pseudo_inf_and_nan() {
        // Pseudo-infinity: exp=0x7FFF, integer bit clear, fraction=0
        let pseudo_inf = x87_encode(false, 0x7FFF, 0);
        let one = from_f64(1.0);
        assert_mul_matches(&pseudo_inf, &one, "pseudo_inf*1");

        // Pseudo-NaN: exp=0x7FFF, integer bit clear, fraction nonzero
        let pseudo_nan = x87_encode(false, 0x7FFF, 1);
        assert_mul_matches(&pseudo_nan, &one, "pseudo_nan*1");

        // Pseudo-NaN with more fraction bits
        let pseudo_nan2 = x87_encode(false, 0x7FFF, 0x7FFF_FFFF_FFFF_FFFF);
        assert_mul_matches(&pseudo_nan2, &one, "pseudo_nan2*1");
    }
}

#[cfg(test)]
mod soft_addsub_tests {
    use super::*;

    fn assert_add_matches(a: &[u8; 16], b: &[u8; 16], label: &str) {
        let hw = x87_add(a, b);
        let sw = x87_add_soft(a, b);
        assert_eq!(hw, sw, "{label}: add mismatch\n  a={:02x?}\n  b={:02x?}\n  hw={:02x?}\n  sw={:02x?}",
            &a[..10], &b[..10], &hw[..10], &sw[..10]);
    }

    fn assert_sub_matches(a: &[u8; 16], b: &[u8; 16], label: &str) {
        let hw = x87_sub(a, b);
        let sw = x87_sub_soft(a, b);
        assert_eq!(hw, sw, "{label}: sub mismatch\n  a={:02x?}\n  b={:02x?}\n  hw={:02x?}\n  sw={:02x?}",
            &a[..10], &b[..10], &hw[..10], &sw[..10]);
    }

    fn from_f64(v: f64) -> [u8; 16] { f64_to_x87_bytes_simple(v) }

    #[test]
    fn test_soft_add_basic() {
        assert_add_matches(&from_f64(1.0), &from_f64(2.0), "1+2");
        assert_add_matches(&from_f64(-1.0), &from_f64(2.0), "-1+2");
        assert_add_matches(&from_f64(1.0), &from_f64(-2.0), "1+-2");
        assert_add_matches(&from_f64(-1.0), &from_f64(-2.0), "-1+-2");
    }

    #[test]
    fn test_soft_sub_basic() {
        assert_sub_matches(&from_f64(3.0), &from_f64(1.0), "3-1");
        assert_sub_matches(&from_f64(1.0), &from_f64(3.0), "1-3");
        assert_sub_matches(&from_f64(-1.0), &from_f64(-3.0), "-1--3");
    }

    #[test]
    fn test_soft_addsub_zero() {
        assert_add_matches(&from_f64(0.0), &from_f64(0.0), "0+0");
        assert_add_matches(&from_f64(-0.0), &from_f64(-0.0), "-0+-0");
        assert_add_matches(&from_f64(0.0), &from_f64(-0.0), "0+-0");
        assert_add_matches(&from_f64(-0.0), &from_f64(0.0), "-0+0");
        assert_add_matches(&from_f64(0.0), &from_f64(5.0), "0+5");
        assert_add_matches(&from_f64(5.0), &from_f64(0.0), "5+0");
        assert_sub_matches(&from_f64(1.0), &from_f64(1.0), "1-1");
        assert_sub_matches(&from_f64(-1.0), &from_f64(-1.0), "-1--1");
    }

    #[test]
    fn test_soft_addsub_inf() {
        let inf = make_x87_infinity(false);
        let neg_inf = make_x87_infinity(true);
        assert_add_matches(&inf, &from_f64(1.0), "inf+1");
        assert_add_matches(&from_f64(1.0), &inf, "1+inf");
        assert_add_matches(&inf, &inf, "inf+inf");
        assert_add_matches(&neg_inf, &neg_inf, "-inf+-inf");
        // inf + -inf = NaN
        assert_add_matches(&inf, &neg_inf, "inf+-inf");
        assert_sub_matches(&inf, &inf, "inf-inf");
    }

    #[test]
    fn test_soft_addsub_various() {
        let values: Vec<f64> = vec![
            1.0, -1.0, 0.5, -0.5, 1.5, -1.5,
            3.14159265358980, -2.72,
            1e10, 1e-10, 1e20, 1e-20,
            1e100, 1e-100, 1e300, 1e-300,
            f64::MIN_POSITIVE, f64::MAX,
            0.1, 0.2, 0.3, 0.7, 0.9,
            1.0 / 3.0, 1.0 / 7.0,
        ];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                let xa = from_f64(a);
                let xb = from_f64(b);
                assert_add_matches(&xa, &xb, &format!("add[{i}][{j}]"));
                assert_sub_matches(&xa, &xb, &format!("sub[{i}][{j}]"));
            }
        }
    }

}

#[cfg(test)]
mod soft_div_tests {
    use super::*;

    fn assert_div_matches(a: &[u8; 16], b: &[u8; 16], label: &str) {
        let hw = x87_div(a, b);
        let sw = x87_div_soft(a, b);
        assert_eq!(hw, sw, "{label}: div mismatch\n  a={:02x?}\n  b={:02x?}\n  hw={:02x?}\n  sw={:02x?}",
            &a[..10], &b[..10], &hw[..10], &sw[..10]);
    }

    fn from_f64(v: f64) -> [u8; 16] { f64_to_x87_bytes_simple(v) }

    #[test]
    fn test_soft_div_basic() {
        assert_div_matches(&from_f64(6.0), &from_f64(2.0), "6/2");
        assert_div_matches(&from_f64(1.0), &from_f64(3.0), "1/3");
        assert_div_matches(&from_f64(-6.0), &from_f64(2.0), "-6/2");
        assert_div_matches(&from_f64(6.0), &from_f64(-2.0), "6/-2");
        assert_div_matches(&from_f64(-6.0), &from_f64(-2.0), "-6/-2");
    }

    #[test]
    fn test_soft_div_special() {
        let inf = make_x87_infinity(false);
        let zero = from_f64(0.0);
        let one = from_f64(1.0);
        assert_div_matches(&inf, &one, "inf/1");
        assert_div_matches(&one, &inf, "1/inf");
        assert_div_matches(&inf, &inf, "inf/inf");
        assert_div_matches(&zero, &zero, "0/0");
        assert_div_matches(&zero, &one, "0/1");
        assert_div_matches(&one, &zero, "1/0");
    }

    #[test]
    fn test_soft_div_various() {
        let values: Vec<f64> = vec![
            1.0, -1.0, 0.5, -0.5, 2.0, 3.0, 7.0, 10.0,
            3.14159265358980, 2.72,
            1e10, 1e-10, 1e100, 1e-100,
            f64::MIN_POSITIVE, f64::MAX,
            0.1, 0.3, 1.0 / 3.0, 1.0 / 7.0,
        ];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                assert_div_matches(&from_f64(a), &from_f64(b), &format!("div[{i}][{j}]"));
            }
        }
    }

}

#[cfg(test)]
mod soft_rem_tests {
    use super::*;

    fn assert_rem_matches(a: &[u8; 16], b: &[u8; 16], label: &str) {
        let hw = x87_rem(a, b);
        let sw = x87_rem_soft(a, b);
        assert_eq!(hw, sw, "{label}: rem mismatch\n  a={:02x?}\n  b={:02x?}\n  hw={:02x?}\n  sw={:02x?}",
            &a[..10], &b[..10], &hw[..10], &sw[..10]);
    }

    fn from_f64(v: f64) -> [u8; 16] { f64_to_x87_bytes_simple(v) }

    #[test]
    fn test_soft_rem_basic() {
        assert_rem_matches(&from_f64(7.0), &from_f64(3.0), "7%3");
        assert_rem_matches(&from_f64(10.0), &from_f64(3.0), "10%3");
        assert_rem_matches(&from_f64(-7.0), &from_f64(3.0), "-7%3");
        assert_rem_matches(&from_f64(7.0), &from_f64(-3.0), "7%-3");
        assert_rem_matches(&from_f64(1.5), &from_f64(1.0), "1.5%1");
    }

    #[test]
    fn test_soft_rem_special() {
        let inf = make_x87_infinity(false);
        let zero = from_f64(0.0);
        let one = from_f64(1.0);
        assert_rem_matches(&inf, &one, "inf%1");
        assert_rem_matches(&one, &inf, "1%inf");
        assert_rem_matches(&zero, &one, "0%1");
        assert_rem_matches(&one, &zero, "1%0");
    }

    #[test]
    fn test_soft_rem_various() {
        let values: Vec<f64> = vec![
            1.0, -1.0, 0.5, 2.0, 3.0, 7.0, 10.0, 100.0,
            3.14159265358980, 2.72,
            0.1, 0.3, 1.0 / 3.0,
        ];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                assert_rem_matches(&from_f64(a), &from_f64(b), &format!("rem[{i}][{j}]"));
            }
        }
    }
}

#[cfg(test)]
mod soft_cmp_tests {
    use super::*;

    fn assert_cmp_matches(a: &[u8; 16], b: &[u8; 16], label: &str) {
        let hw = x87_cmp(a, b);
        let sw = x87_cmp_soft(a, b);
        assert_eq!(hw, sw, "{label}: cmp mismatch (hw={hw}, sw={sw})\n  a={:02x?}\n  b={:02x?}",
            &a[..10], &b[..10]);
    }

    fn from_f64(v: f64) -> [u8; 16] { f64_to_x87_bytes_simple(v) }

    #[test]
    fn test_soft_cmp_basic() {
        assert_cmp_matches(&from_f64(1.0), &from_f64(2.0), "1<2");
        assert_cmp_matches(&from_f64(2.0), &from_f64(1.0), "2>1");
        assert_cmp_matches(&from_f64(1.0), &from_f64(1.0), "1==1");
        assert_cmp_matches(&from_f64(-1.0), &from_f64(1.0), "-1<1");
        assert_cmp_matches(&from_f64(1.0), &from_f64(-1.0), "1>-1");
        assert_cmp_matches(&from_f64(-2.0), &from_f64(-1.0), "-2<-1");
    }

    #[test]
    fn test_soft_cmp_special() {
        let nan = make_x87_nan(false);
        let inf = make_x87_infinity(false);
        let neg_inf = make_x87_infinity(true);
        let zero = from_f64(0.0);
        let neg_zero = from_f64(-0.0);
        assert_cmp_matches(&nan, &from_f64(1.0), "nan<>1");
        assert_cmp_matches(&from_f64(1.0), &nan, "1<>nan");
        assert_cmp_matches(&inf, &from_f64(1.0), "inf>1");
        assert_cmp_matches(&neg_inf, &from_f64(1.0), "-inf<1");
        assert_cmp_matches(&inf, &neg_inf, "inf>-inf");
        assert_cmp_matches(&zero, &neg_zero, "0==-0");
        assert_cmp_matches(&neg_zero, &zero, "-0==0");
    }

    #[test]
    fn test_soft_cmp_various() {
        let values: Vec<f64> = vec![
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
            f64::MIN_POSITIVE, -f64::MIN_POSITIVE,
            f64::MAX, -f64::MAX,
            1e100, -1e100, 1e-100, -1e-100,
        ];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                assert_cmp_matches(&from_f64(a), &from_f64(b), &format!("cmp[{i}][{j}]"));
            }
        }
    }
}

// ======== Known-value tests (platform-independent) ========

#[cfg(test)]
mod known_value_tests {
    use super::*;

    fn from_f64(v: f64) -> [u8; 16] { f64_to_x87_bytes_simple(v) }
    fn enc(sign: bool, exp: u16, mant: u64) -> [u8; 16] { x87_encode(sign, exp, mant) }
    fn one() -> [u8; 16] { from_f64(1.0) }

    // ---- Multiply known values ----

    #[test]
    fn test_mul_identity() {
        for &v in &[1.0, -1.0, 2.0, 0.5, 3.14160, 1e100, 1e-100] {
            let x = from_f64(v);
            assert_eq!(x87_mul(&one(), &x), x, "1*{v}");
            assert_eq!(x87_mul(&x, &one()), x, "{v}*1");
        }
    }

    #[test]
    fn test_mul_known_exact() {
        assert_eq!(x87_mul(&from_f64(2.0), &from_f64(3.0)), from_f64(6.0));
        assert_eq!(x87_mul(&from_f64(0.5), &from_f64(4.0)), from_f64(2.0));
        assert_eq!(x87_mul(&from_f64(-2.0), &from_f64(-3.0)), from_f64(6.0));
        assert_eq!(x87_mul(&from_f64(-2.0), &from_f64(3.0)), from_f64(-6.0));
    }

    #[test]
    fn test_mul_zero_sign() {
        let pz = from_f64(0.0);
        let nz = from_f64(-0.0);
        assert_eq!(x87_mul(&pz, &from_f64(5.0)), pz);
        assert_eq!(x87_mul(&nz, &from_f64(5.0)), nz);
        assert_eq!(x87_mul(&nz, &nz), pz, "-0*-0=+0");
        assert_eq!(x87_mul(&pz, &nz), nz, "+0*-0=-0");
    }

    #[test]
    fn test_mul_inf_cases() {
        let pi = make_x87_infinity(false);
        let ni = make_x87_infinity(true);
        assert_eq!(x87_mul(&pi, &from_f64(2.0)), pi);
        assert_eq!(x87_mul(&ni, &ni), pi, "-inf*-inf=+inf");
        assert_eq!(x87_mul(&pi, &ni), ni, "inf*-inf=-inf");
        // inf * 0 = NaN
        let r = x87_mul(&pi, &from_f64(0.0));
        assert_eq!(x87_decompose(&r).biased_exp, 0x7FFF);
    }

    #[test]
    fn test_mul_nan_propagation() {
        let nan = make_x87_nan(false);
        let r = x87_mul(&nan, &from_f64(2.0));
        let d = x87_decompose(&r);
        assert_eq!(d.biased_exp, 0x7FFF);
        assert!(d.mantissa & (1u64 << 62) != 0, "should be quiet NaN");
    }

    #[test]
    fn test_mul_overflow_to_inf() {
        let max_norm = enc(false, 0x7FFE, u64::MAX);
        assert_eq!(x87_mul(&max_norm, &from_f64(2.0)), make_x87_infinity(false));
    }

    #[test]
    fn test_mul_underflow_to_denormal() {
        let min_norm = enc(false, 1, 1u64 << 63);
        let d = x87_decompose(&x87_mul(&min_norm, &from_f64(0.5)));
        assert_eq!(d.biased_exp, 0, "min_norm*0.5 should be denormal");
    }

    #[test]
    fn test_mul_powers_of_two() {
        for i in -20i32..=20 {
            for j in -20i32..=20 {
                let a = from_f64(2.0f64.powi(i));
                let b = from_f64(2.0f64.powi(j));
                assert_eq!(x87_mul(&a, &b), from_f64(2.0f64.powi(i + j)), "2^{i}*2^{j}");
            }
        }
    }

    #[test]
    fn test_mul_commutativity() {
        let vals = [from_f64(3.15), from_f64(-2.719), from_f64(1e50), from_f64(1e-50)];
        for a in &vals { for b in &vals {
            assert_eq!(x87_mul(a, b), x87_mul(b, a), "mul commutativity");
        }}
    }

    #[test]
    fn test_mul_denormal_identity() {
        let min_d = enc(false, 0, 1);
        assert_eq!(x87_mul(&min_d, &one()), min_d, "min_denorm*1");
        assert_eq!(x87_mul(&min_d, &from_f64(2.0)), enc(false, 0, 2), "min_denorm*2");
    }

    #[test]
    fn test_mul_unnormal_is_nan() {
        let unnormal = enc(false, 100, 0x7FFF_FFFF_FFFF_FFFF);
        let d = x87_decompose(&x87_mul(&unnormal, &one()));
        assert_eq!(d.biased_exp, 0x7FFF);
    }

    // ---- Add/Sub known values ----

    #[test]
    fn test_add_known_exact() {
        assert_eq!(x87_add(&from_f64(1.0), &from_f64(2.0)), from_f64(3.0));
        assert_eq!(x87_add(&from_f64(0.5), &from_f64(0.5)), from_f64(1.0));
        assert_eq!(x87_add(&from_f64(-1.0), &from_f64(1.0)), from_f64(0.0));
        assert_eq!(x87_add(&from_f64(1.0), &from_f64(1.0)), from_f64(2.0));
    }

    #[test]
    fn test_sub_known_exact() {
        assert_eq!(x87_sub(&from_f64(3.0), &from_f64(1.0)), from_f64(2.0));
        assert_eq!(x87_sub(&from_f64(1.0), &from_f64(1.0)), from_f64(0.0));
        assert_eq!(x87_sub(&from_f64(1.0), &from_f64(3.0)), from_f64(-2.0));
        assert_eq!(x87_sub(&from_f64(1.5), &from_f64(1.0)), from_f64(0.5));
    }

    #[test]
    fn test_add_zero_identity() {
        for &v in &[1.0, -1.0, 0.5, 3.15, 1e100, 1e-100] {
            let x = from_f64(v);
            let z = from_f64(0.0);
            assert_eq!(x87_add(&x, &z), x, "{v}+0");
            assert_eq!(x87_add(&z, &x), x, "0+{v}");
        }
    }

    #[test]
    fn test_sub_self_is_zero() {
        for &v in &[1.0, -1.0, 0.5, 1e100, 1e-100, 3.15] {
            let x = from_f64(v);
            assert_eq!(x87_sub(&x, &x), from_f64(0.0), "{v}-{v}");
        }
    }

    #[test]
    fn test_add_commutativity() {
        let vals = [1.0, -2.5, 0.5, 1e100, 1e-100, 3.15];
        for &a in &vals { for &b in &vals {
            assert_eq!(x87_add(&from_f64(a), &from_f64(b)), x87_add(&from_f64(b), &from_f64(a)),
                "{a}+{b} commutativity");
        }}
    }

    #[test]
    fn test_add_neg_is_sub() {
        let vals = [1.0, -2.5, 0.5, 100.0, 3.15];
        for &a in &vals { for &b in &vals {
            let xa = from_f64(a); let xb = from_f64(b);
            assert_eq!(x87_add(&xa, &x87_neg(&xb)), x87_sub(&xa, &xb), "a+(-b)==a-b for {a},{b}");
        }}
    }

    #[test]
    fn test_add_carry_1_plus_1() {
        assert_eq!(x87_add(&from_f64(1.0), &from_f64(1.0)), from_f64(2.0));
    }

    #[test]
    fn test_add_max_overflow() {
        let big = enc(false, 0x7FFE, u64::MAX);
        assert_eq!(x87_add(&big, &big), make_x87_infinity(false), "max+max=inf");
    }

    #[test]
    fn test_add_large_exp_diff() {
        let big = from_f64(1e100);
        let tiny = from_f64(1e-100);
        assert_eq!(x87_add(&big, &tiny), big, "1e100+1e-100≈1e100");
    }

    #[test]
    fn test_add_zero_signs() {
        let pz = from_f64(0.0);
        let nz = from_f64(-0.0);
        assert_eq!(x87_add(&pz, &pz), pz, "+0+0=+0");
        assert_eq!(x87_add(&nz, &nz), nz, "-0+-0=-0");
        assert_eq!(x87_add(&pz, &nz), pz, "+0+-0=+0");
        assert_eq!(x87_add(&nz, &pz), pz, "-0++0=+0");
    }

    #[test]
    fn test_addsub_inf_nan() {
        let pi = make_x87_infinity(false);
        let ni = make_x87_infinity(true);
        assert_eq!(x87_add(&pi, &from_f64(1.0)), pi);
        assert_eq!(x87_add(&ni, &ni), ni);
        let r = x87_sub(&pi, &pi);
        assert_eq!(x87_decompose(&r).biased_exp, 0x7FFF, "inf-inf=NaN");
    }

    #[test]
    fn test_add_denormals() {
        let d1 = enc(false, 0, 100);
        let d2 = enc(false, 0, 200);
        assert_eq!(x87_add(&d1, &d2), enc(false, 0, 300), "denorm+denorm");
        assert_eq!(x87_sub(&d1, &d1), from_f64(0.0), "denorm-denorm=0");
    }

    // ---- Division known values ----

    #[test]
    fn test_div_known_exact() {
        assert_eq!(x87_div(&from_f64(6.0), &from_f64(2.0)), from_f64(3.0));
        assert_eq!(x87_div(&from_f64(10.0), &from_f64(5.0)), from_f64(2.0));
        assert_eq!(x87_div(&from_f64(-6.0), &from_f64(-2.0)), from_f64(3.0));
        assert_eq!(x87_div(&from_f64(6.0), &from_f64(-2.0)), from_f64(-3.0));
    }

    #[test]
    fn test_div_by_one() {
        for &v in &[1.0, -1.0, 2.0, 0.5, 3.15, 1e100, 1e-100] {
            let x = from_f64(v);
            assert_eq!(x87_div(&x, &one()), x, "{v}/1");
        }
    }

    #[test]
    fn test_div_self_is_one() {
        for &v in &[1.0, -1.0, 2.0, 0.5, 3.15, 1e100, 1e-100] {
            assert_eq!(x87_div(&from_f64(v), &from_f64(v)), one(), "{v}/{v}=1");
        }
    }

    #[test]
    fn test_div_zero_inf_nan() {
        let pz = from_f64(0.0);
        let nz = from_f64(-0.0);
        let pi = make_x87_infinity(false);
        assert_eq!(x87_div(&pz, &one()), pz, "0/1=0");
        assert_eq!(x87_div(&one(), &pz), pi, "1/0=inf");
        assert_eq!(x87_div(&from_f64(-1.0), &pz), make_x87_infinity(true), "-1/0=-inf");
        assert_eq!(x87_div(&nz, &one()), nz, "-0/1=-0");
        assert_eq!(x87_decompose(&x87_div(&pz, &pz)).biased_exp, 0x7FFF, "0/0=NaN");
        assert_eq!(x87_decompose(&x87_div(&pi, &pi)).biased_exp, 0x7FFF, "inf/inf=NaN");
        assert_eq!(x87_div(&one(), &pi), pz, "1/inf=0");
    }

    #[test]
    fn test_div_powers_of_two() {
        for i in -20i32..=20 {
            for j in 1i32..=20 {
                let a = from_f64(2.0f64.powi(i));
                let b = from_f64(2.0f64.powi(j));
                assert_eq!(x87_div(&a, &b), from_f64(2.0f64.powi(i - j)), "2^{i}/2^{j}");
            }
        }
    }

    #[test]
    fn test_div_rounding_1_3() {
        let third = x87_div(&one(), &from_f64(3.0));
        let back = x87_mul(&third, &from_f64(3.0));
        // Due to rounding, 1/3*3 might not be exactly 1.0 but should be very close
        let d = x87_decompose(&back);
        assert_eq!(d.biased_exp, 16383, "1/3*3 exponent should be 1.0's exponent");
    }

    // ---- Remainder known values ----

    #[test]
    fn test_rem_known_exact() {
        assert_eq!(x87_rem(&from_f64(7.0), &from_f64(3.0)), from_f64(1.0));
        assert_eq!(x87_rem(&from_f64(10.0), &from_f64(3.0)), from_f64(1.0));
        assert_eq!(x87_rem(&from_f64(1.5), &from_f64(1.0)), from_f64(0.5));
        assert_eq!(x87_rem(&from_f64(5.0), &from_f64(2.0)), from_f64(1.0));
        assert_eq!(x87_rem(&from_f64(4.0), &from_f64(2.0)), from_f64(0.0));
    }

    #[test]
    fn test_rem_sign_is_dividend() {
        assert_eq!(x87_rem(&from_f64(-7.0), &from_f64(3.0)), from_f64(-1.0));
        assert_eq!(x87_rem(&from_f64(7.0), &from_f64(-3.0)), from_f64(1.0));
        assert_eq!(x87_rem(&from_f64(-7.0), &from_f64(-3.0)), from_f64(-1.0));
    }

    #[test]
    fn test_rem_less_than_divisor() {
        assert_eq!(x87_rem(&from_f64(1.0), &from_f64(3.0)), from_f64(1.0));
        assert_eq!(x87_rem(&from_f64(0.5), &from_f64(1.0)), from_f64(0.5));
    }

    #[test]
    fn test_rem_large_quotient() {
        // 10^20 mod 3 = 1 (since 10 ≡ 1 mod 3)
        assert_eq!(x87_rem(&from_f64(1e20), &from_f64(3.0)), from_f64(1.0));
        // 10^20 mod 1 = 0
        assert_eq!(x87_rem(&from_f64(1e20), &from_f64(1.0)), from_f64(0.0));
    }

    #[test]
    fn test_rem_consistency() {
        let pairs = [(7.0, 3.0), (10.0, 3.0), (100.0, 7.0), (256.0, 10.0)];
        for (a, b) in pairs {
            assert_eq!(x87_rem(&from_f64(a), &from_f64(b)), from_f64(a % b), "{a}%{b}");
        }
    }

    // ---- Compare known values ----

    #[test]
    fn test_cmp_ordering() {
        assert_eq!(x87_cmp(&from_f64(1.0), &from_f64(2.0)), -1);
        assert_eq!(x87_cmp(&from_f64(2.0), &from_f64(1.0)), 1);
        assert_eq!(x87_cmp(&from_f64(1.0), &from_f64(1.0)), 0);
        assert_eq!(x87_cmp(&from_f64(-2.0), &from_f64(-1.0)), -1);
        assert_eq!(x87_cmp(&from_f64(-1.0), &from_f64(-2.0)), 1);
    }

    #[test]
    fn test_cmp_zero_equality() {
        assert_eq!(x87_cmp(&from_f64(0.0), &from_f64(-0.0)), 0);
        assert_eq!(x87_cmp(&from_f64(-0.0), &from_f64(0.0)), 0);
    }

    #[test]
    fn test_cmp_nan() {
        let nan = make_x87_nan(false);
        assert_eq!(x87_cmp(&nan, &from_f64(1.0)), i32::MIN);
        assert_eq!(x87_cmp(&from_f64(1.0), &nan), i32::MIN);
        assert_eq!(x87_cmp(&nan, &nan), i32::MIN);
    }

    #[test]
    fn test_cmp_inf() {
        let pi = make_x87_infinity(false);
        let ni = make_x87_infinity(true);
        assert_eq!(x87_cmp(&pi, &from_f64(1e300)), 1);
        assert_eq!(x87_cmp(&ni, &from_f64(-1e300)), -1);
        assert_eq!(x87_cmp(&pi, &ni), 1);
        assert_eq!(x87_cmp(&pi, &pi), 0);
    }

    #[test]
    fn test_cmp_denormals() {
        let d1 = enc(false, 0, 100);
        let d2 = enc(false, 0, 200);
        assert_eq!(x87_cmp(&d1, &d2), -1);
        assert_eq!(x87_cmp(&d2, &d1), 1);
        assert_eq!(x87_cmp(&d1, &d1), 0);
    }

    #[test]
    fn test_cmp_antisymmetry() {
        let vals = [1.0, -1.0, 0.5, 100.0, 1e-100, 0.0];
        for &a in &vals { for &b in &vals {
            let r1 = x87_cmp(&from_f64(a), &from_f64(b));
            let r2 = x87_cmp(&from_f64(b), &from_f64(a));
            assert_eq!(r1, -r2, "antisymmetry cmp({a},{b})");
        }}
    }

    #[test]
    fn test_ldbl_min_parse() {
        // LDBL_MIN = 2^(-16382), the smallest normal long double.
        // GCC full-precision string:
        let f128_bytes = parse_long_double_to_f128_bytes("3.36210314311209350626267781732175260e-4932");
        let (sign, biased_exp, _mantissa) = f128_decompose(&f128_bytes);
        assert!(!sign);
        assert_eq!(biased_exp, 1, "LDBL_MIN should be normal (biased_exp=1), got {}", biased_exp);

        // Verify x87 conversion gives correct result
        let x87_bytes = f128_bytes_to_x87_bytes(&f128_bytes);
        let x87_exp = (x87_bytes[8] as u16) | ((x87_bytes[9] as u16) << 8);
        assert_eq!(x87_exp & 0x7fff, 1, "x87 LDBL_MIN biased_exp should be 1");
        let mantissa64 = u64::from_le_bytes(x87_bytes[0..8].try_into().unwrap());
        assert_eq!(mantissa64, 0x8000000000000000u64, "x87 LDBL_MIN mantissa should be just integer bit");

        // Also verify that the shorter (truncated) string rounds correctly
        let f128_short = parse_long_double_to_f128_bytes("3.36210314311209350626e-4932");
        let (sign2, biased_exp2, _) = f128_decompose(&f128_short);
        assert!(!sign2);
        // The truncated value is slightly below 2^(-16382), so it's the largest
        // subnormal in f128 - this is expected behavior for truncated decimal input
        assert_eq!(biased_exp2, 0, "truncated LDBL_MIN string is subnormal in f128 (expected)");
    }
}

#[cfg(test)]
mod hex_float_tests {
    use super::*;

    fn f128_hex(text: &str) -> u128 {
        u128::from_le_bytes(parse_long_double_to_f128_bytes(text))
    }

    #[test]
    fn test_hex_f128_parse_mantissa_wider_than_128_bits() {
        // 34 digits of ones scaled back to 1.0; the exponent depends on every digit.
        assert_eq!(f128_hex("0xffffffffffffffffffffffffffffffffffp-136L"), 0x3fff0000000000000000000000000000);
        assert_eq!(f128_hex("0x10000000000000000000000000000000000p0L"), 0x40870000000000000000000000000000);
    }

    #[test]
    fn test_hex_f128_parse_rounds_to_nearest_even() {
        assert_eq!(f128_hex("0x1.0000000000000000000000000000ffp0L"), 0x3fff0000000000000000000000000001);
        assert_eq!(f128_hex("0x1.00000000000000000000000000008p0L"), 0x3fff0000000000000000000000000000);
        assert_eq!(f128_hex("0x1.00000000000000000000000000018p0L"), 0x3fff0000000000000000000000000002);
    }

    #[test]
    fn test_hex_f128_parse_subnormal_rounds() {
        assert_eq!(f128_hex("0x1.8p-16494L"), 2);
        assert_eq!(f128_hex("0x1p-16495L"), 0);
        assert_eq!(f128_hex("0x1.0000000000000001p-16495L"), 1);
    }

    #[test]
    fn test_hex_f128_parse_overflow_and_sign() {
        assert_eq!(f128_hex("0x1p16384L"), 0x7fff0000000000000000000000000000);
        assert_eq!(f128_hex("-0x1p16384L"), 0xffff0000000000000000000000000000);
        assert_eq!(f128_hex("-0x1.8p1L"), 0xc0008000000000000000000000000000);
    }
}
