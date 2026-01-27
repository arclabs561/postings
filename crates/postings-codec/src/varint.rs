//! Simple varint encoding for `u32`.
//!
//! Format: little-endian base-128 (LEB128-like): 7 bits per byte, MSB is continuation.

/// Encode a `u32` into varint bytes.
pub fn encode_u32(mut x: u32, out: &mut Vec<u8>) {
    while x >= 0x80 {
        out.push(((x as u8) & 0x7F) | 0x80);
        x >>= 7;
    }
    out.push(x as u8);
}

/// Decode a `u32` from varint bytes, returning (value, bytes_consumed).
pub fn decode_u32(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut x: u32 = 0;
    let mut shift = 0u32;
    for (i, &b) in bytes.iter().enumerate() {
        let low = (b & 0x7F) as u32;
        x |= low.checked_shl(shift)?;
        if (b & 0x80) == 0 {
            return Some((x, i + 1));
        }
        shift = shift.checked_add(7)?;
        if shift > 28 {
            return None;
        }
    }
    None
}
