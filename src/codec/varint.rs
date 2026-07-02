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
#[inline]
pub fn decode_u32(bytes: &[u8]) -> Option<(u32, usize)> {
    let b0 = *bytes.first()?;
    if b0 < 0x80 {
        return Some((b0 as u32, 1));
    }
    let b1 = *bytes.get(1)?;
    let x = ((b0 & 0x7F) as u32) | (((b1 & 0x7F) as u32) << 7);
    if b1 < 0x80 {
        return Some((x, 2));
    }
    let b2 = *bytes.get(2)?;
    let x = x | (((b2 & 0x7F) as u32) << 14);
    if b2 < 0x80 {
        return Some((x, 3));
    }
    let b3 = *bytes.get(3)?;
    let x = x | (((b3 & 0x7F) as u32) << 21);
    if b3 < 0x80 {
        return Some((x, 4));
    }
    let b4 = *bytes.get(4)?;
    if b4 > 0x0F {
        return None;
    }
    Some((x | ((b4 as u32) << 28), 5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrips_u32_boundaries() {
        for value in [0, 1, 127, 128, 16_383, 16_384, u32::MAX] {
            let mut bytes = Vec::new();
            encode_u32(value, &mut bytes);

            assert_eq!(decode_u32(&bytes), Some((value, bytes.len())));
        }
    }

    #[test]
    fn varint_rejects_truncated_values() {
        assert_eq!(decode_u32(&[0x80]), None);
        assert_eq!(decode_u32(&[0x80, 0x80]), None);
        assert_eq!(decode_u32(&[0x80, 0x80, 0x80]), None);
        assert_eq!(decode_u32(&[0x80, 0x80, 0x80, 0x80]), None);
    }

    #[test]
    fn varint_rejects_u32_overflow() {
        assert_eq!(decode_u32(&[0xFF, 0xFF, 0xFF, 0xFF, 0x10]), None);
        assert_eq!(decode_u32(&[0x80, 0x80, 0x80, 0x80, 0x80]), None);
    }
}
