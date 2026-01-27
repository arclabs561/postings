//! `postings-codec`: codecs + iterator contracts for postings lists.
//!
//! This crate is meant to hold *mechanical* building blocks:
//! - integer encodings (varint, delta/gap)
//! - decode iterators
//! - skip/advance contracts (future)
//!
//! It intentionally does **not** define an inverted index; it exists so multiple
//! index structures can share the same low-level encoding/decoding logic.

#![warn(missing_docs)]

pub mod varint;

/// Succinct monotone sequences (feature-gated).
#[cfg(feature = "sbits")]
pub mod ef;

/// Errors for postings codecs.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Input ids were not strictly increasing.
    #[error("ids are not strictly increasing at index {index}: prev={prev}, next={next}")]
    NotStrictlyIncreasing {
        /// Index where the monotonicity violation occurred.
        index: usize,
        /// Previous id (expected smaller than `next`).
        prev: u32,
        /// Next id (must be strictly greater than `prev`).
        next: u32,
    },
    /// Decoding overflowed `u32`.
    #[error("u32 overflow while decoding at index {index}")]
    Overflow {
        /// Index in the gaps stream where overflow occurred.
        index: usize,
    },
}

/// Encode a sorted list of doc ids as gaps (delta encoding).
///
/// This is the **checked** variant: input must be strictly increasing.
pub fn gaps_from_sorted_ids(ids: &[u32]) -> Result<Vec<u32>, Error> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(ids.len());
    out.push(ids[0]);
    for i in 1..ids.len() {
        let prev = ids[i - 1];
        let next = ids[i];
        if next <= prev {
            return Err(Error::NotStrictlyIncreasing {
                index: i,
                prev,
                next,
            });
        }
        out.push(next - prev);
    }
    Ok(out)
}

/// Encode gaps without validating sort order.
///
/// This is faster and may be useful in hot paths where inputs are already proven correct.
/// If inputs are not strictly increasing, the output is **not** a meaningful encoding.
pub fn gaps_from_sorted_ids_unchecked(ids: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut prev = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        if i == 0 {
            out.push(id);
            prev = id;
        } else {
            out.push(id.saturating_sub(prev));
            prev = id;
        }
    }
    out
}

/// Decode gaps back into absolute ids.
///
/// This is the **checked** variant: returns an error on overflow.
pub fn ids_from_gaps(gaps: &[u32]) -> Result<Vec<u32>, Error> {
    let mut out = Vec::with_capacity(gaps.len());
    let mut cur = 0u32;
    for (i, &g) in gaps.iter().enumerate() {
        if i == 0 {
            cur = g;
        } else {
            cur = cur.checked_add(g).ok_or(Error::Overflow { index: i })?;
        }
        out.push(cur);
    }
    Ok(out)
}

/// Decode gaps without overflow checking.
///
/// If overflow occurs, this saturates at `u32::MAX`, which may hide corruption.
pub fn ids_from_gaps_unchecked(gaps: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(gaps.len());
    let mut cur = 0u32;
    for (i, &g) in gaps.iter().enumerate() {
        if i == 0 {
            cur = g;
        } else {
            cur = cur.saturating_add(g);
        }
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn gaps_rejects_unsorted() {
        let err = gaps_from_sorted_ids(&[10, 10]).unwrap_err();
        assert_eq!(
            err,
            Error::NotStrictlyIncreasing {
                index: 1,
                prev: 10,
                next: 10
            }
        );
    }

    #[test]
    fn ids_from_gaps_rejects_overflow() {
        let err = ids_from_gaps(&[u32::MAX, 1]).unwrap_err();
        assert_eq!(err, Error::Overflow { index: 1 });
    }

    proptest! {
        #[test]
        fn gaps_roundtrip_strictly_increasing(mut ids in prop::collection::vec(0u32..1_000_000u32, 0..200)) {
            ids.sort_unstable();
            ids.dedup();
            // Make it strictly increasing by construction.
            let gaps = gaps_from_sorted_ids(&ids).unwrap();
            let back = ids_from_gaps(&gaps).unwrap();
            prop_assert_eq!(back, ids);
        }
    }
}
