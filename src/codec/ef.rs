//! Elias-Fano helpers for monotone sequences.
//!
//! This module is feature-gated behind `postings/sbits`.
//! It provides a compact representation for sorted `u32` ids.

/// Re-export the underlying succinct structure.
pub type EliasFano = sbits::EliasFano;

/// Build an Elias-Fano structure from sorted ids.
///
/// Caller must ensure `ids` are sorted and in `[0, universe_size)`.
pub fn elias_fano_from_sorted_ids(ids: &[u32], universe_size: u32) -> EliasFano {
    try_elias_fano_from_sorted_ids(ids, universe_size)
        .expect("ids must be strictly increasing and inside the universe")
}

/// Build an Elias-Fano structure from sorted ids, validating the public contract.
pub fn try_elias_fano_from_sorted_ids(
    ids: &[u32],
    universe_size: u32,
) -> Result<EliasFano, crate::codec::Error> {
    crate::codec::validate_sorted_ids(ids, universe_size)?;
    let ids64: Vec<u64> = ids.iter().map(|&x| x as u64).collect();
    Ok(EliasFano::new(&ids64, universe_size as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn elias_fano_empty_is_empty() {
        let ef = elias_fano_from_sorted_ids(&[], 0);
        assert_eq!(ef.len(), 0);
        assert!(ef.is_empty());
    }

    #[test]
    fn elias_fano_roundtrip_get_matches_input() {
        let ids = vec![1u32, 5, 10, 20, 50, 100];
        let ef = elias_fano_from_sorted_ids(&ids, 1_000);
        assert_eq!(ef.len(), ids.len());
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(ef.get(i).unwrap(), id as u64);
        }
    }

    #[test]
    fn try_elias_fano_rejects_unsorted_ids() {
        let err = try_elias_fano_from_sorted_ids(&[1, 5, 3], 10).unwrap_err();
        assert_eq!(
            err,
            crate::codec::Error::NotStrictlyIncreasing {
                index: 2,
                prev: 5,
                next: 3
            }
        );
    }

    #[test]
    fn try_elias_fano_rejects_ids_outside_universe() {
        let err = try_elias_fano_from_sorted_ids(&[1, 10], 10).unwrap_err();
        assert_eq!(
            err,
            crate::codec::Error::IdOutOfUniverse {
                index: 1,
                id: 10,
                universe_size: 10
            }
        );
    }

    proptest! {
        #[test]
        fn elias_fano_property_get_matches_input(mut ids in prop::collection::vec(0u32..1_000_000u32, 0..500)) {
            ids.sort_unstable();
            ids.dedup();
            let ef = elias_fano_from_sorted_ids(&ids, 1_000_000);
            prop_assert_eq!(ef.len(), ids.len());
            for (i, &id) in ids.iter().enumerate() {
                prop_assert_eq!(ef.get(i).unwrap(), id as u64);
            }
        }
    }
}
