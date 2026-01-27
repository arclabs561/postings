//! Elias-Fano helpers for monotone sequences.
//!
//! This module is feature-gated behind `postings-codec/sbits`.
//! It provides a compact representation for sorted `u32` ids.

/// Re-export the underlying succinct structure.
pub type EliasFano = sbits::EliasFano;

/// Build an Elias-Fano structure from sorted ids.
///
/// Caller must ensure `ids` are sorted and in `[0, universe_size)`.
pub fn elias_fano_from_sorted_ids(ids: &[u32], universe_size: u32) -> EliasFano {
    EliasFano::new(ids, universe_size)
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
            assert_eq!(ef.get(i).unwrap(), id);
        }
    }

    proptest! {
        #[test]
        fn elias_fano_property_get_matches_input(mut ids in prop::collection::vec(0u32..1_000_000u32, 0..500)) {
            ids.sort_unstable();
            ids.dedup();
            let ef = elias_fano_from_sorted_ids(&ids, 1_000_000);
            prop_assert_eq!(ef.len(), ids.len());
            for (i, &id) in ids.iter().enumerate() {
                prop_assert_eq!(ef.get(i).unwrap(), id);
            }
        }
    }
}
