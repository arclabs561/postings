# Changelog

## [Unreleased]

### Added

- Added `PostingsIndex::top_k_weighted` for sparse inner-product ranking over
  weighted postings.

### Changed

- Sped up multi-term `top_k_weighted` scoring for dense doc ids by using a dense
  accumulator with sparse-id fallback. On the benchmark query workload,
  `weighted_top_k/terms/2` moved from `[1.0914 ms 1.0979 ms 1.1059 ms]` to
  `[171.76 us 171.99 us 172.21 us]`, and `weighted_top_k/terms/5` moved from
  `[2.2738 ms 2.2827 ms 2.2925 ms]` to `[288.46 us 288.98 us 289.53 us]`.
- Sped up positional phrase and unordered proximity matching by avoiding
  per-candidate shifted-list allocation and hash counting. On the positional
  query benchmark, `phrase_3_terms` moved from
  `[1.4912 ms 1.5001 ms 1.5104 ms]` to
  `[1.1631 ms 1.1670 ms 1.1757 ms]`, and
  `near_unordered_3_terms_window_16` moved from
  `[1.6475 ms 1.6708 ms 1.7120 ms]` to
  `[1.2588 ms 1.2643 ms 1.2719 ms]`.
- Added explicit weighted single-term benchmark coverage and skipped query
  weight aggregation for one-term `top_k_weighted` queries. On the learned
  sparse benchmark, `weighted_top_k/single_common_term/1` moved from
  `[88.215 us 90.264 us 91.174 us]` to
  `[87.472 us 88.244 us 89.375 us]`, and
  `weighted_top_k/single_rare_term/1` moved from
  `[324.98 ns 326.51 ns 328.00 ns]` to
  `[136.33 ns 136.67 ns 137.22 ns]`.
- Sped up common single-term `top_k_weighted` queries by keeping only the
  current top `k` postings instead of materializing every scored posting. In
  the focused benchmark, `weighted_top_k/single_common_term/1` moved from
  `[87.704 us 89.188 us 91.114 us]` to
  `[71.472 us 71.688 us 71.913 us]`, while
  `weighted_top_k/single_rare_term/1` stayed flat.
- Sped up two-term positional proximity queries with skewed term frequencies
  by anchoring on the rarer term. On the positional benchmark,
  `near_pair_skewed_window_4` moved from
  `[1.4829 ms 1.5023 ms 1.5174 ms]` to
  `[85.559 us 85.795 us 85.945 us]`.
- Sped up positional candidate filtering by avoiding a redundant re-check of
  the already-matched rarest anchor term. On the positional benchmark,
  `phrase_3_terms` moved from `[1.1363 ms 1.1387 ms 1.1408 ms]` to
  `[931.30 us 935.12 us 940.58 us]`, and
  `near_ordered_3_terms_window_16` moved from
  `[969.81 us 973.69 us 979.73 us]` to
  `[762.03 us 794.17 us 819.46 us]`.
- Added an ordered-proximity cutoff once a candidate span already exceeds the
  requested window. In the focused ordered benchmark,
  `near_ordered_3_terms_window_16` moved from
  `[738.30 us 740.00 us 743.38 us]` to
  `[722.66 us 727.38 us 732.83 us]`.
- Sped up short learned-sparse `top_k_weighted` queries by aggregating query
  weights in a small vector before falling back to a hash map for longer
  queries. In close A/B runs, `weighted_top_k/expanded_terms/16` moved from
  `[599.63 us 600.82 us 601.88 us]` to
  `[526.99 us 527.78 us 529.40 us]`,
  `weighted_top_k/expanded_terms/32` moved from
  `[853.28 us 856.27 us 858.58 us]` to
  `[739.77 us 742.00 us 744.42 us]`, and
  `weighted_top_k/rare_terms/8` moved from
  `[6.3326 us 6.3638 us 6.3793 us]` to
  `[5.9865 us 6.0191 us 6.0414 us]`.
- Sped up dense disjunctive candidate generation for three or more terms by
  marking doc-id slots once instead of building repeated pairwise unions. In
  close A/B runs, `disjunctive/terms/5` moved from
  `[297.42 us 298.42 us 300.25 us]` to
  `[98.885 us 99.443 us 100.39 us]`.
- Sped up conjunctive candidate generation by using a linear two-pointer
  intersection for similarly sized postings lists while keeping galloping
  search for skewed lists. In the focused benchmark,
  `conjunctive/terms/2` moved from `[68.089 us 68.321 us 68.584 us]` to
  `[51.512 us 51.877 us 52.065 us]`, and `conjunctive/terms/5` moved from
  `[498.44 us 501.95 us 503.96 us]` to
  `[394.70 us 399.13 us 401.53 us]`.

### Fixed

- Removed stale postings on delete and kept global postings sorted for
  out-of-order document ids.

## [0.1.8] - 2026-06-10

### Changed

- Documentation and CI polish; no API changes.
