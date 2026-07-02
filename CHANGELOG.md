# Changelog

## [Unreleased]

### Added

- Added `PostingsIndex::top_k_weighted` for sparse inner-product ranking over
  weighted postings.
- Added a property-test quality gate that checks `top_k_weighted` against a
  brute-force sparse dot-product oracle across duplicate query terms and
  dense/sparse document id layouts.

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
- Sped up dense-doc-id multi-term `top_k_weighted` queries by keeping only the
  current top `k` scored documents instead of materializing every touched
  document before selection. In focused runs, `weighted_top_k/terms/2` moved
  from `[171.76 us 171.99 us 172.21 us]` to
  `[147.67 us 147.93 us 148.21 us]`, `weighted_top_k/terms/5` moved from
  `[288.46 us 288.98 us 289.53 us]` to
  `[224.35 us 224.83 us 225.32 us]`, and
  `weighted_top_k/expanded_terms/32` moved from
  `[741.00 us 742.18 us 743.25 us]` to
  `[592.20 us 593.69 us 595.36 us]`.
- Sped up sparse-doc-id `top_k_weighted` queries by feeding accumulated scores
  directly into bounded top-`k` selection instead of collecting every score
  first. In focused runs, `weighted_top_k_sparse_doc_ids/terms/5` moved from
  `[792.09 us 794.14 us 796.18 us]` to
  `[747.24 us 748.95 us 750.79 us]`, and
  `weighted_top_k_sparse_doc_ids/terms/16` moved from
  `[1.8250 ms 1.8285 ms 1.8321 ms]` to
  `[1.7442 ms 1.7477 ms 1.7516 ms]`.
- Sped up three-term unordered proximity by scanning the three sorted position
  lists directly instead of allocating and sorting per-document occurrences. In
  the positional benchmark, `near_unordered_3_terms_window_16` moved from
  `[1.0238 ms 1.0314 ms 1.0412 ms]` to
  `[709.45 us 712.45 us 715.86 us]`.
- Sped up three-term exact phrase matching by anchoring directly on the rarest
  of the three terms instead of building the generic required-term candidate
  set. In focused runs, `phrase_3_terms` moved from
  `[932.04 us 935.40 us 939.11 us]` to
  `[478.68 us 479.73 us 480.77 us]`.
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
- Sped up short candidate queries by deduplicating query terms in a small vector
  before falling back to a `HashSet` for longer queries. In focused runs,
  `disjunctive/terms/5` moved from `[100.75 us 101.13 us 101.49 us]` to
  `[97.965 us 98.222 us 98.472 us]`, and `conjunctive/terms/5` moved from
  `[466.91 us 467.79 us 468.69 us]` to
  `[392.37 us 395.10 us 397.89 us]`.
- Sped up conjunctive candidate generation by collecting posting lists during
  term deduplication instead of sorting by `df` and doing a second set of hash
  lookups. In the focused benchmark, `conjunctive/terms/2` moved from
  `[51.429 us 51.518 us 51.617 us]` to
  `[50.780 us 50.847 us 50.915 us]`, and `conjunctive/terms/5` moved from
  `[390.82 us 395.33 us 400.03 us]` to
  `[330.26 us 334.30 us 338.50 us]`.
- Sped up two-term disjunctive candidate generation by using the dense
  mark-pass union for two dense posting lists, not only for three or more. In
  close A/B runs, `disjunctive/terms/2` moved from
  `[65.849 us 66.063 us 66.272 us]` to
  `[62.431 us 62.552 us 62.675 us]`; `disjunctive/terms/5` stayed within noise.
  The candidate-returning `plan_candidates/terms/2` benchmark moved from
  `[64.801 us 64.939 us 65.092 us]` to
  `[60.962 us 61.259 us 61.584 us]`.

### Fixed

- Removed stale postings on delete and kept global postings sorted for
  out-of-order document ids.

## [0.1.8] - 2026-06-10

### Changed

- Documentation and CI polish; no API changes.
