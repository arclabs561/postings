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

### Fixed

- Removed stale postings on delete and kept global postings sorted for
  out-of-order document ids.

## [0.1.8] - 2026-06-10

### Changed

- Documentation and CI polish; no API changes.
