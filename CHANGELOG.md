# Changelog

## [Unreleased]

### Added

- Added `PostingsIndex::top_k_weighted` for sparse inner-product ranking over
  weighted postings.

### Fixed

- Removed stale postings on delete and kept global postings sorted for
  out-of-order document ids.

## [0.1.8] - 2026-06-10

### Changed

- Documentation and CI polish; no API changes.
