# Changelog

## [Unreleased]

### Added

- Added Criterion coverage for large single-term filtered raw-file top-k search.
- Added checked Elias-Fano constructors for codec ids and positional candidate
  doc-id sets.

### Changed

- Changed large single-term filtered `RawSegmentFile` top-k search to reuse
  block-max pruning while applying visibility before ranking.
- Changed raw positional phrase queries for three distinct terms to use the
  rarest decoded term as the anchor, matching the in-memory fast path.
- Changed cached file-backed raw positional NEAR queries for three distinct
  terms to use the same decoded three-list scan as the in-memory fast path.
- Changed raw positional candidate and three-term fast paths to preserve anchor
  posting order without extra result sorts.

## [0.2.12] - 2026-07-04

### Added

- Added sorted positional shard export helpers for document lengths and borrowed
  term posting lists, preparing positional phrase/proximity data for
  consumer-owned segment writers without cloning position vectors.
- Added `postings::positional::raw` for checked byte-backed positional segment
  encoding, lazy term/posting lookup, and exact phrase/NEAR matching when both
  `positional` and `raw-segment` are enabled.
- Added `RawPositionalSegmentFile` for file-backed positional segments that
  keep metadata resident, range-read term payloads, and validate term payload
  checksums lazily.
- Added exact multi-segment phrase and NEAR helpers for byte-backed and
  file-backed raw positional segments.
- Added `RawPositionalTermCache` for caller-owned decoded-term reuse across
  file-backed raw positional queries.
- Added Criterion coverage for raw positional phrase and NEAR query paths beside
  the existing in-memory positional benchmarks.
- Added Criterion coverage for file-backed raw positional phrase and NEAR query
  paths.
- Added Criterion coverage for multi-file raw positional phrase and NEAR query
  paths.
- Added Criterion coverage for cached file-backed raw positional phrase and
  NEAR query paths.
- Added filtered file-backed raw top-k helpers so lifecycle layers can apply
  tombstone or newer-version masks before local top-k truncation.
- Added Criterion coverage for filtered multi-file raw top-k search.

## [0.2.11] - 2026-07-04

### Changed

- Changed `top_k_weighted_u32_files_and_index` to score the live
  `PostingsIndex` shard first, then use its exact top-k threshold to skip
  low-bound sealed raw files. Added benchmark coverage for that live-dominant
  dynamic/static query shape.

## [0.2.10] - 2026-07-04

### Added

- Added raw-segment benchmark coverage for sealing a live
  `PostingsIndex<u64, u32>` shard directly into a seekable raw segment writer.

### Changed

- Reduced live-index raw-segment sealing memory by borrowing the source index's
  posting slices instead of copying every posting into owned term lists before
  writing.

## [0.2.9] - 2026-07-04

### Added

- Added partitioned raw-file top-k benchmark coverage to make segment-pruning
  wins visible when term vocabularies are segment-local.
- Added `write_u64_u32_segment_from_index_seekable_to` to seal a live
  `PostingsIndex<u64, u32>` shard directly into a raw segment file.
- Added `top_k_weighted_u32_files_and_index` to fuse immutable raw segment
  files with one live `PostingsIndex<u64, u32>` shard for exact top-k search.
- Added a `raw_segment_file` example for writing immutable raw impact files and
  querying them with file-backed multi-segment top-k search.

## [0.2.8] - 2026-07-03

### Added

- Added `RawTermMeta`, `RawSegment::for_each_term_meta`, and
  `RawSegmentFile::for_each_term_meta` for streaming raw term-directory stats
  without allocating term-id lists or doing a second lookup per term.

## [0.2.7] - 2026-07-03

### Added

- Added `RawSegmentFile::resident_metadata_len` and
  `RawSegmentFile::posting_payload_len` so callers can budget file-backed raw
  segments by resident directory bytes versus range-read posting payload bytes.
- Added `RawSegment::for_each_term_id` and `RawSegmentFile::for_each_term_id`
  for streaming term dictionaries without allocating a term-id vector.

## [0.2.6] - 2026-07-03

### Added

- Added term-major raw segment writer APIs for external sort or spill/merge
  pipelines that already have sorted document metadata and posting lists,
  avoiding the encoder's doc-to-term transposition.
- Added a seekable term-major raw segment writer for local-file style sinks that
  can reserve directories, stream posting blocks once, then backpatch metadata.
- Added a raw-segment multi-file top-k benchmark for many small segment files.
- Added `RawSegment::doc_id_range` and `RawSegmentFile::doc_id_range` for
  consumers that need a compact segment-local document-id span.

### Changed

- Changed raw-segment dense candidate and top-k scratch buffers to use the
  segment's document-id span instead of allocating from zero to the maximum
  document id.
- Changed the term-major raw segment sink writer to stream encoded posting
  blocks instead of buffering the full postings payload before writing.

### Fixed

- Preserved existing `raw::Error` implicit discriminant values while adding
  term-major writer error variants.

## [0.2.5] - 2026-07-03

### Added

- Added sorted-document raw segment writer APIs for callers that can provide
  strictly increasing document ids, reducing encoder memory by avoiding the
  whole-corpus document map before postings are written.

## [0.2.4] - 2026-07-03

### Added

- Added `top_k_weighted_u32_files_with_stats` for raw multi-file sparse top-k
  segment-pruning diagnostics.
- Added `write_u64_u32_segment_to` for emitting raw segments into
  caller-provided writers without requiring the final segment as one contiguous
  `Vec<u8>`.
- Added `write_u64_u32_segment_from_iter` and
  `write_u64_u32_segment_from_iter_to` so raw segment callers can stream
  document views into the encoder without first building a `Vec<RawDocument>`.
- Added borrowed-term positional query helpers:
  `PosingsIndex::phrase_match_strs` and `near_match_terms_strs`.

### Changed

- Reduced raw-segment writer peak memory by hashing and writing fixed
  directories incrementally instead of building full directory byte buffers.
- Reduced raw-segment file-backed block scoring allocations by reusing one
  encoded-block buffer during block traversal.
- Sped up raw-segment pruned block top-k scoring by maintaining the current
  threshold incrementally instead of rescanning all touched documents after each
  block.
- Sped up multi-file raw-segment sparse top-k queries by skipping finite
  zero-bound segments before scoring.
- Changed the multi-file raw sparse top-k benchmark to keep segment-reference
  construction outside the timed loop.
- Sped up positional phrase and proximity queries by reusing term posting maps
  inside specialized two- and three-term paths instead of re-hashing term
  strings for every candidate document.

## [0.2.3] - 2026-07-03

### Added

- Added `RawSegment::for_each_posting_block_with_document_len` and
  `RawSegmentFile::for_each_posting_block_with_document_len` for block-scoped
  scorers that need term weight and document length without decoding a full
  posting list.
- Added `RawSegment::for_each_document_len` and
  `RawSegmentFile::for_each_document_len` for scorers that want to build dense
  document-length caches from raw segment metadata.
- Added raw-segment content checksums behind the `raw-segment` feature. New
  writes include directory CRC32s plus per-posting-block CRC32s; readers keep
  accepting legacy flag-zero raw segments and file-backed readers verify posting
  blocks lazily as ranges are read.

## [0.2.2] - 2026-07-03

### Changed

- Sped up raw-segment posting visitors that include document lengths by
  scanning document metadata with a forward cursor instead of doing a binary
  search for every posting. In focused downstream `lexir` BM25 runs, this
  enabled file-backed two-term raw queries around 292 us and eight-term raw
  queries around 1.09 ms while preserving duplicate-heavy query performance.

## [0.2.1] - 2026-07-03

### Added

- Added `RawSegment::for_each_posting`,
  `RawSegment::for_each_posting_with_document_len`,
  `RawSegmentFile::for_each_posting`, and
  `RawSegmentFile::for_each_posting_with_document_len` for streaming raw
  postings to scorers without materializing decoded posting lists.
- Added `postings::raw::top_k_weighted_u32_files` for exact sparse
  inner-product top-k over multiple file-backed raw segments with globally
  unique document ids.
- Added raw-segment benchmark coverage for multi-file sparse top-k merging.

## [0.2.0] - 2026-07-02

### Added

- Added a raw-segment reader design for the first larger-than-memory postings
  path: numeric term ids, consumer-owned segment bytes, block metadata, and a
  filesystem-backed reader before mmap/object-store/GPU variants.
- Added an experimental `raw-segment` feature with a byte-backed reader for
  numeric `u64` terms and `u32` weights. The reader validates the segment
  envelope, answers document metadata and document-frequency lookups from fixed
  directories, and decodes only the posting lists needed by a query.
- Added raw-segment benchmarks for open, document-frequency lookup, posting
  decode, and conjunctive candidate queries against the in-memory path.
- Added `RawSegment::candidates_any_terms` for disjunctive candidate generation
  over byte-backed numeric segments, with property coverage against
  `PostingsIndex::candidates`.
- Added `RawSegment::plan_candidates`, which can return broad-query `ScanAll`
  decisions from fixed term-directory metadata without decoding posting bytes.
- Added raw-segment posting-block metadata for each term, recording block base
  doc id, last doc id, byte range, and max term weight so later readers can
  prune or range-read posting blocks instead of treating each posting list as one
  opaque byte span.
- Added `RawSegment::posting_block_postings` for decoding one raw posting block
  without scanning the rest of the term's posting list.
- Added `RawSegmentFile`, a file-backed raw-segment reader that keeps fixed
  directories in memory and range-reads requested posting lists or posting
  blocks from the segment file.
- Added `RawSegmentFile::top_k_weighted_u32` for exact sparse inner-product
  scoring from a file-backed raw segment without loading full posting payloads
  at open time.
- Added file-backed raw-segment candidate generation and planning with
  `RawSegmentFile::candidates_all_terms`, `candidates_any_terms`, and
  `plan_candidates`.
- Added `RawSegment::term_ids` and `RawSegmentFile::term_ids` for enumerating
  numeric term ids present in a raw segment.
- Added `RawSegment::top_k_weighted_u32` for exact sparse inner-product scoring
  directly from byte-backed numeric raw segments.
- Added raw-segment benchmark coverage for file-backed open, posting-block
  decode, posting-list decode, candidate generation, planning, and exact top-k
  scoring.
- Added raw-segment benchmark coverage for disjunctive candidate generation
  and metadata-only candidate planning against the in-memory path.
- Added `PostingsIndex::top_k_weighted` for sparse inner-product ranking over
  weighted postings.
- Added a property-test quality gate that checks `top_k_weighted` against a
  brute-force sparse dot-product oracle across duplicate query terms and
  dense/sparse document id layouts.
- Added mixed-sign `top_k_weighted` benchmark coverage to keep the exact scorer
  fallback visible while positive-query fast paths improve.

### Changed

- Used a Unix positional-read path for file-backed raw posting-block payloads
  while leaving full posting-list reads on the seek/read path. In the focused
  cached-read benchmark, `file_postings_decode_common_block_0` measured
  `[832.22 ns 837.60 ns 843.21 ns]`, and `file_top_k_weighted_5` measured
  `[758.43 us 760.40 us 762.20 us]`.
- Sped up raw-segment posting consumers by decoding directly into hot-path
  callers instead of routing every posting through an iterator `Result`. In
  focused runs, `raw_segment/raw_top_k_weighted_5` moved from
  `[774.40 us 776.69 us 779.21 us]` to
  `[550.53 us 551.25 us 551.84 us]`, and
  `raw_segment/raw_candidates_all_terms_5` improved by about 29%.
- Refreshed cached float weight bounds when deleting a boundary weighted
  posting, so later positive queries can regain the dense non-negative scorer
  path. In the focused benchmark,
  `weighted_top_k/after_negative_delete/2` moved from
  `[208.00 us 208.50 us 208.95 us]` to
  `[166.84 us 167.39 us 167.97 us]`.
- Sped up raw-segment conjunctive candidate queries by intersecting decoded doc
  id lists in place instead of allocating a fresh candidate vector for each
  query term. In the focused raw-segment benchmark,
  `raw_candidates_all_terms_5` moved from `[930.64 us 947.83 us 958.44 us]`
  to `[901.89 us 910.71 us 920.60 us]`.
- Reduced raw-segment conjunctive query allocations by reusing a decoded doc-id
  scratch buffer across non-anchor posting lists. The focused latency benchmark
  stayed within noise, but multi-term queries no longer allocate a fresh
  scratch vector for each intersected list.
- Capped dense per-query scratch for candidate generation and weighted top-k.
  Dense accumulators are still used for small and medium dense indexes, but
  massive dense doc-id spaces now fall back to sparse accumulation instead of
  allocating scratch proportional to the corpus on every query.
- Sped up three-unique-term positional proximity queries by bypassing the
  generic candidate-set materialization and checking positions directly from
  the rarest term's postings map. Duplicate-term queries still use the generic
  multiplicity-aware path. In focused runs,
  `near_unordered_3_terms_window_16` moved from
  `[719.80 us 723.63 us 728.76 us]` to
  `[342.90 us 343.84 us 344.77 us]`, and
  `near_ordered_3_terms_window_16` moved from
  `[726.54 us 729.49 us 732.68 us]` to
  `[331.20 us 332.07 us 333.01 us]`.
- Sped up positional phrase and two-term proximity queries by reusing the
  already-visited anchor term positions instead of re-looking them up for every
  candidate document. In the positional benchmark, `phrase_3_terms` moved from
  `[500.79 us 502.83 us 504.83 us]` to
  `[337.84 us 338.87 us 339.92 us]`, `near_pair_window_4` moved from
  `[339.59 us 341.14 us 342.66 us]` to
  `[161.56 us 161.81 us 162.06 us]`, and
  `near_pair_skewed_window_4` moved from `[85.515 us 85.774 us 86.030 us]` to
  `[39.087 us 39.148 us 39.211 us]`.
- Sped up deleting documents that appear in long common-term posting lists by
  removing the sorted posting entry with binary search instead of scanning the
  full list. In the delete benchmark, `delete/common_terms_20k` moved from
  `[3.2583 ms 3.3284 ms 3.4373 ms]` to
  `[3.0940 ms 3.1817 ms 3.2701 ms]`; `delete/common_terms_5k` stayed within
  noise.
- Sped up positive multi-term `top_k_weighted` queries by caching per-term
  weight bounds and skipping the dense scorer's separate `seen` bitmap when
  every contribution is non-negative. In focused runs, `weighted_top_k/terms/5`
  moved from `[266.33 us 273.77 us 276.66 us]` to
  `[237.46 us 238.95 us 239.92 us]`,
  `weighted_top_k/expanded_terms/16` moved from
  `[467.64 us 469.66 us 472.90 us]` to
  `[438.44 us 438.91 us 439.75 us]`, and
  `weighted_top_k/rare_terms/8` moved from
  `[7.0772 us 7.1154 us 7.1553 us]` to
  `[5.6593 us 5.6818 us 5.7066 us]`.
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

- Rejected overflowing five-byte `u32` varints instead of accepting payload bits
  above `u32::MAX` in raw posting streams.
- Removed stale postings on delete and kept global postings sorted for
  out-of-order document ids.
- Omitted zero-score documents from multi-term `top_k_weighted` results after
  exact score cancellation, matching the single-term path.
- Fixed same-term positional proximity so `near_match(term, term, window)`
  requires two distinct token positions instead of matching a single occurrence
  against itself.
- Made single-term phrase results deterministic by sorting doc ids before
  returning.

## [0.1.8] - 2026-06-10

### Changed

- Documentation and CI polish; no API changes.
