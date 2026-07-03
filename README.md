# postings

[![crates.io](https://img.shields.io/crates/v/postings.svg)](https://crates.io/crates/postings)
[![Documentation](https://docs.rs/postings/badge.svg)](https://docs.rs/postings)
[![CI](https://github.com/arclabs561/postings/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/postings/actions/workflows/ci.yml)

Inverted index postings lists and codecs.

Supports `u32` term frequencies for classical IR and `f32` weights for learned
sparse retrieval.

## Data Model & Invariants

- **Doc IDs**: `u32`. Sparse ids are supported; smaller gaps compress better
  and dense ids keep dense scratch paths cheap.
- **Ordering**: Postings lists are always sorted by Doc ID.
- **Updates**: `PostingsIndex` supports in-memory add/delete. Raw file segments
  are immutable; a store or application manifest owns deletes and compaction.
- **Storage**: In-memory by default; optional persistence and raw file-backed
  segment readers.

## Usage

```toml
[dependencies]
postings = "0.2"
```

Example (index + candidates):

```rust
use postings::{PostingsIndex, PlannerConfig};

let mut idx = PostingsIndex::new();
idx.add_document(0, &["the".to_string(), "quick".to_string(), "fox".to_string()])
    .unwrap();
idx.add_document(1, &["quick".to_string(), "brown".to_string(), "dog".to_string()])
    .unwrap();

// Conjunctive (AND) candidates.
assert_eq!(
    idx.candidates_all_terms(&["quick".to_string(), "dog".to_string()]),
    vec![1]
);

let cfg = PlannerConfig::default();
let plan = idx.plan_candidates(&["quick".to_string()], cfg);
assert!(matches!(plan, postings::CandidatePlan::Candidates(_)));
```

Example (learned-sparse top-k):

```rust
use postings::PostingsIndex;

let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
idx.add_weighted_document(
    0,
    &[
        ("neural".to_string(), 1.8),
        ("retrieval".to_string(), 0.4),
    ],
)
.unwrap();
idx.add_weighted_document(
    1,
    &[
        ("retrieval".to_string(), 2.6),
        ("search".to_string(), 2.2),
    ],
)
.unwrap();

let ranking = idx.top_k_weighted(&[("neural", 1.5), ("retrieval", 2.0)], 10);
assert_eq!(ranking[0].0, 1);
```

## Examples

Runnable examples live in [`examples/`](examples/):

- `durable_roundtrip` pairs `postings` with `durability` to build a crash-recoverable inverted index: update events go to a record log, snapshots to a checkpoint, and the index rebuilds from both, the persistence pattern a search engine needs to survive restarts.
- `splade_weighted` scores a small learned-sparse collection with `f32` weights
  and verifies top-k sparse inner-product search.

## File-backed segments

The `raw-segment` feature exposes `postings::raw`, a numeric-term segment format
with a byte-backed reader and a file-backed reader. The file reader keeps the
fixed directories in memory and range-reads posting payloads for the query terms.
New raw files carry directory and posting-block checksums; legacy unchecked raw
files remain readable.
It is the path intended for large lexical and learned-sparse indexes whose
posting payloads should not be rebuilt into a full `PostingsIndex` on every
open.

`RawSegmentFile::top_k_weighted_u32` scores one raw file by sparse inner product;
`top_k_weighted_u32_files` merges exact top-k results across raw files when
document ids are globally unique. The file-backed scorer uses block metadata for
bounded reads and safe top-k pruning where the query weights make that possible.
Use `top_k_weighted_u32_files_with_stats` when you need searched/pruned segment
counts for profiling.
Use `lexir::raw` for BM25 over one or more raw files.

This is not a full index lifecycle by itself: callers still own term-id mapping,
commit publication, deletes, and compaction. Pair raw files with `segstore`
sidecars or an application manifest when lifecycle guarantees are needed.

## Features

- `serde`: enable serde for the in-memory structures.
- `persistence`: enable save/load helpers via `durability` + `postcard`.
- `sbits`: enable succinct monotone sequences (Elias-Fano) under `postings::codec::ef`.
- `positional`: enable positional postings (`postings::positional::PositionalIndex`).
- `cnk-compression`: enable optional compressed-candidate helpers under `postings::positional::cnk_candidates`.
- `raw-segment`: enable the experimental checked byte- and file-backed raw segment reader.

## Optional: positional postings

Enable positional postings behind a feature flag:

```toml
[dependencies]
postings = { version = "0.2", features = ["positional"] }
```

Then use `postings::positional::PositionalIndex` for phrase/proximity
evaluation. `phrase_match_strs` and `near_match_terms_strs` accept borrowed
query terms when a parser already holds `&str`s. `PosingsIndex` remains as the
historical name from the older `posings` crate.

## Development

```bash
cargo test
```

## License

MIT OR Apache-2.0
