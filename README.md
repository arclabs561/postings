# postings

[![crates.io](https://img.shields.io/crates/v/postings.svg)](https://crates.io/crates/postings)
[![Documentation](https://docs.rs/postings/badge.svg)](https://docs.rs/postings)
[![CI](https://github.com/arclabs561/postings/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/postings/actions/workflows/ci.yml)

Inverted-index postings lists with segment-style updates.
Supports `u32` term frequencies (classical IR) and `f32` weights
(SPLADE / learned-sparse retrieval).

## Data Model & Invariants

- **Doc IDs**: `u32`. Must be dense/contiguous for optimal compression.
- **Ordering**: Postings lists are always sorted by Doc ID.
- **Updates**: Segment-based. Deletions are tombstones; updates are delete+add.
- **Storage**: In-memory by default. Persistence via `durability` (optional).

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

## File-backed segments

The `raw-segment` feature exposes `postings::raw`, a numeric-term segment format
with a byte-backed reader and a file-backed reader. The file reader keeps the
fixed directories in memory and range-reads posting payloads for the query terms.
It is the path intended for large lexical and learned-sparse indexes whose
posting payloads should not be rebuilt into a full `PostingsIndex` on every
open.

This is not a full index lifecycle by itself: callers still own term-id mapping,
commit publication, deletes, and compaction. Pair it with `segstore` sidecars or
a higher-level search crate when those lifecycle guarantees are needed.

## Features

- `postings/serde`: enable serde for the in-memory structures.
- `postings/persistence`: enable save/load helpers via `durability` + `postcard`.
- `postings/sbits`: enable succinct monotone sequences (Elias–Fano) under `postings::codec::ef`.
- `postings/positional`: enable positional postings (`postings::positional::PosingsIndex`).
- `postings/cnk-compression`: enable optional compressed-candidate helpers under `postings::positional::cnk_candidates`.
- `postings/raw-segment`: enable the experimental byte- and file-backed raw segment reader.

## Optional: positional postings

Enable positional postings behind a feature flag:

```toml
[dependencies]
postings = { version = "0.2", features = ["positional"] }
```

Then use `postings::positional::PosingsIndex` for phrase/proximity evaluation.

## Development

```bash
cargo test
```
