# postings

[![crates.io](https://img.shields.io/crates/v/postings.svg)](https://crates.io/crates/postings)
[![Documentation](https://docs.rs/postings/badge.svg)](https://docs.rs/postings)
[![CI](https://github.com/arclabs561/postings/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/postings/actions/workflows/ci.yml)

Inverted index postings lists and codecs.

## Data Model & Invariants

- **Doc IDs**: `u32`. Must be dense/contiguous for optimal compression.
- **Ordering**: Postings lists are always sorted by Doc ID.
- **Updates**: Segment-based. Deletions are tombstones; updates are delete+add.
- **Storage**: In-memory by default. Persistence via `durability` (optional).

## Usage

```toml
[dependencies]
postings = “0.1”
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

## Features

- `postings/serde`: enable serde for the in-memory structures.
- `postings/persistence`: enable save/load helpers via `durability` + `postcard`.
- `postings/sbits`: enable succinct monotone sequences (Elias–Fano) under `postings::codec::ef`.
- `postings/positional`: enable positional postings (`postings::positional::PosingsIndex`).
- `postings/cnk-compression`: enable optional compressed-candidate helpers under `postings::positional::cnk_candidates`.

## Optional: positional postings

Enable positional postings behind a feature flag:

```toml
[dependencies]
postings = { version = "0.1", features = ["positional"] }
```

Then use `postings::positional::PosingsIndex` for phrase/proximity evaluation.

## Development

```bash
cargo test
```
