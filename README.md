# postings

Postings-list primitives for inverted indices, packaged as a small Rust crate.

## Data Model & Invariants

- **Doc IDs**: `u32`. Must be dense/contiguous for optimal compression.
- **Ordering**: Postings lists are always sorted by Doc ID.
- **Updates**: Segment-based. Deletions are tombstones; updates are delete+add.
- **Storage**: In-memory by default. Persistence via `durability` (optional).

## Stability & Publishing

This crate is stable for internal use. Public API may change.
Use `git` dependencies for now.

## What it is

`postings` is an in-memory, index-only inverted index meant for **candidate generation**:

- it does not store document text
- it supports “segment-style” updates
- it provides **no-false-negative** candidate sets

## Usage

Add `postings` to your `Cargo.toml`:

```toml
[dependencies]
postings = { git = "https://github.com/arclabs561/postings" }
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

## Development

```bash
cargo test
```
