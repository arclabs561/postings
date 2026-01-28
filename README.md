# postings

Postings-list primitives for inverted indices, packaged as a small Rust workspace.

This repo contains two crates:

- `postings`: in-memory postings index with segment-style updates (candidate generation).
- `postings-codec`: low-level codecs (gap/varint; optional Elias–Fano via `sbits`).

Related: `posings` (positional postings for phrase/proximity) lives in its own repo: <https://github.com/arclabs561/posings>

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

let cfg = PlannerConfig::default();
let plan = idx.plan_candidates(&["quick".to_string()], cfg);
assert!(matches!(plan, postings::CandidatePlan::Candidates(_)));
```

## Features

- `postings/serde`: enable serde for the in-memory structures.
- `postings/persistence`: enable save/load helpers via `durability` + `postcard`.

## Development

```bash
cargo test
```
