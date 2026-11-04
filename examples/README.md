# postings examples

These examples are small demonstrations on synthetic documents and weights.
Their output is not benchmark evidence; use `cargo bench` for measurement.

## Where to start

| I want to... | Run |
|---|---|
| Score a learned-sparse (SPLADE-style) query against f32-weighted postings | `splade_weighted` |
| Recover an index from a checkpoint plus a replayed update log | `durable_roundtrip` |
| Write immutable raw segment files and run top-k across them | `raw_segment_file` |

## Default features

| Example | What it shows |
|---|---|
| `splade_weighted` | Builds a `PostingsIndex<String, f32>` from four weighted documents, prints union candidates for a weighted query, ranks them by sparse inner product with `top_k_weighted`, and reads back one stored weight |
| `durable_roundtrip` | Logs add/delete events with `durability::recordlog`, snapshots to a checkpoint, replays the log suffix to rebuild a `PostingsIndex`, asserts every document containing the query term is a candidate, and prints the planner decision for two queries |

```sh
cargo run --release --example splade_weighted
cargo run --release --example durable_roundtrip
```

CI also runs `durable_roundtrip` as an assertion check.

## Feature-gated examples

| Feature | Example | What it shows |
|---|---|---|
| `raw-segment` | `raw_segment_file` | Writes two raw segment files, opens them as `RawSegmentFile`, prints resident metadata and payload sizes, runs weighted top-k across both, and prints segment seen/scored/pruned counts |

```sh
cargo run --release --features raw-segment --example raw_segment_file
```
