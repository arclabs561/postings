---
status: proposed
date: 2026-07-02
scope: postings raw segment format, filesystem-backed readers, segstore raw consumer
grounded-in:
  - /Users/arc/Documents/dev/segstore/docs/design/out-of-core-segment-reader.md
---

# Design: Postings Raw Segment Reader

## Problem

`postings` has fast in-memory candidate and weighted top-k paths, but its
persistence story is still whole-index postcard save/load or operation-log
replay plus rebuild. That is useful for restart safety, but it does not support
massive indexes whose dictionaries, posting blocks, positions, and scoring
metadata cannot all fit in memory.

The first out-of-core consumer for `segstore` should be `postings`, because its
disk format has concrete, measurable requirements: sorted doc-id blocks,
frequencies or learned-sparse weights, optional positions, document statistics,
and block summaries for pruning. If `postings` cannot use the raw segment
contract cleanly, vector and sketch consumers will not either.

## Context

Current `PostingsIndex<Term, W>` stores a flat `HashMap<Term, Vec<(DocId, W)>>`
and eagerly removes deleted postings from that map. It also keeps per-document
lengths, document frequency, total length, and per-term weight bounds. The hot
query paths rely on sorted live posting lists, dense scratch only when doc ids
are compact enough, and bounded top-k selection.

Current `postings::codec` already contains low-level gap and varint helpers, but
there is no segment byte format and no reader that can answer a query without
loading a full `PostingsIndex`.

The governing `segstore` design says the raw path must be a parallel API:
`segstore` owns ids, manifests, WAL lifecycle, tombstones, compaction scheduling,
pinned views, and sidecar GC. The consumer owns segment bytes, metadata, and
query readers. Rebuildable sidecars stay best-effort caches; raw segments are
source-of-truth bytes.

## Non-Goals

- Do not replace `PostingsIndex<Term, W>` or break the current in-memory API.
- Do not add object-store, async I/O, mmap, GPU, or far-memory dependencies in
  the first raw reader.
- Do not make `segstore` parse posting blocks, positions, weights, or term
  dictionaries.
- Do not support arbitrary serde `Term` values in the first byte-native format.
  Numeric term ids come first; higher layers can own text dictionaries.
- Do not solve global optimize/merge across every postings format variant in
  the first commit. The first gate is read without full segment decode.

## Options Considered

### Postcard `PostingsIndex` per segment

Rejected as the out-of-core path. It preserves the current API and is easy to
write, but every query still pays full segment decode or full sidecar load. That
is restart persistence, not larger-than-memory search.

### Generic serde terms in the raw segment

Deferred. It would make `PostingsIndex<String>` feel direct, but byte-native
query readers need deterministic term ordering, compact dictionary lookup, and
stable binary equality. Making arbitrary `Term: Serialize + DeserializeOwned`
part of the raw format would either reintroduce full dictionary decode or force
a generic term codec before we have evidence it is worth the API cost.

### Numeric term ids first

Chosen for the first raw path. `lexir` can map terms to numeric ids, `sporse`
already thinks in sparse dimensions, and learned-sparse models use vocabulary
dimensions naturally. A later text dictionary can be a higher-layer artifact or
a separate section, but the postings reader should start with `u64` term ids and
`u32` doc ids.

### One universal compressed layout

Rejected. CPU WAND/BMW wants compressed sorted blocks and upper bounds. GPU
scan wants flat, aligned arrays. Positional phrase search wants position blocks
and skip metadata. The first format should be versioned and sectioned so a CPU
filesystem reader can land without pretending to be the final layout for every
execution engine.

## Chosen Approach

Add a `raw` design surface to `postings`, initially behind a feature name such
as `raw-segment` once implemented. The first raw segment format is a
filesystem-friendly, numeric-term, CPU-query format:

- header: magic, version, flags, checksum policy, endianness marker;
- segment metadata: term count, doc count, max doc id, total doc length, weight
  kind, block size, offsets to sections;
- term directory: sorted `u64` term ids with document frequency, collection
  term frequency, max score or weight bound, and posting-list offset;
- doc metadata: sorted doc ids plus document lengths, either dense when max doc
  id is compact or blocked sparse when it is not;
- posting blocks: doc-id gaps plus `u32` frequencies or `f32` weights, split
  into fixed target-size blocks with per-block max score and last doc id;
- weighted metadata: per-term and per-block bounds must record enough sign
  information to prove nonnegative contributions; mixed-sign queries must fall
  back to exact scoring rather than unsafe block pruning;
- optional positions section: per-term, per-doc position blocks for phrase and
  proximity queries;
- footer: format version and checksum over the payload or selected sections.

The reader API should expose capabilities already used by the in-memory code:

- `df(term_id)`, `document_len(doc_id)`, `num_docs()`, `avg_doc_len()`;
- `postings(term_id)` as a block iterator, not a `Vec`;
- `candidates_any`, `candidates_all`, and `top_k_weighted` over one or more raw
  segment readers;
- optional positional readers that can retrieve positions for a term/doc pair
  without decoding all positions for that term.

The first implementation can use `std::fs::File` plus `pread`-style range reads
or a small buffered file abstraction. Mmap can come after the format and tests
are stable. `MemoryDirectory` remains a correctness backend only; performance
claims must use a filesystem directory.

## Tradeoffs

Numeric term ids make the first format less ergonomic for users who only have
string terms. That is acceptable because the main consumers (`lexir`, `sporse`,
and learned-sparse workflows) can own a term dictionary above postings, and
because a generic term codec would obscure the real disk-layout work.

The raw reader duplicates some in-memory query logic. That duplication is
intentional until the disk iterator proves itself. Prematurely abstracting over
`Vec<(DocId, W)>` and byte-backed block iterators would risk slowing the current
hot path or hiding I/O costs.

Block metadata increases write complexity. It earns its place only if
benchmarks show fewer decoded postings or lower latency for top-k and
conjunctive queries. If it does not, the raw format should shrink.

## Implementation Plan

1. Define `RawSegmentMeta` and a non-public binary envelope in `postings`
   tests, with golden corrupt-header and version-rejection cases. Reversible.
2. Implement a writer for `u64` terms and `u32` weights over a single immutable
   segment. Gate: round-trip `df`, `document_len`, `postings(term)`, and
   `candidates_all` without constructing `PostingsIndex`. Reversible.
3. Add `f32` weight blocks and `top_k_weighted` over one raw segment. Gate:
   property test against in-memory `PostingsIndex<String, f32>` after mapping
   strings to term ids, including duplicate query terms and mixed-sign weights.
   Partially reversible because it fixes the weight layout.
4. Add multi-segment query helpers over a pinned raw view supplied by segstore.
   Gate: query sees checkpoint-visible segments and tombstones without full
   segment decode. Partially reversible because it touches segstore API.
5. Add positional blocks only after doc-only and weighted raw readers pass.
   Gate: phrase/proximity tests match `PosingsIndex` and benchmark decodes only
   relevant position blocks. Reversible until public.
6. Only after local filesystem evidence exists, revisit mmap, range/vectored
   reads, GPU/HBM layouts, and object-store publication.

## Decision Gates

- If a raw segment query still decodes every posting list in the segment, stop
  and redesign the block directory before adding segstore integration.
- If string-term users become the first real consumer, add a dictionary section
  or higher-layer dictionary artifact before stabilizing the numeric-only API.
- If weighted top-k cannot reuse block max or weight-bound metadata to reduce
  decoded work, keep top-k raw support experimental.
- If a block-pruning path changes results for mixed-sign learned-sparse queries,
  delete that optimization and keep the exact scorer as the only public raw
  weighted path.
- If `MemoryDirectory` is the only passing benchmark backend, do not claim
  out-of-core performance.
- If `segstore` pinning cannot keep raw segment files and sidecars alive across
  a reader view, implement pinning before exposing concurrent raw readers.

## Open Questions

- Should the first raw writer live in `postings` only, or behind a narrow
  `segstore` integration feature that writes raw bytes directly into segment
  files?
- Should the checksum cover the whole raw segment or each section independently
  so a reader can verify only the ranges it touches?
- Should doc lengths live in every postings raw segment or in a shared corpus
  metadata segment owned by `lexir`?
- What is the first public term-id type: `u64`, `u32`, or a newtype that leaves
  room for field ids?
