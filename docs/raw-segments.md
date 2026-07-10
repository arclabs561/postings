# Raw Segments

`raw-segment` exposes numeric-term byte and file segment formats for large
lexical or learned-sparse indexes. The readers keep fixed directories resident
and range-read posting payloads for query terms.

## Metadata

`RawSegmentFile::resident_metadata_len` and
`RawSegmentFile::posting_payload_len` report resident metadata separately from
posting payload bytes. `RawSegment::for_each_term_id` and
`RawSegmentFile::for_each_term_id` stream the term directory without allocating a
full term-id list.

`RawSegment::for_each_term_meta` and `RawSegmentFile::for_each_term_meta` also
stream document frequency, maximum weight, and total weight from the same
directory pass.

Raw segments use separate id domains. Document ids remain `u32`, matching
`postings::DocId` and the current consumer crates. Numeric term ids are `u64`,
matching `postings::raw::RawTermId` and leaving room for learned-sparse
vocabulary dimensions. A `segstore` segment id is a separate `u64` lifecycle id
used only to name and validate sidecars.

## Writers

Raw segments can be encoded from a slice, document iterator, sorted document
iterator, or term-major posting lists into a `Vec<u8>`, caller-provided `Write`
sink, or seekable writer.

The writer API avoids requiring the final segment as one contiguous allocation
when the caller wants to stream bytes into its own durability layer. When
document ids are already strictly increasing, the sorted-iterator writer also
avoids the encoder's whole-corpus document map before postings are written.
When an external sorter or merge already has term-major lists, the term-major
writer avoids doc-to-term transposition too.

## Search

`RawSegmentFile::top_k_weighted_u32` scores one raw file by sparse inner
product. `top_k_weighted_u32_files` merges exact top-k results across raw files
when document ids are globally unique.

The file-backed scorer uses block metadata for bounded reads and safe top-k
pruning where the query weights make that possible. Use
`top_k_weighted_u32_files_with_stats` when you need searched/pruned segment
counts for profiling. Segment pruning is layout-sensitive, so measure it against
representative segment construction before treating it as a storage win.

`top_k_weighted_u32_files_and_index` fuses sealed files with one live
`PostingsIndex<u64, u32>` shard and uses the live shard's current top-k
threshold to skip low-bound sealed files.

Use `lexir::raw` for BM25 over one or more raw files.

## Filtering

Use `RawSegmentFile::top_k_weighted_u32_filtered` or
`top_k_weighted_u32_files_filtered` when a lifecycle layer supplies tombstones
or newer-version masks. The predicate runs before local top-k truncation.

## Embedded Files

Use `RawSegmentFile::from_file_range` when a raw segment is embedded inside a
container file, such as a lifecycle-owned sidecar envelope. The containing layer
validates its header, recipe, and segment id, then passes the raw payload byte
range to `RawSegmentFile`. Raw-format offsets remain relative to the payload
start; posting range reads still hit the original file without copying the whole
payload into memory.

## Segstore Sidecars

`postings::raw` is the byte format a `segstore` consumer can place under
`segstore.idx.<segment-id>.<kind>`. The `segstore` segment remains the durable
source payload; the raw postings bytes are rebuildable derived data keyed by the
stable segment id. Use `segstore::SidecarEnvelope` for the repeated
magic/version/segment-id/recipe/CRC framing, then put the raw segment bytes in
the envelope payload.

Release ordering matters for crates that depend on both `postings` and
`durability`: publish `postings` with the new `durability` version before
consumers such as `lexir` bump their direct `durability` dependency. Otherwise
Cargo links two `durability` versions, and `durability::Directory` arguments
from the consumer do not satisfy the trait expected by the published `postings`
crate.

## Positional Segments

`sorted_document_lengths` and `sorted_term_posting_lists` expose stable,
borrowed, sorted streams for sealing a bounded in-memory positional shard into a
consumer-owned segment format without cloning position vectors.

With `raw-segment` also enabled, `postings::positional::raw` can write and open
checked byte-backed or file-backed positional segments. `RawPositionalSegmentFile`
keeps term/document metadata resident, range-reads term payloads, validates each
term payload with its directory checksum, and can stream-check the whole postings
payload on demand.

Segment-set helpers such as `phrase_match_strs_segment_files` and
`near_match_terms_strs_segment_files` union exact results across sealed
positional files. Use the `_filtered` helpers when a lifecycle layer supplies
tombstones or newer-version masks.

`RawPositionalTermCache` lets serving paths reuse decoded term lists across
file-backed queries without forcing one-shot scans to retain them. Cache-aware
file segment-set helpers take one cache per segment.

## Boundaries

The same list-contiguous storage shape can inform vector-search posting lists
such as IVF list ids and codes: keep fixed directories resident, range-read only
the candidate lists a query probes, and report resident metadata separately from
payload bytes. Vector-specific distance kernels and quantizer state should stay
in the ANN crate; `postings` should remain a lexical/sparse postings library.

This is not a full index lifecycle by itself. Callers own term-id mapping,
commit publication, deletes, compaction, and crash-safety policy.
