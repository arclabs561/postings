//! Byte-backed raw segment reader experiments.
//!
//! This module is intentionally narrow. It supports one local, numeric-term,
//! `u32`-weighted raw segment format so query code can read posting lists
//! without reconstructing a full [`crate::PostingsIndex`].

use crate::codec::varint;
use crate::{CandidatePlan, DocId, PlannerConfig};
use std::collections::{btree_map::Entry, BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"PSTRW001";
const FOOTER_MAGIC: &[u8; 8] = b"PSTRF001";
const VERSION: u32 = 3;
const HEADER_LEN: usize = 72;
const TERM_ENTRY_LEN: usize = 48;
const DOC_ENTRY_LEN: usize = 8;
const BLOCK_ENTRY_LEN: usize = 24;
const FOOTER_LEN: usize = 12;
const DEFAULT_BLOCK_SIZE: u32 = 128;
/// Header flag bit: the segment carries a content-integrity section (section
/// CRC32s + one CRC32 per posting block) between the postings region and the
/// footer. Readers that predate this flag reject it via `UnsupportedFlags`,
/// so they can never silently misread a checksummed segment.
const FLAG_CHECKSUMS: u32 = 1;
/// Fixed prefix of the integrity section: term-dir, doc-meta, and block-dir
/// CRC32s, in that order, followed by one CRC32 per block in directory order.
const INTEGRITY_HEADER_LEN: usize = 12;
// Keep normal local-file queries on one read, but cap pathological high-DF term
// payloads so file-backed traversal cannot allocate an entire huge posting list.
const FILE_FULL_POSTINGS_READ_LIMIT: u64 = 1024 * 1024;

/// A numeric term id in the first raw postings format.
pub type RawTermId = u64;

/// Segment-level diagnostics for multi-file raw sparse top-k search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawTopKSearchStats {
    /// Raw segment files supplied to the search.
    pub segments_seen: usize,
    /// Raw segment files actually scored.
    pub segments_scored: usize,
    /// Raw segment files skipped by a zero bound or current top-k threshold.
    pub segments_pruned: usize,
}

/// Hits and segment-level diagnostics from multi-file raw sparse top-k search.
#[derive(Clone, Debug, PartialEq)]
pub struct RawTopKSearchResult {
    /// Top-k hits sorted by descending score, then document id.
    pub hits: Vec<(DocId, f32)>,
    /// Segment-pruning diagnostics for the search.
    pub stats: RawTopKSearchStats,
}

/// Errors returned by raw segment encoding and decoding.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Segment bytes do not start with the expected magic.
    #[error("raw segment has invalid magic")]
    BadMagic,
    /// Segment footer is missing or does not match the expected magic.
    #[error("raw segment has invalid footer")]
    BadFooter,
    /// Segment version is newer or otherwise unsupported by this reader.
    #[error("unsupported raw segment version: {version}")]
    UnsupportedVersion {
        /// Version found in the segment header.
        version: u32,
    },
    /// Header flags contain bits this reader does not understand.
    #[error("unsupported raw segment flags: {flags:#x}")]
    UnsupportedFlags {
        /// Raw flag bits from the segment header.
        flags: u32,
    },
    /// Segment bytes ended before the named section could be read.
    #[error("truncated raw segment section: {section}")]
    Truncated {
        /// Section being decoded when truncation was found.
        section: &'static str,
    },
    /// Section offsets or lengths are inconsistent.
    #[error("invalid raw segment layout: {reason}")]
    InvalidLayout {
        /// Description of the violated layout invariant.
        reason: &'static str,
    },
    /// Stored content checksum did not match the recomputed value.
    #[error("raw segment checksum mismatch in {section}")]
    ChecksumMismatch {
        /// Section whose recomputed CRC32 disagreed with the stored one.
        section: &'static str,
    },
    /// Two documents in the input used the same id.
    #[error("duplicate raw segment doc id: {doc_id}")]
    DuplicateDocId {
        /// Duplicate document id.
        doc_id: DocId,
    },
    /// Two term posting lists in the input used the same term id.
    #[error("duplicate raw segment term id: {term_id}")]
    DuplicateTermId {
        /// Duplicate term id.
        term_id: RawTermId,
    },
    /// A sorted raw segment writer received a smaller document id after a
    /// larger one.
    #[error("raw segment doc ids must be increasing: previous {previous}, current {current}")]
    UnsortedDocId {
        /// Previous document id in the input stream.
        previous: DocId,
        /// Current document id that violated the ordering requirement.
        current: DocId,
    },
    /// A term-major raw segment writer received a smaller term id after a
    /// larger one.
    #[error("raw segment term ids must be increasing: previous {previous}, current {current}")]
    UnsortedTermId {
        /// Previous term id in the input stream.
        previous: RawTermId,
        /// Current term id that violated the ordering requirement.
        current: RawTermId,
    },
    /// A term-major raw segment writer received a term with no postings.
    #[error("empty raw segment posting list for term {term_id}")]
    EmptyPostingList {
        /// Term with no postings.
        term_id: RawTermId,
    },
    /// A term-major raw segment writer received a posting for a document id
    /// missing from the document-length table.
    #[error("raw segment posting references unknown doc {doc_id} for term {term_id}")]
    UnknownDocId {
        /// Term containing the unknown document id.
        term_id: RawTermId,
        /// Posting document id missing from document metadata.
        doc_id: DocId,
    },
    /// A raw posting used a zero term weight.
    #[error("zero raw segment weight for doc {doc_id}, term {term_id}")]
    ZeroWeight {
        /// Document containing the zero weight.
        doc_id: DocId,
        /// Term with the zero weight.
        term_id: RawTermId,
    },
    /// Per-document weight accumulation overflowed `u32`.
    #[error("raw segment term weight overflow for doc {doc_id}, term {term_id}")]
    WeightOverflow {
        /// Document whose accumulated term weight overflowed.
        doc_id: DocId,
        /// Term whose accumulated weight overflowed.
        term_id: RawTermId,
    },
    /// Document length accumulation overflowed `u32`.
    #[error("raw segment document length overflow for doc {doc_id}")]
    DocLengthOverflow {
        /// Document whose length overflowed.
        doc_id: DocId,
    },
    /// A segment contains more documents than the format can encode.
    #[error("raw segment has too many documents")]
    TooManyDocuments,
    /// A segment contains more distinct terms than the format can encode.
    #[error("raw segment has too many terms")]
    TooManyTerms,
    /// A byte range cannot be represented on this platform.
    #[error("raw segment byte range is too large")]
    SegmentTooLarge,
    /// A posting list varint could not be decoded.
    #[error("invalid varint in raw postings for term {term_id} at posting {index}")]
    InvalidVarint {
        /// Term whose posting list is corrupt.
        term_id: RawTermId,
        /// Posting index within the term's posting list.
        index: u32,
    },
    /// A decoded posting list overflowed `DocId`.
    #[error("doc id overflow in raw postings for term {term_id} at posting {index}")]
    DocIdOverflow {
        /// Term whose posting list is corrupt.
        term_id: RawTermId,
        /// Posting index within the term's posting list.
        index: u32,
    },
    /// A decoded posting list did not stay strictly increasing.
    #[error("non-increasing doc id in raw postings for term {term_id} at posting {index}")]
    NonIncreasingDocId {
        /// Term whose posting list is corrupt.
        term_id: RawTermId,
        /// Posting index within the term's posting list.
        index: u32,
    },
    /// A posting list had bytes left after decoding its declared document count.
    #[error("trailing bytes in raw postings for term {term_id}")]
    TrailingPostingsBytes {
        /// Term whose posting list has trailing bytes.
        term_id: RawTermId,
    },
}

/// Errors returned when writing a raw segment to an external writer.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum RawSegmentWriteError {
    /// Raw segment construction failed before bytes could be written.
    #[error(transparent)]
    Segment {
        /// Source raw segment error.
        #[from]
        source: Error,
    },
    /// Underlying writer I/O failed.
    #[error("raw segment writer I/O failed")]
    Io {
        /// Source I/O error.
        #[source]
        source: std::io::Error,
    },
}

impl From<std::io::Error> for RawSegmentWriteError {
    fn from(source: std::io::Error) -> Self {
        Self::Io { source }
    }
}

/// Errors returned by file-backed raw segment readers.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum RawSegmentFileError {
    /// Underlying file I/O failed.
    #[error("raw segment file I/O failed")]
    Io {
        /// Source I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Raw segment bytes or metadata were invalid.
    #[error(transparent)]
    Segment {
        /// Source raw segment error.
        #[from]
        source: Error,
    },
}

impl From<std::io::Error> for RawSegmentFileError {
    fn from(source: std::io::Error) -> Self {
        Self::Io { source }
    }
}

/// A document accepted by the first raw segment writer.
///
/// Duplicate term ids are accumulated. The document length is the sum of all
/// supplied `u32` weights before duplicate-term accumulation.
#[derive(Debug, Clone, Copy)]
pub struct RawDocument<'a> {
    doc_id: DocId,
    terms: &'a [(RawTermId, u32)],
}

impl<'a> RawDocument<'a> {
    /// Create a raw document view.
    pub fn new(doc_id: DocId, terms: &'a [(RawTermId, u32)]) -> Self {
        Self { doc_id, terms }
    }

    /// Return the document id.
    pub fn doc_id(self) -> DocId {
        self.doc_id
    }

    /// Return the raw weighted terms.
    pub fn terms(self) -> &'a [(RawTermId, u32)] {
        self.terms
    }
}

/// A term-major posting list accepted by the raw segment writer.
///
/// Term ids must be strictly increasing across lists. Posting document ids must
/// be strictly increasing within a list and must exist in the document-length
/// table supplied to the term-major writer.
#[derive(Debug, Clone, Copy)]
pub struct RawTermPostingList<'a> {
    term_id: RawTermId,
    postings: &'a [(DocId, u32)],
}

impl<'a> RawTermPostingList<'a> {
    /// Create a raw term posting-list view.
    pub fn new(term_id: RawTermId, postings: &'a [(DocId, u32)]) -> Self {
        Self { term_id, postings }
    }

    /// Return the raw term id.
    pub fn term_id(self) -> RawTermId {
        self.term_id
    }

    /// Return the raw `(doc_id, weight)` postings.
    pub fn postings(self) -> &'a [(DocId, u32)] {
        self.postings
    }
}

/// Header metadata for a raw segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawSegmentMeta {
    term_count: u32,
    doc_count: u32,
    max_doc_id: DocId,
    block_size: u32,
    total_doc_len: u64,
    term_dir_offset: u64,
    doc_meta_offset: u64,
    postings_offset: u64,
    footer_offset: u64,
    flags: u32,
}

impl RawSegmentMeta {
    fn has_checksums(self) -> bool {
        self.flags & FLAG_CHECKSUMS != 0
    }
}

impl RawSegmentMeta {
    /// Number of distinct terms in the segment.
    pub fn term_count(self) -> u32 {
        self.term_count
    }

    /// Number of documents in the segment.
    pub fn doc_count(self) -> u32 {
        self.doc_count
    }

    /// Highest document id in the segment, or zero for an empty segment.
    pub fn max_doc_id(self) -> DocId {
        self.max_doc_id
    }

    /// Target block size recorded for the segment.
    pub fn block_size(self) -> u32 {
        self.block_size
    }

    /// Sum of document lengths in the segment.
    pub fn total_doc_len(self) -> u64 {
        self.total_doc_len
    }

    /// Average document length.
    pub fn avg_doc_len(self) -> f32 {
        if self.doc_count == 0 {
            return 0.0;
        }
        self.total_doc_len as f32 / self.doc_count as f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TermEntry {
    term_id: RawTermId,
    df: u32,
    max_weight: u32,
    total_weight: u64,
    postings_offset: u64,
    postings_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TermBlockDirectory {
    block_count: u32,
    blocks_offset: u64,
}

struct RawSegmentSections {
    meta: RawSegmentMeta,
    term_entries: Vec<TermEntry>,
    term_block_directories: Vec<TermBlockDirectory>,
    doc_entries: RawDocumentEntries,
    block_entries: Vec<RawPostingBlockMeta>,
    postings_bytes: Vec<u8>,
    block_crcs: Vec<u32>,
    final_len: usize,
}

struct RawSegmentStreamPlan {
    meta: RawSegmentMeta,
    term_entries: Vec<TermEntry>,
    term_block_directories: Vec<TermBlockDirectory>,
    doc_entries: RawDocumentEntries,
    block_entries: Vec<RawPostingBlockMeta>,
    final_len: usize,
}

type RawDocumentMap = BTreeMap<DocId, (u32, BTreeMap<RawTermId, u32>)>;
type RawDocumentEntries = Vec<(DocId, u32)>;

struct EncodedRawPostingBlock {
    base_doc_id: DocId,
    last_doc_id: DocId,
    postings_len: u32,
    max_weight: u32,
    total_weight: u64,
}

struct RawBlockScoringList {
    entry: TermEntry,
    query_weight: f32,
    blocks: Vec<RawPostingBlockMeta>,
    full_postings: Option<Vec<u8>>,
}

struct RawTopKThreshold {
    ranked: Vec<(DocId, f32)>,
    k: usize,
    sorted: bool,
}

impl RawTopKThreshold {
    fn new(k: usize) -> Self {
        Self {
            ranked: Vec::with_capacity(k),
            k,
            sorted: false,
        }
    }

    fn update(&mut self, doc_id: DocId, score: f32) {
        if self.k == 0 || score == 0.0 {
            return;
        }

        if let Some(index) = self
            .ranked
            .iter()
            .position(|(ranked_doc_id, _)| *ranked_doc_id == doc_id)
        {
            self.ranked[index].1 = score;
            if self.sorted {
                self.bubble_up(index);
            }
            return;
        }

        if self.ranked.len() < self.k {
            self.ranked.push((doc_id, score));
            self.sorted = false;
            return;
        }

        self.sort_if_needed();
        let candidate = (doc_id, score);
        if crate::cmp_doc_scores(
            &candidate,
            self.ranked.last().expect("top-k buffer is full"),
        )
        .is_lt()
        {
            let last = self.ranked.len() - 1;
            self.ranked[last] = candidate;
            self.bubble_up(last);
        }
    }

    fn threshold(&mut self) -> Option<f32> {
        if self.ranked.len() < self.k {
            return None;
        }
        self.sort_if_needed();
        self.ranked.last().map(|(_, score)| *score)
    }

    fn sort_if_needed(&mut self) {
        if !self.sorted {
            self.ranked.sort_by(crate::cmp_doc_scores);
            self.sorted = true;
        }
    }

    fn bubble_up(&mut self, mut index: usize) {
        while index > 0
            && crate::cmp_doc_scores(&self.ranked[index], &self.ranked[index - 1]).is_lt()
        {
            self.ranked.swap(index, index - 1);
            index -= 1;
        }
    }
}

/// Metadata for one encoded posting block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawPostingBlockMeta {
    base_doc_id: DocId,
    last_doc_id: DocId,
    postings_offset: u64,
    postings_len: u32,
    max_weight: u32,
}

impl RawPostingBlockMeta {
    /// Previous document id before the first posting in this block.
    pub fn base_doc_id(self) -> DocId {
        self.base_doc_id
    }

    /// Last document id encoded in this block.
    pub fn last_doc_id(self) -> DocId {
        self.last_doc_id
    }

    /// Absolute byte offset of this block's encoded postings payload.
    pub fn postings_offset(self) -> u64 {
        self.postings_offset
    }

    /// Byte length of this block's encoded postings payload.
    pub fn postings_len(self) -> u32 {
        self.postings_len
    }

    /// Maximum term weight encoded in this block.
    pub fn max_weight(self) -> u32 {
        self.max_weight
    }
}

/// A byte-backed raw segment reader.
#[derive(Debug, Clone, Copy)]
pub struct RawSegment<'a> {
    bytes: &'a [u8],
    meta: RawSegmentMeta,
}

impl<'a> RawSegment<'a> {
    /// Open a raw segment from bytes.
    ///
    /// This validates the envelope and section offsets, but it does not decode
    /// every posting list. Posting bytes are decoded only when a term is read.
    pub fn open(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER_LEN {
            return Err(Error::Truncated { section: "header" });
        }
        let meta = parse_header(bytes)?;

        validate_layout(bytes, meta)?;
        let footer = checked_range(meta.footer_offset, FOOTER_LEN as u64, bytes.len(), "footer")?;
        if &bytes[footer.start..footer.start + FOOTER_MAGIC.len()] != FOOTER_MAGIC {
            return Err(Error::BadFooter);
        }
        let footer_version = read_u32_at(bytes, footer.start + FOOTER_MAGIC.len(), "footer")?;
        if footer_version != VERSION {
            return Err(Error::UnsupportedVersion {
                version: footer_version,
            });
        }

        // With all bytes resident, verify every checksum once at open; the
        // read paths then stay untouched. File-backed opens verify blocks
        // lazily instead (a full pass would defeat out-of-core reads).
        if meta.has_checksums() {
            let (integrity_offset, block_count, integrity_len) = integrity_layout(meta)?;
            let integrity =
                &bytes[checked_range(integrity_offset, integrity_len, bytes.len(), "integrity")?];
            let term_dir_len = (meta.term_count as u64)
                .checked_mul(TERM_ENTRY_LEN as u64)
                .ok_or(Error::SegmentTooLarge)?;
            let term_dir = &bytes[checked_range(
                meta.term_dir_offset,
                term_dir_len,
                bytes.len(),
                "term directory",
            )?];
            let doc_meta_len = (meta.doc_count as u64)
                .checked_mul(DOC_ENTRY_LEN as u64)
                .ok_or(Error::SegmentTooLarge)?;
            let doc_meta = &bytes[checked_range(
                meta.doc_meta_offset,
                doc_meta_len,
                bytes.len(),
                "doc metadata",
            )?];
            let block_dir_start = doc_meta_end(meta)?;
            let block_dir_len = block_count
                .checked_mul(BLOCK_ENTRY_LEN as u64)
                .ok_or(Error::SegmentTooLarge)?;
            let block_dir = &bytes[checked_range(
                block_dir_start,
                block_dir_len,
                bytes.len(),
                "block directory",
            )?];
            verify_directory_checksums(term_dir, doc_meta, block_dir, integrity)?;
        }

        Ok(Self { bytes, meta })
    }

    /// Return segment header metadata.
    pub fn meta(&self) -> RawSegmentMeta {
        self.meta
    }

    /// Number of documents in the segment.
    pub fn num_docs(&self) -> u32 {
        self.meta.doc_count
    }

    /// Average document length in the segment.
    pub fn avg_doc_len(&self) -> f32 {
        self.meta.avg_doc_len()
    }

    /// Document length for a document id, if the id is present.
    pub fn document_len(&self, doc_id: DocId) -> Result<Option<u32>, Error> {
        let mut low = 0u32;
        let mut high = self.meta.doc_count;
        while low < high {
            let mid = low + ((high - low) / 2);
            let offset = self.doc_entry_offset(mid)?;
            let mid_doc_id = read_u32_at(self.bytes, offset, "doc metadata")?;
            match mid_doc_id.cmp(&doc_id) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    return Ok(Some(read_u32_at(self.bytes, offset + 4, "doc metadata")?));
                }
            }
        }
        Ok(None)
    }

    /// Visit all document ids and document lengths in ascending document id order.
    pub fn for_each_document_len(&self, visit: impl FnMut(DocId, u32)) -> Result<(), Error> {
        for_each_document_len_in_doc_meta(self.doc_meta_bytes()?, self.meta.doc_count, visit)
    }

    /// Document frequency for a term id.
    pub fn df(&self, term_id: RawTermId) -> Result<u32, Error> {
        Ok(self.term_entry(term_id)?.map_or(0, |entry| entry.df))
    }

    /// Sum of weights for a term id across this segment.
    pub fn total_weight(&self, term_id: RawTermId) -> Result<u64, Error> {
        Ok(self
            .term_entry(term_id)?
            .map_or(0, |entry| entry.total_weight))
    }

    /// Maximum per-document weight for a term id in this segment.
    pub fn max_weight(&self, term_id: RawTermId) -> Result<u32, Error> {
        Ok(self
            .term_entry(term_id)?
            .map_or(0, |entry| entry.max_weight))
    }

    /// Numeric term ids present in this segment, in ascending order.
    pub fn term_ids(&self) -> Result<Vec<RawTermId>, Error> {
        let mut terms = Vec::with_capacity(self.meta.term_count as usize);
        for index in 0..self.meta.term_count {
            terms.push(self.term_entry_at(index)?.term_id);
        }
        Ok(terms)
    }

    /// Return a lazy posting iterator for a term id.
    pub fn postings(&self, term_id: RawTermId) -> Result<RawPostings<'a>, Error> {
        let Some(entry) = self.term_entry(term_id)? else {
            return Ok(RawPostings::empty(term_id));
        };
        let range = checked_range(
            entry.postings_offset,
            entry.postings_len as u64,
            self.bytes.len(),
            "postings",
        )?;
        let bytes = &self.bytes[range];
        self.verify_entry_postings_bytes(entry, bytes)?;
        Ok(RawPostings {
            term_id,
            bytes,
            remaining: entry.df,
            consumed: 0,
            prev_doc_id: 0,
            index: 0,
            failed: false,
        })
    }

    /// Visit decoded postings for a term id without allocating a postings list.
    pub fn for_each_posting(
        &self,
        term_id: RawTermId,
        visit: impl FnMut(DocId, u32),
    ) -> Result<(), Error> {
        let Some(entry) = self.term_entry(term_id)? else {
            return Ok(());
        };
        self.for_each_posting_in_entry(entry, visit)
    }

    /// Visit decoded postings and their document lengths for a term id.
    ///
    /// This is intended for scorers such as BM25 that need both term frequency
    /// and document length while avoiding an intermediate postings vector.
    pub fn for_each_posting_with_document_len(
        &self,
        term_id: RawTermId,
        mut visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), Error> {
        let Some(entry) = self.term_entry(term_id)? else {
            return Ok(());
        };
        let mut lengths = DocLengthCursor::new(self.doc_meta_bytes()?, self.meta.doc_count);
        let mut lookup_error = None;
        self.for_each_posting_in_entry(entry, |doc_id, weight| {
            if lookup_error.is_some() {
                return;
            }
            match lengths.get(doc_id) {
                Ok(Some(doc_len)) => visit(doc_id, weight, doc_len),
                Ok(None) => {
                    lookup_error = Some(Error::InvalidLayout {
                        reason: "posting doc id has no document metadata",
                    });
                }
                Err(err) => lookup_error = Some(err),
            }
        })?;
        if let Some(err) = lookup_error {
            return Err(err);
        }
        Ok(())
    }

    /// Return posting-block metadata for a term id.
    pub fn posting_blocks(&self, term_id: RawTermId) -> Result<Vec<RawPostingBlockMeta>, Error> {
        let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
            return Ok(Vec::new());
        };
        self.validate_block_directory(block_directory)?;
        let mut blocks = Vec::with_capacity(block_directory.block_count as usize);
        for i in 0..block_directory.block_count {
            blocks.push(self.posting_block_at(entry, block_directory, i)?);
        }
        Ok(blocks)
    }

    /// Return a lazy posting iterator for one encoded block of a term id.
    pub fn posting_block_postings(
        &self,
        term_id: RawTermId,
        block_index: u32,
    ) -> Result<RawPostingBlockPostings<'a>, Error> {
        let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
            return Ok(RawPostingBlockPostings::empty(term_id));
        };
        self.validate_block_directory(block_directory)?;
        let block = self.posting_block_at(entry, block_directory, block_index)?;
        let range = checked_range(
            block.postings_offset,
            block.postings_len as u64,
            self.bytes.len(),
            "postings",
        )?;
        let bytes = &self.bytes[range];
        self.verify_block_checksum(block_directory, block_index, block, bytes)?;
        Ok(RawPostingBlockPostings {
            term_id,
            bytes,
            consumed: 0,
            base_doc_id: block.base_doc_id,
            last_doc_id: block.last_doc_id,
            prev_doc_id: block.base_doc_id,
            index: 0,
            done: false,
            failed: false,
        })
    }

    /// Visit decoded postings and document lengths for one encoded block.
    ///
    /// Missing terms visit nothing. A present term with an out-of-range block
    /// index returns the same layout error as [`Self::posting_block_postings`].
    pub fn for_each_posting_block_with_document_len(
        &self,
        term_id: RawTermId,
        block_index: u32,
        visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), Error> {
        let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
            return Ok(());
        };
        self.validate_block_directory(block_directory)?;
        let block = self.posting_block_at(entry, block_directory, block_index)?;
        let range = checked_range(
            block.postings_offset,
            block.postings_len as u64,
            self.bytes.len(),
            "postings",
        )?;
        let bytes = &self.bytes[range];
        self.verify_block_checksum(block_directory, block_index, block, bytes)?;
        for_each_posting_in_block_with_document_len(
            self.doc_meta_bytes()?,
            self.meta.doc_count,
            entry.term_id,
            bytes,
            block,
            visit,
        )
    }

    /// Candidate documents that contain every term id.
    pub fn candidates_all_terms(&self, query_terms: &[RawTermId]) -> Result<Vec<DocId>, Error> {
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut terms = query_terms.to_vec();
        terms.sort_unstable();
        terms.dedup();
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::with_capacity(terms.len());
        for term_id in terms {
            let Some(entry) = self.term_entry(term_id)? else {
                return Ok(Vec::new());
            };
            if entry.df == 0 {
                return Ok(Vec::new());
            }
            entries.push(entry);
        }
        entries.sort_by_key(|entry| entry.df);

        let mut candidates = self.posting_doc_ids(entries[0])?;
        let mut scratch = Vec::with_capacity(entries.last().map_or(0, |entry| entry.df as usize));
        for entry in entries.into_iter().skip(1) {
            self.posting_doc_ids_into(entry, &mut scratch)?;
            intersect_doc_id_lists_in_place(&mut candidates, &scratch);
            if candidates.is_empty() {
                break;
            }
        }
        Ok(candidates)
    }

    /// Candidate documents that contain at least one term id.
    pub fn candidates_any_terms(&self, query_terms: &[RawTermId]) -> Result<Vec<DocId>, Error> {
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut terms = query_terms.to_vec();
        terms.sort_unstable();
        terms.dedup();

        let mut entries = Vec::with_capacity(terms.len());
        for term_id in terms {
            if let Some(entry) = self.term_entry(term_id)? {
                if entry.df != 0 {
                    entries.push(entry);
                }
            }
        }
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        if entries.len() == 1 {
            return self.posting_doc_ids(entries[0]);
        }

        let dense_limit = crate::dense_scratch_limit(self.meta.doc_count as usize);
        let dense_range = self
            .dense_doc_id_range()?
            .filter(|(_, dense_slots)| *dense_slots <= dense_limit);
        if let Some((dense_base, dense_slots)) = dense_range {
            let mut seen = vec![false; dense_slots];
            for entry in entries {
                for posting in RawPostings::from_entry(self.bytes, entry)? {
                    let (doc_id, _) = posting?;
                    let slot = dense_doc_slot(doc_id, dense_base, dense_slots);
                    seen[slot] = true;
                }
            }
            return Ok(seen
                .into_iter()
                .enumerate()
                .filter_map(|(slot, hit)| hit.then_some(dense_base + slot as DocId))
                .collect());
        }

        entries.sort_by_key(|entry| entry.df);
        let mut entries = entries.into_iter();
        let mut out = self.posting_doc_ids(entries.next().expect("entries is not empty"))?;
        let mut scratch = Vec::new();
        for entry in entries {
            self.posting_doc_ids_into(entry, &mut scratch)?;
            out = union_doc_id_lists(&out, &scratch);
        }
        Ok(out)
    }

    /// Plan disjunctive candidate generation, with bailout for broad queries.
    ///
    /// The selectivity estimate reads only fixed term-directory metadata. Posting
    /// bytes are decoded only when the plan returns concrete candidates.
    pub fn plan_candidates(
        &self,
        query_terms: &[RawTermId],
        cfg: PlannerConfig,
    ) -> Result<CandidatePlan, Error> {
        if query_terms.is_empty() || self.meta.doc_count == 0 {
            return Ok(CandidatePlan::Candidates(Vec::new()));
        }

        let mut terms = query_terms.to_vec();
        terms.sort_unstable();
        terms.dedup();

        let mut df_sum = 0u64;
        for term_id in terms {
            df_sum = df_sum.saturating_add(self.df(term_id)? as u64);
            if df_sum >= cfg.max_candidates as u64 {
                return Ok(CandidatePlan::ScanAll);
            }
        }

        let ratio = (df_sum as f32) / (self.meta.doc_count as f32);
        if ratio > cfg.max_candidate_ratio {
            return Ok(CandidatePlan::ScanAll);
        }

        Ok(CandidatePlan::Candidates(
            self.candidates_any_terms(query_terms)?,
        ))
    }

    /// Return the top `k` documents by sparse inner product over raw `u32` weights.
    ///
    /// Duplicate query terms are accumulated before scoring. Ties are broken by
    /// ascending doc id, matching [`crate::PostingsIndex::top_k_weighted`].
    pub fn top_k_weighted_u32(
        &self,
        query_terms: &[(RawTermId, f32)],
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, Error> {
        if k == 0 || query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let query_terms = normalize_weighted_query_terms(query_terms);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut lists = Vec::with_capacity(query_terms.len());
        let mut total_postings = 0usize;
        for (term_id, query_weight) in query_terms {
            let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
                continue;
            };
            if entry.df == 0 {
                continue;
            }
            total_postings = total_postings.saturating_add(entry.df as usize);
            lists.push((entry, block_directory, query_weight));
        }
        if lists.is_empty() {
            return Ok(Vec::new());
        }

        if lists.len() == 1 {
            let (entry, block_directory, query_weight) = lists[0];
            return self.top_k_single_raw_term(entry, block_directory, query_weight, k);
        }

        let dense_limit = crate::dense_scratch_limit(self.meta.doc_count as usize);
        let dense_range = self
            .dense_doc_id_range()?
            .filter(|(_, dense_slots)| *dense_slots <= dense_limit);
        if let Some((dense_base, dense_slots)) = dense_range {
            let mut scores = vec![0.0; dense_slots];
            let mut touched = Vec::with_capacity(total_postings.min(self.meta.doc_count as usize));
            let contributions_are_nonnegative = lists
                .iter()
                .all(|(_, _, query_weight)| *query_weight >= 0.0 && query_weight.is_finite());

            if contributions_are_nonnegative {
                for (entry, _, query_weight) in lists {
                    self.for_each_posting_in_entry(entry, |doc_id, doc_weight| {
                        let contribution = query_weight * doc_weight as f32;
                        if contribution == 0.0 {
                            return;
                        }
                        let slot = dense_doc_slot(doc_id, dense_base, dense_slots);
                        if scores[slot] == 0.0 {
                            touched.push((doc_id, slot));
                        }
                        scores[slot] += contribution;
                    })?;
                }
            } else {
                let mut seen = vec![false; dense_slots];
                for (entry, _, query_weight) in lists {
                    self.for_each_posting_in_entry(entry, |doc_id, doc_weight| {
                        let contribution = query_weight * doc_weight as f32;
                        if contribution == 0.0 {
                            return;
                        }
                        let slot = dense_doc_slot(doc_id, dense_base, dense_slots);
                        if !seen[slot] {
                            seen[slot] = true;
                            touched.push((doc_id, slot));
                        }
                        scores[slot] += contribution;
                    })?;
                }
            }

            return Ok(crate::top_k_scored_docs(
                touched
                    .into_iter()
                    .map(|(doc_id, slot)| (doc_id, scores[slot])),
                k,
            ));
        }

        let mut scores: HashMap<DocId, f32> =
            HashMap::with_capacity(total_postings.min(self.meta.doc_count as usize));
        for (entry, _, query_weight) in lists {
            self.for_each_posting_in_entry(entry, |doc_id, doc_weight| {
                let contribution = query_weight * doc_weight as f32;
                if contribution != 0.0 {
                    *scores.entry(doc_id).or_insert(0.0) += contribution;
                }
            })?;
        }

        Ok(crate::top_k_scored_docs(scores, k))
    }

    fn term_entry(&self, term_id: RawTermId) -> Result<Option<TermEntry>, Error> {
        let Some(index) = self.term_entry_index(term_id)? else {
            return Ok(None);
        };
        Ok(Some(self.term_entry_at(index)?))
    }

    fn term_entry_with_blocks(
        &self,
        term_id: RawTermId,
    ) -> Result<Option<(TermEntry, TermBlockDirectory)>, Error> {
        let Some(index) = self.term_entry_index(term_id)? else {
            return Ok(None);
        };
        Ok(Some((
            self.term_entry_at(index)?,
            self.term_block_directory_at(index)?,
        )))
    }

    fn term_entry_index(&self, term_id: RawTermId) -> Result<Option<u32>, Error> {
        let mut low = 0u32;
        let mut high = self.meta.term_count;
        while low < high {
            let mid = low + ((high - low) / 2);
            let offset = self.term_entry_offset(mid)?;
            let mid_term_id = read_u64_at(self.bytes, offset, "term directory")?;
            match mid_term_id.cmp(&term_id) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    fn term_entry_at(&self, index: u32) -> Result<TermEntry, Error> {
        let offset = self.term_entry_offset(index)?;
        let entry = TermEntry {
            term_id: read_u64_at(self.bytes, offset, "term directory")?,
            df: read_u32_at(self.bytes, offset + 8, "term directory")?,
            max_weight: read_u32_at(self.bytes, offset + 12, "term directory")?,
            total_weight: read_u64_at(self.bytes, offset + 16, "term directory")?,
            postings_offset: read_u64_at(self.bytes, offset + 24, "term directory")?,
            postings_len: read_u32_at(self.bytes, offset + 32, "term directory")?,
        };
        let postings_end = entry
            .postings_offset
            .checked_add(entry.postings_len as u64)
            .ok_or(Error::InvalidLayout {
                reason: "posting range overflows",
            })?;
        if entry.postings_offset < self.meta.postings_offset
            || postings_end > self.meta.footer_offset
        {
            return Err(Error::InvalidLayout {
                reason: "posting range is outside postings section",
            });
        }
        Ok(entry)
    }

    fn term_block_directory_at(&self, index: u32) -> Result<TermBlockDirectory, Error> {
        let offset = self.term_entry_offset(index)?;
        Ok(TermBlockDirectory {
            block_count: read_u32_at(self.bytes, offset + 36, "term directory")?,
            blocks_offset: read_u64_at(self.bytes, offset + 40, "term directory")?,
        })
    }

    fn validate_block_directory(&self, block_directory: TermBlockDirectory) -> Result<(), Error> {
        let block_dir_end = block_directory
            .blocks_offset
            .checked_add(block_directory.block_count as u64 * BLOCK_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "block range overflows",
            })?;
        let doc_meta_end = doc_meta_end(self.meta)?;
        if block_directory.blocks_offset < doc_meta_end || block_dir_end > self.meta.postings_offset
        {
            return Err(Error::InvalidLayout {
                reason: "block range is outside block directory section",
            });
        }
        Ok(())
    }

    fn posting_block_at(
        &self,
        entry: TermEntry,
        block_directory: TermBlockDirectory,
        index: u32,
    ) -> Result<RawPostingBlockMeta, Error> {
        if index >= block_directory.block_count {
            return Err(Error::InvalidLayout {
                reason: "block index out of range",
            });
        }
        let offset = block_directory
            .blocks_offset
            .checked_add(index as u64 * BLOCK_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "block entry offset overflows",
            })?;
        let offset = usize::try_from(offset).map_err(|_| Error::SegmentTooLarge)?;
        let block = RawPostingBlockMeta {
            base_doc_id: read_u32_at(self.bytes, offset, "block directory")?,
            last_doc_id: read_u32_at(self.bytes, offset + 4, "block directory")?,
            postings_offset: read_u64_at(self.bytes, offset + 8, "block directory")?,
            postings_len: read_u32_at(self.bytes, offset + 16, "block directory")?,
            max_weight: read_u32_at(self.bytes, offset + 20, "block directory")?,
        };
        let postings_end = block
            .postings_offset
            .checked_add(block.postings_len as u64)
            .ok_or(Error::InvalidLayout {
                reason: "block posting range overflows",
            })?;
        if block.postings_offset < entry.postings_offset
            || postings_end > entry.postings_offset + entry.postings_len as u64
        {
            return Err(Error::InvalidLayout {
                reason: "block posting range is outside term postings",
            });
        }
        if block.last_doc_id < block.base_doc_id {
            return Err(Error::InvalidLayout {
                reason: "block last doc precedes base doc",
            });
        }
        Ok(block)
    }

    fn verify_entry_postings_bytes(&self, entry: TermEntry, bytes: &[u8]) -> Result<(), Error> {
        if !self.meta.has_checksums() {
            return Ok(());
        }

        let Some((_, block_directory)) = self.term_entry_with_blocks(entry.term_id)? else {
            return Err(Error::InvalidLayout {
                reason: "term block directory missing",
            });
        };
        self.validate_block_directory(block_directory)?;
        for block_index in 0..block_directory.block_count {
            let block = self.posting_block_at(entry, block_directory, block_index)?;
            let start = block
                .postings_offset
                .checked_sub(entry.postings_offset)
                .ok_or(Error::InvalidLayout {
                    reason: "block posting range is outside term postings",
                })?;
            let start = usize::try_from(start).map_err(|_| Error::SegmentTooLarge)?;
            let len = usize::try_from(block.postings_len).map_err(|_| Error::SegmentTooLarge)?;
            let end = start.checked_add(len).ok_or(Error::InvalidLayout {
                reason: "block posting range overflows",
            })?;
            let block_bytes = bytes.get(start..end).ok_or(Error::InvalidLayout {
                reason: "block posting range is outside term postings",
            })?;
            self.verify_block_checksum(block_directory, block_index, block, block_bytes)?;
        }
        Ok(())
    }

    fn verify_block_checksum(
        &self,
        block_directory: TermBlockDirectory,
        block_index: u32,
        block: RawPostingBlockMeta,
        bytes: &[u8],
    ) -> Result<(), Error> {
        if !self.meta.has_checksums() {
            return Ok(());
        }
        if block.postings_len as usize != bytes.len() {
            return Err(Error::InvalidLayout {
                reason: "posting block read does not match block bounds",
            });
        }
        let global_index = self.global_block_index(block_directory, block_index)?;
        let stored_crc = self.block_checksum_at(global_index)?;
        if crc32fast::hash(bytes) != stored_crc {
            return Err(Error::ChecksumMismatch {
                section: "posting block",
            });
        }
        Ok(())
    }

    fn global_block_index(
        &self,
        block_directory: TermBlockDirectory,
        block_index: u32,
    ) -> Result<u64, Error> {
        let block_dir_start = doc_meta_end(self.meta)?;
        let rel = block_directory
            .blocks_offset
            .checked_sub(block_dir_start)
            .ok_or(Error::InvalidLayout {
                reason: "block range is outside block directory section",
            })?;
        if rel % BLOCK_ENTRY_LEN as u64 != 0 {
            return Err(Error::InvalidLayout {
                reason: "block directory offset is not entry-aligned",
            });
        }
        rel.checked_div(BLOCK_ENTRY_LEN as u64)
            .and_then(|base| base.checked_add(block_index as u64))
            .ok_or(Error::SegmentTooLarge)
    }

    fn block_checksum_at(&self, global_index: u64) -> Result<u32, Error> {
        let (integrity_offset, block_count, _) = integrity_layout(self.meta)?;
        if global_index >= block_count {
            return Err(Error::InvalidLayout {
                reason: "posting block has no checksum entry",
            });
        }
        let offset = integrity_offset
            .checked_add(INTEGRITY_HEADER_LEN as u64)
            .and_then(|offset| offset.checked_add(global_index.checked_mul(4)?))
            .ok_or(Error::SegmentTooLarge)?;
        let offset = usize::try_from(offset).map_err(|_| Error::SegmentTooLarge)?;
        read_u32_at(self.bytes, offset, "integrity")
    }

    fn term_entry_offset(&self, index: u32) -> Result<usize, Error> {
        if index >= self.meta.term_count {
            return Err(Error::InvalidLayout {
                reason: "term entry index out of range",
            });
        }
        let offset = self
            .meta
            .term_dir_offset
            .checked_add(index as u64 * TERM_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "term entry offset overflows",
            })?;
        usize::try_from(offset).map_err(|_| Error::SegmentTooLarge)
    }

    fn doc_entry_offset(&self, index: u32) -> Result<usize, Error> {
        if index >= self.meta.doc_count {
            return Err(Error::InvalidLayout {
                reason: "doc entry index out of range",
            });
        }
        let offset = self
            .meta
            .doc_meta_offset
            .checked_add(index as u64 * DOC_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "doc entry offset overflows",
            })?;
        usize::try_from(offset).map_err(|_| Error::SegmentTooLarge)
    }

    fn doc_meta_bytes(&self) -> Result<&'a [u8], Error> {
        let len = (self.meta.doc_count as u64)
            .checked_mul(DOC_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "doc metadata length overflows",
            })?;
        let range = checked_range(
            self.meta.doc_meta_offset,
            len,
            self.bytes.len(),
            "doc metadata",
        )?;
        Ok(&self.bytes[range])
    }

    fn dense_doc_id_range(&self) -> Result<Option<(DocId, usize)>, Error> {
        dense_doc_id_range_in_doc_meta(
            self.doc_meta_bytes()?,
            self.meta.doc_count,
            self.meta.max_doc_id,
        )
    }

    fn posting_doc_ids(&self, entry: TermEntry) -> Result<Vec<DocId>, Error> {
        let mut docs = Vec::with_capacity(entry.df as usize);
        self.posting_doc_ids_into(entry, &mut docs)?;
        Ok(docs)
    }

    fn posting_doc_ids_into(&self, entry: TermEntry, docs: &mut Vec<DocId>) -> Result<(), Error> {
        docs.clear();
        self.for_each_posting_in_entry(entry, |doc_id, _| {
            docs.push(doc_id);
        })?;
        Ok(())
    }

    fn for_each_posting_in_entry(
        &self,
        entry: TermEntry,
        mut visit: impl FnMut(DocId, u32),
    ) -> Result<(), Error> {
        let range = checked_range(
            entry.postings_offset,
            entry.postings_len as u64,
            self.bytes.len(),
            "postings",
        )?;
        let bytes = &self.bytes[range];
        self.verify_entry_postings_bytes(entry, bytes)?;
        let mut consumed = 0usize;
        let mut prev_doc_id: DocId = 0;
        for index in 0..entry.df {
            let Some((gap, gap_len)) = varint::decode_u32(&bytes[consumed..]) else {
                return Err(Error::InvalidVarint {
                    term_id: entry.term_id,
                    index,
                });
            };
            consumed += gap_len;
            let Some((weight, weight_len)) = varint::decode_u32(&bytes[consumed..]) else {
                return Err(Error::InvalidVarint {
                    term_id: entry.term_id,
                    index,
                });
            };
            consumed += weight_len;

            let doc_id = if index == 0 {
                gap
            } else {
                if gap == 0 {
                    return Err(Error::NonIncreasingDocId {
                        term_id: entry.term_id,
                        index,
                    });
                }
                prev_doc_id.checked_add(gap).ok_or(Error::DocIdOverflow {
                    term_id: entry.term_id,
                    index,
                })?
            };
            if weight == 0 {
                return Err(Error::ZeroWeight {
                    doc_id,
                    term_id: entry.term_id,
                });
            }

            prev_doc_id = doc_id;
            visit(doc_id, weight);
        }
        if consumed != bytes.len() {
            return Err(Error::TrailingPostingsBytes {
                term_id: entry.term_id,
            });
        }
        Ok(())
    }

    fn top_k_single_raw_term(
        &self,
        entry: TermEntry,
        block_directory: TermBlockDirectory,
        query_weight: f32,
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, Error> {
        if query_weight == 0.0 || k == 0 {
            return Ok(Vec::new());
        }

        self.validate_block_directory(block_directory)?;
        let mut ranked = Vec::with_capacity(k);
        let mut sorted = false;
        let can_prune_blocks = query_weight > 0.0 && query_weight.is_finite();
        for block_index in 0..block_directory.block_count {
            let block = self.posting_block_at(entry, block_directory, block_index)?;
            if can_prune_blocks && ranked.len() == k {
                if !sorted {
                    ranked.sort_by(crate::cmp_doc_scores);
                    sorted = true;
                }
                let threshold = ranked.last().expect("top-k buffer is full").1;
                if query_weight * (block.max_weight as f32) < threshold {
                    continue;
                }
            }

            let range = checked_range(
                block.postings_offset,
                block.postings_len as u64,
                self.bytes.len(),
                "postings",
            )?;
            let bytes = &self.bytes[range];
            self.verify_block_checksum(block_directory, block_index, block, bytes)?;
            for_each_posting_in_block(
                entry.term_id,
                bytes,
                block.base_doc_id,
                block.last_doc_id,
                |doc_id, doc_weight| {
                    push_top_k_doc(
                        &mut ranked,
                        &mut sorted,
                        (doc_id, query_weight * doc_weight as f32),
                        k,
                    );
                },
            )?;
        }
        if !sorted {
            ranked.sort_by(crate::cmp_doc_scores);
        }
        Ok(ranked)
    }
}

/// File-backed raw segment reader.
///
/// This reader keeps the term, document, and block directories resident but
/// range-reads posting payloads from the file on demand.
#[derive(Debug)]
pub struct RawSegmentFile {
    file: File,
    file_len: usize,
    meta: RawSegmentMeta,
    term_dir: Vec<u8>,
    doc_meta: Vec<u8>,
    block_dir: Vec<u8>,
    block_checksums: Option<BlockChecksums>,
}

#[derive(Debug)]
struct BlockChecksums {
    encoded: Vec<u8>,
    decoded: Option<Vec<RawBlockChecksum>>,
}

impl BlockChecksums {
    fn new(integrity: &[u8]) -> Self {
        Self {
            encoded: integrity[INTEGRITY_HEADER_LEN..].to_vec(),
            decoded: None,
        }
    }

    fn get(&mut self, block_dir: &[u8]) -> Result<&[RawBlockChecksum], Error> {
        if self.decoded.is_none() {
            self.decoded = Some(block_checksum_list(block_dir, &self.encoded)?);
        }
        Ok(self.decoded.as_deref().expect("decoded above"))
    }
}

#[derive(Debug, Clone, Copy)]
struct RawBlockChecksum {
    offset: u64,
    len: u32,
    crc: u32,
}

impl RawSegmentFile {
    /// Open a raw segment file from a path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RawSegmentFileError> {
        Self::from_file(File::open(path)?)
    }

    /// Open a raw segment from an already-open file handle.
    pub fn from_file(mut file: File) -> Result<Self, RawSegmentFileError> {
        let file_len_u64 = file.metadata()?.len();
        let file_len = usize::try_from(file_len_u64).map_err(|_| Error::SegmentTooLarge)?;
        if file_len < HEADER_LEN {
            return Err(Error::Truncated { section: "header" }.into());
        }
        let header = read_exact_at(&mut file, 0, HEADER_LEN as u64)?;
        let meta = parse_header(&header)?;
        validate_layout_len(meta, file_len)?;

        let footer = read_exact_at(&mut file, meta.footer_offset, FOOTER_LEN as u64)?;
        if &footer[..FOOTER_MAGIC.len()] != FOOTER_MAGIC {
            return Err(Error::BadFooter.into());
        }
        let footer_version = read_u32_at(&footer, FOOTER_MAGIC.len(), "footer")?;
        if footer_version != VERSION {
            return Err(Error::UnsupportedVersion {
                version: footer_version,
            }
            .into());
        }

        let term_dir_len = (meta.term_count as u64)
            .checked_mul(TERM_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "term directory length overflows",
            })?;
        let doc_meta_len = (meta.doc_count as u64)
            .checked_mul(DOC_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "doc metadata length overflows",
            })?;
        let block_dir_start = doc_meta_end(meta)?;
        let block_dir_len =
            meta.postings_offset
                .checked_sub(block_dir_start)
                .ok_or(Error::InvalidLayout {
                    reason: "block directory must follow doc metadata",
                })?;

        let term_dir = read_exact_at(&mut file, meta.term_dir_offset, term_dir_len)?;
        let doc_meta = read_exact_at(&mut file, meta.doc_meta_offset, doc_meta_len)?;
        let block_dir = read_exact_at(&mut file, block_dir_start, block_dir_len)?;

        // Directories are resident, so verify them at open; posting blocks
        // verify lazily per range read via the checksum map.
        let block_checksums = if meta.has_checksums() {
            let (integrity_offset, block_count, integrity_len) = integrity_layout(meta)?;
            let integrity = read_exact_at(&mut file, integrity_offset, integrity_len)?;
            verify_directory_checksums(&term_dir, &doc_meta, &block_dir, &integrity)?;
            let block_count = usize::try_from(block_count).map_err(|_| Error::SegmentTooLarge)?;
            if integrity.len() != INTEGRITY_HEADER_LEN + block_count * 4 {
                return Err(Error::InvalidLayout {
                    reason: "integrity section length mismatch",
                }
                .into());
            }
            Some(BlockChecksums::new(&integrity))
        } else {
            None
        };

        Ok(Self {
            file,
            file_len,
            meta,
            term_dir,
            doc_meta,
            block_dir,
            block_checksums,
        })
    }

    /// Return segment header metadata.
    pub fn meta(&self) -> RawSegmentMeta {
        self.meta
    }

    /// Number of documents in the segment.
    pub fn num_docs(&self) -> u32 {
        self.meta.doc_count
    }

    /// Average document length in the segment.
    pub fn avg_doc_len(&self) -> f32 {
        self.meta.avg_doc_len()
    }

    /// Document length for a document id, if the id is present.
    pub fn document_len(&self, doc_id: DocId) -> Result<Option<u32>, Error> {
        document_len_in_doc_meta(&self.doc_meta, self.meta.doc_count, doc_id)
    }

    /// Visit all document ids and document lengths in ascending document id order.
    pub fn for_each_document_len(&self, visit: impl FnMut(DocId, u32)) -> Result<(), Error> {
        for_each_document_len_in_doc_meta(&self.doc_meta, self.meta.doc_count, visit)
    }

    /// Document frequency for a term id.
    pub fn df(&self, term_id: RawTermId) -> Result<u32, Error> {
        Ok(self.term_entry(term_id)?.map_or(0, |entry| entry.df))
    }

    /// Sum of weights for a term id across this segment.
    pub fn total_weight(&self, term_id: RawTermId) -> Result<u64, Error> {
        Ok(self
            .term_entry(term_id)?
            .map_or(0, |entry| entry.total_weight))
    }

    /// Maximum per-document weight for a term id in this segment.
    pub fn max_weight(&self, term_id: RawTermId) -> Result<u32, Error> {
        Ok(self
            .term_entry(term_id)?
            .map_or(0, |entry| entry.max_weight))
    }

    /// Numeric term ids present in this segment, in ascending order.
    pub fn term_ids(&self) -> Result<Vec<RawTermId>, Error> {
        let mut terms = Vec::with_capacity(self.meta.term_count as usize);
        for index in 0..self.meta.term_count {
            terms.push(self.term_entry_at(index)?.term_id);
        }
        Ok(terms)
    }

    /// Return decoded postings for a term id, range-reading only that term's payload.
    pub fn postings(
        &mut self,
        term_id: RawTermId,
    ) -> Result<Vec<(DocId, u32)>, RawSegmentFileError> {
        let Some(entry) = self.term_entry(term_id)? else {
            return Ok(Vec::new());
        };
        if entry.postings_len as u64 > FILE_FULL_POSTINGS_READ_LIMIT {
            let mut out = Vec::with_capacity(entry.df as usize);
            self.for_each_posting_in_entry_blocks(entry, |doc_id, weight| {
                out.push((doc_id, weight));
            })?;
            return Ok(out);
        }

        let bytes = self.read_postings_range(entry.postings_offset, entry.postings_len as u64)?;
        let mut out = Vec::with_capacity(entry.df as usize);
        let postings = RawPostings {
            term_id,
            bytes: &bytes,
            remaining: entry.df,
            consumed: 0,
            prev_doc_id: 0,
            index: 0,
            failed: false,
        };
        for posting in postings {
            out.push(posting?);
        }
        Ok(out)
    }

    /// Visit decoded postings for a term id without allocating a postings list.
    ///
    /// This uses the same range-read strategy as [`Self::postings`]: small term
    /// payloads stay on the single-read fast path, while large payloads are read
    /// through the segment's block directory.
    pub fn for_each_posting(
        &mut self,
        term_id: RawTermId,
        visit: impl FnMut(DocId, u32),
    ) -> Result<(), RawSegmentFileError> {
        let Some(entry) = self.term_entry(term_id)? else {
            return Ok(());
        };
        self.for_each_posting_in_entry(entry, visit)
    }

    /// Visit decoded postings and their document lengths for a term id.
    ///
    /// This is intended for out-of-core scorers such as BM25 that need document
    /// lengths but should not materialize the full decoded postings list.
    pub fn for_each_posting_with_document_len(
        &mut self,
        term_id: RawTermId,
        visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), RawSegmentFileError> {
        let Some(entry) = self.term_entry(term_id)? else {
            return Ok(());
        };
        self.for_each_posting_in_entry_with_document_len(entry, visit)
    }

    /// Return posting-block metadata for a term id.
    pub fn posting_blocks(&self, term_id: RawTermId) -> Result<Vec<RawPostingBlockMeta>, Error> {
        let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
            return Ok(Vec::new());
        };
        self.validate_block_directory(block_directory)?;
        let mut blocks = Vec::with_capacity(block_directory.block_count as usize);
        for i in 0..block_directory.block_count {
            blocks.push(self.posting_block_at(entry, block_directory, i)?);
        }
        Ok(blocks)
    }

    /// Return decoded postings for one block of a term id.
    pub fn posting_block_postings(
        &mut self,
        term_id: RawTermId,
        block_index: u32,
    ) -> Result<Vec<(DocId, u32)>, RawSegmentFileError> {
        let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
            return Ok(Vec::new());
        };
        self.validate_block_directory(block_directory)?;
        let block = self.posting_block_at(entry, block_directory, block_index)?;
        let bytes =
            self.read_posting_block_range(block.postings_offset, block.postings_len as u64)?;
        let mut out = Vec::new();
        for_each_posting_in_block(
            entry.term_id,
            &bytes,
            block.base_doc_id,
            block.last_doc_id,
            |doc_id, weight| out.push((doc_id, weight)),
        )?;
        Ok(out)
    }

    /// Visit decoded postings and document lengths for one encoded block.
    ///
    /// Missing terms visit nothing. A present term with an out-of-range block
    /// index returns the same layout error as [`Self::posting_block_postings`].
    pub fn for_each_posting_block_with_document_len(
        &mut self,
        term_id: RawTermId,
        block_index: u32,
        visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), RawSegmentFileError> {
        let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
            return Ok(());
        };
        self.validate_block_directory(block_directory)?;
        let block = self.posting_block_at(entry, block_directory, block_index)?;
        let bytes =
            self.read_posting_block_range(block.postings_offset, block.postings_len as u64)?;
        for_each_posting_in_block_with_document_len(
            &self.doc_meta,
            self.meta.doc_count,
            entry.term_id,
            &bytes,
            block,
            visit,
        )
        .map_err(Into::into)
    }

    /// Candidate documents that contain every term id.
    pub fn candidates_all_terms(
        &mut self,
        query_terms: &[RawTermId],
    ) -> Result<Vec<DocId>, RawSegmentFileError> {
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut terms = query_terms.to_vec();
        terms.sort_unstable();
        terms.dedup();
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::with_capacity(terms.len());
        for term_id in terms {
            let Some(entry) = self.term_entry(term_id)? else {
                return Ok(Vec::new());
            };
            if entry.df == 0 {
                return Ok(Vec::new());
            }
            entries.push(entry);
        }
        entries.sort_by_key(|entry| entry.df);

        let mut candidates = self.posting_doc_ids(entries[0])?;
        let mut scratch = Vec::with_capacity(entries.last().map_or(0, |entry| entry.df as usize));
        for entry in entries.into_iter().skip(1) {
            self.posting_doc_ids_into(entry, &mut scratch)?;
            intersect_doc_id_lists_in_place(&mut candidates, &scratch);
            if candidates.is_empty() {
                break;
            }
        }
        Ok(candidates)
    }

    /// Candidate documents that contain at least one term id.
    pub fn candidates_any_terms(
        &mut self,
        query_terms: &[RawTermId],
    ) -> Result<Vec<DocId>, RawSegmentFileError> {
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut terms = query_terms.to_vec();
        terms.sort_unstable();
        terms.dedup();

        let mut entries = Vec::with_capacity(terms.len());
        for term_id in terms {
            if let Some(entry) = self.term_entry(term_id)? {
                if entry.df != 0 {
                    entries.push(entry);
                }
            }
        }
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        if entries.len() == 1 {
            return self.posting_doc_ids(entries[0]);
        }

        let dense_limit = crate::dense_scratch_limit(self.meta.doc_count as usize);
        let dense_range = self
            .dense_doc_id_range()?
            .filter(|(_, dense_slots)| *dense_slots <= dense_limit);
        if let Some((dense_base, dense_slots)) = dense_range {
            let mut seen = vec![false; dense_slots];
            for entry in entries {
                self.for_each_posting_in_entry(entry, |doc_id, _| {
                    let slot = dense_doc_slot(doc_id, dense_base, dense_slots);
                    seen[slot] = true;
                })?;
            }
            return Ok(seen
                .into_iter()
                .enumerate()
                .filter_map(|(slot, hit)| hit.then_some(dense_base + slot as DocId))
                .collect());
        }

        entries.sort_by_key(|entry| entry.df);
        let mut entries = entries.into_iter();
        let mut out = self.posting_doc_ids(entries.next().expect("entries is not empty"))?;
        let mut scratch = Vec::new();
        for entry in entries {
            self.posting_doc_ids_into(entry, &mut scratch)?;
            out = union_doc_id_lists(&out, &scratch);
        }
        Ok(out)
    }

    /// Plan disjunctive candidate generation, with bailout for broad queries.
    ///
    /// The selectivity estimate reads only fixed term-directory metadata.
    /// Posting payloads are range-read only when the plan returns concrete
    /// candidates.
    pub fn plan_candidates(
        &mut self,
        query_terms: &[RawTermId],
        cfg: PlannerConfig,
    ) -> Result<CandidatePlan, RawSegmentFileError> {
        if query_terms.is_empty() || self.meta.doc_count == 0 {
            return Ok(CandidatePlan::Candidates(Vec::new()));
        }

        let mut terms = query_terms.to_vec();
        terms.sort_unstable();
        terms.dedup();

        let mut df_sum = 0u64;
        for term_id in terms {
            df_sum = df_sum.saturating_add(self.df(term_id)? as u64);
            if df_sum >= cfg.max_candidates as u64 {
                return Ok(CandidatePlan::ScanAll);
            }
        }

        let ratio = (df_sum as f32) / (self.meta.doc_count as f32);
        if ratio > cfg.max_candidate_ratio {
            return Ok(CandidatePlan::ScanAll);
        }

        Ok(CandidatePlan::Candidates(
            self.candidates_any_terms(query_terms)?,
        ))
    }

    /// Return the top `k` documents by sparse inner product over raw `u32` weights.
    ///
    /// This matches [`RawSegment::top_k_weighted_u32`] while keeping fixed
    /// directories in memory and range-reading only posting payloads needed by
    /// the query.
    pub fn top_k_weighted_u32(
        &mut self,
        query_terms: &[(RawTermId, f32)],
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, RawSegmentFileError> {
        if k == 0 || query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let query_terms = normalize_weighted_query_terms(query_terms);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut lists = Vec::with_capacity(query_terms.len());
        let mut total_postings = 0usize;
        let mut has_large_blocked_list = false;
        for (term_id, query_weight) in query_terms {
            let Some((entry, block_directory)) = self.term_entry_with_blocks(term_id)? else {
                continue;
            };
            if entry.df == 0 {
                continue;
            }
            total_postings = total_postings.saturating_add(entry.df as usize);
            has_large_blocked_list |= entry.postings_len as u64 > FILE_FULL_POSTINGS_READ_LIMIT
                && block_directory.block_count > 1;
            lists.push((entry, block_directory, query_weight));
        }
        if lists.is_empty() {
            return Ok(Vec::new());
        }

        if lists.len() == 1 {
            let (entry, block_directory, query_weight) = lists[0];
            return self.top_k_single_raw_term(entry, block_directory, query_weight, k);
        }

        let dense_limit = crate::dense_scratch_limit(self.meta.doc_count as usize);
        let dense_range = self
            .dense_doc_id_range()?
            .filter(|(_, dense_slots)| *dense_slots <= dense_limit);
        if has_large_blocked_list
            && lists
                .iter()
                .all(|(_, _, query_weight)| *query_weight >= 0.0 && query_weight.is_finite())
        {
            return self.top_k_weighted_u32_pruned_blocks(lists, total_postings, dense_range, k);
        }

        if let Some((dense_base, dense_slots)) = dense_range {
            let mut scores = vec![0.0; dense_slots];
            let mut touched = Vec::with_capacity(total_postings.min(self.meta.doc_count as usize));
            let contributions_are_nonnegative = lists
                .iter()
                .all(|(_, _, query_weight)| *query_weight >= 0.0 && query_weight.is_finite());

            if contributions_are_nonnegative {
                for (entry, _, query_weight) in lists {
                    self.for_each_posting_in_entry(entry, |doc_id, doc_weight| {
                        let contribution = query_weight * doc_weight as f32;
                        if contribution == 0.0 {
                            return;
                        }
                        let slot = dense_doc_slot(doc_id, dense_base, dense_slots);
                        if scores[slot] == 0.0 {
                            touched.push((doc_id, slot));
                        }
                        scores[slot] += contribution;
                    })?;
                }
            } else {
                let mut seen = vec![false; dense_slots];
                for (entry, _, query_weight) in lists {
                    self.for_each_posting_in_entry(entry, |doc_id, doc_weight| {
                        let contribution = query_weight * doc_weight as f32;
                        if contribution == 0.0 {
                            return;
                        }
                        let slot = dense_doc_slot(doc_id, dense_base, dense_slots);
                        if !seen[slot] {
                            seen[slot] = true;
                            touched.push((doc_id, slot));
                        }
                        scores[slot] += contribution;
                    })?;
                }
            }

            return Ok(crate::top_k_scored_docs(
                touched
                    .into_iter()
                    .map(|(doc_id, slot)| (doc_id, scores[slot])),
                k,
            ));
        }

        let mut scores: HashMap<DocId, f32> =
            HashMap::with_capacity(total_postings.min(self.meta.doc_count as usize));
        for (entry, _, query_weight) in lists {
            self.for_each_posting_in_entry(entry, |doc_id, doc_weight| {
                let contribution = query_weight * doc_weight as f32;
                if contribution != 0.0 {
                    *scores.entry(doc_id).or_insert(0.0) += contribution;
                }
            })?;
        }

        Ok(crate::top_k_scored_docs(scores, k))
    }

    fn top_k_weighted_u32_pruned_blocks(
        &mut self,
        lists: Vec<(TermEntry, TermBlockDirectory, f32)>,
        total_postings: usize,
        dense_range: Option<(DocId, usize)>,
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, RawSegmentFileError> {
        let scoring_lists = self.prepare_raw_block_scoring_lists(lists)?;
        if let Some((dense_base, dense_slots)) = dense_range {
            return self.top_k_weighted_u32_pruned_blocks_dense(
                &scoring_lists,
                total_postings,
                dense_base,
                dense_slots,
                k,
            );
        }

        self.top_k_weighted_u32_pruned_blocks_sparse(&scoring_lists, total_postings, k)
    }

    fn prepare_raw_block_scoring_lists(
        &mut self,
        lists: Vec<(TermEntry, TermBlockDirectory, f32)>,
    ) -> Result<Vec<RawBlockScoringList>, RawSegmentFileError> {
        let mut scoring_lists = Vec::with_capacity(lists.len());
        for (entry, block_directory, query_weight) in lists {
            self.validate_block_directory(block_directory)?;
            if entry.df != 0 && block_directory.block_count == 0 {
                return Err(Error::InvalidLayout {
                    reason: "nonempty term has no posting blocks",
                }
                .into());
            }

            let mut blocks = Vec::with_capacity(block_directory.block_count as usize);
            for block_index in 0..block_directory.block_count {
                blocks.push(self.posting_block_at(entry, block_directory, block_index)?);
            }

            let full_postings = if entry.postings_len as u64 <= FILE_FULL_POSTINGS_READ_LIMIT {
                Some(self.read_postings_range(entry.postings_offset, entry.postings_len as u64)?)
            } else {
                None
            };
            scoring_lists.push(RawBlockScoringList {
                entry,
                query_weight,
                blocks,
                full_postings,
            });
        }
        Ok(scoring_lists)
    }

    fn top_k_weighted_u32_pruned_blocks_dense(
        &mut self,
        lists: &[RawBlockScoringList],
        total_postings: usize,
        dense_base: DocId,
        dense_slots: usize,
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, RawSegmentFileError> {
        let mut scores = vec![0.0; dense_slots];
        let mut touched = Vec::with_capacity(total_postings.min(self.meta.doc_count as usize));
        let mut threshold = RawTopKThreshold::new(k);
        let mut block_bytes = Vec::new();

        for list in lists {
            for &block in &list.blocks {
                let upper_bound = raw_block_range_upper_bound(block, lists);
                if threshold
                    .threshold()
                    .is_some_and(|threshold| upper_bound < threshold)
                {
                    continue;
                }

                self.for_each_scoring_block_posting(
                    list,
                    block,
                    &mut block_bytes,
                    |doc_id, doc_weight| {
                        let contribution = list.query_weight * doc_weight as f32;
                        if contribution == 0.0 {
                            return;
                        }
                        let slot = dense_doc_slot(doc_id, dense_base, dense_slots);
                        if scores[slot] == 0.0 {
                            touched.push((doc_id, slot));
                        }
                        scores[slot] += contribution;
                        threshold.update(doc_id, scores[slot]);
                    },
                )?;
            }
        }

        Ok(crate::top_k_scored_docs(
            touched
                .into_iter()
                .map(|(doc_id, slot)| (doc_id, scores[slot])),
            k,
        ))
    }

    fn top_k_weighted_u32_pruned_blocks_sparse(
        &mut self,
        lists: &[RawBlockScoringList],
        total_postings: usize,
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, RawSegmentFileError> {
        let mut scores: HashMap<DocId, f32> =
            HashMap::with_capacity(total_postings.min(self.meta.doc_count as usize));
        let mut threshold = RawTopKThreshold::new(k);
        let mut block_bytes = Vec::new();

        for list in lists {
            for &block in &list.blocks {
                let upper_bound = raw_block_range_upper_bound(block, lists);
                if threshold
                    .threshold()
                    .is_some_and(|threshold| upper_bound < threshold)
                {
                    continue;
                }

                self.for_each_scoring_block_posting(
                    list,
                    block,
                    &mut block_bytes,
                    |doc_id, doc_weight| {
                        let contribution = list.query_weight * doc_weight as f32;
                        if contribution != 0.0 {
                            let score = scores.entry(doc_id).or_insert(0.0);
                            *score += contribution;
                            threshold.update(doc_id, *score);
                        }
                    },
                )?;
            }
        }

        Ok(crate::top_k_scored_docs(scores, k))
    }

    fn for_each_scoring_block_posting(
        &mut self,
        list: &RawBlockScoringList,
        block: RawPostingBlockMeta,
        block_bytes: &mut Vec<u8>,
        visit: impl FnMut(DocId, u32),
    ) -> Result<(), RawSegmentFileError> {
        if let Some(postings) = &list.full_postings {
            let start = block
                .postings_offset
                .checked_sub(list.entry.postings_offset)
                .ok_or(Error::InvalidLayout {
                    reason: "block posting range is outside cached postings",
                })?;
            let start = usize::try_from(start).map_err(|_| Error::SegmentTooLarge)?;
            let len = usize::try_from(block.postings_len).map_err(|_| Error::SegmentTooLarge)?;
            let end = start.checked_add(len).ok_or(Error::InvalidLayout {
                reason: "block posting range overflows cached postings",
            })?;
            let bytes = postings.get(start..end).ok_or(Error::InvalidLayout {
                reason: "block posting range is outside cached postings",
            })?;
            for_each_posting_in_block(
                list.entry.term_id,
                bytes,
                block.base_doc_id,
                block.last_doc_id,
                visit,
            )?;
            return Ok(());
        }

        self.read_posting_block_range_into(
            block.postings_offset,
            block.postings_len as u64,
            block_bytes,
        )?;
        for_each_posting_in_block(
            list.entry.term_id,
            block_bytes,
            block.base_doc_id,
            block.last_doc_id,
            visit,
        )?;
        Ok(())
    }

    fn term_entry(&self, term_id: RawTermId) -> Result<Option<TermEntry>, Error> {
        let Some(index) = self.term_entry_index(term_id)? else {
            return Ok(None);
        };
        Ok(Some(self.term_entry_at(index)?))
    }

    fn term_entry_with_blocks(
        &self,
        term_id: RawTermId,
    ) -> Result<Option<(TermEntry, TermBlockDirectory)>, Error> {
        let Some(index) = self.term_entry_index(term_id)? else {
            return Ok(None);
        };
        Ok(Some((
            self.term_entry_at(index)?,
            self.term_block_directory_at(index)?,
        )))
    }

    fn term_entry_index(&self, term_id: RawTermId) -> Result<Option<u32>, Error> {
        let mut low = 0u32;
        let mut high = self.meta.term_count;
        while low < high {
            let mid = low + ((high - low) / 2);
            let offset = self.term_entry_offset(mid)?;
            let mid_term_id = read_u64_at(&self.term_dir, offset, "term directory")?;
            match mid_term_id.cmp(&term_id) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    fn term_entry_at(&self, index: u32) -> Result<TermEntry, Error> {
        let offset = self.term_entry_offset(index)?;
        let entry = TermEntry {
            term_id: read_u64_at(&self.term_dir, offset, "term directory")?,
            df: read_u32_at(&self.term_dir, offset + 8, "term directory")?,
            max_weight: read_u32_at(&self.term_dir, offset + 12, "term directory")?,
            total_weight: read_u64_at(&self.term_dir, offset + 16, "term directory")?,
            postings_offset: read_u64_at(&self.term_dir, offset + 24, "term directory")?,
            postings_len: read_u32_at(&self.term_dir, offset + 32, "term directory")?,
        };
        let postings_end = entry
            .postings_offset
            .checked_add(entry.postings_len as u64)
            .ok_or(Error::InvalidLayout {
                reason: "posting range overflows",
            })?;
        if entry.postings_offset < self.meta.postings_offset
            || postings_end > self.meta.footer_offset
        {
            return Err(Error::InvalidLayout {
                reason: "posting range is outside postings section",
            });
        }
        Ok(entry)
    }

    fn term_block_directory_at(&self, index: u32) -> Result<TermBlockDirectory, Error> {
        let offset = self.term_entry_offset(index)?;
        Ok(TermBlockDirectory {
            block_count: read_u32_at(&self.term_dir, offset + 36, "term directory")?,
            blocks_offset: read_u64_at(&self.term_dir, offset + 40, "term directory")?,
        })
    }

    fn validate_block_directory(&self, block_directory: TermBlockDirectory) -> Result<(), Error> {
        let block_dir_end = block_directory
            .blocks_offset
            .checked_add(block_directory.block_count as u64 * BLOCK_ENTRY_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "block range overflows",
            })?;
        let block_dir_start = doc_meta_end(self.meta)?;
        if block_directory.blocks_offset < block_dir_start
            || block_dir_end > self.meta.postings_offset
        {
            return Err(Error::InvalidLayout {
                reason: "block range is outside block directory section",
            });
        }
        Ok(())
    }

    fn posting_block_at(
        &self,
        entry: TermEntry,
        block_directory: TermBlockDirectory,
        index: u32,
    ) -> Result<RawPostingBlockMeta, Error> {
        if index >= block_directory.block_count {
            return Err(Error::InvalidLayout {
                reason: "block index out of range",
            });
        }
        let block_dir_start = doc_meta_end(self.meta)?;
        let offset = block_directory
            .blocks_offset
            .checked_sub(block_dir_start)
            .and_then(|offset| offset.checked_add(index as u64 * BLOCK_ENTRY_LEN as u64))
            .ok_or(Error::InvalidLayout {
                reason: "block entry offset overflows",
            })?;
        let offset = usize::try_from(offset).map_err(|_| Error::SegmentTooLarge)?;
        let block = RawPostingBlockMeta {
            base_doc_id: read_u32_at(&self.block_dir, offset, "block directory")?,
            last_doc_id: read_u32_at(&self.block_dir, offset + 4, "block directory")?,
            postings_offset: read_u64_at(&self.block_dir, offset + 8, "block directory")?,
            postings_len: read_u32_at(&self.block_dir, offset + 16, "block directory")?,
            max_weight: read_u32_at(&self.block_dir, offset + 20, "block directory")?,
        };
        let postings_end = block
            .postings_offset
            .checked_add(block.postings_len as u64)
            .ok_or(Error::InvalidLayout {
                reason: "block posting range overflows",
            })?;
        if block.postings_offset < entry.postings_offset
            || postings_end > entry.postings_offset + entry.postings_len as u64
        {
            return Err(Error::InvalidLayout {
                reason: "block posting range is outside term postings",
            });
        }
        if block.last_doc_id < block.base_doc_id {
            return Err(Error::InvalidLayout {
                reason: "block last doc precedes base doc",
            });
        }
        Ok(block)
    }

    fn term_entry_offset(&self, index: u32) -> Result<usize, Error> {
        if index >= self.meta.term_count {
            return Err(Error::InvalidLayout {
                reason: "term entry index out of range",
            });
        }
        let offset = index as u64 * TERM_ENTRY_LEN as u64;
        usize::try_from(offset).map_err(|_| Error::SegmentTooLarge)
    }

    fn read_postings_range(
        &mut self,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, RawSegmentFileError> {
        checked_range(offset, len, self.file_len, "postings")?;
        let bytes = read_exact_at_positional(&mut self.file, offset, len)?;
        if let Some(checksums) = self.decoded_block_checksums()? {
            verify_span_blocks(checksums, offset, &bytes)?;
        }
        Ok(bytes)
    }

    fn read_posting_block_range(
        &mut self,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, RawSegmentFileError> {
        let mut bytes = Vec::new();
        self.read_posting_block_range_into(offset, len, &mut bytes)?;
        Ok(bytes)
    }

    fn read_posting_block_range_into(
        &mut self,
        offset: u64,
        len: u64,
        bytes: &mut Vec<u8>,
    ) -> Result<(), RawSegmentFileError> {
        checked_range(offset, len, self.file_len, "postings")?;
        let len = usize::try_from(len).map_err(|_| Error::SegmentTooLarge)?;
        bytes.clear();
        bytes.resize(len, 0);
        read_exact_at_positional_into(&mut self.file, offset, bytes)?;
        if let Some(checksums) = self.decoded_block_checksums()? {
            verify_block_slice(checksums, offset, bytes)?;
        }
        Ok(())
    }

    fn decoded_block_checksums(&mut self) -> Result<Option<&[RawBlockChecksum]>, Error> {
        let Some(checksums) = &mut self.block_checksums else {
            return Ok(None);
        };
        Ok(Some(checksums.get(&self.block_dir)?))
    }

    fn dense_doc_id_range(&self) -> Result<Option<(DocId, usize)>, Error> {
        dense_doc_id_range_in_doc_meta(&self.doc_meta, self.meta.doc_count, self.meta.max_doc_id)
    }

    fn posting_doc_ids(&mut self, entry: TermEntry) -> Result<Vec<DocId>, RawSegmentFileError> {
        let mut docs = Vec::with_capacity(entry.df as usize);
        self.posting_doc_ids_into(entry, &mut docs)?;
        Ok(docs)
    }

    fn posting_doc_ids_into(
        &mut self,
        entry: TermEntry,
        docs: &mut Vec<DocId>,
    ) -> Result<(), RawSegmentFileError> {
        docs.clear();
        self.for_each_posting_in_entry(entry, |doc_id, _| {
            docs.push(doc_id);
        })?;
        Ok(())
    }

    fn for_each_posting_in_entry(
        &mut self,
        entry: TermEntry,
        mut visit: impl FnMut(DocId, u32),
    ) -> Result<(), RawSegmentFileError> {
        if entry.postings_len as u64 > FILE_FULL_POSTINGS_READ_LIMIT {
            return self.for_each_posting_in_entry_blocks(entry, visit);
        }

        let bytes = self.read_postings_range(entry.postings_offset, entry.postings_len as u64)?;
        let postings = RawPostings {
            term_id: entry.term_id,
            bytes: &bytes,
            remaining: entry.df,
            consumed: 0,
            prev_doc_id: 0,
            index: 0,
            failed: false,
        };
        for posting in postings {
            let (doc_id, weight) = posting?;
            visit(doc_id, weight);
        }
        Ok(())
    }

    fn for_each_posting_in_entry_blocks(
        &mut self,
        entry: TermEntry,
        mut visit: impl FnMut(DocId, u32),
    ) -> Result<(), RawSegmentFileError> {
        let Some((_, block_directory)) = self.term_entry_with_blocks(entry.term_id)? else {
            return Err(Error::InvalidLayout {
                reason: "term block directory missing",
            }
            .into());
        };
        self.validate_block_directory(block_directory)?;
        if entry.df != 0 && block_directory.block_count == 0 {
            return Err(Error::InvalidLayout {
                reason: "nonempty term has no posting blocks",
            }
            .into());
        }

        let mut decoded = 0u32;
        let mut block_bytes = Vec::new();
        for block_index in 0..block_directory.block_count {
            let block = self.posting_block_at(entry, block_directory, block_index)?;
            self.read_posting_block_range_into(
                block.postings_offset,
                block.postings_len as u64,
                &mut block_bytes,
            )?;
            for_each_posting_in_block(
                entry.term_id,
                &block_bytes,
                block.base_doc_id,
                block.last_doc_id,
                |doc_id, weight| {
                    decoded = decoded.saturating_add(1);
                    visit(doc_id, weight);
                },
            )?;
        }
        if decoded != entry.df {
            return Err(Error::InvalidLayout {
                reason: "block directory posting count mismatch",
            }
            .into());
        }
        Ok(())
    }

    fn for_each_posting_in_entry_with_document_len(
        &mut self,
        entry: TermEntry,
        mut visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), RawSegmentFileError> {
        if entry.postings_len as u64 > FILE_FULL_POSTINGS_READ_LIMIT {
            return self.for_each_posting_in_entry_blocks_with_document_len(entry, visit);
        }

        let bytes = self.read_postings_range(entry.postings_offset, entry.postings_len as u64)?;
        let mut lengths = DocLengthCursor::new(&self.doc_meta, self.meta.doc_count);
        let postings = RawPostings {
            term_id: entry.term_id,
            bytes: &bytes,
            remaining: entry.df,
            consumed: 0,
            prev_doc_id: 0,
            index: 0,
            failed: false,
        };
        for posting in postings {
            let (doc_id, weight) = posting?;
            let doc_len = lengths.get(doc_id)?.ok_or(Error::InvalidLayout {
                reason: "posting doc id has no document metadata",
            })?;
            visit(doc_id, weight, doc_len);
        }
        Ok(())
    }

    fn for_each_posting_in_entry_blocks_with_document_len(
        &mut self,
        entry: TermEntry,
        mut visit: impl FnMut(DocId, u32, u32),
    ) -> Result<(), RawSegmentFileError> {
        let Some((_, block_directory)) = self.term_entry_with_blocks(entry.term_id)? else {
            return Err(Error::InvalidLayout {
                reason: "term block directory missing",
            }
            .into());
        };
        self.validate_block_directory(block_directory)?;
        if entry.df != 0 && block_directory.block_count == 0 {
            return Err(Error::InvalidLayout {
                reason: "nonempty term has no posting blocks",
            }
            .into());
        }

        let mut blocks = Vec::with_capacity(block_directory.block_count as usize);
        for block_index in 0..block_directory.block_count {
            blocks.push(self.posting_block_at(entry, block_directory, block_index)?);
        }

        let file_len = self.file_len;
        let file = &mut self.file;
        let block_checksums = match &mut self.block_checksums {
            Some(checksums) => Some(checksums.get(&self.block_dir)?),
            None => None,
        };
        let mut decoded = 0u32;
        let mut lengths = DocLengthCursor::new(&self.doc_meta, self.meta.doc_count);
        let mut block_bytes = Vec::new();
        for block in blocks {
            checked_range(
                block.postings_offset,
                block.postings_len as u64,
                file_len,
                "postings",
            )?;
            let len = usize::try_from(block.postings_len).map_err(|_| Error::SegmentTooLarge)?;
            block_bytes.clear();
            block_bytes.resize(len, 0);
            read_exact_at_positional_into(file, block.postings_offset, &mut block_bytes)?;
            if let Some(checksums) = block_checksums {
                verify_block_slice(checksums, block.postings_offset, &block_bytes)?;
            }
            let mut lookup_error = None;
            for_each_posting_in_block(
                entry.term_id,
                &block_bytes,
                block.base_doc_id,
                block.last_doc_id,
                |doc_id, weight| {
                    if lookup_error.is_some() {
                        return;
                    }
                    decoded = decoded.saturating_add(1);
                    match lengths.get(doc_id) {
                        Ok(Some(doc_len)) => visit(doc_id, weight, doc_len),
                        Ok(None) => {
                            lookup_error = Some(Error::InvalidLayout {
                                reason: "posting doc id has no document metadata",
                            });
                        }
                        Err(err) => lookup_error = Some(err),
                    }
                },
            )?;
            if let Some(err) = lookup_error {
                return Err(err.into());
            }
        }
        if decoded != entry.df {
            return Err(Error::InvalidLayout {
                reason: "block directory posting count mismatch",
            }
            .into());
        }
        Ok(())
    }

    fn top_k_single_raw_term(
        &mut self,
        entry: TermEntry,
        block_directory: TermBlockDirectory,
        query_weight: f32,
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, RawSegmentFileError> {
        if query_weight == 0.0 || k == 0 {
            return Ok(Vec::new());
        }

        self.validate_block_directory(block_directory)?;
        let mut ranked = Vec::with_capacity(k);
        let mut sorted = false;
        let can_prune_blocks = query_weight > 0.0 && query_weight.is_finite();
        let mut block_bytes = Vec::new();
        for block_index in 0..block_directory.block_count {
            let block = self.posting_block_at(entry, block_directory, block_index)?;
            if can_prune_blocks && ranked.len() == k {
                if !sorted {
                    ranked.sort_by(crate::cmp_doc_scores);
                    sorted = true;
                }
                let threshold = ranked.last().expect("top-k buffer is full").1;
                if query_weight * (block.max_weight as f32) < threshold {
                    continue;
                }
            }

            self.read_posting_block_range_into(
                block.postings_offset,
                block.postings_len as u64,
                &mut block_bytes,
            )?;
            for_each_posting_in_block(
                entry.term_id,
                &block_bytes,
                block.base_doc_id,
                block.last_doc_id,
                |doc_id, doc_weight| {
                    push_top_k_doc(
                        &mut ranked,
                        &mut sorted,
                        (doc_id, query_weight * doc_weight as f32),
                        k,
                    );
                },
            )?;
        }
        if !sorted {
            ranked.sort_by(crate::cmp_doc_scores);
        }
        Ok(ranked)
    }
}

/// Return the top `k` documents by sparse inner product across raw segment files.
///
/// This composes [`RawSegmentFile::top_k_weighted_u32`] across immutable files
/// and merges the per-file top-k lists. Segment document ids must already be
/// globally unique among live documents; deletes and newer-version masking are
/// owned by the caller's manifest or store layer.
pub fn top_k_weighted_u32_files(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[(RawTermId, f32)],
    k: usize,
) -> Result<Vec<(DocId, f32)>, RawSegmentFileError> {
    if k == 0 || query_terms.is_empty() {
        return Ok(Vec::new());
    }

    let query_terms = normalize_weighted_query_terms(query_terms);
    if query_terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut candidates = Vec::with_capacity(k.saturating_mul(segments.len()));
    let can_prune_segments = query_terms
        .iter()
        .all(|(_, weight)| *weight >= 0.0 && weight.is_finite());

    if can_prune_segments {
        let mut order = Vec::with_capacity(segments.len());
        for (index, segment) in segments.iter().enumerate() {
            let mut upper_bound = 0.0;
            for &(term_id, query_weight) in &query_terms {
                upper_bound += query_weight * segment.max_weight(term_id)? as f32;
            }
            order.push((index, upper_bound));
        }
        order.retain(|(_, upper_bound)| *upper_bound > 0.0 || !upper_bound.is_finite());
        order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut threshold = 0.0;
        for (index, upper_bound) in order {
            if candidates.len() >= k && upper_bound < threshold {
                continue;
            }
            candidates.extend(segments[index].top_k_weighted_u32(&query_terms, k)?);
            if candidates.len() >= k {
                candidates = crate::top_k_scored_docs(candidates, k);
                threshold = candidates.last().map_or(0.0, |(_, score)| *score);
            }
        }
    } else {
        for segment in segments.iter_mut() {
            candidates.extend(segment.top_k_weighted_u32(&query_terms, k)?);
        }
    }

    Ok(crate::top_k_scored_docs(candidates, k))
}

/// Return the top `k` documents by sparse inner product across raw segment files
/// with segment-pruning diagnostics.
///
/// Segment document ids must already be globally unique among live documents;
/// deletes and newer-version masking are owned by the caller's manifest or
/// store layer.
pub fn top_k_weighted_u32_files_with_stats(
    segments: &mut [&mut RawSegmentFile],
    query_terms: &[(RawTermId, f32)],
    k: usize,
) -> Result<RawTopKSearchResult, RawSegmentFileError> {
    // Keep the hits-only sibling separate so diagnostic counters do not tax the
    // default multi-file search hot path.
    let mut stats = RawTopKSearchStats {
        segments_seen: segments.len(),
        ..RawTopKSearchStats::default()
    };
    if k == 0 || query_terms.is_empty() {
        return Ok(RawTopKSearchResult {
            hits: Vec::new(),
            stats,
        });
    }

    let query_terms = normalize_weighted_query_terms(query_terms);
    if query_terms.is_empty() {
        return Ok(RawTopKSearchResult {
            hits: Vec::new(),
            stats,
        });
    }

    let mut candidates = Vec::with_capacity(k.saturating_mul(segments.len()));
    let can_prune_segments = query_terms
        .iter()
        .all(|(_, weight)| *weight >= 0.0 && weight.is_finite());

    if can_prune_segments {
        let mut order = Vec::with_capacity(segments.len());
        for (index, segment) in segments.iter().enumerate() {
            let mut upper_bound = 0.0;
            for &(term_id, query_weight) in &query_terms {
                upper_bound += query_weight * segment.max_weight(term_id)? as f32;
            }
            if upper_bound > 0.0 || !upper_bound.is_finite() {
                order.push((index, upper_bound));
            } else {
                stats.segments_pruned += 1;
            }
        }
        order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut threshold = 0.0;
        for (index, upper_bound) in order {
            if candidates.len() >= k && upper_bound < threshold {
                stats.segments_pruned += 1;
                continue;
            }
            stats.segments_scored += 1;
            candidates.extend(segments[index].top_k_weighted_u32(&query_terms, k)?);
            if candidates.len() >= k {
                candidates = crate::top_k_scored_docs(candidates, k);
                threshold = candidates.last().map_or(0.0, |(_, score)| *score);
            }
        }
    } else {
        for segment in segments.iter_mut() {
            stats.segments_scored += 1;
            candidates.extend(segment.top_k_weighted_u32(&query_terms, k)?);
        }
    }

    Ok(RawTopKSearchResult {
        hits: crate::top_k_scored_docs(candidates, k),
        stats,
    })
}

fn read_exact_at(file: &mut File, offset: u64, len: u64) -> Result<Vec<u8>, RawSegmentFileError> {
    let len = usize::try_from(len).map_err(|_| Error::SegmentTooLarge)?;
    let mut bytes = vec![0; len];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_exact_at_positional(
    file: &mut File,
    offset: u64,
    len: u64,
) -> Result<Vec<u8>, RawSegmentFileError> {
    let len = usize::try_from(len).map_err(|_| Error::SegmentTooLarge)?;
    let mut bytes = vec![0; len];
    read_exact_at_positional_into(file, offset, &mut bytes)?;
    Ok(bytes)
}

#[cfg(unix)]
fn read_exact_at_positional_into(
    file: &mut File,
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(bytes, offset)
}

#[cfg(not(unix))]
fn read_exact_at_positional_into(
    file: &mut File,
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(bytes)
}

struct DocLengthCursor<'a> {
    doc_meta: &'a [u8],
    doc_count: u32,
    next: u32,
}

impl<'a> DocLengthCursor<'a> {
    fn new(doc_meta: &'a [u8], doc_count: u32) -> Self {
        Self {
            doc_meta,
            doc_count,
            next: 0,
        }
    }

    fn starting_at(doc_meta: &'a [u8], doc_count: u32, min_doc_id: DocId) -> Result<Self, Error> {
        let mut low = 0u32;
        let mut high = doc_count;
        while low < high {
            let mid = low + ((high - low) / 2);
            let offset = doc_meta_offset(mid)?;
            let current = read_u32_at(doc_meta, offset, "doc metadata")?;
            if current < min_doc_id {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(Self {
            doc_meta,
            doc_count,
            next: low,
        })
    }

    fn get(&mut self, doc_id: DocId) -> Result<Option<u32>, Error> {
        while self.next < self.doc_count {
            let offset = doc_meta_offset(self.next)?;
            let current = read_u32_at(self.doc_meta, offset, "doc metadata")?;
            match current.cmp(&doc_id) {
                std::cmp::Ordering::Less => self.next += 1,
                std::cmp::Ordering::Equal => {
                    return Ok(Some(read_u32_at(
                        self.doc_meta,
                        offset + 4,
                        "doc metadata",
                    )?));
                }
                std::cmp::Ordering::Greater => return Ok(None),
            }
        }
        Ok(None)
    }
}

fn document_len_in_doc_meta(
    doc_meta: &[u8],
    doc_count: u32,
    doc_id: DocId,
) -> Result<Option<u32>, Error> {
    let mut low = 0u32;
    let mut high = doc_count;
    while low < high {
        let mid = low + ((high - low) / 2);
        let offset = doc_meta_offset(mid)?;
        let mid_doc_id = read_u32_at(doc_meta, offset, "doc metadata")?;
        match mid_doc_id.cmp(&doc_id) {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
            std::cmp::Ordering::Equal => {
                return Ok(Some(read_u32_at(doc_meta, offset + 4, "doc metadata")?));
            }
        }
    }
    Ok(None)
}

fn dense_doc_id_range_in_doc_meta(
    doc_meta: &[u8],
    doc_count: u32,
    max_doc_id: DocId,
) -> Result<Option<(DocId, usize)>, Error> {
    if doc_count == 0 {
        return Ok(None);
    }

    let min_doc_id = read_u32_at(doc_meta, 0, "doc metadata")?;
    let span = max_doc_id
        .checked_sub(min_doc_id)
        .ok_or(Error::InvalidLayout {
            reason: "max doc id precedes first document metadata",
        })?;
    let dense_slots = usize::try_from(span)
        .ok()
        .and_then(|span| span.checked_add(1))
        .ok_or(Error::SegmentTooLarge)?;
    Ok(Some((min_doc_id, dense_slots)))
}

#[inline]
fn dense_doc_slot(doc_id: DocId, dense_base: DocId, dense_slots: usize) -> usize {
    let slot = doc_id.wrapping_sub(dense_base) as usize;
    debug_assert!(slot < dense_slots);
    slot
}

fn for_each_document_len_in_doc_meta(
    doc_meta: &[u8],
    doc_count: u32,
    mut visit: impl FnMut(DocId, u32),
) -> Result<(), Error> {
    for index in 0..doc_count {
        let offset = doc_meta_offset(index)?;
        let doc_id = read_u32_at(doc_meta, offset, "doc metadata")?;
        let doc_len = read_u32_at(doc_meta, offset + 4, "doc metadata")?;
        visit(doc_id, doc_len);
    }
    Ok(())
}

fn doc_meta_offset(index: u32) -> Result<usize, Error> {
    let offset = (index as u64)
        .checked_mul(DOC_ENTRY_LEN as u64)
        .ok_or(Error::InvalidLayout {
            reason: "doc entry offset overflows",
        })?;
    usize::try_from(offset).map_err(|_| Error::SegmentTooLarge)
}

fn for_each_posting_in_block_with_document_len(
    doc_meta: &[u8],
    doc_count: u32,
    term_id: RawTermId,
    bytes: &[u8],
    block: RawPostingBlockMeta,
    mut visit: impl FnMut(DocId, u32, u32),
) -> Result<(), Error> {
    let mut lengths = DocLengthCursor::starting_at(doc_meta, doc_count, block.base_doc_id)?;
    let mut lookup_error = None;
    for_each_posting_in_block(
        term_id,
        bytes,
        block.base_doc_id,
        block.last_doc_id,
        |doc_id, weight| {
            if lookup_error.is_some() {
                return;
            }
            match lengths.get(doc_id) {
                Ok(Some(doc_len)) => visit(doc_id, weight, doc_len),
                Ok(None) => {
                    lookup_error = Some(Error::InvalidLayout {
                        reason: "posting doc id has no document metadata",
                    });
                }
                Err(err) => lookup_error = Some(err),
            }
        },
    )?;
    if let Some(err) = lookup_error {
        return Err(err);
    }
    Ok(())
}

fn for_each_posting_in_block(
    term_id: RawTermId,
    bytes: &[u8],
    base_doc_id: DocId,
    last_doc_id: DocId,
    mut visit: impl FnMut(DocId, u32),
) -> Result<(), Error> {
    let mut consumed = 0usize;
    let mut prev_doc_id = base_doc_id;
    let mut index = 0u32;
    while consumed < bytes.len() {
        let Some((gap, gap_len)) = varint::decode_u32(&bytes[consumed..]) else {
            return Err(Error::InvalidVarint { term_id, index });
        };
        consumed += gap_len;
        let Some((weight, weight_len)) = varint::decode_u32(&bytes[consumed..]) else {
            return Err(Error::InvalidVarint { term_id, index });
        };
        consumed += weight_len;

        if index == 0 && gap == 0 && base_doc_id != 0 {
            return Err(Error::NonIncreasingDocId { term_id, index });
        }
        if index > 0 && gap == 0 {
            return Err(Error::NonIncreasingDocId { term_id, index });
        }
        let doc_id = prev_doc_id
            .checked_add(gap)
            .ok_or(Error::DocIdOverflow { term_id, index })?;
        if doc_id > last_doc_id {
            return Err(Error::InvalidLayout {
                reason: "block posting exceeds last doc",
            });
        }
        if weight == 0 {
            return Err(Error::ZeroWeight { doc_id, term_id });
        }

        prev_doc_id = doc_id;
        index += 1;
        visit(doc_id, weight);
    }
    if index == 0 || prev_doc_id != last_doc_id {
        return Err(Error::InvalidLayout {
            reason: "block last doc does not match decoded postings",
        });
    }
    Ok(())
}

fn raw_block_range_upper_bound(block: RawPostingBlockMeta, lists: &[RawBlockScoringList]) -> f32 {
    // Use the max contribution from every query term over the same document
    // range. A current-block-only follower bound can understate exact top-k.
    lists
        .iter()
        .map(|list| {
            list.query_weight * max_overlapping_raw_block_weight(&list.blocks, block) as f32
        })
        .sum()
}

fn max_overlapping_raw_block_weight(
    blocks: &[RawPostingBlockMeta],
    target: RawPostingBlockMeta,
) -> u32 {
    let start = blocks.partition_point(|block| block.last_doc_id < target.base_doc_id);
    let mut max_weight = 0u32;
    for block in &blocks[start..] {
        if block.base_doc_id > target.last_doc_id {
            break;
        }
        max_weight = max_weight.max(block.max_weight);
    }
    max_weight
}

fn normalize_weighted_query_terms(query_terms: &[(RawTermId, f32)]) -> Vec<(RawTermId, f32)> {
    let mut terms: Vec<_> = query_terms
        .iter()
        .copied()
        .filter(|(_, weight)| *weight != 0.0)
        .collect();
    terms.sort_unstable_by_key(|(term_id, _)| *term_id);

    let mut normalized = Vec::with_capacity(terms.len());
    for (term_id, weight) in terms {
        if let Some((last_term_id, last_weight)) = normalized.last_mut() {
            if *last_term_id == term_id {
                *last_weight += weight;
                continue;
            }
        }
        normalized.push((term_id, weight));
    }
    normalized.retain(|(_, weight)| *weight != 0.0);
    normalized
}

fn push_top_k_doc(
    ranked: &mut Vec<(DocId, f32)>,
    sorted: &mut bool,
    candidate: (DocId, f32),
    k: usize,
) {
    if candidate.1 == 0.0 || k == 0 {
        return;
    }
    if ranked.len() < k {
        ranked.push(candidate);
        *sorted = false;
        return;
    }
    if !*sorted {
        ranked.sort_by(crate::cmp_doc_scores);
        *sorted = true;
    }
    if crate::cmp_doc_scores(&candidate, ranked.last().expect("top-k buffer is full")).is_lt() {
        let last = ranked.len() - 1;
        ranked[last] = candidate;
        let mut i = last;
        while i > 0 && crate::cmp_doc_scores(&ranked[i], &ranked[i - 1]).is_lt() {
            ranked.swap(i, i - 1);
            i -= 1;
        }
    }
}

/// Lazy iterator over one raw posting list.
#[derive(Debug, Clone)]
pub struct RawPostings<'a> {
    term_id: RawTermId,
    bytes: &'a [u8],
    remaining: u32,
    consumed: usize,
    prev_doc_id: DocId,
    index: u32,
    failed: bool,
}

impl<'a> RawPostings<'a> {
    fn empty(term_id: RawTermId) -> Self {
        Self {
            term_id,
            bytes: &[],
            remaining: 0,
            consumed: 0,
            prev_doc_id: 0,
            index: 0,
            failed: false,
        }
    }

    fn from_entry(bytes: &'a [u8], entry: TermEntry) -> Result<Self, Error> {
        let range = checked_range(
            entry.postings_offset,
            entry.postings_len as u64,
            bytes.len(),
            "postings",
        )?;
        Ok(Self {
            term_id: entry.term_id,
            bytes: &bytes[range],
            remaining: entry.df,
            consumed: 0,
            prev_doc_id: 0,
            index: 0,
            failed: false,
        })
    }
}

impl Iterator for RawPostings<'_> {
    type Item = Result<(DocId, u32), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        if self.remaining == 0 {
            if self.consumed != self.bytes.len() {
                self.failed = true;
                return Some(Err(Error::TrailingPostingsBytes {
                    term_id: self.term_id,
                }));
            }
            return None;
        }

        let Some((gap, gap_len)) = varint::decode_u32(&self.bytes[self.consumed..]) else {
            self.failed = true;
            return Some(Err(Error::InvalidVarint {
                term_id: self.term_id,
                index: self.index,
            }));
        };
        self.consumed += gap_len;
        let Some((weight, weight_len)) = varint::decode_u32(&self.bytes[self.consumed..]) else {
            self.failed = true;
            return Some(Err(Error::InvalidVarint {
                term_id: self.term_id,
                index: self.index,
            }));
        };
        self.consumed += weight_len;

        let doc_id = if self.index == 0 {
            gap
        } else {
            if gap == 0 {
                self.failed = true;
                return Some(Err(Error::NonIncreasingDocId {
                    term_id: self.term_id,
                    index: self.index,
                }));
            }
            match self.prev_doc_id.checked_add(gap) {
                Some(doc_id) => doc_id,
                None => {
                    self.failed = true;
                    return Some(Err(Error::DocIdOverflow {
                        term_id: self.term_id,
                        index: self.index,
                    }));
                }
            }
        };
        if weight == 0 {
            self.failed = true;
            return Some(Err(Error::ZeroWeight {
                doc_id,
                term_id: self.term_id,
            }));
        }

        self.prev_doc_id = doc_id;
        self.index += 1;
        self.remaining -= 1;
        Some(Ok((doc_id, weight)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.failed {
            return (0, Some(0));
        }
        if self.remaining == 0 && self.consumed != self.bytes.len() {
            return (1, Some(1));
        }
        let remaining = self.remaining as usize;
        (remaining, Some(remaining))
    }
}

/// Lazy iterator over one raw posting block.
#[derive(Debug, Clone)]
pub struct RawPostingBlockPostings<'a> {
    term_id: RawTermId,
    bytes: &'a [u8],
    consumed: usize,
    base_doc_id: DocId,
    last_doc_id: DocId,
    prev_doc_id: DocId,
    index: u32,
    done: bool,
    failed: bool,
}

impl<'a> RawPostingBlockPostings<'a> {
    fn empty(term_id: RawTermId) -> Self {
        Self {
            term_id,
            bytes: &[],
            consumed: 0,
            base_doc_id: 0,
            last_doc_id: 0,
            prev_doc_id: 0,
            index: 0,
            done: true,
            failed: false,
        }
    }
}

impl Iterator for RawPostingBlockPostings<'_> {
    type Item = Result<(DocId, u32), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.done {
            return None;
        }
        if self.consumed == self.bytes.len() {
            self.done = true;
            if self.index == 0 || self.prev_doc_id != self.last_doc_id {
                self.failed = true;
                return Some(Err(Error::InvalidLayout {
                    reason: "block last doc does not match decoded postings",
                }));
            }
            return None;
        }

        let Some((gap, gap_len)) = varint::decode_u32(&self.bytes[self.consumed..]) else {
            self.failed = true;
            return Some(Err(Error::InvalidVarint {
                term_id: self.term_id,
                index: self.index,
            }));
        };
        self.consumed += gap_len;
        let Some((weight, weight_len)) = varint::decode_u32(&self.bytes[self.consumed..]) else {
            self.failed = true;
            return Some(Err(Error::InvalidVarint {
                term_id: self.term_id,
                index: self.index,
            }));
        };
        self.consumed += weight_len;

        if self.index == 0 && gap == 0 && self.base_doc_id != 0 {
            self.failed = true;
            return Some(Err(Error::NonIncreasingDocId {
                term_id: self.term_id,
                index: self.index,
            }));
        }
        if self.index > 0 && gap == 0 {
            self.failed = true;
            return Some(Err(Error::NonIncreasingDocId {
                term_id: self.term_id,
                index: self.index,
            }));
        }
        let Some(doc_id) = self.prev_doc_id.checked_add(gap) else {
            self.failed = true;
            return Some(Err(Error::DocIdOverflow {
                term_id: self.term_id,
                index: self.index,
            }));
        };
        if doc_id > self.last_doc_id {
            self.failed = true;
            return Some(Err(Error::InvalidLayout {
                reason: "block posting exceeds last doc",
            }));
        }
        if weight == 0 {
            self.failed = true;
            return Some(Err(Error::ZeroWeight {
                doc_id,
                term_id: self.term_id,
            }));
        }

        self.prev_doc_id = doc_id;
        self.index += 1;
        Some(Ok((doc_id, weight)))
    }
}

/// Encode documents into the first raw `u64` term, `u32` weight segment format.
pub fn write_u64_u32_segment(documents: &[RawDocument<'_>]) -> Result<Vec<u8>, Error> {
    let sections = build_u64_u32_segment_sections(documents)?;
    Ok(raw_segment_sections_to_vec(&sections))
}

/// Encode an iterator of documents into the first raw `u64` term, `u32` weight
/// segment format.
///
/// This avoids a caller-side `Vec<RawDocument>`; the encoder still materializes
/// segment directories and posting payload bytes internally.
pub fn write_u64_u32_segment_from_iter<'a, I>(documents: I) -> Result<Vec<u8>, Error>
where
    I: IntoIterator<Item = RawDocument<'a>>,
{
    let sections = build_u64_u32_segment_sections_from_iter(documents)?;
    Ok(raw_segment_sections_to_vec(&sections))
}

/// Encode an increasing-document-id iterator into the first raw `u64` term,
/// `u32` weight segment format.
///
/// This avoids both a caller-side `Vec<RawDocument>` and the encoder's
/// whole-corpus document map. The encoder still materializes segment
/// directories and posting payload bytes internally. Input document ids must be
/// strictly increasing; duplicate ids return [`Error::DuplicateDocId`] and
/// decreasing ids return [`Error::UnsortedDocId`].
pub fn write_u64_u32_segment_sorted_from_iter<'a, I>(documents: I) -> Result<Vec<u8>, Error>
where
    I: IntoIterator<Item = RawDocument<'a>>,
{
    let sections = build_u64_u32_segment_sections_from_sorted_iter(documents)?;
    Ok(raw_segment_sections_to_vec(&sections))
}

/// Encode sorted document metadata and term-major posting lists into the first
/// raw `u64` term, `u32` weight segment format.
///
/// This is the lowest-level raw writer. It is intended for external sort or
/// spill/merge pipelines that already have term-major postings. Document ids in
/// `document_lengths` must be strictly increasing. Term ids in `terms` must be
/// strictly increasing, and each posting list's document ids must be strictly
/// increasing and present in `document_lengths`.
pub fn write_u64_u32_segment_from_term_postings(
    document_lengths: &[(DocId, u32)],
    terms: &[RawTermPostingList<'_>],
) -> Result<Vec<u8>, Error> {
    let sections = build_u64_u32_segment_sections_from_term_postings(document_lengths, terms)?;
    Ok(raw_segment_sections_to_vec(&sections))
}

fn raw_segment_sections_to_vec(sections: &RawSegmentSections) -> Vec<u8> {
    let mut out = Vec::with_capacity(sections.final_len);
    put_header(&mut out, sections.meta);
    let term_dir_crc = append_term_directory(&mut out, sections);
    let doc_meta_crc = append_document_metadata(&mut out, sections);
    let block_dir_crc = append_block_directory(&mut out, sections);
    out.extend_from_slice(&sections.postings_bytes);
    put_integrity_section(
        &mut out,
        term_dir_crc,
        doc_meta_crc,
        block_dir_crc,
        &sections.block_crcs,
    );
    out.extend_from_slice(FOOTER_MAGIC);
    put_u32(&mut out, VERSION);

    debug_assert_eq!(out.len(), sections.final_len);
    out
}

/// Encode documents into a caller-provided raw segment writer.
///
/// This only emits segment bytes. Callers still own the file path, atomic
/// publication, fsync policy, manifest update, deletes, and compaction.
pub fn write_u64_u32_segment_to<W: Write + ?Sized>(
    documents: &[RawDocument<'_>],
    writer: &mut W,
) -> Result<(), RawSegmentWriteError> {
    let sections = build_u64_u32_segment_sections(documents)?;
    write_raw_segment_sections_to(&sections, writer)?;
    Ok(())
}

/// Encode an iterator of documents into a caller-provided raw segment writer.
///
/// This only emits segment bytes. Callers still own the file path, atomic
/// publication, fsync policy, manifest update, deletes, and compaction.
/// The encoder still materializes segment directories and posting payload bytes
/// internally; use this to avoid a caller-side `Vec<RawDocument>`.
pub fn write_u64_u32_segment_from_iter_to<'a, I, W>(
    documents: I,
    writer: &mut W,
) -> Result<(), RawSegmentWriteError>
where
    I: IntoIterator<Item = RawDocument<'a>>,
    W: Write + ?Sized,
{
    let sections = build_u64_u32_segment_sections_from_iter(documents)?;
    write_raw_segment_sections_to(&sections, writer)?;
    Ok(())
}

/// Encode an increasing-document-id iterator into a caller-provided raw segment
/// writer.
///
/// This only emits segment bytes. Callers still own the file path, atomic
/// publication, fsync policy, manifest update, deletes, and compaction.
/// Compared with [`write_u64_u32_segment_from_iter_to`], this avoids the
/// encoder's whole-corpus document map when the caller can provide strictly
/// increasing document ids.
pub fn write_u64_u32_segment_sorted_from_iter_to<'a, I, W>(
    documents: I,
    writer: &mut W,
) -> Result<(), RawSegmentWriteError>
where
    I: IntoIterator<Item = RawDocument<'a>>,
    W: Write + ?Sized,
{
    let sections = build_u64_u32_segment_sections_from_sorted_iter(documents)?;
    write_raw_segment_sections_to(&sections, writer)?;
    Ok(())
}

/// Encode sorted document metadata and term-major posting lists into a
/// caller-provided raw segment writer.
///
/// This only emits segment bytes. Callers still own the file path, atomic
/// publication, fsync policy, manifest update, deletes, and compaction.
/// Compared with document-major writers, this avoids the encoder's
/// whole-corpus document map and term transposition when the caller can provide
/// already-sorted posting lists.
pub fn write_u64_u32_segment_from_term_postings_to<W: Write + ?Sized>(
    document_lengths: &[(DocId, u32)],
    terms: &[RawTermPostingList<'_>],
    writer: &mut W,
) -> Result<(), RawSegmentWriteError> {
    let plan = build_u64_u32_segment_stream_plan_from_term_postings(document_lengths, terms)?;
    write_raw_segment_stream_plan_to(&plan, terms, writer)?;
    Ok(())
}

/// Encode sorted document metadata and term-major posting lists into a
/// seekable raw segment writer.
///
/// This has the same byte format and validation rules as
/// [`write_u64_u32_segment_from_term_postings_to`]. For local files or other
/// seekable sinks, it reserves the header and directories, streams encoded
/// posting blocks once, then seeks back to publish the metadata. That avoids
/// both the full postings-payload buffer and the generic sink writer's
/// measurement pass.
pub fn write_u64_u32_segment_from_term_postings_seekable_to<W: Write + Seek + ?Sized>(
    document_lengths: &[(DocId, u32)],
    terms: &[RawTermPostingList<'_>],
    writer: &mut W,
) -> Result<(), RawSegmentWriteError> {
    let (doc_entries, total_doc_len, max_doc_id) = validate_raw_document_lengths(document_lengths)?;
    validate_raw_term_posting_lists(&doc_entries, terms)?;
    write_raw_segment_term_postings_seekable_to(
        &doc_entries,
        total_doc_len,
        max_doc_id,
        terms,
        writer,
    )
}

fn build_u64_u32_segment_sections(
    documents: &[RawDocument<'_>],
) -> Result<RawSegmentSections, Error> {
    build_u64_u32_segment_sections_from_docs(collect_raw_documents_from_slice(documents)?)
}

fn build_u64_u32_segment_sections_from_iter<'a, I>(
    documents: I,
) -> Result<RawSegmentSections, Error>
where
    I: IntoIterator<Item = RawDocument<'a>>,
{
    build_u64_u32_segment_sections_from_docs(collect_raw_documents_from_iter(documents)?)
}

fn build_u64_u32_segment_sections_from_sorted_iter<'a, I>(
    documents: I,
) -> Result<RawSegmentSections, Error>
where
    I: IntoIterator<Item = RawDocument<'a>>,
{
    let mut postings: BTreeMap<RawTermId, Vec<(DocId, u32)>> = BTreeMap::new();
    let mut doc_entries = Vec::new();
    let mut total_doc_len = 0u64;
    let mut previous_doc_id = None;

    for doc in documents {
        if let Some(previous) = previous_doc_id {
            match doc.doc_id.cmp(&previous) {
                std::cmp::Ordering::Less => {
                    return Err(Error::UnsortedDocId {
                        previous,
                        current: doc.doc_id,
                    });
                }
                std::cmp::Ordering::Equal => {
                    return Err(Error::DuplicateDocId { doc_id: doc.doc_id });
                }
                std::cmp::Ordering::Greater => {}
            }
        }

        let (doc_len, terms) = collect_raw_document_terms(doc.doc_id, doc.terms)?;
        total_doc_len = total_doc_len.saturating_add(doc_len as u64);
        doc_entries.push((doc.doc_id, doc_len));
        for (term_id, weight) in terms {
            postings
                .entry(term_id)
                .or_default()
                .push((doc.doc_id, weight));
        }
        previous_doc_id = Some(doc.doc_id);
    }

    build_u64_u32_segment_sections_from_parts(
        postings,
        doc_entries,
        total_doc_len,
        previous_doc_id.unwrap_or(0),
    )
}

fn build_u64_u32_segment_sections_from_term_postings(
    document_lengths: &[(DocId, u32)],
    terms: &[RawTermPostingList<'_>],
) -> Result<RawSegmentSections, Error> {
    let (doc_entries, total_doc_len, max_doc_id) = validate_raw_document_lengths(document_lengths)?;
    validate_raw_term_posting_lists(&doc_entries, terms)?;
    build_u64_u32_segment_sections_from_posting_lists(terms, doc_entries, total_doc_len, max_doc_id)
}

fn build_u64_u32_segment_stream_plan_from_term_postings(
    document_lengths: &[(DocId, u32)],
    terms: &[RawTermPostingList<'_>],
) -> Result<RawSegmentStreamPlan, Error> {
    let (doc_entries, total_doc_len, max_doc_id) = validate_raw_document_lengths(document_lengths)?;
    validate_raw_term_posting_lists(&doc_entries, terms)?;
    build_u64_u32_segment_stream_plan_from_posting_lists(
        terms,
        doc_entries,
        total_doc_len,
        max_doc_id,
    )
}

fn write_raw_segment_term_postings_seekable_to<W: Write + Seek + ?Sized>(
    doc_entries: &[(DocId, u32)],
    total_doc_len: u64,
    max_doc_id: DocId,
    terms: &[RawTermPostingList<'_>],
    writer: &mut W,
) -> Result<(), RawSegmentWriteError> {
    let doc_count = u32::try_from(doc_entries.len()).map_err(|_| Error::TooManyDocuments)?;
    let term_count = u32::try_from(terms.len()).map_err(|_| Error::TooManyTerms)?;
    let term_dir_offset = HEADER_LEN as u64;
    let term_dir_len = (term_count as u64)
        .checked_mul(TERM_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let doc_meta_offset = term_dir_offset
        .checked_add(term_dir_len)
        .ok_or(Error::SegmentTooLarge)?;
    let doc_meta_len = (doc_count as u64)
        .checked_mul(DOC_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let block_dir_offset = doc_meta_offset
        .checked_add(doc_meta_len)
        .ok_or(Error::SegmentTooLarge)?;
    let total_block_count = terms.iter().try_fold(0u64, |count, term| {
        count
            .checked_add((term.postings.len() as u64).div_ceil(DEFAULT_BLOCK_SIZE as u64))
            .ok_or(Error::SegmentTooLarge)
    })?;
    let block_dir_len = total_block_count
        .checked_mul(BLOCK_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let postings_offset = block_dir_offset
        .checked_add(block_dir_len)
        .ok_or(Error::SegmentTooLarge)?;

    let segment_start = writer.stream_position()?;
    write_zeroes_to(
        writer,
        (HEADER_LEN as u64)
            .checked_add(term_dir_len)
            .ok_or(Error::SegmentTooLarge)?,
    )?;
    let doc_meta_crc = write_document_metadata_to(writer, doc_entries)?;
    write_zeroes_to(writer, block_dir_len)?;

    let mut term_entries = Vec::with_capacity(term_count as usize);
    let mut term_block_directories = Vec::with_capacity(term_count as usize);
    let mut block_entries = Vec::new();
    let mut block_crcs = Vec::new();
    let mut scratch = Vec::new();
    let mut postings_len_total = 0u64;

    for term in terms {
        let term_postings_offset = postings_offset
            .checked_add(postings_len_total)
            .ok_or(Error::SegmentTooLarge)?;
        let blocks_offset = block_dir_offset
            .checked_add(
                u64::try_from(block_entries.len()).map_err(|_| Error::SegmentTooLarge)?
                    * BLOCK_ENTRY_LEN as u64,
            )
            .ok_or(Error::SegmentTooLarge)?;
        let term_start_len = postings_len_total;
        let mut prev_doc_id = 0;
        let mut first_posting = true;
        let mut max_weight = 0u32;
        let mut total_weight = 0u64;

        for chunk in term.postings.chunks(DEFAULT_BLOCK_SIZE as usize) {
            scratch.clear();
            let block = append_encoded_raw_posting_block(
                chunk,
                &mut first_posting,
                &mut prev_doc_id,
                &mut scratch,
            )?;
            let block_postings_offset = postings_offset
                .checked_add(postings_len_total)
                .ok_or(Error::SegmentTooLarge)?;
            postings_len_total = postings_len_total
                .checked_add(block.postings_len as u64)
                .ok_or(Error::SegmentTooLarge)?;
            max_weight = max_weight.max(block.max_weight);
            total_weight = total_weight.saturating_add(block.total_weight);
            block_entries.push(RawPostingBlockMeta {
                base_doc_id: block.base_doc_id,
                last_doc_id: block.last_doc_id,
                postings_offset: block_postings_offset,
                postings_len: block.postings_len,
                max_weight: block.max_weight,
            });
            block_crcs.push(crc32fast::hash(&scratch));
            writer.write_all(&scratch)?;
        }

        let postings_len = u32::try_from(postings_len_total - term_start_len)
            .map_err(|_| Error::SegmentTooLarge)?;
        term_entries.push(TermEntry {
            term_id: term.term_id,
            df: u32::try_from(term.postings.len()).map_err(|_| Error::TooManyDocuments)?,
            max_weight,
            total_weight,
            postings_offset: term_postings_offset,
            postings_len,
        });
        term_block_directories.push(TermBlockDirectory {
            block_count: u32::try_from(term.postings.len().div_ceil(DEFAULT_BLOCK_SIZE as usize))
                .map_err(|_| Error::TooManyDocuments)?,
            blocks_offset,
        });
    }

    let integrity_offset = postings_offset
        .checked_add(postings_len_total)
        .ok_or(Error::SegmentTooLarge)?;
    let integrity_len = (block_crcs.len() as u64)
        .checked_mul(4)
        .and_then(|crcs| crcs.checked_add(INTEGRITY_HEADER_LEN as u64))
        .ok_or(Error::SegmentTooLarge)?;
    let footer_offset = integrity_offset
        .checked_add(integrity_len)
        .ok_or(Error::SegmentTooLarge)?;
    let final_len = footer_offset
        .checked_add(FOOTER_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let final_len_usize = usize::try_from(final_len).map_err(|_| Error::SegmentTooLarge)?;
    let meta = RawSegmentMeta {
        term_count,
        doc_count,
        max_doc_id,
        block_size: DEFAULT_BLOCK_SIZE,
        total_doc_len,
        term_dir_offset,
        doc_meta_offset,
        postings_offset,
        footer_offset,
        flags: FLAG_CHECKSUMS,
    };

    debug_assert_eq!(writer.stream_position()? - segment_start, integrity_offset);
    writer.seek(SeekFrom::Start(segment_start))?;
    write_header_to(writer, meta)?;
    let term_dir_crc = write_term_directory_to(writer, &term_entries, &term_block_directories)?;
    writer.seek(SeekFrom::Start(segment_position(
        segment_start,
        block_dir_offset,
    )?))?;
    let block_dir_crc = write_block_directory_to(writer, &block_entries)?;
    writer.seek(SeekFrom::Start(segment_position(
        segment_start,
        integrity_offset,
    )?))?;

    let mut integrity = Vec::with_capacity(INTEGRITY_HEADER_LEN + block_crcs.len() * 4);
    put_integrity_section(
        &mut integrity,
        term_dir_crc,
        doc_meta_crc,
        block_dir_crc,
        &block_crcs,
    );
    writer.write_all(&integrity)?;
    writer.write_all(FOOTER_MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    debug_assert_eq!(
        writer.stream_position()? - segment_start,
        final_len_usize as u64
    );
    Ok(())
}

fn validate_raw_document_lengths(
    document_lengths: &[(DocId, u32)],
) -> Result<(RawDocumentEntries, u64, DocId), Error> {
    let mut doc_entries = Vec::with_capacity(document_lengths.len());
    let mut total_doc_len = 0u64;
    let mut previous_doc_id = None;

    for &(doc_id, doc_len) in document_lengths {
        if let Some(previous) = previous_doc_id {
            match doc_id.cmp(&previous) {
                std::cmp::Ordering::Less => {
                    return Err(Error::UnsortedDocId {
                        previous,
                        current: doc_id,
                    });
                }
                std::cmp::Ordering::Equal => return Err(Error::DuplicateDocId { doc_id }),
                std::cmp::Ordering::Greater => {}
            }
        }
        total_doc_len = total_doc_len.saturating_add(doc_len as u64);
        doc_entries.push((doc_id, doc_len));
        previous_doc_id = Some(doc_id);
    }

    Ok((doc_entries, total_doc_len, previous_doc_id.unwrap_or(0)))
}

fn validate_raw_term_posting_lists(
    doc_entries: &[(DocId, u32)],
    terms: &[RawTermPostingList<'_>],
) -> Result<(), Error> {
    let mut previous_term_id = None;
    for term in terms {
        if let Some(previous) = previous_term_id {
            match term.term_id.cmp(&previous) {
                std::cmp::Ordering::Less => {
                    return Err(Error::UnsortedTermId {
                        previous,
                        current: term.term_id,
                    });
                }
                std::cmp::Ordering::Equal => {
                    return Err(Error::DuplicateTermId {
                        term_id: term.term_id,
                    });
                }
                std::cmp::Ordering::Greater => {}
            }
        }
        if term.postings.is_empty() {
            return Err(Error::EmptyPostingList {
                term_id: term.term_id,
            });
        }

        let mut previous_doc_id = None;
        for (index, &(doc_id, weight)) in term.postings.iter().enumerate() {
            if let Some(previous) = previous_doc_id {
                if doc_id <= previous {
                    return Err(Error::NonIncreasingDocId {
                        term_id: term.term_id,
                        index: u32::try_from(index).map_err(|_| Error::TooManyDocuments)?,
                    });
                }
            }
            if weight == 0 {
                return Err(Error::ZeroWeight {
                    doc_id,
                    term_id: term.term_id,
                });
            }
            if doc_entries
                .binary_search_by_key(&doc_id, |&(doc_id, _)| doc_id)
                .is_err()
            {
                return Err(Error::UnknownDocId {
                    term_id: term.term_id,
                    doc_id,
                });
            }
            previous_doc_id = Some(doc_id);
        }
        previous_term_id = Some(term.term_id);
    }
    Ok(())
}

fn collect_raw_documents_from_slice(
    documents: &[RawDocument<'_>],
) -> Result<RawDocumentMap, Error> {
    let mut docs = BTreeMap::new();
    for doc in documents {
        insert_raw_document(&mut docs, doc.doc_id, doc.terms)?;
    }
    Ok(docs)
}

fn collect_raw_documents_from_iter<'a, I>(documents: I) -> Result<RawDocumentMap, Error>
where
    I: IntoIterator<Item = RawDocument<'a>>,
{
    let mut docs = BTreeMap::new();
    for doc in documents {
        insert_raw_document(&mut docs, doc.doc_id, doc.terms)?;
    }
    Ok(docs)
}

fn insert_raw_document(
    docs: &mut RawDocumentMap,
    doc_id: DocId,
    input_terms: &[(RawTermId, u32)],
) -> Result<(), Error> {
    let entry = match docs.entry(doc_id) {
        Entry::Vacant(entry) => entry,
        Entry::Occupied(_) => return Err(Error::DuplicateDocId { doc_id }),
    };

    let (doc_len, terms) = collect_raw_document_terms(doc_id, input_terms)?;
    entry.insert((doc_len, terms));
    Ok(())
}

fn collect_raw_document_terms(
    doc_id: DocId,
    input_terms: &[(RawTermId, u32)],
) -> Result<(u32, BTreeMap<RawTermId, u32>), Error> {
    let mut doc_len = 0u32;
    let mut terms: BTreeMap<RawTermId, u32> = BTreeMap::new();
    for &(term_id, weight) in input_terms {
        if weight == 0 {
            return Err(Error::ZeroWeight { doc_id, term_id });
        }
        doc_len = doc_len
            .checked_add(weight)
            .ok_or(Error::DocLengthOverflow { doc_id })?;
        let accumulated = terms.entry(term_id).or_insert(0);
        *accumulated = accumulated
            .checked_add(weight)
            .ok_or(Error::WeightOverflow { doc_id, term_id })?;
    }

    Ok((doc_len, terms))
}

fn build_u64_u32_segment_sections_from_docs(
    docs: RawDocumentMap,
) -> Result<RawSegmentSections, Error> {
    let mut postings: BTreeMap<RawTermId, Vec<(DocId, u32)>> = BTreeMap::new();
    let mut doc_entries = Vec::with_capacity(docs.len());
    let mut total_doc_len = 0u64;
    let mut max_doc_id = 0;
    for (&doc_id, (doc_len, terms)) in &docs {
        total_doc_len = total_doc_len.saturating_add(*doc_len as u64);
        doc_entries.push((doc_id, *doc_len));
        max_doc_id = doc_id;
        for (&term_id, &weight) in terms {
            postings.entry(term_id).or_default().push((doc_id, weight));
        }
    }

    build_u64_u32_segment_sections_from_parts(postings, doc_entries, total_doc_len, max_doc_id)
}

fn build_u64_u32_segment_sections_from_parts(
    postings: BTreeMap<RawTermId, Vec<(DocId, u32)>>,
    doc_entries: RawDocumentEntries,
    total_doc_len: u64,
    max_doc_id: DocId,
) -> Result<RawSegmentSections, Error> {
    build_u64_u32_segment_sections_from_term_iter(
        postings.len(),
        || {
            postings
                .iter()
                .map(|(&term_id, postings)| (term_id, postings.as_slice()))
        },
        doc_entries,
        total_doc_len,
        max_doc_id,
    )
}

fn build_u64_u32_segment_sections_from_posting_lists(
    posting_lists: &[RawTermPostingList<'_>],
    doc_entries: RawDocumentEntries,
    total_doc_len: u64,
    max_doc_id: DocId,
) -> Result<RawSegmentSections, Error> {
    build_u64_u32_segment_sections_from_term_iter(
        posting_lists.len(),
        || {
            posting_lists
                .iter()
                .map(|term| (term.term_id, term.postings))
        },
        doc_entries,
        total_doc_len,
        max_doc_id,
    )
}

fn build_u64_u32_segment_stream_plan_from_posting_lists(
    posting_lists: &[RawTermPostingList<'_>],
    doc_entries: RawDocumentEntries,
    total_doc_len: u64,
    max_doc_id: DocId,
) -> Result<RawSegmentStreamPlan, Error> {
    let doc_count = u32::try_from(doc_entries.len()).map_err(|_| Error::TooManyDocuments)?;
    let term_count = u32::try_from(posting_lists.len()).map_err(|_| Error::TooManyTerms)?;
    let term_dir_offset = HEADER_LEN as u64;
    let term_dir_len = (term_count as u64)
        .checked_mul(TERM_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let doc_meta_offset = term_dir_offset
        .checked_add(term_dir_len)
        .ok_or(Error::SegmentTooLarge)?;
    let doc_meta_len = (doc_count as u64)
        .checked_mul(DOC_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let block_dir_offset = doc_meta_offset
        .checked_add(doc_meta_len)
        .ok_or(Error::SegmentTooLarge)?;
    let total_block_count = posting_lists.iter().try_fold(0u64, |count, term| {
        count
            .checked_add((term.postings.len() as u64).div_ceil(DEFAULT_BLOCK_SIZE as u64))
            .ok_or(Error::SegmentTooLarge)
    })?;
    let block_dir_len = total_block_count
        .checked_mul(BLOCK_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let postings_offset = block_dir_offset
        .checked_add(block_dir_len)
        .ok_or(Error::SegmentTooLarge)?;

    let mut term_entries = Vec::with_capacity(term_count as usize);
    let mut term_block_directories = Vec::with_capacity(term_count as usize);
    let mut block_entries = Vec::new();
    let mut postings_len_total = 0u64;

    for term in posting_lists {
        let term_postings_offset = postings_offset
            .checked_add(postings_len_total)
            .ok_or(Error::SegmentTooLarge)?;
        let blocks_offset = block_dir_offset
            .checked_add(
                u64::try_from(block_entries.len()).map_err(|_| Error::SegmentTooLarge)?
                    * BLOCK_ENTRY_LEN as u64,
            )
            .ok_or(Error::SegmentTooLarge)?;
        let term_start_len = postings_len_total;
        let mut prev_doc_id = 0;
        let mut first_posting = true;
        let mut max_weight = 0u32;
        let mut total_weight = 0u64;

        for chunk in term.postings.chunks(DEFAULT_BLOCK_SIZE as usize) {
            let block = measure_raw_posting_block(chunk, &mut first_posting, &mut prev_doc_id)?;
            let block_postings_offset = postings_offset
                .checked_add(postings_len_total)
                .ok_or(Error::SegmentTooLarge)?;
            postings_len_total = postings_len_total
                .checked_add(block.postings_len as u64)
                .ok_or(Error::SegmentTooLarge)?;
            max_weight = max_weight.max(block.max_weight);
            total_weight = total_weight.saturating_add(block.total_weight);
            block_entries.push(RawPostingBlockMeta {
                base_doc_id: block.base_doc_id,
                last_doc_id: block.last_doc_id,
                postings_offset: block_postings_offset,
                postings_len: block.postings_len,
                max_weight: block.max_weight,
            });
        }

        let postings_len = u32::try_from(postings_len_total - term_start_len)
            .map_err(|_| Error::SegmentTooLarge)?;
        term_entries.push(TermEntry {
            term_id: term.term_id,
            df: u32::try_from(term.postings.len()).map_err(|_| Error::TooManyDocuments)?,
            max_weight,
            total_weight,
            postings_offset: term_postings_offset,
            postings_len,
        });
        term_block_directories.push(TermBlockDirectory {
            block_count: u32::try_from(term.postings.len().div_ceil(DEFAULT_BLOCK_SIZE as usize))
                .map_err(|_| Error::TooManyDocuments)?,
            blocks_offset,
        });
    }

    let integrity_len = total_block_count
        .checked_mul(4)
        .and_then(|crcs| crcs.checked_add(INTEGRITY_HEADER_LEN as u64))
        .ok_or(Error::SegmentTooLarge)?;
    let footer_offset = postings_offset
        .checked_add(postings_len_total)
        .and_then(|end| end.checked_add(integrity_len))
        .ok_or(Error::SegmentTooLarge)?;
    let final_len = footer_offset
        .checked_add(FOOTER_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let final_len = usize::try_from(final_len).map_err(|_| Error::SegmentTooLarge)?;

    Ok(RawSegmentStreamPlan {
        meta: RawSegmentMeta {
            term_count,
            doc_count,
            max_doc_id,
            block_size: DEFAULT_BLOCK_SIZE,
            total_doc_len,
            term_dir_offset,
            doc_meta_offset,
            postings_offset,
            footer_offset,
            flags: FLAG_CHECKSUMS,
        },
        term_entries,
        term_block_directories,
        doc_entries,
        block_entries,
        final_len,
    })
}

fn build_u64_u32_segment_sections_from_term_iter<'a, I, F>(
    term_count: usize,
    terms: F,
    doc_entries: RawDocumentEntries,
    total_doc_len: u64,
    max_doc_id: DocId,
) -> Result<RawSegmentSections, Error>
where
    I: Iterator<Item = (RawTermId, &'a [(DocId, u32)])>,
    F: Fn() -> I,
{
    let doc_count = u32::try_from(doc_entries.len()).map_err(|_| Error::TooManyDocuments)?;
    let term_count = u32::try_from(term_count).map_err(|_| Error::TooManyTerms)?;
    let term_dir_offset = HEADER_LEN as u64;
    let term_dir_len = (term_count as u64)
        .checked_mul(TERM_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let doc_meta_offset = term_dir_offset
        .checked_add(term_dir_len)
        .ok_or(Error::SegmentTooLarge)?;
    let doc_meta_len = (doc_count as u64)
        .checked_mul(DOC_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let block_dir_offset = doc_meta_offset
        .checked_add(doc_meta_len)
        .ok_or(Error::SegmentTooLarge)?;
    let mut total_block_count = 0u64;
    for (_, list) in terms() {
        let len = list.len() as u64;
        total_block_count = total_block_count
            .checked_add(len.div_ceil(DEFAULT_BLOCK_SIZE as u64))
            .ok_or(Error::SegmentTooLarge)?;
    }
    let block_dir_len = total_block_count
        .checked_mul(BLOCK_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let postings_offset = block_dir_offset
        .checked_add(block_dir_len)
        .ok_or(Error::SegmentTooLarge)?;

    let mut postings_bytes = Vec::new();
    let mut term_entries = Vec::with_capacity(term_count as usize);
    let mut term_block_directories = Vec::with_capacity(term_count as usize);
    let mut block_entries = Vec::new();
    let mut block_crcs: Vec<u32> = Vec::new();
    for (term_id, list) in terms() {
        let offset = postings_offset
            .checked_add(u64::try_from(postings_bytes.len()).map_err(|_| Error::SegmentTooLarge)?)
            .ok_or(Error::SegmentTooLarge)?;
        let blocks_offset = block_dir_offset
            .checked_add(
                u64::try_from(block_entries.len()).map_err(|_| Error::SegmentTooLarge)?
                    * BLOCK_ENTRY_LEN as u64,
            )
            .ok_or(Error::SegmentTooLarge)?;
        let start_len = postings_bytes.len();
        let mut prev_doc_id = 0;
        let mut first_posting = true;
        let mut max_weight = 0u32;
        let mut total_weight = 0u64;
        for chunk in list.chunks(DEFAULT_BLOCK_SIZE as usize) {
            let block_start_len = postings_bytes.len();
            let block = append_encoded_raw_posting_block(
                chunk,
                &mut first_posting,
                &mut prev_doc_id,
                &mut postings_bytes,
            )?;
            max_weight = max_weight.max(block.max_weight);
            total_weight = total_weight.saturating_add(block.total_weight);
            block_entries.push(RawPostingBlockMeta {
                base_doc_id: block.base_doc_id,
                last_doc_id: block.last_doc_id,
                postings_offset: postings_offset
                    .checked_add(
                        u64::try_from(block_start_len).map_err(|_| Error::SegmentTooLarge)?,
                    )
                    .ok_or(Error::SegmentTooLarge)?,
                postings_len: block.postings_len,
                max_weight: block.max_weight,
            });
            block_crcs.push(crc32fast::hash(&postings_bytes[block_start_len..]));
        }
        let postings_len =
            u32::try_from(postings_bytes.len() - start_len).map_err(|_| Error::SegmentTooLarge)?;
        term_entries.push(TermEntry {
            term_id,
            df: u32::try_from(list.len()).map_err(|_| Error::TooManyDocuments)?,
            max_weight,
            total_weight,
            postings_offset: offset,
            postings_len,
        });
        term_block_directories.push(TermBlockDirectory {
            block_count: u32::try_from(list.len().div_ceil(DEFAULT_BLOCK_SIZE as usize))
                .map_err(|_| Error::TooManyDocuments)?,
            blocks_offset,
        });
    }

    let integrity_len = (block_crcs.len() as u64)
        .checked_mul(4)
        .and_then(|crcs| crcs.checked_add(INTEGRITY_HEADER_LEN as u64))
        .ok_or(Error::SegmentTooLarge)?;
    let footer_offset = postings_offset
        .checked_add(u64::try_from(postings_bytes.len()).map_err(|_| Error::SegmentTooLarge)?)
        .and_then(|end| end.checked_add(integrity_len))
        .ok_or(Error::SegmentTooLarge)?;
    let final_len = footer_offset
        .checked_add(FOOTER_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let final_len = usize::try_from(final_len).map_err(|_| Error::SegmentTooLarge)?;

    Ok(RawSegmentSections {
        meta: RawSegmentMeta {
            term_count,
            doc_count,
            max_doc_id,
            block_size: DEFAULT_BLOCK_SIZE,
            total_doc_len,
            term_dir_offset,
            doc_meta_offset,
            postings_offset,
            footer_offset,
            flags: FLAG_CHECKSUMS,
        },
        term_entries,
        term_block_directories,
        doc_entries,
        block_entries,
        postings_bytes,
        block_crcs,
        final_len,
    })
}

fn append_encoded_raw_posting_block(
    chunk: &[(DocId, u32)],
    first_posting: &mut bool,
    prev_doc_id: &mut DocId,
    out: &mut Vec<u8>,
) -> Result<EncodedRawPostingBlock, Error> {
    let start_len = out.len();
    let base_doc_id = *prev_doc_id;
    let mut max_weight = 0u32;
    let mut total_weight = 0u64;

    for &(doc_id, weight) in chunk {
        let gap = if *first_posting {
            *first_posting = false;
            doc_id
        } else {
            doc_id - *prev_doc_id
        };
        varint::encode_u32(gap, out);
        varint::encode_u32(weight, out);
        *prev_doc_id = doc_id;
        max_weight = max_weight.max(weight);
        total_weight = total_weight.saturating_add(weight as u64);
    }

    Ok(EncodedRawPostingBlock {
        base_doc_id,
        last_doc_id: *prev_doc_id,
        postings_len: u32::try_from(out.len() - start_len).map_err(|_| Error::SegmentTooLarge)?,
        max_weight,
        total_weight,
    })
}

fn measure_raw_posting_block(
    chunk: &[(DocId, u32)],
    first_posting: &mut bool,
    prev_doc_id: &mut DocId,
) -> Result<EncodedRawPostingBlock, Error> {
    let base_doc_id = *prev_doc_id;
    let mut postings_len = 0u32;
    let mut max_weight = 0u32;
    let mut total_weight = 0u64;

    for &(doc_id, weight) in chunk {
        let gap = if *first_posting {
            *first_posting = false;
            doc_id
        } else {
            doc_id - *prev_doc_id
        };
        postings_len = postings_len
            .checked_add(varint_u32_len(gap))
            .and_then(|len| len.checked_add(varint_u32_len(weight)))
            .ok_or(Error::SegmentTooLarge)?;
        *prev_doc_id = doc_id;
        max_weight = max_weight.max(weight);
        total_weight = total_weight.saturating_add(weight as u64);
    }

    Ok(EncodedRawPostingBlock {
        base_doc_id,
        last_doc_id: *prev_doc_id,
        postings_len,
        max_weight,
        total_weight,
    })
}

fn varint_u32_len(value: u32) -> u32 {
    match value {
        0..=0x7f => 1,
        0x80..=0x3fff => 2,
        0x4000..=0x1f_ffff => 3,
        0x20_0000..=0x0fff_ffff => 4,
        _ => 5,
    }
}

fn append_term_directory(out: &mut Vec<u8>, sections: &RawSegmentSections) -> u32 {
    let start = out.len();
    for (entry, block_directory) in sections
        .term_entries
        .iter()
        .zip(&sections.term_block_directories)
    {
        put_term_directory_entry(out, entry, block_directory);
    }
    debug_assert_eq!(
        out.len() - start,
        sections.term_entries.len() * TERM_ENTRY_LEN
    );
    crc32fast::hash(&out[start..])
}

fn append_document_metadata(out: &mut Vec<u8>, sections: &RawSegmentSections) -> u32 {
    let start = out.len();
    for &(doc_id, doc_len) in &sections.doc_entries {
        put_document_metadata_entry(out, doc_id, doc_len);
    }
    debug_assert_eq!(
        out.len() - start,
        sections.doc_entries.len() * DOC_ENTRY_LEN
    );
    crc32fast::hash(&out[start..])
}

fn append_block_directory(out: &mut Vec<u8>, sections: &RawSegmentSections) -> u32 {
    let start = out.len();
    for block in &sections.block_entries {
        put_block_directory_entry(out, block);
    }
    debug_assert_eq!(
        out.len() - start,
        sections.block_entries.len() * BLOCK_ENTRY_LEN
    );
    crc32fast::hash(&out[start..])
}

fn put_term_directory_entry(
    out: &mut Vec<u8>,
    entry: &TermEntry,
    block_directory: &TermBlockDirectory,
) {
    let start = out.len();
    put_u64(out, entry.term_id);
    put_u32(out, entry.df);
    put_u32(out, entry.max_weight);
    put_u64(out, entry.total_weight);
    put_u64(out, entry.postings_offset);
    put_u32(out, entry.postings_len);
    put_u32(out, block_directory.block_count);
    put_u64(out, block_directory.blocks_offset);
    debug_assert_eq!(out.len() - start, TERM_ENTRY_LEN);
}

fn put_document_metadata_entry(out: &mut Vec<u8>, doc_id: DocId, doc_len: u32) {
    let start = out.len();
    put_u32(out, doc_id);
    put_u32(out, doc_len);
    debug_assert_eq!(out.len() - start, DOC_ENTRY_LEN);
}

fn put_block_directory_entry(out: &mut Vec<u8>, block: &RawPostingBlockMeta) {
    let start = out.len();
    put_u32(out, block.base_doc_id);
    put_u32(out, block.last_doc_id);
    put_u64(out, block.postings_offset);
    put_u32(out, block.postings_len);
    put_u32(out, block.max_weight);
    debug_assert_eq!(out.len() - start, BLOCK_ENTRY_LEN);
}

fn put_integrity_section(
    out: &mut Vec<u8>,
    term_dir_crc: u32,
    doc_meta_crc: u32,
    block_dir_crc: u32,
    block_crcs: &[u32],
) {
    let start = out.len();
    put_u32(out, term_dir_crc);
    put_u32(out, doc_meta_crc);
    put_u32(out, block_dir_crc);
    for &crc in block_crcs {
        put_u32(out, crc);
    }
    debug_assert_eq!(
        out.len() - start,
        INTEGRITY_HEADER_LEN + block_crcs.len() * 4
    );
}

fn write_raw_segment_sections_to<W: Write + ?Sized>(
    sections: &RawSegmentSections,
    writer: &mut W,
) -> std::io::Result<()> {
    write_header_to(writer, sections.meta)?;
    let term_dir_crc = write_term_directory_to(
        writer,
        &sections.term_entries,
        &sections.term_block_directories,
    )?;
    let doc_meta_crc = write_document_metadata_to(writer, &sections.doc_entries)?;
    let block_dir_crc = write_block_directory_to(writer, &sections.block_entries)?;
    writer.write_all(&sections.postings_bytes)?;

    let mut integrity = Vec::with_capacity(INTEGRITY_HEADER_LEN + sections.block_crcs.len() * 4);
    put_integrity_section(
        &mut integrity,
        term_dir_crc,
        doc_meta_crc,
        block_dir_crc,
        &sections.block_crcs,
    );
    writer.write_all(&integrity)?;
    writer.write_all(FOOTER_MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    Ok(())
}

fn write_raw_segment_stream_plan_to<W: Write + ?Sized>(
    plan: &RawSegmentStreamPlan,
    terms: &[RawTermPostingList<'_>],
    writer: &mut W,
) -> Result<(), RawSegmentWriteError> {
    debug_assert_eq!(
        plan.final_len as u64,
        plan.meta.footer_offset + FOOTER_LEN as u64
    );
    write_header_to(writer, plan.meta)?;
    let term_dir_crc =
        write_term_directory_to(writer, &plan.term_entries, &plan.term_block_directories)?;
    let doc_meta_crc = write_document_metadata_to(writer, &plan.doc_entries)?;
    let block_dir_crc = write_block_directory_to(writer, &plan.block_entries)?;
    let block_crcs = write_term_posting_payloads_to(writer, terms, &plan.block_entries)?;

    let mut integrity = Vec::with_capacity(INTEGRITY_HEADER_LEN + block_crcs.len() * 4);
    put_integrity_section(
        &mut integrity,
        term_dir_crc,
        doc_meta_crc,
        block_dir_crc,
        &block_crcs,
    );
    writer.write_all(&integrity)?;
    writer.write_all(FOOTER_MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    Ok(())
}

fn write_header_to<W: Write + ?Sized>(writer: &mut W, meta: RawSegmentMeta) -> std::io::Result<()> {
    let mut header = Vec::with_capacity(HEADER_LEN);
    put_header(&mut header, meta);
    writer.write_all(&header)
}

fn write_zeroes_to<W: Write + ?Sized>(writer: &mut W, mut len: u64) -> std::io::Result<()> {
    const ZEROES: [u8; 4096] = [0; 4096];
    while len > 0 {
        let chunk_len = if len > ZEROES.len() as u64 {
            ZEROES.len()
        } else {
            len as usize
        };
        writer.write_all(&ZEROES[..chunk_len])?;
        len -= chunk_len as u64;
    }
    Ok(())
}

fn segment_position(segment_start: u64, offset: u64) -> Result<u64, Error> {
    segment_start
        .checked_add(offset)
        .ok_or(Error::SegmentTooLarge)
}

fn write_term_directory_to<W: Write + ?Sized>(
    writer: &mut W,
    term_entries: &[TermEntry],
    term_block_directories: &[TermBlockDirectory],
) -> std::io::Result<u32> {
    let mut hasher = crc32fast::Hasher::new();
    let mut entry_bytes = Vec::with_capacity(TERM_ENTRY_LEN);
    for (entry, block_directory) in term_entries.iter().zip(term_block_directories) {
        entry_bytes.clear();
        put_term_directory_entry(&mut entry_bytes, entry, block_directory);
        hasher.update(&entry_bytes);
        writer.write_all(&entry_bytes)?;
    }
    Ok(hasher.finalize())
}

fn write_document_metadata_to<W: Write + ?Sized>(
    writer: &mut W,
    doc_entries: &[(DocId, u32)],
) -> std::io::Result<u32> {
    let mut hasher = crc32fast::Hasher::new();
    let mut entry_bytes = Vec::with_capacity(DOC_ENTRY_LEN);
    for &(doc_id, doc_len) in doc_entries {
        entry_bytes.clear();
        put_document_metadata_entry(&mut entry_bytes, doc_id, doc_len);
        hasher.update(&entry_bytes);
        writer.write_all(&entry_bytes)?;
    }
    Ok(hasher.finalize())
}

fn write_block_directory_to<W: Write + ?Sized>(
    writer: &mut W,
    block_entries: &[RawPostingBlockMeta],
) -> std::io::Result<u32> {
    let mut hasher = crc32fast::Hasher::new();
    let mut entry_bytes = Vec::with_capacity(BLOCK_ENTRY_LEN);
    for block in block_entries {
        entry_bytes.clear();
        put_block_directory_entry(&mut entry_bytes, block);
        hasher.update(&entry_bytes);
        writer.write_all(&entry_bytes)?;
    }
    Ok(hasher.finalize())
}

fn write_term_posting_payloads_to<W: Write + ?Sized>(
    writer: &mut W,
    terms: &[RawTermPostingList<'_>],
    block_entries: &[RawPostingBlockMeta],
) -> Result<Vec<u32>, RawSegmentWriteError> {
    let mut scratch = Vec::new();
    let mut planned_blocks = block_entries.iter();
    let mut block_crcs = Vec::with_capacity(block_entries.len());
    for term in terms {
        let mut prev_doc_id = 0;
        let mut first_posting = true;
        for chunk in term.postings.chunks(DEFAULT_BLOCK_SIZE as usize) {
            scratch.clear();
            let block = append_encoded_raw_posting_block(
                chunk,
                &mut first_posting,
                &mut prev_doc_id,
                &mut scratch,
            )?;
            let planned = planned_blocks
                .next()
                .expect("raw segment stream plan missing block metadata");
            debug_assert_eq!(block.base_doc_id, planned.base_doc_id);
            debug_assert_eq!(block.last_doc_id, planned.last_doc_id);
            debug_assert_eq!(block.postings_len, planned.postings_len);
            debug_assert_eq!(block.max_weight, planned.max_weight);
            block_crcs.push(crc32fast::hash(&scratch));
            writer.write_all(&scratch)?;
        }
    }
    debug_assert!(planned_blocks.next().is_none());
    Ok(block_crcs)
}

fn put_header(out: &mut Vec<u8>, meta: RawSegmentMeta) {
    out.extend_from_slice(MAGIC);
    put_u32(out, VERSION);
    put_u32(out, meta.flags);
    put_u32(out, meta.term_count);
    put_u32(out, meta.doc_count);
    put_u32(out, meta.max_doc_id);
    put_u32(out, meta.block_size);
    put_u64(out, meta.total_doc_len);
    put_u64(out, meta.term_dir_offset);
    put_u64(out, meta.doc_meta_offset);
    put_u64(out, meta.postings_offset);
    put_u64(out, meta.footer_offset);
    debug_assert_eq!(out.len(), HEADER_LEN);
}

fn parse_header(bytes: &[u8]) -> Result<RawSegmentMeta, Error> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::Truncated { section: "header" });
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(Error::BadMagic);
    }

    let version = read_u32_at(bytes, 8, "header")?;
    if version != VERSION {
        return Err(Error::UnsupportedVersion { version });
    }
    let flags = read_u32_at(bytes, 12, "header")?;
    if flags & !FLAG_CHECKSUMS != 0 {
        return Err(Error::UnsupportedFlags { flags });
    }

    Ok(RawSegmentMeta {
        term_count: read_u32_at(bytes, 16, "header")?,
        doc_count: read_u32_at(bytes, 20, "header")?,
        max_doc_id: read_u32_at(bytes, 24, "header")?,
        block_size: read_u32_at(bytes, 28, "header")?,
        total_doc_len: read_u64_at(bytes, 32, "header")?,
        term_dir_offset: read_u64_at(bytes, 40, "header")?,
        doc_meta_offset: read_u64_at(bytes, 48, "header")?,
        postings_offset: read_u64_at(bytes, 56, "header")?,
        footer_offset: read_u64_at(bytes, 64, "header")?,
        flags,
    })
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Locate the integrity section for a checksummed segment:
/// `(offset, block_count, len)`. The section sits immediately before the
/// footer and its length is fully derived from header fields, so no header
/// layout changed when checksums were introduced.
fn integrity_layout(meta: RawSegmentMeta) -> Result<(u64, u64, u64), Error> {
    let block_dir_start = doc_meta_end(meta)?;
    let block_dir_len =
        meta.postings_offset
            .checked_sub(block_dir_start)
            .ok_or(Error::InvalidLayout {
                reason: "block directory must follow doc metadata",
            })?;
    if block_dir_len % BLOCK_ENTRY_LEN as u64 != 0 {
        return Err(Error::InvalidLayout {
            reason: "block directory length is not a multiple of the entry size",
        });
    }
    let block_count = block_dir_len / BLOCK_ENTRY_LEN as u64;
    let integrity_len = block_count
        .checked_mul(4)
        .and_then(|crcs| crcs.checked_add(INTEGRITY_HEADER_LEN as u64))
        .ok_or(Error::SegmentTooLarge)?;
    let integrity_offset =
        meta.footer_offset
            .checked_sub(integrity_len)
            .ok_or(Error::InvalidLayout {
                reason: "integrity section does not fit before the footer",
            })?;
    if integrity_offset < meta.postings_offset {
        return Err(Error::InvalidLayout {
            reason: "integrity section overlaps the postings region",
        });
    }
    Ok((integrity_offset, block_count, integrity_len))
}

fn verify_directory_checksums(
    term_dir: &[u8],
    doc_meta: &[u8],
    block_dir: &[u8],
    integrity: &[u8],
) -> Result<(), Error> {
    let sections: [(&[u8], &'static str, usize); 3] = [
        (term_dir, "term directory", 0),
        (doc_meta, "doc metadata", 4),
        (block_dir, "block directory", 8),
    ];
    for (bytes, section, at) in sections {
        let want = read_u32_at(integrity, at, "integrity")?;
        if crc32fast::hash(bytes) != want {
            return Err(Error::ChecksumMismatch { section });
        }
    }
    Ok(())
}

/// Per-block checksums keyed by the block's postings byte offset, so range
/// reads can verify by the offset they already hold.
fn block_checksum_list(
    block_dir: &[u8],
    block_crcs: &[u8],
) -> Result<Vec<RawBlockChecksum>, Error> {
    if block_dir.len() % BLOCK_ENTRY_LEN != 0 {
        return Err(Error::InvalidLayout {
            reason: "block directory length is not a multiple of the entry size",
        });
    }
    let block_count = block_dir.len() / BLOCK_ENTRY_LEN;
    if block_crcs.len() != block_count * 4 {
        return Err(Error::InvalidLayout {
            reason: "integrity section length mismatch",
        });
    }
    let mut checksums = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let entry = i * BLOCK_ENTRY_LEN;
        let offset = read_u64_at(block_dir, entry + 8, "block directory")?;
        let len = read_u32_at(block_dir, entry + 16, "block directory")?;
        let crc = read_u32_at(block_crcs, i * 4, "integrity")?;
        checksums.push(RawBlockChecksum { offset, len, crc });
    }
    checksums.sort_unstable_by_key(|checksum| checksum.offset);
    Ok(checksums)
}

/// Verify one block's bytes against its stored checksum.
fn verify_block_slice(
    checksums: &[RawBlockChecksum],
    offset: u64,
    bytes: &[u8],
) -> Result<(), Error> {
    match checksums.binary_search_by_key(&offset, |checksum| checksum.offset) {
        Ok(index) if checksums[index].len as usize == bytes.len() => {
            if crc32fast::hash(bytes) != checksums[index].crc {
                return Err(Error::ChecksumMismatch {
                    section: "posting block",
                });
            }
            Ok(())
        }
        Ok(_) => Err(Error::InvalidLayout {
            reason: "posting block read does not match block bounds",
        }),
        Err(_) => Err(Error::InvalidLayout {
            reason: "posting block has no checksum entry",
        }),
    }
}

/// Verify every checksummed block tiling the span `[offset, offset + len)`.
/// Whole-term reads span consecutive blocks; each is verified against its own
/// checksum and the span must be fully covered.
fn verify_span_blocks(
    checksums: &[RawBlockChecksum],
    offset: u64,
    bytes: &[u8],
) -> Result<(), Error> {
    let end = offset
        .checked_add(bytes.len() as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let mut covered = 0u64;
    let start = checksums.partition_point(|checksum| checksum.offset < offset);
    for checksum in &checksums[start..] {
        if checksum.offset >= end {
            break;
        }
        let rel = usize::try_from(checksum.offset - offset).map_err(|_| Error::SegmentTooLarge)?;
        let block_end = rel
            .checked_add(checksum.len as usize)
            .ok_or(Error::SegmentTooLarge)?;
        if block_end > bytes.len() {
            return Err(Error::InvalidLayout {
                reason: "posting block exceeds the read span",
            });
        }
        if crc32fast::hash(&bytes[rel..block_end]) != checksum.crc {
            return Err(Error::ChecksumMismatch {
                section: "posting block",
            });
        }
        covered = covered.saturating_add(checksum.len as u64);
    }
    if covered != bytes.len() as u64 {
        return Err(Error::InvalidLayout {
            reason: "posting span is not fully covered by checksummed blocks",
        });
    }
    Ok(())
}

fn validate_layout(bytes: &[u8], meta: RawSegmentMeta) -> Result<(), Error> {
    validate_layout_len(meta, bytes.len())
}

fn validate_layout_len(meta: RawSegmentMeta, bytes_len: usize) -> Result<(), Error> {
    if meta.term_dir_offset != HEADER_LEN as u64 {
        return Err(Error::InvalidLayout {
            reason: "term directory must follow header",
        });
    }
    let term_dir_len = (meta.term_count as u64)
        .checked_mul(TERM_ENTRY_LEN as u64)
        .ok_or(Error::InvalidLayout {
            reason: "term directory length overflows",
        })?;
    let term_dir_end =
        meta.term_dir_offset
            .checked_add(term_dir_len)
            .ok_or(Error::InvalidLayout {
                reason: "term directory end overflows",
            })?;
    if term_dir_end != meta.doc_meta_offset {
        return Err(Error::InvalidLayout {
            reason: "doc metadata must follow term directory",
        });
    }
    let doc_meta_len = (meta.doc_count as u64)
        .checked_mul(DOC_ENTRY_LEN as u64)
        .ok_or(Error::InvalidLayout {
            reason: "doc metadata length overflows",
        })?;
    let doc_meta_end = doc_meta_end(meta)?;
    if doc_meta_end > meta.postings_offset {
        return Err(Error::InvalidLayout {
            reason: "block directory must follow doc metadata",
        });
    }
    if meta.postings_offset > meta.footer_offset {
        return Err(Error::InvalidLayout {
            reason: "footer cannot precede postings",
        });
    }
    let footer_end =
        meta.footer_offset
            .checked_add(FOOTER_LEN as u64)
            .ok_or(Error::InvalidLayout {
                reason: "footer end overflows",
            })?;
    if footer_end != bytes_len as u64 {
        return Err(Error::InvalidLayout {
            reason: "footer must end at segment end",
        });
    }
    checked_range(
        meta.term_dir_offset,
        term_dir_len,
        bytes_len,
        "term directory",
    )?;
    checked_range(
        meta.doc_meta_offset,
        doc_meta_len,
        bytes_len,
        "doc metadata",
    )?;
    checked_range(
        doc_meta_end,
        meta.postings_offset - doc_meta_end,
        bytes_len,
        "block directory",
    )?;
    checked_range(
        meta.postings_offset,
        meta.footer_offset - meta.postings_offset,
        bytes_len,
        "postings",
    )?;
    Ok(())
}

fn doc_meta_end(meta: RawSegmentMeta) -> Result<u64, Error> {
    let doc_meta_len = (meta.doc_count as u64)
        .checked_mul(DOC_ENTRY_LEN as u64)
        .ok_or(Error::InvalidLayout {
            reason: "doc metadata length overflows",
        })?;
    meta.doc_meta_offset
        .checked_add(doc_meta_len)
        .ok_or(Error::InvalidLayout {
            reason: "doc metadata end overflows",
        })
}

fn checked_range(
    start: u64,
    len: u64,
    bytes_len: usize,
    section: &'static str,
) -> Result<std::ops::Range<usize>, Error> {
    let end = start.checked_add(len).ok_or(Error::InvalidLayout {
        reason: "range end overflows",
    })?;
    let start = usize::try_from(start).map_err(|_| Error::SegmentTooLarge)?;
    let end = usize::try_from(end).map_err(|_| Error::SegmentTooLarge)?;
    if end > bytes_len || start > end {
        return Err(Error::Truncated { section });
    }
    Ok(start..end)
}

fn read_u32_at(bytes: &[u8], offset: usize, section: &'static str) -> Result<u32, Error> {
    let range = checked_range(offset as u64, 4, bytes.len(), section)?;
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[range]);
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_at(bytes: &[u8], offset: usize, section: &'static str) -> Result<u64, Error> {
    let range = checked_range(offset as u64, 8, bytes.len(), section)?;
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[range]);
    Ok(u64::from_le_bytes(arr))
}

fn intersect_doc_id_lists_in_place(candidates: &mut Vec<DocId>, docs: &[DocId]) {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut write = 0usize;
    while i < candidates.len() && j < docs.len() {
        match candidates[i].cmp(&docs[j]) {
            std::cmp::Ordering::Equal => {
                candidates[write] = candidates[i];
                write += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    candidates.truncate(write);
}

fn union_doc_id_lists(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(a.len().saturating_add(b.len()));
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                push_unique_doc_id(&mut out, a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                push_unique_doc_id(&mut out, a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                push_unique_doc_id(&mut out, b[j]);
                j += 1;
            }
        }
    }
    while i < a.len() {
        push_unique_doc_id(&mut out, a[i]);
        i += 1;
    }
    while j < b.len() {
        push_unique_doc_id(&mut out, b[j]);
        j += 1;
    }
    out
}

#[inline]
fn push_unique_doc_id(out: &mut Vec<DocId>, doc_id: DocId) {
    if out.last().copied() != Some(doc_id) {
        out.push(doc_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PostingsIndex;
    use proptest::prelude::*;
    use std::io::Write;

    fn collect_postings(segment: &RawSegment<'_>, term_id: RawTermId) -> Vec<(DocId, u32)> {
        segment
            .postings(term_id)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn collect_block(
        segment: &RawSegment<'_>,
        term_id: RawTermId,
        block_index: u32,
    ) -> Vec<(DocId, u32)> {
        segment
            .posting_block_postings(term_id, block_index)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn collect_block_with_lens(
        segment: &RawSegment<'_>,
        term_id: RawTermId,
        block_index: u32,
    ) -> Vec<(DocId, u32, u32)> {
        let mut out = Vec::new();
        segment
            .for_each_posting_block_with_document_len(
                term_id,
                block_index,
                |doc_id, weight, len| {
                    out.push((doc_id, weight, len));
                },
            )
            .unwrap();
        out
    }

    fn strip_checksums_for_test(bytes: &[u8]) -> Vec<u8> {
        let meta = parse_header(bytes).unwrap();
        assert!(meta.has_checksums());
        let (integrity_offset, _, _) = integrity_layout(meta).unwrap();
        let integrity_offset = usize::try_from(integrity_offset).unwrap();
        let footer_offset = usize::try_from(meta.footer_offset).unwrap();

        let mut legacy = Vec::with_capacity(integrity_offset + FOOTER_LEN);
        legacy.extend_from_slice(&bytes[..integrity_offset]);
        legacy.extend_from_slice(&bytes[footer_offset..footer_offset + FOOTER_LEN]);
        legacy[12..16].copy_from_slice(&0u32.to_le_bytes());
        legacy[64..72].copy_from_slice(&(integrity_offset as u64).to_le_bytes());
        RawSegment::open(&legacy).unwrap();
        legacy
    }

    #[test]
    fn raw_segment_roundtrips_numeric_terms() {
        let doc_a = vec![(10, 1), (20, 2), (10, 3)];
        let doc_b = vec![(10, 1), (30, 1)];
        let doc_c = vec![(20, 1)];
        let docs = vec![
            RawDocument::new(5, &doc_a),
            RawDocument::new(2, &doc_b),
            RawDocument::new(9, &doc_c),
        ];

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();

        assert!(segment.meta().has_checksums());
        assert_eq!(segment.num_docs(), 3);
        assert_eq!(segment.meta().term_count(), 3);
        assert_eq!(segment.meta().max_doc_id(), 9);
        assert_eq!(segment.meta().total_doc_len(), 9);
        assert_eq!(segment.avg_doc_len(), 3.0);
        assert_eq!(segment.document_len(5).unwrap(), Some(6));
        assert_eq!(segment.document_len(2).unwrap(), Some(2));
        assert_eq!(segment.document_len(999).unwrap(), None);
        let mut document_lengths = Vec::new();
        segment
            .for_each_document_len(|doc_id, len| document_lengths.push((doc_id, len)))
            .unwrap();
        assert_eq!(document_lengths, vec![(2, 2), (5, 6), (9, 1)]);
        assert_eq!(segment.df(10).unwrap(), 2);
        assert_eq!(segment.total_weight(10).unwrap(), 5);
        assert_eq!(segment.max_weight(10).unwrap(), 4);
        assert_eq!(segment.term_ids().unwrap(), vec![10, 20, 30]);
        assert_eq!(collect_postings(&segment, 10), vec![(2, 1), (5, 4)]);
        let mut visited = Vec::new();
        segment
            .for_each_posting(10, |doc_id, weight| visited.push((doc_id, weight)))
            .unwrap();
        assert_eq!(visited, vec![(2, 1), (5, 4)]);
        let mut visited_with_lens = Vec::new();
        segment
            .for_each_posting_with_document_len(10, |doc_id, weight, doc_len| {
                visited_with_lens.push((doc_id, weight, doc_len));
            })
            .unwrap();
        assert_eq!(visited_with_lens, vec![(2, 1, 2), (5, 4, 6)]);
        assert_eq!(collect_postings(&segment, 20), vec![(5, 2), (9, 1)]);
        assert_eq!(
            segment.posting_blocks(10).unwrap(),
            vec![RawPostingBlockMeta {
                base_doc_id: 0,
                last_doc_id: 5,
                postings_offset: segment.term_entry(10).unwrap().unwrap().postings_offset,
                postings_len: segment.term_entry(10).unwrap().unwrap().postings_len,
                max_weight: 4,
            }]
        );
        assert!(collect_postings(&segment, 999).is_empty());
        assert_eq!(
            segment.candidates_any_terms(&[10, 20]).unwrap(),
            vec![2, 5, 9]
        );
        assert_eq!(segment.candidates_any_terms(&[10, 10]).unwrap(), vec![2, 5]);
        assert!(segment.candidates_any_terms(&[999]).unwrap().is_empty());
        assert_eq!(
            segment
                .plan_candidates(&[10, 20], PlannerConfig::default())
                .unwrap(),
            CandidatePlan::ScanAll
        );
        assert_eq!(
            segment
                .plan_candidates(
                    &[10, 20],
                    PlannerConfig {
                        max_candidate_ratio: 2.0,
                        max_candidates: 10,
                    },
                )
                .unwrap(),
            CandidatePlan::Candidates(vec![2, 5, 9])
        );
        assert_eq!(segment.candidates_all_terms(&[10, 20]).unwrap(), vec![5]);
        assert_eq!(segment.candidates_all_terms(&[20]).unwrap(), vec![5, 9]);
        assert_eq!(segment.candidates_all_terms(&[10, 10]).unwrap(), vec![2, 5]);
        assert!(segment.candidates_all_terms(&[999]).unwrap().is_empty());
    }

    #[test]
    fn raw_segment_writer_matches_vec_writer() {
        let doc_a = vec![(10, 1), (20, 2), (10, 3)];
        let doc_b = vec![(10, 1), (30, 1)];
        let doc_c = vec![(20, 1)];
        let docs = vec![
            RawDocument::new(5, &doc_a),
            RawDocument::new(2, &doc_b),
            RawDocument::new(9, &doc_c),
        ];

        let expected = write_u64_u32_segment(&docs).unwrap();
        let mut written = Vec::new();
        write_u64_u32_segment_to(&docs, &mut written).unwrap();

        assert_eq!(written, expected);
        RawSegment::open(&written).unwrap();
    }

    #[test]
    fn raw_segment_iterator_writer_matches_slice_writer() {
        let term_docs = [vec![(10, 1), (20, 2), (10, 3)], vec![(10, 1), (30, 1)]];
        let docs = [
            RawDocument::new(2, term_docs[0].as_slice()),
            RawDocument::new(5, term_docs[1].as_slice()),
        ];

        let expected = write_u64_u32_segment(&docs).unwrap();
        let iter = term_docs
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new((doc_id as DocId) * 3 + 2, terms));
        let from_iter = write_u64_u32_segment_from_iter(iter).unwrap();
        let mut written = Vec::new();
        let iter = term_docs
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new((doc_id as DocId) * 3 + 2, terms));
        write_u64_u32_segment_from_iter_to(iter, &mut written).unwrap();

        assert_eq!(from_iter, expected);
        assert_eq!(written, expected);
    }

    #[test]
    fn raw_segment_sorted_iterator_writer_matches_slice_writer() {
        let term_docs = [
            vec![(10, 1), (20, 2), (10, 3)],
            vec![(10, 1), (30, 1)],
            vec![(20, 1), (40, 5)],
        ];
        let docs = [
            RawDocument::new(2, term_docs[0].as_slice()),
            RawDocument::new(5, term_docs[1].as_slice()),
            RawDocument::new(9, term_docs[2].as_slice()),
        ];

        let expected = write_u64_u32_segment(&docs).unwrap();
        let from_iter = write_u64_u32_segment_sorted_from_iter(docs.iter().copied()).unwrap();
        let mut written = Vec::new();
        write_u64_u32_segment_sorted_from_iter_to(docs.iter().copied(), &mut written).unwrap();

        assert_eq!(from_iter, expected);
        assert_eq!(written, expected);
        RawSegment::open(&written).unwrap();
    }

    #[test]
    fn raw_segment_sorted_iterator_rejects_unsorted_doc_ids() {
        let term = vec![(10, 1)];
        let decreasing = [
            RawDocument::new(5, term.as_slice()),
            RawDocument::new(2, term.as_slice()),
        ];
        assert_eq!(
            write_u64_u32_segment_sorted_from_iter(decreasing).unwrap_err(),
            Error::UnsortedDocId {
                previous: 5,
                current: 2
            }
        );

        let duplicate = [
            RawDocument::new(5, term.as_slice()),
            RawDocument::new(5, term.as_slice()),
        ];
        assert_eq!(
            write_u64_u32_segment_sorted_from_iter(duplicate).unwrap_err(),
            Error::DuplicateDocId { doc_id: 5 }
        );
    }

    #[test]
    fn raw_segment_term_postings_writer_matches_slice_writer() {
        let doc_a = vec![(10, 1), (20, 2), (10, 3)];
        let doc_b = vec![(10, 1), (30, 1)];
        let doc_c = vec![(20, 1), (40, 5)];
        let docs = [
            RawDocument::new(2, doc_a.as_slice()),
            RawDocument::new(5, doc_b.as_slice()),
            RawDocument::new(9, doc_c.as_slice()),
        ];
        let doc_lengths = [(2, 6), (5, 2), (9, 6)];
        let term_10 = [(2, 4), (5, 1)];
        let term_20 = [(2, 2), (9, 1)];
        let term_30 = [(5, 1)];
        let term_40 = [(9, 5)];
        let terms = [
            RawTermPostingList::new(10, &term_10),
            RawTermPostingList::new(20, &term_20),
            RawTermPostingList::new(30, &term_30),
            RawTermPostingList::new(40, &term_40),
        ];

        let expected = write_u64_u32_segment(&docs).unwrap();
        let from_terms = write_u64_u32_segment_from_term_postings(&doc_lengths, &terms).unwrap();
        let mut written = Vec::new();
        write_u64_u32_segment_from_term_postings_to(&doc_lengths, &terms, &mut written).unwrap();
        let mut seekable = std::io::Cursor::new(Vec::new());
        write_u64_u32_segment_from_term_postings_seekable_to(&doc_lengths, &terms, &mut seekable)
            .unwrap();
        let mut prefixed_seekable = std::io::Cursor::new(vec![0xaa; 13]);
        prefixed_seekable.seek(SeekFrom::End(0)).unwrap();
        write_u64_u32_segment_from_term_postings_seekable_to(
            &doc_lengths,
            &terms,
            &mut prefixed_seekable,
        )
        .unwrap();
        let prefixed = prefixed_seekable.into_inner();

        assert_eq!(from_terms, expected);
        assert_eq!(written, expected);
        assert_eq!(seekable.into_inner(), expected);
        assert_eq!(&prefixed[..13], &[0xaa; 13]);
        assert_eq!(&prefixed[13..], expected);
        RawSegment::open(&written).unwrap();
        RawSegment::open(&prefixed[13..]).unwrap();
    }

    #[test]
    fn raw_segment_term_postings_writer_streams_posting_blocks_to_sink() {
        struct WriteLimit {
            bytes: Vec<u8>,
            max_write_len: usize,
            largest_write_len: usize,
        }

        impl Write for WriteLimit {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if buf.len() > self.max_write_len {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "write exceeded test limit",
                    ));
                }
                self.largest_write_len = self.largest_write_len.max(buf.len());
                self.bytes.extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let doc_lengths: Vec<_> = (1..=300).map(|doc_id| (doc_id, 1)).collect();
        let postings: Vec<_> = (1..=300).map(|doc_id| (doc_id, 1)).collect();
        let terms = [RawTermPostingList::new(7, &postings)];
        let expected = write_u64_u32_segment_from_term_postings(&doc_lengths, &terms).unwrap();
        let mut writer = WriteLimit {
            bytes: Vec::new(),
            max_write_len: 300,
            largest_write_len: 0,
        };

        write_u64_u32_segment_from_term_postings_to(&doc_lengths, &terms, &mut writer).unwrap();

        assert_eq!(writer.bytes, expected);
        assert!(writer.largest_write_len <= 300);
        RawSegment::open(&writer.bytes).unwrap();
    }

    #[test]
    fn raw_segment_term_postings_writer_rejects_invalid_doc_metadata() {
        let postings = [(1, 1)];
        let terms = [RawTermPostingList::new(7, &postings)];

        let decreasing_docs = [(5, 1), (2, 1)];
        assert_eq!(
            write_u64_u32_segment_from_term_postings(&decreasing_docs, &terms).unwrap_err(),
            Error::UnsortedDocId {
                previous: 5,
                current: 2
            }
        );

        let duplicate_docs = [(5, 1), (5, 1)];
        assert_eq!(
            write_u64_u32_segment_from_term_postings(&duplicate_docs, &terms).unwrap_err(),
            Error::DuplicateDocId { doc_id: 5 }
        );
    }

    #[test]
    fn raw_segment_term_postings_writer_rejects_invalid_terms() {
        let docs = [(1, 1), (2, 1), (3, 1)];
        let postings = [(1, 1)];
        let term_7 = RawTermPostingList::new(7, &postings);
        let term_5 = RawTermPostingList::new(5, &postings);
        assert_eq!(
            write_u64_u32_segment_from_term_postings(&docs, &[term_7, term_5]).unwrap_err(),
            Error::UnsortedTermId {
                previous: 7,
                current: 5
            }
        );
        assert_eq!(
            write_u64_u32_segment_from_term_postings(&docs, &[term_7, term_7]).unwrap_err(),
            Error::DuplicateTermId { term_id: 7 }
        );

        let empty: [(DocId, u32); 0] = [];
        assert_eq!(
            write_u64_u32_segment_from_term_postings(&docs, &[RawTermPostingList::new(7, &empty)])
                .unwrap_err(),
            Error::EmptyPostingList { term_id: 7 }
        );
    }

    #[test]
    fn raw_segment_term_postings_writer_rejects_invalid_postings() {
        let docs = [(1, 1), (2, 1), (3, 1)];

        let decreasing = [(2, 1), (1, 1)];
        assert_eq!(
            write_u64_u32_segment_from_term_postings(
                &docs,
                &[RawTermPostingList::new(7, &decreasing)]
            )
            .unwrap_err(),
            Error::NonIncreasingDocId {
                term_id: 7,
                index: 1
            }
        );

        let duplicate = [(2, 1), (2, 1)];
        assert_eq!(
            write_u64_u32_segment_from_term_postings(
                &docs,
                &[RawTermPostingList::new(7, &duplicate)]
            )
            .unwrap_err(),
            Error::NonIncreasingDocId {
                term_id: 7,
                index: 1
            }
        );

        let zero = [(2, 0)];
        assert_eq!(
            write_u64_u32_segment_from_term_postings(&docs, &[RawTermPostingList::new(7, &zero)])
                .unwrap_err(),
            Error::ZeroWeight {
                doc_id: 2,
                term_id: 7
            }
        );

        let unknown = [(99, 1)];
        assert_eq!(
            write_u64_u32_segment_from_term_postings(
                &docs,
                &[RawTermPostingList::new(7, &unknown)]
            )
            .unwrap_err(),
            Error::UnknownDocId {
                term_id: 7,
                doc_id: 99
            }
        );
    }

    #[test]
    fn raw_segment_writer_propagates_io_error() {
        struct FailingWriter;

        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let term = vec![(10, 1)];
        let docs = vec![RawDocument::new(1, &term)];
        let err = write_u64_u32_segment_to(&docs, &mut FailingWriter).unwrap_err();

        assert!(matches!(err, RawSegmentWriteError::Io { .. }));
    }

    #[test]
    fn raw_segment_reads_legacy_unchecksummed_fixture() {
        let bytes = include_bytes!("../tests/fixtures/raw_v3_flags0.segment");
        let segment = RawSegment::open(bytes).unwrap();

        assert!(!segment.meta().has_checksums());
        assert_eq!(segment.num_docs(), 3);
        assert_eq!(segment.term_ids().unwrap(), vec![7, 9, 11]);
        assert_eq!(collect_postings(&segment, 7), vec![(1, 3), (4, 1)]);
    }

    #[test]
    fn raw_segment_records_multi_block_postings_metadata() {
        let term = vec![(7, 1)];
        let docs: Vec<_> = (0..130u32)
            .map(|doc_id| RawDocument::new(doc_id, term.as_slice()))
            .collect();

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let blocks = segment.posting_blocks(7).unwrap();

        assert_eq!(segment.meta().block_size(), DEFAULT_BLOCK_SIZE);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].base_doc_id(), 0);
        assert_eq!(blocks[0].last_doc_id(), 127);
        assert_eq!(blocks[0].max_weight(), 1);
        assert_eq!(blocks[1].base_doc_id(), 127);
        assert_eq!(blocks[1].last_doc_id(), 129);
        assert_eq!(blocks[1].max_weight(), 1);
        assert_eq!(collect_postings(&segment, 7).len(), 130);
        let first_block = collect_block(&segment, 7, 0);
        assert_eq!(first_block.len(), 128);
        assert_eq!(first_block[0], (0, 1));
        assert_eq!(first_block[127], (127, 1));
        assert_eq!(collect_block(&segment, 7, 1), vec![(128, 1), (129, 1)]);
        let first_block_with_lens = collect_block_with_lens(&segment, 7, 0);
        assert_eq!(first_block_with_lens.len(), 128);
        assert_eq!(first_block_with_lens[0], (0, 1, 1));
        assert_eq!(first_block_with_lens[127], (127, 1, 1));
        assert_eq!(
            collect_block_with_lens(&segment, 7, 1),
            vec![(128, 1, 1), (129, 1, 1)]
        );
        assert!(collect_block(&segment, 999, 0).is_empty());
        assert!(collect_block_with_lens(&segment, 999, 0).is_empty());
        assert!(segment.posting_block_postings(7, 2).is_err());
        assert!(segment
            .for_each_posting_block_with_document_len(7, 2, |_, _, _| {})
            .is_err());
    }

    #[test]
    fn raw_segment_records_block_max_weights() {
        let weighted_terms: Vec<_> = (0..130u32)
            .map(|doc_id| {
                let weight = if doc_id == 12 {
                    9
                } else if doc_id == 129 {
                    17
                } else {
                    1
                };
                vec![(7, weight)]
            })
            .collect();
        let docs: Vec<_> = weighted_terms
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let blocks = segment.posting_blocks(7).unwrap();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].max_weight(), 9);
        assert_eq!(blocks[1].max_weight(), 17);
    }

    #[test]
    fn raw_segment_file_reads_metadata_and_posting_ranges() {
        let doc_a = vec![(10, 4), (20, 2)];
        let doc_b = vec![(10, 1)];
        let doc_c = vec![(20, 1)];
        let docs = vec![
            RawDocument::new(5, &doc_a),
            RawDocument::new(2, &doc_b),
            RawDocument::new(9, &doc_c),
        ];
        let bytes = write_u64_u32_segment(&docs).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();

        let mut segment = RawSegmentFile::open(&path).unwrap();

        assert_eq!(segment.num_docs(), 3);
        assert_eq!(segment.meta().term_count(), 2);
        assert_eq!(segment.document_len(5).unwrap(), Some(6));
        assert_eq!(segment.document_len(999).unwrap(), None);
        let mut document_lengths = Vec::new();
        segment
            .for_each_document_len(|doc_id, len| document_lengths.push((doc_id, len)))
            .unwrap();
        assert_eq!(document_lengths, vec![(2, 1), (5, 6), (9, 1)]);
        assert_eq!(segment.df(10).unwrap(), 2);
        assert_eq!(segment.total_weight(10).unwrap(), 5);
        assert_eq!(segment.max_weight(10).unwrap(), 4);
        assert_eq!(segment.term_ids().unwrap(), vec![10, 20]);
        assert_eq!(segment.postings(10).unwrap(), vec![(2, 1), (5, 4)]);
        let mut visited = Vec::new();
        segment
            .for_each_posting(10, |doc_id, weight| visited.push((doc_id, weight)))
            .unwrap();
        assert_eq!(visited, vec![(2, 1), (5, 4)]);
        let mut visited_with_lens = Vec::new();
        segment
            .for_each_posting_with_document_len(10, |doc_id, weight, doc_len| {
                visited_with_lens.push((doc_id, weight, doc_len));
            })
            .unwrap();
        assert_eq!(visited_with_lens, vec![(2, 1, 1), (5, 4, 6)]);
        assert!(segment.postings(999).unwrap().is_empty());

        let blocks = segment.posting_blocks(10).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].max_weight(), 4);
        assert_eq!(
            segment.posting_block_postings(10, 0).unwrap(),
            vec![(2, 1), (5, 4)]
        );
        let mut block_with_lens = Vec::new();
        segment
            .for_each_posting_block_with_document_len(10, 0, |doc_id, weight, doc_len| {
                block_with_lens.push((doc_id, weight, doc_len));
            })
            .unwrap();
        assert_eq!(block_with_lens, vec![(2, 1, 1), (5, 4, 6)]);
        let mut missing_block_with_lens = Vec::new();
        segment
            .for_each_posting_block_with_document_len(999, 0, |doc_id, weight, doc_len| {
                missing_block_with_lens.push((doc_id, weight, doc_len));
            })
            .unwrap();
        assert!(missing_block_with_lens.is_empty());
        assert!(segment
            .for_each_posting_block_with_document_len(10, 1, |_, _, _| {})
            .is_err());
        assert_eq!(segment.candidates_all_terms(&[10, 20]).unwrap(), vec![5]);
        assert_eq!(
            segment.candidates_any_terms(&[10, 20]).unwrap(),
            vec![2, 5, 9]
        );
        assert_eq!(
            segment
                .plan_candidates(
                    &[10, 20],
                    PlannerConfig {
                        max_candidate_ratio: 2.0,
                        max_candidates: 10,
                    },
                )
                .unwrap(),
            CandidatePlan::Candidates(vec![2, 5, 9])
        );
    }

    fn checksum_test_segment() -> Vec<u8> {
        let terms_a = vec![(7u64, 3u32), (9, 1)];
        let terms_b = vec![(7u64, 1u32)];
        let terms_c = vec![(9u64, 5u32), (11, 2)];
        let docs = vec![
            RawDocument::new(1, &terms_a),
            RawDocument::new(4, &terms_b),
            RawDocument::new(9, &terms_c),
        ];
        write_u64_u32_segment(&docs).unwrap()
    }

    #[test]
    fn pre_checksum_segments_still_open_and_read() {
        // Fixture written by the pre-checksum writer (flags = 0): readers must
        // keep accepting it, skipping verification.
        let bytes: &[u8] = include_bytes!("../tests/fixtures/raw_v3_flags0.segment");
        let segment = RawSegment::open(bytes).unwrap();
        assert!(!segment.meta().has_checksums());
        assert_eq!(segment.num_docs(), 3);
        assert_eq!(collect_postings(&segment, 7), vec![(1, 3), (4, 1)]);
        assert_eq!(collect_postings(&segment, 9), vec![(1, 1), (9, 5)]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v3.segment");
        std::fs::write(&path, bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        assert_eq!(file_segment.postings(7).unwrap(), vec![(1, 3), (4, 1)]);
    }

    #[test]
    fn new_segments_carry_checksums_and_round_trip() {
        let bytes = checksum_test_segment();
        let meta = parse_header(&bytes).unwrap();
        assert!(meta.has_checksums());
        let segment = RawSegment::open(&bytes).unwrap();
        assert_eq!(collect_postings(&segment, 7), vec![(1, 3), (4, 1)]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v4.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        assert_eq!(file_segment.postings(9).unwrap(), vec![(1, 1), (9, 5)]);
    }

    #[test]
    fn corrupt_directory_sections_fail_at_open() {
        let clean = checksum_test_segment();
        let meta = parse_header(&clean).unwrap();
        let sections = [
            (meta.term_dir_offset as usize, "term directory"),
            (meta.doc_meta_offset as usize, "doc metadata"),
            (doc_meta_end(meta).unwrap() as usize, "block directory"),
        ];
        for (offset, section) in sections {
            let mut corrupt = clean.clone();
            corrupt[offset] ^= 0xFF;
            let err = RawSegment::open(&corrupt).unwrap_err();
            assert_eq!(
                err,
                Error::ChecksumMismatch { section },
                "byte-backed open must reject a corrupt {section}"
            );

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("corrupt.segment");
            std::fs::write(&path, &corrupt).unwrap();
            let err = RawSegmentFile::open(&path).unwrap_err();
            assert!(
                err.to_string().contains("checksum mismatch"),
                "file-backed open must reject a corrupt {section}: {err}"
            );
        }
    }

    #[test]
    fn corrupt_posting_block_fails_reads_but_spares_siblings() {
        let clean = checksum_test_segment();
        let meta = parse_header(&clean).unwrap();
        // First postings byte belongs to term 7's block (terms are written in
        // ascending id order).
        let mut corrupt = clean.clone();
        corrupt[meta.postings_offset as usize] ^= 0xFF;

        // Posting payload checksums verify when the touched block is read; an
        // untouched sibling block still reads.
        let segment = RawSegment::open(&corrupt).unwrap();
        let err = segment.postings(7).unwrap_err();
        assert_eq!(
            err,
            Error::ChecksumMismatch {
                section: "posting block"
            }
        );
        let err = segment
            .for_each_posting_block_with_document_len(7, 0, |_, _, _| {})
            .unwrap_err();
        assert_eq!(
            err,
            Error::ChecksumMismatch {
                section: "posting block"
            }
        );
        assert_eq!(collect_postings(&segment, 11), vec![(9, 2)]);

        // File-backed behavior matches: directories are intact, so open
        // succeeds and the corrupted block fails on first read.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.segment");
        std::fs::write(&path, &corrupt).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        let err = file_segment.postings(7).unwrap_err();
        assert!(
            err.to_string().contains("checksum mismatch"),
            "reading the corrupted block must fail: {err}"
        );
        let err = file_segment
            .for_each_posting_block_with_document_len(7, 0, |_, _, _| {})
            .unwrap_err();
        assert!(
            err.to_string().contains("checksum mismatch"),
            "reading the corrupted block with doc lengths must fail: {err}"
        );
        assert_eq!(
            file_segment.postings(11).unwrap(),
            vec![(9, 2)],
            "a block the corruption did not touch must still read"
        );
    }

    #[test]
    fn unknown_header_flag_bits_are_rejected() {
        let mut bytes = checksum_test_segment();
        bytes[12] |= 0x2;
        let err = RawSegment::open(&bytes).unwrap_err();
        assert!(matches!(err, Error::UnsupportedFlags { .. }));
    }

    #[test]
    fn raw_segment_file_top_k_weighted_matches_byte_backed() {
        let weighted_terms: Vec<Vec<(RawTermId, u32)>> = (0..140u32)
            .map(|doc_id| {
                vec![
                    (7, 1 + (doc_id % 11)),
                    ((doc_id % 5) as RawTermId, 1 + (doc_id % 3)),
                    (100 + (doc_id % 13) as RawTermId, 2),
                ]
            })
            .collect();
        let docs: Vec<_> = weighted_terms
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new((doc_id as DocId) * 37, terms))
            .collect();

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        for query in [
            vec![(7, 1.25)],
            vec![(7, -1.0)],
            vec![(7, 1.0), (3, 2.0), (3, -0.5)],
            vec![(999, 1.0)],
        ] {
            for k in [0usize, 1, 5, 32] {
                assert_eq!(
                    file_segment.top_k_weighted_u32(&query, k).unwrap(),
                    segment.top_k_weighted_u32(&query, k).unwrap(),
                    "query {query:?} k={k}"
                );
            }
        }
    }

    #[test]
    fn raw_segment_file_blocked_entry_traversal_matches_byte_backed() {
        let weighted_terms: Vec<Vec<(RawTermId, u32)>> = (0..260u32)
            .map(|doc_id| vec![(7, 1 + (doc_id % 3)), (11, 1 + (doc_id % 5))])
            .collect();
        let docs: Vec<_> = weighted_terms
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let entry = segment.term_entry(7).unwrap().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        let mut blocked = Vec::new();
        file_segment
            .for_each_posting_in_entry_blocks(entry, |doc_id, weight| {
                blocked.push((doc_id, weight));
            })
            .unwrap();

        assert_eq!(blocked, collect_postings(&segment, 7));

        for block_index in 0..segment.posting_blocks(7).unwrap().len() as u32 {
            let expected = collect_block_with_lens(&segment, 7, block_index);
            let mut got = Vec::new();
            file_segment
                .for_each_posting_block_with_document_len(
                    7,
                    block_index,
                    |doc_id, weight, doc_len| {
                        got.push((doc_id, weight, doc_len));
                    },
                )
                .unwrap();
            assert_eq!(got, expected, "block {block_index}");
        }
    }

    #[test]
    fn block_max_pruning_skips_a_block_and_matches_brute_force() {
        // Oracle test for the block-skip at top_k_single_raw_term: block 0
        // (128 entries, weights ~100) fills the top-k, then block 1's max
        // (weight 3) falls below the threshold and the block is skipped.
        // The result must still equal an unpruned brute force over the raw
        // postings; the skip precondition is asserted explicitly so the test
        // stays pinned to the pruning regime if constants drift.
        let term: RawTermId = 7;
        let query_weight = 1.5f32;
        let k = 10usize;

        let weighted: Vec<Vec<(RawTermId, u32)>> = (0..200u32)
            .map(|doc_id| {
                let weight = if doc_id < DEFAULT_BLOCK_SIZE {
                    100 + (doc_id % 17)
                } else {
                    1 + (doc_id % 3)
                };
                vec![(term, weight)]
            })
            .collect();
        let docs: Vec<_> = weighted
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();
        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        // Brute-force oracle from the raw postings, not the pruned scorer.
        let mut expected: Vec<(DocId, f32)> = collect_postings(&segment, term)
            .into_iter()
            .map(|(doc_id, weight)| (doc_id, query_weight * weight as f32))
            .collect();
        expected.sort_by(crate::cmp_doc_scores);
        expected.truncate(k);

        // Skip precondition: two blocks, and block 1's bounded contribution
        // cannot beat the k-th best score from block 0.
        let blocks = segment.posting_blocks(term).unwrap();
        assert_eq!(blocks.len(), 2, "test corpus must span two blocks");
        let threshold = expected.last().unwrap().1;
        assert!(
            query_weight * (blocks[1].max_weight() as f32) < threshold,
            "block 1 must be prunable for this test to exercise the skip"
        );

        let query = [(term, query_weight)];
        assert_eq!(segment.top_k_weighted_u32(&query, k).unwrap(), expected);
        assert_eq!(
            file_segment.top_k_weighted_u32(&query, k).unwrap(),
            expected
        );
    }

    #[test]
    fn multi_term_block_max_pruning_matches_byte_backed() {
        let k = 10usize;
        let query = [(10, 1.0), (20, 1.0)];
        let weighted_terms: Vec<Vec<(RawTermId, u32)>> = (0..260u32)
            .map(|doc_id| {
                let first_block = doc_id < DEFAULT_BLOCK_SIZE;
                let term_10 = if first_block {
                    1_000 + (doc_id % 11)
                } else {
                    1
                };
                let term_20 = if first_block { 700 + (doc_id % 7) } else { 1 };
                vec![(10, term_10), (20, term_20)]
            })
            .collect();
        let docs: Vec<_> = weighted_terms
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();
        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        let expected = segment.top_k_weighted_u32(&query, k).unwrap();
        let threshold = expected.last().unwrap().1;
        let term_10_blocks = segment.posting_blocks(10).unwrap();
        let term_20_blocks = segment.posting_blocks(20).unwrap();
        assert!(term_10_blocks.len() > 1);
        assert_eq!(term_10_blocks.len(), term_20_blocks.len());
        assert!(
            term_10_blocks[1].max_weight() as f32 + (term_20_blocks[1].max_weight() as f32)
                < threshold,
            "later aligned blocks must be prunable for this test"
        );

        let query_terms = normalize_weighted_query_terms(&query);
        let mut lists = Vec::with_capacity(query_terms.len());
        let mut total_postings = 0usize;
        for (term_id, query_weight) in query_terms {
            let Some((entry, block_directory)) =
                file_segment.term_entry_with_blocks(term_id).unwrap()
            else {
                panic!("test term should be present");
            };
            total_postings = total_postings.saturating_add(entry.df as usize);
            lists.push((entry, block_directory, query_weight));
        }
        let dense_range = file_segment.dense_doc_id_range().unwrap();

        assert_eq!(
            file_segment
                .top_k_weighted_u32_pruned_blocks(lists, total_postings, dense_range, k)
                .unwrap(),
            expected
        );

        assert_eq!(
            file_segment.top_k_weighted_u32(&query, k).unwrap(),
            expected
        );
    }

    #[test]
    fn multi_term_block_pruning_uses_overlapping_range_max() {
        let k = 10usize;
        let query = [(10, 1.0), (20, 1.0)];
        let weighted_terms: Vec<Vec<(RawTermId, u32)>> = (0..328u32)
            .map(|doc_id| {
                let mut terms = Vec::new();
                if doc_id < 256 {
                    let weight = if doc_id < DEFAULT_BLOCK_SIZE {
                        100
                    } else if (200..210).contains(&doc_id) {
                        20
                    } else {
                        1
                    };
                    terms.push((10, weight));
                }
                if doc_id >= 200 {
                    terms.push((20, 90));
                }
                terms
            })
            .collect();
        let docs: Vec<_> = weighted_terms
            .iter()
            .enumerate()
            .map(|(doc_id, terms)| RawDocument::new(doc_id as DocId, terms))
            .collect();
        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        let expected = segment.top_k_weighted_u32(&query, k).unwrap();
        assert_eq!(
            expected,
            (200u32..210u32)
                .map(|doc_id| (doc_id, 110.0f32))
                .collect::<Vec<_>>()
        );

        let query_terms = normalize_weighted_query_terms(&query);
        let mut lists = Vec::with_capacity(query_terms.len());
        let mut total_postings = 0usize;
        for (term_id, query_weight) in query_terms {
            let Some((entry, block_directory)) =
                file_segment.term_entry_with_blocks(term_id).unwrap()
            else {
                panic!("test term should be present");
            };
            total_postings = total_postings.saturating_add(entry.df as usize);
            lists.push((entry, block_directory, query_weight));
        }
        let scoring_lists = file_segment
            .prepare_raw_block_scoring_lists(lists.clone())
            .unwrap();
        assert!(
            raw_block_range_upper_bound(scoring_lists[0].blocks[1], &scoring_lists) >= 110.0,
            "the term-10 second block needs the overlapping term-20 max"
        );

        let dense_range = file_segment.dense_doc_id_range().unwrap();

        assert_eq!(
            file_segment
                .top_k_weighted_u32_pruned_blocks(lists, total_postings, dense_range, k)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn raw_segment_files_top_k_weighted_matches_in_memory_index() {
        let first = [
            (1, vec![(10, 3), (20, 1)]),
            (2, vec![(20, 5)]),
            (3, vec![(30, 2)]),
        ];
        let second = [
            (10, vec![(10, 1), (30, 3)]),
            (11, vec![(30, 2), (40, 2)]),
            (12, vec![(40, 4)]),
        ];

        let mut idx: PostingsIndex<RawTermId> = PostingsIndex::new();
        for (doc_id, terms) in first.iter().chain(second.iter()) {
            let mut expanded = Vec::new();
            for &(term_id, weight) in terms {
                for _ in 0..weight {
                    expanded.push(term_id);
                }
            }
            idx.add_document(*doc_id, &expanded).unwrap();
        }

        let first_docs: Vec<_> = first
            .iter()
            .map(|(doc_id, terms)| RawDocument::new(*doc_id, terms))
            .collect();
        let second_docs: Vec<_> = second
            .iter()
            .map(|(doc_id, terms)| RawDocument::new(*doc_id, terms))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, write_u64_u32_segment(&first_docs).unwrap()).unwrap();
        std::fs::write(&second_path, write_u64_u32_segment(&second_docs).unwrap()).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];

        let query = vec![(10, 1.5), (30, 2.0), (40, -0.25)];
        let memory_query: Vec<(&RawTermId, f32)> = query
            .iter()
            .map(|(term_id, weight)| (term_id, *weight))
            .collect();

        assert_eq!(
            top_k_weighted_u32_files(&mut segments, &query, 4).unwrap(),
            idx.top_k_weighted(&memory_query, 4)
        );
    }

    #[test]
    fn raw_segment_dense_scoring_offsets_segment_doc_id_range() {
        let docs = [
            RawDocument::new(10_000, &[(10, 2), (20, 1)]),
            RawDocument::new(10_002, &[(20, 5)]),
            RawDocument::new(10_003, &[(10, 1)]),
        ];
        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("high-base.raw");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();

        assert_eq!(segment.dense_doc_id_range().unwrap(), Some((10_000, 4)));
        assert_eq!(
            file_segment.dense_doc_id_range().unwrap(),
            Some((10_000, 4))
        );
        assert_eq!(
            segment.candidates_any_terms(&[10, 20]).unwrap(),
            vec![10_000, 10_002, 10_003]
        );
        assert_eq!(
            file_segment.candidates_any_terms(&[10, 20]).unwrap(),
            vec![10_000, 10_002, 10_003]
        );

        let query = [(10, 1.0), (20, 1.0)];
        let expected = vec![(10_002, 5.0), (10_000, 3.0), (10_003, 1.0)];
        assert_eq!(segment.top_k_weighted_u32(&query, 3).unwrap(), expected);
        assert_eq!(
            file_segment.top_k_weighted_u32(&query, 3).unwrap(),
            expected
        );
    }

    #[test]
    fn raw_segment_files_do_not_prune_equal_bound_tie() {
        let first_docs = [RawDocument::new(10, &[(7, 5)])];
        let second_docs = [RawDocument::new(1, &[(7, 5)])];
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, write_u64_u32_segment(&first_docs).unwrap()).unwrap();
        std::fs::write(&second_path, write_u64_u32_segment(&second_docs).unwrap()).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];

        let result = top_k_weighted_u32_files_with_stats(&mut segments, &[(7, 1.0)], 1).unwrap();
        assert_eq!(result.hits, vec![(1, 5.0)]);
        assert_eq!(
            result.stats,
            RawTopKSearchStats {
                segments_seen: 2,
                segments_scored: 2,
                segments_pruned: 0,
            }
        );
    }

    #[test]
    fn raw_segment_files_report_threshold_pruning() {
        let first_docs = [RawDocument::new(10, &[(7, 100)])];
        let second_docs = [RawDocument::new(1, &[(7, 1)])];
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, write_u64_u32_segment(&first_docs).unwrap()).unwrap();
        std::fs::write(&second_path, write_u64_u32_segment(&second_docs).unwrap()).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];

        let result = top_k_weighted_u32_files_with_stats(&mut segments, &[(7, 1.0)], 1).unwrap();

        assert_eq!(result.hits, vec![(10, 100.0)]);
        assert_eq!(
            result.stats,
            RawTopKSearchStats {
                segments_seen: 2,
                segments_scored: 1,
                segments_pruned: 1,
            }
        );
    }

    #[test]
    fn raw_segment_files_return_empty_for_absent_terms() {
        let first_docs = [RawDocument::new(10, &[(7, 5)])];
        let second_docs = [RawDocument::new(1, &[(8, 5)])];
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.raw");
        let second_path = dir.path().join("second.raw");
        std::fs::write(&first_path, write_u64_u32_segment(&first_docs).unwrap()).unwrap();
        std::fs::write(&second_path, write_u64_u32_segment(&second_docs).unwrap()).unwrap();
        let mut first_segment = RawSegmentFile::open(&first_path).unwrap();
        let mut second_segment = RawSegmentFile::open(&second_path).unwrap();
        let mut segments = [&mut first_segment, &mut second_segment];

        let result = top_k_weighted_u32_files_with_stats(&mut segments, &[(99, 1.0)], 10).unwrap();

        assert!(result.hits.is_empty());
        assert_eq!(
            result.stats,
            RawTopKSearchStats {
                segments_seen: 2,
                segments_scored: 0,
                segments_pruned: 2,
            }
        );
    }

    #[test]
    fn raw_segment_file_rejects_short_header_as_raw_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.segment");
        std::fs::write(&path, b"short").unwrap();

        match RawSegmentFile::open(&path).unwrap_err() {
            RawSegmentFileError::Segment {
                source: Error::Truncated { section: "header" },
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn raw_top_k_weighted_scores_sparse_inner_product() {
        let doc_a = vec![(10, 2), (20, 1)];
        let doc_b = vec![(10, 1), (30, 5)];
        let doc_c = vec![(20, 4), (30, 1)];
        let docs = vec![
            RawDocument::new(7, &doc_a),
            RawDocument::new(1_000_000, &doc_b),
            RawDocument::new(42, &doc_c),
        ];

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();

        assert_eq!(
            segment
                .top_k_weighted_u32(&[(10, 2.0), (20, 0.5), (30, 1.0)], 3)
                .unwrap(),
            vec![(1_000_000, 7.0), (7, 4.5), (42, 3.0)]
        );
    }

    #[test]
    fn raw_top_k_weighted_accumulates_duplicate_query_terms() {
        let doc_a = vec![(10, 2)];
        let doc_b = vec![(10, 1)];
        let docs = vec![RawDocument::new(4, &doc_a), RawDocument::new(2, &doc_b)];

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();

        assert_eq!(
            segment
                .top_k_weighted_u32(&[(10, 1.0), (10, 0.5)], 1)
                .unwrap(),
            vec![(4, 3.0)]
        );
        assert!(segment
            .top_k_weighted_u32(&[(10, 1.0), (10, -1.0)], 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn raw_top_k_weighted_handles_cancellation_and_ties() {
        let doc_a = vec![(10, 2), (20, 2)];
        let doc_b = vec![(10, 2)];
        let doc_c = vec![(10, 1), (30, 4)];
        let docs = vec![
            RawDocument::new(4, &doc_a),
            RawDocument::new(2, &doc_b),
            RawDocument::new(3, &doc_c),
        ];

        let bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();

        assert_eq!(
            segment
                .top_k_weighted_u32(&[(10, 1.0), (20, -1.0), (30, 0.25)], 10)
                .unwrap(),
            vec![(2, 2.0), (3, 2.0)]
        );
        assert!(segment
            .top_k_weighted_u32(&[(999, 1.0)], 10)
            .unwrap()
            .is_empty());
        assert!(segment
            .top_k_weighted_u32(&[(10, 1.0)], 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn raw_segment_block_iterator_validates_block_last_doc() {
        let term = vec![(7, 1)];
        let docs: Vec<_> = (0..130u32)
            .map(|doc_id| RawDocument::new(doc_id, term.as_slice()))
            .collect();

        let mut bytes = strip_checksums_for_test(&write_u64_u32_segment(&docs).unwrap());
        let segment = RawSegment::open(&bytes).unwrap();
        let second_block = doc_meta_end(segment.meta()).unwrap() as usize + BLOCK_ENTRY_LEN;
        bytes[second_block + 4..second_block + 8].copy_from_slice(&128u32.to_le_bytes());

        let segment = RawSegment::open(&bytes).unwrap();
        let err = segment
            .posting_block_postings(7, 1)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap_err();
        assert!(matches!(err, Error::InvalidLayout { .. }));
    }

    #[test]
    fn raw_segment_validates_block_metadata_lazily() {
        let terms = vec![(1, 1)];
        let docs = vec![RawDocument::new(1, &terms)];
        let mut bytes = strip_checksums_for_test(&write_u64_u32_segment(&docs).unwrap());
        bytes[HEADER_LEN + 40..HEADER_LEN + 48].copy_from_slice(&0u64.to_le_bytes());

        let segment = RawSegment::open(&bytes).unwrap();
        assert_eq!(segment.df(1).unwrap(), 1);
        assert!(matches!(
            segment.posting_blocks(1),
            Err(Error::InvalidLayout {
                reason: "block range is outside block directory section"
            })
        ));
    }

    #[test]
    fn raw_segment_rejects_bad_magic() {
        let terms = vec![(1, 1)];
        let docs = vec![RawDocument::new(1, &terms)];
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
        bytes[0] = b'X';

        assert_eq!(RawSegment::open(&bytes).unwrap_err(), Error::BadMagic);
    }

    #[test]
    fn raw_segment_rejects_unsupported_version() {
        let terms = vec![(1, 1)];
        let docs = vec![RawDocument::new(1, &terms)];
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
        bytes[8..12].copy_from_slice(&999u32.to_le_bytes());

        assert_eq!(
            RawSegment::open(&bytes).unwrap_err(),
            Error::UnsupportedVersion { version: 999 }
        );
    }

    #[test]
    fn raw_segment_rejects_checksummed_directory_corruption() {
        let terms = vec![(1, 1)];
        let docs = vec![RawDocument::new(1, &terms)];
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
        bytes[HEADER_LEN] ^= 0x01;

        assert_eq!(
            RawSegment::open(&bytes).unwrap_err(),
            Error::ChecksumMismatch {
                section: "term directory"
            }
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        match RawSegmentFile::open(&path).unwrap_err() {
            RawSegmentFileError::Segment {
                source:
                    Error::ChecksumMismatch {
                        section: "term directory",
                    },
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn raw_segment_file_rejects_checksummed_posting_corruption_lazily() {
        let terms = vec![(7, 3)];
        let docs = vec![RawDocument::new(3, &terms)];
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
        let segment = RawSegment::open(&bytes).unwrap();
        let entry = segment.term_entry(7).unwrap().unwrap();
        bytes[entry.postings_offset as usize] ^= 0x01;

        let segment = RawSegment::open(&bytes).unwrap();
        assert_eq!(
            segment.postings(7).unwrap_err(),
            Error::ChecksumMismatch {
                section: "posting block"
            }
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.segment");
        std::fs::write(&path, &bytes).unwrap();
        let mut file_segment = RawSegmentFile::open(&path).unwrap();
        match file_segment.postings(7).unwrap_err() {
            RawSegmentFileError::Segment {
                source:
                    Error::ChecksumMismatch {
                        section: "posting block",
                    },
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn raw_segment_rejects_corrupt_posting_varint() {
        let terms = vec![(7, 1)];
        let docs = vec![RawDocument::new(3, &terms)];
        let mut bytes = strip_checksums_for_test(&write_u64_u32_segment(&docs).unwrap());
        let segment = RawSegment::open(&bytes).unwrap();
        let entry = segment.term_entry(7).unwrap().unwrap();
        let offset = entry.postings_offset as usize;
        bytes[offset] = 0x80;
        bytes[offset + 1] = 0x80;
        let segment = RawSegment::open(&bytes).unwrap();
        let err = segment.postings(7).unwrap().next().unwrap().unwrap_err();

        assert_eq!(
            err,
            Error::InvalidVarint {
                term_id: 7,
                index: 0
            }
        );
    }

    #[test]
    fn raw_segment_rejects_corrupt_zero_weight_posting() {
        let terms = vec![(7, 1)];
        let docs = vec![RawDocument::new(3, &terms)];
        let mut bytes = strip_checksums_for_test(&write_u64_u32_segment(&docs).unwrap());
        let segment = RawSegment::open(&bytes).unwrap();
        let entry = segment.term_entry(7).unwrap().unwrap();
        let offset = entry.postings_offset as usize;
        bytes[offset + 1] = 0;
        let segment = RawSegment::open(&bytes).unwrap();
        let err = segment.postings(7).unwrap().next().unwrap().unwrap_err();

        assert_eq!(
            err,
            Error::ZeroWeight {
                doc_id: 3,
                term_id: 7
            }
        );
    }

    #[test]
    fn raw_candidates_still_validate_current_intersection_list() {
        let term_a = vec![(1, 1)];
        let term_b = vec![(2, 1)];
        let term_c = vec![(2, 1)];
        let docs = vec![
            RawDocument::new(1, &term_a),
            RawDocument::new(2, &term_b),
            RawDocument::new(100, &term_c),
        ];
        let mut bytes = strip_checksums_for_test(&write_u64_u32_segment(&docs).unwrap());
        let segment = RawSegment::open(&bytes).unwrap();
        let entry = segment.term_entry(2).unwrap().unwrap();
        let offset = entry.postings_offset as usize;
        bytes[offset + 3] = 0;

        let segment = RawSegment::open(&bytes).unwrap();
        let err = segment.candidates_all_terms(&[1, 2]).unwrap_err();

        assert_eq!(
            err,
            Error::ZeroWeight {
                doc_id: 100,
                term_id: 2
            }
        );
    }

    #[test]
    fn raw_writer_rejects_duplicate_docs_and_zero_weights() {
        let terms = vec![(1, 1)];
        let docs = vec![RawDocument::new(1, &terms), RawDocument::new(1, &terms)];
        assert_eq!(
            write_u64_u32_segment(&docs).unwrap_err(),
            Error::DuplicateDocId { doc_id: 1 }
        );

        let zero = vec![(1, 0)];
        let docs = vec![RawDocument::new(1, &zero)];
        assert_eq!(
            write_u64_u32_segment(&docs).unwrap_err(),
            Error::ZeroWeight {
                doc_id: 1,
                term_id: 1
            }
        );
    }

    proptest! {
        #[test]
        fn raw_segment_matches_in_memory_candidates_and_postings(
            docs in prop::collection::vec(
                prop::collection::vec((0u8..12, 1u8..4), 0..16),
                0..24
            ),
            query in prop::collection::vec(0u8..12, 0..8),
            weighted_query in prop::collection::vec((0u8..12, -3i8..4), 0..8),
            stride in prop::sample::select(vec![1u32, 37u32]),
        ) {
            let weighted_docs: Vec<Vec<(RawTermId, u32)>> = docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&(term, weight)| (term as RawTermId, weight as u32))
                        .collect()
                })
                .collect();
            let raw_docs: Vec<RawDocument<'_>> = weighted_docs
                .iter()
                .enumerate()
                .map(|(i, terms)| RawDocument::new((i as DocId) * stride, terms))
                .collect();

            let mut idx: PostingsIndex<RawTermId> = PostingsIndex::new();
            for (i, terms) in weighted_docs.iter().enumerate() {
                let mut expanded = Vec::new();
                for &(term_id, weight) in terms {
                    for _ in 0..weight {
                        expanded.push(term_id);
                    }
                }
                idx.add_document((i as DocId) * stride, &expanded).unwrap();
            }

            let bytes = write_u64_u32_segment(&raw_docs).unwrap();
            let segment = RawSegment::open(&bytes).unwrap();
            let query_terms: Vec<RawTermId> = query.iter().map(|&term| term as RawTermId).collect();

            prop_assert_eq!(
                segment.candidates_all_terms(&query_terms).unwrap(),
                idx.candidates_all_terms(&query_terms)
            );
            prop_assert_eq!(
                segment.candidates_any_terms(&query_terms).unwrap(),
                idx.candidates(&query_terms)
            );
            prop_assert_eq!(
                segment.plan_candidates(&query_terms, PlannerConfig::default()).unwrap(),
                idx.plan_candidates(&query_terms, PlannerConfig::default())
            );
            let raw_weighted_query: Vec<(RawTermId, f32)> = weighted_query
                .iter()
                .map(|&(term, weight)| (term as RawTermId, weight as f32))
                .collect();
            let memory_weighted_query: Vec<(&RawTermId, f32)> = raw_weighted_query
                .iter()
                .map(|(term, weight)| (term, *weight))
                .collect();
            prop_assert_eq!(
                segment.top_k_weighted_u32(&raw_weighted_query, 5).unwrap(),
                idx.top_k_weighted(&memory_weighted_query, 5)
            );
            let mut file = tempfile::tempfile().unwrap();
            file.write_all(&bytes).unwrap();
            file.flush().unwrap();
            let mut file_segment = RawSegmentFile::from_file(file).unwrap();
            prop_assert_eq!(
                file_segment.candidates_all_terms(&query_terms).unwrap(),
                segment.candidates_all_terms(&query_terms).unwrap()
            );
            prop_assert_eq!(
                file_segment.candidates_any_terms(&query_terms).unwrap(),
                segment.candidates_any_terms(&query_terms).unwrap()
            );
            prop_assert_eq!(
                file_segment.plan_candidates(&query_terms, PlannerConfig::default()).unwrap(),
                segment.plan_candidates(&query_terms, PlannerConfig::default()).unwrap()
            );
            prop_assert_eq!(
                file_segment.term_ids().unwrap(),
                segment.term_ids().unwrap()
            );
            prop_assert_eq!(
                file_segment.top_k_weighted_u32(&raw_weighted_query, 5).unwrap(),
                segment.top_k_weighted_u32(&raw_weighted_query, 5).unwrap()
            );
            for term_id in 0..12u64 {
                prop_assert_eq!(segment.df(term_id).unwrap(), idx.df(&term_id));
                let raw_postings = collect_postings(&segment, term_id);
                let memory_postings: Vec<(DocId, u32)> = idx.postings_iter(&term_id).collect();
                prop_assert_eq!(raw_postings, memory_postings);
            }
            let mut memory_terms: Vec<_> = idx.terms().copied().collect();
            memory_terms.sort_unstable();
            prop_assert_eq!(segment.term_ids().unwrap(), memory_terms);
            for doc_id in idx.document_ids() {
                prop_assert_eq!(segment.document_len(doc_id).unwrap(), Some(idx.document_len(doc_id)));
            }
        }

        #[test]
        fn raw_segment_sorted_writer_matches_unsorted_writer_for_sorted_input(
            docs in prop::collection::vec(
                prop::collection::vec((0u8..12, 1u8..4), 0..16),
                0..24
            ),
            stride in prop::sample::select(vec![1u32, 37u32]),
        ) {
            let weighted_docs: Vec<Vec<(RawTermId, u32)>> = docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&(term, weight)| (term as RawTermId, weight as u32))
                        .collect()
                })
                .collect();
            let raw_docs: Vec<RawDocument<'_>> = weighted_docs
                .iter()
                .enumerate()
                .map(|(i, terms)| RawDocument::new((i as DocId) * stride, terms))
                .collect();

            let expected = write_u64_u32_segment(&raw_docs).unwrap();
            let sorted = write_u64_u32_segment_sorted_from_iter(raw_docs.iter().copied()).unwrap();
            let mut written = Vec::new();
            write_u64_u32_segment_sorted_from_iter_to(raw_docs.iter().copied(), &mut written).unwrap();

            prop_assert_eq!(&sorted, &expected);
            prop_assert_eq!(&written, &expected);
        }

        #[test]
        fn raw_segment_term_postings_writer_matches_unsorted_writer_for_random_input(
            docs in prop::collection::vec(
                prop::collection::vec((0u8..12, 1u8..4), 0..16),
                0..24
            ),
            stride in prop::sample::select(vec![1u32, 37u32]),
        ) {
            let weighted_docs: Vec<Vec<(RawTermId, u32)>> = docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&(term, weight)| (term as RawTermId, weight as u32))
                        .collect()
                })
                .collect();
            let raw_docs: Vec<RawDocument<'_>> = weighted_docs
                .iter()
                .enumerate()
                .map(|(i, terms)| RawDocument::new((i as DocId) * stride, terms))
                .collect();

            let mut doc_lengths = Vec::with_capacity(weighted_docs.len());
            let mut term_storage: BTreeMap<RawTermId, Vec<(DocId, u32)>> = BTreeMap::new();
            for (i, terms) in weighted_docs.iter().enumerate() {
                let doc_id = (i as DocId) * stride;
                let mut doc_len = 0u32;
                let mut per_doc_terms = BTreeMap::new();
                for &(term_id, weight) in terms {
                    doc_len = doc_len.checked_add(weight).unwrap();
                    *per_doc_terms.entry(term_id).or_insert(0u32) += weight;
                }
                doc_lengths.push((doc_id, doc_len));
                for (term_id, weight) in per_doc_terms {
                    term_storage.entry(term_id).or_default().push((doc_id, weight));
                }
            }
            let term_lists: Vec<_> = term_storage
                .iter()
                .map(|(&term_id, postings)| RawTermPostingList::new(term_id, postings))
                .collect();

            let expected = write_u64_u32_segment(&raw_docs).unwrap();
            let term_major =
                write_u64_u32_segment_from_term_postings(&doc_lengths, &term_lists).unwrap();
            let mut written = Vec::new();
            write_u64_u32_segment_from_term_postings_to(
                &doc_lengths,
                &term_lists,
                &mut written,
            )
            .unwrap();

            prop_assert_eq!(&term_major, &expected);
            prop_assert_eq!(&written, &expected);
        }
    }
}
