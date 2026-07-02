//! Byte-backed raw segment reader experiments.
//!
//! This module is intentionally narrow. It supports one local, numeric-term,
//! `u32`-weighted raw segment format so query code can read posting lists
//! without reconstructing a full [`crate::PostingsIndex`].

use crate::codec::varint;
use crate::{CandidatePlan, DocId, PlannerConfig};
use std::collections::{BTreeMap, HashMap};

const MAGIC: &[u8; 8] = b"PSTRW001";
const FOOTER_MAGIC: &[u8; 8] = b"PSTRF001";
const VERSION: u32 = 3;
const HEADER_LEN: usize = 72;
const TERM_ENTRY_LEN: usize = 48;
const DOC_ENTRY_LEN: usize = 8;
const BLOCK_ENTRY_LEN: usize = 24;
const FOOTER_LEN: usize = 12;
const DEFAULT_BLOCK_SIZE: u32 = 128;

/// A numeric term id in the first raw postings format.
pub type RawTermId = u64;

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
    /// Two documents in the input used the same id.
    #[error("duplicate raw segment doc id: {doc_id}")]
    DuplicateDocId {
        /// Duplicate document id.
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
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(Error::BadMagic);
        }

        let version = read_u32_at(bytes, 8, "header")?;
        if version != VERSION {
            return Err(Error::UnsupportedVersion { version });
        }
        let flags = read_u32_at(bytes, 12, "header")?;
        if flags != 0 {
            return Err(Error::UnsupportedFlags { flags });
        }

        let meta = RawSegmentMeta {
            term_count: read_u32_at(bytes, 16, "header")?,
            doc_count: read_u32_at(bytes, 20, "header")?,
            max_doc_id: read_u32_at(bytes, 24, "header")?,
            block_size: read_u32_at(bytes, 28, "header")?,
            total_doc_len: read_u64_at(bytes, 32, "header")?,
            term_dir_offset: read_u64_at(bytes, 40, "header")?,
            doc_meta_offset: read_u64_at(bytes, 48, "header")?,
            postings_offset: read_u64_at(bytes, 56, "header")?,
            footer_offset: read_u64_at(bytes, 64, "header")?,
        };

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
        Ok(RawPostings {
            term_id,
            bytes: &self.bytes[range],
            remaining: entry.df,
            consumed: 0,
            prev_doc_id: 0,
            index: 0,
            failed: false,
        })
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
        Ok(RawPostingBlockPostings {
            term_id,
            bytes: &self.bytes[range],
            consumed: 0,
            base_doc_id: block.base_doc_id,
            last_doc_id: block.last_doc_id,
            prev_doc_id: block.base_doc_id,
            index: 0,
            done: false,
            failed: false,
        })
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

        let dense_slots = usize::try_from(self.meta.max_doc_id)
            .ok()
            .and_then(|max_doc_id| max_doc_id.checked_add(1))
            .unwrap_or(usize::MAX);
        let dense_limit = crate::dense_scratch_limit(self.meta.doc_count as usize);
        if dense_slots <= dense_limit {
            let mut seen = vec![false; dense_slots];
            for entry in entries {
                for posting in RawPostings::from_entry(self.bytes, entry)? {
                    let (doc_id, _) = posting?;
                    seen[doc_id as usize] = true;
                }
            }
            return Ok(seen
                .into_iter()
                .enumerate()
                .filter_map(|(doc_id, hit)| hit.then_some(doc_id as DocId))
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

        let dense_slots = usize::try_from(self.meta.max_doc_id)
            .ok()
            .and_then(|max_doc_id| max_doc_id.checked_add(1))
            .unwrap_or(usize::MAX);
        let dense_limit = crate::dense_scratch_limit(self.meta.doc_count as usize);
        if dense_slots <= dense_limit {
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
                        let slot = doc_id as usize;
                        if scores[slot] == 0.0 {
                            touched.push(doc_id);
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
                        let slot = doc_id as usize;
                        if !seen[slot] {
                            seen[slot] = true;
                            touched.push(doc_id);
                        }
                        scores[slot] += contribution;
                    })?;
                }
            }

            return Ok(crate::top_k_scored_docs(
                touched
                    .into_iter()
                    .map(|doc_id| (doc_id, scores[doc_id as usize])),
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
            for_each_posting_in_block(
                entry.term_id,
                &self.bytes[range],
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
    let mut docs: BTreeMap<DocId, (u32, BTreeMap<RawTermId, u32>)> = BTreeMap::new();
    for doc in documents {
        if docs.contains_key(&doc.doc_id) {
            return Err(Error::DuplicateDocId { doc_id: doc.doc_id });
        }

        let mut doc_len = 0u32;
        let mut terms: BTreeMap<RawTermId, u32> = BTreeMap::new();
        for &(term_id, weight) in doc.terms {
            if weight == 0 {
                return Err(Error::ZeroWeight {
                    doc_id: doc.doc_id,
                    term_id,
                });
            }
            doc_len = doc_len
                .checked_add(weight)
                .ok_or(Error::DocLengthOverflow { doc_id: doc.doc_id })?;
            let accumulated = terms.entry(term_id).or_insert(0);
            *accumulated = accumulated
                .checked_add(weight)
                .ok_or(Error::WeightOverflow {
                    doc_id: doc.doc_id,
                    term_id,
                })?;
        }

        docs.insert(doc.doc_id, (doc_len, terms));
    }

    let mut postings: BTreeMap<RawTermId, Vec<(DocId, u32)>> = BTreeMap::new();
    let mut total_doc_len = 0u64;
    for (&doc_id, (doc_len, terms)) in &docs {
        total_doc_len = total_doc_len.saturating_add(*doc_len as u64);
        for (&term_id, &weight) in terms {
            postings.entry(term_id).or_default().push((doc_id, weight));
        }
    }

    let doc_count = u32::try_from(docs.len()).map_err(|_| Error::TooManyDocuments)?;
    let term_count = u32::try_from(postings.len()).map_err(|_| Error::TooManyTerms)?;
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
    let total_block_count: u64 = postings
        .values()
        .map(|list| {
            let len = list.len() as u64;
            len.div_ceil(DEFAULT_BLOCK_SIZE as u64)
        })
        .sum();
    let block_dir_len = total_block_count
        .checked_mul(BLOCK_ENTRY_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let postings_offset = block_dir_offset
        .checked_add(block_dir_len)
        .ok_or(Error::SegmentTooLarge)?;

    let mut postings_bytes = Vec::new();
    let mut term_entries = Vec::with_capacity(postings.len());
    let mut term_block_directories = Vec::with_capacity(postings.len());
    let mut block_entries = Vec::new();
    for (&term_id, list) in &postings {
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
            let block_base_doc_id = prev_doc_id;
            let block_start_len = postings_bytes.len();
            let mut block_max_weight = 0u32;
            for &(doc_id, weight) in chunk {
                let gap = if first_posting {
                    first_posting = false;
                    doc_id
                } else {
                    doc_id - prev_doc_id
                };
                varint::encode_u32(gap, &mut postings_bytes);
                varint::encode_u32(weight, &mut postings_bytes);
                prev_doc_id = doc_id;
                max_weight = max_weight.max(weight);
                block_max_weight = block_max_weight.max(weight);
                total_weight = total_weight.saturating_add(weight as u64);
            }
            block_entries.push(RawPostingBlockMeta {
                base_doc_id: block_base_doc_id,
                last_doc_id: prev_doc_id,
                postings_offset: postings_offset
                    .checked_add(
                        u64::try_from(block_start_len).map_err(|_| Error::SegmentTooLarge)?,
                    )
                    .ok_or(Error::SegmentTooLarge)?,
                postings_len: u32::try_from(postings_bytes.len() - block_start_len)
                    .map_err(|_| Error::SegmentTooLarge)?,
                max_weight: block_max_weight,
            });
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

    let footer_offset = postings_offset
        .checked_add(u64::try_from(postings_bytes.len()).map_err(|_| Error::SegmentTooLarge)?)
        .ok_or(Error::SegmentTooLarge)?;
    let final_len = footer_offset
        .checked_add(FOOTER_LEN as u64)
        .ok_or(Error::SegmentTooLarge)?;
    let mut out =
        Vec::with_capacity(usize::try_from(final_len).map_err(|_| Error::SegmentTooLarge)?);

    put_header(
        &mut out,
        RawSegmentMeta {
            term_count,
            doc_count,
            max_doc_id: docs.keys().next_back().copied().unwrap_or(0),
            block_size: DEFAULT_BLOCK_SIZE,
            total_doc_len,
            term_dir_offset,
            doc_meta_offset,
            postings_offset,
            footer_offset,
        },
    );
    for (entry, block_directory) in term_entries.into_iter().zip(term_block_directories) {
        put_u64(&mut out, entry.term_id);
        put_u32(&mut out, entry.df);
        put_u32(&mut out, entry.max_weight);
        put_u64(&mut out, entry.total_weight);
        put_u64(&mut out, entry.postings_offset);
        put_u32(&mut out, entry.postings_len);
        put_u32(&mut out, block_directory.block_count);
        put_u64(&mut out, block_directory.blocks_offset);
    }
    for (&doc_id, (doc_len, _)) in &docs {
        put_u32(&mut out, doc_id);
        put_u32(&mut out, *doc_len);
    }
    for block in block_entries {
        put_u32(&mut out, block.base_doc_id);
        put_u32(&mut out, block.last_doc_id);
        put_u64(&mut out, block.postings_offset);
        put_u32(&mut out, block.postings_len);
        put_u32(&mut out, block.max_weight);
    }
    out.extend_from_slice(&postings_bytes);
    out.extend_from_slice(FOOTER_MAGIC);
    put_u32(&mut out, VERSION);

    debug_assert_eq!(out.len(), final_len as usize);
    Ok(out)
}

fn put_header(out: &mut Vec<u8>, meta: RawSegmentMeta) {
    out.extend_from_slice(MAGIC);
    put_u32(out, VERSION);
    put_u32(out, 0);
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

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn validate_layout(bytes: &[u8], meta: RawSegmentMeta) -> Result<(), Error> {
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
    if footer_end != bytes.len() as u64 {
        return Err(Error::InvalidLayout {
            reason: "footer must end at segment end",
        });
    }
    checked_range(
        meta.term_dir_offset,
        term_dir_len,
        bytes.len(),
        "term directory",
    )?;
    checked_range(
        meta.doc_meta_offset,
        doc_meta_len,
        bytes.len(),
        "doc metadata",
    )?;
    checked_range(
        doc_meta_end,
        meta.postings_offset - doc_meta_end,
        bytes.len(),
        "block directory",
    )?;
    checked_range(
        meta.postings_offset,
        meta.footer_offset - meta.postings_offset,
        bytes.len(),
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

        assert_eq!(segment.num_docs(), 3);
        assert_eq!(segment.meta().term_count(), 3);
        assert_eq!(segment.meta().max_doc_id(), 9);
        assert_eq!(segment.meta().total_doc_len(), 9);
        assert_eq!(segment.avg_doc_len(), 3.0);
        assert_eq!(segment.document_len(5).unwrap(), Some(6));
        assert_eq!(segment.document_len(2).unwrap(), Some(2));
        assert_eq!(segment.document_len(999).unwrap(), None);
        assert_eq!(segment.df(10).unwrap(), 2);
        assert_eq!(segment.total_weight(10).unwrap(), 5);
        assert_eq!(segment.max_weight(10).unwrap(), 4);
        assert_eq!(collect_postings(&segment, 10), vec![(2, 1), (5, 4)]);
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
        assert!(collect_block(&segment, 999, 0).is_empty());
        assert!(segment.posting_block_postings(7, 2).is_err());
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

        let mut bytes = write_u64_u32_segment(&docs).unwrap();
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
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
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
    fn raw_segment_rejects_corrupt_posting_varint() {
        let terms = vec![(7, 1)];
        let docs = vec![RawDocument::new(3, &terms)];
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
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
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
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
        let mut bytes = write_u64_u32_segment(&docs).unwrap();
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
            for term_id in 0..12u64 {
                prop_assert_eq!(segment.df(term_id).unwrap(), idx.df(&term_id));
                let raw_postings = collect_postings(&segment, term_id);
                let memory_postings: Vec<(DocId, u32)> = idx.postings_iter(&term_id).collect();
                prop_assert_eq!(raw_postings, memory_postings);
            }
            for doc_id in idx.document_ids() {
                prop_assert_eq!(segment.document_len(doc_id).unwrap(), Some(idx.document_len(doc_id)));
            }
        }
    }
}
