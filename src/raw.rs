//! Byte-backed raw segment reader experiments.
//!
//! This module is intentionally narrow. It supports one local, numeric-term,
//! `u32`-weighted raw segment format so query code can read posting lists
//! without reconstructing a full [`crate::PostingsIndex`].

use crate::codec::varint;
use crate::DocId;
use std::collections::BTreeMap;

const MAGIC: &[u8; 8] = b"PSTRW001";
const FOOTER_MAGIC: &[u8; 8] = b"PSTRF001";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 72;
const TERM_ENTRY_LEN: usize = 40;
const DOC_ENTRY_LEN: usize = 8;
const FOOTER_LEN: usize = 12;

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
        for entry in entries.into_iter().skip(1) {
            let docs = self.posting_doc_ids(entry)?;
            candidates = intersect_doc_id_lists(&candidates, &docs);
            if candidates.is_empty() {
                break;
            }
        }
        Ok(candidates)
    }

    fn term_entry(&self, term_id: RawTermId) -> Result<Option<TermEntry>, Error> {
        let mut low = 0u32;
        let mut high = self.meta.term_count;
        while low < high {
            let mid = low + ((high - low) / 2);
            let entry = self.term_entry_at(mid)?;
            match entry.term_id.cmp(&term_id) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some(entry)),
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
        for posting in RawPostings::from_entry(self.bytes, entry)? {
            let (doc_id, _) = posting?;
            docs.push(doc_id);
        }
        Ok(docs)
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
    let postings_offset = doc_meta_offset
        .checked_add(doc_meta_len)
        .ok_or(Error::SegmentTooLarge)?;

    let mut postings_bytes = Vec::new();
    let mut term_entries = Vec::with_capacity(postings.len());
    for (&term_id, list) in &postings {
        let offset = postings_offset
            .checked_add(u64::try_from(postings_bytes.len()).map_err(|_| Error::SegmentTooLarge)?)
            .ok_or(Error::SegmentTooLarge)?;
        let start_len = postings_bytes.len();
        let mut prev_doc_id = 0;
        let mut max_weight = 0u32;
        let mut total_weight = 0u64;
        for (index, &(doc_id, weight)) in list.iter().enumerate() {
            let gap = if index == 0 {
                doc_id
            } else {
                doc_id - prev_doc_id
            };
            varint::encode_u32(gap, &mut postings_bytes);
            varint::encode_u32(weight, &mut postings_bytes);
            prev_doc_id = doc_id;
            max_weight = max_weight.max(weight);
            total_weight = total_weight.saturating_add(weight as u64);
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
            block_size: 0,
            total_doc_len,
            term_dir_offset,
            doc_meta_offset,
            postings_offset,
            footer_offset,
        },
    );
    for entry in term_entries {
        put_u64(&mut out, entry.term_id);
        put_u32(&mut out, entry.df);
        put_u32(&mut out, entry.max_weight);
        put_u64(&mut out, entry.total_weight);
        put_u64(&mut out, entry.postings_offset);
        put_u32(&mut out, entry.postings_len);
        put_u32(&mut out, 0);
    }
    for (&doc_id, (doc_len, _)) in &docs {
        put_u32(&mut out, doc_id);
        put_u32(&mut out, *doc_len);
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
    let doc_meta_end =
        meta.doc_meta_offset
            .checked_add(doc_meta_len)
            .ok_or(Error::InvalidLayout {
                reason: "doc metadata end overflows",
            })?;
    if doc_meta_end != meta.postings_offset {
        return Err(Error::InvalidLayout {
            reason: "postings must follow doc metadata",
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
        meta.postings_offset,
        meta.footer_offset - meta.postings_offset,
        bytes.len(),
        "postings",
    )?;
    Ok(())
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

fn intersect_doc_id_lists(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    out
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
        assert!(collect_postings(&segment, 999).is_empty());
        assert_eq!(segment.candidates_all_terms(&[10, 20]).unwrap(), vec![5]);
        assert_eq!(segment.candidates_all_terms(&[20]).unwrap(), vec![5, 9]);
        assert_eq!(segment.candidates_all_terms(&[10, 10]).unwrap(), vec![2, 5]);
        assert!(segment.candidates_all_terms(&[999]).unwrap().is_empty());
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
