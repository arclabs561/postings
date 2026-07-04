//! Byte-backed positional segment helpers.
//!
//! This module stores term -> doc -> token-position lists in an immutable byte
//! segment. It owns only the segment byte format; callers still own term
//! analysis, commit publication, deletes, compaction, and crash-safety policy.

use std::ops::Range;

use crate::codec::varint;
use crate::DocId;

use super::{PosingsIndex, PositionalTermPostings, TokenPos};

const MAGIC: &[u8; 8] = b"PSTP0001";
const VERSION: u32 = 1;
const FLAGS: u32 = 0;
const HEADER_LEN: usize = 72;
const TERM_ENTRY_LEN: usize = 28;

/// Errors returned by raw positional segment encoding and decoding.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The segment magic did not match the raw positional format.
    #[error("raw positional segment has invalid magic")]
    InvalidMagic,
    /// The segment version is not supported by this reader.
    #[error("unsupported raw positional segment version: {0}")]
    UnsupportedVersion(u32),
    /// The segment flags contain unsupported bits.
    #[error("unsupported raw positional segment flags: {0:#x}")]
    UnsupportedFlags(u32),
    /// A fixed-width field or section was truncated.
    #[error("truncated raw positional segment section: {0}")]
    Truncated(&'static str),
    /// A section checksum did not match the encoded bytes.
    #[error("raw positional segment checksum mismatch in {0}")]
    ChecksumMismatch(&'static str),
    /// The encoded layout violated the raw positional segment contract.
    #[error("invalid raw positional segment layout: {0}")]
    InvalidLayout(&'static str),
    /// A varint field could not be decoded.
    #[error("invalid varint in raw positional segment section {section} at index {index}")]
    InvalidVarint {
        /// Section being decoded.
        section: &'static str,
        /// Field index within the section.
        index: usize,
    },
    /// A delta-decoded document id overflowed.
    #[error("raw positional segment doc id overflow at index {0}")]
    DocIdOverflow(usize),
    /// A delta-decoded token position overflowed.
    #[error("raw positional segment token position overflow at index {0}")]
    PositionOverflow(usize),
    /// A term string was not valid UTF-8.
    #[error("raw positional segment term is not UTF-8")]
    InvalidUtf8,
}

/// One decoded raw positional posting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawPositionalPosting {
    /// Document id containing the term.
    pub doc_id: DocId,
    /// Sorted token positions for this term in the document.
    pub positions: Vec<TokenPos>,
}

#[derive(Clone, Debug)]
struct Header {
    doc_count: u32,
    term_count: u32,
    term_dir_len: usize,
    term_bytes_len: usize,
    doc_meta_len: usize,
    postings_len: usize,
    term_dir_crc: u32,
    term_bytes_crc: u32,
    doc_meta_crc: u32,
    postings_crc: u32,
}

#[derive(Clone, Debug)]
struct TermEntry<'a> {
    term: &'a str,
    postings: Range<usize>,
    doc_freq: u32,
}

/// A byte-backed immutable positional segment.
#[derive(Clone, Debug)]
pub struct RawPositionalSegment<'a> {
    bytes: &'a [u8],
    terms: Vec<TermEntry<'a>>,
    document_lengths: Vec<(DocId, u32)>,
}

impl<'a> RawPositionalSegment<'a> {
    /// Open a raw positional segment from bytes.
    ///
    /// The reader validates fixed sections and checksums up front. Posting
    /// payloads are decoded lazily when a term is requested.
    pub fn open(bytes: &'a [u8]) -> Result<Self, Error> {
        let header = read_header(bytes)?;
        let term_dir = checked_range(
            HEADER_LEN,
            header.term_dir_len,
            bytes.len(),
            "term directory",
        )?;
        let term_bytes = checked_range(
            term_dir.end,
            header.term_bytes_len,
            bytes.len(),
            "term bytes",
        )?;
        let doc_meta = checked_range(
            term_bytes.end,
            header.doc_meta_len,
            bytes.len(),
            "document metadata",
        )?;
        let postings = checked_range(doc_meta.end, header.postings_len, bytes.len(), "postings")?;
        if postings.end != bytes.len() {
            return Err(Error::InvalidLayout("trailing bytes"));
        }

        check_crc(
            "term directory",
            &bytes[term_dir.clone()],
            header.term_dir_crc,
        )?;
        check_crc(
            "term bytes",
            &bytes[term_bytes.clone()],
            header.term_bytes_crc,
        )?;
        check_crc(
            "document metadata",
            &bytes[doc_meta.clone()],
            header.doc_meta_crc,
        )?;
        check_crc("postings", &bytes[postings.clone()], header.postings_crc)?;

        let document_lengths =
            decode_document_lengths(&bytes[doc_meta], header.doc_count as usize)?;
        let terms = decode_term_directory(
            bytes,
            term_dir,
            term_bytes,
            postings.clone(),
            header.term_count as usize,
        )?;

        Ok(Self {
            bytes,
            terms,
            document_lengths,
        })
    }

    /// Return `(doc_id, token_count)` pairs sorted by document id.
    pub fn document_lengths(&self) -> &[(DocId, u32)] {
        &self.document_lengths
    }

    /// Return one document's token count, if present.
    pub fn document_len(&self, doc_id: DocId) -> Option<u32> {
        self.document_lengths
            .binary_search_by_key(&doc_id, |&(id, _)| id)
            .ok()
            .map(|index| self.document_lengths[index].1)
    }

    /// Return the number of documents containing a term.
    pub fn df(&self, term: &str) -> u32 {
        self.term_entry(term).map_or(0, |entry| entry.doc_freq)
    }

    /// Return sorted document ids containing a term.
    pub fn docs_with_term(&self, term: &str) -> Result<Vec<DocId>, Error> {
        Ok(self
            .term_postings(term)?
            .into_iter()
            .map(|posting| posting.doc_id)
            .collect())
    }

    /// Return decoded postings for one term.
    pub fn term_postings(&self, term: &str) -> Result<Vec<RawPositionalPosting>, Error> {
        let Some(entry) = self.term_entry(term) else {
            return Ok(Vec::new());
        };
        self.decode_postings(entry)
    }

    /// Return positions for one term in one document.
    pub fn positions(&self, term: &str, doc_id: DocId) -> Result<Vec<TokenPos>, Error> {
        for posting in self.term_postings(term)? {
            if posting.doc_id == doc_id {
                return Ok(posting.positions);
            }
            if posting.doc_id > doc_id {
                break;
            }
        }
        Ok(Vec::new())
    }

    fn term_entry(&self, term: &str) -> Option<&TermEntry<'a>> {
        self.terms
            .binary_search_by_key(&term, |entry| entry.term)
            .ok()
            .map(|index| &self.terms[index])
    }

    fn decode_postings(&self, entry: &TermEntry<'a>) -> Result<Vec<RawPositionalPosting>, Error> {
        let bytes = &self.bytes[entry.postings.clone()];
        let mut consumed = 0usize;
        let mut previous_doc = 0u32;
        let mut out = Vec::with_capacity(entry.doc_freq as usize);

        for posting_index in 0..entry.doc_freq as usize {
            let (doc_gap, doc_gap_len) = decode_varint(bytes, consumed, "postings", posting_index)?;
            consumed += doc_gap_len;
            let doc_id = if posting_index == 0 {
                doc_gap
            } else {
                previous_doc
                    .checked_add(doc_gap)
                    .ok_or(Error::DocIdOverflow(posting_index))?
            };
            if posting_index > 0 && doc_id <= previous_doc {
                return Err(Error::InvalidLayout("non-increasing posting doc id"));
            }
            previous_doc = doc_id;

            let Some(doc_len) = self.document_len(doc_id) else {
                return Err(Error::InvalidLayout("posting references unknown document"));
            };

            let (position_count, position_count_len) =
                decode_varint(bytes, consumed, "postings", posting_index)?;
            consumed += position_count_len;
            if position_count == 0 {
                return Err(Error::InvalidLayout("empty position list"));
            }

            let mut previous_position = 0u32;
            let mut positions = Vec::with_capacity(position_count as usize);
            for position_index in 0..position_count as usize {
                let field_index = posting_index
                    .checked_add(position_index)
                    .ok_or(Error::InvalidLayout("position index overflow"))?;
                let (position_gap, position_gap_len) =
                    decode_varint(bytes, consumed, "postings", field_index)?;
                consumed += position_gap_len;
                let position = if position_index == 0 {
                    position_gap
                } else {
                    previous_position
                        .checked_add(position_gap)
                        .ok_or(Error::PositionOverflow(field_index))?
                };
                if position_index > 0 && position <= previous_position {
                    return Err(Error::InvalidLayout("non-increasing token position"));
                }
                if position >= doc_len {
                    return Err(Error::InvalidLayout(
                        "token position exceeds document length",
                    ));
                }
                previous_position = position;
                positions.push(position);
            }

            out.push(RawPositionalPosting { doc_id, positions });
        }

        if consumed != bytes.len() {
            return Err(Error::InvalidLayout("trailing posting bytes"));
        }
        Ok(out)
    }
}

/// Encode a positional index into a raw positional segment.
///
/// This is a convenience for sealing bounded in-memory shards. It emits only
/// segment bytes; callers still own file paths, atomic publication, manifests,
/// deletes, retention, and compaction.
pub fn write_positional_segment_from_index(index: &PosingsIndex) -> Result<Vec<u8>, Error> {
    let document_lengths = index.sorted_document_lengths();
    let terms = index.sorted_term_posting_lists();
    write_positional_segment(&document_lengths, &terms)
}

/// Encode sorted positional document metadata and term posting lists.
///
/// Document lengths must be sorted by document id. Terms must be sorted
/// lexicographically, and each posting list must be sorted by document id with
/// strictly increasing token positions.
pub fn write_positional_segment(
    document_lengths: &[(DocId, u32)],
    terms: &[PositionalTermPostings<'_>],
) -> Result<Vec<u8>, Error> {
    validate_document_lengths(document_lengths)?;

    let mut term_bytes = Vec::new();
    let mut postings_bytes = Vec::new();
    let mut entries = Vec::with_capacity(terms.len());
    let mut previous_term: Option<&str> = None;

    for term in terms {
        if term.term.is_empty() {
            return Err(Error::InvalidLayout("empty term"));
        }
        if previous_term.is_some_and(|previous| term.term <= previous) {
            return Err(Error::InvalidLayout("terms must be strictly increasing"));
        }
        if term.postings.is_empty() {
            return Err(Error::InvalidLayout("empty term posting list"));
        }

        let term_offset = checked_u32(term_bytes.len(), "term byte offset")?;
        let term_len = checked_u32(term.term.len(), "term byte length")?;
        term_bytes.extend_from_slice(term.term.as_bytes());

        let postings_start = postings_bytes.len();
        let postings_offset = checked_u64(postings_start, "postings offset")?;
        append_postings(document_lengths, &term.postings, &mut postings_bytes)?;
        let postings_len = checked_u64(postings_bytes.len() - postings_start, "postings length")?;
        entries.push((
            term_offset,
            term_len,
            postings_offset,
            postings_len,
            term.postings.len(),
        ));
        previous_term = Some(term.term);
    }

    let mut term_dir = Vec::with_capacity(entries.len() * TERM_ENTRY_LEN);
    for (term_offset, term_len, postings_offset, postings_len, doc_freq) in entries {
        put_u32(&mut term_dir, term_offset);
        put_u32(&mut term_dir, term_len);
        put_u64(&mut term_dir, postings_offset);
        put_u64(&mut term_dir, postings_len);
        put_u32(&mut term_dir, checked_u32(doc_freq, "document frequency")?);
    }

    let mut doc_meta = Vec::new();
    append_document_lengths(document_lengths, &mut doc_meta)?;

    let header = Header {
        doc_count: checked_u32(document_lengths.len(), "document count")?,
        term_count: checked_u32(terms.len(), "term count")?,
        term_dir_len: term_dir.len(),
        term_bytes_len: term_bytes.len(),
        doc_meta_len: doc_meta.len(),
        postings_len: postings_bytes.len(),
        term_dir_crc: crc32fast::hash(&term_dir),
        term_bytes_crc: crc32fast::hash(&term_bytes),
        doc_meta_crc: crc32fast::hash(&doc_meta),
        postings_crc: crc32fast::hash(&postings_bytes),
    };

    let total_len = HEADER_LEN
        .checked_add(term_dir.len())
        .and_then(|len| len.checked_add(term_bytes.len()))
        .and_then(|len| len.checked_add(doc_meta.len()))
        .and_then(|len| len.checked_add(postings_bytes.len()))
        .ok_or(Error::InvalidLayout("segment length overflow"))?;
    let mut out = Vec::with_capacity(total_len);
    put_header(&mut out, &header)?;
    out.extend_from_slice(&term_dir);
    out.extend_from_slice(&term_bytes);
    out.extend_from_slice(&doc_meta);
    out.extend_from_slice(&postings_bytes);
    Ok(out)
}

fn validate_document_lengths(document_lengths: &[(DocId, u32)]) -> Result<(), Error> {
    let mut previous = None;
    for &(doc_id, _) in document_lengths {
        if previous.is_some_and(|previous| doc_id <= previous) {
            return Err(Error::InvalidLayout(
                "document lengths must be strictly increasing",
            ));
        }
        previous = Some(doc_id);
    }
    Ok(())
}

fn append_document_lengths(
    document_lengths: &[(DocId, u32)],
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    let mut previous = 0u32;
    for (index, &(doc_id, doc_len)) in document_lengths.iter().enumerate() {
        let gap = if index == 0 {
            doc_id
        } else {
            doc_id
                .checked_sub(previous)
                .ok_or(Error::InvalidLayout("document id gap underflow"))?
        };
        varint::encode_u32(gap, out);
        varint::encode_u32(doc_len, out);
        previous = doc_id;
    }
    Ok(())
}

fn append_postings(
    document_lengths: &[(DocId, u32)],
    postings: &[super::PositionalPosting<'_>],
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    let mut previous_doc = 0u32;
    for (posting_index, posting) in postings.iter().enumerate() {
        let Some(doc_len) = document_len(document_lengths, posting.doc_id) else {
            return Err(Error::InvalidLayout("posting references unknown document"));
        };
        let doc_gap = if posting_index == 0 {
            posting.doc_id
        } else {
            posting
                .doc_id
                .checked_sub(previous_doc)
                .ok_or(Error::InvalidLayout("posting doc id gap underflow"))?
        };
        if posting_index > 0 && posting.doc_id <= previous_doc {
            return Err(Error::InvalidLayout("posting doc ids must be increasing"));
        }
        if posting.positions.is_empty() {
            return Err(Error::InvalidLayout("empty position list"));
        }

        varint::encode_u32(doc_gap, out);
        varint::encode_u32(checked_u32(posting.positions.len(), "position count")?, out);

        let mut previous_position = 0u32;
        for (position_index, &position) in posting.positions.iter().enumerate() {
            let gap = if position_index == 0 {
                position
            } else {
                position
                    .checked_sub(previous_position)
                    .ok_or(Error::InvalidLayout("position gap underflow"))?
            };
            if position_index > 0 && position <= previous_position {
                return Err(Error::InvalidLayout("positions must be increasing"));
            }
            if position >= doc_len {
                return Err(Error::InvalidLayout("position exceeds document length"));
            }
            varint::encode_u32(gap, out);
            previous_position = position;
        }
        previous_doc = posting.doc_id;
    }
    Ok(())
}

fn document_len(document_lengths: &[(DocId, u32)], doc_id: DocId) -> Option<u32> {
    document_lengths
        .binary_search_by_key(&doc_id, |&(id, _)| id)
        .ok()
        .map(|index| document_lengths[index].1)
}

fn decode_document_lengths(bytes: &[u8], count: usize) -> Result<Vec<(DocId, u32)>, Error> {
    let mut consumed = 0usize;
    let mut previous_doc = 0u32;
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let (gap, gap_len) = decode_varint(bytes, consumed, "document metadata", index)?;
        consumed += gap_len;
        let (doc_len, doc_len_len) = decode_varint(bytes, consumed, "document metadata", index)?;
        consumed += doc_len_len;
        let doc_id = if index == 0 {
            gap
        } else {
            previous_doc
                .checked_add(gap)
                .ok_or(Error::DocIdOverflow(index))?
        };
        if index > 0 && doc_id <= previous_doc {
            return Err(Error::InvalidLayout("non-increasing document id"));
        }
        previous_doc = doc_id;
        out.push((doc_id, doc_len));
    }
    if consumed != bytes.len() {
        return Err(Error::InvalidLayout("trailing document metadata bytes"));
    }
    Ok(out)
}

fn decode_term_directory<'a>(
    bytes: &'a [u8],
    term_dir: Range<usize>,
    term_bytes: Range<usize>,
    postings: Range<usize>,
    count: usize,
) -> Result<Vec<TermEntry<'a>>, Error> {
    let dir = &bytes[term_dir];
    let expected_len = count
        .checked_mul(TERM_ENTRY_LEN)
        .ok_or(Error::InvalidLayout("term directory length overflow"))?;
    if dir.len() != expected_len {
        return Err(Error::InvalidLayout("term directory length mismatch"));
    }

    let mut entries = Vec::with_capacity(count);
    let mut previous_term = None;
    for index in 0..count {
        let offset = index * TERM_ENTRY_LEN;
        let term_offset = read_u32_at(dir, offset, "term directory")? as usize;
        let term_len = read_u32_at(dir, offset + 4, "term directory")? as usize;
        let postings_offset = read_usize_at(dir, offset + 8, "term directory")?;
        let postings_len = read_usize_at(dir, offset + 16, "term directory")?;
        let doc_freq = read_u32_at(dir, offset + 24, "term directory")?;

        let term_start = term_bytes
            .start
            .checked_add(term_offset)
            .ok_or(Error::InvalidLayout("term byte range overflow"))?;
        let term_range = checked_range(term_start, term_len, term_bytes.end, "term bytes")?;
        let term = std::str::from_utf8(&bytes[term_range]).map_err(|_| Error::InvalidUtf8)?;
        if term.is_empty() {
            return Err(Error::InvalidLayout("empty term"));
        }
        if previous_term.is_some_and(|previous| term <= previous) {
            return Err(Error::InvalidLayout("non-increasing term"));
        }
        let relative_postings =
            checked_range(postings_offset, postings_len, postings.len(), "postings")?;
        if doc_freq == 0 {
            return Err(Error::InvalidLayout("empty term posting list"));
        }
        let postings_start = postings
            .start
            .checked_add(relative_postings.start)
            .ok_or(Error::InvalidLayout("posting range overflow"))?;
        let postings_end = postings
            .start
            .checked_add(relative_postings.end)
            .ok_or(Error::InvalidLayout("posting range overflow"))?;
        entries.push(TermEntry {
            term,
            postings: postings_start..postings_end,
            doc_freq,
        });
        previous_term = Some(term);
    }
    Ok(entries)
}

fn put_header(out: &mut Vec<u8>, header: &Header) -> Result<(), Error> {
    out.extend_from_slice(MAGIC);
    put_u32(out, VERSION);
    put_u32(out, FLAGS);
    put_u32(out, header.doc_count);
    put_u32(out, header.term_count);
    put_u64(
        out,
        checked_u64(header.term_dir_len, "term directory length")?,
    );
    put_u64(
        out,
        checked_u64(header.term_bytes_len, "term bytes length")?,
    );
    put_u64(
        out,
        checked_u64(header.doc_meta_len, "document metadata length")?,
    );
    put_u64(out, checked_u64(header.postings_len, "postings length")?);
    put_u32(out, header.term_dir_crc);
    put_u32(out, header.term_bytes_crc);
    put_u32(out, header.doc_meta_crc);
    put_u32(out, header.postings_crc);
    debug_assert_eq!(out.len(), HEADER_LEN);
    Ok(())
}

fn read_header(bytes: &[u8]) -> Result<Header, Error> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::Truncated("header"));
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(Error::InvalidMagic);
    }
    let version = read_u32_at(bytes, 8, "header")?;
    if version != VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    let flags = read_u32_at(bytes, 12, "header")?;
    if flags != FLAGS {
        return Err(Error::UnsupportedFlags(flags));
    }
    Ok(Header {
        doc_count: read_u32_at(bytes, 16, "header")?,
        term_count: read_u32_at(bytes, 20, "header")?,
        term_dir_len: read_usize_at(bytes, 24, "header")?,
        term_bytes_len: read_usize_at(bytes, 32, "header")?,
        doc_meta_len: read_usize_at(bytes, 40, "header")?,
        postings_len: read_usize_at(bytes, 48, "header")?,
        term_dir_crc: read_u32_at(bytes, 56, "header")?,
        term_bytes_crc: read_u32_at(bytes, 60, "header")?,
        doc_meta_crc: read_u32_at(bytes, 64, "header")?,
        postings_crc: read_u32_at(bytes, 68, "header")?,
    })
}

fn check_crc(section: &'static str, bytes: &[u8], want: u32) -> Result<(), Error> {
    if crc32fast::hash(bytes) != want {
        return Err(Error::ChecksumMismatch(section));
    }
    Ok(())
}

fn decode_varint(
    bytes: &[u8],
    offset: usize,
    section: &'static str,
    index: usize,
) -> Result<(u32, usize), Error> {
    if offset > bytes.len() {
        return Err(Error::Truncated(section));
    }
    varint::decode_u32(&bytes[offset..]).ok_or(Error::InvalidVarint { section, index })
}

fn checked_range(
    offset: usize,
    len: usize,
    total_len: usize,
    section: &'static str,
) -> Result<Range<usize>, Error> {
    let end = offset
        .checked_add(len)
        .ok_or(Error::InvalidLayout("section range overflow"))?;
    if end > total_len {
        return Err(Error::Truncated(section));
    }
    Ok(offset..end)
}

fn checked_u32(value: usize, reason: &'static str) -> Result<u32, Error> {
    value.try_into().map_err(|_| Error::InvalidLayout(reason))
}

fn checked_u64(value: usize, reason: &'static str) -> Result<u64, Error> {
    value.try_into().map_err(|_| Error::InvalidLayout(reason))
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u32_at(bytes: &[u8], offset: usize, section: &'static str) -> Result<u32, Error> {
    let range = checked_range(offset, 4, bytes.len(), section)?;
    Ok(u32::from_le_bytes(
        bytes[range]
            .try_into()
            .expect("checked range is exactly four bytes"),
    ))
}

fn read_u64_at(bytes: &[u8], offset: usize, section: &'static str) -> Result<u64, Error> {
    let range = checked_range(offset, 8, bytes.len(), section)?;
    Ok(u64::from_le_bytes(
        bytes[range]
            .try_into()
            .expect("checked range is exactly eight bytes"),
    ))
}

fn read_usize_at(bytes: &[u8], offset: usize, section: &'static str) -> Result<usize, Error> {
    read_u64_at(bytes, offset, section)?
        .try_into()
        .map_err(|_| Error::InvalidLayout("section length is too large"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(terms: &[&str]) -> Vec<String> {
        terms.iter().map(|term| (*term).to_string()).collect()
    }

    fn sample_index() -> PosingsIndex {
        let mut index = PosingsIndex::new();
        index.add_document(7, &strings(&["b", "c"])).unwrap();
        index.add_document(2, &strings(&["a"])).unwrap();
        index.add_document(4, &strings(&["a", "b", "a"])).unwrap();
        index
    }

    #[test]
    fn raw_positional_segment_roundtrips_index_exports() {
        let index = sample_index();
        let bytes = write_positional_segment_from_index(&index).unwrap();
        let segment = RawPositionalSegment::open(&bytes).unwrap();

        assert_eq!(segment.document_lengths(), &[(2, 1), (4, 3), (7, 2)]);
        assert_eq!(segment.document_len(4), Some(3));
        assert_eq!(segment.document_len(99), None);
        assert_eq!(segment.df("a"), 2);
        assert_eq!(segment.df("missing"), 0);
        assert_eq!(segment.docs_with_term("b").unwrap(), vec![4, 7]);
        assert_eq!(segment.positions("a", 4).unwrap(), vec![0, 2]);
        assert!(segment.positions("a", 7).unwrap().is_empty());

        assert_eq!(
            segment.term_postings("a").unwrap(),
            vec![
                RawPositionalPosting {
                    doc_id: 2,
                    positions: vec![0],
                },
                RawPositionalPosting {
                    doc_id: 4,
                    positions: vec![0, 2],
                },
            ]
        );
    }

    #[test]
    fn raw_positional_segment_rejects_corrupt_postings_checksum() {
        let index = sample_index();
        let mut bytes = write_positional_segment_from_index(&index).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0xff;

        assert!(matches!(
            RawPositionalSegment::open(&bytes),
            Err(Error::ChecksumMismatch("postings"))
        ));
    }

    #[test]
    fn raw_positional_writer_rejects_unsorted_document_lengths() {
        let index = sample_index();
        let terms = index.sorted_term_posting_lists();
        let err = write_positional_segment(&[(4, 3), (2, 1)], &terms).unwrap_err();

        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn raw_positional_writer_rejects_position_past_document_length() {
        let postings = vec![super::super::PositionalPosting {
            doc_id: 1,
            positions: &[0, 2],
        }];
        let terms = vec![PositionalTermPostings {
            term: "a",
            postings,
        }];
        let err = write_positional_segment(&[(1, 2)], &terms).unwrap_err();

        assert!(matches!(err, Error::InvalidLayout(_)));
    }
}
