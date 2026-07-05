//! Positional postings (term -> doc -> positions) for phrase/proximity search.
//!
//! This module is behind `feature = "positional"` so the base `postings` crate can stay
//! lightweight for doc-only use-cases.
//!
//! Design notes:
//! - in-memory, index-only structures (no document content)
//! - caller-provided token stream (no tokenization policy)
//! - positions are **token positions** (not byte offsets)

use std::collections::{HashMap, HashSet};

use crate::{CandidatePlan, DocId, PlannerConfig};

/// Token position within a document (0-based).
pub type TokenPos = u32;

/// Errors for positional postings operations.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Document id already exists (caller must delete/update explicitly).
    #[error("document already exists: {0}")]
    DuplicateDocId(DocId),
}

/// Preferred name for the positional postings index.
///
/// `PosingsIndex` is kept for continuity with the historical crate name (`posings`).
pub type PositionalIndex = PosingsIndex;

/// Preferred name for the positional postings error type.
pub type PositionalError = Error;

/// Byte-backed positional segment helpers.
#[cfg(feature = "raw-segment")]
pub mod raw;

/// One document's positions for a positional term posting list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PositionalPosting<'a> {
    /// Document id containing the term.
    pub doc_id: DocId,
    /// Sorted token positions where the term occurs in the document.
    pub positions: &'a [TokenPos],
}

/// A sorted positional posting list for one term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionalTermPostings<'a> {
    /// Term text for this posting list.
    pub term: &'a str,
    /// Posting entries sorted by document id.
    pub postings: Vec<PositionalPosting<'a>>,
}

/// Feature-gated helpers for representing candidate doc-id sets with Elias–Fano (via `sbits`).
///
/// This is intentionally a helper surface rather than a hard dependency of the main index.
#[cfg(feature = "sbits")]
pub mod ef_candidates {
    use crate::{codec, DocId};

    /// Re-export the underlying succinct structure.
    pub type EliasFano = sbits::EliasFano;

    /// Build an Elias–Fano structure from sorted doc ids.
    ///
    /// # Contract
    ///
    /// - `ids` must be **sorted** and **strictly increasing**
    /// - every id must satisfy `id < universe_size`
    pub fn elias_fano_from_sorted_doc_ids(ids: &[DocId], universe_size: u32) -> EliasFano {
        try_elias_fano_from_sorted_doc_ids(ids, universe_size)
            .expect("doc ids must be strictly increasing and inside the universe")
    }

    /// Build an Elias-Fano structure from sorted doc ids, validating the public contract.
    pub fn try_elias_fano_from_sorted_doc_ids(
        ids: &[DocId],
        universe_size: u32,
    ) -> Result<EliasFano, codec::Error> {
        codec::validate_sorted_ids(ids, universe_size)?;
        let ids64: Vec<u64> = ids.iter().map(|&x| x as u64).collect();
        Ok(EliasFano::new(&ids64, universe_size as u64))
    }
}

/// Feature-gated helpers for compressing candidate doc-id sets with `cnk`.
///
/// `cnk::DeltaVarintCompressor` provides delta+varint ID set compression.
/// We keep it behind a feature so higher layers can experiment with compressed
/// candidate sets without imposing a dependency on all users.
#[cfg(feature = "cnk-compression")]
pub mod cnk_candidates {
    use crate::DocId;
    use cnk::{DeltaVarintCompressor, IdSetCompressor};

    pub use cnk::CompressionError;

    /// Compress sorted doc ids into a compact byte vector.
    ///
    /// # Contract
    ///
    /// - `ids` must be **sorted** and **strictly increasing**
    /// - every id must satisfy `id < universe_size`
    pub fn compress_sorted_doc_ids(
        ids: &[DocId],
        universe_size: u32,
    ) -> Result<Vec<u8>, CompressionError> {
        DeltaVarintCompressor::new().compress_set(ids, universe_size)
    }

    /// Decompress a compressed doc-id set.
    ///
    /// Returns doc ids in sorted order.
    pub fn decompress_doc_ids(
        compressed: &[u8],
        universe_size: u32,
    ) -> Result<Vec<DocId>, CompressionError> {
        DeltaVarintCompressor::new().decompress_set(compressed, universe_size)
    }
}

/// A minimal positional postings index.
///
/// Stores per-term postings lists of (doc_id -> positions).
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PosingsIndex {
    // term -> (doc_id -> positions)
    postings: HashMap<String, HashMap<DocId, Vec<TokenPos>>>,
    // doc_id -> length in tokens
    doc_len: HashMap<DocId, u32>,
    // doc_id -> unique terms in that doc (for fast deletes without scanning |V|)
    doc_terms: HashMap<DocId, Vec<String>>,
}

impl PosingsIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document as a stream of terms (tokens).
    ///
    /// Terms are recorded with their token positions (0..len-1).
    pub fn add_document(&mut self, doc_id: DocId, terms: &[String]) -> Result<(), Error> {
        if self.doc_len.contains_key(&doc_id) {
            return Err(Error::DuplicateDocId(doc_id));
        }
        self.doc_len.insert(doc_id, terms.len() as u32);

        let mut seen: HashSet<&str> = HashSet::new();
        let mut uniq_terms: Vec<String> = Vec::new();
        for (i, t) in terms.iter().enumerate() {
            let pos = i as u32;
            if seen.insert(t.as_str()) {
                uniq_terms.push(t.clone());
            }

            // Avoid cloning `t` on every token occurrence: clone only on first insertion
            // of a new vocabulary term.
            match self.postings.get_mut(t.as_str()) {
                Some(docs) => {
                    docs.entry(doc_id).or_default().push(pos);
                }
                None => {
                    let mut docs: HashMap<DocId, Vec<TokenPos>> = HashMap::new();
                    docs.insert(doc_id, vec![pos]);
                    self.postings.insert(t.clone(), docs);
                }
            }
        }

        uniq_terms.sort_unstable();
        self.doc_terms.insert(doc_id, uniq_terms);
        Ok(())
    }

    /// Delete a document (if present).
    ///
    /// Returns true if the document existed.
    pub fn delete_document(&mut self, doc_id: DocId) -> bool {
        if self.doc_len.remove(&doc_id).is_none() {
            return false;
        }

        // Fast path: only touch vocabulary entries that appeared in the deleted doc.
        // (O(|unique_terms_in_doc|) rather than scanning O(|V|).)
        let terms = self.doc_terms.remove(&doc_id).unwrap_or_default();
        for term in terms {
            let empty = match self.postings.get_mut(term.as_str()) {
                Some(docs) => {
                    docs.remove(&doc_id);
                    docs.is_empty()
                }
                None => false,
            };
            if empty {
                self.postings.remove(&term);
            }
        }
        true
    }

    /// Update/replace a document by modeling updates as delete+add.
    pub fn upsert_document(&mut self, doc_id: DocId, terms: &[String]) {
        let _ = self.delete_document(doc_id);
        // Safe by construction.
        let _ = self.add_document(doc_id, terms);
    }

    /// Number of indexed documents.
    pub fn num_docs(&self) -> u32 {
        self.doc_len.len() as u32
    }

    /// Iterate all document ids.
    pub fn document_ids(&self) -> impl Iterator<Item = DocId> + '_ {
        self.doc_len.keys().copied()
    }

    /// Document length in tokens (0 if missing).
    pub fn document_len(&self, doc_id: DocId) -> u32 {
        self.doc_len.get(&doc_id).copied().unwrap_or(0)
    }

    /// Return `(doc_id, token_count)` pairs sorted by document id.
    ///
    /// This is a sealing helper for bounded in-memory positional shards. Callers
    /// that build byte-native or externally sorted segment files can use this as
    /// the stable document metadata stream while keeping publication, deletes,
    /// compaction, and crash-safety policy outside the in-memory index.
    pub fn sorted_document_lengths(&self) -> Vec<(DocId, u32)> {
        let mut lengths: Vec<_> = self
            .doc_len
            .iter()
            .map(|(&doc_id, &len)| (doc_id, len))
            .collect();
        lengths.sort_unstable_by_key(|&(doc_id, _)| doc_id);
        lengths
    }

    /// Positions for a term in a given doc (empty if absent).
    pub fn positions(&self, term: &str, doc_id: DocId) -> &[TokenPos] {
        positions_in_docs(self.postings.get(term), doc_id)
    }

    /// Candidate docs containing a term.
    pub fn docs_with_term(&self, term: &str) -> impl Iterator<Item = DocId> + '_ {
        self.postings
            .get(term)
            .into_iter()
            .flat_map(|m| m.keys().copied())
    }

    /// Document frequency for a term (number of docs containing it).
    pub fn df(&self, term: &str) -> u32 {
        self.postings.get(term).map(|m| m.len() as u32).unwrap_or(0)
    }

    /// Return all positional posting lists sorted by term, then by document id.
    ///
    /// This is intended for sealing bounded in-memory shards into an immutable
    /// segment format. It borrows position slices from the index, so callers can
    /// encode without cloning the per-document position vectors.
    pub fn sorted_term_posting_lists(&self) -> Vec<PositionalTermPostings<'_>> {
        let mut terms: Vec<_> = self.postings.keys().map(String::as_str).collect();
        terms.sort_unstable();

        terms
            .into_iter()
            .map(|term| {
                let mut postings: Vec<_> = self.postings[term]
                    .iter()
                    .map(|(&doc_id, positions)| PositionalPosting {
                        doc_id,
                        positions: positions.as_slice(),
                    })
                    .collect();
                postings.sort_unstable_by_key(|posting| posting.doc_id);
                PositionalTermPostings { term, postings }
            })
            .collect()
    }

    /// Candidate docs containing all required terms (intersection).
    ///
    /// Supports per-term multiplicity requirements: `required_counts[t]` means the term must occur
    /// at least that many times in the doc.
    fn candidates_all_terms(&self, required_counts: &HashMap<&str, usize>) -> Vec<DocId> {
        if required_counts.is_empty() {
            return Vec::new();
        }

        // Pick the rarest term as the anchor.
        let mut anchor: Option<&str> = None;
        let mut anchor_df: usize = usize::MAX;
        for &t in required_counts.keys() {
            let df = self.postings.get(t).map(|m| m.len()).unwrap_or(0);
            if df < anchor_df {
                anchor = Some(t);
                anchor_df = df;
            }
        }
        let Some(anchor) = anchor else {
            return Vec::new();
        };
        let Some(anchor_map) = self.postings.get(anchor) else {
            return Vec::new();
        };
        let req_anchor = *required_counts.get(anchor).unwrap_or(&1);
        let mut required_rest = Vec::with_capacity(required_counts.len().saturating_sub(1));
        for (&term, &count) in required_counts {
            if term != anchor {
                required_rest.push((term, count));
            }
        }

        let mut out: Vec<DocId> = Vec::new();
        'doc: for (&doc_id, pos_anchor) in anchor_map.iter() {
            if pos_anchor.len() < req_anchor {
                continue;
            }
            for &(t, req) in &required_rest {
                let Some(m) = self.postings.get(t) else {
                    continue 'doc;
                };
                let Some(pos) = m.get(&doc_id) else {
                    continue 'doc;
                };
                if pos.len() < req {
                    continue 'doc;
                }
            }
            out.push(doc_id);
        }
        out.sort_unstable();
        out
    }

    /// Plan candidate generation for proximity queries using DF upper bounds.
    ///
    /// Uses \(\sum_t df(t)\) over unique terms (upper bound; can exceed N).
    pub fn plan_candidates_near<'a>(
        &'a self,
        terms: impl IntoIterator<Item = &'a str>,
        cfg: PlannerConfig,
    ) -> CandidatePlan {
        let mut uniq: HashSet<&str> = HashSet::new();
        let mut df_sum: u64 = 0;
        let n = self.num_docs();
        if n == 0 {
            return CandidatePlan::Candidates(Vec::new());
        }
        for t in terms {
            if !uniq.insert(t) {
                continue;
            }
            df_sum = df_sum.saturating_add(self.df(t) as u64);
            if df_sum >= cfg.max_candidates as u64 {
                return CandidatePlan::ScanAll;
            }
        }
        let ratio = (df_sum as f32) / (n as f32);
        if ratio > cfg.max_candidate_ratio {
            return CandidatePlan::ScanAll;
        }

        // For proximity queries, a doc must contain all terms. Use intersection to keep candidate
        // sets tight while preserving the "no false negatives" guarantee.
        let required: HashMap<&str, usize> = uniq.into_iter().map(|t| (t, 1usize)).collect();
        CandidatePlan::Candidates(self.candidates_all_terms(&required))
    }

    /// Exact phrase match: returns docs that contain `terms` as an adjacent sequence.
    ///
    /// Requires positional postings. Complexity depends on term frequencies.
    pub fn phrase_match(&self, phrase: &[String]) -> Vec<DocId> {
        let phrase: Vec<&str> = phrase.iter().map(String::as_str).collect();
        self.phrase_match_refs(&phrase)
    }

    /// Exact phrase match over borrowed term strings.
    ///
    /// This avoids requiring callers to allocate `String` values for query terms
    /// they already hold as borrowed text.
    pub fn phrase_match_strs(&self, phrase: &[&str]) -> Vec<DocId> {
        self.phrase_match_refs(phrase)
    }

    fn phrase_match_refs(&self, phrase: &[&str]) -> Vec<DocId> {
        if phrase.is_empty() {
            return Vec::new();
        }
        if phrase.len() == 1 {
            let t0 = phrase[0];
            let mut docs: Vec<DocId> = self.docs_with_term(t0).collect();
            docs.sort_unstable();
            return docs;
        }
        if let [a, b, c] = phrase {
            let terms = [*a, *b, *c];
            if terms[0] != terms[1] && terms[0] != terms[2] && terms[1] != terms[2] {
                return self.phrase_match_three_unique(terms);
            }
        }

        // Prefilter docs to those that contain all required terms (including multiplicity).
        let mut required: HashMap<&str, usize> = HashMap::new();
        for &t in phrase {
            *required.entry(t).or_insert(0) += 1;
        }
        let candidates = self.candidates_all_terms(&required);

        // For each possible start implied by the rarest position list, verify
        // the exact shifted target in the other sorted position lists.
        let mut out = Vec::new();
        'doc: for doc_id in candidates {
            let mut anchor_i = 0usize;
            let mut anchor_positions: &[TokenPos] = &[];
            for (i, &term) in phrase.iter().enumerate() {
                let ps = self.positions(term, doc_id);
                if ps.is_empty() {
                    continue 'doc;
                }
                if anchor_positions.is_empty() || ps.len() < anchor_positions.len() {
                    anchor_i = i;
                    anchor_positions = ps;
                }
            }

            'start: for &anchor_pos in anchor_positions {
                let Some(start) = anchor_pos.checked_sub(anchor_i as u32) else {
                    continue;
                };
                for (i, &term) in phrase.iter().enumerate() {
                    if i == anchor_i {
                        continue;
                    }
                    let Some(target) = start.checked_add(i as u32) else {
                        continue 'start;
                    };
                    if self.positions(term, doc_id).binary_search(&target).is_err() {
                        continue 'start;
                    }
                }
                out.push(doc_id);
                continue 'doc;
            }
        }
        out
    }

    fn phrase_match_three_unique(&self, terms: [&str; 3]) -> Vec<DocId> {
        let maps = [
            self.postings.get(terms[0]),
            self.postings.get(terms[1]),
            self.postings.get(terms[2]),
        ];
        let mut anchor_i = 0usize;
        let mut anchor_df = maps[0].map_or(0, |m| m.len());
        for (i, map) in maps.iter().enumerate().skip(1) {
            let df = map.map_or(0, |m| m.len());
            if df < anchor_df {
                anchor_i = i;
                anchor_df = df;
            }
        }

        let Some(anchor_map) = maps[anchor_i] else {
            return Vec::new();
        };

        let mut out = Vec::new();
        'doc: for (&doc_id, anchor_positions) in anchor_map {
            let positions = match anchor_i {
                0 => [
                    anchor_positions.as_slice(),
                    positions_in_docs(maps[1], doc_id),
                    positions_in_docs(maps[2], doc_id),
                ],
                1 => [
                    positions_in_docs(maps[0], doc_id),
                    anchor_positions.as_slice(),
                    positions_in_docs(maps[2], doc_id),
                ],
                _ => [
                    positions_in_docs(maps[0], doc_id),
                    positions_in_docs(maps[1], doc_id),
                    anchor_positions.as_slice(),
                ],
            };
            if positions.iter().any(|ps| ps.is_empty()) {
                continue;
            }

            'start: for &anchor_pos in positions[anchor_i] {
                let Some(start) = anchor_pos.checked_sub(anchor_i as u32) else {
                    continue;
                };
                for (i, ps) in positions.iter().enumerate() {
                    if i == anchor_i {
                        continue;
                    }
                    let Some(target) = start.checked_add(i as u32) else {
                        continue 'start;
                    };
                    if ps.binary_search(&target).is_err() {
                        continue 'start;
                    }
                }
                out.push(doc_id);
                continue 'doc;
            }
        }
        out.sort_unstable();
        out
    }

    /// Proximity match: returns docs where `a` and `b` occur within `window` tokens.
    ///
    /// Semantics: unordered NEAR/k, i.e. exists positions `pa` in `a` and `pb` in `b`
    /// such that `|pa - pb| <= window`.
    pub fn near_match(&self, a: &str, b: &str, window: u32) -> Vec<DocId> {
        if window == 0 {
            // window=0 means the same position, which can't happen for two distinct terms.
            return Vec::new();
        }
        if a == b {
            let Some(docs) = self.postings.get(a) else {
                return Vec::new();
            };
            let mut out = Vec::new();
            for (&doc_id, positions) in docs {
                if positions
                    .windows(2)
                    .any(|pair| pair[1].saturating_sub(pair[0]) <= window)
                {
                    out.push(doc_id);
                }
            }
            out.sort_unstable();
            return out;
        }
        let (anchor, other) = if self.df(a) <= self.df(b) {
            (a, b)
        } else {
            (b, a)
        };
        let Some(anchor_map) = self.postings.get(anchor) else {
            return Vec::new();
        };
        let other_map = self.postings.get(other);
        let mut out = Vec::new();
        for (&doc_id, pa) in anchor_map {
            let pb = if anchor == other {
                pa.as_slice()
            } else {
                positions_in_docs(other_map, doc_id)
            };
            if pb.is_empty() {
                continue;
            }
            // Two-pointer scan on sorted position lists.
            let mut i = 0usize;
            let mut j = 0usize;
            let mut hit = false;
            while i < pa.len() && j < pb.len() {
                let x = pa[i];
                let y = pb[j];
                let diff = x.abs_diff(y);
                if diff <= window {
                    hit = true;
                    break;
                }
                if x < y {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            if hit {
                out.push(doc_id);
            }
        }
        out.sort_unstable();
        out
    }

    /// Multi-term proximity: return docs where all `terms` occur within `window` tokens.
    ///
    /// - `ordered=false`: unordered window (`max(pos)-min(pos) <= window`) covering all term occurrences.
    /// - `ordered=true`: the terms must appear in the given order, within window.
    ///
    /// Supports duplicate terms by requiring multiple occurrences.
    pub fn near_match_terms(&self, terms: &[String], window: u32, ordered: bool) -> Vec<DocId> {
        let terms: Vec<&str> = terms.iter().map(String::as_str).collect();
        self.near_match_term_refs(&terms, window, ordered)
    }

    /// Multi-term proximity over borrowed term strings.
    ///
    /// This avoids requiring callers to allocate `String` values for query terms
    /// they already hold as borrowed text.
    pub fn near_match_terms_strs(&self, terms: &[&str], window: u32, ordered: bool) -> Vec<DocId> {
        self.near_match_term_refs(terms, window, ordered)
    }

    fn near_match_term_refs(&self, terms: &[&str], window: u32, ordered: bool) -> Vec<DocId> {
        if terms.len() < 2 || window == 0 {
            return Vec::new();
        }
        if let [a, b, c] = terms {
            let terms = [*a, *b, *c];
            if terms[0] != terms[1] && terms[0] != terms[2] && terms[1] != terms[2] {
                return if ordered {
                    self.near_match_three_unique::<true>(terms, window)
                } else {
                    self.near_match_three_unique::<false>(terms, window)
                };
            }
        }

        // Multiplicity requirements for duplicates.
        let mut required: HashMap<&str, usize> = HashMap::new();
        for &t in terms {
            *required.entry(t).or_insert(0) += 1;
        }

        let candidates = self.candidates_all_terms(&required);
        let required_terms: Vec<(&str, usize)> = required
            .iter()
            .map(|(&term, &count)| (term, count))
            .collect();
        let mut out = Vec::new();
        for doc_id in candidates {
            let hit = if ordered {
                near_doc_ordered(self, doc_id, terms, window)
            } else {
                near_doc_unordered(self, doc_id, &required_terms, window)
            };
            if hit {
                out.push(doc_id);
            }
        }
        out.sort_unstable();
        out
    }

    fn near_match_three_unique<const ORDERED: bool>(
        &self,
        terms: [&str; 3],
        window: u32,
    ) -> Vec<DocId> {
        let maps = [
            self.postings.get(terms[0]),
            self.postings.get(terms[1]),
            self.postings.get(terms[2]),
        ];
        let mut anchor_i = 0usize;
        let mut anchor_df = maps[0].map_or(0, |m| m.len());
        for (i, map) in maps.iter().enumerate().skip(1) {
            let df = map.map_or(0, |m| m.len());
            if df < anchor_df {
                anchor_i = i;
                anchor_df = df;
            }
        }

        let Some(anchor_map) = maps[anchor_i] else {
            return Vec::new();
        };

        let mut out = Vec::new();
        for (&doc_id, anchor_positions) in anchor_map {
            let positions = match anchor_i {
                0 => [
                    anchor_positions.as_slice(),
                    positions_in_docs(maps[1], doc_id),
                    positions_in_docs(maps[2], doc_id),
                ],
                1 => [
                    positions_in_docs(maps[0], doc_id),
                    anchor_positions.as_slice(),
                    positions_in_docs(maps[2], doc_id),
                ],
                _ => [
                    positions_in_docs(maps[0], doc_id),
                    positions_in_docs(maps[1], doc_id),
                    anchor_positions.as_slice(),
                ],
            };
            if positions.iter().any(|ps| ps.is_empty()) {
                continue;
            }
            let hit = if ORDERED {
                near_positions_ordered_three(positions, window)
            } else {
                near_positions_unordered_three(positions, window)
            };
            if hit {
                out.push(doc_id);
            }
        }
        out.sort_unstable();
        out
    }
}

fn positions_in_docs(docs: Option<&HashMap<DocId, Vec<TokenPos>>>, doc_id: DocId) -> &[TokenPos] {
    static EMPTY: [TokenPos; 0] = [];
    docs.and_then(|m| m.get(&doc_id))
        .map(|v| v.as_slice())
        .unwrap_or(&EMPTY)
}

fn near_positions_unordered_three(positions: [&[TokenPos]; 3], window: u32) -> bool {
    let [a, b, c] = positions;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut k = 0usize;
    while i < a.len() && j < b.len() && k < c.len() {
        let pa = a[i];
        let pb = b[j];
        let pc = c[k];
        let min_pos = pa.min(pb).min(pc);
        let max_pos = pa.max(pb).max(pc);
        if max_pos - min_pos <= window {
            return true;
        }
        if pa == min_pos {
            i += 1;
        } else if pb == min_pos {
            j += 1;
        } else {
            k += 1;
        }
    }
    false
}

fn near_positions_ordered_three(positions: [&[TokenPos]; 3], window: u32) -> bool {
    let [a, b, c] = positions;
    for &pa in a {
        let b_i = b.partition_point(|&p| p <= pa);
        let Some(&pb) = b.get(b_i) else {
            return false;
        };
        let c_i = c.partition_point(|&p| p <= pb);
        let Some(&pc) = c.get(c_i) else {
            return false;
        };
        if pc.saturating_sub(pa) <= window {
            return true;
        }
    }
    false
}

fn near_doc_unordered(
    ix: &PosingsIndex,
    doc_id: DocId,
    required: &[(&str, usize)],
    window: u32,
) -> bool {
    if let [(a, 1), (b, 1), (c, 1)] = required {
        return near_positions_unordered_three(
            [
                ix.positions(a, doc_id),
                ix.positions(b, doc_id),
                ix.positions(c, doc_id),
            ],
            window,
        );
    }

    // Build occurrences (pos, term) for all required term strings.
    let mut occ: Vec<(TokenPos, usize)> = Vec::new();
    for (term_i, &(term, _)) in required.iter().enumerate() {
        for &p in ix.positions(term, doc_id) {
            occ.push((p, term_i));
        }
    }
    occ.sort_unstable_by_key(|(p, _)| *p);
    if occ.is_empty() {
        return false;
    }

    // Sliding window over occurrences; maintain counts.
    let mut have = vec![0usize; required.len()];
    let mut satisfied = 0usize;
    let need = required.len();

    let mut l = 0usize;
    for r in 0..occ.len() {
        let (pos_r, term_r) = occ[r];
        have[term_r] += 1;
        if have[term_r] == required[term_r].1 {
            satisfied += 1;
        }

        while satisfied == need {
            let (pos_l, term_l) = occ[l];
            if pos_r.saturating_sub(pos_l) <= window {
                return true;
            }
            // Shrink from left.
            if have[term_l] == required[term_l].1 {
                satisfied -= 1;
            }
            have[term_l] -= 1;
            l += 1;
        }
    }
    false
}

fn near_doc_ordered(ix: &PosingsIndex, doc_id: DocId, terms: &[&str], window: u32) -> bool {
    let first = terms[0];
    let p0 = ix.positions(first, doc_id);
    if p0.is_empty() {
        return false;
    }
    'start: for &start in p0 {
        let mut prev = start;
        for &t in terms.iter().skip(1) {
            let ps = ix.positions(t, doc_id);
            if ps.is_empty() {
                continue 'start;
            }
            // Find first occurrence strictly after `prev` to ensure distinct tokens.
            let target = prev.saturating_add(1);
            let i = ps.partition_point(|&p| p < target);
            let Some(&pn) = ps.get(i) else {
                continue 'start;
            };
            prev = pn;
            if prev.saturating_sub(start) > window {
                continue 'start;
            }
        }
        if prev.saturating_sub(start) <= window {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "sbits")]
    use proptest::prelude::*;

    #[test]
    fn phrase_match_finds_adjacent_sequence() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["new".into(), "york".into(), "city".into()])
            .unwrap();
        ix.add_document(2, &["new".into(), "jersey".into(), "york".into()])
            .unwrap();

        let hits = ix.phrase_match(&["new".into(), "york".into()]);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn phrase_match_accepts_borrowed_terms() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["new".into(), "york".into(), "city".into()])
            .unwrap();
        ix.add_document(2, &["new".into(), "jersey".into(), "york".into()])
            .unwrap();

        let hits = ix.phrase_match_strs(&["new", "york"]);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn phrase_match_handles_duplicate_terms() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["a".into(), "a".into(), "b".into()])
            .unwrap();
        ix.add_document(2, &["a".into(), "x".into(), "a".into()])
            .unwrap();

        let hits = ix.phrase_match(&["a".into(), "a".into()]);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn phrase_match_single_term_is_sorted() {
        let mut ix = PosingsIndex::new();
        ix.add_document(2, &["a".into()]).unwrap();
        ix.add_document(1, &["a".into()]).unwrap();

        let hits = ix.phrase_match(&["a".into()]);
        assert_eq!(hits, vec![1, 2]);
    }

    #[test]
    fn near_match_finds_within_window_unordered() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["new".into(), "york".into(), "city".into()])
            .unwrap();
        ix.add_document(2, &["new".into(), "jersey".into(), "york".into()])
            .unwrap();

        // doc 2: new at 0, york at 2 => within window=2
        let hits = ix.near_match("new", "york", 2);
        assert_eq!(hits, vec![1, 2]);

        // window=1 excludes doc 2 (distance 2), includes doc 1 (distance 1)
        let hits = ix.near_match("new", "york", 1);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn near_match_is_symmetric_for_skewed_terms() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["common".into(), "x".into(), "rare".into()])
            .unwrap();
        ix.add_document(2, &["common".into(), "x".into(), "x".into()])
            .unwrap();
        ix.add_document(3, &["common".into(), "rare".into(), "x".into()])
            .unwrap();

        assert_eq!(ix.near_match("common", "rare", 2), vec![1, 3]);
        assert_eq!(ix.near_match("rare", "common", 2), vec![1, 3]);
    }

    #[test]
    fn near_match_same_term_requires_distinct_positions() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["a".into()]).unwrap();
        ix.add_document(2, &["a".into(), "x".into(), "a".into()])
            .unwrap();
        ix.add_document(3, &["a".into(), "x".into(), "x".into(), "a".into()])
            .unwrap();

        assert_eq!(ix.near_match("a", "a", 2), vec![2]);
        assert_eq!(ix.near_match("a", "a", 3), vec![2, 3]);
    }

    #[test]
    fn near_match_terms_unordered_multiterm() {
        let mut ix = PosingsIndex::new();
        ix.add_document(
            1,
            &["a".into(), "x".into(), "b".into(), "y".into(), "c".into()],
        )
        .unwrap();
        ix.add_document(
            2,
            &["a".into(), "x".into(), "b".into(), "y".into(), "z".into()],
        )
        .unwrap();

        let hits = ix.near_match_terms(&["a".into(), "b".into(), "c".into()], 4, false);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn near_match_terms_accepts_borrowed_terms() {
        let mut ix = PosingsIndex::new();
        ix.add_document(
            1,
            &["a".into(), "x".into(), "b".into(), "y".into(), "c".into()],
        )
        .unwrap();
        ix.add_document(
            2,
            &["a".into(), "x".into(), "b".into(), "y".into(), "z".into()],
        )
        .unwrap();

        let hits = ix.near_match_terms_strs(&["a", "b", "c"], 4, false);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn near_match_terms_ordered_unique_three_terms() {
        let mut ix = PosingsIndex::new();
        ix.add_document(
            1,
            &["a".into(), "x".into(), "b".into(), "y".into(), "c".into()],
        )
        .unwrap();
        ix.add_document(
            2,
            &["a".into(), "x".into(), "c".into(), "y".into(), "b".into()],
        )
        .unwrap();
        ix.add_document(
            3,
            &[
                "a".into(),
                "x".into(),
                "b".into(),
                "y".into(),
                "y".into(),
                "y".into(),
                "c".into(),
            ],
        )
        .unwrap();

        let hits = ix.near_match_terms(&["a".into(), "b".into(), "c".into()], 4, true);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn near_match_terms_ordered_and_duplicates() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["a".into(), "x".into(), "a".into(), "b".into()])
            .unwrap();

        // Requires two occurrences of "a" then "b" within window.
        let hits = ix.near_match_terms(&["a".into(), "a".into(), "b".into()], 10, true);
        assert_eq!(hits, vec![1]);

        // Wrong order: "b" then two "a" does not match.
        let hits = ix.near_match_terms(&["b".into(), "a".into(), "a".into()], 10, true);
        assert!(hits.is_empty());
    }

    #[test]
    fn posings_planner_can_bail_out() {
        let mut ix = PosingsIndex::new();
        for i in 0..100u32 {
            ix.add_document(i, &["common".into(), format!("u{i}")])
                .unwrap();
        }
        let plan = ix.plan_candidates_near(
            ["common", "u1"],
            PlannerConfig {
                max_candidate_ratio: 0.2,
                max_candidates: 10,
            },
        );
        assert_eq!(plan, CandidatePlan::ScanAll);
    }

    #[test]
    fn positions_are_sorted_token_positions() {
        let mut ix = PosingsIndex::new();
        ix.add_document(
            1,
            &[
                "a".into(),
                "x".into(),
                "a".into(),
                "y".into(),
                "a".into(),
                "z".into(),
            ],
        )
        .unwrap();
        assert_eq!(ix.positions("a", 1), &[0, 2, 4]);
    }

    #[test]
    fn sorted_document_lengths_are_doc_ordered() {
        let mut ix = PosingsIndex::new();
        ix.add_document(7, &["b".into(), "c".into()]).unwrap();
        ix.add_document(2, &["a".into()]).unwrap();
        ix.add_document(4, &["a".into(), "b".into(), "a".into()])
            .unwrap();

        assert_eq!(ix.sorted_document_lengths(), vec![(2, 1), (4, 3), (7, 2)]);
    }

    #[test]
    fn sorted_term_posting_lists_are_term_then_doc_ordered() {
        let mut ix = PosingsIndex::new();
        ix.add_document(7, &["b".into(), "c".into()]).unwrap();
        ix.add_document(2, &["a".into()]).unwrap();
        ix.add_document(4, &["a".into(), "b".into(), "a".into()])
            .unwrap();

        let lists = ix.sorted_term_posting_lists();
        assert_eq!(
            lists.iter().map(|list| list.term).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            lists[0].postings,
            vec![
                PositionalPosting {
                    doc_id: 2,
                    positions: &[0],
                },
                PositionalPosting {
                    doc_id: 4,
                    positions: &[0, 2],
                },
            ]
        );
        assert_eq!(
            lists[1].postings,
            vec![
                PositionalPosting {
                    doc_id: 4,
                    positions: &[1],
                },
                PositionalPosting {
                    doc_id: 7,
                    positions: &[0],
                },
            ]
        );
        assert_eq!(
            lists[2].postings,
            vec![PositionalPosting {
                doc_id: 7,
                positions: &[1],
            }]
        );
    }

    #[test]
    fn delete_removes_positions_and_docs() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["a".into(), "b".into()]).unwrap();
        ix.add_document(2, &["a".into()]).unwrap();

        assert_eq!(ix.df("a"), 2);
        assert!(!ix.positions("a", 1).is_empty());

        assert!(ix.delete_document(1));
        assert_eq!(ix.df("a"), 1);
        assert!(ix.positions("a", 1).is_empty());
        assert_eq!(ix.num_docs(), 1);
    }

    #[test]
    fn upsert_replaces_document() {
        let mut ix = PosingsIndex::new();
        ix.add_document(1, &["a".into(), "b".into()]).unwrap();
        assert_eq!(ix.phrase_match(&["a".into(), "b".into()]), vec![1]);

        ix.upsert_document(1, &["a".into(), "x".into()]);
        assert!(ix.phrase_match(&["a".into(), "b".into()]).is_empty());
        assert_eq!(ix.near_match("a", "x", 1), vec![1]);
    }

    #[cfg(feature = "sbits")]
    #[test]
    fn ef_candidates_roundtrip_get() {
        let ids: Vec<DocId> = vec![1, 5, 10, 20, 50];
        let ef = ef_candidates::elias_fano_from_sorted_doc_ids(&ids, 1_000);
        assert_eq!(ef.len(), ids.len());
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(ef.get(i).unwrap(), id as u64);
        }
    }

    #[cfg(feature = "sbits")]
    #[test]
    fn ef_candidates_checked_constructor_rejects_bad_ids() {
        let err = ef_candidates::try_elias_fano_from_sorted_doc_ids(&[2, 2], 10).unwrap_err();
        assert_eq!(
            err,
            crate::codec::Error::NotStrictlyIncreasing {
                index: 1,
                prev: 2,
                next: 2
            }
        );

        let err = ef_candidates::try_elias_fano_from_sorted_doc_ids(&[2, 10], 10).unwrap_err();
        assert_eq!(
            err,
            crate::codec::Error::IdOutOfUniverse {
                index: 1,
                id: 10,
                universe_size: 10
            }
        );
    }

    #[cfg(feature = "sbits")]
    proptest! {
        #[test]
        fn ef_candidates_property_get_matches_ids(mut ids in prop::collection::vec(0u32..1_000_000u32, 0..200)) {
            ids.sort_unstable();
            ids.dedup();
            let ef = ef_candidates::elias_fano_from_sorted_doc_ids(&ids, 1_000_000);
            prop_assert_eq!(ef.len(), ids.len());
            for (i, &id) in ids.iter().enumerate() {
                prop_assert_eq!(ef.get(i).unwrap(), id as u64);
            }
        }
    }

    #[cfg(feature = "cnk-compression")]
    #[test]
    fn cnk_candidates_roundtrip() {
        let ids: Vec<DocId> = vec![1, 5, 10, 20, 50, 100];
        let universe_size = 1_000;
        let compressed = cnk_candidates::compress_sorted_doc_ids(&ids, universe_size).unwrap();
        let back = cnk_candidates::decompress_doc_ids(&compressed, universe_size).unwrap();
        assert_eq!(back, ids);
    }
}
