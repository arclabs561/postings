//! `posings`: positional postings for phrase/proximity search.
//!
//! This is the “positional sibling” of a doc-only inverted index:
//! - doc-only postings: term -> docs (with tf)
//! - positional postings: term -> docs -> positions
//!
//! What this unlocks:
//! - phrase queries: `"new york"` (ordered, adjacent)
//! - proximity queries: `new NEAR/5 york` (unordered within window)
//! - proximity-aware scoring features
//!
//! Scope:
//! - in-memory, index-only structures
//! - caller-provided token stream (no tokenization policy)
//! - Unicode-safe positions are **token positions** (not byte offsets)

#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};

/// Document id type (kept consistent with other lexical crates here).
pub type DocId = u32;

/// Token position within a document (0-based).
pub type TokenPos = u32;

/// Errors for positional postings operations.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Document id already exists (caller must delete/update explicitly).
    #[error("duplicate doc id: {0}")]
    DuplicateDocId(DocId),
}

/// Planner output for positional candidate generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidatePlan {
    /// Use the returned candidates as a search set.
    Candidates(Vec<DocId>),
    /// Bail out: query is too broad, caller should scan all documents (or choose a different strategy).
    ScanAll,
}

/// Feature-gated helpers for representing candidate doc-id sets with Elias–Fano.
///
/// This is intentionally a *helper surface* rather than a core dependency:
/// - the main `posings` index is in-memory and `Vec<DocId>` based
/// - callers can opt into succinct representations for intersection-heavy pipelines
#[cfg(feature = "ef")]
pub mod ef_candidates {
    use crate::DocId;

    /// Re-export the underlying succinct structure.
    pub type EliasFano = postings_codec::ef::EliasFano;

    /// Build an Elias–Fano structure from sorted doc ids.
    ///
    /// # Contract
    ///
    /// - `ids` must be **sorted** and **strictly increasing**
    /// - every id must satisfy `id < universe_size`
    pub fn elias_fano_from_sorted_doc_ids(ids: &[DocId], universe_size: u32) -> EliasFano {
        postings_codec::ef::elias_fano_from_sorted_ids(ids, universe_size)
    }
}

/// Feature-gated helpers for compressing candidate doc-id sets with `cnk`.
///
/// Today `cnk::RocCompressor` is a delta+varint baseline (despite the name).
/// We keep it behind a feature so higher layers can experiment with compressed
/// candidate sets without imposing a dependency on all users.
#[cfg(feature = "roc")]
pub mod roc_candidates {
    use crate::DocId;
    use cnk::{IdSetCompressor, RocCompressor};

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
        RocCompressor::new().compress_set(ids, universe_size)
    }

    /// Decompress a compressed doc-id set.
    ///
    /// Returns doc ids in sorted order.
    pub fn decompress_doc_ids(
        compressed: &[u8],
        universe_size: u32,
    ) -> Result<Vec<DocId>, CompressionError> {
        RocCompressor::new().decompress_set(compressed, universe_size)
    }
}

/// Configuration for candidate planning / bailout.
#[derive(Debug, Clone, Copy)]
pub struct PlannerConfig {
    /// If an upper bound on candidates exceeds this ratio of the corpus, bail out.
    pub max_candidate_ratio: f32,
    /// If an upper bound on candidates exceeds this absolute count, bail out.
    pub max_candidates: u32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_candidate_ratio: 0.6,
            max_candidates: 200_000,
        }
    }
}

/// A minimal positional postings index.
///
/// Stores per-term postings lists of (doc_id -> positions).
#[derive(Debug, Default)]
pub struct PosingsIndex {
    // term -> (doc_id -> positions)
    postings: HashMap<String, HashMap<DocId, Vec<TokenPos>>>,
    // doc_id -> length in tokens
    doc_len: HashMap<DocId, u32>,
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
        for (i, t) in terms.iter().enumerate() {
            let pos = i as u32;
            self.postings
                .entry(t.clone())
                .or_default()
                .entry(doc_id)
                .or_default()
                .push(pos);
        }
        Ok(())
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

    /// Positions for a term in a given doc (empty if absent).
    pub fn positions(&self, term: &str, doc_id: DocId) -> &[TokenPos] {
        static EMPTY: [TokenPos; 0] = [];
        self.postings
            .get(term)
            .and_then(|m| m.get(&doc_id))
            .map(|v| v.as_slice())
            .unwrap_or(&EMPTY)
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
        for (&t, _req) in required_counts {
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

        let mut out: Vec<DocId> = Vec::new();
        'doc: for (&doc_id, pos_anchor) in anchor_map.iter() {
            let req_anchor = *required_counts.get(anchor).unwrap_or(&1);
            if pos_anchor.len() < req_anchor {
                continue;
            }
            for (&t, &req) in required_counts {
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
        let mut out: HashSet<DocId> = HashSet::new();
        for t in uniq {
            out.extend(self.docs_with_term(t));
        }
        let mut v: Vec<DocId> = out.into_iter().collect();
        v.sort_unstable();
        CandidatePlan::Candidates(v)
    }

    /// Exact phrase match: returns docs that contain `terms` as an adjacent sequence.
    ///
    /// Requires positional postings. Complexity depends on term frequencies.
    pub fn phrase_match(&self, phrase: &[String]) -> Vec<DocId> {
        if phrase.is_empty() {
            return Vec::new();
        }
        if phrase.len() == 1 {
            let t0 = &phrase[0];
            return self.docs_with_term(t0).collect();
        }

        let t0 = &phrase[0];
        let mut out = Vec::new();
        for doc_id in self.docs_with_term(t0) {
            let mut starts: Vec<TokenPos> = self.positions(t0, doc_id).to_vec();
            if starts.is_empty() {
                continue;
            }
            // For each subsequent term, filter candidate starts by checking pos+offset exists.
            for (offset, term) in phrase.iter().enumerate().skip(1) {
                let want_pos = offset as u32;
                let pos_set = self.positions(term, doc_id);
                if pos_set.is_empty() {
                    starts.clear();
                    break;
                }
                starts.retain(|&p0| pos_set.binary_search(&(p0 + want_pos)).is_ok());
                if starts.is_empty() {
                    break;
                }
            }
            if !starts.is_empty() {
                out.push(doc_id);
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
        let mut out = Vec::new();
        for doc_id in self.docs_with_term(a) {
            let pa = self.positions(a, doc_id);
            let pb = self.positions(b, doc_id);
            if pa.is_empty() || pb.is_empty() {
                continue;
            }
            // Two-pointer scan on sorted position lists.
            let mut i = 0usize;
            let mut j = 0usize;
            let mut hit = false;
            while i < pa.len() && j < pb.len() {
                let x = pa[i];
                let y = pb[j];
                let diff = if x >= y { x - y } else { y - x };
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
        if terms.len() < 2 || window == 0 {
            return Vec::new();
        }

        // Multiplicity requirements for duplicates.
        let mut required: HashMap<&str, usize> = HashMap::new();
        for t in terms {
            *required.entry(t.as_str()).or_insert(0) += 1;
        }

        let candidates = self.candidates_all_terms(&required);
        let mut out = Vec::new();
        for doc_id in candidates {
            let hit = if ordered {
                near_doc_ordered(self, doc_id, terms, window)
            } else {
                near_doc_unordered(self, doc_id, &required, window)
            };
            if hit {
                out.push(doc_id);
            }
        }
        out.sort_unstable();
        out
    }
}

fn near_doc_unordered(
    ix: &PosingsIndex,
    doc_id: DocId,
    required: &HashMap<&str, usize>,
    window: u32,
) -> bool {
    // Build occurrences (pos, term) for all required term strings.
    let mut occ: Vec<(TokenPos, &str)> = Vec::new();
    for (&t, _req) in required {
        for &p in ix.positions(t, doc_id) {
            occ.push((p, t));
        }
    }
    occ.sort_unstable_by_key(|(p, _)| *p);
    if occ.is_empty() {
        return false;
    }

    // Sliding window over occurrences; maintain counts.
    let mut have: HashMap<&str, usize> = HashMap::new();
    let mut satisfied = 0usize;
    let need = required.len();

    let mut l = 0usize;
    for r in 0..occ.len() {
        let (pos_r, t_r) = occ[r];
        let c = have.entry(t_r).or_insert(0);
        *c += 1;
        if *c == *required.get(t_r).unwrap_or(&1) {
            satisfied += 1;
        }

        while satisfied == need {
            let (pos_l, t_l) = occ[l];
            if pos_r.saturating_sub(pos_l) <= window {
                return true;
            }
            // Shrink from left.
            let c = have.get_mut(t_l).unwrap();
            if *c == *required.get(t_l).unwrap_or(&1) {
                satisfied -= 1;
            }
            *c -= 1;
            l += 1;
        }
    }
    false
}

fn near_doc_ordered(ix: &PosingsIndex, doc_id: DocId, terms: &[String], window: u32) -> bool {
    let first = terms[0].as_str();
    let p0 = ix.positions(first, doc_id);
    if p0.is_empty() {
        return false;
    }
    'start: for &start in p0 {
        let mut prev = start;
        for t in terms.iter().skip(1) {
            let ps = ix.positions(t, doc_id);
            if ps.is_empty() {
                continue 'start;
            }
            // Find first occurrence strictly after `prev` to ensure distinct tokens.
            let mut found: Option<TokenPos> = None;
            if let Ok(i) = ps.binary_search(&(prev + 1)) {
                found = Some(ps[i]);
            } else if let Err(i) = ps.binary_search(&(prev + 1)) {
                if i < ps.len() {
                    found = Some(ps[i]);
                }
            }
            let Some(pn) = found else {
                continue 'start;
            };
            prev = pn;
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
    #[cfg(feature = "ef")]
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
            ["common", "u1"].into_iter(),
            PlannerConfig {
                max_candidate_ratio: 0.2,
                max_candidates: 10,
            },
        );
        assert_eq!(plan, CandidatePlan::ScanAll);
    }

    #[cfg(feature = "ef")]
    #[test]
    fn ef_candidates_roundtrip_get() {
        let ids: Vec<DocId> = vec![1, 5, 10, 20, 50];
        let ef = ef_candidates::elias_fano_from_sorted_doc_ids(&ids, 1_000);
        assert_eq!(ef.len(), ids.len());
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(ef.get(i).unwrap(), id);
        }
    }

    #[cfg(feature = "ef")]
    proptest! {
        #[test]
        fn ef_candidates_property_get_matches_ids(mut ids in prop::collection::vec(0u32..1_000_000u32, 0..200)) {
            ids.sort_unstable();
            ids.dedup();
            let ef = ef_candidates::elias_fano_from_sorted_doc_ids(&ids, 1_000_000);
            prop_assert_eq!(ef.len(), ids.len());
            for (i, &id) in ids.iter().enumerate() {
                prop_assert_eq!(ef.get(i).unwrap(), id);
            }
        }
    }

    #[cfg(feature = "roc")]
    #[test]
    fn roc_candidates_roundtrip() {
        let ids: Vec<DocId> = vec![1, 5, 10, 20, 50, 100];
        let universe_size = 1_000;
        let compressed = roc_candidates::compress_sorted_doc_ids(&ids, universe_size).unwrap();
        let back = roc_candidates::decompress_doc_ids(&compressed, universe_size).unwrap();
        assert_eq!(back, ids);
    }
}
