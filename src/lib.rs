//! # postings
//!
//! A small inverted-index core built around postings lists.
//!
//! ## Scope (deliberate)
//!
//! - This crate is **index-only**: it does not store document content.
//! - It supports **candidate generation** with **no false negatives** (a caller may
//!   choose to verify candidates with an exact matcher).
//! - It uses a Lucene-style mental model: immutable "segments" (here: append-only
//!   batches) and logical deletes.
//!
//! ## Non-goals (for now)
//!
//! - Positional postings (phrase queries)
//! - On-disk persistence / compaction
//! - Rich query language beyond "union of term postings"
//!
//! Related crates:
//! - `posings`: positional postings for phrase/proximity evaluation (token positions).
//!   - Repo: <https://github.com/arclabs561/posings>
//! - `postings::codec`: low-level codecs (varint/gap) for postings payloads (in this repo).

pub mod codec;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Document identifier.
pub type DocId = u32;

/// Errors returned by `postings`.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The caller attempted to add an already-present document id.
    #[error("document already exists: {0}")]
    DuplicateDocId(DocId),
}

/// Planner output for candidate generation.
///
/// This is the simplest encoding of the key invariant:
/// indexing is allowed to bail out (and fall back to scanning) when a query is too broad.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidatePlan {
    /// Use the returned candidates as a search set.
    Candidates(Vec<DocId>),
    /// Bail out: the query is too broad, so the caller should scan all documents.
    ScanAll,
}

/// Configuration for candidate planning / bailout.
#[derive(Debug, Clone, Copy)]
pub struct PlannerConfig {
    /// If an upper bound on candidates exceeds this ratio of the corpus, bail out.
    ///
    /// Note: we estimate using \(\sum_t df(t)\), which is an *upper bound* (can exceed N).
    pub max_candidate_ratio: f32,
    /// If an upper bound on candidates exceeds this absolute count, bail out.
    pub max_candidates: u32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            // Conservative defaults: useful for "index as filter" use-cases.
            max_candidate_ratio: 0.6,
            max_candidates: 200_000,
        }
    }
}

/// Immutable postings for a batch of documents.
///
/// A "segment" here is an append-only batch that is never mutated after creation.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "Term: serde::Serialize",
        deserialize = "Term: serde::Deserialize<'de> + Eq + std::hash::Hash"
    ))
)]
struct Segment<Term> {
    /// term -> sorted postings list of (doc_id, tf)
    postings: HashMap<Term, Vec<(DocId, u32)>>,
    /// doc_id -> doc length in terms
    doc_len: HashMap<DocId, u32>,
    /// doc_id -> unique terms in that doc (for df adjustments on delete)
    doc_terms: HashMap<DocId, Vec<Term>>,
}

impl<Term> Default for Segment<Term> {
    fn default() -> Self {
        Self {
            postings: HashMap::new(),
            doc_len: HashMap::new(),
            doc_terms: HashMap::new(),
        }
    }
}

/// A postings-based inverted index with segment-style updates.
///
/// This is an in-memory MVP:
/// - each `add_document` creates a new (small) segment
/// - deletes are logical (we update global stats; segment remains immutable)
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "Term: serde::Serialize",
        deserialize = "Term: serde::Deserialize<'de> + Eq + std::hash::Hash"
    ))
)]
pub struct PostingsIndex<Term = String> {
    segments: Vec<Segment<Term>>,
    /// live doc -> segment index
    doc_segment: HashMap<DocId, usize>,
    /// live doc -> length
    doc_len: HashMap<DocId, u32>,
    /// term -> df (number of live documents containing term)
    df: HashMap<Term, u32>,
    total_doc_len: u64,
}

impl<Term> Default for PostingsIndex<Term> {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
            doc_segment: HashMap::new(),
            doc_len: HashMap::new(),
            df: HashMap::new(),
            total_doc_len: 0,
        }
    }
}

impl<Term> PostingsIndex<Term>
where
    Term: Clone + Eq + Hash + Ord,
{
    /// Create an empty postings index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of live documents currently indexed.
    pub fn num_docs(&self) -> u32 {
        self.doc_len.len() as u32
    }

    /// Average document length (in terms) over live documents.
    pub fn avg_doc_len(&self) -> f32 {
        let n = self.num_docs() as f32;
        if n == 0.0 {
            return 0.0;
        }
        (self.total_doc_len as f32) / n
    }

    /// Iterate live document ids.
    pub fn document_ids(&self) -> impl Iterator<Item = DocId> + '_ {
        self.doc_len.keys().copied()
    }

    /// Document length (in terms). Returns 0 for unknown doc ids.
    pub fn document_len(&self, doc_id: DocId) -> u32 {
        self.doc_len.get(&doc_id).copied().unwrap_or(0)
    }

    /// Document frequency for a term (count of live documents containing term).
    pub fn df<Q>(&self, term: &Q) -> u32
    where
        Term: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.df.get(term).copied().unwrap_or(0)
    }

    /// Iterate all distinct terms present in live documents.
    pub fn terms(&self) -> impl Iterator<Item = &Term> + '_ {
        self.df.keys()
    }

    /// Add a document by doc id and term stream.
    ///
    /// If `doc_id` already exists, return an error. Call `delete_document` first
    /// to model updates as delete+add (segment-style).
    pub fn add_document(&mut self, doc_id: DocId, terms: &[Term]) -> Result<(), Error> {
        if self.doc_segment.contains_key(&doc_id) {
            return Err(Error::DuplicateDocId(doc_id));
        }

        let doc_length = terms.len() as u32;
        let mut term_freqs: HashMap<Term, u32> = HashMap::new();
        for t in terms {
            *term_freqs.entry(t.clone()).or_insert(0) += 1;
        }

        let mut doc_terms: Vec<Term> = term_freqs.keys().cloned().collect();
        doc_terms.sort_unstable();

        // Build an immutable segment for this doc.
        let mut seg = Segment::<Term>::default();
        seg.doc_len.insert(doc_id, doc_length);
        seg.doc_terms.insert(doc_id, doc_terms.clone());
        for (term, tf) in term_freqs {
            seg.postings.entry(term).or_default().push((doc_id, tf));
        }
        // Ensure postings lists are sorted (future-proof for multi-doc segments).
        for postings in seg.postings.values_mut() {
            postings.sort_unstable_by_key(|(id, _)| *id);
        }

        let seg_idx = self.segments.len();
        self.segments.push(seg);
        self.doc_segment.insert(doc_id, seg_idx);
        self.doc_len.insert(doc_id, doc_length);
        self.total_doc_len += doc_length as u64;

        // Update global df.
        for term in doc_terms {
            *self.df.entry(term).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Logically delete a document (if present).
    ///
    /// Returns true if the doc existed.
    pub fn delete_document(&mut self, doc_id: DocId) -> bool {
        let seg_idx = match self.doc_segment.remove(&doc_id) {
            Some(i) => i,
            None => return false,
        };
        let doc_len = self.doc_len.remove(&doc_id).unwrap_or(0);
        self.total_doc_len = self.total_doc_len.saturating_sub(doc_len as u64);

        let seg = &self.segments[seg_idx];
        if let Some(terms) = seg.doc_terms.get(&doc_id) {
            for term in terms {
                if let Some(df) = self.df.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.df.remove(term);
                    }
                }
            }
        }
        true
    }

    /// Term frequency of `term` in `doc_id`.
    pub fn term_frequency<Q>(&self, doc_id: DocId, term: &Q) -> u32
    where
        Term: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let seg_idx = match self.doc_segment.get(&doc_id) {
            Some(i) => *i,
            None => return 0,
        };
        let seg = &self.segments[seg_idx];
        let postings = match seg.postings.get(term) {
            Some(p) => p,
            None => return 0,
        };
        match postings.binary_search_by_key(&doc_id, |(id, _)| *id) {
            Ok(i) => postings[i].1,
            Err(_) => 0,
        }
    }

    /// Candidate documents that contain at least one query term.
    pub fn candidates<Q>(&self, query_terms: &[Q]) -> Vec<DocId>
    where
        Term: Borrow<Q>,
        Q: Hash + Eq,
    {
        if query_terms.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<DocId> = Vec::new();
        let mut seen: HashSet<DocId> = HashSet::new();
        for term in query_terms {
            for (doc_id, _) in self.postings_iter(term) {
                if seen.insert(doc_id) {
                    out.push(doc_id);
                }
            }
        }
        // Deterministic output: stable ascending doc ids.
        out.sort_unstable();
        out
    }

    /// Candidate documents that contain **all** query terms (conjunctive / AND).
    ///
    /// This is a common first step before:
    /// - exact phrase/proximity verification (positional index), or
    /// - scoring (BM25/TF-IDF) in a higher layer.
    ///
    /// Notes:
    /// - Duplicate terms in `query_terms` are treated as a single requirement.
    /// - Results are returned in sorted order.
    pub fn candidates_all_terms<Q>(&self, query_terms: &[Q]) -> Vec<DocId>
    where
        Term: Borrow<Q>,
        Q: Hash + Eq,
    {
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Deduplicate query terms.
        let mut uniq: Vec<&Q> = Vec::new();
        let mut seen: HashSet<&Q> = HashSet::new();
        for t in query_terms {
            if seen.insert(t) {
                uniq.push(t);
            }
        }
        if uniq.is_empty() {
            return Vec::new();
        }

        // DAAT-style intersection over sorted doc-id lists.
        // Anchor on the rarest term to minimize intermediate sets.
        uniq.sort_by_key(|t| self.df(*t));
        if self.df(uniq[0]) == 0 {
            return Vec::new();
        }

        let mut acc: Vec<DocId> = self.postings_iter(uniq[0]).map(|(id, _)| id).collect();
        acc.sort_unstable();

        for &t in uniq.iter().skip(1) {
            if self.df(t) == 0 {
                return Vec::new();
            }
            let mut docs: Vec<DocId> = self.postings_iter(t).map(|(id, _)| id).collect();
            docs.sort_unstable();
            acc = intersect_sorted(&acc, &docs);
            if acc.is_empty() {
                break;
            }
        }
        acc
    }

    /// Plan candidate generation, with a bailout option for broad queries.
    ///
    /// Returns:
    /// - `CandidatePlan::Candidates` when the query is selective enough.
    /// - `CandidatePlan::ScanAll` when the query is too broad (caller should scan all docs).
    pub fn plan_candidates<Q>(&self, query_terms: &[Q], cfg: PlannerConfig) -> CandidatePlan
    where
        Term: Borrow<Q>,
        Q: Hash + Eq,
    {
        if query_terms.is_empty() {
            return CandidatePlan::Candidates(Vec::new());
        }

        let n = self.num_docs();
        if n == 0 {
            return CandidatePlan::Candidates(Vec::new());
        }

        // Upper bound on candidate count: sum df(t) over unique terms.
        let mut seen_terms: HashSet<&Q> = HashSet::new();
        let mut df_sum: u64 = 0;
        for t in query_terms {
            if !seen_terms.insert(t) {
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

        CandidatePlan::Candidates(self.candidates(query_terms))
    }

    /// Iterate postings for a term across all segments (live docs only).
    pub fn postings_iter<'a, Q>(&'a self, term: &'a Q) -> impl Iterator<Item = (DocId, u32)> + 'a
    where
        Term: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.segments.iter().flat_map(move |seg| {
            seg.postings
                .get(term)
                .into_iter()
                .flat_map(|v| v.iter().copied())
                .filter(|(doc_id, _)| self.doc_segment.contains_key(doc_id))
        })
    }

    /// Save the index to a directory using `durability`.
    #[cfg(feature = "persistence")]
    pub fn save<D: durability::Directory + ?Sized>(
        &self,
        dir: &D,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        Term: serde::Serialize,
    {
        let bytes = postcard::to_allocvec(self)?;
        dir.atomic_write(path, &bytes)?;
        Ok(())
    }

    /// Save the index with stable-storage durability barriers.
    ///
    /// This is strictly stronger than `save()` on filesystem-backed directories:
    /// it fsyncs the temp file and syncs the parent directory after the atomic rename.
    ///
    /// For non-filesystem backends (e.g. `MemoryDirectory`) this returns `NotSupported`.
    #[cfg(feature = "persistence")]
    pub fn save_durable<D: durability::DurableDirectory + ?Sized>(
        &self,
        dir: &D,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        Term: serde::Serialize,
    {
        let bytes = postcard::to_allocvec(self)?;
        dir.atomic_write_durable(path, &bytes)?;
        Ok(())
    }

    /// Load the index from a directory using `durability`.
    #[cfg(feature = "persistence")]
    pub fn load<D: durability::Directory + ?Sized>(
        dir: &D,
        path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        for<'de> Term: serde::Deserialize<'de>,
    {
        use std::io::Read;

        let mut f = dir.open_file(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        let idx: Self = postcard::from_bytes(&bytes)?;
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn add_and_lookup_basic() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(
            0,
            &[
                String::from("the"),
                String::from("quick"),
                String::from("quick"),
            ],
        )
        .unwrap();
        assert_eq!(idx.num_docs(), 1);
        assert_eq!(idx.document_len(0), 3);
        assert_eq!(idx.df("quick"), 1);
        assert_eq!(idx.term_frequency(0, "quick"), 2);
        assert_eq!(idx.term_frequency(0, "missing"), 0);
    }

    #[test]
    fn delete_updates_df() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(0, &[String::from("a"), String::from("b")])
            .unwrap();
        idx.add_document(1, &[String::from("b"), String::from("c")])
            .unwrap();
        assert_eq!(idx.df("b"), 2);
        assert!(idx.delete_document(0));
        assert_eq!(idx.df("b"), 1);
        assert_eq!(idx.df("a"), 0);
        assert_eq!(idx.term_frequency(0, "b"), 0);
        assert_eq!(idx.term_frequency(1, "b"), 1);
    }

    #[test]
    fn multilingual_terms_do_not_panic() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(
            0,
            &[
                String::from("Müller"),                           // Latin + diacritics
                String::from("東京"),                             // CJK
                String::from("مرحبا"),                            // Arabic (RTL)
                String::from("Москва"),                           // Cyrillic
                String::from("cafe\u{0301}"),                     // NFD combining mark
                String::from("👨\u{200D}👩\u{200D}👧\u{200D}👦"), // emoji ZWJ sequence
            ],
        )
        .unwrap();
        assert_eq!(idx.num_docs(), 1);
        assert_eq!(idx.df("東京"), 1);
        assert_eq!(idx.term_frequency(0, "مرحبا"), 1);
    }

    proptest::proptest! {
        #[test]
        fn df_is_never_negative_and_upper_bounded(
            docs in proptest::collection::vec(
                proptest::collection::vec("[a-z]{1,6}", 0..20),
                0..50
            )
        ) {
            use proptest::prelude::*;
            let mut idx: PostingsIndex<String> = PostingsIndex::new();
            for (i, doc) in docs.iter().enumerate() {
                let terms: Vec<String> = doc.to_vec();
                idx.add_document(i as u32, &terms).unwrap();
            }
            let n = idx.num_docs();
            for t in idx.terms() {
                let df = idx.df(t);
                prop_assert!(df <= n);
            }
        }
    }

    proptest! {
        #[test]
        fn candidates_have_no_false_negatives(
            docs in prop::collection::vec(
                prop::collection::vec("[a-z]{1,6}", 0..20),
                0..30
            ),
            query in prop::collection::vec("[a-z]{1,6}", 0..10),
        ) {
            let mut idx: PostingsIndex<String> = PostingsIndex::new();
            for (i, terms) in docs.iter().enumerate() {
                let terms: Vec<String> = terms.to_vec();
                idx.add_document(i as DocId, &terms).unwrap();
            }

            let q_terms: Vec<String> = query.to_vec();
            let cands = idx.candidates(&q_terms);
            let cand_set: std::collections::HashSet<DocId> = cands.into_iter().collect();

            // For every live doc, if it contains at least one query term, it must appear in candidates().
            for doc_id in idx.document_ids() {
                let mut hits = false;
                for t in &q_terms {
                    if idx.term_frequency(doc_id, t) > 0 {
                        hits = true;
                        break;
                    }
                }
                if hits {
                    prop_assert!(cand_set.contains(&doc_id));
                }
            }
        }
    }

    #[test]
    fn planner_can_bail_out() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        // Make a very common term.
        for i in 0..100u32 {
            idx.add_document(i, &["common".to_string(), format!("u{i}")])
                .unwrap();
        }
        let cfg = PlannerConfig {
            max_candidate_ratio: 0.2,
            max_candidates: 10,
        };
        let plan = idx.plan_candidates(&["common".to_string()], cfg);
        assert_eq!(plan, CandidatePlan::ScanAll);
    }

    #[test]
    fn generic_term_type_u32_works() {
        // Smoke test that the generic `Term` machinery isn't String-only.
        let mut idx: PostingsIndex<u32> = PostingsIndex::new();
        idx.add_document(0, &[1, 2, 2, 3]).unwrap();
        idx.add_document(1, &[2, 4]).unwrap();
        assert_eq!(idx.df(&2u32), 2);
        assert_eq!(idx.term_frequency(0, &2u32), 2);
        assert_eq!(idx.term_frequency(1, &2u32), 1);

        // With the default bailout config, a term that appears in all docs is allowed to
        // trigger ScanAll. For this smoke test, use a permissive config.
        let plan = idx.plan_candidates(
            &[2u32],
            PlannerConfig {
                max_candidate_ratio: 1.0,
                max_candidates: 10_000,
            },
        );
        match plan {
            CandidatePlan::Candidates(cands) => {
                assert!(cands.contains(&0));
                assert!(cands.contains(&1));
            }
            CandidatePlan::ScanAll => panic!("unexpected bailout for tiny corpus"),
        }
    }

    #[test]
    fn candidates_all_terms_intersects() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(0, &["a".into(), "b".into(), "b".into()])
            .unwrap();
        idx.add_document(1, &["a".into(), "c".into()]).unwrap();
        idx.add_document(2, &["b".into(), "c".into()]).unwrap();

        assert_eq!(
            idx.candidates_all_terms(&["a".to_string(), "b".to_string()]),
            vec![0]
        );
        assert_eq!(
            idx.candidates_all_terms(&["b".to_string(), "c".to_string()]),
            vec![2]
        );
        assert!(idx
            .candidates_all_terms(&["missing".to_string()])
            .is_empty());
    }

    #[test]
    fn candidates_are_sorted_and_unique() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(2, &["a".into(), "b".into()]).unwrap();
        idx.add_document(1, &["a".into()]).unwrap();
        idx.add_document(3, &["b".into()]).unwrap();

        let c = idx.candidates(&["b".to_string(), "a".to_string()]);
        assert_eq!(c, vec![1, 2, 3]);
    }

    proptest! {
        #[test]
        fn candidates_all_terms_have_no_false_negatives(
            docs in prop::collection::vec(
                prop::collection::vec("[a-z]{1,6}", 0..20),
                0..30
            ),
            query in prop::collection::vec("[a-z]{1,6}", 0..10),
        ) {
            let mut idx: PostingsIndex<String> = PostingsIndex::new();
            for (i, terms) in docs.iter().enumerate() {
                let terms: Vec<String> = terms.to_vec();
                idx.add_document(i as DocId, &terms).unwrap();
            }

            let q_terms: Vec<String> = query.to_vec();
            let cands = idx.candidates_all_terms(&q_terms);
            let cand_set: std::collections::HashSet<DocId> = cands.into_iter().collect();

            // For every live doc, if it contains *all* query terms (at least once), it must appear.
            // (Duplicates in the query are treated as a single requirement.)
            let mut uniq: std::collections::HashSet<&String> = std::collections::HashSet::new();
            for t in &q_terms {
                uniq.insert(t);
            }
            for doc_id in idx.document_ids() {
                let mut ok = !uniq.is_empty();
                for t in &uniq {
                    if idx.term_frequency(doc_id, t.as_str()) == 0 {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    prop_assert!(cand_set.contains(&doc_id));
                }
            }
        }
    }

    proptest! {
        #[test]
        fn plan_candidates_candidates_respects_thresholds(
            docs in prop::collection::vec(
                prop::collection::vec("[a-z]{1,6}", 0..30),
                0..60
            ),
            query in prop::collection::vec("[a-z]{1,6}", 0..12),
            max_ratio in 0.05f32..1.0f32,
            max_abs in 1u32..5000u32,
        ) {
            let mut idx: PostingsIndex<String> = PostingsIndex::new();
            for (i, doc) in docs.iter().enumerate() {
                let terms: Vec<String> = doc.to_vec();
                idx.add_document(i as DocId, &terms).unwrap();
            }

            let q_terms: Vec<String> = query.to_vec();
            let cfg = PlannerConfig { max_candidate_ratio: max_ratio, max_candidates: max_abs };

            let plan = idx.plan_candidates(&q_terms, cfg);
            if let CandidatePlan::Candidates(_cands) = plan {
                // If we didn't bail out, then our computed df upper-bound must be below
                // BOTH bailout thresholds (by construction).
                let n = idx.num_docs();
                if n == 0 || q_terms.is_empty() {
                    return Ok(());
                }

                let mut seen: std::collections::HashSet<&String> = std::collections::HashSet::new();
                let mut df_sum: u64 = 0;
                for t in &q_terms {
                    if !seen.insert(t) { continue; }
                    df_sum = df_sum.saturating_add(idx.df(t) as u64);
                }

                prop_assert!(df_sum < (cfg.max_candidates as u64));
                prop_assert!((df_sum as f32) / (n as f32) <= cfg.max_candidate_ratio);
            }
        }
    }
}

fn intersect_sorted(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        let x = a[i];
        let y = b[j];
        if x == y {
            out.push(x);
            i += 1;
            j += 1;
        } else if x < y {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}
