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
//! ## Optional modules
//!
//! - `postings::codec`: low-level codecs (varint/gap) for postings payloads (in this repo).
//! - `postings::positional` (feature `positional`): positional postings for phrase/proximity evaluation.
//!
//! ## Non-goals (for now)
//!
//! - On-disk persistence / compaction
//! - Rich query language beyond candidate generation and sparse top-k scoring

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod codec;

#[cfg(feature = "positional")]
pub mod positional;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Document identifier.
pub type DocId = u32;

/// Trait for term weight types in posting lists.
///
/// `u32` is the default for classical term frequency counts.
/// `f32` enables learned sparse representations (SPLADE, SPLADE++, etc.)
/// where terms carry continuous weights rather than integer frequencies.
pub trait Weight: Copy + Default + std::fmt::Debug + 'static {
    /// The zero/absent weight.
    fn zero() -> Self;
    /// Accumulate (add) a weight into self.
    fn accumulate(&mut self, other: Self);
    /// Convert to f32 for scoring.
    fn to_f32(self) -> f32;
    /// Convert to u64 for total doc length accumulation.
    fn to_doc_len(self) -> u64;
}

impl Weight for u32 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn accumulate(&mut self, other: Self) {
        *self += other;
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn to_doc_len(self) -> u64 {
        self as u64
    }
}

impl Weight for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn accumulate(&mut self, other: Self) {
        *self += other;
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn to_doc_len(self) -> u64 {
        // For float weights, doc length is the count of non-zero terms
        // (not the sum of weights), keeping avg_doc_len meaningful for BM25
        1
    }
}

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

/// Per-document metadata stored inside a segment (delete support only).
///
/// Slimmed down from the original design: `postings` and `doc_len` are now
/// tracked globally on `PostingsIndex`, so a segment only needs the term list
/// for df adjustment on delete.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "Term: serde::Serialize, W: serde::Serialize",
        deserialize = "Term: serde::Deserialize<'de> + Eq + std::hash::Hash, W: serde::Deserialize<'de>"
    ))
)]
struct Segment<Term, W: Weight = u32> {
    /// doc_id -> unique terms in that doc (for df adjustments on delete).
    doc_terms: HashMap<DocId, Vec<Term>>,
    /// Phantom to keep the W type parameter (serde compat, zero size in practice).
    #[cfg_attr(feature = "serde", serde(skip))]
    _w: std::marker::PhantomData<W>,
}

impl<Term, W: Weight> Default for Segment<Term, W> {
    fn default() -> Self {
        Self {
            doc_terms: HashMap::new(),
            _w: std::marker::PhantomData,
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
        serialize = "Term: serde::Serialize, W: serde::Serialize",
        deserialize = "Term: serde::Deserialize<'de> + Eq + std::hash::Hash, W: serde::Deserialize<'de>"
    ))
)]
pub struct PostingsIndex<Term = String, W: Weight = u32> {
    segments: Vec<Segment<Term, W>>,
    /// live doc -> segment index (used by delete to find doc_terms)
    doc_segment: HashMap<DocId, usize>,
    /// live doc -> length
    doc_len: HashMap<DocId, u32>,
    /// term -> df (number of live documents containing term)
    df: HashMap<Term, u32>,
    total_doc_len: u64,
    /// Flat global postings: term -> sorted live (doc_id, weight) list.
    ///
    /// Deletes remove postings eagerly; query hot paths rely on that private
    /// invariant to avoid a per-posting live-doc check.
    global_postings: HashMap<Term, Vec<(DocId, W)>>,
}

impl<Term, W: Weight> Default for PostingsIndex<Term, W> {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
            doc_segment: HashMap::new(),
            doc_len: HashMap::new(),
            df: HashMap::new(),
            total_doc_len: 0,
            global_postings: HashMap::new(),
        }
    }
}

impl<Term, W: Weight> PostingsIndex<Term, W>
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

    /// Add a document by doc id and weighted term pairs.
    ///
    /// Each `(term, weight)` pair represents a term and its weight in the document.
    /// For classical indexing, weight is the term frequency (u32).
    /// For learned sparse retrieval (SPLADE), weight is a continuous value (f32).
    ///
    /// Duplicate terms are accumulated (weights summed).
    ///
    /// If `doc_id` already exists, return an error. Call `delete_document` first
    /// to model updates as delete+add (segment-style).
    pub fn add_weighted_document(
        &mut self,
        doc_id: DocId,
        weighted_terms: &[(Term, W)],
    ) -> Result<(), Error> {
        if self.doc_segment.contains_key(&doc_id) {
            return Err(Error::DuplicateDocId(doc_id));
        }

        let mut term_weights: HashMap<Term, W> = HashMap::new();
        let mut doc_length: u64 = 0;
        for (t, w) in weighted_terms {
            term_weights
                .entry(t.clone())
                .and_modify(|existing| existing.accumulate(*w))
                .or_insert(*w);
            doc_length += w.to_doc_len();
        }

        self.index_document_inner(doc_id, term_weights, doc_length);
        Ok(())
    }

    /// Core indexing logic shared by `add_weighted_document` and `add_document`.
    ///
    /// Caller must check for duplicate doc_id before calling.
    fn index_document_inner(
        &mut self,
        doc_id: DocId,
        term_weights: HashMap<Term, W>,
        doc_length: u64,
    ) {
        let mut doc_terms: Vec<Term> = term_weights.keys().cloned().collect();
        doc_terms.sort_unstable();

        // Keep global postings sorted so lookup and intersection paths can use
        // binary search and galloping search even when callers insert doc ids
        // out of order.
        for (term, w) in &term_weights {
            let postings = self.global_postings.entry(term.clone()).or_default();
            match postings.binary_search_by_key(&doc_id, |(id, _)| *id) {
                Ok(i) => postings[i].1 = *w,
                Err(i) => postings.insert(i, (doc_id, *w)),
            }
        }

        // Update global df before moving doc_terms into the segment.
        for term in &doc_terms {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }

        // Build a lightweight segment (doc_terms only, for delete support).
        let mut seg = Segment::<Term, W>::default();
        seg.doc_terms.insert(doc_id, doc_terms); // move, not clone

        let seg_idx = self.segments.len();
        self.segments.push(seg);
        self.doc_segment.insert(doc_id, seg_idx);
        self.doc_len.insert(doc_id, doc_length as u32);
        self.total_doc_len += doc_length;
    }
}

/// Backward-compatible methods for integer-weighted (classical) postings.
impl<Term> PostingsIndex<Term, u32>
where
    Term: Clone + Eq + Hash + Ord,
{
    /// Add a document by doc id and term stream (classical indexing).
    ///
    /// Terms are counted to produce integer term frequencies.
    /// For weighted terms (SPLADE), use [`add_weighted_document`](PostingsIndex::add_weighted_document).
    pub fn add_document(&mut self, doc_id: DocId, terms: &[Term]) -> Result<(), Error> {
        if self.doc_segment.contains_key(&doc_id) {
            return Err(Error::DuplicateDocId(doc_id));
        }
        let mut term_counts: HashMap<Term, u32> = HashMap::new();
        for t in terms {
            *term_counts.entry(t.clone()).or_insert(0) += 1;
        }
        let doc_length = terms.len() as u64;
        self.index_document_inner(doc_id, term_counts, doc_length);
        Ok(())
    }
}

impl<Term, W: Weight> PostingsIndex<Term, W>
where
    Term: Clone + Eq + Hash + Ord,
{
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
                if let Some(postings) = self.global_postings.get_mut(term) {
                    postings.retain(|(id, _)| *id != doc_id);
                    if postings.is_empty() {
                        self.global_postings.remove(term);
                    }
                }
            }
        }
        true
    }

    /// Term weight (frequency) of `term` in `doc_id`.
    ///
    /// Returns `W::zero()` if the term is not present or the doc is unknown.
    /// For classical indexes (`W = u32`), this is the term frequency count.
    /// For learned sparse indexes (`W = f32`), this is the SPLADE weight.
    pub fn term_frequency<Q>(&self, doc_id: DocId, term: &Q) -> W
    where
        Term: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Doc must be live.
        if !self.doc_segment.contains_key(&doc_id) {
            return W::zero();
        }
        let postings = match self.global_postings.get(term) {
            Some(p) => p,
            None => return W::zero(),
        };
        match postings.binary_search_by_key(&doc_id, |(id, _)| *id) {
            Ok(i) => postings[i].1,
            Err(_) => W::zero(),
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
        let mut lists: Vec<&[(DocId, W)]> = Vec::new();
        let mut seen_terms: HashSet<&Q> = HashSet::new();
        for term in query_terms {
            if !seen_terms.insert(term) {
                continue;
            }
            if let Some(postings) = self.global_postings.get(term) {
                if !postings.is_empty() {
                    lists.push(postings);
                }
            }
        }
        if lists.is_empty() {
            return Vec::new();
        }

        lists.sort_by_key(|postings| postings.len());
        let mut lists = lists.into_iter();
        let mut out: Vec<DocId> = lists
            .next()
            .map(|postings| postings.iter().map(|(id, _)| *id).collect())
            .unwrap_or_default();
        for docs in lists {
            out = union_sorted_postings(&out, docs);
        }
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

        // Collect doc ids for the rarest term; global postings are sorted and
        // delete removes stale entries eagerly.
        let mut acc: Vec<DocId> = {
            let Some(list) = self.global_postings.get(uniq[0]) else {
                return Vec::new();
            };
            list.iter().map(|(id, _)| *id).collect()
        };

        for &t in uniq.iter().skip(1) {
            if self.df(t) == 0 {
                return Vec::new();
            }
            match self.global_postings.get(t) {
                Some(list) => {
                    acc = intersect_sorted_postings(&acc, list);
                }
                None => return Vec::new(),
            }
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
    ///
    /// Reads from the global flat postings map in O(1) and filters out
    /// logically-deleted documents.
    pub fn postings_iter<'a, Q>(&'a self, term: &'a Q) -> impl Iterator<Item = (DocId, W)> + 'a
    where
        Term: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.global_postings
            .get(term)
            .into_iter()
            .flat_map(|v| v.iter().copied())
            .filter(|(doc_id, _)| self.doc_segment.contains_key(doc_id))
    }

    /// Return the top `k` documents by sparse inner product.
    ///
    /// Query terms are borrowed so a `PostingsIndex<String, f32>` can be queried
    /// with `&str` terms without allocating. Duplicate query terms are
    /// accumulated before scoring. Ties are broken by ascending doc id.
    pub fn top_k_weighted<Q>(&self, query_terms: &[(&Q, f32)], k: usize) -> Vec<(DocId, f32)>
    where
        Term: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if k == 0 || query_terms.is_empty() {
            return Vec::new();
        }
        if query_terms.len() == 1 {
            let (term, query_weight) = query_terms[0];
            if query_weight == 0.0 {
                return Vec::new();
            }
            let Some(postings) = self.global_postings.get(term) else {
                return Vec::new();
            };
            return top_k_single_postings(postings, query_weight, k);
        }

        let mut query_weights: HashMap<&Q, f32> = HashMap::new();
        for &(term, weight) in query_terms {
            if weight == 0.0 {
                continue;
            }
            *query_weights.entry(term).or_insert(0.0) += weight;
        }
        if query_weights.is_empty() {
            return Vec::new();
        }

        let mut lists = Vec::with_capacity(query_weights.len());
        let mut total_postings = 0usize;
        for (term, query_weight) in query_weights {
            if query_weight == 0.0 {
                continue;
            }
            let Some(postings) = self.global_postings.get(term) else {
                continue;
            };
            total_postings = total_postings.saturating_add(postings.len());
            lists.push((postings.as_slice(), query_weight));
        }
        if lists.is_empty() {
            return Vec::new();
        }

        if lists.len() == 1 {
            let (postings, query_weight) = lists[0];
            return top_k_single_postings(postings, query_weight, k);
        }

        let dense_slots = lists
            .iter()
            .filter_map(|(postings, _)| {
                let (last_doc_id, _) = postings.last()?;
                usize::try_from(*last_doc_id).ok()?.checked_add(1)
            })
            .max()
            .unwrap_or(0);
        let dense_limit = self.doc_len.len().saturating_mul(4).max(1024);
        if dense_slots <= dense_limit {
            let mut scores = vec![0.0; dense_slots];
            let mut seen = vec![false; dense_slots];
            let mut touched = Vec::with_capacity(total_postings.min(self.doc_len.len()));

            for (postings, query_weight) in lists {
                for &(doc_id, doc_weight) in postings {
                    let contribution = query_weight * doc_weight.to_f32();
                    if contribution == 0.0 {
                        continue;
                    }
                    let slot = doc_id as usize;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(doc_id);
                    }
                    scores[slot] += contribution;
                }
            }

            let mut ranked: Vec<(DocId, f32)> = touched
                .into_iter()
                .map(|doc_id| (doc_id, scores[doc_id as usize]))
                .collect();
            keep_top_k(&mut ranked, k);
            return ranked;
        }

        let mut scores: HashMap<DocId, f32> =
            HashMap::with_capacity(total_postings.min(self.doc_len.len()));
        for (postings, query_weight) in lists {
            for &(doc_id, doc_weight) in postings {
                let contribution = query_weight * doc_weight.to_f32();
                if contribution != 0.0 {
                    *scores.entry(doc_id).or_insert(0.0) += contribution;
                }
            }
        }

        let mut ranked: Vec<(DocId, f32)> = scores.into_iter().collect();
        keep_top_k(&mut ranked, k);
        ranked
    }

    /// Save the index to a directory using `durability`.
    ///
    /// **Format note**: the on-disk layout changed in 0.2.0 when the internal
    /// `global_postings` field was added for query performance.  Data written
    /// by 0.1.x cannot be read by 0.2.0+ and vice-versa.
    #[cfg(feature = "persistence")]
    pub fn save<D: durability::Directory + ?Sized>(
        &self,
        dir: &D,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        Term: serde::Serialize,
        W: serde::Serialize,
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
    pub fn save_durable<D: durability::Directory + ?Sized>(
        &self,
        dir: &D,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        Term: serde::Serialize,
        W: serde::Serialize,
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
        for<'de> W: serde::Deserialize<'de>,
    {
        use std::io::Read;

        let mut f = dir.open_file(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        let idx: Self = postcard::from_bytes(&bytes)?;
        Ok(idx)
    }
}

fn union_sorted_postings<W>(a: &[DocId], b: &[(DocId, W)]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(a.len().saturating_add(b.len()));
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                push_unique(&mut out, a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                push_unique(&mut out, a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                push_unique(&mut out, b[j].0);
                j += 1;
            }
        }
    }
    while i < a.len() {
        push_unique(&mut out, a[i]);
        i += 1;
    }
    while j < b.len() {
        push_unique(&mut out, b[j].0);
        j += 1;
    }
    out
}

#[inline]
fn push_unique(out: &mut Vec<DocId>, doc_id: DocId) {
    if out.last().copied() != Some(doc_id) {
        out.push(doc_id);
    }
}

fn intersect_sorted_postings<W>(a: &[DocId], b: &[(DocId, W)]) -> Vec<DocId> {
    // Galloping (exponential search) intersection.
    // Invariant: when a[i] != b[j], advance the cursor of the *smaller*
    // value by gallopping it forward to the *larger* value.  Both cursors
    // always make progress (gallop_forward returns strictly > start when
    // the target is not at start), so the loop terminates.
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                // a[i] < b[j]: gallop a forward to reach b[j].
                i = gallop_forward(a, i, b[j].0);
            }
            std::cmp::Ordering::Greater => {
                // b[j] < a[i]: gallop b forward to reach a[i].
                j = gallop_forward_postings(b, j, a[i]);
            }
        }
    }
    out
}

/// Returns the first index `>= start` in `list` such that `list[idx] >= target`,
/// or `list.len()` if no such index exists.  Uses exponential probing + binary
/// search so cost is O(log(gap)) rather than O(gap).
#[inline]
fn gallop_forward(list: &[DocId], start: usize, target: DocId) -> usize {
    if start >= list.len() || list[start] >= target {
        return start;
    }
    let mut step = 1usize;
    // Exponential probe to find an upper bound.
    while start + step < list.len() && list[start + step] < target {
        step <<= 1;
    }
    // Binary search in [start + step/2, start + step).
    let lo = start + (step >> 1);
    let hi = (start + step).min(list.len());
    match list[lo..hi].binary_search(&target) {
        Ok(k) => lo + k,
        Err(k) => lo + k,
    }
}

/// Postings-list equivalent of `gallop_forward`.
#[inline]
fn gallop_forward_postings<W>(list: &[(DocId, W)], start: usize, target: DocId) -> usize {
    if start >= list.len() || list[start].0 >= target {
        return start;
    }
    let mut step = 1usize;
    while start + step < list.len() && list[start + step].0 < target {
        step <<= 1;
    }
    let lo = start + (step >> 1);
    let hi = (start + step).min(list.len());
    match list[lo..hi].binary_search_by_key(&target, |(id, _)| *id) {
        Ok(k) => lo + k,
        Err(k) => lo + k,
    }
}

#[inline]
fn cmp_doc_scores(a: &(DocId, f32), b: &(DocId, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

#[inline]
fn keep_top_k(ranked: &mut Vec<(DocId, f32)>, k: usize) {
    if ranked.len() > k {
        ranked.select_nth_unstable_by(k, cmp_doc_scores);
        ranked.truncate(k);
    }
    ranked.sort_by(cmp_doc_scores);
}

fn top_k_single_postings<W: Weight>(
    postings: &[(DocId, W)],
    query_weight: f32,
    k: usize,
) -> Vec<(DocId, f32)> {
    let mut ranked = Vec::with_capacity(postings.len());
    for &(doc_id, doc_weight) in postings {
        let score = query_weight * doc_weight.to_f32();
        if score != 0.0 {
            ranked.push((doc_id, score));
        }
    }
    keep_top_k(&mut ranked, k);
    ranked
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
    fn term_frequency_handles_out_of_order_doc_ids() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(2, &[String::from("a")]).unwrap();
        idx.add_document(1, &[String::from("a")]).unwrap();

        assert_eq!(idx.candidates(&[String::from("a")]), vec![1, 2]);
        assert_eq!(idx.term_frequency(1, "a"), 1);
        assert_eq!(idx.term_frequency(2, "a"), 1);
    }

    #[test]
    fn delete_then_readd_does_not_revive_stale_postings() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(7, &[String::from("old")]).unwrap();
        assert!(idx.delete_document(7));
        idx.add_document(7, &[String::from("new")]).unwrap();

        assert!(idx.candidates(&[String::from("old")]).is_empty());
        assert_eq!(idx.candidates(&[String::from("new")]), vec![7]);
        assert_eq!(idx.term_frequency(7, "old"), 0);
        assert_eq!(idx.term_frequency(7, "new"), 1);
    }

    #[test]
    fn candidates_skip_deleted_docs() {
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(1, &[String::from("shared")]).unwrap();
        idx.add_document(2, &[String::from("shared")]).unwrap();

        assert!(idx.delete_document(1));

        let query = [String::from("shared")];
        assert_eq!(idx.candidates(&query), vec![2]);
        assert_eq!(idx.candidates_all_terms(&query), vec![2]);
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

    // ── Float-weighted (SPLADE) tests ─────────────────────────────────

    #[test]
    fn float_weighted_index_basic() {
        // SPLADE-style: terms with continuous weights
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(
            0,
            &[
                (String::from("neural"), 0.42),
                (String::from("network"), 0.87),
                (String::from("deep"), 0.15),
            ],
        )
        .unwrap();

        assert_eq!(idx.num_docs(), 1);
        assert!((idx.term_frequency(0, "neural") - 0.42).abs() < 1e-6);
        assert!((idx.term_frequency(0, "network") - 0.87).abs() < 1e-6);
        assert!((idx.term_frequency(0, "missing") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn float_weighted_candidates() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(0, &[(String::from("cat"), 0.9), (String::from("dog"), 0.3)])
            .unwrap();
        idx.add_weighted_document(
            1,
            &[(String::from("dog"), 0.8), (String::from("fish"), 0.5)],
        )
        .unwrap();

        // Query for "dog" -- both docs have it
        let cands = idx.candidates(&[String::from("dog")]);
        assert_eq!(cands.len(), 2);

        // Query for "cat" -- only doc 0
        let cands = idx.candidates(&[String::from("cat")]);
        assert_eq!(cands, vec![0]);

        // df
        assert_eq!(idx.df("dog"), 2);
        assert_eq!(idx.df("cat"), 1);
    }

    #[test]
    fn float_weighted_delete() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(0, &[(String::from("a"), 0.5)])
            .unwrap();
        idx.add_weighted_document(1, &[(String::from("a"), 0.8)])
            .unwrap();

        assert_eq!(idx.df("a"), 2);
        idx.delete_document(0);
        assert_eq!(idx.df("a"), 1);
        assert_eq!(idx.num_docs(), 1);
    }

    #[test]
    fn float_weighted_accumulates_duplicates() {
        // If same term appears twice, weights should accumulate
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(
            0,
            &[(String::from("term"), 0.3), (String::from("term"), 0.4)],
        )
        .unwrap();

        // 0.3 + 0.4 = 0.7
        assert!((idx.term_frequency(0, "term") - 0.7).abs() < 1e-6);
    }

    #[test]
    fn top_k_weighted_scores_sparse_inner_product() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(
            0,
            &[
                (String::from("neural"), 1.8),
                (String::from("network"), 2.1),
                (String::from("deep"), 0.9),
                (String::from("learning"), 1.2),
            ],
        )
        .unwrap();
        idx.add_weighted_document(
            1,
            &[
                (String::from("graph"), 2.4),
                (String::from("network"), 1.1),
                (String::from("node"), 1.7),
            ],
        )
        .unwrap();
        idx.add_weighted_document(
            2,
            &[
                (String::from("neural"), 0.7),
                (String::from("search"), 2.2),
                (String::from("retrieval"), 2.6),
                (String::from("learning"), 0.5),
            ],
        )
        .unwrap();
        idx.add_weighted_document(
            3,
            &[
                (String::from("retrieval"), 1.9),
                (String::from("sparse"), 2.8),
                (String::from("index"), 1.3),
                (String::from("search"), 1.0),
            ],
        )
        .unwrap();

        let ranked = idx.top_k_weighted(&[("neural", 1.5), ("retrieval", 2.0), ("search", 1.0)], 3);

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, 2);
        assert!((ranked[0].1 - 8.45).abs() < 1e-6);
        assert_eq!(ranked[1].0, 3);
        assert!((ranked[1].1 - 4.8).abs() < 1e-6);
        assert_eq!(ranked[2].0, 0);
        assert!((ranked[2].1 - 2.7).abs() < 1e-6);
    }

    #[test]
    fn top_k_weighted_truncates_and_ties_by_doc_id() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(4, &[(String::from("term"), 1.0)])
            .unwrap();
        idx.add_weighted_document(2, &[(String::from("term"), 1.0)])
            .unwrap();
        idx.add_weighted_document(3, &[(String::from("term"), 1.0)])
            .unwrap();

        let ranked = idx.top_k_weighted(&[("term", 1.0)], 2);

        assert_eq!(ranked, vec![(2, 1.0), (3, 1.0)]);
    }

    #[test]
    fn top_k_weighted_accumulates_duplicate_query_terms() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(0, &[(String::from("term"), 2.0)])
            .unwrap();

        let ranked = idx.top_k_weighted(&[("term", 1.0), ("term", 0.5)], 1);

        assert_eq!(ranked, vec![(0, 3.0)]);
    }

    #[test]
    fn top_k_weighted_single_term_zero_weight_and_missing_term_are_empty() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(0, &[(String::from("term"), 2.0)])
            .unwrap();

        assert!(idx.top_k_weighted(&[("term", 0.0)], 10).is_empty());
        assert!(idx.top_k_weighted(&[("missing", 1.0)], 10).is_empty());
    }

    #[test]
    fn top_k_weighted_ignores_deleted_docs() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(0, &[(String::from("term"), 10.0)])
            .unwrap();
        idx.add_weighted_document(1, &[(String::from("term"), 1.0)])
            .unwrap();
        assert!(idx.delete_document(0));

        let ranked = idx.top_k_weighted(&[("term", 1.0)], 10);

        assert_eq!(ranked, vec![(1, 1.0)]);
    }

    #[test]
    fn top_k_weighted_handles_score_cancellation_without_duplicate_docs() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(
            0,
            &[
                (String::from("positive"), 1.0),
                (String::from("negative"), 1.0),
                (String::from("late"), 1.0),
            ],
        )
        .unwrap();
        idx.add_weighted_document(1, &[(String::from("late"), 1.0)])
            .unwrap();

        let ranked =
            idx.top_k_weighted(&[("positive", 1.0), ("negative", -1.0), ("late", 2.0)], 10);

        assert_eq!(ranked, vec![(0, 2.0), (1, 2.0)]);
    }

    #[test]
    fn top_k_weighted_handles_sparse_doc_ids() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(7, &[(String::from("term"), 1.0)])
            .unwrap();
        idx.add_weighted_document(1_000_000, &[(String::from("term"), 2.0)])
            .unwrap();

        let ranked = idx.top_k_weighted(&[("term", 1.0)], 10);

        assert_eq!(ranked, vec![(1_000_000, 2.0), (7, 1.0)]);
    }

    #[test]
    fn top_k_weighted_accumulates_sparse_doc_id_fallback() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(
            7,
            &[
                (String::from("positive"), 1.0),
                (String::from("negative"), 1.0),
                (String::from("late"), 1.0),
            ],
        )
        .unwrap();
        idx.add_weighted_document(1_000_000, &[(String::from("late"), 2.0)])
            .unwrap();

        let ranked =
            idx.top_k_weighted(&[("positive", 1.0), ("negative", -1.0), ("late", 2.0)], 10);

        assert_eq!(ranked, vec![(1_000_000, 4.0), (7, 2.0)]);
    }

    #[test]
    fn top_k_weighted_zero_k_is_empty() {
        let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
        idx.add_weighted_document(0, &[(String::from("term"), 1.0)])
            .unwrap();

        assert!(idx.top_k_weighted(&[("term", 1.0)], 0).is_empty());
    }

    #[test]
    fn classic_u32_still_works_unchanged() {
        // Verify the default u32 path is backward compatible
        let mut idx: PostingsIndex<String> = PostingsIndex::new();
        idx.add_document(0, &[String::from("hello"), String::from("hello")])
            .unwrap();
        // "hello" appears twice -> tf=2
        assert_eq!(idx.term_frequency(0, "hello"), 2);
        assert_eq!(idx.document_len(0), 2); // 2 terms total
    }
}
