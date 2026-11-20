//! The file-backed block-max WAND path must agree with the in-memory scorer.
//!
//! That path runs when a query has at least eight terms, all weights are
//! non-negative, and one posting list is larger than 1 MiB with more than one
//! block, so these tests build segments of several hundred thousand documents.
#![cfg(feature = "raw-segment")]

use postings::raw::{
    top_k_weighted_u32_files, write_u64_u32_segment, RawDocument, RawSegment, RawSegmentFile,
};
use std::io::Write;

fn assert_file_matches_memory(docs: &[(u32, Vec<(u64, u32)>)], query: &[(u64, f32)], what: &str) {
    let raw: Vec<RawDocument> = docs.iter().map(|(d, t)| RawDocument::new(*d, t)).collect();
    let bytes = write_u64_u32_segment(&raw).unwrap();
    let memory = RawSegment::open(&bytes).unwrap();
    let mut handle = tempfile::tempfile().unwrap();
    handle.write_all(&bytes).unwrap();
    let mut file = RawSegmentFile::from_file(handle).unwrap();
    for k in [1usize, 3, 10, 50] {
        let expected = memory.top_k_weighted_u32(query, k).unwrap();
        assert_eq!(
            file.top_k_weighted_u32(query, k).unwrap(),
            expected,
            "{what}: RawSegmentFile, k = {k}"
        );
        let mut files = vec![&mut file];
        assert_eq!(
            top_k_weighted_u32_files(&mut files, query, k).unwrap(),
            expected,
            "{what}: top_k_weighted_u32_files, k = {k}"
        );
    }
}

/// Document 129 sits in a later, heavier block of term 1 than the block at the
/// candidate where the old pivot search evaluated its bound, and was skipped.
#[test]
fn heavy_document_between_cursor_positions_is_not_skipped() {
    let mut docs: Vec<(u32, Vec<(u64, u32)>)> = Vec::new();
    let mut first = vec![(3u64, 5u32)];
    first.extend((4..=9u64).map(|term| (term, 1)));
    docs.push((0, first));
    docs.extend((1..=128u32).map(|d| (d, vec![(1, 1)])));
    docs.push((129, vec![(1, 20)]));
    docs.push((1000, vec![(2, 12)]));
    docs.extend((2000..602_000u32).map(|d| (d, vec![(1, 1)])));
    let query: Vec<(u64, f32)> = (1..=9u64).map(|term| (term, 1.0)).collect();

    let raw: Vec<RawDocument> = docs.iter().map(|(d, t)| RawDocument::new(*d, t)).collect();
    let bytes = write_u64_u32_segment(&raw).unwrap();
    let memory = RawSegment::open(&bytes).unwrap();
    assert_eq!(
        memory.top_k_weighted_u32(&query, 1).unwrap(),
        vec![(129, 20.0)]
    );

    assert_file_matches_memory(&docs, &query, "handcrafted");
}

#[test]
fn random_corpora_agree_with_the_in_memory_scorer() {
    for seed in [1u64, 2, 3] {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let mut docs: Vec<(u32, Vec<(u64, u32)>)> = Vec::new();
        let mut doc_id = 0u32;
        while docs.len() < 600_000 {
            // Irregular gaps, so block boundaries differ between lists.
            doc_id += 1 + next(3) as u32;
            let mut terms: Vec<(u64, u32)> = Vec::new();
            if next(20) != 0 {
                // One long list with rare heavy postings.
                let weight = if next(4000) == 0 {
                    20 + next(40) as u32
                } else {
                    1
                };
                terms.push((1, weight));
            }
            for term in 2..=9u64 {
                if next(1500) == 0 {
                    terms.push((term, 1 + next(30) as u32));
                }
            }
            if !terms.is_empty() {
                docs.push((doc_id, terms));
            }
        }
        let query: Vec<(u64, f32)> = (1..=9u64)
            .map(|term| (term, 0.5 + (term % 3) as f32))
            .collect();
        assert_file_matches_memory(&docs, &query, &format!("seed {seed}"));
    }
}
