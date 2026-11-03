//! Regression tests for documented behavior that the implementation missed.

use postings::{CandidatePlan, PlannerConfig, PostingsIndex};

/// `Weight for f32` defines document length as the count of non-zero terms.
/// Duplicate pairs for one term accumulate into a single term, and a term
/// whose weight is zero is not counted.
#[test]
fn b4_f32_doc_len_counts_duplicates_and_zero_weights() {
    let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
    idx.add_weighted_document(
        0,
        &[
            ("a".to_string(), 0.5),
            ("a".to_string(), 0.5),
            ("b".to_string(), 0.0),
        ],
    )
    .unwrap();
    assert_eq!(idx.document_len(0), 1);
    assert_eq!(idx.avg_doc_len(), 1.0);

    // Weights that cancel to zero leave no non-zero term.
    idx.add_weighted_document(1, &[("c".to_string(), 0.25), ("c".to_string(), -0.25)])
        .unwrap();
    assert_eq!(idx.document_len(1), 0);

    // Deleting restores the total exactly.
    assert!(idx.delete_document(0));
    assert_eq!(idx.avg_doc_len(), 0.0);
}

/// `u32` document length is still the sum of term frequencies, with
/// duplicate pairs summed.
#[test]
fn b4_u32_doc_len_sums_duplicate_weights() {
    let mut idx: PostingsIndex<String, u32> = PostingsIndex::new();
    idx.add_weighted_document(
        0,
        &[
            ("a".to_string(), 2),
            ("a".to_string(), 3),
            ("b".to_string(), 0),
        ],
    )
    .unwrap();
    assert_eq!(idx.document_len(0), 5);
}

/// One more candidate than the threshold (two, for `max_candidates = 1`).
fn planner_cfg() -> PlannerConfig {
    PlannerConfig {
        max_candidate_ratio: 1.0,
        max_candidates: 1,
    }
}

/// `PlannerConfig::max_candidates` bails out when the candidate upper bound
/// exceeds the count; a bound equal to it plans candidates.
#[test]
fn b5_planner_bails_at_equality_not_exceeds() {
    let mut idx: PostingsIndex<String> = PostingsIndex::new();
    for d in 0..10u32 {
        idx.add_document(d, &[format!("u{d}")]).unwrap();
    }
    idx.add_document(10, &["t".to_string()]).unwrap();
    idx.add_document(11, &["s".to_string()]).unwrap();
    assert_eq!(
        idx.plan_candidates(&["t".to_string()], planner_cfg()),
        CandidatePlan::Candidates(vec![10])
    );
    assert_eq!(
        idx.plan_candidates(&["t".to_string(), "s".to_string()], planner_cfg()),
        CandidatePlan::ScanAll
    );

    // Queries with more than 32 terms take a separate deduplication path.
    let mut long_query: Vec<String> = (0..40).map(|i| format!("missing{i}")).collect();
    long_query.push("t".to_string());
    assert_eq!(
        idx.plan_candidates(&long_query, planner_cfg()),
        CandidatePlan::Candidates(vec![10])
    );
    long_query.push("s".to_string());
    assert_eq!(
        idx.plan_candidates(&long_query, planner_cfg()),
        CandidatePlan::ScanAll
    );
}

#[cfg(feature = "positional")]
#[test]
fn b5_positional_planner_bails_at_equality_not_exceeds() {
    use postings::positional::PosingsIndex;
    let mut idx = PosingsIndex::new();
    for d in 0..10u32 {
        idx.add_document(d, &[format!("u{d}")]).unwrap();
    }
    idx.add_document(10, &["t".to_string()]).unwrap();
    idx.add_document(11, &["s".to_string()]).unwrap();
    assert_eq!(
        idx.plan_candidates_near(["t"], planner_cfg()),
        CandidatePlan::Candidates(vec![10])
    );
    assert_eq!(
        idx.plan_candidates_near(["t", "s"], planner_cfg()),
        CandidatePlan::ScanAll
    );
}

#[cfg(feature = "raw-segment")]
fn raw_planner_segment() -> Vec<u8> {
    use postings::raw::{write_u64_u32_segment, RawDocument};
    let filler: Vec<Vec<(u64, u32)>> = (0..10u64).map(|t| vec![(100 + t, 1u32)]).collect();
    let t = vec![(7u64, 1u32)];
    let s = vec![(9u64, 1u32)];
    let mut docs: Vec<RawDocument<'_>> = filler
        .iter()
        .enumerate()
        .map(|(d, terms)| RawDocument::new(d as u32, terms))
        .collect();
    docs.push(RawDocument::new(10, &t));
    docs.push(RawDocument::new(11, &s));
    write_u64_u32_segment(&docs).unwrap()
}

#[cfg(feature = "raw-segment")]
#[test]
fn b5_raw_planner_bails_at_equality_not_exceeds() {
    use postings::raw::{RawSegment, RawSegmentFile};
    use std::io::{Seek, SeekFrom, Write};

    let bytes = raw_planner_segment();
    let seg = RawSegment::open(&bytes).unwrap();
    assert_eq!(
        seg.plan_candidates(&[7], planner_cfg()).unwrap(),
        CandidatePlan::Candidates(vec![10])
    );
    assert_eq!(
        seg.plan_candidates(&[7, 9], planner_cfg()).unwrap(),
        CandidatePlan::ScanAll
    );

    let mut file = tempfile::tempfile().unwrap();
    file.write_all(&bytes).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut seg = RawSegmentFile::from_file(file).unwrap();
    assert_eq!(
        seg.plan_candidates(&[7], planner_cfg()).unwrap(),
        CandidatePlan::Candidates(vec![10])
    );
    assert_eq!(
        seg.plan_candidates(&[7, 9], planner_cfg()).unwrap(),
        CandidatePlan::ScanAll
    );
}
