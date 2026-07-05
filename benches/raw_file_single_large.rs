//! Focused Criterion target for large single-term file-backed raw top-k search.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use postings::raw::{
    write_u64_u32_segment_from_term_postings_seekable_to, RawSegmentFile, RawTermId,
    RawTermPostingList,
};
use std::fs::File;

const DOCS: usize = 600_000;
const TERM: RawTermId = 42;
const HIGH_BLOCK_DOCS: u32 = 128;

type SingleLargeFixture = (tempfile::TempDir, RawSegmentFile, Vec<(RawTermId, f32)>);

fn build_single_large_filtered_fixture() -> SingleLargeFixture {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single-large.raw");
    let document_lengths: Vec<_> = (0..DOCS as u32).map(|doc_id| (doc_id, 1)).collect();
    let postings: Vec<_> = (0..DOCS as u32)
        .map(|doc_id| {
            let weight = if doc_id < HIGH_BLOCK_DOCS {
                10_000 - (doc_id % 17)
            } else {
                1
            };
            (doc_id, weight)
        })
        .collect();
    let terms = [RawTermPostingList::new(TERM, &postings)];
    let mut file = File::create(&path).unwrap();
    write_u64_u32_segment_from_term_postings_seekable_to(&document_lengths, &terms, &mut file)
        .unwrap();

    (dir, RawSegmentFile::open(path).unwrap(), vec![(TERM, 1.0)])
}

fn bench_single_large(c: &mut Criterion) {
    let (_dir, mut segment, query_terms) = build_single_large_filtered_fixture();
    assert!(segment.posting_blocks(TERM).unwrap().len() > 1);
    let result = segment
        .top_k_weighted_u32_filtered(&query_terms, 10, |doc_id| doc_id % 7 != 0)
        .unwrap();
    assert_eq!(result.len(), 10);
    assert!(result
        .iter()
        .all(|(doc_id, score)| *doc_id < HIGH_BLOCK_DOCS && *score > 9_900.0));

    let mut group = c.benchmark_group("raw_file_single_large");
    group.bench_function("filtered", |b| {
        b.iter(|| {
            black_box(
                segment
                    .top_k_weighted_u32_filtered(
                        black_box(query_terms.as_slice()),
                        black_box(10),
                        |doc_id| doc_id % 7 != 0,
                    )
                    .unwrap(),
            );
        });
    });
    group.finish();
}

criterion_group!(benches, bench_single_large);
criterion_main!(benches);
