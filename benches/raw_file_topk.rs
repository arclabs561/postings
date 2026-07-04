//! Focused Criterion target for profiling file-backed raw top-k search.
//!
//! `benches/query.rs` covers the broad query suite. This target keeps filtered
//! raw multi-file timings and profiles from paying unrelated query-suite setup.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use postings::raw::{
    top_k_weighted_u32_files, write_u64_u32_segment, RawDocument, RawSegmentFile, RawTermId,
};
use std::collections::BTreeMap;

const N_DOCS: usize = 50_000;
const VOCAB_SIZE: usize = 10_000;
const TERMS_PER_DOC: usize = 100;
const SEGMENTS: usize = 64;

type WeightedDocs = Vec<Vec<(RawTermId, u32)>>;

fn zipf_sample(rng: &mut u64, vocab_size: usize) -> usize {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let u = (*rng as f64) / (u64::MAX as f64);
    let n = vocab_size as f64;
    let k = (u * (n + 1.0_f64).ln()).exp() - 1.0;
    (k as usize).min(vocab_size - 1)
}

fn build_weighted_docs() -> (WeightedDocs, Vec<u32>) {
    let mut docs = Vec::with_capacity(N_DOCS);
    let mut dfs = vec![0u32; VOCAB_SIZE];
    let mut rng = 0xdeadbeef_cafebabe;

    for _ in 0..N_DOCS {
        let mut counts = BTreeMap::new();
        for _ in 0..TERMS_PER_DOC {
            let term_id = zipf_sample(&mut rng, VOCAB_SIZE) as RawTermId;
            *counts.entry(term_id).or_insert(0u32) += 1;
        }
        for &term_id in counts.keys() {
            dfs[term_id as usize] += 1;
        }
        docs.push(counts.into_iter().collect());
    }

    (docs, dfs)
}

fn raw_segment_bytes_from_docs(
    weighted_docs: &[Vec<(RawTermId, u32)>],
    start_doc_id: u32,
) -> Vec<u8> {
    let raw_docs: Vec<_> = weighted_docs
        .iter()
        .enumerate()
        .map(|(offset, terms)| RawDocument::new(start_doc_id + offset as u32, terms))
        .collect();
    write_u64_u32_segment(&raw_docs).unwrap()
}

fn build_fixture() -> (
    tempfile::TempDir,
    Vec<RawSegmentFile>,
    Vec<(RawTermId, f32)>,
) {
    let (weighted_docs, dfs) = build_weighted_docs();
    let weighted_terms: Vec<_> = dfs
        .iter()
        .enumerate()
        .filter(|(_, df)| **df >= 10)
        .take(5)
        .enumerate()
        .map(|(i, (term_id, _))| (term_id as RawTermId, 1.0 + (i as f32 * 0.1)))
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let chunk_size = N_DOCS.div_ceil(SEGMENTS);
    let mut segments = Vec::new();
    for (chunk_index, chunk) in weighted_docs.chunks(chunk_size).enumerate() {
        let start_doc_id = (chunk_index * chunk_size) as u32;
        let path = dir.path().join(format!("numeric-{chunk_index}.raw"));
        std::fs::write(&path, raw_segment_bytes_from_docs(chunk, start_doc_id)).unwrap();
        segments.push(RawSegmentFile::open(path).unwrap());
    }

    (dir, segments, weighted_terms)
}

fn bench_multi_file_top_k(c: &mut Criterion) {
    let (_dir, mut segments, weighted_terms) = build_fixture();
    let mut group = c.benchmark_group("raw_file_topk");
    group.bench_function("multi_64", |b| {
        let mut segment_refs: Vec<_> = segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                top_k_weighted_u32_files(
                    black_box(segment_refs.as_mut_slice()),
                    black_box(weighted_terms.as_slice()),
                    black_box(10),
                )
                .unwrap(),
            );
        });
    });
    group.finish();
}

criterion_group!(benches, bench_multi_file_top_k);
criterion_main!(benches);
