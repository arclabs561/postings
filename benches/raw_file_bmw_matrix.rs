//! Block-Max WAND workloads with useful and deliberately unhelpful bounds.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use postings::raw::{write_u64_u32_segment, RawDocument, RawSegment, RawSegmentFile};

const DOCS: u32 = 30_000;

#[derive(Clone, Copy)]
enum Pruning {
    None,
    Moderate,
    Strong,
}

impl Pruning {
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Moderate => "moderate",
            Self::Strong => "strong",
        }
    }

    fn weight(self, doc_id: u32, term_id: u64) -> u32 {
        match self {
            Self::None => 1 + ((doc_id as u64 + term_id) % 7) as u32,
            Self::Moderate if doc_id < 1_000 => 10_000 + doc_id % 17,
            Self::Strong if doc_id < 10 => 1_000_000,
            Self::Moderate | Self::Strong => 1,
        }
    }
}

fn build_fixture(
    width: usize,
    pruning: Pruning,
    staggered: bool,
) -> (tempfile::TempDir, RawSegmentFile, Vec<(u64, f32)>) {
    let weighted_docs: Vec<Vec<_>> = (0..DOCS)
        .map(|doc_id| {
            (0..width as u64)
                .filter_map(|term_id| {
                    let present = !staggered
                        || (doc_id + term_id as u32 * 37) % (97 + term_id as u32 % 11) != 0;
                    present.then_some((term_id, pruning.weight(doc_id, term_id)))
                })
                .collect()
        })
        .collect();
    let docs: Vec<_> = weighted_docs
        .iter()
        .enumerate()
        .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms))
        .collect();
    let bytes = write_u64_u32_segment(&docs).unwrap();
    let exhaustive = RawSegment::open(&bytes).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bmw-matrix.raw");
    std::fs::write(&path, &bytes).unwrap();
    let mut file = RawSegmentFile::open(path).unwrap();
    let query: Vec<_> = (0..width as u64).map(|term_id| (term_id, 1.0)).collect();

    for k in [10, 100, 1_000] {
        assert_eq!(
            file.top_k_weighted_u32(&query, k).unwrap(),
            exhaustive.top_k_weighted_u32(&query, k).unwrap()
        );
    }
    (dir, file, query)
}

fn bench_bmw_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_file_bmw_matrix");
    group.sample_size(20);

    for staggered in [false, true] {
        for pruning in [Pruning::None, Pruning::Moderate, Pruning::Strong] {
            for width in [8usize, 16, 32] {
                let (_dir, mut file, query) = build_fixture(width, pruning, staggered);
                for k in [10usize, 100, 1_000] {
                    let case = format!(
                        "{}_{}_w{width}_k{k}",
                        if staggered { "staggered" } else { "equal" },
                        pruning.name()
                    );
                    group.bench_with_input(BenchmarkId::from_parameter(case), &k, |b, &k| {
                        b.iter(|| {
                            black_box(
                                file.top_k_weighted_u32(black_box(&query), black_box(k))
                                    .unwrap(),
                            );
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_bmw_matrix);
criterion_main!(benches);
