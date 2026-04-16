/// Criterion benchmarks for postings query paths.
///
/// Index shape: 50 000 documents, 10 000-term vocabulary, ~100 terms per doc
/// (Zipf-distributed term frequencies so common terms appear in many docs,
/// rare terms in only a few -- realistic IR workload).
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use postings::PostingsIndex;

// ---------------------------------------------------------------------------
// Zipf generator
// ---------------------------------------------------------------------------

/// Inverse-transform Zipf sampler (exponent s=1.0, N=vocab_size).
/// Returns a term index in [0, vocab_size).
fn zipf_sample(rng: &mut u64, vocab_size: usize, _s: f64) -> usize {
    // Xorshift64 for speed
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let u = (*rng as f64) / (u64::MAX as f64); // uniform (0,1)

    // Precomputed harmonic number is expensive for 10k; use approximate inverse-CDF.
    // For s=1: CDF(k) ≈ ln(k+1)/ln(N+1). Invert: k ≈ exp(u * ln(N+1)) - 1.
    let n = vocab_size as f64;
    let k = (u * (n + 1.0_f64).ln()).exp() - 1.0;
    (k as usize).min(vocab_size - 1)
}

// ---------------------------------------------------------------------------
// Index construction helpers
// ---------------------------------------------------------------------------

const N_DOCS: usize = 50_000;
const VOCAB_SIZE: usize = 10_000;
const TERMS_PER_DOC: usize = 100;

fn term_str(id: usize) -> String {
    format!("t{id:05}")
}

fn build_index() -> PostingsIndex<String> {
    let mut idx: PostingsIndex<String> = PostingsIndex::new();
    let mut rng: u64 = 0xdeadbeef_cafebabe;

    for doc_id in 0..N_DOCS as u32 {
        let terms: Vec<String> = (0..TERMS_PER_DOC)
            .map(|_| term_str(zipf_sample(&mut rng, VOCAB_SIZE, 1.0)))
            .collect();
        idx.add_document(doc_id, &terms).unwrap();
    }
    idx
}

/// Choose terms that are guaranteed to exist with a given minimum df.
/// Scans the vocabulary from the most-common end (low term ids = high Zipf rank = high df).
fn query_terms(idx: &PostingsIndex<String>, count: usize, min_df: u32) -> Vec<String> {
    (0..VOCAB_SIZE)
        .map(term_str)
        .filter(|t| idx.df(t.as_str()) >= min_df)
        .take(count)
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_insert_50k(c: &mut Criterion) {
    c.bench_function("insert_50k", |b| {
        b.iter(|| {
            black_box(build_index());
        });
    });
}

fn bench_conjunctive(c: &mut Criterion) {
    let idx = build_index();
    let mut group = c.benchmark_group("conjunctive");

    for n in [2usize, 5] {
        // Pick moderately selective terms (df >= 10) to get real intersection work.
        let terms = query_terms(&idx, n, 10);
        group.bench_with_input(BenchmarkId::new("terms", n), &terms, |b, terms| {
            b.iter(|| {
                black_box(idx.candidates_all_terms(black_box(terms.as_slice())));
            });
        });
    }
    group.finish();
}

fn bench_disjunctive(c: &mut Criterion) {
    let idx = build_index();
    let mut group = c.benchmark_group("disjunctive");

    for n in [2usize, 5] {
        let terms = query_terms(&idx, n, 10);
        group.bench_with_input(BenchmarkId::new("terms", n), &terms, |b, terms| {
            b.iter(|| {
                black_box(idx.candidates(black_box(terms.as_slice())));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_50k,
    bench_conjunctive,
    bench_disjunctive
);
criterion_main!(benches);
