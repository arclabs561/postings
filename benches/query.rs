/// Criterion benchmarks for postings query paths.
///
/// Index shape: 50 000 documents, 10 000-term vocabulary, ~100 terms per doc
/// (Zipf-distributed term frequencies so common terms appear in many docs,
/// rare terms in only a few -- realistic IR workload).
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(all(feature = "positional", feature = "raw-segment"))]
use postings::positional::raw::{
    near_match_terms_strs_segment_files, phrase_match_strs_segment_files,
    write_positional_segment_from_index, RawPositionalSegment, RawPositionalSegmentFile,
};
#[cfg(feature = "positional")]
use postings::positional::PosingsIndex;
#[cfg(feature = "raw-segment")]
use postings::raw::{
    top_k_weighted_u32_files, write_u64_u32_segment, write_u64_u32_segment_from_index_seekable_to,
    write_u64_u32_segment_from_iter, write_u64_u32_segment_from_iter_to,
    write_u64_u32_segment_from_term_postings, write_u64_u32_segment_from_term_postings_seekable_to,
    write_u64_u32_segment_from_term_postings_to, write_u64_u32_segment_sorted_from_iter,
    write_u64_u32_segment_sorted_from_iter_to, write_u64_u32_segment_to, RawDocument, RawSegment,
    RawSegmentFile, RawTermPostingList,
};
use postings::{CandidatePlan, PlannerConfig, PostingsIndex};
#[cfg(all(feature = "positional", feature = "raw-segment"))]
use std::io::Write;

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

#[cfg(feature = "raw-segment")]
type RawWeightedDocs = Vec<Vec<(u64, u32)>>;
#[cfg(feature = "positional")]
const POSITIONAL_DOCS: usize = 25_000;
#[cfg(feature = "positional")]
const POSITIONAL_TERMS_PER_DOC: usize = 128;
const SPARSE_DOCS: usize = 20_000;
const SPARSE_DOC_ID_STRIDE: u32 = 100;

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

fn build_weighted_index() -> PostingsIndex<String, f32> {
    let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
    let mut rng: u64 = 0xdeadbeef_cafebabe;

    for doc_id in 0..N_DOCS as u32 {
        let weighted: Vec<(String, f32)> = (0..TERMS_PER_DOC)
            .map(|position| {
                let term_id = zipf_sample(&mut rng, VOCAB_SIZE, 1.0);
                let weight = 1.0 + ((term_id % 17) as f32 * 0.01) + ((position % 5) as f32 * 0.001);
                (term_str(term_id), weight)
            })
            .collect();
        idx.add_weighted_document(doc_id, &weighted).unwrap();
    }
    idx
}

fn build_sparse_doc_id_weighted_index() -> PostingsIndex<String, f32> {
    let mut idx: PostingsIndex<String, f32> = PostingsIndex::new();
    let mut rng: u64 = 0xdeadbeef_cafebabe;

    for logical_doc_id in 0..SPARSE_DOCS as u32 {
        let weighted: Vec<(String, f32)> = (0..TERMS_PER_DOC)
            .map(|position| {
                let term_id = zipf_sample(&mut rng, VOCAB_SIZE, 1.0);
                let weight = 1.0 + ((term_id % 17) as f32 * 0.01) + ((position % 5) as f32 * 0.001);
                (term_str(term_id), weight)
            })
            .collect();
        idx.add_weighted_document(logical_doc_id * SPARSE_DOC_ID_STRIDE, &weighted)
            .unwrap();
    }
    idx
}

fn build_weighted_index_after_negative_delete() -> PostingsIndex<String, f32> {
    let mut idx = build_weighted_index();
    idx.add_weighted_document(
        N_DOCS as u32,
        &[(term_str(0), -1.0), (String::from("deleted"), 1.0)],
    )
    .unwrap();
    assert!(idx.delete_document(N_DOCS as u32));
    idx
}

#[cfg(feature = "raw-segment")]
fn build_raw_numeric_docs() -> (PostingsIndex<u64>, RawWeightedDocs) {
    let mut idx: PostingsIndex<u64> = PostingsIndex::new();
    let mut weighted_docs: Vec<Vec<(u64, u32)>> = Vec::with_capacity(N_DOCS);
    let mut rng: u64 = 0xdeadbeef_cafebabe;

    for doc_id in 0..N_DOCS as u32 {
        let mut expanded = Vec::with_capacity(TERMS_PER_DOC);
        let mut counts = std::collections::BTreeMap::new();
        for _ in 0..TERMS_PER_DOC {
            let term_id = zipf_sample(&mut rng, VOCAB_SIZE, 1.0) as u64;
            expanded.push(term_id);
            *counts.entry(term_id).or_insert(0u32) += 1;
        }
        idx.add_document(doc_id, &expanded).unwrap();
        weighted_docs.push(counts.into_iter().collect());
    }

    (idx, weighted_docs)
}

#[cfg(feature = "raw-segment")]
fn raw_segment_bytes_from_docs(weighted_docs: &[Vec<(u64, u32)>], start_doc_id: u32) -> Vec<u8> {
    let raw_docs: Vec<_> = weighted_docs
        .iter()
        .enumerate()
        .map(|(offset, terms)| RawDocument::new(start_doc_id + offset as u32, terms))
        .collect();
    write_u64_u32_segment(&raw_docs).unwrap()
}

#[cfg(feature = "raw-segment")]
fn build_raw_numeric_fixture() -> (PostingsIndex<u64>, RawWeightedDocs, Vec<u8>) {
    let (idx, weighted_docs) = build_raw_numeric_docs();
    let bytes = raw_segment_bytes_from_docs(&weighted_docs, 0);
    (idx, weighted_docs, bytes)
}

#[cfg(feature = "raw-segment")]
fn build_raw_prunable_top_k_fixture() -> (tempfile::TempDir, RawSegmentFile, Vec<(u64, f32)>) {
    const PRUNABLE_DOCS: u32 = 180_000;

    let weighted_docs: Vec<Vec<(u64, u32)>> = (0..PRUNABLE_DOCS)
        .map(|doc_id| {
            if doc_id < 128 {
                vec![(1, 2_000_000_000), (2, 1_000_000_000)]
            } else {
                vec![(1, 300_000_000), (2, 100_000_000)]
            }
        })
        .collect();
    let raw_docs: Vec<_> = weighted_docs
        .iter()
        .enumerate()
        .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms))
        .collect();
    let bytes = write_u64_u32_segment(&raw_docs).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prunable.raw");
    std::fs::write(&path, bytes).unwrap();
    let segment = RawSegmentFile::open(&path).unwrap();

    (dir, segment, vec![(1, 1.0), (2, 1.0)])
}

fn build_delete_index(n_docs: u32, common_terms: usize) -> PostingsIndex<String> {
    let mut idx: PostingsIndex<String> = PostingsIndex::new();
    for doc_id in 0..n_docs {
        let mut terms: Vec<String> = (0..common_terms)
            .map(|term| format!("delete_common_{term}"))
            .collect();
        terms.push(format!("delete_unique_{doc_id}"));
        idx.add_document(doc_id, &terms).unwrap();
    }
    idx
}

#[cfg(feature = "positional")]
fn build_positional_index() -> PosingsIndex {
    let mut idx = PosingsIndex::new();
    let mut rng: u64 = 0xdeadbeef_cafebabe;

    for doc_id in 0..POSITIONAL_DOCS as u32 {
        let mut terms: Vec<String> = (0..POSITIONAL_TERMS_PER_DOC)
            .map(|_| term_str(zipf_sample(&mut rng, VOCAB_SIZE, 1.0)))
            .collect();

        if doc_id % 10 == 0 {
            terms[16] = "anchor_alpha".to_string();
            terms[17] = "anchor_beta".to_string();
            terms[18] = "anchor_gamma".to_string();
        } else if doc_id % 10 == 1 {
            terms[16] = "anchor_alpha".to_string();
            terms[20] = "anchor_beta".to_string();
            terms[30] = "anchor_gamma".to_string();
        }
        terms[48] = "near_common".to_string();
        if doc_id % 20 == 0 {
            terms[50] = "near_rare".to_string();
        }

        idx.add_document(doc_id, &terms).unwrap();
    }
    idx
}

#[cfg(all(feature = "positional", feature = "raw-segment"))]
fn build_positional_segment_files(
    shards: u32,
) -> (Vec<tempfile::NamedTempFile>, Vec<RawPositionalSegmentFile>) {
    let mut indexes: Vec<PosingsIndex> = (0..shards).map(|_| PosingsIndex::new()).collect();
    let mut rng: u64 = 0xdeadbeef_cafebabe;

    for doc_id in 0..POSITIONAL_DOCS as u32 {
        let mut terms: Vec<String> = (0..POSITIONAL_TERMS_PER_DOC)
            .map(|_| term_str(zipf_sample(&mut rng, VOCAB_SIZE, 1.0)))
            .collect();

        if doc_id % 10 == 0 {
            terms[16] = "anchor_alpha".to_string();
            terms[17] = "anchor_beta".to_string();
            terms[18] = "anchor_gamma".to_string();
        } else if doc_id % 10 == 1 {
            terms[16] = "anchor_alpha".to_string();
            terms[20] = "anchor_beta".to_string();
            terms[30] = "anchor_gamma".to_string();
        }
        terms[48] = "near_common".to_string();
        if doc_id % 20 == 0 {
            terms[50] = "near_rare".to_string();
        }

        indexes[(doc_id % shards) as usize]
            .add_document(doc_id, &terms)
            .unwrap();
    }

    let mut files = Vec::with_capacity(shards as usize);
    let mut segments = Vec::with_capacity(shards as usize);
    for index in indexes {
        let bytes = write_positional_segment_from_index(&index).unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        segments.push(RawPositionalSegmentFile::open(file.path()).unwrap());
        files.push(file);
    }
    (files, segments)
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

fn weighted_query_terms(
    idx: &PostingsIndex<String, f32>,
    count: usize,
    min_df: u32,
) -> Vec<(String, f32)> {
    (0..VOCAB_SIZE)
        .map(term_str)
        .filter(|t| idx.df(t.as_str()) >= min_df)
        .take(count)
        .enumerate()
        .map(|(i, t)| (t, 1.0 + (i as f32 * 0.1)))
        .collect()
}

fn weighted_query_terms_in_df_range(
    idx: &PostingsIndex<String, f32>,
    count: usize,
    min_df: u32,
    max_df: u32,
) -> Vec<(String, f32)> {
    let terms: Vec<_> = (0..VOCAB_SIZE)
        .rev()
        .map(term_str)
        .filter(|t| {
            let df = idx.df(t.as_str());
            df >= min_df && df <= max_df
        })
        .take(count)
        .enumerate()
        .map(|(i, t)| (t, 1.0 + (i as f32 * 0.1)))
        .collect();
    assert_eq!(
        terms.len(),
        count,
        "benchmark fixture did not find {count} terms with df in [{min_df}, {max_df}]"
    );
    terms
}

#[cfg(feature = "raw-segment")]
fn numeric_query_terms(idx: &PostingsIndex<u64>, count: usize, min_df: u32) -> Vec<u64> {
    (0..VOCAB_SIZE as u64)
        .filter(|term| idx.df(term) >= min_df)
        .take(count)
        .collect()
}

fn query_from_weighted_terms(terms: &[(String, f32)]) -> Vec<(&str, f32)> {
    terms
        .iter()
        .map(|(term, weight)| (term.as_str(), *weight))
        .collect()
}

fn top_weighted_terms(terms: &[(String, f32)], keep: usize) -> Vec<(String, f32)> {
    let mut terms = terms.to_vec();
    terms.sort_by(|(left_term, left_weight), (right_term, right_weight)| {
        right_weight
            .total_cmp(left_weight)
            .then_with(|| left_term.cmp(right_term))
    });
    terms.truncate(keep);
    terms
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

fn bench_plan_candidates(c: &mut Criterion) {
    let idx = build_index();
    let cfg = PlannerConfig {
        max_candidate_ratio: f32::INFINITY,
        max_candidates: u32::MAX,
    };
    let mut group = c.benchmark_group("plan_candidates");

    for n in [2usize, 5] {
        let terms = query_terms(&idx, n, 10);
        group.bench_with_input(BenchmarkId::new("terms", n), &terms, |b, terms| {
            b.iter(
                || match idx.plan_candidates(black_box(terms.as_slice()), cfg) {
                    CandidatePlan::Candidates(candidates) => black_box(candidates.len()),
                    CandidatePlan::ScanAll => black_box(idx.num_docs() as usize),
                },
            );
        });
    }
    group.finish();
}

fn bench_weighted_top_k(c: &mut Criterion) {
    let idx = build_weighted_index();
    let mut group = c.benchmark_group("weighted_top_k");

    let common_term = weighted_query_terms(&idx, 1, 10);
    let common_query = query_from_weighted_terms(&common_term);
    group.bench_with_input(
        BenchmarkId::new("single_common_term", 1),
        &common_query,
        |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        },
    );

    let rare_single_term = weighted_query_terms_in_df_range(&idx, 1, 1, 64);
    let rare_single_query = query_from_weighted_terms(&rare_single_term);
    group.bench_with_input(
        BenchmarkId::new("single_rare_term", 1),
        &rare_single_query,
        |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        },
    );

    for n in [2usize, 5] {
        let terms = weighted_query_terms(&idx, n, 10);
        let query = query_from_weighted_terms(&terms);
        group.bench_with_input(BenchmarkId::new("terms", n), &query, |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        });
    }

    // Learned-sparse retrievers produce longer expanded queries than classical BM25.
    for n in [8usize, 16, 32] {
        let terms = weighted_query_terms(&idx, n, 10);
        let query = query_from_weighted_terms(&terms);
        group.bench_with_input(BenchmarkId::new("expanded_terms", n), &query, |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        });
    }

    let mut mixed_terms = weighted_query_terms(&idx, 16, 10);
    for (i, (_, weight)) in mixed_terms.iter_mut().enumerate() {
        if i % 4 == 0 {
            *weight = -*weight;
        }
    }
    let mixed_query = query_from_weighted_terms(&mixed_terms);
    group.bench_with_input(
        BenchmarkId::new("mixed_sign_terms", 16),
        &mixed_query,
        |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        },
    );

    let expanded_terms = weighted_query_terms(&idx, 32, 10);
    let masked_terms = top_weighted_terms(&expanded_terms, 8);
    let masked_query = query_from_weighted_terms(&masked_terms);
    group.bench_with_input(
        BenchmarkId::new("top_weight_terms_from_32", 8),
        &masked_query,
        |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        },
    );

    let rare_terms = weighted_query_terms_in_df_range(&idx, 8, 1, 64);
    let rare_query = query_from_weighted_terms(&rare_terms);
    group.bench_with_input(
        BenchmarkId::new("rare_terms", 8),
        &rare_query,
        |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        },
    );

    let deleted_negative_idx = build_weighted_index_after_negative_delete();
    let deleted_negative_query = [("t00000", 1.0), ("t00001", 1.0)];
    group.bench_with_input(
        BenchmarkId::new("after_negative_delete", 2),
        &deleted_negative_query.as_slice(),
        |b, query| {
            b.iter(|| {
                black_box(deleted_negative_idx.top_k_weighted(black_box(query), 10));
            });
        },
    );
    group.finish();
}

fn bench_weighted_top_k_sparse_doc_ids(c: &mut Criterion) {
    let idx = build_sparse_doc_id_weighted_index();
    let mut group = c.benchmark_group("weighted_top_k_sparse_doc_ids");

    for n in [5usize, 16] {
        let terms = weighted_query_terms(&idx, n, 10);
        let query = query_from_weighted_terms(&terms);
        group.bench_with_input(BenchmarkId::new("terms", n), &query, |b, query| {
            b.iter(|| {
                black_box(idx.top_k_weighted(black_box(query.as_slice()), 10));
            });
        });
    }
    group.finish();
}

fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete");
    for &(label, n_docs, common_terms, target_doc) in &[
        ("common_terms_5k", 5_000u32, 8usize, 2_500u32),
        ("common_terms_20k", 20_000u32, 8usize, 10_000u32),
    ] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || build_delete_index(n_docs, common_terms),
                |mut idx| {
                    black_box(idx.delete_document(black_box(target_doc)));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

#[cfg(feature = "raw-segment")]
fn bench_raw_segment_queries(c: &mut Criterion) {
    let (idx, weighted_docs, bytes) = build_raw_numeric_fixture();
    let segment = RawSegment::open(&bytes).unwrap();
    let writer_doc_count = 10_000;
    let writer_docs: Vec<_> = weighted_docs
        .iter()
        .take(writer_doc_count)
        .enumerate()
        .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms))
        .collect();
    let mut writer_index: PostingsIndex<u64, u32> = PostingsIndex::new();
    for (doc_id, terms) in weighted_docs.iter().take(writer_doc_count).enumerate() {
        writer_index
            .add_weighted_document(doc_id as u32, terms)
            .unwrap();
    }
    let writer_segment_len = write_u64_u32_segment(&writer_docs).unwrap().len();
    let mut writer_doc_lengths = Vec::with_capacity(writer_docs.len());
    let mut writer_term_storage: std::collections::BTreeMap<u64, Vec<(u32, u32)>> =
        std::collections::BTreeMap::new();
    for doc in &writer_docs {
        let doc_len: u32 = doc.terms().iter().map(|&(_, weight)| weight).sum();
        writer_doc_lengths.push((doc.doc_id(), doc_len));
        for &(term_id, weight) in doc.terms() {
            writer_term_storage
                .entry(term_id)
                .or_default()
                .push((doc.doc_id(), weight));
        }
    }
    let writer_term_postings: Vec<_> = writer_term_storage
        .iter()
        .map(|(&term_id, postings)| RawTermPostingList::new(term_id, postings))
        .collect();
    let raw_dir = tempfile::tempdir().unwrap();
    let raw_path = raw_dir.path().join("numeric.raw");
    std::fs::write(&raw_path, &bytes).unwrap();
    let mut file_segment = RawSegmentFile::open(&raw_path).unwrap();
    let terms = numeric_query_terms(&idx, 5, 10);
    let weighted_terms: Vec<(u64, f32)> = terms
        .iter()
        .enumerate()
        .map(|(i, &term)| (term, 1.0 + (i as f32 * 0.1)))
        .collect();
    let memory_weighted_terms: Vec<(&u64, f32)> = weighted_terms
        .iter()
        .map(|(term, weight)| (term, *weight))
        .collect();
    let common_term = terms[0];
    let chunk_size = N_DOCS.div_ceil(4);
    let mut multi_file_segments = Vec::new();
    for (chunk_index, chunk) in weighted_docs.chunks(chunk_size).enumerate() {
        let start_doc_id = (chunk_index * chunk_size) as u32;
        let path = raw_dir.path().join(format!("numeric-{chunk_index}.raw"));
        std::fs::write(&path, raw_segment_bytes_from_docs(chunk, start_doc_id)).unwrap();
        multi_file_segments.push(RawSegmentFile::open(path).unwrap());
    }
    let chunk_size_64 = N_DOCS.div_ceil(64);
    let mut multi_file_segments_64 = Vec::new();
    for (chunk_index, chunk) in weighted_docs.chunks(chunk_size_64).enumerate() {
        let start_doc_id = (chunk_index * chunk_size_64) as u32;
        let path = raw_dir.path().join(format!("numeric-64-{chunk_index}.raw"));
        std::fs::write(&path, raw_segment_bytes_from_docs(chunk, start_doc_id)).unwrap();
        multi_file_segments_64.push(RawSegmentFile::open(path).unwrap());
    }
    let mut group = c.benchmark_group("raw_segment");

    group.bench_function("write_10k_vec", |b| {
        b.iter(|| {
            black_box(
                write_u64_u32_segment(black_box(&writer_docs))
                    .unwrap()
                    .len(),
            );
        });
    });

    group.bench_function("write_10k_sink_vec", |b| {
        b.iter_batched(
            || Vec::with_capacity(writer_segment_len),
            |mut out| {
                write_u64_u32_segment_to(black_box(&writer_docs), &mut out).unwrap();
                black_box(out.len());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("write_10k_iter_vec", |b| {
        b.iter(|| {
            let docs = weighted_docs
                .iter()
                .take(writer_doc_count)
                .enumerate()
                .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms));
            black_box(
                write_u64_u32_segment_from_iter(black_box(docs))
                    .unwrap()
                    .len(),
            );
        });
    });

    group.bench_function("write_10k_iter_sink_vec", |b| {
        b.iter_batched(
            || Vec::with_capacity(writer_segment_len),
            |mut out| {
                let docs = weighted_docs
                    .iter()
                    .take(writer_doc_count)
                    .enumerate()
                    .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms));
                write_u64_u32_segment_from_iter_to(black_box(docs), &mut out).unwrap();
                black_box(out.len());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("write_10k_sorted_iter_vec", |b| {
        b.iter(|| {
            let docs = weighted_docs
                .iter()
                .take(writer_doc_count)
                .enumerate()
                .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms));
            black_box(
                write_u64_u32_segment_sorted_from_iter(black_box(docs))
                    .unwrap()
                    .len(),
            );
        });
    });

    group.bench_function("write_10k_sorted_iter_sink_vec", |b| {
        b.iter_batched(
            || Vec::with_capacity(writer_segment_len),
            |mut out| {
                let docs = weighted_docs
                    .iter()
                    .take(writer_doc_count)
                    .enumerate()
                    .map(|(doc_id, terms)| RawDocument::new(doc_id as u32, terms));
                write_u64_u32_segment_sorted_from_iter_to(black_box(docs), &mut out).unwrap();
                black_box(out.len());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("write_10k_term_postings_vec", |b| {
        b.iter(|| {
            black_box(
                write_u64_u32_segment_from_term_postings(
                    black_box(&writer_doc_lengths),
                    black_box(&writer_term_postings),
                )
                .unwrap()
                .len(),
            );
        });
    });

    group.bench_function("write_10k_term_postings_sink_vec", |b| {
        b.iter_batched(
            || Vec::with_capacity(writer_segment_len),
            |mut out| {
                write_u64_u32_segment_from_term_postings_to(
                    black_box(&writer_doc_lengths),
                    black_box(&writer_term_postings),
                    &mut out,
                )
                .unwrap();
                black_box(out.len());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("write_10k_term_postings_seekable_cursor", |b| {
        b.iter_batched(
            || std::io::Cursor::new(Vec::with_capacity(writer_segment_len)),
            |mut out| {
                write_u64_u32_segment_from_term_postings_seekable_to(
                    black_box(&writer_doc_lengths),
                    black_box(&writer_term_postings),
                    &mut out,
                )
                .unwrap();
                black_box(out.get_ref().len());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("write_10k_index_seekable_cursor", |b| {
        b.iter_batched(
            || std::io::Cursor::new(Vec::with_capacity(writer_segment_len)),
            |mut out| {
                write_u64_u32_segment_from_index_seekable_to(black_box(&writer_index), &mut out)
                    .unwrap();
                black_box(out.get_ref().len());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("write_10k_sink_discard", |b| {
        b.iter(|| {
            let mut out = std::io::sink();
            write_u64_u32_segment_to(black_box(&writer_docs), &mut out).unwrap();
        });
    });

    group.bench_function("open", |b| {
        b.iter(|| {
            black_box(
                RawSegment::open(black_box(bytes.as_slice()))
                    .unwrap()
                    .num_docs(),
            );
        });
    });

    group.bench_function("file_open", |b| {
        b.iter(|| {
            black_box(
                RawSegmentFile::open(black_box(&raw_path))
                    .unwrap()
                    .num_docs(),
            );
        });
    });

    group.bench_function("df_lookup", |b| {
        b.iter(|| {
            black_box(segment.df(black_box(common_term)).unwrap());
        });
    });

    group.bench_function("plan_candidates_5", |b| {
        b.iter(|| {
            black_box(
                segment
                    .plan_candidates(
                        black_box(terms.as_slice()),
                        black_box(PlannerConfig::default()),
                    )
                    .unwrap(),
            );
        });
    });

    group.bench_function("file_plan_candidates_5", |b| {
        b.iter(|| {
            black_box(
                file_segment
                    .plan_candidates(
                        black_box(terms.as_slice()),
                        black_box(PlannerConfig::default()),
                    )
                    .unwrap(),
            );
        });
    });

    group.bench_function("postings_decode_common", |b| {
        b.iter(|| {
            let postings = segment
                .postings(black_box(common_term))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            black_box(postings.len());
        });
    });

    group.bench_function("postings_decode_common_with_lens", |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            segment
                .for_each_posting_with_document_len(
                    black_box(common_term),
                    |doc_id, weight, len| {
                        checksum = checksum
                            .wrapping_add(doc_id as u64)
                            .wrapping_add(weight as u64)
                            .wrapping_add(len as u64);
                    },
                )
                .unwrap();
            black_box(checksum);
        });
    });

    group.bench_function("postings_decode_common_block_0", |b| {
        b.iter(|| {
            let postings = segment
                .posting_block_postings(black_box(common_term), black_box(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            black_box(postings.len());
        });
    });

    group.bench_function("file_postings_decode_common", |b| {
        b.iter(|| {
            let postings = file_segment.postings(black_box(common_term)).unwrap();
            black_box(postings.len());
        });
    });

    group.bench_function("file_postings_decode_common_with_lens", |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            file_segment
                .for_each_posting_with_document_len(
                    black_box(common_term),
                    |doc_id, weight, len| {
                        checksum = checksum
                            .wrapping_add(doc_id as u64)
                            .wrapping_add(weight as u64)
                            .wrapping_add(len as u64);
                    },
                )
                .unwrap();
            black_box(checksum);
        });
    });

    group.bench_function("file_postings_decode_common_block_0", |b| {
        b.iter(|| {
            let postings = file_segment
                .posting_block_postings(black_box(common_term), black_box(0))
                .unwrap();
            black_box(postings.len());
        });
    });

    group.bench_function("raw_candidates_all_terms_5", |b| {
        b.iter(|| {
            black_box(
                segment
                    .candidates_all_terms(black_box(terms.as_slice()))
                    .unwrap(),
            );
        });
    });

    group.bench_function("raw_candidates_any_terms_5", |b| {
        b.iter(|| {
            black_box(
                segment
                    .candidates_any_terms(black_box(terms.as_slice()))
                    .unwrap(),
            );
        });
    });

    group.bench_function("file_candidates_all_terms_5", |b| {
        b.iter(|| {
            black_box(
                file_segment
                    .candidates_all_terms(black_box(terms.as_slice()))
                    .unwrap(),
            );
        });
    });

    group.bench_function("file_candidates_any_terms_5", |b| {
        b.iter(|| {
            black_box(
                file_segment
                    .candidates_any_terms(black_box(terms.as_slice()))
                    .unwrap(),
            );
        });
    });

    group.bench_function("raw_top_k_weighted_5", |b| {
        b.iter(|| {
            black_box(
                segment
                    .top_k_weighted_u32(black_box(weighted_terms.as_slice()), black_box(10))
                    .unwrap(),
            );
        });
    });

    group.bench_function("file_top_k_weighted_5", |b| {
        b.iter(|| {
            black_box(
                file_segment
                    .top_k_weighted_u32(black_box(weighted_terms.as_slice()), black_box(10))
                    .unwrap(),
            );
        });
    });

    group.bench_function("file_top_k_weighted_5_multi_4", |b| {
        let mut segments: Vec<_> = multi_file_segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                top_k_weighted_u32_files(
                    black_box(segments.as_mut_slice()),
                    black_box(weighted_terms.as_slice()),
                    black_box(10),
                )
                .unwrap(),
            );
        });
    });

    group.bench_function("file_top_k_weighted_5_multi_64", |b| {
        let mut segments: Vec<_> = multi_file_segments_64.iter_mut().collect();
        b.iter(|| {
            black_box(
                top_k_weighted_u32_files(
                    black_box(segments.as_mut_slice()),
                    black_box(weighted_terms.as_slice()),
                    black_box(10),
                )
                .unwrap(),
            );
        });
    });

    group.bench_function("file_top_k_weighted_2_prunable_blocks", |b| {
        let (_dir, mut prunable_segment, prunable_terms) = build_raw_prunable_top_k_fixture();
        b.iter(|| {
            black_box(
                prunable_segment
                    .top_k_weighted_u32(black_box(prunable_terms.as_slice()), black_box(10))
                    .unwrap(),
            );
        });
    });

    group.bench_function("file_top_k_weighted_2_forced_exact_blocks", |b| {
        let (_dir, mut exact_segment, _) = build_raw_prunable_top_k_fixture();
        let exact_terms = [(1, 1.0), (2, -1.0)];
        b.iter(|| {
            black_box(
                exact_segment
                    .top_k_weighted_u32(black_box(exact_terms.as_slice()), black_box(10))
                    .unwrap(),
            );
        });
    });

    group.bench_function("in_memory_candidates_all_terms_5", |b| {
        b.iter(|| {
            black_box(idx.candidates_all_terms(black_box(terms.as_slice())));
        });
    });

    group.bench_function("in_memory_candidates_any_terms_5", |b| {
        b.iter(|| {
            black_box(idx.candidates(black_box(terms.as_slice())));
        });
    });

    group.bench_function("in_memory_top_k_weighted_5", |b| {
        b.iter(|| {
            black_box(
                idx.top_k_weighted(black_box(memory_weighted_terms.as_slice()), black_box(10)),
            );
        });
    });

    group.finish();
}

#[cfg(not(feature = "raw-segment"))]
fn bench_raw_segment_queries(_c: &mut Criterion) {}

#[cfg(feature = "positional")]
fn bench_positional_queries(c: &mut Criterion) {
    let idx = build_positional_index();
    let phrase = vec![
        "anchor_alpha".to_string(),
        "anchor_beta".to_string(),
        "anchor_gamma".to_string(),
    ];
    let mut group = c.benchmark_group("positional_queries");

    group.bench_function("phrase_3_terms", |b| {
        b.iter(|| {
            black_box(idx.phrase_match(black_box(phrase.as_slice())));
        });
    });
    group.bench_function("near_pair_window_4", |b| {
        b.iter(|| {
            black_box(idx.near_match(
                black_box("anchor_alpha"),
                black_box("anchor_beta"),
                black_box(4),
            ));
        });
    });
    group.bench_function("near_pair_skewed_window_4", |b| {
        b.iter(|| {
            black_box(idx.near_match(
                black_box("near_common"),
                black_box("near_rare"),
                black_box(4),
            ));
        });
    });
    group.bench_function("near_unordered_3_terms_window_16", |b| {
        b.iter(|| {
            black_box(idx.near_match_terms(
                black_box(phrase.as_slice()),
                black_box(16),
                black_box(false),
            ));
        });
    });
    group.bench_function("near_ordered_3_terms_window_16", |b| {
        b.iter(|| {
            black_box(idx.near_match_terms(
                black_box(phrase.as_slice()),
                black_box(16),
                black_box(true),
            ));
        });
    });
    group.finish();
}

#[cfg(not(feature = "positional"))]
fn bench_positional_queries(_c: &mut Criterion) {}

#[cfg(all(feature = "positional", feature = "raw-segment"))]
fn bench_raw_positional_queries(c: &mut Criterion) {
    let idx = build_positional_index();
    let bytes = write_positional_segment_from_index(&idx).unwrap();
    let segment = RawPositionalSegment::open(&bytes).unwrap();
    let phrase = ["anchor_alpha", "anchor_beta", "anchor_gamma"];
    let mut group = c.benchmark_group("raw_positional_queries");

    group.bench_function("phrase_3_terms", |b| {
        b.iter(|| {
            black_box(segment.phrase_match_strs(black_box(&phrase)).unwrap());
        });
    });
    group.bench_function("near_pair_window_4", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match(
                        black_box("anchor_alpha"),
                        black_box("anchor_beta"),
                        black_box(4),
                    )
                    .unwrap(),
            );
        });
    });
    group.bench_function("near_pair_skewed_window_4", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match(
                        black_box("near_common"),
                        black_box("near_rare"),
                        black_box(4),
                    )
                    .unwrap(),
            );
        });
    });
    group.bench_function("near_unordered_3_terms_window_16", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match_terms_strs(black_box(&phrase), black_box(16), black_box(false))
                    .unwrap(),
            );
        });
    });
    group.bench_function("near_ordered_3_terms_window_16", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match_terms_strs(black_box(&phrase), black_box(16), black_box(true))
                    .unwrap(),
            );
        });
    });
    group.finish();
}

#[cfg(not(all(feature = "positional", feature = "raw-segment")))]
fn bench_raw_positional_queries(_c: &mut Criterion) {}

#[cfg(all(feature = "positional", feature = "raw-segment"))]
fn bench_raw_positional_file_queries(c: &mut Criterion) {
    let idx = build_positional_index();
    let bytes = write_positional_segment_from_index(&idx).unwrap();
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    let mut segment = RawPositionalSegmentFile::open(file.path()).unwrap();
    let phrase = ["anchor_alpha", "anchor_beta", "anchor_gamma"];
    let mut group = c.benchmark_group("raw_positional_file_queries");

    group.bench_function("phrase_3_terms", |b| {
        b.iter(|| {
            black_box(segment.phrase_match_strs(black_box(&phrase)).unwrap());
        });
    });
    group.bench_function("near_pair_window_4", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match(
                        black_box("anchor_alpha"),
                        black_box("anchor_beta"),
                        black_box(4),
                    )
                    .unwrap(),
            );
        });
    });
    group.bench_function("near_pair_skewed_window_4", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match(
                        black_box("near_common"),
                        black_box("near_rare"),
                        black_box(4),
                    )
                    .unwrap(),
            );
        });
    });
    group.bench_function("near_unordered_3_terms_window_16", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match_terms_strs(black_box(&phrase), black_box(16), black_box(false))
                    .unwrap(),
            );
        });
    });
    group.bench_function("near_ordered_3_terms_window_16", |b| {
        b.iter(|| {
            black_box(
                segment
                    .near_match_terms_strs(black_box(&phrase), black_box(16), black_box(true))
                    .unwrap(),
            );
        });
    });
    group.finish();
}

#[cfg(not(all(feature = "positional", feature = "raw-segment")))]
fn bench_raw_positional_file_queries(_c: &mut Criterion) {}

#[cfg(all(feature = "positional", feature = "raw-segment"))]
fn bench_raw_positional_file_segment_queries(c: &mut Criterion) {
    let (_files, mut segments) = build_positional_segment_files(4);
    let phrase = ["anchor_alpha", "anchor_beta", "anchor_gamma"];
    let mut group = c.benchmark_group("raw_positional_file_segment_queries");

    group.bench_function("phrase_3_terms_multi_4", |b| {
        let mut segment_refs: Vec<_> = segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                phrase_match_strs_segment_files(
                    black_box(segment_refs.as_mut_slice()),
                    black_box(&phrase),
                )
                .unwrap(),
            );
        });
    });
    group.bench_function("near_unordered_3_terms_window_16_multi_4", |b| {
        let mut segment_refs: Vec<_> = segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                near_match_terms_strs_segment_files(
                    black_box(segment_refs.as_mut_slice()),
                    black_box(&phrase),
                    black_box(16),
                    black_box(false),
                )
                .unwrap(),
            );
        });
    });
    group.bench_function("near_ordered_3_terms_window_16_multi_4", |b| {
        let mut segment_refs: Vec<_> = segments.iter_mut().collect();
        b.iter(|| {
            black_box(
                near_match_terms_strs_segment_files(
                    black_box(segment_refs.as_mut_slice()),
                    black_box(&phrase),
                    black_box(16),
                    black_box(true),
                )
                .unwrap(),
            );
        });
    });
    group.finish();
}

#[cfg(not(all(feature = "positional", feature = "raw-segment")))]
fn bench_raw_positional_file_segment_queries(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_insert_50k,
    bench_conjunctive,
    bench_disjunctive,
    bench_plan_candidates,
    bench_weighted_top_k,
    bench_weighted_top_k_sparse_doc_ids,
    bench_delete,
    bench_raw_segment_queries,
    bench_positional_queries,
    bench_raw_positional_queries,
    bench_raw_positional_file_queries,
    bench_raw_positional_file_segment_queries
);
criterion_main!(benches);
