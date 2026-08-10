//! Semantic parity between optimized query paths and direct document scans.
//!
//! The oracle owns the source documents and derives OR, AND, and weighted
//! sparse-inner-product results without consulting postings, frequencies, or
//! any other index output.

use postings::{DocId, PostingsIndex};
use std::collections::{BTreeMap, BTreeSet};

type TextDocs = BTreeMap<DocId, Vec<&'static str>>;

fn oracle_any(docs: &TextDocs, query: &[&str]) -> Vec<DocId> {
    let wanted: BTreeSet<&str> = query.iter().copied().collect();
    if wanted.is_empty() {
        return Vec::new();
    }
    docs.iter()
        .filter(|(_, terms)| terms.iter().any(|term| wanted.contains(term)))
        .map(|(&doc_id, _)| doc_id)
        .collect()
}

fn oracle_all(docs: &TextDocs, query: &[&str]) -> Vec<DocId> {
    let wanted: BTreeSet<&str> = query.iter().copied().collect();
    if wanted.is_empty() {
        return Vec::new();
    }
    docs.iter()
        .filter(|(_, terms)| wanted.iter().all(|term| terms.contains(term)))
        .map(|(&doc_id, _)| doc_id)
        .collect()
}

fn assert_text_query_parity(docs: &TextDocs, query: &[&str]) {
    let mut index = PostingsIndex::new();
    for (&doc_id, terms) in docs {
        let owned: Vec<String> = terms.iter().map(|term| (*term).to_owned()).collect();
        index.add_document(doc_id, &owned).unwrap();
    }
    let owned_query: Vec<String> = query.iter().map(|term| (*term).to_owned()).collect();

    assert_eq!(index.candidates(&owned_query), oracle_any(docs, query));
    assert_eq!(
        index.candidates_all_terms(&owned_query),
        oracle_all(docs, query)
    );
}

#[test]
fn candidate_paths_match_direct_document_scans() {
    let dense_docs: TextDocs = [
        (9, vec!["rust", "rust", "検索"]),
        (1, vec!["検索", "cafe\u{301}", "🦀"]),
        (6, vec!["index", "rust", "東京"]),
        (3, vec![]),
        (4, vec!["🦀", "index", "index"]),
    ]
    .into_iter()
    .collect();

    for query in [
        vec![],
        vec!["rust"],
        vec!["rust", "rust"],
        vec!["rust", "index"],
        vec!["検索", "🦀", "検索"],
        vec!["missing"],
        vec!["東京", "missing"],
    ] {
        assert_text_query_parity(&dense_docs, &query);
    }

    // Large, sparse ids force the sorted-merge OR path. The long query also
    // crosses the implementation's 32-term deduplication boundary.
    let sparse_docs: TextDocs = [
        (4_000_001, vec!["needle", "common"]),
        (17, vec!["common", "rare"]),
        (u32::MAX - 1, vec!["needle", "rare", "needle"]),
    ]
    .into_iter()
    .collect();
    let mut long_query: Vec<String> = (0..34).map(|i| format!("absent-{i}")).collect();
    long_query.extend(["needle".into(), "common".into(), "needle".into()]);
    let borrowed: Vec<&str> = long_query.iter().map(String::as_str).collect();
    assert_text_query_parity(&sparse_docs, &borrowed);
}

type WeightedDocs = BTreeMap<DocId, Vec<(&'static str, f32)>>;

fn oracle_top_k(docs: &WeightedDocs, query: &[(&str, f32)], k: usize) -> Vec<(DocId, f32)> {
    let mut query_weights: BTreeMap<&str, f32> = BTreeMap::new();
    for &(term, weight) in query {
        *query_weights.entry(term).or_default() += weight;
    }

    let mut ranked = Vec::new();
    for (&doc_id, terms) in docs {
        let mut document_weights: BTreeMap<&str, f32> = BTreeMap::new();
        for &(term, weight) in terms {
            *document_weights.entry(term).or_default() += weight;
        }
        let score = query_weights
            .iter()
            .map(|(term, query_weight)| {
                query_weight * document_weights.get(term).copied().unwrap_or_default()
            })
            .sum::<f32>();
        if score != 0.0 {
            ranked.push((doc_id, score));
        }
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

#[test]
fn weighted_top_k_matches_sparse_inner_product_oracle() {
    let mut docs: WeightedDocs = [
        (40, vec![("rust", 1.0), ("rust", 0.5), ("検索", 2.0)]),
        (2, vec![("rust", 2.0), ("🦀", 1.0)]),
        (9_000_000, vec![("検索", -1.0), ("🦀", 4.0)]),
        (7, vec![]),
        (11, vec![("rust", 2.0), ("🦀", 1.0)]),
    ]
    .into_iter()
    .collect();

    let mut index: PostingsIndex<String, f32> = PostingsIndex::new();
    for (&doc_id, terms) in &docs {
        let owned: Vec<(String, f32)> = terms
            .iter()
            .map(|&(term, weight)| (term.to_owned(), weight))
            .collect();
        index.add_weighted_document(doc_id, &owned).unwrap();
    }

    let queries = [
        vec![],
        vec![("rust", 1.0)],
        vec![("rust", 0.5), ("rust", 1.5)],
        vec![("検索", 2.0), ("🦀", -0.5)],
        vec![("missing", 3.0)],
        vec![("rust", 1.0), ("rust", -1.0)],
    ];
    for query in queries {
        let owned: Vec<(String, f32)> = query
            .iter()
            .map(|&(term, weight)| (term.to_owned(), weight))
            .collect();
        let borrowed: Vec<(&String, f32)> =
            owned.iter().map(|(term, weight)| (term, *weight)).collect();
        for k in [0, 1, 3, 20] {
            assert_eq!(
                index.top_k_weighted(&borrowed, k),
                oracle_top_k(&docs, &query, k),
                "query={query:?}, k={k}"
            );
        }
    }

    assert!(index.delete_document(2));
    docs.remove(&2);
    let query = [("rust", 1.0), ("🦀", 1.0)];
    let owned = [(String::from("rust"), 1.0), (String::from("🦀"), 1.0)];
    let borrowed: Vec<(&String, f32)> =
        owned.iter().map(|(term, weight)| (term, *weight)).collect();
    assert_eq!(
        index.top_k_weighted(&borrowed, 20),
        oracle_top_k(&docs, &query, 20)
    );
}
