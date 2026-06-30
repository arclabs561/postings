//! SPLADE-style weighted postings: build f32 term weights, then score a sparse query.
//!
//! Demonstrates `PostingsIndex<String, f32>`, where each term carries a continuous
//! weight (as produced by a learned sparse model such as SPLADE) instead of an
//! integer term frequency. Retrieval scores a sparse query vector against the
//! indexed documents by accumulating `query_weight * doc_weight` over the postings
//! lists -- the inner product SPLADE ranking uses.
//!
//! No dataset required: the document and query weights are synthetic.
//!
//! Run:
//! `cargo run -p postings --example splade_weighted`

use postings::PostingsIndex;

fn main() {
    // A tiny corpus of "documents", each already encoded by a learned sparse model
    // into (term, weight) pairs. Weights are nonnegative term activations.
    let docs: &[(u32, &[(&str, f32)])] = &[
        (
            0,
            &[
                ("neural", 1.8),
                ("network", 2.1),
                ("deep", 0.9),
                ("learning", 1.2),
            ],
        ),
        (1, &[("graph", 2.4), ("network", 1.1), ("node", 1.7)]),
        (
            2,
            &[
                ("neural", 0.7),
                ("search", 2.2),
                ("retrieval", 2.6),
                ("learning", 0.5),
            ],
        ),
        (
            3,
            &[
                ("retrieval", 1.9),
                ("sparse", 2.8),
                ("index", 1.3),
                ("search", 1.0),
            ],
        ),
    ];

    let mut index: PostingsIndex<String, f32> = PostingsIndex::new();
    for &(doc_id, terms) in docs {
        let weighted: Vec<(String, f32)> = terms.iter().map(|&(t, w)| (t.to_string(), w)).collect();
        index
            .add_weighted_document(doc_id, &weighted)
            .expect("doc ids are unique");
    }

    println!("indexed {} documents", index.num_docs());

    // A learned sparse *query*, also a (term, weight) vector. SPLADE expands a
    // query into weighted terms the same way it expands documents.
    let query: &[(&str, f32)] = &[("neural", 1.5), ("retrieval", 2.0), ("search", 1.0)];

    // Candidate generation: union of docs containing any query term. This is the
    // no-false-negative candidate set the index guarantees; a scorer never has to
    // touch a document outside it.
    let query_terms: Vec<String> = query.iter().map(|&(t, _)| t.to_string()).collect();
    let candidates = index.candidates(&query_terms);
    println!("candidates for query: {candidates:?}");

    // Score by sparse inner product:
    //   score(d) = sum_t query_weight(t) * doc_weight(t, d)
    let ranked = index.top_k_weighted(query, 10);

    println!("\nranking (sparse inner product):");
    for &(doc_id, score) in &ranked {
        println!("  doc {doc_id}: score {score:.2}");
    }

    // Spot-check the top document: `term_frequency` returns the stored f32 SPLADE
    // weight (not an integer count) for an f32-weighted index.
    let &(top_doc, _) = ranked.first().expect("query matched at least one doc");
    let neural_w = index.term_frequency(top_doc, "neural");
    println!(
        "\ntop doc {top_doc} stores weight {neural_w:.2} for term \"neural\" (df={})",
        index.df("neural"),
    );
}
