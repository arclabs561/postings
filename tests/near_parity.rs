//! Randomized parity between the positional NEAR paths and a brute-force
//! oracle computed from the raw token sequences.
//!
//! `near_match` (two-term, rarest-term anchored) and `near_match_terms`
//! (specialized three-unique-term path plus the generic multiplicity-aware
//! path) are optimizations over one definition: unordered NEAR holds when
//! some window of `window` tokens covers every required occurrence, ordered
//! NEAR when a strictly increasing position sequence in term order spans at
//! most `window`. The oracle re-derives that definition directly from the
//! documents, so any divergence in a specialized path fails here.

use postings::positional::PositionalIndex;
use postings::DocId;

struct Lcg(u64);

impl Lcg {
    fn next_below(&mut self, bound: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) % bound
    }
}

const VOCAB: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

fn positions(doc: &[&str], term: &str) -> Vec<u32> {
    doc.iter()
        .enumerate()
        .filter(|(_, t)| **t == term)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Unordered oracle: some window of `window` tokens covers `count` distinct
/// occurrences of every required term.
fn oracle_unordered(doc: &[&str], terms: &[&str], window: u32) -> bool {
    let mut required: Vec<(&str, usize)> = Vec::new();
    for t in terms {
        match required.iter_mut().find(|(rt, _)| rt == t) {
            Some((_, c)) => *c += 1,
            None => required.push((t, 1)),
        }
    }
    (0..doc.len() as u32).any(|start| {
        required.iter().all(|(term, count)| {
            positions(doc, term)
                .iter()
                .filter(|&&p| p >= start && p - start <= window)
                .count()
                >= *count
        })
    })
}

/// Ordered oracle: a strictly increasing position sequence, one position per
/// term in the given order, spanning at most `window`.
fn oracle_ordered(doc: &[&str], terms: &[&str], window: u32) -> bool {
    fn extend(doc: &[&str], terms: &[&str], first: u32, prev: u32, window: u32) -> bool {
        let Some((term, rest)) = terms.split_first() else {
            return true;
        };
        positions(doc, term)
            .into_iter()
            .filter(|&p| p > prev && p - first <= window)
            .any(|p| extend(doc, rest, first, p, window))
    }
    let (first_term, rest) = terms.split_first().expect("at least one term");
    positions(doc, first_term)
        .into_iter()
        .any(|p| extend(doc, rest, p, p, window))
}

fn matching_docs(docs: &[Vec<&'static str>], pred: impl Fn(&[&str]) -> bool) -> Vec<DocId> {
    docs.iter()
        .enumerate()
        .filter(|(_, doc)| pred(doc))
        .map(|(i, _)| i as DocId)
        .collect()
}

#[test]
fn near_paths_match_brute_force_oracle() {
    let mut rng = Lcg(0x0dd5_beef);
    let mut docs: Vec<Vec<&'static str>> = Vec::new();
    let mut ix = PositionalIndex::new();
    for doc_id in 0..40u32 {
        let len = 4 + rng.next_below(20) as usize;
        let doc: Vec<&'static str> = (0..len)
            .map(|_| VOCAB[rng.next_below(VOCAB.len() as u32) as usize])
            .collect();
        let tokens: Vec<String> = doc.iter().map(|t| t.to_string()).collect();
        ix.add_document(doc_id, &tokens).unwrap();
        docs.push(doc);
    }

    // Query shapes chosen to hit each dispatch: the two-term anchored path
    // (near_match, including same-term), the specialized three-unique path,
    // and the generic multiplicity path (duplicates, four terms).
    let queries: Vec<Vec<&str>> = vec![
        vec!["a", "b"],
        vec!["c", "f"],
        vec!["a", "a"],
        vec!["a", "b", "c"],
        vec!["d", "f", "b"],
        vec!["e", "a", "c"],
        vec!["a", "b", "a"],
        vec!["b", "b", "c"],
        vec!["a", "b", "c", "d"],
        vec!["a", "a", "b", "b"],
    ];

    for window in [1u32, 2, 3, 5, 8] {
        for terms in &queries {
            let strings: Vec<String> = terms.iter().map(|t| t.to_string()).collect();

            let got = ix.near_match_terms(&strings, window, false);
            let want = matching_docs(&docs, |d| oracle_unordered(d, terms, window));
            assert_eq!(got, want, "unordered {terms:?} window={window}");

            let got = ix.near_match_terms(&strings, window, true);
            let want = matching_docs(&docs, |d| oracle_ordered(d, terms, window));
            assert_eq!(got, want, "ordered {terms:?} window={window}");

            if let [a, b] = terms.as_slice() {
                let got = ix.near_match(a, b, window);
                let want = matching_docs(&docs, |d| oracle_unordered(d, &[a, b], window));
                assert_eq!(got, want, "near_match {a} {b} window={window}");
            }
        }
    }
}
