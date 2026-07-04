//! File-backed raw impact segments.
//!
//! Raw segments are immutable files. A store or application manifest owns
//! publication, deletes, and compaction; `postings::raw` owns the checked file
//! format and query-time scoring over those files.
//!
//! Run:
//! `cargo run -p postings --features raw-segment --example raw_segment_file`

use std::fs::File;
use std::path::Path;

use postings::raw::{
    top_k_weighted_u32_files_with_stats, write_u64_u32_segment_sorted_from_iter_to, RawDocument,
    RawSegmentFile,
};

struct Doc {
    id: u32,
    title: &'static str,
    terms: &'static [(u64, u32)],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let first_generation = [
        doc(
            1,
            "learned sparse retrieval",
            &[(10, 240), (20, 75), (30, 20)],
        ),
        doc(2, "dense vector service", &[(10, 70), (40, 310), (50, 100)]),
    ];
    let second_generation = [
        doc(3, "impact score pruning", &[(20, 200), (30, 180), (60, 50)]),
        doc(4, "lexical bm25 baseline", &[(30, 45), (70, 220)]),
    ];

    let dir = tempfile::tempdir()?;
    let first_path = dir.path().join("gen-000.raw");
    let second_path = dir.path().join("gen-001.raw");
    write_segment(&first_path, &first_generation)?;
    write_segment(&second_path, &second_generation)?;

    let mut first = RawSegmentFile::open(&first_path)?;
    let mut second = RawSegmentFile::open(&second_path)?;
    let resident_metadata = first.resident_metadata_len() + second.resident_metadata_len();
    let posting_payload = first.posting_payload_len()? + second.posting_payload_len()?;

    let query = [(10, 1.2), (20, 0.7), (30, 2.0)];
    let result = {
        let mut segments = vec![&mut first, &mut second];
        top_k_weighted_u32_files_with_stats(&mut segments, &query, 3)?
    };

    assert_eq!(result.hits.first().map(|(doc_id, _)| *doc_id), Some(3));

    println!("resident metadata bytes: {resident_metadata}");
    println!("posting payload bytes:   {posting_payload}");
    println!(
        "segments: seen={}, scored={}, pruned={}",
        result.stats.segments_seen, result.stats.segments_scored, result.stats.segments_pruned
    );
    println!("top-k across raw files:");
    for (doc_id, score) in result.hits {
        println!(
            "  doc {doc_id}: {score:.1}  {}",
            title(doc_id, &first_generation, &second_generation)
        );
    }

    Ok(())
}

fn write_segment(path: &Path, docs: &[Doc]) -> Result<(), Box<dyn std::error::Error>> {
    let raw_docs: Vec<_> = docs
        .iter()
        .map(|doc| RawDocument::new(doc.id, doc.terms))
        .collect();
    let mut file = File::create(path)?;
    write_u64_u32_segment_sorted_from_iter_to(raw_docs.iter().copied(), &mut file)?;
    file.sync_all()?;
    Ok(())
}

fn doc(id: u32, title: &'static str, terms: &'static [(u64, u32)]) -> Doc {
    Doc { id, title, terms }
}

fn title(doc_id: u32, first_generation: &[Doc], second_generation: &[Doc]) -> &'static str {
    first_generation
        .iter()
        .chain(second_generation.iter())
        .find(|doc| doc.id == doc_id)
        .map_or("unknown", |doc| doc.title)
}
