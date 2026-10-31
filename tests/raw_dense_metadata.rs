//! A posting whose doc id is below the first document-metadata id has no
//! metadata row. The per-term path reports this as `InvalidLayout`; the
//! multi-term dense-scratch paths must report it too instead of indexing
//! their scratch arrays with a wrapped slot.
#![cfg(feature = "raw-segment")]

use postings::raw::{write_u64_u32_segment, RawDocument, RawSegment, RawSegmentFile};

/// A checksummed segment for docs 1, 4 and 9 whose first doc-metadata id is
/// rewritten from 1 to 2, with the doc-metadata CRC recomputed so every
/// checksum stays valid. Postings for terms 7 and 9 still name doc 1.
fn segment_missing_first_doc_metadata() -> Vec<u8> {
    let a = vec![(7u64, 3u32), (9, 1)];
    let b = vec![(7u64, 1u32)];
    let c = vec![(9u64, 5u32), (11, 2)];
    let docs = vec![
        RawDocument::new(1, &a),
        RawDocument::new(4, &b),
        RawDocument::new(9, &c),
    ];
    let mut bytes = write_u64_u32_segment(&docs).unwrap();
    let read_u64 =
        |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize;
    let doc_count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let doc_meta = read_u64(&bytes, 48);
    let postings_offset = read_u64(&bytes, 56);
    let footer = read_u64(&bytes, 64);
    let block_count = (postings_offset - (doc_meta + doc_count * 8)) / 24;
    let integrity = footer - (block_count * 4 + 16);
    assert_eq!(
        u32::from_le_bytes(bytes[doc_meta..doc_meta + 4].try_into().unwrap()),
        1
    );
    bytes[doc_meta..doc_meta + 4].copy_from_slice(&2u32.to_le_bytes());
    let crc = crc32fast::hash(&bytes[doc_meta..doc_meta + doc_count * 8]);
    bytes[integrity + 4..integrity + 8].copy_from_slice(&crc.to_le_bytes());
    bytes
}

#[test]
fn byte_reader_dense_paths_return_err() {
    let bytes = segment_missing_first_doc_metadata();
    let seg = RawSegment::open(&bytes).expect("all checksums are valid");
    assert!(seg
        .for_each_posting_with_document_len(7, |_, _, _| {})
        .is_err());
    assert!(seg.candidates_any_terms(&[7, 9]).is_err());
    assert!(seg.top_k_weighted_u32(&[(7, 1.0), (9, 1.0)], 10).is_err());
    assert!(seg.top_k_weighted_u32(&[(7, 1.0), (9, -1.0)], 10).is_err());
}

#[test]
fn file_reader_dense_paths_return_err() {
    let bytes = segment_missing_first_doc_metadata();
    let dir = std::env::temp_dir().join(format!(
        "postings-raw-dense-metadata-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("missing_meta.segment");
    std::fs::write(&path, &bytes).unwrap();

    let mut seg = RawSegmentFile::open(&path).expect("all checksums are valid");
    assert!(seg.candidates_any_terms(&[7, 9]).is_err());
    assert!(seg.top_k_weighted_u32(&[(7, 1.0), (9, 1.0)], 10).is_err());
    assert!(seg.top_k_weighted_u32(&[(7, 1.0), (9, -1.0)], 10).is_err());
    assert!(seg
        .top_k_weighted_u32_filtered(&[(7, 1.0), (9, 1.0)], 10, |_| true)
        .is_err());
    assert!(seg
        .top_k_weighted_u32_filtered(&[(7, 1.0), (9, -1.0)], 10, |_| true)
        .is_err());

    std::fs::remove_dir_all(&dir).unwrap();
}
