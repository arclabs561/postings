//! Saved-index envelope: checksummed files, corruption detection, and
//! validated loading of unframed files from earlier releases.
#![cfg(feature = "persistence")]

use durability::{Directory, MemoryDirectory};
use postings::PostingsIndex;
use std::io::Read;
use std::sync::Arc;

fn index() -> PostingsIndex<String, u32> {
    let mut ix = PostingsIndex::new();
    let docs: &[(u32, &[&str])] = &[
        (1, &["a", "b", "c"]),
        (2, &["b", "c", "d", "d"]),
        (5, &["a", "e"]),
        (9, &["c", "e", "f", "a"]),
    ];
    for (id, terms) in docs {
        let terms: Vec<String> = terms.iter().map(|s| s.to_string()).collect();
        ix.add_document(*id, &terms).unwrap();
    }
    ix.delete_document(5);
    ix
}

fn saved(ix: &PostingsIndex<String, u32>) -> Vec<u8> {
    let dir: Arc<dyn Directory> = MemoryDirectory::arc();
    ix.save(&*dir, "ix").unwrap();
    let mut bytes = Vec::new();
    dir.open_file("ix")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

fn load(bytes: &[u8]) -> Result<PostingsIndex<String, u32>, Box<dyn std::error::Error>> {
    let dir: Arc<dyn Directory> = MemoryDirectory::arc();
    dir.atomic_write("ix", bytes).unwrap();
    PostingsIndex::load(&*dir, "ix")
}

fn query(ix: &mut PostingsIndex<String, u32>) -> Vec<u32> {
    let q: Vec<String> = ["a", "b", "c", "d", "e", "f", "zz"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut out = ix.candidates(&q);
    out.extend(ix.candidates_all_terms(&q));
    let weighted: Vec<(&String, f32)> = q.iter().map(|t| (t, 1.0)).collect();
    out.extend(
        ix.top_k_weighted(&weighted, 3)
            .into_iter()
            .map(|(id, _)| id),
    );
    let mut ids: Vec<u32> = ix.document_ids().collect();
    ids.sort_unstable();
    for id in &ids {
        for t in &q {
            out.push(ix.term_frequency(*id, t));
        }
    }
    if let Some(id) = ids.first() {
        ix.delete_document(*id);
    }
    let _ = ix.add_document(77, &q);
    out.extend(ix.candidates(&q));
    out
}

/// The envelope header is magic, version, payload length and CRC-32.
const HEADER: usize = 24;

#[test]
fn saved_index_is_framed_and_round_trips() {
    let mut ix = index();
    let bytes = saved(&ix);
    assert_eq!(&bytes[..8], b"POSTIDX\0");
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
    let mut loaded = load(&bytes).unwrap();
    assert_eq!(query(&mut loaded), query(&mut ix));
}

#[test]
fn every_single_byte_corruption_is_rejected() {
    let bytes = saved(&index());
    for offset in 0..bytes.len() {
        for value in [bytes[offset].wrapping_add(1), bytes[offset] ^ 0x80] {
            let mut corrupt = bytes.clone();
            corrupt[offset] = value;
            assert!(
                load(&corrupt).is_err(),
                "corruption at byte {offset} loaded"
            );
        }
    }
    assert!(
        load(&bytes[..bytes.len() - 1]).is_err(),
        "truncated file loaded"
    );
}

#[test]
fn unframed_legacy_index_still_loads() {
    let mut ix = index();
    let legacy = saved(&ix)[HEADER..].to_vec();
    let mut loaded = load(&legacy).unwrap();
    assert_eq!(query(&mut loaded), query(&mut ix));
}

/// Unframed files carry no checksum, so loading checks the invariants that
/// queries rely on: a corrupted legacy file fails to load or answers queries,
/// and never panics later.
#[test]
fn corrupted_legacy_index_is_rejected_or_usable_never_panics() {
    let legacy = saved(&index())[HEADER..].to_vec();
    let mut panics = Vec::new();
    for offset in 0..legacy.len() {
        for value in [
            legacy[offset].wrapping_add(1),
            legacy[offset].wrapping_sub(1),
            0x2f,
            0xff,
        ] {
            let mut corrupt = legacy.clone();
            corrupt[offset] = value;
            let outcome = std::panic::catch_unwind(|| {
                if let Ok(mut ix) = load(&corrupt) {
                    query(&mut ix);
                }
            });
            if outcome.is_err() {
                panics.push((offset, value));
            }
        }
    }
    assert!(panics.is_empty(), "panicking corruptions: {panics:?}");
}
