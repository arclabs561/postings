//! Durability integration test for `postings`.

use durability::checkpoint::CheckpointFile;
use durability::recordlog::RecordLogWriter;
use durability::replay::replay_postcard_since;
use durability::storage::MemoryDirectory;
use postings::{DocId, PostingsIndex};
use std::collections::BTreeMap;
use std::sync::Arc;

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum Event {
    AddDoc { doc_id: DocId, terms: Vec<u64> },
    DeleteDoc { doc_id: DocId },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Snapshot {
    last_event_id: u64,
    docs: Vec<(DocId, Vec<u64>)>,
}

#[test]
fn recovered_candidates_have_no_false_negatives() {
    let dir: Arc<dyn durability::Directory> = Arc::new(MemoryDirectory::new());
    let log_path = "log.bin";
    let ckpt_path = "ckpt.bin";

    let mut w = RecordLogWriter::new(dir.clone(), log_path);
    let ckpt = CheckpointFile::new(dir.clone());

    let mut live: BTreeMap<DocId, Vec<u64>> = BTreeMap::new();
    let mut next_event_id: u64 = 0;

    let push = |e: Event,
                w: &mut RecordLogWriter,
                live: &mut BTreeMap<DocId, Vec<u64>>,
                next_event_id: &mut u64| {
        *next_event_id += 1;
        w.append_postcard(&e).unwrap();
        match e {
            Event::AddDoc { doc_id, terms } => {
                live.insert(doc_id, terms);
            }
            Event::DeleteDoc { doc_id } => {
                live.remove(&doc_id);
            }
        }
    };

    push(
        Event::AddDoc {
            doc_id: 1,
            terms: vec![fnv1a64("hello"), fnv1a64("tokyo")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    );
    push(
        Event::AddDoc {
            doc_id: 9,
            terms: vec![fnv1a64("習近平"), fnv1a64("北京")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    );
    push(
        Event::AddDoc {
            doc_id: 2,
            terms: vec![fnv1a64("moscow")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    );
    push(
        Event::AddDoc {
            doc_id: 3,
            terms: vec![fnv1a64("hello"), fnv1a64("berlin")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    );

    // Checkpoint.
    ckpt.write_postcard(
        ckpt_path,
        next_event_id,
        &Snapshot {
            last_event_id: next_event_id,
            docs: live.iter().map(|(k, v)| (*k, v.clone())).collect(),
        },
    )
    .unwrap();

    // Suffix update.
    push(
        Event::DeleteDoc { doc_id: 1 },
        &mut w,
        &mut live,
        &mut next_event_id,
    );

    // Recover.
    let mut recovered_live: BTreeMap<DocId, Vec<u64>> = BTreeMap::new();
    let (last_event_id, s): (u64, Snapshot) = ckpt.read_postcard(ckpt_path).unwrap();
    // The header's last_applied_id is the authoritative suffix cutoff.
    let since = last_event_id;
    assert_eq!(s.last_event_id, last_event_id);
    for (doc_id, terms) in s.docs {
        recovered_live.insert(doc_id, terms);
    }
    replay_postcard_since(dir, log_path, since, |e: Event| {
        match e {
            Event::AddDoc { doc_id, terms } => {
                recovered_live.insert(doc_id, terms);
            }
            Event::DeleteDoc { doc_id } => {
                recovered_live.remove(&doc_id);
            }
        }
        Ok(())
    })
    .unwrap();

    // Rebuild index.
    let mut idx: PostingsIndex<u64> = PostingsIndex::new();
    for (doc_id, terms) in &recovered_live {
        idx.add_document(*doc_id, terms).unwrap();
    }

    let q = vec![fnv1a64("hello")];
    let cand_set: std::collections::HashSet<DocId> = idx.candidates(&q).into_iter().collect();
    for (doc_id, terms) in &recovered_live {
        if terms.iter().any(|t| *t == q[0]) {
            assert!(
                cand_set.contains(doc_id),
                "missing doc_id={doc_id} in candidates()"
            );
        }
    }
}
