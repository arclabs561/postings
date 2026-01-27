//! Demonstrate `postings` + `durability` generic primitives.
//!
//! - store update events in `durability::recordlog`
//! - store snapshots in `durability::checkpoint`
//! - recover by (checkpoint + replay suffix) and rebuild `PostingsIndex`
//!
//! Run:
//! `cargo run -p postings --example durable_roundtrip`

use durability::checkpoint::CheckpointFile;
use durability::recordlog::RecordLogWriter;
use durability::replay::replay_postcard_since;
use durability::storage::FsDirectory;
use postings::{CandidatePlan, PlannerConfig, PostingsIndex};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

// Deterministic term id mapping (example-only). Use a stable hash, not Rust's HashMap hasher.
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
    AddDoc { doc_id: u32, terms: Vec<u64> },
    DeleteDoc { doc_id: u32 },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Snapshot {
    last_event_id: u64,
    docs: Vec<(u32, Vec<u64>)>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let dir: Arc<dyn durability::Directory> = Arc::new(FsDirectory::new(tmp.path())?);

    let log_path = "log/updates.bin";
    let ckpt_path = "ckpt/snap.bin";

    let mut w = RecordLogWriter::new(dir.clone(), log_path);

    // Ground truth (what we believe is live).
    let mut live: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
    let mut next_event_id: u64 = 0;

    let push = |e: Event,
                w: &mut RecordLogWriter,
                live: &mut BTreeMap<u32, Vec<u64>>,
                next_event_id: &mut u64|
     -> Result<(), Box<dyn std::error::Error>> {
        *next_event_id += 1;
        w.append_postcard(&e)?;
        match e {
            Event::AddDoc { doc_id, terms } => {
                live.insert(doc_id, terms);
            }
            Event::DeleteDoc { doc_id } => {
                live.remove(&doc_id);
            }
        }
        Ok(())
    };

    // Some updates.
    push(
        Event::AddDoc {
            doc_id: 1,
            terms: vec![fnv1a64("hello"), fnv1a64("tokyo"), fnv1a64("tokyo")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    )?;
    // Unicode token hashing should be stable and non-panicking.
    push(
        Event::AddDoc {
            doc_id: 9,
            terms: vec![fnv1a64("習近平"), fnv1a64("北京")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    )?;
    push(
        Event::AddDoc {
            doc_id: 2,
            terms: vec![fnv1a64("moscow"), fnv1a64("hello")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    )?;
    push(
        Event::DeleteDoc { doc_id: 2 },
        &mut w,
        &mut live,
        &mut next_event_id,
    )?;

    // Checkpoint current view.
    let ckpt = CheckpointFile::new(dir.clone());
    ckpt.write_postcard(
        ckpt_path,
        next_event_id,
        &Snapshot {
            last_event_id: next_event_id,
            docs: live.iter().map(|(k, v)| (*k, v.clone())).collect(),
        },
    )?;

    // More updates after checkpoint.
    push(
        Event::AddDoc {
            doc_id: 3,
            terms: vec![fnv1a64("hello"), fnv1a64("berlin")],
        },
        &mut w,
        &mut live,
        &mut next_event_id,
    )?;

    // === Recovery path ===
    let mut recovered_live: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
    let mut since: u64 = 0;
    if dir.exists(ckpt_path) {
        let (last_event_id, s): (u64, Snapshot) = ckpt.read_postcard(ckpt_path)?;
        since = last_event_id;
        debug_assert_eq!(s.last_event_id, last_event_id);
        for (doc_id, terms) in s.docs {
            recovered_live.insert(doc_id, terms);
        }
    }

    replay_postcard_since(dir.clone(), log_path, since, |e: Event| {
        match e {
            Event::AddDoc { doc_id, terms } => {
                recovered_live.insert(doc_id, terms);
            }
            Event::DeleteDoc { doc_id } => {
                recovered_live.remove(&doc_id);
            }
        }
        Ok(())
    })?;

    // Rebuild postings index from recovered live docs.
    let mut idx: PostingsIndex<u64> = PostingsIndex::new();
    for (doc_id, terms) in &recovered_live {
        idx.add_document(*doc_id, terms).unwrap();
    }

    // Validate: candidates for "hello" cover all docs containing term.
    let q = vec![fnv1a64("hello")];
    let cands = idx.candidates(&q);
    let cand_set: BTreeSet<u32> = cands.into_iter().collect();
    for (doc_id, terms) in &recovered_live {
        if terms.iter().any(|t| *t == q[0]) {
            assert!(cand_set.contains(doc_id));
        }
    }

    // Also show planner behavior: with default thresholds this may bail out.
    let plan = idx.plan_candidates(&q, PlannerConfig::default());
    match plan {
        CandidatePlan::Candidates(xs) => {
            println!("plan=candidates count={}", xs.len());
        }
        CandidatePlan::ScanAll => {
            println!("plan=scan_all");
        }
    }

    // A more selective query term to demonstrate `Candidates`.
    let q2 = vec![fnv1a64("tokyo")];
    let plan2 = idx.plan_candidates(
        &q2,
        PlannerConfig {
            max_candidate_ratio: 1.0,
            max_candidates: 1_000_000,
        },
    );
    match plan2 {
        CandidatePlan::Candidates(xs) => println!("plan(tokyo)=candidates count={}", xs.len()),
        CandidatePlan::ScanAll => println!("plan(tokyo)=scan_all"),
    }

    println!(
        "live_docs={} recovered_docs={}",
        live.len(),
        recovered_live.len()
    );
    Ok(())
}
