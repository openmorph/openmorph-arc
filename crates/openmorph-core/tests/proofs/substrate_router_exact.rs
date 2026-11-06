// SPDX-License-Identifier: MIT-0
//! v2.1 · the CORE ROUTER, exact half (#64/#146) — CONTENT-ADDRESSED recall + DEDUP by behavioural signature.
//! The fractal `route` is the GROWN nearest-signature half (fuzzy, O(√M)); this is the EXACT half MorphLab's
//! harness `SkillRouter` also has: key each skill by its `fingerprint()` (a stable behavioural signature — a
//! copy matches exactly) and put them in an ordered map, giving O(log M) exact recall and behavioural DEDUP.
//! It composes the CORE `fingerprint` (grown) with a std `BTreeMap` — a usage protocol, not a new core faculty.
//!   (A) DEDUP — a library with duplicate skills collapses to the UNIQUE behaviours (the common sub-morph = 1 node).
//!   (B) EXACT RECALL — a known skill's signature → O(log M) lookup → its canonical node.
//!   (C) BEHAVIOURAL — the recalled node computes the queried function (dedup is by WHAT it does, not how it grew).
use openmorph_core::substrate::Field;
use std::collections::BTreeMap;

/// A distinct leaf skill: cell 0 = input, cell 1 = tanh(w · x). Its BEHAVIOUR is set by `w`.
fn skill(w: f32) -> Field {
    let mut g = Field::new(2);
    g.add_unit(vec![0], vec![w], 0.0, 1);
    g
}

/// A behavioural KEY: quantize the (deterministic) fingerprint so it is Ord/Eq for a BTreeMap. Two skills with
/// the same behaviour get the same key; different behaviours differ.
fn sig(f: &Field) -> Vec<i64> {
    f.fingerprint()
        .iter()
        .map(|&v| (v * 10_000.0).round() as i64)
        .collect()
}

#[test]
fn fingerprint_content_addresses_and_dedups_skills() {
    // A library of M=10 banked skills over K=4 distinct behaviours (weights), with duplicates interleaved.
    let library = [0.3f32, 0.7, 0.7, 1.1, 0.3, 1.5, 1.1, 0.7, 1.5, 0.3];
    let unique_weights = [0.3f32, 0.7, 1.1, 1.5]; // K = 4
    let skills: Vec<Field> = library.iter().map(|&w| skill(w)).collect();

    // ── (A) DEDUP — content-address by behavioural signature; identical behaviours collapse to one entry. ──
    let mut catalog: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
    for (i, s) in skills.iter().enumerate() {
        catalog.entry(sig(s)).or_insert(i); // first occurrence is the canonical node
    }
    eprintln!(
        "[EXACT] library of {} skills over {} distinct behaviours → catalog has {} unique entries",
        library.len(),
        unique_weights.len(),
        catalog.len()
    );
    assert_eq!(
        catalog.len(),
        unique_weights.len(),
        "dedup by fingerprint must collapse the {} skills to the {} UNIQUE behaviours, got {}",
        library.len(),
        unique_weights.len(),
        catalog.len()
    );

    // sanity: distinct behaviours really do get distinct signatures (no accidental collision)
    let distinct_sigs: std::collections::BTreeSet<Vec<i64>> =
        unique_weights.iter().map(|&w| sig(&skill(w))).collect();
    assert_eq!(
        distinct_sigs.len(),
        unique_weights.len(),
        "distinct behaviours must have distinct fingerprints (no collision)"
    );

    // ── (B) EXACT RECALL — a freshly-built copy of a known skill is recalled in O(log M) by its signature. ──
    for &w in &unique_weights {
        let query = skill(w); // a COPY (grown independently) — same behaviour, must match
        let hit = catalog.get(&sig(&query)).copied();
        assert!(
            hit.is_some(),
            "a known behaviour (w={w}) must be recalled exactly by its fingerprint"
        );
        let node = hit.unwrap();
        assert_eq!(
            library[node], w,
            "recall must return a node with the queried behaviour (w={w}, got node {node} = w{})",
            library[node]
        );

        // ── (C) BEHAVIOURAL — the recalled node computes the same function as the query. ──
        let mut recalled = skill(library[node]);
        for &x in &[-0.6f32, 0.4, 0.9] {
            recalled.stimulate(&[(0, x)], 8);
            assert!(
                (recalled.cells()[1] - (w * x).tanh()).abs() < 1e-6,
                "the recalled node must compute tanh({w}·x)"
            );
        }
    }

    // an UNKNOWN behaviour is a miss (not a false recall)
    assert!(
        !catalog.contains_key(&sig(&skill(0.95))),
        "an unseen behaviour must MISS the exact index (no false content-address)"
    );

    eprintln!("[EXACT] VERDICT: the EXACT half of the core router (#64/#146). Skills are content-addressed by their `fingerprint()` — a stable behavioural signature (a copy matches exactly). (A) a library of 10 skills over 4 behaviours DEDUPS to 4 nodes (the common sub-morph collapses to one — the marketplace/mycelium dedup). (B) a known behaviour is recalled in O(log M) by signature (an unseen one MISSES — no false address). (C) recall is by WHAT a skill does, not how it grew — the node computes the queried tanh(w·x). This composes the CORE grown `fingerprint` with a std ordered map — the exact twin of the fuzzy fractal `route`, together backing MorphLab's SkillRouter.");
}
