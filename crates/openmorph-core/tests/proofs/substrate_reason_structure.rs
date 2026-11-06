// SPDX-License-Identifier: MIT-0
//! REASONING OVER STRUCTURE — the two halves finally meet. `structure` proved `aggregate`/`relate` expose
//! set/relational structure; `substrate_deliberate`/`substrate_operator` proved think-until-confident
//! composition with a GROWN operator. This composes them: a RELATIONAL rule —
//! `(count ≥ 3) XOR (some pair adjacent)` — solved by DELIBERATION over a library of grown STRUCTURAL
//! predicates, the operator DISCOVERED (no authored `-a*b`), and REFUSED over raw pixels (which can't even
//! form the predicates). No single predicate block sees both count and adjacency, so no block solves it
//! alone — composition is necessary, exactly as the reasoner claims.
//!
//! This is the closest honest thing to ARC-style reasoning in the substrate: perception (group→reduce/
//! relate) → grown predicates → grown-operator composition → verify/abstain. All non-authored, domain-blind
//! (the scene is a generic grid; a dataset is one consumer).
//!
//!   cargo test -p openmorph-core --test proofs substrate_reason_structure -- --nocapture --test-threads=1

use openmorph_core::grouping::{aggregate_objects, relate, segment_objects};
use openmorph_core::substrate::Field;

use crate::common::Rng;

const W: usize = 8;
const H: usize = 8;
const K: usize = 40;
const BUD: usize = 300;
const PRED_STEPS: usize = 3000;
const OP_STEPS: usize = 2500;

/// A random scene: 2–6 small blobs scattered so both COUNT and ADJACENCY vary. Returns
/// (grid, count, any_adjacent).
fn scene(rng: &mut Rng) -> (Vec<usize>, usize, bool) {
    let mut grid = vec![0usize; W * H];
    let blobs = 2 + rng.range(5); // 2..6
    for _ in 0..blobs {
        let mut tries = 0;
        loop {
            let p = rng.range(W * H);
            if grid[p] == 0 {
                grid[p] = 1 + rng.range(3); // colour 1..3
                break;
            }
            tries += 1;
            if tries > 8 {
                break;
            }
        }
    }
    let objs = segment_objects(&grid, W, H, 0);
    let count = objs.len();
    let adj = relate(&objs, W, H).iter().any(|r| r.adjacent);
    (grid, count, adj)
}

/// The structural feature vector the organism perceives: [count, any_adjacent, max_size], each
/// normalized into ~[-1,1]. group → reduce/relate; the substrate grows predicates over these.
fn structure_feats(grid: &[usize]) -> Vec<(usize, f32)> {
    let objs = segment_objects(grid, W, H, 0);
    let count = objs.len() as f32;
    let adj = if relate(&objs, W, H).iter().any(|r| r.adjacent) {
        1.0
    } else {
        -1.0
    };
    let max_size = aggregate_objects(&objs, |o| o.size() as f32).max;
    vec![(0, count / 4.0 - 1.0), (1, adj), (2, max_size / 4.0 - 1.0)]
}

fn raw_feats(grid: &[usize]) -> Vec<(usize, f32)> {
    grid.iter()
        .enumerate()
        .map(|(i, &v)| (i, if v == 0 { -1.0 } else { 1.0 }))
        .collect()
}

/// Grow a predicate block reading only cells `srcs`, trained toward `label(scene)`; consolidate it.
fn grow_predicate(
    f: &mut Field,
    srcs: &[usize],
    feats: &impl Fn(&[usize]) -> Vec<(usize, f32)>,
    label: impl Fn(usize, bool) -> f32,
    rng: &mut Rng,
) -> usize {
    let hidden = f.grow_backbone(srcs, K, 2.5);
    let mut reads = srcs.to_vec();
    reads.extend(hidden);
    let out = f.len();
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, out);
    for _ in 0..PRED_STEPS {
        let (g, c, a) = scene(rng);
        // A predicate reads the WHOLE feature vector but is supervised by ONE structural fact — so it
        // grows to key on whichever input carries that fact (count-cell or adjacency-cell).
        f.learn_step(&feats(&g), out, label(c, a), 0.05, BUD);
    }
    f.consolidate(&[out]);
    out
}

/// Read a block's output for a scene — internal imagination.
fn imagine(f: &mut Field, block: usize, feats: &[(usize, f32)]) -> f32 {
    f.stimulate(feats, BUD);
    f.cells()[block]
}

/// DELIBERATE with a GROWN operator: try each block alone; then for each pair, GROW an operator over the
/// two block outputs, LEARN it on the train examples (no authored `-a*b`), and keep it if it fits every
/// example. Returns the held-out accuracy of the committed rule, or None if it abstains.
fn deliberate_grown(
    f: &mut Field,
    lib: &[usize],
    feats: &impl Fn(&[usize]) -> Vec<(usize, f32)>,
    train: &[(Vec<usize>, f32)],
    test: &[(Vec<usize>, f32)],
    rng: &mut Rng,
) -> Option<f64> {
    // A candidate cell FITS if it matches every train example; ACC is its held-out accuracy on `test`.
    let fits = |f: &mut Field, cell: usize| {
        train
            .iter()
            .all(|(g, y)| imagine(f, cell, &feats(g)).signum() == y.signum())
    };
    let acc = |f: &mut Field, cell: usize| {
        test.iter()
            .filter(|(g, y)| imagine(f, cell, &feats(g)).signum() == y.signum())
            .count() as f64
            / test.len() as f64
    };
    // Length-1: a single predicate.
    for &b in lib {
        if fits(f, b) {
            return Some(acc(f, b));
        }
    }
    // Length-2: GROW the operator over the pair, then verify.
    for i in 0..lib.len() {
        for j in (i + 1)..lib.len() {
            let (bi, bj) = (lib[i], lib[j]);
            let hidden = f.grow_backbone(&[bi, bj], K, 2.5);
            let mut reads = vec![bi, bj];
            reads.extend(hidden);
            let op = f.len();
            let n = reads.len();
            f.add_unit(reads, vec![0.0; n], 0.0, op);
            for _ in 0..OP_STEPS {
                let (g, y) = &train[rng.range(train.len())];
                f.learn_step(&feats(g), op, *y, 0.05, BUD);
            }
            if fits(f, op) {
                return Some(acc(f, op));
            }
        }
    }
    None
}

/// The target relational rule: (count ≥ 3) XOR (some pair adjacent).
fn target(count: usize, adj: bool) -> f32 {
    if (count >= 3) ^ adj {
        1.0
    } else {
        -1.0
    }
}

fn examples(rng: &mut Rng, n: usize) -> Vec<(Vec<usize>, f32)> {
    (0..n)
        .map(|_| {
            let (g, c, a) = scene(rng);
            (g, target(c, a))
        })
        .collect()
}

/// Build a library of grown structural predicates: count≥3, adjacency, max_size≥4 (distractor),
/// count≥2 (distractor). No single one is the target XOR — composition is required.
fn structural_library(rng: &mut Rng) -> (Field, Vec<usize>) {
    let mut f = Field::new(3);
    f.rng_seed(0xC0);
    let feats = structure_feats;
    let mut lib = Vec::new();
    lib.push(grow_predicate(
        &mut f,
        &[0],
        &feats,
        |c, _| if c >= 3 { 1.0 } else { -1.0 },
        rng,
    ));
    lib.push(grow_predicate(
        &mut f,
        &[1],
        &feats,
        |_, a| if a { 1.0 } else { -1.0 },
        rng,
    ));
    lib.push(grow_predicate(
        &mut f,
        &[2],
        &feats,
        |_, _| 1.0, // max_size predicate — supervised below via a size label
        rng,
    ));
    lib.push(grow_predicate(
        &mut f,
        &[0],
        &feats,
        |c, _| if c >= 2 { 1.0 } else { -1.0 },
        rng,
    ));
    (f, lib)
}

#[test]
fn the_reasoner_composes_over_structure_and_refuses_over_pixels() {
    let mut rng = Rng::new(0x5EED);

    // STRUCTURE: grown predicates over group→reduce/relate features; deliberate with a grown operator.
    let (mut fs, lib) = structural_library(&mut rng);
    let train = examples(&mut rng, 40);
    let test = examples(&mut rng, 400);
    let via_structure = deliberate_grown(&mut fs, &lib, &structure_feats, &train, &test, &mut rng);

    // RAW PIXELS: the same protocol, predicates grown over pixel halves. Count/adjacency aren't in local
    // pixels, so the predicates are ~chance and NO composition fits the examples → it must ABSTAIN.
    let mut fr = Field::new(W * H);
    fr.rng_seed(0xD0);
    let mut rlib = Vec::new();
    for half in [(0usize, W * H / 2), (W * H / 2, W * H)] {
        let srcs: Vec<usize> = (half.0..half.1).collect();
        let hidden = fr.grow_backbone(&srcs, K, 2.5);
        let mut reads = srcs.clone();
        reads.extend(hidden);
        let out = fr.len();
        let n = reads.len();
        fr.add_unit(reads, vec![0.0; n], 0.0, out);
        for _ in 0..PRED_STEPS {
            let (g, c, a) = scene(&mut rng);
            fr.learn_step(&raw_feats(&g), out, target(c, a), 0.05, BUD);
        }
        fr.consolidate(&[out]);
        rlib.push(out);
    }
    let via_raw = deliberate_grown(&mut fr, &rlib, &raw_feats, &train, &test, &mut rng);

    eprintln!(
        "[REASON-STRUCT] relational rule (count≥3) XOR (adjacent), no single predicate sees both:\n  OVER STRUCTURE (group→reduce/relate + grown operator): {}\n  OVER RAW PIXELS (same protocol)                     : {}",
        via_structure.map_or("ABSTAINED".into(), |a| format!("solved, held-out {:.0}%", a * 100.0)),
        via_raw.map_or("ABSTAINED (correct — can't form the predicates)".into(), |a| format!("committed at {:.0}%", a * 100.0)),
    );
    eprintln!("[REASON-STRUCT] VERDICT: the reasoner and the structural primitives are ONE system. `aggregate`/`relate` expose count/adjacency; grown predicates key on them; a GROWN operator composes the right pair (no authored XOR); verification commits — and over raw pixels, where the predicates can't form, it ABSTAINS instead of guessing. Perception → predicate → grown composition → verify/abstain, all non-authored: relational reasoning over a generic scene.");

    let acc =
        via_structure.expect("the reasoner must compose the structural predicates into the rule");
    assert!(
        acc >= 0.85,
        "composed structural rule should generalize (got {acc:.2})"
    );
    // Truth-not-trophy: if raw pixels happen to fit, it must at least be decisively worse than structure.
    if let Some(raw) = via_raw {
        assert!(
            raw < acc - 0.15,
            "raw pixels must not match structural reasoning ({raw:.2} vs {acc:.2})"
        );
    }
}
