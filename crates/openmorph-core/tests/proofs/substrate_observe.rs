// SPDX-License-Identifier: MIT-0
//! INTRINSIC PREDICTIVE CODING — `observe`: the organism learns to model its world from the raw
//! stream, TARGET-FREE. Where `substrate_predict` has the harness compute and supply `target = the
//! next observation` each step, `observe` supplies only the observation itself: reality is the target.
//! Learning is the mismatch between the organism's OWN standing prediction and what actually arrives.
//!
//! The world: two coupled bumps A, B (each a one-hot in {0,1,2}). A rotates; B's direction DEPENDS on
//! A (a conjunction) — so modelling B needs non-linearity, and the SAME `grow_backbone` supplies it.
//! No dynamics are authored; the organism grows the model by predicting and being corrected by reality.
//!
//!   cargo test -p openmorph-core --test proofs substrate_observe -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const K: usize = 64;

/// The (unauthored, hidden) world dynamics: A rotates +1; B shifts by a direction that depends on A.
fn trans(a: usize, b: usize) -> (usize, usize) {
    let na = (a + 1) % 3;
    let db = if a == 0 { 1 } else { 2 }; // +1 when A is at 0, else -1 — a conjunction of A×B
    (na, (b + db) % 3)
}

/// Encode (A, B) as 6 cells: A one-hot in 0..3, B one-hot in 3..6 (±1).
fn enc(a: usize, b: usize) -> Vec<f32> {
    (0..6)
        .map(|i| {
            let hot = if i < 3 { i == a } else { i - 3 == b };
            if hot {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}
fn enc_pairs(a: usize, b: usize) -> Vec<(usize, f32)> {
    enc(a, b).into_iter().enumerate().collect()
}
fn argmax3(cells: &[f32], base: usize) -> usize {
    (0..3)
        .max_by(|&i, &j| cells[base + i].partial_cmp(&cells[base + j]).unwrap())
        .unwrap()
}

fn build(backbone: bool, seed: u64) -> (Field, usize) {
    let mut f = Field::new(6);
    let inputs: Vec<usize> = (0..6).collect();
    let mut reads = inputs.clone();
    if backbone {
        f.rng_seed(seed ^ 0xB047);
        reads.extend(f.grow_backbone(&inputs, K, 2.5));
    }
    let pred = f.len(); // base of the 6 prediction cells
    for p in 0..6 {
        let n = reads.len();
        f.add_unit(reads.clone(), vec![0.0; n], 0.0, pred + p);
    }
    (f, pred)
}

/// Stream the world through `observe` — TARGET-FREE — then measure next-step accuracy over all states.
fn run(backbone: bool, seed: u64) -> f64 {
    let (mut f, pred) = build(backbone, seed);
    let mut rng = Rng::new(seed);
    let (mut a, mut b) = ((rng.u() % 3) as usize, (rng.u() % 3) as usize);
    f.stimulate(&enc_pairs(a, b), 300); // prime: the organism makes its first prediction
    for _ in 0..12000 {
        let (na, nb) = trans(a, b);
        // No target is computed here — we hand `observe` only what actually happened next.
        f.observe(0, pred, 6, &enc(na, nb), 0.03, 300);
        (a, b) = (na, nb);
    }
    let mut correct = 0;
    for a in 0..3 {
        for b in 0..3 {
            f.stimulate(&enc_pairs(a, b), 300);
            let (na, nb) = trans(a, b);
            if argmax3(f.cells(), pred) == na && argmax3(f.cells(), pred + 3) == nb {
                correct += 1;
            }
        }
    }
    correct as f64 / 9.0
}

#[test]
fn morph_models_its_world_target_free_via_observe() {
    let seeds = [0x0C, 0x1D, 0x2E, 0x3F, 0x4A, 0x5B, 0x6C, 0x7D];
    let n = seeds.len() as f64;
    let lin = seeds.iter().map(|&s| run(false, s)).sum::<f64>() / n;
    let world = seeds.iter().map(|&s| run(true, s)).sum::<f64>() / n;

    eprintln!(
        "[OBSERVE] intrinsic predictive coding — model a coupled-bump world from the raw stream, TARGET-FREE, mean over {} seeds:\n  LINEAR predictor       : {:.0}%\n  GROWN world-model      : {:.0}%  (chance ~11%)",
        seeds.len(),
        lin * 100.0,
        world * 100.0,
    );
    eprintln!("[OBSERVE] VERDICT (rc9 — prediction made INTRINSIC): `observe` supplies only the observation — no target argument; the organism's own standing prediction is the 'expected cell state' and reality is the correction. It learns a non-linear world model self-supervised (the coupled bump B needs the grown backbone), and returns the per-cell prediction error as the dense signal. This is the whole thesis in one call: perception, prediction, and the dense acting reward are one comparison — cell vs its own expectation, corrected by the world.");

    assert!(
        lin < 0.80,
        "a linear predictor cannot model the A×B conjunction (got {lin:.2})"
    );
    assert!(
        world >= 0.95,
        "observe should grow a target-free world model (got {world:.2})"
    );
}
