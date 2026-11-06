// SPDX-License-Identifier: MIT-0
//! CONTEXT-GATED ACTING — MorphLab's open wall (issue #92), CRACKED with a shared state backbone.
//!
//! The wall: grow-deep cracks non-linear SUPERVISED structure (XOR 52%→95%), but the SAME growth on
//! the noisy per-action REWARD channel stalled at ~50% (opposite-policy multiplexing). MorphLab's
//! diagnosis: growing hidden features PER action, gated by the exploration-noisy per-action reward,
//! corrupts the surprise that gates hidden-birth. Their candidate fix: grow the features on the
//! SHARED STATE backbone (stable, present every step), and learn each action's value on top.
//!
//! This test proves it. `grow_backbone` births a shared random-feature basis of the STATE (decoupled
//! from the reward); each action-value reads `[state.., backbone..]` and tunes with `learn_step` from
//! reward alone. Opposite policies (best action = context XOR state) then multiplex — feature
//! FORMATION is stable on the state, value CREDIT is learned per action. No authored gate.
//!
//!   cargo test -p openmorph-core --test proofs substrate_context -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// Opposite-policy multiplex: best action = context XOR state — a conjunction (not linearly
/// separable in the raw state), and the two contexts demand OPPOSITE policies.
fn best(ctx: usize, st: usize) -> usize {
    ctx ^ st
}
/// Encode a bit as ±1 (symmetric for the tanh features).
fn pm(b: usize) -> f32 {
    if b == 1 {
        1.0
    } else {
        -1.0
    }
}

const CTX: usize = 0;
const ST: usize = 1;

/// Build a morph whose two action-value cells read the raw state, optionally PLUS a shared random
/// backbone of the state. Returns (field, v0_cell, v1_cell).
fn build(backbone: bool, seed: u64) -> (Field, usize, usize) {
    let mut f = Field::new(2);
    let mut reads = vec![CTX, ST];
    if backbone {
        // Grow the shared non-linear basis of the STATE — ONCE, decoupled from the reward.
        f.rng_seed(seed ^ 0xF17E);
        let hidden = f.grow_backbone(&[CTX, ST], 48, 2.5); // the shared-basis width (the historical 48)
        reads.extend(hidden);
    }
    let (v0, v1) = (f.len(), f.len() + 1);
    let n = reads.len();
    f.add_unit(reads.clone(), vec![0.0; n], 0.0, v0);
    f.add_unit(reads, vec![0.0; n], 0.0, v1);
    (f, v0, v1)
}

/// ε-greedy contextual bandit; returns greedy accuracy over the 4 (ctx,st) cases.
fn run(backbone: bool, seed: u64) -> f64 {
    let (mut f, v0, v1) = build(backbone, seed);
    let mut rng = Rng::new(seed);
    for _ in 0..6000 {
        let (ctx, st) = ((rng.u() % 2) as usize, (rng.u() % 2) as usize);
        let inp = [(CTX, pm(ctx)), (ST, pm(st))];
        f.stimulate(&inp, 50);
        let greedy = usize::from(f.cells()[v1] > f.cells()[v0]);
        let a = if rng.u().is_multiple_of(4) {
            (rng.u() % 2) as usize // explore
        } else {
            greedy
        };
        let reward = if a == best(ctx, st) { 1.0 } else { -1.0 };
        f.learn_step(&inp, if a == 0 { v0 } else { v1 }, reward, 0.03, 50); // tune the taken action only
    }
    let mut correct = 0;
    for ctx in 0..2 {
        for st in 0..2 {
            f.stimulate(&[(CTX, pm(ctx)), (ST, pm(st))], 50);
            if usize::from(f.cells()[v1] > f.cells()[v0]) == best(ctx, st) {
                correct += 1;
            }
        }
    }
    correct as f64 / 4.0
}

#[test]
fn shared_state_backbone_gives_context_gated_acting() {
    let seeds = [0x0C, 0x1D, 0x2E, 0x3F, 0x4A, 0x5B, 0x6C, 0x7D];
    let n = seeds.len() as f64;
    let lin = seeds.iter().map(|&s| run(false, s)).sum::<f64>() / n;
    let back = seeds.iter().map(|&s| run(true, s)).sum::<f64>() / n;

    eprintln!(
        "[CONTEXT] opposite-policy multiplex (best = ctx XOR state), mean over {} seeds:\n  LINEAR value (raw state only): {:.0}%  (chance 50%)\n  SHARED STATE BACKBONE        : {:.0}%",
        seeds.len(),
        lin * 100.0,
        back * 100.0
    );
    eprintln!("[CONTEXT] VERDICT (MorphLab's open wall, CRACKED): a single additive value can't gate context×state, and growing features PER action off the noisy reward stalled. Growing them ONCE on the shared STATE backbone (grow_backbone) decouples feature FORMATION from the reward; each action LEARNS a linear readout over the same stable basis, so ONE morph multiplexes BOTH opposite policies. No authored gate — the fix is the shared backbone + the existing learn_step. This is grown transduction of the state applied to the value channel — the same moat, on reward.");

    assert!(
        lin < 0.80,
        "a linear value cannot gate context×state (got {lin:.2})"
    );
    assert!(
        back >= 0.95,
        "the shared state backbone should multiplex both opposite policies (got {back:.2})"
    );
}
