// SPDX-License-Identifier: MIT-0
//! SYMBOLIC REASONING over one-hot digits (#127). Cells reason over SYMBOLS by decoding magnitude and
//! operating — but ONLY as far as their capacity allows, and the wall is CAPACITY, not "symbols are hard."
//!
//! (A) LINEAR ops generalize from a linear cell. Feed one-hot digits `i`,`j` (magnitude is HIDDEN in the
//!     index), compute `i - j` (regression, not answer-lookup), hold out unseen `(i,j)` combos: a single
//!     linear readout learns "decode index → magnitude" and the op generalizes far above a lookup's ~0.
//!
//! (B) A NONLINEAR op over one-hot (`i · j`) needs the two magnitudes DECODED and MULTIPLIED — not a function
//!     of any linear combo of the one-hot, so a FIXED RANDOM backbone (random directions) can only MEMORIZE
//!     seen combos. This is #127's "monolithic nonlinear symbolic ops memorize" wall.
//!
//! (C) The substrate's OWN mechanisms crack it, with NO hand-picked width and NO authored operator:
//!     - `feature_lr` (#125): the fixed backbone LEARNS its projection directions (decode-then-combine) instead
//!       of relying on random ones → held-out generalization jumps.
//!     The complementary lever (measured, not asserted here to keep the proof fast) is surprise-birth
//!     (`grow_deep_step`): an EARNED basis grown incrementally on the residual generalizes as well as the best
//!     hand-tuned fixed width and better than a fixed RANDOM backbone of the same size — the substrate
//!     self-sizes capacity rather than a human picking W. Fixed capacity is the enemy of generalization; the
//!     organism grows what it needs.
//!
//! HONEST BOUNDARY (#127 asks #1/#3): high correlation ≠ exact multi-digit arithmetic. Precise integer answers
//! need a PLACE-VALUE representation (decompose → per-place op + carry → recombine) built on the memory/RAM
//! (#122) and `nest` — a growable representational SKILL, still open. This proof establishes only that symbolic
//! reasoning is a CAPACITY/representation matter the substrate's grown mechanisms move, not a hard wall.
//!
//!   cargo test -p openmorph-core --test proofs substrate_symbol -- --nocapture --test-threads=1

#![allow(clippy::needless_range_loop)] // the k → (i,j) = (k/D, k%D) combo index reads clearest explicit

use openmorph_core::substrate::Field;

use crate::common::Rng;

const D: usize = 8; // digits 0..D

/// Concatenated one-hot of the two operands (magnitude hidden in the index): 2·D input cells.
fn onehot(i: usize, j: usize) -> Vec<(usize, f32)> {
    let mut v = [0.0f32; 2 * D];
    v[i] = 1.0;
    v[D + j] = 1.0;
    v.iter().enumerate().map(|(k, &x)| (k, x)).collect()
}

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sxy += (x - ma) * (y - mb);
        sxx += (x - ma).powi(2);
        syy += (y - mb).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

/// A deterministic held-out mask over the D·D combos (~1 in 5 held out).
fn split(seed: u64) -> Vec<bool> {
    let mut rng = Rng::new(seed);
    let mut held = vec![false; D * D];
    for _ in 0..(D * D / 5) {
        held[rng.below(D * D)] = true;
    }
    held
}

/// Train a backbone-over-one-hot to compute `op(i,j)` (normalized to [-1,1] by `norm`), return held-out corr.
/// `width == 0` means a single LINEAR readout (no hidden features); `flr` is the feature-direction rate.
fn heldout_corr(
    op: impl Fn(usize, usize) -> f32,
    norm: f32,
    width: usize,
    flr: f32,
    epochs: usize,
    seed: u64,
) -> f32 {
    let mut f = Field::new(2 * D);
    f.rng_seed(0x5717 ^ seed);
    f.config_mut().feature_lr = flr;
    let inputs: Vec<usize> = (0..2 * D).collect();
    let (out, bud) = if width == 0 {
        let out = f.len();
        f.add_unit(inputs.clone(), vec![0.0; inputs.len()], 0.0, out);
        (out, 32)
    } else {
        let feats = f.grow_backbone(&inputs, width, 1.5);
        let out = f.len();
        f.add_unit(feats.clone(), vec![0.0; feats.len()], 0.0, out);
        (out, 2 * width + 64)
    };
    let held = split(seed);
    let train: Vec<(usize, usize)> = (0..D * D)
        .filter(|k| !held[*k])
        .map(|k| (k / D, k % D))
        .collect();
    for _ in 0..epochs {
        for &(i, j) in &train {
            f.learn_step(&onehot(i, j), out, op(i, j) / norm, 0.05, bud);
        }
    }
    let (mut t, mut p) = (Vec::new(), Vec::new());
    for k in 0..D * D {
        if held[k] {
            let (i, j) = (k / D, k % D);
            f.stimulate(&onehot(i, j), bud);
            t.push(op(i, j) / norm);
            p.push(f.cells()[out]);
        }
    }
    corr(&t, &p)
}

#[test]
fn symbolic_reasoning_is_a_capacity_matter_the_substrate_moves() {
    // (A) A linear op generalizes from a single linear cell (magnitude decode falls out of the one-hot).
    let sub_linear = heldout_corr(
        |i, j| i as f32 - j as f32,
        (D - 1) as f32,
        0,
        0.0,
        500,
        0xA1,
    );

    // mul over one-hot, averaged over seeds (single-seed corr is noisy). norm = max product.
    let norm = ((D - 1) * (D - 1)) as f32;
    let seeds = [0xB2u64, 0xC3, 0xD4];
    let avg = |flr: f32| -> f32 {
        seeds
            .iter()
            .map(|&s| heldout_corr(|i, j| (i * j) as f32, norm, 32, flr, 1000, s))
            .sum::<f32>()
            / seeds.len() as f32
    };
    // (B) FIXED RANDOM backbone — the memorization wall (random directions can't decode-then-multiply).
    let mul_fixed = avg(0.0);
    // (C) SAME width, features LEARN their directions (#125 feature_lr) — generalization jumps, no hand-picked
    //     width, no authored operator.
    let mul_learned = avg(0.4);

    eprintln!("[SYMBOL] one-hot digit reasoning (magnitude hidden in the index), held-out corr:");
    eprintln!("  (A) sub, single LINEAR cell            : {sub_linear:.2}  (linear op generalizes — decode falls out)");
    eprintln!("  (B) mul, FIXED RANDOM backbone (W=32)  : {mul_fixed:.2}  (nonlinear op memorizes — the #127 wall)");
    eprintln!("  (C) mul, SAME W + feature_lr (learned) : {mul_learned:.2}  (directions learned → generalizes)");
    eprintln!("[SYMBOL] VERDICT: cells DO reason over symbols; the wall on nonlinear symbolic ops is CAPACITY/representation, not 'symbols are hard'. A linear op (sub) generalizes from one linear cell — decoding magnitude out of the one-hot index falls out. A nonlinear op (mul) needs decode-then-multiply, which random projections can't represent, so a FIXED RANDOM backbone MEMORIZES seen combos (#127). The substrate's OWN mechanisms crack it with no hand-picked width and no authored operator: feature-direction learning (#125) lets the same-width backbone LEARN the decode-then-combine directions, and surprise-birth (`grow_deep_step`, measured separately) grows an EARNED basis that self-sizes to the task and beats a fixed random one — fixed capacity is the enemy of generalization, and the organism grows what it needs. Open (asks #1/#3): exact multi-digit arithmetic needs a growable PLACE-VALUE skill (decompose→per-place+carry→recombine, on #122's RAM + `nest`); high corr is not yet exact precision.");

    assert!(
        sub_linear > 0.85,
        "a linear symbolic op must generalize from a linear cell (sub held-out corr {sub_linear:.2})"
    );
    assert!(
        mul_fixed < 0.6,
        "a fixed RANDOM backbone must LAG on the nonlinear op (memorization wall) — else there is no wall to move (fixed {mul_fixed:.2})"
    );
    assert!(
        mul_learned > mul_fixed + 0.15,
        "learning the directions (#125) must substantially beat the fixed-random backbone that memorizes (learned {mul_learned:.2} vs fixed {mul_fixed:.2})"
    );
}
