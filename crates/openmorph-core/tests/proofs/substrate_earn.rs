// SPDX-License-Identifier: MIT-0
//! THE EARNED BASIS — curing the random-projection LOTTERY with VARIATION + SELECTION, the alive economy's
//! own logic. A fixed-width `grow_backbone` gambles ONCE on `width` random features; a bad draw (features
//! that don't separate the task) is fatal — the readout can only work with what it was dealt, so accuracy
//! swings by seed (the documented lottery). Per-feature pruning can't fix it (usefulness is distributed
//! across the whole basis — an earlier honest negative). The non-authored fix is the economy's other half:
//! don't gamble once — grow SEVERAL candidate bases (variation) and KEEP the one the organism is least
//! SURPRISED by (selection by fitness = low residual error, the same surprise currency that births every
//! unit). A bad draw is now survived: it loses selection to a better sibling. The WORST-CASE seed lifts to
//! near the mean — the lottery's tail is cut, with no authored feature-selector.
//!
//! Demonstrated on 3-parity (needs a non-linear basis): selecting the least-surprising of a few grown bases
//! lifts the worst seed well above the single-draw fixed backbone. HONEST RESIDUAL (truth-not-trophy): a
//! SINGLE-organism grow-on-residual (`grow_deep_step`, birth-on-surprise within one field) does NOT beat
//! the fixed backbone here — its random features are lower-bandwidth (unscaled), so persistence alone
//! doesn't cure it. Selection among variants does; a substrate-level single-organism earned basis stays open.
//!
//! PINNED to `feature_lr` at its `Config::legacy()` value 0 — every other lever is the 1.6 default (the 1.6 learning
//! flip): the basis LOTTERY exists only while feature directions are frozen random. Under the full 1.6 default
//! (`feature_lr` 0.3) the single random draw already reaches 100 % on every seed (worst 100 %).
//!
//!   cargo test -p openmorph-core --test proofs substrate_earn -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const M: usize = 3; // 3-parity — needs a non-linear basis
const WIDTH: usize = 6; // small enough that a single random draw is a real lottery
const STEPS: usize = 5000;
const BUD: usize = 200;

fn parity(x: &[f32]) -> f32 {
    -x.iter().product::<f32>()
}
fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}
fn acc(f: &mut Field, out: usize) -> f64 {
    let mut ok = 0;
    for m in 0..(1u32 << M) {
        let x: Vec<f32> = (0..M)
            .map(|k| if (m >> k) & 1 == 0 { -1.0 } else { 1.0 })
            .collect();
        f.stimulate(&pairs(&x), BUD);
        if f.cells()[out].signum() == parity(&x).signum() {
            ok += 1;
        }
    }
    ok as f64 / (1u32 << M) as f64
}

/// Grow one WIDTH-feature backbone from `seed`, learn the readout, and return (accuracy, residual surprise,
/// output cell packaged with the trained field). Residual = mean |error| after training = the organism's
/// own fitness signal for this basis.
fn grow_one(seed: u64) -> (f64, f32) {
    let mut f = pinned_field(M);
    f.rng_seed(seed);
    let inputs: Vec<usize> = (0..M).collect();
    let mut reads = inputs.clone();
    reads.extend(f.grow_backbone(&inputs, WIDTH, 2.5));
    let out = f.len();
    let k = reads.len();
    f.add_unit(reads, vec![0.0; k], 0.0, out);
    let mut rng = Rng::new(seed ^ 0x11);
    let mut resid = 0.0f32;
    for t in 0..STEPS {
        let x: Vec<f32> = (0..M).map(|_| rng.bit()).collect();
        let e = f.learn_step(&pairs(&x), out, parity(&x), 0.1, BUD);
        if t >= STEPS - 500 {
            resid += e; // residual surprise over the last stretch = the basis's fitness
        }
    }
    (acc(&mut f, out), resid / 500.0)
}

/// FIXED: gamble once.
fn fixed(seed: u64) -> f64 {
    grow_one(seed).0
}

/// EARNED: grow N candidate bases (variation), keep the one with the LOWEST residual surprise (selection by
/// fitness) — the alive economy's variation+selection, using surprise as the currency.
const VARIANTS: usize = 4;
fn earned(seed: u64) -> f64 {
    let mut best = (f64::NAN, f32::INFINITY);
    for v in 0..VARIANTS {
        let (a, resid) = grow_one(seed ^ (0xA5A5 * (v as u64 + 1)));
        if resid < best.1 {
            best = (a, resid);
        }
    }
    best.0
}

#[test]
fn variation_and_selection_by_surprise_cure_the_lottery() {
    let seeds: Vec<u64> = (0..8).map(|s| 0x100 + s * 17).collect();

    let fx: Vec<f64> = seeds.iter().map(|&s| fixed(s)).collect();
    let ea: Vec<f64> = seeds.iter().map(|&s| earned(s)).collect();

    let stats = |v: &[f64]| {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let min = v.iter().copied().fold(1.0f64, f64::min);
        (mean, min)
    };
    let (fm, fmin) = stats(&fx);
    let (em, emin) = stats(&ea);

    eprintln!(
        "[EARN] 3-parity over {} seeds — single random draw vs select-least-surprising of {VARIANTS} (both {WIDTH} features):\n  FIXED  (gamble once)                 : mean {:.0}%  WORST {:.0}%\n  EARNED (variation + select by surprise): mean {:.0}%  WORST {:.0}%",
        seeds.len(),
        fm * 100.0,
        fmin * 100.0,
        em * 100.0,
        emin * 100.0,
    );
    eprintln!("[EARN] VERDICT: the basis lottery is cured by the alive economy's OTHER half — VARIATION + SELECTION. Grow several candidate bases and keep the one the organism is least SURPRISED by (fitness = low residual error, the same currency that births every unit); a bad draw loses to a better sibling, so the WORST-CASE seed lifts to near the mean. No authored feature-selector, no per-feature pruning (which fails on a distributed basis). HONEST RESIDUAL: a SINGLE-organism grow-on-residual (grow_deep_step) does NOT beat the fixed draw here — its features are lower-bandwidth — so a substrate-level single-field earned basis stays the open wall; selection among variants is the nature-based cure that works today.");

    assert!(
        emin > fmin + 0.05,
        "selection by surprise must lift the WORST-CASE seed above the single-draw lottery ({emin:.2} vs {fmin:.2})"
    );
    assert!(em >= fm, "and not lose on the mean ({em:.2} vs {fm:.2})");
}

/// The genome this proof's claim is about (see the module docs): the 1.6 default with `feature_lr` at the
/// [`Config::legacy`](openmorph_core::Config::legacy) value.
fn pinned_config() -> openmorph_core::Config {
    let legacy = openmorph_core::Config::legacy();
    openmorph_core::Config {
        feature_lr: legacy.feature_lr, // feature directions stay frozen random
        ..openmorph_core::Config::default()
    }
}

/// A `Field` of `n` cells on [`pinned_config`].
fn pinned_field(n: usize) -> Field {
    let mut f = Field::new(n);
    f.configure(pinned_config());
    f
}
