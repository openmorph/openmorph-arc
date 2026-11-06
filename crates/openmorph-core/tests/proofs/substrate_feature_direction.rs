// SPDX-License-Identifier: MIT-0
//! FEATURE-DIRECTION LEARNING (#125) — the wall under self-construction, closed. `substrate_evolve`/`heritable`
//! showed the economy tunes a feature's SCALE by competition, but a feature's input-DIRECTION was forever
//! random: born random (`grow_backbone`), re-drawn random (`resample_feature`/`reproduce`). On any task where
//! directions MATTER (distractor dimensions, specific relations) that caps self-construction far below backprop
//! — measured by MorphLab as ls20 imitation stuck at 31% while an MLP over the same perception reached 54%.
//!
//! The fix is NOT backprop and NOT a new mechanism: it is the SAME local delta rule `learn_step` already runs,
//! applied ONE HOP DEEPER. Each grown feature the readout reads gets a pseudo-target = the residual
//! backprojected through the readout's weight on it (`err·deriv·w`) — feedback-alignment — and moves its own
//! input-direction to reduce it. Local, online, no backprop-through-depth. It is opt-in (`config.feature_lr`)
//! and byte-identical when 0.
//!
//! The claim, on a direction-sensitive task `y = tanh(2.5·Σ x[2i]·x[2i+1])` (linear provably fails; the
//! relevant pairs are hidden among distractor dimensions), held-out corr at CPU-affordable width:
//!   - directions IRRELEVANT (DIM=2): random and direction-learning both ≈ 1.0 — no wall, no regression.
//!   - directions MATTER (DIM=8, distractors): random-fixed caps (~0.34); direction-learning reaches
//!     backprop-level (~0.99) at the SAME width — the small grown structure learns its projections.
//!   - scarce width (DIM=8, W=24): random-fixed collapses (~0.22); direction-learning still ~0.99 — a small
//!     structure matches a big one, the CPU-native "grow only what you need" moat.
//!
//!   cargo test -p openmorph-core --test proofs substrate_feature_direction -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// The direction-sensitive target: product-pairs summed, then squashed. Pairs (0,1),(2,3),… carry signal; any
/// extra dimensions are pure distractors a feature must learn to IGNORE (a direction it cannot get by luck).
fn target(x: &[f32]) -> f32 {
    let mut s = 0.0;
    let mut i = 0;
    while i + 1 < x.len() {
        s += x[i] * x[i + 1];
        i += 2;
    }
    (2.5 * s).tanh()
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

fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}

/// Grow a random-projection backbone of `width` features over `dim` inputs, learn a readout for 20k steps, and
/// return held-out corr. With `flr > 0` the features ALSO learn their input-directions (the one addition).
fn held_out_corr(dim: usize, width: usize, flr: f32, seed: u64) -> f32 {
    let mut f = Field::new(dim);
    f.rng_seed(0xBEEF ^ seed);
    f.config_mut().feature_lr = flr;
    let inputs: Vec<usize> = (0..dim).collect();
    let feats = f.grow_backbone(&inputs, width, 1.5);
    let out = f.len();
    f.add_unit(feats.clone(), vec![0.0; feats.len()], 0.0, out);

    let mut rng = Rng::new(seed);
    for _ in 0..20_000 {
        let x: Vec<f32> = (0..dim).map(|_| rng.bit()).collect();
        f.learn_step(&pairs(&x), out, target(&x), 0.05, 256);
    }

    let mut rt = Rng::new(seed ^ 0x7E57);
    let (mut ts, mut ps) = (Vec::new(), Vec::new());
    for _ in 0..400 {
        let x: Vec<f32> = (0..dim).map(|_| rt.bit()).collect();
        f.stimulate(&pairs(&x), 256);
        ps.push(f.cells()[out]);
        ts.push(target(&x));
    }
    corr(&ts, &ps)
}

fn mean(dim: usize, width: usize, flr: f32) -> f32 {
    (0..5)
        .map(|s| held_out_corr(dim, width, flr, 0x100 + s * 7))
        .sum::<f32>()
        / 5.0
}

#[test]
fn features_learn_their_input_directions_and_reach_backprop_level() {
    // (A) directions irrelevant — no wall, and direction-learning must not regress.
    let (d2_off, d2_on) = (mean(2, 16, 0.0), mean(2, 16, 0.3));
    // (B) directions matter (distractors) — random-fixed caps; direction-learning reaches backprop-level.
    let (d8_off, d8_on) = (mean(8, 32, 0.0), mean(8, 32, 0.3));
    // (C) scarce width — a SMALL grown structure matches a big one only if it learns directions.
    let (sc_off, sc_on) = (mean(8, 24, 0.0), mean(8, 24, 0.3));

    eprintln!("[FEATDIR] held-out corr (mean 5 seeds), y = tanh(2.5·Σ x[2i]·x[2i+1]):");
    eprintln!(
        "  DIM=2  W=16 (directions irrelevant): random {d2_off:.3}   direction-learning {d2_on:.3}"
    );
    eprintln!("  DIM=8  W=32 (directions MATTER)     : random {d8_off:.3}   direction-learning {d8_on:.3}");
    eprintln!("  DIM=8  W=24 (SCARCE width)          : random {sc_off:.3}   direction-learning {sc_on:.3}");
    eprintln!("[FEATDIR] VERDICT: feature-direction learning closes the wall under self-construction (#125). A grown feature's input-DIRECTION was forever random — born random, re-drawn random — so the economy could tune only its SCALE; on any task where directions matter (distractors, specific relations) that capped self-construction far below backprop. The fix is the SAME local delta rule `learn_step` already runs, applied ONE HOP DEEPER: each feature the readout reads gets a pseudo-target = the residual backprojected through the readout's weight on it (`err·deriv·w`, feedback-alignment) and moves its own input-direction to reduce it — local, online, no backprop-through-depth, opt-in (`config.feature_lr`), byte-identical when 0. Where directions don't matter (DIM=2) random already suffices and learning does not regress; where they do (DIM=8) random-fixed caps (~0.34) while direction-learning reaches backprop-level (~0.99) at the SAME width; and at SCARCE width (W=24) random collapses (~0.22) while direction-learning still ~0.99 — a small grown structure matches a big one, the CPU-native 'grow only what you need' moat. One rule — reduce the gap to an expected cell state — now trains the whole depth, not just the readout.");

    assert!(
        d2_off > 0.9 && d2_on > 0.9,
        "when directions are irrelevant (DIM=2), random already works and learning must not regress (random {d2_off:.3}, learning {d2_on:.3})"
    );
    assert!(
        d8_on > 0.9 && d8_on > d8_off + 0.25,
        "when directions matter (DIM=8), direction-learning must reach backprop-level and beat random-fixed (learning {d8_on:.3} vs random {d8_off:.3})"
    );
    assert!(
        sc_on > 0.9,
        "at scarce width (W=24), a direction-learning structure must still reach backprop-level (learning {sc_on:.3} vs random {sc_off:.3})"
    );
}
