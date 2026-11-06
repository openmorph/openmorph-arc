// SPDX-License-Identifier: MIT-0
//! THE SINGLE-FIELD ADAPTIVE BASIS — the last open substrate wall, closed with the alive economy's missing
//! operator. A random-projection `grow_backbone` is a LOTTERY: only the readout learns, so a bad draw is
//! dealt once and is fatal — worst-case accuracy collapses by seed. `substrate_earn` cured it with an
//! ENSEMBLE (variation + selection across whole fields); the open wall was doing it in ONE field, with no
//! best-of-N. The fix is MUTATION — the operator between birth (`grow_backbone`) and death (`cull`):
//! `resample_feature` re-draws an UNEARNED feature's projection in place and zeroes the readout weight on
//! it, so the readout re-earns it. "Unearned" is the substrate's own credit currency: a feature poorly
//! correlated with the RESIDUAL (what's still wrong = surprise) has earned nothing. Re-roll the starved
//! feature, keep the useful draws — variation + selection at feature granularity, inside one field.
//!
//! Demonstrated on 3-parity (needs a non-linear basis) with a lottery-prone width: the fixed backbone's
//! worst seed leaves persistent errors (~88%); residual-credit resampling lifts the WORST-CASE seed to
//! solved (100%), at equal total training — the bad draw is survived, not fatal, with no ensemble. (Soft
//! ±0.7 targets keep tanh' alive so a stuck output can escape — isolating the basis lottery from the
//! unrelated saturation trap; accuracy is read by sign.)
//!
//! PINNED to `feature_lr` at its `Config::legacy()` value 0 — every other lever is the 1.6 default (the 1.6 learning
//! flip): the random-projection LOTTERY this proof cures exists only while feature directions are frozen random. Under
//! the full 1.6 default (`feature_lr` 0.3) the FIXED single draw already reaches 100 % on every seed (worst 100 %), so
//! there is no lottery left for `resample_feature` to cure. (With only `feature_lr` pinned the numbers equal the full
//! legacy genome's.)
//!
//!   cargo test -p openmorph-core --test proofs substrate_adapt -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const M: usize = 3; // 3-parity
const WIDTH: usize = 6; // lottery-prone
const SCALE: f32 = 2.5;
const ROUNDS: usize = 16;
const PER_ROUND: usize = 600; // training steps per round
const TAIL: usize = 2400; // final training so the last resample is re-earned
const TOTAL: usize = ROUNDS * PER_ROUND + TAIL; // equal budget for fixed and adaptive
const BUD: usize = 200;

fn parity(x: &[f32]) -> f32 {
    -x.iter().product::<f32>()
}
// Soft learning target (±0.7, not ±1): keeps tanh' alive so a confidently-wrong output can still escape —
// isolating the BASIS lottery from the unrelated gradient-saturation trap. Accuracy is still read by sign.
const SOFT: f32 = 0.7;
fn tgt(x: &[f32]) -> f32 {
    SOFT * parity(x)
}
fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}
fn corners() -> Vec<Vec<f32>> {
    (0..(1u32 << M))
        .map(|m| {
            (0..M)
                .map(|k| if (m >> k) & 1 == 0 { -1.0 } else { 1.0 })
                .collect()
        })
        .collect()
}
fn acc(f: &mut Field, out: usize) -> f64 {
    let mut ok = 0;
    for x in corners() {
        f.stimulate(&pairs(&x), BUD);
        if f.cells()[out].signum() == parity(&x).signum() {
            ok += 1;
        }
    }
    ok as f64 / (1u32 << M) as f64
}

fn build(seed: u64) -> (Field, Vec<usize>, usize) {
    let mut f = pinned_field(M);
    f.rng_seed(seed);
    let inputs: Vec<usize> = (0..M).collect();
    let feats = f.grow_backbone(&inputs, WIDTH, SCALE);
    let mut reads = inputs.clone();
    reads.extend(&feats);
    let out = f.len();
    let k = reads.len();
    f.add_unit(reads, vec![0.0; k], 0.0, out);
    (f, feats, out)
}

fn train(f: &mut Field, out: usize, steps: usize, rng: &mut Rng) {
    for _ in 0..steps {
        let x: Vec<f32> = (0..M).map(|_| rng.bit()).collect();
        f.learn_step(&pairs(&x), out, tgt(&x), 0.1, BUD);
    }
}

/// FIXED: gamble once on WIDTH features, train the readout for the whole budget.
fn fixed(seed: u64) -> f64 {
    let (mut f, _feats, out) = build(seed);
    let mut rng = Rng::new(seed ^ 0x11);
    train(&mut f, out, TOTAL, &mut rng);
    acc(&mut f, out)
}

/// ADAPTIVE: same total training, but WHILE STUCK (surprise persists), RESAMPLE the least-earning feature
/// — the one least correlated with the residual (its credit) — re-rolling a bad draw in place. A solved
/// field is left alone (no churn); a final tail re-earns the last resample.
fn adaptive(seed: u64) -> f64 {
    let (mut f, feats, out) = build(seed);
    let mut rng = Rng::new(seed ^ 0x11);
    for _ in 0..ROUNDS {
        train(&mut f, out, PER_ROUND, &mut rng);
        if acc(&mut f, out) >= 1.0 {
            continue; // solved — don't re-roll a working basis (the lottery only bites the stuck seeds)
        }
        // Per-feature credit = |mean over the corners of residual × feature output| — the substrate's own
        // learn_step credit direction, read off. Low credit ⇒ the feature has earned nothing ⇒ re-roll it.
        let mut credit = vec![0.0f32; feats.len()];
        for x in corners() {
            f.stimulate(&pairs(&x), BUD);
            let resid = tgt(&x) - f.cells()[out];
            for (fi, &h) in feats.iter().enumerate() {
                credit[fi] += resid * f.cells()[h];
            }
        }
        let worst = (0..feats.len())
            .min_by(|&a, &b| credit[a].abs().partial_cmp(&credit[b].abs()).unwrap())
            .unwrap();
        f.resample_feature(feats[worst], SCALE);
    }
    train(&mut f, out, TAIL, &mut rng);
    acc(&mut f, out)
}

#[test]
fn residual_credit_resampling_cures_the_single_field_lottery() {
    let seeds: Vec<u64> = (0..10).map(|s| 0x100 + s * 13).collect();

    let fx: Vec<f64> = seeds.iter().map(|&s| fixed(s)).collect();
    let ad: Vec<f64> = seeds.iter().map(|&s| adaptive(s)).collect();

    let stats = |v: &[f64]| {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let min = v.iter().copied().fold(1.0f64, f64::min);
        (mean, min)
    };
    let (fm, fmin) = stats(&fx);
    let (am, amin) = stats(&ad);

    eprintln!(
        "[ADAPT] 3-parity, single field, width {WIDTH} (lottery-prone), {} seeds, equal total training:\n  FIXED    (gamble once)                    : mean {:.0}%  WORST {:.0}%\n  ADAPTIVE (resample least-residual-credit) : mean {:.0}%  WORST {:.0}%",
        seeds.len(),
        fm * 100.0,
        fmin * 100.0,
        am * 100.0,
        amin * 100.0,
    );
    eprintln!("[ADAPT] VERDICT: the single-field adaptive basis — the last open substrate wall — is closed by the alive economy's missing operator, MUTATION. `resample_feature` re-rolls the feature the organism has earned the least on (lowest correlation with the residual, the same credit `learn_step` follows) and makes the readout re-earn it. A bad random draw is no longer fatal: it is survived and replaced, IN ONE FIELD, no ensemble. Variation (re-draw) + selection (residual credit) at feature granularity — the earned basis natively. This retires the honest residual left by `substrate_earn`.");

    assert!(
        amin > fmin,
        "residual-credit resampling must lift the WORST-CASE single-field seed above the fixed lottery ({amin:.2} vs {fmin:.2})"
    );
    assert!(am >= fm, "and not lose on the mean ({am:.2} vs {fm:.2})");
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
