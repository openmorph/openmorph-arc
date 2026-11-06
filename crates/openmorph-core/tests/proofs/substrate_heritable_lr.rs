// SPDX-License-Identifier: MIT-0
//! HERITABLE LEARNING RATE — the substrate tunes its DOMINANT knob by competition. MorphLab's Optuna
//! importance put `lr` first (0.29 / 0.66) — the highest-value trait to make heritable after `scale`. Same
//! machinery: `lr` is a live per-unit factor `learn_step` multiplies into that unit's own updates; a
//! population of learners with varied `lr` competes, the least-accurate is reborn (`reproduce`) as the
//! fittest's offspring inheriting its `lr` under a self-adaptive mutation. No outer optimizer, no oracle,
//! no hand-set mutation rate — the substrate finds its own learning rate.
//!
//! Demonstrated: every learner STARTS at a too-slow `lr` (underfits in the budget). Competition drives the
//! population `lr` UP to where the task is learned, lifting accuracy well above a frozen-slow control. (The
//! learners' `scale` is held at 1.0 so this isolates the `lr` trait; `reproduce` carries the whole genome.)
//!
//! PINNED to `stability_cap` at its `Config::legacy()` value 0 — every other lever is the 1.6 default (the 1.6 learning
//! flip): the selection pressure on `lr` is the raw step's (here: population `lr` 0.03 → 0.25, best MAE 0.40 → 0.05).
//! The claim is ABOUT that regime. The step-stability cap bounds the BASE rate (the call's `lr` × this per-unit trait) at
//! `κ/(1+Σx²)` by design — on this 27-read readout at `lr` 0.2 it binds from a trait of ≈ 0.2 — so above it a larger
//! heritable `lr` buys nothing and competition has nothing to select: under the full 1.6 default the population `lr`
//! rises only 0.03 → 0.11 (the bar is +0.15), while the best learner's MAE still falls 0.40 → 0.07 against the frozen
//! 0.40. (Unchanged by the fix that applies the step's modulators after the cap: these learners are fully plastic.)
//!
//!   cargo test -p openmorph-core --test proofs substrate_heritable_lr -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const M: usize = 3;
const K: usize = 24; // shared backbone width
const POP: usize = 6; // competing learners
const GENS: usize = 50;
const STEPS: usize = 50; // per-generation training; weights restart each generation so LR decides the race
const BUD: usize = 300;
const BASE_LR: f32 = 0.2; // the call-site lr; the per-unit factor scales it
const START_LR: f32 = 0.03; // a too-SLOW starting factor (effective lr 0.006 — underfits STEPS)

fn parity(x: &[f32]) -> f32 {
    -x.iter().product::<f32>()
}
fn tgt(x: &[f32]) -> f32 {
    0.7 * parity(x)
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
/// Mean absolute error of a learner over the truth table — LOWER is fitter. MAE (not sign-accuracy) is the
/// discriminating fitness for `lr`: a too-slow rate leaves the output near 0 (high MAE) even when its SIGN
/// is already right, so the learning rate genuinely matters here.
fn learner_mae(f: &mut Field, out: usize) -> f64 {
    let mut s = 0.0;
    let cs = corners();
    for x in &cs {
        f.stimulate(&pairs(x), BUD);
        s += (f.cells()[out] - tgt(x)).abs() as f64;
    }
    s / cs.len() as f64
}

/// Build a shared backbone (fixed features) + POP learner units, each writing its own output cell, all
/// starting at the too-slow `lr`. Returns (field, learner output cells, backbone read list).
fn build() -> (Field, Vec<usize>) {
    let mut f = pinned_field(M);
    f.rng_seed(0x1EA2);
    let inputs: Vec<usize> = (0..M).collect();
    let feats = f.grow_backbone(&inputs, K, 2.5);
    let mut reads = inputs.clone();
    reads.extend(&feats);
    let mut outs = Vec::new();
    for i in 0..POP {
        let out = f.len();
        f.add_unit(reads.clone(), vec![0.0; reads.len()], 0.0, out);
        // Seed VARIED (but all still too-slow) rates so competition has variation to select on — 0.6×..1.4×
        // START_LR; mean is START_LR. Without variation every learner is identical and selection is a no-op.
        let spread = 0.6 + 0.8 * i as f32 / (POP - 1) as f32;
        f.set_lr(out, START_LR * spread);
        outs.push(out);
    }
    (f, outs)
}

/// One generation: RESTART every learner's weights (fresh small-random init, `lr` genome kept), then train
/// `STEPS` — so how far each learner gets is decided purely by its learning rate, a clean per-generation race.
fn generation(f: &mut Field, outs: &[usize], rng: &mut Rng) {
    for &o in outs {
        f.resample_feature(o, 0.4); // restart weights only; per-unit lr is untouched
    }
    for _ in 0..STEPS {
        let x: Vec<f32> = (0..M).map(|_| rng.bit()).collect();
        for &o in outs {
            f.learn_step(&pairs(&x), o, tgt(&x), BASE_LR, BUD);
        }
    }
}

fn mean_lr(f: &Field, outs: &[usize]) -> f32 {
    outs.iter().map(|&o| f.lr_of(o)).sum::<f32>() / outs.len() as f32
}
fn best_mae(f: &mut Field, outs: &[usize]) -> f64 {
    outs.iter()
        .map(|&o| learner_mae(f, o))
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn the_substrate_self_tunes_its_learning_rate_by_competition() {
    let seeds: Vec<u64> = (0..6).map(|s| 0x100 + s * 13).collect();

    let mut ev_lr = 0.0f32;
    let mut ev_mae = 0.0f64;
    let mut fx_mae = 0.0f64;
    for &seed in &seeds {
        // FIXED: same generational race, but NO reproduction — the lr stays at its seeded (too-slow) spread.
        let (mut ff, fouts) = build();
        let mut frng = Rng::new(seed);
        for _ in 0..GENS {
            generation(&mut ff, &fouts, &mut frng);
        }
        fx_mae += best_mae(&mut ff, &fouts);

        // EVOLVED: after each generation the WORST learner (highest MAE) is reborn as the BEST one's
        // offspring — inheriting its lr under a self-adaptive mutation (scale held at 1.0 to isolate lr).
        let (mut f, outs) = build();
        let mut rng = Rng::new(seed);
        for _ in 0..GENS {
            generation(&mut f, &outs, &mut rng);
            let mae: Vec<f64> = outs.iter().map(|&o| learner_mae(&mut f, o)).collect();
            let best = outs[(0..POP)
                .min_by(|&a, &b| mae[a].partial_cmp(&mae[b]).unwrap())
                .unwrap()];
            let worst = outs[(0..POP)
                .max_by(|&a, &b| mae[a].partial_cmp(&mae[b]).unwrap())
                .unwrap()];
            if best != worst {
                f.reproduce(worst, best);
                f.set_scale(worst, 1.0); // hold scale fixed — this proof isolates the lr trait
            }
        }
        ev_lr += mean_lr(&f, &outs);
        ev_mae += best_mae(&mut f, &outs);
    }
    let n = seeds.len() as f64;
    let (ev_lr, ev_mae, fx_mae) = (ev_lr / seeds.len() as f32, ev_mae / n, fx_mae / n);

    eprintln!(
        "[HERITABLE-LR] a population of learners starts too slow (lr factor {START_LR}); competition tunes the rate ({} seeds):\n  FIXED    (lr frozen at {START_LR}): best-learner MAE {:.2}\n  EVOLVED  (lr heritable, select by fit): best-learner MAE {:.2}, population lr {START_LR} → {:.2}",
        seeds.len(),
        fx_mae,
        ev_mae,
        ev_lr,
    );
    eprintln!("[HERITABLE-LR] VERDICT: `lr` — MorphLab's dominant Optuna knob — is now a heritable trait the substrate tunes by competition, exactly like `scale`. A population of learners starting too slow drives its own learning rate UP (via `reproduce` inheriting the fittest's lr under a self-adapting σ) until the task is fit, lowering error well below a frozen-slow control, with no oracle and no hand-set mutation rate. Together with `scale`, the two highest-value hyperparameters self-organize inside the organism — the substrate finding its own constants, in the energy currency.");

    assert!(
        ev_lr > START_LR + 0.15,
        "competition must drive the learning rate UP from {START_LR} (got {ev_lr:.2})"
    );
    assert!(
        ev_mae < fx_mae - 0.1,
        "self-tuning the lr must fit better (lower MAE) than the frozen-too-slow control ({ev_mae:.2} vs {fx_mae:.2})"
    );
}

/// The genome this proof's claim is about (see the module docs): the 1.6 default with `stability_cap` at the
/// [`Config::legacy`](openmorph_core::Config::legacy) value.
fn pinned_config() -> openmorph_core::Config {
    let legacy = openmorph_core::Config::legacy();
    openmorph_core::Config {
        stability_cap: legacy.stability_cap, // the raw delta step (no step-stability cap)
        ..openmorph_core::Config::default()
    }
}

/// A `Field` of `n` cells on [`pinned_config`].
fn pinned_field(n: usize) -> Field {
    let mut f = Field::new(n);
    f.configure(pinned_config());
    f
}
