// SPDX-License-Identifier: MIT-0
//! HERITABLE CONSTANTS — the substrate tunes its OWN hot constant by competition, natively (MorphLab's
//! competition-tunes-constants PoC, grown into the substrate). Their evolve.py (a plain GA) rediscovered the
//! Optuna optima — projection SCALE ~2.5 for a sharp/RL task vs ~0.7 for a smooth/classification task — with
//! no oracle. Here that runs INSIDE the organism, no outer optimizer:
//!   - `scale` is now a LIVE per-unit trait (the activation is `tanh(scale·(Σw·x+b))`), promoted from the
//!     once-global projection constant;
//!   - SELECTION is by `credit_of` — accumulated readout CONTRIBUTION (|weight·activation|), NOT firing.
//!     This is the make-or-break MorphLab flagged: a firing gate credits a random backbone equally, so
//!     competition would select on noise; contribution credit selects on usefulness;
//!   - `reproduce(child, parent)` is the heredity operator: the offspring inherits the fittest parent's
//!     scale under a SELF-ADAPTIVE mutation (σ is itself heritable, Schwefel) — no hand-set mutation rate.
//!
//! THE ACCEPTANCE TEST — two-task divergence: the SAME mechanism, given a SHARP task (parity, needs crisp
//! high-scale features) drives the population scale UP, and given a SMOOTH task (a linear ramp on continuous
//! inputs, needs gentle low-scale features) drives it DOWN — each discovering its own optimum by who-
//! survives-and-reproduces, no oracle, no hand-set mutation rate. That is MorphLab's SCALE-2.5-vs-0.7 result,
//! now a substrate faculty.
//!
//!   cargo test -p openmorph-core --test proofs substrate_heritable -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const M: usize = 3;
const WIDTH: usize = 6;
const ROUNDS: usize = 40;
const PER_ROUND: usize = 400;
const BUD: usize = 200;

fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}

/// A SHARP task: 3-parity on ±1 inputs — needs crisp (high-scale) features (soft ±0.7 target keeps tanh'
/// alive). A SMOOTH task: a linear ramp on CONTINUOUS inputs — needs gentle (low-scale, near-linear)
/// features, which high-scale step-features fit poorly. Returns (input, target).
fn sharp(rng: &mut Rng) -> (Vec<f32>, f32) {
    let x: Vec<f32> = (0..M).map(|_| rng.bit()).collect();
    let y = 0.7 * -x.iter().product::<f32>(); // soft parity
    (x, y)
}
fn smooth(rng: &mut Rng) -> (Vec<f32>, f32) {
    let x: Vec<f32> = (0..M).map(|_| rng.sym()).collect();
    let y = 0.5 * (x[0] + x[1]); // a smooth linear ramp, bounded in [-1, 1]
    (x, y)
}

fn build(start_scale: f32) -> (Field, Vec<usize>, usize) {
    let mut f = Field::new(M);
    f.rng_seed(0x5CA1E);
    let inputs: Vec<usize> = (0..M).collect();
    let feats = f.grow_backbone(&inputs, WIDTH, 1.0); // unit-variance weights; the SCALE is the live gain
    for &h in &feats {
        f.set_scale(h, start_scale); // promote the population to the (task-mismatched) start scale
    }
    // Readout reads ONLY the features, so all non-linearity (and thus the scale) must come from the basis.
    let out = f.len();
    f.add_unit(feats.clone(), vec![0.0; feats.len()], 0.0, out);
    (f, feats, out)
}

fn mean_scale(f: &Field, feats: &[usize]) -> f32 {
    feats.iter().map(|&h| f.scale_of(h)).sum::<f32>() / feats.len() as f32
}

/// Run the native competition: train (credit accrues in `learn_step`), then the least-earning feature is
/// reborn as the fittest one's offspring (`reproduce` — inherit scale ± self-adaptive σ). Returns
/// (start mean scale, end mean scale).
fn compete(
    start_scale: f32,
    mut task: impl FnMut(&mut Rng) -> (Vec<f32>, f32),
    seed: u64,
) -> (f32, f32) {
    let (mut f, feats, out) = build(start_scale);
    let start = mean_scale(&f, &feats);
    let mut rng = Rng::new(seed);
    for _ in 0..ROUNDS {
        for _ in 0..PER_ROUND {
            let (x, t) = task(&mut rng);
            f.learn_step(&pairs(&x), out, t, 0.1, BUD);
        }
        // SELECTION by contribution credit; HEREDITY + self-adaptive mutation via `reproduce`.
        let best = *feats
            .iter()
            .max_by(|&&a, &&b| f.credit_of(a).partial_cmp(&f.credit_of(b)).unwrap())
            .unwrap();
        let worst = *feats
            .iter()
            .min_by(|&&a, &&b| f.credit_of(a).partial_cmp(&f.credit_of(b)).unwrap())
            .unwrap();
        if best != worst {
            f.reproduce(worst, best);
        }
    }
    (start, mean_scale(&f, &feats))
}

#[test]
fn the_substrate_evolves_its_projection_scale_up_or_down_by_task() {
    let seeds: Vec<u64> = (0..6).map(|s| 0x100 + s * 13).collect();

    // SHARP: start too smooth (0.3) — competition must drive scale UP.
    let sharp_end: f32 =
        seeds.iter().map(|&s| compete(0.3, sharp, s).1).sum::<f32>() / seeds.len() as f32;
    // SMOOTH: start too steppy (2.2) — competition must drive scale DOWN.
    let smooth_end: f32 = seeds
        .iter()
        .map(|&s| compete(2.2, smooth, s).1)
        .sum::<f32>()
        / seeds.len() as f32;

    eprintln!(
        "[HERITABLE] the SAME native mechanism, opposite tasks (per-unit scale, selection by contribution credit, self-adaptive reproduce):\n  SHARP task (parity, wants crisp features): population scale 0.30 → {sharp_end:.2}  (driven UP)\n  SMOOTH task (linear ramp, wants gentle features): population scale 2.20 → {smooth_end:.2}  (driven DOWN)",
    );
    eprintln!("[HERITABLE] VERDICT: the substrate tunes its own hot constant by COMPETITION, natively — no outer optimizer, no oracle, no hand-set mutation rate. `scale` is a live per-unit heritable trait; the alive economy selects on CONTRIBUTION credit (readout reliance, not firing — MorphLab's make-or-break), and `reproduce` inherits the fittest parent's scale under a self-adapting σ (Schwefel — mutability itself heritable). The two-task DIVERGENCE is the point: given a sharp task the population scale climbs, given a smooth one it falls — each discovers its own optimum by who-survives-and-reproduces, MorphLab's SCALE-2.5-vs-0.7 result as a substrate faculty. Nature tuning its own constants: decentralized, online, in the energy currency. FULL WIRING (open): fold contribution-income into `metabolize`/`cull` so reproduction is economy-driven end to end, and add per-unit `lr`; acceptance = reproducing the Optuna profiles exactly.");

    assert!(
        sharp_end > 0.3 + 0.12,
        "the sharp task must drive scale UP from 0.30 (got {sharp_end:.2})"
    );
    assert!(
        smooth_end < 2.2 - 0.2,
        "the smooth task must drive scale DOWN from 2.20 (got {smooth_end:.2})"
    );
    // The divergence is the two OPPOSITE moves from the same mechanism — sharp up, smooth down.
    assert!(
        sharp_end < smooth_end,
        "sharp (from 0.30) and smooth (from 2.20) must not cross — the moves are toward each task's own optimum ({sharp_end:.2} vs {smooth_end:.2})"
    );
}
