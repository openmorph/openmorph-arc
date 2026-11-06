// SPDX-License-Identifier: MIT-0
//! CAUSAL GROUPING — surprise-reduction reward groups by CAUSAL help, `co_activate` groups by CORRELATION
//! (Gap 7 test #1, `docs/research/arc3-gaps-to-100.md`).
//!
//! THE HYPOTHESIS (Sébastien). If a cell rewards the cells that reduce its SURPRISE, then cells GROUP by predictive
//! help — and because surprise is measured on the RESIDUAL (what the cell cannot already predict), the reward is
//! CONDITIONAL: a cell only pays for information it did not already have. That is exactly what separates a CAUSAL
//! driver from a COMMON-CAUSE correlate. `co_activate` (fire-together) cannot make that distinction — it groups by
//! raw correlation, and a common cause makes two effects correlate MORE than a real driver correlates with its
//! effect.
//!
//! THE TEST (decisive, isolated). Signals: a common cause `Z`, an independent causal driver `A`.
//!   B = tanh(Z + A)   — B is driven by Z AND by A (A→B is a REAL causal link)
//!   C = tanh(Z + εC)  — C is driven by Z only
//!   D = tanh(Z + εD)  — D is driven by Z only  (C,D share the common cause; neither drives the other)
//! Two grouping signals per pair:
//!   * CO-ACTIVATE (correlation, fire-together): `corr(X,Y)`.
//!   * SURPRISE-REDUCTION REWARD (conditional): how much X reduces Y's prediction error GIVEN Y's other input Z —
//!     `err(Y | Z) − err(Y | Z, X)` (grown predictors, held-out).
//!
//! THE CLAIM. Surprise-reduction reward SEPARATES the causal link from the spurious one — reward(A→B) ≫ reward(C→D)
//! — while `co_activate` does the OPPOSITE, scoring the spurious common-cause pair C,D at least as high as the
//! causal pair A,B (the two effects of a common cause correlate more than a driver correlates with its effect). So
//! grouping by surprise-reduction is CAUSAL where co-firing is merely correlational.
//!
//! TRUTH-NOT-TROPHY. The generative model, the seeded PRNG, and the held-out split are authored and identical for
//! both grouping signals; only the signal differs. This is the FIRST of Gap 7's three earn-it tests (grouping win →
//! stabilised co-adaptation → deletion) before any gated core spike. ZERO openmorph-core change: a usage protocol
//! over `grow`/`stimulate`/`cells` (the grown predictor) — surprise-reduction is the error the predictor reports.
//!
//!   cargo test -p openmorph-core --test proofs substrate_causal_grouping -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// One sample of the generative model: `(z, a, b, c, d)`.
fn sample(rng: &mut Rng) -> (f32, f32, f32, f32, f32) {
    let z = rng.sym();
    let a = rng.sym(); // independent causal driver
    let b = (1.5 * z + 1.5 * a).tanh(); // B ← Z AND A  (A→B is the real causal link)
    let c = (1.5 * z + 0.3 * rng.sym()).tanh(); // C ← Z only
    let d = (1.5 * z + 0.3 * rng.sym()).tanh(); // D ← Z only  (C,D: common cause, no direct link)
    (z, a, b, c, d)
}

fn dataset(seed: u64, n: usize) -> Vec<(f32, f32, f32, f32, f32)> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| sample(&mut rng)).collect()
}

/// Held-out MAE of a GROWN predictor of `target` from `inputs` (each row → the input values / the target). Grows a
/// non-linear predictor on the train split and reports mean-absolute-error on the held-out split.
fn fit_mae(
    data: &[(f32, f32, f32, f32, f32)],
    inputs: impl Fn(&(f32, f32, f32, f32, f32)) -> Vec<f32>,
    target: impl Fn(&(f32, f32, f32, f32, f32)) -> f32,
) -> f32 {
    let n_in = inputs(&data[0]).len();
    let pred_cell = n_in; // the writer's target cell, above the inputs
    let mut f = Field::new(n_in + 1);
    let split = data.len() * 3 / 4;
    // TRAIN (a few passes so the grown predictor converges on the train split).
    for _ in 0..6 {
        for row in &data[..split] {
            let iv = inputs(row);
            let pairs: Vec<(usize, f32)> = iv.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            f.grow(&pairs, pred_cell, target(row), 0.05, 200);
        }
    }
    // TEST (held out).
    let (mut sae, mut m) = (0.0f32, 0usize);
    for row in &data[split..] {
        let iv = inputs(row);
        let pairs: Vec<(usize, f32)> = iv.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        f.stimulate(&pairs, 200);
        sae += (f.cells()[pred_cell] - target(row)).abs();
        m += 1;
    }
    sae / m as f32
}

/// Pearson correlation of two signal columns over the dataset — the essence of `co_activate` (fire-together).
fn corr(
    data: &[(f32, f32, f32, f32, f32)],
    x: impl Fn(&(f32, f32, f32, f32, f32)) -> f32,
    y: impl Fn(&(f32, f32, f32, f32, f32)) -> f32,
) -> f32 {
    let n = data.len() as f32;
    let (mx, my) = (
        data.iter().map(&x).sum::<f32>() / n,
        data.iter().map(&y).sum::<f32>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for row in data {
        let (dx, dy) = (x(row) - mx, y(row) - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt() + 1e-9)
}

#[test]
fn surprise_reduction_groups_causally_where_cofire_groups_correlationally() {
    let seeds = [0x81u64, 0x82, 0x83, 0x84];
    let mut r_causal = Vec::new();
    let mut r_spurious = Vec::new();
    let mut co_causal = Vec::new();
    let mut co_spurious = Vec::new();

    for &s in &seeds {
        let data = dataset(s, 1200);

        // SURPRISE-REDUCTION REWARD (conditional): err(Y|Z) − err(Y|Z,X).
        // causal A→B: does A reduce B's surprise beyond Z? (yes — A drives B, Z can't explain it)
        let b_z = fit_mae(&data, |r| vec![r.0], |r| r.2); // err(B | Z)
        let b_za = fit_mae(&data, |r| vec![r.0, r.1], |r| r.2); // err(B | Z, A)
        r_causal.push((b_z - b_za).max(0.0));
        // spurious C→D: does C reduce D's surprise beyond Z? (no — C is redundant with Z)
        let d_z = fit_mae(&data, |r| vec![r.0], |r| r.4); // err(D | Z)
        let d_zc = fit_mae(&data, |r| vec![r.0, r.3], |r| r.4); // err(D | Z, C)
        r_spurious.push((d_z - d_zc).max(0.0));

        // CO-ACTIVATE (correlation, fire-together).
        co_causal.push(corr(&data, |r| r.1, |r| r.2).abs()); // corr(A, B)
        co_spurious.push(corr(&data, |r| r.3, |r| r.4).abs()); // corr(C, D)
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (rc, rs, cc, cs) = (
        mean(&r_causal),
        mean(&r_spurious),
        mean(&co_causal),
        mean(&co_spurious),
    );

    // SURPRISE-REDUCTION reward SEPARATES causal from spurious: reward(A→B) ≫ reward(C→D).
    assert!(
        rc > 0.05 && rc > 3.0 * rs.max(1e-4),
        "surprise-reduction must group the CAUSAL pair far above the spurious one: A→B {rc:.4} vs C→D {rs:.4}"
    );
    // CO-ACTIVATE cannot: the spurious common-cause pair correlates AT LEAST AS STRONGLY as the causal pair —
    // co-firing mis-ranks toward the spurious grouping.
    assert!(
        cs >= cc,
        "co_activate must NOT separate them (common cause correlates ≥ the driver): C~D {cs:.3} vs A~B {cc:.3}"
    );

    println!("\n  CAUSAL GROUPING (Gap 7 test #1 — surprise-reduction vs co-fire):");
    println!("    SURPRISE-REDUCTION reward : causal A→B {rc:.4}   spurious C→D {rs:.4}   ({:.0}× separation)", rc / rs.max(1e-4));
    println!("    CO-ACTIVATE  (correlation): causal A~B {cc:.3}   spurious C~D {cs:.3}   (spurious ranks ≥ causal)");
    println!("  ⇒ grouping by surprise-reduction is CAUSAL; grouping by co-firing is merely correlational.\n");
}
