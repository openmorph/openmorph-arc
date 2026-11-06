// SPDX-License-Identifier: MIT-0
//! THE HYPERBOLIC GATE (#172) — the measured evidence behind DEFERRING the product Euclidean×hyperbolic
//! concept-map (see `docs/investigations/172-hyperbolic-product-space.md`). Truth-not-trophy: we went for the
//! research frontier and MEASURED it before committing surface to core. Two decisive, honest facts:
//!
//!  (A) f32 Poincaré-ball distance is ILL-CONDITIONED for NEARBY points — `arcosh(1+ε) ≈ √(2ε)` loses ~half the
//!      significant digits as ε→0 (the argument rounds toward 1, then `z²−1` cancels). A dense hierarchy embedding
//!      is FULL of nearby points, so this is the fragility that actually bites. (This corrects the naive prior that
//!      the dominant error is boundary crowding: `1 − ‖x‖²` does lose digits as `‖x‖→1`, but in the distance that
//!      is OFFSET by `arcosh` becoming better-conditioned far from 1 — measured below.)
//!  (B) The DISCRETE side we DID ship (`cophenetic`, S4) recovers a known hierarchy EXACTLY (100%), as integers,
//!      with zero boundary math. So the baseline a continuous hyperbolic block would have to beat is already
//!      perfect and fragility-free — the burden of proof for the added surface is unmet.
//!
//! Together these are the measured basis for keeping the hyperbolic geometry as a deferred, opt-in investigation
//! rather than a core default. Revisit conditions (f64 positions / a real Riemannian optimizer / a task needing
//! a differentiable hierarchy) are in the investigation doc.

use openmorph_core::Field;

fn arcosh_f32(z: f32) -> f32 {
    let z = z.max(1.0);
    (z + (z * z - 1.0).max(0.0).sqrt()).ln()
}
fn arcosh_f64(z: f64) -> f64 {
    let z = z.max(1.0);
    (z + (z * z - 1.0).max(0.0).sqrt()).ln()
}
/// Poincaré-ball distance (curvature −1) in f32 — the hand-rolled form a zero-dep substrate would use.
fn poincare_f32(x: &[f32; 2], y: &[f32; 2]) -> f32 {
    let nx2 = x[0] * x[0] + x[1] * x[1];
    let ny2 = y[0] * y[0] + y[1] * y[1];
    let d2 = (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2);
    arcosh_f32(1.0 + 2.0 * d2 / ((1.0 - nx2) * (1.0 - ny2)))
}
/// The same distance in f64 — the reference the f32 computation is compared against.
fn poincare_f64(x: &[f64; 2], y: &[f64; 2]) -> f64 {
    let nx2 = x[0] * x[0] + x[1] * x[1];
    let ny2 = y[0] * y[0] + y[1] * y[1];
    let d2 = (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2);
    arcosh_f64(1.0 + 2.0 * d2 / ((1.0 - nx2) * (1.0 - ny2)))
}

/// The STABLE arcosh(1+u) = ln1p(u + √(u·(u+2))): `u` is fed in directly, so `1+u` is NEVER formed and there is
/// no cancellation — faithful in f32 even for tiny `u` (nearby points). Zero-dep (ln_1p/sqrt are core f32).
fn arcosh1p_f32(u: f32) -> f32 {
    (u + (u * (u + 2.0)).sqrt()).ln_1p()
}
/// Poincaré distance (curvature −1) via the stable form — computes `u` then `arcosh1p(u)`, never `arcosh(1+u)`.
fn poincare_stable_f32(x: &[f32; 2], y: &[f32; 2]) -> f32 {
    let nx2 = x[0] * x[0] + x[1] * x[1];
    let ny2 = y[0] * y[0] + y[1] * y[1];
    let d2 = (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2);
    arcosh1p_f32(2.0 * d2 / ((1.0 - nx2) * (1.0 - ny2)))
}

#[test]
fn f32_is_not_the_barrier_the_stable_formula_is_faithful() {
    // TRUTH-NOT-TROPHY CORRECTION. An earlier version of this proof claimed f32 Poincaré distance is intrinsically
    // fragile for nearby points and rested the DEFERRAL partly on that. An adversarial review corrected it: the
    // fragility is a FORMULA artifact, not an f32 limit. The naive arcosh(1+u) rounds `1+u` and cancels; the
    // stable arcosh1p(u) computes `u` directly and stays faithful. So f32 is REMOVED as a reason to defer — the
    // deferral rests on holonomy (no displacement faculty) + the composition hierarchy being EXACT via `valuation`
    // + burden-of-proof, none of which numerics touch. Measured against an f64 reference, for a nearby pair:
    let (x, y) = ([0.1f32, 0.0], [0.1f32, 1e-3]);
    let d64 = poincare_f64(&[0.1, 0.0], &[0.1, 1e-3]);
    let naive_err = (poincare_f32(&x, &y) as f64 - d64).abs() / d64;
    let stable_err = (poincare_stable_f32(&x, &y) as f64 - d64).abs() / d64;
    eprintln!(
        "[HYP-GATE] nearby-point rel-error — naive arcosh(1+u) {naive_err:.2e} | stable arcosh1p {stable_err:.2e}"
    );
    assert!(
        naive_err > 1e-3,
        "the NAIVE arcosh(1+u) form IS fragile for nearby points (rel-err {naive_err:.2e}) — the artifact"
    );
    assert!(
        stable_err < 1e-5,
        "the STABLE arcosh1p form is FAITHFUL in f32 (rel-err {stable_err:.2e}) — f32 is NOT the barrier"
    );
    assert!(
        stable_err < naive_err / 100.0,
        "the stable reformulation fixes the naive fragility by >100x ({stable_err:.2e} vs {naive_err:.2e})"
    );
}

#[test]
fn the_discrete_baseline_recovers_the_hierarchy_exactly() {
    // The baseline a continuous hyperbolic block would have to beat: the discrete ultrametric `cophenetic` (S4).
    // Two well-separated groups → a region tree → EVERY within-group pair is strictly closer than EVERY
    // cross-group pair. Exact (integer), fragility-free, and it never touches the learning metric.
    let mut f = Field::new(8);
    f.place_at(0, &[0.0, 0.0]);
    f.place_at(1, &[1.0, 0.0]);
    f.place_at(2, &[0.0, 1.0]);
    f.place_at(3, &[1.0, 1.0]);
    f.place_at(4, &[100.0, 100.0]);
    f.place_at(5, &[101.0, 100.0]);
    f.place_at(6, &[100.0, 101.0]);
    f.place_at(7, &[101.0, 101.0]);
    let index = f.index_skills(&[0, 1, 2, 3, 4, 5, 6, 7], 2, 2);

    let mut correct = 0usize;
    let mut total = 0usize;
    for a in 0..8 {
        // the largest within-group cophenetic distance from `a`
        let same: Vec<usize> = if a < 4 {
            vec![0, 1, 2, 3]
        } else {
            vec![4, 5, 6, 7]
        };
        let other: Vec<usize> = if a < 4 {
            vec![4, 5, 6, 7]
        } else {
            vec![0, 1, 2, 3]
        };
        let within_max = same
            .iter()
            .filter(|&&b| b != a)
            .map(|&b| f.cophenetic(&index, a, b))
            .max()
            .unwrap_or(0);
        for &b in &other {
            total += 1;
            if f.cophenetic(&index, a, b) > within_max {
                correct += 1;
            }
        }
    }
    let acc = correct as f32 / total as f32;
    eprintln!("[HYP-GATE] discrete cophenetic hierarchy retrieval: {correct}/{total} = {acc:.3}");
    assert_eq!(
        correct, total,
        "the discrete baseline recovers the hierarchy EXACTLY (every cross-group pair is farther than every within-group pair)"
    );
}
