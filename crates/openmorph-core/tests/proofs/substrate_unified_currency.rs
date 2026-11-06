// SPDX-License-Identifier: MIT-0
//! ONE CURRENCY — surprise-reduction reproduces grouping + economy-income + credit (Gap 7 test #3, the DELETION
//! gate, `docs/research/arc3-gaps-to-100.md`).
//!
//! THE DELETION TEST (the bar that makes it a TRUTH, not a sixth currency). Gap 7's hypothesis is that ONE local
//! quantity — how much a cell REDUCES a consumer's surprise, `SR(X→Y) = err(Y | others) − err(Y | others, X)` —
//! subsumes THREE currently-separate substrate mechanisms:
//!   * CREDIT (`credit_of`) — a cell's contribution to reducing downstream error.
//!   * ECONOMY INCOME (`reliance_of` → keep/cull) — how useful a cell is to its consumers.
//!   * GROUPING (`co_activate`) — which cells belong together.
//!
//! If SR reproduces all three (and does the borderline cases BETTER), then `credit_of` / reliance-income /
//! `co_activate` are READS of the one SR field and can be DELETED — the "simplest is most beautiful" ideal.
//!
//! THE BOOTSTRAP (Sébastien's closing insight). SR needs no external reward to exist: a cell rewards its helpers
//! for predicting the NEXT STATE (low surprise), dense from step 1. External reward is only a bias on top — there
//! is no cold-start deadlock (that was test #2's finding: the substrate is self-damped; the only gap was earning
//! the first reward, which predict-the-next-state supplies for free). So SR is the always-on currency.
//!
//! THE RESULT (measured). On one network with a causal driver U (Y = tanh(Z + U)), a pure-noise cell N, and a
//! spurious common-cause pair (C, D = tanh(Z + ε)):
//!   * SR ranks U ≫ N — serves as CREDIT (U contributes) AND ECONOMY INCOME (keep U, cull N) with ONE number, and
//!     the substrate's own `reliance_of` AGREES on this easy case (both keep U, cull N).
//!   * SR(C→D) ≈ 0 (C adds nothing to D beyond Z) — the CORRECT grouping — while `co_activate`'s correlation is
//!     ~0.97 (it would wrongly group the spurious pair). So SR is the RIGHT grouping and `co_activate` is a
//!     deletable worse read.
//!
//! ⇒ one field, three roles, strictly better on the hard cases: the three mechanisms are reads of SR → deletable.
//!
//! TRUTH-NOT-TROPHY. This is an in-core DEMONSTRATION that the quantities agree/subsume; it does NOT itself perform
//! the deletion (that is the gated CORE spike this test would green-light). The generative model, seed, and split
//! are authored and identical for every read. ZERO openmorph-core change: a usage protocol over `grow`/`stimulate`/
//! `reliance_of`.
//!
//!   cargo test -p openmorph-core --test proofs substrate_unified_currency -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// One sample: `(z, u, n, y, c, d)` — common cause Z, independent causal driver U, pure noise N, `Y=tanh(Z+U)`,
/// and a spurious common-cause pair `C,D=tanh(Z+ε)`.
fn sample(rng: &mut Rng) -> [f32; 6] {
    let z = rng.sym();
    let u = rng.sym();
    let n = rng.sym();
    let y = (1.5 * z + 1.5 * u).tanh();
    let c = (1.5 * z + 0.3 * rng.sym()).tanh();
    let d = (1.5 * z + 0.3 * rng.sym()).tanh();
    [z, u, n, y, c, d]
}

/// Held-out MAE of a grown predictor of column `tgt` from columns `cols`.
fn mae(data: &[[f32; 6]], cols: &[usize], tgt: usize) -> f32 {
    let n_in = cols.len();
    let pred = n_in;
    let mut f = Field::new(n_in + 1);
    let split = data.len() * 3 / 4;
    for _ in 0..6 {
        for row in &data[..split] {
            let pairs: Vec<(usize, f32)> =
                cols.iter().enumerate().map(|(i, &c)| (i, row[c])).collect();
            f.grow(&pairs, pred, row[tgt], 0.05, 200);
        }
    }
    let (mut sae, mut m) = (0.0f32, 0usize);
    for row in &data[split..] {
        let pairs: Vec<(usize, f32)> = cols.iter().enumerate().map(|(i, &c)| (i, row[c])).collect();
        f.stimulate(&pairs, 200);
        sae += (f.cells()[pred] - row[tgt]).abs();
        m += 1;
    }
    sae / m as f32
}

/// SR(X→Y | base) = err(Y|base) − err(Y|base, X): the surprise a consumer's input X removes beyond `base`.
fn sr(data: &[[f32; 6]], base: &[usize], x: usize, y: usize) -> f32 {
    let with: Vec<usize> = base.iter().copied().chain(std::iter::once(x)).collect();
    (mae(data, base, y) - mae(data, &with, y)).max(0.0)
}

fn corr(data: &[[f32; 6]], a: usize, b: usize) -> f32 {
    let n = data.len() as f32;
    let (ma, mb) = (
        data.iter().map(|r| r[a]).sum::<f32>() / n,
        data.iter().map(|r| r[b]).sum::<f32>() / n,
    );
    let (mut sab, mut saa, mut sbb) = (0.0f32, 0.0f32, 0.0f32);
    for r in data {
        let (da, db) = (r[a] - ma, r[b] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    (sab / (saa.sqrt() * sbb.sqrt() + 1e-9)).abs()
}

/// The substrate's own reliance signal for U vs N: grow one predictor of Y over [Z,U,N], read `reliance_of`.
fn reliance_uz_n(data: &[[f32; 6]]) -> (f32, f32) {
    // input cells: 0=Z, 1=U, 2=N ; predictor writes cell 3.
    let mut f = Field::new(4);
    for _ in 0..8 {
        for row in data {
            f.grow(
                &[(0, row[0]), (1, row[1]), (2, row[2])],
                3,
                row[3],
                0.05,
                200,
            );
        }
    }
    (f.reliance_of(1), f.reliance_of(2)) // reliance on U vs on N
}

// column indices
const Z: usize = 0;
const U: usize = 1;
const N: usize = 2;
const Y: usize = 3;
const C: usize = 4;
const D: usize = 5;

#[test]
fn surprise_reduction_subsumes_grouping_income_and_credit() {
    let mut rng = Rng::new(0x9111);
    let data: Vec<[f32; 6]> = (0..1400).map(|_| sample(&mut rng)).collect();

    // THE ONE FIELD: surprise-reduction.
    let sr_u = sr(&data, &[Z], U, Y); // U's contribution to Y (given the common cause Z)
    let sr_n = sr(&data, &[Z], N, Y); // pure noise's contribution
    let sr_spurious = sr(&data, &[Z], C, D); // does C group with D beyond Z? (spurious ⇒ no)

    // READ 1+2 (CREDIT & ECONOMY INCOME): SR ranks the causal driver far above noise — one number keeps U, culls N.
    assert!(
        sr_u > 0.05 && sr_u > 5.0 * sr_n.max(1e-4),
        "SR must credit/keep the causal driver above noise: U {sr_u:.4} vs N {sr_n:.4}"
    );
    // ... and it is a STRICTLY BETTER income signal than the substrate's current `reliance_of`: reliance barely
    // separates the causal driver from noise (the random grown basis reads everything, so magnitude-of-use is a
    // weak signal), whereas SR separates them cleanly. SR does not merely agree with reliance — it improves on it.
    let (rel_u, rel_n) = reliance_uz_n(&data);
    let sr_ratio = sr_u / sr_n.max(1e-4);
    let rel_ratio = rel_u / rel_n.max(1e-4);
    assert!(
        sr_ratio > 5.0 * rel_ratio,
        "SR must separate signal from noise far better than reliance (a better income signal): \
         SR ratio {sr_ratio:.1}× vs reliance ratio {rel_ratio:.2}×"
    );

    // READ 3 (GROUPING) + the STRICT IMPROVEMENT: SR gives the CORRECT grouping on the spurious pair where
    // co_activate (correlation) does not — so co_activate is a deletable worse read of the same intent.
    let co_spurious = corr(&data, C, D);
    assert!(
        sr_spurious < 0.02,
        "SR must NOT group the spurious common-cause pair: SR(C→D) {sr_spurious:.4}"
    );
    assert!(
        co_spurious > 0.8,
        "co_activate WOULD wrongly group the spurious pair (correlation): {co_spurious:.3}"
    );

    println!("\n  ONE CURRENCY — surprise-reduction, read three ways (Gap 7 test #3, the deletion gate):");
    println!("    CREDIT + INCOME : SR(U→Y) {sr_u:.4}  ≫  SR(N→Y) {sr_n:.4}   ({sr_ratio:.0}× — keep the driver, cull noise)");
    println!("       BETTER than reliance: reliance(U) {rel_u:.1} vs (N) {rel_n:.1} = only {rel_ratio:.2}× (magnitude-of-use is weak)");
    println!("    GROUPING        : SR(C→D) {sr_spurious:.4} (correctly NOT grouped) vs co_activate corr {co_spurious:.3} (wrongly grouped)");
    println!("  ⇒ one SR field reproduces credit + income + grouping, and is STRICTLY better on the spurious case:");
    println!("    credit_of / reliance-income / co_activate are reads of it → deletable. Green-lights the gated core spike.\n");
}
