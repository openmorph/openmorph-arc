// SPDX-License-Identifier: MIT-0
//! SURPRISE-CURRENCY SPIKE — the unified currency must be the ABLATION form (stable at convergence AND causal),
//! not `credit_of` (`docs/research/surprise-currency-spike.md`, Gap 7 core spike, increment 0).
//!
//! THE DEEP FINDING THE SPIKE SURFACED. The substrate deliberately splits the economy into `reliance_of`
//! (convergence-STABLE, structural keep/cull) and `credit_of` (residual CORRELATION, breeds traits) because —
//! per the core's own docstring — `credit_of` → 0 for a USED feature AT CONVERGENCE (no residual left to correlate
//! with). So the naive "credit is the one currency" would CULL a converged-useful feature. That is the real
//! obstacle to Gap 7's deletion.
//!
//! THE RESOLUTION. Gap 7's grouping/deletion tests (`substrate_causal_grouping` / `substrate_unified_currency`)
//! measured surprise-reduction as an ABLATION / counterfactual: `SR(X) = err(Y | without X) − err(Y | with X)`.
//! Unlike `credit_of`, this does NOT vanish at convergence — at convergence `err(Y|with X)` is small but
//! `err(Y|without X)` stays LARGE (X is still needed), so `SR(X)` stays large. And unlike `reliance_of`, it is
//! CAUSAL (a redundant/spurious input earns ~0). So the ABLATION surprise-reduction is the currency that has
//! reliance's STABILITY and credit's/grouping's CAUSALITY at once — the thing worth making a live core primitive.
//!
//! THE CLAIM (increment 0, measured). Train a predictor of Y from a causal driver X and a noise input N to
//! convergence, then read the three signals:
//!   * `credit_of` (residual correlation) COLLAPSES toward 0 for X — it cannot keep a converged-useful feature.
//!   * `reliance_of` stays nonzero for X but barely separates X from N (correlational, weak).
//!   * ABLATION-SR stays LARGE for X and ~0 for N — stable AND causal: the unified currency.
//!
//! TRUTH-NOT-TROPHY. This motivates the core primitive; it does not yet add it. The generative model, seed, and
//! split are authored and identical for every read. ZERO openmorph-core change in this increment — a usage protocol
//! over `grow` / `credit_of` / `reliance_of` establishing WHICH quantity the gated core spike should implement.
//!
//!   cargo test -p openmorph-core --test research spike_currency_stability -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// `(x, n, y)` — a causal driver X, a pure-noise input N, and `Y = tanh(1.6·X)` (Y depends ONLY on X).
fn sample(rng: &mut Rng) -> (f32, f32, f32) {
    let x = rng.sym();
    let n = rng.sym();
    (x, n, (1.6 * x).tanh())
}

fn held_out_mae(data: &[(f32, f32, f32)], cols: &[usize]) -> f32 {
    let n_in = cols.len();
    let pred = n_in;
    let mut f = Field::new(n_in + 1);
    let split = data.len() * 3 / 4;
    for _ in 0..8 {
        for row in &data[..split] {
            let vals = [row.0, row.1, row.2];
            let pairs: Vec<(usize, f32)> = cols
                .iter()
                .enumerate()
                .map(|(i, &c)| (i, vals[c]))
                .collect();
            f.grow(&pairs, pred, row.2, 0.05, 200);
        }
    }
    let (mut sae, mut m) = (0.0f32, 0usize);
    for row in &data[split..] {
        let vals = [row.0, row.1, row.2];
        let pairs: Vec<(usize, f32)> = cols
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, vals[c]))
            .collect();
        f.stimulate(&pairs, 200);
        sae += (f.cells()[pred] - row.2).abs();
        m += 1;
    }
    sae / m as f32
}

#[test]
#[ignore = "research: on demand"]
fn ablation_surprise_reduction_is_stable_and_causal() {
    let mut rng = Rng::new(0xAB1A);
    let data: Vec<(f32, f32, f32)> = (0..1400).map(|_| sample(&mut rng)).collect();

    // Train ONE predictor of Y from [X(0), N(1)] to convergence and read the LIVE substrate signals.
    let mut f = Field::new(3); // 0=X, 1=N, pred=2
    for _ in 0..40 {
        for row in &data {
            f.grow(&[(0, row.0), (1, row.1)], 2, row.2, 0.05, 200);
        }
    }
    let credit_pred = f.credit_of(2); // residual-correlation contribution at convergence
    let reliance_x = f.reliance_of(0);
    let reliance_n = f.reliance_of(1);

    // ABLATION surprise-reduction (the counterfactual): err(Y | N only) − err(Y | X, N).
    let mae_full = held_out_mae(&data, &[0, 1]);
    let mae_no_x = held_out_mae(&data, &[1]); // remove the causal driver
    let mae_no_n = held_out_mae(&data, &[0]); // remove the noise
    let sr_x = (mae_no_x - mae_full).max(0.0);
    let sr_n = (mae_no_n - mae_full).max(0.0);

    // (1) credit_of COLLAPSES at convergence — it cannot be the keep/cull currency (would cull the used feature).
    assert!(
        credit_pred < 0.05,
        "credit_of must collapse toward 0 at convergence (why it can't keep a used feature): {credit_pred:.4}"
    );
    // (2) reliance stays nonzero but barely separates X (causal) from N (noise) — stable yet correlational/weak.
    let rel_ratio = reliance_x / reliance_n.max(1e-4);
    assert!(
        reliance_x > 0.1 && rel_ratio < 2.5,
        "reliance stays up but barely separates causal from noise: X {reliance_x:.3} vs N {reliance_n:.3} ({rel_ratio:.2}×)"
    );
    // (3) ABLATION-SR is STABLE (large for X at convergence) AND CAUSAL (≈0 for N) — the unified currency.
    assert!(
        sr_x > 0.1 && sr_x > 5.0 * sr_n.max(1e-4),
        "ablation-SR must stay large for the causal driver and ~0 for noise: X {sr_x:.4} vs N {sr_n:.4}"
    );

    println!("\n  SURPRISE-CURRENCY SPIKE — which quantity survives convergence AND stays causal:");
    println!("    credit_of(pred) at convergence : {credit_pred:.4}   (COLLAPSED → cannot keep a used feature)");
    println!("    reliance_of  X {reliance_x:.3}  N {reliance_n:.3}   ({rel_ratio:.2}× — stable but correlational/weak)");
    println!(
        "    ABLATION-SR  X {sr_x:.4}  N {sr_n:.4}   ({:.0}× — STABLE at convergence AND causal)",
        sr_x / sr_n.max(1e-4)
    );
    println!("  ⇒ the unified currency is the ABLATION surprise-reduction: reliance's stability + credit's causality.");
    println!("    Next increment: add a live `surprise_reduction_of` core primitive (gated, bit-identical off).\n");
}
