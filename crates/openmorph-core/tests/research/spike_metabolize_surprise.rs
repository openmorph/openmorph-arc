// SPDX-License-Identifier: MIT-0
//! SURPRISE-CURRENCY SPIKE — increment 2: the economy runs on ONE currency (`docs/research/surprise-currency-spike`).
//!
//! THE DELETION-BY-DERIVATION. Increments 0/1/1.5 established the unified currency: the END-TO-END ablation
//! surprise-reduction `surprise_reduction_of(c) = (∂output/∂c)²·Var(c)` — stable at convergence AND causal through a
//! random backbone, where `reliance_of` is blind. Increment 2 replaces the economy's keep/cull income with it:
//! `Field::metabolize_surprise` (income ∝ `surprise_reduction_of`), the drop-in for `metabolize_reliance`.
//!
//! THE TEST (where reliance is BLIND but surprise is not). Two DEEP features — U = tanh(X) (carries the signal),
//! Nz = tanh(N) (carries only noise) — are read ONLY by a RANDOM hidden backbone, which feeds the learned readout
//! predicting Y = tanh(X). So `reliance_of(U)` and `reliance_of(Nz)` are both just Σ of the backbone's RANDOM
//! weights — reliance cannot tell them apart. But the end-to-end sensitivity can: the readout learns to extract X
//! and cancel N, so `∂Y/∂U` is large and `∂Y/∂Nz ≈ 0`.
//!
//! THE CLAIM. `metabolize_surprise` CULLS the noise feature Nz (starves it below death) while keeping U — where
//! `metabolize_reliance` keeps BOTH (Nz survives, because reliance can't rank a feature below a random layer). One
//! currency, and it makes the economy's keep/cull CAUSAL. This is the payoff: `metabolize_reliance` becomes a
//! derivable special case, retirable once validated live.
//!
//! TRUTH-NOT-TROPHY. `metabolize_surprise` is purely ADDITIVE (nothing existing calls it) and only meaningful with
//! `Config::surprise_currency` on ⇒ the default path is byte-identical (whole suite green). The generative model,
//! seed, income/upkeep, and death threshold are authored and identical across the two economy arms; only the income
//! RULE differs. Honest residual (unchanged from 1.5): SR still does not discount collinear REDUNDANCY on
//! independent paths. This proof does NOT yet DELETE `metabolize_reliance` (that flips a default ⇒ needs live
//! validation) — it proves the domination that justifies the deletion.
//!
//!   cargo test -p openmorph-core --test research spike_metabolize_surprise -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// Build the topology (X,N → deep features U,Nz → random backbone → learned readout of Y=tanh(X)), train the
/// readout with the surprise-currency on, and return `(field, u_unit, nz_unit, u_cell, nz_cell)`.
fn build_and_train(seed: u64) -> (Field, usize, usize, usize, usize) {
    let mut f = Field::new(2); // 0 = X, 1 = N
    f.config_mut().surprise_currency = true;
    f.rng_seed(seed);
    // deep features: U reads X (signal), Nz reads N (noise), each writing its own cell.
    let (u_cell, nz_cell) = (2usize, 3usize);
    let u_unit = f.add_unit(vec![0], vec![1.6], 0.0, u_cell);
    let nz_unit = f.add_unit(vec![1], vec![1.6], 0.0, nz_cell);
    // a RANDOM hidden backbone reading ONLY the two deep features — these are the sole readers of U, Nz.
    let hidden = f.grow_backbone(&[u_cell, nz_cell], 16, 2.5);
    // the learned readout of Y over the backbone.
    let out = f.len();
    let n = hidden.len();
    f.add_unit(hidden, vec![0.0; n], 0.0, out);

    let mut rng = Rng::new(seed ^ 0xBEEF);
    for _ in 0..4000 {
        let x = rng.sym();
        let nz = rng.sym();
        // set both inputs, settle the whole cascade, and correct the readout toward Y = tanh(1.6·X).
        f.learn_step(&[(0, x), (1, nz)], out, (1.6 * x).tanh(), 0.05, 300);
    }
    (f, u_unit, nz_unit, u_cell, nz_cell)
}

/// Run `steps` of the given economy income rule on the two deep-feature cells and return `(energy_U, energy_Nz)`.
fn run_economy(surprise: bool, seed: u64, income: f32, upkeep: f32, steps: usize) -> (f32, f32) {
    let (mut f, u_unit, nz_unit, u_cell, nz_cell) = build_and_train(seed);
    for _ in 0..steps {
        if surprise {
            f.metabolize_surprise(&[u_cell, nz_cell], income, upkeep);
        } else {
            f.metabolize_reliance(&[u_cell, nz_cell], income, upkeep);
        }
    }
    (f.energy_of(u_unit), f.energy_of(nz_unit))
}

#[test]
#[ignore = "research: on demand"]
fn metabolize_surprise_culls_the_noise_feature_reliance_keeps() {
    let seed = 0x2CAFE;
    let (f, _, _, u_cell, nz_cell) = build_and_train(seed);

    // (1) THE SIGNAL — surprise-reduction separates the deep signal feature from the deep noise feature; reliance,
    // reading their RANDOM immediate readers, cannot.
    let (sr_u, sr_n) = (
        f.surprise_reduction_of(u_cell),
        f.surprise_reduction_of(nz_cell),
    );
    let (rel_u, rel_n) = (f.reliance_of(u_cell), f.reliance_of(nz_cell));
    let sr_ratio = sr_u / sr_n.max(1e-9);
    let rel_ratio = rel_u / rel_n.max(1e-9);
    assert!(
        sr_u > 1e-6 && sr_ratio > 3.0,
        "surprise-reduction must rank the deep SIGNAL feature above the deep NOISE feature: U {sr_u:.6} vs Nz {sr_n:.6}"
    );
    assert!(
        rel_ratio < 2.0,
        "reliance must be ~blind to which deep feature matters (random readers): U {rel_u:.3} vs Nz {rel_n:.3}"
    );

    // (2) THE ECONOMY — pick income/upkeep so the SIGNAL feature earns its keep and the NOISE feature starves under
    // surprise; the SAME dials under reliance keep BOTH (reliance can't cull the noise below a random layer).
    let (income, upkeep, death, steps) = (2.0f32, 0.02f32, 0.05f32, 300);
    let (eu_s, en_s) = run_economy(true, seed, income, upkeep, steps);
    let (eu_r, en_r) = run_economy(false, seed, income, upkeep, steps);

    assert!(
        eu_s > death && en_s <= death,
        "metabolize_surprise must KEEP the signal (E_U {eu_s:.3} > {death}) and CULL the noise (E_Nz {en_s:.3} ≤ {death})"
    );
    assert!(
        en_r > death,
        "metabolize_reliance must FAIL to cull the noise feature (E_Nz {en_r:.3} > {death}) — reliance is blind here"
    );

    println!("\n  SURPRISE-CURRENCY SPIKE — increment 2 (metabolize_surprise):");
    println!("    deep-feature signal : surprise_reduction U {sr_u:.6} vs Nz {sr_n:.6}  ({sr_ratio:.0}×, causal)");
    println!("                          reliance          U {rel_u:.3} vs Nz {rel_n:.3}  ({rel_ratio:.2}×, blind)");
    println!("    economy (income {income}, upkeep {upkeep}, death {death}, {steps} steps):");
    println!("      metabolize_surprise : E_U {eu_s:.3} (kept)  E_Nz {en_s:.3} (CULLED)");
    println!("      metabolize_reliance : E_U {eu_r:.3}         E_Nz {en_r:.3} (noise SURVIVES — reliance blind)");
    println!("  ⇒ one surprise currency makes the economy's keep/cull CAUSAL; reliance-income is the derivable case.\n");
}
