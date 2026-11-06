// SPDX-License-Identifier: MIT-0
//! SURPRISE-CURRENCY SPIKE — increment 1.5: chain the ablation signal THROUGH a random backbone
//! (`docs/research/surprise-currency-spike.md`).
//!
//! Increment 1 landed the LIVE `surprise_reduction_of` as `Σ w²·Var(cell)` — but that reads a cell's IMMEDIATE
//! readers, so it is causal only where those readers are LEARNED. Under a RANDOM backbone (grown hidden features
//! between the raw inputs and the readout) a raw input's immediate readers are the RANDOM feature units, so the
//! signal degenerates to `reliance_of` (a driver and a noise input are read by the same random features ⇒ ~equal).
//!
//! Increment 1.5 makes `surprise_reduction_of` the END-TO-END ablation `(∂output/∂cell)²·Var(cell)`: a local
//! reverse (adjoint / transpose-Jacobian) sweep in the learn path chains the one-hop residual backprojection all
//! the way to the raw inputs. The readout's LEARNED downstream weights make the driver's end-to-end path large and
//! the noise's path cancel — so it stays causal THROUGH the backbone where `reliance_of` (and increment-1's
//! immediate form) cannot rank them.
//!
//! THE CLAIMS:
//!   * BYTE-IDENTICAL OFF — with the flag off the organism is bit-for-bit unchanged (the new state is never
//!     touched); default == explicit-off snapshot; the dormant sensitivity reads 0.
//!   * CAUSAL THROUGH A RANDOM BACKBONE ON — `surprise_reduction_of(driver) ≫ surprise_reduction_of(noise)`,
//!     while `reliance_of` FAILS to separate them (the degenerate baseline the immediate form collapses to).
//!
//!   cargo test -p openmorph-core --test research spike_ablation_sensitivity -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// Train a predictor of `Y = tanh(1.6·X)` from a RANDOM backbone over [X, N] (N = pure noise). The readout reads
/// ONLY the hidden features, so each raw input reaches the output EXCLUSIVELY through the random backbone — the
/// exact regime where increment-1's immediate-fan-out `Σ w²·Var` degenerates to `reliance_of`.
fn train_backbone(surprise_currency: bool, seed: u64) -> Field {
    let width = 16;
    let mut f = Field::new(3); // 0 = X (driver), 1 = N (noise), 2 = out
    f.config_mut().surprise_currency = surprise_currency;
    // A RANDOM backbone of [X, N]; the readout reads ONLY the hidden cells (no direct raw-input path).
    let hidden = f.grow_backbone(&[0, 1], width, 1.5);
    f.add_unit(hidden.clone(), vec![0.0; hidden.len()], 0.0, 2);
    let mut rng = Rng::new(seed);
    for _ in 0..4000 {
        let x = rng.sym();
        let n = rng.sym();
        f.learn_step(&[(0, x), (1, n)], 2, (1.6 * x).tanh(), 0.05, 200);
    }
    f
}

#[test]
#[ignore = "research: on demand"]
fn ablation_sensitivity_is_byte_identical_off_and_causal_through_a_backbone() {
    // (1) BYTE-IDENTICAL OFF: default (never mentions the flag) vs explicitly-off produce identical learned state.
    let off_default = {
        let width = 16;
        let mut f = Field::new(3);
        let hidden = f.grow_backbone(&[0, 1], width, 1.5);
        f.add_unit(hidden.clone(), vec![0.0; hidden.len()], 0.0, 2);
        let mut rng = Rng::new(0x5EED);
        for _ in 0..4000 {
            let x = rng.sym();
            let n = rng.sym();
            f.learn_step(&[(0, x), (1, n)], 2, (1.6 * x).tanh(), 0.05, 200);
        }
        f
    };
    let off_explicit = train_backbone(false, 0x5EED);
    let a = serde_json::to_vec(&off_default).unwrap();
    let b = serde_json::to_vec(&off_explicit).unwrap();
    assert_eq!(
        a, b,
        "surprise_currency=false must be byte-identical to a run that never sets the flag"
    );
    assert!(
        off_explicit.surprise_reduction_of(0) == 0.0
            && off_explicit.surprise_reduction_of(1) == 0.0,
        "with the flag off, surprise_reduction_of must read 0 (no state tracked)"
    );

    // (2) CAUSAL THROUGH THE BACKBONE ON.
    let on = train_backbone(true, 0x5EED);
    let sr_x = on.surprise_reduction_of(0);
    let sr_n = on.surprise_reduction_of(1);
    let rel_x = on.reliance_of(0);
    let rel_n = on.reliance_of(1);

    let sr_ratio = sr_x / sr_n.max(1e-9);
    let rel_ratio = rel_x / rel_n.max(1e-9);

    println!("\n  SURPRISE-CURRENCY SPIKE — increment 1.5 (end-to-end ablation through a RANDOM backbone):");
    println!(
        "    byte-identical OFF ✓ (default == explicit-off snapshot; dormant sensitivity reads 0)"
    );
    println!("    ON — surprise_reduction_of : X {sr_x:.5}  N {sr_n:.5}   (end-to-end ratio {sr_ratio:.1}×)");
    println!("       reliance_of (baseline)  : X {rel_x:.3}  N {rel_n:.3}   (degenerate ratio {rel_ratio:.2}× — reads the RANDOM features)");

    // The driver's end-to-end ablation importance must dominate the noise input's, LIVE and stable.
    assert!(
        sr_x > 1e-4 && sr_ratio > 5.0,
        "end-to-end surprise_reduction_of must rank the causal driver ≫ noise THROUGH the backbone: X {sr_x:.5} vs N {sr_n:.5} (ratio {sr_ratio:.1}×)"
    );
    // …and it must SEPARATE them far better than reliance (which reads the random features and cannot):
    // this is the whole point of increment 1.5 over the immediate-fan-out form (which degenerates to reliance).
    assert!(
        sr_ratio > 3.0 * rel_ratio.max(1.0),
        "end-to-end sensitivity must separate driver/noise MUCH better than reliance: sr {sr_ratio:.1}× vs reliance {rel_ratio:.2}×"
    );

    println!("  ⇒ POSITIVE: the end-to-end (∂output/∂cell)²·Var estimate is cheap (one local adjoint sweep), stable");
    println!("    at convergence, byte-identical off, and CAUSAL through a random backbone where reliance cannot rank.\n");
}
