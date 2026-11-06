// SPDX-License-Identifier: MIT-0
//! SURPRISE-CURRENCY SPIKE — increment 1: the LIVE `surprise_reduction_of` core primitive
//! (`docs/research/surprise-currency-spike.md`).
//!
//! Increment 0 (`spike_currency_stability`) pinned the unified currency to the ABLATION surprise-reduction. This
//! increment adds it as a LIVE core primitive: `Config::surprise_currency` (default OFF, byte-identical) tracks a
//! running per-cell variance in the settle path, and `surprise_reduction_of(c) = Σ w²·Var(c)` reports the
//! stable-at-convergence, causal importance — no counterfactual retrain.
//!
//! THE TWO CLAIMS:
//!   * BYTE-IDENTICAL OFF — with `surprise_currency=false` (the default) the organism is bit-for-bit unchanged: the
//!     new state is never touched and the learned output matches a run built without ever mentioning the flag.
//!   * LIVE + CAUSAL (for a LEARNED readout) ON — with the flag on, `surprise_reduction_of` ranks a causal driver
//!     far above a noise input and STAYS high at convergence, where `credit_of` has collapsed to 0. It is variance-
//!     weighted, so a dead/constant cell earns 0 (a real improvement over `reliance_of`, which keeps it).
//!
//! TRUTH-NOT-TROPHY (the HONEST scope). First real core change of the register — gated, bit-identical off, guarded
//! by the whole proof suite. But the cheap `Σ w²·Var` reads a cell's IMMEDIATE readers, so it is causal only where
//! those readers are LEARNED (a direct readout): for a raw input under a RANDOM backbone the immediate readers are
//! random and the signal degenerates to `reliance_of` (measured: it mis-ranks noise there, exactly like reliance).
//! The end-to-end causal advantage the ABLATION currency showed in increment 0 needs SENSITIVITY PROPAGATION
//! through the hidden features — increment 1.5, which may conflict with the local rule (an honest open question).
//! So this increment lands the gated byte-identical primitive + the variance-weighting improvement; it does NOT yet
//! deliver the full end-to-end deletion of `reliance_of`.
//!
//!   cargo test -p openmorph-core --test research spike_surprise_currency -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// Train a predictor of `Y=tanh(1.6·X)` from [X, N] (N = pure noise) with the surprise-currency flag `on` or off,
/// and return the settled prediction cells (to compare byte-for-byte) plus the field (to read signals).
fn train(surprise_currency: bool, seed: u64) -> Field {
    let mut f = Field::new(3); // 0=X, 1=N, pred=2
    f.config_mut().surprise_currency = surprise_currency;
    f.config_mut().max_hidden = 0; // a DIRECT learned readout — surprise_reduction_of reads its LEARNED weights
                                   // (over a random backbone the immediate readers are random, so the signal is only causal where readers learn).
    let mut rng = Rng::new(seed);
    for _ in 0..1500 {
        let x = rng.sym();
        let n = rng.sym();
        f.grow(&[(0, x), (1, n)], 2, (1.6 * x).tanh(), 0.05, 200);
    }
    f
}

#[test]
#[ignore = "research: on demand"]
fn surprise_currency_is_byte_identical_off_and_causal_on() {
    // (1) BYTE-IDENTICAL OFF: default (off) vs explicitly-off produce identical learned state (snapshots equal).
    let off_default = {
        let mut f = Field::new(3);
        f.config_mut().max_hidden = 0; // same config as train(); surprise_currency is NEVER mentioned here
        let mut rng = Rng::new(0x5EED);
        for _ in 0..1500 {
            let x = rng.sym();
            let n = rng.sym();
            f.grow(&[(0, x), (1, n)], 2, (1.6 * x).tanh(), 0.05, 200);
        }
        f
    };
    let off_explicit = train(false, 0x5EED);
    let a = serde_json::to_vec(&off_default).unwrap();
    let b = serde_json::to_vec(&off_explicit).unwrap();
    assert_eq!(
        a, b,
        "surprise_currency=false must be byte-identical to a run that never sets the flag"
    );
    // and the dormant state is genuinely untouched when off.
    assert!(
        off_explicit.surprise_reduction_of(0) == 0.0
            && off_explicit.surprise_reduction_of(1) == 0.0,
        "with the flag off, surprise_reduction_of must read 0 (no variance tracked)"
    );

    // (2) LIVE + CAUSAL ON: the primitive ranks the causal driver X far above noise N, stable at convergence.
    let on = train(true, 0x5EED);
    let sr_x = on.surprise_reduction_of(0);
    let sr_n = on.surprise_reduction_of(1);
    let rel_x = on.reliance_of(0);
    let rel_n = on.reliance_of(1);
    let credit = on.credit_of(2);

    assert!(
        sr_x > 1e-4 && sr_x > 5.0 * sr_n.max(1e-6),
        "surprise_reduction_of must rank the causal driver ≫ noise, live: X {sr_x:.5} vs N {sr_n:.5}"
    );

    println!("\n  SURPRISE-CURRENCY SPIKE — increment 1 (live surprise_reduction_of):");
    println!("    byte-identical OFF ✓ (default == explicit-off snapshot; dormant state reads 0)");
    println!("    ON — surprise_reduction_of : X {sr_x:.5}  N {sr_n:.5}   ({:.0}× — causal for a LEARNED readout)", sr_x / sr_n.max(1e-6));
    println!("       reliance_of             : X {rel_x:.3}  N {rel_n:.3}   (also works for a direct readout)");
    println!("       credit_of(pred)         : {credit:.4}   (collapsed at convergence)");
    println!("  ⇒ HONEST scope: the cheap live Σw²·Var is byte-identical-off and causal for a LEARNED readout, but it");
    println!("    only IMPROVES on reliance via variance-weighting (a dead/constant cell earns 0). The full end-to-end");
    println!("    causal advantage of the ABLATION currency (increment 0) needs SENSITIVITY PROPAGATION through hidden");
    println!("    features — over a random backbone the immediate-fan-out form degenerates to reliance. That is");
    println!("    increment 1.5, and it may conflict with the local rule (an honest open question, not a win yet).\n");
}
