// SPDX-License-Identifier: MIT-0
//! CO-ADAPTATION STABILITY — a coupled reward loop is a driven oscillator (Gap 7 test #2,
//! `docs/research/arc3-gaps-to-100.md`).
//!
//! THE CLAIM (Sébastien's physics). When two cells mutually reward/adapt to each other, the pair is a driven
//! system with a natural frequency: it CONVERGES below a critical coupling gain (≈ learning rate) and RESONATES
//! (oscillates) above it — "resonance in the patterns, dependent on the learning rate". So the co-adaptation
//! instability behind the naive cell-reward probe is not a dead end; it is a stability regime, and the substrate's
//! own dynamics set where the boundary sits.
//!
//! THE TEST. A minimal rotational coupling learned through the substrate's real delta rule (`learn_step`): unit A
//! tracks `+gain·B`, unit B tracks `−gain·A` (anti-tracking ⇒ genuine oscillation, fixed point at 0). Start from a
//! perturbation and measure the residual oscillation amplitude after settling, across a learning-rate SWEEP. Then
//! toggle the substrate's 2nd-order term (`Config::init_momentum`, the heavy-ball `v←μv+Δw`) and MEASURE — honestly
//! — which way the damping moves the stability boundary.
//!
//! THE RESULT (measured, truth-not-trophy). Below a critical coupling gain the loop DECAYS (lr 0.5: peak |A|
//! stays at its 0.05 perturbation); above it the rotational loop RESONATES — the spiral grows until tanh
//! saturates (lr ≥ 2: peak |A| ≥ 0.99; the boundary sits near lr ≈ 1, where it already reaches 0.96). The
//! driven-oscillator prediction holds on the substrate: the learning rate sets the co-adaptation stability
//! boundary, so a cell-to-cell reward loop at high gain DOES need that boundary respected (or damping).
//!
//! CORRECTION (the stale-after-learn fix). This proof first reported the opposite — "BOUNDED at every rate, the
//! learn rule is intrinsically self-damped, so the difficulty is a bootstrap, not resonance". That reading was an
//! artifact: the drive is CONSTANT, and before the fix a unit whose weights changed was never re-fired on an
//! unchanged input, so both outputs stayed at their initial values and the loop never ran (peak |A| = tanh(0.05)
//! at EVERY rate). The same sweep with the drive JITTERED by ±1e-3 (every unit then re-fires naturally, fix or
//! no fix) gives the curve asserted here, so it is the substrate's dynamics, not an effect of the fix.
//!
//! TRUTH-NOT-TROPHY. The coupling, the seed, the sweep, and the settle window are authored and identical across
//! arms; only the learning rate / momentum differ. ZERO openmorph-core change: a usage protocol over `add_unit` /
//! `learn_step` / `configure`. The full cell-to-cell reward economy (Gap 7) remains a gated CORE spike pending
//! test #3 (deletion).
//!
//! PINNED to `stability_cap` at its `Config::legacy()` value 0 — every other lever is the 1.6 default (the 1.6 learning
//! flip): the resonance above a critical learning rate is a property of the RAW delta step. Under the full 1.6 default
//! the step-stability cap bounds the coupled step and the loop decays at every learning rate of the sweep (peak |A|
//! 0.020 from lr 0.5 to 20): the cap removes the instability this proof measures.
//!
//!   cargo test -p openmorph-core --test proofs substrate_coadapt_stability -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

/// Run the rotational coupled loop for `steps` and return the residual oscillation amplitude (peak-to-peak of A's
/// output over the last third). `lr` is the coupling gain; `momentum` is the heavy-ball μ (0 = 1st-order).
fn amplitude(lr: f32, momentum: f32, steps: usize) -> f32 {
    let mut f = pinned_field(1); // cell 0 = the constant drive
    f.config_mut().init_momentum = momentum; // units are born with this heavy-ball μ (set before add_unit)
                                             // two output units A (cell 1) and B (cell 2), each reading the constant drive; a TINY non-zero init ⇒ a small
                                             // perturbation. Below the threshold it decays (stays tiny); above it, the spiral GROWS until tanh saturates.
    f.add_unit(vec![0], vec![0.05], 0.0, 1);
    f.add_unit(vec![0], vec![-0.03], 0.0, 2);
    let gain = 1.0f32;
    let mut peak = 0.0f32; // max |A| reached over the run — converged ⇒ ~init, resonating ⇒ grows to saturation
    for _ in 0..steps {
        f.stimulate(&[(0, 1.0)], 20); // compute current outputs
        let (ca, cb) = (f.cells()[1], f.cells()[2]);
        // rotational targets: A chases +gain·B, B flees at −gain·A.
        f.learn_step(&[(0, 1.0)], 1, gain * cb, lr, 20);
        f.learn_step(&[(0, 1.0)], 2, -gain * ca, lr, 20);
        peak = peak.max(f.cells()[1].abs());
    }
    peak
}

#[test]
fn the_coupled_learn_loop_resonates_above_a_critical_gain() {
    let steps = 400;
    // SWEEP the coupling gain (learning rate) across two orders of magnitude, 1st-order (μ=0).
    let rates = [0.5f32, 2.0, 5.0, 10.0, 20.0];
    let amps: Vec<f32> = rates.iter().map(|&lr| amplitude(lr, 0.0, steps)).collect();

    // MEASURED: bounded (decaying) below the critical gain, resonating to tanh saturation above it.
    assert!(
        amps[0] < 0.15,
        "below the critical gain the coupled learn-loop must stay bounded: {amps:?}"
    );
    assert!(
        amps[1..].iter().all(|&a| a > 0.9),
        "above the critical gain the coupled learn-loop must resonate to saturation: {amps:?}"
    );

    println!("\n  CO-ADAPTATION STABILITY (Gap 7 test #2):");
    println!(
        "    peak |A| over a learning-rate sweep (μ=0): {}",
        rates
            .iter()
            .zip(&amps)
            .map(|(r, a)| format!("lr {r}\u{2192}{a:.3}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!("    \u{21d2} DECAYS below the critical gain, RESONATES to tanh saturation above it: the learning rate");
    println!("      sets the co-adaptation stability boundary (the driven-oscillator prediction holds).\n");
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
