// SPDX-License-Identifier: MIT-0
//! rc6 step 2 — the ASYNC substrate: no tick. Units fire only on a changed input; a perturbation
//! cascades and SETTLES to a fixed point; and — the nature-like efficiency — a stable region does
//! ZERO work (empty computation is skipped), so events scale with the CHANGE, not the field size.
//!
//!   cargo test -p openmorph-core --test proofs substrate_async -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

/// A chain c0 → c1 → … → cn, each unit relaying tanh(1.5·prev).
fn chain(n: usize) -> Field {
    let mut f = Field::new(n + 1);
    for i in 0..n {
        f.add_unit(vec![i], vec![1.5], 0.0, i + 1);
    }
    f
}

#[test]
fn a_perturbation_cascades_and_settles_deterministically() {
    let mut f = chain(6);
    let events = f.stimulate(&[(0, 1.0)], 1000);
    assert!(
        f.settled(),
        "the cascade must reach a fixed point (empty queue)"
    );
    // Each cell downstream took its predecessor's tanh — a settled spread to the fixed point
    // of x = tanh(1.5·x) ≈ 0.86.
    assert!(
        f.cells()[1] > 0.8 && f.cells()[6] > 0.8,
        "signal spread to the end of the chain ({:?})",
        f.cells()
    );
    assert!(events >= 6, "at least one firing per unit ({events})");

    // Deterministic: a fresh identical field settles byte-identically.
    let mut g = chain(6);
    let _ = g.stimulate(&[(0, 1.0)], 1000);
    assert_eq!(f.cells(), g.cells(), "async settling is byte-reproducible");
    eprintln!("[SUBSTRATE] chain of 6: perturb c0=1.0 → cascade settled in {events} events, byte-reproducible, NO tick.");
}

#[test]
fn a_feedback_loop_settles_to_an_attractor() {
    // Two units in a positive-feedback loop: c0 ⇄ c1. It should settle to a saturated fixed point.
    let mut f = Field::new(2);
    f.add_unit(vec![1], vec![2.0], 0.0, 0); // c0 = tanh(2·c1)
    f.add_unit(vec![0], vec![2.0], 0.0, 1); // c1 = tanh(2·c0)
    let events = f.stimulate(&[(0, 0.3)], 10_000);
    assert!(
        f.settled(),
        "the loop must settle (not oscillate forever) within budget"
    );
    assert!(
        f.cells()[0] > 0.9 && f.cells()[1] > 0.9,
        "positive feedback → the +1 attractor (got {:?})",
        f.cells()
    );
    eprintln!("[SUBSTRATE] feedback loop c0⇄c1: nudged +0.3 → settled to the +attractor {:?} in {events} events.", f.cells());
}

#[test]
fn empty_computation_is_skipped_events_scale_with_change_not_size() {
    // 200 INDEPENDENT 5-unit chains = 1000 units. Perturbing ONE chain's input must fire only that
    // chain — the rest is a stable region that costs nothing (the nature-like sparse-compute win).
    const CHAINS: usize = 200;
    const LEN: usize = 5;
    let mut f = Field::new(CHAINS * (LEN + 1));
    for k in 0..CHAINS {
        let base = k * (LEN + 1);
        for i in 0..LEN {
            f.add_unit(vec![base + i], vec![1.5], 0.0, base + i + 1);
        }
    }
    let total_units = f.units.len();
    // Perturb chain 0's input only.
    let events = f.stimulate(&[(0, 1.0)], 10_000);
    assert!(f.settled(), "settled");
    eprintln!("[SUBSTRATE] {total_units} units across {CHAINS} chains; perturb ONE input → {events} firings (the other {} chains did ZERO work).", CHAINS - 1);
    eprintln!("[SUBSTRATE] VERDICT: async/event-driven is how nature works AND it saves compute globally — a stable region is never recomputed (empty computation skipped). Events scale with the CHANGE, not the field size — the foundation for a morph that thinks only where something is happening.");
    assert!(
        events <= LEN + 1,
        "only the perturbed chain fires ({events}), not all {total_units} units"
    );
}
