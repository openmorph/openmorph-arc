// SPDX-License-Identifier: MIT-0
//! CAUSALITY ⇄ STRATEGY — you must ACT to find the cause; the STRATEGY (how you act) decides whether you
//! can. Correlation is not causation: passive/correlated acting cannot separate a cause from a correlate.
//! The action IS the intervention (Pearl's `do`), and only acting that BREAKS the correlations reveals which
//! feature actually causes the outcome. This is the strategy→causality half of the coupled loop; the
//! causality→strategy half is that a causal (Markov) state is exactly what a strategy needs (see
//! `substrate_causal`, and the controllability filter that attends to the loci the action controls).
//!
//! Two features A and B. The outcome is caused by A alone (`outcome = A`). Under a CORRELATED strategy the
//! organism always sets A = B, so B correlates perfectly with the outcome — a spurious cause. A model
//! trained on that data can't tell A from B, and FAILS on interventional data (A ≠ B). Under an INDEPENDENT
//! strategy (diverse acting — the exploration a curiosity/go-explore actor provides) the correlation breaks,
//! B is revealed non-causal, and the model identifies A — generalizing to interventions. Acting the right
//! way is what makes the cause identifiable.
//!
//!   cargo test -p openmorph-core --test proofs substrate_intervene -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// The world: the outcome is caused by A alone. B is a bystander feature.
fn outcome(a: f32) -> f32 {
    a
}

/// Train a linear predictor of the outcome from (A, B) under a given acting STRATEGY, then test it on
/// INTERVENTIONAL data where A and B are set independently (A ≠ B half the time). Returns interventional
/// accuracy — does the learned model track the true cause A, or the spurious correlate B?
fn interventional_accuracy(correlated_strategy: bool, seed: u64) -> f64 {
    let mut f = Field::new(2);
    f.rng_seed(seed);
    f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, 2); // linear: out = tanh(wA·A + wB·B)
    let mut rng = Rng::new(seed);
    for _ in 0..6000 {
        let a = rng.bit();
        // CORRELATED strategy: always set B = A (a lazy policy that never decorrelates the features).
        // INDEPENDENT strategy: set B independently — diverse acting that breaks the correlation.
        let b = if correlated_strategy { a } else { rng.bit() };
        f.learn_step(&[(0, a), (1, b)], 2, outcome(a), 0.1, 200);
    }
    // INTERVENTIONAL test: A and B independent — the do-operator, breaking any training correlation.
    let (mut ok, mut tot) = (0.0, 0.0);
    for _ in 0..3000 {
        let (a, b) = (rng.bit(), rng.bit());
        f.stimulate(&[(0, a), (1, b)], 200);
        if f.cells()[2].signum() == outcome(a).signum() {
            ok += 1.0;
        }
        tot += 1.0;
    }
    ok / tot
}

#[test]
fn only_intervening_strategy_identifies_the_cause_not_the_correlate() {
    let seeds: Vec<u64> = (0..8).map(|s| 0x100 + s * 13).collect();
    let correlated = seeds
        .iter()
        .map(|&s| interventional_accuracy(true, s))
        .sum::<f64>()
        / seeds.len() as f64;
    let independent = seeds
        .iter()
        .map(|&s| interventional_accuracy(false, s))
        .sum::<f64>()
        / seeds.len() as f64;

    eprintln!(
        "[INTERVENE] identify the cause (outcome = A; B is a bystander), tested under INTERVENTION (A ≠ B):\n  CORRELATED strategy (A=B, passive/lazy): {:.0}%  → fooled by the correlate B\n  INDEPENDENT strategy (diverse acting)    : {:.0}%  → identifies the true cause A",
        correlated * 100.0,
        independent * 100.0,
    );
    eprintln!("[INTERVENE] VERDICT: causality and strategy are one coupled loop. You cannot find a cause by observing — you must ACT (intervene), and the STRATEGY decides whether you can: a correlated/passive policy never decorrelates the features, so a spurious correlate (B) is indistinguishable from the cause (A) and the model FAILS under intervention; an independent/diverse policy breaks the correlation, reveals B as non-causal, and identifies A — generalizing to interventions. The action is Pearl's `do`; diverse acting is how the organism runs its own experiments. This is the strategy→causality half; the causality→strategy half is that the Markov/causal state (grown in substrate_causal) + the controllability filter (attend where the action has leverage) are exactly what a strategy needs. Acting reveals causes; causes guide acting.");

    assert!(
        independent > 0.95,
        "independent (intervening) acting must identify the true cause (got {independent:.2})"
    );
    assert!(
        independent > correlated + 0.15,
        "the correlated strategy must be fooled by the correlate under intervention (indep {independent:.2} vs corr {correlated:.2})"
    );
}
