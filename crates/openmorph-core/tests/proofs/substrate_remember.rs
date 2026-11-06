// SPDX-License-Identifier: MIT-0
//! GROWING THE MEMORY — causality supplies credit-through-time (MorphLab #115 + Sébastien's insight, the
//! close of `substrate_recurrent`'s open frontier). `substrate_recurrent` showed the cause-in-the-past needs
//! MEMORY-OF-VALUE and that the substrate can REPRESENT a latch, but flagged GROWING it as open: the local
//! per-event delta rule can't carry a terminal reward back through the delay (credit-through-time).
//!
//! The resolution is not to solve backprop-through-time — it's that **causal state-identification supplies the
//! per-step target for free.** `substrate_causal`'s determinism metric already *identifies* memory-of-value as
//! the causally-sufficient state (1.00 vs memoryless/recency 0.00). That identification says exactly what the
//! memory must hold at every step — the cue's value — which is the DENSE per-step target the local rule needs.
//! Credit never has to travel through time; the causal state names the target directly.
//!
//! Two regimes on the same recurrent latch (a unit reading `[cue, self]`, grown from zero weights):
//!   - **B — terminal signal only:** correct the memory only at the final step (the outcome). The delta rule
//!     has no per-step target while the value must be held → the latch never forms. Chance.
//!   - **C — dense causal target:** the causal identification supplies the per-step target (hold the cue's
//!     value). The SAME local delta rule now grows the latch — load then hold — to hold the value across the
//!     delay. Solved, no backprop-through-time.
//!
//! So the temporal half is not a separate mechanism: it is `substrate_causal`'s residual→grow loop, extended
//! so the birthed feature can be a MEMORY cell, with the determinism metric supplying its per-step target.
//!
//!   cargo test -p openmorph-core --test proofs substrate_remember -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const DELAY: usize = 6;

/// Build a recurrent latch: one unit reading `[cue, self]` and writing `self`, weights grown from ZERO.
/// `cue` = cell 0, `self`/memory = cell 1.
fn latch() -> Field {
    let mut f = Field::new(2);
    f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, 1); // memory = tanh(w_cue·cue + w_self·self)
    f
}

/// Run one episode's stimulation (cue flashes, then gone for DELAY), returning the memory sign at the end.
fn rollout(f: &mut Field, c: f32) -> f32 {
    f.stimulate(&[(0, c)], 16); // load
    for _ in 0..DELAY {
        f.stimulate(&[(0, 0.0)], 16); // hold
    }
    f.cells()[1]
}

/// Test how often the grown latch holds the cue's sign across the delay.
fn holds(f: &mut Field, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let (mut ok, mut tot) = (0.0f64, 0.0);
    for _ in 0..1000 {
        let c = rng.bit();
        if rollout(f, c).signum() == c {
            ok += 1.0;
        }
        tot += 1.0;
    }
    ok / tot
}

#[test]
fn causal_identification_supplies_the_dense_target_that_grows_the_memory() {
    // ── Regime B — TERMINAL signal only: correct the memory just at the final step. ──
    let mut b = latch();
    let mut rng = Rng::new(0xB000);
    for _ in 0..30000 {
        let c = rng.bit();
        // Roll the episode, correcting ONLY the last step toward the outcome (terminal reward).
        b.stimulate(&[(0, c)], 16);
        for _ in 0..DELAY - 1 {
            b.stimulate(&[(0, 0.0)], 16);
        }
        b.learn_step(&[(0, 0.0)], 1, c, 0.1, 16); // only the final step carries a target
    }
    let b_holds = holds(&mut b, 0xB1);

    // ── Regime C — DENSE CAUSAL target: the determinism metric identifies memory-of-value as the sufficient
    // state, so its per-step target is the held cue value. Correct the memory at EVERY step toward it. ──
    let mut c_field = latch();
    let mut rng = Rng::new(0xC000);
    for _ in 0..30000 {
        let c = rng.bit();
        // The causal identification's per-step target for the memory is `c` (the value that makes the world
        // deterministic) — supplied at the load step AND every hold step. Same local delta rule as everywhere.
        c_field.learn_step(&[(0, c)], 1, c, 0.1, 16); // load: hold the cue
        for _ in 0..DELAY {
            c_field.learn_step(&[(0, 0.0)], 1, c, 0.1, 16); // hold: keep holding it
        }
    }
    let c_holds = holds(&mut c_field, 0xC1);

    eprintln!(
        "[REMEMBER] growing the latch for the cause-in-the-past (cue gone {DELAY} steps):\n  Regime B — terminal signal only : holds the value {:.0}%  (credit can't travel back through the delay)\n  Regime C — dense CAUSAL target  : holds the value {:.0}%  (causal identification names the per-step target)",
        b_holds * 100.0,
        c_holds * 100.0,
    );
    eprintln!("[REMEMBER] VERDICT: the recurrent frontier closes without solving backprop-through-time. `substrate_causal`'s determinism metric identifies memory-of-value as the causally-sufficient state — and that identification IS the dense per-step target the local delta rule needs (the memory must hold the cue's value at every step). With only a terminal signal the latch never forms (Regime B, chance); with the causal target supplied per step the SAME local rule grows the latch — learning to LOAD (w_cue) then HOLD (w_self) — and carries the value across the delay (Regime C, ~100%). So the temporal half is not a new mechanism: it is `substrate_causal`'s residual→grow loop extended to birth a MEMORY cell, with the causal state supplying its target. Causality provides the credit, so credit-through-time is never solved blindly — exactly Sébastien's point.");

    assert!(
        b_holds < 0.65,
        "terminal-only signal cannot grow the latch (holds {b_holds:.2})"
    );
    assert!(
        c_holds > 0.95,
        "the dense causal target grows the latch via the local rule (holds {c_holds:.2})"
    );
}
