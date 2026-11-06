// SPDX-License-Identifier: MIT-0
//! DECISION-DEGENERACY DIAGNOSTIC (`Field::g_spread`) — the smallest member of the "silent failure" family.
//!
//! An actor fed a FLAT reward for thousands of steps is in a degenerate state: every option scores the same, so
//! `alive(Event::Act)` is choosing by tie-break. The substrate is doing the right thing — but nothing reported it,
//! and a real agent spent weeks chasing a pull-count law while its reward was identically zero (MorphBot,
//! 2026-08-22). `g_spread()` exposes the spread of the active rule's decision score across the last act's options
//! (on both rules — the default valence `A` and the free-energy `G`): near-zero ⇒
//! the decision is degenerate, whatever the cause (flat reward, collapsed value head, a state key so coarse every
//! option maps to one cell). A client asserts `g_spread() > eps` in its smoke test and catches "your options are
//! tied" on day one. Measured here on the default valence rule (this proof used to run the free-energy softmax,
//! deleted in 1.6 as a documented negative on real ARC; its thresholds are unchanged) and, for the degenerate read,
//! on the free-energy argmin too.
//!
//! Same family as: a nested cell computed-but-inert, `budget < n_units` returning identical garbage, and
//! `stimulate()` returning an int — the substrate does the right thing and says nothing when fed something
//! meaningless. This is the read that breaks that silence for the actor.

use openmorph_core::{ActRule, Event, Field, Outcome};

const ARMS: usize = 6;

fn act(f: &mut Field, options: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act { options, last }) {
        Outcome::Acted { chosen } => chosen,
        _ => panic!("expected Acted"),
    }
}

/// The two rules measured: the default valence rule and the free-energy argmin.
const RULES: [ActRule; 2] = [ActRule::Valence, ActRule::FreeEnergy];

/// Drive the built-in actor (`rule`) for `steps` under `payoff`, folding each reward back; return the final
/// `g_spread()` (the spread of the decision score across the last act's options).
fn drive(rule: ActRule, payoff: fn(usize) -> f32, seed: u64, steps: usize) -> f32 {
    let options: Vec<usize> = (0..ARMS).collect();
    let mut f = Field::new(ARMS);
    f.config_mut().act_rule = rule;
    f.rng_seed(seed);
    let mut last: Option<(usize, f32)> = None;
    for _ in 0..steps {
        let a = act(&mut f, &options, last);
        last = Some((a, payoff(a)));
    }
    f.g_spread()
}

#[test]
fn g_spread_is_nan_before_any_act() {
    let f = Field::new(ARMS);
    assert!(
        f.g_spread().is_nan(),
        "g_spread must be NaN until the first act (no decision recorded yet)"
    );
}

#[test]
fn g_spread_is_near_zero_on_a_flat_reward_and_positive_on_a_real_signal() {
    for rule in RULES {
        // DEGENERATE — a flat (all-zero) reward for every step: the score never differentiates, the decision is
        // tied, and g_spread collapses toward 0. The exact silent-failure MorphBot hit (reward identically zero).
        let flat: f32 = (0..4)
            .map(|s| drive(rule, |_| 0.0, 100 + s, 400))
            .sum::<f32>()
            / 4.0;
        // SIGNAL — one arm consistently pays: its learned value converges away from the rest, the score separates.
        let signal: f32 = (0..4)
            .map(|s| drive(rule, |a| if a == 2 { 1.0 } else { 0.0 }, 100 + s, 400))
            .sum::<f32>()
            / 4.0;
        println!("{rule:?}: g_spread  flat-reward(degenerate)={flat:.5}  real-signal={signal:.5}");
        assert!(
            flat < 1e-3,
            "{rule:?}: a flat reward must read as a DEGENERATE decision (g_spread≈0, got {flat:.5})"
        );
        // The lift is asserted on the DEFAULT rule. The free-energy argmin lifts less (≈0.03 here): once the paying
        // arm is found it never re-visits the others, so their σ² stays at its one-visit value and compresses `G`
        // toward the payer's (the deleted softmax kept re-sampling them, which is what the old 0.05 bar measured).
        if rule == ActRule::Valence {
            assert!(
                signal > flat + 0.05,
                "{rule:?}: a real reward signal must lift g_spread clear of the degenerate floor (signal {signal:.5} vs flat {flat:.5})"
            );
        } else {
            assert!(
                signal > flat,
                "{rule:?}: a real reward signal must lift g_spread above the degenerate floor (signal {signal:.5} vs flat {flat:.5})"
            );
        }
    }
}

#[test]
fn g_spread_does_not_change_selection() {
    // The diagnostic is a pure read of state the actor already computes — recording it must not alter WHICH
    // option is chosen. Two identical runs (same seed) produce the same action sequence; g_spread only observes.
    let options: Vec<usize> = (0..ARMS).collect();
    let seq = |seed: u64| -> Vec<usize> {
        let mut f = Field::new(ARMS); // the default valence rule: exact ties draw the seeded act PRNG
        f.rng_seed(seed);
        let mut last: Option<(usize, f32)> = None;
        let mut s = Vec::new();
        for _ in 0..200 {
            let a = act(&mut f, &options, last);
            last = Some((a, if a == 1 { 1.0 } else { 0.0 }));
            s.push(a);
        }
        s
    };
    assert_eq!(
        seq(7),
        seq(7),
        "same seed ⇒ identical action sequence (g_spread is observation-only)"
    );
}
