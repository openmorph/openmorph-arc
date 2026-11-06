// SPDX-License-Identifier: MIT-0
//! rc6/rc7 — ASYNC ACTING/RL, with NO new machinery. Acting is just grow-and-learn on a REWARD
//! channel: the morph settles on the state, picks an action (ε-greedy over the settled action cells),
//! observes reward, and grows/tunes a value-unit for the TAKEN action toward that reward. Over a
//! stream it learns the policy — event-driven, per-event, no tick, no separate acting ecology.
//!
//!   cargo test -p openmorph-core --test proofs substrate_act -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const NC: usize = 4; // contexts (states) — one-hot, cells 0..NC
const NA: usize = 3; // actions — value cells NC..NC+NA

fn best_action(context: usize) -> usize {
    // an arbitrary context→best-action map the morph must discover
    [2usize, 0, 1, 2][context]
}

/// The state one-hot for a context, as field inputs.
fn state(context: usize) -> Vec<(usize, f32)> {
    (0..NC)
        .map(|c| (c, if c == context { 1.0 } else { 0.0 }))
        .collect()
}

/// Greedy action = argmax over the settled action-value cells.
fn greedy(f: &mut Field, context: usize) -> usize {
    f.stimulate(&state(context), 200);
    (0..NA)
        .max_by(|&a, &b| {
            f.cells()[NC + a]
                .partial_cmp(&f.cells()[NC + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

#[test]
fn the_async_morph_learns_a_policy_from_reward_no_acting_machinery() {
    let mut f = Field::new(NC + NA);
    f.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    f.config_mut().birth_gate = 0.05;
    let mut rng = Rng::new(0xAC7);

    // BASELINE (control): the greedy policy of an UNTRAINED field — a real floor the learning must beat, so
    // the pass is attributable to reward-learning and not to the task being trivial. An untrained field's
    // action cells are ~0, so greedy is degenerate; a uniformly random policy would average only NC/NA.
    let untrained_correct = (0..NC)
        .filter(|&c| greedy(&mut f, c) == best_action(c))
        .count();

    // STREAM of experience: settle → ε-greedy act → reward → grow/tune the taken action's value.
    for _ in 0..8000 {
        let ctx = rng.below(NC);
        f.stimulate(&state(ctx), 200);
        // ε-greedy over the settled action values.
        let action = if rng.f() < 0.25 {
            rng.below(NA)
        } else {
            (0..NA)
                .max_by(|&a, &b| {
                    f.cells()[NC + a]
                        .partial_cmp(&f.cells()[NC + b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        };
        let reward = if action == best_action(ctx) {
            1.0
        } else {
            -1.0
        };
        // Acting = grow-and-learn on the reward channel: the TAKEN action's value moves toward reward.
        f.grow(&state(ctx), NC + action, reward, 0.1, 200);
    }

    // Greedy policy accuracy: does the morph pick the best action in every context?
    let correct = (0..NC)
        .filter(|&c| greedy(&mut f, c) == best_action(c))
        .count();
    eprintln!(
        "[ACT] contextual bandit (no acting machinery — just grow_and_learn on a reward channel): untrained baseline {untrained_correct}/{NC} → learned greedy policy {correct}/{NC} contexts; grew {} value-units (expected random policy ≈ {:.1}/{NC}).",
        f.n_units(),
        NC as f32 / NA as f32
    );
    eprintln!("[ACT] VERDICT: the async morph LEARNS a policy from reward with NO separate acting ecology — acting is grow-and-learn on a REWARD channel (grow a value for the taken action, tune toward reward), event-driven, no tick. The learned policy is perfect where the untrained/random floor is not, so the pass is reward-learning, not a trivial task. The RL/ARC-3 differentiator on the async substrate, from the same one mechanism.");
    assert_eq!(
        correct, NC,
        "the morph must learn the best action in every context"
    );
    assert!(
        correct > untrained_correct,
        "learning must beat the untrained-policy floor (learned {correct} vs untrained {untrained_correct} of {NC})"
    );
}
