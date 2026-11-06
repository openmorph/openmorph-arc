// SPDX-License-Identifier: MIT-0
//! #150 — native hypothesis-testing, PHASE 1: reuse the EXISTING value/goal system, ZERO new primitive.
//! MorphLab showed a cold-start morph random-walks on deep ARC-3 because it lacks (1) negative-reward memory,
//! (2) commitment, (3) hypothesis-from-a-library. This proves the first two fall out of the value field you
//! already built (`prefer`/`value_of`/`relax_values`) — no new core primitive — and that the reward is
//! SELF-GENERATED from the goal.
//!   (A) SELF-REWARD FROM THE GOAL — `relax_values` is cost-to-go to the `prefer`red goal, so a step's reward =
//!       `value_of(before) − value_of(after)`: POSITIVE toward the goal, NEGATIVE away. A goal-derived ±reward
//!       the morph makes itself (no external signal), and descending it reaches the goal (planning).
//!   (B) NEG-MEMORY makes the search LINEAR not exponential — on a hidden L-step plan (Kᴸ plans), remembering a
//!       failed (step,action) as a persistent repeller (`prefer(+BIG)`) + committing the winners turns a random
//!       ~Kᴸ search into ~K·L. Reproduces MorphLab's gate4_policy_loop, on the value field alone.
use openmorph_core::substrate::Field;

use crate::common::Rng;

const K: usize = 5; // options per step
const BIG: f32 = 1e9; // an aversive clamp: a persistent repeller the argmin policy never picks

#[test]
fn value_field_gives_self_reward_and_neg_memory_beats_random() {
    // ── (A) SELF-REWARD FROM THE GOAL — the value field's cost-to-go IS a goal-derived ±reward. ──
    let n = 8usize; // a line of states 0..7, the goal at 7
    let adj: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut v = Vec::new();
            if i > 0 {
                v.push(i - 1);
            }
            if i + 1 < n {
                v.push(i + 1);
            }
            v
        })
        .collect();
    let mut f = Field::new(n);
    f.prefer(n - 1, 0.0); // the GOAL: a preferred state clamped at value 0 (low = good)
    f.relax_values(&adj, 1.0, 0); // Bellman cost-to-go field to the goal

    // self-reward of a step u→v = value_of(u) − value_of(v): + toward the goal, − away from it.
    let self_reward = |f: &Field, u: usize, v: usize| f.value_of(u) - f.value_of(v);
    for i in 0..n - 1 {
        assert!(
            self_reward(&f, i, i + 1) > 0.0,
            "a step TOWARD the goal must be +self-reward (state {i}→{}): {}",
            i + 1,
            self_reward(&f, i, i + 1)
        );
        assert!(
            self_reward(&f, i + 1, i) < 0.0,
            "a step AWAY from the goal must be −self-reward"
        );
    }
    // descending the self-reward (greedy: the neighbour of lowest cost-to-go) reaches the goal — planning.
    let (mut s, mut steps) = (0usize, 0usize);
    while s != n - 1 && steps < 100 {
        s = *adj[s]
            .iter()
            .min_by(|&&a, &&b| f.value_of(a).total_cmp(&f.value_of(b)))
            .unwrap();
        steps += 1;
    }
    assert_eq!(s, n - 1, "self-reward descent must reach the goal");
    eprintln!("[HYP] (A) self-reward from goal: +toward / −away over the cost-to-go field; descent reaches the goal in {steps} steps (optimal {})", n - 1);

    // ── (B) NEG-MEMORY vs RANDOM on a hidden L-step plan (Kᴸ plans) — linear vs exponential. ──
    eprintln!(
        "[HYP] (B) hidden L-step plan, K={K} options/step (Kᴸ plans): plan-attempts to first solve"
    );
    let mut rows = Vec::new();
    for &l in &[4usize, 5, 6] {
        // average over many seeds so the random column (geometric variance ~Kᴸ) is smooth, not a lucky run
        let seeds: Vec<u64> = (0..24u64).map(|i| 0x1500 + i * 13).collect();
        let avg = |neg: bool| -> f64 {
            seeds
                .iter()
                .map(|&sd| episodes_to_solve(l, neg, sd) as f64)
                .sum::<f64>()
                / seeds.len() as f64
        };
        let (rand, neg) = (avg(false), avg(true));
        let space = (K as f64).powi(l as i32);
        eprintln!("[HYP]   L={l}: search space Kᴸ={space:.0}   RANDOM {rand:>8.0} attempts   NEG-MEMORY {neg:>5.1}   ({:.0}× fewer)", rand / neg);
        rows.push((l, rand, neg));
    }
    // NEG-MEMORY is ~K·L (linear): bounded by a small constant × K·L at every L.
    for (l, _rand, neg) in &rows {
        assert!(
            *neg <= (2 * K * *l) as f64,
            "neg-memory must be ~linear (≤2·K·L={}) at L={l}: got {neg:.1}",
            2 * K * *l
        );
    }
    // and the speedup over random GROWS with depth (random is exponential, neg-memory linear).
    let speedup = |r: &(usize, f64, f64)| r.1 / r.2;
    assert!(
        speedup(&rows[2]) > speedup(&rows[0]) * 3.0,
        "the neg-memory advantage must GROW with plan depth (random exponential, neg-memory linear): L=6 {:.0}× vs L=4 {:.0}×",
        speedup(&rows[2]),
        speedup(&rows[0])
    );

    eprintln!("[HYP] VERDICT: #150's first two faculties fall out of the EXISTING value/goal system — ZERO new primitive. (A) `relax_values` cost-to-go IS a SELF-REWARD from the goal: value_of(before)−value_of(after) is +toward / −away, and descending it plans a path to the goal. (B) NEGATIVE-REWARD MEMORY is a persistent aversive clamp (`prefer(+BIG)` on the failed (step,action) cell, reusing the value store) + committing winners — turning a random ~Kᴸ plan search into ~K·L (the cold-start wall, closed on the value field alone). Next: hypothesis-over-library (route+nest) and wiring the loop native into alive().");
}

/// Solve a hidden L-step plan (correct[d] ∈ 0..K). One EPISODE = one plan attempt from depth 0: pick an action
/// per depth, advance on the correct one (self-reward + toward the goal), fail on a wrong one (dead-end). With
/// `neg_memory`, a failed (depth,action) is clamped `prefer(+BIG)` (a persistent repeller) and a winning action
/// is COMMITTED (win-stay). Returns the number of episodes to first reach depth L (the goal). Pure value-field.
fn episodes_to_solve(l: usize, neg_memory: bool, seed: u64) -> usize {
    let mut rng = Rng::new(seed);
    let correct: Vec<usize> = (0..l).map(|_| rng.below(K)).collect();
    let mut f = Field::new(l * K + 1);
    let goal = l * K;
    f.prefer(goal, 0.0); // the goal: reaching depth L
    let mut committed: Vec<Option<usize>> = vec![None; l]; // win-stay: the known-correct action per depth
    let mut episodes = 0usize;
    loop {
        episodes += 1;
        let mut depth = 0usize;
        while depth < l {
            let base = depth * K; // (depth,action) cell = base + action
            let action = if neg_memory {
                // COMMIT the winner (win-stay), else pick the first action NOT aversively clamped.
                match committed[depth] {
                    Some(a) => a,
                    None => (0..K)
                        .find(|&a| f.value_of(base + a) < BIG * 0.5)
                        .unwrap_or_else(|| rng.below(K)),
                }
            } else {
                rng.below(K) // RANDOM — no memory, no commitment: re-picks (and re-fails) every episode
            };
            if action == correct[depth] {
                if neg_memory {
                    committed[depth] = Some(action); // advanced toward the goal — commit this step
                }
                depth += 1;
            } else {
                if neg_memory {
                    f.prefer(base + action, BIG); // AVERSIVE MEMORY — never take this (depth,action) again
                }
                break; // dead-end (−self-reward) — restart the plan
            }
        }
        if depth == l {
            return episodes;
        }
        if episodes > 2_000_000 {
            return episodes; // safety cap
        }
    }
}
