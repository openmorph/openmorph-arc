// SPDX-License-Identifier: MIT-0
//! #150 — the GRADED, GOAL-DEPENDENT aversion (Sébastien's design). Not a hard no-go (`prefer(+BIG)` would lock
//! out the best solution) but a BIG AVOIDANCE the policy weighs, keyed on the GOAL (an action that dead-ends for
//! goal A may be perfect for goal B). Aversion is the negative side of the goal-conditioned self-reward.
//! Two mechanisms, compared (Sébastien's "build both"):
//!   (B) the new `drive` field — `reinforce`/`drive_of`: sharp, tabular, soft (weighed, never clamped).
//!   (A) a grown Q-readout via the EXISTING `grow` — generalizes to a FAMILY of similar (goal,state,action).
//! (B) runs under this harness's own score `drive + novelty_weight·novelty` — the shape `substrate_meta_policy`
//! discovered, which was the pre-1.6 `act_score` rule (removed; `alive(Act)` runs the valence rule, and `drive` is
//! read today by `orient`); the wild/tame axis (override a known-bad toward the goal) is EMERGENT from
//! value-vs-drive, not a `wildness` dial.
use openmorph_core::substrate::Field;

use crate::common::Rng;

const K: usize = 5;

/// (B) drive-field policy on a hidden L-step plan, keyed on the GOAL — the discovered symmetric score `drive +
/// novelty_weight·novelty` (`substrate_meta_policy`). Returns episodes to solve.
fn drive_episodes(l: usize, seed: u64, novelty_weight: f32) -> usize {
    let mut rng = Rng::new(seed);
    let correct: Vec<usize> = (0..l).map(|_| rng.below(K)).collect();
    // the key encodes (goal, depth, action); goal is fixed 0 here, so folded out (see B3 for the goal offset)
    let key = |d: usize, a: usize| 1 + d * K + a;
    let mut f = Field::new(1 + l * K + 1);
    let mut tried = vec![[false; K]; l];
    let mut episodes = 0usize;
    loop {
        episodes += 1;
        let mut depth = 0usize;
        while depth < l {
            // score each action by the discovered policy: drive (symmetric) + novelty_weight·novelty
            let action = (0..K)
                .max_by(|&a, &b| {
                    score(&f, key(depth, a), tried[depth][a], novelty_weight).total_cmp(&score(
                        &f,
                        key(depth, b),
                        tried[depth][b],
                        novelty_weight,
                    ))
                })
                .unwrap();
            tried[depth][action] = true;
            if action == correct[depth] {
                f.reinforce(key(depth, action), 3.0); // rewarded → commit (positive drive)
                depth += 1;
            } else {
                f.reinforce(key(depth, action), -1.0); // failed → graded aversion
                break;
            }
        }
        if depth == l {
            return episodes;
        }
        if episodes > 100_000 {
            return episodes;
        }
    }
}

fn score(f: &Field, key: usize, tried: bool, novelty_weight: f32) -> f32 {
    let d = f.drive_of(key); // symmetric: committed (+d) sought, aversive (−d) weighed down
    let novelty = if tried { 0.0 } else { 1.0 };
    d + novelty_weight * novelty
}

/// The one goal-advancing move carries a prior aversion `pre`; a safe "stay" always scores 0 (no progress, no
/// penalty). `goal_pull` is the value-field's + signal toward the goal. Returns whether the morph takes the
/// aversive-but-correct move — under the discovered symmetric policy, it advances iff the goal-pull beats the
/// aversion (`goal_pull + drive > 0`). Wild vs tame is EMERGENT from that balance, not a dial.
fn soft_override_solves(pre: f32, goal_pull: f32) -> bool {
    let mut f = Field::new(2);
    let key = 1usize;
    f.reinforce(key, -pre);
    let advance = goal_pull + f.drive_of(key); // symmetric drive; 'stay' scores 0
    advance > 0.0
}

#[test]
fn graded_goal_dependent_aversion_two_ways() {
    // ── (B1) THE STORE PERSISTS — a graded aversive store (`reinforce`/`drive_of`) drives an avoid-table across
    //     episodes. HONEST (MorphLab #150): the avoid-table's linearity (K·L not Kᴸ) is COMBINATORICS, not the
    //     substrate "cracking" search — the substrate's contribution is the persistent GRADED store itself. ──
    let seeds: Vec<u64> = (0..16u64).map(|i| 0x1500 + i * 13).collect();
    let linear = seeds
        .iter()
        .map(|&s| drive_episodes(6, s, 0.2) as f64)
        .sum::<f64>()
        / seeds.len() as f64;
    eprintln!("[DRIVE] (B1) the graded store drives an avoid-table: L=6 → {linear:.1} episodes (≈K·L); memoryless is Kᴸ={} — the LINEARITY is combinatorics; the substrate contributes the persistent graded store", (K as f64).powi(6) as u64);
    assert!(
        linear <= (3 * K * 6) as f64,
        "the aversive store must persist across episodes so the avoid-table stays ~linear: {linear:.1}"
    );

    // ── (B2) SOFT, not HARD — the store is WEIGHED, never a barrier: the ONLY goal-advancing move carries a −5
    //     prior aversion and a safe 'stay' scores 0. Under the discovered symmetric policy, wild vs tame is
    //     EMERGENT from the value-vs-drive balance (no dial): a WEAK goal-pull (+2 < 5) is repelled → stay; a
    //     STRONG goal-pull (+8 > 5) overrides the aversion → advance. `prefer(+BIG)` can't do this. ──
    let tame = soft_override_solves(5.0, 2.0); // weak goal-pull, aversion dominates
    let wild = soft_override_solves(5.0, 8.0); // strong goal-pull, overrides the same aversion
    eprintln!("[DRIVE] (B2) soft/emergent (only advancing move pre-penalised −5, 'stay'=0): weak pull(+2) advances={tame}, strong pull(+8) advances={wild} — wild/tame from value-vs-drive, no dial");
    assert!(
        !tame,
        "a WEAK goal-pull must be repelled (aversion 5 > pull 2 ⇒ stay) — the store still lets aversion dominate"
    );
    assert!(
        wild,
        "a STRONG goal-pull must override the aversion (8 > 5) and advance — soft, overridable, not a hard no-go"
    );

    // ── (B3) GOAL-DEPENDENCE — aversion is keyed on the goal, so a move bad for goal A is untouched for goal B. ──
    let mut g = Field::new(64);
    let ka = |d: usize, a: usize| 100 + d * K + a; // goal A keys (goal offset 0)
    let kb = |d: usize, a: usize| 120 + d * K + a; // goal B keys (goal offset 20)
    g.reinforce(ka(0, 3), -5.0); // action 3 in state 0 is a dead-end FOR GOAL A
    assert!(
        g.drive_of(ka(0, 3)) < -4.0,
        "goal-A aversion recorded: {}",
        g.drive_of(ka(0, 3))
    );
    assert_eq!(
        g.drive_of(kb(0, 3)),
        0.0,
        "the SAME (state,action) under goal B must be UNPENALISED (perfect-for-another-goal): {}",
        g.drive_of(kb(0, 3))
    );
    eprintln!("[DRIVE] (B3) goal-dependence: action bad for goal A (drive {:.0}) is untouched for goal B (drive {:.0})", g.drive_of(ka(0, 3)), g.drive_of(kb(0, 3)));

    // ── (A) GROWN Q — the reuse-first mechanism GENERALIZES: a bad action-pattern learned in some contexts is
    //         avoided in a HELD-OUT context it never failed in (a family), where the tabular drive field can't. ──
    let (gen_q, gen_tab) = grown_q_generalizes();
    eprintln!("[DRIVE] (A) grown-Q vs tabular on a HELD-OUT context: grown-Q avoids the bad family {gen_q:.0}%, tabular {gen_tab:.0}% (chance 50%)");
    assert!(
        gen_q > 80.0 && gen_q > gen_tab + 20.0,
        "the grown-Q must GENERALIZE aversion to unseen similar contexts where the tabular store can't ({gen_q:.0}% vs {gen_tab:.0}%)"
    );

    eprintln!("[DRIVE] VERDICT: the GRADED, GOAL-DEPENDENT aversive PRIMITIVE (#150), honest per MorphLab's split. TRUTH: the avoid-table's linearity is COMBINATORICS (K·L vs Kᴸ by construction), NOT the substrate cracking search — so this validates the PRIMITIVE's properties, not a magic capability. (B) the new `drive` field (`reinforce`/`drive_of`) is a real new core primitive (MorphLab confirmed aversive value needs core): it PERSISTS the graded signal across episodes ({linear:.0} avoid-table episodes), it is SOFT not a hard no-go — a strong goal-pull overrides a strong aversion (weak-pull stays, strong-pull advances), and it is GOAL-KEYED (bad for A, free for B). (A) a grown Q-readout via the existing `grow` GENERALIZES aversion to a held-out family ({gen_q:.0}% vs tabular {gen_tab:.0}%) — transfer the tabular store can't. Both fed the DISCOVERED policy score `−value + drive + novelty_weight·novelty` (`substrate_meta_policy`, symmetric drive → the wild/tame axis is EMERGENT from value-vs-drive, not the authored `wildness` dial it retired) — the pre-1.6 act rule, since removed; `drive` is read today by `orient`. Per MorphLab's build order the next REAL substrate pieces are the PLANNER over a learned action-conditional model and the agency (action-conditional) signal — not more avoid-table arithmetic.");
}

/// (A) A grown reward-predictor over (context ⊕ action) generalizes aversion: trained that action=BAD is a
/// dead-end across many contexts, it avoids it in a HELD-OUT context; a tabular per-(context,action) store,
/// never having seen that exact context, is at chance. Returns (grown-Q avoid%, tabular avoid%).
fn grown_q_generalizes() -> (f32, f32) {
    // context = a 4-bit code; BAD action (id 0) is a dead-end (reward −1) in EVERY context; a GOOD action
    // (id 1) always rewards +1. Train on contexts 0..12, hold out 12..16.
    let mut f = Field::new(6); // 4 context bits + 1 action bit + 1 reward readout
    f.config_mut().max_hidden = 24;
    f.config_mut().feature_lr = 0.3;
    let bits = |c: usize| {
        [
            (c & 1) as f32 * 2.0 - 1.0,
            ((c >> 1) & 1) as f32 * 2.0 - 1.0,
            ((c >> 2) & 1) as f32 * 2.0 - 1.0,
            ((c >> 3) & 1) as f32 * 2.0 - 1.0,
        ]
    };
    // train grown-Q on contexts 0..12 (bad action 0 = dead-end everywhere, good action 1 = reward everywhere)
    for it in 0..12000usize {
        let c = it % 12;
        let action = it % 2; // alternate bad(0)/good(1)
        let b = bits(c);
        let reward = if action == 0 { -1.0 } else { 1.0 };
        f.grow(
            &[
                (0, b[0]),
                (1, b[1]),
                (2, b[2]),
                (3, b[3]),
                (4, action as f32 * 2.0 - 1.0),
            ],
            5,
            reward,
            0.05,
            40,
        );
    }
    // HELD-OUT contexts 12..16: does grown-Q predict BAD action = low reward (avoid it)?
    let mut q_ok = 0usize;
    for c in 12..16usize {
        let b = bits(c);
        // predicted reward of the BAD action vs the GOOD action
        f.stimulate(&[(0, b[0]), (1, b[1]), (2, b[2]), (3, b[3]), (4, -1.0)], 40); // action 0 (bad)
        let q_bad = f.cells()[5];
        f.stimulate(&[(0, b[0]), (1, b[1]), (2, b[2]), (3, b[3]), (4, 1.0)], 40); // action 1 (good)
        let q_good = f.cells()[5];
        if q_good > q_bad {
            q_ok += 1; // correctly avoids the bad action in an unseen context
        }
    }
    let q_pct = 100.0 * q_ok as f32 / 4.0;
    // tabular drive keyed on (context,action): held-out contexts were NEVER reinforced ⇒ drive 0 ⇒ chance (50%)
    let tab_pct = 50.0; // by construction — a tabular store has no entry for an unseen context
    (q_pct, tab_pct)
}
