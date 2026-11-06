// SPDX-License-Identifier: MIT-0
//! #150 — the CLOSED LOOP composes (MorphLab D1 `gate4_agent`), native: a fresh morph on a multi-object world
//! (avatar that moves per action + a goal + a static distractor — NOT told which is which) runs the whole loop:
//!   (A) DISCOVER WHAT IT CONTROLS — the action-conditional AGENCY signal (how much the action IMPROVES a move-
//!       prediction, per object) picks the avatar unprompted (high gap) vs static objects (≈0). Faculty 3.
//!   (B) LEARN its action-causal model, (C) PLAN to the goal by value-iteration over that model (faculty 5).
//! Composes perceive-what-I-control → learn-consequences → plan-to-goal. A usage protocol over `grow` +
//! `prefer`/`relax_values`/`value_of` — zero new core (grown-perception object features are the #145 follow-up).
use openmorph_core::Field;

use crate::common::Rng;

const N: usize = 6;
const DELTAS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

fn cell(r: i32, c: i32) -> usize {
    (r.clamp(0, N as i32 - 1) as usize) * N + c.clamp(0, N as i32 - 1) as usize
}
fn onehot(a: usize, with: bool) -> Vec<(usize, f32)> {
    (0..4)
        .map(|i| (i, if with && i == a { 1.0 } else { -1.0 }))
        .collect()
}

/// The AGENCY of an object = how much knowing the ACTION improves predicting its move. Grow a WITH-action and a
/// WITHOUT-action move-predictor from the object's (action → move) experience; return the accuracy gap.
/// `moves[a]` is the object's true move-delta under action `a` (a controllable object varies with `a`; a static
/// one is `(0,0)` for every `a`).
fn agency(moves: &[(i32, i32); 4], seed: u64) -> f32 {
    let train = |with: bool| -> Field {
        let mut f = Field::new(6);
        f.config_mut().max_hidden = 0; // action→move is linear in the one-hot
        let mut rng = Rng::new(seed);
        for _ in 0..8000 {
            let a = rng.below(4);
            let (dr, dc) = moves[a];
            f.grow(&onehot(a, with), 4, dr as f32, 0.1, 20);
            f.grow(&onehot(a, with), 5, dc as f32, 0.1, 20);
        }
        f
    };
    let acc = |f: &mut Field, with: bool| -> f32 {
        let mut ok = 0;
        for (a, &(dr, dc)) in moves.iter().enumerate() {
            let inp = onehot(a, with);
            f.stimulate(&inp, 20);
            if f.cells()[4].round() as i32 == dr && f.cells()[5].round() as i32 == dc {
                ok += 1;
            }
        }
        ok as f32 / 4.0
    };
    let (mut w, mut wo) = (train(true), train(false));
    (acc(&mut w, true) - acc(&mut wo, false)).max(0.0) // WITH − WITHOUT: the action-conditional gap
}

#[test]
fn a_fresh_morph_discovers_learns_and_plans() {
    // Three objects; the morph is NOT told which is the avatar. avatar moves per action; goal/distractor static.
    let avatar_moves = DELTAS; // move depends on the action
    let static_moves = [(0, 0); 4]; // a wall / the goal / a distractor doesn't move for any action
                                    // ── (A) DISCOVER — the action-conditional agency gap picks the avatar. ──
    let gap_avatar = agency(&avatar_moves, 0x2);
    let gap_static = agency(&static_moves, 0x2);
    eprintln!("[AGENT] (A) agency gap (WITH−WITHOUT action): avatar {gap_avatar:.2} vs static {gap_static:.2} → controllable = the higher");
    assert!(
        gap_avatar > 0.5 && gap_avatar > gap_static + 0.5,
        "the action-conditional gap must pick the avatar unprompted: avatar {gap_avatar:.2} vs static {gap_static:.2}"
    );

    // ── (B) LEARN the discovered avatar's action-causal model (action → delta). ──
    let mut model = Field::new(6);
    model.config_mut().max_hidden = 0;
    let mut rng = Rng::new(0x7);
    for _ in 0..12000 {
        let a = rng.below(4);
        let (dr, dc) = avatar_moves[a];
        model.grow(&onehot(a, true), 4, dr as f32, 0.1, 20);
        model.grow(&onehot(a, true), 5, dc as f32, 0.1, 20);
    }
    let learned: Vec<(i32, i32)> = (0..4)
        .map(|a| {
            model.stimulate(&onehot(a, true), 20);
            (
                model.cells()[4].round() as i32,
                model.cells()[5].round() as i32,
            )
        })
        .collect();
    assert_eq!(
        learned,
        avatar_moves.to_vec(),
        "the avatar model must be learned exactly"
    );

    // ── (C) PLAN — value-iterate over the LEARNED model, then walk the avatar to the goal. AGENT vs RANDOM. ──
    let (start, goal) = (0usize, N * N - 1);
    let mut planner = Field::new(N * N);
    let mut adj = vec![Vec::new(); N * N];
    for (s, adj_s) in adj.iter_mut().enumerate() {
        let (r, c) = ((s / N) as i32, (s % N) as i32);
        for &(dr, dc) in &learned {
            adj_s.push(cell(r + dr, c + dc));
        }
    }
    planner.prefer(goal, 0.0);
    planner.relax_values(&adj, 1.0, 0);
    // AGENT: descend the value field via the learned model
    let (mut s, mut steps) = (start, 0usize);
    while s != goal && steps < 100 {
        let (r, c) = ((s / N) as i32, (s % N) as i32);
        let a = (0..4)
            .min_by(|&a, &b| {
                let (ar, ac) = learned[a];
                let (br, bc) = learned[b];
                planner
                    .value_of(cell(r + ar, c + ac))
                    .total_cmp(&planner.value_of(cell(r + br, c + bc)))
            })
            .unwrap();
        let (dr, dc) = DELTAS[a];
        s = cell(r + dr, c + dc);
        steps += 1;
    }
    let opt = (N - 1) + (N - 1);
    assert_eq!(s, goal, "the AGENT must reach the goal");
    assert_eq!(
        steps, opt,
        "the AGENT must plan the optimal path: {steps} vs {opt}"
    );
    // RANDOM baseline — reach-rate within a FAIR budget (3× the optimal path), over 20 runs.
    let budget = 3 * opt;
    let mut reached = 0;
    for seed in 0..20u64 {
        let mut rng = Rng::new(0x3A + seed);
        let (mut rs, mut rsteps) = (start, 0usize);
        while rs != goal && rsteps < budget {
            let (r, c) = ((rs / N) as i32, (rs % N) as i32);
            let (dr, dc) = DELTAS[rng.below(4)];
            rs = cell(r + dr, c + dc);
            rsteps += 1;
        }
        if rs == goal {
            reached += 1;
        }
    }
    let rand_rate = 100.0 * reached as f32 / 20.0;
    assert!(
        rand_rate < 60.0,
        "a random walk must mostly FAIL within the agent's time budget ({budget} steps): {rand_rate:.0}%"
    );
    eprintln!("[AGENT] (B/C) learned the avatar model {learned:?}; AGENT reached the goal OPTIMALLY in {steps}={opt} steps; RANDOM within {budget} steps: {rand_rate:.0}% of 20 runs");

    eprintln!("[AGENT] VERDICT: the #150 closed loop COMPOSES, native (MorphLab D1) — a fresh morph, NOT told which object it controls, (A) DISCOVERS the avatar by the action-conditional agency gap ({gap_avatar:.2} vs static {gap_static:.2}), (B) LEARNS its action-causal model exactly, and (C) PLANS to the goal by value-iteration over that learned model — OPTIMAL ({steps} steps) where a random walk reaches only {rand_rate:.0}% within 3× the budget. perceive-what-I-control → learn-consequences → plan-to-goal, as a usage protocol over grow + prefer/relax_values/value_of (zero new core). Honest scope: HAND object perception on a synthetic world (grown perception + real ls20 = the #145 acceptance bar); value-iteration is the native planner (faculty 5). The agency signal is faculty 3's mechanism, validated composing.");
}
