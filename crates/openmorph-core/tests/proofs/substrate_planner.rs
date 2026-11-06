// SPDX-License-Identifier: MIT-0
//! #150 faculty 5 — the model-based PLANNER (B1's validated capability, native as a usage protocol). A morph
//! LEARNS an action-conditional world-model (predict the move from the action — agency), then VALUE-ITERATES over
//! its OWN IMAGINED transitions (`relax_values`) to reach a goal near-optimally — a Dyna loop over `grow` +
//! `prefer`/`relax_values`/`value_of`, no external transition table, no new core primitive.
//!   (A) AGENCY — the world-model needs the ACTION to predict the move: WITH-action ~100%, WITHOUT ~chance.
//!   (B) PLAN — value-iteration over the LEARNED model reaches the goal near-optimally; a model-free random walk
//!       takes an order of magnitude more steps. And it REPLANS when the goal moves (Dyna's point).
use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 6; // N×N grid world
const DELTAS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // up, down, left, right

fn cell(r: i32, c: i32) -> usize {
    (r.clamp(0, N as i32 - 1) as usize) * N + c.clamp(0, N as i32 - 1) as usize
}

/// LEARN the world-model: predict the move-delta (dr,dc) from the action. `with_action` gates whether the model
/// even sees the action (the agency ablation). Returns the grown Field (cells 0..3 = action one-hot, 4=dr, 5=dc).
fn learn_model(with_action: bool, seed: u64) -> Field {
    let mut f = Field::new(6);
    f.config_mut().max_hidden = 0; // action→delta is LINEAR in the one-hot; a pure linear readout fits it cleanly
    let mut rng = Rng::new(seed);
    for _ in 0..12000 {
        let a = rng.below(4);
        let (dr, dc) = DELTAS[a];
        // action as a ±1 one-hot (B1's fix); without_action ⇒ all −1 (no distinguishing info).
        let inp: Vec<(usize, f32)> = (0..4)
            .map(|i| (i, if with_action && i == a { 1.0 } else { -1.0 }))
            .collect();
        f.grow(&inp, 4, dr as f32, 0.1, 20); // predict the intended dr (world clamps at the boundary, not the model)
        f.grow(&inp, 5, dc as f32, 0.1, 20); // predict the intended dc
    }
    f
}

/// The model's predicted integer delta for action `a`.
fn model_delta(m: &mut Field, a: usize) -> (i32, i32) {
    let inp: Vec<(usize, f32)> = (0..4)
        .map(|i| (i, if i == a { 1.0 } else { -1.0 }))
        .collect();
    m.stimulate(&inp, 20);
    (m.cells()[4].round() as i32, m.cells()[5].round() as i32)
}

/// PLAN over the LEARNED model: build imagined adjacency (state×action → predicted next), value-iterate to the
/// goal, then walk GREEDILY in the real world. Returns steps to reach the goal (None if it doesn't).
fn plan_and_walk(m: &mut Field, start: usize, goal: usize) -> Option<usize> {
    // learned deltas per action (state-independent; the world does the clamping)
    let learned: Vec<(i32, i32)> = (0..4).map(|a| model_delta(m, a)).collect();
    // imagined transition graph from the model
    let mut adj = vec![Vec::new(); N * N];
    for (s, adj_s) in adj.iter_mut().enumerate() {
        let (r, c) = ((s / N) as i32, (s % N) as i32);
        for &(dr, dc) in &learned {
            adj_s.push(cell(r + dr, c + dc));
        }
    }
    let mut planner = Field::new(N * N);
    planner.prefer(goal, 0.0);
    planner.relax_values(&adj, 1.0, 0); // cost-to-go over the IMAGINED dynamics

    // greedy descent in the REAL world (true deltas)
    let mut s = start;
    for step in 0..(4 * N * N) {
        if s == goal {
            return Some(step);
        }
        let (r, c) = ((s / N) as i32, (s % N) as i32);
        // pick the action whose IMAGINED next-cell has the lowest cost-to-go
        let a = (0..4)
            .min_by(|&a, &b| {
                let (adr, adc) = learned[a];
                let (bdr, bdc) = learned[b];
                planner
                    .value_of(cell(r + adr, c + adc))
                    .total_cmp(&planner.value_of(cell(r + bdr, c + bdc)))
            })
            .unwrap();
        let (dr, dc) = DELTAS[a]; // the REAL world applies the true delta
        s = cell(r + dr, c + dc);
    }
    None
}

#[test]
fn planner_over_a_learned_model_reaches_goals() {
    // ── (A) AGENCY — the model needs the action to predict the move. ──
    let mut with = learn_model(true, 0x9);
    let mut without = learn_model(false, 0x9);
    let acc = |m: &mut Field| -> f32 {
        let mut ok = 0;
        for (a, &delta) in DELTAS.iter().enumerate() {
            if model_delta(m, a) == delta {
                ok += 1;
            }
        }
        100.0 * ok as f32 / 4.0
    };
    let (with_acc, without_acc) = (acc(&mut with), acc(&mut without));
    eprintln!("[PLAN] (A) action-conditional model: WITH-action {with_acc:.0}% vs WITHOUT-action {without_acc:.0}% (the action carries agency)");
    assert!(
        with_acc >= 100.0,
        "the model must learn action→move (agency): WITH-action {with_acc:.0}%"
    );
    assert!(
        without_acc <= 50.0,
        "without the action the model can't predict the move (chance): {without_acc:.0}%"
    );

    // ── (B) PLAN — value-iteration over the LEARNED model reaches goals near-optimally vs a random walk. ──
    let trials = [(0usize, 35usize), (5, 30), (2, 33), (30, 5)];
    let mut rng = Rng::new(0x71A);
    let (mut plan_tot, mut opt_tot, mut rand_tot) = (0usize, 0usize, 0usize);
    for &(start, goal) in &trials {
        let steps = plan_and_walk(&mut with, start, goal).expect("planner must reach the goal");
        let opt = ((start / N) as i32 - (goal / N) as i32).unsigned_abs() as usize
            + ((start % N) as i32 - (goal % N) as i32).unsigned_abs() as usize; // Manhattan optimum
                                                                                // model-free random-walk baseline
        let mut s = start;
        let mut rsteps = 0;
        while s != goal && rsteps < 4000 {
            let (r, c) = ((s / N) as i32, (s % N) as i32);
            let (dr, dc) = DELTAS[rng.below(4)];
            s = cell(r + dr, c + dc);
            rsteps += 1;
        }
        eprintln!("[PLAN]   start {start} → goal {goal}: planner {steps} steps (optimal {opt}), random walk {rsteps}");
        plan_tot += steps;
        opt_tot += opt;
        rand_tot += rsteps;
    }
    assert_eq!(
        plan_tot, opt_tot,
        "planning over the learned model must be OPTIMAL (Manhattan): {plan_tot} vs optimal {opt_tot}"
    );
    assert!(
        rand_tot > 5 * plan_tot,
        "the model-based planner must vastly beat a model-free random walk: {plan_tot} vs {rand_tot}"
    );

    // ── (B2) REPLAN — the goal moves; re-value-iterate over the same learned model and still reach it. ──
    let replanned = plan_and_walk(&mut with, 0, 20).expect("planner must reach the MOVED goal");
    let replan_opt = 20 / N + 20 % N; // Manhattan from 0 to 20
    eprintln!("[PLAN] (B2) replan to a MOVED goal (0→20): {replanned} steps (optimal {replan_opt}) — same model, new value field");
    assert_eq!(replanned, replan_opt, "replanning must stay optimal");

    eprintln!("[PLAN] VERDICT: #150 faculty 5 — the model-based PLANNER, native as a usage protocol (no new core). (A) the morph GROWS an action-conditional world-model: it needs the action to predict the move ({with_acc:.0}% with vs {without_acc:.0}% without = agency, B1). (B) it then VALUE-ITERATES (`relax_values`) over its OWN IMAGINED transitions and walks greedily to the goal — OPTIMAL ({plan_tot}={opt_tot} Manhattan steps) where a model-free random walk wanders {rand_tot} steps; and it REPLANS to a moved goal on the same learned model. This is B1's validated capability closed as a Dyna loop over `grow` + `prefer`/`relax_values`/`value_of`. Next: wire it + the aversive `drive` into a native `alive(Event::Act)`.");
}
