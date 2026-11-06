// SPDX-License-Identifier: MIT-0
//! #150 — the native policy loop through `alive(Event::Act)`. One `Act` call = one loop-step: fold the last
//! action's reward into the option's learned value, then pick the next option by the default VALENCE score
//! `A(o) = −value(o) + μ_o + σ_o²` (the planner's cost-to-go, the signed reward mean, the self-annealing
//! exploration bonus — `substrate_valence`). Ties the shipped pieces — the planner's value field
//! (`prefer`/`relax_values`/`value_of`), the reward memory (the `(μ, visits)` record, `valence_of`) and the policy —
//! into ONE native step.
//!   (A) PLAN — value-iterate to a goal, then `alive(Act)` walks there OPTIMALLY.
//!   (B) REMEMBER + AVOID (soft) — a −reward option, folded through `last`, is avoided; the wild/tame axis
//!       (override a known-bad option toward the goal) is EMERGENT from the value-vs-valence balance, not a dial.
//!
//! (This proof used to pin the pre-1.6 `act_score` rule, whose reward memory was the `reinforce`d `drive`, and
//! carried a (C): `hypothesis_loop = false` made it a plain value-greedy pick. That rule and its levers are
//! removed; `reinforce`/`drive_of` remain, read by `orient`, not by `alive(Act)`.)
use openmorph_core::{Event, Field, Outcome};

const N: usize = 5;
const DELTAS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

fn cell(r: i32, c: i32) -> usize {
    (r.clamp(0, N as i32 - 1) as usize) * N + c.clamp(0, N as i32 - 1) as usize
}

fn act(f: &mut Field, options: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act { options, last }) {
        Outcome::Acted { chosen } => chosen,
        _ => unreachable!(),
    }
}

/// Two options, `far` at cost-to-go `v_far` and `near` at `v_near`; the world punishes `near` with −2 every time it
/// is taken and pays `far` nothing. Returns every pick of a `steps`-long loop through `alive(Act)`.
fn punished_loop(v_far: f32, v_near: f32, steps: usize) -> Vec<usize> {
    let mut m = Field::new(2);
    m.prefer(0, v_far);
    m.prefer(1, v_near);
    let mut last = None;
    (0..steps)
        .map(|_| {
            let a = act(&mut m, &[0, 1], last);
            last = Some((a, if a == 1 { -2.0 } else { 0.0 }));
            a
        })
        .collect()
}

#[test]
fn alive_act_plans_remembers_and_is_soft() {
    // ── (A) PLAN — value-iterate a grid, then walk to the goal via alive(Act) (optimal descent). ──
    let (start, goal) = (0usize, N * N - 1);
    let mut agent = Field::new(N * N);
    let adj: Vec<Vec<usize>> = (0..N * N)
        .map(|s| {
            let (r, c) = ((s / N) as i32, (s % N) as i32);
            DELTAS
                .iter()
                .map(|&(dr, dc)| cell(r + dr, c + dc))
                .collect()
        })
        .collect();
    agent.prefer(goal, 0.0);
    agent.relax_values(&adj, 1.0, 0); // cost-to-go field
    let (mut s, mut steps) = (start, 0usize);
    while s != goal && steps < 100 {
        let (r, c) = ((s / N) as i32, (s % N) as i32);
        let options: Vec<usize> = DELTAS
            .iter()
            .map(|&(dr, dc)| cell(r + dr, c + dc))
            .collect();
        s = act(&mut agent, &options, None); // options are real reachable next-states; alive(Act) descends value
        steps += 1;
    }
    let opt = (N - 1) + (N - 1);
    assert_eq!(s, goal, "alive(Act) must reach the goal");
    assert_eq!(
        steps, opt,
        "alive(Act) planner must be optimal: {steps} vs {opt}"
    );
    eprintln!(
        "[ACT] (A) planner through alive(Act): reached goal in {steps} steps (optimal {opt})"
    );

    // ── (B) REMEMBER + AVOID — SOFT (weighed, not a hard barrier), and the wild/tame axis is EMERGENT from the
    //     value-vs-valence balance, NOT a dial. Same policy, same −2 punishment folded through `last`: a MILD
    //     goal-pull (1.0 vs 2.0) is outweighed once the aversion is learned (tame → avoid); a STRONG goal-pull (0.0
    //     vs 5.0) overrides it (wild → take it anyway). No `wildness` knob — the world's scales decide. ──
    const STEPS: usize = 60;
    let mild = punished_loop(2.0, 1.0, STEPS);
    let punishments = mild.iter().filter(|&&a| a == 1).count();
    let last_punished = mild.iter().rposition(|&a| a == 1).unwrap_or(0);
    eprintln!("[ACT] (B) mild gap (1.0 vs 2.0), −2 on the near option: picks {mild:?}");
    eprintln!("[ACT] (B)   punished {punishments} times, the last at step {last_punished}; avoids it from then on");
    assert_eq!(
        mild[0], 1,
        "with no memory, alive(Act) picks the better-value option"
    );
    assert!(
        mild[STEPS / 2..].iter().all(|&a| a == 0),
        "a MILD goal-pull is outweighed by the learned −2 aversion → AVOID the punished option (tame): {mild:?}"
    );
    // measured (deterministic): 6 punishments, the last at step 13
    assert_eq!(
        (punishments, last_punished),
        (6, 13),
        "the aversion is learned by step 13: {mild:?}"
    );
    let strong = punished_loop(5.0, 0.0, STEPS);
    eprintln!("[ACT] (B) strong gap (0.0 vs 5.0), −2 on the at-goal option: picks {strong:?}");
    assert!(
        strong.iter().all(|&a| a == 1),
        "a STRONG goal-pull OVERRIDES the same −2 aversion → keep taking the punished-but-at-goal option (wild): {strong:?}"
    );

    eprintln!("[ACT] VERDICT: #150's loop is NATIVE in `alive(Event::Act)` — one call = remember (fold the last reward into the option's (μ, visits) record) + select by the valence score `−value + μ + σ²`. (A) over a value field it walks to the goal OPTIMALLY (the planner, native). (B) it REMEMBERS a −reward and AVOIDS the option — SOFT, and the wild/tame axis is EMERGENT from the value-vs-valence balance: a mild goal-pull is outweighed once the aversion is learned ({punishments} punishments, then avoid), a strong one overrides it (take it anyway, every step). Ties the planner + the reward memory + the policy into the one alive() entry — 'we are alive, and alive is this'.");
}
