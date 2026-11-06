// SPDX-License-Identifier: MIT-0
//! FREE-ENERGY ACTOR × OBJECT-RELATIVE STATE = CROSS-INSTANCE GENERALIZATION (the ARC-line answer, in-core).
//!
//! MorphBot's landmark real-ARC result (2026-08-11, `gym --arc3-agent`): an **object-relative** state
//! (agent→goal displacement) + a **generalizing** actor, trained multi-episode across instances, beats random
//! ~2× on the reachable set (21/34 vs 11/34) — where every one-shot tabular harness masked it and every pixel
//! encoder failed. The win used a TD-Q actor. THIS proof asks the openmorph question: does the substrate's OWN
//! principled actor — the **free-energy actor** `G = value + (μ−s*)² − σ²` (no ε, no β, no TD bootstrap; A6
//! `Config::free_energy_actor`) — generalize the same way over the same object-relative state? Truth-not-trophy:
//! it is benchmarked head-to-head with a TD-Q baseline (must match or the gap is explained) and against an
//! absolute-state control that MUST fail (proving the representation, not the actor, carries generalization).
//!
//! THE WORLD — grid navigation, the smallest world with the ARC control structure. Each INSTANCE places an agent
//! and a goal at random absolute cells; the rule ("reduce the agent→goal displacement") is INSTANCE-INVARIANT in
//! object-relative coordinates. Immediate shaped reward r = 1 iff the move reduced Manhattan distance (a world
//! property both actors see). Success = reach the goal within a step budget.
//!   - **Object-relative state** = `(sign(gx−ax), sign(gy−ax))` → 9 buckets shared by ALL instances, so a policy
//!     learned on training layouts applies to held-out layouts.
//!   - **Absolute state** (the control) = the full `(agent, goal)` position pair → every held-out instance is a
//!     NEW state, so nothing transfers.
//!
//! THE ACTOR exercises the SHIPPED free-energy path (`alive(Event::Act)`, `free_energy_actor = true`): the
//! per-`(state,action)` option is a virtual key `state·4 + action`; `alive(Act)` folds the reward into that
//! option's ensemble (μ→reward, σ²→0) and ARGMIN-selects `G`. Across instances the same relative state → same
//! option keys → shared learning.
//!
//! Scored truth-not-trophy — four measured numbers, each with a control that must fail:
//!   1) FE-actor over the object-relative state GENERALIZES to held-out layouts (> 0.80).
//!   2) it BEATS RANDOM through learning (> random + 0.30).
//!   3) it MATCHES the TD-Q baseline (≥ TD − 0.15) — the substrate's ε/β/TD-free actor is not worse.
//!   4) the ABSOLUTE-state control does NOT generalize (≥ 0.30 below the relative actor) — the representation is
//!      load-bearing, exactly MorphBot's object-relative-vs-bag finding, reproduced in-core.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_generalize -- --nocapture

use openmorph_core::{Config, Event, Field, Outcome};
use std::collections::HashMap;

const W: i32 = 12;
const H: i32 = 12;
const N_ACT: usize = 4;
const BUDGET: usize = 28;
const TRAIN: usize = 500;
const EVAL: usize = 200;

/// Deterministic xorshift → unit float (no wall-clock; the whole proof is byte-replayable).
fn u01(s: &mut u64) -> f32 {
    let mut x = *s | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x >> 11) as f32 / (1u64 << 53) as f32
}

fn rand_cell(s: &mut u64) -> (i32, i32) {
    let x = (u01(s) * W as f32) as i32 % W;
    let y = (u01(s) * H as f32) as i32 % H;
    (x, y)
}

/// A random (agent, goal) instance with agent ≠ goal.
fn instance(s: &mut u64) -> (i32, i32, i32, i32) {
    let (ax, ay) = rand_cell(s);
    loop {
        let (gx, gy) = rand_cell(s);
        if (gx, gy) != (ax, ay) {
            return (ax, ay, gx, gy);
        }
    }
}

fn step_pos(ax: i32, ay: i32, a: usize) -> (i32, i32) {
    let (mut x, mut y) = (ax, ay);
    match a {
        0 => y = (y - 1).max(0),     // up
        1 => y = (y + 1).min(H - 1), // down
        2 => x = (x - 1).max(0),     // left
        _ => x = (x + 1).min(W - 1), // right
    }
    (x, y)
}

fn manhattan(ax: i32, ay: i32, gx: i32, gy: i32) -> i32 {
    (ax - gx).abs() + (ay - gy).abs()
}

/// OBJECT-RELATIVE state — sign of the agent→goal displacement, shared across all instances (0..9).
fn rel_state(ax: i32, ay: i32, gx: i32, gy: i32) -> usize {
    let sx = (gx - ax).signum() + 1; // 0,1,2
    let sy = (gy - ay).signum() + 1;
    (sx * 3 + sy) as usize
}

/// ABSOLUTE state (the control) — the full (agent, goal) position pair; every layout is its own state.
fn abs_state(ax: i32, ay: i32, gx: i32, gy: i32) -> usize {
    let cells = (W * H) as usize;
    ((ay * W + ax) as usize) * cells + (gy * W + gx) as usize
}

// ---- the free-energy actor (the SHIPPED `alive(Act)` path) -------------------------------------------------

fn fe_field() -> Field {
    let mut f = Field::new(1);
    f.configure(Config {
        free_energy_actor: true,
        free_energy_sstar: 1.0, // the preferred reward s* = the max achievable shaped reward
        ..Config::default()
    });
    f
}

/// One episode under the free-energy actor over the given state encoder. `learn` = fold reward (train) vs a
/// frozen policy (held-out eval). Returns whether the goal was reached within the budget.
fn fe_episode(
    f: &mut Field,
    encode: fn(i32, i32, i32, i32) -> usize,
    inst: (i32, i32, i32, i32),
    learn: bool,
) -> bool {
    let (mut ax, mut ay, gx, gy) = inst;
    let mut last: Option<(usize, f32)> = None;
    for _ in 0..BUDGET {
        if (ax, ay) == (gx, gy) {
            return true;
        }
        let s = encode(ax, ay, gx, gy);
        let opts: Vec<usize> = (0..N_ACT).map(|a| s * N_ACT + a).collect();
        let chosen = match f.alive(Event::Act {
            options: &opts,
            last: if learn { last } else { None },
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!(),
        };
        let a = chosen % N_ACT;
        let d0 = manhattan(ax, ay, gx, gy);
        let (nx, ny) = step_pos(ax, ay, a);
        ax = nx;
        ay = ny;
        let r = if manhattan(ax, ay, gx, gy) < d0 {
            1.0
        } else {
            0.0
        };
        last = Some((chosen, r));
    }
    (ax, ay) == (gx, gy)
}

fn fe_run(encode: fn(i32, i32, i32, i32) -> usize, train_seed: u64, eval_seed: u64) -> f32 {
    let mut f = fe_field();
    let mut ts = train_seed;
    for _ in 0..TRAIN {
        let inst = instance(&mut ts);
        fe_episode(&mut f, encode, inst, true);
    }
    let mut es = eval_seed;
    let mut solved = 0;
    for _ in 0..EVAL {
        let inst = instance(&mut es);
        if fe_episode(&mut f, encode, inst, false) {
            solved += 1;
        }
    }
    solved as f32 / EVAL as f32
}

// ---- the TD-Q baseline (external; the actor MorphBot used) -------------------------------------------------

fn td_run(train_seed: u64, eval_seed: u64) -> f32 {
    let mut q: HashMap<usize, [f32; N_ACT]> = HashMap::new();
    let lr = 0.2f32;
    let mut ts = train_seed;
    let mut eps_rng = 0xEE_u64;
    for i in 0..TRAIN {
        let (mut ax, mut ay, gx, gy) = instance(&mut ts);
        let eps = 0.3 * (1.0 - i as f32 / TRAIN as f32); // decaying ε
        for _ in 0..BUDGET {
            if (ax, ay) == (gx, gy) {
                break;
            }
            let s = rel_state(ax, ay, gx, gy);
            let row = q.entry(s).or_insert([0.0; N_ACT]);
            let a = if u01(&mut eps_rng) < eps {
                (u01(&mut eps_rng) * N_ACT as f32) as usize % N_ACT
            } else {
                (0..N_ACT)
                    .max_by(|&x, &y| row[x].total_cmp(&row[y]))
                    .unwrap()
            };
            let d0 = manhattan(ax, ay, gx, gy);
            let (nx, ny) = step_pos(ax, ay, a);
            ax = nx;
            ay = ny;
            let r = if manhattan(ax, ay, gx, gy) < d0 {
                1.0
            } else {
                0.0
            };
            let row = q.entry(s).or_insert([0.0; N_ACT]);
            row[a] += lr * (r - row[a]); // contextual-bandit update (immediate shaped reward)
        }
    }
    let mut es = eval_seed;
    let mut solved = 0;
    for _ in 0..EVAL {
        let (mut ax, mut ay, gx, gy) = instance(&mut es);
        let mut reached = false;
        for _ in 0..BUDGET {
            if (ax, ay) == (gx, gy) {
                reached = true;
                break;
            }
            let s = rel_state(ax, ay, gx, gy);
            let a = q
                .get(&s)
                .map(|row| {
                    (0..N_ACT)
                        .max_by(|&x, &y| row[x].total_cmp(&row[y]))
                        .unwrap()
                })
                .unwrap_or(0);
            let (nx, ny) = step_pos(ax, ay, a);
            ax = nx;
            ay = ny;
        }
        if reached || (ax, ay) == (gx, gy) {
            solved += 1;
        }
    }
    solved as f32 / EVAL as f32
}

// ---- the random floor --------------------------------------------------------------------------------------

fn random_run(eval_seed: u64) -> f32 {
    let mut es = eval_seed;
    let mut rng = 0x1234_u64;
    let mut solved = 0;
    for _ in 0..EVAL {
        let (mut ax, mut ay, gx, gy) = instance(&mut es);
        for _ in 0..BUDGET {
            if (ax, ay) == (gx, gy) {
                break;
            }
            let a = (u01(&mut rng) * N_ACT as f32) as usize % N_ACT;
            let (nx, ny) = step_pos(ax, ay, a);
            ax = nx;
            ay = ny;
        }
        if (ax, ay) == (gx, gy) {
            solved += 1;
        }
    }
    solved as f32 / EVAL as f32
}

#[test]
fn free_energy_actor_generalizes_across_instances_over_an_object_relative_state() {
    let (train_seed, eval_seed) = (0xABCDEF, 0x9911); // held-out layouts are a DIFFERENT seed stream

    let fe_rel = fe_run(rel_state, train_seed, eval_seed);
    let fe_abs = fe_run(abs_state, train_seed, eval_seed);
    let td_rel = td_run(train_seed, eval_seed);
    let rand = random_run(eval_seed);

    println!("FE-actor  · object-relative state : held-out {fe_rel:.3}");
    println!("TD-Q      · object-relative state : held-out {td_rel:.3}");
    println!("FE-actor  · ABSOLUTE state (ctrl) : held-out {fe_abs:.3}");
    println!("random                            : held-out {rand:.3}");

    // 1) The free-energy actor over the object-relative state GENERALIZES to held-out layouts.
    assert!(
        fe_rel > 0.80,
        "FE-actor over object-relative state must generalize to held-out layouts (got {fe_rel:.3})"
    );
    // 2) It beats random THROUGH LEARNING.
    assert!(
        fe_rel > rand + 0.30,
        "FE-actor ({fe_rel:.3}) must beat random ({rand:.3}) by a wide margin"
    );
    // 3) It MATCHES the TD-Q baseline (the substrate's ε/β/TD-free actor is not worse) — must-match-or-explain.
    assert!(
        fe_rel >= td_rel - 0.15,
        "FE-actor ({fe_rel:.3}) must be competitive with TD-Q ({td_rel:.3})"
    );
    // 4) The ABSOLUTE-state control does NOT generalize — the object-relative REPRESENTATION is load-bearing
    //    (MorphBot's object-relative-beats-bag finding, reproduced in-core).
    assert!(
        fe_abs < fe_rel - 0.30,
        "absolute-state control ({fe_abs:.3}) must fail to generalize vs the relative actor ({fe_rel:.3})"
    );
}
