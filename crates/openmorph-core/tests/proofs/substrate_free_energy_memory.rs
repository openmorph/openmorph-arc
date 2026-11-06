// SPDX-License-Identifier: MIT-0
//! BEATING THE REACTIVE BASELINE #1 — MEMORY: grown carried-state lets the free-energy actor solve a rule the
//! reactive object-relative state CAN'T (the higher-abstraction games the 21/34 generalist misses).
//!
//! MorphBot's generalist (object-relative displacement state + a generalizing actor) hit 21/34 on real ARC; its
//! misses were the games needing **counting / memory / comparison** — abstractions a REACTIVE (Markov,
//! current-frame) state cannot represent. openmorph already grows those: `aggregate` (count), `delay`/`integrate`
//! (memory), `relate` (compare). This proof is the first rung of BEATING that baseline: on a task that needs
//! MEMORY, the free-energy actor over a reactive state is stuck at chance, and AUGMENTING the state with the
//! substrate's native exact memory (`delay`) closes it — same actor, richer (grown) state.
//!
//! THE TASK — cued binary recall (a T-maze). At step 0 a CUE `c ∈ {0,1}` is observable; for the next `D` steps
//! it is HIDDEN (a neutral observation). At the decision step the actor picks `a ∈ {0,1}`; reward = 1 iff a = c.
//! The rule is NOT a function of the current observation (neutral at decision) — only of the PAST cue.
//!   - **Reactive** state = the current observation (neutral at decision) → one indistinct state for both cues →
//!     the actor cannot condition on `c` → 50% (chance).
//!   - **Memory-augmented** state = a `delay`-chain of length `D` fed by the cue channel → at the decision step
//!     the delayed cell emits `x[t−D]` = the cue → the state separates by `c` → the actor learns `a = c` → ~100%.
//!
//! The actor is the SHIPPED free-energy path (`alive(Event::Act)`, `free_energy_actor`); the ONLY difference
//! between the two runs is whether the state key includes the grown memory. Scored truth-not-trophy:
//!   1) memory-augmented FE-actor SOLVES the recall (> 0.90) — grown memory represents the rule.
//!   2) the reactive FE-actor is STUCK at chance (< 0.65) — the control that must fail (a Markov state can't).
//!   3) the augmented actor beats the reactive one by a wide margin (≥ 0.30) — memory is the lever.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_memory -- --nocapture

use openmorph_core::{Config, Event, Field, Outcome};

const D: usize = 3; // the cue is hidden for D steps before the decision
const TRAIN: usize = 1_200;
const EVAL: usize = 400;

fn u01(s: &mut u64) -> f32 {
    let mut x = *s | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x >> 11) as f32 / (1u64 << 53) as f32
}

fn fe_field(n_cells: usize) -> Field {
    let mut f = Field::new(n_cells);
    f.configure(Config {
        free_energy_actor: true,
        free_energy_sstar: 1.0,
        ..Config::default()
    });
    f
}

/// The MEMORY-augmented actor: a `delay` chain of length `D` on the cue channel carries `x[t−D]`. Returns the
/// held-out recall accuracy. The state at decision = the quantized DELAYED cue (grown carried-state).
fn run_memory(train_seed: u64, eval_seed: u64) -> f32 {
    let mut f = fe_field(1);
    let cue_src = 0usize;
    // Build the delay chain c1←cue_src, c2←c1, …, cD←c(D-1). The last cell emits x[t−D].
    let mut chain = Vec::with_capacity(D);
    let mut src = cue_src;
    for _ in 0..D {
        let c = f.delay(src);
        chain.push(c);
        src = c;
    }
    let last_cell = *chain.last().unwrap();

    // Present a cue, advance the register `D` times through the hidden steps, return the recalled value.
    let recall = |f: &mut Field, cue: f32| -> f32 {
        let mut mem = 0.0;
        for t in 0..=D {
            let obs = if t == 0 { cue } else { 0.5 }; // cue visible only at t=0, neutral after
            f.stimulate(&[(cue_src, obs)], 1);
            let updates = f.advance_delays();
            mem = updates
                .iter()
                .find(|(cell, _)| *cell == last_cell)
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
        }
        mem
    };

    let mut last: Option<(usize, f32)> = None;
    let mut ts = train_seed;
    for _ in 0..TRAIN {
        let c = if u01(&mut ts) < 0.5 { 0usize } else { 1 };
        let m = recall(&mut f, c as f32);
        let s = usize::from(m > 0.5); // the RECALLED cue (grown memory) → the state key
        let opts = [s * 2, s * 2 + 1];
        let chosen = match f.alive(Event::Act {
            options: &opts,
            last,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!(),
        };
        let a = chosen % 2;
        let r = if a == c { 1.0 } else { 0.0 };
        last = Some((chosen, r));
    }
    // Held-out eval (frozen policy — no folding).
    let mut es = eval_seed;
    let mut correct = 0;
    for _ in 0..EVAL {
        let c = if u01(&mut es) < 0.5 { 0usize } else { 1 };
        let m = recall(&mut f, c as f32);
        let s = usize::from(m > 0.5);
        let opts = [s * 2, s * 2 + 1];
        let chosen = match f.alive(Event::Act {
            options: &opts,
            last: None,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!(),
        };
        if chosen % 2 == c {
            correct += 1;
        }
    }
    correct as f32 / EVAL as f32
}

/// The REACTIVE control: identical actor + task, but the state is the CURRENT observation (neutral at the
/// decision) — a Markov state with no memory. One indistinct state for both cues → chance.
fn run_reactive(train_seed: u64, eval_seed: u64) -> f32 {
    let mut f = fe_field(1);
    let decision_state = 0usize; // the current obs at decision is always neutral ⇒ one constant state
    let mut last: Option<(usize, f32)> = None;
    let mut ts = train_seed;
    for _ in 0..TRAIN {
        let c = if u01(&mut ts) < 0.5 { 0usize } else { 1 };
        let opts = [decision_state * 2, decision_state * 2 + 1];
        let chosen = match f.alive(Event::Act {
            options: &opts,
            last,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!(),
        };
        let a = chosen % 2;
        let r = if a == c { 1.0 } else { 0.0 };
        last = Some((chosen, r));
    }
    let mut es = eval_seed;
    let mut correct = 0;
    for _ in 0..EVAL {
        let c = if u01(&mut es) < 0.5 { 0usize } else { 1 };
        let opts = [decision_state * 2, decision_state * 2 + 1];
        let chosen = match f.alive(Event::Act {
            options: &opts,
            last: None,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!(),
        };
        if chosen % 2 == c {
            correct += 1;
        }
    }
    correct as f32 / EVAL as f32
}

#[test]
fn grown_memory_beats_the_reactive_free_energy_actor_on_a_recall_rule() {
    let (train_seed, eval_seed) = (0x5151, 0x9090);
    let memory = run_memory(train_seed, eval_seed);
    let reactive = run_reactive(train_seed, eval_seed);

    println!("FE-actor + grown DELAY memory : recall {memory:.3}");
    println!("FE-actor + reactive state     : recall {reactive:.3}  (chance ceiling)");
    println!("memory − reactive             : {:+.3}", memory - reactive);

    // 1) Grown memory represents the rule.
    assert!(
        memory > 0.90,
        "memory-augmented FE-actor must solve the recall (got {memory:.3})"
    );
    // 2) The reactive control CANNOT (a Markov state can't hold the past cue).
    assert!(
        reactive < 0.65,
        "reactive FE-actor must be stuck near chance (got {reactive:.3})"
    );
    // 3) Memory is the lever — a wide margin.
    assert!(
        memory - reactive >= 0.30,
        "grown memory must beat the reactive baseline by a wide margin (Δ {:.3})",
        memory - reactive
    );
}
