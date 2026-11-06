// SPDX-License-Identifier: MIT-0
//! BEATING THE REACTIVE BASELINE #4 — COMPOSITION: a rule that needs TWO grown abstractions COMPOSED, which no
//! single one can represent. The lever MorphBot's live ladder points at for the "none-of-these" hard games.
//!
//! MorphBot's reachable-34 ablation (2026-08-11) validated the memory/counting/comparison augmentations live
//! (each lifts exactly its games) but found the reachable-ARC number is NOT gated by picking ONE abstraction —
//! it is gated by the 12 "none-of-these" games that need COMPOSITIONAL / multi-step rules. This rung is the
//! in-core mechanism for that: a rule that is a COMPOSITION of two abstractions (`same_color` XOR `count_even`).
//! XOR of the two factors makes EACH single abstraction exactly chance — only the COMPOSED state (their product)
//! is representable. Same shipped free-energy actor; the only difference is whether the state key composes both.
//!
//!   - reactive (neither)                → 50% (the null).
//!   - `relate(same_color)` alone        → 50% (within a same-colour state, the parity is a coin flip → XOR chance).
//!   - `aggregate(count)` parity alone   → 50% (within a parity state, same-colour is a coin flip).
//!   - COMPOSED `same_color × parity`    → 4 states, the XOR table is deterministic → ~100%.
//!
//! Truth-not-trophy: the composed state solves it (> 0.90) where EVERY single abstraction (each a control that
//! must fail) is stuck at chance (< 0.65) — so composition, not any one abstraction, is the lever. This is the
//! bridge from the proven compose primitives (`nest`/reasoning/beam) to the actor line: the hard games need
//! composed rules, and a composed state makes them representable.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_compose -- --nocapture

use openmorph_core::grouping::{aggregate, relate, Object};
use openmorph_core::{Config, Event, Field, Outcome};

const TRAIN: usize = 2_000;
const EVAL: usize = 400;

fn u01(s: &mut u64) -> f32 {
    let mut x = *s | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x >> 11) as f32 / (1u64 << 53) as f32
}

fn fe_field() -> Field {
    let mut f = Field::new(1);
    f.configure(Config {
        free_energy_actor: true,
        free_energy_sstar: 1.0,
        ..Config::default()
    });
    f
}

fn act(f: &mut Field, s: usize, last: Option<(usize, f32)>) -> usize {
    let opts = [s * 2, s * 2 + 1];
    match f.alive(Event::Act {
        options: &opts,
        last,
    }) {
        Outcome::Acted { chosen } => chosen,
        _ => unreachable!(),
    }
}

fn obj(color: usize, base: usize) -> Object {
    Object {
        cells: vec![base, base + 1],
        color,
        bbox_w: 2,
        bbox_h: 1,
    }
}

/// Which state encoding a candidate uses over the composed scene.
#[derive(Clone, Copy)]
enum Code {
    Reactive,
    Relate,
    Aggregate,
    Composed,
}

/// A scene carries BOTH a colour pair (for `relate`) and a set (for `aggregate`); the rule is their XOR.
struct Scene {
    ca: usize,
    cb: usize,
    vals: Vec<f32>,
}

fn scene(s: &mut u64) -> Scene {
    let ca = usize::from(u01(s) < 0.5);
    let cb = usize::from(u01(s) < 0.5);
    let n = 2 + (u01(s) * 4.0) as usize % 4; // 2..=5 objects
    let vals: Vec<f32> = (0..n).map(|_| u01(s)).collect();
    Scene { ca, cb, vals }
}

/// The rule: `a = 1` iff (A,B same colour) XOR (count even) — a COMPOSITION of comparison and counting.
fn target(sc: &Scene) -> usize {
    let same = sc.ca == sc.cb;
    let even = sc.vals.len().is_multiple_of(2);
    usize::from(same ^ even)
}

fn encode(code: Code, sc: &Scene) -> usize {
    let same = usize::from({
        let objs = [obj(sc.ca, 0), obj(sc.cb, 4)];
        relate(&objs, 8, 8)[0].same_color
    });
    let parity = (aggregate(&sc.vals).count as usize) % 2;
    match code {
        Code::Reactive => 0,
        Code::Relate => same,                // 2 states — blind to parity
        Code::Aggregate => parity,           // 2 states — blind to colour
        Code::Composed => same * 2 + parity, // 4 states — both factors
    }
}

fn run(code: Code, train_seed: u64, eval_seed: u64) -> f32 {
    let mut f = fe_field();
    let mut last: Option<(usize, f32)> = None;
    let mut ts = train_seed;
    for _ in 0..TRAIN {
        let sc = scene(&mut ts);
        let chosen = act(&mut f, encode(code, &sc), last);
        let r = if chosen % 2 == target(&sc) { 1.0 } else { 0.0 };
        last = Some((chosen, r));
    }
    let mut es = eval_seed;
    let mut correct = 0;
    for _ in 0..EVAL {
        let sc = scene(&mut es);
        if act(&mut f, encode(code, &sc), None) % 2 == target(&sc) {
            correct += 1;
        }
    }
    correct as f32 / EVAL as f32
}

#[test]
fn a_composed_state_beats_every_single_abstraction_on_a_compositional_rule() {
    let (ts, es) = (0xC0FFEE, 0xC0DE);
    let reactive = run(Code::Reactive, ts, es);
    let relate_only = run(Code::Relate, ts, es);
    let aggregate_only = run(Code::Aggregate, ts, es);
    let composed = run(Code::Composed, ts, es);

    println!("COMPOSITION (same_color XOR count_even):");
    println!("  reactive        : {reactive:.3}");
    println!("  relate  only    : {relate_only:.3}");
    println!("  aggregate only  : {aggregate_only:.3}");
    println!("  COMPOSED        : {composed:.3}  <== representable");

    // The composed state solves the compositional rule.
    assert!(
        composed > 0.90,
        "the composed state must crack the compositional rule (got {composed:.3})"
    );
    // Every SINGLE abstraction is at chance — composition, not any one, is the lever (each a failing control).
    for (name, v) in [
        ("reactive", reactive),
        ("relate", relate_only),
        ("aggregate", aggregate_only),
    ] {
        assert!(
            v < 0.65,
            "the {name}-only control must be near chance on the XOR-composed rule (got {v:.3})"
        );
    }
}
