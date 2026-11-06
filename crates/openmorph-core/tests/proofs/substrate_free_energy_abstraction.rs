// SPDX-License-Identifier: MIT-0
//! BEATING THE REACTIVE BASELINE #2 & #3 — COUNTING and COMPARISON: grown `aggregate` / `relate` features let
//! the free-energy actor solve rules the reactive object-relative state CAN'T.
//!
//! MorphBot's generalist (object-relative displacement + a generalizing actor) hit 21/34 on real ARC; its misses
//! are the higher-abstraction games (counting / memory / comparison) a REACTIVE per-object state can't represent.
//! Rung 1 (memory, `substrate_free_energy_memory`) showed `delay` closes a recall rule. These are rungs 2 & 3:
//!   - **COUNTING** — a rule that depends on the CARDINALITY of a set. The reactive state sees one object's
//!     value (no count); the substrate's `aggregate` folds the set into `count` → the rule becomes representable.
//!   - **COMPARISON** — a rule that depends on a RELATION between two objects (same colour). The reactive state
//!     sees ONE object's colour (can't compare); the substrate's `relate` yields `same_color` → representable.
//!
//! Both use the SHIPPED free-energy actor (`alive(Event::Act)`, `free_energy_actor`); the ONLY difference is
//! whether the state key carries the grown abstraction. Each scored truth-not-trophy with a control that must
//! fail: augmented solves (> 0.90), reactive is stuck near chance (< 0.65), and augmented beats reactive by a
//! wide margin (≥ 0.30). Nothing task-specific is authored — `aggregate`/`relate` are domain-blind statistics.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_abstraction -- --nocapture

use openmorph_core::grouping::{aggregate, relate, Object};
use openmorph_core::{Config, Event, Field, Outcome};

const TRAIN: usize = 1_500;
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

/// Select a binary action for state `s` under the shipped free-energy actor, carrying the previous reward in
/// `last` (which the call folds into that option's ensemble). Returns the chosen option key.
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

// ============================================================================================================
// Rung 2 — COUNTING (via `aggregate`)
// ============================================================================================================

/// One counting run. The rule: `a = 1` iff the set's CARDINALITY is even. `augmented` ⇒ the state key is the
/// `aggregate().count` parity (the grown count); else it is the quantized FIRST value (a reactive per-object
/// view with no count). Returns held-out accuracy.
fn run_count(train_seed: u64, eval_seed: u64, augmented: bool) -> f32 {
    let mut f = fe_field();
    let state = |vals: &[f32]| -> usize {
        if augmented {
            (aggregate(vals).count as usize) % 2 // grown COUNT parity → 2 states
        } else {
            (vals[0] * 4.0) as usize % 4 // reactive: one object's value, no cardinality
        }
    };
    let scene = |s: &mut u64| -> (Vec<f32>, usize) {
        let n = 2 + (u01(s) * 4.0) as usize % 4; // 2..=5 objects
        let vals: Vec<f32> = (0..n).map(|_| u01(s)).collect();
        let target = usize::from(n.is_multiple_of(2)); // a = 1 iff the COUNT is even
        (vals, target)
    };

    let mut last: Option<(usize, f32)> = None;
    let mut ts = train_seed;
    for _ in 0..TRAIN {
        let (vals, target) = scene(&mut ts);
        let chosen = act(&mut f, state(&vals), last);
        let r = if chosen % 2 == target { 1.0 } else { 0.0 };
        last = Some((chosen, r));
    }
    let mut es = eval_seed;
    let mut correct = 0;
    for _ in 0..EVAL {
        let (vals, target) = scene(&mut es);
        if act(&mut f, state(&vals), None) % 2 == target {
            correct += 1;
        }
    }
    correct as f32 / EVAL as f32
}

#[test]
fn grown_count_beats_the_reactive_free_energy_actor_on_a_cardinality_rule() {
    let (ts, es) = (0xC0117, 0x7A11);
    let augmented = run_count(ts, es, true);
    let reactive = run_count(ts, es, false);
    println!("COUNTING  · FE + grown aggregate(count) : {augmented:.3}");
    println!("COUNTING  · FE + reactive per-object    : {reactive:.3}  (chance)");
    println!("           Δ = {:+.3}", augmented - reactive);
    assert!(
        augmented > 0.90,
        "grown count must solve the cardinality rule (got {augmented:.3})"
    );
    assert!(
        reactive < 0.65,
        "reactive control must be near chance (got {reactive:.3})"
    );
    assert!(
        augmented - reactive >= 0.30,
        "grown count must beat the reactive baseline by a wide margin (Δ {:.3})",
        augmented - reactive
    );
}

// ============================================================================================================
// Rung 3 — COMPARISON (via `relate`)
// ============================================================================================================

/// A tiny two-cell object of a given colour at a given base cell — enough for `relate` to read `same_color`.
fn obj(color: usize, base: usize) -> Object {
    Object {
        cells: vec![base, base + 1],
        color,
        bbox_w: 2,
        bbox_h: 1,
    }
}

/// One comparison run. The rule: `a = 1` iff the two objects share a colour. `augmented` ⇒ the state key is
/// `relate(...).same_color` (the grown RELATION); else it is object A's own colour (a reactive per-object view
/// that cannot see B). Colours are drawn from {0,1}, so knowing only A's colour gives NO information about
/// sameness → chance. Returns held-out accuracy.
fn run_compare(train_seed: u64, eval_seed: u64, augmented: bool) -> f32 {
    let mut f = fe_field();
    let state = |ca: usize, cb: usize| -> usize {
        if augmented {
            let objs = [obj(ca, 0), obj(cb, 4)];
            usize::from(relate(&objs, 8, 8)[0].same_color) // grown RELATION → 2 states
        } else {
            ca // reactive: only A's colour, blind to B
        }
    };
    let scene = |s: &mut u64| -> (usize, usize, usize) {
        let ca = usize::from(u01(s) < 0.5);
        let cb = usize::from(u01(s) < 0.5);
        (ca, cb, usize::from(ca == cb)) // a = 1 iff same colour
    };

    let mut last: Option<(usize, f32)> = None;
    let mut ts = train_seed;
    for _ in 0..TRAIN {
        let (ca, cb, target) = scene(&mut ts);
        let chosen = act(&mut f, state(ca, cb), last);
        let r = if chosen % 2 == target { 1.0 } else { 0.0 };
        last = Some((chosen, r));
    }
    let mut es = eval_seed;
    let mut correct = 0;
    for _ in 0..EVAL {
        let (ca, cb, target) = scene(&mut es);
        if act(&mut f, state(ca, cb), None) % 2 == target {
            correct += 1;
        }
    }
    correct as f32 / EVAL as f32
}

#[test]
fn grown_relation_beats_the_reactive_free_energy_actor_on_a_comparison_rule() {
    let (ts, es) = (0xC0FFEE, 0x5A0E);
    let augmented = run_compare(ts, es, true);
    let reactive = run_compare(ts, es, false);
    println!("COMPARISON · FE + grown relate(same_color) : {augmented:.3}");
    println!("COMPARISON · FE + reactive per-object      : {reactive:.3}  (chance)");
    println!("            Δ = {:+.3}", augmented - reactive);
    assert!(
        augmented > 0.90,
        "grown relation must solve the comparison rule (got {augmented:.3})"
    );
    assert!(
        reactive < 0.65,
        "reactive control must be near chance (got {reactive:.3})"
    );
    assert!(
        augmented - reactive >= 0.30,
        "grown relation must beat the reactive baseline by a wide margin (Δ {:.3})",
        augmented - reactive
    );
}
