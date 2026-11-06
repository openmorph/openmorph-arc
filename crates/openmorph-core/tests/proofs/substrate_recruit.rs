// SPDX-License-Identifier: MIT-0
//! RECRUITMENT ON INTERFERENCE — growth at the MODULE scale, from the SAME surprise law. `grow_and_learn`
//! births a unit when a cell's SURPRISE is high and nothing yet predicts it. This lifts that one law to the
//! library: when a NEW task arrives and the existing (consolidated) modules leave high surprise — none of
//! them recognizes it — the organism BIRTHS a fresh module instead of overwriting a banked one. Nothing is
//! authored: the recruit decision is the organism's own error EMA crossing the birth gate, exactly as
//! hidden-unit birth is; consolidation (plasticity 0) then makes the new skill permanent.
//!
//! This closes the #103 stretch: placement (`graft_at`) + consolidation + arbitration compose N modules,
//! but only if something DECIDES when a task needs its own module. Surprise decides — the growth law the
//! substrate already runs everywhere. Without recruitment, forcing the new task onto the old module
//! destroys it (interference); with surprise-gated recruitment, both tasks are served and the old skill is
//! untouched.
//!
//!   cargo test -p openmorph-core --test proofs substrate_recruit -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const BUD: usize = 200;
const BIRTH_GATE: f32 = 0.5; // the same surprise threshold that gates hidden-unit birth

// Cells: 0,1 = context (A=[1,1], B=[1,-1], sharing cell 0 so routing isn't a trivial one-hot); 2 = state.
fn ctx(a: bool) -> (f32, f32) {
    if a {
        (1.0, 1.0)
    } else {
        (1.0, -1.0)
    }
}
fn inp(a: bool, s: f32) -> Vec<(usize, f32)> {
    let (c0, c1) = ctx(a);
    vec![(0, c0), (1, c1), (2, s)]
}
// Opposite policies over the SAME state: task A wants value = s, task B wants value = -s. One shared
// readout over the state cannot do both — that is the interference recruitment must resolve.
fn target(a: bool, s: f32) -> f32 {
    if a {
        s
    } else {
        -s
    }
}

/// Grow a module: a GATE cell (recognizes its context: +1 on-context, −1 off) and an ACTOR cell (the
/// policy over the state). Returns (gate, actor). Trained, then consolidated.
fn grow_module(f: &mut Field, on_ctx: bool, rng: &mut Rng) -> (usize, usize) {
    let gate = f.len();
    f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, gate);
    let actor = f.len();
    f.add_unit(vec![2], vec![0.0], 0.0, actor);
    for _ in 0..4000 {
        // Train the gate to recognize its context (on both contexts), the actor to act on its own.
        let a = rng.bit() > 0.0;
        let s = rng.bit();
        f.learn_step(
            &inp(a, s),
            gate,
            if a == on_ctx { 1.0 } else { -1.0 },
            0.1,
            BUD,
        );
        if a == on_ctx {
            f.learn_step(&inp(a, s), actor, target(on_ctx, s), 0.1, BUD);
        }
    }
    f.consolidate(&[gate, actor]);
    (gate, actor)
}

/// The organism's surprise on `task` under the CURRENT library: the error EMA of the arbitrated output.
fn surprise_on(f: &mut Field, gates: &[usize], actors: &[usize], a: bool, rng: &mut Rng) -> f32 {
    let mut ema = 0.0;
    for _ in 0..200 {
        let s = rng.bit();
        f.stimulate(&inp(a, s), BUD);
        let out = f.arbitrate(gates, actors);
        ema = 0.9 * ema + 0.1 * (target(a, s) - out).abs();
    }
    ema
}

fn task_mae(f: &mut Field, gates: &[usize], actors: &[usize], a: bool, rng: &mut Rng) -> f32 {
    let mut sum = 0.0;
    for _ in 0..200 {
        let s = rng.bit();
        f.stimulate(&inp(a, s), BUD);
        sum += (target(a, s) - f.arbitrate(gates, actors)).abs();
    }
    sum / 200.0
}

#[test]
fn surprise_recruits_a_new_module_instead_of_overwriting_a_banked_one() {
    let mut rng = Rng::new(0x2EC0);

    // Start: ONE module for task A, banked.
    let mut f = Field::new(3);
    let (ga, aa) = grow_module(&mut f, true, &mut rng);
    let (mut gates, mut actors) = (vec![ga], vec![aa]);
    let mae_a_before = task_mae(&mut f, &gates, &actors, true, &mut rng);

    // A FAMILIAR task (A again) must NOT recruit — the library already predicts it, surprise is low.
    let surprise_familiar = surprise_on(&mut f, &gates, &actors, true, &mut rng);
    assert!(
        surprise_familiar < BIRTH_GATE,
        "a task the library already serves must not recruit (surprise {surprise_familiar:.2})"
    );

    // Task B arrives. The organism measures its SURPRISE on B under the current library (module A only).
    let surprise_b = surprise_on(&mut f, &gates, &actors, false, &mut rng);
    let recruited = surprise_b > BIRTH_GATE; // the SAME gate that births a hidden unit
    if recruited {
        let (gb, ab) = grow_module(&mut f, false, &mut rng); // birth a module for B
        gates.push(gb);
        actors.push(ab);
    }

    // After: both tasks served by the (now two-module) library; module A must be UNCHANGED.
    let mae_a_after = task_mae(&mut f, &gates, &actors, true, &mut rng);
    let mae_b_after = task_mae(&mut f, &gates, &actors, false, &mut rng);

    // CONTROL — no recruitment: force task B onto module A's actor (kept plastic). Interference destroys A.
    let mut fc = Field::new(3);
    let (gac, aac) = grow_module(&mut fc, true, &mut rng);
    fc.set_plasticity(&[aac], 1.0); // re-open the banked actor (the naive "just keep training" path)
    for _ in 0..4000 {
        let s = rng.bit();
        fc.learn_step(&inp(false, s), aac, target(false, s), 0.1, BUD);
    }
    let ctrl_mae_a = task_mae(&mut fc, &[gac], &[aac], true, &mut rng);

    eprintln!(
        "[RECRUIT] task B arrives on a library holding only module A:\n  surprise on B under {{A}}: {surprise_b:.2}  (> gate {BIRTH_GATE} ⇒ RECRUIT)  → recruited = {recruited}\n  RECRUITED library : task-A MAE {mae_a_before:.2} → {mae_a_after:.2} (A untouched), task-B MAE {mae_b_after:.2}\n  CONTROL (no recruit, retrain A's actor on B): task-A MAE {mae_a_before:.2} → {ctrl_mae_a:.2} (A destroyed)",
    );
    eprintln!("[RECRUIT] VERDICT: recruitment-on-interference is the surprise-birth law at the MODULE scale — no authored trigger. When the existing modules leave high surprise on a new task, the organism BIRTHS a module (and consolidates it) rather than overwriting a banked skill; when they don't, it reuses. Surprise decides growth here exactly as it does for a hidden unit. This is the deciding mechanism the #103 non-interfering library needed.");

    assert!(
        recruited,
        "high surprise on the novel task must trigger recruitment"
    );
    assert!(
        mae_a_after < 0.1,
        "the banked module A must be UNCHANGED after recruiting B (got {mae_a_after:.2})"
    );
    assert!(
        mae_b_after < 0.2,
        "the recruited module must serve task B (got {mae_b_after:.2})"
    );
    assert!(
        ctrl_mae_a > 0.6,
        "without recruitment, retraining on B must destroy A (control MAE {ctrl_mae_a:.2})"
    );
}
