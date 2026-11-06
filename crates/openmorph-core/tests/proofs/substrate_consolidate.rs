// SPDX-License-Identifier: MIT-0
//! CONSOLIDATION — modular-skill isolation at the source (MorphLab #103, ask #1). A living organism
//! that keeps learning catastrophically INTERFERES with its banked skills: a later `learn_step` (a new
//! task, a shared router) touching a skill's cell overwrites it. `learn_step`/`observe` are already
//! LOCAL (they touch only `writers[output_cell]`), so placement (`graft_at`) stops skills colliding in
//! space — but not in TIME. `consolidate(cells)` closes that gap: it freezes a matured skill's units
//! (plasticity → 0), so later learning cannot degrade it. Isolation guaranteed at the source, not by a
//! caller convention one stray `learn_step` breaks.
//!
//! The probe: grow a skill (O = tanh(1.5·x)); then hammer the SAME cell with the OPPOSITE mapping (the
//! interfering task). Without consolidation the skill is destroyed; consolidated, it survives untouched.
//!
//!   cargo test -p openmorph-core --test proofs substrate_consolidate -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const O: usize = 1; // the skill's output cell
fn skill(x: f32) -> f32 {
    (1.5 * x).tanh()
}

fn mae_on_skill(f: &mut Field, rng: &mut Rng) -> f32 {
    let mut e = 0.0;
    for _ in 0..400 {
        let x = rng.sym();
        f.stimulate(&[(0, x)], 50);
        e += (f.cells()[O] - skill(x)).abs();
    }
    e / 400.0
}

/// Grow the skill, optionally consolidate it, then hammer its cell with the OPPOSITE map (interference).
/// Returns MAE on the ORIGINAL skill (a) right after learning it, (b) after the interference.
fn run(consolidate: bool) -> (f32, f32) {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    f.config_mut().birth_gate = 0.05;
    let mut rng = Rng::new(0xC0FFEE);
    for _ in 0..3000 {
        let x = rng.sym();
        f.grow(&[(0, x)], O, skill(x), 0.1, 50);
    }
    let before = mae_on_skill(&mut f, &mut rng);

    if consolidate {
        f.consolidate(&[O]); // freeze the matured skill
    }
    // A later task (or a stray router) learns the OPPOSITE mapping on the same cell.
    for _ in 0..3000 {
        let x = rng.sym();
        f.learn_step(&[(0, x)], O, (-1.5 * x).tanh(), 0.1, 50);
    }
    let after = mae_on_skill(&mut f, &mut rng);
    (before, after)
}

#[test]
fn consolidate_protects_a_skill_from_catastrophic_interference() {
    let (b0, plain) = run(false);
    let (b1, frozen) = run(true);

    eprintln!(
        "[CONSOLIDATE] a banked skill (O = tanh(1.5·x)) after 3000 steps of the OPPOSITE task on the same cell:\n  learned MAE (both)     : {:.3} / {:.3}\n  WITHOUT consolidation  : MAE {plain:.3}  ← catastrophically overwritten\n  WITH consolidation     : MAE {frozen:.3}  ← frozen, survives untouched",
        b0, b1,
    );
    eprintln!("[CONSOLIDATE] VERDICT (MorphLab #103 ask #1): `consolidate(cells)` freezes a matured skill's units (plasticity → 0) and the learn path skips them, so later learning — a new task, a shared router, a stray learn_step — cannot degrade it. Combined with `graft_at` placement and the already-local learn_step, a library of banked skills is isolated in space AND time. Modular isolation is now a guarantee at the source, not a caller convention. (Plasticity is a `[0,1]` dial; `set_plasticity` gives graded/slow protection, and could decay with age/energy for consolidation-by-thriving.)");

    assert!(
        plain > 0.5,
        "without consolidation the skill should be overwritten (MAE {plain:.3})"
    );
    assert!(
        frozen < 0.05,
        "consolidation should protect the skill (MAE {frozen:.3})"
    );
}

/// A banked skill faces the alive economy too: when another module is active, a dormant skill neither
/// fires nor earns and would STARVE. A consolidated skill must survive dormancy — else the library
/// forgets by DEATH, not just by overwriting. `cull` spares consolidated (plasticity-0) units.
#[test]
fn a_consolidated_skill_survives_dormancy() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    f.config_mut().birth_gate = 0.05;
    let mut rng = Rng::new(0xDEAD);
    for _ in 0..2000 {
        let x = rng.sym();
        f.grow(&[(0, x)], O, skill(x), 0.1, 50);
    }
    // A second, NON-consolidated unit on its own cell — the control that should starve and die.
    f.add_unit(vec![0], vec![0.5], 0.0, 3);

    f.consolidate(&[O]); // bank the skill; leave cell-3 unit plastic

    // Run the economy with BOTH units dormant (nothing fires → nothing earns → both starve).
    let mut culled_total = 0;
    for _ in 0..40 {
        f.stimulate(&[], 10); // no input → no firing → dormant
        f.metabolize(1.0, 1.0); // not fired → −1 upkeep each cycle
        culled_total += f.cull(0.0);
    }

    let skill_alive = f.units.iter().any(|u| u.write == O);
    let control_alive = f.units.iter().any(|u| u.write == 3);
    // The consolidated skill still computes correctly.
    let mae = mae_on_skill(&mut f, &mut rng);

    eprintln!(
        "[CONSOLIDATE] dormancy under the alive economy (40 starving cycles, {culled_total} units culled):\n  consolidated skill alive = {skill_alive} (MAE {mae:.3})   |   non-consolidated control alive = {control_alive}"
    );
    eprintln!("[CONSOLIDATE] VERDICT (robustness for the library): a consolidated skill is a PERMANENT asset — `cull` spares plasticity-0 units, so a banked module survives dormancy while another module is the active one. Consolidation protects against BOTH later learning and starvation; without the death-immunity the library would forget the moment an inactive module ran out of energy.");

    assert!(skill_alive, "the consolidated skill must survive dormancy");
    assert!(mae < 0.05, "and still compute (MAE {mae:.3})");
    assert!(
        !control_alive,
        "the non-consolidated control should starve and be culled"
    );
}
