// SPDX-License-Identifier: MIT-0
//! SLEEP = freeze the WHOLE matured subgraph, not just the readout. A living morph that keeps learning must
//! protect a banked skill from interference — and `consolidate` does that, but only for the cells you pass it.
//! For a GROWN skill (a readout over hidden random-projection features), consolidating just the OUTPUT leaves
//! the HIDDEN features plastic and — crucially — CULLABLE: when the skill goes dormant while a new context is
//! active, its hidden features starve and are culled, and the readout loses its depth (the skill breaks) even
//! though the readout itself survived. The fix needs no new mechanism: sleep = `consolidate` the whole subgraph
//! (the target cell PLUS the grown features feeding it, `target_basis`) at a boundary. Then the whole skill is
//! frozen and cull-immune — it sleeps through dormancy intact. This is the composable answer to MorphLab's
//! "consolidate protects only the readout" (a mechanism gap only if you freeze the readout alone).
//!
//!   cargo test -p openmorph-core --test proofs substrate_sleep -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const Y: usize = 2; // the skill's output cell (inputs 0,1)

// XOR-like: needs GROWN hidden depth (a linear readout can't do it), so the hidden subgraph matters.
fn label(x0: f32, x1: f32) -> f32 {
    if x0 * x1 > 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn build_skill() -> Field {
    let mut f = Field::new(3);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 16;
    let mut rng = Rng::new(0x51ee);
    for _ in 0..8000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.grow(&[(0, x0), (1, x1)], Y, label(x0, x1), 0.05, 50);
    }
    f
}

fn acc(f: &mut Field) -> f64 {
    let mut rng = Rng::new(0x51ee ^ 0xFFFF);
    let mut c = 0usize;
    for _ in 0..1000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], 50);
        let pred = if f.cells()[Y] > 0.0 { 1.0 } else { -1.0 };
        if (pred - label(x0, x1)).abs() < 1e-6 {
            c += 1;
        }
    }
    c as f64 / 1000.0
}

// SLEEP a skill: freeze the whole subgraph feeding `target` — the target cell AND the grown features it relies
// on (target_basis). Composable, no new core.
fn sleep_skill(f: &mut Field, target: usize) {
    let mut cells = vec![target];
    cells.extend(f.target_basis(target));
    f.consolidate(&cells);
}

// Hammer: the skill goes DORMANT while a "new context" is active (a fresh cell drives activity, the skill's
// units don't fire), and the firing-based economy starves + culls whatever isn't protected.
fn hammer(f: &mut Field, cycles: usize) {
    let dummy = f.len();
    f.ensure_cells(dummy + 1);
    for i in 0..cycles {
        f.stimulate(&[(dummy, if i % 2 == 0 { 1.0 } else { -1.0 })], 20); // skill dormant
        f.metabolize(2.0, 1.0); // use-it-or-lose-it: dormant units earn 0, pay upkeep
        f.cull(0.5);
    }
}

#[test]
fn sleep_freezes_the_whole_subgraph_not_just_the_readout() {
    // (1) NO SLEEP — dormant skill is culled away
    let mut a = build_skill();
    let a0 = acc(&mut a);
    let u0 = a.n_units();
    hammer(&mut a, 40);
    let a1 = acc(&mut a);

    // (2) READOUT-ONLY consolidate — the output survives, but the hidden features starve and are culled
    let mut b = build_skill();
    b.consolidate(&[Y]);
    hammer(&mut b, 40);
    let b1 = acc(&mut b);
    let ub = b.n_units();

    // (3) WHOLE-SUBGRAPH SLEEP — freeze the readout AND its grown features
    let mut c = build_skill();
    let c0 = acc(&mut c);
    sleep_skill(&mut c, Y);
    hammer(&mut c, 40);
    let c1 = acc(&mut c);
    let uc = c.n_units();

    eprintln!("[SLEEP] skill built: {:.0}% ({} units)", a0 * 100.0, u0);
    eprintln!("[SLEEP] after dormancy + starve/cull:");
    eprintln!("[SLEEP]   no sleep          → {:.0}%", a1 * 100.0);
    eprintln!(
        "[SLEEP]   readout-only      → {:.0}%  ({ub} units — hidden features culled)",
        b1 * 100.0
    );
    eprintln!(
        "[SLEEP]   WHOLE-subgraph    → {:.0}%  ({uc} units — kept)",
        c1 * 100.0
    );
    eprintln!("[SLEEP] VERDICT: freezing only the readout is NOT enough for a grown skill — its hidden features are cullable and die during dormancy, breaking the skill. Sleep = consolidate the WHOLE subgraph (target + target_basis): the skill sleeps through dormancy intact. Composable, no new core.");

    assert!(c0 > 0.90, "the built skill works ({c0:.2})");
    assert!(
        c1 > 0.85,
        "whole-subgraph sleep keeps the skill through dormancy (got {c1:.2})"
    );
    assert!(
        c1 > b1 + 0.15,
        "whole-subgraph sleep beats readout-only (which loses the hidden depth): {c1:.2} vs {b1:.2}"
    );
    assert!(
        uc >= u0,
        "whole-subgraph sleep culls nothing of the skill ({uc} vs {u0})"
    );
}
