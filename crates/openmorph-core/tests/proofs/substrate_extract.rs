// SPDX-License-Identifier: MIT-0
//! EXTRACT-INTO-BRICK — collapse a matured skill from many active units to ONE nested cell, losslessly. This is
//! the brick-compression `encapsulate` alone couldn't do: `encapsulate` COPIES a subgraph (non-mutating), so the
//! originals stayed in the active graph. The new `detach` removes them. So the sleep-consolidation loop is:
//! `encapsulate` the whole matured subgraph → `detach` the originals → `nest` the brick as one cell. The skill's
//! internals are preserved verbatim inside the frozen, memoizing brick; the active graph drops to a single node.
//! This both compresses (fewer active units) and protects (a nested cell is a frozen brick) — and the banked
//! brick is exactly the reusable intermediate composition (parity-3) needs.
//!
//!   cargo test -p openmorph-core --test proofs substrate_extract -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const Y: usize = 2;
fn label(x0: f32, x1: f32) -> f32 {
    if x0 * x1 > 0.0 {
        1.0
    } else {
        -1.0
    }
}
fn acc(f: &mut Field) -> f64 {
    let mut rng = Rng::new(0x51ee ^ 0xFFFF);
    let mut c = 0usize;
    for _ in 0..1000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], 100);
        if (if f.cells()[Y] > 0.0 { 1.0 } else { -1.0 } - label(x0, x1)).abs() < 1e-6 {
            c += 1;
        }
    }
    c as f64 / 1000.0
}

#[test]
fn extract_collapses_a_skill_to_one_nested_cell() {
    // grow a skill (XOR — needs hidden depth)
    let mut f = Field::new(3);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 16;
    let mut rng = Rng::new(0x51ee);
    for _ in 0..8000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.grow(&[(0, x0), (1, x1)], Y, label(x0, x1), 0.05, 50);
    }
    let a0 = acc(&mut f);
    let u0 = f.n_units();

    // the whole subgraph = the output cell + the grown features feeding it
    let mut skill: Vec<usize> = vec![Y];
    skill.extend(f.target_basis(Y));

    // BANK → SHED → PLANT: encapsulate (copy) the skill, detach the originals, nest the brick as one cell.
    let brick = f.encapsulate(&skill);
    let removed = f.detach(&skill);
    f.nest(brick, &[0, 1], Y, 100);

    let a1 = acc(&mut f);
    let u1 = f.n_units();

    eprintln!(
        "[EXTRACT] skill: {:.0}% in {u0} active units → extracted into a brick: {:.0}% in {u1} active unit(s) (detached {removed})",
        a0 * 100.0,
        a1 * 100.0
    );
    eprintln!("[EXTRACT] VERDICT: encapsulate (copy the matured subgraph) + detach (shed the originals) + nest (plant the brick) collapses a many-unit skill to ONE nested cell, losslessly — sleep-consolidation into a reusable, frozen, memoizing brick. The active graph stays small no matter how many skills accumulate; the brick is the intermediate parity-3 composition needs.");

    assert!(a0 > 0.90, "the skill works before extraction ({a0:.2})");
    assert!(
        a1 > 0.85,
        "the extracted brick reproduces the skill ({a1:.2})"
    );
    assert_eq!(
        u1, 1,
        "the active graph collapsed to a single nested cell (got {u1})"
    );
    assert_eq!(
        removed, u0,
        "detach shed all the original skill units ({removed} vs {u0})"
    );
}
