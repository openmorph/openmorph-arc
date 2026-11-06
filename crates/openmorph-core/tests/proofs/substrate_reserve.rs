// SPDX-License-Identifier: MIT-0
//! NEWBORN ENERGY RESERVE — the "yolk" (MorphLab #92, ask #3). Emergence needs development time at
//! birth: a young unit hasn't wired in or tuned yet, so if the alive economy judges it immediately it
//! starves before it can become useful. `cull_after(death, grace)` spares a unit younger than `grace`
//! events — a birth reserve, measured in the organism's OWN subjective time.
//!
//! The probe: a background unit keeps the organism (and its clock) alive; a newborn is SILENT at first
//! (not yet earning) then becomes active (fires, earns). Without the reserve it is culled while silent
//! and is gone before it can contribute; with the reserve it survives its development and thrives.
//!
//!   cargo test -p openmorph-core --test proofs substrate_reserve -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const SILENT: usize = 6; // cycles the newborn is not yet earning (its development window)
const ACTIVE: usize = 24; // cycles once it becomes useful

/// Returns whether the newborn (the unit writing cell 1) is still alive at the end.
fn newborn_survives(grace: u64) -> bool {
    let mut f = Field::new(4);
    // A background unit, always driven — it keeps the organism alive and ticks its subjective clock.
    f.add_unit(vec![2], vec![1.0], 0.0, 3);
    // The newborn: reads cell 0, writes cell 1. Silent until its input starts changing.
    f.add_unit(vec![0], vec![1.0], 0.0, 1);

    // Phase 1 — the newborn is silent (cell 0 stable); only the background fires and earns.
    for t in 0..SILENT {
        let bg = if t % 2 == 0 { 1.0 } else { -1.0 };
        f.stimulate(&[(2, bg)], 10);
        f.metabolize(2.0, 1.0); // fired: +2 −1 ; silent newborn: −1 (it starves)
        f.cull_after(0.0, grace);
    }
    // Phase 2 — the newborn's input becomes active: it fires and earns (if it survived to here).
    for t in 0..ACTIVE {
        let v = if t % 2 == 0 { 1.0 } else { -1.0 };
        f.stimulate(&[(0, v), (2, v)], 10);
        f.metabolize(2.0, 1.0);
        f.cull_after(0.0, grace);
    }
    f.units.iter().any(|u| u.write == 1)
}

#[test]
fn a_birth_reserve_lets_a_newborn_develop_before_it_must_earn() {
    let without = newborn_survives(0);
    let with = newborn_survives(50);

    eprintln!(
        "[RESERVE] a newborn that is not yet earning, over its development window:\n  WITHOUT reserve (grace 0) : survives = {without}\n  WITH reserve   (grace 50) : survives = {with}"
    );
    eprintln!("[RESERVE] VERDICT (rc9 — the yolk): `cull_after(death, grace)` spares a unit younger than `grace` events, measured in the organism's own subjective time. A newborn gets development time to wire in and tune before the economy is allowed to starve it — so emergence isn't strangled at birth. The birth reserve composes the alive economy with subjective time; over-provision-then-cull can now keep the winning ticket instead of killing it in the cradle.");

    assert!(
        !without,
        "without a reserve the silent newborn should be culled before it can earn"
    );
    assert!(
        with,
        "with a reserve the newborn should survive its development and thrive"
    );
}
