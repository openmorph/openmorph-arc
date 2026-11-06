// SPDX-License-Identifier: MIT-0
//! ONE EVENT, ONE SETTLE — `grow` settles its inputs ONCE, then learns on that settled state.
//!
//! `grow` used to settle, birth, then call `learn_step`, which settled the SAME inputs a second time. The second
//! settle was not free: it advanced every resonator a second step (a band-pass bank rang at twice its tuned rate
//! under `grow` / `alive(Perceive)`), re-ticked the temporal traces, and wiped `last_fired` — so the firing economy
//! (`metabolize`) never paid income after a `grow` (the units that did the work read as idle). Measured on the
//! parent commit: 1 `grow` = 2 resonator steps (u = 1.5 instead of 1.0), and `grow` + `metabolize(income 1)` left
//! the writer at its birth energy 3.00 instead of 4.00.
//!
//! The second settle DID do one thing that must survive: it re-applied every supplied input the first cascade had
//! overwritten, so a readout grown over a TEACHER-FORCED cell (one a unit also writes — a committed carry) learned on
//! the caller's value. `grow` now re-clamps those inputs (no second temporal tick) before it learns.

use openmorph_core::Field;

#[test]
fn grow_advances_the_resonators_once_per_event_like_stimulate() {
    // freq 0, radius 0.5 ⇒ u' = 0.5·u + x: after n events on x = 1, u = (1 − 0.5ⁿ)/(1 − 0.5).
    let mut grown = Field::new(3);
    let rg = grown.resonate(0, 0.0, 0.5);
    let mut settled = Field::new(3);
    let rs = settled.resonate(0, 0.0, 0.5);
    for n in 1..=3 {
        grown.grow(&[(0, 1.0)], 2, 0.3, 0.1, 50);
        settled.stimulate(&[(0, 1.0)], 50);
        let (ug, us) = (grown.cells()[rg], settled.cells()[rs]);
        println!("after {n} event(s): resonator u under grow {ug:.4}, under stimulate {us:.4}");
        assert_eq!(
            ug.to_bits(),
            us.to_bits(),
            "one grow must advance a resonator exactly one step, like one stimulate (event {n})"
        );
    }
}

#[test]
fn the_units_a_grow_fired_earn_their_income() {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0;
    f.grow(&[(0, 1.0)], 1, 0.5, 0.1, 50); // births the writer; the input changed, so it fires
    let birth = f.energy_of(0);
    f.metabolize(1.0, 0.0);
    let after = f.energy_of(0);
    println!("writer energy: birth {birth:.2} → after grow + metabolize(income 1) {after:.2}");
    assert_eq!(
        after,
        birth + 1.0,
        "the writer fired in grow's settle, so the firing economy must pay it"
    );
}

#[test]
fn grow_learns_on_the_teacher_forced_inputs_the_caller_supplied() {
    // U writes cell 1 from cell 0 (a frozen predictor); `grow` supplies cell 1 as a teacher-forced input, which U
    // overwrites whenever cell 0 changes. The readout must learn on the SUPPLIED value.
    let mut f = Field::new(3);
    f.config_mut().max_hidden = 0;
    f.add_unit(vec![0], vec![0.8], 0.0, 1);
    f.consolidate(&[1]);
    let mut off = 0;
    for t in 0..200 {
        let x = if t % 2 == 0 { 0.5 } else { -0.5 };
        let forced = if t % 3 == 0 { 1.0 } else { -1.0 };
        f.grow(&[(0, x), (1, forced)], 2, 0.6 * forced, 0.1, 100);
        if f.cells()[1] != forced {
            off += 1; // the readout just learned on a value the caller never supplied
        }
    }
    let w = f.units[1].weights[1];
    println!(
        "grow learned on a non-supplied cell 1 in {off}/200 events; readout weight on it {w:.3}"
    );
    assert_eq!(off, 0, "grow must learn on the supplied inputs");
    assert!(
        w > 0.3,
        "the readout must learn the teacher-forced relation ({w})"
    );
}
