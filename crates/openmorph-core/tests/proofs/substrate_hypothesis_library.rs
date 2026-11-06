// SPDX-License-Identifier: MIT-0
//! #150 — native hypothesis-testing, PHASE 2: HYPOTHESIS OVER A SKILL LIBRARY (the third faculty).
//! MorphLab (gate4_policy_skills): a BLANK morph learning a DEEP target (parity-6) from RAW inputs is stuck at
//! chance (the cold-start wall) — but a SKILLED morph that reads a prior-skill library (the parity-3 halves)
//! hits 100%, because the library turns a deep search into a SHALLOW "which two skills compose?" question.
//! This proves it on the substrate, reusing GROWN skills + `encapsulate`/`nest` (composition) — zero new
//! primitive. parity-6 over ±1 bits = the PRODUCT of all six = parity3(a,b,c)·parity3(d,e,f): a shallow
//! product of two skill outputs.
//!
//! RE-BASELINED at the 1.6 learning flip — the GAP claim (SKILLED beats BLANK by > 25 points) does not hold and is
//! `#[ignore]`d (`the_library_opens_a_gap_over_a_blank_learner`); the running proof asserts that the library skills
//! generalize and that composing them cracks parity-6. Measured (BLANK / SKILLED held-out %, 12 distinct seeds, each
//! seeding the data streams and the growth PRNG): on the 1.6 default BLANK cracks parity-6 on 12/12 seeds (100 / 100).
//! The "cold-start wall" is the legacy learner's raw delta step with fan-in-blind init at lr 0.05: legacy BLANK is
//! 66 % (0/12 seeds crack; SKILLED 100 %), but with every unit's learning rate halved (`init_lr` 0.5) legacy BLANK
//! cracks 5/6 seeds (94 %), at 0.3× 6/6 (100 %); switching on only the step-stability cap lifts it to 100 % (6/6), only
//! the c/√fan-in init to 89 % (4/6). (At this file's single seed legacy BLANK already cracked parity-6 — 100 % vs
//! 100 % — which is why the test was red before the flip.)
use openmorph_core::substrate::Field;

use crate::common::Rng;

fn parity(x: &[f32]) -> f32 {
    x.iter().product::<f32>().signum() // ±1 parity = product of the bits
}

/// Grow a parity-3 skill over 3 input cells → cell 3, from random ±1 examples.
fn grow_parity3(seed: u64) -> Field {
    let mut f = Field::new(4);
    f.config_mut().max_hidden = 48;
    f.config_mut().feature_lr = 0.3;
    let mut rng = Rng::new(seed);
    for _ in 0..12000 {
        let x: Vec<f32> = (0..3)
            .map(|_| if rng.below(2) == 0 { -1.0 } else { 1.0 })
            .collect();
        f.grow(&[(0, x[0]), (1, x[1]), (2, x[2])], 3, parity(&x), 0.05, 60);
    }
    f
}

fn accuracy(predict: &mut dyn FnMut(&[f32]) -> f32, seed: u64, n: usize) -> f32 {
    let mut rng = Rng::new(seed);
    let mut ok = 0usize;
    for _ in 0..n {
        let x: Vec<f32> = (0..6)
            .map(|_| if rng.below(2) == 0 { -1.0 } else { 1.0 })
            .collect();
        if predict(&x).signum() == parity(&x) {
            ok += 1;
        }
    }
    100.0 * ok as f32 / n as f32
}

/// BLANK — learn parity-6 from RAW 6 inputs (cold start), the deep target. Returns held-out accuracy (%).
fn blank_accuracy() -> f32 {
    // ── BLANK — learn parity-6 from RAW 6 inputs (cold start), the deep target. ──
    let mut blank = Field::new(7);
    blank.config_mut().max_hidden = 96;
    blank.config_mut().feature_lr = 0.3;
    let mut rng = Rng::new(0xB1A);
    for _ in 0..20000 {
        let x: Vec<f32> = (0..6)
            .map(|_| if rng.below(2) == 0 { -1.0 } else { 1.0 })
            .collect();
        let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        blank.grow(&inp, 6, parity(&x), 0.05, 120);
    }
    let blank_acc = {
        let mut f = |x: &[f32]| {
            let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            blank.stimulate(&inp, 120);
            blank.cells()[6]
        };
        accuracy(&mut f, 0xBEEF, 600)
    };

    blank_acc
}

/// SKILLED — a library of two GROWN parity-3 skills composed by `nest`; the hypothesis = compose them, learn the
/// shallow product. Returns (library skill held-out %, composed held-out %).
fn skilled_accuracy() -> (f32, f32) {
    // ── SKILLED — a library of two GROWN parity-3 skills; hypothesis = compose them, learn the shallow product. ──
    let s1 = grow_parity3(0x5111); // parity3(a,b,c)
    let s2 = grow_parity3(0x5222); // parity3(d,e,f)
                                   // sanity: the library skills themselves generalize (else the composition can't work)
    let mut s1c = s1.clone();
    let s1_acc = {
        let mut f = |x: &[f32]| {
            s1c.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2])], 60);
            s1c.cells()[3]
        };
        let mut rng = Rng::new(0x9);
        let mut ok = 0;
        for _ in 0..300 {
            let x: Vec<f32> = (0..3)
                .map(|_| if rng.below(2) == 0 { -1.0 } else { 1.0 })
                .collect();
            if f(&x).signum() == parity(&x) {
                ok += 1;
            }
        }
        100.0 * ok as f32 / 300.0
    };

    // compose: nest the two skills into a host; skill1 reads cells 0,1,2→s1out, skill2 reads 3,4,5→s2out;
    // then GROW a shallow readout over [s1out, s2out] toward parity-6 (= their product).
    let mut host = Field::new(6); // 6 raw inputs
    let s1out = host.nest(s1.clone(), &[0, 1, 2], host.len(), 60);
    let _ = s1out;
    let s1cell = host.len() - 1;
    let s2cell_pre = host.len();
    host.nest(s2.clone(), &[3, 4, 5], s2cell_pre, 60);
    let s2cell = host.len() - 1;
    host.config_mut().max_hidden = 32;
    host.config_mut().feature_lr = 0.3;
    let out = host.len();
    host.ensure_cells(out + 1);
    let mut rng = Rng::new(0x5C1);
    for _ in 0..8000 {
        let x: Vec<f32> = (0..6)
            .map(|_| if rng.below(2) == 0 { -1.0 } else { 1.0 })
            .collect();
        let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        // reading the skill cells settles the sub-morphs; grow the composition over their outputs
        host.stimulate(&inp, 120);
        let (a, b) = (host.cells()[s1cell], host.cells()[s2cell]);
        host.grow(&[(s1cell, a), (s2cell, b)], out, parity(&x), 0.05, 60);
    }
    let skilled_acc = {
        let mut f = |x: &[f32]| {
            let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            host.stimulate(&inp, 120);
            host.cells()[out]
        };
        accuracy(&mut f, 0xBEEF, 600)
    };

    (s1_acc, skilled_acc)
}

#[test]
fn a_skill_library_cracks_the_deep_parity_wall() {
    let (s1_acc, skilled_acc) = skilled_accuracy();

    eprintln!("[LIB] parity-6 (deep target, ±1 bits):");
    eprintln!("[LIB]   library skill parity-3 held-out: {s1_acc:.0}%");
    eprintln!("[LIB]   SKILLED (composes the 2 parity-3 skills):       {skilled_acc:.0}%");

    assert!(
        s1_acc > 90.0,
        "the library's parity-3 skills must generalize (else composition can't work): {s1_acc:.0}%"
    );
    assert!(
        skilled_acc > 90.0,
        "composing the library skills must crack parity-6: {skilled_acc:.0}%"
    );

    eprintln!("[LIB] VERDICT: HYPOTHESIS OVER A LIBRARY — on the substrate, zero new primitive: a SKILLED morph that COMPOSES two grown parity-3 skills (`nest` the library skills, grow the shallow product over their outputs) reaches {skilled_acc:.0}%. The library turns a deep search into a shallow 'which skills compose?' — MorphLab's gate4_policy_skills, on the `encapsulate`/`nest` + router (`index_skills`/`route`) already shipped. (Since the 1.6 re-baseline this is NOT claimed to beat a blank learner: the 1.6 default cracks parity-6 from raw inputs too — see the module docs.)");
}

/// The GAP half of the original proof, kept verbatim but ignored: on the 1.6 default BLANK cracks parity-6 on 12/12
/// seeds, and the legacy cold-start wall exists only at the raw step's lr 0.05 (see the module docs).
#[test]
#[ignore = "1.6 re-baseline: does not hold — BLANK cracks parity-6 on 12/12 seeds on the default genome; the legacy wall was the raw step at lr 0.05 (lr x0.5 -> 94 %); see this file's module docs"]
fn the_library_opens_a_gap_over_a_blank_learner() {
    let blank_acc = blank_accuracy();
    let (_, skilled_acc) = skilled_accuracy();
    eprintln!("[LIB] BLANK {blank_acc:.0}% vs SKILLED {skilled_acc:.0}%");
    assert!(
        skilled_acc > blank_acc + 25.0,
        "the library must open a real gap over cold-start blank ({skilled_acc:.0}% vs {blank_acc:.0}%)"
    );
}
