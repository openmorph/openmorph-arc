// SPDX-License-Identifier: MIT-0
//! REASONING BY FRACTAL COMPOSITION — Sébastien's thesis: program induction is not a missing primitive;
//! it EMERGES from composing crystallized morphs (skills-as-cells) grown by a curriculum. A crystallized
//! skill is a program; reasoning is assembling pattern-recognizers into composed skills; a composed morph
//! becomes a CELL in a bigger morph (micro → macro, fractal). OpenMorph already has the halves — grown
//! transduction (`grow_backbone`) + `consolidate` (a matured skill becomes a permanent cell) + composition.
//!
//! The test uses the classic depth-vs-width separator: HIGH-ORDER PARITY. 6-way parity is orthogonal to
//! every low-degree feature, so a FLAT random-feature readout can't learn it at ANY width — it needs
//! DEPTH. Grown from a CURRICULUM it is easy: three 2-XOR blocks (a⊕b, c⊕d, e⊕f), CONSOLIDATED into cells,
//! then a mid over two of them, then the top — each level a trivial 2-XOR over the blocks below. Same
//! total budget; the fractal organism solves it, the flat one is at chance. Reasoning = assembling grown
//! blocks, inside→outside.
//!
//! RE-BASELINED at the 1.6 learning flip — the DOMINANCE claim does not hold and is `#[ignore]`d
//! (`composition_beats_the_flat_learner_on_six_parity`); the running proof asserts only that the fractal organism solves
//! 6-way parity. Measured: on the 1.6 default FLAT reaches 100 % (FRACTAL 100 %). On the legacy genome FLAT's 66 % is
//! not a depth wall but the RAW delta step diverging on its wide readout (6 + 120 reads): with every unit's learning
//! rate halved (`init_lr` 0.5) legacy FLAT reaches 98 %, at 0.3× 100 %; with only the step-stability cap on and the
//! feature directions still frozen random (`feature_lr` 0) FLAT reaches 100 %. So 6-way parity is NOT out of reach of a
//! flat random-feature readout; the explanation above ("orthogonal to random features at any width") is wrong, and the
//! earlier 66 % → 100 % gap was step instability.
//!
//!   cargo test -p openmorph-core --test proofs substrate_compose_reason -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 6; // input bits
const K: usize = 24; // backbone width per skill
const STEPS: usize = 2500; // learn steps per readout
const BUD: usize = 200;

/// Parity (in ±1) of a slice of bits: −∏ bits.
fn parity(bits: &[f32]) -> f32 {
    -bits.iter().product::<f32>()
}

/// Add a 2-input skill reading `(i, j)` + a grown backbone of them, writing `out`. Returns nothing;
/// the readout is learned by `train`.
fn add_skill(f: &mut Field, i: usize, j: usize, out: usize) {
    let hidden = f.grow_backbone(&[i, j], K, 2.5);
    let mut reads = vec![i, j];
    reads.extend(hidden);
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, out);
}

/// Train the readout writing `out` toward `target(bits)`, presenting ALL N raw bits each step so the
/// whole chain (lower blocks) computes the intermediates.
fn train(f: &mut Field, out: usize, steps: usize, rng: &mut Rng, target: impl Fn(&[f32]) -> f32) {
    for _ in 0..steps {
        let bits: Vec<f32> = (0..N).map(|_| rng.bit()).collect();
        let inp: Vec<(usize, f32)> = bits.iter().enumerate().map(|(k, &v)| (k, v)).collect();
        f.learn_step(&inp, out, target(&bits), 0.05, BUD);
    }
}

fn eval(f: &mut Field, out: usize) -> f64 {
    let mut ok = 0;
    for m in 0..(1u32 << N) {
        let bits: Vec<f32> = (0..N)
            .map(|k| if (m >> k) & 1 == 0 { -1.0 } else { 1.0 })
            .collect();
        let inp: Vec<(usize, f32)> = bits.iter().enumerate().map(|(k, &v)| (k, v)).collect();
        f.stimulate(&inp, BUD);
        if f.cells()[out].signum() == parity(&bits).signum() {
            ok += 1;
        }
    }
    ok as f64 / (1u32 << N) as f64
}

/// FRACTAL: curriculum-grown blocks, consolidated into cells, composed micro→macro.
fn composed() -> f64 {
    // cells 0..N = inputs; 6=P1, 7=P2, 8=P3, 9=M(=P1⊕P2), 10=T(=M⊕P3).
    let (p1, p2, p3, m, t) = (6usize, 7, 8, 9, 10);
    let mut f = Field::new(11);
    let mut rng = Rng::new(0x1);
    // Level 1 — three easy 2-XOR building blocks (the curriculum).
    add_skill(&mut f, 0, 1, p1);
    add_skill(&mut f, 2, 3, p2);
    add_skill(&mut f, 4, 5, p3);
    train(&mut f, p1, STEPS, &mut rng, |b| parity(&b[0..2]));
    train(&mut f, p2, STEPS, &mut rng, |b| parity(&b[2..4]));
    train(&mut f, p3, STEPS, &mut rng, |b| parity(&b[4..6]));
    f.consolidate(&[p1, p2, p3]); // the blocks are now permanent CELLS
                                  // Level 2 — a macro over two blocks (a morph of morphs).
    add_skill(&mut f, p1, p2, m);
    train(&mut f, m, STEPS, &mut rng, |b| parity(&b[0..4]));
    f.consolidate(&[m]);
    // Level 3 — the top over the macro and the third block.
    add_skill(&mut f, m, p3, t);
    train(&mut f, t, STEPS, &mut rng, parity);
    eval(&mut f, t)
}

/// FLAT: one grown backbone over all N bits, same total budget (see the module docs: it learns 6-way parity once its
/// readout step is stable).
fn flat() -> f64 {
    let out = N;
    let mut f = Field::new(N + 1);
    let mut rng = Rng::new(0x1);
    let inputs: Vec<usize> = (0..N).collect();
    let hidden = f.grow_backbone(&inputs, K * 5, 2.5); // same total width as the 5 composed skills
    let mut reads = inputs.clone();
    reads.extend(hidden);
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, out);
    train(&mut f, out, STEPS * 5, &mut rng, parity); // same total steps
    eval(&mut f, out)
}

#[test]
fn reasoning_emerges_from_composing_grown_blocks() {
    let comp = composed();
    let flat = flat();

    eprintln!(
        "[COMPOSE] {N}-way parity, same budget:\n  FLAT (one wide backbone on raw bits)  : {:.0}%  (~50% = chance)\n  FRACTAL (curriculum blocks → macro)   : {:.0}%",
        flat * 100.0,
        comp * 100.0,
    );
    eprintln!("[COMPOSE] VERDICT (Sébastien's fractal thesis): reasoning can be ASSEMBLING grown building blocks. A curriculum grows easy 2-XOR skills; `consolidate` makes them permanent CELLS; a macro grows OVER them and a top over the macro — a morph of morphs — solving 6-way parity. (Since the 1.6 re-baseline this is NOT claimed to beat a flat learner: a flat random-feature readout on a stable step solves it too — see the module docs.)");

    assert!(
        comp >= 0.95,
        "the fractal organism should solve {N}-parity (got {comp:.2})"
    );
}

/// The DOMINANCE half of the original proof, kept verbatim but ignored: on the 1.6 default FLAT reaches 100 %, and the
/// legacy 66 % was raw-step divergence, not a depth wall (see the module docs).
#[test]
#[ignore = "1.6 re-baseline: does not hold — FLAT 100 % on the default genome; the legacy 66 % was raw-step divergence (lr x0.5 -> 98 %); see this file's module docs"]
fn composition_beats_the_flat_learner_on_six_parity() {
    let comp = composed();
    let flat = flat();
    eprintln!("[COMPOSE] fractal {comp:.2} vs flat {flat:.2}");
    assert!(
        comp > flat + 0.3,
        "composition must clear the depth wall the flat substrate can't (fractal {comp:.2} vs flat {flat:.2})"
    );
}
