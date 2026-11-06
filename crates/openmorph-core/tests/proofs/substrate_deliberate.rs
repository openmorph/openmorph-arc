// SPDX-License-Identifier: MIT-0
//! DELIBERATION — the reasoner past the one-shot-selection wall (MorphLab #104; resolves the
//! `substrate_assemble` residual). One-shot learned selection PREDICTS the composition in a single pass
//! and caps (~81%): you cannot know a block belongs until you TRY it and CHECK. Selection-by-VERIFICATION
//! doesn't cap. The organism PROPOSES a candidate composition, SIMULATES it internally (runs the grown
//! cells on the example inputs — imagination, no environment), VERIFIES fit to the examples, and keeps
//! thinking until one candidate is CONFIDENT — then commits, or ABSTAINS if none fits. System-1 proposes,
//! System-2 verifies.
//!
//! Built on primitives already here: grown blocks (`grow_backbone`/`consolidate`), `stimulate` as internal
//! imagination, and a fit score for verification. In-library rules solve 100% (vs 81% one-shot); a rule
//! OUTSIDE the library is refused — the organism knows when it doesn't know.
//!
//!   cargo test -p openmorph-core --test proofs substrate_deliberate -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 6;
const K: usize = 48;
const STEPS: usize = 4000;
const BUD: usize = 500;

fn parity(bits: &[f32]) -> f32 {
    -bits.iter().product::<f32>()
}
fn sub(bits: &[f32], idx: &[usize]) -> Vec<f32> {
    idx.iter().map(|&k| bits[k]).collect()
}

fn grow_block(f: &mut Field, inputs: &[usize], out: usize, rng: &mut Rng) {
    let hidden = f.grow_backbone(inputs, K, 2.5);
    let mut reads = inputs.to_vec();
    reads.extend(hidden);
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, out);
    for _ in 0..STEPS {
        let bits: Vec<f32> = (0..N).map(|_| rng.bit()).collect();
        let inp: Vec<(usize, f32)> = bits.iter().enumerate().map(|(k, &v)| (k, v)).collect();
        f.learn_step(&inp, out, parity(&sub(&bits, inputs)), 0.05, BUD);
    }
}

/// A grown library: relevant blocks (3-parity of the two halves) + distractors that all avoid bit 5.
fn library(rng: &mut Rng) -> (Field, Vec<usize>) {
    let mut f = Field::new(N);
    let mut lib = Vec::new();
    for s in [
        [0usize, 1, 2].as_slice(),
        &[3, 4, 5],
        &[0, 1, 3],
        &[1, 2, 4],
        &[0, 2, 4],
    ] {
        let c = f.len();
        grow_block(&mut f, s, c, rng);
        lib.push(c);
    }
    f.consolidate(&lib);
    (f, lib) // lib[0]=r0(012), lib[1]=r1(345), rest distractors
}

/// Read a block's output for input `x` — internal imagination (run the grown cell, no environment).
fn imagine(f: &mut Field, block: usize, x: &[f32]) -> f32 {
    let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(k, &v)| (k, v)).collect();
    f.stimulate(&inp, BUD);
    f.cells()[block]
}

/// DELIBERATE: propose candidate compositions (a single block, or the XOR of two), SIMULATE each on the
/// example set, VERIFY its fit, and commit to the first that is confident (fits every example) — else
/// ABSTAIN. Returns (chosen composition, thoughts = candidates simulated).
fn deliberate(
    f: &mut Field,
    lib: &[usize],
    examples: &[(Vec<f32>, f32)],
) -> (Option<Vec<usize>>, usize) {
    let mut thoughts = 0;
    // Length-1 candidates first (System-2 tries the cheapest programs first).
    for &b in lib {
        thoughts += 1;
        if examples
            .iter()
            .all(|(x, y)| imagine(f, b, x).signum() == y.signum())
        {
            return (Some(vec![b]), thoughts);
        }
    }
    // Length-2 candidates: the XOR (composition) of two blocks.
    // NB the compose operator here is AUTHORED (`-bi*bj` = XOR) — a deliberate simplification isolated to this
    // harness. It is NOT the substrate claim: `substrate_operator` proves this exact operator is GROWN, not
    // hand-coded, so the honest end-to-end reasoner composes via `grow_operator`. What THIS proof isolates is
    // the deliberate → simulate → verify → abstain loop; the operator is stubbed to keep that loop legible.
    for i in 0..lib.len() {
        for j in (i + 1)..lib.len() {
            thoughts += 1;
            let fits = examples.iter().all(|(x, y)| {
                let (bi, bj) = (imagine(f, lib[i], x), imagine(f, lib[j], x));
                (-bi * bj).signum() == y.signum() // compose = XOR of the two grown cells (authored op, see above)
            });
            if fits {
                return (Some(vec![lib[i], lib[j]]), thoughts);
            }
        }
    }
    (None, thoughts) // nothing fit — abstain
}

/// Held-out accuracy of a chosen composition over all inputs.
fn verify(f: &mut Field, chosen: &[usize], target: impl Fn(&[f32]) -> f32) -> f64 {
    let mut ok = 0;
    for msk in 0..(1u32 << N) {
        let x: Vec<f32> = (0..N)
            .map(|k| if (msk >> k) & 1 == 0 { -1.0 } else { 1.0 })
            .collect();
        let out = match chosen {
            [b] => imagine(f, *b, &x),
            [i, j] => -imagine(f, *i, &x) * imagine(f, *j, &x),
            _ => 0.0,
        };
        if out.signum() == target(&x).signum() {
            ok += 1;
        }
    }
    ok as f64 / (1u32 << N) as f64
}

fn examples_of(rng: &mut Rng, target: impl Fn(&[f32]) -> f32, n: usize) -> Vec<(Vec<f32>, f32)> {
    (0..n)
        .map(|_| {
            let x: Vec<f32> = (0..N).map(|_| rng.bit()).collect();
            let y = target(&x);
            (x, y)
        })
        .collect()
}

#[test]
fn deliberation_solves_selection_and_abstains_when_it_should() {
    let mut rng = Rng::new(0xD1B);
    let (mut f, lib) = library(&mut rng);

    // IN-LIBRARY rule: 6-parity = XOR of the two relevant blocks. Infer it from a few examples.
    let ex = examples_of(&mut rng, parity, 24);
    let (chosen, thoughts) = deliberate(&mut f, &lib, &ex);
    let acc = chosen.as_ref().map_or(0.0, |c| verify(&mut f, c, parity));

    // OUT-OF-LIBRARY rule: 5-parity of bits 0..4 — not any block, not any XOR of two. Must ABSTAIN.
    let oob_target = |b: &[f32]| parity(&b[0..5]);
    let ex_oob = examples_of(&mut rng, oob_target, 24);
    let (chosen_oob, thoughts_oob) = deliberate(&mut f, &lib, &ex_oob);

    eprintln!(
        "[DELIBERATE] think-until-confident over a grown library ({} blocks):\n  IN-LIBRARY (6-parity): chosen = {:?}, held-out acc {:.0}%, thought {thoughts} candidates\n  OUT-OF-LIBRARY (5-parity): {} after thinking {thoughts_oob} candidates",
        lib.len(),
        chosen,
        acc * 100.0,
        if chosen_oob.is_none() { "ABSTAINED (0 wrong acts)" } else { "acted (WRONG)" },
    );
    eprintln!("[DELIBERATE] VERDICT (MorphLab #104 — the reasoner): the one-shot selection wall (~81%) is a REACTION problem; DELIBERATION resolves it. The organism proposes a composition, SIMULATES it internally on the examples (imagination — `stimulate` the grown cells, no environment), VERIFIES fit, and keeps thinking until confident — selection-by-verification, so it solves 100% where one-shot capped. On a rule OUTSIDE the library it stays unconfident and ABSTAINS — it knows when it doesn't know. The `substrate_assemble` gate is best used as a PROPOSER that orders which candidates to simulate first (System-1 intuition), while verification (System-2) keeps accuracy exact. This is reasoning on the primitives already here — grow/compose + `stimulate` as imagination + a confidence gate. TRUTH-NOT-TROPHY: the compose operator used above is AUTHORED (`-bi*bj`), a simplification isolated to this harness so the deliberate/verify/abstain loop stays legible; that operator is GROWN in `substrate_operator`, so the honest full reasoner composes via `grow_operator`. The substrate claim here is the loop, not the operator.");

    assert!(
        chosen.is_some(),
        "deliberation must find the in-library composition"
    );
    assert!(
        acc >= 0.99,
        "and it must be exactly right (verified), got {acc:.2}"
    );
    assert!(
        chosen_oob.is_none(),
        "an out-of-library rule must be ABSTAINED, not acted on wrongly"
    );
}
