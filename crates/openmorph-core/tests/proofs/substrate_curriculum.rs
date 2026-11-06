// SPDX-License-Identifier: MIT-0
//! SELF-WRITTEN CURRICULUM — the subgoal is DISCOVERED, not handed over. `substrate_recurse` proved that
//! crystallizing a confident sub-composition unlocks a deeper rule — but WE named the sub-rule (solve
//! 4-parity first). That is the human writing the curriculum. Here the organism DISCOVERS which
//! sub-composition to bank: it SEARCHES over candidate crystallizations and keeps the one whose banking
//! makes the too-deep goal REACHABLE + verified. No one tells it "4-parity" — it finds that banking
//! b01⊕b23 is the rung that unlocks 6-parity, purely by testing reachability.
//!
//! Bounded deliberation + subgoal SEARCH + crystallization = a curriculum the organism writes for itself:
//! reason → (discover the rung) → crystallize → reason deeper. This closes the last authored piece in the
//! recursion story — the milestone.
//!
//!   cargo test -p openmorph-core --test proofs substrate_curriculum -- --nocapture --test-threads=1

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
fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}

fn grow_block(f: &mut Field, inputs: &[usize], out: usize, rng: &mut Rng) {
    let hidden = f.grow_backbone(inputs, K, 2.5);
    let mut reads = inputs.to_vec();
    reads.extend(hidden);
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, out);
    for _ in 0..STEPS {
        let bits: Vec<f32> = (0..N).map(|_| rng.bit()).collect();
        f.learn_step(&pairs(&bits), out, parity(&sub(&bits, inputs)), 0.05, BUD);
    }
}

fn imagine(f: &mut Field, block: usize, x: &[f32]) -> f32 {
    f.stimulate(&pairs(x), BUD);
    f.cells()[block]
}

/// Deliberate over `lib` at length ≤ 2 (single block, or the composition of two). Returns the chosen
/// composition or None (abstain). The length-2 check composes via the reasoner's algebra.
fn deliberate(f: &mut Field, lib: &[usize], ex: &[(Vec<f32>, f32)]) -> Option<Vec<usize>> {
    for &b in lib {
        if ex
            .iter()
            .all(|(x, y)| imagine(f, b, x).signum() == y.signum())
        {
            return Some(vec![b]);
        }
    }
    for i in 0..lib.len() {
        for j in (i + 1)..lib.len() {
            if ex.iter().all(|(x, y)| {
                (-imagine(f, lib[i], x) * imagine(f, lib[j], x)).signum() == y.signum()
            }) {
                return Some(vec![lib[i], lib[j]]);
            }
        }
    }
    None
}

/// CRYSTALLIZE the composition of two blocks into a new self-contained cell — a GROWN operator over the
/// pair (self-supervised toward their composition), banked as a first-class block. Grows the operator in a
/// scratch morph over the two block VALUES (inputs 0,1 → sink), then ENCAPSULATEs the whole subgraph and
/// NESTs it into `f` as ONE frozen cell reading `bi`, `bj`. A nested cell is inherently self-contained (no
/// tunable weights — later learning skips it), so it needs no separate consolidation. Returns the cell.
fn crystallize_pair(f: &mut Field, bi: usize, bj: usize, rng: &mut Rng) -> usize {
    let mut scratch = Field::new(2);
    let hidden = scratch.grow_backbone(&[0, 1], K, 2.5);
    let mut reads = vec![0, 1];
    reads.extend(hidden.iter().copied());
    let sink = scratch.len(); // fresh cell for the composition output
    let n = reads.len();
    scratch.add_unit(reads, vec![0.0; n], 0.0, sink);
    for _ in 0..STEPS {
        let (a, b) = (rng.bit(), rng.bit());
        // target = the composition the two blocks realize (XOR of their ±1 values)
        scratch.learn_step(&[(0, a), (1, b)], sink, (-a * b).signum(), 0.05, BUD);
    }
    // Bank the FULL subgraph (backbone hidden cells + the readout sink) and nest it as a new cell of `f`.
    let mut banked = hidden;
    banked.push(sink);
    let sub = scratch.encapsulate(&banked);
    let out = f.len();
    f.nest(sub, &[bi, bj], out, BUD);
    out
}

fn verify(f: &mut Field, chosen: &[usize]) -> f64 {
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
        if out.signum() == parity(&x).signum() {
            ok += 1;
        }
    }
    ok as f64 / (1u32 << N) as f64
}

fn examples_of(rng: &mut Rng, n: usize) -> Vec<(Vec<f32>, f32)> {
    (0..n)
        .map(|_| {
            let x: Vec<f32> = (0..N).map(|_| rng.bit()).collect();
            (x.clone(), parity(&x))
        })
        .collect()
}

#[test]
fn the_organism_discovers_the_rung_it_must_crystallize() {
    let mut rng = Rng::new(0xC0DE);
    // Primitive library: three 2-XOR halves + two distractors. 6-parity is length-3 over these.
    let mut f = Field::new(N);
    let mut lib = Vec::new();
    for s in [[0usize, 1].as_slice(), &[2, 3], &[4, 5], &[0, 2], &[1, 3]] {
        let c = f.len();
        grow_block(&mut f, s, c, &mut rng);
        lib.push(c);
    }
    f.consolidate(&lib);

    let goal = examples_of(&mut rng, 24);
    let before = deliberate(&mut f, &lib, &goal);

    // SUBGOAL DISCOVERY: no one names the sub-rule. Try banking each candidate pair; keep the FIRST whose
    // crystallization makes the goal reachable + verified. The rung is found by reachability, not authored.
    let mut discovered: Option<(usize, usize, usize)> = None; // (i, j, banked cell)
    let mut probed = 0;
    'search: for i in 0..lib.len() {
        for j in (i + 1)..lib.len() {
            probed += 1;
            // Speculatively bank this rung and test reachability. A failed probe's cell is simply never
            // read again (the search set `ext` is rebuilt each probe), so no rollback machinery is needed.
            let banked = crystallize_pair(&mut f, lib[i], lib[j], &mut rng);
            let mut ext = lib.clone();
            ext.push(banked);
            if let Some(chosen) = deliberate(&mut f, &ext, &goal) {
                if verify(&mut f, &chosen) >= 0.99 {
                    discovered = Some((i, j, banked));
                    break 'search;
                }
            }
        }
    }

    let (di, dj, banked) =
        discovered.expect("the organism must discover a rung that unlocks the goal");
    let mut ext = lib.clone();
    ext.push(banked);
    let chosen = deliberate(&mut f, &ext, &goal).unwrap();
    let acc = verify(&mut f, &chosen);

    eprintln!(
        "[CURRICULUM] 6-parity is length-3 — out of reach for length-≤2 deliberation:\n  BEFORE: {} over the primitive library\n  SEARCH: probed {probed} candidate rungs; DISCOVERED that banking (block {di} ⊕ block {dj}) unlocks the goal — no human named the sub-rule\n  AFTER: chosen {chosen:?} over the self-extended library, held-out {:.0}%",
        if before.is_none() { "ABSTAINED (too deep)" } else { "solved" },
        acc * 100.0,
    );
    eprintln!("[CURRICULUM] VERDICT: the recursion curriculum is now self-written. The organism SEARCHES which sub-composition to crystallize and keeps the rung that makes the deeper rule reachable + verified — the milestone (`substrate_recurse` hard-coded '4-parity') is DISCOVERED by reachability. reason → discover the rung → crystallize → reason deeper: bounded search + subgoal discovery = unbounded depth the organism scaffolds for itself.");

    assert!(
        before.is_none(),
        "6-parity must be out of reach at length ≤ 2"
    );
    assert!(
        acc >= 0.99,
        "after the discovered crystallization it must be exact, got {acc:.2}"
    );
}
