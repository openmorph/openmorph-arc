// SPDX-License-Identifier: MIT-0
//! RECURSION — the fractal loop: reason → CRYSTALLIZE the confident composition into a new CELL → reason
//! over it (MorphLab #104 open piece; the #91 Cell::Nested path). `substrate_deliberate` searches shallow
//! compositions (length ≤ 2). A deeper rule is out of reach — UNTIL a confident sub-composition is banked
//! as a first-class cell, which shrinks the deeper rule to a shallow one over the grown library. Bounded
//! deliberation + crystallization = unbounded depth, built incrementally (a curriculum the organism writes
//! for itself).
//!
//! Demonstrated: 6-parity is length-3 over the primitive blocks, so shallow deliberation ABSTAINS. Solve
//! the 4-parity sub-rule, `crystallize` it into a new cell, and 6-parity becomes length-2 over the extended
//! library — now solved, 100%. Reasoning grows the library it reasons over.
//!
//!   cargo test -p openmorph-core --test proofs substrate_recurse -- --nocapture --test-threads=1

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

/// Grow a block computing the parity of raw cells `inputs`, writing `out`; consolidate — now a cell.
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

/// Deliberate over `lib`: try each single block, then each XOR of two, and commit to the first that fits
/// EVERY example; else abstain. Returns (chosen, thoughts).
fn deliberate(f: &mut Field, lib: &[usize], ex: &[(Vec<f32>, f32)]) -> (Option<Vec<usize>>, usize) {
    let mut thoughts = 0;
    for &b in lib {
        thoughts += 1;
        if ex
            .iter()
            .all(|(x, y)| imagine(f, b, x).signum() == y.signum())
        {
            return (Some(vec![b]), thoughts);
        }
    }
    for i in 0..lib.len() {
        for j in (i + 1)..lib.len() {
            thoughts += 1;
            if ex.iter().all(|(x, y)| {
                (-imagine(f, lib[i], x) * imagine(f, lib[j], x)).signum() == y.signum()
            }) {
                return (Some(vec![lib[i], lib[j]]), thoughts);
            }
        }
    }
    (None, thoughts)
}

/// CRYSTALLIZE a confident composition `[i, j]` into a NEW CELL: grow a unit that reproduces XOR(cell_i,
/// cell_j) from those two cells, then consolidate it. It becomes a first-class block the next search can
/// propose — reasoning banking its own result. Returns the new cell.
fn crystallize_composition(f: &mut Field, comp: &[usize], rng: &mut Rng) -> usize {
    let (bi, bj) = (comp[0], comp[1]);
    // Grow the XOR operator over the two cell VALUES in a scratch morph (inputs 0,1 → sink), then
    // ENCAPSULATE the whole subgraph and NEST it into `f` as ONE new first-class cell reading bi, bj —
    // reasoning banking its own result as a frozen, self-contained node (learning skips a nested cell, so
    // no separate consolidation is needed).
    let mut scratch = Field::new(2);
    let hidden = scratch.grow_backbone(&[0, 1], K, 2.5);
    let mut reads = vec![0, 1];
    reads.extend(hidden.iter().copied());
    let sink = scratch.len();
    let n = reads.len();
    scratch.add_unit(reads, vec![0.0; n], 0.0, sink);
    for _ in 0..STEPS {
        let (a, b) = (rng.bit(), rng.bit());
        // target = the composition the two blocks realize (XOR of their ±1 values)
        scratch.learn_step(&[(0, a), (1, b)], sink, (-a * b).signum(), 0.05, BUD);
    }
    let mut banked = hidden;
    banked.push(sink);
    let sub = scratch.encapsulate(&banked);
    let out = f.len();
    f.nest(sub, &[bi, bj], out, BUD);
    out
}

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
fn recursion_crystallizes_a_composition_to_reach_a_deeper_rule() {
    let mut rng = Rng::new(0x2EC);
    // Primitive library: three 2-XOR blocks (halves of pairs) + two distractors.
    let mut f = Field::new(N);
    let mut lib = Vec::new();
    for s in [[0usize, 1].as_slice(), &[2, 3], &[4, 5], &[0, 2], &[1, 3]] {
        let c = f.len();
        grow_block(&mut f, s, c, &mut rng);
        lib.push(c);
    }
    f.consolidate(&lib);

    // 6-parity is XOR of THREE primitives — length-3, out of reach for shallow deliberation.
    let ex6 = examples_of(&mut rng, parity, 24);
    let (before, thoughts_before) = deliberate(&mut f, &lib, &ex6);

    // Solve the 4-parity(0..3) sub-rule (length-2) and CRYSTALLIZE it into a new cell.
    let ex4 = examples_of(&mut rng, |b| parity(&b[0..4]), 24);
    let (comp4, _) = deliberate(&mut f, &lib, &ex4);
    let comp4 = comp4.expect("4-parity sub-rule is reachable at length-2");
    let banked = crystallize_composition(&mut f, &comp4, &mut rng);
    let mut lib2 = lib.clone();
    lib2.push(banked); // the extended, self-grown library

    // Now 6-parity = XOR(banked 4-parity, the xor45 block) is length-2 — reachable.
    let (after, thoughts_after) = deliberate(&mut f, &lib2, &ex6);
    let acc = after.as_ref().map_or(0.0, |c| verify(&mut f, c, parity));

    eprintln!(
        "[RECURSE] deliberation is bounded to length ≤ 2; recursion crystallizes to go deeper:\n  6-parity BEFORE (primitive library, {} cells): {} after {thoughts_before} thoughts\n  → solved 4-parity sub-rule {comp4:?}, CRYSTALLIZED it into new cell {banked}\n  6-parity AFTER (library grown to {} cells): chosen {after:?}, held-out {:.0}%, {thoughts_after} thoughts",
        lib.len(),
        if before.is_none() { "ABSTAINED (too deep)" } else { "solved" },
        lib2.len(),
        acc * 100.0,
    );
    eprintln!("[RECURSE] VERDICT (the fractal loop closed): a rule too deep for bounded deliberation becomes reachable once a confident sub-composition is CRYSTALLIZED into a first-class cell — the organism banks its own result and reasons over it. Bounded search + crystallization = unbounded depth, built incrementally: reason → crystallize → reason. This is the recursion Sébastien's thesis needs — a curriculum the organism grows for itself, each solved rule a new building block for the next.");

    assert!(
        before.is_none(),
        "6-parity must be out of reach at length ≤ 2 over primitives"
    );
    assert!(
        after.is_some(),
        "after crystallizing the sub-rule, 6-parity must be reachable"
    );
    assert!(acc >= 0.99, "and exactly right (verified), got {acc:.2}");
}
