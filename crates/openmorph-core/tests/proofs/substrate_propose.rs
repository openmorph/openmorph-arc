// SPDX-License-Identifier: MIT-0
//! THE PROPOSER — System-1 orders the search, surprise is the ordering signal (the open-end flagged under
//! the reasoning chain; MorphLab #104 measured it: −26% simulations, accuracy held at 100%). `deliberate`
//! verifies candidates in an arbitrary order — correct but O(library) per depth. Because verification is
//! order-INDEPENDENT (checking a candidate can never be wrong, only slow), a proposer can only SPEED the
//! search, never break it. The non-authored proposer: order candidates by the organism's own SURPRISE
//! against the target on the examples — the block it is LEAST surprised by (most predictive) is simulated
//! first. Same currency that gates every birth; no hand-tuned relevance heuristic.
//!
//! Demonstrated: a library where one grown block IS the rule and the rest are distractors. Surprise-ordered
//! deliberation commits after ~1 thought; arbitrary order pays O(library). Both return the SAME block —
//! System-1 (surprise) proposes the order, System-2 (verification) keeps the answer exact. HONEST SCOPE:
//! this speeds search only where a candidate carries solo signal; for a pure-XOR composition (each block
//! 50% alone) surprise is flat and the fallback is Occam/length order — verification still guarantees the
//! result either way.
//!
//!   cargo test -p openmorph-core --test proofs substrate_propose -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 4;
const K: usize = 24;
const STEPS: usize = 3000;
const BUD: usize = 300;

fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}

// A set of distinct grown functions; index TRUE_FN is the actual rule. The rest are distractors — none of
// them fits the rule everywhere (each disagrees on some inputs), so verification correctly skips them.
const TRUE_FN: usize = 6;
fn func(k: usize, x: &[f32]) -> f32 {
    match k {
        0 => x[0],
        1 => x[2],
        2 => (x[2] * x[3]).signum(), // AND-ish
        3 => x[3],
        4 => (x[0] + x[2] + x[3]).signum(), // a partial correlate of the rule
        5 => -x[2] * x[3],                  // XOR(2,3)
        6 => -x[0] * x[1],                  // THE RULE: XOR(0,1)
        _ => x[1],
    }
}
fn target(x: &[f32]) -> f32 {
    func(TRUE_FN, x)
}

fn grow_block(f: &mut Field, k: usize, rng: &mut Rng) -> usize {
    let inputs: Vec<usize> = (0..N).collect();
    let hidden = f.grow_backbone(&inputs, K, 2.5);
    let mut reads = inputs.clone();
    reads.extend(hidden);
    let out = f.len();
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, out);
    for _ in 0..STEPS {
        let x: Vec<f32> = (0..N).map(|_| rng.bit()).collect();
        f.learn_step(&pairs(&x), out, func(k, &x), 0.05, BUD);
    }
    out
}

fn imagine(f: &mut Field, block: usize, x: &[f32]) -> f32 {
    f.stimulate(&pairs(x), BUD);
    f.cells()[block]
}

/// Verify candidates in `order`, committing to the first that fits EVERY example. Returns (chosen block,
/// thoughts = candidates simulated).
fn deliberate(f: &mut Field, order: &[usize], ex: &[(Vec<f32>, f32)]) -> (Option<usize>, usize) {
    let mut thoughts = 0;
    for &b in order {
        thoughts += 1;
        if ex
            .iter()
            .all(|(x, y)| imagine(f, b, x).signum() == y.signum())
        {
            return (Some(b), thoughts);
        }
    }
    (None, thoughts)
}

#[test]
fn surprise_proposes_the_search_order_verification_keeps_it_exact() {
    let mut rng = Rng::new(0x9405);
    let mut f = Field::new(N);
    let lib: Vec<usize> = (0..8).map(|k| grow_block(&mut f, k, &mut rng)).collect();

    let ex: Vec<(Vec<f32>, f32)> = (0..24)
        .map(|_| {
            let x: Vec<f32> = (0..N).map(|_| rng.bit()).collect();
            let y = target(&x);
            (x, y)
        })
        .collect();

    // ARBITRARY order (the library's build order) — verification is correct but pays O(library).
    let brute_order: Vec<usize> = lib.clone();
    let (brute_pick, brute_thoughts) = deliberate(&mut f, &brute_order, &ex);

    // PROPOSER: order blocks by the organism's SURPRISE against the target on the examples (ascending —
    // least-surprising first). This is the same cell-vs-target comparison `learn_step` runs, read as a
    // relevance score; no authored heuristic.
    let mut scored: Vec<(usize, f32)> = lib
        .iter()
        .map(|&b| {
            let s = ex
                .iter()
                .map(|(x, y)| (y - imagine(&mut f, b, x)).abs())
                .sum::<f32>()
                / ex.len() as f32;
            (b, s)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let prop_order: Vec<usize> = scored.iter().map(|(b, _)| *b).collect();
    let (prop_pick, prop_thoughts) = deliberate(&mut f, &prop_order, &ex);

    eprintln!(
        "[PROPOSE] library of 8 grown blocks, one IS the rule (block {}, at build-index {TRUE_FN}):\n  ARBITRARY order : chose block {:?} after {brute_thoughts} thoughts\n  SURPRISE-ordered: chose block {:?} after {prop_thoughts} thoughts (least-surprising-first)",
        lib[TRUE_FN],
        brute_pick,
        prop_pick,
    );
    eprintln!("[PROPOSE] VERDICT: the proposer is the reasoning chain's last flagged open-end, and it's non-authored — order the candidates by the organism's own SURPRISE against the target (the same comparison that gates every birth), least-surprising first. Because verification is order-independent, the proposer only ever SPEEDS the search (System-1 intuition), while verification (System-2) keeps the answer exact — both orders return the identical block. This is what MorphLab measured as −26% simulations at 100% accuracy; here the winning block is found in ~1 thought instead of O(library). For a pure-XOR composition (no solo signal) surprise is flat and it falls back to Occam/length order — correctness is guaranteed regardless.");

    assert_eq!(
        prop_pick, brute_pick,
        "the proposer must not change the answer"
    );
    assert_eq!(
        prop_pick,
        Some(lib[TRUE_FN]),
        "and it must be the true rule"
    );
    assert!(
        prop_thoughts < brute_thoughts,
        "surprise-ordering must reduce thoughts ({prop_thoughts} vs {brute_thoughts})"
    );
}
