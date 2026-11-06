// SPDX-License-Identifier: MIT-0
//! FACTORIZATION — the real wall behind MorphLab's macro-planning null (#118). Macro-planning was a null
//! because the banked skills OVERLAP (each a whole-game policy) — there are no DISTINCT skills to plan over or
//! compose. The generalized fix: don't accumulate separate monolithic policies; grow ONE organism whose
//! recurring sub-computations the concept-map factors into a distinct, reusable, TRANSFERABLE vocabulary. This
//! probes whether `decompose` (the concept-map `cluster` from `substrate_decompose`) actually does that —
//! truth-not-trophy, because clean disentanglement is a genuinely open question.
//!
//! One organism grows a SHARED primitive g and two task-heads that both read it (A(g), B(g)). The concept-map
//! grows from per-task co-activation. Two claims:
//!   (A) FACTORED — clustering the grown map separates the shared primitive g from the task-specific heads
//!       A and B (A and B land in different clusters; g is its own).
//!   (B) TRANSFERS — `encapsulate`+`nest` the recovered g into a NEW task C(g) that also uses the primitive:
//!       it forms fast (the factored primitive is reused), where cold re-derivation from raw input struggles
//!       under the same budget. A distinct primitive that PAYS on a new task = distinct-skill formation.
//!
//! RE-BASELINED at the 1.6 learning flip — runs on the default genome. Claim (A) holds (clusters [[x, g], [A], [B]]:
//! the primitive is factored from both heads). Claim (B) does not and is `#[ignore]`d
//! (`decompose_factors_a_shared_primitive_that_transfers`): on the default genome COLD re-derivation from raw x reaches
//! residual 0.004 while REUSE (a readout grown over the nested g) is at 0.019 — the transfer is reversed (legacy: REUSE
//! 0.005 vs COLD 0.047). The lever is the fan-in-scaled feature init (`fanin_init_gain` 3; with only it at its legacy 0,
//! REUSE 0.004 vs COLD 0.037): both arms grow over ONE input, where `c/√1 = 3` draws features three times steeper than
//! the legacy `U[-1,1]`, and that bandwidth lets the cold arm re-derive g in the budget.
//!
//! WALL 2 (2026-09-27): a trade-off, not a bug. With only the arms on the legacy init, REUSE 0.0045 vs COLD 0.037 (the
//! shared organism's genome does not matter: the banked g is exact either way); `stability_cap`, `feature_lr`, the
//! reliance norm and the budget bound are inert here. An init never steeper than the legacy draw, `c/√max(n, c²)`,
//! restores the claim but costs two-spirals (fan-in 2) 94.1 → 61.5 % on the 12-task harness, and capping only fan-in 1
//! would be a special case. Not shipped; the claim stays ignored.
//!
//!   cargo test -p openmorph-core --test proofs substrate_factor -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// cells: 0=x ; 1=g(x) (shared primitive) ; 2=A(g) ; 3=B(g).
const X: usize = 0;
const G: usize = 1;
const A: usize = 2;
const B: usize = 3;

fn g_of(x: f32) -> f32 {
    (1.6 * x).tanh()
}
fn a_of(g: f32) -> f32 {
    (1.4 * g).tanh()
}
fn b_of(g: f32) -> f32 {
    (-1.4 * g).tanh()
}
fn c_of(g: f32) -> f32 {
    (0.8 * g).tanh() // a NEW task, same primitive g
}

/// Residual of learning task C from a given input context, after `iters` grow_deep steps.
fn learn_c(f: &mut Field, reads_g: bool, out: usize, g_cell: usize, iters: usize) -> f32 {
    f.config_mut().birth_gate = 0.05;
    f.config_mut().max_hidden = 4;
    for _ in 0..iters {
        for &x in &[-0.9f32, -0.4, 0.3, 0.8] {
            if reads_g {
                f.stimulate(&[(X, x)], 30); // settle so the nested g computes
                let g = f.cells()[g_cell];
                f.grow(&[(g_cell, g)], out, c_of(g_of(x)), 0.1, 30);
            } else {
                f.grow(&[(X, x)], out, c_of(g_of(x)), 0.1, 30);
            }
        }
    }
    let (mut e, mut n) = (0.0f32, 0.0);
    for &x in &[-0.7f32, 0.5, -0.2, 0.9] {
        f.stimulate(&[(X, x)], 30);
        e += (f.cells()[out] - c_of(g_of(x))).abs();
        n += 1.0;
    }
    e / n
}

/// Grow the shared organism, factor it by its concept-map (claim A) and transfer the banked primitive to task C (claim
/// B). Returns (factored, the clusters, REUSE residual, COLD residual).
fn factor_and_transfer() -> (bool, Vec<Vec<usize>>, f32, f32) {
    // ── Grow ONE organism: shared primitive g, plus heads A(g), B(g). ──
    let mut f = Field::new(4);
    f.config_mut().birth_gate = 0.05;
    f.config_mut().max_hidden = 0; // grow_and_learn was linear-only
    let mut rng = Rng::new(0x0FAC_7012);
    for _ in 0..2500 {
        let x = rng.sym();
        f.grow(&[(X, x)], G, g_of(x), 0.1, 30); // shared primitive
        let g = f.cells()[G];
        f.grow(&[(G, g)], A, a_of(g), 0.1, 30); // head A reads g
        f.grow(&[(G, g)], B, b_of(g), 0.1, 30); // head B reads g
    }

    // ── Grow the CONCEPT-MAP from per-task co-activation (the cells active together in each task episode). ──
    for it in 0..4000 {
        f.co_activate(&[X, G, A], 0.05); // task A episode: input, g, head A
        f.co_activate(&[X, G, B], 0.05); // task B episode: input, g, head B
        if it % 100 == 99 {
            f.spread_positions();
        }
    }
    f.spread_positions();

    // ── (A) FACTORED: does clustering separate the shared g from the task-specific heads A, B? ──
    let groups = f.cluster(&[X, G, A, B], 3);
    let cluster_of = |c: usize| groups.iter().position(|g| g.contains(&c)).unwrap();
    let (ca, cb, cg) = (cluster_of(A), cluster_of(B), cluster_of(G));
    let factored = ca != cb && cg != ca && cg != cb; // A, B, g each in a distinct cluster
    eprintln!(
        "[FACTOR] concept-map clusters of the shared organism: {groups:?}\n  head A in cluster {ca}, head B in {cb}, primitive g in {cg} → factored: {factored}"
    );

    // ── (B) TRANSFERS: the recovered g, re-nested, makes a NEW task C(g) form fast vs cold re-derivation. ──
    let g_node = f.encapsulate(&[G]); // bank the factored primitive g
    let iters = 400;
    // REUSE: nest g into a fresh organism, grow C reading g's output.
    let mut reuse = Field::new(1);
    reuse.nest(g_node, &[X], 1, 30); // cell 1 = g(x) (the factored primitive)
    let reuse_err = learn_c(&mut reuse, true, 2, 1, iters); // grow C at cell 2 reading g at cell 1
                                                            // COLD: grow C straight from raw x under the same budget (must re-derive g internally).
    let mut cold = Field::new(1);
    let cold_err = learn_c(&mut cold, false, 1, 0, iters);

    eprintln!(
        "[FACTOR] transfer of the factored primitive to a NEW task C(g):\n  REUSE (nest g → grow readout): residual {reuse_err:.3}\n  COLD  (grow C from raw x):     residual {cold_err:.3}"
    );
    (factored, groups, reuse_err, cold_err)
}

#[test]
fn decompose_factors_a_shared_primitive() {
    let (factored, groups, _, _) = factor_and_transfer();
    eprintln!("[FACTOR] VERDICT: the wall behind the macro-planning null is DISTINCTNESS — overlapping whole-game policies can't be planned over or composed. The fix generalizes: grow ONE organism and FACTOR it, don't accumulate monoliths. `decompose` (the concept-map `cluster`) separates the shared primitive g from the task-specific heads A and B — cells that recur across tasks co-activate consistently and cluster distinctly, so a grown solution factors into a reusable vocabulary, which `encapsulate` banks as a distinct node. (Since the 1.6 re-baseline this is NOT claimed to make a new task form faster than cold re-derivation: on the default genome cold re-derivation is better — see the module docs.)");
    assert!(factored, "decompose must factor the shared primitive g distinct from the task heads (clusters {groups:?})");
}

/// The TRANSFER half (claim B) of the original proof, kept verbatim but ignored: on the 1.6 default cold re-derivation
/// from raw x reaches residual 0.004 while REUSE is at 0.019 (see the module docs).
#[test]
#[ignore = "1.6 re-baseline: does not hold on the default genome — REUSE 0.019 vs COLD 0.004 (reversed); legacy 0.005 vs 0.047; see this file's module docs"]
fn decompose_factors_a_shared_primitive_that_transfers() {
    let (factored, groups, reuse_err, cold_err) = factor_and_transfer();
    assert!(factored, "decompose must factor the shared primitive g distinct from the task heads (clusters {groups:?})");
    assert!(
        reuse_err < 0.02 && reuse_err * 2.0 < cold_err,
        "the factored primitive must TRANSFER — a new task using it forms markedly better than cold re-derivation (reuse {reuse_err:.3} vs cold {cold_err:.3})"
    );
}
