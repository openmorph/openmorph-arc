// SPDX-License-Identifier: MIT-0
//! SUGGESTIONS — the ARC-2 flow (Sébastien). The autonomous ALIVE organism self-infers its goals; on top of
//! that, the world can SUGGEST goals — and an ARC task is exactly that: the example grids are SUGGESTIONS of
//! the rule, the question is the grid to answer. The morph binds whatever rule its suggestions demonstrate,
//! then completes the question. Same organism, same `bind → prefer → complete`; the goal comes from OUTSIDE
//! (suggested) instead of from within (self-inferred).
//!
//! The sharp test is cross-rule (MorphLab#144 finding 3, where a gradient learner collapses to chance): a
//! FRESH morph must solve WHICHEVER rule its own few suggestions show — 2-D horizontal SYMMETRY or column
//! PERIODICITY — on HELD-OUT colours. `bind` binds "which cells covary", so it infers the rule TYPE from the
//! suggestions themselves; nothing is hard-coded per rule.
//!
//!   cargo test -p openmorph-core --test proofs substrate_suggest -- --nocapture --test-threads=1

use crate::common::Rng;
use openmorph_core::substrate::Field;

const H: usize = 4;
const W: usize = 4;
const N: usize = H * W;
fn cell(r: usize, c: usize) -> usize {
    r * W + c
}

// two rule families; each returns a consistent grid over the given colour alphabet
fn symmetric(colours: &[f32], rng: &mut Rng) -> Vec<f32> {
    let mut g = vec![0.0f32; N];
    for r in 0..H {
        for c in 0..W / 2 {
            let v = colours[rng.below(colours.len())];
            g[cell(r, c)] = v;
            g[cell(r, W - 1 - c)] = v; // horizontal mirror
        }
    }
    g
}
fn periodic(colours: &[f32], rng: &mut Rng) -> Vec<f32> {
    let mut g = vec![0.0f32; N];
    for r in 0..H {
        let seed: Vec<f32> = (0..2).map(|_| colours[rng.below(colours.len())]).collect();
        for c in 0..W {
            g[cell(r, c)] = seed[c % 2]; // column period 2
        }
    }
    g
}

/// The ARC-2 loop: SUGGEST `d` example grids (bind each), then on 300 held-out-colour questions clamp the known
/// cells (prefer) and complete the masked one. Returns accuracy — the morph solving whatever it was suggested.
fn suggest_then_answer(rule: fn(&[f32], &mut Rng) -> Vec<f32>, d: usize, seed: u64) -> f32 {
    let train: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let test: [f32; 4] = [5.0, 6.0, 7.0, 8.0]; // HELD-OUT colours
    let mut f = Field::new(N);
    let mut rng = Rng::new(seed);
    for _ in 0..d {
        let g = rule(&train, &mut rng);
        f.bind(&(0..N).map(|i| (i, g[i])).collect::<Vec<_>>()); // a SUGGESTION: one example of the rule
    }
    let (mut ok, mut tot) = (0, 0);
    for _ in 0..300 {
        let g = rule(&test, &mut rng);
        let m = rng.below(N);
        f.clear_values();
        for (i, &v) in g.iter().enumerate() {
            if i != m {
                f.prefer(i, v);
            }
        }
        if let Some(v) = f.complete(m) {
            if (v - g[m]).abs() < 1e-3 {
                ok += 1;
            }
        }
        tot += 1;
    }
    ok as f32 / tot as f32
}

#[test]
fn suggestions_are_the_arc_goal_channel() {
    let sym = (0..4)
        .map(|s| suggest_then_answer(symmetric, 5, 0x5 + s))
        .sum::<f32>()
        / 4.0;
    let per = (0..4)
        .map(|s| suggest_then_answer(periodic, 5, 0x5 + s))
        .sum::<f32>()
        / 4.0;

    eprintln!("[SUGGEST] a FRESH morph, 5 suggested examples of a rule, then answer held-out-colour questions (chance 25%):");
    eprintln!(
        "[SUGGEST]   suggestions show 2-D SYMMETRY  → morph answers {:.0}%",
        sym * 100.0
    );
    eprintln!(
        "[SUGGEST]   suggestions show PERIODICITY   → morph answers {:.0}%",
        per * 100.0
    );
    eprintln!("[SUGGEST] VERDICT (Sébastien's ARC-2 framing): a SUGGESTION is a goal from OUTSIDE. The autonomous ALIVE organism self-infers goals; the same `bind → prefer → complete` also takes SUGGESTED ones — an ARC task's example grids ARE suggestions of the rule, the question is the grid to answer. A single fresh morph solves WHICHEVER rule its own 5 examples demonstrate (symmetry {:.0}%, periodicity {:.0}%) on HELD-OUT colours — cross-rule (MorphLab#144 finding 3, where a gradient learner collapses to chance), because `bind` infers the rule TYPE from the suggestions (which cells covary), nothing hard-coded. Suggestions are the thin goal-channel layer on the autonomous default: examples → bind → prefer → complete → answer.", sym * 100.0, per * 100.0);

    assert!(
        sym > 0.9,
        "the morph must solve a SYMMETRY rule from its suggestions ({sym:.2})"
    );
    assert!(
        per > 0.9,
        "and a PERIODIC rule from different suggestions — cross-rule ({per:.2})"
    );
}

#[test]
fn a_deployed_crystal_morph_answers_few_shot_on_the_fast_path() {
    // TWO TIMESCALES: `crystallize()` freezes the SLOW grown weights (zero training compute — the deployed /
    // microcontroller path), but the FAST path — bind/prefer/complete — is inference-time and stays live. So a
    // shipped CRYSTAL snapshot still solves FRESH puzzles from their suggestions: consolidated knowledge frozen,
    // this puzzle's rule bound fast.
    let train: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let test: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    let mut f = Field::new(N);
    f.crystallize(); // DEPLOY: freeze the organism
    assert!(f.is_crystal());
    let mut rng = Rng::new(0xDE9107);
    for _ in 0..5 {
        let g = symmetric(&train, &mut rng);
        let obs: Vec<(usize, f32)> = (0..N).map(|i| (i, g[i])).collect();
        f.bind(&obs); // fast bind — works while frozen
        let _ = f.grow(&obs, 0, 0.0, 0.1, 20); // slow path: NO-OP in crystal
    }
    assert_eq!(
        f.n_units(),
        0,
        "crystal: the slow path grew nothing — zero training compute"
    );

    let (mut ok, mut tot) = (0, 0);
    for _ in 0..300 {
        let g = symmetric(&test, &mut rng);
        let m = rng.below(N);
        f.clear_values();
        for (i, &v) in g.iter().enumerate() {
            if i != m {
                f.prefer(i, v);
            }
        }
        if let Some(v) = f.complete(m) {
            if (v - g[m]).abs() < 1e-3 {
                ok += 1;
            }
        }
        tot += 1;
    }
    let acc = ok as f32 / tot as f32;
    eprintln!("[SUGGEST] DEPLOYED CRYSTAL morph (slow weights frozen): answered fresh held-out-colour puzzles {:.0}%, {} units grown.", acc * 100.0, f.n_units());
    eprintln!("[SUGGEST] VERDICT: two timescales in one organism. `crystallize()` freezes the SLOW grown weights (zero training compute — microcontroller / deploy), yet the FAST path (`bind`/`prefer`/`complete`) is inference-time and stays live, so a shipped snapshot still solves FRESH suggestion-puzzles ({:.0}%, zero units grown). Consolidated knowledge frozen; per-puzzle rule bound fast — exactly the two-timescale architecture MorphLab#144's 'bind, don't grind' calls for, now unified with Alive/Crystal.", acc * 100.0);

    assert!(
        acc > 0.9,
        "a frozen (deployed) morph must still answer few-shot via the fast path ({acc:.2})"
    );
}
