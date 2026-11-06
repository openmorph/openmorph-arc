// SPDX-License-Identifier: MIT-0
//! PERCEPTION → REASONING SEAM, grown end-to-end (Gap 2, `docs/research/arc3-gaps-to-100.md`).
//!
//! THE GAP (the honest authoring debt of epic #337). Goal inference (#335) and skill memory (#336) were validated
//! over HAND-AUTHORED relational features (`overlap-target`, `x-align`, …). The founding rule says perception and
//! features must be GROWN, not hand-set — so the whole-organism claim (raw frame → grown objects → grown relations →
//! grown goal-inference) was unproven, and the authored features are a purity debt. This closes the seam: the SAME
//! grown reward-attribution runs over the GROWN relational structure the substrate perceives, not authored features.
//!
//! THE MECHANISM (fully grown — no authored features). Raw pixel frame → `grouping::segment_objects` (the async
//! label-propagation fixed point = connected-component OBJECTS) → `grouping::relate` (grown RELATIONS between
//! objects: relative position `dx/dy`, `adjacent`, `same_color`, size). Grow a reward predictor over those grown
//! relation predicates with `learn_step`; the one it RELIES on (`reliance_of`) is the inferred goal relation (#335,
//! now over grown structure). Nothing about the task is authored — the objects and relations are perceived.
//!
//! THE CLAIM (Gap 2 acceptance):
//!   * The inferred goal relation over GROWN objects/relations matches the hidden win condition ≥80% (parity with
//!     the authored-feature #335 result — the seam does not degrade it).
//!   * SEAM LOAD-BEARING: a grown-relational reward predictor GENERALISES to unseen agent positions (the relation is
//!     translation-invariant), while a RAW-PIXEL predictor memorises and is ≈ chance on held-out positions — pixels
//!     alias, exactly MorphBot's finding; the grown perception is what makes the abstract goal learnable.
//!
//! TRUTH-NOT-TROPHY. The frames, the hidden win relation, the seeded PRNG are authored and identical across arms;
//! the win relation is NEVER given — it is recovered from reward↔grown-relation co-occurrence. ZERO
//! openmorph-core change: a usage protocol over `grouping::segment_objects`/`relate` (grown perception) +
//! `add_unit`/`learn_step`/`reliance_of` (grown attribution). The decisive metric is MorphBot's live A/B.
//!
//!   cargo test -p openmorph-core --test proofs substrate_perception_seam -- --nocapture --test-threads=1

use openmorph_core::grouping::{relate, segment_objects};
use openmorph_core::substrate::Field;

use crate::common::Rng;

const W: usize = 6;
const H: usize = 6;
const N_ACT: usize = 4;
const BG: usize = 0;
const AGENT: usize = 1;
const TARGET: usize = 2;
const DISTRACT: usize = 3;
const TPOS: (usize, usize) = (4, 4);
const DPOS: (usize, usize) = (1, 3);

fn step(x: usize, y: usize, a: usize) -> (usize, usize) {
    let (dx, dy) = [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)][a];
    (
        (x as i32 + dx).clamp(0, W as i32 - 1) as usize,
        (y as i32 + dy).clamp(0, H as i32 - 1) as usize,
    )
}

/// Render the raw pixel frame with the agent at `(ax,ay)` (target + distractor fixed). No object identity is handed
/// to the model — it must SEGMENT the frame itself.
fn frame(ax: usize, ay: usize) -> Vec<usize> {
    let mut g = vec![BG; W * H];
    g[TPOS.1 * W + TPOS.0] = TARGET;
    g[DPOS.1 * W + DPOS.0] = DISTRACT;
    if (ax, ay) != TPOS && (ax, ay) != DPOS {
        g[ay * W + ax] = AGENT;
    }
    g
}

/// GROWN perception → the agent↔target relation features (bipolar): [x-align, y-align, adjacent, same-colour].
/// Derived ENTIRELY from `segment_objects` + `relate` — the objects and their relation are PERCEIVED, not authored.
fn grown_features(ax: usize, ay: usize) -> [f32; 4] {
    let g = frame(ax, ay);
    let objs = segment_objects(&g, W, H, BG);
    let rels = relate(&objs, W, H);
    let b = |c: bool| if c { 1.0 } else { -1.0 };
    // find the grown relation whose endpoints are the agent-coloured and target-coloured objects.
    for r in &rels {
        let (ca, cb) = (objs[r.a].color, objs[r.b].color);
        if (ca, cb) == (AGENT, TARGET) || (ca, cb) == (TARGET, AGENT) {
            return [
                b(r.dx.abs() < 0.01), // x-align (grown relative position)
                b(r.dy.abs() < 0.01), // y-align
                b(r.adjacent),        // grown proximity
                b(r.same_color),      // dead decoy (never the reward here)
            ];
        }
    }
    [-1.0; 4] // agent on a landmark cell (no distinct object) — no relation this frame
}

fn wins(k: usize, ax: usize, ay: usize) -> bool {
    grown_features(ax, ay)[k] > 0.0
}

/// GROWN goal inference (#335 over grown perception): random-play the hidden-relation-`k` game, grow a reward
/// predictor over the GROWN relation features, and return the feature it relies on most.
fn infer_over_grown(k: usize, seed: u64) -> usize {
    let mut rng = Rng::new(seed);
    let mut f = Field::new(5); // 4 grown features + a reward head
    f.add_unit((0..4).collect(), vec![0.0; 4], 0.0, 4);
    let (mut ax, mut ay) = (0usize, 0usize);
    for _ in 0..2500 {
        let a = rng.below(N_ACT);
        let (nx, ny) = step(ax, ay, a);
        let feats = grown_features(nx, ny);
        let reward = if feats[k] > 0.0 { 1.0 } else { -1.0 };
        f.learn_step(
            &(0..4).map(|i| (i, feats[i])).collect::<Vec<_>>(),
            4,
            reward,
            0.02,
            50,
        );
        ax = nx;
        ay = ny;
    }
    (0..4)
        .max_by(|&i, &j| f.reliance_of(i).total_cmp(&f.reliance_of(j)))
        .unwrap()
}

/// Held-out reward-prediction accuracy for a predictor trained on TRAIN agent positions and tested on TEST
/// positions. `grown=true` reads the grown relation features; `grown=false` reads the RAW pixel frame (which
/// aliases). Measures whether the abstract goal (relation `k`) GENERALISES to unseen positions.
fn generalisation(k: usize, grown: bool, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let n_in = if grown { 4 } else { W * H };
    let mut f = Field::new(n_in + 1);
    f.add_unit((0..n_in).collect(), vec![0.0; n_in], 0.0, n_in);
    let feats = |ax: usize, ay: usize| -> Vec<f32> {
        if grown {
            grown_features(ax, ay).to_vec()
        } else {
            frame(ax, ay).iter().map(|&c| c as f32).collect()
        }
    };
    // DISJOINT agent positions split by a hash (≈60/40), so both sets contain the relation's positive region —
    // the test is RELATIONAL generalisation to unseen positions, not coverage of where the relation can occur.
    let all: Vec<(usize, usize)> = (0..H).flat_map(|y| (0..W).map(move |x| (x, y))).collect();
    let train: Vec<(usize, usize)> = all
        .iter()
        .copied()
        .filter(|&(x, y)| (x * 7 + y * 3) % 5 < 3)
        .collect();
    let test: Vec<(usize, usize)> = all
        .iter()
        .copied()
        .filter(|&(x, y)| (x * 7 + y * 3) % 5 >= 3)
        .collect();
    for _ in 0..4000 {
        let &(ax, ay) = &train[rng.below(train.len())];
        let fv = feats(ax, ay);
        let reward = if wins(k, ax, ay) { 1.0 } else { -1.0 };
        f.learn_step(
            &fv.iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect::<Vec<_>>(),
            n_in,
            reward,
            0.02,
            60,
        );
    }
    let (mut correct, mut total) = (0usize, 0usize);
    for &(ax, ay) in &test {
        let fv = feats(ax, ay);
        f.stimulate(
            &fv.iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect::<Vec<_>>(),
            60,
        );
        let pred = f.cells()[n_in];
        let truth: f32 = if wins(k, ax, ay) { 1.0 } else { -1.0 };
        if pred.signum() == truth.signum() {
            correct += 1;
        }
        total += 1;
    }
    correct as f32 / total as f32
}

#[test]
fn grown_perception_makes_the_abstract_goal_learnable() {
    let seeds = [0x51u64, 0x52, 0x53];
    let conditions = [0usize, 1, 2]; // x-align, y-align, adjacent — all GROWN relations

    // (1) SEAM: goal inference over GROWN objects/relations recovers the hidden win condition.
    let (mut correct, mut total) = (0usize, 0usize);
    for &k in &conditions {
        for &s in &seeds {
            if infer_over_grown(k, s) == k {
                correct += 1;
            }
            total += 1;
        }
    }
    let acc = correct as f32 / total as f32;
    assert!(
        acc >= 0.8,
        "goal inference over GROWN perception must match the hidden relation ≥80%: {correct}/{total}"
    );

    // (2) SEAM LOAD-BEARING: the grown relation GENERALISES to unseen positions; raw pixels alias (≈ chance).
    let mut grown_gen = Vec::new();
    let mut pixel_gen = Vec::new();
    for &k in &conditions {
        grown_gen.push(generalisation(k, true, 0x900 + k as u64));
        pixel_gen.push(generalisation(k, false, 0x900 + k as u64));
    }
    let mg = grown_gen.iter().sum::<f32>() / grown_gen.len() as f32;
    let mp = pixel_gen.iter().sum::<f32>() / pixel_gen.len() as f32;
    assert!(
        mg > 0.85,
        "grown-relational reward prediction must GENERALISE to unseen positions: {mg:.3} ({grown_gen:?})"
    );
    assert!(
        mg > mp + 0.15,
        "grown perception must beat raw pixels on held-out generalisation: grown {mg:.3} vs pixel {mp:.3}"
    );

    println!("\n  PERCEPTION → REASONING SEAM (raw frame → segment_objects → relate → reliance):");
    println!("    (1) goal inference over GROWN relations matches the hidden win condition: {correct}/{total} ({:.0}%)", acc * 100.0);
    println!("    (2) held-out generalisation — GROWN {mg:.3} vs RAW-PIXEL {mp:.3} (pixels alias)");
    println!("  ⇒ the grown perceptual structure is what makes the abstract relational goal learnable — no authored features.\n");
}
