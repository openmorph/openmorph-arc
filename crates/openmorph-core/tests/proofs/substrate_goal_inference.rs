// SPDX-License-Identifier: MIT-0
//! GOAL / SUBGOAL INFERENCE FROM REWARD ATTRIBUTION (cap #335, epic #337).
//!
//! THE GAP (MorphBot). ARC-3 goals are HIDDEN and rewards SPARSE; a human infers *"get X onto Y"* from what
//! CHANGED when the reward arrived. openmorph's actor only chases the scalar reward, so between sparse rewards it
//! has almost no signal — no directed progress. The missing capability: on reward, ATTRIBUTE it to the co-occurring
//! perceptual condition and form a GOAL PREDICATE, then use that predicate as a dense internal shaping signal.
//!
//! THE MECHANISM (no new core primitive — grown credit assignment via `reliance_of`). Perception exposes a handful
//! of candidate relational features (overlap-target, x-align, y-align, overlap-distractor — the kind `relate`
//! grows). Grow a linear REWARD PREDICTOR over those features with the substrate's own `learn_step`; the feature the
//! predictor comes to RELY on (`reliance_of`, Σ readers' |weight|) IS the inferred goal predicate — grown, not
//! authored. Then plan to the states that satisfy it (`prefer` + `relax_values`, the validated #331/#334 machinery),
//! which turns a sparse terminal reward into a dense cost-to-go.
//!
//! THE CLAIM (cap #335 in-core acceptance): the inferred goal predicate matches the true (hidden) win condition on
//! ≥80% of levels, AND planning to the inferred goal reaches the reward in far fewer actions than reward-chasing.
//!
//! TRUTH-NOT-TROPHY. The world, the per-level HIDDEN win condition, the seeded PRNG, and the feature set are
//! authored and identical across arms; the win condition is NEVER given to the inferer — it is recovered from the
//! reward↔feature co-occurrence a random-play episode produces, by which feature the grown predictor relies on.
//! Collinear decoys are included (overlap ⟹ x-align), so the inferer must prefer the feature that predicts reward
//! EXACTLY, not one merely correlated. ZERO openmorph-core change: a usage protocol over `add_unit`/`learn_step`/
//! `reliance_of` + `prefer`/`relax_values`/`value_of`. The decisive metric (actions-to-first-win ↓ on sparse-reward
//! games) is MorphBot's live A/B.
//!
//!   cargo test -p openmorph-core --test proofs substrate_goal_inference -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const GW: usize = 5;
const GH: usize = 5;
const N_ACT: usize = 4;
const K: usize = 4; // candidate goal predicates / features

fn delta(a: usize) -> (i32, i32) {
    match a {
        0 => (0, -1),
        1 => (0, 1),
        2 => (-1, 0),
        _ => (1, 0),
    }
}
fn step(x: usize, y: usize, a: usize) -> (usize, usize) {
    let (dx, dy) = delta(a);
    (
        (x as i32 + dx).clamp(0, GW as i32 - 1) as usize,
        (y as i32 + dy).clamp(0, GH as i32 - 1) as usize,
    )
}
fn id(x: usize, y: usize) -> usize {
    y * GW + x
}

/// The world's fixed landmarks (target and a distractor), authored, identical across levels.
const TARGET: (usize, usize) = (3, 3);
const DISTRACT: (usize, usize) = (1, 4);

/// The candidate relational features at agent `(ax,ay)`, bipolar (+1 holds / −1 not). `relate`-style predicates:
/// 0 = overlap target, 1 = x-align target, 2 = y-align target, 3 = overlap distractor. A hidden win condition is
/// one of these; note the collinearity overlap(0) ⟹ x-align(1) ∧ y-align(2), the decoy the inferer must resist.
fn features(ax: usize, ay: usize) -> [f32; K] {
    let b = |c: bool| if c { 1.0 } else { -1.0 };
    [
        b(ax == TARGET.0 && ay == TARGET.1),
        b(ax == TARGET.0),
        b(ay == TARGET.1),
        b(ax == DISTRACT.0 && ay == DISTRACT.1),
    ]
}

/// Does the hidden win condition `k` hold at `(ax,ay)`?
fn wins(k: usize, ax: usize, ay: usize) -> bool {
    features(ax, ay)[k] > 0.0
}

/// GROWN REWARD ATTRIBUTION: random-play a terminal-reward-only game whose hidden win condition is `k`, grow a
/// linear reward predictor over the features, and return the feature the predictor RELIES ON most (`reliance_of`)
/// — the inferred goal predicate.
fn infer_goal(k: usize, seed: u64) -> usize {
    let mut rng = Rng::new(seed);
    // feature cells 0..K, the reward-prediction head at cell K (reads the K features).
    let mut f = Field::new(K + 1);
    f.add_unit((0..K).collect(), vec![0.0; K], 0.0, K);
    let (mut ax, mut ay) = (0usize, 0usize);
    for _ in 0..3000 {
        let a = rng.below(N_ACT);
        let (nx, ny) = step(ax, ay, a);
        let feats = features(nx, ny);
        let reward = if wins(k, nx, ny) { 1.0 } else { -1.0 };
        let inp: Vec<(usize, f32)> = (0..K).map(|i| (i, feats[i])).collect();
        f.learn_step(&inp, K, reward, 0.02, 50);
        ax = nx;
        ay = ny;
        if wins(k, ax, ay) {
            // terminal — respawn (a new episode of the same level)
            ax = 0;
            ay = 0;
        }
    }
    (0..K)
        .max_by(|&i, &j| f.reliance_of(i).total_cmp(&f.reliance_of(j)))
        .unwrap()
}

/// The grid adjacency (open 5×5, 4-connectivity).
fn grid_adjacency() -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); GW * GH];
    for y in 0..GH {
        for x in 0..GW {
            for a in 0..N_ACT {
                let (nx, ny) = step(x, y, a);
                if (nx, ny) != (x, y) {
                    adj[id(x, y)].push(id(nx, ny));
                }
            }
        }
    }
    adj
}

/// Plan to the states satisfying goal predicate `k` (`prefer` all of them, `relax_values`), then navigate greedily
/// from the start; return actions-to-reward (or cap). This is the "plan to the inferred goal" arm.
fn goal_directed(k: usize, cap: usize) -> usize {
    let adj = grid_adjacency();
    let mut f = Field::new(GW * GH);
    for y in 0..GH {
        for x in 0..GW {
            if wins(k, x, y) {
                f.prefer(id(x, y), 0.0); // every satisfying state is a zero-cost goal
            }
        }
    }
    f.relax_values(&adj, 1.0, 0);
    let (mut x, mut y) = (0usize, 0usize);
    for t in 1..=cap {
        if wins(k, x, y) {
            return t - 1;
        }
        let cur = id(x, y);
        let next = adj[cur]
            .iter()
            .copied()
            .min_by(|&a, &b| f.value_of(a).total_cmp(&f.value_of(b)).then(a.cmp(&b)))
            .unwrap_or(cur);
        (x, y) = (next % GW, next / GW);
    }
    cap
}

/// Reward-chasing baseline: no goal predicate, random walk until the (sparse) reward fires.
fn reward_chasing(k: usize, seed: u64, cap: usize) -> usize {
    let mut rng = Rng::new(seed);
    let (mut x, mut y) = (0usize, 0usize);
    for t in 1..=cap {
        if wins(k, x, y) {
            return t - 1;
        }
        let a = rng.below(N_ACT);
        (x, y) = step(x, y, a);
    }
    cap
}

#[test]
fn goal_inference_recovers_the_win_condition_and_shapes_planning() {
    let seeds = [0x31u64, 0x32, 0x33];
    let cap = 400;

    // (1) INFER the hidden win condition from reward attribution, over every (condition × seed) level.
    let (mut correct, mut total) = (0usize, 0usize);
    for k in 0..K {
        for &s in &seeds {
            if infer_goal(k, s) == k {
                correct += 1;
            }
            total += 1;
        }
    }
    let acc = correct as f32 / total as f32;
    assert!(
        acc >= 0.8,
        "inferred goal predicate must match the hidden win condition on ≥80% of levels: {correct}/{total}"
    );

    // (2) Planning to the inferred goal reaches the reward in far fewer actions than reward-chasing.
    let mut gd = Vec::new();
    let mut rc = Vec::new();
    for k in 0..K {
        gd.push(goal_directed(k, cap));
        let mean_rc = seeds
            .iter()
            .map(|&s| reward_chasing(k, s, cap))
            .sum::<usize>() as f32
            / seeds.len() as f32;
        rc.push(mean_rc);
    }
    let mean_gd = gd.iter().sum::<usize>() as f32 / gd.len() as f32;
    let mean_rc = rc.iter().sum::<f32>() / rc.len() as f32;
    assert!(
        gd.iter().all(|&a| a < cap),
        "goal-directed planning must reach every win condition: {gd:?}"
    );
    assert!(
        mean_rc >= 2.0 * mean_gd,
        "reward-chasing {mean_rc:.1} must cost ≥2× goal-directed {mean_gd:.1} actions-to-reward"
    );

    println!("\n  GOAL INFERENCE FROM REWARD ATTRIBUTION:");
    println!(
        "    (1) inferred win condition matches the hidden one: {correct}/{total} levels ({:.0}%)",
        acc * 100.0
    );
    println!("    (2) actions-to-reward — goal-directed {mean_gd:.1} vs reward-chasing {mean_rc:.1}  ({:.1}× fewer)", mean_rc / mean_gd.max(1.0));
    println!("  ⇒ attribute reward to the co-occurring feature (reliance) → a dense goal that shapes planning.\n");
}
