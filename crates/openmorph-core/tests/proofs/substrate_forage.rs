// SPDX-License-Identifier: MIT-0
//! ANT FORAGING — autonomous goal inference by reward PATTERN recognition, with a negative pheromone. The
//! capstone of the navigation arc: `substrate_navigate` reaches a goal, `substrate_autonomy` would set one from
//! novelty — but an ant does something richer. It finds food, then infers WHERE MORE FOOD IS by resemblance:
//! good sources look alike, so they land in the same place in its mind (the concept map clusters them by
//! feature). Reward becomes a smooth, learnable function over that map — find one, predict the rest.
//!
//! And the pheromone "I've been here, nothing good" is the SAME mechanism's negative half. The reward readout
//! is a `tanh` spanning [-1, +1]; we have only ever used the positive side. A NEGATIVE target is just as
//! natural — "this look is a dead end" — and it is exactly what you need when decoys LOOK LIKE FOOD but aren't:
//! with positive examples only you can't separate them and you keep chasing the false leads; the negative
//! pheromone carves the dead-end look out of the food look.
//!
//! Ablation (steps to collect all real food; food and decoys are confusable — close in feature space):
//!   RANDOM goals ...... no pattern at all — foraging by luck.
//!   POSITIVE only ..... learns what food looks like, but can't tell a look-alike decoy from real food, so it
//!                       keeps navigating to decoys — the wall of positive-only reward.
//!   POS + NEG (ant) ... one visit to a decoy marks that look negative; the model then separates food from
//!                       decoy and beelines to real food. Negative reward, added trivially, is the difference.
//!
//! Everything is grown / a usage protocol: a spatial map for navigation (`metric_step`), a reward model over
//! features (`grow` — the morph's own recognition), goal = argmax predicted reward. Zero core code.
//!
//!   cargo test -p openmorph-core --test proofs substrate_forage -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const SIDE: usize = 11;
const F: usize = 5; // feature dimension — what the morph "sees" at a cell
const FOOD_N: usize = 5;
const DECOY_N: usize = 8;

fn neighbors(cell: usize) -> Vec<usize> {
    let (r, c) = (cell / SIDE, cell % SIDE);
    let mut v = Vec::new();
    if r + 1 < SIDE {
        v.push((r + 1) * SIDE + c);
    }
    if r > 0 {
        v.push((r - 1) * SIDE + c);
    }
    if c + 1 < SIDE {
        v.push(r * SIDE + c + 1);
    }
    if c > 0 {
        v.push(r * SIDE + c - 1);
    }
    v
}
fn euclid(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// The morph learns the space by experience → a near-isometric spatial map for navigation.
fn grow_spatial(seed: u64) -> Vec<Vec<f32>> {
    let n = SIDE * SIDE;
    let mut org = Field::new(n);
    for _ in 0..4 {
        org.grow_dimension();
    }
    let mut rng = Rng::new(seed);
    let mut cur = 0usize;
    let steps = 120_000;
    for it in 0..steps {
        let nb = neighbors(cur);
        let nxt = nb[rng.below(nb.len())];
        let rate = 0.2 * (1.0 - it as f32 / steps as f32).max(0.02);
        org.metric_step(&[cur, nxt], rate, 14);
        cur = nxt;
    }
    (0..n).map(|i| org.position(i)).collect()
}

/// Greedy navigation on the spatial map — the walked path from `a` to `goal`.
fn navigate(a: usize, goal: usize, pos: &[Vec<f32>], cap: usize) -> Vec<usize> {
    let mut cur = a;
    let mut path = vec![a];
    let mut seen = vec![0usize; SIDE * SIDE];
    for _ in 0..cap {
        if cur == goal {
            break;
        }
        let nb = neighbors(cur);
        let next = *nb
            .iter()
            .min_by(|&&x, &&y| {
                let fx = euclid(&pos[x], &pos[goal]) + seen[x] as f32 * 0.05;
                let fy = euclid(&pos[y], &pos[goal]) + seen[y] as f32 * 0.05;
                fx.total_cmp(&fy)
            })
            .unwrap();
        seen[cur] += 1;
        cur = next;
        path.push(cur);
    }
    path
}

/// Build a world: features per cell, plus which cells are food vs (look-alike) decoy. Food and decoy share
/// most feature dims and differ in one — confusable, the whole point.
fn make_world(seed: u64) -> (Vec<Vec<f32>>, Vec<bool>, Vec<bool>) {
    let n = SIDE * SIDE;
    let mut rng = Rng::new(seed);
    let rand_feat = |rng: &mut Rng| -> Vec<f32> { (0..F).map(|_| rng.f() * 2.0 - 1.0).collect() };
    let food_proto = vec![0.9, 0.8, -0.2, 0.3, -0.7];
    let mut decoy_proto = food_proto.clone();
    decoy_proto[1] = -0.8; // the ONE dim that separates a real source from a look-alike
    let jitter = |p: &[f32], rng: &mut Rng| -> Vec<f32> {
        p.iter()
            .map(|&x| (x + (rng.f() - 0.5) * 0.2).clamp(-1.0, 1.0))
            .collect()
    };

    let mut is_food = vec![false; n];
    let mut is_decoy = vec![false; n];
    let mut feat: Vec<Vec<f32>> = (0..n).map(|_| rand_feat(&mut rng)).collect();
    // scatter food and decoys on distinct cells (avoid the start cell 0)
    let mut placed = std::collections::HashSet::new();
    placed.insert(0usize);
    let place = |k: usize,
                 food: bool,
                 feat: &mut [Vec<f32>],
                 is_food: &mut [bool],
                 is_decoy: &mut [bool],
                 rng: &mut Rng,
                 placed: &mut std::collections::HashSet<usize>| {
        let mut done = 0;
        while done < k {
            let c = rng.below(n);
            if placed.insert(c) {
                feat[c] = if food {
                    jitter(&food_proto, rng)
                } else {
                    jitter(&decoy_proto, rng)
                };
                if food {
                    is_food[c] = true;
                } else {
                    is_decoy[c] = true;
                }
                done += 1;
            }
        }
    };
    place(
        FOOD_N,
        true,
        &mut feat,
        &mut is_food,
        &mut is_decoy,
        &mut rng,
        &mut placed,
    );
    place(
        DECOY_N,
        false,
        &mut feat,
        &mut is_food,
        &mut is_decoy,
        &mut rng,
        &mut placed,
    );
    (feat, is_food, is_decoy)
}

/// One forager. mode: "random" (ignore the reward model), "pos" (learn positives only), "full" (pos + negative
/// pheromone). Returns steps to collect all food (cap = failed).
fn forage(
    pos: &[Vec<f32>],
    feat: &[Vec<f32>],
    is_food: &[bool],
    mode: &str,
    seed: u64,
    cap: usize,
) -> usize {
    let n = SIDE * SIDE;
    let mut rm = Field::new(F);
    let out = rm.len();
    rm.ensure_cells(out + 1);
    rm.rng_seed(seed ^ 0xF00D);
    rm.config_mut().max_hidden = 16;
    rm.config_mut().feature_lr = 0.3;
    let inp = |c: usize| -> Vec<(usize, f32)> { (0..F).map(|i| (i, feat[c][i])).collect() };
    let predict = |rm: &mut Field, c: usize| -> f32 {
        rm.stimulate(&inp(c), 40);
        rm.cells()[out]
    };

    let mut visited = vec![false; n];
    visited[0] = true;
    let mut collected = 0usize;
    let mut cur = 0usize;
    let mut steps = 0usize;
    let mut rng = Rng::new(seed);
    while collected < FOOD_N && steps < cap {
        let unvisited: Vec<usize> = (0..n).filter(|&c| !visited[c]).collect();
        if unvisited.is_empty() {
            break;
        }
        let goal = if mode == "random" {
            unvisited[rng.below(unvisited.len())]
        } else {
            // goal = the unvisited cell that most LOOKS like reward; tiny spatial-novelty tiebreak so an
            // untrained model still explores rather than stalling on ties.
            *unvisited
                .iter()
                .max_by(|&&x, &&y| {
                    let novx = neighbors(x).len() as f32 * 0.001;
                    let novy = neighbors(y).len() as f32 * 0.001;
                    (predict(&mut rm, x) + novx).total_cmp(&(predict(&mut rm, y) + novy))
                })
                .unwrap()
        };
        let path = navigate(cur, goal, pos, 4 * n);
        for &c in &path {
            if !visited[c] {
                visited[c] = true;
                if is_food[c] {
                    collected += 1;
                    for _ in 0..5 {
                        rm.grow(&inp(c), out, 0.9, 0.1, 40); // attractant: this look = food
                    }
                }
            }
        }
        // the negative pheromone: "I deliberately went THERE and it was a dead end" — only the disappointing
        // GOAL is marked, not every incidental step (marking everything would just flood the model negative).
        if mode == "full" && !is_food[goal] {
            for _ in 0..5 {
                rm.grow(&inp(goal), out, -0.9, 0.1, 40);
            }
        }
        steps += path.len().saturating_sub(1);
        cur = *path.last().unwrap();
    }
    if collected >= FOOD_N {
        steps
    } else {
        cap
    }
}

#[test]
fn the_morph_forages_by_pattern_with_a_negative_pheromone() {
    let cap = 40 * SIDE * SIDE;
    let (mut rnd_s, mut pos_s, mut full_s) = (0.0f32, 0.0f32, 0.0f32);
    let seeds = 5;
    for s in 0..seeds {
        let spatial = grow_spatial(0x5EED + s);
        let (feat, is_food, _is_decoy) = make_world(0xA1 + s);
        rnd_s += forage(&spatial, &feat, &is_food, "random", 0x11 + s, cap) as f32;
        pos_s += forage(&spatial, &feat, &is_food, "pos", 0x11 + s, cap) as f32;
        full_s += forage(&spatial, &feat, &is_food, "full", 0x11 + s, cap) as f32;
    }
    let (rnd, pos, full) = (
        rnd_s / seeds as f32,
        pos_s / seeds as f32,
        full_s / seeds as f32,
    );

    eprintln!("[FORAGE] world: {SIDE}×{SIDE}, {FOOD_N} food + {DECOY_N} LOOK-ALIKE decoys (confusable features); steps to collect all food.");
    eprintln!(
        "[FORAGE]   RANDOM goals ......... {rnd:>4.0} steps   (no pattern — foraging by luck)"
    );
    eprintln!("[FORAGE]   POSITIVE reward only . {pos:>4.0} steps   (learns the food look, but can't separate a look-alike decoy — chases false leads)");
    eprintln!("[FORAGE]   POS + NEG pheromone .. {full:>4.0} steps   (one decoy visit marks that look negative → separates food from decoy → beelines)");
    eprintln!("[FORAGE] VERDICT: autonomy = goal inference by reward PATTERN over the grown map (the ant). Good sources look alike, so they sit in the same place in the morph's mind; a reward readout over features makes reward a smooth learnable function — find one, PREDICT the rest, navigate there. The negative pheromone ('I've been here, nothing good') is the SAME readout's other half: the tanh output spans [-1,+1], so a NEGATIVE target is trivial — and it is exactly what separates a look-alike DECOY from real food (positive-only can't: {pos:.0} steps vs pos+neg {full:.0}). So YES, negative reward is easy and it matters. Curiosity finds the first source; recognition (attractant) and pheromone (repellent), both readouts over the grown geometry, do the rest. Goal inference + navigation + intrinsic reward/punishment = one autonomous loop, zero core code.");

    assert!(
        full < pos * 0.85,
        "the negative pheromone must beat positive-only (full {full:.0} vs pos {pos:.0})"
    );
    assert!(
        full < rnd * 0.8,
        "and pattern foraging must beat random goals (full {full:.0} vs random {rnd:.0})"
    );
    assert!(
        (full as usize) < cap,
        "the ant forager must collect all food within budget"
    );
}

// ---- THE KEY: reward is GOAL-CONDITIONED. A place good for one goal is bad for another. ----

const GF: usize = 5; // feature dim for the typed world

/// A world of two resource TYPES (A, B), each a feature cluster. Returns (features, type: false=A/true=B).
fn typed_world(seed: u64, n: usize) -> (Vec<Vec<f32>>, Vec<bool>) {
    let mut rng = Rng::new(seed);
    let proto_a = vec![0.9, -0.7, 0.4, -0.5, 0.8];
    let proto_b = vec![-0.8, 0.6, -0.3, 0.7, -0.9];
    let mut feat = Vec::with_capacity(n);
    let mut ty = Vec::with_capacity(n);
    for _ in 0..n {
        let b = rng.f() < 0.5;
        let p = if b { &proto_b } else { &proto_a };
        feat.push(
            p.iter()
                .map(|&x| (x + (rng.f() - 0.5) * 0.4).clamp(-1.0, 1.0))
                .collect::<Vec<f32>>(),
        );
        ty.push(b);
    }
    (feat, ty)
}
/// Reward for visiting a cell of type `ty` while pursuing `goal`: +1 if it matches the goal, −1 if it's the
/// OTHER resource (good for the other goal, bad for this one), the whole point.
fn goal_reward(ty: bool, goal: bool) -> f32 {
    if ty == goal {
        0.9
    } else {
        -0.9
    }
}

/// Train a reward model and return sign-accuracy on HELD-OUT cells across BOTH goals. `conditioned` = the model
/// sees the goal (features ⊕ goal-code); otherwise it is goal-blind (features only).
fn goal_value_accuracy(conditioned: bool, seed: u64) -> f32 {
    let n = 220;
    let (feat, ty) = typed_world(seed, n);
    let din = if conditioned { GF + 1 } else { GF };
    let mut rm = Field::new(din);
    let out = rm.len();
    rm.ensure_cells(out + 1);
    rm.rng_seed(seed ^ 0xC0DE);
    rm.config_mut().max_hidden = 24;
    rm.config_mut().feature_lr = 0.3;
    let inp = |c: usize, goal: bool| -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = (0..GF).map(|i| (i, feat[c][i])).collect();
        if conditioned {
            v.push((GF, if goal { 1.0 } else { -1.0 }));
        }
        v
    };
    let split = n / 2; // first half train, second half held out
    for _ in 0..400 {
        for (c, &t) in ty.iter().enumerate().take(split) {
            for &goal in &[false, true] {
                rm.grow(&inp(c, goal), out, goal_reward(t, goal), 0.1, 40);
            }
        }
    }
    let (mut ok, mut tot) = (0, 0);
    for (c, &t) in ty.iter().enumerate().skip(split) {
        for &goal in &[false, true] {
            rm.stimulate(&inp(c, goal), 40);
            let pred = rm.cells()[out];
            if pred.signum() == goal_reward(t, goal).signum() {
                ok += 1;
            }
            tot += 1;
        }
    }
    ok as f32 / tot as f32
}

#[test]
fn reward_is_goal_conditioned_the_same_place_good_for_one_goal_bad_for_another() {
    let (mut cond, mut blind) = (0.0f32, 0.0f32);
    let seeds = 4;
    for s in 0..seeds {
        cond += goal_value_accuracy(true, 0x77 + s);
        blind += goal_value_accuracy(false, 0x77 + s);
    }
    let (cond, blind) = (cond / seeds as f32, blind / seeds as f32);

    eprintln!("[GOAL] held-out sign-accuracy of predicted reward, over BOTH goals (a type-A place is +1 for goal A, −1 for goal B):");
    eprintln!("[GOAL]   GOAL-BLIND value (features only) ...... {:.0}%   (same features get +1 AND −1 depending on the goal → contradictory → can't resolve, ≈ chance)", blind * 100.0);
    eprintln!("[GOAL]   GOAL-CONDITIONED value (⊕ the goal) ... {:.0}%   (the goal gates the reward map — one place, opposite value per goal)", cond * 100.0);
    eprintln!("[GOAL] VERDICT (Sébastien's key). Reward is not a property of a place — it is a property of (place, GOAL). Every experience is good or bad RELATIVE TO the goal you were pursuing, so the morph binds (goal, place)→reward, and the SAME place is sought under one goal and avoided under another. A goal-blind value model gets contradictory targets for the same features ({:.0}% ≈ chance); conditioning the reward readout on the goal resolves it ({:.0}%) — it is literally XOR(type, goal), the context-gating the substrate already cracked (substrate_context / grow_backbone: shared feature basis, per-goal readout). This is what lets ONE organism pursue MANY goals over the SAME grown map: curiosity finds sources, the goal-conditioned reward/pheromone says which are good for the goal at hand, and navigation reaches them. Autonomy = goals × a goal-conditioned value over the grown geometry. Zero core code.", blind * 100.0, cond * 100.0);

    assert!(
        cond > 0.85,
        "goal-conditioned value must predict reward across both goals ({:.2})",
        cond
    );
    assert!(cond > blind + 0.25, "conditioning on the goal must resolve what a goal-blind model can't ({blind:.2}→{cond:.2})");
}
