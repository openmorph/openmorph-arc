// SPDX-License-Identifier: MIT-0
//! COMPOSITIONAL SKILL MEMORY — reusable RELATIONAL priors (cap #336, epic #337; composes the #335 goal inferer).
//!
//! THE GAP + THE MEASURED WALL (MorphBot). openmorph's durable transfer is perception-only; action-learning is
//! transient and does not serialize. Naive `(state, action)` transfer FAILS — controls and landmark positions
//! re-randomise per game, so a warm ABSOLUTE policy scored BELOW random on held-out games. What SHOULD transfer is
//! RELATIONAL strategy ("satisfy relation R with the target"), independent of where the target is or which action
//! index moves where.
//!
//! THE MECHANISM (no new core primitive — serialise the #335 grown goal inferer). A skill is the grown reward
//! predictor whose `reliance_of` identifies the win RELATION (`substrate_goal_inference`). It is a `Field`, so it
//! round-trips through `snapshot()`/`load()`. Transfer = load the skill on a NEW level (target moved), read the
//! RELATION it encodes, and plan to satisfy that relation w.r.t. the new target — the warm agent skips rediscovery.
//!
//! THE CLAIM (cap #336 in-core acceptance): a skill learned on level A transfers to held-out level B — WARM (loaded
//! relational skill) reaches the first reward in fewer observations than COLD (must rediscover the win condition);
//! and the skill round-trips through `snapshot`/`load`. THE HONEST CONTRAST that proves it is RELATIONAL, not
//! memorised: a RAW-ABSOLUTE warm transfer (reuse level A's goal CELLS) FAILS on B because the target moved.
//!
//! TRUTH-NOT-TROPHY. The levels, the (moved) target, the seeded PRNG, and the feature set are authored and
//! identical across arms; the win RELATION is shared A→B (that is what "same family" means) but its GROUNDING (the
//! target cell) differs, so only a relational skill can transfer. ZERO openmorph-core change: a usage protocol over
//! `add_unit`/`learn_step`/`reliance_of` + serde `snapshot`/`load` + `prefer`/`relax_values`. The decisive metric
//! (warm < cold actions-to-first-win on the 25 games) is MorphBot's live A/B.
//!
//!   cargo test -p openmorph-core --test proofs substrate_skill_memory -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const GW: usize = 5;
const GH: usize = 5;
const N_ACT: usize = 4;
const K: usize = 3; // candidate relations: 0=overlap, 1=x-align, 2=y-align (all relative to the target)

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

/// The candidate relations at `(ax,ay)` RELATIVE to `target` (bipolar). Relation `k` is the win when its feature > 0.
fn features(ax: usize, ay: usize, target: (usize, usize)) -> [f32; K] {
    let b = |c: bool| if c { 1.0 } else { -1.0 };
    [
        b(ax == target.0 && ay == target.1),
        b(ax == target.0),
        b(ay == target.1),
    ]
}
fn wins(k: usize, ax: usize, ay: usize, target: (usize, usize)) -> bool {
    features(ax, ay, target)[k] > 0.0
}

/// LEARN a skill on a level: random-play the hidden-relation-`k` game grounded at `target`, growing a reward
/// predictor over the RELATIVE features. Returns the skill `Field` (its `reliance_of` encodes the relation).
fn learn_skill(k: usize, target: (usize, usize), seed: u64) -> Field {
    let mut rng = Rng::new(seed);
    let mut f = Field::new(K + 1);
    f.add_unit((0..K).collect(), vec![0.0; K], 0.0, K);
    let (mut ax, mut ay) = (0usize, 0usize);
    for _ in 0..3000 {
        let a = rng.below(N_ACT);
        let (nx, ny) = step(ax, ay, a);
        let feats = features(nx, ny, target);
        let reward = if wins(k, nx, ny, target) { 1.0 } else { -1.0 };
        let inp: Vec<(usize, f32)> = (0..K).map(|i| (i, feats[i])).collect();
        f.learn_step(&inp, K, reward, 0.02, 50);
        ax = nx;
        ay = ny;
        if wins(k, ax, ay, target) {
            ax = 0;
            ay = 0;
        }
    }
    f
}

/// The relation a skill encodes = the feature it relies on most.
fn skill_relation(skill: &Field) -> usize {
    (0..K)
        .max_by(|&i, &j| skill.reliance_of(i).total_cmp(&skill.reliance_of(j)))
        .unwrap()
}

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

/// Plan to satisfy relation `k` w.r.t. `target` and return actions-to-reward from the start (or cap).
fn plan_to_relation(k: usize, target: (usize, usize), cap: usize) -> usize {
    let adj = grid_adjacency();
    let mut f = Field::new(GW * GH);
    for y in 0..GH {
        for x in 0..GW {
            if wins(k, x, y, target) {
                f.prefer(id(x, y), 0.0);
            }
        }
    }
    f.relax_values(&adj, 1.0, 0);
    let (mut x, mut y) = (0usize, 0usize);
    for t in 1..=cap {
        if wins(k, x, y, target) {
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

/// COLD: no skill — random-chase the (sparse) reward on level B until it fires.
fn cold_first_reward(k: usize, target: (usize, usize), seed: u64, cap: usize) -> usize {
    let mut rng = Rng::new(seed);
    let (mut x, mut y) = (0usize, 0usize);
    for t in 1..=cap {
        if wins(k, x, y, target) {
            return t - 1;
        }
        let a = rng.below(N_ACT);
        (x, y) = step(x, y, a);
    }
    cap
}

#[test]
fn relational_skill_transfers_across_moved_landmarks() {
    let cap = 400;
    let target_a = (3, 3); // level A grounding
    let target_b = (1, 1); // held-out level B: SAME relation family, target MOVED
    let seeds = [0x41u64, 0x42, 0x43];

    let (mut warm_wins, mut roundtrips) = (0usize, 0usize);
    let mut warm_obs = Vec::new();
    let mut cold_obs = Vec::new();
    let mut raw_obs = Vec::new();

    for k in 0..K {
        // learn the skill on level A, then SERIALISE + reload it (snapshot round-trip).
        let skill = learn_skill(k, target_a, 0xA0 + k as u64);
        let bytes = serde_json::to_vec(&skill).expect("skill serialises");
        let reloaded: Field = serde_json::from_slice(&bytes).expect("skill loads");
        if skill_relation(&reloaded) == skill_relation(&skill) {
            roundtrips += 1; // the reloaded skill encodes the same relation
        }
        let relation = skill_relation(&reloaded);
        if relation == k {
            warm_wins += 1;
        }

        // WARM: apply the relational skill on level B (target moved) — plan to satisfy the KNOWN relation w.r.t. B's
        // target. Observations = the plan length (no rediscovery needed).
        warm_obs.push(plan_to_relation(relation, target_b, cap));

        // COLD: no skill — must random-chase B's reward to find it the first time.
        let mc = seeds
            .iter()
            .map(|&s| cold_first_reward(k, target_b, s, cap))
            .sum::<usize>() as f32
            / seeds.len() as f32;
        cold_obs.push(mc);

        // RAW-ABSOLUTE warm transfer (the wall): reuse level A's goal CELLS (relation grounded at target_A) on B.
        // Since B's target moved, planning to A's cells reaches A's grounding, which is NOT a win on B.
        raw_obs.push(plan_to_relation_cells_of_a(k, target_a, target_b, cap));
    }

    let mean = |v: &[usize]| v.iter().sum::<usize>() as f32 / v.len() as f32;
    let meanf = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (mw, mc, mr) = (mean(&warm_obs), meanf(&cold_obs), mean(&raw_obs));

    // Skill round-trips through snapshot/load, and encodes the correct relation.
    assert_eq!(
        roundtrips, K,
        "every skill must round-trip through snapshot/load"
    );
    assert_eq!(
        warm_wins, K,
        "every skill must encode the true relation ({warm_wins}/{K})"
    );
    // WARM (relational) reaches B's first reward in far fewer observations than COLD.
    assert!(
        mc >= 2.0 * mw,
        "cold {mc:.1} must cost ≥2× warm(relational) {mw:.1} observations-to-first-reward"
    );
    // The RAW-ABSOLUTE transfer FAILS on B (the honest wall — it is memorised, not relational).
    assert!(
        mr >= mc,
        "raw-absolute transfer {mr:.1} must be NO BETTER than cold {mc:.1} (positions moved ⇒ memorisation fails)"
    );

    println!("\n  COMPOSITIONAL SKILL MEMORY (learn on A target {target_a:?}, transfer to B target {target_b:?}):");
    println!("    snapshot round-trips: {roundtrips}/{K}; correct relation: {warm_wins}/{K}");
    println!("    observations-to-first-reward on B — WARM(relational) {mw:.1}  vs COLD {mc:.1}  ({:.1}× fewer)", mc / mw.max(1.0));
    println!(
        "    RAW-ABSOLUTE transfer (the wall): {mr:.1}  (≥ cold ⇒ memorisation does NOT transfer)"
    );
    println!("  ⇒ what transfers is the grown RELATION (reliance), not the grounded cells — serialisable + composable.\n");
}

/// The RAW-ABSOLUTE arm: plan to the cells that satisfy relation `k` w.r.t. level A's target (the memorised
/// grounding), but score arrival against level B's win. Returns actions until a B-win (or cap — typically cap,
/// since A's grounding is not a B-win once the target moved).
fn plan_to_relation_cells_of_a(
    k: usize,
    target_a: (usize, usize),
    target_b: (usize, usize),
    cap: usize,
) -> usize {
    let adj = grid_adjacency();
    let mut f = Field::new(GW * GH);
    for y in 0..GH {
        for x in 0..GW {
            if wins(k, x, y, target_a) {
                f.prefer(id(x, y), 0.0); // memorised A-grounded goal cells
            }
        }
    }
    f.relax_values(&adj, 1.0, 0);
    let (mut x, mut y) = (0usize, 0usize);
    for t in 1..=cap {
        if wins(k, x, y, target_b) {
            return t - 1; // a B-win (only if A's grounding happens to coincide)
        }
        let cur = id(x, y);
        let next = adj[cur]
            .iter()
            .copied()
            .min_by(|&a, &b| f.value_of(a).total_cmp(&f.value_of(b)).then(a.cmp(&b)))
            .unwrap_or(cur);
        if next == cur {
            return cap;
        }
        (x, y) = (next % GW, next / GW);
    }
    cap
}
