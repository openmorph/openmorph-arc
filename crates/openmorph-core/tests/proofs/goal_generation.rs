// SPDX-License-Identifier: MIT-0
//! GROWN GOAL-GENERATION — the organism invents its OWN subgoals on the FRONTIER of its grown concept-map, so it
//! keeps making directed progress in the regime MorphBot's Phase-A flagged as the residual ARC-3 tail: SPARSE-
//! first-reward games where the extrinsic reward is effectively ABSENT in-budget (reward only at a deep/rare
//! state a reward-seeker never stumbles onto). Curriculum transfers DYNAMICS, but a reward-seeking actor with
//! nothing to seek STALLS. The lever is to stop seeking reward and instead SELF-GENERATE goals: a self-set goal
//! `s*` = a preferred state placed at an UNDER-VISITED / farthest-point region of the organism's OWN grown key
//! map. Pursuing frontier goals drives COVERAGE of the reachable abstract-state space, and coverage is what
//! stumbles onto the sparse reward far more often than reward-seeking that has nothing to seek.
//!
//! This composes SHIPPED grown primitives, ZERO core change (a usage protocol, exactly like `substrate_discover`
//! / `substrate_skillmap`):
//!   * the GROWN concept-map — `metric_step` over the organism's TRANSITION experience (states experienced
//!     together as a move lie near on the map); the SAME grown metric backbone as `substrate_metric`.
//!   * the GROWN frontier goal — farthest-point novelty over the organism's OWN grown `position`s (the
//!     `substrate_skillmap` frontier, now generating GOALS not just ranking tries): the reachable state whose
//!     min map-distance to everything already visited is largest.
//!   * the REACH — `prefer` the self-set goal, `relax_values` the cost-to-go over the EXPERIENCED adjacency,
//!     descend (`value_of`); the `substrate_free_energy_policy` planner, toward a self-generated `s*`.
//!
//! THE WORLD — a 6×5 GRID bulk (30 densely-connected shallow states) with a long THIN deep TAIL (a 10-state chain
//! hanging off one grid corner); the sole reward sits at the tail TIP. To reach it an actor must commit to a
//! 10-deep dead-end branch — a random walk almost never does, and there is no reward gradient to pull it there
//! (the reward is unobserved until reached). On the grown map the tail lays out as a line extending AWAY from the
//! bulk, so the tip is the single farthest map point — the frontier's FIRST goal.
//!
//! THREE ARMS, matched step budget, scored INSIDE (coverage + did-it-reach-the-rare-reward), 8 seeds:
//!   * EXTRINSIC-ONLY (reward-seeking, no self-goals) — nothing to seek ⇒ an undirected random walk. Covers the
//!     bulk, ~never descends the thin tail. STALLS on the rare reward.
//!   * GROWN GOAL-GENERATION (self-set frontier goals + reach) — the tail tip IS the farthest map point, so it is
//!     the first self-set goal; the planner walks straight down the tail to it. COVERS the space, HITS the reward.
//!   * SHUFFLE control (truth-not-trophy) — the SAME frontier goal-generation + reach, but over PERMUTED map
//!     positions (the grown frontier ordering destroyed). The tail is camouflaged among the bulk, so the frontier
//!     rarely sets a deep-tail goal ⇒ collapses toward extrinsic-only. Proves the GROWN frontier is load-bearing,
//!     not goal-setting per se.
//!
//! AUTHORING LEDGER — GIVEN: the state/action interface (the adjacency the organism experiences by acting; the
//! reward is a single rare state whose LOCATION is unknown until visited). GROWN: the concept-map (`metric_step`),
//! the frontier goal placement (farthest-point over the organism's own `position`s), the reach (`relax_values`).
//! FIXED: the substrate laws. THE LOOP: the pick-goal → plan → walk → mark-visited cycle is a harness protocol
//! (like discovery), not a core faculty. Nothing is an authored waypoint or a hand-set goal curriculum.
//!
//!   cargo test -p openmorph-core --test proofs goal_generation -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const GW: usize = 6; // grid bulk width
const GH: usize = 5; // grid bulk height
const BULK: usize = GW * GH; // 30 shallow, densely-connected states
const TAIL: usize = 10; // a long thin deep tail hanging off grid corner 0
const N: usize = BULK + TAIL; // 40 abstract states
const REWARD: usize = N - 1; // the sole reward — the tail TIP (deepest state)

/// The world's transition structure (undirected adjacency) — the state/action interface the organism EXPERIENCES
/// by acting. A 6×5 4-connected grid bulk + a 10-state chain tail off grid corner 0. This is GIVEN (the task's
/// dynamics); the reward LOCATION inside it is NOT given — it is unobserved until a state is visited.
fn world() -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); N];
    let link = |a: usize, b: usize, adj: &mut Vec<Vec<usize>>| {
        adj[a].push(b);
        adj[b].push(a);
    };
    // grid bulk (4-connectivity)
    for r in 0..GH {
        for c in 0..GW {
            let s = r * GW + c;
            if c + 1 < GW {
                link(s, s + 1, &mut adj);
            }
            if r + 1 < GH {
                link(s, s + GW, &mut adj);
            }
        }
    }
    // thin deep tail: grid corner 0 → BULK → BULK+1 → … → REWARD
    link(0, BULK, &mut adj);
    for k in 0..TAIL - 1 {
        link(BULK + k, BULK + k + 1, &mut adj);
    }
    adj
}

/// GROW the concept-map from TRANSITION experience: sample the world's edges and `metric_step` each experienced
/// pair (states reached together as a move lie near; random negatives carve the metric). Annealed rate unfolds
/// the metric (`substrate_metric`). Returns the grown positions — the tail chain lays out as a line extending
/// away from the bulk, so the tip is the single farthest map point.
fn grow_map(adj: &[Vec<usize>], seed: u64) -> Vec<Vec<f32>> {
    // flatten the adjacency into a uniform edge list (each undirected edge once) — the transition experience.
    let mut edges = Vec::new();
    for (a, nbrs) in adj.iter().enumerate() {
        for &b in nbrs {
            if a < b {
                edges.push((a, b));
            }
        }
    }
    let mut map = Field::new(N);
    let mut rng = Rng::new(seed ^ 0xE0_6E);
    const ITERS: usize = 24000;
    for it in 0..ITERS {
        let (a, b) = edges[rng.below(edges.len())];
        let rate = 0.20 * (1.0 - it as f32 / ITERS as f32).max(0.02); // annealed — unfold the metric
        map.metric_step(&[a, b], rate, 12);
    }
    (0..N).map(|c| map.position(c)).collect()
}

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// GROWN FRONTIER GOAL — self-generate the next subgoal as the UNVISITED state whose min map-distance to the
/// visited set is largest (farthest-point novelty over the organism's OWN grown positions, `substrate_skillmap`).
/// `None` when everything is visited. This is the whole of goal-generation: a preferred state `s*` placed at the
/// under-explored frontier of the grown key map.
fn frontier_goal(pos: &[Vec<f32>], visited: &[bool]) -> Option<usize> {
    (0..N).filter(|&s| !visited[s]).max_by(|&a, &b| {
        let da = (0..N)
            .filter(|&v| visited[v])
            .map(|v| dist(&pos[a], &pos[v]))
            .fold(f32::MAX, f32::min);
        let db = (0..N)
            .filter(|&v| visited[v])
            .map(|v| dist(&pos[b], &pos[v]))
            .fold(f32::MAX, f32::min);
        da.total_cmp(&db)
    })
}

/// REACH a self-set goal — `prefer` it, `relax_values` the cost-to-go over the EXPERIENCED adjacency, then descend
/// greedily from `start` (`substrate_free_energy_policy`). Marks every state on the walk `visited`; returns the
/// state it ends on (the goal, or wherever it stalled within `max_steps`). The planner is a fresh value field.
fn reach(
    adj: &[Vec<usize>],
    start: usize,
    goal: usize,
    visited: &mut [bool],
    max_steps: usize,
) -> usize {
    let mut plan = Field::new(N);
    plan.prefer(goal, 0.0);
    plan.relax_values(adj, 1.0, 0); // value-iterate hop-cost-to-go over the experienced dynamics
    let mut s = start;
    visited[s] = true;
    for _ in 0..max_steps {
        if s == goal {
            break;
        }
        // descend the value field: step to the neighbour with the lowest cost-to-go (grown reach, no argmax label).
        let next = adj[s]
            .iter()
            .copied()
            .min_by(|&x, &y| plan.value_of(x).total_cmp(&plan.value_of(y)));
        match next {
            Some(nx) if plan.value_of(nx) < plan.value_of(s) => {
                s = nx;
                visited[s] = true;
            }
            _ => break, // no descent available (shouldn't happen on this connected world)
        }
    }
    s
}

/// GROWN GOAL-GENERATION arm: repeatedly self-generate a frontier goal on `pos` and reach it. `pos` is either the
/// grown map (the real arm) or a shuffled permutation of it (the control). Returns (coverage fraction, reached
/// the rare reward?, tail depth reached).
fn goal_gen_run(
    adj: &[Vec<usize>],
    pos: &[Vec<f32>],
    episodes: usize,
    steps: usize,
) -> (f32, bool, usize) {
    let mut visited = vec![false; N];
    let mut s = BULK / 2; // start in the middle of the bulk
    visited[s] = true;
    for _ in 0..episodes {
        match frontier_goal(pos, &visited) {
            Some(goal) => s = reach(adj, s, goal, &mut visited, steps),
            None => break,
        }
    }
    score(&visited)
}

/// EXTRINSIC-ONLY arm: reward-seeking with nothing to seek (the reward is unobserved) ⇒ an undirected random walk,
/// matched total step budget. Covers the bulk, ~never commits to the thin deep tail.
fn extrinsic_run(adj: &[Vec<usize>], total_steps: usize, rng: &mut Rng) -> (f32, bool, usize) {
    let mut visited = vec![false; N];
    let mut s = BULK / 2;
    visited[s] = true;
    for _ in 0..total_steps {
        let nbrs = &adj[s];
        s = nbrs[rng.below(nbrs.len())];
        visited[s] = true;
    }
    score(&visited)
}

/// Score a visited set INSIDE: overall coverage, whether the rare reward was reached, and the deepest tail state
/// visited (0 = never entered the tail, TAIL = reached the reward tip).
fn score(visited: &[bool]) -> (f32, bool, usize) {
    let cov = visited.iter().filter(|&&v| v).count() as f32 / N as f32;
    let deep = (0..TAIL)
        .rev()
        .find(|&k| visited[BULK + k])
        .map(|k| k + 1)
        .unwrap_or(0);
    (cov, visited[REWARD], deep)
}

#[test]
fn grown_frontier_goal_generation_lifts_the_no_extrinsic_reward_regime() {
    let adj = world();
    let seeds = 8u64;
    const EPISODES: usize = 5; // self-set goals per life
    const STEPS: usize = 25; // step cap per episode (grid path ≤9 + full tail 10 ≤ 25)
    let budget = EPISODES * STEPS; // matched total step budget for the extrinsic random walk

    // ── Sanity: the GROWN map must separate the deep tail from the bulk (the frontier's leverage). ──
    let pos0 = grow_map(&adj, 0);
    let bulk_ctr: Vec<f32> = {
        let mut c = vec![0.0; pos0[0].len()];
        for p in pos0.iter().take(BULK) {
            for (ci, &v) in p.iter().enumerate() {
                c[ci] += v / BULK as f32;
            }
        }
        c
    };
    let within_bulk = (0..BULK).map(|s| dist(&pos0[s], &bulk_ctr)).sum::<f32>() / BULK as f32;
    let tip_far = dist(&pos0[REWARD], &bulk_ctr);
    let sep = tip_far / within_bulk.max(1e-6);
    eprintln!(
        "[GOALGEN] grown map: deep-tail TIP sits {sep:.1}× farther from the bulk centroid than a bulk state does — the tip is the frontier's first self-set goal."
    );

    let (mut cov_e, mut cov_g, mut cov_s) = (0.0f32, 0.0, 0.0);
    let (mut hit_e, mut hit_g, mut hit_s) = (0usize, 0usize, 0usize);
    let (mut deep_e, mut deep_g, mut deep_s) = (0usize, 0usize, 0usize);
    for seed in 0..seeds {
        let pos = grow_map(&adj, seed);
        // shuffle control: permute the grown positions (destroy the frontier's grown ordering).
        let mut perm: Vec<usize> = (0..N).collect();
        let mut srng = Rng::new(0x5EED ^ seed);
        for i in (1..N).rev() {
            perm.swap(i, srng.below(i + 1));
        }
        let shuf: Vec<Vec<f32>> = (0..N).map(|s| pos[perm[s]].clone()).collect();

        let (cg, hg, dg) = goal_gen_run(&adj, &pos, EPISODES, STEPS);
        let (cs, hs, ds) = goal_gen_run(&adj, &shuf, EPISODES, STEPS);
        let mut ern = Rng::new(0xE0_1D ^ seed);
        let (ce, he, de) = extrinsic_run(&adj, budget, &mut ern);

        cov_g += cg;
        cov_s += cs;
        cov_e += ce;
        hit_g += hg as usize;
        hit_s += hs as usize;
        hit_e += he as usize;
        deep_g += dg;
        deep_s += ds;
        deep_e += de;
    }
    let sf = seeds as f32;
    let (cov_e, cov_g, cov_s) = (cov_e / sf, cov_g / sf, cov_s / sf);
    let (dep_e, dep_g, dep_s) = (deep_e as f32 / sf, deep_g as f32 / sf, deep_s as f32 / sf);

    eprintln!(
        "[GOALGEN] {seeds} seeds, {EPISODES} self-set goals × {STEPS} steps (budget {budget} steps), reward at the {TAIL}-deep tail tip:"
    );
    eprintln!(
        "[GOALGEN]   EXTRINSIC-ONLY (reward-seek, nothing to seek): coverage {cov_e:.2}  tail-depth {dep_e:.1}/{TAIL}  rare-reward {hit_e}/{seeds}  <- STALLS"
    );
    eprintln!(
        "[GOALGEN]   GROWN GOAL-GEN (frontier goals + reach)      : coverage {cov_g:.2}  tail-depth {dep_g:.1}/{TAIL}  rare-reward {hit_g}/{seeds}  <- COVERS + HITS"
    );
    eprintln!(
        "[GOALGEN]   SHUFFLE control (permuted map positions)     : coverage {cov_s:.2}  tail-depth {dep_s:.1}/{TAIL}  rare-reward {hit_s}/{seeds}  <- collapses toward extrinsic"
    );
    eprintln!(
        "[GOALGEN] VERDICT: where the extrinsic reward is absent in-budget, GROWN goal-generation on the concept-map FRONTIER keeps the organism making directed progress — it self-sets the deep tail tip as its farthest-point goal, covers the abstract-state space, and hits the rare reward {hit_g}/{seeds} where reward-seeking (nothing to seek) stalls {hit_e}/{seeds}. The SHUFFLE control ({hit_s}/{seeds}) — same goal-generation + reach over permuted positions — collapses toward extrinsic, so the GROWN frontier ordering is load-bearing, not goal-setting per se. Zero core change; a usage protocol over `metric_step` + farthest-point + `relax_values`."
    );

    // the deep-tail separation the frontier exploits
    assert!(
        sep > 3.0,
        "the grown map must push the deep tail far from the bulk (tip/bulk separation {sep:.1}×)"
    );
    // extrinsic-only STALLS on the rare reward (nothing to seek)
    assert!(
        hit_e <= 1 && dep_e < 5.0,
        "extrinsic-only (reward-seeking) must stall on the rare reward (hits {hit_e}/{seeds}, tail-depth {dep_e:.1})"
    );
    // grown goal-generation reaches the rare reward far more often, and covers more
    assert!(
        hit_g >= 7,
        "grown frontier goal-generation must reach the rare reward (hits {hit_g}/{seeds})"
    );
    assert!(
        cov_g > cov_e,
        "grown goal-generation must cover more of the abstract-state space than the reward-seeker ({cov_g:.2} vs {cov_e:.2})"
    );
    // shuffle collapses toward extrinsic — the grown frontier is the load
    assert!(
        hit_s + 2 <= hit_g,
        "shuffling the map (destroying the grown frontier) must collapse the reward-hit toward extrinsic ({hit_s}/{seeds} vs grown {hit_g}/{seeds})"
    );
}
