// SPDX-License-Identifier: MIT-0
//! COVERAGE RECOVERY — keep BOTH coverage and efficiency (Gap 1, `docs/research/arc3-gaps-to-100.md`).
//!
//! THE GAP (MorphBot live A/B, reproduced here). A greedy planner over the world model RAISES efficiency but LOSES
//! coverage: reactive `act()` won 7/30 games at ~189 actions/win; a greedy planner won only 4/30 at ~32 actions —
//! more efficient, fewer wins, netting BELOW reactive. Efficiency was solved; COVERAGE is the open multiplicand.
//! The #333 honest-negative said the fix is NOT world-model σ² curiosity — it is directed exploration. This proves
//! the concrete lever in-core.
//!
//! THE LEVER (nature-based, no new core primitive). Do NOT greedily EXPLOIT (which narrows exploration and loses
//! coverage). PLAN THE EXPLORATION itself — go-explore over the EXPERIENCED map: expand the frontier, and when the
//! local frontier is exhausted, `relax_values`-PLAN an efficient reach back to a cell that still has unexplored
//! neighbours (instead of random-walking back). This keeps a random walk's COVERAGE (systematic reach of the whole
//! reachable set) at a planner's EFFICIENCY (no wasted re-treading). It is `reachability_options`' archive+reach
//! composed with `relax_values`, measured on the coverage×efficiency PRODUCT.
//!
//! THE CLAIM. Across many hidden-reward games, GO-EXPLORE dominates BOTH baselines on wins×efficiency:
//!   * vs GREEDY-EXPLOIT — far higher COVERAGE (greedy commits to a path and misses most rewards).
//!   * vs RANDOM — same-or-better coverage at far FEWER actions-to-reward (planned reaches, not a diffusive wander).
//!
//! TRUTH-NOT-TROPHY (the honest boundary). The lever is PLANNING THE REACH, not a magic frontier: `reachability_
//! options` already showed farthest-point ≈ random for FINDING a gradient-less reward, so this proof does NOT claim
//! the frontier finds faster — it claims go-explore keeps coverage WHILE gaining efficiency (no re-tread). The
//! worlds, hidden reward locations, seeded PRNG, and budget are authored and identical across arms; only the
//! exploration POLICY differs. ZERO openmorph-core change: a usage protocol over `relax_values`/`value_of` + an
//! experienced adjacency + an archive. The decisive metric (wins×efficiency on the 25 live games) is MorphBot's A/B.
//!
//!   cargo test -p openmorph-core --test proofs substrate_coverage_recovery -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::collections::VecDeque;

use crate::common::Rng;

const W: usize = 6;
const H: usize = 6;

fn id(x: usize, y: usize) -> usize {
    y * W + x
}
fn neighbours(c: usize) -> Vec<usize> {
    let (x, y) = (c % W, c / W);
    let mut v = Vec::new();
    if x + 1 < W {
        v.push(id(x + 1, y));
    }
    if x > 0 {
        v.push(id(x - 1, y));
    }
    if y + 1 < H {
        v.push(id(x, y + 1));
    }
    if y > 0 {
        v.push(id(x, y - 1));
    }
    v
}

/// A game: a hidden reward cell (never given to a policy). `budget` actions to reach it. Returns
/// `(reached, actions)` — actions is the step the reward was first reached (or `budget`).
type Outcome = (bool, usize);

/// RANDOM — a diffusive walk. High coverage eventually, but wastes actions re-treading.
fn random_walk(reward: usize, seed: u64, budget: usize) -> Outcome {
    let mut rng = Rng::new(seed);
    let mut pos = 0usize;
    for t in 1..=budget {
        if pos == reward {
            return (true, t - 1);
        }
        let nbrs = neighbours(pos);
        pos = nbrs[rng.below(nbrs.len())];
    }
    (pos == reward, budget)
}

/// GREEDY-EXPLOIT — a fixed-priority exploit policy (prefer right, then down, then left, then up). With no reward
/// signal it COMMITS to that direction and, at the boundary, oscillates in a corner — directed action that NARROWS
/// exploration to an L-shaped path (MorphBot's coverage-losing planner). Efficient IF the reward sits on that path,
/// else it never covers the rest. (Any fixed exploit-without-exploration policy has this shape.)
fn greedy_exploit(reward: usize, budget: usize) -> Outcome {
    let mut pos = 0usize;
    for t in 1..=budget {
        if pos == reward {
            return (true, t - 1);
        }
        let (x, y) = (pos % W, pos / W);
        pos = if x + 1 < W {
            id(x + 1, y) // right
        } else if y + 1 < H {
            id(x, y + 1) // down
        } else if x > 0 {
            id(x - 1, y) // left (at the far corner ⇒ oscillate with 'right')
        } else {
            id(x, y.saturating_sub(1)) // up
        };
    }
    (pos == reward, budget)
}

/// GO-EXPLORE — plan the EXPLORATION. Expand the frontier locally; when the current cell has no unvisited neighbour,
/// `relax_values`-plan an efficient reach to the nearest visited cell that still has one, then keep expanding. Keeps
/// random's coverage at a planner's efficiency (no re-tread). Uses only the EXPERIENCED adjacency.
fn go_explore(reward: usize, budget: usize) -> Outcome {
    let n = W * H;
    let mut visited = vec![false; n];
    let mut known_adj: Vec<Vec<usize>> = vec![Vec::new(); n]; // edges the agent has traversed/seen
    let mut pos = 0usize;
    visited[pos] = true;
    let mut actions = 0usize;

    // reveal the edges out of a visited cell (experienced on arrival — the agent sees its immediate neighbours).
    let reveal = |c: usize, known_adj: &mut Vec<Vec<usize>>| {
        for nb in neighbours(c) {
            if !known_adj[c].contains(&nb) {
                known_adj[c].push(nb);
            }
        }
    };
    reveal(pos, &mut known_adj);

    while actions < budget {
        if pos == reward {
            return (true, actions);
        }
        // a locally-unvisited neighbour? step to it (explore).
        if let Some(&nb) = known_adj[pos].iter().find(|&&nb| !visited[nb]) {
            pos = nb;
            visited[pos] = true;
            reveal(pos, &mut known_adj);
            actions += 1;
            continue;
        }
        // local frontier exhausted → PLAN a reach to the nearest cell that still has an unvisited neighbour.
        let target = nearest_frontier(pos, &visited, &known_adj);
        let Some(target) = target else {
            break; // everything reachable is visited — the whole component is covered
        };
        // plan over the EXPERIENCED graph and walk the planned path (efficient reach, not a random wander).
        let mut field = Field::new(n);
        field.prefer(target, 0.0);
        field.relax_values(&known_adj, 1.0, 0);
        while pos != target && actions < budget {
            if pos == reward {
                return (true, actions);
            }
            let step = known_adj[pos]
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    field
                        .value_of(a)
                        .total_cmp(&field.value_of(b))
                        .then(a.cmp(&b))
                })
                .unwrap_or(pos);
            if step == pos {
                break;
            }
            pos = step;
            if !visited[pos] {
                visited[pos] = true;
                reveal(pos, &mut known_adj);
            }
            actions += 1;
        }
    }
    (pos == reward, actions.min(budget))
}

/// The nearest visited cell (BFS over the experienced graph) that still has an unvisited neighbour — the reach target.
fn nearest_frontier(from: usize, visited: &[bool], known_adj: &[Vec<usize>]) -> Option<usize> {
    let mut seen = vec![false; visited.len()];
    let mut q = VecDeque::new();
    seen[from] = true;
    q.push_back(from);
    while let Some(c) = q.pop_front() {
        if known_adj[c].iter().any(|&nb| !visited[nb]) {
            return Some(c);
        }
        for &nb in &known_adj[c] {
            if visited[nb] && !seen[nb] {
                seen[nb] = true;
                q.push_back(nb);
            }
        }
    }
    None
}

#[test]
fn go_explore_keeps_coverage_at_planner_efficiency() {
    let budget = 60;
    let games = 24;
    let mut rng = Rng::new(0xC0FFEE);
    let (mut r_win, mut g_win, mut x_win) = (0usize, 0usize, 0usize);
    let (mut r_act, mut g_act, mut x_act) = (0usize, 0usize, 0usize);
    let (mut r_n, mut g_n, mut x_n) = (0usize, 0usize, 0usize);

    for _ in 0..games {
        let reward = 1 + rng.below(W * H - 1); // any cell but the start
        let seed = rng.u();

        let (rw, ra) = random_walk(reward, seed, budget);
        let (gw, ga) = greedy_exploit(reward, budget);
        let (xw, xa) = go_explore(reward, budget);

        if rw {
            r_win += 1;
            r_act += ra;
            r_n += 1;
        }
        if gw {
            g_win += 1;
            g_act += ga;
            g_n += 1;
        }
        if xw {
            x_win += 1;
            x_act += xa;
            x_n += 1;
        }
    }

    let eff = |act: usize, n: usize| {
        if n == 0 {
            0.0
        } else {
            50.0 / (act as f32 / n as f32).max(1.0)
        }
    };
    let cov = |w: usize| w as f32 / games as f32;
    // wins × efficiency (the ARC-3 product): coverage × per-win efficiency, efficiency capped at 1 (50-move parity).
    let product = |w: usize, act: usize, n: usize| cov(w) * eff(act, n).min(50.0) / 50.0;

    let (rp, gp, xp) = (
        product(r_win, r_act, r_n),
        product(g_win, g_act, g_n),
        product(x_win, x_act, x_n),
    );

    // GO-EXPLORE keeps coverage (≥ random — it covers the whole reachable set) ...
    assert!(
        x_win >= r_win,
        "go-explore coverage {x_win} must be ≥ random {r_win} (systematic reach keeps coverage)"
    );
    // ... at far better efficiency (fewer actions-to-reward than a diffusive random walk) ...
    let (ra_mean, xa_mean) = (
        r_act as f32 / r_n.max(1) as f32,
        x_act as f32 / x_n.max(1) as f32,
    );
    assert!(
        xa_mean < ra_mean,
        "go-explore actions/win {xa_mean:.1} must be < random {ra_mean:.1} (planned reach, no re-tread)"
    );
    // ... and it beats the greedy planner on COVERAGE (greedy commits and misses) ...
    assert!(
        x_win > g_win,
        "go-explore coverage {x_win} must exceed greedy-exploit {g_win} (greedy narrows exploration)"
    );
    // ... so it dominates BOTH on the wins×efficiency product — coverage recovered.
    assert!(
        xp > rp && xp > gp,
        "go-explore product {xp:.3} must beat random {rp:.3} AND greedy {gp:.3}"
    );

    println!("\n  COVERAGE RECOVERY (plan the exploration; {games} hidden-reward games, budget {budget}):");
    println!(
        "    RANDOM        : wins {r_win:>2}/{games}  actions/win {ra_mean:>5.1}  product {rp:.3}"
    );
    println!(
        "    GREEDY-EXPLOIT: wins {g_win:>2}/{games}  actions/win {:>5.1}  product {gp:.3}  (commits ⇒ loses coverage)",
        g_act as f32 / g_n.max(1) as f32
    );
    println!(
        "    GO-EXPLORE    : wins {x_win:>2}/{games}  actions/win {xa_mean:>5.1}  product {xp:.3}  (coverage KEPT, efficient)"
    );
    println!(
        "  ⇒ the fix for lost coverage is PLANNING THE EXPLORATION, not greedy exploitation.\n"
    );
}
