// SPDX-License-Identifier: MIT-0
//! REASONING ON THE CONCEPT-MAP — the grown metric doesn't just speed a search, it grants a REASONING ABILITY:
//! finding and VERIFYING a correct multi-step derivation to a goal, within a bounded thought budget that blind
//! search cannot meet on deep problems. This is planning / theorem-proving: from a start concept, chain valid
//! steps (operations) to reach a goal concept; the ANSWER is the derivation (the op sequence), and it is
//! checked by replay — correctness from verification (System-2), reach from the map (System-1).
//!
//! The organism grows the map from its OWN experience (co-activation of concepts used in sequence — no
//! oracle). Then, with a fixed thought budget, it must solve problems of increasing DEPTH:
//!   - BLIND breadth-first exhausts the budget wandering — solves shallow, FAILS deep.
//!   - MAP-guided best-first heads toward the goal by map-distance — solves deep within the SAME budget.
//!   - SHUFFLED map collapses back to blind (the geometry, not the algorithm, grants the ability).
//!
//! Every returned derivation is verified valid. So the map is what lets the organism REASON several steps
//! ahead under a fixed compute budget — the ability, not just the speed.
//!
//!   cargo test -p openmorph-core --test proofs substrate_reason -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::collections::VecDeque;

use crate::common::Rng;

const N: usize = 90; // concepts (states)
const BUDGET: usize = 14; // thoughts (nodes expanded) allowed per problem

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// True graph shortest-path length start→goal (BFS), or None if unreachable — used to GROUP problems by depth
/// and to know the ground truth. NOT available to the reasoner.
fn true_depth(adj: &[Vec<usize>], s: usize, t: usize) -> Option<usize> {
    let mut d = vec![usize::MAX; adj.len()];
    d[s] = 0;
    let mut q = VecDeque::from([s]);
    while let Some(u) = q.pop_front() {
        if u == t {
            return Some(d[u]);
        }
        for &v in &adj[u] {
            if d[v] == usize::MAX {
                d[v] = d[u] + 1;
                q.push_back(v);
            }
        }
    }
    None
}

/// Solve start→goal within `budget` expansions; `h` orders the frontier (None = blind BFS). Returns the
/// DERIVATION (the path of concepts) if found, tracking parents so the answer can be verified.
fn solve(
    adj: &[Vec<usize>],
    s: usize,
    t: usize,
    budget: usize,
    h: Option<&dyn Fn(usize) -> f32>,
) -> Option<Vec<usize>> {
    let mut parent = vec![usize::MAX; adj.len()];
    let mut seen = vec![false; adj.len()];
    let mut frontier: Vec<(f32, usize)> = vec![(0.0, s)];
    seen[s] = true;
    let mut expanded = 0;
    while let Some(pick) = (!frontier.is_empty()).then(|| match h {
        Some(_) => frontier
            .iter()
            .enumerate()
            .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
            .map(|(i, _)| i)
            .unwrap(),
        None => 0,
    }) {
        let (_, u) = frontier.remove(pick);
        expanded += 1;
        if u == t {
            // reconstruct the derivation
            let mut path = vec![t];
            let mut c = t;
            while c != s {
                c = parent[c];
                path.push(c);
            }
            path.reverse();
            return Some(path);
        }
        if expanded >= budget {
            return None; // out of thoughts
        }
        for &v in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                parent[v] = u;
                frontier.push((h.map(|f| f(v)).unwrap_or(0.0), v));
            }
        }
    }
    None
}

/// Verify a derivation: every step is a legal edge and it ends at the goal — the reasoner's answer must be a
/// genuine, replayable proof, not just a claim.
fn valid(path: &[usize], adj: &[Vec<usize>], s: usize, t: usize) -> bool {
    if path.first() != Some(&s) || path.last() != Some(&t) {
        return false;
    }
    path.windows(2).all(|w| adj[w[0]].contains(&w[1]))
}

#[test]
fn the_concept_map_grants_a_bounded_reasoning_ability() {
    let mut rng = Rng::new(0x00C0_FFEE_5EED);
    // ── The world: concepts in a hidden latent space; each enables its k nearest (the valid steps). ──
    let latent: Vec<[f32; 2]> = (0..N).map(|_| [rng.f(), rng.f()]).collect();
    let adj: Vec<Vec<usize>> = (0..N)
        .map(|i| {
            let mut d: Vec<(f32, usize)> = (0..N)
                .filter(|&j| j != i)
                .map(|j| (dist(&latent[i], &latent[j]), j))
                .collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0));
            d.into_iter().take(4).map(|(_, j)| j).collect()
        })
        .collect();

    // ── Grow the map from experience: walks of consecutive concepts co-activate (no oracle). ──
    let mut org = Field::new(N);
    for w in 0..6000 {
        let mut cur = rng.below(N);
        for _ in 0..14 {
            let nxt = adj[cur][rng.below(adj[cur].len())];
            org.co_activate(&[cur, nxt], 0.05);
            cur = nxt;
        }
        if w % 150 == 149 {
            org.spread_positions();
        }
    }
    org.spread_positions();
    let pos: Vec<Vec<f32>> = (0..N).map(|i| org.position(i)).collect();
    let mut shuf = pos.clone();
    for i in (1..N).rev() {
        shuf.swap(i, rng.below(i + 1));
    }

    // ── Solve problems by DEPTH under the fixed budget; verify every derivation. ──
    // buckets: shallow (1-2), mid (3-4), deep (5+)
    let bucket = |d: usize| {
        if d <= 2 {
            0
        } else if d <= 4 {
            1
        } else {
            2
        }
    };
    let names = ["shallow(1-2)", "mid(3-4)", "deep(5+)"];
    let (mut ok, mut tot) = ([[0.0f64; 3]; 3], [0.0f64; 3]); // ok[method][bucket]
    let mut invalid = 0usize;
    let mut solved = 0usize;
    for _ in 0..1500 {
        let (s, t) = (rng.below(N), rng.below(N));
        let depth = match true_depth(&adj, s, t) {
            Some(d) if d >= 1 => d,
            _ => continue,
        };
        let bk = bucket(depth);
        tot[bk] += 1.0;
        // three reasoners, SAME budget
        let hg = |n: usize| dist(&pos[n], &pos[t]);
        let hs = |n: usize| dist(&shuf[n], &shuf[t]);
        for (m, path) in [
            solve(&adj, s, t, BUDGET, None),
            solve(&adj, s, t, BUDGET, Some(&hg)),
            solve(&adj, s, t, BUDGET, Some(&hs)),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(p) = path {
                if valid(&p, &adj, s, t) {
                    ok[m][bk] += 1.0;
                    solved += 1;
                } else {
                    invalid += 1;
                }
            }
        }
    }
    let rate = |m: usize, bk: usize| 100.0 * ok[m][bk] / tot[bk].max(1.0);

    eprintln!("[REASON] solving derivations within a {BUDGET}-thought budget, success% by depth (all answers verified):");
    for bk in 0..3 {
        eprintln!(
            "  {:<12}  BLIND {:>3.0}%   MAP {:>3.0}%   SHUFFLED {:>3.0}%   (n={:.0})",
            names[bk],
            rate(0, bk),
            rate(1, bk),
            rate(2, bk),
            tot[bk]
        );
    }
    eprintln!("[REASON] VERDICT: the grown concept-map grants a REASONING ABILITY, not just speed. Under a fixed thought budget, blind breadth-first solves shallow problems but RUNS OUT on deep ones; the map-guided reasoner heads toward the goal by learned distance and finds VERIFIED multi-step derivations several steps deep within the SAME budget — and a SHUFFLED map collapses back to blind, so the geometry, not the algorithm, is what grants the reach. Every returned derivation was replay-checked (0 invalid). The organism reasons because its concepts finally have a place to stand — it can see where the goal is and step toward it.");

    assert_eq!(
        invalid, 0,
        "every returned derivation must be a valid, replayable proof ({invalid} invalid)"
    );
    assert!(solved > 0, "sanity: some problems were solved");
    // shallow: everyone can do it.
    assert!(
        rate(1, 0) > 80.0,
        "map must solve shallow problems (got {:.0}%)",
        rate(1, 0)
    );
    // DEEP is the ability: the map solves deep derivations blind cannot reach in budget.
    assert!(
        rate(1, 2) > rate(0, 2) + 20.0,
        "the map must grant deep-reasoning reach beyond blind (map {:.0}% vs blind {:.0}%)",
        rate(1, 2),
        rate(0, 2)
    );
    // and the geometry — not the search algorithm — is what does it.
    assert!(
        rate(1, 2) > rate(2, 2) + 20.0,
        "the GROWN map must beat a SHUFFLED one on deep reasoning (map {:.0}% vs shuffled {:.0}%)",
        rate(1, 2),
        rate(2, 2)
    );
}
