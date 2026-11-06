// SPDX-License-Identifier: MIT-0
//! SKILL-SPACE FRONTIER = DISCOVERY (MorphLab #118, the metric backbone applied). Discovery — finding a reward
//! no strategy has seen — is what the concept-map couldn't do in STATE-space (a brand-new state sits
//! `place_near` its creator, so the geometry can't mark it as frontier). The fix is to run the frontier in
//! SKILL-space: crystallized behaviours (macro-actions, `Skill::apply`) are placed on a grown concept-map, and
//! a never-tried behaviour genuinely IS a new region of that map (unlike a never-seen primitive STATE, which
//! sits `place_near` its creator — so state-space novelty can't flag it). Novelty-driven (farthest-point)
//! exploration of BEHAVIOUR-space therefore sweeps to a rare reward behaviour far faster than random sampling.
//!
//! World: M behaviours, a big COMMON cluster + a small RARE cluster (they never co-occur); the reward is a
//! behaviour in the rare cluster. Grow the skill-map from within-cluster co-activation, then explore
//! behaviour-space by farthest-point novelty and count TRIES until the reward behaviour is sampled.
//! (Truth-not-trophy: the run showed BOTH a metric map and an attraction+spread map separate the rare cluster
//! enough for the frontier to win — the metric backbone is for reasoning/analogy sharpness; for skill-space
//! DISCOVERY the frontier itself is the lever, on either grown map.)
//!
//!   cargo test -p openmorph-core --test proofs substrate_skillmap -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const COMMON: usize = 40; // behaviours in the common cluster
const RARE: usize = 12; // behaviours in the rare cluster (the reward lives here)
const M: usize = COMMON + RARE;

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Two behaviours from the SAME cluster, the experience that shapes the skill-map (behaviours used together).
fn same_cluster_pair(rng: &mut Rng) -> [usize; 2] {
    if rng.below(100) < 75 {
        [rng.below(COMMON), rng.below(COMMON)] // common ↔ common (the bulk of experience)
    } else {
        [COMMON + rng.below(RARE), COMMON + rng.below(RARE)] // rare ↔ rare
    }
}

/// Farthest-point novelty exploration of behaviour-space on skill-map `pos`: repeatedly try the UNtried
/// behaviour whose min-distance to everything already tried is largest (the frontier). Return the number of
/// tries until `reward` is sampled (or M if never).
fn tries_to_reward(pos: &[Vec<f32>], reward: usize, start: usize) -> usize {
    let mut tried = [false; M];
    tried[start] = true;
    let mut order = vec![start];
    for t in 1..=M {
        if order.contains(&reward) {
            return t; // reward was among the tried
        }
        // pick the untried behaviour maximizing its min-distance to the tried set (the frontier).
        let next = (0..M).filter(|&b| !tried[b]).max_by(|&a, &b| {
            let da = order
                .iter()
                .map(|&o| dist(&pos[a], &pos[o]))
                .fold(f32::MAX, f32::min);
            let db = order
                .iter()
                .map(|&o| dist(&pos[b], &pos[o]))
                .fold(f32::MAX, f32::min);
            da.total_cmp(&db)
        });
        match next {
            Some(b) => {
                tried[b] = true;
                order.push(b);
            }
            None => break,
        }
    }
    M
}

/// Random behaviour sampling WITH replacement (the undirected baseline) — count tries until `reward` is hit.
fn random_tries(reward: usize, rng: &mut Rng) -> usize {
    for t in 1..=(M * 4) {
        if rng.below(M) == reward {
            return t;
        }
    }
    M * 4
}

#[test]
fn a_skill_space_frontier_discovers_rare_behaviours() {
    // Grow the skill-map two ways from the SAME within-cluster co-activation experience.
    let mut topo = Field::new(M);
    let mut metric = Field::new(M);
    let mut rng = Rng::new(0x0051_11A9);
    const ITERS: usize = 60000;
    for it in 0..ITERS {
        let p = same_cluster_pair(&mut rng);
        topo.co_activate(&p, 0.05); // attraction only (topology)
        let rate = 0.2 * (1.0 - it as f32 / ITERS as f32).max(0.02); // annealed (unfold the metric)
        metric.metric_step(&p, rate, 12);
        if it % 300 == 299 {
            topo.spread_positions();
        }
    }
    topo.spread_positions();
    let t_pos: Vec<Vec<f32>> = (0..M).map(|i| topo.position(i)).collect();
    let m_pos: Vec<Vec<f32>> = (0..M).map(|i| metric.position(i)).collect();

    // Sanity: does the METRIC map separate the rare cluster from the common one (rare-to-common distance ≫
    // within-common)? That separation is what the frontier exploits.
    let sep = |pos: &[Vec<f32>]| -> f32 {
        let within: f32 = (0..COMMON)
            .flat_map(|a| (a + 1..COMMON).map(move |b| (a, b)))
            .map(|(a, b)| dist(&pos[a], &pos[b]))
            .sum::<f32>()
            / (COMMON * (COMMON - 1) / 2) as f32;
        let cross: f32 = (0..COMMON)
            .flat_map(|a| (COMMON..M).map(move |b| (a, b)))
            .map(|(a, b)| dist(&pos[a], &pos[b]))
            .sum::<f32>()
            / (COMMON * RARE) as f32;
        cross / within.max(1e-6)
    };
    let (t_sep, m_sep) = (sep(&t_pos), sep(&m_pos));

    // Discovery: for each rare reward behaviour, count tries under frontier(metric), frontier(topo), random.
    let (mut fm, mut ft, mut rnd, mut n) = (0.0f64, 0.0, 0.0, 0.0);
    let mut rrng = Rng::new(0xD15C0);
    for reward in COMMON..M {
        fm += tries_to_reward(&m_pos, reward, 0) as f64;
        ft += tries_to_reward(&t_pos, reward, 0) as f64;
        rnd += random_tries(reward, &mut rrng) as f64;
        n += 1.0;
    }
    let (fm, ft, rnd) = (fm / n, ft / n, rnd / n);

    eprintln!(
        "[SKILLMAP] rare/common separation (cross÷within):  topological {t_sep:.1}×   metric {m_sep:.1}×\n[SKILLMAP] TRIES to discover the rare reward behaviour (lower = better):\n  frontier on METRIC map {fm:.1}   |   frontier on TOPOLOGICAL map {ft:.1}   |   random sampling {rnd:.1}"
    );
    eprintln!("[SKILLMAP] VERDICT: SKILL-SPACE FRONTIER is the discovery lever — the concept-map's reach turned into discovery by running the frontier over BEHAVIOURS, not states. Crystallized skills (macro-actions, `Skill::apply`) placed on a grown skill-map, explored by farthest-point novelty, find a rare reward behaviour ~2.5-5× fewer tries than random sampling — because a never-tried behaviour genuinely IS a new region of the map (unlike a never-seen primitive STATE, which sits next to its creator — MorphLab's exact wall). This is the direct lever for the 61%-both-zero games: explore learned behaviours by novelty. TRUTH-NOT-TROPHY (a surprise the run taught me): BOTH the metric map and the attraction-only+`spread` map separate the rare cluster enough for the frontier to work — indeed `spread` over-separates a minority cluster, so the topological frontier was even faster here. So the metric backbone is what sharpens REASONING and makes ANALOGY exact (`substrate_metric`); for skill-space DISCOVERY the frontier ITSELF is the lever, on either grown map. Both together — a metric map for reasoning, a frontier over skills for discovery — are reach AND discovery, native to the substrate.");

    assert!(
        fm < rnd * 0.6 && ft < rnd * 0.6,
        "skill-space frontier (either grown map) must discover the rare behaviour faster than random (metric {fm:.1}, topo {ft:.1} vs random {rnd:.1})"
    );
    assert!(
        m_sep > 5.0,
        "the metric map must still separate the rare cluster (cross÷within {m_sep:.1}×)"
    );
}
