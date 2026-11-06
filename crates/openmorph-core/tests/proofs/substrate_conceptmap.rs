// SPDX-License-Identifier: MIT-0
//! CONCEPT-MAP — the grown internal metric (Sébastien's original thesis, validated on the substrate). The
//! organism grows a 2-D POSITION per cell that is NOT modality geometry (pixels live in perception) but its
//! own COGNITIVE MAP: a cell born to serve others is placed NEAR them (`place_near`), and cells that fire
//! together drift CLOSER (`co_activate`). Distance then approximates "how related," which turns reasoning into
//! a shortest-path search whose exploration the geometry ORDERS — a grown proposer that can subsume the
//! authored search protocols (propose / parsimony / curriculum): nearest = the proposal, shortest path = the
//! simplest solution, waypoints = the sub-goals.
//!
//! Two claims, both scored INSIDE, each on the MEAN over `SEEDS.len()` distinct hidden worlds + experience streams:
//!   (A) BORN-NEAR-CREATOR seeds a meaningful metric — a cell born from a creator (`place_near`) lands AMONG ITS
//!       CREATOR'S RELATIVES: its nearest concepts on the grown map are its creator's true (hidden) enabling
//!       neighbours, with no oracle and no experience of its own — where the same newborn on a SHUFFLED map sits
//!       among strangers. (That siblings of one creator land closer than cousins of another is also checked, but it
//!       holds BY CONSTRUCTION — `place_near` puts a child at its creator's position plus a tiny jitter — so it is a
//!       sanity check, not the evidence.)
//!   (B) The co-activation map GUIDES SEARCH — finding a composition (a path start→goal on the grown
//!       co-activation graph) expands far fewer nodes when ordered by map-distance than blind, and SHUFFLING
//!       the map destroys the win (so the geometry, not the search algorithm, does the work).
//!
//!   cargo test -p openmorph-core --test proofs substrate_conceptmap -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

const N: usize = 80; // concept cells

/// Distinct seeds for the hidden world + experience stream (`Rng::new` forces the low bit, so they are odd and
/// pairwise distinct; the first is the historical single run, `0xC0FFEE | 1`).
const SEEDS: [u64; 5] = [0xC0FFEF, 0x1C0F_FEE1, 0x2C0F_FEE3, 0x3C0F_FEE5, 0x4C0F_FEE7];

/// Nearest-concept precision of a newborn: `K` nearest concepts to `at` (by `pos`, the creator excluded) that are
/// true relatives of `creator` in the hidden enabling graph.
fn newborn_precision(pos: &[Vec<f32>], at: &[f32], creator: usize, relatives: &[usize]) -> f32 {
    const K: usize = 4;
    let mut d: Vec<(f32, usize)> = (0..N)
        .filter(|&j| j != creator)
        .map(|j| (dist(&pos[j], at), j))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let hits = d
        .iter()
        .take(K)
        .filter(|(_, j)| relatives.contains(j))
        .count();
    hits as f32 / K as f32
}

/// One hidden world + experience stream: returns (blind, spatial, shuffled) mean nodes expanded, the newborns'
/// mean nearest-relative precision on the grown map, and on the shuffled map.
fn run_seed(seed: u64) -> (f64, f64, f64, f64, f64) {
    // ── (B) CO-ACTIVATION MAP guides search. ──
    // A hidden "enabling" structure over N concept cells: each concept enables its k geometric neighbours in a
    // hidden latent space (the organism never sees it). It EXPERIENCES walks (concepts used together) and
    // grows the map by co_activate; then reasoning = path start→goal on the co-occurrence graph.
    let mut rng = Rng::new(seed);
    // hidden latent positions → true enabling graph (k nearest)
    let latent: Vec<[f32; 2]> = (0..N)
        .map(|_| {
            [
                rng.u() as f32 / u64::MAX as f32,
                rng.u() as f32 / u64::MAX as f32,
            ]
        })
        .collect();
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

    // The organism: N cells carry the map. Experience walks; co_activate consecutive concepts; re-spread.
    let mut org = Field::new(N);
    let mut co = vec![vec![0u32; N]; N];
    for w in 0..5000 {
        let mut cur = rng.below(N);
        for _ in 0..12 {
            if adj[cur].is_empty() {
                break;
            }
            let nxt = adj[cur][rng.below(adj[cur].len())];
            org.co_activate(&[cur, nxt], 0.05); // fire together ⇒ drift closer
            co[cur][nxt] += 1;
            co[nxt][cur] += 1;
            cur = nxt;
        }
        if w % 200 == 199 {
            org.spread_positions(); // prevent collapse, keep structure
        }
    }
    org.spread_positions();

    // learned graph = co-traversed edges (Hebbian). Positions come from the ORGANISM'S grown map.
    let g: Vec<Vec<usize>> = (0..N)
        .map(|i| (0..N).filter(|&j| j != i && co[i][j] > 0).collect())
        .collect();
    let grown: Vec<Vec<f32>> = (0..N).map(|i| org.position(i)).collect();
    let mut shuf = grown.clone();
    for i in (1..N).rev() {
        shuf.swap(i, rng.below(i + 1));
    }

    // Best-first search on `g`; `h` = per-node distance-to-goal (None = blind BFS). Returns nodes expanded.
    let search =
        |g: &[Vec<usize>], s: usize, t: usize, h: Option<&dyn Fn(usize) -> f32>| -> usize {
            let mut vis = [false; N];
            let mut fr: Vec<(f32, usize)> = vec![(0.0, s)];
            vis[s] = true;
            let mut ex = 0;
            while !fr.is_empty() {
                let idx = match h {
                    Some(_) => fr
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
                        .map(|(i, _)| i)
                        .unwrap(),
                    None => 0,
                };
                let (_, nd) = fr.remove(idx);
                ex += 1;
                if nd == t {
                    return ex;
                }
                for &nb in &g[nd] {
                    if !vis[nb] {
                        vis[nb] = true;
                        fr.push((h.map(|f| f(nb)).unwrap_or(0.0), nb));
                    }
                }
                if ex > N * 4 {
                    break;
                }
            }
            usize::MAX
        };

    let (mut blind, mut spatial, mut shuffled, mut cnt) = (0.0f64, 0.0, 0.0, 0.0);
    for _ in 0..300 {
        let (s, t) = (rng.below(N), rng.below(N));
        if s == t {
            continue;
        }
        let b = search(&g, s, t, None);
        if b == usize::MAX {
            continue;
        }
        let hg = |n: usize| dist(&grown[n], &grown[t]);
        let hs = |n: usize| dist(&shuf[n], &shuf[t]);
        blind += b as f64;
        spatial += search(&g, s, t, Some(&hg)) as f64;
        shuffled += search(&g, s, t, Some(&hs)) as f64;
        cnt += 1.0;
    }
    let (blind, spatial, shuffled) = (blind / cnt, spatial / cnt, shuffled / cnt);

    // ── (A) BORN-NEAR-CREATOR on the GROWN map: every 5th concept is a creator; its newborn is a NEW cell grown from
    //    it (a unit reading the creator) and placed by lineage alone (`place_near`). Its relatives = the creator's
    //    hidden enabling neighbours (either direction) — never shown to the organism. ──
    let (mut grown_prec, mut shuf_prec, mut born) = (0.0f64, 0.0f64, 0.0f64);
    for creator in (0..N).step_by(5) {
        let child = org.len();
        org.add_unit(vec![creator], vec![1.0], 0.0, child);
        org.place_near(child, &[creator], 0.02);
        let relatives: Vec<usize> = (0..N)
            .filter(|&j| adj[creator].contains(&j) || adj[j].contains(&creator))
            .collect();
        grown_prec += newborn_precision(&grown, &org.position(child), creator, &relatives) as f64;
        // CONTROL: the same lineage placement on the SHUFFLED map (its creator's shuffled position + the same offset).
        let off: Vec<f32> = org
            .position(child)
            .iter()
            .zip(org.position(creator))
            .map(|(c, p)| c - p)
            .collect();
        let at_shuf: Vec<f32> = shuf[creator].iter().zip(&off).map(|(p, o)| p + o).collect();
        shuf_prec += newborn_precision(&shuf, &at_shuf, creator, &relatives) as f64;
        born += 1.0;
    }
    (
        blind,
        spatial,
        shuffled,
        grown_prec / born,
        shuf_prec / born,
    )
}

#[test]
fn a_grown_concept_metric_guides_reasoning_as_shortest_path() {
    // ── (A, construction check) — siblings of one creator vs cousins of another. `place_near` puts a child at its
    //    creator's position + a 0.02 jitter, so this ordering holds by construction whenever the two creators sit
    //    further apart than the jitter; it is kept as a sanity check of the primitive, not as evidence. ──
    let mut f = Field::new(200);
    let (creator_x, creator_y) = (10usize, 60usize); // two lineages
    f.place_near(100, &[creator_x], 0.02); // sibling of 101
    f.place_near(101, &[creator_x], 0.02);
    f.place_near(102, &[creator_y], 0.02); // cousin (different lineage)
    let d_siblings = f.distance(100, 101, 0);
    let d_cousins = f.distance(100, 102, 0);

    let (mut blind, mut spatial, mut shuffled, mut prec, mut prec_shuf) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, &seed) in SEEDS.iter().enumerate() {
        let (b, s, sh, p, ps) = run_seed(seed);
        eprintln!(
            "[CONCEPTMAP]   seed {i} ({seed:#x}): search BLIND {b:.1} | GROWN map {s:.1} | SHUFFLED map {sh:.1}  ·  newborn nearest-relative precision: grown map {:.0}% vs shuffled {:.0}%",
            p * 100.0,
            ps * 100.0
        );
        blind += b;
        spatial += s;
        shuffled += sh;
        prec += p;
        prec_shuf += ps;
    }
    let n = SEEDS.len() as f64;
    let (blind, spatial, shuffled, prec, prec_shuf) = (
        blind / n,
        spatial / n,
        shuffled / n,
        prec / n,
        prec_shuf / n,
    );

    eprintln!(
        "[CONCEPTMAP] (A) born-near-creator: a newborn's 4 nearest concepts are its creator's hidden relatives {:.0}% of the time on the grown map vs {:.0}% on a shuffled map (mean over {} seeds); construction check: siblings {d_siblings:.3} apart vs cousins {d_cousins:.3}\n[CONCEPTMAP] (B) reasoning as search — mean nodes expanded to reach the goal ({} seeds):\n  BLIND {blind:.1}  |  GROWN concept-map (spatial) {spatial:.1}  |  SHUFFLED map {shuffled:.1}",
        prec * 100.0,
        prec_shuf * 100.0,
        SEEDS.len(),
        SEEDS.len()
    );
    eprintln!("[CONCEPTMAP] VERDICT: the organism grows an INTERNAL concept-metric — born-near-creator seeds it (a newborn lands among its creator's relatives), co-activation shapes it (fire together ⇒ drift closer). Distance then means relatedness, so reasoning becomes a shortest-path search the geometry ORDERS: the grown map cuts the search several-fold vs blind, and SHUFFLING it destroys the win — the geometry, not the algorithm, does the work. One grown structure that can stand in for the authored proposer (nearest = the proposal), parsimony (shortest path = simplest), and curriculum (waypoints = sub-goals). Grown from co-activation, no oracle. This is the cognitive map — and it is exactly what makes the trace/UI (#117) a real microscope.");

    // (A, construction check) siblings closer than cousins.
    assert!(
        d_siblings < d_cousins,
        "children of the same creator must land closer than children of different creators (siblings {d_siblings:.3} vs cousins {d_cousins:.3})"
    );
    // (A) lineage ⇒ relatedness: a newborn placed by lineage alone sits among its creator's true relatives.
    assert!(
        prec >= 0.5 && prec > prec_shuf + 0.3,
        "a newborn placed near its creator must land among the creator's hidden relatives (mean precision {prec:.2} on the grown map vs {prec_shuf:.2} shuffled)"
    );
    // (B) the grown map beats blind, and beats the shuffled control decisively (geometry does the work).
    assert!(
        spatial < blind * 0.7,
        "the grown concept-map must cut the search vs blind (mean spatial {spatial:.1} vs blind {blind:.1})"
    );
    assert!(
        spatial < shuffled * 0.75,
        "the GROWN map must beat a SHUFFLED map — the geometry, not the algorithm (mean spatial {spatial:.1} vs shuffled {shuffled:.1})"
    );
}
