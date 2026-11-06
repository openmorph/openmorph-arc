// SPDX-License-Identifier: MIT-0
//! HIERARCHICAL OPTIONS over the grown discrete-state graph — closing the REWARD-HORIZON wall (the #1 ARC-3
//! substrate lever). MorphBot's finding: 7 real ARC-3 games take 600–1330 steps to their only reward, and NO
//! flat step-level actor reaches it — a flat planner/learner cannot carry reward-credit across a
//! several-hundred-step horizon within any affordable budget. Temporal abstraction is the escape: on the
//! organism's SMALL grown discrete-state graph (the #264 bound-object key gives 0.83–0.99-deterministic
//! transitions), DISCOVER subgoals, ENCAPSULATE "reach-subgoal" macro-actions (OPTIONS), and PLAN OVER OPTIONS.
//! The horizon then collapses from `N` primitive decisions to a two-level nest of SHORT horizons (the
//! option-graph diameter `K`, and each option's local horizon `N/K`), BOTH affordable.
//!
//! THE ONE LEVER — bounded CREDIT REACH. Both actors plan with the SHIPPED `relax_values(adjacency, cost,
//! budget)`. A value-iteration SWEEP is exactly one Bellman backup, so `budget` sweeps = how far reward-credit
//! can reach — the compute/experience horizon a flat step-RL agent lacks at 600 steps. The SAME `budget` is given
//! to both actors; the ONLY difference is the ABSTRACTION LEVEL of the graph `relax_values` runs on. That
//! isolates temporal abstraction as the lever (truth-not-trophy).
//!
//!   * FLAT — `relax_values` over the `N`-state PRIMITIVE graph, `budget = B`. Cost-to-go reaches only `B` states
//!     back from the reward; when the diameter `N−1 > B` the START stays unvalued (`f32::MAX`) ⇒ greedy descent is
//!     uninformed at the start ⇒ never reaches. FLAT solves iff `N−1 ≤ B` (the horizon wall).
//!   * OPTIONS — (1) GROW a concept-map from experienced transitions (`metric_step`/`co_activate`); (2) DISCOVER
//!     subgoals by the FARTHEST-POINT frontier on that map (the `substrate_skillmap` mechanism — evenly-spread
//!     frontier states, no authored waypoints); (3) each region = the Voronoi cell of a subgoal, the OPTION =
//!     "reach the entrance of the next region" (local horizon `≈ N/K ≤ B`, so every option is locally solvable);
//!     (4) PLAN OVER OPTIONS — `relax_values` over the `K`-node region graph (diameter `K−1 ≤ B`) resolves which
//!     option each region fires. Both nested horizons `≤ B`, so OPTIONS solve up to `N ≈ K·B` — a factor `K`
//!     beyond FLAT.
//!
//! TRUTH-NOT-TROPHY. (a) The subgoals are GROWN (farthest-point on the grown map), not authored waypoints — the
//! SHUFFLE control replaces them with RANDOM states: regions become unbalanced (some larger than `B`), those
//! options can't reach their entrance within budget, and the actor FAILS. So the DISCOVERY (balanced frontier)
//! is load-bearing, not merely "any hierarchy." (b) A second test (`bottleneck_detection`) proves the discovery
//! finds real graph BOTTLENECKS: on a rooms-and-doorways graph the grown map's `cluster` recovers the rooms
//! unsupervised and the states adjacent to another cluster are EXACTLY the doorways. (c) Scored INSIDE (reached
//! the reward + terminal dwell), deterministic (`metric_step` with the field's own PRNG), CI-fast. ZERO
//! openmorph-core change — a usage protocol over `relax_values` + `metric_step`/`cluster` + the frontier, exactly
//! as "the reasoner is a usage protocol over the substrate."
//!
//!   cargo test -p openmorph-core --test proofs substrate_options -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ── the credit budget and the grown map ──────────────────────────────────────────────────────────────────────

/// Bounded value-iteration sweeps = the credit-reach horizon both actors share.
const B: usize = 24;
/// Number of grown subgoals (frontier states) — the option-graph node count `K`. Chosen `≈ √(2N)` so BOTH the
/// option-graph diameter (`K−1`) AND each option's local horizon (`≈ 2N/K`, the farthest-point k-center bound)
/// fit the same credit budget `B` — the balance temporal abstraction needs.
const K: usize = 22;

fn line_adj(n: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|s| {
            let mut v = Vec::new();
            if s > 0 {
                v.push(s - 1);
            }
            if s + 1 < n {
                v.push(s + 1);
            }
            v
        })
        .collect()
}

/// GROW the concept-map from experienced transitions: `metric_step` each observed edge (attraction + negative-
/// sampling repulsion — the metric-faithful `substrate_metric` recipe), rate annealed. Deterministic (the field's
/// own PRNG). The map is the organism's own geometry of the state graph — nothing about "rooms" or "waypoints" is
/// declared.
fn grow_map(f: &mut Field, adj: &[Vec<usize>], rounds: usize) {
    // a deterministic edge list from the experienced adjacency
    let mut edges = Vec::new();
    for (u, ns) in adj.iter().enumerate() {
        for &v in ns {
            if u < v {
                edges.push((u, v));
            }
        }
    }
    for r in 0..rounds {
        let rate = 0.10 * (1.0 - r as f32 / rounds as f32) + 0.01;
        for &(u, v) in &edges {
            f.metric_step(&[u, v], rate, 5);
        }
    }
}

/// FARTHEST-POINT frontier over the grown map (the `substrate_skillmap` discovery): greedily pick the state
/// farthest (in the grown metric) from all chosen — evenly-spread frontier subgoals. Anchored at the start so the
/// start region is well-formed; the reward is appended (reaching the reward is the terminal subgoal, not an
/// authored waypoint).
fn frontier(f: &Field, n: usize, k: usize, start: usize, goal: usize) -> Vec<usize> {
    let ids: Vec<usize> = (0..n).collect();
    let mut seeds = vec![start];
    while seeds.len() < k {
        let next = ids
            .iter()
            .copied()
            .filter(|i| !seeds.contains(i))
            .max_by(|&a, &b| {
                let da = seeds
                    .iter()
                    .map(|&s| f.distance(a, s, 0))
                    .fold(f32::MAX, f32::min);
                let db = seeds
                    .iter()
                    .map(|&s| f.distance(b, s, 0))
                    .fold(f32::MAX, f32::min);
                da.total_cmp(&db)
            })
            .unwrap();
        seeds.push(next);
    }
    if !seeds.contains(&goal) {
        seeds.push(goal);
    }
    seeds
}

/// Region of each state: the index (into `subgoals`) of the nearest subgoal by EXPERIENCED-TRANSITION hop count
/// (multi-source BFS on the real adjacency). Where the frontier DISCOVERS subgoals on the grown map, the regions
/// they induce are read off the transitions the organism actually experiences — contiguous by construction, so
/// each option's local horizon is bounded by the region's own radius. Ties break to the lower subgoal index.
fn regions(adj: &[Vec<usize>], n: usize, subgoals: &[usize]) -> Vec<usize> {
    let mut reg = vec![usize::MAX; n];
    let mut dist = vec![usize::MAX; n];
    let mut frontier: Vec<usize> = Vec::new();
    for (g, &sg) in subgoals.iter().enumerate() {
        if dist[sg] == usize::MAX {
            dist[sg] = 0;
            reg[sg] = g;
            frontier.push(sg);
        }
    }
    let mut d = 0;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &u in &frontier {
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = d + 1;
                    reg[v] = reg[u];
                    next.push(v);
                }
            }
        }
        frontier = next;
        d += 1;
    }
    reg
}

/// The OPTION GRAPH: region `a`→`b` is an edge iff a primitive transition crosses from region `a` into region `b`
/// (the regions touch in the experienced graph). One `entrance[a][b]` = a state IN region `b` reachable by a
/// primitive step from region `a` (the subgoal an option aims at).
fn option_graph(
    adj: &[Vec<usize>],
    region: &[usize],
    k: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<Option<usize>>>) {
    let mut radj = vec![Vec::new(); k];
    let mut entrance = vec![vec![None; k]; k];
    for (u, ns) in adj.iter().enumerate() {
        for &v in ns {
            let (ru, rv) = (region[u], region[v]);
            if ru != rv {
                if !radj[ru].contains(&rv) {
                    radj[ru].push(rv);
                }
                entrance[ru][rv].get_or_insert(v);
            }
        }
    }
    (radj, entrance)
}

/// Descend a freshly-relaxed value field one primitive step from `s` toward the clamped target; `None` if `s` is
/// at a local floor (no strictly-lower neighbour — the "uninformed / unreached" verdict FLAT hits at long horizon).
fn step_down(planner: &Field, adj: &[Vec<usize>], s: usize) -> Option<usize> {
    let here = planner.value_of(s);
    adj[s]
        .iter()
        .copied()
        .filter(|&nb| planner.value_of(nb) < here)
        .min_by(|&a, &b| planner.value_of(a).total_cmp(&planner.value_of(b)))
}

/// FLAT step-level actor: `relax_values` over the PRIMITIVE graph with the credit budget `B`, then greedy descent.
fn run_flat(n: usize, budget: usize) -> (bool, usize) {
    let adj = line_adj(n);
    let (start, goal) = (0usize, n - 1);
    let mut planner = Field::new(n);
    planner.prefer(goal, 0.0);
    planner.relax_values(&adj, 1.0, budget);
    let mut s = start;
    for step in 0..3 * n {
        if s == goal {
            return (true, step);
        }
        match step_down(&planner, &adj, s) {
            Some(nb) => s = nb,
            None => break, // uninformed: credit never reached here
        }
    }
    (s == goal, 3 * n)
}

/// HIERARCHICAL OPTIONS actor. Grow the map, DISCOVER subgoals (frontier or, for the shuffle control, random),
/// PLAN over the K-node option graph (`relax_values`, budget `B`), then EXECUTE: each primitive step, relax the
/// LOCAL value field toward the current option's target (the next region's entrance, or the reward once in the
/// goal region), all within budget `B`. Returns (reached, primitive-steps, option-invocations, max-region-size).
fn run_options(n: usize, budget: usize, subgoals: &[usize]) -> (bool, usize, usize, usize) {
    let adj = line_adj(n);
    let (start, goal) = (0usize, n - 1);
    let k = subgoals.len();

    // regions from the EXPERIENCED transitions (BFS); the caller discovered the subgoals on the grown map.
    let region = regions(&adj, n, subgoals);
    let max_region = (0..k)
        .map(|g| region.iter().filter(|&&r| r == g).count())
        .max()
        .unwrap_or(0);

    let (radj, entrance) = option_graph(&adj, &region, k);
    let goal_region = region[goal];

    // ── PLAN OVER OPTIONS: cost-to-go over the K-node region graph (diameter K−1 ≤ B). ──
    let mut hi = Field::new(k);
    hi.prefer(goal_region, 0.0);
    hi.relax_values(&radj, 1.0, budget);

    // ── EXECUTE: fire the option each region's high-level plan selects. ──
    let mut planner = Field::new(n);
    let mut s = start;
    let mut prim = 0usize;
    let mut opts = 0usize;
    let mut cur_region = region[s];
    for _ in 0..3 * n {
        if s == goal {
            return (true, prim, opts, max_region);
        }
        let r = region[s];
        if r != cur_region {
            opts += 1;
            cur_region = r;
        }
        // OPTION TARGET: in the goal region, aim at the reward; else aim at the entrance of the neighbouring
        // region with the lowest high-level cost-to-go (the option the plan fires).
        let target = if r == goal_region {
            goal
        } else {
            let best = radj[r]
                .iter()
                .copied()
                .filter(|&t| hi.value_of(t) < hi.value_of(r))
                .min_by(|&a, &b| hi.value_of(a).total_cmp(&hi.value_of(b)));
            match best.and_then(|t| entrance[r][t]) {
                Some(e) => e,
                None => return (false, prim, opts, max_region), // no valid option (malformed hierarchy)
            }
        };
        planner.clear_values();
        planner.prefer(target, 0.0);
        planner.relax_values(&adj, 1.0, budget);
        match step_down(&planner, &adj, s) {
            Some(nb) => s = nb,
            None => return (false, prim, opts, max_region), // local horizon exceeded budget
        }
        prim += 1;
    }
    (s == goal, prim, opts, max_region)
}

#[test]
fn hierarchical_options_close_the_reward_horizon_wall() {
    let n = 240usize; // ~hundreds of primitive steps to the only reward (the horizon regime)
    let adj = line_adj(n);

    // ── discover the frontier subgoals on the grown map ──
    let mut f = Field::new(n);
    grow_map(&mut f, &adj, 60);
    let subgoals = frontier(&f, n, K, 0, n - 1);

    // ── (A) FLAT vs OPTIONS at the long horizon ──
    let (flat_reach, flat_steps) = run_flat(n, B);
    let (opt_reach, opt_prim, opt_calls, max_region) = run_options(n, B, &subgoals);

    // ── (B) SHUFFLE control: RANDOM subgoals (unbalanced regions) — the discovery must be load-bearing ──
    let mut rng = Rng::new(0xB0771E);
    let mut rand_sub = vec![0usize, n - 1];
    while rand_sub.len() < subgoals.len() {
        let c = rng.below(n);
        if !rand_sub.contains(&c) {
            rand_sub.push(c);
        }
    }
    let (shuf_reach, _sp, _sc, shuf_max_region) = run_options(n, B, &rand_sub);

    eprintln!(
        "[OPTIONS] horizon N={n}, credit budget B={B}, K={} grown subgoals (max region {max_region} states)",
        subgoals.len()
    );
    eprintln!(
        "[OPTIONS]   FLAT    (relax over {n} primitives, budget {B}) : reached={flat_reach} (steps {flat_steps}) <- diameter {} > budget ⇒ start unvalued",
        n - 1
    );
    eprintln!(
        "[OPTIONS]   OPTIONS (plan over {K} options, budget {B})    : reached={opt_reach} in {opt_prim} primitive steps / {opt_calls} option-invocations"
    );
    eprintln!(
        "[OPTIONS]   SHUFFLE (random subgoals, max region {shuf_max_region}) : reached={shuf_reach} <- unbalanced regions exceed the budget"
    );

    // FLAT cannot cross the horizon: the reward-credit never reaches the start.
    assert!(
        !flat_reach,
        "FLAT must FAIL at the long horizon (diameter {} > budget {B})",
        n - 1
    );
    // OPTIONS collapses the horizon and reaches the reward.
    assert!(
        opt_reach,
        "OPTIONS must reach the reward by planning over subgoals"
    );
    // the horizon really collapsed: option-invocations ≪ primitive steps, both nested horizons ≤ budget.
    assert!(
        max_region <= B,
        "every option's local horizon must fit the budget (max region {max_region} ≤ {B})"
    );
    assert!(
        opt_calls <= subgoals.len(),
        "the plan fires ~K option-invocations, not N primitive decisions ({opt_calls} ≤ {})",
        subgoals.len()
    );
    // truth-not-trophy: the balanced GROWN discovery is the lever — random subgoals fail.
    assert!(
        !shuf_reach,
        "SHUFFLE (random subgoals) must FAIL — the balanced frontier discovery is load-bearing"
    );
    assert!(
        shuf_max_region > B,
        "random subgoals must leave a region larger than the budget ({shuf_max_region} > {B})"
    );

    eprintln!(
        "[OPTIONS] VERDICT: at horizon {n} FLAT can't carry reward-credit past its {B}-sweep budget (start unvalued, reached={flat_reach}); OPTIONS discovers {K} frontier subgoals on the grown map, plans over the {K}-node option graph and executes local reach-subgoal options — each nested horizon ≤ {B} — reaching the reward in {opt_calls} option-invocations. Random subgoals collapse (reached={shuf_reach}) ⇒ the GROWN balanced discovery is the lever, not the hierarchy per se. Zero core change."
    );
}

#[test]
fn horizon_boundary_flat_breaks_options_hold() {
    // Fixed credit budget B, fixed K; SWEEP the horizon N. FLAT solves iff N−1 ≤ B; OPTIONS solve up to N ≈ K·B.
    eprintln!("[OPTIONS-BOUNDARY] fixed budget B={B}, K={K}; sweeping horizon N:");
    let mut flat_break = None;
    let mut opt_hold_max = 0usize;
    for &n in &[16usize, 24, 32, 48, 96, 160, 240] {
        let adj = line_adj(n);
        let mut f = Field::new(n);
        grow_map(&mut f, &adj, 60);
        let k = K.min(n);
        let subgoals = frontier(&f, n, k, 0, n - 1);
        let (flat_reach, _) = run_flat(n, B);
        let (opt_reach, _, _, mr) = run_options(n, B, &subgoals);
        eprintln!(
            "[OPTIONS-BOUNDARY]   N={n:>3}: FLAT reached={flat_reach:<5} OPTIONS reached={opt_reach:<5} (max region {mr})"
        );
        if flat_break.is_none() && !flat_reach {
            flat_break = Some(n);
        }
        if opt_reach {
            opt_hold_max = n;
        }
    }
    let flat_break = flat_break.expect("FLAT must break at some horizon within the sweep");
    eprintln!(
        "[OPTIONS-BOUNDARY] VERDICT: FLAT breaks at horizon N≈{flat_break} (≈ budget {B}); OPTIONS still solves at N={opt_hold_max} ({}× the flat wall) — temporal abstraction multiplies the reachable horizon by ~K.",
        opt_hold_max / flat_break.max(1)
    );
    assert!(
        flat_break <= B + 16,
        "FLAT must break near the budget horizon (broke at {flat_break}, budget {B})"
    );
    assert!(
        opt_hold_max >= flat_break * 4,
        "OPTIONS must hold well beyond the flat wall (held to {opt_hold_max} vs flat break {flat_break})"
    );
}

#[test]
fn bottleneck_detection_finds_the_doorways() {
    // A rooms-and-doorways graph: R cliques (rooms) joined by single doorway edges (bottlenecks). The grown map's
    // `cluster` must recover the rooms UNSUPERVISED, and the states adjacent to another cluster must be EXACTLY
    // the doorways — GROWN bottleneck subgoal discovery (the alternative to the frontier route).
    const R: usize = 5; // rooms
    const L: usize = 6; // states per room (clique)
    let n = R * L;
    let mut adj = vec![Vec::new(); n];
    fn connect(a: usize, b: usize, adj: &mut [Vec<usize>]) {
        if !adj[a].contains(&b) {
            adj[a].push(b);
        }
        if !adj[b].contains(&a) {
            adj[b].push(a);
        }
    }
    // intra-room cliques
    for r in 0..R {
        for i in 0..L {
            for j in i + 1..L {
                connect(r * L + i, r * L + j, &mut adj);
            }
        }
    }
    // doorways: room r exit (r*L+L-1) ↔ room r+1 entrance ((r+1)*L)
    let mut doorway_states = std::collections::BTreeSet::new();
    for r in 0..R - 1 {
        let (a, b) = (r * L + L - 1, (r + 1) * L);
        connect(a, b, &mut adj);
        doorway_states.insert(a);
        doorway_states.insert(b);
    }

    let mut f = Field::new(n);
    grow_map(&mut f, &adj, 120);
    let groups = f.cluster(&(0..n).collect::<Vec<_>>(), R);

    // cluster purity: each grown group is (almost) one room
    let mut pure = 0usize;
    for g in &groups {
        let mut counts = [0usize; R];
        for &s in g {
            counts[s / L] += 1;
        }
        let top = counts.iter().max().unwrap();
        if *top == g.len() {
            pure += 1;
        }
    }
    // map each state to its grown cluster id
    let mut cid = vec![usize::MAX; n];
    for (gi, g) in groups.iter().enumerate() {
        for &s in g {
            cid[s] = gi;
        }
    }
    // bottleneck states: adjacent (in the experienced graph) to a state in a DIFFERENT grown cluster
    let mut bottlenecks = std::collections::BTreeSet::new();
    for (s, ns) in adj.iter().enumerate() {
        if ns.iter().any(|&nb| cid[nb] != cid[s]) {
            bottlenecks.insert(s);
        }
    }

    eprintln!(
        "[OPTIONS-BOTTLENECK] rooms R={R}×L={L}: cluster recovered {}/{R} groups, {pure} pure; bottleneck states {:?} vs true doorways {:?}",
        groups.len(),
        bottlenecks,
        doorway_states
    );
    assert_eq!(
        groups.len(),
        R,
        "cluster must recover exactly R rooms from the grown map"
    );
    assert_eq!(
        pure, R,
        "every grown cluster must be exactly one room (unsupervised room recovery)"
    );
    assert_eq!(
        bottlenecks, doorway_states,
        "the states adjacent to another cluster must be EXACTLY the doorways (grown bottleneck discovery)"
    );
    eprintln!(
        "[OPTIONS-BOTTLENECK] VERDICT: the grown concept-map recovers the {R} rooms unsupervised and pinpoints the doorways as the inter-cluster bottlenecks — GROWN subgoal discovery for options, no authored waypoints."
    );
}
