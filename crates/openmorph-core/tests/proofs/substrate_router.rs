// SPDX-License-Identifier: MIT-0
//! v2.1 · the CORE ROUTER (#64) — the fractal map-of-maps index, promoted from the test harnesses into the
//! substrate. A morph with M nested skills routes a query to the competent skill via its OWN grown concept-map
//! (`cluster` → `index_skills` → `route`), settling only the active path — not an external harness index, not an
//! O(M) broadcast. Two claims, by BEHAVIOR:
//!   (A) SUB-LINEAR CORRECT ROUTING — over a GROWN faithful map of M skills, `route` returns the flat-nearest
//!       skill in ≪ M comparisons (~√M). Reproduces `substrate_scale`/`substrate_fractal` via CORE primitives.
//!   (B) ONLY THE ROUTED SKILL SETTLES — route decides the winner by POSITION (no settle); driving the query
//!       into only the winner's band settles exactly ONE sub-morph (`nested_settles` += 1), vs M for a broadcast.
use openmorph_core::substrate::Field;

use crate::common::Rng;

fn dist_vec(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Grow a `side`×`side` grid into a flat faithful metric map (the validated `metric_step` recipe from
/// `substrate_fractal`), returning the whole Field so the core router reads its LIVE grown positions.
fn grow_grid_field(side: usize, seed: u64) -> Field {
    let n = side * side;
    let mut org = Field::new(n);
    let mut rng = Rng::new(seed);
    let iters = 120000 + n * 300;
    for it in 0..iters {
        let (r, c) = (rng.below(side), rng.below(side));
        let mut nb = Vec::new();
        if r + 1 < side {
            nb.push((r + 1) * side + c);
        }
        if r > 0 {
            nb.push((r - 1) * side + c);
        }
        if c + 1 < side {
            nb.push(r * side + c + 1);
        }
        if c > 0 {
            nb.push(r * side + c - 1);
        }
        let m = nb[rng.below(nb.len())];
        let rate = 0.2 * (1.0 - it as f32 / iters as f32).max(0.02);
        org.metric_step(&[r * side + c, m], rate, 14);
    }
    org
}

/// A trivial banked skill: a sub-morph computing `tanh(x)` (source cell 0 → sink cell 1).
fn skill() -> Field {
    let mut g = Field::new(2);
    g.add_unit(vec![0], vec![1.0], 0.0, 1);
    g
}

#[test]
fn router_routes_sublinearly_and_settles_only_the_winner() {
    // ── (A) SUB-LINEAR CORRECT ROUTING over a GROWN faithful map. ──
    let mut rng = Rng::new(0x_C0DE_0507);
    eprintln!("[ROUTE] coarse-to-fine descent vs flat scan over a grown concept-map:");
    let mut cmp1 = Vec::new(); // beam=1 comparison counts per size (for the scaling check)
    for &side in &[8usize, 16] {
        let n = side * side;
        let f = grow_grid_field(side, 0xF00D_0001 + side as u64);
        let ids: Vec<usize> = (0..n).collect();
        let index = f.index_skills(&ids, 16, 4); // leaf 16, branch 4 (same shape as substrate_fractal)
        let trials = 300;
        // pre-draw queries so beam=1 and beam=2 see the SAME set (fair comparison)
        let queries: Vec<(Vec<f32>, usize)> = (0..trials)
            .map(|_| {
                let c = rng.below(n);
                let q: Vec<f32> = f
                    .position(c)
                    .iter()
                    .map(|&x| x + 0.04 * rng.sym())
                    .collect();
                let flat = (0..n)
                    .min_by(|&a, &b| {
                        dist_vec(&q, &f.position(a)).total_cmp(&dist_vec(&q, &f.position(b)))
                    })
                    .unwrap();
                (q, flat)
            })
            .collect();
        for &beam in &[1usize, 2] {
            let (mut correct, mut total_cmp) = (0usize, 0usize);
            for (q, flat) in &queries {
                let (routed, cmps) = f.route(&index, q, beam, 0);
                if routed == *flat {
                    correct += 1;
                }
                total_cmp += cmps;
            }
            let acc = correct as f32 / trials as f32;
            let avg_cmp = total_cmp as f32 / trials as f32;
            eprintln!(
                "[ROUTE]   N={n:>3} beam={beam}: accuracy {acc:.2}, avg comparisons {avg_cmp:.1}  (flat O(M)={n}, √M={:.0})",
                (n as f32).sqrt()
            );
            assert!(
                avg_cmp < n as f32,
                "routing must be cheaper than a flat scan (N={n}, beam={beam}): {avg_cmp:.1} vs {n}"
            );
            if beam == 1 {
                cmp1.push(avg_cmp);
                assert!(
                    avg_cmp < 0.5 * n as f32,
                    "beam=1 must be strongly sub-linear (N={n}): {avg_cmp:.1} vs flat {n}"
                );
            }
            if beam == 2 {
                assert!(
                    acc > 0.95,
                    "beam=2 must recover nearly all flat-nearest (N={n}): accuracy {acc:.2}"
                );
            }
        }
    }
    // SCALING: N quadruples (64→256) but the comparison count must grow FAR slower than flat's 4× (≈√ ⇒ ~2×).
    let growth = cmp1[1] / cmp1[0];
    eprintln!(
        "[ROUTE]   scaling: comparisons grew {growth:.2}× as N went 4× (flat would be 4.0×, √M would be 2.0×)"
    );
    assert!(
        growth < 2.5,
        "the index must scale sub-linearly: comparisons grew {growth:.2}× vs flat 4× for a 4× library"
    );

    // ── (B) ONLY THE ROUTED SKILL SETTLES — route by position, then drive only the winner's band. ──
    const M: usize = 16;
    let mut h = Field::new(M); // cells 0..M-1 are the M skills' distinct query bands
                               // These M skills are IDENTICAL copies of `skill()`, routed by their DISTINCT per-cell position keys. Behaviour-
                               // seeding (#172, on by default) would correctly collapse identical skills to ONE position — so this test, which
                               // needs distinct position keys per skill, opts out (the documented escape hatch for position-routing on clones).
    h.config_mut().nest_seeds_position = false;
    let mut o = Vec::new(); // the M skill (nested-cell) handles
    for k in 0..M {
        let w = h.len();
        h.ensure_cells(w + 1);
        h.nest(skill(), &[k], w, 30); // skill k reads its OWN band cell k, writes o_k
        o.push(w);
    }
    let index = h.index_skills(&o, 4, 4);

    // route an exact query = skill 7's key → must return skill 7 (its own position is nearest)
    let target = 7usize;
    let q = h.position(o[target]);
    let (routed, _c) = h.route(&index, &q, 2, 0);
    assert_eq!(
        routed, o[target],
        "route must return the queried skill's own cell (o[{target}]={}, got {routed})",
        o[target]
    );

    // settle ONLY the routed winner: write just its band → fire-on-change settles one sub-morph.
    let base = h.nested_settles();
    h.stimulate(&[(target, 0.5)], 30);
    let routed_settles = h.nested_settles() - base;
    let out = h.cells()[o[target]];

    // broadcast control: write EVERY band (fresh input so none are memoized) → all M skills settle.
    let base2 = h.nested_settles();
    let all: Vec<(usize, f32)> = (0..M).map(|k| (k, 0.7)).collect();
    h.stimulate(&all, 30);
    let broadcast_settles = h.nested_settles() - base2;

    eprintln!(
        "[ROUTE] selective settle (M={M}): route→one band ⇒ {routed_settles} sub-settle; broadcast all bands ⇒ {broadcast_settles} sub-settles. routed skill output tanh(0.5)={out:.3}"
    );
    assert_eq!(
        routed_settles, 1,
        "routing must settle EXACTLY the one winning skill, got {routed_settles}"
    );
    assert!(
        (out - 0.5f32.tanh()).abs() < 1e-3,
        "the routed skill must compute its function: tanh(0.5)={:.3}, got {out:.3}",
        0.5f32.tanh()
    );
    assert_eq!(
        broadcast_settles, M as u64,
        "the broadcast control must settle all M skills, got {broadcast_settles}"
    );

    eprintln!("[ROUTE] VERDICT: the CORE ROUTER lands (#64) — the fractal map-of-maps index (`cluster`→`index_skills`→`route`) is now a substrate primitive over the grown concept-map, no longer harness-duplicated. (A) coarse-to-fine descent finds the flat-nearest skill in ~√M comparisons, not O(M). (B) it decides the winner by POSITION with no settle, so driving only the winner's band settles EXACTLY ONE sub-morph (1 vs M broadcast) — a big recursive library settles the active path, not the whole tree. arbitrate = route + local select; this backs MorphLab's harness SkillRouter with a grown, in-substrate router.");
}
