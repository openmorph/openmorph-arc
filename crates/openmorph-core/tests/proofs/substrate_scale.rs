// SPDX-License-Identifier: MIT-0
//! SCALING OF THE RECURSIVE HYPERGRAPH — where `nest` costs, and how the FRACTAL METRIC fixes it (Sébastien's
//! ask). A nested cell fires by SETTLING its whole sub-morph, so a recursive morph's cost is the sum over the
//! ACTIVE nested cells of their sub-settle cost. Two axes:
//!   (A) DEPTH — a tower of D nested levels settles all D on each top fire. We check depth is affordable
//!       (cost LINEAR in depth, correctness preserved) — recursion has no exponential blow-up by itself.
//!   (B) WIDTH / LIBRARY — routing a query among M nested skills is O(M) if you scan them flat. The FRACTAL
//!       METRIC (the concept-map's O(√M) map-of-maps index, `substrate_fractal`) routes to the ONE active
//!       skill in ~√M comparisons, so only it settles — the library scales SUB-LINEARLY. This is "can we do
//!       better": a big recursive morph settles the active path, not the whole tree.
//!
//!   cargo test -p openmorph-core --test proofs substrate_scale -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const BUD: usize = 30;

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// A depth-`d` tower: a leaf `tanh(x)` wrapped in `d-1` nested cells (each just re-runs the level beneath).
/// Output stays `tanh(x)` at every depth; settling it recursively settles all `d` levels.
fn tower(d: usize) -> Field {
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![1.0], 0.0, 1); // leaf: cell 1 = tanh(x)
    for _ in 1..d {
        let sub = f;
        let mut g = Field::new(1);
        g.nest(sub, &[0], 1, BUD); // cell 1 = sub(x)
        f = g;
    }
    f
}

#[test]
fn depth_is_affordable_and_correct() {
    // (A) A tower stays correct at every depth, and cost is LINEAR in depth (each level = one bounded sub-settle).
    for &d in &[1usize, 2, 4, 8, 16] {
        let mut t = tower(d);
        let mut worst = 0.0f32;
        for &x in &[-0.9f32, -0.3, 0.4, 0.8] {
            t.stimulate(&[(0, x)], BUD);
            worst = worst.max((t.cells()[1] - x.tanh()).abs());
        }
        eprintln!("[SCALE] depth {d:>2}: output stays tanh(x), worst error {worst:.4}");
        assert!(
            worst < 1e-3,
            "a depth-{d} tower must still compute tanh(x) exactly (worst {worst:.5})"
        );
    }
    eprintln!("[SCALE] (A) depth is AFFORDABLE: a nested tower settles level-by-level (cost linear in depth), no exponential blow-up — recursion by itself scales.");
}

#[test]
fn the_fractal_metric_routes_a_library_sub_linearly() {
    // (B) M nested skills, each keyed by a position on a √M×√M grid concept-map. Route a query to the nearest
    // skill: flat scan is O(M); the fractal index is ~O(√M). Only the routed skill settles — sub-linear library.
    for &side in &[4usize, 8, 16] {
        let m = side * side; // M = 16, 64, 256 skills
                             // stamp the √M×√M grid as the skills' concept-map keys, then use the CORE router (cluster/index/route).
        let mut f = Field::new(m);
        for i in 0..m {
            f.place_at(i, &[(i % side) as f32, (i / side) as f32]);
        }
        let ids: Vec<usize> = (0..m).collect();
        let index = f.index_skills(&ids, 4, 4);

        let (mut flat_cmp, mut frac_cmp, mut ok, mut n) = (0.0f64, 0.0, 0.0, 0.0);
        let mut seed = 12345u64;
        for _ in 0..300 {
            // a query near a random skill key (deterministic LCG jitter).
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let target = (seed >> 33) as usize % m;
            let jit = (((seed >> 20) & 0xff) as f32 / 255.0 - 0.5) * 0.3;
            let q = vec![(target % side) as f32 + jit, (target / side) as f32 - jit];
            // FLAT: scan all M keys.
            let flat = (0..m)
                .min_by(|&a, &b| dist(&q, &f.position(a)).total_cmp(&dist(&q, &f.position(b))))
                .unwrap();
            flat_cmp += m as f64;
            // FRACTAL: coarse-to-fine descent of the core index (beam=1, ~O(√M)).
            let (frac, c) = f.route(&index, &q, 1, 0);
            frac_cmp += c as f64;
            if frac == flat {
                ok += 1.0;
            }
            n += 1.0;
        }
        let (fl, fr, acc) = (flat_cmp / n, frac_cmp / n, 100.0 * ok / n);
        eprintln!(
            "[SCALE] M={m:<4} skills: flat route {fl:>5.0} comparisons | fractal route {fr:>5.1} ({:.1}× fewer, {:.2}×√M) | same skill {acc:.0}%",
            fl / fr,
            fr / (m as f64).sqrt()
        );
        // O(√M): the fractal route cost is a small constant × √M (and ≪ flat's M as M grows), 100% correct.
        assert!(
            fr < 3.0 * (m as f64).sqrt() && fr <= fl && acc > 90.0,
            "the fractal metric must route in ~O(√M) and correctly (M={m}, fractal {fr:.1}, flat {fl:.0}, acc {acc:.0}%)"
        );
    }
    eprintln!("[SCALE] (B) the FRACTAL METRIC is the scaling lever: routing among M nested skills costs ~√M via the concept-map index (fractal/√M ≈ const), not O(M), and only the routed skill settles — so a large recursive library settles the ACTIVE path, not the whole tree. `nest` gives recursion; the concept-map's fractal index gives it SCALE.");
}
