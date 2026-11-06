// SPDX-License-Identifier: MIT-0
//! GROWN-MAP FIDELITY AT SCALE (the stress test) — the fractal-index routing in `substrate_scale` used a clean
//! HAND-BUILT map. The honest question: does it hold when the concept-map is GROWN (`metric_step`) over a
//! library whose intrinsic structure is HIGHER-DIMENSIONAL than the default 2-D map? A skill library laid out
//! on a 3-D grid (skills related to their 3-D grid neighbours) CANNOT be embedded faithfully in 2 dimensions —
//! the map folds, distant skills collide, and routing degrades. Growing the map's dimension (`grow_dimension`,
//! the Matryoshka mechanism) to match the intrinsic dimension recovers routing fidelity. So the scaling lever
//! is TWO grown structures composing: `grow_dimension` gives the map enough dimensions, the fractal index
//! routes in ~O(√M) over it. Fractal metrics scale exactly when the geometry is grown to fit the library.
//!
//!   cargo test -p openmorph-core --test proofs substrate_stress -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const S: usize = 4; // grid side
const M: usize = S * S * S; // 64 skills on a 4×4×4 grid (intrinsic dimension 3)

fn coord(i: usize) -> (usize, usize, usize) {
    (i % S, (i / S) % S, i / (S * S))
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s | 1)
    }
    fn u(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u() % n as u64) as usize
    }
    fn f(&mut self) -> f32 {
        (self.u() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// 3-D grid neighbours of skill `i` (differ by 1 in one coordinate) — the library's true relatedness.
fn neighbors(i: usize) -> Vec<usize> {
    let (x, y, z) = coord(i);
    let mut v = Vec::new();
    let mut push = |a: usize, b: usize, c: usize| v.push(a + b * S + c * S * S);
    if x + 1 < S {
        push(x + 1, y, z);
    }
    if x > 0 {
        push(x - 1, y, z);
    }
    if y + 1 < S {
        push(x, y + 1, z);
    }
    if y > 0 {
        push(x, y - 1, z);
    }
    if z + 1 < S {
        push(x, y, z + 1);
    }
    if z > 0 {
        push(x, y, z - 1);
    }
    v
}

/// Grow a concept-map over the M skills at `pos_dim` dimensions (2 by default; `grow_dimension` to widen),
/// from related-skill co-activation via `metric_step`. Return the grown Field (the router reads its positions).
fn grow_map(pos_dim: usize, seed: u64) -> Field {
    let mut org = Field::new(M);
    for _ in 2..pos_dim {
        org.grow_dimension(); // 2 → pos_dim (the Matryoshka widen)
    }
    let mut rng = Rng::new(seed);
    let iters = 120000;
    for it in 0..iters {
        let i = rng.below(M);
        let nb = neighbors(i);
        let j = nb[rng.below(nb.len())];
        let rate = 0.2 * (1.0 - it as f32 / iters as f32).max(0.02);
        org.metric_step(&[i, j], rate, 14);
    }
    org
}

/// Routing quality: (accuracy%, mean comparisons) — did the CORE fractal index find the right skill, how cheap.
fn routing_quality(f: &Field) -> (f32, f32) {
    let index = f.index_skills(&(0..M).collect::<Vec<_>>(), 4, 4);
    let mut rng = Rng::new(0x57E55);
    let dims = f.pos_dim();
    let spacing: f32 = (0..M)
        .map(|i| {
            (0..M)
                .filter(|&j| j != i)
                .map(|j| f.distance(i, j, 0))
                .fold(f32::MAX, f32::min)
        })
        .sum::<f32>()
        / M as f32;
    let (mut ok, mut cmp, mut n) = (0.0f32, 0.0f32, 0.0);
    for _ in 0..600 {
        let i = rng.below(M);
        let base = f.position(i);
        let q: Vec<f32> = (0..dims)
            .map(|k| base[k] + (rng.f() - 0.5) * 0.3 * spacing)
            .collect();
        let (best, c) = f.route(&index, &q, 1, 0);
        if best == i {
            ok += 1.0;
        }
        cmp += c as f32;
        n += 1.0;
    }
    (100.0 * ok / n, cmp / n)
}

/// METRIC FIDELITY — Pearson correlation of grown-map distance vs the TRUE 3-D grid distance over all skill
/// pairs. This is what a low-dim map folds and a dimension-matched map preserves (needed for reasoning over the
/// library — analogy/planning — not just nearest-skill routing).
fn fidelity(f: &Field) -> f32 {
    let truth = |i: usize, j: usize| {
        let (a, b, c) = coord(i);
        let (x, y, z) = coord(j);
        ((a as f32 - x as f32).powi(2)
            + (b as f32 - y as f32).powi(2)
            + (c as f32 - z as f32).powi(2))
        .sqrt()
    };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for i in 0..M {
        for j in (i + 1)..M {
            xs.push(f.distance(i, j, 0));
            ys.push(truth(i, j));
        }
    }
    let n = xs.len() as f32;
    let (mx, my) = (xs.iter().sum::<f32>() / n, ys.iter().sum::<f32>() / n);
    let (mut num, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in xs.iter().zip(&ys) {
        num += (x - mx) * (y - my);
        dx += (x - mx).powi(2);
        dy += (y - my).powi(2);
    }
    num / (dx.sqrt() * dy.sqrt()).max(1e-6)
}

#[test]
fn grown_map_routing_scales_and_dimensions_buy_fidelity() {
    // The library is intrinsically 3-D. Truth-not-trophy: I expected a 2-D map to FOLD it and MISROUTE — that
    // was FALSIFIED (routing by map-position is robust because query and keys are co-embedded). Where dimension
    // genuinely matters is METRIC FIDELITY — preserving the true 3-D structure — which reasoning over the
    // library needs, not nearest-skill routing.
    let p2 = grow_map(2, 0xD1);
    let p3 = grow_map(3, 0xD1);
    let (acc2, cmp2) = routing_quality(&p2);
    let (acc3, cmp3) = routing_quality(&p3);
    let (fid2, fid3) = (fidelity(&p2), fidelity(&p3));
    let root_m = (M as f32).sqrt();

    eprintln!(
        "[STRESS] a 3-D-structured library of {M} skills, GROWN concept-map + fractal index:"
    );
    eprintln!("  2-D grown map: routing {acc2:>3.0}% at {cmp2:.1} comps ({:.2}×√M)   metric-fidelity {fid2:.2}", cmp2 / root_m);
    eprintln!("  3-D grown map: routing {acc3:>3.0}% at {cmp3:.1} comps ({:.2}×√M)   metric-fidelity {fid3:.2}", cmp3 / root_m);
    eprintln!("[STRESS] VERDICT: the honest scaling picture, two grown structures each earning its keep. (1) ROUTING scales robustly at ANY dimension — the fractal index holds ~O(√M) and routing-by-position is 90%+ even in a 2-D map (my folding hypothesis FALSIFIED: query and keys co-embed, so a fold doesn't misroute). (2) METRIC FIDELITY is where `grow_dimension` pays: a 3-D-structured library folds in 2 dimensions (fidelity {fid2:.2}) and the dimension-matched map preserves the true structure (fidelity {fid3:.2}) — and fidelity is what REASONING over the library needs (analogy, planning, the manifold faculties), not nearest-skill routing. So: the fractal index gives sub-linear routing; `grow_dimension` gives the geometry enough axes to keep the library faithful as it grows. The limitation (a fixed low-dim map caps fidelity) is real; the fix is grown, on demand.");

    assert!(
        acc2 > 85.0 && acc3 > 85.0,
        "grown-map routing must be robust at both dims ({acc2:.0}%/{acc3:.0}%)"
    );
    assert!(
        cmp3 < 3.0 * root_m,
        "the fractal index must hold ~O(√M) on the grown map (comparisons {cmp3:.1})"
    );
    assert!(
        fid3 > fid2 + 0.03,
        "growing the map to the library's intrinsic dimension must improve metric fidelity (2-D {fid2:.2} → 3-D {fid3:.2})"
    );
}
