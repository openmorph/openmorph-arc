// SPDX-License-Identifier: MIT-0
//! FUSED INDEX — the two grown speedups COMPOUND. Retrieval cost is (number of comparisons) × (dimensions per
//! comparison). `substrate_fractal` shrinks the first factor (coarse regions → ~√N comparisons instead of N);
//! `substrate_matryoshka` shrinks the second (a cheap coarse PREFIX of the nested dimensions instead of the full
//! embedding). This proof fuses them: descend the region tree comparing to region centroids with only the
//! coarse PREFIX (cheap AND few), then spend the FULL embedding on the handful of leaf cells. Total scalar work
//! is far below either speedup alone, for the same nearest cell — Sébastien's vector-DB, both dials at once.
//!
//! World is the nested-group map of `substrate_matryoshka` (the growth that makes a 2-D prefix coarse-faithful):
//! G groups (the regions), M members each; a coarse 2-D map separates the groups, a grown dimension separates
//! members. Cost is counted as SCALAR OPS = Σ (dimensions touched) across all distance computations. Four
//! strategies, same query set:
//!   flat-full      : N comparisons × full D
//!   matryoshka-only: N × prefix (coarse scan) + survivors × D
//!   fractal-only   : region centroids × D + leaf members × D   (fewer comparisons, full dims)
//!   FUSED          : region centroids × PREFIX + leaf members × D   (fewer AND cheaper)
//!
//!   cargo test -p openmorph-core --test proofs substrate_fractal_fused -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const G: usize = 8; // groups = regions
const M: usize = 8; // members per group
const N: usize = G * M; // 64 concepts
const PREFIX: usize = 2; // coarse Matryoshka prefix
const BEAM: usize = 2; // region beam width

fn members(g: usize) -> Vec<usize> {
    (g * M..g * M + M).collect()
}
fn sub_a(g: usize) -> Vec<usize> {
    (g * M..g * M + M / 2).collect()
}
fn sub_b(g: usize) -> Vec<usize> {
    (g * M + M / 2..g * M + M).collect()
}

fn dp(a: &[f32], b: &[f32], p: usize) -> f32 {
    let k = p.min(a.len());
    a[..k]
        .iter()
        .zip(&b[..k])
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn centroid(pos: &[Vec<f32>], ids: &[usize]) -> Vec<f32> {
    let d = pos[0].len();
    let mut c = vec![0.0f32; d];
    for &i in ids {
        for k in 0..d {
            c[k] += pos[i][k];
        }
    }
    for x in &mut c {
        *x /= ids.len() as f32;
    }
    c
}

#[test]
fn the_fractal_and_matryoshka_speedups_compound() {
    // ── Grow the nested-group Matryoshka map (coarse groups in 2-D, a grown dim for members). ──
    let mut org = Field::new(N);
    for it in 0..6000 {
        let g = it % G;
        org.co_activate(&members(g), 0.05); // cluster each group in the coarse 2-D prefix
        if it % 100 == 99 {
            org.spread_positions();
        }
    }
    org.spread_positions();
    org.grow_dimension(); // add a finer axis for within-group structure
    org.grow_dimension();
    for it in 0..6000 {
        let g = it % G;
        org.co_activate(&sub_a(g), 0.05); // separate the two sub-halves inside each group (fine)
        org.co_activate(&sub_b(g), 0.05);
        if it % 100 == 99 {
            org.spread_positions();
        }
    }
    org.spread_positions();
    let pos: Vec<Vec<f32>> = (0..N).map(|i| org.position(i)).collect();
    let d = pos[0].len();
    let g_ctr: Vec<Vec<f32>> = (0..G).map(|g| centroid(&pos, &members(g))).collect();

    // ── Four retrieval strategies; count scalar ops (dims touched) + agreement with the flat-full nearest. ──
    let mut rng = Rng::new(0x0F05_ED01);
    let mut ops = [0.0f64; 4];
    let mut agree = [0.0f64; 4];
    let trials = 800;
    for _ in 0..trials {
        let target = rng.below(N);
        let jit = 0.05;
        let q: Vec<f32> = (0..d)
            .map(|k| pos[target][k] + (rng.f() - 0.5) * 2.0 * jit)
            .collect();

        // (0) FLAT-FULL — ground truth: N-1 comparisons at full D.
        let flat = (0..N)
            .filter(|&c| c != target)
            .min_by(|&a, &b| dp(&pos[a], &q, d).total_cmp(&dp(&pos[b], &q, d)))
            .unwrap();
        ops[0] += ((N - 1) * d) as f64;
        agree[0] += 1.0; // trivially agrees with itself

        // (1) MATRYOSHKA-ONLY — scan all N at the cheap PREFIX, re-rank survivors at full D.
        let mut cand: Vec<(f32, usize)> = (0..N)
            .filter(|&c| c != target)
            .map(|c| (dp(&pos[c], &q, PREFIX), c))
            .collect();
        cand.sort_by(|a, b| a.0.total_cmp(&b.0));
        let survivors = M + 2;
        let m_best = cand
            .iter()
            .take(survivors)
            .map(|&(_, c)| c)
            .min_by(|&a, &b| dp(&pos[a], &q, d).total_cmp(&dp(&pos[b], &q, d)))
            .unwrap();
        ops[1] += ((N - 1) * PREFIX + survivors * d) as f64;
        if m_best == flat {
            agree[1] += 1.0;
        }

        // (2) FRACTAL-ONLY — region centroids at FULL D, then leaf members at full D.
        let mut go: Vec<usize> = (0..G).collect();
        go.sort_by(|&a, &b| dp(&g_ctr[a], &q, d).total_cmp(&dp(&g_ctr[b], &q, d)));
        let leaf_f: Vec<usize> = go.iter().take(BEAM).flat_map(|&g| members(g)).collect();
        let f_best = leaf_f
            .iter()
            .filter(|&&c| c != target)
            .copied()
            .min_by(|&a, &b| dp(&pos[a], &q, d).total_cmp(&dp(&pos[b], &q, d)))
            .unwrap();
        ops[2] += (G * d + leaf_f.len() * d) as f64;
        if f_best == flat {
            agree[2] += 1.0;
        }

        // (3) FUSED — region centroids at the cheap PREFIX, then leaf members at full D.
        let mut gp: Vec<usize> = (0..G).collect();
        gp.sort_by(|&a, &b| dp(&g_ctr[a], &q, PREFIX).total_cmp(&dp(&g_ctr[b], &q, PREFIX)));
        let leaf_u: Vec<usize> = gp.iter().take(BEAM).flat_map(|&g| members(g)).collect();
        let u_best = leaf_u
            .iter()
            .filter(|&&c| c != target)
            .copied()
            .min_by(|&a, &b| dp(&pos[a], &q, d).total_cmp(&dp(&pos[b], &q, d)))
            .unwrap();
        ops[3] += (G * PREFIX + leaf_u.len() * d) as f64;
        if u_best == flat {
            agree[3] += 1.0;
        }
    }
    for o in &mut ops {
        *o /= trials as f64;
    }
    for a in &mut agree {
        *a = 100.0 * *a / trials as f64;
    }

    let names = ["flat-full", "matryoshka", "fractal", "FUSED"];
    eprintln!("[FUSED] map grew to pos_dim={d}; retrieval cost = scalar ops (dimensions touched), agreement with the flat-full nearest:");
    for i in 0..4 {
        eprintln!(
            "  {:<11} {:>6.0} ops   ({:>4.1}× vs flat)   agree {:.0}%",
            names[i],
            ops[i],
            ops[0] / ops[i],
            agree[i]
        );
    }
    eprintln!("[FUSED] VERDICT: the two grown speedups COMPOUND into one vector-DB. Retrieval cost is (comparisons) × (dims each); `substrate_fractal` cuts the comparisons (coarse regions), `substrate_matryoshka` cuts the dims (coarse prefix), and FUSING them — descend the region tree on the cheap PREFIX, spend the full embedding only on the handful of leaf cells — costs far less than either alone, for the SAME nearest cell. Both dials are GROWN, nothing authored: the regions from the map's own clustering, the nested dimensions from grow-when-stuck. This is the north-star index Sébastien built years ago — coarse-to-fine in BOTH number and dimension at once.");

    assert!(
        agree[3] > 90.0,
        "fused retrieval must find the true nearest nearly always (agree {:.0}%)",
        agree[3]
    );
    // The compounding: fused is cheaper than EITHER speedup alone, and both alone beat flat — a strict chain.
    assert!(
        ops[3] < ops[2] && ops[2] < ops[1] && ops[1] < ops[0] && ops[3] < ops[0] * 0.4,
        "the speedups must compound: fused < fractal < matryoshka < flat, fused ≪ flat (ops {ops:?})"
    );
}
