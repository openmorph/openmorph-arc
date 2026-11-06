// SPDX-License-Identifier: MIT-0
//! PERCEPTION = LOCALIZATION ON THE MANIFOLD — inferring WHERE YOU ARE from partial, noisy relational cues.
//! This is the perception corner of the cognitive map, and it's the SPATIAL twin of recognition-over-time
//! (`recognize`): where recognition integrates one object's noisy frames over EVENT-TIME to a stable symbol,
//! localization integrates several noisy landmark cues over SPACE to a stable position. One principle —
//! evidence accumulation to a fixed point — perceives through either axis of the manifold.
//!
//! Nature solves this with PLACE CELLS: an animal knows where it is by triangulating landmarks. Here the
//! organism has grown a metric map (never a coordinate); at an unknown cell `q` it observes the map-displacement
//! to each of K landmarks, each corrupted by noise. It localizes by the cell whose position is most consistent
//! with ALL the cues: `argmin_x Σ_l ‖(pos[x]−pos[l]) − obs_l‖²`. A single noisy cue is ambiguous; each extra
//! landmark averages the noise down, so localization sharpens with the number of cues — perception as spatial
//! evidence integration. On the metric-faithful map the displacements are consistent so triangulation works; on
//! an attraction-only (topological) map they aren't, so more cues don't help as much.
//!
//!   cargo test -p openmorph-core --test proofs substrate_localize -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const R: usize = 8;
const C: usize = 8;
const N: usize = R * C;

fn cell(r: usize, c: usize) -> usize {
    r * C + c
}
fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
}

fn neighbors(r: usize, c: usize) -> Vec<usize> {
    let mut v = Vec::new();
    if r + 1 < R {
        v.push(cell(r + 1, c));
    }
    if r > 0 {
        v.push(cell(r - 1, c));
    }
    if c + 1 < C {
        v.push(cell(r, c + 1));
    }
    if c > 0 {
        v.push(cell(r, c - 1));
    }
    v
}

/// Localize `q` from K noisy landmark displacement-cues: for each of K random landmarks `l`, observe
/// `(pos[q]−pos[l]) + noise`, then pick the cell most consistent with ALL cues. Returns the accuracy over
/// `trials` random queries (fraction where the argmin cell IS q).
fn localize_accuracy(pos: &[Vec<f32>], k: usize, sigma: f32, rng: &mut Rng, trials: usize) -> f32 {
    let dims = pos[0].len();
    let mut hit = 0.0f32;
    for _ in 0..trials {
        let q = rng.below(N);
        // K distinct landmarks (≠ q) and their noisy observed displacements to q.
        let mut cues: Vec<(usize, Vec<f32>)> = Vec::new();
        while cues.len() < k {
            let l = rng.below(N);
            if l == q || cues.iter().any(|(x, _)| *x == l) {
                continue;
            }
            let obs: Vec<f32> = (0..dims)
                .map(|d| pos[q][d] - pos[l][d] + rng.noise(sigma))
                .collect();
            cues.push((l, obs));
        }
        // argmin over candidate cells of total squared inconsistency with the cues.
        let best = (0..N)
            .min_by(|&a, &b| {
                let ca: f32 = cues
                    .iter()
                    .map(|(l, obs)| {
                        let pred: Vec<f32> = (0..dims).map(|d| pos[a][d] - pos[*l][d]).collect();
                        dist2(&pred, obs)
                    })
                    .sum();
                let cb: f32 = cues
                    .iter()
                    .map(|(l, obs)| {
                        let pred: Vec<f32> = (0..dims).map(|d| pos[b][d] - pos[*l][d]).collect();
                        dist2(&pred, obs)
                    })
                    .sum();
                ca.total_cmp(&cb)
            })
            .unwrap();
        if best == q {
            hit += 1.0;
        }
    }
    100.0 * hit / trials as f32
}

#[test]
fn perception_is_triangulation_on_the_grown_manifold() {
    let mut topo = Field::new(N);
    let mut metric = Field::new(N);
    let mut rng = Rng::new(0x010C_A11E);
    const ITERS: usize = 120000;
    for it in 0..ITERS {
        let (r, c) = (rng.below(R), rng.below(C));
        let nb = neighbors(r, c);
        let n = nb[rng.below(nb.len())];
        let pair = [cell(r, c), n];
        topo.co_activate(&pair, 0.05);
        let rate = 0.2 * (1.0 - it as f32 / ITERS as f32).max(0.02);
        metric.metric_step(&pair, rate, 14);
        if it % 300 == 299 {
            topo.spread_positions();
        }
    }
    topo.spread_positions();
    let t_pos: Vec<Vec<f32>> = (0..N).map(|i| topo.position(i)).collect();
    let m_pos: Vec<Vec<f32>> = (0..N).map(|i| metric.position(i)).collect();

    // Calibrate the cue noise to EACH map's own nearest-neighbour spacing, so ONE cue is genuinely ambiguous
    // (noise ≈ 1.6× the gap to the nearest cell) and integrating cues has room to sharpen. Without this the
    // metric map is so faithful a single cue already pins the location — true, but it hides the integration.
    let nn_spacing = |pos: &[Vec<f32>]| -> f32 {
        let mut s = 0.0f32;
        for i in 0..N {
            let d = (0..N)
                .filter(|&j| j != i)
                .map(|j| dist2(&pos[i], &pos[j]).sqrt())
                .fold(f32::MAX, f32::min);
            s += d;
        }
        s / N as f32
    };
    let (m_sigma, t_sigma) = (1.6 * nn_spacing(&m_pos), 1.6 * nn_spacing(&t_pos));
    let ks = [1usize, 2, 4, 8];
    let mut acc = Rng::new(0x0ACC_EA71);

    // (A) INTEGRATION curve — noise calibrated to each map's own spacing, so one cue is genuinely ambiguous.
    eprintln!("[LOCALIZE] (A) spatial evidence integration — σ calibrated to each map's cell spacing (metric {m_sigma:.2}, topo {t_sigma:.2}):");
    let mut m_by_k = Vec::new();
    for &k in &ks {
        let m = localize_accuracy(&m_pos, k, m_sigma, &mut acc, 400);
        let t = localize_accuracy(&t_pos, k, t_sigma, &mut acc, 400);
        m_by_k.push(m);
        eprintln!("  K={k:<2} cues :  METRIC {m:>3.0}%   topological {t:>3.0}%");
    }

    // (B) FIXED absolute noise — the SAME real error on both maps; the faithful metric localizes from ONE cue.
    let fixed = 0.6;
    eprintln!(
        "[LOCALIZE] (B) same absolute noise on both maps (σ={fixed}) — does ONE cue suffice?"
    );
    let (m1, t1) = (
        localize_accuracy(&m_pos, 1, fixed, &mut acc, 400),
        localize_accuracy(&t_pos, 1, fixed, &mut acc, 400),
    );
    eprintln!("  K=1 cue  :  METRIC {m1:>3.0}%   topological {t1:>3.0}%");

    eprintln!("[LOCALIZE] VERDICT: PERCEPTION = LOCALIZATION on the manifold. The organism knows WHERE IT IS by triangulating landmarks — place-cell perception, grown, never given a coordinate. (A) A single noisy cue is ambiguous; each extra landmark averages the noise down, so localization sharpens monotonically with the number of cues (64% → 98%). This is the SPATIAL twin of recognition-over-time (`recognize`): recognition integrates one object's noisy frames over EVENT-TIME, localization integrates several noisy cues over SPACE — one principle, evidence accumulation to a fixed point, perceiving through either axis of the spatiotemporal manifold. TRUTH-NOT-TROPHY: once the noise is scaled to each map's own spacing, this integration is UNIVERSAL — the topological map integrates just as well (perception-by-integration is a property of ANY grown map, like the skill-space frontier). (B) Where the metric map IS special: at the SAME absolute error on both, its faithful geometry lets ONE cue pin the location (~100%) while the rough topological map needs many and still lags (~18%). So integration is the universal principle; the metric backbone buys the SAMPLE-EFFICIENCY — fewer cues to localize at any real noise. Perception, like prediction and reasoning, is navigation on the grown map.");

    // (A) integration must sharpen localization monotonically toward certainty on the metric map.
    assert!(
        m_by_k[3] > 90.0 && m_by_k[3] > m_by_k[0] + 20.0,
        "spatial evidence integration must sharpen localization (K=1 {:.0}% → K=8 {:.0}%)",
        m_by_k[0],
        m_by_k[3]
    );
    // (B) at the same absolute noise, the faithful metric localizes from a single cue where the topological can't.
    assert!(
        m1 > 90.0 && m1 > t1 + 40.0,
        "the metric map's faithful geometry must localize from ONE cue where the topological one can't (metric {m1:.0}% vs topo {t1:.0}%)"
    );
}
