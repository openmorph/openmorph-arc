// SPDX-License-Identifier: MIT-0
//! FEW-SHOT SYSTEMATIC GENERALIZATION — a relation learned from ONE example applies correctly EVERYWHERE. This
//! is the hallmark of intelligence (systematicity: know `right` for one concept, know it for all), and on a
//! METRIC-faithful cognitive map it falls out for free: a relation is a consistent DISPLACEMENT vector, so a
//! single (A, right(A)) pair gives the whole vector, and `right(C) = nearest(C + vector)` for every C. This is
//! why `metric_step` matters beyond analogy — the isotropic metric makes ONE example enough; an attraction-only
//! (topological) map has inconsistent displacements, so a single example is noise and generalization fails
//! (it needs to average many examples).
//!
//! Same grid world as `substrate_metric` (the organism only ever co-activates grid
//! neighbours — never a coordinate). Grow the map two ways, learn `right`/`up` from a SINGLE pair, apply to all
//! 64 cells, and measure accuracy. On the metric map one example should already generalize; on the topological
//! map it should not.
//!
//!   cargo test -p openmorph-core --test proofs substrate_fewshot -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const R: usize = 8;
const C: usize = 8;
const N: usize = R * C;

fn cell(r: usize, c: usize) -> usize {
    r * C + c
}
fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
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

/// FEW-SHOT accuracy: learn the relation vector from a SINGLE example `(a, rel(a))`, apply it to every cell,
/// and measure how often `nearest(C + vector)` is the true `rel(C)`. Averaged over all single examples.
fn few_shot(pos: &[Vec<f32>], rel: &dyn Fn(usize, usize) -> Option<usize>) -> f32 {
    let dims = pos[0].len();
    let (mut acc_sum, mut ex) = (0.0f32, 0.0f32);
    for ar in 0..R {
        for ac in 0..C {
            let a = cell(ar, ac);
            let Some(b) = rel(ar, ac) else { continue }; // the ONE example
            let rv: Vec<f32> = (0..dims).map(|k| pos[b][k] - pos[a][k]).collect();
            // apply this single-example vector to EVERY cell.
            let (mut ok, mut tot) = (0.0f32, 0.0);
            for cr in 0..R {
                for cc in 0..C {
                    let c0 = cell(cr, cc);
                    let Some(d) = rel(cr, cc) else { continue };
                    let pred: Vec<f32> = (0..dims).map(|k| pos[c0][k] + rv[k]).collect();
                    let nearest = (0..N)
                        .filter(|&x| x != c0)
                        .min_by(|&a, &b| dist(&pos[a], &pred).total_cmp(&dist(&pos[b], &pred)))
                        .unwrap();
                    if nearest == d {
                        ok += 1.0;
                    }
                    tot += 1.0;
                }
            }
            acc_sum += 100.0 * ok / tot;
            ex += 1.0;
        }
    }
    acc_sum / ex
}

#[test]
fn one_example_generalizes_everywhere_on_a_metric_map() {
    let mut topo = Field::new(N);
    let mut metric = Field::new(N);
    let mut rng = Rng::new(0x00FE_5407);
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

    let right = |r: usize, c: usize| (c + 1 < C).then(|| cell(r, c + 1));
    let up = |r: usize, c: usize| (r + 1 < R).then(|| cell(r + 1, c));

    let (t_r, t_u) = (few_shot(&t_pos, &right), few_shot(&t_pos, &up));
    let (m_r, m_u) = (few_shot(&m_pos, &right), few_shot(&m_pos, &up));

    eprintln!(
        "[FEWSHOT] a relation learned from ONE example, applied to all 64 cells:\n  TOPOLOGICAL map: right {t_r:.0}% up {t_u:.0}%\n  METRIC map     : right {m_r:.0}% up {m_u:.0}%"
    );
    eprintln!("[FEWSHOT] VERDICT: systematic generalization — the hallmark of intelligence — falls out of a metric cognitive map. A relation is a DISPLACEMENT vector, so on the metric-faithful map ONE example `(A, right(A))` gives the whole vector and `right(C)=nearest(C+vector)` holds for EVERY C, from a single shot. On an attraction-only (topological) map the displacement is inconsistent cell-to-cell, so one example is noise and generalization collapses — it would need to average many examples. This is why `metric_step` is the backbone: the isotropic metric turns experience into TRANSFERABLE STRUCTURE, so the organism knows `right` everywhere from knowing it once. king−man+woman=queen from a single example.");

    assert!(
        m_r.min(m_u) > 60.0,
        "one example must generalize on the metric map (right {m_r:.0}% up {m_u:.0}%)"
    );
    assert!(
        m_r.min(m_u) > t_r.max(t_u) + 20.0,
        "the metric map must enable few-shot generalization the topological one cannot (metric {m_r:.0}/{m_u:.0}% vs topo {t_r:.0}/{t_u:.0}%)"
    );
}
