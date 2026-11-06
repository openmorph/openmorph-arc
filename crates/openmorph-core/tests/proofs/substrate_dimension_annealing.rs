// SPDX-License-Identifier: MIT-0
//! DIMENSION ANNEALING — the concept-map grows and shrinks its OWN dimensionality while alive, driven by its
//! own signals. It EXPANDS an axis when concepts are CONFUSED (`grow_dimension` on ambiguity: two nearest
//! memories within a threshold-width) to unfold and separate them, and CONTRACTS the redundant axes with
//! `metabolize_dimension` (drop the lowest-variance axis — the one carrying the least separation). The
//! separation the map grew survives the projection, so it relaxes back to a COMPACT representation: the
//! dimension rises during learning, then falls once the contexts are cleanly separated in fewer axes — while
//! all contexts stay distinct. Expansion and contraction are automatic and only happen while ALIVE (both are
//! no-ops when crystallised). This is the Matryoshka closed both ways: grow on demand, metabolize when redundant.
//!
//!   cargo test -p openmorph-core --test proofs substrate_dimension_annealing -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;

use crate::common::Rng;

const NC: usize = 12;
const NF: usize = NC * 4;
const BLOCK: usize = 150;
fn feats(c: usize) -> [usize; 4] {
    [c * 4, c * 4 + 1, c * 4 + 2, c * 4 + 3]
}
fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}
fn medoid(f: &Field, cells: &[usize]) -> Vec<f32> {
    let d = f.position(0).len();
    (0..d)
        .map(|axis| {
            let mut v: Vec<f32> = cells.iter().map(|&c| f.position(c)[axis]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        })
        .collect()
}
fn purity(route: &[Vec<usize>]) -> f32 {
    route
        .iter()
        .map(|r| {
            (0..NC)
                .map(|c| r.iter().filter(|&&x| x == c).count())
                .max()
                .unwrap() as f32
                / r.len().max(1) as f32
        })
        .sum::<f32>()
        / route.len().max(1) as f32
}

#[test]
fn dimension_expands_on_confusion_then_anneals_back() {
    let mut f = Field::new(NF);
    let mut rng = Rng::new(0x0d15);
    let mut mems: Vec<Vec<f32>> = Vec::new();
    let mut route: Vec<Vec<usize>> = Vec::new();
    let mut cur: Option<usize> = None;
    let mut scatter = 1.0f32;
    let mut peak_dim = 2usize;

    for i in 0..30000usize {
        let c = ((i / BLOCK) * 7) % NC;
        let base = feats(c);
        let drop = (rng.u() % 4) as usize;
        let mut active: Vec<usize> = (0..4).filter(|&k| k != drop).map(|k| base[k]).collect();
        if rng.u().is_multiple_of(5) {
            active.push((rng.u() % NF as u64) as usize);
        }
        f.metric_step(&active, 0.1, 8);
        let q = medoid(&f, &active);
        let thr = 3.0 * scatter;

        // recognize (locality fast-path, then search tracking the two nearest)
        let mut best = usize::MAX;
        let mut bd = f32::MAX;
        if let Some(k) = cur {
            if k < mems.len() && dist(&q, &mems[k]) < thr {
                best = k;
                bd = dist(&q, &mems[k]);
            }
        }
        let (mut d1, mut d2) = (f32::MAX, f32::MAX);
        let mut searched = false;
        if best == usize::MAX {
            searched = true;
            let mut bk = usize::MAX;
            for (k, m) in mems.iter().enumerate() {
                let d = dist(&q, m);
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                    bk = k;
                } else if d < d2 {
                    d2 = d;
                }
            }
            if bk != usize::MAX && d1 < thr {
                best = bk;
                bd = d1;
            }
        }
        // EXPAND on confusion (ambiguous match) — grow an axis, pad stored memories to it
        if searched && best != usize::MAX && d2 - d1 < thr {
            f.grow_dimension();
            let nd = f.position(0).len();
            for m in mems.iter_mut() {
                while m.len() < nd {
                    m.push(0.0);
                }
            }
        }
        let created = best == usize::MAX;
        if created {
            mems.push(q.clone());
            route.push(vec![c]);
            best = mems.len() - 1;
        } else {
            route[best].push(c);
            scatter = 0.995 * scatter + 0.005 * bd;
            for (a, v) in q.iter().enumerate() {
                mems[best][a] += 0.02 * (v - mems[best][a]);
            }
        }
        cur = Some(best);

        // MERGE on create (keep ~one memory per context, so the map doesn't over-grow)
        if created && mems.len() > 1 {
            let mthr = 2.0 * thr;
            'outer: loop {
                for a in 0..mems.len() {
                    for b in (a + 1)..mems.len() {
                        if dist(&mems[a], &mems[b]) < mthr {
                            for ax in 0..mems[a].len() {
                                mems[a][ax] = 0.5 * (mems[a][ax] + mems[b][ax]);
                            }
                            let moved = std::mem::take(&mut route[b]);
                            route[a].extend(moved);
                            mems.remove(b);
                            route.remove(b);
                            if cur == Some(b) {
                                cur = Some(a);
                            } else if cur > Some(b) {
                                cur = cur.map(|x| x - 1);
                            }
                            continue 'outer;
                        }
                    }
                }
                break;
            }
        }

        peak_dim = peak_dim.max(f.position(0).len());

        // CONTRACT — periodically metabolize a redundant axis (infrequent, so a genuinely-needed new axis has
        // time to gain variance and survive). Keep stored memories aligned by dropping the same axis.
        if i.is_multiple_of(300) {
            if let Some(ax) = f.metabolize_dimension(0.2) {
                for m in mems.iter_mut() {
                    if ax < m.len() {
                        m.remove(ax);
                    }
                }
            }
        }
    }

    let final_dim = f.position(0).len();
    let p = purity(&route);
    eprintln!(
        "[ANNEAL] contexts {NC} → {} memories, purity {p:.2}; dimension: peaked at {peak_dim}D during learning, annealed back to {final_dim}D",
        mems.len()
    );
    eprintln!("[ANNEAL] VERDICT: the map GREW its dimension on confusion to separate crowded concepts, then METABOLIZED the redundant axes once they were separated — annealing to a compact representation that keeps every context distinct. Automatic, driven by confusion, only while alive.");

    assert!(
        peak_dim > 3,
        "the map expanded its dimension on confusion (peaked {peak_dim}D)"
    );
    assert!(
        final_dim < peak_dim,
        "it annealed back — contracted redundant axes ({final_dim}D < {peak_dim}D)"
    );
    assert!(final_dim >= 2, "never below the minimum 2D");
    assert!(
        (NC - 2..=NC + 1).contains(&mems.len()),
        "contexts stay separated through the annealing (got {} memories, want ~{NC})",
        mems.len()
    );
    assert!(
        p > 0.88,
        "each memory is essentially one context after annealing (purity {p:.2})"
    );
}
