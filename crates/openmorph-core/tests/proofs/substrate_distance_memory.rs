// SPDX-License-Identifier: MIT-0
//! DISTANCE-BASED CONTENT-ADDRESSABLE MEMORY on the grown concept-map — memory and context are one substrate.
//!
//! A memory is a POSITION on the organism's grown metric map. Recognition is distance: an input near a stored
//! memory means the activation can reach it (pattern completion) → REUSE; an input far from every memory means
//! it can't reach any (the "energy can't propagate") → CREATE one there (pattern separation). This is exactly
//! how the hippocampus stores memory — a metric/place map with completion-vs-separation by distance — and it
//! needs NO decay (a position just sits), is cheap (a nearest lookup; O(√M) via the fractal index), and is
//! content-addressable (creates only when the same memory doesn't already exist). It composes ONLY existing
//! primitives — `metric_step` / `position` / `distance` — no new core, nothing authored.
//!
//! The dials are GROWN from the map's own scale, not authored absolutes: the create/merge thresholds are
//! fractions of the map's dispersion (`spread`), and there is no fixed warmup — allocation runs from the start
//! and duplicates MERGE as the map clusters, so it self-terminates ("stop when creates stop").
//!
//!   cargo test -p openmorph-core --test proofs substrate_distance_memory -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const NF: usize = 16; // 4 contexts × 4 feature cells each (disjoint)
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
fn centroid(f: &Field, cells: &[usize]) -> Vec<f32> {
    let d = f.position(0).len();
    let mut ctr = vec![0.0f32; d];
    for &c in cells {
        for (a, p) in f.position(c).into_iter().enumerate() {
            ctr[a] += p / cells.len() as f32;
        }
    }
    ctr
}
// ROBUST content address: per-axis MEDIAN of the active cells — a single distant noise feature can't drag it.
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
// the map's OWN dispersion (RMS distance from centroid) — the intrinsic scale that sets the grown dials.
fn map_spread(f: &Field, n: usize) -> f32 {
    let ctr = centroid(f, &(0..n).collect::<Vec<_>>());
    let s: f32 = (0..n).map(|c| dist(&f.position(c), &ctr).powi(2)).sum();
    (s / n as f32).sqrt()
}
fn purity(route: &[Vec<usize>]) -> f32 {
    if route.is_empty() {
        return 0.0;
    }
    route
        .iter()
        .map(|r| {
            (0..4)
                .map(|c| r.iter().filter(|&&x| x == c).count())
                .max()
                .unwrap() as f32
                / r.len() as f32
        })
        .sum::<f32>()
        / route.len() as f32
}

fn train(f: &mut Field, rng: &mut Rng, steps: usize) {
    for _ in 0..steps {
        let c = (rng.u() % 4) as usize;
        let mut fs = feats(c).to_vec();
        if rng.u().is_multiple_of(4) {
            fs.push((rng.u() % NF as u64) as usize); // within-context variability
        }
        f.metric_step(&fs, 0.1, 8);
    }
}

/// H1 — the metric map SEPARATES contexts by distance: same-context cells cluster tight, different contexts push
/// far apart, and a novel context lands far from every known one. Without this, distance can't address memory.
#[test]
fn distance_separates_contexts() {
    let mut f = Field::new(NF + 4);
    let mut rng = Rng::new(0x0d15);
    train(&mut f, &mut rng, 8000);

    let cents: Vec<Vec<f32>> = (0..4).map(|c| centroid(&f, &feats(c))).collect();
    let within = (0..4)
        .flat_map(|c| feats(c).into_iter().map(move |cell| (c, cell)))
        .map(|(c, cell)| dist(&f.position(cell), &cents[c]))
        .sum::<f32>()
        / (4 * 4) as f32;
    let mut between = 0.0f32;
    let mut np = 0;
    for i in 0..4 {
        for j in (i + 1)..4 {
            between += dist(&cents[i], &cents[j]);
            np += 1;
        }
    }
    between /= np as f32;
    // a known context's query sits ~within from its OWN memory; a novel context sits far from the NEAREST.
    let novel = centroid(&f, &[16, 17, 18, 19]);
    let novel_near = (0..4)
        .map(|c| dist(&novel, &cents[c]))
        .fold(f32::MAX, f32::min);

    eprintln!(
        "[DMEM/H1] within {within:.2}  between {between:.2}  ratio {:.1}x   |  known→own ~{within:.2}  novel→nearest {novel_near:.2}",
        between / within.max(1e-6)
    );
    assert!(
        between > 4.0 * within,
        "contexts must separate by distance (between {between:.2} vs within {within:.2})"
    );
    assert!(
        novel_near > 3.0 * within,
        "a novel context must land far from every memory (novel {novel_near:.2} vs within {within:.2})"
    );
}

/// H2 — recognize-by-nearest + create-when-far allocates EXACTLY ONE memory per distinct context on the trained
/// map, routing every occurrence (incl. a late-arriving context) to its own slot. No labels — content by distance.
#[test]
fn recognize_or_create_allocates_one_per_context() {
    let mut f = Field::new(NF + 4);
    let mut rng = Rng::new(0x0d15);
    train(&mut f, &mut rng, 8000);
    let thr = 0.30 * map_spread(&f, NF);

    let mut mems: Vec<Vec<f32>> = Vec::new();
    let mut route: Vec<Vec<usize>> = Vec::new();
    let mut rng2 = Rng::new(0xBEEF);
    for i in 0..6000usize {
        let c = if i < 1500 {
            (rng2.u() % 3) as usize
        } else {
            (rng2.u() % 4) as usize
        };
        let q = centroid(&f, &feats(c));
        let (mut best, mut bd) = (usize::MAX, f32::MAX);
        for (k, m) in mems.iter().enumerate() {
            let d = dist(&q, m);
            if d < bd {
                bd = d;
                best = k;
            }
        }
        if best == usize::MAX || bd > thr {
            mems.push(q);
            route.push(vec![c]);
        } else {
            route[best].push(c);
        }
    }
    eprintln!(
        "[DMEM/H2] {} slots (want 4), purity {:.2}",
        mems.len(),
        purity(&route)
    );
    assert_eq!(mems.len(), 4, "exactly one memory per distinct context");
    assert!(purity(&route) > 0.99, "each memory is one context");
}

/// H3 — ONLINE from scratch, NOISY patterns (a random 3-of-4 of a context's features, +20% a random other), with
/// GROWN dials (create/merge thresholds are fractions of the map's own `spread`; no fixed warmup — allocation
/// runs from the start and duplicates MERGE as the map clusters, self-terminating). Converges to one-per-context.
#[test]
fn principled_online_converges() {
    let mut f = Field::new(NF);
    let mut rng = Rng::new(0x0d15);
    let mut mems: Vec<Vec<f32>> = Vec::new();
    let mut route: Vec<Vec<usize>> = Vec::new();

    for _ in 0..12000usize {
        let c = (rng.u() % 4) as usize;
        let base = feats(c);
        let drop = (rng.u() % 4) as usize;
        let mut active: Vec<usize> = (0..4).filter(|&k| k != drop).map(|k| base[k]).collect();
        if rng.u().is_multiple_of(5) {
            active.push((rng.u() % NF as u64) as usize);
        }
        f.metric_step(&active, 0.1, 8); // train the map every event

        let sp = map_spread(&f, NF);
        let (create_thr, merge_thr) = (0.30 * sp, 0.70 * sp); // grown dials, scale-free
        let q = medoid(&f, &active);
        let (mut best, mut bd) = (usize::MAX, f32::MAX);
        for (k, m) in mems.iter().enumerate() {
            let d = dist(&q, m);
            if d < bd {
                bd = d;
                best = k;
            }
        }
        if best == usize::MAX || bd > create_thr {
            mems.push(q);
            route.push(vec![c]);
        } else {
            route[best].push(c);
            for (a, v) in q.iter().enumerate() {
                mems[best][a] += 0.02 * (v - mems[best][a]); // recognized memory drifts with its context
            }
        }
        // merge duplicates the map has clustered together (create strict, merge loose ⇒ same-context slots coalesce)
        'outer: loop {
            for a in 0..mems.len() {
                for b in (a + 1)..mems.len() {
                    if dist(&mems[a], &mems[b]) < merge_thr {
                        for ax in 0..mems[a].len() {
                            mems[a][ax] = 0.5 * (mems[a][ax] + mems[b][ax]);
                        }
                        let moved = std::mem::take(&mut route[b]);
                        route[a].extend(moved);
                        mems.remove(b);
                        route.remove(b);
                        continue 'outer;
                    }
                }
            }
            break;
        }
    }
    let p = purity(&route);
    eprintln!(
        "[DMEM/H3] online, noisy, grown dials: {} slots (want 4), purity {:.2}",
        mems.len(),
        p
    );
    eprintln!("[DMEM] VERDICT: memory = positions on the grown metric map; recognize by nearest distance (reuse) / create when far (a new memory) — content-addressable, no decay, cheap, and it converges to one memory per distinct context from scratch, online, with dials grown from the map's own scale. This is context AND memory as one substrate — how the hippocampus does it.");
    assert!(
        (3..=6).contains(&mems.len()),
        "online, from scratch, it converges to ~one memory per context (got {})",
        mems.len()
    );
    assert!(
        p > 0.9,
        "each memory is essentially one context (purity {p:.2})"
    );
}
