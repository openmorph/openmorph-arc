// SPDX-License-Identifier: MIT-0
//! SEED_COORD — the percept→coordinate seam (OM-5 archive key / OM-9 egocentric key). A cell's concept-map
//! coordinate is DERIVED from a high-dimensional percept by a fixed random projection (Johnson–Lindenstrauss),
//! so the mapping is (1) STABLE/reproducible — same percept ⇒ same coordinate; (2) DISTINCT — different percepts
//! ⇒ different coordinates; (3) DISTANCE-PRESERVING — similar percepts land near, dissimilar percepts land far,
//! so `distance()` over seeded coordinates ranks percept similarity; and therefore (4) a REPEATING state re-seeds
//! to the SAME region (an archive dedups by distance instead of minting one key per raw frame).
//!
//! Truth-not-trophy: `seed_coord` does NOT interpret the percept — it is a distance-preserving hash of whatever
//! representation it is fed. This proof asserts only the seam's own properties (stability, injectivity,
//! JL distance-preservation, dedup); the QUALITY of the resulting key is the quality of the percept, which is
//! the separate upstream M2-latent lever.
//!
//!   cargo test -p openmorph-core --test proofs substrate_seed_coord -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;

use crate::common::Rng;

const D: usize = 64; // percept dimensionality (e.g. a downsampled frame / grown latent)

fn percept(rng: &mut Rng) -> Vec<f32> {
    (0..D)
        .map(|_| (rng.u() % 1000) as f32 / 500.0 - 1.0)
        .collect()
}
fn jitter(base: &[f32], rng: &mut Rng, amp: f32) -> Vec<f32> {
    base.iter()
        .map(|&x| x + amp * ((rng.u() % 1000) as f32 / 500.0 - 1.0))
        .collect()
}
fn coord(f: &Field, cell: usize) -> Vec<f32> {
    f.position(cell)
}
fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn seed_coord_is_stable_distinct_and_distance_preserving() {
    let mut f = Field::new(64);
    let mut rng = Rng::new(0x5eed);

    // (1) STABLE / reproducible — the same percept always seeds the same coordinate.
    let p = percept(&mut rng);
    f.seed_coord(0, &p);
    let c0 = coord(&f, 0);
    f.seed_coord(1, &p); // a DIFFERENT cell, SAME percept
    let c1 = coord(&f, 1);
    assert_eq!(
        c0, c1,
        "same percept must seed identical coordinates (stability)"
    );
    // and re-seeding is idempotent
    f.seed_coord(0, &p);
    assert_eq!(
        coord(&f, 0),
        c0,
        "re-seeding the same percept is idempotent"
    );

    // (2) DISTINCT — different percepts seed different coordinates.
    let q = percept(&mut rng);
    f.seed_coord(2, &q);
    assert!(
        l2(&coord(&f, 2), &c0) > 1e-3,
        "distinct percepts must seed distinct coordinates"
    );

    // (3) DISTANCE-PRESERVING (Johnson–Lindenstrauss). For many percept pairs, the coordinate distance should
    //     RANK-CORRELATE with the true percept distance: a near-duplicate (small jitter) lands closer than an
    //     unrelated percept, reliably. Measure it as a clean margin over many trials.
    let mut near_closer = 0usize;
    let trials = 200usize;
    for _ in 0..trials {
        let base = percept(&mut rng);
        let near = jitter(&base, &mut rng, 0.1); // a small perturbation (same state, sensor noise)
        let far = percept(&mut rng); // an unrelated percept
        f.seed_coord(10, &base);
        f.seed_coord(11, &near);
        f.seed_coord(12, &far);
        let d_near = l2(&coord(&f, 10), &coord(&f, 11));
        let d_far = l2(&coord(&f, 10), &coord(&f, 12));
        if d_near < d_far {
            near_closer += 1;
        }
    }
    let rate = near_closer as f32 / trials as f32;
    eprintln!("[SEED-COORD] JL distance-preservation: near-percept closer than far-percept in {:.0}% of {trials} trials", rate * 100.0);
    assert!(
        rate > 0.9,
        "a near-duplicate percept must map nearer than an unrelated one (JL) — got {rate:.2}"
    );

    // (4) DEDUP — a REPEATING state (noisy re-observations of ONE latent) forms ONE tight cluster on the map,
    //     well separated from OTHER states: distance-dedup collapses the exploding raw stream to few keys.
    let states: Vec<Vec<f32>> = (0..4).map(|_| percept(&mut rng)).collect();
    let mut within = 0.0f32;
    let mut wn = 0.0f32;
    let mut between = f32::MAX;
    // seed 8 noisy observations of each state into cells 20.., record per-state cluster centroids in coords
    let mut centroids: Vec<Vec<f32>> = Vec::new();
    for (s, st) in states.iter().enumerate() {
        let mut cells = Vec::new();
        for k in 0..8 {
            let cell = 20 + s * 8 + k;
            f.seed_coord(cell, &jitter(st, &mut rng, 0.1));
            cells.push(cell);
        }
        // within-state spread
        for i in 0..cells.len() {
            for j in (i + 1)..cells.len() {
                within += l2(&coord(&f, cells[i]), &coord(&f, cells[j]));
                wn += 1.0;
            }
        }
        let dim = f.position(0).len();
        let ctr: Vec<f32> = (0..dim)
            .map(|a| cells.iter().map(|&c| f.position(c)[a]).sum::<f32>() / cells.len() as f32)
            .collect();
        centroids.push(ctr);
    }
    within /= wn;
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            between = between.min(l2(&centroids[i], &centroids[j]));
        }
    }
    eprintln!("[SEED-COORD] dedup: within-state spread {within:.3} ≪ between-state gap {between:.3} (ratio {:.1}×)", between / within.max(1e-6));
    assert!(between > 2.0 * within, "a repeating state must cluster tighter than states are apart (dedup): within {within:.3} vs between {between:.3}");

    eprintln!("[SEED-COORD] VERDICT: the percept→coordinate seam is stable, injective, distance-preserving (JL), and dedups a repeating state to one map region. It carries whatever representation it is fed — the key's quality is the percept's quality (the upstream M2-latent lever).");
}
