// SPDX-License-Identifier: MIT-0
//! STRUCTURAL PERCEPTION — group → reduce / relate, all domain-blind (general primitives, no dataset).
//! The substrate operates on FIXED cell vectors, but a scene has a VARIABLE number of objects. These two
//! general primitives bridge that gap so the organism can grow structural rules:
//!
//! - `aggregate` (REDUCE) — fold a variable object set into fixed stats, so COUNTING rules become
//!   growable (counting OBJECTS needs grouping first — pixel correlations can't).
//! - `relate` (the hypergraph edges) — pairwise structure (adjacency, relative position), so RELATIONAL
//!   rules become growable where the raw grid can't tell if two regions touch.
//!
//! Both expose structure, not a task — the organism grows the meaning. This is the general capability the
//! reasoner's building blocks are grown from; a dataset (ARC or any other) is just one consumer.
//!
//! RE-BASELINED at the 1.6 learning flip — both proofs run on the default genome. COUNTING still holds (REDUCE 82 % vs
//! RAW 57 %). ADJACENCY: `relate` still learns the rule (100 %), but the DOMINANCE claim does not hold and is
//! `#[ignore]`d (`relate_exposes_adjacency_the_raw_grid_cannot`): on the default genome the raw-grid readout reaches
//! 100 % too (legacy: 81 %). The legacy gap needs BOTH frozen random feature directions (`feature_lr` 0) and the raw
//! delta step (`stability_cap` 0) — either lever alone at its legacy value and the raw grid learns it — so it was a
//! property of that learner, not of the raw pixels.
//!
//!   cargo test -p openmorph-core --test proofs structure -- --nocapture --test-threads=1

use openmorph_core::grouping::{aggregate_objects, relate, segment_objects};
use openmorph_core::substrate::Field;

use crate::common::Rng;

const W: usize = 8;
const H: usize = 10;
const SCATTER: usize = 20; // cells scattered per scene (constant pixel-sum ⇒ count needs connectivity)

/// A scene of `SCATTER` same-colour cells scattered at random. The pixel-SUM is constant, so the object
/// COUNT is purely the number of connected components — which needs grouping (2-D connectivity), a thing
/// random pixel-features can't compute. Returns (grid, component_count).
fn count_scene(rng: &mut Rng) -> (Vec<usize>, usize) {
    let mut grid = vec![0usize; W * H];
    let mut placed = 0;
    while placed < SCATTER {
        let p = rng.range(W * H);
        if grid[p] == 0 {
            grid[p] = 1;
            placed += 1;
        }
    }
    let n = segment_objects(&grid, W, H, 0).len(); // ground-truth component count
    (grid, n)
}

/// Train a grow_backbone predictor over the given fixed input representation of each scene; return
/// held-out accuracy on a target derived from each scene.
fn learn_over(
    input_of: impl Fn(&mut Field, &(Vec<usize>, usize)) -> Vec<(usize, f32)>,
    n_in: usize,
    target: impl Fn(&(Vec<usize>, usize)) -> f32,
    seed: u64,
) -> f64 {
    let mut f = Field::new(n_in);
    f.rng_seed(seed ^ 0x57);
    let inputs: Vec<usize> = (0..n_in).collect();
    let mut reads = inputs.clone();
    reads.extend(f.grow_backbone(&inputs, 48, 2.5));
    let out = f.len();
    let k = reads.len();
    f.add_unit(reads, vec![0.0; k], 0.0, out);

    let mut rng = Rng::new(seed);
    for _ in 0..4000 {
        let s = count_scene(&mut rng);
        let inp = input_of(&mut f, &s);
        f.learn_step(&inp, out, target(&s), 0.05, 300);
    }
    let (mut ok, mut tot) = (0usize, 0usize);
    for _ in 0..400 {
        let s = count_scene(&mut rng);
        let inp = input_of(&mut f, &s);
        f.stimulate(&inp, 300);
        if f.cells()[out].signum() == target(&s).signum() {
            ok += 1;
        }
        tot += 1;
    }
    ok as f64 / tot as f64
}

#[test]
fn aggregate_lets_the_organism_count_objects_raw_cannot() {
    // Target: are there ≥ 5 connected components? The pixel-sum is constant (SCATTER cells), so the count
    // is pure 2-D connectivity — grouping gives it exactly; random pixel-features can't.
    let target = |s: &(Vec<usize>, usize)| if s.1 >= 12 { 1.0 } else { -1.0 };

    // REDUCE: group → aggregate object sizes → feed the 5 fixed stats (count/sum/min/max/mean), scaled
    // into ~[-1,1] (any sensor normalizes its channel before the substrate).
    let scale = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0];
    let via_aggregate = learn_over(
        move |_f, s| {
            let objs = segment_objects(&s.0, W, H, 0);
            let ag = aggregate_objects(&objs, |o| o.size() as f32);
            ag.to_vec()
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v * scale[i]))
                .collect()
        },
        5,
        target,
        0x0C,
    );
    // RAW: the whole grid as ±1 pixels — object count needs segmentation, so pixels can't (pixel-sum ≠
    // object-count with variable sizes).
    let via_raw = learn_over(
        |_f, s| {
            s.0.iter()
                .enumerate()
                .map(|(i, &v)| (i, if v == 0 { -1.0 } else { 1.0 }))
                .collect()
        },
        W * H,
        target,
        0x0C,
    );

    eprintln!(
        "[STRUCTURE] ≥12 connected components? (constant pixel-sum ⇒ pure connectivity)\n  REDUCE (group → aggregate stats): {:.0}%\n  RAW (grid pixels)               : {:.0}%",
        via_aggregate * 100.0,
        via_raw * 100.0,
    );
    eprintln!("[STRUCTURE] VERDICT: `aggregate` (the general REDUCE) folds a variable object set into fixed stats, so a counting rule the fixed-width substrate can't express over raw pixels — counting OBJECTS needs grouping first — becomes growable. Group → reduce is the domain-blind bridge from a variable scene to the substrate; a dataset is just one consumer.");

    assert!(
        via_aggregate >= 0.75,
        "counting via aggregate should work (got {via_aggregate:.2}) — the threshold sits at the count mode, so borderline scenes cap it"
    );
    assert!(
        via_aggregate > via_raw + 0.2,
        "the reduce must decisively beat raw pixels at counting components ({via_aggregate:.2} vs {via_raw:.2} ≈ chance)"
    );
}

/// Scene: two objects of DIFFERENT colours, either touching or separated. Target: are they adjacent?
fn adjacency_scene(rng: &mut Rng) -> (Vec<usize>, bool) {
    let mut grid = vec![0usize; W * H];
    let touch = rng.u().is_multiple_of(2);
    let row = 2 + rng.range(4);
    let c0 = 1 + rng.range(3);
    grid[row * W + c0] = 1; // object A (colour 1)
    let c1 = if touch { c0 + 1 } else { c0 + 2 + rng.range(3) };
    if c1 < W {
        grid[row * W + c1] = 2; // object B (colour 2)
    }
    (grid, touch)
}

/// Held-out accuracy of a readout over a 48-wide backbone learning adjacency from the `relate` edge (`use_relate`)
/// or from the raw grid pixels, on the default genome.
fn adjacency(use_relate: bool, seed: u64) -> f64 {
    let n_in = if use_relate { 3 } else { W * H };
    let mut f = Field::new(n_in);
    f.rng_seed(seed ^ 0xAD);
    let inputs: Vec<usize> = (0..n_in).collect();
    let mut reads = inputs.clone();
    reads.extend(f.grow_backbone(&inputs, 48, 2.5));
    let out = f.len();
    let k = reads.len();
    f.add_unit(reads, vec![0.0; k], 0.0, out);
    let feat = |grid: &[usize]| -> Vec<(usize, f32)> {
        if use_relate {
            // RELATE: the pairwise edge — adjacency + relative position (the hypergraph structure).
            let objs = segment_objects(grid, W, H, 0);
            let rels = relate(&objs, W, H);
            let r = rels.first();
            vec![
                (0, r.map_or(-1.0, |r| if r.adjacent { 1.0 } else { -1.0 })),
                (1, r.map_or(0.0, |r| r.dx)),
                (2, r.map_or(0.0, |r| r.dy)),
            ]
        } else {
            grid.iter()
                .enumerate()
                .map(|(i, &v)| (i, if v == 0 { -1.0 } else { 1.0 }))
                .collect()
        }
    };
    let mut rng = Rng::new(seed);
    for _ in 0..3000 {
        let (g, touch) = adjacency_scene(&mut rng);
        let inp = feat(&g);
        f.learn_step(&inp, out, if touch { 1.0 } else { -1.0 }, 0.05, 300);
    }
    let (mut ok, mut tot) = (0, 0);
    for _ in 0..400 {
        let (g, touch) = adjacency_scene(&mut rng);
        let inp = feat(&g);
        f.stimulate(&inp, 300);
        if (f.cells()[out] > 0.0) == touch {
            ok += 1;
        }
        tot += 1;
    }
    ok as f64 / tot as f64
}

#[test]
fn relate_exposes_adjacency() {
    let via_relate = adjacency(true, 0x1D);
    let via_raw = adjacency(false, 0x1D);

    eprintln!(
        "[STRUCTURE] are two objects adjacent? a relational rule\n  RELATE (hypergraph edge): {:.0}%\n  RAW (grid pixels)       : {:.0}%",
        via_relate * 100.0,
        via_raw * 100.0,
    );
    eprintln!("[STRUCTURE] VERDICT: `relate` exposes the raw pairwise structure (the hypergraph edges — adjacency, relative position), so a relational rule is growable from three numbers. group → relate completes the object hypergraph; the organism grows which relations matter. (Since the 1.6 re-baseline this is NOT claimed to beat raw pixels: on the default genome a raw-grid readout learns this adjacency rule too — see the module docs.)");

    assert!(
        via_relate >= 0.9,
        "adjacency via relate should work (got {via_relate:.2})"
    );
}

/// The DOMINANCE half of the original adjacency proof, kept verbatim but ignored: on the 1.6 default the raw-grid
/// readout reaches 100 % too (see the module docs).
#[test]
#[ignore = "1.6 re-baseline: does not hold on the default genome — RAW 100 % = RELATE 100 %; the legacy 81 % needs frozen random directions AND the raw step; see this file's module docs"]
fn relate_exposes_adjacency_the_raw_grid_cannot() {
    let via_relate = adjacency(true, 0x1D);
    let via_raw = adjacency(false, 0x1D);
    eprintln!("[STRUCTURE] relate {via_relate:.2} vs raw {via_raw:.2}");
    assert!(
        via_relate > via_raw + 0.15,
        "relate must beat raw pixels at adjacency ({via_relate:.2} vs {via_raw:.2})"
    );
}
