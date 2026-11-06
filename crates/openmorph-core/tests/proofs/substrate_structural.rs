// SPDX-License-Identifier: MIT-0
//! STRUCTURAL INTROSPECTION — a morph understands another morph by GRAPH SHAPE (openmorph#172, Sébastien's
//! deep idea). The behavioural `embed` reads WHAT a morph computes (probe response); `structural_embed` reads
//! HOW IT IS WIRED — the topology of its cell-graph — with no probe and no I/O port assumption. If perception is
//! graph-embedding, then perceiving another morph is the SAME act as perceiving a scene.
//!
//! Three claims, scored INSIDE:
//!   (a) INVARIANCE  — the structural code is INVARIANT to relabelling a morph's cells + reordering its units
//!                     (an isomorphic re-wiring embeds identically). Tested on ACTUAL random permutations.
//!   (b) DISCRIMINATION — topologically-DIFFERENT morphs (chain / star / tree / fractal-tower) get DIFFERENT
//!                     codes, controlled for node count (chain, star, tree all have the SAME N) so the code is
//!                     NOT a size proxy; isomorphic morphs stay near-zero apart.
//!   (c) MORPH-TEACHES-MORPH — the payoff. A TEACHER reads a STUDENT's structure, DIAGNOSES the structural HOLE
//!                     (a sub-part whose inputs never reach an output — missing units, visible in topology), and
//!                     TRANSPLANTS a matching competent skill-brick THERE (nest). The differentiator is the
//!                     structural understanding of WHERE: the structure-BLIND teacher is handed the SAME brick but
//!                     places it structure-blind and fails. So the win is provably the structural diagnosis, NOT
//!                     the free weights (both arms get them). Beats solo regrow under budget too.
//!
//!   cargo test -p openmorph-core --test proofs substrate_structural -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
// Helpers: build a Field from an abstract edge-set, and relabel/reorder it into an ISOMORPHIC copy.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/// A topology = a list of units, each `(reads, write)` over abstract cell ids `0..n`. Weights are irrelevant to
/// structure, so they are set to 1.0 — the point is the WIRING.
type Topo = Vec<(Vec<usize>, usize)>;

/// Build a `Field` (n cells) from a topology, in the given unit order.
fn build(n: usize, topo: &Topo) -> Field {
    let mut f = Field::new(n);
    for (reads, write) in topo {
        let w = vec![1.0f32; reads.len()];
        f.add_unit(reads.clone(), w, 0.0, *write);
    }
    f
}

/// Build an ISOMORPHIC copy: relabel every cell by permutation `perm` and add the units in a shuffled order.
/// Same graph, different labels + order — the invariance adversary.
fn isomorph(n: usize, topo: &Topo, seed: u64) -> Field {
    let mut rng = Rng::new(seed);
    // A random permutation of the n cell labels.
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        perm.swap(i, j);
    }
    // Relabel every unit through perm, then shuffle unit order.
    let mut relabelled: Topo = topo
        .iter()
        .map(|(reads, write)| (reads.iter().map(|&c| perm[c]).collect(), perm[*write]))
        .collect();
    for i in (1..relabelled.len()).rev() {
        let j = rng.below(i + 1);
        relabelled.swap(i, j);
    }
    build(n, &relabelled)
}

// ── Canonical topologies (chain / star / tree share the SAME node count N to control for size). ──

/// A CHAIN over `n` cells: 0→1→2→…→(n-1). Uniform degree 2 internally, no hub.
fn chain(n: usize) -> Topo {
    (0..n - 1).map(|i| (vec![i], i + 1)).collect()
}
/// A STAR over `n` cells: every leaf 1..n feeds hub 0. Hub in-degree n-1 — one dominant node.
fn star(n: usize) -> Topo {
    (1..n).map(|leaf| (vec![leaf], 0)).collect()
}
/// A balanced BINARY TREE over `n` cells: parent p is written from its two children 2p+1, 2p+2.
fn tree(n: usize) -> Topo {
    let mut t = Topo::new();
    for p in 0..n {
        let (l, r) = (2 * p + 1, 2 * p + 2);
        if r < n {
            t.push((vec![l, r], p));
        } else if l < n {
            t.push((vec![l], p));
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
// (a) INVARIANCE + (b) DISCRIMINATION
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn structural_code_is_permutation_invariant_and_discriminates_topology() {
    let n = 15;
    let topos: [(&str, Topo); 3] = [("chain", chain(n)), ("star", star(n)), ("tree", tree(n))];

    // ── (a) INVARIANCE: same graph, many random relabel+reorder → IDENTICAL code (distance ~0). ──
    let mut worst_within = 0.0f32;
    for (name, topo) in &topos {
        let base = build(n, topo);
        for seed in 0..8u64 {
            let iso = isomorph(n, topo, 0xA11CE ^ (seed.wrapping_mul(0x9E37)));
            let d = base.structural_distance(&iso);
            worst_within = worst_within.max(d);
            eprintln!(
                "[INVARIANCE] {name}: base vs isomorph(seed {seed}) structural distance = {d:.6}"
            );
            assert!(
                d < 1e-5,
                "{name}: structural code must be permutation-invariant, got {d}"
            );
        }
    }

    // ── (b) DISCRIMINATION: different topologies, SAME node count N=15, must separate. ──
    let codes: Vec<(&str, Field)> = topos
        .iter()
        .map(|(name, topo)| (*name, build(n, topo)))
        .collect();
    let mut min_between = f32::MAX;
    for i in 0..codes.len() {
        for j in i + 1..codes.len() {
            let d = codes[i].1.structural_distance(&codes[j].1);
            min_between = min_between.min(d);
            eprintln!(
                "[DISCRIMINATION] {} vs {} (both N={n}) structural distance = {d:.4}",
                codes[i].0, codes[j].0
            );
        }
    }

    // Node-count control: report that all three graphs have the SAME participating-node count, so the
    // separation cannot be a size proxy.
    for (name, f) in &codes {
        eprintln!(
            "[SIZE-CONTROL] {name}: nodes(from embed via structure) — units={}",
            f.units.len()
        );
    }

    eprintln!(
        "[SUMMARY] worst within-topology (isomorph) distance = {worst_within:.6}; \
         min between-topology distance = {min_between:.4}; separation ratio = {:.1}x",
        min_between / worst_within.max(1e-9)
    );
    assert!(
        min_between > 0.05,
        "distinct topologies of equal size must get distinct codes, min_between={min_between}"
    );
    assert!(
        min_between > worst_within * 100.0,
        "between-topology distance ({min_between}) must dwarf within-topology ({worst_within})"
    );
}

#[test]
fn structural_code_sees_fractal_nesting_depth() {
    // A cell can be a MORPH (#91). A FLAT graph and a NESTED (encapsulated) graph computing a similar thing must
    // read as structurally different — the code recurses into nested sub-morphs (nested_frac + child_* slots).
    let mut rng = Rng::new(0xF00D);
    // Grow a small flat morph y = tanh(x0 - x1).
    let mut flat = Field::new(4);
    flat.config_mut().max_hidden = 2;
    for _ in 0..800 {
        let (a, b) = (rng.sym(), rng.sym());
        flat.grow(&[(0, a), (1, b)], 3, (a - b).tanh(), 0.1, 40);
    }
    // A tower: encapsulate the flat morph and NEST it inside a host — same behaviour, fractal structure.
    let brick = flat.encapsulate(&[3]);
    let mut tower = Field::new(4);
    tower.nest(brick, &[0, 1], 3, 40);

    let d_self = flat.structural_distance(&flat.clone());
    let d_nested = flat.structural_distance(&tower);
    eprintln!("[FRACTAL] flat vs clone = {d_self:.6}; flat vs nested-tower = {d_nested:.4}");
    eprintln!(
        "[FRACTAL] flat.nested_frac={:.3}  tower.nested_frac={:.3}",
        flat.structural_embed()[7],
        tower.structural_embed()[7]
    );
    assert!(d_self < 1e-6, "a clone must embed identically");
    assert!(
        d_nested > 0.05,
        "a nested (fractal) rewrite must read as structurally different, got {d_nested}"
    );
}

// ── The structural code is COMPLEMENTARY to the existing exact `structure_key` (a discrete hash of the
//    composition-tree keyed on fan-in multiset + nesting). `structure_key` COLLIDES a chain and a star
//    (both are all-fan-in-1 units over the same cell count → identical hash), where the continuous
//    `structural_embed` separates them on hub-ness. Exact identity vs graded similarity — two different jobs. ──
#[test]
fn structural_embed_separates_what_the_exact_structure_key_collides() {
    let n = 15;
    let (c, s) = (build(n, &chain(n)), build(n, &star(n)));
    let same_key = c.structure_key() == s.structure_key();
    let graded = c.structural_distance(&s);
    eprintln!(
        "[COMPLEMENT] chain.structure_key == star.structure_key : {same_key}  (hash collides)\n\
         [COMPLEMENT] chain vs star structural_distance : {graded:.4}  (graded metric separates)"
    );
    assert!(
        same_key,
        "the fan-in-multiset hash is expected to collide chain vs star"
    );
    assert!(
        graded > 1.0,
        "the graded structural metric must still separate them"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
// (c) MORPH-TEACHES-MORPH — the payoff. A teacher reads a student's structure, finds the HOLE, and TRANSPLANTS
//     a matching competent skill-brick INTO it (nest). The differentiator is the STRUCTURAL understanding of
//     WHERE the hole is: the structure-BLIND teacher is handed the *same* brick but places it structure-blind
//     and fails. So the win is provably the structural diagnosis, NOT the free weights (both arms get them).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/// The hole skill: a PRODUCT tanh(3·x·y) — needs grown hidden features, so a competent brick is a real
/// multi-unit sub-morph (not a one-liner the student could trivially regrow in a few steps).
fn true_hole(x: f32, y: f32) -> f32 {
    (3.0 * x * y).tanh()
}

/// The cells in the backward cone of `out` (out + everything that transitively feeds it), by BFS over the
/// read→write graph. Passed to `encapsulate` so a DEEP sub-morph (writer + all its hidden features) is banked
/// whole — `encapsulate(&[out])` alone would grab only the writer.
fn cone(f: &Field, out: usize) -> Vec<usize> {
    let mut cells = vec![out];
    let mut i = 0;
    while i < cells.len() {
        let c = cells[i];
        i += 1;
        for u in &f.units {
            if u.write == c {
                for &r in &u.reads {
                    if !cells.contains(&r) {
                        cells.push(r);
                    }
                }
            }
        }
    }
    cells
}

/// Train a competent, deep hole-brick as a standalone morph (inputs 0,1 → out 2), then bank its WHOLE
/// backward cone as a reusable sub-morph. This is the library skill both teachers are handed.
fn hole_brick(seed: u64) -> Field {
    let mut m = Field::new(3);
    m.config_mut().max_hidden = 8;
    m.config_mut().birth_gate = 0.05;
    m.config_mut().feature_lr = 0.3;
    m.rng_seed(seed);
    let mut rng = Rng::new(seed ^ 0xB1C);
    for _ in 0..2000 {
        let (x, y) = (rng.sym(), rng.sym());
        m.grow(&[(0, x), (1, y)], 2, true_hole(x, y), 0.1, 40);
    }
    m.encapsulate(&cone(&m, 2))
}

/// A student for the 4-region task. Regions live on input pairs {0,1},{2,3},{4,5},{6,7} writing out-slots
/// 8,9,10,11. Region `hole_region` is left an EMPTY structural hole (its inputs are never read, its out-slot
/// has no writer); the other three are built as simple linear skills. Returns the student.
fn student_with_hole(seed: u64, hole_region: usize) -> Field {
    let mut s = Field::new(12);
    s.rng_seed(seed);
    let mut rng = Rng::new(seed ^ 0x5A5A);
    for region in 0..4 {
        if region == hole_region {
            continue; // the hole: inputs (2*region, 2*region+1) dangle, out-slot 8+region unwritten
        }
        let (i0, i1, out) = (2 * region, 2 * region + 1, 8 + region);
        for _ in 0..300 {
            let (x, y) = (rng.sym(), rng.sym());
            s.grow(&[(i0, x), (i1, y)], out, (x - y).tanh(), 0.1, 30); // a simple, cheap linear skill
        }
    }
    s
}

/// STRUCTURAL diagnosis: the input cells (0..8) that NO unit reads — the dangling pair is the hole's inputs —
/// and the out-slot (8..12) with no writer. Pure topology; no weights, no task labels handed in.
fn diagnose_hole(s: &Field) -> ([usize; 2], usize) {
    let read = |c: usize| s.units.iter().any(|u| u.reads.contains(&c));
    let written = |c: usize| s.units.iter().any(|u| u.write == c);
    let dangling: Vec<usize> = (0..8).filter(|&c| !read(c)).collect();
    let slot = (8..12).find(|&c| !written(c)).unwrap_or(11);
    ([dangling[0], dangling[1]], slot)
}

/// MAE of the hole's output vs its true product, over held-out inputs (the other regions are pre-built for
/// every arm, so this isolates whether the hole was correctly filled). `hole_inputs` are the cells the hole's
/// product is a function of; `out` the cell the arm wrote its brick to.
fn hole_mae(s: &mut Field, hole_inputs: [usize; 2], out: usize, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut e = 0.0;
    let n = 300;
    for _ in 0..n {
        // Randomize ALL eight inputs (so a brick wired to the wrong pair is scored on irrelevant signals).
        let mut vals = [0.0f32; 8];
        for v in vals.iter_mut() {
            *v = rng.sym();
        }
        let inputs: Vec<(usize, f32)> = (0..8).map(|c| (c, vals[c])).collect();
        s.stimulate(&inputs, 400);
        let pred = s.cells()[out];
        let target = true_hole(vals[hole_inputs[0]], vals[hole_inputs[1]]);
        e += (pred - target).abs();
    }
    e / n as f32
}

#[test]
fn a_teacher_transplants_a_skill_brick_by_reading_the_students_structure() {
    let seeds = [0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let (mut struct_mae, mut blind_mae, mut solo_mae) = (0.0f32, 0.0f32, 0.0f32);
    let mut struct_wrong = 0; // times the structural teacher mis-placed (should be ~0)
    let mut blind_wrong = 0; // times the blind teacher mis-placed (should be most)

    for (t, &seed) in seeds.iter().enumerate() {
        let hole_region = t % 4; // rotate WHICH region is the hole, so no arm can hard-code it
        let hole_inputs = [2 * hole_region, 2 * hole_region + 1];
        let out = 8 + hole_region;
        let brick = hole_brick(seed);

        // ── ARM 1: STRUCTURAL teacher — diagnose the hole from the graph, transplant the brick THERE. ──
        let mut s1 = student_with_hole(seed, hole_region);
        let (dinputs, dslot) = diagnose_hole(&s1);
        s1.nest(brick.clone(), &dinputs, dslot, 200);
        let placed_right = dinputs == hole_inputs && dslot == out;
        if !placed_right {
            struct_wrong += 1;
        }
        let m1 = hole_mae(&mut s1, hole_inputs, out, seed ^ 0xEE);

        // ── ARM 2: BLIND teacher — handed the SAME brick, but places it structure-blind: a RANDOM input pair
        //    and the correct out-slot. The control isolating structural understanding (weights are free to both). ──
        let mut s2 = student_with_hole(seed, hole_region);
        let mut rng = Rng::new(seed ^ 0xB11);
        let a = rng.below(8);
        let mut b = rng.below(8);
        while b == a {
            b = rng.below(8);
        }
        let blind_inputs = [a.min(b), a.max(b)];
        s2.nest(brick.clone(), &blind_inputs, out, 200);
        if blind_inputs != hole_inputs {
            blind_wrong += 1;
        }
        let m2 = hole_mae(&mut s2, hole_inputs, out, seed ^ 0xEE);

        // ── ARM 3: SOLO — no brick, no teacher: grow the hole from scratch with a SMALL confirm-budget. ──
        let mut s3 = student_with_hole(seed, hole_region);
        s3.config_mut().max_hidden = 8;
        s3.config_mut().birth_gate = 0.05;
        s3.config_mut().feature_lr = 0.3;
        let mut rng = Rng::new(seed ^ 0xC3);
        for _ in 0..200 {
            let (x, y) = (rng.sym(), rng.sym());
            s3.grow(
                &[(hole_inputs[0], x), (hole_inputs[1], y)],
                out,
                true_hole(x, y),
                0.1,
                40,
            );
        }
        let m3 = hole_mae(&mut s3, hole_inputs, out, seed ^ 0xEE);

        eprintln!(
            "[TEACH] seed {seed:#x} hole=region{hole_region}: struct {m1:.4} (placed {}) | \
             blind {m2:.4} (inputs {blind_inputs:?}) | solo {m3:.4}",
            if placed_right { "RIGHT" } else { "wrong" }
        );
        struct_mae += m1;
        blind_mae += m2;
        solo_mae += m3;
    }
    let ns = seeds.len() as f32;
    struct_mae /= ns;
    blind_mae /= ns;
    solo_mae /= ns;
    eprintln!(
        "[TEACH-SUMMARY] mean hole-MAE — STRUCTURAL {struct_mae:.4}  |  BLIND {blind_mae:.4}  |  SOLO {solo_mae:.4}\n\
         [TEACH-SUMMARY] mis-placements — structural {struct_wrong}/{} | blind {blind_wrong}/{}",
        seeds.len(),
        seeds.len()
    );

    // The structural teacher places the brick correctly every time; the blind teacher (same brick) almost never.
    assert_eq!(
        struct_wrong, 0,
        "structural diagnosis must place the brick correctly every time"
    );
    assert!(
        blind_wrong as f32 >= 0.6 * seeds.len() as f32,
        "the blind baseline must mostly mis-place (else the task is too easy to be a control)"
    );
    // Structural transplant is a decisive win over BOTH the same-brick-but-blind teacher and solo regrow.
    assert!(
        struct_mae < 0.15,
        "the transplanted competent brick must make the student competent, got {struct_mae}"
    );
    assert!(
        struct_mae < blind_mae * 0.5,
        "structural placement ({struct_mae}) must dwarf structure-blind placement ({blind_mae})"
    );
    assert!(
        struct_mae < solo_mae * 0.5,
        "structural transplant ({struct_mae}) must beat solo regrow under budget ({solo_mae})"
    );
}
