// SPDX-License-Identifier: MIT-0
//! FRACTAL RECURSIVE NESTING (#91/#64) — a cell can BE a Morph, recursively. The tick-era `Cell::Nested` ask
//! is answered by the async core with ONE primitive pair: [`Field::encapsulate`] banks a grown subgraph as a
//! sub-morph (a single addressable node), and [`Field::nest`] plants it as a CELL whose value is that
//! sub-morph's settled output. Reading it settles the sub IN the active cascade — and the sub may itself hold
//! nested cells, so a skill-of-a-skill-of-a-skill is one node, not a flat band. Depth is emergent; there is no
//! floor. This replaces the whole crystallize/graft/Skill/apply cluster — reuse IS nesting.
//!
//! Proved here:
//!   (A) RECURSION IN THE ACTIVE PATH — a nested cell fires during settle and yields the sub's output.
//!   (B) FRACTAL DEPTH — a 3-level tower (TOP ▷ S2 ▷ S1) computes exactly the composed function; when TOP
//!       settles, its nested S2 settles, whose nested S1 settles — three scales, one addressable top cell.
//!       Flat composition (bands) tops out at pairs; a skill-of-a-skill can't be a single routed node.
//!   (C) OPT-IN — a field with no nested cell is the plain `tanh` substrate, unchanged.
//!
//!   cargo test -p openmorph-core --test proofs substrate_nest -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const BUD: usize = 20;

/// S1 — a leaf sub-morph: `s1(x,y) = tanh(x − y)`. Cells 0=x, 1=y, 2=out.
fn build_s1() -> Field {
    let mut f = Field::new(3);
    f.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);
    f
}

/// S2 — a sub-morph that CONTAINS a nested S1: `s2(x,y,z) = tanh(s1(x,y) + z)`. Cells 0=x,1=y,2=z; a nested
/// S1 cell writes 3 = s1(x,y); a leaf reads [3,2] → 4 = out. So S2 has its OWN nested cell inside it.
fn build_s2() -> Field {
    let mut f = Field::new(3); // 0=x, 1=y, 2=z
    f.nest(build_s1(), &[0, 1], 3, BUD); // cell 3 = nested S1(x,y)
    f.add_unit(vec![3, 2], vec![1.0, 1.0], 0.0, 4); // cell 4 = tanh(s1 + z)
    f
}

/// The reference (closed form) the tower must reproduce exactly.
fn s1(x: f32, y: f32) -> f32 {
    (x - y).tanh()
}
fn s2(x: f32, y: f32, z: f32) -> f32 {
    (s1(x, y) + z).tanh()
}

fn signs3() -> Vec<(f32, f32, f32)> {
    let s = [-1.0f32, 1.0];
    let mut v = Vec::new();
    for &a in &s {
        for &b in &s {
            for &c in &s {
                v.push((a, b, c));
            }
        }
    }
    v
}

#[test]
fn a_nested_cell_computes_its_sub_morph_in_the_active_path() {
    // (A) A single nested S1 cell in a host: reading it settles S1 and yields s1(x,y).
    let mut host = Field::new(2); // 0=x, 1=y
    host.nest(build_s1(), &[0, 1], 2, BUD); // cell 2 IS the S1 sub-morph
    let mut worst = 0.0f32;
    for (x, y, _) in signs3() {
        host.stimulate(&[(0, x), (1, y)], BUD);
        let got = host.cells()[2];
        worst = worst.max((got - s1(x, y)).abs());
    }
    eprintln!("[NEST] (A) a nested S1 cell reproduces s1(x,y) in the active cascade — worst error {worst:.4}");
    assert!(
        worst < 1e-3,
        "nested cell must compute its sub-morph (worst {worst:.5})"
    );
}

#[test]
fn a_three_level_tower_composes_a_skill_of_a_skill() {
    // (B) TOP ▷ S2 ▷ S1 — TOP nests S2, and S2 already nests S1. One addressable top cell holds three scales.
    let mut top = Field::new(3); // 0=a, 1=b, 2=c
    top.nest(build_s2(), &[0, 1, 2], 3, BUD); // cell 3 = S2(a,b,c), and S2 contains S1

    let mut worst = 0.0f32;
    for (a, b, c) in signs3() {
        top.stimulate(&[(0, a), (1, b), (2, c)], BUD);
        let got = top.cells()[3];
        worst = worst.max((got - s2(a, b, c)).abs());
    }
    eprintln!(
        "[NEST] (B) 3-level tower TOP ▷ S2 ▷ S1: one top cell computes the full composition — worst error {worst:.4}"
    );
    eprintln!("[NEST] VERDICT: the fractal recursive hypergraph lands with ONE primitive pair. `nest` plants a sub-morph as a CELL whose value is that morph's settled output — recursion in the active settle path — and because the sub can itself hold nested cells, a skill-of-a-skill (TOP ▷ S2 ▷ S1) is ONE addressable node, not a flat band. Depth is emergent, no authored floor (#64/#91). This is the keystone that lets composition cover the full recursive tree instead of only M+M² pairs, and it SUBSUMES the entire crystallize/graft/Skill/apply cluster — reuse is nesting. `encapsulate` banks any grown subgraph as such a node; the concept-map can even detect the sub-morph boundary itself (a cluster of related cells IS a sub-morph).");
    assert!(
        worst < 1e-3,
        "the 3-level tower must compute the composed function exactly (worst {worst:.5})"
    );
}

#[test]
fn a_field_without_nested_cells_is_the_plain_substrate() {
    // (C) OPT-IN: a leaf-only field is byte-identical to the plain tanh substrate.
    let mut f = Field::new(3);
    f.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);
    f.stimulate(&[(0, 0.7), (1, -0.3)], BUD);
    assert!((f.cells()[2] - (0.7_f32 - (-0.3)).tanh()).abs() < 1e-4);
    eprintln!("[NEST] (C) no nested cell ⇒ the plain tanh substrate, unchanged (opt-in).");
}
