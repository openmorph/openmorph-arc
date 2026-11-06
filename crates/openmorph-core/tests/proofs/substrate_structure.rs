// SPDX-License-Identifier: MIT-0
//! THE PRIME/TREE STRUCTURE READ-ALGEBRA (#172) — the "prime world", delivered as GROWN read-queries over the
//! composition graph, NOT as a materialized bignum and NOT as a metric swap. The composition graph (a morph's
//! `nested` sub-morphs, recursively) already IS a prime factorization: `nest` multiplies a factor in, a flat
//! morph is an irreducible atom. This proof asserts the algebra does REAL work — shared-subtree detection, the
//! structural gcd, coprimality (independence), and the p-adic valuation (nesting depth × multiplicity) — with
//! truth-not-trophy baselines (a flat morph factorizes to nothing; the key is weight-INVARIANT but shape-SENSITIVE;
//! the whole algebra is `&self`-read, so it never touches the Euclidean learning metric).

use openmorph_core::{Activation, Field};

/// A tiny leaf sub-morph, shape "2 cells, one unit reading 1 cell": out = tanh(w·x). The WEIGHT is a free
/// parameter — structure_key must be blind to it.
fn sub_a(weight: f32) -> Field {
    let mut s = Field::new(2);
    s.add_unit(vec![0], vec![weight], 0.0, 1);
    s
}
/// A structurally DIFFERENT leaf: 3 cells, one unit reading 2 cells → a different shape, hence a different key.
fn sub_b() -> Field {
    let mut s = Field::new(3);
    s.add_unit(vec![0, 1], vec![1.0, 1.0], 0.0, 2);
    s
}

#[test]
fn structure_key_is_weight_invariant_but_shape_sensitive() {
    // WEIGHTS don't move the key (behaviour is embed's job; this is SHAPE) …
    assert_eq!(
        sub_a(1.0).structure_key(),
        sub_a(99.0).structure_key(),
        "structure_key must be blind to weights"
    );
    // … but SHAPE does (different cell count + fan-in).
    assert_ne!(
        sub_a(1.0).structure_key(),
        sub_b().structure_key(),
        "structurally different morphs must differ"
    );
}

#[test]
fn flat_morph_is_an_irreducible_atom() {
    // Baseline: a morph with no nested cells has NO prime factors.
    let mut flat = Field::new(2);
    flat.add_unit(vec![0], vec![1.0], 0.0, 1);
    assert!(
        flat.factorization().is_empty(),
        "a flat morph factorizes to nothing (an atom)"
    );
}

#[test]
fn gcs_finds_the_shared_subtree_and_coprime_detects_independence() {
    let ka = sub_a(1.0).structure_key();

    // h1 = {sub_a, sub_b}; h2 = {sub_a}. Their shared substructure is exactly sub_a.
    let mut h1 = Field::new(6);
    h1.nest(sub_a(1.0), &[0], 3, 20);
    h1.nest(sub_b(), &[0, 1], 4, 20);
    let mut h2 = Field::new(6);
    h2.nest(sub_a(2.0), &[0], 3, 20); // a DIFFERENT weight — same structure, so the same prime

    let g = h1.gcs(&h2);
    assert_eq!(
        g,
        vec![ka],
        "gcs must be exactly the shared subtree (sub_a)"
    );
    assert!(!h1.coprime(&h2), "sharing a subtree ⇒ NOT coprime");

    // h3 = {sub_b}; h4 = {sub_a}. Disjoint substructure ⇒ coprime, empty gcs.
    let mut h3 = Field::new(6);
    h3.nest(sub_b(), &[0, 1], 4, 20);
    let mut h4 = Field::new(6);
    h4.nest(sub_a(1.0), &[0], 3, 20);
    assert!(h3.gcs(&h4).is_empty(), "disjoint subtrees ⇒ empty gcs");
    assert!(h3.coprime(&h4), "disjoint subtrees ⇒ coprime (independent)");
}

#[test]
fn valuation_reports_nesting_depth_and_multiplicity() {
    let ka = sub_a(1.0).structure_key();

    // A 3-level tower: top ▷ mid ▷ sub_a. sub_a sits at depth 2 within top; mid at depth 1.
    let mut mid = Field::new(3);
    mid.nest(sub_a(1.0), &[0], 2, 20);
    let kmid = mid.structure_key();
    let mut top = Field::new(4);
    top.nest(mid, &[0], 3, 20);

    assert_eq!(
        top.valuation(ka),
        (2, 1),
        "sub_a occurs once, at nesting depth 2"
    );
    assert_eq!(
        top.valuation(kmid),
        (1, 1),
        "mid occurs once, at nesting depth 1"
    );
    // An absent structure has valuation (0, 0).
    assert_eq!(top.valuation(sub_b().structure_key()), (0, 0));

    // MULTIPLICITY: two immediate copies of the same prime count twice, at depth 1.
    let mut twin = Field::new(6);
    twin.nest(sub_a(1.0), &[0], 3, 20);
    twin.nest(sub_a(7.0), &[1], 4, 20); // same shape, different weight ⇒ same prime
    assert_eq!(
        twin.valuation(ka),
        (1, 2),
        "two copies of sub_a ⇒ multiplicity 2 at depth 1"
    );
    assert_eq!(
        twin.factorization(),
        vec![ka, ka],
        "factorization lists the prime with multiplicity"
    );
}

#[test]
fn fan_in_and_flags_move_the_key_not_just_cell_count() {
    // DECONFOUND cell-count from the mechanism: `sub_a` vs `sub_b` above differ in BOTH cell-count AND fan-in,
    // so a degenerate cell-count-only key would pass those. These pin that the OTHER terms do real work.

    // Same cell count (3), DIFFERENT fan-in → different key (the fan-in term is load-bearing).
    let fan1 = {
        let mut s = Field::new(3);
        s.add_unit(vec![0], vec![1.0], 0.0, 2);
        s
    };
    let fan2 = {
        let mut s = Field::new(3);
        s.add_unit(vec![0, 1], vec![1.0, 1.0], 0.0, 2);
        s
    };
    assert_eq!(fan1.len(), fan2.len(), "same cell count (control)");
    assert_ne!(
        fan1.structure_key(),
        fan2.structure_key(),
        "fan-in must move the key at EQUAL cell count"
    );

    // Same shape (3 cells, one fan-in-1 unit), DIFFERENT capability flags → different key.
    let plain = {
        let mut s = Field::new(3);
        s.add_unit(vec![0], vec![1.0], 0.0, 2);
        s
    };
    let gated = {
        let mut s = Field::new(3);
        s.add_unit(vec![0], vec![1.0], 0.0, 2);
        s.set_gate(&[2], 1);
        s
    };
    let acted = {
        let mut s = Field::new(3);
        s.add_unit(vec![0], vec![1.0], 0.0, 2);
        s.set_activation(2, Some(Activation::Relu));
        s
    };
    assert_ne!(
        plain.structure_key(),
        gated.structure_key(),
        "the GATE flag must move the key"
    );
    assert_ne!(
        plain.structure_key(),
        acted.structure_key(),
        "the ACTIVATION flag must move the key"
    );
    assert_ne!(
        gated.structure_key(),
        acted.structure_key(),
        "different flags → different keys"
    );
}

#[test]
fn structure_key_is_syntactic_the_collision_is_documented() {
    // TRUTH-NOT-TROPHY: structure_key records fan-in COUNT + flags + cell-count, NOT which cells are read. Two
    // DIFFERENT wirings with identical (cell-count, fan-in, flags) COLLIDE — the syntactic-only limit that
    // `gcs`/`coprime`/`valuation` inherit and that the module docs admit. This test PINS the limit so it can
    // never silently widen into a false-distinctness bug (and documents WHY behaviour uses `embed`, not this).
    let reads0 = {
        let mut s = Field::new(3);
        s.add_unit(vec![0], vec![1.0], 0.0, 2);
        s
    };
    let reads1 = {
        let mut s = Field::new(3);
        s.add_unit(vec![1], vec![1.0], 0.0, 2); // reads cell 1, not 0 — a different WIRING, same SHAPE
        s
    };
    assert_eq!(
        reads0.structure_key(),
        reads1.structure_key(),
        "structure_key is SHAPE (fan-in count), not wiring — the documented syntactic collision"
    );
}

/// A third leaf shape, distinct from `sub_a`/`sub_b`: 4 cells, one unit reading 3 cells.
fn sub_c() -> Field {
    let mut s = Field::new(4);
    s.add_unit(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 0.0, 3);
    s
}
/// One binary composition node: a fixed 6-cell host with `p` then `q` nested with the same reads/writes shape, so
/// the only thing that can differ between two bracketings is the sub-trees themselves.
fn pair(p: Field, q: Field) -> Field {
    let mut h = Field::new(6);
    h.nest(p, &[0, 1], 3, 20);
    h.nest(q, &[0, 1], 4, 20);
    h
}

#[test]
fn structure_key_is_an_unordered_tree_key_bracketing_collides_only_when_isomorphic() {
    // INTENDED SEMANTICS (pinned; checked against the Figueiredo note's L4 side finding): a morph's units are a
    // SET (async, fire-on-change; insertion order carries no meaning), so `structure_key` is the canonical key of
    // the UNORDERED rooted composition tree. There is no planar order for it to see.

    // (1) Sibling insertion order never moves the key: P·Q == Q·P.
    assert_eq!(
        pair(sub_a(1.0), sub_b()).structure_key(),
        pair(sub_b(), sub_a(1.0)).structure_key(),
        "sibling insertion order must not move the key (units are a set)"
    );

    // (2) IDENTICAL-shape leaves: (AB)C and A(BC) are the SAME unordered tree {{a,a},a}. They collide, and that
    // is correct: nothing in the unordered tree tells them apart.
    let ab_c = pair(pair(sub_a(1.0), sub_a(2.0)), sub_a(3.0));
    let a_bc = pair(sub_a(1.0), pair(sub_a(2.0), sub_a(3.0)));
    assert_eq!(
        ab_c.structure_key(),
        a_bc.structure_key(),
        "identical-shape leaves: (AB)C and A(BC) are isomorphic unordered trees, so they share a key"
    );

    // (3) DISTINCT-shape leaves: (AB)C = {{A,B},C} and A(BC) = {A,{B,C}} are NON-isomorphic unordered trees, and
    // the key separates them — the sort is per level, inside the recursion, so it never flattens the nesting.
    let ab_c = pair(pair(sub_a(1.0), sub_b()), sub_c());
    let a_bc = pair(sub_a(1.0), pair(sub_b(), sub_c()));
    assert_ne!(
        ab_c.structure_key(),
        a_bc.structure_key(),
        "distinct leaves: (AB)C and A(BC) are different trees and must get different keys"
    );
    // … and the bracketing is visible to the factor algebra: no immediate factor is shared.
    assert_ne!(ab_c.factorization(), a_bc.factorization());
    assert!(
        ab_c.coprime(&a_bc),
        "the two bracketings share no immediate factor"
    );
}

#[test]
fn structure_key_nested_token_carries_host_fan_in_and_capability_flags() {
    // A NESTED unit is a unit of the host like any leaf: its token must carry the same shape facts a leaf token
    // does (host fan-in + gate/factor/activation flags), not only the sub-morph's key. Regression: the nested
    // token was `fnv(NESTED_TAG, sub_key)` alone, so both pairs below collided.

    // (1) Host fan-in: the same sub nested reading one host cell vs three.
    let mut narrow = Field::new(6);
    narrow.nest(sub_a(1.0), &[0], 4, 20);
    let mut wide = Field::new(6);
    wide.nest(sub_a(1.0), &[0, 1, 2], 4, 20);
    assert_ne!(
        narrow.structure_key(),
        wide.structure_key(),
        "a nested unit's host fan-in is shape and must move the key"
    );

    // (2) Forward factor: the nested output multiplied by a live cell (the flag that changes what it computes).
    let mut unscaled = Field::new(6);
    unscaled.nest(sub_a(1.0), &[0], 4, 20);
    let mut scaled = Field::new(6);
    scaled.nest(sub_a(1.0), &[0], 4, 20);
    scaled.set_factor(4, Some(5));
    assert_ne!(
        unscaled.structure_key(),
        scaled.structure_key(),
        "a nested unit's forward factor is shape and must move the key"
    );

    // (3) Activation flag — inert on a nested cell at runtime, hashed for leaf/nested parity: the same nest with vs
    // without `set_activation(Relu)` on the nested cell.
    let mut plain = Field::new(6);
    plain.nest(sub_a(1.0), &[0], 4, 20);
    let mut relu = Field::new(6);
    relu.nest(sub_a(1.0), &[0], 4, 20);
    relu.set_activation(4, Some(Activation::Relu));
    assert_ne!(
        plain.structure_key(),
        relu.structure_key(),
        "a nested unit's activation override is shape and must move the key"
    );

    // (4) Extra host outputs (`nest_multi`): the same two-sink sub nested writing one host cell vs two. The second
    // write changes what the host sees (review of the fix: this pair still collided).
    let two_sinks = || {
        let mut s = Field::new(3);
        s.add_unit(vec![0], vec![1.0], 0.0, 1);
        s.add_unit(vec![0], vec![-1.0], 0.0, 2);
        s
    };
    let mut one_out = Field::new(6);
    one_out.nest_multi(two_sinks(), &[0], &[4], 20);
    let mut two_out = Field::new(6);
    two_out.nest_multi(two_sinks(), &[0], &[4, 5], 20);
    two_out.stimulate(&[(0, 0.7)], 20);
    assert!(
        two_out.cells()[5].abs() > 0.1,
        "the second output really reaches the host"
    );
    assert_ne!(
        one_out.structure_key(),
        two_out.structure_key(),
        "a nested unit's extra host outputs are shape and must move the key"
    );

    // The sub-morph's own key (its prime factor) is untouched: both hosts still factorize to the same sub.
    assert_eq!(narrow.factorization(), wide.factorization());
    assert_eq!(plain.factorization(), relu.factorization());
    assert_eq!(one_out.factorization(), two_out.factorization());
}
