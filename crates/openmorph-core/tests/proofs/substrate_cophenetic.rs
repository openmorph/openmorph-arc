// SPDX-License-Identifier: MIT-0
//! COPHENETIC ULTRAMETRIC READ-DISTANCE (#172) — the "ultrametric world", delivered SAFELY: a hierarchy distance
//! over the grown region tree (`index_skills`) for RETRIEVAL, with the substrate's Euclidean LEARNING metric left
//! completely untouched (this is the "keep both worlds" contract — the ultrametric is a read-distance, the
//! Euclidean stays the metric that powers displacement faculties). Truth-not-trophy: the proof asserts the
//! ultrametric PROPERTY (strong triangle inequality; within-region variation COLLAPSES to a region height that
//! raw Euclidean distance does NOT), and that calling it never mutates the concept-map.

use openmorph_core::Field;

/// A concept-map with two well-separated groups (A near the origin, B ~140 away), indexed into a region tree.
fn two_group_index() -> (Field, openmorph_core::SkillIndex) {
    let mut f = Field::new(8);
    // Group A (cells 0..4): a tight cluster near the origin, spread ~1.
    f.place_at(0, &[0.0, 0.0]);
    f.place_at(1, &[1.0, 0.0]);
    f.place_at(2, &[0.0, 1.0]);
    f.place_at(3, &[1.0, 1.0]);
    // Group B (cells 4..8): a tight cluster ~140 away.
    f.place_at(4, &[100.0, 100.0]);
    f.place_at(5, &[101.0, 100.0]);
    f.place_at(6, &[100.0, 101.0]);
    f.place_at(7, &[101.0, 101.0]);
    let index = f.index_skills(&[0, 1, 2, 3, 4, 5, 6, 7], 2, 2);
    (f, index)
}

#[test]
fn cophenetic_is_an_ultrametric_that_collapses_within_region_variation() {
    let (f, index) = two_group_index();

    // Self-distance is 0.
    assert_eq!(f.cophenetic(&index, 0, 0), 0);

    // CROSS-region distance is a single region height, identical for EVERY A×B pair — the ultrametric collapse:
    // all within-region variation is flattened to the height at which the two regions merge.
    let across = f.cophenetic(&index, 0, 4);
    for a in 0..4 {
        for b in 4..8 {
            assert_eq!(
                f.cophenetic(&index, a, b),
                across,
                "every A×B pair shares one cophenetic height (ultrametric collapse), a={a} b={b}"
            );
        }
    }

    // WITHIN-region pairs are strictly closer than cross-region pairs.
    for a in 1..4 {
        assert!(
            f.cophenetic(&index, 0, a) < across,
            "same-region cells (0,{a}) must be closer than cross-region"
        );
    }

    // STRONG TRIANGLE INEQUALITY (the ultrametric property): d(a,c) <= max(d(a,b), d(b,c)) for every triple.
    for a in 0..8 {
        for b in 0..8 {
            for c in 0..8 {
                let (dab, dbc, dac) = (
                    f.cophenetic(&index, a, b),
                    f.cophenetic(&index, b, c),
                    f.cophenetic(&index, a, c),
                );
                assert!(
                    dac <= dab.max(dbc),
                    "ultrametric violated: d({a},{c})={dac} > max(d={dab}, d={dbc})"
                );
            }
        }
    }
}

#[test]
fn cophenetic_collapses_what_raw_euclidean_separates() {
    // TRUTH-NOT-TROPHY: the ultrametric does real work the Euclidean metric can't — it COLLAPSES within-region
    // variation. Raw Euclidean distance 0→4 and 3→4 genuinely DIFFER; cophenetic makes them identical.
    let (f, index) = two_group_index();
    let e04 = f.distance(0, 4, 0);
    let e34 = f.distance(3, 4, 0);
    assert!(
        (e04 - e34).abs() > 1e-3,
        "raw Euclidean distances DO differ within a region ({e04} vs {e34})"
    );
    assert_eq!(
        f.cophenetic(&index, 0, 4),
        f.cophenetic(&index, 3, 4),
        "cophenetic COLLAPSES that variation — a hierarchy distance, not a Euclidean one"
    );
}

#[test]
fn cophenetic_never_touches_the_learning_metric() {
    // The "keep both worlds" contract: the ultrametric read is PURE — it must not perturb the Euclidean concept
    // map that powers the displacement faculties.
    let (f, index) = two_group_index();
    let before: Vec<Vec<f32>> = (0..8).map(|c| f.position(c)).collect();
    for a in 0..8 {
        for b in 0..8 {
            let _ = f.cophenetic(&index, a, b);
        }
    }
    let after: Vec<Vec<f32>> = (0..8).map(|c| f.position(c)).collect();
    assert_eq!(
        before, after,
        "cophenetic is a pure read — the Euclidean learning metric is untouched"
    );
}

#[test]
fn cophenetic_absent_cell_reads_maximally_distant() {
    // The documented edge case: a cell NOT in the index reads as maximally distant (the tree height). Honest
    // caveat — self-distance is therefore NOT 0 for an absent cell (it is also the tree height).
    let (f, index) = two_group_index();
    let root_h = f.cophenetic(&index, 0, 4); // a cross-group pair = the tree height
    assert!(root_h >= 1, "the tree has depth (control)");
    assert_eq!(
        f.cophenetic(&index, 0, 42),
        root_h,
        "an absent cell (42) reads as maximally distant (the tree height)"
    );
    assert_eq!(
        f.cophenetic(&index, 42, 42),
        root_h,
        "absent-vs-itself is the tree height too — self-distance is undefined for an absent cell (honest caveat)"
    );
}

#[test]
fn cophenetic_works_over_a_grown_not_authored_map() {
    // The "emergent" framing, backed: positions here are GROWN by co-activation (metric_step), NOT authored via
    // place_at. Two co-active groups cluster; cophenetic then reflects that grown structure. Deterministic
    // (the field's own seeded PRNG), so this is replay-stable, not flaky.
    let mut f = Field::new(6);
    for _ in 0..600 {
        f.metric_step(&[0, 1, 2], 0.15, 6); // group A co-activates + repels random negatives
        f.metric_step(&[3, 4, 5], 0.15, 6); // group B
    }
    let index = f.index_skills(&[0, 1, 2, 3, 4, 5], 2, 2);
    // Within a grown cluster, cells are no farther apart (cophenetic) than across clusters — from GROWN geometry.
    let within = f.cophenetic(&index, 0, 1);
    let across = f.cophenetic(&index, 0, 3);
    assert!(
        across >= within,
        "grown clusters: cross-cluster cophenetic >= within-cluster (emergent geometry, not authored) — {across} vs {within}"
    );
}
