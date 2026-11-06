// SPDX-License-Identifier: MIT-0
//! NEST BEHAVIOUR-SEEDING (#172) — a nested skill lands on the concept-map by WHAT IT DOES, not by scatter noise.
//! `nest` seeds the new nested cell's position from the sub-morph's own `embed` (its coarse spectral prefix,
//! PROJECTED to the current `pos_dim` — never growing the map), unifying core's positional `route` and the cosine
//! retrieval key onto ONE behaviour signal. Truth-not-trophy: the proof asserts the MECHANISM (position IS the
//! embed projection; a behavioural CLONE seeds to the SAME spot; the flag genuinely changes placement) and is
//! HONEST about the coarse-router limitation (it only separates skills that already differ at the pos_dim scale).

use openmorph_core::Field;

/// A near-linear leaf skill: out = tanh(w·x). Cells 0=in, 1=out.
fn sub_lin(w: f32) -> Field {
    let mut s = Field::new(2);
    s.add_unit(vec![0], vec![w], 0.0, 1);
    s
}
/// A structurally- AND behaviourally-distinct skill: 3 cells, 2 inputs, hard-saturating weights.
fn sub_wide() -> Field {
    let mut s = Field::new(3);
    s.add_unit(vec![0, 1], vec![8.0, 8.0], 0.0, 2);
    s
}

fn proj(embed: &[f32], dims: usize) -> Vec<f32> {
    embed.iter().take(dims).copied().collect()
}

#[test]
fn nest_seeds_the_nested_cell_position_from_behaviour() {
    let s = sub_lin(1.0);
    let key = s.embed(); // the behaviour key

    // seeding ON (default): the nested cell's position IS the embed projection (place_at truncation).
    let mut host = Field::new(4);
    let write = 3;
    host.nest(s.clone(), &[0], write, 20);
    let pos = host.position(write);
    let d = host.pos_dim();
    assert_eq!(
        pos,
        proj(&key, d),
        "nest ON seeds the nested cell's position from sub.embed(), projected to pos_dim"
    );
    assert_eq!(
        d, 2,
        "the map stays 2-D — nest PROJECTS, never grows pos_dim"
    );

    // CLONE-INVARIANCE: a freshly-built behavioural clone seeds to the SAME position (by behaviour, not identity).
    let mut host_clone = Field::new(4);
    host_clone.nest(sub_lin(1.0), &[0], write, 20);
    assert_eq!(
        host_clone.position(write),
        pos,
        "a behavioural clone seeds to the same map position"
    );
}

#[test]
fn the_flag_genuinely_changes_placement() {
    let s = sub_lin(1.0);
    let seeded = {
        let mut h = Field::new(4);
        h.nest(s.clone(), &[0], 3, 20);
        h.position(3)
    };
    let unseeded = {
        let mut h = Field::new(4);
        h.config_mut().nest_seeds_position = false;
        h.nest(s.clone(), &[0], 3, 20);
        h.position(3)
    };
    assert_ne!(
        seeded, unseeded,
        "nest_seeds_position=false leaves the scatter seed — the flag genuinely changes placement"
    );
}

#[test]
fn behaviourally_distinct_skills_seed_apart_a_clone_seeds_together() {
    let a = sub_lin(1.0);
    let b = sub_wide();
    // HONEST LIMITATION CHECK: the coarse pos_dim projection can only separate skills that already differ at that
    // scale — confirm these two do (else the assertion below would be vacuous).
    let d = 2usize;
    assert_ne!(
        proj(&a.embed(), d),
        proj(&b.embed(), d),
        "the two skills must differ at the coarse (pos_dim) behaviour scale for the seed to separate them"
    );

    let mut host = Field::new(6);
    host.nest(a, &[0], 4, 20);
    host.nest(b, &[0, 1], 5, 20);
    assert_ne!(
        host.position(4),
        host.position(5),
        "behaviourally-distinct skills seed to different map positions"
    );

    // The retrieval win: a re-built copy of `a` seeds to `a`'s position (not `b`'s).
    let mut host2 = Field::new(6);
    host2.nest(sub_lin(1.0), &[0], 4, 20);
    assert_eq!(
        host2.position(4),
        host.position(4),
        "a re-built copy of `a` seeds to `a`'s position — behaviour-addressed placement"
    );
}

#[test]
fn placement_is_behaviour_not_the_write_index() {
    // DECONFOUND behaviour from the nested cell's WRITE INDEX (the adversarial gap): the tests above nest at a
    // FIXED index, so equal positions could be an index artifact. Vary index and behaviour INDEPENDENTLY.

    // (1) IDENTICAL behaviour at DIFFERENT write indices → SAME position (behaviour sets the VALUE, not the index).
    let pos_at_3 = {
        let mut h = Field::new(8);
        h.nest(sub_lin(1.0), &[0], 3, 20);
        h.position(3)
    };
    let pos_at_6 = {
        let mut h = Field::new(8);
        h.nest(sub_lin(1.0), &[0], 6, 20);
        h.position(6)
    };
    assert_eq!(
        pos_at_3, pos_at_6,
        "same behaviour → same position REGARDLESS of write index (behaviour beats index)"
    );

    // (2) DIFFERENT behaviour at the SAME write index → DIFFERENT position (behaviour decides, not the index).
    let lin_at_4 = {
        let mut h = Field::new(8);
        h.nest(sub_lin(1.0), &[0], 4, 20);
        h.position(4)
    };
    let wide_at_4 = {
        let mut h = Field::new(8);
        h.nest(sub_wide(), &[0, 1], 4, 20);
        h.position(4)
    };
    assert_ne!(
        lin_at_4, wide_at_4,
        "different behaviour → different position at the SAME write index (behaviour beats index)"
    );
}
