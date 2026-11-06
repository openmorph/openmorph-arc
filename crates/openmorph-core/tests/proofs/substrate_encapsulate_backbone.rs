// SPDX-License-Identifier: MIT-0
//! ENCAPSULATE CLOSES THE FEEDING BACKBONE — a banked skill is the SAME function, not a truncated one. A skill's
//! output cell is computed THROUGH intermediate (often grown, caller-invisible) cells; `encapsulate` must pull in
//! that whole feeding backbone or the sub-morph reads cells nothing in it computes and settles to a DIFFERENT
//! function (the reported bug: `encapsulate([output])` kept only the top unit → wrong output, no crash). This is
//! MorphBot's deterministic repro, in-core: a feed chain c0→c1→c2→c3, encapsulate by the OUTPUT cell alone, nest
//! into a fresh host, and assert the fired value reproduces the source EXACTLY.
//!
//!   cargo test -p openmorph-core --test proofs substrate_encapsulate_backbone -- --nocapture

use openmorph_core::substrate::Field;

#[test]
fn encapsulate_pulls_in_the_transitive_backbone() {
    // A feed chain: the skill OUTPUT (cell 3) is computed through a backbone (cells 1, 2) from input cell 0.
    let mut m = Field::new(4);
    m.add_unit(vec![0], vec![3.0], -2.0, 1); // c1 = tanh(3·c0 − 2)   backbone
    m.add_unit(vec![1], vec![2.0], 1.0, 2); // c2 = tanh(2·c1 + 1)   backbone
    m.add_unit(vec![2], vec![1.5], -0.5, 3); // c3 = tanh(1.5·c2 − 0.5) OUTPUT
    m.stimulate(&[(0, 0.9)], 40);
    let true_out = m.cells()[3];

    // Encapsulate by the OUTPUT cell alone — the caller does NOT (and cannot) name the grown backbone.
    let sub = m.encapsulate(&[3]);
    assert_eq!(
        sub.n_units(),
        3,
        "encapsulate([output]) must pull the whole feeding backbone (3 units), not just the top unit"
    );

    // Nest it into a fresh host and fire it: it must reproduce the source's settled output EXACTLY.
    let mut host = Field::new(8);
    host.nest(sub, &[0], 7, 40);
    host.stimulate(&[(0, 0.9)], 40);
    let fired = host.cells()[7];

    eprintln!("[ENCAPSULATE-BACKBONE] source c3 = {true_out:.4}  |  nested skill = {fired:.4}  |  err = {:.2e}", (fired - true_out).abs());
    eprintln!(
        "[ENCAPSULATE-BACKBONE] settles = {} (the skill fired)",
        host.nested_settles()
    );
    assert!(
        (fired - true_out).abs() < 1e-6,
        "a nested skill must reproduce the source EXACTLY (backbone closed): {fired} vs {true_out}"
    );
    assert!(
        host.nested_settles() >= 1,
        "the nested skill must have fired"
    );
}

#[test]
fn encapsulate_a_flat_skill_is_unchanged() {
    // A single-unit skill (output reads inputs directly, no intermediate backbone) must be identical under the
    // closure — backward-compatibility for the common flat case.
    let mut m = Field::new(3);
    m.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2); // c2 = tanh(c0 − c1)
    let sub = m.encapsulate(&[2]);
    assert_eq!(sub.n_units(), 1, "a flat skill stays one unit");

    let mut host = Field::new(4);
    host.nest(sub, &[0, 1], 3, 20);
    host.stimulate(&[(0, 0.6), (1, -0.4)], 20);
    let mut plain = Field::new(3);
    plain.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);
    plain.stimulate(&[(0, 0.6), (1, -0.4)], 20);
    assert!(
        (host.cells()[3] - plain.cells()[2]).abs() < 1e-6,
        "flat-skill encapsulation must be unchanged"
    );
}
