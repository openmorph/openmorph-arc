// SPDX-License-Identifier: MIT-0
//! TRANSCRIPTION — a unit can carry its OWN law, so a foreign network can be the same function here.
//!
//! A grown organism has one genome: every unit it births computes `tanh`, and a bounded cell is what
//! keeps a coupled population stable. But a *trained* network is a mixture — a `Relu` hidden layer
//! under a bare affine logit layer — and copying one in is only honest if each unit is the function
//! the source says it is. That is the whole of this addition: two more `Activation` variants and a
//! per-unit override, both `#[serde(default)]`, both invisible unless a caller asks for them.
//!
//! The claims:
//!   (A) UNTOUCHED    — a field with no override is byte-identical to before the override existed.
//!   (B) EXACT        — a hand-built ReLU/Identity network computes exactly its own arithmetic, with
//!                      no bound clipping the logits.
//!   (C) IT LEARNS    — the local delta rule differentiates through each unit's OWN law, including a
//!                      rectifier's dead half.
//!   (D) IT TRAVELS   — the law survives `encapsulate` (a banked skill) and a snapshot round-trip.
//!
//!   cargo test -p openmorph-core --test proofs substrate_transcribe -- --nocapture

use openmorph_core::{Activation, Field};

/// (A) An organism that never asks for an override behaves exactly as it did before overrides existed:
/// the genome's `tanh` still applies, on every path.
#[test]
fn a_field_without_overrides_is_untouched() {
    let mut f = Field::new(3);
    f.add_unit(vec![0, 1], vec![0.5, -0.25], 0.1, 2);
    f.stimulate(&[(0, 1.0), (1, 2.0)], 32);
    let want = (0.5 * 1.0 + -0.25 * 2.0 + 0.1_f32).tanh();
    assert!(
        (f.cells()[2] - want).abs() < 1e-6,
        "the default law changed: {} vs {want}",
        f.cells()[2]
    );

    // And the genome knob still governs every unit that has no opinion of its own.
    f.config_mut().activation = Activation::HardTanh;
    f.stimulate(&[(0, 4.0), (1, 0.0)], 32);
    assert!(
        (f.cells()[2] - 1.0).abs() < 1e-6,
        "the genome no longer governs an un-overridden unit"
    );
}

/// (B) A transcribed network: two `Relu` cells under an `Identity` readout. The logit must be the plain
/// arithmetic — unbounded, unclipped — or an imported model would silently saturate.
#[test]
fn a_relu_network_computes_its_own_arithmetic() {
    let mut f = Field::new(0);
    f.ensure_cells(4);
    f.add_unit(vec![0], vec![2.0], -0.5, 1); // h1 = relu(2x - 0.5)
    f.set_activation(1, Some(Activation::Relu));
    f.add_unit(vec![0], vec![-3.0], 1.0, 2); // h2 = relu(-3x + 1)
    f.set_activation(2, Some(Activation::Relu));
    f.add_unit(vec![1, 2], vec![4.0, 5.0], 0.25, 3); // y  = 4h1 + 5h2 + 0.25
    f.set_activation(3, Some(Activation::Identity));

    for x in [-2.0f32, -0.1, 0.0, 0.4, 3.0, 7.5] {
        f.stimulate(&[(0, x)], 64);
        let (h1, h2) = ((2.0 * x - 0.5).max(0.0), (-3.0 * x + 1.0).max(0.0));
        let want = 4.0 * h1 + 5.0 * h2 + 0.25;
        assert!((f.cells()[1] - h1).abs() < 1e-6, "relu h1 at x={x}");
        assert!((f.cells()[2] - h2).abs() < 1e-6, "relu h2 at x={x}");
        assert!(
            (f.cells()[3] - want).abs() < 1e-5,
            "logit at x={x}: {} vs {want}",
            f.cells()[3]
        );
    }
    // The point of `Identity`: a logit of 60 stays 60. `tanh` would have returned 1.0.
    f.stimulate(&[(0, 7.5)], 64);
    assert!(
        f.cells()[3] > 50.0,
        "the readout was bounded — an imported logit would be destroyed"
    );
}

/// (C) The delta rule differentiates through each unit's OWN law. An `Identity` readout learns with
/// slope 1 (plain LMS), and a `Relu` unit sitting in its dead half receives nothing — the rectifier's
/// zero gradient, read off the output value like every other variant.
#[test]
fn learning_follows_each_units_own_law() {
    // Identity readout: one step moves the weight by exactly lr·err·x.
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![0.0], 0.0, 1);
    f.set_activation(1, Some(Activation::Identity));
    f.learn_step(&[(0, 2.0)], 1, 10.0, 0.01, 32);
    let w = f.units[0].weights[0];
    assert!(
        (w - 0.01 * 10.0 * 2.0).abs() < 1e-6,
        "identity slope is not 1: w={w}"
    );

    // A Relu unit whose output is 0 has zero slope, so no update reaches it.
    let mut g = Field::new(2);
    g.add_unit(vec![0], vec![1.0], -5.0, 1); // relu(x - 5) = 0 at x = 1
    g.set_activation(1, Some(Activation::Relu));
    let before = g.units[0].weights[0];
    g.learn_step(&[(0, 1.0)], 1, 3.0, 0.5, 32);
    assert!(
        (g.units[0].weights[0] - before).abs() < 1e-9,
        "a dead rectifier still moved: the derivative is not being read from its own law"
    );
    // Alive, it learns.
    g.learn_step(&[(0, 9.0)], 1, 6.0, 0.01, 32);
    assert!(
        g.units[0].weights[0] != before,
        "a live rectifier did not learn"
    );
}

/// (E) THE SECOND-ORDER TERM — a unit can emit the PRODUCT of two live cells.
///
/// A leaf unit is affine in the cells (`Σ w·x`, weights fixed), so no unit could ever multiply two
/// values the organism is currently holding. That is the one thing standing between this substrate and
/// attention, gated linear units, binding and gain modulation. `Unit::factor` closes it by reusing the
/// shape `gate` already had: a live cell that modulates what the unit does — `gate` on what it writes
/// into its WEIGHTS, `factor` on what it writes into its CELL.
#[test]
fn a_unit_can_multiply_two_live_cells() {
    let mut f = Field::new(3);
    f.add_unit(vec![0], vec![1.0], 0.0, 2); // out = identity(x) …
    f.set_activation(2, Some(Activation::Identity));
    f.set_factor(2, Some(1)); // … times cell 1 — one unit, one product

    for (x, y) in [(3.0f32, 4.0f32), (-2.0, 0.5), (0.0, 9.0), (1.5, -6.25)] {
        f.stimulate(&[(0, x), (1, y)], 64);
        assert!(
            (f.cells()[2] - x * y).abs() < 1e-5,
            "{x}·{y} came out {}",
            f.cells()[2]
        );
    }

    // A factor pinned to zero silences the unit — and, because nothing it does can matter, no gradient
    // reaches it either. (This is exactly why `factor` is NOT `gate`: a write-gated register is read
    // while its gate is shut, so gating the OUTPUT would destroy the value it holds.)
    f.stimulate(&[(0, 5.0), (1, 0.0)], 64);
    assert_eq!(f.cells()[2], 0.0, "a zero factor did not silence the unit");
    let before = f.units[0].weights[0];
    f.learn_step(&[(0, 5.0), (1, 0.0)], 2, 7.0, 0.5, 64);
    assert!(
        (f.units[0].weights[0] - before).abs() < 1e-9,
        "a silenced unit still learned — the factor is missing from the chain rule"
    );

    // With the factor open it learns, and the factor is IN the gradient: the same error moves the
    // weight twice as far at factor 2 as at factor 1.
    let step = |k: f32| {
        let mut g = Field::new(3);
        g.add_unit(vec![0], vec![0.0], 0.0, 2);
        g.set_activation(2, Some(Activation::Identity));
        g.set_factor(2, Some(1));
        g.learn_step(&[(0, 1.0), (1, k)], 2, 1.0, 0.01, 64);
        g.units[0].weights[0]
    };
    let (s1, s2) = (step(1.0), step(2.0));
    assert!(
        (s2 / s1 - 2.0).abs() < 1e-3,
        "the factor is not in the chain rule: {s1} vs {s2}"
    );
}

/// (D) A banked skill keeps the law it was grown under. `encapsulate` copies units into a sub-morph, so
/// if the per-unit law did not travel with them, a lifted skill would silently revert to `tanh` and
/// compute something else than it did inside its parent.
#[test]
fn the_law_travels_with_the_skill_and_the_snapshot() {
    let mut f = Field::new(0);
    f.ensure_cells(3);
    f.add_unit(vec![0], vec![3.0], 0.0, 1);
    f.set_activation(1, Some(Activation::Relu));
    f.add_unit(vec![1], vec![2.0], 0.0, 2);
    f.set_activation(2, Some(Activation::Identity));
    f.stimulate(&[(0, 4.0)], 32);
    let inside = f.cells()[2];
    assert!(
        (inside - 24.0).abs() < 1e-5,
        "expected 2·relu(3·4) = 24, got {inside}"
    );

    // As a nested skill in a blank host.
    let sub = f.encapsulate(&[1, 2]);
    let mut host = Field::new(2);
    host.nest(sub, &[0], 1, 256);
    host.stimulate(&[(0, 4.0)], 256);
    assert!(
        (host.cells()[1] - inside).abs() < 1e-5,
        "the nested skill answers {} where it answered {inside} at home",
        host.cells()[1]
    );

    // And through a snapshot.
    let json = serde_json::to_string(&f).expect("serialize");
    let mut back: Field = serde_json::from_str(&json).expect("deserialize");
    back.stimulate(&[(0, 5.0)], 32);
    f.stimulate(&[(0, 5.0)], 32);
    assert!(
        (back.cells()[2] - f.cells()[2]).abs() < 1e-6,
        "the law did not survive the snapshot"
    );

    // A snapshot written BEFORE per-unit laws existed carries no such key at all. Strip it back out of
    // a real snapshot and the organism must still load, with every unit deferring to the genome.
    let mut plain = Field::new(2);
    plain.add_unit(vec![0], vec![1.0], 0.0, 1); // no override — the genome's law, as an old field had
    let plain_json = serde_json::to_string(&plain).expect("serialize");
    let legacy_json = plain_json
        .replace(",\"activation\":null", "")
        .replace("\"activation\":null,", "");
    // (The genome's own `activation` key stays — it is the field that is being stripped from the UNITS.)
    assert!(
        !legacy_json.contains("\"activation\":null"),
        "the test did not actually strip the field it claims to strip"
    );
    let legacy: Field =
        serde_json::from_str(&legacy_json).expect("a pre-override snapshot must still load");
    assert!(
        legacy.units.iter().all(|u| u.activation.is_none()),
        "an old snapshot gained an opinion it never had"
    );
}
