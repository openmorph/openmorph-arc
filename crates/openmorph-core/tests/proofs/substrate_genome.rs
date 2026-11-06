// SPDX-License-Identifier: MIT-0
//! THE FRACTAL GENOME IS REAL (registry C3).
//!
//! A `Field` owns a `Config` — its GENOME, the laws it runs under. A nested sub-morph is itself a `Field`
//! (via `encapsulate`), so the design claim (`physics-architecture.md`: physics = laws; `one-recursive-type.md`:
//! sub-morphs are morphs) is that a sub-morph can carry a DIFFERENT genome and compute/learn under it.
//!
//! This proof is FALSIFIABLE against the exact bug the ONNX note warns about ("or the skill would silently
//! revert to the genome's default"): if `encapsulate` dropped the parent `Config` and the sub reverted to
//! `Config::default()`, both halves below would fail:
//!   (a) INHERITANCE   — an encapsulated Field's `Config` equals the PARENT's, field-for-field, not the default.
//!   (b) DIFFERENT GENOME → DIFFERENT BEHAVIOUR — a nested sub-morph settles under ITS OWN genome, so a
//!       `Relu` sub and a `Tanh` sub produce the measurably-different outputs their laws imply; a sub whose
//!       genome MATCHES the host behaves identically to the standalone case (the control).

use openmorph_core::{Activation, Config, Field};

/// (a) `encapsulate` carries the parent's whole genome — not a fresh default.
#[test]
fn encapsulate_inherits_the_parent_genome() {
    // A genome that differs from the default in several places: a behaviour-affecting categorical
    // (`activation`) AND numeric levers (`feature_lr`, `birth_gate`, `settle_eps`).
    let cfg = Config {
        activation: Activation::Relu,
        feature_lr: 0.42,
        birth_gate: 0.137,
        settle_eps: 5e-4,
        ..Config::default()
    };
    assert_ne!(
        cfg,
        Config::default(),
        "the test genome must actually differ from the default"
    );

    let mut parent = Field::new(3);
    parent.configure(cfg.clone());
    parent.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);

    let sub = parent.encapsulate(&[2]);

    // FIELD-EQUALITY, not "is default": the sub runs under the parent's exact laws.
    assert_eq!(
        sub.config(),
        &cfg,
        "the sub-morph must inherit the parent's full Config"
    );
    assert_eq!(sub.config(), parent.config(), "sub genome == parent genome");
    // Falsifiable: a revert-to-default `encapsulate` would make this fail.
    assert_ne!(
        sub.config(),
        &Config::default(),
        "the sub must NOT revert to the default genome"
    );
}

/// A one-input leaf sub-morph `out = activation(x)` under the given genome. Cell 0 = x, cell 1 = out.
fn leaf(act: Activation) -> Field {
    let mut s = Field::new(2);
    s.config_mut().activation = act;
    // A bare affine unit (activation = None) so it defers to the FIELD's genome — the whole point.
    s.add_unit(vec![0], vec![1.0], 0.0, 1);
    s
}

/// Settle `sub` as a nested cell of a host at input `x`, return the nested cell's value.
fn nested_out(sub: Field, x: f32) -> f32 {
    let mut host = Field::new(3);
    host.nest(sub, &[0], 2, 20);
    host.stimulate(&[(0, x)], 20);
    host.cells()[2]
}

/// (b) A nested sub-morph settles under ITS OWN genome — different genome, different behaviour.
#[test]
fn nested_submorph_runs_under_its_own_genome() {
    let x = -0.7f32; // a NEGATIVE pre-activation: the point where Relu and Tanh diverge hardest.

    let relu = nested_out(leaf(Activation::Relu), x);
    let tanh = nested_out(leaf(Activation::Tanh), x);

    // Relu clips the negative to exactly 0; Tanh passes tanh(-0.7) ≈ -0.604. Different laws, different answers.
    assert!(
        relu.abs() < 1e-6,
        "Relu genome must clip x<0 to 0, got {relu}"
    );
    assert!(
        (tanh - (-0.7f32).tanh()).abs() < 1e-4,
        "Tanh genome must pass tanh(x), got {tanh}"
    );
    assert!(
        (relu - tanh).abs() > 0.5,
        "the two genomes must produce measurably different behaviour"
    );

    // CONTROL: a nested sub whose genome MATCHES the host default (Tanh) behaves exactly like the
    // standalone field — nesting adds no distortion, the difference above is the GENOME, nothing else.
    let mut standalone = leaf(Activation::Tanh);
    standalone.stimulate(&[(0, x)], 20);
    assert!(
        (tanh - standalone.cells()[1]).abs() < 1e-6,
        "a matching-genome nested sub must equal the standalone case"
    );
}

/// (a)+(b) END TO END on the load-bearing path: `encapsulate` a Relu field, then `nest` the encapsulated
/// COPY. It must still clip to 0 — proving the genome survived the encapsulate→nest round-trip. If
/// `encapsulate` reverted to the default Tanh genome, this copy would instead emit tanh(-0.7) ≈ -0.604.
#[test]
fn encapsulated_copy_keeps_its_genome_when_nested() {
    let mut relu = Field::new(2);
    relu.config_mut().activation = Activation::Relu;
    relu.add_unit(vec![0], vec![1.0], 0.0, 1);

    let banked = relu.encapsulate(&[1]); // a "skill" lifted out — must carry Relu with it
    assert_eq!(banked.config().activation, Activation::Relu);

    let out = nested_out(banked, -0.7);
    assert!(
        out.abs() < 1e-6,
        "the encapsulated skill must still run Relu (clip to 0), got {out}"
    );
}
