// SPDX-License-Identifier: MIT-0
//! v2.0-dev · the fitness / health PRIMITIVES for MorphLab's genome search, validated by BEHAVIOR.
//! Core supplies the MORPH-INTRINSIC health terms (a UI/tuner composes them with task generalization):
//!   • `spectral_radius` — STABILITY (settling ρ; sub-critical = well-formed).
//!   • `effective_rank`  — used CAPACITY (participation ratio of unit reliance; ≤ nominal `units.len()`).
//!   • `parsimony`       — nominal size (unit count), the λ term.
//! And `RuleGenome` (= Config) is the plug: `configure(genome)` / `config()`, serde, snapshot round-trip.
use openmorph_core::{Activation, Field, RuleGenome};

use crate::common::Rng;

#[test]
fn fitness_primitives_behave() {
    // ── RuleGenome plugs as data (round-trips through JSON, configures live) ──
    let g = RuleGenome {
        activation: Activation::Softsign,
        lr_fanin_norm: true,
        max_hidden: 17,
        ..RuleGenome::default()
    };
    let json = serde_json::to_string(&g).unwrap();
    let back: RuleGenome = serde_json::from_str(&json).unwrap();
    assert_eq!(g, back, "RuleGenome must round-trip through serde");
    let mut f = Field::new(3);
    f.configure(g.clone());
    assert_eq!(f.config().activation, Activation::Softsign);
    assert!(
        f.config().lr_fanin_norm && f.config().max_hidden == 17,
        "configure(genome) plugs the genome live"
    );

    // ── effective_rank: the PARTICIPATION RATIO of unit reliance, controlled cases ──
    // reliance is DOWNSTREAM: a feature's reliance = the readout's |weight| on it. Build 4 features (each
    // writing its own cell, read by a readout) and vary the readout's weights.
    let make = |weights: &[f32]| -> Field {
        let n = weights.len();
        let mut f = Field::new(n + 2); // cell 0 = input, cells 1..=n = features, cell n+1 = readout
        for i in 0..n {
            f.add_unit(vec![0], vec![1.0], 0.0, 1 + i); // feature i reads input 0, writes cell 1+i
        }
        f.add_unit((1..=n).collect(), weights.to_vec(), 0.0, n + 1); // readout leans on features by weight
        f
    };
    // (A) EQUAL weights ⇒ load spread evenly over 4 features ⇒ effective_rank ≈ 4.
    let er_even = make(&[1.0, 1.0, 1.0, 1.0]).effective_rank();
    // (B) one DOMINANT weight + three negligible ⇒ effective_rank ≈ 1.
    let er_skew = make(&[1.0, 0.01, 0.01, 0.01]).effective_rank();
    eprintln!("[FIT] effective_rank:  4 equal units = {er_even:.2} (≈N=4)   1 dominant + 3 tiny = {er_skew:.2} (≈1)");
    assert!(
        (er_even - 4.0).abs() < 0.1,
        "equal load over 4 units ⇒ effective_rank ≈ 4, got {er_even:.2}"
    );
    assert!(
        er_skew < 1.2,
        "one dominant unit ⇒ effective_rank ≈ 1, got {er_skew:.2}"
    );

    // ── on a GROWN morph: used capacity ≤ nominal size, and it's a real health readout ──
    let mut m = Field::new(3);
    m.config_mut().max_hidden = 32;
    m.config_mut().birth_gate = 0.1;
    m.config_mut().feature_lr = 0.3;
    let mut rng = Rng::new(0xF17);
    for _ in 0..5000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        m.grow(&[(0, x0), (1, x1)], 2, (2.0 * x0 * x1).tanh(), 0.05, 60);
    }
    let (rho, er, par) = (m.spectral_radius(64), m.effective_rank(), m.parsimony());
    eprintln!("[FIT] grown morph HEALTH:  ρ(stability)={rho:.3}   effective_rank(capacity)={er:.2}   parsimony(#units)={par}");
    assert!(
        er <= par as f32 + 1e-3,
        "used capacity (effective_rank {er:.2}) must be ≤ nominal size (parsimony {par})"
    );
    assert!(
        er > 0.5,
        "a working grown morph uses more than one feature: effective_rank {er:.2}"
    );
    assert!(
        rho < 1.0,
        "a feedforward grown morph is nilpotent/stable: ρ {rho:.3}"
    );

    eprintln!("[FIT] VERDICT: the intrinsic fitness primitives behave — RuleGenome (=Config) plugs as serde data via configure(); spectral_radius reads STABILITY (ρ<1), effective_rank reads USED CAPACITY (≤ nominal, ≈N for even load, ≈1 for a dominant unit), parsimony reads nominal size. MorphLab composes these with held-out task generalization: Φ = generalization − λ·parsimony − μ·(1−ρ margin), preferring the smallest stable morph that solves the task.");
}
