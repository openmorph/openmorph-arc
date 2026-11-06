// SPDX-License-Identifier: MIT-0
//! The v1.3.1-era buried constants become Config LEVERS (focused audit follow-up to `substrate_backbone_lever`).
//! The last Config sweep predated #137 (`adapt_scales`) and #122 (`bind`), so two genuine tuning levers were
//! still hard-coded literals in that newer code. This proof asserts they are now `Config` fields — default =
//! historical, in `Config::schema()` — AND that each measurably shifts behaviour (truth-not-trophy: a lever is
//! only worth surfacing if moving it moves the organism). The audit's "stays a const" literals (the value-field
//! convergence ε, the half-life math `0.5^(dt/hl)`, the fingerprint DFT) are deliberately NOT config — they
//! define a mechanism, they are not knobs.
//!
//!   cargo test -p openmorph-core --test proofs substrate_config_v131 -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use openmorph_core::Config;

/// Read a trace's live half-life via the public serialized `traces` field.
fn half_life(f: &Field, cell: usize) -> f32 {
    let v: serde_json::Value = serde_json::to_value(f).unwrap();
    v.get("traces")
        .and_then(|t| t.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|t| t.get("cell").and_then(|c| c.as_u64()) == Some(cell as u64))
                .and_then(|t| t.get("half_life").and_then(|h| h.as_f64()))
        })
        .map(|h| h as f32)
        .unwrap_or(0.0)
}

#[test]
fn v131_levers_default_historical_and_are_in_schema() {
    let d = Config::default();
    assert_eq!(
        d.adapt_step_cap, 0.2,
        "adapt_step_cap default is historical"
    );
    assert_eq!(d.bind_tol, 1e-3, "bind_tol default is historical");
    for name in ["adapt_step_cap", "bind_tol"] {
        assert!(
            Config::schema().iter().any(|l| l.name == name),
            "{name} must be surfaced in schema() (Optuna-ready)"
        );
    }
}

#[test]
fn bind_tol_controls_the_relation_match() {
    // Rule: cells 0 and 2 covary; cell 1 is a near-but-unequal decoy (|1.0 − 1.4| = 0.4).
    let obs = [(0usize, 1.0f32), (1, 1.4), (2, 1.0)];

    // Default tolerance (1e-3): only the EXACTLY-equal pair (0,2) binds; the decoy stays unbound.
    let mut tight = Field::new(3);
    tight.bind(&obs);
    assert!(tight.bound(0, 2) > 0.0, "the equal pair binds");
    assert_eq!(
        tight.bound(0, 1),
        0.0,
        "a 0.4-apart pair does NOT bind at the tight default"
    );

    // A caller on a noisy/continuous channel raises the tolerance so an approximate match still binds.
    let mut loose = Field::new(3);
    loose.config_mut().bind_tol = 0.5;
    loose.bind(&obs);
    assert!(
        loose.bound(0, 1) > 0.0,
        "at bind_tol = 0.5 the near-equal pair binds too — the lever measurably changes what counts as a relation"
    );

    eprintln!("[BIND_TOL] tight 1e-3 → only equal pairs bind (0.4-apart decoy rejected); loose 0.5 → near-equal binds. One Config lever tunes relation-match strictness for continuous/noisy channels.");
}

#[test]
fn adapt_step_cap_bounds_the_half_life_step() {
    // Build one trace with accumulated sensitivity + a linear readout carrying persistent error, so the
    // half-life gradient is large — then take ONE adapt step from the SAME state under two caps.
    let mut f = Field::new(1);
    f.config_mut().max_hidden = 0; // clean linear one-hop backprojection over [x, trace]
    let tr = f.integrate(0, 20.0);
    let pred = f.len();
    f.ensure_cells(pred + 1);
    // Train on a SQUARE WAVE so the trace keeps chasing — a constant source lets its ∂m/∂h sensitivity decay
    // to zero (nothing to learn from a settled trace). A varying source keeps the gradient alive.
    for k in 0..400 {
        let x = if k % 2 == 0 { 1.0 } else { -1.0 };
        f.stimulate(&[(0, x)], 50);
        f.grow(&[(0, x), (tr, f.cells()[tr])], pred, 0.5 * x, 0.02, 50);
    }

    // A large lr makes the raw log-step land between the two caps, so the tight cap clamps it and the loose
    // cap does not — the difference in |ln(h_new/h0)| is exactly the lever at work.
    let step = |cap: f32| -> f32 {
        let mut g = f.clone();
        g.config_mut().adapt_step_cap = cap;
        // Prime a fresh nonzero sensitivity with two opposite stimulations (a definite source change).
        g.stimulate(&[(0, 1.0)], 50);
        g.stimulate(&[(0, -1.0)], 50);
        let h0 = half_life(&g, tr);
        g.adapt_scales(pred, 0.9, 500.0, 1.0, 1000.0); // large gradient (target far from pred)
        (half_life(&g, tr) / h0).ln().abs()
    };
    let small = step(0.05); // a tight cap the large gradient saturates
    let big = step(1.0); // a loose cap the same gradient does NOT reach

    assert!(
        (small - 0.05).abs() < 1e-3,
        "the tight cap clamps the log-step to itself (got {small}, cap 0.05)"
    );
    assert!(
        big > small + 1e-3,
        "a larger cap permits a larger single-event half-life move ({big} > {small}) — the lever works"
    );
    assert!(big <= 1.0 + 1e-3, "the larger cap still bounds the move");

    eprintln!("[ADAPT_STEP_CAP] one adapt step, same gradient: cap 0.05 → |Δln h| = {small:.3} (clamped), cap 1.0 → {big:.3} (unclamped). The half-life adaptation speed is now one Config lever (was the buried STEP_CAP).");
}
