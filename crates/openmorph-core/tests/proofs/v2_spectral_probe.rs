// SPDX-License-Identifier: MIT-0
//! v2.0-dev · M1 · the ρ SELF-DIAGNOSTIC — spectral radius of `J = D·W` by power iteration
//! (`Field::spectral_radius`). Validates the primitive AND the pre-kickoff design-review corrections:
//!   • E1/E2 — measure ρ on the settling **transient**, not the saturated fixed point (`m→±1 ⇒ D→0 ⇒ ρ→0`
//!     trivially), and note fire-on-change is Gauss–Seidel so `ρ(D·W)<1` is a conservative proxy;
//!   • ρ is a **recurrence** property — a feedforward (acyclic) morph is nilpotent ⇒ ρ = 0.
//! The clean test bed is a recurrent LATCH `m ← tanh(w·m)`: its origin (transient) ρ equals `|w|` and crosses
//! 1 exactly when the loop turns bistable (a memory forms), while the stable fixed point always reads ρ<1 —
//! so the memory-forming instability is visible ONLY on the transient. This is H-S1 (corrected) + H-R1 in one.
use openmorph_core::substrate::Field;

use crate::common::Rng;

/// A recurrent self-loop on cell 1 (memory), cued by cell 0: `memory = tanh(w_self·self + w_cue·cue)`.
fn latch(w_cue: f32, w_self: f32) -> Field {
    let mut f = Field::new(2);
    f.add_unit(vec![0, 1], vec![w_cue, w_self], 0.0, 1);
    f
}

/// A grown FEEDFORWARD morph (2 inputs → 1 output, hidden random features) — acyclic ⇒ nilpotent ⇒ ρ≈0.
fn feedforward(seed: u64) -> Field {
    let mut f = Field::new(3);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().feature_lr = 0.3;
    f.config_mut().max_hidden = 24;
    let mut rng = Rng::new(seed);
    for _ in 0..6000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.grow(&[(0, x0), (1, x1)], 2, (2.0 * x0 * x1).tanh(), 0.05, 100);
    }
    f
}

#[test]
fn rho_is_a_transient_recurrence_property() {
    // ── FACT 1 — feedforward ⇒ nilpotent ⇒ ρ ≈ 0 (settling is evaluation, not contraction). ──
    let ff = feedforward(0x51);
    let rho_ff = ff.spectral_radius(80);
    eprintln!(
        "[SPECTRAL] feedforward grown morph ({} units): ρ = {rho_ff:.4}  (expect ≈0 — nilpotent)",
        ff.units.len()
    );
    assert!(
        rho_ff < 0.05,
        "a feedforward (acyclic) morph is nilpotent ⇒ ρ≈0, got {rho_ff:.4}"
    );

    // ── FACT 2 — the recurrent latch: transient (origin) ρ vs fixed-point ρ (review E1/E2, H-S1). ──
    eprintln!(
        "[SPECTRAL] latch m←tanh(w·m):   origin ρ (transient, D=1)  vs  settled ρ (fixed point):"
    );
    let mut saw_bistable_hidden_by_fixed_point = false;
    for &w in &[0.5f32, 0.9, 1.5, 3.0] {
        // ORIGIN / transient: cells at 0 ⇒ D=1 ⇒ ρ should equal |w| (the self-loop eigenvalue).
        let rho_origin = latch(0.3, w).spectral_radius(64);

        // FIXED POINT: nudge off 0 via the cue, relax (cue=0) to a settled state, measure ρ there.
        let mut f = latch(0.3, w);
        f.stimulate(&[(0, 0.02)], 32); // tiny symmetry-breaking cue (w_cue=0.3 lets it bite)
        for _ in 0..8 {
            f.stimulate(&[(0, 0.0)], 64); // hold: iterate m←tanh(w·m) to its fixed point
        }
        let m_star = f.cells()[1];
        let rho_fixed = f.spectral_radius(64);

        eprintln!(
            "[SPECTRAL]   w={w:>4.1}:  origin ρ={rho_origin:>5.2} (≈|w|)   settled m*={m_star:>6.3}   fixed-point ρ={rho_fixed:>5.3}"
        );

        // (a) the primitive is CORRECT: origin ρ = |w|.
        assert!(
            (rho_origin - w.abs()).abs() < 0.06,
            "origin ρ must equal |w|={:.2}, got {rho_origin:.2}",
            w.abs()
        );
        // (b) E2: a STABLE fixed point is always sub-critical (ρ<1) — trivially, regardless of w.
        assert!(
            rho_fixed < 1.0,
            "a stable fixed point is sub-critical (ρ<1), got {rho_fixed:.3} at w={w}"
        );
        // (c) the point of it all: for w>1 the loop is BISTABLE (a memory), origin ρ>1 — yet the
        //     fixed-point ρ<1 HIDES that. You must measure the transient.
        if w > 1.0 {
            assert!(
                rho_origin > 1.0 && m_star.abs() > 0.3,
                "w={w}: origin should be super-critical (ρ>1) and the latch bistable (|m*|>0.3), got ρ={rho_origin:.2}, m*={m_star:.3}"
            );
            saw_bistable_hidden_by_fixed_point = true;
        }
    }
    assert!(saw_bistable_hidden_by_fixed_point);

    eprintln!("[SPECTRAL] VERDICT: ρ is a RECURRENCE property (feedforward ρ≈0). A stable fixed point is ALWAYS sub-critical (ρ<1), so the fixed-point ρ is trivial/misleading — the memory-forming instability (origin ρ=|w|>1 ⇒ bistable) shows ONLY on the transient. Confirms review E1/E2: measure ρ off the saturated fixed point. Primitive correct: origin ρ=|w| to ±0.06.");
}
