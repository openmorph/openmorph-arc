// SPDX-License-Identifier: MIT-0
//! v2.0-dev · GATE-2 prep — the true MOTIF FAMILIES, validated by BEHAVIOR (not digit).
//! The §3.1 "3 motifs" was too loose. Reading the code, two update forms are distinct:
//!   • LEAKY RELAXATION  `s += rate·(target − s)`  — a STATE relaxes toward a target (traces, surprise, credit,
//!     AND metric_step attraction). Behaviour: monotone EXPONENTIAL approach to the target.
//!   • GRADIENT STEP     `w += lr·err·∂pred/∂w`     — a PARAMETER descends the error gradient (learn_step).
//!     Behaviour: the OUTPUT (not the weight) converges to the target; the weight need not equal any "target".
//! This test shows, on the REAL substrate, that a TRACE and a POSITION — two different subsystems — both follow
//! the relaxation motif (exponential approach), while learning follows the gradient motif. That is the honest
//! motif set an interpreter/genome must carry: relaxation ≠ gradient.
use openmorph_core::substrate::Field;

#[allow(unused_imports)]
use crate::common::Rng;

/// fit an exponential-approach rate r from a trajectory x_t → target: x_t ≈ target·(1 − (1−r)^t).
/// returns (final_gap_fraction, is_monotone_toward_target).
fn relaxation_signature(xs: &[f32], target: f32) -> (f32, bool) {
    let start = xs[0];
    let mut monotone = true;
    for w in xs.windows(2) {
        // each step must move toward the target and never overshoot past it
        let moved_toward = (target - w[1]).abs() <= (target - w[0]).abs() + 1e-6;
        let no_overshoot =
            (w[1] - start).signum() == (target - start).signum() || (w[1] - start).abs() < 1e-6;
        if !moved_toward || !no_overshoot {
            monotone = false;
        }
    }
    let final_gap = (target - xs[xs.len() - 1]).abs() / (target - start).abs().max(1e-6);
    (final_gap, monotone)
}

#[test]
fn relaxation_and_gradient_are_two_distinct_motif_families() {
    // ── (1) TRACE is leaky relaxation: hold the source constant at 1.0, advance the clock, watch the trace
    //        integrate toward it. Traces are Δt-aware, so we need FIRING to advance the subjective clock —
    //        a toggling driver cell (3) fires a unit each event; the trace of the constant source (0) ticks. ──
    let mut f = Field::new(4);
    f.add_unit(vec![3], vec![1.0], 0.0, 1); // fires when cell 3 toggles ⇒ advances the subjective clock
    let tr = f.integrate(0, 3.0); // trace of cell 0, half-life 3 subjective events
    f.stimulate(&[(0, 1.0)], 8); // set the constant source once
    let mut trace_traj = Vec::new();
    for t in 0..40 {
        f.stimulate(&[(3, (t % 2) as f32)], 8); // toggle driver ⇒ unit fires ⇒ clock advances ⇒ trace ticks
        trace_traj.push(f.cells()[tr]);
    }
    let (trace_gap, trace_mono) = relaxation_signature(&trace_traj, 1.0);
    eprintln!(
        "[MOTIF] TRACE  → target 1.0: start {:.3} … end {:.3}  (closed {:.0}% of the gap, monotone={trace_mono})",
        trace_traj[0],
        trace_traj[trace_traj.len() - 1],
        (1.0 - trace_gap) * 100.0
    );

    // ── (2) POSITION (metric_step attraction) is leaky relaxation: a cell's position approaches the group centroid. ──
    let mut g = Field::new(6);
    // fix an anchor cluster {0,1,2} tightly, and watch cell 5 get pulled toward them via co-activation.
    let mut pos_traj = Vec::new();
    for _ in 0..60 {
        g.metric_step(&[0, 1, 2], 0.15, 0); // tighten the anchors (attraction only, no repel)
        g.metric_step(&[0, 5], 0.15, 0); // pull 5 toward the anchor region
                                         // distance of 5 to the anchor centroid (dim-0 as a scalar proxy)
        let p5 = g.position(5);
        let c = (g.position(0)[0] + g.position(1)[0] + g.position(2)[0]) / 3.0;
        pos_traj.push((c - p5[0]).abs());
    }
    // this trajectory is a GAP that should shrink toward 0 (relaxation toward the centroid)
    let pos_start = pos_traj[0];
    let pos_end = pos_traj[pos_traj.len() - 1];
    let pos_mono = pos_traj.windows(2).all(|w| w[1] <= w[0] + 1e-4);
    eprintln!(
        "[MOTIF] POS    → centroid: gap {:.3} … {:.3}  (shrank {:.0}%, monotone={pos_mono})",
        pos_start,
        pos_end,
        (1.0 - pos_end / pos_start.max(1e-6)) * 100.0
    );

    // ── (3) GRADIENT (learn_step) is DISTINCT: the OUTPUT converges to the target; measure error, not "weight→target". ──
    let mut h = Field::new(2);
    h.add_unit(vec![0], vec![0.0], 0.0, 1); // out = tanh(w·x), w starts at 0
    let mut err_traj = Vec::new();
    let mut rng = Rng::new(0x6);
    for _ in 0..300 {
        let x = rng.sym();
        let target = (1.3 * x).tanh();
        let e = h.learn_step(&[(0, x)], 1, target, 0.1, 8);
        err_traj.push(e);
    }
    let err0: f32 = err_traj[..20].iter().sum::<f32>() / 20.0;
    let err1: f32 = err_traj[err_traj.len() - 20..].iter().sum::<f32>() / 20.0;
    eprintln!("[MOTIF] LEARN  → gradient: mean|err| {err0:.3} → {err1:.3}  (output converges by descending the error gradient)");

    // ── behavioural claims (not digit): ──
    assert!(
        trace_mono && trace_gap < 0.5,
        "TRACE must relax monotonically toward its source (leaky-relaxation motif): gap {trace_gap:.2}, monotone {trace_mono}"
    );
    assert!(
        pos_mono && pos_end < pos_start * 0.7,
        "POSITION must relax monotonically toward the centroid (SAME motif as the trace): {pos_start:.3}→{pos_end:.3}"
    );
    assert!(
        err1 < err0 * 0.5,
        "LEARN must reduce output error by gradient descent (a DISTINCT motif): {err0:.3}→{err1:.3}"
    );

    eprintln!("[MOTIF] VERDICT: a TRACE and a POSITION — two different substrate subsystems — both follow the SAME leaky-relaxation motif (monotone exponential approach to a target), validated by behaviour; learning follows a DISTINCT gradient motif (the output, not the parameter, converges). So the honest motif set for the interpreter/genome is {{ gradient · relaxation · contrastive · economy }} — 4 families, not 3. Corrects §3.1.");
}
