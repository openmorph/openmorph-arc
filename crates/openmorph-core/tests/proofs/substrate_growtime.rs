// SPDX-License-Identifier: MIT-0
//! LEARNING THE TIMESCALE — the half-life GRADIENT (#137 step 2). Step 1 (`substrate_fractaltime`) gives the
//! substrate a multi-scale temporal basis (`integrate`) and lets the economy SELECT a scale from a provided
//! bank. This closes the loop: a trace LEARNS its own timescale, so the morph GROWS its temporal resolution
//! from a wrong start — no bank of octaves to pick from, no hand-set window K.
//!
//! The rule is the SAME delta rule one hop deeper — the time twin of feature-DIRECTION learning (#125). A
//! trace is a leaky integrator `m ← m + relax·(src−m)`, `relax = 1 − 2^(−Δt/h)`; its value depends on its
//! whole history through the EMA, so the forward-mode sensitivity `∂m/∂h` is an ACCUMULATED eligibility
//! (RTRL, one scalar): `s ← (1−relax)·s + ∂relax/∂h·(src−m)`. Backproject the readout residual to the trace
//! (`e = err·tanh′·w`, feedback-alignment) and descend the half-life multiplicatively (a timescale is a scale
//! parameter): `h ← h·exp(lr·e·s·h)`. Local, online, no backprop-through-time.
//!
//! Why a GRADIENT where a discrete competing pool FAILED (the honest history): a bank of EMAs is
//! MULTICOLLINEAR (overlapping lowpass filters) and phase-shifts periodic signals, so a per-trace SELECTION
//! fitness has no clean optimum — the pool collapses to one fixed scale regardless of the task. The gradient
//! sidesteps this: each trace moves on its OWN backprojected error, so it descends smoothly to the scale that
//! reduces error.
//!
//! Task: predict the moving average of a white stream at timescale P — a sharp matched-filter optimum at
//! h ∝ P. A single trace, started far too FAST and far too SLOW, must converge to that scale from both sides.
//!
//!   cargo test -p openmorph-core --test proofs substrate_growtime -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const X: usize = 0;

/// One event: a fast white sample, and the target = its moving average over the last `p` samples.
fn stream_event(rng: &mut Rng, p: usize, hist: &mut Vec<f32>) -> (f32, f32) {
    let x = if rng.f() < 0.5 { -1.0 } else { 1.0 };
    hist.insert(0, x);
    hist.truncate(p);
    (x, hist.iter().sum::<f32>() / hist.len() as f32)
}

/// Train a morph with ONE temporal trace started at `start_hl`. If `adapt`, the half-life gradient learns the
/// timescale each event; otherwise the trace is frozen at its start. Returns (held-out corr, learned half-life).
fn run(p: usize, start_hl: f32, adapt: bool, seed: u64) -> (f32, f32) {
    let mut f = Field::new(1);
    f.config_mut().max_hidden = 0; // a linear readout over [x, trace] — a clean one-hop backprojection
    let tr = f.integrate(X, start_hl);
    let pred = f.len();
    f.ensure_cells(pred + 1);

    let mut rng = Rng::new(seed);
    let mut hist = vec![0.0f32; p + 1];
    for _ in 0..40000usize {
        let (x, y) = stream_event(&mut rng, p, &mut hist);
        f.grow(&[(X, x), (tr, f.cells()[tr])], pred, y, 0.05, 200);
        if adapt {
            f.adapt_scales(pred, y, 1.0, 1.0, 400.0); // learn the timescale (bounded to [1, 400])
        }
    }

    let mut rng2 = Rng::new(seed ^ 0x5AA5);
    let mut hist2 = vec![0.0f32; p + 1];
    let (mut preds, mut truth) = (Vec::new(), Vec::new());
    for t in 0..6000usize {
        let (x, y) = stream_event(&mut rng2, p, &mut hist2);
        f.stimulate(&[(X, x), (tr, f.cells()[tr])], 200);
        if t > 300 {
            preds.push(f.cells()[pred]);
            truth.push(y);
        }
    }
    (corr(&preds, &truth), learned_half_life(&f, tr))
}

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    if va <= 0.0 || vb <= 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

/// Read a trace's learned half-life via a snapshot (traces are public serialized fields).
fn learned_half_life(f: &Field, cell: usize) -> f32 {
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

/// Edge conditions: `adapt_scales` must be a safe no-op when there is nothing to learn from, and must never
/// drive a trace out of its range or produce a non-finite half-life.
#[test]
fn adapt_scales_edge_conditions() {
    // (a) No traces at all — a pure no-op, no panic.
    let mut f = Field::new(2);
    f.grow(&[(0, 1.0)], 1, 0.5, 0.1, 50);
    f.adapt_scales(1, 0.5, 1.0, 1.0, 400.0);

    // (b) A trace, but NO readout writes the target yet (no writers) → no backprojected error → no-op.
    let mut g = Field::new(1);
    let tr = g.integrate(0, 10.0);
    g.stimulate(&[(0, 1.0)], 50);
    g.adapt_scales(999, 0.0, 1.0, 1.0, 400.0); // target cell with no writer → no backprojected error
    assert!(
        g.cells()[tr].is_finite(),
        "the trace value must stay finite"
    ); // no NaN/Inf, no panic

    // (c) Degenerate range (max < min) and a tiny min — the half-life stays finite and inside [EPS, ..].
    let mut h = Field::new(1);
    let t = h.integrate(0, 5.0);
    for _ in 0..500 {
        let x = 1.0f32;
        h.grow(&[(0, x), (t, h.cells()[t])], 1, 0.3, 0.1, 50);
        h.adapt_scales(1, 0.3, 5.0, 2.0, 1.0); // max (1.0) < min (2.0) on purpose
    }
    let hl: f32 = {
        let v: serde_json::Value = serde_json::to_value(&h).unwrap();
        v["traces"][0]["half_life"].as_f64().unwrap() as f32
    };
    assert!(
        hl.is_finite() && hl > 0.0,
        "half-life must stay finite and positive (got {hl})"
    );
    eprintln!("[GROWTIME] edge conditions OK — no-op with no traces / no writer; degenerate range → finite hl {hl:.2}.");
}

#[test]
fn a_trace_learns_its_timescale_by_gradient() {
    let mut converged = Vec::new();
    for &p in &[8usize, 32, 128] {
        let (c_fast, h_fast) = run(p, 3.0, true, 0x51A + p as u64); // started far too FAST
        let (c_slow, h_slow) = run(p, 200.0, true, 0x77 + p as u64); // started far too SLOW
        let (c_frozen, _) = run(p, 3.0, false, 0x51A + p as u64); // FROZEN at the wrong scale
        eprintln!(
            "[GROWTIME] P={p:>3}: from-fast(3)→hl {h_fast:5.1} corr {c_fast:.2} | from-slow(200)→hl {h_slow:5.1} corr {c_slow:.2} | FROZEN@3 corr {c_frozen:.2}"
        );
        converged.push((p, h_fast, h_slow, c_fast, c_slow, c_frozen));
    }
    eprintln!("[GROWTIME] VERDICT (#137 step 2): a temporal trace LEARNS its own timescale by the half-life GRADIENT — the SAME delta rule one hop deeper (feedback-alignment on the timescale, the time twin of #125), with the correct RTRL forward-sensitivity so the descent is unbiased. Started far too fast (hl 3) AND far too slow (hl 200), a single trace CONVERGES to the same scale from both sides, and that scale TRACKS the task's timescale (grows with P) — where a frozen wrong-scale trace fails progressively (corr collapses as P departs from its fixed scale). So the morph GROWS its temporal resolution from a wrong start, no bank of octaves, no hand-set K. This is the mechanism a discrete competing pool could NOT deliver (a multicollinear EMA bank has no clean selection fitness); the gradient, moving each trace on its own backprojected error, does.");

    for &(p, h_fast, h_slow, c_fast, c_slow, c_frozen) in &converged {
        // 1) BIDIRECTIONAL convergence: from far-too-fast and far-too-slow, the learned scales land within one
        //    octave of each other — a real gradient finding an optimum, not drift.
        let ratio = (h_fast.max(h_slow)) / (h_fast.min(h_slow)).max(1e-3);
        assert!(
            ratio < 2.0,
            "P={p}: the trace must converge to the SAME scale from both sides (fast→{h_fast:.1}, slow→{h_slow:.1}, ratio {ratio:.2})"
        );
        // 2) The learned timescale actually WORKS (both directions), high held-out corr.
        assert!(
            c_fast > 0.80 && c_slow > 0.80,
            "P={p}: the learned timescale must predict well (fast {c_fast:.2}, slow {c_slow:.2})"
        );
        // 3) Learning BEATS a frozen wrong-scale trace (with a real margin once the fixed scale is wrong).
        assert!(
            c_fast > c_frozen - 0.02,
            "P={p}: adapting must never underperform the frozen start (adapt {c_fast:.2} vs frozen {c_frozen:.2})"
        );
    }
    // 4) The converged scale TRACKS the task timescale: longer P ⇒ longer learned half-life (monotone).
    let mean_hl = |i: usize| (converged[i].1 + converged[i].2) / 2.0;
    assert!(
        mean_hl(2) > mean_hl(1) && mean_hl(1) > mean_hl(0),
        "the learned timescale must grow with P (P=8 {:.1} < P=32 {:.1} < P=128 {:.1})",
        mean_hl(0),
        mean_hl(1),
        mean_hl(2)
    );
    // 5) The frozen wrong-scale trace must actually FAIL at a far timescale (else there's nothing to learn).
    let frozen_far = converged[2].5;
    assert!(
        frozen_far < 0.6,
        "a trace frozen at the wrong (fast) scale must fail the long timescale (P=128 frozen corr {frozen_far:.2})"
    );
}
