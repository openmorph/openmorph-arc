// SPDX-License-Identifier: MIT-0
//! NATIVE STREAM MEMORY (#139) — carry (exact) + fractal (trend) on every event, reliance-selected. The
//! substrate now maintains BOTH flavors of temporal memory natively, so a streaming task needs NO hand-built
//! window and NO hand-built cascade — just `advance_delays()` + `grow`.
//!
//! Two carried-state flavors, both native:
//!   - EXACT order — a `delay` is a unit-delay register holding `x[t−1]`; chained, `delay(delay(x))` is
//!     `x[t−2]`, a native shift register carrying the recent EXACT order (`th ≠ ht`). Advanced once per event
//!     by `advance_delays` (a discrete register needs an explicit event boundary the async settle can't give;
//!     `advance_delays` RETURNS the shifted values so a caller feeds them into `grow` as genuine input
//!     changes — no window array, the state lives in native delay cells).
//!   - TREND — the `integrate` trace bank (#137), a leaky-integrator cascade for multi-scale long-range trend.
//!
//! The morph grows one readout over BOTH banks and the reliance economy (#128) SELECTS the flavor per task:
//! exact delays for order/next-symbol, traces for tempo/trend. This unifies exact memory (#122) + fractal
//! trace (#137) + reliance (#128) into one native stream input — `for x in stream { advance; grow(next) }`.
//!
//!   cargo test -p openmorph-core --test proofs substrate_stream -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const X: usize = 0;

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sxy += (x - ma) * (y - mb);
        sxx += (x - ma).powi(2);
        syy += (y - mb).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

/// The native shift register carries the recent EXACT order — `delay_k = x[t−k]`, with no harness window.
#[test]
fn native_shift_register_is_exact() {
    let mut f = Field::new(1);
    let d1 = f.delay(X);
    let d2 = f.delay(d1);
    let d3 = f.delay(d2);
    let seq = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    for (t, &x) in seq.iter().enumerate() {
        f.stimulate(&[(X, x)], 5); // apply the input first (X = x[t])
        let mem = f.advance_delays(); // shift; emits x[t-1..], captures x[t]
        f.stimulate(&mem, 5); // reflect the delayed values in the medium
        let want = |k: usize| if t >= k { seq[t - k] } else { 0.0 };
        assert!(
            (f.cells()[d1] - want(1)).abs() < 1e-6
                && (f.cells()[d2] - want(2)).abs() < 1e-6
                && (f.cells()[d3] - want(3)).abs() < 1e-6,
            "t{t}: ({}, {}, {}) want ({}, {}, {})",
            f.cells()[d1],
            f.cells()[d2],
            f.cells()[d3],
            want(1),
            want(2),
            want(3)
        );
    }
    eprintln!("[STREAM] native shift register EXACT: delay_k = x[t−k], no window");
}

/// Build a morph with BOTH native memories over the input, and grow a readout over both. Returns
/// (field, delay cells, trace cells, out cell).
fn native_memory(lag: usize, max_hidden: usize) -> (Field, Vec<usize>, Vec<usize>, usize) {
    let mut f = Field::new(1);
    f.config_mut().max_hidden = max_hidden;
    let mut d = X;
    let dels: Vec<usize> = (0..lag)
        .map(|_| {
            d = f.delay(d);
            d
        })
        .collect();
    let mut hl = 1.0;
    let trs: Vec<usize> = (0..6)
        .map(|_| {
            let c = f.integrate(X, hl);
            hl *= 2.0;
            c
        })
        .collect();
    let out = f.len();
    f.ensure_cells(out + 1);
    (f, dels, trs, out)
}

/// One native streaming step: apply the input, advance the shift register, and read/grow over BOTH memories.
fn step(f: &mut Field, trs: &[usize], out: usize, x: f32, learn: Option<f32>) {
    f.stimulate(&[(X, x)], 40);
    let mem = f.advance_delays();
    let mut inp = mem;
    for &t in trs {
        inp.push((t, f.cells()[t]));
    }
    match learn {
        Some(tg) => {
            f.grow(&inp, out, tg, 0.1, 80);
        }
        None => {
            f.stimulate(&inp, 80);
        }
    }
}

#[test]
fn native_stream_both_flavors_reliance_selected() {
    // (A) EXACT-ORDER task — deep recall x[t-LAG]. Only the exact delays hold it; an EMA blurs it.
    const LAG: usize = 5;
    let (mut f, dels, trs, out) = native_memory(LAG, 0);
    let mut rng = Rng::new(0x5);
    let mut hist = vec![0.0f32; LAG + 1];
    for _ in 0..16000 {
        let x = if rng.f() < 0.5 { -1.0 } else { 1.0 };
        hist.insert(0, x);
        hist.truncate(LAG + 1);
        step(&mut f, &trs, out, x, Some(hist[LAG] * 0.8)); // predict x[t-LAG]
    }
    let mut rng2 = Rng::new(0x9);
    let mut h2 = vec![0.0f32; LAG + 1];
    let (mut pr, mut tv) = (Vec::new(), Vec::new());
    for _ in 0..3000 {
        let x = if rng2.f() < 0.5 { -1.0 } else { 1.0 };
        h2.insert(0, x);
        h2.truncate(LAG + 1);
        step(&mut f, &trs, out, x, None);
        pr.push(f.cells()[out]);
        tv.push(h2[LAG] * 0.8);
    }
    let exact_corr = corr(&pr, &tv);
    let exact_delay_rel: f32 = dels.iter().map(|&c| f.reliance_of(c)).sum();
    let exact_trace_rel: f32 = trs.iter().map(|&c| f.reliance_of(c)).sum();

    // (B) TREND task — the moving average over ~16 events. Only the trace cascade tracks it; delays can't.
    let (mut g, gdels, gtrs, gout) = native_memory(LAG, 0);
    let mut rng3 = Rng::new(0x3);
    let mut win = vec![0.0f32; 16];
    for _ in 0..16000 {
        let x = if rng3.f() < 0.5 { -1.0 } else { 1.0 };
        win.insert(0, x);
        win.truncate(16);
        let trend = win.iter().sum::<f32>() / 16.0; // long moving average
        step(&mut g, &gtrs, gout, x, Some(trend));
    }
    let mut rng4 = Rng::new(0x7);
    let mut w2 = vec![0.0f32; 16];
    let (mut pr2, mut tv2) = (Vec::new(), Vec::new());
    for _ in 0..3000 {
        let x = if rng4.f() < 0.5 { -1.0 } else { 1.0 };
        w2.insert(0, x);
        w2.truncate(16);
        step(&mut g, &gtrs, gout, x, None);
        pr2.push(g.cells()[gout]);
        tv2.push(w2.iter().sum::<f32>() / 16.0);
    }
    let trend_corr = corr(&pr2, &tv2);
    let trend_delay_rel: f32 = gdels.iter().map(|&c| g.reliance_of(c)).sum();
    let trend_trace_rel: f32 = gtrs.iter().map(|&c| g.reliance_of(c)).sum();

    eprintln!("[STREAM] EXACT task (deep recall x[t-{LAG}], NO window): corr {exact_corr:.2} | reliance delays {exact_delay_rel:.2} vs traces {exact_trace_rel:.2}");
    eprintln!("[STREAM] TREND task (moving average, NO window):        corr {trend_corr:.2} | reliance delays {trend_delay_rel:.2} vs traces {trend_trace_rel:.2}");
    eprintln!("[STREAM] VERDICT (#139): the substrate now carries BOTH flavors of temporal memory NATIVELY, per event, with NO hand-built window and NO hand-built cascade — just `advance_delays()` + `grow`. A `delay` is a native unit-delay register (chained = a shift register) holding the recent EXACT order (`x[t−k]`, `th ≠ ht`); the `integrate` trace bank (#137) holds multi-scale TREND. The morph grows one readout over both, and the SAME reliance economy that self-sizes width (#128) SELECTS the flavor per task: on exact deep-recall the delays win the reliance and the traces are ~0; on a long moving average the traces win and the delays are ~0. This unifies exact memory (#122) + fractal trace (#137) + reliance (#128) into one native stream input — `for x in stream {{ advance_delays(); grow(next) }}`, plug-and-play, no window anywhere.");

    // The right flavor solves each task…
    assert!(
        exact_corr > 0.9,
        "native delays must recall exact order (corr {exact_corr:.2})"
    );
    assert!(
        trend_corr > 0.9,
        "native traces must track the trend (corr {trend_corr:.2})"
    );
    // …and the reliance economy selects it (exact → delays, trend → traces).
    assert!(
        exact_delay_rel > exact_trace_rel + 0.3,
        "the economy must pick the EXACT delays for order (delays {exact_delay_rel:.2} vs traces {exact_trace_rel:.2})"
    );
    assert!(
        trend_trace_rel > trend_delay_rel + 0.3,
        "the economy must pick the TREND traces for the moving average (traces {trend_trace_rel:.2} vs delays {trend_delay_rel:.2})"
    );
}
