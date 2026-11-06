// SPDX-License-Identifier: MIT-0
//! v2.0-dev · the TEMPORAL-WM wall (GATE-4, openmorph#149) — does a grown short-term memory let a world-model
//! capture DELAYED dynamics that a MEMORYLESS one-step predictor can't (ls20's signature)?
//!
//! Task: the target depends on the input LAG events AGO — `target_t = tanh(2·x[t−LAG])` over an i.i.d. stream, so
//! the CURRENT observation carries no information about it. A memoryless world-model (grows over `x_t`) is at
//! chance; a temporal world-model (grows over the native `delay` shift register #139) reads `x[t−LAG]` directly and
//! captures the delayed dynamics. Validated BY BEHAVIOR: memoryless corr ≈ 0, temporal corr ≈ 1. The answer is the
//! #137/#139 temporal primitives — no new mechanism, just feed `delay`/`integrate` into the WM's context.
use openmorph_core::substrate::Field;

use crate::common::Rng;

const X: usize = 0;
const LAG: usize = 3;

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

/// Grow a world-model to predict `target = tanh(2·x[t−LAG])`. `memory` = wire the native `delay` shift register
/// into the WM's context (temporal WM) vs grow over only the current observation (memoryless one-step WM).
fn wm_corr(memory: bool, seed: u64) -> f32 {
    let mut f = Field::new(1);
    f.config_mut().max_hidden = 16;
    f.config_mut().birth_gate = 0.1;
    f.config_mut().feature_lr = 0.3;
    // the native shift register x[t−1..t−LAG] (only if this is the temporal WM)
    if memory {
        let mut d = X;
        for _ in 0..LAG {
            d = f.delay(d);
        }
    }
    let out = f.len();
    f.ensure_cells(out + 1);

    let target_of = |h: &[f32]| (2.0 * h[LAG]).tanh(); // DELAYED dynamics — depends on x[t−LAG]
    let mut rng = Rng::new(seed);
    let mut hist = vec![0.0f32; LAG + 1]; // hist[0]=x_t … hist[LAG]=x[t−LAG]

    // train
    for _ in 0..8000 {
        let x = rng.sym();
        hist.insert(0, x);
        hist.truncate(LAG + 1);
        let tg = target_of(&hist);
        f.stimulate(&[(X, x)], 40); // apply the observation
        let mem = f.advance_delays(); // shift the register; returns the delayed (cell, value) pairs
        let inp: Vec<(usize, f32)> = if memory { mem } else { vec![(X, x)] };
        f.grow(&inp, out, tg, 0.1, 80);
    }
    // held-out (no learning)
    let mut r2 = Rng::new(seed ^ 0xABCD);
    let mut h2 = vec![0.0f32; LAG + 1];
    let (mut ps, mut ts) = (Vec::new(), Vec::new());
    for _ in 0..600 {
        let x = r2.sym();
        h2.insert(0, x);
        h2.truncate(LAG + 1);
        let tg = target_of(&h2);
        f.stimulate(&[(X, x)], 40);
        let mem = f.advance_delays();
        let inp: Vec<(usize, f32)> = if memory { mem } else { vec![(X, x)] };
        f.stimulate(&inp, 80);
        ps.push(f.cells()[out]);
        ts.push(tg);
    }
    corr(&ps, &ts)
}

#[test]
fn temporal_memory_closes_the_delayed_dynamics_gap() {
    // Firm across seeds (GATE-4 caveat: single seed) — the separation must hold for every seed, not on average only.
    let seeds = [0x7u64, 0x11, 0x2b, 0x5f];
    let (mut sum_mem, mut sum_tmp) = (0.0f32, 0.0f32);
    for &s in &seeds {
        let memoryless = wm_corr(false, s);
        let temporal = wm_corr(true, s);
        eprintln!("[TWM] seed {s:#x}:  memoryless WM corr = {memoryless:.3}   temporal WM (delay) corr = {temporal:.3}");
        assert!(
            memoryless.abs() < 0.2,
            "seed {s:#x}: a MEMORYLESS one-step WM is at chance on a target caused LAG events ago: corr {memoryless:.3}"
        );
        assert!(
            temporal > 0.9,
            "seed {s:#x}: a grown TEMPORAL memory (delay shift register) captures the delayed dynamics: corr {temporal:.3}"
        );
        sum_mem += memoryless;
        sum_tmp += temporal;
    }
    let (memoryless, temporal) = (sum_mem / seeds.len() as f32, sum_tmp / seeds.len() as f32);
    eprintln!("[TWM] delayed dynamics (target = tanh(2·x[t−{LAG}])), mean over {} seeds:  memoryless = {memoryless:.3}   temporal (delay) = {temporal:.3}", seeds.len());

    eprintln!("[TWM] VERDICT: YES — a grown short-term memory closes the delayed-dynamics gap GATE-4 surfaced. The memoryless one-step WM is at chance (corr {memoryless:.2}) on a target whose cause is LAG events in the past; wiring the native `delay` shift register (#139) into the world-model's context lets it read x[t−LAG] and capture the delayed dynamics (corr {temporal:.2}). The #137/#139 temporal primitives ARE the substrate answer to ls20's delayed signature — no new mechanism: feed delay (exact order) / integrate (trend) into the WM context, and the reliance economy selects the flavor per task (substrate_stream).");
}
