// SPDX-License-Identifier: MIT-0
//! #155 (OM-6) R4/R5 — the GROWN surprise accessors + saturation predicate. The ARC-3 harness re-derived the
//! organism's prediction error BY HAND (`morph_discover.rs`, `autonomy_probe.rs`) and hand-thresholded a
//! stage-transition trigger with MAGIC CONSTANTS (`ERR_TRIGGER = 0.45`, `agency > 0.008`) — two authored holes
//! (purity rule #1). The substrate already HAS the signal (the per-cell surprise EMA); this exposes it and grows
//! the trigger from it, RELATIVE to the cell's own history, so nothing is hand-thresholded.
//!
//! - `surprise_of(cell)` / `mean_surprise(heads)` — read the morph's OWN grown error (no recomputation).
//! - `learning_progress(cell)` = `(slow − fast) / (slow + ε)` over two surprise timescales: HIGH while error is
//!   dropping, → 0 once surprise has PLATEAUED (dSurprise/dt ≈ 0). Its own history ⇒ no authored magnitude.
//! - `saturated(heads, plateau)` — fires when learning has plateaued (relative fraction), replacing `ERR_TRIGGER`.
//!
//! THE FALSIFIER (truth-not-trophy; measured). The legitimacy win over a hand constant is MODEL-INDEPENDENCE: an
//! absolute threshold (`surprise < 0.45`) only detects saturation on tasks whose irreducible-error floor happens to
//! sit below it; a task with a HIGHER noise floor plateaus ABOVE 0.45 and the magic constant NEVER fires (misses
//! the plateau). The grown RELATIVE trigger fires on BOTH at their own plateau. Two tasks — low-noise (floor <0.45)
//! and high-noise (floor >0.45):
//!   (A) learning_progress SPIKES HIGH while learning (surprise dropping) and → ~0 at the plateau, on BOTH tasks.
//!   (B) grown `saturated` fires on BOTH (each at its own plateau) — model-independent.
//!   (C) the authored `ERR_TRIGGER = 0.45` fires on the low-noise task but MISSES the high-noise one — the magic
//!       constant is model-dependent and wrong. This is the purity win: the grown relative trigger is correct where
//!       the authored absolute threshold fails.
//! Off/irrelevant to existing behaviour (pure read-side accessors + one unused slow trace) — byte-identical.
//! Deterministic (seeded), CI-safe.

use openmorph_core::{Activation, Field};

use crate::common::Rng;

// The grown trigger's RELATIVE trend fraction (meta-evolvable), NOT a per-signal magnitude: "surprise is changing
// by less than 25% of its own level" = plateaued. Model-INDEPENDENT (works at any error floor), the whole point.
const PLATEAU: f32 = 0.25;

/// Train a linear readout toward `y = 0.8·x + noise` for `epochs` via `grow` (which updates the surprise EMAs).
/// Returns (peak learning_progress over the run, saturated-after-a-brief-warmup, final saturated, surprise floor).
fn run(noise: f32, epochs: usize, seed: u64) -> (f32, bool, bool, f32) {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0;
    f.config_mut().lr_fanin_norm = false;
    f.config_mut().gain_fanin_norm = false;
    let mut rng = Rng::new(seed);
    f.grow(&[(0, 0.0)], 1, 0.0, 0.0, 20); // birth the readout
    f.set_activation(1, Some(Activation::Identity));
    let mut peak_lp = 0.0f32;
    let mut warmup_saturated = false;
    let mut floor = 0.0f32;
    for e in 0..epochs {
        let x = rng.sym();
        f.grow(&[(0, x)], 1, 0.8 * x + noise * rng.sym(), 0.3, 20);
        peak_lp = peak_lp.max(f.learning_progress(1));
        if (1..5).contains(&e) && f.saturated(&[1], PLATEAU) {
            warmup_saturated = true; // must NOT fire during the initial surprise RISE (before any learning)
        }
        if e >= epochs - 40 {
            floor += f.surprise_of(1) / 40.0;
        }
    }
    (peak_lp, warmup_saturated, f.saturated(&[1], PLATEAU), floor)
}

#[test]
fn grown_saturation_is_model_independent_where_a_magic_constant_fails() {
    const ERR_TRIGGER: f32 = 0.45; // the harness's authored absolute threshold (autonomy_probe.rs)
    let epochs = 400;
    let seeds = [0x5Au64, 0x5B, 0x5C, 0x5D, 0x5E];

    // mean over seeds; grown-fires / warmup-fires as ALL/ANY-seed booleans (must hold every seed)
    let agg = |noise: f32| {
        let (mut peak, mut floor) = (0.0f32, 0.0f32);
        let (mut all_final, mut any_warmup) = (true, false);
        for &s in &seeds {
            let (p, w, fin, fl) = run(noise, epochs, s);
            peak += p / seeds.len() as f32;
            floor += fl / seeds.len() as f32;
            all_final &= fin;
            any_warmup |= w;
        }
        (peak, floor, all_final, any_warmup)
    };

    let (lo_peak, lo_floor, lo_grown, lo_warmup) = agg(0.10); // low-noise: floor BELOW 0.45
    let (hi_peak, hi_floor, hi_grown, hi_warmup) = agg(1.20); // high-noise: floor ABOVE 0.45
    let lo_magic = lo_floor < ERR_TRIGGER;
    let hi_magic = hi_floor < ERR_TRIGGER;

    eprintln!("[SAT] LOW-noise:  floor {lo_floor:.3}  peak learning_progress {lo_peak:.3}  grown saturated (all seeds): {lo_grown}  ERR_TRIGGER<0.45: {lo_magic}");
    eprintln!("[SAT] HIGH-noise: floor {hi_floor:.3}  peak learning_progress {hi_peak:.3}  grown saturated (all seeds): {hi_grown}  ERR_TRIGGER<0.45: {hi_magic}");
    eprintln!("[SAT] warmup false-positive (saturated fired during the early rise)?  low {lo_warmup}  high {hi_warmup}");
    eprintln!("[SAT] VERDICT: `learning_progress` (relative to each cell's own surprise history) spikes HIGH while the morph learns and → 0 at its OWN plateau, whatever the irreducible-error floor — so `saturated` (|trend|<{PLATEAU}) fires on BOTH tasks at their own floor, and NOT during the warmup rise. The authored absolute `ERR_TRIGGER=0.45` only fires when the floor happens to sit below it: it MISSES the high-noise plateau ({hi_floor:.3} > 0.45). The grown relative trigger is model-INDEPENDENT and correct where the hand constant is wrong — the #155 purity fix (the harness stops re-deriving error and stops hand-thresholding).");

    // (A) learning_progress spikes high during learning, on BOTH tasks.
    assert!(
        lo_peak > 0.3 && hi_peak > 0.3,
        "(A) learning_progress never spiked (lo {lo_peak:.3}, hi {hi_peak:.3})"
    );
    // (B) the grown trigger fires on BOTH (every seed) and NEVER during the warmup rise.
    assert!(
        lo_grown && hi_grown,
        "(B) grown saturated must fire on both tasks (lo {lo_grown}, hi {hi_grown})"
    );
    assert!(
        !lo_warmup && !hi_warmup,
        "(B) grown saturated false-fired during warmup (lo {lo_warmup}, hi {hi_warmup})"
    );
    // (C) the authored absolute constant is model-dependent: fires on low-noise, MISSES high-noise.
    assert!(
        hi_floor > ERR_TRIGGER,
        "(C) high-noise floor must sit ABOVE the magic constant ({hi_floor:.3} vs 0.45)"
    );
    assert!(lo_magic && !hi_magic, "(C) the magic constant should fire on LOW-noise and MISS high-noise (lo {lo_magic}, hi {hi_magic})");
}
