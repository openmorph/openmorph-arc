// SPDX-License-Identifier: MIT-0
//! OM-11 (#172) — the behavioural EMBEDDING, the mycelium library retrieval key. A small morph must interrogate a
//! library for OTHER crystallized skills that help it and reuse/compose them (the marketplace north-star #64); the
//! search key is an embedding vector you drop into any ANN store. The raw `fingerprint` spectrum is structurally
//! blind (deaf to the I/O ports; cosine over a magnitude spectrum is blind to a global GAIN rescale). `embed()` is
//! a FIXED-LENGTH nested key: a spectral prefix (backward-compatible), a SIGNED across-probe shape curve, an ENERGY
//! shape curve (the sign-blind backstop), and one BOUNDED gain coordinate (output/input energy ratio).
//!
//! THE FALSIFIER (truth-not-trophy; measured; would BREAK if the mechanism is wrong). This rev is the redesign that
//! fixed the OM-11 verify's four majors — amplitude dominance, low-energy collapse, cross-arity misalignment, and
//! the overclaims — so the gates below deliberately probe those regimes:
//!   (A) PREFIX + CLONE: `embed()`'s spectral prefix (dims = spectrum_bins) gives exactly `fingerprint_distance`;
//!       a clone embeds at ~0. Fixed length regardless of output arity.
//!   (B) GAIN GRID (amplitude-consistent): y=x vs y=g·x for g ∈ {0.1, 0.5, 2, 10, 100} all embed FAR, and — the
//!       shipped bug — HIGH gain does NOT collapse toward 0 (the shipped raw-prefix version returned ~0.001 at
//!       g=100, degrading back to the gain-blind fingerprint). Shrinks and stretches both separate.
//!   (C) SIGN + FUNCTION: y=x vs y=−x embed FAR (the signed shape curve is sign-aware for a scalar output), and
//!       y=x vs y=x² embed FAR; a copy dedups to ~0.
//!   (D) CROSS-ARITY: a 1-output and a 2-output skill embed to the SAME length (aligned blocks), and a re-GROWN
//!       (not cloned) copy of a skill retrieves that skill over strangers — real growth-invariance, not clone-only.
//!
//! HONEST SCOPE: reads the ACTUAL settled response (nonlinear, not an LTI C·Aᵏ·B, #172 Risk #1). Invariant to
//! internal re-wiring under a FIXED I/O layout + comparable amplitude, NOT to arbitrary port relabelling (probes
//! are index-keyed). The gain axis is monotone-but-saturating (rank, don't threshold). Deterministic, CI-safe.

use openmorph_core::{Activation, Field};

use crate::common::Rng;

/// A 1-in → 1-out linear skill toward `y = gain·x` (Identity readout, no hidden). Cell 0 = input, cell 1 = output.
fn linear_skill(gain: f32, seed: u64) -> Field {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0;
    f.config_mut().lr_fanin_norm = false;
    f.config_mut().gain_fanin_norm = false;
    let mut rng = Rng::new(seed);
    f.grow(&[(0, 0.0)], 1, 0.0, 0.0, 20);
    f.set_activation(1, Some(Activation::Identity));
    for _ in 0..800 {
        let x = rng.sym();
        f.grow(&[(0, x)], 1, gain * x, 0.2, 20);
    }
    f
}

/// A nonlinear skill toward `y = x²` (grows a hidden feature). `max_hidden` lets the internal wiring differ run to
/// run, so re-growing it is a real growth-invariance test, not a clone. Same ports as `linear_skill`.
fn square_skill(seed: u64, hidden: usize) -> Field {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = hidden;
    let mut rng = Rng::new(seed);
    f.grow(&[(0, 0.0)], 1, 0.0, 0.0, 20);
    for _ in 0..1800 {
        let x = rng.sym();
        f.grow(&[(0, x)], 1, x * x, 0.2, 30);
    }
    f
}

/// A 2-in → 2-out skill toward `(y0, y1) = (a, b)` (a passthrough of both inputs). Tests cross-arity alignment.
fn dual_skill(seed: u64) -> Field {
    let mut f = Field::new(4); // cells 0,1 = inputs; 2,3 = outputs
    f.config_mut().max_hidden = 0;
    let mut rng = Rng::new(seed);
    f.grow(&[(0, 0.0)], 2, 0.0, 0.0, 20);
    f.grow(&[(1, 0.0)], 3, 0.0, 0.0, 20);
    f.set_activation(2, Some(Activation::Identity));
    f.set_activation(3, Some(Activation::Identity));
    for _ in 0..800 {
        let (a, b) = (rng.sym(), rng.sym());
        f.grow(&[(0, a), (1, b)], 2, a, 0.2, 20);
        f.grow(&[(0, a), (1, b)], 3, b, 0.2, 20);
    }
    f
}

#[test]
fn embed_prefix_matches_fingerprint_and_clone_is_zero() {
    let f = linear_skill(1.0, 0x11);
    let emb = f.embed();
    let bins = f.config().spectrum_bins;
    let probes = f.config().fingerprint_probes;
    // (A) fixed length = bins + 2·probes + 1, regardless of arity.
    assert_eq!(
        emb.len(),
        bins + 2 * probes + 1,
        "embedding length is not the fixed contract width"
    );
    // matryoshka: the prefix (dims = spectrum_bins) reproduces the spectral fingerprint distance.
    let g = linear_skill(1.0, 0x12);
    let prefix_d = f.embed_distance(&g, bins);
    let fp_d = f.fingerprint_distance(&g);
    assert!(
        (prefix_d - fp_d).abs() < 1e-5,
        "(A) prefix distance {prefix_d:.5} != fingerprint_distance {fp_d:.5}"
    );
    // a clone embeds at ~0.
    assert!(
        f.embed_distance(&f.clone(), 0) < 1e-6,
        "a clone must embed at ~0"
    );
    eprintln!(
        "[EMBED] len {} = bins {bins} + 2·probes {probes} + 1 ✓  prefix==fingerprint ✓  clone ~0 ✓",
        emb.len()
    );
}

#[test]
fn gain_grid_is_amplitude_consistent() {
    // (B) y=x vs y=g·x — the same behaviour up to a global output gain. Every rescale must embed FAR, and HIGH gain
    //     must NOT collapse toward 0 (the shipped raw-prefix bug). All measured against the spectrum, which is BLIND.
    let base = linear_skill(1.0, 0x2A);
    eprintln!("[EMBED] gain grid (y=x vs y=g·x): g | fingerprint_distance (blind) | embed_distance (must stay FAR)");
    let mut worst = f32::MAX;
    for (i, &g) in [0.1f32, 0.5, 2.0, 10.0, 100.0].iter().enumerate() {
        let rescaled = linear_skill(g, 0x2B ^ i as u64);
        let fp = base.fingerprint_distance(&rescaled);
        let em = base.embed_distance(&rescaled, 0);
        eprintln!("[EMBED]   g={g:>6} | fp {fp:.4} | embed {em:.4}");
        worst = worst.min(em);
        assert!(
            em > 0.1,
            "(B) gain g={g} did not embed far ({em:.4}) — the metric lost the rescale"
        );
        assert!(
            em > 3.0 * fp.max(1e-4),
            "(B) embed g={g} ({em:.4}) did not beat the blind spectrum ({fp:.4})"
        );
    }
    // the KEY regression: g=100 (high gain) stayed far — no collapse toward the fingerprint.
    let hi = base.embed_distance(&linear_skill(100.0, 0x2C), 0);
    assert!(
        hi > 0.2,
        "(B) high gain collapsed (embed {hi:.4}) — amplitude dominance regressed"
    );
    eprintln!("[EMBED] VERDICT: bounded output/input-ratio gain — worst-case rescale distance {worst:.3}, high-gain {hi:.3} (no collapse).");
}

#[test]
fn embed_discriminates_sign_function_and_dedups() {
    // (C) sign flip and different function both embed far; a copy dedups to ~0.
    let pos = linear_skill(1.0, 0x3C);
    let neg = linear_skill(-1.0, 0x3D);
    let sq = square_skill(0x3E, 4);
    let sign_d = pos.embed_distance(&neg, 0);
    let func_d = pos.embed_distance(&sq, 0);
    let copy_d = sq.embed_distance(&sq.clone(), 0);
    eprintln!("[EMBED] sign y=x vs y=−x: {sign_d:.4}; function y=x vs y=x²: {func_d:.4}; copy: {copy_d:.4}");
    assert!(copy_d < 1e-6, "(C) a copy must dedup to ~0");
    assert!(
        sign_d > 0.1,
        "(C) sign flip must embed far (scalar signed shape), got {sign_d:.4}"
    );
    assert!(
        func_d > 0.1,
        "(C) different functions must embed far, got {func_d:.4}"
    );
}

#[test]
fn cross_arity_aligns_and_regrown_copy_retrieves() {
    // (D-i) different arities embed to the SAME length (blocks aligned, no layout noise).
    let one = linear_skill(1.0, 0x40);
    let two = dual_skill(0x41);
    assert_eq!(
        one.embed().len(),
        two.embed().len(),
        "(D) cross-arity keys must be the same length"
    );

    // (D-ii) REAL growth-invariance + retrieval: a library {sq_a, lin, dual}; a RE-GROWN copy of sq_a (grown from a
    // different seed AND a different hidden width — genuinely re-wired, not cloned) must retrieve sq_a as top-1.
    let sq_a = square_skill(0x50, 4);
    let lib_lin = linear_skill(1.0, 0x51);
    let lib_dual = dual_skill(0x52);
    let regrown = square_skill(0x9999, 6); // same y=x² function, different seed + wider hidden layer

    let d_a = regrown.embed_distance(&sq_a, 0);
    let d_lin = regrown.embed_distance(&lib_lin, 0);
    let d_dual = regrown.embed_distance(&lib_dual, 0);
    eprintln!(
        "[EMBED] re-grown y=x² retrieval: →sq_a {d_a:.4} | →linear {d_lin:.4} | →dual {d_dual:.4}"
    );
    assert!(d_a < d_lin && d_a < d_dual, "(D) a re-grown skill must retrieve its own family top-1 (sq {d_a:.4} vs lin {d_lin:.4}, dual {d_dual:.4})");
    assert!(d_a < 0.15, "(D) a re-grown copy should embed CLOSE to its stored skill (growth-invariance), got {d_a:.4}");
}
