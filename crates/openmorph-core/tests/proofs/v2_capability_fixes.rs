// SPDX-License-Identifier: MIT-0
//! v2.0-dev · the two capability fixes from the scale battery, as genome (Config) knobs, validated by BEHAVIOR:
//!   • `lr_fanin_norm` (lr∝1/W, Normalized-LMS): a wide dense readout no longer COLLAPSES past ~256 features —
//!     one base lr serves narrow AND wide (the effective step is divided by the input energy Σx² ≈ W).
//!   • `gain_fanin_norm` (g/√N): a coupled recurrent population no longer goes SUPER-CRITICAL purely by growing —
//!     the transient spectral radius ρ ≈ gain·w·√N is pinned by scaling the gain 1/√fan_in.
//! Both default OFF (byte-identical); the full proof suite passing unchanged is the no-regression guarantee.
//!
//! The 1.6 learning flip: the default delta step is now kept stable by `Config::stability_cap`, which replaces the
//! fan-in-128 auto-normalization. The width proofs therefore build their RAW arms explicitly (`stability_cap` 0 on the
//! legacy genome) and add the default genome as its own arm: the cap alone keeps one fixed small `lr` working at
//! narrow AND wide (no collapse), and protects the wide-from-birth readout like the legacy auto-normalization did.
use openmorph_core::substrate::Field;
use openmorph_core::Config;

use crate::common::Rng;

/// Grow a wide dense readout on tanh(2·x0·x1); return held-out MAE. `base_lr` is the caller's lr (NLMS uses a
/// proper μ≈1 that the 1/Σx² normalization scales down per width; the un-normalized path uses the usual small lr).
fn width_mae(width: usize, normalize: bool, base_lr: f32, epochs: usize) -> f32 {
    // The classic regime this helper measures: the legacy genome (the raw step unless `normalize`, fan-in-blind
    // random features) — the flag's own effect, isolated.
    width_mae_on(Config::legacy(), width, normalize, base_lr, epochs)
}

/// [`width_mae`] on an explicit `genome` (the explicit flag, the fixed directions, the width and the fill are set on
/// top of it).
fn width_mae_on(genome: Config, width: usize, normalize: bool, base_lr: f32, epochs: usize) -> f32 {
    let mut f = Field::new(3);
    f.configure(genome);
    f.config_mut().birth_gate = 0.0; // fill to exactly `width` features
    f.config_mut().feature_lr = 0.0; // classic fixed-direction random features; only the readout learns
    f.config_mut().max_hidden = width;
    f.config_mut().lr_fanin_norm = normalize;
    // Isolate the EXPLICIT flag: disable the dimension-aware AUTO-normalization (default `width_norm_min = 128`)
    // so this helper demonstrates `lr_fanin_norm`'s own effect. Otherwise a width-768 readout auto-normalizes
    // even with the flag off, and the raw un-normalized collapse (the thing this proves the flag fixes) never
    // shows. The auto-default is proved separately in `dimension_aware_default_auto_fixes_the_wide_collapse`.
    f.config_mut().width_norm_min = usize::MAX;
    let budget = width + 8; // width must be paid for in settle budget (battery finding)
    let mut rng = Rng::new(0x5EED);
    for _ in 0..epochs {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.grow(
            &[(0, x0), (1, x1)],
            2,
            (2.0 * x0 * x1).tanh(),
            base_lr,
            budget,
        );
    }
    let mut r2 = Rng::new(0x5EED ^ 0xFFFF);
    let (mut e, n) = (0.0f32, 400);
    for _ in 0..n {
        let (x0, x1) = (r2.sym(), r2.sym());
        f.stimulate(&[(0, x0), (1, x1)], budget);
        e += (f.cells()[2] - (2.0 * x0 * x1).tanh()).abs();
    }
    e / n as f32
}

/// Build a coupled all-to-all recurrent reservoir (each cell = tanh(Σ_{j≠i} ±w·x_j)); return transient ρ.
fn reservoir_rho(n: usize, w: f32, normalize: bool) -> f32 {
    let mut f = Field::new(n);
    f.config_mut().gain_fanin_norm = normalize;
    let mut rng = Rng::new(0xC0FFEE);
    for i in 0..n {
        let reads: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        let weights: Vec<f32> = reads
            .iter()
            .map(|_| if rng.bit() > 0.0 { w } else { -w })
            .collect();
        f.add_unit(reads, weights, 0.0, i);
    }
    f.spectral_radius(80) // measured at cells=0 (the transient), per E1/E2
}

#[test]
fn dimension_aware_default_auto_normalizes_a_wide_from_birth_readout() {
    // The DIMENSION-AWARE default (`Config::width_norm_min = 128`) protects the case that actually diverges in
    // practice: a readout that is WIDE FROM BIRTH (a WorldModel predictor reads all `n_in` frame cells from step
    // 0). With the raw step, a base lr that suits a narrow readout gives effective lr `lr·W` and the wide readout
    // diverges silently; with the default auto-normalization it learns WITHOUT the caller ever setting
    // `lr_fanin_norm`. (A readout that GROWS wide through the threshold — `grow` with a high `max_hidden` and a
    // wide-tuned base lr — still passes through the sub-threshold regime and needs the explicit flag; the default
    // targets the wide-from-birth divergence, which is the WorldModel / raw-frame scenario.)
    const W: usize = 256; // a real 16×16 frame; ≥ width_norm_min → auto-normalized by default
    let mae = |auto: bool| -> f32 {
        let mut f = Field::new(W + 1);
        f.config_mut().activation = openmorph_core::Activation::Identity;
        f.config_mut().max_hidden = 0;
        if !auto {
            // the RAW step: no auto-normalization and no step-stability cap (the 1.6 default's guard)
            f.config_mut().width_norm_min = usize::MAX;
            f.config_mut().stability_cap = 0.0;
        }
        f.add_unit((0..W).collect(), vec![0.0; W], 0.0, W); // wide-from-birth linear readout over W inputs
        let mut rng = Rng::new(0x1234);
        // learn a single target dim = x[0] (a trivial linear map) at a lr that suits a narrow readout.
        for _ in 0..300 {
            let x: Vec<f32> = (0..W).map(|_| rng.sym()).collect();
            let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            f.stimulate(&[(0, 0.0)], W + 4); // reset (fire-on-change)
            f.stimulate(&inp, W + 4);
            f.learn_step(&inp, W, x[0], 0.5, W + 4);
        }
        let mut r2 = Rng::new(0x9999);
        let (mut e, n) = (0.0f32, 200);
        for _ in 0..n {
            let x: Vec<f32> = (0..W).map(|_| r2.sym()).collect();
            let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            f.stimulate(&[(0, 0.0)], W + 4);
            f.stimulate(&inp, W + 4);
            e += (f.cells()[W] - x[0]).abs();
        }
        e / n as f32
    };
    let raw = mae(false); // no auto-normalization, no cap → raw step → diverges at W=256
    let auto = mae(true); // the default genome: the step-stability cap bounds the step → learns
                          // The legacy genome's guard, the fan-in-128 auto-normalization, protects it the same way.
    let legacy = {
        let mut f = Field::new(W + 1);
        f.configure(Config::legacy());
        f.config_mut().activation = openmorph_core::Activation::Identity;
        f.config_mut().max_hidden = 0;
        f.add_unit((0..W).collect(), vec![0.0; W], 0.0, W);
        let mut rng = Rng::new(0x1234);
        for _ in 0..300 {
            let x: Vec<f32> = (0..W).map(|_| rng.sym()).collect();
            let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            f.stimulate(&[(0, 0.0)], W + 4);
            f.stimulate(&inp, W + 4);
            f.learn_step(&inp, W, x[0], 0.5, W + 4);
        }
        let mut r2 = Rng::new(0x9999);
        let (mut e, n) = (0.0f32, 200);
        for _ in 0..n {
            let x: Vec<f32> = (0..W).map(|_| r2.sym()).collect();
            let inp: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            f.stimulate(&[(0, 0.0)], W + 4);
            f.stimulate(&inp, W + 4);
            e += (f.cells()[W] - x[0]).abs();
        }
        e / n as f32
    };
    eprintln!(
        "[CAP] wide-from-birth W=256:  raw MAE={raw:.3}   default (stability cap) MAE={auto:.3}   legacy (auto-NLMS) MAE={legacy:.3}"
    );
    assert!(
        legacy.is_finite() && legacy < 0.5,
        "the legacy genome's fan-in-128 auto-normalization protects it too (MAE {legacy:.3})"
    );
    assert!(
        raw > 0.3,
        "the raw un-normalized wide-from-birth readout must diverge/collapse (MAE {raw:.3})"
    );
    assert!(
        auto.is_finite() && auto < 0.5,
        "the dimension-aware default must auto-normalize it (finite, learning, beats the ~1.0 predict-zero) \
         without the explicit flag (MAE {auto:.3} vs raw {raw:.3})"
    );
}

#[test]
fn lr_fanin_norm_fixes_the_wide_readout_collapse() {
    // Un-normalized: a fixed small lr fits narrow but COLLAPSES wide (the battery's width wall).
    let off_narrow = width_mae(64, false, 0.05, 5000);
    let off_wide = width_mae(768, false, 0.05, 5000);
    // Normalized-LMS: a proper base lr (μ≈1) works at BOTH widths (effective step ∝ 1/W automatically).
    let on_narrow = width_mae(64, true, 1.0, 5000);
    let on_wide = width_mae(768, true, 1.0, 5000);
    eprintln!("[CAP] WIDTH (lr∝1/W):  OFF narrow={off_narrow:.3} wide={off_wide:.3}   |   ON narrow={on_narrow:.3} wide={on_wide:.3}");

    assert!(
        off_wide > off_narrow + 0.2,
        "un-normalized: the wide readout must COLLAPSE vs narrow (fixed lr fails at width): {off_narrow:.3}→{off_wide:.3}"
    );
    assert!(
        on_wide < 0.15 && on_narrow < 0.15,
        "lr_fanin_norm: ONE base lr must work at narrow AND wide (no collapse): narrow {on_narrow:.3}, wide {on_wide:.3}"
    );
    // The 1.6 default: the step-stability cap alone keeps the SAME fixed small lr working at both widths (it caps
    // only the wide readout's step, where the raw one diverges).
    let cap_narrow = width_mae_on(Config::default(), 64, false, 0.05, 5000);
    let cap_wide = width_mae_on(Config::default(), 768, false, 0.05, 5000);
    eprintln!("[CAP] WIDTH, default genome (stability cap, lr 0.05): narrow={cap_narrow:.3} wide={cap_wide:.3}");
    assert!(
        cap_wide < 0.15 && cap_narrow < 0.15,
        "the default step-stability cap: one small lr works at narrow AND wide: narrow {cap_narrow:.3}, wide {cap_wide:.3}"
    );

    // ── g/√N: the coupled reservoir. ──
    let mut off_rhos = Vec::new();
    let mut on_rhos = Vec::new();
    for &nn in &[4usize, 16, 64, 256] {
        off_rhos.push((nn, reservoir_rho(nn, 0.15, false)));
        on_rhos.push((nn, reservoir_rho(nn, 0.15, true)));
    }
    eprintln!("[CAP] RECURRENCE (g/√N), raw w=0.15:");
    for ((n, ro), (_, rn)) in off_rhos.iter().zip(on_rhos.iter()) {
        eprintln!(
            "[CAP]   N={n:>3}:  OFF ρ={ro:>5.2} ({})   ON ρ={rn:>5.2} ({})",
            if *ro > 1.0 { "SUPER-CRIT" } else { "sub-crit" },
            if *rn > 1.0 { "SUPER-CRIT" } else { "sub-crit" }
        );
    }
    let off_256 = off_rhos.last().unwrap().1;
    let on_256 = on_rhos.last().unwrap().1;
    assert!(
        off_256 > 1.0,
        "un-normalized coupling must go SUPER-CRITICAL by N=256 (ρ≈w√N): ρ={off_256:.2}"
    );
    assert!(
        on_256 < 1.0 && on_rhos.iter().all(|&(_, r)| r < 1.0),
        "gain_fanin_norm must PIN ρ<1 at every N (g/√N): ρ(N=256)={on_256:.2}"
    );

    eprintln!("[CAP] VERDICT: both capability fixes work BY BEHAVIOR. lr_fanin_norm makes ONE lr serve narrow+wide (wide MAE {off_wide:.2}→{on_wide:.2}); gain_fanin_norm pins the recurrent ρ<1 at every N (N=256: {off_256:.2}→{on_256:.2}). Both are opt-in Config (genome) knobs, default off = byte-identical.");
}
