// SPDX-License-Identifier: MIT-0
//! v2.0-dev · the WAVELET bank (#145) — MorphLab's anti-collapse result, validated in-core. The open item I
//! flagged (a self-tuning resonator bank collapses to the dominant tone under energy-locking alone) is closed
//! WITHOUT a new primitive: a wavelet IS the seed-diversity — `resonate` at log-spaced ω with damping ∝ ω
//! (constant-Q). The log-Q seeds sit in separate basins, so `adapt_freq` polishes each scale instead of all
//! collapsing to one. A pure usage protocol over `resonate` + `adapt_freq` (the reasoning chain, spectral edition).
//!   (A) TEMPORAL — a 2-tone signal sin(0.3t)+sin(1.2t) needs BOTH bands to predict x[t+1]. A FLAT bank (same
//!       seed + adapt_freq) collapses to one tone; a WAVELET bank (log-Q + adapt_freq) keeps both — far lower MAE.
//!   (B) POSITIONAL — a log-Q bank on a space=time scan of a grating is scale-selective (energy peaks on the
//!       diagonal res-scale = grating-scale) and translation-invariant (band energy is phase-blind).
use openmorph_core::substrate::Field;

/// Train a bank (given per-resonator seed ω and damping) on a signal `sig`, linear readout over all [u,v]
/// predicting `sig(t+1)`, trained MANUALLY (one resonator tick per event), `adapt_freq` refining. Held-out MAE.
fn bank_mae(seeds: &[f32], dampings: &[f32], adapt: bool, sig: impl Fn(usize) -> f32) -> f32 {
    let mut f = Field::new(1);
    let rcs: Vec<usize> = seeds
        .iter()
        .zip(dampings)
        .map(|(&w, &d)| f.resonate(0, w, d))
        .collect();
    let reads: Vec<usize> = rcs.iter().flat_map(|&rc| [rc, rc + 1]).collect(); // u and v of each
    let out = f.len();
    f.ensure_cells(out + 1);
    let ru = f.add_unit(reads.clone(), vec![0.0; reads.len()], 0.0, out);
    let steps = 40000usize;
    for t in 0..steps {
        f.stimulate(&[(0, sig(t))], 2); // ONE tick of every resonator
        let pred = f.cells()[out];
        let vals: Vec<f32> = reads.iter().map(|&c| f.cells()[c]).collect();
        let deriv = 1.0 - pred * pred;
        let norm = 1.0 + vals.iter().map(|v| v * v).sum::<f32>(); // NLMS over the whole bank
        let step = 0.3 * (sig(t + 1) - pred) * deriv / norm;
        for (k, &v) in vals.iter().enumerate() {
            f.units[ru].weights[k] += step * v;
        }
        f.units[ru].bias += step;
        if adapt {
            f.adapt_freq(0.001); // gentle refine (log-Q grid is already well-placed; adapt polishes)
        }
    }
    let (mut e, n) = (0.0f32, 3000usize);
    for t in steps..steps + n {
        f.stimulate(&[(0, sig(t))], 2);
        e += (f.cells()[out] - sig(t + 1)).abs();
    }
    e / n as f32
}

/// scan a spatial grating of frequency `k` (phase `phi`) into event-time; return the tuned resonator's mean band
/// energy √(u²+v²) — the scale-selective, phase-blind read-out.
fn pos_energy(res_k: f32, grating_k: f32, phi: f32) -> f32 {
    let p = 12usize;
    let mut f = Field::new(1);
    let rc = f.resonate(0, res_k, 0.2 * res_k.max(0.05)); // constant-Q
    let vc = rc + 1;
    let (mut e, mut n) = (0.0f32, 0usize);
    for t in 0..6000usize {
        let pos = (t % p) as f32;
        f.stimulate(&[(0, (grating_k * pos + phi).sin())], 2);
        if t > 4000 {
            e += (f.cells()[rc].powi(2) + f.cells()[vc].powi(2)).sqrt();
            n += 1;
        }
    }
    e / n as f32
}

#[test]
fn wavelet_bank_beats_flat_collapse() {
    // ── (A) TEMPORAL: a wavelet (log-Q) bank keeps both tones; a flat (same-seed) bank collapses. ──
    let two_tone = |t: usize| 0.5 * ((0.3 * t as f32).sin() + (1.2 * t as f32).sin());
    // FLAT — 4 resonators all seeded at the SAME ω=0.6, uniform damping.
    let flat = bank_mae(&[0.6; 4], &[0.12; 4], true, two_tone);
    // WAVELET — 4 log-spaced seeds (×2 apart) with damping ∝ ω (constant-Q, Q≈5).
    let wl_seeds = [0.2f32, 0.4, 0.8, 1.6];
    let wl_damp: Vec<f32> = wl_seeds.iter().map(|&w| 0.2 * w).collect();
    let wavelet = bank_mae(&wl_seeds, &wl_damp, true, two_tone);
    let wavelet_frozen = bank_mae(&wl_seeds, &wl_damp, false, two_tone); // authored grid, no adapt
    eprintln!("[WL] 2-tone x[t+1] held-out MAE — FLAT(same-seed+adapt) {flat:.3}  WAVELET(logQ+adapt) {wavelet:.3}  WAVELET(logQ frozen) {wavelet_frozen:.3}");

    assert!(
        wavelet < 0.5 * flat,
        "a log-Q wavelet bank must keep both bands where a flat same-seed bank collapses: wavelet {wavelet:.3} vs flat {flat:.3}"
    );
    assert!(
        wavelet < 0.15,
        "the wavelet bank must actually predict the 2-tone signal well (both bands present): MAE {wavelet:.3}"
    );

    // ── (B) POSITIONAL: a resonator on a space=time scan is scale-selective + translation-invariant. ──
    let scales = [core::f32::consts::PI / 6.0, core::f32::consts::PI / 3.0]; // two grating scales
    eprintln!(
        "[WL] positional — band energy, res-scale × grating-scale (diagonal should dominate):"
    );
    let mut diag_ok = true;
    for (i, &rk) in scales.iter().enumerate() {
        let on = pos_energy(rk, scales[i], 0.0);
        let off = pos_energy(rk, scales[1 - i], 0.0);
        eprintln!("[WL]   res-k {rk:.3}: on-scale energy {on:.3}  off-scale {off:.3}");
        diag_ok &= on > 1.5 * off;
    }
    assert!(
        diag_ok,
        "positional resonators must be scale-selective (on-scale energy dominates off-scale)"
    );
    // translation-invariance — band energy is phase-blind, so shifting the grating barely changes it
    let k = scales[1];
    let energies: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0]
        .iter()
        .map(|&phi| pos_energy(k, k, phi))
        .collect();
    let mean = energies.iter().sum::<f32>() / energies.len() as f32;
    let cv = (energies.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / energies.len() as f32)
        .sqrt()
        / mean;
    eprintln!(
        "[WL]   translation: energies across phases {energies:?}  coeff-of-variation {cv:.3}"
    );
    assert!(
        cv < 0.25,
        "the tuned band energy must be ~phase-blind (translation-invariant): CV {cv:.3}"
    );

    eprintln!("[WL] VERDICT: the open collapse item is CLOSED, no new primitive. (A) A log-Q WAVELET bank (`resonate` at log-spaced ω, damping ∝ ω) keeps BOTH tones of sin(0.3t)+sin(1.2t) — held-out MAE {wavelet:.3} vs a flat same-seed bank's {flat:.3} (which collapses to one tone); the log-Q seeds are the seed-diversity that stops the collapse, and adapt_freq polishes (frozen grid {wavelet_frozen:.3}). (B) The same bank on a space=time scan is a POSITIONAL wavelet: scale-selective (diagonal energy dominates) and translation-invariant (band energy phase-blind, CV {cv:.3}). MorphLab's recommendation, validated in-core — the spectral front-end is a usage protocol over resonate + adapt_freq + the reliance economy.");
}
