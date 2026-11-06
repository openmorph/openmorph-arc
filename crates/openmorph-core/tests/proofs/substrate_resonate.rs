// SPDX-License-Identifier: MIT-0
//! v2.0-dev · the RESONATE core spike (#145) — the band-pass twin of `integrate`, validated by BEHAVIOR.
//! A `resonate(source, freq, damping)` cell is a 2nd-order complex-pole oscillator that RINGS at its tuned
//! frequency (pole r·e^{±iω}). Two claims:
//!   (1) BAND-PASS — its steady-state output energy PEAKS when driven at its tuned frequency, low off-band.
//!   (2) GROWN FREQUENCY (the moat gate) — starting DETUNED, `adapt_freq` climbs the resonator's OWN output
//!       energy to LOCK onto the signal frequency from BOTH sides, resonating far more than a frozen control.
//!       Unsupervised frequency estimation — nothing authored; the signal reveals its own frequencies.
use openmorph_core::substrate::Field;

fn learned_freq(f: &Field, cell: usize) -> f32 {
    let v: serde_json::Value = serde_json::to_value(f).unwrap();
    v.get("resonators")
        .and_then(|r| r.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|r| r.get("cell").and_then(|c| c.as_u64()) == Some(cell as u64))
                .and_then(|r| r.get("freq").and_then(|fr| fr.as_f64()))
        })
        .map(|x| x as f32)
        .unwrap_or(-1.0)
}

/// (1) drive a resonator (tuned to `reson_freq`) with a sinusoid at `drive_freq`; return the steady-state RMS of
/// its output — a band-pass response peaks at resonance.
fn ring_rms(reson_freq: f32, damping: f32, drive_freq: f32) -> f32 {
    let mut f = Field::new(1);
    let rc = f.resonate(0, reson_freq, damping);
    let (mut sum_sq, mut n) = (0.0f32, 0usize);
    for t in 0..3000usize {
        let x = (drive_freq * t as f32).sin();
        f.stimulate(&[(0, x)], 2);
        if t > 2200 {
            sum_sq += f.cells()[rc].powi(2);
            n += 1;
        }
    }
    (sum_sq / n as f32).sqrt()
}

const W_SIG: f32 = 0.60; // the signal frequency the resonator must DISCOVER

/// (2) drive a resonator from a DETUNED seed with a single tone at W_SIG; if `adapt`, `adapt_freq` climbs the
/// resonator's own output energy to LOCK onto W_SIG (unsupervised frequency estimation). Returns (learned freq,
/// steady-state output RMS = how strongly it resonates).
fn lock_freq(start_freq: f32, adapt: bool) -> (f32, f32) {
    let mut f = Field::new(1);
    let rc = f.resonate(0, start_freq, 0.05); // r=0.95, high-Q ⇒ a sharp energy peak to climb
    let (mut sum_sq, mut n) = (0.0f32, 0usize);
    for t in 0..12000usize {
        let x = (W_SIG * t as f32).sin();
        f.stimulate(&[(0, x)], 2);
        if adapt {
            // sign-based energy ascent on ω, step ANNEALED 0.004→0.0004 (fast march, then a tight lock)
            let lr = 0.004 * (1.0 - t as f32 / 12000.0).max(0.1);
            f.adapt_freq(lr);
        }
        if t > 11000 {
            sum_sq += f.cells()[rc].powi(2);
            n += 1;
        }
    }
    (learned_freq(&f, rc), (sum_sq / n as f32).sqrt())
}

#[test]
fn resonate_rings_and_grows_its_frequency() {
    // ── (1) BAND-PASS selectivity — a resonator tuned to 0.6 rings hardest when driven at 0.6. ──
    let tuned = 0.6f32;
    eprintln!(
        "[RES] band-pass (resonator tuned {tuned}, high-Q): drive-freq → steady-state output RMS"
    );
    let mut peak_at = (0.0f32, -1.0f32);
    for &dr in &[0.2f32, 0.4, 0.6, 0.8, 1.0] {
        let rms = ring_rms(tuned, 0.03, dr);
        eprintln!("[RES]   drive {dr:.1} → RMS {rms:.3}");
        if rms > peak_at.0 {
            peak_at = (rms, dr);
        }
    }
    let off = ring_rms(tuned, 0.03, 0.2);
    let on = ring_rms(tuned, 0.03, tuned);
    assert!(
        (peak_at.1 - tuned).abs() < 1e-3,
        "the resonator's response must PEAK at its tuned frequency {tuned}, peaked at {}",
        peak_at.1
    );
    assert!(
        on > 3.0 * off,
        "on-band response must dominate off-band (band-pass): on {on:.3} vs off {off:.3}"
    );

    // ── (2) GROWN FREQUENCY (the moat gate) — a detuned resonator LOCKS onto the signal, from both sides. ──
    let (lf_lo, rms_lo) = lock_freq(0.20, true); // seed BELOW the signal
    let (lf_hi, rms_hi) = lock_freq(1.10, true); // seed ABOVE the signal
    let (lf_frozen, rms_frozen) = lock_freq(0.20, false); // frozen-detuned control (same seed as lo)
    eprintln!("[RES] grown frequency (single tone W_SIG={W_SIG}): learned freq / steady-state output RMS:");
    eprintln!("[RES]   seed 0.20 + adapt → freq {lf_lo:.3}, RMS {rms_lo:.3}");
    eprintln!("[RES]   seed 1.10 + adapt → freq {lf_hi:.3}, RMS {rms_hi:.3}");
    eprintln!("[RES]   seed 0.20 FROZEN  → freq {lf_frozen:.3}, RMS {rms_frozen:.3}");

    // converges from BOTH sides to the SAME frequency
    assert!(
        (lf_lo - lf_hi).abs() < 0.08,
        "adapt_freq must converge from both sides to the same frequency: {lf_lo:.3} vs {lf_hi:.3}"
    );
    // and both LOCKED onto the signal frequency, migrating off their detuned seeds
    assert!(
        (lf_lo - W_SIG).abs() < 0.08 && (lf_hi - W_SIG).abs() < 0.08,
        "adapt_freq must discover W_SIG {W_SIG} from both sides: {lf_lo:.3} (from .20), {lf_hi:.3} (from 1.10)"
    );
    // and the locked resonator RESONATES far more strongly than the frozen-detuned control
    assert!(
        rms_lo > 3.0 * rms_frozen,
        "the locked resonator must resonate far more than the frozen-detuned control: RMS {rms_lo:.3} vs {rms_frozen:.3}"
    );

    eprintln!("[RES] VERDICT: `resonate` is a working band-pass resonator (#145) — it RINGS at its tuned frequency (on-band RMS {on:.2} ≫ off-band {off:.2}), and its frequency is GROWN not authored: from detuned seeds 0.20/1.10 `adapt_freq` climbs the resonator's own output energy to LOCK onto the SAME signal frequency ({lf_lo:.2}/{lf_hi:.2}≈{W_SIG}), resonating {:.1}× stronger than a frozen-detuned control. The complex-pole band-pass twin of integrate's real-pole low-pass, advanced once per event — the moat gate holds: the bank tunes itself to the signal's energy, unsupervised.", rms_lo / rms_frozen.max(1e-6));
}
