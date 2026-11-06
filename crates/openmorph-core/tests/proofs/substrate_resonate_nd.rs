// SPDX-License-Identifier: MIT-0
//! v2.0-dev · `resonate` in N-D and in TIME (#145) — the same 1-D primitive is a full SPATIOTEMPORAL spectral
//! front-end with ZERO new core code, via space = time = one dimensional sense (#142/#137). Three claims:
//!   (1) N-D FIELD — one resonator per cell, each locks to ITS OWN cell's frequency (a grown spectral map).
//!   (2) MOTION = TIME ("adding time") — a moving grating makes a fixed pixel oscillate at k·v; `adapt_freq`
//!       recovers the velocity AS a temporal frequency (2 speeds → 2 locked freqs in the right ratio).
//!   (3) SPACE = TIME — scanning a spatial grating into event-time, `resonate` reads the SPATIAL frequency,
//!       and it is TRANSLATION-INVARIANT (a shifted grating locks to the SAME frequency). No new mechanism.
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

/// annealed sign-ascent lock rate (fast march → tight lock), matching `substrate_resonate`
fn anneal_lr(t: usize, steps: usize) -> f32 {
    0.004 * (1.0 - t as f32 / steps as f32).max(0.1)
}

// ── (1) N-D FIELD: one resonator per cell, each locks to its own cell's frequency. ──
fn nd_field() -> Vec<f32> {
    let freqs = [0.40f32, 0.70, 1.00, 1.50];
    let seeds = [0.20f32, 0.35, 0.60, 1.00]; // all detuned away from the truth
    let mut f = Field::new(freqs.len());
    let rcs: Vec<usize> = (0..freqs.len())
        .map(|i| f.resonate(i, seeds[i], 0.05))
        .collect();
    let steps = 14000;
    for t in 0..steps {
        let inp: Vec<(usize, f32)> = (0..freqs.len())
            .map(|i| (i, (freqs[i] * t as f32).sin()))
            .collect();
        f.stimulate(&inp, 2);
        f.adapt_freq(anneal_lr(t, steps)); // every resonator climbs ITS OWN energy → its own source freq
    }
    rcs.iter().map(|&rc| learned_freq(&f, rc)).collect()
}

// ── (2) MOTION: a grating sin(k·(p − v·f)) moving at velocity v; a fixed pixel p0 oscillates at k·v. ──
fn motion_freq(k: f32, v: f32) -> f32 {
    let p0 = 3.0f32;
    let mut f = Field::new(1);
    let rc = f.resonate(0, 0.15, 0.05); // seed low, below both velocities' temporal freqs
    let steps = 14000;
    for fr in 0..steps {
        let x = (k * p0 - k * v * fr as f32).sin(); // the fixed pixel's value across frames
        f.stimulate(&[(0, x)], 2);
        f.adapt_freq(anneal_lr(fr, steps));
    }
    learned_freq(&f, rc)
}

// ── (3) SPACE = TIME: scan a P-cell spatial grating (freq k) cyclically into event-time; `shift` translates it. ──
fn spatial_freq(k: f32, shift: usize) -> f32 {
    let p = 12usize; // k·P is a multiple of 2π ⇒ the cyclic scan is a seamless tone at k
    let mut f = Field::new(1);
    let rc = f.resonate(0, 0.15, 0.05);
    let steps = 14000;
    for t in 0..steps {
        let pos = ((t + shift) % p) as f32; // the scan position (space presented as event-time)
        let x = (k * pos).sin();
        f.stimulate(&[(0, x)], 2);
        f.adapt_freq(anneal_lr(t, steps));
    }
    learned_freq(&f, rc)
}

#[test]
fn resonate_is_spatiotemporal() {
    // (1) N-D spectral field ---------------------------------------------------------------------------------
    let truth = [0.40f32, 0.70, 1.00, 1.50];
    let locked = nd_field();
    eprintln!("[ND] N-D field — per-cell resonators lock to per-cell frequencies:");
    for (i, (&t, &l)) in truth.iter().zip(locked.iter()).enumerate() {
        eprintln!("[ND]   cell {i}: truth {t:.2} → locked {l:.3}");
    }
    for (&t, &l) in truth.iter().zip(locked.iter()) {
        assert!(
            (l - t).abs() < 0.10,
            "each resonator must lock to its own cell's frequency: truth {t:.2}, locked {l:.3}"
        );
    }

    // (2) MOTION = temporal frequency (adding time) ----------------------------------------------------------
    let k = 1.0f32;
    let (slow, fast) = (motion_freq(k, 0.30), motion_freq(k, 0.60)); // temporal freq = k·v = 0.30, 0.60
    eprintln!(
        "[ND] motion (k={k}): v=0.30 → locked {slow:.3} (expect ~{:.2}),  v=0.60 → locked {fast:.3} (expect ~{:.2})",
        k * 0.30,
        k * 0.60
    );
    assert!(
        (slow - 0.30).abs() < 0.08 && (fast - 0.60).abs() < 0.08,
        "a fixed pixel under a moving grating must resonate at k·v: {slow:.3} (v=.3), {fast:.3} (v=.6)"
    );
    assert!(
        fast > 1.6 * slow,
        "faster motion ⇒ proportionally higher locked frequency (velocity read out): {fast:.3} vs {slow:.3}"
    );

    // (3) SPACE = TIME (spatial frequency, translation-invariant) ---------------------------------------------
    let ka = core::f32::consts::PI / 2.0; // period 4 cells (3 periods over 12)
    let kb = core::f32::consts::PI / 3.0; // period 6 cells (2 periods over 12)
    let (a0, a3) = (spatial_freq(ka, 0), spatial_freq(ka, 3)); // same grating, translated by 3 cells
    let b0 = spatial_freq(kb, 0);
    eprintln!(
        "[ND] space=time scan: grating kA={ka:.3} → locked {a0:.3} (shift 0) / {a3:.3} (shift 3);  grating kB={kb:.3} → locked {b0:.3}"
    );
    // reads the spatial frequency
    assert!(
        (a0 - ka).abs() < 0.10 && (b0 - kb).abs() < 0.10,
        "the scan must read the spatial frequency: kA {ka:.3}→{a0:.3}, kB {kb:.3}→{b0:.3}"
    );
    // TRANSLATION-INVARIANT — a shifted grating locks to the SAME frequency (relative representation, #142)
    assert!(
        (a0 - a3).abs() < 0.06,
        "spatial-frequency read-out must be translation-invariant: shift0 {a0:.3} vs shift3 {a3:.3}"
    );

    eprintln!("[ND] VERDICT: `resonate` is a full SPATIOTEMPORAL spectral front-end with ZERO new core code. (1) N-D — one resonator per cell locks to that cell's own frequency (a grown spectral map). (2) TIME — a fixed pixel under a moving grating oscillates at k·v, so `adapt_freq` reads out VELOCITY as a temporal frequency ({slow:.2}/{fast:.2} for v=.3/.6). (3) SPACE=TIME — scanning a spatial grating into event-time reads its SPATIAL frequency ({a0:.2}≈{ka:.2}), TRANSLATION-INVARIANTLY (shift 0 vs 3: {a0:.2}={a3:.2}). Space = time = one dimensional sense (#142/#137): the 1-D band-pass covers N-D and motion by scanning, exactly as arithmetic (#129) and music (#136) fell out of grow + a scan.");
}
