// SPDX-License-Identifier: MIT-0
//! REPRO (scratch): disambiguate issue #148's "three axes → one wall". Test whether each axis is really a DEPTH
//! wall, or a WIDTH (capacity) wall, or a FEATURE-GEOMETRY wall — by the cheap fix that SHOULD clear it if so:
//!   PARITY-5  : sweep max_hidden 32→64→128 — if still ~chance at 128, it's genuine DEPTH (width is futile).
//!   CHECKER   : sweep max_hidden 32→96      — if it clears, it's WIDTH/capacity, not depth.
//!   RINGS-4   : add one QUADRATIC feature r²=x0²+x1² — if it jumps, it's FEATURE-GEOMETRY, not depth.
//! feature_lr=0.3 (matching the ticket), grow (single grown layer), 2 seeds averaged.
use openmorph_core::substrate::Field;

use crate::common::Rng;

fn parity(xs: &[f32]) -> f32 {
    if xs.iter().map(|&x| x.signum()).product::<f32>() > 0.0 {
        1.0
    } else {
        -1.0
    }
}
fn checker(xs: &[f32], c: f32) -> f32 {
    let a = ((xs[0] + 1.0) * 0.5 * c).floor() as i64;
    let b = ((xs[1] + 1.0) * 0.5 * c).floor() as i64;
    if (a + b).rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}
fn rings(xs: &[f32], bands: f32) -> f32 {
    let r2 = xs[0] * xs[0] + xs[1] * xs[1]; // in [0,2]
    if ((r2 * 0.5 * bands).floor() as i64).rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

// n_raw inputs; `quad` adds r²=x0²+x1² as an extra input feature. Returns held-out accuracy.
fn run(
    n_raw: usize,
    quad: bool,
    max_hidden: usize,
    target: &dyn Fn(&[f32]) -> f32,
    seed: u64,
) -> f64 {
    let n_in = n_raw + if quad { 1 } else { 0 };
    let y = n_in;
    let mut f = Field::new(n_in + 1);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().feature_lr = 0.3;
    f.config_mut().max_hidden = max_hidden;
    let build = |xs: &[f32]| -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = xs.iter().enumerate().map(|(i, &x)| (i, x)).collect();
        if quad {
            v.push((n_raw, xs[0] * xs[0] + xs[1] * xs[1]));
        }
        v
    };
    let mut rng = Rng::new(seed);
    for _ in 0..10000 {
        let xs: Vec<f32> = (0..n_raw).map(|_| rng.sym()).collect();
        f.grow(&build(&xs), y, target(&xs), 0.05, 120);
    }
    let mut rng = Rng::new(seed ^ 0xFFFF);
    let mut c = 0usize;
    for _ in 0..2000 {
        let xs: Vec<f32> = (0..n_raw).map(|_| rng.sym()).collect();
        f.stimulate(&build(&xs), 120);
        let pred = if f.cells()[y] > 0.0 { 1.0 } else { -1.0 };
        if (pred - target(&xs)).abs() < 1e-6 {
            c += 1;
        }
    }
    c as f64 / 2000.0
}

fn avg(n_raw: usize, quad: bool, mh: usize, t: &dyn Fn(&[f32]) -> f32) -> f64 {
    (run(n_raw, quad, mh, t, 0x11) + run(n_raw, quad, mh, t, 0x22)) / 2.0
}

#[test]
#[ignore = "research: on demand"]
fn disambiguate_the_three_axes() {
    eprintln!("[PROBE] PARITY-5 (is width futile ⇒ genuine depth?):");
    for mh in [32usize, 64, 128] {
        eprintln!(
            "[PROBE]   max_hidden {mh:>3} → {:.0}%",
            avg(5, false, mh, &parity) * 100.0
        );
    }
    eprintln!("[PROBE] CHECKERBOARD (does width clear it ⇒ capacity, not depth?):");
    for &c in &[4.0f32, 6.0] {
        for mh in [32usize, 96] {
            eprintln!(
                "[PROBE]   {c:.0}×{c:.0}, max_hidden {mh:>3} → {:.0}%",
                avg(2, false, mh, &|xs| checker(xs, c)) * 100.0
            );
        }
    }
    eprintln!(
        "[PROBE] RINGS-4 (does a quadratic feature clear it ⇒ feature-geometry, not depth?):"
    );
    eprintln!(
        "[PROBE]   linear basis        → {:.0}%",
        avg(2, false, 32, &|xs| rings(xs, 4.0)) * 100.0
    );
    eprintln!(
        "[PROBE]   + quadratic r²       → {:.0}%",
        avg(2, true, 32, &|xs| rings(xs, 4.0)) * 100.0
    );
    eprintln!("[PROBE] READ: parity flat at 128 ⇒ depth wall (composition). checker clears with width ⇒ capacity. rings clears with r² ⇒ feature-geometry. If so, #148's 'one wall' is really three different problems.");
}
