// SPDX-License-Identifier: MIT-0
//! REPRO (scratch): is composition the fix for checkerboard/rings (like parity), or are they a different wall?
//! The shared sub-problem of both is a 1-D k-BAND (k alternating intervals of one scalar). Checkerboard =
//! XOR(band(x0), band(x1)); rings = band(r²). So:
//!   • if a flat basis SOLVES a 1-D k-band, then checkerboard/rings are COMPOSITIONAL — grow the band brick,
//!     combine — and #148's depth fix applies to all three.
//!   • if the flat basis WALLS on a 1-D k-band, the sub-skill itself can't be grown, so it's a RESOLUTION /
//!     training-regime wall, NOT depth — composition can't help (nothing to bank).
//! Also: feed the intermediates directly (the arm-C test) to confirm the combine step is the easy part.
use openmorph_core::substrate::Field;

use crate::common::Rng;

fn band1d(x: f32, k: f32) -> f32 {
    if (((x + 1.0) * 0.5 * k).floor() as i64).rem_euclid(2) == 0 {
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
    let r2 = xs[0] * xs[0] + xs[1] * xs[1];
    if ((r2 * 0.5 * bands).floor() as i64).rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

// n_raw inputs + optional computed intermediates; returns held-out accuracy (2 seeds averaged).
fn run(n_raw: usize, extra: &dyn Fn(&[f32]) -> Vec<f32>, target: &dyn Fn(&[f32]) -> f32) -> f64 {
    let mut tot = 0.0;
    for seed in [0x11u64, 0x22] {
        let xex = extra(&vec![0.0; n_raw]).len();
        let n_in = n_raw + xex;
        let y = n_in;
        let mut f = Field::new(n_in + 1);
        f.config_mut().birth_gate = 0.15;
        f.config_mut().feature_lr = 0.3;
        f.config_mut().max_hidden = 32;
        let build = |xs: &[f32]| -> Vec<(usize, f32)> {
            let mut v: Vec<(usize, f32)> = xs.iter().enumerate().map(|(i, &x)| (i, x)).collect();
            for (j, e) in extra(xs).into_iter().enumerate() {
                v.push((n_raw + j, e));
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
        tot += c as f64 / 2000.0;
    }
    tot / 2.0
}

fn none(_: &[f32]) -> Vec<f32> {
    Vec::new()
}

#[test]
#[ignore = "research: on demand"]
fn is_the_shared_subproblem_solvable() {
    eprintln!("[COMP] 1-D k-BAND (the shared sub-skill — is RESOLUTION itself solvable by the flat basis?):");
    for k in [2.0f32, 3.0, 4.0, 6.0] {
        eprintln!(
            "[COMP]   k={k:.0} → {:.0}%",
            run(1, &none, &|xs| band1d(xs[0], k)) * 100.0
        );
    }
    eprintln!("[COMP] CHECKERBOARD-4:");
    eprintln!(
        "[COMP]   flat                 → {:.0}%",
        run(2, &none, &|xs| checker(xs, 4.0)) * 100.0
    );
    eprintln!(
        "[COMP]   + 1-D band(x0),band(x1) fed → {:.0}%  (combine step = a 2-input XOR)",
        run(
            2,
            &|xs| vec![band1d(xs[0], 4.0), band1d(xs[1], 4.0)],
            &|xs| checker(xs, 4.0)
        ) * 100.0
    );
    eprintln!("[COMP] RINGS-4:");
    eprintln!(
        "[COMP]   + r² fed              → {:.0}%",
        run(2, &|xs| vec![xs[0] * xs[0] + xs[1] * xs[1]], &|xs| rings(
            xs, 4.0
        )) * 100.0
    );
    eprintln!("[COMP] READ: if 1-D k-band solves for all k, checkerboard/rings are compositional (depth fix applies). If it walls at k≥4, the sub-skill can't be grown ⇒ resolution/training wall, NOT depth — composition has nothing to bank.");
}
