// SPDX-License-Identifier: MIT-0
//! THE STEP-STABILITY CAP BOUNDS THE RATE, NOT ITS MODULATORS — `Config::stability_cap` (κ, 0.5 by default since
//! 1.6) caps the delta rule's BASE rate — the call's `lr` × the unit's heritable `lr` trait — at `κ/(1+Σx²)`, and only
//! THEN applies the step's modulators: plasticity (`set_plasticity` / `consolidate` / reliability-gated
//! consolidation), the write-gate (`set_gate`) and the surprise-annealed gain (`plasticity_anneal`), each in `[0,1]`.
//! The step therefore stays stable, and every modulator keeps its documented meaning where the cap binds.
//!
//! On a 200-read readout at `lr` 0.05 (raw step `η·(1+Σx²)` ≈ 3.4, so the cap binds), one `learn_step` from zero
//! weights:
//!
//! 1. a half-plastic unit moves EXACTLY half as far as a fully plastic one, a quarter-plastic one a quarter (before
//!    the fix the whole product was clipped, so plasticity 0.5 and 0.25 gave weights bit-identical to 1.0);
//! 2. a write-gate held at 0.5 halves the step the same way;
//! 3. any two base rates above the cap take the same capped step — the cap itself — while on a narrow readout
//!    whose raw step is stable the cap is inert: weights bit-identical to the raw rule (`stability_cap` 0).
//!
//!   cargo test -p openmorph-core --test proofs substrate_stability_cap -- --nocapture

use openmorph_core::{Config, Field};

const N: usize = 200;

/// A deterministic input vector in `[-1, 1]` (xorshift, independent of the organism's PRNG).
fn inputs(n: usize) -> Vec<(usize, f32)> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (i, (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
        })
        .collect()
}

/// One `learn_step` of a zero-initialized `n`-read readout toward 0.5, with the unit's plasticity `p` and (when
/// `Some`) a write-gate cell held at `gate`. Returns the readout's weights after the step.
fn step(cfg: Config, n: usize, lr: f32, p: f32, gate: Option<f32>) -> Vec<f32> {
    let out = n;
    let gate_cell = n + 1;
    let mut f = Field::new(n + 2);
    f.configure(cfg);
    f.add_unit((0..n).collect(), vec![0.0; n], 0.0, out);
    f.set_plasticity(&[out], p);
    let mut x = inputs(n);
    if let Some(g) = gate {
        f.set_gate(&[out], gate_cell);
        x.push((gate_cell, g));
    }
    f.learn_step(&x, out, 0.5, lr, 10_000);
    f.units[0].weights.clone()
}

fn norm(w: &[f32]) -> f32 {
    w.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[test]
fn plasticity_and_the_write_gate_scale_the_capped_step() {
    let d = Config::default();
    assert!(d.stability_cap > 0.0, "the 1.6 default caps the step");
    let energy: f32 = 1.0 + inputs(N).iter().map(|&(_, x)| x * x).sum::<f32>();
    let lr = 0.05;
    eprintln!(
        "[CAP] {N} reads, lr {lr}: raw step·(1+Σx²) = {:.2} > κ {} — the cap binds",
        lr * energy,
        d.stability_cap
    );
    assert!(
        lr * energy > d.stability_cap,
        "the probe must be in the capped regime"
    );

    let full = step(d.clone(), N, lr, 1.0, None);
    for p in [0.5f32, 0.25] {
        let part = step(d.clone(), N, lr, p, None);
        eprintln!(
            "[CAP] plasticity {p}: |Δw| {:.4} vs {:.4} at plasticity 1 (ratio {:.3})",
            norm(&part),
            norm(&full),
            norm(&part) / norm(&full)
        );
        for (a, b) in part.iter().zip(&full) {
            assert_eq!(
                a.to_bits(),
                (b * p).to_bits(),
                "plasticity {p} must take exactly {p}× the capped step"
            );
        }
    }
    let open = step(d.clone(), N, lr, 1.0, Some(1.0));
    let half = step(d.clone(), N, lr, 1.0, Some(0.5));
    eprintln!(
        "[CAP] write-gate 0.5: |Δw| {:.4} vs {:.4} with the gate open",
        norm(&half),
        norm(&open)
    );
    for (a, b) in half.iter().zip(&open) {
        assert_eq!(
            a.to_bits(),
            (b * 0.5).to_bits(),
            "a gate held at 0.5 must take exactly half the capped step"
        );
    }
}

#[test]
fn the_cap_bounds_the_base_rate_and_leaves_a_stable_step_untouched() {
    let d = Config::default();
    // Above the cap every base rate takes the same (capped) step — that is the cap.
    let a = step(d.clone(), N, 0.05, 1.0, None);
    let b = step(d.clone(), N, 0.5, 1.0, None);
    assert_eq!(a, b, "two rates above the cap take the same capped step");

    // A narrow readout at a stable raw step: the cap never enlarges or shrinks it — bit-identical to the raw rule.
    let raw = Config {
        stability_cap: 0.0,
        ..d.clone()
    };
    let n = 3;
    let lr = 0.05;
    let energy: f32 = 1.0 + inputs(n).iter().map(|&(_, x)| x * x).sum::<f32>();
    assert!(
        lr * energy < d.stability_cap,
        "the narrow probe is below the cap"
    );
    for p in [1.0f32, 0.5] {
        let capped = step(d.clone(), n, lr, p, None);
        let unc = step(raw.clone(), n, lr, p, None);
        assert_eq!(
            capped.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
            unc.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
            "below the cap the step is the raw one, bit for bit (plasticity {p})"
        );
    }
}
