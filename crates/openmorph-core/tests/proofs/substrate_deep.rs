// SPDX-License-Identifier: MIT-0
//! rc7 — GROW-DEEP: the morph grows its own DEPTH where a linear map fails. On a NON-linearly-separable
//! task (XOR-like: class = sign(x0·x1)), a single linear unit is stuck near chance; `grow_deep_step`
//! births HIDDEN random-projection features on persistent surprise until the transducer can separate
//! it — a non-linear driver grown from surprise alone, no authored architecture. Closes MorphLab's
//! rc7 ask (the moat reaching the readout number without pre-specified structure) for non-linear data.
//!
//!   cargo test -p openmorph-core --test proofs substrate_deep -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// XOR-like: the classes are the two diagonals — not linearly separable in [x0, x1].
fn label(x0: f32, x1: f32) -> f32 {
    if x0 * x1 > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Train a field with a given hidden budget over a stream; return held-out accuracy.
fn run(max_hidden: usize, seed: u64) -> (usize, f64) {
    let mut f = Field::new(3); // x0=0, x1=1, class=2
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = max_hidden;
    let mut rng = Rng::new(seed);
    for _ in 0..8000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.grow(&[(0, x0), (1, x1)], 2, label(x0, x1), 0.05, 200);
    }
    let mut rng = Rng::new(seed ^ 0xFFFF);
    let mut correct = 0usize;
    for _ in 0..2000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], 200);
        let pred = if f.cells()[2] > 0.0 { 1.0 } else { -1.0 };
        if (pred - label(x0, x1)).abs() < 1e-6 {
            correct += 1;
        }
    }
    (f.n_units(), correct as f64 / 2000.0)
}

#[test]
fn the_morph_grows_depth_to_crack_a_nonlinear_task() {
    let (u_lin, acc_lin) = run(0, 0x111); // linear only: one output unit, no hidden
    let (u_deep, acc_deep) = run(16, 0x111); // grow up to 16 hidden features on surprise

    eprintln!(
        "[DEEP] XOR-like (not linearly separable): LINEAR {u_lin} unit → {:.0}%  |  GROW-DEEP {u_deep} units → {:.0}%  (chance 50%)",
        acc_lin * 100.0,
        acc_deep * 100.0
    );
    eprintln!("[DEEP] VERDICT (rc7): a linear map is stuck near chance; grow_deep_step BIRTHS hidden random-projection features on persistent surprise until the transducer separates the classes — the morph grows its own DEPTH from surprise alone, no authored architecture. The 'grows its own driver' moat now covers non-linear data.");
    assert!(
        acc_lin < 0.70,
        "the linear map cannot crack XOR (got {acc_lin:.2})"
    );
    assert!(
        acc_deep > 0.90,
        "growing hidden features cracks it ({acc_deep:.2})"
    );
    assert!(
        u_deep > u_lin,
        "the deep morph grew hidden units ({u_deep} vs {u_lin})"
    );
}
