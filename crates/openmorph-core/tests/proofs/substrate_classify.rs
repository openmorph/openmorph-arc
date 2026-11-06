// SPDX-License-Identifier: MIT-0
//! rc7 diagnosis (MorphLab's ask) — does surprise-birth grow ENOUGH of a transducer to CLASSIFY, or
//! does it under-grow? A multi-class stream: the morph grows its own classifier (one grown predictor
//! per class via `grow_and_learn`, no caller-specified units) and classifies by the settled argmax.
//! It should reach the linear-readout accuracy without pre-specifying the structure.
//!
//!   cargo test -p openmorph-core --test proofs substrate_classify -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const NC: usize = 3; // classes
const D: usize = 4; // features

// A linearly-separable 3-class problem: label = argmax_j (w_j · x).
const WT: [[f32; D]; NC] = [
    [1.0, 0.2, -0.3, 0.1],
    [-0.4, 0.9, 0.2, -0.2],
    [0.1, -0.3, 0.8, 0.5],
];
fn label(x: &[f32; D]) -> usize {
    (0..NC)
        .max_by(|&a, &b| {
            let sa: f32 = (0..D).map(|k| WT[a][k] * x[k]).sum();
            let sb: f32 = (0..D).map(|k| WT[b][k] * x[k]).sum();
            sa.partial_cmp(&sb).unwrap()
        })
        .unwrap()
}

/// Classify by the settled argmax over the class cells.
fn classify(f: &mut Field, inputs: &[(usize, f32)]) -> usize {
    f.stimulate(inputs, 200);
    (0..NC)
        .max_by(|&a, &b| {
            f.cells()[D + a]
                .partial_cmp(&f.cells()[D + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

#[test]
fn the_morph_grows_its_own_classifier_and_reaches_the_readout_number() {
    let mut f = Field::new(D + NC);
    f.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    f.config_mut().birth_gate = 0.05;

    // STREAM: grow one predictor per class by surprise-birth; one-vs-rest target (+1 true / −1 else).
    let mut rng = Rng::new(0xC1A);
    for _ in 0..6000 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let y = label(&x);
        let inputs: Vec<(usize, f32)> = (0..D).map(|k| (k, x[k])).collect();
        for j in 0..NC {
            let tgt = if j == y { 1.0 } else { -1.0 };
            f.grow(&inputs, D + j, tgt, 0.05, 200);
        }
    }
    let grown_units = f.n_units();

    // Held-out accuracy of the GROWN classifier.
    let mut rng = Rng::new(0xBEE7);
    let mut correct = 0usize;
    for _ in 0..1000 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..D).map(|k| (k, x[k])).collect();
        if classify(&mut f, &inputs) == label(&x) {
            correct += 1;
        }
    }
    let acc = correct as f64 / 1000.0;

    eprintln!("[CLASSIFY] grown units = {grown_units} (expected {NC}: one per class); held-out accuracy {:.3} (chance {:.3})", acc, 1.0 / NC as f64);
    assert_eq!(
        grown_units, NC,
        "grew one predictor per class ({grown_units})"
    );
    assert!(
        acc > 0.85,
        "the grown classifier must reach the linear-readout level, not chance (got {acc:.3})"
    );
}
