// SPDX-License-Identifier: MIT-0
//! FEATURE-DIRECTION BATCHING — `observe` learns every prediction head on ONE settled state, so each shared
//! feature's one-hop direction step is summed over the heads and applied once. That is the same update as moving
//! the feature once per head (every factor but the head's backprojected error is fixed by the settled state), at
//! `O(H·W + W·fan-in)` per observation instead of `O(H·W·fan-in)`. These tests pin it against the per-head
//! sequence written with the public calls `observe_from` is made of:
//! - one head ⇒ bit-identical (nothing to sum);
//! - many heads over a SHARED grown basis ⇒ the same organism up to f32 summation order, and the directions
//!   really moved (the mechanism is doing work, not idling at `feature_lr = 0`);
//! - heads that read each other's cells ⇒ the same, because pending steps land before an affected head learns;
//! - frozen directions (`Config::legacy()`) ⇒ bit-identical.
//!
//!   cargo test -p openmorph-core --test proofs substrate_direction_batch -- --nocapture

use openmorph_core::substrate::Field;
use openmorph_core::Config;

const W: usize = 12; // shared basis width
const LR: f32 = 0.05;
const BUDGET: usize = usize::MAX;

/// `dim` observed cells at 0..dim, `dim` prediction heads at dim..2·dim, all reading ONE grown basis.
fn world_model(dim: usize, cfg: Config) -> Field {
    world_model_wired(dim, cfg, false)
}

/// `cross` ⇒ head `i` also reads head `i+1`'s cell (weight 0.5): the heads are features of each other, so a head's
/// direction step moves the NEXT head's readout before that head learns.
fn world_model_wired(dim: usize, cfg: Config, cross: bool) -> Field {
    let mut f = Field::new(2 * dim);
    f.configure(cfg);
    f.rng_seed(11);
    let obs: Vec<usize> = (0..dim).collect();
    let bb = f.grow_backbone(&obs, W, 1.0);
    for i in 0..dim {
        let mut reads = bb.clone();
        let mut w: Vec<f32> = (0..bb.len())
            .map(|k| (((k * 7 + i * 3) % 11) as f32 / 11.0 - 0.5) * 0.2)
            .collect();
        if cross && i + 1 < dim {
            reads.push(dim + i + 1);
            w.push(0.5);
        }
        f.add_unit(reads, w, 0.0, dim + i);
    }
    f
}

/// A moving bump with a sine-modulated amplitude: every head has something to predict.
fn frame(dim: usize, t: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let on = (i + t) % dim < 2;
            if on {
                0.9 * (0.3 * t as f32).sin().abs() + 0.1
            } else {
                -0.1
            }
        })
        .collect()
}

/// `observe` spelled out head by head through the public calls it is made of — each head's direction step lands
/// before the next head learns (the pre-batching sequence).
fn observe_per_head(f: &mut Field, dim: usize, values: &[f32]) -> Vec<f32> {
    let ctx: Vec<(usize, f32)> = (0..dim).map(|i| (i, f.cells()[i])).collect();
    let mut errs = Vec::with_capacity(dim);
    for (i, &want) in values.iter().enumerate() {
        errs.push(want - f.cells()[dim + i]);
        if i == 0 {
            f.learn_step(&ctx, dim, want, LR, BUDGET);
        } else {
            f.learn_settled(dim + i, want, LR);
        }
    }
    let take: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    f.stimulate(&take, BUDGET);
    errs
}

fn params(f: &Field) -> Vec<f32> {
    f.units
        .iter()
        .flat_map(|u| u.weights.iter().copied().chain(core::iter::once(u.bias)))
        .collect()
}

fn run(dim: usize, cfg: Config, steps: usize) -> (Field, Field, f32, f32) {
    run_wired(dim, cfg, steps, false)
}

fn run_wired(dim: usize, cfg: Config, steps: usize, cross: bool) -> (Field, Field, f32, f32) {
    let mut batched = world_model_wired(dim, cfg.clone(), cross);
    let mut per_head = world_model_wired(dim, cfg, cross);
    let (mut eb, mut ep) = (0.0f32, 0.0f32);
    for t in 0..steps {
        let x = frame(dim, t);
        let a = batched.observe(0, dim, dim, &x, LR, BUDGET);
        let b = observe_per_head(&mut per_head, dim, &x);
        if t + 50 >= steps {
            eb += a.iter().map(|e| e.abs()).sum::<f32>();
            ep += b.iter().map(|e| e.abs()).sum::<f32>();
        }
    }
    (batched, per_head, eb, ep)
}

#[test]
fn one_head_is_bit_identical_to_the_per_head_sequence() {
    let (a, b, _, _) = run(1, Config::default(), 300);
    let (pa, pb) = (params(&a), params(&b));
    assert_eq!(pa.len(), pb.len());
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "one head: nothing to sum, so nothing may differ"
        );
    }
    for (x, y) in a.cells().iter().zip(b.cells()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn shared_basis_heads_match_the_per_head_sequence_up_to_rounding() {
    let dim = 24;
    let before = params(&world_model(dim, Config::default()));
    let (a, b, err_batched, err_per_head) = run(dim, Config::default(), 400);
    let (pa, pb) = (params(&a), params(&b));
    let worst = pa
        .iter()
        .zip(&pb)
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0f32, f32::max);
    // The shared basis really learned its directions (the batch is not an idle no-op).
    let moved = pa
        .iter()
        .zip(&before)
        .take(W * (dim + 1))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "dim {dim}: worst relative param gap {worst:.2e}; basis moved up to {moved:.3}; \
         late |err| batched {err_batched:.4} vs per-head {err_per_head:.4}"
    );
    assert!(
        moved > 1e-2,
        "feature directions must move under the default genome"
    );
    assert!(
        worst < 1e-3,
        "batched and per-head updates are the same update up to f32 order"
    );
    assert!(
        (err_batched - err_per_head).abs() <= 1e-3 * (1.0 + err_per_head),
        "the same organism predicts the same"
    );
}

#[test]
fn frozen_directions_are_bit_identical() {
    let (a, b, _, _) = run(16, Config::legacy(), 200);
    for (x, y) in params(&a).iter().zip(&params(&b)) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn heads_that_read_each_other_still_match_the_per_head_sequence() {
    // A head whose cell is a FEATURE of an earlier head has its readout moved by that head's step before it learns
    // (per head); the batch lands its pending steps first, so the two stay the same update up to f32 order.
    let dim = 24;
    let (a, b, err_batched, err_per_head) = run_wired(dim, Config::default(), 400, true);
    let worst = params(&a)
        .iter()
        .zip(&params(&b))
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0f32, f32::max);
    println!("cross-reading heads: worst relative param gap {worst:.2e}; late |err| {err_batched:.4} vs {err_per_head:.4}");
    assert!(
        worst < 1e-3,
        "cross-reading heads: batched and per-head are the same update"
    );
    assert!((err_batched - err_per_head).abs() <= 1e-3 * (1.0 + err_per_head));
}
