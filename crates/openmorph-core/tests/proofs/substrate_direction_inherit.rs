// SPDX-License-Identifier: MIT-0
//! REPRODUCE DRAWS A FRESH DIRECTION — the #126 direction inheritance, measured and removed (2026-09, wall 2).
//!
//! #126 made `reproduce` copy the parent's LEARNED input-direction (plus a σ jitter) into the offspring whenever
//! `feature_lr > 0`, so "good directions spread". But `reproduce` is `repopulate`'s heredity operator, and there the
//! parent is the fittest SURVIVOR: it lives on. A copy of its direction duplicates a feature the organism already has,
//! and repopulating every starved feature from the one fittest parent collapses the basis. Measured on the 1.6 default
//! (`substrate_evolve`, 3-parity, 6-feature backbone, economy-driven): with inheritance the loop reached 67 % accuracy
//! and a population scale 0.3 → 0.92 (0.43 over six distinct organisms); with a fresh draw 85 % and 0.3 → 2.16 (1.35
//! over distinct organisms). No proof ever showed inheritance helping a task; its proof (this file, before) showed
//! only that the child pointed the parent's way (|cosine| > 0.7).
//!
//! Now the offspring's direction is always a fresh unit-variance draw (the same rng draws as before, so only values
//! change), and `feature_lr` refines it from there. `fork_mutated` (a whole-organism fork, where the parent's copy IS
//! the variant) still jitters learned directions. This file proves the new property: an offspring does not copy its
//! living parent, with or without direction learning.
//!
//!   cargo test -p openmorph-core --test proofs substrate_direction_inherit -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

fn target(x: &[f32]) -> f32 {
    let mut s = 0.0;
    let mut i = 0;
    while i + 1 < x.len() {
        s += x[i] * x[i + 1];
        i += 2;
    }
    (2.5 * s).tanh()
}

fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 1e-6 || nb <= 1e-6 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn weights_of(f: &Field, cell: usize) -> Vec<f32> {
    f.units
        .iter()
        .find(|u| u.write == cell)
        .map(|u| u.weights.clone())
        .unwrap()
}

/// Learn directions on the task, then reproduce a child off a matured parent; return |cosine| between the
/// child's direction and the parent's learned one.
fn inherit_cosine(feature_lr: f32, seed: u64) -> f32 {
    let dim = 8;
    let mut f = Field::new(dim);
    f.rng_seed(0xD1 ^ seed);
    f.config_mut().feature_lr = feature_lr;
    let inputs: Vec<usize> = (0..dim).collect();
    let feats = f.grow_backbone(&inputs, 8, 1.5);
    let out = f.len();
    f.add_unit(feats.clone(), vec![0.0; feats.len()], 0.0, out);

    let mut rng = Rng::new(seed);
    for _ in 0..20_000 {
        let x: Vec<f32> = (0..dim).map(|_| rng.bit()).collect();
        f.learn_step(&pairs(&x), out, target(&x), 0.05, 256);
    }

    // Parent = a matured feature; child = another feature reborn as its offspring.
    let (parent, child) = (feats[0], feats[1]);
    let parent_dir = weights_of(&f, parent);
    f.reproduce(child, parent);
    let child_dir = weights_of(&f, child);
    cosine(&parent_dir, &child_dir).abs()
}

#[test]
fn an_offspring_draws_a_fresh_direction_not_a_copy_of_its_living_parent() {
    let seeds: [u64; 5] = [0x101, 0x115, 0x129, 0x13d, 0x151];
    let learned: f32 =
        seeds.iter().map(|&s| inherit_cosine(0.3, s)).sum::<f32>() / seeds.len() as f32;
    let random: f32 =
        seeds.iter().map(|&s| inherit_cosine(0.0, s)).sum::<f32>() / seeds.len() as f32;

    eprintln!(
        "[REPRODUCE] |cosine(child direction, parent direction)| after reproduce, mean of 5 seeds:"
    );
    eprintln!("  feature_lr = 0.3 (directions learned): {learned:.3}");
    eprintln!("  feature_lr = 0.0 (directions random):  {random:.3}");
    eprintln!("[REPRODUCE] VERDICT: the offspring of a living parent is a NEW feature, not a near-copy of one the organism already has — with or without direction learning (the #126 copy, which scored |cosine| > 0.7 here, cost the economy loop 18 accuracy points in `substrate_evolve`).");

    assert!(
        learned < 0.5,
        "with feature_lr > 0 the offspring must draw a fresh direction, not copy its living parent (|cosine| {learned:.3})"
    );
    assert!(
        random < 0.5,
        "with feature_lr = 0 the offspring draws an uncorrelated random direction (|cosine| {random:.3})"
    );
}
