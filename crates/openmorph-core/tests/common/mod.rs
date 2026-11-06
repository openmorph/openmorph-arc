// SPDX-License-Identifier: MIT-0
//! Shared deterministic test RNG for the openmorph-core integration suite.
//!
//! Every proof test seeds this xorshift generator and asserts against calibrated
//! thresholds, so the sequence MUST stay byte-identical to the per-file copies it
//! replaces. The methods below are the superset used across the suite. Each tier
//! binary (`tests/<tier>/main.rs`) declares this module ONCE
//! (`#[path = "../common/mod.rs"] mod common;`) and its proof modules reach it as
//! `crate::common::…`, using only the subset they need (hence the module-level
//! `allow(dead_code)`).
#![allow(dead_code)]

/// Deterministic xorshift64 PRNG. `Rng::new(seed)` forces the low bit so the
/// state is never zero (`seed | 1`).
pub struct Rng(pub u64);

impl Rng {
    /// Seed the generator (`seed | 1`, so the state is never zero).
    pub fn new(s: u64) -> Self {
        Rng(s | 1)
    }

    /// Raw `u64` draw (xorshift64: `<<13`, `>>7`, `<<17`).
    pub fn u(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn f(&mut self) -> f32 {
        (self.u() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform `f32` in `[0, 1)` (alias of [`Rng::f`]).
    pub fn unit(&mut self) -> f32 {
        (self.u() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform `f32` in `[-1, 1)`.
    pub fn sym(&mut self) -> f32 {
        self.f() * 2.0 - 1.0
    }

    /// Random sign: `-1.0` or `1.0`.
    pub fn bit(&mut self) -> f32 {
        if (self.u() >> 40) & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    }

    /// Uniform `usize` in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.u() % n as u64) as usize
    }

    /// Uniform `usize` in `[0, n)` (alias of [`Rng::below`]).
    pub fn range(&mut self, n: usize) -> usize {
        (self.u() % n as u64) as usize
    }

    /// Approx-gaussian noise (sum of 3 uniforms, centred) scaled by `s`.
    pub fn noise(&mut self, s: f32) -> f32 {
        ((self.f() + self.f() + self.f()) / 3.0 - 0.5) * 2.0 * s
    }
}

// ── grown forward/world model — the `imagine(option)` capability (cap #332, `substrate_imagine`) ─────────────────
//
// A `K`-member ENSEMBLE of grown forward models: each member `rng_seed`-varied so its grown backbone differs, so the
// spread across members IS the substrate's epistemic uncertainty (the same ensemble-disagreement the free-energy
// actor / `select_explore` run on). This is the shared keystone the ARC-3 reflex→plan rungs build on: #333 reads its
// σ² to direct curiosity, #334 rolls it out to plan. Zero core change — `grow_backbone` + `add_unit` + `learn_step`
// (the proven `substrate_predict` idiom). Validated in `substrate_imagine` (MAE 0.038 within 20 obs, σ² 190×
// calibrated, monotone anneal); reused here so #333/#334 build on the SAME mechanism, not a re-derivation.

use openmorph_core::substrate::Field;

/// A grown forward model: predicts an `n_out`-vector from an `n_in`-vector context, online, with a calibrated
/// epistemic σ² from ensemble disagreement.
pub struct WorldModel {
    members: Vec<Field>,
    heads: Vec<usize>,
    n_in: usize,
    lr: f32,
    budget: usize,
}

impl WorldModel {
    /// `k` members, `n_in`→`n_out` map, each member a seed-varied grown backbone of `width` features.
    pub fn new(k: usize, n_in: usize, n_out: usize, width: usize) -> Self {
        let mut heads = Vec::new();
        let members = (0..k)
            .map(|i| {
                let mut f = Field::new(n_in);
                f.rng_seed(0x1234_5678 ^ ((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                let inputs: Vec<usize> = (0..n_in).collect();
                let mut reads = inputs.clone();
                reads.extend(f.grow_backbone(&inputs, width, 2.5));
                let base = f.len();
                let hs: Vec<usize> = (0..n_out).map(|d| base + d).collect();
                for &c in &hs {
                    let n = reads.len();
                    f.add_unit(reads.clone(), vec![0.0; n], 0.0, c);
                }
                heads = hs;
                f
            })
            .collect();
        WorldModel {
            members,
            heads,
            n_in,
            lr: 0.03,
            budget: 300,
        }
    }

    /// Put every member on `genome` (e.g. `Config::legacy()` for a proof calibrated on the pre-1.6 learner).
    pub fn configure(&mut self, genome: &openmorph_core::Config) {
        for f in &mut self.members {
            f.configure(genome.clone());
        }
    }

    fn pairs(&self, ctx: &[f32]) -> Vec<(usize, f32)> {
        (0..self.n_in).map(|i| (i, ctx[i])).collect()
    }

    /// LEARN one real transition: correct every member's heads toward `target` given `ctx`.
    pub fn learn(&mut self, ctx: &[f32], target: &[f32]) {
        let inp = self.pairs(ctx);
        for m in self.members.iter_mut() {
            for (&head, &t) in self.heads.iter().zip(target.iter()) {
                m.learn_step(&inp, head, t, self.lr, self.budget);
            }
        }
    }

    /// REPLAY a buffer of transitions `epochs` times — free internal training (no new world samples).
    pub fn replay(&mut self, buffer: &[(Vec<f32>, Vec<f32>)], epochs: usize) {
        for _ in 0..epochs {
            for (ctx, tgt) in buffer {
                self.learn(ctx, tgt);
            }
        }
    }

    /// IMAGINE `ctx` without acting: `(mean prediction, epistemic σ²)`. σ² = mean-over-heads variance across members.
    pub fn imagine(&mut self, ctx: &[f32]) -> (Vec<f32>, f32) {
        let inp = self.pairs(ctx);
        let n_out = self.heads.len();
        let mut preds: Vec<Vec<f32>> = Vec::with_capacity(self.members.len());
        let heads = self.heads.clone();
        for m in self.members.iter_mut() {
            m.stimulate(&inp, self.budget);
            let c = m.cells();
            preds.push(heads.iter().map(|&h| c[h]).collect());
        }
        let k = preds.len() as f32;
        let mut mean = vec![0.0f32; n_out];
        for p in &preds {
            for d in 0..n_out {
                mean[d] += p[d] / k;
            }
        }
        let mut var = 0.0f32;
        for d in 0..n_out {
            var += preds.iter().map(|p| (p[d] - mean[d]).powi(2)).sum::<f32>() / k / n_out as f32;
        }
        (mean, var)
    }
}
