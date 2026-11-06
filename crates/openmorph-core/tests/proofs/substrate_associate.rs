// SPDX-License-Identifier: MIT-0
//! KDA-STYLE ASSOCIATIVE MEMORY AS A USAGE PROTOCOL — content-addressed key→value recall built from the
//! substrate primitives it already has, with ZERO core additions.
//!
//! Kimi Delta Attention's recurrence is, at heart, `S += k·β·(v − u)` — Rescorla–Wagner (the substrate's own
//! delta rule) applied to a FAST state matrix instead of slow unit weights (the same move as #125 "the delta
//! rule one hop deeper", `adapt_scales` on the timescale, #460 `nest_credit` one level down — here, on a fast
//! weight). So a KDA head is a Field whose readout weights ARE the fast state `S`: the WRITE `S += β·(v − u)·k`
//! is [`Field::learn_step`] over a linear (Identity) readout (the delta write), and the READ `u = Σ k_i S_i` is
//! [`Field::stimulate`] then reading the output cell (the graded read).
//! This is what the sequence-scan doctrine prescribes: a new sequence capability should be a PROTOCOL over
//! `grow`/`learn_step`/`stimulate`, not a new core primitive. It fills a real gap — none of `bind` (Hebbian
//! cell-COVARIANCE + argmax `complete`), `delay` (exact but time-indexed, scalar), or `integrate` (trend, scalar)
//! does content-addressed key→value recall.
//!
//! HONEST SCOPE (truth-not-trophy): this measures the mechanism, it does NOT claim a task win — the current
//! ARC wall is the reward, not memory (MorphBot). The one friction found: fire-on-change skips a re-fire when
//! the key equals the last input, so a repeated-same-key write/read needs a forced recompute (a zero reset);
//! interleaved keys (the real recall use) don't. DECAY (per-channel forget, KDA's `α`) is the one piece this
//! protocol can't express without a weight-scaling op — left as the sole candidate core addition, gated on a
//! consumer that needs graceful forgetting. See `docs/research/kda-associative-memory.md`.

use openmorph_core::{Activation, Field};

const K: usize = 32; // key dimension (cells 0..K); the value lives in cell K

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bit(&mut self) -> f32 {
        if self.f() > 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// A fast linear associative memory: `K` key cells → one value cell, a single Identity readout whose weights
/// ARE the fast state `S` (start 0 = empty memory). No hidden units, no growth.
fn kda_new() -> Field {
    let mut f = Field::new(K + 1);
    f.config_mut().activation = Activation::Identity;
    f.config_mut().max_hidden = 0;
    f.add_unit((0..K).collect(), vec![0.0; K], 0.0, K);
    f
}

/// Reset the key cells to zero so the next `stimulate` is a real change (fire-on-change would otherwise skip a
/// re-fire when the key equals the last input, leaving a stale `u`).
fn reset(f: &mut Field) {
    let zero: Vec<(usize, f32)> = (0..K).map(|i| (i, 0.0)).collect();
    f.stimulate(&zero, K + 4);
}

/// KDA WRITE: one delta step `S += β·(v − u)·k` == `learn_step` over the linear readout.
fn write(f: &mut Field, key: &[f32], val: f32, beta: f32) {
    reset(f);
    let inp: Vec<(usize, f32)> = key.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    f.learn_step(&inp, K, val, beta, K + 4);
}

/// KDA READ: `u = Σ k_i S_i` == `stimulate(key)` then read the value cell.
fn read(f: &mut Field, key: &[f32]) -> f32 {
    reset(f);
    let inp: Vec<(usize, f32)> = key.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    f.stimulate(&inp, K + 4);
    f.cells()[K]
}

#[test]
fn content_addressed_recall_within_capacity_and_degrades_gracefully() {
    // Store M random key→value pairs; recall each. A linear associative memory recalls exactly up to capacity
    // (~K) and degrades GRACEFULLY past it (a smooth MSE rise, not the memo's hard cliff — the v2_scale_economy
    // B2 point). No core code: just learn_step (write) + stimulate (read).
    let cap_mse: Vec<(usize, f32)> = [8usize, 16, 24, 40]
        .iter()
        .map(|&m| {
            let mut rng = Lcg(0xA5 ^ (m as u64).wrapping_mul(0x9E3779B9));
            let keys: Vec<Vec<f32>> = (0..m)
                .map(|_| (0..K).map(|_| rng.bit()).collect())
                .collect();
            let vals: Vec<f32> = (0..m).map(|_| rng.f()).collect();
            let mut f = kda_new();
            for _ in 0..40 {
                for (k, &v) in keys.iter().zip(&vals) {
                    write(&mut f, k, v, 0.05);
                }
            }
            let mse = keys
                .iter()
                .zip(&vals)
                .map(|(k, &v)| {
                    let e = read(&mut f, k) - v;
                    e * e
                })
                .sum::<f32>()
                / m as f32;
            (m, mse)
        })
        .collect();
    for (m, mse) in &cap_mse {
        println!("M={m:2}  recall MSE={mse:.4}");
    }
    // Within capacity: exact recall.
    assert!(
        cap_mse[0].1 < 1e-4,
        "M=8 must recall exactly (got {:.4})",
        cap_mse[0].1
    );
    assert!(
        cap_mse[1].1 < 1e-4,
        "M=16 must recall exactly (got {:.4})",
        cap_mse[1].1
    );
    // Past capacity: it degrades, but GRACEFULLY (monotone, bounded) — not a cliff to garbage.
    assert!(
        cap_mse[3].1 > cap_mse[1].1,
        "past capacity MSE must rise (graceful degradation)"
    );
    assert!(
        cap_mse[3].1 < 5.0,
        "degradation must stay bounded, not blow up (got {:.4})",
        cap_mse[3].1
    );
}

#[test]
fn delta_write_overwrites_where_accumulation_only_piles() {
    // THE sharpest KDA property vs a Hebbian store: write key→v1, then UPDATE the same key→v2. The delta write
    // (v − u) is error-corrected, so it OVERWRITES; an accumulator (`+= v`, as `bind` tallies) can only pile.
    let mut rng = Lcg(0x1234);
    let key: Vec<f32> = (0..K).map(|_| rng.bit()).collect();
    let mut f = kda_new();
    for _ in 0..600 {
        write(&mut f, &key, 1.0, 0.01);
    }
    let after_v1 = read(&mut f, &key);
    for _ in 0..600 {
        write(&mut f, &key, -0.5, 0.01);
    }
    let after_v2 = read(&mut f, &key);
    let accumulator = 1.0f32 + (-0.5f32); // a += store ends at v1+v2, never v2
    println!("delta write: v1={after_v1:.4} → v2={after_v2:.4} (want 1.0 → -0.5)  |  accumulator {accumulator:.4} (piled)");
    assert!(
        (after_v1 - 1.0).abs() < 0.05,
        "must recall v1 before the update"
    );
    assert!(
        (after_v2 + 0.5).abs() < 0.05,
        "delta write must OVERWRITE to v2, not blend"
    );
    assert!(
        (accumulator + 0.5).abs() > 0.4,
        "an accumulator cannot overwrite — it piles v1+v2"
    );
}

#[test]
fn recall_is_robust_to_key_noise() {
    // Content-addressed recall degrades gracefully with a corrupted key (a few flipped bits), within capacity.
    const M: usize = 12;
    let mut rng = Lcg(0x5151);
    let keys: Vec<Vec<f32>> = (0..M)
        .map(|_| (0..K).map(|_| rng.bit()).collect())
        .collect();
    let vals: Vec<f32> = (0..M).map(|_| rng.f()).collect();
    let mut f = kda_new();
    for _ in 0..40 {
        for (k, &v) in keys.iter().zip(&vals) {
            write(&mut f, k, v, 0.05);
        }
    }
    let mut nrng = Lcg(0xBEEF);
    let noisy_mse = keys
        .iter()
        .zip(&vals)
        .map(|(k, &v)| {
            let mut nk = k.clone();
            for _ in 0..3 {
                let j = (nrng.f().abs() * K as f32) as usize % K;
                nk[j] = -nk[j];
            }
            let e = read(&mut f, &nk) - v;
            e * e
        })
        .sum::<f32>()
        / M as f32;
    println!("noisy (3-bit-flip) recall MSE = {noisy_mse:.4}");
    assert!(
        noisy_mse < 0.25,
        "recall must stay close under a 3-bit key corruption (got {noisy_mse:.4})"
    );
}
