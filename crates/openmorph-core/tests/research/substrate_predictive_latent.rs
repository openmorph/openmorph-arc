// SPDX-License-Identifier: MIT-0
//! SPIKE (go/no-go, not yet a shipped proof) — the reward-free predictive LATENT, the M2/OM-1 crux.
//!
//! Claim under test: a grown COMPACT latent, trained ONLY by self-supervised next-frame prediction (no reward,
//! no labels), extracts a STABLE per-state code from a signal BURIED IN NOISE — where a random projection
//! cannot. The world: K states in a fixed cycle; each frame = a SMALL predictive signal `a·v_state` (the state's
//! identity, the only predictable thing) plus LARGER fresh noise `b·ξ` (unpredictable). Predicting the next
//! frame is possible only for the signal part, so minimizing prediction error must drive the encoder to extract
//! `v_state` and ignore the noise — the abstraction ARC needs (collapse the exploding raw stream to a repeating
//! state code).
//!
//! The mechanism is ASSEMBLED from existing primitives, no new core: `grow_backbone` (the encoder = features over
//! the frame → a narrow latent band), a linear predictor latent→next-frame, and `learn_step` with
//! `config.feature_lr` (#125) — which trains the encoder's directions one hop back from the prediction residual
//! (feedback-alignment). The ONLY difference between the two arms is `feature_lr`: 0 = random projection (control),
//! >0 = learned predictive encoder.
//!
//! GO = the learned latent's per-state separation clearly beats the random projection's. NO-GO (honest) = it
//! ties, meaning feature_lr does not extract the buried signal and a different encoder is needed.
//!
//!   cargo test -p openmorph-core --test research substrate_predictive_latent -- --ignored --nocapture --test-threads=1
use openmorph_core::substrate::Field;

use crate::common::Rng;

const F: usize = 32; // frame dimensionality (the raw high-dim observation)
const L: usize = 4; // latent bottleneck width
const K: usize = 3; // number of cyclic world-states
                    // Regime: the signal is recoverable from ONE frame ONLY if the encoder aligns with v_state. Projected onto the
                    // RIGHT direction (v_state) the signal A=1.0 beats the in-direction noise std ≈ B/√3 ≈ 0.58; projected onto a
                    // RANDOM direction the signal shrinks to ≈ A/√F ≈ 0.18 and the noise (0.58) swamps it. So a learned encoder can
                    // separate the states and a random projection cannot — the discriminating regime.
const A: f32 = 1.0; // signal amplitude (predictable state identity)
const B: f32 = 1.0; // noise amplitude (unpredictable)

struct World {
    v: Vec<Vec<f32>>, // K unit signal vectors
}
impl World {
    fn new(rng: &mut Rng) -> Self {
        let v = (0..K)
            .map(|_| {
                let raw: Vec<f32> = (0..F)
                    .map(|_| (rng.u() % 1000) as f32 / 500.0 - 1.0)
                    .collect();
                let n = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                raw.iter().map(|x| x / n).collect()
            })
            .collect();
        World { v }
    }
    fn render(&self, s: usize, rng: &mut Rng) -> Vec<f32> {
        (0..F)
            .map(|i| A * self.v[s][i] + B * ((rng.u() % 1000) as f32 / 500.0 - 1.0))
            .collect()
    }
}

/// Build the encoder (frame → L-cell latent) + linear predictor (latent → F-cell prediction), train it on the
/// self-supervised next-frame stream with the given `feature_lr`, then return the per-state code separation
/// (between-state gap / within-state spread — scale-free; higher = a more stable, distinct per-state code).
fn run(feature_lr: f32, seed: u64) -> f32 {
    let mut f = Field::new(F);
    f.config_mut().feature_lr = feature_lr;
    f.rng_seed(0xC0DE ^ seed);
    let mut rng = Rng::new(seed);
    let world = World::new(&mut rng);

    // encoder: L grown random-projection features over the frame → the latent band. Scale ≈ 1/√F keeps the
    // tanh in its LINEAR regime (a saturated encoder has ~zero derivative, so feature_lr cannot train it).
    let latent = f.grow_backbone(&(0..F).collect::<Vec<_>>(), L, 0.2);
    // predictor: F linear readouts latent → prediction band (small random init so feature_lr has a gradient).
    let pred0 = latent.iter().copied().max().unwrap() + 1;
    for i in 0..F {
        let w: Vec<f32> = (0..L)
            .map(|_| ((rng.u() % 1000) as f32 / 5000.0) - 0.1)
            .collect();
        f.add_unit(latent.clone(), w, 0.0, pred0 + i);
    }

    // TRAIN — self-supervised: predict frame_{t+1} from the latent of frame_t. Fresh noise every step, so the
    // unpredictable noise averages out of the gradient and only the signal trains the encoder.
    let mut s = 0usize;
    let mut frame = world.render(s, &mut rng);
    for _ in 0..14000 {
        let s_next = (s + 1) % K;
        let next = world.render(s_next, &mut rng);
        let inputs: Vec<(usize, f32)> = (0..F).map(|i| (i, frame[i])).collect();
        for (i, &n) in next.iter().enumerate() {
            f.learn_step(&inputs, pred0 + i, n, 0.05, 200);
        }
        s = s_next;
        frame = next;
    }

    // EVALUATE — freeze, and read the latent CODE for many noisy frames of each state.
    f.crystallize();
    let mut codes: Vec<Vec<Vec<f32>>> = vec![Vec::new(); K]; // codes[state] = list of latent vectors
    for (st, code) in codes.iter_mut().enumerate() {
        for _ in 0..24 {
            let fr = world.render(st, &mut rng);
            let inputs: Vec<(usize, f32)> = (0..F).map(|i| (i, fr[i])).collect();
            f.stimulate(&inputs, 200);
            code.push(latent.iter().map(|&c| f.cells()[c]).collect());
        }
    }
    let d = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    };
    // within-state spread
    let mut within = 0.0f32;
    let mut wn = 0.0f32;
    let mut centroids = Vec::new();
    for code in &codes {
        for i in 0..code.len() {
            for j in (i + 1)..code.len() {
                within += d(&code[i], &code[j]);
                wn += 1.0;
            }
        }
        let ctr: Vec<f32> = (0..L)
            .map(|a| code.iter().map(|c| c[a]).sum::<f32>() / code.len() as f32)
            .collect();
        centroids.push(ctr);
    }
    within = (within / wn).max(1e-6);
    let mut between = f32::MAX;
    for i in 0..K {
        for j in (i + 1)..K {
            between = between.min(d(&centroids[i], &centroids[j]));
        }
    }
    between / within
}

#[test]
#[ignore = "research: on demand"]
fn predictive_latent_extracts_a_stable_state_code_from_noise() {
    // Average over a few seeds — the claim is about the mechanism, not one lucky projection.
    let seeds = [0x11u64, 0x22, 0x33];
    let rand_sep: f32 = seeds.iter().map(|&s| run(0.0, s)).sum::<f32>() / seeds.len() as f32;
    let learn_sep: f32 = seeds.iter().map(|&s| run(0.3, s)).sum::<f32>() / seeds.len() as f32;

    eprintln!("[PRED-LATENT] world: {K} cyclic states, signal A={A} buried under noise B={B} (noise/signal = {:.1}×)", B / A);
    eprintln!(
        "[PRED-LATENT] per-state code separation (between/within, higher=stabler distinct code):"
    );
    eprintln!("[PRED-LATENT]   random projection (feature_lr=0.0): {rand_sep:.2}");
    eprintln!("[PRED-LATENT]   learned predictive latent (feature_lr=0.3): {learn_sep:.2}");
    eprintln!(
        "[PRED-LATENT]   → learned/random = {:.1}×",
        learn_sep / rand_sep.max(1e-6)
    );

    // GO condition (the decision: is reward-free predictive growth worth wiring to ARC?): the learned encoder
    // must (a) clearly beat the random-projection null — ≥2× — proving the effect is the LEARNING, not the
    // basis; and (b) reach genuine separability — between-state gap > within-state spread (>1.0) — so the code
    // actually distinguishes the states despite the equal-amplitude noise. (An earlier arbitrary 1.5 absolute bar
    // was replaced by this decision-relevant criterion.)
    let go = learn_sep > 2.0 * rand_sep && learn_sep > 1.0;
    eprintln!(
        "[PRED-LATENT] VERDICT: {} — {}",
        if go { "GO" } else { "NO-GO" },
        if go {
            "reward-free predictive growth (grow_backbone + feature_lr, self-supervised next-frame target) extracts a STABLE, SEPARABLE per-state latent from a signal fully buried in equal noise — where a random projection cannot. The M2 abstraction mechanism works; wire it through seed_coord to the live six."
        } else {
            "feature_lr does not extract the buried signal at this width/regime; the predictive-latent path needs a different encoder before wiring to ARC (honest negative)."
        }
    );
    assert!(go, "predictive latent must beat random projection: learned {learn_sep:.2} vs random {rand_sep:.2}");
}
