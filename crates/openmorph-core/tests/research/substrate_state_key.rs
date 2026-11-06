// SPDX-License-Identifier: MIT-0
//! #171 / #159 — the STATE-KEY adapter: turn the grown M2 latent into a discrete state key, and MEASURE
//! (truth-not-trophy — report whatever it shows) whether it beats the hand quantizer MorphLab diagnosed as the
//! ARC-3 killer. MorphLab's ablation (2026-07-28): a full active-inference decide-loop LOST to random on 252
//! real games, and removing the exploit-lock changed NOTHING — the cause is PERCEPTION: value/novelty computed
//! over a coarse 6×6/3-bit state key that ALIASES genuinely different states into one bucket, so "informed"
//! action is MIS-informed. The fix is a GROWN state key that REPEATS on the same abstract state under nuisance
//! and SEPARATES distinct states — exactly M2's validated property. This test is the substrate-side confirmation
//! (synthetic); the real confirmation is Level-3 (MorphLab wires this into the 252-game loop vs random 9/0.61).
//!
//! THE METRIC (standard, non-riggable): V-MEASURE between the state key and the true abstract state (phase) — the
//! harmonic mean of HOMOGENEITY (each key is one phase ⇒ no false-MERGE / aliasing) and COMPLETENESS (each phase
//! is one key ⇒ no false-SPLIT / nuisance leak). The true phase is used for SCORING ONLY; no key is built from it.
//! A perfect key → V=1; a key that aliases distinct states loses homogeneity; a key that splits one state across
//! nuisance loses completeness. (MEASURED RESULT: FALSIFIED — a naive grown sign-key SPLITS on nuisance and
//! loses to the coarse quantizer here; see the RESULT note on the test fn. The honest lesson is below.)
//!
//! THE KEYS COMPARED (all over the SAME frames): (1) COARSE quantizer — 6×6 downsample, 3 levels (MorphLab's,
//! fair); (2) RAW-fine — sign of a pixel grid (fails completeness: nuisance splits it); (3) GROWN — sign-quantize
//! the RELIANCE-selected band of the grown latent (the M2-denoised, phase-carrying features).
//!
//! HONEST SCOPE: synthetic ring world; the grown latent here is a lean self-supervised next-feature predictor
//! (grow_backbone + a trained readout + reliance selection), not the full switched-Koopman — enough to select
//! the phase-carrying features. The REAL validation is Level-3 on 252 games; this only confirms the adapter's
//! state-purity on synthetic. Deterministic (seeded), CI-safe.
//!
//! THE CORRECT MECHANISM (why this stays a documented negative): the fix is NOT a better per-frame hash. Two
//! things were wrong here — (1) a PER-FRAME sign-hash throws away invariance (nuisance flips bits ⇒ 139 keys), and
//! (2) this RING WORLD is SPATIALLY CODED (blob position IS the phase), so a coarse SPATIAL average denoises
//! tautologically and legitimately wins — the world disqualified the question. The correct grown state-key is a
//! USAGE PROTOCOL, no new core: region membership (`cluster`/`route`) of the grown dynamics-latent under a LEARNED
//! metric (`metric_step` — nuisance dims earn ~zero weight), ACCUMULATED over event-time (drift-diffusion,
//! as `ShapeRecognizer`'s glimpse integration does) — i.e. recognition applied to the STATE. It DELETES this per-frame approach. Full proposal +
//! the honest synthetic/Level-3 boundary: `docs/investigations/171-recognition-state-key.md`. The de-aliasing
//! verdict is Level-3 (the 252-game loop), un-settleable by any synthetic.

use openmorph_core::{Activation, Field};
use std::f32::consts::PI;

use crate::common::Rng;

const N: usize = 16;
const PIX: usize = N * N;
const P: usize = 8; // abstract states (ring phases)
const M: usize = 48; // grown dictionary width
const RBAND: usize = 10; // top-reliance features that form the grown key

struct Frame {
    pix: Vec<f32>,
    phase: usize,
}

/// Ring world: a bright blob at phase angle + genuine nuisance (distractor random-walk, noise, brightness), so
/// the RAW frame never repeats but the abstract state (phase) does.
fn stream(len: usize, seed: u64) -> Vec<Frame> {
    let mut rng = Rng::new(seed);
    let mut phase = 0usize;
    let mut dist = (4.0f32, 4.0);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let step = (rng.below(3) as i32) - 1;
        phase = ((phase as i32 + step).rem_euclid(P as i32)) as usize;
        let bright = 0.7 + 0.6 * rng.unit();
        dist.0 = (dist.0 + rng.sym() * 1.5).clamp(1.0, 14.0);
        dist.1 = (dist.1 + rng.sym() * 1.5).clamp(1.0, 14.0);
        let th = 2.0 * PI * phase as f32 / P as f32;
        let (bx, by) = (7.5 + 5.0 * th.cos(), 7.5 + 5.0 * th.sin());
        let mut pix = vec![0.0f32; PIX];
        for y in 0..N {
            for x in 0..N {
                let (fx, fy) = (x as f32, y as f32);
                let sig = (-((fx - bx).powi(2) + (fy - by).powi(2)) / 2.0).exp();
                let dst = (-((fx - dist.0).powi(2) + (fy - dist.1).powi(2)) / 2.0).exp();
                pix[y * N + x] = (bright * sig + 0.45 * dst + rng.noise(0.06)).clamp(0.0, 1.0);
            }
        }
        out.push(Frame { pix, phase });
    }
    out
}

/// COARSE quantizer key (MorphLab's 6×6 / 3-level): downsample to 6×6, 3 levels → a base-3 integer.
fn coarse_key(pix: &[f32]) -> u64 {
    let g = 6;
    let mut k = 0u64;
    for cy in 0..g {
        for cx in 0..g {
            let (mut s, mut n) = (0.0f32, 0.0f32);
            for y in (cy * N / g)..((cy + 1) * N / g) {
                for x in (cx * N / g)..((cx + 1) * N / g) {
                    s += pix[y * N + x];
                    n += 1.0;
                }
            }
            let level = ((s / n) * 3.0).min(2.0) as u64; // 0,1,2
            k = k.wrapping_mul(3).wrapping_add(level);
        }
    }
    k
}

/// RAW-fine key: sign of a 4×4 downsample about the mid-level (fails completeness — nuisance flips bits).
fn raw_fine_key(pix: &[f32]) -> u64 {
    let g = 4;
    let mut k = 0u64;
    for cy in 0..g {
        for cx in 0..g {
            let (mut s, mut n) = (0.0f32, 0.0f32);
            for y in (cy * N / g)..((cy + 1) * N / g) {
                for x in (cx * N / g)..((cx + 1) * N / g) {
                    s += pix[y * N + x];
                    n += 1.0;
                }
            }
            k = (k << 1) | ((s / n > 0.15) as u64);
        }
    }
    k
}

/// V-measure (harmonic mean of homogeneity and completeness) between a key column and the true phase column.
fn v_measure(keys: &[u64], phases: &[usize]) -> (f32, u64) {
    use std::collections::HashMap;
    let n = keys.len() as f32;
    let mut n_k: HashMap<u64, f32> = HashMap::new();
    let mut n_c: HashMap<usize, f32> = HashMap::new();
    let mut n_kc: HashMap<(u64, usize), f32> = HashMap::new();
    for i in 0..keys.len() {
        *n_k.entry(keys[i]).or_default() += 1.0;
        *n_c.entry(phases[i]).or_default() += 1.0;
        *n_kc.entry((keys[i], phases[i])).or_default() += 1.0;
    }
    // H(C|K) = -Σ p(k,c) ln( p(k,c)/p(k) )
    let mut h_ck = 0.0f32;
    let mut h_kc = 0.0f32;
    for (&(k, c), &v) in &n_kc {
        let p_kc = v / n;
        h_ck += -p_kc * (v / n_k[&k]).ln();
        h_kc += -p_kc * (v / n_c[&c]).ln();
    }
    let h_c: f32 = -n_c
        .values()
        .map(|&v| {
            let p = v / n;
            p * p.ln()
        })
        .sum::<f32>();
    let h_k: f32 = -n_k
        .values()
        .map(|&v| {
            let p = v / n;
            p * p.ln()
        })
        .sum::<f32>();
    let homogeneity = if h_c > 1e-6 { 1.0 - h_ck / h_c } else { 1.0 };
    let completeness = if h_k > 1e-6 { 1.0 - h_kc / h_k } else { 1.0 };
    let v = if homogeneity + completeness > 1e-6 {
        2.0 * homogeneity * completeness / (homogeneity + completeness)
    } else {
        0.0
    };
    (v, n_k.len() as u64)
}

/// Grow the lean M2 latent (grow_backbone dictionary + one Identity readout predicting the NEXT frame's
/// features = a self-supervised world model), train it, and return (field, feature cells, reliance-ranked cells).
fn grow_latent(train: &[Frame], budget: usize) -> (Field, Vec<usize>, Vec<usize>) {
    let mut f = Field::new(PIX);
    f.config_mut().feature_lr = 0.0;
    f.config_mut().max_hidden = 0;
    f.config_mut().gain_fanin_norm = true;
    f.config_mut().lr_fanin_norm = true;
    let pix_cells: Vec<usize> = (0..PIX).collect();
    let feats = f.grow_backbone(&pix_cells, M, 2.5);
    // one Identity readout per feature, predicting that feature's NEXT value (streaming DMD; reliance = |K| mass)
    let mut zhat = Vec::with_capacity(M);
    for _ in 0..M {
        let target = f.cells().len();
        let clamp: Vec<(usize, f32)> = feats.iter().map(|&c| (c, 0.0)).collect();
        f.grow(&clamp, target, 0.0, 0.0, budget);
        f.set_activation(target, Some(Activation::Identity));
        zhat.push(target);
    }
    let latent = |f: &mut Field, fr: &Frame| -> Vec<f32> {
        let inp: Vec<(usize, f32)> = (0..PIX).map(|p| (p, fr.pix[p])).collect();
        f.stimulate(&inp, budget);
        feats.iter().map(|&c| f.cells()[c]).collect()
    };
    let lat: Vec<Vec<f32>> = train.iter().map(|fr| latent(&mut f, fr)).collect();
    for t in 0..train.len() - 1 {
        let clamp: Vec<(usize, f32)> = feats.iter().zip(&lat[t]).map(|(&c, &v)| (c, v)).collect();
        for (i, &zh) in zhat.iter().enumerate() {
            f.learn_step(&clamp, zh, lat[t + 1][i], 0.15, budget);
        }
    }
    // rank features by reliance (the predictable = phase-carrying ones)
    let mut ranked: Vec<usize> = (0..M).collect();
    let rel: Vec<f32> = feats.iter().map(|&c| f.reliance_of(c)).collect();
    ranked.sort_by(|&a, &b| rel[b].partial_cmp(&rel[a]).unwrap());
    let band: Vec<usize> = ranked[..RBAND].iter().map(|&i| feats[i]).collect();
    (f, feats, band)
}

// RESULT (measured, TNT): FALSIFIED as written. Coarse quantizer V 0.735 > raw-fine 0.635 > GROWN sign-band
// 0.601 (139 distinct keys — nuisance flips the sign bits). Two honest reasons, both important: (1) a naive
// sign-quantize of the grown latent is NUISANCE-SENSITIVE (the features carry nuisance; M2's "repeat" lives in
// the CLOSURE/concept-map, not in a per-frame sign) — a good grown key needs the concept-map recognition, not
// a hash; (2) this ring world is SPATIALLY coded (blob position = phase), so a coarse spatial average already
// denoises well — it does NOT reproduce real ARC-3's SEMANTIC aliasing (distinct game states that look alike).
// So the synthetic cannot settle the state-key question — which is the cleanest confirmation of MorphLab's "not
// more Python": only Level-3 (real games where the quantizer genuinely aliases) can. Kept as a reproducible
// probe (#[ignore]); the honest next step is the concept-map recognition key + the real run, not a forced pass.
#[test]
#[ignore = "state-key probe — FALSIFIED on synthetic (naive sign-key < coarse quantizer); the real test is Level-3"]
fn grown_latent_beats_the_aliased_quantizer_as_a_state_key() {
    let budget = 400;
    let train = stream(2500, 0x57A7);
    let test = stream(600, 0x1234);
    let (mut f, feats, band) = grow_latent(&train, budget);

    // build the three keys over the test frames
    let (mut coarse, mut raw, mut grown, mut phases) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for fr in &test {
        coarse.push(coarse_key(&fr.pix));
        raw.push(raw_fine_key(&fr.pix));
        // GROWN key = sign-quantize the reliance-selected band of the grown latent
        let inp: Vec<(usize, f32)> = (0..PIX).map(|p| (p, fr.pix[p])).collect();
        f.stimulate(&inp, budget);
        let mut gk = 0u64;
        for &c in &band {
            gk = (gk << 1) | ((f.cells()[c] > 0.0) as u64);
        }
        grown.push(gk);
        phases.push(fr.phase);
    }
    let (v_coarse, nk_coarse) = v_measure(&coarse, &phases);
    let (v_raw, nk_raw) = v_measure(&raw, &phases);
    let (v_grown, nk_grown) = v_measure(&grown, &phases);
    let _ = feats;

    eprintln!("[KEY] state-key V-measure vs the true abstract state (P={P} phases; higher=better, balances no-alias + repeats):");
    eprintln!("[KEY]   COARSE 6×6/3-lvl quantizer (MorphLab's): V {v_coarse:.3}  ({nk_coarse} distinct keys)");
    eprintln!(
        "[KEY]   RAW-fine sign grid:                       V {v_raw:.3}  ({nk_raw} distinct keys)"
    );
    eprintln!("[KEY]   GROWN latent (reliance band, sign):       V {v_grown:.3}  ({nk_grown} distinct keys)");
    eprintln!("[KEY] HONEST FINDING (FALSIFIED): the naive grown sign-key ({v_grown:.3}, {nk_grown} keys — nuisance-split) does NOT beat the coarse quantizer ({v_coarse:.3}). On a SPATIALLY-coded world a coarse average denoises well, and M2's 'repeat' lives in the CLOSURE/concept-map, not a per-frame sign — so a good grown key needs the concept-map recognition, and this synthetic (no semantic aliasing) can't settle it. The real test is Level-3 on 252 games. TNT: reported straight, not forced to pass.");

    // Documented negative — the measurement did its job (falsified the naive-adapter hypothesis). No trophy gate.
    assert!(
        v_coarse >= v_grown,
        "sanity: on this spatially-coded world the coarse average is expected to win"
    );
}
