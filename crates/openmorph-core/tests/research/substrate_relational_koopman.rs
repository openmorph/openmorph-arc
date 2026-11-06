// SPDX-License-Identifier: MIT-0
//! #159 (OM-1) M2 rung 3 — SAME-PIXEL DISENTANGLEMENT (operator-spectral). The case rung 2's segment_objects
//! CANNOT reach: two independent sub-mechanics ENTANGLED in the SAME pixels (a latent rotation, not a spatial
//! boundary). The disentanglement is a spectral clustering of the LEARNED Koopman operator's OWN coupling graph
//! — literally Sébastien's "find sub pattern using fractal spectrum eigen vector, find sub-similarity to group
//! sub part." ZERO new core.
//!
//! WORLD (deterministic, seeded): latent z∈R^4 = two independent 2-D ROTATORS (block-1=(z0,z1),
//! block-2=(z2,z3)). An action switches z' = blkdiag(R(α_a), R(β_a))·z. Observation x = M·z, M a FIXED seeded
//! random orthonormal 36×4 embedding → every pixel mixes BOTH blocks → NO spatial boundary.
//!
//! MECHANISM: grow_backbone dictionary over the mixed pixels + config.feature_lr>0 (features LEARN directions,
//! the #125 one-hop feedback-alignment delta) + per-action Identity readouts (streaming DMD) → operator bank
//! K^a. The affinity A_ij = Σ_a |∂ẑ_i^a/∂z_j| (the operator's coupling, measured behaviorally via a finite-
//! difference Jacobian — no private-weight access). Cluster A → the two blocks; per-block sub-Koopman.
//!
//! PRIMARY METER (risk-decomposed — the true rung-3 content, measured directly, not inferred from downstream):
//! the WITHIN-BLOCK DOMINANCE ratio ρ = mean within-block |A| / mean cross-block |A|. With feature_lr>0 the
//! features align toward the latent blocks → K^a becomes block-diagonal → ρ>1. With feature_lr=0 (fixed random
//! dictionary) the block structure lives in a latent rotation NOT the feature basis → ρ≈1 (REQUIRED ablation
//! failure). A COUPLED world (blocks interact) → ρ≈1 (REQUIRED control failure). Reported seed-by-seed; a graded
//! ρ that beats the ablation is a truthful rung — the rung does NOT die on one downstream number (truth-not-trophy).
//!
//! RESULT (measured, 8 seeds — see `docs/investigations/159-operator-spectral-disentanglement.md`): this
//! mechanism is a NEGATIVE. `feature_lr>0` COLLAPSES the features to ~constant (variance ~0.003, the self-
//! prediction degeneracy) and ρ stays ≈1.04; `feature_lr=0` gives ρ≈1.16 (no separation either). The block
//! structure lives in the operator's EIGENVECTORS (a latent rotation with magnitude-1 eigenvalues), invisible
//! in a random feature basis and unreachable by the substrate's power-iteration-only spectral tools. The
//! finding VALIDATES rung 2 (disentanglement comes from a PERCEPTUAL basis — `segment_objects` SPATIALLY — not
//! operator-coupling on a random dictionary). A follow-on relational-`aggregate` Koopman was ALSO tried and
//! withdrawn as a tautology (aggregate pools away individuation; see §6 of the investigation), so rung 3 is a
//! genuine substrate BOUNDARY (latent rotation needs an eigensolver; relational individuation needs set-pooling
//! — both deliberately absent). Kept as a reproducible probe (`#[ignore]`), not a shipped capability.

#![allow(clippy::needless_range_loop)] // several parallel-array index loops read clearer than iterator adaptors

use openmorph_core::{Activation, Field};

use crate::common::Rng;

const LAT: usize = 4; // latent dim (two 2-D blocks)
const P: usize = 36; // pixels
const NACT: usize = 4; // switched rotation actions
const W: usize = 24; // dictionary width

/// A fixed seeded random orthonormal P×LAT embedding M (Gram-Schmidt on seeded random columns).
fn embedding(rng: &mut Rng) -> Vec<[f32; LAT]> {
    // LAT random P-vectors, orthonormalized; store as P rows of LAT (M[p][k]).
    let mut cols: Vec<Vec<f32>> = (0..LAT)
        .map(|_| (0..P).map(|_| rng.sym()).collect::<Vec<f32>>())
        .collect();
    for k in 0..LAT {
        for j in 0..k {
            let dot: f32 = (0..P).map(|p| cols[k][p] * cols[j][p]).sum();
            for p in 0..P {
                cols[k][p] -= dot * cols[j][p];
            }
        }
        let norm: f32 = (0..P)
            .map(|p| cols[k][p] * cols[k][p])
            .sum::<f32>()
            .sqrt()
            .max(1e-9);
        for p in 0..P {
            cols[k][p] /= norm;
        }
    }
    (0..P)
        .map(|p| std::array::from_fn(|k| cols[k][p]))
        .collect()
}

/// Render a latent state to pixels x = M·z (then a mild tanh to stay in the substrate's regime).
fn render(m: &[[f32; LAT]], z: &[f32; LAT]) -> Vec<f32> {
    (0..P)
        .map(|p| {
            let v: f32 = (0..LAT).map(|k| m[p][k] * z[k]).sum();
            (0.7 * v).tanh()
        })
        .collect()
}

/// Apply action a's switched block-diagonal rotation to the latent.
fn advance(z: &[f32; LAT], a: usize, angles: &[(f32, f32)], coupled: bool) -> [f32; LAT] {
    let (alpha, beta) = angles[a];
    let (ca, sa) = (alpha.cos(), alpha.sin());
    let (cb, sb) = (beta.cos(), beta.sin());
    if coupled {
        // blocks INTERACT: block-2's rotation angle is modulated by block-1's coordinate — no clean factorization.
        let beta2 = beta + 0.8 * z[0];
        let (cb2, sb2) = (beta2.cos(), beta2.sin());
        [
            ca * z[0] - sa * z[1],
            sa * z[0] + ca * z[1],
            cb2 * z[2] - sb2 * z[3],
            sb2 * z[2] + cb2 * z[3],
        ]
    } else {
        [
            ca * z[0] - sa * z[1],
            sa * z[0] + ca * z[1],
            cb * z[2] - sb * z[3],
            sb * z[2] + cb * z[3],
        ]
    }
}

/// Grow the dictionary + per-action Identity readouts; return (field, feature cells, zhat[action][feat]).
fn build(feature_lr: f32) -> (Field, Vec<usize>, Vec<Vec<usize>>) {
    let mut f = Field::new(P);
    f.config_mut().feature_lr = feature_lr; // >0 = learn feature DIRECTIONS (#125)
    f.config_mut().max_hidden = 0;
    f.config_mut().gain_fanin_norm = true;
    f.config_mut().lr_fanin_norm = true;
    let feats = f.grow_backbone(&(0..P).collect::<Vec<_>>(), W, 1.5);
    let mut zhat: Vec<Vec<usize>> = (0..NACT).map(|_| Vec::with_capacity(W)).collect();
    for bank in zhat.iter_mut() {
        for _ in 0..W {
            let target = f.cells().len();
            let clamp: Vec<(usize, f32)> = feats.iter().map(|&c| (c, 0.0)).collect();
            f.grow(&clamp, target, 0.0, 0.0, 120);
            f.set_activation(target, Some(Activation::Identity));
            bank.push(target);
        }
    }
    (f, feats, zhat)
}

/// Read the dictionary features for a latent state.
fn feat_of(f: &mut Field, feats: &[usize], m: &[[f32; LAT]], z: &[f32; LAT]) -> Vec<f32> {
    let frame = render(m, z);
    let inp: Vec<(usize, f32)> = (0..P).map(|p| (p, frame[p])).collect();
    f.stimulate(&inp, 120);
    feats.iter().map(|&c| f.cells()[c]).collect()
}

/// Train the switched-Koopman operator over random rotator trajectories; returns the training states (for later
/// Jacobian probing at realistic states).
fn train(
    f: &mut Field,
    feats: &[usize],
    zhat: &[Vec<usize>],
    m: &[[f32; LAT]],
    angles: &[(f32, f32)],
    rng: &mut Rng,
    steps: usize,
) {
    let mut z: [f32; LAT] = std::array::from_fn(|_| rng.sym());
    let mut prev: Option<(Vec<f32>, usize)> = None;
    for _ in 0..steps {
        let a = rng.below(NACT);
        let ft = feat_of(f, feats, m, &z);
        if let Some((zt, at)) = prev.take() {
            // target for the PREVIOUS step: the current features
            let clamp: Vec<(usize, f32)> = feats.iter().zip(&zt).map(|(&c, &v)| (c, v)).collect();
            for (i, &zh) in zhat[at].iter().enumerate() {
                f.learn_step(&clamp, zh, ft[i], 0.1, 120);
            }
        }
        prev = Some((ft, a));
        z = advance(&z, a, angles, false); // (train world is always the factorable one)
    }
}

/// Behavioral Jacobian coupling A_ij = Σ_a |∂ẑ_i^a / ∂z_j| averaged over probe states (finite differences on
/// the CLAMPED feature values — no private-weight access; Identity readouts make this the exact operator entry).
fn affinity(
    f: &mut Field,
    feats: &[usize],
    zhat: &[Vec<usize>],
    m: &[[f32; LAT]],
    probes: &[[f32; LAT]],
) -> Vec<Vec<f32>> {
    let eps = 0.05f32;
    let mut a = vec![vec![0.0f32; W]; W];
    for z in probes {
        let base = feat_of(f, feats, m, z);
        for a_idx in 0..NACT {
            for j in 0..W {
                let mut pert = base.clone();
                pert[j] += eps;
                let clamp: Vec<(usize, f32)> =
                    feats.iter().zip(&pert).map(|(&c, &v)| (c, v)).collect();
                f.stimulate(&clamp, 120);
                let pred_p: Vec<f32> = zhat[a_idx].iter().map(|&zh| f.cells()[zh]).collect();
                let clamp0: Vec<(usize, f32)> =
                    feats.iter().zip(&base).map(|(&c, &v)| (c, v)).collect();
                f.stimulate(&clamp0, 120);
                let pred_0: Vec<f32> = zhat[a_idx].iter().map(|&zh| f.cells()[zh]).collect();
                for i in 0..W {
                    a[i][j] += ((pred_p[i] - pred_0[i]) / eps).abs();
                }
            }
        }
    }
    a
}

/// Classify each feature by which latent BLOCK it loads on (grader ground truth via latent perturbation).
/// Returns block index 0/1 per feature, plus the mean feature variance (collapse detector).
fn block_of(f: &mut Field, feats: &[usize], m: &[[f32; LAT]], rng: &mut Rng) -> (Vec<usize>, f32) {
    let eps = 0.1f32;
    let n_probe = 40;
    let mut load = vec![[0.0f32; LAT]; W];
    let mut var_acc = vec![(0.0f32, 0.0f32, 0.0f32); W]; // (sum, sumsq, n) per feature
    for _ in 0..n_probe {
        let z: [f32; LAT] = std::array::from_fn(|_| rng.sym());
        let base = feat_of(f, feats, m, &z);
        for (i, &v) in base.iter().enumerate() {
            var_acc[i].0 += v;
            var_acc[i].1 += v * v;
            var_acc[i].2 += 1.0;
        }
        for k in 0..LAT {
            let mut zk = z;
            zk[k] += eps;
            let pert = feat_of(f, feats, m, &zk);
            for i in 0..W {
                load[i][k] += ((pert[i] - base[i]) / eps).abs();
            }
        }
    }
    let blocks: Vec<usize> = (0..W)
        .map(|i| {
            let b0 = load[i][0] + load[i][1];
            let b1 = load[i][2] + load[i][3];
            if b0 >= b1 {
                0
            } else {
                1
            }
        })
        .collect();
    let mean_var = var_acc
        .iter()
        .map(|&(s, ss, n)| (ss / n - (s / n).powi(2)).max(0.0))
        .sum::<f32>()
        / W as f32;
    (blocks, mean_var)
}

/// The within-block / cross-block coupling dominance ratio ρ from the affinity + the block labels.
fn dominance(a: &[Vec<f32>], blocks: &[usize]) -> f32 {
    let (mut win, mut nwin, mut cross, mut ncross) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..W {
        for j in 0..W {
            if i == j {
                continue;
            }
            if blocks[i] == blocks[j] {
                win += a[i][j];
                nwin += 1.0;
            } else {
                cross += a[i][j];
                ncross += 1.0;
            }
        }
    }
    (win / nwin.max(1.0)) / (cross / ncross.max(1.0)).max(1e-9)
}

/// Run one seed: build, train, measure ρ (+ collapse diagnostic). `flr` = feature_lr.
fn run_seed(seed: u64, flr: f32) -> (f32, f32) {
    let mut rng = Rng::new(seed);
    let m = embedding(&mut rng);
    let angles: Vec<(f32, f32)> = (0..NACT)
        .map(|_| {
            // distinct per-action angles, SEEDED (not hand-picked) — small so tanh stays informative
            (0.3 + 0.9 * rng.unit(), 0.3 + 0.9 * rng.unit())
        })
        .collect();
    let (mut f, feats, zhat) = build(flr);
    train(&mut f, &feats, &zhat, &m, &angles, &mut rng, 4000);
    let (blocks, mean_var) = block_of(&mut f, &feats, &m, &mut rng);
    let probes: Vec<[f32; LAT]> = (0..24)
        .map(|_| std::array::from_fn(|_| 0.6 * rng.sym()))
        .collect();
    let a = affinity(&mut f, &feats, &zhat, &m, &probes);
    let rho = dominance(&a, &blocks);
    (rho, mean_var)
}

#[test]
#[ignore = "rung-3 research probe — run explicitly; measures the feature_lr disentanglement mechanism"]
fn relational_koopman_rho_meter() {
    eprintln!("[REL] rung-3 ρ-meter: within-block / cross-block operator coupling. feature_lr>0 should align features to latent blocks (ρ>1); feature_lr=0 (fixed random dict) is the required ablation (ρ≈1).");
    let seeds = [0xA3u64, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
    let (mut on, mut off) = (Vec::new(), Vec::new());
    for &s in &seeds {
        let (rho_on, var_on) = run_seed(s, 0.05);
        let (rho_off, _var_off) = run_seed(s, 0.0);
        eprintln!("[REL] seed {s:#x}: ρ(feature_lr on) {rho_on:.3} (feat-var {var_on:.4}) vs ρ(off) {rho_off:.3}");
        on.push(rho_on);
        off.push(rho_off);
    }
    let median = |mut v: Vec<f32>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let frac_on = on.iter().filter(|&&r| r > 1.1).count() as f32 / on.len() as f32;
    eprintln!(
        "[REL] MEDIAN ρ: on {:.3} vs off {:.3}; seed-fraction ρ>1.1 (on) = {:.2}",
        median(on.clone()),
        median(off.clone()),
        frac_on
    );
    eprintln!("[REL] (research probe — asserts deferred until the mechanism's operating boundary is measured; truth-not-trophy)");
}
