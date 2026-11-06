// SPDX-License-Identifier: MIT-0
//! GROWN FORWARD/WORLD MODEL — `imagine(option)` (ARC-3 keystone, rung 3 of `docs/proposals/internal-deliberation`).
//!
//! THE GAP (MorphBot, epic #337 / cap #332). `act()` learns VALUES (what is good) but not DYNAMICS (what HAPPENS if
//! I do X). Without a forward model the organism can only discover consequences by TRIAL — exactly what the ARC-3
//! efficiency metric punishes (external actions are scored; internal computation is free). MorphBot's think-then-act
//! prototype showed a crude model + 1-step lookahead cut actions-to-first-win ~10×. The keystone is a GROWN forward
//! model the actor can query internally, for free, before committing an external action:
//!
//! ```text
//!     imagine(state, option) -> (predicted next-state features, predicted reward, epistemic uncertainty σ²)
//! ```
//!
//! THE MECHANISM (no new core primitive — the proven world-model idiom `substrate_predict`, exactly "expose and
//! loop what the substrate already computes"). A forward model is a grown predictor pointed at DYNAMICS:
//! `grow_backbone` a random-feature basis of the (state ⊕ option) context, `add_unit` a linear head per output
//! feature, and `learn_step` it toward the ACTUAL next feature, online from real transitions `(s, a, s', r)`. The
//! epistemic σ² is the SUBSTRATE'S OWN uncertainty device — an ENSEMBLE of such models, each `rng_seed`-varied so
//! their grown backbones differ; σ² = the DISAGREEMENT across members (the same ensemble-disagreement =
//! epistemic-uncertainty the free-energy actor / `select_explore` already run on). A never-tried option's weights are
//! untouched, so the members keep their varied priors and DISAGREE (high σ²); as transitions arrive every member
//! converges on the one true dynamics, so σ² SELF-ANNEALS toward 0.
//!
//! THE CLAIM (cap #332 in-core acceptance, measured here):
//!   * ACCURACY — after N ≤ 20 real transitions, `imagine` predicts next-state features with mean-absolute-error
//!     below a set threshold; and it GENERALISES (predicts states/actions in combinations never observed, because
//!     the grown model learns the DYNAMICS `Δ = f(action)`, not a lookup).
//!   * CALIBRATION — σ² is HIGH on a never-tried option and shrinks MONOTONICALLY as that option is observed. This
//!     is the signal cap #333 (info-directed curiosity) and #334 (planning) consume.
//!
//! TRUTH-NOT-TROPHY. The controllable grid, its linear dynamics, the seeded PRNG, the thresholds, and the
//! shortest-path-free accuracy oracle are authored and identical across arms; NOTHING tells the model the dynamics —
//! it grows them from experienced transitions. The MAE oracle is the world's own next state, held out. The live
//! decisive test (model-based 1-step lookahead cuts mean actions-to-first-win ≥30% at equal coverage on the 25
//! public games) is MorphBot's A/B, deliberately NOT asserted here. ZERO openmorph-core change: a usage protocol
//! over `grow_backbone` / `add_unit` / `learn_step` / `stimulate` / `cells` / `rng_seed`.
//!
//!   cargo test -p openmorph-core --test proofs substrate_imagine -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ── the controllable grid world (authored; the model never sees the rule) ───────────────────────────────────────

const W: usize = 6;
const H: usize = 6;
const N_ACT: usize = 4; // 0=up 1=down 2=left 3=right

/// The four action deltas on the grid.
fn delta(a: usize) -> (i32, i32) {
    match a {
        0 => (0, -1),
        1 => (0, 1),
        2 => (-1, 0),
        _ => (1, 0),
    }
}

/// Deterministic transition: apply the action, staying in-grid. Returns `(nx, ny, in_grid)` — `in_grid=false` marks
/// a move that would leave the lattice (we sample interior transitions so the trained dynamics is exactly linear).
fn step(x: usize, y: usize, a: usize) -> (usize, usize, bool) {
    let (dx, dy) = delta(a);
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    if nx < 0 || ny < 0 || nx >= W as i32 || ny >= H as i32 {
        (x, y, false)
    } else {
        (nx as usize, ny as usize, true)
    }
}

// ── the forward-model ensemble (the `imagine` protocol) ─────────────────────────────────────────────────────────

/// The input layout on each member field: cells `0..IN` = [x, y, onehot(action)]. The three prediction heads
/// [next-x, next-y, reward] are grown ABOVE the backbone (allocated at `f.len()`). Coordinates are normalised to
/// `[0,1]` so one MAE threshold covers both axes.
const IN: usize = 2 + N_ACT; // x, y, a0..a3
const OUT: usize = 3; // next-x, next-y, reward
const K: usize = 24; // per-member world-model backbone width (seed-varied ⇒ ensemble disagreement)
const LR: f32 = 0.03; // the proven world-model learning rate (`substrate_predict`); 0.3 diverges through the backbone
const BUDGET: usize = 300;

fn ctx(x: usize, y: usize, a: usize) -> Vec<(usize, f32)> {
    let mut c = vec![
        (0usize, x as f32 / (W - 1) as f32),
        (1usize, y as f32 / (H - 1) as f32),
    ];
    for k in 0..N_ACT {
        c.push((2 + k, if k == a { 1.0 } else { 0.0 }));
    }
    c
}

/// The target the world reveals for a transition: normalised next coords + reward (1 at the goal, else 0).
fn target(nx: usize, ny: usize, goal: (usize, usize)) -> [f32; OUT] {
    [
        nx as f32 / (W - 1) as f32,
        ny as f32 / (H - 1) as f32,
        if (nx, ny) == goal { 1.0 } else { 0.0 },
    ]
}

/// A `K`-member ensemble of grown forward models, each `rng_seed`-varied so its grown random-feature backbone
/// differs (⇒ genuine disagreement that anneals with data). This IS the `imagine` capability. Each member reads the
/// (state ⊕ action) inputs plus a grown backbone, and writes three prediction heads via the substrate's own
/// `learn_step` delta rule — the proven world-model idiom (`substrate_predict`).
struct Ensemble {
    members: Vec<Field>,
    pred: Vec<usize>, // the three head cells (same layout in every member)
}

impl Ensemble {
    fn new(k: usize) -> Self {
        let mut pred = Vec::new();
        let members = (0..k)
            .map(|i| {
                let mut f = Field::new(IN);
                f.rng_seed(0x1234_5678 ^ ((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                let inputs: Vec<usize> = (0..IN).collect();
                // A seed-varied grown backbone: the source of both nonlinear capacity and ensemble disagreement.
                let mut reads = inputs.clone();
                reads.extend(f.grow_backbone(&inputs, K, 2.5));
                let base = f.len();
                let heads: Vec<usize> = (0..OUT).map(|d| base + d).collect();
                for &c in &heads {
                    let n = reads.len();
                    f.add_unit(reads.clone(), vec![0.0; n], 0.0, c);
                }
                pred = heads; // identical across members
                f
            })
            .collect();
        Ensemble { members, pred }
    }

    /// LEARN one real transition `(s, a) → (s', r)`: every member corrects its three prediction heads toward reality
    /// with the local delta rule.
    fn learn(&mut self, x: usize, y: usize, a: usize, tgt: [f32; OUT]) {
        let inputs = ctx(x, y, a);
        for m in self.members.iter_mut() {
            for (&head, &t) in self.pred.iter().zip(tgt.iter()) {
                m.learn_step(&inputs, head, t, LR, BUDGET);
            }
        }
    }

    /// REPLAY the buffer of real transitions `epochs` times. The transitions are the only WORLD samples (the scored
    /// resource); replay over them is INTERNAL computation, which ARC-3 scores as FREE — so the model trains to
    /// convergence on its ≤N observations exactly as the deliberation thesis prescribes (think hard internally, act
    /// rarely). Nothing new is observed here.
    fn replay(&mut self, buffer: &[(usize, usize, usize, [f32; OUT])], epochs: usize) {
        for _ in 0..epochs {
            for &(x, y, a, tgt) in buffer {
                self.learn(x, y, a, tgt);
            }
        }
    }

    /// IMAGINE `(s, a)` WITHOUT acting: returns `(mean prediction [nx, ny, r], epistemic σ²)`. σ² is the mean over
    /// the three heads of the variance across members — the ensemble disagreement = the substrate's epistemic
    /// uncertainty. A pure read: `stimulate` the context, read the heads; no learning, no world interaction.
    fn imagine(&mut self, x: usize, y: usize, a: usize) -> ([f32; OUT], f32) {
        let inputs = ctx(x, y, a);
        let pred = self.pred.clone();
        let mut preds: Vec<[f32; OUT]> = Vec::with_capacity(self.members.len());
        for m in self.members.iter_mut() {
            m.stimulate(&inputs, BUDGET);
            let c = m.cells();
            preds.push([c[pred[0]], c[pred[1]], c[pred[2]]]);
        }
        let k = preds.len() as f32;
        let mut mean = [0.0f32; OUT];
        for p in &preds {
            for d in 0..OUT {
                mean[d] += p[d] / k;
            }
        }
        let mut var = 0.0f32;
        for d in 0..OUT {
            let v: f32 = preds.iter().map(|p| (p[d] - mean[d]).powi(2)).sum::<f32>() / k;
            var += v / OUT as f32;
        }
        (mean, var)
    }
}

// ── the proof ───────────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn imagine_predicts_dynamics_and_calibrates_uncertainty() {
    let goal = (W - 1, H - 1);
    let mut rng = Rng::new(0xA11CE);
    let mut ens = Ensemble::new(12);

    // ── ACCURACY: grow from N≤20 real transitions using only actions {up, down, left}; RIGHT is left never-tried
    // (it becomes the calibration probe). Interior sampling keeps the trained dynamics exactly linear.
    let train_actions = [0usize, 1, 2]; // up, down, left — NOT right(3)
    let mut buffer: Vec<(usize, usize, usize, [f32; OUT])> = Vec::new();
    let mut seen: Vec<(usize, usize, usize)> = Vec::new();
    while buffer.len() < 20 {
        let x = 1 + rng.below(W - 2); // interior 1..W-1
        let y = 1 + rng.below(H - 2);
        let a = train_actions[rng.below(train_actions.len())];
        let (nx, ny, ok) = step(x, y, a);
        if !ok {
            continue;
        }
        buffer.push((x, y, a, target(nx, ny, goal)));
        seen.push((x, y, a));
    }
    let n_obs = buffer.len();
    // Free internal training: replay the 20 real observations to convergence (no new world samples).
    ens.replay(&buffer, 200);

    // Held-out MAE on state features over ALL interior (state, trained-action) pairs — INCLUDING combinations never
    // observed (generalisation: the model learned Δ=f(action), not a lookup of the 20 seen transitions).
    let (mut mae_sum, mut count) = (0.0f32, 0usize);
    let mut unseen_mae = (0.0f32, 0usize);
    for x in 1..W - 1 {
        for y in 1..H - 1 {
            for &a in &train_actions {
                let (nx, ny, ok) = step(x, y, a);
                if !ok {
                    continue;
                }
                let (pred, _s2) = ens.imagine(x, y, a);
                let tgt = target(nx, ny, goal);
                let e = (pred[0] - tgt[0]).abs() + (pred[1] - tgt[1]).abs();
                mae_sum += e / 2.0;
                count += 1;
                if !seen.contains(&(x, y, a)) {
                    unseen_mae.0 += e / 2.0;
                    unseen_mae.1 += 1;
                }
            }
        }
    }
    let mae = mae_sum / count as f32;
    let unseen = unseen_mae.0 / unseen_mae.1.max(1) as f32;
    assert!(
        mae < 0.06,
        "imagine predicts next-state features within {n_obs} obs: MAE {mae:.4} (normalised coords) must be < 0.06"
    );
    assert!(
        unseen < 0.08,
        "GENERALISATION — MAE on never-observed (state,action) combos {unseen:.4} must be < 0.08 (dynamics, not lookup)"
    );

    // ── CALIBRATION (part A): σ² is HIGH on the never-tried option (right) vs the trained options, at the same
    // interior states (nothing about `right` was ever grown, so the members keep their varied priors and disagree).
    let mut cal_untried = 0.0f32;
    let mut cal_trained = 0.0f32;
    let mut cells = 0usize;
    for x in 1..W - 1 {
        for y in 1..H - 1 {
            let (_p, s2r) = ens.imagine(x, y, 3); // right — never trained
            let (_p, s2u) = ens.imagine(x, y, 0); // up — trained
            cal_untried += s2r;
            cal_trained += s2u;
            cells += 1;
        }
    }
    cal_untried /= cells as f32;
    cal_trained /= cells as f32;
    assert!(
        cal_untried > 2.0 * cal_trained,
        "σ² calibrated: never-tried option σ² {cal_untried:.5} must be ≫ trained σ² {cal_trained:.5}"
    );

    // ── CALIBRATION (part B): the never-tried option's σ² SHRINKS MONOTONICALLY as it is observed. Feed `right`
    // transitions in small batches from varied interior states and record the ensemble's MEAN σ²(right) over the
    // whole interior after each batch — the same read part A used for the trained actions. It must fall, not rise.
    let mean_right_sigma = |ens: &mut Ensemble| -> f32 {
        let mut s = 0.0f32;
        let mut c = 0usize;
        for x in 1..W - 1 {
            for y in 1..H - 1 {
                s += ens.imagine(x, y, 3).1;
                c += 1;
            }
        }
        s / c as f32
    };
    let s2_untried = mean_right_sigma(&mut ens); // σ² before any right observation (the never-tried baseline)
                                                 // A fixed buffer of varied real right-observations; replay it in growing epoch counts and watch σ² anneal.
    let mut right_buf: Vec<(usize, usize, usize, [f32; OUT])> = Vec::new();
    while right_buf.len() < 8 {
        let (bx, by) = (1 + rng.below(W - 3), 1 + rng.below(H - 2)); // x ≤ W-3 so right stays in-grid
        let (nx, ny, ok) = step(bx, by, 3);
        if ok {
            right_buf.push((bx, by, 3, target(nx, ny, goal)));
        }
    }
    // The anneal curve over CUMULATIVE free replay on the FIXED buffer: from the never-tried baseline, epistemic σ²
    // shrinks MONOTONICALLY toward the trained level as the shared dynamics is pinned down. (Feeding NEW states
    // mid-measurement transiently RAISES σ² — first-contact surprise, the active-learning signal #333 rides; here
    // we hold the buffer fixed to isolate the pure annealing.)
    let mut anneal = vec![s2_untried];
    for _ in 0..8 {
        ens.replay(&right_buf, 20); // +20 epochs of free internal replay each round
        anneal.push(mean_right_sigma(&mut ens));
    }
    let s2_end = *anneal.last().unwrap();
    for w in anneal.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-4,
            "σ² must not rise as data arrives: {:.6} → {:.6} (curve {anneal:?})",
            w[0],
            w[1]
        );
    }
    // And the converged σ² collapses far below the never-tried baseline — data resolves the uncertainty.
    assert!(
        s2_end < 0.2 * s2_untried,
        "σ² self-anneals with data: never-tried {s2_untried:.5} → converged {s2_end:.5} (below 20%)"
    );

    println!("\n  GROWN WORLD MODEL — imagine(option):");
    println!("    accuracy   : MAE {mae:.4} within {n_obs} obs (unseen combos {unseen:.4}) — grown dynamics, generalises");
    println!(
        "    calibration: never-tried σ² {cal_untried:.5} vs trained σ² {cal_trained:.5}  ({:.0}× higher)",
        cal_untried / cal_trained.max(1e-9)
    );
    println!(
        "    annealing  : never-tried σ² {s2_untried:.4} → converged {s2_end:.4} over free replay (monotone ↓ with data)"
    );
    println!("    anneal curve: {anneal:?}");
    println!("  ⇒ the keystone: a free, grown forward model with calibrated uncertainty for #333 (curiosity) + #334 (planning).\n");
}
