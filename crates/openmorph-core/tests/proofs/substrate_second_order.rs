// SPDX-License-Identifier: MIT-0
//! Sébastien's keystone: NATURE USES 2ND-ORDER DIFFERENTIAL EQUATIONS FOR STABILITY, embedded in the substrate
//! itself. The core delta rule is now optionally 2nd-order — a heritable per-unit momentum μ makes `learn_step`
//! a damped oscillator `v ← μ·v + Δw; w ← w + v` (heavy-ball = the discretized 2nd-order ODE `ẍ + (1−μ)ẋ = −∇`),
//! `μ=0` byte-identical to the historical 1st-order rule. This is the first seam of "the metabolism of surprise"
//! (docs/research/the-metabolism-of-surprise.md, §7.2): the tuning loop's stability, grown not authored.
//!
//! THE FALSIFIER (truth-not-trophy — this test would BREAK if the hypothesis is wrong; the numbers are measured):
//! on an ILL-CONDITIONED target (correlated/unequal-variance inputs — a ravine), 1st-order gradient descent
//! crawls in the low-curvature direction; 2nd-order momentum accelerates it. But the damping is a genuine
//! INTERIOR OPTIMUM — too little is slow (1st-order), too much is an unstable pole that DIVERGES. So the claims:
//!   (A) 2ND-ORDER HELPS: a well-damped μ reaches a markedly lower held-out error than μ=0 (1st-order) IN THE
//!       SAME BUDGET on the ill-conditioned world (a heavy-ball speedup — both converge to ~0 given enough steps).
//!   (B) DAMPING IS AN INTERIOR OPTIMUM (why it must be TUNED, not fixed): the well-damped μ beats BOTH a
//!       1st-order neighbour AND an OVER-damped μ (≈0.98) whose pole magnitude >1 makes the delta rule DIVERGE.
//!       So μ is not "more is better"; there is a right value — a U-curve.
//!   (C) GROWN, NOT AUTHORED (bounded honestly): a competition on error DRIFTS the pool's mean μ from a broad
//!       seed into the stable band where a static control does not (μ is DISCOVERABLE by selection), AND the
//!       in-core `Field::reproduce` actually INHERITS μ under the same self-adaptive σ as scale/lr (asserted
//!       directly). CAVEAT: unlike scale/lr, μ is NOT (yet) wired into the in-core fitness path (credit/energy),
//!       so the selection shown here is a HARNESS surrogate on an external oracle, not closed through the economy.
//!
//! HONEST SCOPE: this validates the 2nd-order STABILITY of the LEARNING loop on a synthetic ill-conditioned
//! regression — it is the tuning-loop half of the keystone, not yet the recurrent-belief/economy half. Momentum
//! is OFF by default (byte-identical); the whole existing suite is unchanged. Deterministic (seeded), CI-safe.

use openmorph_core::{Activation, Field};

use crate::common::Rng;

/// Learn a LINEAR target y = x0 + x1 over ILL-CONDITIONED inputs (x0 full-variance, x1 tiny-variance ⇒ a ravine,
/// condition number ≈ 1/σ²) with a fixed per-unit momentum μ. Returns held-out MAE after `steps`.
fn learn_ill_conditioned(mu: f32, steps: usize, seed: u64) -> f32 {
    let mut f = Field::new(3);
    f.config_mut().max_hidden = 0;
    f.config_mut().lr_fanin_norm = false; // control the effective lr exactly (no normalization)
    f.config_mut().gain_fanin_norm = false;
    f.config_mut().init_momentum = mu; // the unit is born with this damping
    let mut rng = Rng::new(seed);
    let sigma = 0.18f32; // x1 variance ≪ x0 ⇒ ill-conditioned (κ ≈ 30)
                         // birth the linear (Identity) readout of cells {0,1} → cell 2
    f.grow(&[(0, 0.0), (1, 0.0)], 2, 0.0, 0.0, 20);
    f.set_activation(2, Some(Activation::Identity));
    for _ in 0..steps {
        let x0 = rng.sym();
        let x1 = sigma * rng.sym();
        f.learn_step(&[(0, x0), (1, x1)], 2, x0 + x1, 0.5, 20);
    }
    // held-out MAE
    let (mut e, mut n) = (0.0f32, 0.0f32);
    for _ in 0..300 {
        let x0 = rng.sym();
        let x1 = sigma * rng.sym();
        f.stimulate(&[(0, x0), (1, x1)], 20);
        e += (f.cells()[2] - (x0 + x1)).abs();
        n += 1.0;
    }
    e / n
}

/// SELECTABLE μ: a population of `POP` learners with heritable momentum (seeded broadly around `seed_mu`). When
/// `select` is true the fittest (lowest error) breeds the worst each round by the SAME multiplicative-Schwefel rule
/// the in-core `reproduce` uses for μ; when false the pool is STATIC (a no-selection control). Returns
/// `(best_mae, initial_mean_μ, final_mean_μ)`. HONEST SCOPE: this is a harness-level SELECTION surrogate on an
/// external held-out-error oracle — it does NOT call `Field::reproduce` and μ is NOT closed through the in-core
/// energy account (unlike `scale`/`lr`). Its job is only to show the stabilizing μ is DISCOVERABLE by competition:
/// with selection ON, the population's mean μ must DRIFT from its broad seed into the stable band (the effect a
/// static pool cannot show). That the real `Field::reproduce` inherits μ is asserted separately and directly by
/// `reproduce_inherits_grown_momentum`.
fn grown_momentum(seed_mu: f32, rounds: usize, seed: u64, select: bool) -> (f32, f32, f32) {
    const POP: usize = 8;
    let mut rng = Rng::new(seed);
    // Seed the pool BROADLY across the whole μ range (not clustered at the optimum), so any drift toward the
    // stable band is selection's doing, not a lucky narrow start.
    let mut mus: Vec<f32> = (0..POP)
        .map(|_| (seed_mu + 0.45 * rng.sym()).clamp(0.0, 0.97))
        .collect();
    let initial_mean = mus.iter().sum::<f32>() / POP as f32;
    let mut best = f32::MAX;
    for _ in 0..rounds {
        let errs: Vec<f32> = mus
            .iter()
            .enumerate()
            .map(|(i, &m)| learn_ill_conditioned(m, 250, seed ^ (i as u64 + 1)))
            .collect();
        let (bi, &be) = errs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (wi, _) = errs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        best = best.min(be);
        if select {
            // heredity: the worst is reborn as the fittest's offspring (μ inherited, self-adaptive jitter)
            mus[wi] = (mus[bi] * (0.25 * rng.sym()).exp()).clamp(0.0, 0.97);
        }
    }
    (best, initial_mean, mus.iter().sum::<f32>() / POP as f32)
}

/// TWO opposite episodes back-to-back with a high-inertia unit (μ). Episode A is a SHORT, non-converging burst
/// toward y = +x (a "game" — you never converge per episode, so it ends with a LARGE carried velocity, still
/// climbing); episode B then wants −x. Returns B's FINAL convergence error `|predict(1) − (−1)|` after `b_steps`
/// (lower = reached the new target). `relax_between` runs the unforced relaxing period between the episodes.
fn two_episodes(mu: f32, relax_between: bool, a_steps: usize, b_steps: usize, seed: u64) -> f32 {
    let mut f = Field::new(2);
    f.config_mut().max_hidden = 0;
    f.config_mut().lr_fanin_norm = false;
    f.config_mut().gain_fanin_norm = false;
    f.config_mut().init_momentum = mu;
    let mut rng = Rng::new(seed);
    f.grow(&[(0, 0.0)], 1, 0.0, 0.0, 20);
    f.set_activation(1, Some(Activation::Identity));
    for _ in 0..a_steps {
        let x = rng.sym();
        f.learn_step(&[(0, x)], 1, x, 0.3, 20); // toward +x — SHORT, so velocity is still large at the switch
    }
    // The relaxing period between games — repeated unforced steps bleed the carried momentum toward rest.
    if relax_between {
        for _ in 0..80 {
            f.relax();
        }
    }
    for _ in 0..b_steps {
        let x = rng.sym();
        f.learn_step(&[(0, x)], 1, -x, 0.3, 20); // opposite target — stale +momentum overshoots before turning
    }
    f.stimulate(&[(0, 1.0)], 20);
    (f.cells()[1] - (-1.0)).abs() // final convergence error toward the new target
}

#[test]
fn relax_clears_stale_momentum_between_episodes() {
    // The GAME regime (Sébastien's use case): SHORT episodes that never converge, so the carried velocity is
    // large at the switch — where a "relaxing period" actually earns its keep.
    let seeds = [0x51u64, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58];
    let mean = |relax: bool| {
        seeds
            .iter()
            .map(|&s| two_episodes(0.9, relax, 4, 12, s))
            .sum::<f32>()
            / seeds.len() as f32
    };
    let no_relax = mean(false);
    let with_relax = mean(true);
    eprintln!("[RELAX] episode-B FINAL convergence error after switching to the OPPOSITE target");
    eprintln!(
        "[RELAX] (μ=0.9, SHORT non-converging episodes = the game regime, 8 seeds; lower=better):"
    );
    eprintln!("[RELAX]   momentum carried over (no rest):   {no_relax:.4}");
    eprintln!(
        "[RELAX]   relaxing period between (relax()): {with_relax:.4}  ({:.2}× lower)",
        no_relax / with_relax.max(1e-6)
    );
    eprintln!("[RELAX] VERDICT: `relax()` — the unforced 2nd-order dynamics, momentum bleeding to rest at the grown damping — clears the carried velocity between SHORT episodes so a stale, wrong-direction momentum does not overshoot the next task. Between games/exercises the organism RESTS, then starts the next clean. No new mechanism: relaxation is the delta rule at zero surprise. HONEST SCOPE: the benefit is regime-specific — a CONVERGED episode self-relaxes its own momentum (gradient→0 ⇒ v→0), so relax() matters for SHORT/truncated episodes (games), which is exactly where Sébastien flagged it.");
    assert!(with_relax < no_relax, "relax() should reduce cross-episode overshoot in the game regime ({with_relax:.4} vs {no_relax:.4})");
}

#[test]
fn grown_second_order_stabilizes_the_delta_rule() {
    let steps = 500;
    let seeds = [0xF1u64, 0xF2, 0xF3, 0xF4, 0xF5];
    let mean =
        |f: &dyn Fn(u64) -> f32| seeds.iter().map(|&s| f(s)).sum::<f32>() / seeds.len() as f32;

    let mae_1st = mean(&|s| learn_ill_conditioned(0.0, steps, s)); // 1st-order (μ=0)
    let mae_well = mean(&|s| learn_ill_conditioned(0.8, steps, s)); // WELL-damped 2nd-order (near, not at, optimum)
    let mae_low = mean(&|s| learn_ill_conditioned(0.5, steps, s)); // a LOWER μ — still 2nd-order, less inertia
    let mae_div = mean(&|s| learn_ill_conditioned(0.98, steps, s)); // OVER-damped pole |>1| ⇒ DIVERGES
                                                                    // (C) SELECTION vs a no-selection control: does breeding DRIFT the pool's mean μ into the stable band?
    let (grown_mae, mu_init, mu_final) = grown_momentum(0.5, 10, 0xA11CE, true);
    let (_, ctl_init, ctl_final) = grown_momentum(0.5, 10, 0xA11CE, false); // static pool (no breeding)

    eprintln!("[2ND] ILL-CONDITIONED held-out MAE (5 seeds, {steps} steps; lower=better):");
    eprintln!("[2ND]   1st-order (μ=0):                    {mae_1st:.4}");
    eprintln!(
        "[2ND]   2nd-order well-damped (μ=0.8):       {mae_well:.4}  ({:.1}× lower than 1st-order)",
        mae_1st / mae_well.max(1e-6)
    );
    eprintln!(
        "[2ND]   2nd-order lower μ (μ=0.5):          {mae_low:.4}  (interior of the U-curve)"
    );
    eprintln!(
        "[2ND]   2nd-order OVER-damped (μ=0.98):      {mae_div:.4}  (pole |>1| ⇒ DIVERGES — damping must be tuned)"
    );
    eprintln!("[2ND]   SELECTION: pool mean μ {mu_init:.2} → {mu_final:.2}  (no-selection control stays {ctl_final:.2})");
    eprintln!("[2ND] VERDICT: embedding a 2nd-order (heavy-ball) delta rule in the core — μ a HERITABLE per-unit trait, off-by-default and byte-identical — stabilizes the tuning loop as Sébastien's keystone predicts: on an ill-conditioned ravine a well-damped μ reaches {:.1}× lower held-out error than the 1st-order rule IN THE SAME BUDGET (a heavy-ball speedup on κ≈30; both converge to ~0 given enough steps), while an OVER-damped μ (pole magnitude >1) DIVERGES — so the damping is a genuine interior optimum that must be TUNED, not fixed. A competition on error DRIFTS the population's mean μ from a broad seed into the stable band (a static pool does not drift), showing the stabilizing μ is DISCOVERABLE by selection. HONEST SCOPE (truth-not-trophy): (i) the LEARNING-loop half of the keystone on a synthetic ravine — the recurrent-belief/economy half is next; (ii) μ is INHERITED by the real `Field::reproduce` (asserted directly by `reproduce_inherits_grown_momentum`) but is NOT yet closed through the in-core energy account like `scale`/`lr` — the selection here is a harness surrogate on an external oracle, not economic selection.", mae_1st / mae_well.max(1e-6));

    // ── GATES (the falsifier) ─────────────────────────────────────────────────────────────────────────────
    // (A) 2nd-order is a real fixed-budget speedup over 1st-order on the ravine.
    assert!(
        mae_well < 0.7 * mae_1st,
        "(A) 2nd-order did not beat 1st-order on the ravine ({mae_well:.4} vs {mae_1st:.4})"
    );
    // (B) DAMPING IS AN INTERIOR OPTIMUM (not 'more is better'): a well-damped μ beats BOTH a 1st-order (μ=0) and
    // an over-damped (μ=0.98, which diverges) neighbour — a genuine U-curve, so μ must be tuned. (The old gate
    // `mae_under > mae_crit` was near-vacuous — any divergence trivially exceeds a small number.)
    assert!(
        mae_well < mae_1st && mae_well < mae_div,
        "(B) μ=0.8 is not an interior optimum ({mae_well:.4} vs 1st {mae_1st:.4} / over-damped {mae_div:.4})"
    );
    assert!(
        mae_div > 1.0,
        "(B) μ=0.98 should DIVERGE (pole |>1|), got {mae_div:.4}"
    );
    // (C) SELECTION EFFECT (not best-of-initial-pool): breeding must DRIFT the mean μ up into the stable band,
    // where the static control does not. This fails if μ is not actually selected — the earlier gate
    // `grown_mae < mae_1st` passed even with a pre-seeded good member and zero selection.
    assert!(
        mu_final > mu_init + 0.05 && (0.55..=0.9).contains(&mu_final),
        "(C) selection did not drift mean μ into the stable band (init {mu_init:.2} → final {mu_final:.2})"
    );
    assert!(
        (mu_final - ctl_final).abs() > 0.05 && (ctl_init - ctl_final).abs() < 0.02,
        "(C) selection did not move μ vs the static no-selection control (sel {mu_final:.2} vs ctl {ctl_final:.2})"
    );
    assert!(
        grown_mae < mae_1st,
        "(C-sanity) selected μ did not beat the 1st-order baseline ({grown_mae:.4} vs {mae_1st:.4})"
    );
}

/// The DIRECT core-path assertion behind claim (C): `Field::reproduce` actually inherits the grown damping μ from
/// parent to child, through the SAME self-adaptive (Schwefel) rule as `scale`/`lr` — no surrogate. A parent with
/// high μ breeds a 1st-order child; the child must come out carrying (an inherited, jittered, clamped) μ.
#[test]
fn reproduce_inherits_grown_momentum() {
    let mut f = Field::new(4);
    f.config_mut().init_momentum = 0.8; // the parent feature is born high-inertia
    f.grow(&[(0, 0.0)], 2, 0.0, 0.0, 20); // parent writes cell 2
    f.config_mut().init_momentum = 0.0; // the child feature is born 1st-order
    f.grow(&[(1, 0.0)], 3, 0.0, 0.0, 20); // child writes cell 3
    assert!(
        f.momentum_of(2) > 0.5,
        "parent should carry high μ (got {})",
        f.momentum_of(2)
    );
    assert_eq!(f.momentum_of(3), 0.0, "child born 1st-order");
    f.reproduce(3, 2); // breed: child becomes the parent's offspring
    let child_mu = f.momentum_of(3);
    eprintln!("[REPRO] Field::reproduce inherited μ: parent 0.80 → child {child_mu:.3} (Schwefel-jittered, clamped ≤0.99)");
    assert!(
        child_mu > 0.4 && child_mu <= 0.99,
        "reproduce did not inherit μ from the parent (got {child_mu} — expected an inherited high μ)"
    );
}

/// The BYTE-IDENTICAL guard: a purely 1st-order lineage (μ=0 everywhere, the default) must stay μ=0 through
/// `reproduce` — and consume NO extra rng draw for the (absent) momentum mutation, so the whole existing evolve/
/// heritable suite is unperturbed. If this regressed, momentum would silently leak into every organism.
#[test]
fn reproduce_off_stays_first_order() {
    let mut f = Field::new(4); // init_momentum defaults to 0.0
    f.grow(&[(0, 0.0)], 2, 0.0, 0.0, 20);
    f.grow(&[(1, 0.0)], 3, 0.0, 0.0, 20);
    f.reproduce(3, 2);
    assert_eq!(
        f.momentum_of(3),
        0.0,
        "a 1st-order lineage must stay μ=0 through reproduce (byte-identical guarantee)"
    );
}

/// REGRESSION for the correctness bug the adversarial verify caught: `cull`/`detach` compact `self.units` and
/// every parallel trait vector — the μ (`momentum`) and 2nd-order state (`velocities`) MUST compact with them, or
/// a survivor whose index shifts down would silently inherit a REMOVED unit's damping. Grow three features with
/// DISTINCT μ, remove the middle one (shifting the third's index), and each survivor must still report ITS OWN μ.
/// (This is exactly the living-pool path claim (C) leans on, and no earlier proof exercised cull with momentum on.)
#[test]
fn cull_keeps_momentum_aligned_with_survivors() {
    let mut f = Field::new(6);
    f.config_mut().max_hidden = 0; // one writer per grow, so the indices are unambiguous
    for (cell, mu) in [(3usize, 0.3f32), (4, 0.6), (5, 0.9)] {
        f.config_mut().init_momentum = mu;
        f.grow(&[(cell - 3, 0.0)], cell, 1.0, 0.0, 20); // cell 3←0, 4←1, 5←2; each born with its own μ
    }
    assert!((f.momentum_of(3) - 0.3).abs() < 1e-6);
    assert!((f.momentum_of(4) - 0.6).abs() < 1e-6);
    assert!((f.momentum_of(5) - 0.9).abs() < 1e-6);
    // Remove the MIDDLE feature — cell 5's writer index shifts down by one. With the bug, momentum_of(5) would
    // read the stale (un-compacted) vector and return the removed unit's μ (0.6) instead of its own (0.9).
    f.detach(&[4]);
    assert!(
        (f.momentum_of(3) - 0.3).abs() < 1e-6,
        "first survivor's μ drifted after detach (got {})",
        f.momentum_of(3)
    );
    assert!(
        (f.momentum_of(5) - 0.9).abs() < 1e-6,
        "shifted survivor's μ misaligned after detach (got {} — the cull/detach compaction bug)",
        f.momentum_of(5)
    );
}
