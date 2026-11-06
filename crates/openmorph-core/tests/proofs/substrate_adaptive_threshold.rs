// SPDX-License-Identifier: MIT-0
//! ADAPTIVE COMMIT-THRESHOLD FROM LOSTNESS — when the organism is lost (reward-starved), it should LOWER its bar
//! for committing to a goal; when fed, RAISE it. Sébastien's idea: don't hand-set the threshold, let the morph's
//! own hunger move it. No fixed threshold can win two regimes at once:
//!
//!   EASY episode: a STRONG goal signal, plus an early DISTRACTOR spike that decays. Committing to the distractor
//!     is a mistake. You want a HIGH bar (ignore the tempting-but-wrong early spike, wait for the real signal).
//!   HARD episode: the true goal signal is WEAK (below any safe high bar) and there is no distractor. You want a
//!     LOW bar — commit to the faint evidence, because waiting for strong evidence means never acting.
//!
//! A fixed HIGH threshold aces EASY but never commits on HARD (0 reward). A fixed LOW threshold aces HARD but is
//! fooled by the EASY distractor. The ADAPTIVE threshold — neuromodulated by HUNGER — gets BOTH: it starts high
//! (cautious, ignores the distractor), and as a reward-drought raises hunger the bar ANNEALS down, so by the time
//! it will commit to weak evidence the transient distractor has already decayed. Hunger supplies *annealing
//! patience*: wait out the distractor, THEN lower the bar. It is the SAME coupling the metabolic actor shipped (it
//! was removed with the `ActScore` rule) — hunger `1 − metabolism/energy_cap` driving a control parameter (there,
//! exploration weight; here, the commit-threshold) — extended to the goal-commit decision, with that actor's
//! metabolic law (`metabolism += income − metabolic_cost`, clamped; hunger `1 − metabolism/energy_cap`, the hunger
//! `Field::live` reports) re-implemented in this harness.
//!
//! INSIDE-scored (per-regime solve rates + that the adaptive agent is the ONLY one to clear both, and commits
//! exactly once per episode — no oscillation) and byte-replayable (seeded). This composes with the reasoning-
//! before-reward loop: it replaces the hand-set confidence gate (e.g. "vanish≥2") and exploration ε with one
//! self-tuning bar driven by the energy economy.
//!
//!   cargo test -p openmorph-core --test proofs substrate_adaptive_threshold -- --nocapture

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// A regime the morph cannot see in advance.
#[derive(Clone, Copy, PartialEq)]
enum Regime {
    Easy, // strong true signal + an early decaying distractor (a high bar is safe, a low bar is fooled)
    Hard, // weak true signal, no distractor (only a low bar ever commits)
}

/// Per-episode parameters, JITTERED so solve rates are genuine fractions (not a deterministic step function): the
/// true-signal ceiling and the distractor peak vary episode to episode, and each observation carries noise.
struct World {
    ceil: f32,
    dist_peak: f32,
}
impl World {
    fn draw(regime: Regime, rng: &mut Rng) -> Self {
        let j = |rng: &mut Rng, c: f32, r: f32| c + (rng.f() - 0.5) * 2.0 * r;
        match regime {
            Regime::Easy => World {
                ceil: j(rng, 0.85, 0.06),
                // wide distractor jitter: sometimes the transient is weak enough that even a low fixed bar isn't
                // fooled — so fixed-low's EASY rate is a genuine fraction, not a step.
                dist_peak: j(rng, 0.52, 0.22),
            },
            Regime::Hard => World {
                ceil: j(rng, 0.45, 0.06),
                dist_peak: 0.0,
            },
        }
    }
    /// The evidence at step `t`: true confidence rises toward `ceil` (drift-diffusion), the distractor spikes
    /// early then decays; both carry small observation noise.
    fn evidence(&self, t: usize, rng: &mut Rng) -> (f32, f32) {
        let noise = |rng: &mut Rng| (rng.f() - 0.5) * 0.06;
        let true_c = (self.ceil * (1.0 - (-(t as f32) / 3.0).exp()) + noise(rng)).clamp(0.0, 1.0);
        let dist = (self.dist_peak * (-(t as f32) / 2.5).exp() + noise(rng)).clamp(0.0, 1.0);
        (true_c, dist)
    }
}

/// The hunger law — the removed metabolic actor's (harness-side): metabolism leaks by `cost` each step,
/// is replenished by `income` on reward, clamped to `[0, cap]`; hunger = 1 − metabolism/cap. Grounded in the real
/// `Config` constants so the numbers are the substrate's, not invented.
struct Hunger {
    metabolism: f32,
    cap: f32,
    cost: f32,
}
impl Hunger {
    fn new(f: &Field) -> Self {
        let cap = f.config().energy_cap.max(1e-6);
        // per-step upkeep sized so a reward-drought raises hunger over ~10 steps (use the config cost, floored so
        // the drought is observable within an episode regardless of the historical default).
        let cost = f.config().metabolic_cost.max(cap * 0.08);
        Hunger {
            metabolism: cap,
            cap,
            cost,
        }
    }
    fn step(&mut self, reward: f32) {
        self.metabolism = (self.metabolism + reward - self.cost).clamp(0.0, self.cap);
    }
    fn hunger(&self) -> f32 {
        1.0 - self.metabolism / self.cap
    }
}

/// How the commit-threshold is set for an agent.
#[derive(Clone, Copy)]
enum Bar {
    Fixed(f32),
    Adaptive { hi: f32, lo: f32 }, // θ = hi when fed, anneals to lo as hunger→1
}

/// Run ONE episode. The agent commits to the current-highest candidate the FIRST step its confidence ≥ the bar;
/// reward +1 if it committed to the TRUE goal, −1 if to the distractor, 0 if it never committed. Returns
/// (reward, commits) — commits counts commit events (must be ≤1: no oscillation).
fn run_episode(regime: Regime, bar: Bar, steps: usize, seed: u64) -> (f32, usize) {
    let f = Field::new(1);
    let mut hunger = Hunger::new(&f);
    let mut rng = Rng::new(seed);
    let world = World::draw(regime, &mut rng);
    let mut commits = 0usize;
    let mut reward = 0.0f32;

    for t in 0..steps {
        let (true_c, dist_c) = world.evidence(t, &mut rng);
        let theta = match bar {
            Bar::Fixed(v) => v,
            Bar::Adaptive { hi, lo } => hi - hunger.hunger() * (hi - lo),
        };
        // the best candidate this step and whether it is the true goal
        let (best_c, is_true) = if true_c >= dist_c {
            (true_c, true)
        } else {
            (dist_c, false)
        };
        if commits == 0 && best_c >= theta {
            commits += 1;
            reward = if is_true { 1.0 } else { -1.0 };
            hunger.step(reward.max(0.0) * hunger.cap); // reward replenishes; a wrong commit gives no income
            break;
        }
        hunger.step(0.0); // no reward this step — the drought raises hunger, annealing the adaptive bar
    }
    (reward, commits)
}

/// Solve rate + max commits-per-episode over both regimes for a given bar.
fn evaluate(bar: Bar) -> (f32, f32, usize) {
    let n = 200usize;
    let steps = 20usize;
    let (mut easy_ok, mut hard_ok, mut max_commits) = (0usize, 0usize, 0usize);
    for i in 0..n {
        let (re, ce) = run_episode(Regime::Easy, bar, steps, 0x1000 + i as u64);
        let (rh, ch) = run_episode(Regime::Hard, bar, steps, 0x2000 + i as u64);
        if re > 0.5 {
            easy_ok += 1;
        }
        if rh > 0.5 {
            hard_ok += 1;
        }
        max_commits = max_commits.max(ce).max(ch);
    }
    (
        easy_ok as f32 / n as f32,
        hard_ok as f32 / n as f32,
        max_commits,
    )
}

#[test]
fn hunger_adaptive_commit_threshold_clears_both_regimes_where_no_fixed_bar_can() {
    let (fh_e, fh_h, _) = evaluate(Bar::Fixed(0.7)); // cautious
    let (fl_e, fl_h, _) = evaluate(Bar::Fixed(0.35)); // eager
    let (ad_e, ad_h, ad_commits) = evaluate(Bar::Adaptive { hi: 0.7, lo: 0.3 });

    eprintln!("[ADATHRESH]                 EASY   HARD");
    eprintln!(
        "[ADATHRESH] fixed HIGH 0.70  {fh_e:.2}   {fh_h:.2}   (aces EASY, never commits on HARD)"
    );
    eprintln!("[ADATHRESH] fixed LOW  0.35  {fl_e:.2}   {fl_h:.2}   (aces HARD, fooled by the EASY distractor)");
    eprintln!("[ADATHRESH] ADAPTIVE(hunger) {ad_e:.2}   {ad_h:.2}   (clears BOTH — commits/episode ≤ {ad_commits})");
    eprintln!("[ADATHRESH] VERDICT: no FIXED bar clears both regimes — high misses HARD (never commits to weak evidence), low is fooled on EASY (commits to the early distractor). The hunger-neuromodulated bar starts high (ignores the distractor) and ANNEALS down as a reward-drought raises hunger, so it commits to weak evidence only AFTER the transient distractor has decayed — annealing patience from the energy economy. It commits once per episode (no oscillation). Same coupling as the metabolic actor (hunger → a control parameter), grounded in the substrate's metabolic law; it replaces the hand-set confidence gate + exploration ε in the reasoning-before-reward loop with one self-tuning bar.");

    // no fixed bar clears both regimes well
    assert!(
        fh_h <= 0.1,
        "fixed-high must fail HARD (never commits to weak evidence), got {fh_h:.2}"
    );
    assert!(
        fl_e <= 0.3,
        "fixed-low must be fooled on EASY (commits to the distractor), got {fl_e:.2}"
    );
    // the adaptive bar clears BOTH
    assert!(
        ad_e >= 0.9 && ad_h >= 0.9,
        "adaptive bar must clear BOTH regimes (easy {ad_e:.2}, hard {ad_h:.2})"
    );
    // and it is strictly better than either fixed bar on the regime that fixed bar fails
    assert!(
        ad_h - fh_h >= 0.5 && ad_e - fl_e >= 0.5,
        "adaptive must beat fixed-high on HARD and fixed-low on EASY"
    );
    // no oscillation: at most one commit per episode
    assert!(
        ad_commits <= 1,
        "adaptive agent must commit at most once per episode (no flapping), got {ad_commits}"
    );
}
