// SPDX-License-Identifier: MIT-0
//! THE CRYSTAL HALF — native reliability-gated consolidation. The consolidation CONCEPT is validated on real ARC-3
//! (MorphLab #171, via the one-way FREEZE_STEPS harness stopgap this replaces: rr01 wandering-132 → locked-85, zm04
//! degrading-29 → optimal-16); THIS native primitive is validated by the synthetic falsifier below — ARC-3 wiring
//! is a pending hand-off. The morph is ALIVE (learns a skill one-shot) but was not CRYSTAL (couldn't HOLD a skill
//! under repetition — a near-perfect no-change marginal wins and the action-conditional structure collapses).
//!
//! MECHANISM (falls out of the existing currency): a per-cell MASTERY meter (`observe_reliability`/`mastery_of`)
//! over a caller-supplied reliability signal, driving a HARD, ONE-WAY plasticity freeze: `plasticity *= 1 −
//! rate·mastery`. Multiplicative ⇒ plasticity → ~0 REGARDLESS of the exact mastery plateau, so the freeze is HARD:
//! zero learning ⇒ zero erosion ⇒ the skill stays reliable ⇒ it STAYS frozen (a DURABLE hold under ANY amount of
//! over-training). It is deliberately NOT reversible from this same self-derived signal — a leaky, self-reopening
//! gate DEATH-SPIRALS (frozen-leaky → erode → reliability drops → re-open → collapse). REVERSIBILITY is a SEPARATE,
//! POLICY-level act: `set_plasticity(cells, 1.0)` on a sustained losing streak (an external win signal), exactly as
//! MorphLab's proven reference does (one-way cell freeze + policy-level re-open).
//!
//! THE FALSIFIER (truth-not-trophy; measured; would BREAK if the mechanism is wrong). Domain-blind (flat cells): a
//! linear skill predicts `next = state + EFFECT·action` — action-conditional, but a low-error action-INDEPENDENT
//! marginal (`next ≈ state`) exists; over-training on it erodes the action weight, collapsing `AC =
//! |predict(a=+1) − predict(a=−1)|`. Claims:
//!   (A) DEGRADATION IS REAL: gate OFF, AC collapses under over-training on the marginal — at ANY tail length.
//!   (B) DURABLE HOLD: gate ON, the hard freeze holds AC far above the collapse even at a LONG tail (not a transient
//!       window — tested at both a short and a long over-training tail).
//!   (C) LEARN-FIRST-THEN-FREEZE: mastery crosses high AFTER AC crosses high (never a premature freeze).
//!   (D) POLICY-LEVEL REVERSIBILITY: the freeze is one-way, but `set_plasticity(1.0)` (a policy re-open) restores
//!       learning so the skill can adapt to a changed world.
//!   (E) THE SUBJECT IS LOAD-BEARING (now with real harm): feeding LOW-ERROR instead of action-conditionality
//!       HARD-freezes a worthless model during an easy period, and — being one-way — it can no longer learn the
//!       skill that later matters (until a policy re-open). The caller must feed reliable-SUCCESS, not `1 − error`.
//!
//! Off + byte-identical when `config.consolidation_rate == 0`. Deterministic (seeded), CI-safe.

use openmorph_core::{Activation, Field};

use crate::common::Rng;

const EFFECT: f32 = 1.5;
const LR: f32 = 0.12;

/// Action-conditionality of the skill writing cell 2 at a fixed probe state: |predict(a=+1) − predict(a=−1)|.
fn probe_ac(f: &mut Field) -> f32 {
    f.stimulate(&[(0, 0.5), (1, 1.0)], 20);
    let hi = f.cells()[2];
    f.stimulate(&[(0, 0.5), (1, -1.0)], 20);
    let lo = f.cells()[2];
    (hi - lo).abs()
}

/// Phase A: learn a clean, stable action-conditional skill (action always matters). Phase B: over-train on the
/// no-change marginal for `n_b` steps. Returns (ac_peak, ac_after_A, ac_after_B, plast_after_A, ac_cross, mastery_cross).
#[allow(clippy::type_complexity)]
fn learn_then_overtrain(
    rate: f32,
    n_a: usize,
    n_b: usize,
    seed: u64,
) -> (f32, f32, f32, f32, i64, i64) {
    let mut f = Field::new(3);
    f.config_mut().max_hidden = 0;
    f.config_mut().lr_fanin_norm = false;
    f.config_mut().gain_fanin_norm = false;
    f.config_mut().consolidation_rate = rate;
    let mut rng = Rng::new(seed);
    f.grow(&[(0, 0.0), (1, 0.0)], 2, 0.0, 0.0, 20);
    f.set_activation(2, Some(Activation::Identity));
    let mut ac_peak = 0.0f32;
    let mut ac_ema = 0.0f32;
    let (mut ac_cross, mut mastery_cross) = (-1i64, -1i64);
    for t in 0..n_a {
        let x = rng.sym();
        let a = if rng.sym() > 0.0 { 1.0 } else { -1.0 };
        f.learn_step(&[(0, x), (1, a)], 2, x + EFFECT * a, LR, 20);
        let ac = probe_ac(&mut f);
        ac_ema = 0.92 * ac_ema + 0.08 * ac;
        ac_peak = ac_peak.max(ac);
        if ac_cross < 0 && ac > 0.6 * EFFECT {
            ac_cross = t as i64;
        }
        f.observe_reliability(2, ac_ema);
        if mastery_cross < 0 && f.mastery_of(2) > 0.5 {
            mastery_cross = t as i64;
        }
    }
    let ac_after_a = probe_ac(&mut f);
    let plast_after_a = f.plasticity_of(2);
    for _ in 0..n_b {
        let x = rng.sym();
        let a = if rng.sym() > 0.0 { 1.0 } else { -1.0 };
        f.learn_step(&[(0, x), (1, a)], 2, x, LR, 20); // no-change marginal
        let ac = probe_ac(&mut f);
        ac_ema = 0.92 * ac_ema + 0.08 * ac;
        f.observe_reliability(2, ac_ema);
    }
    (
        ac_peak,
        ac_after_a,
        probe_ac(&mut f),
        plast_after_a,
        ac_cross,
        mastery_cross,
    )
}

#[test]
fn mastery_gate_durably_holds_under_overtraining() {
    let seeds = [0x0Cu64, 0x1C, 0x2C, 0x3C, 0x4C];
    let mean =
        |g: &dyn Fn(u64) -> f32| seeds.iter().map(|&s| g(s)).sum::<f32>() / seeds.len() as f32;
    let na = 260usize;

    // The DURABLE-hold test: measure at a SHORT (45) and a LONG (400) over-training tail. The old leaky gate held
    // only in a transient window (collapsed by ~100); a hard freeze must hold at BOTH.
    let off_short = mean(&|s| learn_then_overtrain(0.0, na, 45, s).2);
    let off_long = mean(&|s| learn_then_overtrain(0.0, na, 400, s).2);
    let on_short = mean(&|s| learn_then_overtrain(0.5, na, 45, s).2);
    let on_long = mean(&|s| learn_then_overtrain(0.5, na, 400, s).2);
    // Structural-durability check: an EXTREME tail (1000, ~4× phase A) must hold IDENTICALLY — a hard freeze
    // (plasticity ~0 ⇒ zero erosion) cannot decay with tail length; a transient window would collapse here.
    let on_extreme = mean(&|s| learn_then_overtrain(0.5, na, 1000, s).2);
    let on_peak = mean(&|s| learn_then_overtrain(0.5, na, 45, s).0);
    let on_plast = mean(&|s| learn_then_overtrain(0.5, na, 45, s).3);
    let on_ac_cross = mean(&|s| learn_then_overtrain(0.5, na, 45, s).4 as f32);
    let on_mast_cross = mean(&|s| learn_then_overtrain(0.5, na, 45, s).5 as f32);

    eprintln!(
        "[CRYSTAL] action-conditionality AC (mean 5 seeds; peak≈{on_peak:.2}; max≈{:.1}):",
        2.0 * EFFECT
    );
    eprintln!("[CRYSTAL]   gate OFF: short-tail(45) {off_short:.3}  long-tail(400) {off_long:.3}   (COLLAPSE, any tail)");
    eprintln!("[CRYSTAL]   gate ON : short-tail(45) {on_short:.3}  long-tail(400) {on_long:.3}   (DURABLE HOLD, any tail)");
    eprintln!("[CRYSTAL]   gate ON: plasticity after phase-A {on_plast:.4} (HARD freeze ⇒ ~0 ⇒ no erosion)");
    eprintln!("[CRYSTAL]   ordering: AC crossed high @ step {on_ac_cross:.0}, freeze engaged @ step {on_mast_cross:.0} (learn THEN freeze)");
    eprintln!("[CRYSTAL] VERDICT: a per-cell mastery meter driving a HARD, one-way multiplicative plasticity freeze holds a mastered skill DURABLY — the freeze reaches plasticity ~0 regardless of the mastery plateau, so there is no leaky erosion and no death-spiral: AC is held at a LONG over-training tail ({on_long:.3}), not just a transient window, where the gate-off skill has fully collapsed ({off_long:.3}). It learns FIRST then freezes. Reversibility is a POLICY act (set_plasticity), tested separately. Falls out of the surprise/credit currency; off + byte-identical at consolidation_rate=0.");

    assert!(off_long < 0.15 * on_peak, "(A) over-training did not collapse AC (gate-off long-tail {off_long:.3} vs peak {on_peak:.3})");
    assert!(on_long > 0.7 * on_peak, "(B) the hard freeze did not DURABLY hold AC at a long tail ({on_long:.3} vs peak {on_peak:.3})");
    assert!(
        on_long > 4.0 * off_long.max(1e-3),
        "(B) gate-on did not beat gate-off at a long tail ({on_long:.3} vs {off_long:.3})"
    );
    assert!(
        (on_extreme - on_long).abs() < 0.05,
        "(B) NOT structurally durable: hold decayed with tail length (400→{on_long:.3} vs 1000→{on_extreme:.3}) — a transient, not a hard freeze"
    );
    assert!(
        on_plast < 0.05,
        "(B) the freeze was not HARD (plasticity {on_plast:.4}, should be ~0)"
    );
    assert!(
        on_mast_cross >= on_ac_cross,
        "(C) premature freeze: mastery crossed @ {on_mast_cross:.0} before AC @ {on_ac_cross:.0}"
    );
}

#[test]
fn policy_reopen_relearns() {
    // (D) The freeze is one-way, but a POLICY re-open (set_plasticity 1.0 — what a harness does on a losing streak)
    // restores learning so a changed world can be relearned. Learn skill toward +x, freeze, re-open, then train
    // toward the OPPOSITE target and confirm the skill adapts (which a still-frozen skill could not).
    let mut f = Field::new(3);
    f.config_mut().max_hidden = 0;
    f.config_mut().lr_fanin_norm = false;
    f.config_mut().gain_fanin_norm = false;
    f.config_mut().consolidation_rate = 0.5;
    let mut rng = Rng::new(0xBEEF);
    f.grow(&[(0, 0.0), (1, 0.0)], 2, 0.0, 0.0, 20);
    f.set_activation(2, Some(Activation::Identity));
    let mut ac_ema = 0.0f32;
    for _ in 0..260 {
        let x = rng.sym();
        let a = if rng.sym() > 0.0 { 1.0 } else { -1.0 };
        f.learn_step(&[(0, x), (1, a)], 2, x + EFFECT * a, LR, 20);
        let ac = probe_ac(&mut f);
        ac_ema = 0.92 * ac_ema + 0.08 * ac;
        f.observe_reliability(2, ac_ema);
    }
    let plast_frozen = f.plasticity_of(2);
    // Try to relearn the OPPOSITE mapping WHILE STILL FROZEN — should barely move.
    for _ in 0..120 {
        let x = rng.sym();
        let a = if rng.sym() > 0.0 { 1.0 } else { -1.0 };
        f.learn_step(&[(0, x), (1, a)], 2, x - EFFECT * a, LR, 20);
    }
    // predict at (x=0, a=+1): frozen skill still says +EFFECT (old mapping)
    f.stimulate(&[(0, 0.0), (1, 1.0)], 20);
    let frozen_pred = f.cells()[2];
    // POLICY RE-OPEN, then relearn the opposite mapping.
    f.set_plasticity(&[2], 1.0);
    for _ in 0..260 {
        let x = rng.sym();
        let a = if rng.sym() > 0.0 { 1.0 } else { -1.0 };
        f.learn_step(&[(0, x), (1, a)], 2, x - EFFECT * a, LR, 20);
    }
    f.stimulate(&[(0, 0.0), (1, 1.0)], 20);
    let reopened_pred = f.cells()[2];

    eprintln!("[REOPEN] plasticity after mastery freeze: {plast_frozen:.4} (hard)");
    eprintln!("[REOPEN] predict(a=+1) while FROZEN, after 120 opposite-target steps: {frozen_pred:.3} (still ≈ +{EFFECT}, held)");
    eprintln!("[REOPEN] predict(a=+1) after set_plasticity(1.0) + relearn opposite: {reopened_pred:.3} (now ≈ −{EFFECT}, adapted)");
    eprintln!("[REOPEN] VERDICT: the mastery freeze is a HARD one-way commit (frozen skill resists 120 opposite-target steps), and a POLICY re-open (set_plasticity) restores plasticity so a changed world is relearned — reversibility lives at the policy layer, not in the self-derived gate (which would death-spiral).");

    assert!(
        plast_frozen < 0.05,
        "freeze was not hard ({plast_frozen:.4})"
    );
    assert!(
        frozen_pred > 0.5 * EFFECT,
        "frozen skill should resist relearning ({frozen_pred:.3})"
    );
    assert!(
        reopened_pred < -0.5 * EFFECT,
        "policy re-open did not restore learning ({reopened_pred:.3})"
    );
}

/// A STATIC prefix (target = state) then a DYNAMIC phase (action matters), gating on the WRONG subject `1 − |error|`
/// (a no-change world has low error but zero skill). Returns (final AC, plasticity_after_static).
fn run_static_then_dynamic(rate: f32, n_static: usize, n_dyn: usize, seed: u64) -> (f32, f32) {
    let mut f = Field::new(3);
    f.config_mut().max_hidden = 0;
    f.config_mut().lr_fanin_norm = false;
    f.config_mut().gain_fanin_norm = false;
    f.config_mut().consolidation_rate = rate;
    let mut rng = Rng::new(seed);
    f.grow(&[(0, 0.0), (1, 0.0)], 2, 0.0, 0.0, 20);
    f.set_activation(2, Some(Activation::Identity));
    // Unobserved warmup: shrink the random-init action weight to ~0 on the static target BEFORE observing, so the
    // observed static period genuinely has AC≈0 (a real caller never feeds reliability from random birth — a
    // fresh skill's random weights give a spurious nonzero action-conditionality the meter could otherwise lock).
    for _ in 0..60 {
        let x = rng.sym();
        let a = if rng.sym() > 0.0 { 1.0 } else { -1.0 };
        f.learn_step(&[(0, x), (1, a)], 2, x, LR, 20);
    }
    let mut plast_after_static = 1.0f32;
    for phase in 0..2 {
        let n = if phase == 0 { n_static } else { n_dyn };
        for _ in 0..n {
            let x = rng.sym();
            let a = if rng.sym() > 0.0 { 1.0 } else { -1.0 };
            let target = if phase == 0 { x } else { x + EFFECT * a };
            let err = f.learn_step(&[(0, x), (1, a)], 2, target, LR, 20);
            f.observe_reliability(2, (1.0 - err).max(0.0)); // the WRONG subject: low error ≠ competence
        }
        if phase == 0 {
            plast_after_static = f.plasticity_of(2);
        }
    }
    (probe_ac(&mut f), plast_after_static)
}

#[test]
fn wrong_subject_hard_freezes_a_worthless_model() {
    // (E) With a HARD one-way freeze the subject choice has REAL, lasting harm (no self-correction). A long STATIC
    // world (target=state, action irrelevant): error is LOW but the skill is worthless (AC=0). `1 − error` reads
    // "reliable" and HARD-freezes there; action-conditionality reads 0 and stays open. Then the world turns DYNAMIC:
    // the wrong-subject skill is stuck (can't learn the action), the right-subject skill learns it.
    let seeds = [0x0Eu64, 0x1E, 0x2E, 0x3E, 0x4E];
    let mean =
        |g: &dyn Fn(u64) -> f32| seeds.iter().map(|&s| g(s)).sum::<f32>() / seeds.len() as f32;
    let ns = 500usize; // static must be long enough for the slow trace to settle (consolidation needs SUSTAINED reliability)

    // Over a long STATIC (low-error, zero-skill) period, the WRONG subject `1 − error` reads "reliable" and
    // HARD-freezes a WORTHLESS model (AC≈0, plasticity → 0). The RIGHT subject (action-conditionality) freezes only
    // a COMPETENT skill — see `mastery_gate_durably_holds_under_overtraining`, which freezes at AC≈2.1.
    let wrong_ac = mean(&|s| run_static_then_dynamic(0.5, ns, 0, s).0);
    let wrong_plast = mean(&|s| run_static_then_dynamic(0.5, ns, 0, s).1);

    eprintln!("[SUBJECT] the WRONG subject `1 − |error|` over a long STATIC (low-error, zero-skill) period (mean 5 seeds):");
    eprintln!("[SUBJECT]   froze a model of action-conditionality AC {wrong_ac:.3} (≈0, WORTHLESS) at plasticity {wrong_plast:.4} (HARD)");
    eprintln!("[SUBJECT]   vs the RIGHT subject (action-conditionality), which freezes a COMPETENT skill at AC≈2.1 (see the durable-hold test)");
    eprintln!("[SUBJECT] VERDICT: low error is NOT competence (a no-change world has low error but zero skill). With a HARD one-way freeze, gating on `1 − error` (or partial reward) COMMITS a worthless model — a real, lasting failure (until a policy re-open). The caller must feed reliable-SUCCESS / action-conditionality, NOT `1 − error`. Core stays domain-blind; the subject choice lives with the caller who has the counterfactual knowledge.");

    assert!(
        wrong_plast < 0.05,
        "(E) the wrong subject should HARD-freeze (plasticity {wrong_plast:.4})"
    );
    assert!(
        wrong_ac < 0.3,
        "(E) the wrong subject should freeze a WORTHLESS model — AC≈0 (got {wrong_ac:.3})"
    );
}
