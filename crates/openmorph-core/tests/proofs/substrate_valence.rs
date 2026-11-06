// SPDX-License-Identifier: MIT-0
//! substrate_valence — THE VALENCE FACE OF `V`: a grown SIGNED, self-annealing, LINEAR currency that
//! COMPLETED `act_score` where the free-energy `G` could not (research spike #5 → the field-theory-completing build),
//! and has replaced it: the valence rule is the default since 1.6, and `act_score` is removed.
//!
//! A research probe (`substrate_grown_act`, deleted with `act_score`) + `docs/research/last-authored-audit.md`
//! pinned the last authored residue of the acting policy `act_score(o) = −value(o) + drive(o) + om·novelty(o)` to
//! ONE currency-less term — the binary novelty step `1[drive==0]` — and proved two things term by term:
//!   • the binary flag does NOT grow in place: swapping it for graded σ² at weight `om` STILL traps the deceptive
//!     bandit (0.300 = the binary's 0.300) — a bounded additive bonus cannot outweigh a decoy's pragmatic drive;
//!   • the currency only pays inside the WHOLE reformulated `G = value + (μ−s*)² − σ²` (→ 0.000) — BUT `G` LOSES a
//!     faculty: its SQUARED `(μ−s*)²` turns a learned aversion into a BARRIER, so it goes TAME where `act_score`'s
//!     SOFT, LINEAR `+drive` lets the value gradient override and go WILD.
//! So the irreducible authored core is the SIGNED, soft-additive (LINEAR) drive composition, and the missing
//! currency is a grown SIGNED VALENCE (self-annealing aversion/commitment) that enters the objective LINEARLY —
//! which neither μ (unsigned, squared) nor σ² (signless, epistemic) supplies ON ITS OWN.
//!
//! THE GROWN VALENCE (this build) — `Config::valence_actor`, `Field::valence_of`:
//!   valence_of(o) = μ_o + σ_o²   (the option's learned `(μ, visits)` record: the same fold `μ ← μ + lr·(r − μ)`
//!                                 the free-energy ensemble takes, and σ_o = σ0·(1−lr)^visits — the ensemble's own
//!                                 spread schedule, kept as an exact visit count since 1.6)
//!   A(o) = −value_of(o) + valence_of(o)   (ARGMAX)
//! It is the exact structural sibling of `G = value + (μ−s*)² − σ²`, differing in ONE place: the pragmatic enters
//! as the LINEAR SIGNED μ, not the SQUARED (μ−s*)². That one change is load-bearing BOTH ways:
//!   • SIGNED + LINEAR μ ⇒ a learned aversion (μ<0) is a soft HILL a value gradient overrides ("wild"), NOT the
//!     quadratic barrier `G` builds ("tame") — the faculty `G` loses is KEPT.
//!   • self-annealing +σ² at UNIT weight (prior σ0=1) ⇒ ESCAPES the deceptive bandit `act_score`'s bounded
//!     `om·novelty` traps on — the escape `G` buys is KEPT.
//! SIGNED (μ's sign = aversion/commitment), SELF-ANNEALING (σ² → 0 as the option is visited — the
//! exploratory face fades, leaving the committed sign — grounded in the visit count, no new schedule), LINEAR
//! (superposable, overridable). It dissolves BOTH authored constants `act_score`/`G` still carry — `om` AND `s*`.
//!
//! FOUR MEASURED CLAIMS (INSIDE-scored, deterministic, chance-anchored, same organism):
//!   (A) the deceptive bandit — the valence rule with its bonus DELETED (σ0 = 0) traps (0.300, as the removed
//!       `act_score` did), G escapes (~0.000), VALENCE escapes (~0.000).
//!   (B) the aversion-override — G tame (picks the far safe option), VALENCE wild (picks the aversive at-goal
//!       option, as the removed `act_score` did, UNLIKE G) — linear+signed does what squared can't.
//!   (C) the valence is genuinely SIGNED — negative for a learned aversion, positive for a commitment.
//!   (D) the valence is genuinely SELF-ANNEALING — its exploratory face σ² decays from σ0²=1 (hot/untried) → 0
//!       (cold/committed) as the option is folded, grounded in its visit count (an existing currency).
//!
//! Since 1.6 the valence rule is the DEFAULT `alive(Act)` rule (`ActRule::Valence`); the `valence_actor` flag used
//! below is its deprecated alias. The AUTHORED `act_score` arms this proof used to carry (deceptive bandit 0.300,
//! aversion-override wild) went with that rule; (A)'s control is now the valence rule itself with σ0 = 0.

use openmorph_core::{Event, Field, Outcome};

const ARMS: usize = 8;
const SSTAR: f32 = 0.9;

fn act(f: &mut Field, options: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act { options, last }) {
        Outcome::Acted { chosen } => chosen,
        _ => unreachable!(),
    }
}

/// arm 0 = dead (0.0), the PEAK arm ≈ s* (0.90) is placed by the caller, every other arm is the decoy (0.45).
fn payoff(a: usize, peak: usize) -> f32 {
    if a == peak {
        0.90
    } else if a == 0 {
        0.0
    } else {
        0.45
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// The deceptive bandit under an actor `mode`:
//   "greedy"   — the valence rule with its exploration bonus deleted (σ0 = 0 ⇒ A = −value + μ): the control.
//   "fe"       — the shipped free-energy actor G = value + (μ−s*)² − σ² (argmin).
//   "valence"  — the grown valence actor A = −value + μ + σ² (argmax).
// All three fold the last reward through their OWN native path inside `alive` (fe→the value ensemble,
// valence/greedy→the (μ, visits) record). Returns mean |o−s*| over the 2nd half (lower = reaches the preferred
// peak).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
fn run_bandit(mode: &str, peak: usize) -> f32 {
    let options: Vec<usize> = (0..ARMS).collect();
    let mut f = Field::new(ARMS);
    match mode {
        "fe" => f.config_mut().free_energy_actor = true,
        "valence" => f.config_mut().valence_actor = true,
        // the CONTROL: the same valence rule with its exploration bonus deleted (σ0 = 0 ⇒ A = −value + μ)
        "greedy" => f.config_mut().free_energy_prior_std = 0.0,
        _ => unreachable!("unknown mode {mode}"),
    }
    let (mut err, mut cnt) = (0.0f32, 0usize);
    let mut last: Option<(usize, f32)> = None;
    for step in 0..1200 {
        let a = act(&mut f, &options, last);
        let r = payoff(a, peak);
        last = Some((a, r));
        if step >= 700 {
            err += (payoff(a, peak) - SSTAR).abs();
            cnt += 1;
        }
    }
    err / cnt as f32
}

fn avg_bandit(mode: &str) -> f32 {
    // vary the PEAK so a pass cannot be tie-break luck — the actor must FIND it wherever it sits.
    [7usize, 4, 1]
        .iter()
        .map(|&p| run_bandit(mode, p))
        .sum::<f32>()
        / 3.0
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (A)  THE DECEPTIVE BANDIT — the valence ESCAPES like G (where the same rule without its σ² bonus traps).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn a_valence_escapes_the_deceptive_bandit() {
    let greedy = avg_bandit("greedy");
    let fe = avg_bandit("fe");
    let valence = avg_bandit("valence");
    eprintln!("[VALENCE A] deceptive bandit, mean |o−s*| over the 2nd half (peak 0.90, decoy 0.45, dead 0.0):");
    eprintln!("[VALENCE A]   GREEDY    (−value + μ,      σ0 = 0, argmax)  : {greedy:.3}  <- TRAPPED (no bonus: the first decoy found out-scores every untried arm)");
    eprintln!("[VALENCE A]   G         (value + (μ−s*)² − σ²,  argmin)           : {fe:.3}  <- ESCAPES (epistemic σ² at UNIT weight)");
    eprintln!("[VALENCE A]   VALENCE   (−value + μ + σ²,       argmax)           : {valence:.3}  <- ESCAPES (self-annealing +σ² at UNIT weight)");
    assert!(
        greedy > 0.25,
        "the valence rule with its bonus deleted must be trapped on the deceptive bandit (got {greedy:.3})"
    );
    assert!(
        fe < 0.10,
        "the free-energy G must escape the trap (got {fe:.3})"
    );
    assert!(
        valence < 0.10,
        "the grown VALENCE must escape the trap like G — its self-annealing +σ² pays at unit weight (got {valence:.3})"
    );
    assert!(
        valence < greedy * 0.4,
        "VALENCE must beat its bonus-deleted control decisively ({valence:.3} vs {greedy:.3})"
    );
    eprintln!("[VALENCE A] VERDICT: the grown valence escapes the deceptive bandit (like G; deleting its σ² face traps it, as the removed act_score's bounded novelty was trapped) — the σ² face enters at UNIT weight, self-annealing.");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (B)  THE AVERSION-OVERRIDE — the valence stays WILD (UNLIKE G). This is the win: linear+signed
//      does what squared can't. value gap 5.0 to a far option; a learned −2 aversion on the option AT the goal.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn b_valence_keeps_the_soft_symmetric_drive_faculty() {
    // G — the aversion becomes a quadratic barrier `(μ−s*)²` no value gap can cross (TAME).
    let tame_fe = {
        let mut w = Field::new(2);
        w.config_mut().free_energy_actor = true;
        w.prefer(0, 5.0);
        w.prefer(1, 0.0);
        for _ in 0..40 {
            act(&mut w, &[0, 1], Some((1, -2.0))); // fold the aversion to convergence (μ≈−2)
        }
        act(&mut w, &[0, 1], None)
    };
    // VALENCE — the aversion is the SIGNED LINEAR μ (a soft hill), so the value gradient overrides it (WILD).
    let (wild_valence, s0, s1, v0, v1) = {
        let mut w = Field::new(2);
        w.config_mut().valence_actor = true;
        w.prefer(0, 5.0);
        w.prefer(1, 0.0);
        for _ in 0..40 {
            act(&mut w, &[0, 1], Some((1, -2.0))); // fold the aversion into its record (μ≈−2, σ²→0)
        }
        let picked = act(&mut w, &[0, 1], None);
        // reconstruct the linear score to expose the composition: A(o) = −value(o) + valence_of(o).
        let sc = |o: usize| -> f32 { -w.value_of(o) + w.valence_of(o) };
        (picked, sc(0), sc(1), w.valence_of(0), w.valence_of(1))
    };
    eprintln!("[VALENCE B] value gap 5.0, learned aversion −2.0:");
    eprintln!("[VALENCE B]   G         picks option {tame_fe}  (TAME — the squared (μ−s*)² aversion is a barrier value can't cross)");
    eprintln!(
        "[VALENCE B]   VALENCE   picks option {wild_valence}  (WILD — the SIGNED LINEAR μ is a soft hill: A(far)={s0:+.2} < A(goal)={s1:+.2}; valence_of far={v0:+.2}, goal={v1:+.2})"
    );
    assert_eq!(
        tame_fe, 0,
        "G's squared pragmatic makes the aversion a barrier — it stays tame (the faculty G drops)"
    );
    assert_eq!(
        wild_valence, 1,
        "the grown VALENCE keeps the faculty: its SIGNED LINEAR μ lets value override the aversion (wild, UNLIKE G)"
    );
    // the composition is genuinely additive/linear: the at-goal option wins on the value gradient despite μ<0.
    assert!(v1 < 0.0, "the at-goal option carries a NEGATIVE valence (aversion), yet is still chosen (got {v1:+.2})");
    assert!(s1 > s0, "the linear score prefers the at-goal option ({s1:+.2} > {s0:+.2}) — value overrides the soft aversion hill");
    eprintln!("[VALENCE B] VERDICT: linear+signed does what squared can't — the valence keeps the soft-symmetric drive faculty G loses (the faculty the removed act_score was kept for).");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (C)  THE VALENCE IS GENUINELY SIGNED — negative for a learned aversion, positive for a commitment.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn c_valence_is_signed() {
    let mut f = Field::new(2);
    f.config_mut().valence_actor = true;
    // fold a persistent AVERSION into option 0 and a persistent COMMITMENT into option 1 (converge σ²→0).
    for _ in 0..80 {
        act(&mut f, &[0, 1], Some((0, -0.9))); // option 0: repeated punishment
        act(&mut f, &[0, 1], Some((1, 0.9))); //  option 1: repeated reward
    }
    let aversion = f.valence_of(0);
    let commitment = f.valence_of(1);
    eprintln!("[VALENCE C] valence_of after 80 folds of −0.9 (option 0) and +0.9 (option 1):");
    eprintln!(
        "[VALENCE C]   option 0 (punished) valence_of = {aversion:+.3}  <- AVERSION (negative)"
    );
    eprintln!(
        "[VALENCE C]   option 1 (rewarded) valence_of = {commitment:+.3}  <- COMMITMENT (positive)"
    );
    assert!(
        aversion < -0.5,
        "a learned aversion must give a NEGATIVE valence (got {aversion:+.3})"
    );
    assert!(
        commitment > 0.5,
        "a commitment must give a POSITIVE valence (got {commitment:+.3})"
    );
    eprintln!("[VALENCE C] VERDICT: the valence carries a genuine SIGN (aversion − / commitment +) — the currency neither μ-squared nor σ² supplies.");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (D)  THE VALENCE IS GENUINELY SELF-ANNEALING — its exploratory face σ² decays from σ0²=1 (hot/untried) → 0
//      (cold/committed) as the option is folded, grounded in its visit count (an existing currency). Fold a
//      NEUTRAL reward (r=0) so μ stays 0 and valence_of = σ² EXACTLY — isolating the exploratory face.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn d_valence_self_anneals() {
    let mut f = Field::new(1);
    f.config_mut().valence_actor = true;
    let checkpoints = [0usize, 5, 20, 60];
    let mut trace: Vec<(usize, f32)> = Vec::new();
    let mut folds = 0usize;
    // untried (never folded) — no record yet; valence_of reads the prior σ0²=1.
    trace.push((0, f.valence_of(0)));
    for &target in &checkpoints[1..] {
        while folds < target {
            act(&mut f, &[0], Some((0, 0.0))); // fold a NEUTRAL reward — μ stays 0, σ² anneals
            folds += 1;
        }
        trace.push((folds, f.valence_of(0)));
    }
    eprintln!("[VALENCE D] the exploratory face σ² (= valence_of at neutral μ=0) as the option is folded:");
    for &(n, s) in &trace {
        eprintln!(
            "[VALENCE D]   folds {n:>2}: σ² = {s:.4}  ({})",
            if s > 0.5 {
                "HOT — exploratory"
            } else if s > 0.05 {
                "warming"
            } else {
                "COLD — committed"
            }
        );
    }
    // hot at the prior (untried), cold after convergence, and strictly monotone in between (self-annealing).
    assert!(
        (trace[0].1 - 1.0).abs() < 1e-4,
        "an untried option's exploratory face must be the prior σ0²=1 (got {:.4})",
        trace[0].1
    );
    for w in trace.windows(2) {
        assert!(
            w[1].1 < w[0].1,
            "σ² must anneal monotonically as the option is visited ({:.4} → {:.4} at folds {}→{})",
            w[0].1,
            w[1].1,
            w[0].0,
            w[1].0
        );
    }
    assert!(
        trace.last().unwrap().1 < 0.05,
        "the exploratory face must anneal toward 0 (cold/committed) after convergence (got {:.4})",
        trace.last().unwrap().1
    );
    eprintln!("[VALENCE D] VERDICT: the valence self-anneals σ0²=1 → 0 with NO external schedule — grounded in the visit count (soft when hot/uncertain, sharp when cold/committed).");
}
