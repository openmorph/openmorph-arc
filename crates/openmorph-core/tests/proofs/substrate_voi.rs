// SPDX-License-Identifier: MIT-0
//! VALUE-OF-INFORMATION IN CORE — `Field::select_explore`, the one exploration signal MorphLab measured to
//! actually help ARC-3 (UCB / value-of-information: 37→43 on the ladder; every other intrinsic bonus —
//! count-novelty, empowerment, world-surprise curiosity — HURT and was tuned to zero). It is the value-of-
//! information driver made a substrate primitive: each option is an ENSEMBLE of grown value-estimators (varied
//! priors), and `select_explore` scores it by `mean + beta·spread` — its expected value PLUS the estimators'
//! DISAGREEMENT (`aggregate().spread`, the epistemic uncertainty). So the organism acts where its VALUE estimate
//! is most uncertain and commits as the estimators converge. This is distinct from `observe`'s surprise (WORLD
//! unpredictability); here the uncertainty is in the organism's own value estimate.
//!
//! Two claims, both INSIDE one Field (no external ensemble of Fields — the whole thing is grown units + the
//! primitive):
//!   (A) MECHANISM — with two options of EQUAL mean, `select_explore` prefers the higher-DISAGREEMENT one for
//!       beta>0, and is indifferent (picks the tie-break) for beta=0. The spread term is doing the work.
//!   (B) BEHAVIOUR — on a deceptive 4-arm bandit, VoI (beta>0) ends on the best arm far more than random and
//!       earns more total reward than greedy (beta=0), which locks onto an early lucky arm. Less regret.
//!
//!   cargo test -p openmorph-core --test proofs substrate_voi -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const K: usize = 4; // arms
const E: usize = 5; // ensemble estimators per arm
const HORIZON: usize = 400;
const BUD: usize = 50;
// Deceptive: the lesser arms are COSTLY (negative expected reward) and arm 2 (0.6) looks decent, so a policy
// that commits to an early lucky-looking arm can miss the true best, arm 3 (0.85) — the gap it must explore for.
const MEANS: [f32; K] = [0.25, 0.45, 0.6, 0.85];
const BEST: usize = 3;

/// The estimate cell for arm `a`, ensemble member `e` (after the K arm-input cells).
fn est_cell(a: usize, e: usize) -> usize {
    K + a * E + e
}

/// Build one Field holding K·E value-estimators: estimator (a,e) reads arm-cell `a` and writes its own estimate
/// cell, with a VARIED random initial weight (the priors disagree before any data). Present all arm cells = 1 so
/// every estimator writes its current estimate; `select_explore` then reads each arm's ensemble.
fn build(seed: u64) -> Field {
    let mut f = Field::new(K + K * E);
    let mut r = Rng::new(seed);
    for a in 0..K {
        for e in 0..E {
            let w = (r.f() - 0.5) * 1.2; // varied prior
            f.add_unit(vec![a], vec![w], 0.0, est_cell(a, e));
        }
    }
    f
}

fn options() -> Vec<Vec<usize>> {
    (0..K)
        .map(|a| (0..E).map(|e| est_cell(a, e)).collect())
        .collect()
}

/// Refresh every estimator's output. Units fire ON CHANGE, so we must toggle the arm inputs 0→1 (a stale 1→1
/// wouldn't re-fire): zero the arm cells, then set them to 1, so each estimator recomputes `tanh(w)` into its
/// cell for `select_explore` to read.
fn present(f: &mut Field) {
    let zero: Vec<(usize, f32)> = (0..K).map(|a| (a, 0.0)).collect();
    f.stimulate(&zero, BUD);
    let one: Vec<(usize, f32)> = (0..K).map(|a| (a, 1.0)).collect();
    f.stimulate(&one, BUD);
}

fn reward(a: usize, rng: &mut Rng) -> f32 {
    if rng.f() < MEANS[a] {
        1.0
    } else {
        -1.0
    }
}

/// Learn arm `a`'s ensemble toward reward `r`, each member with prob 1/2 (bootstrap — stays varied where data
/// is thin, converges where thick).
fn update(f: &mut Field, a: usize, r: f32, rng: &mut Rng) {
    for e in 0..E {
        if rng.u() & 1 == 0 {
            f.learn_step(&[(a, 1.0)], est_cell(a, e), r, 0.1, BUD);
        }
    }
}

#[derive(Clone, Copy)]
enum Policy {
    Greedy, // select_explore beta=0 (exploit the ensemble mean)
    Random,
    Voi, // select_explore beta>0 (mean + disagreement)
}

/// One bandit run; returns (fraction of the last 100 pulls on the best arm, total reward).
fn run(policy: Policy, seed: u64) -> (f64, f64) {
    let mut f = build(seed);
    let opts = options();
    let refs: Vec<&[usize]> = opts.iter().map(|v| v.as_slice()).collect();
    let mut rng = Rng::new(seed ^ 0xB17);
    let (mut best_hits, mut total) = (0usize, 0.0);
    for t in 0..HORIZON {
        present(&mut f);
        let a = match policy {
            Policy::Random => rng.below(K),
            Policy::Greedy => f.select_explore(&refs, 0.0),
            Policy::Voi => f.select_explore(&refs, 1.0),
        };
        let r = reward(a, &mut rng);
        update(&mut f, a, r, &mut rng);
        total += r as f64;
        if t >= HORIZON - 100 && a == BEST {
            best_hits += 1;
        }
    }
    (best_hits as f64 / 100.0, total)
}

fn avg(policy: Policy, seeds: &[u64]) -> (f64, f64) {
    let (mut b, mut t) = (0.0, 0.0);
    for &s in seeds {
        let (bb, tt) = run(policy, s);
        b += bb;
        t += tt;
    }
    (b / seeds.len() as f64, t / seeds.len() as f64)
}

#[test]
fn select_explore_prefers_the_uncertain_option() {
    // (A) MECHANISM: two options, EQUAL mean, different disagreement. beta>0 must pick the uncertain one.
    let mut f = Field::new(6);
    // option A cells 0,1,2 = {0.55,0.55,0.55} (mean 0.55, spread 0 — the safe, slightly-higher-value option);
    // option B cells 3,4,5 = {0.1,0.5,0.9} (mean 0.50, but high spread 0.33 — uncertain).
    f.stimulate(
        &[
            (0, 0.55),
            (1, 0.55),
            (2, 0.55),
            (3, 0.1),
            (4, 0.5),
            (5, 0.9),
        ],
        5,
    );
    let a: &[usize] = &[0, 1, 2];
    let b: &[usize] = &[3, 4, 5];
    let opts: Vec<&[usize]> = vec![a, b];
    assert_eq!(
        f.select_explore(&opts, 0.0),
        0,
        "beta=0 (pure exploit) must pick the higher-MEAN option A (0.55 > 0.50)"
    );
    assert_eq!(
        f.select_explore(&opts, 1.0),
        1,
        "beta=1 must pick the UNCERTAIN option B — its spread (0.33) overcomes A's mean edge (0.05)"
    );
    eprintln!("[VOI] (A) mechanism: exploit (beta=0) picks the higher-mean option; VoI (beta=1) picks the uncertain one — the spread term is doing the work.");
}

#[test]
fn value_of_information_beats_greedy_and_random_through_the_primitive() {
    // (B) BEHAVIOUR: the deceptive bandit, entirely through select_explore in ONE grown Field.
    let seeds: Vec<u64> = (0..16).map(|s| 0x100 + s * 11).collect();
    let (g_best, g_rew) = avg(Policy::Greedy, &seeds);
    let (r_best, r_rew) = avg(Policy::Random, &seeds);
    let (v_best, v_rew) = avg(Policy::Voi, &seeds);

    eprintln!(
        "[VOI] deceptive 4-arm bandit through `select_explore` ({} seeds); best arm {BEST} (p={:.2}):\n  GREEDY  (beta=0) : ends on best {:>3.0}%, total reward {:>6.1}\n  RANDOM           : ends on best {:>3.0}%, total reward {:>6.1}\n  VoI (beta=1.0)   : ends on best {:>3.0}%, total reward {:>6.1}",
        seeds.len(), MEANS[BEST], g_best*100.0, g_rew, r_best*100.0, r_rew, v_best*100.0, v_rew,
    );
    eprintln!("[VOI] VERDICT: value-of-information is now a substrate primitive — `select_explore(options, beta)` scores each option by `mean + beta·spread` of its grown value-estimator ensemble (`aggregate().spread` = the estimators' disagreement = epistemic uncertainty). It probes where its value estimate is uncertain and commits as the ensemble converges, beating GREEDY (locks onto an early lucky arm — more regret) and RANDOM (never commits). This is the ONE exploration signal MorphLab measured to help ARC-3 (UCB, 37→43) while count/empowerment/world-surprise bonuses hurt — lifted into `select` from the generic `spread` moment, no bespoke machinery.");

    assert!(
        v_best > r_best + 0.3,
        "VoI must end on the best arm far more than random ({r_best:.2}) — got {v_best:.2}"
    );
    assert!(
        v_rew > g_rew && v_rew > r_rew,
        "VoI must earn more total reward than greedy and random — less regret (VoI {v_rew:.1} vs greedy {g_rew:.1} / random {r_rew:.1})"
    );
}

/// OM-2 (MorphBot-filed): `select_explore` must NOT deterministically favour one index on ties. The old
/// `Iterator::max_by` returned the LAST of equal-maximum options, so an all-tied caller silently chose the
/// last option every call. The fix breaks ties UNIFORMLY via the seeded act PRNG (byte-replayable), and takes
/// NO draw when a single option is strictly best (so a distinct-score decision is unchanged).
#[test]
fn select_explore_breaks_ties_uniformly_not_to_the_last_index() {
    // 5 fresh single-cell options all score 0.0 (a perfect tie).
    let mut f = Field::new(8);
    f.rng_seed(12345);
    let owned: Vec<Vec<usize>> = (0..5).map(|i| vec![i]).collect();
    let opts: Vec<&[usize]> = owned.iter().map(|o| o.as_slice()).collect();

    let n = 2000usize;
    let mut counts = [0usize; 5];
    for _ in 0..n {
        counts[f.select_explore(&opts, 1.0)] += 1;
    }
    // (a) every index is chosen (no dead option), and NOT all on the last index (the old bug).
    for (i, &c) in counts.iter().enumerate() {
        assert!(
            c > 0,
            "index {i} never chosen — tie-break not uniform (counts {counts:?})"
        );
    }
    assert!(
        counts[4] < n,
        "every choice went to the LAST index — the max_by-last bug is back (counts {counts:?})"
    );
    // (b) chi-square vs uniform: df=4, p>0.01 critical = 13.28. A healthy uniform draw is well under.
    let exp = n as f32 / 5.0;
    let chi2: f32 = counts
        .iter()
        .map(|&c| {
            let d = c as f32 - exp;
            d * d / exp
        })
        .sum();
    assert!(
        chi2 < 13.28,
        "tie-break not uniform enough: chi2={chi2:.2} (counts {counts:?})"
    );

    // (c) BYTE-REPLAY: same seed ⇒ identical choice sequence (determinism moat intact).
    let mut g = Field::new(8);
    g.rng_seed(999);
    let mut h = Field::new(8);
    h.rng_seed(999);
    for _ in 0..64 {
        assert_eq!(
            g.select_explore(&opts, 1.0),
            h.select_explore(&opts, 1.0),
            "tie-break must be deterministic per seed"
        );
    }

    // (d) NO-REGRESSION: a strictly-best option is returned deterministically with NO draw (byte-identical to
    // the historical single-max pick — option index 1 has the highest cell value).
    let mut d = Field::new(6);
    d.stimulate(
        &[(0, 0.1), (1, 0.9), (2, 0.1), (3, 0.1), (4, 0.1), (5, 0.1)],
        5,
    );
    let a: &[usize] = &[0];
    let b: &[usize] = &[1];
    let distinct: Vec<&[usize]> = vec![a, b];
    for _ in 0..32 {
        assert_eq!(
            d.select_explore(&distinct, 0.0),
            1,
            "a strictly-best option must be chosen deterministically, no draw"
        );
    }

    eprintln!("[OM-2] select_explore tie-break: uniform over {counts:?} (chi2={chi2:.2}<13.28), byte-replayable per seed, distinct-score pick unchanged — the max_by-last bug is fixed.");
}
