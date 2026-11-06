// SPDX-License-Identifier: MIT-0
//! A6 — THE NEXT RUNG: give `alive(Event::Act)` a live per-option value ENSEMBLE (grown μ/σ²) so it can route
//! through the EXPECTED-FREE-ENERGY functional `G = pragmatic − epistemic`, the addition #198/#200 named as the
//! prerequisite for even *considering* deleting the authored `act_score`. This proof BUILDS that ensemble in-core
//! and then, truth-not-trophy, NET-ASSESSES the deletion — it does not force it.
//!
//! THE MECHANISM (`Config::free_energy_actor`, opt-in, byte-identical off). Each option grows an ensemble of
//! varied-prior reward estimators (mean 0, std `free_energy_prior_std` = the epistemic prior σ0). When a reward
//! folds in, every member steps toward it (`m ← m + lr·(r − m)`) so a sampled option CONVERGES (σ²→0) while an
//! untried one keeps its prior disagreement. `alive(Act)` then picks
//!
//!     G(o) = value_of(o) + (μ_o − s*)² − σ_o²        [ARGMIN]
//!            \_ plan _/    \_ pragmatic _/   \_ epistemic: ensemble disagreement, via `aggregate` _/
//!
//! — the SAME `spread` moment `select_explore` reads (no new statistic), the pragmatic/epistemic terms in one
//! currency (o²) at coefficient ONE (no β, no binary novelty flag). `value_of(o)` (cost-to-go, `f32::MAX`⇒0) lets
//! the ONE functional also descend a planner's value field where one exists.
//!
//! WHAT IS MEASURED (two parts, each scored INSIDE):
//!   A) DECEPTIVE BANDIT — the epistemic term's habitat. Ensemble-`G` finds the peak for EVERY peak position
//!      (≈0.000). The epistemic term is ISOLATED: the SAME ensemble actor with the epistemic prior deleted (σ0=0 ⇒
//!      `G` is pragmatic-only) is trapped at 0.900 — deleting `−σ²` reproduces the free-energy proof's greedy trap
//!      INSIDE `alive(Act)`. (The pre-1.6 `act_score` rule was trapped here too, ≈0.30: its binary
//!      `novelty=1[untried]` stopped exploring once a decoy out-paid its 0.2 novelty weight. That rule is removed.)
//!   B) GRID PLAN — ensemble-`G` must NOT regress planning: the unified `G`'s `value_of` term descends the
//!      cost-to-go field to the goal OPTIMALLY (as the default valence rule does, `substrate_alive_act` A).
//!
//! THE DELETION, EXECUTED. This proof used to carry a part C: `G`'s SQUARED pragmatic term turns a learned aversion
//! (μ≈−2) into a quadratic barrier (μ−s*)²≈8 that a value gap of 5 cannot overcome, so `G` goes "tame" where
//! `act_score`'s soft-symmetric `drive` went "wild" — the measured reason `act_score` was NOT deleted when this
//! ensemble landed. The faculty `G` lacks is now carried by the default VALENCE rule (`−value + μ + σ²`: the signed
//! LINEAR μ is a soft hill a value gradient overrides — `substrate_valence` B, G tame vs valence wild), so
//! `act_score` was deleted and part C with it.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_alive -- --nocapture --test-threads=1

use openmorph_core::{Event, Field, Outcome};

const ARMS: usize = 8;
const SSTAR: f32 = 0.9;

/// arm 0 = dead (0.0), arms 1..=6 = the decoy (0.45), the PEAK arm (0.9 ≈ s*) is placed by the caller.
fn payoff(a: usize, peak: usize) -> f32 {
    if a == peak {
        0.90
    } else if a == 0 {
        0.0
    } else {
        0.45
    }
}

fn act(f: &mut Field, options: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act { options, last }) {
        Outcome::Acted { chosen } => chosen,
        _ => unreachable!(),
    }
}

/// mode: "fe" (ensemble-G), "greedy" (epistemic prior deleted). Returns mean |o−s*|
/// over the 2nd half (lower = the organism reaches the preferred peak).
fn run_bandit(mode: &str, peak: usize) -> f32 {
    let options: Vec<usize> = (0..ARMS).collect();
    let mut f = Field::new(ARMS);
    match mode {
        "fe" => f.config_mut().free_energy_actor = true,
        "greedy" => {
            f.config_mut().free_energy_actor = true;
            f.config_mut().free_energy_prior_std = 0.0; // DELETE the epistemic prior ⇒ G is pragmatic-only
        }
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
    // Deterministic actor; vary the PEAK position so a pass cannot be index-order luck (the sweep must FIND it).
    let peaks = [7usize, 4, 1];
    peaks.iter().map(|&p| run_bandit(mode, p)).sum::<f32>() / peaks.len() as f32
}

#[test]
fn a6_ensemble_g_actor_routes_alive_act_through_free_energy() {
    // ─────────────────────────── (A) DECEPTIVE BANDIT — the epistemic term's habitat ───────────────────────────
    let fe = avg_bandit("fe");
    let greedy = avg_bandit("greedy");
    eprintln!("[A6] (A) deceptive bandit, mean |o−s*| over the 2nd half (peak at 0.90, decoy 0.45, dead 0.0):");
    eprintln!("[A6]   ensemble-G  (μ,σ² live)        : {fe:.3}   <- explores past the decoy, finds the peak (all peaks)");
    eprintln!("[A6]   greedy      (epistemic deleted): {greedy:.3}   <- no −σ² ⇒ no exploration ⇒ TRAPPED (isolates the term)");
    assert!(fe < 0.10, "ensemble-G must find the peak (got {fe:.3})");
    // Isolate the epistemic term: the SAME ensemble actor with −σ² deleted (σ0=0) is trapped.
    assert!(
        greedy > 0.30,
        "deleting the epistemic prior must TRAP the ensemble actor — isolates the −σ² term (got {greedy:.3})"
    );

    // ─────────────────────────── (B) GRID PLAN — act_score's habitat, must NOT regress ──────────────────────────
    const N: usize = 5;
    let cell = |r: i32, c: i32| -> usize {
        (r.clamp(0, N as i32 - 1) as usize) * N + c.clamp(0, N as i32 - 1) as usize
    };
    let deltas = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)];
    let (start, goal) = (0usize, N * N - 1);
    let mut agent = Field::new(N * N);
    agent.config_mut().free_energy_actor = true; // the SAME functional descends the value field
    let adj: Vec<Vec<usize>> = (0..N * N)
        .map(|s| {
            let (r, c) = ((s / N) as i32, (s % N) as i32);
            deltas
                .iter()
                .map(|&(dr, dc)| cell(r + dr, c + dc))
                .collect()
        })
        .collect();
    agent.prefer(goal, 0.0);
    agent.relax_values(&adj, 1.0, 0);
    let (mut s, mut steps) = (start, 0usize);
    while s != goal && steps < 100 {
        let (r, c) = ((s / N) as i32, (s % N) as i32);
        let options: Vec<usize> = deltas
            .iter()
            .map(|&(dr, dc)| cell(r + dr, c + dc))
            .collect();
        s = act(&mut agent, &options, None);
        steps += 1;
    }
    let optimal = (N - 1) * 2;
    assert_eq!(
        s, goal,
        "ensemble-G must reach the goal (no plan regression)"
    );
    assert_eq!(
        steps, optimal,
        "ensemble-G's value_of term must descend the plan OPTIMALLY ({steps} vs {optimal})"
    );
    eprintln!("[A6] (B) grid plan through ensemble-G: reached the goal in {steps} steps (optimal {optimal}) — the unified G's value_of term does not regress planning.");

    eprintln!("[A6] VERDICT: the per-option value ENSEMBLE is BUILT in `alive(Act)` (`ActRule::FreeEnergy`) — it grows a live μ/σ² and routes selection through G = value_of + (μ−s*)² − σ², reusing the substrate's own `aggregate` disagreement (no new statistic). (A) it finds the deceptive bandit's peak ({fe:.3}), and the epistemic −σ² term is isolated (deleting it traps the actor at {greedy:.3}). (B) the SAME unified G's value_of term descends a plan OPTIMALLY. G's squared pragmatic cannot carry the soft-symmetric drive faculty (a learned aversion is a barrier value cannot override — tame); the default VALENCE rule carries it (substrate_valence B), which is why the authored act_score — kept until then for that faculty — is now deleted.");
}
