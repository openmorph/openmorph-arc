// SPDX-License-Identifier: MIT-0
//! THE UNIFIED FREE-ENERGY ACTOR — perceive, act, and EXPLORE fall out of ONE functional, minimised by ONE
//! rule, with ZERO exploration knobs. The project had been ACCRETING acting machinery — a hunger-gated
//! openmindedness (ε; removed with the `ActScore` rule), an ensemble-disagreement weight (β in `select_explore`), a
//! novelty term, a reward memory.
//! Active inference DELETES the glue: exploration and exploitation are the SAME quantity — the EXPECTED FREE
//! ENERGY of an action — over the organism's OWN learned world model. For a Gaussian belief `o ~ N(μ_a, σ_a²)`
//! about the next observation under action `a`, and a preferred observation `s*`:
//!
//!   G(a) = pragmatic(a) − epistemic(a) = (μ_a − s*)²  −  σ_a²
//!            \__ expected surprise about reaching the preferred state __/   \__ expected information gain __/
//!
//! Both terms are in the SAME currency — units of the preferred observation SQUARED (o²) — so they add with
//! coefficient ONE. There is NO β to weight exploration against exploitation: the trade-off is the model's OWN
//! epistemic variance (the disagreement of a varied-prior ensemble — the substrate's `select_explore`
//! uncertainty), which SELF-ANNEALS. Where the organism has NOT been, the ensemble disagrees (σ² large) ⇒ the
//! epistemic term dominates ⇒ it EXPLORES; where it HAS been, the ensemble has converged (σ² → 0) ⇒ only the
//! pragmatic term remains ⇒ it EXPLOITS. Exploration is not a bonus bolted on — it is what "minimise expected
//! free energy" MEANS while the model is still ignorant. REWARD is the preferred prior `s*`; HUNGER would be its
//! precision (weight on the pragmatic term); EXPLORATION is the epistemic term — one quantity.
//!
//! The world is a DECEPTIVE bandit — the discriminator that forces the point. One arm is the true peak (payoff
//! ≈ s*), several are a DECOY (a middling payoff), the rest are dead. A value-greedy actor (the SAME varied-prior
//! ensemble & learned model, epistemic term DELETED) samples until it finds a decoy, then — every UNSAMPLED arm's
//! expected payoff sitting just below the decoy — exploits the decoy forever and never discovers the peak. The
//! free-energy actor's epistemic term keeps it sampling the uncertain arms (their broad prior makes the peak
//! PLAUSIBLE, so the information is worth the risk), finds the peak, and — its estimate now certain — the SAME
//! functional exploits it. You MUST explore to find the preferred state, and exploration is a free consequence
//! of the one functional. (This is exactly `substrate_voi`'s deceptive bandit, but with exploration folded INTO
//! the single free-energy functional — no separate `select_explore` β.)
//!
//! Truth-not-trophy — the mechanism is scored INSIDE, exactly like `substrate_purpose`:
//!   * the ONLY difference between the two actors is the epistemic term (same varied-prior ensemble, same
//!     learned model, same preferred prior) — a win isolates the term, nothing else;
//!   * SCRAMBLED control — an action-BLIND ensemble (the model can't tell the arms apart) collapses the
//!     free-energy actor: no valid model ⇒ no purposeful choice. The learned model IS the load;
//!   * the metric is intrinsic free energy (mean |o − s*|), never an external label;
//!   * the world model is GROWN online (`grow`), never given.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_actor -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const ARMS: usize = 8;
const SSTAR: f32 = 0.9; // preferred observation — "how the world should be" (kept in the tanh genome's range)
const ENSEMBLE: usize = 24; // varied-prior world models; their DISAGREEMENT is the epistemic variance

/// The world's payoff per arm: arm 0 is DEAD (0.0), arms 1..=6 are the DECOY (0.45), arm 7 is the true PEAK
/// (0.9 ≈ s*). Declared by the world (what "preferred" means), never learned.
fn payoff(a: usize) -> f32 {
    match a {
        7 => 0.90,     // the true peak — only reachable by exploring
        1..=6 => 0.45, // the decoy: middling, found first, traps the greedy actor
        _ => 0.0,      // dead
    }
}

/// A one-hot arm encoding over the input cells `0..ARMS` (every cell supplied so the input actually CHANGES as
/// the arm switches — this substrate is fire-on-change, a constant input never drives its units).
fn onehot(a: usize) -> Vec<(usize, f32)> {
    (0..ARMS)
        .map(|i| (i, if i == a { 1.0 } else { 0.0 }))
        .collect()
}

/// The cell holding arm `a`'s predicted payoff. Each arm is an INDEPENDENT unit (cell `ARMS + a`, reading the
/// one-hot bit `a`) — so a sampled arm's estimate converges WITHOUT dragging the others (a shared readout would
/// couple them and collapse the epistemic variance of the unvisited arms). The scramble control routes EVERY arm
/// to ONE shared cell (ARMS): the model can no longer tell the arms apart.
fn arm_cell(a: usize, ablind: bool) -> usize {
    ARMS + if ablind { 0 } else { a }
}

/// One ensemble member: `ARMS` INDEPENDENT linear units (max_hidden = 0 — a bandit is linear), arm `a`'s unit
/// reads one-hot bit `a` and writes cell `ARMS + a`. All arms share this member's prior weight `w` (an
/// EXCHANGEABLE prior — before data the arms are indistinguishable), so a sampled arm's estimate converges
/// WITHOUT dragging the unvisited arms (their disagreement — the epistemic variance — is preserved). Grown
/// online by `grow`; a sampled arm's ensemble converges to the truth, self-annealing its variance to 0.
fn model(seed: u64, w: f32) -> Field {
    let mut f = Field::new(ARMS); // cells 0..ARMS = the one-hot input
    f.ensure_cells(2 * ARMS);
    f.rng_seed(seed);
    f.config_mut().max_hidden = 0;
    for a in 0..ARMS {
        // pred = tanh(w + b); w varies per ensemble member (the disagreement), b centres it just below the decoy.
        f.add_unit(vec![a], vec![w], 0.406, ARMS + a);
    }
    f
}

/// Build the varied-prior ensemble with ANTITHETIC weight pairs (`+w`, `−w`): each pair's predictions straddle
/// the centre, so the ensemble MEAN is pinned near `tanh(0.406) ≈ 0.38` (a touch below the decoy) on EVERY seed
/// — only the spread (the epistemic variance) varies with the seed. Without this, a low random draw would make
/// the whole ensemble pessimistic and even the free-energy actor would (rationally) decline to explore.
fn ensemble(seed: u64) -> Vec<Field> {
    let mut r = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xF00D);
    r.u(); // warm up (a fresh xorshift avalanches its first sample poorly)
    let mut ens = Vec::with_capacity(ENSEMBLE);
    for j in 0..ENSEMBLE / 2 {
        let w = r.sym() * 1.3; // one magnitude per pair
        ens.push(model(seed ^ (0xA51 * (2 * j as u64 + 1)), w)); //  +w
        ens.push(model(seed ^ (0xA51 * (2 * j as u64 + 2)), -w)); //  −w (antithetic twin)
    }
    ens
}

fn pred(f: &mut Field, a: usize, ablind: bool) -> f32 {
    f.stimulate(&onehot(a), 20);
    f.cells()[arm_cell(a, ablind)]
}

/// THE ONE FUNCTIONAL — Gaussian expected free energy. `pragmatic` = squared surprise of the mean predicted
/// outcome against the preferred state; `epistemic` = the ensemble's predictive VARIANCE (expected information
/// gain). Both in o², so G = pragmatic − epistemic with coefficient ONE — no β. Greedy drops the epistemic term.
fn g_of(ens: &mut [Field], a: usize, greedy: bool, ablind: bool) -> f32 {
    let os: Vec<f32> = ens.iter_mut().map(|f| pred(f, a, ablind)).collect();
    let n = os.len() as f32;
    let mean = os.iter().sum::<f32>() / n;
    let var = os.iter().map(|o| (o - mean) * (o - mean)).sum::<f32>() / n;
    let pragmatic = (mean - SSTAR) * (mean - SSTAR);
    if greedy {
        pragmatic // value-greedy: exploit the learned model, NO epistemic term (gets trapped)
    } else {
        pragmatic - var // free-energy actor: (μ−s*)² − σ², coefficient ONE, no β
    }
}

/// Score one run. `mode`: "efe" | "greedy" | "scrambled". Every mode grows the SAME varied-prior ensemble online
/// from the SAME real payoffs; they differ ONLY in the action rule. Returns mean intrinsic free energy over the
/// 2nd half (lower = the world is how the organism prefers it).
fn run(mode: &str, seed: u64) -> f32 {
    let greedy = mode == "greedy";
    let ablind = mode == "scrambled";
    let mut ens = ensemble(seed);

    let (mut err, mut cnt) = (0.0f32, 0usize);
    for step in 0..1200 {
        // pick the action minimising expected free energy G (greedy: pragmatic only) over the learned model.
        let a = (0..ARMS)
            .min_by(|&x, &y| {
                g_of(&mut ens, x, greedy, ablind).total_cmp(&g_of(&mut ens, y, greedy, ablind))
            })
            .unwrap();
        // The world reveals the chosen arm's payoff; grow every model toward it (the model is GROWN, never given).
        let r = payoff(a);
        for f in ens.iter_mut() {
            for _ in 0..2 {
                f.grow(&onehot(a), arm_cell(a, ablind), r, 0.15, 20);
            }
        }
        if step >= 700 {
            err += (r - SSTAR).abs(); // intrinsic free energy vs the preferred state
            cnt += 1;
        }
    }
    err / cnt as f32
}

fn avg(mode: &str, seeds: u64) -> f32 {
    (0..seeds).map(|s| run(mode, 0x1CE + s)).sum::<f32>() / seeds as f32
}

#[test]
fn exploration_falls_out_of_one_free_energy_functional_with_no_knobs() {
    let seeds = 6;
    let greedy = avg("greedy", seeds);
    let efe = avg("efe", seeds);
    let scrambled = avg("scrambled", seeds);

    eprintln!("[FREE-ENERGY ACTOR] intrinsic free energy = mean |o − s*| over the 2nd half (lower = the world is how the organism prefers it); the decoy sits at |o−s*|=0.45, the true peak at |o−s*|=0.00:");
    eprintln!("[FREE-ENERGY ACTOR]   value-greedy (pragmatic only) : {greedy:.3}   <- trapped on the decoy");
    eprintln!("[FREE-ENERGY ACTOR]   free-energy actor (G=prag-epi): {efe:.3}   <- explores, finds the peak, exploits it");
    eprintln!("[FREE-ENERGY ACTOR]   scrambled (action-blind model): {scrambled:.3}   <- no valid model => no purposeful choice");
    eprintln!("[FREE-ENERGY ACTOR] VERDICT: one functional G(a) = (μ-s*)² - σ² — pragmatic minus epistemic, SAME currency (o²), coefficient ONE — reproduces exploration with ZERO knobs. The value-greedy actor (identical ensemble & learned model, epistemic term DELETED) exploits the first decoy it finds and never discovers the peak ({greedy:.2}); the free-energy actor's epistemic variance keeps it sampling the uncertain arms until the peak is found, then the SAME functional exploits it ({efe:.2}). No ε, no β, no novelty weight, no TD target — the explore/exploit trade-off is the model's own epistemic variance, which self-anneals. Truth-not-trophy: an action-BLIND model collapses it ({scrambled:.2}) — the learned model IS the load.");

    // the free-energy actor finds the peak; the value-greedy actor is trapped on the decoy
    assert!(
        efe < 0.10,
        "the free-energy actor must find the preferred state (got {efe:.3})"
    );
    assert!(
        greedy > 0.30,
        "the value-greedy baseline must be trapped short of the peak (got {greedy:.3})"
    );
    assert!(
        efe < greedy * 0.4,
        "the epistemic term (the ONLY difference between the two actors) must break the trap ({efe:.3} vs {greedy:.3})"
    );
    // the model is the load — an action-blind ensemble cannot choose purposefully
    assert!(
        scrambled > 0.30,
        "a scrambled (action-blind) model must collapse the actor back to no-better-than-the-decoy ({scrambled:.3})"
    );
}
