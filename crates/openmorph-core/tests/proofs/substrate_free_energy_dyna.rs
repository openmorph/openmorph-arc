// SPDX-License-Identifier: MIT-0
//! DYNA INTERLEAVE — EXPLORE and PLAN in ONE loop, from the ONE free-energy functional. The two prior proofs
//! isolate the two halves SEPARATELY: `substrate_free_energy_actor` (the epistemic `−σ²` term ⇒ exploration to
//! find a hidden bandit peak) and `substrate_free_energy_policy` (`relax_values` ⇒ planning depth to cross a
//! valley toward a KNOWN goal). Neither alone solves a world that needs BOTH: a preferred state whose LOCATION is
//! UNKNOWN (you must EXPLORE to discover where high-preference is) AND ACROSS A VALLEY (you must PLAN multi-step
//! to reach it, first moving AWAY from the local optimum). This proof unifies them in ONE Dyna loop.
//!
//! THE FUNCTIONAL (unchanged). `G(π) = Σ_t [ (μ_t − s*)² − σ_t² ]` — pragmatic minus epistemic, SAME currency
//! (o²), coefficient ONE. In the Dyna loop BOTH terms are active during planning:
//!   * the ORGANISM'S GOAL each step is the state minimising the immediate free energy `g(s) = (μ(s)−s*)² − σ(s)²`
//!     — so where the model is still IGNORANT (`σ²` large) an un-mapped state OUTSCORES the local pragmatic
//!     optimum and BECOMES the goal (epistemic drive); where the model is CERTAIN (`σ²→0`) only the pragmatic
//!     term survives and the PREFERRED state is the goal (exploit). The `−σ²` term FOLDS INTO the value iteration
//!     by choosing the clamped attractor `relax_values` spreads its cost-to-go from — the value field descended
//!     therefore rewards states that are BOTH preferred AND informative.
//!   * `relax_values` gives the PLANNING DEPTH: the cost-to-go to that goal has no interior local minimum, so
//!     greedy descent walks THROUGH the valley a one-step actor traps on — whether the goal is the far frontier
//!     (explore-to-find) or the discovered preferred state (plan-to-reach).
//!
//! As the model is learned the epistemic term SELF-ANNEALS (σ→0) and pragmatic planning sharpens — ONE loop, no
//! explore/exploit switch, NO ε/β/TD/novelty.
//!
//! THE WORLD needs BOTH (the discriminator). A corridor of `N` states. `h(s)`: a LOCAL peak near the start, then a
//! long deep VALLEY/plain, then the GLOBAL peak `≈ s*` as a NEEDLE at the far end. The near region THROUGH the valley is already
//! KNOWN (a brief identical "prior life" random walk grows every actor's model there); the far ASCENT to the
//! global peak is TERRA INCOGNITA (never visited ⇒ the varied-prior ensemble keeps a broad `σ²`). So the valley is
//! KNOWN-bad terrain and the unknown lies BEYOND it — the far frontier's epistemic value is INVISIBLE to a myopic
//! one-step view (it is not an adjacent novel step; the adjacent steps are the KNOWN valley). Only PLANNING sees it.
//!
//!   * PURE-EXPLORE (one-step `g(s')` over immediate neighbours, the `substrate_free_energy_actor` form) — climbs
//!     to the LOCAL peak (finds the high-preference direction), then TRAPS: its neighbours are the KNOWN valley
//!     (bad pragmatic, no epistemic — already mapped), and the far frontier's `σ²` is not one step away. FAILS to reach.
//!   * PURE-PLAN (`relax_values` with the PRAGMATIC-ONLY goal `argmin (μ−s*)²`, the `substrate_free_energy_policy`
//!     form) — plans competently to the best KNOWN state = the local peak, and with no epistemic drive NEVER
//!     ventures into the unknown, so it never discovers where the global peak is. FAILS to find.
//!   * UNIFIED DYNA (goal = `argmin g(s) = (μ−s*)² − σ²`, then `relax_values`) — the `−σ²` makes the un-mapped far
//!     region OUTSCORE the local peak, so the goal is the frontier; `relax_values` PLANS a path through the known
//!     valley to it; crossing, it DISCOVERS the global peak, whose certain pragmatic pull then holds it. SUCCEEDS
//!     at BOTH — discovers the goal AND crosses the valley.
//!   * SCRAMBLE (truth-not-trophy) — the UNIFIED actor over an action-BLIND transition model (it can't tell the
//!     moves apart). No valid model ⇒ the imagined rollout is meaningless ⇒ planning COLLAPSES. The learned MODEL
//!     is the load (the `substrate_free_energy_actor` / `substrate_free_energy_policy` inside-check).
//!
//! The ONLY difference PURE-PLAN → UNIFIED is the `−σ²` term in the goal (isolates the EPISTEMIC drive); the ONLY
//! difference PURE-EXPLORE → UNIFIED is `relax_values` (isolates PLANNING DEPTH). So the proof isolates BOTH
//! halves of the unification at once. Scored INTRINSICALLY (reached the preferred state + terminal dwell |h−s*|),
//! over multiple seeds; the world model is GROWN online (`grow`), never given. ZERO core change — a usage protocol
//! over `grow` + `prefer`/`relax_values`/`value_of`. Companion: `docs/research/unified-free-energy-actor.md` §5c.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_dyna -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 16; // corridor states 0..N
const SSTAR: f32 = 0.90; // preferred observation (the global peak's height; kept in the tanh genome's range)
const ENSEMBLE: usize = 24; // varied-prior observation models; their DISAGREEMENT is the epistemic σ²
const KNOWN_END: usize = 9; // "prior life" maps states 0..=KNOWN_END (near + valley); 10..N is TERRA INCOGNITA
const START: usize = 0; // the organism starts at the near end
const STEPS: usize = 96; // rollout length (6·N): room to explore-then-plan-then-settle

/// The landscape: observation height `h(s)`. A LOCAL peak at s=2 (0.52) near the start; a long deep VALLEY/plain
/// (s=5..14, ~0.07–0.12); the GLOBAL peak a NEEDLE at s=15 (0.90 ≈ s*). Declared by the world (what "preferred"
/// means), never learned. The needle is the ONLY high-preference state and its LOCATION is UNKNOWN until you
/// explore the far plain — and reaching it means committing across a long stretch FARTHER from s* than the local
/// peak. A needle (no intermediate "good enough" distractor on the far side) so nothing halts the search short of
/// the true optimum: the both-hard instance is UNKNOWN-location AND across-a-valley.
const H: [f32; N] = [
    0.30, 0.45, 0.52, 0.38, 0.24, 0.14, 0.09, 0.07, 0.07, 0.08, 0.10, 0.11, 0.10, 0.12, 0.11, 0.90,
];

const DELTAS: [i32; 2] = [-1, 1]; // 0 = left, 1 = right; the world clamps at the boundary

fn clamp_s(s: i32) -> usize {
    s.clamp(0, N as i32 - 1) as usize
}

fn onehot(s: usize) -> Vec<(usize, f32)> {
    (0..N)
        .map(|i| (i, if i == s { 1.0 } else { 0.0 }))
        .collect()
}

/// One ensemble member: `N` INDEPENDENT linear units (max_hidden = 0), unit `s` reads one-hot bit `s`, writes cell
/// `N + s`. All members share bias 0.406 (prior mean ≈ tanh(0.406) ≈ 0.385, a touch below the peaks) and differ
/// only in weight `w` (the disagreement). INDEPENDENT units ⇒ visiting state `s` converges its estimate WITHOUT
/// dragging the unvisited states — their disagreement (the epistemic σ²) is preserved for the frontier.
fn model(w: f32) -> Field {
    let mut f = Field::new(N);
    f.ensure_cells(2 * N);
    f.config_mut().max_hidden = 0;
    for s in 0..N {
        f.add_unit(vec![s], vec![w], 0.406, N + s);
    }
    f
}

/// Varied-prior ensemble with ANTITHETIC weight pairs (`+w`, `−w`): the mean is pinned near tanh(0.406) on every
/// seed, only the spread (the epistemic variance) varies — exactly `substrate_free_energy_actor`'s construction.
fn ensemble(seed: u64) -> Vec<Field> {
    let mut r = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xF00D);
    r.u();
    let mut ens = Vec::with_capacity(ENSEMBLE);
    for _ in 0..ENSEMBLE / 2 {
        let w = r.sym() * 1.7;
        ens.push(model(w));
        ens.push(model(-w));
    }
    ens
}

fn pred(f: &mut Field, s: usize) -> f32 {
    f.stimulate(&onehot(s), 20);
    f.cells()[N + s]
}

/// Grow every ensemble member toward the observed height at state `s` (the model is GROWN online, never given).
fn observe_state(ens: &mut [Field], s: usize) {
    for f in ens.iter_mut() {
        for _ in 0..3 {
            f.grow(&onehot(s), N + s, H[s], 0.2, 20);
        }
    }
}

/// "PRIOR LIFE" — a brief identical random walk over the KNOWN region (states 0..=KNOWN_END) that grows every
/// actor's model there (σ² → 0 in the near + valley band). The far ascent (10..N) is NEVER visited ⇒ its σ² stays
/// broad = the unknown goal location. This defines the both-hard INSTANCE; it is identical across all actors.
fn prior_life(ens: &mut [Field], seed: u64) {
    let mut r = Rng::new(seed ^ 0xB1A5);
    let mut s: i32 = START as i32;
    for _ in 0..600 {
        observe_state(ens, s as usize);
        let d = DELTAS[r.below(2)];
        s = (s + d).clamp(0, KNOWN_END as i32);
    }
}

/// The ensemble's mean and variance of the predicted observation at each state — μ(s) and the epistemic σ²(s).
fn belief(ens: &mut [Field]) -> (Vec<f32>, Vec<f32>) {
    let (mut mu, mut var) = (vec![0.0f32; N], vec![0.0f32; N]);
    for (s, (m, v)) in mu.iter_mut().zip(var.iter_mut()).enumerate() {
        let os: Vec<f32> = ens.iter_mut().map(|f| pred(f, s)).collect();
        let n = os.len() as f32;
        *m = os.iter().sum::<f32>() / n;
        *v = os.iter().map(|o| (o - *m) * (o - *m)).sum::<f32>() / n;
    }
    (mu, var)
}

/// LEARN the action-conditional transition model (predict the move-delta from the action — agency). `with_action`
/// gates whether the model even SEES the action (the scramble ablation). Cells 0,1 = action ±1 one-hot; cell 2 =
/// predicted delta. Linear (max_hidden = 0). Same construction as `substrate_free_energy_policy`.
fn learn_move(with_action: bool, seed: u64) -> Field {
    let mut mv = Field::new(2);
    mv.config_mut().max_hidden = 0;
    mv.ensure_cells(3);
    let mut r = Rng::new(seed);
    for _ in 0..8000 {
        let a = r.below(2);
        let inp: Vec<(usize, f32)> = (0..2)
            .map(|i| (i, if with_action && i == a { 1.0 } else { -1.0 }))
            .collect();
        mv.grow(&inp, 2, DELTAS[a] as f32, 0.1, 20);
    }
    mv
}

fn pred_delta(mv: &mut Field, a: usize) -> i32 {
    let inp: Vec<(usize, f32)> = (0..2)
        .map(|i| (i, if i == a { 1.0 } else { -1.0 }))
        .collect();
    mv.stimulate(&inp, 20);
    mv.cells()[2].round() as i32
}

/// Score a trajectory INTRINSICALLY: did it REACH the global preferred state, and its terminal dwell mean|h − s*|
/// over the last `N` steps (0 = sitting at the global optimum).
fn score(traj: &[usize]) -> (bool, f32) {
    let reached = traj.contains(&(N - 1));
    let tail = &traj[traj.len() - N..];
    let dwell = tail.iter().map(|&s| (H[s] - SSTAR).abs()).sum::<f32>() / N as f32;
    (reached, dwell)
}

/// THE UNIFIED DYNA ACTOR (and its pure-plan ablation). Each step: grow the model at the current state (interleave),
/// recompute the belief, SELECT the goal by the free-energy functional `g(s) = (μ−s*)² − epi·σ²` (epi=1 unified,
/// epi=0 pure-plan), `relax_values` the cost-to-go over the LEARNED transitions, and descend it. `epi` is the ONLY
/// thing separating pure-plan from unified — the epistemic term.
fn dyna(ens: &mut [Field], mv: &mut Field, epi: f32) -> Vec<usize> {
    let learned: Vec<i32> = (0..2).map(|a| pred_delta(mv, a)).collect();
    let adj: Vec<Vec<usize>> = (0..N)
        .map(|s| learned.iter().map(|&d| clamp_s(s as i32 + d)).collect())
        .collect();
    let mut planner = Field::new(N);
    let mut s: i32 = START as i32;
    let mut traj = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        observe_state(ens, s as usize);
        let (mu, var) = belief(ens);
        // free-energy goal: the state minimising (μ−s*)² − epi·σ² (pragmatic − epistemic, coefficient one).
        let goal = (0..N)
            .min_by(|&x, &y| {
                let gx = (mu[x] - SSTAR).powi(2) - epi * var[x];
                let gy = (mu[y] - SSTAR).powi(2) - epi * var[y];
                gx.total_cmp(&gy)
            })
            .unwrap();
        planner.clear_values();
        planner.prefer(goal, 0.0);
        planner.relax_values(&adj, 1.0, 0); // value-iterate the cost-to-go over the IMAGINED dynamics
        let a = (0..2)
            .min_by(|&x, &y| {
                let vx = planner.value_of(clamp_s(s + learned[x]));
                let vy = planner.value_of(clamp_s(s + learned[y]));
                vx.total_cmp(&vy)
            })
            .unwrap();
        s = clamp_s(s + DELTAS[a]) as i32; // the REAL world applies the true delta
        traj.push(s as usize);
    }
    traj
}

/// THE ONE-STEP FREE-ENERGY ACTOR (pure-explore, the `substrate_free_energy_actor` form). Grow the model at the
/// current state, then pick the action whose IMAGINED next state minimises `g(s') = (μ(s')−s*)² − σ²(s')` — a
/// ONE-STEP horizon, no `relax_values`. It climbs the local peak, then traps: the far frontier's σ² is not one
/// step away (the adjacent steps are the KNOWN valley).
fn one_step(ens: &mut [Field], mv: &mut Field) -> Vec<usize> {
    let learned: Vec<i32> = (0..2).map(|a| pred_delta(mv, a)).collect();
    let mut s: i32 = START as i32;
    let mut traj = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        observe_state(ens, s as usize);
        let (mu, var) = belief(ens);
        let a = (0..2)
            .min_by(|&x, &y| {
                let sx = clamp_s(s + learned[x]);
                let sy = clamp_s(s + learned[y]);
                let gx = (mu[sx] - SSTAR).powi(2) - var[sx];
                let gy = (mu[sy] - SSTAR).powi(2) - var[sy];
                gx.total_cmp(&gy)
            })
            .unwrap();
        s = clamp_s(s + DELTAS[a]) as i32;
        traj.push(s as usize);
    }
    traj
}

#[test]
fn dyna_interleave_explores_to_find_and_plans_to_reach_in_one_loop() {
    let seeds = 8u64;

    // ── (A) AGENCY — the transition model needs the action to predict the move (else planning has no model). ──
    let mut mv_with = learn_move(true, 0x9);
    let mut mv_blind = learn_move(false, 0x9);
    let acc = |mv: &mut Field| -> f32 {
        let ok = (0..2).filter(|&a| pred_delta(mv, a) == DELTAS[a]).count();
        100.0 * ok as f32 / 2.0
    };
    let (with_acc, blind_acc) = (acc(&mut mv_with), acc(&mut mv_blind));
    eprintln!(
        "[EFE-DYNA] (A) action-conditional transition model: WITH-action {with_acc:.0}% vs action-BLIND {blind_acc:.0}%"
    );
    assert!(
        with_acc >= 100.0,
        "the model must learn action→move (agency): WITH-action {with_acc:.0}%"
    );
    assert!(
        blind_acc <= 50.0,
        "an action-blind model can't distinguish the moves (chance): {blind_acc:.0}%"
    );

    // ── (B) the both-hard world — pure-explore TRAPS, pure-plan can't FIND, unified DOES BOTH, scramble COLLAPSES. ──
    let (mut expl_reach, mut plan_reach, mut uni_reach, mut scram_reach) = (0, 0, 0, 0);
    let (mut expl_d, mut plan_d, mut uni_d, mut scram_d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for seed in 0..seeds {
        let sd = 0x100 + seed;
        let mut mv = learn_move(true, sd);
        let mut mvb = learn_move(false, sd);

        // each actor gets its OWN copy of the SAME prior-life-primed ensemble (identical instance across actors).
        let make_ens = || {
            let mut e = ensemble(sd);
            prior_life(&mut e, sd);
            e
        };
        let (mut e_expl, mut e_plan, mut e_uni, mut e_scr) =
            (make_ens(), make_ens(), make_ens(), make_ens());

        let (er, ed) = score(&one_step(&mut e_expl, &mut mv));
        let (pr, pd) = score(&dyna(&mut e_plan, &mut mv, 0.0)); // pure-plan: no epistemic term
        let (ur, ud) = score(&dyna(&mut e_uni, &mut mv, 1.0)); // unified: pragmatic − epistemic
        let (sr, sd2) = score(&dyna(&mut e_scr, &mut mvb, 1.0)); // unified over an action-blind model

        expl_reach += er as usize;
        plan_reach += pr as usize;
        uni_reach += ur as usize;
        scram_reach += sr as usize;
        expl_d += ed;
        plan_d += pd;
        uni_d += ud;
        scram_d += sd2;
    }
    let sf = seeds as f32;
    let (expl_d, plan_d, uni_d, scram_d) = (expl_d / sf, plan_d / sf, uni_d / sf, scram_d / sf);

    eprintln!(
        "[EFE-DYNA] (B) UNKNOWN goal ACROSS a valley — reached the global preferred state / {seeds} seeds, terminal dwell mean|h−s*| (local-peak trap ≈ {:.2}, global peak = 0.00):",
        (H[2] - SSTAR).abs()
    );
    eprintln!(
        "[EFE-DYNA]   pure-explore (one-step g=(μ−s*)²−σ²) : reached {expl_reach}/{seeds}  dwell {expl_d:.3}   <- climbs the local peak, TRAPS (far σ² invisible one step away)"
    );
    eprintln!(
        "[EFE-DYNA]   pure-plan    (relax_values, no −σ²)  : reached {plan_reach}/{seeds}  dwell {plan_d:.3}   <- plans to the best KNOWN state, never ventures ⇒ never FINDS"
    );
    eprintln!(
        "[EFE-DYNA]   unified Dyna (relax_values, −σ² goal): reached {uni_reach}/{seeds}  dwell {uni_d:.3}   <- explores to the frontier AND plans across the valley"
    );
    eprintln!(
        "[EFE-DYNA]   scramble     (action-blind model)   : reached {scram_reach}/{seeds}  dwell {scram_d:.3}   <- no valid model ⇒ planning collapses"
    );
    eprintln!(
        "[EFE-DYNA] VERDICT: ONE Dyna loop over the ONE functional G(π)=Σ[(μ−s*)²−σ²]. pure-explore (one-step) climbs the local peak but CAN'T reach the unknown far goal — its epistemic value is not one step away ({expl_reach}/{seeds}, dwell {expl_d:.2}); pure-plan reaches the best KNOWN state and, with no −σ² drive, never discovers the far goal ({plan_reach}/{seeds}, dwell {plan_d:.2}); the UNIFIED actor's −σ² goal makes the un-mapped far region OUTSCORE the local peak and relax_values PLANS across the valley to it, discovering the global optimum ({uni_reach}/{seeds}, dwell {uni_d:.2}). pure-plan→unified differ ONLY by the epistemic term; pure-explore→unified ONLY by relax_values — so this isolates BOTH halves of the unification. Truth-not-trophy: an action-blind model collapses it ({scram_reach}/{seeds}, dwell {scram_d:.2}) — the learned model IS the load. No ε/β/TD/novelty; explore self-anneals as σ→0."
    );

    // pure-explore climbs the local-preference direction but TRAPS — never reaching the unknown global peak
    assert_eq!(
        expl_reach, 0,
        "pure-explore (one-step) must TRAP at the local peak, never reaching the unknown global goal ({expl_reach}/{seeds})"
    );
    assert!(
        expl_d > 0.25,
        "the trapped one-step explorer must dwell far from the preferred observation ({expl_d:.3})"
    );
    // pure-plan reaches the best KNOWN state but never DISCOVERS the far goal
    assert_eq!(
        plan_reach, 0,
        "pure-plan (no epistemic drive) must never discover the unknown global goal ({plan_reach}/{seeds})"
    );
    assert!(
        plan_d > 0.25,
        "pure-plan must dwell at the local peak, far from the preferred observation ({plan_d:.3})"
    );
    // the unified actor does BOTH — discovers the goal AND crosses the valley, every (or nearly every) seed
    assert!(
        uni_reach >= seeds as usize * 7 / 8,
        "the unified actor must discover AND reach the global goal (reached {uni_reach}/{seeds})"
    );
    assert!(
        uni_d < 0.12,
        "the unified actor must settle AT the preferred observation ({uni_d:.3})"
    );
    // both halves are load-bearing: unified beats each one-sided baseline decisively
    assert!(
        uni_d < expl_d * 0.5 && uni_d < plan_d * 0.5,
        "unified (explore AND plan) must beat both one-sided baselines ({uni_d:.3} vs explore {expl_d:.3}, plan {plan_d:.3})"
    );
    // truth-not-trophy: the learned model is the load — a scrambled (action-blind) model collapses the actor
    assert_eq!(
        scram_reach, 0,
        "a scrambled (action-blind) model must collapse the actor ({scram_reach}/{seeds})"
    );
    assert!(
        scram_d > 0.25,
        "the scrambled actor must dwell far from the preferred observation ({scram_d:.3})"
    );
}
