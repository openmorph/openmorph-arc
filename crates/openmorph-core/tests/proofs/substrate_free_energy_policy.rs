// SPDX-License-Identifier: MIT-0
//! MULTI-STEP FREE ENERGY — the SAME free-energy functional, now over POLICIES (multi-step rollouts) instead of
//! single actions. `substrate_free_energy_actor` proved a ONE-STEP actor minimising `G(a) = (μ_a − s*)² − σ_a²`
//! (pragmatic − epistemic, ONE currency, coefficient ONE) beats a value-greedy baseline on a deceptive BANDIT,
//! deleting ε/β/TD/novelty. Its DOCUMENTED EDGE: a one-step actor CANNOT cross a VALLEY — to reach a distant
//! global optimum it must temporarily move AWAY from the preferred observation, and the one-step pragmatic term
//! `(μ − s*)²` forbids exactly that. This proof closes that gap WITHOUT re-introducing any knob.
//!
//! THE THESIS. Expected free energy over a POLICY π (a multi-step rollout `a₁,a₂,…` under the organism's learned
//! world model) is just the SAME functional summed along the imagined trajectory:
//!
//!   G(π) = Σ_t [ (μ_t − s*)²  −  σ_t² ]      (pragmatic-to-reach-s* propagated along the rollout − epistemic)
//!
//! Minimising `G(π)` over policies is DYNAMIC PROGRAMMING toward the preferred state — and the substrate already
//! has it: `relax_values` value-iterates the cost-to-reach a clamped preferred cell over the organism's OWN
//! learned transition model (`substrate_planner`). The pragmatic term is no longer the one-step `(μ − s*)²`; it
//! becomes the RELAXED cost-to-go, which — being a value field with NO interior local minima — descends across a
//! valley the one-step term traps on. The epistemic `−σ²` term is unchanged (it still drives exploration of
//! un-mapped states, `substrate_free_energy_actor`); here the model is already learned (σ→0, the doc's
//! self-annealing), so what remains to isolate is the PLANNING-DEPTH half. STILL no ε/β/TD/novelty — DEPTH is
//! what crosses the valley.
//!
//! THE WORLD — a VALLEY corridor (the discriminator that forces the point). States `0..N` on a line; observation
//! `h(s)` is a DOUBLE HUMP: a LOCAL peak near the start, a deep VALLEY, then the GLOBAL peak `≈ s*` at the far
//! end. The organism starts on the local-peak side. To reach `s*` it must descend INTO the valley (lower `h`,
//! FARTHER from the preferred observation) before climbing the far side.
//!
//!   * ONE-STEP actor (Route-1 form, the honest baseline that FAILS) — pick the action whose imagined next
//!     observation is closest to `s*` (`argmin (μ(s') − s*)²` over the SAME learned model). It climbs to the
//!     local peak and TRAPS: every step off it raises `(μ − s*)²`. It NEVER reaches the global optimum.
//!   * MULTI-STEP actor (succeeds) — identify the preferred STATE from `s*` (the state whose learned observation
//!     is closest to `s*`), clamp it (`prefer`), `relax_values` the cost-to-go over the learned transitions, and
//!     descend. The relaxed field has no interior minimum, so it walks THROUGH the valley to the global optimum.
//!   * SCRAMBLE control (truth-not-trophy) — the SAME multi-step actor over an action-BLIND transition model (it
//!     can't tell the actions apart). No valid model ⇒ the imagined rollout is meaningless ⇒ planning COLLAPSES.
//!     The learned MODEL is the load (the `substrate_free_energy_actor` / `substrate_purpose` inside-check).
//!
//! Scored INTRINSICALLY — did the actor REACH the global preferred state, and its terminal dwell `|h − s*|`
//! (a life metric, never an external label); the world model is GROWN online (`grow`), never given. The one-step
//! and multi-step actors share the SAME learned models, so a win isolates the ACTION RULE (one-step pragmatic vs
//! multi-step value iteration) and nothing else — exactly like the bandit proof isolates the epistemic term.
//!
//! ZERO core change — a usage protocol over `grow` (the world model) + `prefer`/`relax_values`/`value_of` (the
//! preferred prior + policy-level value iteration), exactly per "the organism is the persistent entity; skills
//! are disposable." Companion doc: `docs/research/unified-free-energy-actor.md` §"Multi-step: EFE over policies".
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_policy -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 13; // states 0..N on a line
const SSTAR: f32 = 0.95; // preferred observation — "how the world should be" (the global peak's height)

/// The VALLEY landscape: observation height `h(s)` per state. A LOCAL peak at s=2 (0.45), a deep VALLEY at s=6
/// (0.05), the GLOBAL peak at s=12 (0.95 ≈ s*). Declared by the world (what "preferred" means), never learned.
/// To get from the local peak to the global one you MUST cross the valley — the one-step pragmatic term forbids it.
const H: [f32; N] = [
    0.20, 0.35, 0.45, 0.35, 0.22, 0.12, 0.05, 0.12, 0.30, 0.50, 0.70, 0.88, 0.95,
];

/// Actions on the corridor: 0 = left (−1), 1 = right (+1). The world clamps at the boundary.
const DELTAS: [i32; 2] = [-1, 1];

fn clamp_s(s: i32) -> usize {
    s.clamp(0, N as i32 - 1) as usize
}

/// LEARN the action-conditional TRANSITION model: predict the move-delta from the action (agency). `with_action`
/// gates whether the model even SEES the action (the scramble ablation — action-blind ⇒ can't tell moves apart).
/// Cells 0,1 = action one-hot (±1); cell 2 = predicted delta. Linear (max_hidden = 0): a one-hot→delta map.
fn learn_move(with_action: bool, seed: u64) -> Field {
    let mut mv = Field::new(2);
    mv.config_mut().max_hidden = 0;
    mv.ensure_cells(3);
    let mut rng = Rng::new(seed);
    for _ in 0..8000 {
        let a = rng.below(2);
        // action as a ±1 one-hot (the planner's agency encoding); action-blind ⇒ all −1 (no distinguishing info).
        let inp: Vec<(usize, f32)> = (0..2)
            .map(|i| (i, if with_action && i == a { 1.0 } else { -1.0 }))
            .collect();
        mv.grow(&inp, 2, DELTAS[a] as f32, 0.1, 20);
    }
    mv
}

/// LEARN the OBSERVATION model `h(s)`: the grown predictor of a state's observation (a one-hot state → its
/// height). This is the world model the organism reads BOTH to identify the preferred STATE from the preferred
/// OBSERVATION `s*` AND to drive the one-step actor's pragmatic term. Grown online by a random walk that visits
/// every state (max_hidden = 0 — one independent readout weight per state). Action-INDEPENDENT (not scrambled).
fn learn_obs(seed: u64) -> Field {
    let mut ob = Field::new(N);
    ob.config_mut().max_hidden = 0;
    ob.ensure_cells(N + 1);
    let mut rng = Rng::new(seed ^ 0xB1A5);
    let mut s: i32 = 0;
    for _ in 0..12000 {
        let sp = clamp_s(s);
        let oh: Vec<(usize, f32)> = (0..N)
            .map(|i| (i, if i == sp { 1.0 } else { 0.0 }))
            .collect();
        ob.grow(&oh, N, H[sp], 0.1, 20); // learn this state's observation
        s = clamp_s(s + DELTAS[rng.below(2)]) as i32; // random walk covers the whole corridor
    }
    ob
}

/// The observation the model PREDICTS for state `s` (its grown estimate of `h(s)`).
fn pred_obs(ob: &mut Field, s: usize) -> f32 {
    let oh: Vec<(usize, f32)> = (0..N)
        .map(|i| (i, if i == s { 1.0 } else { 0.0 }))
        .collect();
    ob.stimulate(&oh, 20);
    ob.cells()[N]
}

/// The transition model's predicted integer delta for action `a`.
fn pred_delta(mv: &mut Field, a: usize) -> i32 {
    let inp: Vec<(usize, f32)> = (0..2)
        .map(|i| (i, if i == a { 1.0 } else { -1.0 }))
        .collect();
    mv.stimulate(&inp, 20);
    mv.cells()[2].round() as i32
}

/// Roll out an actor for `4·N` steps from the start (s=0) and score it: did it REACH the global preferred state,
/// and its terminal dwell `mean |h − s*|` over the last `N` steps (0 = sitting at the global optimum). `pick`
/// chooses the action at each state (the ONLY thing that differs between the actors).
fn roll_out(mut pick: impl FnMut(i32) -> usize) -> (bool, f32) {
    let mut s: i32 = 0;
    let mut traj = Vec::new();
    let mut reached = false;
    for _ in 0..(4 * N) {
        let a = pick(s);
        s = clamp_s(s + DELTAS[a]) as i32; // the REAL world applies the TRUE delta
        if s as usize == N - 1 {
            reached = true;
        }
        traj.push(s as usize);
    }
    let tail = &traj[traj.len() - N..];
    let dwell = tail.iter().map(|&s| (H[s] - SSTAR).abs()).sum::<f32>() / N as f32;
    (reached, dwell)
}

/// The ONE-STEP free-energy actor (Route-1 form). Pick the action whose IMAGINED next observation is closest to
/// `s*`: `argmin (μ(s') − s*)²` over the learned models. (The epistemic `−σ²` term has self-annealed to ~0 now
/// the model is learned — the doc's self-annealing — so `G` reduces to the pragmatic term.) It climbs to the
/// local peak and traps: any step off it raises `(μ − s*)²`.
fn one_step(mv: &mut Field, ob: &mut Field) -> (bool, f32) {
    let learned: Vec<i32> = (0..2).map(|a| pred_delta(mv, a)).collect();
    // pre-read every state's predicted observation once (deterministic, avoids borrow churn in the closure)
    let obs: Vec<f32> = (0..N).map(|s| pred_obs(ob, s)).collect();
    roll_out(|s| {
        (0..2)
            .min_by(|&x, &y| {
                let gx = (obs[clamp_s(s + learned[x])] - SSTAR).powi(2); // one-step pragmatic G, no lookahead
                let gy = (obs[clamp_s(s + learned[y])] - SSTAR).powi(2);
                gx.total_cmp(&gy)
            })
            .unwrap()
    })
}

/// The MULTI-STEP actor — expected free energy over POLICIES. Identify the preferred STATE from `s*` (grown), clamp
/// it, `relax_values` the cost-to-go over the LEARNED transitions (policy-level pragmatic term), and descend the
/// value field. The relaxed field has no interior local minimum, so it crosses the valley the one-step term traps
/// on. `scramble` uses the action-blind transition model — the inside-check that the learned MODEL is the load.
fn multi_step(mv: &mut Field, ob: &mut Field) -> (bool, f32) {
    let learned: Vec<i32> = (0..2).map(|a| pred_delta(mv, a)).collect();
    // GROWN goal identification: the state whose learned observation is closest to the preferred observation s*.
    let goal = (0..N)
        .min_by(|&x, &y| {
            (pred_obs(ob, x) - SSTAR)
                .abs()
                .total_cmp(&(pred_obs(ob, y) - SSTAR).abs())
        })
        .unwrap();
    // imagined transition graph from the LEARNED model: state × action → predicted next state.
    let adj: Vec<Vec<usize>> = (0..N)
        .map(|s| learned.iter().map(|&d| clamp_s(s as i32 + d)).collect())
        .collect();
    let mut planner = Field::new(N);
    planner.prefer(goal, 0.0);
    planner.relax_values(&adj, 1.0, 0); // value-iterate the cost-to-go over the IMAGINED dynamics
                                        // descend the value field in the REAL world (pick the action whose imagined next-state is lowest cost-to-go).
    roll_out(|s| {
        (0..2)
            .min_by(|&x, &y| {
                let vx = planner.value_of(clamp_s(s + learned[x]));
                let vy = planner.value_of(clamp_s(s + learned[y]));
                vx.total_cmp(&vy)
            })
            .unwrap()
    })
}

#[test]
fn multi_step_free_energy_crosses_the_valley_a_one_step_actor_traps_on() {
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
        "[EFE-POLICY] (A) action-conditional transition model: WITH-action {with_acc:.0}% vs action-BLIND {blind_acc:.0}% (the action carries agency)"
    );
    assert!(
        with_acc >= 100.0,
        "the model must learn action→move (agency): WITH-action {with_acc:.0}%"
    );
    assert!(
        blind_acc <= 50.0,
        "an action-blind model can't distinguish the moves (chance): {blind_acc:.0}%"
    );

    // ── (B) the VALLEY — one-step TRAPS, multi-step CROSSES, scramble COLLAPSES. Averaged over seeds. ──
    let (mut one_reach, mut multi_reach, mut scram_reach) = (0usize, 0usize, 0usize);
    let (mut one_dwell, mut multi_dwell, mut scram_dwell) = (0.0f32, 0.0f32, 0.0f32);
    for seed in 0..seeds {
        let mut mv = learn_move(true, 0x100 + seed);
        let mut mvb = learn_move(false, 0x100 + seed);
        let mut ob = learn_obs(0x100 + seed);

        let (o_r, o_d) = one_step(&mut mv, &mut ob);
        let (m_r, m_d) = multi_step(&mut mv, &mut ob);
        let (s_r, s_d) = multi_step(&mut mvb, &mut ob); // SAME multi-step actor, action-BLIND model

        one_reach += o_r as usize;
        multi_reach += m_r as usize;
        scram_reach += s_r as usize;
        one_dwell += o_d;
        multi_dwell += m_d;
        scram_dwell += s_d;
    }
    let sf = seeds as f32;
    let (one_dwell, multi_dwell, scram_dwell) =
        (one_dwell / sf, multi_dwell / sf, scram_dwell / sf);

    eprintln!(
        "[EFE-POLICY] (B) VALLEY corridor — reached the global preferred state / {seeds} seeds, and terminal dwell mean|h−s*| (the local-peak trap sits at |h−s*|≈0.50, the global peak at 0.00):"
    );
    eprintln!(
        "[EFE-POLICY]   one-step  (argmin (μ−s*)²)         : reached {one_reach}/{seeds}  dwell {one_dwell:.3}   <- TRAPS on the local peak (can't descend into the valley)"
    );
    eprintln!(
        "[EFE-POLICY]   multi-step(relax_values to s*)     : reached {multi_reach}/{seeds}  dwell {multi_dwell:.3}   <- value iteration crosses the valley to the global optimum"
    );
    eprintln!(
        "[EFE-POLICY]   scramble  (action-blind model)     : reached {scram_reach}/{seeds}  dwell {scram_dwell:.3}   <- no valid model => planning collapses"
    );
    eprintln!(
        "[EFE-POLICY] VERDICT: the SAME free-energy functional, over POLICIES instead of single actions, crosses a valley the one-step form provably traps on. Pragmatic-to-reach-s* becomes the `relax_values` cost-to-go (no interior local minimum) propagated over the organism's OWN learned transition model; the epistemic term is unchanged (self-annealed to 0 post-learning). The one-step and multi-step actors share the SAME learned models — the ONLY difference is the action rule (one-step pragmatic vs policy value iteration), so the win ({one_reach}/{seeds}→{multi_reach}/{seeds} reached, dwell {one_dwell:.2}→{multi_dwell:.2}) isolates PLANNING DEPTH. Truth-not-trophy: an action-blind model collapses it ({scram_reach}/{seeds}, dwell {scram_dwell:.2}) — the learned model IS the load. No ε, no β, no TD, no novelty; DEPTH crosses the valley."
    );

    // the one-step actor traps on the local peak — it never reaches the global preferred state
    assert_eq!(
        one_reach, 0,
        "the one-step actor must TRAP on the local peak, never reaching the global optimum (reached {one_reach}/{seeds})"
    );
    assert!(
        one_dwell > 0.40,
        "the trapped one-step actor must dwell far from the preferred observation ({one_dwell:.3})"
    );
    // the multi-step actor crosses the valley to the global preferred state, every seed
    assert_eq!(
        multi_reach, seeds as usize,
        "the multi-step actor must cross the valley and reach the global optimum every seed (reached {multi_reach}/{seeds})"
    );
    assert!(
        multi_dwell < 0.05,
        "the multi-step actor must settle AT the preferred observation ({multi_dwell:.3})"
    );
    // policy value iteration is the load, and the model is the load under it — the scramble must collapse
    assert!(
        multi_dwell < one_dwell * 0.25,
        "policy value iteration (the ONLY difference from the one-step actor) must break the trap ({multi_dwell:.3} vs {one_dwell:.3})"
    );
    assert_eq!(
        scram_reach, 0,
        "a scrambled (action-blind) model must collapse the planner — it cannot reach the goal ({scram_reach}/{seeds})"
    );
    assert!(
        scram_dwell > 0.40,
        "the scrambled planner must dwell far from the preferred observation ({scram_dwell:.3})"
    );
}
