// SPDX-License-Identifier: MIT-0
//! EMERGENT PHASE SWITCH — one free-energy actor explores-below-reward and exploits-above with NO authored gate
//! (`docs/research/unified-free-energy-actor.md`; MorphBot's UNIFIED_RELATIONAL_PLANNER "two phases of one system").
//!
//! ## The wall (MorphBot, real ARC-3, 2026-08-15) — Seam B, goal-discovery BELOW the first reward
//! The relational representation + effect-action solved the STATE/ACTION seam, but the unified planner still scored
//! 0 on real games: below level-1 there is no reward, so `infer_goal` has nothing to latch onto and a fixed
//! "approach the salient object" prior does not match the game's unknown win-mechanic. The empirical fix was a
//! TWO-PHASE system — explore-to-discover-the-goal below the first reward, exploit the inferred relation above it —
//! but wired with an AUTHORED gate ("the coverage probe fires only below the first reward").
//!
//! ## The claim (nothing authored): the phase switch is EMERGENT, not gated
//! The free-energy functional `g(s) = (μ(s) − s*)² − σ²(s)` already IS the two-phase system, if the preferred
//! observation `s*` is treated as a PRIOR with a PRECISION that starts at ZERO (a flat / uninformative preference,
//! the principled active-inference starting point) and is set only by the first reward EVENT (`infer_goal ⇒
//! prefer`). With precision 0 the pragmatic term `(μ−s*)²` contributes NOTHING to the goal `argmin g`, so the goal
//! is `argmin(−σ²) = argmax σ²` — the epistemic frontier ⇒ EXPLORE. The reward event sets `s*` (precision 1) ⇒ the
//! pragmatic term activates ⇒ the goal becomes the preferred state ⇒ EXPLOIT. SAME formula every step; the ONLY
//! variable is whether `prefer` has been informed — driven by the reward, not by a step counter.
//!
//! ## The test — a world whose win-mechanic the "approach" prior does NOT match
//! A corridor of `N` states; observation height `h(s)` has a LOCAL peak (the salient DECOY the approach prior
//! points at, s=2, h 0.52) and the true REWARD state a far NEEDLE (s=15, h 0.90) reachable only by crossing a long
//! low plain. Reward fires only AT the needle. Four actors share ONE prior-life-primed ensemble instance:
//!   - REACTIVE-APPROACH — `s*` fixed to the decoy height (the "approach salient" prior); plans to the decoy, never explores ⇒ never reaches reward (the real-games failure).
//!   - AUTHORED TWO-PHASE — explicit gate `if !reward_seen { goal = argmax σ² } else { goal = argmin(μ−s*)² }`; discovers then exploits (MorphBot's current design).
//!   - UNIFIED (NO GATE) — one `g(s) = (μ−s*)² − σ²` with `s*` unset until the reward event; below reward the flat prior makes the pragmatic term vanish ⇒ `−σ²` drives coverage; the needle's reward sets `s*` ⇒ exploit.
//!   - ABLATION (flat `s*`, `−σ²` REMOVED) — no drive at all ⇒ never reaches reward (isolates the epistemic term as the below-reward explorer).
//!
//! The world/ensemble/prior-life/transition-model are authored and IDENTICAL across actors; only the goal rule
//! differs. `relax_values` gives the planning depth to reach the chosen goal across the plain (the same as
//! `substrate_free_energy_dyna`); this test isolates the EMERGENCE of the explore→exploit switch.
//!
//! THE CLAIM ASSERTED. UNIFIED (no gate) MATCHES AUTHORED-two-phase on reach + dwell, and both ≫ reactive and
//! ablation. So the authored below/above-reward gate is DELETABLE — a fix that removes machinery. Zero core change:
//! a usage protocol over `grow`/`prefer`/`relax_values`/`value_of` + the ensemble `σ²` (the validated epistemic
//! currency, `substrate_voi` — NOT a count/world-surprise bonus, which MorphBot measured HURTS real ARC).
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_discover -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N: usize = 16;
const ENSEMBLE: usize = 24;
const KNOWN_END: usize = 9; // prior life maps the near band + plain; the far ascent stays terra incognita
const START: usize = 0;
const STEPS: usize = 96;
const REWARD: usize = 15; // the true win state (the needle) — its LOCATION is unknown until explored
const DECOY: usize = 2; // the salient local peak the "approach" prior fixates on

/// Observation height: a LOCAL decoy peak at s=2 (0.52), a long low plain, the REWARD needle at s=15 (0.90).
const H: [f32; N] = [
    0.30, 0.45, 0.52, 0.38, 0.24, 0.14, 0.09, 0.07, 0.07, 0.08, 0.10, 0.11, 0.10, 0.12, 0.11, 0.90,
];

const DELTAS: [i32; 2] = [-1, 1];

fn clamp_s(s: i32) -> usize {
    s.clamp(0, N as i32 - 1) as usize
}

fn onehot(s: usize) -> Vec<(usize, f32)> {
    (0..N).map(|i| (i, (i == s) as u8 as f32)).collect()
}

/// One ensemble member: N independent linear units, member differs only in weight (the disagreement = σ²).
fn model(w: f32) -> Field {
    let mut f = Field::new(N);
    f.ensure_cells(2 * N);
    f.config_mut().max_hidden = 0;
    for s in 0..N {
        f.add_unit(vec![s], vec![w], 0.406, N + s);
    }
    f
}

/// Antithetic varied-prior ensemble: mean pinned, only the spread (epistemic σ²) varies.
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

fn observe_state(ens: &mut [Field], s: usize) {
    for f in ens.iter_mut() {
        for _ in 0..3 {
            f.grow(&onehot(s), N + s, H[s], 0.2, 20);
        }
    }
}

/// Prior life — an identical random walk over the KNOWN band (0..=KNOWN_END); the far ascent (10..N incl. the
/// reward needle) is NEVER visited, so its σ² stays broad = the unknown goal location.
fn prior_life(ens: &mut [Field], seed: u64) {
    let mut r = Rng::new(seed ^ 0xB1A5);
    let mut s: i32 = START as i32;
    for _ in 0..600 {
        observe_state(ens, s as usize);
        s = (s + DELTAS[r.below(2)]).clamp(0, KNOWN_END as i32);
    }
}

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

/// Action-conditional transition model (predict the move-delta from the action) — the same agency model as
/// `substrate_free_energy_dyna`, learned online, never given.
fn learn_move(seed: u64) -> Field {
    let mut mv = Field::new(2);
    mv.config_mut().max_hidden = 0;
    mv.ensure_cells(3);
    let mut r = Rng::new(seed);
    for _ in 0..8000 {
        let a = r.below(2);
        let inp: Vec<(usize, f32)> = (0..2)
            .map(|i| (i, if i == a { 1.0 } else { -1.0 }))
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

/// How the goal is chosen this step.
#[derive(Clone, Copy)]
enum Mode {
    /// `s*` fixed to the decoy height from the start (the "approach the salient object" prior).
    ReactiveApproach,
    /// Explicit gate: explore (argmax σ²) below reward, exploit (argmin (μ−s*)²) above it.
    AuthoredTwoPhase,
    /// One formula `g = (μ−s*)² − σ²`; `s*` unset (flat prior, precision 0) until the reward event.
    UnifiedNoGate,
    /// Flat `s*` AND the `−σ²` term removed — no drive (isolates the epistemic explorer).
    AblationNoEpistemic,
}

/// Run one actor to convergence; return `(reached_reward, terminal_dwell)` where dwell = mean |h − h(REWARD)| over
/// the last N steps (0 = sitting on the reward needle).
fn run(ens: &mut [Field], mv: &mut Field, mode: Mode) -> (bool, f32) {
    let learned: Vec<i32> = (0..2).map(|a| pred_delta(mv, a)).collect();
    let adj: Vec<Vec<usize>> = (0..N)
        .map(|s| learned.iter().map(|&d| clamp_s(s as i32 + d)).collect())
        .collect();
    let mut planner = Field::new(N);
    let mut s: i32 = START as i32;
    let mut traj = Vec::with_capacity(STEPS);
    // The preferred observation s*: None = flat prior (precision 0). The reward EVENT sets it (infer_goal ⇒ prefer).
    let mut sstar: Option<f32> = match mode {
        Mode::ReactiveApproach => Some(H[DECOY]), // approach prior: prefer the salient decoy, fixed
        _ => None,
    };

    for _ in 0..STEPS {
        observe_state(ens, s as usize);
        let (mu, var) = belief(ens);

        // The goal state, by mode. The UNIFIED formula g(s) = pragmatic(s) − σ²(s) with pragmatic(s) = 0 when s* is
        // unset (flat prior contributes nothing to argmin) — the emergent switch.
        let pragmatic = |x: usize| match sstar {
            Some(v) => (mu[x] - v).powi(2),
            None => 0.0,
        };
        let goal = match mode {
            Mode::AuthoredTwoPhase => {
                if sstar.is_none() {
                    (0..N).max_by(|&a, &b| var[a].total_cmp(&var[b])).unwrap() // explore: frontier
                } else {
                    (0..N)
                        .min_by(|&a, &b| pragmatic(a).total_cmp(&pragmatic(b)))
                        .unwrap() // exploit
                }
            }
            // Pure exploiters — pragmatic ONLY, no epistemic term. ReactiveApproach fixates on the decoy (wrong
            // fixed s*); AblationNoEpistemic has a flat s* so it is indifferent (stays). Both model "exploit with
            // no discovery drive" — the real-games reactive failure.
            Mode::ReactiveApproach | Mode::AblationNoEpistemic => (0..N)
                .min_by(|&a, &b| pragmatic(a).total_cmp(&pragmatic(b)))
                .unwrap(),
            // UnifiedNoGate: the ONE functional g = pragmatic − σ² (flat s* ⇒ pure −σ² below reward).
            Mode::UnifiedNoGate => (0..N)
                .min_by(|&a, &b| (pragmatic(a) - var[a]).total_cmp(&(pragmatic(b) - var[b])))
                .unwrap(),
        };

        planner.clear_values();
        planner.prefer(goal, 0.0);
        planner.relax_values(&adj, 1.0, 0);
        let a = (0..2)
            .min_by(|&x, &y| {
                planner
                    .value_of(clamp_s(s + learned[x]))
                    .total_cmp(&planner.value_of(clamp_s(s + learned[y])))
            })
            .unwrap();
        s = clamp_s(s + DELTAS[a]) as i32;
        traj.push(s as usize);

        // The reward EVENT: reaching the reward state informs the preferred prior (infer_goal ⇒ prefer). No step
        // counter — the reward, not the clock, flips exploration to exploitation. Reactive never updates s*.
        if s as usize == REWARD && sstar.is_none() {
            sstar = Some(H[REWARD]);
        }
    }

    let reached = traj.contains(&REWARD);
    let tail = &traj[traj.len() - N..];
    let dwell = tail.iter().map(|&x| (H[x] - H[REWARD]).abs()).sum::<f32>() / N as f32;
    (reached, dwell)
}

#[test]
fn free_energy_actor_switches_explore_to_exploit_with_no_authored_gate() {
    let seeds = 8u64;
    let (mut react_r, mut auth_r, mut uni_r, mut abl_r) = (0, 0, 0, 0);
    let (mut auth_d, mut uni_d) = (0.0f32, 0.0f32);

    for seed in 0..seeds {
        let sd = 0x100 + seed;
        let mut mv = learn_move(sd);
        let make_ens = || {
            let mut e = ensemble(sd);
            prior_life(&mut e, sd);
            e
        };
        let (r_react, _) = run(&mut make_ens(), &mut mv, Mode::ReactiveApproach);
        let (r_auth, d_auth) = run(&mut make_ens(), &mut mv, Mode::AuthoredTwoPhase);
        let (r_uni, d_uni) = run(&mut make_ens(), &mut mv, Mode::UnifiedNoGate);
        let (r_abl, _) = run(&mut make_ens(), &mut mv, Mode::AblationNoEpistemic);

        react_r += r_react as u32;
        auth_r += r_auth as u32;
        uni_r += r_uni as u32;
        abl_r += r_abl as u32;
        auth_d += d_auth;
        uni_d += d_uni;
    }
    let sf = seeds as f32;
    println!("\n  EMERGENT PHASE SWITCH (explore→exploit, one free-energy actor, {seeds} seeds):");
    println!("    reached reward   : reactive-approach {react_r}/{seeds}   authored-2phase {auth_r}/{seeds}   UNIFIED-no-gate {uni_r}/{seeds}   ablation(no −σ²) {abl_r}/{seeds}");
    println!(
        "    terminal dwell   : authored {:.3}   unified {:.3}   (0 = sits on the reward needle)",
        auth_d / sf,
        uni_d / sf
    );
    println!("  ⇒ the unified actor (flat s* until the reward event) reproduces the AUTHORED two-phase gate with");
    println!("    NO below/above-reward branch — the −σ² term explores, the reward sets s*, exploitation follows.\n");

    // (1) UNIFIED (no gate) discovers + exploits — reaches the reward needle across the plain, like the authored gate.
    assert!(
        uni_r >= 7,
        "unified (no gate) must reach the reward on ≥7/8 seeds: {uni_r}/{seeds}"
    );
    // (2) It MATCHES the authored two-phase gate (the gate is deletable) — reach parity and comparable dwell.
    assert!(
        uni_r >= auth_r && (uni_d / sf) <= (auth_d / sf) + 0.05,
        "unified must match authored two-phase: reach {uni_r} vs {auth_r}, dwell {:.3} vs {:.3}",
        uni_d / sf,
        auth_d / sf
    );
    // (3) The APPROACH prior fails (fixates on the decoy — the real-games failure) and the epistemic term is the
    // below-reward driver (ablation without −σ² never reaches).
    assert!(
        react_r <= 1 && abl_r <= 1,
        "reactive-approach and the no-epistemic ablation must fail to reach reward: reactive {react_r}, ablation {abl_r}"
    );
    // (4) Unified sits ON the reward once found (exploitation is real, not a fly-by).
    assert!(
        uni_d / sf < 0.1,
        "unified must DWELL on the reward needle (exploit): dwell {:.3}",
        uni_d / sf
    );
}
