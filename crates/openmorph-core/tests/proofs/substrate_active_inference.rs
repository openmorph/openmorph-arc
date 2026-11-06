// SPDX-License-Identifier: MIT-0
//! #171 (OM-10) — the F↔G SEAM on the real substrate: ε-free self-bootstrapping active inference via the
//! native VoI, and the DARK-ROOM control (the sign of the epistemic term is the mechanism). MorphLab measured
//! the unification on a Python surrogate; this executes its shippable clause (D) on the substrate — and finds
//! it needs LESS machinery than the surrogate did.
//!
//! THE UNIFICATION (MorphLab): one signed SURPRISE currency, two functionals — F (perceive/learn: minimize
//! surprise over beliefs) and G (act: minimize expected free energy = pragmatic value − β·epistemic, where the
//! epistemic term SEEKS *resolvable* surprise). The shippable seam is: couple the epistemic weight to F's own
//! residual uncertainty so exploration →0 as the model learns → ε-free self-bootstrap (their clause D, return
//! 0→1.31, no authored exploration constant). The dark-room finding: the epistemic term must be over the
//! REWARD-RELEVANT belief (value uncertainty), NOT raw world-surprise — else the agent camps in a noisy corner.
//!
//! WHERE THE SUBSTRATE IS SIMPLER (the review the owner asked for — scoped honestly to THIS seam): MorphLab's
//! clause-D reads β live from F's residual (an `esc` coupling formula) to make exploration self-anneal. The
//! substrate's `select_explore` scores `mean + β·spread`, and `spread` — the DISAGREEMENT of a bootstrap
//! ensemble of grown value-estimators — IS F's residual uncertainty about VALUE, which self-vanishes as the
//! ensemble converges. So β·spread self-anneals with a FIXED β: no coupling formula, no schedule, no ε-seed. And
//! because the epistemic term is over VALUE (not raw observation), the dark room is avoided by construction.
//! CAVEAT (do not overclaim): this seam has NO hidden cause and NO reward delay, so it does NOT test or replace
//! MorphLab's recursive belief filter / pragmatic-C — the harder regime that motivated them. It is the SELECTION
//! seam only.
//!
//! THE WORLD (deterministic, seeded): K repeated choices. WIN pays +1; the rest pay 0. One option is a NOISE /
//! DARK-ROOM trap: it pays 0 but emits a RANDOM observation, so a grown world-model can never predict it → its
//! world-surprise is IRREDUCIBLE. Value must be GROWN from the reward stream (no authored values).
//!
//! WHAT IS ASSERTED (score INSIDE, controls are required failures — this proof was REBUILT after an adversarial
//! verify caught the first cut's dark-room baseline as an inert strawman; the baseline is now a genuine explorer):
//!   BOOTSTRAP — VoI (β>0 over the value ensemble) finds and exploits WIN with NO ε-seed; reward ≫ greedy (β=0),
//!       which freezes on a zero-reward option it has no reason to leave.
//!   DARK-ROOM (the load-bearing SUBJECT control, WITH TEETH) — an EQUALLY-exploratory curiosity agent (optimistic
//!       world-surprise, same β) explores the same options but is LURED into the noise corner: it FLEES WIN once
//!       WIN becomes predictable and CAMPS on the option whose surprise is IRREDUCIBLE, earning LESS than greedy.
//!       VoI does not, because value-uncertainty is RESOLVABLE. Measured by LATE NOISE-pull fraction: world-
//!       surprise ≈1.0 (camps) vs VoI ≈0 (leaves). The SUBJECT (resolvable value vs irreducible world) is the mechanism.
//!   SELF-ANNEAL — VoI's exploration DRIVE decays: non-WIN pull fraction early≫late with a FIXED β (not the
//!       winner's convergence — the drive itself). Clause D for free.
//!   GROWN VALUE — WIN's ensemble mean moves from its random prior to ≈+0.9 (learned from reward), and WIN was
//!       actually visited ≫ (discovered, not authored).
//!
//! HONEST SCOPE: β=1.5→2.5 is a FIXED epistemic weight (meta-evolvable like `epistemic_weight`); "ε-free" means no
//! exploration SCHEDULE/seed, NOT a free β. Synthetic bandit selection seam; the grown epistemic-SEARCH policy
//! (the real frontier) and the real-data value arm (blocked on a reward-bearing rollout) stay open. Deterministic.
//!
//! CRITICAL CAVEAT (measured by MorphLab on REAL ARC-3, 2026-07-28): the SELF-ANNEALING celebrated here is
//! optimal for a STATIONARY bandit (one reward, find-and-exploit) but is exactly the EXPLOIT-LOCK that LOST to
//! random on 252 real ARC-3 games (EFE 6 wins / 0.33 levels vs random 9 / 0.61). On a SEQUENTIAL task where
//! progress needs sustained COVERAGE, collapsing exploration on the first foothold is death — random keeps
//! covering the action space and wins. So this proof shows the decide/VoI SEAM works in its regime, NOT that it
//! lifts ARC-3; the load-bearing piece is the grown epistemic-SEARCH (coverage that does NOT exploit-lock) +
//! grown perception (an aliased state key falsely converges the value ⇒ SPURIOUS annealing). Do not read the
//! self-anneal as a win for real games — it is the failure mode there.

use openmorph_core::substrate::Field;

use crate::common::Rng;

const K: usize = 5; // options
const WIN: usize = 0; // pays +1
const NOISE: usize = 4; // pays 0 but emits a random observation (the dark-room trap)
const E: usize = 6; // ensemble value-estimators per option
const HORIZON: usize = 500;
const BUD: usize = 60;

fn val_cell(a: usize, e: usize) -> usize {
    K + a * E + e
}
fn pred_cell(a: usize) -> usize {
    K + K * E + a
}

/// Build a Field: per option, E value-estimators (varied bootstrap priors) + one world-model predictor of the
/// option's observation. All read the option's arm cell.
fn build(seed: u64) -> Field {
    let mut f = Field::new(K + K * E + K);
    let mut r = Rng::new(seed);
    for a in 0..K {
        for e in 0..E {
            let w = (r.f() - 0.5) * 1.4; // varied prior ⇒ disagreement before any data
            f.add_unit(vec![a], vec![w], 0.0, val_cell(a, e));
        }
        f.add_unit(vec![a], vec![0.0], 0.0, pred_cell(a)); // world-model predictor of the observation
    }
    f
}

/// Refresh every estimator/predictor output (fire-on-change: toggle the arm cells 0→1).
fn present(f: &mut Field) {
    let zero: Vec<(usize, f32)> = (0..K).map(|a| (a, 0.0)).collect();
    f.stimulate(&zero, BUD);
    let one: Vec<(usize, f32)> = (0..K).map(|a| (a, 1.0)).collect();
    f.stimulate(&one, BUD);
}

fn reward(a: usize, rng: &mut Rng) -> f32 {
    if a == WIN && rng.f() < 0.9 {
        1.0
    } else {
        0.0
    }
}

/// The option's observation: NOISE emits random (unpredictable ⇒ high world-surprise); all others emit 0.
fn observation(a: usize, rng: &mut Rng) -> f32 {
    if a == NOISE {
        rng.sym()
    } else {
        0.0
    }
}

fn ensemble_mean(f: &Field, a: usize) -> f32 {
    (0..E).map(|e| f.cells()[val_cell(a, e)]).sum::<f32>() / E as f32
}

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    Greedy,     // select_explore β=0 (exploit value mean)
    Voi, // select_explore β>0 (mean + value-DISAGREEMENT) — the reward-relevant epistemic term
    WorldNovel, // mean + β·world-surprise (the DARK-ROOM baseline — raw observation unpredictability)
}

const WINDOW: usize = 120; // early/late measurement window

struct Stats {
    total: f32,
    noise_late: f32,   // fraction of the LAST window pulls that were the NOISE option
    nonwin_early: f32, // fraction of the FIRST window pulls that were NOT WIN (exploration)
    nonwin_late: f32,  // fraction of the LAST window pulls that were NOT WIN (residual exploration)
    win_pre: f32,      // WIN ensemble mean BEFORE any learning (the random prior)
    win_post: f32,     // WIN ensemble mean AFTER (learned from reward)
    win_visits: usize,
}

/// One run. The world-surprise curiosity baseline is OPTIMISTIC-init (all bands start maximally surprising) so
/// it is a GENUINE explorer — it visits every option, learns WIN pays, but a curiosity agent FLEES an option
/// once it becomes predictable, so it abandons the now-boring WIN for the NOISE option whose surprise is
/// IRREDUCIBLE. VoI's value-uncertainty, by contrast, is RESOLVABLE, so it settles and exploits.
fn run(policy: Policy, seed: u64) -> Stats {
    let mut f = build(seed);
    let opts: Vec<Vec<usize>> = (0..K)
        .map(|a| (0..E).map(|e| val_cell(a, e)).collect())
        .collect();
    let refs: Vec<&[usize]> = opts.iter().map(|v| v.as_slice()).collect();
    let mut rng = Rng::new(seed ^ 0xEFE);
    let mut world_surprise = [1.0f32; K]; // OPTIMISTIC init ⇒ a genuine curiosity explorer (not greedy-until-noise)
    present(&mut f);
    let win_pre = ensemble_mean(&f, WIN);
    let mut total = 0.0f32;
    let (mut noise_late, mut nonwin_early, mut nonwin_late, mut win_visits) =
        (0usize, 0usize, 0usize, 0usize);
    for t in 0..HORIZON {
        present(&mut f);
        let a = match policy {
            Policy::Greedy => f.select_explore(&refs, 0.0),
            Policy::Voi => f.select_explore(&refs, 2.5),
            Policy::WorldNovel => (0..K)
                .max_by(|&x, &y| {
                    (ensemble_mean(&f, x) + 2.5 * world_surprise[x])
                        .partial_cmp(&(ensemble_mean(&f, y) + 2.5 * world_surprise[y]))
                        .unwrap()
                })
                .unwrap(),
        };
        let r = reward(a, &mut rng);
        for e in 0..E {
            if rng.u() & 1 == 0 {
                f.learn_step(&[(a, 1.0)], val_cell(a, e), r, 0.1, BUD);
            }
        }
        let o = observation(a, &mut rng);
        let resid = f.learn_step(&[(a, 1.0)], pred_cell(a), o, 0.1, BUD);
        world_surprise[a] = 0.85 * world_surprise[a] + 0.15 * resid;
        total += r;
        if a == WIN {
            win_visits += 1;
        }
        if t < WINDOW && a != WIN {
            nonwin_early += 1;
        }
        if t >= HORIZON - WINDOW {
            if a == NOISE {
                noise_late += 1;
            }
            if a != WIN {
                nonwin_late += 1;
            }
        }
    }
    present(&mut f);
    Stats {
        total,
        noise_late: noise_late as f32 / WINDOW as f32,
        nonwin_early: nonwin_early as f32 / WINDOW as f32,
        nonwin_late: nonwin_late as f32 / WINDOW as f32,
        win_pre,
        win_post: ensemble_mean(&f, WIN),
        win_visits,
    }
}

fn avg(policy: Policy, seeds: &[u64], f: impl Fn(&Stats) -> f32) -> f32 {
    seeds.iter().map(|&s| f(&run(policy, s))).sum::<f32>() / seeds.len() as f32
}

#[test]
fn voi_self_bootstraps_and_avoids_the_dark_room() {
    let seeds = [0xA1u64, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8];
    let voi_r = avg(Policy::Voi, &seeds, |s| s.total);
    let greedy_r = avg(Policy::Greedy, &seeds, |s| s.total);
    let world_r = avg(Policy::WorldNovel, &seeds, |s| s.total);
    // THE LOAD-BEARING DISCRIMINATOR: where does each policy spend its LATE pulls?
    let voi_noise = avg(Policy::Voi, &seeds, |s| s.noise_late);
    let world_noise = avg(Policy::WorldNovel, &seeds, |s| s.noise_late);
    // exploration DRIVE annealing (not winner convergence): non-WIN pull fraction early vs late, VoI
    let voi_explore_early = avg(Policy::Voi, &seeds, |s| s.nonwin_early);
    let voi_explore_late = avg(Policy::Voi, &seeds, |s| s.nonwin_late);
    // grown value: WIN's ensemble mean pre→post, and it WAS actually visited
    let win_pre = avg(Policy::Voi, &seeds, |s| s.win_pre);
    let win_post = avg(Policy::Voi, &seeds, |s| s.win_post);
    let win_visits = avg(Policy::Voi, &seeds, |s| s.win_visits as f32);

    eprintln!("[AIF] REWARD (8 seeds, {HORIZON} steps): VoI {voi_r:.0}  greedy {greedy_r:.0}  world-surprise-curiosity {world_r:.0}");
    eprintln!("[AIF] DARK-ROOM discriminator — LATE pulls on the NOISE option: VoI {voi_noise:.2}  vs  world-surprise {world_noise:.2} (both EXPLORE early; only world-surprise CAMPS on irreducible noise)");
    eprintln!("[AIF] EXPLORATION ANNEALS (VoI): non-WIN pull fraction early {voi_explore_early:.2} → late {voi_explore_late:.2} (FIXED β; the DRIVE decays, not just the winner)");
    eprintln!("[AIF] GROWN VALUE (VoI): WIN ensemble mean {win_pre:.2} (prior) → {win_post:.2} (learned), WIN visited {win_visits:.0}× — value discovered from reward");
    eprintln!("[AIF] VERDICT: the substrate's native VoI (`select_explore`, mean + β·value-DISAGREEMENT) self-bootstraps exploration with NO ε-SEED and NO schedule — the exploration DRIVE anneals ({voi_explore_early:.2}→{voi_explore_late:.2}) because the value-disagreement (F's own residual uncertainty about value) self-vanishes, so it reproduces MorphLab's F↔G clause (D) WITHOUT their `esc` coupling formula. The DARK-ROOM control is now genuine: an EQUALLY-exploratory curiosity agent (optimistic world-surprise) explores the same options but is LURED into the noise corner — it FLEES the WIN once WIN becomes predictable and camps on the option whose surprise is IRREDUCIBLE (late NOISE pulls {world_noise:.2} vs VoI {voi_noise:.2}), earning LESS reward than even greedy ({world_r:.0} vs {greedy_r:.0}). VoI does not, because value-uncertainty is RESOLVABLE: the SUBJECT of the epistemic term (resolvable value vs irreducible world) is the mechanism that keeps the organism out of the dark room. HONEST SCOPE: β=1.5 is a FIXED epistemic weight (meta-evolvable like `epistemic_weight`); 'ε-free' means no exploration SCHEDULE/seed, not a free β. This is the bandit SELECTION seam — it deliberately has NO hidden cause or reward delay, so it does NOT test (and does not claim to replace) MorphLab's belief filter / pragmatic C, which that harder regime needs; the grown epistemic-SEARCH policy stays the open frontier. Synthetic; real-data value arm blocked on a reward-bearing rollout.");

    // ── GATES (only claims that survive an adversarial verify) ────────────────────────────────────────────
    assert!(
        voi_r > 2.0 * greedy_r.max(1.0),
        "BOOTSTRAP: VoI {voi_r:.0} did not ε-freely beat greedy {greedy_r:.0}"
    );
    // dark-room WITH TEETH: the curiosity agent forgoes reward (below greedy) AND camps on noise where VoI leaves
    assert!(world_r < greedy_r, "DARK-ROOM: world-surprise curiosity {world_r:.0} did not forgo reward vs greedy {greedy_r:.0} (the trap must have teeth)");
    assert!(world_noise > 0.5 && voi_noise < 0.15, "DARK-ROOM subject: world-surprise must CAMP on noise ({world_noise:.2}) while VoI LEAVES it ({voi_noise:.2}) — the resolvable-vs-irreducible mechanism");
    assert!(voi_explore_late < 0.5 * voi_explore_early, "SELF-ANNEAL: exploration drive did not decay ({voi_explore_early:.2} → {voi_explore_late:.2})");
    assert!(win_visits > 40.0 && win_post > win_pre + 0.4, "GROWN VALUE: WIN not visited+learned (visits {win_visits:.0}, {win_pre:.2} → {win_post:.2})");
}
