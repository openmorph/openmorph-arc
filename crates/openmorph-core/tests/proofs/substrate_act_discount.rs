// SPDX-License-Identifier: MIT-0
//! substrate_act_discount — MULTI-STEP CREDIT ASSIGNMENT inside `alive(Event::Act)` (`Config::act_discount`).
//!
//! THE WALL. `alive(Act)` folds the reward of the LAST (option, reward) into that option's learned value. On a
//! sparse multi-step task only the action that ENTERS the goal is ever paid, so the actor can stumble onto the
//! goal (exploration finds it) and never learn its way back: nothing upstream of the last step learns that it
//! leads anywhere. At γ = 0 the valence rule's return visits come only from its visit-count exploration sweeping
//! the grid again — about a quarter of the late episodes, at ≈180 of the 200 capped steps
//! (`examples/actor_bench.rs` in openmorph-perception measures it next to the textbook baselines).
//!
//! THE FIX (no API change). The options an `Act` call receives ARE the successors of the last (option, reward),
//! so the fold can bootstrap from them: target = `r + γ·max_{o' ∈ options} (μ_{o'} + σ_{o'}²)` — one-step Q-learning over
//! option cells, the same local delta rule toward the successor's value. `γ = act_discount`, default `0.0` = the
//! immediate-reward fold (byte-identical).
//!
//! THREE MEASURED CLAIMS (seeded, per-seed ACTION PERMUTATION so no positional tie-break can pass for a policy):
//!   (A) γ = 0 FINDS the goal but does not learn the way back: late success far below reliable, late episodes
//!       run close to the step cap (many times the optimal length).
//!   (B) γ = 0.95 reaches it RELIABLY, at NEAR-OPTIMAL length (the shortest path is 14 steps).
//!   (C) BANDITS ARE UNCHANGED at γ = 0: an explicit `act_discount = 0.0` is bit-identical to the default, and the
//!       folded value is the arm's own reward (no bootstrap). A γ > 0 on a bandit bootstraps an arm from itself
//!       (every call re-offers the same options) — why the default is 0.
//!   (D) DEEP EXPLORATION on the deceptive chain (10 states, start next to a near end that pays 0.2, the far end
//!       pays 1.0 eight steps away). The Valence fold bootstraps the successor's full `μ + σ²`, so an unexplored
//!       state's bonus flows back up the path to it, and each episode ends with an `Act` over NO options (the
//!       final reward folds with no bootstrap). With both, γ = 0.95 reaches the far end on every late episode.
//!       Without the episode end, the final step bootstraps from the restart state, the one-step trap pays
//!       0.2 per step against the far end's 0.125, and the actor keeps the trap. Before this change the bootstrap read
//!       `μ` alone and the chain was 0% for every rule even with episode ends (`actor_bench` ablation).
//!
//!   cargo test -p openmorph-core --test proofs substrate_act_discount -- --nocapture

use openmorph_core::{Config, Event, Field, Outcome};

const W: usize = 8;
const H: usize = 8;
const ACTIONS: usize = 4;
const CAP: usize = 200;
const EPISODES: usize = 200;
const OPTIMAL: usize = 14;
const SEEDS: u64 = 8;

/// The `rng_seed` of run `s` — odd and distinct (`Field::rng_seed` ORs the seed with 1, so an even seed and its
/// successor would share a tie-break stream).
fn seed_of(s: u64) -> u64 {
    101 + 2 * s
}

fn act(f: &mut Field, options: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act { options, last }) {
        Outcome::Acted { chosen } => chosen,
        _ => unreachable!("alive(Act) always acts"),
    }
}

/// splitmix64 — the harness's own stream (the action permutation only).
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The per-seed action permutation (the organism's action index → the world's move).
fn permutation(seed: u64) -> [usize; ACTIONS] {
    let mut p = [0, 1, 2, 3];
    let mut z = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xA5A5;
    for i in (1..ACTIONS).rev() {
        z = mix(z);
        p.swap(i, (z % (i as u64 + 1)) as usize);
    }
    p
}

/// Two rooms: a wall at column W/2 except a door at row H−2. Start (0,0), goal (W−1,H−1) pays 1 and ends.
fn step(x: usize, y: usize, mv: usize) -> (usize, usize) {
    let (dx, dy): (i32, i32) = [(0, -1), (0, 1), (-1, 0), (1, 0)][mv];
    let nx = (x as i32 + dx).clamp(0, W as i32 - 1) as usize;
    let ny = (y as i32 + dy).clamp(0, H as i32 - 1) as usize;
    if nx == W / 2 && ny != H - 2 {
        (x, y) // the wall
    } else {
        (nx, ny)
    }
}

/// Run `EPISODES` episodes of the gridworld through `alive(Act)` over `(state, action)` option cells with the given
/// discount. Returns (first successful episode or EPISODES, success rate over the last quarter, mean steps per
/// episode over the last quarter).
fn run_grid(gamma: f32, seed: u64) -> (usize, f32, f32) {
    let mut f = Field::new(W * H * ACTIONS);
    f.rng_seed(seed);
    f.config_mut().act_discount = gamma;
    let perm = permutation(seed);
    let late = EPISODES - EPISODES / 4;
    let (mut first, mut succ, mut steps_late) = (EPISODES, 0usize, 0usize);
    let mut last: Option<(usize, f32)> = None;
    for ep in 0..EPISODES {
        let (mut x, mut y) = (0usize, 0usize);
        let mut reached = false;
        let mut steps = 0;
        while steps < CAP {
            let s = y * W + x;
            let options: Vec<usize> = (0..ACTIONS).map(|a| s * ACTIONS + a).collect();
            let chosen = act(&mut f, &options, last);
            (x, y) = step(x, y, perm[chosen - s * ACTIONS]);
            steps += 1;
            reached = x == W - 1 && y == H - 1;
            last = Some((chosen, if reached { 1.0 } else { 0.0 }));
            if reached {
                break;
            }
        }
        if reached && first == EPISODES {
            first = ep;
        }
        if ep >= late {
            succ += usize::from(reached);
            steps_late += steps;
        }
    }
    let n = (EPISODES - late) as f32;
    (first, succ as f32 / n, steps_late as f32 / n)
}

#[test]
fn a_b_the_discount_turns_a_found_goal_into_a_reliable_short_path() {
    let mut rows = Vec::new();
    for gamma in [0.0f32, 0.95] {
        let runs: Vec<(usize, f32, f32)> =
            (0..SEEDS).map(|s| run_grid(gamma, seed_of(s))).collect();
        let found = runs.iter().filter(|r| r.0 < EPISODES).count();
        let late = runs.iter().map(|r| r.1).sum::<f32>() / SEEDS as f32;
        let steps = runs.iter().map(|r| r.2).sum::<f32>() / SEEDS as f32;
        let first = runs.iter().map(|r| r.0 as f32).sum::<f32>() / SEEDS as f32;
        eprintln!(
            "[ACT-DISCOUNT] γ={gamma:.2}: goal found in {found}/{SEEDS} seeds (first at episode {first:.1} on average); \
             late success {:.1}%, late steps/episode {steps:.1} (optimal {OPTIMAL}, cap {CAP})",
            100.0 * late
        );
        rows.push((gamma, found, late, steps));
    }
    let (_, found0, late0, steps0) = rows[0];
    let (_, found1, late1, steps1) = rows[1];
    // (A) γ = 0 does find the goal — the failure is credit assignment, not exploration — but does not learn the
    //     way back: its late episodes succeed only by re-exploring (well under half) and run near the step cap.
    assert!(
        found0 as u64 >= SEEDS - 1,
        "γ=0 must FIND the goal on (nearly) every seed — its failure is credit, not exploration ({found0}/{SEEDS})"
    );
    assert!(
        late0 < 0.5,
        "γ=0 must NOT return to the goal reliably after finding it (late success {:.1}%)",
        100.0 * late0
    );
    assert!(
        steps0 > 5.0 * OPTIMAL as f32,
        "γ=0's late episodes must stay far from the short path ({steps0:.1} steps vs optimal {OPTIMAL})"
    );
    // (B) γ = 0.95 reaches it reliably, at near-optimal length.
    assert_eq!(
        found1 as u64, SEEDS,
        "γ=0.95 must find the goal on every seed"
    );
    assert!(
        late1 >= 0.95,
        "γ=0.95 must reach the goal RELIABLY (late success {:.1}%)",
        100.0 * late1
    );
    assert!(
        steps1 <= OPTIMAL as f32 * 1.25,
        "γ=0.95 must reach it at NEAR-OPTIMAL length ({steps1:.1} steps vs optimal {OPTIMAL})"
    );
    // …and the discount is what makes the difference, by a wide margin.
    assert!(
        late1 - late0 >= 0.5,
        "γ=0.95 must beat γ=0 by a wide margin in late success ({:.1}% vs {:.1}%)",
        100.0 * late1,
        100.0 * late0
    );
    eprintln!("[ACT-DISCOUNT] VERDICT: bootstrapping the fold from the options the next call offers (r + γ·max (μ + σ²)) carries the sparse goal back along the path — the SAME alive(Act), no API change: γ=0 {:.0}% at {steps0:.1} steps → γ=0.95 {:.0}% at {steps1:.1} steps (optimal {OPTIMAL}).", 100.0 * late0, 100.0 * late1);
}

/// A deterministic 3-arm bandit through `alive(Act)`. Returns the choice sequence and the final `valence_of`
/// of each arm.
fn run_bandit(cfg: Config, seed: u64) -> (Vec<usize>, Vec<f32>) {
    const PAY: [f32; 3] = [0.2, 0.5, 0.8];
    let mut f = Field::new(3);
    f.rng_seed(seed);
    f.configure(cfg);
    let options = [0usize, 1, 2];
    let mut last = None;
    let mut picks = Vec::new();
    for _ in 0..600 {
        let a = act(&mut f, &options, last);
        picks.push(a);
        last = Some((a, PAY[a]));
    }
    let _ = act(&mut f, &options, last); // fold the final reward
    (picks, (0..3).map(|o| f.valence_of(o)).collect())
}

#[test]
fn c_bandits_are_unchanged_at_gamma_zero() {
    let (default_picks, default_val) = run_bandit(Config::default(), 7);
    let explicit = Config {
        act_discount: 0.0,
        ..Config::default()
    };
    let (zero_picks, zero_val) = run_bandit(explicit, 7);
    assert_eq!(
        default_picks, zero_picks,
        "an explicit act_discount = 0.0 must choose exactly as the default"
    );
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&default_val),
        bits(&zero_val),
        "…and fold bit-identical values"
    );
    // At γ = 0 the folded value IS the arm's reward (the best arm is exploited, its σ² annealed to ~0).
    let late_best = default_picks[400..].iter().filter(|&&a| a == 2).count();
    eprintln!(
        "[ACT-DISCOUNT C] bandit γ=0: valence_of = {default_val:?} (payoffs 0.2/0.5/0.8), best arm in the last 200 picks {late_best}/200"
    );
    assert!(
        (default_val[2] - 0.8).abs() < 1e-3,
        "at γ=0 the best arm's value is its own reward, no bootstrap ({:.4})",
        default_val[2]
    );
    assert!(
        late_best >= 190,
        "at γ=0 the bandit exploits the best arm ({late_best}/200)"
    );
    // The documented hazard: a γ > 0 on a bandit bootstraps an arm from ITSELF (the same options are re-offered),
    // so its value inflates toward r/(1−γ). This is why the default is 0.
    let (_, gamma_val) = run_bandit(
        Config {
            act_discount: 0.5,
            ..Config::default()
        },
        7,
    );
    eprintln!(
        "[ACT-DISCOUNT C] bandit γ=0.5: valence_of = {gamma_val:?} (inflated toward r/(1−γ) = 1.6)"
    );
    assert!(
        gamma_val[2] > 1.2,
        "a γ>0 on a bandit bootstraps an arm from itself — its value inflates past its reward ({:.3})",
        gamma_val[2]
    );
}

const CHAIN: usize = 10;
const CHAIN_CAP: usize = 30;
const CHAIN_EPISODES: usize = 300;

/// The deceptive chain through `alive(Act)` at γ = 0.95: states `0..CHAIN`, start 1, action 0 = left, 1 = right
/// (swapped on every other seed, so a positional tie-break cannot pass for a policy). State 0 pays 0.2 and ends; state
/// `CHAIN − 1` pays 1.0 and ends. `episode_end` folds each episode's final reward with an `Act` over no options.
/// Returns the share of the last quarter of episodes that reached the far end.
fn run_chain(seed: u64, episode_end: bool) -> f32 {
    let mut f = Field::new(CHAIN * 2);
    f.rng_seed(seed);
    f.config_mut().act_discount = 0.95;
    let right = usize::from(seed % 4 != 1); // seed_of(s) is odd; alternate the action layout across seeds
    let late = CHAIN_EPISODES - CHAIN_EPISODES / 4;
    let mut far = 0usize;
    let mut last: Option<(usize, f32)> = None;
    for ep in 0..CHAIN_EPISODES {
        let mut pos = 1usize;
        for _ in 0..CHAIN_CAP {
            let options = [pos * 2, pos * 2 + 1];
            let chosen = act(&mut f, &options, last);
            pos = if chosen - pos * 2 == right {
                pos + 1
            } else {
                pos - 1
            };
            let r = match pos {
                0 => 0.2,
                p if p == CHAIN - 1 => 1.0,
                _ => 0.0,
            };
            last = Some((chosen, r));
            if pos == 0 || pos == CHAIN - 1 {
                break;
            }
        }
        if ep >= late {
            far += usize::from(pos == CHAIN - 1);
        }
        if episode_end {
            let end = act(&mut f, &[], last.take());
            assert_eq!(end, usize::MAX, "an Act over no options chooses nothing");
        }
    }
    far as f32 / (CHAIN_EPISODES - late) as f32
}

#[test]
fn d_the_deceptive_chain_needs_optimism_and_an_episode_end() {
    let with_end: Vec<f32> = (0..SEEDS).map(|s| run_chain(seed_of(s), true)).collect();
    let without: Vec<f32> = (0..SEEDS).map(|s| run_chain(seed_of(s), false)).collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    eprintln!(
        "[ACT-DISCOUNT D] deceptive chain, far end in the last quarter: with episode ends {:.1}% (per seed {with_end:?}), \
         without {:.1}% (per seed {without:?})",
        100.0 * mean(&with_end),
        100.0 * mean(&without)
    );
    assert!(
        with_end.iter().all(|&x| x >= 0.95),
        "with episode ends the optimistic bootstrap must reach the far end on every seed ({with_end:?})"
    );
    assert!(
        mean(&without) <= 0.1,
        "without episode ends the one-step trap must win (it pays more per step): {without:?}"
    );
}
