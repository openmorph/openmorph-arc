// SPDX-License-Identifier: MIT-0
//! ACTION DWELL / PERSISTENCE — the actor's missing TEMPORAL axis (substrate completeness, opt-in, byte-identical
//! off). `alive(Event::Act)` decides among the options EVERY step, so an actor whose estimates FLUCTUATE (two arms
//! that both pay a noisy reward: the running means keep crossing) flips between them on noise. `Config::action_dwell`
//! carries a GROWN commitment DURATION per option: a chosen action is HELD for a learned number of steps (starts at
//! 1 ≡ off; extended by reward, released when it stops earning). The temporal analogue of grown structural DEPTH
//! (`bank_skill`): grow widens one layer, dwell holds one choice. Since 1.6 dwell applies to EVERY acting rule; this
//! proof runs the default (valence) rule. (It used to run the free-energy softmax, deleted in 1.6 as a documented
//! negative on real ARC; the claim and its thresholds are unchanged.)
//!
//! HONEST SCOPE: this proof asserts the BEHAVIOUR (dwell makes a re-deciding actor commit — longer action runs),
//! NOT a reward benefit; morphbot-35's real-game sticky-action test found none. Opt-in completeness.
//!
//!   cargo test -p openmorph-core --test proofs substrate_dwell -- --nocapture

use openmorph_core::substrate::{Event, Field, Outcome};
use openmorph_core::Config;

const ARM_A: usize = 0;
const ARM_B: usize = 1;
const T: usize = 600;

fn step(f: &mut Field, options: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act { options, last }) {
        Outcome::Acted { chosen } => chosen,
        _ => 0,
    }
}

/// A seeded reward-noise draw in `[0, 1)` (the harness's own stream, separate from the organism's PRNGs).
fn noise(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 40) as f32 / (1u64 << 24) as f32
}

/// Mean consecutive-action run length under the default (valence) rule when BOTH arms pay a noisy positive
/// reward of the same mean (0.05–0.15) — the running estimates keep crossing, so a re-deciding actor flips on
/// noise (the persistence signature).
fn mean_run_len(dwell: bool, seed: u64) -> f32 {
    let mut f = Field::new(2);
    f.rng_seed(seed);
    f.configure(Config {
        action_dwell: dwell,
        ..Config::default()
    });
    let options = [ARM_A, ARM_B];
    let mut last: Option<(usize, f32)> = None;
    let mut prev: Option<usize> = None;
    let mut runs = 0usize;
    let mut world = seed ^ 0x05EE_D0FD_3E11;
    for _ in 0..T {
        let chosen = step(&mut f, &options, last);
        let reward = 0.05 + 0.1 * noise(&mut world);
        last = Some((chosen, reward));
        if Some(chosen) != prev {
            runs += 1;
            prev = Some(chosen);
        }
    }
    T as f32 / runs.max(1) as f32
}

#[test]
fn action_dwell_makes_a_re_deciding_actor_commit_longer_runs() {
    let seeds = [1u64, 7, 42, 99, 5, 2024, 0xBEEF, 3];
    let off: f32 = seeds.iter().map(|&s| mean_run_len(false, s)).sum::<f32>() / seeds.len() as f32;
    let on: f32 = seeds.iter().map(|&s| mean_run_len(true, s)).sum::<f32>() / seeds.len() as f32;
    eprintln!(
        "[DWELL] mean action-run length over {} seeds: valence alone {off:.2}  vs  valence + action_dwell {on:.2}",
        seeds.len()
    );
    assert!(
        on > off * 1.5,
        "action_dwell must make the re-deciding actor COMMIT — mean run {on:.2} should clearly exceed re-deciding {off:.2}"
    );
    eprintln!("[DWELL] VERDICT: the grown commitment duration lets the actor HOLD a working choice instead of flipping on reward noise every step (mean run {off:.2}->{on:.2}). HONEST SCOPE: NO reward benefit claimed (morphbot-35's real-game sticky test found none). Opt-in completeness, byte-identical off.");
}

#[test]
fn action_dwell_off_is_byte_identical() {
    assert!(
        !Config::default().action_dwell,
        "action_dwell must default OFF"
    );
    let a = mean_run_len(false, 777);
    let b = mean_run_len(false, 777);
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "the off path must be byte-reproducible"
    );
}

/// Mean consecutive-action run length under a CONSTANT ZERO-reward stream (the sparse-reward case — measured on
/// 23/25 ARC-3 games). Nothing is EARNING, so a correct grown dwell stays neutral and the actor keeps mixing (the
/// valence rule alternates toward the less-visited option). (Regression for the `r.signum()`-at-zero perseveration
/// bug: `f32::signum(+0.0) == 1.0` used to ratchet `dwell_len` to `dwell_max` every zero-reward step.)
fn mean_run_len_zero_reward(dwell: bool, seed: u64) -> f32 {
    let mut f = Field::new(2);
    f.rng_seed(seed);
    f.configure(Config {
        action_dwell: dwell,
        ..Config::default()
    });
    let options = [ARM_A, ARM_B];
    let mut last: Option<(usize, f32)> = None;
    let mut prev: Option<usize> = None;
    let mut runs = 0usize;
    for _ in 0..T {
        let chosen = step(&mut f, &options, last);
        last = Some((chosen, 0.0)); // ZERO reward — nothing is earning
        if Some(chosen) != prev {
            runs += 1;
            prev = Some(chosen);
        }
    }
    T as f32 / runs.max(1) as f32
}

#[test]
fn zero_reward_does_not_perseverate() {
    let (mut on, mut off) = (0.0f32, 0.0f32);
    for s in 0..8 {
        on += mean_run_len_zero_reward(true, s);
        off += mean_run_len_zero_reward(false, s);
    }
    on /= 8.0;
    off /= 8.0;
    eprintln!("[DWELL/ZERO] zero-reward run length: dwell-ON {on:.2} vs dwell-OFF {off:.2} (dwell_max=32)");
    // A grown dwell must not GROW under zero reward (nothing earns): ON must stay near OFF, not blow toward
    // dwell_max. Pre-fix this failed (ON ≈ dwell_max ≫ OFF); post-fix ON ≈ OFF.
    assert!(
        on < off * 3.0 + 2.0,
        "zero reward must not perseverate: dwell-ON {on:.2} should be ~dwell-OFF {off:.2}, not near dwell_max"
    );
}
