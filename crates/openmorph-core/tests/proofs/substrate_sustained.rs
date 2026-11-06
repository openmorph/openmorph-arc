// SPDX-License-Identifier: MIT-0
//! substrate_sustained — SUSTAINED INTERNAL THINKING: decide, or HOLD for more information (registry D8).
//!
//! The deliberation loop that sits on top of the autonomous heartbeat (D6/D7 `Field::live`), episodic
//! replay/imagine (#189), and the pragmatic free-energy score `G` (`unified-free-energy-actor.md`). Between
//! external events the organism runs its OWN subjective time, REPLAYING its stored memories into a fresh
//! judgment and ACCUMULATING evidence toward a decision — a drift-diffusion process in event-time. It COMMITS
//! when confidence crosses a threshold, and HOLDS (abstains — "no action yet") when the evidence is genuinely
//! insufficient, RESUMING and committing only when new async information tips it over. ZERO new core: the whole
//! faculty is a usage protocol over `integrate` (the accumulator), `store_episode`/`imagine` (memory replayed
//! internally), and `stimulate` (open a fresh judgment / stream new evidence). No `argmax`, no authored driver.
//!
//! THE MODEL. A hidden binary state `y ∈ {−1,+1}` emits a stream of NOISY glimpses `c = y·signal + noise`. A
//! single glimpse is ambiguous. The organism gathers the glimpses into episodic memory and, to decide, opens a
//! FRESH judgment — it resets its accumulator to the undecided prior `s* = 0` and DELIBERATES: it internally
//! replays (`imagine`) its memories, each replay nudging a leaky-integrator accumulator `μ` (`integrate`). The
//! confidence is the pragmatic free-energy distance from the undecided prior, `G = (μ − s*)² = μ²` (the same `G`
//! the beam/actor proofs use, s*=0). It COMMITS `sign(μ)` once `G ≥ Θ²`, else it HOLDS.
//!
//! Each claim (a)-(c) is scored INSIDE, multi-seed, each with a control that MUST fail:
//!   (a) DELIBERATION BEATS REACTION — a reactive single-settle (one glimpse) is unreliable; the deliberating
//!       organism reaches the correct decision far more often. Load-bearing control: WITHOUT the internal replay
//!       loop the fresh accumulator never leaves the prior, so the organism can make NO decision at all — the
//!       thinking, not the perception, produces the answer.
//!   (b) CORRECT HOLD — when evidence is genuinely insufficient the organism HOLDS instead of guessing; a
//!       control FORCED to decide immediately guesses at chance. Hold rate is high when ambiguous, low when clear.
//!   (c) RESUME ON NEW INFO — after holding, feeding the disambiguating async input makes it COMMIT correctly;
//!       and the hold is a real "not yet", NOT a timer: 10× more internal thinking on the same memories never
//!       crosses the threshold, only the new information does.
//!
//! Truth-not-trophy: the reactive baseline is the SAME organism deciding on a single settle (a genuine baseline,
//! not a strawman); the hold is evidence-gated (bounded internal re-examination cannot manufacture a decision).
//!
//!   cargo test -p openmorph-core --test proofs substrate_sustained -- --nocapture

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ── the deliberation protocol (a usage protocol over integrate + store_episode + imagine; ZERO core) ─────────
const HL: f32 = 4.0; // the accumulator's leak half-life, in the organism's own subjective event-time
const N: usize = 64; // glimpses gathered into episodic memory before deciding
const THETA: f32 = 0.20; // commit when confidence G = μ² ≥ Θ² (distance of the accumulated evidence from s*=0)
const SWEEPS: usize = 6; // internal deliberation sweeps (replaying the whole memory) per decision attempt

/// A decision the organism can reach.
#[derive(PartialEq, Debug, Clone, Copy)]
enum Decision {
    Commit(f32), // committed to a sign (±1)
    Hold,        // abstained — evidence insufficient ("no action yet")
}

/// Build the memory-morph: cell 0 = the glimpse channel; a unit echoes it into cell 1 (so every distinct
/// glimpse advances subjective time); the accumulator `μ` is a leaky integrator (`integrate`) over cell 1.
fn make() -> (Field, usize) {
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![1.0], 0.0, 1); // echo: fires on a changed glimpse, advancing the subjective clock
    let mu = f.integrate(1, HL); // the accumulator — the organism's decision variable, in event-time
    (f, mu)
}

/// OBSERVE — stream the noisy glimpses into episodic memory (`store_episode`). No decision is taken here.
fn observe(cues: &[f32]) -> (Field, usize, Vec<usize>) {
    let (mut f, mu) = make();
    let ids = cues
        .iter()
        .map(|&c| {
            f.stimulate(&[(0, c)], 16);
            f.store_episode(&[0], &[])
        })
        .collect();
    (f, mu, ids)
}

/// DELIBERATE — open a fresh judgment (reset the accumulator + its source to the undecided prior s*=0) and
/// think internally: replay the memories `sweeps` times, each `imagine` nudging `μ`. Returns the accumulator.
fn deliberate(f: &mut Field, mu_cell: usize, ids: &[usize], sweeps: usize) -> f32 {
    f.stimulate(&[(1, 0.0), (mu_cell, 0.0)], 4); // reset to the prior s* = 0 (a fresh judgment)
    for _ in 0..sweeps {
        for &id in ids {
            f.imagine(id, 16); // internal reactivation — memory recombined with no external input
        }
    }
    f.cells()[mu_cell]
}

/// The COMMIT-OR-HOLD rule: confidence is the pragmatic free-energy distance `G = (μ − s*)² = μ²`.
fn decide(mu: f32) -> Decision {
    if mu * mu >= THETA * THETA {
        Decision::Commit(mu.signum())
    } else {
        Decision::Hold
    }
}

/// REACT — the reactive baseline: settle ONCE on a single glimpse and read out its sign. No memory, no
/// accumulation (the "no sustained internal state" organism).
fn react(glimpse: f32) -> f32 {
    let (mut f, _) = make();
    f.stimulate(&[(0, glimpse)], 16);
    f.cells()[1].signum()
}

/// Draw a trial's glimpse stream for hidden state `y`.
fn stream(y: f32, signal: f32, noise: f32, rng: &mut Rng) -> Vec<f32> {
    (0..N).map(|_| y * signal + rng.noise(noise)).collect()
}

// ── (a) DELIBERATION BEATS REACTION ──────────────────────────────────────────────────────────────────────
/// On a hard case (real signal buried in per-glimpse noise) a single settle is unreliable, while sustained
/// internal replay integrates the evidence to a confident, correct decision. The load-bearing control: WITHOUT
/// the internal loop the fresh accumulator sits at the prior, so the organism decides NOTHING — the thinking is
/// what produces the answer. Multi-seed (each trial an independent seed).
#[test]
fn deliberation_beats_reaction() {
    const SIGNAL: f32 = 0.5;
    const NOISE: f32 = 3.0; // std = NOISE/3 = 1.0 ≈ 2× the signal ⇒ one glimpse is barely better than a coin
    let trials = 240;
    let (mut react_ok, mut commit, mut commit_ok, mut no_think_commit) = (0usize, 0, 0, 0);
    let mut determinism_seen = false;

    for t in 0..trials {
        let mut rng = Rng::new(1 + t as u64 * 0x100);
        let y = rng.bit();
        let cues = stream(y, SIGNAL, NOISE, &mut rng);

        // REACTION: a single settle on the latest glimpse.
        if react(*cues.last().unwrap()) == y {
            react_ok += 1;
        }

        // DELIBERATION: gather → open a fresh judgment → replay-integrate → decide.
        let (mut f, mu_cell, ids) = observe(&cues);
        // CONTROL (load-bearing): a fresh judgment with ZERO deliberation sweeps decides nothing.
        let no_think = deliberate(&mut f, mu_cell, &ids, 0);
        if decide(no_think) != Decision::Hold {
            no_think_commit += 1;
        }
        // now actually think:
        let mu = deliberate(&mut f, mu_cell, &ids, SWEEPS);
        if let Decision::Commit(s) = decide(mu) {
            commit += 1;
            if s == y {
                commit_ok += 1;
            }
        }

        // DETERMINISM: re-deliberating the same memories reproduces the same μ byte-for-byte (once suffices).
        if !determinism_seen {
            let mut g = f.clone();
            let mu_again = deliberate(&mut g, mu_cell, &ids, SWEEPS);
            assert_eq!(
                mu.to_bits(),
                mu_again.to_bits(),
                "deliberation is deterministic (subjective-clock replay, never wall-clock)"
            );
            determinism_seen = true;
        }
    }

    let react_acc = react_ok as f32 / trials as f32;
    let commit_rate = commit as f32 / trials as f32;
    let commit_acc = if commit > 0 {
        commit_ok as f32 / commit as f32
    } else {
        0.0
    };
    eprintln!(
        "[SUSTAINED] (a) REACTION single-settle acc {:.0}%  |  DELIBERATION committed {}/{} ({:.0}%) at acc {:.0}%  |  CONTROL no-thinking commits {}/{}",
        100.0 * react_acc,
        commit,
        trials,
        100.0 * commit_rate,
        100.0 * commit_acc,
        no_think_commit,
        trials
    );
    assert!(
        react_acc < 0.80,
        "a reactive single glimpse is genuinely unreliable here: {react_acc:.2}"
    );
    assert!(
        commit_acc > 0.90,
        "when the organism deliberates to a decision it is nearly always right: {commit_acc:.2}"
    );
    assert!(
        commit_acc - react_acc > 0.15,
        "deliberation must clearly beat reaction ({commit_acc:.2} vs {react_acc:.2})"
    );
    assert!(
        commit_rate > 0.5,
        "and it commits on a majority of the hard trials: {commit_rate:.2}"
    );
    assert_eq!(
        no_think_commit, 0,
        "CONTROL: without the internal replay loop the organism decides NOTHING — thinking is load-bearing"
    );
    eprintln!("[SUSTAINED] (a) VERDICT: sustained internal replay integrates ambiguous glimpses to a correct decision where a reactive single-settle fails; strip the loop and there is no decision at all.");
}

// ── (b) CORRECT HOLD ─────────────────────────────────────────────────────────────────────────────────────
/// When the evidence is genuinely insufficient (near-zero signal) the organism HOLDS rather than guess; when it
/// is clear it decides. Control: an organism FORCED to commit `sign(μ)` at ambiguity guesses at chance.
#[test]
fn correct_hold_vs_forced_guess() {
    let clear_hold = hold_rate(0.5, 1.5, 21); // clear world: strong signal, low noise
    let (ambig_hold, forced_acc) = hold_and_forced(0.03, 3.0, 42); // ambiguous world: ~no signal

    eprintln!(
        "[SUSTAINED] (b) HOLD RATE  clear {:.0}%  vs  ambiguous {:.0}%   |   FORCED-decision acc at ambiguity {:.0}% (≈ chance)",
        100.0 * clear_hold,
        100.0 * ambig_hold,
        100.0 * forced_acc
    );
    assert!(
        clear_hold < 0.15,
        "when the evidence is clear the organism decides (low hold): {clear_hold:.2}"
    );
    assert!(
        ambig_hold > 0.55,
        "when the evidence is ambiguous the organism mostly HOLDS: {ambig_hold:.2}"
    );
    assert!(
        ambig_hold - clear_hold > 0.40,
        "the hold is evidence-gated: far higher when ambiguous than when clear"
    );
    assert!(
        forced_acc < 0.60,
        "CONTROL: an organism FORCED to decide at ambiguity is no better than a coin ({forced_acc:.2}) — \
         holding is what avoids the wrong guesses"
    );
    eprintln!("[SUSTAINED] (b) VERDICT: the abstention is a real evidence-gated 'not yet' — high when ambiguous, near-zero when clear; forcing a decision there just guesses.");
}

/// Fraction of trials the organism HOLDS in a world of the given difficulty.
fn hold_rate(signal: f32, noise: f32, seed_salt: u64) -> f32 {
    let trials = 240;
    let mut hold = 0usize;
    for t in 0..trials {
        let mut rng = Rng::new(seed_salt + t as u64 * 0x100);
        let y = rng.bit();
        let cues = stream(y, signal, noise, &mut rng);
        let (mut f, mu_cell, ids) = observe(&cues);
        let mu = deliberate(&mut f, mu_cell, &ids, SWEEPS);
        if decide(mu) == Decision::Hold {
            hold += 1;
        }
    }
    hold as f32 / trials as f32
}

/// Hold rate AND the accuracy of a control forced to commit `sign(μ)` regardless of confidence.
fn hold_and_forced(signal: f32, noise: f32, seed_salt: u64) -> (f32, f32) {
    let trials = 240;
    let (mut hold, mut forced_ok) = (0usize, 0usize);
    for t in 0..trials {
        let mut rng = Rng::new(seed_salt + t as u64 * 0x100);
        let y = rng.bit();
        let cues = stream(y, signal, noise, &mut rng);
        let (mut f, mu_cell, ids) = observe(&cues);
        let mu = deliberate(&mut f, mu_cell, &ids, SWEEPS);
        if decide(mu) == Decision::Hold {
            hold += 1;
        }
        if mu.signum() == y {
            forced_ok += 1; // the forced control commits anyway
        }
    }
    (
        hold as f32 / trials as f32,
        forced_ok as f32 / trials as f32,
    )
}

// ── (c) RESUME ON NEW INFO (the hold is a "not yet", not a timer) ─────────────────────────────────────────
/// After holding on ambiguous evidence, feeding the disambiguating async input makes the organism COMMIT to the
/// correct decision. Two controls prove the hold is evidence-gated, not a timeout: (i) 10× more internal
/// thinking on the SAME memories never crosses the threshold; (ii) that same extended thinking commits to
/// nothing. Only the new information resolves the hold. Studied on the trials that genuinely held.
#[test]
fn resume_on_new_info_not_a_timer() {
    let trials = 200;
    let (mut held, mut stay_held, mut resume_ok, mut think_only_commit) = (0usize, 0, 0, 0);

    for t in 0..trials {
        let mut rng = Rng::new(0xC001 + t as u64 * 0x100);
        let y = rng.bit();
        let cues = stream(y, 0.03, 3.0, &mut rng); // ambiguous: ~no signal
        let (mut f, mu_cell, mut ids) = observe(&cues);
        let mu = deliberate(&mut f, mu_cell, &ids, SWEEPS);
        if decide(mu) != Decision::Hold {
            continue; // only study genuinely-held trials
        }
        held += 1;

        // CONTROL (not a timer): 10× the thinking budget on the SAME memories.
        let mut fx = f.clone();
        let mu_extra = deliberate(&mut fx, mu_cell, &ids, SWEEPS * 10);
        if decide(mu_extra) == Decision::Hold {
            stay_held += 1;
        }
        if decide(mu_extra) != Decision::Hold {
            think_only_commit += 1;
        }

        // RESUME: new disambiguating glimpses arrive (async), are stored, and the organism re-deliberates.
        for _ in 0..12 {
            let c = 0.6 * y + rng.noise(1.0); // strong, consistent — distinct draws so each advances event-time
            f.stimulate(&[(0, c)], 16);
            ids.push(f.store_episode(&[0], &[]));
        }
        let mu2 = deliberate(&mut f, mu_cell, &ids, SWEEPS);
        if decide(mu2) == Decision::Commit(y) {
            resume_ok += 1;
        }
    }

    eprintln!(
        "[SUSTAINED] (c) of {held} genuinely-held trials:  stay-held after 10× thinking {stay_held}/{held}  |  commit-after-NEW-info {resume_ok}/{held} (correct)  |  CONTROL commit-after-more-thinking-only {think_only_commit}/{held}",
    );
    assert!(
        held as f32 / trials as f32 > 0.5,
        "the ambiguous world genuinely holds a majority of trials: {held}/{trials}"
    );
    assert!(
        stay_held as f32 / held as f32 > 0.9,
        "the hold is NOT a timer: extra internal thinking leaves it held ({stay_held}/{held})"
    );
    assert_eq!(
        think_only_commit, 0,
        "CONTROL: more thinking with no new information manufactures NO decision — the hold is evidence-gated"
    );
    assert!(
        resume_ok as f32 / held as f32 > 0.85,
        "the new async information RESOLVES the hold to the correct decision ({resume_ok}/{held})"
    );
    eprintln!("[SUSTAINED] (c) VERDICT: holding is a real 'not yet' — bounded internal re-examination never crosses the threshold; only the disambiguating information tips the organism to the correct commit.");
}
