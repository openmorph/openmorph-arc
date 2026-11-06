// SPDX-License-Identifier: MIT-0
//! GROWN RECURSIVE BELIEF FILTER — the tractable slice of #171 (OM-10). The full OM-10 vision fuses the fovea
//! (attend) and rc2-purpose (decide) into ONE expected-free-energy scalar `G = E_q[-log C(o|x)] − β·IG(x)`. The
//! SELECTION half of that (the `argmin-G` currency) is already proven on the substrate by
//! `substrate_active_inference`, which is explicit that it has "NO hidden cause and NO reward delay, so it does
//! NOT test or replace MorphLab's recursive belief filter". THIS proof is exactly that missing half: the belief
//! `q` itself — a posterior over a PERSISTING, only partially-observed hidden cause, carried and updated across
//! events, so the epistemic term of `G` finally has a belief to reduce and a K-delayed cause becomes
//! representable (requirement 1 of #171). The `argmin-G` FUSION on top of both halves stays gated on the ARC-3
//! loop and is NOT attempted here.
//!
//! This is NOT new machinery — it composes shipped primitives, and it makes an HONEST distinction the two
//! carried-state flavors force:
//!   - `substrate_recurrent` diagnosed that a cause-in-the-past needs MEMORY carried on the async medium.
//!   - `substrate_remember` GREW a recurrent latch (a unit reading `[cue, self]`). But that latch is the EXACT /
//!     register flavor: with a clean cue the delta rule drives its self-weight into a hard bistable LATCH that
//!     COMMITS on the first observation and HOLDS — perfect for a clean value, useless for a NOISY one (it locks
//!     onto the first, usually-wrong, observation instead of integrating).
//!   - A belief over a NOISY, persistent cause is the TREND / smoothed flavor, which the substrate ships as
//!     `integrate` — a native leaky integrator, an EMA in the organism's own event-time, `q_t = q_{t-1} +
//!     α·(o_t − q_{t-1})`. That IS a recursive belief filter, stable by construction. Its rate `α` (≡ a Kalman
//!     gain) is set by the half-life, so — to keep NOTHING authored — the half-life is GROWN by `adapt_scales`
//!     (the #137 timescale gradient), not hand-set. Integrating the noisy observation over event-time recovers
//!     the hidden state: `substrate_localize`'s evidence-accumulation (64%→98% as cues add up), run over
//!     EVENT-TIME instead of over space. Perception, memory and decision are one act: integrate evidence.
//!
//! GROWN, NOT AUTHORED: the belief carrier is a native `integrate` trace; its filter-gain (half-life) is grown
//! by `adapt_scales` from the residual of a linear readout that tries to name the causal state (the same one-hop
//! delta rule, `substrate_growtime`); the belief itself is read as `sign(trace)`. No authored Kalman gain, no
//! argmax, no hand-set retention. Zero core change — a pure usage protocol + this proof.
//!
//! THE TASK (synthetic POMDP, seeded): a hidden state `s ∈ {−1,+1}` is drawn per episode and PERSISTS. Each
//! event yields a noisy binary observation (the true sign with probability `P_REVEAL`, else the opposite) — one
//! observation is only weakly informative. Score INSIDE with a required-failure control whose ONLY difference is
//! the integration depth:
//!   RECURRENT  (the grown belief filter): integrates over event-time; accuracy rises with the event count K.
//!   MEMORYLESS (integration depth 0 — the raw sensor, the same filter at `α=1`): can only ever reflect the LAST
//!       observation, so it is pinned at the single-observation rate — flat, for every K.
//!
//!   cargo test -p openmorph-core --test proofs substrate_believe -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const OBS: usize = 0; // the observation channel (the world writes here)

// A noisy BINARY sensor: each event reports the true state with probability P_REVEAL, else the opposite — so a
// single observation is only weakly informative and the state must be integrated over many events.
const P_REVEAL: f32 = 0.62;
const HL0: f32 = 2.0; // the belief filter's INITIAL half-life — grown from here by adapt_scales
const MIN_HL: f32 = 1.0;
const MAX_HL: f32 = 64.0;
const BUDGET: usize = 8;
const LR: f32 = 0.05; // readout learning rate
const ADAPT_LR: f32 = 1.0; // half-life gradient rate (as in substrate_growtime)
const T: f32 = 0.5; // belief target magnitude (<1) — sign is all that is scored; keeps the readout in range

/// A belief filter: OBS, one `integrate` trace of it (the belief carrier, half-life grown), and a linear readout
/// `[obs, trace] → pred` whose residual drives the half-life gradient. Returns `(field, trace_cell, pred_cell)`.
fn belief_filter() -> (Field, usize, usize) {
    let mut f = Field::new(1); // cell 0 = OBS
    f.config_mut().max_hidden = 0; // a LINEAR readout — a clean one-hop backprojection for adapt_scales
    let tr = f.integrate(OBS, HL0); // the recursive belief filter (native leaky integrator)
    let pred = f.cells().len();
    f.ensure_cells(pred + 1);
    (f, tr, pred) // grow() births the readout writing `pred` on first call
}

/// A clean episode boundary: zero OBS, the trace and the readout, then fire — the trace relaxes toward the
/// (now-zero) source and stays 0, so the belief is exactly 0. All through the public `stimulate` path.
fn reset(f: &mut Field, tr: usize, pred: usize) {
    f.stimulate(&[(OBS, 0.0), (tr, 0.0), (pred, 0.0)], BUDGET);
}

/// One noisy observation of the persistent hidden state `s`: the true sign with probability `P_REVEAL`.
fn observe(s: f32, rng: &mut Rng) -> f32 {
    if rng.f() < P_REVEAL {
        s
    } else {
        -s
    }
}

/// Train online: episodes of a persistent hidden `s`. Each event grows the readout toward the causal state's own
/// value (exactly `substrate_remember`'s regime C) and lets `adapt_scales` grow the filter's half-life.
fn train(f: &mut Field, tr: usize, pred: usize, episodes: usize, ep_len: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    for _ in 0..episodes {
        let s = rng.bit();
        reset(f, tr, pred);
        for _ in 0..ep_len {
            let o = observe(s, &mut rng);
            f.grow(&[(OBS, o), (tr, f.cells()[tr])], pred, T * s, LR, BUDGET);
            f.adapt_scales(pred, T * s, ADAPT_LR, MIN_HL, MAX_HL);
        }
    }
}

/// Accuracy of the BELIEF `sign(trace)` after integrating exactly `k` noisy observations of one hidden state.
fn accuracy_at(f: &mut Field, tr: usize, pred: usize, k: usize, trials: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let (mut ok, mut tot) = (0.0f64, 0.0);
    for _ in 0..trials {
        let s = rng.bit();
        reset(f, tr, pred);
        for _ in 0..k {
            let o = observe(s, &mut rng);
            f.stimulate(&[(OBS, o)], BUDGET); // tick_traces integrates the belief; the readout fires (clock++)
        }
        if f.cells()[tr].signum() == s {
            ok += 1.0;
        }
        tot += 1.0;
    }
    ok / tot
}

/// The MEMORYLESS ceiling: a single-observation observer — `sign(o_k)` after the same k events. Integration
/// depth 0 (the same filter at α=1); it can only ever reflect the last observation, so it is flat in k.
fn accuracy_memoryless(k: usize, trials: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let (mut ok, mut tot) = (0.0f64, 0.0);
    for _ in 0..trials {
        let s = rng.bit();
        let mut last = 0.0;
        for _ in 0..k {
            last = observe(s, &mut rng);
        }
        if last.signum() == s {
            ok += 1.0;
        }
        tot += 1.0;
    }
    ok / tot
}

/// Accuracy after building a belief over `seen` observations, then holding across `gap` events of PURE NOISE
/// (uninformative ±1) — the persistence a delayed cause needs (#171 req 1). The recurrent belief carries its
/// sign; a single-observation observer sees only the current noise and is at chance.
fn accuracy_after_gap(
    f: &mut Field,
    tr: usize,
    pred: usize,
    seen: usize,
    gap: usize,
    trials: usize,
    seed: u64,
) -> (f64, f64) {
    let mut rng = Rng::new(seed);
    let (mut rec_ok, mut mem_ok, mut tot) = (0.0f64, 0.0, 0.0);
    for _ in 0..trials {
        let s = rng.bit();
        reset(f, tr, pred);
        for _ in 0..seen {
            let o = observe(s, &mut rng);
            f.stimulate(&[(OBS, o)], BUDGET);
        }
        let mut last = 0.0f32;
        for _ in 0..gap {
            last = rng.bit(); // the cause is now in the PAST; the world ticks but carries no information about s
            f.stimulate(&[(OBS, last)], BUDGET);
        }
        if f.cells()[tr].signum() == s {
            rec_ok += 1.0;
        }
        if last.signum() == s {
            mem_ok += 1.0;
        }
        tot += 1.0;
    }
    (rec_ok / tot, mem_ok / tot)
}

/// Read the grown half-life of a trace (traces are public serialized fields — same helper as `substrate_growtime`).
fn learned_half_life(f: &Field, cell: usize) -> f32 {
    let v: serde_json::Value = serde_json::to_value(f).unwrap();
    v.get("traces")
        .and_then(|t| t.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|t| t.get("cell").and_then(|c| c.as_u64()) == Some(cell as u64))
                .and_then(|t| t.get("half_life").and_then(|h| h.as_f64()))
        })
        .map(|h| h as f32)
        .unwrap_or(0.0)
}

#[test]
fn a_grown_recurrent_belief_filter_integrates_a_partial_observation_over_event_time() {
    let ks = [1usize, 2, 4, 8, 16, 32];

    let (mut f, tr, pred) = belief_filter();
    train(&mut f, tr, pred, 8000, 32, 0xB0135EED);
    let grown_hl = learned_half_life(&f, tr);

    let rec_acc: Vec<f64> = ks
        .iter()
        .map(|&k| accuracy_at(&mut f, tr, pred, k, 4000, 0x5EED))
        .collect();
    let mem_acc: Vec<f64> = ks
        .iter()
        .map(|&k| accuracy_memoryless(k, 4000, 0x5EED))
        .collect();

    // (B) HOLD across a gap of pure-noise events — the delayed-cause representability (#171 req 1).
    let (rec_gap, mem_gap) = accuracy_after_gap(&mut f, tr, pred, 32, 8, 4000, 0x6A9);

    eprintln!("[BELIEVE] grown belief filter — half-life grown by adapt_scales: {HL0} → {grown_hl:.1} events (its filter-gain α, not authored)");
    eprintln!("[BELIEVE] (A) evidence integration over event-time (persistent hidden s, noisy binary sensor P_REVEAL={P_REVEAL}):");
    eprintln!("  events K :   1     2     4     8    16    32");
    eprint!("  RECURRENT :");
    for a in &rec_acc {
        eprint!("  {:.2}", a);
    }
    eprintln!("   ← the belief integrates evidence (rises with K)");
    eprint!("  MEMORYLESS:");
    for a in &mem_acc {
        eprint!("  {:.2}", a);
    }
    eprintln!("   ← flat: only ever the last observation");
    eprintln!(
        "[BELIEVE] (B) hold across a gap: build belief over 32 obs, then 8 PURE-NOISE events — RECURRENT {:.2}  MEMORYLESS {:.2}",
        rec_gap, mem_gap
    );
    eprintln!("[BELIEVE] VERDICT: a grown recursive belief filter integrates a noisy, partial observation of a PERSISTING hidden cause into a settled belief over event-time. The recurrence is the substrate's native leaky integrator (`integrate`) — the TREND/smoothed carried-state; its filter-gain (half-life) is GROWN by adapt_scales, so nothing is authored (a hand-set gain would BE the authored Kalman gain #171 forbids). Integrating over event-time lifts accuracy monotonically from the single-observation rate toward certainty (like `substrate_localize` over space), while the memoryless observer — integration depth 0, the same read with no carried state — is pinned at the single-observation rate for every K, and is blind the instant the cause moves into the past (B). HONEST DISTINCTION: `substrate_remember`'s grown latch is the EXACT/register flavor — it HOLDS a clean value but hard-latches (commits early) on a noisy one; a belief over noise is the TREND flavor, which is `integrate`. This is the perception/epistemic half of #171's one `argmin-G`: the posterior `q` the expected-information-gain term reduces. The fovea⇄purpose FUSION into one scalar stays gated on the ARC-3 loop. Nothing authored: no Kalman gain, no argmax, no hand-set retention — zero core change.");

    // The filter-gain (half-life) was GROWN well above its small seed — the integrator learned to average.
    assert!(
        grown_hl > 2.0 * HL0,
        "adapt_scales must grow the filter's half-life to integrate the persistent state (grown {grown_hl:.1} from {HL0})"
    );
    // (A) INTEGRATION: accuracy rises (non-decreasing) with evidence and reaches high confidence.
    for i in 1..rec_acc.len() {
        assert!(
            rec_acc[i] >= rec_acc[i - 1] - 0.02,
            "recurrent accuracy must rise (non-decreasing) with evidence: K={} {:.2} → K={} {:.2}",
            ks[i - 1],
            rec_acc[i - 1],
            ks[i],
            rec_acc[i]
        );
    }
    assert!(
        rec_acc[0] < 0.72,
        "a SINGLE observation must be weakly informative (K=1 {:.2}) — else there is nothing to integrate",
        rec_acc[0]
    );
    assert!(
        *rec_acc.last().unwrap() > 0.85,
        "integrating evidence must reach high confidence (K=32 {:.2})",
        rec_acc.last().unwrap()
    );
    assert!(
        rec_acc.last().unwrap() - rec_acc[0] > 0.18,
        "integration must lift accuracy substantially over a single observation (Δ={:.2})",
        rec_acc.last().unwrap() - rec_acc[0]
    );
    // CONTROL: the memoryless observer is flat and at the single-observation rate — it cannot integrate.
    let mem_max = mem_acc.iter().cloned().fold(0.0f64, f64::max);
    let mem_min = mem_acc.iter().cloned().fold(1.0f64, f64::min);
    assert!(
        mem_max - mem_min < 0.05,
        "memoryless accuracy must be FLAT in K (range {:.2}) — no integration",
        mem_max - mem_min
    );
    assert!(
        (mem_max as f32 - P_REVEAL).abs() < 0.04,
        "memoryless accuracy is pinned at the single-observation rate P_REVEAL={P_REVEAL} (max {mem_max:.2})"
    );
    assert!(
        rec_acc.last().unwrap() - mem_acc.last().unwrap() > 0.18,
        "the recurrent belief filter must STRICTLY beat the memoryless control (Δ={:.2})",
        rec_acc.last().unwrap() - mem_acc.last().unwrap()
    );
    // (B) HOLD: the recurrent belief survives the gap; the single-observation observer is at chance.
    assert!(
        rec_gap > 0.80 && rec_gap - mem_gap > 0.25,
        "the belief must persist across missing/uninformative events (recurrent {rec_gap:.2} vs memoryless {mem_gap:.2})"
    );
}
