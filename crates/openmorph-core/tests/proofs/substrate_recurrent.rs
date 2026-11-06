// SPDX-License-Identifier: MIT-0
//! RECURRENT CAUSALITY — when the cause is in the PAST (MorphLab #115, the flagged-open temporal half).
//! `substrate_causal` grows the sufficient state when the disambiguating cause is present in the CURRENT
//! input (a spatial feature the residual births). But the deeper case is a HIDDEN cause: the thing that makes
//! the outcome predictable happened earlier and is GONE from the current observation. Then no spatial feature
//! over the present can split the aliasing — the state must carry MEMORY of the past value.
//!
//! The world: a cue `c = ±1` flashes on one cell at step 0, then that cell is 0 for `DELAY` steps; the
//! outcome at the final step is the cue `c`. The present observation at the final step is identical (0)
//! whatever `c` was — so a MEMORYLESS state is fully aliased. Three states, scored by STATE-DETERMINISM (the
//! #115 metric — fraction of (state,action) groups with a unique outcome):
//!   - MEMORYLESS (current obs): 0.0 — one group (obs=0), both outcomes. Predictor at chance.
//!   - RECENCY of the cue cell: still ~0 — the async organism's subjective-time trace (`recency`) says WHEN
//!     the cue fired (identical here, fixed delay) but NOT its VALUE/sign. This is the honest boundary of
//!     `substrate_time`: recency is a clock, not a memory of what happened.
//!   - MEMORY-OF-VALUE (a latch that holds `c`): 1.0 — the outcome is a function of it. Predictor 100%.
//!
//! So memory-of-value is exactly the missing ingredient, and the substrate CAN REPRESENT it: a recurrent unit
//! (reads its own cell + the cue, on the async medium) LATCHES the cue and holds it across the delay — Part C
//! demonstrates a wired latch retaining the sign. What is still OPEN (truth-not-trophy): GROWING that latch
//! autonomously from the residual — credit assignment THROUGH time — which the local per-event delta rule
//! does not do. The diagnosis is closed (memory-of-value, not recency); the growth of it is the frontier.
//!
//!   cargo test -p openmorph-core --test proofs substrate_recurrent -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::collections::HashMap;

use crate::common::Rng;

const DELAY: usize = 6;

/// STATE-DETERMINISM (the #115 metric): fraction of state-key groups whose outcome is unique.
fn determinism(obs: &[(i64, i64)]) -> f64 {
    let mut groups: HashMap<i64, Vec<i64>> = HashMap::new();
    for &(s, o) in obs {
        groups.entry(s).or_default().push(o);
    }
    let unique = groups
        .values()
        .filter(|v| v.iter().all(|&o| o == v[0]))
        .count();
    unique as f64 / groups.len().max(1) as f64
}

#[test]
fn the_cause_in_the_past_needs_memory_of_value_not_recency() {
    let mut rng = Rng::new(0x5EC0DE);

    // ── Part A: STATE-DETERMINISM of the three candidate states. ──
    let (mut memoryless, mut recency_state, mut memory_state) =
        (Vec::new(), Vec::new(), Vec::new());
    // A tiny field just to measure recency of the cue cell honestly through the async clock.
    for _ in 0..2000 {
        let c = rng.bit();
        let mut f = Field::new(2); // cell 0 = cue, cell 1 = spare
        f.stimulate(&[(0, c)], 8); // cue flashes
        for _ in 0..DELAY {
            f.stimulate(&[(0, 0.0)], 8); // cue gone; clock advances
        }
        let o = c as i64;
        // MEMORYLESS: the current observation of the cue cell (always 0 at the final step).
        memoryless.push((0i64, o));
        // RECENCY: the subjective-time trace of the cue cell — quantized. Carries time, not sign.
        let r = f.recency(0, DELAY as f32);
        recency_state.push(((r * 100.0) as i64, o));
        // MEMORY-OF-VALUE: a state that held the cue's sign.
        memory_state.push((c as i64, o));
    }
    let d_mem0 = determinism(&memoryless);
    let d_rec = determinism(&recency_state);
    let d_mem = determinism(&memory_state);

    // ── Part B: predictors — memoryless (chance) vs memory-of-value (100%). ──
    let mut memoryless_p = Field::new(2);
    memoryless_p.add_unit(vec![0], vec![0.0], 0.0, 1); // out = tanh(w·obs)
    let mut memory_p = Field::new(2);
    memory_p.add_unit(vec![0], vec![0.0], 0.0, 1); // out = tanh(w·mem)
    let mut rng = Rng::new(0x1A7C4);
    for _ in 0..8000 {
        let c = rng.bit();
        memoryless_p.learn_step(&[(0, 0.0)], 1, c, 0.1, 8); // present obs is 0 → no signal
        memory_p.learn_step(&[(0, c)], 1, c, 0.1, 8); // the held value → learnable
    }
    let (mut m0_ok, mut mem_ok, mut tot) = (0.0f64, 0.0, 0.0);
    for _ in 0..2000 {
        let c = rng.bit();
        memoryless_p.stimulate(&[(0, 0.0)], 8);
        if memoryless_p.cells()[1].signum() == c {
            m0_ok += 1.0;
        }
        memory_p.stimulate(&[(0, c)], 8);
        if memory_p.cells()[1].signum() == c {
            mem_ok += 1.0;
        }
        tot += 1.0;
    }
    let (m0_acc, mem_acc) = (m0_ok / tot, mem_ok / tot);

    // ── Part C: the substrate CAN REPRESENT the memory — a recurrent latch holds the cue across the delay. ──
    // A unit reads [cue, self] and writes self; a strong self-weight makes it a bistable latch: the cue loads
    // it, and when the cue returns to 0 the self-connection holds the sign. Growing these weights from the
    // residual is the open frontier; that the async medium can HOLD state is not.
    let mut held_ok = 0.0f64;
    let (cue, mem) = (0usize, 1usize);
    for _ in 0..200 {
        let c = rng.bit();
        let mut f = Field::new(2);
        f.add_unit(vec![cue, mem], vec![4.0, 4.0], 0.0, mem); // latch: tanh(4·cue + 4·self)
        f.stimulate(&[(cue, c)], 16); // load
        for _ in 0..DELAY {
            f.stimulate(&[(cue, 0.0)], 16); // cue gone — the latch must hold
        }
        if f.cells()[mem].signum() == c {
            held_ok += 1.0;
        }
    }
    let held = held_ok / 200.0;

    eprintln!(
        "[RECURRENT] the cause is in the PAST (cue flashes, then gone {DELAY} steps, outcome = the cue):\n  STATE-DETERMINISM  memoryless {d_mem0:.2}  recency(time-not-value) {d_rec:.2}  memory-of-value {d_mem:.2}\n  PREDICTS outcome   memoryless {:.0}%  vs  memory-of-value {:.0}%\n  substrate CAN HOLD the value: recurrent latch retains the sign across the delay {:.0}%",
        m0_acc * 100.0,
        mem_acc * 100.0,
        held * 100.0,
    );
    eprintln!("[RECURRENT] VERDICT: when the disambiguating cause happened in the PAST, no spatial feature over the present can split the aliasing — the state needs MEMORY OF THE PAST VALUE. State-determinism localizes it exactly: memoryless 0.0, memory-of-value 1.0. Crucially recency ~0.0 too — the async organism's subjective-time trace (substrate_time) says WHEN the cue fired, not WHAT it was; a clock is not a memory. The substrate can REPRESENT the memory (a recurrent latch on the async medium holds the sign across the delay, Part C), so the mechanism exists. OPEN (truth-not-trophy): GROWING that latch autonomously — credit assignment through time — is not done by the local per-event delta rule; that recurrent-growth is the real frontier of #115. This proof closes the DIAGNOSIS (memory-of-value, not recency) and marks the growth honestly.");

    assert!(
        d_mem0 < 0.5,
        "memoryless state must be aliased (det {d_mem0:.2})"
    );
    assert!(
        d_rec < 0.5,
        "recency carries time not value — still aliased (det {d_rec:.2})"
    );
    assert!(
        d_mem > 0.99,
        "memory-of-value must be Markov (det {d_mem:.2})"
    );
    assert!(
        m0_acc < 0.6,
        "memoryless predictor is at chance (acc {m0_acc:.2})"
    );
    assert!(
        mem_acc > 0.95,
        "memory-of-value predictor recovers the outcome (acc {mem_acc:.2})"
    );
    assert!(
        held > 0.95,
        "the recurrent latch must hold the past value across the delay (held {held:.2})"
    );
}
