// SPDX-License-Identifier: MIT-0
//! WORKING MEMORY / BELIEF STATE — reward from the unobserved past (Gap 3, `docs/research/arc3-gaps-to-100.md`).
//!
//! THE GAP (MorphBot wall-map: ~17 games). Some ARC-3 games are beyond-Markov: the reward depends on a PAST state
//! the current observation does not contain (a hidden counter, a switch pressed earlier, a delay-line). The grown
//! discrete state-key (#264) is built from the CURRENT frame, so it cannot hold that history — a memoryless policy
//! is blind to it. Prove that gap, then close it with the substrate's native carried state.
//!
//! THE MECHANISM (nature-based — the native unit-delay register #139). `delay(source)` grows a cell that holds the
//! PREVIOUS value of `source`; `advance_delays()` shifts it once per external event. The carried delay cell IS the
//! belief variable: feed it (alongside the current observation) into the same grown reward predictor, and the past
//! becomes readable. No backprop-through-time, no window — a native register on the async medium.
//!
//! THE CLAIM (Gap 3 acceptance). On a task whose reward is determined by the PREVIOUS symbol (absent from the
//! current observation), a MEMORYLESS predictor is at chance, while a predictor that reads the carried `delay` state
//! recovers it (~100%).
//!
//! TRUTH-NOT-TROPHY. Balanced two-symbol stream (reward = the previous symbol, INDEPENDENT of the current one), so
//! the memoryless ceiling is exactly chance (0.5) — no class-imbalance inflation. The stream, seed, and rule are
//! authored and identical across arms; only whether the predictor reads the carried state differs. ZERO
//! openmorph-core change: a usage protocol over `delay`/`advance_delays` + `add_unit`/`learn_step`. Wiring the
//! carried belief into the #264 state-key on real games is MorphBot's live step.
//!
//!   cargo test -p openmorph-core --test proofs substrate_belief_state -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

/// Train a reward predictor on a balanced ±1 symbol stream whose reward = the PREVIOUS symbol (independent of the
/// current one). `with_memory` gives the predictor the carried `delay` state (previous symbol) in addition to the
/// current symbol. Returns held-out reward-prediction accuracy.
fn run(with_memory: bool, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let mut f = Field::new(1); // cell 0 = the current symbol (±1)
    let delay_cell = f.delay(0); // a native unit-delay register holding the PREVIOUS symbol (x[t−1])
    let reward_cell = delay_cell + 1;
    let reads: Vec<usize> = if with_memory {
        vec![0, delay_cell] // current + carried past
    } else {
        vec![0] // memoryless: current only
    };
    let nr = reads.len();
    f.add_unit(reads, vec![0.0; nr], 0.0, reward_cell);

    let sym = |r: &mut Rng| if r.u() & 1 == 0 { 1.0f32 } else { -1.0f32 };
    let bud = 20;

    // Advance one symbol through the stream, leaving cell 0 = current and `delay_cell` = previous.
    let advance = |f: &mut Field, cur: f32| {
        f.stimulate(&[(0, cur)], bud); // apply x[t]
        let mem = f.advance_delays(); // shift; emits x[t−1]
        f.stimulate(&mem, bud); // reflect the delayed value in the medium (delay_cell = x[t−1])
    };

    let mut prev = sym(&mut rng);
    advance(&mut f, prev);

    // TRAIN — reward is the PREVIOUS symbol (absent from the current observation).
    for _ in 0..4000 {
        let cur = sym(&mut rng);
        advance(&mut f, cur); // cell 0 = cur, delay_cell = prev
        f.learn_step(&[(0, cur)], reward_cell, prev, 0.05, bud);
        prev = cur;
    }

    // TEST (held-out stream): predict reward = the previous symbol.
    let (mut correct, mut total) = (0usize, 0usize);
    for _ in 0..1000 {
        let cur = sym(&mut rng);
        advance(&mut f, cur);
        let pred = f.cells()[reward_cell];
        if pred.signum() == prev.signum() {
            correct += 1;
        }
        total += 1;
        prev = cur;
    }
    correct as f32 / total as f32
}

#[test]
fn carried_delay_state_recovers_the_unobserved_past() {
    let seeds = [0x61u64, 0x62, 0x63, 0x64];
    let mem = seeds.iter().map(|&s| run(true, s)).sum::<f32>() / seeds.len() as f32;
    let nomem = seeds.iter().map(|&s| run(false, s)).sum::<f32>() / seeds.len() as f32;

    // The memoryless predictor CANNOT beat chance (reward is the previous, unobserved symbol).
    assert!(
        nomem < 0.6,
        "memoryless predictor must be ~chance on the beyond-Markov task: {nomem:.3}"
    );
    // The carried delay state recovers it almost perfectly.
    assert!(
        mem > 0.95,
        "carried delay state must recover the unobserved past: {mem:.3}"
    );

    println!("\n  WORKING MEMORY / BELIEF STATE (reward = the previous, unobserved symbol):");
    println!("    MEMORYLESS (current obs only) : {nomem:.3}  (≈ chance — beyond-Markov, blind to the past)");
    println!("    CARRIED delay state (#139)    : {mem:.3}  (the native register holds the past)");
    println!("  ⇒ the belief variable is the substrate's own carried state; wire it into the #264 key on real games.\n");
}
