// SPDX-License-Identifier: MIT-0
//! TEMPORAL WORLD MODEL — composing subjective time INTO intrinsic prediction (rc9). `observe_from`
//! lets the world model's context hold more than the observed value — here the `recency` of the
//! observed cell (its subjective-time trace). So the SAME target-free predictive coding models a
//! TIME-DEPENDENT world: a metronome whose next value the state alone can't predict (it's constant
//! within a period) but whose recency reveals the phase. With time in the context it nails the rhythm;
//! without, it can only guess. Prediction, perception, and time are one grown mechanism.
//!
//!   cargo test -p openmorph-core --test proofs substrate_temporal -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const P: usize = 4; // metronome period
const K: usize = 32; // world-model backbone width
const HL: f32 = 32.0; // recency half-life (events)
const BUD: usize = 200;

fn c_at(t: usize) -> f32 {
    if (t / P).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Build a world model predicting the metronome's next value. `timed` adds the recency cell (1) to the
/// context; otherwise it sees only the value cell (0).
fn build(timed: bool, seed: u64) -> (Field, usize, Vec<usize>) {
    let mut f = Field::new(2); // cell 0 = value X, cell 1 = recency of X
    f.rng_seed(seed ^ 0x71ED);
    let context: Vec<usize> = if timed { vec![0, 1] } else { vec![0] };
    let mut reads = context.clone();
    reads.extend(f.grow_backbone(&context, K, 2.5));
    let pred = f.len();
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, pred);
    (f, pred, context)
}

/// Model the metronome world through `observe_from` (target-free); return online next-value accuracy.
fn run(timed: bool, seed: u64) -> f64 {
    let (mut f, pred, context) = build(timed, seed);
    let total = 8000;
    let eval_from = total - 400;
    let (mut correct, mut seen) = (0usize, 0usize);
    let mut x = c_at(0);
    f.stimulate(&[(0, x)], BUD);
    for t in 0..total {
        let rec = f.recency(0, HL); // the substrate's own timing signal for cell 0
        let mut ctx_in = vec![(0, x)];
        if timed {
            ctx_in.push((1, rec));
        }
        f.stimulate(&ctx_in, BUD); // present the context (value, and recency if timed)
        let nx = c_at(t + 1);
        if t >= eval_from {
            let predicted = if f.cells()[pred] > 0.0 { 1.0 } else { -1.0 };
            if predicted == nx {
                correct += 1;
            }
            seen += 1;
        }
        f.observe_from(&context, 0, pred, 1, &[nx], 0.05, BUD); // learn [x,(rec)] -> nx, take nx in
        x = nx;
    }
    correct as f64 / seen as f64
}

#[test]
fn observe_models_a_time_dependent_world_with_recency() {
    let state_only = run(false, 0x0C);
    let timed = run(true, 0x0C);

    eprintln!(
        "[TEMPORAL] model a metronome world (period {P}) target-free via observe_from:\n  context = value only     : {:.0}%  (constant within a period → ~{:.0}% majority)\n  context = value + recency : {:.0}%",
        state_only * 100.0,
        100.0 * (P - 1) as f64 / P as f64,
        timed * 100.0,
    );
    eprintln!("[TEMPORAL] VERDICT (rc9): observe_from lets the intrinsic world model read the subjective-time trace (recency) alongside the value, so the SAME target-free predictive coding models a TIME-DEPENDENT world — it predicts WHEN the metronome flips, which the value alone can't. Time composes into prediction: perception, prediction, and time placement are one grown mechanism over a shared basis.");

    assert!(
        state_only < 0.85,
        "value alone cannot predict the phase (got {state_only:.2})"
    );
    assert!(
        timed >= 0.95,
        "recency in the context should let the world model predict the rhythm (got {timed:.2})"
    );
}
