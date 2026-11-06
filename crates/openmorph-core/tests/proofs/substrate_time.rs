// SPDX-License-Identifier: MIT-0
//! TIME PLACEMENT — timing is information the state alone can't carry. Because the substrate is async
//! (fire-on-change, no global tick), *when* a cell changed is meaningful: two moments with the SAME
//! value but different timing are genuinely different situations. The organism's clock is its own
//! SUBJECTIVE time — a deterministic, machine-independent count of events it has processed (never a
//! wall-clock, which varies by computer and breaks replay). Each cell is stamped with the event it
//! last changed at; `recency` is the bounded, sense-able trace of that age.
//!
//! The proof: a metronome cell flips every P steps. From its VALUE alone you cannot predict the flip —
//! every step inside a period looks identical (the value is constant). But its RECENCY encodes the
//! phase (time since the last flip), so the SAME predictor, given the recency, nails it. Timing is the
//! missing information — an axis only the async substrate has.
//!
//!   cargo test -p openmorph-core --test proofs substrate_time -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const P: usize = 4; // the metronome period
const HL: f32 = 3.0; // recency half-life (in events)

/// The metronome's value at step `t`: flips every P steps.
fn c_at(t: usize) -> f32 {
    if (t / P).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// A one-unit predictor of "will the metronome flip next step?". `timed` gives it the recency cell (1)
/// in addition to the value cell (0); otherwise it sees only the value.
fn build(timed: bool) -> (Field, usize) {
    let mut f = Field::new(2);
    let reads: Vec<usize> = if timed { vec![0, 1] } else { vec![0] };
    let pred = 2;
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, pred);
    (f, pred)
}

/// Online accuracy over the last window: predict the flip, then learn. Deterministic (no RNG).
fn run(timed: bool) -> f64 {
    let (mut f, pred) = build(timed);
    let total = 8000;
    let eval_from = total - 400;
    let (mut correct, mut seen) = (0usize, 0usize);
    for t in 0..total {
        let c = c_at(t);
        f.stimulate(&[(0, c)], 50); // feed the metronome; stamps changed_at[0] on flips, ticks the clock
        let rec = f.recency(0, HL); // the substrate's own timing signal
        let inp: Vec<(usize, f32)> = if timed {
            vec![(0, c), (1, rec)]
        } else {
            vec![(0, c)]
        };
        let flip = if c_at(t + 1) != c { 1.0 } else { -1.0 };
        if t >= eval_from {
            f.stimulate(&inp, 50);
            let predicted = if f.cells()[pred] > 0.0 { 1.0 } else { -1.0 };
            if predicted == flip {
                correct += 1;
            }
            seen += 1;
        }
        f.learn_step(&inp, pred, flip, 0.05, 50);
    }
    correct as f64 / seen as f64
}

#[test]
fn recency_reveals_rhythm_the_value_alone_cannot() {
    let state_only = run(false);
    let timed = run(true);

    eprintln!(
        "[TIME] predict a metronome flip (period {P}); the value is constant within a period, so:\n  STATE ONLY (value)        : {:.0}%  (can only predict 'never flip' ≈ {:.0}%)\n  + RECENCY (subjective time): {:.0}%",
        state_only * 100.0,
        100.0 * (P - 1) as f64 / P as f64,
        timed * 100.0,
    );
    eprintln!("[TIME] VERDICT: because the substrate is async (fire-on-change, no tick), WHEN a cell changed is information the value can't carry. Time is the organism's OWN event count — deterministic, machine-independent, no wall-clock — so `recency` is byte-replayable. Given it, the SAME predictor learns the rhythm and predicts the flip; without it, it can only guess the majority. Timing is a second axis of prediction, unique to the async organism.");

    assert!(
        state_only < 0.85,
        "value alone cannot predict the phase (got {state_only:.2})"
    );
    assert!(
        timed >= 0.95,
        "recency should reveal the rhythm (got {timed:.2})"
    );
}
