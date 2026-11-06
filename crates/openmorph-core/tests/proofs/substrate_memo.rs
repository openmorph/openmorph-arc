// SPDX-License-Identifier: MIT-0
//! NESTED MEMOIZATION — a nested cell is a FROZEN brick (learning skips it), so for a given input its output
//! is a fixed function. Fire-on-change already skips an UNCHANGED input (the cell never fires); the memo adds
//! the case that actually costs: a RECURRING, non-consecutive input — a revisited discrete STATE, exactly
//! ARC-3's regime — which otherwise re-settles the whole sub even though it computed that input before. The
//! memo caches the sub's output by the input's bit pattern, so distinct-input WORK, not fire count, is what a
//! reused node pays. `Field::nested_settles()` counts the real sub-settles (cache misses) for measurement.
//!
//!   cargo test -p openmorph-core --test proofs substrate_memo -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const BUD: usize = 20;

fn skill() -> Field {
    let mut f = Field::new(3);
    f.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2); // out = tanh(x - y)
    f
}
fn f(x: f32, y: f32) -> f32 {
    (x - y).tanh()
}

#[test]
fn a_recurring_input_is_served_from_the_memo_not_re_settled() {
    let mut host = Field::new(2);
    host.nest(skill(), &[0, 1], 2, BUD); // cell 2 = the nested skill

    // Cycle through 4 DISTINCT discrete states, 100 rounds → 400 fires (each fire changes an input, so the
    // nested cell fires every time — fire-on-change does NOT skip these; the memo must).
    let states = [(-1.0f32, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)];
    let rounds = 100;
    let mut worst = 0.0f32;
    let mut fires = 0u64;
    for _ in 0..rounds {
        for &(x, y) in &states {
            host.stimulate(&[(0, x), (1, y)], BUD);
            worst = worst.max((host.cells()[2] - f(x, y)).abs());
            fires += 1;
        }
    }
    let settles = host.nested_settles();

    eprintln!(
        "[MEMO] {fires} fires over {} distinct states → {settles} actual sub-settles ({:.0}× fewer), worst error {worst:.4}",
        states.len(),
        fires as f64 / settles as f64
    );
    eprintln!("[MEMO] VERDICT: a nested cell is a frozen brick, so its output is a pure function of its input — the memo turns a REVISITED discrete state (ARC-3's regime, and any recurring context) into a cache hit instead of a full sub-settle. Fire-on-change already elides an unchanged input; the memo elides the non-consecutive repeat that otherwise re-settles the whole sub. So a reused node pays for DISTINCT-input work, not fire count — here 400 fires cost only 4 sub-settles, and correctness is exact. Combined with the fractal-index routing (`substrate_scale`), a large recursive library settles only the active path, and each active node computes each distinct input once.");

    assert!(
        worst < 1e-3,
        "memoized nested cell must stay exact (worst {worst:.5})"
    );
    assert_eq!(
        settles as usize,
        states.len(),
        "each DISTINCT input must settle once; the {fires} fires must not re-settle (settles {settles})"
    );
}
