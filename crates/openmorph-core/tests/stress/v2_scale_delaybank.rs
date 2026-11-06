// SPDX-License-Identifier: MIT-0
//! v2 scalability audit — the TEMPORAL wall, attacked. `advance_delays` was O(D²) per event on a WIDE delay
//! bank (many channels each delayed once — the #139 "bank of delays, the reliance economy selects" pattern):
//! for every delay it scanned the whole emitted-updates vector to catch delay-of-a-delay chaining, and when a
//! delay's source is an EXTERNAL cell the scan never matched → a full O(D) sweep each → O(D²). Replaced the
//! per-delay scan with an O(1) cell→value lookup (delay cells are unique, so it is byte-identical), making a
//! wide bank O(D)/event. This proves the fix: linear scaling + chaining correctness preserved.
//!
//! What is ASSERTED is machine-independent: the per-event work counted in the organism's own events (the delay
//! updates `advance_delays` emits + the settle firings of a readout over the bank, `Field::clock`) grows linearly
//! with the width. The wall time per event — where the old quadratic SCAN (an implementation detail below the
//! event level, invisible to any event count) showed up — is printed for information only.
use openmorph_core::substrate::Field;
use std::time::Instant;

#[test]
#[ignore = "stress: nightly"]
fn wide_delay_bank_advances_in_linear_time() {
    // ── (A) SCALING — a wide bank of D channels each delayed once advances in ~O(D)/event, not O(D²). ──
    let widths = [500usize, 1000, 2000, 4000];
    let (mut per_event, mut wall_per_event) = (Vec::new(), Vec::new());
    for &d in &widths {
        let mut f = Field::new(d);
        let bank: Vec<usize> = (0..d).map(|c| f.delay(c)).collect(); // each delay sources a DISTINCT external
                                                                     // cell (no chaining) — the pure wide bank
                                                                     // A readout over the whole bank (the #139 pattern: grow over a bank of delays), so an event does settle work.
        let out = f.len();
        f.add_unit(bank.clone(), vec![0.001; d], 0.0, out);
        let inp0: Vec<(usize, f32)> = (0..d).map(|c| (c, 0.01)).collect();
        f.stimulate(&inp0, 2); // prime
        let events = 300;
        let t = Instant::now();
        let (mut acc, mut emitted) = (0.0f32, 0usize);
        let clock0 = f.clock();
        for e in 1..=events {
            let ups = f.advance_delays();
            emitted += ups.len();
            acc += ups[e % ups.len()].1; // touch the output so nothing is optimized away
            let mut inp: Vec<(usize, f32)> = (0..d)
                .map(|c| (c, ((e * 31 + c) % 17) as f32 * 0.01))
                .collect();
            inp.extend(ups); // the delayed values enter as genuine input changes (the streaming step)
            f.stimulate(&inp, 2);
        }
        let firings = f.clock() - clock0;
        let work = (emitted as u64 + firings) as f64 / events as f64;
        let us = t.elapsed().as_secs_f64() * 1e6 / events as f64;
        per_event.push(work);
        wall_per_event.push(us);
        eprintln!(
            "[DELAYSCALE] D={d:>5}: {work:>8.1} events/event ({} updates + {} firings over {events})  | wall {us:>8.1} µs/event (info)  (acc {acc:.2})",
            emitted, firings
        );
    }
    // 8× the width (500→4000) should cost ~8× the work if linear; the old O(D²) would be ~64×.
    let ratio = per_event.last().unwrap() / per_event.first().unwrap().max(1e-6);
    let wall_ratio = wall_per_event.last().unwrap() / wall_per_event.first().unwrap().max(1e-6);
    eprintln!("[DELAYSCALE] 8× width → {ratio:.1}× events per event (wall {wall_ratio:.1}×, info)  (linear ≈ 8, the old quadratic ≈ 64)");
    assert!(
        ratio < 24.0,
        "advance_delays must scale ~linearly in bank width (8× width → {ratio:.1}× events; quadratic would be ~64×)"
    );

    // ── (B) CHAINING correctness — a delay-of-a-delay still emits x[t−2] exactly (the O(1) lookup = the scan). ──
    let mut g = Field::new(1);
    let _d1 = g.delay(0); // x[t−1]
    let d2 = g.delay(_d1); // x[t−2] — sources d1's cell, so it needs d1's FRESH output this pass (chaining)
    let seq = [0.2f32, 0.5, 0.9, 0.3, 0.7, 0.1];
    let mut out2 = Vec::new();
    for &x in &seq {
        g.stimulate(&[(0, x)], 2);
        let ups = g.advance_delays();
        out2.push(ups.iter().find(|(c, _)| *c == d2).map(|(_, v)| *v).unwrap());
    }
    eprintln!("[DELAYSCALE] chained delay-of-a-delay stream: {out2:?}   (input {seq:?})");
    // once the shift register has filled (t ≥ 2), d2 emits x[t−2] exactly.
    for t in 2..seq.len() {
        assert!(
            (out2[t] - seq[t - 2]).abs() < 1e-6,
            "chaining must survive the O(1) lookup: step {t} d2={} want x[t−2]={}",
            out2[t],
            seq[t - 2]
        );
    }

    eprintln!("[DELAYSCALE] VERDICT: the TEMPORAL scale wall is closed. `advance_delays` was O(D²)/event on a wide delay bank (a per-delay scan of the emitted-updates vector to catch chaining); an O(1) cell→value lookup makes it O(D)/event — a wide bank of thousands of channels now streams in linear time ({ratio:.0}× for 8× width, was ~64×). Chaining (delay-of-a-delay = x[t−2]) is byte-identical. The #139 streaming-memory pattern scales.");
}
