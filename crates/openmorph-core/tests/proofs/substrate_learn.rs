// SPDX-License-Identifier: MIT-0
//! rc6 step 4 — the async organism LEARNS from a stream, per-event and local (no epoch). The morph
//! finds the RELATIONSHIP between two queryable spaces X → Y purely from experience: each sample is
//! one event (feed X, settle, delta-update toward the observed Y). This is grown transduction —
//! learning the relationship, not authoring a decoder — on the async substrate.
//!
//!   cargo test -p openmorph-core --test proofs substrate_learn -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const N_IN: usize = 4;

/// The (unknown to the morph) relationship Y = tanh(Σ w*·x) between space X (4 cells) and space Y (1).
fn truth(x: &[f32; N_IN]) -> f32 {
    const W: [f32; N_IN] = [0.8, -0.5, 0.3, 0.6];
    (0..N_IN).map(|k| W[k] * x[k]).sum::<f32>().tanh()
}

#[test]
fn the_async_morph_learns_a_relationship_between_two_spaces_from_a_stream() {
    // X = cells 0..4, prediction of Y = cell 4. One unit reads X, writes the prediction (weights 0).
    let mut f = Field::new(N_IN + 1);
    f.add_unit((0..N_IN).collect(), vec![0.0; N_IN], 0.0, N_IN);

    // STREAM of experience: each sample is ONE event — feed X, settle, learn toward the observed Y.
    let mut rng = Rng::new(0x5A11);
    let mut window = Vec::new();
    let mut acc = 0.0f32;
    let (mut first, mut last) = (0.0f32, 0.0f32);
    let total = 4000usize;
    for t in 0..total {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..N_IN).map(|k| (k, x[k])).collect();
        let e = f.learn_step(&inputs, N_IN, truth(&x), 0.1, 100);
        acc += e;
        if (t + 1) % 500 == 0 {
            window.push(acc / 500.0);
            acc = 0.0;
        }
    }
    first = *window.first().unwrap_or(&first);
    last = *window.last().unwrap_or(&last);

    // HELD-OUT: the learned unit predicts Y from X it never saw.
    let mut rng = Rng::new(0xBEEF);
    let mut mae = 0.0f32;
    for _ in 0..500 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..N_IN).map(|k| (k, x[k])).collect();
        f.stimulate(&inputs, 100);
        mae += (f.cells()[N_IN] - truth(&x)).abs();
    }
    mae /= 500.0;

    eprintln!(
        "[LEARN] async stream, per-event local delta rule (no epoch): mean|err| {:.3} → {:.3} over training; held-out MAE {:.3}",
        first, last, mae
    );
    eprintln!("[LEARN] VERDICT (rc6 step 4): the async morph LEARNED the relationship between space X and space Y from a stream of experience — one event per sample, plasticity local to each firing unit, no epoch and no tick. Finding the relationship IS the transduction (grown, not authored). This is the seed of the morph growing its own driver for any channel.");
    assert!(
        last < first * 0.5,
        "error must fall as the morph learns ({first:.3} → {last:.3})"
    );
    assert!(
        mae < 0.05,
        "the learned relationship must generalize to held-out X (MAE {mae:.3})"
    );
}

/// `learn_heads` — several heads, ONE event. (A) one head is exactly `learn_step`, bit for bit. (B) H heads
/// settle the shared inputs ONCE: a trace over an input ticks once per event, as after a single settle, where
/// `learn_step` per head re-settles and ticks it H times.
#[test]
fn learn_heads_is_one_event_for_many_heads() {
    let build = || {
        let mut f = Field::new(8);
        f.rng_seed(7);
        for out in 4..8 {
            f.add_unit(vec![0, 1, 2, 3], vec![0.1, -0.2, 0.3, 0.05], 0.0, out);
        }
        let trace = f.integrate(0, 4.0);
        (f, trace)
    };
    let inputs = [(0usize, 1.0f32), (1, 0.5), (2, -0.25), (3, 0.75)];
    let heads = [(4usize, 0.3f32), (5, -0.4), (6, 0.1), (7, 0.6)];
    let budget = 10_000;

    // (A) one head
    let (mut a, _) = build();
    let (mut b, _) = build();
    let ea = a.learn_heads(&inputs, &heads[..1], 0.1, budget);
    let eb = b.learn_step(&inputs, heads[0].0, heads[0].1, 0.1, budget);
    assert_eq!(ea[0].to_bits(), eb.to_bits());
    assert_eq!(a.snapshot(), b.snapshot(), "one head is exactly learn_step");

    // (B) one settle per event
    let (mut many, trace) = build();
    many.learn_heads(&inputs, &heads, 0.1, budget);
    let (mut once, _) = build();
    once.stimulate(&inputs, budget);
    let (mut per_head, _) = build();
    for &(cell, target) in &heads {
        per_head.learn_step(&inputs, cell, target, 0.1, budget);
    }
    let t = |f: &Field| f.cells()[trace];
    eprintln!(
        "[LEARN-HEADS] trace after one event: learn_heads {:.4}, one settle {:.4}, learn_step per head {:.4}",
        t(&many),
        t(&once),
        t(&per_head)
    );
    assert_eq!(
        t(&many).to_bits(),
        t(&once).to_bits(),
        "the trace ticks once per event"
    );
    assert!(
        t(&per_head) != t(&once),
        "learn_step per head ticks the trace once per head"
    );
}
