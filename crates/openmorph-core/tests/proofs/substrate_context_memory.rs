// SPDX-License-Identifier: MIT-0
//! CONTEXT IS MEMORY — the exact/REGISTER flavor. Context and memory are not two mechanisms: the context a
//! morph conditions on is its own carried state, held in cells. This proves the EXACT flavor (a latch that
//! HOLDS a discrete context across absence) — the one an EMA cannot do (an `integrate` trace decays; it works
//! for slowly-varying/persistent context but its magnitude collapses when the signal is sparse).
//!
//! The context is SET by a sparse marker at each switch and must be HELD across the (mostly blank) block. A
//! bistable self-recurrent cell — `L ← tanh(HOLD·L + SET·marker)` with `HOLD > 1` so ±1 are stable attractors
//! — latches the marker and holds it at FULL magnitude until the next one (a flip-flop; how nature holds state,
//! e.g. `substrate_recurrent`/`substrate_remember`). Two latches at two rates give HIERARCHICAL context (slow
//! domain, fast sub). One `alive()` channel conditions on the held registers and learns all four
//! context-gated rules (94–99%); the raw-marker baseline (no memory) sees 0 between switches and goes
//! context-blind. Composes existing primitives — a recurrent unit + `consolidate` + `grow`/`alive` — NO new core.
//!
//! (The TREND flavor is the twin: an `integrate` trace conditions just as well on PERSISTENT context. Together
//! they are the two flavors of carried state the substrate already has — trend + exact — now shown to BE the
//! context substrate. Nothing is a separate "context cell".)
//!
//!   cargo test -p openmorph-core --test proofs substrate_context_memory -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;
use openmorph_core::Event;

use crate::common::Rng;

const X0: usize = 0;
const X1: usize = 1;
const DOM_M: usize = 2; // domain marker (±1 only at a domain switch, else 0)
const SUB_M: usize = 3; // sub marker
const DOM_L: usize = 4; // domain latch (holds the domain between markers)
const SUB_L: usize = 5; // sub latch

fn rule(dom: bool, sub: bool, x0: f32, x1: f32) -> f32 {
    let v = match (dom, sub) {
        (false, false) => x0,
        (false, true) => x1,
        (true, false) => -x0,
        (true, true) => -x1,
    };
    if v > 0.0 {
        1.0
    } else {
        -1.0
    }
}

const DOM_BLOCK: usize = 400;
const SUB_BLOCK: usize = 100;

fn train_and_eval(memory: bool) -> [f64; 4] {
    let mut f = Field::new(6);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 40;
    if memory {
        // Bistable latches: L ← tanh(HOLD·L + SET·marker). HOLD > 1 makes ±1 stable attractors (holds); a
        // marker (large SET weight) flips it. Consolidate so the alive economy can never cull the register.
        f.add_unit(vec![DOM_L, DOM_M], vec![2.5, 4.0], 0.0, DOM_L);
        f.add_unit(vec![SUB_L, SUB_M], vec![2.5, 4.0], 0.0, SUB_L);
        f.consolidate(&[DOM_L, SUB_L]);
    }
    let y = f.len();
    f.ensure_cells(y + 1);

    let build = |f: &Field, x0: f32, x1: f32, md: f32, ms: f32| -> Vec<(usize, f32)> {
        if memory {
            // pass the latch cells at their CURRENT value (no reset); the marker drives the latch this event
            let (dl, sl) = (f.cells()[DOM_L], f.cells()[SUB_L]);
            vec![
                (X0, x0),
                (X1, x1),
                (DOM_M, md),
                (SUB_M, ms),
                (DOM_L, dl),
                (SUB_L, sl),
            ]
        } else {
            vec![(X0, x0), (X1, x1), (DOM_M, md), (SUB_M, ms)]
        }
    };
    let marker = |i: usize, block: usize, v: bool| -> f32 {
        if i.is_multiple_of(block) {
            if v {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        }
    };

    let mut rng = Rng::new(0x3033);
    for i in 0..48_000usize {
        let (dom, sub) = ((i / DOM_BLOCK) % 2 == 1, (i / SUB_BLOCK) % 2 == 1);
        let (x0, x1) = (rng.sym(), rng.sym());
        let (md, ms) = (marker(i, DOM_BLOCK, dom), marker(i, SUB_BLOCK, sub));
        let inp = build(&f, x0, x1, md, ms);
        f.alive(Event::Perceive {
            inputs: &inp,
            target_cell: y,
            target: rule(dom, sub, x0, x1),
            lr: 0.05,
            budget: 200,
        });
    }

    // Eval each context: SET it with one marker, then hold through blank queries.
    let mut acc = [0.0f64; 4];
    for (k, &(dom, sub)) in [(false, false), (false, true), (true, false), (true, true)]
        .iter()
        .enumerate()
    {
        let mut rng = Rng::new(0xE0 + k as u64);
        // one marker event sets both latches to this context
        let (md, ms) = (if dom { 1.0 } else { -1.0 }, if sub { 1.0 } else { -1.0 });
        let inp = build(&f, rng.sym(), rng.sym(), md, ms);
        f.stimulate(&inp, 200);
        let mut c = 0usize;
        for _ in 0..1000 {
            let (x0, x1) = (rng.sym(), rng.sym());
            let inp = build(&f, x0, x1, 0.0, 0.0); // NO marker — the register must hold the context
            f.stimulate(&inp, 200);
            let pred = if f.cells()[y] > 0.0 { 1.0 } else { -1.0 };
            if (pred - rule(dom, sub, x0, x1)).abs() < 1e-6 {
                c += 1;
            }
        }
        acc[k] = c as f64 / 1000.0;
    }
    acc
}

#[test]
fn context_is_register_memory() {
    let names = ["(L,0) +x0", "(L,1) +x1", "(R,0) -x0", "(R,1) -x1"];
    let mem = train_and_eval(true);
    let raw = train_and_eval(false);
    for k in 0..4 {
        eprintln!(
            "[LATCH] {:<9} latch(register memory) {:.0}%   raw-marker(no memory) {:.0}%",
            names[k],
            mem[k] * 100.0,
            raw[k] * 100.0
        );
    }
    let mem_avg = mem.iter().sum::<f64>() / 4.0;
    let raw_avg = raw.iter().sum::<f64>() / 4.0;
    eprintln!("[LATCH] VERDICT: a consolidated bistable latch HOLDS the sparse context marker at full magnitude across the blank block; one alive() channel conditions on the held register (avg {:.0}%) while raw markers (no memory) go context-blind (avg {:.0}%). Context IS memory — the exact/register flavor. Two latches at two rates = hierarchical context. No new core.", mem_avg * 100.0, raw_avg * 100.0);
    for k in 0..4 {
        assert!(
            mem[k] > 0.85,
            "the register must hold context {} across the blank block (got {:.2})",
            names[k],
            mem[k]
        );
    }
    assert!(
        mem_avg - raw_avg > 0.30,
        "register memory must do the distinctive context-carrying work vs no-memory (avg {:.2} vs {:.2})",
        mem_avg,
        raw_avg
    );
}
