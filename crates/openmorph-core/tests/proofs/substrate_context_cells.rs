// SPDX-License-Identifier: MIT-0
//! HIERARCHICAL CONTEXT CELLS — the RIGHT way to give a morph context (the LLM-Morph shape). The context is a
//! first-class part of the observation: context cells at two levels (a coarse DOMAIN and a fine SUB-context),
//! so `y = f(task_inputs, context)` is a WELL-DEFINED FUNCTION. With the context present, ONE `alive()` channel
//! learns all four context-conditioned rules — including the nonlinear XOR/XNOR and the adjacent contexts that
//! an inferred bare-cue recognizer collapsed on — with NO recruiter, NO separate channels, NO disambiguation.
//!
//! This is the good-practice counterpart to the pathological toys: forcing a morph to disambiguate identical
//! inputs mapped to CONFLICTING targets (context withheld, or crammed into one ambiguous scalar) is exactly
//! the setup a living organism cannot behave well in — the mapping is not a function without the context.
//! Provide the context, and hierarchical context-conditioning falls out of `grow` on one channel. (Explicit
//! channels-per-skill still work and isolate harder; context cells are how you SCALE past a channel per
//! context — condition on running context at multiple scales, one output, as an LLM-like Morph must.)
//!
//!   cargo test -p openmorph-core --test proofs substrate_context_cells -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;
use openmorph_core::Event;

use crate::common::Rng;

// inputs
const X0: usize = 0;
const X1: usize = 1;
const DOM: usize = 2; // coarse context cell (domain)
const SUB: usize = 3; // fine context cell (sub-context within the domain)
const Y: usize = 4; // ONE shared output channel

// Four contexts in a 2-level tree, tagged by (dom, sub) ∈ {−1,+1}²:
//   (−,−) sign(x0)   (−,+) sign(x1)   (+,−) XOR   (+,+) XNOR
fn target(dom: f32, sub: f32, x0: f32, x1: f32) -> f32 {
    let yes = match (dom > 0.0, sub > 0.0) {
        (false, false) => x0 > 0.0,
        (false, true) => x1 > 0.0,
        (true, false) => x0 * x1 > 0.0,
        (true, true) => x0 * x1 < 0.0,
    };
    if yes {
        1.0
    } else {
        -1.0
    }
}

const CTX: [(f32, f32); 4] = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)];
const NAME: [&str; 4] = ["sign(x0)", "sign(x1)", "XOR", "XNOR"];

fn eval(f: &mut Field, ctx: usize, budget: usize, seed: u64) -> f64 {
    let (dom, sub) = CTX[ctx];
    let mut rng = Rng::new(seed);
    let mut c = 0usize;
    for _ in 0..1000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(X0, x0), (X1, x1), (DOM, dom), (SUB, sub)], budget);
        let pred = if f.cells()[Y] > 0.0 { 1.0 } else { -1.0 };
        if (pred - target(dom, sub, x0, x1)).abs() < 1e-6 {
            c += 1;
        }
    }
    c as f64 / 1000.0
}

#[test]
fn hierarchical_context_cells_one_channel() {
    let budget = 300;
    let mut f = Field::new(5);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 40;
    let mut rng = Rng::new(0x2025);
    // GOOD PRACTICE: the context is always present as input, and the four contexts are interleaved.
    for i in 0..48000usize {
        let (dom, sub) = CTX[i % 4];
        let (x0, x1) = (rng.sym(), rng.sym());
        f.alive(Event::Perceive {
            inputs: &[(X0, x0), (X1, x1), (DOM, dom), (SUB, sub)],
            target_cell: Y,
            target: target(dom, sub, x0, x1),
            lr: 0.05,
            budget,
        });
    }
    let acc: Vec<f64> = (0..4)
        .map(|c| eval(&mut f, c, budget, 0x11 ^ c as u64))
        .collect();
    eprint!("[CTX] one alive() channel, hierarchical context cells: ");
    for c in 0..4 {
        eprint!("{}={:.0}% ", NAME[c], acc[c] * 100.0);
    }
    eprintln!("(hidden {})", f.n_units().saturating_sub(1));
    eprintln!("[CTX] VERDICT: giving the context as first-class hierarchical cells makes y=f(inputs,context) a well-defined function, so ONE alive() channel conditions on it and learns all four rules (incl. nonlinear XOR/XNOR) — no recruiter, no per-skill channels, no disambiguation. This is how context SCALES for an LLM-like Morph: condition on running context at multiple levels.");
    for c in 0..4 {
        assert!(
            acc[c] > 0.82,
            "context {} not learned ({:.2}) — hierarchical context cells should make it a clean function",
            NAME[c],
            acc[c]
        );
    }
}
