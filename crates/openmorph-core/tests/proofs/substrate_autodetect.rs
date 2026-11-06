// SPDX-License-Identifier: MIT-0
//! AUTONOMOUS GROWN DETECTION — a morph allocates its OWN modules over a stream of contexts, with no
//! pre-built structure and no supplied task-id. It decides HOW MANY skills to grow, recruits on genuine
//! novelty, routes each context to its module, and learns each skill (including the nonlinear XOR) with no
//! interference. This is the autonomous extension of `substrate_arbitrate` (which pre-builds the modules).
//!
//! It is a USAGE PROTOCOL over existing primitives — `add_unit` (recruit), `grow` (recognizer + skill),
//! `select`/`arbitrate` (route) — NO new core. The design is what makes it converge (the two naive versions
//! failed): the ACTOR reads only task inputs + grows depth (so one module cannot cheat by fitting every
//! context through the cue — that masked the surprise and under-recruited); the GATE recognizes the module's
//! own HOME context, grown to depth (an interior prototype is a non-monotonic bump a linear gate cannot form);
//! NOVELTY = "no module RECOGNIZES this context" (max gate low) — which CONVERGES (once every context has a
//! recognizer, recruiting stops), unlike an actor-error signal (a nonlinear skill keeps a residual and would
//! over-recruit); a warmup/cooldown/cap prevents the runaway the "recruit-if-unconfident" version had.
//!
//! HONEST BOUNDARY (truth-not-trophy): this converges for WELL-SEPARATED contexts. A grown-tanh recognizer
//! over the raw cue OVER-GENERALIZES to ADJACENT prototypes (measured: with cues −1/0/+1, the −1 and +1
//! recognizers both fire on 0, so the middle context is never seen as novel and never gets a module). Sharp
//! recognition of adjacent/overlapping contexts is a LOCAL (RBF / concept-map / localization) job — a proven
//! openmorph capability not wired in here. So: autonomous allocation WORKS; the recognizer's representational
//! capacity is the frontier, and the concept-map is the next lever for dense context spaces.
//!
//!   cargo test -p openmorph-core --test proofs substrate_autodetect -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;

use crate::common::Rng;

const X0: usize = 0;
const X1: usize = 1;
const CUE: usize = 2;
const BASE: usize = 3;

fn target(ctx_b: bool, x0: f32, x1: f32) -> f32 {
    let yes = if ctx_b { x0 * x1 > 0.0 } else { x0 > 0.0 };
    if yes {
        1.0
    } else {
        -1.0
    }
}

struct Module {
    gate: usize,
    actor: usize,
    home: f32,
}
struct Auto {
    f: Field,
    m: Vec<Module>,
    lr: f32,
    budget: usize,
    since_recruit: usize,
    cap: usize,
}

impl Auto {
    fn new() -> Self {
        let mut f = Field::new(BASE);
        f.config_mut().max_hidden = 16;
        f.config_mut().birth_gate = 0.15;
        Auto {
            f,
            m: Vec::new(),
            lr: 0.05,
            budget: 150,
            since_recruit: 0,
            cap: 6,
        }
    }
    fn recruit(&mut self, cue: f32) -> usize {
        let gate = self.f.len();
        self.f.ensure_cells(gate + 1);
        self.f.add_unit(vec![CUE], vec![0.0], 0.0, gate);
        let actor = self.f.len();
        self.f.ensure_cells(actor + 1);
        self.f.add_unit(vec![X0, X1], vec![0.0, 0.0], 0.0, actor);
        let idx = self.m.len();
        self.m.push(Module {
            gate,
            actor,
            home: cue,
        });
        self.since_recruit = 0;
        for _ in 0..80 {
            self.f.grow(&[(CUE, cue)], gate, 1.0, 0.2, 30);
        }
        idx
    }
    fn step(&mut self, x0: f32, x1: f32, cue: f32, tgt: f32) {
        let inp = [(X0, x0), (X1, x1), (CUE, cue)];
        if self.m.is_empty() {
            self.recruit(cue);
        }
        self.f.stimulate(&inp, self.budget);
        let gates: Vec<usize> = self.m.iter().map(|m| m.gate).collect();
        let best = self.f.select(&gates);
        let conf = self.f.cells()[self.m[best].gate];
        self.since_recruit += 1;
        let novel = conf < 0.5 && self.since_recruit > 300 && self.m.len() < self.cap;
        let winner = if novel { self.recruit(cue) } else { best };
        for k in 0..self.m.len() {
            let (g, home) = (self.m[k].gate, self.m[k].home);
            let want = if (cue - home).abs() < 0.5 { 1.0 } else { -1.0 };
            self.f.grow(&[(CUE, cue)], g, want, self.lr, 30);
        }
        let actor = self.m[winner].actor;
        self.f
            .grow(&[(X0, x0), (X1, x1)], actor, tgt, self.lr, self.budget);
    }
    fn answer(&mut self, x0: f32, x1: f32, cue: f32) -> f32 {
        self.f
            .stimulate(&[(X0, x0), (X1, x1), (CUE, cue)], self.budget);
        let gates: Vec<usize> = self.m.iter().map(|m| m.gate).collect();
        let actors: Vec<usize> = self.m.iter().map(|m| m.actor).collect();
        self.f.arbitrate(&gates, &actors)
    }
}

fn eval(a: &mut Auto, ctx_b: bool, seed: u64) -> f64 {
    let cue = if ctx_b { 1.0 } else { -1.0 };
    let mut rng = Rng::new(seed);
    let mut c = 0usize;
    for _ in 0..1000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        let pred = if a.answer(x0, x1, cue) > 0.0 {
            1.0
        } else {
            -1.0
        };
        if (pred - target(ctx_b, x0, x1)).abs() < 1e-6 {
            c += 1;
        }
    }
    c as f64 / 1000.0
}

#[test]
fn autonomous_detection() {
    let mut a = Auto::new();
    let mut rng = Rng::new(0x1234);
    for i in 0..24000 {
        let ctx_b = i % 2 == 0;
        let cue = if ctx_b { 1.0 } else { -1.0 };
        let (x0, x1) = (rng.sym(), rng.sym());
        a.step(x0, x1, cue, target(ctx_b, x0, x1));
    }
    let (aa, bb) = (eval(&mut a, false, 0x1), eval(&mut a, true, 0x2));
    eprintln!(
        "[AUTO] modules {}   A(sign) {:.0}%   B(XOR) {:.0}%",
        a.m.len(),
        aa * 100.0,
        bb * 100.0
    );
    eprintln!("[AUTO] VERDICT: the morph autonomously allocated one module per context (no supplied task-id), recruited on novelty (unrecognized context) and CONVERGED, routed each context to its module via grown recognizers, and learned the nonlinear XOR through the router with no interference. Composes add_unit/grow/select/arbitrate — no new core.");
    assert_eq!(
        a.m.len(),
        2,
        "must allocate exactly one module per distinct context (got {})",
        a.m.len()
    );
    assert!(aa > 0.90, "linear context routed + learned (got {aa:.2})");
    assert!(
        bb > 0.85,
        "nonlinear context (XOR) routed + learned through the grown router (got {bb:.2})"
    );
}
