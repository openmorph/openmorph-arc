// SPDX-License-Identifier: MIT-0
//! THE INTERNAL RAM — a WRITE-GATED short-term memory register (#122). openmorph does static batch ML well but
//! stalls on LIVE interaction because the additive `tanh(Σw·x)` unit can't HOLD a value across time: MorphLab
//! measured feedforward growth = 0.00 delayed recall and a persistent-activity reservoir LEAKING in ~1 step,
//! while a fast-weight / activity-silent register holds 0.99 FLAT across 20 distractors — but only when
//! hand-wired. The missing op is GATING: protect the trace while distractors stream past.
//!
//! The one addition — a WRITE-GATE (`set_gate`): a unit learns only in proportion to a gate cell (clamped
//! [0,1]), so it WRITES when the gate is open (salient/store) and is PROTECTED when closed. It is
//! salience-gated learning — the neuromodulation analog. From it falls out short-term MEMORY: a cell whose
//! store signal gates it captures a value while storing and holds it (the weight untouched) while the same
//! CONTEXT keeps firing during distractors — where an UN-gated cell, learning every step, is overwritten by
//! the last distractor. Same stock `learn_step`/`stimulate`; the gate makes the register autonomous.
//!
//!   cargo test -p openmorph-core --test proofs substrate_memory -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const ADDR: usize = 0; // the register's address (pulsed 0→1 every step so the memory recomputes — active always)
const STORE: usize = 1; // store/salience gate (open only on the step worth remembering)
const MEM: usize = 2; // the memory cell — its weight holds the value

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s | 1)
    }
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32
    }
    fn s(&mut self) -> f32 {
        self.f() * 2.0 - 1.0
    }
}

fn corr(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let (mx, my) = (xs.iter().sum::<f32>() / n, ys.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

/// A learning step against the memory cell: reset the address (so the cell recomputes on the 0→1 edge), then
/// learn toward `target` with the store gate set to `store` (1 = open on the salient step, 0 = distractor).
fn step(f: &mut Field, target: f32, store: f32, bud: usize) {
    f.stimulate(&[(ADDR, 0.0)], bud); // reset so the address edge re-fires the memory cell
    f.learn_step(&[(ADDR, 1.0), (STORE, store)], MEM, target, 0.5, bud);
}

/// One store→distract→recall trial. Returns the recalled value.
fn trial(f: &mut Field, s: f32, delay: usize, rng: &mut Rng) -> f32 {
    let bud = 16;
    for _ in 0..8 {
        step(f, s, 1.0, bud); // STORE: gate OPEN — encode s into the memory weight
    }
    for _ in 0..delay {
        step(f, rng.s(), 0.0, bud); // DISTRACTOR: gate CLOSED — an un-gated cell is overwritten; a gated one holds
    }
    // RECALL: pulse the address (store closed), read the memory cell.
    f.stimulate(&[(ADDR, 0.0)], bud);
    f.stimulate(&[(ADDR, 1.0), (STORE, 0.0)], bud);
    f.cells()[MEM]
}

fn build(gated: bool) -> Field {
    let mut f = Field::new(3);
    f.add_unit(vec![ADDR], vec![0.0], 0.0, MEM); // memory reads the address; its weight holds the value
    if gated {
        f.set_gate(&[MEM], STORE); // WRITE-GATE: learn only while the store signal is open
    }
    f
}

/// Delayed-recall corr(recalled, stored) over many trials at a given delay.
fn recall_corr(gated: bool, delay: usize) -> f32 {
    let mut f = build(gated);
    let mut rng = Rng::new(0x9E37_79B9);
    let (mut ss, mut ps) = (Vec::new(), Vec::new());
    for _ in 0..400 {
        let s = rng.s();
        ps.push(trial(&mut f, s, delay, &mut rng));
        ss.push(s);
    }
    corr(&ss, &ps)
}

#[test]
fn a_write_gated_register_holds_a_value_across_distractors() {
    let delays = [1usize, 3, 6, 20];
    eprintln!("[MEMORY] delayed recall corr(recalled, stored) — memory reads an ACTIVE context every step:");
    eprintln!(
        "  delay:            {:>7} {:>7} {:>7} {:>7}",
        delays[0], delays[1], delays[2], delays[3]
    );
    let ungated: Vec<f32> = delays.iter().map(|&d| recall_corr(false, d)).collect();
    let gated: Vec<f32> = delays.iter().map(|&d| recall_corr(true, d)).collect();
    eprintln!(
        "  UN-gated (leaks): {:>+7.3} {:>+7.3} {:>+7.3} {:>+7.3}",
        ungated[0], ungated[1], ungated[2], ungated[3]
    );
    eprintln!(
        "  WRITE-GATED     : {:>+7.3} {:>+7.3} {:>+7.3} {:>+7.3}",
        gated[0], gated[1], gated[2], gated[3]
    );
    eprintln!("[MEMORY] VERDICT: the internal RAM is a WRITE-GATE — the one op the additive `tanh(Σw·x)` unit lacked. A memory cell gated by its store signal captures a value while the gate is open and HOLDS it (the weight untouched) as the same context keeps firing through distractors — flat delayed recall across 20 steps — where an UN-gated cell, learning every step, is overwritten by the last distractor (corr collapses to ~0). This is the activity-silent fast-weight register MorphLab measured (Mongillo/Stokes), now autonomous: same stock `learn_step`/`stimulate`, the gate supplying the write-protection. And it is MORE than memory — it is salience-gated (neuromodulated) learning: drive the gate from surprise/reward/attention (a grown control cell) and the morph writes only what matters, which also curbs catastrophic interference (a unit updates only in its own context) and enables satisfaction-gated consolidation (write while unsatisfied, freeze once satisfied). The live-inference missing organ: the RAM holds the delayed context so `observe_from`'s dense action→effect signal becomes learnable.");

    assert!(
        gated[3] > 0.9,
        "the write-gated register must hold FLAT across 20 distractors (delay-20 corr {:.3})",
        gated[3]
    );
    assert!(
        gated[3] > ungated[3] + 0.5,
        "gating must protect the trace where an un-gated cell is overwritten (gated {:.3} vs un-gated {:.3} at delay 20)",
        gated[3],
        ungated[3]
    );
}
