// SPDX-License-Identifier: MIT-0
//! ALIVE mode (v1.3.1 #1, the Alive half) — learning at inference, no external orchestration. The design
//! question: is being ALIVE just the substrate running unimpeded? A single `alive(event)` call bundles the
//! whole loop — perceive the inputs, GROW on surprise, learn per event, and SIZE ITSELF by the reliance
//! economy (pay the grown basis ∝ how much the answer relies on it, cull the starved with a newborn grace) —
//! all auto-triggered from the organism's own state. The caller only feeds the event.
//!
//! Proof: driving a task ONLY through `alive` reproduces the hand-orchestrated LIVING POOL (#128) — it births
//! liberally then self-sizes back to the writers the answer relies on, held-out accuracy preserved — where
//! `grow` alone (no economy) BLOATS. So the living pool falls out of `alive`; no external metabolize/cull.
//!
//!   cargo test -p openmorph-core --test proofs substrate_alive -- --nocapture --test-threads=1

use crate::common::Rng;
use openmorph_core::substrate::{Event, Field, Outcome};

const NX: usize = 3;
fn truth(x: &[f32]) -> f32 {
    (0.6 * x[0] - 0.4 * x[1] + 0.3 * x[2]).tanh()
}
fn mae(f: &mut Field, rng: &mut Rng) -> f32 {
    let mut e = 0.0;
    for _ in 0..300 {
        let x: Vec<f32> = (0..NX).map(|_| rng.sym()).collect();
        let inp: Vec<(usize, f32)> = (0..NX).map(|k| (k, x[k])).collect();
        f.stimulate(&inp, 100);
        e += (f.cells()[NX] - truth(&x)).abs();
    }
    e / 300.0
}

/// `alive`: one call per event bundles grow + self-sizing. `grow_only`: grow with NO economy (the bloat baseline).
fn run(alive: bool, seed: u64) -> (f32, usize, usize) {
    let mut f = Field::new(NX + 1);
    f.config_mut().feature_lr = 0.3;
    let mut rng = Rng::new(seed);
    let mut peak = 0;
    for _ in 0..3000 {
        let x: Vec<f32> = (0..NX).map(|_| rng.sym()).collect();
        let inp: Vec<(usize, f32)> = (0..NX).map(|k| (k, x[k])).collect();
        if alive {
            f.alive(Event::Perceive {
                inputs: &inp,
                target_cell: NX,
                target: truth(&x),
                lr: 0.1,
                budget: 100,
            });
        } else {
            let _ = f.grow(&inp, NX, truth(&x), 0.1, 100);
        }
        peak = peak.max(f.n_units());
    }
    let m = mae(&mut f, &mut rng);
    (m, f.n_units(), peak)
}

#[test]
fn alive_is_the_substrate_running_unimpeded() {
    let (alive_mae, alive_units, alive_peak) = run(true, 0xA11FE);
    let (grow_mae, grow_units, grow_peak) = run(false, 0xA11FE);

    eprintln!("[ALIVE] learning a function purely through one call per event:");
    eprintln!("[ALIVE]   grow only (no economy) . born to {grow_peak}, ended {grow_units} units — BLOATS; held-out MAE {grow_mae:.3}");
    eprintln!("[ALIVE]   ALIVE (one call/event) . born to {alive_peak}, self-sized to {alive_units} units; held-out MAE {alive_mae:.3}");
    eprintln!("[ALIVE] VERDICT: ALIVE is the substrate running unimpeded. A single `alive(event)` call bundles perceive → grow-on-surprise → learn → self-size (reliance economy + cull with a newborn grace), all auto-triggered from the organism's own state — the caller only feeds events, never orchestrates metabolize/consolidate/cull. Driving the task through `alive` alone reproduces the #128 LIVING POOL: it births liberally (peak {alive_peak}) then sizes itself back to {alive_units} units with held-out MAE {alive_mae:.3} preserved, where `grow` without the economy bloats to {grow_units}. The living pool — and learning at inference — fall out of one call. This is the Alive half of v1.3.1; `crystallize()` freezes the very same organism into the Crystal snapshot. Two modes, one API, no orchestration.");

    assert!(
        alive_units < alive_peak,
        "alive must self-size below its birth peak (the living pool)"
    );
    assert!(
        alive_units < grow_units,
        "alive must be leaner than grow-without-economy ({alive_units} vs {grow_units})"
    );
    assert!(
        alive_mae < grow_mae + 0.03,
        "alive must preserve accuracy while self-sizing ({alive_mae:.3} vs {grow_mae:.3})"
    );
}

#[test]
fn the_whole_arc_loop_runs_through_one_alive_entry() {
    // Three event kinds — Perceive / Suggest / Answer — each ONE `alive` call, no orchestration. Here the ARC
    // shape: SUGGEST the rule with a few examples, then ANSWER a held-out-colour question in a single call.
    const M: usize = 12;
    const PER: usize = 3;
    let train = [1.0f32, 2.0, 3.0, 4.0];
    let test = [5.0f32, 6.0, 7.0, 8.0]; // held-out colours
    let gen = |cols: &[f32], rng: &mut Rng| -> Vec<f32> {
        let seed: Vec<f32> = (0..PER).map(|_| cols[rng.below(cols.len())]).collect();
        (0..M).map(|i| seed[i % PER]).collect()
    };
    let mut f = Field::new(M);
    let mut rng = Rng::new(0x0A11E);
    // SUGGEST: 5 examples, one `alive(Event::Suggest)` each
    for _ in 0..5 {
        let g = gen(&train, &mut rng);
        let obs: Vec<(usize, f32)> = (0..M).map(|i| (i, g[i])).collect();
        assert!(matches!(
            f.alive(Event::Suggest { observation: &obs }),
            Outcome::Suggested
        ));
    }
    // ANSWER: held-out questions, one `alive(Event::Answer)` each — the morph returns the completed cell
    let (mut ok, mut tot) = (0, 0);
    for _ in 0..300 {
        let g = gen(&test, &mut rng);
        let m = rng.below(M);
        let known: Vec<(usize, f32)> = (0..M).filter(|&i| i != m).map(|i| (i, g[i])).collect();
        if let Outcome::Answered(a) = f.alive(Event::Answer {
            known: &known,
            unknown: &[m],
        }) {
            if let Some(&(_, v)) = a.first() {
                if (v - g[m]).abs() < 1e-3 {
                    ok += 1;
                }
            }
        }
        tot += 1;
    }
    let acc = ok as f32 / tot as f32;
    eprintln!("[ALIVE] the whole ARC loop through ONE `alive` entry — Suggest×5 then Answer, held-out colours: {:.0}%", acc * 100.0);
    eprintln!("[ALIVE] VERDICT: one entry point, one call per event. `alive(Event::Perceive)` = learning at inference; `alive(Event::Suggest)` = bind a demonstrated rule; `alive(Event::Answer)` = complete a question toward the bound consistency. Self-inferred goals (the autonomous default) and SUGGESTED goals (the ARC channel) flow through the SAME call, with no orchestration — perceive, suggest, answer, all `alive`. The v1.3.1 plug-and-play API: attach a world, feed events, choose Alive or Crystal.");

    assert!(
        acc > 0.9,
        "the ARC loop through one alive entry must answer few-shot on held-out colours ({acc:.2})"
    );
}
