// SPDX-License-Identifier: MIT-0
//! NO CATASTROPHIC FORGETTING under `alive()` — the SEQUENTIAL-block curriculum. Each of six skills is trained
//! ALONE in its own block, in order (sign(x0) → … → XOR), never revisited. At the end every skill still holds
//! its peak accuracy: the per-target economy ([`Field::target_basis`]) plus separate output channels give
//! modular isolation FOR FREE — a dormant skill's readout is untouched (learn_step only touches the active
//! target's writer) and its features keep their energy (metabolism scopes to the active target), so a later
//! block cannot overwrite an earlier skill. This is why episodic replay is NOT needed to avoid forgetting in
//! the multi-channel setting: the substrate already isolates skills in time. (Replay/detection remain the
//! levers for the AUTONOMOUS case, where the task-id is grown, not supplied.)
//!
//!   cargo test -p openmorph-core --test proofs substrate_alive_sequential -- --nocapture --test-threads=1
use openmorph_core::substrate::Field;
use openmorph_core::Event;

use crate::common::Rng;

const OUT: [usize; 6] = [2, 3, 4, 5, 6, 7];
const NAME: [&str; 6] = ["sign(x0)", "sign(x1)", "AND", "OR", "quadrant", "XOR"];

fn target(task: usize, x0: f32, x1: f32) -> f32 {
    let (s0, s1) = (x0 > 0.0, x1 > 0.0);
    let yes = match task {
        0 => s0,
        1 => s1,
        2 => s0 && s1,
        3 => s0 || s1,
        4 => x0 + x1 > 0.0,
        _ => x0 * x1 > 0.0,
    };
    if yes {
        1.0
    } else {
        -1.0
    }
}

fn eval(f: &mut Field, task: usize, budget: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut correct = 0usize;
    for _ in 0..1000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], budget);
        let pred = if f.cells()[OUT[task]] > 0.0 {
            1.0
        } else {
            -1.0
        };
        if (pred - target(task, x0, x1)).abs() < 1e-6 {
            correct += 1;
        }
    }
    correct as f64 / 1000.0
}

#[test]
fn sequential_blocks_do_they_forget() {
    let budget = 300;
    let block = 3000usize;
    let mut f = Field::new(8);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 16;
    let mut rng = Rng::new(0x777);

    // Track each skill's accuracy RIGHT AFTER its own block (peak), to compare with the final accuracy.
    let mut peak = [0.0f64; 6];
    for task in 0..6 {
        for _ in 0..block {
            let (x0, x1) = (rng.sym(), rng.sym());
            f.alive(Event::Perceive {
                inputs: &[(0, x0), (1, x1)],
                target_cell: OUT[task],
                target: target(task, x0, x1),
                lr: 0.05,
                budget,
            });
        }
        peak[task] = eval(&mut f, task, budget, 0x999 ^ task as u64);
    }

    eprintln!("[SEQ] sequential blocks (each skill trained ALONE, in order):");
    let mut worst_drop = 0.0f64;
    for t in 0..6 {
        let fin = eval(&mut f, t, budget, 0x999 ^ t as u64);
        let drop = peak[t] - fin;
        worst_drop = worst_drop.max(drop);
        eprintln!(
            "[SEQ] {:<9} peak-after-its-block {:.0}%  →  FINAL {:.0}%   (drop {:.0}pts)",
            NAME[t],
            peak[t] * 100.0,
            fin * 100.0,
            drop * 100.0
        );
        assert!(
            fin > 0.80,
            "{} was forgotten by the end of the curriculum (final {:.2})",
            NAME[t],
            fin
        );
    }
    eprintln!("[SEQ] VERDICT: worst forgetting drop {:.0}pts — the per-target economy + separate channels isolate each skill in TIME, so sequential (non-interleaved) training does not overwrite earlier skills. Episodic replay is not required to retain skills here.", worst_drop * 100.0);
    assert!(
        worst_drop < 0.12,
        "a skill was catastrophically forgotten (worst drop {:.2})",
        worst_drop
    );
}
