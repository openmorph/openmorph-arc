// SPDX-License-Identifier: MIT-0
//! ONE `alive()` MORPH ACCUMULATES A CURRICULUM OF SKILLS — no gating, no orchestration. Six skills on six
//! output channels (sign(x0), sign(x1), AND, OR, quadrant, XOR), trained interleaved through the single
//! `alive(Event::Perceive)` call. All six coexist at the end, INCLUDING the nonlinear XOR.
//!
//! This closes the curriculum wall MorphLab hit: the five LINEAR skills learned but XOR stuck at chance. The
//! cause was the living-pool economy running GLOBALLY every event: a skill's readout builds reliance only on
//! ITS OWN events, but a global upkeep drained a skill's features on EVERY other skill's event too — so the
//! minority nonlinear skill (XOR, 1/6 of an interleaved stream) paid upkeep 6× faster than it earned reliance,
//! and its specialist features died before they matured. The fix is `alive` metering only the ACTIVE target's
//! features (`target_basis`) — which is exactly where `learn_step` updates that event — so each skill's
//! sub-pool self-sizes on its own events. Additive skill-building falls out of the one loop.
//!
//! Truth-not-trophy: a multi-skill morph is BIGGER, so its settle needs a larger anytime `budget` (too small a
//! budget starves the settle → surprise never falls → births run away). The budget scales with the morph, as
//! the anytime substrate intends; here 300 settles a ~60-unit morph.
//!
//!   cargo test -p openmorph-core --test proofs substrate_alive_curriculum -- --nocapture --test-threads=1

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
        _ => x0 * x1 > 0.0, // XOR — parity, not linearly separable
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
fn one_alive_morph_accumulates_six_skills_including_xor() {
    let budget = 300;
    let mut f = Field::new(8); // inputs 0,1 ; outputs 2..8
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 16;

    // Interleave all six skills through ONE alive() loop — round-robin, no gating, no consolidation.
    let mut rng = Rng::new(0x777);
    for i in 0..18_000usize {
        let task = i % 6;
        let (x0, x1) = (rng.sym(), rng.sym());
        f.alive(Event::Perceive {
            inputs: &[(0, x0), (1, x1)],
            target_cell: OUT[task],
            target: target(task, x0, x1),
            lr: 0.05,
            budget,
        });
    }

    let acc: Vec<f64> = (0..6)
        .map(|t| eval(&mut f, t, budget, 0x999 ^ t as u64))
        .collect();
    for t in 0..6 {
        eprintln!(
            "[CURRIC] {:<9} → {:.0}%   (basis {})",
            NAME[t],
            acc[t] * 100.0,
            f.target_basis(OUT[t]).len()
        );
    }
    eprintln!("[CURRIC] VERDICT: one alive() loop accumulates all six skills — the five linear ones AND the nonlinear XOR — because the reliance economy is metered per ACTIVE target, so a minority nonlinear skill's features are judged on its own events, not drained by the other skills'. Additive skill-building, no gating.");

    for t in 0..6 {
        assert!(
            acc[t] > 0.80,
            "skill {} did not accumulate ({:.2}) — a curriculum skill was lost",
            NAME[t],
            acc[t]
        );
    }
    // The crux: XOR (nonlinear, the one that used to stick at chance) genuinely learned.
    assert!(
        acc[5] > 0.80,
        "XOR must crack in the curriculum, not stick at chance (got {:.2})",
        acc[5]
    );
}
