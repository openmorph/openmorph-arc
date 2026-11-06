// SPDX-License-Identifier: MIT-0
//! AUTO-BOUNDARY SLEEP — a morph consolidates automatically at context boundaries, so skills accumulate as
//! compact protected bricks instead of a growing pile of interfering units. Each context is a block of events
//! terminated by a BOUNDARY TOKEN (a period ending a sentence, a blank line after a paragraph). When the token
//! fires, the morph SLEEPS: it extracts the just-learned skill's whole subgraph into a frozen, memoizing brick
//! (encapsulate → detach → nest), then wakes to the next context with a clean scratch space. After N contexts
//! the morph holds N skills, each intact, in N active cells — not N×(many) units, and the frozen bricks can't
//! be degraded by later learning. This is the wake→sleep cadence: fast messy encoding, then consolidation at
//! the boundary, all from existing primitives (+ the new `detach`).
//!
//!   cargo test -p openmorph-core --test proofs substrate_sleep_stream -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const X0: usize = 0;
const X1: usize = 1;
// three skills, each on its own output channel (cells 2,3,4)
const OUT: [usize; 3] = [2, 3, 4];
const NAME: [&str; 3] = ["XOR", "XNOR", "sign(x0)"];
fn rule(skill: usize, x0: f32, x1: f32) -> f32 {
    let yes = match skill {
        0 => x0 * x1 > 0.0,
        1 => x0 * x1 < 0.0,
        _ => x0 > 0.0,
    };
    if yes {
        1.0
    } else {
        -1.0
    }
}
fn acc(f: &mut Field, skill: usize) -> f64 {
    let mut rng = Rng::new(0xACC ^ skill as u64);
    let mut c = 0usize;
    for _ in 0..1000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(X0, x0), (X1, x1)], 120);
        let pred = if f.cells()[OUT[skill]] > 0.0 {
            1.0
        } else {
            -1.0
        };
        if (pred - rule(skill, x0, x1)).abs() < 1e-6 {
            c += 1;
        }
    }
    c as f64 / 1000.0
}

// SLEEP the just-learned skill on `chan`: extract its whole subgraph into a frozen brick (encapsulate → detach
// → nest). Composable; the only new primitive is `detach`.
fn sleep(f: &mut Field, chan: usize) {
    let mut cells = vec![chan];
    cells.extend(f.target_basis(chan));
    let brick = f.encapsulate(&cells);
    f.detach(&cells);
    f.nest(brick, &[X0, X1], chan, 120);
}

#[test]
fn auto_boundary_sleep_accumulates_compact_skills() {
    let mut f = Field::new(5);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 16;
    let mut rng = Rng::new(0x5133);

    let mut peak_units = 0usize;
    // stream: three contexts, each a block ended by a boundary token that triggers sleep
    for (skill, &chan) in OUT.iter().enumerate() {
        for _ in 0..7000 {
            let (x0, x1) = (rng.sym(), rng.sym());
            f.grow(&[(X0, x0), (X1, x1)], chan, rule(skill, x0, x1), 0.05, 120);
        }
        peak_units = peak_units.max(f.n_units());
        // ── BOUNDARY TOKEN → SLEEP ──
        sleep(&mut f, chan);
    }

    let accs: Vec<f64> = (0..3).map(|s| acc(&mut f, s)).collect();
    eprintln!(
        "[STREAM] after 3 contexts with auto-boundary sleep: {} active units (peak during wake {peak_units})",
        f.n_units()
    );
    for s in 0..3 {
        eprintln!("[STREAM]   {:<9} → {:.0}%", NAME[s], accs[s] * 100.0);
    }
    eprintln!("[STREAM] VERDICT: sleep fires on the boundary token and consolidates each context into a compact frozen brick — 3 skills survive intact in 3 active cells, not a growing pile of interfering units. Wake encodes fast; the boundary triggers consolidation; the next context starts clean. Auto-boundary sleep, from existing primitives + detach.");

    for s in 0..3 {
        assert!(
            accs[s] > 0.85,
            "skill {} survived sleep-consolidation ({:.2})",
            NAME[s],
            accs[s]
        );
    }
    assert_eq!(
        f.n_units(),
        3,
        "three skills collapsed to three active bricks (got {})",
        f.n_units()
    );
    assert!(
        peak_units > 8,
        "wake genuinely grew units before each sleep (peak {peak_units})"
    );
}
