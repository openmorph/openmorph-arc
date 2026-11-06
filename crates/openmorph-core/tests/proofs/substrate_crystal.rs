// SPDX-License-Identifier: MIT-0
//! CRYSTAL mode (v1.3.1) — the frozen snapshot of competence, the complement of the ALIVE loop. An organism
//! learns at inference while ALIVE; `crystallize()` freezes it into a still photograph at a moment in time —
//! no growth, learning, culling, or map/economy change, only settle (pure inference). It costs zero training
//! compute (the microcontroller / evaluation / deployment path), preserves per-unit plasticity, and round-
//! trips through a snapshot. `thaw()` returns it to ALIVE. Two modes, one API.
//!
//!   cargo test -p openmorph-core --test proofs substrate_crystal -- --nocapture --test-threads=1

use crate::common::Rng;
use openmorph_core::substrate::Field;

const IN: usize = 0;
const OUT: usize = 1;
fn truth(x: f32) -> f32 {
    0.6 * x
}

#[test]
fn crystal_freezes_a_snapshot_of_competence() {
    let mut f = Field::new(2);
    let mut rng = Rng::new(0xC0FFEE);
    // ALIVE: learn y = 0.6·x over varying x
    for _ in 0..2500 {
        let x = rng.sym();
        let _ = f.grow(&[(IN, x)], OUT, truth(x), 0.1, 100);
    }
    f.stimulate(&[(IN, 0.6)], 100);
    let trained = f.cells()[OUT];
    let units = f.n_units();

    // CRYSTALLISE — a snapshot of competence at this moment
    f.crystallize();
    assert!(f.is_crystal(), "crystallize must enter crystal mode");
    // hammer it with the OPPOSITE task + starvation — all no-ops while frozen
    for _ in 0..3000 {
        let x = rng.sym();
        let _ = f.grow(&[(IN, x)], OUT, -truth(x) - 0.5, 0.3, 100);
        f.cull(100.0);
        f.metabolize(0.0, 100.0);
    }
    f.stimulate(&[(IN, 0.6)], 100);
    let frozen = f.cells()[OUT];
    assert!(
        (frozen - trained).abs() < 1e-6,
        "crystal inference must be IDENTICAL after hammering ({trained:.4} vs {frozen:.4})"
    );
    assert_eq!(
        f.n_units(),
        units,
        "crystal must be immune to culling (a banked skill can't die)"
    );

    // SNAPSHOT round-trip — a crystal is a portable competence artifact
    let json = serde_json::to_string(&f).unwrap();
    let mut g: Field = serde_json::from_str(&json).unwrap();
    assert!(
        g.is_crystal(),
        "crystal mode must round-trip through a snapshot"
    );
    g.stimulate(&[(IN, 0.6)], 100);
    assert!(
        (g.cells()[OUT] - trained).abs() < 1e-6,
        "the snapshot must preserve the frozen competence"
    );

    // THAW — back to ALIVE, learning resumes
    f.thaw();
    assert!(!f.is_crystal(), "thaw must return to alive");
    for _ in 0..2500 {
        let x = rng.sym();
        let _ = f.grow(&[(IN, x)], OUT, -truth(x), 0.1, 100);
    }
    f.stimulate(&[(IN, 0.6)], 100);
    let relearned = f.cells()[OUT];

    eprintln!("[CRYSTAL] predict(0.6) [truth {:.2}] → ALIVE-trained {trained:.3} | CRYSTAL after 3000 opposite-task hits {frozen:.3} (identical) | after THAW+relearn {relearned:.3}", truth(0.6));
    eprintln!("[CRYSTAL] VERDICT: CRYSTAL is the frozen snapshot of competence — `crystallize()` sets a mode flag turning every learning/growth/culling/economy mutator into a no-op, so 3000 opposite-task grows + starvation leave the output byte-identical ({trained:.3}={frozen:.3}) and no unit dies; only settle (inference) runs, at zero training compute (microcontroller / evaluation / deploy). It round-trips through a snapshot (a portable competence artifact), and `thaw()` returns it to ALIVE where learning resumes ({trained:.3} → {relearned:.3} toward the opposite target). Two modes over one organism — the Crystal half of v1.3.1; the Alive autonomous loop is the complement.");

    assert!(
        relearned < trained - 0.2,
        "after thaw, learning must resume and move the output ({trained:.3} → {relearned:.3})"
    );
}
