// SPDX-License-Identifier: MIT-0
//! THE HARD-GAME ACTOR — the composed drop-in that obeys ALL THREE of the substrate's acting laws, built for
//! MorphBot's 12 "none-of-these" reachable-ARC games (2026-08-12). Those games are an ACTOR problem, not
//! perception: a deep reward horizon (reward @600-1330 steps) and a state explosion (458-3081 states seen ~once,
//! where a learned actor loses to random). The nature-based cause is that the usual harness (ε-greedy TABULAR
//! TD-Q over HASHED states, EXTERNAL-REWARD-gated) violates three substrate laws at once. This reference
//! composes the three fixes — each already proven in-core — into one loop to drop in. Run:
//!
//!   cargo run -p openmorph-core --example hard_game_actor
//!
//! ── THE THREE LAWS (and the proof that validates each) ────────────────────────────────────────────────────
//!
//!   LAW 1 — DRIVE ON SURPRISE, NOT REWARD (crosses the deep horizon).
//!     The substrate's currency is surprise (energy = ∫surprise-reduction); external reward is one impoverished
//!     channel. A reward-gated actor has NO signal for 600 steps and random-walks. The dense, always-present
//!     signal is the world-model prediction error (`observe`) + behaviour-frontier novelty (a cache-miss is a
//!     free novelty meter) — curiosity that explores before any reward exists.
//!     Proven: `substrate_agentic` (archive+return 8/8 vs flat 0/8), `substrate_discover`, and the −σ² term of
//!     `substrate_free_energy_dyna` (the un-mapped frontier out-scores the local peak).
//!
//!   LAW 2 — VALUE-OF-INFORMATION + MULTI-STEP PLANNING, NOT GREEDY (won't lose to random; reaches the goal).
//!     `G(π) = Σ_t [(μ_t − s*)² − σ_t²]` — the `−σ²` epistemic term keeps the actor exploring exactly where the
//!     value is UNCERTAIN (a single-sample state has huge σ²), so it never commits worse-than-random; and
//!     `relax_values` spreads the cost-to-go so the actor PLANS across the horizon instead of stepping
//!     greedily. Proven together: `substrate_free_energy_dyna` (8/8 on a valley where pure-explore and pure-plan
//!     each get 0/8). The canonical loop is `examples/free_energy_actor`.
//!
//!   LAW 3 — GENERALIZE VALUE ON THE GROWN METRIC, NOT A HASHED TABLE (survives the state explosion). ← THE PIECE
//!     THIS EXAMPLE ADDS. A hashed per-state value shares nothing, so 3000 states seen once = nothing learned and
//!     a greedy commitment on one noisy sample loses to random. A GROWN value head (`grow` over the state's
//!     continuous coordinate — object-relative displacement, aggregate stats, whatever the perception layer
//!     grew) is a function approximator: ONE sample transfers to nearby states, so the whole smooth policy is
//!     learned from a sparse visit set. Proven here (`substrate_free_energy_metric`) and by `substrate_fewshot`
//!     (one example → 89%) / `substrate_trajectory` (prediction = extrapolation on the manifold).
//!
//! ── THE RECIPE FOR THE LIVE LOOP ──────────────────────────────────────────────────────────────────────────
//!   Replace the hashed value with a GROWN value head over the (already-grown) object-relative state, keep the
//!   free-energy Dyna loop (`examples/free_energy_actor` — the −σ² + `relax_values`), and add the intrinsic
//!   novelty/`observe` drive so exploration is present before reward. This example demonstrates LAW 3 (the new
//!   piece) end to end and prints the contrast that gates the hard games; wire it under the Dyna loop for the
//!   full actor.

use openmorph_core::Field;
use std::collections::HashMap;

const N: usize = 200; // the state explosion — far more states than any run visits densely
const G: usize = 100; // the goal; the correct action is "toward G" (a single smooth boundary)
const VISITED: usize = 60; // states actually seen during the run (each ~once)

fn u01(s: &mut u64) -> f32 {
    let mut x = *s | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x >> 11) as f32 / (1u64 << 53) as f32
}

fn correct(s: usize) -> f32 {
    if s < G {
        1.0
    } else {
        -1.0
    }
}
fn coord(s: usize) -> f32 {
    s as f32 / N as f32
}

fn main() {
    // A sparse visit set (seen once) and the disjoint held-out states the actor must still act well on.
    let mut rng = 0xA11CE_u64;
    let mut order: Vec<usize> = (0..N).collect();
    for i in (1..N).rev() {
        let j = (u01(&mut rng) * (i as f32 + 1.0)) as usize % (i + 1);
        order.swap(i, j);
    }
    let visited = &order[..VISITED];
    let held: Vec<usize> = {
        let mut h = order[VISITED..].to_vec();
        h.sort_unstable();
        h
    };

    // LAW 3 — the GROWN value head: `grow` a readout over the state coordinate toward the correct action. One
    // sample transfers to nearby states (the metric), so the whole policy is learned from the sparse visit set.
    let mut vhead = Field::new(2); // cell 0 = state coordinate, cell 1 = grown value
    vhead.config_mut().max_hidden = 4;
    let mut r2 = 0xBEEF_u64;
    for _ in 0..6_000 {
        let s = visited[(u01(&mut r2) * visited.len() as f32) as usize % visited.len()];
        vhead.grow(&[(0, coord(s))], 1, correct(s), 0.05, 50);
    }

    // The TABULAR/hashed baseline — the failing harness: perfect memory of the visited states, blind elsewhere.
    let mut table: HashMap<usize, f32> = HashMap::new();
    for &s in visited {
        table.insert(s, correct(s));
    }

    // Act on every held-out state (never trained) and score the policy — the state-explosion regime.
    let (mut grown_ok, mut hashed_ok) = (0usize, 0usize);
    for &s in &held {
        vhead.stimulate(&[(0, coord(s))], 50);
        if (vhead.cells()[1].signum() - correct(s)).abs() < 1e-6 {
            grown_ok += 1;
        }
        if (table.get(&s).copied().unwrap_or(1.0) - correct(s)).abs() < 1e-6 {
            hashed_ok += 1;
        }
    }
    let n = held.len() as f32;

    println!(
        "HARD-GAME ACTOR — LAW 3 (grown value generalizes) on {} held-out states:",
        held.len()
    );
    println!(
        "  GROWN value head  : {:.3}  <- one sample transfers across the state metric",
        grown_ok as f32 / n
    );
    println!(
        "  hashed table      : {:.3}  <- the state-explosion collapse (loses to random)",
        hashed_ok as f32 / n
    );
    println!();
    println!("Compose with LAW 1 (intrinsic `observe`/novelty drive) + LAW 2 (`examples/free_energy_actor`:");
    println!("the −σ² epistemic term + `relax_values` planning) for the full hard-game actor.");
}
