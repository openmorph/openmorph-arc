// SPDX-License-Identifier: MIT-0
//! THE THIRD ACTOR LAW — METRIC GENERALIZATION: a GROWN value head generalizes across the state space where a
//! TABULAR/hashed one cannot. The direct in-core explanation of MorphBot's "state-explosion → learned actor
//! loses to random" (dv01/co01/fe02: 458-3081 states each seen ~once).
//!
//! MorphBot's 12-hard-games probe (2026-08-12) found the reachable wall is an ACTOR problem, and the striking
//! class is state-explosion where the LEARNED actor does WORSE than random. Nature-based cause: the harness kept
//! a value over a HASHED discrete state index — no cross-state sharing — so with each state seen ~once there is
//! nothing to learn from, and a greedy commitment on one noisy sample is worse than uniform. The substrate's law
//! is the opposite: GENERALIZE value on the grown metric, so ONE sample transfers to nearby states
//! (`substrate_fewshot`, `substrate_trajectory`). This proof isolates that law in the value head itself — the
//! third leg of the composed drop-in actor (the other two — intrinsic `−σ²` exploration and multi-step
//! `relax_values` planning — are proven in `substrate_free_energy_dyna`).
//!
//! THE WORLD — navigation over N states with a smooth optimal policy: the goal sits at G, so the correct action
//! is "toward G" (a single boundary at G). We TRAIN both value heads on a small subset of states (seen once) and
//! evaluate the policy on the DISJOINT held-out states (never trained) — the state-explosion regime.
//!   - GROWN head (`grow` over the state coordinate) learns the boundary and GENERALIZES to held-out states.
//!   - TABULAR head (a hashed per-state table) memorizes the trained states and is BLIND on the held-out ones —
//!     exactly the hashed-index collapse.
//!
//! Truth-not-trophy: the grown head's held-out policy accuracy is high (> 0.90) where the tabular head is at
//! chance (< 0.65) — so a grown (metric-generalizing) value head is the fix for the state-explosion games; a
//! hashed table is why the learned actor loses to random.
//!
//!   cargo test -p openmorph-core --test proofs substrate_free_energy_metric -- --nocapture

use openmorph_core::Field;
use std::collections::HashMap;

const N: usize = 200; // states 0..N (the "explosion" — far more than any run visits densely)
const G: usize = 100; // the goal position; correct action = move toward it
const N_TRAIN: usize = 60; // states actually visited during training (each ~once)

fn u01(s: &mut u64) -> f32 {
    let mut x = *s | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x >> 11) as f32 / (1u64 << 53) as f32
}

/// The correct action at state `s`: `+1` (go right / toward G) if `s < G`, else `−1` (go left). A single
/// smooth boundary at G — the kind of structure the metric SHOULD generalize over.
fn correct(s: usize) -> f32 {
    if s < G {
        1.0
    } else {
        -1.0
    }
}

/// Normalize a state index to the grown head's input coordinate.
fn coord(s: usize) -> f32 {
    s as f32 / N as f32
}

fn main_seed() -> (Vec<usize>, Vec<usize>) {
    // Split the N states into a TRAIN subset (visited) and the DISJOINT held-out remainder (never visited).
    let mut rng = 0xA11CE_u64;
    let mut order: Vec<usize> = (0..N).collect();
    // Fisher-Yates with the deterministic rng.
    for i in (1..N).rev() {
        let j = (u01(&mut rng) * (i as f32 + 1.0)) as usize % (i + 1);
        order.swap(i, j);
    }
    let train = order[..N_TRAIN].to_vec();
    let mut held: Vec<usize> = order[N_TRAIN..].to_vec();
    held.sort_unstable();
    (train, held)
}

/// The GROWN value head: `grow` a readout over the state COORDINATE toward the correct action. It learns the
/// boundary at G as a function of the coordinate, so it generalizes to states it never saw.
fn grown_heldout_accuracy(train: &[usize], held: &[usize]) -> f32 {
    let mut f = Field::new(2); // cell 0 = state coordinate, cell 1 = value output
    f.config_mut().max_hidden = 4; // a little depth to place the boundary crisply
    let mut rng = 0xBEEF_u64;
    for _ in 0..6_000 {
        let s = train[(u01(&mut rng) * train.len() as f32) as usize % train.len()];
        f.grow(&[(0, coord(s))], 1, correct(s), 0.05, 50);
    }
    let mut ok = 0;
    for &s in held {
        f.stimulate(&[(0, coord(s))], 50);
        let pred = if f.cells()[1] >= 0.0 { 1.0 } else { -1.0 };
        if (pred - correct(s)).abs() < 1e-6 {
            ok += 1;
        }
    }
    ok as f32 / held.len() as f32
}

/// The TABULAR value head: a hashed per-state table. It memorizes the trained states perfectly but has NO entry
/// for a held-out state → it must fall back to a default → blind on the explosion (the hashed-index collapse).
fn tabular_heldout_accuracy(train: &[usize], held: &[usize]) -> f32 {
    let mut table: HashMap<usize, f32> = HashMap::new();
    for &s in train {
        table.insert(s, correct(s)); // perfect memory of the visited states
    }
    let mut ok = 0;
    for &s in held {
        // Unseen state → no entry → the best a table can do is a fixed default (say "go right").
        let pred = *table.get(&s).unwrap_or(&1.0);
        if (pred - correct(s)).abs() < 1e-6 {
            ok += 1;
        }
    }
    ok as f32 / held.len() as f32
}

#[test]
fn a_grown_value_head_generalizes_where_a_hashed_table_collapses() {
    let (train, held) = main_seed();
    let grown = grown_heldout_accuracy(&train, &held);
    let tabular = tabular_heldout_accuracy(&train, &held);

    println!("held-out states: {} (never trained)", held.len());
    println!("  GROWN value head (over the coordinate) : {grown:.3}");
    println!("  TABULAR/hashed table                   : {tabular:.3}  (blind on the explosion)");

    // The grown metric-generalizing head solves the held-out policy.
    assert!(
        grown > 0.90,
        "the grown value head must generalize to held-out states (got {grown:.3})"
    );
    // The hashed table is at chance on the held-out states — the state-explosion collapse.
    assert!(
        tabular < 0.65,
        "the tabular/hashed head must be blind on held-out states (got {tabular:.3})"
    );
    // And the gap is wide — generalization is the lever.
    assert!(
        grown - tabular >= 0.30,
        "a grown value head must beat the hashed table by a wide margin (Δ {:.3})",
        grown - tabular
    );
}
