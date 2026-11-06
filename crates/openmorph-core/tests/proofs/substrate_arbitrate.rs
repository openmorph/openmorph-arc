// SPDX-License-Identifier: MIT-0
//! ARBITRATION — many banked skills resolving into ONE actor (MorphLab #103, ask #2). A library
//! organism with N per-task modules in disjoint cell bands has no native answer to *which module drives
//! the action*. `select`/`arbitrate` are the competitive read: route to the module whose gate (its
//! recognition of the current context) is highest, and read ITS output. The gates are grown units, so
//! the routing is learned — a gate is just more cells.
//!
//! This proves the full non-interfering library: train module A on task A, CONSOLIDATE it, then train
//! module B on task B — B's learning can't touch A (consolidation + local learn), and `arbitrate` routes
//! each context to its own module. Two opposite policies live in one organism, selected natively.
//!
//!   cargo test -p openmorph-core --test proofs substrate_arbitrate -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// Two contexts that SHARE a feature (f0) so routing isn't a trivial one-hot: A = [1, 1], B = [1, -1].
fn ctx(a: bool) -> [(usize, f32); 2] {
    if a {
        [(0, 1.0), (1, 1.0)]
    } else {
        [(0, 1.0), (1, -1.0)]
    }
}
// Module A: gate cell 2, action cell 3. Module B: gate cell 4, action cell 5.
const GA: usize = 2;
const AA: usize = 3;
const GB: usize = 4;
const AB: usize = 5;

fn build() -> Field {
    let mut f = Field::new(6);
    f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, GA); // A recognizer
    f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, AA); // A actor
    f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, GB); // B recognizer
    f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, AB); // B actor
    f
}

#[test]
fn arbitration_routes_to_the_competent_module_no_interference() {
    let mut f = build();
    let mut rng = Rng::new(0x5E1EC7);

    // Phase 1 — learn MODULE A: recognise context A (+1 on A, −1 on B) and act (+1) on its context.
    for _ in 0..3000 {
        let a = rng.u().is_multiple_of(2);
        let inp = ctx(a);
        f.learn_step(&inp, GA, if a { 1.0 } else { -1.0 }, 0.1, 50);
        if a {
            f.learn_step(&inp, AA, 1.0, 0.1, 50); // A's policy: action +1
        }
    }
    // Consolidate module A — frozen; nothing learned later can degrade it.
    f.consolidate(&[GA, AA]);

    // Snapshot A's competence, then Phase 2 — learn MODULE B (opposite policy) on the SAME cells' organism.
    let a_action_after_a = {
        f.stimulate(&ctx(true), 50);
        f.cells()[AA]
    };
    for _ in 0..3000 {
        let a = rng.u().is_multiple_of(2);
        let inp = ctx(a);
        f.learn_step(&inp, GB, if a { -1.0 } else { 1.0 }, 0.1, 50); // B recognises context B
        if !a {
            f.learn_step(&inp, AB, -1.0, 0.1, 50); // B's policy: action −1 (opposite of A)
        }
    }
    let a_action_after_b = {
        f.stimulate(&ctx(true), 50);
        f.cells()[AA]
    };

    // Arbitrate: each context routes to its own module's action.
    f.stimulate(&ctx(true), 50);
    let act_a = f.arbitrate(&[GA, GB], &[AA, AB]);
    f.stimulate(&ctx(false), 50);
    let act_b = f.arbitrate(&[GA, GB], &[AA, AB]);

    eprintln!(
        "[ARBITRATE] a 2-module library, opposite policies, one organism:\n  routing: context A → action {act_a:+.2} (want +1),  context B → action {act_b:+.2} (want −1)\n  no-forgetting: module A's action was {a_action_after_a:+.3} after learning A, {a_action_after_b:+.3} after learning B (consolidated)"
    );
    eprintln!("[ARBITRATE] VERDICT (MorphLab #103 ask #2): `select`/`arbitrate` route the action to the module whose gate (grown recognition of the context) is highest — the competent module drives, no hand-rolled argmax outside. With `graft_at` (placement) + `consolidate` (no forgetting) + this competitive select, N per-task modules compose into ONE non-interfering actor. The routing is grown (a gate is just cells) and learnable. The composition half of the marketplace.");

    assert!(
        act_a > 0.5,
        "context A must route to A's action +1 (got {act_a:+.2})"
    );
    assert!(
        act_b < -0.5,
        "context B must route to B's action −1 (got {act_b:+.2})"
    );
    assert!(
        (a_action_after_a - a_action_after_b).abs() < 0.05,
        "module A must be unchanged by learning B (consolidation): {a_action_after_a:.3} → {a_action_after_b:.3}"
    );
}
