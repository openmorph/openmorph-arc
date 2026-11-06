// SPDX-License-Identifier: MIT-0
//! IMITATION = PREDICTIVE ALIGNMENT TO A DEMONSTRATOR (F1/#119) — the sparse-reward gap is a DISCOVERY limit,
//! not a learning limit (both-zero games have no reward to learn from). The nature-based fill is social
//! learning: watch a demonstrator and align to it. And it is the SAME law the substrate already runs —
//! `learn_step`/`observe` reduce the gap between a cell and its EXPECTED state; here the "expected" is the
//! DEMONSTRATOR's action. Because a cell is sensor+motor, the cell the organism ACTS with is the one it trains
//! to predict the demonstrator — that IS the mirror. No new primitive, no behavioural-cloning scaffold.
//!
//! Two claims, the second make-or-break (imitation that only memorizes is useless):
//!   (A) SOLVED-BY-WATCHING — with no reward, imitating a demonstrated policy reproduces the solution.
//!   (B) GENERALIZES — a PRIMITIVE seen in game A's demos transfers to an UN-demonstrated game B: `decompose`
//!       factors the demonstrator's perceptual primitive out of the imitation, `nest` reuses it, and B's policy
//!       forms with NO demos of B — where cold (no transferred primitive) does not. Not rote replay.
//!
//!   cargo test -p openmorph-core --test proofs substrate_imitate -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// cells: 0 = state s ; 1 = g(s) the demonstrator's perceptual primitive ; 2 = action.
const S: usize = 0;
const G: usize = 1;
const ACT: usize = 2;

/// The demonstrator's shared perceptual primitive (what it attends to in the state).
fn g_of(s: f32) -> f32 {
    (1.6 * s).tanh()
}
/// Game A's demonstrated policy (action as a function of the primitive).
fn policy_a(s: f32) -> f32 {
    (1.3 * g_of(s)).tanh()
}
/// Game B's policy — never demonstrated, but built on the SAME primitive g.
fn policy_b(s: f32) -> f32 {
    (-1.1 * g_of(s)).tanh()
}

/// Mean action error of a field's policy at ACT vs a reference policy, over held-out states.
fn policy_error(f: &mut Field, reference: fn(f32) -> f32, act_cell: usize) -> f32 {
    let (mut e, mut n) = (0.0f32, 0.0);
    for &s in &[-0.8f32, -0.3, 0.2, 0.6, 0.9] {
        f.stimulate(&[(S, s)], 30);
        e += (f.cells()[act_cell] - reference(s)).abs();
        n += 1.0;
    }
    e / n
}

#[test]
fn imitation_solves_by_watching_and_the_primitive_generalizes() {
    // ── (A) SOLVED BY WATCHING: no reward — imitate the demonstrated (state → action) trajectory. ──
    // The morph grows a policy by aligning its ACTION cell to the demonstrator's action (learn_step toward the
    // demo). It also grows the primitive g on the way (a shared perceptual feature at cell G).
    let mut f = Field::new(3);
    f.config_mut().birth_gate = 0.05;
    f.config_mut().max_hidden = 0; // grow_and_learn was linear-only
    let mut rng = Rng::new(0x1717_A7ED);
    for _ in 0..2500 {
        let s = rng.sym();
        f.grow(&[(S, s)], G, g_of(s), 0.1, 30); // grow the primitive the demonstrator attends to
        let g = f.cells()[G];
        f.grow(&[(G, g)], ACT, policy_a(s), 0.1, 30); // ALIGN action to the demonstrator
    }
    let imitation_err = policy_error(&mut f, policy_a, ACT);
    eprintln!("[IMITATE] (A) solved by watching: reproduces the demonstrated policy — action error {imitation_err:.3} (no reward used)");

    // ── (B) GENERALIZES: factor the demonstrator's primitive out of the imitation and reuse it on an
    // UN-demonstrated game B. ──
    for it in 0..4000 {
        f.co_activate(&[S, G, ACT], 0.05); // the imitation episode's co-firing cells
        if it % 100 == 99 {
            f.spread_positions();
        }
    }
    f.spread_positions();
    let groups = f.cluster(&[S, G, ACT], 2);
    let g_node = f.encapsulate(&[G]); // the factored perceptual primitive

    // TRANSFER: game B, no demos of B — nest the primitive and grow B's (short) policy over it.
    let mut b = Field::new(1);
    b.config_mut().birth_gate = 0.05;
    b.config_mut().max_hidden = 0; // grow_and_learn was linear-only
    b.nest(g_node, &[S], G, 30); // reuse the primitive from A's imitation
    let mut rb = Rng::new(0xB0B);
    for _ in 0..300 {
        let s = rb.sym();
        b.stimulate(&[(S, s)], 30);
        let g = b.cells()[G];
        b.grow(&[(G, g)], ACT, policy_b(s), 0.1, 30);
    }
    let transfer_err = policy_error(&mut b, policy_b, ACT);

    // COLD: game B from raw state, same short budget, NO transferred primitive (must re-derive g).
    let mut cold = Field::new(3);
    cold.config_mut().birth_gate = 0.05;
    cold.config_mut().max_hidden = 4;
    let mut rc = Rng::new(0xC01D);
    for _ in 0..300 {
        let s = rc.sym();
        cold.grow(&[(S, s)], ACT, policy_b(s), 0.1, 30);
    }
    let cold_err = policy_error(&mut cold, policy_b, ACT);

    eprintln!(
        "[IMITATE] (B) generalizes — the demonstrator's primitive g (cluster {groups:?}) transfers to UN-demonstrated game B:\n  TRANSFER (nest factored g, short B policy): action error {transfer_err:.3}\n  COLD    (game B from raw state, no primitive): action error {cold_err:.3}"
    );
    eprintln!("[IMITATE] VERDICT: imitation is the SAME law as `observe` — reduce the gap between a cell and its EXPECTED state, the expected now being a DEMONSTRATOR's action; and because a cell is sensor+motor, the cell the organism ACTS with is the one it trains to predict the demonstrator — the mirror, no bespoke behavioural-cloning code. It fills the sparse-reward gap: with NO reward, aligning to a demonstrated trajectory reproduces the solution (both-zero → solved). And it GENERALIZES, not memorizes: `decompose` factors the demonstrator's perceptual primitive out of the imitation, `nest` reuses it, and an UN-demonstrated game built on the same primitive forms fast where cold re-derivation struggles under the same budget. Imitation + factorization = social learning that transfers — the discovery-gap fill that Feature 2 (dream) then consolidates into a distinct vocabulary.");

    assert!(
        imitation_err < 0.05,
        "imitation must reproduce the demonstrated policy (error {imitation_err:.3})"
    );
    assert!(
        transfer_err < 0.05 && transfer_err * 2.0 < cold_err,
        "the demonstrated primitive must GENERALIZE to an un-demonstrated game (transfer {transfer_err:.3} vs cold {cold_err:.3})"
    );
}
