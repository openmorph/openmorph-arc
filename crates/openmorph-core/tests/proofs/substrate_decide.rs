// SPDX-License-Identifier: MIT-0
//! rc6 step 5 — THINK-AND-COMMIT natively on the async substrate. A recurrent winner-take-all field
//! (two outputs, self-exciting + mutually inhibiting) SETTLES to a decision. Clear evidence resolves
//! FAST (few events); ambiguous evidence resolves SLOWLY (more events = more thinking) — the
//! drift-diffusion / reaction-time signature, emerging from the event dynamics themselves, no tick.
//! The commit is the settled winner; the number of firings is the thinking time.
//!
//!   cargo test -p openmorph-core --test proofs substrate_decide -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

/// A 2-choice winner-take-all decision field: in0,in1 = evidence; o0,o1 = the competing decisions.
/// Each output self-excites and inhibits the other → a bistable attractor that commits to one.
fn wta() -> Field {
    let mut f = Field::new(4); // in0=0, in1=1, o0=2, o1=3
    f.add_unit(vec![0, 2, 3], vec![1.0, 1.6, -1.6], 0.0, 2); // o0 = tanh(in0 + 1.6·o0 − 1.6·o1)
    f.add_unit(vec![1, 3, 2], vec![1.0, 1.6, -1.6], 0.0, 3); // o1 = tanh(in1 + 1.6·o1 − 1.6·o0)
    f
}

/// Present evidence; settle; return (winner, thinking-time = firings, margin |o0−o1|).
fn decide(in0: f32, in1: f32) -> (usize, usize, f32) {
    let mut f = wta();
    let events = f.stimulate(&[(0, in0), (1, in1)], 100_000);
    assert!(
        f.settled(),
        "the decision field must settle (not run to budget)"
    );
    let (a, b) = (f.cells()[2], f.cells()[3]);
    (if a > b { 0 } else { 1 }, events, (a - b).abs())
}

#[test]
fn a_decision_field_settles_faster_on_clearer_evidence() {
    let (w_easy, e_easy, m_easy) = decide(0.90, 0.10); // strong evidence for 0
    let (w_hard, e_hard, m_hard) = decide(0.54, 0.46); // barely-for-0

    eprintln!(
        "[DECIDE] async WTA: EASY (0.90 vs 0.10) → winner {w_easy}, {e_easy} firings, margin {m_easy:.2}  |  HARD (0.54 vs 0.46) → winner {w_hard}, {e_hard} firings, margin {m_hard:.2}",
    );
    eprintln!("[DECIDE] VERDICT (rc6 step 5): the async substrate THINKS by settling — a clear stimulus commits fast, an ambiguous one takes more firings to resolve (the reaction-time signature), all from the event dynamics with no tick. Committing = the settled winner; thinking-time = the firings. Answering-as-act (deliberate) folds onto this natively.");
    assert_eq!(w_easy, 0, "easy evidence resolves to the right choice");
    assert_eq!(
        w_hard, 0,
        "hard evidence still resolves to the (slight) leader"
    );
    assert!(
        e_hard > e_easy,
        "the field THINKS LONGER on ambiguous evidence ({e_hard} vs {e_easy} firings)"
    );
    assert!(
        m_easy > 1.0 && m_hard > 1.0,
        "the attractor commits fully either way (WTA)"
    );
}
