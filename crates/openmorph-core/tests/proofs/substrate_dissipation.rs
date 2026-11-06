// SPDX-License-Identifier: MIT-0
//! ENERGY DISSIPATION ACCOUNTING — the metabolize paths cap a unit's energy at `energy_cap` with
//! `.min(energy_cap)`, and the overflow (`income − upkeep` above the ceiling) was SILENTLY DISCARDED: energy left
//! the economy with no ledger, so there was no denominator for "where did the energy go". `dissipated_overflow`
//! now accounts exactly that discarded overflow, at every metabolize site. It is pure observability — never read
//! back into any dynamics — so `energy_of` is bit-for-bit unchanged (the byte-identity oracle, `physics_*_engine`,
//! stays green; run those separately). SCOPE (truth-not-trophy): this is the cap-overflow ledger ONLY — culled-unit
//! energy and metabolic-pool burns are separate conservation questions, deliberately not folded in.
//!
//!   cargo test -p openmorph-core --test proofs substrate_dissipation -- --nocapture

use openmorph_core::Field;

/// A→cell1→B graph: unit 0 (A) writes cell 1, unit 1 (B) reads cell 1 with weight 0.8, so cell 1's writer A has a
/// non-zero reliance (0.8) and thus earns income under `metabolize_reliance`.
fn two_unit_field() -> Field {
    let mut f = Field::new(4);
    f.add_unit(vec![0], vec![0.5], 0.0, 1); // unit 0 (A): reads cell 0 → writes cell 1
    f.add_unit(vec![1], vec![0.8], 0.0, 2); // unit 1 (B): reads cell 1 → writes cell 2  (relies on A)
    f
}

#[test]
fn cap_overflow_is_accounted_not_silently_discarded() {
    let mut f = two_unit_field();
    let cap = f.config().energy_cap;
    let relied = [1usize]; // cell 1 — A is its writer, reliance = B's |weight| = 0.8

    // Baseline: nothing has overflowed yet.
    assert_eq!(
        f.dissipated_overflow(),
        0.0,
        "a fresh field has discarded no energy"
    );

    // A huge income drives A's energy far past the cap in one metabolize.
    f.metabolize_reliance(&relied, 10_000.0, 0.0);
    let e_after = f.energy_of(0);
    let d1 = f.dissipated_overflow();
    eprintln!("[DISSIPATION] after big income: energy_of(A)={e_after} (cap={cap}), dissipated_overflow={d1}");

    assert_eq!(
        e_after.to_bits(),
        cap.to_bits(),
        "A's energy must be pinned at energy_cap (the cap behaviour is unchanged): {e_after} vs {cap}"
    );
    assert!(
        d1 > 0.0,
        "the overflow above energy_cap must be ACCOUNTED, not discarded: {d1}"
    );

    // Already pinned at the cap: another income step overflows entirely, so the ledger keeps growing.
    f.metabolize_reliance(&relied, 10_000.0, 0.0);
    let d2 = f.dissipated_overflow();
    eprintln!("[DISSIPATION] after a second capped step: dissipated_overflow={d2}");
    assert!(
        d2 > d1,
        "an income step while already at the cap dissipates entirely and must keep accruing: {d1} → {d2}"
    );
    assert_eq!(
        f.energy_of(0).to_bits(),
        cap.to_bits(),
        "energy stays pinned at the cap across the second step"
    );

    eprintln!("[DISSIPATION] VERDICT: energy discarded by the energy_cap ceiling is now on a ledger (`dissipated_overflow`), giving the economy a denominator for where its energy goes — while `energy_of` stays bit-for-bit, so no dynamics and no byte-identity proof are touched.");
}

#[test]
fn no_overflow_no_dissipation() {
    // Income kept well under the headroom to the cap ⇒ never overflows ⇒ the ledger stays exactly 0 (a control
    // that shows the accumulator only counts genuine cap-overflow, not ordinary metabolism).
    let mut f = two_unit_field();
    let cap = f.config().energy_cap;
    for _ in 0..3 {
        // tiny income · reliance 0.8 keeps energy below the cap; upkeep 0 so nothing is lost either
        f.metabolize_reliance(&[1usize], 0.001, 0.0);
        assert!(
            f.energy_of(0) < cap,
            "sanity: energy stays below the cap in this control"
        );
    }
    assert_eq!(
        f.dissipated_overflow(),
        0.0,
        "no cap-overflow ⇒ the dissipation ledger must stay exactly 0"
    );
    eprintln!("[DISSIPATION] control: sub-cap metabolism dissipates nothing (ledger stays 0.0).");
}
