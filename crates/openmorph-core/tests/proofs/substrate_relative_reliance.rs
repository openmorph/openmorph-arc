// SPDX-License-Identifier: MIT-0
//! RELATIVE RELIANCE breakeven (opt-in `Config::relative_reliance`, byte-identical off) — the Defect-1 fix
//! (MorphTrading probe 0001). The living pool pays a grown feature income ∝ `reliance_of` (Σ|reader weight|) and
//! culls it when energy starves; the survival breakeven is the ABSOLUTE `pool_upkeep / pool_income = 0.125`. But
//! `reliance_of` is unnormalised, so on a LOW-AMPLITUDE task (the readout's solution weights are ~0.04) the whole
//! WORKING basis sits below 0.125 and is culled indiscriminately — a silent false negative (the mechanism works,
//! the economy throws the answer away). Normalising reliance to a SHARE of the pool's total makes the breakeven
//! amplitude-INVARIANT: "carry ≥ 12.5% of what the readout relies on", which a working basis does at any scale.
//!
//! Since 1.6 the default economy is `RelianceNorm::Mean` (reliance relative to the pool MEAN), which keeps the
//! low-amplitude basis too and — unlike the share of the TOTAL (`relative_reliance`, now `RelianceNorm::Share`) — has
//! no survivor cap: a share of `1/n` pays `pool_income/n`, so 12 or more EQUALLY useful features all starve under
//! Share, while each earns the full `pool_income` under Mean. Each arm pins its rule explicitly.
//!
//!   cargo test -p openmorph-core --test proofs substrate_relative_reliance -- --nocapture

use openmorph_core::{Field, RelianceNorm};

/// Three low-amplitude features (cells 1,2,3), each read by a single readout (cell 4) with weight 0.04 — so each
/// feature's `reliance_of` is exactly 0.04, far below the absolute breakeven 0.125. Run the living-pool economy
/// (`metabolize_reliance` + `cull`) and return (surviving unit count, effective_rank).
fn run(rule: RelianceNorm) -> (usize, f32) {
    let mut f = Field::new(5);
    f.config_mut().reliance_norm = rule;
    f.add_unit(vec![0], vec![0.5], 0.0, 1); // feature units (their own weights are irrelevant to reliance_of)
    f.add_unit(vec![0], vec![0.5], 0.0, 2);
    f.add_unit(vec![0], vec![0.5], 0.0, 3);
    f.add_unit(vec![1, 2, 3], vec![0.04, 0.04, 0.04], 0.0, 4); // the readout: reliance_of(feature) = 0.04

    let basis = [1usize, 2, 3];
    // default living-pool rates: income 2.0, upkeep 0.25 (breakeven 0.125), death 0.1, grace 0 (no reserve).
    for _ in 0..40 {
        f.metabolize_reliance(&basis, 2.0, 0.25);
        f.cull_after(0.1, 0);
    }
    (f.n_units(), f.effective_rank())
}

#[test]
fn absolute_breakeven_culls_a_low_amplitude_basis_relative_keeps_it() {
    let (abs_units, abs_rank) = run(RelianceNorm::Absolute); // absolute (the pre-1.6 default)
    let (rel_units, rel_rank) = run(RelianceNorm::Share); // relative reliance (the share of the total)
    let (mean_units, mean_rank) = run(RelianceNorm::Mean); // the 1.6 default: relative to the pool mean

    eprintln!("[REL-RELIANCE] low-amplitude basis (feature reliance 0.04, breakeven 0.125):");
    eprintln!(
        "[REL-RELIANCE]   ABSOLUTE: surviving units {abs_units}  effective_rank {abs_rank:.2}"
    );
    eprintln!(
        "[REL-RELIANCE]   RELATIVE: surviving units {rel_units}  effective_rank {rel_rank:.2}"
    );
    eprintln!(
        "[REL-RELIANCE]   MEAN:     surviving units {mean_units}  effective_rank {mean_rank:.2}"
    );

    // ABSOLUTE breakeven: each feature earns 2·0.04 − 0.25 = −0.17/event and starves → the 3 features are culled,
    // leaving only the readout (which is not metabolised). The economy destroyed the working basis.
    assert_eq!(
        abs_units, 1,
        "absolute breakeven culls the entire low-amplitude basis (only the readout survives): {abs_units} units"
    );
    assert!(
        abs_rank < 0.5,
        "with the basis culled, effective_rank collapses: {abs_rank:.3}"
    );

    // RELATIVE breakeven: each feature carries share 0.04/0.12 = 0.333 ≥ 0.125 and earns 2·0.333 − 0.25 = +0.417 →
    // all three survive regardless of the tiny absolute weight. The working basis is kept.
    assert_eq!(
        rel_units, 4,
        "relative reliance keeps the whole low-amplitude basis (3 features + readout): {rel_units} units"
    );
    assert!(
        rel_rank > 2.5,
        "the surviving basis carries near-full effective_rank (~3): {rel_rank:.3}"
    );
    // MEAN (the 1.6 default): each feature carries 0.04/0.04 = 1 of the mean and earns 2 − 0.25 → all three survive.
    assert_eq!(
        mean_units, 4,
        "the mean-relative economy keeps the whole low-amplitude basis too: {mean_units} units"
    );
    assert!(
        mean_rank > 2.5,
        "the surviving basis carries near-full effective_rank (~3): {mean_rank:.3}"
    );

    eprintln!("[REL-RELIANCE] VERDICT: an absolute upkeep/income breakeven against unnormalised reliance culls a low-amplitude working basis (units 4→1, rank {abs_rank:.2}); normalising reliance to a pool SHARE makes the breakeven amplitude-invariant and keeps the basis (units 4, rank {rel_rank:.2}). Byte-identical when relative_reliance=off. Fix VALIDATED in-core; needs a pre-registered live replication before becoming the default.");
}

#[test]
fn a_high_amplitude_basis_survives_either_way() {
    // Sanity / no-harm: when the solution IS high-amplitude (weight 0.5 > breakeven), both modes keep it — the fix
    // only changes the low-SNR regime, it doesn't destabilise the well-scaled one.
    let build = |rule: RelianceNorm| {
        let mut f = Field::new(3);
        f.config_mut().reliance_norm = rule;
        f.add_unit(vec![0], vec![0.5], 0.0, 1);
        f.add_unit(vec![1], vec![0.5], 0.0, 2); // reliance_of(1) = 0.5 (>> 0.125)
        for _ in 0..40 {
            f.metabolize_reliance(&[1usize], 2.0, 0.25);
            f.cull_after(0.1, 0);
        }
        f.n_units()
    };
    assert_eq!(
        build(RelianceNorm::Absolute),
        2,
        "absolute keeps a high-amplitude feature"
    );
    assert_eq!(
        build(RelianceNorm::Share),
        2,
        "relative also keeps it (share 1.0)"
    );
    assert_eq!(
        build(RelianceNorm::Mean),
        2,
        "mean keeps it (1.0 of the mean)"
    );
}

/// The deprecated bool is an alias of `Share` and wins over `reliance_norm`.
#[test]
fn relative_reliance_is_an_alias_of_share() {
    let mut f = Field::new(5);
    f.config_mut().reliance_norm = RelianceNorm::Mean;
    f.config_mut().relative_reliance = true;
    assert_eq!(f.config().effective_reliance_norm(), RelianceNorm::Share);
}

/// `k` EQUALLY useful features (each read with weight 0.3, well above the absolute breakeven 0.125), metabolised
/// under `rule`: how many units survive (features + the readout)?
fn equal_pool(k: usize, rule: RelianceNorm) -> usize {
    let mut f = Field::new(k + 2);
    f.config_mut().reliance_norm = rule;
    for i in 0..k {
        f.add_unit(vec![0], vec![0.5], 0.0, 1 + i);
    }
    f.add_unit((1..=k).collect(), vec![0.3; k], 0.0, k + 1);
    let basis: Vec<usize> = (1..=k).collect();
    for _ in 0..40 {
        f.metabolize_reliance(&basis, 2.0, 0.25);
        f.cull_after(0.1, 0);
    }
    f.n_units()
}

/// The share of the TOTAL caps the pool at `pool_income/pool_upkeep = 8` equally useful features: 9 break even,
/// 12 all starve. The MEAN rule (the default) and the absolute rule keep any number of them.
#[test]
fn the_mean_economy_has_no_survivor_cap() {
    for k in [3usize, 8, 12, 32] {
        let (a, s, m) = (
            equal_pool(k, RelianceNorm::Absolute),
            equal_pool(k, RelianceNorm::Share),
            equal_pool(k, RelianceNorm::Mean),
        );
        eprintln!(
            "[REL-RELIANCE] {k:2} equally useful features: survivors (incl. readout) absolute {a}, share {s}, mean {m}"
        );
        assert_eq!(
            a,
            k + 1,
            "absolute keeps {k} features that clear its breakeven"
        );
        assert_eq!(m, k + 1, "mean keeps all {k} equally useful features");
        if k >= 12 {
            assert_eq!(s, 1, "share of the total starves all {k} equal features");
        } else {
            assert_eq!(s, k + 1, "share keeps {k} ≤ 8 equal features");
        }
    }
}
