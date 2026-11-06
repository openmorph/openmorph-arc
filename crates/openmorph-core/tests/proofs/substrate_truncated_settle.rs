// SPDX-License-Identifier: MIT-0
//! A TRUNCATED SETTLE IS REPORTED — the anytime `budget` of a settle used to cut the cascade short in silence: the
//! call returned a not-yet-settled prediction and nothing said so (gap-learning F5: with 64 grown features a budget
//! of 50 left XOR at chance). `Field::truncated_settles()` now counts every settle that stopped at its budget with
//! firings still pending, at no cost to a settle that completes.
//!
//! The proof grows an XOR pool through `alive`, then settles one fresh input under a sweep of budgets around the
//! cost of the full settle (the grown hidden features + 2 firings — the readout fires first on the stale features,
//! then again once they have settled):
//!
//! 1. a budget that covers the settle reports nothing and gives the settled value;
//! 2. every smaller budget is counted, and its output differs from the settled one;
//! 3. the counter is a diagnostic: it never enters a snapshot.
//!
//! And the 1.6 default avoids the truncation in the first place: `Config::grow_within_budget` births a feature only
//! while a settle of the inputs still fits the budget, so a budget below the pool's settle cost BOUNDS the pool
//! instead of cutting every settle short (`a_small_budget_bounds_the_pool_instead_of_truncating`).
//!
//!   cargo test -p openmorph-core --test proofs substrate_truncated_settle -- --nocapture

use openmorph_core::{Config, Event, Field};

fn grown_xor() -> Field {
    let mut f = Field::new(3);
    f.rng_seed(5);
    let mut s = 0x1234u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    for _ in 0..3000 {
        let (a, b) = (r(), r());
        f.alive(Event::Perceive {
            inputs: &[(0, a), (1, b)],
            target_cell: 2,
            target: if a * b > 0.0 { 0.8 } else { -0.8 },
            lr: 0.1,
            budget: 1_000_000,
        });
    }
    f
}

#[test]
fn a_settle_cut_short_by_its_budget_is_counted() {
    let f = grown_xor();
    assert_eq!(
        f.truncated_settles(),
        0,
        "the unbounded training settles never truncate"
    );
    let hidden = f.n_units() - 1;
    assert!(hidden >= 4, "the XOR pool must have grown hidden features");
    let probe = [(0, 0.3), (1, -0.7)];

    let mut full = f.clone();
    let cost = full.stimulate(&probe, usize::MAX);
    let settled = full.cells()[2];
    eprintln!("[TRUNCATED] {hidden} hidden features, a full settle fires {cost} units");
    assert_eq!(
        full.truncated_settles(),
        0,
        "an unbounded settle is never truncated"
    );
    assert_eq!(
        cost,
        hidden + 2,
        "the readout fires, the features, the readout again"
    );

    for budget in 1..=cost + 2 {
        let mut g = f.clone();
        g.stimulate(&probe, budget);
        let cut = g.truncated_settles();
        let dy = (g.cells()[2] - settled).abs();
        if budget >= cost {
            assert_eq!(
                cut, 0,
                "budget {budget} covers the settle — nothing to report"
            );
            assert_eq!(
                dy, 0.0,
                "budget {budget}: a complete settle is the fixed point"
            );
        } else {
            assert_eq!(
                cut, 1,
                "budget {budget} < {cost} must be reported as truncated"
            );
            assert!(
                dy > 0.0,
                "budget {budget}: the truncated output should differ from the settled one"
            );
        }
    }

    // A diagnostic, not state: a snapshot round trip does not carry it.
    let mut g = f.clone();
    g.stimulate(&probe, 1);
    assert_eq!(g.truncated_settles(), 1);
    let back: Field = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    assert_eq!(back.truncated_settles(), 0, "the counter is not serialized");
}

/// Train XOR through `alive` at settle `budget` on `cfg` and score it on fresh inputs at the same budget. Returns
/// (units, truncated settles over training + evaluation, sign accuracy).
fn xor_at_budget(cfg: Config, budget: usize) -> (usize, u64, f32) {
    let mut f = Field::new(3);
    f.configure(cfg);
    f.rng_seed(5);
    let mut s = 0x1234u64;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    for _ in 0..6000 {
        let (a, b) = (r(), r());
        f.alive(Event::Perceive {
            inputs: &[(0, a), (1, b)],
            target_cell: 2,
            target: if a * b > 0.0 { 0.8 } else { -0.8 },
            lr: 0.1,
            budget,
        });
    }
    let mut hits = 0;
    const TEST: usize = 1000;
    for _ in 0..TEST {
        let (a, b) = (r(), r());
        f.stimulate(&[(0, a), (1, b)], budget);
        hits += usize::from((f.cells()[2] > 0.0) == (a * b > 0.0));
    }
    (
        f.n_units(),
        f.truncated_settles(),
        hits as f32 / TEST as f32,
    )
}

#[test]
fn a_small_budget_bounds_the_pool_instead_of_truncating() {
    const BUDGET: usize = 20; // the default pool would grow to 64 features — a 66-firing settle
    let d = Config::default();
    assert!(
        d.grow_within_budget,
        "the 1.6 default bounds growth by the budget"
    );
    let (units, cut, acc) = xor_at_budget(d.clone(), BUDGET);
    let unbounded = Config {
        grow_within_budget: false,
        ..d.clone()
    };
    let (u_units, u_cut, u_acc) = xor_at_budget(unbounded, BUDGET);
    let (big_units, _, big_acc) = xor_at_budget(d, 1_000_000);
    eprintln!(
        "[BUDGET-BOUND] XOR at budget {BUDGET}: bounded {units} units, {cut} truncated settles, acc {acc:.3} | \
         unbounded {u_units} units, {u_cut} truncated, acc {u_acc:.3} | budget 1e6: {big_units} units, acc {big_acc:.3}"
    );
    assert!(
        units < BUDGET,
        "the bounded pool fits its budget ({units} units, budget {BUDGET})"
    );
    assert_eq!(cut, 0, "a bounded pool never truncates a settle");
    assert!(
        u_units >= BUDGET && u_cut > 0,
        "without the bound the pool outgrows the budget and its settles are cut short ({u_units} units, {u_cut})"
    );
    assert!(
        acc > 0.9,
        "the bounded pool learns XOR on complete settles ({acc:.3})"
    );
    assert!(
        acc > u_acc + 0.2,
        "bounding the pool beats truncating its settles ({acc:.3} vs {u_acc:.3})"
    );
}

/// The bound holds at EVERY budget, the smallest included: the check compares the settle cost AFTER a birth with the
/// budget. It used to compare the cost before, so at budget 2 the first feature (cost 1 → 3: writer, feature, writer
/// re-fire) was born and every later settle truncated (review of #494, core #4: 298 of 300 settles cut).
#[test]
fn no_budget_is_too_small_for_the_bound() {
    for budget in 1..=8 {
        let (units, cut, _) = xor_at_budget(Config::default(), budget);
        assert_eq!(
            cut, 0,
            "budget {budget}: {units} units, {cut} truncated settles"
        );
    }
}
