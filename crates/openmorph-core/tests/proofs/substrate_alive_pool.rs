// SPDX-License-Identifier: MIT-0
//! THE LIVING-POOL ECONOMY under `alive` — the fix for MorphLab's "alive strangles XOR" finding.
//!
//! `alive(Event::Perceive)` grows the basis, then self-sizes it with the reliance economy each event. The
//! survival BREAKEVEN is `pool_upkeep / pool_income` in reliance units. With the old buried rates (upkeep 1.0,
//! income 2.0) that breakeven was 0.5 — so the economy culled the used-but-MODEST features of a DISTRIBUTED
//! solution (a nonlinear task like XOR spreads its answer across many random-projection features, each with a
//! small readout weight), and alive plateaued at chance (~45%) while raw `grow` (no economy) hit ~95%. The fix
//! is NOT new machinery: the economy rates are `Config` levers, and the tuned default upkeep (0.25 → breakeven
//! 0.125) prunes only IGNORED capacity (|weight| ~ 0), never a distributed solution mid-formation. alive now
//! matches raw grow on XOR AND still self-sizes an over-provisioned linear task.
//!
//! PINNED to `reliance_norm` at its `Config::legacy()` value (`RelianceNorm::Absolute`) — every other lever is the 1.6
//! default (the 1.6 learning flip): both proofs are about the ABSOLUTE reliance economy's rates (here: tuned 13 units →
//! 94 %, OLD rates 14 units → 51 %; the linear task keeps 1 unit). Under the full 1.6 default (the mean-relative economy)
//! the OLD rates no longer strangle XOR (15 units → 97 %; the tuned rates: 17 units → 94 %) because a mean-relative
//! breakeven is amplitude-invariant, and the over-provisioned linear task keeps 11 of its 16 units (read-back err 0.028):
//! the mean-relative economy does not prune features the readout ignores EQUALLY.
//!
//!   cargo test -p openmorph-core --test proofs substrate_alive_pool -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use openmorph_core::Event;

use crate::common::Rng;

// XOR-like: classes are the two diagonals — not linearly separable, needs grown depth.
fn label(x0: f32, x1: f32) -> f32 {
    if x0 * x1 > 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn xor_acc(f: &mut Field, seed: u64) -> f64 {
    let mut rng = Rng::new(seed ^ 0xFFFF);
    let mut correct = 0usize;
    for _ in 0..2000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], 200);
        let pred = if f.cells()[2] > 0.0 { 1.0 } else { -1.0 };
        if (pred - label(x0, x1)).abs() < 1e-6 {
            correct += 1;
        }
    }
    correct as f64 / 2000.0
}

/// Train an alive morph on XOR; `old_rates` uses the pre-fix buried economy (upkeep 1.0, death 0.5).
fn alive_xor(old_rates: bool, seed: u64) -> (usize, f64) {
    let mut f = pinned_field(3);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = 16;
    if old_rates {
        f.config_mut().pool_upkeep = 1.0;
        f.config_mut().pool_death = 0.5;
    }
    let mut rng = Rng::new(seed);
    for _ in 0..8000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.alive(Event::Perceive {
            inputs: &[(0, x0), (1, x1)],
            target_cell: 2,
            target: label(x0, x1),
            lr: 0.05,
            budget: 200,
        });
    }
    let u = f.n_units();
    (u, xor_acc(&mut f, seed))
}

#[test]
fn alive_cracks_xor_with_the_tuned_pool_economy() {
    let (u_new, acc_new) = alive_xor(false, 0x111); // tuned default rates
    let (u_old, acc_old) = alive_xor(true, 0x111); // the pre-fix buried rates

    eprintln!(
        "[POOL] alive on XOR — tuned default: {u_new} units → {:.0}%   |   OLD rates (upkeep 1.0/death 0.5): {u_old} units → {:.0}%",
        acc_new * 100.0,
        acc_old * 100.0
    );
    eprintln!("[POOL] VERDICT: the old upkeep set the survival breakeven at reliance 0.5, culling the used-but-modest features of XOR's DISTRIBUTED basis (alive plateaued at chance). The tuned upkeep (breakeven 0.125) prunes only IGNORED capacity, so alive self-sizes WITHOUT destroying a nonlinear solution.");

    assert!(
        acc_new > 0.90,
        "alive cracks XOR with the tuned pool economy (got {acc_new:.2})"
    );
    assert!(
        acc_old < 0.75,
        "the old rates strangle the distributed solution (got {acc_old:.2}) — documents the bug the fix closes"
    );
}

#[test]
fn alive_still_self_sizes_an_overprovisioned_linear_task() {
    // A purely LINEAR map (y = 0.6·x) with a generous hidden ceiling. The hidden features are IGNORED (the
    // readout leans on the linear term), so the living pool must prune them — self-sizing, not hoarding.
    let mut f = pinned_field(2);
    f.config_mut().max_hidden = 16;
    let mut rng = Rng::new(0x222);
    for _ in 0..6000 {
        let x = rng.sym();
        f.alive(Event::Perceive {
            inputs: &[(0, x)],
            target_cell: 1,
            target: 0.6 * x,
            lr: 0.1,
            budget: 50,
        });
    }
    // Read it back.
    f.stimulate(&[(0, 1.0)], 50);
    let err = (f.cells()[1] - 0.6).abs();
    eprintln!(
        "[POOL] linear task: alive kept {} unit(s), read-back err {err:.3} (self-sized, ceiling was 16)",
        f.n_units()
    );
    assert!(
        err < 0.1,
        "alive still learns the linear map (err {err:.3})"
    );
    assert!(
        f.n_units() < 8,
        "the pool prunes IGNORED capacity — self-sizes well below the 16 ceiling (got {} units)",
        f.n_units()
    );
}

/// The genome this proof's claim is about (see the module docs): the 1.6 default with `reliance_norm` at the
/// [`Config::legacy`](openmorph_core::Config::legacy) value.
fn pinned_config() -> openmorph_core::Config {
    let legacy = openmorph_core::Config::legacy();
    openmorph_core::Config {
        reliance_norm: legacy.reliance_norm, // the ABSOLUTE reliance economy
        ..openmorph_core::Config::default()
    }
}

/// A `Field` of `n` cells on [`pinned_config`].
fn pinned_field(n: usize) -> Field {
    let mut f = Field::new(n);
    f.configure(pinned_config());
    f
}
