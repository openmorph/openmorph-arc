// SPDX-License-Identifier: MIT-0
//! v2.0 STRESS TEST — the HARD SCALABILITY LIMIT of the substrate on ONE axis: HYPERGRAPH WIDTH.
//!
//! A feedforward morph grown by `grow` on a fixed nonlinear target `tanh(2·x0·x1)` is, structurally, ONE
//! output cell (a dense readout) fed by W grown random-projection feature cells. W = the width. We sweep
//! `max_hidden` LARGE (32 → 8192 → until impractical) and measure, per width:
//!   • #units (`f.units.len()`), which equals W+1 once the pool is filled (birth_gate lowered so `grow` fills
//!     liberally — isolating the WIDTH cost from the surprise-gating rate);
//!   • wall-time to train a fixed schedule (reported only);
//!   • the cost of ONE stimulate/settle, in the organism's own unit-firings (asserted) and in wall-clock
//!     microseconds (reported only — a wall-clock varies by machine and load, so asserting on it makes a
//!     test that flakes under CI contention rather than one that measures the substrate);
//!   • held-out MAE;
//!   • whether the LIVING POOL (`metabolize_reliance` + `cull` each epoch over the target's basis) keeps
//!     #units bounded below max_hidden (self-sizing) or lets the basis blow up.
//!
//! THREE limits fall out, all real and quantitative:
//!   (1) A HARD CEILING at width ≈ settle-budget. The output unit is enqueued BEFORE the W features fire, so
//!       it needs the budget to reach ~W+2 firings to see fresh features. At a FIXED budget (the 100 the
//!       canonical proofs use) any width past ~budget silently truncates: the output never re-fires with the
//!       grown features, prediction goes stale, MAE collapses. Width is not "unhelpful" past the budget — it
//!       is BROKEN. The budget MUST scale with width.
//!   (2) With the budget scaled, accuracy PLATEAUS (random features: diminishing returns) while settle-cost
//!       grows ~LINEARLY in W — capacity stops buying accuracy long before it stops costing time.
//!   (3) The living pool DOES keep #units bounded — but on a genuinely nonlinear (DISTRIBUTED) basis it
//!       over-prunes: the answer leans a LITTLE on MANY features, each below the reliance/upkeep breakeven,
//!       so cull starves most of them and MAE degrades. Self-sizing works for a linear channel (substrate_grow),
//!       not for a distributed nonlinear one — the honest boundary of the reliance economy.
//!
//! PINNED to `Config::legacy()` (the 1.6 learning flip): every table here measures the RAW delta step. Under the 1.6 default
//! the step-stability cap removes the WIDTH-2 collapse (MAE ≈ 0.08 from width 32 to 2048, no peak at 128), and caps a
//! step that the raw rule took stably: at width 96 under the fixed budget, MAE 0.090 where the raw step reaches 0.042.
//!
//!   cargo test -p openmorph-core --test stress v2_scale_width -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::time::Instant;

use crate::common::Rng;

const NX: usize = 2; // input cells 0,1
const TGT: usize = NX; // the readout target cell = 2

fn truth(x: &[f32]) -> f32 {
    (2.0 * x[0] * x[1]).tanh()
}

/// The grown hidden-feature cells (written by a unit, beyond the raw inputs + the target). The prunable pool.
fn hidden_cells(f: &Field) -> Vec<usize> {
    let mut h: Vec<usize> = f
        .units
        .iter()
        .map(|u| u.write)
        .filter(|&c| c > TGT)
        .collect();
    h.sort_unstable();
    h.dedup();
    h
}

struct Metrics {
    peak_units: usize,
    final_units: usize,
    train_ms: f64,
    settle_us: f64,
    /// Mean unit-firings per settle — the organism's OWN subjective cost, returned by `stimulate`.
    /// Deterministic and machine-independent, unlike `settle_us`, so it is what the linearity claim is
    /// asserted on; the microseconds are kept for the printed report only.
    settle_events: f64,
    mae: f64,
}

/// Grow a width-`width` morph on `truth`, settling with `budget`, optionally running the living pool each
/// epoch after the fill. Returns the measured metrics. `birth_gate=0` forces `grow` to fill to `width`
/// (one feature per call) so we measure the cost of the WIDTH itself, not the surprise-gated birth rate.
fn run(width: usize, budget: usize, pool: bool, train_extra: usize, seed: u64, lr: f32) -> Metrics {
    let mut f = legacy_field(TGT + 1);
    f.config_mut().birth_gate = 0.0; // fill liberally to `width`
    f.config_mut().max_hidden = width;
    f.config_mut().feature_lr = 0.0; // classic random-feature regime: directions fixed, only the readout learns
                                     // This battery CHARACTERIZES the raw un-normalized wide-readout behaviour (the budget ceiling, the
                                     // un-normalized-lr collapse, the accuracy plateau). Disable the dimension-aware auto-normalization
                                     // (default `width_norm_min = 128`) so a width ≥ 128 keeps the raw step and the measured limits still show.
    f.config_mut().width_norm_min = usize::MAX;

    let mut rng = Rng::new(seed);
    let epochs = width + train_extra; // ~width calls to fill, then train_extra tuning steps
    let mut peak = 0usize;
    let t0 = Instant::now();
    for t in 0..epochs {
        let x = [rng.sym(), rng.sym()];
        let inputs = [(0usize, x[0]), (1usize, x[1])];
        f.grow(&inputs, TGT, truth(&x), lr, budget);
        peak = peak.max(f.n_units());
        if pool && t >= width {
            let hid = hidden_cells(&f);
            f.metabolize_reliance(&hid, 2.0, 1.0);
            f.cull(0.5);
        }
    }
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;
    let final_units = f.n_units();

    // Held-out MAE + per-settle wall-time (measured at the same `budget`).
    let mut rng2 = Rng::new(0xBEEF ^ seed);
    let n = 400usize;
    let mut mae = 0.0f64;
    let mut events = 0usize;
    let ts = Instant::now();
    for _ in 0..n {
        let x = [rng2.sym(), rng2.sym()];
        events += f.stimulate(&[(0, x[0]), (1, x[1])], budget);
        mae += (f.cells()[TGT] - truth(&x)).abs() as f64;
    }
    let settle_us = ts.elapsed().as_secs_f64() * 1e6 / n as f64;
    let settle_events = events as f64 / n as f64;
    mae /= n as f64;

    Metrics {
        peak_units: peak,
        final_units,
        train_ms,
        settle_us,
        settle_events,
        mae,
    }
}

#[test]
#[ignore = "stress: nightly"]
fn fixed_budget_settle_is_a_hard_width_ceiling() {
    // LIMIT (1): hold the settle budget at 100 (what the canonical proofs pass) and sweep width. The output
    // is enqueued before the W features, so it re-fires with fresh features only if the budget reaches ~W+2.
    // Past width ≈ budget the settle truncates BEFORE the output sees its own grown features → stale
    // prediction → MAE collapses. Width beyond the budget is not merely unhelpful, it is broken.
    eprintln!(
        "\n[WIDTH-1] FIXED settle budget = 100. The width ceiling is the budget, not capacity:"
    );
    eprintln!(
        "  {:>6} | {:>10} | {:>10} | {:>8}",
        "width", "#units", "settle_us", "MAE"
    );
    let mut res: Vec<(usize, f64)> = Vec::new();
    for &w in &[16usize, 64, 96, 128, 256, 512, 2048] {
        let m = run(w, 100, false, 1200, 0xA1CE + w as u64, 0.05);
        eprintln!(
            "  {:>6} | {:>10} | {:>9.1}  | {:>7.4}",
            w, m.final_units, m.settle_us, m.mae
        );
        res.push((w, m.mae));
    }
    let mae_at = |w: usize| res.iter().find(|(x, _)| *x == w).unwrap().1;
    eprintln!("[WIDTH-1] VERDICT: at a fixed budget of 100 the morph works only while width < budget-2 (~98): widths 64/96 fit (MAE ~0.04-0.05), but at width 128 (past the budget) the output never re-fires with its grown features — the settle exhausts the budget on the features first, the prediction stays stale, and MAE jumps to ~0.9. The settle BUDGET is a HARD width ceiling; width must be paid for in budget. (Width 16 is a separate, milder UNDER-capacity effect, MAE ~0.20.)");
    // Within the budget the readout fits; one step past the budget it COLLAPSES — the sharp ceiling.
    assert!(
        mae_at(96) < 0.07,
        "width 96 (< budget) must fit (MAE {:.4})",
        mae_at(96)
    );
    assert!(
        mae_at(128) > 0.5,
        "width 128 (> budget) must COLLAPSE — the settle truncates before the output sees its features (MAE {:.4})",
        mae_at(128)
    );
}

#[test]
#[ignore = "stress: nightly"]
fn scaled_budget_accuracy_plateaus_while_cost_grows_linearly() {
    // LIMIT (2): scale the budget with width (budget = width+8, so the output always re-fires with fresh
    // features — the LIMIT-(1) ceiling removed). Now width helps... only up to a point, then a SECOND,
    // independent collapse hits. Measure the accuracy curve and the per-settle cost curve, escalating width
    // until a single fill+train run exceeds a practical wall-time bound (the impractical frontier).
    eprintln!("\n[WIDTH-2] SCALED settle budget = width+8. Accuracy vs cost as width grows:");
    eprintln!(
        "  {:>6} | {:>8} | {:>9} | {:>10} | {:>10} | {:>10} | {:>8}",
        "width", "#units", "train_ms", "settle_us", "us/width", "fires/width", "MAE"
    );

    let global = Instant::now();
    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new(); // (width, settle_us, settle_events, mae)
    let mut best_mae = f64::MAX;
    let mut peak_width = 0usize;
    let mut impractical_at: Option<usize> = None;
    let mut last_train_ms = 0.0f64;

    for &w in &[32usize, 128, 512, 2048, 8192, 16384, 32768, 65536, 131072] {
        // Guard the total wall-time: if we're already deep, or the previous width's train ran long (the next
        // octave is ~4× the work — training is O(W^2)), stop and record THIS width as the impractical frontier.
        if global.elapsed().as_secs_f64() > 150.0 || last_train_ms > 20_000.0 {
            impractical_at = Some(w);
            eprintln!(
                "  {:>6} | (SKIPPED — projected fill+train ~ {:.0} ms > practical per-run bound; prior width took {:.0} ms)",
                w,
                last_train_ms * 4.0,
                last_train_ms
            );
            break;
        }
        let budget = w + 8;
        let m = run(w, budget, false, 800, 0x5EED + w as u64, 0.05);
        last_train_ms = m.train_ms;
        eprintln!(
            "  {:>6} | {:>8} | {:>8.0} | {:>9.1}  | {:>9.4} | {:>10.4} | {:>7.4}",
            w,
            m.final_units,
            m.train_ms,
            m.settle_us,
            m.settle_us / w as f64,
            m.settle_events / w as f64,
            m.mae
        );
        rows.push((w, m.settle_us, m.settle_events, m.mae));
        assert_eq!(
            m.final_units,
            w + 1,
            "with birth_gate 0 the pool fills to exactly width+1 (writer + W features), got {} at width {w}",
            m.final_units
        );
        if m.mae < best_mae - 0.002 {
            best_mae = m.mae;
            peak_width = w;
        }
    }

    // Cost linearity, measured in the organism's OWN time. A settle fires each of the W features once
    // and then the readout, so unit-firings per width is ~1 at every width — that IS the linearity claim,
    // and unlike wall-clock microseconds it is deterministic and machine-independent (the substrate's rule:
    // subjective event count, never a wall-clock, or the measurement stops being replayable and starts
    // failing on a loaded machine).
    let ratios: Vec<f64> = rows.iter().map(|&(w, _, e, _)| e / w as f64).collect();
    let rmin = ratios.iter().cloned().fold(f64::MAX, f64::min);
    let rmax = ratios.iter().cloned().fold(0.0, f64::max);
    let us_ratios: Vec<f64> = rows.iter().map(|&(w, s, _, _)| s / w as f64).collect();
    let umin = us_ratios.iter().cloned().fold(f64::MAX, f64::min);
    let umax = us_ratios.iter().cloned().fold(0.0, f64::max);
    let mae_at = |w: usize| {
        rows.iter()
            .find(|(x, _, _, _)| *x == w)
            .map(|&(_, _, _, m)| m)
    };
    eprintln!(
        "[WIDTH-2] unit-firings per width stays in [{:.4}, {:.4}] across the whole sweep -> the feedforward SETTLE is ~LINEAR in width (no super-linear settle blow-up). Wall-clock over the same sweep was [{:.4}, {:.4}] us/width — reported, not asserted, because it varies with machine load.",
        rmin, rmax, umin, umax
    );
    eprintln!(
        "[WIDTH-2] accuracy PEAKS at width ~ {peak_width} (best MAE {:.4}) and then COLLAPSES: past ~256 features the dense readout over the un-normalized wide random basis saturates and can no longer learn (see WIDTH-4), so MAE climbs back toward ~0.8-1.0 while settle/train cost keeps growing linearly. Capacity beyond ~128-256 is not merely wasted — it actively breaks the readout.",
        best_mae
    );
    match impractical_at {
        Some(w) => eprintln!(
            "[WIDTH-2] IMPRACTICAL FRONTIER: width {w} — a single fill+train run projected past the ~20 s practical bound (fill is O(W) grow-calls, each an O(W)-cost settle => training is O(W^2)). The last width that FINISHED did so in {:.1} s. Beyond the frontier the substrate is not settle-broken, it is TIME-broken (quadratic fill).",
            last_train_ms / 1000.0
        ),
        None => eprintln!(
            "[WIDTH-2] the sweep completed all widths within budget; the impractical frontier is above the largest width run ({:.1} s for the largest).",
            last_train_ms / 1000.0
        ),
    }

    // Assertions that DOCUMENT the limits (not hide them):
    assert!(
        rmax / rmin < 1.2,
        "settle cost per width should be constant in the organism's own time (linear settle): firings/width spread {:.3}",
        rmax / rmin
    );
    assert!(
        best_mae < 0.12,
        "even the BEST width should barely fit this random-feature regime (best MAE {best_mae:.4})"
    );
    assert!(
        peak_width <= 256,
        "accuracy should peak at a SMALL width and then collapse, not keep improving with capacity (peak {peak_width})"
    );
    if let Some(m) = mae_at(2048) {
        assert!(
            m > 0.5,
            "at width 2048 (scaled budget) accuracy must COLLAPSE despite the removed settle ceiling (MAE {m:.4}) — the second, learning-dynamics limit"
        );
    }
}

#[test]
#[ignore = "stress: nightly"]
fn the_living_pool_bounds_width_but_over_prunes_a_distributed_basis() {
    // LIMIT (3): does `metabolize_reliance` + `cull` keep #units bounded below max_hidden (self-sizing), or
    // does the basis blow up? It DOES bound it — but a genuinely nonlinear target needs a DISTRIBUTED basis
    // (many small weights), most below the reliance/upkeep breakeven, so the pool over-prunes and MAE degrades.
    eprintln!("\n[WIDTH-3] LIVING POOL (metabolize_reliance income=2 upkeep=1 + cull death=0.5) vs no pool:");
    eprintln!(
        "  {:>6} | {:>10} | {:>12} | {:>12} | {:>10} | {:>10}",
        "width", "peak", "final(pool)", "final(noPool)", "MAE pool", "MAE noPool"
    );
    let mut bounded_ok = true;
    let mut degraded = false;
    for &w in &[128usize, 512, 2048] {
        let budget = w + 8;
        let mp = run(w, budget, true, 1500, 0x9001 + w as u64, 0.05);
        let mn = run(w, budget, false, 1500, 0x9001 + w as u64, 0.05);
        eprintln!(
            "  {:>6} | {:>10} | {:>12} | {:>13} | {:>9.4} | {:>9.4}",
            w, mp.peak_units, mp.final_units, mn.final_units, mp.mae, mn.mae
        );
        // The pool bounds the population well below the filled width.
        if mp.final_units >= mp.peak_units {
            bounded_ok = false;
        }
        if mp.mae > mn.mae + 0.02 {
            degraded = true;
        }
    }
    eprintln!("[WIDTH-3] VERDICT: the reliance economy DOES self-size — cull keeps #units bounded far below max_hidden, so the basis never blows up. But on this DISTRIBUTED nonlinear target the answer leans a little on many features (most weights below the income·reliance>upkeep breakeven of 0.5), so the pool starves and culls features the readout was actually using → MAE degrades vs the un-pruned basis. Self-sizing to the true rank works for a LINEAR channel (substrate_grow: sized to the writers); for a distributed nonlinear basis it is a lossy regularizer, not free. That is the honest boundary of reliance-pruning.");
    assert!(
        bounded_ok,
        "the living pool must keep #units bounded below the filled width (self-sizing, no blow-up)"
    );
    // We assert the OBSERVED behaviour (documented, not hidden): the pool did prune the distributed basis.
    eprintln!(
        "[WIDTH-3] distributed-basis over-prune observed at the fitting width (128): {}",
        if degraded {
            "YES — pool MAE worse than no-pool MAE"
        } else {
            "NO — pool matched no-pool this run"
        }
    );
    assert!(
        degraded,
        "on a distributed nonlinear basis the reliance economy should over-prune (pool worse than no-pool at the fitting width)"
    );
}

#[test]
#[ignore = "stress: nightly"]
fn wide_readout_collapse_traces_to_the_unnormalized_learning_rate() {
    // DIAGNOSIS of LIMIT (2): the scaled-budget collapse past width ~256 is the DENSE readout's EFFECTIVE
    // step growing with width. The readout weights start at 0 (net = 0, output unsaturated); each learn_step
    // nudges ALL W weights, so the output's pre-activation moves by ~lr·Σ f_i² — which grows with W. Past a
    // few hundred features a fixed per-weight lr overshoots, drives the output tanh into saturation
    // (grad -> 0), and it sticks. The substrate does NOT normalize the readout by width. Shrink lr with
    // width and accuracy RETURNS — confirming the collapse is learning-dynamics, not representational.
    eprintln!("\n[WIDTH-4] DIAGNOSIS — is the wide collapse the un-normalized learning rate? Shrink lr, watch MAE recover:");
    eprintln!("  {:>6} | {:>10} | {:>8}", "width", "lr", "MAE");
    let mut recovered = f64::MAX;
    for &w in &[512usize, 2048] {
        let budget = w + 8;
        let mut best_w = f64::MAX;
        for &lr in &[0.05f32, 0.01, 0.002, 0.0005] {
            let m = run(w, budget, false, 2500, 0x77 + w as u64, lr);
            eprintln!("  {:>6} | {:>10.4} | {:>7.4}", w, lr, m.mae);
            best_w = best_w.min(m.mae);
        }
        if w == 512 {
            recovered = best_w;
        }
    }
    eprintln!(
        "[WIDTH-4] VERDICT: at the DEFAULT lr 0.05 a width-512 readout collapses (~0.77 MAE), but shrinking lr recovers it to {:.4}. The collapse is the local delta rule's effective step scaling with width — the substrate has NO width-normalization of the dense readout, so a single fixed `lr`/`Config` cannot serve small AND large morphs. The usable-width band (this task, default lr) is only ~[64, 256]; beyond it you must co-scale lr ~ 1/width or the readout saturates.",
        recovered
    );
    assert!(
        recovered < 0.12,
        "a smaller lr must recover the wide readout (best width-512 MAE {recovered:.4}) — confirming the collapse is the un-normalized learning rate, not lost capacity"
    );
}

/// A `Field` of `n` cells on [`Config::legacy`](openmorph_core::Config::legacy) — the pre-1.6 learning regime this
/// proof's claim is about (see the module docs).
fn legacy_field(n: usize) -> Field {
    let mut f = Field::new(n);
    f.configure(openmorph_core::Config::legacy());
    f
}
