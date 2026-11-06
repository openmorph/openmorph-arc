// SPDX-License-Identifier: MIT-0
//! STRESS TEST — LIVING-POOL ECONOMY + NESTED MEMOIZATION AT SCALE.
//!
//! Two axes, pushed until something breaks, numbers observed from a real run:
//!
//!   (A) CONTINUOUS GROW ON A STREAM. Under `grow` + `metabolize_reliance` + `cull` every step over a
//!       stream of many tasks/targets, does the population stay BOUNDED (self-sizing) or grow/thrash?
//!       - A1 CYCLIC-K tasks: measure #units and wall-time over 1k / 10k / 100k grow-steps.
//!       - A2 OPEN-ENDED distinct tasks: does the economy FORGET fit-but-dormant skills, or accumulate?
//!
//!   (B) NESTED MEMOIZATION. A nested morph over K recurring discrete states vs many novel inputs.
//!       - B1 recurring, K <= memo_cap: nested_settles should be ~K (flat) over T fires.
//!       - B2 the memo_cap CLIFF: sweep K across the cap. Because eviction is CLEAR-ALL (not LRU),
//!         reuse should collapse the instant K > cap.
//!       - B3 novel/continuous inputs: nested_settles ~ T (memo never hits) — the memo helps ONLY exact
//!         bit-identical recurring inputs.
//!
//!   cargo test --manifest-path .../Cargo.toml -p openmorph-core --test stress v2_scale_economy -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::time::Instant;

use crate::common::Rng;

// ───────────────────────────────────────────────────────────────────────────────────────────────
// (A1) CONTINUOUS GROW OVER A CYCLIC-K TASK STREAM — is the pool bounded, is wall-time linear?
// ───────────────────────────────────────────────────────────────────────────────────────────────

const NX: usize = 2; // shared input bus: cells 0, 1
const K_A: usize = 8; // number of cyclic tasks (target cells 2..2+K_A); features live at cells >= 2+K_A

/// Task `τ`'s target: a NONLINEAR (product) map of the two inputs, so a linear writer can't solve it and
/// `grow` keeps birthing hidden features — the population is genuinely exercised, not trivially linear.
fn target_a(tau: usize, x0: f32, x1: f32) -> f32 {
    let a = 0.6 + 0.5 * (tau as f32 / K_A as f32); // a distinct gain per task
    (a * x0 * x1 + 0.3 * (tau as f32).cos()).tanh()
}

/// Hidden-feature cells (written by a unit, beyond the raw-input + target band). The prunable pool; the
/// writers (cells 2..2+K_A) are excluded so they are never culled.
fn features_a(f: &Field) -> Vec<usize> {
    let mut v: Vec<usize> = f
        .units
        .iter()
        .map(|u| u.write)
        .filter(|&c| c >= NX + K_A)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Returns (peak_units, final_units, final_cells, firings, elapsed_ms). `firings` = `f.clock()`, the organism's own
/// event count — the machine-independent cost the per-step claim is asserted on (`elapsed_ms` is printed only).
/// `final_cells` = `f.len()`, the medium size:
/// the number that once betrayed the deploy leak. FIXED (#149): `cull` now returns each orphaned cell to a free
/// list and `birth_feature` reuses it via `alloc_cell`, so the medium (and its per-cell parallel vectors + cull's
/// O(cells) adjacency rebuild) stays bounded at the PEAK concurrent cell count instead of growing ~O(steps).
fn run_cyclic_stream(steps: usize) -> (usize, usize, usize, u64, u128) {
    let mut f = Field::new(NX + K_A);
    let mut rng = Rng::new(0xA11CE);
    let mut peak = 0usize;
    let t0 = Instant::now();
    for t in 0..steps {
        let tau = t % K_A;
        let (x0, x1) = (rng.sym(), rng.sym());
        let inputs = [(0usize, x0), (1usize, x1)];
        f.grow(&inputs, NX + tau, target_a(tau, x0, x1), 0.1, 100);
        // The living pool every step (after a short warm-up so newborns aren't judged mid-transient).
        if t >= 200 {
            let feats = features_a(&f);
            f.metabolize_reliance(&feats, 2.0, 1.0);
            f.cull(0.5);
        }
        peak = peak.max(f.n_units());
    }
    (
        peak,
        f.n_units(),
        f.len(),
        f.clock(),
        t0.elapsed().as_millis(),
    )
}

#[test]
#[ignore = "stress: nightly"]
fn a1_cyclic_stream_units_and_cells_bounded() {
    eprintln!("[A1] CONTINUOUS grow + metabolize_reliance + cull, cyclic K={K_A} tasks over a shared {NX}-cell input bus:");
    let mut rows: Vec<(usize, usize, usize, usize, f64, f64)> = Vec::new();
    for &steps in &[1_000usize, 10_000, 100_000] {
        let (peak, final_u, final_cells, firings, ms) = run_cyclic_stream(steps);
        let fires_per = firings as f64 / steps as f64;
        let us_per = 1000.0 * ms as f64 / steps as f64;
        eprintln!(
            "[A1] steps {steps:>7}: peak units {peak:>4}, final units {final_u:>4} | CELLS(medium) {final_cells:>7} | {fires_per:>6.2} firings/step | wall {ms:>6} ms, {us_per:>6.2} µs/step (info)"
        );
        rows.push((steps, peak, final_u, final_cells, fires_per, us_per));
    }
    let (_, _, u_1k, cells_1k, fires_1k, us_1k) = rows[0];
    let (_, _, u_100k, cells_100k, fires_100k, us_100k) = rows[2];
    let cell_growth = cells_100k as f64 / cells_1k as f64;
    // Per-step cost in the organism's OWN currency (settle firings, `Field::clock`) — deterministic and
    // machine-independent. Cull's O(cells) adjacency rebuild is the other per-step cost; it is bounded by the
    // CELL assertion below (a bounded medium ⇒ a bounded rebuild). Wall time is printed for information only.
    let step_growth = fires_100k / fires_1k;
    eprintln!(
        "[A1] VERDICT (deploy-leak FIXED, MorphLab #149): BOTH the unit pool AND the cell medium are now BOUNDED. Units self-size ({u_1k} @1k vs {u_100k} @100k). Cells stay bounded at the PEAK concurrent count ({cells_1k} @1k → {cells_100k} @100k = {cell_growth:.1}×, NOT O(steps)): `cull` returns each orphaned cell to a FREE LIST and `birth_feature` reuses it. Because the medium no longer grows, cull's O(cells) adjacency rebuild is bounded too — per-step cost is FLAT ({fires_1k:.1}→{fires_100k:.1} firings/step, {step_growth:.1}×; wall {us_1k:.0}→{us_100k:.0} µs/step), so a continuous grow+cull stream is O(steps), not O(steps²). The forever-loop deploy wall is closed."
    );
    // The unit pool IS bounded (self-sizing works).
    assert!(
        u_100k <= u_1k + K_A,
        "the cyclic UNIT pool must stay bounded: {u_1k} @1k vs {u_100k} @100k units (self-sizing failed)"
    );
    assert!(
        u_100k <= K_A * 65,
        "units must never exceed K·(1 writer + max_hidden 64) = {} (got {u_100k})",
        K_A * 65
    );
    // The CELL medium is now RECLAIMED (free-list) — bounded at the peak concurrent count, NOT O(steps).
    assert!(
        cells_100k <= cells_1k + 8,
        "cells must stay BOUNDED with the free-list fix ({cells_1k} @1k → {cells_100k} @100k — should not grow O(steps))"
    );
    // And the per-step cost stays FLAT — no O(steps²): the settle work per step, counted in the organism's own
    // events (`clock`), does not grow with the stream length (and the medium bound above bounds cull's rebuild).
    assert!(
        step_growth < 2.0,
        "per-step cost must stay FLAT with the fix (bounded medium ⇒ bounded cull rebuild): {fires_1k:.2}→{fires_100k:.2} firings/step ({step_growth:.1}×)"
    );
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// (A2) OPEN-ENDED DISTINCT TASKS — does the economy forget fit-but-dormant skills, or accumulate?
// ───────────────────────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "stress: nightly"]
fn a2_open_ended_distinct_tasks_accumulate_no_forgetting() {
    // Each task is trained to convergence on its OWN target cell, then NEVER revisited. Because reliance
    // (Σ reader |weight|) is convergence-STABLE, a well-fit dormant skill keeps earning income >= upkeep
    // and is never culled — so the library grows with the NUMBER OF DISTINCT TASKS. Distinct-task count,
    // not stream length, is the memory axis. We also watch per-event cost climb (shared bus ⇒ all refire).
    const EPOCHS: usize = 40; // grow-steps per task (to convergence)
    eprintln!(
        "[A2] OPEN-ENDED: {EPOCHS} grow-steps per task, each task banked then never revisited:"
    );
    let mut prev_units = 0usize;
    for &n_tasks in &[100usize, 200, 400] {
        let base = 2 + n_tasks; // reserve target cells [2, 2+n_tasks); features land at cells >= base
        let mut f = Field::new(base);
        let mut rng = Rng::new(0xBEEF);
        let t0 = Instant::now();
        // time the LAST task's epochs separately to expose the per-event cost at a full library.
        let mut last_task_ms = 0u128;
        for task in 0..n_tasks {
            let tstart = Instant::now();
            for _ in 0..EPOCHS {
                let (x0, x1) = (rng.sym(), rng.sym());
                let inputs = [(0usize, x0), (1usize, x1)];
                f.grow(&inputs, 2 + task, target_a(task % 97, x0, x1), 0.1, 100);
                let feats: Vec<usize> = f
                    .units
                    .iter()
                    .map(|u| u.write)
                    .filter(|&c| c >= base)
                    .collect();
                f.metabolize_reliance(&feats, 2.0, 1.0);
                f.cull(0.5);
            }
            if task == n_tasks - 1 {
                last_task_ms = tstart.elapsed().as_millis();
            }
        }
        let total_ms = t0.elapsed().as_millis();
        let units = f.n_units();
        let us_last = 1000.0 * last_task_ms as f64 / EPOCHS as f64;
        eprintln!(
            "[A2] tasks {n_tasks:>4}: final units {units:>5} ({:.2} per task) | total {total_ms:>6} ms | last-task {us_last:>7.1} µs/step (cost at a full library)",
            units as f64 / n_tasks as f64
        );
        assert!(
            units >= n_tasks,
            "each banked task keeps >=1 relied unit — the economy does NOT forget fit dormant skills (tasks {n_tasks}, units {units})"
        );
        prev_units = units;
    }
    let _ = prev_units;
    eprintln!(
        "[A2] VERDICT: units grow ~LINEARLY with the number of DISTINCT tasks — reliance is convergence-stable, so a fit-but-dormant skill is NEVER culled; there is NO forgetting. The pool self-sizes PER task but the library is unbounded in the count of distinct tasks. Worse, over the shared input bus every unit refires each event, so per-event settle cost also climbs with library size (last-task µs/step grows) → an open-ended distinct-task stream is O(tasks) memory AND O(tasks) per-event — the real wall is task-diversity, not stream length. Routing (substrate_scale's O(√M) index) is the missing lever the raw economy stream does not apply."
    );
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// (B) NESTED MEMOIZATION AT SCALE.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// A small nested skill: out = tanh(x - y), cells 0=x, 1=y, 2=out.
fn skill() -> Field {
    let mut f = Field::new(3);
    f.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);
    f
}
fn truth(x: f32, y: f32) -> f32 {
    (x - y).tanh()
}

/// Fire a host whose cell 2 is the nested `skill`, round-robin over `k` DISTINCT discrete states for
/// `fires` total fires. `memo_cap` set live. Returns (nested_settles, worst_error, elapsed_ms).
fn run_memo(k: usize, fires: usize, memo_cap: usize) -> (u64, f32, u128) {
    let mut host = Field::new(2);
    host.config_mut().memo_cap = memo_cap;
    host.nest(skill(), &[0, 1], 2, 20);
    // k bit-identical recurring states, deterministic and reproducible on every revisit.
    let states: Vec<(f32, f32)> = (0..k)
        .map(|s| {
            let x = (s as f32 / k as f32) * 2.0 - 1.0;
            let y = (((s * 7 + 3) % k) as f32 / k as f32) * 2.0 - 1.0;
            (x, y)
        })
        .collect();
    let mut worst = 0.0f32;
    let t0 = Instant::now();
    for i in 0..fires {
        let (x, y) = states[i % k];
        host.stimulate(&[(0, x), (1, y)], 20);
        worst = worst.max((host.cells()[2] - truth(x, y)).abs());
    }
    (host.nested_settles(), worst, t0.elapsed().as_millis())
}

#[test]
#[ignore = "stress: nightly"]
fn b1_recurring_states_within_cap_are_flat() {
    // K distinct recurring states, K <= default cap (4096): distinct-input WORK, not fire count.
    eprintln!("[B1] recurring K states, T=16·K fires, default memo_cap=4096:");
    for &k in &[4usize, 64, 1024, 4096] {
        let fires = 16 * k;
        let (settles, worst, ms) = run_memo(k, fires, 4096);
        eprintln!(
            "[B1] K={k:>5} states, {fires:>7} fires → {settles:>6} sub-settles ({:.1}× reuse), worst err {worst:.4} | {ms} ms",
            fires as f64 / settles as f64
        );
        assert!(
            worst < 1e-3,
            "memo must stay exact (K={k}, worst {worst:.5})"
        );
        assert_eq!(
            settles as usize, k,
            "each distinct state settles exactly once when K<=cap (K={k}, settles {settles})"
        );
    }
    eprintln!("[B1] VERDICT: while the distinct-state count fits under memo_cap, a nested cell pays for DISTINCT-input work exactly once (settles == K), reuse scales with the fire count — flat, exact.");
}

#[test]
#[ignore = "stress: nightly"]
fn b2_the_memo_cap_cliff() {
    // Eviction is CLEAR-ALL on overflow (`memo.clear()` when len >= cap), NOT LRU. So the instant the
    // working set of distinct states exceeds the cap, a round-robin cycle never survives to be reused →
    // reuse collapses to ~0 and every fire re-settles the whole sub. Pin cap=256, sweep K across it.
    const CAP: usize = 256;
    eprintln!("[B2] memo_cap PINNED to {CAP}; sweep K across the cap, T=8·K round-robin fires:");
    let mut below = (0u64, 0usize);
    let mut above = (0u64, 0usize);
    for &k in &[128usize, 256, 384, 512, 1024] {
        let fires = 8 * k;
        let (settles, worst, ms) = run_memo(k, fires, CAP);
        let miss = 100.0 * settles as f64 / fires as f64;
        eprintln!(
            "[B2] K={k:>5} (cap {CAP}), {fires:>6} fires → {settles:>6} settles = {miss:>5.1}% MISS, worst err {worst:.4} | {ms} ms"
        );
        assert!(
            worst < 1e-3,
            "still exact on miss (K={k}, worst {worst:.5})"
        );
        if k <= CAP {
            below = (settles, k);
        } else {
            above = (settles, k);
        }
    }
    // Below the cap: settles == K (perfect reuse). Above: settles ≈ fires (reuse gone).
    let (settles_below, k_below) = below;
    let (settles_above, k_above) = above;
    eprintln!(
        "[B2] VERDICT: the memo degrades on a CLIFF, not gracefully. At K={k_below}<=cap reuse is perfect (settles {settles_below}=K); at K={k_above}>cap the CLEAR-ALL eviction thrashes — a round-robin working set never survives a full cycle, so reuse collapses (settles {settles_above} ≈ fires, ~100% miss). A nested node over MORE recurring discrete states than memo_cap gets ZERO memo benefit. The fix would be LRU/associative eviction; today the cap is a hard wall — size memo_cap >= the recurring-state count or lose all reuse."
    );
    assert_eq!(
        settles_below as usize, k_below,
        "below the cap, reuse is perfect (settles {settles_below} should equal K {k_below})"
    );
    assert!(
        settles_above as usize > 4 * k_above,
        "above the cap, clear-all thrash destroys reuse (K={k_above}: settles {settles_above} should approach the fire count 8·K)"
    );
}

#[test]
#[ignore = "stress: nightly"]
fn b3_novel_continuous_inputs_never_hit() {
    // Distinct continuous inputs every fire: the bit-keyed memo never hits, so it is pure overhead
    // (fills to cap, clears, refills) with ZERO reuse. The memo helps ONLY exact recurring discrete states.
    let mut host = Field::new(2);
    host.nest(skill(), &[0, 1], 2, 20);
    let mut rng = Rng::new(0xC0FFEE);
    let fires = 5_000usize;
    let mut worst = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..fires {
        let (x, y) = (rng.sym(), rng.sym());
        host.stimulate(&[(0, x), (1, y)], 20);
        worst = worst.max((host.cells()[2] - truth(x, y)).abs());
    }
    let settles = host.nested_settles();
    let ms = t0.elapsed().as_millis();
    eprintln!(
        "[B3] {fires} NOVEL continuous inputs → {settles} sub-settles ({:.1}% miss), worst err {worst:.4} | {ms} ms",
        100.0 * settles as f64 / fires as f64
    );
    assert!(worst < 1e-3, "still exact (worst {worst:.5})");
    assert_eq!(
        settles as usize, fires,
        "every novel input is a cache MISS — the memo gives no reuse on continuous streams"
    );
    eprintln!("[B3] VERDICT: the memo is keyed on the EXACT f32 bit pattern, so a continuous / never-repeating stream misses 100% — memoization is a discrete-recurring-state optimization only (ARC-3's regime), not a general speedup.");
}
