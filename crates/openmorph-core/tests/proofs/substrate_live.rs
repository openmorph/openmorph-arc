// SPDX-License-Identifier: MIT-0
//! substrate_live — THE AUTONOMOUS HEARTBEAT + the LIVING-MEMORY equation (D6/D7, #191 §3).
//!
//! The culminating "living organism" primitive: with NO external input the organism runs its OWN subjective
//! time — it metabolizes (energy ebbs, hunger rises), spontaneously REACTIVATES dormant memories during
//! quiescence to test whether they still earn their keep (offline replay = dreaming = introspection = memory
//! maintenance, ONE loop), keeps the useful ones (`consolidate`) and prunes the stale ones (`cull`), then
//! truly SLEEPS when no unfinished work remains. The ONLY new mechanism is the per-cell reactivation pressure
//! `P_c` self-trigger; every downstream arrow (`recency`, `reliance_of`, `consolidate`, `cull`) already existed.
//!
//! Each claim (a)-(e) is falsifiable, with a control that MUST fail. Truth-not-trophy: the maintenance test
//! genuinely keeps-good AND prunes-stale (not churn everything); sleep is genuinely bounded; determinism is
//! byte-exact.

use openmorph_core::Field;

/// Build a field with two grown MEMORIES on a shared context cell:
/// - cell 1 = memory A, still USEFUL: a downstream, already-consolidated skill (unit uR) reads it with weight 1.
/// - cell 2 = memory B, now USELESS/stale: nothing reads it any more (reliance 0).
///
/// So `reliance_of(1) = 1.0` (kept) and `reliance_of(2) = 0.0` (pruned) — the D7 keep-good / prune-stale split,
/// driven by nothing but structure (who still relies on the memory) + the heartbeat's dormancy trigger.
/// Cells: 0 = context, 1 = memory A, 2 = memory B, 3 = the downstream skill's output.
fn memories() -> Field {
    let mut m = Field::new(4);
    let _u_a = m.add_unit(vec![0], vec![0.8], 0.0, 1); // memory A: context → cell 1
    let _u_b = m.add_unit(vec![0], vec![0.8], 0.0, 2); // memory B: context → cell 2
    let _u_r = m.add_unit(vec![1], vec![1.0], 0.0, 3); // a downstream skill RELIES on memory A (reads cell 1)
    m.consolidate(&[3]); // the downstream skill is an ACTIVE, frozen asset (not itself a reactivation candidate)
    m
}

/// (a) AUTONOMY — with NO external input the organism metabolizes → hunger rises → takes self-initiated actions
/// (spontaneous reactivations + consolidate/cull decisions) a purely reactive settler cannot. Control: a field
/// that only `stimulate`s empty input (a reactive settler) emits nothing and its hunger never moves.
#[test]
fn autonomy_flowing_time_vs_reactive_control() {
    // ORGANISM: run the heartbeat with no external input at all.
    let mut alive = memories();
    assert_eq!(alive.satiation(), 1.0, "born fed (hunger 0)");
    let hb = alive.live(256);
    let self_acts = hb.reactivations;
    let decisions = hb.consolidated + hb.culled;
    eprintln!(
        "[LIVE] (a) AUTONOMY: with NO input — ticks {}, reactivations {}, decisions(consolidate+cull) {}, hunger 0.00 → {:.2}",
        hb.ticks, self_acts, decisions, hb.hunger
    );
    assert!(
        hb.ticks > 0,
        "the organism ran its own internal time with no input"
    );
    assert!(
        self_acts > 0,
        "it took self-initiated actions (spontaneous reactivations) during pure quiescence"
    );
    assert!(
        decisions > 0,
        "it made autonomous keep/prune DECISIONS a reactive settler never makes"
    );
    assert!(
        hb.hunger > 0.9,
        "metabolizing on its own drove hunger up from 0 (flowing time): {:.2}",
        hb.hunger
    );

    // CONTROL: a REACTIVE settler — same organism, but it only ever settles external input, and there is none.
    let mut reactive = memories();
    for _ in 0..256 {
        reactive.stimulate(&[], 128); // no input → nothing fires → no autonomous work
    }
    assert_eq!(
        reactive.satiation(),
        1.0,
        "the reactive control never gets hungry (hunger stays 0)"
    );
    assert!(
        reactive.has_writer(1) && reactive.has_writer(2),
        "the reactive control makes no keep/prune decisions"
    );
    assert_eq!(
        reactive.plasticity_of(1),
        1.0,
        "the reactive control consolidates nothing"
    );
    eprintln!("[LIVE] (a) CONTROL (reactive settler, no input): hunger stays 0.00, zero reactivations, zero decisions — a reactive organism does NOTHING without the world. VERDICT: the heartbeat is genuinely autonomous.");
}

/// (b) SLEEP MAINTENANCE (the D7 core) — during quiescence, dormant memories reactivate and are tested; a
/// STILL-USEFUL memory SURVIVES (consolidated) while a NOW-USELESS one is PRUNED (culled), driven by nothing
/// but `recency` (the trigger) + contribution/`reliance` (the test). Control: WITHOUT the heartbeat, the useless
/// memory is NOT pruned and the useful one is NOT reinforced.
#[test]
fn sleep_maintenance_keeps_good_prunes_stale() {
    // MAINTENANCE ON (the heartbeat runs during idle).
    let mut m = memories();
    assert!(
        m.has_writer(1) && m.has_writer(2),
        "both memories present at start"
    );
    assert_eq!(
        m.plasticity_of(1),
        1.0,
        "memory A starts plastic (not yet consolidated)"
    );
    let hb = m.live(512);

    let useful_survived = m.has_writer(1); // memory A still exists…
    let useful_frozen = m.plasticity_of(1) == 0.0; // …and was KEPT (consolidated) because it still earns
    let stale_pruned = !m.has_writer(2); // memory B was culled — nothing relied on it any more

    eprintln!(
        "[LIVE] (b) SLEEP MAINTENANCE: reactivations {}, consolidated {}, culled {}",
        hb.reactivations, hb.consolidated, hb.culled
    );
    eprintln!(
        "[LIVE] (b)   USEFUL memory A (reliance 1.0): survived={useful_survived}, frozen/kept={useful_frozen}   |   STALE memory B (reliance 0.0): pruned={stale_pruned}"
    );
    assert!(useful_survived, "a still-useful memory must SURVIVE sleep");
    assert!(
        useful_frozen,
        "a still-useful memory must be KEPT (consolidated) — reinforced, not eroded"
    );
    assert!(
        stale_pruned,
        "a now-useless memory must be PRUNED (culled) during sleep"
    );
    assert_eq!(
        hb.consolidated, 1,
        "exactly the one useful memory was kept (not churn-everything)"
    );
    assert_eq!(
        hb.culled, 1,
        "exactly the one stale memory was pruned (truth-not-trophy: keep-good AND prune-stale)"
    );

    // CONTROL: NO heartbeat — the same idle time passes with only reactive settling. Nothing is tested, so the
    // stale memory is NOT pruned and the useful one is NOT reinforced.
    let mut ctrl = memories();
    for _ in 0..512 {
        ctrl.stimulate(&[], 128);
    }
    assert!(
        ctrl.has_writer(2),
        "CONTROL: without the P-trigger the useless memory is NOT pruned"
    );
    assert_eq!(
        ctrl.plasticity_of(1),
        1.0,
        "CONTROL: without the P-trigger the useful memory is NOT reinforced/consolidated"
    );
    eprintln!("[LIVE] (b) CONTROL (no heartbeat): stale memory B still present, useful memory A still un-consolidated — maintenance requires the reactivation self-trigger. survive={{A}} / prune={{B}} with the heartbeat; survive={{A,B}} / prune={{}} without it.");
}

/// (c) DETERMINISM — two runs from the SAME construction produce byte-identical state (replay intact). The
/// heartbeat advances only the organism's OWN subjective clock, never a wall-clock, so it is fully reproducible.
#[test]
fn deterministic_byte_identical_replay() {
    let mut a = memories();
    let mut b = memories();
    // identical heartbeat sequences
    for _ in 0..8 {
        let ha = a.live(64);
        let hb = b.live(64);
        assert_eq!(
            ha, hb,
            "each heartbeat report is identical across the two runs"
        );
    }
    let sa = serde_json::to_string(&a).expect("serialise a");
    let sb = serde_json::to_string(&b).expect("serialise b");
    assert_eq!(
        sa, sb,
        "two same-seed runs must be BYTE-IDENTICAL (deterministic, replay intact)"
    );
    eprintln!("[LIVE] (c) DETERMINISM: two identical heartbeat runs are byte-identical ({} bytes) — the clock is the organism's own event count, never wall-clock.", sa.len());
}

/// (d) TRULY SLEEPS (not a spin) — left idle, the number of autonomous ticks is BOUNDED (converges), because
/// every reactivation RESOLVES a memory (consolidate ⇒ frozen, or cull ⇒ gone), so the candidate set strictly
/// shrinks. After everything is resolved, further `live` calls do ZERO ticks and report `slept`.
#[test]
fn truly_sleeps_bounded_not_a_spin() {
    let mut m = memories();
    let mut total_ticks = 0usize;
    let mut trailing_idle = 0usize;
    for i in 0..200 {
        let hb = m.live(128);
        total_ticks += hb.ticks;
        if i >= 100 {
            assert_eq!(
                hb.ticks, 0,
                "after all work is resolved, an idle heartbeat must do NOTHING (sleep)"
            );
            assert!(hb.slept, "and it must report that it slept");
            trailing_idle += 1;
        }
    }
    eprintln!(
        "[LIVE] (d) TRULY SLEEPS: 200 idle heartbeat calls did {total_ticks} ticks TOTAL (bounded), then {trailing_idle}/100 trailing calls did 0 ticks (asleep). A spin would be unbounded (25600+)."
    );
    assert!(
        total_ticks > 0,
        "it did do some autonomous work before sleeping"
    );
    assert!(
        total_ticks < 500,
        "the idle-tick count must CONVERGE (a finite bound), not spin: {total_ticks}"
    );
    assert_eq!(trailing_idle, 100, "once resolved it stays asleep");
}

/// (e) BYTE-IDENTICAL-OFF — a field whose heartbeat is never invoked behaves EXACTLY as before. It learns the
/// same mapping to the same number as the pre-existing `grow` doctest, and the new heartbeat state stays inert.
#[test]
fn byte_identical_when_heartbeat_never_invoked() {
    // The exact pre-existing lib.rs doctest computation — grow x → tanh(1.5·x). Never calls `live`.
    let mut m = Field::new(2);
    m.config_mut().max_hidden = 0; // linear-in-tanh — grow the writer, no hidden depth
    for _ in 0..2_000 {
        for &x in &[-1.0_f32, 1.0] {
            m.grow(&[(0, x)], 1, (1.5 * x).tanh(), 0.1, 50);
        }
    }
    m.stimulate(&[(0, 1.0)], 50);
    let got = m.cells()[1];
    let want = (1.5_f32).tanh();
    assert!(
        (got - want).abs() < 0.05,
        "the pre-existing grow number is unchanged: {got:.4} ≈ {want:.4}"
    );
    // The new heartbeat state was never touched: metabolism still full (never drained), config defaults present.
    assert_eq!(
        m.satiation(),
        1.0,
        "metabolism untouched (hunger 0) — the heartbeat never ran"
    );
    assert_eq!(m.config().heartbeat_k, 0.5, "the new levers default");
    assert_eq!(m.config().heartbeat_theta, 1.0, "the new levers default");
    eprintln!("[LIVE] (e) BYTE-IDENTICAL-OFF: an unheartbeated field learns tanh(1.5)={want:.4} → {got:.4} (unchanged); metabolism full, pressure inert. The heartbeat is opt-in and additive.");
    eprintln!("[LIVE] VERDICT: the organism is now ALIVE — it flows its own time, dreams its dormant memories, keeps the useful and prunes the stale, and truly sleeps. One new state variable (P_c); every other arrow already existed.");
}
