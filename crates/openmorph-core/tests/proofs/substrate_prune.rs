// SPDX-License-Identifier: MIT-0
//! CULL AS A CLEAN CAPACITY REGULARIZER (#128). #128 asks to kill the fixed authored backbone: grow the
//! capacity a task needs and CULL/regularize the rest, so the morph can't memorize with excess width. The
//! economy already starves a low-contribution feature (`metabolize_contribution` pays income ∝ `credit_of`, all
//! pay upkeep, the starved `cull`). But culling a feature used to leave its CELL behind with a stale value that
//! the readout kept reading — so pruning a feature CORRUPTED the readout instead of regularizing it.
//!
//! The fix (in `cull`/`cull_after`): a cell ORPHANED by a cull (written by a culled unit, by no survivor) is
//! DETACHED from every reader (dropped from their reads/weights) and cleared. So a starved feature can be
//! pruned without dragging a dead input into the answer — cull is now a clean regularizer.
//!
//! This proves it directly: a readout over ONE useful feature + several DEAD (noise) features learns the task;
//! the dead features are pruned; and the readout's prediction stays CORRECT afterwards (it would be corrupted
//! by stale orphaned cells without the detach), with capacity actually removed.
//!
//! TRUTH-NOT-TROPHY (an honest finding this surfaced): the SELECTION signal for "what to prune" must be
//! RELIANCE — how much the readout leans on a feature (`reliance_of`) — NOT the economy's `credit_of`. At
//! convergence `credit_of` (residual-correlation) VANISHES for a USED feature (its gradient → 0), so both the
//! useful and the dead features read ~0 credit and a credit-metered economy culls EVERYTHING. So the STRUCTURAL
//! economy `metabolize_reliance` meters income by reliance instead: the dead features starve and cull, the
//! useful one survives — no `consolidate`, no picking. The two-fitness split stands: residual-credit
//! (`metabolize_contribution`) drives EVOLUTION, reliance (`metabolize_reliance`) drives structural KEEP/CULL.
//!
//!   cargo test -p openmorph-core --test proofs substrate_prune -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const SIG: usize = 0; // the signal input (target depends only on this)
const NOISE: usize = 4; // cells 1..=NOISE are pure noise (dead-feature inputs)

fn mae(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

#[test]
fn cull_prunes_dead_capacity_without_corrupting_the_readout() {
    let n_in = 1 + NOISE;
    let mut f = Field::new(n_in);
    let mut rng = Rng::new(0x9D1);

    // One USEFUL feature (reads the signal) + several DEAD features (each reads a noise cell). No fixed
    // backbone — the point is that EXCESS capacity is prunable.
    let useful = f.len();
    f.ensure_cells(useful + 1);
    f.add_unit(vec![SIG], vec![1.2], 0.0, useful);
    let mut dead = Vec::new();
    for d in 0..NOISE {
        let c = f.len();
        f.ensure_cells(c + 1);
        f.add_unit(vec![1 + d], vec![rng.sym()], rng.sym() * 0.3, c); // reads noise input 1+d
        dead.push(c);
    }
    let feats: Vec<usize> = std::iter::once(useful)
        .chain(dead.iter().copied())
        .collect();
    let out = f.len();
    f.ensure_cells(out + 1);
    f.add_unit(feats.clone(), vec![0.0; feats.len()], 0.0, out);

    let target = |x0: f32| (1.5 * x0).tanh();
    // Every cell gets an independent random value; SIG carries the signal (the target reads it), the rest
    // are noise the readout must learn to ignore.
    let stim = |rng: &mut Rng| -> Vec<(usize, f32)> { (0..n_in).map(|c| (c, rng.sym())).collect() };

    // Train the readout: it must learn to rely on the useful feature and ignore the noise ones.
    for _ in 0..1500 {
        let inp = stim(&mut rng);
        let x0 = inp[SIG].1;
        f.learn_step(&inp, out, target(x0), 0.1, 64);
    }

    // Held-out MAE BEFORE pruning.
    let mut rt = Rng::new(0x7E57);
    let before_mae = {
        let (mut t, mut p) = (Vec::new(), Vec::new());
        for _ in 0..300 {
            let inp = stim(&mut rt);
            let x0 = inp[SIG].1;
            f.stimulate(&inp, 64);
            t.push(target(x0));
            p.push(f.cells()[out]);
        }
        mae(&t, &p)
    };
    let units_before = f.n_units();
    let useful_reliance = f.reliance_of(useful);
    let dead_reliance: f32 = dead.iter().map(|&c| f.reliance_of(c)).sum::<f32>() / NOISE as f32;

    // ── The RELIANCE economy prunes the dead capacity: income ∝ how much the readout leans on a feature
    //    (`metabolize_reliance`), all pay upkeep; a feature the answer ignores earns ~0 → starves → cull. No
    //    `consolidate`, no picking — reliance itself distinguishes the useful feature from the dead ones (where
    //    `credit_of` cannot, since it vanishes at convergence). ──
    for _ in 0..40 {
        f.metabolize_reliance(&feats, 2.0, 1.0);
        f.cull(0.5);
    }
    let units_after = f.n_units();

    // Held-out MAE AFTER pruning — must stay correct (orphaned dead cells were detached, not left stale).
    let mut rt2 = Rng::new(0x7E57);
    let after_mae = {
        let (mut t, mut p) = (Vec::new(), Vec::new());
        for _ in 0..300 {
            let inp = stim(&mut rt2);
            let x0 = inp[SIG].1;
            f.stimulate(&inp, 64);
            t.push(target(x0));
            p.push(f.cells()[out]);
        }
        mae(&t, &p)
    };

    // The surviving readout must no longer read any pruned (dead) cell — the detach worked.
    let ro2 = f.units.iter().find(|u| u.write == out).unwrap();
    let still_reads_dead = dead.iter().any(|c| ro2.reads.contains(c));

    eprintln!("[PRUNE] reliance (readout |weight|) — useful {useful_reliance:.3} vs dead {dead_reliance:.3}; units {units_before} → {units_after}; readout still reads a dead cell: {still_reads_dead}");
    eprintln!("[PRUNE] held-out MAE — before {before_mae:.4}   after {after_mae:.4}  (correct across the prune)");
    eprintln!("[PRUNE] VERDICT: cull is now a CLEAN capacity regularizer (#128), driven by the RELIANCE economy. `metabolize_reliance` pays a feature income ∝ how much its readers lean on it (convergence-stable, unlike `credit_of` which vanishes at convergence), so the dead features starve and `cull` removes them; and because an orphaned cell is DETACHED from its readers (not left with a stale value the readout keeps reading), the pruned readout stays CORRECT — held-out MAE preserved, capacity actually removed, no corruption, no `consolidate`/picking. This is the missing 'cull/regularize the rest' half of #128: growth (grow_deep_step) births what a task demands and reliance-metered cull removes the excess, so an over-provisioned basis SELF-SIZES and `max_hidden` stops being a fixed-capacity dial — the morph prunes its own width without memorizing. The two-fitness split stands: residual-`credit` (`metabolize_contribution`) breeds heritable traits (EVOLUTION), `reliance` (`metabolize_reliance`) keeps/culls STRUCTURE.");

    assert!(
        units_after < units_before,
        "the economy must actually prune dead capacity (units {units_before} → {units_after})"
    );
    assert!(
        !still_reads_dead,
        "the pruned readout must not read any culled dead cell (the orphaned-cell detach)"
    );
    assert!(
        after_mae < 0.1 && after_mae < before_mae + 0.03,
        "the pruned readout must stay correct — no stale-cell corruption (before {before_mae:.4}, after {after_mae:.4})"
    );
    assert!(
        useful_reliance > dead_reliance + 0.2,
        "the readout must rely on the useful feature far more than the dead ones (useful {useful_reliance:.3} vs dead {dead_reliance:.3})"
    );
}
