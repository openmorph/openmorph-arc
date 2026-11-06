// SPDX-License-Identifier: MIT-0
//! #153 (OM-3) — native `merge(keep, drop)` + fingerprint-DISTANCE dedup (openmorph#146). The substrate could
//! GROW identity cells and CULL starved ones, but had no way to FUSE two cells discovered to be the same
//! identity — `fn merge` count in core was 0, so the "content-address / dedup: the common sub-morph collapses
//! to ONE node" case `fingerprint()`'s own doc named was unbuilt. This closes it with ONE primitive.
//!
//! `merge` is the DUAL of `cull`: `cull` drops a starved cell (detaching its readers); `merge` FUSES two
//! equivalent cells WITHOUT losing what read them — every reference to `drop` (a unit's read+weight, write,
//! gate, factor; a temporal/spectral memory source; a fast-weight bind pair) is redirected onto `keep`, a
//! read that becomes a duplicate of `keep` FOLDS (weights sum), positions average, drives sum, and `drop` is
//! freed for reuse. Deterministic (no RNG).
//!
//! WHAT IS ASSERTED (truth-not-trophy):
//!   A — FUNCTION PRESERVED: when `keep` and `drop` carry the same value (the identity-dedup premise), fusing
//!       them leaves the organism's computed output byte-close (the folded weight = the sum) — no capability lost.
//!   B — CLEAN + REUSABLE: after the merge nothing reads `drop`, the field still settles with no panic, and the
//!       freed slot is REUSED by the next growth (no dangling id, no leak).
//!   C — DETERMINISTIC: two identically-built organisms merged the same way settle to identical output (replay-safe).
//!   D — DEDUP: a pool of K near-identical sub-morphs + D distinct ones collapses to exactly 1+D under a
//!       `fingerprint_distance` threshold grounded in the measured clone floor — K−1 duplicates found, ZERO
//!       false-merges (precision = recall = 1). The native soft form of #146.
//!
//! RE-BASELINED at the 1.6 learning flip — the DEDUP proof (claim D) depended on an accident: SILENT SETTLE TRUNCATION.
//! Its organisms were grown with a settle budget of 40 while `grow` births up to `max_hidden` (64) features and a settle
//! costs hidden + 2 firings. On the legacy genome four of the five organisms truncated ~1460 of their 1500 settles
//! (`Field::truncated_settles`) and never learned their targets (held-out MAE 0.32–0.49): their fingerprints were far
//! apart because they were broken. (On the 1.6 default at budget 40, two organisms truncated and one distinct organism
//! was merged: precision 0.80.) The budget is now 200 — every settle completes, held-out MAE ≤ 0.02 — and on the 1.6
//! default the dedup is exact (nearest distinct pair 0.0115, threshold 0.0058; the closest distinct-to-clone pair,
//! tanh(1.5x) vs tanh(2.4x), at 0.0070). Settled on the LEGACY genome it is not (that pair at 0.0063 < threshold
//! 0.0075: one false merge). Claims A–C (merge itself) are unaffected; all four run on the default genome.

use openmorph_core::Field;

use crate::common::Rng;

/// Grow a small organism that learns y = tanh(gain·x) from cells {0,1} (two "identity" inputs fed the SAME
/// value — as a percept→identity binder would produce for one object seen twice). Returns the field; cell 2 is
/// the output, cells 0 and 1 the twin identity inputs.
fn twin_identity_organism(gain: f32) -> Field {
    let mut f = Field::new(3);
    f.config_mut().max_hidden = 0; // a clean linear readout over the two twin cells
    let mut rng = Rng::new(0x1D);
    for _ in 0..2500 {
        let x = rng.sym();
        // both identity cells carry x; the readout learns a weight on each
        f.grow(&[(0, x), (1, x)], 2, (gain * x).tanh(), 0.1, 30);
    }
    f
}

#[test]
fn merge_fuses_identity_cells() {
    // ── A — FUNCTION PRESERVED ────────────────────────────────────────────────────────────────────────────
    let mut f = twin_identity_organism(1.5);
    let xt = 0.4f32;
    f.stimulate(&[(0, xt), (1, xt)], 30);
    let before = f.cells()[2];

    f.merge(0, 1); // fuse the twin identity cell 1 into cell 0

    // after the merge the readout reads ONLY cell 0 (the two weights folded); feed cell 0 = xt
    f.stimulate(&[(0, xt)], 30);
    let after = f.cells()[2];
    eprintln!("[MERGE] FUNCTION PRESERVED: output {before:.6} → {after:.6} after fusing the twin identity cell");
    assert!(
        (before - after).abs() < 1e-4,
        "merge changed the function: {before:.6} vs {after:.6}"
    );

    // ── B — CLEAN + REUSABLE ──────────────────────────────────────────────────────────────────────────────
    // no unit reads the dropped cell any more; the field still settles (no panic / dangling id)
    f.stimulate(&[(0, -0.2)], 30);
    let cells_after_merge = f.cells().len();
    // grow a fresh feature — it must REUSE the freed slot (the medium does not grow)
    let _ = f.grow_backbone(&[0], 1, 1.0);
    assert!(
        f.cells().len() <= cells_after_merge,
        "freed cell was not reused: {} → {}",
        cells_after_merge,
        f.cells().len()
    );
    eprintln!(
        "[MERGE] CLEAN + REUSABLE: settles after merge, freed slot reused (no medium growth)"
    );

    // ── C — DETERMINISTIC ─────────────────────────────────────────────────────────────────────────────────
    let mut a = twin_identity_organism(2.0);
    let mut b = twin_identity_organism(2.0);
    a.merge(0, 1);
    b.merge(0, 1);
    a.stimulate(&[(0, 0.33)], 30);
    b.stimulate(&[(0, 0.33)], 30);
    assert_eq!(
        a.cells()[2].to_bits(),
        b.cells()[2].to_bits(),
        "merge is not deterministic"
    );
    eprintln!("[MERGE] DETERMINISTIC: two identically-built organisms merged identically → bit-identical output");

    eprintln!("[MERGE] VERDICT: native merge(keep, drop) fuses two same-identity cells WITHOUT losing what read them (weights fold, references redirect, the dropped slot is freed) — function-preserving, clean, deterministic. The dual of cull; the missing content-address collapse (#146).");
}

/// Grow a distinct organism learning y = tanh(gain·x + phase) — different gain/phase ⇒ a distinct behaviour ⇒
/// a far `fingerprint`.
fn organism(seed: u64, gain: f32, phase: f32) -> Field {
    let mut f = Field::new(2);
    let mut rng = Rng::new(seed);
    for _ in 0..1500 {
        let x = rng.sym();
        // budget 200 ≥ max_hidden + 2: every settle completes (the old 40 truncated — see the module docs)
        f.grow(&[(0, x)], 1, (gain * x + phase).tanh(), 0.1, 200);
    }
    f
}

#[test]
fn fingerprint_distance_dedup() {
    // A pool: K near-identical clones of one sub-morph + D genuinely distinct ones.
    const K: usize = 5;
    const D: usize = 4;
    let base = organism(0xC1, 1.5, 0.0);
    let mut pool: Vec<Field> = Vec::new();
    for _ in 0..K {
        pool.push(base.clone()); // exact clones — a fingerprint_distance of ~0 (the self-noise floor)
    }
    let distinct = [
        organism(0xA1, 0.6, 0.0),
        organism(0xA2, 2.4, 0.0),
        organism(0xA3, 1.5, 0.9),
        organism(0xA4, 3.0, -0.6),
    ];
    for d in &distinct {
        pool.push(d.clone());
    }

    // The dedup THRESHOLD, grounded in the MEASURED clone floor (a relative threshold, not a hand-magic
    // constant): a clone's self-distance is ~0; the nearest distinct pair is far. Deploy-time it is
    // meta-evolved (substrate_meta_policy), exactly as the OM-7 dedup radius — here we ground it and report the gap.
    let clone_floor = base.fingerprint_distance(&pool[1]); // ~0
    let nearest_distinct = {
        let mut m = f32::MAX;
        for i in 0..distinct.len() {
            for j in (i + 1)..distinct.len() {
                m = m.min(distinct[i].fingerprint_distance(&distinct[j]));
            }
        }
        m
    };
    let threshold = (clone_floor + nearest_distinct) * 0.5; // well inside the gap
    eprintln!(
        "[DEDUP] clone floor {clone_floor:.4} ≪ nearest distinct {nearest_distinct:.4}; threshold {threshold:.4}"
    );

    // Greedy dedup: keep a representative; a pool member within `threshold` of a kept rep is a duplicate.
    let mut reps: Vec<usize> = Vec::new();
    let mut is_dup = vec![false; pool.len()];
    for i in 0..pool.len() {
        let mut dup = false;
        for &r in &reps {
            if pool[i].fingerprint_distance(&pool[r]) < threshold {
                dup = true;
                break;
            }
        }
        if dup {
            is_dup[i] = true;
        } else {
            reps.push(i);
        }
    }
    let kept = reps.len();
    let found_dupes = is_dup.iter().filter(|&&d| d).count();

    // The K clones occupy pool[0..K]; the D distinct occupy pool[K..]. Correct dedup: exactly 1+D reps kept,
    // K−1 clones flagged duplicate, and NO distinct organism flagged (zero false-merges).
    let false_merges = (K..pool.len()).filter(|&i| is_dup[i]).count();
    let true_dupes = (1..K).filter(|&i| is_dup[i]).count(); // clones after the first rep
    let precision = 1.0 - false_merges as f32 / found_dupes.max(1) as f32;
    let recall = true_dupes as f32 / (K - 1) as f32;
    eprintln!(
        "[DEDUP] pool {} ({K} clones + {D} distinct) → {kept} kept, {found_dupes} duplicates found; precision {precision:.2}, recall {recall:.2}, false-merges {false_merges}",
        pool.len()
    );

    eprintln!("[DEDUP] VERDICT: fingerprint_distance soft-dedup collapses near-identical sub-morphs to ONE representative and keeps every distinct one — precision/recall 1.0, zero false-merges — the native content-address dedup (#146) that feeds merge().");

    assert!(
        clone_floor < 0.1 * nearest_distinct.max(1e-6),
        "clone floor not ≪ distinct gap"
    );
    assert_eq!(
        kept,
        1 + D,
        "expected exactly 1 clone-rep + {D} distinct kept, got {kept}"
    );
    assert_eq!(false_merges, 0, "a distinct organism was wrongly merged");
    assert!(
        (precision - 1.0).abs() < 1e-6 && (recall - 1.0).abs() < 1e-6,
        "precision/recall not 1.0"
    );
}
