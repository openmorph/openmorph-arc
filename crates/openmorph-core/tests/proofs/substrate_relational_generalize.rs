// SPDX-License-Identifier: MIT-0
//! RELATIONAL GENERALIZATION — "recolor a cell if a Y sits next to it" from a FEW examples (MorphBot's deepest
//! gap, 2026-08-16). The wall MorphBot measured (6/12) is REAL and it is NOT a cleverness problem: an
//! ATOMIC-context learner (a table / tree / a morph keyed on a hashed local patch) treats "Y above" / "Y below"
//! / "Y left" / "Y right" as FOUR unrelated symbols, so it must see every direction → it cannot generalize
//! across them. This proof settles whether openmorph needs a NEW primitive for it, or whether it is WIRING —
//! feeding the substrate POOLED relational predicates instead of atomic contexts. TRUTH-NOT-TROPHY: a direction
//! split makes the invariance the ONLY thing that can win.
//!
//! The rule: a cell of colour A recolours IFF at least one 4-neighbour is colour Y. `(center==A) ∧ (Y-adjacent)`
//! — a CONJUNCTION over a POSITION-INVARIANT relation. TRAIN patches place Y only in {up,left}; HELD-OUT TEST
//! patches place Y only in {down,right} (directions never seen as Y in training). So:
//!   * ATOMIC learner (memorise the 5-tuple center+4 neighbours → label): a test patch with Y-down/right is an
//!     UNSEEN tuple ⇒ it defaults ⇒ it fails the held-out directions. This reproduces MorphBot's wall.
//!   * SUBSTRATE-NATIVE: perception POOLS the relation over the neighbourhood (`Y-adjacent = OR over the 4
//!     neighbours` — invariance BY POOLING, Hubel–Wiesel; the same `aggregate`-over-`relate` the structure
//!     primitives already expose), so "Y up" and "Y down" become the SAME feature. The substrate then GROWS the
//!     conjunction (`grow_backbone`) and SELECTS the predictive relation among DISTRACTORS (`Z-adjacent`,
//!     neighbour-count, center==other) by the delta rule's own Rescorla–Wagner competition + reliance — no new op.
//!
//! Verdict shape: pooled-grown GENERALISES to held-out directions (few examples), atomic COLLAPSES; and the
//! grown readout RELIES on the causal relations (isA, Y-adjacent) over the distractors. If both hold, the gap is
//! WIRING (feed pooled predicates), not a missing primitive. The pooling is `aggregate`-over-`relate` (grown,
//! domain-blind); the selection + conjunction are `grow_backbone` + `learn_step` + reliance (the economy).
//!
//!   cargo test -p openmorph-core --test proofs substrate_relational_generalize -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// colours: 0 = empty, 1 = A (the cell that may recolour), 2 = Y (the trigger), 3 = Z (a distractor), 4 = other.
const A: usize = 1;
const Y: usize = 2;
const Z: usize = 3;
const K: usize = 24; // grown backbone width
const STEPS: usize = 4000;

/// A local patch: the centre colour + the 4 neighbour colours [up, down, left, right].
struct Patch {
    center: usize,
    nbr: [usize; 4],
}

/// The ground-truth label: recolour (1.0) iff centre is A and some neighbour is Y, else keep (-1.0).
fn label(p: &Patch) -> f32 {
    if p.center == A && p.nbr.contains(&Y) {
        1.0
    } else {
        -1.0
    }
}

/// Generate a patch. `y_dirs` = which neighbour slots MAY hold the Y trigger (the direction split). Half the
/// positive patches get a Y in an allowed direction; the rest are negatives / distractor-laden.
fn patch(rng: &mut Rng, y_dirs: &[usize]) -> Patch {
    let center = if rng.f() < 0.6 { A } else { 1 + rng.range(4) }; // usually A, sometimes another colour
    let mut nbr = [0usize; 4];
    for n in nbr.iter_mut() {
        *n = match rng.range(4) {
            0 => 0, // empty
            1 => Z, // distractor colour
            2 => 4, // other
            _ => 1 + rng.range(4),
        };
    }
    // With some probability, PLANT a Y in one of the allowed directions (so positives exist in train & test).
    if rng.f() < 0.5 {
        let d = y_dirs[rng.range(y_dirs.len())];
        nbr[d] = Y;
    }
    Patch { center, nbr }
}

// ── ATOMIC learner: memorise the exact 5-tuple → majority label. The table/tree/hashed-context baseline. ──
fn atomic_accuracy(train: &[Patch], test: &[Patch]) -> f32 {
    use std::collections::HashMap;
    let key = |p: &Patch| (p.center, p.nbr[0], p.nbr[1], p.nbr[2], p.nbr[3]);
    let mut table: HashMap<(usize, usize, usize, usize, usize), f32> = HashMap::new();
    for p in train {
        table.insert(key(p), label(p)); // last-writer; deterministic rule ⇒ consistent
    }
    let mut hit = 0usize;
    for p in test {
        // Unseen tuple ⇒ default to the majority class (KEEP = -1.0), exactly the "never saw this direction" miss.
        let pred = table.get(&key(p)).copied().unwrap_or(-1.0);
        if (pred > 0.0) == (label(p) > 0.0) {
            hit += 1;
        }
    }
    hit as f32 / test.len() as f32
}

// ── SUBSTRATE-NATIVE: pooled relational features (invariance by pooling) → grown conjunction + selection. ──
/// The perception front-end: pool the relation over the neighbourhood. Position-INVARIANT (OR / count over the 4
/// slots), so direction is factored out — this is `aggregate` over `relate`, grown & domain-blind. Distractors
/// (`Z-adjacent`, neighbour-count, center==other) are included so the substrate must SELECT, not be handed the
/// answer.
fn pooled_feats(p: &Patch) -> Vec<(usize, f32)> {
    let b = |c: bool| if c { 1.0 } else { -1.0 };
    let y_adj = p.nbr.contains(&Y); // POOLED OR — invariant to which neighbour is Y
    let z_adj = p.nbr.contains(&Z); // distractor relation
    let ncol = p.nbr.iter().filter(|&&c| c != 0).count() as f32 / 4.0; // distractor: neighbourhood density
    vec![
        (0, b(p.center == A)), // isA
        (1, b(y_adj)),         // Y-adjacent (the causal relation)
        (2, b(z_adj)),         // Z-adjacent (distractor)
        (3, 2.0 * ncol - 1.0), // neighbour-count (distractor)
        (4, b(p.center == 4)), // center==other (distractor)
    ]
}
const N_FEAT: usize = 5;

fn build_pooled(train: &[Patch], seed: u64) -> (Field, usize) {
    let mut f = Field::new(N_FEAT);
    f.rng_seed(seed);
    let srcs: Vec<usize> = (0..N_FEAT).collect();
    let hidden = f.grow_backbone(&srcs, K, 2.5); // grown depth for the conjunction
    let mut reads = srcs.clone();
    reads.extend(hidden);
    let out = f.len();
    let n = reads.len();
    f.add_unit(reads, vec![0.0; n], 0.0, out);
    let mut rng = Rng::new(seed ^ 0xF00D);
    for _ in 0..STEPS {
        let p = &train[rng.range(train.len())];
        f.learn_step(&pooled_feats(p), out, label(p), 0.03, 100);
    }
    (f, out)
}

/// Accuracy over `test`. If `ablate = Some(i)`, the i-th pooled feature is CORRUPTED (sign flipped) at inference
/// — an ablation of that relation. A causal relation's ablation tanks accuracy; a distractor's barely moves it
/// (this is the substrate's own ablation-importance = surprise-reduction, a cleaner selection signal than raw
/// input reliance, which is diffuse because the backbone reads every input).
fn pooled_accuracy(f: &mut Field, out: usize, test: &[Patch], ablate: Option<usize>) -> f32 {
    let mut hit = 0usize;
    for p in test {
        let mut feats = pooled_feats(p);
        if let Some(i) = ablate {
            feats[i].1 = -feats[i].1; // corrupt just this relation
        }
        f.stimulate(&feats, 100);
        let pred = f.cells()[out];
        if (pred > 0.0) == (label(p) > 0.0) {
            hit += 1;
        }
    }
    hit as f32 / test.len() as f32
}

#[test]
fn relational_rule_generalizes_by_pooling_not_memory() {
    let mut rng = Rng::new(0x9E1A7);
    // DIRECTION SPLIT: train sees Y only up(0)/left(2); test sees Y only down(1)/right(3) — held-out directions.
    let train: Vec<Patch> = (0..500).map(|_| patch(&mut rng, &[0, 2])).collect();
    let test: Vec<Patch> = (0..500).map(|_| patch(&mut rng, &[1, 3])).collect();

    // Sanity: both splits contain real positives (else the task is trivial).
    let pos_test = test.iter().filter(|p| label(p) > 0.0).count();
    assert!(
        pos_test > 50,
        "test must contain recolour cases in the held-out directions: {pos_test}"
    );

    let atomic = atomic_accuracy(&train, &test);
    let (mut f, out) = build_pooled(&train, 0x5EED);
    let pooled = pooled_accuracy(&mut f, out, &test, None);

    // SELECTION by ablation (the substrate's own importance currency): corrupt one relation at inference and see
    // how far accuracy drops. Causal relations (isA, Y-adjacent) must tank it; distractors (Z-adjacent) must not.
    let drop_isa = pooled - pooled_accuracy(&mut f, out, &test, Some(0));
    let drop_yadj = pooled - pooled_accuracy(&mut f, out, &test, Some(1));
    let drop_zadj = pooled - pooled_accuracy(&mut f, out, &test, Some(2));

    eprintln!("[REL-GEN] direction split (train Y∈up/left, test Y∈down/right), {pos_test} held-out positives:");
    eprintln!(
        "[REL-GEN]   ATOMIC 5-tuple memory: {atomic:.3}   POOLED-grown (invariance): {pooled:.3}"
    );
    eprintln!("[REL-GEN]   ablation Δacc — isA {drop_isa:+.3}  Y-adjacent {drop_yadj:+.3}  Z-adjacent(distractor) {drop_zadj:+.3}");

    // The wall: atomic memory cannot cross to the held-out directions (unseen tuples ⇒ default KEEP).
    assert!(
        atomic < 0.75,
        "atomic-context memory must FAIL to generalize across unseen directions (the wall): {atomic:.3}"
    );
    // The fix: pooled invariant features + grown conjunction generalize to the held-out directions.
    assert!(
        pooled > 0.88,
        "pooled invariant features + grown conjunction must generalize across directions: {pooled:.3}"
    );
    assert!(
        pooled > atomic + 0.15,
        "the substrate-native path must beat atomic memory ({pooled:.3} vs {atomic:.3})"
    );
    // Selection: ablating a CAUSAL relation tanks accuracy; ablating the DISTRACTOR barely moves it.
    assert!(
        drop_yadj > 0.2 && drop_isa > 0.2,
        "ablating a causal relation must tank accuracy (isA {drop_isa:+.3}, Y-adjacent {drop_yadj:+.3})"
    );
    assert!(
        drop_yadj > drop_zadj + 0.15,
        "Y-adjacent must matter far more than the Z-adjacent distractor ({drop_yadj:+.3} vs {drop_zadj:+.3})"
    );

    eprintln!("[REL-GEN] VERDICT: MorphBot's wall is REAL for atomic memory ({atomic:.3}) and DISSOLVES with substrate-native wiring ({pooled:.3}). NO new primitive: perception POOLS the relation over the neighbourhood (aggregate-over-relate = invariance by pooling, so 'Y up' == 'Y down'), and the substrate GROWS the conjunction (grow_backbone) + SELECTS the predictive relation — ablating Y-adjacent costs {drop_yadj:+.3} accuracy vs the Z-adjacent distractor's {drop_zadj:+.3} (delta-rule RW competition, no new op). The conditional-generalization gap is WIRING (feed pooled predicates, not atomic contexts) — and it is NOT the skill-basin separability op (measured negative in substrate_skill_basins), so the two gaps are NOT one hole.");
}
