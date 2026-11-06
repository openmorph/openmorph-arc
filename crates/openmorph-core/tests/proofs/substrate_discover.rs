// SPDX-License-Identifier: MIT-0
//! GROWN DISCOVERY = THE BEHAVIOR FRONTIER (MorphLab's discovery wall, the substrate answer). Reuse is solved
//! (the collapse: a library of nested skills); DISCOVERY — finding a reward you've never seen — is the wall.
//! The honest trap: earlier measurement showed STATE-level intrinsic bonuses HURT (count/empowerment/
//! world-model-surprise all underperformed and were tuned to zero) — the world-model is starved per-game so its
//! surprise is noise, and a NEW STATE sits near its creator on the map, so it can't be flagged as frontier.
//!
//! The signal that escapes this is BEHAVIOR-level novelty, and the memoization the collapse added gives it for
//! FREE: a `nested` cell's cache-MISS (`nested_settles` ticks) is exactly "this behavior is new" — a skill
//! applied where it hasn't been IS a genuinely new region (unlike a new state). This proof establishes the
//! SUBSTRATE claim: `nested_settles` is an EXACT behaviour-novelty meter — a distinct behaviour is one
//! cache-miss, a repeat is a cache-hit, with no counting machinery of its own.
//!
//! TRUTH-NOT-TROPHY: the *policy* built on this meter — "prefer the un-memoized behaviour, systematically cover
//! behaviour-space, find the sparse reward" — is a HARNESS PROTOCOL (like the rest of the reasoner chain), not a
//! substrate faculty, and its "covers 100% vs random ~63% under a covering budget" is coupon-collector
//! ARITHMETIC about exhaustive coverage, not a grown result. So it is NOT asserted here; only the grown novelty
//! meter is. The frontier policy is validated where it belongs — on real ARC-3 by MorphLab (epsilon 12% →
//! behaviour-frontier 20%, new-solved 2→4).
//!
//!   cargo test -p openmorph-core --test proofs substrate_discover -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const M: usize = 6; // skills (nested cells)
const S: usize = 50; // distinct input states
const BUD: usize = 12; // sub settle budget

/// A leaf skill i: `out = tanh(scale_i · x)` — a distinct behavior. Cells 0=in, 1=out.
fn skill(scale: f32) -> Field {
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![scale], 0.0, 1);
    f
}

/// Build a host with M nested skills, skill i reading its OWN input cell `i` and writing cell `M+i`, so a
/// behavior (skill i, state s) is fired independently by `stimulate(&[(i, state_value(s))])`.
fn host() -> Field {
    let mut f = Field::new(M);
    for i in 0..M {
        f.nest(skill(0.5 + i as f32 * 0.3), &[i], M + i, BUD);
    }
    f
}

/// A distinct input value per state index (deterministic, spread in [-1,1]).
fn state_value(s: usize) -> f32 {
    -1.0 + 2.0 * (s as f32 + 0.5) / S as f32
}

/// Fire behavior (skill `i`, state `s`) and return whether it was a cache-MISS (a NEW behavior).
fn fire(f: &mut Field, i: usize, s: usize) -> bool {
    let before = f.nested_settles();
    f.stimulate(&[(i, state_value(s))], BUD);
    f.nested_settles() > before
}

#[test]
fn cache_miss_is_an_exact_behavior_novelty_meter() {
    // The memoization counter IS the novelty meter: covering all M·S distinct behaviours ticks nested_settles
    // exactly M·S times (each is one cache-MISS), and every subsequent repeat is a cache-HIT (no new settle).
    let mut f = host();
    let mut first_pass_misses = 0usize;
    for i in 0..M {
        for s in 0..S {
            if fire(&mut f, i, s) {
                first_pass_misses += 1;
            }
        }
    }
    let after_cover = f.nested_settles();
    for i in 0..M {
        for s in 0..S {
            let miss = fire(&mut f, i, s); // all repeats now — must be cache HITS
            assert!(
                !miss,
                "a repeated behavior ({i},{s}) must be a cache-hit, not a new settle"
            );
        }
    }

    eprintln!(
        "[DISCOVER] behavior space = {M} skills × {S} states = {} behaviors:\n  first pass: {first_pass_misses} cache-MISSES (= distinct behaviors)\n  second pass: 0 new settles (all cache-HITS) — nested_settles held at {}",
        M * S,
        f.nested_settles()
    );
    eprintln!("[DISCOVER] VERDICT: `nested_settles` is an EXACT behaviour-novelty meter, and it falls out of the collapse's memoization — no new primitive. A cache-MISS on a nested cell is novelty at the behaviour level: a skill applied where it hasn't been is a genuinely new region (unlike a new STATE, which sits near its creator on the map — why state-level curiosity/count/world-surprise bonuses HURT and were tuned to zero). The DISCOVERY POLICY that navigates to this frontier is a harness protocol built on the meter (its coverage-vs-random number is coupon-collector arithmetic, not asserted here); it is validated where it counts — on real ARC-3 by MorphLab (epsilon 12% → behaviour-frontier 20%, new-solved 2→4). The substrate's contribution is the free, exact novelty signal.");

    assert_eq!(
        first_pass_misses,
        M * S,
        "each distinct behavior must be exactly one cache-miss"
    );
    assert_eq!(
        after_cover as usize,
        M * S,
        "nested_settles must equal the number of distinct behaviors covered"
    );
    assert_eq!(
        f.nested_settles(),
        after_cover,
        "repeats must be cache-hits (no new settles)"
    );
}
