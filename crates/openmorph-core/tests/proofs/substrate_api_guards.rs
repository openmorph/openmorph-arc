// SPDX-License-Identifier: MIT-0
//! PUBLIC-API GUARDS — bad input fails fast at the boundary, or reads as absent, instead of surfacing later as an
//! out-of-bounds panic deep in the settle/learn path (or as a silent pick of cell 0). Every guard is inert on valid
//! input, so the default path stays byte-identical.
//!
//! - `add_unit` checks one weight per read (a mismatch used to be accepted and panic later inside `learn_step`).
//! - `think` / `dream` read a cell outside the medium as `0.0` (they indexed the medium directly and panicked).
//! - `Field::sample` reads an out-of-range slot as empty, like `Membrane::read`.
//! - `grouping::relate` / `Object::centroid` treat `width == 0` as a 1-wide column (a remainder by zero before).
//! - In debug builds, the index fallbacks that return `0` for an EMPTY option list (`select_explore`, `route`) now
//!   `debug_assert!` — `0` is a real cell/index, so the failure was invisible. `alive(Act)` over no options is the
//!   documented EPISODE END instead: it folds `last` with no bootstrap on every rule and returns `usize::MAX`.
//!
//!   cargo test -p openmorph-core --release --test proofs substrate_api_guards -- --nocapture

use openmorph_core::grouping::{relate, segment_objects, Object};
use openmorph_core::{Binding, Field, Flow};

#[test]
#[should_panic(expected = "add_unit: weights.len()")]
fn add_unit_rejects_a_weight_count_that_differs_from_the_read_count() {
    let mut f = Field::new(3);
    // Two reads, one weight. Before the guard this was accepted, and the first `learn_step` on cell 2 then
    // indexed `weights[1]` out of bounds.
    f.add_unit(vec![0, 1], vec![0.5], 0.0, 2);
}

#[test]
fn add_unit_accepts_matching_lengths_unchanged() {
    let mut f = Field::new(3);
    let id = f.add_unit(vec![0, 1], vec![0.5, -0.25], 0.1, 2);
    assert_eq!(id, 0);
    f.stimulate(&[(0, 1.0), (1, 1.0)], 10);
    let expect = (0.5f32 - 0.25 + 0.1).tanh();
    assert_eq!(f.cells()[2].to_bits(), expect.to_bits());
}

#[test]
fn think_reads_cells_outside_the_medium_as_zero() {
    let mut f = Field::new(2);
    f.stimulate(&[(1, 0.5)], 10);
    // state cell 99 and predicted-next cell 98 do not exist: they read 0.0 and writes to them are dropped
    // (as `stimulate` already drops them). Before the guard: index-out-of-bounds panic.
    let out = f.think(&[0, 99], &[1, 98], 3, 0.0, 10);
    eprintln!("[GUARD] think over out-of-range cells → {out:?}");
    assert_eq!(out, vec![0.5, 0.0]);
}

#[test]
fn think_on_a_frozen_field_reads_cells_outside_the_medium_as_zero() {
    let mut f = Field::new(2);
    f.crystallize();
    assert_eq!(f.think(&[0, 99], &[1, 98], 3, 0.0, 10), vec![0.0, 0.0]);
}

#[test]
fn dream_over_cells_outside_the_medium_does_not_panic() {
    let mut f = Field::new(2);
    let hb = f.dream(&[0, 50], &[1, 51], &[], &[vec![1.0, 1.0]], 2, 0.1, 1, 10);
    eprintln!("[GUARD] dream over out-of-range cells → {hb:?}");
    assert!(hb.slept, "the dream completes its bout and sleeps");
    assert_eq!(f.cells().len(), 2, "out-of-range cells are never allocated");
}

#[test]
fn sample_reads_an_out_of_range_slot_as_empty() {
    let f = Field::new(2);
    let past_end = Binding {
        slots: vec![(1, 4, Flow::Out)],
        ports: Vec::new(),
    };
    assert!(
        f.sample(&past_end, 0).is_empty(),
        "a slot past the medium reads empty (Membrane::read's guard)"
    );
    // An in-range slot reads exactly as before.
    let mut g = Field::new(4);
    g.stimulate(&[(2, 0.5)], 10);
    let ok = Binding {
        slots: vec![(1, 3, Flow::Out)],
        ports: Vec::new(),
    };
    assert_eq!(g.sample(&ok, 0), &[0.0, 0.5, 0.0]);
    assert!(
        g.sample(&ok, 7).is_empty(),
        "an unknown port is (0, 0), as before"
    );
}

#[test]
fn grouping_treats_a_zero_width_grid_as_one_column() {
    let o = Object {
        cells: vec![0, 1, 2],
        color: 1,
        bbox_w: 1,
        bbox_h: 3,
    };
    // width 0 used to be a remainder-by-zero panic; it now reads as the width-1 grid.
    assert_eq!(o.centroid(0), o.centroid(1));
    assert_eq!(o.centroid(1), (0.0, 1.0));
    let rels = relate(&[o.clone(), o], 0, 0);
    assert_eq!(rels.len(), 2);
    assert!(rels.iter().all(|r| r.adjacent && r.same_color));
    assert!(segment_objects(&[1, 1], 0, 2, 0).is_empty());
    // A real width is untouched.
    let grid = [0, 1, 1, 0, 0, 2, 3, 0, 2];
    let objs = segment_objects(&grid, 3, 3, 0);
    assert_eq!(objs.len(), 3);
    assert_eq!(objs[0].centroid(3), (1.5, 0.0));
}

/// Pins the tie behaviour the `select` doc now states (ties go to the LAST tied gate, `Iterator::max_by`). Not a
/// fix: the tie policy is an open owner decision. Change this test together with the doc if the policy changes.
#[test]
fn select_breaks_exact_ties_to_the_last_gate() {
    let f = Field::new(4); // every cell is 0.0, so every gate ties
    assert_eq!(f.select(&[0, 1, 2]), 2);
    assert_eq!(f.select(&[3, 0]), 1);
}

/// The empty-option fallbacks are `debug_assert!`s: they fire only in a debug-assertions build (release output is
/// unchanged). Run with `CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true` to exercise them under `--release`.
#[cfg(debug_assertions)]
mod empty_options {
    use openmorph_core::substrate::{Event, Field, Outcome};

    /// An `Act` over no options is the episode end: on every rule path it folds `last`, chooses nothing
    /// (`usize::MAX`, so a caller that indexes with it fails loudly) and leaves the next `Act` working.
    fn episode_end(mut f: Field) {
        let chosen =
            |f: &mut Field, options: &[usize], last| match f.alive(Event::Act { options, last }) {
                Outcome::Acted { chosen } => chosen,
                _ => unreachable!("alive(Act) always acts"),
            };
        let a = chosen(&mut f, &[0, 1], None);
        assert!(a < 2);
        assert_eq!(chosen(&mut f, &[], Some((a, 1.0))), usize::MAX);
        assert_eq!(chosen(&mut f, &[], None), usize::MAX);
        assert!(chosen(&mut f, &[0, 1], None) < 2);
    }

    /// The episode end only folds: the decision diagnostic and the act PRNG are untouched, so a run with an
    /// episode end chooses exactly as one whose last reward is folded by the next `Act` instead (γ = 0).
    #[test]
    fn the_episode_end_only_folds() {
        let run = |end: bool| {
            let mut f = Field::new(4);
            let mut picks = Vec::new();
            let mut last = None;
            for step in 0..40 {
                let a = match f.alive(Event::Act {
                    options: &[0, 1, 2, 3],
                    last,
                }) {
                    Outcome::Acted { chosen } => chosen,
                    _ => unreachable!(),
                };
                picks.push(a);
                last = Some((a, if a == 2 { 1.0 } else { 0.0 }));
                if end && step % 5 == 4 {
                    let spread = f.g_spread();
                    assert!(
                        matches!(f.alive(Event::Act { options: &[], last: last.take() }), Outcome::Acted { chosen } if chosen == usize::MAX)
                    );
                    assert_eq!(
                        f.g_spread().to_bits(),
                        spread.to_bits(),
                        "g_spread is untouched"
                    );
                }
            }
            picks
        };
        assert_eq!(
            run(true),
            run(false),
            "at γ = 0 an episode end changes no choice"
        );
    }

    #[test]
    fn valence_path() {
        episode_end(Field::new(2));
    }

    #[test]
    fn free_energy_path() {
        let mut f = Field::new(2);
        f.config_mut().act_rule = openmorph_core::ActRule::FreeEnergy;
        episode_end(f);
    }

    #[test]
    #[should_panic(expected = "select_explore: no options")]
    fn select_explore_empty() {
        let _ = Field::new(2).select_explore(&[], 1.0);
    }

    #[test]
    #[should_panic(expected = "route: empty")]
    fn route_empty_leaf() {
        let f = Field::new(2);
        let index = f.index_skills(&[], 4, 2);
        let _ = f.route(&index, &[0.0, 0.0], 1, 0);
    }
}
