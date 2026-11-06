// SPDX-License-Identifier: MIT-0
//! A TRAINED ACTOR SURVIVES snapshot/load — the free-energy actor's per-option value ENSEMBLE is its LEARNED
//! POLICY, so it must persist. It used to be `#[serde(skip)]` ("transient scratch"), so a trained preference
//! EVAPORATED on reload (verified by MorphBot: a deterministic learned choice flipped across snapshot/load). This
//! is that repro inverted to GREEN: train the actor to a deterministic preference, snapshot → load, and the
//! reloaded organism makes the SAME choice — while a never-trained control makes a DIFFERENT (default) one, so the
//! test proves real learned state crossed the boundary, not a coincidence.
//!
//!   cargo test -p openmorph-core --test proofs substrate_actor_persistence -- --nocapture

use openmorph_core::substrate::{Event, Field, Outcome};

/// Drive the free-energy actor one step over `opts`, folding the previous `(chosen, reward)`; returns the choice.
fn act(f: &mut Field, opts: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act {
        options: opts,
        last,
    }) {
        Outcome::Acted { chosen } => chosen,
        _ => 0,
    }
}

/// The deterministic greedy choice (no reward folded) — repeated `n`× it should be a single settled option.
fn greedy(f: &mut Field, opts: &[usize], n: usize) -> usize {
    let mut choice = opts[0];
    for _ in 0..n {
        choice = act(f, opts, None);
    }
    choice
}

#[test]
fn a_trained_free_energy_actor_survives_snapshot_load() {
    let opts = [100usize, 200usize];

    // Build + enable the free-energy actor.
    let mut f = Field::new(1);
    let mut cfg = f.config().clone();
    cfg.free_energy_actor = true;
    f.configure(cfg);

    // A never-trained control (same config) — its default greedy choice, to contrast against the trained one.
    let mut fresh = Field::new(1);
    let mut cfg2 = fresh.config().clone();
    cfg2.free_energy_actor = true;
    fresh.configure(cfg2);
    let default_choice = greedy(&mut fresh, &opts, 20);

    // TRAIN: reward option 200 whenever chosen, penalise 100 — folds a clear preference into the ensemble.
    let mut last = None;
    for _ in 0..300 {
        let chosen = act(&mut f, &opts, last);
        let r = if chosen == 200 { 1.0 } else { -1.0 };
        last = Some((chosen, r));
    }
    let pre = greedy(&mut f, &opts, 20);

    // SNAPSHOT → LOAD into a fresh organism.
    let bytes = serde_json::to_vec(&f).expect("Field serializes");
    let mut loaded: Field = serde_json::from_slice(&bytes).expect("Field deserializes");
    let post = greedy(&mut loaded, &opts, 20);

    eprintln!("[ACTOR-PERSIST] default(untrained)={default_choice}  trained(pre-save)={pre}  reloaded(post-load)={post}");
    eprintln!("[ACTOR-PERSIST] VERDICT: the free-energy actor's learned per-option ensemble is now part of the Field snapshot, so a TRAINED preference survives save/load (pre==post) where it used to revert to the untrained default. A warm-started brain round-trips.");

    assert_eq!(
        pre, post,
        "the trained preference must SURVIVE snapshot/load (pre {pre} vs reloaded {post})"
    );
    assert_ne!(
        pre, default_choice,
        "sanity: training must have MOVED the choice off the untrained default ({pre} vs {default_choice}), else the round-trip is trivially true"
    );

    // A load→save round-trip must be byte-stable — the deterministic BTreeMap keeps the ensemble bytes ordered.
    let bytes2 = serde_json::to_vec(&loaded).expect("re-serializes");
    assert_eq!(
        bytes, bytes2,
        "a load→save round-trip must be byte-stable (deterministic BTreeMap serialization)"
    );
}
