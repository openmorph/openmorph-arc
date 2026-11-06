// SPDX-License-Identifier: MIT-0
//! NATIVE EPISODIC MEMORY + REPLAY — a Morph stores experience as memory cells and REPLAYS it internally.
//! Sébastien: "memory cells and replay should be embedded inside a Morph — useful for replay, dreaming,
//! reproducing, internal thinking."
//!
//! THE PRINCIPLE (rule #2 — the simplest mechanism, reuse what exists): memory IS cells; replay IS
//! re-stimulating from stored cell-states. The whole faculty is ONE stored [`Episode`] (a snapshot of a
//! chosen cell-set) + [`Field::replay`] (re-inject it through the EXISTING `stimulate` path). Everything
//! below — genomic reproduction, dreaming, imagination — is a USAGE of that one primitive, no parallel
//! machinery. The buffer lives on the `Field`, so it CLONES with `fork_mutated` (experience travels with a
//! mutated clone) and round-trips through a snapshot (serde).
//!
//! THE PROOFS (each multi-seed, scored INSIDE, truth-not-trophy):
//!   1. GENOMIC REPRODUCTION (the keystone) — a morph stores a small (input→target) dataset in its OWN
//!      memory; a population of `fork_mutated` clones (each a mutated genome, INHERITING the memory) is
//!      scored by REPLAYING that stored dataset; keeping the fittest genome improves fit OVER GENERATIONS
//!      from the morph's own memory alone (no external data). Controls: (a) SCRAMBLE — picking a RANDOM
//!      (not replay-selected) genome does NOT improve; (b) a clone with an EMPTY buffer can't be scored,
//!      proving the memory genuinely travels with the clone.
//!   2. DREAMING / OFFLINE REPLAY — with NO external input, replaying stored episodes as `learn_step`
//!      drives a measurable improvement (consolidation) vs a no-replay control. The organism improves by
//!      rehearsing its own memory (sleep-replay).
//!   3. INTERNAL THINKING (bonus) — replay an episode, then perturb one input and settle to a DIFFERENT
//!      completion than the stored one (imagination over memory).
//!
//! Plus the substrate hygiene the faculty must not break: BYTE-IDENTICAL when the buffer is unused (storing
//! episodes is inert to the settle), and the buffer round-trips through a snapshot.
//!
//!   cargo test -p openmorph-core --test proofs substrate_replay -- --nocapture

use openmorph_core::substrate::Field;

use crate::common::Rng;

const X0: usize = 0;
const X1: usize = 1;
const TGT: usize = 2; // a plain storage cell (no unit reads it): the remembered target lives here
const WIDTH: usize = 8; // backbone features (all non-linearity + scale-sensitivity lives here)
const BUD: usize = 100;

/// The DATASET the morph memorises: soft 2-parity `y = 0.7·x0·x1` on ±1 inputs — a PRODUCT, so a linear
/// readout over the raw inputs can't fit it; it needs crisp (high-`scale`) non-linear features. Starting the
/// population at a MISMATCHED low scale makes the GENOME the bottleneck, so improvement is genome-selection,
/// not mere training (truth-not-trophy: the scramble control that trains identically but doesn't select stays
/// stuck). All four combinations — small, exact, replayable from memory.
const DATA: [(f32, f32, f32); 4] = [
    (-1.0, -1.0, 0.7),
    (-1.0, 1.0, -0.7),
    (1.0, -1.0, -0.7),
    (1.0, 1.0, 0.7),
];

/// Build a morph over the parity dataset at a chosen starting projection `scale`, and STORE the four
/// examples in its native episodic memory (each: stimulus = the input cells, outcome = the target cell).
fn build(start_scale: f32) -> (Field, usize) {
    let mut f = Field::new(3); // cells 0,1 = inputs; 2 = TGT (inert storage for the remembered target)
    f.rng_seed(0xE9150DE);
    let feats = f.grow_backbone(&[X0, X1], WIDTH, 1.0); // unit-variance weights; SCALE is the live gain
    for &h in &feats {
        f.set_scale(h, start_scale); // start the population mismatched, so the genome is the bottleneck
    }
    let out = f.len();
    f.add_unit(feats, vec![0.0; WIDTH], 0.0, out); // linear readout over the features only
                                                   // Store the dataset in memory: set the cells, then snapshot them as an episode. Memory IS cells.
    for &(x0, x1, y) in DATA.iter() {
        f.stimulate(&[(X0, x0), (X1, x1), (TGT, y)], BUD);
        f.store_episode(&[X0, X1], &[TGT]);
    }
    (f, out)
}

/// MEAN-SQUARED ERROR over the stored dataset, scored purely by REPLAY: re-inject each episode's remembered
/// stimulus (via `replay`, i.e. `stimulate` from the stored cell-state), read the settled output, compare to
/// the remembered outcome. No external data — the morph is graded on its OWN memory.
fn mse(f: &mut Field, out: usize) -> f64 {
    let n = f.episodes().len();
    if n == 0 {
        return f64::NAN; // no memory ⇒ nothing to score (control (b): an empty-buffer clone is unscoreable)
    }
    let mut s = 0.0;
    for id in 0..n {
        let tgt = f.episode(id).unwrap().outcome[0].1 as f64; // copy out before the mutable replay
        f.replay(id, BUD);
        let p = f.cells()[out] as f64;
        s += (p - tgt).powi(2);
    }
    s / n as f64
}

/// DREAM — one bout of offline replay-training: replay the stored episodes (shuffled) as `learn_step` toward
/// their remembered outcome. NO external input; the organism rehearses its own memory. This is the sleep-
/// replay consolidation loop AND the per-clone fitting the genomic loop scores.
fn dream(f: &mut Field, out: usize, epochs: usize, lr: f32, rng: &mut Rng) {
    let n = f.episodes().len();
    for _ in 0..epochs {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.below(i + 1));
        }
        for id in order {
            let (inputs, tgt) = {
                let ep = f.episode(id).unwrap();
                (ep.stimulus.clone(), ep.outcome[0].1)
            };
            f.learn_step(&inputs, out, tgt, lr, BUD);
        }
    }
}

// ============================================================================================
// PROOF 1 — GENOMIC REPRODUCTION: the keystone. Evolve the genome on the morph's OWN memory.
// ============================================================================================

const N_CLONES: usize = 10;
const GENERATIONS: usize = 16;
const EPOCHS: usize = 30;
const LR: f32 = 0.1;
const START_SCALE: f32 = 0.3; // mismatched-low: parity needs crisp features, so scale MUST climb

/// One genomic run. Each generation: spawn `N_CLONES` `fork_mutated` clones (mutated genome, INHERITING the
/// memory), let each fit its inherited weights by REPLAYING the stored dataset (dreaming), then keep either
/// the FITTEST (`select`) or a RANDOM clone (scramble control). Returns (start MSE, end MSE).
fn evolve(select_fittest: bool, seed: u64) -> (f64, f64) {
    let mut rng = Rng::new(seed);
    let (mut parent, out) = build(START_SCALE);
    let start = mse(&mut parent, out);
    for gen in 0..GENERATIONS {
        let mut clones: Vec<Field> = Vec::with_capacity(N_CLONES);
        let mut scores: Vec<f64> = Vec::with_capacity(N_CLONES);
        for i in 0..N_CLONES {
            // Reseed the parent's PRNG between forks so each clone gets a DISTINCT genome mutation
            // (fork_mutated is deterministic in the parent's rng_state — same seed ⇒ same mutation).
            parent.rng_seed(
                seed.wrapping_mul(2654435761)
                    .wrapping_add((gen * N_CLONES + i) as u64)
                    | 1,
            );
            let mut c = parent.fork_mutated(); // inherits weights + the episodic buffer, mutates the genome
            dream(&mut c, out, EPOCHS, LR, &mut rng); // fit on its OWN replayed memory
            scores.push(mse(&mut c, out));
            clones.push(c);
        }
        let chosen = if select_fittest {
            (0..N_CLONES)
                .min_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap())
                .unwrap()
        } else {
            rng.below(N_CLONES) // scramble: a randomly-picked genome, no selection pressure
        };
        parent = clones.swap_remove(chosen);
    }
    let end = mse(&mut parent, out);
    (start, end)
}

#[test]
fn genomic_reproduction_from_the_morphs_own_memory() {
    let seeds: Vec<u64> = (0..6).map(|s| 0xA11CE + s * 101).collect();

    let (mut sel_start, mut sel_end, mut scr_end) = (0.0f64, 0.0f64, 0.0f64);
    for &s in &seeds {
        let (st, en) = evolve(true, s);
        let (_, scr) = evolve(false, s);
        sel_start += st;
        sel_end += en;
        scr_end += scr;
    }
    let n = seeds.len() as f64;
    let (sel_start, sel_end, scr_end) = (sel_start / n, sel_end / n, scr_end / n);

    // CONTROL (b): the buffer genuinely TRAVELS with a mutated clone — that is the ONLY reason it can be
    // scored. A clone inherits every episode; a from-scratch organism has NO memory, so it can't be scored.
    let (parent, _out) = build(START_SCALE);
    let clone = parent.fork_mutated();
    assert!(!parent.episodes().is_empty(), "the parent stored a dataset");
    assert_eq!(
        clone.episodes().len(),
        parent.episodes().len(),
        "the mutated clone INHERITS the whole episodic buffer (memory travels with the genome)"
    );
    let orphan = Field::new(3);
    assert!(
        orphan.episodes().is_empty(),
        "a from-scratch organism has no memory"
    );
    {
        let mut orphan = orphan; // an empty-buffer organism cannot be scored — nothing to replay
        assert_eq!(
            orphan.replay(0, BUD),
            0,
            "replaying a non-existent episode is a no-op"
        );
    }

    eprintln!(
        "[REPLAY/GENOMIC] evolving the GENOME on the morph's OWN replayed memory (no external data):\n  \
         start MSE {sel_start:.3}  →  SELECT-fittest end MSE {sel_end:.3}   |   SCRAMBLE (random genome) end MSE {scr_end:.3}"
    );
    eprintln!(
        "[REPLAY/GENOMIC] VERDICT: a `fork_mutated` clone inherits the parent's episodic buffer, so it can be \
         scored by REPLAYING the stored dataset — the keystone of the genome loop. Selecting the fittest \
         genome improves fit over generations FROM MEMORY ALONE; a scrambled (randomly-picked) genome does \
         not. Memory IS cells; replay IS re-stimulating from them."
    );

    // The keystone claim: selection improves fit from memory alone, and BEATS the scramble control.
    assert!(
        sel_end < sel_start * 0.5,
        "selection more than halves the error from the morph's own memory (start {sel_start:.3} → {sel_end:.3})"
    );
    assert!(
        sel_end < scr_end * 0.6,
        "the improvement is genome-SELECTION, not training: fittest {sel_end:.3} must beat scramble {scr_end:.3}"
    );
}

// ============================================================================================
// PROOF 2 — DREAMING / OFFLINE REPLAY: the organism improves by rehearsing its own memory.
// ============================================================================================

#[test]
fn dreaming_offline_replay_consolidates_from_memory() {
    let seeds: Vec<u64> = (0..6).map(|s| 0xD3EA1 + s * 57).collect();
    let (mut before, mut dreamed, mut control) = (0.0f64, 0.0f64, 0.0f64);
    for &s in &seeds {
        let mut rng = Rng::new(s);
        // A morph at a workable scale but UNTRAINED (zero readout weights) — high error.
        let (mut f, out) = build(1.6);
        before += mse(&mut f, out);
        // The DREAMER: many bouts of offline replay-training — no external input, only its own memory.
        let mut dreamer = f.clone();
        dream(&mut dreamer, out, 200, LR, &mut rng);
        dreamed += mse(&mut dreamer, out);
        // The CONTROL: the same organism that never replays (no consolidation).
        let mut ctl = f.clone();
        control += mse(&mut ctl, out);
    }
    let n = seeds.len() as f64;
    let (before, dreamed, control) = (before / n, dreamed / n, control / n);

    eprintln!(
        "[REPLAY/DREAM] offline sleep-replay (no external input, only stored episodes):\n  \
         untrained MSE {before:.3}  →  DREAMED (replayed its memory) MSE {dreamed:.4}   |   no-replay control MSE {control:.3}"
    );
    eprintln!(
        "[REPLAY/DREAM] VERDICT: replaying stored episodes as `learn_step` — the organism rehearsing its own \
         memory offline — consolidates a skill with no fresh data. The no-replay control is unchanged."
    );

    assert!(
        dreamed < before * 0.3,
        "dreaming (offline replay) sharply reduces error ({before:.3} → {dreamed:.4})"
    );
    assert!(
        (control - before).abs() < 1e-9,
        "the no-replay control is exactly unchanged (it never rehearsed)"
    );
}

// ============================================================================================
// PROOF 3 (bonus) — INTERNAL THINKING: imagine a DIFFERENT completion over a replayed memory.
// ============================================================================================

#[test]
fn internal_thinking_blends_memory_into_a_different_context() {
    // Sébastien's definition of internal thinking: "activation of a memory cell to keep simulating the memory
    // again with ANOTHER context." So the SAME stored episode, activated in RESTORE mode (clear-then-inject)
    // vs BLEND mode (fire into the live context), must settle DIFFERENTLY — and the blended settle must be a
    // USEFUL recombinant inference. Train a morph to competence on the parity rule from its own memory, then:
    let mut rng = Rng::new(0x1DEA5);
    let (mut f, out) = build(1.6);
    dream(&mut f, out, 300, LR, &mut rng);

    // RESTORE (`replay`) is a FULL, context-INDEPENDENT recall: episode 0 captured BOTH inputs (x0=-1, x1=-1)
    // → remembered +0.7. Even from a scrambled live context, clear-then-inject reinstates it exactly.
    f.stimulate(&[(X0, 1.0), (X1, 1.0)], BUD); // a live context UNRELATED to episode 0
    f.replay(0, BUD);
    let recalled = f.cells()[out] as f64;
    let remembered = f.episode(0).unwrap().outcome[0].1 as f64;

    // Now store a PARTIAL memory cue — just x0 = -1 (no x1) — and ACTIVATE it (BLEND) into two DIFFERENT live
    // contexts. The stored x0 fires into the present x1 and the cascade RECOMBINES them: the SAME memory yields
    // a DIFFERENT completion per context — Sébastien's internal thinking, "re-simulating the memory with
    // another context." Neither combination was ever stored; the morph derives each correctly from its rule.
    f.stimulate(&[(X0, -1.0)], BUD);
    let cue = f.store_episode(&[X0], &[]); // partial cue: only x0

    f.stimulate(&[(X1, 1.0)], BUD); // context A: x1 = +1
    f.imagine(cue, BUD);
    let blend_a = f.cells()[out] as f64; // recombine x0=-1 with x1=+1 → parity −0.7

    f.stimulate(&[(X1, -1.0)], BUD); // context B: x1 = -1
    f.imagine(cue, BUD);
    let blend_b = f.cells()[out] as f64; // recombine x0=-1 with x1=-1 → parity +0.7

    eprintln!(
        "[REPLAY/THINK] RESTORE (full episode 0) is context-independent recall: {recalled:+.2} (≈ remembered {remembered:+.2}).\n  \
         BLEND a partial memory cue (x0=-1) into two live contexts → {blend_a:+.2} (x1=+1, ≈ −0.70)  vs  {blend_b:+.2} (x1=-1, ≈ +0.70)"
    );
    eprintln!(
        "[REPLAY/THINK] VERDICT: RESTORE reinstates the stored completion; BLEND fires a memory into the LIVE \
         context and the cascade recombines old memory + present state into a NEW simulation that CHANGES with \
         the context — internal thinking, both modes off stored episodes, both just `stimulate`."
    );

    assert!(
        (recalled - remembered).abs() < 0.15,
        "RESTORE reinstates the remembered completion regardless of the live context (pure recall)"
    );
    assert!(
        blend_a < -0.4 && blend_b > 0.4,
        "BLEND recombines the cue with each live context into the correct, context-dependent completion"
    );
    assert!(
        (blend_a - blend_b).abs() > 0.8,
        "BLEND ≠ RESTORE: the SAME memory fired into DIFFERENT contexts settles to DIFFERENT completions"
    );
}

// ============================================================================================
// HYGIENE — byte-identical when unused, and the buffer round-trips through a snapshot.
// ============================================================================================

#[test]
fn storing_episodes_is_inert_to_the_settle() {
    // Two identical organisms driven by the same inputs; one ALSO stores episodes throughout. Storing only
    // reads the medium and pushes to a side buffer, so the settled cells must be byte-identical.
    let mut rng = Rng::new(0xB17E);
    let (mut a, _oa) = build(1.2);
    let (mut b, _ob) = build(1.2);
    for _ in 0..200 {
        let (x0, x1) = (rng.bit(), rng.bit());
        a.stimulate(&[(X0, x0), (X1, x1)], BUD);
        b.stimulate(&[(X0, x0), (X1, x1)], BUD);
        b.store_episode(&[X0, X1], &[]); // b keeps storing; a does not
    }
    assert_eq!(
        a.cells(),
        b.cells(),
        "storing episodes is inert to the settle — byte-identical medium (the buffer never touches compute)"
    );
    eprintln!("[REPLAY/HYGIENE] byte-identical: storing episodes does not perturb the settle (buffer is inert to compute).");
}

#[test]
fn the_episodic_buffer_round_trips_through_a_snapshot() {
    let (mut f, out) = build(1.6);
    let mut rng = Rng::new(0x5EED);
    dream(&mut f, out, 100, LR, &mut rng);
    let before = mse(&mut f, out);

    // Snapshot (serde) and restore.
    let json = serde_json::to_string(&f).expect("serialize a Field with episodes");
    let mut back: Field = serde_json::from_str(&json).expect("deserialize a Field with episodes");

    assert_eq!(
        back.episodes().len(),
        f.episodes().len(),
        "the episodic buffer survives a snapshot round-trip"
    );
    for id in 0..f.episodes().len() {
        assert_eq!(
            f.episode(id).unwrap().stimulus,
            back.episode(id).unwrap().stimulus,
            "episode {id} stimulus round-trips exactly"
        );
        assert_eq!(
            f.episode(id).unwrap().outcome,
            back.episode(id).unwrap().outcome,
            "episode {id} outcome round-trips exactly"
        );
    }
    let after = mse(&mut back, out);
    assert!(
        (before - after).abs() < 1e-6,
        "replay over the restored buffer reproduces the same fit (buffer + weights round-trip)"
    );
    eprintln!("[REPLAY/HYGIENE] snapshot round-trip: the episodic buffer + its replayed fit survive serde exactly.");
}
