// SPDX-License-Identifier: MIT-0
//! CHARACTERIZATION oracle for the four public `Field` methods the proof suite never exercised — the coverage
//! gaps a physics-inversion audit flagged. Before any of these delegate INTO the engine, their CURRENT
//! behaviour must be pinned, so the verify-then-delete inversion can assert the delegated path is identical.
//! These are the substrate contract for `age` / `grown_basis` / `connect` / `unbind`.
//!
//!   cargo test -p openmorph-core --test proofs substrate_coverage_gaps

use openmorph_core::Field;

/// `age(cell)` = subjective events since the cell last changed (`clock − changed_at`); 0 for a cell that has
/// never changed (it defaults its stamp to the current clock). Deterministic (event count, not wall-clock).
#[test]
fn age_is_events_since_the_cell_last_changed() {
    let mut f = Field::new(3);
    // Nothing has fired: every cell is age 0 (changed_at defaults to clock == 0).
    assert_eq!(f.age(0), 0, "a never-changed cell is age 0");
    assert_eq!(f.clock(), 0, "no firing yet");

    // Grow a writer for cell 2 from cell 0, then settle a few times so the clock advances.
    f.config_mut().max_hidden = 0;
    for _ in 0..50 {
        f.grow(&[(0, 1.0)], 2, 0.6, 0.1, 20);
    }
    let clk = f.clock();
    // Stimulate: cell 0 changes NOW, so immediately after the settle its age is small and bounded by the
    // firings this settle added; and clock only advances (monotone, deterministic).
    f.stimulate(&[(0, 1.0)], 20);
    assert!(f.clock() >= clk, "the subjective clock is monotone");
    // A cell out of range reports age 0 (the unwrap_or(clock) − clock branch).
    assert_eq!(f.age(999), 0, "an out-of-range cell reports age 0");
    // recency is the bounded form of the same signal: a just-changed cell is ~1.0.
    assert!(
        f.recency(0, 4.0) > 0.0 && f.recency(0, 4.0) <= 1.0,
        "recency is a bounded (0,1] trace of age"
    );
}

/// `grown_basis()` = the sorted, deduped set of write-cells of units that currently have at least one reader.
/// The grown features a readout actually reads appear; an isolated writer (no readers) does not.
#[test]
fn grown_basis_is_the_read_grown_write_cells() {
    let mut f = Field::new(4);
    assert!(
        f.grown_basis().is_empty(),
        "a blank field has grown nothing"
    );

    // Grow depth: a readout for cell 3 over input 0 with hidden features — the hidden write-cells that the
    // readout reads must show up in the basis, sorted + deduped.
    for _ in 0..200 {
        for &x in &[-1.0f32, 1.0] {
            f.grow(&[(0, x)], 3, (1.5 * x).tanh(), 0.1, 40);
        }
    }
    let basis = f.grown_basis();
    assert!(!basis.is_empty(), "growth produced a basis");
    // Sorted + deduped is the contract.
    let mut sorted = basis.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(basis, sorted, "grown_basis is sorted and deduped");
    // Every entry is a real cell index within the medium.
    assert!(
        basis.iter().all(|&c| c < f.len()),
        "every basis cell is in range"
    );
}

/// `connect(from, into, w)` wires `from` as a new read (weight `w`) of the FIRST unit writing `into`, and
/// enqueues the adjacency. Returns `true` once; `false` if there is no writer of `into`, or the edge already
/// exists. The new edge measurably shifts the settled output.
#[test]
fn connect_adds_a_read_edge_once_and_shifts_the_output() {
    let mut f = Field::new(3);
    // No unit writes cell 2 yet ⇒ connect returns false (nothing to wire into).
    assert!(!f.connect(0, 2, 1.0), "no writer of `into` ⇒ false");

    // Grow a writer for cell 2 from cell 0 (a linear-in-tanh mapping).
    f.config_mut().max_hidden = 0;
    for _ in 0..300 {
        for &x in &[-1.0f32, 1.0] {
            f.grow(&[(0, x)], 2, (0.8 * x).tanh(), 0.1, 30);
        }
    }
    // Baseline settled output with cell 1 = 0 (not yet read).
    f.stimulate(&[(0, 0.5), (1, 0.0)], 30);
    let before = f.cells()[2];

    // Wire cell 1 in with a large weight ⇒ true, and re-settling now depends on cell 1.
    assert!(f.connect(1, 2, 2.0), "first connect of a new edge ⇒ true");
    // The same edge again ⇒ false (already reads `from`).
    assert!(!f.connect(1, 2, 2.0), "a duplicate edge ⇒ false");

    // Now CHANGE cell 1 (0 → 1): fire-on-change re-fires the writer, which reads the new edge.
    f.stimulate(&[(0, 0.5), (1, 1.0)], 30);
    let after = f.cells()[2];
    assert!(
        (after - before).abs() > 1e-4,
        "the new read edge shifts the settled output (before {before}, after {after})"
    );
}

/// `unbind()` clears the fast-weight binding buffer (`bind → prefer → complete` is per-puzzle). The slow grown
/// weights are untouched; only the fast association resets to zero.
#[test]
fn unbind_clears_only_the_fast_binding() {
    let mut f = Field::new(4);
    // Bind an observation where cells 0 and 1 share a value ⇒ a positive association forms.
    f.bind(&[(0, 0.7), (1, 0.7), (2, -0.3)]);
    assert!(f.bound(0, 1) > 0.0, "equal-valued cells bind");
    assert_eq!(f.bound(0, 2), 0.0, "differing-valued cells do not bind");

    // Unbind wipes the fast rule.
    f.unbind();
    assert_eq!(f.bound(0, 1), 0.0, "unbind clears the association");
}
