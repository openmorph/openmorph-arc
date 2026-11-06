// SPDX-License-Identifier: MIT-0
//! A forward-factor DEPENDENCY survives every structural rewrite of the adjacency. `set_factor` registers the
//! factor cell as a reader edge (a change to the factor must re-fire the unit), but the adjacency rebuilds used by
//! `cull`/`cull_after`, `detach` and `merge` re-indexed from `reads` only — so after any of them a product unit
//! silently stopped updating when only its factor changed (a stale product, e.g. every `grow_products` feature after
//! the next `alive` event's cull, or a transcribed attention product after a merge). Also: replacing the factor must
//! not delete a genuine READ edge on the old factor cell.

use openmorph_core::Field;

/// cells: 0 = x, 1 = k (factor), 2 = out = k · tanh(x). Returns the field settled at x = 0.5, k = 1.
fn product_field() -> Field {
    let mut f = Field::new(3);
    f.add_unit(vec![0], vec![1.0], 0.0, 2);
    f.set_factor(2, Some(1));
    f.stimulate(&[(0, 0.5), (1, 1.0)], 100);
    f
}

fn expect(f: &Field, k: f32) {
    let want = k * 0.5f32.tanh();
    let got = f.cells()[2];
    assert!(
        (got - want).abs() < 1e-5,
        "factor edge lost: k = {k} ⇒ out {got}, want {want}"
    );
}

#[test]
fn factor_edge_survives_a_no_death_cull() {
    let mut f = product_field();
    assert_eq!(f.cull(-1e9), 0); // nothing starves — a structural no-op
    f.stimulate(&[(1, 3.0)], 100); // change ONLY the factor
    expect(&f, 3.0);
}

#[test]
fn factor_edge_survives_a_cull_that_kills_another_unit() {
    let mut f = product_field();
    // a second, unrelated unit that will starve (energy 0 < death) — the survivor's ids shift
    let dead = f.add_unit(vec![0], vec![0.3], 0.0, f.cells().len());
    let _ = dead;
    f.metabolize(0.0, 1e9); // drain everyone's energy …
    f.consolidate(&[2]); // … but the product unit is a banked skill (immune to cull)
    assert!(f.cull(0.0) >= 1);
    f.stimulate(&[(1, -2.0)], 100);
    expect(&f, -2.0);
}

#[test]
fn factor_edge_survives_merge() {
    let mut f = product_field();
    // a duplicate writer of a scratch cell, merged away — `merge` rebuilds the adjacency
    let a = f.cells().len();
    f.add_unit(vec![0], vec![0.1], 0.0, a);
    let b = f.cells().len();
    f.add_unit(vec![0], vec![0.1], 0.0, b);
    f.merge(a, b);
    f.stimulate(&[(1, 4.0)], 100);
    expect(&f, 4.0);
}

#[test]
fn replacing_a_factor_keeps_a_real_read_edge_on_the_old_cell() {
    // out = k2 · tanh(w0·x + w1·k1): the unit READS k1 (cell 1) and first uses it as the factor too.
    let mut f = Field::new(4);
    f.add_unit(vec![0, 1], vec![1.0, 1.0], 0.0, 3);
    f.set_factor(3, Some(1));
    f.set_factor(3, Some(2)); // move the factor to cell 2 — cell 1 is still a READ
    f.stimulate(&[(0, 0.2), (1, 0.1), (2, 1.0)], 100);
    f.stimulate(&[(1, 0.7)], 100); // change ONLY the read cell 1
    let want = 1.0 * (0.2f32 + 0.7).tanh();
    assert!(
        (f.cells()[3] - want).abs() < 1e-5,
        "read edge on the old factor cell was deleted: {} vs {want}",
        f.cells()[3]
    );
}
