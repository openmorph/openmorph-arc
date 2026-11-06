// SPDX-License-Identifier: MIT-0
//! THE SURPRISE CURRENCY SEES TEMPORAL CELLS. `Config::surprise_currency` keeps a running per-cell variance, the
//! `Var(c)` in `surprise_reduction_of(c) = (∂out/∂c)²·Var(c)`. It was noted for input writes and unit outputs but
//! NOT for the cells the temporal steps write — `integrate` traces and `resonate` resonators — so a temporal cell
//! always read `Var = 0`, `surprise_reduction_of` = 0, and `metabolize_surprise` would starve every temporal
//! feature however much the readout relied on it. The trace and resonator steps now write through the same
//! write-stamp-note-enqueue path as every other cell write.
//!
//! Two claims:
//!   * ON — a trace and both resonator cells read a non-zero surprise reduction, like the raw input they filter.
//!   * The dynamics are untouched — noting a value feeds nothing back, so the medium is bit-identical with the flag
//!     on or off (and with it off, the default, nothing is tracked at all).
//!
//!   cargo test -p openmorph-core --release --test proofs spike_surprise_temporal -- --nocapture

use openmorph_core::substrate::Field;

struct Run {
    field: Field,
    trace: usize,
    res: usize,
}

fn run(currency: bool) -> Run {
    let mut f = Field::new(2); // cell 0 = the raw source, cell 1 = a readout
    f.config_mut().surprise_currency = currency;
    let trace = f.integrate(0, 2.0);
    let res = f.resonate(0, 0.7, 0.2); // `res` = in-phase u, `res + 1` = quadrature v
    f.add_unit(
        vec![0, trace, res, res + 1],
        vec![0.3, 0.5, 0.4, 0.2],
        0.0,
        1,
    );
    for t in 0..200 {
        let x = (0.3 * t as f32).sin();
        f.stimulate(&[(0, x)], 50);
    }
    Run {
        field: f,
        trace,
        res,
    }
}

#[test]
fn temporal_cells_carry_surprise_currency() {
    let on = run(true);
    let f = &on.field;
    let (input, trace, u, v) = (
        f.surprise_reduction_of(0),
        f.surprise_reduction_of(on.trace),
        f.surprise_reduction_of(on.res),
        f.surprise_reduction_of(on.res + 1),
    );
    eprintln!(
        "[CURRENCY] surprise_reduction_of: input {input:.5}  trace {trace:.5}  resonator u {u:.5}  v {v:.5}"
    );
    assert!(input > 0.0, "control: the raw input is noted by stimulate");
    assert!(
        trace > 0.0,
        "a trace cell must carry its variance into the currency"
    );
    assert!(
        u > 0.0 && v > 0.0,
        "both resonator cells must carry their variance into the currency"
    );
}

#[test]
fn noting_temporal_cells_leaves_the_dynamics_bit_identical() {
    let (on, off) = (run(true), run(false));
    let bits = |f: &Field| f.cells().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&on.field),
        bits(&off.field),
        "the medium must not depend on the meter"
    );
    assert_eq!(
        off.field.surprise_reduction_of(off.trace),
        0.0,
        "off: nothing is tracked"
    );
    assert_eq!(
        off.field.surprise_reduction_of(off.res),
        0.0,
        "off: nothing is tracked"
    );
}
