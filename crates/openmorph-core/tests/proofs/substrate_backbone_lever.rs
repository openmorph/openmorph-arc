// SPDX-License-Identifier: MIT-0
//! `grow_backbone`'s shared-basis width is the caller's EXPLICIT argument — the one capacity dial of the shared
//! basis (the living pool self-sizes the supervised `grow` case). A stored `Config::backbone_width` convention
//! (#128 debt) was read by no substrate code and has been removed; a caller passes the width it wants. `width`
//! is NOT overloaded — `0` keeps its literal meaning (a pure linear readout, no features).
//!
//!   cargo test -p openmorph-core --test proofs substrate_backbone_lever -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

#[test]
fn grow_backbone_width_is_the_explicit_argument() {
    // the width argument sets the size of the shared basis
    let mut c = Field::new(4);
    c.rng_seed(7);
    assert_eq!(
        c.grow_backbone(&[0, 1], 16, 2.5).len(),
        16,
        "the shared-basis width is the explicit argument"
    );

    // width is NOT overloaded — 0 still means a pure linear readout (no features)
    let mut z = Field::new(4);
    assert_eq!(
        z.grow_backbone(&[0, 1], 0, 2.5).len(),
        0,
        "0 must keep its literal meaning (no features)"
    );

    eprintln!("[BACKBONE] grow_backbone width is the caller's explicit argument (16 → 16 features); 0 keeps its literal meaning (no features) — no overload. The living pool self-sizes the supervised case.");
}
