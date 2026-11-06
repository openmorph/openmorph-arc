// SPDX-License-Identifier: MIT-0
//! BYTE-IDENTITY GUARD (#172, S1/S2) — the constants-sweep promoted several buried literals to `Config`, and the
//! probe-harness/cosine-kernel DRY is byte-identical ONLY if those defaults equal the historical hard-coded
//! values. An adversarial review noted the byte-identity was an inspection claim, not a guarded one. This pins
//! it: if any promoted default is ever edited, this test fails LOUDLY before a behavioural-key or metric silently
//! drifts from the value it replaced.

use openmorph_core::Config;

#[test]
fn promoted_defaults_equal_the_historical_literals() {
    let d = Config::default();
    // fingerprint/embed probe harness: budget was `n_cells + 32`, key resolution 16 bins over 8 probes.
    assert_eq!(
        d.probe_settle_pad, 32,
        "fingerprint/embed settle budget was n+32"
    );
    assert_eq!(d.spectrum_bins, 16, "behavioural-key spectral resolution");
    assert_eq!(d.fingerprint_probes, 8, "behavioural-key probe count");
    // concept-map: floor 2, metric_step repulsion softening `rate/(1.0+dist2)`.
    assert_eq!(
        d.pos_dim_init, 2,
        "the concept-map dimensionality floored at 2"
    );
    assert_eq!(
        d.metric_repel_softening, 1.0,
        "metric_step repulsion used rate/(1.0 + dist2)"
    );
    // #172 structure algebra recursion guard.
    assert_eq!(
        d.struct_walk_max_depth, 64,
        "structure-walk recursion guard"
    );
    // nest behaviour-seeding default (a capability toggle, documented on the feature-flags page).
    assert!(
        d.nest_seeds_position,
        "nest_seeds_position default (documented on the capabilities page)"
    );
}
