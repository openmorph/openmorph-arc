// SPDX-License-Identifier: MIT-0
//! BEHAVIOURAL FINGERPRINT (openmorph#146 — the core seam for genetic dedup). `Field::fingerprint()` is a
//! probe-free, domain-blind content-address by WHAT A MORPH DOES: perturb it with a fixed CANONICAL probe
//! (deterministic, no caller/task input), settle, take the magnitude SPECTRUM of the response. So a marketplace
//! can dedup shared sub-morphs by behaviour — the common sub-morph collapses to one node.
//!
//! Three properties: (1) a COPY has an IDENTICAL fingerprint (dedup collapses copies); (2) morphs that behave
//! DIFFERENTLY have DIFFERENT fingerprints; (3) it is behaviour-level — two INDEPENDENT morphs trained to the
//! SAME function are closer than to a different-function morph.
//!
//!   cargo test -p openmorph-core --test proofs substrate_fingerprint -- --nocapture --test-threads=1

use crate::common::Rng;
use openmorph_core::substrate::Field;

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}
/// Grow a morph to y = k·x0 − k·x1 over 3 inputs, seeded.
fn morph(k: f32, seed: u64) -> Field {
    let mut f = Field::new(4);
    f.rng_seed(seed);
    f.config_mut().feature_lr = 0.3;
    let mut rng = Rng::new(seed ^ 0xF);
    for _ in 0..1500 {
        let x = [rng.sym(), rng.sym(), rng.sym()];
        let _ = f.grow(
            &[(0, x[0]), (1, x[1]), (2, x[2])],
            3,
            (k * x[0] - k * x[1]).tanh(),
            0.1,
            60,
        );
    }
    f
}

#[test]
fn fingerprint_content_addresses_by_behaviour() {
    let a = morph(0.6, 0x1);
    let fp_a = a.fingerprint();

    // (1) a COPY → identical fingerprint (dedup)
    let a_copy = a.clone();
    let fp_copy = a_copy.fingerprint();
    let copy_d = dist(&fp_a, &fp_copy);

    // (2) a DIFFERENT behaviour → different fingerprint
    let b = morph(-0.5, 0x2); // opposite-sign function
    let fp_b = b.fingerprint();
    let diff_d = dist(&fp_a, &fp_b);

    // (3) an INDEPENDENT morph trained to the SAME function (different seed) → closer than a different one
    let a2 = morph(0.6, 0x99);
    let fp_a2 = a2.fingerprint();
    let same_fn_d = dist(&fp_a, &fp_a2);

    eprintln!("[FINGERPRINT] fingerprint distances (probe-free spectral behavioural signature):");
    eprintln!("[FINGERPRINT]   COPY of A vs A ................. {copy_d:.5}   (dedup: a copy is byte-identical)");
    eprintln!(
        "[FINGERPRINT]   independent SAME-function vs A . {same_fn_d:.4}   (behaviour-close)"
    );
    eprintln!("[FINGERPRINT]   DIFFERENT-function vs A ........ {diff_d:.4}   (behaviour-far)");
    eprintln!("[FINGERPRINT] VERDICT: `Field::fingerprint()` content-addresses a morph by WHAT IT DOES, probe-free and domain-blind (a fixed canonical probe → settle → magnitude spectrum, #145's spectral rep in minimal core form). A COPY collapses to an IDENTICAL fingerprint (dist {copy_d:.5} ≈ 0) so a marketplace dedups shared sub-morphs to one node (openmorph#146); a DIFFERENT behaviour is far ({diff_d:.3}); and it is behaviour-level — an independent morph of the SAME function is closer ({same_fn_d:.3}) than a different one ({diff_d:.3}). This is the same behavioural signature memoization/#118-discovery/the concept-map already use; mycelium wraps it for content-addressing + lineage. No task-probe, no new search machinery in core.");

    assert!(
        copy_d < 1e-6,
        "a COPY must have an identical fingerprint (dedup) — got {copy_d}"
    );
    assert!(diff_d > same_fn_d * 2.0, "a DIFFERENT behaviour must be far, and same-function nearer ({diff_d:.3} vs {same_fn_d:.3})");
}
