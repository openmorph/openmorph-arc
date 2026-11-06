// SPDX-License-Identifier: MIT-0
//! SNAPSHOT ROBUSTNESS — a grown organism serialized before the heritable-trait fields existed must LOAD and
//! RUN without panicking. `scale`/`lr`/`mut_rate`/`credit` are `#[serde(default)]`, so an older snapshot
//! deserializes them EMPTY while `units` is full; the hot paths then index them directly (`scale[u]`,
//! `credit[fj]`, …). Without self-healing that is an out-of-bounds panic the moment the loaded organism is
//! used. `Field::hydrate_traits` pads the trait vectors to `units.len()` with their defaults at the entry of
//! every direct-indexing path, so a legacy snapshot is safe. This is the regression test for that.
//!
//!   cargo test -p openmorph-core --test proofs substrate_snapshot -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

/// Grow a small field with a few units so the parallel trait vectors are non-empty.
fn grown() -> Field {
    let mut f = Field::new(4);
    f.rng_seed(0x5AFE);
    f.add_unit(vec![0, 1], vec![0.4, -0.3], 0.1, 2);
    f.add_unit(vec![2], vec![0.6], 0.0, 3);
    for _ in 0..50 {
        f.learn_step(&[(0, 0.5), (1, -0.2)], 3, 0.3, 0.1, 64);
    }
    f
}

#[test]
fn a_snapshot_predating_the_trait_fields_loads_and_runs() {
    let mut f = grown();
    let after = f.stimulate(&[(0, 0.7), (1, 0.1)], 64);
    let _ = after;

    // Serialize, then STRIP the heritable-trait keys to simulate a snapshot written before they existed.
    let mut json: serde_json::Value = serde_json::to_value(&f).expect("serialize");
    let obj = json.as_object_mut().expect("field is a map");
    for k in ["scale", "lr", "mut_rate", "credit"] {
        assert!(obj.remove(k).is_some(), "expected key {k} in the snapshot");
    }
    // Confirm the legacy shape: units present, trait vectors absent.
    assert!(obj.get("units").and_then(|u| u.as_array()).map(|a| a.len()) == Some(2));

    // Deserialize the legacy snapshot and USE it — this is the exact path that used to panic.
    let mut loaded: Field = serde_json::from_value(json).expect("deserialize legacy snapshot");
    let ev = loaded.stimulate(&[(0, 0.7), (1, 0.1)], 64); // was: index out of bounds on scale[u]
    loaded.learn_step(&[(0, 0.3), (1, 0.9)], 3, -0.2, 0.1, 64); // credit[fj] path
    loaded.metabolize_contribution(&[2, 3], 0.5, 0.01); // energy/credit path
    loaded.reproduce(1, 0); // scale/lr/mut_rate path
    let culled = loaded.cull_after(0.0, 0); // scale/lr/mut_rate/credit rebuild path
    let _ = (ev, culled);

    // The heritable accessors must also be safe on a just-loaded organism (they are `&self`, cannot hydrate).
    let mut fresh: Field = serde_json::from_value({
        let mut j = serde_json::to_value(grown()).unwrap();
        let o = j.as_object_mut().unwrap();
        for k in ["scale", "lr", "mut_rate", "credit"] {
            o.remove(k);
        }
        j
    })
    .unwrap();
    let s0 = fresh.scale_of(2); // &self read on an unhydrated snapshot — returns the default, no panic
    let c = fresh.credit_of(2);
    let l = fresh.lr_of(2);
    let m = fresh.mut_rate_of(2);
    fresh.set_scale(2, 1.3);
    fresh.set_lr(2, 0.2);
    let s1 = fresh.scale_of(2); // after set

    eprintln!("[SNAPSHOT] legacy snapshot (no trait fields) loaded + ran: scale_of {s0:.2}→{s1:.2} credit_of {c:.2} lr_of {l:.2} mut_rate_of {m:.2}; stimulate/learn/metabolize/reproduce/cull all safe.");
    eprintln!("[SNAPSHOT] VERDICT: hydrate_traits self-heals the serde(default) heritable-trait vectors on load, so an organism serialized before those fields existed can be loaded and used without an out-of-bounds panic. A grown organism is the persistent entity — it must survive the format evolving underneath it.");

    assert_eq!(
        s0, 1.0,
        "unhydrated read accessor returns the documented default (no panic)"
    );
    assert_eq!(s1, 1.3, "set_scale after load must take effect");
    assert!((0.0..=8.0).contains(&l), "lr accessor returns a sane value");
}
