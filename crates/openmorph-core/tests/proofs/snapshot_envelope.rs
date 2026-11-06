// SPDX-License-Identifier: MIT-0
//! THE VERSIONED SNAPSHOT ENVELOPE — an organism saved by one release loads in the next. `Field::snapshot` writes
//! `b"OMSN"` + a little-endian `u16` format + a JSON payload; `Field::load` reads it, reads the bare JSON every
//! earlier release wrote, fills fields an older snapshot lacks from their `#[serde(default)]` (and absent 1.6 genome
//! keys from `Config::legacy()`), and REFUSES a newer format, a non-snapshot, a truncated or a structurally corrupt
//! one with an error — never a panic. (Byte-identical continuation through the envelope is proven alongside JSON
//! and bincode in `substrate_snapshot_continue`.)
//!
//!   cargo test -p openmorph-core --test proofs snapshot_envelope

use openmorph_core::snapshot::{self, FORMAT, MAGIC};
use openmorph_core::{Config, Field, OpenMorphError};
use serde_json::Value;

/// A grown organism with nested/gated structure absent, so every unit carries the optional keys as `null`.
fn grown() -> Field {
    let mut f = Field::new(4);
    f.rng_seed(11);
    for i in 0..80 {
        let x = (i as f32 * 0.37).sin();
        f.grow(&[(0, x), (1, -0.5 * x)], 3, 0.6 * x, 0.1, 64);
    }
    f
}

#[test]
fn header_is_magic_format_then_json() {
    let f = grown();
    let s = f.snapshot();
    assert_eq!(
        &s[..6],
        b"OMSN\x01\x00",
        "the header is pinned: a FORMAT bump must be deliberate"
    );
    assert_eq!((MAGIC, FORMAT), (*b"OMSN", 1));
    assert_eq!(snapshot::format_of(&s), Some(FORMAT));
    assert_eq!(
        &s[6..],
        serde_json::to_vec(&f).unwrap().as_slice(),
        "the payload is the organism's JSON"
    );
}

#[test]
fn pre_envelope_json_loads_and_reenvelopes_identically() {
    let f = grown();
    let legacy = serde_json::to_vec(&f).unwrap();
    assert_eq!(snapshot::format_of(&legacy), Some(0));
    let back = Field::load(&legacy).expect("a pre-envelope JSON snapshot loads");
    assert_eq!(
        back.snapshot(),
        f.snapshot(),
        "the legacy load is the same organism, byte for byte"
    );
    // Leading whitespace (a pretty-printed or hand-edited file) is still JSON.
    let mut spaced = b"\n  ".to_vec();
    spaced.extend_from_slice(&legacy);
    assert_eq!(Field::load(&spaced).unwrap().snapshot(), f.snapshot());
}

/// An OLDER release's snapshot: drop keys a newer release added (per-unit optional wiring, 1.6 genome levers). It
/// loads — bare or enveloped — with the defaults filled and the old genome's legacy levers kept.
#[test]
fn older_snapshot_missing_fields_loads_with_defaults() {
    let f = grown();
    let mut v: Value = serde_json::to_value(&f).unwrap();
    for u in v["units"].as_array_mut().unwrap() {
        let u = u.as_object_mut().unwrap();
        for k in ["gate", "factor", "activation", "nested"] {
            assert!(
                u.remove(k).is_some(),
                "the fixture must actually drop `{k}`"
            );
        }
    }
    let cfg = v["config"].as_object_mut().unwrap();
    for k in [
        "stability_cap",
        "fanin_init_gain",
        "reliance_norm",
        "grow_within_budget",
        "act_rule",
    ] {
        assert!(
            cfg.remove(k).is_some(),
            "the fixture must actually drop config `{k}`"
        );
    }
    let bare = serde_json::to_vec(&v).unwrap();
    let mut enveloped = MAGIC.to_vec();
    enveloped.extend_from_slice(&FORMAT.to_le_bytes());
    enveloped.extend_from_slice(&bare);
    let legacy = Config::legacy();
    for bytes in [bare, enveloped] {
        let g = Field::load(&bytes).expect("an older snapshot loads");
        assert_eq!(g.n_units(), f.n_units());
        assert!(g
            .units
            .iter()
            .all(|u| u.gate.is_none() && u.factor.is_none() && u.nested.is_none()));
        let c = g.config();
        assert_eq!(
            c.stability_cap.to_bits(),
            legacy.stability_cap.to_bits(),
            "absent 1.6 lever ⇒ legacy value"
        );
        assert_eq!(c.grow_within_budget, legacy.grow_within_budget);
        assert_eq!(
            c.fanin_init_gain.to_bits(),
            legacy.fanin_init_gain.to_bits()
        );
    }
}

#[test]
fn newer_format_is_refused_clearly() {
    let mut s = grown().snapshot();
    s[4..6].copy_from_slice(&(FORMAT + 1).to_le_bytes());
    match Field::load(&s) {
        Err(OpenMorphError::InvalidSnapshot(m)) => assert!(m.contains("newer"), "{m}"),
        other => panic!("a newer format must be refused, got {other:?}"),
    }
}

#[test]
fn garbage_and_truncation_are_errors_not_panics() {
    let f = grown();
    let s = f.snapshot();
    let bin = bincode::serialize(&f).unwrap();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("short magic", b"OMS".to_vec()),
        ("magic, half a version", b"OMSN\x01".to_vec()),
        ("header only", s[..6].to_vec()),
        ("truncated payload", s[..s.len() / 2].to_vec()),
        (
            "truncated legacy json",
            serde_json::to_vec(&f).unwrap()[..40].to_vec(),
        ),
        ("raw bincode", bin),
        ("binary noise", (0u8..=255).collect()),
    ];
    for (name, bytes) in cases {
        assert!(Field::load(&bytes).is_err(), "{name}: must be an error");
    }
}

/// A crafted snapshot that decodes but breaks an invariant (a unit reading a cell that does not exist) is rejected
/// by `load` itself — it runs `Field::validate`, so the caller never holds an organism that panics later.
#[test]
fn structurally_corrupt_snapshot_is_rejected_on_load() {
    let f = grown();
    let mut v: Value = serde_json::to_value(&f).unwrap();
    v["units"][0]["reads"][0] = Value::from(f.len() + 5);
    let bytes = snapshot::encode(&v).unwrap();
    assert!(matches!(
        Field::load(&bytes),
        Err(OpenMorphError::InvalidSnapshot(_))
    ));
    // …while the generic `decode` (no invariants to know about) still hands the value back.
    assert!(snapshot::decode::<Field>(&bytes).is_ok());
}
