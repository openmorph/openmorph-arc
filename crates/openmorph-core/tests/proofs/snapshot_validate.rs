// SPDX-License-Identifier: MIT-0
//! WAVE-2 defensive hardening (security review C1/PY1): `Field::validate()` for UNTRUSTED snapshots.
//!
//! `serde` reconstructs a `Field` from raw bytes WITHOUT enforcing the structural invariants the
//! settle/learn paths assume. A crafted snapshot (a unit reading a non-existent cell, a
//! length-mismatched parallel array, a hostile adjacency unit id) is memory-SAFE — Rust bounds-checks
//! — but PANICS on a later call. `validate()` is the opt-in, up-front check that turns that latent
//! panic into a descriptive rejection, WITHOUT touching the (byte-identical) deserialization path.

use openmorph_core::{Field, OpenMorphError};
use serde_json::Value;

/// A normally-grown organism: a small living pool over two inputs predicting a target cell.
fn grown() -> Field {
    let mut f = Field::new(4);
    for i in 0..60 {
        let x = (i as f32 * 0.37).sin();
        f.grow(&[(0, x), (1, -0.5 * x)], 3, 0.6 * x, 0.1, 32);
    }
    // exercise the event-driven side channels so the field carries traces/delays too.
    let t = f.integrate(3, 8.0);
    let _ = f.delay(0);
    assert!(
        t < f.len(),
        "sanity: integrate returned an in-range cell id"
    );
    f
}

#[test]
fn normal_field_validates() {
    let f = grown();
    assert!(!f.units.is_empty(), "the grown field should have units");
    assert!(
        f.validate().is_ok(),
        "a normally-grown field must pass validate()"
    );

    // A clean JSON round-trip (the honest snapshot path) also validates.
    let json = serde_json::to_string(&f).unwrap();
    let back: Field = serde_json::from_str(&json).unwrap();
    assert!(
        back.validate().is_ok(),
        "a clean round-trip must still validate"
    );
}

/// A unit reading a cell that does not exist. Memory-SAFE (compute is `.get()`-guarded ⇒ silently
/// wrong, no panic), but a corrupt organism — `validate()` rejects it up front.
#[test]
fn hostile_reads_index_rejected() {
    let f = grown();
    let n = f.len();
    let mut v: Value = serde_json::to_value(&f).unwrap();
    v["units"][0]["reads"][0] = Value::from(n + 5);
    let hostile: Field = serde_json::from_value(v).unwrap();

    let err = hostile.validate().unwrap_err();
    assert!(matches!(err, OpenMorphError::InvalidSnapshot(_)));
    assert!(
        format!("{err}").contains("reads"),
        "error should name the violated invariant: {err}"
    );
}

/// A length-mismatched parallel array (a unit's `weights` shorter than its `reads`). `validate()`
/// rejects it (the `compute` zip would otherwise silently drop the extra reads).
#[test]
fn hostile_length_mismatch_rejected() {
    let f = grown();
    let mut v: Value = serde_json::to_value(&f).unwrap();
    // Drop the last weight of unit 0 so weights.len() != reads.len().
    let w = v["units"][0]["weights"].as_array_mut().unwrap();
    assert!(w.len() >= 2, "need a multi-input unit for this case");
    w.pop();
    let hostile: Field = serde_json::from_value(v).unwrap();

    let err = hostile.validate().unwrap_err();
    assert!(matches!(err, OpenMorphError::InvalidSnapshot(_)));
    assert!(
        format!("{err}").contains("length"),
        "error should mention the length mismatch: {err}"
    );
}

/// A hostile adjacency bucket referencing a non-existent unit. This is the case that PANICS on use:
/// `stimulate` enqueues the readers of a changed cell, and an out-of-range unit id indexes `queued`
/// out of bounds. `validate()` rejects it; we also CONFIRM the un-validated field would panic — the
/// exact robustness/DoS gap this method closes.
#[test]
fn hostile_adjacency_unit_id_rejected_and_would_panic() {
    let f = grown();
    let nu = f.units.len();
    let mut v: Value = serde_json::to_value(&f).unwrap();

    // Inject a bogus unit id into cell 0's readers bucket (cell 0 is a live input, so it fires on
    // stimulate and the bogus id is dereferenced).
    let readers0 = v["readers"][0].as_array_mut().expect("readers[0] bucket");
    assert!(
        !readers0.is_empty(),
        "cell 0 should have readers in a grown field"
    );
    readers0.push(Value::from(nu + 3));
    let hostile: Field = serde_json::from_value(v).unwrap();

    // validate() catches it.
    let err = hostile.validate().unwrap_err();
    assert!(matches!(err, OpenMorphError::InvalidSnapshot(_)));
    assert!(
        format!("{err}").contains("units.len()"),
        "error should name the bad adjacency: {err}"
    );

    // And the SAME field, if a careless caller had skipped validate(), PANICS on the first stimulate.
    let mut victim = hostile.clone();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the expected panic's backtrace
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        victim.stimulate(&[(0, 0.9)], 64);
    }))
    .is_err();
    std::panic::set_hook(prev);
    assert!(
        panicked,
        "the un-validated hostile field must panic on stimulate (the gap validate closes)"
    );
}

/// A hostile trace writing a cell past the medium. `tick_traces` writes the trace cell directly
/// (`self.mem.medium[cell] = ..`), so this panics on `stimulate`; `validate()` rejects it.
#[test]
fn hostile_trace_cell_rejected() {
    let f = grown();
    let n = f.len();
    let mut v: Value = serde_json::to_value(&f).unwrap();
    let traces = v["traces"].as_array_mut().expect("traces present");
    assert!(!traces.is_empty(), "grown() registers an integrate() trace");
    traces[0]["cell"] = Value::from(n + 9);
    let hostile: Field = serde_json::from_value(v).unwrap();

    let err = hostile.validate().unwrap_err();
    assert!(matches!(err, OpenMorphError::InvalidSnapshot(_)));
    assert!(
        format!("{err}").contains("trace"),
        "error should name the trace: {err}"
    );
}

/// The pending-refire set (`dirty`, serialized since the staleness refresh): a crafted one — a flag past the last unit,
/// a listed unit outside the flags, or a count that disagrees with the flags — passed `validate()` and then panicked
/// in `drain` (review of #494, core #3). `validate()` now rejects each.
#[test]
fn hostile_pending_refire_set_rejected() {
    let f = grown();
    let nu = f.units.len();
    let cases = [
        // a flag for a unit that does not exist
        serde_json::json!({"flag": vec![false; nu + 1].iter().enumerate().map(|(i, _)| i == nu).collect::<Vec<bool>>(), "list": [nu], "n": 1}),
        // a listed unit outside the flags
        serde_json::json!({"flag": [true], "list": [0, nu + 3], "n": 1}),
        // a count that disagrees with the flags (would underflow on the next clear)
        serde_json::json!({"flag": [false], "list": [], "n": 1}),
        // a set flag missing from the list
        serde_json::json!({"flag": [true], "list": [], "n": 1}),
    ];
    for (i, dirty) in cases.into_iter().enumerate() {
        let mut v: Value = serde_json::to_value(&f).unwrap();
        v["dirty"] = dirty;
        let hostile: Field = serde_json::from_value(v).unwrap();
        let err = hostile.validate().unwrap_err();
        assert!(
            matches!(err, OpenMorphError::InvalidSnapshot(_)),
            "case {i}: {err:?}"
        );
    }
    // an honest pending set still validates
    let mut v: Value = serde_json::to_value(&f).unwrap();
    v["dirty"] = serde_json::json!({"flag": [true], "list": [0], "n": 1});
    let ok: Field = serde_json::from_value(v).unwrap();
    assert!(ok.validate().is_ok(), "{:?}", ok.validate());
}
