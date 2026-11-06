// SPDX-License-Identifier: MIT-0
//! RETIRED FIELDS STILL LOAD — a JSON snapshot written before a dead or validated-NEGATIVE piece of state was
//! deleted must still load, and load to EXACTLY the organism it would be today. The retired state:
//!
//! - the mass/gravity cluster (validated NEGATIVE — gravity neutral/hurts, does not transfer): the per-cell `mass`
//!   array on `Field`, and the `Config` levers `mass_retain`, `gravity_softening`, `gravity_conceptmap`,
//!   `gravity_conceptmap_rate`;
//! - the dead `Config::backbone_width` (never read by the substrate);
//! - the removed pre-1.6 acting rule: `"act_rule": "ActScore"` (it loads as `Valence`, through a serde alias) and its
//!   four `Config` levers `hypothesis_loop`, `openmindedness`, `drive_decay`, `metabolic_actor`;
//! - the host/UI metadata a `Port` carried (`kind`, `unit`, `range`, `rate`, `desc`, `viz`), serialized inside every
//!   `Binding` of the membrane.
//!
//! None of `Field`, `Config`, `Membrane`, `Binding`, `Port`, `Unit` or `Nested` denies unknown fields, so serde skips
//! the retired keys. Checked at every level a snapshot nests them (the host, and a nested sub-morph planted as a
//! cell): the load succeeds, re-serializes to the SAME bytes as the snapshot without the retired keys, and the
//! loaded organism then lives the same life (`.to_bits()`-identical cells) — a stale `gravity_conceptmap: true` or
//! `metabolic_actor: true` turns nothing back on, and a stale `"ActScore"` re-serializes as the `"Valence"` it
//! runs. (JSON only: the bincode `.morph` wire format is positional, so ANY removed field breaks an older `.morph`;
//! that break is documented for each such release.)
//!
//!   cargo test -p openmorph-core --release --test proofs substrate_retired_fields -- --nocapture

use openmorph_core::substrate::Field;
use openmorph_core::{Config, Port};
use serde_json::{json, Value};

/// A trained organism whose snapshot holds every place a retired key used to live: a membrane binding with port
/// descriptors, grown units, and a nested sub-morph (a whole `Field`, with its own `Config`, inside a unit).
fn organism() -> Field {
    let mut f = Field::new(0);
    f.rng_seed(5);
    f.attach(&[Port::input("x", 2).with_shape(2, 1), Port::output("y", 1)]);
    for t in 0..60usize {
        let x = if t.is_multiple_of(2) { 1.0 } else { -1.0 };
        let _ = f.grow(&[(0, x), (1, -x)], 2, 0.6 * x, 0.1, 100);
    }
    // a leaf sub-morph s(x, y) = tanh(x − y), planted as host cell `w`
    let mut sub = Field::new(3);
    sub.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);
    let w = f.len();
    f.nest(sub, &[0, 1], w, 20);
    f
}

/// Put the retired `Field`/`Config` keys back into one serialized `Field` object, as an older build wrote them.
fn add_retired_field_keys(field: &mut Value) {
    let n = field["mem"]["medium"].as_array().expect("medium").len();
    field["mass"] = json!(vec![0.25_f32; n]);
    let cfg = field["config"].as_object_mut().expect("config");
    cfg.insert("mass_retain".into(), json!(0.99));
    cfg.insert("gravity_softening".into(), json!(1.0));
    cfg.insert("gravity_conceptmap".into(), json!(true));
    cfg.insert("gravity_conceptmap_rate".into(), json!(0.02));
    cfg.insert("backbone_width".into(), json!(16));
    // the removed ActScore rule, by name, and its four levers (at non-default values)
    cfg.insert("act_rule".into(), json!("ActScore"));
    cfg.insert("hypothesis_loop".into(), json!(false));
    cfg.insert("openmindedness".into(), json!(0.4));
    cfg.insert("drive_decay".into(), json!(0.0));
    cfg.insert("metabolic_actor".into(), json!(true));
}

/// The retired `Port` host metadata, as an older build serialized it.
fn add_retired_port_keys(port: &mut Value) {
    let p = port.as_object_mut().expect("port");
    p.insert("kind".into(), json!("Image"));
    p.insert("unit".into(), json!("px"));
    p.insert("range".into(), json!([0.0, 1.0]));
    p.insert("rate".into(), json!(30.0));
    p.insert("desc".into(), json!("a camera"));
    p.insert("viz".into(), json!("Frame"));
}

fn bits(f: &Field) -> Vec<u32> {
    f.cells().iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_snapshot_with_retired_fields_loads_to_the_same_organism() {
    let mut original = organism();
    let today = serde_json::to_string(&original).expect("Field serializes");

    // Re-create the snapshot an older build would have written: the retired keys at every level.
    let mut old: Value = serde_json::from_str(&today).unwrap();
    add_retired_field_keys(&mut old);
    let ports = old["mem"]["bindings"][0]["ports"]
        .as_array_mut()
        .expect("binding ports");
    assert_eq!(ports.len(), 2, "setup: the binding carries its two ports");
    for port in ports.iter_mut() {
        add_retired_port_keys(port);
    }
    let nested_sub = old["units"]
        .as_array_mut()
        .expect("units")
        .iter_mut()
        .find_map(|u| u.get_mut("nested").and_then(|n| n.get_mut("sub")))
        .expect("setup: one unit holds a nested sub-morph");
    add_retired_field_keys(nested_sub);
    let old_json = serde_json::to_string(&old).unwrap();
    for key in [
        "\"mass\"",
        "\"mass_retain\"",
        "\"gravity_conceptmap\"",
        "\"backbone_width\"",
        "\"viz\"",
        "\"ActScore\"",
        "\"metabolic_actor\"",
    ] {
        assert!(
            old_json.contains(key),
            "setup: the old snapshot carries {key}"
        );
    }

    // It loads, and it IS today's organism: the retired keys are dropped and nothing else moves.
    let mut loaded: Field =
        serde_json::from_str(&old_json).expect("an old snapshot with retired fields must load");
    let reserialized = serde_json::to_string(&loaded).unwrap();
    assert_eq!(
        reserialized, today,
        "loading drops exactly the retired keys — the snapshot re-serializes byte-identical"
    );

    // …and lives the same life: a stale `gravity_conceptmap: true` switches nothing back on.
    for t in 0..20usize {
        let x = if t.is_multiple_of(3) { 0.5 } else { -0.75 };
        let a = original.grow(&[(0, x), (1, -x)], 2, 0.6 * x, 0.1, 100);
        let b = loaded.grow(&[(0, x), (1, -x)], 2, 0.6 * x, 0.1, 100);
        assert_eq!((a.0.to_bits(), a.1), (b.0.to_bits(), b.1), "step {t}: grow");
        original.co_activate(&[0, 1, 2], 0.1); // the map update the gravity hook used to ride on
        loaded.co_activate(&[0, 1, 2], 0.1);
        assert_eq!(bits(&original), bits(&loaded), "step {t}: cells");
    }
    assert_eq!(
        serde_json::to_string(&original).unwrap(),
        serde_json::to_string(&loaded).unwrap(),
        "the loaded organism continues byte-identically"
    );
    eprintln!(
        "[RETIRED] old snapshot ({} bytes, with mass/gravity/backbone_width/ActScore/port metadata at host + nested level) \
         loads to today's organism ({} bytes) and continues bit-identically for 20 steps",
        old_json.len(),
        today.len()
    );
}

#[test]
fn a_config_with_retired_levers_loads_its_live_levers() {
    let old = r#"{"max_hidden": 7, "mass_retain": 0.9, "gravity_softening": 2.0, "gravity_conceptmap": true,
                  "gravity_conceptmap_rate": 0.1, "backbone_width": 16, "feature_lr": 0.02,
                  "act_rule": "ActScore", "hypothesis_loop": false, "openmindedness": 0.4, "drive_decay": 0.0,
                  "metabolic_actor": true}"#;
    let cfg: Config =
        serde_json::from_str(old).expect("an old config with retired levers must load");
    let want = Config {
        max_hidden: 7,
        feature_lr: 0.02,
        ..Config::default()
    };
    assert_eq!(
        cfg, want,
        "the live levers load, the retired ones are dropped"
    );
    assert_eq!(
        cfg.act_rule,
        openmorph_core::ActRule::Valence,
        "the removed ActScore rule loads as Valence"
    );
    for retired in [
        "mass_retain",
        "gravity_softening",
        "gravity_conceptmap_rate",
        "backbone_width",
        "openmindedness",
        "drive_decay",
    ] {
        assert!(
            !Config::schema().iter().any(|l| l.name == retired),
            "{retired} is retired — it must not be offered to a tuner"
        );
    }
}
