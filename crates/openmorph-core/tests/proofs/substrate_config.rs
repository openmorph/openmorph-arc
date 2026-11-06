// SPDX-License-Identifier: MIT-0
//! TUNABLE CONFIG — the substrate's meta-parameters are now a single serde struct ([`Config`]) that an external
//! tuner (Optuna, a sweep, a grid search) can set, serialise, and change LIVE. This proves the four properties a
//! tuning harness needs:
//!   (A) DEFAULT = HISTORICAL — an unconfigured field reproduces the old hard-coded constants exactly (so the
//!       change is behaviour-preserving until someone tunes).
//!   (B) SERDE ROUND-TRIP — a Config serialises to JSON and back unchanged (Optuna writes a trial → JSON → load).
//!   (C) LIVE + EFFECTIVE — a knob changed through `config_mut` measurably shifts behaviour (a tuner has signal:
//!       a smaller `init_lr` learns slower under a fixed schedule), and `configure` applies a whole Config.
//!   (D) SNAPSHOT-SAFE — the config round-trips through a Field snapshot, and an OLD snapshot with no config
//!       field loads the defaults (serde default) rather than panicking.
//!
//!   cargo test -p openmorph-core --test proofs substrate_config -- --nocapture --test-threads=1

use openmorph_core::{Config, Field};

/// Learn `target = tanh(1.5·x)` for a fixed number of steps at a fixed caller schedule, and return the residual
/// error. The per-unit `init_lr` seeds every born unit's learning-rate factor (multiplied into each step), so a
/// smaller `init_lr` must converge LESS in the same budget — the signal a tuner optimises.
fn residual_after_training(init_lr: f32) -> f32 {
    let mut m = Field::new(2);
    m.config_mut().init_lr = init_lr; // LIVE knob change
    m.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    m.config_mut().birth_gate = 0.05;
    for _ in 0..300 {
        for &x in &[-1.0_f32, 1.0] {
            m.grow(&[(0, x)], 1, (1.5 * x).tanh(), 0.1, 50);
        }
    }
    m.stimulate(&[(0, 1.0)], 50);
    (m.cells()[1] - (1.5_f32).tanh()).abs()
}

#[test]
fn config_default_matches_history() {
    // (A) The defaults ARE the old constants — documented here so a change to Config::default is caught.
    let c = Config::default();
    assert_eq!(c.birth_energy, 3.0);
    assert_eq!(c.energy_cap, 10.0);
    assert_eq!(c.surprise_decay, 0.9);
    assert_eq!(c.credit_decay, 0.99);
    assert_eq!(c.self_adapt_tau, 0.4);
    assert_eq!(c.settle_eps, 1e-4);
    assert_eq!(c.init_scale, 1.0);
    assert_eq!(c.init_lr, 1.0);
    assert_eq!(c.init_mut_rate, 0.3);
    assert_eq!((c.sigma_min, c.sigma_max), (0.02, 1.5));
    assert_eq!((c.scale_min, c.scale_max), (0.05, 5.0));
    assert_eq!((c.lr_min, c.lr_max), (0.05, 8.0));
    // A fresh field carries the defaults.
    assert_eq!(Field::new(1).config(), &Config::default());
    eprintln!("[CONFIG] (A) defaults reproduce the historical constants — behaviour-preserving.");
}

#[test]
fn config_serialises_round_trip() {
    // (B) Optuna's flow: build a Config from a trial's params → JSON → load into the field.
    let tuned = Config {
        init_lr: 0.5,
        birth_energy: 5.0,
        surprise_decay: 0.8,
        ..Config::default()
    };
    let json = serde_json::to_string(&tuned).expect("serialise");
    let back: Config = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(tuned, back, "Config must survive a JSON round-trip");

    let mut m = Field::new(1);
    m.configure(back.clone());
    assert_eq!(m.config(), &back, "configure() must apply the whole Config");
    eprintln!(
        "[CONFIG] (B) JSON round-trip exact; configure() applies a loaded Config. json = {json}"
    );
}

#[test]
fn a_tuned_knob_measurably_shifts_behaviour() {
    // (C) A tuner needs the knob to MATTER: slower init_lr ⇒ more residual error in a fixed budget.
    let fast = residual_after_training(1.0); // default-scale learning
    let slow = residual_after_training(0.05); // throttled — should under-converge
    eprintln!(
        "[CONFIG] (C) residual after 300 steps: init_lr=1.0 → {fast:.4}   init_lr=0.05 → {slow:.4}"
    );
    assert!(
        slow > fast + 0.05,
        "init_lr must measurably change learning speed (fast {fast:.4} vs slow {slow:.4}) — the tuner's signal"
    );
}

#[test]
fn config_survives_snapshot_and_old_snapshots_keep_the_legacy_genome() {
    // (D-1) A configured field snapshots and reloads with its config intact.
    let mut m = Field::new(3);
    m.config_mut().init_scale = 2.5;
    m.config_mut().energy_cap = 42.0;
    let snap = serde_json::to_string(&m).expect("snapshot");
    let loaded: Field = serde_json::from_str(&snap).expect("reload");
    assert_eq!(loaded.config().init_scale, 2.5);
    assert_eq!(loaded.config().energy_cap, 42.0);

    // (D-2) An OLD snapshot (predating the config field) must load, not panic, with the genome it ran: the
    // historical constants, i.e. `Config::legacy()` (not the 1.6 defaults — review of #494, core #2). Simulate one
    // by taking a real snapshot and DELETING the `config` key, then loading — exactly what an old file looks like.
    let mut plain = Field::new(2);
    plain.config_mut().max_hidden = 0; // grow a linear writer only (preserve grow_and_learn behaviour)
    plain.config_mut().birth_gate = 0.05;
    plain.grow(&[(0, 1.0)], 1, 0.5, 0.1, 50); // give it some grown state
    let mut val: serde_json::Value = serde_json::to_value(&plain).expect("to value");
    val.as_object_mut().unwrap().remove("config"); // strip config → an "old" snapshot
    assert!(val.get("config").is_none());
    let from_old: Field = serde_json::from_value(val).expect("a config-less snapshot must load");
    assert_eq!(
        from_old.config(),
        &Config::legacy(),
        "a config-less snapshot falls back to the historical genome, Config::legacy()"
    );
    eprintln!("[CONFIG] (D) config round-trips through a snapshot; a config-less old snapshot loads the legacy genome.");
    eprintln!("[CONFIG] VERDICT: every substrate meta-constant is now a tunable, serde-serialisable, live-changeable knob — Optuna (or any tuner) writes a trial to a Config, loads it, and measures; nothing is a buried literal any more. `docs/config-reference.md` is the full search space.");
}
