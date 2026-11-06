// SPDX-License-Identifier: MIT-0
//! A RECLAIMED CELL SLOT IS PRISTINE — a feature a `cull`/`merge` retires hands its slot to the free-cell pool, and
//! the next grown feature reuses it (the medium stays bounded at the peak concurrent cell count). The reused slot
//! must carry NOTHING of its previous life: every per-cell array (the slow surprise trace, the plasticity
//! controller, the var-gated read statistics, the surprise-currency moments, drive, dwell, heartbeat pressure,
//! …) and every cell-keyed map entry (value ensemble, valence record, product co-credit, fast-weight binding) reads
//! exactly what a never-used cell reads. A stale entry biases whatever consults it next — `orient` (drive + learning
//! progress), `saturated`, the heartbeat (pressure), `metabolize_surprise` (variance/sensitivity), the dwell actor,
//! the acting rules.
//!
//!   cargo test -p openmorph-core --release --test proofs substrate_pristine_slot -- --nocapture

use openmorph_core::substrate::{Event, Field};
use serde_json::Value;

/// The per-cell arrays of the serialized field, with the value a never-used cell holds in each.
const PRISTINE: &[(&str, f64)] = &[
    ("surprise", 0.0),
    ("surprise_slow", 0.0),
    ("err_fast", 0.0),
    ("err_slow", 0.0),
    ("learn_gain", 1.0),
    ("read_min", 0.0),
    ("read_max", 0.0),
    ("read_count", 0.0),
    ("cell_mean", 0.0),
    ("cell_var", 0.0),
    ("cell_sens2", 0.0),
    ("drive", 0.0),
    ("dwell_len", 1.0),
    ("pressure", 0.0),
];

/// Every serialized per-cell entry at `c` that differs from a never-used cell (empty ⇒ pristine).
fn stale_entries(f: &Field, c: usize) -> Vec<String> {
    let v: Value = serde_json::to_value(f).unwrap();
    let mut stale = Vec::new();
    for &(key, want) in PRISTINE {
        if let Some(x) = v[key].as_array().and_then(|a| a.get(c)) {
            if x.as_f64() != Some(want) {
                stale.push(format!("{key}[{c}]={x}"));
            }
        }
    }
    let medium = &v["mem"]["medium"][c];
    if medium.as_f64() != Some(0.0) {
        stale.push(format!("medium[{c}]={medium}"));
    }
    if v["clamped"][c].as_bool() == Some(true) {
        stale.push(format!("clamped[{c}]"));
    }
    if v["values"][c].as_f64() != Some(f32::MAX as f64) {
        stale.push(format!("values[{c}]={}", v["values"][c]));
    }
    if let Some(r) = v["reliability"].as_array().and_then(|a| a.get(c)) {
        if [&r["fast"], &r["slow"], &r["peak"]]
            .iter()
            .any(|x| x.as_f64() != Some(0.0))
        {
            stale.push(format!("reliability[{c}]={r}"));
        }
    }
    if v["value_ensemble"].get(c.to_string()).is_some() {
        stale.push(format!("value_ensemble[{c}]"));
    }
    if v["valence"].get(c.to_string()).is_some() {
        stale.push(format!("valence[{c}]"));
    }
    stale
}

/// The cell-keyed accessors, read the way the rest of the organism reads them.
fn accessor_view(f: &Field, c: usize, other: usize) -> [f32; 6] {
    [
        f.learning_progress(c),
        f.plasticity_gain_of(c),
        f.drive_of(c),
        f.co_credit_of(other, c),
        f.bound(other, c),
        f.surprise_reduction_of(c),
    ]
}

const PRISTINE_VIEW: [f32; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

/// Give feature cell `h` (read by the readout of cell 2 from inputs 0/1) a full life of per-cell state.
fn live_a_life(f: &mut Field, h: usize) {
    let cfg = f.config_mut();
    cfg.max_hidden = 0; // no more births while the slot is being dirtied
    cfg.plasticity_anneal = 0.05; // err_fast/err_slow/learn_gain on a learned cell
    cfg.lr_fanin_norm = true; // with var_gated_norm: read_min/read_max/read_count on every read cell
    cfg.var_gated_norm = true;
    cfg.surprise_currency = true; // cell_mean/cell_var/cell_sens2
    cfg.heartbeat_theta = 1e9; // `live` only ramps pressure, never reactivates or culls
    for t in 0..40usize {
        let x = if t.is_multiple_of(2) { 1.0 } else { -1.0 };
        let _ = f.grow(&[(0, x), (1, -x)], 2, 0.7 * x, 0.1, 200); // readout 2 reads h (read stats, sensitivity)
        let _ = f.grow(
            &[(0, x), (1, -x)],
            h,
            if t < 5 { 0.5 } else { -0.5 },
            0.1,
            200,
        ); // h itself a target
    }
    f.reinforce(h, 0.7); // drive
    f.accrue_co_credit(&[0, h], 0.5, 0.5); // product co-credit
    f.bind(&[(0, 3.0), (h, 3.0)]); // fast-weight rule
    f.observe_reliability(h, 0.8); // mastery meter
    f.prefer(h, 0.0); // a clamped value
    let _ = f.live(3); // heartbeat pressure on grown cells
    f.config_mut().action_dwell = true; // dwell_len
    let _ = f.alive(Event::Act {
        options: &[h, 0],
        last: Some((h, 1.0)),
    }); // the default valence rule: h's (μ, visits) record
    f.config_mut().action_dwell = false;
    f.config_mut().free_energy_actor = true; // value ensemble
    let _ = f.alive(Event::Act {
        options: &[h, 0],
        last: None,
    });
    f.config_mut().free_energy_actor = false;
}

#[test]
fn a_culled_slot_is_reused_pristine() {
    let mut f = Field::new(3);
    f.rng_seed(11);
    f.config_mut().max_hidden = 1;
    for t in 0..10usize {
        let x = if t.is_multiple_of(2) { 1.0 } else { -1.0 };
        let _ = f.grow(&[(0, x), (1, -x)], 2, 0.7 * x, 0.1, 200);
    }
    let h = f.grown_basis()[0]; // the one grown feature
    live_a_life(&mut f, h);
    let dirty = stale_entries(&f, h);
    let before = accessor_view(&f, h, 0);
    eprintln!("[PRISTINE] feature {h} before its death: {dirty:?}  accessors {before:?}");
    // Setup is meaningful: the slot really carries a life. (Learning progress may read 0 while h's own surprise is
    // still above its slow trace; the trace itself is required dirty through the serialized view below.)
    for (i, (&b, &p)) in before.iter().zip(PRISTINE_VIEW.iter()).enumerate().skip(1) {
        assert_ne!(b, p, "setup: accessor {i} must be dirty before the cull");
    }
    for key in [
        "surprise_slow",
        "read_count",
        "cell_var",
        "drive",
        "dwell_len",
        "pressure",
        "value_ensemble",
        "valence",
    ] {
        assert!(
            dirty.iter().any(|d| d.starts_with(key)),
            "setup: {key} must be dirty before the cull"
        );
    }

    // Starve ONLY h's unit and cull it: h is orphaned, detached from the readout, and its slot reclaimed.
    f.metabolize_reliance(&[h], 0.0, 100.0);
    assert_eq!(f.cull(0.0), 1, "exactly the starved feature dies");
    let after = accessor_view(&f, h, 0);
    let stale = stale_entries(&f, h);
    eprintln!("[PRISTINE] reclaimed slot {h}: stale {stale:?}  accessors {after:?}");
    assert!(
        stale.is_empty(),
        "a reclaimed slot must be pristine — stale: {stale:?}"
    );
    assert_eq!(
        after, PRISTINE_VIEW,
        "a reclaimed slot reads like a never-used cell"
    );

    // Someone touches the free slot, then a new feature is born into it: the allocation resets it again.
    f.reinforce(h, 0.4);
    f.config_mut().max_hidden = 1;
    // Budget 0: birth only, no settle/learn yet. The 1.6 default's `grow_within_budget` refuses a feature no settle
    // can fire (budget 0), so this one step lifts the bound — the claim here is the slot's reuse, not the bound.
    f.config_mut().grow_within_budget = false;
    let _ = f.grow(&[(0, 1.0), (1, -1.0)], 2, -5.0, 0.1, 0);
    assert_eq!(
        f.grown_basis(),
        vec![h],
        "the newborn feature REUSES the freed slot"
    );
    assert_eq!(f.drive_of(h), 0.0, "the newborn inherits no drive");
    assert_eq!(
        f.learning_progress(h),
        0.0,
        "the newborn inherits no learning progress"
    );
}

/// core-debt E4: `merge` frees the dropped cell; its slow surprise trace must not survive into the slot.
#[test]
fn a_merged_away_slot_is_pristine() {
    let mut f = Field::new(4);
    f.config_mut().max_hidden = 0;
    for t in 0..200usize {
        let x = if t.is_multiple_of(2) { 1.0 } else { -1.0 };
        let _ = f.grow(
            &[(0, x)],
            3,
            if t < 100 { 0.9 * x } else { -0.9 * x },
            0.1,
            50,
        );
    }
    let lp_before = f.learning_progress(3);
    f.merge(2, 3); // cell 3 is reset + freed
    let (lp, stale) = (f.learning_progress(3), stale_entries(&f, 3));
    eprintln!("[PRISTINE] merge: learning_progress(3) {lp_before:.4} → {lp:.4}  stale {stale:?}");
    assert!(lp_before > 0.0, "setup: cell 3 had learning progress");
    assert_eq!(lp, 0.0, "a merged-away slot reads no learning progress");
    assert!(
        stale.is_empty(),
        "a merged-away slot must be pristine — stale: {stale:?}"
    );
}
