// SPDX-License-Identifier: MIT-0
//! rc6 step 6 (part 1) — GROWN TRANSDUCTION on the async substrate, AUTONOMOUS (#128). The morph starts
//! with ZERO units and, presented with a raw channel it has never seen, GROWS its own transducer where it
//! fails to predict (surprise-birth) and tunes it per-event — no authored encoder, no pre-placed driver, no
//! epoch. This is the moat: plug in any raw channel, the organism grows the driver itself.
//!
//! And it needs NO capacity dial. The one `grow` verb births LIBERALLY — on the training transient (before
//! the writer converges) it grows far more hidden features than a linear-in-tanh channel needs. The LIVING
//! POOL then sizes itself: the reliance economy (`metabolize_reliance` over the grown features + `cull`) pays
//! a feature income ∝ how much the readout leans on it, so the ones the answer ignores starve and are culled
//! — the field self-sizes to exactly the writers the task relies on, held-out error preserved. Birth is
//! variation, the economy is selection; `max_hidden` stops being a hand-set dial. `grow` only, no knob.
//!
//! PINNED to `reliance_norm` at its `Config::legacy()` value (`RelianceNorm::Absolute`) — every other lever is the 1.6
//! default (the 1.6 learning flip): both proofs assert the ABSOLUTE reliance economy's self-sizing (a liberal peak —
//! 195 here — sculpted back to 3 writers; a one-unit skill). Under the full 1.6 default (the mean-relative economy) the
//! pool keeps 7 writers for the 3-cell channel and 11 units for the one-unit skill: a pool whose features the readout
//! ignores EQUALLY earns the mean each, so the mean-relative economy does not prune it to the minimum.
//!
//!   cargo test -p openmorph-core --test proofs substrate_grow -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const NX: usize = 4; // X space: cells 0..4
const NY: usize = 3; // Y space: cells 4..7

// The unknown relationship between the two spaces (three linear maps → tanh).
const W: [[f32; NX]; NY] = [
    [0.7, -0.4, 0.2, 0.5],
    [0.3, 0.6, -0.5, 0.1],
    [-0.2, 0.4, 0.7, -0.3],
];
fn truth_y(x: &[f32; NX], j: usize) -> f32 {
    (0..NX).map(|k| W[j][k] * x[k]).sum::<f32>().tanh()
}

/// The grown HIDDEN-feature cells (written by a unit, beyond the raw X + target Y band). These are the
/// prunable pool — the writers (writing the NX..NX+NY targets) are deliberately excluded so they survive.
fn grown_features(f: &Field) -> Vec<usize> {
    let mut hid: Vec<usize> = f
        .units
        .iter()
        .map(|u| u.write)
        .filter(|&c| c >= NX + NY)
        .collect();
    hid.sort_unstable();
    hid.dedup();
    hid
}

#[test]
fn the_morph_grows_its_own_transducer_for_an_unseen_channel() {
    let mut f = pinned_field(NX + NY);
    // NO capacity dial — DEFAULT config. This channel is linear-in-tanh (a writer per Y cell solves it),
    // but `grow` births hidden features LIBERALLY on the transient. The reliance economy sizes the pool back.
    assert_eq!(
        f.n_units(),
        0,
        "the morph starts with NO units — nothing pre-placed"
    );

    let mut rng = Rng::new(0x9E0);
    let mut peak_units = 0usize;
    let mut first_err = 0.0f32;
    let mut last_err = 0.0f32;
    let (mut acc, mut win, mut nwin) = (0.0f32, 0.0f32, 0.0f32);
    for t in 0..3000 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..NX).map(|k| (k, x[k])).collect();
        for j in 0..NY {
            let (e, _born) = f.grow(&inputs, NX + j, truth_y(&x, j), 0.1, 100);
            acc += e;
        }
        // The LIVING POOL sizes itself: pay the grown FEATURES income ∝ reliance, cull the starved. Writers
        // (writing NX..NX+NY) are never metered here, so they are never culled — only the excess basis is.
        let feats = grown_features(&f);
        peak_units = peak_units.max(f.n_units());
        if t >= 300 {
            f.metabolize_reliance(&feats, 2.0, 1.0);
            f.cull(0.5);
        }
        win += acc;
        acc = 0.0;
        nwin += 1.0;
        if (t + 1) % 600 == 0 {
            let m = win / (nwin * NY as f32);
            if t < 600 {
                first_err = m;
            }
            last_err = m;
            win = 0.0;
            nwin = 0.0;
        }
    }

    // Birth was liberal (the pool over-grew on the transient) but the economy SIZED it back to exactly the
    // writers the linear channel relies on — no hand-set `max_hidden`.
    assert!(
        peak_units > NY,
        "grow births liberally on the transient (peak {peak_units} should exceed the {NY} writers)"
    );
    assert_eq!(
        f.n_units(),
        NY,
        "the reliance economy self-sized the pool to one writer per channel cell (got {})",
        f.n_units()
    );

    // Held-out: the GROWN transducer predicts the whole Y channel from unseen X.
    let mut rng = Rng::new(0xC0DE);
    let mut mae = 0.0f32;
    for _ in 0..500 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..NX).map(|k| (k, x[k])).collect();
        f.stimulate(&inputs, 100);
        for j in 0..NY {
            mae += (f.cells()[NX + j] - truth_y(&x, j)).abs();
        }
    }
    mae /= 500.0 * NY as f32;

    eprintln!(
        "[GROW] started 0 units → born LIBERALLY (peak {peak_units}) → reliance economy self-sized to {} writers; mean|err| {:.3} → {:.3}; held-out MAE {:.3}",
        f.n_units(), first_err, last_err, mae
    );
    eprintln!("[GROW] VERDICT (rc6 step 6 / #128): plugged a raw channel the morph had never seen; it GREW its own transducer where it could not predict (surprise-birth) and tuned it per-event, from ZERO units — no authored encoder, no tick, and NO capacity dial. `grow` births liberally (peak {peak_units}); the LIVING POOL sizes itself by the reliance economy (`metabolize_reliance` + `cull`) back to exactly the {NY} writers the channel relies on, held-out MAE preserved. Birth is variation, the economy is selection; `max_hidden` is retired. The morph grows AND sizes its own driver — the moat, on the async membrane.");
    assert!(
        last_err < first_err * 0.5,
        "error falls as the transducer grows + tunes ({first_err:.3} → {last_err:.3})"
    );
    assert!(
        mae < 0.05,
        "the grown transducer generalizes to unseen X (MAE {mae:.3})"
    );
}

#[test]
fn a_grown_skill_grafts_into_another_morph_instantly() {
    // Morph A grows a transducer for the X→Y0 relationship — DEFAULT config, no capacity dial. It births
    // liberally, and the reliance economy sizes the pool back to the single writer the skill relies on, so
    // what gets banked is a CLEAN minimal skill (not a bag of transient random features).
    let mut a = pinned_field(NX + 1);
    let mut rng = Rng::new(0x11);
    for t in 0..2000 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..NX).map(|k| (k, x[k])).collect();
        a.grow(&inputs, NX, truth_y(&x, 0), 0.1, 100);
        // Prune the excess basis: features written beyond the NX+1 raw+target band, metered by reliance.
        let feats: Vec<usize> = a
            .units
            .iter()
            .map(|u| u.write)
            .filter(|&c| c > NX)
            .collect();
        if t >= 300 {
            a.metabolize_reliance(&feats, 2.0, 1.0);
            a.cull(0.5);
        }
    }
    let sub = a.encapsulate(&[NX]); // bank the grown transducer for cell NX as a self-contained sub-morph
    assert_eq!(
        sub.n_units(),
        1,
        "the pool self-sized to one grown unit — a clean minimal skill"
    );

    // Morph B is BLANK — nest the skill as ONE encapsulated cell; it predicts immediately, with NO training
    // of its own. The nested cell reads the same input space (0..NX) and writes NX, settled with budget 100.
    let mut b = pinned_field(NX + 1);
    assert_eq!(b.n_units(), 0);
    b.nest(sub, &(0..NX).collect::<Vec<_>>(), NX, 100);
    assert_eq!(b.n_units(), 1, "B nested the skill");

    let mut rng = Rng::new(0xFEED);
    let (mut mae_b, mut mae_blank) = (0.0f32, 0.0f32);
    let mut blank = pinned_field(NX + 1); // an ungrafted control
    for _ in 0..500 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let inputs: Vec<(usize, f32)> = (0..NX).map(|k| (k, x[k])).collect();
        b.stimulate(&inputs, 100);
        blank.stimulate(&inputs, 100);
        mae_b += (b.cells()[NX] - truth_y(&x, 0)).abs();
        mae_blank += (blank.cells()[NX] - truth_y(&x, 0)).abs();
    }
    mae_b /= 500.0;
    mae_blank /= 500.0;
    eprintln!("[GROW] skill transfer: nested morph MAE {mae_b:.3} vs blank control MAE {mae_blank:.3} — instant competence, no retraining.");
    assert!(
        mae_b < 0.05,
        "the nested skill works immediately (MAE {mae_b:.3})"
    );
    assert!(
        mae_blank > 0.2,
        "the blank control has no skill (MAE {mae_blank:.3})"
    );
}

/// The genome this proof's claim is about (see the module docs): the 1.6 default with `reliance_norm` at the
/// [`Config::legacy`](openmorph_core::Config::legacy) value.
fn pinned_config() -> openmorph_core::Config {
    let legacy = openmorph_core::Config::legacy();
    openmorph_core::Config {
        reliance_norm: legacy.reliance_norm, // the ABSOLUTE reliance economy
        ..openmorph_core::Config::default()
    }
}

/// A `Field` of `n` cells on [`pinned_config`].
fn pinned_field(n: usize) -> Field {
    let mut f = Field::new(n);
    f.configure(pinned_config());
    f
}
