// SPDX-License-Identifier: MIT-0
//! ECONOMY-DRIVEN EVOLUTION — the heritable-constants loop closed through the ENERGY ACCOUNT, no test-side
//! schedule. `substrate_heritable` tuned `scale` by a caller picking best/worst each round; this removes the
//! caller: the substrate runs `metabolize_contribution` (a unit earns its CONTRIBUTION credit, pays upkeep —
//! the make-or-break fix, since a firing gate would credit a random backbone equally) and `repopulate` (a
//! STARVED unit is reborn as the FITTEST survivor's offspring). Selection, heredity, and variation all flow
//! through energy — decentralized, online, in the one currency. Nothing outside the organism decides who
//! wins; the projection scale tunes itself because the useful-scale features bank energy and the useless
//! ones starve and are overwritten by offspring of the fittest.
//!
//! Demonstrated: 3-parity needs crisp (high-scale) features; the population starts too smooth (scale 0.3).
//! With ONLY `metabolize_contribution` + `repopulate` driving it, the population scale climbs and accuracy
//! rises above a no-economy control — the economy tunes the constant on its own.
//!
//! RE-BASELINED at the 1.6 learning flip — runs on the default genome. The SCALE claim holds: the energy account alone
//! drives the population scale 0.3 → 2.16 on the historical organism, and 0.3 → 1.35 averaged over six DISTINCT
//! organisms (0.80 / 1.26 / 0.41 / 1.14 / 3.51 / 0.96). The ACCURACY claim (`economy_driven_tuning_beats_the_no_economy_control`)
//! still does not hold and stays `#[ignore]`d: the no-economy control reaches 100 % (its readout learns the feature
//! directions, so the too-smooth start is no wall on this genome) and the economy-driven loop reaches 85 %, so "+10
//! points over the control" cannot hold. Legacy genome (38074e7): economy 79 % vs no-economy 56 %, scale 0.3 → 2.07.
//!
//! WALL 2 (2026-09-27): until then `reproduce` copied the fittest parent's LEARNED direction into every starved feature
//! whenever `feature_lr > 0` (#126). The parent survives `repopulate`, so the copies duplicated a living feature and
//! collapsed the 6-feature basis: economy 67 %, scale 0.3 → 0.92 on the historical organism (per stream 0.20 / 0.48 /
//! 1.15 / 0.13 / 3.25 / 0.31) and 0.43 over distinct organisms (0.20 / 0.13 / 0.61 / 0.33 / 0.76 / 0.58), below the bar.
//! An offspring now draws a fresh direction (`substrate_direction_inherit`), and the distinct-organism claim is back.
//!
//!   cargo test -p openmorph-core --test proofs substrate_evolve -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const M: usize = 3;
const WIDTH: usize = 6;
const ROUNDS: usize = 60;
const PER_ROUND: usize = 400;
const BUD: usize = 200;
const START_SCALE: f32 = 0.3; // population MEAN start (seeded with a small spread so selection can grip)
                              // Economy constants: income scales the (small) contribution credit into energy units; a useful feature
                              // banks energy, a useless one drains past DEATH and is repopulated. The newborn BIRTH reserve is its grace.
const INCOME: f32 = 25.0;
const UPKEEP: f32 = 1.0;
const DEATH: f32 = 0.5;

fn parity(x: &[f32]) -> f32 {
    -x.iter().product::<f32>()
}
fn tgt(x: &[f32]) -> f32 {
    0.7 * parity(x)
}
fn pairs(x: &[f32]) -> Vec<(usize, f32)> {
    x.iter().enumerate().map(|(k, &v)| (k, v)).collect()
}
fn corners() -> Vec<Vec<f32>> {
    (0..(1u32 << M))
        .map(|m| {
            (0..M)
                .map(|k| if (m >> k) & 1 == 0 { -1.0 } else { 1.0 })
                .collect()
        })
        .collect()
}
fn acc(f: &mut Field, out: usize) -> f64 {
    let mut ok = 0;
    for x in corners() {
        f.stimulate(&pairs(&x), BUD);
        if f.cells()[out].signum() == parity(&x).signum() {
            ok += 1;
        }
    }
    ok as f64 / (1u32 << M) as f64
}

fn build(field_seed: u64) -> (Field, Vec<usize>, usize) {
    let mut f = Field::new(M);
    f.rng_seed(field_seed);
    let inputs: Vec<usize> = (0..M).collect();
    let feats = f.grow_backbone(&inputs, WIDTH, 1.0);
    // Seed a small scale spread around START_SCALE (0.2..0.5) so the economy has variation to select on
    // from the first round; without any spread every feature is symmetric and selection has no grip.
    for (i, &h) in feats.iter().enumerate() {
        f.set_scale(h, 0.2 + 0.3 * i as f32 / (WIDTH - 1) as f32);
    }
    let out = f.len();
    f.add_unit(feats.clone(), vec![0.0; feats.len()], 0.0, out); // readout over features only
    (f, feats, out)
}

fn mean_scale(f: &Field, feats: &[usize]) -> f32 {
    feats.iter().map(|&h| f.scale_of(h)).sum::<f32>() / feats.len() as f32
}

fn train(f: &mut Field, out: usize, steps: usize, rng: &mut Rng) {
    for _ in 0..steps {
        let x: Vec<f32> = (0..M).map(|_| rng.bit()).collect();
        f.learn_step(&pairs(&x), out, tgt(&x), 0.1, BUD);
    }
}

/// The HISTORICAL seeds: 6 training streams (`0x100 + 13·s`), every one on the SAME organism (`0xE5C0`: the same
/// backbone directions and the same `reproduce` mutation draws).
const HISTORICAL: [(u64, u64); 6] = [
    (0x100, 0xE5C0),
    (0x10D, 0xE5C0),
    (0x11A, 0xE5C0),
    (0x127, 0xE5C0),
    (0x134, 0xE5C0),
    (0x141, 0xE5C0),
];

/// DISTINCT runs: `(training-stream seed, organism seed)` — each pair a different stream AND a different organism
/// (its backbone directions and its `reproduce` mutations). Both generators force the low bit (`seed | 1`), so the
/// values are odd and pairwise distinct; the streams are the historical ones, the first organism the historical one.
const DISTINCT: [(u64, u64); 6] = [
    (0x101, 0xE5C1),
    (0x10D, 0x1E5C3),
    (0x11B, 0x2E5C5),
    (0x127, 0x3E5C7),
    (0x135, 0x4E5C9),
    (0x141, 0x5E5CB),
];

/// Run both arms on the historical seeds. Returns (economy-driven population scale, economy-driven accuracy,
/// no-economy accuracy), seed means.
fn evolve() -> (f32, f64, f64) {
    evolve_over(&HISTORICAL)
}

/// Run both arms on every `(stream, organism)` seed pair; seed means as [`evolve`].
fn evolve_over(seeds: &[(u64, u64)]) -> (f32, f64, f64) {
    let (mut ev_scale, mut ev_acc, mut fx_acc) = (0.0f32, 0.0f64, 0.0f64);
    for &(seed, field_seed) in seeds {
        // NO ECONOMY: train only — scale frozen at the wrong start.
        let (mut ff, _feats, fout) = build(field_seed);
        let mut frng = Rng::new(seed);
        train(&mut ff, fout, ROUNDS * PER_ROUND, &mut frng);
        fx_acc += acc(&mut ff, fout);

        // ECONOMY-DRIVEN: each round, train (credit accrues), then let the ENERGY ACCOUNT select —
        // metabolize by contribution, repopulate the starved from the fittest. No best/worst picked here.
        let (mut f, feats, out) = build(field_seed);
        let mut rng = Rng::new(seed);
        for _ in 0..ROUNDS {
            train(&mut f, out, PER_ROUND, &mut rng);
            f.metabolize_contribution(&feats, INCOME, UPKEEP);
            f.repopulate(&feats, DEATH);
        }
        let (scale, a) = (mean_scale(&f, &feats), acc(&mut f, out));
        eprintln!(
            "[EVOLVE]   seed ({seed:#x}, {field_seed:#x}): economy-driven scale {START_SCALE} → {scale:.2}, acc {:.0}%",
            a * 100.0
        );
        ev_scale += scale;
        ev_acc += a;
    }
    let n = seeds.len() as f64;
    let (ev_scale, ev_acc, fx_acc) = (ev_scale / seeds.len() as f32, ev_acc / n, fx_acc / n);

    eprintln!(
        "[EVOLVE] 3-parity, scale starts too smooth ({START_SCALE}); tuned ONLY by the energy account ({} seeds):\n  NO ECONOMY  (train only)                        : acc {:.0}%, scale stays {START_SCALE}\n  ECONOMY-DRIVEN (metabolize_contribution + repopulate): acc {:.0}%, population scale {START_SCALE} → {:.2}",
        seeds.len(),
        fx_acc * 100.0,
        ev_acc * 100.0,
        ev_scale,
    );
    (ev_scale, ev_acc, fx_acc)
}

#[test]
fn the_economy_tunes_the_scale_with_no_external_schedule() {
    let (ev_scale, _, _) = evolve();
    eprintln!("[EVOLVE] VERDICT: the heritable-constants loop is closed through the ENERGY ACCOUNT — no external schedule. `metabolize_contribution` credits a unit by its usefulness (residual-correlation `credit`, not firing — the make-or-break), `repopulate` reseeds the starved from the fittest survivor via `reproduce` (self-adaptive mutation), and the newborn BIRTH reserve is its grace. Given a task that needs crisp features, the population scale climbs with nothing outside the organism picking winners: variation + selection + heredity, decentralized and online, in the one currency. (Since the 1.6 re-baseline this is NOT claimed to raise accuracy over the no-economy control: on the default genome the control is at 100 % and the loop reaches 85 % — see the module docs.)");

    assert!(
        ev_scale > START_SCALE + 0.15,
        "the energy account alone must drive the scale UP from {START_SCALE} (got {ev_scale:.2})"
    );
}

/// The ACCURACY half of the original proof, kept verbatim but ignored: on the 1.6 default the no-economy control reaches
/// 100 % and the economy-driven loop 85 % (67 % before the fresh-direction `reproduce`; see the module docs), so the
/// loop cannot beat the control by 10 points on this genome.
#[test]
#[ignore = "1.6 re-baseline: does not hold on the default genome — economy 85 % vs no-economy 100 % (the control already solves the task); see this file's module docs"]
fn economy_driven_tuning_beats_the_no_economy_control() {
    let (_, ev_acc, fx_acc) = evolve();
    assert!(
        ev_acc > fx_acc + 0.1,
        "economy-driven tuning must beat the frozen-wrong-scale control ({ev_acc:.2} vs {fx_acc:.2})"
    );
}

/// The SCALE claim over DISTINCT organisms (a different backbone and mutation stream per run, not only a different
/// training stream). Holds since the fresh-direction `reproduce` (2026-09-27): the population scale ends at 0.80 /
/// 1.26 / 0.41 / 1.14 / 3.51 / 0.96 (mean 1.35). Before it, with the parent's direction copied, 0.20 / 0.13 / 0.61 /
/// 0.33 / 0.76 / 0.58 (mean 0.43 < the 0.45 bar), and this test was ignored.
#[test]
fn the_economy_tunes_the_scale_over_distinct_organisms() {
    let (ev_scale, _, _) = evolve_over(&DISTINCT);
    assert!(
        ev_scale > START_SCALE + 0.15,
        "the energy account alone must drive the scale UP from {START_SCALE} (mean over distinct organisms {ev_scale:.2})"
    );
}
