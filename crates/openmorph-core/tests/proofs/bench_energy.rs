// SPDX-License-Identifier: MIT-0
//! #165/#167 (BM-1/BM-3) — the ENERGY-PER-CAPABILITY ratchet: the true metric is LESS ENERGY for MORE POWER
//! (Sébastien). A capability-only bench can be "passed" by ADDING machinery (a founding-rule-#2 violation); this
//! gates capability PER COST. Each rung measures, on a FRESH organism, three numbers read from the deployed Field
//! itself (`deployed_is_computed`): the CAPABILITY (a held-out score in [0,1]), the ENERGY (`clock()` — the
//! substrate's own subjective-time = total settle-firings spent), and the MACHINERY (`n_units()` grown). The
//! RATCHET (a committed floor table): capability may only go UP, energy and machinery only DOWN — a regression on
//! ANY axis fails, so the substrate can never buy a capability by getting fatter. Deterministic (seeded xorshift,
//! fixed data), so CI is cache-safe and reproducible bit-for-bit.
//!
//! TWO GENOMES, TWO TABLES (the 1.6 learning flip). The committed floor table ([`RATCHET`]) is the historical gate
//! and runs on [`Config::legacy`], so it keeps guarding the substrate's efficiency on the computation it was
//! calibrated for. The 1.6 default genome gets its own table ([`RATCHET_DEFAULT`]): on these three saturated rungs it
//! reaches the same capability for 2.1–2.7× the firings and 2.2–2.5× the units — its mean-relative economy keeps
//! features the readout ignores EQUALLY instead of pruning them (the price of the accuracy the flip buys on harder
//! tasks; with the absolute economy the flipped genome is leaner than legacy here). Every rung asserts that NO settle
//! was truncated: a genome that saves firings by leaving its settles unfinished (a pool larger than the rung's
//! budget) is not more efficient. The rung budgets (40–60) sit below the default pool's `max_hidden + 2` = 66, so
//! under the default genome they BOUND the pool (`Config::grow_within_budget`); at a budget ≥ 66, R2 grows to 57
//! units and 262 074 firings (2.38× / 2.5× legacy).
use openmorph_core::{Config, Event, Field};

use crate::common::Rng;

/// One bench rung: (capability score in [0,1], energy = firings, machinery = units grown, settles the anytime
/// budget cut short — must be 0, or the energy was saved by never finishing the computation).
struct Rung {
    cap: f32,
    energy: u64,
    units: usize,
    truncated: u64,
}

/// R1 — LEARN a nonlinear map y = tanh(2·x) from experience; capability = held-out correlation.
fn r1_learn(genome: &Config) -> Rung {
    let mut f = Field::new(2);
    f.configure(genome.clone());
    let mut rng = Rng::new(0xB1);
    for _ in 0..3000 {
        let x = rng.sym();
        f.alive(Event::Perceive {
            inputs: &[(0, x)],
            target_cell: 1,
            target: (2.0 * x).tanh(),
            lr: 0.1,
            budget: 50,
        });
    }
    // held-out correlation
    let (mut sxy, mut sxx, mut syy, mut mx, mut my, mut n) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut preds = Vec::new();
    for i in 0..40 {
        let x = -1.0 + 2.0 * i as f32 / 39.0;
        f.stimulate(&[(0, x)], 50);
        preds.push((f.cells()[1], (2.0 * x).tanh()));
    }
    for &(p, t) in &preds {
        mx += p;
        my += t;
        n += 1.0;
    }
    mx /= n;
    my /= n;
    for &(p, t) in &preds {
        sxy += (p - mx) * (t - my);
        sxx += (p - mx) * (p - mx);
        syy += (t - my) * (t - my);
    }
    let corr = (sxy / (sxx.sqrt() * syy.sqrt()).max(1e-9)).clamp(0.0, 1.0);
    Rung {
        cap: corr,
        energy: f.clock(),
        units: f.n_units(),
        truncated: f.truncated_settles(),
    }
}

/// R2 — GROW DEPTH: learn XOR (not linearly separable); capability = accuracy over the 4 cases.
fn r2_xor(genome: &Config) -> Rung {
    let mut f = Field::new(3);
    f.configure(genome.clone());
    let mut rng = Rng::new(0xB2);
    let cases = [
        ([-1.0f32, -1.0], -1.0f32),
        ([-1.0, 1.0], 1.0),
        ([1.0, -1.0], 1.0),
        ([1.0, 1.0], -1.0),
    ];
    for _ in 0..6000 {
        let c = rng.below(4);
        let (x, t) = cases[c];
        f.alive(Event::Perceive {
            inputs: &[(0, x[0]), (1, x[1])],
            target_cell: 2,
            target: t,
            lr: 0.1,
            budget: 60,
        });
    }
    let mut ok = 0;
    for (x, t) in cases {
        f.stimulate(&[(0, x[0]), (1, x[1])], 60);
        if f.cells()[2].signum() == t.signum() {
            ok += 1;
        }
    }
    Rung {
        cap: ok as f32 / 4.0,
        energy: f.clock(),
        units: f.n_units(),
        truncated: f.truncated_settles(),
    }
}

/// R3 — COMPOSE by nesting: grow a skill s(x,y)=tanh(x−y), encapsulate it, nest it as one cell, apply.
/// Capability = correctness of the composed output on a held-out point.
fn r3_compose(genome: &Config) -> Rung {
    let mut s = Field::new(3);
    s.configure(genome.clone());
    let mut rng = Rng::new(0xB3);
    for _ in 0..2000 {
        let (x, y) = (rng.sym(), rng.sym());
        s.alive(Event::Perceive {
            inputs: &[(0, x), (1, y)],
            target_cell: 2,
            target: (x - y).tanh(),
            lr: 0.1,
            budget: 40,
        });
    }
    let skill = s.encapsulate(&[2]);
    let mut host = Field::new(3);
    host.nest(skill, &[0, 1], 2, 40);
    let (x, y) = (0.4f32, -0.3);
    host.stimulate(&[(0, x), (1, y)], 40);
    let err = (host.cells()[2] - (x - y).tanh()).abs();
    let cap = (1.0 - err).clamp(0.0, 1.0);
    // energy/machinery of the WHOLE pipeline (grow the skill + compose)
    Rung {
        cap,
        energy: s.clock() + host.clock(),
        units: s.n_units() + host.n_units(),
        truncated: s.truncated_settles() + host.truncated_settles(),
    }
}

// ── THE RATCHET FLOOR (committed; the never-decrease gate). Capability floors: current − a small tolerance.
//    Energy/machinery ceilings: current + a small headroom. Improving the substrate = raising CAP_MIN or lowering
//    ENERGY_MAX/UNITS_MAX in a reviewed bump (recorded here). Deterministic, so these are stable across runs. ──
struct Floor {
    name: &'static str,
    cap_min: f32,
    energy_max: u64,
    units_max: usize,
}
// Calibrated 2026-07-27 @ de4096a-era: R1 (cap 1.000, 25 063 firings, 7 units), R2 (0.750, 107 764, 23),
// R3 (0.760, 12 471, 6). Floors = cap just below / energy+units just above — the never-decrease gate with headroom.
// Runs on `Config::legacy()` since the 1.6 flip (the genome it was calibrated on).
const RATCHET: &[Floor] = &[
    Floor {
        name: "R1-learn-tanh",
        cap_min: 0.98,
        energy_max: 30_000,
        units_max: 10,
    },
    Floor {
        name: "R2-grow-xor",
        cap_min: 0.70,
        energy_max: 130_000,
        units_max: 30,
    },
    Floor {
        name: "R3-compose-nest",
        cap_min: 0.70,
        energy_max: 16_000,
        units_max: 10,
    },
];

// The 1.6 DEFAULT genome's table, on complete settles (no truncation, asserted): R1 (cap 1.000, 58 364 firings, 16
// units), R2 (1.000, 236 202, 51 — the budget-60 settle bounds the pool), R3 (0.968, 34 147, 15). Same capability
// floors as `RATCHET`. The ceilings were set at the flip (+~20 % over R2's 214 977 / 46, which came from settles the
// budget cut short) and are kept: the complete-settle numbers sit under them.
const RATCHET_DEFAULT: &[Floor] = &[
    Floor {
        name: "R1-learn-tanh",
        cap_min: 0.98,
        energy_max: 70_000,
        units_max: 19,
    },
    Floor {
        name: "R2-grow-xor",
        cap_min: 0.70,
        energy_max: 258_000,
        units_max: 55,
    },
    Floor {
        name: "R3-compose-nest",
        cap_min: 0.70,
        energy_max: 41_000,
        units_max: 18,
    },
];

// ── THE PIN (the determinism half of the gate). The floors above only say "no worse than"; they absorb a one-ulp
//    drift of every number. These are the EXACT firings and units each rung spends — the bench is deterministic
//    (seeded, fixed data), so any change to them is a change of the organism's behaviour. If it was intended (a
//    deliberate change to a default, a kernel, the settle order), update the pin from the `--nocapture` table in
//    the same change and record it in `CHANGELOG.md`; if it was not, it is a regression. Asserted on the CI host
//    (x86_64 Linux glibc) only: under `std` the substrate's `tanh`/`exp` come from the platform libm (see
//    `substrate_determinism`). ──
struct Pin {
    name: &'static str,
    energy: u64,
    units: usize,
}
/// Whether this is the host the pins were recorded on.
const PINNED_HOST: bool = cfg!(all(
    target_arch = "x86_64",
    target_os = "linux",
    target_env = "gnu"
));
// Recorded 2026-09-25 @ b63d3ac (identical at opt-level 0, 2 and 3). The legacy R2 spends 110 212 firings, not
// the 107 764 of the 2026-07-27 calibration above: a drift the ceilings absorbed, now pinned.
const PIN_LEGACY: &[Pin] = &[
    Pin {
        name: "R1-learn-tanh",
        energy: 25_063,
        units: 7,
    },
    Pin {
        name: "R2-grow-xor",
        energy: 110_212,
        units: 23,
    },
    Pin {
        name: "R3-compose-nest",
        energy: 12_471,
        units: 6,
    },
];
const PIN_DEFAULT: &[Pin] = &[
    Pin {
        name: "R1-learn-tanh",
        energy: 58_364,
        units: 16,
    },
    Pin {
        name: "R2-grow-xor",
        energy: 236_202,
        units: 51,
    },
    Pin {
        name: "R3-compose-nest",
        energy: 34_147,
        units: 15,
    },
];

#[test]
fn energy_per_capability_ratchet() {
    ratchet("legacy", &Config::legacy(), RATCHET, PIN_LEGACY);
}

#[test]
fn energy_per_capability_ratchet_default_genome() {
    ratchet("default", &Config::default(), RATCHET_DEFAULT, PIN_DEFAULT);
}

fn ratchet(genome_name: &str, genome: &Config, table: &[Floor], pins: &[Pin]) {
    let rungs = [
        ("R1-learn-tanh", r1_learn(genome)),
        ("R2-grow-xor", r2_xor(genome)),
        ("R3-compose-nest", r3_compose(genome)),
    ];
    eprintln!("[BENCH] ENERGY-PER-CAPABILITY (true metric: less energy for more power), {genome_name} genome. deployed_is_computed — every number read from the Field.");
    eprintln!(
        "[BENCH]   {:<16} {:>7} {:>10} {:>7} {:>12}",
        "rung", "cap", "energy", "units", "power/energy"
    );
    let mut composite_cap = 0.0f32;
    let mut composite_eff = 0.0f32;
    for (name, r) in &rungs {
        let eff = 1000.0 * r.cap / r.energy.max(1) as f32; // capability per 1000 firings — the efficiency
        eprintln!(
            "[BENCH]   {name:<16} {:>7.3} {:>10} {:>7} {:>12.4}  (truncated settles {})",
            r.cap, r.energy, r.units, eff, r.truncated
        );
        composite_cap += r.cap;
        composite_eff += eff;
    }
    composite_cap /= rungs.len() as f32;
    composite_eff /= rungs.len() as f32;
    eprintln!("[BENCH]   COMPOSITE: capability {composite_cap:.3}, efficiency (power/energy) {composite_eff:.4}");

    // THE RATCHET GATE — a regression on ANY axis (capability down, energy up, machinery up) fails CI.
    for ((name, r), floor) in rungs.iter().zip(table) {
        assert_eq!(*name, floor.name);
        assert_eq!(
            r.truncated, 0,
            "TRUNCATED SETTLES on {name}: energy saved by cutting settles short is not efficiency"
        );
        assert!(
            r.cap >= floor.cap_min,
            "CAPABILITY REGRESSION on {name}: {:.3} < floor {:.3}",
            r.cap,
            floor.cap_min
        );
        assert!(
            r.energy <= floor.energy_max,
            "ENERGY REGRESSION on {name} (bought capability with more compute): {} > ceiling {}",
            r.energy,
            floor.energy_max
        );
        assert!(r.units <= floor.units_max, "MACHINERY REGRESSION on {name} (bought capability with more units — rule #2): {} > ceiling {}", r.units, floor.units_max);
    }

    // THE PIN — the exact firings and units (see `Pin`).
    for ((name, r), pin) in rungs.iter().zip(pins) {
        assert_eq!(*name, pin.name);
        if PINNED_HOST {
            assert_eq!(
                (r.energy, r.units),
                (pin.energy, pin.units),
                "BEHAVIOUR CHANGED on {name} ({genome_name} genome): (firings, units) moved off the pin. If intended, \
                 update the pin from this run's table and record the change in CHANGELOG.md; if not, it is a regression."
            );
        }
    }

    eprintln!("[BENCH] VERDICT: the ENERGY-PER-CAPABILITY ratchet holds — capability at/above floor AND energy/machinery at/below ceiling on every rung. The true metric (less energy for more power) is now a HARD gate: the substrate cannot buy a capability by getting fatter (more units) or slower (more firings). Every number is read from the deployed Field (deployed_is_computed), deterministic (seeded, cache-safe). Improving the substrate = raising a cap floor or lowering an energy/units ceiling in a reviewed bump. This is the missing energy denominator on MorphLab's capability-only bench; new rungs (temporal, abstraction, agency) accrete the same three-number shape.");
}
