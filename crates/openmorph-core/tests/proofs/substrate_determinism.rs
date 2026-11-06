// SPDX-License-Identifier: MIT-0
//! DETERMINISM GUARD — the byte-identical-replay promise, enforced ACROSS COMMITS. Every other "byte-identical"
//! check in the suite compares two runs inside one process (`a.to_bits() == b.to_bits()`) or the native `Field`
//! against the physics engine; both pass unchanged if a shared kernel or a default code path shifts every number
//! by one ulp, because the proofs' thresholds absorb it. This proof does not: it runs ONE small fixed stream that
//! touches every faculty class and PINS the result.
//!
//! THE STREAM (deterministic: seeded xorshift, fixed data, no wall-clock), run under the default genome AND under
//! [`Config::legacy`]:
//!   1. GROW    — `grow` toward a non-linear target (surprise-birth + hidden random-projection features).
//!   2. ALIVE   — `alive(Event::Perceive)` toward a second target (the living pool: grow + the per-event economy).
//!   3. OBSERVE — target-free predictive coding of a rotating world over a grown backbone (`observe`).
//!   4. CULL    — `metabolize_reliance` + `cull` over every cell (the reliance economy prunes the unused).
//!   5. ACT     — a planned value field (`prefer` + `relax_values`) walked by `alive(Event::Act)` with rewards.
//!   6. NEST    — `encapsulate` the alive-grown skill and `nest` it as one cell of a host, fired in its cascade.
//!
//! THE PIN. Each run yields five numbers: a hash of every per-step output (errors, births, culls, choices, nested
//! outputs), a FULL-STATE hash of each of the three organisms (its canonical serialized state — every serialized
//! field, key-sorted, floats by their bits — plus every cell's `to_bits()`, its clock, size and unit count), and
//! the total firings. They are compared with `assert_eq!` against the constants below.
//!
//! A FAILURE HERE MEANS THE ORGANISM'S BEHAVIOUR CHANGED. If that was NOT intended, it is a regression: find it.
//! If it WAS intended (a deliberate change to a default, a kernel, the settle order, the serialized state), update
//! the pin in the same change — run `cargo test -p openmorph-core --test proofs substrate_determinism --
//! --nocapture`, copy the printed values into [`PIN_DEFAULT`]/[`PIN_LEGACY`] — AND record the behaviour change in
//! `CHANGELOG.md`. A pure refactor (splitting a module, renaming a private helper) must leave the pin untouched.
//!
//! THE HOST. Under `std` the substrate's `tanh`/`exp` come from the platform libm, so the exact bits are pinned on
//! the CI host (`x86_64` Linux glibc, where the constants were recorded). On any other host the pin is not
//! asserted (only printed); in-process replay (two runs, bit-for-bit) is asserted everywhere. If the pin breaks on
//! the CI host with no code change, the runner's libm moved — the cross-host float issue `src/compat.rs` names —
//! re-pin with a note in the CHANGELOG.
//!
//!   cargo test -p openmorph-core --test proofs substrate_determinism -- --nocapture

use openmorph_core::{Config, Event, Field, Outcome};
use serde_json::Value;

use crate::common::Rng;

/// The pinned result of [`stream`] on the default genome (recorded 2026-09-25 @ b63d3ac on x86_64 Linux glibc 2.39,
/// the CI host's libm; identical at opt-level 0, 2 and 3). Re-pinned 2026-09-28 for the serialized state only: the
/// opt-in RLS readout added the `readout_rls` / `rls_prior` config keys and `Field::rls`. With those three keys
/// stripped both genomes hash to the previous pins, and `trace` and `firings` did not move. Re-pinned again the
/// same day for the `pool_surprise` config key alone (stripped, the previous pins; `trace` and `firings` unchanged).
/// Re-pinned 2026-10-06 for the serialized state only: the opt-in per-target product discovery added the
/// `products_per_target` config key and `Field::target_credit` / `Field::target_ticks` (stripped, the previous pins
/// on both genomes; `trace` and `firings` unchanged). Re-pinned 2026-10-08 for the serialized state only: the removal
/// of `ActRule::ActScore` took the `hypothesis_loop`, `openmindedness`, `drive_decay` and `metabolic_actor` keys out of
/// every serialized `Config` (with those four keys stripped from the old state, `organism`, `actor` and `host` hash
/// to these values; `trace` and `firings` unchanged).
const PIN_DEFAULT: Prints = Prints {
    trace: 0xcf4f_05d9_decf_887c,
    organism: 0x0892_639b_b636_b2d5,
    actor: 0x1d33_66ed_57f4_ac0c,
    host: 0xc241_5504_7a9f_6b20,
    firings: 61_721,
};

/// The pinned result of [`stream`] on [`Config::legacy`]. Re-pinned 2026-10-08 for a BEHAVIOUR change: `ActScore`
/// was removed, so `Config::legacy()` acts with the default `Valence` rule. `trace` (stage 5's 40 choices) and `actor`
/// (the agent's `(μ, visits)` records instead of `drive`) moved for that; `organism` and `host` moved only because the
/// four removed keys left the serialized `Config` and `act_rule` reads `"Valence"`. Each value equals what the
/// previous revision produced for `Config { act_rule: Valence, ..Config::legacy() }` with the four keys stripped;
/// `firings` unchanged (acting fires no unit).
const PIN_LEGACY: Prints = Prints {
    trace: 0x54d3_16b4_f624_3b64,
    organism: 0x4e8c_51e7_c9a0_3b60,
    actor: 0x1057_20f4_d2e6_b2b2,
    host: 0x958e_50f0_da40_77e8,
    firings: 36_796,
};

/// Whether this is the host the pins were recorded on (the platform libm supplies `tanh`/`exp` under `std`).
const PINNED_HOST: bool = cfg!(all(
    target_arch = "x86_64",
    target_os = "linux",
    target_env = "gnu"
));

/// What one run of the stream produced (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Prints {
    trace: u64,
    organism: u64,
    actor: u64,
    host: u64,
    firings: u64,
}

/// FNV-1a over bytes — a tiny, dependency-free, platform-independent hash.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.bytes(&v.to_bits().to_le_bytes());
    }
    /// A CANONICAL walk of a serialized value: object keys sorted (independent of struct field order and of
    /// serde_json's `preserve_order`), numbers by their exact bits (independent of float formatting).
    fn value(&mut self, v: &Value) {
        match v {
            Value::Null => self.u64(0),
            Value::Bool(b) => {
                self.u64(1);
                self.u64(*b as u64);
            }
            Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    self.u64(2);
                    self.u64(u);
                } else if let Some(i) = n.as_i64() {
                    self.u64(3);
                    self.u64(i as u64);
                } else {
                    self.u64(4);
                    self.u64(
                        n.as_f64()
                            .expect("a JSON number is u64, i64 or f64")
                            .to_bits(),
                    );
                }
            }
            Value::String(s) => {
                self.u64(5);
                self.u64(s.len() as u64);
                self.bytes(s.as_bytes());
            }
            Value::Array(a) => {
                self.u64(6);
                self.u64(a.len() as u64);
                for x in a {
                    self.value(x);
                }
            }
            Value::Object(m) => {
                self.u64(7);
                self.u64(m.len() as u64);
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                for k in keys {
                    self.u64(k.len() as u64);
                    self.bytes(k.as_bytes());
                    self.value(&m[k]);
                }
            }
        }
    }
}

/// The FULL-STATE hash of one organism: its canonical serialized state + every cell's bits + clock/size/units.
fn state_hash(f: &Field) -> u64 {
    let mut h = Fnv::new();
    h.value(&serde_json::to_value(f).expect("a Field serializes"));
    for &c in f.cells() {
        h.f32(c);
    }
    h.u64(f.clock());
    h.u64(f.len() as u64);
    h.u64(f.n_units() as u64);
    h.0
}

/// A 4×4 grid world: each state's four (clamped) neighbours.
fn grid(n: usize) -> Vec<Vec<usize>> {
    let at = |r: i32, c: i32| {
        (r.clamp(0, n as i32 - 1) as usize) * n + c.clamp(0, n as i32 - 1) as usize
    };
    (0..n * n)
        .map(|s| {
            let (r, c) = ((s / n) as i32, (s % n) as i32);
            vec![at(r - 1, c), at(r + 1, c), at(r, c - 1), at(r, c + 1)]
        })
        .collect()
}

/// What the stream exercised — asserted non-trivial so the pin guards real work, not an idle organism.
struct Exercised {
    grow_births: usize,
    alive_births: usize,
    culled: usize,
    distinct_choices: usize,
    nested_settles: u64,
}

/// Run the fixed mixed stream under `genome`.
fn stream(genome: &Config) -> (Prints, Exercised) {
    let mut trace = Fnv::new();
    let mut ex = Exercised {
        grow_births: 0,
        alive_births: 0,
        culled: 0,
        distinct_choices: 0,
        nested_settles: 0,
    };

    // ── 1. GROW + 2. ALIVE — cells 0,1 = inputs; 2 = the `grow` target; 3 = the `alive` target. ──
    let mut f = Field::new(4);
    f.configure(genome.clone());
    f.rng_seed(0xDE7E_0001);
    let mut rng = Rng::new(0xD17E_0003);
    for _ in 0..300 {
        let (a, b) = (rng.sym(), rng.sym());
        let (err, born) = f.grow(&[(0, a), (1, b)], 2, (2.0 * a * b).tanh(), 0.1, 200);
        trace.f32(err);
        trace.u64(born as u64);
        ex.grow_births += born as usize;
        match f.alive(Event::Perceive {
            inputs: &[(0, a), (1, b)],
            target_cell: 3,
            target: 0.8 * (a - b),
            lr: 0.1,
            budget: 200,
        }) {
            Outcome::Perceived { error, born } => {
                trace.f32(error);
                trace.u64(born as u64);
                ex.alive_births += born as usize;
            }
            _ => unreachable!("a Perceive event reports Perceived"),
        }
    }

    // ── 3. OBSERVE — predict the next state of a rotating world, target-free, over a grown backbone. ──
    let mut reads = vec![0, 1];
    reads.extend(f.grow_backbone(&[0, 1], 6, 2.0));
    let pred = f.len();
    for p in 0..2 {
        f.add_unit(reads.clone(), vec![0.0; reads.len()], 0.0, pred + p);
    }
    let (mut x, mut y) = (0.8f32, 0.0f32);
    f.stimulate(&[(0, x), (1, y)], 200);
    for _ in 0..200 {
        let (nx, ny) = (0.955 * x - 0.296 * y, 0.296 * x + 0.955 * y);
        for e in f.observe(0, pred, 2, &[nx, ny], 0.05, 200) {
            trace.f32(e);
        }
        (x, y) = (nx, ny);
    }

    // ── 4. CULL — the reliance economy starves what no reader leans on. ──
    let cells: Vec<usize> = (0..f.len()).collect();
    for _ in 0..4 {
        f.metabolize_reliance(&cells, 2.0, 1.0);
        let n = f.cull(0.5);
        trace.u64(n as u64);
        ex.culled += n;
    }

    // ── 5. ACT — plan a 4×4 grid's cost-to-go, then walk it through the native policy loop with rewards. ──
    let adj = grid(4);
    let goal = 15;
    let mut agent = Field::new(16);
    agent.configure(genome.clone());
    agent.rng_seed(0xAC7_0005);
    agent.prefer(goal, 0.0);
    agent.relax_values(&adj, 1.0, 0);
    let (mut s, mut last) = (0usize, None);
    let mut seen = [false; 16];
    for _ in 0..40 {
        let chosen = match agent.alive(Event::Act {
            options: &adj[s],
            last,
        }) {
            Outcome::Acted { chosen } => chosen,
            _ => unreachable!("an Act event reports Acted"),
        };
        trace.u64(chosen as u64);
        seen[chosen] = true;
        let reward = if chosen == goal { 1.0 } else { -0.05 };
        last = Some((chosen, reward));
        s = if chosen == goal { 0 } else { chosen };
    }
    ex.distinct_choices = seen.iter().filter(|&&v| v).count();

    // ── 6. NEST — bank the alive-grown skill (cell 3) as a sub-morph and fire it as ONE cell of a host. ──
    let skill = f.encapsulate(&[3]);
    let mut host = Field::new(3);
    host.configure(genome.clone());
    host.nest(skill, &[0, 1], 2, 200);
    for i in 0..12 {
        let (a, b) = ((i % 4) as f32 * 0.5 - 0.75, (i / 4) as f32 * 0.5 - 0.5);
        host.stimulate(&[(0, a), (1, b)], 200);
        trace.f32(host.cells()[2]);
    }
    trace.u64(host.nested_settles());
    ex.nested_settles = host.nested_settles();

    let prints = Prints {
        trace: trace.0,
        organism: state_hash(&f),
        actor: state_hash(&agent),
        host: state_hash(&host),
        firings: f.clock() + agent.clock() + host.clock(),
    };
    (prints, ex)
}

fn check(name: &str, genome: &Config, pin: Prints) -> Prints {
    let (a, ex) = stream(genome);
    let (b, _) = stream(genome);
    eprintln!(
        "[DETERMINISM] {name}: Prints {{ trace: {:#018x}, organism: {:#018x}, actor: {:#018x}, host: {:#018x}, firings: {} }}",
        a.trace, a.organism, a.actor, a.host, a.firings
    );
    eprintln!(
        "[DETERMINISM] {name}: exercised — grow births {}, alive births {}, culled {}, distinct actions {}, nested settles {}",
        ex.grow_births, ex.alive_births, ex.culled, ex.distinct_choices, ex.nested_settles
    );
    // The stream does real work in every faculty, so the pin guards behaviour, not an idle organism.
    assert!(ex.grow_births > 0, "{name}: `grow` must birth units");
    assert!(ex.alive_births > 0, "{name}: `alive` must birth units");
    assert!(
        ex.culled > 0,
        "{name}: the reliance economy must cull something"
    );
    assert!(
        ex.distinct_choices >= 4,
        "{name}: the actor must move through the grid"
    );
    assert!(
        ex.nested_settles > 0,
        "{name}: the nested skill must settle in the host's cascade"
    );
    // In-process replay: the same stream twice is bit-for-bit the same (every host).
    assert_eq!(
        a, b,
        "{name}: the same stream replayed in one process must be bit-for-bit identical"
    );
    if PINNED_HOST {
        assert_eq!(
            a, pin,
            "{name}: the organism's behaviour CHANGED against the pinned replay. If unintended, this is a regression. \
             If intended, update PIN_{} in this file with the printed values and record the change in CHANGELOG.md.",
            name.to_uppercase()
        );
    } else {
        eprintln!("[DETERMINISM] {name}: not the pinned host (x86_64 Linux glibc) — pin printed, not asserted");
    }
    a
}

#[test]
fn a_fixed_mixed_stream_replays_to_the_pinned_state_on_the_default_genome() {
    let d = check("default", &Config::default(), PIN_DEFAULT);
    // The two genomes run different learning rules, so the stream must tell them apart (a pin that could not
    // see the genome would guard nothing).
    let (l, _) = stream(&Config::legacy());
    assert_ne!(
        d.trace, l.trace,
        "the stream must distinguish the default genome from the legacy one"
    );
}

#[test]
fn a_fixed_mixed_stream_replays_to_the_pinned_state_on_the_legacy_genome() {
    check("legacy", &Config::legacy(), PIN_LEGACY);
}
