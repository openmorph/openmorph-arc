// SPDX-License-Identifier: MIT-0
//! substrate_act_rule — ONE actor path (`Config::act_rule`, 1.6): the default valence rule, its seeded unbiased
//! tie-break, "untried" meaning NEVER VISITED, the deprecated aliases, and config back-compatibility.
//!
//! THE CLAIMS (each a contract the default `alive(Event::Act)` now keeps):
//!   (A) the default rule is `Valence`, and `Config::legacy()` keeps it (the pre-1.6 `ActScore` rule is removed:
//!       legacy differs from the default only in the learning levers); the deprecated bools keep their pre-1.6
//!       meaning and precedence (`free_energy_actor` wins, then `valence_actor`, then the enum).
//!   (B) exact ties among the best options are broken UNIFORMLY by the seeded act PRNG — no positional bias — and
//!       the draw is replayable from the seed.
//!   (C) "untried" = never visited: an option that paid ZERO is visited, so the valence rule moves on to the
//!       never-visited ones. (The removed `ActScore` rule's `1[drive == 0]` novelty treated a zero-reward option as
//!       untried forever and re-picked it.)
//!   (D) `act_rule = FreeEnergy` and the deprecated `free_energy_actor = true` are the SAME rule, bit for bit.
//!   (E) a config JSON from before 1.6 still loads: the deleted `free_energy_softmax` key is ignored, and absent
//!       `act_rule` / `act_discount` keys take their defaults; a config or snapshot naming the removed `"ActScore"`
//!       rule loads as `Valence`.
//!   (F) the visit count is EXACT over hundreds of zero-reward visits: a less-visited option always scores at least
//!       as high as a more-visited one (strictly while the bonus is representable), `alive(Act)` picks the
//!       less-visited, and a flat zero-reward bandit is an exact round-robin, then uniform once the bonus underflows.
//!       (Before the `(μ, visits)` record the valence read a 24-member ensemble whose f32 prior mean was −1.99e-8:
//!       past ~115 zero-reward visits that residue outweighed σ² and the argmax preferred the MOST-visited option —
//!       one arm of a flat 4-arm bandit took 888 of 1200 picks.)
//!   (G) the snapshot migration: an organism whose values live in a value ENSEMBLE (a pre-1.6 `valence_actor`
//!       snapshot, or one that ran `FreeEnergy`) is read by the Valence rule as the `(μ, visits)` it encodes — the
//!       visit count recovered exactly, a zero-reward history's f32 residue read as μ = 0. (Its snapshot pins the
//!       removed `"ActScore"`, so it also exercises the alias.)
//!   (H) a TRAINED valence organism survives `snapshot`/`load`: the reloaded organism continues bit for bit (its
//!       `(μ, visits)` records and its tie-break stream are serialised).
//!
//!   cargo test -p openmorph-core --test proofs substrate_act_rule -- --nocapture

use openmorph_core::{ActRule, Config, Event, Field, Outcome, RelianceNorm};

fn act(f: &mut Field, options: &[usize], last: Option<(usize, f32)>) -> usize {
    match f.alive(Event::Act { options, last }) {
        Outcome::Acted { chosen } => chosen,
        _ => unreachable!("alive(Act) always acts"),
    }
}

#[test]
fn a_default_rule_legacy_genome_and_alias_precedence() {
    let d = Config::default();
    assert_eq!(
        d.act_rule,
        ActRule::Valence,
        "the 1.6 default rule is Valence"
    );
    assert_eq!(d.effective_act_rule(), ActRule::Valence);
    assert_eq!(
        d.act_discount, 0.0,
        "the TD discount defaults to the immediate-reward fold"
    );
    assert_eq!(
        Config::legacy().effective_act_rule(),
        ActRule::Valence,
        "legacy() acts with the default rule (the pre-1.6 one is removed)"
    );
    assert_eq!(
        Config::legacy(),
        Config {
            // the 1.6 learning flip, undone (pinned field by field in `substrate_legacy_replay`)
            stability_cap: 0.0,
            feature_lr: 0.0,
            fanin_init_gain: 0.0,
            reliance_norm: RelianceNorm::Absolute,
            grow_within_budget: false,
            ..Config::default()
        },
        "legacy() is the default with the 1.6 learning levers undone, and nothing else"
    );
    let with = |fe: bool, va: bool, rule: ActRule| {
        Config {
            free_energy_actor: fe,
            valence_actor: va,
            act_rule: rule,
            ..Config::default()
        }
        .effective_act_rule()
    };
    assert_eq!(
        with(true, true, ActRule::Valence),
        ActRule::FreeEnergy,
        "free_energy_actor wins (pre-1.6 precedence)"
    );
    assert_eq!(
        with(false, true, ActRule::FreeEnergy),
        ActRule::Valence,
        "valence_actor overrides the enum"
    );
    assert_eq!(
        with(false, false, ActRule::FreeEnergy),
        ActRule::FreeEnergy,
        "no flag ⇒ the enum decides"
    );
}

#[test]
fn b_exact_ties_are_broken_uniformly_by_the_seeded_prng() {
    const K: usize = 4;
    const SEEDS: u64 = 400;
    let options: Vec<usize> = (0..K).collect();
    // A fresh organism faces K never-visited options: every score is tied. Count the first pick per seed.
    let first_pick = |cfg: Config, seed: u64| {
        let mut f = Field::new(K);
        f.rng_seed(seed);
        f.configure(cfg);
        act(&mut f, &options, None)
    };
    let mut valence = [0usize; K];
    for seed in 0..SEEDS {
        valence[first_pick(Config::default(), seed)] += 1;
    }
    eprintln!(
        "[ACT-RULE B] first pick among {K} tied options over {SEEDS} seeds: valence {valence:?}"
    );
    let expect = SEEDS as f32 / K as f32;
    for (o, &n) in valence.iter().enumerate() {
        assert!(
            (n as f32 - expect).abs() < 0.3 * expect,
            "the valence tie-break must be ~uniform: option {o} picked {n}/{SEEDS} (expected ≈{expect})"
        );
    }
    // Replayable: the same seed gives the same sequence of tie-breaks; the draw depends on the seed.
    let run = |seed: u64| -> Vec<usize> {
        let mut f = Field::new(K);
        f.rng_seed(seed);
        let mut last = None;
        (0..64)
            .map(|_| {
                let a = act(&mut f, &options, last);
                last = Some((a, 0.0));
                a
            })
            .collect()
    };
    assert_eq!(
        run(11),
        run(11),
        "the tie-break is deterministic and replayable from the seed"
    );
    assert_ne!(
        run(11),
        run(12),
        "different seeds draw different tie-breaks"
    );
}

#[test]
fn c_untried_means_never_visited_not_zero_reward() {
    const K: usize = 3;
    let options: Vec<usize> = (0..K).collect();
    // ZERO reward on every act: nothing ever pays.
    let visits = |cfg: Config| -> Vec<usize> {
        let mut f = Field::new(K);
        f.configure(cfg);
        let mut last = None;
        let mut seen = Vec::new();
        for _ in 0..K {
            let a = act(&mut f, &options, last);
            seen.push(a);
            last = Some((a, 0.0));
        }
        seen
    };
    let mut v = visits(Config::default());
    eprintln!("[ACT-RULE C] first {K} picks under zero reward: valence {v:?}");
    v.sort_unstable();
    assert_eq!(
        v,
        (0..K).collect::<Vec<_>>(),
        "a zero-reward option is VISITED: the valence rule tries every never-visited option before any repeat"
    );
    // The exploration bonus is a visit count: one zero-reward visit already lowers it below a never-visited option's.
    let mut f = Field::new(K);
    let _ = act(&mut f, &options, None);
    let _ = act(&mut f, &options, Some((0, 0.0)));
    assert!(
        f.valence_of(0) < f.valence_of(1),
        "a visited zero-reward option ({:.4}) must score below a never-visited one ({:.4})",
        f.valence_of(0),
        f.valence_of(1)
    );
}

#[test]
fn d_free_energy_rule_equals_the_deprecated_flag_bit_for_bit() {
    let options = [0usize, 1, 2, 3, 4];
    let pay = |a: usize, t: usize| ((a * 7 + t * 3) % 5) as f32 * 0.2 - 0.3;
    let run = |cfg: Config| -> (Vec<usize>, Vec<u32>) {
        let mut f = Field::new(5);
        f.rng_seed(3);
        f.configure(cfg);
        let mut last = None;
        let mut picks = Vec::new();
        let mut spreads = Vec::new();
        for t in 0..300 {
            let a = act(&mut f, &options, last);
            picks.push(a);
            spreads.push(f.g_spread().to_bits());
            last = Some((a, pay(a, t)));
        }
        (picks, spreads)
    };
    let by_enum = run(Config {
        act_rule: ActRule::FreeEnergy,
        ..Config::default()
    });
    let by_flag = run(Config {
        free_energy_actor: true,
        ..Config::default()
    });
    assert_eq!(
        by_enum, by_flag,
        "act_rule = FreeEnergy and free_energy_actor = true are the same rule"
    );
}

#[test]
fn e_a_pre_1_6_config_json_still_loads() {
    let mut v = serde_json::to_value(Config::default()).unwrap();
    let obj = v.as_object_mut().unwrap();
    // a pre-1.6 config: the deleted softmax lever present, the new keys absent.
    obj.insert("free_energy_softmax".into(), serde_json::Value::Bool(true));
    obj.remove("act_rule");
    obj.remove("act_discount");
    let cfg: Config =
        serde_json::from_value(v.clone()).expect("an old config JSON must still load");
    assert_eq!(
        cfg,
        Config::default(),
        "unknown keys are ignored, absent keys take their defaults"
    );
    // …and inside a whole organism snapshot, where an absent 1.6 key means the organism predates 1.6: it ran the
    // pre-1.6 `ActScore` rule, which is removed, so it acts with the default `Valence` (the legacy genome's rule).
    let mut snap = serde_json::to_value(Field::new(4)).unwrap();
    snap["config"] = v.clone();
    let f: Field = serde_json::from_value(snap.clone()).expect("an old snapshot must still load");
    assert_eq!(f.config().act_rule, ActRule::Valence);
    // A config or snapshot that NAMES the removed rule loads as `Valence` (a serde alias), and its four levers are
    // ignored like every retired key (`substrate_retired_fields`).
    let mut named = v;
    named["act_rule"] = serde_json::Value::String("ActScore".into());
    let cfg: Config = serde_json::from_value(named.clone()).expect("\"ActScore\" must still load");
    assert_eq!(cfg.act_rule, ActRule::Valence);
    snap["config"] = named;
    let f: Field =
        serde_json::from_value(snap).expect("a snapshot naming ActScore must still load");
    assert_eq!(f.config().act_rule, ActRule::Valence);
    assert!(
        serde_json::to_string(&f.config())
            .unwrap()
            .contains("\"act_rule\":\"Valence\""),
        "it re-serialises as the rule it runs"
    );
    // The rule round-trips by name.
    let pinned = serde_json::to_string(&Config {
        act_rule: ActRule::FreeEnergy,
        ..Config::default()
    })
    .unwrap();
    assert!(
        pinned.contains("\"act_rule\":\"FreeEnergy\""),
        "the rule serialises by name: {pinned}"
    );
    let back: Config = serde_json::from_str(&pinned).unwrap();
    assert_eq!(back.act_rule, ActRule::FreeEnergy);
}

/// Fold `n` zero rewards into option `o` alone (`alive(Act)` offered only `o`, reward 0 each time).
fn visit_with_zero(f: &mut Field, o: usize, n: usize) {
    let mut last = None;
    for _ in 0..n {
        let _ = act(f, &[o], last);
        last = Some((o, 0.0));
    }
    let _ = act(f, &[o], last); // fold the n-th
}

#[test]
fn f_the_visit_count_is_exact_over_hundreds_of_zero_reward_visits() {
    // (1) The score never RISES with visits: n visits vs n+1, from 0 to 1000 zero-reward visits.
    let mut f = Field::new(2);
    let (mut strict, mut n_a) = (0usize, 0usize);
    let mut last_a: Option<(usize, f32)> = None;
    for n in 0..1000usize {
        // option 0 visited n times, option 1 visited n+1 times (fold one extra into option 1 on the first round)
        let _ = act(&mut f, &[0], last_a);
        last_a = Some((0, 0.0));
        if n == 0 {
            visit_with_zero(&mut f, 1, 1);
        }
        let _ = act(&mut f, &[1], Some((1, 0.0)));
        n_a += 1;
        let (a, b) = (f.valence_of(0), f.valence_of(1));
        assert!(
            a >= b,
            "a less-visited zero-reward option must never score below a more-visited one ({a:e} < {b:e} at {n_a} vs {} visits)",
            n_a + 1
        );
        if a > b {
            strict += 1;
        }
        if n_a < 300 {
            assert!(
                a > b,
                "while the bonus is representable the less-visited option scores STRICTLY higher ({a:e} vs {b:e} at {n_a} visits)"
            );
        }
    }
    eprintln!("[ACT-RULE F] n vs n+1 zero-reward visits, n ≤ 1000: never inverted, strictly ordered for {strict}/1000 (the bonus (σ0·(1−lr)^n)² underflows f32 past ~320 visits, then they tie)");

    // (2) alive(Act) itself: the reviewer's case, 150 vs 300 zero-reward visits — the less-visited is picked.
    let mut g = Field::new(2);
    visit_with_zero(&mut g, 0, 300);
    visit_with_zero(&mut g, 1, 150);
    let picked = act(&mut g, &[0, 1], None);
    eprintln!(
        "[ACT-RULE F] 300 vs 150 zero-reward visits: valence_of {:e} vs {:e}, alive(Act) picks option {picked}",
        g.valence_of(0),
        g.valence_of(1)
    );
    assert!(g.valence_of(1) > g.valence_of(0));
    assert_eq!(
        picked, 1,
        "alive(Act) must pick the LESS-visited zero-reward option"
    );

    // (3) A flat zero-reward 4-arm bandit: an exact round-robin while the bonus separates the counts (≤ ~300
    //     visits per arm), then uniform (the seeded tie-break) once every bonus has underflowed to 0.
    const K: usize = 4;
    let options: Vec<usize> = (0..K).collect();
    // (odd seeds: `rng_seed` ORs the seed with 1, so 2k and 2k+1 would replay one tie-break stream)
    for seed in [1u64, 3, 5, 7, 9] {
        let mut h = Field::new(K);
        h.rng_seed(seed);
        let mut counts = [0usize; K];
        let mut last = None;
        let mut early = [0usize; K];
        for t in 0..4000 {
            let a = act(&mut h, &options, last);
            counts[a] += 1;
            if t < 1200 {
                early[a] += 1;
            }
            last = Some((a, 0.0));
        }
        eprintln!("[ACT-RULE F] flat zero-reward bandit, seed {seed}: first 1200 picks {early:?}, all 4000 {counts:?}");
        assert!(
            early.iter().all(|&c| c == 300),
            "an exact visit count makes a flat bandit an exact round-robin: {early:?}"
        );
        for &c in &counts {
            assert!(
                (c as f32 - 1000.0).abs() < 100.0,
                "past the bonus underflow the ties are broken uniformly: {counts:?}"
            );
        }
    }
}

#[test]
fn g_an_ensemble_snapshot_migrates_to_the_valence_record() {
    const ZERO_VISITS: usize = 200;
    const PAID_VISITS: usize = 30;
    // A FreeEnergy organism: its values live in the 24-member ensemble. Option 0 paid 0 two hundred times (its
    // ensemble mean is the f32 prior residue, not 0), option 1 paid 0.6 thirty times, option 2 was offered, never taken.
    let mut fe = Field::new(3);
    fe.config_mut().act_rule = ActRule::FreeEnergy;
    visit_with_zero(&mut fe, 0, ZERO_VISITS);
    let mut last = None;
    for _ in 0..PAID_VISITS {
        let _ = act(&mut fe, &[1, 2], last);
        last = Some((1, 0.6));
    }
    let _ = act(&mut fe, &[1, 2], last);
    // The twin a fresh Valence organism grows from the same history.
    let mut twin = Field::new(3);
    visit_with_zero(&mut twin, 0, ZERO_VISITS);
    let mut last = None;
    for _ in 0..PAID_VISITS {
        let _ = act(&mut twin, &[1], last);
        last = Some((1, 0.6));
    }
    let _ = act(&mut twin, &[1], last);

    // A pre-1.6 snapshot: no `valence` map, the old `valence_actor` flag selecting the rule.
    let mut snap = serde_json::to_value(&fe).unwrap();
    snap.as_object_mut().unwrap().remove("valence");
    snap["config"]["act_rule"] = serde_json::Value::String("ActScore".into());
    snap["config"]["valence_actor"] = serde_json::Value::Bool(true);
    let old: Field = serde_json::from_value(snap).expect("a pre-1.6 valence snapshot must load");
    assert_eq!(old.config().effective_act_rule(), ActRule::Valence);
    for o in 0..3 {
        eprintln!(
            "[ACT-RULE G] option {o}: migrated valence_of {:e}, fresh-valence twin {:e}",
            old.valence_of(o),
            twin.valence_of(o)
        );
    }
    assert_eq!(
        old.valence_of(0).to_bits(),
        twin.valence_of(0).to_bits(),
        "a zero-reward history migrates to μ = 0 and its EXACT visit count (bit-equal to the twin)"
    );
    assert!(
        (old.valence_of(1) - twin.valence_of(1)).abs() < 1e-5,
        "a paid history migrates to its μ and visit count"
    );
    assert_eq!(
        old.valence_of(2).to_bits(),
        twin.valence_of(2).to_bits(),
        "an offered-but-never-taken option migrates as untried (σ0²)"
    );
    // …and the migrated organism keeps acting like the twin: the never-taken option first, then the less-visited.
    let mut old = old;
    assert_eq!(act(&mut old, &[0, 2], None), 2, "the untried option first");
    let picked = act(&mut old, &[0, 2], Some((2, 0.0)));
    assert_eq!(
        picked, 2,
        "then the less-visited zero-reward option (1 visit vs 200), not the one carrying a residue"
    );
}

#[test]
fn h_a_trained_valence_organism_survives_snapshot_load() {
    let pay = |a: usize, t: usize| {
        if (a + t).is_multiple_of(3) {
            0.8
        } else {
            0.1 * a as f32
        }
    };
    let options = [0usize, 1, 2, 3];
    let run = |f: &mut Field, from: usize, last: &mut Option<(usize, f32)>| -> Vec<(usize, u32)> {
        (from..from + 200)
            .map(|t| {
                let a = act(f, &options, *last);
                *last = Some((a, pay(a, t)));
                (a, f.valence_of(a).to_bits())
            })
            .collect()
    };
    let mut f = Field::new(4);
    f.rng_seed(5);
    let mut last = None;
    let _ = run(&mut f, 0, &mut last);
    let json = serde_json::to_string(&f).unwrap();
    let mut g: Field = serde_json::from_str(&json).unwrap();
    assert_eq!(
        serde_json::to_string(&g).unwrap(),
        json,
        "a load→save round-trip is byte-stable"
    );
    let (mut last_f, mut last_g) = (last, last);
    let cont_f = run(&mut f, 200, &mut last_f);
    let cont_g = run(&mut g, 200, &mut last_g);
    assert_eq!(
        cont_f, cont_g,
        "the reloaded organism continues exactly like the original (choices and valences, bit for bit)"
    );
}
