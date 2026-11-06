// SPDX-License-Identifier: MIT-0
//! FRESH AFTER LEARNING — a unit whose PARAMETERS change outside of firing (the delta rule, feature-direction
//! learning, nested credit, mutation, heredity, a prior/scale setter) must RE-FIRE on the next settle, even when
//! none of the cells it reads changed.
//!
//! Fire-on-change fires a unit only when a READ changes, and a weight update is not a read change — so before the
//! fix the cell a learned unit writes kept the value computed under the OLD weights until the input happened to
//! move. Repeated `learn_step` on an UNCHANGED input therefore read the SAME stale prediction every event: the
//! error was a frozen accumulator and the weight overshot (1 unit, target 0.5, 60 reps: output stuck at 0.0997,
//! reported error stuck at 0.4003, weight 0.1 → 2.48, true output after a real re-settle 0.9999). The fix is a
//! transient per-unit DIRTY flag the settle drains once its natural cascade is quiescent.
//!
//! The ground truth for "fresh" is the organism's OWN settle: perturb every input and restore it on a clone, so
//! every unit fires naturally under its current parameters. A fresh organism settles on an unchanged input to
//! exactly that state; a stale one does not.
//!
//! Two rules keep the refresh from leaking beyond the stale cells it corrects: it never overwrites a cell the caller
//! supplied to that settle (a teacher-forced or committed value holds, as it did before the fix), and it is not an
//! event of subjective time (the clock — and so every trace, recency and grace window measured in firings — keeps
//! the pace it had before the fix, whether or not an input repeats).

use openmorph_core::Field;

/// Settle `x` on a CLONE after first perturbing every input, so every unit re-fires under its current parameters —
/// the fresh cell state an unchanged-input settle must reproduce.
fn fresh_cells(f: &Field, x: &[(usize, f32)]) -> Vec<f32> {
    let mut g = f.clone();
    let bumped: Vec<(usize, f32)> = x.iter().map(|&(c, v)| (c, v + 0.5)).collect();
    g.stimulate(&bumped, 100_000);
    g.stimulate(x, 100_000);
    g.cells().to_vec()
}

/// Settle `f` on the (unchanged) input `x` and return the largest deviation of any cell from the fresh state.
fn staleness(f: &mut Field, x: &[(usize, f32)]) -> f32 {
    let truth = fresh_cells(f, x);
    f.stimulate(x, 100_000);
    f.cells()
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// A small grown organism (XOR over cells 0,1 → readout cell 2, with hidden features), settled on `x` so every
/// unit has fired under its current parameters.
fn grown(feature_lr: f32, x: &[(usize, f32)]) -> Field {
    let mut f = Field::new(3);
    f.rng_seed(7);
    f.config_mut().feature_lr = feature_lr;
    for t in 0..200 {
        let a = if t % 2 == 0 { 1.0 } else { -1.0 };
        let b = if (t / 2) % 2 == 0 { 1.0 } else { -1.0 };
        f.grow(&[(0, a), (1, b)], 2, 0.8 * a * b, 0.1, 200);
    }
    f.stimulate(x, 100_000);
    assert!(
        f.n_units() >= 3,
        "the XOR grow must have born hidden features"
    );
    f
}

const X: [(usize, f32); 2] = [(0, 0.7), (1, -0.4)];
const TOL: f32 = 1e-3;

#[test]
fn repeated_learning_on_an_unchanged_input_converges_instead_of_overshooting() {
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![0.1], 0.0, 1);
    let mut err = 0.0;
    for _ in 0..60 {
        err = f.learn_step(&[(0, 1.0)], 1, 0.5, 0.1, 50);
    }
    let out = f.cells()[1];
    println!(
        "60x learn_step on one input: reported err {err:.4}, output {out:.4}, weight {:.3}",
        f.units[0].weights[0]
    );
    assert!(
        err < 0.01,
        "the error must track the live output, not freeze ({err})"
    );
    assert!(
        (out - 0.5).abs() < 0.01,
        "the output must reach the target ({out})"
    );
    let s = staleness(&mut f, &[(0, 1.0)]);
    assert!(
        s < TOL,
        "the written output must be the fresh one (staleness {s})"
    );
}

#[test]
fn repeated_grow_on_an_unchanged_input_converges() {
    // A static screen presented N times — the perception `WorldModel` regime ("learns only ONCE" before the fix).
    let mut f = Field::new(2);
    let mut first = 0.0;
    let mut last = 0.0;
    for k in 0..200 {
        let (e, _) = f.grow(&[(0, 1.0)], 1, 0.5, 0.1, 200);
        if k == 0 {
            first = e;
        }
        last = e;
    }
    let truth = fresh_cells(&f, &[(0, 1.0)]);
    f.stimulate(&[(0, 1.0)], 100_000);
    let out_stale = (f.cells()[1] - truth[1]).abs();
    // NOT fixed here (reported): a feature born on an unchanged input has never fired, so its cell still reads 0
    // until an input it reads changes. The readout's weight on it stays 0 meanwhile (its input is 0), so the
    // readout is unaffected — the newborn is just idle capacity until the input moves.
    let newborn_idle = f
        .cells()
        .iter()
        .zip(&truth)
        .filter(|(a, b)| (*a - *b).abs() >= TOL)
        .count();
    println!(
        "200x grow on one input: err {first:.4} → {last:.4}; output staleness {out_stale:.2e}; \
         {newborn_idle} of {} units never fired (newborn, idle)",
        f.n_units()
    );
    assert!(
        last < 0.01,
        "repeated growth must converge ({first} → {last})"
    );
    assert!(
        out_stale < TOL,
        "the grown output must be fresh (staleness {out_stale})"
    );
}

#[test]
fn every_parameter_mutation_is_re_fired_on_an_unchanged_input() {
    // Each mutator changes a unit's parameters outside of firing; an unchanged-input settle must re-fire it.
    type Mutate = fn(&mut Field) -> Field;
    let cases: [(&str, f32, Mutate); 7] = [
        ("learn_step", 0.0, |f| {
            f.learn_step(&X, 2, -0.9, 0.5, 100_000);
            f.clone()
        }),
        ("feature_lr", 0.5, |f| {
            f.learn_step(&X, 2, -0.9, 0.5, 100_000);
            f.clone()
        }),
        ("set_scale", 0.0, |f| {
            let h = f.units[1].write;
            f.set_scale(h, 3.0);
            f.clone()
        }),
        ("resample_feature", 0.0, |f| {
            let h = f.units[1].write;
            f.resample_feature(h, 1.0);
            f.clone()
        }),
        ("reproduce", 0.0, |f| {
            let (child, parent) = (f.units[1].write, f.units[2].write);
            f.reproduce(child, parent);
            f.clone()
        }),
        ("seed_writer_priors", 0.0, |f| {
            f.seed_writer_priors(2, 1, 0.8);
            f.clone()
        }),
        ("fork_mutated", 0.0, |f| f.fork_mutated()),
    ];
    let mut stale = Vec::new();
    for (name, feature_lr, mutate) in cases {
        let mut f = grown(feature_lr, &X);
        assert!(staleness(&mut f, &X) < TOL, "{name}: fresh before");
        let mut g = mutate(&mut f);
        let s = staleness(&mut g, &X);
        println!("{name}: staleness after an unchanged-input settle {s:.2e}");
        if s >= TOL {
            stale.push(name);
        }
    }
    assert!(
        stale.is_empty(),
        "mutated units were not re-fired: {stale:?}"
    );
}

#[test]
fn nested_credit_re_fires_the_tuned_nested_cell() {
    // A frozen brick retuned by the composed error (`nest_credit`) computes a new function; its host cell must
    // follow on an unchanged input (its memo is cleared, but nothing re-fired it).
    let mut sub = Field::new(3);
    sub.add_unit(vec![0, 1], vec![0.3, -0.2], 0.0, 2);
    let mut host = Field::new(4);
    host.config_mut().nest_credit = true;
    host.nest(sub, &[0, 1], 2, 50);
    host.add_unit(vec![2], vec![1.0], 0.0, 3);
    for _ in 0..30 {
        host.learn_step(&X, 3, 0.9, 0.2, 1_000);
    }
    let s = staleness(&mut host, &X);
    println!("nest_credit: staleness after an unchanged-input settle {s:.2e}");
    assert!(
        s < TOL,
        "the tuned nested cell must re-fire (staleness {s})"
    );
}

#[test]
fn a_refresh_never_overwrites_a_cell_the_caller_supplies() {
    // unit 0: cell 0 → cell 1; unit 1: cell 1 → cell 2. Learn unit 0, then settle the SAME input while the caller
    // clamps cell 1 (teacher forcing / a committed decision).
    let mut f = Field::new(3);
    f.add_unit(vec![0], vec![0.5], 0.0, 1);
    f.add_unit(vec![1], vec![1.0], 0.0, 2);
    f.stimulate(&[(0, 1.0)], 50);
    f.learn_step(&[(0, 1.0)], 1, 0.9, 0.1, 50); // unit 0 is now dirty
    f.stimulate(&[(0, 1.0), (1, -1.0)], 50);
    let (held, reader) = (f.cells()[1], f.cells()[2]);
    println!("clamped settle: cell1 {held:.4} (supplied -1), reader cell2 {reader:.4}");
    assert_eq!(held, -1.0, "the refresh overwrote the supplied cell");
    assert_eq!(
        reader,
        (-1.0f32).tanh(),
        "the reader must see the supplied value"
    );
    // Released: the next settle that does not supply cell 1 refreshes unit 0 — it waited, still stale.
    let s = staleness(&mut f, &[(0, 1.0)]);
    println!("released settle: staleness {s:.2e}");
    assert!(
        s < TOL,
        "the held unit must refresh once released (staleness {s})"
    );

    // The same through a CASCADE: a refreshed feature re-drives the writer of the held cell, which must wait too.
    let mut g = Field::new(4);
    g.add_unit(vec![0], vec![0.8], 0.0, 3); // feature: cell 0 → cell 3
    g.add_unit(vec![3], vec![1.0], 0.0, 1); // writer of the held cell: cell 3 → cell 1
    g.add_unit(vec![1], vec![1.0], 0.0, 2); // reader of the held cell
    g.stimulate(&[(0, 1.0)], 50);
    g.set_scale(3, 2.0); // the feature's gain changes → dirty
    g.stimulate(&[(0, 1.0), (1, -1.0)], 50);
    println!(
        "cascade: feature {:.4} (fresh {:.4}), held cell1 {:.4}, reader {:.4}",
        g.cells()[3],
        (1.6f32).tanh(),
        g.cells()[1],
        g.cells()[2]
    );
    assert_eq!(g.cells()[3], (1.6f32).tanh(), "the feature must refresh");
    assert_eq!(
        g.cells()[1],
        -1.0,
        "the refresh cascade overwrote the supplied cell"
    );
    assert_eq!(g.cells()[2], (-1.0f32).tanh());
    let s = staleness(&mut g, &[(0, 1.0)]);
    assert!(
        s < TOL,
        "released, the waiting writer must refresh (staleness {s})"
    );
}

#[test]
fn a_refresh_takes_no_subjective_time() {
    // A static screen, learned on 40 times: the output converges (it is refreshed every event), but no input
    // changed, so no subjective time passes — the clock and a trace over the learned output stay where the last
    // input-driven settle left them, exactly as before the fix.
    let mut f = Field::new(2);
    f.add_unit(vec![0], vec![0.1], 0.0, 1);
    let tr = f.integrate(1, 4.0);
    f.stimulate(&[(0, 1.0)], 50);
    f.stimulate(&[(0, 1.0)], 50); // the trace consumes the one firing of the settle above
    let (clock, trace) = (f.clock(), f.cells()[tr]);
    let mut err = 1.0;
    for _ in 0..40 {
        err = f.learn_step(&[(0, 1.0)], 1, 0.5, 0.1, 50);
    }
    println!(
        "40 learns on one input: err {err:.4}, clock {clock} → {}, trace {trace:.4} → {:.4}",
        f.clock(),
        f.cells()[tr]
    );
    assert!(err < 0.01, "the refreshed output must converge ({err})");
    assert_eq!(
        f.clock(),
        clock,
        "a refresh must not advance the subjective clock"
    );
    assert_eq!(
        f.cells()[tr],
        trace,
        "a trace must not relax on refresh-only time"
    );
}

/// A refresh PENDING at snapshot time crosses the load: learn on an input (the writer is now stale), snapshot, load,
/// and keep learning on the SAME input — the reloaded organism must re-fire the stale writer exactly as the one that
/// stayed in memory, bit for bit (the stale flags are part of the snapshot, like every other piece of dynamics
/// state). Without them the reloaded copy read the pre-update output once and diverged (0.34839 vs 0.31961).
#[test]
fn a_pending_refresh_survives_a_snapshot_round_trip() {
    let mut live = Field::new(2);
    live.add_unit(vec![0], vec![0.1], 0.0, 1);
    let x = [(0, 1.0f32)];
    for _ in 0..3 {
        live.learn_step(&x, 1, 0.5, 0.1, 50); // the last update leaves the writer stale
    }
    let json = serde_json::to_vec(&live).expect("Field serializes (JSON)");
    let mut loaded: Field = serde_json::from_slice(&json).expect("Field loads (JSON)");
    for step in 0..5 {
        let a = live.learn_step(&x, 1, 0.5, 0.1, 50);
        let b = loaded.learn_step(&x, 1, 0.5, 0.1, 50);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "step {step}: error differs after reload"
        );
        assert_eq!(
            live.cells()[1].to_bits(),
            loaded.cells()[1].to_bits(),
            "step {step}: output differs after reload"
        );
        assert_eq!(
            live.units[0].weights[0].to_bits(),
            loaded.units[0].weights[0].to_bits(),
            "step {step}: weight differs after reload"
        );
    }
}
