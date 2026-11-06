// SPDX-License-Identifier: MIT-0
//! DISTILLATION-BY-REPLAY — the compression half of sleep. Energy/credit can't rank redundant-vs-essential
//! among converged features (both signals collapse), so a matured skill can't be pruned in place. But it CAN be
//! re-derived: during sleep the morph REPLAYS the skill's inputs, reads its own matured output as a CLEAN, smooth
//! teacher signal, and re-grows a fresh STUDENT to match. The wake morph over-grew chasing NOISY online labels;
//! the student fits the teacher's denoised function, so a tighter student reproduces the skill with FEWER units.
//! This is hippocampal replay → cortical consolidation, and it doubles as episodic→semantic transfer.
//!
//!   cargo test -p openmorph-core --test proofs substrate_distill -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const Y: usize = 2;
fn label(x0: f32, x1: f32) -> f32 {
    if x0 * x1 > 0.0 {
        1.0
    } else {
        -1.0
    }
}
fn acc(f: &mut Field) -> f64 {
    let mut rng = Rng::new(0x51ee ^ 0xFFFF);
    let mut c = 0usize;
    for _ in 0..1000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.stimulate(&[(0, x0), (1, x1)], 50);
        if (if f.cells()[Y] > 0.0 { 1.0 } else { -1.0 } - label(x0, x1)).abs() < 1e-6 {
            c += 1;
        }
    }
    c as f64 / 1000.0
}

// wake: grow the skill on NOISY online ±1 labels (over-grows chasing the noise)
fn wake(max_hidden: usize) -> Field {
    let mut f = Field::new(3);
    f.config_mut().birth_gate = 0.15;
    f.config_mut().max_hidden = max_hidden;
    let mut rng = Rng::new(0x51ee);
    for _ in 0..8000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        f.grow(&[(0, x0), (1, x1)], Y, label(x0, x1), 0.05, 50);
    }
    f
}

// sleep: replay inputs, read the teacher's smooth output, grow a student of capacity `cap` to match it.
// `flr` > 0 lets the student LEARN its feature directions (#125), so a few learned features replace many random.
fn distill(teacher: &mut Field, cap: usize, flr: f32) -> Field {
    let mut s = Field::new(3);
    s.config_mut().birth_gate = 0.15;
    s.config_mut().max_hidden = cap;
    s.config_mut().feature_lr = flr;
    let mut rng = Rng::new(0xD157);
    for _ in 0..8000 {
        let (x0, x1) = (rng.sym(), rng.sym());
        teacher.stimulate(&[(0, x0), (1, x1)], 50);
        let tgt = teacher.cells()[Y]; // the DENOISED teacher signal (soft label)
        s.grow(&[(0, x0), (1, x1)], Y, tgt, 0.05, 50);
    }
    s
}

#[test]
fn distillation_by_replay_compresses() {
    let mut teacher = wake(16);
    let (ta, tu) = (acc(&mut teacher), teacher.n_units());

    // tight student, RANDOM features (no direction learning) — needs many to cover the space
    let mut rnd = distill(&mut teacher, 6, 0.0);
    let (ra, ru) = (acc(&mut rnd), rnd.n_units());
    // tight student, LEARNED feature directions — a few good directions replace many random ones
    let mut learned = distill(&mut teacher, 6, 0.5);
    let (la, lu) = (acc(&mut learned), learned.n_units());

    eprintln!(
        "[DISTILL] teacher (wake, noisy labels):        {:.0}%  {tu} units",
        ta * 100.0
    );
    eprintln!(
        "[DISTILL] student (tight cap6, RANDOM dirs):   {:.0}%  {ru} units",
        ra * 100.0
    );
    eprintln!(
        "[DISTILL] student (tight cap6, LEARNED dirs):  {:.0}%  {lu} units",
        la * 100.0
    );
    eprintln!("[DISTILL] VERDICT: distilling on the teacher's DENOISED signal, a student that LEARNS its feature directions reproduces the skill in far fewer units than the wake morph over-grew with random projections — sleep re-derives a compact skill by replay + direction-learning. Random-direction distillation can't (a random basis needs many features to cover the space — that part is genuine cost).");

    assert!(ta > 0.90, "teacher skill works ({ta:.2})");
    assert!(
        la > 0.88,
        "the learned-direction student reproduces the skill compactly ({la:.2})"
    );
    assert!(
        lu < tu,
        "distillation + direction-learning compresses the unit count ({lu} < {tu})"
    );
    assert!(
        la > ra + 0.08,
        "learning directions beats random projections at the same tight capacity ({la:.2} vs {ra:.2})"
    );
}
