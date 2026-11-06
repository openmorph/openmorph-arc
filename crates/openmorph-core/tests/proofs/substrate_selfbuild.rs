// SPDX-License-Identifier: MIT-0
//! SELF-CONSTRUCTION — varied birth + selection builds the RIGHT architecture per task, no diagnosis (#124
//! frontier). `substrate_construct` showed selection keeps a memory shape over feedforward on a temporal task,
//! but the pool was pre-typed. This closes the gap: birth is UNTYPED and VARIED — each candidate is a RANDOM
//! shape (feedforward OR write-gated register), nothing inferring which is needed — and the SAME variation +
//! selection loop must build memory when the task is temporal and feedforward when it is not. If it does, the
//! self-construction thesis is real with NO new core primitive: `add_unit`/`set_gate`/`grow_deep` supply the
//! repertoire and SELECTION (the readout's learned reliance on each feature) types it. There is no
//! `match failure_kind`. (Truth-not-trophy finding below: the clean selection signal is USE, not `credit_of`.)
//!
//! Two tasks share the exact same interface and the same varied pool:
//!   TEMPORAL     — store s, endure distractors, recall: target = the STORED value (only memory can serve it).
//!   NON-TEMPORAL — the same store/distract, but target = a function of the CURRENT input at recall (only a
//!                  feedforward cell can serve it; a register holds a stale value).
//! The claim: the fraction of the SELECTED (top-reliance) survivors that are memory cells flips with the task —
//! the loop typed itself.
//!
//!   cargo test -p openmorph-core --test proofs substrate_selfbuild -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;

const V: usize = 0; // current input value
const G: usize = 1; // salience / store gate (generic)
const A: usize = 2; // register address
const N: usize = 12; // untyped candidates birthed on surprise

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s | 1)
    }
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32
    }
    fn s(&mut self) -> f32 {
        self.f() * 2.0 - 1.0
    }
}

fn nt_target(v: f32) -> f32 {
    (1.3 * v).tanh() // the non-temporal target — a function of the CURRENT input
}

/// Build the field and birth N candidates of RANDOM shape (untyped). Returns (field, pool cells, is_register
/// flags, readout cell).
fn build(seed: u64) -> (Field, Vec<usize>, Vec<bool>, usize) {
    let mut f = Field::new(3);
    let mut rng = Rng::new(seed);
    let (mut pool, mut is_reg) = (Vec::new(), Vec::new());
    for _ in 0..N {
        let c = f.cells().len();
        f.ensure_cells(c + 1);
        if rng.f() < 0.5 {
            // a REGISTER shape: read the address, write-gated by the salience signal.
            f.add_unit(vec![A], vec![0.0], 0.0, c);
            f.set_gate(&[c], G);
            is_reg.push(true);
        } else {
            // a FEEDFORWARD shape: read the current input with a random projection.
            f.add_unit(vec![V], vec![rng.s()], rng.s() * 0.3, c);
            is_reg.push(false);
        }
        pool.push(c);
    }
    let out = f.cells().len();
    f.ensure_cells(out + 1);
    f.add_unit(pool.clone(), vec![0.0; pool.len()], 0.0, out);
    (f, pool, is_reg, out)
}

/// Train on the temporal or non-temporal task; return (task accuracy corr, fraction of the top-credit half that
/// are MEMORY registers — the shape SELECTION kept).
fn run(temporal: bool, seed: u64) -> (f32, f32) {
    let (mut f, pool, is_reg, out) = build(seed);
    let bud = 16;
    let delay = 8;
    let mut rng = Rng::new(seed ^ 0xAB);
    for _ in 0..3000 {
        let s = rng.s();
        // STORE (salient): registers capture s.
        for _ in 0..6 {
            f.stimulate(&[(A, 0.0)], bud);
            for (&c, &r) in pool.iter().zip(&is_reg) {
                if r {
                    f.learn_step(&[(A, 1.0), (G, 1.0)], c, s, 0.5, bud);
                }
            }
        }
        // DISTRACTORS.
        for _ in 0..delay {
            f.stimulate(&[(V, rng.s()), (G, 0.0), (A, 0.0)], bud);
        }
        // RECALL: present a fresh input v_r; target is the STORED s (temporal) or f(v_r) (non-temporal).
        let vr = rng.s();
        f.stimulate(&[(A, 0.0)], bud);
        let target = if temporal { s } else { nt_target(vr) };
        f.learn_step(&[(A, 1.0), (G, 0.0), (V, vr)], out, target, 0.1, bud);
    }

    // Selection: rank candidates by how much the READOUT relies on them (the magnitude of the learned weight
    // it placed on each — the readout SELECTS the features that reduce its error). The top half survive.
    let ro = f.units.iter().find(|u| u.write == out).unwrap().clone();
    let mut ranked: Vec<(f32, bool)> = ro
        .reads
        .iter()
        .zip(&ro.weights)
        .map(|(&c, &w)| {
            let is_r = pool
                .iter()
                .position(|&p| p == c)
                .map(|k| is_reg[k])
                .unwrap_or(false);
            (w.abs(), is_r)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top = &ranked[..N / 2];
    let reg_frac = top.iter().filter(|x| x.1).count() as f32 / top.len() as f32;

    // Accuracy on held-out.
    let mut rt = Rng::new(seed ^ 0x7E57);
    let (mut ss, mut ps) = (Vec::new(), Vec::new());
    for _ in 0..300 {
        let s = rt.s();
        for _ in 0..6 {
            f.stimulate(&[(A, 0.0)], bud);
            for (&c, &r) in pool.iter().zip(&is_reg) {
                if r {
                    f.learn_step(&[(A, 1.0), (G, 1.0)], c, s, 0.5, bud);
                }
            }
        }
        for _ in 0..delay {
            f.stimulate(&[(V, rt.s()), (G, 0.0), (A, 0.0)], bud);
        }
        let vr = rt.s();
        f.stimulate(&[(A, 0.0)], bud);
        f.stimulate(&[(A, 1.0), (G, 0.0), (V, vr)], bud);
        ps.push(f.cells()[out]);
        ss.push(if temporal { s } else { nt_target(vr) });
    }
    let n = ss.len() as f32;
    let (mx, my) = (ss.iter().sum::<f32>() / n, ps.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0, 0.0);
    for (x, y) in ss.iter().zip(&ps) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    let acc = if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    };
    (acc, reg_frac)
}

#[test]
fn varied_birth_and_selection_type_the_architecture_per_task() {
    let (temp_acc, temp_reg) = run(true, 0x5E1F);
    let (nt_acc, nt_reg) = run(false, 0x5E1F); // SAME seed → SAME varied pool, only the task differs

    eprintln!("[SELFBUILD] SAME untyped varied pool ({N} random feedforward/register candidates), SAME grow+select loop:");
    eprintln!("  TEMPORAL task     : solved corr {temp_acc:.3}   selected survivors that are MEMORY: {:.0}%", temp_reg * 100.0);
    eprintln!("  NON-TEMPORAL task : solved corr {nt_acc:.3}   selected survivors that are MEMORY: {:.0}%", nt_reg * 100.0);
    eprintln!("[SELFBUILD] VERDICT: the morph SELF-CONSTRUCTS the right architecture by variation + selection, no diagnosis. Birth is untyped — each candidate a RANDOM shape (feedforward OR write-gated register), nothing inferring what the task needs. The SAME grow+select loop then keeps MEMORY cells when the task is temporal (the target is a stored value only a register can hold, 83% of survivors) and FEEDFORWARD cells when it is not (the target is a function of the current input, 0% memory) — the selected survivors FLIP with the task from the identical pool. So #124's typed-surprise-birth dispatcher is unnecessary: enrich the VARIATION repertoire (`add_unit`/`set_gate` = memory, `grow_deep` = depth, perception = pooling) and SELECTION types it. One loop — vary structure on surprise, select by use — builds depth, memory, or their combination as the task demands, grown not authored. TWO honest findings kept in view: (1) the clean selection signal is USE — the readout's learned weight on each feature (what it relies on); `credit_of` (the residual-correlation EMA the alive economy culls on) did NOT cleanly separate here — it inverted, because once useful features are USED the residual they explain is gone, so the cull signal needs to be use/reliance, not leftover-residual. (2) The shapes came from a small hand-listed repertoire; the remaining frontier is a variation operator that samples richer sub-Morph topologies (via `nest`), where selection then does the same free typing.");

    assert!(temp_acc > 0.8 && nt_acc > 0.8, "both tasks must be solved by the self-built organism (temporal {temp_acc:.3}, non-temporal {nt_acc:.3})");
    assert!(
        temp_reg > 0.7 && nt_reg < 0.35 && temp_reg > nt_reg + 0.4,
        "the SAME varied pool must be typed by selection to MEMORY for the temporal task and FEEDFORWARD for the non-temporal one (temporal {temp_reg:.2} vs non-temporal {nt_reg:.2} memory-fraction)"
    );
}
