// SPDX-License-Identifier: MIT-0
//! BEHAVIOURAL SUB-MORPH BOUNDARIES — experiment 2 of `docs/research/figueiredo-amplitude-geometry.md` (§4).
//!
//! CLAIM UNDER TEST. `substrate_decompose` hands the module groups to the organism (`co_activate(&[0,1,4])`,
//! `co_activate(&[2,3,5])`): the decomposition is partly authored. Can the organism find its own module boundaries
//! from BEHAVIOUR alone — where the cross-terms of its input→output map vanish (interaction detection, the honest
//! reading of the hidden-zeros/splits link; not a physics claim)?
//!
//! SIX ROUTES, one truth (cells {0,1,4} = M1, {2,3,5} = M2), 10 seeds each, on four fields:
//!   (a) AUTHORED  — the `substrate_decompose` protocol: `co_activate` the true groups, `spread_positions`, `cluster`.
//!   (b) AFFINITY  — the same `cluster` route, but the concept-map is shaped only by the WIRING: every unit
//!       `co_activate`s its write with its strong reads (|w| ≥ 0.1·max|w|). Nothing authored. This comparison was
//!       CONSTRUCTED for this experiment (the repo has no weight-affinity grouping; only `cluster` exists), so beating
//!       it is a weak claim.
//!   (c) INTERACTION — finite-difference cross-differences |ΔᵢΔⱼ f_o|/h² of each OUTPUT cell (h = 0.05, median over
//!       500 base points in [-1,1]⁴); an input edge when the strongest output's median exceeds 0.1 × the largest
//!       pair's; connected components = proposed sub-morphs; each output joins the component it is most sensitive to.
//!       (Pre-activations would be the note's first choice, but the decompose writers are LINEAR in their inputs,
//!       so their pre-activations have zero cross-differences WITHIN a module too; each output is its own module's
//!       tanh, so reading the outputs couples only within a module.)
//!   (d) SPLIT — the split-style variant over coordinate subspaces: for each of the 7 input bipartitions (A|B),
//!       the 16×16 matrix M[p][q] = f_o(x_A^p ⊕ x_B^q); the cut "splits" F when σ₂/σ₁ < 1e-3 for every output
//!       (the note's criterion). Inputs are linked unless some split cut separates them.
//!   (e) SENSITIVITY — the skeptic's first-order baseline: each output is its own module and each input joins the
//!       output it has the largest median first-order sensitivity |Δᵢ f_o|/h to (same probe points; no second
//!       differences, no threshold).
//!   (f) SURPRISE — `Field::group_by_surprise`, the repo's own non-authored (causal) grouping: on a clone with
//!       `surprise_currency` on, per output warm its end-to-end sensitivity (600 `learn_step`s at lr 0), run 2000
//!       `group_by_surprise` steps over {inputs, that output} at rate 0.5 (`spread_positions` every 100), then
//!       `cluster` the six cells into two.
//! Every partition is then realised: `encapsulate` each component's outputs and `nest` it in a fresh host that feeds
//! it ONLY its component's inputs (any other source the sub reads is grounded at 0) — the nested outputs must match
//! the original field's to 1e-6. Probing and the exactness check run on a clone with `settle_eps = 0` (the shipped
//! 1e-4 propagation dead-band is larger than both a second difference at h = 0.05 and the 1e-6 bar; the default-eps
//! nesting error is reported alongside).
//!
//! FIELDS. `decompose` = the proof's field (linear writers, `max_hidden = 0`). `grown` = a GROWN two-module task:
//! both targets (M1 = a·b, M2 = tanh(c+d)) are grown with `grow` over ALL FOUR inputs, so the writer and every
//! surprise-born random-projection feature read a, b, c, d. `backbone` = a shared `grow_backbone` (24 features over all
//! four inputs) with two delta-rule readouts (M1 = a·b, M2 = tanh(c+d)). `additive` = the grown protocol with an
//! ADDITIVE second module, M2 = 0.5·(c+d) (no within-module interaction; the skeptic's extra task).
//!
//! MEASURED (10 seeds per field; ARI / exact recovery / max |nested − field| at settle_eps 0):
//!   decompose: AUTHORED 1.00 10/10 · AFFINITY 1.00 10/10 · INTERACTION 1.00 10/10 (0/40 false edges) · SPLIT 1.00
//!              10/10 (true-cut σ₂/σ₁ ≈ 8e-9). Every route nests EXACTLY (max err 0, also at the default settle_eps).
//!   grown:     AUTHORED 1.00 10/10 · AFFINITY −0.04 0/10 (purity 0.63) · INTERACTION 0.90 9/10 (purity 0.95; 1/40
//!              false edges, 0/20 missed) · SPLIT 0.00 0/10 (true-cut σ₂/σ₁ median 4.9e-2 — 49× the 1e-3 bar; the
//!              true cut is even the lowest-ratio cut in only 7/10 seeds).
//!   backbone:  AUTHORED 1.00 10/10 · AFFINITY −0.11 0/10 (purity 0.58) · INTERACTION 1.00 10/10 (0/40 false edges) ·
//!              SPLIT 0.00 0/10 (true-cut σ₂/σ₁ median 2.9e-2; lowest-ratio cut in 9/10).
//!   Nesting on the grown fields: NO route's boundary is exact — the authored (= the interaction) partition nests with
//!   max |nested − field| 0.124 (grown) / 0.130 (backbone), because every grown feature reads all four inputs and the
//!   outputs carry a first-order leak (out-of-module / in-module sensitivity median 0.073 grown, 0.040 backbone). The
//!   cut is functionally free, though: the nested sub-morphs' max error against the TASK is 0.331 / 0.239 vs the
//!   original field's 0.331 / 0.258.
//!   Margins: the one grown miss (seed 3) is a cross pair at exactly 0.100 of the largest interaction against the
//!   pre-registered 0.1 threshold (grown median 0.072, backbone 0.048); threshold-free, the strongest cross pair is
//!   below the weakest within pair in 30/30 fields (within ≥ 0.54, cross ≤ 0.077 on the grown fields).
//!   SENSITIVITY (e): 10/10 on EVERY field (40/40, ARI 1.00), including `additive`, where INTERACTION drops to 7/10
//!   (ARI 0.70; 3/40 false edges). SURPRISE (f): decompose 0/10 (ARI 0.00), grown 6/10 (0.66), backbone 4/10
//!   (0.59), additive 7/10 (0.73). Its decompose failure follows from the code path: an input with exactly zero
//!   sensitivity to the warmed output falls back to `Σ w²·Var` over ALL its readers, so the other module's inputs
//!   look load-bearing. Three other step/rate settings (60 @ 1.0, 200 @ 0.2, 20 @ 1.0) were no better (≤ 7/10 on
//!   every field, 0/10 on decompose).
//! VERDICT: PARTIAL, leaning negative. The interaction probe is a non-authored boundary detector that recovers the
//! modules in 29/30 of the original fields, and the grown random-projection backbone does not make its graph dense
//! (1/80 false edges over the grown and backbone fields). But the plain first-order sensitivity baseline recovers 40/40, including the additive
//! task where the probe fails 3/10, so THIS benchmark cannot show any value of second-order interaction detection
//! over sensitivity: with one output per module, "which output does this input drive" already names the module.
//! A fair test needs modules that sensitivity cannot separate: outputs shared across modules (an output that reads
//! both), or more modules than outputs. The weight-affinity baseline it beats was constructed here; the repo's own
//! `group_by_surprise` does worse than both. The split-style rank-1 variant FAILS at the note's 1e-3 bar on every
//! grown field, and the EXACTNESS half fails on grown fields for every route that finds the true partition (0.12-0.13,
//! not 1e-6; SPLIT is exact only trivially, one group, ARI 0; affinity/surprise are off by 0.74-0.94): grown modules are
//! only approximately separable until the leaked reads are pruned. No claim is made that the probe could replace
//! `co_activate`; no library API is added.
//!
//!   cargo test -p openmorph-core --test research behavioural_boundaries -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

const INPUTS: [usize; 4] = [0, 1, 2, 3];
const OUTS: [usize; 2] = [4, 5];
const CELLS: [usize; 6] = [0, 1, 2, 3, 4, 5];
/// The true module of each cell 0..=5.
const TRUTH: [usize; 6] = [0, 0, 1, 1, 0, 1];
const GROUND: usize = 6; // a host cell nothing writes: a grounded (0) input
const PROBE_BUDGET: usize = 4000;
const SEEDS: u64 = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Task {
    Decompose,
    Grown,
    Backbone,
    /// The grown protocol with an ADDITIVE second module, M2 = 0.5·(c+d): a module with no within-module interaction.
    GrownAdditive,
}

fn seed_rng(base: u64, s: u64) -> Rng {
    Rng::new(base ^ s.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Build a trained field for `task` and seed `s` (seed 0 of `Decompose` is the proof's field bit for bit).
/// Returns (field, training-fit RMSE of the two outputs vs their targets on 200 fresh points).
fn target(task: Task, x: &[f32; 4]) -> [f32; 2] {
    let m1 = match task {
        Task::Decompose => (x[0] - x[1]).tanh(),
        _ => x[0] * x[1],
    };
    let m2 = match task {
        Task::GrownAdditive => 0.5 * (x[2] + x[3]),
        _ => (x[2] + x[3]).tanh(),
    };
    [m1, m2]
}

fn build(task: Task, s: u64) -> (Field, f32) {
    let m1 = |a: f32, b: f32| target(task, &[a, b, 0.0, 0.0])[0];
    let m2 = |c: f32, d: f32| target(task, &[0.0, 0.0, c, d])[1];
    let mut f = Field::new(6);
    if s > 0 {
        f.rng_seed(0xB0DA_2026 ^ s.wrapping_mul(0xD1B5_4A32_D192_ED03));
    }
    match task {
        Task::Decompose => {
            f.config_mut().max_hidden = 0;
            f.config_mut().birth_gate = 0.05;
            let mut rng = seed_rng(0x00DE_C0D3, s);
            for _ in 0..2500 {
                let (a, b, c, d) = (rng.sym(), rng.sym(), rng.sym(), rng.sym());
                f.grow(&[(0, a), (1, b)], 4, m1(a, b), 0.1, 40);
                f.grow(&[(2, c), (3, d)], 5, m2(c, d), 0.1, 40);
            }
        }
        Task::Grown | Task::GrownAdditive => {
            f.config_mut().max_hidden = 16;
            f.config_mut().birth_gate = 0.05;
            let base = if task == Task::Grown {
                0x0636_0A11
            } else {
                0x0ADD_0A11
            };
            let mut rng = seed_rng(base, s);
            for _ in 0..4000 {
                let (a, b, c, d) = (rng.sym(), rng.sym(), rng.sym(), rng.sym());
                let all = [(0, a), (1, b), (2, c), (3, d)];
                f.grow(&all, 4, m1(a, b), 0.1, 200);
                f.grow(&all, 5, m2(c, d), 0.1, 200);
            }
        }
        Task::Backbone => {
            let bb = f.grow_backbone(&INPUTS, 24, 1.0);
            f.add_unit(bb.clone(), vec![0.0; bb.len()], 0.0, 4);
            f.add_unit(bb.clone(), vec![0.0; bb.len()], 0.0, 5);
            let mut rng = seed_rng(0x0BAC_4B0E, s);
            for _ in 0..6000 {
                let (a, b, c, d) = (rng.sym(), rng.sym(), rng.sym(), rng.sym());
                let all = [(0, a), (1, b), (2, c), (3, d)];
                f.learn_step(&all, 4, m1(a, b), 0.05, 200);
                f.learn_step(&all, 5, m2(c, d), 0.05, 200);
            }
        }
    }
    let mut p = exact_clone(&f);
    let mut rng = seed_rng(0xF17_7E57, s);
    let mut se = 0.0f32;
    for _ in 0..200 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let y = eval(&mut p, &x);
        se += (y[0] - m1(x[0], x[1])).powi(2) + (y[1] - m2(x[2], x[3])).powi(2);
    }
    (f, (se / 400.0).sqrt())
}

/// A probing clone: `settle_eps = 0`, so a settle computes the units' function exactly (no propagation dead-band).
fn exact_clone(f: &Field) -> Field {
    let mut p = f.clone();
    p.config_mut().settle_eps = 0.0;
    p
}

fn eval(p: &mut Field, x: &[f32; 4]) -> [f32; 2] {
    p.stimulate(&[(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])], PROBE_BUDGET);
    [p.cells()[4], p.cells()[5]]
}

// ── partitions ─────────────────────────────────────────────────────────────────────────────────────────────

/// Labels for cells 0..=5 from `cluster` groups.
fn labels_from_groups(groups: &[Vec<usize>]) -> [usize; 6] {
    let mut l = [usize::MAX; 6];
    for (g, cells) in groups.iter().enumerate() {
        for &c in cells {
            if c < 6 {
                l[c] = g;
            }
        }
    }
    l
}

fn choose2(n: f64) -> f64 {
    n * (n - 1.0) / 2.0
}

/// Adjusted Rand index of `l` against the truth (1 = identical, ~0 = chance).
fn ari(l: &[usize; 6]) -> f32 {
    let k = l.iter().copied().max().unwrap_or(0) + 1;
    let mut table = vec![[0f64; 2]; k];
    for c in 0..6 {
        table[l[c]][TRUTH[c]] += 1.0;
    }
    let sum_ij: f64 = table
        .iter()
        .flat_map(|r| r.iter())
        .map(|&n| choose2(n))
        .sum();
    let a: f64 = table.iter().map(|r| choose2(r[0] + r[1])).sum();
    let b: f64 = (0..2)
        .map(|t| choose2(table.iter().map(|r| r[t]).sum()))
        .sum();
    let total = choose2(6.0);
    let expected = a * b / total;
    let max = 0.5 * (a + b);
    if (max - expected).abs() < 1e-12 {
        return if (sum_ij - expected).abs() < 1e-12 {
            1.0
        } else {
            0.0
        };
    }
    ((sum_ij - expected) / (max - expected)) as f32
}

/// Purity: fraction of cells in the majority true module of their proposed group.
fn purity(l: &[usize; 6]) -> f32 {
    let k = l.iter().copied().max().unwrap_or(0) + 1;
    let mut hit = 0;
    for g in 0..k {
        let mut cnt = [0; 2];
        for c in 0..6 {
            if l[c] == g {
                cnt[TRUTH[c]] += 1;
            }
        }
        hit += cnt[0].max(cnt[1]);
    }
    hit as f32 / 6.0
}

fn exact(l: &[usize; 6]) -> bool {
    (0..6).all(|i| (0..6).all(|j| (l[i] == l[j]) == (TRUTH[i] == TRUTH[j])))
}

/// Union-find components over the 4 inputs given a symmetric `linked(i, j)`; outputs join the component with the
/// largest summed first-order sensitivity `sens[o][i]`. Returns labels for cells 0..=5.
fn components(linked: &dyn Fn(usize, usize) -> bool, sens: &[[f32; 4]; 2]) -> [usize; 6] {
    let mut parent = [0usize, 1, 2, 3];
    fn find(p: &mut [usize; 4], i: usize) -> usize {
        let mut i = i;
        while p[i] != i {
            i = p[i];
        }
        i
    }
    for i in 0..4 {
        for j in i + 1..4 {
            if linked(i, j) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }
    let mut roots: Vec<usize> = Vec::new();
    let mut l = [0usize; 6];
    for (i, li) in l.iter_mut().take(4).enumerate() {
        let r = find(&mut parent, i);
        let id = match roots.iter().position(|&x| x == r) {
            Some(p) => p,
            None => {
                roots.push(r);
                roots.len() - 1
            }
        };
        *li = id;
    }
    for (o, s) in sens.iter().enumerate() {
        let best = (0..roots.len())
            .max_by(|&a, &b| {
                let sa: f32 = (0..4).filter(|&i| l[i] == a).map(|i| s[i]).sum();
                let sb: f32 = (0..4).filter(|&i| l[i] == b).map(|i| s[i]).sum();
                sa.total_cmp(&sb)
            })
            .unwrap();
        l[4 + o] = best;
    }
    l
}

// ── the routes ────────────────────────────────────────────────────────────────────────────────────────────

fn route_authored(f: &Field) -> [usize; 6] {
    let mut g = f.clone();
    for it in 0..4000 {
        g.co_activate(&[0, 1, 4], 0.05);
        g.co_activate(&[2, 3, 5], 0.05);
        if it % 100 == 99 {
            g.spread_positions();
        }
    }
    g.spread_positions();
    labels_from_groups(&g.cluster(&CELLS, 2))
}

fn route_affinity(f: &Field) -> [usize; 6] {
    let mut g = f.clone();
    let sets: Vec<Vec<usize>> = g
        .units
        .iter()
        .map(|u| {
            let maxw = u.weights.iter().fold(0.0f32, |m, w| m.max(w.abs()));
            let mut s: Vec<usize> = u
                .reads
                .iter()
                .zip(&u.weights)
                .filter(|(_, w)| maxw > 0.0 && w.abs() >= 0.1 * maxw)
                .map(|(&r, _)| r)
                .collect();
            s.push(u.write);
            s
        })
        .collect();
    for it in 0..4000 {
        for s in &sets {
            g.co_activate(s, 0.05);
        }
        if it % 100 == 99 {
            g.spread_positions();
        }
    }
    g.spread_positions();
    labels_from_groups(&g.cluster(&CELLS, 2))
}

/// The finite-difference probe's raw measurements.
struct Probe {
    /// inter[i][j] = max over outputs of median |ΔᵢΔⱼ f_o| / h².
    inter: [[f32; 4]; 4],
    /// sens[o][i] = median |Δᵢ f_o| / h.
    sens: [[f32; 4]; 2],
}

fn probe(f: &Field, s: u64) -> Probe {
    let h = 0.05f32;
    let n = 500;
    let mut p = exact_clone(f);
    let mut rng = seed_rng(0x9120_BE00, s);
    let mut d2: Vec<Vec<[f32; 2]>> = (0..16).map(|_| Vec::with_capacity(n)).collect();
    let mut d1: Vec<Vec<[f32; 2]>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let f0 = eval(&mut p, &x);
        let mut fi = [[0f32; 2]; 4];
        for i in 0..4 {
            let mut xi = x;
            xi[i] += h;
            fi[i] = eval(&mut p, &xi);
            d1[i].push([(fi[i][0] - f0[0]).abs() / h, (fi[i][1] - f0[1]).abs() / h]);
        }
        for i in 0..4 {
            for j in i + 1..4 {
                let mut xij = x;
                xij[i] += h;
                xij[j] += h;
                let fij = eval(&mut p, &xij);
                let dd = |o: usize| (fij[o] - fi[i][o] - fi[j][o] + f0[o]).abs() / (h * h);
                d2[i * 4 + j].push([dd(0), dd(1)]);
            }
        }
    }
    assert_eq!(p.truncated_settles(), 0, "probe settles must complete");
    let median = |v: &[[f32; 2]], o: usize| {
        let mut a: Vec<f32> = v.iter().map(|x| x[o]).collect();
        a.sort_by(f32::total_cmp);
        a[a.len() / 2]
    };
    let mut inter = [[0f32; 4]; 4];
    for i in 0..4 {
        for j in i + 1..4 {
            let v = &d2[i * 4 + j];
            let m = median(v, 0).max(median(v, 1));
            inter[i][j] = m;
            inter[j][i] = m;
        }
    }
    let mut sens = [[0f32; 4]; 2];
    for (o, row) in sens.iter_mut().enumerate() {
        for (i, v) in row.iter_mut().enumerate() {
            *v = median(&d1[i], o);
        }
    }
    Probe { inter, sens }
}

/// First-order-sensitivity baseline (the skeptic's): each output is its own module; each input joins the output it
/// has the largest median first-order sensitivity `|Δᵢ f_o| / h` to. No second differences, no threshold.
fn route_sensitivity(pr: &Probe) -> [usize; 6] {
    let mut l = [0usize, 0, 0, 0, 0, 1];
    for (i, li) in l.iter_mut().take(4).enumerate() {
        *li = (pr.sens[1][i] > pr.sens[0][i]) as usize;
    }
    l
}

/// `Field::group_by_surprise` (the in-repo causal analogue of `co_activate`, gated by `surprise_currency`): on a
/// clone with the currency on, warm the end-to-end sensitivity of ONE output (learn_step at lr 0: the adjoint sweep
/// and the cell-variance EMA run, no weight moves), group {inputs, that output} by surprise-reduction, then the same
/// for the other output; `cluster` the six cells into two. Nothing authored beyond which cells are outputs.
fn route_surprise(task: Task, f: &Field, s: u64) -> [usize; 6] {
    let mut g = f.clone();
    g.config_mut().surprise_currency = true;
    let mut rng = seed_rng(0x5A2B_0000, s);
    for (k, &o) in OUTS.iter().enumerate() {
        for _ in 0..600 {
            let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
            let all = [(0, x[0]), (1, x[1]), (2, x[2]), (3, x[3])];
            g.learn_step(&all, o, target(task, &x)[k], 0.0, PROBE_BUDGET);
        }
        let set = [0, 1, 2, 3, o];
        for it in 0..2000 {
            g.group_by_surprise(&set, 0.5);
            if it % 100 == 99 {
                g.spread_positions();
            }
        }
    }
    g.spread_positions();
    labels_from_groups(&g.cluster(&CELLS, 2))
}

fn route_interaction(pr: &Probe) -> [usize; 6] {
    let maxi = (0..4)
        .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
        .map(|(i, j)| pr.inter[i][j])
        .fold(0.0f32, f32::max);
    let tau = 0.1 * maxi;
    components(&|i, j| maxi > 0.0 && pr.inter[i][j] > tau, &pr.sens)
}

/// Eigenvalues of a symmetric matrix (cyclic Jacobi, f64), descending.
#[allow(clippy::needless_range_loop)] // Jacobi rotations index two rows at once
fn sym_eig(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _sweep in 0..60 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut e: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    e.sort_by(|x, y| y.total_cmp(x));
    e
}

/// σ₂/σ₁ of an m×m matrix (via the eigenvalues of M·Mᵀ).
fn sv_ratio(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    let g: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..m[0].len()).map(|k| m[i][k] * m[j][k]).sum())
                .collect()
        })
        .collect();
    let e = sym_eig(g);
    if e[0] <= 0.0 {
        return 0.0;
    }
    (e[1].max(0.0) / e[0]).sqrt()
}

/// Split probe: σ₂/σ₁ per cut (A contains input 0), max over the two outputs. Returns (cuts as bitmask of A, ratio).
fn split_ratios(f: &Field, s: u64) -> Vec<(u8, f64)> {
    let m = 16;
    let mut p = exact_clone(f);
    let mut rng = seed_rng(0x5B11_7000, s);
    let mut out = Vec::new();
    for mask in 1u8..16 {
        if mask & 1 == 0 || mask == 15 {
            continue; // A always holds input 0; B non-empty ⇒ the 7 distinct bipartitions
        }
        let xa: Vec<[f32; 4]> = (0..m)
            .map(|_| [rng.sym(), rng.sym(), rng.sym(), rng.sym()])
            .collect();
        let xb: Vec<[f32; 4]> = (0..m)
            .map(|_| [rng.sym(), rng.sym(), rng.sym(), rng.sym()])
            .collect();
        let mut mats = [vec![vec![0f64; m]; m], vec![vec![0f64; m]; m]];
        for pi in 0..m {
            for qi in 0..m {
                let mut x = [0f32; 4];
                for (i, xv) in x.iter_mut().enumerate() {
                    *xv = if mask >> i & 1 == 1 {
                        xa[pi][i]
                    } else {
                        xb[qi][i]
                    };
                }
                let y = eval(&mut p, &x);
                mats[0][pi][qi] = y[0] as f64;
                mats[1][pi][qi] = y[1] as f64;
            }
        }
        out.push((mask, sv_ratio(&mats[0]).max(sv_ratio(&mats[1]))));
    }
    out
}

const SPLIT_TAU: f64 = 1e-3;

fn route_split(ratios: &[(u8, f64)], sens: &[[f32; 4]; 2]) -> [usize; 6] {
    let separates = |mask: u8, i: usize, j: usize| (mask >> i & 1) != (mask >> j & 1);
    components(
        &|i, j| {
            !ratios
                .iter()
                .any(|&(mask, r)| r < SPLIT_TAU && separates(mask, i, j))
        },
        sens,
    )
}

/// Realise a partition: `encapsulate` each component's outputs, `nest` it in a fresh host fed ONLY that component's
/// inputs (other sources grounded at 0), and return (max |host − field|, max |host − target|, max |field − target|)
/// over 200 fresh points.
fn nest_error(task: Task, f: &Field, l: &[usize; 6], eps: Option<f32>, s: u64) -> (f32, f32, f32) {
    let mut orig = f.clone();
    if let Some(e) = eps {
        orig.config_mut().settle_eps = e;
    }
    let mut host = Field::new(GROUND + 1);
    if let Some(e) = eps {
        host.config_mut().settle_eps = e;
    }
    let k = l.iter().copied().max().unwrap_or(0) + 1;
    for g in 0..k {
        let outs: Vec<usize> = OUTS.iter().copied().filter(|&o| l[o] == g).collect();
        if outs.is_empty() {
            continue;
        }
        let sub = orig.encapsulate(&outs);
        // the sub's SOURCE cells (read, written by none), ascending — what `nest` maps `reads` onto.
        let written: Vec<usize> = sub.units.iter().map(|u| u.write).collect();
        let mut sources: Vec<usize> = sub
            .units
            .iter()
            .flat_map(|u| u.reads.iter().copied().chain(u.factor))
            .filter(|c| !written.contains(c))
            .collect();
        sources.sort_unstable();
        sources.dedup();
        let reads: Vec<usize> = sources
            .iter()
            .map(|&c| if c < 4 && l[c] == g { c } else { GROUND })
            .collect();
        host.nest_multi(sub, &reads, &outs, PROBE_BUDGET);
    }
    let mut rng = seed_rng(0x0E57_0000, s);
    let (mut worst, mut host_t, mut orig_t) = (0.0f32, 0.0f32, 0.0f32);
    for _ in 0..200 {
        let x = [rng.sym(), rng.sym(), rng.sym(), rng.sym()];
        let yo = eval(&mut orig, &x);
        let yh = eval(&mut host, &x);
        let yt = target(task, &x);
        for o in 0..2 {
            worst = worst.max((yo[o] - yh[o]).abs());
            host_t = host_t.max((yh[o] - yt[o]).abs());
            orig_t = orig_t.max((yo[o] - yt[o]).abs());
        }
    }
    (worst, host_t, orig_t)
}

// ── the comparison ────────────────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Tally {
    ari: f32,
    purity: f32,
    exact: usize,
    nest_max: f32,
    nest_default_eps_max: f32,
    /// max |nested host − target| and max |original field − target| (is the cut functionally free?)
    host_target_max: f32,
    orig_target_max: f32,
}

impl Tally {
    fn add(&mut self, task: Task, f: &Field, l: &[usize; 6], s: u64) {
        self.ari += ari(l);
        self.purity += purity(l);
        self.exact += exact(l) as usize;
        let (e0, ht, ot) = nest_error(task, f, l, Some(0.0), s);
        self.nest_max = self.nest_max.max(e0);
        self.host_target_max = self.host_target_max.max(ht);
        self.orig_target_max = self.orig_target_max.max(ot);
        self.nest_default_eps_max = self
            .nest_default_eps_max
            .max(nest_error(task, f, l, None, s).0);
    }
}

struct TaskResult {
    authored: Tally,
    affinity: Tally,
    interaction: Tally,
    split: Tally,
    sensitivity: Tally,
    surprise: Tally,
    false_edges: usize,
    missed_edges: usize,
    split_false: usize,
    cross_over_within: Vec<f32>,
    cross_above_within: usize,
    true_cut_ratio: Vec<f64>,
    /// min σ₂/σ₁ over the 6 cuts that separate a true module (the best WRONG cut)
    wrong_cut_ratio: Vec<f64>,
    /// seeds where the true cut has the lowest σ₂/σ₁ of all 7
    true_cut_lowest: usize,
    /// first-order leak: max over outputs of max_{i ∉ module} |Δᵢf_o| / max_{i ∈ module} |Δᵢf_o| (medians)
    leak: Vec<f32>,
    /// max cross / max within interaction (the margin against the pre-registered 0.1 edge threshold)
    cross_over_max: Vec<f32>,
    fit: Vec<f32>,
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

fn run_task(task: Task) -> TaskResult {
    let mut r = TaskResult {
        authored: Tally::default(),
        affinity: Tally::default(),
        interaction: Tally::default(),
        split: Tally::default(),
        sensitivity: Tally::default(),
        surprise: Tally::default(),
        false_edges: 0,
        missed_edges: 0,
        split_false: 0,
        cross_over_within: Vec::new(),
        cross_above_within: 0,
        true_cut_ratio: Vec::new(),
        wrong_cut_ratio: Vec::new(),
        true_cut_lowest: 0,
        leak: Vec::new(),
        cross_over_max: Vec::new(),
        fit: Vec::new(),
    };
    let cross = [(0, 2), (0, 3), (1, 2), (1, 3)];
    let within = [(0, 1), (2, 3)];
    for s in 0..SEEDS {
        let (f, fit) = build(task, s);
        r.fit.push(fit);

        let la = route_authored(&f);
        r.authored.add(task, &f, &la, s);
        let lb = route_affinity(&f);
        r.affinity.add(task, &f, &lb, s);

        let pr = probe(&f, s);
        let lc = route_interaction(&pr);
        r.interaction.add(task, &f, &lc, s);
        let le = route_sensitivity(&pr);
        r.sensitivity.add(task, &f, &le, s);
        let lf = route_surprise(task, &f, s);
        r.surprise.add(task, &f, &lf, s);
        let maxi = cross
            .iter()
            .chain(within.iter())
            .map(|&(i, j)| pr.inter[i][j])
            .fold(0.0f32, f32::max);
        r.false_edges += cross
            .iter()
            .filter(|&&(i, j)| pr.inter[i][j] > 0.1 * maxi)
            .count();
        r.missed_edges += within
            .iter()
            .filter(|&&(i, j)| pr.inter[i][j] <= 0.1 * maxi)
            .count();
        let max_cross = cross
            .iter()
            .map(|&(i, j)| pr.inter[i][j])
            .fold(0.0, f32::max);
        let min_within = within
            .iter()
            .map(|&(i, j)| pr.inter[i][j])
            .fold(f32::MAX, f32::min);
        let mean_cross = cross.iter().map(|&(i, j)| pr.inter[i][j]).sum::<f32>() / 4.0;
        let mean_within = within.iter().map(|&(i, j)| pr.inter[i][j]).sum::<f32>() / 2.0;
        r.cross_over_within
            .push(mean_cross / mean_within.max(1e-12));
        r.cross_above_within += (max_cross > min_within) as usize;
        r.cross_over_max.push(max_cross / maxi.max(1e-12));
        let leak = (0..2)
            .map(|o| {
                let own = |i: usize| TRUTH[i] == o;
                let out_max = (0..4)
                    .filter(|&i| !own(i))
                    .map(|i| pr.sens[o][i])
                    .fold(0.0, f32::max);
                let in_max = (0..4)
                    .filter(|&i| own(i))
                    .map(|i| pr.sens[o][i])
                    .fold(0.0, f32::max);
                out_max / in_max.max(1e-12)
            })
            .fold(0.0, f32::max);
        r.leak.push(leak);

        let ratios = split_ratios(&f, s);
        let ld = route_split(&ratios, &pr.sens);
        r.split.add(task, &f, &ld, s);
        // the true cut {0,1}|{2,3} is mask 0b0011
        let tc = ratios.iter().find(|&&(m, _)| m == 0b0011).unwrap().1;
        r.true_cut_ratio.push(tc);
        let wrong = ratios
            .iter()
            .filter(|&&(m, _)| m != 0b0011)
            .map(|&(_, x)| x)
            .fold(f64::MAX, f64::min);
        r.wrong_cut_ratio.push(wrong);
        r.true_cut_lowest += (tc < wrong) as usize;
        r.split_false += cross.iter().filter(|&&(i, j)| ld[i] == ld[j]).count();

        eprintln!(
            "[BOUNDARY] {task:?} seed {s}: fit {fit:.3} | authored {la:?} affinity {lb:?} interaction {lc:?} split {ld:?} sensitivity {le:?} surprise {lf:?} | \
             inter within ({:.3},{:.3}) cross ({:.3},{:.3},{:.3},{:.3}) | true-cut σ₂/σ₁ {tc:.2e}",
            pr.inter[0][1], pr.inter[2][3], pr.inter[0][2], pr.inter[0][3], pr.inter[1][2], pr.inter[1][3]
        );
    }
    let n = SEEDS as f32;
    for (name, t) in [
        ("AUTHORED   ", &r.authored),
        ("AFFINITY   ", &r.affinity),
        ("INTERACTION", &r.interaction),
        ("SPLIT      ", &r.split),
        ("SENSITIVITY", &r.sensitivity),
        ("SURPRISE   ", &r.surprise),
    ] {
        eprintln!(
            "[BOUNDARY] {task:?} {name}: ARI {:.2} purity {:.2} exact {}/{} | nest max err eps0 {:.2e} (default eps {:.2e}) \
             | vs target: nested {:.3} original {:.3}",
            t.ari / n,
            t.purity / n,
            t.exact,
            SEEDS,
            t.nest_max,
            t.nest_default_eps_max,
            t.host_target_max,
            t.orig_target_max
        );
    }
    let mut cw = r.cross_over_within.clone();
    let mut tcr: Vec<f32> = r.true_cut_ratio.iter().map(|&x| x as f32).collect();
    let mut fit = r.fit.clone();
    let mut wcr: Vec<f32> = r.wrong_cut_ratio.iter().map(|&x| x as f32).collect();
    let mut leak = r.leak.clone();
    let mut com = r.cross_over_max.clone();
    eprintln!(
        "[BOUNDARY] {task:?} margins: max-cross/max-interaction median {:.3} max {:.3} (edge threshold 0.1) | first-order leak \
         median {:.3} max {:.3} | best wrong-cut σ₂/σ₁ median {:.2e} | true cut is the lowest-ratio cut in {}/{} seeds",
        median(&mut com),
        r.cross_over_max.iter().copied().fold(0.0, f32::max),
        median(&mut leak),
        r.leak.iter().copied().fold(0.0, f32::max),
        median(&mut wcr),
        r.true_cut_lowest,
        SEEDS
    );
    eprintln!(
        "[BOUNDARY] {task:?} probe: false edges {}/{} missed {}/{} | split false links {}/{} | mean cross/within interaction \
         median {:.3} | strongest cross > weakest within in {}/{} seeds | true-cut σ₂/σ₁ median {:.2e} | fit RMSE median {:.3}",
        r.false_edges,
        4 * SEEDS,
        r.missed_edges,
        2 * SEEDS,
        r.split_false,
        4 * SEEDS,
        median(&mut cw),
        r.cross_above_within,
        SEEDS,
        median(&mut tcr),
        median(&mut fit)
    );
    r
}

#[test]
#[ignore = "research: on demand"]
fn behavioural_boundaries_compare() {
    let dec = run_task(Task::Decompose);
    let grown = run_task(Task::Grown);
    let bb = run_task(Task::Backbone);
    let add = run_task(Task::GrownAdditive);
    // The skeptic's first-order baseline: argmax median sensitivity recovers every module on every field (40/40),
    // including the additive one — so this benchmark cannot show second-order interaction detection adds value.
    for r in [&dec, &grown, &bb, &add] {
        assert_eq!(r.sensitivity.exact, SEEDS as usize);
    }
    // The interaction probe is strictly worse on the additive task (a module with no within-module interaction).
    assert!(add.interaction.exact < SEEDS as usize);
    // `group_by_surprise` (the in-repo non-authored grouping) does not recover the modules reliably on any field.
    for r in [&dec, &grown, &bb, &add] {
        assert!(r.surprise.exact < SEEDS as usize);
    }
    // Assert only what was measured (see the module docs).
    // Hand-wired, exactly separable field: every route recovers the modules and nests exactly.
    for t in [&dec.authored, &dec.affinity, &dec.interaction, &dec.split] {
        assert_eq!(t.exact, SEEDS as usize);
        assert!(
            t.nest_max <= 1e-6,
            "decompose nests exactly ({:.2e})",
            t.nest_max
        );
    }
    // Grown fields: the interaction probe recovers the modules in 19/20, weight-affinity in 0/20, split in 0/20.
    assert!(grown.interaction.exact + bb.interaction.exact >= 19);
    assert_eq!(grown.affinity.exact + bb.affinity.exact, 0);
    assert_eq!(grown.split.exact + bb.split.exact, 0);
    assert!(
        grown.false_edges + bb.false_edges <= 1,
        "≤ 1/80 grown false edges"
    );
    assert_eq!(grown.cross_above_within + bb.cross_above_within, 0);
    // …but no boundary on a grown field is exact: the modules leak at first order.
    for t in [
        &grown.authored,
        &grown.interaction,
        &bb.authored,
        &bb.interaction,
    ] {
        assert!(
            t.nest_max > 0.1,
            "grown boundaries are not exact ({:.3})",
            t.nest_max
        );
        assert!(
            t.host_target_max <= t.orig_target_max + 1e-6,
            "…yet the cut costs no task accuracy"
        );
    }
}
