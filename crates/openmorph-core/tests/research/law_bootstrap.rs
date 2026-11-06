// SPDX-License-Identifier: MIT-0
//! LAW BOOTSTRAP — do a few exact constraints single out the shipped delta rule, the way Rodina's hidden zeros
//! single out Tr φ³? (Experiment 1 of `docs/research/figueiredo-amplitude-geometry.md` §4.)
//!
//! THE SHIPPED RULE (`substrate/learn.rs::learn_settled`), modulo scalar step size. Per synapse `k` of a unit,
//! `Δw_k = η_u · err · f′(y) · x_k` (bias: `x ≡ 1`). Everything else the code multiplies in is ONE per-unit,
//! per-event scalar `η_u`, the same for every synapse of the unit: plasticity · heritable `lr` factor · call `lr`
//! · write-gate · anneal gain, or, when the stability cap bites, plasticity · `κ/(1+Σx²)` · gate · gain (the cap
//! reads every read of the unit, so it is non-local, but it is still a common scalar). Momentum (default μ = 0)
//! is a filter on top and is excluded. So "the shipped rule" = the synapse-local direction `L = e·f′·x`;
//! `probe_shipped_update_is_a_common_scalar_times_err_deriv_pre` checks this against a live `Field` (the ratio
//! `Δw_k / (e·f′·x_k)` is one number across `k` and the bias, both below and above the cap).
//!
//! ANSATZ. `L = Σ a_m · m` over monomials `m` in the five synapse-local symbols `x` (pre), `y` (post), `e` (err),
//! `w` (the synapse's own weight), `f′` (the unit's slope, a FORMAL symbol: the law must serve every activation
//! the substrate ships, so `f′` is not tied to `1−y²`). Degree bound `D` on the total degree:
//!   * A1 = the literal note: total degree ≤ 2 → 21 monomials;
//!   * A2 = degree ≤ 2 in (x, y, e, w) times f′^{0,1} → 30 monomials;
//!   * A3..A5 = total degree ≤ 3, 4, 5 → 56, 126, 252 monomials. A3 is the smallest ansatz that CONTAINS the
//!     shipped rule `e·f′·x` (degree 3).
//!
//! CONSTRAINTS, each an EXACT linear condition on the `a_m`, written as evaluation rows at deterministic integer
//! points (never a fit):
//!   * (Z) `Δw = 0` when `e = 0` — rows `m(p|e:=0)`.
//!   * (G) the weight×gain gauge `(w, b, g) → (w/α, b/α, α·g)` (conservation-symmetry.md §2a; `compute` in
//!     `substrate/mod.rs` evaluates `act(g·(Σw·x+b))`). `x, y, e, f′` are gauge-invariant, only `w` moves. The
//!     gain `g` is NOT a synapse-local symbol of the ansatz, so the gauge weight must be carried either by `L` or by
//!     the per-unit step. Covariance (`Δw` transforms like `w`) therefore means `L(w/α) = α^{−d}·L(w)` with
//!     `d = 1 − (gauge weight carried by η_u)`: **G0** (`d = 0`) — the step carries `1/g` (or the heritable `lr`
//!     co-transforms `lr → lr/α`), `L` is gauge-invariant; **G1** (`d = 1`) — the step is gauge-invariant, which is
//!     what the code does (no readout step reads `g`). Rows `2^d·m(p|w:=w/2) − m(p)` (α = 2 separates every
//!     w-degree).
//!   * (S) sign-equivariance, two exact symmetries of the forward pass: S_in `(x_k, w_k) → (−x_k, −w_k)` leaves
//!     `w_k·x_k` unchanged; S_out `(w, b, y, target) → (−w, −b, −y, −target)` for an ODD activation (tanh,
//!     softsign, hardtanh, identity), which flips `e` and keeps `f′` (even). Equivariance: `Δw → −Δw`. Rows
//!     `m(σp) + m(p)`.
//!   * (F) fixed-point consistency under `encapsulate` for a two-layer morph: learning at a cell must not depend on
//!     whether the layer below is flat or banked + `nest`ed. Rows `m(s_flat) − m(s_nested)` with the 5-tuple read
//!     off REAL fields (`encapsulate_constraint_is_vacuous_on_a_local_law`).
//!   * extra, to collapse what survives: (N) `Δw = 0` when `f′ = 0` — the second zero (a unit whose output cannot
//!     respond does not move; the code has it: `slope_of` returns 0 for a zero forward factor) — rows `m(p|f′:=0)`;
//!     and, at degree ≥ 5 only, (H) linearity in pre `L(2x) = 2·L(x)`.
//!
//! EXACTNESS. Rank is computed by Gaussian elimination mod p = 2^61−1; `rank_p ≤ rank_Q`, so `nullity_Q ≤
//! nullity_p`. Every null vector mod p is lifted and re-checked against every row in exact i128 integer arithmetic;
//! exhibiting `nullity_p` exactly-verified independent solutions proves `nullity_Q ≥ nullity_p`. So each reported
//! dimension is the exact dimension over Q.
//!
//! MEASURED (run 2026-09, `--nocapture`; dimension after each constraint, S = S_in and S_out together):
//!   * A3 (smallest ansatz containing the shipped rule), chain Z → G0 → S → F: **56 → 21 → 15 → 2 → 2**, survivors
//!     `{e·x, e·f′·x}`. F adds rank **0**: the flat and nested readout see bit-identical `(x, y, e, w, f′)` at 16
//!     inputs, and one `learn_step` moves the readout bit-identically (the flat hidden layer moves 9.9e-3 by feature
//!     directions while the nested one is a frozen brick — a convention, not a constraint on `L`). Then (N) →
//!     **1**, spanned by `e·f′·x` = THE SHIPPED RULE. The shipped rule satisfies Z, G0, S, F and N.
//!   * A1 (the note's literal degree-≤2 ansatz): 21 → 6 → 5 → 1 → 1, survivor `e·x` — plain LMS WITHOUT the
//!     activation slope. The shipped rule is not in that ansatz at all, so this dimension 1 is a misleading PASS.
//!   * A2 (degree ≤ 2 × f′^{0,1}): 30 → 10 → 8 → 2 → 2 → (N) 1 = `e·f′·x`.
//!   * A4: 126 → 56 → 35 → 7 → 7 → (N) **2** = `{e·f′·x, e·f′²·x}`. A5: 252 → 126 → 70 → 12 → 12 → (N) **7**; every
//!     survivor is `e·f′·x` times something, `e·f′·x³` among them; (H) linearity in pre removes that one (→ **6**),
//!     leaving `e·f′·x × {1, f′, f′², e², y·e, y²}`. So the N-collapse to dimension 1 is an artefact of CHOOSING
//!     degree 3: at degree 4 and 5 the survivors are not unique, even with H. What the constraints fix is the
//!     FACTORISATION `e·f′·x · (an even per-unit factor)`, not the rule itself; "determined up to step size" holds
//!     only in that factorised sense.
//!   * G1 (the gauge weight NOT carried by the step — what the code does): A3 chain Z → G1 → S → F = 56 → 21 → 5 →
//!     2 → 2, survivors `{e²·w, y·e·w}` — pure weight-rescaling laws with no `x` dependence. That "no input-dependent
//!     gauge-covariant law" reading holds ONLY at degree ≤ 3: A5 under G1 gives 252 → 126 → 35 → 12 → 12 and keeps
//!     `x²·e²·w` and `x²·y·e·w` (degree 5, input-dependent, exact Z + G1 + S + F solutions). The shipped rule lives in
//!     G0 only, so its step must carry `1/g`.
//!   * Step identification (live `Field`): `Δw_k/(e·f′·x_k)` = 0.0500 for all three synapses and the bias
//!     uncapped (lr 0.05), 0.23557 = κ/(1+Σx²) capped (lr 2.0) — one common per-unit scalar, as claimed.
//!   * Live gauge probe (α = 2, bit-identical forward): plain gauge `|Δout|` = 2.09e-2 after ONE `learn_step`; the
//!     extended gauge `(w/α, b/α, αg, lr/α)` is EXACT for the uncapped readout rule (`|Δout|` = 0, bit-identical);
//!     under the stability cap `κ/(1+Σx²)` it breaks (`|Δout|` = 1.07e-1) because the cap does not read `g`. The
//!     feature-direction law (`apply_backprojection`) is the SAME `L` with err := `e·f′·w_readout`, but its step
//!     multiplies by `g` (the true gradient): under the extended gauge the hidden output diverges (`|Δh|` = 6.4e-3).
//!
//! CAVEATS ON THE CONSTRAINTS THEMSELVES.
//!   * G0 is a strong modelling assumption, not a free symmetry: it demands `L` be gauge-INVARIANT, which bans every
//!     term of nonzero w-degree — weight-decay-like terms (`−λ·w`, `e²·w`, …) are excluded by fiat, not derived.
//!   * S_out holds only for ODD activations (tanh, softsign, hardtanh, identity); for `Relu` (or any non-odd
//!     activation) it is not a symmetry of the forward pass and its rows do not apply.
//!   * (F) is vacuous for any synapse-local law (rank 0), so it does no work in the chain.
//!
//! VERDICT: **FAIL by the note's criterion.** Z, G0, S and F leave dimension 2 (`{e·x, e·f′·x}`) in the smallest
//! ansatz that contains the shipped rule. Adding a second zero, (N) `Δw = 0` when `f′ = 0`, gives dimension 1 =
//! `e·f′·x` at degree 3 only; that collapse follows from the choice of degree (degree 4: `e·f′·x·{1, f′}`; degree 5:
//! 7 survivors, 6 even with H). So the organism's law is pinned only as the factorisation `e·f′·x · (even per-unit
//! factor)`, not uniquely, and only under G0 (itself a strong assumption) and odd activations (S_out).
//!
//! OPEN (implementation finding, measured; NOT changed here): the implemented steps do not share one gauge weight.
//! The readout step carries g⁰, the feature-direction step g¹, and the stability cap ignores `g`; gauge covariance
//! would need g⁻¹ on the step and the cap `κ/(g·(1+Σx²))` (which is also the true LMS stability bound
//! `η·g·f′²·(1+Σx²) < 2` of the shipped step). Latent at scale 1 (grown writers are born at `init_scale = 1`), live
//! wherever the heritable scale ≠ 1 (features, `set_scale`). A gain-aware step is a candidate behaviour change that
//! needs its own measurement.
//!
//! run: cargo test -p openmorph-core --test research law_bootstrap -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

// ── the ansatz ───────────────────────────────────────────────────────────────────────────────────────

/// Symbol order of an exponent vector.
const NAMES: [&str; 5] = ["x", "y", "e", "w", "f'"];
const X: usize = 0;
const Y: usize = 1;
const E: usize = 2;
const W: usize = 3;
const F: usize = 4;

type Mono = [u32; 5];

/// Every exponent vector of total degree ≤ `deg` (graded, lexicographic within a degree).
fn basis_total(deg: u32) -> Vec<Mono> {
    let mut out = Vec::new();
    for d in 0..=deg {
        for a in 0..=d {
            for b in 0..=d - a {
                for c in 0..=d - a - b {
                    for e in 0..=d - a - b - c {
                        let f = d - a - b - c - e;
                        out.push([a, b, c, e, f]);
                    }
                }
            }
        }
    }
    out
}

/// Degree ≤ 2 in (x, y, e, w), times f′^0 or f′^1 (ansatz A2).
fn basis_a2() -> Vec<Mono> {
    let mut out = Vec::new();
    for fp in 0..=1 {
        for m in basis_total(2) {
            if m[F] == 0 {
                let mut m = m;
                m[F] = fp;
                out.push(m);
            }
        }
    }
    out
}

fn name(m: &Mono) -> String {
    let mut s = String::new();
    for (v, &k) in m.iter().enumerate() {
        if k == 0 {
            continue;
        }
        if !s.is_empty() {
            s.push('·');
        }
        s.push_str(NAMES[v]);
        if k > 1 {
            s.push_str(&format!("^{k}"));
        }
    }
    if s.is_empty() {
        s.push('1');
    }
    s
}

fn eval(m: &Mono, p: &[i64; 5]) -> i128 {
    let mut v: i128 = 1;
    for k in 0..5 {
        for _ in 0..m[k] {
            v *= p[k] as i128;
        }
    }
    v
}

// ── exact constraints as evaluation rows ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Con {
    /// Δw = 0 when err = 0.
    Z,
    /// Gauge covariance with the law carrying w-degree `d` (G0: d=0, G1: d=1).
    G(u32),
    /// Sign-equivariance: S_in and S_out.
    S,
    /// Second zero: Δw = 0 when f′ = 0.
    N,
    /// Linearity in pre: L(2x) = 2·L(x).
    H,
}

/// Deterministic generic integer points, every coordinate in ±{1..=5}, `w` even (so `w/2` stays integral).
fn points(n: usize, seed: u64) -> Vec<[i64; 5]> {
    let mut s = seed | 1;
    let mut draw = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = (s >> 33) % 10;
        let mag = (v % 5 + 1) as i64;
        if v < 5 {
            mag
        } else {
            -mag
        }
    };
    (0..n)
        .map(|_| {
            let mut p = [0i64; 5];
            for c in p.iter_mut() {
                *c = draw();
            }
            p[W] *= 2;
            p
        })
        .collect()
}

fn rows(c: Con, basis: &[Mono], seed: u64) -> Vec<Vec<i128>> {
    let pts = points(basis.len() + 16, seed);
    pts.iter()
        .map(|p| {
            basis
                .iter()
                .map(|m| match c {
                    Con::Z => {
                        let mut q = *p;
                        q[E] = 0;
                        eval(m, &q)
                    }
                    Con::N => {
                        let mut q = *p;
                        q[F] = 0;
                        eval(m, &q)
                    }
                    Con::G(d) => {
                        let mut q = *p;
                        q[W] /= 2; // w even ⇒ exact
                        (1i128 << d) * eval(m, &q) - eval(m, p)
                    }
                    Con::S => {
                        // S_in here; S_out's rows come from `rows_s_out` (stacked in `constraint_rows`).
                        let mut q = *p;
                        q[X] = -q[X];
                        q[W] = -q[W];
                        eval(m, &q) + eval(m, p)
                    }
                    Con::H => {
                        let mut q = *p;
                        q[X] *= 2;
                        eval(m, &q) - 2 * eval(m, p)
                    }
                })
                .collect()
        })
        .collect()
}

fn rows_s_out(basis: &[Mono], seed: u64) -> Vec<Vec<i128>> {
    let pts = points(basis.len() + 16, seed);
    pts.iter()
        .map(|p| {
            let mut q = *p;
            q[Y] = -q[Y];
            q[E] = -q[E];
            q[W] = -q[W];
            basis.iter().map(|m| eval(m, &q) + eval(m, p)).collect()
        })
        .collect()
}

fn constraint_rows(c: Con, basis: &[Mono], seed: u64) -> Vec<Vec<i128>> {
    let mut r = rows(c, basis, seed);
    if let Con::S = c {
        r.extend(rows_s_out(basis, seed ^ 0x5eed));
    }
    r
}

// ── exact rank: elimination mod p = 2^61−1, null vectors re-verified over the integers ──────────────────

const P: u64 = (1u64 << 61) - 1;

fn md(v: i128) -> u64 {
    v.rem_euclid(P as i128) as u64
}
fn mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P as u128) as u64
}
fn pw(mut a: u64, mut e: u64) -> u64 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, a);
        }
        a = mul(a, a);
        e >>= 1;
    }
    r
}
fn inv(a: u64) -> u64 {
    pw(a, P - 2)
}

/// Null space mod P of `rows` (RREF), lifted to symmetric integer representatives.
fn null_space(rows: &[Vec<i128>], n: usize) -> Vec<Vec<i128>> {
    let mut m: Vec<Vec<u64>> = rows
        .iter()
        .map(|r| r.iter().map(|&v| md(v)).collect())
        .collect();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r = 0;
    for col in 0..n {
        let Some(pr) = (r..m.len()).find(|&i| m[i][col] != 0) else {
            continue;
        };
        m.swap(r, pr);
        let iv = inv(m[r][col]);
        for v in m[r].iter_mut() {
            *v = mul(*v, iv);
        }
        let pivot_row = m[r].clone();
        for (i, row) in m.iter_mut().enumerate() {
            if i != r && row[col] != 0 {
                let f = row[col];
                for (v, &pv) in row.iter_mut().zip(&pivot_row) {
                    *v = (*v + P - mul(f, pv)) % P;
                }
            }
        }
        pivots.push(col);
        r += 1;
        if r == m.len() {
            break;
        }
    }
    let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
    free.iter()
        .map(|&fc| {
            let mut v = vec![0u64; n];
            v[fc] = 1;
            for (ri, &pc) in pivots.iter().enumerate() {
                v[pc] = (P - m[ri][fc]) % P;
            }
            v.iter()
                .map(|&x| {
                    if x > P / 2 {
                        x as i128 - P as i128
                    } else {
                        x as i128
                    }
                })
                .collect()
        })
        .collect()
}

/// Solve the stacked constraints; return the exactly-verified basis of the solution space.
fn solve(all: &[Vec<i128>], n: usize) -> Vec<Vec<i128>> {
    let ns = null_space(all, n);
    for v in &ns {
        for r in all {
            let dot: i128 = r.iter().zip(v).map(|(a, b)| a * b).sum();
            assert_eq!(dot, 0, "a mod-p null vector failed the exact integer check");
        }
    }
    ns
}

fn describe(ns: &[Vec<i128>], basis: &[Mono]) -> Vec<String> {
    ns.iter()
        .map(|v| {
            let terms: Vec<String> = v
                .iter()
                .enumerate()
                .filter(|(_, &c)| c != 0)
                .map(|(i, &c)| {
                    if c == 1 {
                        name(&basis[i])
                    } else {
                        format!("{c}·{}", name(&basis[i]))
                    }
                })
                .collect();
            terms.join(" + ")
        })
        .collect()
}

/// Run a constraint chain on `basis`, printing the dimension after each step. `f_rows` are the measured (F) rows.
fn chain(
    label: &str,
    basis: &[Mono],
    steps: &[Con],
    f_rows: &[Vec<i128>],
    f_after: usize,
) -> (Vec<usize>, Vec<String>) {
    let n = basis.len();
    let mut all: Vec<Vec<i128>> = Vec::new();
    let mut dims = vec![n];
    let mut trace = format!("{label}: {n}");
    for (k, &c) in steps.iter().enumerate() {
        all.extend(constraint_rows(c, basis, 0xB007 + k as u64 * 7919));
        let d = solve(&all, n).len();
        dims.push(d);
        trace.push_str(&format!(" →{c:?} {d}"));
        if k + 1 == f_after {
            all.extend(f_rows.iter().cloned());
            let d = solve(&all, n).len();
            dims.push(d);
            trace.push_str(&format!(" →F {d}"));
        }
    }
    let ns = solve(&all, n);
    let surv = describe(&ns, basis);
    println!("{trace}\n    survivors: {surv:?}");
    (dims, surv)
}

// ── (F) measured on real fields ─────────────────────────────────────────────────────────────────────────

/// Two-layer morph: hidden `h = tanh(g·(u·x + b1))` (cell 2), readout `y = tanh(a·h + b2)` (cell 3).
fn two_layer(nested: bool) -> Field {
    let mut src = Field::new(4);
    src.add_unit(vec![0, 1], vec![0.8, -0.55], 0.12, 2);
    if !nested {
        src.add_unit(vec![2], vec![0.7], -0.05, 3);
        return src;
    }
    let sub = src.encapsulate(&[2]);
    let mut host = Field::new(4);
    host.nest(sub, &[0, 1], 2, 64);
    host.add_unit(vec![2], vec![0.7], -0.05, 3);
    host
}

/// The readout synapse's local tuple `(x, y, e, w, f′)` at input `(x0, x1)` and `target`.
fn readout_state(f: &mut Field, x0: f32, x1: f32, target: f32) -> [f32; 5] {
    f.stimulate(&[(0, x0), (1, x1)], 64);
    let x = f.cells()[2];
    let y = f.cells()[3];
    let ro = f.units.iter().position(|u| u.write == 3).unwrap();
    [x, y, target - y, f.units[ro].weights[0], 1.0 - y * y]
}

/// Measure (F): the readout's local state flat vs nested at 16 inputs; returns the rows for every basis monomial
/// (as the exact f64 difference, then asserted to be 0 and converted), plus the post-learn comparison.
fn measure_f(basis: &[Mono]) -> (Vec<Vec<i128>>, bool, f32) {
    let mut flat = two_layer(false);
    let mut nest = two_layer(true);
    let mut rows = Vec::new();
    let mut identical = true;
    for i in 0..16 {
        let x0 = -0.9 + 0.113 * i as f32;
        let x1 = 0.7 - 0.087 * i as f32;
        let t = 0.3 * (i as f32 * 0.7).sin();
        let a = readout_state(&mut flat, x0, x1, t);
        let b = readout_state(&mut nest, x0, x1, t);
        identical &= a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits());
        rows.push(
            basis
                .iter()
                .map(|m| {
                    let ev = |s: &[f32; 5]| -> f64 {
                        (0..5).map(|k| (s[k] as f64).powi(m[k] as i32)).product()
                    };
                    let d = ev(&a) - ev(&b);
                    assert_eq!(d, 0.0, "flat vs nested local state differ");
                    0i128
                })
                .collect(),
        );
    }
    // One learn_step on each: the readout moves identically; the hidden layer does not (nested = frozen brick).
    flat.learn_step(&[(0, 0.4), (1, -0.3)], 3, 0.5, 0.1, 64);
    nest.learn_step(&[(0, 0.4), (1, -0.3)], 3, 0.5, 0.1, 64);
    let ro = |f: &Field| f.units.iter().position(|u| u.write == 3).unwrap();
    let same_readout = flat.units[ro(&flat)].weights[0].to_bits()
        == nest.units[ro(&nest)].weights[0].to_bits()
        && flat.units[ro(&flat)].bias.to_bits() == nest.units[ro(&nest)].bias.to_bits();
    let hid = flat.units.iter().position(|u| u.write == 2).unwrap();
    let hidden_moved = (flat.units[hid].weights[0] - 0.8).abs();
    println!(
        "(F) 16 inputs: flat/nested readout tuples bit-identical = {identical}; after one learn_step readout \
         bit-identical = {same_readout}; flat hidden weight moved {hidden_moved:.3e} (feature_lr {}), nested hidden \
         frozen",
        flat.config().feature_lr
    );
    (rows, identical && same_readout, hidden_moved)
}

// ── the tests ───────────────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "research: on demand"]
fn bootstrap_dimension_chain() {
    let a3 = basis_total(3);
    let (f3, f_ok, _) = measure_f(&a3);
    assert!(
        f_ok,
        "(F) must be measured on bit-identical flat/nested states"
    );

    // Main chain on A3 (the smallest ansatz containing the shipped rule).
    let steps = [Con::Z, Con::G(0), Con::S, Con::N];
    let (d3, s3) = chain("A3 deg≤3 [Z,G0,S,F,N]", &a3, &steps, &f3, 3);
    assert_eq!(d3, vec![56, 21, 15, 2, 2, 1]);
    assert_eq!(
        s3,
        vec!["x·e·f'".to_string()],
        "the survivor is the shipped rule"
    );
    // Before N: the two survivors.
    let (d3b, s3b) = chain("A3 deg≤3 [Z,G0,S,F]", &a3, &steps[..3], &f3, 3);
    assert_eq!(*d3b.last().unwrap(), 2);
    assert_eq!(s3b, vec!["x·e".to_string(), "x·e·f'".to_string()]);

    // G1: the gauge weight NOT carried by the step (what the code does).
    let (g1, s_g1) = chain(
        "A3 deg≤3 [Z,G1,S,F]",
        &a3,
        &[Con::Z, Con::G(1), Con::S],
        &f3,
        3,
    );
    assert_eq!(g1, vec![56, 21, 5, 2, 2]);
    assert!(
        s_g1.iter().all(|s| s.contains('w') && !s.contains('x')),
        "G1 survivors are pure weight-rescaling laws: {s_g1:?}"
    );

    // The literal ansatz of the note (A1) and the split one (A2).
    let a1 = basis_total(2);
    let (f1, _, _) = measure_f(&a1);
    let (d1, s1) = chain("A1 deg≤2 [Z,G0,S,F]", &a1, &steps[..3], &f1, 3);
    assert_eq!(d1, vec![21, 6, 5, 1, 1]);
    assert_eq!(
        s1,
        vec!["x·e".to_string()],
        "A1 picks LMS without the slope"
    );
    let a2 = basis_a2();
    let (f2, _, _) = measure_f(&a2);
    let (d2, s2) = chain("A2 deg≤2×f'^{0,1} [Z,G0,S,F,N]", &a2, &steps, &f2, 3);
    assert_eq!(d2, vec![30, 10, 8, 2, 2, 1]);
    assert_eq!(s2, vec!["x·e·f'".to_string()]);

    // Higher degree: does the collapse survive, and what does it leave?
    let a4 = basis_total(4);
    let (f4, _, _) = measure_f(&a4);
    let (d4, s4) = chain("A4 deg≤4 [Z,G0,S,F,N]", &a4, &steps, &f4, 3);
    let a5 = basis_total(5);
    let (f5, _, _) = measure_f(&a5);
    let (d5, s5) = chain("A5 deg≤5 [Z,G0,S,F,N]", &a5, &steps, &f5, 3);
    let (d5h, s5h) = chain(
        "A5 deg≤5 [Z,G0,S,F,N,H]",
        &a5,
        &[Con::Z, Con::G(0), Con::S, Con::N, Con::H],
        &f5,
        3,
    );
    // Every survivor at every degree contains the shipped factor e·f′·x.
    for s in s4.iter().chain(&s5).chain(&s5h) {
        assert!(
            s.contains('x') && s.contains('e') && s.contains("f'"),
            "{s}"
        );
    }
    // With H, the survivors are e·f′·x times a per-unit scalar (no further x or w dependence).
    for s in &s5h {
        assert!(!s.contains("x^") && !s.contains('w'), "{s}");
    }
    println!("dims A4 {d4:?}  A5 {d5:?}  A5+H {d5h:?}");
    assert_eq!(d4, vec![126, 56, 35, 7, 7, 2]);
    assert_eq!(d5, vec![252, 126, 70, 12, 12, 7]);
    assert_eq!(d5h, vec![252, 126, 70, 12, 12, 7, 6]);

    // G1 beyond degree 3: the "no input-dependent gauge-covariant law" reading of the A3 G1 chain is a degree
    // artefact. `e²·w·x²` (degree 5) satisfies Z, G1, S_in, S_out and F exactly and depends on the input.
    let (g1_5, s_g1_5) = chain(
        "A5 deg≤5 [Z,G1,S,F]",
        &a5,
        &[Con::Z, Con::G(1), Con::S],
        &f5,
        3,
    );
    println!("dims A5 G1 {g1_5:?}");
    assert_eq!(g1_5, vec![252, 126, 35, 12, 12]);
    let target: Mono = [2, 0, 2, 1, 0];
    let idx = a5.iter().position(|m| *m == target).expect("x²·e²·w in A5");
    let mut all = Vec::new();
    for (k, &c) in [Con::Z, Con::G(1), Con::S].iter().enumerate() {
        all.extend(constraint_rows(c, &a5, 0xB007 + k as u64 * 7919));
    }
    all.extend(f5.iter().cloned());
    assert!(
        all.iter().all(|r| r[idx] == 0),
        "x²·e²·w is an exact Z+G1+S+F solution at degree 5"
    );
    assert!(
        s_g1_5.iter().any(|s| s.contains('x')),
        "at degree 5 an input-dependent G1 law survives: {s_g1_5:?}"
    );
}

/// The shipped update IS `η_u · e · f′ · x_k` with one common scalar across synapses and the bias.
#[test]
#[ignore = "research: on demand"]
fn probe_shipped_update_is_a_common_scalar_times_err_deriv_pre() {
    for (lr, label) in [(0.05f32, "uncapped"), (2.0, "capped")] {
        let mut f = Field::new(4);
        f.add_unit(vec![0, 1, 2], vec![0.3, -0.2, 0.45], 0.1, 3);
        let before = f.units[0].clone();
        let (xs, t) = ([0.6f32, -0.8, 0.35], -0.4f32);
        f.learn_step(&[(0, xs[0]), (1, xs[1]), (2, xs[2])], 3, t, lr, 64);
        let y = f.cells()[3];
        let (e, fp) = (t - y, 1.0 - y * y);
        let mut ratios: Vec<f32> = (0..3)
            .map(|k| (f.units[0].weights[k] - before.weights[k]) / (e * fp * xs[k]))
            .collect();
        ratios.push((f.units[0].bias - before.bias) / (e * fp));
        let energy = 1.0 + xs.iter().map(|x| x * x).sum::<f32>();
        let expect = if lr * energy > 0.5 { 0.5 / energy } else { lr };
        println!("{label}: η per synapse+bias = {ratios:?}  (expected common η = {expect:.6})");
        for r in &ratios {
            assert!(
                (r - expect).abs() <= 1e-4 * expect,
                "{label}: {r} vs {expect}"
            );
        }
    }
}

/// (F) on its own: flat vs nested readout local state is bit-identical, so (F) adds rank 0 to ANY local ansatz.
#[test]
#[ignore = "research: on demand"]
fn encapsulate_constraint_is_vacuous_on_a_local_law() {
    let (rows, ok, hidden_moved) = measure_f(&basis_total(3));
    assert!(ok);
    assert!(rows.iter().all(|r| r.iter().all(|&v| v == 0)));
    assert!(
        hidden_moved > 0.0,
        "flat hidden learns by feature directions"
    );
}

/// Gauge-equivalent writers, one learn_step, then a probe input: does the shipped dynamics respect the gauge?
fn gauge_probe(lr: f32, alpha: f32, lr_cotransform: bool) -> (f32, bool) {
    let build = |a: f32, co: bool| {
        let mut f = Field::new(3);
        f.add_unit(vec![0, 1], vec![0.9 / a, -0.6 / a], 0.2 / a, 2);
        f.set_scale(2, a);
        if co {
            f.set_lr(2, 1.0 / a);
        }
        f
    };
    let mut fa = build(1.0, false);
    let mut fb = build(alpha, lr_cotransform);
    fa.stimulate(&[(0, 0.5), (1, 0.25)], 64);
    fb.stimulate(&[(0, 0.5), (1, 0.25)], 64);
    assert_eq!(
        fa.cells()[2].to_bits(),
        fb.cells()[2].to_bits(),
        "gauge-equivalent forward must be bit-identical (α a power of 2)"
    );
    fa.learn_step(&[(0, 0.5), (1, 0.25)], 2, -0.3, lr, 64);
    fb.learn_step(&[(0, 0.5), (1, 0.25)], 2, -0.3, lr, 64);
    fa.stimulate(&[(0, -0.35), (1, 0.8)], 64);
    fb.stimulate(&[(0, -0.35), (1, 0.8)], 64);
    let (a, b) = (fa.cells()[2], fb.cells()[2]);
    ((a - b).abs(), a.to_bits() == b.to_bits())
}

#[test]
#[ignore = "research: on demand"]
fn shipped_rule_breaks_the_weight_gain_gauge() {
    // α = 2: forward bit-identical. lr 0.05 → base·(1+Σx²) = 0.066 < κ = 0.5 (uncapped); lr 1.0 → capped.
    let (d_plain, _) = gauge_probe(0.05, 2.0, false);
    let (d_ext, exact_ext) = gauge_probe(0.05, 2.0, true);
    let (d_cap, exact_cap) = gauge_probe(1.0, 2.0, true);
    println!(
        "readout gauge α=2: plain gauge |Δout| = {d_plain:.3e}; extended gauge (lr/α) uncapped |Δout| = {d_ext:.3e} \
         (bit-identical {exact_ext}); extended gauge under the stability cap |Δout| = {d_cap:.3e} (bit-identical \
         {exact_cap})"
    );
    assert!(
        d_plain > 1e-3,
        "the plain gauge is broken by the shipped step"
    );
    assert!(
        exact_ext,
        "the extended gauge (w/α, b/α, αg, lr/α) is exact uncapped"
    );
    assert!(
        d_cap > 1e-3,
        "the cap κ/(1+Σx²) ignores g and breaks the extended gauge"
    );

    // Feature-direction law: gauge-transform the HIDDEN unit of a two-layer morph (extended gauge), learn once.
    let build = |a: f32| {
        let mut f = Field::new(4);
        f.add_unit(vec![0, 1], vec![0.8 / a, -0.55 / a], 0.12 / a, 2);
        f.set_scale(2, a);
        f.set_lr(2, 1.0 / a);
        f.add_unit(vec![2], vec![0.7], -0.05, 3);
        f
    };
    let (mut fa, mut fb) = (build(1.0), build(2.0));
    fa.learn_step(&[(0, 0.4), (1, -0.3)], 3, 0.5, 0.02, 64);
    fb.learn_step(&[(0, 0.4), (1, -0.3)], 3, 0.5, 0.02, 64);
    fa.stimulate(&[(0, -0.6), (1, 0.9)], 64);
    fb.stimulate(&[(0, -0.6), (1, 0.9)], 64);
    let dh = (fa.cells()[2] - fb.cells()[2]).abs();
    println!(
        "feature law, extended gauge α=2 on the hidden unit: |Δh| = {dh:.3e} (the step carries g¹, covariance needs g⁻¹)"
    );
    assert!(
        dh > 1e-4,
        "the feature-direction law is not gauge-covariant"
    );
}
