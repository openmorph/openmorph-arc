// SPDX-License-Identifier: MIT-0
//! TROPICAL FAN — can a grown zero-bias ReLU morph recover the n = 5 amplitude fan?
//! (Experiment 3 of `docs/research/figueiredo-amplitude-geometry.md` §4.)
//!
//! CLAIM TESTED. The only structural link the note allows: a grown zero-bias `Relu` morph trained on the five
//! pentagon headlight functions α_C(t), t ∈ ℝ², by values alone, grows the amplitude's OWN fan — PASS needs both
//! (a) the enumerated linear-region fan of t ↦ α̂ has exactly 5 maximal cones whose rays match the g-vectors
//! within 1e-3 rad, and (b) the curve integral ∫d²t e^{−Σ α̂_C X_C} reproduces A5 = Σ_{5 triangulations}
//! 1/(X_a X_b) to 1e-3 relative at 20 random positive X.
//!
//! STEP 0 — THE TARGET, DERIVED AND CHECKED FIRST (`tropical_fan_ground_truth`). g-vectors of the A₂ fan, in
//! angular order: X13 (1,0), X14 (0,1), X24 (−1,1), X25 (−1,0), X35 (0,−1); consecutive pairs are exactly the five
//! triangulations and every cone is unimodular. The headlights (linear per cone, α_C(g_D) = δ_CD) are, t = (x, y):
//! α13 = max(0,x), α14 = max(0, y+min(0,x)), α24 = max(0, min(−x,y)), α25 = max(0, −x−max(0,y)), α35 = max(0,−y).
//! The integral equals A5: brute Cartesian grid 6.4e-5 (3 X), polar quadrature ∫dθ/h(θ)² 6.0e-11 (20 X), and the
//! stratified-MC estimator used below 6.0e-7 (20 X). The note's [unverified] convention is thus confirmed up to
//! the choice of orientation (the mirror fan with (1,−1) in place of (−1,1) works equally).
//!
//! INSTRUMENT CHECK (`tropical_fan_instrument_on_authored_net`). The tropical formulas wired by hand as a depth-2
//! zero-bias ReLU Field PASS both criteria (5/5/5 functional rays, ray error 0, integral worst 1.4e-6) — the
//! enumerator discards the one raw flip (relu(x+y) at (1,−1)) that the map does not feel. So a FAIL below is a
//! statement about the organism, not the ruler.
//!
//! THE STRUCTURAL OBSTRUCTION (derived, then measured). `grow` builds ONE hidden layer (features read the inputs
//! only). A zero-bias one-hidden-layer ReLU map kinks along whole LINES through the origin, with the same gradient
//! jump on both halves, so its fan is centrally symmetric — an even number of rays. The g-fan has 5 rays, so
//! (a) is impossible for `grow`'s shape, and the missing half-line asymmetry also bounds the values:
//! `tropical_fan_one_layer_floor` (least squares over 720 zero-bias ReLU directions) reaches only rel-RMSE 0.155,
//! with SIX rays — the five g-rays plus the antipode (1,−1) of g24 — and a curve integral off by up to 19 %.
//!
//! RESULTS (3 seeds each; 3×20 000 samples, lr 0.05 → 0.01 → 0.002):
//!   • ARM A, `grow` under the `Relu` genome, biases zeroed by the harness after every event
//!     (`tropical_fan_grown_zero_bias`): rel-RMSE 0.159–0.162 (within 3–5 % of the one-layer floor), 584–606
//!     functional rays at tol 1e-3, all in antipodal pairs, curve integral worst 0.15–0.31. (a) FAIL, (b) FAIL.
//!   • ARM B, `grow` as shipped, biases free (`tropical_fan_grown_free_bias`): rel-RMSE 0.018–0.022 on the
//!     training annulus — the biases buy the asymmetry — but it is not a fan (kink lines 0.31–0.35 from the origin,
//!     median) and the curve integral is off 12–17 %. (a) n/a, (b) FAIL.
//!   • ARM C, a WIRED depth-2 zero-bias ReLU morph (24 + 24 units, authored topology), learned only by
//!     `learn_step` (`tropical_fan_wired_depth2`): rel-RMSE 0.008–0.011, but 88–90 functional rays (layer 1 sits two
//!     hops below the readout, so its random kink lines never move), curve integral worst 0.5–1.4 %. (a) FAIL,
//!     (b) FAIL. A one-off 5× longer run (3×100 000) reached rel-RMSE 0.004–0.006 and integral worst 0.41–0.55 % —
//!     still FAIL at 1e-3, with the ray count unchanged (88–92).
//!   • `tropical_fan_relu_writer_is_born_dead`: under a `Relu` genome, `grow`'s own writer is born at 0 with slope 0
//!     and never learns (every output stays exactly 0). Arms A/B therefore pre-create each writer with an `Identity`
//!     law — the one authored piece besides the bias pinning. This is a limitation of running `grow` under the
//!     `Relu` genome outside its import use (`Activation::Relu` exists to transcribe foreign networks, `config.rs`),
//!     not a confirmed core bug.
//!
//! VERDICT: FAIL. Values can be fit (arm C to <1 %), but "same values" is not "same fan": no arm grows the five
//! g-rays and nothing else, and none reaches the amplitude to 1e-3. What is PROVED is narrow: a one-hidden-layer
//! zero-bias ReLU map has a centrally symmetric fan (an even ray count), the target has 5 rays, so `grow`'s shape
//! cannot pass (a). What is NOT shown: that grown depth is the only route. No learnable-depth arm and no pruning
//! baseline were run (arm C's layer 1 is frozen random directions), and criterion (a) — exactly 5 rays within 1e-3
//! rad — is strict enough that even a backprop-trained depth-2 MLP might fail it; that was not tested. The
//! structural link of the note stays decorative.
//!
//! Authored, stated plainly: the harness zeroes biases (grow cannot pin them); the output writers' `Identity` law;
//! arm C's topology and its random layer-1 directions; the lr schedule. Pure usage of the public API — no core
//! change.
//!
//!   cargo test -p openmorph-core --test research tropical_fan -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use openmorph_core::Activation;

use crate::common::Rng;

// ───────────────────────────────────── the n = 5 fan, derived from first principles ──────────────────────────
//
// The five chords of the pentagon, in the cyclic order in which consecutive chords are compatible (non-crossing):
// {13,14}, {14,24}, {24,25}, {25,35}, {35,13} are exactly the five triangulations. The g-vector fan of the A₂
// cluster algebra (principal coefficients, exchange matrix [[0,1],[−1,0]]) has the five rays below, in the same
// angular order, so each maximal cone is spanned by the g-vectors of one triangulation. Every cone is unimodular
// (det = 1, checked in `ground_truth`), which is what turns the Laplace integral over a cone into 1/(X_a X_b).

/// Chord labels in fan order.
const CHORDS: [&str; 5] = ["X13", "X14", "X24", "X25", "X35"];
/// g-vectors in angular order (0°, 90°, 135°, 180°, 270°).
const G: [[f64; 2]; 5] = [
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0],
];

/// The headlight functions by DEFINITION: linear on each cone of the fan, α_C(g_D) = δ_CD.
fn alpha_hat(t: [f64; 2]) -> [f64; 5] {
    let mut a = [0.0; 5];
    for k in 0..5 {
        let (p, q) = (G[k], G[(k + 1) % 5]);
        let det = p[0] * q[1] - p[1] * q[0];
        let u = (t[0] * q[1] - t[1] * q[0]) / det;
        let v = (p[0] * t[1] - p[1] * t[0]) / det;
        if u >= -1e-12 && v >= -1e-12 {
            a[k] = u.max(0.0);
            a[(k + 1) % 5] = v.max(0.0);
            return a;
        }
    }
    unreachable!("the fan is complete")
}

/// The same five functions as tropical (max/min-plus) expressions, t = (x, y) — derived cone by cone.
fn alpha_tropical(t: [f64; 2]) -> [f64; 5] {
    let (x, y) = (t[0], t[1]);
    [
        x.max(0.0),                 // α13
        (y + x.min(0.0)).max(0.0),  // α14
        (-x).min(y).max(0.0),       // α24
        (-x - y.max(0.0)).max(0.0), // α25
        (-y).max(0.0),              // α35
    ]
}

/// The tree amplitude: Σ over the five triangulations {C_k, C_{k+1}} of 1/(X_a X_b).
fn a5(x: &[f64; 5]) -> f64 {
    (0..5).map(|k| 1.0 / (x[k] * x[(k + 1) % 5])).sum()
}

fn random_x(rng: &mut Rng) -> [f64; 5] {
    let mut x = [0.0; 5];
    for v in x.iter_mut() {
        *v = 0.5 + 1.5 * rng.f() as f64; // positive kinematics in [0.5, 2)
    }
    x
}

/// ∫_{R²} e^{−Σ α_C X_C} d²t for a POSITIVELY HOMOGENEOUS α (degree 1): the radial integral is exact,
/// ∫₀^∞ r e^{−r h(θ)} dr = 1/h(θ)², leaving A = ∫₀^{2π} dθ / h(θ)². `alphas` are α at the unit vectors of
/// `angles`; the estimator is the plain average × 2π (a midpoint rule on a uniform grid, or a stratified
/// Monte-Carlo when the angles are jittered). `None` if h ≤ 0 somewhere (the integral diverges).
fn curve_integral_homog(alphas: &[[f64; 5]], x: &[f64; 5]) -> Option<f64> {
    let mut s = 0.0;
    for a in alphas {
        let h: f64 = (0..5).map(|c| a[c] * x[c]).sum();
        if h <= 0.0 {
            return None;
        }
        s += 1.0 / (h * h);
    }
    Some(s * core::f64::consts::TAU / alphas.len() as f64)
}

/// Stratified-jittered angles: one uniform draw in each of `m` equal strata of [0, 2π).
fn jittered_angles(m: usize, rng: &mut Rng) -> Vec<f64> {
    (0..m)
        .map(|k| (k as f64 + rng.f() as f64) * core::f64::consts::TAU / m as f64)
        .collect()
}

fn dir(th: f64) -> [f64; 2] {
    [th.cos(), th.sin()]
}

// ─────────────────────────────── a read-only f64 mirror of a feed-forward Field ─────────────────────────────
// Reads the PUBLIC `units` (reads, weights, bias, activation) and `scale_of`, evaluates `act(g·(Σw·x+b))` in
// topological order with forward-mode tangents (∂/∂t₀, ∂/∂t₁), and exposes each ReLU unit's pre-activation for
// the fan enumeration. Checked against `stimulate` before it is trusted.

#[derive(Clone)]
struct MUnit {
    reads: Vec<usize>,
    w: Vec<f64>,
    b: f64,
    g: f64,
    act: Activation,
    write: usize,
}

struct Mirror {
    units: Vec<MUnit>, // topological order
    n_cells: usize,
    outputs: [usize; 5],
}

impl Mirror {
    fn from_field(f: &Field, outputs: [usize; 5]) -> Self {
        let n_cells = f.cells().len();
        let mut writer = vec![usize::MAX; n_cells];
        for (i, u) in f.units.iter().enumerate() {
            assert!(u.nested.is_none() && u.factor.is_none() && u.gate.is_none());
            assert_eq!(writer[u.write], usize::MAX, "one writer per cell");
            writer[u.write] = i;
        }
        let genome = f.config().activation;
        assert!(!f.config().gain_fanin_norm);
        // DFS topological order over cells.
        let mut order = Vec::new();
        let mut state = vec![0u8; n_cells];
        fn visit(c: usize, writer: &[usize], f: &Field, st: &mut [u8], order: &mut Vec<usize>) {
            if st[c] == 2 || writer[c] == usize::MAX {
                return;
            }
            assert_ne!(st[c], 1, "feed-forward only");
            st[c] = 1;
            for &r in &f.units[writer[c]].reads {
                visit(r, writer, f, st, order);
            }
            st[c] = 2;
            order.push(writer[c]);
        }
        for &o in &outputs {
            visit(o, &writer, f, &mut state, &mut order);
        }
        let units = order
            .iter()
            .map(|&i| {
                let u = &f.units[i];
                let act = u.activation.unwrap_or(genome);
                assert!(matches!(act, Activation::Relu | Activation::Identity));
                MUnit {
                    reads: u.reads.clone(),
                    w: u.weights.iter().map(|&w| w as f64).collect(),
                    b: u.bias as f64,
                    g: f.scale_of(u.write) as f64,
                    act,
                    write: u.write,
                }
            })
            .collect();
        Mirror {
            units,
            n_cells,
            outputs,
        }
    }

    /// Values, Jacobian (5×2) and the pre-activation of every unit (in `units` order).
    fn eval(&self, t: [f64; 2]) -> ([f64; 5], [[f64; 2]; 5], Vec<f64>) {
        let mut v = vec![0.0; self.n_cells];
        let mut d = vec![[0.0f64; 2]; self.n_cells];
        v[0] = t[0];
        v[1] = t[1];
        d[0] = [1.0, 0.0];
        d[1] = [0.0, 1.0];
        let mut pre = Vec::with_capacity(self.units.len());
        for u in &self.units {
            let mut s = u.b;
            let mut ds = [0.0; 2];
            for (&c, &w) in u.reads.iter().zip(&u.w) {
                s += w * v[c];
                ds[0] += w * d[c][0];
                ds[1] += w * d[c][1];
            }
            s *= u.g;
            ds = [ds[0] * u.g, ds[1] * u.g];
            pre.push(s);
            let (y, dy) = match u.act {
                Activation::Relu if s > 0.0 => (s, ds),
                Activation::Relu => (0.0, [0.0, 0.0]),
                _ => (s, ds),
            };
            v[u.write] = y;
            d[u.write] = dy;
        }
        let mut a = [0.0; 5];
        let mut j = [[0.0; 2]; 5];
        for k in 0..5 {
            a[k] = v[self.outputs[k]];
            j[k] = d[self.outputs[k]];
        }
        (a, j, pre)
    }

    fn relu_mask(&self) -> Vec<bool> {
        self.units
            .iter()
            .map(|u| u.act == Activation::Relu)
            .collect()
    }
}

/// Max |mirror − stimulate| over `n` random points (the mirror must be the Field, not a model of it).
fn mirror_error(f: &mut Field, m: &Mirror, n: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut worst = 0.0f64;
    for _ in 0..n {
        let t = [rng.sym() as f64 * 1.5, rng.sym() as f64 * 1.5];
        f.stimulate(&[(0, t[0] as f32), (1, t[1] as f32)], 1_000_000);
        let (a, _, _) = m.eval(t);
        for (&o, &ak) in m.outputs.iter().zip(&a) {
            worst = worst.max((f.cells()[o] as f64 - ak).abs());
        }
    }
    worst
}

/// The LINEAR-REGION FAN of a zero-bias net, enumerated on the unit circle.
/// Raw rays: every angle where some ReLU unit's pre-activation changes sign (grid of `m` angles, then bisection
/// to ~1e-12 rad). Functional rays: raw rays across which the 5×2 Jacobian of t ↦ α̂ actually jumps by more than
/// `tol` × max‖J‖_F (a flip that the downstream weights cancel is not a region boundary of the map).
struct Fan {
    raw: Vec<f64>,
    jumps: Vec<f64>, // relative ‖ΔJ‖_F at each raw ray
}

impl Fan {
    fn functional(&self, tol: f64) -> Vec<f64> {
        self.raw
            .iter()
            .zip(&self.jumps)
            .filter(|(_, &j)| j > tol)
            .map(|(&r, _)| r)
            .collect()
    }
}

fn enumerate_fan(m: &Mirror, grid: usize) -> Fan {
    let relu = m.relu_mask();
    let tau = core::f64::consts::TAU;
    let sign = |th: f64| -> Vec<bool> { m.eval(dir(th)).2.iter().map(|&p| p > 0.0).collect() };
    // The grid is offset by an irrational fraction of a step so no ray (the g-vectors are at rational angles) can
    // sit exactly on a grid point, where a sign flip would be invisible.
    let at = |k: usize| tau * (k as f64 + 0.377_964_473) / grid as f64;
    let mut raw: Vec<f64> = Vec::new();
    let mut prev = sign(at(0));
    for k in 1..=grid {
        let (th0, th1) = (at(k - 1), at(k));
        let cur = sign(th1);
        for u in 0..prev.len() {
            if relu[u] && prev[u] != cur[u] {
                let (mut lo, mut hi) = (th0, th1);
                for _ in 0..50 {
                    let mid = 0.5 * (lo + hi);
                    if (m.eval(dir(mid)).2[u] > 0.0) == prev[u] {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                raw.push((0.5 * (lo + hi)).rem_euclid(tau));
            }
        }
        prev = cur;
    }
    raw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    raw.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let jn = |j: &[[f64; 2]; 5]| {
        j.iter()
            .map(|r| r[0] * r[0] + r[1] * r[1])
            .sum::<f64>()
            .sqrt()
    };
    let jmax = (0..grid)
        .map(|k| jn(&m.eval(dir(tau * (k as f64 + 0.5) / grid as f64)).1))
        .fold(0.0, f64::max)
        .max(1e-30);
    let delta = 1e-7;
    let jumps = raw
        .iter()
        .map(|&r| {
            let (_, a, _) = m.eval(dir(r - delta));
            let (_, b, _) = m.eval(dir(r + delta));
            let mut s = 0.0;
            for k in 0..5 {
                for c in 0..2 {
                    s += (a[k][c] - b[k][c]).powi(2);
                }
            }
            s.sqrt() / jmax
        })
        .collect();
    Fan { raw, jumps }
}

fn ang_dist(a: f64, b: f64) -> f64 {
    let tau = core::f64::consts::TAU;
    let d = (a - b).rem_euclid(tau);
    d.min(tau - d)
}

fn g_angles() -> [f64; 5] {
    let mut a = [0.0; 5];
    for k in 0..5 {
        a[k] = G[k][1].atan2(G[k][0]).rem_euclid(core::f64::consts::TAU);
    }
    a
}

/// Worst angular distance from each g-vector ray to its nearest functional ray.
fn ray_match(rays: &[f64]) -> f64 {
    g_angles()
        .iter()
        .map(|&g| {
            rays.iter()
                .map(|&r| ang_dist(r, g))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

/// Is a ray set centrally symmetric (every ray has an antipode within `eps`)?
fn centrally_symmetric(rays: &[f64], eps: f64) -> bool {
    rays.iter().all(|&r| {
        rays.iter()
            .any(|&s| ang_dist(s, r + core::f64::consts::PI) < eps)
    })
}

/// The whole scorecard for a zero-bias net: value fit, fan (a), curve integral (b).
struct Score {
    rel_rmse: f64,
    raw_rays: usize,
    func_rays: [usize; 3], // at tol 1e-2, 1e-3, 1e-4
    ray_err: f64,          // at tol 1e-3
    symmetric: bool,       // functional rays (tol 1e-3) centrally symmetric
    worst_int_err: f64,    // max over 20 X of |Â − A5| / A5 (∞ if divergent)
    median_int_err: f64,
}

fn score(m: &Mirror, seed: u64) -> Score {
    let mut rng = Rng::new(seed);
    // value fit on the unit circle (α is homogeneous, so the circle is the whole function)
    let (mut se, mut ss) = (0.0, 0.0);
    for th in jittered_angles(4096, &mut rng) {
        let (a, _, _) = m.eval(dir(th));
        let t = alpha_hat(dir(th));
        for k in 0..5 {
            se += (a[k] - t[k]).powi(2);
            ss += t[k] * t[k];
        }
    }
    let fan = enumerate_fan(m, 1 << 14);
    let f3 = fan.functional(1e-3);
    let angles = jittered_angles(1 << 16, &mut rng);
    let alphas: Vec<[f64; 5]> = angles.iter().map(|&th| m.eval(dir(th)).0).collect();
    let mut errs: Vec<f64> = (0..20)
        .map(|_| {
            let x = random_x(&mut rng);
            curve_integral_homog(&alphas, &x).map_or(f64::INFINITY, |v| (v - a5(&x)).abs() / a5(&x))
        })
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Score {
        rel_rmse: (se / ss).sqrt(),
        raw_rays: fan.raw.len(),
        func_rays: [
            fan.functional(1e-2).len(),
            f3.len(),
            fan.functional(1e-4).len(),
        ],
        ray_err: ray_match(&f3),
        symmetric: centrally_symmetric(&f3, 1e-6),
        worst_int_err: errs[19],
        median_int_err: 0.5 * (errs[9] + errs[10]),
    }
}

fn report(name: &str, s: &Score) {
    let pass_a = s.func_rays[1] == 5 && s.ray_err < 1e-3;
    let pass_b = s.worst_int_err < 1e-3;
    println!(
        "{name}: value rel-RMSE {:.2e} | raw rays {} | functional rays @1e-2/1e-3/1e-4 = {}/{}/{} | \
         g-ray match {:.2e} rad | centrally symmetric {} | curve integral rel err median {:.2e} worst {:.2e} \
         | (a) {} (b) {}",
        s.rel_rmse,
        s.raw_rays,
        s.func_rays[0],
        s.func_rays[1],
        s.func_rays[2],
        s.ray_err,
        s.symmetric,
        s.median_int_err,
        s.worst_int_err,
        if pass_a { "PASS" } else { "FAIL" },
        if pass_b { "PASS" } else { "FAIL" },
    );
}

// ─────────────────────────────────────────────────── tests ───────────────────────────────────────────────────

/// STEP 0 — the target, verified BEFORE anything is trained.
#[test]
#[ignore = "research: on demand"]
fn tropical_fan_ground_truth() {
    // unimodular cones, and the tropical formulas equal the hat definition everywhere
    for k in 0..5 {
        let (p, q) = (G[k], G[(k + 1) % 5]);
        assert_eq!(p[0] * q[1] - p[1] * q[0], 1.0, "cone {k} unimodular");
        for (d, g) in G.iter().enumerate() {
            let a = alpha_tropical(*g);
            for c in 0..5 {
                assert_eq!(
                    a[c],
                    if c == d { 1.0 } else { 0.0 },
                    "α_{}(g_{})",
                    CHORDS[c],
                    CHORDS[d]
                );
            }
        }
    }
    let mut rng = Rng::new(0xFA11);
    let mut worst_form = 0.0f64;
    for _ in 0..100_000 {
        let t = [rng.sym() as f64 * 3.0, rng.sym() as f64 * 3.0];
        let (h, tr) = (alpha_hat(t), alpha_tropical(t));
        for c in 0..5 {
            worst_form = worst_form.max((h[c] - tr[c]).abs());
        }
    }
    // (i) brute-force 2-D Cartesian midpoint rule over [−L, L]² — no polar reduction, no cone decomposition
    let (l, step) = (50.0f64, 0.025f64);
    let n = (2.0 * l / step) as usize;
    let mut worst_grid = 0.0f64;
    for _ in 0..3 {
        let x = random_x(&mut rng);
        let mut s = 0.0;
        for i in 0..n {
            let t0 = -l + (i as f64 + 0.5) * step;
            for j in 0..n {
                let t1 = -l + (j as f64 + 0.5) * step;
                let a = alpha_tropical([t0, t1]);
                s += (-(0..5).map(|c| a[c] * x[c]).sum::<f64>()).exp();
            }
        }
        let v = s * step * step;
        worst_grid = worst_grid.max((v - a5(&x)).abs() / a5(&x));
    }
    // (ii) the polar reduction ∫dθ/h², midpoint on 2²⁰ angles, at 20 random X; and (iii) the stratified MC
    // estimator used on the grown nets (2¹⁶ jittered angles), to know its own error floor
    let m = 1 << 20;
    let quad: Vec<[f64; 5]> = (0..m)
        .map(|k| alpha_tropical(dir((k as f64 + 0.5) * core::f64::consts::TAU / m as f64)))
        .collect();
    let mc: Vec<[f64; 5]> = jittered_angles(1 << 16, &mut rng)
        .into_iter()
        .map(|th| alpha_tropical(dir(th)))
        .collect();
    let (mut worst_polar, mut worst_mc) = (0.0f64, 0.0f64);
    for _ in 0..20 {
        let x = random_x(&mut rng);
        let a = a5(&x);
        worst_polar = worst_polar.max((curve_integral_homog(&quad, &x).unwrap() - a).abs() / a);
        worst_mc = worst_mc.max((curve_integral_homog(&mc, &x).unwrap() - a).abs() / a);
    }
    println!(
        "ground truth: |hat − tropical| ≤ {worst_form:.1e}; ∫e^(−Σα_C X_C) vs A5 rel err: Cartesian grid {worst_grid:.2e} \
         (3 X), polar quadrature {worst_polar:.2e} (20 X), stratified MC estimator {worst_mc:.2e} (20 X)"
    );
    assert!(worst_form < 1e-12);
    assert!(worst_grid < 1e-3);
    assert!(worst_polar < 1e-6);
    assert!(worst_mc < 1e-4);
}

/// Instrument check: an AUTHORED depth-2 zero-bias ReLU Field (the tropical formulas wired unit for unit) must
/// PASS both criteria — otherwise a FAIL below would say nothing about growth. Raw ReLU flips include the
/// spurious ray (1,−1) (the unit relu(x+y) flips there), which the functional test must discard.
#[test]
#[ignore = "research: on demand"]
fn tropical_fan_instrument_on_authored_net() {
    let mut f = Field::new(10);
    f.config_mut().activation = Activation::Relu;
    f.add_unit(vec![0], vec![-1.0], 0.0, 2); // relu(−x)
    f.add_unit(vec![0, 1], vec![1.0, 1.0], 0.0, 3); // relu(x+y)
    f.add_unit(vec![1], vec![1.0], 0.0, 4); // relu(y)
    f.add_unit(vec![0], vec![1.0], 0.0, 5); // α13 = relu(x)
    f.add_unit(vec![1, 2], vec![1.0, -1.0], 0.0, 6); // α14 = relu(y − relu(−x))
    f.add_unit(vec![1, 3], vec![1.0, -1.0], 0.0, 7); // α24 = relu(y − relu(x+y)) = relu(min(−x, y))
    f.add_unit(vec![0, 4], vec![-1.0, -1.0], 0.0, 8); // α25 = relu(−x − relu(y))
    f.add_unit(vec![1], vec![-1.0], 0.0, 9); // α35 = relu(−y)
    let m = Mirror::from_field(&f, [5, 6, 7, 8, 9]);
    let me = mirror_error(&mut f, &m, 500, 7);
    let s = score(&m, 11);
    println!("mirror vs stimulate max |Δ| = {me:.1e}");
    report("authored depth-2 net", &s);
    assert!(me < 1e-6);
    assert!(s.rel_rmse < 1e-6);
    assert_eq!(s.raw_rays, 6);
    assert_eq!(s.func_rays, [5, 5, 5]);
    assert!(s.ray_err < 1e-9);
    assert!(s.worst_int_err < 1e-4);
}

// ─────────────────────────────────────────── the grown / learned arms ──────────────────────────────────────────

const OUT: [usize; 5] = [2, 3, 4, 5, 6];
const BUDGET: usize = 1_000_000;

/// A training sample: t on an annulus (the target is homogeneous, so the radius only exercises that property).
fn sample(rng: &mut Rng) -> ([f64; 2], [f64; 5]) {
    let th = rng.f() as f64 * core::f64::consts::TAU;
    let r = 0.5 + rng.f() as f64;
    let t = [r * th.cos(), r * th.sin()];
    (t, alpha_hat(t))
}

fn pin_biases(f: &mut Field) {
    for u in f.units.iter_mut() {
        u.bias = 0.0;
    }
}

/// lr schedule: three plateaus (0.05 → 0.01 → 0.002), `steps` samples each.
fn lr_at(k: usize, steps: usize) -> f32 {
    [0.05, 0.01, 0.002][(k / steps).min(2)]
}

/// ARM A/B — `grow` under a `Relu` genome, one grown pool per headlight output. The one authored piece: each
/// output's writer is pre-created (zero weights, as `grow` would) with an `Identity` law, because a `Relu` writer
/// born at zero has slope 0 and never learns (measured: every output stays 0). Hidden units are born, weighted
/// and directed by `grow` alone. `pin` = the harness zeroes every bias after each event (grow cannot pin them).
fn grow_arm(seed: u64, steps: usize, pin: bool) -> Field {
    let mut f = Field::new(7);
    f.rng_seed(seed);
    f.config_mut().activation = Activation::Relu;
    for &o in &OUT {
        f.add_unit(vec![0, 1], vec![0.0, 0.0], 0.0, o);
        f.set_activation(o, Some(Activation::Identity));
    }
    let mut rng = Rng::new(seed ^ 0x5EED);
    for k in 0..3 * steps {
        let (t, a) = sample(&mut rng);
        let inp = [(0, t[0] as f32), (1, t[1] as f32)];
        for c in 0..5 {
            f.grow(&inp, OUT[c], a[c] as f32, lr_at(k, steps), BUDGET);
        }
        if pin {
            pin_biases(&mut f);
        }
    }
    f
}

/// The negative control of the pre-created writer: `grow` with its own (Relu) writer never moves off 0.
#[test]
#[ignore = "research: on demand"]
fn tropical_fan_relu_writer_is_born_dead() {
    let mut f = Field::new(7);
    f.rng_seed(1);
    f.config_mut().activation = Activation::Relu;
    let mut rng = Rng::new(2);
    for _ in 0..2000 {
        let (t, a) = sample(&mut rng);
        let inp = [(0, t[0] as f32), (1, t[1] as f32)];
        for c in 0..5 {
            f.grow(&inp, OUT[c], a[c] as f32, 0.05, BUDGET);
        }
    }
    let mut max_out = 0.0f32;
    for _ in 0..200 {
        let (t, _) = sample(&mut rng);
        f.stimulate(&[(0, t[0] as f32), (1, t[1] as f32)], BUDGET);
        for &o in &OUT {
            max_out = max_out.max(f.cells()[o].abs());
        }
    }
    let w_nonzero = OUT
        .iter()
        .flat_map(|&o| f.units.iter().filter(move |u| u.write == o))
        .flat_map(|u| u.weights.iter())
        .filter(|w| **w != 0.0)
        .count();
    println!("Relu-genome writer: max |output| {max_out:.1e}, nonzero writer weights {w_nonzero}, units {}", f.units.len());
    assert_eq!(max_out, 0.0);
    assert_eq!(w_nonzero, 0);
}

/// ARM A — grown, biases pinned to 0.
#[test]
#[ignore = "research: on demand"]
fn tropical_fan_grown_zero_bias() {
    for seed in [1u64, 3, 5] {
        let mut f = grow_arm(seed, 20_000, true);
        let m = Mirror::from_field(&f, OUT);
        let me = mirror_error(&mut f, &m, 300, seed);
        let s = score(&m, seed + 100);
        println!("seed {seed}: units {}, mirror |Δ| {me:.1e}", f.units.len());
        report("  grown zero-bias (one hidden layer)", &s);
        // measured: rel-RMSE 0.159–0.162 (the one-layer floor is 0.155), 584–606 functional rays, all in
        // antipodal pairs, curve integral worst 0.15–0.31
        assert!(
            s.symmetric,
            "a one-hidden-layer zero-bias fan is centrally symmetric"
        );
        assert!(s.func_rays[1] > 5);
        assert!(s.rel_rmse > 0.15 && s.rel_rmse < 0.17);
        assert!(s.worst_int_err > 0.1);
    }
}

/// The REPRESENTATION FLOOR of the grown topology: the least-squares best fit of the headlights by ONE hidden layer
/// of zero-bias ReLUs over 720 directions (every 0.5°, so every g-vector line and its antipode is present),
/// written into a Field as an authored readout. No learning is involved: this bounds what any one-hidden-layer
/// zero-bias net — `grow`'s shape — can reach.
#[test]
#[ignore = "research: on demand"]
fn tropical_fan_one_layer_floor() {
    let k = 720;
    let tau = core::f64::consts::TAU;
    let u: Vec<[f64; 2]> = (0..k).map(|i| dir(tau * i as f64 / k as f64)).collect();
    let pts: Vec<[f64; 2]> = (0..8192)
        .map(|i| dir(tau * (i as f64 + 0.5) / 8192.0))
        .collect();
    let phi = |t: [f64; 2]| -> Vec<f64> {
        u.iter()
            .map(|d| (d[0] * t[0] + d[1] * t[1]).max(0.0))
            .collect()
    };
    // normal equations (ΦᵀΦ + λI) w = Φᵀy, Cholesky
    let mut ata = vec![0.0f64; k * k];
    let mut aty = vec![[0.0f64; 5]; k];
    for &t in &pts {
        let p = phi(t);
        let y = alpha_hat(t);
        for i in 0..k {
            if p[i] == 0.0 {
                continue;
            }
            for j in 0..k {
                ata[i * k + j] += p[i] * p[j];
            }
            for c in 0..5 {
                aty[i][c] += p[i] * y[c];
            }
        }
    }
    for i in 0..k {
        ata[i * k + i] += 1e-9;
    }
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = ata[i * k + j];
            for q in 0..j {
                s -= l[i * k + q] * l[j * k + q];
            }
            l[i * k + j] = if i == j { s.sqrt() } else { s / l[j * k + j] };
        }
    }
    let mut f = Field::new(7);
    f.config_mut().activation = Activation::Relu;
    let hidden: Vec<usize> = (0..k)
        .map(|i| {
            let c = 7 + i;
            f.add_unit(vec![0, 1], vec![u[i][0] as f32, u[i][1] as f32], 0.0, c);
            c
        })
        .collect();
    for c in 0..5 {
        let mut z = vec![0.0f64; k];
        for i in 0..k {
            let mut s = aty[i][c];
            for q in 0..i {
                s -= l[i * k + q] * z[q];
            }
            z[i] = s / l[i * k + i];
        }
        let mut w = vec![0.0f64; k];
        for i in (0..k).rev() {
            let mut s = z[i];
            for q in i + 1..k {
                s -= l[q * k + i] * w[q];
            }
            w[i] = s / l[i * k + i];
        }
        f.add_unit(
            hidden.clone(),
            w.iter().map(|&x| x as f32).collect(),
            0.0,
            OUT[c],
        );
        f.set_activation(OUT[c], Some(Activation::Identity));
    }
    let m = Mirror::from_field(&f, OUT);
    let s = score(&m, 77);
    report("LS floor, one zero-bias ReLU layer (720 dirs)", &s);
    // measured: the five g-rays PLUS the antipode (1,−1) of g24 — six rays, rel-RMSE 0.155, integral worst 0.19
    assert_eq!(s.func_rays, [6, 6, 6]);
    assert!(s.symmetric && s.ray_err < 1e-9);
    assert!(s.rel_rmse > 0.15 && s.rel_rmse < 0.16);
    assert!(s.worst_int_err > 0.1);
}

/// ARM B — grown as shipped: `Relu` genome, biases NOT pinned (random at birth, learned). Not a fan (the kink
/// lines miss the origin), so only the value fit and the general curve integral are scored.
#[test]
#[ignore = "research: on demand"]
fn tropical_fan_grown_free_bias() {
    for seed in [1u64, 3, 5] {
        let mut f = grow_arm(seed, 20_000, false);
        let m = Mirror::from_field(&f, OUT);
        let me = mirror_error(&mut f, &m, 300, seed);
        let mut rng = Rng::new(seed + 200);
        let (mut se, mut ss) = (0.0, 0.0);
        for _ in 0..4096 {
            let (t, a) = sample(&mut rng);
            let (p, _, _) = m.eval(t);
            for c in 0..5 {
                se += (p[c] - a[c]).powi(2);
                ss += a[c] * a[c];
            }
        }
        let off: Vec<f64> = m
            .units
            .iter()
            .filter(|u| u.act == Activation::Relu)
            .map(|u| u.b.abs() / u.w.iter().map(|w| w * w).sum::<f64>().sqrt().max(1e-30))
            .collect();
        let mut sorted = off.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // general curve integral: θ stratified (1024), r trapezoid on [0, 40] (2000 steps)
        let angles = jittered_angles(1024, &mut rng);
        let nr = 2000;
        let rmax = 40.0;
        let hs: Vec<Vec<[f64; 5]>> = angles
            .iter()
            .map(|&th| {
                (0..=nr)
                    .map(|i| {
                        m.eval([
                            rmax * i as f64 / nr as f64 * th.cos(),
                            rmax * i as f64 / nr as f64 * th.sin(),
                        ])
                        .0
                    })
                    .collect()
            })
            .collect();
        let gt: Vec<Vec<[f64; 5]>> = angles
            .iter()
            .map(|&th| {
                (0..=nr)
                    .map(|i| {
                        alpha_hat([
                            rmax * i as f64 / nr as f64 * th.cos(),
                            rmax * i as f64 / nr as f64 * th.sin(),
                        ])
                    })
                    .collect()
            })
            .collect();
        let integ = |tab: &Vec<Vec<[f64; 5]>>, x: &[f64; 5]| -> f64 {
            let dr = rmax / nr as f64;
            let mut s = 0.0;
            for row in tab {
                for (i, a) in row.iter().enumerate() {
                    let r = i as f64 * dr;
                    let w = if i == 0 || i == nr { 0.5 } else { 1.0 };
                    s += w * r * (-(0..5).map(|c| a[c] * x[c]).sum::<f64>()).exp();
                }
            }
            s * dr * core::f64::consts::TAU / tab.len() as f64
        };
        let (mut worst, mut worst_gt) = (0.0f64, 0.0f64);
        for _ in 0..20 {
            let x = random_x(&mut rng);
            worst = worst.max((integ(&hs, &x) - a5(&x)).abs() / a5(&x));
            worst_gt = worst_gt.max((integ(&gt, &x) - a5(&x)).abs() / a5(&x));
        }
        let rel = (se / ss).sqrt();
        let med_off = sorted[sorted.len() / 2];
        println!(
            "seed {seed}: units {}, mirror |Δ| {me:.1e}, value rel-RMSE (annulus) {rel:.2e}, ReLU kink-line offset from origin median {med_off:.2e} max {:.2e}, curve integral worst rel err {worst:.2e} (estimator on truth {worst_gt:.2e})",
            f.units.len(),
            sorted[sorted.len() - 1]
        );
        // measured: rel-RMSE 0.018–0.022 on the training annulus, kink lines 0.31–0.35 from the origin (median),
        // curve integral worst 0.12–0.17 (the estimator's own error on the truth ≤ 1.2e-3)
        assert!(rel < 0.025);
        assert!(med_off > 0.2);
        assert!(worst > 0.1 && worst_gt < 2e-3);
    }
}

/// ARM C — a WIRED depth-2 zero-bias ReLU morph (authored topology: 24 random-direction ReLUs on t, 24 ReLUs on
/// those, an Identity readout per headlight — the depth the tropical formulas need), learned ONLY through the
/// library's `learn_step` (readout delta rule + the one-hop feature-direction step; layer 1 is two hops down and
/// stays at its random init). Biases pinned by the harness after every event.
#[test]
#[ignore = "research: on demand"]
fn tropical_fan_wired_depth2() {
    let (k1, k2) = (24usize, 24usize);
    for seed in [1u64, 3, 5] {
        let mut f = Field::new(7);
        f.rng_seed(seed);
        f.config_mut().activation = Activation::Relu;
        let mut rng = Rng::new(seed ^ 0xC0FFEE);
        let l1: Vec<usize> = (0..k1)
            .map(|i| {
                let th = rng.f() * core::f32::consts::TAU;
                f.add_unit(vec![0, 1], vec![th.cos(), th.sin()], 0.0, 7 + i);
                7 + i
            })
            .collect();
        let gain = 3.0 / (k1 as f32).sqrt();
        let l2: Vec<usize> = (0..k2)
            .map(|i| {
                let w: Vec<f32> = (0..k1).map(|_| rng.sym() * gain).collect();
                f.add_unit(l1.clone(), w, 0.0, 7 + k1 + i);
                7 + k1 + i
            })
            .collect();
        for &o in &OUT {
            f.add_unit(l2.clone(), vec![0.0; k2], 0.0, o);
            f.set_activation(o, Some(Activation::Identity));
        }
        let steps = 20_000;
        for k in 0..3 * steps {
            let (t, a) = sample(&mut rng);
            let inp = [(0, t[0] as f32), (1, t[1] as f32)];
            for c in 0..5 {
                f.learn_step(&inp, OUT[c], a[c] as f32, lr_at(k, steps), BUDGET);
            }
            pin_biases(&mut f);
        }
        let m = Mirror::from_field(&f, OUT);
        let me = mirror_error(&mut f, &m, 300, seed);
        let s = score(&m, seed + 300);
        println!("seed {seed}: units {}, mirror |Δ| {me:.1e}", f.units.len());
        report("  wired depth-2 zero-bias", &s);
        // measured: rel-RMSE 0.008–0.011, 88–90 functional rays, curve integral worst 0.005–0.014
        assert!(s.rel_rmse < 0.012);
        assert!(s.func_rays[1] > 5);
        assert!(s.worst_int_err > 1e-3 && s.worst_int_err < 0.015);
    }
}
