// SPDX-License-Identifier: MIT-0
//! CURVATURE & TIME — the general-relativity arc of the "Field Theory of the Digital Organism" spike.
//! Two coupled claims, each tested with a MASS *PROXY* so nothing here depends on the parallel mass spike:
//!
//!   (A) CURVATURE — if "mass"/importance warps the concept-map, HIERARCHY wants NEGATIVE (hyperbolic)
//!       curvature. Trees embed in hyperbolic space (the Poincaré disk) with far less distortion than in
//!       flat Euclidean space, because hyperbolic volume grows exponentially with radius — the same rate a
//!       tree's node-count grows with depth. We embed a known TREE and a known GRID under the SAME stress
//!       loss, changing ONLY the distance function (Euclidean vs Poincaré), and measure distortion +
//!       retrieval. The scientific claim is not "hyperbolic is better" but "hyperbolic is better *for
//!       hierarchy specifically*": the advantage must be LARGE on the tree and SMALL on the flat grid.
//!       This revisits the S6 hyperbolic block that #172 DEFERRED (`docs/investigations/172-...`).
//!
//!   (B) TIME DILATION — the organism's subjective time is its event count. A busy / "massive"
//!       (high-activity) region emits MORE events, so a fixed spatial motion is sampled into SMALL
//!       per-event steps there and LARGE steps in sparse regions. Proper time ≠ coordinate (event) time.
//!       We test whether warping event-time by a LOCALLY-MEASURED activity field (the co-activation /
//!       reliance frequency the organism already accumulates — NOT the hidden mass) improves next-state
//!       prediction over the uniform-event-time momentum extrapolation of `substrate_trajectory`. Guarded
//!       by a FLAT-mass control where the warp must NOT help.
//!
//! TRUTH-NOT-TROPHY: both results are reported with honest numbers and a falsification control. The mass
//! variable is a PROXY here (tree structure in A, measured event-density in B); the finale is when these
//! compose with the REAL grown mass/importance variable from the parallel spike.
//!
//!   cargo test -p openmorph-core --test research substrate_curvature_time -- --ignored --nocapture --test-threads=1

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ======================================================================================
//  Shared geometry helpers (test-local — a hyperbolic distance is a pure function).
// ======================================================================================

/// Flat Euclidean distance.
fn euclid(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt()
}

/// Poincaré-ball (curvature −1) distance, in the NUMERICALLY STABLE form #172 identified as the fix:
/// `d = arcosh(1+u)` computed as `ln1p(u + √(u·(u+2)))`, where `u = 2‖x−y‖² / ((1−‖x‖²)(1−‖y‖²))` is
/// formed DIRECTLY (never as `1+u`, which rounds away the small-`u` information for nearby points).
fn poincare(x: &[f32], y: &[f32]) -> f32 {
    let (nx2, ny2, dxy2) = ball_terms(x, y);
    let denom = ((1.0 - nx2) * (1.0 - ny2)).max(1e-9);
    let u = 2.0 * dxy2 / denom;
    (u + (u * (u + 2.0)).sqrt()).ln_1p()
}

/// The NAIVE Poincaré form `arcosh(1+u) = ln(z + √(z²−1))`, `z = 1+u` — kept only to MEASURE its
/// fragility against the stable form (the #172 numerical finding).
fn poincare_naive(x: &[f32], y: &[f32]) -> f32 {
    let (nx2, ny2, dxy2) = ball_terms(x, y);
    let denom = ((1.0 - nx2) * (1.0 - ny2)).max(1e-9);
    let u = 2.0 * dxy2 / denom;
    let z = 1.0 + u;
    (z + (z * z - 1.0).sqrt()).ln()
}

fn ball_terms(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
    let nx2 = x.iter().map(|a| a * a).sum::<f32>();
    let ny2 = y.iter().map(|a| a * a).sum::<f32>();
    let dxy2 = x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
    (nx2, ny2, dxy2)
}

/// f64 reference of the stable Poincaré distance (the ground truth the f32 forms are graded against).
fn poincare_ref64(x: &[f32], y: &[f32]) -> f64 {
    let nx2: f64 = x.iter().map(|a| (*a as f64) * (*a as f64)).sum();
    let ny2: f64 = y.iter().map(|a| (*a as f64) * (*a as f64)).sum();
    let dxy2: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum();
    let u = 2.0 * dxy2 / ((1.0 - nx2) * (1.0 - ny2));
    (u + (u * (u + 2.0)).sqrt()).ln_1p()
}

// ======================================================================================
//  (A) CURVATURE — embed a hierarchy in Euclidean vs hyperbolic space, same stress loss.
// ======================================================================================

/// A weighted-graph dataset: adjacency (unit edges) + all-pairs shortest-path "true" distances.
struct Graph {
    n: usize,
    adj: Vec<Vec<usize>>,
    dist: Vec<Vec<f32>>,
}

impl Graph {
    fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        // BFS all-pairs (unit weights).
        let mut dist = vec![vec![f32::INFINITY; n]; n];
        #[allow(clippy::needless_range_loop)]
        for s in 0..n {
            let mut q = std::collections::VecDeque::new();
            dist[s][s] = 0.0;
            q.push_back(s);
            while let Some(u) = q.pop_front() {
                for &v in &adj[u] {
                    if dist[s][v].is_infinite() {
                        dist[s][v] = dist[s][u] + 1.0;
                        q.push_back(v);
                    }
                }
            }
        }
        Graph { n, adj, dist }
    }

    /// A complete b-ary tree of `depth` levels (root = 0). Deeply hierarchical.
    fn bary_tree(b: usize, depth: usize) -> Self {
        let mut edges = Vec::new();
        let mut nodes = vec![0usize]; // level 0
        let mut next = 1usize;
        for _ in 0..depth {
            let mut children = Vec::new();
            for &p in &nodes {
                for _ in 0..b {
                    edges.push((p, next));
                    children.push(next);
                    next += 1;
                }
            }
            nodes = children;
        }
        Graph::from_edges(next, &edges)
    }

    /// An `l×l` grid graph (4-connectivity). FLAT — Euclidean's home turf; the negative control.
    fn grid(l: usize) -> Self {
        let mut edges = Vec::new();
        let idx = |r: usize, c: usize| r * l + c;
        for r in 0..l {
            for c in 0..l {
                if c + 1 < l {
                    edges.push((idx(r, c), idx(r, c + 1)));
                }
                if r + 1 < l {
                    edges.push((idx(r, c), idx(r + 1, c)));
                }
            }
        }
        Graph::from_edges(l * l, &edges)
    }
}

/// Best global scale α = argmin Σ(d_emb − α·d_true)², closed form.
fn best_scale(demb: &[f32], dtrue: &[f32]) -> f32 {
    let num: f32 = demb.iter().zip(dtrue).map(|(d, t)| d * t).sum();
    let den: f32 = dtrue.iter().map(|t| t * t).sum::<f32>().max(1e-9);
    num / den
}

/// Pearson correlation of two equal-length vectors.
fn corr(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let (mx, my) = (xs.iter().sum::<f32>() / n, ys.iter().sum::<f32>() / n);
    let mut num = 0.0f32;
    let (mut dx2, mut dy2) = (0.0f32, 0.0f32);
    for (x, y) in xs.iter().zip(ys) {
        num += (x - mx) * (y - my);
        dx2 += (x - mx) * (x - mx);
        dy2 += (y - my) * (y - my);
    }
    num / (dx2.sqrt() * dy2.sqrt()).max(1e-9)
}

/// One embedding's quality against a graph's true distances.
struct Fit {
    /// Pearson corr(embedded dist, true dist) over all pairs — higher = more faithful ordering.
    correlation: f32,
    /// Mean multiplicative distortion |d_emb/(α·d_true) − 1| over all pairs — lower = less warped.
    distortion: f32,
    /// Retrieval fidelity: fraction of nodes whose single nearest neighbour is a TRUE graph neighbour.
    retrieval: f32,
}

fn evaluate(g: &Graph, coords: &[Vec<f32>], dist_fn: &dyn Fn(&[f32], &[f32]) -> f32) -> Fit {
    let (mut de, mut dt) = (Vec::new(), Vec::new());
    for i in 0..g.n {
        for j in (i + 1)..g.n {
            de.push(dist_fn(&coords[i], &coords[j]));
            dt.push(g.dist[i][j]);
        }
    }
    let alpha = best_scale(&de, &dt);
    let correlation = corr(&de, &dt);
    let distortion = de
        .iter()
        .zip(&dt)
        .map(|(d, t)| ((d / (alpha * t)) - 1.0).abs())
        .sum::<f32>()
        / de.len() as f32;
    // Retrieval: is each node's nearest embedded neighbour a real graph edge?
    let mut hits = 0usize;
    for i in 0..g.n {
        let nn = (0..g.n)
            .filter(|&j| j != i)
            .min_by(|&a, &b| {
                dist_fn(&coords[i], &coords[a]).total_cmp(&dist_fn(&coords[i], &coords[b]))
            })
            .unwrap();
        if g.adj[i].contains(&nn) {
            hits += 1;
        }
    }
    Fit {
        correlation,
        distortion,
        retrieval: hits as f32 / g.n as f32,
    }
}

/// Refit the global scale α for the current coords.
fn refit_alpha(g: &Graph, coords: &[Vec<f32>], dist_fn: &dyn Fn(&[f32], &[f32]) -> f32) -> f32 {
    let (mut de, mut dt) = (Vec::new(), Vec::new());
    for i in 0..g.n {
        for j in (i + 1)..g.n {
            de.push(dist_fn(&coords[i], &coords[j]));
            dt.push(g.dist[i][j]);
        }
    }
    best_scale(&de, &dt)
}

/// ∂d_ij/∂x_i for the Euclidean metric: `(x_i − x_j)/d`.
fn dgrad_euclid(xi: &[f32], xj: &[f32]) -> Vec<f32> {
    let d = euclid(xi, xj).max(1e-6);
    xi.iter().zip(xj).map(|(a, b)| (a - b) / d).collect()
}

/// ∂d_ij/∂x_i for the Poincaré metric (curvature −1). Analytic chain rule through
/// `d = arcosh(1+u)`, `u = 2D/(A·B)` with `D=‖xi−xj‖²`, `A=1−‖xi‖²`, `B=1−‖xj‖²`.
fn dgrad_poincare(xi: &[f32], xj: &[f32]) -> Vec<f32> {
    let (nx2, ny2, dd) = ball_terms(xi, xj);
    let a = (1.0 - nx2).max(1e-6);
    let b = (1.0 - ny2).max(1e-6);
    let u = 2.0 * dd / (a * b);
    let dddu = 1.0 / (u * (u + 2.0)).sqrt().max(1e-6); // d/du arcosh(1+u)
    let k = 4.0 / (b * a * a); // ∂u/∂xi = k·[A·(xi−xj) + D·xi]
    xi.iter()
        .zip(xj)
        .map(|(&pi, &pj)| dddu * k * (a * (pi - pj) + dd * pi))
        .collect()
}

/// Embed `g` into `dim` dimensions by RIEMANNIAN gradient descent on the stress
/// `Σ(d_emb − α·d_true)²` (α refit each step). `hyperbolic` switches the distance, its analytic
/// gradient, the Riemannian rescale `((1−‖x‖²)²/4)`, and the ball clamp. The SAME optimiser drives both
/// geometries — only the metric differs, so the race is fair (this is the Nickel–Kiela recipe). O(N²)/iter.
fn embed(g: &Graph, dim: usize, hyperbolic: bool, rng: &mut Rng, iters: usize) -> Vec<Vec<f32>> {
    let dist_fn: &dyn Fn(&[f32], &[f32]) -> f32 = if hyperbolic { &poincare } else { &euclid };
    let mut coords: Vec<Vec<f32>> = (0..g.n)
        .map(|_| (0..dim).map(|_| rng.sym() * 0.05).collect())
        .collect();
    for it in 0..iters {
        let alpha = refit_alpha(g, &coords, dist_fn);
        let lr = 0.5 * (1.0 - 0.9 * it as f32 / iters as f32);
        let mut grad = vec![vec![0.0f32; dim]; g.n];
        for i in 0..g.n {
            for j in 0..g.n {
                if i == j {
                    continue;
                }
                let d = dist_fn(&coords[i], &coords[j]);
                let resid = 2.0 * (d - alpha * g.dist[i][j]);
                let dg = if hyperbolic {
                    dgrad_poincare(&coords[i], &coords[j])
                } else {
                    dgrad_euclid(&coords[i], &coords[j])
                };
                for a in 0..dim {
                    grad[i][a] += resid * dg[a];
                }
            }
        }
        for i in 0..g.n {
            // Riemannian rescale: hyperbolic steps shrink near the boundary (the conformal factor).
            let scale = if hyperbolic {
                let n2: f32 = coords[i].iter().map(|x| x * x).sum();
                let a = 1.0 - n2;
                a * a / 4.0
            } else {
                1.0
            };
            for a in 0..dim {
                coords[i][a] -= lr * scale * grad[i][a] / g.n as f32;
            }
            if hyperbolic {
                let n = coords[i].iter().map(|x| x * x).sum::<f32>().sqrt();
                let cap = 1.0 - 1e-3;
                if n > cap {
                    for c in &mut coords[i] {
                        *c *= cap / n;
                    }
                }
            }
        }
    }
    coords
}

/// The organism's OWN native (Euclidean) concept-map, grown by `metric_step` over the graph's edges
/// (co-activating adjacent nodes) — the SHIPPED substrate arm, for grounding. Reports corr to true dist.
fn native_map_corr(g: &Graph, seed: u64) -> f32 {
    let mut f = Field::new(g.n);
    let mut rng = Rng::new(seed);
    let mut edges = Vec::new();
    for i in 0..g.n {
        for &j in &g.adj[i] {
            if i < j {
                edges.push([i, j]);
            }
        }
    }
    let iters = 15_000usize;
    for it in 0..iters {
        let e = edges[rng.below(edges.len())];
        let rate = 0.2 * (1.0 - it as f32 / iters as f32).max(0.02);
        f.metric_step(&e, rate, 10);
    }
    let pos: Vec<Vec<f32>> = (0..g.n).map(|i| f.position(i)).collect();
    let (mut de, mut dt) = (Vec::new(), Vec::new());
    for i in 0..g.n {
        for j in (i + 1)..g.n {
            de.push(euclid(&pos[i], &pos[j]));
            dt.push(g.dist[i][j]);
        }
    }
    corr(&de, &dt)
}

// --------------------------------------------------------------------------------------
//  Sarkar (2011) closed-form tree→Poincaré-disk construction — isolates the GEOMETRY from
//  the optimizer. Points are placed by exact Möbius isometries (f64), so any residual
//  distortion is the geometry's, not gradient descent's. This is what an ideal (Lorentz /
//  Riemannian) optimizer would approach — the #172 "revisit" machinery.
// --------------------------------------------------------------------------------------

type Cx = (f64, f64);
fn cx_div(a: Cx, b: Cx) -> Cx {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
/// Isometry sending `z0`→0: `(z − z0)/(1 − z̄0·z)`.
fn to_origin(z: Cx, z0: Cx) -> Cx {
    cx_div(
        (z.0 - z0.0, z.1 - z0.1),
        (1.0 - (z0.0 * z.0 + z0.1 * z.1), z0.1 * z.0 - z0.0 * z.1),
    )
}
/// Its inverse, sending 0→`z0`: `(w + z0)/(1 + z̄0·w)`.
fn from_origin(w: Cx, z0: Cx) -> Cx {
    // denominator = 1 + z̄0·w, with z̄0·w = (z0.0·w.0 + z0.1·w.1, z0.0·w.1 − z0.1·w.0).
    let znum = (w.0 + z0.0, w.1 + z0.1);
    let zden = (1.0 + (z0.0 * w.0 + z0.1 * w.1), z0.0 * w.1 - z0.1 * w.0);
    cx_div(znum, zden)
}
fn hdist64(x: Cx, y: Cx) -> f64 {
    let nx = x.0 * x.0 + x.1 * x.1;
    let ny = y.0 * y.0 + y.1 * y.1;
    let dd = (x.0 - y.0).powi(2) + (x.1 - y.1).powi(2);
    let u = 2.0 * dd / ((1.0 - nx) * (1.0 - ny));
    (u + (u * (u + 2.0)).sqrt()).ln_1p()
}
fn hdist32(x: Cx, y: Cx) -> f64 {
    // Same distance evaluated in f32 (cast the coords first) — the substrate's only numeric type.
    let (x, y) = ((x.0 as f32, x.1 as f32), (y.0 as f32, y.1 as f32));
    let nx = x.0 * x.0 + x.1 * x.1;
    let ny = y.0 * y.0 + y.1 * y.1;
    let dd = (x.0 - y.0).powi(2) + (x.1 - y.1).powi(2);
    let u = 2.0 * dd / (((1.0 - nx) * (1.0 - ny)).max(1e-9));
    ((u + (u * (u + 2.0)).sqrt()).ln_1p()) as f64
}

/// Embed the tree `g` (rooted at 0) into the Poincaré disk by Sarkar's construction at edge-scale `s`.
fn sarkar(g: &Graph, s: f64) -> Vec<Cx> {
    // BFS from root 0 gives parents + placement order (for a tree, the BFS tree IS the tree).
    let mut parent = vec![usize::MAX; g.n];
    let mut order = vec![0usize];
    let mut seen = vec![false; g.n];
    seen[0] = true;
    let mut q = std::collections::VecDeque::new();
    q.push_back(0usize);
    while let Some(u) = q.pop_front() {
        for &v in &g.adj[u] {
            if !seen[v] {
                seen[v] = true;
                parent[v] = u;
                order.push(v);
                q.push_back(v);
            }
        }
    }
    let mut pos = vec![(0.0f64, 0.0f64); g.n];
    let r = (s / 2.0).tanh(); // Euclidean radius of a point at hyperbolic distance s from origin
    for &u in &order {
        let children: Vec<usize> = g.adj[u]
            .iter()
            .copied()
            .filter(|&v| parent[v] == u)
            .collect();
        if children.is_empty() {
            continue;
        }
        let zu = pos[u];
        // Angle back to the parent, measured in the frame where u is at the origin.
        let theta_p = if parent[u] == usize::MAX {
            None
        } else {
            let zm = to_origin(pos[parent[u]], zu);
            Some(zm.1.atan2(zm.0))
        };
        let k = children.len();
        for (i, &c) in children.iter().enumerate() {
            let ang = match theta_p {
                None => std::f64::consts::TAU * i as f64 / k as f64, // root: full circle
                Some(tp) => {
                    // Distribute the k children EVENLY within an angular window centred opposite the
                    // parent (a gap of one slot is left for the parent so cones never collide).
                    let win = std::f64::consts::TAU * k as f64 / (k + 1) as f64;
                    tp + std::f64::consts::PI + win * ((i as f64 + 0.5) / k as f64 - 0.5)
                }
            };
            pos[c] = from_origin((r * ang.cos(), r * ang.sin()), zu);
        }
    }
    pos
}

/// Scale-free multiplicative distortion of a hyperbolic embedding (best global scale α refit).
fn distort_h(g: &Graph, pos: &[Cx], dist: &dyn Fn(Cx, Cx) -> f64) -> f32 {
    let (mut de, mut dt) = (Vec::new(), Vec::new());
    for i in 0..g.n {
        for j in (i + 1)..g.n {
            de.push(dist(pos[i], pos[j]) as f32);
            dt.push(g.dist[i][j]);
        }
    }
    let alpha = best_scale(&de, &dt);
    de.iter()
        .zip(&dt)
        .filter(|(d, _)| d.is_finite())
        .map(|(d, t)| ((d / (alpha * t)) - 1.0).abs())
        .sum::<f32>()
        / de.len() as f32
}

/// Retrieval fidelity of a hyperbolic embedding: fraction of nodes whose nearest neighbour is a real edge.
fn retrieval_h(g: &Graph, pos: &[Cx]) -> f32 {
    let mut hits = 0usize;
    for i in 0..g.n {
        let nn = (0..g.n)
            .filter(|&j| j != i)
            .min_by(|&a, &b| hdist64(pos[i], pos[a]).total_cmp(&hdist64(pos[i], pos[b])))
            .unwrap();
        if g.adj[i].contains(&nn) {
            hits += 1;
        }
    }
    hits as f32 / g.n as f32
}
fn max_radius(pos: &[Cx]) -> f64 {
    pos.iter()
        .map(|p| (p.0 * p.0 + p.1 * p.1).sqrt())
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "research: on demand"]
fn hyperbolic_beats_euclidean_for_hierarchy_but_not_flat_data() {
    // ---- numerical-stability sub-check: the #172 finding, reproduced. ------------------------------
    // A CLOSE pair near the boundary is where the naive arcosh(1+u) cancels; the stable form must not.
    let x = [0.55f32, 0.30];
    let y = [0.55f32, 0.3006];
    let refd = poincare_ref64(&x, &y);
    let stable_err = ((poincare(&x, &y) as f64 - refd) / refd).abs();
    let naive_err = ((poincare_naive(&x, &y) as f64 - refd) / refd).abs();
    eprintln!(
        "[NUMERIC] close-pair Poincaré rel-error:  stable {stable_err:.2e}   naive {naive_err:.2e}   (ref d={refd:.3e})"
    );
    assert!(
        stable_err < 1e-4 && stable_err < naive_err * 0.2,
        "stable arcosh1p must be far more accurate than naive for a near pair (stable {stable_err:.2e} vs naive {naive_err:.2e})"
    );

    // A deep hierarchy (127 nodes) and a flat grid (36 nodes) — the two regimes.
    let tree = Graph::bary_tree(2, 6); // 127 nodes, depth 6 — deep enough to crowd 2-D Euclidean
    let grid = Graph::grid(6); // 36 nodes, flat control
    const DIM: usize = 2;
    const ITERS: usize = 900;

    // ---- PART 1: the GROWN / tractable optimizer (Riemannian SGD, f32) — the kind the organism runs. --
    // Same init seed for both geometries on each dataset → identical start, fair race; only the metric differs.
    let te = evaluate(
        &tree,
        &embed(&tree, DIM, false, &mut Rng::new(0x7EE1), ITERS),
        &euclid,
    );
    let th = evaluate(
        &tree,
        &embed(&tree, DIM, true, &mut Rng::new(0x7EE1), ITERS),
        &poincare,
    );
    let ge = evaluate(
        &grid,
        &embed(&grid, DIM, false, &mut Rng::new(0x621D), ITERS),
        &euclid,
    );
    let gh = evaluate(
        &grid,
        &embed(&grid, DIM, true, &mut Rng::new(0x621D), ITERS),
        &poincare,
    );
    let native_tree = native_map_corr(&tree, 0x1111);

    eprintln!("[CURVATURE-1] GROWN optimizer (Euclidean-parametrized Riemannian SGD, f32) — the tractable kind:");
    eprintln!(
        "  TREE (b=2,d=6, 127 nodes):  corr  E {:.3}  H {:.3}   | distortion  E {:.3}  H {:.3}   | retrieval  E {:.0}%  H {:.0}%",
        te.correlation, th.correlation, te.distortion, th.distortion, 100.0 * te.retrieval, 100.0 * th.retrieval
    );
    eprintln!(
        "  GRID (6x6, flat control):   corr  E {:.3}  H {:.3}   | distortion  E {:.3}  H {:.3}   | retrieval  E {:.0}%  H {:.0}%",
        ge.correlation, gh.correlation, ge.distortion, gh.distortion, 100.0 * ge.retrieval, 100.0 * gh.retrieval
    );
    eprintln!(
        "  (grounding) shipped native Euclidean metric_step map corr on the tree: {native_tree:.3}"
    );

    // ---- PART 2: the GEOMETRY itself — Sarkar's closed-form construction (f64), no gradient descent. --
    // As the edge-scale s grows, the tree is pushed toward the boundary and distortion → 0.
    let mut best = None;
    for &s in &[1.0f64, 2.0, 3.0] {
        let pos = sarkar(&tree, s);
        let d = distort_h(&tree, &pos, &hdist64);
        let r = retrieval_h(&tree, &pos);
        eprintln!(
            "[CURVATURE-2] Sarkar construction (f64), tree scale s={s}:  distortion {d:.3}   retrieval {:.0}%   maxRadius {:.6}",
            100.0 * r,
            max_radius(&pos)
        );
        if s == 3.0 {
            best = Some((d, r, pos));
        }
    }
    let (sark_dist, sark_ret, sark_pos) = best.unwrap();

    // ---- PART 3: the f32 BARRIER — the win lives at the boundary, where f32 collapses (#172's reason). --
    // At s=3 the tree is f32-safe; push to s=4 (lower f64 distortion) and f32 loses it.
    let pos4 = sarkar(&tree, 4.0);
    let d4_64 = distort_h(&tree, &pos4, &hdist64);
    let d4_32 = distort_h(&tree, &pos4, &hdist32);
    let d3_32 = distort_h(&tree, &sark_pos, &hdist32);
    eprintln!(
        "[CURVATURE-3] f32 boundary barrier:  s=3 distortion  f64 {sark_dist:.3} / f32 {d3_32:.3} (maxR {:.6})   |   s=4 distortion  f64 {d4_64:.3} / f32 {d4_32:.3} (maxR {:.7})   — f32 sheds the gain as points crowd the boundary; at the f64-optimal s=4, f32 ({d4_32:.3}) is WORSE than the Euclidean baseline ({:.3})",
        max_radius(&sark_pos),
        max_radius(&pos4),
        te.distortion
    );

    eprintln!(
        "[CURVATURE] VERDICT: (1) the GROWN optimizer TIES — hyperbolic ≈ Euclidean on the tree (distortion H {:.3} vs E {:.3}), so the tractable f32 gradient rule does NOT realize a hyperbolic win. (2) The GEOMETRY genuinely wins for hierarchy: Sarkar's closed-form construction reaches distortion {sark_dist:.3} & retrieval {:.0}% vs the Euclidean optimizer's {:.3} & {:.0}% — negative curvature DOES fit a tree far better (f64), confirming the hypothesis at the geometry level, while the flat grid shows NO hyperbolic advantage (control). (3) BUT the win lives at the boundary (maxRadius→1): in f32 it is already mostly gone at s=3 (f64 {sark_dist:.3} → f32 {d3_32:.3}, still < E {:.3}) and at the f64-optimal s=4 f32 ({d4_32:.3}) is WORSE than plain Euclidean — the f32 sweet-spot is narrow and untunable. This independently re-derives #172's DEFERRAL: the advantage is real but needs a proper Riemannian/Lorentz optimizer AND f64 boundary precision — exactly the two documented 'revisit-when' conditions. Composes with the real mass variable: importance-warped curvature would let busy sub-trees claim more hyperbolic volume, but only once the Lorentz+f64 machinery lands.",
        th.distortion, te.distortion, 100.0 * sark_ret, te.distortion, 100.0 * te.retrieval, te.distortion
    );

    // Core claims (all checked against measured numbers before hard-coding):
    // 1. Numerical stability handled above.
    // 2. The GROWN optimizer does NOT realize a hyperbolic win on the tree — it ties Euclidean.
    assert!(
        (th.distortion - te.distortion).abs() < 0.03,
        "the tractable Riemannian-SGD optimizer should TIE (H {:.3} vs E {:.3}) — the honest negative",
        th.distortion, te.distortion
    );
    // 3. The GEOMETRY (Sarkar construction) DOES beat the Euclidean optimizer for the hierarchy.
    assert!(
        sark_dist < te.distortion - 0.05 && sark_ret > te.retrieval + 0.3,
        "hyperbolic geometry must embed the TREE far better via construction (dist {sark_dist:.3} vs E {:.3}, retrieval {:.0}% vs E {:.0}%)",
        te.distortion, 100.0 * sark_ret, 100.0 * te.retrieval
    );
    // 4. Flat control: on the grid the grown optimizer gives hyperbolic NO advantage.
    assert!(
        gh.correlation <= ge.correlation + 0.03 && (gh.distortion - ge.distortion).abs() < 0.03,
        "on FLAT data hyperbolic must not beat Euclidean (corr H {:.3} vs E {:.3})",
        gh.correlation,
        ge.correlation
    );
    // 5. The f32 barrier: at the f64-optimal scale the hyperbolic win is LOST in f32 — WORSE than plain
    //    Euclidean — because points crowd the boundary (maxR→1). This IS #172's deferral reason, measured.
    assert!(
        d4_32 > d4_64 + 0.05 && d4_32 > te.distortion && max_radius(&pos4) > 0.9999,
        "the boundary win must collapse in f32 to below the Euclidean baseline (f32 {d4_32:.3} vs f64 {d4_64:.3} vs E {:.3}, maxR {:.6})",
        te.distortion,
        max_radius(&pos4)
    );
}

// ======================================================================================
//  (B) TIME DILATION — warp event-time by locally-measured activity (a mass proxy).
// ======================================================================================

/// A 1-D world where a particle moves at CONSTANT proper speed, but events are emitted at a rate
/// proportional to the local "mass" m(x). So per-event displacement ∝ 1/m: busy (massive) regions are
/// finely sampled (small steps), sparse regions coarsely (big steps). Returns the observed event
/// positions AND the true mass field over bins (the mass field is used ONLY to build the world and for
/// an oracle bound — the predictor never sees it).
struct TimeWorld {
    xs: Vec<f32>,   // observed event positions
    mass: Vec<f32>, // true m per bin (hidden from the predictor)
    bins: usize,
    length: f32,
}

impl TimeWorld {
    fn bin(&self, x: f32) -> usize {
        ((x / self.length * self.bins as f32) as usize).min(self.bins - 1)
    }

    /// Build a world. `amp = 0` → FLAT mass (the control): uniform emission, constant steps.
    fn build(bins: usize, amp: f32, n_events: usize) -> Self {
        let length = 1.0f32;
        // Smoothly-varying mass: m(x) = 1 + amp·(0.5+0.5 sin(2π·2x)). amp=0 ⇒ flat.
        let mass: Vec<f32> = (0..bins)
            .map(|i| {
                let x = (i as f32 + 0.5) / bins as f32;
                1.0 + amp * (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 2.0 * x).sin())
            })
            .collect();
        let m_at = |x: f32| {
            let b = ((x / length * bins as f32) as usize).min(bins - 1);
            mass[b]
        };
        let base_speed = 0.02f32; // proper-time spatial speed
        let mut x = 0.13f32;
        let mut dir = 1.0f32;
        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            xs.push(x);
            let step = base_speed / m_at(x); // dilation: dense events where mass is high
            x += dir * step;
            if x >= length {
                x = 2.0 * length - x;
                dir = -dir;
            } else if x <= 0.0 {
                x = -x;
                dir = -dir;
            }
        }
        TimeWorld {
            xs,
            mass,
            bins,
            length,
        }
    }
}

/// Accumulate the ACTIVITY field = per-bin event count over `xs[..split]` — the mass PROXY the organism
/// measures from its own event stream (co-activation / reliance frequency), never the hidden m.
fn activity_field(w: &TimeWorld, split: usize) -> Vec<f32> {
    let mut act = vec![1e-3f32; w.bins]; // tiny floor to avoid div-by-zero
    for &x in &w.xs[..split] {
        act[w.bin(x)] += 1.0;
    }
    act
}

/// Mean-absolute next-step prediction error for a predictor `pred(prev, cur) -> next_hat` over held-out
/// triples. Triples that cross a wall reflection are kept (both models fail them equally — a fair floor).
fn predict_mae(w: &TimeWorld, split: usize, pred: &dyn Fn(f32, f32) -> f32) -> f32 {
    let (mut err, mut cnt) = (0.0f32, 0.0f32);
    for t in (split + 1)..(w.xs.len() - 1) {
        let (prev, cur, next) = (w.xs[t - 1], w.xs[t], w.xs[t + 1]);
        err += (pred(prev, cur) - next).abs();
        cnt += 1.0;
    }
    err / cnt.max(1.0)
}

#[test]
#[ignore = "research: on demand"]
fn warping_event_time_by_activity_helps_only_in_a_massive_world() {
    let bins = 40usize;
    let n = 6000usize;
    let split = 4000usize; // train the activity field on the first 4000 events, test on the rest

    for &(amp, label) in &[
        (3.0f32, "MASSIVE (mass-modulated sampling)"),
        (0.0f32, "FLAT (uniform, control)"),
    ] {
        let w = TimeWorld::build(bins, amp, n);
        let act = activity_field(&w, split);

        // Baselines & models (all use ONLY prev, cur, and the learned activity field — no peeking).
        let persistence = |_p: f32, c: f32| c; // chance anchor: "it won't move"
        let uniform = |p: f32, c: f32| c + (c - p); // substrate_trajectory momentum in event-time
                                                    // DILATED: displacement ∝ 1/activity, so rescale the last step by act(prev)/act(cur).
        let dilated = |p: f32, c: f32| {
            let bp = ((p / w.length * bins as f32) as usize).min(bins - 1);
            let bc = ((c / w.length * bins as f32) as usize).min(bins - 1);
            c + (c - p) * (act[bp] / act[bc])
        };
        // ORACLE: same warp but with the TRUE hidden mass ratio — upper bound on what the proxy can reach.
        let oracle = |p: f32, c: f32| {
            let bp = ((p / w.length * bins as f32) as usize).min(bins - 1);
            let bc = ((c / w.length * bins as f32) as usize).min(bins - 1);
            c + (c - p) * (w.mass[bp] / w.mass[bc])
        };

        let mae_pers = predict_mae(&w, split, &persistence);
        let mae_unif = predict_mae(&w, split, &uniform);
        let mae_dila = predict_mae(&w, split, &dilated);
        let mae_orac = predict_mae(&w, split, &oracle);

        eprintln!("[TIME] world = {label}");
        eprintln!(
            "  next-step MAE:  persistence(chance) {mae_pers:.5}   uniform-velocity {mae_unif:.5}   activity-DILATED {mae_dila:.5}   oracle(true mass) {mae_orac:.5}"
        );
        let improve = 100.0 * (mae_unif - mae_dila) / mae_unif;
        eprintln!("  dilation vs uniform: {improve:+.1}% MAE change");

        if amp > 0.0 {
            // In the MASSIVE world the measured-activity warp must beat uniform event-time momentum,
            // and both must beat the persistence chance anchor.
            assert!(
                mae_unif < mae_pers,
                "momentum must beat persistence (unif {mae_unif:.5} vs pers {mae_pers:.5})"
            );
            assert!(
                mae_dila < mae_unif * 0.9,
                "activity-dilated event-time must cut next-step MAE >=10% vs uniform (dila {mae_dila:.5} vs unif {mae_unif:.5})"
            );
            // The proxy should approach the oracle (measured event-density ≈ hidden mass).
            assert!(
                mae_dila < mae_unif && mae_orac <= mae_dila + 1e-4,
                "the measured-activity proxy should approach the true-mass oracle (dila {mae_dila:.5}, oracle {mae_orac:.5})"
            );
        } else {
            // FLAT control: warping must NOT meaningfully help (activity is ~uniform, ratio ~1).
            assert!(
                mae_dila >= mae_unif * 0.98,
                "on FLAT mass the warp must not help (dila {mae_dila:.5} vs unif {mae_unif:.5}) — else the effect is an artefact"
            );
        }
    }

    eprintln!("[TIME] VERDICT: warping event-time by a LOCALLY-MEASURED activity field (the co-activation / reliance frequency, a mass proxy) improves next-state prediction in a world where mass modulates event density, and does NOT help when mass is flat (the falsification control). Proper time ≠ coordinate time: `substrate_trajectory` extrapolates constant velocity per EVENT; here the velocity is constant per PROPER time and the organism recovers the warp from its own event statistics — no hidden mass. Composes with the real mass variable: local mass IS the dilation factor.");
}
