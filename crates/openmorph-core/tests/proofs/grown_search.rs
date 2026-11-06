// SPDX-License-Identifier: MIT-0
//! CONSTRAINT SATISFACTION AS SUBSTRATE RELAXATION DYNAMICS — the nature-based solver (openmorph's half of
//! MorphBot's #1 ARC-3 lever, the `algorithmic-reasoning` class: SAT / colouring / cover / tiling, where
//! terminal-only reward sits at the end of a long solution SEQUENCE over a large combinatorial space and no
//! reward-shaping / curriculum / options-depth unlocks it).
//!
//! THE PIVOT (Sébastien): do NOT wrap an authored tree-search/GF(2)/CSP solver. Physical and biological
//! systems solve combinatorial constraints by RELAXING to a low-energy attractor, not by tree search —
//! Ising/Hopfield energy minimisation, mean-field/belief propagation on a factor graph, simulated annealing
//! whose temperature cools as the system orders. Every one of those IS openmorph's physics: RELAX +
//! THRESHOLD-JUMP + SELECT, driven by surprise + energy, settling the async fire-on-change cascade to a
//! fixed point. So we make constraint-satisfaction the SUBSTRATE'S OWN relaxation dynamics.
//!
//! THE MECHANISM (graph K-colouring as a Potts model; K=3):
//!   1. COUPLING = native substrate units. Each vertex holds K spin cells (soft one-hot colour). For every
//!      vertex/colour a LINEAR field unit sums the neighbour spin-mass on that colour (h[v][k] = Σ_{u~v}
//!      s[u][k]) — the local conflict field. For every vertex/colour an ENERGY unit uses the 2nd-order
//!      `factor` product s[v][k]·h[v][k] — the substrate's OWN violation-energy. The async settle IS the
//!      local message-passing that computes both (couplings → fields → energy), each unit firing on change.
//!   2. SETTLE = evaluate. `stimulate` fires the cascade to a fixed point; reading the energy cells gives the
//!      organism's violation-energy E, reading the field cells gives the per-colour conflict pressure.
//!   3. ENDOGENOUS ANNEALING. A Gibbs heat-bath resamples each vertex's colour ∝ exp(−h[v][k]/T) — the
//!      stochastic THRESHOLD-JUMP / SELECT at temperature T. T is the organism's OWN violation-energy
//!      (T = T_MIN + T_SCALE·E/E_max): high violation ⇒ HOT ⇒ explore, violation falls ⇒ COLD ⇒ commit. It
//!      SELF-ANNEALS with no hand-set schedule (surprise = energy = temperature).
//!   4. RESTART via the economy. A configuration that stops improving is starved and RESEEDED (a grown random
//!      restart) — local relaxation has no completeness guarantee, so a stuck basin is abandoned.
//!   5. SELECT the assignment when the violation-energy = 0 (the fixed point IS the solution).
//!
//! ARMS (INSIDE-scored, deterministic seeded PRNG, chance-anchored, SAME coupling organism):
//!   BLIND     (i.i.d. random complete assignments) — terminal-only reward over a combinatorial space defeats
//!             reaction: a valid colouring is found by chance ~3% of the budget.
//!   REACTIVE  (greedy settle, T→0, NO jump/restart) — min-conflicts descent STICKS in a local minimum at
//!             partial violation ⇒ fails ~half the frustrated instances.
//!   ENDOGENOUS(the settle with the grown violation-energy temperature, NO restart) — the stochastic
//!             threshold-jump at T = the organism's own violation-energy escapes the traps greedy sticks in.
//!   ABLATION-T(a FIXED temperature — annealing frozen) — a HOT fixed T never commits (~0), a hand-tuned COLD
//!             fixed T ties the endogenous one (the honest boundary): the endogenous temperature's edge is
//!             being PARAMETER-FREE (self-adapting to violation), not beating a tuned constant here.
//!   FULL      (endogenous annealing + economy restart) — the restart closes the residual ⇒ 100% on
//!             satisfiable instances.
//!
//! AUTHORING LEDGER. GIVEN = the constraint structure (the edge set) + the goal predicate (zero violation).
//! SUBSTRATE (fixed laws) = the native coupling units (`factor`/linear), the async settle that computes the
//! fields + energy, the stochastic SELECT/threshold-jump. GROWN / self-organising = the annealing temperature
//! (the organism's own violation-energy, self-cooling) and the economy restart. NOT AUTHORED = there is no
//! DFS, no backtracking stack, no GF(2) elimination, no CSP propagation — the solve is the SETTLE. The
//! relaxation LOOP is a usage-protocol over the substrate (like the reasoner/discovery loops); the point
//! proven is that the substrate's own relaxation dynamics make the combinatorial search TRACTABLE. NO core
//! change (pure composition of shipped primitives — the async settle + `factor` + the stochastic SELECT).
//!
//!   cargo test -p openmorph-core --test proofs grown_search -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use openmorph_core::Activation;

use crate::common::Rng;

const V: usize = 12; // vertices
const K: usize = 3; // colours

// ---------------------------------------------------------------------------------------------------------
// The task: random 3-colourable graphs (rejection-sampled), dense enough that greedy relaxation gets stuck.
// ---------------------------------------------------------------------------------------------------------

struct Graph {
    adj: Vec<Vec<usize>>,
    edges: Vec<(usize, usize)>,
}

/// An exact 3-colourability oracle (backtracking) — TEST SCAFFOLDING for instance generation / the "is
/// satisfiable" ground truth, NOT an arm. The arms never call it.
fn three_colourable(g: &Graph) -> bool {
    fn go(g: &Graph, col: &mut [i8], v: usize) -> bool {
        if v == V {
            return true;
        }
        for c in 0..K as i8 {
            if g.adj[v].iter().all(|&u| col[u] != c) {
                col[v] = c;
                if go(g, col, v + 1) {
                    return true;
                }
                col[v] = -1;
            }
        }
        false
    }
    go(g, &mut [-1i8; V], 0)
}

/// A random graph with `m` edges, kept only if 3-colourable (hard-satisfiable near the colouring threshold ⇒
/// greedy relaxation frequently traps, so annealing is the lever). Bounded: at a density where 3-colourable
/// graphs are vanishingly rare, PANIC fast rather than spin forever (a mis-set `M` fails CI, never hangs it).
fn random_sat_graph(rng: &mut Rng, m: usize) -> Graph {
    for _ in 0..20_000 {
        let mut adj = vec![Vec::new(); V];
        let mut edges = Vec::new();
        let mut tries = 0;
        while edges.len() < m && tries < 1000 {
            tries += 1;
            let a = rng.below(V);
            let b = rng.below(V);
            if a != b && !adj[a].contains(&b) {
                adj[a].push(b);
                adj[b].push(a);
                edges.push((a.min(b), a.max(b)));
            }
        }
        let g = Graph { adj, edges };
        if three_colourable(&g) {
            return g;
        }
    }
    panic!("no 3-colourable graph with {m} edges found in 20000 tries — density set too high");
}

// ---------------------------------------------------------------------------------------------------------
// The coupling organism — native substrate units + the async settle compute the fields and violation-energy.
// ---------------------------------------------------------------------------------------------------------

struct Coupler {
    f: Field,
    h_cell: Vec<usize>, // h_cell[v*K+k] = local conflict field Σ_{u~v} s[u][k]
    e_cell: Vec<usize>, // e_cell[v*K+k] = s[v][k]·h[v][k] (the 2nd-order factor product = violation energy)
    budget: usize,
    edges: Vec<(usize, usize)>,
}

fn spin(v: usize, k: usize) -> usize {
    v * K + k
}

impl Coupler {
    /// Build the coupling network for a graph: the coupling TOPOLOGY is the given edge set, instantiated as
    /// native substrate units (linear field sums + `factor` products), settled by the async cascade.
    fn build(g: &Graph) -> Coupler {
        let mut f = Field::new(V * K);
        // Identity genome so a field unit is the EXACT neighbour-mass sum and a factor unit the EXACT product
        // (no tanh squashing) — `gain_fanin_norm` stays false (default) so gain = 1.
        f.config_mut().activation = Activation::Identity;
        let mut h_cell = vec![0usize; V * K];
        let mut e_cell = vec![0usize; V * K];
        for v in 0..V {
            for k in 0..K {
                // FIELD unit: h[v][k] = Σ_{u~v} s[u][k] (the local conflict pressure of colour k at v).
                let reads: Vec<usize> = g.adj[v].iter().map(|&u| spin(u, k)).collect();
                let hc = f.len();
                let n = reads.len();
                f.add_unit(reads, vec![1.0; n], 0.0, hc);
                h_cell[v * K + k] = hc;
            }
        }
        for v in 0..V {
            for k in 0..K {
                // ENERGY unit: e[v][k] = s[v][k] · h[v][k] via the 2nd-order `factor` (the substrate's own
                // violation-energy; Σ e = 2·conflict-edges, so the organism SENSES its own frustration).
                let hc = h_cell[v * K + k];
                let ec = f.len();
                f.add_unit(vec![hc], vec![1.0], 0.0, ec);
                f.set_factor(ec, Some(spin(v, k)));
                e_cell[v * K + k] = ec;
            }
        }
        // The coupling graph is a shallow DAG (spin → field → energy, 2 hops), so a few passes fully
        // settle it; 4×len is generous headroom and leaves the converged fields/energy byte-identical to a
        // larger budget (it only stops EARLY if under-settling, which this is not). Keeps the proof CI-fast.
        let budget = 4 * f.len();
        Coupler {
            f,
            h_cell,
            e_cell,
            budget,
            edges: g.edges.clone(),
        }
    }

    /// Set the spins to the one-hot of a discrete colouring and settle the cascade — the substrate computes
    /// every field and energy cell. Returns nothing; read via `field`/`energy`.
    fn settle(&mut self, colour: &[usize]) {
        let mut inputs: Vec<(usize, f32)> = Vec::with_capacity(V * K);
        for (v, &c) in colour.iter().enumerate() {
            for k in 0..K {
                inputs.push((spin(v, k), if k == c { 1.0 } else { 0.0 }));
            }
        }
        self.f.stimulate(&inputs, self.budget);
    }

    fn field(&self, v: usize, k: usize) -> f32 {
        self.f.cells()[self.h_cell[v * K + k]]
    }

    /// The organism's OWN settled violation-energy = ½ Σ_{v,k} e[v][k] = # conflict edges under `colour`.
    fn energy(&self) -> f32 {
        0.5 * self.e_cell.iter().map(|&c| self.f.cells()[c]).sum::<f32>()
    }

    /// Discrete conflict count (the goal predicate: zero = a valid colouring).
    fn conflicts(&self, colour: &[usize]) -> usize {
        self.edges
            .iter()
            .filter(|&&(a, b)| colour[a] == colour[b])
            .count()
    }
}

// ---------------------------------------------------------------------------------------------------------
// The relaxation dynamics (usage-protocol over the settle).
// ---------------------------------------------------------------------------------------------------------

/// Deterministic greedy commit: the colour of least conflict field (argmin h) — the T→0 limit, no jump.
fn argmin_field(c: &Coupler, v: usize) -> usize {
    (0..K)
        .min_by(|&a, &b| c.field(v, a).total_cmp(&c.field(v, b)))
        .unwrap_or(0)
}

/// Heat-bath resample of vertex `v`: colour ∝ exp(−h[v][k]/T). At T→0 this is argmin-conflict (greedy
/// commit); at high T it is uniform (explore) — the stochastic SELECT / THRESHOLD-JUMP at temperature T.
fn heatbath(c: &Coupler, v: usize, t: f32, rng: &mut Rng) -> usize {
    let t = t.max(1e-3);
    let mut p = [0f32; K];
    let mut sum = 0.0;
    // subtract the min field for numerical stability
    let mn = (0..K).map(|k| c.field(v, k)).fold(f32::MAX, f32::min);
    for (k, pk) in p.iter_mut().enumerate() {
        *pk = (-(c.field(v, k) - mn) / t).exp();
        sum += *pk;
    }
    let r = rng.f() * sum;
    let mut acc = 0.0;
    for (k, &pk) in p.iter().enumerate() {
        acc += pk;
        if r <= acc {
            return k;
        }
    }
    K - 1
}

/// Endogenous temperature = the organism's OWN violation-energy (self-annealing): hot while frustrated,
/// cold as the conflicts vanish. `frozen` pins T to a fixed value (the ablation — annealing removed).
fn temperature(conflicts: usize, e_max: usize, frozen: Option<f32>) -> f32 {
    const T_MIN: f32 = 0.05;
    const T_SCALE: f32 = 1.5;
    match frozen {
        Some(t) => t,
        None => T_MIN + T_SCALE * conflicts as f32 / e_max.max(1) as f32,
    }
}

struct Outcome {
    solved: bool,
    sweeps: usize, // configurations evaluated (settles) to the solution
}

/// The full nature-based solver: sequential Gibbs relaxation at the endogenous temperature, with an economy
/// restart when a basin stops improving. `anneal=false` freezes T (ablation); `greedy=true` is the reactive
/// single-descent (T→0, no restart).
fn relax(
    c: &mut Coupler,
    rng: &mut Rng,
    max_sweeps: usize,
    frozen_t: Option<f32>,
    patience: usize,
    reactive: bool,
) -> Outcome {
    let e_max = c.edges.len();
    let mut colour: Vec<usize> = (0..V).map(|_| rng.below(K)).collect();
    let mut best = usize::MAX;
    let mut stall = 0;
    let mut settles = 0usize;
    for sweep in 0..max_sweeps {
        c.settle(&colour);
        settles += 1;
        let conf = c.conflicts(&colour);
        // sanity: the substrate's settled energy equals the discrete conflict count.
        debug_assert!((c.energy() - conf as f32).abs() < 1e-2);
        if conf == 0 {
            return Outcome {
                solved: true,
                sweeps: settles,
            };
        }
        if conf < best {
            best = conf;
            stall = 0;
        } else {
            stall += 1;
        }
        // RESTART via the economy: a stuck basin is abandoned and reseeded (grown random restart).
        if !reactive && stall > patience {
            colour = (0..V).map(|_| rng.below(K)).collect();
            best = usize::MAX;
            stall = 0;
            continue;
        }
        let t = temperature(conf, e_max, frozen_t);
        // one sweep: resample every vertex (sequential — settle after each so the field is current). REACTIVE
        // is deterministic argmin-conflict descent (no annealing, no jump); it halts at the first fixed point.
        let mut changed = false;
        for v in 0..V {
            let nc = if reactive {
                argmin_field(c, v)
            } else {
                heatbath(c, v, t, rng)
            };
            if nc != colour[v] {
                colour[v] = nc;
                changed = true;
                c.settle(&colour); // incremental re-settle keeps the local fields current for the next vertex
            }
        }
        // REACTIVE: a full sweep with no change is a local minimum — greedy descent is TRAPPED (partial
        // violation), no way out without the stochastic jump.
        if reactive && !changed {
            return Outcome {
                solved: false,
                sweeps: settles,
            };
        }
        let _ = sweep;
    }
    Outcome {
        solved: false,
        sweeps: settles,
    }
}

/// BLIND baseline: i.i.d. random complete assignments until zero violation (or the cap). Returns
/// (solved, configurations_evaluated).
fn blind(c: &mut Coupler, rng: &mut Rng, cap: usize) -> (bool, usize) {
    for i in 1..=cap {
        let colour: Vec<usize> = (0..V).map(|_| rng.below(K)).collect();
        if c.conflicts(&colour) == 0 {
            return (true, i);
        }
    }
    (false, cap)
}

// ---------------------------------------------------------------------------------------------------------
// The proof.
// ---------------------------------------------------------------------------------------------------------

/// Run a relax-family arm over the instance bank; returns (solve_rate, mean_settles_over_solved).
fn arm(
    instances: &[Graph],
    max_sweeps: usize,
    patience: usize,
    frozen_t: Option<f32>,
    reactive: bool,
) -> (f64, f64) {
    let mut solved = 0usize;
    let mut settles = 0usize;
    for (i, g) in instances.iter().enumerate() {
        let mut c = Coupler::build(g);
        let mut rng = Rng::new(0x51E5 + i as u64 * 2654435761);
        let o = relax(&mut c, &mut rng, max_sweeps, frozen_t, patience, reactive);
        if o.solved {
            solved += 1;
            settles += o.sweeps;
        }
    }
    let n = instances.len() as f64;
    (
        solved as f64 / n,
        if solved > 0 {
            settles as f64 / solved as f64
        } else {
            0.0
        },
    )
}

#[test]
fn constraint_satisfaction_is_substrate_relaxation() {
    let mut gen = Rng::new(0xC0FFEE);
    const M: usize = 30; // edges — near the 3-colouring threshold for V=12 (avg degree 5)
    const N_INST: usize = 16;
    const MAX_SWEEPS: usize = 120;
    const PATIENCE: usize = 6;
    const BLIND_CAP: usize = 3000;

    let instances: Vec<Graph> = (0..N_INST).map(|_| random_sat_graph(&mut gen, M)).collect();

    const NO_RESTART: usize = 100_000; // patience so large the economy restart never triggers (isolation)

    // REACTIVE — greedy deterministic descent (min-conflicts), no annealing, no jump, no restart.
    let (react_rate, _) = arm(&instances, MAX_SWEEPS, PATIENCE, None, true);
    // --- ISOLATE the annealing: NO restart, so the temperature schedule is the ONLY escape from a trap. ---
    // ENDOGENOUS (violation-driven T) vs a PINNED T at both ends. A fixed temperature is brittle: COLD (0.1)
    // traps like greedy, HOT (2.5) never commits; only the endogenous schedule escapes AND commits.
    let (anneal_rate, anneal_mean) = arm(&instances, MAX_SWEEPS, NO_RESTART, None, false);
    let (cold_rate, cold_mean) = arm(&instances, MAX_SWEEPS, NO_RESTART, Some(0.1), false);
    let (hot_rate, hot_mean) = arm(&instances, MAX_SWEEPS, NO_RESTART, Some(2.5), false);
    // --- FULL solver: the endogenous annealing PLUS the economy restart (the additive nature-based layer). ---
    let (relax_rate, relax_mean) = arm(&instances, MAX_SWEEPS, PATIENCE, None, false);

    // BLIND — i.i.d. random complete assignments.
    let mut blind_solved = 0usize;
    let mut blind_cfgs = 0usize;
    for (i, g) in instances.iter().enumerate() {
        let mut c = Coupler::build(g);
        let mut rng = Rng::new(0x51E5 + i as u64 * 2654435761);
        let (ok, cfgs) = blind(&mut c, &mut rng, BLIND_CAP);
        if ok {
            blind_solved += 1;
            blind_cfgs += cfgs;
        }
    }
    let blind_rate = blind_solved as f64 / instances.len() as f64;
    let blind_mean = if blind_solved > 0 {
        blind_cfgs as f64 / blind_solved as f64
    } else {
        0.0
    };

    eprintln!("[RELAX-SOLVER] random 3-colourable graphs, V={V} K={K} edges={M}, {N_INST} satisfiable instances");
    eprintln!("  BLIND      (i.i.d. random assignments)          : solve {blind_rate:.2}  mean configs {blind_mean:.0}   (terminal-only reward over a combinatorial space defeats reaction)");
    eprintln!("  REACTIVE   (greedy settle, no jump/restart)     : solve {react_rate:.2}  (min-conflicts descent TRAPS in local minima)");
    eprintln!("  -- annealing isolated (NO restart — temperature is the only escape) --");
    eprintln!("  ABLATION hot-T=2.5                              : solve {hot_rate:.2}  mean settles {hot_mean:.0}   (a mis-set fixed T never commits)");
    eprintln!("  ABLATION cold-T=0.1  (hand-tuned)               : solve {cold_rate:.2}  mean settles {cold_mean:.0}   (a TUNED fixed T ties endogenous — the honest boundary)");
    eprintln!("  ENDOGENOUS anneal (T = own violation-energy)    : solve {anneal_rate:.2}  mean settles {anneal_mean:.0}   (PARAMETER-FREE: escapes AND commits, no tuned schedule)");
    eprintln!("  -- + economy restart (the additive nature-based layer) --");
    eprintln!("  FULL       endogenous anneal + restart          : solve {relax_rate:.2}  mean settles-to-solution {relax_mean:.0}");
    eprintln!("  VERDICT: constraint satisfaction is the SETTLE — native `factor`/linear couplings relax to a zero-violation fixed point; the organism's OWN violation-energy is the self-annealing temperature (hot⇒explore, cold⇒commit); the economy reseeds a stuck basin. No DFS, no GF(2), no CSP propagation. The stochastic threshold-jump is load-bearing (greedy T→0 traps); a mis-set fixed T fails (hot never commits); the endogenous temperature lands in the viable range with NO tuned constant. HONEST BOUNDARY: a hand-tuned low fixed T ties it here — the endogenous edge is parameter-freeness; deeper-minima regimes (larger/denser, or frustrated max-cut) are where a fixed low T should break and self-annealing separate.");

    // --- asserts (truth-not-trophy) ---
    assert!(
        blind_rate < 0.3,
        "the space must be genuinely combinatorial — blind random assignment near-fails ({blind_rate:.2})"
    );
    assert!(
        react_rate < 0.8,
        "reactive greedy relaxation must TRAP in local minima on frustrated instances ({react_rate:.2})"
    );
    // The stochastic threshold-jump at the endogenous temperature escapes the traps greedy (T→0) sticks in.
    assert!(
        anneal_rate > react_rate + 0.2,
        "endogenous annealing (no restart) must escape traps greedy cannot ({anneal_rate:.2} vs {react_rate:.2})"
    );
    // The temperature is a real control that must land in a viable range: a mis-set fixed (HOT) temperature
    // fails to commit. The endogenous schedule reaches the viable range from the organism's own violation —
    // no tuned constant. (A hand-tuned COLD fixed T ties it here; that tie is reported, not asserted away.)
    assert!(
        hot_rate < 0.3 && anneal_rate > hot_rate + 0.5,
        "a mis-set fixed temperature must fail where the endogenous one succeeds (hot {hot_rate:.2} vs anneal {anneal_rate:.2})"
    );
    // FULL: the economy restart closes the residual — the substrate settles satisfiable instances to zero.
    assert!(
        relax_rate >= 0.95,
        "endogenous annealing + economy restart must settle satisfiable instances to zero violation ({relax_rate:.2})"
    );
}
