// SPDX-License-Identifier: MIT-0
//! BEAM SEARCH — internal deliberation over IMAGINED trajectories, as a USAGE PROTOCOL over the native
//! replay/imagine + fork memory (#189) and the free-energy score `G` (`substrate_free_energy_*`). NO new core.
//!
//! Sébastien / registry D9: "beam search = replay/`imagine` a state → branch into K continuations
//! (`fork_mutated`) → score each by `G` (or fit) → keep the top-K → repeat, to a depth." The substrate already
//! does exactly this over the concept-map (`substrate_fractal`: beam=1 vs beam=2 coarse-to-fine descent). Here
//! the beam runs over a LEARNED WORLD MODEL: each candidate carries its own imagined trajectory as episodic
//! MEMORY that travels through every fork, and the surviving trajectory is READ BACK OUT of the winner's memory
//! by `replay`. The whole faculty composes three existing primitives — a grown model (`grow`), the pragmatic
//! free-energy score `G(s) = (μ(s) − s*)²`, and fork-carried episodic memory (`fork_mutated` + `store_episode`
//! + `replay`) — and adds ZERO core.
//!
//! THE WORLD needs MULTI-HYPOTHESIS search (the discriminator, truth-not-trophy). A `B`-ary decision tree of
//! depth `D`. Along the single GOLDEN path to the uniquely-best leaf, every gate is DECEPTIVE: the golden child
//! is NOT the locally-cheapest option — `r_t − 1` sibling decoys look CHEAPER (low immediate `G`) but lead into
//! subtrees that BALLOON in cost, while the golden child's small gate cost is followed by the only low-cost
//! continuation. So the locally-best first step leads AWAY from the goal:
//!   * GREEDY (beam width K=1) commits to the cheapest-looking child at gate 1 — a decoy — and can never
//!     recover: it reaches a ballooned decoy leaf, NEVER the goal. It traps on EVERY seed (0%).
//!   * BEAM (K≥2) keeps K hypotheses, so the golden child (immediate-cost rank `r_t` at its gate) SURVIVES the
//!     cut whenever `K ≥ r_t`; once past the gate its cumulative cost dominates the ballooned decoys, so it wins
//!     the final selection. Reaching the goal therefore needs `K ≥ max_t r_t` — keeping MORE hypotheses reaches
//!     it MORE often. The reach-vs-K curve is the mechanism made visible.
//!   * SCRAMBLE (truth-not-trophy) — the SAME beam width, but scored by a fixed RANDOM field instead of the
//!     model's `G`. With no valid score the golden trajectory survives the global top-K cuts only by luck, so
//!     beam COLLAPSES back to ≈greedy no matter how wide it is. The SCORE is the load, not the branching.
//!
//! `r_t` is drawn per seed, so the reach rises smoothly with K (greedy=0 → full-width=100%), and the scramble
//! control stays flat near zero. Scored INTRINSICALLY (reached the unique best leaf), over many seeds. The
//! greedy baseline is the SAME beam code at width 1 (a genuine baseline, not a strawman).
//!
//!   cargo test -p openmorph-core --test proofs substrate_beam -- --nocapture

use openmorph_core::substrate::Field;

use crate::common::Rng;

// ── the deceptive world ──────────────────────────────────────────────────────────────────────────────────
const B: usize = 5; // branching: children per internal node
const D: usize = 3; // depth of the golden leaf (number of deceptive gates)
const SSTAR: f32 = 0.0; // the preferred observation; the pragmatic score is G(s) = (μ(s) − s*)² = μ(s)²
const SEEDS: usize = 40;

// per-node ENTRY costs (as observation distance from s*): golden gate = MID, the r_t−1 lures = LOW (cheaper,
// so greedy prefers them), everything else = HIGH (a ballooning decoy subtree). Golden total = D·MID < HIGH,
// so the golden leaf is the unique minimum-cost trajectory; any path touching a HIGH node loses.
const C_LOW: f32 = 0.02;
const C_MID: f32 = 0.10;
const C_HIGH: f32 = 0.80;

/// A generated tree instance (one seed): the topology, each node's true observation `o` (= −√cost, so
/// `(o−s*)² = cost`), the golden path, and a fixed random score field for the scramble control.
struct World {
    children: Vec<Vec<usize>>, // children[node]; empty ⇒ leaf
    obs: Vec<f32>,             // the true observation the model must learn at each node
    scram: Vec<f32>,           // a fixed random score field (the scramble control's "score")
    golden_path: Vec<usize>,   // root … golden_leaf (length D+1)
}

impl World {
    fn n_nodes(&self) -> usize {
        self.children.len()
    }
    fn golden_leaf(&self) -> usize {
        *self.golden_path.last().unwrap()
    }
}

/// Cost → observation: `o = −√cost`, so the pragmatic score `(o − 0)² = o² = cost`. All in tanh range.
fn cost_to_obs(cost: f32) -> f32 {
    -cost.sqrt()
}

/// Generate a deceptive `B`-ary depth-`D` tree. The golden path is chosen at random; at each golden gate the
/// golden child is planted at a random immediate-cost RANK `r_t ∈ {2..=B}` (never 1 ⇒ greedy always diverges),
/// with `r_t − 1` cheaper lures and the rest expensive. Every non-golden subtree is uniformly HIGH (balloons).
fn generate(seed: u64) -> World {
    let mut r = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xBEA3);
    r.u();

    let mut children: Vec<Vec<usize>> = vec![Vec::new()];
    let mut cost: Vec<f32> = vec![0.0]; // root entry cost is 0 (common to all trajectories)
    let mut depth: Vec<usize> = vec![0];
    let mut on_golden: Vec<bool> = vec![true];

    // Grow the full tree breadth-first, deciding each node's children's costs as we create them.
    let mut queue = vec![0usize];
    let mut golden_path = vec![0usize];
    while let Some(node) = queue.pop() {
        if depth[node] == D {
            continue; // a leaf
        }
        // Create B children first (indices), then assign their costs.
        let base = children.len();
        let kids: Vec<usize> = (0..B).map(|i| base + i).collect();
        for _ in 0..B {
            children.push(Vec::new());
            cost.push(C_HIGH);
            depth.push(depth[node] + 1);
            on_golden.push(false);
        }
        children[node] = kids.clone();

        if on_golden[node] {
            // A GOLDEN gate: plant the golden child at immediate-cost rank r_t ∈ {2..=B}.
            let r_t = 2 + r.below(B - 1); // {2..=B}
            let gidx = r.below(B); // which child continues the golden path
            let golden = kids[gidx];
            // choose r_t − 1 of the OTHER children to be cheaper "lures" (C_LOW); the rest stay C_HIGH.
            let mut others: Vec<usize> = kids.iter().copied().filter(|&k| k != golden).collect();
            for i in (1..others.len()).rev() {
                others.swap(i, r.below(i + 1)); // shuffle
            }
            for &lure in others.iter().take(r_t - 1) {
                cost[lure] = C_LOW; // looks cheaper than the golden child ⇒ lures greedy away
            }
            // the golden child's immediate cost is C_MID (rank r_t) at every INTERNAL gate, but the golden
            // LEAF is the PRIZE (cost 0 = s*): so the golden trajectory's TOTAL (D−1 gates · C_MID) is the
            // unique global minimum, strictly below any lure leaf (which pays C_LOW ON TOP of the gates it
            // branched from). Greedy is still lured off at the internal gates; only reaching the leaf pays off.
            cost[golden] = if depth[golden] == D { 0.0 } else { C_MID };
            on_golden[golden] = true;
            golden_path.push(golden);
        }
        // push children to keep expanding
        for &k in &kids {
            queue.push(k);
        }
    }

    let obs: Vec<f32> = cost.iter().map(|&c| cost_to_obs(c)).collect();
    let scram: Vec<f32> = (0..children.len()).map(|_| r.sym()).collect();
    World {
        children,
        obs,
        scram,
        golden_path,
    }
}

// ── the grown world model ────────────────────────────────────────────────────────────────────────────────
/// A GROWN model that predicts each node's observation `o(node)` from a one-hot node code — one independent
/// linear unit per node (`max_hidden = 0`), grown online with `grow` (never given). μ(node) drives the score.
fn grow_model(w: &World) -> Field {
    let n = w.n_nodes();
    let mut f = Field::new(n);
    f.ensure_cells(2 * n);
    f.config_mut().max_hidden = 0;
    for node in 0..n {
        for _ in 0..60 {
            // clear the bit first so each grow is a real 0→1 CHANGE that re-fires the unit against the CURRENT
            // weights (a unit fires only on change; without the reset the prediction stays stale and diverges).
            f.stimulate(&[(node, 0.0)], 4);
            f.grow(&[(node, 1.0)], n + node, w.obs[node], 0.3, 20);
        }
    }
    f
}

/// The model's predicted observation μ(node): fire the node's one-hot bit, read its output cell. Each unit
/// reads ONLY its own bit, so this is cross-talk-free and needs no clearing.
fn mu(model: &mut Field, n: usize, node: usize) -> f32 {
    // Clear the bit first so the re-injection is always a real 0→1 CHANGE that fires the unit (a unit fires
    // only on change; leaving other one-hot bits high is harmless since each unit reads only its own bit).
    model.stimulate(&[(node, 0.0)], 4);
    model.stimulate(&[(node, 1.0)], 20);
    model.cells()[n + node]
}

// ── the beam-search protocol (a usage protocol; greedy = width 1) ────────────────────────────────────────
/// One beam candidate: a partial trajectory. Its path lives as EPISODIC MEMORY inside a tiny memory-morph
/// (`mm`, one cell holding the visited node id), so the trajectory travels through every `fork_mutated`.
struct Cand {
    mm: Field,   // episodic memory of the path so far (node ids), travels with the fork
    node: usize, // current node
    cost: f32,   // cumulative score Σ G along the path
}

/// Store the current node id into a fresh episode of the memory-morph — memory IS cells.
fn remember(mm: &mut Field, node: usize) {
    mm.stimulate(&[(0, node as f32)], 4);
    mm.store_episode(&[0], &[]);
}

/// BEAM SEARCH over imagined trajectories. Width `K` (K=1 ⇒ greedy). `scramble` swaps the model's `G` for the
/// fixed random score field. Returns the winning candidate (lowest cumulative `G`). Composes: `grow`n model →
/// `G` score → `fork_mutated` branch + `store_episode` memory → global top-K.
fn beam(w: &World, model: &mut Field, k: usize, scramble: bool) -> Cand {
    let n = w.n_nodes();
    let root = 0usize;
    let mut mm0 = Field::new(1);
    remember(&mut mm0, root);
    let mut frontier = vec![Cand {
        mm: mm0,
        node: root,
        cost: 0.0,
    }];

    for _ in 0..D {
        let mut next: Vec<Cand> = Vec::new();
        for c in &frontier {
            for &ch in &w.children[c.node] {
                // SCORE each imagined continuation by the pragmatic free-energy term G(ch) = (μ(ch) − s*)².
                let m = if scramble {
                    w.scram[ch]
                } else {
                    mu(model, n, ch)
                };
                let g = (m - SSTAR) * (m - SSTAR);
                // BRANCH: fork the candidate's memory (the path travels with the fork) and append the child.
                let mut fm = c.mm.fork_mutated();
                remember(&mut fm, ch);
                next.push(Cand {
                    mm: fm,
                    node: ch,
                    cost: c.cost + g,
                });
            }
        }
        // KEEP the top-K over the WHOLE frontier (global beam), by lowest cumulative score.
        next.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        next.truncate(k);
        frontier = next;
    }
    // the winning trajectory: lowest cumulative G.
    frontier
        .into_iter()
        .min_by(|a, b| a.cost.total_cmp(&b.cost))
        .unwrap()
}

/// Recover the winning trajectory PURELY from the surviving candidate's episodic memory, by `replay` — proving
/// the path traveled through every fork (memory that travels, #189). Returns the recovered node sequence.
fn recover_path(win: &mut Cand) -> Vec<usize> {
    let n_ep = win.mm.episodes().len();
    let mut path = Vec::with_capacity(n_ep);
    for i in 0..n_ep {
        win.mm.replay(i, 4);
        path.push(win.mm.cells()[0].round() as usize);
    }
    path
}

#[test]
fn beam_search_over_imagined_trajectories_beats_greedy_and_needs_the_score() {
    // ── (A) the grown model must recover each node's observation (the score is a REAL prediction). ──
    {
        let w = generate(0x51);
        let mut model = grow_model(&w);
        let mae: f32 = (0..w.n_nodes())
            .map(|node| (mu(&mut model, w.n_nodes(), node) - w.obs[node]).abs())
            .sum::<f32>()
            / w.n_nodes() as f32;
        eprintln!(
            "[BEAM] (A) grown world model: mean |μ − o| = {mae:.4} over {} nodes",
            w.n_nodes()
        );
        assert!(
            mae < 0.05,
            "the grown model must predict each node's observation so G ranks the moves faithfully (MAE {mae:.4})"
        );
    }

    // ── (B) reach-vs-K: greedy (K=1) traps; wider beams reach the goal more often; scramble collapses. ──
    let widths = [1usize, 2, 3, 4, 5];
    let mut reach_real = [0usize; 5];
    let mut reach_scram = [0usize; 5];
    // one verified composition check: recover a winning path from memory and confirm it IS the golden path.
    let mut verified_memory = false;

    for seed in 0..SEEDS as u64 {
        let w = generate(0x1000 + seed);
        let mut model = grow_model(&w);
        let goal = w.golden_leaf();

        for (wi, &k) in widths.iter().enumerate() {
            let mut win = beam(&w, &mut model, k, false);
            if win.node == goal {
                reach_real[wi] += 1;
                // COMPOSITION check: the winner's memory, replayed, reconstructs the exact golden trajectory.
                if !verified_memory {
                    let path = recover_path(&mut win);
                    assert_eq!(
                        path, w.golden_path,
                        "the winning trajectory recovered from the candidate's OWN episodic memory (via replay) \
                         must equal the golden path — memory traveled through every fork"
                    );
                    assert_eq!(*path.last().unwrap(), goal, "and it ends at the goal leaf");
                    verified_memory = true;
                }
            }
            let scr = beam(&w, &mut model, k, true);
            if scr.node == goal {
                reach_scram[wi] += 1;
            }
        }
    }

    eprintln!("[BEAM] (B) reached the unique best leaf / {SEEDS} seeds, by beam width K:");
    for (wi, &k) in widths.iter().enumerate() {
        let tag = if k == 1 { "  (greedy)" } else { "" };
        eprintln!(
            "[BEAM]   K={k}{tag:9}  real-G {:>3}/{SEEDS} ({:>3.0}%)   scrambled-score {:>3}/{SEEDS} ({:>3.0}%)",
            reach_real[wi],
            100.0 * reach_real[wi] as f32 / SEEDS as f32,
            reach_scram[wi],
            100.0 * reach_scram[wi] as f32 / SEEDS as f32,
        );
    }
    eprintln!(
        "[BEAM] VERDICT: beam search = a USAGE PROTOCOL over grown model (`grow`) + free-energy score `G` + \
         fork-carried episodic memory (`fork_mutated`/`store_episode`/`replay`), ZERO core change. On a world \
         whose locally-best first step leads AWAY from the goal, GREEDY (K=1) traps on every seed ({}/{SEEDS}); \
         keeping K hypotheses reaches the goal MORE often as K rises, up to {}/{SEEDS} at full width. \
         Truth-not-trophy: the SAME beam scored by a RANDOM field (scramble) collapses back to ≈greedy \
         ({}/{SEEDS} at full width) — the SCORE is the load, not the branching. The winning trajectory was \
         recovered from the surviving candidate's OWN memory by replay.",
        reach_real[0], reach_real[4], reach_scram[4]
    );

    // greedy genuinely traps (not a strawman): the SAME beam code at width 1 never reaches the goal.
    assert_eq!(
        reach_real[0], 0,
        "greedy (beam width 1) must TRAP on the deceptive first step, every seed (reached {}/{SEEDS})",
        reach_real[0]
    );
    // keeping MORE hypotheses reaches the goal MORE often — the multi-hypothesis lever, made visible.
    assert!(
        reach_real[4] >= reach_real[2] && reach_real[2] >= reach_real[0],
        "reach must be non-decreasing in beam width K ({:?})",
        reach_real
    );
    assert!(
        reach_real[4] > reach_real[2] && reach_real[2] > reach_real[0],
        "wider beams must strictly reach the goal more often ({:?})",
        reach_real
    );
    // full-width beam reaches the goal on (nearly) every seed.
    assert!(
        reach_real[4] >= SEEDS * 9 / 10,
        "a full-width beam (K=B) must reach the goal on nearly every seed (reached {}/{SEEDS})",
        reach_real[4]
    );
    // truth-not-trophy: the score is load-bearing — a scrambled score collapses beam back to ≈greedy.
    assert!(
        reach_scram[4] <= SEEDS / 8,
        "a scrambled-score beam must collapse to ≈greedy even at full width (reached {}/{SEEDS})",
        reach_scram[4]
    );
    assert!(
        reach_real[4] >= reach_scram[4] * 4 + SEEDS / 2,
        "real-G beam must beat the scrambled-score beam decisively (real {} vs scrambled {} of {SEEDS})",
        reach_real[4],
        reach_scram[4]
    );
    assert!(
        verified_memory,
        "at least one winning trajectory must have been recovered from episodic memory (composition check)"
    );
}
