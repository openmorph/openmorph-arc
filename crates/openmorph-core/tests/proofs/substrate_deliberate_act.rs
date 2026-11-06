// SPDX-License-Identifier: MIT-0
//! INTERNAL DELIBERATION — "think until confident, then act" (validates `docs/proposals/internal-deliberation.md`).
//!
//! THE FINDING THAT MOTIVATES IT (MorphBot, ARC-AGI-3 Kaggle). ARC-3 scores EFFICIENCY: a level's score falls the
//! more EXTERNAL actions the agent spends beating it (100% = human action-count, 0 = not beaten). The key
//! asymmetry: **external actions are scored; internal computation is FREE.** So the highest-leverage capability is
//! to think HARDER internally and act LESS often — deliberate to a confident choice instead of discovering it by
//! trial. Today the native actor (`alive(Event::Act)`) is ONE-SHOT/reactive: it reads the grown ensemble and
//! returns a pick, with no internal deliberation loop and no exposed confidence. That is the ceiling this proves a
//! way past — WITHOUT any new core primitive.
//!
//! THE CLAIM (measured here, three rungs of the proposal):
//!   * RUNG 1 — DELIBERATION BUDGET. The substrate already SETTLES a value field to a fixed point
//!     (`relax_values`). "Deliberation" is just running MORE internal sweeps before each external move. More free
//!     internal computation ⇒ STRICTLY FEWER external actions to the goal (the efficiency multiplicand ARC-3
//!     rewards), flooring at the shortest path.
//!   * RUNG 2 — CONFIDENCE READOUT. The relaxed field exposes a per-state confidence for free: the MARGIN between
//!     the best and second-best neighbour value (how decisive the gradient is). Pure read of `value_of`.
//!   * RUNG 4 — CONFIDENCE-GATED (ADAPTIVE) DELIBERATION. Deliberate only WHILE the margin is below a threshold,
//!     then commit. This reaches the goal in the same few external actions as always-deliberate-hard, yet spends
//!     FEWER TOTAL internal sweeps — think a lot when the choice is ambiguous, barely when it is obvious.
//!
//! COVERAGE IS NOT TRADED FOR EFFICIENCY (the proposal's honest worry, answered IN-SUBSTRATE). The Python bolt-on
//! traded coverage for efficiency because its directed acting narrowed exploration. Here deliberation routes
//! through `relax_values`, whose relaxed field has NO interior local minima (its own contract) — so EVERY arm,
//! from reactive to deep-deliberate, still REACHES the goal through a dead-end maze. Deliberation only changes HOW
//! FAST (external actions), never WHETHER (coverage). That is the whole point of doing it in the substrate rather
//! than as a tabular bolt-on.
//!
//! TRUTH-NOT-TROPHY. The world/harness (maze, costs, cap, seeded PRNG, the shortest-path oracle) is authored and
//! IDENTICAL across arms; the only thing that differs between arms is the DELIBERATION BUDGET — the number of free
//! internal `relax_values` sweeps. Nothing tells a policy the path; the value field is GROWN by relaxation over the
//! experienced adjacency. HONEST RESIDUAL: rung 3 (a GROWN forward model `imagine(option)` for internal rollouts in
//! worlds without a handed adjacency) needs `observe` + a learned transition model and belongs to MorphBot's live
//! ARC loop — it is deliberately NOT asserted here. ZERO openmorph-core change: a usage protocol over
//! `prefer`/`relax_values`/`value_of`, exactly as "the reasoner is a usage protocol over the substrate."
//!
//!   cargo test -p openmorph-core --test proofs substrate_deliberate_act -- --nocapture --test-threads=1

use openmorph_core::substrate::Field;
use std::collections::VecDeque;

// ── the maze world ──────────────────────────────────────────────────────────────────────────────────────────────

/// A grid maze on a `w×h` lattice with a set of blocked cells. Returns `(adjacency, start, goal)` where adjacency
/// is 4-connectivity minus walls — exactly the connectivity an organism would EXPERIENCE by moving. An obstacle is
/// a cell with no edges (unreachable), so the relaxed field routes around it. The reward (goal) location is handed
/// to the WORLD only; a policy sees it solely as a clamped value-0 cell it must descend to.
fn maze(w: usize, h: usize, walls: &[(usize, usize)]) -> (Vec<Vec<usize>>, usize, usize) {
    let id = |x: usize, y: usize| y * w + x;
    let blocked = |x: usize, y: usize| walls.contains(&(x, y));
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); w * h];
    for y in 0..h {
        for x in 0..w {
            if blocked(x, y) {
                continue;
            }
            let push = |nx: usize, ny: usize, adj: &mut Vec<Vec<usize>>| {
                if !blocked(nx, ny) {
                    adj[id(x, y)].push(id(nx, ny));
                }
            };
            if x + 1 < w {
                push(x + 1, y, &mut adj);
            }
            if x > 0 {
                push(x - 1, y, &mut adj);
            }
            if y + 1 < h {
                push(x, y + 1, &mut adj);
            }
            if y > 0 {
                push(x, y - 1, &mut adj);
            }
        }
    }
    (adj, id(0, 0), id(w - 1, h - 1))
}

/// The true shortest-path length start→goal over the maze adjacency (BFS) — the oracle the efficiency floor is
/// measured against. Authored, identical across arms, never given to a policy.
fn shortest_path(adj: &[Vec<usize>], start: usize, goal: usize) -> usize {
    let mut dist = vec![usize::MAX; adj.len()];
    let mut q = VecDeque::new();
    dist[start] = 0;
    q.push_back(start);
    while let Some(c) = q.pop_front() {
        if c == goal {
            return dist[c];
        }
        for &nb in &adj[c] {
            if dist[nb] == usize::MAX {
                dist[nb] = dist[c] + 1;
                q.push_back(nb);
            }
        }
    }
    usize::MAX
}

// ── the deliberating actor (a usage protocol over the substrate) ────────────────────────────────────────────────

/// The greedy step: move to the neighbour with the LOWEST grown value (descend the relaxed field). Ties and
/// all-unknown (`f32::MAX`, before relaxation has reached here) break to the lowest cell id — deterministic, so a
/// under-deliberated agent WANDERS rather than choosing randomly (makes the efficiency gap a property of
/// deliberation, not of luck). Returns the next cell (or `pos` if it has no neighbours).
fn descend(field: &Field, adj: &[Vec<usize>], pos: usize) -> usize {
    adj[pos]
        .iter()
        .copied()
        .min_by(|&a, &b| {
            field
                .value_of(a)
                .total_cmp(&field.value_of(b))
                .then(a.cmp(&b))
        })
        .unwrap_or(pos)
}

/// RUNG 2 — the CONFIDENCE MARGIN at `pos`: (second-best − best) neighbour value on the relaxed field. Large = a
/// decisive gradient (commit); ~0 = an ambiguous / not-yet-relaxed choice (deliberate more). A pure read of
/// `value_of` — the substrate already computes everything this needs. `f32::MAX` values (unreached) ⇒ margin 0.
fn confidence(field: &Field, adj: &[Vec<usize>], pos: usize) -> f32 {
    let mut vs: Vec<f32> = adj[pos].iter().map(|&n| field.value_of(n)).collect();
    vs.sort_by(f32::total_cmp);
    match (vs.first(), vs.get(1)) {
        (Some(&b0), Some(&b1)) if b0.is_finite() && b1.is_finite() => b1 - b0,
        _ => 0.0,
    }
}

/// RUNGS 1 & 4 — walk start→goal, returning `(external_actions, total_internal_sweeps, reached)`.
///
/// `budget`:
///   * `Budget::Fixed(n)` — RUNG 1: run exactly `n` internal `relax_values` sweeps before every external move
///     (`n = 0` ⇒ run to convergence). The value field PERSISTS across moves (you don't forget your plan), so a
///     small budget relaxes the field a little per step and the agent commits before the goal's influence has
///     spread — it wanders. A large budget converges the field first — it walks the shortest path.
///   * `Budget::GatedUntil(theta, cap)` — RUNG 4: deliberate one sweep at a time until the confidence MARGIN at the
///     current cell reaches `theta` (or `cap` sweeps this step), then commit. Adaptive: heavy deliberation early
///     (far, field unrelaxed, ambiguous), almost none late (on a sharp gradient).
#[derive(Clone, Copy)]
enum Budget {
    Fixed(usize),
    GatedUntil(f32, usize),
}

fn walk(
    adj: &[Vec<usize>],
    start: usize,
    goal: usize,
    budget: Budget,
    cap: usize,
) -> (usize, usize, bool) {
    let mut field = Field::new(adj.len());
    field.prefer(goal, 0.0); // the found reward: clamp its cost-to-go at 0
    let mut pos = start;
    let mut actions = 0usize;
    let mut sweeps = 0usize;
    while pos != goal && actions < cap {
        match budget {
            Budget::Fixed(n) => {
                field.relax_values(adj, 1.0, n);
                // count real sweeps: n, or (for converge) up to the field diameter — approximate by adjacency len.
                sweeps += if n == 0 { adj.len() } else { n };
            }
            Budget::GatedUntil(theta, sweep_cap) => {
                let mut s = 0;
                while s < sweep_cap {
                    field.relax_values(adj, 1.0, 1);
                    s += 1;
                    if confidence(&field, adj, pos) >= theta {
                        break; // confident enough — stop thinking, act
                    }
                }
                sweeps += s;
            }
        }
        pos = descend(&field, adj, pos);
        actions += 1;
    }
    (actions, sweeps, pos == goal)
}

// ── the proof ───────────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn deliberation_cuts_external_actions_without_losing_coverage() {
    // A dead-end maze: a serpentine wall forces the path to snake, so a under-deliberated agent (committing before
    // the value front arrives) wanders into pockets. 7×7 with two comb walls.
    let (w, h) = (7, 7);
    let walls = [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5), // left comb tooth (gap at top row 0)
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5), // right comb tooth
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6), // third tooth (gap at row 1)
    ];
    let (adj, start, goal) = maze(w, h, &walls);
    let sp = shortest_path(&adj, start, goal);
    assert!(sp != usize::MAX, "maze must be solvable");
    let cap = 400; // generous — every arm must reach WELL under this (coverage), the interest is HOW FAST

    // RUNG 1 — sweep the deliberation budget. More free internal sweeps ⇒ fewer external actions, monotonically.
    let mut prev = usize::MAX;
    let mut row = Vec::new();
    for &n in &[1usize, 2, 4, 8, 16] {
        let (actions, _sw, reached) = walk(&adj, start, goal, Budget::Fixed(n), cap);
        assert!(
            reached,
            "deliberate={n}: coverage — must reach the goal (relaxed field has no local minima)"
        );
        row.push((n, actions));
        // monotone non-increasing: thinking more never costs more external actions.
        assert!(
            actions <= prev,
            "deliberate={n}: {actions} external actions must be ≤ the lower budget's {prev}"
        );
        prev = actions;
    }

    // Converge-before-acting = the shortest path exactly (optimal efficiency: internal thought fully cashed in).
    let (opt_actions, _sw, opt_reached) = walk(&adj, start, goal, Budget::Fixed(0), cap);
    assert!(opt_reached, "converge: coverage");
    assert_eq!(
        opt_actions, sp,
        "full deliberation walks the shortest path ({sp}) — the efficiency floor"
    );

    // The efficiency WIN is large: the least-deliberate arm spends many more external actions than the floor.
    let reactive = row[0].1; // deliberate=1
    assert!(
        reactive as f32 >= 1.5 * sp as f32,
        "reactive (deliberate=1) {reactive} should be ≥1.5× the shortest path {sp} — the flailing the metric punishes"
    );

    // RUNG 4 — confidence-gated adaptive deliberation. Reaches the goal in ~shortest-path external actions (matches
    // the always-hard arm) while spending FEWER total internal sweeps than fixing the budget high every step.
    let (gated_actions, gated_sweeps, gated_reached) =
        walk(&adj, start, goal, Budget::GatedUntil(0.5, 64), cap);
    let (_hard_actions, hard_sweeps, _r) = walk(&adj, start, goal, Budget::Fixed(64), cap);
    assert!(gated_reached, "gated: coverage");
    assert!(
        gated_actions <= (sp as f32 * 1.5).ceil() as usize,
        "gated stays near the efficiency floor: {gated_actions} external actions vs shortest path {sp}"
    );
    assert!(
        gated_sweeps < hard_sweeps,
        "gated spends FEWER internal sweeps ({gated_sweeps}) than always-hard ({hard_sweeps}) — think only when unsure"
    );

    println!("\n  INTERNAL DELIBERATION — external actions to goal (shortest path = {sp}):");
    for (n, a) in &row {
        println!("    deliberate={n:<3} → {a} external actions");
    }
    println!("    deliberate=∞   → {opt_actions} external actions (= shortest path, optimal)");
    println!(
        "    confidence-gated → {gated_actions} external actions, {gated_sweeps} internal sweeps \
         (vs always-hard {hard_sweeps} sweeps for the same reach)"
    );
    println!("  ⇒ free internal thought buys external-action efficiency; coverage never traded.\n");
}
