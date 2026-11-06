// SPDX-License-Identifier: MIT-0
//! The VALUE field and goals (`prefer`, `relax_values`, `orient`) plus the fast-weight rule
//! (`bind`/`complete`).

use super::*;

impl Field {
    /// Clamp a cell's graded VALUE as a PREFERRED / source state (a high-precision prior — the "goal" of
    /// purpose). [`Self::relax_values`] propagates toward it and never overwrites it. The organism acts to
    /// bring the world into the preferred state; the value field says how far each state is from it.
    ///
    /// # Example
    /// ```
    /// use openmorph_core::Field;
    /// // A 1-D line of 5 cells; the goal is cell 4. Adjacency = each cell's neighbours.
    /// let mut morph = Field::new(5);
    /// morph.prefer(4, 0.0); // clamp the goal (preferred state) at value 0
    /// let adj: Vec<Vec<usize>> = (0..5usize)
    ///     .map(|c| [c.wrapping_sub(1), c + 1].into_iter().filter(|&n| n < 5).collect())
    ///     .collect();
    /// morph.relax_values(&adj, 1.0, 0); // local correction to a fixed point
    /// // the value field is now distance-to-goal: descend it to reach the goal (no local minima)
    /// assert!(morph.value_of(0) > morph.value_of(3)); // cell 0 is farther than cell 3
    /// assert_eq!(morph.value_of(4), 0.0);             // the goal
    /// ```
    pub fn prefer(&mut self, cell: usize, value: f32) {
        self.ensure_cells(cell + 1);
        self.values[cell] = value;
        self.clamped[cell] = true;
    }

    /// The graded value of a cell (`f32::MAX` = unknown / unreachable).
    #[doc(hidden)]
    pub fn value_of(&self, cell: usize) -> f32 {
        self.values.get(cell).copied().unwrap_or(f32::MAX)
    }

    /// REINFORCE — accumulate a persistent GRADED action-drive onto `cell`: `drive[cell] += amount`.
    /// Negative `amount` is AVERSION (remember a failed move), positive is COMMITMENT (a rewarded one). Unlike
    /// [`Self::prefer`] this does NOT clamp — a reader merely WEIGHS it (soft, not a hard no-go). Goal-dependence is
    /// by the KEY: reinforce a `(goal,state,action)` cell, so the same action under a different goal is unaffected.
    /// Persistent (serde). Read by [`Self::orient`] (the extrinsic goal score); `alive(Event::Act)` does not read it
    /// (its reward memory is the per-option `(μ, visits)` record, [`Self::valence_of`]; the `ActScore` rule that
    /// read `drive` is removed).
    pub fn reinforce(&mut self, cell: usize, amount: f32) {
        self.ensure_cells(cell + 1);
        self.drive[cell] += amount;
    }

    /// Read a cell's accumulated action-drive (0.0 if never reinforced) — the aversive/reinforcing reward-memory
    /// [`Self::orient`] scores goals by. The graded twin of [`Self::value_of`].
    #[doc(hidden)]
    pub fn drive_of(&self, cell: usize) -> f32 {
        self.drive.get(cell).copied().unwrap_or(0.0)
    }

    /// LOCAL correction of the value field — the per-cell, online-correctable representation the plan-and-
    /// correct investigations settled on. Sweep every non-clamped cell to `cost + min(value over its connected
    /// neighbours)` toward a fixed point, over the `adjacency` the organism experiences. DOMAIN-BLIND: the
    /// caller supplies its grown / experienced connectivity (`adjacency[c]` = the cells reachable from `c`); an
    /// obstacle is simply a cell with empty adjacency, which stays `f32::MAX`. The relaxed field has NO interior
    /// local minima, so gradient descent on it crosses barriers a straight-line heuristic traps on; and because
    /// it is per-cell and local, an ONLINE re-relax after discovering a blocked edge touches only the cells whose
    /// value actually shifts (lazy). `max_sweeps == 0` ⇒ run to convergence. Returns the correction footprint
    /// (how many cells changed) — planning that corrects only where it must.
    pub fn relax_values(
        &mut self,
        adjacency: &[Vec<usize>],
        cost: f32,
        max_sweeps: usize,
    ) -> usize {
        // Value-field convergence tolerance: a numerical fixed-point ε (also the footprint threshold), NOT a
        // behavioural lever — kept a named const rather than a Config knob (audit: it doesn't move a task score).
        const VALUE_EPS: f32 = 1e-4;
        let n = self.mem.medium.len();
        self.ensure_cells(n);
        let before: Vec<f32> = self.values.clone();
        let cap = if max_sweeps == 0 { n * 2 } else { max_sweeps };
        for _ in 0..cap {
            let mut changed = false;
            for c in 0..n.min(adjacency.len()) {
                if self.clamped.get(c).copied().unwrap_or(false) {
                    continue;
                }
                let best = adjacency[c]
                    .iter()
                    .map(|&nb| self.value_of(nb))
                    .fold(f32::MAX, f32::min);
                let t = if best.is_finite() {
                    cost + best
                } else {
                    f32::MAX
                };
                if (t - self.values[c]).abs() > VALUE_EPS {
                    self.values[c] = t;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        (0..n)
            .filter(|&c| {
                (self.values[c] - before.get(c).copied().unwrap_or(f32::MAX)).abs() > VALUE_EPS
            })
            .count()
    }

    /// Reset the value field: clear every clamp and set all values to unknown (`f32::MAX`) — start a fresh
    /// preference / goal.
    pub fn clear_values(&mut self) {
        for v in self.values.iter_mut() {
            *v = f32::MAX;
        }
        for c in self.clamped.iter_mut() {
            *c = false;
        }
    }

    /// ORIENT (OM-6) — the organism GENERATES ITS OWN preferred state x*, then clamps it as the goal. This
    /// is the piece the harness used to author (`prefer(nearest_object, cost)`); now the morph SELECTS the goal
    /// from its OWN grown quantities. Among the `candidates` (the addressable state cells — a port, NOT a goal),
    /// it maximises the grown free-energy score
    ///
    /// ```text
    ///     score(c) = drive_of(c) + epistemic_weight · learning_progress(c)
    /// ```
    ///
    /// — EXTRINSIC value (accumulated reward-memory [`Self::drive_of`], from `reinforce`) PLUS EPISTEMIC value
    /// ([`Self::learning_progress`] = the *reducible* surprise, dSurprise/dt relative to the cell's own history).
    /// Both terms are GROWN; nothing is a hand feature. `epistemic_weight` is the META-EVOLVED coefficient
    /// (`Config::epistemic_weight`, discovered by `substrate_meta_drives`, not authored — the twin of the removed
    /// act rule's `openmindedness`). Using learning-PROGRESS (not raw surprise) is what dodges the DARK-ROOM: a
    /// fully-predictable do-nothing state has zero reducible surprise and zero drive, so it is never preferred — and
    /// the noisy-TV trap (high raw surprise, nothing learnable) is not chased either.
    ///
    /// The winner is picked by the GROWN winner-take-all [`Self::select`] (no authored argmax): the per-candidate
    /// scores are materialised into scratch cells select reads, and the slots are released back to the free pool
    /// (bounded — no medium growth per call). x* is then clamped via [`Self::prefer`] at cost-to-go `0.0`, so the
    /// planner ([`Self::relax_values`]) descends toward it. x* persists (it lives in the serialized value field).
    /// In CRYSTAL (`frozen`) the preferred state is HELD, not re-selected. Returns the chosen goal cell (`None`
    /// only if `candidates` is empty, or, when frozen, nothing is currently clamped).
    pub fn orient(&mut self, candidates: &[usize]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        // CRYSTAL: hold x*, don't re-update it (a frozen morph is a fixed competence — R7).
        if self.frozen {
            return candidates
                .iter()
                .copied()
                .find(|&c| self.clamped.get(c).copied().unwrap_or(false));
        }
        let w_epi = self.config.epistemic_weight;
        // The grown free-energy score per candidate (immutable reads first, so the mutable materialise below
        // does not alias): extrinsic drive + epistemic reducible surprise.
        let scores: Vec<f32> = candidates
            .iter()
            .map(|&c| self.drive_of(c) + w_epi * self.learning_progress(c))
            .collect();
        // Let the GROWN `select` pick the argmax: materialise the scores into scratch cells, select over them,
        // then hand the slots back to the free pool (reused next call — the medium does not grow per orient).
        let mut scratch = Vec::with_capacity(scores.len());
        for &v in &scores {
            let s = self.alloc_cell();
            self.mem.medium[s] = v;
            scratch.push(s);
        }
        let win = self.select(&scratch);
        for &s in &scratch {
            self.reset_cell(s);
            self.free_cells.push(s);
        }
        let goal = candidates[win];
        self.prefer(goal, 0.0);
        Some(goal)
    }

    /// ORIENT to MULTIPLE goals — generate the top-`k` self-set goals at once and clamp them as SIMULTANEOUS
    /// subgoals, so the planner ([`Self::relax_values`], which is multi-source) descends a cost-to-go field seeded
    /// by all of them. Same grown free-energy score as [`Self::orient`] (`drive_of + epistemic_weight ·
    /// learning_progress`) and the same grown winner-take-all [`Self::select`], applied `k` times WITHOUT
    /// replacement (no authored top-`k` sort — each round re-scores the remaining candidates and picks the WTA
    /// winner). Each chosen goal is clamped via [`Self::prefer`] at cost-to-go `0.0`. Returns the goal cells in
    /// selection order (fewer than `k` if candidates run out; empty if `candidates` is empty or `k == 0`). In
    /// CRYSTAL (`frozen`) it re-selects nothing and returns the already-clamped subset. The single-goal
    /// [`Self::orient`] is exactly `orient_many(candidates, 1)`.
    pub fn orient_many(&mut self, candidates: &[usize], k: usize) -> Vec<usize> {
        if candidates.is_empty() || k == 0 {
            return Vec::new();
        }
        if self.frozen {
            return candidates
                .iter()
                .copied()
                .filter(|&c| self.clamped.get(c).copied().unwrap_or(false))
                .collect();
        }
        let w_epi = self.config.epistemic_weight;
        let mut remaining: Vec<usize> = candidates.to_vec();
        let mut goals = Vec::new();
        let kk = k.min(remaining.len());
        for _ in 0..kk {
            // Score the remaining candidates by the grown free-energy quantity (immutable reads first).
            let scores: Vec<f32> = remaining
                .iter()
                .map(|&c| self.drive_of(c) + w_epi * self.learning_progress(c))
                .collect();
            // Grown winner-take-all over scratch cells (handed back to the pool — no medium growth per call).
            let mut scratch = Vec::with_capacity(scores.len());
            for &v in &scores {
                let s = self.alloc_cell();
                self.mem.medium[s] = v;
                scratch.push(s);
            }
            let win = self.select(&scratch);
            for &s in &scratch {
                self.reset_cell(s);
                self.free_cells.push(s);
            }
            let goal = remaining.remove(win);
            self.prefer(goal, 0.0);
            goals.push(goal);
        }
        goals
    }

    /// The cumulative number of SELF-SET goals the organism has REACHED (a taken option landed on a clamped
    /// preferred cell while [`Config::goal_reward`] is on). `0` when the goal-completion reward is off.
    pub fn goals_reached(&self) -> u64 {
        self.goals_reached
    }

    fn assoc_key(a: usize, b: usize) -> (usize, usize) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Fast-weight BIND one demonstration: strengthen the association between every pair of observed cells that
    /// share a value. This is RELATION-level (colour-independent), so it generalises to held-out values — the
    /// key to few-shot rule inference. One call per demo, ONE pass; over demos the truly-linked pairs (always
    /// equal) outweigh coincidental ones. The bound structure is the disposable rule for this puzzle.
    ///
    /// # Example
    /// ```
    /// use openmorph_core::Field;
    /// // Rule: cell 0 and cell 2 always share a value (a "copy" relation). SUGGEST it with a few examples...
    /// let mut morph = Field::new(3);
    /// for &v in &[1.0_f32, 2.0, 3.0] {
    ///     morph.bind(&[(0, v), (1, 9.0), (2, v)]); // 0 and 2 covary; 1 is unrelated
    /// }
    /// // ...then ANSWER a fresh, held-out-value question by completing the masked cell.
    /// morph.clear_values();
    /// morph.prefer(0, 7.0); // cell 0 is known (a held-out value never seen in the demos)
    /// morph.prefer(1, 5.0); // cell 1 is known
    /// assert_eq!(morph.complete(2), Some(7.0)); // cell 2 is bound to cell 0 → completes to 7.0
    /// ```
    pub fn bind(&mut self, observation: &[(usize, f32)]) {
        let tol = self.config.bind_tol;
        for a in 0..observation.len() {
            for b in (a + 1)..observation.len() {
                if (observation[a].1 - observation[b].1).abs() < tol {
                    *self
                        .assoc
                        .entry(Self::assoc_key(observation[a].0, observation[b].0))
                        .or_insert(0.0) += 1.0;
                }
            }
        }
    }

    /// Clear the fast-weight rule — disposable, a new puzzle. (The slow grown weights are untouched; only the
    /// fast per-puzzle binding resets.)
    pub fn unbind(&mut self) {
        self.assoc.clear();
    }

    /// The bound association strength between two cells (0 if never co-observed equal).
    pub fn bound(&self, a: usize, b: usize) -> f32 {
        self.assoc
            .get(&Self::assoc_key(a, b))
            .copied()
            .unwrap_or(0.0)
    }

    /// COMPLETE a cell from the bound rule: return the value of the CLAMPED (known — set via [`Self::prefer`])
    /// cell most strongly bound to `cell`. This applies the fast-bound relation to fill a masked cell
    /// consistently with the known ones — the `bind → prefer → complete` loop (bind the rule, clamp what's
    /// known, complete the rest). `None` if nothing is bound to `cell`.
    pub fn complete(&self, cell: usize) -> Option<f32> {
        let mut best: Option<f32> = None;
        let mut best_w = 0.0f32;
        for k in 0..self.clamped.len() {
            if k != cell && self.clamped.get(k).copied().unwrap_or(false) {
                let w = self.bound(cell, k);
                if w > best_w {
                    best_w = w;
                    best = self.values.get(k).copied();
                }
            }
        }
        best
    }
}
