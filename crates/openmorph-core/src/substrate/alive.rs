// SPDX-License-Identifier: MIT-0
//! The ONE alive entry (`alive`), the crystal mode switch, and the grown-basis reads it meters.

use super::*;

impl Field {
    /// Enter CRYSTAL mode — freeze the organism into a snapshot of competence at THIS moment. No growth,
    /// learning, culling, or map/economy change happens while crystallised; only fire-on-change → settle (pure
    /// inference) runs. Per-unit plasticity is untouched (a thaw restores exactly what was alive), so this
    /// costs zero training compute — the microcontroller / evaluation / deployment path. The complement of the
    /// ALIVE loop.
    ///
    /// # Example
    /// ```
    /// use openmorph_core::{Event, Field};
    /// let mut morph = Field::new(2);
    /// for _ in 0..500 { for &x in &[-1.0_f32, 1.0] {
    ///     morph.alive(Event::Perceive { inputs: &[(0, x)], target_cell: 1, target: 0.6 * x, lr: 0.1, budget: 100 });
    /// }}
    /// morph.stimulate(&[(0, 1.0)], 100);
    /// let learned = morph.cells()[1];
    ///
    /// morph.crystallize();                    // freeze the snapshot
    /// assert!(morph.is_crystal());
    /// // learning is now a no-op — hammer it with the opposite task, inference is unchanged
    /// for _ in 0..500 {
    ///     morph.alive(Event::Perceive { inputs: &[(0, 1.0)], target_cell: 1, target: -0.9, lr: 0.3, budget: 100 });
    /// }
    /// morph.stimulate(&[(0, 1.0)], 100);
    /// assert_eq!(morph.cells()[1], learned); // byte-identical — a frozen snapshot of competence
    /// morph.thaw();                           // back to alive; learning resumes
    /// ```
    pub fn crystallize(&mut self) {
        self.frozen = true;
    }

    /// Leave CRYSTAL mode — return to ALIVE (learning at inference). Growth/learning/economy resume with the
    /// per-unit plasticity that was preserved through the freeze.
    pub fn thaw(&mut self) {
        self.frozen = false;
    }

    /// Whether the organism is CRYSTAL (frozen, inference-only) vs ALIVE (learning at inference).
    pub fn is_crystal(&self) -> bool {
        self.frozen
    }

    /// The GROWN BASIS — the internal grown features the organism can prune: cells that a unit writes AND that
    /// at least one other unit reads (they feed forward). Terminal outputs (read only by the world, never by a
    /// unit) are excluded, so the reliance economy meters and culls only the excess basis, never the answer.
    /// Domain-blind: no index convention, just the grown graph's own shape.
    pub fn grown_basis(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .units
            .iter()
            .map(|u| u.write)
            .filter(|&c| self.readers.get(c).is_some_and(|r| !r.is_empty()))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The grown features a specific `target_cell` relies on: the cells its writer(s) READ that are themselves
    /// GROWN (written by a unit — i.e. not raw input/world cells). This is the per-target slice of the
    /// [`Self::grown_basis`], so the reliance economy can meter a skill's OWN features on that skill's events
    /// (see [`Self::alive`]) instead of draining every skill's features on every event — the key to
    /// accumulating multiple skills through one `alive` loop.
    pub fn target_basis(&self, target_cell: usize) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .writers
            .get(target_cell)
            .into_iter()
            .flatten()
            .flat_map(|&w| self.units[w].reads.iter().copied())
            .filter(|&c| self.writers.get(c).is_some_and(|w| !w.is_empty()))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    // ── core loop ──
    /// THE ONE ALIVE ENTRY — the whole v1.3.1 loop in a single call. Feed an [`Event`]; the organism decides
    /// what to do, with NO external orchestration:
    /// - [`Event::Perceive`] → learning at inference: GROW on surprise, learn, and SIZE ITSELF by the reliance
    ///   economy (pay the grown basis ∝ reliance, cull the starved with a newborn grace) — the substrate
    ///   running unimpeded. In CRYSTAL it degrades to pure inference (settle only).
    /// - [`Event::Suggest`] → bind the demonstrated rule on the FAST path (works even in CRYSTAL — a deployed
    ///   morph few-shot-adapts).
    /// - [`Event::Answer`] → clamp the known cells and complete the unknown ones toward the bound consistency.
    ///
    /// Self-inferred goals (the autonomous default) and SUGGESTED goals (the ARC channel) go through the same
    /// entry. The living-pool economy rates are `Config` levers (`pool_income`/`pool_upkeep`/`pool_death`/
    /// `pool_grace`); their defaults keep the survival breakeven at a near-ZERO reliance so alive prunes only
    /// IGNORED capacity, never the distributed features of a nonlinear solution (it matches raw `grow` on XOR).
    /// The economy is metered on the ACTIVE target's features ([`Self::target_basis`]) — where this event's
    /// learning happens — so training MANY skills (separate output channels) through one alive loop accumulates
    /// them additively: a skill's features are judged on ITS events, not drained by other interleaved skills'
    /// (a global economy strands a minority nonlinear skill like XOR). A bigger multi-skill morph settles a
    /// larger `budget`. A settle of a grown target costs its hidden-feature count + 2 firings, so a `budget` of
    /// `max_hidden + 2` (66 at the default 64) per grown target that the changed inputs reach never binds. A smaller
    /// one BOUNDS the pool under the default genome ([`Config::grow_within_budget`]: a feature is born only while
    /// the settle still fits); with the bound off ([`Config::legacy`]) the pool outgrows it and every settle returns
    /// a TRUNCATED, not-yet-settled prediction without an error — [`Self::truncated_settles`] counts each one.
    ///
    /// # Example
    /// ```
    /// use openmorph_core::{Event, Field};
    /// let mut morph = Field::new(2); // cell 0 = input, cell 1 = target
    /// // ALIVE: learn x -> tanh(1.5·x) at inference — grows on surprise, self-sizes, no orchestration.
    /// for _ in 0..800 {
    ///     for &x in &[-1.0_f32, 1.0] {
    ///         morph.alive(Event::Perceive {
    ///             inputs: &[(0, x)], target_cell: 1, target: (1.5 * x).tanh(), lr: 0.1, budget: 100,
    ///         });
    ///     }
    /// }
    /// morph.stimulate(&[(0, 1.0)], 100);
    /// assert!((morph.cells()[1] - (1.5_f32).tanh()).abs() < 0.1);
    ///
    /// // CRYSTAL: freeze a snapshot of competence — inference only, zero training compute.
    /// morph.crystallize();
    /// assert!(morph.is_crystal());
    /// ```
    pub fn alive(&mut self, event: Event) -> Outcome {
        match event {
            Event::Perceive {
                inputs,
                target_cell,
                target,
                lr,
                budget,
            } => {
                if self.frozen {
                    self.stimulate(inputs, budget);
                    return Outcome::Perceived {
                        error: 0.0,
                        born: false,
                    };
                }
                let (error, born) = self.grow(inputs, target_cell, target, lr, budget);
                // The living-pool economy self-sizes the grown basis: pay each feature ∝ how much the readout
                // RELIES on it, cull the starved (with a newborn grace). The upkeep is deliberately SMALL vs
                // income, so the survival breakeven is a near-ZERO reliance — the economy prunes only features
                // the readout genuinely IGNORES (|weight| ~ 0), NOT the used-but-modest features of a DISTRIBUTED
                // basis (a nonlinear task like XOR spreads its solution across many random-projection features,
                // each with a small weight). A large upkeep would cull that distributed solution mid-formation.
                // The rates are `Config` levers (defaults tuned so alive matches raw `grow` on XOR).
                let (income, upkeep, death, grace) = (
                    self.config.pool_income,
                    self.config.pool_upkeep,
                    self.config.pool_death,
                    self.config.pool_grace,
                );
                // Meter only the features THIS event's target relies on, not the global pool. In a CURRICULUM
                // of skills on separate channels, a skill's readout learns (builds reliance) only on ITS OWN
                // events, but a global upkeep would drain its specialist features on EVERY other skill's event
                // too — so a minority skill (e.g. XOR, 1/6 of an interleaved stream) pays upkeep 6× faster than
                // it earns reliance and its features die before they mature (XOR stuck at chance while the
                // linear skills thrive). Scoping the economy to the active target makes each skill's sub-pool
                // self-size on its own events — additive skill-building through one `alive` loop, no gating.
                let basis = self.target_basis(target_cell);
                let fitness = if self.config.pool_surprise {
                    super::economy::Fitness::PoolSurprise
                } else {
                    super::economy::Fitness::Reliance
                };
                self.metabolize_cells(&basis, income, upkeep, fitness);
                self.cull_after(death, grace);
                Outcome::Perceived { error, born }
            }
            Event::Suggest { observation } => {
                self.bind(observation);
                Outcome::Suggested
            }
            Event::Answer { known, unknown } => {
                self.clear_values();
                for &(c, v) in known {
                    self.prefer(c, v);
                }
                let answer = unknown
                    .iter()
                    .filter_map(|&u| self.complete(u).map(|v| (u, v)))
                    .collect();
                Outcome::Answered(answer)
            }
            Event::Act { options, last } => {
                // EPISODE END: an empty `options` offers no successor. Only `last` is folded (no bootstrap), and
                // nothing is chosen: `usize::MAX` makes a caller that indexes with it fail loudly.
                if options.is_empty() {
                    self.end_episode(last);
                    return Outcome::Acted { chosen: usize::MAX };
                }
                // ONE actor path, two scores (`Config::act_rule`; the deprecated `free_energy_actor` /
                // `valence_actor` bools still select their rule — `Config::effective_act_rule`). Both rules fold
                // the last reward (+ the goal-completion self-reward) through their own currency, score the offered
                // options, record the `g_spread` diagnostic, read off one option and then apply `action_dwell`.
                let chosen = match self.config.effective_act_rule() {
                    ActRule::Valence => self.act_valence(options, last),
                    ActRule::FreeEnergy => self.act_free_energy(options, last),
                };
                // ACTION DWELL (opt-in, every rule) — hold a working choice for its GROWN duration instead of
                // re-deciding every step. Off (default) ⇒ `chosen` passes through untouched, byte-identical.
                let chosen = if self.config.action_dwell {
                    self.apply_dwell(chosen, options, last)
                } else {
                    chosen
                };
                Outcome::Acted { chosen }
            }
        }
    }
}
