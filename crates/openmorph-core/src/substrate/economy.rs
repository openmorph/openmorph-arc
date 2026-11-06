// SPDX-License-Identifier: MIT-0
//! The ENERGY economy — metabolize (credit / reliance / surprise), sensitivity and reliance meters,
//! `repopulate`.

use super::*;

/// Which grown FITNESS meters a unit's metabolic income in the one shared metabolize loop: residual `credit`
/// (contribution), reader `reliance`, or end-to-end `surprise`-reduction. Selects only the income FACTOR — the
/// energy update, order, and guards are identical across all three, so the choice never changes the arithmetic.
#[derive(Clone, Copy)]
pub(super) enum Fitness {
    Credit,
    Reliance,
    Surprise,
    /// Surprise reduction scaled by the pool rule, as `Reliance` is (`Config::pool_surprise`).
    PoolSurprise,
}

impl Field {
    /// Apply the `energy_cap` ceiling to a freshly-computed raw energy, ACCOUNTING the discarded overflow into
    /// [`Self::dissipated_overflow`]. Returns `raw.min(energy_cap)` — byte-identical to the bare `.min` it replaces
    /// (the accounting is a passive side-count, never fed back), so the metabolize paths stay bit-for-bit.
    fn cap_energy(&mut self, raw: f32) -> f32 {
        let cap = self.config.energy_cap;
        if raw > cap {
            self.dissipated_overflow += raw - cap;
        }
        raw.min(cap)
    }

    /// The running total of energy DISCARDED by the `energy_cap` ceiling in the metabolize paths (the overflow
    /// `.min(energy_cap)` silently drops). A denominator for where the economy's energy goes — observability only,
    /// never fed back into dynamics. Counts the metabolize cap-overflow ONLY (see the field docs for what is out
    /// of scope). Diagnostic, never a selection gate.
    pub fn dissipated_overflow(&self) -> f32 {
        self.dissipated_overflow
    }

    /// Shared cell-metabolism: pay each cell's writer `income` scaled by an income FACTOR, minus `upkeep`,
    /// capped. The fitnesses differ ONLY in the factor — residual `credit` (breeds traits/evolution), `reliance`
    /// (keeps/culls structure) and surprise-reduction (the gated currency spike) — so they share one body.
    pub(super) fn metabolize_cells(
        &mut self,
        cells: &[usize],
        income: f32,
        upkeep: f32,
        fitness: Fitness,
    ) {
        if self.frozen {
            return;
        }
        self.hydrate_traits();
        // RELATIVE RELIANCE (`Config::reliance_norm`, see `RelianceNorm`): normalise reliance by the metabolised
        // pool's MEAN (or, for the deprecated `relative_reliance`, its TOTAL), so the survival breakeven
        // `upkeep/income` is amplitude-INVARIANT ("carry ≥ that fraction of the average feature") instead of an
        // absolute Σ|w| that culls a low-amplitude working basis. `Absolute` ⇒ `norm = 1.0`, the raw `reliance_of`.
        // Reliance of every metered cell up front (a pure read of weights/adjacency, which the energy updates below
        // never touch — so hoisting it is byte-identical), batched so a shared reader's reads are indexed ONCE.
        let rel: Vec<f32> = match fitness {
            Fitness::Reliance => self.reliance_batch(cells),
            Fitness::PoolSurprise => cells
                .iter()
                .map(|&c| self.surprise_reduction_of(c))
                .collect(),
            _ => Vec::new(),
        };
        let pooled = matches!(fitness, Fitness::Reliance | Fitness::PoolSurprise);
        let rule = self.config.effective_reliance_norm();
        let norm = if pooled && rule != RelianceNorm::Absolute {
            let (mut total, mut live) = (0.0f32, 0usize);
            for (k, &c) in cells.iter().enumerate() {
                if self.writers.get(c).and_then(|w| w.first()).is_some() {
                    total += rel[k];
                    live += 1;
                }
            }
            match rule {
                RelianceNorm::Mean => (total / live.max(1) as f32).max(1e-9), // relative to the AVERAGE feature
                _ => total.max(1e-9), // Share: relative to the pool total
            }
        } else {
            1.0
        };
        for (k, &c) in cells.iter().enumerate() {
            if let Some(&i) = self.writers.get(c).and_then(|w| w.first()) {
                let factor = match fitness {
                    Fitness::Reliance | Fitness::PoolSurprise => rel[k] / norm,
                    Fitness::Credit => self.credit[i].abs(),
                    Fitness::Surprise => self.surprise_reduction_of(c),
                };
                self.energy[i] = self.cap_energy(self.energy[i] + income * factor - upkeep);
            }
        }
    }

    // ── economy ──
    /// METABOLIZE BY CONTRIBUTION — the alive economy with the RIGHT income for tuning constants: a unit in
    /// `cells` earns its accumulated CONTRIBUTION credit (`credit_of`, residual correlation) and pays
    /// `upkeep`, so a useful unit banks energy and a useless one starves — regardless of firing. This is the
    /// make-or-break fix ([`Self::metabolize`]'s firing gate credits a random backbone equally, so the
    /// economy would select on noise). Scoped to `cells` so only the competing population is judged.
    /// `income` scales contribution into energy units (credit magnitudes are small relative to the birth
    /// reserve); a unit earns `income · |credit|` and pays `upkeep`.
    #[doc(hidden)]
    pub fn metabolize_contribution(&mut self, cells: &[usize], income: f32, upkeep: f32) {
        self.metabolize_cells(cells, income, upkeep, Fitness::Credit);
    }

    /// SURPRISE-CURRENCY end-to-end sensitivity (Gap 7 spike, increment 1.5) — accumulate a running EMA of the
    /// SQUARED end-to-end sensitivity `(∂output/∂cell)²` for EVERY cell, by ONE local reverse (adjoint) sweep of the
    /// leaf graph seeded at `output_cell`. This is the transpose of the forward Jacobian already used by
    /// [`Self::spectral_radius`]: it chains the ONE-hop residual backprojection of [`Self::learn_feature_directions`]
    /// all the way to the raw inputs, so `surprise_reduction_of` reflects how much removing a cell would raise the
    /// OUTPUT's error — causal THROUGH a random backbone (where the immediate-fan-out `Σ w²·Var` degenerates to
    /// reliance). A NO-OP touching no state unless [`Config::surprise_currency`] is on ⇒ byte-identical when off.
    ///
    /// The adjoint is a triangular fixed point on a feed-forward (leaf) graph — `adj[c] = [c==out] + Σ_{u reads c}
    /// adj[u.write]·slope·gain·w` — so a fixed number of Jacobi passes (bounded by the leaf depth; nested cells are
    /// opaque bricks and end a path) reaches it exactly. Cheap and local (no backprop-through-time, no global solve).
    pub(super) fn accumulate_sensitivity(&mut self, output_cell: usize) {
        if !self.config.tracks_surprise() {
            return; // dormant ⇒ byte-identical
        }
        let n = self.mem.medium.len();
        if output_cell >= n {
            return;
        }
        if self.cell_sens2.len() < n {
            self.cell_sens2.resize(n, 0.0);
        }
        // Reverse (adjoint) sweep: adj[c] = ∂output_value/∂(value at c). Seed the output at 1 and push sensitivity
        // from each unit's WRITE cell back to its READ cells through the local slope·gain·weight, iterating to the
        // fixed point (a DAG of leaf units is triangular ⇒ it converges in ≤ depth passes; break early on stall).
        let mut adj = vec![0.0f32; n];
        adj[output_cell] = 1.0;
        // Bound: the number of leaf units caps the longest leaf path; small nets stop far sooner via the stall break.
        let max_passes = self.units.len().max(1);
        for _ in 0..max_passes {
            let mut next = vec![0.0f32; n];
            next[output_cell] = 1.0;
            for (i, u) in self.units.iter().enumerate() {
                if u.nested.is_some() {
                    continue; // a nested cell is a frozen sub-morph brick — opaque, it ends the path
                }
                let a_w = adj.get(u.write).copied().unwrap_or(0.0);
                if a_w == 0.0 {
                    continue;
                }
                let mut gain = self.scale.get(i).copied().unwrap_or(1.0);
                if self.config.gain_fanin_norm && !u.reads.is_empty() {
                    gain /= (u.reads.len() as f32).sqrt(); // honor the g/√N gain in the Jacobian (as spectral_radius does)
                }
                let m = self.mem.medium.get(u.write).copied().unwrap_or(0.0);
                let slope = self.slope_of(i, m); // activation′ · any forward factor, at the settled state
                let coeff = a_w * slope * gain;
                for (k, &c) in u.reads.iter().enumerate() {
                    if c < n {
                        next[c] += coeff * u.weights[k];
                    }
                }
            }
            // Stall break: stop once the sweep no longer changes the adjoint (feed-forward reaches this exactly).
            let mut delta = 0.0f32;
            for c in 0..n {
                delta += (next[c] - adj[c]).abs();
            }
            adj = next;
            if delta <= 1e-12 {
                break;
            }
        }
        // EMA of the SQUARED end-to-end sensitivity (reuse the surprise decay — no new constant).
        let a = self.config.surprise_decay;
        for c in 0..n {
            let s2 = adj[c] * adj[c];
            self.cell_sens2[c] += a * (s2 - self.cell_sens2[c]);
        }
    }

    /// SURPRISE-CURRENCY (Gap 7 spike) — update the running per-cell mean/variance EMA. A no-op (touching NO state)
    /// unless [`Config::surprise_currency`] is on, so the settle path is BYTE-IDENTICAL when off. The variance is
    /// the ingredient of the ablation importance `Σ w²·Var(cell)` — the stable-at-convergence, causal currency.
    #[inline]
    pub(super) fn note_value(&mut self, cell: usize, v: f32) {
        if !self.config.tracks_surprise() {
            return;
        }
        if self.cell_var.len() <= cell {
            self.cell_mean.resize(cell + 1, 0.0);
            self.cell_var.resize(cell + 1, 0.0);
        }
        // Welford-style EMA (reuse the surprise decay so no new constant): mean ← mean + a·d; var ← var + a·(d²−var).
        let a = self.config.surprise_decay;
        let d = v - self.cell_mean[cell];
        self.cell_mean[cell] += a * d;
        self.cell_var[cell] += a * (d * d - self.cell_var[cell]);
    }

    /// SURPRISE-REDUCTION of cell `c` (Gap 7 spike, [`Config::surprise_currency`]) — the END-TO-END ABLATION
    /// importance `(∂output/∂c)²·Var(c)`: how much removing this cell would raise the OUTPUT's error. Unlike
    /// [`Self::credit_of`] (residual correlation, → 0 at convergence) it STAYS nonzero for a needed feature; unlike
    /// [`Self::reliance_of`] (Σ|weight|) it weights by the cell's own VARIANCE and by its END-TO-END sensitivity, so
    /// a constant/ignored cell earns ~0 AND a noise input under a random backbone earns ~0 (the readout's learned
    /// downstream weights zero its path). The unified currency with reliance's stability and credit's causality
    /// (`spike_currency_stability`, `spike_ablation_sensitivity`).
    ///
    /// Increment 1.5: the squared sensitivity is the running EMA chained through the hidden backbone by
    /// `accumulate_sensitivity` (a local adjoint sweep). It falls back to the increment-1 immediate-fan-out
    /// form `Σ w²·Var` only if no sensitivity has warmed yet (e.g. `surprise_reduction_of` read before any learn
    /// event). Returns 0 when the flag is off (no state tracked). Still does NOT robustly discount collinear
    /// REDUNDANCY: a duplicate's individual importance is split seed-dependently by the random backbone's routing
    /// (a conditional-variance `Var(c | siblings)` discount was investigated and found NOT to earn its place — the
    /// fast surprise EMA cannot hold a stable correlation; see `docs/research/surprise-currency-spike.md` §3). The
    /// stable keeper of redundant structure remains `reliance_of` (it holds every copy).
    #[doc(hidden)]
    pub fn surprise_reduction_of(&self, c: usize) -> f32 {
        let var = self.cell_var.get(c).copied().unwrap_or(0.0);
        if var <= 0.0 {
            return 0.0;
        }
        // END-TO-END: (∂output/∂c)² · Var(c), the sensitivity chained through the hidden backbone.
        if let Some(&sens2) = self.cell_sens2.get(c) {
            if sens2 > 0.0 {
                return sens2 * var;
            }
        }
        // Fallback (no sensitivity warmed yet): the increment-1 immediate-fan-out form Σ (weight on c)² · Var(c).
        let w2: f32 = self
            .readers
            .get(c)
            .map(|rs| {
                rs.iter()
                    .map(|&r| {
                        let u = &self.units[r];
                        u.reads
                            .iter()
                            .position(|&rc| rc == c)
                            .map(|k| u.weights[k] * u.weights[k])
                            .unwrap_or(0.0)
                    })
                    .sum()
            })
            .unwrap_or(0.0);
        w2 * var
    }

    /// The RELIANCE of feature cell `c` — how much its READERS lean on it: `Σ over readers of |weight on c|`.
    /// Convergence-STABLE (weights settle), unlike `credit_of` (residual-correlation, which → 0 for a USED
    /// feature at convergence). This is the structural keep/cull fitness (the two-fitness split).
    #[doc(hidden)]
    pub fn reliance_of(&self, c: usize) -> f32 {
        self.readers
            .get(c)
            .map(|rs| {
                rs.iter()
                    .map(|&r| {
                        let u = &self.units[r];
                        u.reads
                            .iter()
                            .position(|&rc| rc == c)
                            .map(|k| u.weights[k].abs())
                            .unwrap_or(0.0)
                    })
                    .sum()
            })
            .unwrap_or(0.0)
    }

    /// [`Self::reliance_of`] for every cell in `cells`, bit-for-bit — batched. `reliance_of` finds `c`'s weight in
    /// each reader by a linear `position` scan of that reader's reads, so metering a grown basis of `H` features all
    /// read by ONE wide readout (the living pool's shape) cost `O(H · fan-in)`. Here a cell with a SINGLE reader
    /// looks its first-occurrence index up in a dense map built once per distinct reader (`O(fan-in)`), and sums the
    /// one term through the same `Sum` fold `reliance_of` uses; a multi-reader cell just calls `reliance_of`.
    fn reliance_batch(&self, cells: &[usize]) -> Vec<f32> {
        let n = self.mem.medium.len();
        let mut first: Vec<usize> = Vec::new(); // cell → 1 + first index in the cached reader's reads (0 = absent)
        let mut cached: Option<usize> = None;
        let mut out = Vec::with_capacity(cells.len());
        for &c in cells {
            let rs = match self.readers.get(c) {
                Some(rs) if rs.len() == 1 && c < n && self.units.get(rs[0]).is_some() => rs,
                _ => {
                    out.push(self.reliance_of(c));
                    continue;
                }
            };
            let r = rs[0];
            if cached != Some(r) {
                if first.len() < n {
                    first.resize(n, 0);
                }
                if let Some(old) = cached {
                    for &x in &self.units[old].reads {
                        if x < n {
                            first[x] = 0;
                        }
                    }
                }
                for (k, &x) in self.units[r].reads.iter().enumerate() {
                    if x < n && first[x] == 0 {
                        first[x] = k + 1;
                    }
                }
                cached = Some(r);
            }
            let term = match first[c] {
                0 => 0.0,
                k => self.units[r].weights[k - 1].abs(),
            };
            out.push(core::iter::once(term).sum::<f32>()); // the same one-term `Sum` fold as `reliance_of`
        }
        out
    }

    /// EFFECTIVE RANK — a **fitness / health primitive** (v2). The participation ratio of the units' reliance,
    /// `PR = (Σ rᵢ)² / Σ rᵢ²` with `rᵢ = reliance_of(unit i's write cell)`: **how many units effectively carry
    /// the load** (≈1 if one unit dominates, ≈N if the load is spread evenly). Unlike `units.len()` (the *nominal*
    /// size), this is the *used* dimensionality — and, per **H-S4**, it is what reliance-cull does NOT reduce to
    /// (a distributed nonlinear basis keeps a high effective rank the economy over-prunes). Bare-metal, data-free.
    /// A fitness health term: prefer a genome that solves a task at LOWER effective rank (parsimony) while keeping
    /// the settling `ρ` sub-critical (stability) — see [`Self::spectral_radius`].
    pub fn effective_rank(&self) -> f32 {
        let rels: Vec<f32> = self
            .units
            .iter()
            .map(|u| self.reliance_of(u.write))
            .collect();
        let s1: f32 = rels.iter().sum();
        let s2: f32 = rels.iter().map(|r| r * r).sum();
        if s2 <= f32::EPSILON {
            0.0
        } else {
            s1 * s1 / s2
        }
    }

    /// PARSIMONY — the structural size (unit count), the fitness `λ` term. The smallest morph that solves a task
    /// (fewest units, lowest [`Self::effective_rank`]) generalizes best — the energy economy IS this pressure
    /// (`substrate_parsimony`). A convenience alias for `self.units.len()`, named for the fitness composition.
    #[doc(hidden)]
    pub fn parsimony(&self) -> usize {
        self.units.len()
    }

    /// METABOLIZE by RELIANCE — the STRUCTURAL-selection economy: a feature earns income ∝ [`Self::reliance_of`]
    /// (how much its readers lean on it), all pay upkeep. Compose with [`Self::cull`] and it PRUNES the capacity
    /// the readout does not use: a grown feature the answer ignores (weight ~0) earns ~0, starves, and is
    /// cleanly culled — so an over-provisioned basis SELF-SIZES to what it relies on and `max_hidden` stops
    /// being a fixed-capacity dial. This is the counterpart to `metabolize_contribution` for the OTHER fitness:
    /// residual-`credit` breeds heritable traits (evolution), `reliance` keeps/culls structure (regularization)
    /// — and reliance is the one that survives convergence, which is exactly when pruning must decide.
    #[doc(hidden)]
    pub fn metabolize_reliance(&mut self, cells: &[usize], income: f32, upkeep: f32) {
        self.metabolize_cells(cells, income, upkeep, Fitness::Reliance);
    }

    /// METABOLIZE by SURPRISE-REDUCTION (Gap 7 spike, increment 2) — the DELETION-by-derivation of
    /// `metabolize_reliance`. A feature earns income ∝ [`Self::surprise_reduction_of`] (the END-TO-END ablation
    /// importance `(∂output/∂c)²·Var(c)`, increments 1/1.5) instead of `reliance_of` (Σ|reader weight|). This is the
    /// STABLE-at-convergence, CAUSAL fitness: a feature earns for the surprise it REMOVES from the output, so a
    /// noise / dead-variance / below-a-random-layer feature that `reliance_of` cannot rank (its immediate readers
    /// are random) earns ~0 and is cleanly culled. Requires [`Config::surprise_currency`] (else
    /// `surprise_reduction_of` reads 0 and every cell starves — inert off, like the other gated actors), so it adds
    /// NOTHING to the default path (nothing calls it) ⇒ byte-identical. Compose with [`Self::cull`] exactly as
    /// `metabolize_reliance`. The intent: once validated live, retire `metabolize_reliance` — this is the one
    /// currency the economy keeps.
    #[doc(hidden)]
    pub fn metabolize_surprise(&mut self, cells: &[usize], income: f32, upkeep: f32) {
        self.metabolize_cells(cells, income, upkeep, Fitness::Surprise);
    }

    /// REPOPULATE — close the economy loop: every STARVED unit in `cells` (energy `<= death`) is reborn as
    /// the OFFSPRING of the FITTEST survivor ([`Self::reproduce`]), instead of just dying. Selection (who
    /// starved) + heredity (from the fittest) + variation (reproduce's self-adaptive mutation) run entirely
    /// through the energy account — no external schedule picks winners. A newborn's `BIRTH_ENERGY` reserve
    /// (the "yolk") buys it time to earn before it can starve again; consolidated (plasticity-0) units are
    /// spared. Compose `metabolize_contribution` + `repopulate` each step and the substrate tunes its own
    /// heritable constants by pure competition — decentralized, online, in the energy currency.
    #[doc(hidden)]
    pub fn repopulate(&mut self, cells: &[usize], death: f32) {
        if self.frozen {
            return;
        }
        let energy_of = |s: &Self, c: usize| {
            s.writers
                .get(c)
                .and_then(|w| w.first())
                .map(|&i| s.energy[i])
                .unwrap_or(f32::MIN)
        };
        let fittest = cells.iter().copied().max_by(|&a, &b| {
            energy_of(self, a)
                .partial_cmp(&energy_of(self, b))
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let Some(fittest) = fittest else { return };
        let starved: Vec<usize> = cells
            .iter()
            .copied()
            .filter(|&c| {
                self.writers
                    .get(c)
                    .and_then(|w| w.first())
                    .map(|&i| self.energy[i] <= death && self.plasticity[i] != 0.0)
                    .unwrap_or(false)
            })
            .collect();
        for c in starved {
            if c != fittest {
                self.reproduce(c, fittest);
            }
        }
    }

    /// A grown unit's `(reads, weights, bias)`, for `openmorph-physics`'s bit-for-bit `.to_bits()` comparison of its
    /// engine-backed façade (`EngineField`'s lazy-grow verb) against this `Field`. Read-only; `#[doc(hidden)]`
    /// because it exists for that mirror, not as organism API.
    #[doc(hidden)]
    pub fn unit_params(&self, i: usize) -> (&[usize], &[f32], f32) {
        let u = &self.units[i];
        (&u.reads, &u.weights, u.bias)
    }

    /// The energy of unit `i` (the alive economy account).
    #[doc(hidden)]
    pub fn energy_of(&self, i: usize) -> f32 {
        self.energy.get(i).copied().unwrap_or(0.0)
    }

    /// The organism's SATIATION ∈ `[0,1]` — `metabolism / energy_cap`; its complement is the HUNGER [`Self::live`]
    /// reports. [`Self::should_dream`] gates dreaming on it; [`Self::live`] and a metered [`Self::dream`] drain it.
    /// This reads it for a trace / a tuner.
    #[doc(hidden)]
    pub fn satiation(&self) -> f32 {
        (self.metabolism / self.config.energy_cap.max(1e-6)).clamp(0.0, 1.0)
    }

    /// WHETHER THE ORGANISM IS RESTED ENOUGH TO DREAM — its [`Self::satiation`] has reached the
    /// [`Config::dream_satiation`] floor. Dreaming ([`Self::dream`] — internal thinking) is a LUXURY of the fed: an
    /// organism turns inward to process its experience only while its energy account is above the floor. With the
    /// default floor `0.0` this is always `true` (dreaming ungated). `dream` gates its bouts on this, so a hungry
    /// organism conserves instead of dreaming.
    pub fn should_dream(&self) -> bool {
        self.satiation() >= self.config.dream_satiation
    }

    /// METABOLIZE — the alive economy, one step, event-driven: a unit that FIRED in the last settle
    /// (did useful work) earns `income`; every unit pays `upkeep`. So a unit whose input channel goes
    /// silent stops firing, earns nothing, and STARVES — use-it-or-lose-it. Energy is capped. This is
    /// the retired tick-era organism's energy law, now driven by firings instead of an epoch.
    #[doc(hidden)]
    pub fn metabolize(&mut self, income: f32, upkeep: f32) {
        if self.frozen {
            return;
        }
        for i in 0..self.units.len() {
            let fired = self.last_fired.get(i).copied().unwrap_or(false);
            let raw = self.energy[i] + if fired { income } else { 0.0 } - upkeep;
            self.energy[i] = self.cap_energy(raw);
        }
    }
}
