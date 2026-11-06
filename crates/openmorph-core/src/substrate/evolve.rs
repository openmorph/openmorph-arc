// SPDX-License-Identifier: MIT-0
//! Heritable traits and heredity — trait reads/setters, `resample_feature`, `reproduce`, `fork_mutated`.

use super::*;

impl Field {
    /// RESAMPLE a feature — the alive economy's missing operator, MUTATION, between birth ([`Self::grow_backbone`])
    /// and death ([`Self::cull`]). A random-projection basis is a LOTTERY: a bad draw is dealt once and is
    /// fatal, since only the readout learns, not the projection. Resampling RE-DRAWS the projection of the
    /// unit(s) writing `cell` (`tanh(scale·r·x + b)` with fresh random `r`, `b`) IN PLACE, and ZEROES the
    /// weight every reader placed on `cell` so the readout must RE-EARN the new feature from scratch. Choose
    /// which cell by the substrate's own credit currency (a feature poorly correlated with the residual has
    /// earned nothing — it is starved), and the basis becomes EARNED against error within ONE field: a bad
    /// draw is survived, not fatal — variation + selection at feature granularity, no ensemble. Consolidated
    /// (plasticity-0) writers are left untouched — a banked feature is not re-rolled.
    #[doc(hidden)]
    pub fn resample_feature(&mut self, cell: usize, scale: f32) {
        if self.frozen {
            return;
        }
        if let Some(ids) = self.writers.get(cell).cloned() {
            for i in ids {
                if self.plasticity.get(i).copied().unwrap_or(1.0) == 0.0 {
                    continue; // a consolidated feature is a banked asset — never re-rolled
                }
                let n = self.units[i].reads.len();
                for k in 0..n {
                    self.units[i].weights[k] = self.rand_weight() * scale;
                }
                self.units[i].bias = self.rand_weight() * scale;
                self.energy[i] = self.config.birth_energy;
                self.born_at[i] = self.clock;
                self.dirty.mark(self.units.len(), i);
            }
        }
        // The readers must re-earn the mutated feature: zero the weight each placed on `cell`.
        self.zero_reader_weights_on(cell);
    }

    /// Zero the weight every reader placed on `cell`, so a re-rolled / re-born feature must be RE-EARNED by the
    /// readout (it no longer coasts on the weight the old feature earned). Shared by `resample_feature` (mutation)
    /// and `reproduce` (heredity). Pure (sets weights to 0.0, no rng, order-independent) → byte-replay-safe.
    fn zero_reader_weights_on(&mut self, cell: usize) {
        if let Some(readers) = self.readers.get(cell).cloned() {
            for r in readers {
                let u = &mut self.units[r];
                let mut moved = false;
                for k in 0..u.reads.len() {
                    if u.reads[k] == cell {
                        moved |= u.weights[k] != 0.0;
                        u.weights[k] = 0.0;
                    }
                }
                if moved {
                    self.dirty.mark(self.units.len(), r);
                }
            }
        }
    }

    // ── telemetry / introspection ──
    /// Read a per-unit trait of the first unit writing `cell` (traits are parallel `Vec`s indexed by unit),
    /// falling back to `default` when no unit writes the cell or the trait vector is un-hydrated. The shared
    /// shape behind `scale_of`/`lr_of`/`mut_rate_of`/`credit_of` — the fallback is the live `Config` default,
    /// so a tuned config stays consistent on the edge path.
    fn trait_of(&self, traits: &[f32], cell: usize, default: f32) -> f32 {
        self.writers
            .get(cell)
            .and_then(|w| w.first())
            .map(|&i| traits.get(i).copied().unwrap_or(default))
            .unwrap_or(default)
    }

    /// The projection SCALE (live gain) of the feature writing `cell` — the heritable bandwidth trait.
    #[doc(hidden)]
    pub fn scale_of(&self, cell: usize) -> f32 {
        self.trait_of(&self.scale, cell, self.config.init_scale)
    }

    /// The LEARNING-RATE factor of the unit writing `cell` — the heritable rate trait.
    #[doc(hidden)]
    pub fn lr_of(&self, cell: usize) -> f32 {
        self.trait_of(&self.lr, cell, self.config.init_lr)
    }

    /// The self-adapting MUTATION strength σ of the feature writing `cell`.
    #[doc(hidden)]
    pub fn mut_rate_of(&self, cell: usize) -> f32 {
        self.trait_of(&self.mut_rate, cell, self.config.init_mut_rate)
    }

    /// The 2nd-order damping / MOMENTUM μ of the unit writing `cell` — the heritable stability trait (`0` = the
    /// 1st-order delta rule). INHERITED through `reproduce` under the same self-adaptive σ as `scale`/`lr`. NOTE
    /// (truth-not-trophy): unlike `scale`/`lr`, μ is not (yet) wired into any in-core fitness path
    /// (`credit`/`reliance`/`energy`), so the economy does not itself select ON μ — heredity carries it, but
    /// selection toward critical damping is a follow-on (today shown only by a harness surrogate on held-out error).
    #[doc(hidden)]
    pub fn momentum_of(&self, cell: usize) -> f32 {
        self.trait_of(&self.momentum, cell, self.config.init_momentum)
    }

    /// The CONTRIBUTION credit of the feature writing `cell` — the magnitude of its accumulated correlation
    /// with the residual (the gradient it receives), the fitness to select constants on (a useful-but-quiet
    /// unit earns; a firing gate would not). The make-or-break signal for evolving constants without
    /// selecting on noise.
    #[doc(hidden)]
    pub fn credit_of(&self, cell: usize) -> f32 {
        self.trait_of(&self.credit, cell, 0.0).abs()
    }

    /// SET the projection scale (gain) of every unit writing `cell`. Promotes the once-global projection
    /// constant to a live per-unit trait the substrate can carry, inherit, and tune.
    pub fn set_scale(&mut self, cell: usize, s: f32) {
        self.hydrate_traits();
        if let Some(ids) = self.writers.get(cell).cloned() {
            for i in ids {
                if self.scale[i] != s {
                    self.dirty.mark(self.units.len(), i);
                }
                self.scale[i] = s;
            }
        }
    }

    /// SET the learning-rate factor of every unit writing `cell` — promotes the once-call-site `lr`
    /// constant to a live per-unit trait the substrate can carry, inherit, and tune by competition.
    pub fn set_lr(&mut self, cell: usize, factor: f32) {
        self.hydrate_traits();
        if let Some(ids) = self.writers.get(cell).cloned() {
            for i in ids {
                self.lr[i] = factor;
            }
        }
    }

    /// REPRODUCE — the alive economy's heredity operator: re-draw the feature writing `child` as an
    /// OFFSPRING of the feature writing `parent`. It inherits the parent's projection `scale` under a
    /// SELF-ADAPTIVE mutation — Schwefel's rule σ' = σ·exp(τ·N) is applied FIRST, then the scale mutates by
    /// the NEW σ, so the mutation strength is itself heritable and there is no hand-set mutation constant
    /// (the lineage tunes its own mutability). The child's projection weights are re-drawn (unit-variance —
    /// the scale is the live gain, not baked in), the readers' weight on it is zeroed so the readout
    /// re-earns it, and its credit/energy/age reset. Compose with `credit_of` (select the fittest parent /
    /// the starved child) + `cull` and the substrate tunes its own hot constants by competition (variation,
    /// selection and heredity), decentralized and online, all in the energy currency. `τ` is scale-free (the
    /// only constant, and it is NOT task-tuned).
    pub fn reproduce(&mut self, child: usize, parent: usize) {
        if self.frozen {
            return;
        }
        self.hydrate_traits(); // safe against a freshly-loaded snapshot (see hydrate_traits)
        let cfg = self.config.clone();
        let parent_first = self.writers.get(parent).and_then(|w| w.first()).copied();
        let (pscale, plr, psigma) = match parent_first {
            Some(p) => (self.scale[p], self.lr[p], self.mut_rate[p]),
            None => (cfg.init_scale, cfg.init_lr, cfg.init_mut_rate),
        };
        let pmom = match parent_first {
            Some(p) => self.momentum[p],
            None => cfg.init_momentum,
        };
        // The offspring's input direction is always a FRESH random draw, also when directions are learned
        // (`feature_lr > 0`): the parent LIVES ON (`repopulate` reseeds the starved from the fittest SURVIVOR), so
        // copying its learned direction only duplicates a feature the organism already has. A fresh direction adds
        // a new one, which `feature_lr` then refines. (#126 inherited the parent's direction; on the 1.6 default
        // that collapsed the basis: `substrate_evolve` economy accuracy 67 % with it, 85 % without.)
        // Self-adaptation first (Schwefel), then the genome mutates by the NEW σ: `scale` additively,
        // `lr` multiplicatively (a rate lives on a log scale). Both are inherited from the fittest parent.
        let sigma = (psigma * (cfg.self_adapt_tau * self.rand_weight()).exp())
            .clamp(cfg.sigma_min, cfg.sigma_max);
        let new_scale = (pscale + sigma * self.rand_weight()).clamp(cfg.scale_min, cfg.scale_max);
        let new_lr = (plr * (sigma * self.rand_weight()).exp()).clamp(cfg.lr_min, cfg.lr_max);
        // Momentum μ mutates multiplicatively (0 is absorbing ⇒ a lineage that is OFF stays off, and consumes
        // NO rng draw ⇒ byte-identical when momentum is unused); when seeded >0 it evolves toward critical damping.
        let new_mom = if pmom == 0.0 && cfg.init_momentum == 0.0 {
            0.0
        } else {
            (pmom * (sigma * self.rand_weight()).exp()).clamp(cfg.momentum_min, cfg.momentum_max)
        };
        if let Some(ids) = self.writers.get(child).cloned() {
            for i in ids {
                if self.plasticity[i] == 0.0 {
                    continue; // a consolidated feature is a banked asset — never re-rolled
                }
                let n = self.units[i].reads.len();
                for k in 0..n {
                    self.units[i].weights[k] = self.rand_weight(); // unit-variance; `scale` is the live gain
                }
                self.units[i].bias = self.rand_weight();
                self.scale[i] = new_scale;
                self.lr[i] = new_lr;
                self.mut_rate[i] = sigma;
                self.momentum[i] = new_mom;
                if i < self.velocities.len() {
                    self.velocities[i].clear(); // fresh 2nd-order state for the offspring
                }
                if let Some(st) = self.rls.get_mut(i) {
                    *st = RlsState::default(); // re-rolled weights: the covariance starts over
                }
                self.credit[i] = 0.0;
                self.energy[i] = self.config.birth_energy;
                self.born_at[i] = self.clock;
                self.dirty.mark(self.units.len(), i);
            }
        }
        // The readers must re-earn the offspring: zero the weight each placed on `child`.
        self.zero_reader_weights_on(child);
    }

    /// FORK a WHOLE living organism with variation — the population-scale twin of [`Self::reproduce`] (which
    /// varies one unit): clone the Field, then perturb EVERY plastic unit's heritable genome by that unit's OWN
    /// self-adaptive σ — `scale` additively, `lr` multiplicatively (Schwefel), and the LEARNED input-direction when
    /// directions are heritable (`config.feature_lr != 0`, inherit+refine, never re-drawn). CONSOLIDATED units
    /// (`plasticity == 0`) are carried VERBATIM (a banked asset is never re-rolled), and each varied unit's economy
    /// state (`credit`/`energy`/`born_at`) resets to newborn. Byte-deterministic from the parent's `rng_state`; the
    /// child ends with a distinct advanced state, so a LINEAGE of forks keeps diverging. This is what selection
    /// needs — a population of grown VARIANTS, not identical `.clone()` twins.
    pub fn fork_mutated(&self) -> Field {
        let mut child = self.clone();
        child.hydrate_traits();
        let cfg = child.config.clone();
        for i in 0..child.units.len() {
            if child.plasticity[i] == 0.0 {
                continue; // a consolidated feature is a banked asset — carried verbatim
            }
            // self-adapt σ first (Schwefel), then the genome mutates by the NEW σ.
            let r_sig = child.rand_weight();
            let sigma = (child.mut_rate[i] * (cfg.self_adapt_tau * r_sig).exp())
                .clamp(cfg.sigma_min, cfg.sigma_max);
            let r_sc = child.rand_weight();
            let old_scale = child.scale[i];
            child.scale[i] = (child.scale[i] + sigma * r_sc).clamp(cfg.scale_min, cfg.scale_max);
            if child.scale[i] != old_scale || cfg.feature_lr != 0.0 {
                // the varied unit computes a new function — re-fire it on the child's next settle
                child.dirty.mark(child.units.len(), i);
            }
            let r_lr = child.rand_weight();
            child.lr[i] = (child.lr[i] * (sigma * r_lr).exp()).clamp(cfg.lr_min, cfg.lr_max);
            child.mut_rate[i] = sigma;
            if cfg.feature_lr != 0.0 {
                // directions are heritable → jitter the LEARNED direction (inherit + refine), never re-draw.
                for k in 0..child.units[i].reads.len() {
                    let r = child.rand_weight();
                    child.units[i].weights[k] += sigma * r;
                }
                let rb = child.rand_weight();
                child.units[i].bias += sigma * rb;
            }
            child.credit[i] = 0.0;
            child.energy[i] = cfg.birth_energy;
            child.born_at[i] = child.clock;
        }
        child
    }
}
