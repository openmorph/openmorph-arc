// SPDX-License-Identifier: MIT-0
//! The deterministic PRNG streams (growth and act-sampling) and their seeding.

use super::*;

impl Field {
    /// Seed the deterministic growth PRNG — reproducible but varied grown features across runs. Also re-derives
    /// the act-sampling PRNG (`Self::act_rng`, decorrelated by a golden-ratio mix) so a seeded organism's
    /// tie-breaks (`alive(Event::Act)` under the valence rule, [`Self::select_explore`]) are reproducible and vary
    /// with the seed.
    pub fn rng_seed(&mut self, seed: u64) {
        self.rng_state = seed | 1;
        self.act_rng = (seed | 1).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xAC7_5EED;
    }

    /// A deterministic act-sampling draw in `[0, 1)` (byte-reproducible; advances [`Self::act_rng`]). Used ONLY to
    /// break an exact tie among the best options (the valence rule of `alive(Event::Act)`, [`Self::select_explore`]).
    pub(super) fn act_rand(&mut self) -> f32 {
        self.act_rng = self
            .act_rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.act_rng >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Seed a small random PRIOR onto the weights + bias of the units writing cells `base..base+n`, drawn from
    /// the growth PRNG (deterministic per seed). For an ENSEMBLE of independently-seeded organisms this makes the
    /// members start DIVERGED (varied priors): a heavily-trained input region converges the members (low output
    /// disagreement), while a rarely-updated one keeps its prior spread — the observation-count uncertainty that
    /// value-of-information exploration reads. A lone organism never calls this and is byte-unchanged; when called,
    /// it advances the growth PRNG exactly `Σ(weights+1)` times, so replay stays deterministic.
    pub fn seed_writer_priors(&mut self, base: usize, n: usize, scale: f32) {
        for c in base..base + n {
            let Some(&u) = self.writers.get(c).and_then(|w| w.first()) else {
                continue;
            };
            let len = self.units[u].weights.len();
            for k in 0..len {
                let r = self.rand_weight();
                self.units[u].weights[k] += scale * r;
            }
            let rb = self.rand_weight();
            self.units[u].bias += scale * rb;
            if scale != 0.0 {
                self.dirty.mark(self.units.len(), u);
            }
        }
    }

    /// A deterministic weight in `[-1, 1]` for a grown hidden feature (byte-reproducible).
    pub(super) fn rand_weight(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// A deterministic random cell index (byte-reproducible) — for the concept-metric's negative sampling.
    pub(super) fn rand_cell(&mut self) -> usize {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as usize) % self.mem.medium.len().max(1)
    }

    /// A deterministic uniform `f32` in `[0, 1)` (byte-reproducible) — for the spontaneous drift in [`Self::think`].
    pub(super) fn rand_u01(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng_state >> 33) as f32 / (1u64 << 31) as f32
    }
}
