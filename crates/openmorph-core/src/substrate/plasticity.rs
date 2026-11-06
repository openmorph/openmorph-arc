// SPDX-License-Identifier: MIT-0
//! Plasticity and mastery — `consolidate`, the reliability/mastery meters, surprise meters,
//! `saturated`/`at_capacity`.

use super::*;

impl Field {
    /// CONSOLIDATE a matured skill — FREEZE the units that write any of `cells` (plasticity → 0), so
    /// later learning cannot degrade them. `learn_step`/`observe` skip a frozen unit's weight update,
    /// so a banked skill survives a stray `learn_step`, a shared router, or a new task touching its
    /// cells — modular isolation guaranteed AT THE SOURCE, not by caller convention. Reversible with
    /// [`Self::set_plasticity`] (e.g. `1.0` to re-open). This is the EXPLICIT, caller-triggered freeze; for the
    /// GROWN, reliability-gated freeze ("consolidation by thriving") see [`Self::observe_reliability`].
    pub fn consolidate(&mut self, cells: &[usize]) {
        for &c in cells {
            if let Some(ids) = self.writers.get(c) {
                for &i in ids {
                    self.plasticity[i] = 0.0;
                }
            }
        }
    }

    /// Set the plasticity (learning rate multiplier in `[0, 1]`) of every unit writing any of `cells` —
    /// `0.0` freezes (consolidate), `1.0` fully plastic, in between = slow, protected learning.
    pub fn set_plasticity(&mut self, cells: &[usize], plasticity: f32) {
        let p = plasticity.clamp(0.0, 1.0);
        for &c in cells {
            if let Some(ids) = self.writers.get(c) {
                for &i in ids {
                    self.plasticity[i] = p;
                }
            }
        }
    }

    /// OBSERVE the RELIABILITY of the skill writing `cell` and let it CRYSTALLIZE by thriving (the crystal half).
    /// The caller feeds a reliability signal `r` (higher = the skill is doing its job well); this updates a per-cell
    /// mastery meter and — when `config.consolidation_rate > 0` — decays the plasticity of every unit writing `cell`
    /// multiplicatively by `1 − rate·mastery`. So a skill freezes ITSELF (plasticity → ~0) once its reliability
    /// reaches a HIGH, STABLE plateau, reaching the frozen state [`Self::consolidate`] sets — but grown, not caller-
    /// triggered. Because the decay is MULTIPLICATIVE it reaches ~0 regardless of the exact mastery plateau, so the
    /// freeze is HARD: zero learning ⇒ zero erosion ⇒ the skill stays reliable ⇒ it STAYS frozen (a DURABLE hold
    /// under any amount of over-training). `mastery` is a relative `height·settled` meter (see [`Self::mastery_of`]):
    /// high only near the signal's OWN peak AND after the fast trace has caught the slow one (its derivative → 0),
    /// so it NEVER fires at a collapse (where the signal has crashed) — which is why the SUBJECT must be the caller's:
    ///
    /// **Feed an action-conditional / reliable-SUCCESS signal (e.g. a WIN), NOT raw prediction error or partial/
    /// cumulative reward.** A collapsed no-change model has LOW error but is worthless; partial reward commits too
    /// early on hard tasks. The core stays domain-blind (it sees only a float); the domain choice lives with the
    /// caller who has the counterfactual knowledge (e.g. `openmorph-perception` feeds controllability = action-
    /// conditionality). The freeze is deliberately ONE-WAY here: a gate that re-opens from its OWN self-derived
    /// reliability death-spirals (leaky-frozen → erode → reliability drops → re-open → collapse). REVERSIBILITY is a
    /// SEPARATE, POLICY-level act — re-open a skill with `set_plasticity(cells, 1.0)` on a sustained LOSING streak
    /// (an external win signal), exactly as MorphLab's proven ARC-3 reference does (a one-way cell freeze + policy-
    /// level re-open). Off + byte-identical when `consolidation_rate == 0` (the meter still updates so `mastery_of`
    /// is readable). NOTE: the consolidation CONCEPT is validated on real ARC-3 (via the FREEZE_STEPS harness
    /// stopgap this replaces); this native primitive is validated by a synthetic falsifier — ARC-3 wiring is a
    /// pending hand-off.
    pub fn observe_reliability(&mut self, cell: usize, r: f32) {
        if cell >= self.mem.medium.len() {
            return;
        }
        if self.reliability.len() <= cell {
            self.reliability.resize(cell + 1, Reliability::default());
        }
        // Two timescales — REUSE the substrate's own EMA constants (no new decay knob). Traces START AT ZERO (a
        // fresh meter is Reliability::default) and BUILD UP, so a cold start reads LOW mastery: the fast trace lags
        // its own peak (height ≪ 1) until the signal has been sustained, and only once fast catches slow (the
        // signal has PLATEAUED) does mastery approach 1. (Seeding fast=slow=peak=r would make the very first sample
        // read mastery 1 — a spurious premature freeze — so we do not.)
        let (fast_dec, slow_dec) = (self.config.surprise_decay, self.config.credit_decay);
        let m = &mut self.reliability[cell];
        m.fast = fast_dec * m.fast + (1.0 - fast_dec) * r;
        m.slow = slow_dec * m.slow + (1.0 - slow_dec) * r;
        // Leaky running peak: jumps up instantly, decays slowly, so height is relative to the recent best.
        m.peak = r.max(slow_dec * m.peak + (1.0 - slow_dec) * r);
        let rate = self.config.consolidation_rate;
        if rate > 0.0 {
            // ONE-WAY mastery freeze: plasticity DECAYS MULTIPLICATIVELY by `1 − rate·mastery`, so a mastered skill
            // reaches a HARD freeze (plasticity → ~0) — and crucially it gets there REGARDLESS of the mastery
            // plateau's exact value (multiplicative → 0, not a floor at `1 − mastery`). A hard freeze means zero
            // learning ⇒ zero erosion ⇒ the skill's reliability stays put ⇒ it STAYS frozen (a durable hold under
            // any amount of over-training). This is deliberately NOT reversible from this same self-derived signal:
            // a leaky, self-reopening gate death-spirals (frozen-leaky → erode → reliability drops → re-open →
            // collapse). Reversibility is a SEPARATE, POLICY-level decision — re-open a skill via `set_plasticity`
            // on a sustained LOSING streak (an external win signal), exactly as MorphLab's proven reference does
            // (a one-way cell freeze + policy-level re-open). `rate` sets how fast the commit completes.
            let m = self.mastery_of(cell);
            if let Some(ids) = self.writers.get(cell).cloned() {
                for i in ids {
                    self.plasticity[i] = (self.plasticity[i] * (1.0 - rate * m)).clamp(0.0, 1.0);
                }
            }
        }
    }

    /// The MASTERY of the skill writing `cell` in `[0, 1]` — how reliably-and-stably it is doing its job, read from
    /// the [`Self::observe_reliability`] meter. `mastery = height · settled`, both RELATIVE (no authored threshold):
    /// `height` = fast-EMA / its own leaky peak (near 1 only when the signal is near its OWN best), `settled` =
    /// `1 − |fast − slow| / peak` (near 1 only when the fast trace has caught the slow one — the reliability has
    /// PLATEAUED, its derivative → 0). Product ⇒ near 1 only at a HIGH, STABLE plateau; 0 for an un-observed cell.
    #[doc(hidden)]
    pub fn mastery_of(&self, cell: usize) -> f32 {
        let Some(m) = self.reliability.get(cell) else {
            return 0.0;
        };
        let denom = m.peak.abs() + 1e-6;
        let height = (m.fast / denom).clamp(0.0, 1.0);
        let settled = 1.0 - ((m.fast - m.slow).abs() / denom).clamp(0.0, 1.0);
        height * settled
    }

    /// The PLASTICITY (learn-rate multiplier in `[0, 1]`) of the first unit writing `cell` — `1.0` fully plastic,
    /// `0.0` frozen (consolidated). Read-only view of what `consolidate`/`set_plasticity`/`observe_reliability` set.
    #[doc(hidden)]
    pub fn plasticity_of(&self, cell: usize) -> f32 {
        self.writers
            .get(cell)
            .and_then(|w| w.first())
            .and_then(|&i| self.plasticity.get(i).copied())
            .unwrap_or(1.0)
    }

    /// The organism's OWN grown SURPRISE at `cell` — the per-cell prediction-error EMA. The signal the
    /// harness used to RE-DERIVE by hand; now read it here so triggers key on the morph's own error, not a
    /// recomputation. `0.0` for an unwritten/out-of-range cell.
    #[doc(hidden)]
    pub fn surprise_of(&self, cell: usize) -> f32 {
        self.surprise.get(cell).copied().unwrap_or(0.0)
    }

    /// Mean grown surprise over a set of head cells — the aggregate error the trigger reads.
    pub fn mean_surprise(&self, heads: &[usize]) -> f32 {
        if heads.is_empty() {
            return 0.0;
        }
        heads.iter().map(|&c| self.surprise_of(c)).sum::<f32>() / heads.len() as f32
    }

    /// SIGNED surprise TREND at `cell`, RELATIVE to the cell's own history: `(slow − fast) / (slow + ε)`, where
    /// `slow` is a lagging surprise trace (decay `credit_decay`) and `fast` is `surprise` (decay `surprise_decay`).
    /// POSITIVE ⇒ surprise is DROPPING (learning); NEGATIVE ⇒ RISING (the warmup / a new surprise); ≈0 ⇒ PLATEAUED.
    /// The sign is what separates "warmup" from "plateau" (both have low progress) — see `saturated`.
    fn surprise_trend(&self, cell: usize) -> f32 {
        let fast = self.surprise_of(cell);
        let slow = self.surprise_slow.get(cell).copied().unwrap_or(0.0);
        if slow <= 1e-6 {
            return 0.0; // nothing observed here yet
        }
        (slow - fast) / (slow + 1e-6)
    }

    /// SURPRISE-DRIVEN PLASTICITY controller for a learned `cell` (opt-in `Config::plasticity_anneal`). Maintains a
    /// two-timescale error trace at the cell — fast (`surprise_decay`) and slow (`credit_decay`) EMAs of `|err|` —
    /// and returns the learning GAIN the delta step is scaled by: it DECAYS by `plasticity_anneal` each event (so
    /// the readout SETTLES on a stationary — even noisy — target), and RESETS to 1.0 the moment the fast error
    /// breaks above `plasticity_reset_ratio · slow` (the target CHANGED), re-baselining `slow = fast` so the same
    /// break does not re-trigger. The LEVEL of error carries no signal — only a CHANGE does. Not called when off.
    pub(super) fn update_plasticity_gain(&mut self, cell: usize, err_abs: f32) -> f32 {
        let n = self.mem.medium.len();
        if self.learn_gain.len() < n {
            self.err_fast.resize(n, 0.0);
            self.err_slow.resize(n, 0.0);
            self.learn_gain.resize(n, 1.0);
        }
        if cell >= n {
            return 1.0;
        }
        let (fast_dec, slow_dec) = (self.config.surprise_decay, self.config.credit_decay);
        self.err_fast[cell] = fast_dec * self.err_fast[cell] + (1.0 - fast_dec) * err_abs;
        self.err_slow[cell] = slow_dec * self.err_slow[cell] + (1.0 - slow_dec) * err_abs;
        let broke = self.err_slow[cell] > 1e-9
            && self.err_fast[cell] > self.config.plasticity_reset_ratio * self.err_slow[cell];
        if broke {
            self.learn_gain[cell] = 1.0; // the target changed — re-open plasticity to base
            self.err_slow[cell] = self.err_fast[cell]; // re-baseline so the same break does not re-trigger
        } else {
            self.learn_gain[cell] *= 1.0 - self.config.plasticity_anneal; // anneal — settle on a stationary target
        }
        self.learn_gain[cell]
    }

    /// The current surprise-driven plasticity GAIN at `cell` (opt-in `Config::plasticity_anneal`) — 1.0 = fully
    /// plastic (just reset / never annealed), → 0 as a stationary target settles. Observability only; 1.0 when the
    /// feature is off or the cell has never been learned.
    pub fn plasticity_gain_of(&self, cell: usize) -> f32 {
        self.learn_gain.get(cell).copied().unwrap_or(1.0)
    }

    /// LEARNING PROGRESS at `cell` in `[0, 1]` — how much surprise is STILL being REDUCED, RELATIVE to
    /// the cell's OWN recent history (no authored threshold): the positive part of `Self::surprise_trend`. HIGH
    /// while error is dropping (learning), → 0 once surprise has PLATEAUED (dSurprise/dt ≈ 0 — learned all it can
    /// here) AND also 0 while surprise is RISING (warmup — use `saturated` to tell those two 0s apart).
    #[doc(hidden)]
    pub fn learning_progress(&self, cell: usize) -> f32 {
        self.surprise_trend(cell).max(0.0).clamp(0.0, 1.0)
    }

    /// SATURATED — the grown plateau/saturation predicate: the mean surprise trend over `heads` is FLAT
    /// (`|trend| < plateau`) — surprise is neither still dropping (learning) NOR rising (a new surprise), RELATIVE
    /// to each cell's own history. Replaces the harness's authored `ERR_TRIGGER = 0.45` / `agency > 0.008` magic
    /// constants: the trigger keys on the morph's OWN plateau, and `plateau` is an explicit RELATIVE fraction
    /// (meta-evolvable), not a hidden per-signal magnitude. `false` before learning (nothing observed) and during
    /// the warmup RISE (trend negative), TRUE once error has settled. Using `|trend|` (not just low progress) is
    /// what stops a false positive during the initial rise. NOTE: the capacity-ceiling half of R5
    /// (`n_hidden == max_hidden`) is a follow-on — the plateau is the load-bearing grown signal here.
    pub fn saturated(&self, heads: &[usize], plateau: f32) -> bool {
        if heads.is_empty() {
            return false;
        }
        // Require SOME learning to have happened (slow trace above the floor) so a never-trained head is not
        // "saturated". `saturation_floor` interacts with signal scale (a low-magnitude-surprise task's trace may
        // sit below it), so it is a tunable, not a pure epsilon.
        let observed = heads.iter().any(|&c| {
            self.surprise_slow.get(c).copied().unwrap_or(0.0) > self.config.saturation_floor
        });
        let mean_abs_trend = heads
            .iter()
            .map(|&c| self.surprise_trend(c).abs())
            .sum::<f32>()
            / heads.len() as f32;
        observed && mean_abs_trend < plateau
    }

    /// The CAPACITY-CEILING half of the saturation trigger (the follow-on [`Self::saturated`] flagged):
    /// every `head`'s writer has grown its FULL hidden allotment (`n_hidden == max_hidden`), so surprise can no
    /// longer be reduced by GROWING here — only structural change (recruit a module) can. Compose it with
    /// `saturated` (the plateau) for the full "solved vs stuck-at-capacity" predicate the harness hand-thresholded
    /// (`0.75·MAXH`). GROWN: it reads the substrate's own hidden count against `Config::max_hidden`; a hidden read
    /// is one whose cell is itself written by a unit (a grown feature) vs a raw external input. A head with no
    /// writer, or `max_hidden == 0`, counts as at-capacity. Read-only, so byte-identical when unused.
    pub fn at_capacity(&self, heads: &[usize]) -> bool {
        if heads.is_empty() {
            return false;
        }
        let max_hidden = self.config.max_hidden;
        heads.iter().all(|&c| {
            let Some(&u) = self.writers.get(c).and_then(|w| w.first()) else {
                return true;
            };
            let n_hidden = self.units[u]
                .reads
                .iter()
                .filter(|&&r| self.writers.get(r).is_some_and(|w| !w.is_empty()))
                .count();
            n_hidden >= max_hidden
        })
    }
}
