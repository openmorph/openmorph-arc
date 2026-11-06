// SPDX-License-Identifier: MIT-0
//! The two acting rules behind `alive(Event::Act)` (valence, free-energy) and the selectors
//! (`select`, `select_explore`, `arbitrate`).

use super::*;

/// The population std-dev of an actor's decision scores across the offered options — the `g_spread` diagnostic
/// (`0` for fewer than two options). Near-zero ⇒ the decision was degenerate (every option tied).
fn score_spread(scores: &[f32]) -> f32 {
    if scores.len() < 2 {
        0.0
    } else {
        let n = scores.len() as f32;
        let mean = scores.iter().sum::<f32>() / n;
        (scores.iter().map(|g| (g - mean) * (g - mean)).sum::<f32>() / n).sqrt()
    }
}

impl Field {
    /// The VALENCE rule of `alive(Event::Act)` (the default, [`ActRule::Valence`]). Fold the last reward toward
    /// `r + act_discount·max (μ + σ²)(options)` into its option's [`ValenceRecord`] (and count the visit), then
    /// ARGMAX `A(o) = −value_of(o) + μ_o + σ_o²` with `σ_o² = (σ0·(1−lr)^visits)²` — exact ties broken uniformly by the act
    /// PRNG. Returns the choice before `action_dwell`.
    pub(super) fn act_valence(&mut self, options: &[usize], last: Option<(usize, f32)>) -> usize {
        self.fold_last(options, last, ActRule::Valence);
        let scores: Vec<f32> = options.iter().map(|&o| self.valence_score(o)).collect();
        self.last_g_spread = Some(score_spread(&scores));
        self.argmax_seeded_ties(options, &scores)
    }

    /// The FREE-ENERGY rule of `alive(Event::Act)` ([`ActRule::FreeEnergy`]). The same fold as
    /// [`Self::act_valence`] but into the option's value ENSEMBLE, then ARGMIN `G = value + (μ−s*)² − σ²` with ties
    /// to the FIRST option (the pre-1.6 read-off, unchanged, so a free-energy organism — MorphBot's `Player` — is
    /// bit-identical). Returns the choice before `action_dwell`.
    pub(super) fn act_free_energy(
        &mut self,
        options: &[usize],
        last: Option<(usize, f32)>,
    ) -> usize {
        self.fold_last(options, last, ActRule::FreeEnergy);
        for &o in options {
            self.ensure_ensemble(o);
        }
        // DIAGNOSTIC (byte-identical): the spread of `G` across the offered options, so a client can read
        // `g_spread()` and catch a DEGENERATE decision (all options tied). Pure reads; no state change.
        let gs: Vec<f32> = options.iter().map(|&o| self.free_energy_score(o)).collect();
        self.last_g_spread = Some(score_spread(&gs));
        options
            .iter()
            .zip(&gs)
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&o, _)| o)
            .unwrap_or(0)
    }

    /// Fold `last = (o, r)` into option `o`'s learned value — its [`ValenceRecord`] under `rule =`
    /// [`ActRule::Valence`], its value ensemble under [`ActRule::FreeEnergy`]: the goal-completion self-reward is
    /// added, the target is bootstrapped by `act_discount` from the options THIS call offers (the successors of `o`)
    /// through the same rule's mean, and a reached goal is consumed AFTER the credit. `act_discount = 0` folds `r`
    /// itself (no bootstrap read) — for FreeEnergy byte-identical to the pre-1.6 fold.
    fn fold_last(&mut self, options: &[usize], last: Option<(usize, f32)>, rule: ActRule) {
        if let Some((o, r)) = last {
            // GOAL-COMPLETION REWARD (#155): a self-set goal is credited through the SAME channel the actor
            // folds external reward into. Byte-identical when goal_reward = 0.
            let (r, reached) = self.goal_reward_for(o, r);
            let valence = rule == ActRule::Valence;
            let gamma = self.config.act_discount;
            let target = if gamma == 0.0 {
                r
            } else {
                // the successor's value: the best score among the options offered now, read BEFORE the fold so
                // `o` bootstraps from its pre-update estimate. Valence bootstraps its full `μ + σ²`: an unexplored
                // successor's bonus flows back up the path that leads to it (optimism carried by the bootstrap),
                // so a deep unexplored branch is not abandoned for a shallow sure reward. FreeEnergy reads `μ`.
                let best = options
                    .iter()
                    .map(|&p| {
                        if valence {
                            self.valence_of(p)
                        } else {
                            self.option_mean_var(p).0
                        }
                    })
                    .fold(f32::NEG_INFINITY, f32::max);
                if best.is_finite() {
                    r + gamma * best
                } else {
                    r
                }
            };
            if valence {
                self.fold_valence(o, target);
            } else {
                self.fold_ensemble(o, target);
            }
            if reached {
                self.consume_goal(o);
            }
        }
    }

    /// ARGMAX over `scores` (parallel to `options`) whose EXACT ties among the best score are broken UNIFORMLY by
    /// the act PRNG ([`Self::act_rand`]) — deterministic and replayable, and with no positional bias (a plain
    /// `max_by` returns the last tied option, `min_by` the first). A single best option draws nothing. The same
    /// read-off [`Self::select_explore`] uses.
    fn argmax_seeded_ties(&mut self, options: &[usize], scores: &[f32]) -> usize {
        let Some(best) = scores.iter().copied().max_by(|a, b| a.total_cmp(b)) else {
            return 0;
        };
        let winners: Vec<usize> = options
            .iter()
            .zip(scores)
            .filter(|&(_, &s)| s == best || s.total_cmp(&best).is_eq())
            .map(|(&o, _)| o)
            .collect();
        if winners.len() <= 1 {
            winners.first().copied().unwrap_or(0)
        } else {
            let r = self.act_rand();
            let k = ((r * winners.len() as f32) as usize).min(winners.len() - 1);
            winners[k]
        }
    }

    /// GOAL-COMPLETION REWARD (#155, `Config::goal_reward`) — the AUGMENTED reward for `last` landing on option
    /// `o`: if `o` is a SELF-SET goal (a clamped preferred cell the organism wanted activated) and the reward is
    /// enabled, add `goal_reward` so the reaching action is credited through whatever channel the active actor uses
    /// for external reward. Pure read (no state change) — the caller CONSUMES the goal via [`Field::consume_goal`]
    /// AFTER crediting, so each actor path (valence / free-energy) keeps its own reward-then-consume order.
    /// `goal_reward = 0` (default) ⇒ returns `r` unchanged, `reached = false` — byte-identical.
    fn goal_reward_for(&self, o: usize, r: f32) -> (f32, bool) {
        let reached =
            self.config.goal_reward != 0.0 && self.clamped.get(o).copied().unwrap_or(false);
        if reached {
            (r + self.config.goal_reward, true)
        } else {
            (r, false)
        }
    }

    /// CONSUME a reached self-set goal: unclamp it (so `orient` can set the next) and bump the `goals_reached`
    /// observability counter. Called by every actor path ONLY after a reached goal's reward has been credited, so
    /// the reward → consume order is identical across paths. `o` is always in bounds (a `reached` goal was clamped).
    fn consume_goal(&mut self, o: usize) {
        self.clamped[o] = false;
        self.goals_reached = self.goals_reached.wrapping_add(1);
    }

    /// ACTION DWELL — wrap a freshly-decided action with a GROWN commitment duration (opt-in
    /// `Config::action_dwell`). Learns each option's dwell from reward (earning ⇒ hold longer, not ⇒ release),
    /// and while a commitment is live AND still earning, HOLDS it instead of re-deciding. `dwell_len` starts at 1
    /// (re-decide every step ≡ off), so the persistence is grown, not authored.
    pub(super) fn apply_dwell(
        &mut self,
        fresh: usize,
        options: &[usize],
        last: Option<(usize, f32)>,
    ) -> usize {
        self.learn_dwell(last);
        // HOLD the current commitment while it is still earning (reward ≥ 0 folds through zero-reward steps toward
        // a delayed payoff) and still an available option; a penalty (r < 0) releases it.
        if let Some(c) = self.dwell_committed {
            let still_earning = matches!(last, Some((o, r)) if o == c && r >= 0.0);
            if self.dwell_remaining > 1.0 && still_earning && options.contains(&c) {
                self.dwell_remaining -= 1.0;
                return c;
            }
        }
        // Otherwise take the fresh decision and COMMIT for its learned duration (this step counts as the first).
        self.dwell_committed = Some(fresh);
        self.dwell_remaining = self.dwell_len.get(fresh).copied().unwrap_or(1.0);
        fresh
    }

    /// The EPISODE END of `alive(Event::Act)` (an empty `options`): fold `last` into the active rule's own record
    /// with no successor to bootstrap from, and nothing else. No option is scored or chosen, so the act PRNG and
    /// `g_spread` are untouched; with `action_dwell`, the last reward still teaches its dwell and the commitment is
    /// released (the next episode starts undecided).
    pub(super) fn end_episode(&mut self, last: Option<(usize, f32)>) {
        self.fold_last(&[], last, self.config.effective_act_rule());
        if self.config.action_dwell {
            self.learn_dwell(last);
            self.dwell_committed = None;
            self.dwell_remaining = 0.0;
        }
    }

    /// Move the last action's learned dwell with its reward sign (the LEARN half of [`Self::apply_dwell`]).
    fn learn_dwell(&mut self, last: Option<(usize, f32)>) {
        if self.dwell_len.len() < self.mem.medium.len() {
            self.dwell_len.resize(self.mem.medium.len(), 1.0);
        }
        // LEARN: the last action's dwell moves with its reward sign — a repeat that EARNS is worth holding. Use a
        // three-way sign, NOT `f32::signum`: `signum(+0.0) == 1.0`, so a sparse-reward (all-zero) stream would
        // ratchet the dwell up to `dwell_max` every step and perseverate on an action that never earned. Zero must
        // be neutral (nothing earned ⇒ no change); a penalty (r < 0) still shrinks it.
        if let Some((o, r)) = last {
            if let Some(d) = self.dwell_len.get_mut(o) {
                let sign = if r > 0.0 {
                    1.0
                } else if r < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                *d = (*d + self.config.dwell_rate * sign).clamp(1.0, self.config.dwell_max);
            }
        }
    }

    /// A6 — lazily create option `o`'s free-energy value ENSEMBLE: `Config::free_energy_ensemble` members with mean
    /// 0 (uninformative) and standard deviation `Config::free_energy_prior_std` (the epistemic prior σ0). The
    /// members are a symmetric linspace ([`Self::prior_members`]), so every never-sampled option is EXCHANGEABLE
    /// (identical μ and σ²) — the epistemic/pragmatic prior terms then CANCEL in the argmin over any value field,
    /// so a planner still descends its cost-to-go. Deterministic (no RNG — the prior is the same on every seed).
    /// The mean is 0 only up to f32 rounding (−1.99e-8 at 24 members); the FreeEnergy rule keeps that residue bit
    /// for bit, the Valence rule does not use the ensemble ([`ValenceRecord`]).
    fn ensure_ensemble(&mut self, o: usize) {
        if self.value_ensemble.contains_key(&o) {
            return;
        }
        let members = self.prior_members(self.config.free_energy_ensemble);
        self.value_ensemble.insert(o, members);
    }

    /// The prior value ensemble of `n` members (at least 2): an evenly spaced symmetric linspace on `[−L, L]` whose
    /// standard deviation is `Config::free_energy_prior_std`.
    fn prior_members(&self, n: usize) -> Vec<f32> {
        let n = n.max(2);
        let nf = n as f32;
        // variance of n evenly-spaced symmetric points on [−L, L] is L²(n+1)/(3(n−1)); solve for L so std = σ0.
        let l = self.config.free_energy_prior_std * (3.0 * (nf - 1.0) / (nf + 1.0)).sqrt();
        (0..n)
            .map(|j| -l + 2.0 * l * (j as f32) / (nf - 1.0))
            .collect()
    }

    /// Fold the target `r` into option `o`'s [`ValenceRecord`] (the Valence rule): `μ ← μ + lr·(r − μ)` with
    /// `lr = Config::free_energy_lr`, and one more visit. A never-visited option starts from `(0, 0)` — or from the
    /// record derived from its value ensemble when it has one ([`Self::valence_record`]).
    fn fold_valence(&mut self, o: usize, r: f32) {
        let lr = self.config.free_energy_lr;
        let mut rec = self.valence_record(o);
        rec.mean += lr * (r - rec.mean);
        rec.visits = rec.visits.saturating_add(1);
        self.valence.insert(o, rec);
    }

    /// Option `o`'s [`ValenceRecord`]: its own; else, when it has a value ENSEMBLE (an organism that ran the
    /// FreeEnergy rule, or a pre-1.6 `valence_actor` snapshot, which kept its valence there), the record DERIVED
    /// from it; else the prior `(μ 0, 0 visits)`.
    ///
    /// The derivation (the snapshot migration). Every ensemble member takes the same step, so the ensemble is the
    /// prior linspace scaled by `s = (1−lr)^n` plus the common EMA of the fold targets: its RANGE (max − min, blind
    /// to the common shift) is the prior's range times `s`, so `n = ln s / ln|1−lr|` (rounded; no range left ⇒
    /// saturated), and `μ` is the ensemble mean. A mean the ensemble cannot resolve — within its own f32 rounding
    /// (`members·ε·√(n+1)·range`), or subnormal — is the prior's f32 residue plus rounding, i.e. a zero-reward
    /// history, and reads as exactly 0 (carried into the record it would outweigh σ² after enough further visits).
    /// Equal to the ensemble's own `(μ, σ²)` up to f32 rounding, not bit for bit.
    fn valence_record(&self, o: usize) -> ValenceRecord {
        if let Some(rec) = self.valence.get(&o) {
            return *rec;
        }
        let Some(members) = self.value_ensemble.get(&o) else {
            return ValenceRecord::default();
        };
        let a = crate::grouping::aggregate(members);
        let prior = crate::grouping::aggregate(&self.prior_members(members.len()));
        let prior_range = prior.max - prior.min;
        let range = a.max - a.min;
        let s = if prior_range > 0.0 {
            range / prior_range
        } else {
            0.0
        };
        let q = (1.0 - self.config.free_energy_lr).abs();
        let visits = if s <= 0.0 {
            u32::MAX // no range left: converged past what f32 can count
        } else if q > 0.0 && q != 1.0 {
            (s.ln() / q.ln()).round().max(0.0) as u32
        } else {
            0 // lr = 0 never anneals (and lr = 1 leaves no range after one visit): nothing to count
        };
        let resolution =
            members.len() as f32 * f32::EPSILON * (visits.min(1 << 20) as f32 + 1.0).sqrt() * range;
        let mean = if a.mean.abs() <= resolution || a.mean.abs() < f32::MIN_POSITIVE {
            0.0
        } else {
            a.mean
        };
        ValenceRecord { mean, visits }
    }

    /// The Valence rule's exploration bonus for an option visited `visits` times: `σ² = (σ0·(1−lr)^visits)²` with
    /// `σ0 = Config::free_energy_prior_std`, `lr = Config::free_energy_lr` — `σ0²` never visited, then shrinking
    /// with every visit and never growing (non-increasing in the count; exactly 0 once it underflows f32, ~320
    /// visits at the defaults, after which equal-mean options tie and the seeded tie-break spreads the choices).
    fn visit_bonus(&self, visits: u32) -> f32 {
        let decay = (1.0 - self.config.free_energy_lr).powi(visits.min(i32::MAX as u32) as i32);
        let sigma = self.config.free_energy_prior_std * decay;
        sigma * sigma
    }

    /// A6 — fold the target `r` into option `o`'s ensemble: every member steps toward `r` by
    /// `Config::free_energy_lr` (`m ← m + lr·(r − m)`), so the members CONVERGE on the target and the option's
    /// σ² self-anneals to 0 (exploit once known). The grown twin of `reinforce` for the free-energy rule.
    fn fold_ensemble(&mut self, o: usize, r: f32) {
        self.ensure_ensemble(o);
        let lr = self.config.free_energy_lr;
        if let Some(members) = self.value_ensemble.get_mut(&o) {
            for m in members.iter_mut() {
                *m += lr * (r - *m);
            }
        }
    }

    /// A6 — the EXPECTED-FREE-ENERGY score of option `o` (ARGMIN picks it): `G = value_of(o) + (μ−s*)² − σ²`.
    /// `μ`/`σ²` are the mean and VARIANCE of `o`'s grown ensemble (via the substrate's own
    /// [`crate::grouping::aggregate`] — `spread` is the std, squared here to a variance), `value_of(o)` (cost-to-go,
    /// `f32::MAX` ⇒ 0) is the planner's pragmatic term. PRAGMATIC (reach the preferred `s*` / descend the value
    /// field) minus EPISTEMIC (the ensemble disagreement) — same currency, coefficient ONE, no β / no novelty flag.
    fn free_energy_score(&self, o: usize) -> f32 {
        let v = self.value_of(o);
        let value = if v == f32::MAX { 0.0 } else { v };
        let sstar = self.config.free_energy_sstar;
        let (mean, var) = self.option_mean_var(o);
        value + (mean - sstar) * (mean - sstar) - var
    }

    /// The `(mean, variance)` of option `o`'s grown value ensemble — the read behind [`Self::free_energy_score`]
    /// and the FreeEnergy rule's `act_discount` bootstrap (a never-sampled option keeps the prior `(0, σ0²)` from
    /// `Config::free_energy_prior_std`). Uses the substrate's own [`crate::grouping::aggregate`], so `var = spread²`.
    fn option_mean_var(&self, o: usize) -> (f32, f32) {
        match self.value_ensemble.get(&o) {
            Some(members) => {
                let a = crate::grouping::aggregate(members);
                (a.mean, a.spread * a.spread)
            }
            None => {
                let s0 = self.config.free_energy_prior_std;
                (0.0, s0 * s0)
            }
        }
    }

    /// The GROWN SIGNED VALENCE of option `o` — the signed, self-annealing currency that replaced the removed
    /// `act_score`'s binary novelty and `drive` — the score of the default [`crate::config::ActRule::Valence`] rule.
    /// It reads option `o`'s learned `(μ, visits)` record and returns `μ_o + σ_o²`: the **signed** reward mean `μ_o`
    /// (negative = AVERSION, positive = COMMITMENT — the valence sign) plus the **self-annealing** exploration bonus
    /// `σ_o² = (σ0·(1−lr)^visits)²` (the exploratory face — the prior σ0² for a never-visited option, shrinking with
    /// every visit). A never-visited option reads `(0, σ0²)` (uninformative mean, full bonus), so an option that
    /// paid ZERO is not "untried": its visits already lowered its bonus. Unlike the free-energy `G`, the mean enters
    /// LINEARLY (a soft hill a value gradient can override), not squared into a barrier — that is the whole point of
    /// the valence face of `V`. On an organism that ran the FreeEnergy rule it reads the record derived from the
    /// option's ensemble (≈ that ensemble's `μ + σ²`). Read-only; changes nothing.
    #[doc(hidden)]
    pub fn valence_of(&self, o: usize) -> f32 {
        let rec = self.valence_record(o);
        rec.mean + self.visit_bonus(rec.visits)
    }

    /// DECISION-DEGENERACY DIAGNOSTIC — the spread (population std-dev) of the active rule's decision score (the
    /// valence `A` or the free-energy `G`) across the options offered to the LAST `alive(Event::Act)`. **Near-zero
    /// means the decision was degenerate**: every option scored the same, so selection was effectively a tie-break —
    /// a silent failure no other surface reports (a flat reward that never varies, a collapsed value head, or a
    /// state key so coarse every option maps to one cell all present identically to the caller). Read it in an agent's smoke test and assert `g_spread() > eps` to catch "your
    /// options are tied" on day one instead of after months of chasing a pull-count. `NaN` until the first act. A
    /// pure read of a diagnostic recorded during `Act`; it never influences selection.
    pub fn g_spread(&self) -> f32 {
        self.last_g_spread.unwrap_or(f32::NAN)
    }

    /// The VALENCE-ACTOR score of option `o` (ARGMAX picks it): `A(o) = −value_of(o) + valence_of(o)`. The value is
    /// negated (low cost-to-go = good) and the signed valence competes ADDITIVELY with it — so a strong value gradient
    /// can override a learned aversion (`μ<0`) instead of being blocked by it.
    fn valence_score(&self, o: usize) -> f32 {
        let v = self.value_of(o);
        let value = if v == f32::MAX { 0.0 } else { -v };
        value + self.valence_of(o)
    }

    /// SELECT — the winning module: the index of the highest `gates` cell (a competitive/winner-take-all
    /// read over the modules' recognition / confidence / value on the current context). The gates are grown
    /// units, so the routing itself is learned — a gate is just more cells.
    ///
    /// TIES go to the LAST of the tied gates (`Iterator::max_by` keeps the last maximum), so an all-tied read
    /// returns `gates.len() − 1`; [`Self::orient`], [`Self::orient_many`] and [`Self::arbitrate`] inherit this.
    /// ([`Self::select_explore`] breaks ties uniformly instead.) A gate cell outside the medium reads as `f32::MIN`.
    /// Empty `gates` returns `0` (debug-asserted: `0` is a real index, so the fallback is otherwise silent).
    pub fn select(&self, gates: &[usize]) -> usize {
        debug_assert!(
            !gates.is_empty(),
            "select called with no gates — falls back to cell 0"
        );
        gates
            .iter()
            .enumerate()
            .max_by(|&(_, &a), &(_, &b)| {
                let (va, vb) = (
                    self.mem.medium.get(a).copied().unwrap_or(f32::MIN),
                    self.mem.medium.get(b).copied().unwrap_or(f32::MIN),
                );
                va.partial_cmp(&vb).unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// SELECT WITH VALUE-OF-INFORMATION — like [`Self::select`], but each option is an ENSEMBLE of grown
    /// value-estimator cells (varied priors), scored by `mean + beta·spread` of the ensemble: its expected
    /// value PLUS its epistemic UNCERTAINTY — the estimators' DISAGREEMENT ([`crate::grouping::Aggregate::spread`]).
    /// So the organism acts where its VALUE estimate is most uncertain and commits as the estimators converge on
    /// the reward — the value-of-information (UCB) explore/exploit driver: "don't commit while information is
    /// still valuable." `beta` weights exploration (`0` ⇒ pure exploit = argmax mean; larger ⇒ optimism under
    /// uncertainty). Distinct from `observe`'s surprise (which measures WORLD unpredictability): here the
    /// uncertainty is in the organism's OWN value estimate — the intrinsic signal that steers acting toward what
    /// it would most learn to do (the one exploration bonus measured to actually help, vs count/empowerment/
    /// world-surprise bonuses that hurt). Empty ⇒ `0` (debug-asserted — `0` is a real index).
    ///
    /// TIE-BREAK: among options with the *same* top score it picks UNIFORMLY at random via the deterministic
    /// act-sampling PRNG (`act_rand`, byte-replayable). This is why the method takes `&mut self` — an exploration
    /// chooser must NOT deterministically favour one index (the old `Iterator::max_by` returned the *last* of
    /// equal maxima, silently steering every all-tied caller to the last option). When exactly one option has the
    /// top score — the normal case for genuinely disagreeing ensembles — NO draw is taken and the PRNG is
    /// untouched, so a distinct-score decision is byte-identical to before.
    pub fn select_explore(&mut self, options: &[&[usize]], beta: f32) -> usize {
        debug_assert!(
            !options.is_empty(),
            "select_explore: no options — falls back to index 0"
        );
        if options.is_empty() {
            return 0;
        }
        // Score every option first (immutable read of the medium), so the tie-break draw below is the only
        // mutation. Score = ensemble mean + beta·disagreement, exactly as before.
        let scores: Vec<f32> = options
            .iter()
            .map(|cells| {
                let vals: Vec<f32> = cells
                    .iter()
                    .map(|&c| self.mem.medium.get(c).copied().unwrap_or(0.0))
                    .collect();
                let a = crate::grouping::aggregate(&vals);
                a.mean + beta * a.spread
            })
            .collect();
        let mut best = f32::NEG_INFINITY;
        for &s in &scores {
            if s > best {
                best = s;
            }
        }
        // The argmax SET (exact ties on the top score). One winner ⇒ deterministic, no PRNG draw (byte-identical
        // to the historical single-max pick). Several ⇒ break the tie uniformly with the act PRNG.
        let winners: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s == best)
            .map(|(i, _)| i)
            .collect();
        if winners.len() <= 1 {
            winners.first().copied().unwrap_or(0)
        } else {
            let r = self.act_rand();
            let k = ((r * winners.len() as f32) as usize).min(winners.len() - 1);
            winners[k]
        }
    }

    /// ARBITRATE — a library of banked skills resolving into ONE actor: return the `outputs` value of
    /// the module whose `gates` cell is highest (via [`Self::select`], so ties go to the last tied gate). The
    /// module that best recognizes the context drives the action — no hand-rolled argmax outside the organism.
    /// With [`Self::nest`] (a banked skill is one addressable cell — encapsulation IS isolation),
    /// [`Self::consolidate`] (no forgetting), and this competitive select, many per-task modules compose into
    /// one non-interfering actor: the composition half of the marketplace.
    pub fn arbitrate(&self, gates: &[usize], outputs: &[usize]) -> f32 {
        debug_assert!(
            !gates.is_empty() && gates.len() == outputs.len(),
            "arbitrate: gates/outputs must be non-empty and equal length — else silently returns 0.0"
        );
        let win = self.select(gates);
        outputs
            .get(win)
            .and_then(|&c| self.mem.medium.get(c).copied())
            .unwrap_or(0.0)
    }
}
