// SPDX-License-Identifier: MIT-0
//! Autonomous life — episodes and replay, `think`, `dream`, and the `live` heartbeat.

use super::*;

impl Field {
    /// STORE an EPISODE — native episodic memory. Snapshot the CURRENT values of the `stimulus` cells (what a
    /// [`Self::replay`] will re-inject to drive the cascade from this remembered state) and the `outcome` cells
    /// (the remembered result, so a replayed clone can SCORE its fit against what actually happened — dreaming,
    /// genomic evaluation, imagination). Memory IS cells: an episode is just a captured cell-state. `outcome` may
    /// be empty (a pure stimulus memory). Returns the new episode's id (its index in the buffer). The buffer
    /// clones with [`Self::fork_mutated`] and round-trips through a snapshot, so experience travels with the
    /// organism. Reads only the medium; never grows or learns.
    pub fn store_episode(&mut self, stimulus: &[usize], outcome: &[usize]) -> usize {
        let capture = |cells: &[usize], medium: &[f32]| -> Vec<(usize, f32)> {
            cells
                .iter()
                .map(|&c| (c, medium.get(c).copied().unwrap_or(0.0)))
                .collect()
        };
        let ep = Episode {
            stimulus: capture(stimulus, &self.mem.medium),
            outcome: capture(outcome, &self.mem.medium),
        };
        self.episodes.push(ep);
        self.episodes.len() - 1
    }

    /// REPLAY an episode in RESTORE mode — reinstate the EXACT remembered state and settle: a pure, deterministic
    /// recall. The episode's stimulus cells are first CLEARED, then re-injected through the EXISTING
    /// [`Self::stimulate`] path, so the reinstated state (and the downstream settle) is independent of whatever
    /// was live before — the mode a genomic evaluation needs to score a clone on the exact stored dataset. There
    /// is NO parallel settle path: replay is `stimulate` over a stored cell-state. Its blended twin is
    /// [`Self::imagine`] (fire the memory INTO the live context — internal thinking). Returns the settle events
    /// (0 for an unknown id).
    pub fn replay(&mut self, id: usize, budget: usize) -> usize {
        let Some(stimulus) = self.episodes.get(id).map(|e| e.stimulus.clone()) else {
            return 0;
        };
        // RESTORE: clear the episode's cells first so the re-injection always registers as a change and the
        // recall is deterministic regardless of the live state (reuse `stimulate` — no new settle path). A
        // cleared value equal to the stored one is simply a no-op, so a settle only does the work it must.
        let clear: Vec<(usize, f32)> = stimulus.iter().map(|&(c, _)| (c, 0.0)).collect();
        self.stimulate(&clear, budget);
        self.stimulate(&stimulus, budget)
    }

    /// IMAGINE an episode in BLEND mode — fire the memory INTO the CURRENT context (no clear) and settle, so the
    /// cascade RECOMBINES the remembered values with whatever is presently live: a new, recombinant simulation
    /// rather than a pure recall. This is Sébastien's internal thinking — "activation of a memory cell to keep
    /// simulating the memory again with ANOTHER context" — and the substrate of dreaming/imagination. It reuses
    /// the SAME [`Self::stimulate`] path as [`Self::replay`]; the only difference is it does not reset the
    /// episode's cells first, so old memory + present state settle together. Returns the settle events (0 for an
    /// unknown id).
    pub fn imagine(&mut self, id: usize, budget: usize) -> usize {
        let Some(stimulus) = self.episodes.get(id).map(|e| e.stimulus.clone()) else {
            return 0;
        };
        self.stimulate(&stimulus, budget)
    }

    /// The stored episodes (the memory buffer) — read-only, for a caller scoring a replay against the
    /// remembered `outcome` (its own memory, no external data).
    pub fn episodes(&self) -> &[Episode] {
        &self.episodes
    }

    /// One stored episode by id (its `stimulus`/`outcome` cell-states), or `None` for an unknown id.
    pub fn episode(&self, id: usize) -> Option<&Episode> {
        self.episodes.get(id)
    }

    /// THINK — internal, self-sustained dynamics: the GENERATIVE half of the organism's autonomous life. Where the
    /// heartbeat ([`Self::live`]) spontaneously REACTIVATES dormant memory to maintain and consolidate what
    /// already exists, `think` free-runs the organism's own recurrent MODEL: it feeds its OWN predicted next-state
    /// (`pred_cells`) back in as its state (`state_cells`) and re-settles, a closed-loop trajectory through its
    /// learned dynamics — the organism *treating* what it received rather than waiting for more. This is not replay
    /// of a stored episode ([`Self::replay`]); it is the substrate as a real dynamical system, free-running when
    /// unclamped, SIMULATING the consequences of its fragmentary experience: adjacent transitions it learned online
    /// compose into long-range structure it never directly saw — which the ordinary `learn_step` can then
    /// CONSOLIDATE into direct knowledge (systems consolidation: the endpoint the caller reads is a target it never
    /// experienced). Proof: `substrate_think`.
    ///
    /// `state_cells` are the cells the state is written to; `pred_cells` the cells the model predicts the next
    /// state into (same length as `state_cells`). `steps` transitions are rolled forward from the CURRENT state
    /// (set it with a `stimulate` first); `noise ∈ [0,1]` injects a little spontaneous drift each step so the
    /// dynamics do not freeze on a fixed point (0 = deterministic rollout). Returns the state after `steps`
    /// internal transitions — the simulated consequence. No external input is consumed; the frozen (crystal)
    /// organism does not think. A cell outside the medium reads as `0.0` (and, as in `stimulate`, a write to it
    /// is dropped), so a bad index cannot panic.
    pub fn think(
        &mut self,
        state_cells: &[usize],
        pred_cells: &[usize],
        steps: usize,
        noise: f32,
        budget: usize,
    ) -> Vec<f32> {
        let n = state_cells.len().min(pred_cells.len());
        if self.frozen {
            return state_cells
                .iter()
                .take(n)
                .map(|&c| self.mem.medium.get(c).copied().unwrap_or(0.0))
                .collect();
        }
        for _ in 0..steps {
            // read the organism's OWN predicted next-state, then feed it back in as its state — a free-running
            // recurrent transition. After `steps` iterations the state holds the `steps`-ahead consequence.
            let next: Vec<f32> = pred_cells
                .iter()
                .take(n)
                .map(|&c| self.mem.medium.get(c).copied().unwrap_or(0.0))
                .collect();
            let inputs: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let drift = if noise > 0.0 {
                        noise * self.rand_u01()
                    } else {
                        0.0
                    };
                    (state_cells[i], next[i] + drift)
                })
                .collect();
            self.stimulate(&inputs, budget);
        }
        state_cells
            .iter()
            .take(n)
            .map(|&c| self.mem.medium.get(c).copied().unwrap_or(0.0))
            .collect()
    }

    /// DREAM — the autonomous DISCOVERY LOOP as a first-class substrate faculty. Left alone with NO external
    /// input, the organism lives its own wake-process-sleep cycle, composing the substrate's own dynamics — the
    /// discovery loop is the organism's OWN resting dynamics (the original openmorph thesis), not a hand-written
    /// harness. For `bouts` rounds, from each `seed` state it THINKS the model forward ([`Self::think`] — free-run
    /// its recurrent dynamics to SIMULATE the consequence of its fragmentary experience) and CONSOLIDATES that
    /// simulated consequence into the `longrange` cells (the discovered structure becomes direct knowledge via the
    /// ordinary `learn_step`); then it SLEEPS — the maintenance heartbeat ([`Self::live`]) consolidates the matured,
    /// relied-upon memory and prunes what starved, RESTING when nothing remains (the emergent stop). Returns the
    /// sleep's [`Heartbeat`]. `state_cells`/`pred_cells` are the recurrent model (state ↔ predicted-next),
    /// `longrange` the head that learns the multi-step consequence (empty to skip), `think_steps` the horizon,
    /// `lr` the consolidation rate. Domain-blind (it discovers over the organism's OWN state); a modality can bank
    /// a skill VOCABULARY over the same dream with `openmorph-perception`'s crystalliser + canonicaliser. The
    /// frozen (crystal) organism does not dream.
    #[allow(clippy::too_many_arguments)]
    pub fn dream(
        &mut self,
        state_cells: &[usize],
        pred_cells: &[usize],
        longrange: &[usize],
        seeds: &[Vec<f32>],
        think_steps: usize,
        lr: f32,
        bouts: usize,
        budget: usize,
    ) -> Heartbeat {
        if self.frozen {
            return Heartbeat {
                slept: true,
                hunger: 1.0 - self.metabolism / self.config.energy_cap.max(1e-6),
                ..Heartbeat::default()
            };
        }
        // WAKE / PROCESS: think each seed forward and consolidate the simulated consequence (discovery).
        // WHEN TO DREAM (the energy coupling, opt-in via `Config::dream_satiation` > 0): dreaming is a luxury of
        // the fed — stop and conserve once hunger crosses the floor, and BURN `metabolic_cost` per bout so dreaming
        // spends the reserve it needs and SELF-LIMITS (a real homeostatic cycle, not a fixed bout count). With the
        // default floor 0.0 the coupling is inert and every bout runs — byte-identical to the ungated dream.
        let metered = self.config.dream_satiation > 0.0;
        let cap = self.config.energy_cap.max(1e-6);
        for _ in 0..bouts {
            if metered {
                if !self.should_dream() {
                    break; // hungry: conserve — the organism stops dreaming
                }
                self.metabolism = (self.metabolism - self.config.metabolic_cost).clamp(0.0, cap);
            }
            for seed in seeds {
                let clamp: Vec<(usize, f32)> = state_cells
                    .iter()
                    .copied()
                    .zip(seed.iter().copied())
                    .collect();
                self.stimulate(&clamp, budget);
                let endpoint = self.think(state_cells, pred_cells, think_steps, 0.0, budget);
                if !longrange.is_empty() {
                    // re-clamp the seed (think left the field at the endpoint) — ONE settle — and learn every
                    // long-range head on that settled state (as in `observe_from`: a re-settle of the same clamp
                    // per head would only re-tick the temporal basis and refresh heads no other head reads)
                    self.stimulate(&clamp, budget);
                    self.begin_direction_batch();
                    for (j, &c) in longrange.iter().enumerate() {
                        if let Some(&target) = endpoint.get(j) {
                            self.learn_settled(c, target, lr);
                        }
                    }
                    self.flush_direction_batch();
                }
            }
        }
        // SLEEP: the maintenance heartbeat — consolidate what matured, prune what starved, then rest. A second
        // call reaches REST if the first exhausted its budget while still ramping dormancy pressure.
        let mut hb = self.live(budget);
        if !hb.slept {
            let hb2 = self.live(budget);
            hb.ticks += hb2.ticks;
            hb.reactivations += hb2.reactivations;
            hb.consolidated += hb2.consolidated;
            hb.culled += hb2.culled;
            hb.slept = hb2.slept;
            hb.hunger = hb2.hunger;
        }
        hb
    }

    /// THE AUTONOMOUS HEARTBEAT — the culminating "living organism" primitive (D6/D7 §3). With NO
    /// external input, the organism runs its OWN internal time: it metabolizes (energy ebbs, hunger rises),
    /// spontaneously REACTIVATES dormant memories during quiescence to test whether they still earn their keep
    /// (offline replay = dreaming = introspection = memory maintenance, ALL one loop), then truly SLEEPS when no
    /// unfinished work remains. One call runs up to `budget` internal ticks and returns a [`Heartbeat`] report.
    ///
    /// Each awake tick:
    /// 1. **Metabolic step** — being awake and thinking burns energy: `metabolism −= metabolic_cost`, so HUNGER
    ///    (`1 − metabolism/energy_cap`) rises. This is "flowing time": the organism changes on its own, which a
    ///    purely reactive settler (one that only [`Self::stimulate`]s on external input) cannot.
    /// 2. **Dormancy-driven pressure** — for every grown, still-plastic cell, ramp its reactivation pressure
    ///    `P_c += heartbeat_k·(1 − recency_c)` (recency at the unit half-life: a just-fired cell adds ~0, a
    ///    long-idle one adds ~`heartbeat_k`). A leaky integrate-and-fire on dormancy, in the organism's OWN
    ///    subjective event-time — never a wall-clock (that would break replay).
    /// 3. **Reactivation** — when the longest-dormant cell's `P_c` exceeds `heartbeat_theta`, it spontaneously
    ///    fires (pattern completion via the shared settle path, `Self::reactivate`); the EXISTING contribution
    ///    economy scores it by [`Self::reliance_of`] (how much downstream still leans on it): if it still EARNS
    ///    (`income·reliance ≥ upkeep`) it is KEPT, frozen ([`Self::consolidate`]); otherwise it is left to the
    ///    reliance economy, which starves and [`Self::cull`]s it. Then `P_c` discharges to 0.
    ///
    /// **WAKE/REST DRIVE (the D6 open decision — the DEFAULT chosen here).** The organism is AWAKE while an
    /// UNFINISHED THOUGHT remains — a grown, unconsolidated cell still to be tested — and RESTS the moment none
    /// do. This is the composite drive: hunger is the *autonomy signal* (it rises as the organism works, so a
    /// closed box still "gets hungry on its own"), but the *bound* on sleep is the unfinished-thought term —
    /// because every reactivation RESOLVES a cell (consolidate ⇒ frozen, or cull ⇒ gone), the candidate set
    /// strictly shrinks, so idle ticks CONVERGE (the organism truly sleeps, never spins). Pure hunger alone was
    /// rejected as the wake driver: with no food a closed organism would stay hungry forever and never sleep
    /// (an unbounded spin, failing D6's "truly sleeps"); a hunger-driven forage needs the WORLD to respond, so
    /// with no external input the correct behaviour after finishing its thoughts is to WAIT (sleep) for input.
    /// So: think until nothing is pending, growing hungry as you go, then sleep — external input or fresh
    /// dormancy re-crossing `θ` wakes it again.
    ///
    /// Deterministic (subjective clock, no randomness) and byte-identical-OFF: a field whose heartbeat is never
    /// invoked never touches `pressure` or `metabolism` and behaves exactly as before. In CRYSTAL it only reports
    /// (no growth/economy change). `budget` bounds both the internal ticks and each reactivation's settle.
    pub fn live(&mut self, budget: usize) -> Heartbeat {
        let mut hb = Heartbeat::default();
        let cap = self.config.energy_cap.max(1e-6);
        if self.frozen {
            hb.slept = true;
            hb.hunger = 1.0 - self.metabolism / cap;
            return hb;
        }
        self.hydrate_traits();
        self.ensure_cells(self.mem.medium.len()); // size `pressure` (an old snapshot loads it empty via serde)
        let (k, theta) = (self.config.heartbeat_k, self.config.heartbeat_theta);
        let (income, upkeep, death) = (
            self.config.pool_income,
            self.config.pool_upkeep,
            self.config.pool_death,
        );
        for _ in 0..budget {
            // WAKE/REST: awake iff an unfinished thought remains (a grown, still-plastic cell to test). This is
            // the term that BOUNDS sleep — the set shrinks as reactivations resolve cells (consolidate or cull).
            let candidates = self.candidate_cells();
            if candidates.is_empty() {
                hb.slept = true; // REST — nothing pending; drop to basal metabolism (no drain) until input
                break;
            }
            hb.ticks += 1;
            // 1. metabolic step — being awake IS the passage of the organism's OWN time: advance the subjective
            //    clock (so idle memories age and grow dormant with nothing external firing — the self-driving
            //    "flowing time" D6 asks for) and burn energy, so HUNGER rises. Deterministic; never a wall-clock.
            self.clock += 1;
            self.metabolism = (self.metabolism - self.config.metabolic_cost).clamp(0.0, cap);
            // 2. dormancy-driven pressure ramp (we are quiescent between external events) + find longest-dormant
            let mut best: Option<(u64, usize)> = None;
            for &c in &candidates {
                let dormancy = 1.0 - self.recency(c, 1.0); // unit half-life: age 0 ⇒ 0, a few events ⇒ ~1
                self.pressure[c] += k * dormancy;
                let age = self.age(c);
                // longest-dormant wins; tie-break by lowest cell index (deterministic)
                if best.is_none_or(|(ba, bc)| age > ba || (age == ba && c < bc)) {
                    best = Some((age, c));
                }
            }
            // 3. reactivation: the most-dormant cell fires once its pressure crosses θ
            let Some((_, c)) = best else { continue };
            if self.pressure[c] <= theta {
                continue; // still ramping — not ready to fire this tick
            }
            self.reactivate(c, budget);
            hb.reactivations += 1;
            // Score by the EXISTING contribution economy (reliance = how much downstream still leans on c).
            self.metabolize_reliance(&candidates, income, upkeep);
            if income * self.reliance_of(c) >= upkeep {
                // still earns ⇒ KEEP it, frozen (A6) — spared from the cull below (consolidated units are spared)
                self.consolidate(&[c]);
                hb.consolidated += 1;
            }
            // the reliance economy prunes whatever starved (a memory nothing relies on any more) — E2
            hb.culled += self.cull_after(death, 0);
            self.pressure[c] = 0.0; // discharge
        }
        hb.hunger = 1.0 - self.metabolism / cap;
        hb
    }

    /// The grown MEMORY cells eligible for autonomous reactivation ([`Self::live`]): every cell WRITTEN by a
    /// unit that is still plastic (an already-[`Self::consolidate`]d skill is a resolved, permanent asset — it is
    /// not re-tested). In cell order, so the heartbeat is deterministic.
    fn candidate_cells(&self) -> Vec<usize> {
        let mut v = Vec::new();
        for (c, ws) in self.writers.iter().enumerate() {
            if let Some(&i) = ws.first() {
                if self.plasticity.get(i).copied().unwrap_or(1.0) > 0.0 {
                    v.push(c);
                }
            }
        }
        v
    }

    /// REACTIVATE a cell — the spontaneous internal FIRE of the autonomous heartbeat: push cell `c`'s current
    /// value to its readers (offline pattern completion, the [`Self::imagine`] flavour) and settle through the
    /// SHARED [`Self::drain`] path, so a dormant memory recombines with the present context. Resets the scheduler
    /// exactly as [`Self::stimulate`] does (so `last_fired` reflects THIS settle) and re-stamps `c`'s change time
    /// (it "fired", so its dormancy resets). No `tick_traces`/`tick_resonators`: an internal reactivation must not
    /// advance the external temporal basis. Returns the settle events.
    fn reactivate(&mut self, c: usize, budget: usize) -> usize {
        if self.queued.len() != self.units.len() {
            self.queued = vec![false; self.units.len()];
        } else {
            self.queued.iter_mut().for_each(|q| *q = false);
        }
        if self.last_fired.len() != self.units.len() {
            self.last_fired = vec![false; self.units.len()];
        } else {
            self.last_fired.iter_mut().for_each(|q| *q = false);
        }
        self.queue.clear();
        self.changed_at[c] = self.clock; // the memory just fired → its dormancy resets
        self.enqueue_readers(c);
        self.drain(budget)
    }
}
