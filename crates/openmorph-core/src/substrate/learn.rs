// SPDX-License-Identifier: MIT-0
//! LEARNING — the per-event delta rule (`learn_step`), feature-direction and nested credit, and
//! predictive coding (`observe`).

use super::*;

impl Field {
    /// One SUPERVISED, LOCAL, PER-EVENT learning step — the async organism learns from a stream of
    /// experience with no epoch. Feed `inputs`, settle, then delta-update the units that write
    /// `output_cell` toward `target` (Rescorla-Wagner / LMS with the tanh derivative — the same one
    /// plasticity law, local to each firing unit). The "target" is a value from ANOTHER space (a
    /// reward, a paired observation): learning is FINDING THE RELATIONSHIP between spaces, grown, not
    /// authored. Returns `|error|` before the update. One call = one event in the stream.
    pub fn learn_step(
        &mut self,
        inputs: &[(usize, f32)],
        output_cell: usize,
        target: f32,
        lr: f32,
        budget: usize,
    ) -> f32 {
        if self.frozen {
            return 0.0;
        }
        self.stimulate(inputs, budget);
        self.learn_settled(output_cell, target, lr)
    }

    /// MANY HEADS, ONE EVENT — [`Self::learn_step`] for several outputs that read the same `inputs`: settle once,
    /// then correct every head `(output_cell, target)` against that one settled state, with their feature-direction
    /// steps summed and applied once. Calling `learn_step` per head instead re-settles the unchanged inputs per head
    /// (re-ticking any trace or resonator once per head) and moves the shared features once per head:
    /// `O(H·(W + units))` settle work and `O(H·W·fan-in)` direction work per event instead of one settle and
    /// `O(H·W + W·fan-in)`. One head is exactly [`Self::learn_step`]. Returns each head's error before its update.
    pub fn learn_heads(
        &mut self,
        inputs: &[(usize, f32)],
        heads: &[(usize, f32)],
        lr: f32,
        budget: usize,
    ) -> Vec<f32> {
        if self.frozen {
            return vec![0.0; heads.len()];
        }
        self.begin_direction_batch();
        let errs = heads
            .iter()
            .enumerate()
            .map(|(k, &(cell, target))| {
                if k == 0 {
                    self.learn_step(inputs, cell, target, lr, budget)
                } else {
                    self.learn_settled(cell, target, lr)
                }
            })
            .collect();
        self.flush_direction_batch();
        errs
    }

    /// The post-settle half of [`Self::learn_step`]: delta-update the units writing `output_cell` toward `target`
    /// from the CURRENT (already settled) cell state — no stimulate. Shared with [`Self::grow`] (which has just
    /// settled on the same inputs, so one event is one settle: a second, redundant settle would tick the
    /// traces/resonators again and wipe `last_fired`), with `observe_from`/`dream` (every head after the first learns
    /// on the one settled state) and with openmorph-physics's bit-for-bit world-model mirror. Touches weights
    /// / traits / learning meters and the staleness flags only — never the medium, the clock, the scheduler or the
    /// unit topology. Returns `|error|` before the update.
    #[doc(hidden)]
    pub fn learn_settled(&mut self, output_cell: usize, target: f32, lr: f32) -> f32 {
        let pred = self.mem.medium.get(output_cell).copied().unwrap_or(0.0);
        let err = target - pred;
        let deriv = self.slope_of_cell(output_cell, pred); // activation′ · any forward factor
                                                           // SURPRISE-DRIVEN PLASTICITY (opt-in `Config::plasticity_anneal`, default 0 = off ⇒ constant 1.0,
                                                           // byte-identical): scale the delta step by a gain that anneals as the target settles and RE-OPENS when the
                                                           // error breaks away (the target changed) — a change-detector cure for constant-rate divergence on a noisy
                                                           // target.
        let learn_gain = if self.config.plasticity_anneal > 0.0 {
            self.update_plasticity_gain(output_cell, err.abs())
        } else {
            1.0
        };
        let rls = self.config.readout_rls;
        if rls > 0.0 && self.rls.len() < self.units.len() {
            self.rls.resize(self.units.len(), RlsState::default()); // lazily: an organism without RLS carries none
        }
        let medium = &self.mem.medium;
        if let Some(ids) = self.writers.get(output_cell) {
            for &i in ids {
                if self.units[i].nested.is_some() {
                    continue; // a nested cell IS a banked sub-morph — a frozen brick, no weights to tune
                }
                let p = self.plasticity.get(i).copied().unwrap_or(1.0);
                if p == 0.0 {
                    continue; // consolidated: frozen — later learning can't degrade this skill
                }
                // WRITE-GATE: if this unit is gated, scale the delta by the gate cell's value (clamped [0,1]) —
                // it writes when the gate is open (salient) and is protected when closed. The one op the
                // additive unit lacked; short-term memory / neuromodulated learning fall out of it.
                let gate = self.units[i]
                    .gate
                    .map(|g| medium.get(g).copied().unwrap_or(0.0).clamp(0.0, 1.0))
                    .unwrap_or(1.0);
                if gate == 0.0 {
                    continue; // gate closed — the trace is protected this event
                }
                // Each writer differentiates through ITS OWN law (identical to `deriv` unless this
                // unit was transcribed with an override or carries a forward factor).
                let deriv = self.slope_of(i, pred);
                let ul = self.lr.get(i).copied().unwrap_or(1.0); // the unit's own heritable lr factor
                if rls > 0.0 {
                    // RECURSIVE LEAST SQUARES (`Config::readout_rls`): the Kalman gain replaces the delta step. The
                    // rate only gates it; plasticity, the write-gate and the anneal gain weight the event.
                    let w = p * gate * learn_gain;
                    if ul * lr == 0.0 || w <= 0.0 {
                        continue;
                    }
                    let u = &mut self.units[i];
                    let st = &mut self.rls[i];
                    st.align(&u.reads, self.config.rls_prior);
                    let g: Vec<f32> = u
                        .reads
                        .iter()
                        .map(|&c| deriv * medium.get(c).copied().unwrap_or(0.0))
                        .chain(core::iter::once(deriv))
                        .collect();
                    let k = st.step(&g, w, rls, self.config.rls_prior);
                    let n = u.reads.len();
                    for (wk, kk) in u.weights.iter_mut().zip(&k[..n]) {
                        *wk += kk * err;
                    }
                    u.bias += k[n] * err;
                    if k.iter().any(|&kk| kk * err != 0.0) {
                        self.dirty.mark(self.units.len(), i);
                    }
                    continue;
                }
                let mut step = p * ul * lr * gate * learn_gain;
                // Normalized-LMS when the flag is set — or, with the step-stability cap OFF (the legacy genome), once
                // the fan-in is wide enough to diverge (`reads.len() ≥ width_norm_min`, 128). The un-normalized step
                // scales with the input energy `Σx² ~ fan-in`, so a fixed `lr` blows up on a wide readout; the cap
                // (below) is the continuous successor of that threshold, so while it is on the threshold is not read.
                if self.config.lr_fanin_norm
                    || (self.config.stability_cap <= 0.0
                        && self.units[i].reads.len() >= self.config.width_norm_min)
                {
                    // Divide the step by the input energy Σx² so the effective learning rate is width-independent
                    // — one lr serves narrow and wide readouts. VARIANCE-MASKED (opt-in): sum x² only over reads
                    // that actually VARY, so a constant (never-firing) read on a dense band drops out of the
                    // denominator and stops taxing the lr for the reads that carry information. Variance is a
                    // per-cell EMA detached from the weights (no feedback), and the −1 evidence is untouched (only
                    // the denominator changes). Reduces to plain Σx² when every read varies.
                    let energy: f32 = if self.config.var_gated_norm {
                        let n = self.mem.medium.len();
                        if self.read_count.len() < n {
                            self.read_min.resize(n, 0.0);
                            self.read_max.resize(n, 0.0);
                            self.read_count.resize(n, 0);
                        }
                        let mut e = 0.0f32;
                        for &c in &self.units[i].reads {
                            let x = self.mem.medium.get(c).copied().unwrap_or(0.0);
                            if c < self.read_count.len() {
                                if self.read_count[c] == 0 {
                                    self.read_min[c] = x;
                                    self.read_max[c] = x;
                                } else {
                                    self.read_min[c] = self.read_min[c].min(x);
                                    self.read_max[c] = self.read_max[c].max(x);
                                }
                                self.read_count[c] = self.read_count[c].saturating_add(1);
                                // Constant ⇒ dropped, but only once CONFIRMED (seen ≥2× with zero spread); a
                                // just-seen read is still charged so the denominator can't collapse at warm-up.
                                let constant = self.read_count[c] >= 2
                                    && (self.read_max[c] - self.read_min[c]) <= 1e-9;
                                if !constant {
                                    e += x * x; // effective fan-in: only varying reads charge the denominator
                                }
                            }
                        }
                        e
                    } else {
                        self.units[i]
                            .reads
                            .iter()
                            .map(|&c| {
                                let x = medium.get(c).copied().unwrap_or(0.0);
                                x * x
                            })
                            .sum()
                    };
                    step /= self.config.nlms_eps + energy;
                } else if self.config.stability_cap > 0.0 {
                    // STEP-STABILITY CAP (`Config::stability_cap` κ): the raw LMS step is stable only while
                    // `step·(1+Σx²) < 2`, so cap the BASE RATE (the unit's heritable `lr` × the call's `lr`) at
                    // `κ/(1+Σx²)` and THEN apply the modulators (plasticity, write-gate, anneal gain — each in
                    // `[0,1]`, so the step stays stable): a half-plastic unit still learns at half the capped rate
                    // instead of being clipped to the same step as a fully plastic one. Never enlarging: a base rate
                    // already below the cap leaves the step untouched bit for bit. (`1+` is the bias read.)
                    let energy: f32 = self.units[i]
                        .reads
                        .iter()
                        .map(|&c| {
                            let x = medium.get(c).copied().unwrap_or(0.0);
                            x * x
                        })
                        .sum::<f32>()
                        + 1.0;
                    let cap = self.config.stability_cap;
                    let base = ul * lr;
                    if base.abs() * energy > cap {
                        step = p * (cap / energy).copysign(base) * gate * learn_gain;
                    }
                }
                // 2nd-order (heavy-ball) delta rule when this unit carries momentum μ: `v ← μ·v + Δw; w ← w+v`.
                // μ=0 (default) makes `upd = Δw` and applies it directly — BYTE-IDENTICAL to the 1st-order rule.
                let mu = self.momentum.get(i).copied().unwrap_or(0.0);
                // Reuse a scratch buffer for Δw instead of cloning `reads` and allocating a fresh Vec every
                // writer. `medium` (&self.mem.medium) and `ids` (&self.writers) borrow DISJOINT fields, so the
                // scratch — a third field — can be taken here (same disjointness the `&mut self.units[i]` apply
                // and `&mut self.velocities[i]` below already rely on). Byte-identical: same Δw values, same
                // order, same bias slot, same momentum update, same apply — only the backing store is reused.
                let mut upd = core::mem::take(&mut self.learn_scratch);
                upd.clear();
                let reads = &self.units[i].reads;
                let reads_len = reads.len();
                upd.extend(
                    reads
                        .iter()
                        .map(|&c| step * err * deriv * medium.get(c).copied().unwrap_or(0.0)),
                );
                upd.push(step * err * deriv); // bias gradient
                if mu != 0.0 {
                    if self.velocities.len() != self.units.len() {
                        self.velocities.resize(self.units.len(), Vec::new());
                    }
                    let vel = &mut self.velocities[i];
                    if vel.len() != upd.len() {
                        vel.clear();
                        vel.resize(upd.len(), 0.0);
                    }
                    for k in 0..upd.len() {
                        vel[k] = mu * vel[k] + upd[k]; // the damped-oscillator velocity
                        upd[k] = vel[k];
                    }
                }
                {
                    let u = &mut self.units[i];
                    for k in 0..reads_len {
                        u.weights[k] += upd[k];
                    }
                    u.bias += upd[reads_len];
                }
                if upd.iter().any(|&d| d != 0.0) {
                    // A weight change is not a read change: flag it so the next settle re-fires it (`dirty`).
                    self.dirty.mark(self.units.len(), i);
                }
                self.learn_scratch = upd; // keep the capacity for the next writer/event
            }
        }
        if rls <= 0.0 {
            self.learn_feature_directions(output_cell, err, deriv); // an RLS readout keeps its basis fixed
        }
        self.learn_nested_credit(output_cell, err, deriv);
        self.accrue_credit(output_cell, err, deriv);
        self.accumulate_sensitivity(output_cell); // Gap 7 spike 1.5: end-to-end ∂output/∂cell (no-op unless surprise_currency)
        err.abs()
    }

    /// RELAX — one UNFORCED step of the 2nd-order dynamics: the "relaxing period" between games/exercises. With
    /// no surprise to drive it, each unit's carried momentum (velocity) bleeds toward rest at that unit's OWN
    /// grown damping μ (`v ← μ·v`), so the 2nd-order state a task built up returns to equilibrium and momentum
    /// from one episode does not CONTAMINATE the next (a stale velocity would otherwise overshoot the new
    /// target). This is NOT a new mechanism: relaxation is the delta rule at ZERO error (`upd = 0 ⇒ v ← μ·v`) —
    /// it falls straight out of "the metabolism of surprise" (no surprise ⇒ the loop settles). The RATE is grown,
    /// not authored: a high-inertia lineage relaxes slowly, a light one snaps back. Call it repeatedly over the
    /// rest period; in the limit the velocity reaches rest (0). A no-op when momentum is unused (`μ=0`, the
    /// default), so it is safe to call unconditionally between episodes — byte-identical for a 1st-order organism.
    pub fn relax(&mut self) {
        for (i, vel) in self.velocities.iter_mut().enumerate() {
            let mu = self.momentum.get(i).copied().unwrap_or(0.0);
            for v in vel.iter_mut() {
                *v *= mu;
            }
        }
    }

    /// FEATURE-DIRECTION learning — the SAME delta rule as the readout, ONE HOP DEEPER. When
    /// `config.feature_lr > 0`, each grown feature the readout reads also moves its OWN input-direction toward
    /// reducing error: its pseudo-target is the residual backprojected through the readout's weight on it
    /// (`err·deriv·w`) — feedback-alignment / a one-hop delta, local and online, NO backprop-through-depth. It
    /// lets a SMALL grown structure learn its projection directions and match backprop, instead of leaning on
    /// random directions + selection (which caps below backprop when directions matter). OFF (byte-identical)
    /// at 0.0. Guards mirror the readout path (nested / plasticity / gate / heritable lr).
    fn learn_feature_directions(&mut self, output_cell: usize, err: f32, deriv: f32) {
        if self.config.feature_lr == 0.0 {
            return;
        }
        let mut acc = core::mem::take(&mut self.backproj);
        if self.dir_batching && acc.slot.get(output_cell).is_some_and(|&s| s != 0) {
            // An earlier head of this batch READS this head's cell, so its pending step moves THIS head's readout —
            // and this head backprojects (and credits nested cells) through those very weights. Land the pending
            // steps first, exactly as the per-head sequence would have, then keep batching.
            self.apply_backprojection(&mut acc);
        }
        self.backproject(&mut acc, output_cell, err, deriv);
        if !self.dir_batching {
            self.apply_backprojection(&mut acc);
        }
        self.backproj = acc;
    }

    /// Accumulate into `acc` the backprojected pseudo-error per FEATURE CELL = Σ over readout units writing
    /// `output_cell` of `err·deriv·w_readout`. Only cells WRITTEN by a unit can receive the delta (the apply loop
    /// skips the rest), so a raw input read is dropped here instead of being accumulated; the per-cell slot is a
    /// dense cell-indexed scratch (a linear `find` was O(fan-in²) per event on a wide readout, and a hash map spent
    /// most of the time hashing). Slots are in first-encounter order and each sums in encounter order.
    fn backproject(&self, acc: &mut Backproj, output_cell: usize, err: f32, deriv: f32) {
        let Some(ids) = self.writers.get(output_cell) else {
            return;
        };
        for &i in ids {
            if self.units[i].nested.is_some() {
                continue; // a nested readout is a frozen sub-morph — no leaf weights to backproject through
            }
            let u = &self.units[i];
            for (k, &c) in u.reads.iter().enumerate() {
                if self.writers.get(c).and_then(|w| w.first()).is_none() {
                    continue; // no feature unit writes `c` (a raw input) — nothing to tune
                }
                if c == output_cell {
                    // A recurrent SELF-read (a latch reading its own cell) is not a feature below the readout:
                    // the delta rule above already moved this unit, and a second, one-hop update of the same
                    // weights through its own `w_self` would double-count the error (it breaks the latch of
                    // `substrate_remember`: 100 % → 49 %).
                    continue;
                }
                let pe = err * deriv * u.weights[k];
                if c >= acc.slot.len() {
                    acc.slot.resize(c + 1, 0);
                }
                match acc.slot[c] {
                    0 => {
                        acc.errs.push((c, pe));
                        acc.slot[c] = acc.errs.len();
                    }
                    s => acc.errs[s - 1].1 += pe,
                }
            }
        }
    }

    /// Apply the one-hop delta to each feature unit writing a cell of `acc` (same guards as the readout path),
    /// then empty `acc`.
    fn apply_backprojection(&mut self, acc: &mut Backproj) {
        let flr = self.config.feature_lr;
        for &(c, _) in &acc.errs {
            acc.slot[c] = 0;
        }
        for (c, pe) in acc.errs.drain(..) {
            let fj = match self.writers.get(c).and_then(|w| w.first()) {
                Some(&fj) => fj,
                None => continue,
            };
            if self.units[fj].nested.is_some() {
                continue; // frozen brick — no direction to tune
            }
            let p = self.plasticity.get(fj).copied().unwrap_or(1.0);
            if p == 0.0 {
                continue; // consolidated — protected from later learning
            }
            let gate = self.units[fj]
                .gate
                .map(|g| {
                    self.mem
                        .medium
                        .get(g)
                        .copied()
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0)
                })
                .unwrap_or(1.0);
            if gate == 0.0 {
                continue; // gate closed — the trace is protected this event
            }
            let feat_val = self.mem.medium.get(c).copied().unwrap_or(0.0);
            let feat_deriv = self.slope_of(fj, feat_val); // activation′ · any forward factor
            let gain = self.scale.get(fj).copied().unwrap_or(1.0); // the projection gain enters the chain rule
            let ul = self.lr.get(fj).copied().unwrap_or(1.0);
            let fstep = p * ul * flr * gate * pe * feat_deriv * gain;
            self.apply_direction_step(fj, fstep);
        }
    }

    /// Move feature unit `fj`'s input-direction by `fstep` along its current read values (the one-hop delta of
    /// [`Self::learn_feature_directions`]) and flag it for re-firing.
    fn apply_direction_step(&mut self, fj: usize, fstep: f32) {
        // (split borrow of the unit's disjoint `reads`/`weights` fields — no per-feature `reads.clone()`)
        let medium = &self.mem.medium;
        let Unit { reads, weights, .. } = &mut self.units[fj];
        for (k, &cc) in reads.iter().enumerate() {
            let x = medium.get(cc).copied().unwrap_or(0.0);
            weights[k] += fstep * x;
        }
        self.units[fj].bias += fstep;
        if fstep != 0.0 {
            self.dirty.mark(self.units.len(), fj); // re-fire the moved feature
        }
    }

    /// Open a feature-direction BATCH before several heads learn on ONE settled state. Every factor of a feature's
    /// one-hop step except the backprojected error — its read values, its settled slope, gain, plasticity, gate and
    /// lr — is fixed by that state, so the per-head moves `Σ_h c·e_h·x` are ONE move `c·(Σ_h e_h)·x`: the heads'
    /// errors accumulate per feature cell (exactly as one head's writers already do) and
    /// [`Self::flush_direction_batch`] moves each feature once. When a head's OWN cell is a feature of an earlier head
    /// (heads that read each other), the pending steps land before that head learns, as they would per head. On a SHARED basis read by `H` heads that is
    /// `O(H·W + W·fan-in)` per observation instead of `O(H·W·fan-in)` (observe on a 4096-cell frame: billions of
    /// instructions per frame → tens of millions). The same update up to f32 summation order; one head is
    /// bit-identical. A no-op when directions are frozen (`feature_lr == 0`, the legacy genome) or under
    /// `surprise_currency` (its sensitivity sweep reads the feature weights between heads, so there each head's step
    /// still lands before the next).
    pub(super) fn begin_direction_batch(&mut self) {
        self.dir_batching = self.config.feature_lr != 0.0 && !self.config.tracks_surprise();
    }

    /// Apply and close the batch opened by [`Self::begin_direction_batch`] (a no-op if none is open).
    pub(super) fn flush_direction_batch(&mut self) {
        if core::mem::take(&mut self.dir_batching) {
            let mut acc = core::mem::take(&mut self.backproj);
            self.apply_backprojection(&mut acc);
            self.backproj = acc;
        }
    }

    /// CREDIT INTO NESTED CELLS (#composition) — route the composed error ONE HOP into a frozen nested sub's
    /// readout, the fix for "the sub-goal is not missing, it is UNREACHABLE." A nested cell is a banked
    /// sub-morph the learn path treats as a frozen brick (both [`Self::learn_step`] and
    /// [`Self::learn_feature_directions`] SKIP it), so a sub-skill trained in isolation before nesting only ever
    /// saw its own inputs against a label that depends on blocks it cannot see — an impossible problem, and why a
    /// grown sub lands BELOW a flat model. When `config.nest_credit` is on, the host readout's residual —
    /// backprojected through its weight on the nested cell (feedback-alignment, the SAME one-hop delta as
    /// `learn_feature_directions`, one nesting-level down) — becomes a pseudo-target for the sub's OUTPUT, and
    /// one `learn_step` tunes the sub's EXISTING units toward it. The sub's STRUCTURE stays frozen (it grows
    /// nothing), so it remains a banked, addressable skill; only its weights move. A tuned sub is no longer the
    /// same function, so its memo is cleared while credit flows. Requires the sub to have grown hidden capacity
    /// (a plastic linear readout over raw inputs cannot carve a non-separable sub-function — documented on
    /// [`crate::Config::nest_credit`]). A NO-OP (byte-identical) when the flag is off — the sub is never taken
    /// out to learn. Proof `substrate_nest_credit`.
    fn learn_nested_credit(&mut self, output_cell: usize, err: f32, deriv: f32) {
        if !self.config.nest_credit {
            return;
        }
        let lr = self.config.nest_credit_lr;
        // Backproject the composed error to each NESTED cell the host readout reads (leaf readouts only).
        let mut backproj: Vec<(usize, f32)> = Vec::new();
        if let Some(ids) = self.writers.get(output_cell) {
            for &i in ids {
                if self.units[i].nested.is_some() {
                    continue; // the host readout itself must be a leaf that READS the nested cell
                }
                let u = &self.units[i];
                for (k, &c) in u.reads.iter().enumerate() {
                    let is_nested = self
                        .writers
                        .get(c)
                        .and_then(|w| w.first())
                        .map(|&fj| self.units[fj].nested.is_some())
                        .unwrap_or(false);
                    if !is_nested {
                        continue;
                    }
                    let pe = err * deriv * u.weights[k];
                    match backproj.iter_mut().find(|(cc, _)| *cc == c) {
                        Some(slot) => slot.1 += pe,
                        None => backproj.push((c, pe)),
                    }
                }
            }
        }
        // For each nested cell, apply one composed-error delta to its sub's readout (structure frozen).
        for (c, pe) in backproj {
            let fj = match self.writers.get(c).and_then(|w| w.first()) {
                Some(&fj) => fj,
                None => continue,
            };
            if self.plasticity.get(fj).copied().unwrap_or(1.0) == 0.0 {
                continue; // a consolidated nested skill stays frozen even with credit on
            }
            // Take the sub out (as `fire_nested` does), tune it, invalidate its memo, restore.
            let mut nst = match self.units[fj].nested.take() {
                Some(n) => n,
                None => continue,
            };
            let vals: Vec<f32> = self.units[fj]
                .reads
                .iter()
                .map(|&rc| self.mem.medium.get(rc).copied().unwrap_or(0.0))
                .collect();
            let ins: Vec<(usize, f32)> = nst
                .sources
                .iter()
                .copied()
                .zip(vals.iter().copied())
                .collect();
            let cur = self.mem.medium.get(c).copied().unwrap_or(0.0);
            // Feedback-aligned pseudo-target: nudge the sub output in the direction that reduces host error.
            // The sub's own `learn_step` error is then (cur + pe) − pred_sub ≈ pe (pred_sub ≈ cur at this input).
            let sub_target = cur + pe;
            nst.sub
                .learn_step(&ins, nst.sink, sub_target, lr, nst.budget);
            nst.memo.clear(); // a tuned sub is no longer the same frozen function — the cache is stale
            self.units[fj].nested = Some(nst);
            self.dirty.mark(self.units.len(), fj); // …and so is the cell it wrote
        }
    }

    /// Accumulate CONTRIBUTION credit: each feature the readout reads earns by its correlation with the
    /// RESIDUAL — the gradient `err·deriv·activation` it receives (SIGNED, so a consistently-useful feature
    /// builds a large-magnitude EMA while an uncorrelated one averages to ~0). This is the fitness the economy
    /// selects constants on: usefulness-for-reducing-error, not mere firing. NB this is *residual-correlation*
    /// fitness (drives heritable-trait EVOLUTION — `substrate_evolve`/`heritable`, where a crisper `scale`
    /// earns more residual credit and breeds); it is deliberately NOT the readout's convergence-stable RELIANCE
    /// (`|weight·activation|`). Those are two different fitnesses: for keeping grown STRUCTURE at convergence
    /// (`substrate_construct`/`selfbuild`), select on the readout's learned weight directly — reliance is read
    /// off `units` at selection time, without overloading this signal.
    fn accrue_credit(&mut self, output_cell: usize, err: f32, deriv: f32) {
        // Apply each credit EMA INLINE — `self.credit` is a disjoint field from `self.writers`/`self.units`/
        // `self.mem`, so no collect-then-apply Vec is needed. Byte-identical: the (feature, grad) encounter
        // order is unchanged (outer writers × inner reads), and the EMA is applied sequentially in that same
        // order, so a feature touched twice accumulates exactly as the old two-phase loop did.
        let cd = self.config.credit_decay;
        if let Some(ids) = self.writers.get(output_cell) {
            for &i in ids {
                let u = &self.units[i];
                for &c in &u.reads {
                    let x = self.mem.medium.get(c).copied().unwrap_or(0.0);
                    if let Some(&fj) = self.writers.get(c).and_then(|w| w.first()) {
                        let grad = err * deriv * x;
                        self.credit[fj] = cd * self.credit[fj] + (1.0 - cd) * grad;
                    }
                }
            }
        }
    }

    /// OBSERVE — the intrinsic predictive-coding step: **no target is supplied — REALITY is the target.**
    /// The organism's predictor units have already written their expectation of the next observation into
    /// the prediction band `pred` (from the last settle, using the `obs` band as context). `observe`
    /// corrects those predictors toward what ACTUALLY arrived (`values`), then takes the observation in
    /// on the `obs` band and re-settles so the organism predicts the NEXT one. It returns the per-cell
    /// prediction error — the dense, always-present signal (surprise / curiosity / an intrinsic acting
    /// reward), even when any extrinsic reward is sparse.
    ///
    /// This is the whole thesis in one call: learning is a comparison between a cell and its expected
    /// state; here the "expected" is the organism's OWN prediction and the correction is the world, so a
    /// morph learns to model its world from the raw stream with no labels. `dim` is the band width; the
    /// predictor units (grown by e.g. [`Self::grow_backbone`] over `obs`) must already write `pred`.
    pub fn observe(
        &mut self,
        obs: usize,
        pred: usize,
        dim: usize,
        values: &[f32],
        lr: f32,
        budget: usize,
    ) -> Vec<f32> {
        let context: Vec<usize> = (0..dim).map(|i| obs + i).collect();
        self.observe_from(&context, obs, pred, dim, values, lr, budget)
    }

    /// OBSERVE FROM an explicit context — [`Self::observe`] generalized. The standing prediction is made
    /// from the `context` cells, which may hold MORE than the observed band: a `recency` time signal (so
    /// the world model predicts TIME-DEPENDENT dynamics — rhythm, delays), an action (action-conditioned
    /// prediction for acting), or another channel. `values` is the actual next observation, written into
    /// the `obs` band; `pred` is the prediction band. Still target-free — reality is the target — and it
    /// returns the per-cell prediction error. The predictor units (grown over `context`) must write `pred`.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_from(
        &mut self,
        context: &[usize],
        obs: usize,
        pred: usize,
        dim: usize,
        values: &[f32],
        lr: f32,
        budget: usize,
    ) -> Vec<f32> {
        let need = context
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(obs.max(pred) + dim.saturating_sub(1));
        self.ensure_cells(need + 1);
        // The context the standing prediction was made from (its cells' current values).
        let ctx: Vec<(usize, f32)> = context
            .iter()
            .map(|&c| (c, self.mem.medium.get(c).copied().unwrap_or(0.0)))
            .collect();
        let mut errs = Vec::with_capacity(dim);
        // ONE observation = ONE settle: the first head's `learn_step` settles the context, and every other head
        // learns on that SAME settled state (`learn_settled`). A further settle of the unchanged `ctx` would only
        // re-tick the temporal basis (a trace/resonator stepping once per head — the double-tick `grow` also no
        // longer does) and refresh the heads just learned, which no other head reads. So the heads are corrected
        // together against the one prediction the organism made, the per-frame cost is O(dim · fan-in) instead of
        // O(dim · (|ctx| + units)), and the learned heads are refreshed by the observation settle below.
        // The heads share that settled state, so their feature-direction steps are summed and applied once.
        self.begin_direction_batch();
        for i in 0..dim {
            let want = values.get(i).copied().unwrap_or(0.0);
            errs.push(want - self.mem.medium.get(pred + i).copied().unwrap_or(0.0));
            // correct the predictor toward reality
            if i == 0 {
                self.learn_step(&ctx, pred, want, lr, budget);
            } else if !self.frozen {
                self.learn_settled(pred + i, want, lr);
            }
        }
        self.flush_direction_batch();
        // Take the observation in on the obs band and settle → the predictor writes the next prediction.
        let take: Vec<(usize, f32)> = (0..dim)
            .map(|i| (obs + i, values.get(i).copied().unwrap_or(0.0)))
            .collect();
        self.stimulate(&take, budget);
        errs
    }

    /// The slope of unit `i`'s OUTPUT with respect to its pre-activation, given the current value
    /// `pred` of the cell it writes. Without a factor this is just the activation's derivative (read
    /// off the output, as every variant is defined). With one, `out = k·f(z)`, so the chain rule
    /// contributes `k` and the activation is evaluated at `out/k` — and a factor pinned to zero means
    /// the unit cannot influence anything, so no gradient reaches it.
    pub(super) fn slope_of(&self, i: usize, pred: f32) -> f32 {
        let act = self.units[i].act(self.config.activation);
        match self.units[i].factor {
            None => act.deriv(pred),
            Some(c) => {
                let k = self.mem.medium.get(c).copied().unwrap_or(0.0);
                if k == 0.0 {
                    0.0
                } else {
                    k * act.deriv(pred / k)
                }
            }
        }
    }

    /// [`Self::slope_of`] for the first unit writing `cell`; the genome's own derivative when the world
    /// writes that cell directly.
    pub(super) fn slope_of_cell(&self, cell: usize, pred: f32) -> f32 {
        match self.writers.get(cell).and_then(|w| w.first()) {
            Some(&i) => self.slope_of(i, pred),
            None => self.config.activation.deriv(pred),
        }
    }
}
