// SPDX-License-Identifier: MIT-0
//! The TEMPORAL sense — the subjective clock, traces (`integrate`), resonators, delays, and the learned
//! time scales (`adapt_scales`, `adapt_freq`).

use super::*;

impl Field {
    /// The organism's current SUBJECTIVE time — the total events (firings) it has processed. Not a
    /// wall-clock (that varies by machine); a deterministic, machine-independent count of its own activity.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// AGE of a cell — events since it last changed. This is "time placement": a just-changed cell
    /// (age 0) and a long-stable one (large age) are identical by value but different in time — an
    /// extra axis the async substrate has that a tick-based one cannot. Deterministic (events, not
    /// wall-clock). See [`Self::recency`] for a bounded, sense-able form.
    pub fn age(&self, cell: usize) -> u64 {
        // saturating: `changed_at <= clock` always holds in normal operation (stamps come from `clock`), so this
        // is identical then — the guard only avoids a debug underflow-panic on a hostile/corrupt snapshot.
        self.clock
            .saturating_sub(self.changed_at.get(cell).copied().unwrap_or(self.clock))
    }

    /// RECENCY of a cell — a bounded decaying trace of its age in `(0, 1]`: 1.0 the moment it changed,
    /// halving every `half_life` events. This is the readable timing signal the organism can SENSE and
    /// PREDICT (rhythm, intervals, "about to change") — like a neuron's time since its last spike.
    /// ⚠️ `half_life` is in FIRINGS (the subjective clock), not input steps — see [`Self::integrate`] for the
    /// firings-per-input conversion and why an un-scaled bank collapses.
    pub fn recency(&self, cell: usize, half_life: f32) -> f32 {
        0.5_f32.powf(self.age(cell) as f32 / half_life.max(EPS))
    }

    /// GROW A TEMPORAL TRACE — plant a leaky-integrator cell that summarizes `source` over ~`half_life`
    /// subjective events (an EMA in event-time), refreshed once per settle. Returns the new trace cell.
    ///
    /// This is the substrate's answer to "grow the timescales a task needs": where a fixed context
    /// window K sees exactly one scale, a BANK of traces at geometrically-spaced half-lives
    /// (`integrate(x, 1)`, `integrate(x, 2)`, `…4, 8, 16`) is a MULTI-SCALE temporal basis, and a trace whose
    /// source is another trace (`integrate(integrate(x, W), W)`) is a summary-of-a-summary — recursive nested
    /// pooling with exponentially-expanding receptive fields and logarithmic depth (the fractal map-of-maps
    /// on TIME; the time-twin of [`Self::grow_backbone`]). Grow features over the trace cells with
    /// [`Self::grow`] and let the reliance economy ([`Self::metabolize_reliance`] + [`Self::cull`]) keep the
    /// timescales the stream relies on and cull the rest — the morph self-sizes its temporal depth, no
    /// hand-set K. `half_life` is in the organism's own subjective time (events), like [`Self::recency`].
    ///
    /// ⚠️ UNITS: "events" here means unit FIRINGS — the subjective clock ticks once per firing as the cascade
    /// settles (see [`Self::clock`]), NOT once per input/`stimulate` call. If the organism fires `F` times per input
    /// step (often ≫ 1 for a wide grown body), a nominal `half_life` of `H` spans only `H / F` INPUT steps. A trace
    /// bank meant to separate scales over the input stream must therefore scale its half-lives by `F` (or the taps
    /// COLLAPSE to near-duplicates of the raw input — a silent, un-warned footgun: measure `corr(trace, source)`
    /// once if you rely on scale separation). Convert with your measured firings-per-input before choosing values.
    ///
    /// # Example
    /// ```
    /// use openmorph_core::Field;
    /// let mut morph = Field::new(1);
    /// let trace = morph.integrate(0, 4.0); // integrate ALLOCATES a leaky-integrator trace cell of cell 0
    /// assert_ne!(trace, 0);                 // a new cell, distinct from the source
    /// morph.stimulate(&[(0, 1.0)], 10);
    /// let _smoothed = morph.cells()[trace]; // read the trace — a smoothed EMA in subjective event-time
    /// ```
    pub fn integrate(&mut self, source: usize, half_life: f32) -> usize {
        let cell = self.mem.medium.len();
        self.ensure_cells(cell + 1);
        self.traces.push(Trace {
            cell,
            source,
            half_life,
            ticked_at: self.clock,
            sens: 0.0,
        });
        cell
    }

    /// Plant a RESONATOR — the band-pass twin of [`Self::integrate`]. Rings at `freq` (radians per step,
    /// clamped to `[0, π]`) with pole radius `1 − damping` (higher ⇒ sharper/longer ring). Returns the output cell
    /// (the in-phase state `u`; the quadrature `v` is the very next cell, and `√(u²+v²)` is the band energy).
    /// Advanced one step per settle (each `stimulate` call, so a verb that settles `k` times advances it `k` steps)
    /// by `Self::tick_resonators`; grow the frequency with [`Self::adapt_freq`].
    pub fn resonate(&mut self, source: usize, freq: f32, damping: f32) -> usize {
        let cell = self.mem.medium.len();
        self.ensure_cells(cell + 2);
        self.resonators.push(Resonator {
            cell,
            vcell: cell + 1,
            source,
            freq: freq.clamp(0.0, core::f32::consts::PI),
            radius: (1.0 - damping).clamp(0.0, self.config.resonator_radius_max),
            u: 0.0,
            v: 0.0,
            su: 0.0,
            sv: 0.0,
        });
        cell
    }

    /// Advance every resonator one step (the 2nd-order coupled-quadrature update) — the band-pass twin of
    /// [`Self::tick_traces`], called once per settle (every `stimulate`). Unlike a trace it is NOT Δt-aware: it
    /// takes exactly one step per call, so an API that settles `k` times advances it `k` steps.
    /// `u' = r(cosω·u − sinω·v) + x`, `v' = r(sinω·u + cosω·v)` (state-transition eigenvalues `r·e^{±iω}`), plus
    /// the RTRL frequency-sensitivity for [`Self::adapt_freq`]. Writes `u`/`v` to their cells via
    /// [`Self::write_cell`] on change.
    pub(super) fn tick_resonators(&mut self) {
        for i in 0..self.resonators.len() {
            let (r, w) = (self.resonators[i].radius, self.resonators[i].freq);
            let (c, s) = (w.cos(), w.sin());
            let x = self
                .mem
                .medium
                .get(self.resonators[i].source)
                .copied()
                .unwrap_or(0.0);
            let (u, v) = (self.resonators[i].u, self.resonators[i].v);
            let nu = r * (c * u - s * v) + x;
            let nv = r * (s * u + c * v);
            // ω-sensitivity RTRL: s' = J·s + (∂J/∂ω)·[u,v], with ∂J/∂ω = r·[[−sinω,−cosω],[cosω,−sinω]].
            let (su, sv) = (self.resonators[i].su, self.resonators[i].sv);
            let nsu = r * (c * su - s * sv) + r * (-s * u - c * v);
            let nsv = r * (s * su + c * sv) + r * (c * u - s * v);
            let (cell, vcell) = (self.resonators[i].cell, self.resonators[i].vcell);
            self.resonators[i].u = nu;
            self.resonators[i].v = nv;
            self.resonators[i].su = nsu;
            self.resonators[i].sv = nsv;
            let eps = self.config.settle_eps;
            if cell < self.mem.medium.len() && (nu - self.mem.medium[cell]).abs() > eps {
                self.write_cell(cell, nu);
            }
            if vcell < self.mem.medium.len() && (nv - self.mem.medium[vcell]).abs() > eps {
                self.write_cell(vcell, nv);
            }
        }
    }

    /// Self-TUNE every resonator's centre frequency by climbing its own output ENERGY — a resonator LOCKS
    /// onto the dominant frequency in its input, unsupervised: adaptive band-pass / frequency estimation, the
    /// perceptual salience a spectral front-end needs (the signal reveals its own frequencies — nothing authored).
    /// This is the frequency analog of [`Self::adapt_scales`] (which grows a trace's half-life = the real pole's
    /// radius; here the complex pole's ANGLE moves). From the RTRL sensitivity `∂[u,v]/∂ω`, the energy gradient is
    /// `½ ∂E/∂ω = u·∂u/∂ω + v·∂v/∂ω`; an energy-normalized ASCENT step in `ω`. (Supervised, task-driven tuning of a
    /// resonator's frequency by back-prop through the pole angle is a notoriously non-convex optimization; energy
    /// ascent has a clean unimodal landscape and is the natural self-tuning for a perceptual resonator.)
    pub fn adapt_freq(&mut self, lr: f32) {
        if self.frozen {
            return;
        }
        for i in 0..self.resonators.len() {
            let r = &self.resonators[i];
            let grad = r.u * r.su + r.v * r.sv; // ½ ∂E/∂ω (E = u²+v²)
            if !grad.is_finite() || grad == 0.0 {
                continue;
            }
            // SIGN-based ascent (Rprop-style), step size `lr`. The RTRL sensitivity blows up resonantly at the
            // pole, so its magnitude is unusable, but its SIGN cleanly points toward the signal's frequency from
            // both sides (verified: + below resonance, − above). ω marches to the signal's band and dithers there.
            let step = lr * grad.signum();
            self.resonators[i].freq =
                (self.resonators[i].freq + step).clamp(0.0, core::f32::consts::PI);
        }
    }

    /// PLANT A DELAY — a native unit-delay register that holds `source`'s value from the PREVIOUS event
    /// (`x[t−1]`), the EXACT-memory twin of [`Self::integrate`]. Chain it for depth: `delay(x)` is
    /// `x[t−1]`, `delay(delay(x))` is `x[t−2]`, … — a native shift register carrying the recent EXACT order
    /// that a trace (an EMA) blurs. Returns the new delay cell. Advance the whole shift register ONCE per
    /// external event with [`Self::advance_delays`] (a delay is a discrete register, so it must not shift
    /// inside a settle). Grow a readout over a bank of delays (exact) and traces (trend) and let the reliance
    /// economy ([`Self::metabolize_reliance`] + [`Self::cull`]) select the flavor per task — native streaming
    /// memory with no hand-set window.
    pub fn delay(&mut self, source: usize) -> usize {
        let cell = self.mem.medium.len();
        self.ensure_cells(cell + 1);
        self.delays.push(Delay {
            cell,
            source,
            prev: 0.0,
        });
        cell
    }

    /// ADVANCE THE DELAY SHIFT REGISTER by one external event — call once per event (the explicit event
    /// boundary a discrete register needs, which the async settle can't supply). Each delay emits the value
    /// it captured last event (`x[t−k]`) and captures its source's value for next event; in registration
    /// order, so a delay-of-a-delay becomes `x[t−2]` in one pass (each reads its predecessor's fresh output).
    ///
    /// Returns the `(cell, value)` updates rather than writing the medium, so a caller feeds them straight
    /// into the next [`Self::grow`]/[`Self::stimulate`] as genuine input CHANGES — otherwise a re-applied
    /// same value would not re-fire the readout in the async settle. The natural streaming step is thus:
    /// `let mem = field.advance_delays(); field.grow(&[(x_cell, x)].into_iter().chain(mem)…, next, …)` — the
    /// exact-order memory is native (delay cells the readout grows over), with no hand-built window.
    #[must_use]
    pub fn advance_delays(&mut self) -> Vec<(usize, f32)> {
        let mut updates: Vec<(usize, f32)> = Vec::with_capacity(self.delays.len());
        // Cell → value emitted so far this pass (the LATEST emission of that cell), for O(1) chaining lookup:
        // exactly the old `updates.iter().rev().find(cell == source)`. A dense per-cell slot (1 + index into
        // `updates`, 0 = not emitted) instead of a SipHash map — it was ~30% of a streamed delay bank's event;
        // an out-of-medium cell (corrupt snapshot only) falls back to a map. Byte-identical.
        let n = self.mem.medium.len();
        let mut latest: Vec<usize> = vec![0; n];
        let mut emitted_oob: HashMap<usize, f32> = HashMap::new();
        for i in 0..self.delays.len() {
            let (cell, source) = (self.delays[i].cell, self.delays[i].source);
            // The source's value: an earlier delay's fresh output this pass (chaining), else the medium.
            let chained = if source < n {
                match latest[source] {
                    0 => None,
                    k => Some(updates[k - 1].1),
                }
            } else {
                emitted_oob.get(&source).copied()
            };
            let src =
                chained.unwrap_or_else(|| self.mem.medium.get(source).copied().unwrap_or(0.0));
            let out = self.delays[i].prev;
            self.delays[i].prev = src;
            updates.push((cell, out));
            if cell < n {
                latest[cell] = updates.len();
            } else {
                emitted_oob.insert(cell, out);
            }
        }
        updates
    }

    /// Refresh every temporal trace by one Δt-aware event-time step (called at the top of [`Self::stimulate`]
    /// after inputs are applied, so a trace sees the current source). Each trace relaxes toward its source by
    /// `1 − 0.5^(Δt/half_life)` where `Δt` is the subjective clock elapsed since it last ticked — so a trace
    /// touched by several settles in one external event still relaxes by exactly the elapsed time, and it
    /// never over-reacts after a quiet spell. Registration order = dependency order, so a trace-of-a-trace
    /// pools an already-updated summary in the same pass.
    pub(super) fn tick_traces(&mut self) {
        for i in 0..self.traces.len() {
            let (cell, source, hl, last) = {
                let t = &self.traces[i];
                (t.cell, t.source, t.half_life, t.ticked_at)
            };
            let dt = self.clock.saturating_sub(last) as f32;
            self.traces[i].ticked_at = self.clock;
            if dt == 0.0 {
                continue; // no subjective time has elapsed since the last tick — nothing to integrate
            }
            let h = hl.max(EPS);
            let relax = 1.0 - 0.5_f32.powf(dt / h);
            let src = self.mem.medium.get(source).copied().unwrap_or(0.0);
            let old = self.mem.medium.get(cell).copied().unwrap_or(0.0);
            let new = old + relax * (src - old);
            // Forward-mode sensitivity ∂m/∂h, ACCUMULATED (RTRL, one scalar): because a trace depends on its
            // whole history through the EMA, the sensitivity is a recurrence, not an instantaneous term —
            // `s ← (1−relax)·s + ∂relax/∂h·(src−m)`, with `∂relax/∂h = −(1−relax)·ln2·(Δt/h²)`. (Using only the
            // instantaneous term biases the gradient toward endless smoothing.) Read by `adapt_scales`.
            let d_relax_dh = -(1.0 - relax) * core::f32::consts::LN_2 * (dt / (h * h));
            self.traces[i].sens = (1.0 - relax) * self.traces[i].sens + d_relax_dh * (src - old);
            if (new - old).abs() > self.config.settle_eps {
                self.write_cell(cell, new);
            }
        }
    }

    /// WRITE a cell from the substrate's own temporal dynamics (a trace or resonator step): set the value, stamp
    /// the change with the current subjective time, note it for the surprise currency ([`Self::note_value`], a
    /// no-op unless [`Config::surprise_currency`]) and enqueue the cell's readers — the same four steps an input
    /// write in `stimulate` and a unit output in the settle take, so a temporal cell is visible to every per-cell
    /// meter. The caller has already applied the `settle_eps` change test (and any bounds check).
    fn write_cell(&mut self, c: usize, v: f32) {
        self.mem.medium[c] = v;
        self.changed_at[c] = self.clock;
        self.note_value(c, v);
        self.enqueue_readers(c);
    }

    /// HALF-LIFE GRADIENT — learn each temporal trace's TIMESCALE, the same delta rule one hop deeper (the
    /// time-twin of feature-DIRECTION learning). After a prediction at `target_cell`, backproject the
    /// residual to each trace a unit-writing-`target_cell` reads (`e = err·tanh′·w`, feedback-alignment) and
    /// descend the half-life along its captured sensitivity `∂m/∂h`. The step is MULTIPLICATIVE (gradient in
    /// `ln h`, `dL/d(ln h) = dL/dh · h`) because a timescale is a scale parameter — this cancels one factor of
    /// the `1/h²` sensitivity decay, so a trace started far too SLOW descends as readily as one too fast.
    /// Local, online, no backprop-through-time, and — unlike a competing pool over a MULTICOLLINEAR bank —
    /// each trace moves on its OWN backprojected error, so the timescale descends smoothly toward the one the
    /// stream needs instead of the pool collapsing to a single scale. The per-event log-step is bounded to
    /// `±STEP_CAP` for stability and `half_life` is clamped to `[min_hl, max_hl]`. Call once per event after
    /// [`Self::grow`] with `target` the taught value. This GROWS the temporal resolution where the
    /// discrete pool could not.
    pub fn adapt_scales(
        &mut self,
        target_cell: usize,
        target: f32,
        lr: f32,
        min_hl: f32,
        max_hl: f32,
    ) {
        if self.frozen {
            return;
        }
        // Max per-event log-step (≈ ±22% at the default), for stability — one Config lever (was a buried const).
        let step_cap = self.config.adapt_step_cap;
        // A half-life is a positive scale; floor the range at EPS so the multiplicative step can never drive
        // a trace to a degenerate ≤ 0 timescale, and treat a collapsed [max < min] range gracefully.
        let min_hl = min_hl.max(EPS);
        let max_hl = max_hl.max(min_hl);
        let pred = self.mem.medium.get(target_cell).copied().unwrap_or(0.0);
        let err = target - pred;
        let deriv = self.slope_of_cell(target_cell, pred); // activation′ · any forward factor, at the readout
        let base = err * deriv;
        if !base.is_finite() || base == 0.0 {
            return;
        }
        for i in 0..self.traces.len() {
            let (cell, sens, h0) = (
                self.traces[i].cell,
                self.traces[i].sens,
                self.traces[i].half_life,
            );
            if !sens.is_finite() || sens == 0.0 {
                continue;
            }
            // Sum the backprojected error over every unit that writes the target and reads this trace cell.
            let mut e = 0.0f32;
            if let Some(ws) = self.writers.get(target_cell) {
                for &u in ws {
                    let unit = &self.units[u];
                    for (k, &rc) in unit.reads.iter().enumerate() {
                        if rc == cell {
                            e += base * unit.weights[k];
                        }
                    }
                }
            }
            if !e.is_finite() || e == 0.0 {
                continue;
            }
            // log-space gradient: dL/d(ln h) ∝ e · (∂m/∂h) · h. Ascend error-reduction ⇒ h ← h·exp(step).
            let raw = lr * e * sens * h0;
            if !raw.is_finite() {
                continue;
            }
            let step = raw.clamp(-step_cap, step_cap);
            self.traces[i].half_life = (h0 * step.exp()).clamp(min_hl, max_hl);
        }
    }
}
