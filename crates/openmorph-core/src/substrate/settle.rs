// SPDX-License-Identifier: MIT-0
//! The fire-on-change CASCADE — membrane ports (`attach`/`tap`/`feed`/`sample`/`exchange`), `stimulate`, and
//! the settle/drain loop.

use super::*;

impl Field {
    /// ATTACH a new interface: claim FRESH cells for its ports (grows the medium) and return the
    /// binding — a game, a webcam, an ARC task plugged into the living morph on the fly.
    pub fn attach(&mut self, ports: &[Port]) -> Binding {
        let b = self.mem.attach(ports);
        self.ensure_cells(self.mem.medium.len());
        b
    }

    /// TAP existing cells `[base, base+dim)` as an interface with the given flow — WITHOUT allocating.
    /// Any pattern the morph GREW (a concept: an organ's output band, a settled representation) becomes
    /// a live port: `Flow::Out` to LISTEN to it (read the morph's own representation — introspection),
    /// `Flow::In` to TRIGGER it (write those cells to fire the concept from outside), `Flow::Both` for
    /// both. The morph's inner representations are addressable — you can watch a concept or evoke it.
    pub fn tap(&mut self, base: usize, dim: usize, flow: Flow) -> Binding {
        self.ensure_cells(base + dim);
        let port = match flow {
            Flow::In => Port::input("tap", dim),
            Flow::Out => Port::output("tap", dim),
            Flow::Both => Port::both("tap", dim),
        };
        Binding {
            slots: vec![(base, dim, flow)],
            ports: vec![port],
        }
    }

    /// FEED raw values into a binding's port `port` and SETTLE — the world writing an `In`/`Both`
    /// channel (or triggering a tapped concept). Returns the firings (the work it took to settle).
    pub fn feed(&mut self, at: &Binding, port: usize, values: &[f32], budget: usize) -> usize {
        let (off, dim) = at.slot(port);
        let inputs: Vec<(usize, f32)> = values
            .iter()
            .take(dim)
            .enumerate()
            .map(|(k, &v)| (off + k, v))
            .collect();
        self.stimulate(&inputs, budget)
    }

    /// SAMPLE a binding's port `port` — the world reading an `Out`/`Both` channel (or listening to a
    /// tapped concept). A slot that does not fit in the medium (a binding from another field, an unknown
    /// port) reads as an empty slice, exactly as [`Membrane::read`] does.
    pub fn sample(&self, at: &Binding, port: usize) -> &[f32] {
        self.mem.read(at, port)
    }

    /// EXCHANGE an attached interface against the field's medium (write its `In`/`Both`, read its
    /// `Out`/`Both`). The world touches only the channels it exposes — e.g. a queryable 2-D interface
    /// reads only the pixel under the morph's fovea, not the whole image (active, compute-saving
    /// sensing: the organism travels the space and asks for the value where it matters).
    pub fn exchange<I: crate::membrane::Interface + ?Sized>(
        &mut self,
        iface: &mut I,
        at: &Binding,
    ) {
        iface.exchange(&mut self.mem, at);
    }

    /// STIMULATE + SETTLE: write the `(cell, value)` perturbations, enqueue affected units, and drain
    /// the event queue until QUIESCENT (a fixed point) or `budget` firings (anytime). Returns the
    /// firings. Deterministic: FIFO, fixed reader order, no randomness.
    ///
    /// # Example
    /// ```
    /// use openmorph_core::Field;
    /// let mut morph = Field::new(2);
    /// // (grow a unit first so there is something to fire — see `grow`/`alive`)
    /// for _ in 0..500 { for &x in &[-1.0_f32, 1.0] { morph.grow(&[(0, x)], 1, 0.5 * x, 0.1, 100); } }
    /// // perturb cell 0 and settle: the cascade fires only where cells changed, to a fixed point.
    /// morph.stimulate(&[(0, 1.0)], 100);
    /// assert!((morph.cells()[1] - 0.5).abs() < 0.2); // learned 0.5·x → ~0.5 at x = 1
    /// ```
    pub fn stimulate(&mut self, inputs: &[(usize, f32)], budget: usize) -> usize {
        self.perturb(inputs);
        self.settle_held(inputs, budget)
    }

    /// The perturbation half of [`Self::stimulate`] — everything before the drain: reset the scheduler, write the
    /// supplied inputs (fire-on-change), tick the temporal basis.
    pub(super) fn perturb(&mut self, inputs: &[(usize, f32)]) {
        self.hydrate_traits(); // self-heal a snapshot loaded before the heritable-trait fields existed
        self.queue.clear();
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
        // `last_enq`: the last input cell whose readers were enqueued in THIS loop. Nothing pops the queue until the
        // drain, so an input whose reader list EQUALS that one would only re-find every reader already queued —
        // skip the scan (a memcmp instead of a branchy per-reader check). A random-projection backbone gives every
        // input the SAME reader list, so a dense frame pays one scan instead of one per input. Byte-identical.
        let mut last_enq: Option<usize> = None;
        for &(c, v) in inputs {
            if c < self.mem.medium.len() && (v - self.mem.medium[c]).abs() > self.config.settle_eps
            {
                self.mem.medium[c] = v;
                self.changed_at[c] = self.clock; // stamp the change with the current subjective time
                self.note_value(c, v); // Gap 7 spike: running per-cell variance (no-op unless surprise_currency)
                                       // (Checked only while the queue is not already full — a full queue makes `enqueue_readers` O(1).)
                let same = self.queue.len() >= self.queued.len()
                    || last_enq.is_some_and(|p| {
                        p == c
                            || self.readers.get(p).map(Vec::as_slice)
                                == self.readers.get(c).map(Vec::as_slice)
                    });
                if !same {
                    self.enqueue_readers(c);
                    last_enq = Some(c);
                }
            }
        }
        self.tick_traces(); // refresh the multi-scale temporal basis from the (now-updated) sources (#137)
        self.tick_resonators(); // advance the band-pass resonators one step (the spectral twin, #145)
    }

    /// Write one supplied input `v` into cell `c` when it moves the cell by more than `settle_eps`, stamp it and
    /// enqueue its readers — fire-on-change.
    pub(super) fn apply_input(&mut self, c: usize, v: f32) {
        if c < self.mem.medium.len() && (v - self.mem.medium[c]).abs() > self.config.settle_eps {
            self.mem.medium[c] = v;
            self.changed_at[c] = self.clock; // stamp the change with the current subjective time
            self.note_value(c, v); // Gap 7 spike: running per-cell variance (no-op unless surprise_currency)
            self.enqueue_readers(c);
        }
    }

    /// The settle half of [`Self::stimulate`]: drain with the supplied `inputs` [`Self::held`] (a refresh must not
    /// overwrite what the caller just supplied), releasing them after. Returns the events fired.
    pub(super) fn settle_held(&mut self, inputs: &[(usize, f32)], budget: usize) -> usize {
        self.hold_seq += 1;
        if self.dirty.any() {
            // no stale unit ⇒ no refresh to guard — a settle pays nothing for it
            if self.held.len() < self.mem.medium.len() {
                self.held.resize(self.mem.medium.len(), 0);
            }
            for &(c, _) in inputs {
                if let Some(h) = self.held.get_mut(c) {
                    *h = self.hold_seq;
                }
            }
        }
        let events = self.drain(budget);
        self.hold_seq += 1; // release: nothing is held between settles
        events
    }

    /// Whether cell `c` is [`Self::held`] by the settle in progress.
    fn is_held(&self, c: usize) -> bool {
        self.held.get(c) == Some(&self.hold_seq)
    }

    /// DRAIN the event queue — the async settle loop, factored out of [`Self::stimulate`] so the autonomous
    /// heartbeat ([`Self::reactivate`]) shares the ONE settle path (a unit fires only when a cell it reads
    /// changed; firing enqueues its readers; the cascade settles to a fixed point or the anytime `budget`).
    /// Byte-identical to the inline body it replaced. Once the cascade is QUIESCENT with budget left, the units
    /// still [`Self::dirty`] are re-fired too ([`Self::enqueue_dirty`]) — the REFRESH phase; none dirty ⇒ nothing
    /// extra fires. A refresh never overwrites a cell the caller supplied to this settle ([`Self::held`]): a unit
    /// the refresh reaches that writes a held cell does not fire and stays dirty. Returns the events fired.
    pub(super) fn drain(&mut self, budget: usize) -> usize {
        let mut events = 0usize;
        let mut refresh = false; // past the natural cascade: re-firing the stale units
        loop {
            let u = match self.queue.pop_front() {
                Some(u) => u,
                None if events < budget && self.enqueue_dirty() => {
                    refresh = true;
                    continue;
                }
                None => break,
            };
            self.queued[u] = false;
            if refresh && self.is_held(self.units[u].write) {
                // the caller's value holds for this settle; the unit waits, still stale, for the next one
                self.dirty.mark(self.units.len(), u);
                continue;
            }
            self.dirty.clear(u); // it fires under its CURRENT parameters — no longer stale
            self.last_fired[u] = true; // this unit did work this settle (for the metabolism income)
            if !refresh {
                self.clock += 1; // one event = one tick of the organism's subjective time (a refresh is not an event)
            }
            let out = if self.units[u].nested.is_some() {
                self.fire_nested(u) // the recursive case: settle the sub-morph and read its sink
            } else {
                let act = self.units[u].act(self.config.activation);
                self.units[u].compute(
                    &self.mem.medium,
                    self.scale[u],
                    self.config.gain_fanin_norm,
                    act,
                )
            };
            // FORWARD FACTOR (the second-order term): scale the output by a live cell. Applied here
            // rather than inside `compute` so a nested sub-morph can be modulated the same way.
            let out = match self.units[u].factor {
                Some(c) => out * self.mem.medium.get(c).copied().unwrap_or(0.0),
                None => out,
            };
            let w = self.units[u].write;
            if w < self.mem.medium.len()
                && (out - self.mem.medium[w]).abs() > self.config.settle_eps
            {
                self.mem.medium[w] = out;
                self.changed_at[w] = self.clock;
                self.note_value(w, out); // Gap 7 spike: running per-cell variance (no-op unless surprise_currency)
                self.enqueue_readers(w);
            }
            events += 1;
            if events >= budget {
                break;
            }
        }
        if events >= budget && (!self.queue.is_empty() || self.refresh_pending()) {
            self.truncated_settles += 1; // the anytime budget cut the cascade short
        }
        events
    }

    /// Whether a settle's REFRESH phase still has work: a [`Self::dirty`] unit that is neither queued nor held (the
    /// same test [`Self::enqueue_dirty`] applies, without enqueuing).
    /// Read-only (the serialized stale list is not re-sorted); scans the flags, so call it only off the hot path.
    fn refresh_pending(&self) -> bool {
        self.dirty.any()
            && self.dirty.flag.iter().enumerate().any(|(u, &stale)| {
                stale
                    && u < self.units.len()
                    && !self.queued.get(u).copied().unwrap_or(false)
                    && self.held.get(self.units[u].write) != Some(&self.hold_seq)
            })
    }

    /// How many settles (of [`Self::stimulate`], [`Self::grow`], [`Self::learn_step`], `alive`, …) stopped at
    /// their `budget` with firings still pending — a TRUNCATED settle, whose cells are not the fixed point: a
    /// prediction read from it can be far from the settled one (with 64 grown features a budget of 50 leaves the
    /// readout up to 1.99 off). A settle of a target [`Self::grow`] has grown costs its hidden-feature count + 2
    /// firings, so size `budget` to at least `max_hidden + 2` per grown target. Cumulative over the organism's
    /// life in memory (not serialized: 0 after a snapshot load); compare it before and after a call to detect a
    /// truncation, or check it once after a run. Counts this organism's own settles, not a nested sub-morph's.
    pub fn truncated_settles(&self) -> u64 {
        self.truncated_settles
    }

    /// Enqueue every still-[`Self::dirty`] unit whose write cell is not [`Self::held`], in ascending unit order
    /// (their writes then cascade as usual). Returns whether any was enqueued.
    fn enqueue_dirty(&mut self) -> bool {
        if !self.dirty.any() {
            return false;
        }
        let mut any = false;
        for &u in self.dirty.pending() {
            let held = self.held.get(self.units[u].write) == Some(&self.hold_seq); // `is_held`
            if !held && !self.queued[u] {
                self.queued[u] = true;
                self.queue.push_back(u);
                any = true;
            }
        }
        any
    }

    /// Whether the last stimulate reached quiescence (an empty queue = a settled fixed point).
    pub fn settled(&self) -> bool {
        self.queue.is_empty()
    }

    pub(super) fn enqueue_readers(&mut self, cell: usize) {
        // Every unit already queued ⇒ nothing to add (the queue holds DISTINCT ids `< queued.len()`, so a full
        // queue means every `queued[u]` is set). The wide-input case: once the first changed input has queued
        // the whole grown pool, the remaining inputs' reader scans are pure no-ops. Byte-identical.
        if self.queue.len() >= self.queued.len() {
            return;
        }
        for &u in &self.readers[cell] {
            if !self.queued[u] {
                self.queued[u] = true;
                self.queue.push_back(u);
            }
        }
    }
}
