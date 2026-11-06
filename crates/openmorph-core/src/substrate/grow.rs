// SPDX-License-Identifier: MIT-0
//! GROWTH — the one `grow` verb, the shared backbone, feature birth, and the cell-slot allocator.

use super::*;

impl Field {
    /// GROW — the ONE growth verb, the AUTONOMOUS default. Grow toward `target` at `target_cell`,
    /// letting the morph size its own capacity: ensure a linear WRITER exists (surprise-birth of the
    /// transducer), then, while error persists, birth HIDDEN random-projection features up to the
    /// [`Config`] ceiling (`birth_gate`, `max_hidden`) and tune the readout. A caller says only "grow toward
    /// this" — a LINEAR task grows just the writer (stays exact), a NON-LINEAR one grows the depth surprise
    /// demands (the XOR/`substrate_deep` regime). This dissolved the old `grow_and_learn` (writer-only) and
    /// `grow_deep_step` (writer + hidden) into one verb: set `max_hidden = 0` for writer-only. It is also the
    /// self-sizing alternative to [`Self::grow_backbone`] (a caller-sized fixed random basis, still the shared
    /// state backbone for many readouts). The grown features' directions also learn, one hop below the readout
    /// (`config.feature_lr`, on by default since 1.6; `0` freezes them random), and `metabolize_reliance` + `cull`
    /// prune the excess so the ceiling stops mattering. Under the default `config.grow_within_budget` a feature is
    /// born only while a settle of the inputs still fits `budget` (the budget bounds the pool instead of truncating
    /// its settles). Returns `(|error| before, born_a_feature_this_step)`.
    ///
    /// Lower-level than [`Self::alive`] — prefer `alive(Event::Perceive { .. })`, which wraps `grow` with the
    /// self-sizing economy and respects Crystal mode. Use `grow` directly only when you want the raw growth
    /// step without the economy.
    ///
    /// # Example
    /// ```
    /// use openmorph_core::Field;
    /// let mut morph = Field::new(2);
    /// for _ in 0..800 {
    ///     for &x in &[-1.0_f32, 1.0] {
    ///         morph.grow(&[(0, x)], 1, (1.5 * x).tanh(), 0.1, 100); // birth-on-surprise + tune, per event
    ///     }
    /// }
    /// morph.stimulate(&[(0, 1.0)], 100);
    /// assert!((morph.cells()[1] - (1.5_f32).tanh()).abs() < 0.1);
    /// ```
    pub fn grow(
        &mut self,
        inputs: &[(usize, f32)],
        target_cell: usize,
        target: f32,
        lr: f32,
        budget: usize,
    ) -> (f32, bool) {
        if self.frozen {
            return (0.0, false);
        }
        let birth_gate = self.config.birth_gate;
        let max_hidden = self.config.max_hidden;
        self.ensure_cells(target_cell + 1);
        // Ensure a linear output unit over the inputs exists (surprise-birth of the WRITER).
        if !self.has_writer(target_cell) {
            let reads: Vec<usize> = inputs.iter().map(|(c, _)| *c).collect();
            let n = reads.len();
            self.add_unit(reads, vec![0.0; n], 0.0, target_cell);
        }
        self.perturb(inputs);
        let primed_at = self.clock; // a cell the cascade writes from here on is stamped later — see `reclamp`
        self.settle_held(inputs, budget);
        let pred = self.mem.medium.get(target_cell).copied().unwrap_or(0.0);
        let err = target - pred;
        let sd = self.config.surprise_decay;
        self.surprise[target_cell] = sd * self.surprise[target_cell] + (1.0 - sd) * err.abs();
        // SLOW surprise trace (the lagging reference for learning_progress). Lazily sized; only ever READ by the
        // #155 accessors, so writing it changes no existing output — byte-identical.
        if self.surprise_slow.len() < self.surprise.len() {
            self.surprise_slow.resize(self.surprise.len(), 0.0);
        }
        let cd = self.config.credit_decay;
        self.surprise_slow[target_cell] =
            cd * self.surprise_slow[target_cell] + (1.0 - cd) * err.abs();

        // GROWN PRODUCT DISCOVERY (opt-in `Config::grow_products`): accrue the pairwise product-synergy `err·xᵢ·xⱼ`
        // so a dominant interacting pair can be discovered at birth below. Uses the FAST `surprise_decay` (not the
        // slow `credit_decay`): surprise-birth fills the `max_hidden` budget within a few steps, so the interaction
        // must be discoverable within that window or the product feature is never born. Off ⇒ never accrued (and no
        // pair is ever consulted) — byte-identical.
        if self.config.grow_products {
            let input_cells: Vec<usize> = inputs.iter().map(|(c, _)| *c).collect();
            if self.config.products_per_target {
                self.accrue_target_credit(
                    target_cell,
                    &input_cells,
                    err,
                    self.config.surprise_decay,
                );
            } else {
                self.accrue_co_credit(&input_cells, err, self.config.surprise_decay);
            }
        }

        let out_idx = self.writers[target_cell][0]; // the sole unit writing this cell (just ensured)
                                                    // Hidden features are the reads BEYOND the raw inputs. `saturating_sub` because a writer may have been
                                                    // seeded with FEWER inputs than a later `grow` supplies (e.g. a cell wired from a subset of the inputs,
                                                    // then grown with the full set) — then there are simply zero hidden yet, not an underflow panic. Identical
                                                    // to the bare subtraction whenever `reads.len() >= inputs.len()` (every non-panicking case).
        let n_hidden = self.units[out_idx].reads.len().saturating_sub(inputs.len());
        let mut born = false;
        if self.surprise[target_cell] > birth_gate
            && n_hidden < max_hidden
            && (!self.config.grow_within_budget || self.birth_fits_budget(inputs, out_idx, budget))
        {
            // Birth a hidden feature and wire the output to read it (weight learned from 0). A newborn hidden cell
            // is maximally flexible; the economy (reliance/cull) keeps only the ones that earn. Normally a random-
            // projection feature (scale 1.0 — its gain is the heritable `scale` trait); with `grow_products` on, if
            // an input PAIR's product-synergy dominates, a PRODUCT feature `xⱼ·f(xᵢ)` on that pair instead.
            let input_cells: Vec<usize> = inputs.iter().map(|(c, _)| *c).collect();
            let product = if !self.config.grow_products {
                None
            } else if self.config.products_per_target {
                self.top_target_pair(target_cell, self.config.product_gate)
            } else {
                self.top_synergy_pair(&input_cells, self.config.product_gate)
            };
            let h = match product {
                Some((i, j)) => self.birth_product_feature(i, j),
                None => {
                    // FAN-IN-SCALED INIT (`Config::fanin_init_gain` c): input weights `U[-1,1]·c/√fan_in`, so the
                    // pre-activation spread does not grow with the fan-in; 0 ⇒ the fan-in-blind `U[-1,1]`.
                    let c = self.config.fanin_init_gain;
                    let w_scale = if c > 0.0 {
                        c / (input_cells.len().max(1) as f32).sqrt()
                    } else {
                        1.0
                    };
                    self.birth_feature_scaled(&input_cells, w_scale, 1.0)
                }
            };
            self.units[out_idx].reads.push(h);
            self.units[out_idx].weights.push(0.0);
            // `h` is freshly allocated, so (canonically) no unit read it: `[out_idx]` stays canonical. Defensive:
            // anything already listed there means the adjacency was not what the flag claims.
            if !self.readers[h].is_empty() {
                self.adj_canonical = false;
            }
            self.readers[h].push(out_idx);
            born = true;
        }
        // Tune ONLY the output (hidden features are fixed) on the state settled above — ONE settle per event — with
        // the supplied inputs held: a supplied cell that a unit also writes (teacher forcing, e.g. a committed carry)
        // may have been overwritten by the cascade, so re-apply it first (`reclamp`, no second temporal tick).
        self.reclamp(inputs, primed_at, budget);
        self.learn_settled(target_cell, target, lr);
        (err.abs(), born)
    }

    /// Whether [`Self::settle_cost_after_birth`] fits `budget`, without computing it when a cheap upper bound
    /// already does: the cost is `r + refire + 1 + starts_refiring` with `r` the widest input's reader count and
    /// `refire ≤ r` (a filter over those readers), so it is at most `2r + 2`. Finding `r` is O(inputs); the exact
    /// refire count scans every reader's reads, O(r · fan-in), which made one `grow` O(N²) and a perception world
    /// model's per-cell `observe` O(N³). A perception model's budget (`units + Σ fan-in`) clears the bound, so the
    /// exact count only runs when the budget is tight. Same answer either way.
    fn birth_fits_budget(&self, inputs: &[(usize, f32)], out_idx: usize, budget: usize) -> bool {
        let widest = inputs
            .iter()
            .filter_map(|&(c, _)| self.readers.get(c).map(Vec::len))
            .max()
            .unwrap_or(0);
        widest
            .checked_mul(2)
            .and_then(|b| b.checked_add(2))
            .is_some_and(|bound| bound <= budget)
            || self.settle_cost_after_birth(inputs, out_idx) <= budget
    }

    /// [`Self::input_settle_cost`] AFTER a feature is born for `out_idx`: the newborn reads every input (one more
    /// firing), and a writer that read no grown cell yet becomes one that re-fires after its features (one more,
    /// unless a unit reads its cell). The bound compares THIS with the budget, so the birth that would first push a
    /// settle past the budget is refused (comparing the pre-birth cost let the first birth through at `budget = 2`,
    /// after which every settle truncated: review of #494, core #4).
    fn settle_cost_after_birth(&self, inputs: &[(usize, f32)], out_idx: usize) -> usize {
        let unit = &self.units[out_idx];
        let grown = |c: usize| self.writers.get(c).is_some_and(|w| !w.is_empty());
        let leaf = self.readers.get(unit.write).is_none_or(Vec::is_empty);
        let starts_refiring = leaf && !unit.reads.iter().any(|&c| grown(c));
        self.input_settle_cost(inputs) + 1 + usize::from(starts_refiring)
    }

    /// The firings a change of `inputs` costs to settle a grown pool — what [`Config::grow_within_budget`] compares
    /// with the budget: every unit reading the most-read input fires once, and each of them that fires LAST — its
    /// output no unit reads (a grown target's writer) and it reads a grown cell (a feature) — fires once more, after
    /// its features. Exact for the pools [`Self::grow`] builds (`hidden + 2` per grown target with features, 1 for a
    /// writer with none yet: the writer and every feature read all the inputs), so a budget that settles the current
    /// pool exactly — a perception model's `units + Σ fan-in` — always leaves room for its next feature; a lower
    /// bound for a hand-wired graph whose inputs are read by different units, or deeper chains.
    fn input_settle_cost(&self, inputs: &[(usize, f32)]) -> usize {
        let widest = inputs
            .iter()
            .filter_map(|&(c, _)| self.readers.get(c))
            .max_by_key(|r| r.len());
        let grown = |c: usize| self.writers.get(c).is_some_and(|w| !w.is_empty());
        widest.map_or(0, |r| {
            let refire = r
                .iter()
                .filter(|&&u| {
                    let unit = &self.units[u];
                    self.readers.get(unit.write).is_none_or(Vec::is_empty)
                        && unit.reads.iter().rev().any(|&c| grown(c)) // a grown writer's features come last
                })
                .count();
            r.len() + refire
        })
    }

    /// RE-CLAMP before [`Self::grow`] learns: every supplied input the CASCADE overwrote (a unit wrote its cell
    /// after `primed_at`, the clock when the drain began — the refresh never writes a supplied cell and takes no
    /// time, so only a natural firing stamps one later) is re-applied and the settle continues, WITHOUT ticking the
    /// traces/resonators again and without resetting `last_fired` — so the readout learns on the values the caller
    /// gave, and one `grow` is one step of the temporal basis. A supplied trace/resonator cell keeps its tick.
    /// Nothing overwritten (the common case: inputs no unit writes) ⇒ a no-op.
    fn reclamp(&mut self, inputs: &[(usize, f32)], primed_at: u64, budget: usize) {
        let overwritten = |f: &Self, c: usize| f.changed_at.get(c).is_some_and(|&t| t > primed_at);
        if !inputs.iter().any(|&(c, _)| overwritten(self, c)) {
            return;
        }
        self.queued.resize(self.units.len(), false); // a unit born since the settle joins the scheduler
        self.last_fired.resize(self.units.len(), false);
        for &(c, v) in inputs {
            if overwritten(self, c) {
                self.apply_input(c, v);
            }
        }
        self.settle_held(inputs, budget);
    }

    /// GROW A BACKBONE — expand raw input cells into a SHARED non-linear feature space: birth `width`
    /// random-projection features `tanh(scale·r·x + b)` of `inputs`, and return their cells. Unlike
    /// [`Self::grow`] (features grown *per output*, gated by that output's error), a backbone
    /// is grown ONCE from the inputs' own structure — decoupled from any downstream target — so many
    /// outputs can LEARN a readout over the SAME stable basis.
    ///
    /// This is the fix for value credit on a NOISY reward channel: growing hidden features per action
    /// gated by the exploration-noisy per-action reward stalls (the surprise that gates birth is
    /// corrupted), but growing them ONCE on the shared STATE backbone separates feature FORMATION
    /// (stable, state-driven) from value CREDIT (learned per action). Each action reads
    /// `[state.. , backbone..]` and tunes with [`Self::learn_step`] from reward — opposite policies
    /// multiplex. `scale` sets the feature bandwidth (larger ⇒ crisper separation of discrete states).
    pub fn grow_backbone(&mut self, inputs: &[usize], width: usize, scale: f32) -> Vec<usize> {
        if self.frozen {
            return Vec::new();
        }
        // The shared-basis width is the caller's explicit argument (e.g. `0` = a pure linear readout). `width` is
        // NOT overloaded — `0` keeps its literal meaning (no features).
        (0..width)
            .map(|_| self.birth_feature(inputs, scale))
            .collect()
    }

    /// BIRTH A FEATURE — the atom of growth (shared by `grow` and `grow_backbone`): a fresh
    /// unit reading `inputs` with random weights (×`scale`) and a random bias, writing a NEW cell as
    /// `tanh(gain·(Σ r·x + b))`. A maximally-flexible newborn cell — born random, to be ATTRACTED to useful
    /// structure (`feature_lr`, its direction migrates) and SELECTED by the economy (reliance/cull). Returns
    /// the new cell. (Byte-identical rand-draw order to the inlined births it replaced.)
    fn birth_feature(&mut self, inputs: &[usize], scale: f32) -> usize {
        self.birth_feature_scaled(inputs, scale, scale)
    }

    /// [`Self::birth_feature`] with separate input-weight and bias scales (the fan-in-scaled init of `grow`, see
    /// [`Config::fanin_init_gain`], scales only the input weights). Same rand-draw order.
    fn birth_feature_scaled(&mut self, inputs: &[usize], w_scale: f32, b_scale: f32) -> usize {
        let weights: Vec<f32> = inputs
            .iter()
            .map(|_| self.rand_weight() * w_scale)
            .collect();
        let bias = self.rand_weight() * b_scale;
        let h = self.alloc_cell();
        self.add_unit(inputs.to_vec(), weights, bias, h);
        h
    }

    /// Birth a PRODUCT (second-order) hidden feature `x_j · f(x_i)` on the pair `(i, j)` and return its cell — a
    /// grown [`Unit::factor`] (`reads=[i], weights=[1.0], factor=Some(j)`). Captures a low-order INTERACTION that
    /// no affine unit over a random-projection basis can represent ("x_i matters only when x_j is active").
    /// [`Self::set_factor`] registers `j` as a read so fire-on-change tracks it.
    fn birth_product_feature(&mut self, i: usize, j: usize) -> usize {
        let h = self.alloc_cell();
        self.add_unit(vec![i], vec![1.0], 0.0, h);
        self.set_factor(h, Some(j));
        h
    }

    /// The input-cell PAIR with the largest accrued product-synergy `|co_credit|`, if it clears `threshold` — the
    /// pair whose product most explains the residual a linear readout leaves (opt-in `Config::grow_products`).
    /// `None` (⇒ `grow` falls back to a random feature) when no pair clears the bar. Deterministic: scans pairs in
    /// `(a < b)` order and a strict `>` keeps the first on a tie.
    fn top_synergy_pair(&self, cells: &[usize], threshold: f32) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, f32)> = None;
        for a in 0..cells.len() {
            for b in (a + 1)..cells.len() {
                let s = self.co_credit_of(cells[a], cells[b]).abs();
                if best.is_none_or(|(_, _, bs)| s > bs) {
                    best = Some((cells[a], cells[b], s));
                }
            }
        }
        best.filter(|&(_, _, s)| s > threshold)
            .map(|(i, j, _)| (i, j))
    }

    /// Accrue `target`'s own co-credit (`Config::products_per_target`): one step of its EMA of `err · x_i · x_j`,
    /// over the pairs of `cells` whose values are both nonzero. Every other pair's sample is zero, so its EMA only
    /// decays; that decay is applied lazily ([`Self::target_credit_now`]) instead of walking every pair.
    fn accrue_target_credit(&mut self, target: usize, cells: &[usize], err: f32, decay: f32) {
        let t = {
            let e = self.target_ticks.entry(target).or_insert(0);
            *e += 1;
            *e
        };
        let active: Vec<(usize, f32)> = cells
            .iter()
            .map(|&c| (c, self.mem.medium.get(c).copied().unwrap_or(0.0)))
            .filter(|&(_, x)| x != 0.0)
            .collect();
        for a in 0..active.len() {
            for b in (a + 1)..active.len() {
                let ((ci, xi), (cj, xj)) = (active[a], active[b]);
                let key = if ci < cj {
                    (target, ci, cj)
                } else {
                    (target, cj, ci)
                };
                let e = self.target_credit.entry(key).or_insert((0.0, t - 1));
                e.0 = decay_by(e.0, decay, t - 1 - e.1) * decay + (1.0 - decay) * err * xi * xj;
                e.1 = t;
            }
        }
    }

    /// `target`'s pair `(i, j)` co-credit as of its latest step (the lazy decay of the steps it was not accrued at).
    fn target_credit_now(&self, key: (usize, usize, usize), v: (f32, u64)) -> f32 {
        let t = self.target_ticks.get(&key.0).copied().unwrap_or(v.1);
        decay_by(v.0, self.config.surprise_decay, t.saturating_sub(v.1))
    }

    /// The input pair with `target`'s own largest co-credit, if it clears `threshold`. Deterministic: the largest
    /// magnitude, a tie to the smaller pair.
    fn top_target_pair(&self, target: usize, threshold: f32) -> Option<(usize, usize)> {
        let mut best: Option<((usize, usize), f32)> = None;
        for (&key, &v) in self
            .target_credit
            .range((target, 0, 0)..=(target, usize::MAX, usize::MAX))
        {
            let s = self.target_credit_now(key, v).abs();
            let pair = (key.1, key.2);
            if best.is_none_or(|(bp, bs)| s > bs || (s == bs && pair < bp)) {
                best = Some((pair, s));
            }
        }
        best.filter(|&(_, s)| s > threshold).map(|(p, _)| p)
    }

    /// Allocate a cell for a grown feature — REUSE a slot vacated by [`Self::cull`] (bounding the medium at the
    /// peak concurrent cell count) before growing the medium. A reused slot is reset to a pristine state.
    pub(super) fn alloc_cell(&mut self) -> usize {
        while let Some(c) = self.free_cells.pop() {
            // defensive: only reuse a slot that is genuinely free (no writer/reader took it since it was freed).
            if c < self.mem.medium.len() && self.writers[c].is_empty() && self.readers[c].is_empty()
            {
                self.reset_cell(c);
                return c;
            }
        }
        let h = self.mem.medium.len();
        self.ensure_cells(h + 1);
        h
    }

    /// Reset every per-cell parallel state for a cell being (re)used, so a reclaimed slot carries nothing stale:
    /// each entry reads what a never-used cell reads (the lazily-sized arrays' resize value; a no-op on an array
    /// not yet allocated). THE per-cell registry — a new per-cell array or cell-keyed map must be listed here, or a
    /// reused slot inherits the previous feature's life (stale learning progress biasing `orient`, stale pressure
    /// firing the heartbeat, stale variance paid to a newborn). `readers`/`writers` are empty by construction (a
    /// slot is freed only once nothing references it); `pos` is zeroed, as before.
    pub(super) fn reset_cell(&mut self, c: usize) {
        fn put<T: Copy>(v: &mut [T], c: usize, x: T) {
            if let Some(s) = v.get_mut(c) {
                *s = x;
            }
        }
        put(&mut self.mem.medium, c, 0.0);
        put(&mut self.surprise, c, 0.0);
        put(&mut self.surprise_slow, c, 0.0);
        put(&mut self.err_fast, c, 0.0);
        put(&mut self.err_slow, c, 0.0);
        put(&mut self.learn_gain, c, 1.0);
        put(&mut self.read_min, c, 0.0);
        put(&mut self.read_max, c, 0.0);
        put(&mut self.read_count, c, 0);
        put(&mut self.cell_mean, c, 0.0);
        put(&mut self.cell_var, c, 0.0);
        put(&mut self.cell_sens2, c, 0.0);
        put(&mut self.changed_at, c, self.clock);
        put(&mut self.values, c, f32::MAX); // "unknown", the graded-value sentinel
        put(&mut self.clamped, c, false);
        put(&mut self.drive, c, 0.0);
        put(&mut self.dwell_len, c, 1.0);
        put(&mut self.pressure, c, 0.0);
        if let Some(m) = self.reliability.get_mut(c) {
            *m = Reliability::default(); // a reused slot must not inherit stale mastery (no-op when unused)
        }
        let d = self.pos_dim;
        if d > 0 && (c + 1) * d <= self.pos.len() {
            for a in 0..d {
                self.pos[c * d + a] = 0.0;
            }
        }
        // Cell-keyed maps: the option ensemble / valence record, and every pair (product co-credit, fast-weight
        // binding) touching c.
        self.value_ensemble.remove(&c);
        self.valence.remove(&c);
        if !self.co_credit.is_empty() {
            self.co_credit.retain(|&(a, b), _| a != c && b != c);
        }
        if !self.target_credit.is_empty() {
            self.target_credit
                .retain(|&(t, a, b), _| t != c && a != c && b != c);
            self.target_ticks.remove(&c);
        }
        if !self.assoc.is_empty() {
            self.assoc.retain(|&(a, b), _| a != c && b != c);
        }
    }

    /// Whether a cell is pinned by a non-unit memory structure (a trace or delay reads/writes it), so `cull` must
    /// NOT reclaim its slot even when no unit references it.
    pub(super) fn cell_pinned_by_memory(&self, c: usize) -> bool {
        self.traces.iter().any(|t| t.cell == c || t.source == c)
            || self.delays.iter().any(|d| d.cell == c || d.source == c)
            || self
                .resonators
                .iter()
                .any(|r| r.cell == c || r.vcell == c || r.source == c)
    }
}

/// `v` after `n` steps of an EMA whose samples were all zero: `v · decay^n`, by repeated multiplication (the order
/// an accrued zero sample would apply it in, so the lazy decay matches the eager one bit for bit for small `n`).
fn decay_by(v: f32, decay: f32, n: u64) -> f32 {
    let mut v = v;
    for _ in 0..n {
        if v == 0.0 {
            break;
        }
        v *= decay;
    }
    v
}
