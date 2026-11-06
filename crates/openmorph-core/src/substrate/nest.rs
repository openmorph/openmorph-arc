// SPDX-License-Identifier: MIT-0
//! The fractal cell — `encapsulate`, `nest`/`nest_multi`, `bank_skill`, and the memoized nested firing.

use super::*;

/// The SOURCE cells of a set of units — an input of some unit but written by none (the inputs), ascending. An
/// input is a read OR a forward [`Unit::factor`]: a product term over a raw cell must be fed that cell by the
/// host, or it multiplies by 0.
fn source_cells_of(units: &[Unit]) -> Vec<usize> {
    let written: alloc::collections::BTreeSet<usize> = units.iter().map(|u| u.write).collect();
    let mut src: Vec<usize> = units
        .iter()
        .flat_map(|u| u.reads.iter().copied().chain(u.factor))
        .filter(|c| !written.contains(c))
        .collect();
    src.sort_unstable();
    src.dedup();
    src
}

/// The SINK cells of a set of units — written by some unit but an input (read or factor) of none (the outputs),
/// ascending.
fn sink_cells_of(units: &[Unit]) -> Vec<usize> {
    let read: alloc::collections::BTreeSet<usize> = units
        .iter()
        .flat_map(|u| u.reads.iter().copied().chain(u.factor))
        .collect();
    let mut sink: Vec<usize> = units
        .iter()
        .map(|u| u.write)
        .filter(|c| !read.contains(c))
        .collect();
    sink.sort_unstable();
    sink.dedup();
    sink
}

impl Field {
    /// Fire a NESTED unit: settle its sub-morph on the read values and return the sub's `sink`. Recursive —
    /// if the sub contains nested units, THEY settle their subs, and so on (fractal depth). The sub is taken
    /// out to be settled (`&mut`) then restored.
    pub(super) fn fire_nested(&mut self, u: usize) -> f32 {
        let mut nst = self.units[u]
            .nested
            .take()
            .expect("fire_nested on a leaf unit");
        // MEMO: a frozen brick's output is fixed by its input (to within `settle_eps` — a hit replays the FIRST
        // settle, see `Nested::memo`) — serve a recurring input from cache.
        // The cached value is the FULL output vector: `[sink, extra[0].sink, extra[1].sink, …]`. The key (the read
        // values' bit patterns) is built in a reusable scratch buffer and looked up as a slice, and a hit is read in
        // place — a memo HIT allocates nothing (it used to clone the reads, the values, the key and the cached
        // vector). The input values are recovered from the key with `from_bits` (exact), so a miss settles on the
        // very same bits.
        let mut key = core::mem::take(&mut self.memo_key);
        key.clear();
        key.extend(
            self.units[u]
                .reads
                .iter()
                .map(|&c| self.mem.medium.get(c).copied().unwrap_or(0.0).to_bits()),
        );
        let out0 = if let Some(outs) = nst.memo.get(key.as_slice()) {
            self.write_nested_extras(&nst.extra, outs);
            outs[0]
        } else {
            let ins: Vec<(usize, f32)> = nst
                .sources
                .iter()
                .copied()
                .zip(key.iter().map(|&b| f32::from_bits(b)))
                .collect();
            nst.sub.stimulate(&ins, nst.budget);
            let mut o = Vec::with_capacity(1 + nst.extra.len());
            o.push(nst.sub.cells().get(nst.sink).copied().unwrap_or(0.0));
            for &(sink, _host) in &nst.extra {
                o.push(nst.sub.cells().get(sink).copied().unwrap_or(0.0));
            }
            // (the memo lives in the taken-out `nst`, the extras in the host medium — independent, so writing the
            // extras before caching is the same as the historical cache-then-write order)
            self.write_nested_extras(&nst.extra, &o);
            let out0 = o[0];
            if nst.memo.len() >= self.config.memo_cap {
                nst.memo.clear();
            }
            nst.memo.insert(key.clone(), o);
            self.nested_settles += 1; // a real sub-settle happened (a cache MISS) — for measuring the win
            out0
        };
        self.memo_key = key;
        self.units[u].nested = Some(nst);
        out0
    }

    /// Write a nested cell's ADDITIONAL vector outputs (`outs[1..]`) to their host cells (the PRIMARY `outs[0]`
    /// is written by the caller via `fire_nested`'s return value, so it flows through the same factor /
    /// settle_eps / enqueue path as a leaf unit).
    fn write_nested_extras(&mut self, extra: &[(usize, usize)], outs: &[f32]) {
        for (i, &(_sink, host)) in extra.iter().enumerate() {
            let v = outs[i + 1];
            if host < self.mem.medium.len()
                && (v - self.mem.medium[host]).abs() > self.config.settle_eps
            {
                self.mem.medium[host] = v;
                self.changed_at[host] = self.clock;
                self.note_value(host, v);
                self.enqueue_readers(host);
            }
        }
    }

    /// Cumulative count of nested-cell SUB-SETTLES that actually ran (cache misses) — the memo turns a recurring
    /// input into a cache hit, so this counts distinct-input work, not fire count. For measuring reuse cost.
    pub fn nested_settles(&self) -> u64 {
        self.nested_settles
    }

    // ── composition ──
    /// ENCAPSULATE a grown subgraph into its own MORPH — the reusable node. Given the output `cells` of a skill
    /// (a grown transducer / concept / policy), copies the units that produce them — the unit writing each output
    /// AND its whole transitive FEEDING BACKBONE (every unit that, following `reads` and forward `factor`s
    /// backward, contributes to an output) — into a fresh sub-[`Field`], nesting and all. The closure stops at
    /// cells no unit writes: those are the sub's own SOURCE cells, its inputs. So the sub is a COMPLETE,
    /// self-contained computation that reproduces the skill EXACTLY (naming only the output cells is enough — the
    /// grown backbone is invisible to the caller, so it must be pulled in automatically). Each lifted unit keeps
    /// everything it carries into the settle and the learn path: its `nested` sub, `activation`, `factor` (wired
    /// as a reader edge, exactly as [`Self::set_factor`] does), and its heritable traits (`scale`, `lr`,
    /// `mut_rate`, `momentum`, `plasticity`), under the parent's genome. Its write-`gate` travels when the gate
    /// cell lives INSIDE the skill (a lifted unit writes it, or it is one of the skill's inputs); a gate on an
    /// outside cell is dropped — the sub could never see that cell, and a gate stuck at 0 would freeze the unit
    /// for good — so it learns ungated, as a lifted unit always did. Bank a skill; the sub-morph is a single
    /// addressable node you [`Self::nest`] into any host, and it's serializable — share it over the mycelium.
    /// (Replaces the old crystallize/skill machinery: a skill is just a Morph.)
    pub fn encapsulate(&self, cells: &[usize]) -> Field {
        // Close the FEEDING BACKBONE transitively. A skill's requested output cells are computed THROUGH
        // intermediate cells (often grown, so NOT caller-visible), so we must pull in every unit that
        // transitively feeds an output — follow each included unit's inputs (its `reads` AND its forward
        // `factor`) backward and add whatever writes them, to a fixpoint — or the sub would read cells nothing in
        // it computes and settle to a DIFFERENT function. Cells that no unit writes are the sub's own SOURCE cells
        // (its inputs = its natural boundary), so this "full" closure is already bounded there; it never pulls in
        // more than what actually computes the outputs. (Filtering on `write ∈ cells` alone dropped the backbone
        // — a silent wrong-function bug.) O(units + edges): a cell is visited once (`seen`) and its writers come
        // from the `writers` adjacency, not a scan of every unit per cell.
        let n = self.mem.medium.len();
        let mut seen: Vec<bool> = vec![false; n];
        let mut want: Vec<usize> = Vec::with_capacity(cells.len());
        for &c in cells {
            if c < n && !seen[c] {
                seen[c] = true;
                want.push(c);
            }
        }
        let mut keep: Vec<bool> = vec![false; self.units.len()];
        while let Some(c) = want.pop() {
            for &i in self.writers.get(c).into_iter().flatten() {
                if keep[i] {
                    continue;
                }
                keep[i] = true;
                let u = &self.units[i];
                for r in u.reads.iter().copied().chain(u.factor) {
                    if r < n && !seen[r] {
                        seen[r] = true;
                        want.push(r);
                    }
                }
            }
        }
        // Preserve the parent's unit order (deterministic; matches the old `filter` order over `self.units`).
        let kept: Vec<usize> = (0..self.units.len()).filter(|&i| keep[i]).collect();
        // The cells the skill lives on — every input (read or factor) and write of a lifted unit. Size the sub to
        // them (a factor cell too, or it would read — and `validate` — out of range), and carry a gate only if its
        // cell is one of them (the closure is done with `seen`, so its buffer is reused as the marker).
        let mut inside = seen;
        inside.fill(false);
        let mut max_cell = 0;
        for &i in &kept {
            let u = &self.units[i];
            for c in u
                .reads
                .iter()
                .copied()
                .chain(u.factor)
                .chain(core::iter::once(u.write))
            {
                max_cell = max_cell.max(c + 1);
                if c < n {
                    inside[c] = true;
                }
            }
        }
        let mut sub = Field::new(max_cell);
        // A banked skill keeps the genome it was grown under: the sub is settled on its own `Config`, so the
        // whole point of encapsulation (this subgraph computes what it computed) needs the laws too. Set it
        // BEFORE the units are added, so anything `add_unit` seeds from the genome is seeded from THIS one.
        sub.config = self.config.clone();
        for &i in &kept {
            let u = &self.units[i];
            let id = sub.add_unit(u.reads.clone(), u.weights.clone(), u.bias, u.write);
            sub.units[id].nested = u.nested.clone(); // carry a nested cell verbatim (fractal encapsulation)
            sub.units[id].activation = u.activation; // a unit's own law travels with it, or the skill
                                                     // would silently revert to the genome's default
            if let Some(f) = u.factor {
                // Its forward factor (the second-order term) — AND the reader edge `set_factor` wires, or a
                // factor-only change would never re-fire the unit and the sub would settle on a stale product.
                sub.units[id].factor = Some(f);
                if !sub.readers[f].contains(&id) {
                    sub.readers[f].push(id);
                }
            }
            // Its write-gate — learning only, so not a dependency the closure follows: kept iff its cell is inside.
            sub.units[id].gate = u.gate.filter(|&g| g < n && inside[g]);
            // Its heritable traits: `scale` is the unit's live projection gain (it enters the settle), and
            // `lr`/`mut_rate`/`momentum`/`plasticity` are what a sub that learns (`nest_credit`) or breeds inherits.
            // `get` + the genome default = what `hydrate_traits` would fill for a not-yet-hydrated snapshot.
            sub.scale[id] = self.scale.get(i).copied().unwrap_or(self.config.init_scale);
            sub.lr[id] = self.lr.get(i).copied().unwrap_or(self.config.init_lr);
            sub.mut_rate[id] = self
                .mut_rate
                .get(i)
                .copied()
                .unwrap_or(self.config.init_mut_rate);
            sub.momentum[id] = self
                .momentum
                .get(i)
                .copied()
                .unwrap_or(self.config.init_momentum);
            sub.plasticity[id] = self.plasticity.get(i).copied().unwrap_or(1.0);
        }
        sub
    }

    /// NEST a sub-morph as a CELL — plant `sub` as a single node in this host, reading `reads` (mapped to the
    /// sub's source cells in ascending order) and writing `write`, settled with `budget` per firing. Reading
    /// that cell settles the sub-morph — recursion in the ACTIVE path. This is the ONE composition operator:
    /// a banked skill becomes a first-class node a higher morph routes to, and the sub may itself hold nested
    /// cells, so composition covers the full recursive tree (not just flat pairs). Returns the new unit index.
    /// (Replaces graft / graft_at / graft_skill / graft_skill_at / Skill::apply — reuse is nesting.)
    ///
    /// # Example
    /// ```
    /// use openmorph_core::Field;
    /// // A leaf sub-morph: s1(x, y) = tanh(x - y). Cells 0=x, 1=y, 2=out.
    /// let mut s1 = Field::new(3);
    /// s1.add_unit(vec![0, 1], vec![1.0, -1.0], 0.0, 2);
    /// // Plant it as ONE cell in a host: host cell 2 = s1(host[0], host[1]), fired in the host's cascade.
    /// let mut host = Field::new(2);
    /// host.nest(s1, &[0, 1], 2, 20); // host cell 2 IS the sub-morph, fired in the cascade
    /// host.stimulate(&[(0, 0.5), (1, -0.5)], 20);
    /// assert!(host.cells()[2].abs() <= 1.0); // its settled tanh output
    /// ```
    pub fn nest(&mut self, sub: Field, reads: &[usize], write: usize, budget: usize) -> usize {
        self.nest_multi(sub, reads, &[write], budget)
    }

    /// MULTI-OUTPUT NEST — plant a sub-morph as a node exposing its FULL VECTOR output. The sub's sink cells
    /// (`sink_cells_of`, ascending) are mapped POSITIONALLY to `writes`: `writes[0]` is the primary output (the
    /// owning unit's cell, fired in the host cascade exactly like a scalar nest), and each further `writes[i]`
    /// receives the sub's `i`-th sink on the SAME settle. This is what lets a nested cell be a real SUB-MORPH (a
    /// vector transducer) rather than a scalar feature — a morph whose output is `f: D → V` with `|V|>1` can now
    /// be composed. Extra sinks beyond `writes.len()` are dropped; `writes` beyond the sub's sinks stay 0. With a
    /// single write it is exactly [`Self::nest`] (byte-identical scalar node). Returns the nested unit id.
    pub fn nest_multi(
        &mut self,
        sub: Field,
        reads: &[usize],
        writes: &[usize],
        budget: usize,
    ) -> usize {
        debug_assert!(
            !writes.is_empty(),
            "nest_multi called with no write cells — primary_write silently falls back to cell 0"
        );
        let sources = source_cells_of(&sub.units);
        let sinks = sink_cells_of(&sub.units);
        let primary_sink = sinks.first().copied().unwrap_or(0);
        let primary_write = writes.first().copied().unwrap_or(0);
        // Pair each ADDITIONAL sub sink with an additional host write cell (positional, min-length).
        let extra: Vec<(usize, usize)> = sinks
            .iter()
            .skip(1)
            .copied()
            .zip(writes.iter().skip(1).copied())
            .collect();
        // Make sure every host output cell exists in the medium (add_unit ensures the primary; extras too).
        for &w in writes {
            self.ensure_cells(w + 1);
        }
        // BEHAVIOUR-SEED the node's concept-map POSITION from the sub-morph's OWN embedding (#172) — its coarse
        // spectral prefix is a coarse position — computed BEFORE `sub` is moved. So a nested skill lands on the map
        // by WHAT IT DOES (unifying the positional router with the cosine retrieval key), not scatter noise.
        // `place_at` PROJECTS the key to the current `pos_dim` (truncate, never grow), so routing proofs that
        // `place_at` a 2-D grid key AFTER nest — and any explicit place_at — still override this seed. Gated by
        // `config.nest_seeds_position` (default on; off = strict byte-identity + skip the per-nest embed cost).
        let seed = self.config.nest_seeds_position.then(|| sub.embed());
        let id = self.add_unit(reads.to_vec(), vec![0.0; reads.len()], 0.0, primary_write);
        self.units[id].nested = Some(Box::new(Nested {
            sub,
            sources,
            sink: primary_sink,
            extra,
            budget,
            memo: HashMap::new(),
        }));
        if let Some(key) = seed {
            self.place_at(primary_write, &key); // truncates to pos_dim — projects, never grows the map
        }
        id
    }

    /// BANK a grown skill as a reusable FROZEN node — AUTONOMOUS NESTED GROWTH, the depth-building complement to
    /// scalar [`Self::grow`]. Growth alone only ever widens ONE layer (every born feature is a scalar projection,
    /// `nested: None`); this lets the organism, once a subgraph is relied-upon, bank it as a single addressable
    /// node so it gains a SECOND STOREY on its own. Encapsulates the whole feeding backbone of `output_cell`
    /// ([`Self::encapsulate`]), derives the sub's own SOURCE cells as its inputs, and plants it as a nested cell
    /// reading those sources — returned as a fresh banked cell whose value settles in its OWN sub-cascade.
    ///
    /// Because a nested cell is FROZEN and isolated, the banked skill is INTERFERENCE-PROOF: a later task grows a
    /// fresh readout OVER it instead of re-tuning (and degrading) it. Proof `nested_transfer` measures the
    /// difference — a flat/consolidated skill transfers at chance while the banked (nested) skill converts
    /// interference into transfer. The economy says WHICH subgraph earned banking (high [`Self::reliance_of`]);
    /// this is HOW to bank it. Assumes the skill's source cells carry its inputs in the host (true when banking a
    /// subgraph grown IN this field). Returns the banked cell.
    pub fn bank_skill(&mut self, output_cell: usize, budget: usize) -> usize {
        let sub = self.encapsulate(&[output_cell]);
        let sources = source_cells_of(&sub.units);
        let banked = self.len();
        self.ensure_cells(banked + 1);
        self.nest(sub, &sources, banked, budget);
        banked
    }
}
