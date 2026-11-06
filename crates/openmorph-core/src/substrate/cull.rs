// SPDX-License-Identifier: MIT-0
//! Death and compaction — `cull`, `cull_after`, `detach`, and the unit compaction behind them.

use super::*;

impl Field {
    /// CULL the starved — units with energy `<= death`. DEATH closes the alive loop (birth grows,
    /// metabolism selects, death prunes), so the population tracks what is actually useful with no
    /// tick. Rebuilds the reader adjacency for the survivors.
    #[doc(hidden)]
    pub fn cull(&mut self, death: f32) -> usize {
        if self.frozen {
            return 0;
        }
        self.cull_after(death, 0)
    }

    /// COMPACT the unit vector and EVERY parallel trait array (energy/born_at/plasticity/scale/lr/mut_rate/credit/
    /// momentum/velocities) down to the `keep` survivors, dropping any `orphaned` cell from each survivor's
    /// reads/weights (and, when `clear_factor_on_orphan`, its forward factor — an orphaned cell is never recomputed,
    /// so a unit still multiplying by it would be frozen against a dead value), then rebuild the reader/writer
    /// adjacency for the new indices. This is the SHARED spine of `cull_after` and `detach` — kept in ONE place so a
    /// trait vector can never again be forgotten in one copy and silently misalign with its unit (the momentum/
    /// velocity class of bug the 2nd-order work exposed). Byte-replay-safe: the drain order is preserved verbatim, so
    /// both callers are byte-identical to their former inline bodies (the ONE difference — cull clears orphaned
    /// factors, detach did not — is the explicit `clear_factor_on_orphan` flag). The differing tails (cull reclaims
    /// orphaned cells to the free list; detach only zeros them) and the `keep`/`orphaned` computation stay in each caller.
    fn compact_units(
        &mut self,
        keep: &[bool],
        orphaned: &HashSet<usize>,
        clear_factor_on_orphan: bool,
    ) {
        // FAST PATH (perf, byte-identical): nobody dies and nothing is orphaned — the full compaction would copy
        // every unit and every trait verbatim (the traits are already `units.len()` long: both callers
        // `hydrate_traits` first), so its ONLY observable effect is the adjacency re-index. Skip the ten Vec
        // rebuilds (the per-event cost of the living pool's `cull_after` when nothing starved).
        if orphaned.is_empty() && keep.iter().all(|&k| k) {
            // …and when the adjacency is already canonical that re-index would reproduce it verbatim: skip it too.
            if !self.adj_canonical {
                self.reindex_adjacency();
            }
            #[cfg(debug_assertions)]
            self.debug_assert_canonical();
            return;
        }
        // INCREMENTAL adjacency (perf, byte-identical): with a canonical adjacency, `readers[o]` lists EXACTLY the
        // units reading `o`, so the survivors to filter are found there (no scan of every survivor's reads), and
        // the re-indexed adjacency is the old one with dead ids dropped and the rest shifted down (a monotone
        // remap keeps each bucket ascending) — a streaming pass per bucket instead of a scattered rebuild.
        let n_before = self.units.len();
        // `detach` keeps a forward factor that points at an orphaned cell (its reader edge must survive), so its
        // orphan buckets are not simply cleared — take the full re-index there (detach is rare, cull is hot).
        let incremental = self.adj_canonical
            && keep.len() == n_before
            && (clear_factor_on_orphan || orphaned.is_empty())
            && orphaned.iter().all(|&o| o < self.readers.len());
        let reads_orphan: Vec<bool> = if incremental && !orphaned.is_empty() {
            let mut m = vec![false; n_before];
            for &o in orphaned {
                for &id in &self.readers[o] {
                    m[id] = true;
                }
            }
            m
        } else {
            Vec::new()
        };
        // Orphan membership as a dense per-cell marker instead of a per-read `HashSet::contains` (SipHash per
        // survivor read was the dominant cost of a churning `alive` loop). Same membership ⇒ byte-identical.
        // (An out-of-medium index — only possible in a corrupt, unvalidated snapshot — falls back to the set.)
        let n_cells = self.mem.medium.len();
        let orphan_mark: Vec<bool> = if orphaned.is_empty() {
            Vec::new()
        } else {
            let mut m = vec![false; n_cells];
            for &c in orphaned {
                if c < n_cells {
                    m[c] = true;
                }
            }
            m
        };
        let is_orphan = |c: usize| {
            if c < n_cells {
                orphan_mark.get(c).copied().unwrap_or(false)
            } else {
                orphaned.contains(&c)
            }
        };
        let n_keep = keep.iter().filter(|&&k| k).count();
        let mut new_units = Vec::with_capacity(n_keep);
        let (mut ne, mut nb, mut np, mut ns, mut nl, mut nm, mut nc, mut nmo, mut nve) = (
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
            Vec::with_capacity(n_keep),
        );
        let mut nrls = Vec::new(); // the RLS state, compacted only once it exists
        for (i, mut u) in self.units.drain(..).enumerate() {
            if keep[i] {
                if !orphaned.is_empty() {
                    // Drop any orphaned cell from this survivor's reads (and its parallel weight). Only a survivor
                    // that actually READS an orphaned cell (or whose weights are not parallel to its reads — the
                    // filter would truncate them) is rebuilt; any other survivor's reads/weights are exactly what
                    // the filter would reproduce, so the per-read copy is skipped (perf, byte-identical).
                    let reads_an_orphan = if incremental {
                        reads_orphan[i]
                    } else {
                        u.reads.iter().any(|&c| is_orphan(c))
                    };
                    if u.weights.len() != u.reads.len() || reads_an_orphan {
                        let (mut nr, mut nw) = (
                            Vec::with_capacity(u.reads.len()),
                            Vec::with_capacity(u.reads.len()),
                        );
                        for (k, &c) in u.reads.iter().enumerate() {
                            if !is_orphan(c) {
                                nr.push(c);
                                nw.push(u.weights[k]);
                            }
                        }
                        u.reads = nr;
                        u.weights = nw;
                    }
                    if clear_factor_on_orphan && u.factor.is_some_and(is_orphan) {
                        u.factor = None;
                    }
                }
                new_units.push(u);
                ne.push(self.energy[i]);
                nb.push(self.born_at[i]);
                np.push(self.plasticity[i]);
                ns.push(self.scale[i]);
                nl.push(self.lr[i]);
                nm.push(self.mut_rate[i]);
                nc.push(self.credit[i]);
                nmo.push(self.momentum.get(i).copied().unwrap_or(0.0));
                nve.push(
                    self.velocities
                        .get_mut(i)
                        .map(core::mem::take) // core:: not std:: — keeps the bare-metal (no_std) build compiling
                        .unwrap_or_default(),
                );
                if !self.rls.is_empty() {
                    nrls.push(self.rls.get_mut(i).map(core::mem::take).unwrap_or_default());
                }
            }
        }
        self.units = new_units;
        self.energy = ne;
        self.born_at = nb;
        self.plasticity = np;
        self.scale = ns;
        self.lr = nl;
        self.mut_rate = nm;
        self.credit = nc;
        self.momentum = nmo;
        self.velocities = nve;
        if !self.rls.is_empty() {
            self.rls = nrls;
        }
        // The stale flags follow their units.
        self.dirty.compact(keep);
        // unit indices changed
        if incremental {
            // new id of each survivor = old id − dead ids below it (monotone ⇒ buckets stay ascending)
            let mut new_id = vec![0usize; n_before];
            let mut next = 0usize;
            for (i, &k) in keep.iter().enumerate() {
                new_id[i] = next;
                if k {
                    next += 1;
                }
            }
            let first_dead = keep.iter().position(|&k| !k).unwrap_or(n_before);
            let remap = |bucket: &mut Vec<usize>| {
                // an ascending bucket whose largest id sits below the first dead id is unchanged
                if bucket.last().is_some_and(|&l| l >= first_dead) {
                    bucket.retain(|&id| keep[id]);
                    for id in bucket.iter_mut() {
                        *id = new_id[*id];
                    }
                }
            };
            for (c, bucket) in self.readers.iter_mut().enumerate() {
                if !orphaned.is_empty() && is_orphan(c) {
                    bucket.clear(); // every reader of an orphan died or had it filtered from its reads
                } else {
                    remap(bucket);
                }
            }
            for bucket in self.writers.iter_mut() {
                remap(bucket);
            }
            self.adj_canonical = true;
        } else {
            self.reindex_adjacency();
        }
        #[cfg(debug_assertions)]
        self.debug_assert_canonical();
    }

    /// Debug-build check of the [`Self::adj_canonical`] invariant: the adjacency equals a fresh re-index of
    /// `units`. Fires if a caller mutated the `pub units[i].reads`/`write` directly (bypassing the adjacency).
    #[cfg(debug_assertions)]
    fn debug_assert_canonical(&self) {
        let mut r: Vec<Vec<usize>> = vec![Vec::new(); self.readers.len()];
        let mut w: Vec<Vec<usize>> = vec![Vec::new(); self.writers.len()];
        for (id, u) in self.units.iter().enumerate() {
            push_unit_edges(&mut r, &mut w, id, u);
        }
        debug_assert!(
            r == self.readers && w == self.writers,
            "Field adjacency diverged from `units` (was `units[i].reads`/`write` mutated directly?)"
        );
    }

    /// Re-index the reader/writer adjacency IN PLACE from `units`, keeping the adjacency's current length: the
    /// canonical form — each bucket lists unit ids ascending, a unit once per read occurrence plus its forward-factor
    /// edge ([`push_unit_edges`]). Shared by both [`Self::compact_units`] paths; allocation-free once warm (the
    /// buckets keep their capacity).
    fn reindex_adjacency(&mut self) {
        for r in self.readers.iter_mut() {
            r.clear();
        }
        for w in self.writers.iter_mut() {
            w.clear();
        }
        for (id, u) in self.units.iter().enumerate() {
            push_unit_edges(&mut self.readers, &mut self.writers, id, u);
        }
        self.adj_canonical = true;
    }

    /// CULL, but SPARE newborns AND consolidated skills. A unit younger than `grace` events is never
    /// culled (the birth reserve / "yolk": a young unit needs development time to grow useful before the
    /// economy judges it). A CONSOLIDATED unit (plasticity 0) is also never culled: a banked skill is a
    /// permanent asset, not subject to use-it-or-lose-it — it must survive DORMANCY (when another module
    /// is the active one it neither fires nor earns), or the library would forget by DEATH, not just by
    /// overwriting. So `consolidate` protects a skill from both later learning and starvation. (`cull` is
    /// `grace` 0.) Uses the organism's own subjective time ([`Self::clock`]).
    #[doc(hidden)]
    pub fn cull_after(&mut self, death: f32, grace: u64) -> usize {
        if self.frozen {
            return 0;
        }
        self.hydrate_traits(); // safe against a freshly-loaded snapshot (see hydrate_traits)
        let keep: Vec<bool> = self
            .energy
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                e > death
                    || self.clock.saturating_sub(self.born_at[i]) < grace
                    || self.plasticity[i] == 0.0
            })
            .collect();
        // Cells this cull ORPHANS — written by a culled unit and by NO survivor. A reader still listing an
        // orphaned cell would read a STALE value forever (nothing recomputes the cell), so the readout would
        // keep dragging a dead feature. Detach them (drop from every reader's reads/weights, clear the value)
        // so `cull` is a CLEAN capacity regularizer: a starved (low-`credit_of`) feature can be pruned without
        // corrupting the readout (#128). Raw input cells keep no writer legitimately and are never orphaned
        // here (they were not written by a culled unit).
        // FAST PATH (perf, byte-identical): nobody starved ⇒ `orphaned` (the writes of CULLED units) is empty, so
        // skip building it; `compact_units` then only re-indexes the adjacency, exactly as the full path did.
        if keep.iter().all(|&k| k) {
            self.compact_units(&keep, &HashSet::new(), true);
            return 0;
        }
        let orphaned: HashSet<usize> = {
            // Kept-writer membership as a dense per-cell marker (was a SipHash `HashSet` over every survivor). An
            // out-of-medium write (only in a corrupt, unvalidated snapshot) falls back to a linear list.
            let n_cells = self.mem.medium.len();
            let mut written_by_kept = vec![false; n_cells];
            let mut kept_oob: Vec<usize> = Vec::new();
            for (i, u) in self.units.iter().enumerate() {
                if keep[i] {
                    if u.write < n_cells {
                        written_by_kept[u.write] = true;
                    } else {
                        kept_oob.push(u.write);
                    }
                }
            }
            let kept_writes = |w: usize| {
                if w < n_cells {
                    written_by_kept[w]
                } else {
                    kept_oob.contains(&w)
                }
            };
            self.units
                .iter()
                .enumerate()
                .filter(|(i, u)| !keep[*i] && !kept_writes(u.write))
                .map(|(_, u)| u.write)
                .collect()
        };
        let before = self.units.len();
        self.compact_units(&keep, &orphaned, true); // cull clears orphaned forward-factors
                                                    // RECLAIM each orphaned cell (deploy wall, MorphLab #149): a cell whose unit died and that nothing
                                                    // references any more is reset and returned to the free list, so `birth_feature` reuses the slot instead
                                                    // of growing the medium — the medium (+ its per-cell vectors + this O(cells) adjacency rebuild) stays
                                                    // bounded at the PEAK concurrent cell count rather than leaking O(steps). A cell pinned by a trace/delay is
                                                    // kept (its stale value only zeroed).
                                                    // Iterate a SORTED copy: `orphaned` is a hash set (non-deterministic order under std), and the push
                                                    // order sets which slot a future feature reuses — sorting keeps the free list, and thus the whole run,
                                                    // byte-replayable (the determinism moat).
        let mut reclaim: Vec<usize> = orphaned.iter().copied().collect();
        reclaim.sort_unstable();
        for &c in &reclaim {
            if c < self.mem.medium.len()
                && self.readers[c].is_empty()
                && self.writers[c].is_empty()
                && !self.cell_pinned_by_memory(c)
            {
                self.reset_cell(c); // zeros the value + all per-cell parallel state
                self.free_cells.push(c);
            } else if let Some(v) = self.mem.medium.get_mut(c) {
                *v = 0.0;
            }
        }
        before - self.units.len()
    }

    /// DETACH — remove every unit writing any of `cells` from the active graph, UNCONDITIONALLY (the targeted
    /// counterpart of [`Self::cull`], which removes by starvation). Cleanly re-indexes and orphan-detaches
    /// exactly like `cull` (a cell left with no writer is dropped from every reader and its value cleared, so no
    /// reader drags a stale output). Use it to SHED a banked subgraph after [`Self::encapsulate`]: encapsulate
    /// COPIES the subgraph into a sub-morph, `detach` removes the originals, [`Self::nest`] plants the brick — so
    /// a consolidated skill collapses from many active units to ONE nested cell (sleep-consolidation / the
    /// fractal collapse). Returns the number of units removed. No-op when frozen. O(units·reads).
    pub fn detach(&mut self, cells: &[usize]) -> usize {
        if self.frozen {
            return 0;
        }
        self.hydrate_traits();
        let target: HashSet<usize> = cells.iter().copied().collect();
        let keep: Vec<bool> = self
            .units
            .iter()
            .map(|u| !target.contains(&u.write))
            .collect();
        let orphaned: HashSet<usize> = {
            let written_by_kept: HashSet<usize> = self
                .units
                .iter()
                .enumerate()
                .filter(|(i, _)| keep[*i])
                .map(|(_, u)| u.write)
                .collect();
            let mut o: HashSet<usize> = self
                .units
                .iter()
                .enumerate()
                .filter(|(i, u)| !keep[*i] && !written_by_kept.contains(&u.write))
                .map(|(_, u)| u.write)
                .collect();
            // A cell NO unit writes is droppable too. The energy economy can only judge what the
            // organism PRODUCES — an input channel the world feeds it has no writer, earns no reliance
            // and can never be culled, however dead it is. This is the one way to prune one.
            o.extend(
                target
                    .iter()
                    .copied()
                    .filter(|&c| self.writers.get(c).is_none_or(|w| w.is_empty())),
            );
            o
        };
        let before = self.units.len();
        self.compact_units(&keep, &orphaned, false); // detach historically does NOT clear orphaned factors
        for &c in &orphaned {
            if let Some(v) = self.mem.medium.get_mut(c) {
                *v = 0.0;
            }
        }
        before - self.units.len()
    }
}
