// SPDX-License-Identifier: MIT-0
//! Graph construction and rewiring — `add_unit`, activation/factor/gate/connect setters, `merge`, and the
//! adjacency rebuild.

use super::*;

impl Field {
    /// Add a unit; returns its index. Registers it as a reader of each input cell.
    ///
    /// # Panics
    /// If `weights.len() != reads.len()` — one weight per read cell. Checked here, at the boundary, because a
    /// mismatch would otherwise surface much later as an out-of-bounds panic inside [`Self::learn_step`].
    pub fn add_unit(
        &mut self,
        reads: Vec<usize>,
        weights: Vec<f32>,
        bias: f32,
        write: usize,
    ) -> usize {
        assert!(
            weights.len() == reads.len(),
            "add_unit: weights.len() ({}) must equal reads.len() ({}) — one weight per read cell",
            weights.len(),
            reads.len()
        );
        let max_cell = reads
            .iter()
            .copied()
            .chain(core::iter::once(write))
            .max()
            .unwrap_or(0);
        self.ensure_cells(max_cell + 1);
        let id = self.units.len();
        for &c in &reads {
            self.readers[c].push(id);
        }
        self.writers[write].push(id);
        self.units.push(Unit {
            reads,
            weights,
            bias,
            write,
            nested: None,
            gate: None,
            activation: None,
            factor: None,
        });
        self.energy.push(self.config.birth_energy);
        self.born_at.push(self.clock);
        self.plasticity.push(1.0);
        self.scale.push(self.config.init_scale);
        self.lr.push(self.config.init_lr);
        self.mut_rate.push(self.config.init_mut_rate);
        self.momentum.push(self.config.init_momentum);
        self.velocities.push(Vec::new());
        if !self.rls.is_empty() {
            self.rls.push(RlsState::default()); // lock-step once any unit has RLS state
        }
        self.credit.push(0.0);
        id
    }

    /// SET the nonlinearity of every unit writing `cell`, overriding the field's genome
    /// ([`Config::activation`]); `None` restores it. For TRANSCRIBING a foreign network whose layers do not
    /// all share one law — an imported `Relu` hidden layer under an `Identity` logit layer. A grown organism
    /// never calls this: it births every unit under the one genome it carries.
    pub fn set_activation(&mut self, cell: usize, act: Option<Activation>) {
        if let Some(ids) = self.writers.get(cell).cloned() {
            for i in ids {
                self.units[i].activation = act;
            }
        }
    }

    /// WIRE a new connection: the unit writing `into` also reads `from`, with weight `w`.
    ///
    /// [`Self::grow`] adds a NODE; this adds an EDGE, and it is what lets an organism take on a shape its
    /// source never had. A transcribed network is a DAG because the format it came from can only express
    /// one — but a `Field` settles to a fixed point rather than running a pass, so an edge BACKWARD or
    /// SIDEWAYS (two cells of the same layer reading each other) is a legal structure here, and recurrent
    /// or laterally-coupled organisms are reachable from a feed-forward import.
    ///
    /// A cycle is only useful if it CONVERGES. Check [`Self::spectral_radius`] — above ~1 the loop
    /// amplifies and the settle will run to its budget instead of reaching a fixed point.
    /// Returns `false` if `into` has no writer, or if the edge already exists.
    pub fn connect(&mut self, from: usize, into: usize, w: f32) -> bool {
        self.ensure_cells(from.max(into) + 1);
        let Some(&i) = self.writers.get(into).and_then(|u| u.first()) else {
            return false;
        };
        if self.units[i].reads.contains(&from) {
            return false;
        }
        self.units[i].reads.push(from);
        self.units[i].weights.push(w);
        self.readers[from].push(i);
        self.adj_canonical = false; // an appended edge may break ascending bucket order
        true
    }

    /// SET the forward factor of every unit writing `cell` — its output becomes `cells[factor] · f(…)`
    /// (`None` removes it). The one way to make a unit's value depend on the PRODUCT of two live cells;
    /// see [`Unit::factor`] for why this is separate from [`Unit::gate`].
    pub fn set_factor(&mut self, cell: usize, factor: Option<usize>) {
        if let Some(f) = factor {
            self.ensure_cells(f + 1);
        }
        if let Some(ids) = self.writers.get(cell).cloned() {
            if !ids.is_empty() {
                self.adj_canonical = false; // an appended factor edge may break ascending bucket order
            }
            for i in ids {
                // The factor is a DEPENDENCY, so the unit must be a READER of it — otherwise a change to
                // the factor cell would not fire this unit and the cascade would settle on a stale
                // product. (Fire-on-change only knows what it is told; an invisible edge is a wrong
                // answer, not a slow one.) Every adjacency rebuild re-adds this edge (`push_unit_edges`).
                if let Some(old) = self.units[i].factor {
                    // drop only the FACTOR edge: a unit that also READS the old cell keeps that read edge
                    if !self.units[i].reads.contains(&old) {
                        if let Some(rs) = self.readers.get_mut(old) {
                            rs.retain(|&r| r != i);
                        }
                    }
                }
                self.units[i].factor = factor;
                if let Some(f) = factor {
                    if !self.readers[f].contains(&i) {
                        self.readers[f].push(i);
                    }
                }
            }
        }
    }

    /// GATE the WRITE of every unit writing any of `cells` by cell `gate` — the unit then learns only in
    /// proportion to `gate`'s value (clamped `[0,1]`): it writes while the gate is OPEN and is PROTECTED while
    /// closed. This is the salience/write-enable the additive unit lacked; a memory register is a unit gated by
    /// its store signal, so it captures a value while storing and holds it (the weight untouched) through
    /// distractors, read back by pinging its address. `gate` is a cell — drive it from surprise/reward/attention
    /// (grown), and learning becomes neuromodulated. Pass the same cell as gate to make a self-gating unit.
    pub fn set_gate(&mut self, cells: &[usize], gate: usize) {
        for &c in cells {
            if let Some(ids) = self.writers.get(c).cloned() {
                for i in ids {
                    self.units[i].gate = Some(gate);
                }
            }
        }
    }

    /// MERGE two cells discovered to be the SAME identity: collapse `drop` into `keep`. Every reference to
    /// `drop` — a unit's read (with its weight), write, gate, or factor, and any temporal/spectral memory
    /// structure's source/cell — is redirected onto `keep`; a read that becomes a DUPLICATE of `keep` FOLDS
    /// (its weight sums into the existing one) and a read that becomes a SELF-read (a unit reading its own
    /// write) is dropped; the two cells' concept-map positions AVERAGE and their accumulated `drive` sums;
    /// `drop` is reset and returned to the free list for reuse. Deterministic (no RNG) so replay is intact,
    /// and a no-op on equal / out-of-range ids.
    ///
    /// This is the native content-address COLLAPSE: after a soft [`Self::fingerprint_distance`]
    /// dedup finds near-duplicate sub-morphs (or a percept→identity binder finds the same object under a
    /// transform), `merge` makes the common node ONE — so `route()` to either old cell reaches the single
    /// surviving identity. The dual of [`Self::cull`] (which DROPS a starved cell): `merge` FUSES two equivalent
    /// ones without losing what read them.
    pub fn merge(&mut self, keep: usize, drop: usize) {
        let n = self.mem.medium.len();
        if self.frozen || keep == drop || keep >= n || drop >= n {
            return;
        }
        for u in self.units.iter_mut() {
            if u.gate == Some(drop) {
                u.gate = Some(keep);
            }
            if u.factor == Some(drop) {
                u.factor = Some(keep);
            }
            if u.write == drop {
                u.write = keep;
            }
            // Redirect reads drop→keep, folding a duplicate read of keep (sum weights) and dropping a self-read.
            let (mut nr, mut nw) = (
                Vec::with_capacity(u.reads.len()),
                Vec::with_capacity(u.reads.len()),
            );
            for (k, &c) in u.reads.iter().enumerate() {
                let c = if c == drop { keep } else { c };
                if c == u.write {
                    continue; // a merge-induced self-read: a unit cannot read the cell it writes
                }
                match nr.iter().position(|&x| x == c) {
                    Some(p) => nw[p] += u.weights[k],
                    None => {
                        nr.push(c);
                        nw.push(u.weights[k]);
                    }
                }
            }
            u.reads = nr;
            u.weights = nw;
        }
        // Redirect the temporal / spectral memory structures that source or write `drop`.
        for t in self.traces.iter_mut() {
            if t.cell == drop {
                t.cell = keep;
            }
            if t.source == drop {
                t.source = keep;
            }
        }
        for d in self.delays.iter_mut() {
            if d.cell == drop {
                d.cell = keep;
            }
            if d.source == drop {
                d.source = keep;
            }
        }
        for r in self.resonators.iter_mut() {
            if r.cell == drop {
                r.cell = keep;
            }
            if r.vcell == drop {
                r.vcell = keep;
            }
            if r.source == drop {
                r.source = keep;
            }
        }
        // Redirect fast-weight bind pairs (`assoc`), folding a collision.
        if !self.assoc.is_empty() {
            let redir = |c: usize| if c == drop { keep } else { c };
            let mut moved: Vec<((usize, usize), f32)> = Vec::new();
            self.assoc.retain(|&(a, b), &mut w| {
                if a == drop || b == drop {
                    let (na, nb) = (redir(a), redir(b));
                    if na != nb {
                        moved.push(((na, nb), w));
                    }
                    false
                } else {
                    true
                }
            });
            moved.sort_by_key(|&(k, _)| k); // fold in a fixed key order — deterministic despite HashMap iteration
            for (k, w) in moved {
                *self.assoc.entry(k).or_insert(0.0) += w;
            }
        }
        // Combine the two identities on the concept map (average) and their accumulated drive (sum).
        let d = self.pos_dim;
        if d > 0 && (keep + 1) * d <= self.pos.len() && (drop + 1) * d <= self.pos.len() {
            for a in 0..d {
                self.pos[keep * d + a] = 0.5 * (self.pos[keep * d + a] + self.pos[drop * d + a]);
            }
        }
        let dd = self.drive.get(drop).copied().unwrap_or(0.0);
        if let Some(dk) = self.drive.get_mut(keep) {
            *dk += dd;
        }
        if let Some(d) = self.drive.get_mut(drop) {
            *d = 0.0; // clear the freed cell's drive so a reused slot never inherits a stale set-point
        }
        // Free the dropped cell for reuse and rebuild the adjacency from the units (the source of truth).
        self.reset_cell(drop);
        if !self.free_cells.contains(&drop) {
            self.free_cells.push(drop);
        }
        self.rebuild_adjacency();
    }

    /// Rebuild the `readers`/`writers` adjacency from the units. Used after a structural rewire ([`Self::merge`])
    /// so the O(1) writer/reader lookups stay exact. O(units · fan-in), deterministic.
    fn rebuild_adjacency(&mut self) {
        let n = self.mem.medium.len();
        self.readers = vec![Vec::new(); n];
        self.writers = vec![Vec::new(); n];
        for (i, u) in self.units.iter().enumerate() {
            push_unit_edges(&mut self.readers, &mut self.writers, i, u);
        }
        self.adj_canonical = true; // same content a re-index at this length produces
    }

    /// Whether any unit writes `cell` — the O(1) writer lookup (mirror of the reader adjacency). Public so
    /// an observer (a UI, the trace Snapshot) can ask "is this a grown cell?" without scanning `units`.
    pub fn has_writer(&self, cell: usize) -> bool {
        self.writers.get(cell).is_some_and(|v| !v.is_empty())
    }
}
