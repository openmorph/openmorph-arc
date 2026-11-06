// SPDX-License-Identifier: MIT-0
//! `validate` — the opt-in structural check of a deserialized (untrusted) snapshot.

use super::*;

impl Field {
    /// VALIDATE the structural invariants of a DESERIALIZED field — the opt-in defence for an UNTRUSTED
    /// snapshot (security review C1/PY1). `serde` reconstructs a [`Field`] from raw bytes WITHOUT checking
    /// that a unit's reader/writer indices point at cells that exist, that its parallel `weights`/`reads`
    /// arrays agree in length, or that a `gate`/`factor`/adjacency index is in range. A crafted snapshot
    /// with a hostile index is memory-SAFE (Rust bounds-checks; the settle hot path is `.get()`-guarded)
    /// but will PANIC on a later call (e.g. `stimulate` enqueuing an out-of-range cell's readers, or a
    /// trace writing a cell past the medium). That is a robustness / denial-of-service gap.
    ///
    /// Call this ONCE, right after deserializing bytes you do not control (`let f: Field =
    /// serde_json::from_slice(bytes)?; f.validate()?;`). It is deliberately NOT run inside `Deserialize` —
    /// wiring it into the load path would change how a `Field` deserializes (and the physics engine's
    /// bit-for-bit oracle depends on that path being untouched). A trusted, self-produced snapshot need
    /// not call it. Returns [`OpenMorphError::InvalidSnapshot`](crate::OpenMorphError::InvalidSnapshot) describing the FIRST violation found:
    ///
    /// - adjacency length: `readers`/`writers` must be parallel to the medium (`== n_cells`);
    /// - per unit: `weights.len() == reads.len()`, and every `reads` / `write` / `gate` / `factor` cell
    ///   index `< n_cells`;
    /// - every unit id listed in a `readers[c]` / `writers[c]` bucket `< units.len()`;
    /// - `traces` / `delays` / `resonators`: their `cell` / `source` / `vcell` indices `< n_cells`;
    /// - a NESTED unit's `sources` / `sink` are in range of its sub-field, every memo entry holds one output
    ///   per sink (`1 + extra.len()`), and the sub-field itself validates (recursively — a fractal cell is a
    ///   whole `Field`);
    /// - the plasticity controller's `err_fast` / `err_slow` / `learn_gain` are parallel (sized together).
    ///
    /// The per-unit HERITABLE trait vectors (`energy`/`scale`/`lr`/…) are intentionally NOT length-checked:
    /// they are `#[serde(default)]` and lazily (re)sized by `hydrate_traits`, so a legitimately-empty or
    /// short vector on an old snapshot is not corruption.
    pub fn validate(&self) -> Result<(), crate::OpenMorphError> {
        let n = self.mem.medium.len();
        let nu = self.units.len();
        let bad = |m: String| Err(crate::OpenMorphError::InvalidSnapshot(m));

        // Adjacency must be parallel to the medium (indexed by cell in the settle/enqueue path).
        if self.readers.len() != n {
            return bad(format!(
                "readers adjacency length {} != n_cells {}",
                self.readers.len(),
                n
            ));
        }
        if self.writers.len() != n {
            return bad(format!(
                "writers adjacency length {} != n_cells {}",
                self.writers.len(),
                n
            ));
        }

        // Per-unit index/length invariants.
        for (i, u) in self.units.iter().enumerate() {
            if u.weights.len() != u.reads.len() {
                return bad(format!(
                    "unit {i}: weights length {} != reads length {}",
                    u.weights.len(),
                    u.reads.len()
                ));
            }
            for &c in &u.reads {
                if c >= n {
                    return bad(format!("unit {i}: reads cell {c} >= n_cells {n}"));
                }
            }
            if u.write >= n {
                return bad(format!("unit {i}: write cell {} >= n_cells {n}", u.write));
            }
            if let Some(g) = u.gate {
                if g >= n {
                    return bad(format!("unit {i}: gate cell {g} >= n_cells {n}"));
                }
            }
            if let Some(f) = u.factor {
                if f >= n {
                    return bad(format!("unit {i}: factor cell {f} >= n_cells {n}"));
                }
            }
            if let Some(nested) = &u.nested {
                let sub_n = nested.sub.mem.medium.len();
                for &s in &nested.sources {
                    if s >= sub_n {
                        return bad(format!(
                            "unit {i}: nested source cell {s} >= sub n_cells {sub_n}"
                        ));
                    }
                }
                if nested.sink >= sub_n {
                    return bad(format!(
                        "unit {i}: nested sink cell {} >= sub n_cells {sub_n}",
                        nested.sink
                    ));
                }
                // A memo hit is indexed per output (`fire_nested`), so a short cached vector would panic.
                let outs = 1 + nested.extra.len();
                if let Some(o) = nested.memo.values().find(|o| o.len() != outs) {
                    return bad(format!(
                        "unit {i}: nested memo entry holds {} outputs, expected {outs}",
                        o.len()
                    ));
                }
                // A fractal cell IS a Field — validate it the same way, recursively.
                nested.sub.validate()?;
            }
        }

        // Every unit id in an adjacency bucket must reference a real unit (else `queued[u]` panics).
        for (c, bucket) in self.readers.iter().enumerate() {
            for &id in bucket {
                if id >= nu {
                    return bad(format!("readers[{c}] lists unit {id} >= units.len() {nu}"));
                }
            }
        }
        for (c, bucket) in self.writers.iter().enumerate() {
            for &id in bucket {
                if id >= nu {
                    return bad(format!("writers[{c}] lists unit {id} >= units.len() {nu}"));
                }
            }
        }

        // The event-driven side channels (refreshed once per event) write cells directly — out-of-range
        // indices panic in `tick_traces`/`advance_delays`/`tick_resonators`, so check them too.
        for (i, t) in self.traces.iter().enumerate() {
            if t.cell >= n || t.source >= n {
                return bad(format!(
                    "trace {i}: cell {} / source {} out of range (n_cells {n})",
                    t.cell, t.source
                ));
            }
        }
        for (i, d) in self.delays.iter().enumerate() {
            if d.cell >= n || d.source >= n {
                return bad(format!(
                    "delay {i}: cell {} / source {} out of range (n_cells {n})",
                    d.cell, d.source
                ));
            }
        }
        for (i, r) in self.resonators.iter().enumerate() {
            if r.cell >= n || r.vcell >= n || r.source >= n {
                return bad(format!(
                    "resonator {i}: cell {} / vcell {} / source {} out of range (n_cells {n})",
                    r.cell, r.vcell, r.source
                ));
            }
        }

        // The plasticity controller is resized as ONE (keyed on `learn_gain`), then indexed in lock-step.
        let g = self.learn_gain.len();
        if self.err_fast.len() != g || self.err_slow.len() != g {
            return bad(format!(
                "plasticity controller lengths differ: err_fast {} / err_slow {} / learn_gain {g}",
                self.err_fast.len(),
                self.err_slow.len()
            ));
        }

        // The pending-refire set (serialized since the staleness refresh): a crafted one panics `drain`.
        if let Err(m) = self.dirty.check(nu) {
            return bad(m);
        }
        Ok(())
    }
}
