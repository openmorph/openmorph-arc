// SPDX-License-Identifier: MIT-0
//! Introspection — the spectral radius and the behavioural / structural keys (`fingerprint`, `embed`,
//! `structural_embed`) with their distances.

use super::*;

/// Magnitude SPECTRUM of a real signal — the first `bins` DFT magnitude coefficients, normalised. Deterministic
/// and machine-independent; used by [`Field::fingerprint`]/[`Field::embed`] to turn a settled response into an
/// order/phase-invariant behavioural signature (a spectral representation, minimal in-core form). `bins` is
/// the behavioural-key resolution — `Config::spectrum_bins` (OM-11): a wider key discriminates near-identical
/// behaviours more finely at higher cost, and is a per-library retrieval CONTRACT (change it ⇒ re-embed the library).
fn magnitude_spectrum(x: &[f32], bins: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![0.0; bins];
    }
    let bins = bins.min(n);
    (0..bins)
        .map(|k| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (t, &v) in x.iter().enumerate() {
                let ang = -2.0 * core::f32::consts::PI * k as f32 * t as f32 / n as f32;
                re += v * ang.cos();
                im += v * ang.sin();
            }
            (re * re + im * im).sqrt() / n as f32
        })
        .collect()
}

impl Field {
    /// SPECTRAL RADIUS `ρ` of the local Jacobian `J = D·W` at the CURRENT cell state — the organism's own
    /// contraction-rate self-diagnostic, by **power iteration** (a Gelfand growth-rate estimate, robust to
    /// complex-conjugate dominant pairs). It is bare-metal: only sparse matrix–vector products and a norm,
    /// no linear-algebra dependency — *the same matvec machinery settling already runs*, so a morph can
    /// measure its own stability on a microcontroller.
    ///
    /// The linearized cell→cell operator is `W[wr_i, r] = scale_i · w_{i,k}` (the unit's contribution from
    /// read `r` to its write cell) and `D = diag(1 − m_c²)` are the `tanh` slopes at the current state, so
    /// `(J v)[c] = (1 − m_c²) · Σ_{units i writing c} scale_i · Σ_k w_{i,k} · v[reads_{i,k}]`.
    ///
    /// **Two things to know (design review E1/E2):**
    /// - Fire-on-change is a *Gauss–Seidel* sweep, so `ρ(J)<1` is a **conservative sufficient** signal for
    ///   settling, not the exact async criterion — measure it as a proxy, corroborate with the empirical
    ///   settle rate.
    /// - At a **saturated** fixed point `m→±1 ⇒ D→0 ⇒ ρ→0` *trivially*; the informative value is `ρ` along
    ///   the settling **transient** (unsaturated state), not at rest. Feedforward morphs are **nilpotent**
    ///   (`ρ=0`): `ρ` is a property of RECURRENCE (feedback cycles), which is exactly where settling can fail.
    ///
    /// `iters` power-iteration steps (≈64 is ample for small operators); nested sub-morph units are treated
    /// as opaque (skipped) in this first version — assemble their block Jacobian separately if needed.
    pub fn spectral_radius(&self, iters: usize) -> f32 {
        let n = self.mem.medium.len();
        if n == 0 {
            return 0.0;
        }
        let norm = |x: &[f32]| -> f32 { x.iter().map(|a| a * a).sum::<f32>().sqrt() };
        // Deterministic, generic seed vector (no external RNG) — a spread of signs so it is not orthogonal
        // to the dominant eigenvector for the operators we measure.
        let mut v: Vec<f32> = (0..n)
            .map(|i| (((i as u64 * 2_654_435_761) % 1000) as f32) / 500.0 - 1.0)
            .collect();
        let n0 = norm(&v);
        if n0 == 0.0 {
            return 0.0;
        }
        v.iter_mut().for_each(|a| *a /= n0);

        let iters = iters.max(2);
        let warmup = iters / 2; // let v align with the dominant eigenspace before measuring the rate
        let (mut log_growth, mut steps) = (0.0f32, 0.0f32);
        for k in 0..iters {
            // wv = W v  (sparse; leaf units only — nested treated as opaque)
            let mut wv = vec![0.0f32; n];
            for (i, u) in self.units.iter().enumerate() {
                if u.nested.is_some() {
                    continue;
                }
                let mut s = self.scale.get(i).copied().unwrap_or(1.0);
                if self.config.gain_fanin_norm && !u.reads.is_empty() {
                    s /= (u.reads.len() as f32).sqrt(); // honor the g/√N gain in the Jacobian
                }
                let mut acc = 0.0f32;
                for (&c, &w) in u.reads.iter().zip(u.weights.iter()) {
                    acc += w * v.get(c).copied().unwrap_or(0.0);
                }
                if u.write < n {
                    wv[u.write] += s * acc;
                }
            }
            // apply D = diag(1 − m²): the tanh slopes at the CURRENT state
            for (c, wvc) in wv.iter_mut().enumerate() {
                let m = self.mem.medium[c];
                *wvc *= self.slope_of_cell(c, m); // D from the writing unit's own slope (activation · factor)
            }
            let nk = norm(&wv);
            if nk <= f32::EPSILON {
                return 0.0; // J annihilates v: nilpotent (feedforward) or fully saturated ⇒ ρ = 0
            }
            if k >= warmup {
                log_growth += nk.ln(); // v is unit-norm each step ⇒ nk is the per-step growth factor
                steps += 1.0;
            }
            v.iter_mut().zip(wv.iter()).for_each(|(a, b)| *a = b / nk);
        }
        if steps == 0.0 {
            return 0.0;
        }
        (log_growth / steps).exp()
    }

    /// A probe-free, domain-blind BEHAVIOURAL FINGERPRINT — a content-address by WHAT THE MORPH DOES, not its
    /// weights. Perturb the organism with a fixed CANONICAL probe (deterministic, machine-independent — no task
    /// probe, no caller input, so it stays domain-blind and replay-safe), settle, and take the magnitude
    /// SPECTRUM of the response (invariant to sample order/phase — the behavioural character, not the raw
    /// activations). Two morphs that behave the same get near-identical fingerprints; a copy matches exactly.
    ///
    /// This is the same object memoization (`nested_settles`), discovery (cache-miss), and the concept-map
    /// already use — ONE behavioural signature — so a marketplace (`mycelium`) can content-address / DEDUP
    /// sub-morphs by behaviour: the common sub-morph collapses to one node. Reuses the spectral
    /// representation. Reads only (clones internally to avoid mutating self).
    ///
    /// # Example
    /// ```
    /// use openmorph_core::{Event, Field};
    /// let mut a = Field::new(2);
    /// for _ in 0..300 { for &x in &[-1.0_f32, 1.0] {
    ///     a.alive(Event::Perceive { inputs: &[(0, x)], target_cell: 1, target: 0.6 * x, lr: 0.1, budget: 100 });
    /// }}
    /// let b = a.clone();
    /// // a COPY has an IDENTICAL behavioural fingerprint — content-address by what it does (dedup)
    /// assert_eq!(a.fingerprint(), b.fingerprint());
    /// ```
    pub fn fingerprint(&self) -> Vec<f32> {
        // The spectral prefix of the shared probe harness — `flat` is the historical probe-outer/cell-inner
        // response order, so this is byte-identical to the hand-rolled loop it replaces.
        let (_, flat, _) = self.probe_response();
        magnitude_spectrum(&flat, self.config.spectrum_bins.max(1))
    }

    /// The shared canonical-probe HARNESS behind [`Self::fingerprint`] and [`Self::embed`]: perturb the grown
    /// I/O ports with the fixed low-discrepancy probe sequence, settle each clone under pure inference (never
    /// learn/grow), and read the OUTPUT response. Returns the per-output-cell response matrix `R[cell][probe]`,
    /// the same values FLATTENED probe-outer/cell-inner (the spectral input, = the historical `fingerprint`
    /// response order), and the per-probe input energy `‖input_k‖` (canonical, morph-independent). ONE harness,
    /// so the two behavioural keys can never drift apart. `budget = n_cells + config.probe_settle_pad`.
    ///
    /// The behavioural INTERFACE is read off the grown graph: INPUTs are cells no unit writes (the world
    /// perturbs them); OUTPUTs are cells a unit writes but no unit reads (terminal — the world reads them).
    /// Reading the OUTPUT response to a probe on the INPUTs captures WHAT THE MORPH COMPUTES, not its hidden
    /// internals, so two morphs with the same function match regardless of how they grew.
    fn probe_response(&self) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let probes = self.config.fingerprint_probes.max(1);
        let n = self.mem.medium.len();
        let budget = n + self.config.probe_settle_pad;
        let probe_cells: Vec<usize> = (0..n)
            .filter(|&c| self.writers.get(c).is_none_or(|w| w.is_empty()))
            .collect();
        let read_cells: Vec<usize> = {
            let outs: Vec<usize> = (0..n)
                .filter(|&c| {
                    self.writers.get(c).is_some_and(|w| !w.is_empty())
                        && self.readers.get(c).is_none_or(|r| r.is_empty())
                })
                .collect();
            if outs.is_empty() {
                (0..n).collect()
            } else {
                outs
            }
        };
        let mut r: Vec<Vec<f32>> = vec![Vec::with_capacity(probes); read_cells.len()];
        let mut flat: Vec<f32> = Vec::with_capacity(read_cells.len() * probes);
        let mut in_norm: Vec<f32> = Vec::with_capacity(probes); // input energy per probe (canonical, same for all)
        for k in 0..probes {
            // canonical probe: a fixed deterministic pattern (the low-discrepancy seed), identical for every
            // morph — the caller supplies nothing.
            let inputs: Vec<(usize, f32)> =
                probe_cells.iter().map(|&c| (c, scatter(c, k))).collect();
            in_norm.push(inputs.iter().map(|&(_, v)| v * v).sum::<f32>().sqrt());
            let mut probe = self.clone();
            probe.frozen = true; // fingerprinting/embedding must not learn/grow — pure inference
            probe.stimulate(&inputs, budget);
            for (ci, &c) in read_cells.iter().enumerate() {
                let v = probe.mem.medium[c];
                r[ci].push(v);
                flat.push(v); // probe-outer / cell-inner: the fingerprint-identical order
            }
        }
        (r, flat, in_norm)
    }

    /// NATIVE behavioural DISTANCE between two organisms — a scale-normalised (cosine) distance over their
    /// [`Self::fingerprint`] spectra: `~0` for a `.clone()` (the self-noise floor is exactly 0, fingerprint being
    /// deterministic) and growing with behavioural difference. Replaces a marketplace's authored quantise-to-`u64`
    /// exact-hash dedup with a GRADED metric, so near-identical behaviours COLLAPSE (dedup by what
    /// a morph does, not a lucky bit-pattern). Reads only.
    pub fn fingerprint_distance(&self, other: &Field) -> f32 {
        cosine_distance(&self.fingerprint(), &other.fingerprint(), 0)
    }

    /// The behavioural EMBEDDING — the mycelium library retrieval KEY (OM-11). A NESTED, coarse→fine,
    /// FIXED-LENGTH vector that content-addresses a sub-morph by WHAT IT DOES, so a small morph can query a library
    /// for helper skills and reuse/compose them instead of rebuilding (the marketplace north-star). It reads
    /// the I/O ports the raw [`Self::fingerprint`] spectrum is deaf to, and catches a global gain rescale the
    /// spectrum's cosine is blind to.
    ///
    /// LAYOUT (ordered coarse→fine; a PREFIX is itself a valid cheaper key via [`Self::embed_distance`]'s `dims`).
    /// Length is `spectrum_bins + 2·fingerprint_probes + 1` (default 16+16+1 = 33), CONSTANT regardless of how many
    /// output cells a skill has — every block is a fixed-width reduction over cells, so keys ALIGN block-for-block
    /// across arities (a heterogeneous library mixes scalar- and multi-output skills):
    /// - **SPECTRAL PREFIX (`spectrum_bins`, unit)** — the magnitude spectrum of the flattened probe response,
    ///   padded to a fixed width. `embed_distance(dims = spectrum_bins)` equals [`Self::fingerprint_distance`]
    ///   (cosine is scale-free), so the old spectral index still works on the prefix; the cheapest recall key.
    /// - **SIGNED SHAPE (`fingerprint_probes`, unit)** — the across-probe MEAN signed response curve. Keeps the
    ///   SIGN/shape → separates odd vs even maps (y=x vs y=x²) AND a sign flip (y=x vs y=−x) for scalar/symmetric
    ///   outputs. Scale-free, so it reads the I/O map, not the gain.
    /// - **ENERGY SHAPE (`fingerprint_probes`, unit)** — the across-probe response-ENERGY curve. Sign-blind and
    ///   never-cancelling: the backstop that separates skills the signed block collapses (balanced anti-symmetric
    ///   multi-output).
    /// - **GAIN (1, BOUNDED ±`embed_gain_weight`)** — `w·tanh(mean log response-energy)`. Carries a global gain
    ///   rescale (y=x vs y=0.1·x embed FAR); bounded, so a silent skill maps to a constant instead of an amplitude
    ///   coordinate that swamps the key.
    ///
    /// GROWTH-INVARIANCE (bounded honestly, TNT): invariant to internal re-wiring under a FIXED I/O port layout and
    /// COMPARABLE output amplitude — every block is a function of the probe RESPONSE only, so a clone / `fork_mutated`
    /// re-grow that computes the same function embeds identically (distance ~0). It is NOT a universal address across
    /// arbitrary re-labellings: the canonical probes are index-keyed, so relocating the I/O ports changes the key.
    ///
    /// HONEST SCOPE (TNT): the NONLINEAR NATIVE port of the "behavioural hologram" — it reads the morph's ACTUAL
    /// (possibly nonlinear) settled response, not an LTI `C·Aᵏ·B` (that ticket's first risk); the LTI Hankel-SV block is
    /// deferred. The GAIN coordinate is monotone in log-gain but SATURATES (tanh), so rank on it, don't threshold.
    /// The metric is amplitude-CONSISTENT by construction (all shape blocks unit, gain bounded), fixing the shipped
    /// version's high-gain discrimination loss and low-energy collapse. `spectrum_bins`/`fingerprint_probes`/
    /// `embed_gain_weight` are `Config` levers (sweepable; the first two are a per-library key CONTRACT). Distance is
    /// cosine — see [`Self::embed_distance`].
    pub fn embed(&self) -> Vec<f32> {
        let probes = self.config.fingerprint_probes.max(1);
        let bins = self.config.spectrum_bins.max(1);
        let gain_w = self.config.embed_gain_weight;
        // Shared probe harness (identical to `fingerprint`'s): the response MATRIX R[cell][probe] — each output
        // cell's response across the canonical probe sequence is its input→output MAP (odd vs even, linear vs
        // nonlinear, …); the SHAPE of that map is where behavioural discrimination lives, its overall MAGNITUDE is
        // the gain. `flat` is the same values probe-outer/cell-inner (the spectral input), `in_norm` the per-probe
        // input energy. Reading both behavioural keys off ONE harness keeps them from drifting.
        let (r, flat, in_norm) = self.probe_response();
        fn unit(b: &[f32]) -> Vec<f32> {
            let n = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            b.iter().map(|v| v / n).collect()
        }
        let ncell = r.len().max(1) as f32;
        // BLOCK 1 — SPECTRAL PREFIX (fixed width `bins`, unit): the magnitude spectrum of the flattened response,
        // PADDED to exactly `bins` so blocks after it land at a constant offset regardless of arity. Cosine is
        // scale-free, so `embed_distance(dims=bins)` equals `fingerprint_distance` (the backward-compatible key)
        // whether or not the prefix is unit-normalised — normalising is what stops the prefix from dominating the
        // full-resolution cosine at high amplitude (the shipped raw overwrite did dominate — OM-11 verify).
        let mut prefix = magnitude_spectrum(&flat, bins);
        prefix.resize(bins, 0.0);
        let prefix = unit(&prefix);
        // BLOCK 2 — SIGNED SHAPE (width `probes`, unit): the across-probe MEAN signed response curve. Keeps the
        // SIGN/shape, so it separates an odd map (y=x) from an even one (y=x²) AND a sign flip (y=x vs y=−x) for a
        // scalar or symmetric output. Scale-free (unit) → carries the map, not the gain. (For a balanced
        // ANTI-symmetric multi-output skill the signed mean can cancel — the energy block is the backstop.)
        let mut signed = vec![0.0f32; probes];
        for row in &r {
            for (s, v) in signed.iter_mut().zip(row.iter()) {
                *s += v / ncell;
            }
        }
        let signed = unit(&signed);
        // BLOCK 3 — ENERGY SHAPE (width `probes`, unit): the across-probe response-ENERGY curve `‖R[:,k]‖`. Sign-
        // blind and never-cancelling, it is the backstop that separates skills the signed block collapses.
        let mut energy = vec![0.0f32; probes];
        for (k, e) in energy.iter_mut().enumerate() {
            *e = r.iter().map(|row| row[k] * row[k]).sum::<f32>().sqrt();
        }
        let energy = unit(&energy);
        // BLOCK 4 — GAIN (one BOUNDED scalar): `gain_w · tanh(mean_k ln(‖R[:,k]‖ / ‖input_k‖))`. The per-probe
        // OUTPUT/INPUT energy ratio is the map's actual GAIN — normalising by the (canonical, morph-independent)
        // input energy CENTRES the axis so a unit-gain map (y=x) reads 0 and a rescale reads symmetric about it, so
        // a SHRINK (y=0.1·x) and a STRETCH (y=10·x) both embed FAR (an absolute-energy axis put shrinks on the same
        // side as unit gain — OM-11 verify). The scale-free shape blocks are blind to this global rescale; this
        // coordinate carries it. BOUNDED in ±gain_w by the tanh, so a silent skill maps to a constant instead of a
        // huge log-floor that swamps the key (the shipped raw ln-energy caused BOTH high-gain discrimination loss
        // AND low-energy collapse). Monotone in log-gain but SATURATES — rank on it, don't threshold.
        let mean_log_gain = (0..probes)
            .map(|k| {
                let e = r.iter().map(|row| row[k] * row[k]).sum::<f32>().sqrt();
                (e + 1e-9).ln() - (in_norm[k] + 1e-9).ln()
            })
            .sum::<f32>()
            / probes as f32;
        let gain = gain_w * mean_log_gain.tanh();
        // FIXED-length nested key: bins + 2·probes + 1, independent of output arity. Coarse→fine, so a prefix is a
        // valid cheaper key (`embed_distance`'s `dims`): dims=`bins` is exactly the spectral fingerprint distance.
        let mut v = prefix;
        v.extend(signed);
        v.extend(energy);
        v.push(gain);
        v
    }

    /// Cosine distance between two behavioural [`Self::embed`]dings, over the first `dims` components (`0` = the
    /// full vector). Because the embedding is ordered coarse→fine, a smaller `dims` is a cheaper, coarser key
    /// (Matryoshka): `dims = fingerprint().len()` compares on the spectral prefix alone (= [`Self::fingerprint_distance`]),
    /// larger `dims` adds the I/O direction + gain envelope that separate same-spectrum look-alikes. `~0` for a clone.
    pub fn embed_distance(&self, other: &Field, dims: usize) -> f32 {
        cosine_distance(&self.embed(), &other.embed(), dims)
    }

    /// STRUCTURAL SELF-MODEL — a permutation-invariant fingerprint of the morph's OWN GRAPH SHAPE, the
    /// STRUCTURAL twin of the behavioural [`Self::embed`] (Sébastien's morph-understands-morph).
    /// Where `embed` reads WHAT a morph computes (probe response), this reads HOW IT IS WIRED — the topology of
    /// the cell-graph (nodes = cells that any unit touches; a directed edge `read/gate/factor → write` per unit),
    /// with NO probe and NO I/O port assumption. If perception is graph-embedding, then perceiving another morph
    /// is the SAME operation as perceiving a scene: fold the graph's local structure into a fixed-width code.
    ///
    /// The code is built from ORDER-INVARIANT aggregates (moments over nodes/edges), so it is INVARIANT to any
    /// relabelling of cells or reordering of units (an isomorphic re-wiring embeds identically, distance ~0), and
    /// it is SIZE-CONTROLLED (every coordinate is a ratio / shape statistic, not a raw count — a chain and a star
    /// of the SAME node count separate on hub-ness / degree-heterogeneity, not on size). Coordinates:
    /// `[density, hubness, deg_cv, in_cv, out_cv, source_frac, sink_frac, nested_frac,
    ///   child_density, child_hubness, child_nested_frac, (spread,max,min)×3 diffusion rounds]` — `density`=E/N,
    /// `hubness`=log1p(maxdeg/meandeg), `*_cv`=coefficient of variation, `*_frac`=fraction of nodes, the diffusion
    /// block is a 3-round Weisfeiler-Lehman-style neighbour-degree relaxation read out mean-normalised. The
    /// `child_*` block RECURSES into nested sub-morphs (a cell can be a MORPH), so the code is a DEEP
    /// self-model of the fractal graph, not just its top level. Compare with [`Self::structural_distance`].
    pub fn structural_embed(&self) -> Vec<f32> {
        self.structural_features(3)
    }

    /// Euclidean distance between two [`Self::structural_embed`]dings — small for isomorphic (same-shape) morphs,
    /// large for topologically-different ones (chain vs star vs tree vs fractal tower). The metric for
    /// morph-understands-morph: which teacher's structure matches a student's, which sub-morph a graph contains.
    pub fn structural_distance(&self, other: &Field) -> f32 {
        let (a, b) = (self.structural_embed(), other.structural_embed());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// The fixed-width local-graph feature block for [`Self::structural_embed`]; `depth` bounds the recursion
    /// into nested sub-morphs (a cell that IS a morph contributes its own structure to the `child_*` slots).
    fn structural_features(&self, depth: usize) -> Vec<f32> {
        const WIDTH: usize = 20;
        // ── Build the directed cell-graph from the units: an edge from every input (read / gate / factor) to the
        //    cell the unit writes. Nodes are only the cells some unit touches, so a wider membrane with the same
        //    wiring embeds identically (invariant to unused width). ──
        let mut nodeset = alloc::collections::BTreeSet::new();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut children: Vec<&Field> = Vec::new();
        for u in &self.units {
            nodeset.insert(u.write);
            for &r in &u.reads {
                nodeset.insert(r);
                edges.push((r, u.write));
            }
            if let Some(g) = u.gate {
                nodeset.insert(g);
                edges.push((g, u.write));
            }
            if let Some(fc) = u.factor {
                nodeset.insert(fc);
                edges.push((fc, u.write));
            }
            if let Some(nst) = &u.nested {
                children.push(&nst.sub);
            }
        }
        let n = nodeset.len();
        if n == 0 {
            return vec![0.0; WIDTH];
        }
        // Compact, DETERMINISTIC index per participating cell (BTreeSet is sorted), so relabelling the cells
        // cannot change any downstream aggregate.
        let nodes: Vec<usize> = nodeset.iter().copied().collect();
        let mut compact = alloc::collections::BTreeMap::new();
        for (i, &c) in nodes.iter().enumerate() {
            compact.insert(c, i);
        }
        let mut indeg = vec![0.0f32; n];
        let mut outdeg = vec![0.0f32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n]; // undirected neighbours (for the diffusion)
        for &(s, d) in &edges {
            let (si, di) = (compact[&s], compact[&d]);
            outdeg[si] += 1.0;
            indeg[di] += 1.0;
            adj[si].push(di);
            adj[di].push(si);
        }
        let total: Vec<f32> = (0..n).map(|i| indeg[i] + outdeg[i]).collect();
        // mean / std / max / min of a slice (order-invariant).
        fn moments(x: &[f32]) -> (f32, f32, f32, f32) {
            if x.is_empty() {
                return (0.0, 0.0, 0.0, 0.0);
            }
            let m = x.iter().sum::<f32>() / x.len() as f32;
            let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / x.len() as f32;
            let mx = x.iter().copied().fold(f32::MIN, f32::max);
            let mn = x.iter().copied().fold(f32::MAX, f32::min);
            (m, var.sqrt(), mx, mn)
        }
        let nf = n as f32;
        let density = edges.len() as f32 / nf;
        let (tm, tstd, tmax, _tmin) = moments(&total);
        let hubness = if tm > 1e-9 { (tmax / tm).ln_1p() } else { 0.0 };
        let deg_cv = if tm > 1e-9 { tstd / tm } else { 0.0 };
        let (im, istd, _, _) = moments(&indeg);
        let (om, ostd, _, _) = moments(&outdeg);
        let in_cv = if im > 1e-9 { istd / im } else { 0.0 };
        let out_cv = if om > 1e-9 { ostd / om } else { 0.0 };
        let source_frac = indeg.iter().filter(|&&d| d == 0.0).count() as f32 / nf;
        let sink_frac = outdeg.iter().filter(|&&d| d == 0.0).count() as f32 / nf;
        let nested_frac = if self.units.is_empty() {
            0.0
        } else {
            children.len() as f32 / self.units.len() as f32
        };
        // ── Fractal recursion: fold nested sub-morphs' OWN structure into the child_* slots (deep self-model). ──
        let (mut cden, mut chub, mut cnest) = (0.0f32, 0.0f32, 0.0f32);
        if depth > 0 && !children.is_empty() {
            for sub in &children {
                let cf = sub.structural_features(depth - 1);
                cden += cf[0]; // child density
                chub += cf[1]; // child hubness
                cnest += cf[7]; // child nested_frac
            }
            let k = children.len() as f32;
            cden /= k;
            chub /= k;
            cnest /= k;
        }
        // ── Diffusion (Weisfeiler-Lehman-style): relax each node's degree by its neighbours' mean, 3 rounds;
        //    read out mean-normalised (spread, max-gap, min-gap) each round. A hub, a chain and a tree diffuse
        //    differently, so this separates topologies of equal size. ──
        let mut s = total.clone();
        let mut diff = Vec::with_capacity(9);
        for _ in 0..3 {
            let (m, std, mx, mn) = moments(&s);
            let denom = if m.abs() > 1e-9 { m } else { 1.0 };
            diff.push(std / denom);
            diff.push((mx - m) / denom);
            diff.push((m - mn) / denom);
            let mut ns = s.clone();
            for i in 0..n {
                if !adj[i].is_empty() {
                    ns[i] = adj[i].iter().map(|&j| s[j]).sum::<f32>() / adj[i].len() as f32;
                }
            }
            s = ns;
        }
        let mut v = vec![
            density,
            hubness,
            deg_cv,
            in_cv,
            out_cv,
            source_frac,
            sink_frac,
            nested_frac,
            cden,
            chub,
            cnest,
        ];
        v.extend(diff);
        debug_assert_eq!(v.len(), WIDTH);
        v
    }
}
