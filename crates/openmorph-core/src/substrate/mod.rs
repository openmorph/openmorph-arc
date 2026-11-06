// SPDX-License-Identifier: MIT-0
//! The ASYNC SUBSTRATE — the organism with NO global tick. A [`Field`] OWNS a [`Membrane`]: its cells
//! ARE the medium. A [`Unit`] (organ) reads some cells and writes one; it FIRES only when a cell it
//! reads has CHANGED. A change enqueues the units reading that cell; firing writes an output cell; if
//! that output changed, its readers enqueue — an event cascade that spreads and **settles** to a
//! fixed point (the queue empties) or stops at an anytime **budget**. Like neurons: each fires on its
//! own timeline, no clock.
//!
//! Determinism (so research stays replayable): a single event queue drained in FIFO order, readers in
//! a fixed order, no randomness — byte-identical at any concurrency.
//!
//! Interfaces plug in two ways: [`Field::attach`] claims FRESH cells for a new interface; [`Field::tap`]
//! wraps EXISTING cells — so any pattern the morph GREW (a concept) becomes a live interface you can
//! LISTEN to (read its representation) or TRIGGER (write it to fire downstream). rc6 step 3.

use crate::config::{ActRule, Activation, Config, RelianceNorm};
use crate::membrane::{Binding, Flow, Membrane, Port};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::compat::Float;
use crate::compat::{HashMap, HashSet};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

// The concerns of `Field`, one child module each (every child is an `impl Field` block over the struct below).
mod act;
mod alive;
mod cull;
mod economy;
mod evolve;
mod geometry;
mod grow;
mod index;
mod introspect;
mod learn;
mod live;
mod nest;
mod plasticity;
mod rng;
mod settle;
mod structure;
mod temporal;
mod validate;
mod value;
mod wiring;

/// Numeric floor (division/pow guard). The SETTLE quiescence threshold is a tunable knob now —
/// [`Config::settle_eps`]; this constant is kept only for pure numeric hygiene (e.g. `half_life.max(EPS)`).
const EPS: f32 = 1e-4;

/// A UNIT (organ): reads cells `reads`, writes one cell. A LEAF unit computes `tanh(Σ w·x + b)` (a bounded
/// neuron). A NESTED unit — the fractal case — instead runs a whole sub-morph: reading it settles `nested`'s
/// sub-[`Field`] on the read values and yields one output. So a cell can BE a Morph, recursively (its sub-cells
/// may themselves be nested), and a banked skill is a single addressable node a higher morph routes to — depth
/// emergent, no flat-DAG floor. Fires only on a change to a read cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unit {
    /// Input cells.
    pub reads: Vec<usize>,
    /// One weight per read cell (leaf units).
    pub weights: Vec<f32>,
    /// Constant input (leaf units).
    pub bias: f32,
    /// The single cell this unit writes.
    pub write: usize,
    /// If `Some`, this cell IS a sub-morph — the recursive node. Reading it settles the nested sub-field
    /// instead of `tanh(Σw·x+b)`. `#[serde(default)]` so a snapshot predating nesting loads as a leaf.
    #[serde(default)]
    pub nested: Option<Box<Nested>>,
    /// WRITE-GATE — if `Some(g)`, this unit LEARNS only in proportion to gate cell `g`'s value (clamped to
    /// `[0,1]`): the delta step is scaled by `g`, so the unit WRITES when the gate is open and is PROTECTED
    /// (frozen) when it's closed. The one thing the additive `tanh(Σw·x)` unit lacked. It is salience-gated
    /// learning — the neuromodulation / dopamine analog: open the write on surprise/reward/attention, close it
    /// otherwise. From it fall out short-term MEMORY (an activity-silent fast-weight register: store a value in
    /// the weight while the gate is open, protect it while closed, read it by pinging its address), reduced
    /// catastrophic INTERFERENCE (a unit updates only in its own context), and satisfaction-gated CONSOLIDATION
    /// (write while unsatisfied, freeze once satisfied). The gate is itself a cell — grown, not authored.
    /// `#[serde(default)]` (an ungated unit always writes, exactly as before).
    #[serde(default)]
    pub gate: Option<usize>,
    /// PER-UNIT nonlinearity — `None` (the default) means "use the field's genome", [`Config::activation`].
    /// A grown organism leaves this `None`: one genome, one law for every unit it births. It exists for the
    /// TRANSCRIBED case — a foreign network imported layer by layer mixes nonlinearities (a `Relu` hidden
    /// layer under an `Identity` logit layer), and the transcription is only honest if each unit is the
    /// function the source graph says it is. `#[serde(default)]` ⇒ every pre-existing snapshot loads as
    /// `None` and behaves byte-identically.
    #[serde(default)]
    pub activation: Option<Activation>,
    /// FORWARD FACTOR — if `Some(c)`, this unit's output is MULTIPLIED by cell `c`:
    /// `out = cells[c] · activation(gain·(Σw·x + b))`.
    ///
    /// It is [`Unit::gate`]'s twin on the other path. A `gate` modulates what a unit WRITES INTO ITS
    /// WEIGHTS (learning); a `factor` modulates what it WRITES INTO ITS CELL (activation). They are
    /// deliberately separate fields because the two are opposites at a closed gate: a write-gated
    /// memory register is READ while its gate is shut (that is what "activity-silent" means — see
    /// `substrate_memory`), so gating the output too would destroy the value it is holding.
    ///
    /// This is the substrate's only SECOND-ORDER term: a leaf unit is otherwise affine in the cells,
    /// so no single unit could ever emit the product of two LIVE values. With a factor it can, in one
    /// unit — `reads=[x], weights=[1.0], factor=Some(y)` is `x·y` — and from that fall out multiplicative
    /// attention, gated linear units, binding, and gain modulation. `#[serde(default)]` ⇒ an ungated
    /// unit multiplies by nothing and every existing snapshot loads byte-identically.
    #[serde(default)]
    pub factor: Option<usize>,
}

/// A cell that is a Morph — an encapsulated sub-organism reused as ONE addressable node. Reading the owning
/// unit maps its `reads` (positionally) into `sources` of `sub`, settles `sub`, and returns `sub`'s `sink`
/// cell. `sub` may itself contain nested units, so composition is a recursive tree, not a flat band: the
/// keystone of the fractal recursive hypergraph. Built by [`Field::encapsulate`], planted by
/// [`Field::nest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nested {
    /// The encapsulated sub-organism (its own membrane + units; cells may themselves be nested).
    pub sub: Field,
    /// The sub's input cells, in ascending order — matched positionally to the owning unit's `reads`.
    pub sources: Vec<usize>,
    /// The sub cell whose settled value is this node's PRIMARY output (written to the owning unit's cell).
    pub sink: usize,
    /// ADDITIONAL vector outputs — `(sub sink cell, host cell)` pairs written on the SAME sub-settle as `sink`,
    /// so a nested cell can expose a sub-morph's FULL vector output, not just one scalar (multi-output `nest`).
    /// Empty for a single-output nest ⇒ byte-identical to the historical scalar node. `#[serde(default)]` so an
    /// old snapshot loads with no extra outputs.
    #[serde(default)]
    pub extra: Vec<(usize, usize)>,
    /// Settle budget for the sub-morph per firing.
    pub budget: usize,
    /// MEMO — a nested cell is a FROZEN brick (learning skips it), so for a given input its output is fixed.
    /// Cache the FULL output vector (`sink` then each `extra` sink) by the input's bit pattern, so a RECURRING
    /// input (a revisited discrete state — ARC-3's regime) is served without re-settling the whole sub.
    /// Fire-on-change already skips an UNCHANGED input; this adds the non-consecutive-repeat case. Bounded by
    /// `Config::memo_cap`. NOT a pure cache: a re-settle starts from the sub's CURRENT state, so it can land a
    /// different fixed point (within `settle_eps`) than the settle a hit replays — a cold memo after a reload would
    /// drift the host cells. So it is serialized as a key-sorted list (`#[serde(default)]` — an older snapshot
    /// loads it empty).
    #[serde(default, with = "sorted_map")]
    memo: HashMap<Vec<u32>, Vec<f32>>,
}

/// A TEMPORAL TRACE — a leaky integrator (an EMA in the organism's OWN subjective event-time) of a source
/// cell, refreshed once per settle. It is the time-axis twin of a grown feature: a bank of traces at
/// geometrically-spaced half-lives is a MULTI-SCALE temporal basis the morph grows features over, and a
/// trace whose source is ANOTHER trace is a summary-of-a-summary — recursive nested pooling, the fractal
/// map-of-maps on the TIME axis. The half-life (in subjective events) sets the timescale;
/// grow features over the trace cells and let the reliance economy size the timescales the stream needs —
/// no hand-set context window K. Updated Δt-aware in clock time (like [`Field::recency`]), so however many
/// times a settle touches it, it relaxes by exactly the elapsed subjective time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// The cell this trace writes (its running summary).
    pub cell: usize,
    /// The source cell it integrates.
    pub source: usize,
    /// Integration half-life in subjective events — the timescale (larger ⇒ slower, longer memory).
    pub half_life: f32,
    /// Subjective clock at the trace's last refresh (so Δt-aware relaxation is robust to multiple settles
    /// per event). `#[serde(default)]` — an old snapshot restores traces as freshly ticked.
    #[serde(default)]
    ticked_at: u64,
    /// Local sensitivity of the trace value to its own half-life, `∂m/∂h`, captured at the last refresh.
    /// Used by half-life-gradient learning ([`Field::adapt_scales`]) to nudge the timescale toward the one
    /// that reduces the readout's error — the temporal twin of feature-direction learning. An accumulated RTRL
    /// recurrence (it carries the trace's whole history), so it is SERIALIZED — a snapshot resumes the same
    /// half-life descent; `#[serde(default)]` loads an older snapshot at 0.
    #[serde(default)]
    sens: f32,
}

/// A DELAY — the EXACT-memory twin of a [`Trace`]. Where a trace holds a smoothed *summary* of the
/// past (lossy for order), a delay holds a *faithful copy* of its source one event ago (`x[t−1]`), so a
/// chain of delays (`delay(delay(x))`) is a native shift register carrying the recent EXACT order — the
/// order `th ≠ ht` that an EMA blurs. Unlike a trace (refreshed Δt-aware inside `stimulate`), a delay is a
/// DISCRETE register that must shift exactly once per external event, so it advances only in
/// [`Field::advance_delays`] (the per-event tick), never inside the settle. Together, a delay bank (exact)
/// and a trace bank (trend) are the native stream memory; the reliance economy selects the flavor per task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delay {
    /// The cell this delay writes (its source's value from the previous event).
    pub cell: usize,
    /// The source cell it delays.
    pub source: usize,
    /// The value to emit at the next shift — the source's value captured at the last shift.
    prev: f32,
}

/// A RESONATOR — the band-pass twin of a [`Trace`]. Where [`Field::integrate`] is a 1st-order real-pole
/// LOW-PASS (a leaky EMA), a resonator is a 2nd-order COMPLEX-pole BAND-PASS: it RINGS at a tuned frequency. Its
/// coupled quadrature state (`u` in-phase, `v` quadrature, 90° apart — a grown quadrature oscillator) is advanced
/// by `Field::tick_resonators` exactly ONE step per `stimulate` call (per settle) — outside the settle cascade
/// (which converges to a fixed point; it can't ring, and the saturating unit activation can't carry a clean
/// linear ring). Unlike a `Trace` (Δt-aware) or a `Delay` (shifted by `advance_delays`), a step is not tied to
/// elapsed time or to an external event: an API that settles `k` times per call advances it `k` steps. The
/// state-transition is a rotation by `freq` scaled by `radius = 1 − damping`, so the pole is
/// `radius·e^{±i·freq}` — a complex-conjugate pair whose distance from the unit circle IS the
/// organism's own stability (the modulus is what [`Field::spectral_radius`] measures). Grow `freq` by gradient with
/// [`Field::adapt_freq`] — the frequency analog of [`Field::adapt_scales`] (which grows a trace's half-life = its
/// real pole radius). Nothing authored: the bank tunes to where the signal's energy is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resonator {
    /// The cell this resonator writes its IN-PHASE state `u` to (the band-pass response of `source`).
    pub cell: usize,
    /// The cell for the QUADRATURE state `v` (90° from `u`); `√(u²+v²)` is the band's energy/envelope.
    pub vcell: usize,
    /// The source cell it band-passes.
    pub source: usize,
    /// Centre frequency = pole ANGLE `ω ∈ [0, π]` (radians per step — one step per settle).
    pub freq: f32,
    /// Pole RADIUS `r = 1 − damping ∈ [0, 1)`; closer to 1 ⇒ higher-Q (sharper, longer ring).
    pub radius: f32,
    /// In-phase state `u`.
    #[serde(default)]
    u: f32,
    /// Quadrature state `v`.
    #[serde(default)]
    v: f32,
    /// Forward-mode frequency sensitivities `∂u/∂ω`, `∂v/∂ω` (RTRL — the ω twin of `Trace::sens`), for
    /// [`Field::adapt_freq`]. Accumulated recurrences, so SERIALIZED (a snapshot resumes the same frequency
    /// ascent); `#[serde(default)]` loads an older snapshot at 0.
    #[serde(default)]
    su: f32,
    #[serde(default)]
    sv: f32,
}

/// A stored EPISODE — native EPISODIC MEMORY (the experience twin of the streaming [`Delay`]/[`Trace`]). Where a
/// delay/trace is a rolling window a step wide, an episode is an ADDRESSABLE, replayable snapshot of a chosen
/// cell-set: memory IS cells. It holds the `stimulus` cells' values (what a [`Field::replay`] RE-INJECTS through
/// the existing `stimulate` path to drive the cascade from a remembered state) and the `outcome` cells' values
/// (the remembered result, so a replay can be SCORED against what actually happened — the substrate of dreaming,
/// genomic evaluation, and imagination). Because it lives on the [`Field`], it CLONES with [`Field::fork_mutated`]
/// (a mutated clone inherits the parent's experience — the keystone of the genome loop: a clone can only be scored
/// on the parent's remembered data if that data travels with it) and round-trips through a snapshot (serde).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Episode {
    /// The remembered STIMULUS — cell → value pairs a [`Field::replay`] re-injects via `stimulate` to drive the
    /// cascade from this remembered state. Memory IS cells: an episode is just a captured cell-state.
    pub stimulus: Vec<(usize, f32)>,
    /// The remembered OUTCOME — the target/output cells' values at store time (may be empty for a pure stimulus
    /// memory). A replayed clone reads these back from its OWN memory to score its fit — no external data needed.
    pub outcome: Vec<(usize, f32)>,
}

/// A fractal MAP-OF-MAPS index over a set of skill cells' concept-map positions — the substrate's own
/// router. Built by [`Field::index_skills`] from the ONE self-similar `cluster` operator; descended by
/// [`Field::route`] to the nearest skill in ~O(√M) comparisons instead of a flat O(M) scan. A `Branch` stores
/// one centroid (mean position) per child; a `Leaf` holds the skill cell ids. Serializable so a live library
/// persists (and rebuilds when skills are added).
///
/// Not to be confused with `openmorph_mycelium::SkillLibrary` — that is a flat cosine k-NN store over
/// behavioural `embed` keys for cross-node retrieval; this is the in-organism grown region tree over
/// concept-map *positions*. Different structure, different key, different crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillIndex {
    /// A bucket of skill cell ids (small enough to scan flat).
    Leaf(Vec<usize>),
    /// A region: one centroid position per child sub-region + the children.
    Branch {
        centroids: Vec<Vec<f32>>,
        kids: Vec<SkillIndex>,
    },
}

impl Unit {
    /// This unit's nonlinearity: its own override if it has one, else the field's genome `default`.
    #[inline]
    fn act(&self, default: Activation) -> Activation {
        self.activation.unwrap_or(default)
    }

    /// Activation `tanh(gain·(Σ w·x + b))`. `gain` is the unit's live projection scale (a heritable trait
    /// on the field); at the default 1.0 this is the plain `tanh(Σ w·x + b)`.
    fn compute(&self, cells: &[f32], gain: f32, fanin_norm: bool, act: Activation) -> f32 {
        let mut s = self.bias;
        for (&c, &w) in self.reads.iter().zip(self.weights.iter()) {
            s += cells.get(c).copied().unwrap_or(0.0) * w;
        }
        // g/√N (v2 capability fix): scaling the gain by 1/√fan_in keeps the pre-activation and the recurrent
        // spectral radius bounded as fan-in grows (a coupled population no longer goes super-critical by size).
        let g = if fanin_norm && !self.reads.is_empty() {
            gain / (self.reads.len() as f32).sqrt()
        } else {
            gain
        };
        act.apply(g * s) // v2 genome knob: the unit's own nonlinearity, else Config::activation (default Tanh)
    }
}

/// Push unit `id`'s adjacency edges — one reader edge per read occurrence, a reader edge on its forward
/// [`Unit::factor`] cell unless it already reads that cell (the edge [`Field::set_factor`] registers: a change to
/// the factor must re-fire the unit), and its writer edge. The ONE canonical edge set every adjacency rebuild uses,
/// so a cull/merge/re-index can never silently drop a factor dependency. Out-of-range cells are skipped.
fn push_unit_edges(readers: &mut [Vec<usize>], writers: &mut [Vec<usize>], id: usize, u: &Unit) {
    for &c in &u.reads {
        if c < readers.len() {
            readers[c].push(id);
        }
    }
    if let Some(f) = u.factor {
        if f < readers.len() && !u.reads.contains(&f) {
            readers[f].push(id);
        }
    }
    if u.write < writers.len() {
        writers[u.write].push(id);
    }
}

/// The three EMAs behind the per-cell mastery meter (the crystal half). `fast`/`slow` track a caller-supplied
/// reliability signal on the substrate's own two timescales (`surprise_decay`/`credit_decay`); `peak` is a leaky
/// running maximum. Mastery = height·settled is derived from these (see [`Field::mastery_of`]) — both terms are
/// RELATIVE to `peak`, so there is no authored threshold.
///
/// Public but `#[doc(hidden)]`: it exists for `openmorph-physics`'s bit-for-bit mirror (its `Scheduler` carries the
/// same per-cell meter); it is not part of the organism's API.
#[doc(hidden)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Reliability {
    /// The fast EMA of the reliability signal (`surprise_decay` timescale).
    pub fast: f32,
    /// The slow EMA of the reliability signal (`credit_decay` timescale).
    pub slow: f32,
    /// The leaky running maximum the fast/slow terms are read relative to.
    pub peak: f32,
}

/// Serde for a serialized map (`co_credit`/`assoc` keyed by a cell pair, `Nested::memo` by an input's bits) as a
/// KEY-SORTED `[(key, value), …]` list: JSON cannot key a map by a tuple or a vector, and a `HashMap`'s iteration
/// order must never reach the snapshot bytes (the replay moat).
mod sorted_map {
    use crate::compat::HashMap;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    use core::hash::Hash;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<K: Ord + Serialize, V: Serialize, S: Serializer>(
        m: &HashMap<K, V>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let mut v: Vec<(&K, &V)> = m.iter().collect();
        v.sort_unstable_by(|a, b| a.0.cmp(b.0)); // keys are unique ⇒ a total, deterministic order
        v.serialize(s)
    }

    pub(super) fn deserialize<'de, K, V, D>(d: D) -> Result<HashMap<K, V>, D::Error>
    where
        K: Ord + Hash + Deserialize<'de>,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Ok(Vec::<(K, V)>::deserialize(d)?.into_iter().collect())
    }
}

/// A `BTreeMap` serialised as its sorted `(key, value)` list: the bytes [`sorted_map`] writes for the same entries.
mod ordered_map {
    use alloc::collections::BTreeMap;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<K: Ord + Serialize, V: Serialize, S: Serializer>(
        m: &BTreeMap<K, V>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        m.iter().collect::<Vec<(&K, &V)>>().serialize(s)
    }

    pub(super) fn deserialize<'de, K, V, D>(d: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        K: Ord + Deserialize<'de>,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Ok(Vec::<(K, V)>::deserialize(d)?.into_iter().collect())
    }
}

/// The fixed seed of the deterministic act-sampling PRNG ([`Field::act_rng`]) — a snapshot predating the field
/// loads this, so an old organism resumes on the same stream a fresh one starts on.
fn default_act_rng() -> u64 {
    0xD1B54A32D192ED03
}

/// The backprojected pseudo-error per feature cell, in first-encounter order — see
/// [`Field::learn_feature_directions`].
#[derive(Debug, Clone, Default)]
struct Backproj {
    errs: Vec<(usize, f32)>,
    /// cell → `1 + slot` in `errs` (0 = no slot yet); all-zero again once `errs` is applied.
    slot: Vec<usize>,
}

/// The inverse input covariance `P` of one recursive-least-squares readout (`Config::readout_rls`), row-major
/// `(reads + 1)²` with the bias last, built over `reads`. Empty until the unit's first RLS step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RlsState {
    reads: Vec<usize>,
    p: Vec<f32>,
}

impl RlsState {
    /// Re-key `P` to the unit's current `reads` (a `grow` birth appends one, a cull drops orphans). A kept read keeps
    /// its rows and columns (dropping a read's row and column marginalises it out of the Gaussian posterior `P` is the
    /// covariance of); a new read enters uncorrelated at the prior variance `p0`. A no-op when nothing changed.
    fn align(&mut self, reads: &[usize], p0: f32) {
        let n = reads.len() + 1;
        if self.reads == reads && self.p.len() == n * n {
            return;
        }
        let m = self.reads.len() + 1;
        let built = self.p.len() == m * m;
        let old = |slot: usize| -> Option<usize> {
            if !built {
                None
            } else if slot == n - 1 {
                Some(m - 1) // the bias
            } else {
                self.reads.iter().position(|&c| c == reads[slot])
            }
        };
        let map: Vec<Option<usize>> = (0..n).map(old).collect();
        let mut p = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                p[a * n + b] = match (map[a], map[b]) {
                    (Some(i), Some(j)) => self.p[i * m + j],
                    _ if a == b => p0,
                    _ => 0.0,
                };
            }
        }
        self.reads = reads.to_vec();
        self.p = p;
    }

    /// One weighted RLS step on the regressor `g` (the unit's local gradient: `f′·x`, then `f′` for the bias) with
    /// observation weight `w > 0` and forgetting factor `lambda`: returns the gain `k = P·g / (λ/w + gᵀP·g)` (the
    /// weight step is `k·err`) and updates `P ← (P − k·gᵀP)/λ`, kept exactly symmetric.
    fn step(&mut self, g: &[f32], w: f32, lambda: f32, p0: f32) -> Vec<f32> {
        let n = g.len();
        let pg: Vec<f32> = (0..n)
            .map(|a| (0..n).map(|b| self.p[a * n + b] * g[b]).sum())
            .collect();
        let denom = lambda / w + g.iter().zip(&pg).map(|(x, y)| x * y).sum::<f32>();
        if denom <= 0.0 || !denom.is_finite() {
            return vec![0.0; n];
        }
        for a in 0..n {
            for b in 0..n {
                self.p[a * n + b] = (self.p[a * n + b] - pg[a] * pg[b] / denom) / lambda;
            }
        }
        // WIND-UP BOUND: with `λ < 1`, directions the stream no longer excites grow as `1/λ` per event without bound
        // (the classic RLS blow-up). Scale `P` back whenever its trace passes the prior's (`n·P₀`), which keeps it
        // symmetric positive definite. Never read at `λ = 1`, where `P` only shrinks.
        if lambda < 1.0 {
            let tr: f32 = (0..n).map(|a| self.p[a * n + a]).sum();
            let cap = n as f32 * p0;
            if tr > cap {
                let s = cap / tr;
                self.p.iter_mut().for_each(|v| *v *= s);
            }
        }
        pg.iter().map(|x| x / denom).collect()
    }
}

/// The units whose parameters changed OUTSIDE of firing, so the cells they write are STALE until the refresh of
/// `Field::drain` re-fires them (see `Field::dirty`): per-unit flags (lazily sized), the list of flagged units (so a
/// refresh visits only those) and their count (so a field with none pays nothing). Serialized with the field (a
/// flag pending at snapshot time must survive the load, or a reloaded organism would read one stale output that the
/// in-memory one refreshes — snapshot→load→continue is byte-identical to continuing).
///
/// Public but `#[doc(hidden)]`: it exists for `openmorph-physics`'s bit-for-bit mirror (its `Scheduler` carries
/// `Field`'s own staleness set); it is not part of the organism's API.
#[doc(hidden)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stale {
    flag: Vec<bool>,
    list: Vec<usize>,
    n: usize,
}

impl Stale {
    /// Flag unit `i` (of `n_units`) stale.
    pub fn mark(&mut self, n_units: usize, i: usize) {
        if self.flag.len() < n_units {
            self.flag.resize(n_units, false);
        }
        if let Some(f) = self.flag.get_mut(i) {
            if !*f {
                *f = true;
                self.list.push(i);
                self.n += 1;
            }
        }
    }

    /// Unit `i` fired under its current parameters — it is fresh again.
    pub fn clear(&mut self, i: usize) {
        if let Some(f) = self.flag.get_mut(i) {
            if *f {
                *f = false;
                self.n -= 1;
            }
        }
    }

    /// Check the invariants a loaded snapshot must hold before a settle reads this set: every flag belongs to a
    /// unit, every listed unit is inside the flags, and the count matches the set flags (a crafted snapshot that
    /// breaks one would otherwise index out of bounds or underflow in `Field::drain`).
    pub fn check(&self, n_units: usize) -> Result<(), String> {
        if self.flag.len() > n_units {
            return Err(format!("stale flags {} > units {n_units}", self.flag.len()));
        }
        if let Some(&u) = self.list.iter().find(|&&u| u >= self.flag.len()) {
            return Err(format!("stale list entry {u} >= flags {}", self.flag.len()));
        }
        let set = self.flag.iter().filter(|&&f| f).count();
        if set != self.n {
            return Err(format!("stale count {} != set flags {set}", self.n));
        }
        if self
            .flag
            .iter()
            .enumerate()
            .any(|(u, &f)| f && !self.list.contains(&u))
        {
            return Err("a stale flag is missing from the stale list".into());
        }
        Ok(())
    }

    /// Whether any unit is stale.
    pub fn any(&self) -> bool {
        self.n > 0
    }

    /// The stale units in ascending order (entries a firing already cleared are dropped).
    pub fn pending(&mut self) -> &[usize] {
        let flag = &self.flag;
        self.list.retain(|&u| flag[u]);
        self.list.sort_unstable();
        self.list.dedup();
        &self.list
    }

    /// Follow a unit compaction: `keep[i]` says whether old unit `i` survives (survivors keep their order).
    pub fn compact(&mut self, keep: &[bool]) {
        let mut k = 0;
        self.flag.retain(|_| {
            k += 1;
            keep.get(k - 1).copied().unwrap_or(false)
        });
        self.list = (0..self.flag.len()).filter(|&i| self.flag[i]).collect();
        self.n = self.list.len();
    }
}

/// The async FIELD — a [`Membrane`] (the cell medium) + units + the event scheduler. No tick: perturb
/// cells and the cascade settles. Deterministic in construction order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    /// The membrane — its `medium` IS the field's cells; interfaces attach here.
    pub mem: Membrane,
    /// The units (organs).
    pub units: Vec<Unit>,
    /// Adjacency: `readers[c]` = indices of units that read cell `c` (fixed order ⇒ deterministic).
    readers: Vec<Vec<usize>>,
    /// Adjacency: `writers[c]` = indices of units that write cell `c` (the mirror of `readers`). A
    /// unit writes exactly one cell, so this is usually 0 or 1 units — it turns the per-event "which
    /// unit writes this cell" lookups (learn/grow) from an O(units) scan into O(1).
    writers: Vec<Vec<usize>>,
    /// Per-cell surprise EMA (prediction-error magnitude) — drives surprise-birth of new units.
    surprise: Vec<f32>,
    /// Per-cell SLOW surprise EMA (decay `credit_decay`) — the lagging reference for `learning_progress`: when the
    /// fast `surprise` has caught this slow trace, surprise has PLATEAUED (dSurprise/dt ≈ 0) = the organism has
    /// learned all it can here. Its own history, so the plateau predicate needs no authored threshold.
    /// `#[serde(default)]` + lazily sized, so a fresh organism and old snapshots are byte-identical.
    #[serde(default)]
    surprise_slow: Vec<f32>,
    /// SURPRISE-DRIVEN PLASTICITY controller state (opt-in `Config::plasticity_anneal`) — per learned CELL: a
    /// two-timescale error trace (`err_fast` decay `surprise_decay`, `err_slow` decay `credit_decay`) and the
    /// annealing `learn_gain` the delta step is scaled by. Lazily sized; empty and untouched when the feature is
    /// off, so it never affects the byte-identity path. It is CONTROLLER STATE (the gain a loaded organism learns
    /// with next), so it is serialized — `#[serde(default)]` loads an older snapshot empty (gain 1.0).
    #[serde(default)]
    err_fast: Vec<f32>,
    #[serde(default)]
    err_slow: Vec<f32>,
    #[serde(default)]
    learn_gain: Vec<f32>,
    /// Per-unit ENERGY — the alive economy: a unit earns by firing (doing useful work), pays upkeep,
    /// and DIES (is culled) when starved. Parallel to `units`.
    energy: Vec<f32>,
    /// Which units fired during the last `stimulate` (for the metabolism income).
    last_fired: Vec<bool>,
    /// Subjective time each unit was BORN (in events; parallel to `units`). A newborn's age gates
    /// culling — the "yolk"/development reserve so a young unit gets time to earn before it can starve.
    born_at: Vec<u64>,
    /// Per-unit PLASTICITY in `[0, 1]` (parallel to `units`): the learn rule multiplies it into every
    /// weight update, so a CONSOLIDATED unit (plasticity 0) is frozen — later learning, even a stray
    /// `learn_step` or a shared router touching its cell, cannot degrade a matured skill. Isolation at
    /// the source, not a caller convention. Default 1.0 (fully plastic).
    plasticity: Vec<f32>,
    /// Per-CELL reliability meter (keyed by the cell passed to [`Field::observe_reliability`]) — the crystal
    /// half. Three EMAs (fast/slow/leaky-peak) of a caller-supplied reliability signal on the substrate's OWN
    /// two timescales; [`Field::mastery_of`] reads a relative `height·settled` from it, and once mastery is
    /// high+stable the writers of that cell auto-consolidate (`config.consolidation_rate`). Left EMPTY (never
    /// allocated) until the first `observe_reliability` call, so a 1st-order organism and old snapshots are
    /// byte-identical (`#[serde(default)]`).
    #[serde(default)]
    reliability: Vec<Reliability>,
    /// Per-unit projection SCALE / gain (parallel to `units`): the activation is `tanh(scale·(Σw·x+b))`,
    /// so `scale` sets the feature bandwidth as a LIVE, HERITABLE trait (not baked into the weights). A hot
    /// hyperparameter the substrate can tune by competition instead of a hand-set constant. Default 1.0.
    #[serde(default)]
    scale: Vec<f32>,
    /// Per-unit LEARNING-RATE factor (parallel to `units`): `learn_step` multiplies it into this unit's own
    /// weight updates, so `lr` is a LIVE, HERITABLE trait — the dominant tuning knob, promoted from the
    /// call-site constant so the substrate can compete over it too. Default 1.0.
    #[serde(default)]
    lr: Vec<f32>,
    /// Per-unit MUTATION strength σ (parallel to `units`): how far an offspring's constants stray from a
    /// parent's on [`Self::reproduce`]. It is ITSELF heritable and self-adapts (Schwefel), so there is no
    /// hand-set mutation constant — the lineage tunes its own mutability. Default 0.3.
    #[serde(default)]
    mut_rate: Vec<f32>,
    /// Per-unit MOMENTUM / damping μ (parallel to `units`): makes [`Self::learn_step`] a 2nd-order (heavy-ball)
    /// delta rule `v ← μ·v + Δw; w ← w + v` instead of the 1st-order `w ← w + Δw`. `μ=0` (default) is
    /// byte-identical to the historical rule. A HERITABLE trait, so the organism grows its own critical damping.
    #[serde(default)]
    momentum: Vec<f32>,
    /// Per-unit weight-VELOCITY buffer (the 2nd-order state; parallel to `units`, each a `weights+1` vector for
    /// the bias). Lazily sized in `learn_step` only when `μ != 0`. SERIALIZED with the snapshot, so a heavy-ball
    /// learner resumes with its momentum; `serde(default)` only lets a snapshot that predates the field load it
    /// empty (it then re-warms). This is what makes the tuning loop a damped oscillator.
    #[serde(default)]
    velocities: Vec<Vec<f32>>,
    /// Per-unit RECURSIVE-LEAST-SQUARES state (parallel to `units`; `Config::readout_rls`): the inverse input
    /// covariance each RLS-corrected unit carries, keyed to the reads it was built on. EMPTY until the first RLS
    /// step (so an organism that never turns the lever on carries nothing), then kept in lock-step with `units`.
    /// Serialized, so an RLS readout resumes with its covariance.
    #[serde(default)]
    rls: Vec<RlsState>,
    /// Reusable Δw scratch for [`Self::learn_step`]'s writer loop — filled and applied per writer, then kept
    /// for the next event so the per-writer weight-update Vec is allocated once, not once per writer per event.
    /// Pure transient (`#[serde(skip)]` ⇒ never serialized, so snapshot bytes are unchanged); re-grows on use.
    #[serde(skip)]
    learn_scratch: Vec<f32>,
    /// [`Self::learn_feature_directions`]'s backprojection accumulator (reused, empty between uses). Transient
    /// (`#[serde(skip)]`).
    #[serde(skip)]
    backproj: Backproj,
    /// `true` while several heads learn on ONE settled state (`observe_from`, `dream`): their backprojected errors
    /// accumulate in [`Self::backproj`] and apply once — see [`Self::begin_direction_batch`]. Transient
    /// (`#[serde(skip)]`): a batch never outlives the call that opened it.
    #[serde(skip)]
    dir_batching: bool,
    /// How many settles stopped at their `budget` with work still pending — see [`Self::truncated_settles`].
    /// A diagnostic (`#[serde(skip)]` ⇒ never serialized, so snapshot bytes are unchanged; 0 after a load).
    #[serde(skip)]
    truncated_settles: u64,
    /// Reusable memo-KEY scratch for `fire_nested` (the read values' bit patterns), so a nested cell's memo HIT
    /// allocates nothing. Pure transient (`#[serde(skip)]`, never read before being refilled).
    #[serde(skip)]
    memo_key: Vec<u32>,
    /// Whether `readers`/`writers` are in CANONICAL form — exactly what [`Self::reindex_adjacency`] would rebuild
    /// from `units` right now (each bucket ascending, one entry per read occurrence plus its forward-factor edge). Every
    /// `Field` method that edits the unit topology or the adjacency maintains it (`add_unit`/`grow` append in
    /// canonical order; `connect`/`set_factor` clear it; a re-index/rebuild sets it), so a cull that kills nobody
    /// can skip a re-index that would reproduce the adjacency verbatim, and a cull that does kill can REMAP the
    /// buckets instead of rebuilding them. Transient (`#[serde(skip)]` ⇒ `false` on load: the first cull
    /// re-indexes, exactly as before). NOT tracked: a caller mutating the `pub units[i].reads`/`write` directly
    /// (which already bypasses the fire-on-change adjacency) — debug builds assert the invariant at every cull.
    #[serde(skip)]
    adj_canonical: bool,
    /// Per-unit DIRTY set ([`Stale`]: lazily-sized flags + their list + count): the unit's parameters (weights /
    /// bias / scale) changed OUTSIDE of firing — the delta rule, feature-direction learning, nested credit, mutation,
    /// heredity, a prior/scale setter — so the value in the cell it writes is STALE. Fire-on-change re-fires a unit
    /// only when a cell it READS changes, so without this a repeated input kept reading the old output: the error
    /// froze and the weight overshot. [`Self::drain`] re-fires the still-dirty units once the natural cascade is
    /// quiescent (the REFRESH) and any firing clears the flag — so a settle in which every dirty unit fires
    /// naturally is byte-identical. The refresh is not an event of subjective time (no clock tick) and never writes
    /// a cell the caller supplied ([`Self::held`]). A newborn is NOT marked (it first fires when an input it reads
    /// changes). Serialized (`#[serde(default)]`: an older snapshot loads with nothing stale), so a flag pending at
    /// snapshot time is carried across a load.
    #[serde(default)]
    dirty: Stale,
    /// Per-CELL HOLD stamps (lazily sized): a cell is HELD while `held[c] == hold_seq` — the cells the caller
    /// SUPPLIED to the settle in progress ([`Self::settle_held`] bumps `hold_seq` before and after the drain, so
    /// nothing is held between settles). The staleness REFRESH of [`Self::drain`] never writes a held cell, so a
    /// freshness re-fire never overwrites a value the caller just gave (teacher forcing, a committed decision): a
    /// unit it reaches that writes one stays dirty until a settle that does not supply that cell. The natural
    /// cascade (a unit whose reads the inputs changed) still overwrites a held cell, exactly as before. Transient.
    #[serde(skip)]
    held: Vec<u64>,
    /// The current hold stamp for [`Self::held`] (transient).
    #[serde(skip)]
    hold_seq: u64,
    /// Per-CELL observed min / max / count for [`Config::var_gated_norm`] — the DETACHED input-variation signal
    /// `learn_step` reads to mask CONSTANT reads out of the width-norm denominator. A read is treated as constant
    /// (dropped) only once it has been seen ≥ 2 times with ZERO spread (`read_max == read_min`); a read seen <2
    /// times is still charged, so the denominator never collapses during warm-up (which an EMA-variance-from-0
    /// would, mistaking a just-seen input for a constant one). Exact and parameter-free — no rate, no threshold
    /// beyond float-equality. Grown + updated in `learn_step` only when `var_gated_norm` is on (empty and
    /// untouched otherwise ⇒ byte-identical off). serde-default.
    #[serde(default)]
    read_min: Vec<f32>,
    #[serde(default)]
    read_max: Vec<f32>,
    #[serde(default)]
    read_count: Vec<u32>,
    /// Per-unit CONTRIBUTION credit EMA (parallel to `units`): how much the downstream readout RELIES on
    /// this unit (|weight·activation|), accumulated in [`Self::learn_step`]. This is the fitness the alive
    /// economy should select on — a USEFUL-but-quiet unit earns here even if a firing gate would starve it
    /// (the make-or-break for evolving constants: select on contribution, not mere firing). Default 0.0.
    #[serde(default)]
    credit: Vec<f32>,
    /// SURPRISE-CURRENCY (Gap 7 spike) — per-CELL running mean/variance EMAs (parallel to the medium), updated in
    /// the settle path ONLY when [`Config::surprise_currency`] is on (else empty and never touched ⇒ byte-identical).
    /// `surprise_reduction_of` reads `cell_var` for the ablation importance `Σ w²·Var(cell)`. Transient-safe: a
    /// snapshot loads them empty (serde-default) and they re-warm.
    #[serde(default)]
    cell_mean: Vec<f32>,
    #[serde(default)]
    cell_var: Vec<f32>,
    /// SURPRISE-CURRENCY end-to-end sensitivity (Gap 7 spike, increment 1.5) — per-CELL running EMA of the SQUARED
    /// end-to-end sensitivity `(∂output/∂cell)²`, chained through the hidden backbone by a local reverse
    /// (adjoint / transpose-Jacobian) sweep in the learn path. Touched ONLY when [`Config::surprise_currency`] is on
    /// (else empty and never written ⇒ byte-identical). `surprise_reduction_of` reads it for the END-TO-END ablation
    /// importance `(∂output/∂cell)²·Var(cell)`, which stays causal THROUGH a random backbone where the immediate
    /// `Σ w²·Var` degenerates to reliance. Serde-default so a snapshot loads it empty and it re-warms.
    #[serde(default)]
    cell_sens2: Vec<f32>,
    /// Deterministic PRNG state — for the random-projection weights of grown HIDDEN units (diverse
    /// features), so growth stays byte-reproducible (no external randomness).
    rng_state: u64,
    /// Deterministic ACT-SAMPLING PRNG state — advanced ONLY to break an EXACT tie among the best-scoring options
    /// ([`Self::select_explore`], and the default [`crate::config::ActRule::Valence`] rule of `alive(Event::Act)`);
    /// a decision with one best option never draws. Separate from `rng_state` so growth and acting never
    /// interleave; `#[serde(default)]` and serialised, so a snapshot round-trip replays the same tie-breaks.
    /// Seeded from a fixed constant (and re-derived by [`Self::rng_seed`]) — never wall-clock, so it is machine-
    /// independent and deterministic, the founding invariant.
    #[serde(default = "default_act_rng")]
    act_rng: u64,
    /// DIAGNOSTIC — the spread (population std-dev) of the active rule's decision score across the options offered
    /// to the LAST `alive(Event::Act)` (the valence `A` or the free-energy `G`). Near-zero means the decision was
    /// DEGENERATE: every option scored the same, so selection was effectively a tie-break —
    /// the silent-failure signature of a flat reward, a collapsed value head, or a state key so coarse every option
    /// maps to one cell. A client reads [`Self::g_spread`] and asserts it is `> eps` to catch "your options are
    /// tied" that no other surface reports. `NaN` until the first act (`None` internally, so a snapshot round-trip
    /// stays "no act yet" rather than the misleading 0.0 a bare `f32` skip would default to).
    /// Transient/observability only (`#[serde(skip)]`), written from a pure read of the scores the rule already
    /// computes ⇒ it never influences a choice.
    #[serde(skip)]
    last_g_spread: Option<f32>,
    /// The organism's SUBJECTIVE time — a monotonic count of events (firings) it has processed. There
    /// is no wall-clock (that varies by machine and breaks determinism) and no compute-tick; time is
    /// the substrate's OWN activity. Deterministic and machine-independent, so `changed_at`/`age`/
    /// `recency` are byte-replayable. Time is DATA here, not a clock that drives compute.
    clock: u64,
    /// Per-cell subjective time of last change (in events; parallel to the medium). A cell's `age` —
    /// events since it last changed — is "time placement": information the value alone doesn't carry
    /// (a just-flipped cell vs a long-stable one look identical by value, but not by age).
    changed_at: Vec<u64>,
    /// GRADED VALUE per cell — the substrate's per-cell value/preference store (v1.3.x purpose/planning). A
    /// value is corrected LOCALLY ([`Self::relax_values`]: `cost + min` of connected neighbours) toward a fixed
    /// point; a PREFERRED cell is CLAMPED ([`Self::prefer`]) as a source. It is PER-CELL — an online correction
    /// of one cell does not drag the others — which is why it works where a grow/`learn_step` readout (one
    /// shared bias couples every cell) does not. `f32::MAX` = unknown/unreachable. serde-default so old
    /// snapshots load empty and hydrate on first use.
    #[serde(default)]
    values: Vec<f32>,
    /// Which value cells are CLAMPED — a preferred/source value [`Self::relax_values`] must not overwrite.
    #[serde(default)]
    clamped: Vec<bool>,
    /// GRADED ACTION-DRIVE per cell — the aversive/reinforcing twin of `values`. [`Self::reinforce`]
    /// ACCUMULATES onto it (−=aversion, +=commitment); [`Self::drive_of`] reads it. Unlike `values` it is NOT
    /// clamped and NOT relaxed — the discovered policy simply WEIGHS it (`+ drive`, symmetric), so a strongly-
    /// avoided action can still win on the value gradient when nothing better remains (soft, not a hard barrier).
    /// Goal-dependence is by the
    /// KEY the caller reinforces (a `(goal,state,action)` cell). serde-default so old snapshots load empty.
    #[serde(default)]
    drive: Vec<f32>,
    /// ACTION-DWELL (opt-in `Config::action_dwell`) — the GROWN commitment DURATION per option cell, learned by
    /// reward: an action that keeps earning when repeated has its dwell extended, one that stops earning released.
    /// Starts at 1 (re-decide every step). serde-default empty ⇒ byte-identical when the actor doesn't use dwell.
    #[serde(default)]
    dwell_len: Vec<f32>,
    /// The option the actor is currently COMMITTED to (opt-in dwell), and how many held steps remain. The live
    /// commitment decides the NEXT act, so it is serialized (a reloaded organism keeps holding what it held);
    /// `#[serde(default)]` loads an older snapshot uncommitted (it re-decides).
    #[serde(default)]
    dwell_committed: Option<usize>,
    #[serde(default)]
    dwell_remaining: f32,
    /// FAST-WEIGHT relational binding — the DISPOSABLE per-puzzle RULE store. [`Self::bind`] accumulates,
    /// in ONE pass, which cells COVARY across demonstrations (a relation, colour-independent — so it generalises
    /// to held-out values), and [`Self::complete`] fills a cell from its strongest-bound known cell. Few-shot
    /// rule inference is a CONVERGENCE problem, not a sample-complexity one: bind the relation
    /// once on the fast path instead of grinding the values a hundred times on the slow grown path. A rule for
    /// THIS puzzle ([`Self::unbind`] discards it), but a snapshot taken mid-puzzle must still answer the same, so it
    /// is serialized as a key-sorted list (`#[serde(default)]` — an older snapshot loads it empty).
    #[serde(default, with = "sorted_map")]
    assoc: HashMap<(usize, usize), f32>,
    /// SECOND-ORDER CO-CREDIT (the pairwise sibling of the per-unit credit `learn_step` accrues). Keyed by an
    /// ordered cell pair `(i, j)` with `i < j`, each entry is an EMA of `err · x_i · x_j` — the gradient a
    /// virtual product weight `w_ij·x_i·x_j` would receive. A pair whose PRODUCT consistently explains the
    /// residual error accumulates large `|co_credit|` = SYNERGY, which `co_activate`'s co-occurrence cannot see.
    /// Read by [`Self::co_credit_step`] to shape the concept-map and by `grow` (`Config::grow_products`) to pick
    /// the pair a PRODUCT feature is born on. Untouched unless [`Self::accrue_co_credit`] is called — every
    /// existing path stays byte-identical. It steers future births, so it is serialized as a key-sorted list
    /// (`#[serde(default)]` — an older snapshot loads it empty).
    #[serde(default, with = "sorted_map")]
    co_credit: HashMap<(usize, usize), f32>,
    /// PER-TARGET CO-CREDIT (`Config::products_per_target`): keyed `(target, i, j)` with `i < j`, the EMA of
    /// `err · x_i · x_j` for that target's own error, stored as `(value, the target's step it was last updated at)`
    /// so a pair not accrued since decays lazily on read. Empty unless the mode is on.
    /// Ordered, so a birth reads one target's pairs as a range; serialised as the same sorted pair list.
    #[serde(default, with = "ordered_map")]
    target_credit: alloc::collections::BTreeMap<(usize, usize, usize), (f32, u64)>,
    /// Per target cell, how many times its per-target co-credit has been accrued (the clock of `target_credit`).
    #[serde(default, with = "sorted_map")]
    target_ticks: HashMap<usize, u64>,
    /// CRYSTAL mode (v1.3.1). When `true` the organism is FROZEN — a snapshot of competence at a moment in
    /// time: no growth, no learning, no culling, no map/economy change; only fire-on-change → settle (pure
    /// inference) remains. It is [`Self::crystallize`] / [`Self::thaw`], a mode flag over the primitives we
    /// have (not per-unit plasticity, which is preserved), so freezing costs no training compute — the
    /// microcontroller / evaluation / deployment path. serde-default `false` (Alive) so old snapshots load
    /// alive. The complement of the (forthcoming) ALIVE loop: alive is learning-at-inference, crystal is
    /// inference only.
    #[serde(default)]
    frozen: bool,
    /// GROWN CONCEPT-METRIC — an internal MATRYOSHKA position per cell (a flat `n_cells · pos_dim` buffer,
    /// row-major). This is NOT modality geometry (pixels live in the perception layer); it is the organism's
    /// OWN cognitive map: a cell born to reduce another's residual is placed NEAR its creators
    /// ([`Self::place_near`]) and cells that fire together drift CLOSER ([`Self::co_activate`]), so co-useful
    /// concepts cluster and distance means "how related." The embedding is NESTED (Matryoshka): the FIRST
    /// dimensions carry the most general structure (they are shaped by all the early co-activation), and a new
    /// dimension is GROWN ([`Self::grow_dimension`]) only when the map gets stuck — each added axis captures a
    /// finer residual distinction. So a PREFIX of the dimensions is a valid coarse map ([`Self::distance`]
    /// with `dims < pos_dim`): cheap coarse search first, more dimensions only for the power you need — the
    /// vector-DB coarse-to-fine selector. Domain-blind, opt-in (lazily seeded), serde-defaulted.
    #[serde(default)]
    pos: Vec<f32>,
    /// Current dimensionality of the concept-metric (grows on demand; starts at 2 the first time it is used).
    #[serde(default)]
    pos_dim: usize,
    queue: alloc::collections::VecDeque<usize>,
    queued: Vec<bool>,
    /// The tunable META-parameters (energy economy, EMA smoothing, evolution bounds, heritable-trait seeds).
    /// An old snapshot keeps its genome: a 1.6 lever it has no key for takes its `Config::legacy()` value, and a
    /// snapshot with no config at all is `Config::legacy()` (`config::snapshot_config`).
    #[serde(
        default = "Config::legacy",
        deserialize_with = "crate::config::snapshot_config"
    )]
    config: Config,
    /// The organism's METABOLISM — a scalar energy account for the whole organism (distinct from the per-unit
    /// `energy`): [`Self::live`] and a metered [`Self::dream`] burn `metabolic_cost` from it, [`Self::live`] reports
    /// the HUNGER `1 − metabolism / energy_cap`, and [`Self::should_dream`] gates dreaming on it. Acting does not
    /// read or write it (the metabolic actor that did was removed with the `ActScore` rule), so nothing in core
    /// refills it. Starts fed at `energy_cap`. serde-default (an old snapshot without it loads 0 = starving).
    #[serde(default)]
    metabolism: f32,
    /// Cumulative count of SELF-SET goals the organism has REACHED (a taken option landed on a clamped preferred
    /// cell while [`Config::goal_reward`] is on). Measurement/observability only; stays 0 when the reward is off.
    #[serde(default, skip)]
    goals_reached: u64,
    /// Running total of energy DISCARDED by the `energy_cap` ceiling: when `income − upkeep` would push a unit
    /// past `energy_cap`, the excess is silently dropped by `.min(energy_cap)` in the metabolize paths. Accounted
    /// here so the energy economy has a denominator for where its energy goes (an efficiency reference). Pure
    /// observability — never read back into any dynamics, so `energy_of`/cull/traits are unchanged and the
    /// byte-identity oracle is unaffected. SCOPE: this counts the metabolize cap-overflow ONLY; culled-unit energy
    /// and metabolic-pool burns are separate conservation questions, deliberately NOT folded in here.
    #[serde(default, skip)]
    dissipated_overflow: f32,
    /// Cumulative count of nested-cell sub-settles that actually ran (memo misses) — measurement only, so not
    /// serialized: a loaded organism counts from 0 (its memo crosses the snapshot, so the per-step misses match).
    #[serde(default, skip)]
    nested_settles: u64,
    /// TEMPORAL TRACES — leaky integrators refreshed once per settle, in registration order (so a
    /// trace-of-a-trace pools an already-updated summary in ONE pass). The multi-scale, fractal TIME basis.
    /// `#[serde(default)]` so an old snapshot loads with no traces.
    #[serde(default)]
    traces: Vec<Trace>,
    /// DELAYS — exact unit-delay registers shifted once per external event by [`Self::advance_delays`], the
    /// native EXACT-order memory. In registration order, so a delay-of-a-delay is `x[t−2]` in one
    /// pass. `#[serde(default)]` so an old snapshot loads with no delays.
    #[serde(default)]
    delays: Vec<Delay>,
    /// FREE-CELL list — indices of cells vacated by [`Self::cull`] (a grown feature whose unit died and whose
    /// slot nothing references any more). `birth_feature` reuses these before growing the medium, so a continuous
    /// grow+cull stream keeps the medium (and its per-cell parallel vectors + the O(cells) cull adjacency rebuild)
    /// bounded at the PEAK concurrent cell count instead of leaking O(steps) — the deploy wall. The pool decides
    /// WHICH slot the next grown feature takes (and whether the medium grows), so it is serialized in pop order
    /// (`#[serde(default)]` — an older snapshot loads it empty and simply appends).
    #[serde(default)]
    free_cells: Vec<usize>,
    /// RESONATORS — 2nd-order complex-pole band-pass cells, advanced one step per settle (`stimulate` call) by
    /// [`Self::tick_resonators`]. The band-pass twin of `traces`; `#[serde(default)]` so an old snapshot loads none.
    #[serde(default)]
    resonators: Vec<Resonator>,
    /// EPISODIC MEMORY — a small buffer of stored [`Episode`]s (snapshots of chosen cell-sets). The addressable,
    /// replayable experience the organism carries: [`Self::store_episode`] captures one, [`Self::replay`] re-injects
    /// it through the existing `stimulate` path (replay IS re-stimulating from a remembered cell-state). It CLONES
    /// with [`Self::fork_mutated`] (a mutated clone inherits the parent's experience — the keystone of the genome
    /// loop) and round-trips through a snapshot. `#[serde(default)]` + never touched unless a caller stores an
    /// episode, so a 1st-order organism and old snapshots are byte-identical.
    #[serde(default)]
    episodes: Vec<Episode>,
    /// PER-CELL REACTIVATION PRESSURE `P_c` — the autonomous heartbeat / living-memory self-trigger (D6/D7,
    /// §3). Parallel to the medium. While the field is quiescent, [`Self::live`] ramps it with dormancy
    /// (`P_c += heartbeat_k·(1−recency_c)`) — a leaky integrate-and-fire; when it crosses `heartbeat_theta` the
    /// longest-dormant grown cell spontaneously fires (offline pattern completion / "dreaming"), the EXISTING
    /// contribution economy scores the result, and the memory is kept ([`Self::consolidate`]) or pruned
    /// ([`Self::cull`]); then `P_c` discharges to 0. This is the ONLY new state the heartbeat adds. `#[serde(
    /// default)]` and touched ONLY by `live`, so a field whose heartbeat is never invoked is byte-identical to
    /// before this existed.
    #[serde(default)]
    pressure: Vec<f32>,
    /// Per-option value ENSEMBLE of the FreeEnergy acting rule — option-cell → a set of varied-prior reward
    /// estimators. Their MEAN is the pragmatic prediction `μ_o` and their VARIANCE the epistemic `σ_o²` (read via
    /// [`crate::grouping::aggregate`], the same disagreement `select_explore` uses). [`Self::alive`]`(Act)` folds the
    /// last reward (bootstrapped by `Config::act_discount`) into the chosen option's members
    /// (`m ← m + lr·(target − m)`) so a visited option converges (σ² shrinks by `(1−lr)²` per visit — σ is a visit
    /// count, not a reward-noise estimate) while a never-visited one keeps its prior spread. This is the actor's
    /// LEARNED POLICY (not transient scratch like `assoc`, which is a rule for one puzzle) — it is SERIALISED so a
    /// TRAINED organism survives `snapshot`/`load` (without it a learned preference evaporated on reload).
    /// `#[serde(default)]` so a snapshot taken with the actor OFF (or an old one) loads an empty map = byte-identical;
    /// a `BTreeMap` keeps the serialised bytes deterministic (option-id order), never observed by the computation
    /// (keyed access only). A pre-1.6 `valence_actor` organism kept its valence here too; the Valence rule reads such
    /// an entry through the migration in [`Self::valence_record`].
    #[serde(default)]
    value_ensemble: alloc::collections::BTreeMap<usize, Vec<f32>>,
    /// The Valence (default) acting rule's LEARNED POLICY — option-cell → its [`ValenceRecord`]: the reward estimate
    /// `μ_o` (the EMA of the fold targets, from exactly 0) and the EXACT visit count `n_o`. The exploration bonus is
    /// a pure function of the count, `σ_o² = (σ0·(1−lr)^n_o)²`, so "untried" is exactly `n_o = 0` and a zero-reward
    /// option's `μ_o` stays exactly 0 (the 24-member ensemble this replaces for the Valence rule carried two numbers
    /// in 24 floats and an f32 prior-mean residue that, past ~115 zero-reward visits, outweighed σ² and made the
    /// argmax prefer the MOST-visited option). An option with no record here but a [`Self::value_ensemble`] entry
    /// (an old `valence_actor` snapshot, or an organism that ran `FreeEnergy`) is read by deriving `(μ, n)` from that
    /// ensemble ([`Self::valence_record`]). Serialised with the same `BTreeMap` determinism; `#[serde(default)]` so
    /// an old snapshot loads an empty map.
    #[serde(default)]
    valence: alloc::collections::BTreeMap<usize, ValenceRecord>,
}

/// One option's record under the Valence acting rule ([`Field::valence_of`]): the signed reward estimate and the
/// exact number of times the option was taken (and its reward folded).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct ValenceRecord {
    /// `μ` — the EMA of the fold targets (`μ ← μ + free_energy_lr·(target − μ)`), starting at exactly 0.
    mean: f32,
    /// `n` — how many times the option's reward was folded (saturating). `0` = never visited = untried.
    visits: u32,
}

/// An event the organism LIVES against — the single input to [`Field::alive`]. The whole v1.3.1 loop is one
/// call: feed an event, get an outcome. There is no orchestration — the morph decides what to do with it.
pub enum Event<'a> {
    /// A perceptual event: learn to predict `target` at `target_cell` from `inputs` (learning at inference —
    /// grows on surprise and self-sizes). In CRYSTAL this degrades to pure inference (settle only).
    Perceive {
        inputs: &'a [(usize, f32)],
        target_cell: usize,
        target: f32,
        lr: f32,
        budget: usize,
    },
    /// A SUGGESTION — an example demonstrating a rule (an ARC training grid): bound on the fast path, relation-
    /// level, so it generalises to held-out values. Works in CRYSTAL too (the deployed morph few-shot-adapts).
    Suggest { observation: &'a [(usize, f32)] },
    /// A QUESTION — the KNOWN cells of a puzzle; the morph completes the `unknown` cells toward the consistency
    /// its suggestions bound (clamp known, complete unknown). Returns the filled-in answer.
    Answer {
        known: &'a [(usize, f32)],
        unknown: &'a [usize],
    },
    /// An ACT event: one step of the native policy loop. Given candidate `options` (action or library-skill
    /// cells, each carrying a goal cost-to-go via [`Field::value_of`] set by the planner), fold the `last`
    /// action's reward (`None` on the first step) and return the next option; the world executes it and reports
    /// its reward on the next `Act`. ONE rule runs — [`crate::Config::effective_act_rule`]:
    ///
    /// 1. [`crate::config::ActRule::Valence`] (the default) — fold the reward toward
    ///    `r + act_discount·max_{o'∈options} (μ_{o'} + σ_{o'}²)` into the option's `(μ, visits)` record (the
    ///    successor's bonus is bootstrapped too, so an unexplored state far down a path keeps the path
    ///    attractive), then ARGMAX
    ///    `−value + μ + σ²` with `σ² = (σ0·(1−lr)^visits)²` (shrinks with every visit); exact ties are broken
    ///    uniformly by the seeded act PRNG.
    /// 2. [`crate::config::ActRule::FreeEnergy`] (or the deprecated `free_energy_actor`) — the same fold into the
    ///    option's value ensemble, then ARGMIN `G = value + (μ−s*)² − σ²` (ties → first).
    ///
    /// On both rules `goal_reward` adds its self-reward, [`Field::g_spread`] records the spread of the decision
    /// score, and `action_dwell` then holds the choice.
    ///
    /// **Episode end.** Pass the final step's reward with an EMPTY `options`: the world offers no successor, so
    /// `last` is folded with no `act_discount` bootstrap (both rules), and `chosen` is `usize::MAX` (no option).
    /// Only the fold runs: nothing is scored, so the act PRNG and `g_spread` are untouched, and an `action_dwell`
    /// commitment is released.
    /// Without it, the next episode's first `Act` bootstraps the final step from the RESTART state's options,
    /// which on a multi-step task makes a short episode look better than a long one that pays more.
    Act {
        options: &'a [usize],
        last: Option<(usize, f32)>,
    },
}

/// What an [`Event`] produced.
pub enum Outcome {
    /// A `Perceive` event: `|error|` before the update, and whether a feature was born this step.
    Perceived { error: f32, born: bool },
    /// A `Suggest` event was bound.
    Suggested,
    /// An `Answer` event: the completed `(cell, value)` pairs for the unknown cells.
    Answered(Vec<(usize, f32)>),
    /// An `Act` event: the option the morph chose to execute next.
    Acted { chosen: usize },
}

/// The report of one [`Field::live`] heartbeat call — measurement only (what the organism did while
/// autonomous, with no external input). Deterministic; serde so a trace/UI can record the sleep.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    /// Internal ticks the organism ran before it rested (each an autonomous step of its OWN subjective time).
    pub ticks: usize,
    /// Spontaneous reactivations — a dormant memory cell that crossed the pressure threshold and self-fired.
    pub reactivations: usize,
    /// Reactivations whose memory still EARNED (readers still rely on it) and was frozen/kept ([`Field::consolidate`]).
    pub consolidated: usize,
    /// Grown cells the sleep economy PRUNED (a reactivated memory that earned nothing starved and was culled).
    pub culled: usize,
    /// Whether this call REST/slept — no unfinished work remained, so it dropped to basal metabolism and stopped.
    pub slept: bool,
    /// The organism's HUNGER `1 − metabolism/energy_cap` after the call (rises as being-awake burns energy).
    pub hunger: f32,
}

/// Coordinate `axis` of cell `i`'s deterministic low-discrepancy SEED position in `[-1, 1]` — the position
/// before growth or co-activation shapes it (per-axis golden-ratio additive recurrence, machine-independent,
/// no PRNG). Later axes use finer multipliers, so a freshly-grown dimension starts as a small distinct jitter.
fn scatter(i: usize, axis: usize) -> f32 {
    // distinct irrational multipliers per axis (golden-ratio-family), shrinking amplitude for higher axes so
    // a new (fine) dimension perturbs less than the coarse ones.
    let mult = [
        0.754_877_6f32,
        0.569_840_3,
        0.409_558_1,
        0.313_042_1,
        0.245_122_5,
    ];
    let m = mult[axis % mult.len()] + axis as f32 * 0.017;
    let amp = 1.0 / (1.0 + axis as f32); // coarse axes span [-1,1]; fine axes are tighter
    ((i as f32 * m).fract() * 2.0 - 1.0) * amp
}

/// Cosine DISTANCE `1 − cos(a,b)` over the first `dims` components (`0` = the longer of the two), with a
/// `1e-12` denominator floor and clamped to `[0, 2]` (the true range of `1 − cos`). The ONE cosine kernel
/// behind both [`Field::fingerprint_distance`] and [`Field::embed_distance`] — factored out of the two
/// (previously identical) loops so the behavioural distances can never drift apart. `~0` for identical keys.
pub fn cosine_distance(a: &[f32], b: &[f32], dims: usize) -> f32 {
    let full = a.len().max(b.len());
    let n = if dims == 0 { full } else { dims.min(full) };
    if n == 0 {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0.0),
            b.get(i).copied().unwrap_or(0.0),
        );
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    (1.0 - dot / denom).clamp(0.0, 2.0)
}

/// The empty organism — exactly [`Field::new`]`(0)` (the same PRNG seeds and a fed metabolism), so there is one
/// "empty field", not a second one with zeroed PRNG streams.
impl Default for Field {
    fn default() -> Self {
        Field::new(0)
    }
}

impl Field {
    /// A field of `n_cells` cells (all 0) and no units.
    pub fn new(n_cells: usize) -> Self {
        let mut mem = Membrane::new();
        mem.medium.resize(n_cells, 0.0);
        Field {
            mem,
            units: Vec::new(),
            readers: vec![Vec::new(); n_cells],
            writers: vec![Vec::new(); n_cells],
            surprise: vec![0.0; n_cells],
            surprise_slow: Vec::new(), // lazily sized on first grow (serde-default; byte-identical when unused)
            err_fast: Vec::new(), // surprise-driven-plasticity controller state (opt-in; empty when off)
            err_slow: Vec::new(),
            learn_gain: Vec::new(),
            energy: Vec::new(),
            last_fired: Vec::new(),
            born_at: Vec::new(),
            plasticity: Vec::new(),
            reliability: Vec::new(),
            scale: Vec::new(),
            lr: Vec::new(),
            mut_rate: Vec::new(),
            momentum: Vec::new(),
            velocities: Vec::new(),
            rls: Vec::new(),
            learn_scratch: Vec::new(),
            backproj: Backproj::default(),
            dir_batching: false,
            truncated_settles: 0,
            memo_key: Vec::new(),
            adj_canonical: true, // no units ⇒ the all-empty adjacency IS the canonical re-index
            dirty: Stale::default(),
            held: Vec::new(),
            hold_seq: 0,
            read_min: Vec::new(),
            read_max: Vec::new(),
            read_count: Vec::new(),
            credit: Vec::new(),
            cell_mean: Vec::new(),
            cell_var: Vec::new(),
            cell_sens2: Vec::new(),
            rng_state: 0x9E3779B97F4A7C15,
            act_rng: default_act_rng(),
            last_g_spread: None,
            clock: 0,
            changed_at: vec![0; n_cells],
            values: vec![f32::MAX; n_cells],
            clamped: vec![false; n_cells],
            drive: vec![0.0; n_cells],
            dwell_len: Vec::new(),
            dwell_committed: None,
            dwell_remaining: 0.0,
            assoc: HashMap::new(),
            co_credit: HashMap::new(),
            target_credit: alloc::collections::BTreeMap::new(),
            target_ticks: HashMap::new(),
            frozen: false,
            pos: Vec::new(),
            pos_dim: 0,
            queue: alloc::collections::VecDeque::new(),
            queued: Vec::new(),
            config: Config::default(),
            metabolism: Config::default().energy_cap, // born fed
            goals_reached: 0,
            dissipated_overflow: 0.0,
            nested_settles: 0,
            traces: Vec::new(),
            delays: Vec::new(),
            free_cells: Vec::new(),
            resonators: Vec::new(),
            episodes: Vec::new(),
            pressure: vec![0.0; n_cells], // autonomous-heartbeat reactivation pressure (D6/D7; live-only)
            value_ensemble: alloc::collections::BTreeMap::new(), // free-energy per-option LEARNED policy
            valence: alloc::collections::BTreeMap::new(), // valence per-option LEARNED policy: (μ, visits)
        }
    }

    /// The tunable substrate constants (energy economy, EMA smoothing, evolution bounds, heritable-trait
    /// seeds). See [`crate::Config`] and `docs/config-reference.md`.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Mutable access to the tunable constants — change them LIVE (dynamic tuning): `field.config_mut().init_lr
    /// = 0.5`. Takes effect on the next call that reads that constant.
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    /// Replace the whole config (e.g. deserialise an Optuna trial's parameters into a [`Config`] and apply it).
    /// Returns `&mut Self` for chaining.
    pub fn configure(&mut self, config: Config) -> &mut Self {
        self.config = config;
        self
    }

    /// The cell values (the medium).
    pub fn cells(&self) -> &[f32] {
        &self.mem.medium
    }
    /// Number of cells.
    pub fn len(&self) -> usize {
        self.mem.medium.len()
    }
    /// Whether the field has no cells.
    pub fn is_empty(&self) -> bool {
        self.mem.medium.is_empty()
    }

    /// Number of units (grown organs).
    pub fn n_units(&self) -> usize {
        self.units.len()
    }

    /// Pad the per-unit trait vectors to `units.len()` with their defaults. A no-op in normal operation
    /// (`add_unit` keeps them in lock-step), but it SELF-HEALS a snapshot deserialized before these fields
    /// existed: `scale`/`lr`/`mut_rate`/`credit` are `#[serde(default)]`, so an old snapshot loads them EMPTY
    /// while `units` is full — then a direct index (`scale[u]`, `credit[fj]`) would panic. Called at the entry
    /// of every path that indexes them directly, so a loaded organism can never crash on first use.
    fn hydrate_traits(&mut self) {
        let n = self.units.len();
        let (be, isc, ilr, imr) = (
            self.config.birth_energy,
            self.config.init_scale,
            self.config.init_lr,
            self.config.init_mut_rate,
        );
        self.energy.resize(n, be);
        self.born_at.resize(n, self.clock);
        self.plasticity.resize(n, 1.0);
        self.scale.resize(n, isc);
        self.lr.resize(n, ilr);
        self.mut_rate.resize(n, imr);
        self.momentum.resize(n, self.config.init_momentum);
        self.velocities.resize(n, Vec::new());
        self.credit.resize(n, 0.0);
    }

    /// Grow the cell pool (and adjacency) to at least `n` cells.
    pub fn ensure_cells(&mut self, n: usize) {
        if self.mem.medium.len() < n {
            self.mem.medium.resize(n, 0.0);
        }
        if self.readers.len() < self.mem.medium.len() {
            self.readers.resize(self.mem.medium.len(), Vec::new());
        }
        if self.writers.len() < self.mem.medium.len() {
            self.writers.resize(self.mem.medium.len(), Vec::new());
        }
        if self.surprise.len() < self.mem.medium.len() {
            self.surprise.resize(self.mem.medium.len(), 0.0);
        }
        if self.changed_at.len() < self.mem.medium.len() {
            self.changed_at.resize(self.mem.medium.len(), 0);
        }
        if self.values.len() < self.mem.medium.len() {
            self.values.resize(self.mem.medium.len(), f32::MAX);
        }
        if self.clamped.len() < self.mem.medium.len() {
            self.clamped.resize(self.mem.medium.len(), false);
        }
        if self.drive.len() < self.mem.medium.len() {
            self.drive.resize(self.mem.medium.len(), 0.0);
        }
        if self.pressure.len() < self.mem.medium.len() {
            self.pressure.resize(self.mem.medium.len(), 0.0);
        }
    }
}
