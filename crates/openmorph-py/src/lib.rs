// SPDX-License-Identifier: MIT-0
//! # openmorph (Python) — a Morph in Python
//!
//! A thin, disposable interface layer over the STABLE `openmorph_core::Field` surface (the organism is
//! persistent, the interface is disposable). Touches nothing in the core. Built with PyO3 + maturin into
//! wheels (`pip install openmorph`). Mirrors the WASM binding (`openmorph-wasm`) so JS and Python users
//! share one mental model.
//!
//! ```python
//! from openmorph import Morph
//! import math
//! m = Morph(2)                       # a 2-cell membrane
//! m.configure('{"max_hidden": 0}')   # linear-in-tanh: no hidden depth
//! for _ in range(2000):
//!     for x in (-1.0, 1.0):
//!         m.grow([0], [x], 1, math.tanh(1.5 * x), 0.1, 50)
//! m.stimulate([0], [1.0], 50)
//! print(m.cells()[1])                # ≈ tanh(1.5)
//! ```

// PyO3's `#[pymethods]` macro expands `PyResult`-returning methods with an `.into()` that clippy flags as a
// useless conversion — a known macro artifact, not our code.
#![allow(clippy::useless_conversion)]

use std::borrow::Cow;
use std::ops::Deref;

use openmorph_core::substrate::{Event, Field, Outcome};
use openmorph_core::Config;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};

/// Snapshot bytes handed to Python as a `bytes` object. (A `Vec<u8>` return becomes a `list[int]` under pyo3
/// 0.22: one Python int per byte, ~8× the heap of the blob and most of the snapshot time.)
type PyBytesOut = Cow<'static, [u8]>;

/// Reject a cell index outside the membrane (`0..len`) with a `ValueError` naming it — where the core would
/// otherwise silently ignore the cell or answer for a cell that does not exist.
fn check_cell(field: &Field, cell: usize, what: &str) -> PyResult<()> {
    let n = field.len();
    if cell < n {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{what} {cell} is out of range for a {n}-cell membrane"
        )))
    }
}

/// Pair parallel `idx`/`val` lists into the `(cell, value)` inputs a `Field` takes, REJECTING what the core would
/// silently drop: a length mismatch (a `zip` used to truncate the tail) or an input cell at or past `limit` — the
/// size of the membrane the core writes the inputs into (such a cell is never written, so the input was lost
/// without a word). `stimulate` / `learn_step` pass `field.len()`; `grow` passes [`grow_input_limit`].
fn inputs(idx: &[usize], val: &[f32], limit: usize) -> PyResult<Vec<(usize, f32)>> {
    if idx.len() != val.len() {
        return Err(PyValueError::new_err(format!(
            "idx and val must have the same length (got {} indices, {} values)",
            idx.len(),
            val.len()
        )));
    }
    if let Some(&c) = idx.iter().find(|&&c| c >= limit) {
        return Err(PyValueError::new_err(format!(
            "input cell {c} is out of range for a {limit}-cell membrane"
        )));
    }
    Ok(idx.iter().zip(val).map(|(&i, &v)| (i, v)).collect())
}

/// The first input cell `Field::grow` toward `target_cell` would silently DROP (`usize::MAX` = none). While the
/// target has no writer, the core births one that READS every input, growing the membrane to hold them — so no
/// input is lost (the `Field::new(0)` + `grow` idiom of the core proofs). Once it has a writer, the core grows the
/// membrane only to `target_cell + 1` before writing the inputs, so a cell past that is dropped from the step.
fn grow_input_limit(field: &Field, target_cell: usize) -> usize {
    if field.has_writer(target_cell) {
        field.len().max(target_cell + 1)
    } else {
        usize::MAX
    }
}

/// Merge a partial config JSON object onto `base`, so a caller only sets the keys they care about.
///
/// `Morph.configure` passes the organism's CURRENT config, so repeated calls accumulate (a second call used to
/// silently revert every earlier key to its default); the actors pass `Config::default()` (their documented
/// contract — see their `configure`). The patch must be a JSON OBJECT: `null`, arrays and scalars are rejected
/// (they used to be accepted and reset the whole config). An unknown key (e.g. a typo like `max_hiden`) is
/// REJECTED rather than silently dropped by serde — otherwise `configure('{"max_hiden": 0}')` would do nothing.
fn merge_config(base: &Config, json: &str) -> Result<Config, String> {
    let mut merged = serde_json::to_value(base).map_err(|e| e.to_string())?;
    let patch: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let Some(p) = patch.as_object() else {
        let kind = match patch {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "a boolean",
            serde_json::Value::Number(_) => "a number",
            serde_json::Value::String(_) => "a string",
            _ => "an array",
        };
        return Err(format!(
            "config patch must be a JSON object such as {{\"max_hidden\": 0}}, got {kind}"
        ));
    };
    let b = merged
        .as_object_mut()
        .ok_or("Config did not serialize to a JSON object")?;
    for (k, v) in p {
        if !b.contains_key(k) {
            return Err(format!("unknown config key: {k}"));
        }
        b.insert(k.clone(), v.clone());
    }
    serde_json::from_value(merged).map_err(|e| e.to_string())
}

/// The dtype kind (`'f'`, `'i'`, `'u'`, `'b'`, …) of `ob` if it is a 1-D numpy array (it exposes
/// `__array_interface__`, `ndim == 1` and `dtype.kind`), else `None`. Lists, tuples, masked arrays and every other
/// object return `None` and keep the element-wise extraction they always had.
fn ndarray_kind(ob: &Bound<'_, PyAny>) -> PyResult<Option<char>> {
    if ob.is_instance_of::<PyList>()
        || ob.is_instance_of::<PyTuple>()
        || !ob.hasattr("__array_interface__")?
        || ob.hasattr("mask")?
    {
        return Ok(None);
    }
    let ndim: Option<usize> = ob.getattr("ndim").and_then(|d| d.extract()).ok();
    if ndim != Some(1) {
        return Ok(None);
    }
    let kind: Option<String> = ob
        .getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract())
        .ok();
    Ok(kind.and_then(|k| k.chars().next()))
}

/// `ob.astype(dtype, copy=False).tobytes()` — the whole array across the FFI in ONE call, in native byte order.
fn ndarray_raw<'py>(ob: &Bound<'py, PyAny>, dtype: &str) -> PyResult<Bound<'py, PyBytes>> {
    let kwargs = PyDict::new_bound(ob.py());
    kwargs.set_item("copy", false)?;
    Ok(ob
        .call_method("astype", (dtype,), Some(&kwargs))?
        .call_method0("tobytes")?
        .downcast_into::<PyBytes>()?)
}

/// The 8-byte native-endian words of a `tobytes()` buffer (`chunks_exact` guarantees every chunk is 8 bytes).
fn words(raw: &[u8]) -> impl Iterator<Item = [u8; 8]> + '_ {
    raw.chunks_exact(8)
        .map(|c| <[u8; 8]>::try_from(c).expect("chunks_exact(8) yields 8-byte chunks"))
}

/// A 1-D float-vector argument: a `list`/`tuple` of numbers (extracted element-wise, exactly as before) OR a 1-D
/// numeric numpy array, which crosses the FFI in ONE `astype(float64).tobytes()` call instead of one Python object
/// per element. Each value then takes the same `f64 → f32` rounding the element-wise path applies (pyo3 reads a
/// Python float as `f64` and casts), so a numpy input is bit-identical to passing `arr.tolist()`.
struct Floats(Vec<f32>);

impl Deref for Floats {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        &self.0
    }
}

impl<'py> FromPyObject<'py> for Floats {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        match ndarray_kind(ob)? {
            Some('b' | 'i' | 'u' | 'f') => {
                let raw = ndarray_raw(ob, "float64")?;
                Ok(Floats(
                    words(raw.as_bytes())
                        .map(|w| f64::from_ne_bytes(w) as f32)
                        .collect(),
                ))
            }
            _ => ob.extract().map(Floats),
        }
    }
}

/// A 1-D cell-index (or colour-grid) argument: a `list`/`tuple` of ints (element-wise, as before) OR a 1-D integer
/// numpy array (one `astype(int64|uint64).tobytes()` crossing). A negative or oversized entry raises
/// `OverflowError`, as the element-wise path does.
struct Indices(Vec<usize>);

impl Deref for Indices {
    type Target = Vec<usize>;
    fn deref(&self) -> &Vec<usize> {
        &self.0
    }
}

impl<'py> FromPyObject<'py> for Indices {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let overflow = |_| PyOverflowError::new_err("can't convert the int to a usize cell index");
        match ndarray_kind(ob)? {
            Some('i') => words(ndarray_raw(ob, "int64")?.as_bytes())
                .map(|w| usize::try_from(i64::from_ne_bytes(w)).map_err(overflow))
                .collect::<PyResult<_>>()
                .map(Indices),
            Some('u') => words(ndarray_raw(ob, "uint64")?.as_bytes())
                .map(|w| usize::try_from(u64::from_ne_bytes(w)).map_err(overflow))
                .collect::<PyResult<_>>()
                .map(Indices),
            _ => ob.extract().map(Indices),
        }
    }
}

/// Snapshot bytes from a `bytes` (borrowed, no copy) or `bytearray` (one copy) — or, for backward compatibility,
/// the `list[int]` that `snapshot()` returned before it returned `bytes` (element-wise, as before).
fn snapshot_bytes<'a>(ob: &'a Bound<'_, PyAny>) -> PyResult<Cow<'a, [u8]>> {
    if let Ok(b) = ob.downcast::<PyBytes>() {
        return Ok(Cow::Borrowed(b.as_bytes()));
    }
    if let Ok(b) = ob.downcast::<PyByteArray>() {
        return Ok(Cow::Owned(b.to_vec()));
    }
    Ok(Cow::Owned(ob.extract::<Vec<u8>>()?))
}

/// A digital organism — an async membrane of cells that fires on change, settles, learns, and grows.
#[pyclass]
pub struct Morph {
    field: Field,
}

#[pymethods]
impl Morph {
    /// A Morph with `n_cells` cells (all 0) and no units.
    #[new]
    fn new(n_cells: usize) -> Self {
        Morph {
            field: Field::new(n_cells),
        }
    }

    /// Configure the tunable substrate constants from a JSON object string (see `docs/config-reference.md`).
    /// Only the keys you set are changed: the patch is merged onto the organism's CURRENT config, so successive
    /// calls accumulate (`configure('{"max_hidden": 0}')` then `configure('{"feature_lr": 0.02}')` keeps both).
    /// `reset=True` merges onto the historical defaults instead (`configure('{}', reset=True)` resets every
    /// lever; `apply_preset` uses it, so a preset is always defaults + its overlay). A non-object patch (`null`,
    /// an array, a scalar) or an unknown key raises `ValueError` and changes nothing.
    #[pyo3(signature = (json, reset=false))]
    fn configure(&mut self, json: &str, reset: bool) -> PyResult<()> {
        let base = if reset {
            Config::default()
        } else {
            self.field.config().clone()
        };
        let cfg = merge_config(&base, json).map_err(PyValueError::new_err)?;
        self.field.configure(cfg);
        Ok(())
    }

    /// The CURRENT config as a JSON object string — every lever, including the defaults you never set.
    /// `json.loads(m.config_json())["max_hidden"]` reads one; it round-trips through `configure`.
    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(self.field.config()).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Seed the deterministic growth PRNG (for byte-reproducible runs).
    fn rng_seed(&mut self, seed: u64) {
        self.field.rng_seed(seed);
    }

    /// Perturb the given cells and settle within `budget` events. Returns the number of events processed.
    /// `idx`/`val` are lists or 1-D numpy arrays of equal length; a length mismatch or a cell outside the
    /// membrane raises `ValueError`.
    fn stimulate(&mut self, idx: Indices, val: Floats, budget: usize) -> PyResult<usize> {
        let pairs = inputs(&idx, &val, self.field.len())?;
        Ok(self.field.stimulate(&pairs, budget))
    }

    /// The current membrane values.
    fn cells(&self) -> Vec<f32> {
        self.field.cells().to_vec()
    }

    /// The value of ONE cell — a single float read, without the whole-membrane copy `cells()` makes. A cell
    /// outside the membrane raises `ValueError`.
    fn cell(&self, cell: usize) -> PyResult<f32> {
        check_cell(&self.field, cell, "cell")?;
        Ok(self.field.cells()[cell])
    }

    /// Whether the last cascade reached a fixed point within budget.
    fn settled(&self) -> bool {
        self.field.settled()
    }

    /// Number of grown units (organs).
    fn n_units(&self) -> usize {
        self.field.n_units()
    }

    /// How many settles (of `stimulate`, `grow`, `learn_step`, `alive`, …) stopped at their `budget` with firings
    /// still pending — a TRUNCATED settle, whose cells are not the fixed point. Compare it before and after a call
    /// to detect one. Cumulative for this organism's life in memory (0 after `load`).
    fn truncated_settles(&self) -> u64 {
        self.field.truncated_settles()
    }

    // ── world-model builders (#332/#334): grow a forward model and plan over it ─────────────────────────────────

    /// GROW_BACKBONE — a fixed-width shared random-feature basis of `inputs` (`width` features at projection
    /// `scale`). Returns the new feature cell ids. The building block of the grown world model `imagine` (#332):
    /// reads = inputs ∪ backbone, then an `add_unit` readout per predicted feature. (`substrate_predict` idiom.)
    fn grow_backbone(&mut self, inputs: Vec<usize>, width: usize, scale: f32) -> Vec<usize> {
        self.field.grow_backbone(&inputs, width, scale)
    }

    /// ADD_UNIT — a linear readout unit that writes `write` from `reads` (init `weights` + `bias`); tune it with
    /// `learn_step`. Returns the unit id. Use one per world-model prediction head (next-feature / reward). A `write`
    /// or read cell past the membrane GROWS it to fit (how a new head claims its output cell); `weights` must have
    /// one entry per read cell (`ValueError` otherwise).
    fn add_unit(
        &mut self,
        reads: Vec<usize>,
        weights: Vec<f32>,
        bias: f32,
        write: usize,
    ) -> PyResult<usize> {
        if weights.len() != reads.len() {
            return Err(PyValueError::new_err(format!(
                "add_unit needs one weight per read cell (got {} reads, {} weights)",
                reads.len(),
                weights.len()
            )));
        }
        Ok(self.field.add_unit(reads, weights, bias, write))
    }

    // ── temporal abstraction / macro-actions (#91/#64): bank a skill, plant it as one addressable cell ───────────

    /// ENCAPSULATE — bank the grown subgraph that writes `cells` as a fresh standalone organism (a macro-action /
    /// skill). Returns a new `Morph` you can `nest` back in, snapshot, or transfer. The temporal-abstraction
    /// building block for the options loop: discover a subgoal, encapsulate the "reach it" pathway, plan over the
    /// macro-actions with `relax_values`. The source organism is unchanged.
    fn encapsulate(&self, cells: Vec<usize>) -> Morph {
        Morph {
            field: self.field.encapsulate(&cells),
        }
    }

    /// NEST — plant a banked `sub` skill as a SINGLE cell whose value is that sub-morph's settled output, fired in
    /// the active cascade (recursion in one node; a nested cell memoises by input). `reads` feed the sub's inputs,
    /// `write` receives its output, `budget` bounds its inner settle. Returns the new unit id. This is the ONE
    /// composition operator — reuse IS nesting; a skill is a Morph. `sub` is cloned, so the passed `Morph` stays
    /// usable (bank once, nest into many).
    fn nest(&mut self, sub: &Morph, reads: Vec<usize>, write: usize, budget: usize) -> usize {
        self.field.nest(sub.field.clone(), &reads, write, budget)
    }

    /// NEST_MULTI — plant a banked `sub` skill exposing its FULL VECTOR output. The sub's sink cells map
    /// positionally to `writes`: `writes[0]` is the primary output (the nested unit's cell), each further
    /// `writes[i]` receives the sub's i-th sink on the same settle — so a nested cell is a real vector sub-morph,
    /// not a scalar feature. A single-element `writes` is exactly `nest`. Returns the new unit id.
    fn nest_multi(
        &mut self,
        sub: &Morph,
        reads: Vec<usize>,
        writes: Vec<usize>,
        budget: usize,
    ) -> usize {
        self.field
            .nest_multi(sub.field.clone(), &reads, &writes, budget)
    }

    /// BANK_SKILL — autonomous nested growth: encapsulate the transitive backbone of `output_cell` and nest it as
    /// a FROZEN reusable node (a second storey the organism grows itself), returning the banked cell. The caller
    /// names WHICH cell is the skill boundary (reliance says which subgraph earned it); banking a single-output
    /// net's final cell banks the whole net, so name an INTERMEDIATE cell to bank a sub-skill. Interference-proof:
    /// a later task grows a fresh readout over the banked node instead of overwriting it (proof `nested_transfer`).
    fn bank_skill(&mut self, output_cell: usize, budget: usize) -> usize {
        self.field.bank_skill(output_cell, budget)
    }

    /// The cumulative count of nested sub-settles — a cache-MISS meter (a recurring discrete state is served from
    /// the nested cell's memo, not re-settled), and a free behaviour-NOVELTY signal for the discovery frontier.
    fn nested_settles(&self) -> u64 {
        self.field.nested_settles()
    }

    /// THINK — internal, self-sustained dynamics: the organism free-runs its own recurrent MODEL when not
    /// receiving the world, feeding its predicted next-state (`pred_cells`) back in as its state (`state_cells`)
    /// and re-settling — a closed-loop trajectory that SIMULATES the consequences of its fragmentary experience.
    /// Set the start state with a `stimulate` first, then `think(steps)` rolls that many transitions forward and
    /// returns the state after them (the simulated consequence), which you can consolidate with `grow`/`learn`.
    /// `noise ∈ [0,1]` adds a little spontaneous drift so the dynamics do not freeze (0 = deterministic). No
    /// external input is consumed. The generative complement of the autonomous heartbeat (memory maintenance).
    fn think(
        &mut self,
        state_cells: Vec<usize>,
        pred_cells: Vec<usize>,
        steps: usize,
        noise: f32,
        budget: usize,
    ) -> Vec<f32> {
        self.field
            .think(&state_cells, &pred_cells, steps, noise, budget)
    }

    /// DREAM — the autonomous discovery loop as a substrate faculty. Left alone with no external input, the
    /// organism lives a wake-process-sleep cycle: for `bouts` rounds it THINKS each `seed` state forward and
    /// CONSOLIDATES the simulated consequence into `longrange` (discovered structure → direct knowledge), then
    /// SLEEPS — the maintenance heartbeat consolidates the relied-upon memory and RESTS. Returns
    /// `(ticks, reactivations, consolidated, culled, rested, hunger)` of the sleep. Domain-blind; a modality can
    /// bank a skill vocabulary over the same dream with the perception crystalliser.
    #[allow(clippy::too_many_arguments)]
    fn dream(
        &mut self,
        state_cells: Vec<usize>,
        pred_cells: Vec<usize>,
        longrange: Vec<usize>,
        seeds: Vec<Vec<f32>>,
        think_steps: usize,
        lr: f32,
        bouts: usize,
        budget: usize,
    ) -> (usize, usize, usize, usize, bool, f32) {
        let hb = self.field.dream(
            &state_cells,
            &pred_cells,
            &longrange,
            &seeds,
            think_steps,
            lr,
            bouts,
            budget,
        );
        (
            hb.ticks,
            hb.reactivations,
            hb.consolidated,
            hb.culled,
            hb.slept,
            hb.hunger,
        )
    }

    /// SATIATION ∈ [0,1] — the organism's energy account `metabolism / energy_cap` (fed ⇒ 1, starving ⇒ 0).
    fn satiation(&self) -> f32 {
        self.field.satiation()
    }

    /// WHETHER THE ORGANISM IS RESTED ENOUGH TO DREAM — `satiation >= config.dream_satiation`. Set
    /// `dream_satiation` (via `configure`) to make dreaming a fed-luxury: a hungry organism conserves instead.
    fn should_dream(&self) -> bool {
        self.field.should_dream()
    }

    // ── planning (#331/#334): a value field over an experienced/predicted adjacency ─────────────────────────────

    /// PREFER — clamp `cell`'s cost-to-go at `value` (a goal is `prefer(cell, 0.0)`), the attractor `relax_values`
    /// descends toward.
    fn prefer(&mut self, cell: usize, value: f32) {
        self.field.prefer(cell, value);
    }

    /// ORIENT (#155) — the organism GENERATES ITS OWN goal: among `candidates` (addressable state cells) it picks
    /// the one maximising `drive_of + epistemic_weight·learning_progress` (extrinsic reward-memory + reducible
    /// surprise — dodges the dark-room) by the grown winner-take-all, and CLAMPS it as the goal. Returns the chosen
    /// cell (`None` if `candidates` is empty). This is the self-set goal `goal_reward` / `epistemic_weight` need.
    fn orient(&mut self, candidates: Vec<usize>) -> Option<usize> {
        self.field.orient(&candidates)
    }

    /// ORIENT_MANY — generate the top-`k` self-set goals at once and clamp them as SIMULTANEOUS subgoals (#155);
    /// `relax_values` (multi-source) then descends a cost-to-go field seeded by all of them. Same grown score as
    /// `orient`, the grown WTA applied `k` times without replacement. Returns the goal cells in selection order.
    fn orient_many(&mut self, candidates: Vec<usize>, k: usize) -> Vec<usize> {
        self.field.orient_many(&candidates, k)
    }

    /// The cumulative number of SELF-SET goals reached (a taken option landed on a clamped goal while
    /// `goal_reward` is on). `0` when the goal-completion reward is off.
    fn goals_reached(&self) -> u64 {
        self.field.goals_reached()
    }

    /// REINFORCE — accumulate a graded action-drive on `cell` (`+` commit / `−` aversion). Seeds `orient`'s
    /// extrinsic term and is the channel the goal-completion reward folds through.
    fn reinforce(&mut self, cell: usize, amount: f32) {
        self.field.reinforce(cell, amount);
    }

    /// The accumulated action-drive of a cell (`0.0` if never reinforced).
    fn drive_of(&self, cell: usize) -> f32 {
        self.field.drive_of(cell)
    }

    /// RELAX_VALUES — settle the value field toward its fixed point over `adjacency` (`adjacency[c]` = cells
    /// reachable from `c`), `max_sweeps` internal sweeps (`0` = to convergence). The DELIBERATION primitive: more
    /// sweeps = more free internal planning ⇒ fewer external actions (#331/#334). Returns the correction footprint.
    fn relax_values(&mut self, adjacency: Vec<Vec<usize>>, cost: f32, max_sweeps: usize) -> usize {
        self.field.relax_values(&adjacency, cost, max_sweeps)
    }

    /// The graded cost-to-go value of a cell (`f32::MAX` = unknown/unreachable). Descend it to reach a goal, or
    /// read it as the rung-2 confidence signal (best − second-best neighbour value).
    fn value_of(&self, cell: usize) -> f32 {
        self.field.value_of(cell)
    }

    /// G_SPREAD — the decision-degeneracy diagnostic. The spread (std-dev) of the active rule's decision score
    /// (the valence or the free-energy `G`) across the options offered to the LAST `act()`.
    /// Near-zero ⇒ the decision was DEGENERATE (every option tied ⇒ a tie-break) — the silent-failure signature of
    /// a flat reward, a collapsed value head, or a state key so coarse every option maps to one cell. Assert
    /// `g_spread() > eps` in an agent smoke test to catch "your options are tied" instead of chasing it for weeks.
    /// `NaN` until the first act. Observation-only.
    fn g_spread(&self) -> f32 {
        self.field.g_spread()
    }

    /// Reset the value field (clear every clamp; all values unknown) — start a fresh goal.
    fn clear_values(&mut self) {
        self.field.clear_values();
    }

    // ── exploration + attribution (#333 reframe / #335): the COVERAGE levers the honest-negative points to ──────

    /// SELECT_EXPLORE — value-of-information choice: among `options` (each a list of grown value-estimator cells),
    /// pick the one maximising `mean + beta·spread` — value PLUS ensemble DISAGREEMENT (epistemic uncertainty). The
    /// VoI exploration lever (`substrate_voi`); the #333 honest-negative says DIRECT exploration with THIS (+ the
    /// metric-map frontier below), NOT world-model σ². Returns the chosen option index. Ties among equal-top
    /// options break UNIFORMLY via the organism's deterministic act PRNG (so it takes `&mut self`); a single
    /// strict winner takes no draw. An empty `options` raises `ValueError` (there is nothing to choose).
    fn select_explore(&mut self, options: Vec<Vec<usize>>, beta: f32) -> PyResult<usize> {
        if options.is_empty() {
            return Err(PyValueError::new_err(
                "select_explore needs at least one option",
            ));
        }
        let refs: Vec<&[usize]> = options.iter().map(|o| o.as_slice()).collect();
        Ok(self.field.select_explore(&refs, beta))
    }

    /// RELIANCE_OF — how much the grown readouts rely on cell `c` (Σ readers' |weight|). Over a reward predictor's
    /// feature cells, argmax `reliance_of` = the inferred GOAL predicate (grown reward attribution, #335).
    fn reliance_of(&self, cell: usize) -> f32 {
        self.field.reliance_of(cell)
    }

    // ── the grown concept-map (metric-map frontier — the coverage lever, #333 reframe) ──────────────────────────

    /// The grown concept-map POSITION of a cell (its coordinates on the organism's own cognitive map). A cell
    /// outside the membrane raises `ValueError` (it has no position to report).
    fn position(&self, cell: usize) -> PyResult<Vec<f32>> {
        check_cell(&self.field, cell, "cell")?;
        Ok(self.field.position(cell))
    }

    /// The concept-map DISTANCE between two cells over the first `dims` axes (relatedness; farthest-point over this
    /// is the metric-map frontier — the #333/`substrate_skillmap` coverage lever the agent should drive exploration
    /// with). A cell outside the membrane raises `ValueError`.
    fn distance(&self, a: usize, b: usize, dims: usize) -> PyResult<f32> {
        check_cell(&self.field, a, "cell")?;
        check_cell(&self.field, b, "cell")?;
        Ok(self.field.distance(a, b, dims))
    }

    /// SEED_COORD — place `cell` on the concept-map from a percept vector (a repeating state → a repeating key), the
    /// percept→map seam the frontier is built over. A cell outside the membrane raises `ValueError` (the core would
    /// silently skip it).
    fn seed_coord(&mut self, cell: usize, percept: Floats) -> PyResult<()> {
        check_cell(&self.field, cell, "cell")?;
        self.field.seed_coord(cell, &percept);
        Ok(())
    }

    /// METRIC_STEP — one contrastive step that sharpens the concept-map metric: pull `positive` cells together, push
    /// `n_neg` random negatives apart, at `rate`. Anneal `rate` to unfold a faithful metric (analogy / frontier).
    fn metric_step(&mut self, positive: Vec<usize>, rate: f32, n_neg: usize) {
        self.field.metric_step(&positive, rate, n_neg);
    }

    /// CO_ACTIVATE — cells that fire together drift CLOSER on the concept-map (Hebbian map shaping): pull the
    /// positions of `cells` toward their centroid at `rate`. Called each time a set of cells co-activates while the
    /// organism works, so working-together cells cluster — the geometry a decomposition is then read off.
    fn co_activate(&mut self, cells: Vec<usize>, rate: f32) {
        self.field.co_activate(&cells, rate);
    }

    /// SPREAD_POSITIONS — per-axis standardize the concept-map so it fills the space (prevents collapse), the
    /// counter-force to `co_activate`'s attraction. Call it before `cluster` to get well-separated groups.
    fn spread_positions(&mut self) {
        self.field.spread_positions();
    }

    /// CLUSTER — farthest-point clustering over the grown concept-map: partition `ids` into `k` groups by their
    /// positions (seed 0 = ids[0], then repeatedly the id whose minimum distance to the current seeds is maximal;
    /// each id joins its nearest seed; empty groups dropped). This is the boundary DETECTOR of the discovery loop:
    /// cluster a morph's OWN grown cells → each group is a candidate SUB-MORPH (`substrate_decompose`), which you
    /// then `encapsulate` + `nest` (or `bank_skill` its output). Deterministic. Returns the groups as lists of ids.
    fn cluster(&self, ids: Vec<usize>, k: usize) -> Vec<Vec<usize>> {
        self.field.cluster(&ids, k)
    }

    /// GROWN_CELLS — the cells written by a grown unit (a grown feature/skill/output), in cell order. The set to
    /// feed `cluster` in the discovery loop: `grow` the model, then `cluster(grown_cells(), k)` to read candidate
    /// sub-morph boundaries off the concept-map, then `encapsulate`/`bank_skill` each. (A raw input cell has no
    /// writer and is excluded.)
    fn grown_cells(&self) -> Vec<usize> {
        (0..self.field.len())
            .filter(|&c| self.field.has_writer(c))
            .collect()
    }

    // ── SYNERGY grouping — the interaction-not-co-occurrence shapers for the discovery loop ──────────────────────

    /// GROUP_BY_SURPRISE — the causal analog of `co_activate`: pull each pair in `cells` together in proportion to
    /// `min(SR_i, SR_j)` over the per-cell surprise-reduction (end-to-end ablation importance), so a pair coheres
    /// only when BOTH members are causally load-bearing (a spurious common-cause pair does not group). Gated on the
    /// `surprise_currency` config (set it via `configure`) AND a warmed target predictor — a NO-OP (byte-identical)
    /// when off. Use it in place of `co_activate` when the map must reflect causal grouping, not mere co-firing.
    fn group_by_surprise(&mut self, cells: Vec<usize>, rate: f32) {
        self.field.group_by_surprise(&cells, rate);
    }

    /// SURPRISE_REDUCTION_OF — the per-cell end-to-end ablation importance `(∂output/∂c)²·Var(c)` relative to the
    /// currently-warmed output (the raw signal `group_by_surprise` gates on). `0.0` unless `surprise_currency` is on
    /// and a predictor has been warmed. Read it to rank which grown cells are causally load-bearing for a target.
    fn surprise_reduction_of(&self, cell: usize) -> f32 {
        self.field.surprise_reduction_of(cell)
    }

    /// ACCRUE_CO_CREDIT — the SECOND-ORDER sibling of the per-unit delta rule, and the shaper that recovers SYNERGY
    /// modules a per-cell signal cannot. For every pair `(i, j)` in `cells` it accrues an EMA of `err · x_i · x_j`
    /// (the gradient a virtual product weight would get): a pair whose PRODUCT explains the residual `err`
    /// accumulates large `|co_credit|`, an additive pair averages to ~0. Call each event during training with the
    /// readout's residual (a SLOW `decay`, e.g. 0.9999, averages the signal cleanly), then `co_credit_step` to shape
    /// the map. Unlike `group_by_surprise`'s `min(SR)` (which cannot separate interacting cells when every cell is
    /// individually load-bearing), this is a true pairwise-interaction signal. Transient; byte-identical unless
    /// called (`substrate_co_credit`).
    fn accrue_co_credit(&mut self, cells: Vec<usize>, err: f32, decay: f32) {
        self.field.accrue_co_credit(&cells, err, decay);
    }

    /// CO_CREDIT_STEP — shape the concept-map by accrued synergy: a SIGNED per-pair force around the mean
    /// `accrue_co_credit` magnitude (above-mean pairs attract, below-mean repel), carving a metric where distance
    /// means "how much these cells INTERACT." `cluster` over it then recovers synergy modules `co_activate` /
    /// `group_by_surprise` miss. No-op when nothing was accrued (byte-identical off).
    fn co_credit_step(&mut self, rate: f32) {
        self.field.co_credit_step(rate);
    }

    /// CO_CREDIT_OF — the accrued second-order co-credit of a cell pair (order-insensitive; `0.0` if unseen).
    fn co_credit_of(&self, a: usize, b: usize) -> f32 {
        self.field.co_credit_of(a, b)
    }

    /// One local delta-learning event toward `target` at `output_cell`. Returns the error before the update.
    /// `idx`/`val` are lists or 1-D numpy arrays of equal length; a length mismatch, or an input or output cell
    /// outside the membrane, raises `ValueError` (nothing could be learned there).
    fn learn_step(
        &mut self,
        idx: Indices,
        val: Floats,
        output_cell: usize,
        target: f32,
        lr: f32,
        budget: usize,
    ) -> PyResult<f32> {
        let pairs = inputs(&idx, &val, self.field.len())?;
        check_cell(&self.field, output_cell, "output cell")?;
        Ok(self
            .field
            .learn_step(&pairs, output_cell, target, lr, budget))
    }

    /// GROW — the one growth verb: grow the transducer toward `target` at `target_cell` (surprise-birth of a
    /// writer, then hidden depth up to `config.max_hidden`), and tune it. Returns the error before the step.
    /// `idx`/`val` are lists or 1-D numpy arrays of equal length (`ValueError` otherwise). A `target_cell` past the
    /// membrane grows it to fit (a fresh output cell), and so does an input cell while `target_cell` has no writer
    /// yet (the writer born to read it claims it — `Morph(0).grow([0], [x], 1, ..)` works); once the target has a
    /// writer, an input cell past the membrane raises `ValueError` (the core would drop it from the step).
    fn grow(
        &mut self,
        idx: Indices,
        val: Floats,
        target_cell: usize,
        target: f32,
        lr: f32,
        budget: usize,
    ) -> PyResult<f32> {
        let pairs = inputs(&idx, &val, grow_input_limit(&self.field, target_cell))?;
        let (err, _born) = self.field.grow(&pairs, target_cell, target, lr, budget);
        Ok(err)
    }

    /// Target-free predictive coding: correct the standing prediction toward `values` on the `obs` band, and
    /// return the per-cell prediction error (the dense intrinsic signal). `values` (a list or 1-D numpy array)
    /// must hold exactly `dim` entries (`ValueError` otherwise — it used to be silently zero-padded/truncated).
    fn observe(
        &mut self,
        obs: usize,
        pred: usize,
        dim: usize,
        values: Floats,
        lr: f32,
        budget: usize,
    ) -> PyResult<Vec<f32>> {
        if values.len() != dim {
            return Err(PyValueError::new_err(format!(
                "observe needs exactly dim={dim} values, got {}",
                values.len()
            )));
        }
        Ok(self.field.observe(obs, pred, dim, &values, lr, budget))
    }

    /// One step of the native POLICY loop (#150). `options` are option cells; encode `(state, action)` into each
    /// (e.g. `state_key * n_actions + action`) so selection is CONTEXTUAL. `last` = `(chosen_option, reward)` from
    /// the previous step (`None` on the first) — its reward is credited to that option. Returns the chosen option
    /// cell; an empty `options` with a `last` ends the episode and returns `None`, and with no `last` raises
    /// `ValueError`. The rule is the config's
    /// `act_rule`: `"Valence"` (the DEFAULT since 1.6 — argmax `−value + μ + σ²` over each option's reward estimate
    /// μ and exact visit count, σ² = (σ0·(1−lr)^visits)² shrinking with every visit, exact ties broken by the seeded
    /// act PRNG; for rewards of scale c set `free_energy_prior_std` ≈ √c), `"FreeEnergy"` (argmin
    /// `G = value + (μ−s*)² − σ²`, a set-point matcher; ties → first). The pre-1.6 `"ActScore"` rule is removed: a
    /// config that names it loads as `"Valence"`, and its four keys (`openmindedness`, `hypothesis_loop`,
    /// `drive_decay`, `metabolic_actor`) are unknown to `configure`. The deprecated bools
    /// `free_energy_actor` / `valence_actor` still select their rule (free energy wins). `act_discount` (default
    /// 0 = immediate reward) folds `r + act_discount·max V(options)` on the valence (`V = μ + σ²`) and
    /// free-energy (`V = μ`) rules — set 0.9–0.99 for a multi-step task. END AN EPISODE with `act([], last)`: it folds the final reward
    /// with no bootstrap and returns `None`.
    #[pyo3(signature = (options, last=None))]
    fn act(&mut self, options: Vec<usize>, last: Option<(usize, f32)>) -> PyResult<Option<usize>> {
        if options.is_empty() && last.is_none() {
            return Err(PyValueError::new_err(
                "act needs at least one option cell (or a `last` to end the episode)",
            ));
        }
        match self.field.alive(Event::Act {
            options: &options,
            last,
        }) {
            Outcome::Acted { .. } if options.is_empty() => Ok(None),
            Outcome::Acted { chosen } => Ok(Some(chosen)),
            _ => Err(PyRuntimeError::new_err("act: the organism did not act")),
        }
    }

    /// Snapshot the whole organism to `bytes` — the versioned envelope (magic, format, JSON payload), so it keeps
    /// loading after a release adds fields. Persistence, mycelium-compatible.
    fn snapshot(&self) -> PyBytesOut {
        self.field.snapshot().into()
    }

    /// Load an organism from a snapshot produced by `snapshot()`: `bytes` or `bytearray`, or the legacy
    /// `list[int]` older releases' `snapshot()` returned; pre-envelope JSON snapshots load too. Malformed,
    /// corrupt or newer-format data raises `ValueError`.
    #[staticmethod]
    fn load(bytes: &Bound<'_, PyAny>) -> PyResult<Morph> {
        let field = Field::load(&snapshot_bytes(bytes)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Morph { field })
    }
}

/// One segmented object record for Python: `(colour, size, centroid_x, centroid_y, bbox_w, bbox_h, cells)`.
/// `centroid` is in absolute grid coords (the precise-click point); `cells` are flat `row*width+col` indices.
type PyObject = (usize, usize, f32, f32, usize, usize, Vec<usize>);

fn object_record(o: &openmorph_core::grouping::Object, width: usize) -> PyObject {
    let (cx, cy) = o.centroid(width);
    (
        o.color,
        o.size(),
        cx,
        cy,
        o.bbox_w,
        o.bbox_h,
        o.cells.clone(),
    )
}

/// SEGMENT a flat integer-colour grid into connected components (4-connectivity, `background` excluded) — the
/// authored label-propagation primitive (`openmorph_core::grouping::segment_objects`). Returns one record per
/// object: `(colour, size, centroid_x, centroid_y, bbox_w, bbox_h, cells)`. The centroid is the precise-click
/// target for enclosure/flood-fill/whack games; `cells` gives the exact mask. Domain-blind — the organism grows
/// the meaning; this only exposes the structure. For the GROWN (non-authored) segmenter, see `GrownSegmenter`.
#[pyfunction]
fn segment_objects(grid: Indices, width: usize, height: usize, background: usize) -> Vec<PyObject> {
    openmorph_core::grouping::segment_objects(&grid, width, height, background)
        .iter()
        .map(|o| object_record(o, width))
        .collect()
}

/// REDUCE a variable-length value list to fixed scalar statistics — `(count, sum, min, max, mean, spread)`
/// (`openmorph_core::grouping::aggregate`). `spread` is the dispersion moment (and, over an ensemble of estimators,
/// the value-of-information uncertainty). This is how a variable-cardinality scene meets the fixed-width substrate,
/// and the scalar-field (thermostat) state is just `aggregate(frame).mean`.
#[pyfunction]
fn aggregate(values: Floats) -> (f32, f32, f32, f32, f32, f32) {
    let a = openmorph_core::grouping::aggregate(&values);
    (a.count, a.sum, a.min, a.max, a.mean, a.spread)
}

/// RELATE the objects of a grid pairwise — the raw relational structure (`openmorph_core::grouping::relate` over
/// `segment_objects`). Returns `(a, b, dx, dy, size_ratio, same_colour, adjacent)` per ordered pair, where `a`/`b`
/// index the `segment_objects(grid, …)` order. Exposes structure (adjacency / relative position / size ratio /
/// colour equality), not a task — the organism grows the meaning.
#[pyfunction]
fn relate(
    grid: Indices,
    width: usize,
    height: usize,
    background: usize,
) -> Vec<(usize, usize, f32, f32, f32, bool, bool)> {
    let objs = openmorph_core::grouping::segment_objects(&grid, width, height, background);
    openmorph_core::grouping::relate(&objs, width, height)
        .into_iter()
        .map(|r| (r.a, r.b, r.dx, r.dy, r.size_ratio, r.same_color, r.adjacent))
        .collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Morph>()?;
    m.add_function(wrap_pyfunction!(segment_objects, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(relate, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
