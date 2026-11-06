// SPDX-License-Identifier: MIT-0
//! Tunable META-parameters of the substrate — the knobs Optuna (or any tuner) sweeps.
//!
//! These are NOT authored task weights (those are grown). They are the constants that govern the substrate's
//! own DYNAMICS — the energy economy, the surprise/credit smoothing, the evolution bounds, and the initial
//! values of the heritable traits. Previously each was a magic literal buried in a method; collecting them into
//! one serde-serialisable struct makes the whole substrate configurable from the outside (load a JSON of a
//! trial's parameters), tunable (a flat, bounded search space), and changeable LIVE ([`crate::Field::configure`]
//! / [`crate::Field::config_mut`]). This was the original goal of the retired `openmorph-config` crate; it lives
//! in core now because the constants belong to the substrate they configure.
//!
//! Note on the heritable traits (`init_scale` / `init_lr` / `init_mut_rate`): the substrate ALSO tunes `scale`,
//! `lr` and `mut_rate` from the inside, per-unit, by competition (see `reproduce` / `metabolize_contribution`).
//! The config sets their INITIAL values and the evolution BOUNDS; a tuner can fix the ones that are not (yet)
//! self-tuning and let competition refine the rest. See `docs/config-reference.md` for the full table, ranges,
//! and which knobs are self-tuning.

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::compat::Float;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// The unit's NONLINEARITY — a genome choice (v2). A unit computes `activation(gain·(Σw·x + b))`, and the local
/// gradient rule uses the derivative **expressed in terms of the OUTPUT** `y` (which every gradient site already
/// has as the settled cell value), so swapping the activation is one coordinated `apply`/`deriv` pair, not a
/// scattered edit. `Tanh` is the default (byte-identical to before this existed).
///
/// - `Tanh`     — bounded, smooth; `y=tanh(x)`, `y'=1−y²`. The historical default.
/// - `Softsign` — bounded, **no `exp`** (cheaper + more bit-portable), **gentler saturation** (polynomial tails):
///   `y=x/(1+|x|)`, `y'=(1−|y|)²`. A strong bare-metal candidate; its slower `deriv→0` keeps `D` better-conditioned
///   (less of the E2 saturation collapse) — compare via the `ρ` self-diagnostic.
/// - `HardTanh`  — cheapest (a clamp): `y=clamp(x,−1,1)`, `y'=1 for |y|<1 else 0`.
/// - `Identity` — **unbounded**, `y=x`, `y'=1`: a bare affine unit (a `Gemm` with no activation, a logit
///   layer). The grown organism never picks it — a bounded cell is what keeps the coupled population stable —
///   but a TRANSCRIBED foreign network (an imported ONNX graph) needs it to be the same function it was.
/// - `Relu` — **unbounded**, `y=max(0,x)`, `y'=1 for y>0 else 0`: the near-universal DNN nonlinearity, here for
///   the same transcription reason.
/// - `Exp` / `Recip` / `Rsqrt` — the TRANSCENDENTAL trio a transformer needs, and nothing else: `softmax` is
///   `Exp` over the scores, a linear sum, then `Recip` of that sum multiplied back in (via [`crate::Unit`]'s
///   forward factor); `LayerNorm`/`RMSNorm` is a linear mean, a squared deviation (the factor again), and
///   `Rsqrt` of the variance. With them the whole transformer is expressible — no new unit kind, only laws.
///
/// Every variant from `Identity` down is **unbounded**, and `Recip`/`Rsqrt` are singular at 0 (guarded to a
/// finite value rather than `inf`, so a settle can never poison the medium with `NaN`). They remove the
/// substrate's own amplitude bound, so a recurrent field of them can diverge — check `ρ`
/// ([`crate::Field::spectral_radius`]) before growing on one. A GROWN organism should never pick them; they
/// exist so a transcription can be exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Activation {
    /// `tanh` — the historical default (bounded, smooth, `exp`-based).
    #[default]
    Tanh,
    /// `x/(1+|x|)` — bounded, `exp`-free, gentler saturation.
    Softsign,
    /// `clamp(x,−1,1)` — bounded, cheapest, non-smooth.
    HardTanh,
    /// `x` — unbounded affine (an imported `Gemm`/logit layer). See the type docs for the stability caveat.
    Identity,
    /// `max(0,x)` — unbounded rectifier (an imported `Relu`). See the type docs for the stability caveat.
    Relu,
    /// `eˣ` — the softmax numerator. Saturating-guarded so a large score yields `f32::MAX` rather than `inf`.
    Exp,
    /// `1/x` — division by a computed value (the softmax denominator). Guarded at 0.
    Recip,
    /// `1/√x` — the normalisation scale of `LayerNorm`/`RMSNorm` in one unit. Guarded at/below 0.
    Rsqrt,
}

/// Below this magnitude a reciprocal is treated as division by `SINGULAR` — a settle must never put `inf`
/// or `NaN` on the medium, because every downstream cell would inherit it for the rest of the organism's life.
const SINGULAR: f32 = 1e-12;

impl Activation {
    /// Forward: `y = f(x)`.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Softsign => x / (1.0 + x.abs()),
            Activation::HardTanh => x.clamp(-1.0, 1.0),
            Activation::Identity => x,
            Activation::Relu => x.max(0.0),
            // `exp` overflows to `inf` past ~88 in f32; clamp the ARGUMENT so the result stays finite.
            Activation::Exp => x.min(88.0).exp(),
            Activation::Recip => {
                1.0 / if x.abs() < SINGULAR {
                    SINGULAR.copysign(x)
                } else {
                    x
                }
            }
            Activation::Rsqrt => 1.0 / x.max(SINGULAR).sqrt(),
        }
    }
    /// Derivative `dy/dx` **as a function of the output `y`** (so gradient sites use the settled cell value).
    #[inline]
    pub fn deriv(self, y: f32) -> f32 {
        match self {
            Activation::Tanh => 1.0 - y * y,
            Activation::Softsign => {
                let a = 1.0 - y.abs();
                a * a
            }
            Activation::HardTanh => {
                if y.abs() < 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Identity => 1.0,
            // Each of these has a derivative expressible in the OUTPUT, like every other variant:
            //   y = eˣ      ⇒ dy/dx = y
            //   y = 1/x     ⇒ dy/dx = −1/x² = −y²
            //   y = 1/√x    ⇒ dy/dx = −½·x^(−3/2) = −½·y³
            Activation::Exp => y,
            Activation::Recip => -y * y,
            Activation::Rsqrt => -0.5 * y * y * y,
            // `y > 0 ⇒ x > 0`, so the rectifier's slope reads off the output like every other variant.
            Activation::Relu => {
                if y > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// The RULE [`crate::Field::alive`]`(Event::Act)` selects an option with — one actor path, two scores over the
/// same inputs (the offered option cells, the planner's `value_of`, the folded `last` reward). Read it through
/// [`Config::effective_act_rule`], which also honours the deprecated `free_energy_actor` / `valence_actor` bools.
///
/// - `Valence` (**the default since 1.6**) — ARGMAX `A(o) = −value_of(o) + μ_o + σ_o²` over a per-option record of
///   the reward estimate `μ_o` (folded `μ ← μ + lr·(target − μ)` from exactly 0) and the EXACT visit count `n_o`:
///   a reward MAXIMISER whose exploration bonus `σ_o² = (σ0·(1−lr)^n_o)²` shrinks with every visit (never-visited
///   ⇒ the prior `σ0²`; a zero-reward option keeps `μ = 0` exactly, so a less-visited option never scores below a
///   more-visited one). Exact ties among the best scores are broken UNIFORMLY by the seeded act PRNG
///   (deterministic, replayable). The fold target is `r + act_discount·max_{o'} (μ_{o'} + σ_{o'}²)`: the
///   successor's bonus is bootstrapped too, so optimism about an unexplored state flows back along the path to it
///   (an `Act` with empty options ends an episode: no bootstrap). The bonus is in reward² units: for rewards of
///   scale `c`, set `free_energy_prior_std ≈ √c` (the scores then scale uniformly). The unit-consistent
///   alternative `μ + S·σ` (`S` a running `|r|` scale) was measured and not adopted: it scores higher on the
///   13-world bench mean without a discount (0.70 vs 0.66; 0.68 vs 0.82 with one) but its `S` decays through
///   zero-reward droughts and takes every option's bonus with it (`examples/actor_bench.rs -- all 40 --variants`
///   in `openmorph-perception`).
/// - `FreeEnergy` — ARGMIN `G(o) = value_of(o) + (μ_o − s*)² − σ_o²` over a per-option value ENSEMBLE (24
///   members folded like `μ`; their mean and variance): a SET-POINT matcher (it prefers the reward nearest
///   `free_energy_sstar`, not the largest). Ties go to the first option. MorphBot's `Player` runs it. Same
///   `act_discount` fold.
///
/// The pre-1.6 default `ActScore` (ARGMAX `−value + drive + openmindedness·1[drive == 0]`, ties → last; normalised
/// score 0.14 on the 13-world bench vs Valence's 0.66, random = 0 — `examples/actor_bench.rs` in
/// `openmorph-perception`) is REMOVED, with its four levers (`hypothesis_loop`, `openmindedness`, `drive_decay`,
/// `metabolic_actor`). A config or snapshot that names it loads as `Valence` (a serde alias), and the four keys are
/// ignored on load like every retired key.
///
/// serde-default `Valence`: an old config/snapshot JSON without the key loads the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ActRule {
    /// Reward-maximising grown valence (the default). The removed pre-1.6 `"ActScore"` loads as this.
    #[default]
    #[serde(alias = "ActScore")]
    Valence,
    /// Set-point-matching expected free energy `G`.
    FreeEnergy,
}

/// How the living-pool reliance economy turns a grown feature's RELIANCE (`Σ|reader weight|`, see
/// [`crate::Field::reliance_of`]) into income — [`crate::Field::metabolize_reliance`], which
/// [`crate::Field::alive`]`(Event::Perceive)` runs on every event. The survival breakeven is always
/// `pool_upkeep / pool_income` (0.125 at the defaults); the variant decides what it is measured against. Read it
/// through [`Config::effective_reliance_norm`], which also honours the deprecated `relative_reliance` bool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelianceNorm {
    /// Income `∝` the ABSOLUTE reliance: a feature survives while the readout leans on it with `Σ|w| ≥ 0.125`. The
    /// pre-1.6 default ([`Config::legacy`]). Culls a LOW-AMPLITUDE working basis indiscriminately (solution weights
    /// ~0.04 all sit below the constant), and a newborn (readout weight 0) must reach `|w| ≈ 0.125` within about 12
    /// events or starve — measured as a birth/death TREADMILL of 0.6–0.97 births and deaths per step that holds the
    /// pool at ~15 units.
    Absolute,
    /// Income `∝` reliance / the pool MEAN reliance (the default since 1.6): a feature survives while it carries at
    /// least 12.5 % of what the AVERAGE feature of its pool carries. AMPLITUDE-INVARIANT (scale every weight by `k`
    /// and nothing changes) with NO cap on how many equally useful features survive (an equal pool earns
    /// `pool_income` each, whatever its size).
    Mean,
    /// Income `∝` reliance / the pool TOTAL — the share each feature carries (the deprecated `relative_reliance`).
    /// Amplitude-invariant, but a share of `1/n` pays `pool_income/n`, so at most `pool_income/pool_upkeep = 8`
    /// equally useful features can break even: with 12 or more ALL of them are culled. Kept only so a
    /// `relative_reliance: true` config replays exactly.
    Share,
}

fn default_adapt_step_cap() -> f32 {
    0.2
}
fn default_bind_tol() -> f32 {
    1e-3
}
fn default_pool_income() -> f32 {
    2.0
}
fn default_pool_upkeep() -> f32 {
    0.25
}
fn default_pool_death() -> f32 {
    0.1
}
fn default_pool_grace() -> u64 {
    200
}
fn default_true() -> bool {
    true
}
fn default_metabolic_cost() -> f32 {
    0.5
}
fn default_dwell_rate() -> f32 {
    0.5
}
fn default_dwell_max() -> f32 {
    32.0
}
fn default_nest_credit_lr() -> f32 {
    0.1
}
fn default_free_energy_sstar() -> f32 {
    0.9
}
fn default_free_energy_prior_std() -> f32 {
    1.0
}
fn default_free_energy_ensemble() -> usize {
    24
}
fn default_free_energy_lr() -> f32 {
    0.15
}
fn default_epistemic_weight() -> f32 {
    0.5
}
fn default_plasticity_anneal() -> f32 {
    0.0
}
fn default_plasticity_reset_ratio() -> f32 {
    1.6
}
fn default_product_gate() -> f32 {
    0.02
}
fn default_goal_reward() -> f32 {
    0.0
}
fn default_fingerprint_probes() -> usize {
    8
}
fn default_spectrum_bins() -> usize {
    16
}
fn default_embed_gain_weight() -> f32 {
    2.0
}
fn default_momentum_min() -> f32 {
    0.0
}
fn default_momentum_max() -> f32 {
    0.99
}
fn default_resonator_radius_max() -> f32 {
    0.999
}
fn default_nlms_eps() -> f32 {
    1.0
}
fn default_rls_prior() -> f32 {
    100.0
}
fn default_saturation_floor() -> f32 {
    1e-4
}
fn default_width_norm_min() -> usize {
    128
}
fn default_pos_dim_init() -> usize {
    2
}
fn default_metric_repel_softening() -> f32 {
    1.0
}
fn default_probe_settle_pad() -> usize {
    32
}
fn default_struct_walk_max_depth() -> usize {
    64
}
fn default_heartbeat_k() -> f32 {
    0.5
}
fn default_heartbeat_theta() -> f32 {
    1.0
}

/// **The RULE GENOME** — the plug-and-play name for [`Config`]. The v2 finding (GATE-2) was that the substrate is
/// already near-minimal, so "making it configurable" is NOT a new type: the genome *is* the tunable `Config`. It
/// carries the swappable LAWS — the [`Activation`] nonlinearity (+ its paired derivative), the capability toggles
/// (`lr_fanin_norm` = lr∝1/W, `gain_fanin_norm` = g/√N), and every coefficient — as serde data. Plug it live with
/// [`crate::Field::configure`], read it with [`crate::Field::config`]; it round-trips through snapshots. A tuner
/// (Optuna) sweeps the f32 fields ([`Config::schema`]); a UI toggles the categorical ones (bools + `activation`).
pub type RuleGenome = Config;

// `#[serde(default)]` on the container: a partial config JSON loads, every absent knob taking its `Config::default()`
// value (a field's own `default = "…"` function, where present, still applies first). An ORGANISM snapshot is read
// differently (`snapshot_config`): there an absent 1.6 lever means the snapshot predates it, so it takes its
// `Config::legacy()` value.
/// The substrate's tunable constants, the rule genome. `Config::default()` is the measured 1.6 genome;
/// [`Config::legacy`] restores the pre-1.6 values of the levers 1.6 flipped. Every field is a scalar with a tuning
/// range documented in `docs/core/config-reference.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Energy a newborn unit starts with — the development "yolk" that lets it earn before it can starve.
    /// Higher ⇒ longer grace before the economy judges a unit. (was `BIRTH_ENERGY`, 3.0)
    pub birth_energy: f32,
    /// Upper bound on a unit's energy account, so a thriving unit stays bounded. (was `ENERGY_CAP`, 10.0)
    pub energy_cap: f32,
    /// Surprise EMA retention: `surprise ← surprise_decay·surprise + (1−surprise_decay)·|error|`. Higher ⇒
    /// smoother, slower-to-fire surprise (fewer, more-confident births). (was the `0.9` in the surprise EMA)
    pub surprise_decay: f32,
    /// Contribution-credit EMA retention: `credit ← credit_decay·credit + (1−credit_decay)·gradient`. Higher ⇒
    /// a longer memory of a feature's usefulness (steadier fitness for selection). (was the `0.99` credit EMA)
    pub credit_decay: f32,
    /// SURPRISE-DRIVEN PLASTICITY (opt-in, default 0 = OFF/byte-identical): the per-`learn_step` anneal rate of a
    /// learned cell's plasticity GAIN. The delta step is scaled by a gain that DECAYS by this fraction each event —
    /// so a readout SETTLES on a stationary (even NOISY) target instead of being shaken loose by a constant rate —
    /// and RESETS to 1.0 the moment the fast error EMA breaks above `plasticity_reset_ratio · slow` (the target
    /// CHANGED). A CHANGE-detector, not a level-scaler: the level of error carries no signal, only a change does.
    /// 0 ⇒ the gain is a constant 1.0 and the controller state is never allocated — byte-identical.
    #[serde(default = "default_plasticity_anneal")]
    pub plasticity_anneal: f32,
    /// The break-away threshold for [`Self::plasticity_anneal`]: plasticity re-opens when the fast error EMA
    /// exceeds this multiple of the slow one (the target changed relative to the habitat's own noise floor). Read
    /// only when `plasticity_anneal > 0`. serde-default 1.6.
    #[serde(default = "default_plasticity_reset_ratio")]
    pub plasticity_reset_ratio: f32,
    /// GROWN PRODUCT DISCOVERY (opt-in, default false = OFF/byte-identical): when `grow` would birth a hidden
    /// feature, first check whether an input-cell PAIR's product-synergy (`co_credit`, the EMA of `err·xᵢ·xⱼ`)
    /// dominates; if so birth a PRODUCT feature `xⱼ·f(xᵢ)` (a grown `Unit::factor`) on that pair instead of a
    /// random projection — a low-order INTERACTION ("xᵢ matters only when xⱼ is active") that no random-projection
    /// basis of an affine unit can represent, discovered rather than authored. Off ⇒ `grow` births only random
    /// features exactly as before and no synergy is accrued — byte-identical.
    ///
    /// USAGE (validated the hard way): point `grow` — and therefore this — at a SUPERVISED, STATIONARY target
    /// (e.g. predicting the next value of an observed cell), NEVER at a bootstrapped TD value head. Growing
    /// features from a moving/bootstrapped target migrates shared feature directions and corrupts every readout at
    /// once (feature FORMATION must be decoupled from value CREDIT — cf. `grow_backbone`); an RL value head should
    /// learn a readout over a FIXED basis, not grow its own. Nothing in the API prevents the misuse, so it is
    /// called out here.
    #[serde(default)]
    pub grow_products: bool,
    /// PER-TARGET, ACTIVE-PAIR product discovery (opt-in, default false; read only with [`Self::grow_products`]). The
    /// shared synergy map keys a pair by its two input cells alone, so every target `grow` trains writes its own
    /// error into the same entry (evidence of opposite signs from different targets cancels), and it walks every
    /// input pair on every call (`O(n²)` per target, almost all of it on inputs that are zero). With this on, each
    /// TARGET cell keeps its own synergy record, accrued only over the pairs whose inputs are both nonzero (an
    /// untouched pair's EMA decays lazily, exactly as an accrued zero sample would), and a birth picks the target's
    /// own strongest pair. Measured on a world model over one-hot cells: the interaction "this action AND that
    /// neighbour" a random projection cannot express (`docs/research/grown-interactions.md`). Off ⇒ the shared map
    /// as before, byte-identical.
    #[serde(default)]
    pub products_per_target: bool,
    /// The product-synergy threshold for [`Self::grow_products`]: a pair births a PRODUCT feature only when its
    /// accrued `|co_credit|` (EMA of `err·xᵢ·xⱼ`) exceeds this. Separate from `birth_gate` because synergy is a
    /// different scale than surprise (a product of error and two activations, not `|error|`). Small by default so a
    /// genuine interaction is discovered while the economy culls a spurious one. Read only when `grow_products`.
    /// serde-default 0.02.
    #[serde(default = "default_product_gate")]
    pub product_gate: f32,
    /// Autonomous-growth BIRTH GATE: surprise (EMA of |error|) above this at a target cell births a new
    /// hidden feature there. Higher ⇒ births only on persistent, large surprise (fewer features, a linear
    /// task grows ~0); lower ⇒ eager growth. The `grow` default reads it so a caller never hand-sets it.
    pub birth_gate: f32,
    /// Autonomous-growth capacity CEILING: the maximum hidden features `grow` will birth for one target.
    /// A ceiling, not a target — surprise-birth grows only as many as the task needs up to this. The `grow`
    /// default reads it so a caller never hand-sets a width.
    pub max_hidden: usize,
    /// BUDGET-BOUNDED GROWTH: when `true`, [`crate::Field::grow`] (and so `alive(Event::Perceive)`) births a hidden
    /// feature only while a settle of its inputs still fits the `budget` it was called with — every unit reading
    /// the inputs fires once and every grown writer once more after its features (`hidden + 2` per grown target),
    /// and the newborn adds one firing. A feature the budget cannot fire is dead weight that also TRUNCATES every
    /// later settle: the readout's last firing is the one dropped, so the prediction (and the error it learns
    /// from) is not the settled one. With the bound, a caller's `budget` sizes the organism instead of silently
    /// cutting its settles; a `budget` that covers the pool (`≥ max_hidden + 2` per grown target) never binds.
    /// `false` ⇒ births up to [`Self::max_hidden`] whatever the budget (the pre-1.6 behaviour, [`Config::legacy`]).
    ///
    /// ⚠️ DEFAULT BEHAVIOUR CHANGE (1.6): the default is `true`. The mean-relative economy (the 1.6
    /// [`Self::reliance_norm`]) keeps a pool the absolute one churned down to ~15 units, so without the bound a
    /// budget sized for the old pool truncates: on the 12-task learning benchmark (5 seeds) the other flipped levers
    /// alone took reg2d at budget 50 from NMSE 0.27 (old default) to 7.4, and multi10 (10 grown targets) at budget
    /// 100 from 67 % to 28 %. With the bound no settle is truncated at any budget and the flipped default is at
    /// least as good as the old one on every task at budgets 50, 100 and 200 (reg2d 0.011 at budget 50, multi10
    /// 98.9 % at budget 100); at a budget the pool fits in it never binds. Proof `substrate_truncated_settle`.
    /// serde-default `true`: a snapshot written before the lever existed loads with the bound.
    pub grow_within_budget: bool,
    /// Memoization cache capacity for a nested cell (a frozen sub-morph): when a nested cell's input→output
    /// memo reaches this many entries it is cleared wholesale. The memo is serialized with the organism (a
    /// reload must replay the same hits), so this bounds both live memory and a nested cell's share of a
    /// snapshot. (was the buried `MEMO_CAP`, 4096.) Higher ⇒ more cache retention (fewer re-settles on a large
    /// input alphabet), more memory.
    pub memo_cap: usize,
    /// FEATURE-DIRECTION learning rate (#125). When > 0, `learn_step` also nudges each grown feature's OWN
    /// input-direction one hop back through the readout — the SAME delta rule, applied a layer deeper, with the
    /// feature's pseudo-target being the residual backprojected through the readout's weight on it
    /// (`err·deriv·w_readout`). This is feedback-alignment / a one-hop delta: local, online, no
    /// backprop-through-depth. It lets a SMALL grown structure learn its projection directions and match
    /// backprop, instead of relying on random directions + selection (which caps below backprop when directions
    /// matter). `0.0` ⇒ OFF (directions stay random, the economy tunes only scale).
    ///
    /// ⚠️ DEFAULT BEHAVIOUR CHANGE (1.6): the default is `0.3` (was `0.0`; [`Config::legacy`] restores it). Over the
    /// 12-task learning benchmark (`alive`, 5 seeds) the flipped default depends on it: without it parity8 falls
    /// 100 → 64 %, pairs16 NMSE 0.04 → 0.91, drift16 0.08 → 1.0 and two-spirals 94 → 60 %; 0.3 beat 0.1 and 0.05.
    /// It needs a stable readout step ([`Self::stability_cap`]): under the raw step the flipped levers end worse
    /// than the old default.
    pub feature_lr: f32,
    /// FAN-IN-SCALED FEATURE INIT `c`: a hidden feature [`crate::Field::grow`] births draws its input weights from
    /// `U[-1,1]·c/√fan_in` instead of `U[-1,1]` (its bias stays `U[-1,1]`), so the feature's pre-activation has a
    /// fan-in-INDEPENDENT spread instead of one that grows with `√fan_in`. It is a normalization, not a task
    /// constant. Only `grow`'s surprise-birth reads it; [`crate::Field::grow_backbone`] keeps the caller's explicit
    /// `scale`. `0.0` ⇒ OFF: the fan-in-blind `U[-1,1]` draw of [`Config::legacy`].
    ///
    /// ⚠️ DEFAULT BEHAVIOUR CHANGE (1.6): the default is `3.0` (was `0.0`). The fan-in-blind draw saturates wide
    /// features (51 % of activations past |0.95| at 64 inputs) and under-resolves narrow ones (2 inputs): with the
    /// other flipped levers, `c = 3` takes two-spirals 62 → 94 % over `c = 0` (71 % at `c = 2`) and pairs64 NMSE
    /// 1.26 → 0.73. serde-default `3.0`: a snapshot written before the lever existed loads with the new init (set
    /// `0.0` to keep the fan-in-blind draw).
    pub fanin_init_gain: f32,
    /// RELIABILITY-GATED CONSOLIDATION rate (the crystal half). When > 0, `observe_reliability(cell, r)` decays the
    /// plasticity of the writers of `cell` MULTIPLICATIVELY by `1 − consolidation_rate·mastery` — so a skill
    /// CRYSTALLIZES to a HARD freeze (plasticity → ~0, reaching it regardless of the exact mastery plateau) once its
    /// caller-supplied reliability signal reaches a HIGH, STABLE plateau (`mastery`, a relative height·settled meter
    /// — see [`Field::mastery_of`](crate::Field::mastery_of)). A hard freeze ⇒ zero erosion ⇒ a DURABLE hold under any over-training. This is
    /// deliberately ONE-WAY (a self-reopening gate death-spirals); REVERSIBILITY is a POLICY-level act —
    /// `set_plasticity(cells, 1.0)` on a sustained losing streak — matching MorphLab's proven reference. `0.0` (the
    /// default) ⇒ OFF and byte-identical (the meter is never even allocated). The SUBJECT is the caller's: feed
    /// reliable-SUCCESS (a WIN), NOT raw error or cumulative reward. See `substrate_consolidate_reliability`. NOTE:
    /// the CONCEPT is ARC-3-validated (#171, via the FREEZE_STEPS stopgap this replaces); the native primitive is
    /// validated by a synthetic falsifier — ARC-3 wiring is a pending hand-off.
    pub consolidation_rate: f32,
    /// BEHAVIOURAL-KEY probe count (OM-11) — how many canonical probes [`Field::fingerprint`](crate::Field::fingerprint) / [`Field::embed`](crate::Field::embed)
    /// perturb the morph with to read its I/O behaviour. More probes ⇒ finer skill discrimination at higher cost.
    /// A per-library retrieval CONTRACT: changing it re-shapes every stored key, so a library must be re-embedded.
    /// Default `8` = historical, byte-identical. Does NOT affect what the organism learns/grows/acts.
    #[serde(default = "default_fingerprint_probes")]
    pub fingerprint_probes: usize,
    /// BEHAVIOURAL-KEY spectral resolution (OM-11) — the number of DFT magnitude bins in the fingerprint/embed
    /// prefix. Wider ⇒ near-identical behaviours collapse less coarsely, at a larger key. A per-library retrieval
    /// CONTRACT (change ⇒ re-embed). Default `16` = historical, byte-identical.
    #[serde(default = "default_spectrum_bins")]
    pub spectrum_bins: usize,
    /// EMBEDDING GAIN weight (OM-11) — the weight of the single bounded gain coordinate `W·tanh(mean log response-
    /// energy)` in [`Field::embed`](crate::Field::embed). It sets how much a global GAIN rescale (a skill scaled ×k) separates two
    /// otherwise-identical behaviours, relative to the scale-free shape blocks. BOUNDED (∈ ±W) by construction, so
    /// it never lets absolute amplitude dominate the metric (the shipped raw-log-energy did — OM-11 verify). Higher
    /// ⇒ more gain-sensitive. Default `2.0`.
    #[serde(default = "default_embed_gain_weight")]
    pub embed_gain_weight: f32,
    /// Schwefel self-adaptation rate τ for the per-unit mutation σ in `reproduce` (σ mutates by `exp(τ·N)`).
    /// Higher ⇒ faster-drifting mutability. (was `TAU`, 0.4)
    pub self_adapt_tau: f32,
    /// Settle QUIESCENCE threshold: a firing whose output cell moves less than this does NOT spread, so a
    /// stable input settles instead of jittering. Lower ⇒ more precise fixed points but longer cascades (more
    /// compute per settle); higher ⇒ faster, coarser settling. (was the settle-path `EPS`, 1e-4)
    pub settle_eps: f32,
    /// TEMPORAL half-life gradient step cap (#137, `adapt_scales`): the maximum per-event LOG-step a trace's
    /// half-life may move, `h ← h·exp(clamp(step, −cap, +cap))`. Bounds how fast a temporal timescale drifts
    /// in one event — higher ⇒ faster half-life adaptation, less stability. (was the buried `STEP_CAP`, 0.2 ≈
    /// ±22%/event.) serde-default so old snapshots load 0.2.
    #[serde(default = "default_adapt_step_cap")]
    pub adapt_step_cap: f32,
    /// BIND equality tolerance (#122, `bind`): two observed cells count as SHARING a value (a bound relation)
    /// when their values differ by less than this. Tight for clean discrete observations (ARC colours); a
    /// caller on CONTINUOUS / noisy channels raises it so an approximate match still binds the relation. (was
    /// the buried `1e-3`.) serde-default so old snapshots load 1e-3.
    #[serde(default = "default_bind_tol")]
    pub bind_tol: f32,
    /// LIVING-POOL economy rates that [`crate::Field::alive`] runs each `Perceive` event to self-size the grown
    /// basis. `pool_income` is paid to a feature ∝ how much the readout RELIES on it (`reliance_of`); every
    /// feature pays `pool_upkeep`; a feature whose energy falls to `pool_death` is culled, unless younger than
    /// `pool_grace` events (the newborn reserve). The SURVIVAL BREAKEVEN is `pool_upkeep / pool_income` in
    /// reliance units — keep it SMALL so the economy prunes only features the readout IGNORES, not the
    /// used-but-modest features of a DISTRIBUTED basis (a large upkeep culls a nonlinear solution like XOR
    /// mid-formation — alive 45% vs raw grow 95%; the tuned default breakeven 0.125 restores 95%). serde-default
    /// so old snapshots load the tuned rates.
    #[serde(default = "default_pool_income")]
    pub pool_income: f32,
    /// Living-pool upkeep every grown feature pays per `alive` event — see [`Self::pool_income`]. (was the
    /// buried `1.0` in `alive`, retuned to `0.25`: breakeven `0.25/2.0 = 0.125` reliance.)
    #[serde(default = "default_pool_upkeep")]
    pub pool_upkeep: f32,
    /// Living-pool death threshold: a feature whose energy falls to this is culled by `alive` — see
    /// [`Self::pool_income`]. (was the buried `0.5` in `alive`, retuned to `0.1`.)
    #[serde(default = "default_pool_death")]
    pub pool_death: f32,
    /// Living-pool newborn GRACE: a feature younger than this many events is spared from the `alive` cull (the
    /// development reserve) — see [`Self::pool_income`]. (was the buried `200` in `alive`.)
    /// ⚠️ UNITS: "events" is unit FIRINGS (the subjective clock), NOT input/`stimulate` steps — at `F` firings per
    /// input a grace of `200` is only `200 / F` input steps (a fraction of the runway a stream-time reader expects).
    /// See `Field::integrate` for the conversion.
    #[serde(default = "default_pool_grace")]
    pub pool_grace: u64,
    /// **Deprecated alias of `reliance_norm = Share`** (kept so existing configs replay exactly; removed next
    /// release). RELATIVE RELIANCE breakeven — the Defect-1 fix (MorphTrading probe 0001): reliance normalised to a
    /// SHARE of the metabolised pool's TOTAL, so a low-amplitude working basis (solution weights ~0.04) is no longer
    /// culled by the absolute breakeven. When `true` it WINS over [`Self::reliance_norm`] — see
    /// [`Config::effective_reliance_norm`]. Superseded by [`RelianceNorm::Mean`], which is amplitude-invariant too
    /// but has no survivor cap (a share-of-total pool culls ALL of 12 or more equally useful features).
    /// serde-default `false`.
    #[serde(default)]
    pub relative_reliance: bool,
    /// How the living-pool economy scales reliance into income — see [`RelianceNorm`]. A categorical lever, so it
    /// is not in the f32 [`Config::schema`]. The deprecated [`Self::relative_reliance`] overrides it (to
    /// [`RelianceNorm::Share`]) when set; read the rule that runs through [`Config::effective_reliance_norm`].
    ///
    /// ⚠️ DEFAULT BEHAVIOUR CHANGE (1.6): the default is [`RelianceNorm::Mean`] (was `Absolute`). The absolute
    /// economy ran a birth/death treadmill (0.85 births AND deaths per `alive` step, averaged over the 12-task
    /// benchmark) that held the pool at a few newborn features; with the flipped learning levers the mean-relative
    /// one cuts churn 4× (0.21 per step) and matches a no-cull pool's accuracy with ~20 % fewer units, where the
    /// absolute one costs parity8 100 → 90 %, pairs16 NMSE 0.04 → 0.58, drift16 0.08 → 0.75 and two-spirals
    /// 94 → 72 %. The trade: a pool whose features are all EQUALLY ignored earns the mean each, so an
    /// over-provisioned LINEAR task is no longer pruned down to its writer (`substrate_grow`: 7 units kept where the
    /// absolute economy keeps 3). serde-default `Mean`: a snapshot written before the lever existed loads with the
    /// new economy (set `Absolute` to keep the old one).
    pub reliance_norm: RelianceNorm,

    /// Initial per-unit projection `scale`/gain (the activation is `tanh(scale·(Σw·x+b))`). Heritable
    /// thereafter. (was the `1.0` default `scale`)
    pub init_scale: f32,
    /// Initial per-unit learning-rate factor (multiplied into this unit's weight updates). Heritable
    /// thereafter. (was the `1.0` default `lr`)
    pub init_lr: f32,
    /// Initial per-unit mutation strength σ. Itself heritable and self-adapting. (was the `0.3` default
    /// `mut_rate`)
    pub init_mut_rate: f32,
    /// Initial per-unit MOMENTUM / damping μ for the delta rule (2nd-order / heavy-ball: `v←μ·v + Δw; w←w+v`).
    /// `0.0` (the default) = the historical 1st-order delta rule, BYTE-IDENTICAL. `μ∈(0,1)` makes learning a
    /// damped 2nd-order system (nature's stability): μ→1 is under-damped (fast but rings), small μ over-damped.
    /// Heritable thereafter, so the organism GROWS its own critical damping by competition — the 2nd-order
    /// stability of the tuning loop, with no authored constant. Serde-default 0 (old snapshots load 1st-order).
    #[serde(default)]
    pub init_momentum: f32,

    /// Evolution clamp: minimum mutation σ an offspring may inherit. (was the `0.02` in `reproduce`)
    pub sigma_min: f32,
    /// Evolution clamp: maximum mutation σ. (was `1.5`)
    pub sigma_max: f32,
    /// Evolution clamp: minimum offspring `scale`. (was `0.05`)
    pub scale_min: f32,
    /// Evolution clamp: maximum offspring `scale`. (was `5.0`)
    pub scale_max: f32,
    /// Evolution clamp: minimum offspring `lr`. (was `0.05`)
    pub lr_min: f32,
    /// Evolution clamp: maximum offspring `lr`. (was `8.0`)
    pub lr_max: f32,
    /// Evolution clamp: minimum offspring MOMENTUM μ. (was the buried `0.0` in `reproduce`)
    #[serde(default = "default_momentum_min")]
    pub momentum_min: f32,
    /// Evolution clamp: maximum offspring MOMENTUM μ — the ceiling on 2nd-order damping an evolving lineage may
    /// reach (μ≥1 is unstable). Parity with `sigma_max`/`scale_max`/`lr_max`. (was the buried `0.99` in `reproduce`)
    #[serde(default = "default_momentum_max")]
    pub momentum_max: f32,
    /// Maximum grown RESONATOR pole radius (`resonate`) — the ring-duration/sharpness ceiling `< 1` (a radius ≥ 1
    /// never decays). Parity with the other stability clamps. (was the buried `0.999` in `resonate`)
    #[serde(default = "default_resonator_radius_max")]
    pub resonator_radius_max: f32,
    /// Normalized-LMS regularization floor — the `ε` in `step /= ε + Σx²`, active whenever width-normalization is:
    /// [`Self::lr_fanin_norm`] is set, OR — only while [`Self::stability_cap`] is `0` (the [`Config::legacy`] genome)
    /// — the unit's fan-in is `>=` [`Self::width_norm_min`]. Under the 1.6 default (cap on) it is read only with the
    /// explicit flag. NOT a tiny epsilon — it sets how aggressively the step is width-normalized
    /// (small ⇒ hard normalize; large ⇒ barely normalize until Σx²≫ε), shaping the width-independence the
    /// capability promises. (was the buried `1.0`)
    #[serde(default = "default_nlms_eps")]
    pub nlms_eps: f32,
    /// SATURATION observed-floor (`saturated`): the slow surprise trace must exceed this for a head to count as
    /// "trained" (so a never-trained head is not reported saturated). Interacts with signal scale (a
    /// low-magnitude-surprise task's trace may sit below it), so it is a tunable, not a pure epsilon. (was `1e-4`)
    #[serde(default = "default_saturation_floor")]
    pub saturation_floor: f32,

    /// WIDTH NORMALIZATION (v2 capability fix — `lr∝1/W`). When `true`, `learn_step` divides the delta step by
    /// the input energy `Σ x²` of the unit's reads (Normalized-LMS), so the effective learning rate is
    /// **width-independent**: one `lr` serves a narrow *and* a wide readout — at EVERY fan-in.
    ///
    /// The flag is off by default. By default the raw step is instead kept stable by [`Self::stability_cap`], which
    /// never enlarges a step (so `lr` keeps its raw-LMS meaning wherever the raw step was stable). When the flag is
    /// set it wins over the cap. (With the cap off — [`Config::legacy`] — normalized-LMS also switches on
    /// automatically at fan-in `>=` [`Self::width_norm_min`].) Measured motivation: `v2_scale_width`.
    #[serde(default)]
    pub lr_fanin_norm: bool,

    /// DIMENSION-AWARE width-normalization threshold — the fan-in (`reads.len()`) at/above which `learn_step`
    /// AUTO-enables normalized-LMS even when [`Self::lr_fanin_norm`] is off. The un-normalized delta step scales
    /// with the input energy `Σx² ~ fan-in`, so the effective learning rate is `lr·fan-in` — stable while the
    /// fan-in is small but **DIVERGENT past ≈128** (identity-map RMSE → ∞ at 256), and the failure is SILENT (a
    /// diverged wide readout produces plausible-but-wrong numbers — it once read as "no action causality" and as
    /// "the full-frame world model saturates"). Normalizing automatically once the fan-in is large enough to be
    /// at risk removes that footgun for EVERY `grow`/`learn_step` consumer. The default `128` sits just below the
    /// empirical divergence onset.
    ///
    /// Read ONLY while [`Self::stability_cap`] is `0` (the [`Config::legacy`] genome). Since 1.6 the default cap
    /// replaces this threshold: the threshold made the effective step jump by `1/(1+Σx²)` between fan-in 127 and
    /// 128 (at `lr` 0.01 a readout that converged at 127 reads stopped learning at 128 — train MSE 0.0000 → 0.10),
    /// where the cap is continuous in the fan-in and leaves every stable step untouched. So a readout of `>= 128`
    /// reads that the legacy genome normalized to `lr/(1+Σx²)` steps `min(lr, κ/(1+Σx²))` under the default — up to
    /// `κ/lr` times larger (5× at `lr` 0.1, 50× at `lr` 0.01): a wide readout learns FASTER than before 1.6. Set
    /// [`Self::lr_fanin_norm`] to keep the normalized step at every fan-in. `usize::MAX` keeps the
    /// legacy raw step at every fan-in (only the explicit `lr_fanin_norm` flag then normalizes; the pre-#468
    /// behaviour).
    #[serde(default = "default_width_norm_min")]
    pub width_norm_min: usize,

    /// STEP-STABILITY CAP `κ` of the delta rule. The raw (un-normalized) LMS step `η` on a unit reading `x` is
    /// stable only while `η·(1+Σx²) < 2`; past that the readout diverges SILENTLY. With `κ > 0`, `learn_step` caps
    /// the BASE rate — the call's `lr` × the unit's heritable `lr` trait — at `κ/(1+Σx²)` whenever it exceeds it,
    /// and then applies the step's modulators (plasticity, the write-gate, the `plasticity_anneal` gain, each in
    /// `[0,1]`), so a half-plastic or half-gated unit still takes half the capped step. The cap never ENLARGES a
    /// step relative to the raw rule, so wherever the base rate was already below it the step is untouched, bit
    /// for bit, and `lr` keeps its raw-LMS meaning there; above it, every base rate takes the same capped step (that
    /// is the cap — heritable-`lr` competition acts only below it). It replaces the fan-in threshold
    /// [`Self::width_norm_min`] (ignored while the cap is on — which makes a readout of `>= 128` reads step
    /// `min(lr, κ/(1+Σx²))` instead of the legacy `lr/(1+Σx²)`, i.e. larger whenever `lr < κ`); an explicit
    /// [`Self::lr_fanin_norm`] still wins. `0.0` ⇒ OFF (the raw step, plus the threshold, of [`Config::legacy`]).
    /// Proof: `substrate_stability_cap`.
    ///
    /// ⚠️ DEFAULT BEHAVIOUR CHANGE (1.6): the default is `0.5` (was `0.0`). With the documented `lr` 0.1 and the
    /// default `max_hidden` 64, a grown readout's raw step measured `η·Σx²` = 3.2–4.4 and ended worse than
    /// predicting the mean on every 16- and 64-input benchmark task (NMSE 2.4–2.9) — below the auto-NLMS threshold,
    /// so nothing caught it; with the other flipped levers and no cap, reg2d ends at NMSE 1.69 and seq at 0.74
    /// (0.011 and 0.006 with it). `κ = 0.5` beat `κ = 1.0` on the benchmark (two-spirals 94 vs 78 %, pairs64 NMSE
    /// 0.73 vs 1.05). serde-default `0.5`: a snapshot written before the lever existed loads with the cap on (it
    /// never enlarges a step; set `0.0` to replay the raw step).
    pub stability_cap: f32,

    /// RECURSIVE-LEAST-SQUARES READOUT (opt-in), the online form of ridge regression. `0.0` (default) ⇒ OFF: the
    /// delta rule, byte-identical. `λ ∈ (0, 1]` ⇒ every unit `learn_step` corrects keeps an inverse input
    /// covariance `P` over its reads and its bias and steps by the Kalman gain `k = P·g/(λ + gᵀP·g)`, `g = f′·x`,
    /// instead of the delta rule; `P ← (P − k·gᵀP)/λ`. `λ` is the forgetting factor: `1.0` weighs every event
    /// equally (a linear readout then holds the closed-form ridge solution after every event, prior
    /// [`Self::rls_prior`]); `< 1` forgets with a memory of `≈ 1/(1−λ)` events, for a drifting target.
    ///
    /// It assumes a FIXED basis: an RLS readout makes no feature-direction step ([`Self::feature_lr`] is not read
    /// for it), so the features it reads stay the random projections they were born as, and `grow` still adds
    /// new ones (each enters `P` at the prior). Build the frozen-basis regressor with [`crate::Field::grow_backbone`]
    /// and one zero-weight readout per output (`add_unit`), or let `grow` size the basis. The call's `lr` only
    /// gates the step (`0` ⇒ no update; the gain sets its size); plasticity, the write-gate and the
    /// `plasticity_anneal` gain weight the event (weighted least squares); momentum and [`Self::stability_cap`]
    /// are not read. With `λ < 1`, `P`'s trace is capped at the prior's (`n·P₀`), the bound on RLS wind-up (at
    /// `λ = 0.999` with no bound, `reg2d` diverged to NMSE 7.4). Cost `O(fan-in²)` time and memory per corrected
    /// unit per event (a 66-read readout: 4 356 floats).
    ///
    /// Measured on the 12-task harness, 64-feature backbone, `λ = 1`, `P₀ = 100`, median of 5 seeds at 10k examples:
    /// a regression specialist. `reg2d` NMSE 0.0026 (the default organism 0.0111, the same frozen basis under the
    /// delta rule 0.169, batch random-feature ridge 0.0007), `reg1d` < 1e-5 (2e-4), `seq` 0.0039 (0.0053), `xor`
    /// 99.3 % (96.9). It loses every task that needs learned features: `spirals` 83 % (95), `parity8` 73 % (100),
    /// `pairs16` NMSE 0.88 (0.04), `drift16` 0.91 (0.08). Proof `substrate_rls_readout`.
    #[serde(default)]
    pub readout_rls: f32,

    /// The RLS readout's prior variance `P₀` on each weight ([`Self::readout_rls`]): `P` starts at `P₀·I`, which
    /// is ridge regression with penalty `1/P₀`. Larger trusts the first events more. Measured on `reg2d` (NMSE at
    /// 10k examples): `P₀` 10 → 0.0067, 100 → 0.0026, 1 000 → 0.0018, 10 000 diverges on 2 of 5 seeds (f32 loses
    /// `P`'s small eigenvalues); the default 100 keeps two decades of margin. Not read while `readout_rls` is `0`.
    #[serde(default = "default_rls_prior")]
    pub rls_prior: f32,

    /// RECURRENT-STABILITY NORMALIZATION (v2 capability fix — `g/√N`). When `true`, a unit's effective gain is
    /// scaled by `1/√fan_in`, so the pre-activation and the recurrent spectral radius `ρ ≈ gain·w·√N` stay
    /// bounded as fan-in grows — a coupled recurrent population no longer goes super-critical purely by
    /// *growing* (measured: `v2_scale_recurrence`). Off by default (byte-identical). A genome default for
    /// recurrent morphs. (The `ρ` self-diagnostic honors this normalization too.)
    #[serde(default)]
    pub gain_fanin_norm: bool,

    /// VARIANCE-MASKED width-normalization (opt-in refinement of [`Self::lr_fanin_norm`]). Plain width-norm
    /// divides the delta step by `Σ x²` over ALL of a unit's reads — but on a DENSE band an input that never
    /// fires reads a CONSTANT background value (e.g. `−1`), so it still charges `x² = 1` to the denominator and
    /// taxes the effective lr for the inputs that DO vary, for no information (MorphBot measured whole 18-dim
    /// augmentation blocks at EXACTLY zero variance over an episode — a purely illegitimate tax). When `true`,
    /// `learn_step` sums `x²` only over reads not yet seen to be CONSTANT (a constant read is dropped) — an
    /// "effective fan-in" that keeps NLMS's units (for a ±1 band, the COUNT of non-constant dims) instead of
    /// collapsing the scale the way a raw `Σ variance` would. "Constant" is decided EXACTLY, not by an EMA: each
    /// read cell keeps its observed min / max / count, and a read is dropped once it has been seen at least twice
    /// with `max − min <= 1e-9` (a read seen fewer than twice is still charged, so warm-up never collapses the
    /// denominator; a read that has ever varied is charged for good). Those statistics are DETACHED from the
    /// weights, so this has none of the reliance-weighted denominator's feedback instability, and it preserves
    /// the `−1` evidence (it only changes the step-size denominator, not what feeds the output or the gradient).
    /// Only active WHEN width-norm is (this flag AND `lr_fanin_norm`, or — only while [`Self::stability_cap`] is `0`,
    /// the [`Config::legacy`] genome — the `width_norm_min` auto-threshold; under the 1.6 default it needs the flag).
    /// Off by default (byte-identical): a new opt-in path, so every existing proof and the physics `.to_bits()`
    /// oracle are untouched. NOTE it can raise the effective lr several-fold when much of the band is idle — a
    /// larger intervention than the augmentation tax alone — so gate + measure before adopting as a default.
    #[serde(default)]
    pub var_gated_norm: bool,

    /// The unit NONLINEARITY (v2 genome knob) — `activation(gain·(Σw·x+b))`. Default `Tanh` = byte-identical.
    /// Swap to `Softsign` (bounded, `exp`-free, gentler saturation) or `HardTanh`; the gradient rule follows via
    /// the paired [`Activation::deriv`], and the `ρ` diagnostic lets you compare their stability. serde-default
    /// so old snapshots load `Tanh`.
    #[serde(default)]
    pub activation: Activation,

    /// #155 (OM-6) — the EPISTEMIC WEIGHT `w_epi` in the self-generated free-energy goal score
    /// [`crate::Field::orient`] maximises: `drive_of(c) + w_epi·learning_progress(c)` — extrinsic reward-memory
    /// PLUS epistemic REDUCIBLE surprise. It is a DISCOVERED coefficient, not authored: `substrate_meta_drives`
    /// meta-evolves the graded intrinsic-drive policy and — forced by the
    /// noisy-TV / dark-room anti-hack — keeps LEARNING-PROGRESS (not raw surprise) with a positive weight; the
    /// default here is that recorded GA output. It weights the *reducible* uncertainty term, which is what dodges
    /// the dark-room (a fully-predictable do-nothing state has zero learning-progress, so it is never preferred).
    /// `0.0` disables the epistemic term (pure extrinsic goal), byte-identical to a drive-only orient. serde-default.
    #[serde(default = "default_epistemic_weight")]
    pub epistemic_weight: f32,

    /// GOAL-COMPLETION REWARD (#155 close-the-loop) — the intrinsic reward the organism pays ITSELF for reaching a
    /// SELF-SET goal. A goal is a CLAMPED preferred cell ([`crate::Field::prefer`] / [`crate::Field::orient`] — a
    /// cell it wants activated); when `alive(Event::Act)`'s taken option lands on such a cell, the organism emits
    /// `goal_reward` and folds it through the SAME channel as an external reward (the learned option value of
    /// [`ActRule::Valence`] or [`ActRule::FreeEnergy`]), then CONSUMES the goal (unclamps it) so `orient` can set
    /// the next. This closes the loop create-goal → pursue → ACHIEVE → reward with no external reward needed. `0.0`
    /// (default) = OFF, byte-identical: no self-reward, no consume. serde-default.
    #[serde(default = "default_goal_reward")]
    pub goal_reward: f32,

    /// METABOLIC cost — the energy the organism's metabolism loses per internal tick of [`crate::Field::live`] and
    /// per bout of a metered [`crate::Field::dream`] ([`Self::dream_satiation`] `> 0`). Higher ⇒ hungrier sooner.
    /// (Acting no longer feeds or drains the metabolism: the metabolic actor that did went with the removed
    /// `ActScore` rule.) serde-default 0.5.
    #[serde(default = "default_metabolic_cost")]
    pub metabolic_cost: f32,

    /// ACTION DWELL / PERSISTENCE (opt-in) — the actor's missing TEMPORAL axis. `alive(Event::Act)` re-decides
    /// among the options every step, so a policy that must COMMIT to one action for many steps has to rediscover
    /// that choice independently each time. With `action_dwell` on, the actor carries a GROWN commitment DURATION
    /// per option: a chosen action is held for a learned number of steps before re-deciding. The duration is not
    /// authored — it starts at 1 (re-decide every step ⇒ identical to off) and is tuned by reward
    /// ([`Self::dwell_rate`], [`Self::dwell_max`]): repeating an action that keeps EARNING extends its dwell,
    /// one that stops earning releases it. This is the TEMPORAL analogue of grown structural DEPTH — grow widens
    /// one layer, dwell holds one choice. `false` (default) ⇒ the actor re-decides every step exactly as before,
    /// byte-identical (the dwell state is never touched).
    #[serde(default)]
    pub action_dwell: bool,
    /// How fast a repeated action's grown dwell DURATION moves with reward (± per act, by reward sign). Read only
    /// when [`Self::action_dwell`]. serde-default 0.5.
    #[serde(default = "default_dwell_rate")]
    pub dwell_rate: f32,
    /// The ceiling on a grown dwell duration (how many steps an action can be held before a forced re-decide).
    /// Read only when [`Self::action_dwell`]. serde-default 32.
    #[serde(default = "default_dwell_max")]
    pub dwell_max: f32,

    /// CREDIT INTO NESTED CELLS (opt-in) — the composition CAPABILITY lever. A nested cell is a FROZEN brick:
    /// [`crate::Field::learn_step`] and feature-direction learning both SKIP it, so a sub-skill trained in
    /// isolation before nesting only ever saw its own inputs against a label that depends on other blocks it
    /// cannot see — an impossible learning problem, and why a grown sub-skill lands BELOW a flat model. With
    /// `nest_credit` on, the composed error at the host readout is routed ONE HOP into the frozen sub's readout:
    /// the sub's output cell gets a feedback-aligned pseudo-target (the residual backprojected through the host
    /// readout's weight on the nested cell — the SAME one-hop delta as [`Self::feature_lr`], one nesting-level
    /// down), and the sub's EXISTING units are tuned by one `learn_step` toward it. The sub's STRUCTURE stays
    /// frozen (no growth/birth); only its weights move, so it remains a banked, addressable skill. REQUIREMENT:
    /// the sub needs grown hidden capacity (`max_hidden > 0` when it was grown) — a plastic LINEAR readout over
    /// the raw inputs cannot carve the sub-function (it collapses to chance on a non-linearly-separable target).
    /// A plastic nested cell cannot memoize (its function changes each step), so its memo is cleared while credit
    /// flows. `false` (default) ⇒ nested cells stay frozen bricks exactly as before, byte-identical (the sub is
    /// never taken out to learn). Proof `substrate_nest_credit`.
    ///
    /// SCOPE (honest bound, measured in the proof): one-hop credit ALIGNS a SEPARABLE sub-function (AND: sub↔f
    /// 0.693→0.965 from the composed error alone) but does NOT BOOTSTRAP a NON-separable one (2-way XOR: 0.73,
    /// ≈ the linear ceiling, vs the 0.91 the same hidden sub reaches under DIRECT supervision). The cause is the
    /// backprojected credit itself: `pe = err·deriv·w_host`, so a still-useless sub gets down-weighted by the host
    /// (`w_host → 0`) which collapses its own credit before it can carve a non-linear corner — a chicken-and-egg
    /// the separable case escapes because a noise-sub already carries partial signal. So this ships the MECHANISM
    /// (a nested sub can be REACHED at all — 0.717 below-flat becomes a working composition), not a claim that it
    /// learns arbitrary sub-functions. The natural next lever is the BOOTSTRAP itself — keeping the backprojected
    /// credit alive while the sub is still useless (e.g. a warmup that floors `|w_host|` for the credit term) so a
    /// non-separable sub can get off the ground; that is a separate, un-built follow-up, not this flag.
    #[serde(default)]
    pub nest_credit: bool,
    /// The learning rate applied to a nested sub's readout when [`Self::nest_credit`] is on (the step size of the
    /// one-hop composed-error delta into the frozen sub). Read only when `nest_credit`. serde-default 0.1.
    #[serde(default = "default_nest_credit_lr")]
    pub nest_credit_lr: f32,

    /// WHEN TO DREAM — the SATIATION floor an organism must reach before it turns inward to [`crate::Field::dream`]
    /// (internal thinking). Dreaming is a LUXURY of the fed: the organism processes its experience only while its
    /// energy account is above this fraction of `energy_cap`. When the coupling is active (`> 0`) each dream bout
    /// also BURNS [`Self::metabolic_cost`] from the metabolism, so dreaming spends the reserve it requires and
    /// SELF-LIMITS — the organism dreams until hungry. Nothing in core refills the reserve since the metabolic actor
    /// (which fed it the `Act` reward) was removed with the `ActScore` rule: a new field starts fed, and once the
    /// reserve falls below the floor it stays there. `0.0` (default) ⇒ dreaming is ungated and free, byte-identical
    /// to before the coupling. Read by [`crate::Field::should_dream`] and `dream`.
    #[serde(default)]
    pub dream_satiation: f32,

    /// **Deprecated alias of `act_rule = FreeEnergy`** (kept so existing configs, MorphBot's `Player` and the
    /// free-energy proofs keep their exact behaviour; removed next release). When `true`,
    /// [`crate::Field::alive`]`(Event::Act)` runs [`ActRule::FreeEnergy`] whatever [`Self::act_rule`] says, and it
    /// WINS over [`Self::valence_actor`] if both are set (the pre-1.6 precedence) — see
    /// [`Config::effective_act_rule`]. The rule is the EXPECTED-FREE-ENERGY functional over a GROWN per-option value
    /// ENSEMBLE (μ/σ² read by the substrate's own [`crate::grouping::aggregate`]):
    ///
    /// ```text
    ///     G(o) = value_of(o)   +   (μ_o − s*)²   −   σ_o²          (ARGMIN, ties → first)
    ///            \_ plan _/        \_ pragmatic _/     \_ epistemic (shrinks with every visit) _/
    /// ```
    ///
    /// It is a SET-POINT matcher: the pragmatic term prefers the reward nearest [`Self::free_energy_sstar`], so it
    /// is not monotone in reward once rewards leave `[0, s*]` (on a 10-arm bandit with rewards ×10 it is worse than
    /// uniform random — `examples/actor_bench.rs` in `openmorph-perception`). serde-default `false`.
    #[serde(default)]
    pub free_energy_actor: bool,
    /// FREE-ENERGY preferred observation `s*` — the reward the organism prefers to reach (the pragmatic target in
    /// `G = (μ−s*)² − σ²`). Read only by [`ActRule::FreeEnergy`]. Default `0.9` (the tanh-genome reward range of the
    /// free-energy proofs). This is `reward`-as-a-`prefer`red-prior: `s*` IS the reward signal.
    #[serde(default = "default_free_energy_sstar")]
    pub free_energy_sstar: f32,
    /// Prior epistemic uncertainty `σ0` of an option's learned value — the exploration bonus `σ0²` of a
    /// NEVER-VISITED option (mean 0 = uninformative) before visits anneal it (`σ = σ0·(1−free_energy_lr)^visits`);
    /// larger ⇒ explores longer. Under [`ActRule::Valence`] it also sets the reward SCALE the bonus is measured
    /// against (`μ + σ²` adds reward to reward²): for rewards of scale `c` use `σ0 ≈ √c`. Under
    /// [`ActRule::FreeEnergy`] it is the standard deviation of the prior value ensemble. Default `1.0`.
    #[serde(default = "default_free_energy_prior_std")]
    pub free_energy_prior_std: f32,
    /// SIZE of the per-option value ensemble of [`ActRule::FreeEnergy`]. Every member takes the same step, so the
    /// ensemble carries exactly two numbers — the mean `μ` (an EMA of the fold targets from 0, up to an f32 prior
    /// residue of −1.99e-8 at 24 members) and the spread `σ0·(1−lr)^visits` (a visit count): σ is NOT a
    /// reward-noise estimate. [`ActRule::Valence`] stores those two numbers directly (`μ` and the visit count), so
    /// it does not read this. Default `24`.
    #[serde(default = "default_free_energy_ensemble")]
    pub free_energy_ensemble: usize,
    /// LEARNING RATE of an option's learned value — the step `μ` (or, under [`ActRule::FreeEnergy`], each ensemble
    /// member) takes toward the fold target (`μ ← μ + lr·(target − μ)`). Governs how fast μ tracks the reward and
    /// how fast σ² anneals per visit (`σ = σ0·(1−lr)^visits`). Read by [`ActRule::Valence`] and
    /// [`ActRule::FreeEnergy`]. Default `0.15`.
    #[serde(default = "default_free_energy_lr")]
    pub free_energy_lr: f32,

    /// **Deprecated alias of `act_rule = Valence`** (removed next release). When `true` (and
    /// [`Self::free_energy_actor`] is `false`), `alive(Event::Act)` runs [`ActRule::Valence`] whatever
    /// [`Self::act_rule`] says — see [`Config::effective_act_rule`]. Since 1.6 Valence is the default rule, so the
    /// flag only matters to override a config that pins another rule. serde-default `false`. Proof
    /// `substrate_valence`.
    #[serde(default)]
    pub valence_actor: bool,

    /// THE ACTING RULE of [`crate::Field::alive`]`(Event::Act)` — see [`ActRule`]. Default [`ActRule::Valence`]
    /// (reward-maximising `−value + μ + σ²`, exact ties broken by the seeded act PRNG). The deprecated bools
    /// [`Self::free_energy_actor`] / [`Self::valence_actor`] override it when set; read the rule that actually runs
    /// through [`Config::effective_act_rule`]. [`Config::legacy`] keeps the default (the pre-1.6 `ActScore` rule is
    /// removed; the name still loads, as `Valence`). A categorical lever, so it is not in the f32
    /// [`Config::schema`]. serde-default `Valence`.
    #[serde(default)]
    pub act_rule: ActRule,

    /// TEMPORAL-DIFFERENCE DISCOUNT `γ` of the learned-value fold ([`ActRule::Valence`] and
    /// [`ActRule::FreeEnergy`]). `alive(Event::Act { options, last })`
    /// folds the last option's reward toward the target `r + γ·max_{o' ∈ options} μ_{o'}` — the options THIS call
    /// offers are the successors of the last (option, reward) — so a reward flows back along the options that
    /// led to it (one-step Q-learning over option cells, no API change). `0.0` (default) = the immediate-reward fold,
    /// BYTE-IDENTICAL to before this lever existed: right for a bandit (every call offers the same options, so any
    /// `γ > 0` would bootstrap an arm from itself). Set `0.9`–`0.99` for a multi-step task (a sparse goal several
    /// actions away). An episodic caller leaks one bootstrap across an episode boundary (the terminal action is
    /// folded toward the next episode's first options); there is no `done` flag. Proof `substrate_act_discount`.
    #[serde(default)]
    pub act_discount: f32,

    /// SURPRISE-CURRENCY (Gap 7 spike) — track a running per-cell variance so `surprise_reduction_of` can report
    /// the ablation importance `Σ_readers w² · Var(cell)`: the STABLE-at-convergence, CAUSAL currency that unifies
    /// credit + economy-income + grouping (`docs/research/surprise-currency-spike.md`). `false` (default) keeps the
    /// per-cell variance state EMPTY and never updated ⇒ BYTE-IDENTICAL (the learn/settle path is untouched).
    /// serde-default. Proof `spike_surprise_currency`.
    #[serde(default)]
    pub surprise_currency: bool,

    /// POOL CURRENCY (opt-in): `alive(Event::Perceive)` pays its living pool by SURPRISE REDUCTION instead of reliance.
    /// A feature earns `pool_income` × its end-to-end ablation importance `(∂output/∂c)²·Var(c)` (how much output
    /// error removing it would add, [`crate::Field::surprise_reduction_of`]) over the pool's mean ([`Self::reliance_norm`]
    /// applies to it as to reliance), and pays `pool_upkeep`. Turns on the variance and sensitivity tracking that
    /// currency needs (as [`Self::surprise_currency`] does). `false` (default) ⇒ the reliance economy, byte-identical.
    ///
    /// A parsimony lever for tasks whose answer fits in a few features: on the 12-task harness it shrinks the pool
    /// 2–4× at equal or better error on `reg1d`, `reg2d`, `seq`, `parity4`/`parity8`, `xor4` and `multi10`, and loses
    /// where the task needs a wide basis (`spirals` 95 → 92 %, `pairs16` NMSE 0.04 → 0.07, `pairs64` 0.73 → 1.20,
    /// `drift16` 0.08 → 0.24). Rule of thumb: if capping [`Self::max_hidden`] at 16 costs the task nothing, this makes
    /// the pool smaller for free. It values each feature alone, so it undersizes a distributed basis and, on wide
    /// inputs, culls newborns before they earn (a birth/death treadmill). One backward sweep per learning event
    /// (`multi10`: half the units, about 25 % slower). Proof `substrate_pool_surprise`; analysis
    /// `docs/research/pool-surprise-currency.md`.
    #[serde(default)]
    pub pool_surprise: bool,

    /// CONCEPT-MAP dimensionality FLOOR — the initial number of axes the grown concept-metric uses the first
    /// time it is touched (`position`/`distance`/`metric_step`). Grown finer on demand by `grow_dimension`
    /// (Matryoshka nesting), so this is only the floor. Concept-map capacity is a suspected ARC-3 ceiling
    /// contributor, so a tuner may raise it. Default `2` = historical, byte-identical. (was the buried `2`.)
    #[serde(default = "default_pos_dim_init")]
    pub pos_dim_init: usize,
    /// METRIC-STEP repulsion SOFTENING — the `s` in the negative-sampling repulsion force `rate/(s + dist²)`
    /// (`metric_step`). It sets the length-scale at which repulsion saturates: small ⇒ a sharp short-range push
    /// (crisper local metric), large ⇒ a gentle long-range one. Shapes map fidelity (`metric_step` lifts
    /// map-vs-grid 0.56→0.96). Default `1.0` = historical, byte-identical. (was the buried `1.0`.)
    #[serde(default = "default_metric_repel_softening")]
    pub metric_repel_softening: f32,
    /// BEHAVIOURAL-KEY settle PADDING — the extra settle budget over the cell count when reading a morph's probe
    /// response for `fingerprint`/`embed` (`budget = n_cells + probe_settle_pad`). Too small silently truncates
    /// the settle on a deep nested skill, degrading the key; larger costs more per probe. A per-library key
    /// CONTRACT (changing it re-shapes stored keys). Default `32` = historical, byte-identical. (was the buried `32`.)
    #[serde(default = "default_probe_settle_pad")]
    pub probe_settle_pad: usize,
    /// STRUCTURE-WALK recursion GUARD — the maximum composition-tree depth the prime/tree read-algebra descends
    /// (`structure_key`/`factorization`/`gcs`/`coprime`/`valuation`). The `nested` tree is a tree by construction
    /// (no back-edges), so this only bounds pathologically deep towers; mirrors the mycelium lineage walk's
    /// `max_depth`. Default `64`. (was the buried `64`.)
    #[serde(default = "default_struct_walk_max_depth")]
    pub struct_walk_max_depth: usize,

    /// NEST BEHAVIOUR-SEEDING (#172) — when `true` (default), [`crate::Field::nest`] seeds the new nested cell's
    /// concept-map POSITION from the sub-morph's own [`crate::Field::embed`] (its coarse spectral prefix = a coarse
    /// position, PROJECTED to the current `pos_dim`, never growing the map), so a nested skill lands on the map by
    /// WHAT IT DOES rather than by scatter noise — unifying core's positional `route` and mycelium's cosine
    /// retrieval onto ONE behaviour key. NOT byte-identical (a nested cell's position materialises), but
    /// behaviour-preserving for every existing caller (routing proofs `place_at` their grid keys AFTER nest, and
    /// nothing reads a nested cell's position). Set `false` to recover strict byte-identity / skip the per-nest
    /// embed cost in bulk-nesting. serde-default `true` so old snapshots keep it on. A bool → not a numeric lever.
    #[serde(default = "default_true")]
    pub nest_seeds_position: bool,

    /// AUTONOMOUS-HEARTBEAT dormancy RAMP RATE `k` (D6/D7). In [`crate::Field::live`], while the field is
    /// quiescent, each cell's reactivation pressure ramps `P_c += k·(1−recency_c)` — a leaky integrate-and-fire
    /// on dormancy. Higher ⇒ dormant memories cross the fire threshold ([`Self::heartbeat_theta`]) sooner (more
    /// eager offline reactivation / "dreaming"); lower ⇒ the organism sleeps longer between spontaneous
    /// completions. Only read by `live`, so a field whose heartbeat is never invoked is byte-identical to before.
    /// Default `0.5` (≈2 quiescent ticks to fire at unit dormancy). See `docs/research/organism-differential-
    /// equations.md` §3.
    #[serde(default = "default_heartbeat_k")]
    pub heartbeat_k: f32,
    /// AUTONOMOUS-HEARTBEAT fire THRESHOLD `θ` (D6/D7). In [`crate::Field::live`], when a cell's reactivation
    /// pressure `P_c` (ramped by [`Self::heartbeat_k`]) exceeds this AND the field is quiescent, the longest-
    /// dormant grown cell spontaneously fires (pattern completion), the existing contribution economy scores it
    /// (`reliance_of`), and it is kept (`consolidate`) or pruned (`cull`); then `P_c` discharges to 0. Higher ⇒
    /// rarer, more-deliberate reactivations; lower ⇒ twitchier. Only read by `live` (byte-identical when the
    /// heartbeat is never invoked). Default `1.0`.
    #[serde(default = "default_heartbeat_theta")]
    pub heartbeat_theta: f32,
}

/// Whether a [`Config`] lever is a fixed dial the caller sets, or the SEED of a trait the substrate then
/// tunes itself. A Tuning UI renders `Authored` as an editable slider and `Heritable` as a read-only seed
/// (its live per-unit value drifts by competition — see `reproduce` / `metabolize_contribution`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeverKind {
    /// A fixed dial the caller owns (the substrate never changes it).
    Authored,
    /// The INITIAL value of a per-unit heritable trait; the substrate auto-tunes the trait from here.
    Heritable,
}

/// Machine-readable metadata for one [`Config`] lever — name, default, a sensible tuning range, and whether
/// it is authored or heritable. Returned by [`Config::schema`] so a UI (or Optuna) stays in sync with the
/// substrate's real knobs without hard-coding them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeverInfo {
    /// The field name (matches the `Config` JSON key).
    pub name: &'static str,
    /// The default value (from [`Config::default`]), as `f32`.
    pub default: f32,
    /// A sensible `[min, max]` tuning range (the Optuna search span; see `docs/config-reference.md`).
    pub range: (f32, f32),
    /// Authored (a fixed dial) or Heritable (a seed the substrate auto-tunes).
    pub kind: LeverKind,
}

impl Config {
    /// The pre-1.6 LEARNING genome: [`Config::default`] with every LEARNING lever 1.6 flipped undone (the raw delta
    /// step with the fan-in-128 auto-normalization, random feature directions, the fan-in-blind feature init, the
    /// absolute reliance economy, births up to `max_hidden` whatever the settle budget). Acting is
    /// [`ActRule::Valence`] on every genome: the pre-1.6 `ActScore` rule 1.6 replaced is removed, so an organism
    /// that acts no longer replays its pre-1.6 choices. On the learning path a field built with it is bit-identical to
    /// the pre-FLIP default organism (proof `substrate_legacy_replay` pins it to a hash at `38074e7`). It does not
    /// undo the 2026-09 correctness fixes (the stale-output refresh, one settle per observation, exact
    /// `encapsulate`/`nest`), which run on every genome: an organism from before them replays exactly only where they
    /// did not fire. Use it to replay an old organism's learning or to reproduce a pre-1.6 result. The physics engine
    /// (`openmorph-physics`) models this genome.
    pub fn legacy() -> Self {
        Config {
            stability_cap: 0.0,
            width_norm_min: 128,
            feature_lr: 0.0,
            fanin_init_gain: 0.0,
            reliance_norm: RelianceNorm::Absolute,
            grow_within_budget: false,
            ..Config::default()
        }
    }

    /// The rule `alive(Event::Act)` actually runs: [`ActRule::FreeEnergy`] if the deprecated
    /// [`Self::free_energy_actor`] is set, else [`ActRule::Valence`] if the deprecated [`Self::valence_actor`] is set,
    /// else [`Self::act_rule`]. (The bools keep their pre-1.6 meaning and precedence — FreeEnergy wins if both.)
    pub fn effective_act_rule(&self) -> ActRule {
        if self.free_energy_actor {
            ActRule::FreeEnergy
        } else if self.valence_actor {
            ActRule::Valence
        } else {
            self.act_rule
        }
    }

    /// Whether the substrate tracks the per-cell variance and end-to-end sensitivity of the surprise currency
    /// ([`Self::surprise_currency`] or [`Self::pool_surprise`]).
    pub(crate) fn tracks_surprise(&self) -> bool {
        self.surprise_currency || self.pool_surprise
    }

    /// The reliance economy [`crate::Field::metabolize_reliance`] actually runs: [`RelianceNorm::Share`] if the
    /// deprecated [`Self::relative_reliance`] is set, else [`Self::reliance_norm`].
    pub fn effective_reliance_norm(&self) -> RelianceNorm {
        if self.relative_reliance {
            RelianceNorm::Share
        } else {
            self.reliance_norm
        }
    }

    /// The lever SCHEMA — every tunable field with its default, a sensible range, and authored-vs-heritable
    /// kind. A Tuning UI renders a slider per authored lever (with the right range) and shows heritable
    /// levers read-only with their live value; it stays correct automatically as `Config` gains/loses
    /// fields, because this is the single source of truth. See `docs/config-reference.md` for the full table.
    pub fn schema() -> Vec<LeverInfo> {
        use LeverKind::{Authored, Heritable};
        let d = Config::default();
        vec![
            LeverInfo {
                name: "birth_energy",
                default: d.birth_energy,
                range: (0.5, 10.0),
                kind: Authored,
            },
            LeverInfo {
                name: "energy_cap",
                default: d.energy_cap,
                range: (1.0, 50.0),
                kind: Authored,
            },
            LeverInfo {
                name: "surprise_decay",
                default: d.surprise_decay,
                range: (0.5, 0.999),
                kind: Authored,
            },
            LeverInfo {
                name: "credit_decay",
                default: d.credit_decay,
                range: (0.9, 0.9999),
                kind: Authored,
            },
            LeverInfo {
                name: "plasticity_anneal",
                default: d.plasticity_anneal,
                range: (0.0, 0.1),
                kind: Authored,
            },
            LeverInfo {
                name: "plasticity_reset_ratio",
                default: d.plasticity_reset_ratio,
                range: (1.0, 5.0),
                kind: Authored,
            },
            LeverInfo {
                name: "product_gate",
                default: d.product_gate,
                range: (0.0, 0.5),
                kind: Authored,
            },
            LeverInfo {
                name: "birth_gate",
                default: d.birth_gate,
                range: (0.01, 0.5),
                kind: Authored,
            },
            LeverInfo {
                name: "max_hidden",
                default: d.max_hidden as f32,
                range: (0.0, 256.0),
                kind: Authored,
            },
            LeverInfo {
                name: "memo_cap",
                default: d.memo_cap as f32,
                range: (256.0, 65536.0),
                kind: Authored,
            },
            LeverInfo {
                name: "feature_lr",
                default: d.feature_lr,
                range: (0.0, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "fanin_init_gain",
                default: d.fanin_init_gain,
                range: (0.0, 6.0),
                kind: Authored,
            },
            LeverInfo {
                name: "consolidation_rate",
                default: d.consolidation_rate,
                range: (0.0, 0.5),
                kind: Authored,
            },
            LeverInfo {
                name: "fingerprint_probes",
                default: d.fingerprint_probes as f32,
                range: (4.0, 32.0),
                kind: Authored,
            },
            LeverInfo {
                name: "spectrum_bins",
                default: d.spectrum_bins as f32,
                range: (8.0, 64.0),
                kind: Authored,
            },
            LeverInfo {
                name: "embed_gain_weight",
                default: d.embed_gain_weight,
                range: (0.0, 8.0),
                kind: Authored,
            },
            LeverInfo {
                name: "self_adapt_tau",
                default: d.self_adapt_tau,
                range: (0.05, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "settle_eps",
                default: d.settle_eps,
                range: (1e-6, 1e-2),
                kind: Authored,
            },
            LeverInfo {
                name: "adapt_step_cap",
                default: d.adapt_step_cap,
                range: (0.01, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "bind_tol",
                default: d.bind_tol,
                range: (1e-6, 1e-1),
                kind: Authored,
            },
            LeverInfo {
                name: "pool_income",
                default: d.pool_income,
                range: (0.5, 5.0),
                kind: Authored,
            },
            LeverInfo {
                name: "pool_upkeep",
                default: d.pool_upkeep,
                range: (0.05, 2.0),
                kind: Authored,
            },
            LeverInfo {
                name: "pool_death",
                default: d.pool_death,
                range: (0.01, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "pool_grace",
                default: d.pool_grace as f32,
                range: (0.0, 5000.0),
                kind: Authored,
            },
            // The substrate tunes the per-unit `scale`/`lr`/`mut_rate` traits by competition; these are the seeds.
            LeverInfo {
                name: "init_scale",
                default: d.init_scale,
                range: (0.05, 5.0),
                kind: Heritable,
            },
            LeverInfo {
                name: "init_lr",
                default: d.init_lr,
                range: (0.05, 8.0),
                kind: Heritable,
            },
            LeverInfo {
                name: "init_mut_rate",
                default: d.init_mut_rate,
                range: (0.02, 1.5),
                kind: Heritable,
            },
            LeverInfo {
                name: "init_momentum",
                default: d.init_momentum,
                range: (0.0, 0.97),
                kind: Heritable,
            },
            // Evolution clamps — authored BOUNDS within which the heritable traits are kept.
            LeverInfo {
                name: "sigma_min",
                default: d.sigma_min,
                range: (0.001, 0.5),
                kind: Authored,
            },
            LeverInfo {
                name: "sigma_max",
                default: d.sigma_max,
                range: (0.1, 3.0),
                kind: Authored,
            },
            LeverInfo {
                name: "scale_min",
                default: d.scale_min,
                range: (0.01, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "scale_max",
                default: d.scale_max,
                range: (1.0, 10.0),
                kind: Authored,
            },
            LeverInfo {
                name: "lr_min",
                default: d.lr_min,
                range: (0.01, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "lr_max",
                default: d.lr_max,
                range: (1.0, 16.0),
                kind: Authored,
            },
            LeverInfo {
                name: "momentum_min",
                default: d.momentum_min,
                range: (0.0, 0.5),
                kind: Authored,
            },
            LeverInfo {
                name: "momentum_max",
                default: d.momentum_max,
                range: (0.9, 0.999),
                kind: Authored,
            },
            LeverInfo {
                name: "resonator_radius_max",
                default: d.resonator_radius_max,
                range: (0.9, 0.9999),
                kind: Authored,
            },
            LeverInfo {
                name: "nlms_eps",
                default: d.nlms_eps,
                range: (0.001, 10.0),
                kind: Authored,
            },
            LeverInfo {
                name: "width_norm_min",
                default: d.width_norm_min as f32,
                range: (16.0, 4096.0),
                kind: Authored,
            },
            LeverInfo {
                name: "stability_cap",
                default: d.stability_cap,
                range: (0.0, 2.0),
                kind: Authored,
            },
            LeverInfo {
                name: "readout_rls",
                default: d.readout_rls,
                range: (0.0, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "rls_prior",
                default: d.rls_prior,
                range: (0.01, 1000.0),
                kind: Authored,
            },
            LeverInfo {
                name: "saturation_floor",
                default: d.saturation_floor,
                range: (1e-6, 0.1),
                kind: Authored,
            },
            LeverInfo {
                name: "metabolic_cost",
                default: d.metabolic_cost,
                range: (0.0, 3.0),
                kind: Authored,
            },
            LeverInfo {
                name: "dwell_rate",
                default: d.dwell_rate,
                range: (0.0, 2.0),
                kind: Authored,
            },
            LeverInfo {
                name: "dwell_max",
                default: d.dwell_max,
                range: (1.0, 128.0),
                kind: Authored,
            },
            LeverInfo {
                name: "nest_credit_lr",
                default: d.nest_credit_lr,
                range: (0.0, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "dream_satiation",
                default: d.dream_satiation,
                range: (0.0, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "epistemic_weight",
                default: d.epistemic_weight,
                range: (0.0, 2.0),
                kind: Authored,
            },
            LeverInfo {
                name: "goal_reward",
                default: d.goal_reward,
                range: (0.0, 2.0),
                kind: Authored,
            },
            LeverInfo {
                name: "pos_dim_init",
                default: d.pos_dim_init as f32,
                range: (2.0, 8.0),
                kind: Authored,
            },
            LeverInfo {
                name: "metric_repel_softening",
                default: d.metric_repel_softening,
                range: (0.1, 10.0),
                kind: Authored,
            },
            LeverInfo {
                name: "probe_settle_pad",
                default: d.probe_settle_pad as f32,
                range: (8.0, 256.0),
                kind: Authored,
            },
            LeverInfo {
                name: "struct_walk_max_depth",
                default: d.struct_walk_max_depth as f32,
                range: (4.0, 1024.0),
                kind: Authored,
            },
            LeverInfo {
                name: "heartbeat_k",
                default: d.heartbeat_k,
                range: (0.0, 2.0),
                kind: Authored,
            },
            LeverInfo {
                name: "heartbeat_theta",
                default: d.heartbeat_theta,
                range: (0.1, 10.0),
                kind: Authored,
            },
            // Valence / free-energy actor (A6) — the preferred prior s* (FreeEnergy), the exploration prior σ0, the
            // ensemble size (FreeEnergy), the value lr.
            LeverInfo {
                name: "free_energy_sstar",
                default: d.free_energy_sstar,
                range: (-1.0, 1.0),
                kind: Authored,
            },
            LeverInfo {
                name: "free_energy_prior_std",
                default: d.free_energy_prior_std,
                range: (0.1, 3.0),
                kind: Authored,
            },
            LeverInfo {
                name: "free_energy_ensemble",
                default: d.free_energy_ensemble as f32,
                range: (4.0, 64.0),
                kind: Authored,
            },
            LeverInfo {
                name: "free_energy_lr",
                default: d.free_energy_lr,
                range: (0.01, 1.0),
                kind: Authored,
            },
            // The TD discount of the Valence/FreeEnergy fold (0 = immediate reward). `act_rule` itself is a
            // categorical lever, which `LeverInfo` (an f32 default + range) cannot express — like the bool toggles.
            LeverInfo {
                name: "act_discount",
                default: d.act_discount,
                range: (0.0, 0.99),
                kind: Authored,
            },
        ]
    }
}

impl Default for Config {
    /// The historical hard-coded values, except where a measured improvement replaced one — each marked a DEFAULT
    /// BEHAVIOUR CHANGE on its field: [`Config::width_norm_min`] = 128 (#468); and in 1.6 the acting rule
    /// ([`Config::act_rule`] = [`ActRule::Valence`]) and the learning levers [`Config::stability_cap`] = 0.5,
    /// [`Config::feature_lr`] = 0.3, [`Config::fanin_init_gain`] = 3, [`Config::reliance_norm`] =
    /// [`RelianceNorm::Mean`] and [`Config::grow_within_budget`]. [`Config::legacy`] restores every 1.6 LEARNING
    /// change, bit for bit (the pre-1.6 acting rule is removed).
    fn default() -> Self {
        Self {
            birth_energy: 3.0,
            energy_cap: 10.0,
            surprise_decay: 0.9,
            credit_decay: 0.99,
            plasticity_anneal: default_plasticity_anneal(),
            plasticity_reset_ratio: default_plasticity_reset_ratio(),
            grow_products: false,
            products_per_target: false,
            product_gate: default_product_gate(),
            birth_gate: 0.1, // autonomous-grow surprise threshold to birth a feature (#128)
            max_hidden: 64, // autonomous-grow capacity ceiling (surprise births only what's needed)
            grow_within_budget: true, // 1.6 — a birth must fit the settle budget (legacy false: births to max_hidden)
            memo_cap: 4096,           // nested-cell memo cache bound (was the buried MEMO_CAP)
            feature_lr: 0.3, // 1.6 — feature directions learn one hop below the readout (#125; legacy 0)
            fanin_init_gain: 3.0, // 1.6 — grown features drawn U[-1,1]·3/√fan_in (legacy 0 = U[-1,1])
            consolidation_rate: 0.0, // OFF by default — the mastery meter is never allocated unless a caller opts in
            fingerprint_probes: default_fingerprint_probes(), // 8 — behavioural-key probe count (byte-identical)
            spectrum_bins: default_spectrum_bins(), // 16 — behavioural-key spectral resolution (byte-identical)
            embed_gain_weight: default_embed_gain_weight(), // 2.0 — OM-11 embedding gain-coordinate weight
            self_adapt_tau: 0.4,
            settle_eps: 1e-4,
            adapt_step_cap: 0.2, // temporal half-life gradient step cap (#137 adapt_scales; was STEP_CAP)
            bind_tol: 1e-3,      // bind equality tolerance (#122 bind; was the buried 1e-3)
            pool_income: 2.0,    // living-pool reliance income (alive)
            pool_upkeep: 0.25, // living-pool upkeep — small vs income so breakeven is near-0 reliance (was 1.0)
            pool_death: 0.1,   // living-pool cull threshold (was 0.5)
            pool_grace: 200,   // living-pool newborn grace (events)
            relative_reliance: false, // deprecated alias of reliance_norm = Share
            reliance_norm: RelianceNorm::Mean, // 1.6 — income relative to the pool mean (legacy Absolute)
            init_scale: 1.0,
            init_lr: 1.0,
            init_mut_rate: 0.3,
            init_momentum: 0.0, // off = historical 1st-order delta rule (byte-identical)
            sigma_min: 0.02,
            sigma_max: 1.5,
            scale_min: 0.05,
            scale_max: 5.0,
            lr_min: 0.05,
            lr_max: 8.0,
            momentum_min: default_momentum_min(), // 0.0 — evolution clamp on the heritable momentum trait
            momentum_max: default_momentum_max(), // 0.99 — parity with sigma/scale/lr max clamps
            resonator_radius_max: default_resonator_radius_max(), // 0.999 — grown-resonator ring ceiling
            nlms_eps: default_nlms_eps(), // 1.0 — width-norm regularizer (lr_fanin_norm, or fan-in ≥ width_norm_min with the cap off)
            saturation_floor: default_saturation_floor(), // 1e-4 — saturated() observed-floor
            lr_fanin_norm: false, // v2 knob — off ⇒ raw step below width_norm_min (auto-NLMS at/above it)
            width_norm_min: 128, // auto-normalize at/above this fan-in — read only while stability_cap is 0
            stability_cap: 0.5, // 1.6 — cap the delta step at 0.5/(1+Σx²) (legacy 0 = the raw step)
            readout_rls: 0.0, // opt-in RLS readout (forgetting factor λ) — 0 = the delta rule, byte-identical
            rls_prior: default_rls_prior(), // 100 — the RLS readout's prior weight variance (ridge penalty 0.01)
            gain_fanin_norm: false,         // v2 capability knob — off = historical behavior
            var_gated_norm: false, // opt-in variance-masked width-norm — off = historical behavior
            activation: Activation::Tanh, // v2 genome knob — Tanh = historical behavior
            metabolic_cost: 0.5, // energy each `live` tick / metered dream bout drains from the metabolism
            action_dwell: false, // grown action-persistence axis; off = re-decide every step, byte-identical
            dwell_rate: 0.5, // how fast a repeated action's grown dwell duration moves with reward
            dwell_max: 32.0, // ceiling on a grown dwell duration (forced re-decide)
            nest_credit: false, // route composed error into a frozen nested sub's readout; off = frozen brick, byte-identical
            nest_credit_lr: 0.1, // step size of the one-hop composed-error delta into the nested sub
            dream_satiation: 0.0, // 0 = dreaming ungated/free (byte-identical); >0 = a fed-luxury with a hunger floor
            free_energy_actor: false, // deprecated alias of act_rule = FreeEnergy (wins over valence_actor)
            free_energy_sstar: default_free_energy_sstar(), // 0.9 — preferred observation s* (reward-as-prior)
            free_energy_prior_std: default_free_energy_prior_std(), // 1.0 — never-sampled option's epistemic σ0
            free_energy_ensemble: default_free_energy_ensemble(), // 24 — FreeEnergy's value estimators per option
            free_energy_lr: default_free_energy_lr(), // 0.15 — option-value step toward the fold target
            valence_actor: false, // deprecated alias of act_rule = Valence (the default rule since 1.6)
            act_rule: ActRule::Valence, // 1.6 default — reward-maximising valence (every genome; "ActScore" loads as it)
            act_discount: 0.0, // TD discount of the option-value fold — 0 = immediate reward, byte-identical
            pool_surprise: false, // opt-in: alive's pool pays by surprise reduction — off = the reliance economy
            surprise_currency: false, // Gap 7 spike — off = no per-cell variance tracked, byte-identical (learn path untouched)
            epistemic_weight: default_epistemic_weight(), // #155 — meta-evolved learning-progress weight in orient()
            goal_reward: default_goal_reward(), // 0.0 — intrinsic reward on reaching a self-set goal; off = byte-identical
            pos_dim_init: default_pos_dim_init(), // 2 — concept-map dimensionality floor (was the buried 2)
            metric_repel_softening: default_metric_repel_softening(), // 1.0 — metric_step repulsion softening
            probe_settle_pad: default_probe_settle_pad(), // 32 — fingerprint/embed settle padding (was +32)
            struct_walk_max_depth: default_struct_walk_max_depth(), // 64 — prime/tree read-algebra recursion guard
            nest_seeds_position: true, // #172 — nest seeds concept-map position from sub.embed() (behaviour key)
            heartbeat_k: default_heartbeat_k(), // 0.5 — autonomous-heartbeat dormancy ramp rate (D6/D7; live-only)
            heartbeat_theta: default_heartbeat_theta(), // 1.0 — autonomous-heartbeat fire threshold (D6/D7; live-only)
        }
    }
}

/// How a [`crate::Field`] snapshot reads its genome. A self-describing snapshot (JSON) written before 1.6 has no key
/// for the levers 1.6 added; each absent one takes its [`Config::legacy`] value, so the old organism keeps its old
/// genome instead of a mix of stored values and 1.6 defaults (review of #494, core #2). A key that IS present keeps
/// its value, and a config absent altogether is `Config::legacy()`. A binary snapshot (bincode) carries every field,
/// so it is read as is. (A standalone partial `Config` JSON still fills absent keys from `Config::default()`.)
pub(crate) fn snapshot_config<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Config, D::Error> {
    #[derive(Deserialize)]
    struct Snapshot {
        stability_cap: Option<f32>,
        fanin_init_gain: Option<f32>,
        reliance_norm: Option<RelianceNorm>,
        grow_within_budget: Option<bool>,
        #[serde(flatten)]
        rest: Config,
    }
    if !d.is_human_readable() {
        return Config::deserialize(d);
    }
    let s = Snapshot::deserialize(d)?;
    let l = Config::legacy();
    Ok(Config {
        stability_cap: s.stability_cap.unwrap_or(l.stability_cap),
        fanin_init_gain: s.fanin_init_gain.unwrap_or(l.fanin_init_gain),
        reliance_norm: s.reliance_norm.unwrap_or(l.reliance_norm),
        grow_within_budget: s.grow_within_budget.unwrap_or(l.grow_within_budget),
        ..s.rest
    })
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    /// The schema must stay IN SYNC with `Config`: every serialized field appears exactly once, by name.
    #[test]
    fn schema_covers_every_config_field() {
        let value = serde_json::to_value(Config::default()).unwrap();
        // The schema is the f32 (Optuna) search space. Non-numeric fields — boolean MODE flags (`lr_fanin_norm`,
        // `gain_fanin_norm`) and the `activation` enum — are categorical genome choices, not scalar levers to
        // sweep, so they are intentionally NOT in the f32 schema. Check every NUMERIC field is covered.
        let fields: alloc::collections::BTreeSet<String> = value
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, v)| v.is_number())
            .map(|(k, _)| k.clone())
            .collect();
        let levers: alloc::collections::BTreeSet<String> = Config::schema()
            .iter()
            .map(|l| l.name.to_string())
            .collect();
        assert_eq!(
            fields, levers,
            "Config::schema() is out of sync with the Config fields (a field was added/removed without updating the schema)"
        );
        // defaults in the schema match Config::default()
        let d = Config::default();
        let dv = serde_json::to_value(&d).unwrap();
        for l in Config::schema() {
            let got = dv[l.name].as_f64().unwrap() as f32;
            assert!(
                (got - l.default).abs() < 1e-9
                    || (got - l.default).abs() / got.abs().max(1.0) < 1e-6,
                "schema default for {} ({}) != Config::default() ({got})",
                l.name,
                l.default
            );
            assert!(l.range.0 <= l.range.1, "{} range is inverted", l.name);
        }
    }
}
