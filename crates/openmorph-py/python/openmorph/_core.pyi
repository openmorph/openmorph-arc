# SPDX-License-Identifier: MIT-0
# Type stubs for the `openmorph` extension module (IDE / type-checker support).
# Kept in lockstep with `crates/openmorph-py/src/lib.rs` by `scripts/check_py_binding_coverage.py` (CI).
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Vector arguments accept a list/tuple OR a 1-D numpy array; an array crosses into Rust in one call (no per-element
# conversion) and gives bit-identical results to passing `arr.tolist()`.
_FloatVec = Union[Sequence[float], np.ndarray]
_IndexVec = Union[Sequence[int], np.ndarray]

__version__: str

class Morph:
    """A digital organism — an async membrane of cells that fires on change, settles, learns, and grows."""

    def __init__(self, n_cells: int) -> None:
        """A Morph with `n_cells` cells (all 0) and no units."""

    def configure(self, json: str, reset: bool = False) -> None:
        """Configure the tunable substrate constants from a JSON OBJECT string (see docs/config-reference). The
        patch merges onto the CURRENT config, so successive calls accumulate; `reset=True` merges onto the
        historical defaults instead (what `apply_preset` does). A non-object patch or an unknown key raises
        ValueError and changes nothing."""

    def config_json(self) -> str:
        """The current config as a JSON object string (every lever, including untouched defaults)."""

    def rng_seed(self, seed: int) -> None:
        """Seed the deterministic growth PRNG (for byte-reproducible runs)."""

    def stimulate(self, idx: _IndexVec, val: _FloatVec, budget: int) -> int:
        """Perturb the given cells and settle within `budget` events. Returns events processed. ValueError on a
        length mismatch or a cell outside the membrane."""

    def cells(self) -> List[float]:
        """The current membrane values."""

    def cell(self, cell: int) -> float:
        """The value of one cell, without copying the membrane (ValueError if out of range)."""

    def settled(self) -> bool:
        """Whether the last cascade reached a fixed point within budget."""

    def n_units(self) -> int:
        """Number of grown units (organs)."""

    def truncated_settles(self) -> int:
        """How many settles stopped at their `budget` with firings still pending (a truncated, not-settled
        prediction). Compare before/after a call to detect one; cumulative in memory (0 after `load`)."""

    def learn_step(
        self, idx: _IndexVec, val: _FloatVec, output_cell: int, target: float, lr: float, budget: int
    ) -> float:
        """One local delta-learning event toward `target`. Returns the error before the update. ValueError on a
        length mismatch or an input/output cell outside the membrane."""

    def grow(
        self, idx: _IndexVec, val: _FloatVec, target_cell: int, target: float, lr: float, budget: int
    ) -> float:
        """Grow the transducer toward `target` at `target_cell` and tune it. Returns the error before. A
        `target_cell` past the membrane grows it, and so does an input cell while `target_cell` has no writer yet;
        once it has one, an input cell past the membrane raises ValueError (the core would drop it)."""

    def observe(
        self, obs: int, pred: int, dim: int, values: _FloatVec, lr: float, budget: int
    ) -> List[float]:
        """Target-free predictive coding: correct the standing prediction; return per-cell error. `values` must
        hold exactly `dim` entries."""

    def grow_backbone(self, inputs: List[int], width: int, scale: float) -> List[int]:
        """A fixed-width shared random-feature basis of `inputs` (`width` features at `scale`). Returns the new
        feature cell ids."""

    def add_unit(self, reads: List[int], weights: List[float], bias: float, write: int) -> int:
        """A linear readout unit writing `write` from `reads`; returns the unit id. A cell past the membrane grows
        it; `weights` needs one entry per read (ValueError otherwise)."""

    def encapsulate(self, cells: List[int]) -> "Morph":
        """Bank the subgraph writing `cells` as a fresh standalone Morph (a macro-action / skill). Source unchanged."""

    def nest(self, sub: "Morph", reads: List[int], write: int, budget: int) -> int:
        """Plant a banked skill as ONE cell fired in the cascade (the composition operator). `sub` is cloned."""

    def nest_multi(self, sub: "Morph", reads: List[int], writes: List[int], budget: int) -> int:
        """Plant a banked skill exposing its full vector output: sink i -> writes[i] (writes[0] is the primary)."""

    def bank_skill(self, output_cell: int, budget: int) -> int:
        """Encapsulate the backbone of `output_cell` and nest it as a frozen reusable node; returns the banked cell."""

    def nested_settles(self) -> int:
        """Cumulative nested sub-settles — a memo cache-miss / behaviour-novelty meter."""

    def think(self, state_cells: List[int], pred_cells: List[int], steps: int, noise: float, budget: int) -> List[float]:
        """Free-run the recurrent model: feed predicted state back as state, roll `steps` transitions, return the
        simulated consequence (internal thinking / imagination — the generative complement of the heartbeat)."""

    def dream(self, state_cells: List[int], pred_cells: List[int], longrange: List[int], seeds: List[List[float]], think_steps: int, lr: float, bouts: int, budget: int) -> Tuple[int, int, int, int, bool, float]:
        """The autonomous discovery loop as a substrate faculty: think -> consolidate -> sleep, until rest.
        Returns a 6-tuple, in THIS order:
            0. ticks         (int)   — internal maintenance ticks spent
            1. reactivations (int)   — dormant memories reactivated / replayed
            2. consolidated  (int)   — matured, relied-upon cells frozen this sleep
            3. culled        (int)   — starved cells pruned this sleep
            4. rested        (bool)  — True once nothing remained pending (the emergent stop)
            5. hunger        (float) — the organism's hunger at the end of the bout
        Set config.dream_satiation > 0 (via configure) to gate dreaming on satiation: the organism only
        dreams when fed, each bout burns metabolic_cost, so dreaming self-limits (a wake/conserve/dream cycle)."""

    def satiation(self) -> float:
        """The organism's energy account in [0,1]: metabolism / energy_cap (fed -> 1, starving -> 0)."""

    def should_dream(self) -> bool:
        """Whether satiation has reached config.dream_satiation — i.e. the organism is fed enough to dream."""

    def snapshot(self) -> bytes:
        """Snapshot the whole organism to bytes (JSON) — persistence."""

    @staticmethod
    def load(bytes: Union[bytes, bytearray, Sequence[int]]) -> "Morph":
        """Load an organism from a snapshot produced by `snapshot()` (bytes/bytearray, or the legacy list[int]
        older releases returned). ValueError on malformed data."""

    def act(self, options: List[int], last: Optional[Tuple[int, float]] = None) -> Optional[int]:
        """One native policy step — returns the chosen option cell. `act([], last)` ends an episode: it folds the
        final reward with no `act_discount` bootstrap and returns None (ValueError on `act([])` with no `last`). `last` =
        (previous option, its reward). The rule is config `act_rule`: "Valence" (the DEFAULT since 1.6 — argmax
        -value + mu + sigma^2 over each option's reward estimate mu and exact visit count n, sigma^2 =
        (sigma0 * (1 - lr)^n)^2 shrinking with every visit; exact ties are broken by the seeded act PRNG; for
        rewards of scale c set free_energy_prior_std ~ sqrt(c)), "FreeEnergy" (argmin G = value + (mu - s*)^2 -
        sigma^2, a set-point matcher). The pre-1.6 "ActScore" rule is removed: a config naming it loads as
        "Valence", and its keys `openmindedness`, `hypothesis_loop`, `drive_decay` and `metabolic_actor` are unknown
        to `configure` (ValueError). The deprecated bools `free_energy_actor` / `valence_actor` still select their
        rule. `act_discount` (default
        0 = immediate reward) folds r + act_discount * max mu(options) — set 0.9-0.99 for a multi-step task."""

    def g_spread(self) -> float:
        """Std-dev of the active rule's decision score across the last act()'s options (~0 = a degenerate, tied
        decision; NaN before the first act)."""

    def select_explore(self, options: List[List[int]], beta: float) -> int:
        """Value-of-information choice: the option maximising mean + beta*spread over its estimator cells. Returns
        the option index (ValueError on an empty `options`)."""

    def reliance_of(self, cell: int) -> float:
        """How much the grown readouts rely on `cell` (sum of readers' |weight|) — grown reward attribution."""

    def position(self, cell: int) -> List[float]:
        """The grown concept-map position of `cell` (ValueError if out of range)."""

    def distance(self, a: int, b: int, dims: int) -> float:
        """Concept-map distance between two cells over the first `dims` axes (0 = all; ValueError if out of range)."""

    def seed_coord(self, cell: int, percept: _FloatVec) -> None:
        """Place `cell` on the concept-map from a percept vector (a repeating state -> a repeating key)."""

    def metric_step(self, positive: List[int], rate: float, n_neg: int) -> None:
        """One contrastive step: pull `positive` together, push `n_neg` random negatives apart, at `rate`."""

    def co_activate(self, cells: List[int], rate: float) -> None:
        """Hebbian map shaping: pull the positions of co-active `cells` toward their centroid at `rate`."""

    def spread_positions(self) -> None:
        """Per-axis standardize the concept-map so it fills the space (counter-force to co_activate)."""

    def cluster(self, ids: List[int], k: int) -> List[List[int]]:
        """Farthest-point clustering of `ids` into `k` groups on the grown concept-map (sub-morph boundaries)."""

    def grown_cells(self) -> List[int]:
        """The cells written by a grown unit, in cell order (the set to feed `cluster`)."""

    def group_by_surprise(self, cells: List[int], rate: float) -> None:
        """Causal map shaping: pull pairs together by min surprise-reduction (no-op unless surprise_currency)."""

    def surprise_reduction_of(self, cell: int) -> float:
        """Per-cell end-to-end ablation importance for the warmed output (0.0 unless surprise_currency)."""

    def accrue_co_credit(self, cells: List[int], err: float, decay: float) -> None:
        """Accrue the second-order co-credit EMA of err*x_i*x_j for every pair in `cells`."""

    def co_credit_step(self, rate: float) -> None:
        """Shape the concept-map by accrued synergy (above-mean pairs attract, below-mean repel)."""

    def co_credit_of(self, a: int, b: int) -> float:
        """The accrued co-credit of a cell pair (order-insensitive; 0.0 if unseen)."""

    def prefer(self, cell: int, value: float) -> None:
        """Clamp `cell`'s cost-to-go at `value` (a goal is `prefer(cell, 0.0)`)."""

    def orient(self, candidates: List[int]) -> Optional[int]:
        """Self-generate a goal: clamp the candidate maximising drive + epistemic_weight·learning_progress (#155)."""

    def orient_many(self, candidates: List[int], k: int) -> List[int]:
        """Self-generate the top-k goals at once and clamp them as simultaneous subgoals (#155)."""

    def goals_reached(self) -> int:
        """Cumulative self-set goals reached (0 when goal_reward is off)."""

    def relax_values(self, adjacency: List[List[int]], cost: float, max_sweeps: int) -> int:
        """Settle the cost-to-go value field over `adjacency` (0 sweeps = to convergence). Returns footprint."""

    def value_of(self, cell: int) -> float:
        """The graded cost-to-go value of a cell (f32::MAX ~= 3.4028235e38 = unknown/unreachable; not inf)."""

    def clear_values(self) -> None:
        """Clear every clamp; all values unknown — start a fresh goal."""

    def reinforce(self, cell: int, amount: float) -> None:
        """Accumulate graded action-drive on `cell` (+ commit / − aversion); seeds `orient`'s extrinsic term."""

    def drive_of(self, cell: int) -> float:
        """The accumulated action-drive of a cell (0.0 if never reinforced)."""

class RelativeCode:
    """Grown object-relative, reward-bisimilar discrete state code + effect-action relabelling (#350)."""

    def __init__(self, width: int, height: int, background: int, novelty: float) -> None:
        """A relative code for width×height integer-colour frames; `novelty` mints a new discrete code."""

    def origin_color(self) -> Optional[int]:
        """The grown egocentric origin colour (the controllable object), if learned/seeded yet."""

    def n_states(self) -> int:
        """The number of grown discrete states."""

    def freeze(self) -> None:
        """Freeze the prototype set (pure evaluation — grow none)."""

    def learn_controllable(
        self, transitions: List[Tuple[List[int], int, List[int]]], n_actions: int
    ) -> Optional[int]:
        """Grow the egocentric origin (agent) from an action/displacement stream. Returns the origin colour."""

    def seed_origin_singular(self, frame: _IndexVec) -> Optional[int]:
        """Seed the origin without action history (the rarest-colour object). Returns the origin colour."""

    def learn_action_effects(
        self, transitions: List[Tuple[List[int], int, List[int]]], n_actions: int
    ) -> List[Optional[Tuple[float, float]]]:
        """Effect-action table: the grown agent's mean self-displacement (dx, dy) per action id (#350)."""

    def relative_features(self, frame: _IndexVec) -> List[float]:
        """Position-relative feature vector (translation-invariant): [colour, dx, dy] per object vs the origin."""

    def bag_features(self, frame: _IndexVec) -> List[float]:
        """Position-invariant object-bag features (identity + counts) — the failing control baseline."""

    def quantize(self, feat: _FloatVec) -> int:
        """Quantize a feature vector to a discrete code (nearest prototype within novelty, else a new code)."""

    def encode(self, frame: _IndexVec) -> int:
        """Encode a frame to a discrete object-relative state code (quantize ∘ relative_features)."""

    def snapshot(self) -> bytes:
        """Serialize the encoder (learned self-model + prototypes) to bytes. Round-trips with load()."""

    @staticmethod
    def load(bytes: bytes) -> "RelativeCode":
        """Restore an encoder from snapshot() bytes."""

def greedy_effect_action(
    to_goal: Tuple[float, float], effects: List[Optional[Tuple[float, float]]]
) -> Optional[int]:
    """Control-independent policy step: argmin_a ‖to_goal − effect_a‖ over the effect table (#350)."""

def effect_graph(
    n_actions: int,
    categories: List[int],
    observations: List[Tuple[int, int, str, float]],
    threshold: float,
) -> Tuple[int, List[int], List[Tuple[int, int, int]]]:
    """Rule-aware action->category effect graph (level-invariant). Returns (n, node_labels, edges) for StructuralKey.embed."""

def role_effect_graph(
    n_actions: int,
    observations: List[Tuple[int, int, str, float]],
    threshold: float,
) -> Tuple[int, List[int], List[Tuple[int, int, int]]]:
    """Palette-invariant (cross-game) effect graph: category nodes anonymised, so WL derives each one's ROLE from its
    effect edges. Same mechanic under a different palette -> same signature. No `categories` arg (inferred from
    observations); inert colours dropped. Returns (n, node_labels, edges) for StructuralKey.embed."""

def category_roles(
    categories: List[int],
    observations: List[Tuple[int, int, str, float]],
    threshold: float,
) -> List[Tuple[int, int]]:
    """Palette-invariant per-category functional ROLE labels (mobile=agent, vanishing=collectible, inert=wall) from
    the effect wiring — the per-category twin of role_effect_graph, for recoding the STATE so a warm-started value
    head transfers across palettes. Returns (category, role) pairs sorted by category."""

def role_of_effects(effects: List[str]) -> int:
    """The canonical ROLE hash for a SET of effect strings — the same hash category_roles assigns a category that is
    a target of exactly these effects. Use it to NAME a role to match against category_roles output:
    role_of_effects(["vanished"]) is the collectible role, ["moved"] the agent role. Empty ⇒ the static role."""

def is_static_role(role: int) -> bool:
    """Whether a role hash is the STATIC role (a category no action affects — a wall/background), so a consumer can
    skip inert loci in navigation without hard-coding a colour."""

def segment_objects(
    grid: _IndexVec, width: int, height: int, background: int
) -> List[Tuple[int, int, float, float, int, int, List[int]]]:
    """Connected-component segmentation of a flat integer-colour grid (4-connectivity, background excluded).
    Returns one record per object: (colour, size, centroid_x, centroid_y, bbox_w, bbox_h, cells). The centroid is
    the precise-click target; cells are flat row*width+col indices. For the grown twin, see GrownSegmenter."""

def aggregate(values: _FloatVec) -> Tuple[float, float, float, float, float, float]:
    """Reduce a value list to fixed scalar stats. Returns a 6-float tuple, in THIS order:
        0. count  — number of values
        1. sum    — Σ values
        2. min    — minimum value
        3. max    — maximum value
        4. mean   — arithmetic mean (the scalar-field / thermostat state is aggregate(frame)[4])
        5. spread — dispersion / value-of-information moment
    So a variable-cardinality scene meets the fixed-width substrate."""

def relate(
    grid: _IndexVec, width: int, height: int, background: int
) -> List[Tuple[int, int, float, float, float, bool, bool]]:
    """Pairwise object relations over segment_objects(grid,...): (a, b, dx, dy, size_ratio, same_colour, adjacent),
    where a/b index the segment_objects order. Raw relational structure, not a task."""

class GrownSegmenter:
    """Grown (non-authored) segmenter — segment_objects' substrate-aligned twin (grown colour-equality gate)."""

    def __init__(self) -> None: ...
    def segment(
        self, grid: _IndexVec, width: int, height: int, background: int
    ) -> List[Tuple[int, int, float, float, int, int, List[int]]]:
        """Segment a grid; same record shape as segment_objects."""

    def gate_accuracy(self) -> float:
        """The grown colour-equality gate's current accuracy (diagnostic)."""


class ImaginationPlanner:
    """Plan multi-step IN IMAGINATION over a reward-free learned forward model, so the agent sequences a combo-lock
    (actuate gate -> THEN collect) before any reward. observe() learns the model reward-free; plan(current, goal,
    horizon) rolls it forward and relaxes cost-to-go from the goal, returning the first (precondition) action.
    Cracks combo-locked floor games a single-step reactive actor can't."""

    def __init__(self, n_actions: int) -> None: ...
    def observe(self, state: int, action: int, next: int) -> None:
        """Learn a reward-free state -(action)-> next transition."""
    def imagine_next(self, state: int, action: int) -> Optional[int]:
        """The model's predicted next-state for (state, action), or None if never observed."""
    def plan(self, current: int, goal: int, horizon: int) -> Optional[int]:
        """First (precondition) action of the least-cost imagined path to goal, or None if not imagined-reachable."""
    def plan_seq(self, current: int, goal: int, horizon: int) -> Optional[List[int]]:
        """The whole imagined action sequence to goal (plan_seq(..)[0] == plan(..)), or None."""
    def edge_reliability(self, state: int, action: int) -> float:
        """Confirmations / (confirmations + contradictions) for (state, action) in [0,1]; 0.0 if unobserved."""
    def plan_confident(self, current: int, goal: int, min_reliability: float) -> Optional[int]:
        """First action of the least-cost path over edges trusted at `min_reliability`, or None."""
    def plan_seq_confident(self, current: int, goal: int, min_reliability: float) -> Optional[List[int]]:
        """The whole-sequence twin of plan_confident."""
    def n_edges(self) -> int: ...
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(data: bytes) -> "ImaginationPlanner": ...
class Agent:
    """The perception path: RelativeCode state -> FrontierExplorer narrows the options -> the core Valence act rule
    decides and learns. A colour frame in, a move out; one snapshot = the whole agent. The frontier carries a learned
    action model (inverse / commutation counts); on the in-repo ARC bench the agent scores +0.527 lvl-norm, +54 net vs
    the floor, beating the tabular FrontierExplorer (+0.383 / +42; config='{"predict": false}' restores it)."""

    def __init__(
        self,
        width: int,
        height: int,
        seed: int,
        config: Optional[str] = None,
        carry: Optional[str] = None,
    ) -> None:
        """`carry`: what earlier games taught (the JSON string `carry()` returned by the last game's agent)."""
    def carry(self) -> str:
        """What this agent carries to its next game (JSON): the self's colour and each action's direction."""
    def act(self, frame: _IndexVec, simple: List[int], click: bool) -> Tuple[Any, ...]:
        """("simple", k), ("click", x, y) or ("reset",) (restart the level)."""
    def between(self, frame: _IndexVec) -> None:
        """An intermediate frame the last move returned before its final one (an animation): call it for each, in
        order, before `observe` with the final frame. Only the click model reads it."""
    def observe(self, next: _IndexVec, reward: float, done: bool) -> None: ...
    def stats(self) -> Dict[str, int]: ...
    def state(self) -> Optional[int]: ...
    def reseed(self, seed: int) -> None: ...
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(data: bytes) -> "Agent": ...

class FrontierExplorer:
    """Directed reward-free exploration that first-traverses a multi-step combo random never reaches. observe()
    learns the model + marks (config,action) tried; explore_action(current) returns an untried action here else the
    first action toward the nearest config with an untried action (the behaviour frontier). Pairs with
    ImaginationPlanner over the same reward-free observe stream."""

    def __init__(self, n_actions: int) -> None: ...
    def with_prediction(self) -> "FrontierExplorer":
        """Turn on the learned action model (learned inverse/commutation counts demote predictable untried pairs)."""
    def close(self, config: int, keys: List[int]) -> None:
        """Close a terminal state: its keys become tried self-loops, without teaching the learned model."""
    def predicts(self) -> bool: ...
    def predicted(self, config: int, key: int) -> Optional[int]: ...
    def observe(self, config: int, action: int, next: int) -> None: ...
    def explore_action(self, current: int) -> Optional[int]: ...
    def explore_among(self, current: int, candidates: List[int]) -> Optional[int]: ...
    def frontier_exhausted(self, current: int) -> bool: ...
    def n_edges(self) -> int: ...
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(data: bytes) -> "FrontierExplorer": ...

class Player:
    """A whole trained ARC organism as ONE persistable bundle: perceive (encode) + discrete free-energy act +
    grounded click, with a single snapshot()/load() = one .morph. Honest: the discrete actor is flat free-energy —
    it learns one-step decisions (click), not sparse multi-step paths."""

    def __init__(self, width: int, height: int, background: int, novelty: float, ensemble: int, beta: float, seed: int, lr: float) -> None: ...
    def encode(self, frame: _IndexVec) -> int:
        """Perceive: raw frame -> discrete object-relative state code."""
    def learn_controllable(self, transitions: List[Tuple[List[int], int, List[int]]], n_actions: int) -> Optional[int]:
        """Learn the controllable self object from (frame, action, next_frame) transitions."""
    def act(self, options: List[int], last: Optional[Tuple[int, float]] = None) -> int:
        """Act-discrete: one free-energy choice over state x action option cells (ValueError if empty)."""
    def g_spread(self) -> float:
        """Spread of G across the last act()'s options (~0 = a degenerate decision; NaN before the first act)."""
    def origin_color(self) -> Optional[int]:
        """The grown egocentric origin colour (the controllable self) once learned, else None."""
    def click(self, frame: _IndexVec, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Act-click: grounded (x, y) from the organism's own segmentation, or None."""
    def learn_click(self, reward: float) -> None:
        """Credit the last click (negative -> avoid that locus)."""
    def set_preferred(self, sstar: float) -> None: ...
    def set_background(self, background: int) -> None:
        """Adapt the click head to a new game's background without resetting its learned policy."""
    def learn_effects(self, transitions: List[Tuple[List[int], int, List[int]]], n_actions: int) -> None:
        """Navigate-to-goal: grow the controllable self + per-action effect table from a transition stream (per-game)."""
    def objects(self, frame: _IndexVec) -> List[List[float]]:
        """Per-object egocentric features [colour, dx, dy] (the goal candidates); index feeds navigate/learn_goal."""
    def navigate(self, frame: _IndexVec) -> Optional[int]:
        """Native navigate-to-goal: perceive -> infer which object is the goal -> action stepping the agent toward it."""
    def learn_goal(self, frame: _IndexVec, reached: int, reward: float) -> None:
        """Credit reward to the reached object's relational uniqueness; the signature transfers across games."""
    def goal_credit(self, dim: int) -> float:
        """Learned goal attribution on uniqueness dim (0=colour, 1=dx, 2=dy) — the transferable signal."""
    def learn_hazard(self, frame: _IndexVec, action: int, died: bool) -> None:
        """Learn from an OVER event: grow the per-(state, action) death signal (died=True on the OVER transition)."""
    def hazard_of(self, frame: _IndexVec, action: int) -> float:
        """The learned death risk of action in frame ([0,1], 0.0 if unseen)."""
    def navigate_safe(self, frame: _IndexVec, hazard_bar: float) -> Optional[int]:
        """Hazard-aware navigate: goal-descent among low-hazard actions (detour) — reach the goal without dying."""
    def navigate_retry(self, frame: _IndexVec, patience: int) -> Optional[int]:
        """Navigate with retry: blacklist a candidate not reached in `patience` steps (transient, no negative credit) and try the next-best."""
    def reset_navigation(self) -> None:
        """Clear the transient navigate-retry blacklist + counter (call on reach / level-up / new game)."""
    def snapshot(self) -> bytes:
        """Snapshot the WHOLE player (policy + perception + click + navigate signature + hazard) — one .morph."""
    @staticmethod
    def load(data: bytes) -> "Player":
        """Restore a whole player from snapshot() bytes — every faculty's trained state preserved."""
    def reseed(self, seed: int) -> None:
        """Reseed the exploration streams (click tie-break + policy PRNG) without touching learned faculties —
        call right after load() with a per-run seed."""

class RelationalGoal:
    """Relational goal-inference: which object is the goal when the rule is relational (the odd one out / unlike the
    others), not a fixed per-object feature. Attributes reward to per-dimension UNIQUENESS over the object set, so it
    learns e.g. 'unique-colour target' even when the colour varies per game. Objects are grown feature vectors."""

    def __init__(self, lr: float) -> None: ...
    def learn(self, objects: List[List[float]], reached: int, reward: float) -> None:
        """Credit reward to the uniqueness dimensions of the reached object."""
    def infer_goal(self, objects: List[List[float]]) -> Optional[int]:
        """The object whose uniqueness pattern best matches the learned attribution, or None."""
    def credit_of(self, dim: int) -> float:
        """The learned attribution to being unique on dimension dim."""
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(data: bytes) -> "RelationalGoal": ...

class RuleInduction:
    """Induce an abstract rule over grown structural predicates (aggregate count>=k, relate adjacent), test it on a
    held-out split, and abstain if it doesn't generalise. induce(examples, epochs, lr, budget) grows a predictor
    (deep enough for a non-linear rule like XOR) on the fit split and returns held-out accuracy; confident() is True
    iff it generalises. depth=0 is a linear control that can only induce linearly-separable rules."""

    def __init__(self, n_features: int, bar: float, depth: int, seed: int) -> None: ...
    def induce(self, examples: List[Tuple[List[float], float]], epochs: int, lr: float, budget: int) -> float:
        """Hypothesize+test: grow a rule on the fit split, score held-out (labels are +/-1). Returns held-out accuracy."""
    def confident(self) -> bool:
        """Whether the induced rule generalised (held-out >= bar). If False, apply() is an ABSTENTION."""
    def held_out_accuracy(self) -> float: ...
    def apply(self, features: _FloatVec, budget: int) -> float:
        """Predicted label for new structural predicates (sign = class). Meaningful only when confident()."""
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(data: bytes) -> "RuleInduction": ...

class StructuralGoal:
    """Infer a transform game's goal from regularity alone (zero reward/positive): the min-description-length
    reachable config, over generic codecs (uniformity/runs/symmetry). canonical_targets(current_grid) is the
    target_grid<-regularity(current) seam a ConstraintField (GF(2)/Lights-Out) consumer feeds. Works only for
    impose-order games; on an arbitrary target min-DL picks the regular config and fails."""

    def __init__(self, width: int, height: int) -> None: ...
    def description_length(self, grid: List[int]) -> float: ...
    def canonical_targets(self, current: List[int]) -> List[List[int]]: ...
    def infer_goal(self, configs: List[List[int]]) -> Optional[int]: ...

class ProgressReadout:
    """Reward-free structured-state delta between frames, consumed by an abduction loop as a LIKELIHOOD over
    goal-hypotheses (never a reward). observe(frame) -> (magnitude, palette_delta, region_deltas): magnitude =
    cells changed (frame Hamming); palette_delta = [(colour, delta_count)]; region_deltas = [(cx, cy, colour,
    delta_size, color_changed, appeared, disappeared)] per grown region (a depleting meter = negative delta_size
    at a stable centroid). Grown, not authored (segment_objects + histogram + diff); no meter hardcoded."""

    def __init__(self, width: int, height: int, background: int, match_radius: float) -> None: ...
    def observe(
        self, frame: _IndexVec
    ) -> Tuple[int, List[Tuple[int, int]], List[Tuple[float, float, int, int, bool, bool, bool]]]: ...
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(data: bytes) -> "ProgressReadout": ...

class DisplayedTargetMatcher:
    """Route A for construction games that DRAW their goal (verified eg01: an isolated 6x6 icon, colour-5 frame
    enclosing a colour-4 interior, on a colour-2 canvas). detect(frame) reads the shown icon → (frame_colour,
    interior_colour, (x,y,w,h), shape_mask, side) or None (ABSTAIN); shape_similarity(cells) scores a play-area
    candidate by COLOUR-INVARIANT shape. Match over shape, never colour; propose a target, the loop confirms by
    lvl++ or abstains."""

    def __init__(self, width: int, height: int, side: int, max_area_frac: float) -> None: ...
    def detect(
        self, frame: _IndexVec
    ) -> Optional[Tuple[int, int, Tuple[int, int, int, int], List[bool], int]]: ...
    def shape_similarity(self, candidate_cells: List[int]) -> float: ...

class SpatialActor:
    """Grown spatial action output: act(frame,w,h) segments the frame into the organism's own objects, scores each
    locus by its grown value ensemble (free-energy G), and returns the chosen object's centroid as a grounded
    (x,y) click. learn(reward) credits the last click (negative -> avoid that locus). The click tracks a moving
    target (grounded, not memorised) and absorbs the hand-built Python click into the substrate."""

    def __init__(self, ensemble: int, beta: float, seed: int, background: int, lr: float) -> None: ...
    def snapshot(self) -> bytes:
        """Serialize the trained state to bytes (JSON) so it survives save/load. Round-trips with load()."""
    @staticmethod
    def load(data: bytes) -> "SpatialActor":
        """Restore from snapshot() bytes — the learned state is preserved, not re-initialised."""
    def set_preferred(self, sstar: float) -> None:
        """Set the preferred outcome s* (the reward the actor prefers to reach)."""
    def set_background(self, background: int) -> None:
        """Adapt the click head to a new game background without resetting the learned policy."""
    def act(self, frame: _IndexVec, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Perceive -> select -> emit: returns the grounded (x, y) = (col, row) click, or None if no object."""
    def act_with_change(self, prev: _IndexVec, frame: _IndexVec, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Like act(), but adds the centroids of regions that CHANGED since prev as extra loci — so an
        off-object click (a background cell just activated/cleared) becomes selectable. Same learn path."""
    def learn(self, reward: float) -> None:
        """Credit the last emitted click (negative reward -> avoid that kind of locus)."""
    def last_click(self) -> Optional[Tuple[int, int]]:
        """The last emitted (x, y), if any."""


class Codebook:
    """Match-or-grow vocabulary: nearest prototype within novelty (match) else grow. The alphabet of a scale."""
    def __init__(self, novelty: float, pull: float) -> None: ...
    def snapshot(self) -> bytes:
        """Serialize the trained state to bytes (JSON) so it survives save/load. Round-trips with load()."""
    @staticmethod
    def load(data: bytes) -> "Codebook":
        """Restore from snapshot() bytes — the learned state is preserved, not re-initialised."""
    def assign(self, sig: _FloatVec) -> int: ...
    def vocab_size(self) -> int: ...
    def fires(self, i: int) -> int: ...
    def freeze(self) -> None: ...

class LivingLoop:
    """Autonomous internal life: think (free-run model) -> crystallize (bank recurring) -> consolidate -> live
    (heartbeat maintenance), until the organism rests. Composes the three faculties over a Morph."""
    def __init__(self, scale: int, novelty: float, pull: float, think_steps: int, budget: int) -> None: ...
    def run(self, host: "Morph", state_cells: List[int], pred_cells: List[int], longrange: List[int], seeds: List[List[float]], lr: float, bouts: int) -> Tuple[int, int, int, int, int, bool]: ...
    def vocab_size(self) -> int: ...
    def snapshot(self) -> bytes:
        """Serialize the loop's banked skill vocabulary to bytes. Round-trips with load()."""
    @staticmethod
    def load(data: bytes) -> "LivingLoop": ...

class Crystallizer:
    """Fractal compress-and-reuse operator: assign a segment signature to the scale vocabulary, then compose the
    segment's grown cells up as a reusable nested cell (crystallize_up = encapsulate + nest)."""
    def __init__(self, scale: int, novelty: float, pull: float) -> None: ...
    def snapshot(self) -> bytes:
        """Serialize the trained state to bytes (JSON) so it survives save/load. Round-trips with load()."""
    @staticmethod
    def load(data: bytes) -> "Crystallizer":
        """Restore from snapshot() bytes — the learned state is preserved, not re-initialised."""
    def assign(self, signature: _FloatVec) -> int: ...
    def crystallize_up(self, host: "Morph", seg_cells: List[int], signature: _FloatVec, reads: List[int], write: int, budget: int) -> Tuple[int, int]: ...
    def vocab_size(self) -> int: ...
    def scale(self) -> int: ...
    def freeze(self) -> None: ...
class PredictiveLatent:
    """Reward-free predictive encoder: self-supervised next-frame prediction extracts a stable per-state latent
    from a signal buried in noise (feature_lr>0 learns the encoder; 0 = random projection). Feeds the Crystallizer."""
    def __init__(self, frame_dim: int, latent_width: int, feature_lr: float, scale: float, seed: int) -> None: ...
    def train(self, frame_now: _FloatVec, frame_next: _FloatVec, lr: float, budget: int) -> None: ...
    def train_contrastive(
        self, frame_now: _FloatVec, frame_next: _FloatVec, lr: float, budget: int
    ) -> None:
        """Opt-in CONTRASTIVE step: separate consecutive DIFFERING states in latent regardless of pixel magnitude,
        so a subtle recolour/toggle/cover is encoded as strongly as a salient move (which next-frame MSE ignores).
        Feed action transitions; pair with train() when the stream is noisy."""
    def encode(self, frame: _FloatVec, budget: int) -> List[float]: ...
    def freeze(self) -> None: ...
    def latent_width(self) -> int: ...
    def snapshot(self) -> bytes:
        """Serialize the trained encoder to bytes so a curriculum-trained latent persists (train once, ship it)."""
    @staticmethod
    def load(data: bytes) -> "PredictiveLatent": ...

def skill_signature(frames: List[List[int]], actions: List[int], width: int, height: int, background: int, n_actions: int, ends_reward: bool, ends_levelup: bool) -> List[float]:
    """Self-relative skill canonicaliser: discovers the controllable object (RelativeCode.learn_controllable) then
    keys on its goal-relation (converge/overlap/toggle, board-normalised) — nuisance-invariant. Feed to Crystallizer.assign."""

class ConstraintField:
    """Constraint-satisfaction as annealed substrate relaxation (colouring / exact-cover / SAT / GF(2)). Declare
    categorical variables + constraints; solve settles to a zero-violation attractor. No authored DFS/GF(2)."""

    def __init__(self) -> None: ...
    def var(self, domain: int) -> int:
        """Declare a categorical variable with `domain` values (>=2). Returns its id."""

    def bit(self) -> int:
        """Declare a binary variable (value 1 = true). Returns its id."""

    def not_equal(self, a: int, b: int) -> None:
        """Two same-domain variables must differ (colouring / cut / all-different pair)."""

    def equal(self, a: int, b: int) -> None:
        """Two same-domain variables must match."""

    def cardinality(self, literals: List[Tuple[int, int]], n: int) -> None:
        """Exactly n of the (var, value) literals hold — one-hot / exact-cover / tiling / degree / count."""

    def exactly_one(self, literals: List[Tuple[int, int]]) -> None:
        """Exactly one of the literals holds."""

    def clause(self, literals: List[Tuple[int, int]]) -> None:
        """At least one of the (var, value) literals holds — a SAT clause / set-cover."""

    def parity(self, vars: List[int], odd: bool) -> None:
        """Count of true among these binary vars has the given parity — GF(2) / Lights-Out."""

    def n_vars(self) -> int: ...
    def n_constraints(self) -> int: ...
    def solve(self, max_sweeps: int, patience: int, seed: int) -> Tuple[Optional[List[int]], int]:
        """Solve by annealed relaxation. Returns (assignment, settles) at zero violation, or (None, _) if it
        cannot settle in budget (never a wrong answer)."""

    def violation(self, assignment: List[int]) -> float:
        """Total violation-energy of an assignment (0.0 iff every constraint holds)."""

class StructuralKey:
    """Port-invariant multi-scale WL fingerprint of a relational graph — the cross-game skill address."""

    def __init__(self, scales: int, buckets: int) -> None: ...
    def dim(self) -> int:
        """Signature width (scales × buckets)."""

    def embed(
        self, n: int, node_labels: List[int], edges: List[Tuple[int, int, int]]
    ) -> List[float]:
        """The port-invariant signature (byte-identical under any node relabelling)."""

    def node_embed(
        self, n: int, node_labels: List[int], edges: List[Tuple[int, int, int]], v: int
    ) -> List[float]:
        """The coarse-to-fine signature of a single node (partial matching)."""

    def structural_roles(
        self, n: int, node_labels: List[int], edges: List[Tuple[int, int, int]]
    ) -> List[int]:
        """Group nodes by structural role (WL colour classes ≈ automorphism orbits)."""

    def suggest_next(
        self,
        n: int,
        node_labels: List[int],
        edges: List[Tuple[int, int, int]],
        active: List[int],
    ) -> Optional[int]:
        """Suggest the next node to activate from the active set (frontier + structural affinity)."""

    @staticmethod
    def distance(a: List[float], b: List[float]) -> float:
        """Cosine distance in [0,2] between two signatures."""

class SkillMetric:
    """Learned diagonal metric over structural signatures — contrastive, sleep-trained (metric_step lifted)."""

    def __init__(self, dim: int, lr: float) -> None: ...
    def distance(self, a: List[float], b: List[float]) -> float:
        """Weighted squared distance (lower = more likely same strategy)."""

    def pull(self, a: List[float], b: List[float]) -> None:
        """Pull a same-strategy pair together."""

    def push(self, a: List[float], b: List[float]) -> None:
        """Push a different-strategy pair apart."""

class RecurrentActor:
    """A value-ensemble selector over a belief-state grown by a GrownRecurrence (#384/#387)."""

    def __init__(
        self,
        obs_dim: int,
        delay_depths: List[int],
        half_lives: List[float],
        resonator_freqs: List[float],
        birth_gate: float,
        grow_recurrence: bool,
        n_actions: int,
        ensemble: int,
        max_hidden: int,
        beta: float,
        lr: float,
        seed: int,
    ) -> None: ...
    def sense(self, obs: _FloatVec) -> List[float]:
        """Advance the belief one event; returns the grown belief coordinate (born taps' values)."""

    def act(self, belief_coord: _FloatVec) -> int:
        """Choose an action by the composed free-energy policy over [belief ++ action] candidates."""

    def value(self, belief_coord: _FloatVec, action: int) -> float:
        """The actor's value estimate μ for `action` at the current belief coordinate."""

    def learn_action(self, belief_coord: _FloatVec, action: int, reward: float) -> None:
        """Grow the actor's value for the taken action toward `reward`."""

    def observe_next(self, next_obs: float) -> float:
        """Self-supervised coupling: reinforce the belief toward the next observation."""

    def observe_reward(self, belief_coord: _FloatVec, action: int, reward: float) -> None:
        """Value-residual coupling for sparse reward: grow the belief on reward − Q(belief, action). Call BEFORE learn_action."""

    def n_born(self) -> int:
        """How many recurrent taps the belief has grown."""

    def n_taps(self) -> int:
        """The fixed belief-coordinate width = obs_dim·(n_delays + n_half_lives + 2·n_resonators)."""

    def reset_carrier(self) -> None:
        """Reset the belief's carrier state at an episode boundary; keep born taps + learned value."""

    def born_kinds(self) -> List[str]:
        """The flavours of the grown taps as debug strings (e.g. ['Delay(4)'])."""

    def configure(self, json: str) -> None:
        """Apply a partial-JSON Config object to BOTH the actor heads and the belief predictor (e.g. feature_lr
        #125). Merged onto Config::default() — each call starts from the defaults, so pass every key in one call.
        ValueError on a non-object patch or an unknown key."""

    def set_birth_gate(self, gate: float) -> None:
        """Tune the tap-birth correlation threshold."""

    def set_warmup(self, warmup: int) -> None:
        """Events observed before any tap may be born."""

    def set_surprise_floor(self, floor: float) -> None:
        """Residual-variance floor below which no tap is born (lower for sparse surprise)."""

    def set_auto_grow(self, on: bool) -> None:
        """Enable the dynamic state dimension: grow a new tap + widen the coordinate when the pool can't explain surprise."""

    def set_grow_after(self, n: int) -> None:
        """Consecutive unexplained-surprise events before growing a dimension (default 80)."""

    def n_grown(self) -> int:
        """How many state dimensions the organism grew at runtime (beyond the initial pool)."""

class FovealWorldModel:
    """The recommended world model at real frame scale: predicts only the (2r+1)^2 patch the fovea reads at an
    attention point. Frames are flat row-major w*h vectors of any size (pass w, h per call); a frame whose length is
    not w*h, or a radius outside 0..=64, raises ValueError."""

    def __init__(self, radius: int, action_dim: int, seed: int) -> None: ...
    def tuned(self, lr: float, birth_gate: float, max_hidden: int) -> "FovealWorldModel":
        """Tune the inner patch predictor; returns the model."""
    def with_budget(self, budget: int) -> "FovealWorldModel":
        """Cap the inner predictor's per-settle firing budget; returns the model."""
    def side(self) -> int:
        """The patch side length 2*radius+1."""
    def patch(self, frame: _FloatVec, w: int, h: int, cx: int, cy: int) -> List[float]:
        """The fovea patch centred at (cx, cy), flat row-major (side^2 values)."""
    def observe(
        self, frame: _FloatVec, w: int, h: int, cx: int, cy: int, action: _FloatVec, next: _FloatVec
    ) -> float:
        """Learn the transition at attention (cx, cy); returns the per-cell patch prediction error."""
    def predict(self, frame: _FloatVec, w: int, h: int, cx: int, cy: int, action: _FloatVec) -> List[float]:
        """The attended patch's predicted next appearance, flat row-major (side^2 values)."""
    def controllability(
        self, frame: _FloatVec, w: int, h: int, cx: int, cy: int, actions: List[List[float]]
    ) -> float:
        """How much the prediction at (cx, cy) depends on the action (mean pairwise |delta| across actions)."""
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(bytes: Union[bytes, bytearray, Sequence[int]]) -> "FovealWorldModel":
        """ValueError on a corrupt or wrong-type snapshot."""

def attend(frame: _FloatVec, w: int, h: int, background: float) -> Tuple[int, int]:
    """Centroid of the frame's active (non-background) cells, or the frame centre when nothing is active."""
def attend_regions(frame: _FloatVec, w: int, h: int, background: float) -> List[Tuple[int, int]]:
    """One attention point per connected region of active cells."""
def attend_changes(prev: _FloatVec, cur: _FloatVec, w: int, h: int) -> List[Tuple[int, int]]:
    """One attention point per connected region of changed cells."""
def controllable_regions(
    prev: _FloatVec,
    cur: _FloatVec,
    w: int,
    h: int,
    model: FovealWorldModel,
    actions: List[List[float]],
    keep_frac: float,
) -> List[Tuple[int, int]]:
    """Change regions at least keep_frac as controllable as the frame's most controllable one."""

class WorldModelEnsemble:
    """Parallel k-member world-model ensemble (rayon, byte-identical to serial). Frames/actions are flat f32 lists."""
    def __init__(self, k: int, w: int, h: int, action_dim: int, seed: int) -> None: ...
    def with_budget(self, budget: int) -> "WorldModelEnsemble":
        """Cap the per-settle firing budget (default: settle to the fixed point). Returns self, for chaining."""
    def width_normalized(self) -> None:
        """Force width-normalized (NLMS) learning on every member, in place (members reading >=128 inputs already
        use it; below that the default step-stability cap keeps the raw step stable, NLMS makes it width-independent)."""
    def observe(self, frame: _FloatVec, action: _FloatVec, next: _FloatVec) -> float: ...
    def predict_variance(self, frame: _FloatVec, action: _FloatVec) -> Tuple[List[float], List[float]]: ...
    def epistemic(self, frame: _FloatVec, action: _FloatVec) -> float: ...
    def epistemic_batch(self, frame: _FloatVec, actions: List[List[float]]) -> List[float]: ...
    def predict_batch(self, frame: _FloatVec, actions: List[List[float]]) -> List[Tuple[List[float], float]]: ...
    def n_members(self) -> int: ...
    def snapshot(self) -> bytes: ...
    @staticmethod
    def load(data: bytes, w: int, h: int) -> "WorldModelEnsemble": ...

class GoalBand:
    """Streaming goal-abduction band: the unsigned dynamics residual rho_t grows hypothesis MEMBERS (living pool);
    the signed reward learns a valence readout over them; the abducted goal is the member it relies on most."""

    def __init__(self, n_features: int) -> None: ...
    def with_budget(self, budget: int) -> "GoalBand":
        """Cap the per-settle firing budget (default: settle to the fixed point). Returns self, for chaining."""
    def observe_dynamics(self, features: _FloatVec, rho_t: float, lr: float, budget: int) -> None:
        """Feed one step of the dynamics residual: grow members toward it (birth on surprise)."""
    def metabolize(self, income: float, upkeep: float, death: float) -> None:
        """Run the reliance economy over the members and cull the starved."""
    def observe_reward(self, features: _FloatVec, reward: float, lr: float) -> None:
        """Feed one step of the signed reward: update the valence readout (no member is born here)."""
    def valence_prediction(self, features: _FloatVec) -> float:
        """The band's signed valence prediction for a state (score/rank candidates)."""
    def abducted_goal(self) -> Optional[int]:
        """The member cell the valence readout relies on most (None until a member is grown)."""
    def members_and_valence(self) -> Tuple[List[int], List[float]]:
        """The grown member cells and their valence weights."""
    def member_activation(self, cell: int) -> float:
        """A member cell's activation under the current settle (read after valence_prediction)."""
    def n_grown_members(self) -> int:
        """The number of grown dynamics members (emergent, set by the stream)."""
