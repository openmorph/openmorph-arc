# SPDX-License-Identifier: MIT-0
"""Reflex → plan → remember, from Python — the ARC-3 epic (#337) as thin helpers over the core.

These mirror the in-core proofs one-for-one, so the agent composes the *validated* substrate instead of
re-deriving it:

    WorldModel   ← substrate_imagine        (#332) grown forward model + calibrated epistemic σ²
    plan_step    ← substrate_planning        (#334) deliberation over a learned/experienced adjacency
    Explorer     ← substrate_voi / _skillmap (#333 reframe) VoI + metric-map frontier — the COVERAGE levers
    infer_goal   ← substrate_goal_inference  (#335) grown reward attribution via reliance_of

Design notes carried from the proofs (truth-not-trophy):
  * The world model predicts DISPLACEMENT (translation-invariant, learns near-exactly); the environment keeps
    its own hard constraints (walls/bounds). Predicting absolute next-state caps fidelity ~0.75 and planning
    fails — see substrate_planning.
  * σ² is calibrated at the OPTION level but is NOT a state-coverage director (it saturates when dynamics
    generalise). Direct exploration with `Explorer` (VoI + frontier), NOT world-model σ² — see
    docs/archive/investigations/333-world-model-curiosity.md.
"""

from __future__ import annotations

import base64
import json
from typing import Callable, Sequence

from ._core import Morph

_GOLDEN = 0x9E3779B97F4A7C15


class WorldModel:
    """A grown forward model with a calibrated epistemic σ² (#332 `imagine`).

    An ensemble of `k` seed-varied Morphs, each a `grow_backbone` basis + one linear head per output feature
    (the `substrate_predict` idiom). `imagine` returns the mean prediction and the across-member variance —
    high on never-modelled options, self-annealing with data.
    """

    def __init__(self, n_in: int, n_out: int, k: int = 8, width: int = 24, scale: float = 2.5,
                 lr: float = 0.03, budget: int = 300, seed: int = 0x12345678):
        self.n_in, self.n_out = n_in, n_out
        self.lr, self.budget = lr, budget
        self.members: list[Morph] = []
        self.heads: list[int] = []
        for i in range(k):
            m = Morph(n_in)
            m.rng_seed((seed ^ ((i + 1) * _GOLDEN)) & 0xFFFFFFFFFFFFFFFF)
            reads = list(range(n_in)) + m.grow_backbone(list(range(n_in)), width, scale)
            base = len(m.cells())
            self.heads = [base + d for d in range(n_out)]
            for c in self.heads:
                m.add_unit(reads, [0.0] * len(reads), 0.0, c)
            self.members.append(m)

    def learn(self, ctx: Sequence[float], target: Sequence[float]) -> None:
        """Correct every member's heads toward `target` given context `ctx` (one real transition)."""
        idx = list(range(self.n_in))
        val = list(ctx)
        for m in self.members:
            for d, t in zip(self.heads, target):
                m.learn_step(idx, val, d, float(t), self.lr, self.budget)

    def replay(self, buffer: Sequence[tuple[Sequence[float], Sequence[float]]], epochs: int) -> None:
        """Free internal training: replay the buffer of real transitions `epochs` times (no new samples)."""
        for _ in range(epochs):
            for ctx, tgt in buffer:
                self.learn(ctx, tgt)

    def imagine(self, ctx: Sequence[float]) -> tuple[list[float], float]:
        """Predict `(mean_output, sigma2)` for context `ctx` without acting. σ² = mean-over-heads variance."""
        idx = list(range(self.n_in))
        val = list(ctx)
        preds: list[list[float]] = []
        for m in self.members:
            m.stimulate(idx, val, self.budget)
            cells = m.cells()
            preds.append([cells[h] for h in self.heads])
        k = len(preds)
        mean = [sum(p[d] for p in preds) / k for d in range(self.n_out)]
        var = 0.0
        for d in range(self.n_out):
            var += sum((p[d] - mean[d]) ** 2 for p in preds) / k / self.n_out
        return mean, var

    def snapshot(self) -> bytes:
        """Serialize the trained world model (every ensemble member + its heads) to bytes — a curriculum-trained
        prior PERSISTS, so it can be trained once and SHIPPED (e.g. to Kaggle) instead of cold-starting each run.
        Round-trips with ``WorldModel.load``: ``imagine``/``learn``/``replay`` resume exactly."""
        blob = {
            "n_in": self.n_in,
            "n_out": self.n_out,
            "lr": self.lr,
            "budget": self.budget,
            "heads": self.heads,
            "members": [base64.b64encode(bytes(m.snapshot())).decode("ascii") for m in self.members],
        }
        return json.dumps(blob).encode("utf-8")

    @staticmethod
    def load(data: bytes) -> "WorldModel":
        """Restore a ``WorldModel`` from ``snapshot`` bytes (bypasses re-growing the ensemble)."""
        blob = json.loads(bytes(data).decode("utf-8"))
        obj = WorldModel.__new__(WorldModel)
        obj.n_in, obj.n_out = int(blob["n_in"]), int(blob["n_out"])
        obj.lr, obj.budget = float(blob["lr"]), int(blob["budget"])
        obj.heads = [int(h) for h in blob["heads"]]
        obj.members = [Morph.load(base64.b64decode(s)) for s in blob["members"]]
        return obj


def plan_step(field: Morph, adjacency: Sequence[Sequence[int]], goal: int, pos: int,
              deliberate: int = 0, cost: float = 1.0) -> tuple[int, list[float]]:
    """One deliberation step over an adjacency (#331/#334): `relax_values` for `deliberate` sweeps
    (`0` = to convergence), then return `(best_neighbour, values)`. More sweeps ⇒ fewer external actions.
    `field` persists the value field across calls; call `field.prefer(goal, 0.0)` once before the loop."""
    field.relax_values([list(a) for a in adjacency], cost, deliberate)
    best, best_v = pos, float("inf")
    for nb in adjacency[pos]:
        v = field.value_of(nb)
        if v < best_v or (v == best_v and nb < best):
            best, best_v = nb, v
    return best, [field.value_of(c) for c in range(len(field.cells()))]


class Explorer:
    """The COVERAGE lever (#333 reframe): direct exploration by value-of-information, NOT world-model σ².

    `voi(options, beta)` scores each option by `mean + beta·spread` over its grown value-estimator cells
    (ensemble disagreement = epistemic uncertainty), the `select_explore` primitive. `frontier(...)` picks the
    farthest cell on the grown concept-map (the `substrate_skillmap` behaviour-novelty frontier).
    """

    def __init__(self, field: Morph):
        self.field = field

    def voi(self, options: Sequence[Sequence[int]], beta: float) -> int:
        """Choose the option index maximising value + β·disagreement (VoI)."""
        return self.field.select_explore([list(o) for o in options], beta)

    def frontier(self, candidates: Sequence[int], visited: Sequence[int], dims: int) -> int:
        """Farthest-point on the grown concept-map: the candidate maximising its min distance to `visited`
        (behaviour-space novelty). Falls back to the first candidate if nothing is visited yet."""
        if not visited:
            return int(candidates[0])
        best, best_d = int(candidates[0]), -1.0
        for c in candidates:
            d = min(self.field.distance(c, v, dims) for v in visited)
            if d > best_d:
                best, best_d = int(c), d
        return best


def infer_goal(field: Morph, feature_cells: Sequence[int]) -> int:
    """The inferred GOAL predicate (#335): the feature the grown reward predictor RELIES on most
    (`reliance_of`). Grow a linear reward head over `feature_cells` with `learn_step` first, then call this."""
    return max(feature_cells, key=field.reliance_of)


def build_reward_predictor(field: Morph, feature_cells: Sequence[int], reward_cell: int) -> None:
    """Attach a linear reward-prediction head reading `feature_cells` and writing `reward_cell`. Train it with
    `field.learn_step(feature_idx, feature_vals, reward_cell, reward, lr, budget)` per step; then `infer_goal`."""
    field.add_unit(list(feature_cells), [0.0] * len(feature_cells), 0.0, reward_cell)


# A convenience so callers can discover the intended composition order.
PIPELINE: tuple[str, ...] = (
    "perceive → seed_coord",
    "WorldModel.imagine (learn dynamics)",
    "plan_step (deliberate over the model / experienced adjacency)",
    "Explorer.voi / Explorer.frontier (coverage when the plan is uncertain)",
    "infer_goal (attribute sparse reward → dense goal)",
    "snapshot (serialise the relational skill)",
)

__all__ = [
    "WorldModel",
    "plan_step",
    "Explorer",
    "infer_goal",
    "build_reward_predictor",
    "PIPELINE",
]
