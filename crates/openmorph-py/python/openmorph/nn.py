# SPDX-License-Identifier: MIT-0
"""openmorph.nn — a PyTorch-like function-approximation layer over a digital organism.

A :class:`ValueHead` learns ``y ≈ f(x)`` the openmorph way: **by growing**. Each ``learn`` call is one
local, backprop-free delta step — a surprise-birth of structure followed by a tune — so the model grows
exactly the capacity the data demands (watch ``head.n_units`` climb). The interface stays familiar:
``predict(x)`` / ``learn(x, y)`` / ``fit(X, y)``.
"""

from __future__ import annotations

import json

import numpy as np

from ._core import Morph


class ValueHead:
    """A grown, backprop-free scalar function approximator ``y ≈ f(x)``.

    Args:
        in_dim: input feature dimension.
        hidden: max hidden depth the organism may grow (``0`` = linear-in-tanh). Think model capacity —
            but the organism only grows toward it when surprised, so small tasks stay small.
        budget: settle events per forward pass (how long the membrane relaxes to a fixed point).
        seed: growth-PRNG seed, for byte-reproducible runs.

    Example:
        >>> import numpy as np, openmorph as om
        >>> head = om.ValueHead(in_dim=1, hidden=4)
        >>> for _ in range(2000):
        ...     x = np.random.uniform(-1, 1, size=1)
        ...     _ = head.learn(x, np.tanh(1.5 * x[0]))
        >>> round(float(head.predict(np.array([1.0]))), 1)
        0.9
    """

    def __init__(self, in_dim: int, hidden: int = 8, budget: int = 64, seed: int = 0) -> None:
        self.in_dim = int(in_dim)
        self._out = self.in_dim  # a dedicated output cell just past the input band
        self.budget = int(budget)
        self.morph = Morph(self.in_dim + 1)
        self.morph.configure(json.dumps({"max_hidden": int(hidden)}))
        self.morph.rng_seed(int(seed))
        self._idx = np.arange(self.in_dim)  # input band cells (an array crosses the FFI in one call)

    def _vals(self, x) -> np.ndarray:
        # A 1-D float64 array crosses into the organism in ONE call (no per-element list); the binding applies
        # the same f64 -> f32 rounding a list of Python floats gets, so this is bit-identical to `a.tolist()`.
        a = np.asarray(x, dtype=np.float64).ravel()
        if a.size != self.in_dim:
            raise ValueError(f"ValueHead expected {self.in_dim} features, got {a.size}")
        return a

    def predict(self, x) -> float:
        """Forward pass: stimulate the input band, settle, and read the output cell."""
        self.morph.stimulate(self._idx, self._vals(x), self.budget)
        return float(self.morph.cell(self._out))

    def learn(self, x, target: float, lr: float = 0.1) -> float:
        """One grow-step toward ``target``. Returns the error BEFORE the update (a loss you can log)."""
        return float(
            self.morph.grow(self._idx, self._vals(x), self._out, float(target), float(lr), self.budget)
        )

    def fit(self, X, y, epochs: int = 1, lr: float = 0.1) -> list:
        """Convenience training loop over ``(X, y)``. Returns per-epoch mean-absolute error."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        history = []
        for _ in range(epochs):
            errs = [abs(self.learn(X[i], y[i], lr)) for i in range(len(X))]
            history.append(float(np.mean(errs)) if errs else 0.0)
        return history

    @property
    def n_units(self) -> int:
        """How much structure the organism has grown so far — its emergent capacity."""
        return int(self.morph.n_units())

    def snapshot(self) -> bytes:
        """Serialize the organism to JSON ``bytes`` for persistence."""
        return self.morph.snapshot()

    @classmethod
    def from_snapshot(cls, blob: bytes, in_dim: int, budget: int = 64) -> "ValueHead":
        """Rebuild a ValueHead from :meth:`snapshot` bytes (a legacy ``list[int]`` snapshot also loads)."""
        head = cls.__new__(cls)
        head.in_dim = int(in_dim)
        head._out = head.in_dim
        head.budget = int(budget)
        head.morph = Morph.load(blob)
        head._idx = np.arange(head.in_dim)
        return head
