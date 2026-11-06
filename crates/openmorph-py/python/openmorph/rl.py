# SPDX-License-Identifier: MIT-0
"""openmorph.rl — a backprop-free reinforcement-learning actor built from grown value heads.

:class:`Actor` keeps one grown :class:`~openmorph.nn.ValueHead` per action (``Q_a(x)``) plus an intrinsic
**novelty drive** that rewards visiting under-explored states — so it explores on its own, with no
hand-tuned schedule. Everything is local and online: no replay buffer, no target network, no backprop.

    actor = openmorph.Actor(n_features=8, n_actions=4)
    a = actor.act(state, mask=legal_actions)
    actor.learn(state, a, reward, next_state, done)
"""

from __future__ import annotations

import numpy as np

from .nn import ValueHead


class Actor:
    """A value-based actor over a set of grown openmorph value heads.

    Args:
        n_features: dimension of the state feature vector.
        n_actions: number of discrete actions.
        gamma: reward discount.
        lr: grow learning rate.
        beta: intrinsic-novelty weight (``0`` disables curiosity, leaving plain TD).
        hidden: per-head capacity (see :class:`~openmorph.nn.ValueHead`).
        seed: reproducibility.
        budget: settle events per forward pass.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        gamma: float = 0.95,
        lr: float = 0.2,
        beta: float = 0.3,
        hidden: int = 8,
        seed: int = 0,
        budget: int = 64,
    ) -> None:
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.beta = float(beta)
        self.heads = [
            ValueHead(n_features, hidden=hidden, budget=budget, seed=seed + a)
            for a in range(self.n_actions)
        ]
        self._visits: dict = {}
        self._rng = np.random.default_rng(seed)

    def _key(self, x) -> int:
        return hash(np.asarray(x, dtype=np.float64).round(2).tobytes())

    def values(self, x) -> np.ndarray:
        """The grown Q-value of every action at state ``x`` (a numpy array of length ``n_actions``)."""
        return np.array([h.predict(x) for h in self.heads], dtype=np.float64)

    def act(self, x, epsilon: float = 0.05, mask=None) -> int:
        """Pick an action: epsilon-greedy over grown Q-values.

        Args:
            x: state feature vector.
            epsilon: exploration probability.
            mask: optional iterable of legal action indices (defaults to all).
        """
        legal = list(range(self.n_actions)) if mask is None else [int(a) for a in mask]
        if not legal:
            legal = list(range(self.n_actions))
        if self._rng.random() < epsilon:
            return int(self._rng.choice(legal))
        q = self.values(x)
        return int(max(legal, key=lambda a: q[a]))

    def learn(self, x, a: int, reward: float, x_next, done: bool) -> float:
        """One online TD grow-step for action ``a``, with an intrinsic novelty bonus on the next state.

        Returns the value-head error before the update (a loss you can log).
        """
        k = self._key(x_next)
        self._visits[k] = self._visits.get(k, 0) + 1
        intrinsic = self.beta / np.sqrt(1.0 + self._visits[k])
        bootstrap = 0.0 if done else self.gamma * float(self.values(x_next).max())
        target = float(reward) + intrinsic + bootstrap
        return float(self.heads[int(a)].learn(x, target, self.lr))

    @property
    def n_units(self) -> int:
        """Total grown structure across all action heads — the actor's emergent capacity."""
        return sum(h.n_units for h in self.heads)
