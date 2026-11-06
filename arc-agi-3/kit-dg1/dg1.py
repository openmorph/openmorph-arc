# SPDX-License-Identifier: MIT-0
"""Stream DG1's switch over kit BH (`bs.BsThink`), off by default; off, `DgThink` plays exactly as `BsThink`.

* `DG1_REL=1` -- RELATIONAL NOVELTY. A state's relation cell is the set of colour pairs that touch in its grid (the
  cells the state key paints out excluded). The planner's P(new state) of an untried action at a reached state is
  multiplied by 1 + w / sqrt(n), n the decisions this level taken in states of the same relation cell (w =
  `DG1_REL_W`, default 1), once the level has taken `DG1_REL_AFTER` actions (default 300). Untried actions where
  objects touch in a way seldom seen are worth more than one more position in open space (count-based exploration
  over a coarse cell, as Go-Explore's). Before 300 actions the level plays as kit BH: the early hits carry the score.

AUTHORED (labelled): the relation cell (touching colour pairs), the weight form and the 300-action threshold.
LEARNED per level: the counts. Nothing names a game, a colour or a mechanic.
"""
import collections
import os

import numpy as np

import bs


def relation_cell(h):
    """The set of unordered colour pairs that touch (4-neighbours) in `h`; cells < 0 (painted out) take no part."""
    out = set()
    for a, b in ((h[:, 1:], h[:, :-1]), (h[1:, :], h[:-1, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        if m.any():
            lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
            out |= set(np.unique(lo * 64 + hi).tolist())
    return frozenset(out)


class DgThink(bs.BsThink):
    def __init__(self, game_id, seed=1, verifier=None, arm="B"):
        super().__init__(game_id, seed, verifier=verifier, arm=arm)
        self.dg_rel = bool(int(os.environ.get("DG1_REL", "0")))
        self.dg_w = float(os.environ.get("DG1_REL_W", "1"))
        self.dg_after = int(os.environ.get("DG1_REL_AFTER", "300"))   # actions into a level before the weight applies
        self.dg_cell = {}                               # state key -> relation cell
        self.dg_n = collections.Counter()               # relation cell -> decisions this level
        self.dg_level = None
        self.dg_start = 0
        self.dg_stats = collections.Counter()

    def _masked(self, g):
        h = g.copy()
        cm = self.clock_mask(g.shape)
        if cm.shape == g.shape:
            h[cm] = -1
        if self.bs_key and getattr(self, "bs_shape", None) == g.shape and self.bs_mask.size == h.size:
            h.reshape(-1)[self.bs_mask] = -1
        return h

    def plan(self, key, acts_here, g, comp_of):
        if self.dg_rel:
            if self.dg_level is None or self.dg_level[0] != self.level:
                self.dg_start = self.n_steps
            if self.dg_level != (self.level, self.raw):
                self.dg_level = (self.level, self.raw)
                self.dg_cell, self.dg_n = {}, collections.Counter()
            c = relation_cell(self._masked(g))
            if key not in self.dg_cell:
                self.dg_stats["cells"] += c not in self.dg_n
            self.dg_cell[key] = c
            self.dg_n[c] += 1
        return super().plan(key, acts_here, g, comp_of)

    def predict_next(self, n, a, t, par, graph=True):
        pred = super().predict_next(n, a, t, par, graph)
        if self.dg_rel and not graph and pred["source"] == "rates" and self.dg_steps() >= self.dg_after:
            c = self.dg_cell.get(n)
            if c is not None:
                pred = dict(pred, p_new=pred["p_new"] * (1.0 + self.dg_w / np.sqrt(self.dg_n[c])))
        return pred

    def dg_steps(self):
        """Actions this level (the raw switch restarts TK's level count, not this one)."""
        return self.n_steps - self.dg_start

    def _remap(self):
        super()._remap()
        if self.dg_rel:
            # keys changed: re-read every known state's cell from the grid that made it
            self.dg_cell = {k: relation_cell(self._masked(g)) for k, g in self.bs_key_grid.items()}
