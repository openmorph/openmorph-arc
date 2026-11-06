# SPDX-License-Identifier: MIT-0
"""Stream DG3's switch over kit BH (`../kit-bh/bs.py`, `BsThink`). Off by default; off, `Dg3Think` plays exactly as
`BsThink`. (A second switch, no click candidate on a painted cell, cost su15's level on the public 25 and was deleted:
`docs/research/1.9-streams/dg3-diagnosis.md`.)

* `DG3_BAR=1` -- A BUDGET BAR IS NOT STATE. A bar that shortens by a cell every action (or every few) changes WITH the
  action's own effect, never alone, so the tick test (a kind that was a move's only change) never fires on it and no
  cell of it changes in most transitions (v3's clock). Every state is then new: the planner's P(new) stays near 1, a
  toggle pressed twice (on, off) or a move and its undo look like discoveries, and the search never closes a loop.
  The test, per change kind (a cell's old and new colour), over the current life:
  - ISOLATED: only transitions where its cells form small parts (<= 3 cells) touching no other change count;
  - MONOTONE: a cell it changed did not change again in the life (a refill, the inverse kind, is ignored), and its
    inverse never came in the same transition (an object's leading and trailing edges);
  - A LINE: every cell it changed this life lies in one row or one column, the cells a run without gaps over 2;
  - >= 4 times in the life under >= 2 action classes (keys, or distinct clicks), or >= 8 times under one.
  A kind that passes paints out its line's run of its two colours (BS's tick cells: the key and the effect drop them,
  and every state-keyed structure is remapped). A kind that broke MONOTONE in this level never passes in it.

AUTHORED (labelled, as the rules ask): the four conditions and their numbers. LEARNED per game: which kinds, which
cells. Nothing names a game, a colour or a mechanic.
"""
import collections
import os

import numpy as np

import bs

MIN_TIMES = 4       # occurrences of a kind in one life before it can be a bar
MAX_PART = 3        # cells in one isolated part of one transition
MAX_GAP = 2         # largest gap between the bar's cells along its line


class Dg3Think(bs.BsThink):
    def __init__(self, game_id, seed=1, verifier=None, arm="B"):
        super().__init__(game_id, seed, verifier=verifier, arm=arm)
        self.dg3_bar = bool(int(os.environ.get("DG3_BAR", "0")))
        self.dg3_stats = collections.Counter()
        self._dg3_reset_level(None)

    def _dg3_reset_level(self, shape):
        self.dg3_shape, self.dg3_level = shape, getattr(self, "level", 0)
        self.dg3_banned = set()          # kinds that broke MONOTONE in this level
        self.dg3_bars = set()            # kinds painted in this level
        self._dg3_reset_life()

    def _dg3_reset_life(self):
        self.dg3_cells = collections.defaultdict(set)      # kind -> cells it changed this life
        self.dg3_times = collections.Counter()             # kind -> transitions it appeared in this life
        self.dg3_classes = collections.defaultdict(set)    # kind -> action classes it appeared under
        self.dg3_touched = collections.defaultdict(set)    # cell -> the kinds that changed it this life (no refill)

    def choose_action(self, frames, latest):
        if self.dg3_bar:
            from arcengine import GameState
            if latest.state is GameState.GAME_OVER:
                self._dg3_reset_life()
        return super().choose_action(frames, latest)

    def effect(self, a, b):
        e = super().effect(a, b)
        if not self.dg3_bar or not self.bs_key or a.shape != b.shape or self.prev is None:
            return e
        if self.dg3_shape != a.shape or self.dg3_level != self.level:
            self._dg3_reset_level(a.shape)
        if self._dg3_observe(a, b, self.prev[1]) and self._recompute():
            self._remap()
            # the effect as BS reads it, with the new painted cells
            d = (a != b).reshape(-1) & ~self.clock_mask(a.shape).reshape(-1) & ~self.bs_mask
            return bool(d.any())
        return e

    def _dg3_observe(self, a, b, act):
        h, w = a.shape
        d = a != b
        if not d.any():
            return False
        ys, xs = np.nonzero(d)
        cells = [int(y) * w + int(x) for y, x in zip(ys, xs)]
        kind_of = {p: (int(a.flat[p]), int(b.flat[p])) for p in cells}
        parts = bs._components(cells, w, h)
        isolated = collections.defaultdict(lambda: True)
        here = collections.defaultdict(set)
        for comp in parts:
            ks = {kind_of[p] for p in comp}
            for k in ks:
                here[k].update(p for p in comp if kind_of[p] == k)
                if len(ks) > 1 or len(comp) > MAX_PART:
                    isolated[k] = False
        cls = act if " " in act else self._class(act)
        for k, ps in here.items():
            inv = (k[1], k[0])
            if inv in here:                                 # an object's leading and trailing edges: never a bar
                self.dg3_banned |= {k, inv}
                continue
            if inv in self.dg3_cells and ps <= self.dg3_cells[inv]:
                continue                                    # a refill of a bar: not a contradiction, not a tick
            for p in ps:
                if self.dg3_touched[p]:                     # changed twice in one life: no kind there is a bar
                    self.dg3_banned |= self.dg3_touched[p] | {k}
                self.dg3_touched[p].add(k)
            if not isolated[k]:
                continue                                    # next to another change: not counted
            self.dg3_cells[k] |= ps
            self.dg3_times[k] += 1
            self.dg3_classes[k].add(cls)
        new = []
        for k in here:
            if k in self.dg3_banned or k in self.dg3_bars:
                continue
            if self.dg3_times[k] < MIN_TIMES or (len(self.dg3_classes[k]) < 2 and self.dg3_times[k] < 2 * MIN_TIMES):
                continue
            line = self._line(self.dg3_cells[k], w)
            if line is not None:
                new.append((k, line))
        if not new:
            return False
        grew = False
        for k, (axis, idx, lo, hi) in new:
            self.dg3_bars.add(k)
            self.dg3_stats["bars"] += 1
            for t in range(lo, hi + 1):
                p = idx * w + t if axis == "row" else t * w + idx
                if int(b.flat[p]) in k and not self.bs_tick[p]:
                    self.bs_tick[p] = True
                    grew = True
            # extend along the line over the run of the bar's two colours
            for step in (-1, 1):
                t = lo - 1 if step < 0 else hi + 1
                while 0 <= t < (w if axis == "row" else h):
                    p = idx * w + t if axis == "row" else t * w + idx
                    if int(b.flat[p]) not in k:
                        break
                    if not self.bs_tick[p]:
                        self.bs_tick[p] = True
                        grew = True
                    t += step
        if grew:
            self.dg3_stats["painted"] = int(self.bs_tick.sum())
            self._revote()
        return grew

    @staticmethod
    def _line(cells, w):
        ys = {p // w for p in cells}
        xs = {p % w for p in cells}
        if len(ys) == 1:
            axis, idx, along = "row", next(iter(ys)), sorted(xs)
        elif len(xs) == 1:
            axis, idx, along = "col", next(iter(xs)), sorted(ys)
        else:
            return None
        if len(along) < 2 or any(b - a > MAX_GAP + 1 for a, b in zip(along, along[1:])):
            return None
        return axis, idx, along[0], along[-1]

    def dump(self):
        super().dump()
        import json
        rec = dict(step=self.n_steps, level=self.level, raw=self.raw, shape=self.dg3_shape,
                   bars=sorted(map(list, self.dg3_bars)), stats=dict(self.dg3_stats),
                   ticks=sorted(map(list, self.bs_ticks)), tick_cells=int(self.bs_tick.sum()))
        with open(os.path.join(self.diag, f"{self.game_id}.dg3.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
