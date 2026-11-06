# SPDX-License-Identifier: MIT-0
"""Stream HY: the screen as many small hypotheses.

Before every action the `Verifier` predicts many separate facts about the next frame; after the action it checks
each one against the frame that came back. Every fact is a confirmed or refuted hypothesis, and both are learning
signal. The verifier is an observer: it never chooses an action (the agent it wraps does), so the same layer scores
any agent, and an agent may ask it what it expects (`expect`) before choosing.

The fact types (all yes/no; the prediction is the sign of a `Field` RLS readout, `weights/eval.py` RlsHead, unless
the line says COMPOSED):

  effect        a cell outside the clock changes
  death         the game is over after the action
  level         the level is completed by the action
  reset         the grid returns to the level's first grid (outside the clock) without a game over
  clock_tick    a clock cell changes (the clock: a cell changing in >= 60 % of the level's transitions, after 8)
  novel         a cell that had never changed in this level changes
  cell_change   per active cell (one that has changed in this level): it changes now
  cell_value    per active cell that changed: its new value is the predicted one (a learned table, below)
  self_moves    the self is displaced
  self_pos      COMPOSED: the self's next position (mode displacement of the action if `self_moves`, else none)
  contact       COMPOSED: the self touches another object after the action (from the predicted position)
  obj_gone      per object: no cell of it keeps its colour and no translated copy exists
  obj_moved     per object: it moved rigidly
  obj_recolour  per object: all its cells now hold one other colour
  obj_reshape   per object: its colour's component over it has another cell set (not a translation)

What is AUTHORED (game-blind, nothing names a game, a colour or a mechanic): the logical grid reading (`muse.Grid`,
kit X's), the clock rule (kit X's), the list of fact types above and their truth tests, the feature lists below, the
choice of the self (the colour that moves rigidly most often under non-click actions), the 4-connected object
reading, and the cell-value table's key (old value -> new value, per cell, then for the whole game). LEARNED per game:
every readout, the self's colour, its displacement per action, the value table, the clock, and every rate.

Scores (`report`): per level and per fact type a confusion matrix. The understanding score U is the macro mean over
fact types of the AUC (the chance that a random "yes" case got a higher predicted probability than a random "no"
case), taken over the types that saw both answers: a constant predictor scores 0.5 on every type, so U = 0.5 is no
understanding and U = 1 is every fact ranked right. The composed `self_pos` and `cell_value` have no probability and
report a raw accuracy only (with `self_pos` the baseline "the self stays" is reported beside it).
U_acc is the macro mean of the raw accuracy (flattered by facts that are almost always no), kept for reference.
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "kit-x", "muse"), os.path.join(HERE, "..", "kit-x", "muse", "weights")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import muse  # noqa: E402  (kit X's grid reader and components)

LEARNED = ("effect", "death", "level", "reset", "clock_tick", "novel", "cell_change", "self_moves",
           "obj_gone", "obj_moved", "obj_recolour", "obj_reshape")
COMPOSED = ("cell_value", "self_pos", "self_pos_base", "contact")
TYPES = LEARNED + COMPOSED
N_ACT = 25            # kit X's action features: id one-hot (7), clicked colour (16), log size, one cell
N_LIFE = 5            # steps in this life (log), / mean past life, past it, clock cells used since the life began, none
N_AHEAD = 18          # the colour the self would enter (16), out of the grid, nothing known
N_CELL = 7 + 10
N_OBJ = 7 + 8 + 16
MAX_OBJECTS = int(os.environ.get("HY_MAX_OBJECTS", "48"))     # objects with facts per step (cost)
MAX_CELLS = int(os.environ.get("HY_MAX_CELLS", "160"))       # active cells with facts per step (cost)
WIN = 50              # steps per window of the understanding curve


class Head:
    """One yes/no readout: the morph's RLS readout (`Config::readout_rls`), target +-0.9."""

    def __init__(self, k, seed):
        from eval import RlsHead
        self.h = RlsHead(k + 1, prior=1.0, backbone=0, seed=seed)
        self.n = 0

    def predict(self, f):
        return self.h.predict(np.append(f, 1.0)) if self.n else 0.0

    def learn(self, f, yes, times=1):
        for _ in range(times):
            self.h.learn(np.append(f, 1.0), 0.9 if yes else -0.9)
        self.n += 1


NB = 20               # coarse probability bins: the trust verdicts and the arms' rates
NBF = 200             # fine bins of the raw readout over [-1.5, 1.5]: every reported AUC (rare facts sit near -0.9)


class Confusion:
    __slots__ = ("tp", "fp", "fn", "tn", "surprise", "pos", "neg", "posf", "negf")

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = self.surprise = 0
        self.pos, self.neg = [0] * NB, [0] * NB        # histogram of the predicted probability, by the truth
        self.posf, self.negf = None, None               # the same over the raw readout, fine (allocated on use)

    def add(self, pred, truth, conf, prob=None, raw=None):
        if raw is not None:
            if self.posf is None:
                self.posf, self.negf = [0] * NBF, [0] * NBF
            b = min(NBF - 1, max(0, int((raw + 1.5) / 3.0 * NBF)))
            (self.posf if truth else self.negf)[b] += 1
        if prob is not None:
            b = min(NB - 1, max(0, int(prob * NB)))
            (self.pos if truth else self.neg)[b] += 1
        if pred and truth:
            self.tp += 1
        elif pred:
            self.fp += 1
        elif truth:
            self.fn += 1
        else:
            self.tn += 1
        if pred != truth and conf >= 0.5:
            self.surprise += 1

    @property
    def n(self):
        return self.tp + self.fp + self.fn + self.tn

    def acc(self):
        return (self.tp + self.tn) / self.n if self.n else None

    def bal(self):
        pos, neg = self.tp + self.fn, self.tn + self.fp
        if not pos or not neg:
            return None
        return 0.5 * (self.tp / pos + self.tn / neg)

    @staticmethod
    def _auc(pos, neg):
        P, N = sum(pos), sum(neg)
        if not P or not N:
            return None
        below = a = 0
        for p, q in zip(pos, neg):
            a += p * (below + 0.5 * q)
            below += q
        return a / (P * N)

    def auc(self):
        """Threshold-free discrimination on the 20 coarse bins (what the arms read as a fact type's rate)."""
        return self._auc(self.pos, self.neg)

    def aucf(self):
        """The reported AUC: the chance a random yes got a higher readout than a random no, on 200 fine bins of the
        raw readout (the coarse bins tie every rare prediction near P = 0 and pull the AUC toward 0.5)."""
        if self.posf is None:
            return self.auc()
        return self._auc(self.posf, self.negf)

    def merge(self, o):
        for s in ("tp", "fp", "fn", "tn", "surprise"):
            setattr(self, s, getattr(self, s) + getattr(o, s))
        self.pos = [a + b for a, b in zip(self.pos, o.pos)]
        self.neg = [a + b for a, b in zip(self.neg, o.neg)]
        if o.posf is not None:
            if self.posf is None:
                self.posf, self.negf = [0] * NBF, [0] * NBF
            self.posf = [a + b for a, b in zip(self.posf, o.posf)]
            self.negf = [a + b for a, b in zip(self.negf, o.negf)]

    def row(self):
        return dict(n=self.n, tp=self.tp, fp=self.fp, fn=self.fn, tn=self.tn, surprise=self.surprise,
                    acc=_r(self.acc()), bal=_r(self.bal()), auc=_r(self.aucf()), auc20=_r(self.auc()),
                    base=_r((self.tp + self.fn) / self.n if self.n else None))


def _r(v):
    return None if v is None else round(v, 4)


def understanding(conf):
    """(U, U_acc) of {type: Confusion}: macro AUC over types that saw both answers; macro raw accuracy."""
    bals = [c.aucf() for c in conf.values() if c.aucf() is not None]
    accs = [c.acc() for c in conf.values() if c.n >= 5]
    return (sum(bals) / len(bals) if bals else None), (sum(accs) / len(accs) if accs else None)


def prob(p):
    """A readout's output (target +-0.9) as a probability of yes."""
    return min(1.0, max(0.0, 0.5 + p / 1.8))


def one_hot(i, n):
    v = np.zeros(n)
    if 0 <= i < n:
        v[i] = 1.0
    return v


def background(g):
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def objects(g, bg):
    """4-connected same-colour components that are not the background, smallest first (walls last), capped."""
    out = []
    big = 0.25 * g.size
    for c in [int(v) for v in np.unique(g) if v != bg]:
        for comp in muse._components(g == c):
            if len(comp) <= big:
                out.append((c, frozenset(comp)))
    out.sort(key=lambda o: (len(o[1]), min(o[1])))
    return out[:MAX_OBJECTS]


def shape_key(cells):
    y0 = min(p[0] for p in cells)
    x0 = min(p[1] for p in cells)
    return frozenset((y - y0, x - x0) for y, x in cells), (y0, x0)


class Verifier:
    """Predicts the facts of the next frame before each action, checks them after, learns from both."""

    def __init__(self, seed=0, replay_surprise=0):
        self.frames, self.G = [], None
        self.replay_surprise = int(replay_surprise)       # arm +4: extra learning passes on a refuted confident fact
        dims = dict(effect=N_ACT, death=N_ACT + N_AHEAD + N_LIFE, level=N_ACT, reset=N_ACT + N_AHEAD + N_LIFE, clock_tick=N_ACT,
                    novel=N_ACT + N_AHEAD, cell_change=N_CELL, self_moves=7 + N_AHEAD, obj_gone=N_OBJ,
                    obj_moved=N_OBJ, obj_recolour=N_OBJ, obj_reshape=N_OBJ)
        self.heads = {t: Head(d, seed + i) for i, (t, d) in enumerate(sorted(dims.items()))}
        self.level = 0
        self.conf = collections.defaultdict(lambda: collections.defaultdict(Confusion))   # level -> type -> C
        self.steps = collections.Counter()                 # level -> checked steps
        self.game = collections.defaultdict(Confusion)     # type -> Confusion over the whole game (the rates)
        self.img = collections.defaultdict(collections.Counter)   # level -> imagined-frame scores
        self.horizon = {}                                  # checked steps -> U and per-type AUC then
        self.win = collections.defaultdict(lambda: collections.defaultdict(Confusion))   # 50-step window -> type -> C
        self.info = collections.defaultdict(collections.Counter)   # 100-step bin -> information per action
        self.cur = collections.Counter()
        self.level_up_at = []                              # checked-step index of each level completion
        self.tau = float(os.environ.get("HY_TAU", "0.95"))
        self.pending = None
        self.frozen = False                                # a probe: readouts are read, never trained
        self.lite = bool(int(os.environ.get("HY_LITE", "0")))
        self._read_key, self._read_val, self._obj_cache = None, None, {}   # only the step facts and the self (the kit's cost)
        self.errors = 0
        self.replay = []                                   # (type, features, truth) of refuted confident facts
        # carried across levels: the self, its displacement per action, the value table
        self.self_votes = collections.Counter()
        self.lives = []                                    # past life lengths (steps from a level start or a death)
        self.life = 0
        self.life_start = None
        self.disp = collections.defaultdict(collections.Counter)      # action id -> Counter((dx, dy))
        self.value_game = collections.defaultdict(collections.Counter)  # old -> Counter(new)
        self._new_level()

    # ------------------------------------------------------------------------------------------- level state
    def _new_level(self):
        self.frames, self.G = [], None
        self.change_count, self.n_trans = None, 0
        self.last_diff = None
        self.start = None
        self.self_cells = None
        self.value_cell = collections.defaultdict(collections.Counter)  # (y, x, old) -> Counter(new)

    def grid_of(self, frame):
        f = np.asarray(frame, dtype=np.int64)
        if len(self.frames) < 12 and not any((f == x).all() for x in self.frames):
            self.frames.append(f)
            self.G = muse.Grid(self.frames)
        return self.read(f)

    def read(self, f):
        """`self.G.grid(f)`, cached for the last frame read under this grid (the same frame is read up to 3 times a
        step). The returned grid is shared: callers copy before writing."""
        key = (id(self.G), f.tobytes())
        if self._read_key != key:
            self._read_key, self._read_val = key, self.G.grid(f)
        return self._read_val

    def objects_of(self, g):
        """(background, objects) of g, cached for the last few grids (every candidate action reads the same grid)."""
        key = (g.shape, g.tobytes())
        hit = self._obj_cache.get(key)
        if hit is None:
            bg = background(g)
            hit = (bg, objects(g, bg))
            if len(self._obj_cache) >= 4:
                self._obj_cache.pop(next(iter(self._obj_cache)))
            self._obj_cache[key] = hit
        return hit

    def clock(self, shape):
        if self.change_count is None or self.change_count.shape != shape or self.n_trans < 8:
            return np.zeros(shape, bool)
        return self.change_count >= 0.6 * self.n_trans

    def active(self, shape):
        if self.change_count is None or self.change_count.shape != shape:
            return np.zeros(shape, bool)
        return self.change_count > 0

    @property
    def self_colour(self):
        if not self.self_votes:
            return None
        c, v = self.self_votes.most_common(1)[0]
        return c if v >= 2 else None

    def locate_self(self, g):
        """The self's cells in g: the component of the self's colour nearest its last known cells."""
        c = self.self_colour
        if c is None or not (g == c).any():
            return None
        comps = muse._components(g == c)
        if self.self_cells:
            cy = sum(p[0] for p in self.self_cells) / len(self.self_cells)
            cx = sum(p[1] for p in self.self_cells) / len(self.self_cells)
        else:
            cy, cx = g.shape[0] / 2, g.shape[1] / 2

        def d(comp):
            return min((p[0] - cy) ** 2 + (p[1] - cx) ** 2 for p in comp), -len(comp)
        return frozenset(min(comps, key=d))

    # ------------------------------------------------------------------------------------------- features
    def action_of(self, act):
        """(id, logical click cell or None) of a GameAction."""
        aid = int(getattr(act, "value", act))
        xy = None
        if aid == 6 and self.G is not None:
            d = act.action_data.model_dump() if hasattr(act, "action_data") else {}
            xy = self.G.cell_of_pixel(int(d.get("x", 0)), int(d.get("y", 0)))
        return aid, xy

    @staticmethod
    def act_features(aid, xy, g, comp_size):
        f = np.zeros(N_ACT)
        if 1 <= aid <= 7:
            f[aid - 1] = 1.0
        if xy is not None:
            x, y = xy
            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
                f[7 + int(g[y, x])] = 1.0
                s = comp_size.get((y, x), 1)
                f[23] = math.log1p(s) / 5.0
                f[24] = 1.0 if s == 1 else 0.0
        return f

    def mode_disp(self, aid):
        c = self.disp.get(aid)
        if not c:
            return None
        (d, n), = c.most_common(1)
        return d if n >= 1 else None

    def ahead(self, g, aid, cells):
        """The colour the self would enter under the action's usual displacement (one-hot), out of the grid, unknown."""
        f = np.zeros(N_AHEAD)
        d = self.mode_disp(aid) if cells else None
        if d is None or d == (0, 0):
            f[17] = 1.0
            return f, None
        dx, dy = d
        h, w = g.shape
        new = [(y + dy, x + dx) for y, x in cells]
        if any(not (0 <= y < h and 0 <= x < w) for y, x in new):
            f[16] = 1.0
            return f, None
        entered = collections.Counter(int(g[y, x]) for y, x in new if (y, x) not in cells)
        if entered:
            f[entered.most_common(1)[0][0]] = 1.0
        return f, frozenset(new)

    # ------------------------------------------------------------------------------------------- predict
    def expect(self, g, aid, xy, light=False):
        """Every fact predicted for (g, action): list of (type, key, features, p). p > 0 means yes. `light`: only the
        global facts and the self (what an agent asks for each candidate action)."""
        out = []
        bg, objs = self.objects_of(g)
        comp_size = {}
        clicked_obj = None
        for c, cells in objs:
            for p in cells:
                comp_size[p] = len(cells)
        if xy is not None:
            x, y = xy
            for o in objs:
                if (y, x) in o[1]:
                    clicked_obj = o
                    break
        fa = self.act_features(aid, xy, g, comp_size)
        aid_hot = one_hot(aid - 1, 7)
        cells_self = self.locate_self(g)
        fahead, new_self = self.ahead(g, aid, cells_self)
        flife = self.life_features(g)
        for t, f in (("effect", fa), ("death", np.concatenate([fa, fahead, flife])), ("level", fa),
                     ("reset", np.concatenate([fa, fahead, flife])), ("clock_tick", fa),
                     ("novel", np.concatenate([fa, fahead]))):
            out.append((t, None, f, self.heads[t].predict(f)))
        # the self
        moves_p = None
        if cells_self is not None:
            f = np.concatenate([aid_hot, fahead])
            moves_p = self.heads["self_moves"].predict(f)
            out.append(("self_moves", None, f, moves_p))
            d = self.mode_disp(aid) if moves_p > 0 and new_self is not None else (0, 0)
            pred_self = new_self if (moves_p > 0 and new_self is not None) else cells_self
            out.append(("self_pos", d, None, 1.0))
            out.append(("self_pos_base", (0, 0), None, 1.0))     # the baseline: the self stays
            touch = self._touches(g, pred_self, bg)
            out.append(("contact", None, None, 1.0 if touch else -1.0))
        else:
            pred_self = None
        if light:
            return out
        # active cells
        act = self.active(g.shape)
        clk = self.clock(g.shape)
        last = self.last_diff if self.last_diff is not None and self.last_diff.shape == g.shape else None
        rate = (self.change_count / max(1, self.n_trans)) if act.any() else None
        ys, xs = np.nonzero(act)
        for y, x in list(zip(ys.tolist(), xs.tolist()))[:MAX_CELLS]:
            nb = 0
            if last is not None:
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < g.shape[0] and 0 <= xx < g.shape[1] and last[yy, xx]:
                        nb += 1
            in_self = cells_self is not None and (y, x) in cells_self
            near_self = cells_self is not None and any(abs(y - p[0]) + abs(x - p[1]) == 1 for p in cells_self)
            f = np.concatenate([aid_hot, [rate[y, x], float(last is not None and last[y, x]), nb / 4.0,
                                          float(clk[y, x]), float(in_self), float(near_self),
                                          float(xy is not None and (y, x) == (xy[1], xy[0])),
                                          float(clicked_obj is not None and (y, x) in clicked_obj[1]),
                                          float(pred_self is not None and (y, x) in pred_self),
                                          float(int(g[y, x]) == bg)]])
            p = self.heads["cell_change"].predict(f)
            out.append(("cell_change", (y, x), f, p))
            out.append(("cell_value", (y, x), None, self._value(y, x, int(g[y, x]))))
        # objects
        sc = self.self_colour
        for c, cells in objs:
            if cells_self is not None and cells == cells_self:
                continue
            touched = pred_self is not None and bool(cells & pred_self)
            adj = pred_self is not None and any((y + dy, x + dx) in cells for y, x in pred_self
                                                 for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            ch = last is not None and any(last[p] for p in cells)
            f = np.concatenate([aid_hot, [float(clicked_obj is not None and clicked_obj[1] == cells),
                                          float(clicked_obj is not None and clicked_obj[0] == c),
                                          float(touched), float(adj), math.log1p(len(cells)) / 5.0,
                                          float(len(cells) == 1), float(ch), float(c == sc)], one_hot(c, 16)])
            for t in ("obj_gone", "obj_moved", "obj_recolour", "obj_reshape"):
                out.append((t, (c, cells), f, self.heads[t].predict(f)))
        return out

    def life_features(self, g):
        f = np.zeros(N_LIFE)
        f[0] = math.log1p(self.life) / 7.0
        if self.lives:
            m = sum(self.lives[-8:]) / len(self.lives[-8:])
            f[1] = min(2.0, self.life / max(1.0, m))
            f[2] = 1.0 if self.life + 1 >= 0.9 * min(self.lives[-8:]) else 0.0
        clk = self.clock(g.shape)
        if clk.any() and self.life_start is not None and self.life_start.shape == g.shape:
            f[3] = float((clk & (g != self.life_start)).sum()) / float(clk.sum())
        else:
            f[4] = 1.0
        return f

    def _value(self, y, x, old):
        """The predicted new value of a cell if it changes (learned table: this cell's own history, then the game's)."""
        cnt = self.value_cell.get((y, x, old)) or self.value_game.get(old)
        return cnt.most_common(1)[0][0] if cnt else None

    @staticmethod
    def _touches(g, cells, bg):
        h, w = g.shape
        for y, x in cells:
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and (yy, xx) not in cells and int(g[yy, xx]) != bg:
                    return True
        return False

    def before(self, latest, act):
        """Called with the frame the agent saw and the action it chose: predicts and keeps the facts."""
        self.pending = None
        if not getattr(latest, "frame", None) or int(getattr(act, "value", act)) == 0:
            return
        g = self.grid_of(latest.frame[-1])
        aid, xy = self.action_of(act)
        if self.start is None:
            self.start = g.copy()
        if self.life_start is None:
            self.life_start = g.copy()
        fs = self.expect(g, aid, xy, light=self.lite)
        self.pending = dict(frame=np.asarray(latest.frame[-1], dtype=np.int64), g=g, aid=aid, xy=xy,
                            G=self.G, facts=fs, cells_self=self.locate_self(g), img=self.imagine(g, fs))

    # ------------------------------------------------------------------------------------------- check
    def after(self, latest):
        """Called with the next frame: checks every pending fact, learns, updates the level state."""
        from arcengine import GameState
        p, self.pending = self.pending, None
        if p is None:
            return
        self.cur = collections.Counter()
        lv = int(getattr(latest, "levels_completed", 0) or 0)
        st = getattr(latest, "state", None)
        dead = st is GameState.GAME_OVER
        up = lv > self.level
        conf = self.conf[self.level]
        self.steps[self.level] += 1
        g0 = p["g"]
        g1 = None
        if not dead and not up and getattr(latest, "frame", None) and self.G is p["G"]:
            g1 = self.read(np.asarray(latest.frame[-1], dtype=np.int64))
            if g1.shape != g0.shape:
                g1 = None
        truth = dict(death=dead, level=up)
        if g1 is not None:
            d = g0 != g1
            clk = self.clock(g0.shape)
            act = self.active(g0.shape)
            truth["effect"] = bool((d & ~clk).any())
            truth["clock_tick"] = bool((d & clk).any())
            truth["novel"] = bool((d & ~act).any())
            st0 = self.start if self.start is not None and self.start.shape == g0.shape else None
            truth["reset"] = st0 is not None and bool(((g1 == st0) | clk).all()) and not bool(((g0 == st0) | clk).all())
        facts = p["facts"]
        for t, key, f, pr in facts:
            if t in ("death", "level"):
                self._score(conf, t, f, pr, truth[t])
            elif g1 is None:
                continue
            elif t in ("effect", "clock_tick", "novel", "reset"):
                self._score(conf, t, f, pr, truth[t])
            elif t == "cell_change":
                self._score(conf, t, f, pr, bool(d[key]))
            elif t == "cell_value":
                y, x = key
                if d[y, x]:
                    self._add(conf, t, pr is not None and pr == int(g1[y, x]), True, 1.0)
        if g1 is not None:
            img, unk, flags = p["img"]
            m = self.img[self.level]
            known_ok = bool((img[~unk] == g1[~unk]).all())
            m["n"] += 1
            m["known_right"] += known_ok                       # every committed cell right
            m["full_right"] += known_ok and not unk.any()      # the whole frame committed and right
            m["unknown_cells"] += int(unk.sum())
            m["cells"] += int(unk.size)
            m["wrong_cells"] += int((img[~unk] != g1[~unk]).sum())
            m["persist_right"] += bool((g0 == g1).all())       # the baseline: nothing changes
            m["novel_unknown"] += flags["novel"] is not False
            m["death_verdict"] += flags["death"] is not None
            m["death_verdict_right"] += flags["death"] is not None and flags["death"] == dead
            self._check_self_and_objects(conf, p, g0, g1, d, facts)
            # learn the level's dynamics (after scoring: predictions never see their own answer)
            if self.change_count is None or self.change_count.shape != d.shape:
                self.change_count, self.n_trans = np.zeros(d.shape), 0
            self.change_count += d
            self.n_trans += 1
            self.last_diff = d
            for y, x in zip(*np.nonzero(d)):
                old, new = int(g0[y, x]), int(g1[y, x])
                self.value_cell[(int(y), int(x), old)][new] += 1
                self.value_game[old][new] += 1
        for t, f, yes in ([] if self.frozen else self.replay[-8:]):   # arm +4: replay the latest refuted confident facts
            self.heads[t].learn(f, yes, times=self.replay_surprise)
        self.replay = []
        tot = sum(self.steps.values())
        c = self.cur
        c["steps"] += 1
        c["wasted"] += int(not c["refuted"] and c["err_drop"] < 1e-3 and not up)
        self.info[(tot - 1) // 100].update(c)
        if up:
            self.level_up_at.append(tot)
        if tot in (100, 300):                               # U at a fixed horizon: agents compared on equal data
            u, _ = understanding(self.game)
            self.horizon[str(tot)] = dict(U=_r(u), auc={t: _r(c.aucf()) for t, c in self.game.items()})
        self.life += 1
        if up or dead:
            if dead:
                self.lives.append(self.life)
            self.life, self.life_start = 0, None
        if up:
            self.level = lv
            self._new_level()
        elif dead:
            self.self_cells = None

    def _add(self, conf, t, pred, truth, c, p=None, raw=None):
        conf[t].add(pred, truth, c, p, raw)
        self.game[t].add(pred, truth, c, p, raw)
        self.win[sum(self.steps.values()) // WIN][t].add(pred, truth, c, p, raw)
        self.cur["predicted"] += 1
        self.cur["confirmed" if pred == truth else "refuted"] += 1

    def _score(self, conf, t, f, pr, yes):
        pred = pr > 0
        self._add(conf, t, pred, yes, abs(pr), prob(pr), pr)
        P = prob(pr)
        self.cur["bits"] += -math.log2(max(1e-3, P if yes else 1.0 - P))
        if self.frozen:
            return
        self.heads[t].learn(f, yes)
        tgt = 0.9 if yes else -0.9
        self.cur["err_drop"] += (tgt - pr) ** 2 - (tgt - self.heads[t].predict(f)) ** 2
        if self.replay_surprise and pred != yes and abs(pr) >= 0.5:
            self.replay.append((t, f, yes))

    def _check_self_and_objects(self, conf, p, g0, g1, d, facts):
        aid = p["aid"]
        # which colours moved rigidly: votes for the self (non-click actions only)
        moved = {}
        for c in [int(v) for v in np.unique(g0[d])] if d.any() else []:
            tr = muse.translation(g0, g1, c)
            if tr is not None:
                moved[c] = tr
        if aid != 6:
            for c in moved:
                self.self_votes[c] += 1
        cs = p["cells_self"]
        sc = self.self_colour
        if cs is not None and sc is not None:
            col = next(iter(g0[y, x] for y, x in cs))
            disp = moved.get(int(col), (0, 0)) if int(col) == sc else (0, 0)
            if int(col) == sc and disp == (0, 0) and any(d[y, x] for y, x in cs):
                disp = None                                 # changed but not rigidly: neither moved nor still
            for t, key, f, pr in facts:
                if t == "self_moves" and disp is not None:
                    self._score(conf, t, f, pr, disp != (0, 0))
                elif t in ("self_pos", "self_pos_base") and disp is not None:
                    self._add(conf, t, True, key == disp, 1.0)
                elif t == "contact" and disp is not None:
                    actual = frozenset((y + disp[1], x + disp[0]) for y, x in cs)
                    if all(0 <= y < g1.shape[0] and 0 <= x < g1.shape[1] for y, x in actual):
                        self._add(conf, t, pr > 0, self._touches(g1, actual, background(g1)), 1.0, prob(pr), pr)
            if disp is not None and disp != (0, 0) and aid != 6:
                self.disp[aid][disp] += 1
            if disp is not None:
                self.self_cells = frozenset((y + disp[1], x + disp[0]) for y, x in cs)
        elif cs is None and sc is not None:
            self.self_cells = None
        # objects (the new grid's components, once per colour)
        cache = {}

        def comps(c):
            if c not in cache:
                cache[c] = [frozenset(cp) for cp in muse._components(g1 == c)]
            return cache[c]
        for t, key, f, pr in facts:
            if not t.startswith("obj_"):
                continue
            c, cells = key
            self._score(conf, t, f, pr, self._obj_truth(t, c, cells, g0, g1, comps))

    @staticmethod
    def _obj_truth(t, c, cells, g0, g1, comps):
        keep = [p for p in cells if g1[p] == c]
        if t == "obj_recolour":
            vals = {int(g1[p]) for p in cells}
            return len(vals) == 1 and vals != {c}
        shape, (y0, x0) = shape_key(cells)
        if t in ("obj_moved", "obj_gone"):
            found_moved = False
            for comp in comps(c):
                if len(comp) != len(cells):
                    continue
                s2, (y1, x1) = shape_key(comp)
                if s2 == shape and (y1, x1) != (y0, x0):
                    found_moved = True
                    break
            if t == "obj_moved":
                return found_moved and len(keep) < len(cells)
            return not keep and not found_moved
        # obj_reshape: the component of colour c over the object's kept cells is another set, not a translation
        if not keep:
            return False
        if len(keep) == len(cells):
            comp = next((cp for cp in comps(c) if keep[0] in cp), None)
            return comp is not None and comp != cells
        return True

    # ------------------------------------------------------------------------------------------- rates and report
    def rate(self, t):
        """A fact type's verified rate over the game so far: its AUC (a learned type) or its accuracy (a composed
        one), None while it has fewer than 20 checks or (AUC) has not seen both answers."""
        c = self.game.get(t)
        if c is None or c.n < 20:
            return None
        return c.acc() if t in ("self_pos", "self_pos_base", "cell_value") else c.auc()

    def trusted(self, t):
        r = self.rate(t)
        return r is not None and r >= self.tau

    def verdict(self, t, p):
        """A fact's answer when it is TRUSTED, else None. Trust is verified, not assumed: predictions of type t that
        fell in the same probability bin (of 20) were right >= tau of the time over this game, with >= 10 checks."""
        c = self.game.get(t)
        if c is None:
            return None
        b = min(NB - 1, max(0, int(prob(p) * NB)))
        n = c.pos[b] + c.neg[b]
        if n < 10:
            return None
        if c.pos[b] >= self.tau * n:
            return True
        if c.neg[b] >= self.tau * n:
            return False
        return None

    def bin_rate(self, t, p):
        """The share of "yes" among this game's checked predictions of type t in p's probability bin (Laplace), or
        None under 10 checks: a verified probability, where `verdict` is a verified certainty."""
        c = self.game.get(t)
        if c is None:
            return None
        b = min(NB - 1, max(0, int(prob(p) * NB)))
        n = c.pos[b] + c.neg[b]
        return (c.pos[b] + 0.5) / (n + 1.0) if n >= 10 else None

    def imagine(self, g, facts):
        """The next frame composed from TRUSTED facts only: (grid, unknown mask, flags). A cell is committed only
        when every fact touching it has a verified answer (`verdict`); everything else is unknown."""
        img, unk = g.copy(), np.zeros(g.shape, bool)
        vals = {k: p for t, k, f, p in facts if t == "cell_value"}
        value_ok = (self.rate("cell_value") or 0) >= self.tau
        for t, key, f, p in facts:
            if t == "cell_change":
                v = self.verdict(t, p)
                if v is False:
                    continue
                if v is True and value_ok and vals.get(key) is not None:
                    img[key] = vals[key]
                else:
                    unk[key] = True
            elif t.startswith("obj_"):
                if self.verdict(t, p) is not False:
                    for q in key[1]:
                        unk[q] = True
        pr = {t: p for t, k, f, p in facts if k is None and f is not None}
        novel = self.verdict("novel", pr["novel"]) if "novel" in pr else None
        if novel is not False:
            unk |= ~self.active(g.shape)                    # a cell that never changed may change: unknown
        moves = pr.get("self_moves")
        disp = next((k for t, k, f, p in facts if t == "self_pos"), None)
        cs = self.locate_self(g)
        if cs is not None and moves is not None:
            v = self.verdict("self_moves", moves)
            pos_ok = (self.rate("self_pos") or 0) >= self.tau
            if v is True and pos_ok and disp not in (None, (0, 0)):
                dx, dy = disp
                new = {(y + dy, x + dx) for y, x in cs}
                if all(0 <= y < g.shape[0] and 0 <= x < g.shape[1] for y, x in new):
                    c = g[next(iter(cs))]
                    for q in cs:
                        if q not in new:
                            unk[q] = True                   # what the self uncovers is not predicted
                    for q in new:
                        img[q], unk[q] = c, False
                else:
                    v = None
            if v is None or (v is True and not pos_ok):
                for y, x in cs:
                    for yy, xx in ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= yy < g.shape[0] and 0 <= xx < g.shape[1]:
                            unk[yy, xx] = True
        flags = dict(novel=novel, death=self.verdict("death", pr.get("death", 0.0)),
                     level=self.verdict("level", pr.get("level", 0.0)))
        return img, unk, flags

    def report(self):
        tot = collections.defaultdict(Confusion)
        levels = {}
        for lv, cs in sorted(self.conf.items()):
            for t, c in cs.items():
                tot[t].merge(c)
            u, ua = understanding(cs)
            levels[str(lv)] = dict(U=_r(u), U_acc=_r(ua), steps=self.steps[lv], imagine=dict(self.img[lv]),
                                   types={t: c.row() for t, c in sorted(cs.items())})
        u, ua = understanding(tot)
        im = collections.Counter()
        for c in self.img.values():
            im.update(c)
        return dict(U=_r(u), U_acc=_r(ua), steps=sum(self.steps.values()), self_colour=self.self_colour,
                    imagine=dict(im), tau=self.tau, horizon=self.horizon, level_up_at=self.level_up_at,
                    info={str(b): {k: round(v, 3) for k, v in c.items()} for b, c in sorted(self.info.items())},
                    windows={str(w): dict(U=_r(understanding(cs)[0]),
                                          auc={t: _r(c.aucf()) for t, c in cs.items() if c.aucf() is not None})
                             for w, cs in sorted(self.win.items())},
                    types={t: c.row() for t, c in sorted(tot.items())}, levels=levels)


def probe(v, path, steps=300):
    """Score v's readouts, frozen, on a shared reference trajectory (`path`, an .npz written by `Verified` with
    HY_RECORD): the understanding score of what v learned, on the SAME transitions for every agent. v is not changed:
    the probe has its own level state and copies of v's tables; its readouts are v's, never trained."""
    import copy
    from arcengine import GameAction, GameState
    z = np.load(path)
    q = Verifier()
    q.heads, q.frozen = v.heads, True
    q.self_votes, q.disp = copy.deepcopy(v.self_votes), copy.deepcopy(v.disp)
    q.value_game, q.lives = copy.deepcopy(v.value_game), list(v.lives)

    class F:
        pass
    n = min(steps, len(z["actions"]))
    for i in range(n):
        f = F()
        f.frame, f.state, f.levels_completed = [z["frames"][i]], GameState[str(z["states"][i])], int(z["levels"][i])
        q.after(f)
        a = GameAction.from_id(int(z["actions"][i]))
        if int(z["actions"][i]) == 6:
            a.set_data({"x": int(z["xy"][i][0]), "y": int(z["xy"][i][1])})
        if f.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            q.pending = None
            continue
        q.before(f, a)
    u, ua = understanding(q.game)
    return dict(U=_r(u), U_acc=_r(ua), steps=n, auc={t: _r(c.aucf()) for t, c in q.game.items()})


class Verified:
    """Wraps a policy (`choose_action(frames, latest)` / `is_done`) with a Verifier that observes every step. The
    Verifier is exposed as `.agent` with `stats()` and `carry()`, the hooks the gate's `replay.py` reads into its row."""

    def __init__(self, policy, verifier, game_id=""):
        self.policy, self.v, self.game_id = policy, verifier, game_id
        self.agent = self
        self.record = os.environ.get("HY_RECORD", "")         # write this game's trajectory (the probe set)
        self.probe_dir = os.environ.get("HY_PROBE", "")       # score the readouts on a recorded trajectory
        self.tape = []
        self.probes = {}

    def is_done(self, frames, latest):
        return self.policy.is_done(frames, latest)

    def choose_action(self, frames, latest):
        try:
            self.v.after(latest)
        except Exception as e:  # the observer must never break the agent
            self._error(e)
        if self.probe_dir and sum(self.v.steps.values()) == 300 and "300" not in self.probes:
            self._probe("300")
        act = self.policy.choose_action(frames, latest)
        if self.record and getattr(latest, "frame", None):
            d = act.action_data.model_dump() if hasattr(act, "action_data") else {}
            self.tape.append((np.asarray(latest.frame[-1], dtype=np.int8), latest.state.name,
                              int(latest.levels_completed or 0), int(act.value), (int(d.get("x", 0)), int(d.get("y", 0)))))
        try:
            self.v.before(latest, act)
        except Exception as e:
            self._error(e)
        return act

    def _error(self, e):
        import traceback
        self.v.pending = None
        self.v.errors += 1
        self.v.last_error = traceback.format_exc()[-600:]

    def _probe(self, key):
        path = os.path.join(self.probe_dir, f"{self.game_id}.npz")
        if os.path.exists(path):
            try:
                self.probes[key] = probe(self.v, path)
            except Exception as e:
                self._error(e)

    def stats(self):
        if self.record and self.tape:
            os.makedirs(self.record, exist_ok=True)
            fr, st, lv, ac, xy = zip(*self.tape)
            np.savez_compressed(os.path.join(self.record, f"{self.game_id}.npz"), frames=np.stack(fr),
                                states=np.array(st), levels=np.array(lv), actions=np.array(ac), xy=np.array(xy))
        if self.probe_dir:
            self._probe("end")
        r = self.v.report()
        r["probe"] = self.probes
        if hasattr(self.policy, "hy_stats"):
            r["hy"] = dict(self.policy.hy_stats)
        if getattr(self.policy, "trace", None):
            r["hy_trace"] = self.policy.trace
        r["errors"] = self.v.errors
        if getattr(self.v, "last_error", None):
            r["last_error"] = self.v.last_error
        return r

    def carry(self):
        return json.dumps({})
