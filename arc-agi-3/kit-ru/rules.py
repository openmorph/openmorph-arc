# SPDX-License-Identifier: MIT-0
"""Stream RU: an object-level rule model, learned from every transition and kept across levels.

Kit DG13 learns STATES (a graph of whole-screen hashes, wiped at every level). This module learns RULES: what an
action does to the objects, keyed by the self's relation to the object it meets, so a rule seen on level 1 predicts a
screen never seen on level 2.

    m = RuleModel()
    m.observe(g0, action, g1, died=False)        # every transition (g1 the next grid; None after a death)
    m.observe_win(g0, action, g_final)           # the transition that completed a level (g_final = frame[0])
    g1_hat, info = m.predict(g0, action)         # the predicted next grid, or None when no rule covers it

The world, as this module reads it (AUTHORED, game-blind):
* OBJECTS: 4-connected components of one colour on the logical grid; the background is the most common colour; the
  type of an object is its colour (colours keep their meaning across the levels of one game). Cells < 0 are masked.
* THE SELF: the colour that moves rigidly most often under non-click actions (HY's rule); its displacement per action
  is the mode of the observed ones, stored in units of the self's own extent (a level may draw cells larger).
* THE RELATION (the rule's key): for a move, the colour the self would enter (`ahead`), the colour one step further
  (`ahead2`), and whether each is the last object of its colour; for a click, the clicked colour; for any other
  action, the action alone. Keys back off from the full relation to `ahead` alone.
* THE EFFECT (the rule's outcome), relative to the self: how far the self went (0, 1 step, or `slide`: until the next
  non-background cell), what happened to the object ahead (`same`, `push`: moved one step with the self, `gone`,
  `recolour` to a colour), the remote colour changes (old -> new, `all` cells of the old colour or a count) and death.
LEARNED: the self, its displacements, every rule's outcome counts and its VERIFIED rate (each rule predicts before
it learns; a prediction is checked against the next grid), and the win condition (`WinModel`).
"""
import collections
from fractions import Fraction

import numpy as np

EDGE = -2          # outside the grid
MASK = -1          # a masked cell (the clock)


# ------------------------------------------------------------------------------------------------ objects (AUTHORED)
def background(g):
    v = g[g >= 0]
    if not v.size:
        return 0
    vals, cnt = np.unique(v, return_counts=True)
    return int(vals[np.argmax(cnt)])


try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover - the pure-python reading below is the same components
    _ndi = None


def components(mask):
    """4-connected components of a boolean mask: a label array (-1 outside) and the list of (ys, xs) arrays."""
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        lab = lab.astype(np.int32) - 1
        if n == 0:
            return lab, []
        ys, xs = np.nonzero(lab >= 0)
        ids = lab[ys, xs]
        order = np.argsort(ids, kind="stable")
        ys, xs, ids = ys[order], xs[order], ids[order]
        cuts = np.flatnonzero(np.diff(ids)) + 1
        # components ordered by their first cell in row-major order (as the python reading below)
        comps = list(zip(np.split(ys, cuts), np.split(xs, cuts)))
        return lab, comps
    h, w = mask.shape
    lab = -np.ones((h, w), np.int32)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0] >= 0:
            continue
        i = len(comps)
        stack = [(y0, x0)]
        lab[y0, x0] = i
        cy, cx = [], []
        while stack:
            y, x = stack.pop()
            cy.append(y)
            cx.append(x)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = i
                    stack.append((yy, xx))
        comps.append((np.array(cy), np.array(cx)))
    return lab, comps


def surround(g, ys, xs, exclude=()):
    """The most common colour 4-adjacent to the cells (ys, xs), outside them and not in `exclude` (masked excluded)."""
    h, w = g.shape
    own = set(zip(ys.tolist(), xs.tolist()))
    c = collections.Counter()
    for y, x in own:
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and (yy, xx) not in own:
                v = int(g[yy, xx])
                if v >= 0 and v not in exclude:
                    c[v] += 1
    return c.most_common(1)[0][0] if c else None


def colour_at(g, y, x):
    h, w = g.shape
    return int(g[y, x]) if 0 <= y < h and 0 <= x < w else EDGE


class Scene:
    """One grid read as objects (cached per grid)."""

    def __init__(self, g):
        self.g = g
        self.bg = background(g)
        v = g[g >= 0].ravel()
        vals, first, cnt = np.unique(v, return_index=True, return_counts=True)
        order = np.argsort(first, kind="stable")             # first-appearance order, as a Counter built by a scan
        self.counts = collections.Counter({int(vals[i]): int(cnt[i]) for i in order})
        self._lab = {}
        self._n = {}
        self._self = {}

    def comps(self, c):
        if c not in self._lab:
            self._lab[c] = components(self.g == c)
        return self._lab[c]

    def object_at(self, y, x):
        """(colour, ys, xs) of the component through (y, x), or None for background / outside / masked."""
        c = colour_at(self.g, y, x)
        if c < 0 or c == self.bg:
            return None
        lab, comps = self.comps(c)
        ys, xs = comps[lab[y, x]]
        return c, ys, xs

    def n_objects(self, c):
        if c in self._lab:
            return len(self._lab[c][1])
        if c not in self._n:
            self._n[c] = _ndi.label(self.g == c)[1] if _ndi is not None else len(self.comps(c)[1])
        return self._n[c]


# ------------------------------------------------------------------------------------------------ the model
def _sign(v):
    return (v > 0) - (v < 0)


class RuleModel:
    def __init__(self):
        self.self_votes = collections.Counter()         # colour -> rigid moves under non-click actions
        self.disp = collections.defaultdict(collections.Counter)   # action -> Counter((dy/h, dx/w) as Fractions)
        self.rules = collections.defaultdict(collections.Counter)  # key -> Counter(outcome)
        self.checked = collections.defaultdict(lambda: [0, 0])     # key -> [right, checked] (verified rate)
        self.self_cells = None                                       # (ys, xs) of the self last seen
        self.floor = None                                            # the static layer: last colour seen per cell
        self.movers = set()                                          # colours seen pushed (they leave floor behind)
        self.pair_steps = collections.Counter()                      # remote (old, new) -> steps it happened in
        self.n_steps = 0                                              # steps read (the autonomous rate's base)
        self.ptr = {}                                                 # (old, new) -> (last cells, step delta)
        self._rem = {}
        self.pair_noop = collections.Counter()                       # ... steps where the action itself did nothing
        self.pair_keys = collections.defaultdict(set)                # ... the rule keys it came under
        self.pair_gap = collections.defaultdict(collections.Counter)  # ... steps between two of its changes
        self.pair_last = {}                                           # ... the step it last happened
        self.ptr_ok = collections.defaultdict(lambda: [0, 0])         # ... its pointer's predictions [right, made]
        self.n_obs = 0
        self.log = collections.Counter()

    # ---- the self ------------------------------------------------------------------------------------------------
    @property
    def self_colour(self):
        if not self.self_votes:
            return None
        c, v = self.self_votes.most_common(1)[0]
        return c if v >= 1.5 else None

    def locate_self(self, sc):
        c = self.self_colour
        sk = self.self_cells
        key = (c, None if sk is None else (sk[0].tobytes(), sk[1].tobytes()))
        hit = sc._self.get(key, False)
        if hit is not False:
            return hit                                   # the same reading of the same scene (a search's actions)
        if c is None or not (sc.g == c).any():
            sc._self[key] = None
            return None
        lab, comps = sc.comps(c)
        if sk is not None and len(sk[0]):
            cy, cx = sk[0].mean(), sk[1].mean()
        else:
            cy, cx = sc.g.shape[0] / 2, sc.g.shape[1] / 2
        out = min(comps, key=lambda p: (np.min((p[0] - cy) ** 2 + (p[1] - cx) ** 2), -len(p[0])))
        sc._self[key] = out
        return out

    @staticmethod
    def extent(cells):
        ys, xs = cells
        return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)

    def step_of(self, a, cells):
        """The self's usual displacement (dy, dx) in cells for action a, scaled to its current extent; None if unknown."""
        c = self.disp.get(a)
        if not c or cells is None:
            return None
        (fy, fx), _ = c.most_common(1)[0]
        h, w = self.extent(cells)
        dy, dx = fy * h, fx * w
        if dy.denominator != 1 or dx.denominator != 1:
            return None
        return int(dy), int(dx)

    # ---- the relation: what the self meets -----------------------------------------------------------------------
    def relation(self, sc, a, xy=None):
        """(keys from specific to general, context) for action a in scene sc."""
        if a == 6:
            if xy is None:
                return [("click", None)], dict(kind="click")
            x, y = xy
            o = sc.object_at(y, x)
            c = colour_at(sc.g, y, x)
            last = o is not None and sc.n_objects(o[0]) == 1
            return [("click", c, last), ("click", c)], dict(kind="click", obj=o, xy=xy)
        cells = self.locate_self(sc)
        d = self.step_of(a, cells)
        if d is None or d == (0, 0):
            return [("act", a)], dict(kind="act", cells=cells, d=d)
        ys, xs = cells
        dy, dx = d
        own = set(zip(ys.tolist(), xs.tolist()))
        ahead = [(y + dy, x + dx) for y, x in own if (y + dy, x + dx) not in own]
        # a jump (a move longer than the self): the colour passed over decides as much as where it lands
        n = max(abs(dy), abs(dx))
        via = None
        if n > 1:
            sy, sx = _sign(dy), _sign(dx)
            land = set(ahead)
            vc = collections.Counter(colour_at(sc.g, y + j * sy, x + j * sx) for y, x in own for j in range(1, n)
                                     if (y + j * sy, x + j * sx) not in own and (y + j * sy, x + j * sx) not in land)
            if vc:
                via = vc.most_common(1)[0][0]
                via = "bg" if via == sc.bg else via
        cols = collections.Counter(colour_at(sc.g, y, x) for y, x in ahead)
        c1 = cols.most_common(1)[0][0] if cols else EDGE
        o1 = None
        if c1 >= 0 and c1 != sc.bg:
            y, x = next(p for p in ahead if colour_at(sc.g, *p) == c1)
            o1 = sc.object_at(y, x)
        # one step beyond the object ahead (or beyond the self when the way is free)
        if o1 is not None:
            oc = set(zip(o1[1].tolist(), o1[2].tolist()))
            beyond = [(y + dy, x + dx) for y, x in oc if (y + dy, x + dx) not in oc and (y + dy, x + dx) not in own]
        else:
            beyond = [(y + 2 * dy, x + 2 * dx) for y, x in own if (y + 2 * dy, x + 2 * dx) not in own]
        cols2 = collections.Counter(colour_at(sc.g, y, x) for y, x in beyond)
        c2 = cols2.most_common(1)[0][0] if cols2 else EDGE
        k1 = "bg" if c1 == sc.bg else c1
        k2 = "bg" if c2 == sc.bg else c2
        last1 = o1 is not None and sc.n_objects(o1[0]) == 1
        o2 = None
        if c2 >= 0 and c2 != sc.bg and beyond:
            y, x = next(p for p in beyond if colour_at(sc.g, *p) == c2)
            o2 = sc.object_at(y, x)
        last2 = o2 is not None and sc.n_objects(o2[0]) == 1
        if via is not None:
            k1 = (via, k1)                             # a jump: what it passes over, then where it lands
        keys = [("mv", k1, k2, last1, last2), ("mv", k1, k2)]
        cls2 = self.solidity(k2)
        if cls2 is not None:
            keys.append(("mv", k1, cls2))              # what lies beyond, by what it does to the self (generalises)
        keys.append(("mv", k1))
        return keys, dict(kind="mv", cells=cells, d=d, o1=o1, c1=c1, c2=c2)

    def solidity(self, k):
        """How a colour treats the self, learned: `solid` (the self stays), `free` (it enters), `edge`, else None."""
        if k == EDGE:
            return "edge"
        c = self.rules.get(("mv", k))
        if not c:
            return None
        moves = sum(n for o, n in c.items() if o[0] not in ("death",) and o[0] != 0)
        stays = sum(n for o, n in c.items() if o[0] == 0)
        if moves + stays < 2:
            return None
        return "free" if moves > stays else "solid"

    # ---- the effect of an observed transition ----------------------------------------------------------------------
    def outcome(self, sc0, g1, ctx, died):
        """The observed effect, in the model's terms (a hashable tuple), or None when it cannot be read."""
        if died:
            return ("death",)
        g0 = sc0.g
        if g1 is None or g1.shape != g0.shape:
            return None
        valid = (g0 >= 0) & (g1 >= 0)
        diff = (g0 != g1) & valid
        local = np.zeros(g0.shape, bool)
        k, eff1 = 0, "same"
        if ctx["kind"] == "mv":
            ys, xs = ctx["cells"]
            dy, dx = ctx["d"]
            sc1 = Scene(g1)
            c = self.self_colour
            # how far the self went along d (0 .. the grid's size)
            span = max(g0.shape)
            k = None
            for j in range(0, span):
                yy, xx = ys + j * dy, xs + j * dx
                if yy.min() < 0 or xx.min() < 0 or yy.max() >= g0.shape[0] or xx.max() >= g0.shape[1]:
                    break
                if (g1[yy, xx] == c).all():
                    k = j
                    if j > 0:
                        break
                    # stayed? only if no displaced copy exists further along
                    continue
            if k is None:
                return None
            # the self's old and new cells are local
            for j in (0, k):
                local[np.clip(ys + j * dy, 0, g0.shape[0] - 1), np.clip(xs + j * dx, 0, g0.shape[1] - 1)] = True
            if k > 1:
                # a slide: the cells passed over are local too, and the slide is `slide` if it stopped at a non-bg cell
                for j in range(1, k):
                    local[ys + j * dy, xs + j * dx] = True
                nxt = [colour_at(g0, y, x) for y, x in zip((ys + (k + 1) * dy).tolist(), (xs + (k + 1) * dx).tolist())]
                k = "slide" if any(v != sc0.bg and v not in (MASK,) for v in nxt if (v != c)) else k
            o1 = ctx["o1"]
            if o1 is not None:
                col, oy, ox = o1
                local[oy, ox] = True
                ny, nx = oy + dy, ox + dx
                inside = ny.min() >= 0 and nx.min() >= 0 and ny.max() < g0.shape[0] and nx.max() < g0.shape[1]
                dest = g1[ny, nx] if inside else None
                moved = inside and k == 1 and not (g1[oy, ox] == col).all() and len(set(dest.tolist())) == 1 \
                    and int(dest[0]) != c and (g0[ny, nx] != dest).any() \
                    and (int(dest[0]) == col or (g0[ny, nx] != dest).all())
                if moved:
                    eff1 = "push" if int(dest[0]) == col else ("push", int(dest[0]))
                    local[ny, nx] = True
                elif (g1[oy, ox] == col).all():
                    eff1 = "same"
                else:
                    covered = (g1[oy, ox] == c)
                    rest = g1[oy, ox][~covered]
                    if (rest == col).all() and rest.size:
                        eff1 = "same"                          # partly under the self
                    elif rest.size == 0:
                        eff1 = "under"                         # wholly under the self: unknown
                    else:
                        vals = collections.Counter(int(v) for v in rest)
                        nv = vals.most_common(1)[0][0]
                        fl = surround(g0, oy, ox, (c, col))
                        eff1 = "gone" if nv in (sc1.bg, fl) else ("recolour", nv)
        elif ctx["kind"] == "click":
            o = ctx.get("obj")
            if o is not None:
                col, oy, ox = o
                local[oy, ox] = True
                vals = collections.Counter(int(v) for v in g1[oy, ox])
                nv = vals.most_common(1)[0][0]
                eff1 = "same" if nv == col else ("gone" if nv in (sc0.bg, surround(g0, oy, ox, (col,))) else ("recolour", nv))
        # remote changes: colour pairs, `all` when every cell of the old colour outside the local region changed
        rem = diff & ~local
        ry, rx = np.nonzero(rem)
        cells = collections.defaultdict(list)
        for y, x in zip(ry.tolist(), rx.tolist()):
            cells[(int(g0[y, x]), int(g1[y, x]))].append((y, x))
        self._rem = {p: frozenset(v) for p, v in cells.items()}
        remote = []
        for (o, n), v in sorted(cells.items()):
            if self.autonomous((o, n)):
                continue                                     # the world's own step, not the action's (Ticks)
            tot = int(((g0 == o) & ~local & valid).sum())
            remote.append((o, n, "all" if len(v) == tot else "some"))
        return (k, eff1, tuple(remote))

    # ---- the world's own steps (pointers) ------------------------------------------------------------------------
    def autonomous(self, p):
        """A remote change that happens whatever the action (a bar, a clock, a patrol): in most steps, or on steps
        where the action itself did nothing and under three or more relations."""
        if self.n_steps >= 6 and self.pair_steps[p] >= 0.5 * self.n_steps:
            return True
        return self.pair_noop[p] >= 2 and len(self.pair_keys[p]) >= 3

    def due(self, p):
        """Whether an autonomous change is due on the next step: its usual gap since its last one has passed."""
        gaps = self.pair_gap.get(p)
        if not gaps or p not in self.pair_last:
            return True
        gap = gaps.most_common(1)[0][0]
        return self.n_steps - self.pair_last[p] >= gap

    def learn_ticks(self, key=None, noop=False):
        """After a read step: count every remote pair, move its pointer (the cells it changed, and their shift)."""
        self.n_steps += 1
        for p, cs in self._rem.items():
            self.pair_steps[p] += 1
            self.pair_noop[p] += int(noop)
            if key is not None:
                self.pair_keys[p].add(key)
            if p in self.pair_last:
                self.pair_gap[p][self.n_steps - self.pair_last[p]] += 1
            self.pair_last[p] = self.n_steps
            last = self.ptr.get(p)
            delta = None
            if last is not None and len(last[0]) == len(cs):
                a = np.array(sorted(last[0]))
                b = np.array(sorted(cs))
                dd = b.mean(0) - a.mean(0)
                if np.allclose(dd, np.round(dd)):
                    delta = (int(round(dd[0])), int(round(dd[1])))
            self.ptr[p] = (cs, delta if delta is not None else (last[1] if last is not None else None))

    def tick_cells(self, g, p):
        """Where pair p = (old, new) changes next: its last cells shifted by its last step, among cells of the old
        colour; reversed when that is empty (a bounce); the last cells' neighbours of the old colour otherwise."""
        last = self.ptr.get(p)
        if last is None:
            return []
        cs, d = last
        h, w = g.shape
        o = p[0]

        def ok(c):
            return [(y, x) for y, x in c if 0 <= y < h and 0 <= x < w and g[y, x] == o]
        if d is not None and d != (0, 0):
            for s in (1, -1):
                c = ok([(y + s * d[0], x + s * d[1]) for y, x in cs])
                if len(c) == len(cs):
                    return c
        return []

    # ---- learning --------------------------------------------------------------------------------------------------
    def learn_self(self, g0, a, g1):
        """HY's self rule: colours that moved rigidly under a non-click action vote; the self's displacement."""
        if g1 is None or g1.shape != g0.shape or a in (0, 6):
            return
        d = (g0 != g1) & (g0 >= 0) & (g1 >= 0)
        if not d.any():
            return
        for c in set(int(v) for v in g0[d]) | set(int(v) for v in g1[d]):
            t = _translation(g0, g1, c)
            if t is not None and _whole(g0, g1, c, t):
                # one vote per step, shared by the colour's objects: the self is one thing, a dotted floor many
                self.self_votes[c] += 1.0 / max(1, len(components(g0 == c)[1]))
        c = self.self_colour
        if c is None:
            return
        t = _translation(g0, g1, c)
        sc = Scene(g0)
        cells = self.locate_self(sc)
        if t is not None and cells is not None:
            h, w = self.extent(cells)
            dy, dx = t
            self.disp[a][(Fraction(dy, h), Fraction(dx, w))] += 1

    def observe(self, g0, a, g1, died=False, xy=None):
        """Learn from one transition: check the prediction first (the verified rate), then count the outcome."""
        self.n_obs += 1
        if not died:
            self.learn_self(g0, a, g1)
        sc0 = Scene(g0)
        keys, ctx = self.relation(sc0, a, xy)
        out = self.outcome(sc0, g1, ctx, died)
        if out is None:
            self.log["unreadable"] += 1
            return None
        pred = self.best(keys)
        if pred is not None:
            key, o, _ = pred
            r = self.checked[key]
            r[0] += int(o == out)
            r[1] += 1
        for k in keys:
            self.rules[k][out] += 1
        if ctx["kind"] == "mv" and not died and g1 is not None:
            sc1 = Scene(g1)
            self.self_cells = self.locate_self(sc1)
        if out[0] != "death" and len(out) > 1 and isinstance(out[1], tuple) and out[1][0] == "push":
            self.movers.add(ctx["o1"][0])
        elif out[0] != "death" and out[1] == "push":
            self.movers.add(ctx["o1"][0])
        self.see(g0)
        if g1 is not None:
            self.see(g1)
            for p in list(self.ptr):
                if not self.due(p):
                    continue
                pc = frozenset(self.tick_cells(g0, p))
                if pc:
                    r = self.ptr_ok[p]
                    r[0] += int(pc == self._rem.get(p, frozenset()))
                    r[1] += 1
            self.learn_ticks(keys[-1], out != ("death",) and out[0] == 0 and out[1] == "same")
        return out

    def see(self, g):
        """Update the static layer: every cell not covered by the self or a mover keeps its last colour."""
        if self.floor is None or self.floor.shape != g.shape:
            self.floor = np.full(g.shape, MASK, np.int64)
        movers = set(self.movers)
        if self.self_colour is not None:
            movers.add(self.self_colour)
        m = (g >= 0) & ~np.isin(g, list(movers))
        self.floor[m] = g[m]

    def new_level(self):
        """Positions reset at a level change; the rules, the self and the movers are kept."""
        self.floor = None
        self.self_cells = None
        self.ptr = {}
        self.pair_last = {}

    def best(self, keys):
        """(key, outcome, its share) of the most specific key with data."""
        for k in keys:
            c = self.rules.get(k)
            if c:
                o, n = c.most_common(1)[0]
                return k, o, n / sum(c.values())
        return None

    def rate(self, key):
        r, n = self.checked[key]
        return (r + 1.0) / (n + 2.0)

    # ---- prediction ------------------------------------------------------------------------------------------------
    def predict(self, g0, a, xy=None, ticks=True):
        """(predicted next grid or None, info). info: key, outcome, share, verified rate, death."""
        sc0 = Scene(g0)
        keys, ctx = self.relation(sc0, a, xy)
        b = self.best(keys)
        if b is None:
            return None, dict(key=None)
        key, out, share = b
        info = dict(key=key, outcome=out, share=share, rate=self.rate(key), death=out == ("death",))
        if out == ("death",):
            return None, info
        g = apply(sc0, ctx, out, self.self_colour, self.floor)
        if g is None:
            return None, info
        # the world's own steps and the action's counted changes, where their pointers say
        if not ticks:
            return g, info
        todo = [p for p in self.ptr if self.autonomous(p) and self.due(p)] + \
            [(o, n) for o, n, how in out[2] if how != "all"]
        for p in dict.fromkeys(todo):
            r = self.ptr_ok[p]
            if r[1] < 3 or r[0] < 0.6 * r[1]:
                continue                                    # a pointer is used only once verified
            for y, x in self.tick_cells(g0, p):
                if g[y, x] == g0[y, x]:
                    g[y, x] = p[1]
        return g, info


def apply(sc0, ctx, out, self_colour, floor=None):
    """The grid an outcome produces from scene sc0 (the model's imagination)."""
    g0 = sc0.g
    g = g0.copy()
    k, eff1, remote = out
    local = np.zeros(g0.shape, bool)
    fl = floor if floor is not None and floor.shape == g0.shape else None

    def vacate(ys, xs, col=None, under=None):
        dflt = under if under is not None else surround(g0, ys, xs, (self_colour, col))
        dflt = sc0.bg if dflt is None else dflt
        if fl is None:
            g[ys, xs] = dflt
        else:
            v = fl[ys, xs]
            g[ys, xs] = np.where((v >= 0) & (v != col), v, dflt)
    if ctx["kind"] == "mv":
        ys, xs = ctx["cells"]
        dy, dx = ctx["d"]
        o1 = ctx["o1"]
        h, w = g0.shape
        if k == "slide":
            j = 0
            while True:
                ny, nx = ys + (j + 1) * dy, xs + (j + 1) * dx
                if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
                    break
                v = g0[ny, nx]
                if ((v != sc0.bg) & (v != self_colour) & (v >= 0)).any():
                    break
                j += 1
            k = j
        if o1 is not None:
            col, oy, ox = o1
            local[oy, ox] = True
            if eff1 == "push" or (isinstance(eff1, tuple) and eff1[0] == "push"):
                ny, nx = oy + dy, ox + dx
                if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
                    return None
                vacate(oy, ox, col)
                g[ny, nx] = col if eff1 == "push" else eff1[1]
                local[ny, nx] = True
            elif eff1 == "gone":
                vacate(oy, ox, col)
            elif isinstance(eff1, tuple) and eff1[0] == "recolour":
                g[oy, ox] = eff1[1]
        if k:
            ny, nx = ys + k * dy, xs + k * dx
            if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
                return None
            c1 = ctx.get("c1")
            # what the self leaves behind, where never seen: the floor it steps onto, if it steps onto floor
            under = c1 if (o1 is None or eff1 == "under") and c1 is not None and c1 >= 0 else None
            vacate(ys, xs, under=under)
            g[ny, nx] = self_colour
            for j in range(0, k + 1):
                local[ys + j * dy, xs + j * dx] = True
        else:
            local[ys, xs] = True
    elif ctx["kind"] == "click":
        o = ctx.get("obj")
        if o is not None:
            col, oy, ox = o
            local[oy, ox] = True
            if eff1 == "gone":
                vacate(oy, ox, col)
            elif isinstance(eff1, tuple):
                g[oy, ox] = eff1[1]
    # remote colour changes, simultaneously (a swap is two pairs); a counted change is not localised: left as is
    out_g = g.copy()
    for o, n, how in remote:
        if how == "all":
            out_g[(g0 == o) & ~local] = n
    return out_g


def _whole(a, b, c, t):
    """Whether the objects of colour c that moved moved whole: each component the move left, shifted by t, lies on
    colour c in b (a moving thing; the floor the self walks over "moves" the other way, but not whole)."""
    A = a == c
    gone = A & ~(b == c)
    lab, comps = components(A)
    h, w = a.shape
    dy, dx = t
    for i in set(lab[gone].tolist()):
        ys, xs = comps[i]
        ny, nx = ys + dy, xs + dx
        if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w or not (b[ny, nx] == c).all():
            return False
    return True


def _translation(a, b, c):
    """(dy, dx) by which the cells of colour c that changed moved rigidly from a to b, else None (muse.translation's
    test, on the changed part of the colour)."""
    A, B = a == c, b == c
    gone, new = A & ~B, B & ~A
    if not gone.any() or not new.any() or gone.sum() != new.sum():
        return None
    gy, gx = np.nonzero(gone)
    ny, nx = np.nonzero(new)
    for dy, dx in {(int(ny[0] - y), int(nx[0] - x)) for y, x in zip(gy.tolist(), gx.tolist())}:
        if dy == 0 and dx == 0:
            continue
        sh = np.zeros_like(A)
        h, w = A.shape
        ys, xs = np.nonzero(A)
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        moved = A & ~(A & B & ~gone)          # every cell of the colour
        # the moved part: the old cells that left, shifted, must equal the new cells plus the old cells kept under it
        src = np.zeros_like(A)
        src[gy, gx] = True
        # include old cells that stay covered by the moved copy (overlap of old and new positions)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if B[y, x] and 0 <= y - dy < h and 0 <= x - dx < w and A[y - dy, x - dx] and not gone[y, x]:
                pass
        sy, sx = np.nonzero(src)
        ty, tx = sy + dy, sx + dx
        if (ty < 0).any() or (tx < 0).any() or (ty >= h).any() or (tx >= w).any():
            continue
        sh[ty, tx] = True
        if (sh & ~B).any():
            continue
        if ((B & ~A) & ~sh).any():
            continue
        del moved, ok, yy, xx
        return dy, dx
    return None
