# SPDX-License-Identifier: MIT-0
"""Stream BS's effect-graph pieces as switches over kit HV's planner (`hy_arms.HyThink`, itself kit TK's `think.
ThinkAgent` over kit X's explore v3). A port of `crates/openmorph-perception/src/effect_graph.rs` (PR #813) to the
logical grid. Both switches are off by default; off, `BsThink` plays exactly as `HyThink`.

* `BS_KEY=1` -- EFFECT-EQUIVALENT STATES. The state key also paints out the cells that tests show never change what
  an action does:
  - TICK TEST: a change kind (a cell's old and new colour) that was a transition's only change under two simple
    actions, or (click-only) that appeared in every one of >= 8 transitions of the level and once alone: every cell
    it changes, grown to its same-colour run (and its bounding box when small), is painted out;
  - EFFECT TEST: two grids that differ in a few cells, under the same action, changed the same cells the same way
    elsewhere (and changed something): a vote for painting those cells out. A disagreement on a one-part region votes
    against. A cell is painted out with support from two action classes and more than twice its contradictions, or
    four non-empty agreements and no contradiction. Votes are judged against the ticks (a tick is no effect).
  When the painted set grows, every state-keyed structure of the planner is remapped to the new keys.
  The painted cells are not an effect either (v3's effect head and TK's outcome rates read `effect`).
* `BS_CLICK=1` -- CLICK PROBING BY EFFECT. Click candidates: a cell of every object (no 32 cap) and a coarse grid of
  cells; a click that changed nothing on a colour is presumed inert where that colour is under it again, and every
  click on a colour inert at 3 places and effective nowhere; two clicks that took one state to the same next state
  are one button (only its first is offered).

AUTHORED (labelled, as the rules ask): the tests, the vote rule, the thresholds, the grid spacing. LEARNED per game:
the painted cells, the buttons, the inert colours. Nothing names a game, a colour or a mechanic.
"""
import collections
import hashlib
import os

import numpy as np

import hy_arms as hy

SOLO = 4            # non-empty agreements that paint a cell under one action class, with no contradiction
EVERY = 8           # click-only: transitions of a level before a kind seen in all of them is a tick
INERT_COLOUR = 3    # places where clicks on one colour changed nothing before that colour is presumed inert


def _neighbours(p, w, h):
    y, x = divmod(p, w)
    if x > 0:
        yield p - 1
    if x + 1 < w:
        yield p + 1
    if y > 0:
        yield p - w
    if y + 1 < h:
        yield p + w


def _components(cells, w, h):
    cells = set(cells)
    seen, out = set(), []
    for p in sorted(cells):
        if p in seen:
            continue
        comp, stack = [p], [p]
        seen.add(p)
        while stack:
            c = stack.pop()
            for n in _neighbours(c, w, h):
                if n in cells and n not in seen:
                    seen.add(n)
                    comp.append(n)
                    stack.append(n)
        out.append(comp)
    return out


def _flood(f, p, w, h, cap):
    col = f[p]
    seen, stack = {p}, [p]
    while stack:
        c = stack.pop()
        for n in _neighbours(c, w, h):
            if n not in seen and f[n] == col:
                seen.add(n)
                if len(seen) > cap:
                    return None
                stack.append(n)
    return seen


class LifeClock:
    """Stream HY's life-periodic filter (from `Verifier.time_log` / `time_mask`, same rule and
    defaults): cells driven by life time, not by actions -- across >= `lives` finished lives of this level, each changed
    at exactly the same life-relative steps in every life since it first changed (a budget bar). A fading floor changes
    where and when the self has been, so it fails the test unless the agent replays the same path every life.
    AUTHORED rule; the cells are learned per level."""

    def __init__(self, lives=3, min_cells=4):
        self.lives, self.min_cells = lives, min_cells
        self.reset()

    def reset(self):                        # at each level-up
        self.log, self.life, self.step, self._cache = {}, 0, 0, (None, None)

    def observe(self, g0, g1):              # a transition within a life (not the death, not the level-up)
        if g0.shape == g1.shape:
            for y, x in zip(*np.nonzero(g0 != g1)):
                self.log.setdefault((int(y), int(x)), {}).setdefault(self.life, []).append(self.step)
        self.step += 1

    def end_life(self):                     # a death: the next life starts its step count at 0
        self.life += 1
        self.step = 0

    def mask(self, shape):
        key = (shape, self.life)
        if self._cache[0] == key:
            return self._cache[1]
        m = np.zeros(shape, bool)
        for (y, x), per in self.log.items():
            if not (0 <= y < shape[0] and 0 <= x < shape[1]):
                continue
            done = {li: ts for li, ts in per.items() if li < self.life}      # finished lives only
            if len(done) < self.lives:
                continue
            first = min(done)
            if any(li not in done for li in range(first, self.life)):        # it must change in every life since
                continue
            if len({tuple(ts) for ts in done.values()}) == 1:                # at the same steps of each life
                m[y, x] = True
        if m.sum() < self.min_cells:                                         # a bar is many cells
            m[:] = False
        self._cache = (key, m)
        return m


class BsThink(hy.HyThink):
    def __init__(self, game_id, seed=1, verifier=None, arm="B"):
        super().__init__(game_id, seed, verifier=verifier, arm=arm)
        self.bs_key = bool(int(os.environ.get("BS_KEY", "0")))
        self.bs_click = bool(int(os.environ.get("BS_CLICK", "0")))
        # which key tests: "all" (the tick test and the effect test), "tick" (the tick test only), "tick+life" (a
        # ticking cell painted out only once HY's LifeClock also finds it life-periodic), "tick+chain" (that only for
        # cells painted by a colour chain a->b->c, plain ticks as "tick")
        self.bs_key_mode = os.environ.get("BS_KEY_MODE", "all")
        self.bs_life = LifeClock()
        # which candidates: "all" (every object + a coarse grid), "nogrid" (every object), "cap" (v3's first 32
        # objects); the inert and button filters apply to each
        self.bs_click_mode = os.environ.get("BS_CLICK_MODE", "all")
        # which inert rule: "both" (the position on a colour, and a colour inert at 3 places), "pos" (the position on
        # a colour only), "none"; "soft" / "softpos": as "both" / "pos", but at a state whose other clicks were all
        # tried, the presumed-inert ones are offered again
        self.bs_click_inert = os.environ.get("BS_CLICK_INERT", "both")
        self._bs_reset(None)
        self.bs_stats = collections.Counter()

    # ---- per-shape state (a new level or the raw reading may change the grid's shape) ----------------------------
    def _bs_reset(self, shape):
        self.bs_shape = shape
        n = 0 if shape is None else int(np.prod(shape))
        self.bs_mask = np.zeros(n, bool)
        self.bs_tick = np.zeros(n, bool)
        self.bs_tick_kinds = collections.defaultdict(set)   # cell -> the tick kinds that painted it
        self.bs_votes = []                          # (cells, class bits, support?, effect cells)
        self.bs_alone = collections.defaultdict(int)
        self.bs_ticks = set()
        self.bs_kind_n = collections.Counter()
        self.bs_at_click = set()                    # kinds seen at or next to the clicked cell
        self.bs_kind_clicks = collections.defaultdict(set)   # kind -> the clicks it appeared under (up to 3)
        self.bs_trans = []                          # this shape's transitions: (flat g0, action, flat g1)
        self.bs_by_act = collections.defaultdict(list)
        self.bs_key_grid = {}                       # key -> the grid that made it
        # clicks
        self.bs_inert = set()                       # (action, colour)
        self.bs_col_inert = collections.defaultdict(set)
        self.bs_col_eff = set()
        self.bs_group = {}                          # click -> its button's first click

    def _bs_shape(self, shape):
        if shape != self.bs_shape:
            self._bs_reset(shape)

    # ---- the key ---------------------------------------------------------------------------------------------------
    def key(self, g):
        if not self.bs_key:
            return super().key(g)
        self._bs_shape(g.shape)
        h = g.copy()
        h[self.clock_mask(g.shape)] = -1
        h.reshape(-1)[self.bs_mask] = -1
        k = hashlib.blake2b(h.tobytes(), digest_size=8).hexdigest()
        if k not in self.bs_key_grid:
            self.bs_key_grid[k] = g.copy()
        return k

    # ---- the hook: every non-terminal transition passes through `effect(g0, g1)` with `self.prev` its action ----------
    def effect(self, a, b):
        e = super().effect(a, b)
        if (not self.bs_key and not self.bs_click) or a.shape != b.shape or self.prev is None:
            return e
        act = self.prev[1]
        self._bs_shape(a.shape)
        if getattr(self, "_bs_level", None) != self.level:
            # a new level: its transitions and clicks are its own (the painted cells and the votes are kept)
            self._bs_level = self.level
            self.bs_trans, self.bs_by_act = [], collections.defaultdict(list)
            self.bs_inert, self.bs_col_inert, self.bs_col_eff, self.bs_group = (
                set(), collections.defaultdict(set), set(), {})
            self.bs_kind_n = collections.Counter()
            self.bs_at_click = set()
            self.bs_kind_clicks = collections.defaultdict(set)
            self.bs_life.reset()
        d = (a != b).reshape(-1)
        outside = d & ~self.clock_mask(a.shape).reshape(-1)
        if self.bs_key:
            outside &= ~self.bs_mask
        changed = bool(outside.any())
        if self.bs_click and " " in act:
            self._note_click(a, act, changed)
        if self.bs_key:
            if self.bs_key_mode in ("tick+life", "tick+chain"):
                self.bs_life.observe(a, b)
            f0, f1 = a.reshape(-1).copy(), b.reshape(-1).copy()
            self.bs_trans.append((f0, act, f1))
            self.bs_by_act[act].append(len(self.bs_trans) - 1)
            grew = self._ticks(f0, act, f1)
            if self.bs_key_mode != "tick":
                grew |= self._pair(len(self.bs_trans) - 1)
            if grew:
                self._remap()
            return changed
        return e

    # ---- clicks ------------------------------------------------------------------------------------------------------
    def _note_click(self, g, act, changed):
        _, x, y = act.split()
        c = int(g[int(y), int(x)])
        if changed:
            self.bs_col_eff.add(c)
        else:
            self.bs_inert.add((act, c))
            self.bs_col_inert[c].add(act)

    def _presumed_inert(self, g, act):
        _, x, y = act.split()
        c = int(g[int(y), int(x)])
        if self.bs_click_inert == "none":
            return False
        if self.bs_click_inert in ("pos", "softpos"):
            return (act, c) in self.bs_inert
        return (act, c) in self.bs_inert or (c not in self.bs_col_eff and len(self.bs_col_inert[c]) >= INERT_COLOUR)

    def cells(self, g, cap=32):
        """Click candidates: every object's cell (no cap), then a coarse grid, without presumed-inert clicks and
        without a button's other clicks."""
        import muse
        self._bs_shape(g.shape) if self.bs_key else None
        vals, counts = np.unique(g, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        objs = []
        for c in [int(v) for v in vals if v != bg]:
            for comp in muse._components(g == c):
                my = sum(p[0] for p in comp) / len(comp)
                mx = sum(p[1] for p in comp) / len(comp)
                y, x = min(comp, key=lambda p: ((p[0] - my) ** 2 + (p[1] - mx) ** 2, p))
                objs.append((x, y))
        objs.sort(key=lambda p: (p[1], p[0]))
        if self.bs_click_mode == "cap":
            objs = objs[:cap]
        h, w = g.shape
        s = max(1, min(h, w) // 4)
        grid = [] if self.bs_click_mode != "all" else [
            (x, y) for y in range(s // 2, h, s) for x in range(s // 2, w, s)]
        out, kept, seen = [], [], set()
        for x, y in objs + grid:
            if (x, y) in seen:
                continue
            seen.add((x, y))
            act = f"6 {x} {y}"
            if self.bs_group.get(act, act) != act:
                continue
            kept.append((x, y))
            if self._presumed_inert(g, act):
                continue
            out.append((x, y))
        if self.bs_click_inert.startswith("soft"):
            # soft: presumed-inert clicks come back at a state once its other clicks were all tried there
            tried = self.tried.get(self.key(g), set())
            if all(f"6 {x} {y}" in tried for x, y in out):
                return kept
        return out

    def _think(self, latest):
        if not self.bs_click:
            return super()._think(latest)
        import play
        orig = play.click_cells
        play.click_cells = lambda g, cap=32: self.cells(g, cap)
        try:
            return super()._think(latest)
        finally:
            play.click_cells = orig

    def observe(self, k_now, died, leveled):
        # a button: two clicks that took one state to the same next state
        if self.bs_click and self.pend is not None and not died and not leveled and k_now is not None:
            k0, a = self.pend[0], self.pend[1]
            if " " in a and k_now != k0 and a not in self.bs_group:
                other = next((b for b, n in self.graph.get(k0, {}).items() if b != a and " " in b and n == k_now),
                             None)
                self.bs_group[a] = self.bs_group.get(other, other) if other else a
        super().observe(k_now, died, leveled)

    # ---- the tick test -----------------------------------------------------------------------------------------------
    @staticmethod
    def _class(act):
        if " " in act:
            return 1 << (8 + int(hashlib.blake2b(act.encode(), digest_size=2).hexdigest(), 16) % 8)
        return 1 << min(int(act), 7)

    def _ticks(self, f0, act, f1):
        idx = np.nonzero(f0 != f1)[0]
        kinds = {(int(f0[p]), int(f1[p])) for p in idx}
        for k in kinds:
            self.bs_kind_n[k] += 1
        for k in kinds:
            if " " in act and len(self.bs_kind_clicks[k]) < 3:
                self.bs_kind_clicks[k].add(act)
        if " " in act:
            # a kind that changed the clicked cell or a cell next to it is the click's own effect, never a clock
            _, x, y = act.split()
            h, w = self.bs_shape
            c = int(y) * w + int(x)
            near = {c} | set(_neighbours(c, w, h)) if c < h * w else set()
            for p in idx:
                if int(p) in near:
                    self.bs_at_click.add((int(f0[p]), int(f1[p])))
        if len(kinds) == 1:
            k = next(iter(kinds))
            self.bs_alone[k] |= 1 if " " in act else self._class(act)
        n = len(self.bs_trans)
        only_clicks = all(" " in a for a in self.bs_by_act)
        new = [k for k in kinds if k not in self.bs_ticks and (
            bin(self.bs_alone[k] & ~1).count("1") >= 2
            or (only_clicks and self.bs_alone[k] and n >= EVERY and self.bs_kind_n[k] == n
                and k not in self.bs_at_click and len(self.bs_kind_clicks[k]) >= 3))]
        span = range(n) if new else range(n - 1, n)
        self.bs_ticks.update(new)
        h, w = self.bs_shape
        cap = max(1, h * w // 16)
        bg = int(np.bincount(f0).argmax())
        grew = False
        for i in span:
            g0, _, g1 = self.bs_trans[i]
            seed = [int(p) for p in np.nonzero(g0 != g1)[0]
                    if (int(g0[p]), int(g1[p])) in self.bs_ticks and not self.bs_tick[p]]
            kinds_here = {(int(g0[p]), int(g1[p])) for p in seed}
            for comp in _components(seed, w, h):
                for p in self._grow(comp, (g0, g1), cap, bg):
                    self.bs_tick_kinds[p] |= kinds_here
                    if not self.bs_tick[p]:
                        self.bs_tick[p] = True
                        grew = True
        if grew:
            self._revote()
        return grew and self._recompute()

    def _grow(self, seed, frames, cap, bg):
        h, w = self.bs_shape
        region = set(seed)
        for p in seed:
            for f in frames:
                if f[p] != bg:
                    run = _flood(f, p, w, h, cap)
                    if run is not None:
                        region |= run
        ys, xs = zip(*(divmod(p, w) for p in region))
        if (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) <= cap:
            region = {y * w + x for y in range(min(ys), max(ys) + 1) for x in range(min(xs), max(xs) + 1)}
        elif len(region) > cap:
            region = set(seed)
        return region

    # ---- the effect test ---------------------------------------------------------------------------------------------
    def _pair(self, ti):
        g, act, g2 = self.bs_trans[ti]
        h, w = self.bs_shape
        limit = max(4, h * w // 128)
        clock = self.clock_mask(self.bs_shape).reshape(-1)
        free = ~self.bs_mask & ~clock
        dg = (g != g2) & free
        votes = []
        for oi in self.bs_by_act[act][:-1]:
            f, _, f2 = self.bs_trans[oi]
            r = (f != g) & free
            nr = int(r.sum())
            if nr == 0 or nr > limit:
                continue
            region = set(int(p) for p in np.nonzero(r)[0])
            df = (f != f2) & free
            # disagreements: cells changed in only one move, or to different colours
            dis = np.nonzero((df != dg) | (df & dg & (f2 != g2)))[0]
            dis = [int(p) for p in dis if p not in region]
            grew = True
            while grew and dis:
                grew = False
                keep = []
                for p in dis:
                    if any(n in region for n in _neighbours(p, w, h)):
                        region.add(p)
                        grew = True
                    else:
                        keep.append(p)
                dis = keep
            reg = np.zeros(h * w, bool)
            reg[list(region)] = True
            ef, eg = df & ~reg, dg & ~reg
            same = bool((ef == eg).all() and (f2[ef] == g2[ef]).all())
            if same:
                effect = [int(p) for p in np.nonzero(ef)[0]]
            else:
                if len(_components(region, w, h)) > 1:
                    continue
                effect = [int(p) for p in np.nonzero((ef != eg) | (ef & eg & (f2 != g2)))[0]]
            votes.append((sorted(region), same, effect))
        if not votes:
            return False
        bits = self._class(act)
        for region, support, effect in votes:
            self.bs_votes.append((region, bits, support, effect))
        self._revote()
        return self._recompute()

    def _revote(self):
        n = self.bs_mask.size
        self.bs_sup = np.zeros(n, np.int64)
        self.bs_sup_n = np.zeros(n, np.int64)
        self.bs_full_n = np.zeros(n, np.int64)
        self.bs_con_n = np.zeros(n, np.int64)
        for region, bits, support, effect in self.bs_votes:
            full = any(not self.bs_tick[p] for p in effect)
            if not full:
                continue
            if support:
                self.bs_sup[region] |= bits
                self.bs_full_n[region] += 1
            else:
                self.bs_con_n[region] += 1

    def _recompute(self):
        if not hasattr(self, "bs_sup") or self.bs_sup.size != self.bs_mask.size:
            self._revote()
        classes = np.array([bin(int(b)).count("1") for b in self.bs_sup])
        m = ((classes >= 2) & (self.bs_full_n > 2 * self.bs_con_n)) \
            | ((self.bs_full_n >= SOLO) & (self.bs_con_n == 0)) \
            | (self._tick_mask() & (self.bs_con_n == 0))
        changed = bool((m != self.bs_mask).any())
        self.bs_mask = m
        if changed:
            self.bs_stats["mask_changes"] += 1
        return changed

    def _tick_mask(self):
        """The tick cells as the key mode reads them (HY's life filter in the "+life" / "+chain" modes)."""
        if self.bs_key_mode not in ("tick+life", "tick+chain") or self.bs_shape is None:
            return self.bs_tick
        life = self.bs_life.mask(self.bs_shape).reshape(-1)
        if self.bs_key_mode == "tick+life":
            return self.bs_tick & life
        chain = {k for k in self.bs_ticks if any(j[1] == k[0] or k[1] == j[0] for j in self.bs_ticks if j != k)}
        out = self.bs_tick.copy()
        for p, ks in self.bs_tick_kinds.items():
            if ks & chain and not life[p]:
                out[p] = False
        return out

    def choose_action(self, frames, latest):
        if self.bs_key and self.bs_key_mode in ("tick+life", "tick+chain"):
            from arcengine import GameState
            if latest.state is GameState.GAME_OVER and self.prev is not None:
                self.bs_life.end_life()
                if self.bs_shape is not None and self._recompute():
                    self._remap()
        return super().choose_action(frames, latest)

    # ---- remap every state-keyed structure to the new keys ------------------------------------------------------------
    def _remap(self):
        old = dict(self.bs_key_grid)
        self.bs_key_grid = {}
        m = {k: self.key(g) for k, g in old.items()}
        f = lambda k: m.get(k, k)  # noqa: E731
        tried = collections.defaultdict(set)
        for k, v in self.tried.items():
            tried[f(k)] |= v
        self.tried = tried
        graph = collections.defaultdict(dict)
        for k, edges in self.graph.items():
            for a, n in edges.items():
                graph[f(k)].setdefault(a, f(n))
        self.graph = graph
        acts, kinds = {}, {}
        for k, v in self.acts_of.items():
            acts.setdefault(f(k), v)
        for k, v in self.kinds_of.items():
            kinds.setdefault(f(k), v)
        self.acts_of, self.kinds_of = acts, kinds
        parent = {}
        for k, (pk, t) in self.parent.items():
            if f(k) != f(pk):
                parent.setdefault(f(k), (f(pk), t))
        self.parent = parent
        self.dead = {(f(k), a) for k, a in self.dead}
        if self.start_key is not None:
            self.start_key = f(self.start_key)
        if self.goal is not None:
            self.goal = (f(self.goal[0]), self.goal[1])
        if self.pend is not None:
            k0, a, t, first, par = self.pend
            self.pend = (f(k0), a, t, first, None if par is None else (f(par[0]), par[1]))
        if self.pend_key is not None:
            self.pend_key = f(self.pend_key)
        if self.pend_pred is not None:
            src, nxt, cls = self.pend_pred
            self.pend_pred = (src, f(nxt) if isinstance(nxt, str) else nxt, cls)
        self.route = []
        self.bs_stats["remaps"] += 1
