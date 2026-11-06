# SPDX-License-Identifier: MIT-0
"""Stream RU's agent: kit DG13 with an object-level rule model that is kept across levels and planned over.

`RuThink` is kit DG13's policy (`dg13.Dg13Think`) plus three things, all in this kit (nothing in a crate):
* LEARN RULES (`rules.RuleModel`): every transition, on the logical grid with DG13's clock and bars painted out,
  updates the rules (what an action does to the objects, keyed by the self's relation to what it meets). Rules,
  the self and its displacements are KEPT across levels; only positions reset.
* LEARN THE WIN (`WinModel`): at each level completion, which object-level conditions became true (AUTHORED
  vocabulary: the self entered a colour, a colour reached zero cells, a colour was clicked); a condition that became
  true on a step that did not finish the level is refuted once; the conditions true at every win so far, least
  refuted first, are the goal.
* PLAN WITH THEM (`imagine`): at a level whose start the rules cover, a breadth-first search over PREDICTED grids
  (the rules as the model; predicted deaths pruned; the life's verified budget respected) for an action sequence
  whose last step makes the goal true. The plan is executed while each step's prediction comes true; on a miss the
  rules learn and the plan is searched again (predict, verify, learn). No plan: kit DG13 decides, as before.
AUTHORED (labelled): the object reading, the relation key, the effect classes, the win vocabulary, the search and its
caps. GROWN: the rules, their verified rates, the win condition and which conditions matter.

Switches (fixed in `my_agent.py` for a kit; the environment for screens): RU_PLAN (1: plan to the learned win from
level 2 on), RU_NODES (search cap, 2000), RU_OPT (1: when the verified rules give no plan, one more search in which
one untested move per plan steps onto what it meets; three such plans a level at most, none after one missed), RU_PROBE (0: also walk to
the nearest untested relation), RU_L1 (0: also plan on level 1), RU_PRUNE (0: the rules score DG13's untried
actions), RU_DEBUG (0). An optimistic plan that misses ends optimism for its level.
"""
import collections
import os
import time

import numpy as np

import dg13
import rules


class WinModel:
    """The level's win condition as object-level hypotheses, supported at wins and refuted on steps."""

    def __init__(self):
        self.support = collections.Counter()
        self.refute = collections.Counter()
        self.level_refute = collections.Counter()     # refutations in the current level (reset at each level)
        self.level_dead = collections.Counter()       # deaths while a plan toward it was walked, this level
        self.wins = 0

    @staticmethod
    def hyps(m, g0, a, xy, g1, final):
        """The conditions the transition g0 -a-> g1 made true (`final`: g1 is the completed level's frame)."""
        out = set()
        sc0 = rules.Scene(g0)
        keys, ctx = m.relation(sc0, a, xy)
        sc = m.self_colour
        if ctx["kind"] == "mv" and ctx.get("c1") is not None and ctx["c1"] >= 0 and ctx["c1"] != sc0.bg:
            if final:
                out.add(("enter", int(ctx["c1"])))
            elif g1 is not None:
                o = m.outcome(sc0, g1, ctx, False)
                if o is not None and o != ("death",) and o[0] != 0:
                    out.add(("enter", int(ctx["c1"])))
        if ctx["kind"] == "click" and xy is not None:
            out.add(("click", rules.colour_at(g0, xy[1], xy[0])))
        if g1 is not None and g1.shape == g0.shape:
            for c in set(int(v) for v in np.unique(g0)) - set(int(v) for v in np.unique(g1)):
                if c >= 0 and c != sc and c != sc0.bg:
                    out.add(("gone", c))
        return out

    def observe_win(self, hs):
        self.wins += 1
        self.support.update(hs)
        self.level_refute = collections.Counter()
        self.level_dead = collections.Counter()

    def observe_step(self, hs):
        for h in hs:
            if self.support[h]:
                self.refute[h] += 1
                self.level_refute[h] += 1

    def goals(self):
        """Conditions true at every win so far, least refuted first (ties: the support, then the name)."""
        if not self.wins:
            return []
        cand = [h for h, n in self.support.items()
                if n == self.wins and self.level_refute[h] < 2 and self.level_dead[h] < 3]
        return sorted(cand, key=lambda h: (self.refute[h] / (self.support[h] + self.refute[h]), str(h)))


def holds(goal, ctx, pred_g, out, a, xy, g0):
    """Whether an imagined step makes the goal true."""
    kind, c = goal
    if kind == "enter":
        return ctx["kind"] == "mv" and ctx.get("c1") == c and out is not None and out[0] != 0
    if kind == "gone":
        return pred_g is not None and not (pred_g == c).any() and (g0 == c).any()
    if kind == "click":
        return xy is not None and rules.colour_at(g0, xy[1], xy[0]) == c
    return False


class RuThink(dg13.Dg13Think):
    def __init__(self, game_id, seed=1, verifier=None, arm="B"):
        super().__init__(game_id, seed, verifier=verifier, arm=arm)
        self.ru = rules.RuleModel()
        self.win = WinModel()
        self.ru_on = bool(int(os.environ.get("RU_PLAN", "1")))
        self.ru_nodes = int(os.environ.get("RU_NODES", "2000"))
        self.ru_probe = bool(int(os.environ.get("RU_PROBE", "0")))
        self.ru_l1 = bool(int(os.environ.get("RU_L1", "0")))
        self.ru_debug = bool(int(os.environ.get("RU_DEBUG", "0")))     # print the first failed searches (off)
        self.ru_prune = bool(int(os.environ.get("RU_PRUNE", "0")))     # the rules score DG13's untried actions (off)
        self.ru_opt = bool(int(os.environ.get("RU_OPT", "1")))         # an optimistic re-search (on)
        self.ru_plan_bold = False         # the plan in hand came from the optimistic search
        self.ru_goals = []                # the goals the plan in hand was searched for
        self._ru_cache = {}
        self.ru_bold = {}                # level -> optimistic plans made
        self.ru_prev = None              # (G, raw grid, action, xy) of the last decision
        self.ru_plan = []                # [(action string, action id, xy, predicted masked grid)]
        self.ru_fail = None              # (level, model size) at the last failed search: no search until it changes
        self.ru_stats = collections.Counter()
        self.ru_s = 0.0
        self.ru_level = 0

    # ---- reading ------------------------------------------------------------------------------------------------
    def _ru_mask(self, g):
        try:
            return self._masked(g)
        except Exception:
            return g.copy()

    @staticmethod
    def _ru_action(a):
        if " " in a:
            _, x, y = a.split()
            return 6, (int(x), int(y))
        return int(a), None

    # ---- the hook: every step's outcome --------------------------------------------------------------------------
    def choose_action(self, frames, latest):
        t0 = time.perf_counter()
        try:
            self._ru_observe(latest)
        except Exception as e:  # pragma: no cover - the rule layer must never break the agent
            self.ru_stats["errors"] += 1
            self.ru_last_error = repr(e)
            self.ru_prev = None
        self.ru_s += time.perf_counter() - t0
        act = super().choose_action(frames, latest)
        return act

    def _ru_observe(self, latest):
        from arcengine import GameState
        p, self.ru_prev = self.ru_prev, None
        if p is None:
            return
        G, g0raw, a, xy = p
        aid, axy = self._ru_action(a)
        g0 = self._ru_mask(g0raw)
        lv = int(latest.levels_completed or 0)
        if latest.state is GameState.GAME_OVER:
            self.ru.observe(g0, aid, None, died=True, xy=axy)
            if self.ru_plan:
                for h in self.ru_goals:
                    self.win.level_dead[h] += 1               # a life lost on the way: three, and it is dropped
            self.ru_plan = []
            self.ru.self_cells = None
            return
        if not latest.frame:
            return
        if lv > self.ru_level:
            # the completed level's last frame is the response's first frame
            g1 = None
            try:
                g1raw = G.grid(np.asarray(latest.frame[0], dtype=np.int64))
                if g1raw.shape == g0raw.shape:
                    g1 = self._ru_mask_like(g1raw, g0raw, g0)
            except Exception:
                g1 = None
            hs = WinModel.hyps(self.ru, g0, aid, axy, g1, True)
            self.win.observe_win(hs)
            self.ru_stats["wins"] += 1
            if g1 is not None:
                self.ru.observe(g0, aid, g1, xy=axy)
            self.ru_level = lv
            self.ru.new_level()
            self.ru_plan, self.ru_fail = [], None
            return
        g1raw = G.grid(np.asarray(latest.frame[-1], dtype=np.int64))
        if g1raw.shape != g0raw.shape:
            return
        g1 = self._ru_mask_like(g1raw, g0raw, g0)
        self.win.observe_step(WinModel.hyps(self.ru, g0, aid, axy, g1, False))
        self.ru.observe(g0, aid, g1, xy=axy)

    @staticmethod
    def _ru_mask_like(g1raw, g0raw, g0):
        g1 = g1raw.copy()
        g1[g0 < 0] = -1
        return g1

    # ---- the decision ---------------------------------------------------------------------------------------------
    def plan(self, key, acts_here, g, comp_of):
        a = None
        t0 = time.perf_counter()
        try:
            a = self._ru_plan(g, acts_here)
        except Exception as e:  # pragma: no cover
            self.ru_stats["errors"] += 1
            self.ru_last_error = repr(e)
            self.ru_plan = []
        self.ru_s += time.perf_counter() - t0
        if a is not None:
            self.ru_stats["ru_steps"] += 1
            self.goal = None
            out = a
        else:
            out = super().plan(key, acts_here, g, comp_of)
        return out

    def _think(self, latest):
        a = super()._think(latest)
        # remember the decision (the grid it was taken on and its reading) for the next step's observation
        if self.prev is not None and not self.raw:
            self.ru_prev = (self.G, self.prev[0], self.prev[1], None)
        else:
            self.ru_prev = None
        return a

    def _ru_plan(self, g, acts_here):
        if not self.ru_on or self.raw:
            return None
        gm = self._ru_mask(g)
        # on track: the last planned step came true
        if self.ru_plan:
            want = self.ru_plan[0][3]
            if want is not None and want.shape == gm.shape and ((want == gm) | (want < 0) | (gm < 0)).all():
                self.ru_plan.pop(0)
            else:
                self.ru_stats["plan_miss"] += 1
                if self.ru_plan_bold:
                    self.ru_bold[self.level] = 3             # an optimistic plan missed: no more optimism this level
                self.ru_plan = []
        if self.ru_plan:
            if self.ru_plan and self.ru_plan[0][0] == "#done":
                self.ru_plan = []
            else:
                a = self._ru_match(self.ru_plan[0][0], acts_here, gm)
                if a is not None:
                    self.ru_plan[0] = (a,) + self.ru_plan[0][1:]
                    self.ru_stats["plan_steps"] += 1
                    return a
                self.ru_plan = []
        goals = self.win.goals()
        if self.level == 0 and not self.ru_l1:
            goals = []
        if self.ru_probe:
            goals = goals[:3] + [("unknown", None)]
        if not goals:
            return None
        sig = (self.level, len(self.ru.rules), sum(sum(c.values()) for c in self.ru.rules.values()) // 25)
        if self.ru_fail == sig:
            return None
        budget = self.budget()
        left = None if budget is None else budget - self.life
        path = self.imagine(gm, acts_here, goals, left)
        if not path:
            self.ru_fail = sig
            self.ru_stats["search_fail"] += 1
            if self.ru_debug and self.ru_stats["search_fail"] <= 3:
                import sys
                print(f"RU_DEBUG {self.game_id} level {self.level} goals {goals} self {self.ru.self_colour} "
                      f"cells {self.ru.self_cells} left {left}\n" + "\n".join(" ".join(f"{v:2d}" for v in r) for r in gm),
                      file=sys.stderr, flush=True)
                for a in self._ru_actions(gm, acts_here)[:6]:
                    aid, axy = self._ru_action(a)
                    keys, ctx = self.ru.relation(rules.Scene(gm), aid, axy)
                    b = self.ru.best(keys)
                    print("   ", a, keys[0], b, None if b is None else round(self.ru.rate(b[0]), 2), file=sys.stderr)
            return None
        self.ru_stats["search_ok"] += 1
        self.ru_goals = [h for h in goals if h[0] != "unknown"]
        self.ru_plan = path
        a = self._ru_match(path[0][0], acts_here, gm)
        if a is None:
            self.ru_plan = []
            return None
        self.ru_plan[0] = (a,) + self.ru_plan[0][1:]
        return a

    @staticmethod
    def _ru_match(a, acts_here, g):
        """The real candidate for an imagined action: itself, or for a click another candidate on the same object."""
        if a in acts_here:
            return a
        if " " not in a:
            return None
        _, x, y = a.split()
        x, y = int(x), int(y)
        c = rules.colour_at(g, y, x)
        sc = rules.Scene(g)
        o = sc.object_at(y, x)
        if o is None:
            return None
        cells = set(zip(o[1].tolist(), o[2].tolist()))
        for b in acts_here:
            if " " in b:
                _, bx, by = b.split()
                if (int(by), int(bx)) in cells and rules.colour_at(g, int(by), int(bx)) == c:
                    return b
        return None

    def _ru_actions(self, g, acts_here):
        """The imagined action set: the keyboard actions here, and one click per object (its first cell)."""
        keys = [a for a in acts_here if " " not in a]
        if any(" " in a for a in acts_here):
            sc = rules.Scene(g)
            clicks = []
            for c in sorted(sc.counts):
                if c == sc.bg or c < 0:
                    continue
                for ys, xs in sc.comps(c)[1][:12]:
                    clicks.append(f"6 {int(xs[0])} {int(ys[0])}")
            keys += clicks[:48]
        return keys

    def imagine(self, g, acts_here, goals, left):
        """Breadth-first over predicted grids (`imagine`, below), with this agent's action set and cap."""
        acts = lambda gs: self._ru_actions(gs, acts_here)  # noqa: E731
        path, n, unknown = imagine(self.ru, g, acts, goals, left, self.ru_nodes)
        self.ru_plan_bold = False
        if not path and unknown and self.ru_opt and self.ru_bold.get(self.level, 0) < 3:
            # no plan the verified rules give: one more search where one untested move steps onto what it meets
            # (at most three such plans a level: a wrong one costs the life it is walked in)
            path, n2, _ = imagine(self.ru, g, acts, goals, left, self.ru_nodes, optimist=True)
            n += n2
            self.ru_stats["optimist_ok" if path else "optimist_fail"] += 1
            if path:
                self.ru_bold[self.level] = self.ru_bold.get(self.level, 0) + 1
                self.ru_plan_bold = True
        self.ru_stats["nodes"] += n
        return path

    # ---- the rules inside DG13's own explorer (RU_PRUNE) ------------------------------------------------------------
    def predict_next(self, n, a, t, par, graph=True):
        """DG13's prediction for an untried action, unless the rules predict its outcome confidently (a verified rate
        and a share of at least 0.8): then a predicted death is a death, a predicted known screen (or no change) is
        not new, and a predicted unseen screen is new."""
        pred = super().predict_next(n, a, t, par, graph)
        if not self.ru_prune or graph or pred["source"] != "rates" or self.raw:
            return pred
        g = getattr(self, "bs_key_grid", {}).get(n)
        if g is None:
            return pred
        ck = (n, a, self.ru.n_obs // 16)
        hit = self._ru_cache.get(ck)
        if hit is None:
            hit = ("?",)
            gm = self._ru_mask(g)
            aid, xy = self._ru_action(a)
            keep = self.ru.self_cells
            self.ru.self_cells = None
            try:
                pg, info = self.ru.predict(gm, aid, xy, ticks=False)
            finally:
                self.ru.self_cells = keep
            k = info.get("key")
            if k is not None and self.ru.rate(k) >= 0.8 and info.get("share", 0) >= 0.8:
                if info.get("death"):
                    hit = ("death",)
                elif pg is not None and pg.shape == g.shape:
                    h = g.copy()
                    h[pg >= 0] = pg[pg >= 0]
                    hit = ("next", self.key(h))
            if len(self._ru_cache) > 50000:
                self._ru_cache.clear()
            self._ru_cache[ck] = hit
        if hit[0] == "death":
            self.ru_stats["prune_death"] += 1
            return dict(pred, p_new=0.0, p_death=1.0)
        if hit[0] == "next":
            k = hit[1]
            known = k == n or k in self.graph or k in self.parent or k == self.start_key
            self.ru_stats["prune_known" if known else "prune_new"] += 1
            return dict(pred, p_new=0.0 if known else 1.0)
        return pred

    def ru_report(self):
        g = self.win.goals()
        return dict(stats=dict(self.ru_stats), ms=round(1000 * self.ru_s / max(1, self.n_steps), 2),
                    rules=len(self.ru.rules), self_colour=self.ru.self_colour, wins=self.win.wins,
                    goals=[list(map(str, h)) for h in g[:4]], unreadable=self.ru.log["unreadable"],
                    error=getattr(self, "ru_last_error", None))


def _action(a):
    if " " in a:
        _, x, y = a.split()
        return 6, (int(x), int(y))
    return int(a), None


def imagine(m, g, actions, goals, left, max_nodes, optimist=False):
    """Breadth-first over the grids the rule model `m` predicts from g: the shortest action sequence whose last step
    makes a goal true (`holds`). `actions(grid)` lists the imagined actions; predicted deaths and rules verified under
    1/3 are not taken; `left` caps the depth (the life's budget). Returns ([(action, id, xy, predicted grid)], the
    predictions made, whether an untested move was met). A goal ("unknown", None) also accepts a keyboard step no rule covers (a probe)."""
    probe = any(gl[0] == "unknown" for gl in goals)
    goals = [gl for gl in goals if gl[0] != "unknown"]
    keep = m.self_cells
    start = g.tobytes()
    parent = {start: None}
    grids = {start: (g, keep, rules.Scene(g))}
    q = collections.deque([start])
    depth = {start: 0}
    bold = {start: False}                                    # the path to this grid took an untested step
    unknown = [False]                                        # an untested move was met (else optimism adds nothing)
    n = 0
    found = None
    try:
        while q and n < max_nodes and found is None:
            s = q.popleft()
            gs, cells, sc0 = grids[s]
            if left is not None and depth[s] + 1 >= left:
                continue
            for a in actions(gs):
                aid, xy = _action(a)
                m.self_cells = cells
                keys, ctx = m.relation(sc0, aid, xy)
                b = m.best(keys)
                n += 1
                if b is None:
                    if ctx["kind"] == "mv":
                        unknown[0] = True
                    if probe and " " not in a:
                        found = (s, a, aid, xy, None)          # an untested relation: go and see
                        break
                    if not (optimist and ctx["kind"] == "mv") or bold[s]:
                        continue
                    out = (1, "under", ())                     # optimism, once per plan: the self steps onto it
                else:
                    key, out, share = b
                    if out == ("death",) or m.rate(key) < 0.34:
                        continue
                pg = rules.apply(sc0, ctx, out, m.self_colour, m.floor)
                if any(holds(gl, ctx, pg, out, aid, xy, gs) for gl in goals):
                    found = (s, a, aid, xy, pg)
                    break
                if pg is None:
                    continue
                t = pg.tobytes()
                if t in parent:
                    continue
                parent[t] = (s, a, aid, xy)
                m.self_cells = cells
                sc1 = rules.Scene(pg)                              # one reading per imagined grid
                ncells = m.locate_self(sc1)
                grids[t] = (pg, ncells, sc1)
                depth[t] = depth[s] + 1
                bold[t] = bold[s] or b is None
                q.append(t)
    finally:
        m.self_cells = keep
    if found is None:
        return [], n, unknown[0]
    s, a, aid, xy, pg = found
    path = [(a, aid, xy, pg)]
    while parent[s] is not None:
        ps, pa, paid, pxy = parent[s]
        path.append((pa, paid, pxy, grids[s][0]))
        s = ps
    path.reverse()
    return path, n, unknown[0]
