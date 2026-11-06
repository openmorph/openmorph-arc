# SPDX-License-Identifier: MIT-0
"""`think`: kit X's explorer (`explore.ExploreAgent`, v3) that plans before it acts (stream TK, 2026-10-26).

Compute is free and actions are scored, so each decision searches the agent's own models for the action with the
most expected discovery per real action spent. All of it is a usage protocol over v3's state graph and heads; nothing
here is in a crate.

The imagined world, per decision:
* ON THE GRAPH (never wrong): the observed transitions `graph[state][action] -> state`, searched breadth-first from
  the current state up to `TK_DEPTH` real actions (1, 4, 16, 64). Known deaths are edges the search never takes.
* OFF THE GRAPH (an untried action in a reached state): its outcome is predicted only from rates VERIFIED on this
  game's own real tries, per action type (the key, or a click's colour and whether it is one cell): the share of
  first tries that reached a NEW state, a known one, no change, or a death. One more verified rate, per ordered pair
  of action types: how often `b` right after `a` returned to the state before `a` (an undo). An untried `b` in a
  state entered by `a` is predicted to lead back at that rate.
* VALUE: gain = P(new state) + w_p * progress(action) (v3's progress head, learned from completions);
  cost = the path's actions + 1 + P(death) * (1 + the known distance from the level's start back to that state).
  The pick is the best gain / cost; the first action of its path is taken, its real outcome checked, and the next
  decision re-plans (MPC). Nothing in reach -> v3's own choice (its frontier route, then its noise).

AUTHORED (labelled, as the rules ask): the outcome classes (new / known / none / death), the Beta(1,1)-style priors,
the gain / cost form, the undo pairing and the depth horizon. LEARNED per game: every rate, the graph, v3's two
heads and v3's clock. Nothing names a game, a colour or a mechanic.

Switches (environment): TK_DEPTH (default 16), TK_PROGRESS (w_p, default 1.0), TK_UNDO (1), TK_DEATH (1).
"""
import collections
import os
import time

import numpy as np

import explore
from explore import features


class ThinkAgent(explore.ExploreAgent):
    def __init__(self, game_id, seed=1):
        super().__init__(game_id, seed)
        self.depth = int(os.environ.get("TK_DEPTH", "16"))
        self.w_p = float(os.environ.get("TK_PROGRESS", "1.0"))
        self.use_undo = bool(int(os.environ.get("TK_UNDO", "1")))
        self.use_death = bool(int(os.environ.get("TK_DEATH", "1")))
        # verified outcome counts of first tries, per action type: [new, known, none, death]
        self.outcome = collections.defaultdict(lambda: [0, 0, 0, 0])
        self.undo = collections.defaultdict(lambda: [0, 0])     # (type a, type b) -> [undid, tries]
        self.parent = {}                                          # state -> (previous state, action type into it)
        self.dead = set()                                         # (state, action) known deaths
        self.start_key = None                                     # the level's first state (where a death returns)
        self.pend = None                                          # (state key, action, type, first try?, parent)
        self.think_s = 0.0
        self.pred_log = collections.defaultdict(lambda: [0, 0])   # source -> [one-step predictions right, made]
        self.pend_key, self.pend_pred = None, None
        self.n_steps = 0
        self.game_id = game_id
        self.diag = os.environ.get("TK_DIAG", "")                # a directory: one JSONL line per 100 steps (off)
        self.kinds_of = {}                                        # state -> {action: type}
        self.awaiting_start = True
        self.use_raw = bool(int(os.environ.get("TK_RAW", "1")))
        self.raw = False                                          # reading the raw frame (after the grid was blind)
        self.level_steps = 0
        self.use_budget = bool(int(os.environ.get("TK_BUDGET", "1")))
        self.use_commit = bool(int(os.environ.get("TK_COMMIT", "1")))
        self.life = 0                                             # actions since this life began
        self.life_ends = []                                       # life lengths that ended in a death
        self.goal = None                                          # the committed target (state, untried action)
        self.v3_order = bool(int(os.environ.get("TK_V3ORDER", "0")))  # untried actions here: v3's order (off)
        self.arbiter = bool(int(os.environ.get("TK_ARBITER", "0")))   # ... when v3's head is better calibrated (off)
        self.brier = [0.0, 0.0, 0]                                    # P(new) squared error: planner, v3, count
        self.pend_brier, self.pend_first = None, False
        self.line = int(os.environ.get("TK_LINE", "1"))           # v4's line clock: 0 off, 1 in raw mode, 2 always

    # ---- the blind-reading fallback (AUTHORED rule) ----------------------------------------------------------------
    class _Raw:
        """The identity reading: one cell per pixel (the `Grid` interface `to_action` uses)."""
        k, ox, oy, x0, y0 = 1, 0, 0, 0, 0

    def grid_of(self, frame):
        if self.raw:
            return np.asarray(frame, dtype=np.int64)
        return super().grid_of(frame)

    def blind(self):
        """The logical grid shows (almost) nothing this level: every known state's actions tried, or under one state
        per ten actions after 200. Verified on this game's own play, so only a stagnant level switches."""
        if not self.use_raw or self.raw or self.level_steps < 50:
            return False
        front = any(any(x not in self.tried[k] for x in acts) for k, acts in self.acts_of.items())
        return not front or (self.level_steps >= 200 and 10 * len(self.acts_of) < self.level_steps)

    def go_raw(self):
        self.raw = True
        self.G = self._Raw()
        self.frames, self.change_count, self.row_count = [], None, None
        self.tried = collections.defaultdict(set)
        self.graph, self.acts_of, self.route = collections.defaultdict(dict), {}, []
        self.parent, self.dead, self.kinds_of = {}, set(), {}
        self.start_key, self.prev, self.pend, self.goal = None, None, None, None
        self.stats["went_raw"] = self.n_steps

    # ---- v4's line clock (gpu-track 34e1a31d, AUTHORED there: border lines changing by <= 2 cells at a time) -------
    def line_on(self):
        return self.line == 2 or (self.line == 1 and self.raw)

    def effect(self, a, b):
        if not self.line_on() or a.shape != b.shape:
            return super().effect(a, b)
        d = a != b
        if (self.change_count is None or self.change_count.shape != d.shape or getattr(self, "row_count", None) is None
                or self.row_count.shape[0] != d.shape[0] or self.col_count.shape[0] != d.shape[1]):
            self.change_count = np.zeros(d.shape)
            self.row_count, self.col_count = np.zeros(d.shape[0]), np.zeros(d.shape[1])
            self.row_small, self.col_small = np.zeros(d.shape[0]), np.zeros(d.shape[1])
            self.n_trans = 0
        self.change_count += d
        rows, cols = d.sum(1), d.sum(0)
        self.row_count += rows > 0
        self.col_count += cols > 0
        self.row_small += (rows > 0) & (rows <= 2)
        self.col_small += (cols > 0) & (cols <= 2)
        self.n_trans += 1
        return bool((d & ~self.clock_mask(d.shape)).any())

    def clock_mask(self, shape):
        m = super().clock_mask(shape)
        if not self.line_on() or self.n_trans < 8 or getattr(self, "row_count", None) is None \
                or self.row_count.shape[0] != shape[0] or self.col_count.shape[0] != shape[1]:
            return m
        n = self.n_trans
        h, w = shape
        edge_r = np.zeros(h, bool)
        edge_r[:2] = edge_r[-2:] = True
        edge_c = np.zeros(w, bool)
        edge_c[:2] = edge_c[-2:] = True
        m = m.copy()
        m[edge_r & (self.row_count >= 0.6 * n) & (self.row_small >= 0.9 * self.row_count), :] = True
        m[:, edge_c & (self.col_count >= 0.6 * n) & (self.col_small >= 0.9 * self.col_count)] = True
        return m

    # ---- action types -------------------------------------------------------------------------------------------
    @staticmethod
    def kind(a, g, comp_of):
        if " " in a:
            _, x, y = a.split()
            x, y = int(x), int(y)
            c = int(g[y, x]) if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] else 0
            return f"6:{c}:{int(comp_of.get((x, y), 1) == 1)}"
        return a

    def rates(self, t):
        n, k, z, d = self.outcome[t]
        tot = n + k + z + d
        p_new = (n + 1.0) / (tot + 2.0)
        p_death = (d + 0.1) / (tot + 2.0) if self.use_death else 0.0
        return p_new, p_death

    def undo_rate(self, ta, tb):
        u, n = self.undo[(ta, tb)]
        return (u + 0.0) / (n + 1.0)

    # ---- the one-step prediction (THE HOOK: what will the next state be) --------------------------------------------
    def predict_next(self, n, a, t, par, graph=True):
        """What taking `a` in state `n` (action type `t`, entered from `par` = (state, type) or None) leads to.

        Returns {"source", "next", "p_new", "p_death", "p_undo"}: on a seen edge, source "graph" and `next` the exact
        next state key; a known death, source "dead"; otherwise source "rates", `next` None and the probabilities
        verified on this game's own tries (`graph=False`: the search's call, which scores untried actions by rates as
        measured, even where a clock-mask change left an edge under the same key). The search reads only `p_new` and
        `p_death` off the graph, so a model of
        trusted facts (stream HY) can replace this method (subclass or assign) without touching the search."""
        if (n, a) in self.dead:
            return dict(source="dead", next=None, p_new=0.0, p_death=1.0, p_undo=0.0)
        nxt = self.graph.get(n, {}).get(a) if graph else None
        if nxt is not None:
            return dict(source="graph", next=nxt, p_new=0.0, p_death=0.0, p_undo=1.0 if par and nxt == par[0] else 0.0)
        p_new, p_death = self.rates(t)
        u = self.undo_rate(par[1], t) if self.use_undo and par is not None else 0.0
        if self.use_undo and par is not None:
            p_new *= 1.0 - u
        return dict(source="rates", next=None, p_new=p_new, p_death=p_death, p_undo=u)

    def predicted_class(self, pred, t):
        """The single most likely outcome, for scoring the prediction (logging only; the search never reads it)."""
        if pred["source"] == "dead":
            return "death"
        if pred["source"] == "graph":
            return "none" if pred["next"] == self.pend_key else "known"
        n, k, z, d = self.outcome[t]
        tot = n + k + z + d + 2.0
        q = {"new": (n + 0.5) / tot, "known": (k + 0.5) / tot, "none": (z + 0.5) / tot, "death": (d + 0.5) / tot}
        q["known"] += q["new"] * pred["p_undo"]
        q["new"] *= 1.0 - pred["p_undo"]
        return max(("new", "known", "none", "death"), key=lambda c: q[c])

    def note_prediction(self, key, a, t, par):
        self.pend_key = key
        self.pend_first = bool(self.pend is not None and self.pend[3])     # this is the action's first try here
        pred = self.predict_next(key, a, t, par)
        self.pend_pred = (pred["source"], pred["next"], self.predicted_class(pred, t))
        # both orderers' P(new state) for this action, scored on first tries off the graph (the arbiter)
        f = self.prev[2] if self.prev is not None else None
        v3 = min(1.0, max(0.0, (self.model.predict(f) + 1.0) / 2.0)) if f is not None else None
        self.pend_brier = (pred["p_new"], v3) if pred["source"] == "rates" and v3 is not None else None

    def score_prediction(self, k0, k_now, died, known_before):
        src, nxt, cls = self.pend_pred
        actual = "death" if died else ("none" if k_now == k0 else ("known" if known_before else "new"))
        ok = (nxt == k_now) if (src == "graph" and not died) else (cls == actual)
        if self.pend_brier is not None and k0 is not None and self.pend_first:
            y = 1.0 if actual == "new" else 0.0
            self.brier[0] += (self.pend_brier[0] - y) ** 2
            self.brier[1] += (self.pend_brier[1] - y) ** 2
            self.brier[2] += 1
        rec = self.pred_log[src]
        rec[0] += int(ok)
        rec[1] += 1

    # ---- the real outcome of the last action -----------------------------------------------------------------------
    def observe(self, k_now, died, leveled):
        if self.pend is None:
            return
        k0, a, t, first, par = self.pend
        self.pend = None
        if leveled:
            return
        if died:
            self.score_prediction(k0, None, True, False)
            self.dead.add((k0, a))
            if first:
                self.outcome[t][3] += 1
            return
        known_before = k_now in self.graph or k_now in self.parent or k_now == self.start_key
        self.score_prediction(k0, k_now, False, known_before)
        if first:
            self.outcome[t][2 if k_now == k0 else (1 if known_before else 0)] += 1
        if par is not None and k_now != k0:
            pk, pt = par
            self.undo[(pt, t)][1] += 1
            if k_now == pk:
                self.undo[(pt, t)][0] += 1
        if k_now != k0 and k_now not in self.parent and k_now != self.start_key:
            self.parent[k_now] = (k0, t)

    # ---- the search -------------------------------------------------------------------------------------------------
    def bfs(self, start, limit):
        dist, first = {start: 0}, {start: None}
        q = collections.deque([start])
        while q:
            k = q.popleft()
            if dist[k] >= limit:
                continue
            for a, n in self.graph[k].items():
                if n not in dist and (k, a) not in self.dead:
                    dist[n] = dist[k] + 1
                    first[n] = a if k == start else first[k]
                    q.append(n)
        return dist, first

    def budget(self):
        """A life's step budget, once verified: the two longest lives ended within 2 actions of each other."""
        if not self.use_budget or len(self.life_ends) < 2:
            return None
        a, b = sorted(self.life_ends)[-2:]
        return a if b - a <= 2 else None

    def plan(self, key, acts_here, g, comp_of):
        if self.v3_order or (self.arbiter and self.brier[2] >= 10 and self.brier[1] < self.brier[0]):
            here = [a for a in acts_here if a not in self.tried[key] and (key, a) not in self.dead]
            if here:                                   # v3's own pick where v3 had one (its effect + progress order)
                self.goal = None
                return max((self.model.predict(features(a, g, comp_of))
                            + self.w_progress * self.progress.predict(features(a, g, comp_of)), -i, a)
                           for i, a in enumerate(here))[2]
        dist, first = self.bfs(key, self.depth)
        budget = self.budget()
        left = None if budget is None else budget - self.life       # actions before the budget kills this life
        if self.use_commit and self.goal is not None:
            n, a = self.goal
            if (n in dist and a not in self.tried[n] and (n, a) not in self.dead
                    and (left is None or dist[n] + 1 < left)):
                return a if n == key else first[n]
            self.goal = None
        back = self.bfs(self.start_key, 10 ** 6)[0] if self.start_key is not None else {}
        best = None
        for n, d in dist.items():
            if left is not None and d + 1 >= left:
                continue
            acts = acts_here if n == key else self.acts_of.get(n, [])
            par = self.parent.get(n)
            for i, a in enumerate(acts):
                if a in self.tried[n] or (n, a) in self.dead:
                    continue
                t = self.kind(a, g, comp_of) if n == key else self.kinds_of.get(n, {}).get(a, a)
                pred = self.predict_next(n, a, t, par, graph=False)
                p_new, p_death = pred["p_new"], pred["p_death"]
                gain = p_new
                if self.w_p and n == key:
                    gain += self.w_p * max(0.0, self.progress.predict(features(a, g, comp_of)))
                cost = d + 1 + p_death * (1 + back.get(n, d))
                s = (gain / cost, -d, -i)
                if best is None or s > best[0]:
                    best = (s, n, a)
        if best is None:
            return None
        _, n, a = best
        self.goal = (n, a) if n != key else None
        return a if n == key else first[n]

    # ---- one decision -------------------------------------------------------------------------------------------------
    def choose_action(self, frames, latest):
        from arcengine import GameState
        t0 = time.perf_counter()
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or not latest.frame:
            if latest.state is GameState.GAME_OVER:
                self.observe(None, True, False)
                if not self.awaiting_start:
                    self.life_ends.append(self.life)
                self.goal = None
            self.awaiting_start = True
            a = super().choose_action(frames, latest)       # v3: learns the death, returns RESET
        else:
            a = self._think(latest)
        self.think_s += time.perf_counter() - t0
        self.n_steps += 1
        if self.diag and self.n_steps % 100 == 0:
            self.dump()
        return a

    def dump(self):
        import json
        front = sum(1 for k, acts in self.acts_of.items() if any(x not in self.tried[k] for x in acts))
        rec = dict(game=self.game_id, step=self.n_steps, level=self.level, nodes=len(self.acts_of), frontier=front,
                   edges=sum(len(v) for v in self.graph.values()), dead=len(self.dead), stats=dict(self.stats),
                   outcome={t: v for t, v in self.outcome.items()}, pred=dict(self.pred_log), brier=self.brier, think_ms=round(1000 * self.think_s / self.n_steps, 3))
        with open(os.path.join(self.diag, f"{self.game_id}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _think(self, latest):
        self.level_steps += 1
        if self.blind():
            self.go_raw()
        import muse
        from play import click_cells
        # v3's bookkeeping (identical to explore.ExploreAgent.choose_action up to its choice)
        g = self.grid_of(latest.frame[-1])
        lv = int(latest.levels_completed or 0)
        if self.prev is not None:
            g0, a0, f0 = self.prev
            if lv > self.level:
                self.model.learn(f0, 0.9)
                for i, f in enumerate(reversed((self.recent + [f0])[-20:])):
                    self.progress.learn(f, 0.9 * (0.85 ** i))
                self.recent = []
                self.level = lv
                self.frames, self.G, self.change_count = [], None, None
                self.tried = collections.defaultdict(set)
                self.graph, self.acts_of, self.route = collections.defaultdict(dict), {}, []
                g = self.grid_of(latest.frame[-1])
                self.stats["levels"] += 1
                self.parent, self.dead, self.kinds_of = {}, set(), {}
                self.start_key, self.goal = None, None
                self.level_steps = 0
                self.life_ends = []
                if self.raw:
                    self.G = self._Raw()
                self.observe(None, False, True)
                self.awaiting_start = True
            else:
                eff = self.effect(g0, g)
                self.model.learn(f0, 0.9 if eff else -0.9)
                self.stats["effective" if eff else "no_effect"] += 1
                self.graph[self.key(g0)][a0] = self.key(g)
        avail = sorted(set(int(v) for v in (latest.available_actions or [])))
        acts = [str(a) for a in avail if a not in (0, 6)]
        comp_of = {}
        if 6 in avail:
            cells = click_cells(g, cap=64 if self.raw else 32)
            for x, y in cells:
                c = int(g[y, x])
                size = next((len(comp) for comp in muse._components(g == c) if (y, x) in comp), 1)
                comp_of[(x, y)] = size
            acts += [f"6 {x} {y}" for x, y in cells]
        if not acts:
            # no candidate at all (a click-only level whose logical grid is one colour): the reading is blind now,
            # not after 50 steps; v3 (and this agent before) raised here and the wrapper clicked the frame's centre
            if self.use_raw and not self.raw:
                self.go_raw()
                return self._think(latest)
            if 6 in avail:
                acts = [f"6 {g.shape[1] // 2} {g.shape[0] // 2}"]
            else:
                from arcengine import GameAction
                return GameAction.RESET
        key = self.key(g)
        if self.awaiting_start:
            self.start_key = key
            self.awaiting_start = False
            self.life = 0
        self.life += 1
        self.observe(key, False, False)
        self.acts_of[key] = acts
        self.kinds_of[key] = {a: self.kind(a, g, comp_of) for a in acts}
        a = self.plan(key, acts, g, comp_of)
        if a is not None and a in acts:
            self.stats["planned"] += 1
            self.route = []
            first = a not in self.tried[key]
            if first:
                self.tried[key].add(a)
                self.recent.append(features(a, g, comp_of))
            self.prev = (g, a, features(a, g, comp_of))
            self.pend = (key, a, self.kinds_of[key][a], first, self.parent.get(key))
            self.note_prediction(key, a, self.kinds_of[key][a], self.parent.get(key))
            return self.to_action(a)
        # nothing untried within reach: v3's own choice (its frontier route, then its noise)
        untried = [x for x in acts if x not in self.tried[key]]
        if self.frontier and not untried:
            if not self.route:
                self.route = self.route_to_frontier(key)
            if self.route:
                a = self.route.pop(0)
                if a in acts:
                    self.stats["route_steps"] += 1
                    self.prev = (g, a, features(a, g, comp_of))
                    self.pend = (key, a, self.kinds_of[key][a], False, self.parent.get(key))
                    self.note_prediction(key, a, self.kinds_of[key][a], self.parent.get(key))
                    return self.to_action(a)
                self.route = []
        else:
            self.route = []
        pool = untried or acts
        scored = [(self.model.predict(features(x, g, comp_of))
                   + self.w_progress * self.progress.predict(features(x, g, comp_of))
                   + (0 if untried else self.rng.random() * 0.3), -i, x) for i, x in enumerate(pool)]
        a = max(scored)[2]
        first = a not in self.tried[key]
        self.tried[key].add(a)
        self.prev = (g, a, features(a, g, comp_of))
        self.recent.append(self.prev[2])
        self.pend = (key, a, self.kinds_of[key][a], first, self.parent.get(key))
        self.note_prediction(key, a, self.kinds_of[key][a], self.parent.get(key))
        self.stats["fallback"] += 1
        return self.to_action(a)
