# SPDX-License-Identifier: MIT-0
"""`explore`: an agent with no authored solver. It grows a model of WHICH actions change the world and spends its
budget on them.

    python explore.py ENV [--games a,b] [--jobs 4] --out runs/explore.json

Per step, on the logical grid (`muse.Grid`):
* CLOCK. Cells that change in most transitions (the step/energy bar every real game draws, a timer) are learned as
  the clock: a cell is clock when it changed in >= 60 % of the level's transitions (after 8 of them). An action's
  EFFECT is the change outside the clock.
* EFFECT MODEL. A `Field` readout (the RLS readout, `Config::readout_rls` 1, P0 1: the morph's ridge) learns, from
  every transition, how likely an action is to have an effect: inputs are the action's id (one-hot), and for a click
  the clicked object's colour (one-hot), log size and whether it is one cell; target +0.9 effect, -0.9 none, -0.9
  for a death.
* PROGRESS. A second RLS readout over the same features learns from completions: the completing action gets +0.9,
  the 19 before it 0.9 * 0.85^k. It carries what finished one level into the next.
* CHOICE. Score = predicted effect + predicted progress. Among the actions not yet tried in this exact grid, the one the model predicts most effective (ties: the
  first); when every action was tried here, the most effective one with a small seeded noise. A death marks the
  (grid, action) as tried. RESET only after a game over.
Nothing names a game, a colour or a mechanic; what is effective is learned per game from its own transitions.
"""
import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "gate"), os.path.join(HERE, "weights")]
import muse  # noqa: E402
from play import click_cells  # noqa: E402

N_FEAT = 7 + 16 + 2


class EffectModel:
    """The morph's RLS readout over action features (see `weights/eval.py` RlsHead)."""

    def __init__(self, seed=0):
        from eval import RlsHead
        self.head = RlsHead(N_FEAT, prior=1.0, backbone=0, seed=seed)
        self.n = 0

    def learn(self, f, target):
        self.head.learn(f, target)
        self.n += 1

    def predict(self, f):
        return self.head.predict(f) if self.n else 0.0


def features(a, g, comp_of):
    f = np.zeros(N_FEAT)
    if " " in a:
        _, x, y = a.split()
        x, y = int(x), int(y)
        f[5] = 1.0                                   # action 6
        c = int(g[y, x]) if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] else 0
        f[7 + c] = 1.0
        size = comp_of.get((x, y), 1)
        f[23] = np.log1p(size) / 5.0
        f[24] = 1.0 if size == 1 else 0.0
    else:
        f[int(a) - 1] = 1.0
    return f


class ExploreAgent:
    def __init__(self, game_id, seed=1):
        self.rng = random.Random(f"{game_id}:{seed}:explore")
        self.model = EffectModel()
        self.progress = EffectModel(seed=1)         # credit from completions, discounted backward
        self.recent = []                             # this level's action features, newest last
        self.w_progress = float(os.environ.get("EXPLORE_PROGRESS", "1.0"))
        self.frames, self.G = [], None
        self.level = 0
        self.prev = None                              # (grid, action, features)
        self.tried = collections.defaultdict(set)
        self.change_count = None
        self.n_trans = 0
        self.graph = collections.defaultdict(dict)    # state key -> {action: next state key}
        self.acts_of = {}                             # state key -> its candidate actions
        self.route = []                               # actions toward the nearest state with untried actions
        self.frontier = bool(int(os.environ.get("EXPLORE_FRONTIER", "1")))
        self.stats = collections.Counter()

    def grid_of(self, frame):
        f = np.asarray(frame, dtype=np.int64)
        if len(self.frames) < 12 and not any((f == x).all() for x in self.frames):
            self.frames.append(f)
            self.G = muse.Grid(self.frames)
        return self.G.grid(f)

    def to_action(self, a):
        from arcengine import GameAction
        if " " in a:
            _, x, y = a.split()
            G = self.G
            act = GameAction.from_id(6)
            act.set_data({"x": int(min(63, max(0, G.ox + (int(x) + G.x0) * G.k + G.k // 2))),
                          "y": int(min(63, max(0, G.oy + (int(y) + G.y0) * G.k + G.k // 2)))})
            return act
        return GameAction.from_id(int(a))

    def effect(self, a, b):
        """Changed cells outside the learned clock."""
        if a.shape != b.shape:
            return True
        d = a != b
        if self.change_count is None or self.change_count.shape != d.shape:
            self.change_count = np.zeros(d.shape)
            self.n_trans = 0
        self.change_count += d
        self.n_trans += 1
        clock = (self.change_count >= 0.6 * self.n_trans) if self.n_trans >= 8 else np.zeros(d.shape, bool)
        return bool((d & ~clock).any())

    def clock_mask(self, shape):
        if self.change_count is None or self.change_count.shape != shape or self.n_trans < 8:
            return np.zeros(shape, bool)
        return self.change_count >= 0.6 * self.n_trans

    def key(self, g):
        """The state with the learned clock masked out (the step bar must not make every state new)."""
        h = g.copy()
        h[self.clock_mask(g.shape)] = -1
        return hashlib.blake2b(h.tobytes(), digest_size=8).hexdigest()

    def route_to_frontier(self, start):
        """Shortest known action path from `start` to a state that still has untried actions."""
        prev = {start: None}
        q = collections.deque([start])
        while q:
            k = q.popleft()
            if k != start and any(a not in self.tried[k] for a in self.acts_of.get(k, [])):
                path = []
                while prev[k] is not None:
                    k, a = prev[k]
                    path.append(a)
                return path[::-1]
            for a, n in self.graph[k].items():
                if n not in prev:
                    prev[n] = (k, a)
                    q.append(n)
        return []

    def is_done(self, frames, latest):
        from arcengine import GameState
        return latest.state is GameState.WIN

    def choose_action(self, frames, latest):
        from arcengine import GameAction, GameState
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or not latest.frame:
            if self.prev is not None and latest.state is GameState.GAME_OVER:
                g0, a0, f0 = self.prev
                self.model.learn(f0, -0.9)
                self.stats["deaths"] += 1
            self.prev = None
            return GameAction.RESET
        g = self.grid_of(latest.frame[-1])
        lv = int(latest.levels_completed or 0)
        if self.prev is not None:
            g0, a0, f0 = self.prev
            if lv > self.level:
                self.model.learn(f0, 0.9)
                for i, f in enumerate(reversed((self.recent + [f0])[-20:])):  # the completing action first
                    self.progress.learn(f, 0.9 * (0.85 ** i))
                self.recent = []
                self.level = lv
                self.frames, self.G, self.change_count = [], None, None
                self.tried = collections.defaultdict(set)
                self.graph, self.acts_of, self.route = collections.defaultdict(dict), {}, []
                g = self.grid_of(latest.frame[-1])
                self.stats["levels"] += 1
            else:
                eff = self.effect(g0, g)
                self.model.learn(f0, 0.9 if eff else -0.9)
                self.stats["effective" if eff else "no_effect"] += 1
                self.graph[self.key(g0)][a0] = self.key(g)
        avail = sorted(set(int(v) for v in (latest.available_actions or [])))
        acts = [str(a) for a in avail if a not in (0, 6)]
        comp_of = {}
        if 6 in avail:
            cells = click_cells(g, cap=32)
            vals, counts = np.unique(g, return_counts=True)
            for x, y in cells:
                c = int(g[y, x])
                size = next((len(comp) for comp in muse._components(g == c) if (y, x) in comp), 1)
                comp_of[(x, y)] = size
            acts += [f"6 {x} {y}" for x, y in cells]
        key = self.key(g)
        self.acts_of[key] = acts
        untried = [a for a in acts if a not in self.tried[key]]
        if self.frontier and not untried:
            if not self.route:
                self.route = self.route_to_frontier(key)
            if self.route:
                a = self.route.pop(0)
                if a in acts:
                    self.stats["route_steps"] += 1
                    self.prev = (g, a, features(a, g, comp_of))
                    return self.to_action(a)
                self.route = []
        else:
            self.route = []
        pool = untried or acts
        scored = [(self.model.predict(features(a, g, comp_of))
                   + self.w_progress * self.progress.predict(features(a, g, comp_of))
                   + (0 if untried else self.rng.random() * 0.3), -i, a)
                  for i, a in enumerate(pool)]
        a = max(scored)[2]
        self.tried[key].add(a)
        self.prev = (g, a, features(a, g, comp_of))
        self.recent.append(self.prev[2])
        return self.to_action(a)


def main():
    import replay

    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--games", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="all", choices=["all", "dev", "sealed"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    orig = replay.make_agent

    def make_agent(kind, game_id, seed, env, meta):
        return ExploreAgent(game_id, seed) if kind == "explore" else orig(kind, game_id, seed, env, meta)

    replay.make_agent = make_agent
    games = replay.discover(args.env, 40, "openmorph-gate-1")
    if args.split != "all":
        games = [g for g in games if g["split"] == args.split]
    if args.games:
        keep = set(args.games.split(","))
        games = [g for g in games if g["base"] in keep or g["game_id"] in keep]
    opts = dict(env=args.env, max_actions=1000, offline_resets=False)
    rows = []
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        for r in pool.imap_unordered(replay._run_safe, [("explore", args.seed, g, opts) for g in games]):
            rows.append(r)
            print(f"{r.get('game'):18} [{r.get('split')}] score {r.get('score', 0):6.2f}  levels "
                  f"{r.get('levels', 0)}/{r.get('n_levels', '?')}  deaths {r.get('deaths', 0)}  {r.get('error', '')}",
                  flush=True)
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("split")].append(r)
    for s, rs in sorted(by.items()):
        print(f"FINAL {s:7} {sum(r.get('score', 0) for r in rs) / len(rs):6.3f}  levels "
              f"{sum(r.get('levels', 0) for r in rs)}")
    print(f"FINAL all     {sum(r.get('score', 0) for r in rows) / len(rows):6.3f}  levels "
          f"{sum(r.get('levels', 0) for r in rows)}")
    json.dump(dict(rows=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
