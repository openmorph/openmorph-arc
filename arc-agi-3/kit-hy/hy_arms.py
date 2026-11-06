# SPDX-License-Identifier: MIT-0
"""Kit HY's acting arms: kit X's explore v3 with one added term in its choice, read from the fact verifier.

`HY_ARM` (the gate rows say which arm ran):
  B     explore v3 unchanged (the verifier only observes);
  1     + TRUSTED facts gate the choice: an action whose death verdict is a verified yes is skipped while another is
        left, and an action predicted to change a never-changed cell gains `HY_W1` x P(novel), only once the novel
        fact's verified rate (AUC) is >= 0.75;
  13    + 1 + EXPERIMENTS: each candidate gains `HY_W3` x its expected information, the sum over the facts that touch
        the self, the counters, the effect and death of H(P) x (1 - skill), where H is the binary entropy of the
        predicted probability and skill = (AUC - 0.5) / 0.5 of that fact type over the game so far (0 while unrated),
        decaying as 1 / (1 + steps / 500): early in a game and fewer as the rates rise;
  134   + 13 + learn fastest from SURPRISE: a refuted confident fact is replayed `HY_REPLAY` (default 3) more passes.
AUTHORED: the arm rules above (a death is bad, a never-changed cell changing is worth seeing, the facts that matter),
the weights and the decay. LEARNED: every probability and every rate. Nothing names a game, a colour or a mechanic.
"""
import collections
import math
import os

import numpy as np

import explore
import facts
from explore import features


def entropy(P):
    P = min(1 - 1e-6, max(1e-6, P))
    return -(P * math.log2(P) + (1 - P) * math.log2(1 - P))


MATTER = ("effect", "self_moves", "death", "clock_tick", "novel")


class HyExplore(explore.ExploreAgent):
    """Explore v3, whose final scoring line adds `self.bonus(a)`; every other line is v3's."""

    def __init__(self, game_id, seed=1, verifier=None, arm="B"):
        super().__init__(game_id, seed)
        self.v, self.arm = verifier, arm
        self.w1 = float(os.environ.get("HY_W1", "0.5"))
        self.w3 = float(os.environ.get("HY_W3", "0.5"))
        self.n_steps = 0
        self._gv = None

    def bonus(self, a, cache):
        if self.arm == "B" or self._gv is None:
            return 0.0
        if a in cache:
            return cache[a]
        if " " in a:
            _, x, y = a.split()
            aid, xy = 6, (int(x), int(y))
        else:
            aid, xy = int(a), None
        if self._gv.shape != self.G.grid(self._frame).shape:
            return 0.0
        fs = {t: p for t, k, f, p in self.v.expect(self._gv, aid, xy, light=True) if f is not None}
        b = 0.0
        if self.v.verdict("death", fs["death"]) is True:
            b -= 5.0
        r = self.v.rate("novel")
        if r is not None and r >= 0.75:
            b += self.w1 * facts.prob(fs["novel"])
        if "3" in self.arm:
            info = 0.0
            for t in MATTER:
                if t in fs:
                    r = self.v.rate(t)
                    skill = 0.0 if r is None else max(0.0, (r - 0.5) / 0.5)
                    info += entropy(facts.prob(fs[t])) * (1.0 - skill)
            b += self.w3 * info / (1.0 + self.n_steps / 500.0)
        cache[a] = b
        return b

    def choose_action(self, frames, latest):
        self.n_steps += 1
        self._gv = None
        if self.v is not None and latest.frame:
            try:
                self._frame = np.asarray(latest.frame[-1], dtype=np.int64)
                self._gv = self.v.grid_of(latest.frame[-1])
            except Exception:
                self._gv = None
        return self._choose(frames, latest)

    # ---- v3's choose_action, with `+ self.bonus(a, cache)` in the score (marked HY)
    def _choose(self, frames, latest):
        import collections
        import muse
        from arcengine import GameAction, GameState
        from play import click_cells
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
                for i, f in enumerate(reversed((self.recent + [f0])[-20:])):
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
            for x, y in cells:
                c = int(g[y, x])
                size = next((len(comp) for comp in muse._components(g == c) if (y, x) in comp), 1)
                comp_of[(x, y)] = size
            acts += [f"6 {x} {y}" for x, y in cells]
        key = self.key(g)
        self.acts_of[key] = acts
        untried = [a for a in acts if a not in self.tried[key]]
        cache = {}
        if self.arm != "B":                                       # HY: a verified-death action is not "untried"
            safe = [a for a in untried if self.bonus(a, cache) > -4.0]
            untried = safe if safe else untried
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
                   + (0 if untried else self.rng.random() * 0.3)
                   + self.bonus(a, cache), -i, a)                 # HY
                  for i, a in enumerate(pool)]
        a = max(scored)[2]
        self.tried[key].add(a)
        self.prev = (g, a, features(a, g, comp_of))
        self.recent.append(self.prev[2])
        return self.to_action(a)


# ---------------------------------------------------------------------------------------------------------------------
# The arms on TK's planner (kit TK, `think.ThinkAgent`): B = kit TK itself. The search reads `predict_next(...)`'s
# p_new and p_death; for the CURRENT state only (the verifier has facts for the grid it sees, not for remembered
# states) the arms replace TK's per-type rates with the verifier's facts:
#   1    a TRUSTED verdict wins where one exists: effect verified "no" -> p_new 0 (nothing changes, so no new state);
#        novel verified "yes" -> p_new 1; death verified "yes" -> p_death 1, "no" -> p_death min(TK's, 1 - tau);
#        otherwise TK's rates;
#   13   + the expected information (as for explore, MATTER facts, decaying with steps) added to p_new;
#   134  + surprise replay.
#   1v   as 1, but death is not a verdict: p_death = the death rate VERIFIED in this prediction's probability bin over
#        the game (Laplace, >= 10 checks), else TK's. A "no" verdict on a rare fact says nothing beyond its base rate
#        (sp80: every one of 21 deaths fell under a trusted "no"), so arm 1 made risky moves look cheap.
# AUTHORED: these substitutions. LEARNED: every probability, rate and verdict.
# ---------------------------------------------------------------------------------------------------------------------
import think  # noqa: E402


class HyThink(think.ThinkAgent):
    def __init__(self, game_id, seed=1, verifier=None, arm="B"):
        super().__init__(game_id, seed)
        self.v, self.arm = verifier, arm
        self.w3 = float(os.environ.get("HY_W3", "0.5"))
        self._plan_key, self._gv, self._cache = None, None, {}
        self.hy_stats = collections.Counter()
        self.trace = [] if os.environ.get("HY_TRACE") else None   # (step, action, p_new TK -> HY, p_death TK -> HY)

    def choose_action(self, frames, latest):
        self._gv, self._cache = None, {}
        if self.v is not None and latest.frame:
            try:
                self._gv = self.v.grid_of(latest.frame[-1])
            except Exception:
                self._gv = None
        return super().choose_action(frames, latest)

    def plan(self, key, acts_here, g, comp_of):
        self._plan_key = key if (self._gv is not None and not self.raw and g.shape == self._gv.shape) else None
        return super().plan(key, acts_here, g, comp_of)

    def facts_for(self, a):
        if a in self._cache:
            return self._cache[a]
        if " " in a:
            _, x, y = a.split()
            aid, xy = 6, (int(x), int(y))
        else:
            aid, xy = int(a), None
        fs = {t: p for t, k, f, p in self.v.expect(self._gv, aid, xy, light=True) if f is not None}
        self._cache[a] = fs
        return fs

    def predict_next(self, n, a, t, par, graph=True):
        pred = super().predict_next(n, a, t, par, graph)
        if self.arm == "B" or self.v is None or n != self._plan_key or pred["source"] != "rates":
            return pred
        fs = self.facts_for(a)
        v = self.v
        p_new, p_death = pred["p_new"], pred["p_death"]
        if v.verdict("effect", fs["effect"]) is False:
            p_new = 0.0
            self.hy_stats["effect_no"] += 1
        elif v.verdict("novel", fs["novel"]) is True:
            p_new = 1.0
            self.hy_stats["novel_yes"] += 1
        dv = v.verdict("death", fs["death"])
        if "v" in self.arm:                                       # arm 1v: the verified rate of this bin, not a verdict
            r = v.bin_rate("death", fs["death"])
            if r is not None:
                p_death = r
                self.hy_stats["death_bin"] += 1
        elif dv is True:
            p_death = 1.0
            self.hy_stats["death_yes"] += 1
        elif dv is False:
            p_death = min(p_death, 1.0 - v.tau)
            self.hy_stats["death_no"] += 1
        if "3" in self.arm:
            info = 0.0
            for ft in MATTER:
                if ft in fs:
                    r = v.rate(ft)
                    skill = 0.0 if r is None else max(0.0, (r - 0.5) / 0.5)
                    info += entropy(facts.prob(fs[ft])) * (1.0 - skill)
            p_new = p_new + self.w3 * info / (1.0 + self.n_steps / 500.0)
        if self.trace is not None and len(self.trace) < 4000 and (p_new, p_death) != (pred["p_new"], pred["p_death"]):
            self.trace.append((self.n_steps, a, round(pred["p_new"], 3), round(p_new, 3), round(pred["p_death"], 3),
                               round(p_death, 3)))
        return dict(pred, p_new=p_new, p_death=p_death)
