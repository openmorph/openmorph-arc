# SPDX-License-Identifier: MIT-0
"""The muse in the loop: an LLM writes the game's rules as code, a planner searches them, reality corrects them.

    python play.py ENV --base-url http://host:port/v1 --model NAME [--games a,b] [--jobs 8] --out runs/x.json

Plays real `arcengine` games through the real-engine gate (`../gate/replay.py`: the real `arc_agi` scorecard,
competition resets, MAX_ACTIONS 1000), with the agent `muse`:

1. probe: every simple action once (and clicks on object centres) from the level's first frames;
2. model: the LLM writes `step(grid, action)` and `is_goal(grid)` from the object scene, the grid and the recorded
   transitions of this level (`muse.py` describes them);
3. verify: `step` is run (sandboxed) on every recorded transition; mismatches go back for a revision (2 rounds);
4. plan: breadth-first search over the LLM's `step` (in the sandbox) to a grid where `is_goal` holds;
5. act: the plan is executed; each real grid is compared with the predicted one; the first surprise ends the plan,
   records the transition and asks for a revision;
6. no plan, or the LLM budget spent: explore an action not yet tried in the current state.

Every prompt and answer is appended to `--log` (JSONL). The LLM is any OpenAI-compatible server (`ask.py`).
"""
import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "gate")]
import ask  # noqa: E402
import muse  # noqa: E402

SYSTEM = ("You are the muse of a game-playing agent. The agent plays an unknown grid game with no instructions. "
          "You write the game's rules as Python code; the agent checks your code against everything it has seen and "
          "plans with it, so exact code matters more than prose.")

ASK = """Write the game's rules as two Python functions:

def step(grid, action):
    # grid: list of rows, each a list of ints 0-15 (grid[y][x]); action: an int ({simple}) or a tuple (6, x, y)
    # for a click on cell (x, y). Return the grid right after the action, within the same level.

def is_goal(grid):
    # Return True if this grid means the level is complete. The agent will search for actions that reach it.

Standard library only, no input/output. Reply with exactly one ```python code block."""

RUNNER = r'''
import json, sys, resource, collections, time
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
req = json.loads(sys.stdin.read())
ns = {}
try:
    exec(compile(req["code"], "<muse>", "exec"), ns)
    step, is_goal = ns["step"], ns["is_goal"]
except Exception as e:
    print(json.dumps({"error": "compile: %s: %s" % (type(e).__name__, e)})); sys.exit()
def act(a):
    return tuple(int(v) for v in a.split()) if " " in a else int(a)
def norm(g):
    return [list(map(int, r)) for r in g]
out = {}
if req["mode"] == "verify":
    res = []
    for c in req["cases"]:
        try:
            res.append(norm(step([list(r) for r in c["grid"]], act(c["action"]))))
        except Exception as e:
            res.append("%s: %s" % (type(e).__name__, e))
    out["post"] = res
else:  # plan: BFS over the model to is_goal
    t0 = time.time()
    start = norm(req["grid"])
    key = lambda g: json.dumps(g)
    try:
        done0 = bool(is_goal(start))
    except Exception:
        done0 = False
    seen = {key(start)}
    frontier = [(start, [])]
    path = None
    nodes = 0
    while frontier and path is None and len(frontier[0][1]) < req["depth"]:
        nxt = []
        for g, p in frontier:
            for a in req["actions"]:
                try:
                    h = norm(step([list(r) for r in g], act(a)))
                    nodes += 1
                    if is_goal(h):
                        path = p + [(a, h)]
                        break
                except Exception:
                    continue
                k = key(h)
                if k not in seen:
                    seen.add(k)
                    nxt.append((h, p + [(a, h)]))
                if nodes >= req["nodes"] or time.time() - t0 > req["seconds"]:
                    break
            if path is not None or nodes >= req["nodes"] or time.time() - t0 > req["seconds"]:
                break
        frontier = nxt
    out = {"path": path, "nodes": nodes, "goal_at_start": done0}
print(json.dumps(out))
'''


def sandbox(req, timeout=40):
    with tempfile.TemporaryDirectory() as d:
        runner = os.path.join(d, "run.py")
        open(runner, "w").write(RUNNER)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", runner], input=json.dumps(req), capture_output=True,
                               text=True, timeout=timeout, cwd=d, env={})
            return json.loads(p.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": f"crash: {type(e).__name__}"}


def click_cells(g, cap=24):
    """Logical cells to try a click on: one per 4-connected object of a non-modal colour (its member nearest the
    centroid), in reading order."""
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out = []
    for c in [int(v) for v in vals if v != bg]:
        for comp in muse._components(g == c):
            my = sum(p[0] for p in comp) / len(comp)
            mx = sum(p[1] for p in comp) / len(comp)
            y, x = min(comp, key=lambda p: ((p[0] - my) ** 2 + (p[1] - mx) ** 2, p))
            out.append((x, y))
    return sorted(out, key=lambda p: (p[1], p[0]))[:cap]


class MuseAgent:
    """The closed loop above, behind the framework's `is_done` / `choose_action`."""

    def __init__(self, game_id, cfg, log_path, max_calls=16, seed=1):
        self.game_id, self.cfg, self.log_path, self.max_calls = game_id, cfg, log_path, max_calls
        self.rng = random.Random(f"{game_id}:{seed}")
        self.frames = []          # distinct frames seen (for the block grid)
        self.G = None
        self.level = 0
        self.trans = []           # this level's transitions: (grid, action, event, grid)
        self.tried = collections.defaultdict(set)
        self.code = None
        self.plan = []            # [(action, predicted grid)]
        self.prev = None          # (grid, action, predicted)
        self.calls = 0
        self.need_model = True
        self.stats = collections.Counter()

    # -- perception
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
            px = G.ox + (int(x) + G.x0) * G.k + G.k // 2
            py = G.oy + (int(y) + G.y0) * G.k + G.k // 2
            act = GameAction.from_id(6)
            act.set_data({"x": int(min(63, max(0, px))), "y": int(min(63, max(0, py)))})
            return act
        return GameAction.from_id(int(a))

    def actions(self, g, avail):
        acts = [str(a) for a in avail if a not in (0, 6)]
        if 6 in avail:
            acts += [f"6 {x} {y}" for x, y in click_cells(g)]
        return acts

    # -- the LLM
    def prompt(self, g, avail, failures):
        simple = ", ".join(str(a) for a in avail if a not in (0, 6)) or "none"
        hist = []
        for a, ev, b, act in [(t[0], t[2], t[3], t[1]) for t in self.trans[-14:]]:
            hist.append((act, ev, muse.object_diff_text(a, b)))
        parts = [muse.GRID_HELP.format(w=g.shape[1], h=g.shape[0]), muse.actions_text(avail),
                 "Current scene:\n" + muse.objects_text(g), "Current grid:\n" + muse.grid_text(g),
                 muse.history_text(hist, "objects")]
        if self.level:
            parts.append(f"The agent has completed {self.level} level(s) of this game; this is a new level.")
        if self.code and failures:
            parts.append("Your previous code:\n```python\n" + self.code + "\n```\nIt was WRONG on these recorded "
                         "transitions (action: what really happened / what your step predicted):\n" + failures +
                         "\nFix it.")
        elif self.code:
            parts.append("Your previous code (it matched every recorded transition, but planning with it did not "
                         "finish the level, so is_goal or step is still wrong):\n```python\n" + self.code + "\n```")
        parts.append(ASK.format(simple=simple))
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    def ask(self, msgs):
        self.calls += 1
        t0 = time.time()
        try:
            content, reasoning, finish, usage = ask.stream(self.cfg["base_url"].rstrip("/") + "/chat/completions",
                                                           dict(model=self.cfg["model"], messages=msgs,
                                                                max_tokens=self.cfg.get("max_tokens", 4096),
                                                                temperature=self.cfg.get("temperature", 0.6),
                                                                seed=self.calls, **self.cfg.get("extra", {})), 900)
            err = None
        except Exception as e:
            content, reasoning, finish, usage, err = "", None, None, {}, f"{type(e).__name__}: {e}"
        with open(self.log_path, "a") as f:
            f.write(json.dumps(dict(game=self.game_id, level=self.level, call=self.calls, messages=msgs,
                                    answer=content, reasoning=reasoning, finish=finish, usage=usage, error=err,
                                    latency_s=round(time.time() - t0, 2), ts=time.time())) + "\n")
        return muse.extract_code(content or "")

    def failures(self, code):
        cases = [dict(grid=a.tolist(), action=act) for a, act, ev, b in self.trans if ev == "none"]
        if not cases:
            return 0, ""
        res = sandbox(dict(mode="verify", code=code, cases=cases))
        if "error" in res:
            return len(cases), res["error"]
        bad = []
        for (a, act, ev, b), post in zip([t for t in self.trans if t[2] == "none"], res["post"]):
            if post != b.tolist():
                pred = post if isinstance(post, str) else muse.object_diff_text(a, np.array(post))
                bad.append(f"- action {act}: really {muse.object_diff_text(a, b)} / yours {pred}")
        return len(bad), "\n".join(bad[:6])

    def model(self, g, avail):
        fails = ""
        for _ in range(3):
            if self.calls >= self.max_calls:
                return
            code = self.ask(self.prompt(g, avail, fails))
            if not code or "def step" not in code:
                self.stats["no_code"] += 1
                continue
            n_bad, fails = self.failures(code)
            self.code = code
            if n_bad == 0:
                self.stats["verified"] += 1
                return
            self.stats["revise"] += 1

    def make_plan(self, g, avail):
        if not self.code:
            return
        res = sandbox(dict(mode="plan", code=self.code, grid=g.tolist(), actions=self.actions(g, avail),
                           depth=40, nodes=30000, seconds=25))
        path = res.get("path") if isinstance(res, dict) else None
        if path:
            self.plan = [(a, np.array(h)) for a, h in path]
            self.stats["plans"] += 1
        else:
            self.stats["no_plan"] += 1

    # -- the framework interface
    def is_done(self, frames, latest):
        from arcengine import GameState
        return latest.state is GameState.WIN

    def choose_action(self, frames, latest):
        from arcengine import GameAction, GameState
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or not latest.frame:
            if self.prev and latest.state is GameState.GAME_OVER:
                a, act, pred = self.prev
                self.trans.append((a, act, "death", a))
                self.prev, self.plan = None, []
            self.prev = None
            return GameAction.RESET
        g = self.grid_of(latest.frame[-1])
        avail = sorted(set(int(v) for v in (latest.available_actions or [])))
        lv = int(latest.levels_completed or 0)
        if self.prev is not None:
            a, act, pred = self.prev
            if lv > self.level:
                self.trans.append((a, act, "complete", g))
                self.stats["levels"] += 1
                self.level, self.trans, self.tried, self.plan = lv, [], collections.defaultdict(set), []
                self.frames, self.G = [], None
                g = self.grid_of(latest.frame[-1])
                self.need_model = True
            else:
                self.trans.append((a, act, "none", g))
                if pred is not None and (pred.shape != g.shape or (pred != g).any()):
                    self.stats["surprise"] += 1
                    self.plan, self.need_model = [], True
        key = hashlib.blake2b(g.tobytes(), digest_size=8).hexdigest()
        if not self.plan and self.need_model and self.calls < self.max_calls and len(self.trans) >= len(avail):
            self.need_model = False
            self.model(g, avail)
            self.make_plan(g, avail)
        if self.plan:
            a, pred = self.plan.pop(0)
            self.stats["planned_steps"] += 1
        else:
            pred = None
            acts = self.actions(g, avail)
            untried = [x for x in acts if x not in self.tried[key]]
            simple = [x for x in untried if " " not in x]
            a = (simple or untried or acts)[0] if (simple or untried) else self.rng.choice(acts)
            if not untried:
                a = self.rng.choice(acts)
                if self.code and self.calls < self.max_calls:
                    self.need_model = True
            self.stats["explore_steps"] += 1
        self.tried[key].add(a)
        self.prev = (g, a, pred)
        return self.to_action(a)


def main():
    import replay

    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", default="{}", help="JSON merged into each request (e.g. chat_template_kwargs)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-calls", type=int, default=16, help="LLM calls per game")
    ap.add_argument("--games", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max-actions", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    cfg = dict(base_url=args.base_url, model=args.model, max_tokens=args.max_tokens, extra=json.loads(args.extra))

    orig = replay.make_agent

    def make_agent(kind, game_id, seed, env, meta):
        if kind == "muse":
            return MuseAgent(game_id, cfg, args.log, max_calls=args.max_calls, seed=seed)
        return orig(kind, game_id, seed, env, meta)

    replay.make_agent = make_agent
    games = replay.discover(args.env, 40, "openmorph-gate-1")
    if args.games:
        keep = set(args.games.split(","))
        games = [g for g in games if g["base"] in keep or g["game_id"] in keep]
    opts = dict(env=args.env, max_actions=args.max_actions, offline_resets=False)
    tasks = [("muse", 1, g, opts) for g in games]
    ctx = multiprocessing.get_context("fork")
    rows = []
    with ctx.Pool(args.jobs) as pool:
        for r in pool.imap_unordered(replay._run_safe, tasks):
            rows.append(r)
            print(f"{r.get('game'):18} score {r.get('score', 0):7.2f}  levels {r.get('levels', 0)}/"
                  f"{r.get('n_levels', '?')}  actions {r.get('actions', 0)}  {r.get('error', '')}", flush=True)
    final = sum(r.get("score", 0) for r in rows) / max(1, len(rows))
    json.dump(dict(rows=rows, final=final, cfg={k: v for k, v in cfg.items()}), open(args.out, "w"), indent=1)
    print(f"FINAL muse {final:.3f} over {len(rows)} games; levels {sum(r.get('levels', 0) for r in rows)}")


if __name__ == "__main__":
    main()
