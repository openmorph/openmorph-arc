# SPDX-License-Identifier: MIT-0
"""The muse claim-quality test: prompts from recorded stuck states, engine ground truth, scoring.

Pipeline (no network, no GPU):

    python muse.py build data/arcengine.jsonl data/bench.jsonl --out data/      # -> prompts.jsonl, gt.jsonl.gz
    python muse.py score data/ answers/*.jsonl                                   # -> the table per model config
    python muse.py selftest data/                                                # oracle and null answers

Inputs are episode records (`record_arcengine.py`, `muse_export.rs`): the 64x64 frames of a level, the actions and
events, and every candidate action PROBED from every state (event + next frame). Everything below is derived from
those transitions, generically, with no hand labels:

* GRID. The frame is reduced to its logical grid: the largest block size k (with offsets) whose k x k blocks are
  uniform in >= 90 % of cases over all the record's frames, each block its modal colour; constant border rows and
  columns of the letterbox colour (the frame's corner pixel) are cropped. This is the description a model sees; a click (x, y) is a logical cell.
* SELF. A colour "translates" in a transition when the objects of it that changed moved rigidly (other objects of
  that colour unchanged). The self is the colour that translates in the most non-terminal simple-action
  transitions of the level (>= 2); the self group adds colours that translate
  at least half as often and always with the self's shift.
* ACTION RULES (simple actions). Over the probes where the self translated: the dominant direction (sign of dx, dy),
  if it holds in >= 80 % of them (else `other`).
  An action that never moves the self is `none` when it never changes the grid, else `other` (not scored).
* GOAL. Candidate `reach(c)`: a move action completes the level iff the self, shifted one cell by the action's
  direction, overlaps a cell of colour c. A candidate is ground truth iff it predicts EVERY probe's completion event
  of the level exactly (F1 = 1, at least one completion). Levels no candidate explains have no claim-form goal truth
  (their goal is scored in the code form only).

Two answer forms (the same state, the same known facts):

* `claims`: JSON {self, actions, goal}. A fact the prompt already gives as verified is not scored (not novel);
  every withheld fact is scored correct / wrong / abstained. `novel_correct` = correct / withheld.
* `code`: Python `step(grid, action)` and `is_goal(grid)`, executed in a sandboxed subprocess over every probe of
  the level: step accuracy on the transitions shown in the prompt (what the morph can verify) and on the held-out
  probes, and the goal as `is_goal(step(grid, a))` against the engine's completion events.
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile

import numpy as np

SIZE = 64


def opener(path, mode="rt"):
    """open() that reads and writes `.gz` transparently."""
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)
STATE_TIMES = (0, 10, 20, 39)
HISTORY = 6
MAX_DIFF_CELLS = 12

# --------------------------------------------------------------------------------------------------------- grids


def decode(s):
    a = np.array([int(ch, 16) for ch in s], dtype=np.int64)
    return a.reshape(SIZE, SIZE)


class Grid:
    """Frame -> logical grid for one record: block size k, offsets, crop."""

    def __init__(self, frames):
        # the largest block size whose cropped grid is at least 4 x 4 (a near-uniform frame can pass the uniformity
        # test at a large k and then crop to nothing)
        kmax = 8
        while True:
            self.k, self.ox, self.oy = self._block(frames, kmax)
            grids = [self._reduce(f) for f in frames]
            self.y0, self.y1, self.x0, self.x1 = self._crop(grids, int(frames[0][0, 0]))
            if (self.y1 - self.y0 >= 4 and self.x1 - self.x0 >= 4) or self.k == 1:
                break
            kmax = self.k - 1

    @staticmethod
    def _block(frames, kmax=8):
        for k in [k for k in (8, 7, 6, 5, 4, 3, 2) if k <= kmax]:
            best = None
            for oy in range(k):
                for ox in range(k):
                    uni = tot = 0
                    for f in frames:
                        sub = f[oy:oy + ((SIZE - oy) // k) * k, ox:ox + ((SIZE - ox) // k) * k]
                        h, w = sub.shape[0] // k, sub.shape[1] // k
                        if h == 0 or w == 0:
                            continue
                        b = sub.reshape(h, k, w, k).transpose(0, 2, 1, 3).reshape(h, w, k * k)
                        uni += int((b == b[:, :, :1]).all(axis=2).sum())
                        tot += h * w
                    if tot and uni / tot >= 0.9 and (best is None or uni / tot > best[0]):
                        best = (uni / tot, ox, oy)
            if best:
                return k, best[1], best[2]
        return 1, 0, 0

    def _reduce(self, f):
        k, ox, oy = self.k, self.ox, self.oy
        h, w = (SIZE - oy) // k, (SIZE - ox) // k
        sub = f[oy:oy + h * k, ox:ox + w * k]
        b = sub.reshape(h, k, w, k).transpose(0, 2, 1, 3).reshape(h, w, k * k)
        out = np.zeros((h, w), dtype=np.int64)
        for y in range(h):
            for x in range(w):
                out[y, x] = np.bincount(b[y, x], minlength=16).argmax()
        return out

    @staticmethod
    def _crop(grids, letterbox):
        """Drop constant border rows/columns of the letterbox colour (the frame's corner pixel); a border of any
        other colour (a maze's outer wall) is part of the game."""
        g0 = grids[0]
        h, w = g0.shape

        def const_row(y):
            v = g0[y, 0]
            return all((g[y, :] == v).all() and g[y, 0] == v for g in grids)

        def const_col(x):
            v = g0[0, x]
            return all((g[:, x] == v).all() and g[0, x] == v for g in grids)

        corner = letterbox
        y0, y1, x0, x1 = 0, h, 0, w
        while y0 < y1 - 1 and const_row(y0) and g0[y0, 0] == corner:
            y0 += 1
        while y1 - 1 > y0 and const_row(y1 - 1) and g0[y1 - 1, 0] == corner:
            y1 -= 1
        while x0 < x1 - 1 and const_col(x0) and g0[0, x0] == corner:
            x0 += 1
        while x1 - 1 > x0 and const_col(x1 - 1) and g0[0, x1 - 1] == corner:
            x1 -= 1
        return y0, y1, x0, x1

    def grid(self, f):
        return self._reduce(f)[self.y0:self.y1, self.x0:self.x1]

    def cell_of_pixel(self, px, py):
        return (px - self.ox) // self.k - self.x0, (py - self.oy) // self.k - self.y0


def grid_text(g):
    return "\n".join("".join("%x" % v for v in row) for row in g)


def diff_text(a, b):
    if a.shape != b.shape:
        return "the whole grid changed (new size %dx%d)" % (b.shape[1], b.shape[0])
    ys, xs = np.nonzero(a != b)
    if len(ys) == 0:
        return "no change"
    parts = ["(%d,%d) %x->%x" % (x, y, a[y, x], b[y, x]) for y, x in zip(ys.tolist(), xs.tolist())]
    more = len(parts) - MAX_DIFF_CELLS
    return ", ".join(parts[:MAX_DIFF_CELLS]) + (f", ... {more} more cells" if more > 0 else "")


# --------------------------------------------------------------------------------------------------- truth


def _components(mask):
    """4-connected components of a boolean mask, as lists of (y, x)."""
    seen = np.zeros(mask.shape, bool)
    comps = []
    for y0, x0 in np.argwhere(mask).tolist():
        if seen[y0, x0]:
            continue
        stack, comp = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        comps.append(comp)
    return comps


def translation(a, b, c):
    """The (dx, dy) by which the objects of colour c that changed moved rigidly from a to b (the rest of colour c
    unchanged), else None. Other objects of the same colour may stay put."""
    if a.shape != b.shape:
        return None
    A, B = a == c, b == c
    gone, new = A & ~B, B & ~A
    if not gone.any() or not new.any():
        return None
    moved = [comp for comp in _components(A) if any(gone[y, x] for y, x in comp)]
    cells = [p for comp in moved for p in comp]
    gy, gx = np.argwhere(gone)[0]
    h, w = a.shape
    for ny, nx in np.argwhere(new).tolist():
        for py, px in cells:
            if (py, px) != (gy, gx):
                continue
            dy, dx = ny - py, nx - px
            if dy == 0 and dx == 0:
                continue
            exp = A.copy()
            for y, x in cells:
                exp[y, x] = False
            ok = True
            for y, x in cells:
                yy, xx = y + dy, x + dx
                if not (0 <= yy < h and 0 <= xx < w):
                    ok = False
                    break
                exp[yy, xx] = True
            if ok and (exp == B).all():
                return int(dx), int(dy)
    return None


def sign(v):
    return (v > 0) - (v < 0)


def action_token(tok, G):
    """Engine token -> (kind, logical): '3' -> (3, '3'); '6@px,py' -> (6, '6 x y')."""
    if tok.startswith("6@"):
        px, py = map(int, tok[2:].split(","))
        x, y = G.cell_of_pixel(px, py)
        return 6, f"6 {x} {y}", (x, y)
    return int(tok), tok, None


def level_truth(recs):
    """Ground truth of one (game, seed, level) from its episodes (walk + demo)."""
    frames = [decode(s) for r in recs for s in r["pool"]]
    G = Grid(frames)
    trans = []  # (pre grid, token, event, post grid)
    for r in recs:
        gs = [G.grid(decode(s)) for s in r["pool"]]
        for t, probes in enumerate(r["probes"]):
            pre = gs[r["frames"][t]]
            for tok, (ev, fi) in probes.items():
                trans.append((pre, tok, ev, gs[fi]))
    colours = sorted({int(v) for g in (G.grid(f) for f in frames) for v in np.unique(g)})
    tcount, tshift = collections.Counter(), collections.defaultdict(list)
    for pre, tok, ev, post in trans:
        if ev != "none" or tok == "0" or tok.startswith("6@"):
            continue  # the self is what the simple actions move
        for c in colours:
            d = translation(pre, post, c)
            if d:
                tcount[c] += 1
                tshift[c].append((id(pre), tok, d))
    truth = dict(block=G.k, dims=None, self=None, self_group=[], rules={}, goal=None, goal_candidates=[])
    if tcount and tcount.most_common(1)[0][1] >= 2:
        me = tcount.most_common(1)[0][0]
        truth["self"] = me
        mine = {(p, t): d for p, t, d in tshift[me]}
        group = [me]
        for c, n in tcount.items():
            if c != me and n >= tcount[me] / 2:
                if all(mine.get((p, t)) == d for p, t, d in tshift[c]):
                    group.append(c)
        truth["self_group"] = sorted(group)
        # action rules
        per = collections.defaultdict(collections.Counter)
        changed = collections.Counter()
        seen_act = set()
        for pre, tok, ev, post in trans:
            if tok.startswith("6@") or tok == "0":
                continue
            a = int(tok)
            seen_act.add(a)
            if ev == "none":
                d = translation(pre, post, me)
                if d:
                    per[a][(sign(d[0]), sign(d[1]))] += 1
                elif pre.shape != post.shape or (pre != post).any():
                    changed[a] += 1
        for a in sorted(seen_act):
            if per[a]:
                (dx, dy), k = per[a].most_common(1)[0]
                if k >= 0.8 * sum(per[a].values()):
                    truth["rules"][str(a)] = {"moves": me, "dx": dx, "dy": dy}
                else:  # no consistent direction (e.g. a jump to a fixed place): not a scorable rule
                    truth["rules"][str(a)] = {"effect": "other"}
            elif changed[a]:
                truth["rules"][str(a)] = {"effect": "other"}
            else:
                truth["rules"][str(a)] = {"effect": "none"}
        # goal: reach(c)
        group_set = set(group)
        for c in colours:
            if c in group_set:
                continue
            tp = fp = fn = 0
            for pre, tok, ev, post in trans:
                if ev == "death" or tok == "0":
                    continue
                pred = False
                rule = truth["rules"].get(tok)
                if rule and "moves" in rule:
                    cells = np.argwhere(np.isin(pre, list(group_set)))
                    for y, x in cells.tolist():
                        yy, xx = y + rule["dy"], x + rule["dx"]
                        if 0 <= yy < pre.shape[0] and 0 <= xx < pre.shape[1] and pre[yy, xx] == c:
                            pred = True
                            break
                real = ev == "complete"
                tp += pred and real
                fp += pred and not real
                fn += real and not pred
            if tp and not fp and not fn:
                truth["goal_candidates"].append(c)
        if truth["goal_candidates"]:
            truth["goal"] = {"kind": "reach", "targets": truth["goal_candidates"]}
    truth["completions"] = sum(ev == "complete" for _, _, ev, _ in trans)
    return G, trans, truth


# -------------------------------------------------------------------------------------------------- prompts

SYSTEM = (
    "You are the muse of a game-playing agent. The agent plays an unknown grid game with no instructions. "
    "It records every transition, and it tests every claim you make against them before acting on it, so a "
    "wrong claim costs little but a vague one is useless. Be concrete."
)

GRID_HELP = (
    "The grid is {w} wide and {h} high. Each cell is a colour 0-15 written as one hex digit; row 0 is the top, "
    "x grows to the right, y grows downward."
)


def actions_text(avail):
    names = []
    for a in avail:
        if a == 6:
            names.append("ACTION6 x y (click the cell at x, y)")
        elif 1 <= a <= 7:
            names.append(f"ACTION{a}")
    return "Available actions: " + ", ".join(names) + ". RESET restarts the level."


def objects_text(g):
    """The grid as objects: the most common colour, then every 4-connected one-colour component, grouped when a
    colour has several objects of the same shape. Generic: no colour or shape is named."""
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    h, w = g.shape
    lines = [f"Most common colour: {bg} ({int(counts.max())} of {h * w} cells), not listed below. Objects:"]
    for c in [int(v) for v in vals if v != bg]:
        comps = _components(g == c)
        shapes = collections.defaultdict(list)
        for comp in comps:
            ys, xs = [p[0] for p in comp], [p[1] for p in comp]
            x0, y0 = min(xs), min(ys)
            shapes[tuple(sorted((y - y0, x - x0) for y, x in comp))].append((x0, y0, max(xs), max(ys), len(comp)))
        for shape, objs in sorted(shapes.items(), key=lambda kv: (-len(kv[1]), kv[1][0])):
            n = objs[0][4]
            bw, bh = objs[0][2] - objs[0][0] + 1, objs[0][3] - objs[0][1] + 1
            if n == 1:
                desc = "1 cell"
            elif n == bw * bh:
                desc = f"a solid {bw}x{bh} block"
            elif n <= 8:
                desc = f"{n} cells, shape " + " ".join(f"({x},{y})" for y, x in shape) + " from its top-left"
            else:
                desc = f"{n} cells spanning {bw}x{bh} (a large irregular shape: walls, a frame or a region)"
            where = ", ".join(f"({x0},{y0})" for x0, y0, _, _, _ in objs[:12]) + (" ..." if len(objs) > 12 else "")
            if len(objs) == 1:
                lines.append(f"- colour {c}: one object, {desc}, top-left at {where}")
            else:
                lines.append(f"- colour {c}: {len(objs)} objects, each {desc}, top-left at {where}")
    return "\n".join(lines)


def object_diff_text(a, b):
    """What changed between two grids, per colour: a rigid move of its changed objects, or cells (dis)appearing.
    The most common colour's changes (the ground filling in behind a moving object) are reported only when nothing
    else changed."""
    if a.shape != b.shape:
        return "the whole grid changed (new size %dx%d)" % (b.shape[1], b.shape[0])
    if (a == b).all():
        return "no change"
    vals, counts = np.unique(a, return_counts=True)
    common = int(vals[np.argmax(counts)])
    changed = sorted({int(v) for v in np.unique(np.concatenate([a[a != b], b[a != b]]))})
    if len(changed) > 1:
        changed = [c for c in changed if c != common]
    parts = []
    for c in changed:
        d = translation(a, b, c)
        if d:
            parts.append(f"colour {c} moved by dx={d[0]}, dy={d[1]}")
            continue
        gone, new = int(((a == c) & (b != c)).sum()), int(((b == c) & (a != c)).sum())
        bits = []
        if new:
            bits.append(f"{new} cell{'s' if new > 1 else ''} appeared")
        if gone:
            bits.append(f"{gone} cell{'s' if gone > 1 else ''} disappeared")
        parts.append(f"colour {c}: " + " and ".join(bits))
    return "; ".join(parts)


def history_text(hist, style="grid"):
    if not hist:
        return "Recent transitions: none yet (this is the level's first frame)."
    lines = ["Recent transitions, oldest first (cell changes as (x,y) old->new):" if style == "grid"
             else "Recent transitions, oldest first (what each colour's objects did):"]
    for i, (act, ev, d) in enumerate(hist, 1):
        if act == "0":
            name = "RESET"
        elif act == "outside":
            name = "ACTION6 outside the grid"
        else:
            name = "ACTION" + act if " " not in act else "ACTION6 " + act[2:]
        tail = {"complete": " -> the level was COMPLETED", "death": " -> the agent DIED (game over)"}.get(ev, "")
        lines.append(f"{i}. {name}: {d}{tail}")
    return "\n".join(lines)


def known_text(known, truth):
    facts = []
    if "self" in known:
        facts.append(f"the agent's own piece (the self) is colour {truth['self']:x}")
    for a in sorted(k for k in known if k != "self"):
        r = truth["rules"][a]
        if "moves" in r:
            facts.append(f"ACTION{a} moves the self by dx={r['dx']}, dy={r['dy']} (when not blocked)")
        else:
            facts.append(f"ACTION{a} does nothing")
    if not facts:
        return "Already verified: nothing yet."
    return "Already verified (do not repeat these): " + "; ".join(facts) + "."


CLAIMS_ASK = """The agent has not completed this level and has no idea what to do. Reply with ONE JSON object and nothing else:
{{"self": <the colour of the piece the agent controls, as an integer 0-15>,
 "actions": {{"<action number>": {{"moves": <colour>, "dx": <-1|0|1>, "dy": <-1|0|1>}} or {{"effect": "none"}} or {{"effect": "other", "note": "<short>"}}}},
 "goal": {{"kind": "reach", "target": <colour the self must touch to complete the level>}} or {{"kind": "other", "note": "<short>"}}}}
Give "actions" for every simple action ({simple}) that is not already verified. Colours are integers (hex b = 11)."""

CODE_ASK = """The agent has not completed this level and has no idea what to do. Write the game's rules as two Python functions:

def step(grid, action):
    # grid: list of rows, each a list of ints 0-15 (grid[y][x]); action: an int ({simple}) or a tuple (6, x, y) for a click.
    # Return the grid right after the action, within the same level.

def is_goal(grid):
    # Return True if this grid means the level is complete.

Standard library only, no input/output. Reply with exactly one ```python code block."""


def prompt_messages(form, G, grid, avail, hist, known, truth, style="grid"):
    """`grid` style: the hex grid and raw cell diffs. `objects` style: the objects and what each colour did (the
    code form keeps the grid too, since its functions take one)."""
    simple = ", ".join(str(a) for a in avail if a != 6) or "none"
    h, w = grid.shape
    if style == "grid":
        scene = "Current grid:\n" + grid_text(grid)
    elif form == "claims":
        scene = "Current scene:\n" + objects_text(grid)
    else:
        scene = "Current scene:\n" + objects_text(grid) + "\n\nThe same, as the grid:\n" + grid_text(grid)
    body = "\n\n".join([
        GRID_HELP.format(w=w, h=h),
        actions_text(avail),
        scene,
        history_text(hist, style),
        known_text(known, truth),
        (CLAIMS_ASK if form == "claims" else CODE_ASK).format(simple=simple),
    ])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def avail_list(rec, t):
    if "available" in rec:
        return list(rec["available"][t])
    m = rec["available_mask"][t]
    return [i for i in range(1, 8) if m & (1 << i)]


def build(paths, out_dir, style="grid"):
    recs = [json.loads(l) for p in paths for l in opener(p) if l.strip()]
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["source"], r["set"], r["game"], r["seed"], r["level"])].append(r)
    os.makedirs(out_dir, exist_ok=True)
    np_ = gt_ = 0
    with open(os.path.join(out_dir, "prompts.jsonl"), "w") as fp, \
            opener(os.path.join(out_dir, "gt.jsonl.gz"), "wt") as fg, \
            open(os.path.join(out_dir, "levels.jsonl"), "w") as fl:
        for key, rs in sorted(groups.items()):
            src, set_, game, seed, level = key
            G, trans, truth = level_truth(rs)
            lid = f"{src}:{game}:s{seed}:L{level}"
            fl.write(json.dumps(dict(level_id=lid, source=src, set=set_, game=game, seed=seed, level=level,
                                     block=G.k, truth=truth)) + "\n")
            walk = next((r for r in rs if r["kind"] == "walk"), None)
            if walk is None:
                continue
            gs = [G.grid(decode(s)) for s in walk["pool"]]
            # every probe of the level, as logical cases, for the code form
            cases = []
            for r in rs:
                rg = [G.grid(decode(s)) for s in r["pool"]]
                for t, probes in enumerate(r["probes"]):
                    for tok, (ev, fi) in probes.items():
                        _, logical, xy = action_token(tok, G)
                        g0 = rg[r["frames"][t]]
                        if xy and not (0 <= xy[0] < g0.shape[1] and 0 <= xy[1] < g0.shape[0]):
                            continue  # a click on the letterbox
                        cases.append(dict(ep=r["kind"], t=t, grid=rg[r["frames"][t]].tolist(), action=logical,
                                          event=ev, post=rg[fi].tolist()))
            T = len(walk["actions"])
            for si, t in enumerate(sorted({min(x, T - 1) for x in STATE_TIMES})):
                if not walk["probes"][t]:
                    continue  # game over: only RESET
                grid = gs[walk["frames"][t]]
                hist, shown = [], []
                for u in range(max(0, t - HISTORY), t):
                    tok = walk["actions"][u]
                    a, b = gs[walk["frames"][u]], gs[walk["frames"][u + 1]]
                    _, logical, xy = action_token(tok, G) if tok != "0" else (0, "0", None)
                    if xy and not (0 <= xy[0] < a.shape[1] and 0 <= xy[1] < a.shape[0]):
                        logical = "outside"
                    hist.append((logical, walk["events"][u], diff_text(a, b) if style == "grid" else object_diff_text(a, b)))
                    shown.append(dict(t=u, action=logical))
                avail = avail_list(walk, t)
                rng = random.Random(f"{lid}:{t}")
                variants = [("none", set())]
                if truth["self"] is not None and si % 2 == 1:  # every other state also asks with some facts known
                    scored = [a for a in truth["rules"] if truth["rules"][a].get("effect") != "other"]
                    half = set(rng.sample(scored, len(scored) // 2)) if scored else set()
                    variants.append(("some", {"self"} | half))
                for vname, known in variants:
                    for form in ("claims", "code"):
                        pid = f"{lid}:t{t}:{vname}:{form}" + ("" if style == "grid" else f":{style}")
                        msgs = prompt_messages(form, G, grid, avail, hist, known, truth, style)
                        fp.write(json.dumps(dict(prompt_id=pid, level_id=lid, form=form, known=vname,
                                                 messages=msgs)) + "\n")
                        fg.write(json.dumps(dict(prompt_id=pid, level_id=lid, form=form, t=t,
                                                 known=sorted(known), truth=truth, shown=shown,
                                                 cases=cases if form == "code" else None)) + "\n")
                        np_ += 1
            gt_ += 1
    print(f"{gt_} levels, {np_} prompts -> {out_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------------- scoring


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = [m.group(1)] if m else []
    i = text.find("{")
    if i >= 0:
        depth = 0
        for j in range(i, len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                cand.append(text[i:j + 1])
                break
    for c in cand:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def as_colour(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return int(s, 16) if len(s) == 1 else int(s)
        except ValueError:
            return None
    return None


def score_claims(ans, g):
    truth, known = g["truth"], set(g["known"])
    out = collections.Counter()
    obj = parse_json(ans)
    out["parsed"] = obj is not None and isinstance(obj, dict)
    obj = obj if isinstance(obj, dict) else {}
    # (c) self
    if truth["self"] is not None and "self" not in known:
        out["self_n"] += 1
        c = as_colour(obj.get("self"))
        if c is None:
            out["self_abstain"] += 1
        elif c in truth["self_group"]:
            out["self_ok"] += 1
        else:
            out["self_wrong"] += 1
    # (a) actions
    acts = obj.get("actions") if isinstance(obj.get("actions"), dict) else {}
    acts = {str(k).replace("ACTION", ""): v for k, v in acts.items()}
    for a, r in truth["rules"].items():
        if r.get("effect") == "other":
            continue
        if a in known:
            if a in acts:
                out["act_repeat"] += 1
            continue
        out["act_n"] += 1
        claim = acts.get(a)
        if not isinstance(claim, dict) or claim.get("effect") == "other":
            out["act_abstain"] += 1
            continue
        if "moves" in r:
            ok = (as_colour(claim.get("moves")) in truth["self_group"]
                  and sign(int(claim.get("dx", 0) or 0)) == r["dx"] and sign(int(claim.get("dy", 0) or 0)) == r["dy"])
        else:
            ok = claim.get("effect") == "none"
        out["act_ok" if ok else "act_wrong"] += 1
    # (b) goal
    if truth["goal"]:
        out["goal_n"] += 1
        goal = obj.get("goal") if isinstance(obj.get("goal"), dict) else {}
        if goal.get("kind") != "reach":
            out["goal_abstain"] += 1
        elif as_colour(goal.get("target")) in truth["goal"]["targets"]:
            out["goal_ok"] += 1
        else:
            out["goal_wrong"] += 1
    return out


RUNNER = r'''
import json, sys, resource
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
src = sys.stdin.readline()
cases = json.loads(sys.stdin.readline())
out = []
try:
    ns = {}
    exec(compile(json.loads(src), "<muse>", "exec"), ns)
    step, is_goal = ns.get("step"), ns.get("is_goal")
except Exception as e:
    print(json.dumps({"error": "compile: %s" % e})); sys.exit()
for c in cases:
    a = c["action"]
    act = tuple(int(v) for v in a.split()) if " " in a else int(a)
    r = {}
    try:
        g = [list(row) for row in c["grid"]]
        post = step(g, act)
        r["post"] = [list(map(int, row)) for row in post]
    except Exception as e:
        r["err"] = type(e).__name__
    try:
        r["goal"] = bool(is_goal(r["post"])) if "post" in r else None
    except Exception as e:
        r["goal"] = None
    out.append(r)
print(json.dumps({"results": out}))
'''


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else (text if "def step" in text else None)
    return code


def run_code(code, cases, timeout=20):
    with tempfile.TemporaryDirectory() as d:
        runner = os.path.join(d, "run.py")
        open(runner, "w").write(RUNNER)
        payload = json.dumps(code) + "\n" + json.dumps([dict(grid=c["grid"], action=c["action"]) for c in cases]) + "\n"
        try:
            p = subprocess.run([sys.executable, "-I", "-S", runner], input=payload, capture_output=True, text=True,
                               timeout=timeout, cwd=d, env={})
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return {"error": "crash: " + p.stderr[-200:]}


def score_code(ans, g):
    out = collections.Counter()
    code = extract_code(ans)
    out["parsed"] = bool(code and "def step" in code)
    if not out["parsed"]:
        return out
    cases = g["cases"]
    res = run_code(code, cases)
    if "error" in res:
        out["exec_error"] = 1
        return out
    out["ran"] = 1
    shown = {(s["t"], s["action"]) for s in g["shown"]}
    for c, r in zip(cases, res["results"]):
        if c["event"] == "death":
            continue
        is_shown = c["ep"] == "walk" and (c["t"], c["action"]) in shown
        if c["event"] == "none":
            ok = r.get("post") == c["post"]
            key = "seen" if is_shown else "held"
            out[f"step_{key}_n"] += 1
            out[f"step_{key}_ok"] += ok
            if not is_shown and c["post"] != c["grid"]:
                out["step_moved_n"] += 1
                out["step_moved_ok"] += ok
        real = c["event"] == "complete"
        pred = bool(r.get("goal"))
        out["goal_tp"] += pred and real
        out["goal_fp"] += pred and not real
        out["goal_fn"] += real and not pred
    out["verified"] = int(out["step_seen_n"] == out["step_seen_ok"])
    return out


def load_jsonl(path):
    return [json.loads(l) for l in opener(path) if l.strip()]


def score(data_dir, answer_paths):
    gt = {g["prompt_id"]: g for g in load_jsonl(os.path.join(data_dir, "gt.jsonl.gz"))}
    tallies = collections.defaultdict(collections.Counter)
    n = collections.Counter()
    rows = []
    for p in answer_paths:
        for a in load_jsonl(p):
            g = gt.get(a["prompt_id"])
            if g is None or a.get("error"):
                continue
            cfg = a["config"]
            s = score_claims(a["answer"], g) if g["form"] == "claims" else score_code(a["answer"], g)
            tallies[(cfg, g["form"])].update(s)
            n[(cfg, g["form"])] += 1
            n[(cfg, "tokens")] += a.get("usage", {}).get("completion_tokens", 0) or 0
            n[(cfg, "latency")] += a.get("latency_s", 0) or 0
            rows.append(dict(prompt_id=a["prompt_id"], config=cfg, form=g["form"], **{k: int(v) for k, v in s.items()}))
    cfgs = sorted({c for c, _ in tallies})
    pct = lambda a, b: f"{100 * a / b:5.1f}" if b else "    -"
    print("CLAIMS FORM (novel = withheld facts only; ok / wrong / abstain, % of withheld)")
    print(f"{'config':34} {'n':>4} {'parse':>6} | {'self ok':>7} {'wrong':>6} | {'act ok':>6} {'wrong':>6} | {'goal ok':>7} {'wrong':>6} | {'tok/ans':>7} {'s/ans':>6}")
    for c in cfgs:
        t, k = tallies[(c, "claims")], n[(c, "claims")]
        if not k:
            continue
        tot = n[(c, "claims")] + n[(c, "code")]
        print(f"{c:34} {k:4d} {pct(t['parsed'], k)} | {pct(t['self_ok'], t['self_n'])}   {pct(t['self_wrong'], t['self_n'])} |"
              f" {pct(t['act_ok'], t['act_n'])} {pct(t['act_wrong'], t['act_n'])} | {pct(t['goal_ok'], t['goal_n'])}  "
              f" {pct(t['goal_wrong'], t['goal_n'])} | {n[(c, 'tokens')] / tot:7.0f} {n[(c, 'latency')] / tot:6.1f}")
    print("\nCODE FORM (step = exact next grid on held-out probes, all / only those that change the grid; "
          "goal = is_goal(step(grid, a)) vs the engine's completions)")
    print(f"{'config':34} {'n':>4} {'parse':>6} {'ran':>6} | {'seen ok':>7} {'held ok':>7} {'changed':>7} {'verified':>8} | {'goal P':>6} {'goal R':>6}")
    for c in cfgs:
        t, k = tallies[(c, "code")], n[(c, "code")]
        if not k:
            continue
        print(f"{c:34} {k:4d} {pct(t['parsed'], k)} {pct(t['ran'], k)} | {pct(t['step_seen_ok'], t['step_seen_n'])}  "
              f" {pct(t['step_held_ok'], t['step_held_n'])}   {pct(t['step_moved_ok'], t['step_moved_n'])}    {pct(t['verified'], k)} |"
              f" {pct(t['goal_tp'], t['goal_tp'] + t['goal_fp'])} {pct(t['goal_tp'], t['goal_tp'] + t['goal_fn'])}")
    return rows


# ------------------------------------------------------------------------------------------------- selftest


def oracle_claims(g):
    t = g["truth"]
    return json.dumps({"self": t["self"], "actions": {a: r for a, r in t["rules"].items()},
                       "goal": {"kind": "reach", "target": t["goal"]["targets"][0]} if t["goal"] else {"kind": "other"}})


def selftest(data_dir):
    """Oracle claims must score 100 % where truth exists; a null answer 0 %; a trivial step (identity) shows the
    floor of the code form (unchanged grids are common)."""
    gts = load_jsonl(os.path.join(data_dir, "gt.jsonl.gz"))
    tmp = os.path.join(data_dir, "selftest_answers.jsonl")
    ident = "```python\ndef step(grid, action):\n    return grid\n\ndef is_goal(grid):\n    return False\n```"
    with open(tmp, "w") as f:
        for g in gts:
            if g["form"] == "claims":
                f.write(json.dumps(dict(prompt_id=g["prompt_id"], config="oracle", answer=oracle_claims(g))) + "\n")
                f.write(json.dumps(dict(prompt_id=g["prompt_id"], config="null", answer="I don't know.")) + "\n")
            else:
                f.write(json.dumps(dict(prompt_id=g["prompt_id"], config="identity", answer=ident)) + "\n")
    score(data_dir, [tmp])
    levels = load_jsonl(os.path.join(data_dir, "levels.jsonl"))
    print("\nLEVEL TRUTH")
    for l in levels:
        t = l["truth"]
        print(f"{l['level_id']:42} block {l['block']}  self {t['self']} group {t['self_group']}  "
              f"rules {' '.join(f'{a}:' + (('%+d%+d' % (r['dx'], r['dy'])) if 'moves' in r else r['effect']) for a, r in t['rules'].items())}"
              f"  goal {t['goal']['targets'] if t['goal'] else None}  completions {t['completions']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("records", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--style", default="grid", choices=["grid", "objects"])
    s = sub.add_parser("score")
    s.add_argument("data")
    s.add_argument("answers", nargs="+")
    s.add_argument("--rows", help="write per-answer scores as JSONL")
    t = sub.add_parser("selftest")
    t.add_argument("data")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.records, args.out, args.style)
    elif args.cmd == "score":
        rows = score(args.data, sorted({p for a in args.answers for p in glob.glob(a)}))
        if args.rows:
            with open(args.rows, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    else:
        selftest(args.data)


if __name__ == "__main__":
    main()
