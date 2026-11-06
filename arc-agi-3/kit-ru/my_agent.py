# SPDX-License-Identifier: MIT-0
"""Kit RU: kit DG13 with stream RU's object-level rule model (`ru.RuThink`, `rules.py`; docs/research/1.9-streams/
ru-rules.md): rules learned from every transition and kept across levels, the win condition learned at each level
completion, and a search over predicted grids that plans toward it from level 2 on. Everything below is kit DG13's.

Kit DG13: kit DG3 composed with kit DG1, fixed in code: relational novelty (`dg1.DgThink`, stream DG1;
docs/research/1.9-streams/dg1-goal.md, read 3) over the bar test (`dg3.Dg3Think`), as `dg13.Dg13Think`. Once a level has
taken 300 actions, the planner's P(new state) for an untried action at a reached state is multiplied by
1 + 1 / sqrt(n), n the decisions this level in states whose touching colour pairs are the same.

Kit DG3: kit BH with one change, fixed in code: a budget bar is not state (`dg3.Dg3Think`, stream DG3;
docs/research/1.9-streams/dg3-diagnosis.md). A colour change that comes in small isolated parts, changes no cell
twice in a life, never comes with its inverse and stays on one row or column is painted out of the state key, like
BS's ticks (a bar shortens with every action's own effect, never alone, so the tick test misses it).

Kit BH: kit HV with stream BS's effect-graph switches (`bs.py`), fixed in code: the tick test paints out what every
action changes the same way (BS_KEY, tick only), and clicks are probed by effect (BS_CLICK: every object, the soft inert
rule). `BS_FREE=1` reads the switches from the environment instead; both off = kit HV.

Kit HV: kit TK (below) with stream HY's verified facts (`facts.py`, `agent.py`; `docs/research/1.9-streams/
hy-verify.md` reads 6-7). Before each action a verifier predicts the next frame's step facts and the self, checks them
after the action, and keeps per-fact verified rates. TK's search reads, for the current state, the death rate VERIFIED
in each prediction's probability bin (arm 1v) and the trusted effect / novelty verdicts, in place of its per-type rates.
Settings, fixed in code (`make_policy`): arm 1v, and the verifier in lite mode (it checks only the facts arm 1v reads).
Everything else is kit TK's wrapper as it was.

Kit TK: kit X (below) with stream TK's planner, `muse/think.py` (`ThinkAgent`, a subclass of v3's `ExploreAgent`;
`docs/research/1.9-streams/tk-think-prereg.md`).

Kit X: the gpu-track `explore` agent, v3, as it scored 0.238 on the public 25 (commit `a446b4d1`, default switches: `EXPLORE_PROGRESS=1.0`, `EXPLORE_FRONTIER=1`, seed 1).

`muse/` holds v3's five modules byte for byte (`explore.py`, `muse.py`, `play.py`, `ask.py`, `weights/eval.py`).
This file only adapts `explore.ExploreAgent` to the competition framework's `Agent`:
* `MAX_ACTIONS = 3000` (the framework's default is 80). v3 was measured at 1,000; at 3,000 its first 1,000 actions
  are the same play, three more public-25 levels fall after action 1,000 and no game scores less (README);
* a global 8.3 h guard, like every earlier kit (v3's own `is_done` stops only on a WIN);
* on an exception in a step (none on the public 25), a logged legal move instead of a dead game thread.
The seed is v3's: `random.Random(f"{game_id}:1:explore")`, which Python seeds by SHA-512, not by `hash()`, so it
repeats across processes. Nothing here changes a decision v3 makes.
"""
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _root in (os.environ.get("OM_KIT_X_ROOT"), _HERE, os.path.join(_HERE, "..", "kit-x"), "/kaggle/working"):
    if _root and os.path.isdir(os.path.join(_root, "muse")):
        sys.path.insert(0, os.path.join(_root, "muse"))
        break

os.environ.setdefault("TK_ARBITER", "1")   # kit TK's default since its gate read (tk-think-prereg.md §8.1)
for _root in (os.environ.get("OM_KIT_HV_ROOT"), _HERE, os.path.join(_HERE, "..", "kit-hy"), "/kaggle/working"):
    if _root and os.path.isfile(os.path.join(_root, "facts.py")):
        sys.path.insert(0, _root)
        break

import explore  # noqa: E402  (v3; it puts muse/ and muse/weights on sys.path itself)
import think  # noqa: E402  (stream TK's planner over v3's own models)
import hy_arms as hy  # noqa: E402  (stream HY: the arms over think.ThinkAgent)
import facts  # noqa: E402  (stream HY: the fact verifier)
sys.path.insert(0, _HERE)
if os.path.isfile(os.path.join(_HERE, "..", "kit-bh", "bs.py")):   # the repo layout; on Kaggle bs.py sits beside
    sys.path.insert(0, os.path.join(_HERE, "..", "kit-bh"))
for _kit in ("kit-dg13", "kit-dg3", "kit-dg1"):                                  # the repo layout; on Kaggle these sit beside
    if os.path.isdir(os.path.join(_HERE, "..", _kit)):
        sys.path.insert(0, os.path.join(_HERE, "..", _kit))
import bs  # noqa: E402  (stream BS: the effect-graph switches, BS_KEY / BS_CLICK, off by default)
import dg3  # noqa: E402  (stream DG3: a budget bar is not state)
import dg13  # noqa: E402  (stream DG1: kit DG1 over kit DG3)
import ru  # noqa: E402  (stream RU: the rule model over kit DG13)
import play  # noqa: E402  (kit X's click candidates; kit BH's decision swaps its click_cells)
import threadsafe  # noqa: E402  (stream OL-W6: that swap made per game thread; one thread reads as before)
threadsafe.install(play, "click_cells")
from arcengine import FrameData, GameAction, GameState  # noqa: E402
threadsafe.install_action_data(GameAction)   # stream RU: a click's data per game thread (GameAction members are shared)

try:
    from agents.agent import Agent  # provided by the competition kernel
except Exception:
    Agent = object  # type: ignore[assignment,misc]

logger = logging.getLogger()
_RUN_START = time.time()
_BUDGET_S = float(os.environ.get("OM_BUDGET_S", str(8.3 * 3600)))
import openmorph as _om  # noqa: E402
MAX_ACTIONS = int(os.environ.get("KIT_X_MAX_ACTIONS", "3000"))
print(f">>> kit RU-T2 | rules + win + plan over kit DG13, click_cells and click data per thread | explore v3 (a446b4d1) + think + HY facts (arm 1v) + BS ticks + clicks (nogrid, soft) + DG3 bars + DG1 relational novelty | openmorph {_om.__version__} | "
      f"MAX_ACTIONS {MAX_ACTIONS}", flush=True)


def make_policy(game_id):
    """Kit BH's policy: kit HV's (kit TK's planner with HY's arm 1v and a lite verifier) with stream BS's states by the
    tick test and click probing by effect (`bs.py`: BS_KEY tick only; every object, the soft inert rule), fixed in code; `BS_FREE=1` reads the BS_* switches instead
    (both off = kit HV, row for row on the public 25)."""
    v = facts.Verifier()
    v.lite = True
    a = ru.RuThink(game_id, seed=1, verifier=v, arm="1v")
    a.ru_on, a.ru_nodes, a.ru_probe, a.ru_l1, a.ru_opt = True, 2000, False, False, True   # kit RU: plan from level 2
    a.dg3_bar = True   # kit DG3: a budget bar is not state
    a.dg_rel, a.dg_w, a.dg_after = True, 1.0, 300   # kit DG1: relational novelty after 300 actions of a level, w 1
    if not int(os.environ.get("BS_FREE", "0")):
        # kit BH's setting, fixed in code: states by the tick test (BS_KEY, tick only) and click probing by effect
        # (every object, the soft inert rule)
        a.bs_key, a.bs_key_mode = True, "tick"
        a.bs_click, a.bs_click_mode, a.bs_click_inert = True, "nogrid", "soft"
    return _RuVerified(a, v, game_id=game_id)


class _RuVerified(facts.Verified):
    """HY's observer wrapper, with the rule layer's report in the gate's row."""

    def stats(self):
        r = super().stats()
        try:
            r["ru"] = self.policy.ru_report()
        except Exception as e:  # pragma: no cover
            r["ru"] = dict(error=repr(e))
        return r


class MorphAgent(Agent):
    """The `explore` agent v3 with stream TK's planner and stream HY's verified facts, in the official framework."""

    MAX_ACTIONS = MAX_ACTIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = make_policy(self.game_id)
        self.errors = 0
        logger.info(f"MorphAgent kit HV: HyThink 1v over explore v3 (game={self.game_id})")

    def is_done(self, frames, latest_frame: FrameData) -> bool:
        return self.policy.is_done(frames, latest_frame) or (time.time() - _RUN_START) > _BUDGET_S

    def choose_action(self, frames, latest_frame: FrameData) -> GameAction:
        try:
            return self.policy.choose_action(frames, latest_frame)
        except Exception as e:  # pragma: no cover
            self.errors += 1
            logger.warning(f"kit HV choose_action error #{self.errors} (game={self.game_id}), legal fallback: {e!r}")
            avail = sorted(int(getattr(a, "value", a)) for a in (getattr(latest_frame, "available_actions", None) or []))
            simple = [a for a in avail if a not in (0, 6)]
            if simple:
                return GameAction.from_id(simple[0])
            if 6 in avail:
                act = GameAction.from_id(6)
                act.set_data({"x": 32, "y": 32})
                return act
            return GameAction.RESET
