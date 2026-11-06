# SPDX-License-Identifier: MIT-0
"""(Taken byte for byte in logic from kit OL's threadsafe.py, stream OL-W6.)
Per-thread module attributes, for the one global that kit RU's decision swaps: `play.click_cells`.

Kit BH's `bs.BsThink._think` (inherited by kit RU) sets `play.click_cells` to the game's own click generator for the
length of one decision and restores it after. The competition framework plays every game as a thread of one
process, so two games' decisions can overlap. One game can then read another's generator. And when the restores
interleave (A sets, B sets, A restores, B restores A's), the module keeps a dead game's generator for every later
decision. Kit OL makes the window longer (an LLM call can sit inside a decision), so it makes the attribute per
thread: an assignment is seen only by the thread that made it; a thread that set nothing reads the module's own
function. Run in one thread, every read returns what it returned before (the gate rows are unchanged), and the
module's own globals are untouched.

`install(play, "click_cells")` is idempotent. `install_action_data(GameAction)` (stream RU) does the same for the
click data arcengine's `GameAction` members carry (below).
"""
import threading
import types


class _PerThread(types.ModuleType):
    def __getattribute__(self, name):
        d = types.ModuleType.__getattribute__(self, "__dict__")
        if name in d.get("_ol_names", ()):
            tl = d["_ol_tl"].__dict__
            if name in tl:
                return tl[name]
        return types.ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        d = types.ModuleType.__getattribute__(self, "__dict__")
        if name in d.get("_ol_names", ()):
            setattr(d["_ol_tl"], name, value)
            return
        types.ModuleType.__setattr__(self, name, value)


def install(module, *names):
    """Make `names` per-thread attributes of `module` (reads fall back to the module's own value)."""
    d = module.__dict__
    if "_ol_tl" not in d:
        d["_ol_tl"] = threading.local()
        d["_ol_names"] = frozenset()
    d["_ol_names"] = d["_ol_names"] | frozenset(names)
    if not isinstance(module, _PerThread):
        module.__class__ = _PerThread
    return module


def install_action_data(game_action_cls):
    """Make `action_data` per thread on the members of arcengine's `GameAction` enum (stream RU, 2026-10-28).

    `GameAction.ACTION6.set_data({...})` writes the click's coordinates onto the enum member, one object shared by
    every game of the process. With games as threads, one game's `set_data` can land between another's and that
    game's read of its own action (the verifier's read, the framework's submission). A thread now reads the data it
    set itself; a thread that set nothing reads the member's own default. Idempotent; one thread reads as before."""
    if getattr(game_action_cls, "_ru_tl", None) is not None:
        return game_action_cls
    tl = threading.local()

    def _get(self):
        d = tl.__dict__.get("data")
        if d is not None and self.name in d:
            return d[self.name]
        return self.__dict__["action_data"]

    def _set(self, value):
        d = tl.__dict__.setdefault("data", {})
        d[self.name] = value

    type.__setattr__(game_action_cls, "_ru_tl", tl)
    type.__setattr__(game_action_cls, "action_data", property(_get, _set))
    return game_action_cls
