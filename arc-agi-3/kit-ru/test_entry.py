# SPDX-License-Identifier: MIT-0
"""The agent's checks: the notebook writes the tree's files byte for byte, its settings clear every switch the code
reads, the rule model learns and keeps its rules across levels, the classes compose in order, the notebook's
pre-flight passes on the files it writes, and a click's cells and data are per game thread."""
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
KITS = HERE.parent
NB = HERE / "openmorph-on-arc-agi-3.ipynb"
_spec = importlib.util.spec_from_file_location("entry_build", HERE / "build_notebook.py")
BN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BN)
DIRS = [HERE, KITS / "kit-dg13", KITS / "kit-dg1", KITS / "kit-dg3", KITS / "kit-bh", KITS / "kit-hy", KITS / "kit-x" / "muse"]


def _cells():
    return ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"]]


def _written():
    out = {}
    for s in _cells():
        if s.startswith("%%writefile "):
            head, body = s.split("\n", 1)
            out[head.split()[1].replace("/kaggle/working/", "")] = body
    return out


def _imports():
    sys.path[:0] = [str(d) for d in DIRS]


def test_the_notebook_is_what_the_build_script_builds():
    assert NB.read_text() == BN.build()


def test_the_notebook_writes_every_file_byte_for_byte():
    w = _written()
    assert set(w) == set(BN.FILES)
    for path, rel in BN.FILES.items():
        assert w[path] == (KITS / rel).read_text(), path


def test_the_settings_cell_clears_every_switch_the_code_reads():
    settings = next(s for s in _cells() if s.startswith("# SETTINGS FOR THIS SUBMISSION"))
    popped = set(re.findall(r"'([A-Z0-9_]+)'", settings))
    read = set()
    for rel in BN.FILES.values():
        t = (KITS / rel).read_text()
        read |= set(re.findall(r'environ\.get\("((?:HY|TK|EXPLORE|BS|DG1|DG3|RU)_[A-Z0-9_]+)"', t))
        read |= set(re.findall(r'environ\.setdefault\("((?:HY|TK|EXPLORE|BS|DG1|DG3|RU)_[A-Z0-9_]+)"', t))
    assert any(k.startswith("RU_") for k in read), "no RU switch found: the pattern is stale"
    assert read <= popped, sorted(read - popped)


def test_the_run_cell_copies_only_my_agent():
    run = next(s for s in _cells() if "KAGGLE_IS_COMPETITION_RERUN" in s and "main.py" in s)
    cps = [line.strip() for line in run.splitlines() if line.strip().startswith("!cp")]
    assert len(cps) == 2 and "my_agent.py" in cps[1]


def test_the_rule_model_learns_and_carries_over():
    _imports()
    import rules
    m = rules.RuleModel()
    g = np.full((8, 8), 3)
    g[2, 2] = 5
    h = g.copy()
    h[2, 2], h[2, 3] = 3, 5
    for _ in range(3):
        m.observe(g, 4, h)
    assert m.self_colour == 5
    m.new_level()                                          # positions reset, rules kept
    g2 = np.full((8, 8), 3)
    g2[5, 1] = 5
    pred, info = m.predict(g2, 4)
    want = g2.copy()
    want[5, 1], want[5, 2] = 3, 5
    assert info["key"] is not None and (pred == want).all()


def test_the_class_order():
    pytest.importorskip("openmorph")
    _imports()
    for k in [k for k in os.environ if k.startswith(("DG1_", "DG3_", "RU_"))]:
        os.environ.pop(k)
    import ru
    names = [c.__name__ for c in ru.RuThink.__mro__]
    assert names[:5] == ["RuThink", "Dg13Think", "DgThink", "Dg3Think", "BsThink"]


def test_the_preflight_passes_on_the_written_files(tmp_path):
    pytest.importorskip("arcengine")
    pytest.importorskip("openmorph")
    for path, body in _written().items():
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    pre = next(s for s in _cells() if s.startswith("# PRE-FLIGHT")).replace("/kaggle/working", str(tmp_path))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HY_", "TK_", "EXPLORE_", "BS_", "DG1_", "DG3_", "RU_"))}
    env.update(OM_KIT_X_ROOT=str(tmp_path), OM_KIT_HV_ROOT=str(tmp_path))
    r = subprocess.run([sys.executable, "-c", pre], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert ">>> kit RU-T2 | rules + win + plan over kit DG13, click_cells and click data per thread" in r.stdout
    assert r.stdout.rstrip().endswith("kit RU-T2 pre-flight OK")


def test_click_cells_is_per_thread():
    _imports()
    import threading
    import play
    import threadsafe
    threadsafe.install(play, "click_cells")
    own = play.click_cells
    seen = {}

    def game(i):
        play.click_cells = lambda g, cap=32, i=i: i
        seen[i] = play.click_cells(None)

    ts = [threading.Thread(target=game, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen == {0: 0, 1: 1, 2: 2, 3: 3}
    assert play.click_cells is own


def test_click_data_is_per_thread():
    pytest.importorskip("arcengine")
    _imports()
    import threading
    import time
    import threadsafe
    from arcengine import GameAction
    threadsafe.install_action_data(GameAction)
    GameAction.ACTION6.set_data({"x": 1, "y": 2})
    seen = {}

    def game(i):
        a = GameAction.from_id(6)
        a.set_data({"x": i, "y": i})
        time.sleep(0.01)
        seen[i] = (a.action_data.x, a.action_data.y)

    ts = [threading.Thread(target=game, args=(i,)) for i in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen == {i: (i, i) for i in range(5)}
    assert (GameAction.ACTION6.action_data.x, GameAction.ACTION6.action_data.y) == (1, 2)
