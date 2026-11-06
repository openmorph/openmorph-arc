# SPDX-License-Identifier: MIT-0
"""Rebuild the notebook's `%%writefile` cells from the files of this tree; `--check` fails if the notebook differs.

Every other cell is kept as it is. The 16 files are the agent's, each at the path the notebook writes it to.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
NB = HERE / "openmorph-on-arc-agi-3.ipynb"
FILES = {
    "my_agent.py": "kit-ru/my_agent.py",
    "ru.py": "kit-ru/ru.py",
    "rules.py": "kit-ru/rules.py",
    "threadsafe.py": "kit-ru/threadsafe.py",
    "dg13.py": "kit-dg13/dg13.py",
    "dg3.py": "kit-dg3/dg3.py",
    "dg1.py": "kit-dg1/dg1.py",
    "bs.py": "kit-bh/bs.py",
    "facts.py": "kit-hy/facts.py",
    "hy_arms.py": "kit-hy/hy_arms.py",
    "muse/muse.py": "kit-x/muse/muse.py",
    "muse/ask.py": "kit-x/muse/ask.py",
    "muse/play.py": "kit-x/muse/play.py",
    "muse/explore.py": "kit-x/muse/explore.py",
    "muse/think.py": "kit-x/muse/think.py",
    "muse/weights/eval.py": "kit-x/muse/weights/eval.py"
}


def build():
    nb = json.loads(NB.read_text())
    seen = set()
    for c in nb["cells"]:
        s = "".join(c["source"])
        if s.startswith("%%writefile "):
            path = s.split("\n", 1)[0].split()[1].replace("/kaggle/working/", "")
            body = (HERE.parent / FILES[path]).read_text()
            c["source"] = (f"%%writefile /kaggle/working/{path}\n" + body).splitlines(keepends=True)
            seen.add(path)
    assert seen == set(FILES), sorted(set(FILES) ^ seen)
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        if NB.read_text() != text:
            sys.exit("openmorph-on-arc-agi-3.ipynb does not write the files of this tree: run build_notebook.py")
        print("notebook up to date")
    else:
        NB.write_text(text)
        print("wrote", NB)
