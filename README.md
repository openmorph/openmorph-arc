# openmorph-arc

The ARC-AGI-3 entry of [OpenMorph](https://openmorph.org): kit RU-T2, an agent for the ARC Prize 2026 ARC-AGI-3
competition, and the part of the OpenMorph library it runs on. Everything here is **MIT-0**.

| path | what |
|---|---|
| `arc-agi-3/kit-ru/openmorph-on-arc-agi-3.ipynb` | the Kaggle notebook: it installs the wheel, writes the agent's 16 files and runs them |
| `arc-agi-3/kit-*/` | those 16 files (`kit-ru/my_agent.py` is the entry point) |
| `arc-agi-3/kit-ru/build_notebook.py` | rebuilds the notebook's `%%writefile` cells from the files; `--check` fails on a difference |
| `arc-agi-3/kit-ru/test_entry.py` | the agent's tests |
| `crates/openmorph-core` | the organism, `openmorph-core` 1.8.0 (Rust, `no_std`, zero ML dependencies) |
| `crates/openmorph-py` | the Python binding: `Morph`, `Law`, the core-only functions and the Python package |
| `scripts/build_wheel.sh` | builds the `openmorph` 1.8.0 wheel the agent installs |
| `REPLAY.md` | the gate replay: this tree's wheel plays exactly like the submitted one |

## What the agent uses

The agent (`my_agent.py`) is written in Python over NumPy. It reads frames, keeps a graph of the states it has
seen, learns which actions change what, verifies yes/no facts about the game and plans over them. The learned
readouts are OpenMorph organisms (`openmorph.Morph`, its recursive-least-squares readout), so `import openmorph` is
the one native dependency. The submitted wheel also contained OpenMorph's perception classes, which the agent never
calls. They are not part of this repository, and the wheel built here leaves them out.

## Build the wheel

```bash
pip install maturin ziglang
scripts/build_wheel.sh            # -> dist/openmorph-1.8.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
pip install dist/openmorph-*.whl
python -c "import openmorph; print(openmorph.__version__, openmorph.HAS_PERCEPTION)"   # 1.8.0 False
```

Rust 1.87 or later. Without zig, `cd crates/openmorph-py && maturin build --release` builds for this machine's glibc.

## Run the agent

- **On Kaggle:** attach the wheel as a dataset to the notebook and set `OM_WHEEL_VERSION = '1.8.0'` and
  `OM_WHEEL_SHA256` to the wheel's `sha256sum` (the notebook installs only the wheel with that hash). In the
  competition's scored rerun the notebook plays every hidden game through the competition's gateway.
- **Elsewhere:** with the competition's `ARC-AGI-3-Agents` framework, put `arc-agi-3/kit-ru/my_agent.py` in its
  `agents/templates/`, set `OM_KIT_X_ROOT` and `OM_KIT_HV_ROOT` to a directory holding the other 15 files as the
  notebook writes them (or run it from this tree, where it finds them itself), and run
  `python main.py --agent morphagent`. It needs `arc-agi`, `arcengine`, NumPy and SciPy.

## Test

```bash
pip install pytest numpy scipy arc-agi==0.9.9 arcengine==0.9.3 dist/openmorph-*.whl
pytest arc-agi-3/kit-ru/test_entry.py
python arc-agi-3/kit-ru/build_notebook.py --check
cargo test -p openmorph-core     # the organism's own proofs
```

## Company and contact

OpenMorph is built by Morphe.ai (Morphic Intelligence Limited, New Zealand). Company, investors and contact:
https://morphe.ai

## Licence

MIT-0 (MIT No Attribution), `LICENSE-MIT0`. Copyright (c) 2026 Petitgen Limited. Patent pending (NZ provisional,
Petitgen Limited).

OpenMorph is developed and commercialised by Morphic Intelligence Limited (Morphe.ai) under licence from Petitgen
Limited.

Third-party code is not vendored. The wheel links Rust crates under MIT, Apache-2.0, Unlicense and Unicode-3.0
terms (its SBOM, `openmorph-1.8.0.dist-info/sboms/`); at run time the agent uses NumPy and SciPy (BSD-3-Clause), and
`arc-agi`, `arcengine` and the `ARC-AGI-3-Agents` framework (MIT) from the competition.
