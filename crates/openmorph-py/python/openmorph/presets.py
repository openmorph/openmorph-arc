# SPDX-License-Identifier: MIT-0
"""Config PRESETS for openmorph — ready-made, use-case-tuned `Config` overlays.

Each preset is a small YAML file under ``openmorph/presets/`` listing only the levers it changes; everything else
keeps its historical default (``apply_preset`` merges the overlay onto ``Config::default()``, so a preset is a
whole starting point and ``default`` resets). Layer your own overrides on top AFTERWARDS with ``Morph.configure``,
which merges onto the current config. The YAML files are the canonical, human-editable source — add or edit one
and it is picked up automatically.

Quick use::

    import openmorph
    from openmorph import Morph, RecurrentActor

    openmorph.list_presets()                     # ['arc_bc', 'curious_scientist', 'default', ...]
    print(openmorph.describe("arc_bc"))          # the preset's one-paragraph doc
    cfg = openmorph.load_preset("arc_bc")        # -> dict of {lever: value}

    m = Morph(64)
    openmorph.apply_preset(m, "arc_bc")          # == m.configure(json.dumps(cfg), reset=True)
    m.configure('{"feature_lr": 0.05}')          # an override on top keeps the rest of arc_bc

    ra = RecurrentActor(1, [1,2,3,4,5,6], [4.0,16.0], [0.26,0.5], 0.35, True, 4, 8, 12, 0.5, 0.1, 0)
    openmorph.apply_preset(ra, "arc_bc")         # reaches the actor's value heads + belief predictor

The presets set only VALID `Config` keys (see ``docs/core/config-reference.md`` for every lever, its range, and whether
it is byte-identical off). Presets are advisory starting points — tune against your real benchmark.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from ._core import Morph

_PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")


def _coerce(token: str) -> Any:
    """Coerce a scalar YAML token to bool / int / float / str."""
    t = token.strip()
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    # strip surrounding quotes if present (e.g. activation: "Relu")
    if len(t) >= 2 and t[0] in "'\"" and t[-1] == t[0]:
        return t[1:-1]
    return t


def _parse(path: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a flat preset YAML into (description, {lever: value}). Description = the leading comment block."""
    desc_lines: List[str] = []
    cfg: Dict[str, Any] = {}
    seen_config = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if not seen_config:
                    desc_lines.append(stripped.lstrip("#").strip())
                continue
            if ":" not in stripped:
                continue
            seen_config = True
            key, _, value = stripped.partition(":")
            key = key.strip()
            # drop any trailing inline comment on the value
            value = value.split(" #", 1)[0].strip()
            if key:
                cfg[key] = _coerce(value)
    return " ".join(desc_lines), cfg


def _path(name: str) -> str:
    p = os.path.join(_PRESET_DIR, name + ".yaml")
    if not os.path.isfile(p):
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(list_presets())}"
        )
    return p


def list_presets() -> List[str]:
    """The names of every available preset (the ``.yaml`` files under ``openmorph/presets/``), sorted."""
    if not os.path.isdir(_PRESET_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(_PRESET_DIR) if f.endswith(".yaml")
    )


def load_preset(name: str) -> Dict[str, Any]:
    """The `Config` overlay of preset ``name`` as a ``{lever: value}`` dict (merge it onto defaults)."""
    return _parse(_path(name))[1]


def describe(name: str) -> str:
    """The one-paragraph description of preset ``name`` (its leading comment)."""
    return _parse(_path(name))[0]


def preset_json(name: str) -> str:
    """The preset as a JSON string ready for ``Morph.configure`` / ``actor.configure``."""
    return json.dumps(load_preset(name))


def apply_preset(target: Any, name: str) -> Dict[str, Any]:
    """Apply preset ``name`` to any object with a ``configure(json_str)`` method — a ``Morph`` or a
    ``RecurrentActor``. The target's config becomes ``Config::default()`` + the overlay (earlier ``configure``
    calls are replaced, so the empty ``default`` preset resets). Returns the applied overlay."""
    cfg = load_preset(name)
    if isinstance(target, Morph):
        # Morph.configure merges onto the CURRENT config; reset=True keeps a preset = defaults + overlay.
        target.configure(json.dumps(cfg), reset=True)
    else:
        target.configure(json.dumps(cfg))  # the actors' configure already starts from the defaults
    return cfg


__all__ = [
    "list_presets",
    "load_preset",
    "describe",
    "preset_json",
    "apply_preset",
]
