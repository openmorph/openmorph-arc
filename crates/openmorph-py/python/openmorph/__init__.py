# SPDX-License-Identifier: MIT-0
"""openmorph — digital organisms that learn, grow, and compose, from Python.

The Rust type ``Field`` is exposed in Python as ``Morph`` — same organism, one object.

OpenMorph is **not** a neural network. There is no backprop, no gradient tape, no optimizer. An organism
is an async membrane of cells that *fires on change, settles to a fixed point, learns locally, and grows
new structure when it is surprised*. Yet the Python interface feels like a familiar ML library:

    import numpy as np
    import openmorph as om

    # A grown, backprop-free function approximator — learns y = f(x) by GROWING capacity.
    head = om.ValueHead(in_dim=1, hidden=4)
    for _ in range(2000):
        x = np.random.uniform(-1, 1, size=1)
        head.learn(x, np.tanh(1.5 * x[0]))
    head.predict(np.array([1.0]))            # ≈ tanh(1.5)

    # A value-based RL actor with an intrinsic curiosity drive — online, no replay buffer.
    actor = om.Actor(n_features=8, n_actions=4)
    a = actor.act(state)
    actor.learn(state, a, reward, next_state, done)

Three layers, low → high:
    om.Agent       the perception path: a colour frame in, a move out (act / observe / snapshot).
    om.Morph       the raw organism (Rust core): stimulate / observe / grow / learn_step / snapshot.
    om.ValueHead   a grown scalar function approximator: predict / learn / fit.
    om.Actor       a value-based RL actor (grown Q-heads + intrinsic novelty).
"""

from ._core import (
    Morph,
    segment_objects,
    aggregate,
    relate,
    __version__,
)

try:
    from ._core import (
        Agent,
        RelativeCode,
        RecurrentActor,
        SpatialActor,
        Player,
        ImaginationPlanner,
        FrontierExplorer,
        RelationalGoal,
        RuleInduction,
        StructuralGoal,
        ProgressReadout,
        DisplayedTargetMatcher,
        Codebook,
        Crystallizer,
        LivingLoop,
        PredictiveLatent,
        StructuralKey,
        SkillMetric,
        GrownSegmenter,
        ConstraintField,
        WorldModelEnsemble,
        FovealWorldModel,
        GoalBand,
        greedy_effect_action,
        skill_signature,
        effect_graph,
        attend,
        attend_regions,
        attend_changes,
        controllable_regions,
        role_effect_graph,
        category_roles,
        role_of_effects,
        is_static_role,
    )

    HAS_PERCEPTION = True
except ImportError:  # this build: MIT-0 sources only, no perception
    HAS_PERCEPTION = False
from . import nn, planning, presets, rl
from .nn import ValueHead
from .planning import Explorer, WorldModel, infer_goal, plan_step
from .presets import apply_preset, describe, list_presets, load_preset, preset_json
from .rl import Actor

__all__ = [
    "Morph",
    "segment_objects",
    "aggregate",
    "relate",
    "list_presets",
    "load_preset",
    "describe",
    "preset_json",
    "apply_preset",
    "presets",
    "ValueHead",
    "Actor",
    "WorldModel",
    "plan_step",
    "Explorer",
    "infer_goal",
    "nn",
    "rl",
    "planning",
    "__version__",
]
