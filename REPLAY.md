<!-- SPDX-License-Identifier: MIT-0 -->
# The gate replay

The real-engine gate plays the agent on offline ARC-AGI-3-style games through the real `arc_agi` scorecard (the competition's rules: one play per game, its resets, 1000 actions). Six sets: the four ARCEngine example games, the toolkit's two test games, and 22 games with per-life step budgets. Two seeds (192, 193; the agent seeds itself, so they are the same play).

A row is one game and one seed. Two runs are equal when every field of every row is equal, except three wall-clock timings (`ms_step`, `ms_env`, `stats.ru.ms`). The digest of a row is the first 16 hex digits of the SHA-256 of `json.dumps(row, sort_keys=True)` without those three fields.

| wheel | levels | mean of set means | rows equal to the submitted wheel |
|---|---|---|---|
| the submitted wheel (`45bfde8b…`, it links perception code the agent never calls) | 110 | 25.2733 | 58 / 58 |
| **this repository's wheel**, built by `scripts/build_wheel.sh` from this tree | 110 | 25.2733 | **58 / 58** |

| set | levels | mean | row digests |
|---|---|---|---|
| `env_examples` | 16 | 5.4339 | `complex_maze-63b63b26b211 s192` cb8c6743d35f38f1, `complex_maze-63b63b26b211 s193` d992a613a7f300ee, `merge-3e1f05ad2468 s192` bad09ae46680910c, `merge-3e1f05ad2468 s193` 5783288a5c79bff4, `merge_detach-c928a7c37c2b s192` 2cb413f9ad34e5d6, `merge_detach-c928a7c37c2b s193` 0d3bcf3499929da2, `simple_maze-24e94ec6c7d5 s192` 5cb4570c2bcfbd7f, `simple_maze-24e94ec6c7d5 s193` 878ad6f9efe2e69c |
| `env_tests` | 20 | 68.2823 | `bt11-fd9df0622a1a s192` b8e82d42e83d7629, `bt11-fd9df0622a1a s193` b4d305098cb9316b, `bt33-a7c3f9d18b4e s192` 12a6e28f337a250d, `bt33-a7c3f9d18b4e s193` ac83d6588c1552f0 |
| `env_budget` | 22 | 36.9529 | `cinder-65eeaf10da78 s192` b2f0e7809c8a47e8, `cinder-65eeaf10da78 s193` 18b16ff60000c68e, `dusk-9e9b32f8c290 s192` db8310bb07febfa6, `dusk-9e9b32f8c290 s193` b864372f069fe328, `hush-237f96f4df2c s192` af4d00b0ec9dadc4, `hush-237f96f4df2c s193` 5a33bd17a4b6d338, `tally-3b34011f6666 s192` 3f7c7ffad6204393, `tally-3b34011f6666 s193` f74753b9aea5e967, `wick-26522aae029a s192` f930da949ad35d73, `wick-26522aae029a s193` 3bd7cdaeb3831ff9 |
| `env_budget2` | 18 | 15.2019 | `fray-4549c7353932 s192` 13961edcf1b438ab, `fray-4549c7353932 s193` b08e8254a6b345f6, `haul-edf3a8754076 s192` 1855102f41ffb23f, `haul-edf3a8754076 s193` 4b8b1a8ccf9783b8, `latch-c2e2c5449fc5 s192` c14754cd8b340005, `latch-c2e2c5449fc5 s193` c77abc2299a546a9, `marsh-e91e20431d83 s192` c961fe006f219cb8, `marsh-e91e20431d83 s193` 15d7fca31bf873f7, `mire-eadd1effbe55 s192` dbe68074b5ebddf1, `mire-eadd1effbe55 s193` 10586023683360da, `shove-c87ade62c7db s192` 0b5a7daf7e2cb16c, `shove-c87ade62c7db s193` 8427f91a7a00133d |
| `env_budget3` | 26 | 25.7578 | `ferry-abe1e20fdd31 s192` 988a030852213f5e, `ferry-abe1e20fdd31 s193` c670520cc34dc88c, `lumen-5fe403b74bb6 s192` 4458e0f710383fa6, `lumen-5fe403b74bb6 s193` e7bf8e22c2d01166, `pallor-c420064bf8bc s192` 4bf3fdc4d229bbe5, `pallor-c420064bf8bc s193` 7c218da881ce46e5, `tick-37990397d571 s192` 46726617ac6763f8, `tick-37990397d571 s193` be1da9e4771a518e, `tumbler-618554b9f601 s192` 46ec51b0278db2d9, `tumbler-618554b9f601 s193` 8a9acf4a2152cf86, `twain-3ff06c74c176 s192` c7c6cf3dd9fb902c, `twain-3ff06c74c176 s193` e1626a0a4b59834e |
| `env_budget4` | 8 | 0.011 | `gemini-cd506091e635 s192` f7103d4b93a0aee8, `gemini-cd506091e635 s193` 21587fe4676a8ffc, `mimic-47e5c65f93ab s192` 5c9375ff82d37140, `mimic-47e5c65f93ab s193` c503337a247806de, `postern-baa739e8d2be s192` 115e2fddd96b958c, `postern-baa739e8d2be s193` 0e9639aeb40bc6d3, `sleet-ddddec160fb1 s192` 0fde734952b540ed, `sleet-ddddec160fb1 s193` 0c2266841bc0fdf3, `strobe-19d06aeebc73 s192` 79e589db046687cf, `strobe-19d06aeebc73 s193` 508b5724cb444139, `treadle-38ccec35fdfe s192` d10a31b2154a8d96, `treadle-38ccec35fdfe s193` ec8924d159c06cd1 |

Versions: {"openmorph": "1.8.0", "arc-agi": "0.9.9", "arcengine": "0.9.3", "my_agent.py sha256": "3b7c7630506eec674c73a1df9b6405af828c9704d002938add3797a0775480d9"}. The harness (`replay.py`) and the 22 budget games are not in this repository; the four example games and the two test games are the ARCEngine and ARC-AGI toolkit's own (MIT).
