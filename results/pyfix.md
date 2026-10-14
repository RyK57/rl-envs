# pyfix

Source: [environments/code/pyfix/README.md](../environments/code/pyfix/README.md). Not re-run.

`deepseek/deepseek-v4-flash`, `bash` harness, docker runtime (`python:3.11-slim`), temperature 1.0, 13 tasks × 2 rollouts, $0.024.

| Reward | file_changed | syntax_ok | Model calls per rollout |
| --- | --- | --- | --- |
| 1.000 | 1.000 | 1.000 | 4 to 8 |

26 of 26 solved. The README calls the set saturated for this model: the planted bugs are one-liners.

Agent judge (`configs/pyfix_judged.toml`, `shared-agentic-judge`), first run, 3 tasks: the judge agreed with the hidden tests 3 of 3, all solved. Whether it says no when the fix is missing was not tested.
