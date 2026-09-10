# pyfix

Fix one planted bug in a small Python function. The model gets a shell and an editor inside a
container; the reward runs hidden tests against its final file, inside that container, after the
run. The gold never enters the box during the run.

The new mechanism in this environment is the runtime. `setup()` writes the buggy file into the
box, `finalize()` reads the result back into the trace (`info.solution`), and the reward writes
and runs the hidden tests there. `validate()` proves inside a runtime that the reference fix
passes the tests and the buggy source fails them, so the tests discriminate before any model runs.

## Taskset

- **Source:** `pyfix/catalog.py`, thirteen functions with one bug each and hidden tests that go
  beyond the docstring's examples
- **Size:** 13 tasks, keys `pyfix:<name>`

## Config

No knobs. Runtime and harness are run settings: the smoke config uses the `bash` harness in the
docker runtime (default image `python:3.11-slim`). Without Docker, `--env.agent.runtime.type
subprocess` runs the model's commands on your own machine in a temporary directory.

## Signals

- **Reward** `tests_pass` (weight 1.0): the hidden test script exits 0 against the final file.
- **Metric** `file_changed`: the final file differs from the buggy one.
- **Metric** `syntax_ok`: the final file compiles.

## Baseline

`deepseek/deepseek-v4-flash`, `bash` harness, docker runtime (`python:3.11-slim`), temperature 1.0,
13 tasks x 2 rollouts, $0.024.

| Reward | file_changed | syntax_ok | Model calls per rollout | What happened |
| --- | --- | --- | --- | --- |
| 1.000 | 1.000 | 1.000 | 4 to 8 | every bug fixed; turns wasted guessing the working directory (`/testbed`, `/workspace`) before `pwd` |

Saturated for this model: the planted bugs are one-liners. The mechanics are what this
environment proves (files staged in the box, hidden tests run there, the final file captured on
the trace). A harder catalog, or a small efficiency signal on model calls, is the lever if it is
ever used for training.

## Run

```bash
uv run validate pyfix --runtime.type subprocess     # every row: fix passes, bug fails
uv run eval @ configs/pyfix.toml --no-rich          # 3x1 smoke test in docker, needs a model key
```

## Changelog

- 2026-09-10: Baseline recorded in docker: 26/26.
- 2026-09-10: Initial v1 taskset.
