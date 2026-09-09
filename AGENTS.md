# AGENTS.md

Conventions for this repo, distilled from the verifiers and prime-envs `AGENTS.md` files.

## Writing environments

- Base everything on verifiers v1: `import verifiers.v1 as vf`. Never mix v0 objects
  (`Environment`, `Rubric`, `Parser`, `ToolEnv`) into a taskset.
- Scaffold with `uv run init <name> -p environments/<group>`; do not hand-write the package.
- Keep tasksets small: typed data/task/config classes, `load()`, decorated scoring on the task.
  Don't override `Taskset.__init__`; implement `load()`.
- `TaskData` holds immutable, serialisable row values only. `Task` holds behaviour. Load-time
  knobs go on `TasksetConfig`, per-task run-time knobs on `TaskConfig`.
- Prefer deterministic rewards grounded in the answer or artifact. Metrics are for
  observability and never contribute to reward. Raise ordinary exceptions from hooks.
- Register each environment in the root `pyproject.toml` (`dependencies` + `[tool.uv.sources]`).
  Package dependencies go in the environment's own `pyproject.toml`, pinned `verifiers>=0.3.1`.
- Keep the environment README current: what it is, source and size, config knobs, signals,
  and a dated changelog.

## Prompting

- No role prompts. Minimal prompts: what the task is and how to answer.
- Instructions and reward must match.
- Prompt only for behaviour specific to this environment.
- Tool descriptions are prompts too: describe what the tool does, not how to use it here.

## Running code

- Always `uv run`, never raw `python`. Run from the repo root.
- Local runs use `--env.agent.runtime.type subprocess` (or `docker`) and `push = false`.
- Smoke test: `uv run eval <taskset-id> -n 3 -r 1 --no-rich -v`. Validate TOML configs with
  `--dry-run` before running them.
- Lint: `uv run ruff check --fix . && uv run ruff format .`. Tests: `uv run pytest`.

## Code style

- Minimal try/except; let errors propagate unless fault tolerance is intentional.
- Targeted comments only; no narration of the work process.
- Tests are plain functions with pytest fixtures, no test classes.

## Git

- Branch prefixes: `feat/`, `fix/`, `chore/`.
- Pull requests are drafts. No "test plan" section unless tests were actually run.
