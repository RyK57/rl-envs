# rl-envs

A practice catalog of RL environments for LLMs, built on
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 and laid out like Prime
Intellect's production catalog, [prime-envs](https://github.com/PrimeIntellect-ai/prime-envs).
Each environment adds one concept on top of the previous one; the goal is to end up able to
write production-grade environments, not to collect toys.

## The landscape

Three reference repos sit next to this one. They play different roles:

| Repo | What it is | Use it for |
| --- | --- | --- |
| `verifiers` (`prime-int-verifiers`) | The framework: `Taskset`/`Task`/`TaskData`, harnesses, runtimes, the interception server, the `init`/`eval`/`validate` CLIs. Its `environments/` are small teaching examples. | The spec. Read `docs/v1/` (human docs) and `skills/` (the agent-facing, more complete version) before writing code. |
| `prime-envs` | ~90 production tasksets on verifiers: one package per env under `environments/<group>/<name>`, README with changelog, pinned dataset revisions, CI that installs and runs every env. | The shape to imitate. When unsure how something "should" look, find the closest env there. |
| `HuggingEnvs` | The ecosystem tour: the same environment implemented in six frameworks (OpenEnv, ORS, NeMo Gym, verifiers, SkyRL Gym, GEM) plus training recipes. Its verifiers ports use the older in-process toolkit style, not v1. | Understanding what is framework-specific versus inherent, and how an env plugs into a trainer. |

## Mental model

An environment in verifiers v1 is a **taskset**: a Python package exporting one `vf.Taskset`
subclass. Everything else is provided by the framework and chosen at run time.

| Piece | Owns | Where |
| --- | --- | --- |
| `TaskData` | One immutable row: prompt, references, image, network policy. It is what gets stored on the trace. | your package |
| `Task` | Behaviour for a row: `@vf.stop`, `@vf.reward`, `@vf.metric`, the `setup`/`finalize`/`validate` hooks, optional toolsets. | your package |
| `Taskset` | Loading: `load()` yields tasks. `TasksetConfig` holds load-time knobs (size, seed, split); `TaskConfig` holds per-task run-time knobs (judge model, thresholds). | your package |
| Harness | The program the model runs in: `bash` (default, a bash + edit agent), `null` (plain chat, no tools), `claude_code`, `codex`, `mini_swe_agent`, and more. The taskset does not define tools; harnesses bring them. | framework, `--env.agent.harness.id` |
| Runtime | Where the harness process runs: `subprocess` (local, for debugging), `docker`, `prime` (remote sandbox, the default), `modal`. | framework, `--env.agent.runtime.type` |
| Interception server | Every model call the harness makes goes through it. That is how traces are captured, sampling is controlled, and reward hacks can be blocked. | framework |
| `Trace` | The message graph plus rewards, metrics, errors and per-call records for one agent run. `traces.jsonl` holds one episode per line. | output |
| `Env` | Control flow between agents. `SingleAgentEnv` by default; `best-of-n`, `agentic-judge`, `user-sim` are bundled; a custom `run()` scripts a user or a game engine. | framework or your package |

One rollout, in order: load a task → provision the runtime → `task.setup` → the harness runs
the model (calls flow through interception, `@vf.stop` predicates end it) → `task.finalize`
→ metrics and rewards → the trace is written.

Design rules that matter for training, distilled from the prime-envs `AGENTS.md`:

- No role prompts. Keep prompts minimal: what the task is and how to answer, not how to solve it.
- Instructions and reward must agree; any mismatch trains the model to ignore instructions.
- Only prompt for behaviour specific to this environment; behaviour that should generalise is
  rewarded, not prompted.
- Prefer deterministic verification of an artifact or answer; reach for an LLM judge only when
  the judgement is semantic.
- Almost no environment needs custom tools. A harness already provides a shell, an editor, and
  often web search.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv, the only tool you need
uv sync                                            # verifiers + every environment in the catalog
cp .env.example .env                               # then fill in a key (see below)
```

Model calls need a key. `PRIME_API_KEY` covers Prime inference (the default client) and Prime
sandboxes (the default runtime). Any OpenAI-compatible endpoint works instead:
`--client.base-url <url> --client.api-key-var OPENAI_API_KEY`.

## Workflow

The inner loop for one environment, from scaffold to a scored run:

```bash
uv run init my-env -p environments/<group>              # 1. scaffold (never hand-write the package)
#   implement load() and the @vf.reward in environments/<group>/my_env/my_env/taskset.py
#   register it: add "my-env" to the root pyproject dependencies + [tool.uv.sources]
uv sync                                                 # 2. install it, editable
uv run ruff check --fix . && uv run ruff format .       # 3. lint
uv run pytest                                           # 4. offline scoring tests (no model)
uv run validate my-env --runtime.type subprocess        # 5. model-free gold check per row
uv run eval @ configs/my_env.toml --dry-run             # 6. resolve the run config, no calls
uv run --env-file .env eval @ configs/my_env.toml --no-rich -v   # 7. 3x1 smoke test
```

Every run writes to `outputs/<run-dir>/`: the resolved config, `traces.jsonl`, and logs.
Read a few traces before trusting a number. `scripts/summarize.py` prints the outcome counts,
mean reward, metric means, cost, and one line per rollout with the gold answer next to the
model's reply:

```bash
uv run python scripts/summarize.py outputs/<run-dir>
```

Scale only after loading, harness, runtime and scoring are all correct: more rollouts (`-r`),
more tasks (drop `-n`), other harnesses and runtimes, then publish (`prime env push`) or train
against it with prime-rl.

## Environments

| Taskset | Group | Task | Concepts it teaches |
| --- | --- | --- | --- |
| [`count-letters`](environments/toy/count_letters/) | toy | Count how often a letter appears in a short passage; answer in `<answer>` tags. | Typed `TaskData`/`Task`/`Taskset`, procedural seeded rows, config knobs, tagged-answer parsing, binary reward plus metrics, `validate`, offline scoring tests. |

## Roadmap

Each step adds exactly one new concept. The pattern to copy is named for each.

1. **Anatomy** (done): `count-letters`. Scaffold, load, score, validate, test offline.
2. **First real rollout** (done): add a key, run the smoke config, read `traces.jsonl`. Then run the
   same taskset under the `bash` harness and see whether the model reaches for a shell; that
   is the taskset-versus-harness split in action. Results are in the `count-letters` README:
   0.63 without tools with mixed rewards per task, 0.93 with a shell.
3. **Multi-turn with a scripted user**: an `Env.run()` that drives `interaction.turn()`, for
   example a number-guessing game with per-turn feedback. Pattern:
   `verifiers/environments/alphabet_sort`.
4. **Dataset-backed rows**: a Hugging Face dataset with a pinned revision and a lazy `load()`
   generator. Pattern: `prime-envs/environments/math/math500` and `reasoning/unscramble`.
5. **Verification inside the sandbox**: the docker runtime, a reward that runs a script with
   `runtime.run_uv_script`, `setup()` pre-provisioning, `network_allow=[]`. Pattern:
   `verifiers/environments/gsm8k`.
6. **An LLM judge**: `vf.Judge` with its config on `TaskConfig` so the judge model is a CLI
   knob. Pattern: `docs/v1/tasksets.md`, "Using Judges".
7. **Multi-agent envs**: `--env.id best-of-n` and `agentic-judge`, then a custom `Env` with
   `finalize()`. Pattern: `verifiers/environments/code_golf`.
8. **Production hygiene**: a CI smoke test like `prime-envs/tests/test_envs.py`, a changelog
   in each README, `prime env push`, a training run with prime-rl.

## Gotchas

- The verifiers repo itself needs `uv >= 0.11.1`; older `uv` fails to parse its `pyproject.toml`.
- Defaults are production defaults: runtime `prime`, client Prime inference, `push = true`
  (uploads the run to the platform). Local configs set `runtime.type = "subprocess"` or
  `"docker"` and `push = false`.
- `--no-rich` gives plain logs, `-v` prints prompts and completions. Both belong in a smoke test.
- Run from the repo root, never from inside `.venv`: NLTK refuses imports resolved beneath the
  working directory.
- `TasksetConfig.system_prompt` is reserved (a file path override). Use another name for an
  inline default prompt.
- Hosted Claude Code sessions route outbound HTTPS through an egress policy fixed when the
  environment was created. `ProviderError: 403 Forbidden` on every model call, with
  `connect_rejected` for `api.pinference.ai:443` in the proxy status, means the host is blocked,
  not that the key is wrong. Allow the host in the environment's network settings and start a
  new session, or run the eval on your own machine.
- A `@vf.stop` on `trace.num_turns` counts model calls, and a tool call is a model call. A "single turn" stop ends a `bash`-harness episode on the first tool call, before any answer. Let the harness decide when the agent is done and bound runaways with `--env.agent.max-turns`.
