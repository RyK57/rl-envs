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

A judge-scored run can be re-scored offline with `uv run replay <run-dir> -r 3 --rich false`
(the same judge three times, its noise) or with `--taskset.id <id> --taskset.task.judge.model
<other>` (another judge). `scripts/judges.py <replay-dir> <replay-dir>` prints the summaries
with both judges' verdicts side by side, so a human can decide which judge to trust.

Scale only after loading, harness, runtime and scoring are all correct: more rollouts (`-r`),
more tasks (drop `-n`), other harnesses and runtimes, then publish (`prime env push`) or train
against it with prime-rl.

## Dashboard

`dashboard/` is a live observer over `outputs/`: every run with its rewards, metrics, cost and
errors; each run's task × rollout grid, rollout table and `eval.log`; each trace's full
conversation with reasoning, tool calls, model calls and timing. It re-reads the folder every
five seconds, so a run in progress fills in as it goes.

```bash
cd dashboard && npm install && npm run dev      # http://localhost:3000
```

## Environments

| Taskset | Group | Task | Concepts it teaches |
| --- | --- | --- | --- |
| [`count-letters`](environments/toy/count_letters/) | toy | Count how often a letter appears in a short passage; answer in `<answer>` tags. | Typed `TaskData`/`Task`/`Taskset`, procedural seeded rows, config knobs, tagged-answer parsing, binary reward plus metrics, `validate`, offline scoring tests. |
| [`number-guess`](environments/toy/number_guess/) | toy | Find a secret number from higher/lower feedback; guesses in `<guess>` tags. | An exported `Env` whose `run()` drives the conversation turn by turn, a binary episode reward, metrics over the whole exchange, a model-free winnability check. |
| [`gsm8k`](environments/math/gsm8k/) | math | Grade-school word problems from Hugging Face; final answer in `\boxed{}`. | A dataset at a pinned revision, a lazy `load()` generator, stable task keys from dataset coordinates, math-equivalence scoring with the same checker prime-envs uses. |
| [`pyfix`](environments/code/pyfix/) | code | Fix one planted bug in a small Python function with a shell and editor; hidden tests run in the sandbox. | `setup()`/`finalize()` hooks that write and read files in the runtime, a reward that runs commands in the box, the docker runtime, a `validate` that proves the tests discriminate. |
| [`summarize`](environments/text/summarize/) | text | Two-sentence summaries of short passages, graded by an LLM judge for faithfulness and coverage. | A `vf.Judge` subclass with a prompt template and a strict parser, judge config as a run-time knob, one cached judge call shared by reward and metrics, a deterministic gate before the judge. |
| [`guess-golf`](environments/toy/guess_golf/) | toy | An env, not a taskset: four games of one number-guess task per episode, the fewest guesses wins. | A multi-attempt `vf.Env` with a named role (`guesser`), `run()` fanning out games with a task group, `finalize()` recording an episode-level reward that compares siblings. |

## Roadmap

Each step adds exactly one new concept. The pattern to copy is named for each.

1. **Anatomy** (done): `count-letters`. Scaffold, load, score, validate, test offline.
2. **First real rollout** (done): add a key, run the smoke config, read `traces.jsonl`. Then run the
   same taskset under the `bash` harness and see whether the model reaches for a shell; that
   is the taskset-versus-harness split in action. Results are in the `count-letters` README:
   0.63 without tools with mixed rewards per task, 0.93 with a shell.
3. **Multi-turn with a scripted user** (done): `number-guess`, an `Env.run()`
   that drives `interaction.turn()` with higher/lower feedback. Saturated at 1..100, a
   0.90 baseline with mixed groups at 1..5000 / 13. Pattern: `verifiers/environments/alphabet_sort`.
4. **Dataset-backed rows** (done): `gsm8k`, a Hugging Face dataset with a
   pinned revision and a lazy `load()` generator. Pattern: `prime-envs/environments/math/math500`
   and `reasoning/unscramble`.
5. **Verification inside the sandbox** (done): `pyfix`, the docker runtime,
   `setup()`/`finalize()` file hooks, a reward that runs hidden tests in the box. Pattern:
   `verifiers/environments/gsm8k` and `docs/v1/env.md`.
6. **An LLM judge** (done): `summarize`, a `vf.Judge` with its config on `TaskConfig` so the
   judge model is a CLI knob, then judge calibration: `replay -r 3` for the noise floor (7 of 26
   rollouts move by one key point), a second judge via `replay --taskset.task.judge.model`, and a
   human read of every disagreement with `scripts/judges.py`, which found the second judge wrong
   on all of them and led to a clearer rubric. Pattern: `docs/v1/tasksets.md`, "Using Judges".
7. **Multi-agent envs** (done): `--env.id best-of-n` over count-letters first, which turned
   pass@1 0.58 into pass@4 1.00 and showed `finalize()` reading a whole episode; then
   `guess-golf`, our own `Env` with a `guesser` role and an episode-level `fewest` reward (it
   turned 4 mixed groups of 10 into 10 of 10); then `configs/pyfix_judged.toml`, the built-in
   `shared-agentic-judge` env: a second agent grades the first inside its box, and on 3 of 3
   tasks it wrote its own checks rather than running the hidden tests and agreed with them.
   Pattern: `verifiers/environments/code_golf`.
8. **Production hygiene** (in progress): private material off `TaskData` (done for pyfix: the
   hidden tests and reference fix live in the package, and the test file is removed from the box
   after it runs); `.github/workflows/ci.yml` (done: lint, offline tests, `validate` for every
   taskset, a dry run of every config, the dashboard typecheck, and a model smoke rollout only
   when the repository has a `PRIME_API_KEY` secret); then `prime env push`; then a training run.

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
- A run-level `--env.agent.max-turns` is a hard cap on model calls. On an env-driven game it must
  never be below the number of turns the prompt promises, or the reward punishes rollouts that
  were about to win. Let the env bound its own episodes.

- `trace.info["judge"]` belongs to the framework: every judge call appends its raw response there.
  Record your own parsed verdicts under another key. Metrics run concurrently, so a cache shared
  by hooks must cache the pending call, not just its result.
- `replay` lifts the taskset from the saved run only when no `--taskset.*` flag is given. Any
  override, such as another judge model, needs `--taskset.id <id>` as well, or the id is empty
  and the command fails with `ValueError: Empty module name`.
- `TaskData` is serialized onto every trace. Anything in it is visible to every agent handed the
  trace record, the agentic judge included, and to everyone who reads `traces.jsonl`. Keep
  private material out of `TaskData` and look it up by key at scoring time, as pyfix now does.
