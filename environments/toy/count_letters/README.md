# count-letters

Count how often a letter appears in a short passage. Single-turn, no tools, no sandbox: the
model replies once with an integer inside `<answer></answer>` tags and is scored by exact match.

Rows are generated from a seed, so the taskset needs no dataset download and is fully
reproducible.

## Taskset

- **Source:** procedural (`count_letters/taskset.py`), passages of common English words
- **Size:** `num_tasks` rows (default 100)

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.num-tasks` | 100 | how many rows to generate |
| `--env.taskset.seed` | 0 | row generator seed |
| `--env.taskset.min-words` | 8 | shortest passage, in words |
| `--env.taskset.max-words` | 24 | longest passage, in words (difficulty) |

## Signals

- **Reward** `correct` (weight 1.0): parsed count equals the truth.
- **Metric** `formatted`: the reply carried a parseable `<answer>` tag.
- **Metric** `abs_error`: distance between the parsed count and the truth (unparseable counts as 0).

## Baseline

`deepseek/deepseek-v4-flash`, subprocess runtime, provider-default sampling, 10 tasks x 3 rollouts
(`-n 10 -r 3`). Each run cost under a cent.

| Harness | Reward | Formatted | What the misses were |
| --- | --- | --- | --- |
| `null` (no tools) | 0.633 | 1.000 | 9 of 11 misses answered `0` without counting; the same task passes and fails across rollouts |
| `bash` (shell) | 0.933 | 0.933 | both misses were the right count inside a corrupted tag (`<｜DSML｜answer>`); every parsed answer was correct |

With a shell the model solves the task with `grep -o | wc -l`, so the harness decides what this
environment trains: counting without tools (`null`, mixed rewards, a training signal) or tool use
(`bash`, nearly solved). The strict parser is kept on purpose: a polluted tag scores 0, which
matches the instruction and pushes training toward clean output.

## Run

```bash
uv run validate count-letters --runtime.type subprocess          # model-free gold check
uv run eval @ configs/count_letters.toml --no-rich -v            # 3x1 smoke test, needs a model key
```

## Changelog

- 2026-09-10: Baseline recorded for the `null` and `bash` harnesses.
- 2026-09-10: Drop the single-turn stop. It counted model calls, so under a tool-using harness (`bash`) the episode ended on the model's first tool call, before any answer. The harness now decides when the agent is done.
- 2026-09-09: Initial v1 taskset.
