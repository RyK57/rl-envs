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

## Run

```bash
uv run validate count-letters --runtime.type subprocess          # model-free gold check
uv run eval @ configs/count_letters.toml --no-rich -v            # 3x1 smoke test, needs a model key
```

## Changelog

- 2026-09-09: Initial v1 taskset.
