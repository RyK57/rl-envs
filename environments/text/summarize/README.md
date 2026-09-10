# summarize

Summarize a short passage in at most two sentences. Single-turn, no tools. The reward asks a
judge model whether the summary is faithful to the passage and how many of the passage's three
key points it conveys, then scores `covered / 3` when faithful and 0 otherwise. A reply over the
sentence limit scores 0 before the judge is called.

The new mechanism in this environment is the judge. It is a run-time knob (`SummaryJudge` reads
`--env.taskset.task.judge.*`), it grades only against the passage and is told the summary is
untrusted, and a verdict it cannot parse raises so the model is never scored for the judge's
failure. One judge call per rollout; the parsed verdict is recorded as `info.verdict` (the framework keeps the raw judge responses under `info.judge`).

## Taskset

- **Source:** `summarize/catalog.py`, thirteen fictional passages (notices, memos, reports) with three
  key points and a reference summary each
- **Size:** 13 tasks, keys `summarize:<name>`

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.task.judge.model` | `deepseek/deepseek-v4-flash` | the grader |
| `--env.taskset.task.judge.sampling.temperature` | provider default | judge sampling |
| `--env.taskset.task.max-sentences` | 2 | limit stated in the prompt and enforced by the reward |

## Signals

- **Reward** `summary` (weight 1.0): 0 over the sentence limit or if unfaithful, else key points covered / 3.
- **Metric** `sentences`: sentences in the reply.
- **Metric** `faithful`: the judge found no unsupported claim.
- **Metric** `covered`: key points the judge counted.

## Run

```bash
uv run validate summarize --runtime.type subprocess     # structural checks, no model
uv run eval @ configs/summarize.toml --no-rich          # 3x1 smoke test, needs a model key
```

## Changelog

- 2026-09-10: Fix a double judge call under concurrent metrics and a clash with the framework's `info.judge` key.
- 2026-09-10: Initial v1 taskset.
