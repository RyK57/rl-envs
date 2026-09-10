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
uv run replay outputs/<run-dir> -r 3 --rich false       # re-judge saved traces 3x: judge noise
uv run replay outputs/<run-dir> --taskset.id summarize --taskset.task.judge.model openai/gpt-5.4-nano --rich false
```

Any `--taskset.*` override on `replay` must come with `--taskset.id`; without it the taskset is
not lifted from the saved run and the command fails with `ValueError: Empty module name`.

## Baseline

`deepseek/deepseek-v4-flash` at temperature 1.0, no tools, judged by the same model. 13 tasks x 2
rollouts, re-scored three times with `replay -r 3` (78 scores):

- mean reward 0.76, `faithful` 1.00, `covered` 2.73 of 3, `sentences` 2.15
- 4 of 26 rollouts wrote three sentences and scored 0 from the length gate, judge not involved;
  the judge counted all three key points in every one of them
- 9 of 26 rollouts got different rewards from the same judge across the three re-scores; every
  disagreement was one borderline key point counted or not, a swing of 0.33

So the noise floor of this reward is about a third of its range. The length gate is the only
clean signal; coverage differences between two rollouts of one task are as likely to be judge
noise as real. Fixes to try: a stricter judge prompt with a rubric per key point, judge
temperature 0, or several judge samples with a majority vote.

## Changelog

- 2026-09-10: Fix a double judge call under concurrent metrics and a clash with the framework's `info.judge` key.
- 2026-09-10: Initial v1 taskset.
