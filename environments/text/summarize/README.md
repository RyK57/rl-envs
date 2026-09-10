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
- **Info** `verdict`: the parsed verdict, with the judge's quoted `issue` when it found one.

## Run

```bash
uv run validate summarize --runtime.type subprocess     # structural checks, no model
uv run eval @ configs/summarize.toml --no-rich          # 3x1 smoke test, needs a model key
uv run replay outputs/<run-dir> -r 3 --rich false       # re-judge saved traces 3x: judge noise
uv run replay outputs/<run-dir> --taskset.id summarize --taskset.task.judge.model openai/gpt-5.4-nano --rich false
uv run python scripts/judges.py outputs/<replay-dir> outputs/<other-replay-dir>   # verdicts side by side
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

The same 26 summaries re-judged by `openai/gpt-5.4-nano` (`replay --taskset.task.judge.model`):

- mean reward 0.40, `faithful` 0.54, `covered` 2.50
- 12 of 26 summaries called unfaithful, against 0 of 26 by deepseek

So the noise floor of this reward is about a third of its range, and the two judges do not even
agree on what the reward measures. Which judge is right was decided by reading the summaries
against the passages with `scripts/judges.py` and the judge's quoted `issue`:

- a second nano pass objected to 15 of 26 summaries; every objection was an omission (6), a
  paraphrase or generalization of something the passage does say (8), or a rationale that
  concluded "faithful" and then answered false (1). None was an unsupported claim.
- deepseek's 26 faithful verdicts were all right; its noise is only in the coverage count.

The judge prompt now says explicitly that omissions, paraphrase and generalization are not
unfaithfulness and that a key point counts without its details. Re-judged with that prompt:

| | deepseek, 3 re-scores | gpt-5.4-nano |
| --- | --- | --- |
| mean reward | 0.80 (was 0.76) | 0.67 (was 0.40) |
| `faithful` | 0.99, one borderline verdict in 78 | 0.85 |
| `covered` of 3 | 2.87 (was 2.73) | 2.73 |
| rollouts re-scored differently | 7 of 26 (was 9) | |
| wrong "unfaithful" objections | 0 | 4 of 26 (was 15), all omissions |

Decision: the default judge stays `deepseek/deepseek-v4-flash`, one call per rollout, with a noise
floor of about 7 in 26 rollouts moving by one key point. `gpt-5.4-nano` is unsuitable for this
rubric: even told not to, it still calls omissions unfaithful. Levers not yet tried: a rubric per
key point, judge temperature 0, several judge samples with a majority vote.

## Changelog

- 2026-09-10: Judge prompt: omissions and paraphrase are not unfaithfulness, after adjudicating a second judge's objections.
- 2026-09-10: The judge quotes the unsupported claim (`issue`) so verdicts can be audited; judge comparison recorded.
- 2026-09-10: Fix a double judge call under concurrent metrics and a clash with the framework's `info.judge` key.
- 2026-09-10: Initial v1 taskset.
