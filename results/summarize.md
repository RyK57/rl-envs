# summarize

Source: [environments/text/summarize/README.md](../environments/text/summarize/README.md). Not re-run.

`deepseek/deepseek-v4-flash` at temperature 1.0, no tools, judged by the same model. 13 tasks × 2 rollouts, re-scored three times with `replay -r 3` (78 scores).

Before the rubric fix:

- mean reward 0.76, `faithful` 1.00, `covered` 2.73 of 3, `sentences` 2.15
- 4 of 26 rollouts wrote three sentences and scored 0 from the length gate
- 9 of 26 rollouts got different rewards from the same judge across the three re-scores; each disagreement was one key point (a swing of 0.33)

The same 26 summaries re-judged by `openai/gpt-5.4-nano`:

- mean reward 0.40, `faithful` 0.54, `covered` 2.50
- 12 of 26 summaries called unfaithful, against 0 of 26 by deepseek

After the judge prompt was changed so omissions, paraphrase, and generalization are not unfaithfulness:

| | deepseek, 3 re-scores | gpt-5.4-nano |
| --- | --- | --- |
| mean reward | 0.80 (was 0.76) | 0.67 (was 0.40) |
| `faithful` | 0.99, one borderline verdict in 78 | 0.85 |
| `covered` of 3 | 2.87 (was 2.73) | 2.73 |
| rollouts re-scored differently | 7 of 26 (was 9) | |
| wrong "unfaithful" objections | 0 | 4 of 26 (was 15), all omissions |

The README keeps `deepseek/deepseek-v4-flash` as the default judge, with a noise floor of about 7 in 26 rollouts moving by one key point.
