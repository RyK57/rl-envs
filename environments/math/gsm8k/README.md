# gsm8k

Grade-school math word problems. Single-turn: the model reasons and puts its final answer as a
number inside `\boxed{}`; the reward is math equivalence against the gold number, so `18`,
`18.0` and `$18` all count. No tools, no sandbox.

The new mechanism in this environment is data. Rows come from a Hugging Face dataset at a pinned
revision through a lazy loader, so `-n 3` builds three tasks and the same revision always yields
the same rows. Each task's identity is `gsm8k:<split>:<row>`, stable across runs.

## Taskset

- **Source:** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), config `main`,
  revision pinned in `gsm8k/taskset.py`
- **Size:** `test` 1319 problems (default), `train` 7473

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.split` | `test` | `test` or `train` |

## Signals

- **Reward** `correct` (weight 1.0): the boxed answer is mathematically equal to the gold (`vf.verify_boxed_math_answer`).
- **Metric** `boxed`: the reply carried a well-formed `\boxed{}`.
- **Metric** `abs_error`: distance between the boxed number and the gold (missing or non-numeric counts as 0).

## Baseline

`deepseek/deepseek-v4-flash`, `null` harness, subprocess runtime, temperature 1.0, 10 tasks x 3
rollouts, $0.012.

| Harness | Reward | boxed | abs_error | What the misses were |
| --- | --- | --- | --- | --- |
| `null` (no tools) | 0.933 | 0.967 | 2.50 | one reasoning error (15 for 45); one empty reply with no usage record, a provider anomaly rather than a model mistake |

Near-saturated for this model, as GSM8K is for most current models: one mixed group in ten.
Useful as a regression eval and as a training environment only for weaker policies.

## Run

```bash
uv run validate gsm8k --runtime.type subprocess -n 50     # gold parses and scores 1.0 through the reward
uv run eval @ configs/gsm8k.toml --no-rich                 # 3x1 smoke test, needs a model key
```

## Changelog

- 2026-09-10: Baseline recorded on the `null` harness.
- 2026-09-10: Initial v1 taskset.
