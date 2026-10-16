# gsm8k

Source: [environments/math/gsm8k/README.md](../environments/math/gsm8k/README.md). Not re-run.

`deepseek/deepseek-v4-flash`, `null` harness, subprocess runtime, temperature 1.0, 10 tasks × 3 rollouts, $0.012.

| Harness | Reward | boxed | abs_error | Misses |
| --- | --- | --- | --- | --- |
| `null` (no tools) | 0.933 | 0.967 | 2.50 | one reasoning error (15 for 45); one empty reply with no usage record |

One mixed group in ten. The README calls this near-saturated for this model.
