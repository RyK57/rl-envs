# count-letters

Source: [environments/toy/count_letters/README.md](../environments/toy/count_letters/README.md). Not re-run.

`deepseek/deepseek-v4-flash`, subprocess runtime, provider-default sampling, 10 tasks × 3 rollouts. Each run cost under a cent.

| Harness | Reward | Formatted | Misses |
| --- | --- | --- | --- |
| `null` (no tools) | 0.633 | 1.000 | 9 of 11 misses answered `0` without counting; the same task passes and fails across rollouts |
| `bash` (shell) | 0.933 | 0.933 | both misses were the right count inside a corrupted tag; every parsed answer was correct |

A separate best-of-n note in the previous catalog README (model not named there) turned pass@1 0.58 into pass@4 1.00.
