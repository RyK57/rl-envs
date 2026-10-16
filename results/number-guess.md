# number-guess

Source: [environments/toy/number_guess/README.md](../environments/toy/number_guess/README.md). Not re-run.

`deepseek/deepseek-v4-flash`, `null` harness, subprocess runtime, provider-default sampling, 10 tasks × 3 rollouts, $0.016.

| Range / guesses | Reward | consistent | guesses_used | What happened |
| --- | --- | --- | --- | --- |
| 1..100 / 7 | 1.000 | 1.000 | 5.23 | binary search on every rollout |
| 1..100 / 7, temperature 1.0 | 1.000 | 1.000 | 5.23 | identical guesses; sampling did not change the result |
| 1..5000 / 13, temperature 1.0 | 0.833 | 0.933 | 10.97 | 4 of the 5 losses were cut short by a 12-turn cap in the run config, since removed |
| 1..5000 / 13, temperature 1.0, no cap (default) | 0.900 | 0.900 | 11.70 | every loss is a guess outside the feedback interval; 3 of 10 groups mixed |

At 1..100 every group passed and there is no training signal. The default difficulty is 1..5000 / 13.
