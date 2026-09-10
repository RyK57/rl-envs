# number-guess

Find a secret whole number from "higher" / "lower" feedback. Multi-turn: the model guesses inside
`<guess></guess>` tags and the environment answers after every guess. The model never sees the
secret. No tools, no sandbox.

The package exports `NumberGuessEnv` next to the taskset, so a plain `uv run eval number-guess`
plays the game: the env holds the rollout open turn by turn and decides when the episode ends
(solved, guesses exhausted, or a turn without a guess).

## Taskset

- **Source:** procedural (`number_guess/taskset.py`), secrets drawn from a seed
- **Size:** `num_tasks` rows (default 100)

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.num-tasks` | 100 | how many rows to generate |
| `--env.taskset.seed` | 0 | secret generator seed |
| `--env.taskset.max-number` | 5000 | secrets are drawn from 1..max_number |
| `--env.taskset.max-guesses` | 13 | guesses per episode (difficulty, with `max_number`) |

1..5000 in 13 guesses is exactly what binary search needs (2^13 = 8192): a perfect player always
wins and one slip loses. That is where deepseek-v4-flash starts to fail; 1..100 in 7 is saturated.

## Signals

- **Reward** `solved` (weight 1.0): a guess equalled the secret within the guess limit.
- **Metric** `guesses_used`: guesses up to and including the solve, or all of them when unsolved.
- **Metric** `consistent`: every guess stayed inside the interval implied by the feedback before it.

## Baseline

`deepseek/deepseek-v4-flash`, `null` harness, subprocess runtime, provider-default sampling,
10 tasks x 3 rollouts, $0.016.

| Range / guesses | Reward | consistent | guesses_used | What happened |
| --- | --- | --- | --- | --- |
| 1..100 / 7 | 1.000 | 1.000 | 5.23 | textbook binary search on every rollout; the three rollouts of a task made identical guesses |
| 1..100 / 7, temperature 1.0 | 1.000 | 1.000 | 5.23 | identical guesses again; sampling is not the lever for this model |
| 1..5000 / 13, temperature 1.0 | 0.833 | 0.933 | 10.97 | two rollouts guessed outside their own feedback interval (4-digit range tracking slips); 4 of the 5 losses were cut one guess short by a 12-turn cap in the run config, since removed |
| 1..5000 / 13, temperature 1.0, no cap (default) | 0.900 | 0.900 | 11.70 | every loss is a guess outside the feedback interval; 3 of 10 groups mixed, so there is a training signal |

At 1..100 every group is all-pass and there is no training signal; temperature does not change
that. At 1..5000 / 13 the model starts to slip while tracking the range, which is the signal.
Never cap `--env.agent.max-turns` below `max_guesses`: the env already bounds the game, and a
lower cap ends episodes before the last guess the prompt promises.

## Run

```bash
uv run validate number-guess --runtime.type subprocess          # every row winnable by binary search
uv run eval @ configs/number_guess.toml --no-rich                # 3x1 smoke test, needs a model key
```

## Changelog

- 2026-09-10: Defaults changed to 1..5000 / 13 guesses; baseline 0.90 for deepseek-v4-flash at temperature 1.0.
- 2026-09-10: Drop the 12-turn cap from the smoke config; it ended episodes below `max_guesses`. Baseline at 1..5000 / 13 recorded.
- 2026-09-10: Baseline recorded at 1..100 / 7 guesses: saturated for deepseek-v4-flash.
- 2026-09-10: Initial v1 taskset with its env.
