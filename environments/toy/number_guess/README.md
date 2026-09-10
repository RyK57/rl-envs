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
| `--env.taskset.max-number` | 100 | secrets are drawn from 1..max_number |
| `--env.taskset.max-guesses` | 7 | guesses per episode (difficulty, with `max_number`) |

1..100 in 7 guesses is exactly what binary search needs: a perfect player always wins, a careless
one often loses.

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

Saturated: every group is all-pass, so there is no training signal at this setting. The two
levers that need no code change are the knobs (`max_number`, `max_guesses`) and the sampling
temperature, which at provider defaults produced no variance between rollouts.

## Run

```bash
uv run validate number-guess --runtime.type subprocess          # every row winnable by binary search
uv run eval @ configs/number_guess.toml --no-rich                # 3x1 smoke test, needs a model key
```

## Changelog

- 2026-09-10: Baseline recorded at 1..100 / 7 guesses: saturated for deepseek-v4-flash.
- 2026-09-10: Initial v1 taskset with its env.
