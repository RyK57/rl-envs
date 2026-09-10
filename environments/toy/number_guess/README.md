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

## Run

```bash
uv run validate number-guess --runtime.type subprocess          # every row winnable by binary search
uv run eval @ configs/number_guess.toml --no-rich                # 3x1 smoke test, needs a model key
```

## Changelog

- 2026-09-10: Initial v1 taskset with its env.
