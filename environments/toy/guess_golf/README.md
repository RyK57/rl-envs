# guess-golf

An env, not a taskset. It plays `--env.attempts` games of one `number-guess` task per episode and
scores them against each other: every game keeps the task's own `solved` reward, and `finalize()`
adds `fewest` (weight 0.5) for the solved game that used the fewest guesses, ties sharing. A failed
game earns nothing on the comparison; when no game is solved, nobody is paid.

The new mechanism is the episode-level reward. Everything before this scored one trace at a time;
`finalize(task, episode)` runs after every game is played and scored, sees all of them, and may
record more rewards and metrics on each. That is how "best of the group" signals (shortest correct
program, fastest passing solution, fewest guesses) are built. Pattern: `verifiers/environments/code_golf`.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.id` | set in the config to `number-guess` | the taskset whose tasks are played |
| `--env.attempts` | 4 | games per episode |
| `--env.guesser.harness.id` | `null` in the config | the guesser's harness; the role is named `guesser` |

## Signals

- **Reward** `solved` (weight 1.0): from `number-guess`, per game.
- **Reward** `fewest` (weight 0.5): 1 for the solved game with the fewest guesses in its episode.
- **Metric** `any_solved`: 1 on every game of an episode where at least one game was solved.
- Plus `number-guess`'s `guesses_used` and `consistent`.

## Run

```bash
uv run eval @ configs/guess_golf.toml --dry-run --no-rich    # resolve the config, no calls
uv run --env-file .env eval @ configs/guess_golf.toml --no-rich
```

## Changelog

- 2026-09-10: Initial env.
