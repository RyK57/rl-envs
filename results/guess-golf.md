# guess-golf

Source: [environments/toy/guess_golf/README.md](../environments/toy/guess_golf/README.md). Not re-run.

`deepseek/deepseek-v4-flash`, no tools, temperature 1.0, 10 tasks × 4 games, 1..5000 in 13 guesses:

- mean reward 1.14 of 1.5
- `any_solved` 1.00
- 36 of 40 games solved
- `guesses_used` 11.2
- with `solved` alone, 4 of 10 groups had mixed rewards; with `fewest`, 10 of 10 do
- 2 losses guessed wrong at the limit, 2 ended with an empty reply mid-game

The README notes that all four games share one secret, so `fewest` pays a lucky off-midpoint guess as readily as a better strategy.
