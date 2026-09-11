# Algorithm overlays

One file per training algorithm, layered on any base run config with a second `@`:

```bash
uv run rl @ hts_classify_rl.toml                      # the base file alone is grpo
uv run rl @ hts_classify_rl.toml @ algo/gspo.toml     # same run, same data, GSPO
```

prime-rl deep-merges the files left to right, so an overlay changes only the algorithm. The
advantage (credit assignment) is `[orchestrator.algo]`; the loss the trainer applies to those
advantages is `[trainer.loss]`. The `custom` losses live in `rl_losses/` and must be installed in
the prime-rl venv first (`uv pip install -e ~/rl-envs/rl_losses`).

| Overlay | Advantage | Loss | Paper |
| --- | --- | --- | --- |
| `grpo.toml` | reward minus group mean | IPO (prime-rl's default) | DeepSeekMath, Dr. GRPO |
| `grpo_length_penalty.toml` | as above, rewards shaped by output length | IPO | prime-rl |
| `max_rl.toml` | reward minus group mean, over the mean | IPO | MaxRL (arXiv:2602.02710) |
| `reinforce_ema.toml` | reward minus a running per-agent baseline, no group | IPO | REINFORCE; SPIRAL's RAE |
| `ppo_clip.toml` | group mean | PPO clipped surrogate, token level | PPO (arXiv:1707.06347) |
| `dapo.toml` | group mean | PPO clip with the higher clip raised (clip-higher) | DAPO (arXiv:2503.14476) |
| `gspo.toml` | group mean | sequence-level importance ratio, clipped | GSPO (arXiv:2507.18071) |
| `cispo.toml` | group mean | clipped importance weight on the REINFORCE term | MiniMax-M1 (arXiv:2506.13585) |
| `pg_is.toml` | group mean | importance-weighted policy gradient, no clipping | REINFORCE with importance sampling |
| `reinforce.toml` | group mean | plain policy gradient, no importance ratio | REINFORCE (Williams 1992) |
| `opd.toml` | none: per-token reverse KL to a teacher | ref_kl | on-policy distillation |

What is not here: PPO with a learned value network. prime-rl is critic-free by design, so every
variant above estimates the baseline from the group (or a running mean) instead of from a
critic. DAPO's dynamic sampling, dropping groups whose rollouts all scored the same, is what
prime-rl does by default when it filters zero-advantage samples and refills the batch.
