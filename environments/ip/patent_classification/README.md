# patent-classification

Give a patent application its main Cooperative Patent Classification (CPC) symbol from its
title and abstract. Single-turn, no tools. Every application is classified so that it reaches
the right examiner and so that prior art can be searched; patent offices do it by hand and train
their own models for it, and IP firms and analytics vendors need it for every portfolio.

## Taskset

Source: the Harvard USPTO Patent Dataset (`HUPD/hupd`), the January 2016 sample: every utility
application filed to the USPTO that month, as submitted by the applicant, with the main CPC
symbol the office assigned and the eventual decision. Licence CC BY-NC-SA 4.0, so a commercial
deployment needs its own data (any patent office's published applications carry the same
labels). The sample is a 390 MB archive of one JSON file per application with the full text; the
loader reads it once and keeps a table of application number, title, abstract, symbol and
decision under `~/.cache/huggingface/rl-envs/`, so later runs start in seconds.

Applications without an abstract or without a well-formed main symbol are dropped. Splits by
hash of the application number: one in ten to `test`, one in ten to `validation`, the rest to
`train`.

The CPC scheme (`mhurhangee/cpc-classifications`, 262,000 symbols with titles) backs the
validity metric. Symbols are compared in the form `G06F17/30`: spaces are ignored and a dash
before the subgroup counts as a slash.

Task keys: `cpc:<application number>`. Gold stays off `TaskData` and is looked up at scoring
time. Dataset revisions are unpinned until the first fetch; the CI "Dataset revisions" step and
`uv run python scripts/pin_revisions.py patent_classification` print the commits to pin in
`taskset.py`.

Data facts: to be recorded after the first `validate` run (applications per split, dropped for a
missing abstract or symbol).

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |

## Signals

- **Reward** `hierarchical` (weight 1.0): leading CPC levels shared with the office's symbol,
  over five: section 0.2, class 0.4, subclass 0.6, main group 0.8, subgroup 1.0. No symbol in
  `\boxed{}` scores 0.
- **Metric** `exact`: the full symbol.
- **Metric** `subclass`: the four-character subclass, the level examiners are assigned by.
- **Metric** `formatted`: a symbol was given in the required form.
- **Metric** `valid_symbol`: the symbol exists in the CPC scheme, right or wrong.

## Run

```bash
uv run validate patent-classification --runtime.type subprocess -n 20
uv run --env-file .env eval @ configs/patent_classification.toml --no-rich
uv run --env-file .env eval @ configs/patent_classification.toml --no-rich --env.taskset.split test -n 500
```

## Baseline

To be recorded: a frontier model and Qwen3-1.7B on 500 held-out applications. Published
reference: the HUPD paper reports subclass-level classification from the abstract at about 60%
accuracy for fine-tuned encoders.

## Changelog

- 2026-09-12: Initial v1 taskset.
