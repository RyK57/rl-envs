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

HUPD writes the symbol without its slash, the main group and subgroup digits run together
(`H04N214312` for H04N 21/4312), which is ambiguous on its own. The loader resolves each label
through the CPC scheme (`mhurhangee/cpc-classifications`, 262,000 symbols): the one way of
splitting the digits that names a real symbol is the label, and an application whose digits
read as two real symbols, or as none, is left out. The same scheme backs the validity metric.
Symbols are compared in the form `G06F17/30`: spaces are ignored and a dash before the subgroup
counts as a slash.

Task keys: `cpc:<application number>`. Gold stays off `TaskData` and is looked up at scoring
time. Both datasets are pinned to a commit in `taskset.py`.

Data facts: the January 2016 archive holds 26,808 applications. CI validates five tasks on every
push (5 of 5 valid on 2026-09-12). Applications per split, and how many were dropped for a
missing abstract or an unresolvable symbol, to be recorded from the first complete `validate`
run.

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
