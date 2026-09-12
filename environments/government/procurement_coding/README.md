# procurement-coding

Code a federal contract action to the NAICS industry of the work, or to the four-character
Product or Service Code (PSC) of what was bought. Single-turn, no tools. Contracting officers
assign both codes to every action by hand; set-aside eligibility, size standards and spend
analysis all key off them, and government-contracting vendors search opportunities by them.

## Taskset

Source: USAspending's FY2024 contract transactions (`zzsi/usaspending_2024_all_contracts`, US
government data, public domain), the first of the archive's 24 parquet shards, read with only the
six columns the task needs. An action is kept when its description, with the
inherently-governmental-function tags removed, is at least `min_chars` long and mostly letters,
its NAICS code has six digits and its PSC four characters; descriptions are deduplicated, first
occurrence kept. Splits by hash of the award key: one in ten to `test`, one in ten to
`validation`, the rest to `train`.

These labels are the ones officers actually recorded, not a curated benchmark: some are wrong or
lazy, and that noise is part of what a model has to learn through. The README of a run should
say how often the frontier model and the record disagree on the sector.

Task keys: `procurement:<field>:<award key>:<row>`. Gold stays off `TaskData` and is looked up
at scoring time. The dataset is pinned to a commit in `taskset.py`.

Data facts: CI validates five tasks for each code on every push (5 of 5 valid in both on
2026-09-12). Actions kept per split, and how many were dropped as short or numeric descriptions
or duplicates, to be recorded from the first complete `validate` run.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.field` | `naics` | `naics` (six-digit industry) or `psc` (four-character product or service code) |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |
| `--env.taskset.min-chars` | 60 | shortest description kept |

## Signals

- **Reward** `hierarchical` (weight 1.0): leading characters shared with the recorded code, over
  its length: for NAICS the sector earns 0.33 and the full code 1.0; for PSC each character is a
  quarter. No well-formed code in `\boxed{}` scores 0.
- **Metric** `exact`: the full code.
- **Metric** `sector`: the first two characters.
- **Metric** `formatted`: a code was given in the required form.

## Run

```bash
uv run validate procurement-coding --runtime.type subprocess -n 20
uv run validate procurement-coding --runtime.type subprocess -n 20 --taskset.field psc
uv run --env-file .env eval @ configs/procurement_coding.toml --no-rich
uv run --env-file .env eval @ configs/procurement_coding.toml --no-rich --env.taskset.split test -n 500
uv run --env-file .env eval @ configs/procurement_coding.toml --no-rich --env.taskset.field psc --env.taskset.split test -n 500
```

## Baseline

To be recorded: a frontier model and Qwen3-1.7B on 500 held-out actions for each code.

## Changelog

- 2026-09-12: Initial v1 taskset, both codes.
