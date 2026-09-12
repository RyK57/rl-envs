# medical-coding

Assign ICD-10-CM diagnosis codes to a clinical history. Single-turn, no tools. Every hospital
and clinic encounter is coded before it is billed, by certified coders reading the chart, and the
code decides the payment and the audit risk; medical coding is one of the largest outsourced
back-office functions in US healthcare.

## Taskset

Two sources.

- **`cases`**: `mkurman/clinical-case-icd10-diagnosis`, 1,798 de-identified clinical histories
  drawn from open-access case reports in PubMed Central, each paired with the ICD-10-CM code of
  its principal diagnosis. The `commercial` configuration (1,549 rows whose source articles are
  CC BY or CC0) is used, so every case is licensed for commercial use. The presentation only: the
  diagnosis is removed from the text. Splits by hash of the case id: one in ten to `test`, one in
  ten to `validation`, the rest to `train`.
- **`codiesp`**: the CodiEsp corpus (`bigbio/codiesp`, CC BY 4.0), 1,000 Spanish clinical cases
  coded by professional clinical coders for the CLEF 2020 shared task, with an inter-annotator
  agreement of 88.6% on diagnosis codes. The diagnosis sub-task: every ICD-10-CM code that applies
  to a case, several per case. Its own splits: 500 train, 250 validation, 250 test.

The full ICD-10-CM table (`awacke1/ICD10-Clinical-Terminology`, 72,800 codes, MIT packaging of
the public code set) backs the validity metric. Codes are compared without dots: `J05.11` and
`j0511` are the same code.

Task keys: `icd10:<source>:<case id>`. Gold stays off `TaskData` and is looked up at scoring
time. Dataset revisions are unpinned until the first fetch; the CI "Dataset revisions" step and
`uv run python scripts/pin_revisions.py medical_coding` print the commits to pin in `taskset.py`.

Data facts: to be recorded after the first `validate` run in each source (rows per split, any
gold code that is not well formed).

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.source` | `cases` | `cases` (one principal code, English) or `codiesp` (all diagnosis codes, Spanish) |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |

## Signals

- **Reward** `coding` (weight 1.0). `cases`: the three-character category right earns 0.5 and
  every further matching character of the gold code shares the other 0.5, so `J05.1` against
  `J05.11` scores 0.75; more than one code, or no code in `\boxed{}`, scores 0. `codiesp`: F1
  between the set of codes given and the coders' set.
- **Metric** `exact`: the reward is 1.0.
- **Metric** `category`: the three-character category is right (`cases`), or F1 over categories
  (`codiesp`).
- **Metric** `formatted`: codes were given in the required form.
- **Metric** `valid_code`: every code given exists in the ICD-10-CM table, right or wrong.

## Run

```bash
uv run validate medical-coding --runtime.type subprocess -n 20
uv run validate medical-coding --runtime.type subprocess -n 20 --taskset.source codiesp
uv run --env-file .env eval @ configs/medical_coding.toml --no-rich
uv run --env-file .env eval @ configs/medical_coding.toml --no-rich --env.taskset.split test
uv run --env-file .env eval @ configs/medical_coding.toml --no-rich --env.taskset.source codiesp --env.taskset.split test
```

## Baseline

To be recorded: a frontier model and Qwen3-1.7B on the held-out case histories and the CodiEsp
test cases. Published reference: CodiEsp's best shared-task system reached a MAP of about 0.59
on diagnosis coding in 2020.

## Changelog

- 2026-09-12: Initial v1 taskset, both sources.
