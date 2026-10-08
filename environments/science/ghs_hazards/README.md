# ghs-hazards

Give a substance its GHS hazard statements (the H-codes of its safety data sheet) from its
structure. Single-turn, no tools by default. Every substance that is sold, shipped or stored
carries a GHS classification, and the hazard statements are the part that decides the label, the
pictograms, the storage class and the handling rules; vendors and EHS teams assign them by hand
from the structure, the literature and the regulatory lists.

## Taskset

- **Source:** [`chemNLP/msds_sigma_aldrich`](https://huggingface.co/datasets/chemNLP/msds_sigma_aldrich),
  the ChemNLP project's table of Sigma-Aldrich safety data (MIT as published): one row per
  substance with a SMILES string, a CAS registry number and the H-codes its data sheet lists, as
  58 one-hot columns. The hazard statements themselves are GHS/CLP regulatory classifications; the
  compilation comes from vendor data sheets, so treat the provenance as "as published by ChemNLP".
- **Size:** 6,420 rows in the file; 5,866 usable after dropping 431 rows without a structure, 52
  with non-standard SMILES (OpenEye coordination-bond notation) and 72 duplicate structures.
  Splits by a hash of the CAS number: 4,679 train, 598 validation, 589 test. 1,454 of the usable
  substances (1,144 / 164 / 146) carry no hazard statement; their right answer is `none`, and the
  `negatives` knob drops them.
- **Labels:** 58 distinct codes; H319, H315, H335, H318, H302 and H314 account for most of them,
  a hazardous substance carries 3.5 on average. `H211` occurs once; it is not a GHS statement (it
  most likely came from the EU supplemental statement EUH211) and is kept as its own hazard class.
  Two rows carry `H200` (unstable explosive) on ordinary phenols, which is label noise in the source;
  the labels are kept as published and the noise is part of what a model has to learn through.
- **Pins:** the file is pinned by its SHA-256 in `taskset.py`, so a changed upload fails loudly;
  `REVISION` is the dataset commit and is filled in by `scripts/pin_revisions.py` on a machine that
  reaches Hugging Face.

Task keys: `ghs:<split>:<cas>`. Gold stays off `TaskData` and is looked up by key at scoring
time; the prompt shows the SMILES only, so the task is classification from structure. The `cas`
knob adds the CAS number, which turns it into recognition of a known substance.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |
| `--env.taskset.cas` | false | show the CAS registry number next to the SMILES |
| `--env.taskset.negatives` | true | keep the substances without hazard statements (`none` is the answer) |

## Signals

- **Reward** `hazard_f1` (weight 1.0): set F1 over the hazard statements, where a predicted code
  of the right hazard class but the wrong category (H301 for H302, both acute oral toxicity; H272
  for H271, both oxidizers) earns half a match. Partial matches pair one-to-one within a class.
  `none` scores 1.0 on a substance without statements and 0 on one with them. A reply without a
  boxed list of H-codes or a boxed `none` scores 0. The hazard classes follow GHS Rev. 9, Annex 3.
- **Metric** `exact`: the predicted set is the data sheet's set.
- **Metric** `strict`: set F1 with no credit for a wrong category.
- **Metric** `hazard_class`: F1 over hazard classes, the level a pictogram is chosen at.
- **Metric** `precision`, `recall`: over exact codes.
- **Metric** `formatted`: a boxed list of H-codes or a boxed `none` was given.
- **Metric** `valid_codes`: every predicted code is a GHS hazard statement, right or wrong.
- **Metric** `num_codes`: how many codes were predicted.

## Run

```bash
uv run validate ghs-hazards --runtime.type subprocess -n 20   # rows, structures, well-formed codes, gold scores 1.0
uv run --env-file .env eval @ configs/ghs_hazards.toml --no-rich
```

## Validation

Offline, on the pinned file: 5,866 rows parse, every CAS is unique, every one-hot row agrees with
its `h_statements` list, and the gold of every row, written as instructed, scores 1.0 through the
reward. `uv run validate` on a machine that reaches Hugging Face runs the same checks per row.

## Baseline

To be recorded from the first runs: a frontier model and Qwen3-1.7B on the 589 held-out substances,
with and without the `cas` knob, next to the `formatted`, `exact`, `strict` and `hazard_class`
numbers.

## Changelog

- 2026-10-08: Initial v1 taskset.
