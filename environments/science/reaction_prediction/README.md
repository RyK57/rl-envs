# reaction-prediction

Write the product of an organic reaction from its reactants. Single-turn, no tools by default.
Predicting what a reaction makes is the daily work of process and medicinal chemists, and it has an
exact answer in the patent record: two SMILES are the same product when RDKit canonicalizes them to
the same string. Pairs with `iupac-structure` and `mol-properties`: the same molecules are named,
counted and now made.

## Taskset

- **Source:** [`bisectgroup/USPTO_50K`](https://huggingface.co/datasets/bisectgroup/USPTO_50K),
  the Hub mirror of USPTO-50K (Schneider et al., 2016): 50,016 single-product reactions extracted
  from US patents by Lowe (CC0) with the standard train / val / test split (40,008 / 5,001 / 5,007),
  as six parquet shards per split with atom-mapped reactant and product SMILES.
- **Filters:** the atom-map numbers are stripped from both sides before anything else, since the
  mapping would reveal which atoms survive into the product; a row is kept when it has reactants
  and exactly one product fragment. The exact number of usable rows per split is to be recorded on
  the first fetch.
- **Pins:** `REVISION` is the dataset commit, filled in by `scripts/pin_revisions.py` on a machine
  that reaches Hugging Face.

Task keys: `rxn:<split>:<row>:<patent id>`. Gold stays off `TaskData` and is looked up by key at
scoring time; the prompt shows the reactants only, as dot-separated SMILES.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |
| `--env.taskset.task.ignore-stereo` | false | compare the product with stereochemistry stripped from both sides |

## Signals

- **Reward** `correct` (weight 1.0): 1.0 when RDKit canonicalizes the answer to the recorded
  product (atom maps cleared on both sides, stereochemistry included unless `ignore_stereo`),
  otherwise half the Tanimoto similarity of Morgan fingerprints (radius 2, 2048 bits) between a
  parseable answer and the product, and 0 for an answer RDKit cannot read. A reply without one
  boxed token scores 0.
- **Metric** `formatted`: one boxed token was given.
- **Metric** `exact`: the product is exactly right.
- **Metric** `parsed`: the answer is a valid SMILES, right or wrong.
- **Metric** `similarity`: the Tanimoto similarity to the product.
- **Metric** `formula_match`: the answer has the product's molecular formula, right or wrong
  structure (the right atoms in the wrong place).
- `trace.info["verify"]` keeps the script's full report for every scored rollout.

The comparison runs as a PEP 723 script (`verify.py`, RDKit 2026.3.6) through
`runtime.run_uv_script`, installed once per runtime in `setup()` while egress is open; it prints one
JSON line and exits non-zero only on a malformed call, so an infrastructure failure never scores as
a wrong answer. Because the score needs the runtime, `replay` cannot re-score runs.

## Run

```bash
uv run validate reaction-prediction --runtime.type subprocess -n 20     # rows, no maps left, products RDKit reads back as themselves
uv run --env-file .env eval @ configs/reaction_prediction.toml --no-rich
uv run --env-file .env eval @ configs/reaction_prediction.toml --env.taskset.split test -n 500 --no-rich
```

## Validation

To be recorded on the first fetch: the usable rows per split and `uv run validate` over each
(every product canonicalizes to itself under the script with its maps cleared).

## Baseline

To be recorded from the first runs: a frontier model and Qwen3-1.7B on the 5,007 test reactions,
next to the `formatted`, `exact`, `similarity` and `formula_match` numbers and the published
top-1 accuracies of template-free models on this split.

## Changelog

- 2026-10-08: Initial v1 taskset.
