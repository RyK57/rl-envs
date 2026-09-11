# contract-review

The clause review a lawyer does on a commercial contract, on CUAD. Single-turn, no tools. Two modes
share one taskset:

- **`classify`**: one labelled clause, which of the 41 categories is it?
- **`extract`**: an excerpt of a contract and one category; quote the passage that belongs to the
  category, or say none.

Contract review in a deal is finding these passages across hundreds of pages; the categories are
the ones attorneys look for (governing law, change of control, non-compete, cap on liability...).

## Taskset

Source: the Contract Understanding Atticus Dataset v1 (The Atticus Project, CC-BY-4.0, free for
commercial use): 510 commercial contracts from SEC filings, 13,000+ passages labelled by trained
law students under attorney review, 41 categories. Two forms of it are read:

- `classify` uses `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification`, the labels as
  13,155 (clause, category) rows over 509 contracts. Rows with the same clause text are merged, so a
  clause the annotators filed under several categories (a license grant that is also perpetual)
  accepts any of them.
- `extract` uses the SQuAD-style `CUAD_v1.json` from `theatticusproject/cuad`: every contract's
  full text and, for each category, the annotated spans or none. Each (contract, category) becomes
  an excerpt of `window` characters; for a positive the excerpt contains the first annotated span at
  a hashed offset (so the passage is not always at the top), for a negative it is a hashed excerpt
  of a contract that has no such clause. The gold for a positive is every annotated span inside the
  excerpt, clipped to it.

Splits are by contract, not by row: a hash of the contract name sends one in ten contracts to
`test`, one in ten to `validation`, the rest to `train`, so no clause of a held-out contract is ever
trained on.

Task keys: `contract:classify:<split>:<row>` and `contract:extract:<split>:<contract>:<category>`.
Gold stays off `TaskData` and is looked up at scoring time. Dataset revisions are unpinned until
the first fetch; `uv run python scripts/pin_revisions.py contract_review` prints the commits to pin
in `taskset.py`.

Data facts: to be recorded after the first `validate` run in each mode (rows per split, positives
and negatives).

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.mode` | `classify` | `classify` a labelled clause, or `extract` the clause from an excerpt |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test`, by contract |
| `--env.taskset.window` | 3000 | excerpt length in characters (`extract`) |
| `--env.taskset.negatives` | true | in `extract`, also ask about categories the contract lacks; the answer is none |

## Signals

- **Reward** `match` (weight 1.0). `classify`: the category in `\boxed{}` is one the annotators
  gave the clause. `extract`: SQuAD token F1 between the quote in `<clause></clause>` and the
  best annotated passage in the excerpt, or 1.0 for `<clause>none</clause>` on an excerpt whose
  contract has no such clause; a quote on a negative or a none on a positive scores 0.
- **Metric** `exact`: the whole passage (or the correct none), not a partial overlap.
- **Metric** `formatted`: a category name or a clause tag was given in the required form.
- **Metric** `abstained`: the reply said none (`extract`).

## Run

```bash
uv run validate contract-review --runtime.type subprocess -n 20
uv run validate contract-review --runtime.type subprocess -n 20 --taskset.mode extract
uv run --env-file .env eval @ configs/contract_review.toml --no-rich
uv run --env-file .env eval @ configs/contract_review.toml --no-rich --env.taskset.split test -n 1000
uv run --env-file .env eval @ configs/contract_review.toml --no-rich --env.taskset.mode extract --env.taskset.split test -n 500
```

## Baseline

To be recorded: a frontier model and Qwen3-1.7B on the held-out contracts in both modes. Published
reference on the extraction task: CUAD's own fine-tuned DeBERTa-xlarge reached 47.8% precision at
80% recall in the original paper.

## Changelog

- 2026-09-11: Initial v1 taskset, both modes.
