# sanctions-screening

Decide whether two records from sanctions and watch lists describe the same real-world person or
organization. Single-turn, no tools. This is the alert-resolution step of know-your-customer and
payment screening: matching a name against a list is cheap, deciding whether the hit is the listed
person is an analyst's job, and every bank does it thousands of times a day.

## Taskset

Source: the OpenSanctions Pairs corpus, the cross-referencing judgements behind their entity
resolution, as snapshotted on Hugging Face in `sanctions-er-anon/opensanctions_pairs` (snapshot of
2025-12-09). 755,540 judged pairs of person and organization records from OFAC, UK, EU, UN and other
lists, with names, aliases, birth dates, nationalities, identifiers and addresses; 76.9% are the
same entity, 23.1% different, and the negatives are near misses by construction. Licence
CC-BY-NC-4.0, inherited from OpenSanctions: fine for this learning repo, and a commercial
deployment needs an OpenSanctions licence.

Splits: `validation` and `test` are the two halves, 500 pairs each, of the 1,000-pair stratified
sample the snapshot publishes for exactly this purpose. `train` is the first `size` pairs of the
full corpus that are not in the sample, half per verdict, interleaved: the corpus is 77% `same`, and
a model trained on it raw would learn to say `same`.

Each record is rendered with identity fields only: names, aliases, dates and places of birth,
nationality, identifiers, addresses, positions. Provenance fields (source URLs, list identifiers,
programme codes, timestamps) are left out, so the model cannot learn which list a record came from
instead of who it is.

Task keys: `sanctions:<split>:<left id>|<right id>`. Verdicts stay off `TaskData` and are looked
up at scoring time. The dataset revision is unpinned until the first fetch; `uv run python
scripts/pin_revisions.py sanctions_screening` prints the commit to pin in `taskset.py`.

Data facts: to be recorded after the first `validate` run (usable pairs per split, any dropped).

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.split` | `validation` | `train` (the corpus), `validation` or `test` (sample halves) |
| `--env.taskset.size` | 4000 | train pairs, half `same` and half `different` |
| `--env.taskset.scan-limit` | 100000 | corpus rows read at most while filling the train pairs |

## Signals

- **Reward** `verdict` (weight 1.0): the analyst's verdict reproduced, `\boxed{same}` or
  `\boxed{different}`. Anything else scores 0.
- **Metric** `formatted`: a verdict was given in the required form.
- **Metric** `false_clear`: the pair is the same entity and the reply said different. The costly
  error: a listed person waved through.
- **Metric** `false_hit`: the pair is different and the reply said same. The friction error: an
  innocent namesake blocked.

## Run

```bash
uv run validate sanctions-screening --runtime.type subprocess -n 20
uv run --env-file .env eval @ configs/sanctions_screening.toml --no-rich
uv run --env-file .env eval @ configs/sanctions_screening.toml --no-rich --env.taskset.split test -n 500
```

## Baseline

To be recorded: a frontier model and Qwen3-1.7B on the 500 test pairs, with both error rates.

## Changelog

- 2026-09-11: Initial v1 taskset.
