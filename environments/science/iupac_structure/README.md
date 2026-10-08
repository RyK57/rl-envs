# iupac-structure

Write the SMILES string, or the molecular formula, of a compound from its IUPAC name. Single-turn,
no tools by default. Reading a systematic name into a structure is the first step of reading any
chemistry text, patent claim or safety data sheet, and it is checkable exactly: two SMILES are the
same compound when RDKit canonicalizes them to the same string.

## Taskset

- **Source:** [`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC),
  a CC0 re-packaging of the January 2026 PubChem dump (public domain data) with each compound's
  RDKit-canonical SMILES, molecular formula and preferred IUPAC name. One shard
  (`data/shard_001.parquet`, 168 MB, about a million compounds in a globally shuffled order) is read
  by column and reduced once to a 20,000-row table next to the Hugging Face cache. The dump's
  85 MB shards (`shard_003`, `shard_011`) carry no IUPAC names or InChI at all, so a full-sized
  shard is the one to read.
- **Filters:** a compound is kept when it has a name, a structure and a formula; the name is ASCII,
  at most 120 characters and not lambda notation; the SMILES is one fragment (no salts) without
  isotopes or dummy atoms and at most 120 characters. The first 20,000 compounds that pass, in shard
  order, form the pool; splits are a hash of the PubChem CID (about 80 / 10 / 10). The exact
  split sizes are to be recorded on the first fetch.
- **Pins:** `REVISION` is the dataset commit, filled in by `scripts/pin_revisions.py` on a machine
  that reaches Hugging Face; the cache file is keyed by the shard's size and the pool size.

Task keys: `iupac:<target>:<cid>`. Gold stays off `TaskData` and is looked up by key at scoring
time.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.target` | `smiles` | `smiles` (scored by RDKit in the runtime) or `formula` (scored offline) |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |
| `--env.taskset.max-smiles-chars` | 60 | keep compounds whose canonical SMILES is at most this long; the difficulty knob |
| `--env.taskset.formula-hint` | false | in `smiles` mode, show the molecular formula next to the name |
| `--env.taskset.task.ignore-stereo` | false | score SMILES answers with stereochemistry stripped from both sides |

## Signals

- **Reward** `correct` (weight 1.0). `smiles`: 1.0 when RDKit canonicalizes the answer to the gold
  structure (stereochemistry included unless `ignore_stereo`), otherwise half the Tanimoto
  similarity of Morgan fingerprints (radius 2, 2048 bits) between a parseable answer and the gold,
  and 0 for an answer RDKit cannot read. An enantiomer or a missing stereocentre therefore scores
  0.5, a wrong substituent position about 0.3, an unrelated molecule close to 0. `formula`: 1.0 for
  the gold Hill formula (subscript digits and a unicode minus are normalized), else 0.
- **Metric** `formatted`: one boxed token was given.
- **Metric** `exact`: the structure, or the formula, is exactly right.
- **Metric** `parsed`: the answer is a valid SMILES (or a well-formed formula), right or wrong.
- **Metric** `similarity`: the Tanimoto similarity (0 or 1 for a formula).
- `trace.info["verify"]` keeps the script's full report for every scored rollout.

The comparison runs as a PEP 723 script (`verify.py`, RDKit 2026.3.6) through
`runtime.run_uv_script`, the same path the verifiers `gsm8k` reference uses: it is installed once
per runtime in `setup()` while egress is open, prints one JSON line, and exits non-zero only on a
malformed call, so an infrastructure failure never scores as a wrong answer. Because the score
needs the runtime, `replay` cannot re-score `smiles` runs; `formula` runs re-score offline.

## Run

```bash
uv run validate iupac-structure --runtime.type subprocess -n 20     # names, structures RDKit reads back as themselves
uv run --env-file .env eval @ configs/iupac_structure.toml --no-rich
uv run --env-file .env eval @ configs/iupac_structure.toml --env.taskset.target formula --no-rich
```

## Validation

To be recorded on the first fetch: the pool's split sizes, how many names the filters dropped, and
`uv run validate` over the pool (every gold SMILES canonicalizes to itself under the script).

## Baseline

To be recorded from the first runs: a frontier model and Qwen3-1.7B on the held-out compounds,
both targets, next to the `formatted`, `exact` and `similarity` numbers.

## Changelog

- 2026-10-08: Initial v1 taskset.
