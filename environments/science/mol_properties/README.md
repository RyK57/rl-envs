# mol-properties

Give a computable property of a molecule from its SMILES: the molecular formula, the average
molecular weight, the monoisotopic mass, or the number of heavy atoms, rings, aromatic rings or
stereocentres. Single-turn, no tools by default. Counting what is in a structure is the arithmetic
of chemistry, and every value here is a pure function of the structure, so the gold is never
stored: RDKit computes it inside the rollout runtime from the same SMILES the prompt shows. It is
the third leg of the chemistry trio: `iupac-structure` reads a name into a structure,
`ghs-hazards` classifies a structure, this one counts it.

## Taskset

- **Source:** the PubChem compounds of [`iupac-structure`](../iupac_structure/), through its
  cached 20,000-row table (CC0, the January 2026 PubChem dump as re-packaged by
  [`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC)).
  This package depends on `iupac-structure` for its reader, filters and hash splits, so the two
  tasksets share one download and one cache file.
- **Gold:** computed, not read. `props.py` (RDKit 2026.3.6) returns the Hill formula, the average
  molecular weight, the monoisotopic mass, the heavy-atom count, the ring count (smallest set of
  smallest rings), the aromatic ring count and the number of tetrahedral stereocentres, assigned or
  not, for the row's SMILES. One run per task, shared by every rollout and every hook; a failed run
  is forgotten so the next rollout retries it.
- **Pins:** `iupac-structure`'s `REVISION` and cache key; `props.py` pins its RDKit version.

Task keys: `molprop:<property>:<cid>`. `TaskData` carries the property, the split, the row index,
the PubChem CID and the SMILES cap the row was selected under; the structure is looked up by key at
scoring time.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.property` | `formula` | `formula`, `molecular_weight`, `monoisotopic_mass`, `heavy_atoms`, `rings`, `aromatic_rings` or `stereocenters`; every row of a run asks the same one |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |
| `--env.taskset.max-smiles-chars` | 60 | keep compounds whose canonical SMILES is at most this long; the difficulty knob |
| `--env.taskset.show-name` | false | show the IUPAC name next to the SMILES |

## Signals

- **Reward** `correct` (weight 1.0). `formula`: 1.0 for the gold Hill formula (subscript digits
  and a unicode minus are normalized), else 0. Counts: 1.0 for the exact number, else 0. Masses:
  1.0 within a tolerance (0.1 g/mol for the molecular weight, 0.005 Da for the monoisotopic mass),
  then a linear decay to 0 at five percent relative error, so a weight 1% off scores 0.8. A reply
  without one boxed token scores 0.
- **Metric** `formatted`: one boxed token was given.
- **Metric** `exact`: the value is right (within tolerance for a mass).
- **Metric** `abs_error`: the distance from the gold for a number, 0 or 1 for a formula, the gold
  itself when nothing parseable was given.
- `trace.info["gold"]` keeps the computed value for every scored rollout.

The calculator runs as a PEP 723 script through `runtime.run_uv_script`, installed once per runtime
in `setup()` while egress is open; it exits non-zero only on a malformed call or a SMILES RDKit
cannot read, so an infrastructure failure never scores as a wrong answer. Because the gold needs
the runtime, `replay` cannot re-score runs.

## Run

```bash
uv run validate mol-properties --runtime.type subprocess -n 20     # rows, structures, a finite value per property
uv run --env-file .env eval @ configs/mol_properties.toml --no-rich
uv run --env-file .env eval @ configs/mol_properties.toml --env.taskset.property molecular_weight --no-rich
```

## Validation

To be recorded on the first fetch, with `iupac-structure`'s pool: `uv run validate` per property
(every structure reads, every value is finite, every formula is a string).

## Baseline

To be recorded from the first runs: a frontier model and Qwen3-1.7B on the held-out compounds, per
property, next to the `formatted`, `exact` and `abs_error` numbers.

## Changelog

- 2026-10-08: Initial v1 taskset.
