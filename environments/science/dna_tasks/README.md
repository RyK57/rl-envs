# dna-tasks

Translate a coding DNA sequence, write its reverse complement, find and translate its longest open
reading frame, or give its GC content. Single-turn, no tools by default. These are the first four
exercises of any bioinformatics course and the primitives behind every sequence pipeline; each has
one right answer that a few lines of code compute, so the gold is a function of the prompt and the
taskset needs no data at all. It is the catalog's first infinite, procedural taskset.

## Taskset

- **Source:** none. Every row is a random sequence generated from the seed, the operation, the
  length and the row index, so the same config always yields the same rows and a run bounds the
  taskset with `-n`. `translate` rows are complete coding sequences (ATG, coding codons, one stop);
  `orf` rows plant one complete ATG-to-stop frame of random length in random flanks; `gc_content`
  rows draw each base under a random GC bias between 25 and 75%; `reverse_complement` rows are
  uniform.
- **Gold:** the standard genetic code (translation stops at the first stop codon, which is not
  written), Watson-Crick complementarity, the longest complete forward-strand ORF (earliest when
  tied), or the GC percentage to one decimal place. Recomputed from the sequence at scoring time,
  never stored.
- **Pins:** none to pin; the generator is the data.

Task keys: `dna:<operation>:<length>:<seed>:<idx>`.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.operation` | `translate` | `translate`, `reverse_complement`, `orf` or `gc_content`; every row of a run asks the same one |
| `--env.taskset.length` | 60 | sequence length in bases (12 to 3000; a coding sequence is rounded down to whole codons); the difficulty knob |
| `--env.taskset.seed` | 0 | generator seed; eval sources use another seed than training |

## Signals

- **Reward** `correct` (weight 1.0). A sequence answer (protein or DNA) scores 1.0 when it is the
  gold and otherwise its sequence identity to the gold (`difflib` ratio, so a single wrong residue
  in a 19-residue protein scores about 0.95 and a frame shift much less). A GC percentage scores
  1.0 within 0.05 points, then decays linearly to 0 at ten points off. A reply without one boxed
  token scores 0. Sequence answers are upper-cased and a trailing `%` is dropped before scoring.
- **Metric** `exact`: the answer is the gold.
- **Metric** `formatted`: one boxed token was given.
- **Metric** `length_error`: how many characters the answer is off in length (a missing answer
  counts the whole gold; 0 for `gc_content`).

## Run

```bash
uv run validate dna-tasks --runtime.type subprocess -n 50          # clean sequences, complete frames, gold scores 1.0
uv run --env-file .env eval @ configs/dna_tasks.toml --no-rich
uv run --env-file .env eval @ configs/dna_tasks.toml --env.taskset.operation orf --env.taskset.length 90 --no-rich
```

## Validation

Offline: `uv run validate dna-tasks --runtime.type subprocess -n 50` passes for each operation (every sequence is ACGT, a
coding sequence starts with ATG and ends in a stop with the right protein length, an ORF row has an
ORF, and the gold of every row, written as instructed, scores 1.0 through the reward); the same seed
yields the same rows across runs and another seed different ones.

## Baseline

To be recorded from the first runs: a frontier model and Qwen3-1.7B on 200 sequences of each
operation at 60 and 150 bases, next to the `exact`, `formatted` and `length_error` numbers.

## Changelog

- 2026-10-08: Initial v1 taskset.
