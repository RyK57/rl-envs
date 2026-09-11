# commercial-underwriting

The decisions a small-commercial underwriter makes on an application, against published
guidelines. Single-turn, no tools. Three task types:

- **`limits`**: the per-occurrence and aggregate policy limits for the line of business,
- **`deductibles`**: the deductible for the line of business,
- **`classification`**: the company's six-digit NAICS code from its description.

## Taskset

Two sources.

**`snorkel`**, the expert-verified cases: Snorkel AI's Multi-Turn Insurance Underwriting benchmark
(Apache-2.0), where a Chartered Property Casualty Underwriter validated each company, task and
reference answer. Both releases are read (`snorkelai/Multi-Turn-Insurance-Underwriting`, 380
conversations, and `...-Code-Gen`, 1,800) and reduced to one case per company-task id, keeping
the application facts (description, state, revenue, employees, payroll, vehicles, building
construction, line of business) and the reference answer. The benchmark's six task types split
into two groups:

- Decidable from the application plus the guidelines the benchmark publishes: policy limits,
  deductibles, business classification. These are the tasks here.
- Needing the carrier's own tables, which the public benchmark does not ship: appetite checks
  (an appetite matrix by NAICS, state and line), small-business eligibility (a size-standard table
  by NAICS) and product recommendations (the appetite matrix again). Those are left out, and they
  are exactly the tables a carrier would bring to a private version of this environment.

Splits by hash of the case id: one in five to `test`, one in five to `validation`, the rest to
`train`. The expert set is small (low hundreds of cases), which is why the second source exists.

**`synthetic`**, rule-generated training cases: the guidelines in `rulebook.py` are the ones the
experts decided under, verbatim from the benchmark. `limits_for` and `deductible_for` encode them
(standard $1M/$2M, $3M/$5M for cyber in telecoms, data processing, finance, ambulatory health,
specialty hospitals, nursing care and public administration; $500 deductible, $1,000 for auto,
$5,000 for auto in freight trucking, taxi and limousine, school and charter bus, scenic transport).
Sixty industries with descriptions, a seeded generator for names, locations and financials, and
the special sectors and lines drawn half the time so the answers stay mixed. The expert cases are
never generated from, so training on `synthetic` and testing on `snorkel` measures transfer from
the rulebook to expert-written applications.

The prompt for `limits` and `deductibles` carries the guidelines in full; the model has to infer
the sector from the description, as the experts' assistant did. `classification` gets the
application alone.

Task keys: `underwriting:<source>:<case id>`. Gold stays off `TaskData` and is looked up at
scoring time. Dataset revisions are unpinned until the first fetch; `uv run python
scripts/pin_revisions.py commercial_underwriting` prints the commits to pin in `taskset.py`.

Data facts: to be recorded after the first `validate` run (expert cases per task and split,
references that did not parse).

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.source` | `snorkel` | `snorkel` expert cases (all three tasks) or `synthetic` rule cases (limits, deductibles) |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |
| `--env.taskset.size` | 2000 | synthetic cases per split |
| `--env.taskset.seed` | 0 | synthetic generator seed |

## Signals

- **Reward** `answer` (weight 1.0). `limits`: both dollar amounts in `\boxed{}` match the
  reference, in any order and any dollar spelling. `deductibles`: the amount matches.
  `classification`: leading NAICS digits shared with the reference, over six: the sector (two
  digits) earns 0.33, the full code 1.0.
- **Metric** `exact`: the answer matches the reference completely.
- **Metric** `formatted`: an answer was given in the required form.

## Run

```bash
uv run validate commercial-underwriting --runtime.type subprocess -n 20
uv run validate commercial-underwriting --runtime.type subprocess -n 20 --taskset.source synthetic
uv run --env-file .env eval @ configs/commercial_underwriting.toml --no-rich
uv run --env-file .env eval @ configs/commercial_underwriting.toml --no-rich --env.taskset.split test
```

## Baseline

To be recorded: a frontier model and Qwen3-1.7B on the expert test cases and on 200 synthetic test
cases. Published reference: Snorkel's leaderboard scores frontier models on the full six-task
benchmark with tools, not directly comparable to this tool-less subset.

## Changelog

- 2026-09-11: Initial v1 taskset, expert and synthetic sources.
