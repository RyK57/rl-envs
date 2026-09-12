# injury-coding

Code a workplace injury narrative with the Occupational Injury and Illness Classification System
(OIICS): the nature of the injury, the part of body, the event that caused it, or its source.
Single-turn, no tools. OSHA codes every severe injury report this way, and workers' compensation
insurers code every claim the same way; reserving, rating and loss control key off the codes.

## Taskset

Source: OSHA's Severe Injury Reports, the export covering January 2015 to August 2025, downloaded
from osha.gov (US government data, public domain) into `~/.cache/huggingface/rl-envs/` on first
use. Every row is one report an employer had to file (a hospitalization, amputation or loss of an
eye): the narrative the employer wrote, the establishment's NAICS code, and the four OIICS codes
OSHA assigned with their titles. Columns are matched by name, so a re-export with different
capitalization still loads.

A report is kept for a field when it carries that field's code with a title and its narrative is
at least `min_chars` long. Splits by hash of the report id: one in ten to `test`, one in ten to
`validation`, the rest to `train`. The set of codes OSHA used for a field, across the whole
export, backs the validity metric.

Task keys: `oiics:<field>:<report id>`. Gold stays off `TaskData` and is looked up at scoring
time. The export is a fixed file rather than a pinned dataset revision; its URL is in
`taskset.py`.

Data facts: to be recorded after the first `validate` run in each field (reports kept per split,
dropped for short narratives or missing codes).

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.field` | `event` | `nature`, `part`, `event` or `source` |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` |
| `--env.taskset.min-chars` | 60 | shortest narrative kept |

## Signals

- **Reward** `hierarchical` (weight 1.0): leading digits shared with OSHA's code over its length,
  the OIICS division first: `4310` against `4312` scores 0.75. No code in `\boxed{}` scores 0.
- **Metric** `exact`: the full code.
- **Metric** `division`: the first digit.
- **Metric** `formatted`: a code was given in the required form.
- **Metric** `valid_code`: the code is one OSHA uses for the field, right or wrong.

## Run

```bash
uv run validate injury-coding --runtime.type subprocess -n 20
uv run validate injury-coding --runtime.type subprocess -n 20 --taskset.field nature
uv run --env-file .env eval @ configs/injury_coding.toml --no-rich
uv run --env-file .env eval @ configs/injury_coding.toml --no-rich --env.taskset.split test -n 500
```

## Baseline

To be recorded: a frontier model and Qwen3-1.7B on 500 held-out reports for the event code and
the nature code.

## Changelog

- 2026-09-12: Initial v1 taskset, four fields.
