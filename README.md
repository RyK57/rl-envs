# rl-envs

RL environments for LLM post-training, from [Reinforcement Learning at Berkeley](https://rlatberkeley.com) (RL@B). Each environment is a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 taskset: a scored task a lab can evaluate and train on.

Industry environments are grouped by domain. Small procedural environments live under `environments/toy/` and are separate from the industry set. Recorded baselines are in [`results/`](results/README.md). No environment has a training run yet.

## Environments

| Environment | Domain | Task | Status |
| --- | --- | --- | --- |
| [count-letters](environments/toy/count_letters/) | toy | Count a letter in a short passage. | [0.633](results/count-letters.md) without tools, 0.933 with a shell |
| [number-guess](environments/toy/number_guess/) | toy | Find a secret number from higher/lower feedback. | [0.900](results/number-guess.md) at 1..5000 / 13 guesses |
| [guess-golf](environments/toy/guess_golf/) | toy | Four number-guess games; fewest guesses wins. | [1.14 / 1.5](results/guess-golf.md), 36/40 games solved |
| [gsm8k](environments/math/gsm8k/) | math | Grade-school word problems, answer in `\boxed{}`. | [0.933](results/gsm8k.md) on 10×3 |
| [pyfix](environments/code/pyfix/) | code | Fix one planted bug; hidden tests run in a sandbox. | [1.000](results/pyfix.md) (26/26), saturated |
| [summarize](environments/text/summarize/) | text | Two-sentence summary, scored by an LLM judge. | [0.80](results/summarize.md) after the rubric fix |
| [hts-classify](environments/trade/hts_classify/) | trade | 10-digit US tariff code for a product. | [0.492](results/hts-classify.md) CROSS / 0.471 listings |
| [sanctions-screening](environments/banking/sanctions_screening/) | banking | Same person or organization, on sanctions-list pairs. | [pending](results/sanctions-screening.md) |
| [contract-review](environments/legal/contract_review/) | legal | Classify a clause, or extract one, on CUAD. | [pending](results/contract-review.md) |
| [commercial-underwriting](environments/insurance/commercial_underwriting/) | insurance | Limits, deductibles, and NAICS for a small-business application. | [pending](results/commercial-underwriting.md) |
| [medical-coding](environments/healthcare/medical_coding/) | healthcare | ICD-10-CM codes for a clinical history. | [pending](results/medical-coding.md) |
| [patent-classification](environments/ip/patent_classification/) | ip | Main CPC symbol from a patent title and abstract. | [pending](results/patent-classification.md) |
| [procurement-coding](environments/government/procurement_coding/) | government | NAICS or PSC code for a federal contract action. | [pending](results/procurement-coding.md) |
| [injury-coding](environments/safety/injury_coding/) | safety | OIICS codes for an OSHA injury narrative. | [pending](results/injury-coding.md) |
| [ghs-hazards](environments/science/ghs_hazards/) | science | GHS hazard statements of a substance from its SMILES. | [pending](results/ghs-hazards.md) |
| [iupac-structure](environments/science/iupac_structure/) | science | SMILES, or the formula, of a compound from its IUPAC name. | [pending](results/iupac-structure.md) |
| [mol-properties](environments/science/mol_properties/) | science | A computable property of a molecule from its SMILES. | [pending](results/mol-properties.md) |
| [dna-tasks](environments/science/dna_tasks/) | science | Translate, reverse-complement, ORF or GC content of generated DNA. | [pending](results/dna-tasks.md) |
| [reaction-prediction](environments/science/reaction_prediction/) | science | Product of an organic reaction from its reactants, on USPTO-50K. | [pending](results/reaction-prediction.md) |

Baselines above are inference evals, mostly `deepseek/deepseek-v4-flash`. They were copied from the environment READMEs and were not re-run for this catalog.

## Quickstart

```bash
uv sync
cp .env.example .env          # PRIME_API_KEY, or another OpenAI-compatible key
uv run pytest
uv run eval @ configs/hts_classify.toml --dry-run --no-rich
uv run --env-file .env eval @ configs/hts_classify.toml --no-rich -v
```

Smoke configs already set a local runtime (`subprocess` or `docker`) and `push = false`. `uv run validate <taskset-id> --runtime.type subprocess` checks rows without a model. Summarize a finished run with `uv run python scripts/summarize.py outputs/<run-dir>`.

## Layout

| Path | What it is |
| --- | --- |
| `environments/<domain>/<name>/` | One installable package per environment. `toy/` is the procedural set. |
| `configs/*.toml` | Eval configs. `configs/train/` is prime-rl, with algorithm overlays in `configs/train/algo/`. |
| `results/` | Recorded baselines. Pending where no eval has been written down. |
| `tests/` | Offline scoring tests. No model calls. |
| `scripts/` | Summarize a run, compare training runs, compare judges, print dataset revisions. |
| `dashboard/` | Local viewer over `outputs/`. |
| `rl_losses/` | Custom policy-gradient losses for prime-rl. |

## License

Code in this repository is [MIT](LICENSE), copyright 2026 Reinforcement Learning at Berkeley.

Datasets keep their own licenses, stated in each environment README. Two are non-commercial: [sanctions-screening](environments/banking/sanctions_screening/) (OpenSanctions pairs, CC-BY-NC) and [patent-classification](environments/ip/patent_classification/) (HUPD, CC-BY-NC-SA).
