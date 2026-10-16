# Results

Baselines copied from the environment READMEs on 2026-10-14. Nothing here was re-run, and no number was added that those READMEs do not already state. Per-environment notes are linked below.

No environment has a training run. There is no before/after comparison.

| Environment | Status | Headline |
| --- | --- | --- |
| [count-letters](count-letters.md) | recorded | 0.633 without tools, 0.933 with a shell (`deepseek-v4-flash`, 10×3) |
| [number-guess](number-guess.md) | recorded | 0.900 at 1..5000 / 13 guesses; 1.000 at 1..100 / 7 |
| [guess-golf](guess-golf.md) | recorded | mean reward 1.14 of 1.5; 36 of 40 games solved |
| [gsm8k](gsm8k.md) | recorded | 0.933 on 10×3 |
| [pyfix](pyfix.md) | recorded | 1.000 (26/26), saturated |
| [summarize](summarize.md) | recorded | 0.80 after the rubric fix (was 0.76) |
| [hts-classify](hts-classify.md) | recorded | CROSS test reward 0.492 (10.7% exact); listings reward 0.471 (13.1% exact) |
| [sanctions-screening](sanctions-screening.md) | pending | no model baseline |
| [contract-review](contract-review.md) | pending | no model baseline |
| [commercial-underwriting](commercial-underwriting.md) | pending | no model baseline |
| [medical-coding](medical-coding.md) | pending | no model baseline |
| [patent-classification](patent-classification.md) | pending | no model baseline |
| [procurement-coding](procurement-coding.md) | pending | no model baseline |
| [injury-coding](injury-coding.md) | pending | no model baseline |
| [ghs-hazards](ghs-hazards.md) | pending | no model baseline |
| [iupac-structure](iupac-structure.md) | pending | no model baseline |
| [mol-properties](mol-properties.md) | pending | no model baseline |
| [dna-tasks](dna-tasks.md) | pending | no model baseline |
| [reaction-prediction](reaction-prediction.md) | pending | no model baseline |
| [analog-sizing](analog-sizing.md) | pending | no model baseline |

Where a README quotes an outside paper or leaderboard, that figure is labeled a published reference. It is not a run of this repository.
