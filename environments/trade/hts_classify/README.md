# hts-classify

Classify a product for import into the United States to its 10-digit Harmonized Tariff Schedule
code. Single-turn, no tools by default. Every imported shipment needs this code; brokers and
importers assign it by hand, and a wrong code means the wrong duty, penalties, or a held shipment.

## Taskset

Two sources, both real and both licensed for this use:

- **`cross`**: US customs rulings from CBP's Rulings Online Search System, as packaged by
  Flexify.AI in the CROSS Rulings HTS Dataset for Tariff Classification (Apache-2.0). Each row is
  the product a ruling described and the code the ruling assigned. Splits: 18,254 train, 200
  validation, 200 test. Published reference points on the test split: GPT-5-Thinking 25.0% exact
  and 55.5% at six digits; a fine-tuned Llama-3.3-70B 40.0% and 57.5%.
- **`hscodecomp`**: 632 real e-commerce listings labeled by customs experts with over 98%
  agreement, from Alibaba's HSCodeComp (Apache-2.0). Test only, never trained on. The best
  published agent scores 46.8% exact; human experts 95%.

The HS nomenclature (6,900 headings, CC-BY-SA-4.0) backs the validity metric and, with the
`rulebook` knob, is written into the box for a harness with a shell to search.

Attribution required by the ruling dataset's licence: "CROSS Rulings HTS Dataset for Tariff
Classification by Flexify.AI Inc. (https://www.flexify.ai)",
https://huggingface.co/datasets/flexifyai/cross_rulings_hts_dataset_for_tariffs, used
unmodified except for parsing each row into a description and a code.

Task keys: `hts:<source>:<split>:<row>`. Gold codes stay off `TaskData` and are looked up by key
at scoring time.

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.source` | `cross` | `cross` rulings or `hscodecomp` listings |
| `--env.taskset.split` | `validation` | `train`, `validation` or `test` (`hscodecomp` has `test` only) |
| `--env.taskset.task.rulebook` | false | write the HS nomenclature into the box and say so in the prompt |

## Signals

- **Reward** `hierarchical` (weight 1.0): leading digits shared with the gold code, over ten. The
  right chapter earns 0.2, the right heading 0.4, the right six-digit subheading 0.6, the right
  US code 1.0. A reply without a ten-digit code in `\boxed{}` scores 0.
- **Metric** `exact`: all ten digits right.
- **Metric** `hs6`: the six-digit subheading right, the number comparable across countries.
- **Metric** `heading`: the four-digit heading right.
- **Metric** `formatted`: a ten-digit code was given in the required form.
- **Metric** `valid_hs6`: the predicted subheading exists in the nomenclature, right or wrong.

## Run

```bash
uv run validate hts-classify --runtime.type subprocess -n 20   # gold codes well formed, subheadings exist
uv run --env-file .env eval @ configs/hts_classify.toml --no-rich
```

## Changelog

- 2026-09-11: Initial v1 taskset.
