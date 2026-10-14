# hts-classify

Source: [environments/trade/hts_classify/README.md](../environments/trade/hts_classify/README.md). Not re-run.

`deepseek/deepseek-v4-flash`, no tools, default sampling, one rollout per task.

| Set | exact | hs6 | heading | formatted | valid_hs6 | mean reward | cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CROSS rulings, test, 197 usable | 10.7% | 53.3% | 66.0% | 97.5% | 91.4% | 0.492 | $0.67 |
| HSCodeComp listings, 632 | 13.1% | 50.3% | 66.0% | 97.6% | 94.0% | 0.471 | $1.97 |

Reward histogram on the rulings, from no digit right to all ten: 35, 32, 25, 38, 46, 21. 67 of 197 replies had the first eight digits right, and 46 of those missed only the two-digit statistical suffix.

Published references (not runs of this repository): on the same 200 rulings, GPT-5-Thinking 25.0% exact and 55.5% at six digits, a fine-tuned Llama-3.3-70B 40.0% and 57.5%; on the same listings, the best search-enabled agent 46.8% exact, human experts 95%.
