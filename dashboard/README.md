# dashboard

A live observer over `outputs/`: every eval run with traces, its rewards and metrics per rollout,
each trace's full conversation (reasoning, tool calls, tool results), model calls with usage and
cost, timing, errors, and the run's `eval.log`. It re-reads the folder every 5 seconds, so a run
in progress fills in as it goes.

```bash
cd dashboard
npm install
npm run dev          # http://localhost:3000
```

It reads `../outputs` by default. Point it elsewhere with `OUTPUTS_DIR=/path/to/outputs npm run dev`.

Pages:

- `/` all runs: model, harness, runtime, rollouts, errors, mean reward, metric means, tokens, cost.
- `/runs/<run>` one run: summary tiles, the task × rollout grid (click a cell), every rollout with
  its reward breakdown, metrics, turns, tokens, cost, stop condition and last reply, plus the log.
- `/runs/<run>/traces/<id>` one trace: the conversation node by node, rewards, metrics, task data,
  model calls, timing, tools, agent config, raw JSON.
