"""Summarize an eval run from its traces.jsonl.

    uv run python scripts/summarize.py                     # newest run under outputs/
    uv run python scripts/summarize.py outputs/<run-dir>   # a specific run

Prints outcome counts, mean reward, per-metric means, token usage and cost, then one line per
rollout with the gold answer, the number of model calls, the stop condition and the model's
final reply, so a zero reward can be traced to its cause (format, wrong answer, cut off, error)
without opening the JSON.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUTPUTS = Path("outputs")


def newest_run() -> Path:
    runs = [p.parent for p in OUTPUTS.rglob("traces.jsonl")]
    if not runs:
        raise SystemExit(f"no traces.jsonl under {OUTPUTS}/")
    return max(runs, key=lambda p: (p / "traces.jsonl").stat().st_mtime)


def load_traces(path: Path) -> list[dict]:
    path = path / "traces.jsonl" if path.is_dir() else path
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [trace for line in lines for trace in json.loads(line)["traces"]]


def reward_of(trace: dict) -> float:
    return sum(r["score"] * r["weight"] for r in trace["rewards"].values())


def assistant_messages(trace: dict) -> list[dict]:
    return [n["message"] for n in trace["nodes"] if n.get("sampled") and n["message"]["role"] == "assistant"]


def last_reply(trace: dict) -> str:
    messages = assistant_messages(trace)
    return (messages[-1].get("content") or "") if messages else ""


def main(argv: list[str]) -> None:
    run = Path(argv[1]) if len(argv) > 1 else newest_run()
    print(f"run: {run}")
    traces = load_traces(run)
    ok = [t for t in traces if t["ok"]]
    print(f"rollouts: {len(traces)}  ok: {len(ok)}  errored: {len(traces) - len(ok)}")
    print("stop conditions:", dict(Counter(t["stop_condition"] for t in traces)))
    if not ok:
        return

    print(f"mean reward: {sum(map(reward_of, ok)) / len(ok):.3f}")
    metrics: dict[str, list[float]] = defaultdict(list)
    for trace in ok:
        for name, value in trace["metrics"].items():
            metrics[name].append(value)
    for name, values in sorted(metrics.items()):
        print(f"mean {name}: {sum(values) / len(values):.3f}")

    usage = [call["usage"] for trace in ok for call in trace["calls"] if call.get("usage")]
    prompt_tokens = sum(u["prompt_tokens"] for u in usage)
    completion_tokens = sum(u["completion_tokens"] for u in usage)
    cost = sum(u.get("cost") or 0.0 for u in usage)
    print(
        f"model calls: {len(usage)}  prompt tokens: {prompt_tokens}  completion tokens: {completion_tokens}  cost: ${cost:.4f}"
    )

    print()
    print(f"{'task':>4}  {'reward':>6}  {'gold':>5}  {'calls':>5}  {'stop':<16}  reply (last 50 chars)")
    for trace in sorted(ok, key=lambda t: (t["task"]["data"].get("idx") or 0, t["id"])):
        data = trace["task"]["data"]
        calls = len(assistant_messages(trace))
        print(
            f"{data.get('idx', '?'):>4}  {reward_of(trace):>6.2f}  {str(data.get('answer', '')):>5}  {calls:>5}  "
            f"{trace['stop_condition']:<16}  {last_reply(trace)[-50:]!r}"
        )


if __name__ == "__main__":
    main(sys.argv)
