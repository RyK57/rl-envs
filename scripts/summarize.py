"""Summarize an eval run from its traces.jsonl.

    uv run python scripts/summarize.py                     # newest run under outputs/
    uv run python scripts/summarize.py outputs/<run-dir>   # a specific run

Prints outcome counts, mean reward, per-metric means, token usage and cost, then one line per
rollout with the gold answer, the number of model calls, the stop condition and the model's
final reply, so a zero reward can be traced to its cause (format, wrong answer, cut off, error)
without opening the JSON. A replay run scored with `-r N` keeps the source trace ids, so its
rows are grouped per rollout and rows whose rewards differ between re-scores are marked `*`:
those are the judge's noise, not the model's.
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


def gold(data: dict) -> str:
    return str(data.get("answer", data.get("secret", data.get("name", ""))))[:14]


def by_rollout(ok: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for trace in ok:
        groups[trace["id"]].append(trace)
    return sorted(groups.values(), key=lambda g: (g[0]["task"]["data"].get("idx") or 0, g[0]["id"]))


def print_rollouts(ok: list[dict]) -> None:
    print(f"{'task':>4}  {'reward':>6}  {'gold':>14}  {'calls':>5}  {'stop':<16}  reply (last 50 chars)")
    for trace in sorted(ok, key=lambda t: (t["task"]["data"].get("idx") or 0, t["id"])):
        data = trace["task"]["data"]
        print(
            f"{data.get('idx', '?'):>4}  {reward_of(trace):>6.2f}  {gold(data):>14}  {len(assistant_messages(trace)):>5}  "
            f"{trace['stop_condition']:<16}  {last_reply(trace)[-50:]!r}"
        )


def print_rescores(groups: list[list[dict]]) -> None:
    noisy = {g[0]["id"] for g in groups if len({round(reward_of(t), 3) for t in g}) > 1}
    print(
        f"re-scores per rollout: {max(map(len, groups))}  "
        f"rollouts scored differently by the same judge: {len(noisy)}/{len(groups)}"
    )
    print(f"{'task':>4}  {'gold':>14}    {'rewards per re-score':<22}  reply (last 40 chars)")
    for group in groups:
        data = group[0]["task"]["data"]
        rewards = " ".join(f"{reward_of(t):.2f}" for t in group)
        mark = "*" if group[0]["id"] in noisy else " "
        print(f"{data.get('idx', '?'):>4}  {gold(data):>14}  {mark} {rewards:<22}  {last_reply(group[0])[-40:]!r}")


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
    groups = by_rollout(ok)
    if len(groups) < len(ok):
        print_rescores(groups)
    else:
        print_rollouts(ok)


if __name__ == "__main__":
    main(sys.argv)
