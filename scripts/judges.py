"""Lay judge verdicts for the same rollouts side by side.

    uv run python scripts/judges.py <run-dir> [<run-dir> ...] [--all]

Replay copies keep the source trace id, so rollouts are matched across runs by id. For each
rollout the summary is printed once, then each run's verdicts (`info.verdict`: faithful, covered,
the quoted issue, the judge model), one entry per re-score. Only rollouts where the verdicts
differ are shown unless `--all`. Read the summary against its passage in `summarize/catalog.py`
to decide which judge is right; that is the calibration step every judge reward needs.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_traces(run: Path) -> list[dict]:
    lines = [line for line in (run / "traces.jsonl").read_text().splitlines() if line.strip()]
    return [trace for line in lines for trace in json.loads(line)["traces"] if trace["ok"]]


def last_reply(trace: dict) -> str:
    replies = [n["message"] for n in trace["nodes"] if n.get("sampled") and n["message"]["role"] == "assistant"]
    return (replies[-1].get("content") or "") if replies else ""


def reward_of(trace: dict) -> float:
    return sum(r["score"] * r["weight"] for r in trace["rewards"].values())


def verdict_line(trace: dict) -> str:
    verdict = trace["info"].get("verdict")
    if verdict is None:
        return f"reward {reward_of(trace):.2f}, no verdict recorded"
    issue = f" issue={verdict['issue']!r}" if verdict.get("issue") else ""
    return f"reward {reward_of(trace):.2f} faithful={verdict['faithful']} covered={verdict['covered']}{issue}"


def main(argv: list[str]) -> None:
    show_all = "--all" in argv
    runs = [Path(a) for a in argv[1:] if a != "--all"]
    if not runs:
        raise SystemExit(__doc__)
    by_run: list[dict[str, list[dict]]] = []
    for run in runs:
        groups: dict[str, list[dict]] = defaultdict(list)
        for trace in load_traces(run):
            groups[trace["id"]].append(trace)
        by_run.append(groups)
    for i, run in enumerate(runs):
        models = {t["info"].get("verdict", {}).get("model") for g in by_run[i].values() for t in g}
        print(f"run {i + 1}: {run}  judge: {', '.join(sorted(m for m in models if m)) or '?'}")

    first = sorted(by_run[0].values(), key=lambda g: (g[0]["task"]["data"].get("idx") or 0, g[0]["id"]))
    shown = 0
    for group in first:
        tid = group[0]["id"]
        entries = [t for groups in by_run for t in groups.get(tid, [])]
        differs = len({verdict_line(t) for t in entries}) > 1
        if not (show_all or differs):
            continue
        shown += 1
        data = group[0]["task"]["data"]
        print()
        print(f"== task {data.get('idx', '?')} {data.get('name', '')}  {'DIFFERS' if differs else 'agree'}")
        print(f"summary: {last_reply(group[0])}")
        for i, groups in enumerate(by_run):
            for trace in groups.get(tid, []):
                print(f"  run {i + 1}: {verdict_line(trace)}")
    print()
    print(f"rollouts shown: {shown} of {len(first)}")


if __name__ == "__main__":
    main(sys.argv)
