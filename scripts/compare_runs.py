"""Compare prime-rl runs, one per algorithm, from the metrics each wrote.

    uv run python scripts/compare_runs.py outputs/grpo outputs/gspo outputs/max_rl

Each argument is a prime-rl output directory (a run started with `[monitors.file]` in its config)
or the `metrics.jsonl` inside it. Prints the train reward at every logged step and every eval score
at the steps it was logged, one column per run, then a summary line per run with its first and last
eval scores. Keys are found by pattern rather than by name: a key ending in `/reward/mean` under
`train/` is the train reward, and a key with `avg@` under `eval/` is an eval score, so the script
survives renamed environments and agents.
"""

import json
import re
import sys
from pathlib import Path

TRAIN_KEY = re.compile(r"^train/[^/]+/effective/[^/]+/reward/mean$")
TRAIN_FALLBACK = re.compile(r"^train/.*reward/mean$")
EVAL_KEY = re.compile(r"^eval/(?P<name>[^/]+)/.*/avg@\d+$")


def metrics_path(arg: str) -> Path:
    path = Path(arg)
    if path.is_file():
        return path
    for candidate in (path / "monitors" / "file" / "metrics.jsonl", path / "metrics.jsonl"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no metrics.jsonl under {path}; was the run started with [monitors.file]?")


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("step") is not None]


def train_curve(rows: list[dict]) -> dict[int, float]:
    keys = {k for row in rows for k in row if TRAIN_KEY.match(k)} or {
        k for row in rows for k in row if TRAIN_FALLBACK.match(k)
    }
    curve: dict[int, list[float]] = {}
    for row in rows:
        values = [row[k] for k in keys if k in row]
        if values:
            curve.setdefault(row["step"], []).extend(values)
    return {step: sum(v) / len(v) for step, v in sorted(curve.items())}


def eval_curves(rows: list[dict]) -> dict[str, dict[int, float]]:
    curves: dict[str, dict[int, float]] = {}
    for row in rows:
        for key, value in row.items():
            match = EVAL_KEY.match(key)
            if match and "/effective/" not in key:
                curves.setdefault(match.group("name"), {})[row["step"]] = value
    return curves


def print_table(title: str, curves: dict[str, dict[int, float]]) -> None:
    steps = sorted({step for curve in curves.values() for step in curve})
    if not steps:
        return
    names = list(curves)
    width = max(12, *(len(n) for n in names))
    print(f"\n{title}")
    print("step".ljust(6) + "".join(n.rjust(width) for n in names))
    for step in steps:
        cells = [f"{curves[n][step]:.3f}" if step in curves[n] else "" for n in names]
        print(str(step).ljust(6) + "".join(c.rjust(width) for c in cells))


def main(args: list[str]) -> None:
    if not args:
        raise SystemExit(__doc__)
    runs = {Path(a).name if not Path(a).is_file() else Path(a).parent.name: load_rows(metrics_path(a)) for a in args}
    print_table("train reward (mean of the trained batch)", {name: train_curve(rows) for name, rows in runs.items()})
    evals: dict[str, dict[str, dict[int, float]]] = {}
    for name, rows in runs.items():
        for eval_name, curve in eval_curves(rows).items():
            evals.setdefault(eval_name, {})[name] = curve
    for eval_name, curves in evals.items():
        print_table(f"eval {eval_name}", curves)
    print("\nsummary (first -> last eval score per set)")
    for name in runs:
        parts = []
        for eval_name, curves in evals.items():
            curve = curves.get(name) or {}
            if curve:
                first, last = curve[min(curve)], curve[max(curve)]
                parts.append(f"{eval_name} {first:.3f} -> {last:.3f}")
        print(f"  {name}: " + ("; ".join(parts) if parts else "no eval rows yet"))


if __name__ == "__main__":
    main(sys.argv[1:])
