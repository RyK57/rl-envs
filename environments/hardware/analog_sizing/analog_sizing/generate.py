"""Generate sizing tasks: sample a gold sizing, measure it, and set the spec just below it.

Every task is feasible by construction, and the gold sizing stays hidden in the task data so
`validate` can prove feasibility without a model. Needs ngspice on this machine.

    uv run python -m analog_sizing.generate -n 300 --seed 0                               # data/train.jsonl
    uv run python -m analog_sizing.generate -n 100 --seed 1 -o <package>/data/eval.jsonl   # the held-out set
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path

from analog_sizing import sim

DATA_PATH = Path(__file__).parent / "data" / "train.jsonl"
# Log-uniform sampling ranges: the part of sim.BOUNDS where the topology behaves like an op-amp.
SAMPLE_RANGES = {
    "W1": (2e-6, 100e-6), "L1": (0.18e-6, 1e-6),
    "W3": (1e-6, 60e-6), "L3": (0.18e-6, 1e-6),
    "W5": (2e-6, 60e-6), "L5": (0.3e-6, 2e-6),
    "W6": (5e-6, 300e-6), "L6": (0.18e-6, 1e-6),
    "W7": (2e-6, 100e-6), "L7": (0.3e-6, 2e-6),
    "CC": (0.3e-12, 5e-12), "IB": (5e-6, 100e-6),
}  # fmt: skip


def sample_sizes(rng: random.Random) -> dict[str, float]:
    return {key: math.exp(rng.uniform(math.log(low), math.log(high))) for key, (low, high) in SAMPLE_RANGES.items()}


def round_sig(value: float, digits: int, *, up: bool) -> float:
    quantum = 10 ** (math.floor(math.log10(abs(value))) - (digits - 1))
    return (math.ceil if up else math.floor)(value / quantum) * quantum


def sane(measurements: dict) -> bool:
    return bool(
        measurements["ok"]
        and 40 <= measurements["gain_db"] <= 110
        and measurements["ugf_hz"] >= 1e6
        and 45 <= measurements["pm_deg"] <= 89
        and measurements["idd_a"] <= 2e-3
    )


def derive_spec(measurements: dict, slack: float) -> dict[str, float]:
    return {
        "gain_db_min": float(math.floor(measurements["gain_db"] - 3.0)),
        "ugf_hz_min": round_sig(measurements["ugf_hz"] * (1 - slack), 2, up=False),
        "pm_deg_min": float(min(math.floor(measurements["pm_deg"] - 5.0), 75)),
        "idd_a_max": round_sig(measurements["idd_a"] * (1 + slack), 2, up=True),
        "area_um2_max": round_sig(measurements["area_um2"] * (1 + slack), 2, up=True),
    }


def generate(count: int, seed: int, slack: float) -> tuple[list[dict], int]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    attempts = 0
    with tempfile.TemporaryDirectory() as tmp:
        baseline = sim.simulate(tmp, sim.DEFAULT_SIZES)
        while len(tasks) < count:
            attempts += 1
            gold = sample_sizes(rng)
            measurements = sim.simulate(tmp, gold)
            if not sane(measurements):
                continue
            spec = derive_spec(measurements, slack)
            # The gold must meet its own spec and the starting sizes must not.
            if not sim.score(measurements, spec)["solved"] or sim.score(baseline, spec)["solved"]:
                continue
            del measurements["log"]
            tasks.append(
                {
                    "name": f"opamp2-s{seed}-{len(tasks):05d}",
                    "spec": spec,
                    "gold": gold,
                    "gold_measurements": measurements,
                }
            )
    return tasks, attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("-n", "--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--slack",
        type=float,
        default=0.2,
        help="spec margin below the gold's measurements",
    )
    parser.add_argument("-o", "--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    tasks, attempts = generate(args.count, args.seed, args.slack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for task in tasks:
            out.write(json.dumps(task) + "\n")
    print(f"wrote {len(tasks)} tasks to {args.output} ({attempts} samples drawn)")


if __name__ == "__main__":
    main()
