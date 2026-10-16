"""analog-sizing — size a two-stage CMOS op-amp to a spec, graded by ngspice.

Each task is a specification (DC gain, unity-gain frequency, phase margin, supply current,
active area) for a fixed two-stage Miller-compensated op-amp. The agent edits `sizes.json` in
a sandbox that has ngspice and a helper (`sim.py`) that simulates the current sizes, so any
shell-capable harness can drive it. Scoring re-simulates the submitted sizes in a fresh
sandbox with pristine copies of the netlist, models, and helper: nothing the agent changed
in its own box is trusted, and the reward is the same formula `sim.py` prints.

Tasks come from `analog_sizing.generate`: sample a gold sizing, measure it, and set the spec
just below what it achieves, so every task is feasible and `validate` proves it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime

from analog_sizing import sim

DATA_DIR = Path(__file__).parent / "data"


def task_file(split: str, source: str | None) -> Path:
    return Path(source) if source else DATA_DIR / f"{split}.jsonl"


@lru_cache(maxsize=None)
def rows_in(path: Path) -> dict[str, dict]:
    """The generated tasks of one file, by name: the spec, the gold sizing and the gold's measurements."""
    with path.open(encoding="utf-8") as f:
        return {row["name"]: row for row in map(json.loads, f)}


SIM_SOURCE = Path(sim.__file__).read_text()
WORKDIR = "/workspace"
MAX_SIZES_BYTES = 64 * 1024

PROMPT = """\
Size the two-stage Miller-compensated CMOS op-amp in `opamp.sp` (current directory) so that \
it meets this specification at VDD = 1.8 V, CL = 2 pF, 27 C:

- DC gain >= {gain_db_min:.0f} dB
- unity-gain frequency >= {ugf_mhz_min:g} MHz
- phase margin >= {pm_deg_min:.0f} degrees
- supply current <= {idd_ua_max:g} uA
- active area (sum of W x L over all transistors) <= {area_um2_max:g} um^2

The topology, models, and testbench are fixed; only the device sizes are graded. They are read \
from `sizes.json` (keys W1, L1, W3, L3, W5, L5, W6, L6, W7, L7, CC, IB; SI units, e.g. 2e-05 \
for 20 um). Allowed ranges: W in [0.5 um, 500 um], L in [0.18 um, 5 um], CC in [0.1 pF, 20 pF], \
IB in [1 uA, 500 uA]. `python3 sim.py` simulates the sizes currently in `sizes.json` with \
ngspice and prints the measured values and each spec line's status. Write your final sizes to \
`sizes.json`."""


def render_prompt(spec: dict[str, float]) -> str:
    return PROMPT.format(
        gain_db_min=spec["gain_db_min"],
        ugf_mhz_min=spec["ugf_hz_min"] / 1e6,
        pm_deg_min=spec["pm_deg_min"],
        idd_ua_max=spec["idd_a_max"] * 1e6,
        area_um2_max=spec["area_um2_max"],
    )


def workspace_files(spec: dict[str, float], sizes: dict[str, float]) -> dict[str, str]:
    """Everything a box needs to simulate `sizes` against `spec`, relative to its workdir."""
    return {
        "sim.py": SIM_SOURCE,
        "spec.json": json.dumps(spec, indent=2) + "\n",
        "sizes.json": json.dumps(sizes, indent=2) + "\n",
        "opamp.sp": sim.NETLIST,
        "sizes.inc": sim.render_sizes(sizes),
    }


def grader_config(config: vf.RuntimeConfig, data: vf.TaskData) -> vf.RuntimeConfig:
    """The solver's runtime config pointed at the task image and workdir, for a fresh box."""
    fields = type(config).model_fields
    updates = {key: value for key, value in (("image", data.image), ("workdir", data.workdir)) if key in fields}
    return config.model_copy(update=updates)


class AnalogSizingData(vf.TaskData):
    spec: dict[str, float]
    split: str
    """The bundled task file the row comes from. The gold sizing stays in that file and is looked up by
    `name` at scoring time, so it never reaches a trace."""
    source: str | None = None
    """A task file given at load time in place of the bundled split."""


class AnalogSizingTask(vf.Task[AnalogSizingData]):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.name or self.hash

    @property
    def _gold(self) -> dict[str, float]:
        """A sizing that meets the spec; read from the task file, never from `TaskData`."""
        return rows_in(task_file(self.data.split, self.data.source))[self.data.name]["gold"]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for name, text in workspace_files(self.data.spec, sim.DEFAULT_SIZES).items():
            await runtime.write(name, text.encode())

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # Capture the submission before the solver box is torn down; scoring simulates it elsewhere.
        try:
            submission = await runtime.read("sizes.json", max_bytes=MAX_SIZES_BYTES)
        except vf.SandboxError as exc:
            trace.info["sizes_error"] = str(exc)
            return
        trace.info["sizes_json"] = submission.decode(errors="replace")

    async def validate(self, runtime: vf.Runtime) -> bool:
        measurements = await self._simulate(runtime, self._gold)
        return sim.score(measurements, self.data.spec)["solved"]

    @vf.reward(weight=1.0)
    async def spec_score(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Performance ratios times budget ratios (see `sim.score`), 1.0 iff every line is met."""
        submission = trace.info.get("sizes_json")
        if submission is None:  # finalize already recorded why there is no submission
            trace.record_metrics({"valid_sizes": 0.0, "solved": 0.0})
            return 0.0
        try:
            sizes = sim.check_sizes(json.loads(submission))
        except (TypeError, ValueError) as exc:
            trace.info["sizes_error"] = str(exc)
            trace.record_metrics({"valid_sizes": 0.0, "solved": 0.0})
            return 0.0
        measurements = await self._grade(runtime, sizes)
        result = sim.score(measurements, self.data.spec)
        trace.info["measurements"] = measurements
        trace.info["ratios"] = result["ratios"]
        metrics = {
            "valid_sizes": 1.0,
            "sim_ok": float(measurements["ok"]),
            "solved": float(result["solved"]),
            "area_um2": measurements["area_um2"],
        }
        for name, scale in (
            ("gain_db", 1.0),
            ("ugf_mhz", 1e-6),
            ("pm_deg", 1.0),
            ("idd_ua", 1e6),
        ):
            value = measurements[name.replace("_mhz", "_hz").replace("_ua", "_a")]
            if value is not None:
                metrics[name] = value * scale
        trace.record_metrics(metrics)
        return result["score"]

    async def _grade(self, runtime: vf.Runtime, sizes: dict[str, float]) -> dict:
        """Simulate `sizes` in a fresh box of the solver's provider so the agent's box is never trusted."""
        name = f"analog-sizing-grader-{uuid.uuid4().hex[:8]}"
        async with provision_runtime(grader_config(runtime.config, self.data), name=name) as grader:
            await grader.prepare_execution([])
            return await self._simulate(grader, sizes)

    async def _simulate(self, runtime: vf.Runtime, sizes: dict[str, float]) -> dict:
        for name, text in workspace_files(self.data.spec, sizes).items():
            await runtime.write(name, text.encode())
        result = await runtime.run(["python3", "sim.py"], {})
        return json.loads(result.stdout)


class AnalogSizingConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    """`train`: the 300 bundled tasks from seed 0. `eval`: 100 tasks from seed 1, never trained on."""
    source: str | None = None
    """JSONL of generated tasks, in place of the bundled split."""
    image: str = "analog-sizing:latest"
    """Container image with python3 and ngspice (build it from this package's Dockerfile)."""


class AnalogSizingTaskset(vf.Taskset[AnalogSizingTask, AnalogSizingConfig]):
    def load(self) -> Iterator[AnalogSizingTask]:
        c = self.config
        for idx, (name, row) in enumerate(rows_in(task_file(c.split, c.source)).items()):
            yield AnalogSizingTask(
                AnalogSizingData(
                    idx=idx,
                    name=name,
                    prompt=render_prompt(row["spec"]),
                    image=c.image,
                    workdir=WORKDIR,
                    spec=row["spec"],
                    split=c.split,
                    source=c.source,
                ),
                c.task,
            )
