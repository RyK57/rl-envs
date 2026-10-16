"""Simulate the two-stage op-amp for one set of device sizes and report what it measures.

Standalone on purpose: this file is copied verbatim into the solver sandbox and into the
grading sandbox, where only python3 and ngspice exist. Keep it stdlib-only.

Inside a workspace, `python3 sim.py` reads `sizes.json`, writes `opamp.sp` and `sizes.inc`,
runs ngspice, and prints the measurements as JSON. When `spec.json` is present it also prints
each spec line's satisfaction ratio and the resulting score, which is exactly what grading
computes.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

PARAMS = ("W1", "L1", "W3", "L3", "W5", "L5", "W6", "L6", "W7", "L7", "CC", "IB")
BOUNDS = {
    "W": (0.5e-6, 500e-6),
    "L": (0.18e-6, 5e-6),
    "CC": (0.1e-12, 20e-12),
    "IB": (1e-6, 500e-6),
}
# The sizes every workspace starts from. Tasks whose spec these already meet are discarded
# at generation time, so doing nothing never scores 1.0.
DEFAULT_SIZES = {
    "W1": 20e-6, "L1": 0.5e-6, "W3": 10e-6, "L3": 0.5e-6, "W5": 10e-6, "L5": 1e-6,
    "W6": 60e-6, "L6": 0.5e-6, "W7": 20e-6, "L7": 1e-6, "CC": 1e-12, "IB": 20e-6,
}  # fmt: skip
SPEC_KEYS = ("gain_db_min", "ugf_hz_min", "pm_deg_min", "idd_a_max", "area_um2_max")

# Level-1 MOSFET models keep the simulation fast, deterministic, and free of PDK downloads.
# Swapping in a real PDK (e.g. SKY130) only changes the .model lines and the include path.
NETLIST = """\
* Two-stage Miller-compensated CMOS op-amp: open-loop AC response with DC unity feedback.
* Device sizes come from sizes.inc, rendered from sizes.json by sim.py.
.include sizes.inc
.param VDD=1.8 VCM=0.9 CL=2p

.model nch NMOS (LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.1 GAMMA=0.4 PHI=0.7 TOX=4n CGSO=0.3n CGDO=0.3n)
.model pch PMOS (LEVEL=1 VTO=-0.5 KP=80u LAMBDA=0.15 GAMMA=0.5 PHI=0.7 TOX=4n CGSO=0.3n CGDO=0.3n)

Vdd vdd 0 {VDD}
Vinp inp 0 dc {VCM} ac 1
* DC unity-gain feedback through an ideal inductor; the ideal capacitor makes inn an AC ground.
Lfb out inn 1G
Cfb inn 0 1G

Iref vdd nb {IB}
M8 nb nb 0 0 nch W={W5} L={L5}
M5 s1 nb 0 0 nch W={W5} L={L5}
M1 d1 inn s1 0 nch W={W1} L={L1}
M2 o1 inp s1 0 nch W={W1} L={L1}
M3 d1 d1 vdd vdd pch W={W3} L={L3}
M4 o1 d1 vdd vdd pch W={W3} L={L3}
M6 out o1 vdd vdd pch W={W6} L={L6}
M7 out nb 0 0 nch W={W7} L={L7}
Cc o1 out {CC}
CL out 0 {CL}

.control
set filetype=ascii
set units=degrees
op
print i(vdd)
ac dec 20 1 10G
meas ac dcgain find vdb(out) at=1
meas ac ugf when vdb(out)=0 cross=1
meas ac pm_raw find vp(out) when vdb(out)=0 cross=1
quit
.endc
.end
"""

_NUMBER = r"([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)"
_MEASURE = re.compile(r"^(dcgain|ugf|pm_raw)\s*=\s*" + _NUMBER, re.MULTILINE)
_SUPPLY = re.compile(r"^i\(vdd\)\s*=\s*" + _NUMBER, re.MULTILINE)


def check_sizes(raw: object) -> dict[str, float]:
    """The twelve sizes as floats, or a TypeError/ValueError naming the first problem."""
    if not isinstance(raw, dict):
        raise TypeError("sizes must be a JSON object")
    sizes = {}
    for key in PARAMS:
        if key not in raw:
            raise ValueError(f"missing {key}")
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} is not a number") from None
        if not math.isfinite(value):
            raise ValueError(f"{key} is not finite")
        low, high = BOUNDS[key if key in BOUNDS else key[0]]
        if not low <= value <= high:
            raise ValueError(f"{key}={value:g} is outside [{low:g}, {high:g}]")
        sizes[key] = value
    return sizes


def area_um2(sizes: dict[str, float]) -> float:
    """Active gate area: M1/M2, M3/M4, and M5/M8 are matched pairs."""
    devices = (
        ("W1", "L1", 2),
        ("W3", "L3", 2),
        ("W5", "L5", 2),
        ("W6", "L6", 1),
        ("W7", "L7", 1),
    )
    return sum(count * sizes[w] * sizes[l] for w, l, count in devices) * 1e12


def render_sizes(sizes: dict[str, float]) -> str:
    return ".param " + " ".join(f"{key}={sizes[key]:.6g}" for key in PARAMS) + "\n"


def parse(output: str) -> dict:
    """Measurements out of ngspice's batch output; anything unmeasured is None."""
    values = {name: float(text) for name, text in _MEASURE.findall(output)}
    supply = _SUPPLY.search(output)
    measurements: dict = {
        "gain_db": values.get("dcgain"),
        "ugf_hz": values.get("ugf"),
        "pm_deg": None,
        "idd_a": abs(float(supply.group(1))) if supply else None,
    }
    if "pm_raw" in values:
        # Non-inverting drive: phase starts at 0 and falls, so the margin is 180 + phase at
        # the unity-gain crossing. ngspice wraps phase into (-180, 180].
        margin = 180.0 + values["pm_raw"]
        measurements["pm_deg"] = margin - 360.0 if margin > 180.0 else margin
    measurements["ok"] = all(value is not None for value in measurements.values())
    return measurements


def simulate(
    workdir: str | Path,
    sizes: dict[str, float],
    ngspice: str = "ngspice",
    timeout: float = 60,
) -> dict:
    """Write the netlist for `sizes` into `workdir`, run ngspice there, and parse the result."""
    workdir = Path(workdir)
    (workdir / "opamp.sp").write_text(NETLIST)
    (workdir / "sizes.inc").write_text(render_sizes(sizes))
    try:
        proc = subprocess.run(
            [ngspice, "-b", "opamp.sp"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        output = f"ngspice timed out after {timeout:g}s"
    measurements = parse(output)
    measurements["area_um2"] = area_um2(sizes)
    measurements["log"] = output[-2000:]
    return measurements


def score(measurements: dict, spec: dict[str, float]) -> dict:
    """Per-spec satisfaction ratios in [0, 1], the score, and whether every line is met.

    A lower bound scores value/bound and an upper bound scores bound/value, both clipped at 1,
    so overshooting a spec earns nothing extra and an unmeasurable quantity scores 0. The
    score is the mean performance ratio (gain, unity-gain frequency, phase margin) times the
    mean budget ratio (current, area): a circuit that does not amplify scores 0 no matter how
    small and cheap it is, and the score is 1 iff every line is met.
    """

    def at_least(value: float | None, bound: float) -> float:
        if value is None:
            return 0.0
        return 1.0 if value >= bound else max(0.0, value / bound)

    def at_most(value: float | None, bound: float) -> float:
        if value is None:
            return 0.0
        return 1.0 if value <= bound else bound / value

    ratios = {
        "gain_db": at_least(measurements.get("gain_db"), spec["gain_db_min"]),
        "ugf_hz": at_least(measurements.get("ugf_hz"), spec["ugf_hz_min"]),
        "pm_deg": at_least(measurements.get("pm_deg"), spec["pm_deg_min"]),
        "idd_a": at_most(measurements.get("idd_a"), spec["idd_a_max"]),
        "area_um2": at_most(measurements.get("area_um2"), spec["area_um2_max"]),
    }
    performance = (ratios["gain_db"] + ratios["ugf_hz"] + ratios["pm_deg"]) / 3
    budget = (ratios["idd_a"] + ratios["area_um2"]) / 2
    return {
        "ratios": ratios,
        "score": performance * budget,
        "solved": all(ratio >= 1.0 for ratio in ratios.values()),
    }


def main() -> int:
    here = Path.cwd()
    try:
        sizes = check_sizes(json.loads((here / "sizes.json").read_text()))
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"sizes.json: {exc}"}, indent=2))
        return 1
    measurements = simulate(here, sizes)
    if measurements["ok"]:
        del measurements["log"]
    spec_path = here / "spec.json"
    if spec_path.exists():
        measurements["spec"] = score(measurements, json.loads(spec_path.read_text()))
    print(json.dumps(measurements, indent=2))
    return 0 if measurements["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
