"""Offline checks for analog-sizing: the size rules, the score formula, the bundled specs and the loader;
with ngspice on this machine, also the simulator and the whole reward path through subprocess runtimes."""

import json
import shutil
from pathlib import Path

import pytest
import verifiers.v1 as vf
from analog_sizing import sim
from analog_sizing.taskset import AnalogSizingConfig, AnalogSizingTask, AnalogSizingTaskset, rows_in, task_file
from verifiers.v1.runtimes import SubprocessConfig, provision_runtime

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice is not installed")

SPEC = {"gain_db_min": 60.0, "ugf_hz_min": 10e6, "pm_deg_min": 60.0, "idd_a_max": 100e-6, "area_um2_max": 200.0}
MET = {"gain_db": 70.0, "ugf_hz": 20e6, "pm_deg": 65.0, "idd_a": 50e-6, "area_um2": 150.0}


def make_trace(task: AnalogSizingTask) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()), task=vf.TraceTask(type="AnalogSizingTask", data=task.data)
    )


@pytest.fixture
def tasks() -> list[AnalogSizingTask]:
    return list(AnalogSizingTaskset(AnalogSizingConfig()))


def test_check_sizes_accepts_the_defaults_and_names_the_first_problem():
    assert sim.check_sizes(sim.DEFAULT_SIZES) == sim.DEFAULT_SIZES
    with pytest.raises(TypeError):
        sim.check_sizes([20e-6, 0.5e-6])
    with pytest.raises(ValueError, match="missing CC"):
        sim.check_sizes({key: value for key, value in sim.DEFAULT_SIZES.items() if key != "CC"})
    with pytest.raises(ValueError, match="W1 is not a number"):
        sim.check_sizes({**sim.DEFAULT_SIZES, "W1": "wide"})
    with pytest.raises(ValueError, match="L1=1e-07 is outside"):
        sim.check_sizes({**sim.DEFAULT_SIZES, "L1": 1e-7})
    with pytest.raises(ValueError, match="IB is not finite"):
        sim.check_sizes({**sim.DEFAULT_SIZES, "IB": float("nan")})


def test_area_counts_the_matched_pairs_twice():
    # 2 x (20 x 0.5) + 2 x (10 x 0.5) + 2 x (10 x 1) + 60 x 0.5 + 20 x 1 = 100 um^2
    assert sim.area_um2(sim.DEFAULT_SIZES) == pytest.approx(100.0)


def test_parse_reads_the_measurements_and_wraps_the_phase():
    output = "i(vdd) = -8.03357e-05\ndcgain = 8.558056e+01\nugf = 5.000508e+07\npm_raw = -1.530544e+02\n"
    measurements = sim.parse(output)
    assert measurements["ok"]
    assert measurements["gain_db"] == pytest.approx(85.58056) and measurements["idd_a"] == pytest.approx(8.03357e-05)
    assert measurements["pm_deg"] == pytest.approx(26.9456)
    assert sim.parse("pm_raw = 1.700000e+02\n")["pm_deg"] == pytest.approx(-10.0)
    assert sim.parse("")["ok"] is False and sim.parse("")["gain_db"] is None


def test_score_is_one_iff_every_line_is_met_and_overshoot_earns_nothing():
    met = sim.score(MET, SPEC)
    assert met["solved"] and met["score"] == 1.0 and all(ratio == 1.0 for ratio in met["ratios"].values())
    assert sim.score({**MET, "gain_db": 700.0, "idd_a": 1e-9}, SPEC)["score"] == 1.0
    half_gain = sim.score({**MET, "gain_db": 30.0}, SPEC)
    assert not half_gain["solved"] and half_gain["ratios"]["gain_db"] == 0.5
    assert half_gain["score"] == pytest.approx(2.5 / 3)
    over_budget = sim.score({**MET, "idd_a": 200e-6}, SPEC)
    assert over_budget["ratios"]["idd_a"] == 0.5 and over_budget["score"] == pytest.approx(0.75)
    assert sim.score({**MET, "gain_db": None}, SPEC)["ratios"]["gain_db"] == 0.0


def test_a_circuit_that_does_not_amplify_scores_zero_however_cheap():
    dead = {"gain_db": 0.0, "ugf_hz": None, "pm_deg": None, "idd_a": 1e-9, "area_um2": 1.0}
    assert sim.score(dead, SPEC)["score"] == 0.0


def test_bundled_splits_are_well_formed_and_keep_the_gold_off_the_trace(tasks):
    held_out = list(AnalogSizingTaskset(AnalogSizingConfig(split="eval")))
    assert len(tasks) == 300 and len(held_out) == 100
    keys = [task.key for task in tasks + held_out]
    assert len(set(keys)) == 400
    assert all(key.startswith("opamp2-s0-") for key in keys[:300]) and all(
        key.startswith("opamp2-s1-") for key in keys[300:]
    )
    for task in tasks + held_out:
        data = task.data
        assert set(data.spec) == set(sim.SPEC_KEYS)
        assert data.image == "analog-sizing:latest" and data.workdir == "/workspace"
        assert f"{data.spec['gain_db_min']:.0f} dB" in data.prompt_text and "`sizes.json`" in data.prompt_text
        assert set(data.model_dump()).isdisjoint({"gold", "gold_measurements"}), "the solution stays off the trace"
        row = rows_in(task_file(data.split, None))[data.name]
        assert sim.check_sizes(row["gold"]) == task._gold
        assert sim.score(row["gold_measurements"], data.spec)["solved"], "the recorded gold meets its own spec"


async def test_a_missing_or_malformed_submission_scores_zero(tasks):
    task = tasks[0]
    trace = make_trace(task)
    assert await task.spec_score(trace, None) == 0.0
    trace.info["sizes_json"] = "not json"
    assert await task.spec_score(trace, None) == 0.0 and "sizes_error" in trace.info
    trace.info["sizes_json"] = json.dumps({**sim.DEFAULT_SIZES, "W1": 1.0})
    assert await task.spec_score(trace, None) == 0.0 and "outside" in trace.info["sizes_error"]


@needs_ngspice
def test_the_simulator_measures_the_default_sizes(tmp_path: Path):
    measurements = sim.simulate(tmp_path, sim.DEFAULT_SIZES)
    assert measurements["ok"]
    assert 40 < measurements["gain_db"] < 110 and measurements["ugf_hz"] > 1e6 and 0 < measurements["idd_a"] < 2e-3
    assert measurements["area_um2"] == pytest.approx(100.0)
    assert (tmp_path / "opamp.sp").is_file() and (tmp_path / "sizes.inc").is_file()


@needs_ngspice
def test_every_bundled_gold_meets_its_spec_and_the_start_does_not(tmp_path: Path):
    """What `validate` proves inside a box, on this machine: every task is feasible and none is solved at the start."""
    start = sim.simulate(tmp_path, sim.DEFAULT_SIZES)
    for split in ("train", "eval"):
        for task in AnalogSizingTaskset(AnalogSizingConfig(split=split)):
            assert sim.score(sim.simulate(tmp_path, task._gold), task.data.spec)["solved"], task.key
            assert not sim.score(start, task.data.spec)["solved"], task.key


@needs_ngspice
async def test_the_reward_re_simulates_the_captured_sizes_in_a_fresh_runtime(tasks):
    """`setup` writes the workspace, `finalize` captures `sizes.json`, the reward grades it in a second box."""
    task = tasks[0]
    async with provision_runtime(SubprocessConfig(), name="analog-sizing-test") as runtime:
        untouched = make_trace(task)
        await task.setup(untouched, runtime)
        assert json.loads((await runtime.read("sizes.json")).decode()) == sim.DEFAULT_SIZES
        await task.finalize(untouched, runtime)
        assert 0.0 < await task.spec_score(untouched, runtime) < 1.0

        solved = make_trace(task)
        await runtime.write("sizes.json", json.dumps(task._gold).encode())
        await task.finalize(solved, runtime)
        assert await task.spec_score(solved, runtime) == 1.0
        assert solved.info["measurements"]["ok"] and all(ratio == 1.0 for ratio in solved.info["ratios"].values())
        assert await task.validate(runtime)
