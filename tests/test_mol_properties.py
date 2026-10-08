"""Offline checks for mol-properties: answer parsing, the scoring rules, the hooks with fixture rows, and the script."""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import verifiers.v1 as vf
from iupac_structure.taskset import Row
from mol_properties import taskset as mp
from mol_properties.taskset import (
    SPECS,
    MolPropertiesConfig,
    MolPropertiesTask,
    MolPropertiesTaskset,
    parse_answer,
    parse_number,
    score,
)
from verifiers.v1.graph import MessageNode

ROWS = (Row("2244", "2-acetyloxybenzoic acid", "CC(=O)Oc1ccccc1C(O)=O", "C9H8O4"),)
PROPERTIES = {
    "formula": "C9H8O4",
    "molecular_weight": 180.159,
    "monoisotopic_mass": 180.042259,
    "heavy_atoms": 13,
    "rings": 1,
    "aromatic_rings": 1,
    "stereocenters": 0,
}


class FakeRuntime:
    """Stands in for the sandbox: answers like props.py would and records what was asked of it."""

    def __init__(self, fail_first: bool = False):
        self.calls = []
        self.fail_first = fail_first

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        self.calls.append(("prepare", len(script)))
        return []

    async def run_uv_script(self, script, args=None, env=None):
        self.calls.append(("run", args[0]))
        if self.fail_first and len([call for call in self.calls if call[0] == "run"]) == 1:
            return vf.ProgramResult(exit_code=1, stdout="", stderr="boom")
        return vf.ProgramResult(exit_code=0, stdout=json.dumps(PROPERTIES) + "\n", stderr="")


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    """No network: one fixture compound stands in for the split at any SMILES cap."""
    monkeypatch.setattr(mp, "rows_for", lambda split, cap: ROWS)


def make_trace(task: MolPropertiesTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MolPropertiesTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_parse_and_score():
    assert parse_answer("about \\boxed{180.16}.") == "180.16" and parse_answer("\\boxed{180.16 g/mol}") is None
    assert parse_number("1,234.5") == 1234.5 and parse_number("−3") == -3.0 and parse_number("x") is None
    weight = SPECS["molecular_weight"]
    assert score(weight, "180.16", 180.159) == 1.0, "within tolerance"
    assert score(weight, "180.0", 180.159) == pytest.approx(1 - 0.159 / (0.05 * 180.159)), "linear decay"
    assert score(weight, "190", 180.159) == 0.0, "more than five percent off"
    assert score(SPECS["heavy_atoms"], "13", 13) == 1.0 and score(SPECS["heavy_atoms"], "13.0", 13) == 1.0
    assert score(SPECS["heavy_atoms"], "12", 13) == 0.0 and score(SPECS["heavy_atoms"], "many", 13) == 0.0
    assert score(SPECS["formula"], "C₉H₈O₄", "C9H8O4") == 1.0 and score(SPECS["formula"], "C9H10O4", "C9H8O4") == 0.0
    assert score(SPECS["monoisotopic_mass"], "180.0423", 180.042259) == 1.0


async def test_load_prompt_identity_and_offline_limits():
    task = list(MolPropertiesTaskset(MolPropertiesConfig()).load())[0]
    assert task.key == "molprop:formula:2244"
    assert "molecular formula" in task.data.prompt_text and "IUPAC" not in task.data.prompt_text
    assert set(task.data.model_dump()).isdisjoint({"smiles", "formula", "gold", "iupac"})
    assert await task.formatted(make_trace(task, "\\boxed{C9H8O4}")) == 1.0
    assert await task.formatted(make_trace(task, "C9H8O4")) == 0.0
    assert await task.validate(None), "shape checks only without a runtime"
    with pytest.raises(RuntimeError, match="runtime"):
        await task.correct(make_trace(task, "\\boxed{C9H8O4}"), None)
    assert await task.correct(make_trace(task, "no box"), None) == 0.0, "nothing to compute for an unformatted reply"
    named = list(MolPropertiesTaskset(MolPropertiesConfig(show_name=True, property="rings")).load())[0]
    assert (
        "IUPAC name: 2-acetyloxybenzoic acid" in named.data.prompt_text and "number of rings" in named.data.prompt_text
    )


async def test_properties_are_computed_once_per_task():
    task = list(MolPropertiesTaskset(MolPropertiesConfig(property="molecular_weight")).load())[0]
    runtime = FakeRuntime()
    close = make_trace(task, "\\boxed{180.16}")
    far = make_trace(task, "\\boxed{170}")
    await task.setup(close, runtime)
    scores = await asyncio.gather(
        task.correct(close, runtime),
        task.exact(close, runtime),
        task.abs_error(close, runtime),
        task.correct(far, runtime),
        task.abs_error(far, runtime),
    )
    assert scores[:3] == [1.0, 1.0, pytest.approx(0.001)]
    assert scores[3] == 0.0 and scores[4] == pytest.approx(10.159)
    assert [call for call in runtime.calls if call[0] == "run"] == [("run", "CC(=O)Oc1ccccc1C(O)=O")], "one RDKit run"
    assert close.info["gold"] == 180.159


async def test_failed_computation_is_retried_and_validate_uses_the_script():
    task = list(MolPropertiesTaskset(MolPropertiesConfig(property="rings")).load())[0]
    runtime = FakeRuntime(fail_first=True)
    with pytest.raises(RuntimeError, match="props.py failed"):
        await task.correct(make_trace(task, "\\boxed{1}"), runtime)
    assert await task.correct(make_trace(task, "\\boxed{1}"), runtime) == 1.0, "the failure was not cached"
    assert await task.validate(runtime)
    wrong = MolPropertiesTask(task.data.model_copy(update={"cid": "999"}), task.config)
    assert not await wrong.validate(None)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv runs the property script")
def test_props_script_runs_through_uv():
    script = Path(mp.__file__).with_name("props.py")
    run = subprocess.run(
        ["uv", "run", "--no-config", "--script", str(script), "CC(=O)Oc1ccccc1C(O)=O"], capture_output=True, text=True
    )
    assert run.returncode == 0
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["formula"] == "C9H8O4" and report["heavy_atoms"] == 13 and report["rings"] == 1
    assert report["stereocenters"] == 0 and 180.1 < report["molecular_weight"] < 180.2
    bad = subprocess.run(["uv", "run", "--no-config", "--script", str(script), "bad("], capture_output=True, text=True)
    assert bad.returncode == 2
