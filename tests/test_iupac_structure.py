"""Offline checks for iupac-structure: answer parsing, the row filters, both targets, and the RDKit script."""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import verifiers.v1 as vf
from iupac_structure import taskset as iu
from iupac_structure.taskset import (
    IupacStructureConfig,
    IupacStructureTask,
    IupacStructureTaskset,
    Row,
    normalize_formula,
    parse_answer,
    rows_for,
    select_rows,
    split_of,
    usable,
)
from verifiers.v1.graph import MessageNode

ROWS = (
    Row("2244", "2-acetyloxybenzoic acid", "CC(=O)Oc1ccccc1C(O)=O", "C9H8O4"),
    Row("5950", "(2S)-2-aminopropanoic acid", "C[C@H](N)C(O)=O", "C3H7NO2"),
)
REPORTS = {
    "OC(=O)c1ccccc1OC(C)=O": {"parsed": True, "exact": True, "tanimoto": 1.0, "formula_match": True},
    "CC(=O)Oc1ccccc1C(=O)OC": {"parsed": True, "exact": False, "tanimoto": 0.6, "formula_match": False},
    "not-a-smiles(": {"parsed": False, "exact": False, "tanimoto": 0.0, "formula_match": False},
}
SAME = {"parsed": True, "exact": True, "tanimoto": 1.0, "formula_match": True}


class FakeRuntime:
    """Stands in for the sandbox: answers like verify.py would and records what was asked of it."""

    def __init__(self):
        self.calls = []

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        self.calls.append(("prepare", len(script)))
        return []

    async def run_uv_script(self, script, args=None, env=None):
        gold, answer, stereo = args
        self.calls.append(("run", answer, stereo))
        report = REPORTS.get(answer, SAME if answer == gold else REPORTS["not-a-smiles("])
        return vf.ProgramResult(exit_code=0, stdout=json.dumps(report) + "\n", stderr="")


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    """No network: two fixture compounds stand in for the split at any SMILES cap."""
    monkeypatch.setattr(iu, "rows_for", lambda split, cap: ROWS)


def make_trace(task: IupacStructureTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="IupacStructureTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_parse_answer_and_formula_normalization():
    assert parse_answer("The structure is \\boxed{CC(=O)Oc1ccccc1C(O)=O}.") == "CC(=O)Oc1ccccc1C(O)=O"
    assert parse_answer("\\boxed{`C9H8O4`}") == "C9H8O4" and parse_answer("\\boxed{CC} then \\boxed{CCO}") == "CCO"
    assert parse_answer("\\boxed{SMILES: CCO}") is None, "one token only"
    assert parse_answer("CCO without a box") is None and parse_answer("\\boxed{ }") is None
    assert normalize_formula("C₉H₈O₄") == "C9H8O4" and normalize_formula("C2H3O2−") == "C2H3O2-"
    assert normalize_formula(" C9 H8 O4 ") == "C9H8O4"


def test_usable_and_select_rows():
    good = {"CID": 1, "SMILES_Canonical": "CCO", "formula": "C2H6O", "iupac": "ethanol"}
    salt = {**good, "CID": 2, "SMILES_Canonical": "[Na+].[Cl-]", "formula": "ClNa"}
    isotope = {**good, "CID": 3, "SMILES_Canonical": "[13CH3]O"}
    greek = {**good, "CID": 4, "iupac": "λ5-phosphane"}
    unnamed = {**good, "CID": 5, "iupac": None}
    long_name = {**good, "CID": 6, "iupac": "x" * 121}
    assert usable(good) and not usable(salt) and not usable(isotope) and not usable(greek)
    assert not usable(unnamed) and not usable(long_name)
    chosen = select_rows([salt, good, isotope, {**good, "CID": 7}, {**good, "CID": 8}], pool=2)
    assert [row["cid"] for row in chosen] == ["1", "7"] and chosen[0]["smiles"] == "CCO"
    assert {split_of(str(i)) for i in range(300)} == {"train", "validation", "test"}


def test_rows_for_reads_the_compact_table(monkeypatch, tmp_path):
    table = tmp_path / "pubchem.jsonl"
    entries = [
        {"cid": str(i), "iupac": f"compound {i}", "smiles": "C" * (i % 7 + 1), "formula": "CH4"} for i in range(60)
    ]
    table.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    monkeypatch.setattr(iu, "compact_table", lambda: table)
    rows_for.cache_clear()
    everything = sum(len(rows_for(split, 120)) for split in ("train", "validation", "test"))
    assert everything == 60 and all(len(row.smiles) <= 3 for row in rows_for("train", 3))
    assert isinstance(rows_for("train", 120)[0], Row)
    rows_for.cache_clear()


async def test_formula_target_scores_offline():
    tasks = list(IupacStructureTaskset(IupacStructureConfig(target="formula")).load())
    task = tasks[0]
    assert task.key == "iupac:formula:2244" and "molecular formula" in task.data.prompt_text
    assert set(task.data.model_dump()).isdisjoint({"smiles", "formula", "iupac", "gold"})
    assert await task.correct(make_trace(task, "\\boxed{C9H8O4}")) == 1.0
    assert await task.correct(make_trace(task, "\\boxed{C₉H₈O₄}")) == 1.0
    assert await task.correct(make_trace(task, "\\boxed{C9H10O4}")) == 0.0
    assert await task.exact(make_trace(task, "\\boxed{C9H8O4}")) == 1.0
    assert (
        await task.parsed(make_trace(task, "\\boxed{C9H10O4}")) == 1.0
        and await task.parsed(make_trace(task, "\\boxed{??}")) == 0.0
    )
    assert await task.formatted(make_trace(task, "C9H8O4")) == 0.0
    assert all([await t.validate(runtime=None) for t in tasks])


async def test_smiles_target_uses_the_runtime_once_per_trace():
    task = list(IupacStructureTaskset(IupacStructureConfig()).load())[0]
    assert (
        task.key == "iupac:smiles:2244" and "SMILES" in task.data.prompt_text and "formula" not in task.data.prompt_text
    )
    runtime = FakeRuntime()
    exact = make_trace(task, "\\boxed{OC(=O)c1ccccc1OC(C)=O}")
    await task.setup(exact, runtime)
    assert runtime.calls == [("prepare", len(iu.VERIFY))]
    scores = await asyncio.gather(
        task.correct(exact, runtime),
        task.exact(exact, runtime),
        task.parsed(exact, runtime),
        task.similarity(exact, runtime),
    )
    assert scores == [1.0, 1.0, 1.0, 1.0], "a differently written SMILES of the same structure"
    assert [call for call in runtime.calls if call[0] == "run"] == [("run", "OC(=O)c1ccccc1OC(C)=O", "0")], (
        "one RDKit run shared"
    )
    near = make_trace(task, "\\boxed{CC(=O)Oc1ccccc1C(=O)OC}")
    assert await task.correct(near, runtime) == pytest.approx(0.3), "half the Tanimoto similarity"
    assert await task.exact(near, runtime) == 0.0 and await task.similarity(near, runtime) == pytest.approx(0.6)
    assert near.info["verify"]["tanimoto"] == 0.6
    garbage = make_trace(task, "\\boxed{not-a-smiles(}")
    assert await task.correct(garbage, runtime) == 0.0 and await task.parsed(garbage, runtime) == 0.0
    unformatted = make_trace(task, "CC(=O)Oc1ccccc1C(O)=O")
    runs = len(runtime.calls)
    assert await task.correct(unformatted, runtime) == 0.0 and await task.formatted(unformatted) == 0.0
    assert len(runtime.calls) == runs, "nothing to verify without a boxed answer"


async def test_smiles_scoring_needs_a_runtime_and_honours_the_stereo_knob():
    task = list(IupacStructureTaskset(IupacStructureConfig()).load())[1]
    with pytest.raises(RuntimeError, match="runtime"):
        await task.correct(make_trace(task, "\\boxed{CC(N)C(O)=O}"), None)
    config = IupacStructureConfig()
    config.task.ignore_stereo = True
    lenient = list(IupacStructureTaskset(config).load())[1]
    runtime = FakeRuntime()
    await lenient.correct(make_trace(lenient, "\\boxed{CC(N)C(O)=O}"), runtime)
    assert runtime.calls[-1] == ("run", "CC(N)C(O)=O", "1")


async def test_validate_runs_the_gold_through_the_script():
    task = list(IupacStructureTaskset(IupacStructureConfig()).load())[0]
    runtime = FakeRuntime()
    assert await task.validate(runtime)
    assert runtime.calls[-1] == ("run", "CC(=O)Oc1ccccc1C(O)=O", "0"), "the gold against itself"
    assert await task.validate(None), "shape checks only without a runtime"
    wrong = IupacStructureTask(task.data.model_copy(update={"cid": "999"}), task.config)
    assert not await wrong.validate(None)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv runs the verifier script")
def test_verify_script_runs_through_uv():
    script = Path(iu.__file__).with_name("verify.py")
    run = lambda *args: subprocess.run(
        ["uv", "run", "--no-config", "--script", str(script), *args], capture_output=True, text=True
    )
    same = run("CC(=O)Oc1ccccc1C(O)=O", "OC(=O)c1ccccc1OC(C)=O", "0")
    assert same.returncode == 0 and json.loads(same.stdout.strip().splitlines()[-1])["exact"] is True
    stereo = json.loads(run("C[C@H](N)C(O)=O", "CC(N)C(O)=O", "0").stdout.strip().splitlines()[-1])
    assert stereo["exact"] is False and stereo["tanimoto"] == 1.0
    assert json.loads(run("C[C@H](N)C(O)=O", "CC(N)C(O)=O", "1").stdout.strip().splitlines()[-1])["exact"] is True
    assert json.loads(run("CCO", "not-a-smiles(", "0").stdout.strip().splitlines()[-1])["parsed"] is False
    assert run("bad(", "CC", "0").returncode == 2, "an unreadable gold is an infrastructure error"
