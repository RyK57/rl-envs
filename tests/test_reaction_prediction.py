"""Offline checks for reaction-prediction: map stripping, the shard reader, the hooks with fixture rows, and the script."""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import verifiers.v1 as vf
from reaction_prediction import taskset as rp
from reaction_prediction.taskset import (
    ReactionPredictionConfig,
    ReactionPredictionTask,
    ReactionPredictionTaskset,
    Row,
    parse_answer,
    read_shards,
    rows_for,
    strip_maps,
)
from verifiers.v1.graph import MessageNode

MAPPED = (
    "CC(C)(C)OC(=O)O[C:12](=[O:13])[O:14][C:15]([CH3:16])([CH3:17])[CH3:18]"
    ".[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9][cH:10][nH:11]2)[cH:19]1",
    "[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9][cH:10][n:11]2[C:12](=[O:13])[O:14][C:15]"
    "([CH3:16])([CH3:17])[CH3:18])[cH:19]1",
)
ROWS = (Row("US07928231B2", strip_maps(MAPPED[0]), strip_maps(MAPPED[1])),)
PRODUCT = "CC(=O)c1ccc2c(c1)ccn2C(=O)OC(C)(C)C"
REPORTS = {
    PRODUCT: {"parsed": True, "exact": True, "tanimoto": 1.0, "formula_match": True},
    "CC(=O)c1ccc2c(c1)cc[nH]2": {"parsed": True, "exact": False, "tanimoto": 0.4, "formula_match": False},
    "not-a-smiles(": {"parsed": False, "exact": False, "tanimoto": 0.0, "formula_match": False},
}


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
        same = {"parsed": True, "exact": True, "tanimoto": 1.0, "formula_match": True}
        report = REPORTS.get(answer, same if answer == gold else REPORTS["not-a-smiles("])
        return vf.ProgramResult(exit_code=0, stdout=json.dumps(report) + "\n", stderr="")


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    """No network: one fixture reaction stands in for the split."""
    monkeypatch.setattr(rp, "rows_for", lambda split: ROWS)


def make_trace(task: ReactionPredictionTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="ReactionPredictionTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def write_shard(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_strip_maps_and_parse_answer():
    assert strip_maps("[CH3:1][C:2](=[O:3])O") == "[CH3][C](=[O])O" and strip_maps("CCO") == "CCO"
    assert ":" not in ROWS[0].reactants and ":" not in ROWS[0].product
    assert parse_answer("so \\boxed{CC(=O)O}") == "CC(=O)O" and parse_answer("\\boxed{CC O}") is None
    assert parse_answer("CC(=O)O") is None


def test_read_shards_strips_maps_and_keeps_single_products(tmp_path):
    rows = [
        {"id": ROWS[0].reaction_id, "class": "UNK", "reactants": MAPPED[0], "product": MAPPED[1]},
        {"id": "US2", "class": "UNK", "reactants": "CCO.CC(=O)O", "product": "CC(=O)OCC.O"},
        {"id": "US3", "class": "UNK", "reactants": "", "product": "CCO"},
    ]
    write_shard(tmp_path / "a.parquet", rows[:2])
    write_shard(tmp_path / "b.parquet", rows[2:])
    table = read_shards([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")])
    assert table == ROWS, "mapped row kept and stripped; a two-product row and an empty row dropped"
    with pytest.raises(ValueError, match="no usable reactions"):
        read_shards([str(tmp_path / "b.parquet")])


def test_rows_for_downloads_every_shard_of_the_split(monkeypatch, tmp_path):
    asked = []

    def fake_download(repo_id, filename, repo_type=None, revision=None):
        asked.append(filename)
        path = tmp_path / Path(filename).name
        write_shard(path, [{"id": filename, "class": "UNK", "reactants": "CCO.CC(=O)O", "product": "CC(=O)OCC"}])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    rows_for.cache_clear()
    table = rows_for("validation")
    rows_for.cache_clear()
    assert asked == [f"data/val-{i:05d}-of-00006.parquet" for i in range(6)], "the Hub's split is called val"
    assert len(table) == 6 and table[0].product == "CC(=O)OCC"


async def test_load_keys_hides_gold_and_validates():
    (task,) = list(ReactionPredictionTaskset(ReactionPredictionConfig()).load())
    assert task.key == "rxn:validation:0:US07928231B2"
    assert "Reactants: " + ROWS[0].reactants in task.data.prompt_text and ":1]" not in task.data.prompt_text
    assert set(task.data.model_dump()).isdisjoint({"product", "reactants", "gold"})
    assert await task.validate(None)
    runtime = FakeRuntime()
    assert await task.validate(runtime) and runtime.calls[-1] == ("run", ROWS[0].product, "0")
    wrong = ReactionPredictionTask(task.data.model_copy(update={"reaction_id": "US0"}), task.config)
    assert not await wrong.validate(None)


async def test_reward_and_metrics_share_one_run():
    (task,) = list(ReactionPredictionTaskset(ReactionPredictionConfig()).load())
    runtime = FakeRuntime()
    exact = make_trace(task, f"\\boxed{{{PRODUCT}}}")
    await task.setup(exact, runtime)
    scores = await asyncio.gather(
        task.correct(exact, runtime),
        task.exact(exact, runtime),
        task.parsed(exact, runtime),
        task.similarity(exact, runtime),
        task.formula_match(exact, runtime),
    )
    assert scores == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert [call for call in runtime.calls if call[0] == "run"] == [("run", PRODUCT, "0")]
    near = make_trace(task, "\\boxed{CC(=O)c1ccc2c(c1)cc[nH]2}")
    assert await task.correct(near, runtime) == pytest.approx(0.2) and await task.similarity(near, runtime) == 0.4
    assert near.info["verify"]["exact"] is False
    garbage = make_trace(task, "\\boxed{not-a-smiles(}")
    assert await task.correct(garbage, runtime) == 0.0 and await task.parsed(garbage, runtime) == 0.0
    assert await task.correct(make_trace(task, PRODUCT), runtime) == 0.0, "no box, no score"
    with pytest.raises(RuntimeError, match="runtime"):
        await task.correct(make_trace(task, f"\\boxed{{{PRODUCT}}}"), None)
    config = ReactionPredictionConfig()
    config.task.ignore_stereo = True
    (lenient,) = list(ReactionPredictionTaskset(config).load())
    await lenient.correct(make_trace(lenient, f"\\boxed{{{PRODUCT}}}"), runtime)
    assert runtime.calls[-1] == ("run", PRODUCT, "1")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv runs the verifier script")
def test_verify_script_clears_atom_maps():
    script = Path(rp.__file__).with_name("verify.py")

    def run(*args):
        return subprocess.run(
            ["uv", "run", "--no-config", "--script", str(script), *args], capture_output=True, text=True
        )

    report = json.loads(run(MAPPED[1], PRODUCT, "0").stdout.strip().splitlines()[-1])
    assert report["exact"] is True and report["tanimoto"] == 1.0, "a mapped gold equals its unmapped product"
    other = json.loads(run(MAPPED[1], "CC(=O)c1ccc2c(c1)cc[nH]2", "0").stdout.strip().splitlines()[-1])
    assert other["exact"] is False and 0.0 < other["tanimoto"] < 1.0
    assert run("bad(", "CC", "0").returncode == 2
