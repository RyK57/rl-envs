"""Offline checks for ghs-hazards: code parsing, the soft F1, the reader on a fixture table, and the hooks."""

import hashlib

import pytest
import verifiers.v1 as vf
from ghs_hazards import taskset as ghs
from ghs_hazards.taskset import (
    GhsHazardsConfig,
    GhsHazardsTask,
    GhsHazardsTaskset,
    Row,
    class_f1,
    parse_codes,
    parse_table,
    prompt_for,
    soft_f1,
    split_of,
    strict_f1,
)
from verifiers.v1.graph import MessageNode

ROWS = (
    Row("7789-38-0", "[Na+].[O-][Br](=O)=O", frozenset({"H271", "H301", "H315", "H319", "H335"})),
    Row("57-50-1", "OC[C@H]1O[C@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", frozenset()),
    Row("917-92-0", "CC(C)(C)C#C", frozenset({"H225"})),
)
CSV = (
    "\n".join(
        [
            "SMILES,cas_nr,h_statements,H225,H301,H302,H319",
            "CC(C)(C)C#C,917-92-0,['H225'],1,0,0,0",
            "[Cl-].[K+],7447-40-7,,0,0,0,0",
            ",9000-65-1,,0,0,0,0",
            "[Li+]|O=[Ni][O-],12031-65-1,\"['H302']\",0,0,1,0",
            "CC(C)(C)C#C,999-99-9,\"['H301', 'H319']\",0,1,0,1",
            "CCCCNCCO,111-75-1,\"['H302', 'H319']\",0,0,1,1",
        ]
    )
    + "\n"
).encode()


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    """No network: three fixture substances stand in for the split."""
    monkeypatch.setattr(ghs, "rows_for", lambda split: ROWS)


def make_trace(task: GhsHazardsTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="GhsHazardsTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_parse_codes_reads_the_last_box():
    assert parse_codes("\\boxed{H225, H319}") == {"H225", "H319"}
    assert parse_codes("\\boxed{h225; H319 and H360FD}") == {"H225", "H319", "H360"}, "case and sub-statements"
    assert parse_codes("\\boxed{H301+H311}") == {"H301", "H311"}
    assert parse_codes("\\boxed{H225} no, \\boxed{H226}") == {"H226"}
    assert parse_codes("\\boxed{none}") == frozenset() and parse_codes("\\boxed{ None }") == frozenset()
    assert parse_codes("\\boxed{nothing}") is None and parse_codes("H225 without a box") is None


def test_soft_f1_ladder():
    gold = frozenset({"H271", "H301", "H315"})
    assert soft_f1(gold, gold) == 1.0
    assert soft_f1(frozenset({"H301"}), frozenset({"H302"})) == 0.5, "right class, wrong category"
    assert soft_f1(frozenset({"H271", "H301", "H316"}), gold) == pytest.approx(5 / 6), "two exact, one half"
    assert soft_f1(frozenset({"H271", "H301", "H400"}), gold) == pytest.approx(2 / 3), "a wrong class earns nothing"
    assert soft_f1(frozenset({"H301", "H303"}), frozenset({"H302"})) == pytest.approx(1 / 3), (
        "one gold pairs with one guess"
    )
    assert soft_f1(frozenset(), frozenset()) == 1.0, "none for a substance without statements"
    assert soft_f1(frozenset(), gold) == 0.0 and soft_f1(gold, frozenset()) == 0.0


def test_strict_and_class_f1():
    gold = frozenset({"H301", "H315"})
    assert strict_f1(frozenset({"H302", "H315"}), gold) == pytest.approx(0.5)
    assert class_f1(frozenset({"H302", "H315"}), gold) == 1.0
    assert class_f1(frozenset({"H999"}), gold) == 0.0, "an unknown code is its own class"
    assert {split_of(f"{i}-00-0") for i in range(300)} == {"train", "validation", "test"}


def test_parse_table_filters_and_pins(monkeypatch):
    monkeypatch.setattr(ghs, "FILE_SHA256", hashlib.sha256(CSV).hexdigest())
    rows = parse_table(CSV)
    assert [row.cas for row in rows] == ["917-92-0", "7447-40-7", "111-75-1"], (
        "no structure, odd notation, duplicate dropped"
    )
    assert rows[0].codes == {"H225"} and rows[1].codes == frozenset() and rows[2].codes == {"H302", "H319"}
    monkeypatch.setattr(ghs, "FILE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="pinned"):
        parse_table(CSV)


async def test_load_keys_hides_gold_and_validates():
    tasks = list(GhsHazardsTaskset(GhsHazardsConfig()).load())
    assert [t.key for t in tasks] == ["ghs:validation:7789-38-0", "ghs:validation:57-50-1", "ghs:validation:917-92-0"]
    assert "SMILES: [Na+].[O-][Br](=O)=O" in tasks[0].data.prompt_text and "CAS" not in tasks[0].data.prompt_text
    assert set(tasks[0].data.model_dump()).isdisjoint({"codes", "h_statements", "gold", "smiles"})
    assert all([await t.validate(runtime=None) for t in tasks])
    hazardous = list(GhsHazardsTaskset(GhsHazardsConfig(negatives=False)).load())
    assert [t.data.cas for t in hazardous] == ["7789-38-0", "917-92-0"] and hazardous[1].data.row == 2
    with_cas = list(GhsHazardsTaskset(GhsHazardsConfig(cas=True)).load())
    assert "CAS: 7789-38-0" in with_cas[0].data.prompt_text


def test_prompt_shape():
    assert "\\boxed{none}" in prompt_for("CC", None) and "CAS" not in prompt_for("CC", None)
    assert prompt_for("CC", "74-84-0").endswith("SMILES: CC\nCAS: 74-84-0")


async def test_reward_and_metrics_on_a_hazardous_substance():
    task = list(GhsHazardsTaskset(GhsHazardsConfig()).load())[0]
    exact = make_trace(task, "\\boxed{H271, H301, H315, H319, H335}")
    near = make_trace(task, "\\boxed{H272, H301, H315, H319, H335}")
    missing = make_trace(task, "\\boxed{H301, H315}")
    none = make_trace(task, "\\boxed{none}")
    unformatted = make_trace(task, "H271, H301, H315, H319, H335")
    invalid = make_trace(task, "\\boxed{H999}")
    assert await task.hazard_f1(exact) == 1.0 and await task.exact(exact) == 1.0
    assert await task.precision(exact) == 1.0 and await task.recall(exact) == 1.0
    assert await task.hazard_f1(near) == pytest.approx(0.9), "H272 is oxidizing like H271, half credit"
    assert await task.strict(near) == pytest.approx(0.8) and await task.hazard_class(near) == 1.0
    assert await task.exact(near) == 0.0
    assert await task.hazard_f1(missing) == pytest.approx(2 * 1.0 * 0.4 / 1.4)
    assert await task.recall(missing) == pytest.approx(0.4) and await task.precision(missing) == 1.0
    assert await task.hazard_f1(none) == 0.0 and await task.formatted(none) == 1.0
    assert await task.hazard_f1(unformatted) == 0.0 and await task.formatted(unformatted) == 0.0
    assert await task.valid_codes(exact) == 1.0 and await task.valid_codes(invalid) == 0.0
    assert await task.num_codes(exact) == 5.0 and await task.num_codes(none) == 0.0


async def test_reward_and_metrics_on_a_substance_without_statements():
    task = list(GhsHazardsTaskset(GhsHazardsConfig()).load())[1]
    none = make_trace(task, "\\boxed{none}")
    guess = make_trace(task, "\\boxed{H319}")
    assert await task.hazard_f1(none) == 1.0 and await task.exact(none) == 1.0
    assert await task.precision(none) == 1.0 and await task.recall(none) == 1.0
    assert await task.hazard_f1(guess) == 0.0 and await task.precision(guess) == 0.0 and await task.recall(guess) == 0.0
    assert await task.validate(runtime=None)
