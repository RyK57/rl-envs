"""Offline checks for medical-coding: code parsing, the two rewards, and the hooks with fixture rows."""

import pytest
import verifiers.v1 as vf
from medical_coding import taskset as mc
from medical_coding.taskset import (
    MedicalCodingConfig,
    MedicalCodingTask,
    MedicalCodingTaskset,
    hierarchical,
    normalize_code,
    parse_codes,
    set_f1,
    split_of,
)
from verifiers.v1.graph import MessageNode

CASES = (("ED0002", "A man in his late 60s presented with throat pain.", ("J0511",)),)
CODIESP = (("S0004-1", "Paciente de 70 años con hematuria.", ("N329", "R300", "C679")),)


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    monkeypatch.setattr(mc, "rows_for", lambda source, split: CASES if source == "cases" else CODIESP)
    monkeypatch.setattr(mc, "icd10_codes", lambda: frozenset({"J0511", "J05", "N329", "R300", "C679"}))


def make_trace(task: MedicalCodingTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MedicalCodingTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_normalize_and_parse_codes():
    assert normalize_code("j05.11") == "J0511" and normalize_code("H40.61X0") == "H4061X0"
    assert parse_codes("\\boxed{J05.11}") == ["J0511"]
    assert parse_codes("\\boxed{n32.9, r30.0; N32.9 c67.9}") == ["N329", "R300", "C679"]
    assert parse_codes("\\boxed{not a code}") == []
    assert parse_codes("J05.11 without a box") is None


def test_hierarchical_ladder():
    assert hierarchical("J0511", "J0511") == 1.0
    assert hierarchical("J051", "J0511") == pytest.approx(0.75)
    assert hierarchical("J05", "J0511") == pytest.approx(0.5)
    assert hierarchical("J0512", "J0511") == pytest.approx(0.75), "four shared characters of five"
    assert hierarchical("J06", "J0511") == 0.0 and hierarchical(None, "J0511") == 0.0
    assert hierarchical("E11", "E11") == 1.0
    assert hierarchical("H4061X0", "H4061X0") == 1.0 and hierarchical("H40", "H4061X0") == pytest.approx(0.5)


def test_set_f1_and_split():
    assert set_f1(["N329", "R300"], ("N329", "R300")) == 1.0
    assert set_f1(["N329", "X999"], ("N329", "R300")) == pytest.approx(0.5)
    assert set_f1(["X999"], ("N329",)) == 0.0
    assert {split_of(f"case-{i}") for i in range(300)} == {"train", "validation", "test"}


async def test_cases_source():
    (task,) = list(MedicalCodingTaskset(MedicalCodingConfig()).load())
    assert task.key == "icd10:cases:ED0002" and "principal diagnosis" in task.data.prompt_text
    assert set(task.data.model_dump()).isdisjoint({"gold", "icd10_code", "diagnosis"})
    assert await task.validate(runtime=None)
    assert await task.coding(make_trace(task, "\\boxed{J05.11}")) == 1.0
    assert await task.coding(make_trace(task, "\\boxed{J05.1}")) == pytest.approx(0.75)
    assert await task.category(make_trace(task, "\\boxed{J05.1}")) == 1.0
    assert await task.coding(make_trace(task, "\\boxed{J05.11, J05.10}")) == 0.0, "one code was asked for"
    assert await task.formatted(make_trace(task, "\\boxed{J05.11, J05.10}")) == 0.0
    assert await task.valid_code(make_trace(task, "\\boxed{J05.11}")) == 1.0
    assert await task.valid_code(make_trace(task, "\\boxed{J99.99}")) == 0.0
    assert await task.exact(make_trace(task, "\\boxed{J05.11}")) == 1.0


async def test_codiesp_source():
    (task,) = list(MedicalCodingTaskset(MedicalCodingConfig(source="codiesp")).load())
    assert task.key == "icd10:codiesp:S0004-1" and "separated by commas" in task.data.prompt_text
    assert await task.validate(runtime=None)
    assert await task.coding(make_trace(task, "\\boxed{N32.9, R30.0, C67.9}")) == 1.0
    assert await task.coding(make_trace(task, "\\boxed{N32.9}")) == pytest.approx(0.5)
    assert await task.category(make_trace(task, "\\boxed{N32.8}")) == pytest.approx(0.5)
    assert await task.formatted(make_trace(task, "\\boxed{N32.9, R30.0}")) == 1.0
    assert await task.exact(make_trace(task, "\\boxed{N32.9}")) == 0.0
