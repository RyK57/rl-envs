"""Offline checks for procurement-coding: description cleaning, code parsing, the shard reader, and the hooks."""

import pytest
import verifiers.v1 as vf
from procurement_coding import taskset as pcod
from procurement_coding.taskset import (
    Action,
    ProcurementCodingConfig,
    ProcurementCodingTask,
    ProcurementCodingTaskset,
    clean_description,
    matched_prefix,
    parse_code,
    usable,
)
from verifiers.v1.graph import MessageNode

ACTIONS = (
    Action(
        "AWD-1",
        "Custom software development and maintenance for the case management system.",
        "541511",
        "Custom Computer Programming Services",
        "D302",
        "IT and Telecom - Systems Development",
    ),
    Action(
        "AWD-2",
        "Purchase of forty office chairs with adjustable arms for the regional headquarters.",
        "337214",
        "Office Furniture (except Wood) Manufacturing",
        "7110",
        "Office Furniture",
    ),
)


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    monkeypatch.setattr(pcod, "rows_for", lambda split, min_chars: ACTIONS)


def make_trace(task: ProcurementCodingTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="ProcurementCodingTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_clean_and_usable():
    assert clean_description("IGF::OT::IGF  Custom software   development") == "Custom software development"
    assert usable("Custom software development and maintenance for the case management system.", 60)
    assert not usable("PART 123-456-789 QTY 4", 10), "mostly digits"
    assert not usable("Short", 60)


def test_parse_code_and_prefix():
    assert parse_code("naics", "\\boxed{541511}") == "541511" and parse_code("naics", "\\boxed{5415}") is None
    assert parse_code("psc", "\\boxed{d302}") == "D302" and parse_code("psc", "\\boxed{D3}") is None
    assert parse_code("naics", "541511") is None
    assert matched_prefix("541511", "541511") == 1.0
    assert matched_prefix("541519", "541511") == pytest.approx(5 / 6)
    assert matched_prefix("541330", "541511") == pytest.approx(3 / 6)
    assert matched_prefix("722511", "541511") == 0.0 and matched_prefix(None, "541511") == 0.0
    assert matched_prefix("D399", "D302") == pytest.approx(0.5) and matched_prefix("7110", "7110") == 1.0


def test_actions_reads_and_filters_a_shard(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    records = {
        "contract_award_unique_key": ["A", "B", "C", "D", "E"],
        "transaction_description": [
            "IGF::OT::IGF Custom software development and maintenance for the case management system.",
            "custom software development and maintenance for the case management system.",
            "PART 12345 QTY 2",
            "Purchase of forty office chairs with adjustable arms for the regional headquarters.",
            "Purchase of forty office desks with adjustable legs for the regional headquarters.",
        ],
        "naics_code": ["541511", "541511", "541511", "337214", "3372"],
        "naics_description": ["a", "a", "a", "b", "b"],
        "product_or_service_code": ["D302", "D302", "D302", "7110", "7110"],
        "product_or_service_code_description": ["x", "x", "x", "y", "y"],
        "unused": ["", "", "", "", ""],
    }
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.table(records), path)
    monkeypatch.setattr(pcod, "hf_hub_download", lambda *a, **k: str(path), raising=False)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    pcod.actions.cache_clear()
    rows = pcod.actions.__wrapped__(60)
    assert [r.award for r in rows] == ["A", "D"], "tag stripped, duplicate dropped, junk dropped, bad NAICS dropped"
    assert rows[0].description.startswith("Custom software")


async def test_naics_and_psc_fields():
    naics_tasks = list(ProcurementCodingTaskset(ProcurementCodingConfig()).load())
    assert [t.key for t in naics_tasks] == ["procurement:naics:AWD-1:0", "procurement:naics:AWD-2:1"]
    assert "six-digit NAICS" in naics_tasks[0].data.prompt_text
    assert set(naics_tasks[0].data.model_dump()).isdisjoint({"naics", "psc", "gold"})
    assert all([await t.validate(runtime=None) for t in naics_tasks])
    software = naics_tasks[0]
    assert await software.hierarchical(make_trace(software, "\\boxed{541511}")) == 1.0
    assert await software.hierarchical(make_trace(software, "\\boxed{541512}")) == pytest.approx(5 / 6)
    assert await software.sector(make_trace(software, "\\boxed{541990}")) == 1.0
    assert await software.exact(make_trace(software, "\\boxed{541990}")) == 0.0
    assert await software.formatted(make_trace(software, "\\boxed{D302}")) == 0.0
    psc_tasks = list(ProcurementCodingTaskset(ProcurementCodingConfig(field="psc")).load())
    chairs = psc_tasks[1]
    assert "Product or Service Code" in chairs.data.prompt_text
    assert await chairs.hierarchical(make_trace(chairs, "\\boxed{7110}")) == 1.0
    assert await chairs.hierarchical(make_trace(chairs, "\\boxed{7195}")) == pytest.approx(0.5)
    assert await chairs.sector(make_trace(chairs, "\\boxed{7195}")) == 1.0
    assert await chairs.validate(runtime=None)
