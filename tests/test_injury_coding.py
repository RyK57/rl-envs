"""Offline checks for injury-coding: header mapping, code parsing, the export reader, and the hooks."""

import csv
import io
import zipfile

import pytest
import verifiers.v1 as vf
from injury_coding import taskset as ic
from injury_coding.taskset import (
    InjuryCodingConfig,
    InjuryCodingTask,
    InjuryCodingTaskset,
    Report,
    header_key,
    matched_prefix,
    normalize_code,
    parse_code,
    read_reports,
)
from verifiers.v1.graph import MessageNode

NARRATIVE = "An employee was standing at the register and tripped over a power cord hanging from a display, fracturing an ankle."
REPORTS = (
    Report(
        "2025088697",
        NARRATIVE,
        "447110",
        {
            "nature": ("124", "Fractures"),
            "part": ("53", "Ankle(s)"),
            "event": ("4312", "Fall on same level due to slip or trip"),
            "source": ("6624", "Other constructed surface"),
        },
    ),
)


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    monkeypatch.setattr(ic, "rows_for", lambda field, split, min_chars: REPORTS)
    monkeypatch.setattr(ic, "known_codes", lambda field: frozenset({"4312", "4310", "124", "53", "6624"}))


def make_trace(task: InjuryCodingTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="InjuryCodingTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_header_and_code_normalization():
    assert header_key("Part of Body Title") == "partofbodytitle" and header_key("Final Narrative") == "finalnarrative"
    assert normalize_code("4312.0") == "4312" and normalize_code(" 124 ") == "124"
    assert parse_code("\\boxed{4312}") == "4312" and parse_code("\\boxed{4312.0}") == "4312"
    assert parse_code("\\boxed{04312}") is None and parse_code("\\boxed{43120}") is None and parse_code("4312") is None


def test_matched_prefix():
    assert matched_prefix("4312", "4312") == 1.0
    assert matched_prefix("4310", "4312") == pytest.approx(0.75)
    assert matched_prefix("4400", "4312") == pytest.approx(0.25)
    assert matched_prefix("12", "124") == pytest.approx(2 / 3)
    assert matched_prefix("5", "4312") == 0.0 and matched_prefix(None, "4312") == 0.0


def test_read_reports_maps_headers_and_normalizes(tmp_path):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "ID",
            "Event Date",
            "Employer",
            "Primary NAICS",
            "Final Narrative",
            "Nature",
            "NatureTitle",
            "Part of Body",
            "Part of Body Title",
            "Event",
            "EventTitle",
            "Source",
            "SourceTitle",
        ]
    )
    writer.writerow(
        [
            "2025088697",
            "2025-08-31",
            "Global Partners LP",
            "447110",
            "  An employee   tripped over a cord.  ",
            "124",
            "Fractures",
            "53",
            "Ankle(s)",
            "4312.0",
            "Fall on same level",
            "6624",
            "Other constructed surface",
        ]
    )
    writer.writerow(
        [
            "2025088698",
            "2025-08-31",
            "H-E-B",
            "445110",
            "An employee cut a finger.",
            "",
            "",
            "4429",
            "Other finger(s)",
            "659",
            "Struck by running powered equipment",
            "",
            "",
        ]
    )
    archive = tmp_path / "reports.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("readme.txt", "not the data")
        bundle.writestr("severeinjury.csv", "﻿" + buffer.getvalue())
    first, second = list(read_reports(archive))
    assert (
        first.id == "2025088697" and first.narrative == "An employee tripped over a cord." and first.naics == "447110"
    )
    assert first.codes["event"] == ("4312", "Fall on same level") and first.codes["nature"] == ("124", "Fractures")
    assert set(second.codes) == {"part", "event"}, "missing nature and source are absent, not empty"


async def test_load_reward_and_metrics():
    (task,) = list(InjuryCodingTaskset(InjuryCodingConfig()).load())
    assert task.key == "oiics:event:2025088697"
    assert "the event or exposure that caused it" in task.data.prompt_text
    assert "Industry (NAICS): 447110" in task.data.prompt_text and "tripped over a power cord" in task.data.prompt_text
    assert set(task.data.model_dump()).isdisjoint({"code", "gold", "codes"})
    assert await task.validate(runtime=None)
    assert await task.hierarchical(make_trace(task, "\\boxed{4312}")) == 1.0
    assert await task.hierarchical(make_trace(task, "\\boxed{4310}")) == pytest.approx(0.75)
    assert await task.division(make_trace(task, "\\boxed{4400}")) == 1.0
    assert await task.exact(make_trace(task, "\\boxed{4310}")) == 0.0
    assert await task.formatted(make_trace(task, "4312")) == 0.0
    assert await task.valid_code(make_trace(task, "\\boxed{4310}")) == 1.0
    assert await task.valid_code(make_trace(task, "\\boxed{4399}")) == 0.0
    (nature,) = list(InjuryCodingTaskset(InjuryCodingConfig(field="nature")).load())
    assert nature.key == "oiics:nature:2025088697" and "the nature of the injury" in nature.data.prompt_text
    assert await nature.hierarchical(make_trace(nature, "\\boxed{124}")) == 1.0
