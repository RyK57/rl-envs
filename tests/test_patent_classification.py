"""Offline checks for patent-classification: symbol parsing, the level score, the archive reader, and the hooks."""

import io
import json
import tarfile

import pytest
import verifiers.v1 as vf
from patent_classification import taskset as pc
from patent_classification.taskset import (
    PatentClassificationConfig,
    PatentClassificationTask,
    PatentClassificationTaskset,
    application_rows,
    candidates,
    canonical,
    levels,
    matched_levels,
    parse_label,
    resolve,
    split_of,
)
from patent_classification.taskset import rows_for as real_rows_for
from verifiers.v1.graph import MessageNode

ROWS = (("14000001", "Data storage system", "A system stores data in a tree.", "G06F17/30"),)


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    monkeypatch.setattr(pc, "rows_for", lambda split: ROWS)
    monkeypatch.setattr(pc, "scheme_keys", lambda: frozenset({"G06F17/30", "G06F17/00", "H04L9/08"}))


def make_trace(task: PatentClassificationTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="PatentClassificationTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_canonical_and_levels():
    assert canonical("G06F 17/30") == "G06F17/30" and canonical("g06f17/30") == "G06F17/30"
    assert canonical("G06F17-30") == "G06F17/30" and canonical("H04L 9/0894") == "H04L9/0894"
    assert canonical("G06F17") is None and canonical("") is None and canonical("Z06F17/30") is None
    assert levels("G06F17/30") == ("G", "G06", "G06F", "G06F17", "G06F17/30")


def test_hupd_labels_resolve_through_the_scheme():
    assert candidates("H04N214312") == ["H04N2/14312", "H04N21/4312", "H04N214/312", "H04N2143/12"]
    assert candidates("B41J2938") == ["B41J2/938", "B41J29/38"]
    assert candidates("G06F17/30") == [] and candidates("A61B0123") == []
    scheme = frozenset({"H04N21/4312", "B41J29/38", "A61B1/00082", "A61B10/0082"})
    assert resolve("H04N214312", scheme) == "H04N21/4312"
    assert resolve("B41J2938", scheme) == "B41J29/38"
    assert resolve("G06F 17/30", scheme) == "G06F17/30", "a symbol written with a slash needs no scheme"
    assert resolve("A61B100082", scheme) is None, "two readings in the scheme: ambiguous, dropped"
    assert resolve("C01G314", scheme) is None, "no reading in the scheme"


def test_matched_levels():
    assert matched_levels("G06F17/30", "G06F17/30") == 5
    assert matched_levels("G06F17/00", "G06F17/30") == 4
    assert matched_levels("G06F3/06", "G06F17/30") == 3
    assert matched_levels("G06Q10/00", "G06F17/30") == 2
    assert matched_levels("G11C7/10", "G06F17/30") == 1
    assert matched_levels("H04L9/08", "G06F17/30") == 0 and matched_levels(None, "G06F17/30") == 0


def test_parse_label_and_split():
    assert parse_label("Probably \\boxed{H04L 9/08} or \\boxed{G06F 17/30}") == "G06F17/30"
    assert parse_label("\\boxed{unknown}") is None and parse_label("G06F17/30") is None
    assert {split_of(str(14000000 + i)) for i in range(300)} == {"train", "validation", "test"}


def test_application_rows_reads_the_archive(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for number, abstract in (("14000001", "A system stores data in a tree."), ("14000002", "")):
            payload = json.dumps(
                {
                    "application_number": number,
                    "title": "Data storage system",
                    "abstract": abstract,
                    "main_cpc_label": "G06F 17/30",
                    "decision": "ACCEPTED",
                    "full_description": "x" * 1000,
                }
            ).encode()
            info = tarfile.TarInfo(f"sample/2016/{number}.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    rows = list(application_rows(archive))
    assert [r["application"] for r in rows] == ["14000001", "14000002"]
    assert rows[0]["label"] == "G06F 17/30" and rows[0]["abstract"].startswith("A system")
    assert "full_description" not in rows[0]


def test_check_table_names_the_problem(tmp_path):
    archive = tmp_path / "empty.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("sample/2016/readme.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
    with pytest.raises(ValueError, match="no application JSON files"):
        pc.check_table([], archive)
    with pytest.raises(ValueError, match="CPC symbol in the expected form"):
        pc.check_table([{"application": "1", "title": "t", "abstract": "a", "label": "G06F", "decision": ""}], archive)
    pc.check_table(
        [{"application": "1", "title": "t", "abstract": "a", "label": "H04N214312", "decision": ""}], archive
    )
    with pytest.raises(ValueError, match="has an abstract"):
        pc.check_table(
            [{"application": "1", "title": "t", "abstract": "", "label": "G06F 17/30", "decision": ""}], archive
        )


def test_rows_for_filters_the_table(tmp_path, monkeypatch):
    table = tmp_path / "table.jsonl"
    entries = [
        {"application": "1", "title": "t", "abstract": "a", "label": "G06F1730", "decision": ""},
        {"application": "2", "title": "t", "abstract": "", "label": "G06F 17/30", "decision": ""},
        {"application": "3", "title": "t", "abstract": "a", "label": "", "decision": ""},
        {"application": "4", "title": "t", "abstract": "a", "label": "H04L908", "decision": ""},
    ]
    table.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    monkeypatch.setattr(pc, "compact_table", lambda: table)
    monkeypatch.setattr(pc, "split_of", lambda application: "validation")
    real_rows_for.cache_clear()
    rows = real_rows_for("validation")
    assert rows == (("1", "t", "a", "G06F17/30"), ("4", "t", "a", "H04L9/08")), "resolved through the scheme"


async def test_load_reward_and_metrics():
    (task,) = list(PatentClassificationTaskset(PatentClassificationConfig()).load())
    assert task.key == "cpc:14000001" and "Abstract: A system stores data" in task.data.prompt_text
    assert set(task.data.model_dump()).isdisjoint({"label", "cpc", "gold"})
    assert await task.validate(runtime=None)
    assert await task.hierarchical(make_trace(task, "\\boxed{G06F 17/30}")) == 1.0
    assert await task.hierarchical(make_trace(task, "\\boxed{G06F 17/00}")) == pytest.approx(0.8)
    assert await task.subclass(make_trace(task, "\\boxed{G06F 3/06}")) == 1.0
    assert await task.subclass(make_trace(task, "\\boxed{G06Q 10/00}")) == 0.0
    assert await task.exact(make_trace(task, "\\boxed{G06F17/30}")) == 1.0
    assert await task.formatted(make_trace(task, "no box")) == 0.0
    assert await task.valid_symbol(make_trace(task, "\\boxed{H04L 9/08}")) == 1.0
    assert await task.valid_symbol(make_trace(task, "\\boxed{H04L 9/99}")) == 0.0
