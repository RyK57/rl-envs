"""Offline checks for sanctions-screening: parsing, rendering, the balanced reader, and the hooks."""

import pytest
import verifiers.v1 as vf
from sanctions_screening import taskset as ss
from sanctions_screening.taskset import (
    SanctionsScreeningConfig,
    SanctionsScreeningTask,
    SanctionsScreeningTaskset,
    balanced_pairs,
    pair_id,
    parse_verdict,
    prompt_for,
    render_record,
)
from verifiers.v1.graph import MessageNode


def entity(id: str, caption: str, **properties) -> dict:
    return {"id": id, "caption": caption, "schema": "Person", "properties": {k: list(v) for k, v in properties.items()}}


SAME = {
    "left": entity(
        "ofac-1", "Aliasghar Norouzi", birthDate=["1962-11-11"], nationality=["ir"], sourceUrl=["https://x"]
    ),
    "right": entity("gb-1", "Ali Asghar Nourouzi", birthDate=["1962-11-11"], passportNumber=["Y53914915"]),
    "judgement": "positive",
}
DIFFERENT = {
    "left": entity("eu-2", "Viktor Zubkov", birthDate=["1941-09-15"]),
    "right": entity("ru-2", "Viktor Zubkov", birthDate=["1975-03-02"]),
    "judgement": "negative",
}
ROWS = (SAME, DIFFERENT)


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    monkeypatch.setattr(ss, "rows_for", lambda split, size, scan_limit: ROWS)


def make_trace(task: SanctionsScreeningTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="SanctionsScreeningTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_parse_verdict_reads_the_last_box():
    assert parse_verdict("I think \\boxed{different}... no, \\boxed{same}") == "same"
    assert parse_verdict("\\boxed{Same.}") == "same"
    assert parse_verdict("\\boxed{Different}") == "different"
    assert parse_verdict("\\boxed{maybe}") is None and parse_verdict("same") is None


def test_render_record_shows_identity_and_hides_provenance():
    text = render_record(SAME["left"])
    assert text.startswith("Type: Person\nName: Aliasghar Norouzi")
    assert "Born: 1962-11-11" in text and "Nationality: ir" in text
    assert "https://x" not in text and "sourceUrl" not in text


def test_prompt_has_both_records_and_the_answer_form():
    prompt = prompt_for(SAME["left"], SAME["right"])
    assert "Record A" in prompt and "Record B" in prompt and "Passport: Y53914915" in prompt
    assert "\\boxed{same}" in prompt and "\\boxed{different}" in prompt


async def test_load_keys_and_hides_verdict():
    tasks = list(SanctionsScreeningTaskset(SanctionsScreeningConfig()).load())
    assert [t.key for t in tasks] == ["sanctions:validation:ofac-1|gb-1", "sanctions:validation:eu-2|ru-2"]
    assert set(tasks[0].data.model_dump()).isdisjoint({"judgement", "verdict", "gold"})
    assert all([await t.validate(runtime=None) for t in tasks])


async def test_reward_and_metrics():
    same, different = list(SanctionsScreeningTaskset(SanctionsScreeningConfig()).load())
    assert await same.verdict(make_trace(same, "\\boxed{same}")) == 1.0
    assert await same.verdict(make_trace(same, "\\boxed{different}")) == 0.0
    assert await same.false_clear(make_trace(same, "\\boxed{different}")) == 1.0
    assert await same.false_clear(make_trace(same, "no box")) == 0.0
    assert await same.formatted(make_trace(same, "no box")) == 0.0
    assert await different.verdict(make_trace(different, "\\boxed{different}")) == 1.0
    assert await different.false_hit(make_trace(different, "\\boxed{same}")) == 1.0
    assert await different.false_hit(make_trace(different, "\\boxed{different}")) == 0.0


def test_balanced_pairs_interleaves_and_skips_the_sample(monkeypatch):
    corpus = (
        [SAME, DIFFERENT]
        + [{**SAME, "left": entity(f"p{i}", "P"), "right": entity(f"q{i}", "Q")} for i in range(4)]
        + [{**DIFFERENT, "left": entity(f"n{i}", "N"), "right": entity(f"m{i}", "M")} for i in range(4)]
    )
    monkeypatch.setattr(ss, "stream_pairs", lambda: iter(corpus))
    monkeypatch.setattr(ss, "sample_pairs", lambda: (SAME, DIFFERENT))
    rows = balanced_pairs(4, scan_limit=100)
    assert [ss.verdict_of(p) for p in rows] == ["same", "different", "same", "different"]
    assert pair_id(SAME) not in {pair_id(p) for p in rows}
    with pytest.raises(ValueError, match="short of a balanced set"):
        balanced_pairs(4, scan_limit=3)
