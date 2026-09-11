"""Offline checks for contract-review: parsing, the F1 score, excerpt placement, and both modes' hooks."""

import pytest
import verifiers.v1 as vf
from contract_review import taskset as cr
from contract_review.categories import CATEGORIES
from contract_review.taskset import (
    ContractReviewConfig,
    ContractReviewTask,
    ContractReviewTaskset,
    Passage,
    excerpt_start,
    normalize,
    parse_category,
    parse_clause,
    passages_of,
    split_of,
    token_f1,
)
from verifiers.v1.graph import MessageNode

CLAUSES = (
    ("This Agreement shall be governed by the laws of the State of Nevada.", ("Governing Law",)),
    (
        "Licensor grants Licensee a perpetual, irrevocable license.",
        ("License Grant", "Irrevocable Or Perpetual License"),
    ),
)
CONTRACT = "PREAMBLE. " * 20 + "This Agreement shall be governed by the laws of the State of Nevada. " + "FILLER. " * 40
GOLD = "This Agreement shall be governed by the laws of the State of Nevada."
PASSAGES = (
    Passage("Acme#0", "Governing Law", CATEGORIES["Governing Law"], 0, (GOLD,)),
    Passage("Acme#0", "Insurance", CATEGORIES["Insurance"], 0, ()),
)


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    monkeypatch.setattr(cr, "clause_rows", lambda split: CLAUSES)
    monkeypatch.setattr(cr, "passage_rows", lambda split, window: PASSAGES)
    monkeypatch.setattr(cr, "documents", lambda: {"Acme#0": CONTRACT})


def make_trace(task: ContractReviewTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="ContractReviewTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_normalize_and_f1():
    assert normalize("The Governing Law, of a State!") == "governing law of state"
    assert token_f1("governed by the laws of Nevada", "governed by the laws of Nevada") == 1.0
    assert token_f1("laws of Nevada", "governed by the laws of Nevada") == pytest.approx(0.75)
    assert token_f1("unrelated words", "governed by the laws of Nevada") == 0.0


def test_parse_category_maps_spelling_variants_to_canonical_labels():
    assert parse_category("\\boxed{governing law}") == "Governing Law"
    assert parse_category("\\boxed{ROFR/ROFO/ROFN}") == "Rofr/Rofo/Rofn"
    assert parse_category("\\boxed{Insurance} then \\boxed{Audit Rights}") == "Audit Rights"
    assert parse_category("\\boxed{Something Else}") is None and parse_category("Governing Law") is None


def test_parse_clause():
    assert parse_clause("<clause>  the quote </clause>") == "the quote"
    assert parse_clause("<clause>None</clause>") == "none"
    assert parse_clause("<clause>a</clause> <clause>b</clause>") == "b"
    assert parse_clause("no tags") is None


def test_split_is_deterministic_and_covers_all_three():
    names = [f"contract-{i}.pdf" for i in range(300)]
    splits = {split_of(n) for n in names}
    assert splits == {"train", "validation", "test"}
    assert split_of("x.pdf") == split_of("x.pdf")


def test_excerpt_start_keeps_the_span_inside_the_window():
    text = "a" * 10_000
    for span_start in (0, 100, 4_000, 9_950):
        start = excerpt_start("doc", "Insurance", text, 3000, (span_start, 40))
        assert start <= span_start and span_start + 40 <= start + 3000
    assert 0 <= excerpt_start("doc", "Insurance", text, 3000, None) <= 7000
    assert excerpt_start("short", "Insurance", "abc", 3000, (0, 3)) == 0


def test_passages_of_builds_positives_and_negatives():
    start = CONTRACT.index(GOLD)
    paragraph = {
        "qas": [
            {
                "question": 'Highlight the parts related to "Governing Law". Details: Which law governs',
                "answers": [{"text": GOLD, "answer_start": start}],
            },
            {"question": 'Highlight the parts related to "Insurance". Details: Any insurance', "answers": []},
        ]
    }
    positive, negative = passages_of("Acme#0", CONTRACT, paragraph, 200)
    assert positive.category == "Governing Law" and positive.description == CATEGORIES["Governing Law"]
    assert positive.golds and GOLD.startswith(positive.golds[0][:20])
    assert CONTRACT[positive.start : positive.start + 200].find(positive.golds[0]) >= 0
    assert negative.category == "Insurance" and negative.golds == ()


async def test_classify_mode():
    tasks = list(ContractReviewTaskset(ContractReviewConfig()).load())
    assert [t.key for t in tasks] == ["contract:classify:validation:0", "contract:classify:validation:1"]
    assert "- Governing Law: " in tasks[0].data.prompt_text and "State of Nevada" in tasks[0].data.prompt_text
    assert set(tasks[0].data.model_dump()).isdisjoint({"label", "labels", "gold"})
    assert all([await t.validate(runtime=None) for t in tasks])
    law, licence = tasks
    assert await law.match(make_trace(law, "\\boxed{Governing Law}")) == 1.0
    assert await law.match(make_trace(law, "\\boxed{Insurance}")) == 0.0
    assert await law.formatted(make_trace(law, "Governing Law")) == 0.0
    assert await licence.match(make_trace(licence, "\\boxed{License Grant}")) == 1.0
    assert await licence.match(make_trace(licence, "\\boxed{Irrevocable or Perpetual License}")) == 1.0


async def test_extract_mode():
    tasks = list(ContractReviewTaskset(ContractReviewConfig(mode="extract", window=3000)).load())
    assert [t.key for t in tasks] == [
        "contract:extract:validation:Acme#0:Governing Law",
        "contract:extract:validation:Acme#0:Insurance",
    ]
    positive, negative = tasks
    assert '"Governing Law"' in positive.data.prompt_text and "PREAMBLE." in positive.data.prompt_text
    assert all([await t.validate(runtime=None) for t in tasks])
    assert await positive.match(make_trace(positive, f"<clause>{GOLD}</clause>")) == 1.0
    assert await positive.exact(make_trace(positive, f"<clause>{GOLD}</clause>")) == 1.0
    partial = await positive.match(make_trace(positive, "<clause>governed by the laws of the State of Nevada</clause>"))
    assert 0.5 < partial < 1.0
    assert await positive.match(make_trace(positive, "<clause>none</clause>")) == 0.0
    assert await positive.match(make_trace(positive, "no tags")) == 0.0
    assert await negative.match(make_trace(negative, "<clause>none</clause>")) == 1.0
    assert await negative.abstained(make_trace(negative, "<clause>none</clause>")) == 1.0
    assert await negative.match(make_trace(negative, "<clause>FILLER.</clause>")) == 0.0
    only_positives = list(ContractReviewTaskset(ContractReviewConfig(mode="extract", negatives=False)).load())
    assert [t.data.category for t in only_positives] == ["Governing Law"]
