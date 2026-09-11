"""Offline checks for commercial-underwriting: amount parsing, the rules, synthetic cases, and the hooks."""

import random

import pytest
import verifiers.v1 as vf
from commercial_underwriting import taskset as uw
from commercial_underwriting.rulebook import GUIDELINES, INDUSTRIES, deductible_for, limits_for
from commercial_underwriting.taskset import (
    Case,
    CommercialUnderwritingConfig,
    CommercialUnderwritingTask,
    CommercialUnderwritingTaskset,
    amounts_in,
    canonical,
    matched_naics,
    parse_answer,
    synthetic_case,
)
from verifiers.v1.graph import MessageNode


def case(id: str, task: str, gold: str, lob: str = "cyber") -> Case:
    return Case(
        id,
        task,
        "Delta HVAC Supply",
        "Delta HVAC Supply distributes heating equipment.",
        "Georgia",
        3_700_000,
        27,
        980_000,
        7,
        "Non-combustible",
        lob,
        gold,
    )


ROWS = (
    case("1097", "limits", "1000000 2000000", lob="general liability"),
    case("731", "deductibles", "5000", lob="auto"),
    case("522", "classification", "522292"),
)


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    monkeypatch.setattr(uw, "rows_for", lambda source, split, size, seed: ROWS)


def make_trace(task: CommercialUnderwritingTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="CommercialUnderwritingTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_amounts_in_reads_every_dollar_form_once():
    assert amounts_in("You should offer a $5000.0 deductible.") == (5000,)
    assert amounts_in("$1M per occurrence / $2M aggregate") == (1_000_000, 2_000_000)
    assert amounts_in("$3 million per-occurence and $5 million aggregate") == (3_000_000, 5_000_000)
    assert amounts_in("limits of 1 million and 2 million") == (1_000_000, 2_000_000)
    assert amounts_in("$1,000,000 and $2,000,000") == (1_000_000, 2_000_000)
    assert amounts_in("27 employees and 7 vehicles") == ()
    assert amounts_in("1000000 2000000", bare=True) == (1_000_000, 2_000_000)


def test_canonical_and_parse_answer():
    assert canonical("limits", "$1M per occurrence, $2M aggregate") == "1000000 2000000"
    assert canonical("limits", "$1M only") is None
    assert canonical("deductibles", "You should offer a $500 deductible.") == "500"
    assert canonical("classification", "The company's NAICS code is 522292.") == "522292"
    assert canonical("classification", "code 5222") is None
    assert parse_answer("limits", "\\boxed{$1,000,000 per occurrence, $2,000,000 aggregate}") == "1000000 2000000"
    assert parse_answer("limits", "\\boxed{1000000 / 2000000}") == "1000000 2000000"
    assert parse_answer("deductibles", "\\boxed{$1,000}") == "1000"
    assert parse_answer("classification", "\\boxed{522292}") == "522292"
    assert parse_answer("deductibles", "$500") is None


def test_matched_naics_scores_by_level():
    assert matched_naics("522292", "522292") == 1.0
    assert matched_naics("522291", "522292") == pytest.approx(5 / 6)
    assert matched_naics("522110", "522292") == pytest.approx(3 / 6)
    assert matched_naics("523150", "522292") == pytest.approx(2 / 6)
    assert matched_naics("611210", "522292") == 0.0


def test_rules_follow_the_guidelines():
    assert limits_for("522292", "cyber") == (3_000_000, 5_000_000)
    assert limits_for("522292", "property") == (1_000_000, 2_000_000)
    assert limits_for("722511", "cyber") == (1_000_000, 2_000_000)
    assert deductible_for("485510", "auto") == 5000
    assert deductible_for("722511", "auto") == 1000
    assert deductible_for("485510", "property") == 500
    assert len({ind.naics for ind in INDUSTRIES}) == len(INDUSTRIES)


def test_synthetic_cases_are_deterministic_and_rule_consistent():
    rng, again_rng = random.Random(7), random.Random(7)
    cases = [synthetic_case(i, rng) for i in range(60)]
    again = [synthetic_case(i, again_rng) for i in range(60)]
    assert cases == again
    assert len({c.description for c in cases}) > 50
    assert {c.task for c in cases} == {"limits", "deductibles"}
    for c in cases:
        industry = next(ind for ind in INDUSTRIES if ind.phrase in c.description)
        expected = limits_for(industry.naics, c.lob) if c.task == "limits" else (deductible_for(industry.naics, c.lob),)
        assert c.gold == " ".join(str(a) for a in expected)
    assert {c.gold for c in cases if c.task == "limits"} == {"1000000 2000000", "3000000 5000000"}
    assert {c.gold for c in cases if c.task == "deductibles"} == {"500", "1000", "5000"}


async def test_load_prompts_keys_and_hidden_gold():
    tasks = list(CommercialUnderwritingTaskset(CommercialUnderwritingConfig()).load())
    assert [t.key for t in tasks] == [
        "underwriting:snorkel:1097",
        "underwriting:snorkel:731",
        "underwriting:snorkel:522",
    ]
    limits, deductible, naics = tasks
    assert (
        GUIDELINES.splitlines()[0] in limits.data.prompt_text
        and "Line of business: general liability" in limits.data.prompt_text
    )
    assert "Underwriting guidelines" not in naics.data.prompt_text and "six-digit NAICS" in naics.data.prompt_text
    assert set(limits.data.model_dump()).isdisjoint({"gold", "reference", "answer"})
    assert all([await t.validate(runtime=None) for t in tasks])
    assert await limits.answer(make_trace(limits, "\\boxed{$1,000,000 per occurrence, $2,000,000 aggregate}")) == 1.0
    assert await limits.answer(make_trace(limits, "\\boxed{$3M / $5M}")) == 0.0
    assert await limits.formatted(make_trace(limits, "\\boxed{$1M}")) == 0.0
    assert await deductible.answer(make_trace(deductible, "\\boxed{$5,000}")) == 1.0
    assert await deductible.exact(make_trace(deductible, "\\boxed{$1000}")) == 0.0
    assert await naics.answer(make_trace(naics, "\\boxed{522110}")) == pytest.approx(0.5)
    assert await naics.exact(make_trace(naics, "\\boxed{522292}")) == 1.0
