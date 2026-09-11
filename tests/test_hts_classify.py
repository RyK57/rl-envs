"""Offline checks for hts-classify: parsing, the hierarchical score, and the hooks with fixture rows."""

import pytest
import verifiers.v1 as vf
from hts_classify import taskset as hts
from hts_classify.taskset import (
    HtsClassifyConfig,
    HtsClassifyData,
    HtsClassifyTask,
    HtsClassifyTaskset,
    describe_listing,
    matched_digits,
    parse_code,
    parse_ruling,
    prompt_for,
)
from verifiers.v1.graph import MessageNode

RULING = [
    {"role": "user", "content": "What is the HTS US Code for cast carbon steel fittings made to ASTM A216?"},
    {"role": "assistant", "content": "HTS US Code -> 7307.19.9060\nReasoning -> The product is classified under..."},
]
LISTING = {
    "product_name": "Green Dragon Magnetic Pendant",
    "product_attributes": '{"Material": "Metal", "Metals Type": "Copper"}',
    "price": 141.0,
    "currency_code": "CNY",
    "cate_lv1_desc": "Jewelry & Accessories",
    "cate_lv2_desc": "Fashion Jewelry",
    "cate_lv3_desc": "Pendants",
    "cate_lv4_desc": "Pendants",
    "cate_lv5_desc": "Pendants",
    "hs_code": 7117199000,
}
ROWS = (("cast carbon steel fittings made to ASTM A216", "7307199060"), ("a copper pendant", "7117199000"))


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    """No network: the two fixture rows stand in for the dataset, and a tiny nomenclature for HS6."""
    monkeypatch.setattr(hts, "rows_for", lambda source, split: ROWS)
    monkeypatch.setattr(hts, "hs6_codes", lambda: frozenset({"730719", "711719"}))


def make_trace(task: HtsClassifyTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HtsClassifyTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_parse_code_takes_the_last_boxed_ten_digit_code():
    assert parse_code("The code is \\boxed{7307.19.9060}.") == "7307199060"
    assert parse_code("\\boxed{7307199060}") == "7307199060"
    assert parse_code("\\boxed{1111111111} no, \\boxed{7307 19 9060}") == "7307199060"
    assert parse_code("\\boxed{73071990}") is None, "eight digits is not a US code"
    assert parse_code("7307.19.9060 without a box") is None


def test_matched_digits_counts_whole_hs_levels():
    gold = "7307199060"
    assert matched_digits("7307199060", gold) == 10
    assert matched_digits("7307199061", gold) == 8, "nine shared digits count as the eight-digit level"
    assert matched_digits("7307190000", gold) == 6
    assert matched_digits("7307910000", gold) == 4
    assert matched_digits("7308000000", gold) == 2
    assert matched_digits("8307199060", gold) == 0
    assert matched_digits(None, gold) == 0


def test_parse_ruling_and_describe_listing():
    assert parse_ruling(RULING) == ("cast carbon steel fittings made to ASTM A216", "7307199060")
    assert parse_ruling([{"role": "user", "content": "unrelated"}]) is None
    text = describe_listing(LISTING)
    assert text.startswith("Title: Green Dragon Magnetic Pendant")
    assert "Listed under: Jewelry & Accessories > Fashion Jewelry > Pendants" in text
    assert "Material: Metal; Metals Type: Copper" in text and "Price: 141.0 CNY" in text


def test_prompt_mentions_the_rulebook_only_when_enabled():
    assert "hs_nomenclature.txt" not in prompt_for("a pendant", with_rulebook=False)
    assert "hs_nomenclature.txt" in prompt_for("a pendant", with_rulebook=True)
    assert "boxed" in prompt_for("a pendant", with_rulebook=False)


async def test_load_keys_and_hides_gold():
    tasks = list(HtsClassifyTaskset(HtsClassifyConfig()).load())
    assert [t.key for t in tasks] == ["hts:cross:validation:0", "hts:cross:validation:1"]
    assert "cast carbon steel fittings" in tasks[0].data.prompt_text
    assert set(tasks[0].data.model_dump()).isdisjoint({"gold", "code", "hs_code"})
    assert all([await t.validate(runtime=None) for t in tasks])


async def test_reward_and_metrics():
    task = list(HtsClassifyTaskset(HtsClassifyConfig()).load())[0]
    exact = make_trace(task, "\\boxed{7307.19.9060}")
    subheading = make_trace(task, "\\boxed{7307.19.0000}")
    wrong = make_trace(task, "\\boxed{8307199060}")
    unformatted = make_trace(task, "7307199060")
    assert await task.hierarchical(exact) == 1.0 and await task.exact(exact) == 1.0
    assert await task.hierarchical(subheading) == pytest.approx(0.6)
    assert await task.hs6(subheading) == 1.0 and await task.exact(subheading) == 0.0
    assert await task.hierarchical(wrong) == 0.0 and await task.heading(wrong) == 0.0
    assert await task.formatted(unformatted) == 0.0 and await task.hierarchical(unformatted) == 0.0
    assert await task.valid_hs6(exact) == 1.0 and await task.valid_hs6(wrong) == 0.0


def test_data_shape():
    data = HtsClassifyData(idx=0, prompt="p", source="cross", split="validation", row=0)
    assert data.row == 0 and data.source == "cross"
