"""Offline checks for summarize: the catalog's structure, sentence counting, judge parsing, and the
reward composition with a faked judge call."""

import asyncio
import json

import pytest
import verifiers.v1 as vf
from summarize.catalog import CATALOG
from summarize.taskset import SummarizeConfig, SummarizeTask, SummarizeTaskset, SummaryJudge, count_sentences
from verifiers.v1.graph import MessageNode


def make_trace(task: SummarizeTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="SummarizeTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


@pytest.fixture
def fake_judge(monkeypatch):
    """Replace the judge's model call with a canned verdict and count the calls."""
    state = {"verdict": {"faithful": True, "covered": 2}, "calls": 0, "prompts": []}

    async def fake_complete(self, messages, *, trace=None, schema=None, parse=None, **sampling):
        state["calls"] += 1
        state["prompts"].append(messages)
        response = vf.JudgeResponse(text=json.dumps(state["verdict"]))
        if parse is not None:
            response.parsed = parse(response)
        return response

    monkeypatch.setattr(vf.Judge, "complete", fake_complete)
    return state


@pytest.fixture
def task() -> SummarizeTask:
    return SummarizeTaskset(SummarizeConfig()).load()[0]


async def test_catalog_is_structurally_valid():
    tasks = SummarizeTaskset(SummarizeConfig()).load()
    assert len(tasks) == len(CATALOG) == 13 and len({t.key for t in tasks}) == 13
    assert all([await t.validate(runtime=None) for t in tasks])
    assert all("at most 2 sentences" in t.data.prompt_text for t in tasks)


def test_count_sentences():
    assert count_sentences("One. Two! Three?") == 3
    assert count_sentences("Just one sentence.") == 1
    assert count_sentences("") == 0
    assert count_sentences("Rehearsals are in St. Anne's hall. Dr. Kim leads them.") == 2


def test_judge_parse_is_strict():
    judge = SummaryJudge()
    assert judge.parse(
        vf.JudgeResponse(text='Sure: {"faithful": false, "covered": 1, "issue": "opens at 7"} done')
    ) == {
        "faithful": False,
        "covered": 1,
        "issue": "opens at 7",
    }
    assert judge.parse(vf.JudgeResponse(text='{"faithful": true, "covered": 3}'))["issue"] == ""
    with pytest.raises(ValueError):
        judge.parse(vf.JudgeResponse(text="looks fine to me"))
    with pytest.raises(ValueError):
        judge.parse(vf.JudgeResponse(text='{"faithful": "yes", "covered": 2}'))


async def test_reward_composition(task: SummarizeTask, fake_judge):
    good = make_trace(
        task, "The library opens earlier on weekdays and ends late fees. The children's wing closes for renovation."
    )
    assert await task.summary(good) == pytest.approx(2 / 3)
    assert await task.faithful(good) == 1.0 and await task.covered(good) == 2.0
    assert fake_judge["calls"] == 1, "one judge call per rollout, shared by reward and metrics"
    assert "Summary:\nThe library opens earlier" in fake_judge["prompts"][0]
    assert good.info["verdict"] == {"faithful": True, "covered": 2, "issue": "", "model": task.config.judge.model}
    assert "judge" not in good.info, "the framework owns info['judge']; the task must not write it"

    fake_judge["verdict"] = {"faithful": False, "covered": 3}
    assert await task.summary(make_trace(task, "The library is closing forever. Nothing else changes.")) == 0.0

    fake_judge["verdict"] = {"faithful": True, "covered": 7}
    assert await task.covered(make_trace(task, "Everything. Everything.")) == 3.0, (
        "covered is clipped to the key point count"
    )


async def test_over_length_scores_zero_without_calling_the_judge(task: SummarizeTask, fake_judge):
    assert await task.summary(make_trace(task, "One. Two. Three.")) == 0.0
    assert await task.summary(make_trace(task, "")) == 0.0
    assert fake_judge["calls"] == 0


async def test_concurrent_hooks_share_one_judge_call(task: SummarizeTask, fake_judge):
    trace = make_trace(task, "The library opens earlier and ends late fees. The children's wing closes for renovation.")
    results = await asyncio.gather(task.summary(trace), task.faithful(trace), task.covered(trace))
    assert results == [pytest.approx(2 / 3), 1.0, 2.0]
    assert fake_judge["calls"] == 1
