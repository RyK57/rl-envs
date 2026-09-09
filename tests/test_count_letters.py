"""Offline checks for count-letters: no model, no runtime — just the taskset and its scoring.

A `vf.Trace` is a message graph; scoring only reads it. So a reward can be exercised by
building a trace with the user prompt and a hand-written assistant reply, exactly the shape a
real rollout would leave behind.
"""

import pytest
import verifiers.v1 as vf
from count_letters.taskset import CountLettersConfig, CountLettersTask, CountLettersTaskset, parse_answer
from verifiers.v1.graph import MessageNode


def make_trace(task: CountLettersTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="CountLettersTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


@pytest.fixture
def task() -> CountLettersTask:
    return next(iter(CountLettersTaskset(CountLettersConfig(num_tasks=1))))


def test_rows_are_deterministic_and_consistent():
    a = [t.data for t in CountLettersTaskset(CountLettersConfig(num_tasks=20, seed=7))]
    b = [t.data for t in CountLettersTaskset(CountLettersConfig(num_tasks=20, seed=7))]
    assert a == b
    assert all(d.text.count(d.letter) == d.answer >= 1 for d in a)
    assert all(d.letter in d.prompt_text and d.text in d.prompt_text for d in a)


def test_parse_answer_takes_the_last_tag():
    assert parse_answer("<answer>3</answer> wait, <answer> 5 </answer>") == 5
    assert parse_answer("five") is None
    assert parse_answer("<answer>five</answer>") is None


async def test_scoring(task: CountLettersTask):
    gold = task.data.answer
    correct = make_trace(task, f"I count {gold}.\n\n<answer>{gold}</answer>")
    off_by_two = make_trace(task, f"<answer>{gold + 2}</answer>")
    unformatted = make_trace(task, f"The answer is {gold}.")

    assert await task.correct(correct) == 1.0
    assert await task.correct(off_by_two) == 0.0
    assert await task.correct(unformatted) == 0.0

    assert await task.formatted(correct) == 1.0
    assert await task.formatted(unformatted) == 0.0
    assert await task.abs_error(off_by_two) == 2.0
    assert await task.abs_error(unformatted) == float(gold)

    assert await task.single_turn(correct) is True


async def test_validate_accepts_every_generated_row():
    tasks = list(CountLettersTaskset(CountLettersConfig(num_tasks=50, seed=3)))
    assert all([await t.validate(runtime=None) for t in tasks])
