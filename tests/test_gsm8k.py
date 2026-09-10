"""Offline checks for gsm8k: row parsing, task identity, scoring. No network, no model."""

import verifiers.v1 as vf
from gsm8k.taskset import GSM8KTask, boxed_number, build_task, gold_answer
from verifiers.v1.graph import MessageNode

JANET = {
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18",
}


def make_trace(task: GSM8KTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="GSM8KTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_gold_answer_strips_marker_and_commas():
    assert gold_answer(JANET["answer"]) == "18"
    assert gold_answer("blah\n#### 1,200") == "1200"


def test_build_task_shape_and_identity():
    task = build_task(0, JANET, "test", vf.TaskConfig())
    assert task.data.answer == "18"
    assert task.data.prompt_text.startswith("Janet's ducks") and "\\boxed{}" in task.data.prompt_text
    assert task.key == "gsm8k:test:0"
    assert build_task(0, JANET, "train", vf.TaskConfig()).key == "gsm8k:train:0"


def test_boxed_number_tolerates_money_and_commas():
    assert boxed_number("so \\boxed{$1,200}") == 1200.0
    assert boxed_number("\\boxed{eighteen}") is None
    assert boxed_number("18") is None


async def test_scoring():
    task = build_task(0, JANET, "test", vf.TaskConfig())
    assert await task.correct(make_trace(task, "9 eggs times $2 is $18.\n\n\\boxed{18}")) == 1.0
    assert await task.correct(make_trace(task, "\\boxed{18.0}")) == 1.0
    assert await task.correct(make_trace(task, "\\boxed{\\$18}")) == 1.0
    assert await task.correct(make_trace(task, "\\boxed{16}")) == 0.0
    assert await task.correct(make_trace(task, "The answer is 18.")) == 0.0

    assert await task.boxed(make_trace(task, "\\boxed{16}")) == 1.0
    assert await task.boxed(make_trace(task, "The answer is 18.")) == 0.0
    assert await task.abs_error(make_trace(task, "\\boxed{16}")) == 2.0
    assert await task.abs_error(make_trace(task, "no box")) == 18.0


async def test_validate_accepts_numeric_gold_only():
    assert await build_task(0, JANET, "test", vf.TaskConfig()).validate(runtime=None)
    broken = build_task(1, {"question": "q", "answer": "#### twelve"}, "test", vf.TaskConfig())
    assert not await broken.validate(runtime=None)
