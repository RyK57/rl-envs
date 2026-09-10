"""Offline checks for number-guess: the taskset, its scoring, and the env loop with a fake agent."""

import pytest
import verifiers.v1 as vf
from number_guess.taskset import (
    NumberGuessConfig,
    NumberGuessData,
    NumberGuessEnv,
    NumberGuessTask,
    NumberGuessTaskset,
    parse_guess,
    prompt_for,
)
from verifiers.v1.graph import MessageNode


def make_trace(task: NumberGuessTask, replies: list[str]) -> vf.Trace:
    """A trace shaped like a real episode: the prompt, then alternating model turns and feedback."""
    nodes = [MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False)]
    for i, reply in enumerate(replies):
        nodes.append(MessageNode(parent=len(nodes) - 1, message=vf.AssistantMessage(content=reply), sampled=True))
        if i < len(replies) - 1:
            nodes.append(MessageNode(parent=len(nodes) - 1, message=vf.UserMessage(content="higher"), sampled=False))
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="NumberGuessTask", data=task.data),
        nodes=nodes,
    )


def task_with(secret: int, max_number: int = 100, max_guesses: int = 7) -> NumberGuessTask:
    return NumberGuessTask(
        NumberGuessData(
            idx=0,
            prompt=prompt_for(max_number, max_guesses),
            secret=secret,
            max_number=max_number,
            max_guesses=max_guesses,
        )
    )


class BinarySearchInteraction:
    """A fake agent that plays perfect binary search and records what the env sent it."""

    def __init__(self, data: NumberGuessData):
        self.lo, self.hi = 1, data.max_number
        self.sent: list[str] = []
        self.guesses: list[int] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def turn(self, message: str | None = None) -> vf.Segment:
        if message is not None:
            self.sent.append(message)
            last = self.guesses[-1]
            if message == "higher":
                self.lo = last + 1
            else:
                self.hi = last - 1
        guess = (self.lo + self.hi) // 2
        self.guesses.append(guess)
        return vf.Segment(messages=[vf.AssistantMessage(content=f"I'll try <guess>{guess}</guess>")])


class FakeAgents:
    def __init__(self, data: NumberGuessData):
        self.interaction_obj = BinarySearchInteraction(data)

    @property
    def agent(self):
        return self

    def interaction(self, task):
        return self.interaction_obj


def test_rows_are_deterministic_and_in_range():
    a = [t.data for t in NumberGuessTaskset(NumberGuessConfig(num_tasks=30, seed=5))]
    b = [t.data for t in NumberGuessTaskset(NumberGuessConfig(num_tasks=30, seed=5))]
    assert a == b
    assert all(1 <= d.secret <= d.max_number and "<guess></guess>" in d.prompt_text for d in a)


def test_parse_guess_takes_the_last_tag():
    assert parse_guess("<guess>50</guess> no wait <guess>75</guess>") == 75
    assert parse_guess("75") is None


async def test_env_loop_answers_until_solved():
    task = task_with(secret=37)
    agents = FakeAgents(task.data)
    await NumberGuessEnv(vf.SingleAgentEnvConfig(taskset=NumberGuessConfig(id="number-guess"))).run(task, agents)
    played = agents.interaction_obj
    assert played.guesses[-1] == 37
    assert len(played.guesses) <= 7
    # one feedback per wrong guess, none after the solve, each one truthful
    assert len(played.sent) == len(played.guesses) - 1
    assert all(msg == ("higher" if g < 37 else "lower") for msg, g in zip(played.sent, played.guesses))


async def test_env_loop_stops_at_the_guess_limit():
    task = task_with(secret=1, max_number=1_000_000, max_guesses=3)
    agents = FakeAgents(task.data)
    await NumberGuessEnv(vf.SingleAgentEnvConfig(taskset=NumberGuessConfig(id="number-guess"))).run(task, agents)
    assert len(agents.interaction_obj.guesses) == 3


async def test_scoring():
    task = task_with(secret=37)
    win = make_trace(task, ["<guess>50</guess>", "<guess>25</guess>", "<guess>37</guess>"])
    lose = make_trace(task, [f"<guess>{g}</guess>" for g in (50, 25, 30, 33, 35, 36, 38)])
    late = make_trace(task, [f"<guess>{g}</guess>" for g in (1, 2, 3, 4, 5, 6, 7, 37)])
    sloppy = make_trace(task, ["<guess>50</guess>", "<guess>80</guess>", "<guess>37</guess>"])
    silent = make_trace(task, ["<guess>50</guess>", "I give up"])

    assert await task.solved(win) == 1.0
    assert await task.solved(lose) == 0.0
    assert await task.solved(late) == 0.0  # the 8th guess is past the limit
    assert await task.solved(silent) == 0.0

    assert await task.guesses_used(win) == 3.0
    assert await task.guesses_used(lose) == 7.0
    assert await task.guesses_used(silent) == 1.0

    assert await task.consistent(win) == 1.0
    assert await task.consistent(sloppy) == 0.0  # 80 after "lower than 50"


async def test_validate_flags_unwinnable_rows():
    assert all([await t.validate(runtime=None) for t in NumberGuessTaskset(NumberGuessConfig(num_tasks=100))])
    hard = NumberGuessTaskset(NumberGuessConfig(num_tasks=100, max_number=1000, max_guesses=7))
    assert not all([await t.validate(runtime=None) for t in hard])


@pytest.mark.parametrize("secret", [1, 50, 100])
async def test_validate_edges(secret: int):
    assert await task_with(secret).validate(runtime=None)
