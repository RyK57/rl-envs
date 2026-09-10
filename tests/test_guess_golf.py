"""Offline checks for guess-golf: the fan-out in `run()` and the sibling comparison in `finalize()`."""

import pytest
import verifiers.v1 as vf
from guess_golf import GuessGolfEnv, GuessGolfEnvConfig
from number_guess.taskset import NumberGuessConfig, NumberGuessTask
from test_number_guess import BinarySearchInteraction, task_with


class FakeAgents:
    """A `guesser` whose every `interaction()` is a fresh perfect binary search."""

    def __init__(self, task: NumberGuessTask):
        self.task = task
        self.games: list[BinarySearchInteraction] = []

    @property
    def guesser(self):
        return self

    def interaction(self, task):
        game = BinarySearchInteraction(self.task.data)
        self.games.append(game)
        return game


def scored_trace(task: NumberGuessTask, solved: bool, guesses_used: int) -> vf.Trace:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="guesser"),
        task=vf.TraceTask(type="NumberGuessTask", data=task.data),
    )
    trace.record_reward("solved", float(solved))
    trace.record_metric("guesses_used", float(guesses_used))
    return trace


@pytest.fixture
def env() -> GuessGolfEnv:
    return GuessGolfEnv(GuessGolfEnvConfig(taskset=NumberGuessConfig(id="number-guess"), attempts=3))


async def test_run_plays_every_attempt(env: GuessGolfEnv):
    task = task_with(secret=37)
    agents = FakeAgents(task)
    await env.run(task, agents)
    assert len(agents.games) == 3
    assert all(game.guesses[-1] == 37 for game in agents.games)


async def test_finalize_pays_the_fewest_guesses_among_solved(env: GuessGolfEnv):
    task = task_with(secret=37)
    traces = [
        scored_trace(task, solved=True, guesses_used=5),
        scored_trace(task, solved=True, guesses_used=3),
        scored_trace(task, solved=False, guesses_used=2),
        scored_trace(task, solved=True, guesses_used=3),
    ]
    await env.finalize(task, vf.Episode(task=vf.TraceTask(type="NumberGuessTask", data=task.data), traces=traces))
    assert [t.rewards["fewest"].score for t in traces] == [0.0, 1.0, 0.0, 1.0]
    assert [t.reward for t in traces] == [1.0, 1.5, 0.0, 1.5]
    assert all(t.metrics["any_solved"] == 1.0 for t in traces)


async def test_finalize_pays_nobody_when_nobody_solved(env: GuessGolfEnv):
    task = task_with(secret=37)
    traces = [scored_trace(task, solved=False, guesses_used=7), scored_trace(task, solved=False, guesses_used=7)]
    await env.finalize(task, vf.Episode(task=vf.TraceTask(type="NumberGuessTask", data=task.data), traces=traces))
    assert [t.rewards["fewest"].score for t in traces] == [0.0, 0.0]
    assert all(t.metrics["any_solved"] == 0.0 for t in traces)
