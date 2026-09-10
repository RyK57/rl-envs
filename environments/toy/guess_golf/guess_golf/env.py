"""guess-golf: several games of one number-guess task, scored against each other.

Environment six. The new mechanism is an `Env` whose reward depends on the other attempts in
the episode. `run()` plays `--env.attempts` independent games of the same task concurrently,
each driven by the higher/lower loop from `number-guess`; `finalize()` then sees every finished
game at once and pays `fewest` to the solved game that used the fewest guesses (ties share). A
game that failed earns nothing on the comparison, and a group where nobody solved it pays nobody.
No per-trace reward can compute that: it exists only at the episode level. The agent role is
`guesser`, so its flags are `--env.guesser.*`. Pattern: `verifiers/environments/code_golf`.
"""

import asyncio

import verifiers.v1 as vf
from number_guess.taskset import play
from pydantic import Field


class GuessGolfEnvConfig(vf.EnvConfig):
    guesser: vf.AgentConfig = vf.AgentConfig()
    attempts: int = Field(4, ge=1)
    """Independent games per episode, scored against each other."""


class GuessGolfEnv(vf.Env[GuessGolfEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with asyncio.TaskGroup() as tg:
            for _ in range(self.config.attempts):
                tg.create_task(play(task, agents.guesser))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Runs after every game is scored: `solved` and `guesses_used` are already on the traces."""
        solved = [t for t in episode.traces if t.rewards["solved"].score]
        fewest = min((t.metrics["guesses_used"] for t in solved), default=None)
        for trace in episode.traces:
            won = trace in solved and trace.metrics["guesses_used"] == fewest
            trace.record_reward("fewest", float(won), 0.5)
            trace.record_metric("any_solved", float(bool(solved)))
