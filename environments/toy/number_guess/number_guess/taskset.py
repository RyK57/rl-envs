"""number-guess: find a secret integer from "higher"/"lower" feedback (multi-turn, env-driven).

Environment two. The new mechanism is control flow: the task carries the opening prompt, the
model guesses, and the exported `NumberGuessEnv` answers "higher" or "lower" after every guess
by driving the interaction turn by turn. The model never sees the secret. The reward is binary
(solved within the guess limit); two metrics record how many guesses were used and whether
every guess respected the feedback before it. Rows are seeded; `max_number` and `max_guesses`
are the difficulty knobs. The defaults, 1..5000 in 13 guesses, are the smallest range at which
deepseek-v4-flash starts to slip while tracking the interval; 1..100 in 7 is saturated for it.
"""

import random
import re
from collections.abc import Iterator

import verifiers.v1 as vf

GUESS_RE = re.compile(r"<guess>\s*(-?\d+)\s*</guess>")


def parse_guess(text: str) -> int | None:
    """The integer inside the last well-formed `<guess>` tag, or None if there is none."""
    matches = GUESS_RE.findall(text or "")
    return int(matches[-1]) if matches else None


def guesses(trace: vf.Trace) -> list[int]:
    """Every model turn's guess in order, stopping at the first turn without one."""
    out: list[int] = []
    for message in trace.assistant_messages:
        guess = parse_guess(message.content or "")
        if guess is None:
            break
        out.append(guess)
    return out


def prompt_for(max_number: int, max_guesses: int) -> str:
    return (
        f"I'm thinking of a whole number between 1 and {max_number}. Guess it. After each guess I will "
        f'reply "higher" or "lower". You have {max_guesses} guesses. Reply with each guess as a single '
        "integer inside <guess></guess> tags."
    )


class NumberGuessData(vf.TaskData):
    secret: int
    """The number to find. Never shown to the model."""
    max_number: int
    """The range is 1..max_number."""
    max_guesses: int
    """Guesses allowed before the episode ends."""


class NumberGuessTask(vf.Task[NumberGuessData]):
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(self.data.secret in guesses(trace)[: self.data.max_guesses])

    @vf.metric
    async def guesses_used(self, trace: vf.Trace) -> float:
        """Guesses up to and including the solve, or all of them when unsolved."""
        used = guesses(trace)[: self.data.max_guesses]
        return float(used.index(self.data.secret) + 1) if self.data.secret in used else float(len(used))

    @vf.metric
    async def consistent(self, trace: vf.Trace) -> float:
        """1.0 when every guess stayed inside the interval implied by all feedback before it."""
        lo, hi = 1, self.data.max_number
        for guess in guesses(trace)[: self.data.max_guesses]:
            if not lo <= guess <= hi:
                return 0.0
            if guess < self.data.secret:
                lo = guess + 1
            elif guess > self.data.secret:
                hi = guess - 1
            else:
                break
        return 1.0

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Valid iff the secret is in range and a midpoint binary search finds it within the
        guess limit, so every row is winnable and a zero reward is always the model's doing."""
        d = self.data
        if not 1 <= d.secret <= d.max_number:
            return False
        lo, hi, steps = 1, d.max_number, 0
        while lo <= hi:
            steps += 1
            mid = (lo + hi) // 2
            if mid == d.secret:
                return steps <= d.max_guesses
            if mid < d.secret:
                lo = mid + 1
            else:
                hi = mid - 1
        return False


class NumberGuessEnv(vf.SingleAgentEnv):
    """Plays the game: answers each guess with "higher" or "lower" until it is solved, the
    guesses run out, or a turn carries no guess."""

    async def run(self, task, agents):
        data = task.data
        async with agents.agent.interaction(task) as interaction:
            # The task is prompted, so the model guesses first.
            segment = await interaction.turn()
            for n in range(1, data.max_guesses + 1):
                if segment.terminated:
                    break
                guess = parse_guess(segment.last_reply)
                if guess is None or guess == data.secret or n == data.max_guesses:
                    break
                segment = await interaction.turn("higher" if guess < data.secret else "lower")


class NumberGuessConfig(vf.TasksetConfig):
    num_tasks: int = 100
    """How many rows to generate."""
    seed: int = 0
    """Seed for the secrets; the same seed always yields the same rows."""
    max_number: int = 5000
    """Secrets are drawn from 1..max_number."""
    max_guesses: int = 13
    """Guesses allowed per episode."""


class NumberGuessTaskset(vf.Taskset[NumberGuessTask, NumberGuessConfig]):
    def load(self) -> Iterator[NumberGuessTask]:
        c = self.config
        assert c.max_number >= 1 and c.max_guesses >= 1, "need max_number >= 1 and max_guesses >= 1"
        rng = random.Random(c.seed)
        for i in range(c.num_tasks):
            yield NumberGuessTask(
                NumberGuessData(
                    idx=i,
                    prompt=prompt_for(c.max_number, c.max_guesses),
                    secret=rng.randint(1, c.max_number),
                    max_number=c.max_number,
                    max_guesses=c.max_guesses,
                ),
                c.task,
            )
