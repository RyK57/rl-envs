"""count-letters: count how often a letter appears in a short passage (single-turn).

The smallest complete v1 taskset: rows are generated procedurally from a seed (no dataset
download, no tools, no sandbox), the model answers once with an integer inside `<answer>`
tags, and the reward is exact match. Two metrics record *why* a rollout scored what it did
(was the answer parseable? how far off was it?), and `validate` checks every row without a
model. Difficulty is a config knob: longer passages mean more letters to keep track of.
"""

import random
import re
from collections.abc import Iterator

import verifiers.v1 as vf

WORDS = (
    "the of and to in is you that it he was for on are as with his they at be this have from "
    "or one had by word but not what all were we when your can said there use an each which "
    "she do how their if will up other about out many then them these so some her would make "
    "like him into time has look two more write go see number no way could people my than "
    "first water been call who oil its now find long down day did get come made may part over "
    "new sound take only little work know place year live me back give most very after thing "
    "our just name good sentence man think say great where help through much before line right "
    "too mean old any same tell boy follow came want show also around form three small set put "
    "end does another well large must big even such because turn here why ask went men read "
    "need land different home us move try kind hand picture again change off play spell air "
    "away animal house point page letter mother answer found study still learn should world"
).split()

ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def parse_answer(text: str) -> int | None:
    """The integer inside the last well-formed `<answer>` tag, or None if there is none."""
    matches = ANSWER_RE.findall(text)
    return int(matches[-1]) if matches else None


class CountLettersData(vf.TaskData):
    text: str
    """The passage the letter is counted in (also embedded in the prompt)."""
    letter: str
    """The letter to count."""
    answer: int
    """The ground-truth count."""


class CountLettersTask(vf.Task[CountLettersData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return float(parse_answer(trace.last_reply) == self.data.answer)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        """1.0 when the reply carries a parseable `<answer>` tag, else 0.0."""
        return float(parse_answer(trace.last_reply) is not None)

    @vf.metric
    async def abs_error(self, trace: vf.Trace) -> float:
        """Distance between the parsed count and the truth; an unparseable reply counts as 0."""
        predicted = parse_answer(trace.last_reply) or 0
        return float(abs(predicted - self.data.answer))

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Valid iff the row is self-consistent and the gold answer, formatted as instructed,
        parses back to itself: the model-free counterpart of the `correct` reward."""
        recount = self.data.text.count(self.data.letter)
        gold_reply = f"<answer>{self.data.answer}</answer>"
        return recount == self.data.answer and parse_answer(gold_reply) == self.data.answer


class CountLettersConfig(vf.TasksetConfig):
    num_tasks: int = 100
    """How many rows to generate."""
    seed: int = 0
    """Seed for the row generator; the same seed always yields the same rows."""
    min_words: int = 8
    """Shortest passage, in words."""
    max_words: int = 24
    """Longest passage, in words."""


class CountLettersTaskset(vf.Taskset[CountLettersTask, CountLettersConfig]):
    def load(self) -> Iterator[CountLettersTask]:
        c = self.config
        assert 1 <= c.min_words <= c.max_words, "need 1 <= min_words <= max_words"
        rng = random.Random(c.seed)
        for i in range(c.num_tasks):
            text = " ".join(rng.choices(WORDS, k=rng.randint(c.min_words, c.max_words)))
            letter = rng.choice(sorted(set(text) - {" "}))
            yield CountLettersTask(
                CountLettersData(
                    idx=i,
                    prompt=(
                        f"How many times does the letter '{letter}' appear in the following text?\n\n"
                        f"{text}\n\n"
                        "Reply with the count as an integer inside <answer></answer> tags."
                    ),
                    text=text,
                    letter=letter,
                    answer=text.count(letter),
                ),
                c.task,
            )
