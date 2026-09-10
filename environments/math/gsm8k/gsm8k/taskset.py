"""gsm8k: grade-school math word problems from a pinned Hugging Face dataset (single-turn).

Environment three. The new mechanism is data: rows come from `openai/gsm8k` at a pinned
revision through a lazy `load()` generator, so `-n 3` builds three tasks and the same revision
always yields the same rows. The model reasons and puts its final answer inside `\\boxed{}`;
the reward is math equivalence against the gold number through `vf.verify_boxed_math_answer`,
the scorer prime-envs uses for MATH-500, so `18`, `18.0` and `$18` all count.
"""

from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
INSTRUCTION = "\n\nReason step by step, then put your final answer as a number inside \\boxed{}."


def gold_answer(answer_field: str) -> str:
    """The number after GSM8K's `####` marker, without thousands separators."""
    return answer_field.rsplit("####", 1)[-1].strip().replace(",", "")


def boxed_number(text: str) -> float | None:
    """The boxed answer as a number, or None when there is no box or it is not numeric."""
    boxed = vf.extract_boxed_answer(text or "", strict=True)
    cleaned = boxed.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


class GSM8KData(vf.TaskData):
    split: str
    """Dataset split the row came from; part of the task's stable identity."""
    answer: str
    """The gold number as text (GSM8K's value after `####`)."""


class GSM8KTask(vf.Task[GSM8KData]):
    @property
    def key(self) -> str:
        return f"gsm8k:{self.data.split}:{self.data.idx}"

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(trace.last_reply, self.data.answer)

    @vf.metric
    async def boxed(self, trace: vf.Trace) -> float:
        """1.0 when the reply carries a well-formed `\\boxed{}`."""
        return float(bool(vf.extract_boxed_answer(trace.last_reply or "", strict=True)))

    @vf.metric
    async def abs_error(self, trace: vf.Trace) -> float:
        """Distance between the boxed number and the gold; a missing or non-numeric box counts as 0."""
        predicted = boxed_number(trace.last_reply) or 0.0
        return abs(predicted - float(self.data.answer))

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Valid iff the gold is a number and, boxed as instructed, scores 1.0 through the reward."""
        try:
            float(self.data.answer)
        except ValueError:
            return False
        return vf.verify_boxed_math_answer(f"\\boxed{{{self.data.answer}}}", self.data.answer) == 1.0


class GSM8KConfig(vf.TasksetConfig):
    split: Literal["train", "test"] = "test"
    """`test` has 1319 problems, `train` 7473."""


def build_task(idx: int, row: dict, split: str, config: vf.TaskConfig) -> GSM8KTask:
    return GSM8KTask(
        GSM8KData(
            idx=idx, split=split, prompt=row["question"].strip() + INSTRUCTION, answer=gold_answer(row["answer"])
        ),
        config,
    )


class GSM8KTaskset(vf.Taskset[GSM8KTask, GSM8KConfig]):
    def load(self) -> Iterator[GSM8KTask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=self.config.split, revision=DATASET_REVISION)
        for idx, row in enumerate(rows):
            yield build_task(idx, row, self.config.split, self.config.task)
