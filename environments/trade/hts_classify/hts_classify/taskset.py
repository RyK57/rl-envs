"""hts-classify: classify a product for import into the United States to its 10-digit HTS code.

Environment seven, the first on a real vertical. The rows are real: US customs rulings from CBP's
CROSS system, as packaged by Flexify.AI (product description and the code the ruling assigned),
and, as a second and harder test set, the 632 expert-labeled e-commerce listings of HSCodeComp.
The reward is deterministic and hierarchical: the number of leading digits the predicted code
shares with the gold code, out of ten, so the right chapter earns 0.2, the right heading 0.4, the
right subheading 0.6 and the full US code 1.0. Gold codes stay off `TaskData`; they are looked up by
key at scoring time from the same pinned rows. `validate()` checks that every gold code has ten
digits and that its six-digit prefix exists in the HS nomenclature.
"""

import json
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import verifiers.v1 as vf

CROSS_DATASET = "Dayanand314Krishna/cross_rulings_hts_dataset_for_tariffs"
HSCODECOMP_DATASET = "ATH-MaaS/HSCodeComp"
NOMENCLATURE_DATASET = "ronnieaban/hs-code"
REVISIONS: dict[str, str] = {
    CROSS_DATASET: "0a66ad345d3fc9828f362836d86b31e20798513c",
    HSCODECOMP_DATASET: "ce9119795acef4ca537b2175e10a3feb7a0ecae9",
    NOMENCLATURE_DATASET: "571f171a0464e659e7f246df4f5d0472c826e2ff",
}
"""The dataset commits every row comes from; the same commit always yields the same rows."""

Source = Literal["cross", "hscodecomp"]
Split = Literal["train", "validation", "test"]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
RULING_QUESTION_RE = re.compile(r"^\s*What is the HTS US Code for\s*(.*?)\s*\??\s*$", re.DOTALL)
RULING_CODE_RE = re.compile(r"HTS US Code\s*->\s*([0-9.]+)")
RULEBOOK_FILE = "hs_nomenclature.txt"


def parse_code(text: str) -> str | None:
    """The digits of the last `\\boxed{}` in the reply when they form a 10-digit code, else None."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    digits = re.sub(r"[^0-9]", "", matches[-1])
    return digits if len(digits) == 10 else None


def matched_digits(predicted: str | None, gold: str) -> int:
    """Leading digits shared with the gold code, counted in the HS levels of two: 0, 2, 4, 6, 8 or 10."""
    if predicted is None:
        return 0
    common = 0
    for a, b in zip(predicted, gold):
        if a != b:
            break
        common += 1
    return min(common, 10) // 2 * 2


def parse_ruling(messages: list[dict]) -> tuple[str, str] | None:
    """(product description, 10-digit gold code) from one CROSS row, or None if the row is malformed."""
    if len(messages) < 2:
        return None
    question = RULING_QUESTION_RE.match(messages[0].get("content") or "")
    code = RULING_CODE_RE.search(messages[1].get("content") or "")
    if question is None or code is None:
        return None
    digits = re.sub(r"[^0-9]", "", code.group(1))
    description = question.group(1).strip()
    return (description, digits) if description and len(digits) == 10 else None


def describe_listing(row: dict) -> str:
    """A product description built from an HSCodeComp listing: title, category path, attributes, price."""
    try:
        attributes = json.loads(row.get("product_attributes") or "{}")
    except json.JSONDecodeError:
        attributes = {}
    categories = [row.get(f"cate_lv{i}_desc") for i in range(1, 6)]
    path = " > ".join(dict.fromkeys(c for c in categories if c))
    lines = [f"Title: {row['product_name']}"]
    if path:
        lines.append(f"Listed under: {path}")
    if attributes:
        lines.append("Attributes: " + "; ".join(f"{k}: {v}" for k, v in attributes.items()))
    if row.get("price") is not None:
        lines.append(f"Price: {row['price']} {row.get('currency_code') or ''}".rstrip())
    return "\n".join(lines)


@lru_cache(maxsize=None)
def rows_for(source: str, split: str) -> tuple[tuple[str, str], ...]:
    """(description, gold code) for every usable row of a split, in dataset order."""
    from datasets import load_dataset

    if source == "cross":
        dataset = load_dataset(CROSS_DATASET, split=split, revision=REVISIONS[CROSS_DATASET])
        parsed = (parse_ruling(row["messages"]) for row in dataset)
        return tuple(pair for pair in parsed if pair is not None)
    if split != "test":
        raise ValueError("hscodecomp has only a test split; it is a held-out set, never train on it")
    dataset = load_dataset(HSCODECOMP_DATASET, split="test", revision=REVISIONS[HSCODECOMP_DATASET])
    return tuple((describe_listing(row), str(row["hs_code"]).zfill(10)) for row in dataset)


@lru_cache(maxsize=None)
def hs6_codes() -> frozenset[str]:
    """Every six-digit subheading in the HS nomenclature."""
    from datasets import load_dataset

    dataset = load_dataset(NOMENCLATURE_DATASET, split="train", revision=REVISIONS[NOMENCLATURE_DATASET])
    return frozenset(row["text"][:6] for row in dataset if row["aggrlevel"] == 6)


@lru_cache(maxsize=None)
def rulebook() -> bytes:
    """The HS nomenclature as one line per heading, for a harness with a shell to search."""
    from datasets import load_dataset

    dataset = load_dataset(NOMENCLATURE_DATASET, split="train", revision=REVISIONS[NOMENCLATURE_DATASET])
    return "\n".join(row["text"] for row in dataset).encode()


def prompt_for(description: str, with_rulebook: bool) -> str:
    rulebook_note = (
        f" The file `{RULEBOOK_FILE}` in the working directory lists the HS nomenclature, one heading per line."
        if with_rulebook
        else ""
    )
    return (
        "Classify the product below for import into the United States. Reply with its 10-digit HTS code "
        f"inside \\boxed{{}}.{rulebook_note}\n\n{description}"
    )


class HtsClassifyData(vf.TaskData):
    source: str
    """`cross` (customs rulings) or `hscodecomp` (expert-labeled listings)."""
    split: str
    row: int
    """Index of the usable row within its split; the gold code is looked up from it at scoring time."""


class HtsClassifyTaskConfig(vf.TaskConfig):
    rulebook: bool = False
    """Write the HS nomenclature into the box and say so in the prompt (for harnesses with a shell)."""


class HtsClassifyTask(vf.Task[HtsClassifyData, vf.State, HtsClassifyTaskConfig]):
    @property
    def key(self) -> str:
        return f"hts:{self.data.source}:{self.data.split}:{self.data.row}"

    @property
    def _gold(self) -> str:
        return rows_for(self.data.source, self.data.split)[self.data.row][1]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.config.rulebook:
            await runtime.write(RULEBOOK_FILE, rulebook())

    @vf.reward(weight=1.0)
    async def hierarchical(self, trace: vf.Trace) -> float:
        """Leading digits shared with the gold code, over ten."""
        return matched_digits(parse_code(trace.last_reply), self._gold) / 10

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        return float(parse_code(trace.last_reply) == self._gold)

    @vf.metric
    async def hs6(self, trace: vf.Trace) -> float:
        """The globally harmonized six-digit subheading is right."""
        return float(matched_digits(parse_code(trace.last_reply), self._gold) >= 6)

    @vf.metric
    async def heading(self, trace: vf.Trace) -> float:
        return float(matched_digits(parse_code(trace.last_reply), self._gold) >= 4)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        return float(parse_code(trace.last_reply) is not None)

    @vf.metric
    async def valid_hs6(self, trace: vf.Trace) -> float:
        """The predicted subheading exists in the nomenclature, right or wrong."""
        code = parse_code(trace.last_reply)
        return float(code is not None and code[:6] in hs6_codes())

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Ten gold digits whose subheading exists, and a description to classify."""
        description, gold = rows_for(self.data.source, self.data.split)[self.data.row]
        return len(gold) == 10 and gold.isdigit() and gold[:6] in hs6_codes() and bool(description.strip())


class HtsClassifyConfig(vf.TasksetConfig):
    source: Source = "cross"
    """Which rows: customs rulings (train/validation/test) or the HSCodeComp listings (test only)."""
    split: Split = "validation"
    task: HtsClassifyTaskConfig = HtsClassifyTaskConfig()


class HtsClassifyTaskset(vf.Taskset[HtsClassifyTask, HtsClassifyConfig]):
    def load(self) -> Iterator[HtsClassifyTask]:
        c = self.config
        for row, (description, _gold) in enumerate(rows_for(c.source, c.split)):
            yield HtsClassifyTask(
                HtsClassifyData(
                    idx=row,
                    prompt=prompt_for(description, c.task.rulebook),
                    source=c.source,
                    split=c.split,
                    row=row,
                ),
                c.task,
            )
