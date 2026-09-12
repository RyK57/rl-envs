"""procurement-coding: code a federal contract action to its NAICS industry or its product or service code.

Environment thirteen, government contracting. Every federal contract action is tagged with the
NAICS code of the work and a four-character Product or Service Code (PSC); contracting officers
pick them by hand, set-aside eligibility and spend analysis depend on them, and vendors search
opportunities by them. The rows are real: FY2024 contract transactions from USAspending (public
domain), one parquet shard of the archive, filtered to actions with a readable description and
deduplicated by description. Rewards are hierarchical: leading NAICS digits over six, leading
PSC characters over four. Gold stays off `TaskData` and is looked up by key at scoring time.
"""

import hashlib
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal, NamedTuple

import verifiers.v1 as vf
from pydantic import Field

DATASET = "zzsi/usaspending_2024_all_contracts"
SHARD = "data/train-00000-of-00024.parquet"
REVISIONS: dict[str, str | None] = {DATASET: None}
"""Dataset commit the rows come from. None follows the default branch; `scripts/pin_revisions.py`
prints the commit to pin once the rows have been fetched."""
COLUMNS = (
    "contract_award_unique_key",
    "transaction_description",
    "naics_code",
    "naics_description",
    "product_or_service_code",
    "product_or_service_code_description",
)

Field_ = Literal["naics", "psc"]
Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NAICS_RE = re.compile(r"^[0-9]{6}$")
PSC_RE = re.compile(r"^[A-Z0-9][0-9A-Z]{3}$")
TAG_RE = re.compile(r"IGF::[A-Z]+::IGF")
QUESTIONS = {
    "naics": "Give the six-digit NAICS code of the industry the work belongs to.",
    "psc": "Give the four-character Product or Service Code (PSC) of what was bought.",
}


def clean_description(text: str) -> str:
    """The description without the inherently-governmental-function tags, single-spaced."""
    return " ".join(TAG_RE.sub(" ", text or "").split())


def usable(description: str, min_chars: int) -> bool:
    """Long enough to describe the work, and mostly words rather than part numbers."""
    letters = sum(ch.isalpha() for ch in description)
    return len(description) >= min_chars and letters >= 0.6 * len(description)


def parse_code(field: str, text: str) -> str | None:
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    code = re.sub(r"[^0-9A-Za-z]", "", matches[-1]).upper()
    return code if (NAICS_RE if field == "naics" else PSC_RE).match(code) else None


def matched_prefix(predicted: str | None, gold: str) -> float:
    """Leading characters shared with the gold code, over the gold code's length; 0 without a first match."""
    if predicted is None:
        return 0.0
    common = 0
    for a, b in zip(predicted, gold):
        if a != b:
            break
        common += 1
    return common / len(gold)


def split_of(award: str) -> str:
    bucket = hashlib.sha1(award.encode()).digest()[0] % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


class Action(NamedTuple):
    award: str
    description: str
    naics: str
    naics_name: str
    psc: str
    psc_name: str


@lru_cache(maxsize=None)
def actions(min_chars: int) -> tuple[Action, ...]:
    """Every usable action of the shard, once per description, in file order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, SHARD, repo_type="dataset", revision=REVISIONS[DATASET])
    table = pq.read_table(path, columns=list(COLUMNS))
    seen: set[str] = set()
    rows = []
    for record in table.to_pylist():
        description = clean_description(record["transaction_description"])
        naics, psc = str(record["naics_code"] or "").strip(), str(record["product_or_service_code"] or "").strip()
        key = description.lower()
        if not usable(description, min_chars) or key in seen or not NAICS_RE.match(naics) or not PSC_RE.match(psc):
            continue
        seen.add(key)
        rows.append(
            Action(
                award=str(record["contract_award_unique_key"] or ""),
                description=description,
                naics=naics,
                naics_name=record["naics_description"] or "",
                psc=psc,
                psc_name=record["product_or_service_code_description"] or "",
            )
        )
    return tuple(rows)


@lru_cache(maxsize=None)
def rows_for(split: str, min_chars: int) -> tuple[Action, ...]:
    return tuple(action for action in actions(min_chars) if split_of(action.award) == split)


def prompt_for(field: str, description: str) -> str:
    return (
        f"The description of a US federal contract action follows. {QUESTIONS[field]} Reply with the code "
        f"inside \\boxed{{}}.\n\n{description}"
    )


class ProcurementCodingData(vf.TaskData):
    field: str
    split: str
    min_chars: int
    row: int
    award: str


class ProcurementCodingTask(vf.Task[ProcurementCodingData]):
    @property
    def key(self) -> str:
        return f"procurement:{self.data.field}:{self.data.award}:{self.data.row}"

    @property
    def _action(self) -> Action:
        return rows_for(self.data.split, self.data.min_chars)[self.data.row]

    @property
    def _gold(self) -> str:
        return self._action.naics if self.data.field == "naics" else self._action.psc

    @vf.reward(weight=1.0)
    async def hierarchical(self, trace: vf.Trace) -> float:
        """Leading characters shared with the recorded code: NAICS digits over six, PSC characters over four."""
        return matched_prefix(parse_code(self.data.field, trace.last_reply), self._gold)

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        return float(parse_code(self.data.field, trace.last_reply) == self._gold)

    @vf.metric
    async def sector(self, trace: vf.Trace) -> float:
        """The two-digit NAICS sector, or the PSC's first two characters, is right."""
        code = parse_code(self.data.field, trace.last_reply)
        return float(code is not None and code[:2] == self._gold[:2])

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        return float(parse_code(self.data.field, trace.last_reply) is not None)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """A well-formed recorded code on a usable description, for the award the key promises."""
        action = self._action
        pattern = NAICS_RE if self.data.field == "naics" else PSC_RE
        return action.award == self.data.award and pattern.match(self._gold) is not None and bool(action.description)


class ProcurementCodingConfig(vf.TasksetConfig):
    field: Field_ = "naics"
    """Which code to give: the six-digit `naics` industry or the four-character `psc` product or service code."""
    split: Split = "validation"
    min_chars: int = Field(60, ge=1)
    """Shortest description kept; shorter ones are part numbers and boilerplate."""


class ProcurementCodingTaskset(vf.Taskset[ProcurementCodingTask, ProcurementCodingConfig]):
    def load(self) -> Iterator[ProcurementCodingTask]:
        c = self.config
        for row, action in enumerate(rows_for(c.split, c.min_chars)):
            yield ProcurementCodingTask(
                ProcurementCodingData(
                    idx=row,
                    prompt=prompt_for(c.field, action.description),
                    field=c.field,
                    split=c.split,
                    min_chars=c.min_chars,
                    row=row,
                    award=action.award,
                ),
                c.task,
            )
