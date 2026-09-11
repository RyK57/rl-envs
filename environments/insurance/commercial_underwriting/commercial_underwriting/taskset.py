"""commercial-underwriting: the decisions a small-commercial underwriter makes on an application.

Environment ten, the fourth real vertical: insurance. The expert-verified rows come from Snorkel's
Multi-Turn Insurance Underwriting benchmark (Apache-2.0), where a chartered underwriter validated
each company and its reference answer. Of its six task types, three are decidable from the
application plus the guidelines the benchmark publishes, and those are the tasks here:

- `limits`: per-occurrence and aggregate policy limits for the line of business,
- `deductibles`: the deductible for the line of business,
- `classification`: the company's six-digit NAICS code.

The other three (appetite, small-business eligibility, product recommendations) need the
carrier's appetite matrix and size-standard table, which the public benchmark does not ship;
they are exactly the tables a customer would bring. The `synthetic` source generates training
cases for the two rule tasks from the same guidelines over sixty industries, so the expert rows
stay held out. Rewards: an exact match on the amounts, a hierarchical match on the NAICS digits.
Gold stays off `TaskData` and is looked up by key at scoring time.
"""

import hashlib
import random
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal, NamedTuple

import verifiers.v1 as vf
from pydantic import Field

from commercial_underwriting.rulebook import (
    CITIES,
    CONSTRUCTIONS,
    CYBER_HIGH_LIMIT_PREFIXES,
    GUIDELINES,
    HIGH_DEDUCTIBLE_PREFIXES,
    INDUSTRIES,
    LOBS,
    NAME_PREFIXES,
    NAME_SUFFIXES,
    STATES,
    deductible_for,
    limits_for,
)

SNORKEL_DATASETS = (
    "snorkelai/Multi-Turn-Insurance-Underwriting",
    "snorkelai/Multi-Turn-Insurance-Underwriting-Code-Gen",
)
REVISIONS: dict[str, str | None] = {name: None for name in SNORKEL_DATASETS}
"""Dataset commits the rows come from. None follows the default branch; pin with
`uv run python scripts/pin_revisions.py` once the rows have been fetched and inspected."""

Source = Literal["snorkel", "synthetic"]
Split = Literal["train", "validation", "test"]
TASKS = {"Policy Limits": "limits", "Deductibles": "deductibles", "Business Classification": "classification"}
QUESTIONS = {
    "limits": (
        "What per-occurrence and aggregate policy limits should be offered for this line of business? "
        "Reply with both dollar amounts inside \\boxed{}, per-occurrence first."
    ),
    "deductibles": (
        "What deductible should be offered for this line of business? Reply with the dollar amount inside \\boxed{}."
    ),
    "classification": "What is the company's six-digit NAICS code? Reply with the code inside \\boxed{}.",
}
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
DOLLAR_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([A-Za-z]*)")
WORDED_RE = re.compile(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|thousand|billion)\b", re.IGNORECASE)
NAICS_RE = re.compile(r"(?<![0-9])([0-9]{6})(?![0-9])")
BARE_RE = re.compile(r"(?<![0-9.])([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{3,})(?![0-9])")
GOLD_SHAPES = {
    "limits": re.compile(r"[0-9]+ [0-9]+"),
    "deductibles": re.compile(r"[0-9]+"),
    "classification": re.compile(r"[0-9]{6}"),
}
SCALES = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "mm": 1_000_000,
    "mil": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
}


def amounts_in(text: str, bare: bool = False) -> tuple[int, ...]:
    """Every dollar amount in the text as whole dollars, sorted: `$1M`, `$1,000,000` and `1 million` all count.
    With `bare`, plain numbers of three or more digits count too, for a reply that skipped the dollar sign."""
    found = [(m.start(1), m.group(1), m.group(2)) for m in DOLLAR_RE.finditer(text or "")]
    found += [(m.start(1), m.group(1), m.group(2)) for m in WORDED_RE.finditer(text or "")]
    if bare and not found:
        found = [(m.start(1), m.group(1), "") for m in BARE_RE.finditer(text or "")]
    amounts = {}
    for position, number, unit in found:
        amounts[position] = round(float(number.replace(",", "")) * SCALES.get(unit.lower(), 1))
    return tuple(sorted(amounts.values()))


def canonical(task: str, text: str, bare: bool = False) -> str | None:
    """The scorable form of an answer text: `<occurrence> <aggregate>`, `<deductible>` or `<naics>`; None if malformed."""
    if task == "classification":
        codes = NAICS_RE.findall(text or "")
        return codes[-1] if codes else None
    amounts = amounts_in(text, bare)
    wanted = 2 if task == "limits" else 1
    return " ".join(str(a) for a in amounts) if len(amounts) == wanted else None


def parse_answer(task: str, reply: str) -> str | None:
    matches = BOXED_RE.findall(reply or "")
    return canonical(task, matches[-1], bare=True) if matches else None


def matched_naics(predicted: str, gold: str) -> float:
    """Leading digits shared with the gold NAICS code over six: sector 2, subsector 3, ..., national industry 6."""
    common = 0
    for a, b in zip(predicted, gold):
        if a != b:
            break
        common += 1
    return common / 6 if common >= 2 else 0.0


def score(task: str, reply: str, gold: str) -> float:
    answer = parse_answer(task, reply)
    if answer is None:
        return 0.0
    return matched_naics(answer, gold) if task == "classification" else float(answer == gold)


class Case(NamedTuple):
    id: str
    task: str
    name: str
    description: str
    state: str
    revenue: int
    employees: int
    payroll: int
    vehicles: int
    construction: str
    lob: str
    gold: str


def describe(case: Case) -> str:
    return "\n".join(
        [
            f"Company: {case.name}",
            f"Description: {case.description}",
            f"State: {case.state}",
            f"Annual revenue: ${case.revenue:,}",
            f"Employees: {case.employees}",
            f"Annual payroll: ${case.payroll:,}",
            f"Vehicles: {case.vehicles}",
            f"Building construction: {case.construction}",
            f"Line of business: {case.lob}",
        ]
    )


def prompt_for(case: Case) -> str:
    if case.task == "classification":
        return f"An insurance application from a small business follows. {QUESTIONS[case.task]}\n\n{describe(case)}"
    return (
        "An insurance application from a small business and the underwriting guidelines it is reviewed "
        f"under follow. {QUESTIONS[case.task]}\n\nUnderwriting guidelines:\n{GUIDELINES}\nApplication:\n{describe(case)}"
    )


def split_of(case_id: str) -> str:
    """One in five cases to test, one in five to validation, the rest to train, by id hash."""
    bucket = hashlib.sha1(case_id.encode()).digest()[0] % 5
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


@lru_cache(maxsize=None)
def snorkel_cases() -> tuple[Case, ...]:
    """Every expert-verified company-task with a scorable reference, once, across both benchmark releases."""
    from datasets import load_dataset

    cases: dict[str, Case] = {}
    for name in SNORKEL_DATASETS:
        dataset = load_dataset(name, split="train", revision=REVISIONS[name]).remove_columns(["trace"])
        for row in dataset:
            task = TASKS.get(row["task"])
            case_id = str(row["company task id"])
            if task is None or case_id in cases:
                continue
            gold = canonical(task, row["reference answer"])
            if gold is None:
                continue
            cases[case_id] = Case(
                id=case_id,
                task=task,
                name=row["company name"],
                description=row["company description"],
                state=row["state"],
                revenue=int(row["annual revenue"]),
                employees=int(row["number of employees"]),
                payroll=int(row["total payroll"]),
                vehicles=int(row["number of vehicles"]),
                construction=row["building construction"],
                lob=row["lob"],
                gold=gold,
            )
    return tuple(cases.values())


def synthetic_case(i: int, rng: random.Random) -> Case:
    """One rule-derived case: the special-case sectors and lines of business are drawn half the time."""
    task = rng.choice(("limits", "deductibles"))
    special_lob = "cyber" if task == "limits" else "auto"
    special_prefixes = CYBER_HIGH_LIMIT_PREFIXES if task == "limits" else HIGH_DEDUCTIBLE_PREFIXES
    lob = special_lob if rng.random() < 0.5 else rng.choice(LOBS)
    pool = [ind for ind in INDUSTRIES if ind.naics.startswith(special_prefixes)] if rng.random() < 0.5 else INDUSTRIES
    industry = rng.choice(pool)
    name = f"{rng.choice(NAME_PREFIXES)} {industry.noun} {rng.choice(NAME_SUFFIXES)}"
    state, city = rng.choice(STATES), rng.choice(CITIES)
    employees = rng.randint(2, 150)
    payroll = employees * rng.randint(35, 95) * 1000
    revenue = round(payroll * rng.uniform(1.5, 6.0) / 10_000) * 10_000
    vehicles = rng.randint(3, 25) if industry.naics[:2] == "48" else rng.randint(0, 6)
    if task == "limits":
        gold = " ".join(str(a) for a in limits_for(industry.naics, lob))
    else:
        gold = str(deductible_for(industry.naics, lob))
    return Case(
        id=f"synthetic-{i}",
        task=task,
        name=name,
        description=f"{name} is {industry.phrase} based in {city}, {state}. {rng.choice(industry.details)}",
        state=state,
        revenue=revenue,
        employees=employees,
        payroll=payroll,
        vehicles=vehicles,
        construction=rng.choice(CONSTRUCTIONS),
        lob=lob,
        gold=gold,
    )


@lru_cache(maxsize=None)
def rows_for(source: str, split: str, size: int, seed: int) -> tuple[Case, ...]:
    """The cases of a source and split in a fixed order; synthetic splits use their own seed offsets."""
    if source == "snorkel":
        return tuple(case for case in snorkel_cases() if split_of(case.id) == split)
    rng = random.Random(seed * 3 + ("train", "validation", "test").index(split))
    return tuple(synthetic_case(i, rng) for i in range(size))


class CommercialUnderwritingData(vf.TaskData):
    source: str
    split: str
    size: int
    """Synthetic cases generated; the expert source ignores it."""
    seed: int
    row: int
    case_id: str
    task: str
    """`limits`, `deductibles` or `classification`."""


class CommercialUnderwritingTask(vf.Task[CommercialUnderwritingData]):
    @property
    def key(self) -> str:
        return f"underwriting:{self.data.source}:{self.data.case_id}"

    @property
    def _case(self) -> Case:
        return rows_for(self.data.source, self.data.split, self.data.size, self.data.seed)[self.data.row]

    @vf.reward(weight=1.0)
    async def answer(self, trace: vf.Trace) -> float:
        """Amounts exactly right, or NAICS digits matched from the sector down."""
        return score(self.data.task, trace.last_reply, self._case.gold)

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        return float(parse_answer(self.data.task, trace.last_reply) == self._case.gold)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        return float(parse_answer(self.data.task, trace.last_reply) is not None)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """A gold in the shape the task scores, on a described company, for the case the key promises."""
        case = self._case
        well_formed = GOLD_SHAPES[case.task].fullmatch(case.gold) is not None
        return (
            case.id == self.data.case_id
            and case.task == self.data.task
            and well_formed
            and bool(case.description.strip())
        )


class CommercialUnderwritingConfig(vf.TasksetConfig):
    source: Source = "snorkel"
    """`snorkel`: expert-verified cases (limits, deductibles, classification). `synthetic`: rule-generated limits and deductibles."""
    split: Split = "validation"
    size: int = Field(2000, ge=1)
    """Synthetic cases per split."""
    seed: int = 0


class CommercialUnderwritingTaskset(vf.Taskset[CommercialUnderwritingTask, CommercialUnderwritingConfig]):
    def load(self) -> Iterator[CommercialUnderwritingTask]:
        c = self.config
        for row, case in enumerate(rows_for(c.source, c.split, c.size, c.seed)):
            yield CommercialUnderwritingTask(
                CommercialUnderwritingData(
                    idx=row,
                    prompt=prompt_for(case),
                    source=c.source,
                    split=c.split,
                    size=c.size,
                    seed=c.seed,
                    row=row,
                    case_id=case.id,
                    task=case.task,
                ),
                c.task,
            )
