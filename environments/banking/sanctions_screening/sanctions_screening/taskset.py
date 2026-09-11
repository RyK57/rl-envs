"""sanctions-screening: decide whether two watch-list records describe the same person or organization.

Environment eight, the second real vertical: banking compliance. Every payment and every new
customer is screened against sanctions and politically-exposed-person lists, and an analyst
resolves each hit by hand: the same person under a transliterated name, or a different one who
happens to share it. The rows are real: OpenSanctions' cross-referencing judgements, pairs of
entity records with an analyst's verdict, as snapshotted on Hugging Face. The reward is the
verdict itself, deterministic. Verdicts stay off `TaskData` and are looked up by key at scoring
time. `validate()` checks that every pair carries a verdict and two named records.
"""

import gzip
import json
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

DATASET = "sanctions-er-anon/opensanctions_pairs"
REVISION: str | None = None
"""The dataset commit every row comes from. None follows the default branch; pin it with
`uv run python scripts/pin_revisions.py` once the rows have been fetched and inspected."""
SAMPLE_FILE = "sample_1000.json"
PAIRS_FILE = "pairs.json.gz"

Split = Literal["train", "validation", "test"]
VERDICTS = {"positive": "same", "negative": "different"}
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "Names"),
    ("alias", "Aliases"),
    ("weakAlias", "Weak aliases"),
    ("firstName", "First name"),
    ("secondName", "Second name"),
    ("middleName", "Middle name"),
    ("lastName", "Last name"),
    ("fatherName", "Father's name"),
    ("motherName", "Mother's name"),
    ("gender", "Gender"),
    ("birthDate", "Born"),
    ("birthPlace", "Birth place"),
    ("birthCountry", "Birth country"),
    ("deathDate", "Died"),
    ("nationality", "Nationality"),
    ("citizenship", "Citizenship"),
    ("country", "Country"),
    ("jurisdiction", "Jurisdiction"),
    ("legalForm", "Legal form"),
    ("incorporationDate", "Incorporated"),
    ("dissolutionDate", "Dissolved"),
    ("address", "Address"),
    ("position", "Position"),
    ("title", "Title"),
    ("idNumber", "ID number"),
    ("passportNumber", "Passport"),
    ("taxNumber", "Tax number"),
    ("registrationNumber", "Registration number"),
    ("innCode", "INN"),
    ("ogrnCode", "OGRN"),
    ("leiCode", "LEI"),
    ("swiftBic", "SWIFT/BIC"),
    ("imoNumber", "IMO number"),
    ("email", "Email"),
    ("phone", "Phone"),
    ("website", "Website"),
    ("notes", "Notes"),
)
"""Record properties shown to the model, in this order. Provenance fields (source URLs, list
identifiers, programme codes, timestamps) are left out: they identify the list, not the person."""
LINE_LIMIT = 400


def parse_verdict(text: str) -> str | None:
    """`same` or `different` from the last `\\boxed{}` in the reply, else None."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    word = re.sub(r"[^a-z]", "", matches[-1].lower())
    return word if word in ("same", "different") else None


def render_record(entity: dict) -> str:
    """One record as labelled lines: its type, display name, then the identity properties it has."""
    lines = [f"Type: {entity.get('schema') or 'Unknown'}", f"Name: {entity.get('caption') or ''}"]
    properties = entity.get("properties") or {}
    for field, label in IDENTITY_FIELDS:
        values = [str(value) for value in properties.get(field) or [] if value]
        if values:
            lines.append(f"{label}: {'; '.join(values)}"[:LINE_LIMIT])
    return "\n".join(lines)


def prompt_for(left: dict, right: dict) -> str:
    return (
        "Two records from sanctions and watch lists follow. Decide whether they refer to the same "
        "real-world person or organization. Reply with \\boxed{same} or \\boxed{different}.\n\n"
        f"Record A\n{render_record(left)}\n\nRecord B\n{render_record(right)}"
    )


def pair_id(pair: dict) -> str:
    return f"{pair['left']['id']}|{pair['right']['id']}"


def verdict_of(pair: dict) -> str | None:
    return VERDICTS.get(pair.get("judgement"))


def dataset_file(name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION)


@lru_cache(maxsize=None)
def sample_pairs() -> tuple[dict, ...]:
    """The 1,000-pair stratified sample published with the corpus: the held-out rows."""
    with open(dataset_file(SAMPLE_FILE), encoding="utf-8") as f:
        return tuple(json.load(f)["pairs"])


def stream_pairs() -> Iterator[dict]:
    """Every judged pair of the full corpus, one JSON object per line, in file order."""
    with gzip.open(dataset_file(PAIRS_FILE), "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def balanced_pairs(size: int, scan_limit: int) -> tuple[dict, ...]:
    """The first `size` pairs of the corpus outside the sample, half per verdict, interleaved."""
    held_out = {pair_id(pair) for pair in sample_pairs()}
    wanted = {"same": size // 2, "different": size - size // 2}
    kept: dict[str, list[dict]] = {"same": [], "different": []}
    for n, pair in enumerate(stream_pairs()):
        if n >= scan_limit or all(len(kept[v]) >= wanted[v] for v in wanted):
            break
        verdict = verdict_of(pair)
        if verdict is None or pair_id(pair) in held_out or len(kept[verdict]) >= wanted[verdict]:
            continue
        kept[verdict].append(pair)
    short = {v: wanted[v] - len(kept[v]) for v in wanted if len(kept[v]) < wanted[v]}
    if short:
        raise ValueError(
            f"{short} pairs short of a balanced set of {size} within the first {scan_limit} rows of "
            f"{PAIRS_FILE}; raise scan_limit or lower size"
        )
    return tuple(pair for both in zip(kept["same"], kept["different"]) for pair in both)


@lru_cache(maxsize=None)
def rows_for(split: str, size: int, scan_limit: int) -> tuple[dict, ...]:
    """The pairs of a split, in a fixed order: the corpus for train, alternate sample rows otherwise."""
    if split == "train":
        return balanced_pairs(size, scan_limit)
    return sample_pairs()[0 if split == "validation" else 1 :: 2]


class SanctionsScreeningData(vf.TaskData):
    split: str
    size: int
    """Balanced train pairs wanted; the sample splits ignore it."""
    scan_limit: int
    row: int
    pair_id: str
    """`<left id>|<right id>`, the public record identifiers; the verdict is looked up at scoring time."""


class SanctionsScreeningTask(vf.Task[SanctionsScreeningData]):
    @property
    def key(self) -> str:
        return f"sanctions:{self.data.split}:{self.data.pair_id}"

    @property
    def _pair(self) -> dict:
        return rows_for(self.data.split, self.data.size, self.data.scan_limit)[self.data.row]

    @vf.reward(weight=1.0)
    async def verdict(self, trace: vf.Trace) -> float:
        """The analyst's verdict, same or different, reproduced."""
        return float(parse_verdict(trace.last_reply) == verdict_of(self._pair))

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        return float(parse_verdict(trace.last_reply) is not None)

    @vf.metric
    async def false_clear(self, trace: vf.Trace) -> float:
        """A listed person waved through: the pair is the same entity and the reply said different."""
        return float(verdict_of(self._pair) == "same" and parse_verdict(trace.last_reply) == "different")

    @vf.metric
    async def false_hit(self, trace: vf.Trace) -> float:
        """An innocent namesake blocked: the pair is different and the reply said same."""
        return float(verdict_of(self._pair) == "different" and parse_verdict(trace.last_reply) == "same")

    async def validate(self, runtime: vf.Runtime) -> bool:
        """A verdict, two named records, and the pair the key promises."""
        pair = self._pair
        named = all((pair[side].get("caption") or "").strip() for side in ("left", "right"))
        return verdict_of(pair) is not None and named and pair_id(pair) == self.data.pair_id


class SanctionsScreeningConfig(vf.TasksetConfig):
    split: Split = "validation"
    """`train` reads the full corpus; `validation` and `test` are the two halves of the published sample."""
    size: int = Field(4000, ge=2)
    """Train pairs, half `same` and half `different`."""
    scan_limit: int = Field(100_000, ge=2)
    """Corpus rows read at most while filling the train pairs."""


class SanctionsScreeningTaskset(vf.Taskset[SanctionsScreeningTask, SanctionsScreeningConfig]):
    def load(self) -> Iterator[SanctionsScreeningTask]:
        c = self.config
        for row, pair in enumerate(rows_for(c.split, c.size, c.scan_limit)):
            yield SanctionsScreeningTask(
                SanctionsScreeningData(
                    idx=row,
                    prompt=prompt_for(pair["left"], pair["right"]),
                    split=c.split,
                    size=c.size,
                    scan_limit=c.scan_limit,
                    row=row,
                    pair_id=pair_id(pair),
                ),
                c.task,
            )
