"""medical-coding: assign ICD-10-CM diagnosis codes to a clinical history.

Environment eleven, healthcare revenue cycle. Every hospital encounter is coded before it is
billed, by hand, by certified coders, and the code decides the payment. Two sources:

- `cases`: 1,500 de-identified clinical histories from open-access case reports (CC BY), each
  with the ICD-10-CM code of its principal diagnosis. One code per case; the reward is
  hierarchical, the three-character category earning half credit and every further character
  of the gold code the rest.
- `codiesp`: the CodiEsp corpus (CC BY 4.0), 1,000 Spanish clinical cases coded by professional
  clinical coders with published inter-annotator agreement, several diagnosis codes per case.
  The reward is set F1 over the codes, the shared task's own measure.

The full ICD-10-CM code table backs a validity metric. Gold stays off `TaskData` and is looked up
by key at scoring time.
"""

import hashlib
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import verifiers.v1 as vf

CASES_DATASET = "mkurman/clinical-case-icd10-diagnosis"
CASES_CONFIG = "commercial"
CODIESP_DATASET = "bigbio/codiesp"
CODIESP_CONFIG = "codiesp_D_bigbio_text"
CODES_DATASET = "awacke1/ICD10-Clinical-Terminology"
BRANCHES: dict[str, str] = {CODIESP_DATASET: "refs/convert/parquet"}
"""CodiEsp is published as a loading script, which `datasets` no longer runs; the Hub's parquet
conversion of it lives on this branch and is read file by file."""
REVISIONS: dict[str, str | None] = {
    CASES_DATASET: None,
    CODIESP_DATASET: BRANCHES[CODIESP_DATASET],
    CODES_DATASET: None,
}
"""Dataset commits the rows come from. None follows the default branch; `scripts/pin_revisions.py`
prints the commits to pin once the rows have been fetched."""
CODIESP_FILES = {split: f"{CODIESP_CONFIG}/{split}/0000.parquet" for split in ("train", "validation", "test")}

Source = Literal["cases", "codiesp"]
Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CODE_RE = re.compile(r"^[A-Z][0-9][0-9A-Z]{1,5}$")


def normalize_code(code: str) -> str:
    """`j05.11` and `J05.11` both become `J0511`, the form the code table uses."""
    return re.sub(r"[^0-9A-Z]", "", code.upper())


def parse_codes(text: str) -> list[str] | None:
    """The well-formed codes in the last `\\boxed{}`, in order without repeats, or None without a box."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    codes = [normalize_code(part) for part in re.split(r"[,;\s]+", matches[-1]) if part.strip()]
    return list(dict.fromkeys(code for code in codes if CODE_RE.match(code)))


def hierarchical(predicted: str | None, gold: str) -> float:
    """Half credit for the right three-character category, the rest per further matching character."""
    if predicted is None:
        return 0.0
    common = 0
    for a, b in zip(predicted, gold):
        if a != b:
            break
        common += 1
    if common < 3:
        return 0.0
    if len(gold) <= 3:
        return 1.0
    return 0.5 + 0.5 * (min(common, len(gold)) - 3) / (len(gold) - 3)


def set_f1(predicted: list[str], gold: tuple[str, ...]) -> float:
    hits = len(set(predicted) & set(gold))
    if hits == 0:
        return 0.0
    precision, recall = hits / len(set(predicted)), hits / len(set(gold))
    return 2 * precision * recall / (precision + recall)


def split_of(key: str) -> str:
    bucket = hashlib.sha1(key.encode()).digest()[0] % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


@lru_cache(maxsize=None)
def rows_for(source: str, split: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """(id, clinical text, gold codes) for a split, in dataset order."""
    from datasets import load_dataset

    if source == "cases":
        dataset = load_dataset(CASES_DATASET, CASES_CONFIG, split="train", revision=REVISIONS[CASES_DATASET])
        rows = ((row["case_id"], row["input"], (normalize_code(row["icd10_code"]),)) for row in dataset)
        return tuple(row for row in rows if split_of(row[0]) == split)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        CODIESP_DATASET, CODIESP_FILES[split], repo_type="dataset", revision=REVISIONS[CODIESP_DATASET]
    )
    table = pq.read_table(path, columns=["document_id", "text", "labels"])
    return tuple(
        (row["document_id"], row["text"], tuple(dict.fromkeys(normalize_code(c) for c in row["labels"])))
        for row in table.to_pylist()
    )


@lru_cache(maxsize=None)
def icd10_codes() -> frozenset[str]:
    from datasets import load_dataset

    dataset = load_dataset(CODES_DATASET, split="train", revision=REVISIONS[CODES_DATASET])
    return frozenset(normalize_code(row["Code"]) for row in dataset)


def prompt_for(source: str, text: str) -> str:
    if source == "cases":
        return (
            "A patient's clinical history follows. Give the ICD-10-CM code of the principal diagnosis. "
            f"Reply with the one code inside \\boxed{{}}.\n\n{text}"
        )
    return (
        "A clinical case in Spanish follows. List every ICD-10-CM diagnosis code that applies to it. "
        f"Reply with the codes inside \\boxed{{}}, separated by commas.\n\n{text}"
    )


class MedicalCodingData(vf.TaskData):
    source: str
    split: str
    row: int
    case_id: str


class MedicalCodingTask(vf.Task[MedicalCodingData]):
    @property
    def key(self) -> str:
        return f"icd10:{self.data.source}:{self.data.case_id}"

    @property
    def _gold(self) -> tuple[str, ...]:
        return rows_for(self.data.source, self.data.split)[self.data.row][2]

    def _score(self, reply: str) -> float:
        codes = parse_codes(reply)
        if not codes:
            return 0.0
        if self.data.source == "cases":
            return hierarchical(codes[0], self._gold[0]) if len(codes) == 1 else 0.0
        return set_f1(codes, self._gold)

    @vf.reward(weight=1.0)
    async def coding(self, trace: vf.Trace) -> float:
        """Cases: hierarchical match on the one code. CodiEsp: F1 over the set of codes."""
        return self._score(trace.last_reply)

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        return float(self._score(trace.last_reply) == 1.0)

    @vf.metric
    async def category(self, trace: vf.Trace) -> float:
        """Three-character categories: right for the one code, or F1 over categories for CodiEsp."""
        codes = parse_codes(trace.last_reply)
        if not codes:
            return 0.0
        predicted = [c[:3] for c in codes]
        gold = tuple(dict.fromkeys(c[:3] for c in self._gold))
        if self.data.source == "cases":
            return float(len(codes) == 1 and predicted[0] == gold[0])
        return set_f1(predicted, gold)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        codes = parse_codes(trace.last_reply)
        return float(bool(codes) and (self.data.source != "cases" or len(codes) == 1))

    @vf.metric
    async def valid_code(self, trace: vf.Trace) -> float:
        """Every predicted code exists in the ICD-10-CM table, right or wrong."""
        codes = parse_codes(trace.last_reply)
        return float(bool(codes) and all(code in icd10_codes() for code in codes))

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Well-formed gold codes on a non-empty case, for the case the key promises."""
        case_id, text, gold = rows_for(self.data.source, self.data.split)[self.data.row]
        well_formed = bool(gold) and all(CODE_RE.match(code) for code in gold)
        return case_id == self.data.case_id and well_formed and bool(text.strip())


class MedicalCodingConfig(vf.TasksetConfig):
    source: Source = "cases"
    """`cases`: one principal ICD-10-CM code per English case history. `codiesp`: all diagnosis codes of a Spanish case."""
    split: Split = "validation"


class MedicalCodingTaskset(vf.Taskset[MedicalCodingTask, MedicalCodingConfig]):
    def load(self) -> Iterator[MedicalCodingTask]:
        c = self.config
        for row, (case_id, text, _gold) in enumerate(rows_for(c.source, c.split)):
            yield MedicalCodingTask(
                MedicalCodingData(
                    idx=row, prompt=prompt_for(c.source, text), source=c.source, split=c.split, row=row, case_id=case_id
                ),
                c.task,
            )
