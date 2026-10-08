"""ghs-hazards: the GHS hazard statements (H-codes) of a substance from its structure.

Environment fifteen, chemical safety. Every substance that is sold, shipped or stored carries a
GHS classification on its safety data sheet, and the hazard statements are the part of it that
decides the label, the storage class and the handling rules. The rows are the ChemNLP project's
table of Sigma-Aldrich safety data (MIT as published): a SMILES string, a CAS registry number and
the H-codes the data sheet lists, for 6,420 substances. A quarter of them carry no hazard
statement at all, so abstaining (`none`) is a real answer. The reward is set F1 over the codes,
with half credit for a code of the right hazard class but the wrong category (H301 for H302, both
acute oral toxicity). Gold stays off `TaskData` and is looked up by key at scoring time; the file is
pinned by its SHA-256 as well as by dataset commit.
"""

import csv
import hashlib
import io
import re
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal, NamedTuple

import verifiers.v1 as vf

DATASET = "chemNLP/msds_sigma_aldrich"
FILE = "msds.csv"
REVISION: str | None = None
"""The dataset commit the rows come from; `uv run python scripts/pin_revisions.py ghs_hazards` prints
the current one. The file is also pinned by content below, so a changed file fails loudly either way."""
FILE_SHA256 = "c49216183330efd9bd28a1bb51dbd07fbe3b2fd3f34452e850229b06eff7002b"

Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CODE_RE = re.compile(r"(?<![A-Za-z0-9])H(\d{3})[A-Za-z]{0,2}(?![A-Za-z0-9])")
"""An H-code, with or without a sub-statement suffix (H360FD counts as H360)."""

HAZARD_CLASSES: dict[str, tuple[str, ...]] = {
    "explosives": ("H200", "H201", "H202", "H203", "H204", "H205"),
    "desensitized explosives": ("H206", "H207", "H208"),
    "flammable gases": ("H220", "H221", "H230", "H231", "H232"),
    "aerosols": ("H222", "H223", "H229"),
    "flammable liquids": ("H224", "H225", "H226", "H227"),
    "flammable solids": ("H228",),
    "self-reactive substances and organic peroxides": ("H240", "H241", "H242"),
    "pyrophoric liquids and solids": ("H250",),
    "self-heating substances": ("H251", "H252"),
    "substances which in contact with water emit flammable gases": ("H260", "H261"),
    "oxidizing gases": ("H270",),
    "oxidizing liquids and solids": ("H271", "H272"),
    "gases and chemicals under pressure": ("H280", "H281", "H282", "H283", "H284"),
    "corrosive to metals": ("H290",),
    "acute toxicity, oral": ("H300", "H301", "H302", "H303"),
    "aspiration hazard": ("H304", "H305"),
    "acute toxicity, dermal": ("H310", "H311", "H312", "H313"),
    "skin corrosion or irritation": ("H314", "H315", "H316"),
    "skin sensitization": ("H317",),
    "serious eye damage or eye irritation": ("H318", "H319", "H320"),
    "acute toxicity, inhalation": ("H330", "H331", "H332", "H333"),
    "respiratory sensitization": ("H334",),
    "specific target organ toxicity, single exposure": ("H335", "H336", "H370", "H371"),
    "germ cell mutagenicity": ("H340", "H341"),
    "carcinogenicity": ("H350", "H351"),
    "reproductive toxicity": ("H360", "H361", "H362"),
    "specific target organ toxicity, repeated exposure": ("H372", "H373"),
    "hazardous to the aquatic environment, acute": ("H400", "H401", "H402"),
    "hazardous to the aquatic environment, chronic": ("H410", "H411", "H412", "H413"),
    "hazardous to the ozone layer": ("H420",),
}
"""GHS hazard classes and the statements that belong to each (GHS Rev. 9, Annex 3)."""
CLASS_OF: dict[str, str] = {code: name for name, codes in HAZARD_CLASSES.items() for code in codes}


class Row(NamedTuple):
    cas: str
    smiles: str
    codes: frozenset[str]


def hazard_class(code: str) -> str:
    """The hazard class of a statement; a code the table does not know is its own class."""
    return CLASS_OF.get(code, code)


def parse_codes(text: str) -> frozenset[str] | None:
    """The H-codes in the last `\\boxed{}`, an empty set for `none`, or None without a box or an answer."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1]
    codes = frozenset(f"H{digits}" for digits in CODE_RE.findall(answer.upper()))
    if codes:
        return codes
    return frozenset() if answer.strip().lower() == "none" else None


def f1(hits: float, predicted: int, gold: int) -> float:
    if predicted == 0 and gold == 0:
        return 1.0
    if hits == 0:
        return 0.0
    precision, recall = hits / predicted, hits / gold
    return 2 * precision * recall / (precision + recall)


def strict_f1(predicted: frozenset[str], gold: frozenset[str]) -> float:
    return f1(len(predicted & gold), len(predicted), len(gold))


def class_f1(predicted: frozenset[str], gold: frozenset[str]) -> float:
    """F1 over hazard classes: H301 and H302 are the same acute oral toxicity class."""
    predicted_classes = frozenset(map(hazard_class, predicted))
    gold_classes = frozenset(map(hazard_class, gold))
    return f1(len(predicted_classes & gold_classes), len(predicted_classes), len(gold_classes))


def soft_f1(predicted: frozenset[str], gold: frozenset[str]) -> float:
    """Set F1 where an exact code counts one and a code of the right class but the wrong category half.
    Partial matches pair up one-to-one within a class, so two wrong categories cannot cover one gold code."""
    exact = len(predicted & gold)
    spare_predicted = Counter(hazard_class(code) for code in predicted - gold)
    spare_gold = Counter(hazard_class(code) for code in gold - predicted)
    partial = sum(min(count, spare_gold[name]) for name, count in spare_predicted.items())
    return f1(exact + 0.5 * partial, len(predicted), len(gold))


def split_of(cas: str) -> str:
    bucket = hashlib.sha1(cas.encode()).digest()[0] % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def parse_table(data: bytes) -> tuple[Row, ...]:
    """Every substance with a standard SMILES, one row per structure, in file order; refuses a changed file."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != FILE_SHA256:
        raise ValueError(f"{DATASET}/{FILE} is not the pinned file: sha256 {digest}, expected {FILE_SHA256}")
    rows: list[Row] = []
    seen: set[str] = set()
    dropped: Counter[str] = Counter()
    for record in csv.DictReader(io.StringIO(data.decode("utf-8"))):
        smiles = (record.get("SMILES") or "").strip()
        cas = (record.get("cas_nr") or "").strip()
        if not smiles:
            dropped["no structure"] += 1
        elif "|" in smiles:
            dropped["non-standard SMILES"] += 1
        elif not cas:
            dropped["no CAS number"] += 1
        elif smiles in seen:
            dropped["duplicate structure"] += 1
        else:
            seen.add(smiles)
            codes = frozenset(
                column for column, value in record.items() if CODE_RE.fullmatch(column or "") and value == "1"
            )
            rows.append(Row(cas, smiles, codes))
    if not rows:
        raise ValueError(f"no usable substances in {DATASET}/{FILE}; dropped {dict(dropped)}")
    return tuple(rows)


@lru_cache(maxsize=None)
def table() -> tuple[Row, ...]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, FILE, repo_type="dataset", revision=REVISION)
    with open(path, "rb") as f:
        return parse_table(f.read())


@lru_cache(maxsize=None)
def rows_for(split: str) -> tuple[Row, ...]:
    """The substances of a split, in file order; the split is a hash of the CAS number."""
    return tuple(row for row in table() if split_of(row.cas) == split)


def prompt_for(smiles: str, cas: str | None) -> str:
    identity = f"SMILES: {smiles}" + (f"\nCAS: {cas}" if cas else "")
    return (
        "Classify the substance below under the GHS (Globally Harmonized System of Classification and "
        "Labelling of Chemicals). Reply with its GHS hazard statement codes inside \\boxed{}, separated by "
        "commas (for example \\boxed{H225, H319}), or \\boxed{none} if it carries no hazard statement.\n\n"
        f"{identity}"
    )


class GhsHazardsData(vf.TaskData):
    split: str
    row: int
    """Index of the substance within its split; the hazard statements are looked up from it at scoring time."""
    cas: str
    """CAS registry number, the row's durable identity; shown in the prompt only with the `cas` knob."""


class GhsHazardsTask(vf.Task[GhsHazardsData]):
    @property
    def key(self) -> str:
        return f"ghs:{self.data.split}:{self.data.cas}"

    @property
    def _gold(self) -> frozenset[str]:
        return rows_for(self.data.split)[self.data.row].codes

    @vf.reward(weight=1.0)
    async def hazard_f1(self, trace: vf.Trace) -> float:
        """F1 over the hazard statements; a code of the right class but the wrong category earns half."""
        predicted = parse_codes(trace.last_reply)
        return 0.0 if predicted is None else soft_f1(predicted, self._gold)

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        """The predicted set is the data sheet's set, `none` included."""
        return float(parse_codes(trace.last_reply) == self._gold)

    @vf.metric
    async def strict(self, trace: vf.Trace) -> float:
        """Set F1 with no credit for a wrong category."""
        predicted = parse_codes(trace.last_reply)
        return 0.0 if predicted is None else strict_f1(predicted, self._gold)

    @vf.metric
    async def hazard_class(self, trace: vf.Trace) -> float:
        """F1 over hazard classes, the level a label pictogram is chosen at."""
        predicted = parse_codes(trace.last_reply)
        return 0.0 if predicted is None else class_f1(predicted, self._gold)

    @vf.metric
    async def precision(self, trace: vf.Trace) -> float:
        predicted = parse_codes(trace.last_reply)
        if not predicted:
            return float(predicted is not None and not self._gold)
        return len(predicted & self._gold) / len(predicted)

    @vf.metric
    async def recall(self, trace: vf.Trace) -> float:
        predicted = parse_codes(trace.last_reply)
        if not self._gold:
            return float(predicted is not None and not predicted)
        return 0.0 if predicted is None else len(predicted & self._gold) / len(self._gold)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        """A boxed list of H-codes or a boxed `none` was given."""
        return float(parse_codes(trace.last_reply) is not None)

    @vf.metric
    async def valid_codes(self, trace: vf.Trace) -> float:
        """Every predicted code is a GHS hazard statement, right or wrong."""
        predicted = parse_codes(trace.last_reply)
        return float(bool(predicted) and all(code in CLASS_OF for code in predicted))

    @vf.metric
    async def num_codes(self, trace: vf.Trace) -> float:
        predicted = parse_codes(trace.last_reply)
        return float(len(predicted)) if predicted else 0.0

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The row the key promises, with a structure, well-formed codes, and a gold that scores 1.0 as written."""
        row = rows_for(self.data.split)[self.data.row]
        well_formed = all(CODE_RE.fullmatch(code) for code in row.codes)
        gold = ", ".join(sorted(row.codes)) or "none"
        return (
            row.cas == self.data.cas
            and bool(row.smiles)
            and well_formed
            and soft_f1(parse_codes(f"\\boxed{{{gold}}}"), row.codes) == 1.0
        )


class GhsHazardsConfig(vf.TasksetConfig):
    split: Split = "validation"
    cas: bool = False
    """Also show the CAS registry number, so a substance can be recognized rather than classified from its structure."""
    negatives: bool = True
    """Keep the substances without any hazard statement, whose right answer is `none`."""


class GhsHazardsTaskset(vf.Taskset[GhsHazardsTask, GhsHazardsConfig]):
    def load(self) -> Iterator[GhsHazardsTask]:
        c = self.config
        for row, entry in enumerate(rows_for(c.split)):
            if not entry.codes and not c.negatives:
                continue
            yield GhsHazardsTask(
                GhsHazardsData(
                    idx=row,
                    prompt=prompt_for(entry.smiles, entry.cas if c.cas else None),
                    split=c.split,
                    row=row,
                    cas=entry.cas,
                ),
                c.task,
            )
