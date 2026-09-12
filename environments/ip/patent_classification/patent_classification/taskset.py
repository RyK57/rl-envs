"""patent-classification: give a patent application its main CPC symbol from the title and abstract.

Environment twelve, intellectual property. Every application is classified so that it reaches the
right examiner and so that prior art can be searched; patent offices and IP firms do it by hand
and train their own models for it. The rows are the Harvard USPTO Patent Dataset's January 2016
sample (CC BY-NC-SA 4.0): every utility application filed that month, as submitted, with the
main Cooperative Patent Classification symbol the office assigned. The reward is hierarchical
over the five CPC levels: section, class, subclass, main group, subgroup. The CPC scheme backs a
validity metric. The sample archive holds the full text of every application; the loader keeps
only what the task needs in a small cache next to the Hugging Face cache, built on first use.
"""

import hashlib
import json
import re
import tarfile
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

DATASET = "HUPD/hupd"
SAMPLE_FILE = "data/sample-jan-2016.tar.gz"
SCHEME_DATASET = "mhurhangee/cpc-classifications"
SCHEME_FILE = "data/train-00000-of-00001.parquet"
REVISIONS: dict[str, str | None] = {
    DATASET: "f570a84b03663180b6034c1f7f4c15864f94385e",
    SCHEME_DATASET: "c1df8e3bac8a5b8c6363763c29d2ab3bd3d95a1f",
}
"""The dataset commits every row comes from; the same commits always yield the same rows.
`uv run python scripts/pin_revisions.py patent_classification` shows whether a branch has moved."""

Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CPC_RE = re.compile(r"^([A-HY])(\d{2})([A-Z])(\d{1,4})/(\d{2,6})$")
RAW_RE = re.compile(r"^([A-HY]\d{2}[A-Z])([1-9]\d{2,9})$")
"""HUPD's form of a symbol: the subclass, then the main group and subgroup digits run together, no slash."""


def canonical(label: str) -> str | None:
    """`G06F 17/30`, `g06f17/30` and `G06F17-30` all become `G06F17/30`; None if not a CPC symbol."""
    text = re.sub(r"\s+", "", (label or "").upper()).replace("-", "/")
    return text if CPC_RE.match(text) else None


def candidates(raw: str) -> list[str]:
    """Every symbol a slashless HUPD label could mean: one to four group digits, at least two subgroup digits."""
    match = RAW_RE.match(raw or "")
    if not match:
        return []
    subclass, digits = match.groups()
    return [f"{subclass}{digits[:n]}/{digits[n:]}" for n in range(1, 5) if len(digits) - n >= 2]


def resolve(raw: str, scheme: frozenset[str]) -> str | None:
    """The canonical symbol of a label: as written when it has a slash, else the one reading the CPC scheme knows."""
    label = canonical(raw)
    if label is not None:
        return label
    known = [symbol for symbol in candidates(raw) if symbol in scheme]
    return known[0] if len(known) == 1 else None


def levels(label: str) -> tuple[str, str, str, str, str]:
    """Section, class, subclass, main group and subgroup of a canonical symbol."""
    section, cls, subclass, group, subgroup = CPC_RE.match(label).groups()
    return (section, section + cls, section + cls + subclass, f"{section}{cls}{subclass}{group}", label)


def matched_levels(predicted: str | None, gold: str) -> int:
    """How many leading CPC levels the prediction shares with the gold symbol, 0 to 5."""
    if predicted is None:
        return 0
    count = 0
    for a, b in zip(levels(predicted), levels(gold)):
        if a != b:
            break
        count += 1
    return count


def parse_label(text: str) -> str | None:
    matches = BOXED_RE.findall(text or "")
    return canonical(matches[-1]) if matches else None


def split_of(application: str) -> str:
    bucket = hashlib.sha1(application.encode()).digest()[0] % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def application_rows(archive: Path) -> Iterator[dict]:
    """The fields the task needs from every application JSON in the sample archive, in archive order."""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            application = json.load(tar.extractfile(member))
            yield {
                "application": str(application["application_number"]),
                "title": (application.get("title") or "").strip(),
                "abstract": (application.get("abstract") or "").strip(),
                "label": application.get("main_cpc_label") or "",
                "decision": application.get("decision") or "",
            }


def check_table(rows: list[dict], archive: Path) -> None:
    """Refuse an empty or unusable extraction with the evidence needed to fix the reader."""
    if not rows:
        with tarfile.open(archive, "r:gz") as tar:
            names = [member.name for _, member in zip(range(5), tar)]
        raise ValueError(f"no application JSON files found in {archive}; first members: {names}")
    if not any(canonical(row["label"]) or candidates(row["label"]) for row in rows):
        examples = [row["label"] for row in rows[:5]]
        raise ValueError(f"none of the {len(rows)} applications has a CPC symbol in the expected form: {examples}")
    if not any(row["abstract"] for row in rows):
        raise ValueError(f"none of the {len(rows)} applications has an abstract")


def compact_table() -> Path:
    """The extracted table, one JSON line per application, built once from the 390 MB archive."""
    from huggingface_hub import constants, hf_hub_download

    archive = Path(hf_hub_download(DATASET, SAMPLE_FILE, repo_type="dataset", revision=REVISIONS[DATASET]))
    cache = Path(constants.HF_HUB_CACHE).parent / "rl-envs" / f"hupd-sample-jan-2016-{archive.stat().st_size}.jsonl"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        rows = list(application_rows(archive))
        check_table(rows, archive)
        partial = cache.with_suffix(".partial")
        with open(partial, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        partial.replace(cache)
    return cache


@lru_cache(maxsize=None)
def rows_for(split: str) -> tuple[tuple[str, str, str, str], ...]:
    """(application number, title, abstract, canonical main CPC symbol) for every usable row of a split.
    Rows whose label the scheme cannot resolve to exactly one symbol are left out."""
    scheme = scheme_keys()
    rows = []
    with open(compact_table(), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = resolve(row["label"], scheme)
            if label is None or not row["abstract"] or split_of(row["application"]) != split:
                continue
            rows.append((row["application"], row["title"], row["abstract"], label))
    return tuple(rows)


@lru_cache(maxsize=None)
def scheme_keys() -> frozenset[str]:
    """Every symbol in the CPC scheme, in the canonical form with a slash; only the key column is read."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SCHEME_DATASET, SCHEME_FILE, repo_type="dataset", revision=REVISIONS[SCHEME_DATASET])
    keys = pq.read_table(path, columns=["key"]).column("key").to_pylist()
    return frozenset(symbol for symbol in map(canonical, keys) if symbol)


def prompt_for(title: str, abstract: str) -> str:
    return (
        "The title and abstract of a patent application follow. Give its main Cooperative Patent "
        "Classification symbol down to the subgroup, in the form G06F 17/30. Reply with the symbol inside "
        f"\\boxed{{}}.\n\nTitle: {title}\nAbstract: {abstract}"
    )


class PatentClassificationData(vf.TaskData):
    split: str
    row: int
    application: str
    """The USPTO application number; the symbol is looked up at scoring time."""


class PatentClassificationTask(vf.Task[PatentClassificationData]):
    @property
    def key(self) -> str:
        return f"cpc:{self.data.application}"

    @property
    def _gold(self) -> str:
        return rows_for(self.data.split)[self.data.row][3]

    @vf.reward(weight=1.0)
    async def hierarchical(self, trace: vf.Trace) -> float:
        """Leading CPC levels shared with the office's symbol, over five."""
        return matched_levels(parse_label(trace.last_reply), self._gold) / 5

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        return float(parse_label(trace.last_reply) == self._gold)

    @vf.metric
    async def subclass(self, trace: vf.Trace) -> float:
        """The four-character subclass is right, the level examiners are assigned by."""
        return float(matched_levels(parse_label(trace.last_reply), self._gold) >= 3)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        return float(parse_label(trace.last_reply) is not None)

    @vf.metric
    async def valid_symbol(self, trace: vf.Trace) -> float:
        """The predicted symbol exists in the CPC scheme, right or wrong."""
        label = parse_label(trace.last_reply)
        return float(label is not None and label in scheme_keys())

    async def validate(self, runtime: vf.Runtime) -> bool:
        """A canonical symbol on an application with an abstract, for the application the key promises."""
        application, _title, abstract, gold = rows_for(self.data.split)[self.data.row]
        return application == self.data.application and canonical(gold) == gold and bool(abstract)


class PatentClassificationConfig(vf.TasksetConfig):
    split: Split = "validation"


class PatentClassificationTaskset(vf.Taskset[PatentClassificationTask, PatentClassificationConfig]):
    def load(self) -> Iterator[PatentClassificationTask]:
        c = self.config
        for row, (application, title, abstract, _gold) in enumerate(rows_for(c.split)):
            yield PatentClassificationTask(
                PatentClassificationData(
                    idx=row, prompt=prompt_for(title, abstract), split=c.split, row=row, application=application
                ),
                c.task,
            )
