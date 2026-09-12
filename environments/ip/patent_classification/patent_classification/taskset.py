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
REVISIONS: dict[str, str | None] = {DATASET: None, SCHEME_DATASET: None}
"""Dataset commits the rows come from. None follows the default branch; `scripts/pin_revisions.py`
prints the commits to pin once the rows have been fetched."""

Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CPC_RE = re.compile(r"^([A-HY])(\d{2})([A-Z])(\d{1,4})/(\d{2,6})$")


def canonical(label: str) -> str | None:
    """`G06F 17/30`, `g06f17/30` and `G06F17-30` all become `G06F17/30`; None if not a CPC symbol."""
    text = re.sub(r"\s+", "", (label or "").upper()).replace("-", "/")
    return text if CPC_RE.match(text) else None


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


def compact_table() -> Path:
    """The extracted table, one JSON line per application, built once from the 390 MB archive."""
    from huggingface_hub import constants, hf_hub_download

    archive = Path(hf_hub_download(DATASET, SAMPLE_FILE, repo_type="dataset", revision=REVISIONS[DATASET]))
    cache = Path(constants.HF_HUB_CACHE).parent / "rl-envs" / f"hupd-sample-jan-2016-{archive.stat().st_size}.jsonl"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix(".partial")
        with open(partial, "w", encoding="utf-8") as f:
            for row in application_rows(archive):
                f.write(json.dumps(row) + "\n")
        partial.replace(cache)
    return cache


@lru_cache(maxsize=None)
def rows_for(split: str) -> tuple[tuple[str, str, str, str], ...]:
    """(application number, title, abstract, canonical main CPC symbol) for every usable row of a split."""
    rows = []
    with open(compact_table(), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = canonical(row["label"])
            if label is None or not row["abstract"] or split_of(row["application"]) != split:
                continue
            rows.append((row["application"], row["title"], row["abstract"], label))
    return tuple(rows)


@lru_cache(maxsize=None)
def scheme_keys() -> frozenset[str]:
    """Every symbol in the CPC scheme, in the canonical form with a slash."""
    from datasets import load_dataset

    dataset = load_dataset(SCHEME_DATASET, split="train", revision=REVISIONS[SCHEME_DATASET])
    return frozenset(key for row in dataset for key in [canonical(row["key"])] if key)


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
