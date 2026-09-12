"""injury-coding: code a workplace injury narrative the way OSHA and workers' compensation insurers do.

Environment fourteen, workplace safety. Every severe injury an employer reports to OSHA is coded
with the Occupational Injury and Illness Classification System (OIICS): the nature of the injury,
the part of body, the event that caused it, and its source. Workers' compensation insurers code
every claim the same way, and reserving, rating and loss control key off the codes. The rows are
OSHA's Severe Injury Reports (US government data, public domain), downloaded from osha.gov: the
narrative the employer wrote and the codes OSHA assigned. Rewards are hierarchical over the code
digits, the OIICS division first. Gold stays off `TaskData` and is looked up by key at scoring time.
"""

import csv
import hashlib
import io
import re
import zipfile
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

import verifiers.v1 as vf
from pydantic import Field

REPORTS_URL = "https://www.osha.gov/sites/default/files/January2015toAugust2025.zip"
"""OSHA's severe injury report export, January 2015 to August 2025; a fixed file, not a live feed."""
FIELDS: dict[str, tuple[str, str, str]] = {
    "nature": ("nature", "naturetitle", "the nature of the injury or illness"),
    "part": ("partofbody", "partofbodytitle", "the part of body affected"),
    "event": ("event", "eventtitle", "the event or exposure that caused it"),
    "source": ("source", "sourcetitle", "the source of the injury"),
}
"""Per task field: the report's code column, its title column (header names normalized), and the question."""

Field_ = Literal["nature", "part", "event", "source"]
Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CODE_RE = re.compile(r"^[1-9][0-9]{0,3}$")


def header_key(name: str) -> str:
    """`Part of Body Title` and `part_of_body_title` both become `partofbodytitle`."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def normalize_code(value: str) -> str:
    """`4312`, `4312.0` and ` 4312 ` all become `4312`."""
    return re.sub(r"\.0+$", "", (value or "").strip())


def parse_code(text: str) -> str | None:
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    code = normalize_code(re.sub(r"[^0-9.]", "", matches[-1]))
    return code if CODE_RE.match(code) else None


def matched_prefix(predicted: str | None, gold: str) -> float:
    """Leading digits shared with the gold code, over the gold code's length; 0 without the division."""
    if predicted is None:
        return 0.0
    common = 0
    for a, b in zip(predicted, gold):
        if a != b:
            break
        common += 1
    return common / len(gold)


def split_of(report_id: str) -> str:
    bucket = hashlib.sha1(report_id.encode()).digest()[0] % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


class Report(NamedTuple):
    id: str
    narrative: str
    naics: str
    codes: dict[str, tuple[str, str]]
    """Per field: (code, title) as OSHA recorded them, normalized; missing fields are absent."""


USER_AGENT = "Mozilla/5.0 (compatible; rl-envs/0.1; +https://github.com/RyK57/rl-envs)"
"""osha.gov answers Python's default user agent with 403; a named client is let through."""


def reports_file() -> Path:
    """The export, downloaded once next to the Hugging Face cache."""
    import shutil
    from urllib.request import Request, urlopen

    from huggingface_hub import constants

    cache = Path(constants.HF_HUB_CACHE).parent / "rl-envs" / "osha-severe-injury-reports-2015-2025.zip"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix(".partial")
        with urlopen(Request(REPORTS_URL, headers={"User-Agent": USER_AGENT})) as response, open(partial, "wb") as f:
            shutil.copyfileobj(response, f)
        partial.replace(cache)
    return cache


def read_reports(archive: Path) -> Iterator[Report]:
    """Every report of the first CSV in the archive, in file order, with the columns the tasks need."""
    with zipfile.ZipFile(archive) as bundle:
        name = next(n for n in bundle.namelist() if n.lower().endswith(".csv"))
        with bundle.open(name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace"))
            columns = {header_key(h): h for h in reader.fieldnames or []}
            for row in reader:
                codes = {}
                for field, (code_key, title_key, _question) in FIELDS.items():
                    code = normalize_code(row.get(columns.get(code_key, ""), ""))
                    title = (row.get(columns.get(title_key, ""), "") or "").strip()
                    if CODE_RE.match(code) and title:
                        codes[field] = (code, title)
                yield Report(
                    id=(row.get(columns.get("id", ""), "") or "").strip(),
                    narrative=" ".join((row.get(columns.get("finalnarrative", ""), "") or "").split()),
                    naics=normalize_code(row.get(columns.get("primarynaics", ""), "")),
                    codes=codes,
                )


@lru_cache(maxsize=None)
def all_reports() -> tuple[Report, ...]:
    return tuple(read_reports(reports_file()))


@lru_cache(maxsize=None)
def rows_for(field: str, split: str, min_chars: int) -> tuple[Report, ...]:
    """Reports of a split that carry the field's code and a narrative of at least `min_chars`."""
    return tuple(
        report
        for report in all_reports()
        if report.id and field in report.codes and len(report.narrative) >= min_chars and split_of(report.id) == split
    )


@lru_cache(maxsize=None)
def known_codes(field: str) -> frozenset[str]:
    """Every code OSHA recorded for the field anywhere in the export."""
    return frozenset(report.codes[field][0] for report in all_reports() if field in report.codes)


def prompt_for(field: str, report: Report) -> str:
    question = FIELDS[field][2]
    industry = f"Industry (NAICS): {report.naics}\n" if report.naics else ""
    return (
        "The narrative of a severe workplace injury reported to OSHA follows. Give the OIICS code for "
        f"{question}. Reply with the numeric code inside \\boxed{{}}.\n\n{industry}Narrative: {report.narrative}"
    )


class InjuryCodingData(vf.TaskData):
    field: str
    split: str
    min_chars: int
    row: int
    report_id: str


class InjuryCodingTask(vf.Task[InjuryCodingData]):
    @property
    def key(self) -> str:
        return f"oiics:{self.data.field}:{self.data.report_id}"

    @property
    def _report(self) -> Report:
        return rows_for(self.data.field, self.data.split, self.data.min_chars)[self.data.row]

    @property
    def _gold(self) -> str:
        return self._report.codes[self.data.field][0]

    @vf.reward(weight=1.0)
    async def hierarchical(self, trace: vf.Trace) -> float:
        """Leading digits shared with OSHA's code, over its length: the division first, then each finer level."""
        return matched_prefix(parse_code(trace.last_reply), self._gold)

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        return float(parse_code(trace.last_reply) == self._gold)

    @vf.metric
    async def division(self, trace: vf.Trace) -> float:
        """The first digit, the OIICS division, is right."""
        code = parse_code(trace.last_reply)
        return float(code is not None and code[0] == self._gold[0])

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        return float(parse_code(trace.last_reply) is not None)

    @vf.metric
    async def valid_code(self, trace: vf.Trace) -> float:
        """The code is one OSHA uses for this field, right or wrong."""
        code = parse_code(trace.last_reply)
        return float(code is not None and code in known_codes(self.data.field))

    async def validate(self, runtime: vf.Runtime) -> bool:
        """A well-formed recorded code with a title, on a narrative, for the report the key promises."""
        report = self._report
        code, title = report.codes[self.data.field]
        return (
            report.id == self.data.report_id
            and CODE_RE.match(code) is not None
            and bool(title)
            and bool(report.narrative)
        )


class InjuryCodingConfig(vf.TasksetConfig):
    field: Field_ = "event"
    """Which OIICS code to give: `nature`, `part` (of body), `event` or `source`."""
    split: Split = "validation"
    min_chars: int = Field(60, ge=1)
    """Shortest narrative kept."""


class InjuryCodingTaskset(vf.Taskset[InjuryCodingTask, InjuryCodingConfig]):
    def load(self) -> Iterator[InjuryCodingTask]:
        c = self.config
        for row, report in enumerate(rows_for(c.field, c.split, c.min_chars)):
            yield InjuryCodingTask(
                InjuryCodingData(
                    idx=row,
                    prompt=prompt_for(c.field, report),
                    field=c.field,
                    split=c.split,
                    min_chars=c.min_chars,
                    row=row,
                    report_id=report.id,
                ),
                c.task,
            )
