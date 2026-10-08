"""iupac-structure: the SMILES (or the molecular formula) of a compound from its IUPAC name.

Environment sixteen, chemistry. Reading a systematic name into a structure is the first step of
reading any chemistry text, patent claim or safety data sheet, and the reverse of what every
chemist is taught to do by hand. The rows are PubChem compounds that carry a preferred IUPAC name,
from a CC0 re-packaging of the January 2026 PubChem dump (public domain data), one of its shards
read by column and reduced once to a small cached table. Two targets: `smiles`, scored in the
runtime by RDKit through a PEP 723 script, 1.0 for the same canonical structure and otherwise half
the Morgan-fingerprint Tanimoto similarity of a parseable guess; `formula`, scored by exact match
on the Hill formula with no runtime at all. Gold stays off `TaskData` and is looked up by key at
scoring time.
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

import verifiers.v1 as vf
from pydantic import Field

DATASET = "hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC"
SHARD = "data/shard_001.parquet"
REVISION: str | None = None
"""The dataset commit the rows come from; `uv run python scripts/pin_revisions.py iupac_structure`
prints the current one."""
POOL = 20_000
"""Usable compounds kept from the shard, in shard order (the shards are globally shuffled upstream)."""
MAX_NAME_CHARS = 120
MAX_CACHED_SMILES_CHARS = 120
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()
"""The RDKit comparison, run in the rollout runtime as a uv script."""

Target = Literal["smiles", "formula"]
Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+(?:[+-]\d*)?$")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉−", "0123456789-")


class Row(NamedTuple):
    cid: str
    iupac: str
    smiles: str
    formula: str


def parse_answer(text: str) -> str | None:
    """The content of the last `\\boxed{}` as one token, or None without a box or with a multi-word answer."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1].strip().strip("`'\"").strip()
    return answer if answer and not re.search(r"\s", answer) else None


def normalize_formula(text: str) -> str:
    """`C₉H₈O₄` and `C9H8O4` compare equal; a unicode minus is a charge sign."""
    return re.sub(r"\s+", "", text or "").translate(SUBSCRIPTS)


def usable(record: dict) -> bool:
    """One neutral-looking fragment with a short ASCII name: no salts, isotopes, dummy atoms or lambda notation."""
    cid, smiles, formula, iupac = (
        record.get("CID"),
        record.get("SMILES_Canonical"),
        record.get("formula"),
        record.get("iupac"),
    )
    if not (cid and smiles and formula and iupac):
        return False
    return (
        iupac.isascii()
        and iupac.isprintable()
        and len(iupac) <= MAX_NAME_CHARS
        and "lambda" not in iupac
        and len(smiles) <= MAX_CACHED_SMILES_CHARS
        and "." not in smiles
        and "*" not in smiles
        and re.search(r"\[\d", smiles) is None
        and FORMULA_RE.match(formula) is not None
    )


def select_rows(records: Iterable[dict], pool: int = POOL) -> list[dict]:
    """The first `pool` usable records, in order."""
    rows: list[dict] = []
    for record in records:
        if usable(record):
            rows.append(
                {
                    "cid": str(record["CID"]),
                    "iupac": record["iupac"],
                    "smiles": record["SMILES_Canonical"],
                    "formula": record["formula"],
                }
            )
            if len(rows) == pool:
                break
    return rows


def split_of(cid: str) -> str:
    bucket = hashlib.sha1(cid.encode()).digest()[0] % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def compact_table() -> Path:
    """The usable compounds as JSON lines, built once from the shard next to the Hugging Face cache."""
    import pyarrow.parquet as pq
    from huggingface_hub import constants, hf_hub_download

    shard = Path(hf_hub_download(DATASET, SHARD, repo_type="dataset", revision=REVISION))
    cache = Path(constants.HF_HUB_CACHE).parent / "rl-envs" / f"pubchem-iupac-{shard.stat().st_size}-{POOL}.jsonl"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        columns = ["CID", "SMILES_Canonical", "formula", "iupac"]
        batches = pq.ParquetFile(shard).iter_batches(batch_size=65_536, columns=columns)
        rows = select_rows(record for batch in batches for record in batch.to_pylist())
        if len(rows) < POOL:
            raise ValueError(f"only {len(rows)} usable compounds in {DATASET}/{SHARD}, expected {POOL}")
        partial = cache.with_suffix(".partial")
        with open(partial, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        partial.replace(cache)
    return cache


@lru_cache(maxsize=None)
def rows_for(split: str, max_smiles_chars: int) -> tuple[Row, ...]:
    """The compounds of a split whose canonical SMILES fits the cap, in table order; the split is a hash of the CID."""
    rows = []
    with open(compact_table(), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if len(row["smiles"]) <= max_smiles_chars and split_of(row["cid"]) == split:
                rows.append(Row(row["cid"], row["iupac"], row["smiles"], row["formula"]))
    return tuple(rows)


def prompt_for(target: str, name: str, formula_hint: str | None) -> str:
    if target == "formula":
        return (
            "Give the molecular formula of the compound named below, in Hill order as PubChem writes it "
            f"(for example C9H8O4). Reply with the formula inside \\boxed{{}}.\n\nIUPAC name: {name}"
        )
    hint = f"\nMolecular formula: {formula_hint}" if formula_hint else ""
    return (
        "Write the SMILES string of the compound named below. Reply with the SMILES inside \\boxed{}."
        f"\n\nIUPAC name: {name}{hint}"
    )


class IupacStructureData(vf.TaskData):
    target: str
    """`smiles` or `formula`."""
    split: str
    row: int
    """Index of the compound within its split at this SMILES cap; the structure is looked up from it at scoring time."""
    cid: str
    """PubChem compound id, the row's durable identity."""
    max_smiles_chars: int
    """The cap the rows were selected under, so the lookup is reproducible from the trace."""


class IupacStructureTaskConfig(vf.TaskConfig):
    ignore_stereo: bool = False
    """Score SMILES answers with stereochemistry stripped from both the answer and the gold."""


class IupacStructureTask(vf.Task[IupacStructureData, vf.State, IupacStructureTaskConfig]):
    @property
    def key(self) -> str:
        return f"iupac:{self.data.target}:{self.data.cid}"

    @property
    def _gold(self) -> Row:
        return rows_for(self.data.split, self.data.max_smiles_chars)[self.data.row]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.data.target == "smiles":
            await runtime.prepare_uv_script(VERIFY)

    async def _run_verify(self, runtime: vf.Runtime, answer: str | None) -> dict:
        if answer is None:
            return {"parsed": False, "exact": False, "tanimoto": 0.0, "formula_match": False}
        stereo = "1" if self.config.ignore_stereo else "0"
        result = await runtime.run_uv_script(VERIFY, args=[self._gold.smiles, answer, stereo])
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-1000:]}")
        return json.loads(result.stdout.strip().splitlines()[-1])

    async def _report(self, trace: vf.Trace, runtime: vf.Runtime | None) -> dict:
        """One RDKit run per trace, shared by the reward and the metrics, which run concurrently."""
        if runtime is None:
            raise RuntimeError("a SMILES answer is scored by RDKit in the rollout runtime; it cannot be scored offline")
        pending = self.__dict__.setdefault("_pending", {})
        if trace.id not in pending:
            pending[trace.id] = asyncio.ensure_future(self._run_verify(runtime, parse_answer(trace.last_reply)))
        report = await pending[trace.id]
        trace.info["verify"] = report
        return report

    def _formula_right(self, trace: vf.Trace) -> bool:
        answer = parse_answer(trace.last_reply)
        return answer is not None and normalize_formula(answer) == normalize_formula(self._gold.formula)

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """`formula`: exact match. `smiles`: 1.0 for the same canonical structure, else half the Tanimoto similarity."""
        if self.data.target == "formula":
            return float(self._formula_right(trace))
        if parse_answer(trace.last_reply) is None:
            return 0.0
        report = await self._report(trace, runtime)
        if report["exact"]:
            return 1.0
        return 0.5 * float(report["tanimoto"]) if report["parsed"] else 0.0

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        """One boxed token was given."""
        return float(parse_answer(trace.last_reply) is not None)

    @vf.metric
    async def exact(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """The structure (or the formula) is exactly right."""
        if self.data.target == "formula":
            return float(self._formula_right(trace))
        if parse_answer(trace.last_reply) is None:
            return 0.0
        return float((await self._report(trace, runtime))["exact"])

    @vf.metric
    async def parsed(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """The answer is a valid SMILES (or a well-formed formula), right or wrong."""
        answer = parse_answer(trace.last_reply)
        if answer is None:
            return 0.0
        if self.data.target == "formula":
            return float(FORMULA_RE.match(normalize_formula(answer)) is not None)
        return float((await self._report(trace, runtime))["parsed"])

    @vf.metric
    async def similarity(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """Tanimoto similarity of the answer to the gold structure (0 or 1 for a formula)."""
        if self.data.target == "formula":
            return float(self._formula_right(trace))
        if parse_answer(trace.last_reply) is None:
            return 0.0
        return float((await self._report(trace, runtime))["tanimoto"])

    async def validate(self, runtime: vf.Runtime | None) -> bool:
        """The compound the key promises, with a name, a structure RDKit reads back as itself, and a Hill formula."""
        gold = self._gold
        if gold.cid != self.data.cid or not gold.iupac or not gold.smiles:
            return False
        if normalize_formula(gold.formula) != gold.formula or FORMULA_RE.match(gold.formula) is None:
            return False
        if self.data.target == "formula" or runtime is None:
            return True
        report = await self._run_verify(runtime, gold.smiles)
        return bool(report["exact"])


class IupacStructureConfig(vf.TasksetConfig):
    target: Target = "smiles"
    """`smiles`: write the structure, scored by RDKit in the runtime. `formula`: write the molecular formula, scored offline."""
    split: Split = "validation"
    max_smiles_chars: int = Field(60, ge=5, le=MAX_CACHED_SMILES_CHARS)
    """Keep compounds whose canonical SMILES is at most this long; the difficulty knob."""
    formula_hint: bool = False
    """In `smiles` mode, show the molecular formula next to the name."""
    task: IupacStructureTaskConfig = IupacStructureTaskConfig()


class IupacStructureTaskset(vf.Taskset[IupacStructureTask, IupacStructureConfig]):
    def load(self) -> Iterator[IupacStructureTask]:
        c = self.config
        for row, entry in enumerate(rows_for(c.split, c.max_smiles_chars)):
            hint = entry.formula if c.target == "smiles" and c.formula_hint else None
            yield IupacStructureTask(
                IupacStructureData(
                    idx=row,
                    prompt=prompt_for(c.target, entry.iupac, hint),
                    target=c.target,
                    split=c.split,
                    row=row,
                    cid=entry.cid,
                    max_smiles_chars=c.max_smiles_chars,
                ),
                c.task,
            )
