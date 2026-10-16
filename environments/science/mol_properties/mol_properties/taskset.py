"""mol-properties: a computable property of a molecule from its SMILES.

Environment seventeen, chemistry. Reading a structure and counting what is in it, from the formula
and the molecular weight to rings and stereocentres, is the arithmetic of chemistry; every value here
is a pure function of the structure, so the gold is never stored. It is computed by RDKit inside the
rollout runtime, through a PEP 723 script, from the same SMILES the prompt shows. The rows are the
PubChem compounds of `iupac-structure`, which this package depends on for its cached table. Exact
properties (formula, counts) score 1 or 0; the two masses score 1.0 within a tolerance and then
decay linearly to 0 at five percent relative error.
"""

import asyncio
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NamedTuple

import verifiers.v1 as vf
from iupac_structure import taskset as iupac
from iupac_structure.taskset import MAX_CACHED_SMILES_CHARS, Row, normalize_formula
from pydantic import Field

PROPS = (Path(__file__).parent / "props.py").read_bytes()
"""The RDKit property calculator, run in the rollout runtime as a uv script."""

Split = Literal["train", "validation", "test"]
Property = Literal[
    "formula", "molecular_weight", "monoisotopic_mass", "heavy_atoms", "rings", "aromatic_rings", "stereocenters"
]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
RELATIVE_SLACK = 0.05
"""A numeric answer earns nothing from this relative error on."""


class Spec(NamedTuple):
    question: str
    kind: str
    tolerance: float


SPECS: dict[str, Spec] = {
    "formula": Spec("its molecular formula in Hill order (for example C9H8O4)", "formula", 0.0),
    "molecular_weight": Spec("its average molecular weight in g/mol, to two decimal places", "float", 0.1),
    "monoisotopic_mass": Spec("its monoisotopic mass in Da, to four decimal places", "float", 0.005),
    "heavy_atoms": Spec("its number of heavy (non-hydrogen) atoms", "int", 0.0),
    "rings": Spec("its number of rings, counted as the smallest set of smallest rings", "int", 0.0),
    "aromatic_rings": Spec("its number of aromatic rings", "int", 0.0),
    "stereocenters": Spec(
        "its number of tetrahedral stereocentres, whether or not their configuration is given", "int", 0.0
    ),
}


def rows_for(split: str, max_smiles_chars: int) -> tuple[Row, ...]:
    """The PubChem compounds of `iupac-structure`, through its cached table."""
    return iupac.rows_for(split, max_smiles_chars)


def parse_answer(text: str) -> str | None:
    """The content of the last `\\boxed{}` as one token, or None without a box or with a multi-word answer."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1].strip().strip("`'\"").strip()
    return answer if answer and not re.search(r"\s", answer) else None


def parse_number(answer: str) -> float | None:
    try:
        return float(answer.replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def score(spec: Spec, answer: str, gold: object) -> float:
    """1.0 for the right formula or count; a mass within tolerance, then a linear decay to 0 at 5% off."""
    if spec.kind == "formula":
        return float(normalize_formula(answer) == normalize_formula(str(gold)))
    value = parse_number(answer)
    if value is None:
        return 0.0
    if spec.kind == "int":
        return float(value == float(gold))
    error = abs(value - float(gold))
    if error <= spec.tolerance:
        return 1.0
    return max(0.0, 1.0 - error / (RELATIVE_SLACK * abs(float(gold)))) if gold else 0.0


def prompt_for(property_name: str, smiles: str, name: str | None) -> str:
    identity = f"SMILES: {smiles}" + (f"\nIUPAC name: {name}" if name else "")
    return (
        f"A molecule follows as SMILES. Give {SPECS[property_name].question}. "
        f"Reply with the value inside \\boxed{{}}.\n\n{identity}"
    )


class MolPropertiesData(vf.TaskData):
    property: str
    split: str
    row: int
    """Index of the compound within its split at this SMILES cap; the structure is looked up from it at scoring time."""
    cid: str
    """PubChem compound id, the row's durable identity."""
    max_smiles_chars: int


class MolPropertiesTask(vf.Task[MolPropertiesData]):
    @property
    def key(self) -> str:
        return f"molprop:{self.data.property}:{self.data.cid}"

    @property
    def _row(self) -> Row:
        return rows_for(self.data.split, self.data.max_smiles_chars)[self.data.row]

    @property
    def _spec(self) -> Spec:
        return SPECS[self.data.property]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(PROPS)

    async def _compute(self, runtime: vf.Runtime) -> dict:
        result = await runtime.run_uv_script(PROPS, args=[self._row.smiles])
        if result.exit_code != 0:
            raise RuntimeError(f"props.py failed: {result.stderr.strip()[-1000:]}")
        return json.loads(result.stdout.strip().splitlines()[-1])

    async def _properties(self, runtime: vf.Runtime | None) -> dict:
        """The structure's properties, computed once per task and shared by every rollout and hook;
        a failed computation is forgotten so the next rollout retries it."""
        if runtime is None:
            raise RuntimeError("the gold is computed by RDKit in the rollout runtime; it cannot be scored offline")
        pending = self.__dict__.setdefault("_pending", {})
        if "props" not in pending:
            pending["props"] = asyncio.ensure_future(self._compute(runtime))
        try:
            return await pending["props"]
        except BaseException:
            pending.pop("props", None)
            raise

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """Exact for a formula or a count; within tolerance for a mass, decaying to 0 at 5% off."""
        answer = parse_answer(trace.last_reply)
        if answer is None:
            return 0.0
        gold = (await self._properties(runtime))[self.data.property]
        trace.info["gold"] = gold
        return score(self._spec, answer, gold)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        """One boxed token was given."""
        return float(parse_answer(trace.last_reply) is not None)

    @vf.metric
    async def exact(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """The value is right (within tolerance for a mass)."""
        answer = parse_answer(trace.last_reply)
        if answer is None:
            return 0.0
        gold = (await self._properties(runtime))[self.data.property]
        return float(score(self._spec, answer, gold) == 1.0)

    @vf.metric
    async def abs_error(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """Distance from the gold for a numeric property; 0 or 1 for a formula; the gold itself when unparseable."""
        answer = parse_answer(trace.last_reply)
        gold = (await self._properties(runtime))[self.data.property]
        if self._spec.kind == "formula":
            return float(answer is None or normalize_formula(answer) != normalize_formula(str(gold)))
        value = parse_number(answer) if answer is not None else None
        return abs(value - float(gold)) if value is not None else abs(float(gold))

    async def validate(self, runtime: vf.Runtime | None) -> bool:
        """The compound the key promises, with a structure RDKit reads and a finite value for the property."""
        row = self._row
        if row.cid != self.data.cid or not row.smiles:
            return False
        if runtime is None:
            return True
        value = (await self._compute(runtime))[self.data.property]
        return isinstance(value, str) if self._spec.kind == "formula" else value == value and abs(value) < float("inf")


class MolPropertiesConfig(vf.TasksetConfig):
    property: Property = "formula"
    """Which property to ask for; every row of the split is asked the same one."""
    split: Split = "validation"
    max_smiles_chars: int = Field(60, ge=5, le=MAX_CACHED_SMILES_CHARS)
    """Keep compounds whose canonical SMILES is at most this long; the difficulty knob."""
    show_name: bool = False
    """Show the IUPAC name next to the SMILES."""


class MolPropertiesTaskset(vf.Taskset[MolPropertiesTask, MolPropertiesConfig]):
    def load(self) -> Iterator[MolPropertiesTask]:
        c = self.config
        for row, entry in enumerate(rows_for(c.split, c.max_smiles_chars)):
            yield MolPropertiesTask(
                MolPropertiesData(
                    idx=row,
                    prompt=prompt_for(c.property, entry.smiles, entry.iupac if c.show_name else None),
                    property=c.property,
                    split=c.split,
                    row=row,
                    cid=entry.cid,
                    max_smiles_chars=c.max_smiles_chars,
                ),
                c.task,
            )
