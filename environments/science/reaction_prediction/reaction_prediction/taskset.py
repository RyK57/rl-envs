"""reaction-prediction: the product of an organic reaction from its reactants.

Environment nineteen, synthetic chemistry. Predicting what a reaction makes is the daily work of
process and medicinal chemists, and it has an exact answer in the patent record. The rows are
USPTO-50K (Schneider et al., 2016), the 50,016 single-product reactions extracted from US patents
by Lowe (CC0), as mirrored on the Hub with its standard train / val / test split. The reactants reach
the prompt with their atom-map numbers stripped, since the mapping would reveal which atoms survive
into the product, and both sides are compared by RDKit in the rollout runtime with the maps cleared:
1.0 for the same canonical product, otherwise half the Tanimoto similarity of a parseable guess.
Gold stays off `TaskData` and is looked up by key at scoring time.
"""

import asyncio
import json
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

import verifiers.v1 as vf

DATASET = "bisectgroup/USPTO_50K"
SHARDS = 6
FILE_SPLITS = {"train": "train", "validation": "val", "test": "test"}
REVISION: str | None = None
"""The dataset commit the rows come from; `uv run python scripts/pin_revisions.py reaction_prediction`
prints the current one."""
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()
"""The RDKit comparison, run in the rollout runtime as a uv script."""

Split = Literal["train", "validation", "test"]
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
MAP_RE = re.compile(r":\d+\]")


class Row(NamedTuple):
    reaction_id: str
    reactants: str
    product: str


def strip_maps(smiles: str) -> str:
    """`[CH3:1]` becomes `[CH3]`: the atom mapping is the answer key, not part of the question."""
    return MAP_RE.sub("]", smiles or "")


def parse_answer(text: str) -> str | None:
    """The content of the last `\\boxed{}` as one token, or None without a box or with a multi-word answer."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1].strip().strip("`'\"").strip()
    return answer if answer and not re.search(r"\s", answer) else None


def read_shards(paths: list[str]) -> tuple[Row, ...]:
    """(patent id, reactants, product) for every single-product reaction in the shards, in file order."""
    import pyarrow.parquet as pq

    rows: list[Row] = []
    for path in paths:
        table = pq.read_table(path, columns=["id", "reactants", "product"])
        for record in table.to_pylist():
            reactants, product = strip_maps(record["reactants"]), strip_maps(record["product"])
            if reactants and product and "." not in product:
                rows.append(Row(str(record["id"]), reactants, product))
    if not rows:
        raise ValueError(f"no usable reactions in {len(paths)} shards of {DATASET}")
    return tuple(rows)


@lru_cache(maxsize=None)
def rows_for(split: str) -> tuple[Row, ...]:
    from huggingface_hub import hf_hub_download

    name = FILE_SPLITS[split]
    paths = [
        hf_hub_download(DATASET, f"data/{name}-{i:05d}-of-{SHARDS:05d}.parquet", repo_type="dataset", revision=REVISION)
        for i in range(SHARDS)
    ]
    return read_shards(paths)


def prompt_for(reactants: str) -> str:
    return (
        "The reactants of an organic reaction follow as SMILES, separated by dots. Write the SMILES of the "
        f"product. Reply with the product SMILES inside \\boxed{{}}.\n\nReactants: {reactants}"
    )


class ReactionPredictionData(vf.TaskData):
    split: str
    row: int
    """Index of the reaction within its split; the product is looked up from it at scoring time."""
    reaction_id: str
    """The patent the reaction was extracted from (one patent holds several reactions)."""


class ReactionPredictionTaskConfig(vf.TaskConfig):
    ignore_stereo: bool = False
    """Compare the product with stereochemistry stripped from both the answer and the gold."""


class ReactionPredictionTask(vf.Task[ReactionPredictionData, vf.State, ReactionPredictionTaskConfig]):
    @property
    def key(self) -> str:
        return f"rxn:{self.data.split}:{self.data.row}:{self.data.reaction_id}"

    @property
    def _gold(self) -> Row:
        return rows_for(self.data.split)[self.data.row]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(VERIFY)

    async def _run_verify(self, runtime: vf.Runtime, answer: str | None) -> dict:
        if answer is None:
            return {"parsed": False, "exact": False, "tanimoto": 0.0, "formula_match": False}
        stereo = "1" if self.config.ignore_stereo else "0"
        result = await runtime.run_uv_script(VERIFY, args=[self._gold.product, answer, stereo])
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-1000:]}")
        return json.loads(result.stdout.strip().splitlines()[-1])

    async def _report(self, trace: vf.Trace, runtime: vf.Runtime | None) -> dict:
        """One RDKit run per trace, shared by the reward and the metrics, which run concurrently."""
        if runtime is None:
            raise RuntimeError("a product is scored by RDKit in the rollout runtime; it cannot be scored offline")
        pending = self.__dict__.setdefault("_pending", {})
        if trace.id not in pending:
            pending[trace.id] = asyncio.ensure_future(self._run_verify(runtime, parse_answer(trace.last_reply)))
        report = await pending[trace.id]
        trace.info["verify"] = report
        return report

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """1.0 for the recorded product, else half the Tanimoto similarity of a parseable guess."""
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
        if parse_answer(trace.last_reply) is None:
            return 0.0
        return float((await self._report(trace, runtime))["exact"])

    @vf.metric
    async def parsed(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """The answer is a valid SMILES, right or wrong."""
        if parse_answer(trace.last_reply) is None:
            return 0.0
        return float((await self._report(trace, runtime))["parsed"])

    @vf.metric
    async def similarity(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """Tanimoto similarity of the answer to the recorded product."""
        if parse_answer(trace.last_reply) is None:
            return 0.0
        return float((await self._report(trace, runtime))["tanimoto"])

    @vf.metric
    async def formula_match(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> float:
        """The answer has the product's molecular formula, right or wrong structure."""
        if parse_answer(trace.last_reply) is None:
            return 0.0
        return float((await self._report(trace, runtime))["formula_match"])

    async def validate(self, runtime: vf.Runtime | None) -> bool:
        """The reaction the key promises, with reactants, a single product RDKit reads back as itself, and no maps left."""
        gold = self._gold
        if gold.reaction_id != self.data.reaction_id or not gold.reactants or not gold.product:
            return False
        if MAP_RE.search(gold.reactants) or MAP_RE.search(gold.product) or "." in gold.product:
            return False
        if runtime is None:
            return True
        return bool((await self._run_verify(runtime, gold.product))["exact"])


class ReactionPredictionConfig(vf.TasksetConfig):
    split: Split = "validation"
    """USPTO-50K's own split: `train` (40,008), `validation` (5,001) or `test` (5,007)."""
    task: ReactionPredictionTaskConfig = ReactionPredictionTaskConfig()


class ReactionPredictionTaskset(vf.Taskset[ReactionPredictionTask, ReactionPredictionConfig]):
    def load(self) -> Iterator[ReactionPredictionTask]:
        c = self.config
        for row, entry in enumerate(rows_for(c.split)):
            yield ReactionPredictionTask(
                ReactionPredictionData(
                    idx=row, prompt=prompt_for(entry.reactants), split=c.split, row=row, reaction_id=entry.reaction_id
                ),
                c.task,
            )
