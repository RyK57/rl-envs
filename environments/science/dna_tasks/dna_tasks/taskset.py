"""dna-tasks: translate, reverse-complement, find the longest open reading frame, or measure GC content.

Environment eighteen, molecular biology, and the catalog's first infinite taskset. Every row is a
random DNA sequence generated from a seed and an index, so the taskset never ends and a run bounds
it with `-n`; the same seed, operation and length always yield the same sequences, so `--resume`
and a training run's eval sources see the same rows. The gold is a pure function of the sequence
(the standard genetic code, Watson-Crick complementarity, the ORF rule, a count) and is recomputed
at scoring time, never stored. Sequence answers earn their identity to the gold when not exact; the
GC percentage earns 1.0 within 0.05 points and decays to 0 at ten points off.
"""

import itertools
import random
import re
from collections.abc import Iterator
from difflib import SequenceMatcher
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

Operation = Literal["translate", "reverse_complement", "orf", "gc_content"]
BASES = "TCAG"
AMINO_ACIDS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: dict[str, str] = {
    a + b + c: AMINO_ACIDS[16 * i + 4 * j + k]
    for i, a in enumerate(BASES)
    for j, b in enumerate(BASES)
    for k, c in enumerate(BASES)
}
"""The standard genetic code, codon to one-letter amino acid, `*` for a stop."""
STOP_CODONS = tuple(codon for codon, aa in CODON_TABLE.items() if aa == "*")
CODING_CODONS = tuple(codon for codon, aa in CODON_TABLE.items() if aa != "*")
COMPLEMENT = str.maketrans("ACGT", "TGCA")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
GC_TOLERANCE = 0.05
GC_SLACK = 10.0


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def translate(sequence: str) -> str:
    """Codons from the first base up to the first stop codon (excluded); a trailing partial codon is ignored."""
    protein = []
    for i in range(0, len(sequence) - 2, 3):
        amino_acid = CODON_TABLE[sequence[i : i + 3]]
        if amino_acid == "*":
            break
        protein.append(amino_acid)
    return "".join(protein)


def longest_orf(sequence: str) -> str:
    """The protein of the longest complete ATG-to-stop frame on the forward strand, the earliest when
    tied, or the empty string when no frame reaches a stop codon."""
    best_start, best_length = None, 0
    for start in range(len(sequence) - 2):
        if sequence[start : start + 3] != "ATG":
            continue
        for end in range(start + 3, len(sequence) - 2, 3):
            if sequence[end : end + 3] in STOP_CODONS:
                if end + 3 - start > best_length:
                    best_start, best_length = start, end + 3 - start
                break
    return translate(sequence[best_start:]) if best_start is not None else ""


def gc_content(sequence: str) -> float:
    """GC bases as a percentage of the sequence."""
    return 100.0 * sum(1 for base in sequence if base in "GC") / len(sequence)


def gold_for(operation: str, sequence: str) -> str:
    if operation == "translate":
        return translate(sequence)
    if operation == "reverse_complement":
        return reverse_complement(sequence)
    if operation == "orf":
        return longest_orf(sequence)
    return f"{gc_content(sequence):.1f}"


def generate(operation: str, length: int, rng: random.Random) -> str:
    """One random sequence for the operation: a complete coding sequence, a sequence with a planted
    complete ORF, a sequence with a random GC bias, or a uniformly random one."""
    if operation == "translate":
        body = "".join(rng.choice(CODING_CODONS) for _ in range(length // 3 - 2))
        return "ATG" + body + rng.choice(STOP_CODONS)
    if operation == "orf":
        codons = rng.randint(2, max(2, length // 3 - 3))
        orf = "ATG" + "".join(rng.choice(CODING_CODONS) for _ in range(codons)) + rng.choice(STOP_CODONS)
        offset = rng.randint(0, length - len(orf))
        flanks = "".join(rng.choice("ACGT") for _ in range(length - len(orf)))
        return flanks[:offset] + orf + flanks[offset:]
    if operation == "gc_content":
        bias = rng.uniform(0.25, 0.75)
        return "".join(rng.choice("GC") if rng.random() < bias else rng.choice("AT") for _ in range(length))
    return "".join(rng.choice("ACGT") for _ in range(length))


def parse_answer(operation: str, text: str) -> str | None:
    """The content of the last `\\boxed{}` as one token, upper-cased for a sequence, or None."""
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1].strip().strip("`'\"").strip()
    if not answer or re.search(r"\s", answer):
        return None
    return answer.rstrip("%") if operation == "gc_content" else answer.upper()


def score(operation: str, answer: str, gold: str) -> float:
    """Exact 1.0; a sequence otherwise earns its identity to the gold, a GC percentage decays to 0 at ten points off."""
    if operation == "gc_content":
        try:
            value = float(answer)
        except ValueError:
            return 0.0
        error = abs(value - float(gold))
        return 1.0 if error <= GC_TOLERANCE else max(0.0, 1.0 - error / GC_SLACK)
    if answer == gold:
        return 1.0
    return SequenceMatcher(None, answer, gold).ratio() if gold else 0.0


PROMPTS = {
    "translate": (
        "Translate the coding DNA sequence below with the standard genetic code, from its first codon to the "
        "stop codon. Reply with the protein in one-letter amino acid codes, without the stop, inside \\boxed{}."
    ),
    "reverse_complement": "Give the reverse complement of the DNA sequence below. Reply with the sequence inside \\boxed{}.",
    "orf": (
        "Find the longest open reading frame on the forward strand of the DNA sequence below (from ATG to the "
        "first in-frame stop codon, standard genetic code) and translate it. Reply with the protein in one-letter "
        "amino acid codes, without the stop, inside \\boxed{}."
    ),
    "gc_content": (
        "Give the GC content of the DNA sequence below as a percentage, to one decimal place. Reply with the "
        "number inside \\boxed{}."
    ),
}


def prompt_for(operation: str, sequence: str) -> str:
    return f"{PROMPTS[operation]}\n\nDNA: {sequence}"


class DnaTasksData(vf.TaskData):
    operation: str
    sequence: str
    """The sequence the prompt shows; the gold is a pure function of it."""
    seed: int
    length: int


class DnaTasksTask(vf.Task[DnaTasksData]):
    @property
    def key(self) -> str:
        return f"dna:{self.data.operation}:{self.data.length}:{self.data.seed}:{self.data.idx}"

    @property
    def _gold(self) -> str:
        return gold_for(self.data.operation, self.data.sequence)

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        """Exact 1.0; else the sequence identity to the gold, or the GC percentage's decayed closeness."""
        answer = parse_answer(self.data.operation, trace.last_reply)
        return 0.0 if answer is None else score(self.data.operation, answer, self._gold)

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        answer = parse_answer(self.data.operation, trace.last_reply)
        return float(answer is not None and score(self.data.operation, answer, self._gold) == 1.0)

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        """One boxed token was given."""
        return float(parse_answer(self.data.operation, trace.last_reply) is not None)

    @vf.metric
    async def length_error(self, trace: vf.Trace) -> float:
        """How many characters the answer is off in length (a missing answer counts the whole gold)."""
        answer = parse_answer(self.data.operation, trace.last_reply)
        if self.data.operation == "gc_content":
            return 0.0
        return float(abs(len(answer) - len(self._gold))) if answer is not None else float(len(self._gold))

    async def validate(self, runtime: vf.Runtime | None) -> bool:
        """A clean sequence whose gold, written as instructed, scores 1.0; a coding sequence is complete and an
        ORF row has one."""
        sequence, operation = self.data.sequence, self.data.operation
        if not sequence or set(sequence) - set("ACGT"):
            return False
        gold = self._gold
        if operation == "translate":
            complete = sequence.startswith("ATG") and len(sequence) % 3 == 0 and sequence[-3:] in STOP_CODONS
            if not complete or len(gold) * 3 != len(sequence) - 3:
                return False
        if operation == "orf" and not gold:
            return False
        return score(operation, parse_answer(operation, f"\\boxed{{{gold}}}"), gold) == 1.0


class DnaTasksConfig(vf.TasksetConfig):
    operation: Operation = "translate"
    """Which operation every row asks for."""
    length: int = Field(60, ge=12, le=3000)
    """Sequence length in bases (a coding sequence is rounded down to whole codons); the difficulty knob."""
    seed: int = 0
    """Generator seed; the same seed, operation and length always yield the same sequences."""


class DnaTasksTaskset(vf.Taskset[DnaTasksTask, DnaTasksConfig]):
    INFINITE = True

    def load(self) -> Iterator[DnaTasksTask]:
        c = self.config
        for idx in itertools.count():
            rng = random.Random(f"{c.seed}:{c.operation}:{c.length}:{idx}")
            sequence = generate(c.operation, c.length, rng)
            yield DnaTasksTask(
                DnaTasksData(
                    idx=idx,
                    prompt=prompt_for(c.operation, sequence),
                    operation=c.operation,
                    sequence=sequence,
                    seed=c.seed,
                    length=c.length,
                ),
                c.task,
            )
