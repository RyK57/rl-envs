"""contract-review: the clause review a lawyer does on a commercial contract, on CUAD.

Environment nine, the third real vertical: legal. CUAD is 510 commercial contracts from SEC
filings in which attorneys and trained law students marked every passage that matters in a
deal review, under 41 categories. Two modes share one taskset:

- `classify`: one labelled clause, which of the 41 categories is it? Reward: the category named
  in `\\boxed{}` is one the annotators gave that clause (a clause can carry several).
- `extract`: an excerpt of a contract and one category; quote the passage that belongs to the
  category, or say none. Reward: token F1 against the annotated passage inside the excerpt, or
  1.0 for a correct none on an excerpt from a contract that has no such clause.

Both split by contract, never by row, so a contract's clauses are all train or all held out. Gold
stays off `TaskData` and is looked up by key at scoring time.
"""

import hashlib
import json
import re
import string
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal, NamedTuple

import verifiers.v1 as vf
from pydantic import Field

from contract_review.categories import CATEGORIES

CLAUSES_DATASET = "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification"
CUAD_DATASET = "theatticusproject/cuad"
CUAD_FILE = "CUAD_v1/CUAD_v1.json"
REVISIONS: dict[str, str | None] = {CLAUSES_DATASET: None, CUAD_DATASET: None}
"""Dataset commits the rows come from. None follows the default branch; pin with
`uv run python scripts/pin_revisions.py` once the rows have been fetched and inspected."""

Mode = Literal["classify", "extract"]
Split = Literal["train", "validation", "test"]
NONE = "none"
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CLAUSE_RE = re.compile(r"<clause>(.*?)</clause>", re.DOTALL)
CATEGORY_RE = re.compile(r'related to "([^"]+)"')
DETAILS_RE = re.compile(r"Details:\s*(.+)$", re.DOTALL)
ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """SQuAD-style: lower case, no punctuation, no articles, single spaces."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(word for word in text.split() if word not in ARTICLES)


LABELS_BY_KEY = {normalize(label): label for label in CATEGORIES}


def token_f1(predicted: str, gold: str) -> float:
    """Token overlap F1 between two normalized strings, the SQuAD measure."""
    p, g = normalize(predicted).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def parse_category(text: str) -> str | None:
    """The canonical category named in the last `\\boxed{}`, or None."""
    matches = BOXED_RE.findall(text or "")
    return LABELS_BY_KEY.get(normalize(matches[-1])) if matches else None


def parse_clause(text: str) -> str | None:
    """The last `<clause>` quote, `none` for an explicit none, or None when the reply has no clause tag."""
    matches = CLAUSE_RE.findall(text or "")
    if not matches:
        return None
    quote = matches[-1].strip()
    return NONE if quote.lower() == NONE else quote


def split_of(contract: str) -> str:
    """One in ten contracts to test, one in ten to validation, the rest to train, by name hash."""
    bucket = hashlib.sha1(contract.encode()).digest()[0] % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def classify_prompt(clause: str) -> str:
    listing = "\n".join(f"- {label}: {description}" for label, description in CATEGORIES.items())
    return (
        "The clause below comes from a commercial contract. Which one of these categories does it "
        f"fall under?\n\n{listing}\n\nReply with the category name inside \\boxed{{}}.\n\nClause:\n{clause}"
    )


def extract_prompt(category: str, description: str, excerpt: str) -> str:
    return (
        "Below is an excerpt from a commercial contract. Quote, exactly as written, the part of the "
        f'excerpt that relates to the category "{category}": {description}. If nothing in the '
        "excerpt relates to it, reply with <clause>none</clause>. Put the quote inside "
        f"<clause></clause>.\n\nExcerpt:\n{excerpt}"
    )


@lru_cache(maxsize=None)
def clause_rows(split: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(clause, categories the annotators gave it) for every distinct clause of a split."""
    from datasets import load_dataset

    dataset = load_dataset(CLAUSES_DATASET, split="train", revision=REVISIONS[CLAUSES_DATASET])
    labels: dict[tuple[str, str], list[str]] = {}
    for row in dataset:
        if split_of(row["file_name"]) != split:
            continue
        found = labels.setdefault((row["file_name"], row["clause"]), [])
        if row["label"] not in found:
            found.append(row["label"])
    return tuple((clause, tuple(found)) for (_, clause), found in labels.items())


class Passage(NamedTuple):
    doc: str
    category: str
    description: str
    start: int
    """Where the excerpt begins in the contract text."""
    golds: tuple[str, ...]
    """Annotated passages of this category inside the excerpt, clipped to it; empty for a negative."""


@lru_cache(maxsize=None)
def cuad() -> list[dict]:
    """The SQuAD-style CUAD file: one entry per contract, its full text, and 41 questions."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CUAD_DATASET, CUAD_FILE, repo_type="dataset", revision=REVISIONS[CUAD_DATASET])
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


@lru_cache(maxsize=None)
def documents() -> dict[str, str]:
    return {
        f"{entry['title']}#{p}": paragraph["context"]
        for entry in cuad()
        for p, paragraph in enumerate(entry["paragraphs"])
    }


def question_parts(question: str) -> tuple[str, str]:
    """(category, description) from a CUAD question; the description falls back to the question's own details."""
    category = CATEGORY_RE.search(question)
    name = category.group(1) if category else question
    label = LABELS_BY_KEY.get(normalize(name), name)
    details = DETAILS_RE.search(question)
    return label, CATEGORIES.get(label, details.group(1).strip() if details else name)


def excerpt_start(doc: str, category: str, text: str, window: int, first_span: tuple[int, int] | None) -> int:
    """Where the excerpt begins: a hashed offset so an annotated span sits anywhere in it, not always first."""
    seed = int.from_bytes(hashlib.sha1(f"{doc}|{category}".encode()).digest()[:4], "big")
    if first_span is None:
        return seed % max(1, len(text) - window + 1)
    span_start, span_len = first_span
    start = max(0, span_start - seed % (max(0, window - span_len) + 1))
    return min(start, max(0, len(text) - window))


def passages_of(doc: str, text: str, paragraph: dict, window: int) -> Iterator[Passage]:
    for qa in paragraph["qas"]:
        category, description = question_parts(qa["question"])
        spans = sorted((a["answer_start"], len(a["text"])) for a in qa["answers"] if a["text"].strip())
        start = excerpt_start(doc, category, text, window, spans[0] if spans else None)
        end = start + window
        golds = tuple(
            gold
            for s, n in spans
            if s < end and s + n > start
            for gold in [text[max(s, start) : min(s + n, end)]]
            if normalize(gold)
        )
        if spans and not golds:
            continue
        yield Passage(doc, category, description, start, golds)


@lru_cache(maxsize=None)
def passage_rows(split: str, window: int) -> tuple[Passage, ...]:
    """Every (contract, category) of a split as an excerpt, positives and negatives, in file order."""
    return tuple(
        passage
        for entry in cuad()
        if split_of(entry["title"]) == split
        for p, paragraph in enumerate(entry["paragraphs"])
        for passage in passages_of(f"{entry['title']}#{p}", paragraph["context"], paragraph, window)
    )


def excerpt_of(passage: Passage, window: int) -> str:
    return documents()[passage.doc][passage.start : passage.start + window]


class ContractReviewData(vf.TaskData):
    mode: str
    split: str
    window: int
    """Excerpt length in characters (extract mode); part of what defines the row."""
    row: int
    doc: str = ""
    category: str = ""


class ContractReviewTask(vf.Task[ContractReviewData]):
    @property
    def key(self) -> str:
        d = self.data
        if d.mode == "classify":
            return f"contract:classify:{d.split}:{d.row}"
        return f"contract:extract:{d.split}:{d.doc}:{d.category}"

    @property
    def _labels(self) -> tuple[str, ...]:
        return clause_rows(self.data.split)[self.data.row][1]

    @property
    def _passage(self) -> Passage:
        return passage_rows(self.data.split, self.data.window)[self.data.row]

    def _score(self, reply: str) -> float:
        if self.data.mode == "classify":
            return float(parse_category(reply) in self._labels)
        quote = parse_clause(reply)
        golds = self._passage.golds
        if quote is None:
            return 0.0
        if not golds:
            return float(quote == NONE)
        if quote == NONE:
            return 0.0
        return max(token_f1(quote, gold) for gold in golds)

    @vf.reward(weight=1.0)
    async def match(self, trace: vf.Trace) -> float:
        """Classify: the annotators' category. Extract: token F1 with the annotated passage, or a correct none."""
        return self._score(trace.last_reply)

    @vf.metric
    async def exact(self, trace: vf.Trace) -> float:
        if self.data.mode == "classify":
            return self._score(trace.last_reply)
        quote = parse_clause(trace.last_reply)
        golds = self._passage.golds
        if quote is None or quote == NONE:
            return float(quote == NONE and not golds)
        return float(any(normalize(quote) == normalize(gold) for gold in golds))

    @vf.metric
    async def formatted(self, trace: vf.Trace) -> float:
        parsed = parse_category(trace.last_reply) if self.data.mode == "classify" else parse_clause(trace.last_reply)
        return float(parsed is not None)

    @vf.metric
    async def abstained(self, trace: vf.Trace) -> float:
        """Extract mode only: the reply said none."""
        return float(self.data.mode == "extract" and parse_clause(trace.last_reply) == NONE)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Known categories on a non-empty clause; or a non-empty excerpt whose positives appear in it verbatim."""
        if self.data.mode == "classify":
            clause, labels = clause_rows(self.data.split)[self.data.row]
            return bool(clause.strip()) and bool(labels) and all(label in CATEGORIES for label in labels)
        passage = self._passage
        excerpt = excerpt_of(passage, self.data.window)
        return bool(excerpt.strip()) and all(gold in excerpt for gold in passage.golds)


class ContractReviewConfig(vf.TasksetConfig):
    mode: Mode = "classify"
    """`classify` a labelled clause into one of 41 categories, or `extract` the clause from an excerpt."""
    split: Split = "validation"
    window: int = Field(3000, ge=200)
    """Excerpt length in characters for `extract`."""
    negatives: bool = True
    """In `extract`, also ask about categories a contract does not contain (the right answer is none)."""


class ContractReviewTaskset(vf.Taskset[ContractReviewTask, ContractReviewConfig]):
    def load(self) -> Iterator[ContractReviewTask]:
        c = self.config
        if c.mode == "classify":
            for row, (clause, _labels) in enumerate(clause_rows(c.split)):
                data = ContractReviewData(
                    idx=row, prompt=classify_prompt(clause), mode=c.mode, split=c.split, window=c.window, row=row
                )
                yield ContractReviewTask(data, c.task)
            return
        for row, passage in enumerate(passage_rows(c.split, c.window)):
            if not passage.golds and not c.negatives:
                continue
            prompt = extract_prompt(passage.category, passage.description, excerpt_of(passage, c.window))
            data = ContractReviewData(
                idx=row,
                prompt=prompt,
                mode=c.mode,
                split=c.split,
                window=c.window,
                row=row,
                doc=passage.doc,
                category=passage.category,
            )
            yield ContractReviewTask(data, c.task)
