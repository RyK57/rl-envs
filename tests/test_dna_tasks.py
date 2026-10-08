"""Offline checks for dna-tasks: the genetic code, the four operations, the generator, and the hooks."""

import random

import pytest
import verifiers.v1 as vf
from dna_tasks.taskset import (
    CODON_TABLE,
    STOP_CODONS,
    DnaTasksConfig,
    DnaTasksTask,
    DnaTasksTaskset,
    gc_content,
    generate,
    gold_for,
    longest_orf,
    parse_answer,
    reverse_complement,
    score,
    translate,
)
from verifiers.v1.graph import MessageNode


def make_trace(task: DnaTasksTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DnaTasksTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def test_genetic_code_and_operations():
    assert len(CODON_TABLE) == 64 and CODON_TABLE["ATG"] == "M" and CODON_TABLE["TGG"] == "W"
    assert set(STOP_CODONS) == {"TAA", "TAG", "TGA"}
    assert translate("ATGGCCTAAGGG") == "MA", "stops at the first stop codon"
    assert translate("ATGGC") == "M", "a trailing partial codon is ignored"
    assert reverse_complement("ATGC") == "GCAT" and reverse_complement(reverse_complement("AACCGGTT")) == "AACCGGTT"
    assert longest_orf("CCATGAAATAGATGGCCGCCTAACC") == "MAA", "the longer of two complete frames"
    assert longest_orf("ATGAAAATGTAG") == "MKM", "the earliest start when nested"
    assert longest_orf("ATGAAACCC") == "" and longest_orf("CCCCCC") == "", "a frame without a stop is not an ORF"
    assert gc_content("GGCCAT") == pytest.approx(200 / 3) and gold_for("gc_content", "GGCCAT") == "66.7"


def test_generate_shapes():
    rng = random.Random(0)
    coding = generate("translate", 60, rng)
    assert coding.startswith("ATG") and coding[-3:] in STOP_CODONS and len(coding) == 60
    assert "*" not in translate(coding) and len(translate(coding)) == 19
    with_orf = generate("orf", 90, rng)
    assert len(with_orf) == 90 and longest_orf(with_orf)
    biased = generate("gc_content", 100, rng)
    assert len(biased) == 100 and not set(biased) - set("ACGT")
    assert len(generate("reverse_complement", 33, rng)) == 33


def test_parse_and_score():
    assert parse_answer("translate", "\\boxed{mfh}") == "MFH" and parse_answer("translate", "MFH") is None
    assert parse_answer("gc_content", "\\boxed{41.7%}") == "41.7" and parse_answer("translate", "\\boxed{M F}") is None
    assert score("translate", "MFH", "MFH") == 1.0 and score("translate", "MFX", "MFH") == pytest.approx(2 / 3)
    assert score("reverse_complement", "", "ACGT") == 0.0
    assert score("gc_content", "41.7", "41.7") == 1.0 and score("gc_content", "46.7", "41.7") == pytest.approx(0.5)
    assert score("gc_content", "60", "41.7") == 0.0 and score("gc_content", "many", "41.7") == 0.0


def test_config_keeps_the_framework_task_field():
    assert isinstance(DnaTasksConfig().task, vf.TaskConfig) and DnaTasksConfig().operation == "translate"


async def test_taskset_is_infinite_deterministic_and_validates():
    taskset = DnaTasksTaskset(DnaTasksConfig(operation="orf", length=90))
    assert taskset.INFINITE
    tasks = list(taskset.head(5))
    again = list(DnaTasksTaskset(DnaTasksConfig(operation="orf", length=90)).head(5))
    assert [t.data.sequence for t in tasks] == [t.data.sequence for t in again], "the same seed yields the same rows"
    other = list(DnaTasksTaskset(DnaTasksConfig(operation="orf", length=90, seed=1)).head(5))
    assert [t.data.sequence for t in tasks] != [t.data.sequence for t in other]
    assert [t.key for t in tasks] == [f"dna:orf:90:0:{i}" for i in range(5)]
    assert all([await t.validate(runtime=None) for t in tasks])
    assert set(tasks[0].data.model_dump()).isdisjoint({"gold", "protein", "answer"})
    for operation in ("translate", "reverse_complement", "gc_content"):
        rows = list(DnaTasksTaskset(DnaTasksConfig(operation=operation)).head(20))
        assert all([await t.validate(runtime=None) for t in rows]) and len({t.key for t in rows}) == 20


async def test_reward_and_metrics():
    task = list(DnaTasksTaskset(DnaTasksConfig(operation="translate", length=30)).head(1))[0]
    gold = task._gold
    exact = make_trace(task, f"The protein is \\boxed{{{gold.lower()}}}")
    partial = make_trace(task, f"\\boxed{{{gold[:-1]}X}}")
    unformatted = make_trace(task, gold)
    assert await task.correct(exact) == 1.0 and await task.exact(exact) == 1.0 and await task.formatted(exact) == 1.0
    assert 0.0 < await task.correct(partial) < 1.0 and await task.exact(partial) == 0.0
    assert await task.length_error(partial) == 0.0
    assert await task.correct(unformatted) == 0.0 and await task.formatted(unformatted) == 0.0
    assert await task.length_error(unformatted) == float(len(gold))
    gc = list(DnaTasksTaskset(DnaTasksConfig(operation="gc_content", length=40)).head(1))[0]
    assert await gc.correct(make_trace(gc, f"\\boxed{{{gc._gold}}}")) == 1.0
    assert await gc.correct(make_trace(gc, f"\\boxed{{{float(gc._gold) + 5:.1f}}}")) == pytest.approx(0.5)
