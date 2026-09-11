"""Offline checks for pyfix: the catalog discriminates, tasks are well formed, metrics behave."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import verifiers.v1 as vf
from pyfix.catalog import CATALOG
from pyfix.taskset import PRIVATE, PyfixTaskset, compiles


def run_tests(tmp_path: Path, name: str, source: str, tests: str) -> bool:
    (tmp_path / f"{name}.py").write_text(source)
    (tmp_path / f"test_{name}.py").write_text(tests)
    shutil.rmtree(tmp_path / "__pycache__", ignore_errors=True)
    return subprocess.run([sys.executable, "-B", f"test_{name}.py"], cwd=tmp_path, capture_output=True).returncode == 0


@pytest.mark.parametrize("entry", CATALOG, ids=lambda e: e["name"])
def test_fix_passes_and_bug_fails(entry, tmp_path: Path):
    assert run_tests(tmp_path, entry["name"], entry["fixed"], entry["tests"])
    assert not run_tests(tmp_path, entry["name"], entry["buggy"], entry["tests"])


def test_tasks_are_well_formed():
    tasks = PyfixTaskset(vf.TasksetConfig()).load()
    assert len(tasks) == len(CATALOG) == 13
    assert len({t.key for t in tasks}) == 13
    for t in tasks:
        assert f"`{t.data.name}.py`" in t.data.prompt_text and "bug" in t.data.prompt_text
        assert compiles(t.data.buggy) and compiles(PRIVATE[t.data.name]["fixed"])
        assert set(t.data.model_dump()).isdisjoint({"fixed", "tests"}), "private material stays off the trace"


async def test_metrics_read_the_captured_solution():
    task = PyfixTaskset(vf.TasksetConfig()).load()[0]
    trace = vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()), task=vf.TraceTask(type="PyfixTask", data=task.data))
    trace.info["solution"] = task.data.buggy
    assert await task.file_changed(trace) == 0.0 and await task.syntax_ok(trace) == 1.0
    trace.info["solution"] = PRIVATE[task.data.name]["fixed"]
    assert await task.file_changed(trace) == 1.0 and await task.syntax_ok(trace) == 1.0
    trace.info["solution"] = "def is_palindrome(s:"
    assert await task.syntax_ok(trace) == 0.0
