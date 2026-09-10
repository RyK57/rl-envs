"""pyfix: fix one planted bug in a small Python function, verified by hidden tests in a sandbox.

Environment four. The new mechanism is the runtime: `setup()` writes the buggy file into the
box, the model edits it with a shell and an editor (the `bash` harness), `finalize()` reads the
result back into the trace, and the reward writes the hidden tests into the box and runs them
there. The gold never enters the box during the run, and the reward is "the tests pass", not
"the answer matches". `validate()` proves, inside a runtime, that the reference fix passes the
hidden tests and the buggy version fails them.
"""

import verifiers.v1 as vf

from pyfix.catalog import CATALOG, Entry


def prompt_for(entry: Entry) -> str:
    return (
        f"The file `{entry['name']}.py` in the current directory defines `{entry['signature']}`, "
        f"which should {entry['description']}. It has a bug. Fix the function in place, keeping its "
        "name and signature."
    )


def compiles(source: str) -> bool:
    try:
        compile(source, "<solution>", "exec")
    except SyntaxError:
        return False
    return True


class PyfixData(vf.TaskData):
    name: str
    """Module and function name; the file is `<name>.py`."""
    buggy: str
    """The source the model starts from."""
    fixed: str
    """A reference fix, used only by `validate`."""
    tests: str
    """Hidden test script: imports the function and asserts; exit code 0 means pass."""


class PyfixTask(vf.Task[PyfixData]):
    @property
    def key(self) -> str:
        return f"pyfix:{self.data.name}"

    @property
    def _file(self) -> str:
        return f"{self.data.name}.py"

    @property
    def _test_file(self) -> str:
        return f"test_{self.data.name}.py"

    async def _run_tests(self, runtime: vf.Runtime) -> bool:
        # Stale bytecode can outlive an edit made within the same second as the import that
        # cached it (pyc validity is mtime + size), so never trust or write the cache here.
        await runtime.write(self._test_file, self.data.tests.encode())
        await runtime.run(["rm", "-rf", "__pycache__"], {})
        result = await runtime.run(["python3", "-B", self._test_file], {})
        return result.exit_code == 0

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.write(self._file, self.data.buggy.encode())

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # The model may have deleted or renamed the file; that is a valid (losing) outcome.
        try:
            trace.info["solution"] = (await runtime.read(self._file)).decode()
        except vf.SandboxError:
            trace.info["solution"] = ""

    @vf.reward(weight=1.0)
    async def tests_pass(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return float(await self._run_tests(runtime))

    @vf.metric
    async def file_changed(self, trace: vf.Trace) -> float:
        """1.0 when the final file differs from the buggy one at all."""
        return float(trace.info.get("solution", "") != self.data.buggy)

    @vf.metric
    async def syntax_ok(self, trace: vf.Trace) -> float:
        """1.0 when the final file still compiles."""
        return float(compiles(trace.info.get("solution", "")))

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The hidden tests discriminate: the reference fix passes them and the buggy source fails."""
        await runtime.write(self._file, self.data.fixed.encode())
        if not await self._run_tests(runtime):
            return False
        await runtime.write(self._file, self.data.buggy.encode())
        return not await self._run_tests(runtime)


class PyfixTaskset(vf.Taskset[PyfixTask, vf.TasksetConfig]):
    def load(self) -> list[PyfixTask]:
        return [
            PyfixTask(PyfixData(idx=i, prompt=prompt_for(entry), **entry), self.config.task)
            for i, entry in enumerate(CATALOG)
        ]
