# analog-sizing

Size a two-stage Miller-compensated CMOS op-amp to a specification, graded by ngspice. Agentic: the
model works in a box that has ngspice and a simulator script, under any shell-capable harness. Sizing
an op-amp to a spec is the first exercise of every analog designer, and it has an exact grader: in
simulation the circuit either meets each line of the spec or it does not.

Each task is a spec with five lines: DC gain, unity-gain frequency, phase margin, supply current and
active area. The topology, device models and testbench are fixed; the agent chooses twelve numbers
(W and L for the five distinct devices, the compensation capacitor and the bias current) and writes
them to `sizes.json`. `python3 sim.py` in the box simulates the current sizes and prints the same
per-line satisfaction ratios and score that grading uses.

## Taskset

- **Source:** generated, not downloaded. `analog_sizing.generate` samples device sizes log-uniformly,
  simulates them, keeps the ones that behave like an op-amp (40 to 110 dB, over 1 MHz, 45 to 89
  degrees of phase margin, under 2 mA) and sets each spec line 20% (`--slack`) inside what that gold
  sizing achieves. A task whose spec the starting sizes already meet is discarded, so doing nothing
  never scores 1.0.
- **Splits:** `data/train.jsonl`, 300 tasks from seed 0, and `data/eval.jsonl`, 100 tasks from seed 1
  (338 samples drawn), never trained on. Both ask for 50 to 104 dB of gain, 40 to 75 degrees of phase
  margin and 16 µA to 2.4 mA of supply current; the training specs span 2.2 to 240 MHz and 24 to
  540 µm², the held-out ones 2.5 to 84 MHz and 26 to 400 µm².
- **Models:** Level-1 MOSFETs, so one simulation takes about 12 ms and needs no PDK download. A real
  PDK such as SKY130 would change the `.model` lines and add an include; the reward, the generator and
  the grader stay the same.
- **Image:** `Dockerfile`, `python:3.12-slim` plus ngspice. Build it once for the docker runtime, or
  push it with `prime images push` and pass the reference as `--env.taskset.image`.

Task keys are the task names, `opamp2-s<seed>-<index>`. The gold sizing stays off `TaskData`: it is
read from the task file by name when `validate` needs it, so it never reaches a trace.

Regenerate or extend the task files with ngspice on this machine:

```bash
uv run python -m analog_sizing.generate -n 300 --seed 0 [--slack 0.2]                 # data/train.jsonl
uv run python -m analog_sizing.generate -n 100 --seed 1 -o <package>/data/eval.jsonl
```

## Config

| Knob | Default | Meaning |
| --- | --- | --- |
| `--env.taskset.split` | `train` | `train` (300 tasks, seed 0) or `eval` (100 tasks, seed 1) |
| `--env.taskset.source` | none | a JSONL of generated tasks in place of the bundled split |
| `--env.taskset.image` | `analog-sizing:latest` | the container image with python3 and ngspice |

## Signals

- **Reward** `spec_score` (weight 1.0). Each spec line gets a satisfaction ratio clipped to [0, 1]:
  `value / bound` for a lower bound and `bound / value` for an upper bound, so overshooting a line
  earns nothing extra. The score is the mean performance ratio (gain, unity-gain frequency, phase
  margin) times the mean budget ratio (supply current, area): 1.0 iff every line is met, and 0 for a
  circuit that does not amplify however small and cheap it is. Sizes outside the allowed ranges, and a
  missing or malformed `sizes.json`, score 0.
- **Metric** `solved`: every line met.
- **Metric** `valid_sizes`: `sizes.json` parsed into twelve in-range numbers.
- **Metric** `sim_ok`: ngspice measured every quantity.
- **Metrics** `gain_db`, `ugf_mhz`, `pm_deg`, `idd_ua`, `area_um2`: what the submitted sizes measure.

Why it is hard to game: grading re-simulates the submitted sizes in a fresh box of the same image
with pristine copies of the netlist, the models and `sim.py`, so nothing the agent changed in its own
box (netlist, models, helper or the ngspice binary) is trusted; and `validate` proves every task
feasible by simulating its gold sizing.

## Run

```bash
docker build -t analog-sizing:latest environments/hardware/analog_sizing
uv run validate analog-sizing --runtime.type docker -n 5          # the gold sizing meets every spec
uv run --env-file .env eval @ configs/analog_sizing.toml --no-rich -v
```

`validate` needs a container runtime, because the tasks name an image. Without Docker,
`eval --env.agent.runtime.type subprocess` runs the agent and the grader on this machine, which then
needs ngspice itself (`apt-get install ngspice`); the offline tests do the same and simulate every
bundled gold sizing.

## Validation

2026-10-16, on a machine with ngspice 42 and no Docker daemon: all 400 bundled gold sizings meet
their spec in simulation and the starting sizes meet none of them; the whole reward path, `setup`
through `finalize` to a fresh grading runtime, scores the gold 1.0 and the starting sizes below it
(`tests/test_analog_sizing.py`). The docker `validate` has not been run yet.

## Baseline

To be recorded from the first runs: a frontier model and Qwen3-1.7B under the `bash` harness on the
100 held-out specs, next to `solved`, `valid_sizes` and `sim_ok`.

## Layout

- `analog_sizing/sim.py`: the netlist, the size bounds, the ngspice call, output parsing and the
  score formula. Stdlib-only because it is copied verbatim into every box.
- `analog_sizing/generate.py`: the task generator (needs ngspice on the host).
- `analog_sizing/taskset.py`: data, the task hooks (`setup`, `finalize`, `validate`), the reward and
  the loader.
- `Dockerfile`: `python:3.12-slim` plus ngspice.

## Next steps

Grade across process, voltage and temperature corners the agent never sees; add slew rate and output
swing from a transient analysis; add other topology families (folded cascode, telescopic, comparator)
with their own generators.

## Changelog

- 2026-10-16: Moved into the catalog from a branch of the verifiers fork: a held-out `eval` split
  (seed 1), the gold sizing off `TaskData`, offline tests, smoke and prime-rl configs.
- 2026-09-09: Initial v1 taskset, op-amp sizing graded by ngspice.
