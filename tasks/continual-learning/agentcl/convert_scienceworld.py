"""Convert AgentCL's AgentBoard ScienceWorld streams into Harbor tasks.

Two streams over the same 90 episodes: ``naive`` is the released order,
``block`` groups them so related tasks arrive together. One episode =
one task, name order = stream order, and the difference between the two
orderings is the experiment.

The simulator, the subgoal patterns and the score live in the judge
sidecar. Reward is progress -- the fraction of the episode's subgoal
patterns that matched an observation, AgentCL's PR -- with the
simulator's own completion reported beside it as SR.
"""

import json
import shutil
from pathlib import Path

import sidecar

#: The released files, and the name each stream's tasks take.
STREAMS = {"naive": "naive", "block": "block"}

#: ScienceWorld is a JVM simulator behind a Python wrapper, so the judge
#: image needs a runtime as well as the package. The wrapper is not
#: version-pinned upstream either; pin it here once a build is proven.
DOCKERFILE_JUDGE = """FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends default-jre \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir scienceworld
RUN mkdir -p /judge
COPY judge_http.py scienceworld_server.py instance.json /judge/
"""

PREAMBLE = """You are working through a stream of ScienceWorld tasks, one per task, in
order. Each drops you into a house of connected rooms with instruments,
substances and living things, and one goal to carry out.

Your persistent workspace is `$REEF_EVAL_STATE_DIR` — where things live,
which phrasings the simulator accepts, what a thermometer reading looks
like. Nothing else in this container carries over, and you have no
network except the simulator."""

PROTOCOL = """The house runs in a sidecar. Look, then act:

    curl -s "$JUDGE_URL/observe"
    curl -s "$JUDGE_URL/actions"
    curl -s "$JUDGE_URL/step" -d '{"action": "go to kitchen"}'

`/actions` lists the action templates and object names the simulator
knows; `teleport` is refused, because moving through the house is part
of the task. You have 30 steps.

The episode is scored on progress: the fraction of its checkpoints you
reach. The checkpoints are not shown to you — carrying out the goal is
what reaches them. Save what you learned to `$REEF_EVAL_STATE_DIR`
before your time is up."""

TASK_TOML = """version = "1.0"

[metadata]
benchmark = "agentcl"
subset = "agentboard_scienceworld"
stream = "{stream}"
source_id = {source_id}
env_name = "{env_name}"
var_num = {var_num}
difficulty = "{difficulty}"
position = {position}
reference = "pipeline-check"
license_note = "AgentCL (osunlp/AgentCL), CC-BY-NC-4.0; AgentBoard ScienceWorld episodes, see the dataset card"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 1800.0
# Offline but for the simulator: ScienceWorld's goals are not lookups,
# and egress would only add a channel the benchmark does not measure.
network_mode = "allowlist"
allowed_hosts = ["judge"]

[environment]
build_timeout_sec = 3600.0
"""


def load_stream(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"AgentCL ScienceWorld stream is empty: {path}")
    for index, row in enumerate(rows):
        missing = {"goal", "subgoals", "additional_info"} - row.keys()
        if missing:
            raise ValueError(f"ScienceWorld row {index} is missing {sorted(missing)}")
        if not {"env_name", "var"} <= row["additional_info"].keys():
            raise ValueError(f"ScienceWorld row {index} has invalid additional_info")
        if not row["subgoals"]:
            raise ValueError(f"ScienceWorld row {index} has no subgoals to score")
    return rows


def task_name(stream: str, position: int) -> str:
    """``sciworld-naive-007``: name order is stream order."""
    return f"sciworld-{stream}-{position + 1:03d}"


def render_instruction(row: dict, *, position: int, total: int) -> str:
    return (
        "\n\n".join(
            [
                PREAMBLE,
                f"--- Episode {position + 1}/{total} ---",
                f"Your task: {row['goal']}",
                PROTOCOL,
            ]
        )
        + "\n"
    )


def convert_episode(
    row: dict,
    dest_root: Path,
    *,
    stream: str,
    position: int,
    total: int,
) -> Path:
    """Write one episode as a Harbor task with its simulator sidecar."""
    info = row["additional_info"]
    task_dir = Path(dest_root) / task_name(stream, position)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    (task_dir / "task.toml").write_text(
        TASK_TOML.format(
            stream=stream,
            source_id=int(row.get("id", position)),
            env_name=info["env_name"],
            var_num=int(info["var"]),
            difficulty=row.get("difficulty", "unknown"),
            position=position,
        )
    )
    (task_dir / "instruction.md").write_text(
        render_instruction(row, position=position, total=total)
    )
    sidecar.scaffold(
        task_dir,
        server="scienceworld_server.py",
        judge_dockerfile=DOCKERFILE_JUDGE,
        instance={
            "goal": row["goal"],
            "env_name": info["env_name"],
            "var": int(info["var"]),
            "subgoals": list(row["subgoals"]),
        },
    )
    # ScienceWorld publishes a gold action sequence through the simulator,
    # but handing it to the solution container would hand it to the agent
    # too -- same network. So the reference proves the pipeline instead:
    # the house boots, steps, and scores. task.toml says so in `reference`.
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\n"
        "# Pipeline check: the gold sequence cannot be shipped to the agent's\n"
        "# side of the network, so this looks once and leaves the score.\n"
        'curl -s "$JUDGE_URL/observe"\n'
    )
    return task_dir


def convert_stream(
    rows: list[dict],
    dest_root: Path,
    *,
    stream: str,
    limit: int | None = None,
) -> list[Path]:
    """Convert one stream in file order, which is the order it is run.

    ``limit`` takes a prefix whose tasks are byte-identical to a full
    conversion's, so a smoke run measures the same episodes.
    """
    total = len(rows)
    selected = rows if limit is None else rows[:limit]
    return [
        convert_episode(row, dest_root, stream=stream, position=position, total=total)
        for position, row in enumerate(selected)
    ]
