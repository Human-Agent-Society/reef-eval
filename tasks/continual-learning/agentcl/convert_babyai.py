"""Convert AgentCL's AgentBoard BabyAI streams into Harbor tasks.

Two streams: ``naive`` is 40 episodes in the released order, and
``compositional`` is 56 that pair a source episode with a target episode
needing the same skill, so the second half is where carried state should
show. One episode = one task, name order = stream order.

The grid, the subgoals and the score live in the judge sidecar; the
agent sends actions over HTTP and never sees what it is being scored
against. Reward is progress -- the fraction of the episode's subgoals
reached -- which is AgentCL's PR.

AgentCL publishes each episode's opening observation but not the seed
that produces it, so the seed manifest ships beside this file (see
``SEEDS``). It is checked rather than trusted: the judge rebuilds the
room at startup and refuses to serve if the observation differs.
"""

import json
import shutil
from pathlib import Path

import sidecar

#: The released files, and the name each stream's tasks take.
STREAMS = {"naive": "naive", "comp": "compositional"}

#: Episode id -> BabyAI seed, for the 76 episodes the two streams draw
#: on. Not part of the dataset: AgentCL ships the opening observation a
#: seed has to reproduce, and this manifest is the one the AgentCL
#: harness this conversion follows runs with. Nothing here is taken on
#: faith -- ``babyai_server`` re-derives the observation and stops on a
#: mismatch, so a wrong seed fails loudly instead of grading another room.
SEEDS = Path(__file__).parent / "babyai_seeds.json"

#: AgentGym's BabyAI environment, pinned to the commit AgentCL's servers
#: were built against. It pulls gym/gymnasium/minigrid with it.
AGENTGYM_REPO = "https://github.com/WooooDyy/AgentGym"
AGENTGYM_COMMIT = "d014732d9fe39b975c368c03749bfd50950067f6"

DOCKERFILE_JUDGE = f"""FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \\
    "agentenv_babyai @ git+{AGENTGYM_REPO}@{AGENTGYM_COMMIT}#subdirectory=agentenv-babyai"
RUN mkdir -p /judge
COPY judge_http.py babyai_server.py instance.json /judge/
"""

PREAMBLE = """You are working through a stream of BabyAI gridworld episodes, one per
task, in order. Each episode is a fresh room; what you learned about how
this world answers to commands is not.

Your persistent workspace is `$REEF_EVAL_STATE_DIR` — the phrasings that
worked, the layouts you met, the mistakes that cost you steps. Nothing
else in this container carries over, and you have no network except the
environment."""

PROTOCOL = """The room runs in a sidecar. Look, then act:

    curl -s "$JUDGE_URL/observe"
    curl -s "$JUDGE_URL/step" -d '{"action": "go to the red ball"}'

`/observe` returns what you can see and the exact actions the
environment accepts right now; `/step` returns the same after acting.
You have 30 steps.

The episode is scored on progress: the fraction of its checkpoints you
reach. The checkpoints are not shown to you — pursuing the goal is what
reaches them. Save what you learned to `$REEF_EVAL_STATE_DIR` before
your time is up."""

TASK_TOML = """version = "1.0"

[metadata]
benchmark = "agentcl"
subset = "agentboard_babyai"
stream = "{stream}"
source_id = {source_id}
subtask = "{subtask}"
difficulty = "{difficulty}"
position = {position}
{extra}reference = "pipeline-check"
license_note = "AgentCL (osunlp/AgentCL), CC-BY-NC-4.0; AgentBoard BabyAI episodes, see the dataset card"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 1800.0
# Offline but for the environment: there is nothing on the internet this
# room can be solved from, and egress would only add a lookup channel.
network_mode = "allowlist"
allowed_hosts = ["judge"]

[environment]
build_timeout_sec = 3600.0
"""


def load_stream(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"AgentCL BabyAI stream is empty: {path}")
    for index, row in enumerate(rows):
        missing = {"id", "goal", "subgoals", "additional_info"} - row.keys()
        if missing:
            raise ValueError(f"BabyAI row {index} is missing {sorted(missing)}")
        if not {"init_obs", "subtask"} <= row["additional_info"].keys():
            raise ValueError(f"BabyAI row {index} has invalid additional_info")
        if not row["subgoals"]:
            raise ValueError(f"BabyAI row {index} has no subgoals to score")
    return rows


def load_seeds(path: str | Path) -> dict[int, int]:
    return {
        int(key): int(value)
        for key, value in json.loads(Path(path).read_text()).items()
    }


def task_name(stream: str, position: int) -> str:
    """``babyai-naive-007``: name order is stream order."""
    return f"babyai-{stream}-{position + 1:03d}"


def render_instruction(row: dict, *, position: int, total: int) -> str:
    return (
        "\n\n".join(
            [
                PREAMBLE,
                f"--- Episode {position + 1}/{total} ---",
                f"Your goal: {row['goal']}",
                PROTOCOL,
            ]
        )
        + "\n"
    )


def _metadata_extra(row: dict) -> str:
    """The compositional stream's pairing, where the row carries it."""
    fields = {
        "compositional_split": row.get("compositional_split"),
        "compositional_pair_id": row.get("compositional_pair_id"),
        "compositional_pair_role": row.get("compositional_pair_role"),
    }
    return "".join(
        f'{key} = "{value}"\n' for key, value in fields.items() if value is not None
    )


def convert_episode(
    row: dict,
    seeds: dict[int, int],
    dest_root: Path,
    *,
    stream: str,
    position: int,
    total: int,
) -> Path:
    """Write one episode as a Harbor task with its environment sidecar."""
    source_id = int(row["id"])
    if source_id not in seeds:
        raise ValueError(f"BabyAI seed manifest has no entry for row id {source_id}")
    task_dir = Path(dest_root) / task_name(stream, position)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    info = row["additional_info"]
    (task_dir / "task.toml").write_text(
        TASK_TOML.format(
            stream=stream,
            source_id=source_id,
            subtask=info["subtask"],
            difficulty=row.get("difficulty", "unknown"),
            position=position,
            extra=_metadata_extra(row),
        )
    )
    (task_dir / "instruction.md").write_text(
        render_instruction(row, position=position, total=total)
    )
    sidecar.scaffold(
        task_dir,
        server="babyai_server.py",
        judge_dockerfile=DOCKERFILE_JUDGE,
        instance={
            "source_id": source_id,
            "subtask": info["subtask"],
            "seed": seeds[source_id],
            "goal": row["goal"],
            "init_obs": info["init_obs"],
            "subgoals": list(row["subgoals"]),
        },
    )
    # No gold trajectory is published for these episodes, so the reference
    # solution proves the pipeline -- the room boots, steps, and scores --
    # rather than a ceiling. task.toml says so in `reference`.
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\n"
        "# Pipeline check: no gold trajectory ships with AgentCL's BabyAI\n"
        "# episodes, so this looks once and leaves the score where it lands.\n"
        'curl -s "$JUDGE_URL/observe"\n'
    )
    return task_dir


def convert_stream(
    rows: list[dict],
    seeds: dict[int, int],
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
        convert_episode(
            row, seeds, dest_root, stream=stream, position=position, total=total
        )
        for position, row in enumerate(selected)
    ]
