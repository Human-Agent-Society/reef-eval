"""Convert AgentCL's CodeEval-Pro streams into Harbor tasks.

CodeEval-Pro pairs each raw HumanEval/MBPP/BigCodeBench problem with a
composed problem that needs the raw one solved inside it. AgentCL
streams that two ways: ``naive`` is the composed problems alone, and
``compositional`` puts all the raw problems first, so what an agent
learned early is exactly what the second half needs. Both are converted;
one problem = one task, name order = stream order.

The tests never enter the agent's container. They live in the judge
sidecar, which meters submissions the way upstream meters retries: ten
tries with the failure output returned, 1.0 if any of them passes.
"""

import json
import re
import shutil
from pathlib import Path

import sidecar

#: AgentCL's three CodeEval-Pro sources, and the file stem each uses.
DATASETS = {
    "humaneval": "humaneval_pro",
    "mbpp": "mbpp_pro",
    "bigcodebench": "bigcodebench_lite_pro",
}
#: ``naive`` = composed problems only; ``comp`` = raw problems, then composed.
STREAMS = {"naive": "naive", "comp": "compositional"}

#: What the hidden tests import, per dataset, mined from the released
#: test code. The judge image installs these; the agent's has nothing to
#: run tests with, because it never runs them.
JUDGE_PACKAGES = {
    "humaneval": ("numpy",),
    "mbpp": ("numpy",),
    "bigcodebench": (
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "seaborn",
        "statsmodels",
        "holidays",
        "regex",
        "python-dateutil",
        "Faker",
    ),
}

_DEF = re.compile(r"(?m)^[ \t]*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(")
_SIGNATURE = re.compile(
    r"(?m)^[ \t]*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\([^\n]*\)"
    r"(?:\s*->\s*[^:\n]+)?\s*:\s*$"
)

#: Problems whose own published reference cannot pass their own tests, so
#: no completion can: the ceiling is 0, not 1. Upstream scores them the
#: same way -- it builds the test program exactly like the judge does --
#: so they are converted as they are rather than repaired, and the task
#: records ``oracle_score = 0.0`` instead of claiming a solution exists.
#: Measured over the pinned files with ``fetch.py codeeval
#: --check-references``, which re-runs every reference and fails if this
#: table has drifted. BigCodeBench needs the judge image's packages, so
#: run the check there; the entries below are humaneval and mbpp.
REFERENCE_FAILURES = {
    ("humaneval", "new", "0"): "prompt annotates List without importing typing",
    ("humaneval", "new", "4"): "prompt annotates List without importing typing",
    ("humaneval", "new", "6"): "prompt annotates List without importing typing",
    ("humaneval", "new", "9"): "prompt annotates Tuple without importing typing",
    ("humaneval", "new", "29"): "prompt annotates List without importing typing",
    (
        "humaneval",
        "new",
        "41",
    ): "test code redefines the entry point after the candidate",
    (
        "mbpp",
        "new",
        "50",
    ): "reference calls the test's helper with an already-flat list",
}


PREAMBLE = """You are working through a stream of Python problems, one per task, in
order. Each is graded by hidden tests you cannot read.

Your persistent workspace is `$REEF_EVAL_STATE_DIR` — the helpers you
wrote, the problems you already solved, and what the tests rejected are
the only things that reach the next problem. Nothing else in this
container carries over, and you have no network."""

PROTOCOL = """Write the completion to `/app/completion.py` — the body that goes after
the prefix, at the prefix's indentation, without repeating it — and
submit:

    submit /app/completion.py

The judge runs the hidden tests and prints what they said. You have 10
submissions; the problem scores 1.0 if one of them passes and 0
otherwise, so read the failure and fix it rather than guessing again."""

TASK_TOML = """version = "1.0"

[metadata]
benchmark = "agentcl"
subset = "codeeval_pro"
dataset = "{dataset}"
stream = "{stream}"
category = "{category}"
problem_id = "{problem_id}"
position = {position}
oracle_score = {oracle_score}
{reference_note}license_note = "AgentCL (osunlp/AgentCL), CC-BY-NC-4.0; CodeEval-Pro and its source benchmarks, see the dataset card"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 1800.0
# Offline but for the judge: these problems are public, so egress would
# measure lookup rather than what the stream taught.
network_mode = "allowlist"
allowed_hosts = ["judge"]

[environment]
build_timeout_sec = 3600.0
"""

AGENT_DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*
COPY submit.py /usr/local/bin/submit
RUN chmod +x /usr/local/bin/submit
WORKDIR /app
"""

DOCKERFILE_JUDGE = """FROM python:3.12-slim
RUN pip install --no-cache-dir {packages}
RUN mkdir -p /judge
COPY judge_http.py codeeval_server.py instance.json /judge/
"""


def load_stream(path: str | Path) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"CodeEval stream must be a non-empty list: {path}")
    required = {"id", "problem", "solution", "test_code", "category"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"CodeEval row {index} is missing {sorted(missing)}")
        if row["category"] not in {"raw", "new"}:
            raise ValueError(f"CodeEval row {index} has category {row['category']!r}")
    return rows


def split_prefix(row: dict) -> tuple[str, str]:
    """The published prefix and the reference completion that follows it.

    Three HumanEval rows are released truncated: ``problem`` holds only
    the leading comment and the rest of the public prompt sits at the
    top of ``solution``. Upstream's ``problem_prefix`` repairs those by
    cutting ``solution`` at its first signature; splitting at the same
    point is what keeps the reference solution out of the prompt.
    """
    problem = str(row["problem"]).rstrip() + "\n"
    solution = str(row.get("solution") or "")
    if _DEF.search(problem):
        return problem, solution
    signature = _SIGNATURE.search(solution)
    if not signature:
        raise ValueError(f"row {row['id']} has no Python function signature")
    return (
        problem + solution[: signature.end()].strip("\n") + "\n",
        solution[signature.end() :].lstrip("\n"),
    )


def task_name(dataset: str, stream: str, position: int) -> str:
    """``code-humaneval-naive-007``: name order is stream order."""
    return f"code-{dataset}-{stream}-{position + 1:03d}"


def render_instruction(
    row: dict, prefix: str, *, dataset: str, position: int, total: int
) -> str:
    return (
        "\n\n".join(
            [
                PREAMBLE,
                f"--- Problem {position + 1}/{total} ({dataset}) ---",
                "Complete the Python code below.",
                f"```python\n{prefix.rstrip()}\n```",
                PROTOCOL,
            ]
        )
        + "\n"
    )


def convert_problem(
    row: dict,
    dest_root: Path,
    *,
    dataset: str,
    stream: str,
    position: int,
    total: int,
) -> Path:
    """Write one problem as a Harbor task with its judge sidecar."""
    prefix, reference = split_prefix(row)
    task_dir = Path(dest_root) / task_name(dataset, stream, position)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()

    problem_id = f"{dataset}-{row['category']}-{row['id']}"
    broken = REFERENCE_FAILURES.get((dataset, row["category"], str(row["id"])))
    (task_dir / "task.toml").write_text(
        TASK_TOML.format(
            dataset=dataset,
            stream=stream,
            category=row["category"],
            problem_id=problem_id,
            position=position,
            oracle_score=0.0 if broken else 1.0,
            reference_note=f'reference_note = "{broken}"\n' if broken else "",
        )
    )
    (task_dir / "instruction.md").write_text(
        render_instruction(row, prefix, dataset=dataset, position=position, total=total)
    )
    # The tests and the prefix go to the judge's build context only: the
    # agent's image copies nothing from it but the submit helper.
    sidecar.scaffold(
        task_dir,
        server="codeeval_server.py",
        judge_dockerfile=DOCKERFILE_JUDGE.format(
            packages=" ".join(JUDGE_PACKAGES[dataset])
        ),
        instance={
            "problem_id": problem_id,
            "prefix": prefix,
            "test_code": row["test_code"],
        },
        agent_dockerfile=AGENT_DOCKERFILE,
        extra_files=("submit.py",),
    )
    reference_comment = (
        f"# The released reference completion. It scores 0 here: {broken}.\n"
        if broken
        else "# The released reference completion: passes the hidden tests, 1.0.\n"
    )
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\n"
        + reference_comment
        + "cat > /app/completion.py <<'AGENTCL_EOF'\n"
        + f"{reference.rstrip()}\n"
        + "AGENTCL_EOF\n"
        + "submit /app/completion.py\n"
    )
    return task_dir


def convert_stream(
    rows: list[dict],
    dest_root: Path,
    *,
    dataset: str,
    stream: str,
    limit: int | None = None,
) -> list[Path]:
    """Convert one dataset stream in file order, which is stream order.

    ``limit`` takes a prefix and leaves the tasks in it byte-identical to
    a full conversion's, so a smoke run measures the same tasks.
    """
    total = len(rows)
    selected = rows if limit is None else rows[:limit]
    return [
        convert_problem(
            row,
            dest_root,
            dataset=dataset,
            stream=stream,
            position=position,
            total=total,
        )
        for position, row in enumerate(selected)
    ]
