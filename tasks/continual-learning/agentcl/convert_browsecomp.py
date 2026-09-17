"""Convert AgentCL's BrowseComp+ streams into Harbor tasks.

Two streams over the same research questions: ``naive`` is the 100
original questions, and ``compositional`` puts the 308 subqueries they
decompose into first, so the originals arrive after their own evidence
has been seen. One question = one task, name order = stream order.

The corpus and the answer live in the judge sidecar; the agent searches
and opens documents through it, with upstream's budget of a hundred tool
calls, and answering is terminal. Scoring is upstream's normalised exact
match.

The corpus itself is published separately from AgentCL and is not
redistributable here, so it is not baked into the image: ``fetch.py
browsecomp --corpus <file>`` indexes it once and the judge mounts that
index read-only.
"""

import json
import shutil
from pathlib import Path

import sidecar

#: ``naive`` is the originals alone; ``comp`` is subqueries then originals.
STREAMS = ("naive", "comp")

#: The compose variable each task's judge mounts its corpus index from.
CORPUS_ENV_VAR = "AGENTCL_BROWSECOMP_CORPUS"
CORPUS_MOUNT = f"${{{CORPUS_ENV_VAR}}}:/judge/corpus.sqlite:ro"

DOCKERFILE_JUDGE = """FROM python:3.12-slim
RUN mkdir -p /judge
COPY judge_http.py browsecomp_corpus.py browsecomp_server.py instance.json /judge/
"""

PREAMBLE = """You are working through a stream of research questions, one per task, in
order. Each is answered from an offline corpus you search through a
sidecar — there is no internet here, and the corpus is the only source.

Your persistent workspace is `$REEF_EVAL_STATE_DIR` — the documents that
turned out to matter, the entities you pinned down, the searches that
found nothing. Later questions in this stream lean on the same corpus,
and often on the same facts."""

PROTOCOL = """Search, open, then answer:

    curl -s "$JUDGE_URL/search" -d '{"query": "<words to look for>"}'
    curl -s "$JUDGE_URL/open"   -d '{"doc_id": "<id from a search hit>"}'
    curl -s "$JUDGE_URL/answer" -d '{"answer": "<the answer itself>"}'

Search returns the five best matching documents; opening one returns up
to 12,000 characters. You have 100 tool calls in total; answering is
free but terminal, and there is no second answer.

The answer is graded by exact match after casefolding and dropping
punctuation, so give the name itself — not a sentence about it. A wrong
or missing answer scores 0."""

TASK_TOML = """version = "1.0"

[metadata]
benchmark = "agentcl"
subset = "browsecomp_plus"
stream = "{stream}"
query_id = "{query_id}"
query_type = "{query_type}"
parent_query_id = "{parent_query_id}"
position = {position}
oracle_score = 1.0
license_note = "AgentCL (osunlp/AgentCL), CC-BY-NC-4.0; BrowseComp-Plus queries, see the dataset card"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 1800.0
# Offline but for the corpus service: these questions are answerable on
# the open web, so egress would measure search engines, not this corpus.
network_mode = "allowlist"
allowed_hosts = ["judge"]

[environment]
build_timeout_sec = 1800.0
"""


def load_jsonl(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    required = {"query_id", "query", "answer", "query_type", "parent_query_id"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"BrowseComp+ row {index} is missing {sorted(missing)}")
        if not str(row["answer"]).strip():
            raise ValueError(f"BrowseComp+ row {index} has an empty answer")
    if not rows:
        raise ValueError(f"BrowseComp+ file is empty: {path}")
    return rows


def load_stream(data_dir: str | Path, stream: str) -> list[dict]:
    """The released stream, with upstream's own shape checks.

    ``naive`` is ``ground_truth_original.jsonl``; ``comp`` is the
    subqueries followed by those same originals, which is the ordering
    that makes the second half a test of what the first half taught.
    """
    root = Path(data_dir)
    originals = load_jsonl(root / "ground_truth_original.jsonl")
    if len(originals) != 100 or any(
        row["query_type"] != "original" for row in originals
    ):
        raise ValueError("BrowseComp+ originals must be 100 rows of type 'original'")
    if stream == "naive":
        return originals
    if stream != "comp":
        raise ValueError(f"unknown BrowseComp+ stream {stream!r}; have {STREAMS}")
    subqueries = load_jsonl(root / "ground_truth_subqueries.jsonl")
    if len(subqueries) != 308 or any(
        row["query_type"] != "subquery" for row in subqueries
    ):
        raise ValueError("BrowseComp+ subqueries must be 308 rows of type 'subquery'")
    return [*subqueries, *originals]


def task_name(stream: str, position: int) -> str:
    """``browsecomp-comp-007``: name order is stream order."""
    return f"browsecomp-{stream}-{position + 1:03d}"


def render_instruction(row: dict, *, position: int, total: int) -> str:
    return (
        "\n\n".join(
            [
                PREAMBLE,
                f"--- Question {position + 1}/{total} ---",
                f"Question: {row['query']}",
                PROTOCOL,
            ]
        )
        + "\n"
    )


def convert_query(
    row: dict,
    dest_root: Path,
    *,
    stream: str,
    position: int,
    total: int,
) -> Path:
    """Write one question as a Harbor task with its corpus sidecar."""
    task_dir = Path(dest_root) / task_name(stream, position)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    (task_dir / "task.toml").write_text(
        TASK_TOML.format(
            stream=stream,
            query_id=row["query_id"],
            query_type=row["query_type"],
            parent_query_id=row["parent_query_id"],
            position=position,
        )
    )
    (task_dir / "instruction.md").write_text(
        render_instruction(row, position=position, total=total)
    )
    sidecar.scaffold(
        task_dir,
        server="browsecomp_server.py",
        judge_dockerfile=DOCKERFILE_JUDGE,
        instance={
            "query_id": row["query_id"],
            "query": row["query"],
            "answer": row["answer"],
        },
        extra_files=("browsecomp_corpus.py",),
        judge_volumes=(CORPUS_MOUNT,),
    )
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\n"
        "# The published answer, submitted directly: exact match, 1.0.\n"
        "curl -s \"$JUDGE_URL/answer\" -d @- <<'AGENTCL_EOF'\n"
        f"{json.dumps({'answer': row['answer']})}\n"
        "AGENTCL_EOF\n"
    )
    return task_dir


def convert_stream(
    rows: list[dict],
    dest_root: Path,
    *,
    stream: str,
    limit: int | None = None,
) -> list[Path]:
    """Convert one stream in order; ``limit`` takes an identical prefix."""
    total = len(rows)
    selected = rows if limit is None else rows[:limit]
    return [
        convert_query(row, dest_root, stream=stream, position=position, total=total)
        for position, row in enumerate(selected)
    ]
