"""Convert AgentCL's MMLU-Pro stream into Harbor tasks.

AgentCL runs MMLU-Pro as three 100-question domain streams (economics,
engineering, philosophy) taken from ``mmlu_pro/test_300.json``. One
question = one task named ``mmlu-<domain>-qNNN``, so name order replays
the stream and ``reef-eval stream`` carries state across it exactly as
upstream carries a memory across the conversation.

Upstream reveals the correct option after every question; that feedback
is what makes the stream learnable. Here it arrives at the top of the
next task's instruction, the way CL-Bench's dbx questions carry the
previous answer, because each task is its own container. What the agent
does with it -- what it writes into ``$REEF_EVAL_STATE_DIR`` -- is the
thing under measurement.
"""

import json
import shutil
from pathlib import Path

SCORER = Path(__file__).parent / "score_mmlu.py"

#: The three domains AgentCL cuts out of MMLU-Pro, 100 questions each.
DOMAINS = ("economics", "engineering", "philosophy")

#: How much of the previous question the feedback block quotes. Upstream
#: fed feedback into a conversation that still held the question; a short
#: quote is what replaces that adjacency.
FEEDBACK_QUOTE_CHARS = 240

PREAMBLE = """You are working through a stream of MMLU-Pro questions from a single
subject, one question per task, in order. You see each question once.

Your persistent workspace is `$REEF_EVAL_STATE_DIR` — notes you save
there are the only thing that reaches the next question: the subject's
conventions, the traps you fell into, the corrections you were given.
Nothing else in this container carries over, and you have no network.

After you answer, the next task tells you which option was correct, so a
mistake is worth writing down."""

# Upstream's question block (mmlu_runner.render_question), verbatim.
QUESTION_PREAMBLE = (
    "Answer this multiple-choice question. Select exactly one listed option."
)

# The schema shows a placeholder, never a sample letter: a literal letter
# in the prompt is a nudge, and on the questions keyed to it, a free point.
ANSWER_PROTOCOL = """Write your answer as JSON to `/app/answer.json`:

    {"answer": "<letter>"}

where `<letter>` is the letter at the start of the option you choose. The
grader reads only that file, and only the letter."""

TASK_TOML = """version = "1.0"

[metadata]
benchmark = "agentcl"
subset = "mmlu_pro"
domain = "{domain}"
question_id = {question_id}
position = {position}
license_note = "AgentCL (osunlp/AgentCL), CC-BY-NC-4.0; MMLU-Pro questions, see the dataset card"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 600.0
# Offline. The questions are public MMLU-Pro items, so egress is a
# lookup channel: it would measure search, not what the stream taught.
network_mode = "allowlist"

[environment]
build_timeout_sec = 600.0
"""

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
"""

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/score.py
"""


def load_domain(rows: list[dict], domain: str) -> list[dict]:
    """One domain's questions, in file order — the order upstream streams."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown MMLU-Pro domain {domain!r}; have {list(DOMAINS)}")
    return [row for row in rows if row.get("category") == domain]


def option_labels(row: dict) -> list[str]:
    options = row["options"]
    if not options:
        raise ValueError(f"question {row['question_id']} has no options")
    if len(options) > 26:
        raise ValueError(f"question {row['question_id']} has more options than letters")
    return [chr(ord("A") + index) for index in range(len(options))]


def check_row(row: dict) -> list[str]:
    """Validate one question and return its labels; a bad row stops the run."""
    labels = option_labels(row)
    if not str(row.get("question", "")).strip():
        raise ValueError(f"question {row['question_id']} has empty text")
    answer = str(row["answer"]).upper()
    if answer not in labels:
        raise ValueError(
            f"question {row['question_id']} keys {answer!r}, not one of {labels}"
        )
    index = row.get("answer_index")
    if index is not None and labels[int(index)] != answer:
        raise ValueError(
            f"question {row['question_id']} disagrees with itself: "
            f"answer {answer!r} but answer_index {index}"
        )
    return labels


def task_name(domain: str, position: int) -> str:
    """``mmlu-economics-q007``: name order is stream order."""
    return f"mmlu-{domain}-q{position + 1:03d}"


def _quote(text: str, limit: int = FEEDBACK_QUOTE_CHARS) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def render_feedback(previous: dict, previous_position: int) -> str:
    """The correct option for the question just answered.

    Upstream says "Correct." or "Incorrect. The correct option was X.";
    naming the option unconditionally is the same information, since the
    agent's own answer is in its memory and not in ours.
    """
    return (
        "## Feedback on the previous question\n\n"
        f"Question {previous_position + 1} asked: “{_quote(previous['question'])}”\n\n"
        f"The correct option was `{str(previous['answer']).upper()}`. Check it "
        "against the answer you recorded before reusing the same reasoning."
    )


def render_instruction(
    row: dict,
    labels: list[str],
    *,
    domain: str,
    position: int,
    total: int,
    previous: dict | None,
) -> str:
    options = "\n".join(
        f"{label}. {value}" for label, value in zip(labels, row["options"], strict=True)
    )
    blocks = [
        PREAMBLE,
        f"--- Question {position + 1}/{total} ({domain}) ---",
        QUESTION_PREAMBLE,
        f"Question: {row['question']}",
        f"Options:\n{options}",
        ANSWER_PROTOCOL,
    ]
    if previous is not None:
        blocks.append(render_feedback(previous, position - 1))
    return "\n\n".join(blocks) + "\n"


def convert_question(
    row: dict,
    dest_root: Path,
    *,
    domain: str,
    position: int,
    total: int,
    previous: dict | None,
) -> Path:
    """Write one question as a Harbor task folder; returns the folder path.

    The answer key goes to ``tests/truth.json`` and nowhere else: the
    agent's container never holds the thing it is being asked for.
    """
    labels = check_row(row)
    task_dir = Path(dest_root) / task_name(domain, position)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()

    (task_dir / "task.toml").write_text(
        TASK_TOML.format(
            domain=domain,
            question_id=int(row["question_id"]),
            position=position,
        )
    )
    (task_dir / "instruction.md").write_text(
        render_instruction(
            row,
            labels,
            domain=domain,
            position=position,
            total=total,
            previous=previous,
        )
    )
    (task_dir / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task_dir / "tests" / "test.sh").write_text(TEST_SH)
    (task_dir / "tests" / "truth.json").write_text(
        json.dumps(
            {
                "answer": str(row["answer"]).upper(),
                "labels": labels,
                "question_id": int(row["question_id"]),
                "domain": domain,
            },
            indent=1,
        )
    )
    shutil.copy(SCORER, task_dir / "tests" / "score.py")
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\n"
        "# The keyed option: scores 1.0, the only score this task hands out.\n"
        f"""echo '{{"answer": "{str(row["answer"]).upper()}"}}' > /app/answer.json\n"""
    )
    return task_dir


def convert_domain(
    rows: list[dict], dest_root: Path, *, domain: str, limit: int | None = None
) -> list[Path]:
    """Convert one domain stream; each task carries the previous key.

    ``limit`` takes a prefix of the stream, and the questions in it are
    byte-identical to the ones a full conversion writes: the position
    line still counts against the whole domain. A smoke run of ten must
    not quietly become ten different tasks.
    """
    questions = load_domain(rows, domain)
    total = len(questions)
    if limit is not None:
        questions = questions[:limit]
    return [
        convert_question(
            row,
            dest_root,
            domain=domain,
            position=position,
            total=total,
            previous=questions[position - 1] if position else None,
        )
        for position, row in enumerate(questions)
    ]
