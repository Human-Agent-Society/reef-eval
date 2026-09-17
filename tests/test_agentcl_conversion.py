"""AgentCL MMLU-Pro conversion + scorer: real fixture questions, known keys.

AgentCL is CC-BY-NC-4.0, so its converted tasks are never committed to
this repo and `tests/test_task_suite.py` never sees them. Everything that
suite would enforce for a committed task is enforced here instead,
against tasks built in a tmp dir: stock-Harbor validation, the reference
solution scoring exactly 1.0, and the zero-reward cases that try to game
the grader.

The fixture holds three questions quoted from ``mmlu_pro/test_300.json``
for testing: the first two economics items and the first engineering one.
"""

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

AGENTCL = Path(__file__).parent.parent / "tasks" / "continual-learning" / "agentcl"
FIXTURES = Path(__file__).parent / "fixtures" / "agentcl"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"agentcl_{name}", AGENTCL / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


convert = _load("convert_mmlu")
scorer = _load("score_mmlu")

ROWS = json.loads((FIXTURES / "mmlu_pro_sample.json").read_text())
ECONOMICS = [row for row in ROWS if row["category"] == "economics"]


def _convert(tmp_path: Path, domain: str = "economics") -> list[Path]:
    return convert.convert_domain(ROWS, tmp_path, domain=domain)


def _truth(task_dir: Path) -> dict:
    return json.loads((task_dir / "tests" / "truth.json").read_text())


def _oracle_answer(task_dir: Path) -> str:
    """The submission the reference solution writes, as the agent would."""
    script = (task_dir / "solution" / "solve.sh").read_text()
    echo = next(line for line in script.splitlines() if line.startswith("echo "))
    return echo.split("'")[1]


# ---------------------------------------------------------------- converter


def test_convert_writes_a_complete_task(tmp_path):
    first = _convert(tmp_path)[0]
    assert first.name == "mmlu-economics-q001"
    for piece in (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "tests/test.sh",
        "tests/score.py",
        "tests/truth.json",
        "solution/solve.sh",
    ):
        assert (first / piece).exists(), piece


def test_task_toml_records_where_in_the_stream_this_is(tmp_path):
    tasks = _convert(tmp_path)
    config = tomllib.loads((tasks[1] / "task.toml").read_text())
    assert config["metadata"]["benchmark"] == "agentcl"
    assert config["metadata"]["subset"] == "mmlu_pro"
    assert config["metadata"]["domain"] == "economics"
    assert config["metadata"]["position"] == 1
    assert config["metadata"]["question_id"] == ECONOMICS[1]["question_id"]
    # Offline: the questions are public, so egress would measure search.
    assert config["agent"]["network_mode"] == "allowlist"
    assert "allowed_hosts" not in config["agent"]


def test_names_sort_into_stream_order(tmp_path):
    """`reef-eval stream` runs a folder in path order, so the zero padding
    is what keeps question 2 from following question 19."""
    names = [task.name for task in _convert(tmp_path)]
    assert names == sorted(names)
    assert names == ["mmlu-economics-q001", "mmlu-economics-q002"]
    assert convert.task_name("philosophy", 99) == "mmlu-philosophy-q100"


def test_instruction_matches_the_upstream_prompt_shape(tmp_path):
    row = ECONOMICS[0]
    text = (_convert(tmp_path)[0] / "instruction.md").read_text()
    assert "Answer this multiple-choice question" in text  # upstream's wording
    assert f"Question: {row['question']}" in text
    for label, option in zip("ABCDEFGH", row["options"], strict=True):
        assert f"{label}. {option}" in text
    assert "--- Question 1/2 (economics) ---" in text
    assert "/app/answer.json" in text
    assert "$REEF_EVAL_STATE_DIR" in text  # the deviation that carries memory


def test_convert_is_valid_stock_harbor(tmp_path):
    pytest.importorskip("harbor")
    from harbor.models.task.config import TaskConfig

    for task_dir in _convert(tmp_path):
        TaskConfig.model_validate(tomllib.loads((task_dir / "task.toml").read_text()))


def test_engineering_is_its_own_stream(tmp_path):
    """Domains are separate streams upstream; converting one leaves the
    others alone, and each restarts the numbering."""
    tasks = _convert(tmp_path, domain="engineering")
    assert [task.name for task in tasks] == ["mmlu-engineering-q001"]
    assert _truth(tasks[0])["labels"] == list("ABCDEFGHIJ")  # 10 options here


def test_load_domain_keeps_file_order_and_rejects_unknown_domains():
    assert [row["question_id"] for row in convert.load_domain(ROWS, "economics")] == [
        row["question_id"] for row in ECONOMICS
    ]
    with pytest.raises(ValueError, match="unknown MMLU-Pro domain"):
        convert.load_domain(ROWS, "astrology")


@pytest.mark.parametrize(
    ("broken", "match"),
    [
        ({"answer": "Z"}, "not one of"),
        ({"answer_index": 3}, "disagrees with itself"),
        ({"options": []}, "no options"),
        ({"question": "  "}, "empty text"),
    ],
)
def test_a_self_contradicting_question_stops_the_run(broken, match, tmp_path):
    """Measurement code fails loudly: a source row that does not agree with
    itself is a changed benchmark, not a task to convert quietly."""
    with pytest.raises(ValueError, match=match):
        convert.convert_question(
            {**ECONOMICS[0], **broken},
            tmp_path,
            domain="economics",
            position=0,
            total=2,
            previous=None,
        )


# ----------------------------------------------------------------- feedback


def test_the_stream_reveals_the_previous_key_and_only_that(tmp_path):
    first, second = _convert(tmp_path)
    assert (
        "## Feedback on the previous question"
        not in (first / "instruction.md").read_text()
    )
    text = (second / "instruction.md").read_text()
    assert "## Feedback on the previous question" in text
    assert f"`{ECONOMICS[0]['answer']}`" in text  # question 1's key: G
    assert ECONOMICS[0]["question"][:40] in text  # quoted, so it is unambiguous
    assert f"`{ECONOMICS[1]['answer']}`" not in text  # never this question's key


def test_the_answer_key_never_reaches_the_agent(tmp_path):
    """The key lives in ``tests/``, mounted for the verifier only. Nothing
    the agent can read before answering names an option the way the grader
    reads one, so a leak would have to get past this test to pay."""
    for task_dir in _convert(tmp_path):
        truth = _truth(task_dir)
        readable = [task_dir / "instruction.md", *(task_dir / "environment").iterdir()]
        for path in readable:
            text = path.read_text()
            assert scorer.parse_answer(text, truth["labels"]) is None, path.name
            assert scorer.grade(text, truth)["score"] == 0.0, path.name


# ------------------------------------------------------------------- scorer


def test_reference_solution_scores_exactly_one(tmp_path):
    for task_dir in _convert(tmp_path) + _convert(tmp_path, domain="engineering"):
        truth = _truth(task_dir)
        assert scorer.grade(_oracle_answer(task_dir), truth)["score"] == 1.0


def test_a_wrong_option_scores_zero_and_says_so(tmp_path):
    truth = _truth(_convert(tmp_path)[0])  # keyed G
    result = scorer.grade('{"answer": "B"}', truth)
    assert result["score"] == 0.0
    assert result["answer"] == "B"
    assert "correct option was G" in result["reason"]


@pytest.mark.parametrize(
    "submission",
    [
        None,  # never wrote the file
        "",
        "{not json",
        '{"answer": "Z"}',  # not a listed option
        '{"reasoning": "it is clearly G"}',  # names no answer field
        "A B C D E F G H",  # every option at once
        '{"answer": ["A", "B", "C", "D", "E", "F", "G", "H"]}',
        "The answer is G.",  # prose, not the declared format
    ],
)
def test_submissions_that_name_no_option_score_zero_with_a_reason(submission, tmp_path):
    """Rule: a submission that does not commit to one letter gets nothing.
    Spraying every option is the cheapest attack on a letter grader, so it
    is the one the grader has to be boring about."""
    truth = _truth(_convert(tmp_path)[0])  # keyed G
    result = scorer.grade(submission, truth)
    assert result["score"] == 0.0
    assert result["reason"]


@pytest.mark.parametrize(
    "submission",
    ['{"answer": "G"}', '{"answer": "g"}', '{"answer":"G"}\n', "G", '"G"\n'],
)
def test_the_shapes_an_answer_may_take(submission, tmp_path):
    """Format is not what this benchmark measures: the JSON object it asks
    for, its near-misses, and a file holding just the letter all count."""
    truth = _truth(_convert(tmp_path)[0])  # keyed G
    assert scorer.grade(submission, truth)["score"] == 1.0


def test_a_missing_file_is_a_zero_not_a_crash(tmp_path):
    raw, status = scorer.read_submission(str(tmp_path / "answer.json"))
    assert raw is None and "no answer file" in status
    written = tmp_path / "answer.json"
    written.write_text('{"answer": "G"}')
    raw, status = scorer.read_submission(str(written))
    assert raw == '{"answer": "G"}' and status == "ok"


def test_a_broken_answer_key_raises(tmp_path):
    """Leniency is for submissions only; a bad key is our bug and stops."""
    with pytest.raises(ValueError, match="answer key"):
        scorer.grade('{"answer": "A"}', {"answer": "Z", "labels": ["A", "B"]})
